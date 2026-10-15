"""Stage 2 (seeding) + Stage 3 (run/read-out) of the PPR graph leg.

This module turns a query into the PPR **personalization vector** ``e`` over the
graph built in :mod:`dc_available.graph.retrieval.ppr_graph`, then (Stage 3, added next)
runs the power iteration and reads out chunks.

Seeding (Stage 2) mixes two independent signals so the walk starts from the right
places without over-trusting any single one:

  * **entity seeds** — the query-matched entities (same gating the current leg
    uses), each weighted by ``signal × idf``. The ``idf`` factor is HippoRAG's
    *node specificity*: generic entities ("billing", "sales order") get little
    mass, specific ones get a lot. This is the fix for generic-entity mis-routing.
  * **passage seeds** *(optional, on by default)* — the vector leg's top chunks,
    weighted by their retrieval score. This grounds the walk in the strong
    (card-boosted) dense signal without merging the legs; a single knob
    ``PPR_PASSAGE_SEED_WEIGHT`` controls how much of the seed mass they get.

The two groups are normalised to a configurable mass split, so neither drowns the
other, then combined into one probability vector that sums to 1.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from pathlib import Path

from dc_available.config import data_config
from dc_available.graph.retrieval.ppr_graph import PPRGraph, EdgeWeights, get_ppr_graph
from dc_available.graph.retrieval.entity_hop_retriever import (
    GraphResult,
    _match_seed_entities,
    _signal_score,
)


# Defaults live here until Stage 5 moves them into data_config; getattr keeps this
# module runnable whether or not those knobs have been added yet.
def _cfg(name: str, default):
    return getattr(data_config, name, default)


@dataclass
class SeedResult:
    """The personalization vector plus a small provenance report (for debug)."""

    vector: np.ndarray | None                 # length = graph.num_nodes(); sums to 1, or None
    abstained: bool
    n_entity_seeds: int = 0
    n_passage_seeds: int = 0
    entity_mass: float = 0.0
    passage_mass: float = 0.0
    raw_weight: float = 0.0                    # pre-normalisation seed strength (cross-topic share)
    seeds: list[dict] = field(default_factory=list)  # [{id, kind, weight}]


def _accumulate(weights: dict[int, float], idx: int | None, w: float) -> None:
    if idx is not None and w > 0.0:
        weights[idx] = weights.get(idx, 0.0) + w


# ===========================================================================
# Seed selection v2 (Stages 1-3): threshold -> boilerplate filter -> merge.
# Gated by PPR_SEED_MODE="merge". Query-time only; the stored graph is untouched.
# ===========================================================================

@dataclass
class SeedGroup:
    """A merged concept seed: several entity fragments sharing one canonical key."""

    key: str                 # canonical concept key (e.g. "sales order")
    weight: float            # group weight = strongest member's signal x idf
    members: list[dict]      # the entity dicts folded into this group


def _seed_signal(ent: dict) -> float:
    """max(fused, dense) — the confidence used for the threshold and ranking."""
    fused = float(ent.get("fused_score") or 0.0)
    dense = float(ent.get("dense_score") or 0.0)
    return max(fused, dense)


def _member_weight(ent: dict) -> float:
    """Per-entity seed weight = max(fused,dense) x idf (HippoRAG node specificity)."""
    return _seed_signal(ent) * float(ent.get("idf") or 0.0)


def _canonical_key(ent: dict) -> str:
    """Concept key for merging: the label, lowercased and whitespace-normalised
    (entity_type stripped). Safe exact-label merge: unites "Sales Order
    [transaction/process/order/...]" but keeps "Standard Sales Order" separate."""
    return re.sub(r"\s+", " ", str(ent.get("label") or "").strip().lower())


def _seed_dbg(on: bool, msg: str) -> None:
    if on:
        print(msg)


def _select_seeds(
    query: str, debug: bool = False, entity_collection: str | None = None
) -> tuple[list[SeedGroup], list[dict], float, float]:
    """Stages 1-3 of the new PPR seeding.

    1. FETCH  ``PPR_SEED_FETCH_K`` entities from the index.
    2. FILTER keep ``max(fused,dense) >= PPR_SEED_DENSE_THRESHOLD`` (with a top-N
       backstop so the set is never empty), then drop boilerplate
       (``chunk_frequency > PPR_SEED_MAX_CF``) unless that would empty it.
    3. MERGE  same-concept fragments into one seed (weight = strongest member).

    Returns ``(groups, members_flat, best_signal, concentration)``.
    """
    from dc_available.graph.construct.entity_index import search_entities_scored

    dbg = debug or bool(_cfg("PPR_SEED_DEBUG", False))
    fetch_k = int(_cfg("PPR_SEED_FETCH_K", 50))
    thr = float(_cfg("PPR_SEED_DENSE_THRESHOLD", 0.6))
    backstop = int(_cfg("PPR_SEED_TOPN_BACKSTOP", 5))
    max_cf = int(_cfg("PPR_SEED_MAX_CF", 150))
    do_merge = bool(_cfg("PPR_SEED_MERGE", True))

    # --- Stage 1: fetch + threshold ---------------------------------------
    entities = search_entities_scored(query, k=fetch_k, collection_name=entity_collection)
    _seed_dbg(dbg, "\n[seed] ── Stage 1: fetch + threshold ──")
    _seed_dbg(dbg, f"[seed] fetched {len(entities)} entities (k={fetch_k})")
    if not entities:
        _seed_dbg(dbg, "[seed] no entity matches -> abstain")
        return [], [], 0.0, 0.0

    kept = [e for e in entities if _seed_signal(e) >= thr]
    if len(kept) < backstop:
        kept = sorted(entities, key=_seed_signal, reverse=True)[:backstop]
        _seed_dbg(dbg, f"[seed] < {backstop} passed threshold {thr}; "
                       f"backstop -> top {len(kept)} by max(fused,dense)")
    else:
        _seed_dbg(dbg, f"[seed] threshold max(fused,dense) >= {thr} -> {len(kept)} kept")
    for e in kept:
        _seed_dbg(dbg, f"[seed]    keep  {e.get('label')} [{e.get('entity_type')}]  "
                       f"max={_seed_signal(e):.3f} idf={e.get('idf')} cf={e.get('chunk_frequency')}")

    # --- Stage 2: boilerplate filter (conditional; never empties the set) ---
    _seed_dbg(dbg, f"[seed] ── Stage 2: boilerplate filter (cf > {max_cf}) ──")
    filtered = [e for e in kept if int(e.get("chunk_frequency") or 0) <= max_cf]
    if filtered and len(filtered) < len(kept):
        for e in kept:
            if e not in filtered:
                _seed_dbg(dbg, f"[seed]    drop  {e.get('label')} [{e.get('entity_type')}] "
                               f"cf={e.get('chunk_frequency')} (boilerplate)")
        _seed_dbg(dbg, f"[seed] dropped {len(kept) - len(filtered)} boilerplate -> {len(filtered)} kept")
        kept = filtered
    elif not filtered:
        _seed_dbg(dbg, f"[seed] all {len(kept)} are boilerplate; keeping them (never empty)")
    else:
        _seed_dbg(dbg, "[seed] no boilerplate found (nothing dropped)")

    # --- Stage 3: concept merge -------------------------------------------
    _seed_dbg(dbg, f"[seed] ── Stage 3: concept merge (merge={do_merge}) ──")
    groups: dict[str, SeedGroup] = {}
    order: list[str] = []
    for e in kept:
        key = _canonical_key(e) if do_merge else str(e.get("entity_id"))
        w = _member_weight(e)
        if key not in groups:
            groups[key] = SeedGroup(key=key, weight=w, members=[e])
            order.append(key)
        else:
            g = groups[key]
            g.members.append(e)
            g.weight = max(g.weight, w)  # group weight = strongest member (max, not sum)
    group_list = [groups[k] for k in order]
    for g in group_list:
        members_desc = ", ".join(
            f"{m.get('label')} [{m.get('entity_type')}]" for m in g.members
        )
        _seed_dbg(dbg, f"[seed]    group '{g.key}' weight={g.weight:.3f} "
                       f"members={len(g.members)}: {members_desc}")
    _seed_dbg(dbg, f"[seed] {len(kept)} entities -> {len(group_list)} groups")

    members_flat = [m for g in group_list for m in g.members]
    best = max((_seed_signal(m) for m in members_flat), default=0.0)

    # topic concentration (trace/log only; drives nothing yet).
    mass: dict[str | None, float] = {}
    for m in members_flat:
        mass[m.get("topic_key")] = mass.get(m.get("topic_key"), 0.0) + _member_weight(m)
    total_mass = sum(mass.values())
    concentration = (max(mass.values()) / total_mass) if total_mass > 0 else 0.0

    return group_list, members_flat, best, concentration


def build_personalization_vector(
    graph: PPRGraph,
    kept_entities: Sequence[dict],
    vector_hits: Sequence[tuple[str, float]] | None = None,
    *,
    passage_seed_weight: float | None = None,
    entity_seed_weights: dict[str, float] | None = None,
) -> SeedResult:
    """Build the PPR personalization vector ``e`` for one query.

    Args:
        graph: the topic's :class:`PPRGraph`.
        kept_entities: query-matched entities (from ``graph_retriever._match_seed_entities``),
            each a dict with ``entity_id``, ``idf`` and the score fields ``_signal_score``
            reads. Weight per seed = ``signal × idf``.
        vector_hits: the vector leg's ``(chunk_id, score)`` pairs, used as passage seeds.
        passage_seed_weight: fraction (0..1) of the total seed mass given to passage
            seeds; the rest goes to entity seeds. Defaults to
            ``data_config.PPR_PASSAGE_SEED_WEIGHT`` (0.2). 0 disables passage seeds.

    Returns:
        A :class:`SeedResult`. ``abstained`` is True (and ``vector`` is None) when
        no seed maps into this graph.
    """
    lam = float(passage_seed_weight if passage_seed_weight is not None
                else _cfg("PPR_PASSAGE_SEED_WEIGHT", 0.2))
    lam = min(max(lam, 0.0), 1.0)

    n = graph.num_nodes()

    # --- entity seeds: weight = signal × idf (node specificity) -------------
    # When ``entity_seed_weights`` is given (merge mode), the weights are already
    # computed per entity_id (group weight split across members); otherwise fall
    # back to the legacy per-entity signal × idf.
    ent_w: dict[int, float] = {}
    if entity_seed_weights is not None:
        for eid, w in entity_seed_weights.items():
            _accumulate(ent_w, graph.idx_of(str(eid)), float(w))
    else:
        for ent in kept_entities:
            idx = graph.idx_of(str(ent.get("entity_id")))
            weight = _signal_score(ent) * float(ent.get("idf") or 0.0)
            _accumulate(ent_w, idx, weight)

    # --- passage seeds: weight = vector retrieval score ---------------------
    pass_w: dict[int, float] = {}
    if lam > 0.0 and vector_hits:
        for chunk_id, score in vector_hits:
            _accumulate(pass_w, graph.idx_of(str(chunk_id)), float(score))

    have_ent = bool(ent_w)
    have_pass = bool(pass_w)
    if not have_ent and not have_pass:
        return SeedResult(vector=None, abstained=True)

    # --- decide the mass split between the two groups -----------------------
    # Both present -> entity mass = (1-lam), passage mass = lam. Only one present
    # -> it takes all the mass (so an abstaining entity match still lets the
    # passage seeds drive, and vice-versa).
    if have_ent and have_pass:
        ent_share, pass_share = (1.0 - lam), lam
    elif have_ent:
        ent_share, pass_share = 1.0, 0.0
    else:
        ent_share, pass_share = 0.0, 1.0

    e = np.zeros(n, dtype=np.float64)
    seeds: list[dict] = []

    def _fill(weights: dict[int, float], share: float, kind: str) -> float:
        total = sum(weights.values())
        if total <= 0.0 or share <= 0.0:
            return 0.0
        for idx, w in weights.items():
            mass = share * (w / total)
            e[idx] += mass
            seeds.append({"id": graph.id_of(idx), "kind": kind, "weight": mass})
        return share

    ent_mass = _fill(ent_w, ent_share, "entity")
    pass_mass = _fill(pass_w, pass_share, "passage")

    # Final safety normalisation (guards against float drift).
    s = e.sum()
    if s <= 0.0:
        return SeedResult(vector=None, abstained=True)
    e /= s

    # Cross-topic share: how strongly the query's seeds landed in THIS graph.
    # Entities dominate the signal; fall back to passage strength when passage-only.
    raw_weight = sum(ent_w.values()) if have_ent else sum(pass_w.values())

    return SeedResult(
        vector=e,
        abstained=False,
        n_entity_seeds=len(ent_w),
        n_passage_seeds=len(pass_w),
        entity_mass=round(ent_mass, 4),
        passage_mass=round(pass_mass, 4),
        raw_weight=raw_weight,
        seeds=seeds,
    )


# ===========================================================================
# Stage 3 — run Personalized PageRank + read out chunks -> GraphResult
# ===========================================================================

def _edge_weights() -> EdgeWeights:
    """Edge weights for the PPR graph, read from config (Stage 5) with defaults."""
    return EdgeWeights(
        mentions=float(_cfg("PPR_MENTIONS_EDGE_WEIGHT", 1.0)),
        semantic=float(_cfg("PPR_SEMANTIC_EDGE_WEIGHT", 0.8)),
        related_to=float(_cfg("PPR_RELATED_EDGE_WEIGHT", 0.5)),
        structural=float(_cfg("PPR_STRUCTURAL_EDGE_WEIGHT", 0.3)),
    )


def _run_power_iteration(S, e: np.ndarray, alpha: float, max_iter: int, tol: float) -> np.ndarray:
    """Personalized PageRank ``π = α·S·π + (1−α)·e`` by power iteration.

    ``S`` is column-stochastic, so on a fully connected graph the mass stays at 1;
    isolated/dangling nodes leak a little, which does not affect the ranking.
    """
    pi = e.copy()
    one_minus = (1.0 - alpha) * e
    for _ in range(max_iter):
        nxt = alpha * (S @ pi) + one_minus
        if np.abs(nxt - pi).sum() < tol:
            pi = nxt
            break
        pi = nxt
    return pi


def _collect_pi_nodes(
    topic_pi: dict[str, tuple[PPRGraph, np.ndarray]],
    topic_raw: dict[str, float],
    total_raw: float,
    top_n: int,
) -> list[dict]:
    """Top-``top_n`` graph nodes by (share-weighted) PPR mass π, across topics.

    Unlike the chunk read-out in :func:`_assemble_result`, this keeps **every**
    node type (Entity, Chunk, Document, ...). It is what the retrieval-graph view
    consumes: the seeds where water was poured, the connector entities it flowed
    through, and the chunks/documents it pooled in — each with its π ("water").

    Pure read-out: it only inspects the π vectors already computed, so it never
    changes retrieval. Uses ``argpartition`` to grab each topic's top nodes cheaply
    instead of sorting the whole (13k+ node) vector.
    """
    if top_n <= 0:
        return []
    collected: list[dict] = []
    for topic, (graph, pi) in topic_pi.items():
        share = (topic_raw.get(topic, 0.0) / total_raw) if total_raw > 0 else 0.0
        if share <= 0.0:
            continue
        k = min(top_n, int(pi.shape[0]))
        if k <= 0:
            continue
        top_idx = np.argpartition(pi, -k)[-k:]
        for idx in top_idx:
            i = int(idx)
            score = float(pi[i]) * share
            if score <= 0.0:
                continue
            nid = graph.id_of(i)
            collected.append(
                {
                    "id": nid,
                    "type": graph.node_type[i],
                    "topic": topic,
                    "score": score,
                    "document": graph.chunk_document.get(nid, ""),  # set for Chunk nodes
                }
            )
    collected.sort(key=lambda d: -d["score"])
    return collected[:top_n]


def _assemble_result(
    scored: list[tuple[str, float]],
    doc_of: dict[str, str],
    best: float,
    kept: list,
    concentration: float,
    has_entities: bool,
) -> GraphResult:
    """Turn a (chunk_id, score) list into a GraphResult, reusing the existing
    floor / per-document-cap / top-k config so downstream behaviour is unchanged.
    """
    if not scored:
        return GraphResult(
            chunks=[], list_weight=0.0, abstained=True,
            matched_entities=kept, concentration=concentration,
        )
    scored.sort(key=lambda x: -x[1])
    top = scored[0][1]

    # Relative floor.
    floor = float(_cfg("GRAPH_SCORE_FLOOR_RATIO", 0.1)) * top
    survivors = [(c, s) for c, s in scored if s >= floor]

    # Per-document diversity cap.
    doc_cap = max(1, int(_cfg("GRAPH_MAX_CHUNKS_PER_DOC", 2)))
    capped: list[tuple[str, float]] = []
    counts: dict[str, int] = {}
    for cid, s in survivors:
        d = doc_of.get(cid)
        if counts.get(d, 0) >= doc_cap:
            continue
        counts[d] = counts.get(d, 0) + 1
        capped.append((cid, s))

    # Top-k (normal vs. low-confidence fallback), same knobs as the legacy leg.
    fallback = (best < float(_cfg("GRAPH_ENTITY_MATCH_FLOOR", 0.6))) if has_entities else True
    k = int(_cfg("GRAPH_FALLBACK_K", 3)) if fallback else int(_cfg("GRAPH_CANDIDATE_K", 30))
    ranked = capped[:k]

    # Top-N recommended documents by MaxP over the (pre-cut) scores.
    doc_best: dict[str, float] = {}
    for cid, s in scored:
        d = doc_of.get(cid)
        if d and s > doc_best.get(d, 0.0):
            doc_best[d] = s
    documents = [
        {"document": d, "score": s}
        for d, s in sorted(doc_best.items(), key=lambda x: -x[1])[: int(_cfg("GRAPH_DOC_TOPN", 10))]
    ]

    list_weight = max(best, float(_cfg("GRAPH_LIST_WEIGHT_MIN", 0.4)))
    return GraphResult(
        chunks=ranked,
        list_weight=list_weight,
        abstained=False,
        documents=documents,
        matched_entities=kept,
        concentration=concentration,
        fallback_mode=fallback,
    )


def _candidate_topics(graph_root: Path, kept: list, debug: bool = False) -> list[str]:
    """Topics to run PPR over.

    Base rule: the topics the query's entities live in (else all topics).

    Stage 6 (``PPR_TOPIC_ISOLATION``): sum seed weight (``max(fused,dense) x idf``)
    per topic and keep only the dominant topic plus any within
    ``PPR_TOPIC_KEEP_RATIO`` of it — so a stray seed in a foreign topic (e.g. a
    same-named ``SAP S/4HANA``) can't pull the walk there.
    """
    subdirs = {d.name for d in graph_root.iterdir() if d.is_dir()}
    if not subdirs:
        return []
    wanted = {str(e.get("topic_key")).lower() for e in kept if e.get("topic_key")}
    # match case-insensitively against the folder names
    chosen = [name for name in subdirs if name.lower() in wanted]
    if not chosen:
        return sorted(subdirs)

    if not bool(_cfg("PPR_TOPIC_ISOLATION", False)):
        return chosen

    # --- Stage 6: keep the dominant seed topic(s) by summed seed mass ---------
    mass: dict[str, float] = {}
    for e in kept:
        t = str(e.get("topic_key") or "").lower()
        if t:
            mass[t] = mass.get(t, 0.0) + _member_weight(e)
    if not mass:
        return chosen
    top = max(mass.values())
    ratio = float(_cfg("PPR_TOPIC_KEEP_RATIO", 0.5))
    keep = {t for t, m in mass.items() if top > 0.0 and m >= ratio * top}
    isolated = [name for name in chosen if name.lower() in keep]
    if debug:
        print(f"[seed] ── Stage 6: topic isolation (keep >= {ratio:.0%} of top) ──")
        for t, m in sorted(mass.items(), key=lambda x: -x[1]):
            print(f"[seed]    topic {t:<22} mass={m:8.3f}  "
                  f"{'KEEP' if t in keep else 'drop'}")
    return isolated or chosen


def ppr_retrieve(
    query: str,
    graph_root: Path | str,
    vector_hits: list[tuple[str, float]] | None = None,
    debug: bool = False,
    topk_pi_nodes: int = 0,
    entity_collection: str | None = None,
) -> GraphResult:
    """Stage-3 entry point: query -> PPR over the (semantic + structural) graph -> chunks.

    Drop-in for ``graph_retriever.retrieve``: returns the same ``GraphResult`` so the
    fusion pool and reranker are untouched. ``vector_hits`` are the vector leg's
    ``(chunk_id, score)`` pairs; pass them for dense passage seeds (hybrid), or omit
    for a fully independent entity-seeded graph leg (the graph-only ablation).

    ``topk_pi_nodes`` (0 = off): when > 0, also attach the top-N graph nodes by π
    (all node types) to ``result.pi_nodes`` for the retrieval-graph view. Pure
    read-out — it does not affect the returned chunks in any way.
    """
    graph_root = Path(graph_root)
    weights = _edge_weights()
    alpha = float(_cfg("PPR_ALPHA", 0.5))
    max_iter = int(_cfg("PPR_MAX_ITER", 100))
    tol = float(_cfg("PPR_TOL", 1e-6))

    # 1. Seed selection. "merge" = new pipeline (threshold -> boilerplate filter
    #    -> concept merge, Stages 1-3); "legacy" = the old top-8-by-fused gate.
    seed_mode = str(_cfg("PPR_SEED_MODE", "legacy")).lower()
    dbg = debug or bool(_cfg("PPR_SEED_DEBUG", False))
    entity_seed_weights: dict[str, float] | None = None
    from time import perf_counter as _pc
    _tg = _pc()

    if seed_mode == "merge":
        groups, kept, best, concentration = _select_seeds(query, debug=debug, entity_collection=entity_collection)
        has_entities = bool(groups)
        # Stage 4: pour each group's weight, split evenly across its member entity
        # nodes (this pools their chunks). Group total = its (max) weight.
        entity_seed_weights = {}
        for g in groups:
            n = max(1, len(g.members))
            for m in g.members:
                eid = str(m.get("entity_id"))
                entity_seed_weights[eid] = entity_seed_weights.get(eid, 0.0) + g.weight / n
        if dbg:
            print("[seed] ── Stage 4: pour water (entity seed weights) ──")
            for eid, w in sorted(entity_seed_weights.items(), key=lambda x: -x[1])[:15]:
                print(f"[seed]    pour {w:.4f}  {eid}")
            print(f"[seed] ── Stage 5: seeding ppr_retrieve (mode=merge, "
                  f"{len(groups)} groups, {len(entity_seed_weights)} nodes) ──")
    else:
        # Entity gating (reuse the legacy match: floor, cap, idf checks).
        matched = _match_seed_entities(query, entity_collection=entity_collection)
        if matched is None:
            kept, best, concentration = [], 0.0, 0.0
        else:
            kept, best, concentration = matched
        has_entities = bool(kept)

    # Abstain early only when there is nothing at all to seed from.
    if not has_entities and not vector_hits:
        return GraphResult(chunks=[], list_weight=0.0, abstained=True)

    _t_seeds = _pc() - _tg; _tg = _pc()
    # 2. Per topic: seed -> PPR -> collect chunk scores. Topics are scored
    #    independently (their graphs share no edges); we combine them by the
    #    seed strength that landed in each (cross-topic share).
    per_chunk: dict[str, float] = {}
    doc_of: dict[str, str] = {}
    topic_raw: dict[str, float] = {}
    topic_pi: dict[str, tuple[PPRGraph, np.ndarray]] = {}

    for topic in _candidate_topics(graph_root, kept, debug=dbg):
        graph = get_ppr_graph(graph_root, topic, weights)
        seed = build_personalization_vector(
            graph, kept, vector_hits, entity_seed_weights=entity_seed_weights
        )
        if seed.abstained or seed.vector is None:
            _seed_dbg(dbg, f"[seed]    topic {topic}: no seed landed (skipped)")
            continue
        _seed_dbg(dbg, f"[seed]    topic {topic}: {seed.n_entity_seeds} entity + "
                       f"{seed.n_passage_seeds} passage nodes seeded, raw_weight={seed.raw_weight:.3f}")
        pi = _run_power_iteration(graph.transition_matrix(), seed.vector, alpha, max_iter, tol)
        topic_raw[topic] = seed.raw_weight
        topic_pi[topic] = (graph, pi)

    if not topic_pi:
        return GraphResult(
            chunks=[], list_weight=0.0, abstained=(not has_entities),
            matched_entities=kept, concentration=concentration,
        )

    total_raw = sum(topic_raw.values()) or 1.0
    for topic, (graph, pi) in topic_pi.items():
        share = topic_raw.get(topic, 0.0) / total_raw
        if share <= 0.0:
            continue
        for idx in graph.chunk_idx:
            score = float(pi[idx]) * share
            if score <= 0.0:
                continue
            cid = graph.id_of(int(idx))
            if score > per_chunk.get(cid, 0.0):
                per_chunk[cid] = score
                doc_of[cid] = graph.chunk_document.get(cid, "")

    _t_ppr = _pc() - _tg; _tg = _pc()
    scored = list(per_chunk.items())
    result = _assemble_result(scored, doc_of, best, kept, concentration, has_entities)
    print(f"[LAT] graph_phases  seeds={_t_seeds:.2f}s  ppr={_t_ppr:.2f}s  "
          f"assemble={_pc()-_tg:.2f}s", flush=True)

    # Optional: expose the top-π nodes (all types) for the retrieval-graph view.
    if topk_pi_nodes > 0:
        result.pi_nodes = _collect_pi_nodes(topic_pi, topic_raw, total_raw, topk_pi_nodes)

    if debug:
        result.trace = {
            "query": query,
            "mode": "ppr",
            "seed_mode": seed_mode,
            "topics": list(topic_pi.keys()),
            "entity_seeds": len(kept),
            "seed_groups": len(entity_seed_weights) if entity_seed_weights is not None else None,
            "passage_seeds": len(vector_hits or []),
            "alpha": alpha,
            "final_chunks": len(result.chunks),
            "pi_nodes": len(result.pi_nodes),
        }
    return result
