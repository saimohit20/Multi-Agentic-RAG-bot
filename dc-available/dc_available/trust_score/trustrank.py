"""Stage 2 — TrustRank: propagate the metadata prior over a corroboration graph.

Same PageRank math as the PPR retriever, but the graph is document<->document
(edges = shared high-IDF entities) and the seed is the metadata *prior* instead
of query entities. Trust flows from reliable sources to documents that
corroborate their SPECIFIC claims.

    t = α · S · t + (1 − α) · p

    S : column-normalized doc<->doc adjacency  (A · D⁻¹)
    p : normalized metadata prior (from score_prior.py)
    A_ab = Σ IDF(entities shared by docs a and b)

Only the 183 documents that appear in the graph propagate. The 94 text-less
videos carry no entities, so they keep their prior (no corroboration possible).

Input : metadata_prior.jsonl  +  graph nodes.jsonl / edges.jsonl
Output: metadata_trust.jsonl  (adds in_graph, trust_raw, trust)

RUN (from the dc-available directory):
    python -m dc_available.trust_score.trustrank
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

# --- tunables --------------------------------------------------------------- #
ALPHA = 0.5
MAX_ITER = 100
TOL = 1e-6
# Final blend: keep the metadata prior in charge, let corroboration nudge it.
# (w_prior + w_trust must = 1, so non-graph docs collapse to final_trust = prior.)
W_PRIOR = 0.7
W_TRUST = 0.3
MAX_DF_FRACTION = 0.5   # drop entities present in >50% of docs (stopword-like)
MIN_DF = 2              # an entity in only 1 doc can't corroborate anything

_DATA = Path(__file__).resolve().parent.parent / "data"
_TRUST = _DATA / "Artifacts" / "resources" / "trust" / "grownfield"
_GRAPH = _DATA / "Artifacts" / "resources" / "graph" / "grownfield"
_PRIOR = _TRUST / "metadata_prior.jsonl"
_OUT = _TRUST / "metadata_trust.jsonl"


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8")]


# --------------------------------------------------------------------------- #
# 1. entity <-> document from the graph
# --------------------------------------------------------------------------- #
def _entity_document_maps(graph_dir: Path = _GRAPH) -> tuple[dict[str, set], dict[str, set]]:
    nodes = _read_jsonl(graph_dir / "nodes.jsonl")
    edges = _read_jsonl(graph_dir / "edges.jsonl")

    node_type = {str(n["id"]): n.get("type", "") for n in nodes}
    chunk_doc: dict[str, str] = {}
    for n in nodes:
        if n.get("type") == "Chunk":
            d = (n.get("properties") or {}).get("document_id")
            if d:
                chunk_doc[str(n["id"])] = str(d).strip().lower()

    doc_entities: dict[str, set] = defaultdict(set)
    entity_docs: dict[str, set] = defaultdict(set)
    for e in edges:
        if e.get("type") != "mentions":
            continue
        s, t = str(e.get("source")), str(e.get("target"))
        if node_type.get(s) == "Chunk" and node_type.get(t) == "Entity":
            chunk, ent = s, t
        elif node_type.get(t) == "Chunk" and node_type.get(s) == "Entity":
            chunk, ent = t, s
        else:
            continue
        doc = chunk_doc.get(chunk)
        if not doc:
            continue
        doc_entities[doc].add(ent)
        entity_docs[ent].add(doc)
    return doc_entities, entity_docs


# --------------------------------------------------------------------------- #
# 2. doc<->doc transition matrix (IDF-weighted shared entities)
# --------------------------------------------------------------------------- #
def _build_transition(doc_entities, entity_docs):
    docs = sorted(doc_entities)
    idx = {d: i for i, d in enumerate(docs)}
    n = len(docs)

    pair_w: dict[tuple[int, int], float] = defaultdict(float)
    for ent, ds in entity_docs.items():
        df = len(ds)
        if df < MIN_DF or df > MAX_DF_FRACTION * n:
            continue
        idf = math.log(n / df)
        members = sorted(idx[d] for d in ds)
        for i in range(len(members)):
            a = members[i]
            for j in range(i + 1, len(members)):
                pair_w[(a, members[j])] += idf

    # Dense is fine: n≈183, so A is ~183x183 — tiny.
    A = np.zeros((n, n), dtype=np.float64)
    for (a, b), w in pair_w.items():
        A[a, b] += w
        A[b, a] += w

    deg = A.sum(axis=0)
    inv = np.divide(1.0, deg, out=np.zeros_like(deg), where=deg > 0)
    S = A * inv          # column-normalize: column j scaled by 1/deg[j]
    return docs, idx, S, deg


# --------------------------------------------------------------------------- #
# 3. power iteration
# --------------------------------------------------------------------------- #
def _power_iteration(S, p) -> tuple[np.ndarray, int]:
    t = p.copy()
    for it in range(1, MAX_ITER + 1):
        t_new = ALPHA * (S @ t) + (1.0 - ALPHA) * p
        if np.abs(t_new - t).sum() < TOL:
            return t_new, it
        t = t_new
    return t, MAX_ITER


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def run(
    graph_dir: Path = _GRAPH,
    prior_path: Path = _PRIOR,
    out_path: Path = _OUT,
) -> None:
    prior_rows = _read_jsonl(prior_path)
    prior_by_key = {r["match_key"]: r["prior"] for r in prior_rows}

    doc_entities, entity_docs = _entity_document_maps(graph_dir)
    docs, idx, S, deg = _build_transition(doc_entities, entity_docs)
    n = len(docs)

    # seed = normalized prior over the graph docs
    p = np.array([prior_by_key.get(d, 0.0) for d in docs], dtype=np.float64)
    if p.sum() <= 0:
        raise RuntimeError("prior vector is empty — check the join keys")
    p = p / p.sum()

    t, iters = _power_iteration(S, p)
    trust_norm = t / t.max()          # 0–1, best doc = 1.0

    trust_by_doc = {docs[i]: (float(t[i]), float(trust_norm[i])) for i in range(n)}

    # write all 277: graph docs get propagated trust, videos keep their prior
    n_isolated = int((deg == 0).sum())
    n_video_fallback = 0
    for r in prior_rows:
        tv = trust_by_doc.get(r["match_key"])
        if tv is not None:
            r["in_graph"] = True
            r["trust_raw"] = round(tv[0], 8)
            r["trust"] = round(tv[1], 4)
        else:
            r["in_graph"] = False
            r["trust_raw"] = None
            r["trust"] = r["prior"]      # fallback: no entities to corroborate
            n_video_fallback += 1
        # Final trust = prior kept in charge, corroboration nudges it.
        r["final_trust"] = round(W_PRIOR * r["prior"] + W_TRUST * r["trust"], 4)

    with out_path.open("w", encoding="utf-8") as fh:
        for r in prior_rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    # report
    print(f"graph docs propagated : {n}")
    print(f"  edges (nnz)         : {int(np.count_nonzero(S))}")
    print(f"  isolated (no shared entity): {n_isolated}")
    print(f"  converged in        : {iters} iterations")
    print(f"non-graph (video) docs: {n_video_fallback}  (trust = prior)")
    print(f"-> {out_path.name}")

    finals = [r["final_trust"] for r in prior_rows]
    priors = [r["prior"] for r in prior_rows]
    mf, mp = sum(finals) / len(finals), sum(priors) / len(priors)
    cov = sum((a - mf) * (b - mp) for a, b in zip(finals, priors))
    den = (sum((a - mf) ** 2 for a in finals) * sum((b - mp) ** 2 for b in priors)) ** 0.5
    print(f"\nfinal_trust  min/mean/max : "
          f"{min(finals):.3f} / {mf:.3f} / {max(finals):.3f}")
    print(f"corr(final_trust, prior)  : {cov / den:.3f}   "
          f"(should be high — prior stays in charge)")

    print("\nTOP 5 final_trust:")
    for r in sorted(prior_rows, key=lambda x: x["final_trust"], reverse=True)[:5]:
        print(f"  final={r['final_trust']:.3f}  prior={r['prior']:.3f}  "
              f"trust={r['trust']:.3f}  {r['match_key'][:44]}")
    print("\nBOTTOM 5 final_trust:")
    for r in sorted(prior_rows, key=lambda x: x["final_trust"])[:5]:
        print(f"  final={r['final_trust']:.3f}  prior={r['prior']:.3f}  "
              f"trust={r['trust']:.3f}  {r['match_key'][:44]}")


if __name__ == "__main__":
    run()
