"""Step 3 — full trust (prior + corroboration) for the 40 test docs.

Good docs keep their REAL final_trust (already computed over the 277-doc graph).
Twins get corroboration by DERIVATION (approach b): a twin = its good doc's graph
entities MINUS the flipped facts (the made-up wrong values would be df<2 orphans,
so we simply don't add them). We inject the 20 twins into the grownfield graph and
run the same TrustRank PPR, then  twin_final = 0.7·prior + 0.3·trust.

We expect the twin's corroboration to sit CLOSE to its good doc (it's a near-clone)
— that's the point: corroboration can't separate a clone; only the prior can.

Outputs (rag_ab/), keyed by the TEST ids (g01 / g01_twin) so final_trust resolves them:
  - test_trust_prior.jsonl   final_trust = PRIOR only        (ablation arm A1)
  - test_trust_full.jsonl    final_trust = prior+corroboration (arm A2)

RUN (from the dc-available directory):
    python -m evaluation.trust_score.rag_ab.graph_twins
"""
from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np

from dc_available.trust_score.trustrank import (
    _GRAPH,
    _PRIOR,
    _build_transition,
    _power_iteration,
    _read_jsonl,
    W_PRIOR,
    W_TRUST,
)
from dc_available.trust_score.components.score_content import _TRUST_DIR
from dc_available.trust_score.components.score_prior import (
    AUTHORITY,
    WEIGHTS,
    authority_score,
    format_score,
    freshness_score,
)
from evaluation.trust_score.rag_ab.config import AB_DIR, DATASET_FILE, DOCS_FILE

TEST_TRUST_PRIOR = AB_DIR / "test_trust_prior.jsonl"
TEST_TRUST_FULL = AB_DIR / "test_trust_full.jsonl"
_META_TRUST = _TRUST_DIR / "metadata_trust.jsonl"
_BACKDATE_YEARS = 4


def _backdate(modified_iso: str, years: int) -> str:
    if not modified_iso:
        return modified_iso
    try:
        dt = datetime.fromisoformat(modified_iso.replace("Z", "+00:00"))
    except ValueError:
        return modified_iso
    try:
        return dt.replace(year=dt.year - years).isoformat()
    except ValueError:
        return dt.replace(year=dt.year - years, day=28).isoformat()


def _twin_prior(d: dict, now: datetime) -> float:
    content = float(d.get("content_score") or 1.0)
    fmt = format_score(d.get("file_type", ""))
    auth = authority_score(d.get("created_by", ""), d.get("modified_by", ""))
    fresh = freshness_score(d.get("modified", ""), now)
    if d["lever"] == "freshness":
        fresh = freshness_score(_backdate(d.get("modified", ""), _BACKDATE_YEARS), now)
    else:
        auth = AUTHORITY["unknown"]
    return round(WEIGHTS["content"] * content + WEIGHTS["authority"] * auth
                 + WEIGHTS["format"] * fmt + WEIGHTS["freshness"] * fresh, 4)


def _graph_maps():
    """doc(lower) -> set(entity_id), entity_id -> set(doc), doc -> {entity_id: label}."""
    nodes = _read_jsonl(_GRAPH / "nodes.jsonl")
    edges = _read_jsonl(_GRAPH / "edges.jsonl")
    chunk_doc = {}
    for n in nodes:
        if n.get("type") == "Chunk":
            d = (n.get("properties") or {}).get("document_id")
            if d:
                chunk_doc[str(n["id"])] = str(d).strip().lower()

    doc_entities: dict[str, set] = defaultdict(set)
    entity_docs: dict[str, set] = defaultdict(set)
    doc_labels: dict[str, dict] = defaultdict(dict)
    for e in edges:
        if e.get("type") != "mentions":
            continue
        s, t = str(e.get("source")), str(e.get("target"))
        doc = chunk_doc.get(s) or chunk_doc.get(t)
        ent = t if chunk_doc.get(s) else s
        if not doc:
            continue
        doc_entities[doc].add(ent)
        entity_docs[ent].add(doc)
        lbl = (e.get("properties") or {}).get("entity_label", "")
        if lbl:
            doc_labels[doc][ent] = lbl
    return doc_entities, entity_docs, doc_labels


def run() -> None:
    docs = [json.loads(l) for l in DOCS_FILE.open(encoding="utf-8")]
    data = [json.loads(l) for l in DATASET_FILE.open(encoding="utf-8")]
    now = datetime.now(timezone.utc)

    flipped: dict[str, set] = defaultdict(set)
    for r in data:
        flipped[r["good_doc_id"]].add(str(r["correct_value"]).lower())

    prior_by_key = {r["match_key"]: r["prior"] for r in _read_jsonl(_PRIOR)}
    final_by_key = {r["match_key"]: r["final_trust"] for r in _read_jsonl(_META_TRUST)}
    doc_entities, entity_docs, doc_labels = _graph_maps()

    # inject twins: good entities minus the flipped ones
    twin_info = {}
    for d in docs:
        did, mk, lever = d["doc_id"], d["match_key"], d["lever"]
        good_ents = set(doc_entities.get(mk, set()))
        labels = doc_labels.get(mk, {})
        cvs = flipped.get(did, set())
        drop = {e for e in good_ents
                if any(cv in labels.get(e, "").lower() or cv in e.lower() for cv in cvs)}
        twin_ents = good_ents - drop
        tk = mk + "__twin"
        if twin_ents:                    # in-graph only; else prior fallback (like trustrank)
            doc_entities[tk] = twin_ents
            for e in twin_ents:
                entity_docs[e].add(tk)
        twin_info[tk] = {"doc_id": did, "prior": _twin_prior(d, now),
                         "n_ent": len(good_ents), "n_drop": len(drop),
                         "in_graph": bool(twin_ents)}

    # PPR over the extended graph (277 real + 20 twins)
    docs_list, idx, S, deg = _build_transition(doc_entities, entity_docs)
    prior_map = dict(prior_by_key)
    for tk, info in twin_info.items():
        prior_map[tk] = info["prior"]
    p = np.array([prior_map.get(dk, 0.0) for dk in docs_list], dtype=np.float64)
    p = p / p.sum()
    t, iters = _power_iteration(S, p)
    trust_norm = t / t.max()
    trust_by_doc = {docs_list[i]: float(trust_norm[i]) for i in range(len(docs_list))}

    prior_rows, full_rows, report = [], [], []
    for d in docs:
        did, mk = d["doc_id"], d["match_key"]
        tk = mk + "__twin"
        gp = prior_by_key.get(mk, 0.0)
        gt = trust_by_doc.get(mk)                    # None -> good has no corroboration
        good_final = round(W_PRIOR * gp + W_TRUST * gt, 4) if gt is not None else round(gp, 4)
        tp = twin_info[tk]["prior"]
        tt = trust_by_doc.get(tk)                    # None -> twin fallback = prior
        twin_final = round(W_PRIOR * tp + W_TRUST * tt, 4) if tt is not None else round(tp, 4)

        prior_rows += [{"match_key": did, "final_trust": round(gp, 4)},
                       {"match_key": f"{did}_twin", "final_trust": round(tp, 4)}]
        full_rows += [{"match_key": did, "final_trust": good_final},
                      {"match_key": f"{did}_twin", "final_trust": twin_final}]
        report.append((did, twin_info[tk], gt, tt, good_final, twin_final, final_by_key.get(mk)))

    with TEST_TRUST_PRIOR.open("w", encoding="utf-8") as fh:
        for r in prior_rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    with TEST_TRUST_FULL.open("w", encoding="utf-8") as fh:
        for r in full_rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_in = sum(1 for _, i, *_ in report if i["in_graph"])
    print(f"PPR over {len(docs_list)} docs, converged in {iters} iters")
    print(f"good docs with entity mentions: {n_in}/20  (others -> prior fallback)\n")
    print(f"{'id':4} {'ent':>4} {'drop':>4} {'good_tr':>8} {'twin_tr':>8} "
          f"{'good_fin':>9} {'twin_fin':>9} {'real_fin':>9}")
    print("-" * 66)
    for did, info, gt, tt, gf, tfin, realf in report:
        gts = f"{gt:.3f}" if gt is not None else "   -"
        tts = f"{tt:.3f}" if tt is not None else "   -"
        rfs = f"{realf:.3f}" if realf is not None else "   -"
        print(f"{did:4} {info['n_ent']:>4} {info['n_drop']:>4} {gts:>8} {tts:>8} "
              f"{gf:>9.3f} {tfin:>9.3f} {rfs:>9}")
    print(f"\n-> {TEST_TRUST_PRIOR.name} + {TEST_TRUST_FULL.name}")


if __name__ == "__main__":
    run()
