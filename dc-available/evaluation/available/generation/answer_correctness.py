"""Compute RAGAS answer_correctness for AVAILABLE 2.0 (hybrid answers).

AVAILABLE 1.0 reported answer_correctness, but the 2.0 RAGAS files saved
context_recall instead. This script fills the gap so the fourth generation
metric can be compared across versions. It scores the answers already generated
for 2.0 (no re-retrieval, no re-generation) with the SAME default RAGAS
answer_correctness metric used elsewhere in the repo (weights 0.75 factual /
0.25 semantic), the project judge, and the local BGE-M3 embeddings.

Comparability note: the metric definition matches 1.0 (default answer_correctness).
The judge is set by JUDGE_MODEL below (gpt-4o). This should match the judge used for
the 1.0 answer_correctness run; change it if 1.0 used a different judge.

Inputs (read only, already contain the generated answers):
  evaluation/results/RAGAS/grownfield/grownfield_with_answers.json   (n=23)
  evaluation/results/RAGAS/hydraulics/hydraulics_with_answers.json   (n=59)
  evaluation/results/synthetic/generation.json                       (n=1215)

Outputs (new folder evaluation/results/RAGAS/answer_correctness/):
  grownfield.jsonl / hydraulics.jsonl / synthetic.jsonl   per-row scores
  summary.json   per-dataset mean + synthetic mean by question_type

Resumable per (dataset, question_id): rerun continues where it stopped.

RUN (from the dc-available directory):
  poetry run python -m evaluation.available.generation.answer_correctness 6   # dry-run: 6 rows/dataset
  poetry run python -m evaluation.available.generation.answer_correctness     # full
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Keep a local "datasets" folder from shadowing HuggingFace's "datasets".
_HERE = Path(__file__).resolve().parent
_BAD = {str(_HERE), str(_HERE.parents[0])}
sys.path[:] = [p for p in sys.path if str(Path(p or ".").resolve()) not in _BAD]

RES = Path("evaluation/results")
OUT = RES / "RAGAS" / "answer_correctness"
JUDGE_MODEL = "gpt-4o"

# (name, path, id_field, answer_field, context_field, golden_field, qtype_field)
DATASETS = [
    ("grownfield", RES / "RAGAS" / "grownfield" / "grownfield_with_answers.json"),
    ("hydraulics", RES / "RAGAS" / "hydraulics" / "hydraulics_with_answers.json"),
    ("synthetic", RES / "synthetic" / "generation.json"),
]


def _build_ragas_llm_emb():
    from dc_available.config.llm_config import get_langchain_llm
    from dc_available.RAG.retrieval_helpers import get_embeddings
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    llm = LangchainLLMWrapper(get_langchain_llm(model_name=JUDGE_MODEL))
    emb = LangchainEmbeddingsWrapper(get_embeddings())
    return llm, emb


def _load_done(path: Path) -> set:
    done = set()
    if path.is_file():
        for line in path.open(encoding="utf-8"):
            done.add(json.loads(line)["question_id"])
    return done


def _usable(r: dict) -> bool:
    ans = (r.get("generated_answer") or "").strip()
    ctx = [c for c in (r.get("retrieved_contexts") or []) if c]
    gold = (r.get("golden_answer") or "").strip()
    return bool(ans and ctx and gold)


def score_dataset(name: str, path: Path, llm, emb, metric, limit: int | None):
    from datasets import Dataset

    if not path.is_file():
        print(f"[{name}] MISSING {path} — skipped")
        return
    rows = json.loads(path.read_text(encoding="utf-8"))
    if limit:
        rows = rows[:limit]

    out_path = OUT / f"{name}.jsonl"
    done = _load_done(out_path)
    todo = [r for r in rows if r["question_id"] not in done and _usable(r)]
    skipped = [r for r in rows if r["question_id"] not in done and not _usable(r)]
    if not todo:
        print(f"[{name}] nothing to score ({len(done)} done, {len(skipped)} unusable)")
        return

    print(f"[{name}] scoring {len(todo)} rows (many LLM calls) ...", flush=True)
    ds = Dataset.from_dict({
        "question": [r["question"] for r in todo],
        "answer": [r["generated_answer"] for r in todo],
        "contexts": [[c for c in r["retrieved_contexts"] if c] for r in todo],
        "ground_truth": [r["golden_answer"] for r in todo],
    })

    try:
        from ragas.run_config import RunConfig
        run_config = RunConfig(max_workers=4, timeout=300)
    except Exception:
        run_config = None

    kwargs = dict(metrics=[metric], llm=llm, embeddings=emb, raise_exceptions=False)
    if run_config:
        kwargs["run_config"] = run_config

    from ragas import evaluate
    result = evaluate(ds, **kwargs)
    scores = result.to_pandas()["answer_correctness"].tolist()

    with out_path.open("a", encoding="utf-8") as fh:
        for r, s in zip(todo, scores):
            val = None if s is None or (isinstance(s, float) and s != s) else round(float(s), 4)
            fh.write(json.dumps({
                "question_id": r["question_id"],
                "question_type": r.get("question_type"),
                "answer_correctness": val,
            }, ensure_ascii=False) + "\n")
    ok = [s for s in scores if s is not None and s == s]
    mean = f"{sum(ok) / len(ok):.4f}" if ok else "n/a"
    print(f"[{name}] {len(ok)}/{len(todo)} scored, batch mean={mean}", flush=True)


def write_summary():
    from collections import defaultdict

    summary = {}
    for name, _ in DATASETS:
        path = OUT / f"{name}.jsonl"
        if not path.is_file():
            continue
        rows = [json.loads(line) for line in path.open(encoding="utf-8")]
        vals = [r["answer_correctness"] for r in rows
                if isinstance(r.get("answer_correctness"), (int, float))]
        overall = round(sum(vals) / len(vals), 4) if vals else None
        if name == "synthetic":
            by_type = {}
            groups = defaultdict(list)
            for r in rows:
                if isinstance(r.get("answer_correctness"), (int, float)):
                    groups[r.get("question_type")].append(r["answer_correctness"])
            for t, vs in groups.items():
                by_type[t] = round(sum(vs) / len(vs), 4)
            summary[name] = {"n": len(vals), "answer_correctness": overall,
                             "by_question_type": by_type}
        else:
            summary[name] = {"n": len(vals), "answer_correctness": overall}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nSUMMARY")
    print(json.dumps(summary, indent=2))


def main(limit: int | None = None):
    OUT.mkdir(parents=True, exist_ok=True)
    from ragas.metrics import answer_correctness   # default weights [0.75, 0.25]
    llm, emb = _build_ragas_llm_emb()
    for name, path in DATASETS:
        score_dataset(name, path, llm, emb, answer_correctness, limit)
    write_summary()


if __name__ == "__main__":
    main(limit=int(sys.argv[1]) if len(sys.argv) > 1 else None)
