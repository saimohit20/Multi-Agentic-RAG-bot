"""Step 1 of RAGAS pipeline: generate answers for hybrid rows using Azure LLM.

Reads  evaluation/results/available_2.0/{topic}/{topic}.json
Writes evaluation/results/RAGAS/{topic}/{topic}_with_answers.json

Answers are produced from the retrieved_contexts already stored in the JSON
(no second retrieval pass). Safe to re-run: rows with an existing
generated_answer are skipped.

Standalone usage (from project root):
  poetry run python -m evaluation.available.generation.generate
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

TOPICS = ["grownfield", "hydraulics"]
AVAILABLE_DIR = Path("evaluation/results/available_2.0")
RAGAS_DIR = Path("evaluation/results/RAGAS")
JUDGE_MODEL = "gpt-4o-mini"

_SYSTEM = (
    "You are a technical assistant. Answer the question based solely on the "
    "provided context. Be concise, accurate, and self-contained. "
    "Do not say 'according to the context' or similar phrases."
)


def _build_llm():
    from dc_available.config.llm_config import get_langchain_llm
    return get_langchain_llm(model_name=JUDGE_MODEL)


def _generate(llm, question: str, contexts: list[str]) -> str:
    from langchain.schema import HumanMessage, SystemMessage
    ctx = "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(contexts))
    msgs = [
        SystemMessage(content=_SYSTEM),
        HumanMessage(content=f"Context:\n{ctx}\n\nQuestion: {question}"),
    ]
    return llm.invoke(msgs).content.strip()


def generate_for_topic(
    topic: str,
    llm,
    src_dir: Path = AVAILABLE_DIR,
    ragas_dir: Path = RAGAS_DIR,
    src_file: Path | None = None,
    out_file: Path | None = None,
) -> Path:
    # Explicit paths (src_file/out_file) override the default {dir}/{topic}/ layout
    # so callers can use a custom folder (e.g. retrieval_eye/<topic>/{retrieval,generation}/).
    src = src_file or (src_dir / topic / f"{topic}.json")
    if not src.exists():
        sys.exit(f"Missing {src}  —  run available.py first.")

    all_records = json.loads(src.read_text(encoding="utf-8"))
    hybrid = [r for r in all_records if r.get("retrieval_type") == "hybrid"]
    print(f"  {len(hybrid)} hybrid rows")

    out_file = out_file or (ragas_dir / topic / f"{topic}_with_answers.json")
    cached: dict[str, str] = {}
    if out_file.exists():
        for r in json.loads(out_file.read_text(encoding="utf-8")):
            if r.get("generated_answer"):
                cached[r["question_id"]] = r["generated_answer"]
        print(f"  {len(cached)} already generated (will skip)")

    results = []
    for i, row in enumerate(hybrid, 1):
        qid = row["question_id"]
        if qid in cached:
            print(f"  [{i}/{len(hybrid)}] {qid}: cached")
            results.append({**row, "generated_answer": cached[qid]})
            continue
        contexts = [c for c in (row.get("retrieved_contexts") or []) if c]
        if not contexts:
            print(f"  [{i}/{len(hybrid)}] {qid}: no contexts — skipped")
            results.append({**row, "generated_answer": ""})
            continue
        try:
            ans = _generate(llm, row["question"], contexts)
            print(f"  [{i}/{len(hybrid)}] {qid}: OK ({len(ans)} chars)")
        except Exception as exc:
            print(f"  [{i}/{len(hybrid)}] {qid}: ERROR {exc}")
            ans = ""
        results.append({**row, "generated_answer": ans})

    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  → {out_file}")
    return out_file


def main(
    topics: list[str] | None = None,
    src_dir: Path = AVAILABLE_DIR,
    ragas_dir: Path = RAGAS_DIR,
) -> None:
    topics = topics or TOPICS
    llm = _build_llm()
    for topic in topics:
        print(f"\n=== [generate] {topic} ===")
        generate_for_topic(topic, llm, src_dir=src_dir, ragas_dir=ragas_dir)


if __name__ == "__main__":
    # Drop the eval dirs (so local evaluation/datasets can't shadow HF "datasets"),
    # but KEEP the project root (dc-available) on sys.path so dc_available imports.
    _here = Path(__file__).resolve().parent
    _root = _here.parents[2]                                   # .../dc-available
    _bad = {str(_here), str(_here.parent), str(_here.parents[1])}  # generation, available, evaluation
    sys.path[:] = [p for p in sys.path if str(Path(p or ".").resolve()) not in _bad]
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

    # CLI: generate <topic> [src_file] [out_file]
    #   no args -> default two-topic run on results/available_2.0 -> results/RAGAS
    args = sys.argv[1:]
    if args:
        topic = args[0]
        src_file = Path(args[1]) if len(args) > 1 else None
        out_file = Path(args[2]) if len(args) > 2 else None
        print(f"=== [generate] {topic} ===")
        generate_for_topic(topic, _build_llm(), src_file=src_file, out_file=out_file)
    else:
        main()
