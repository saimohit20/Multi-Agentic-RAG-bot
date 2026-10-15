"""Step 2 of RAGAS pipeline: score generated answers with 4 RAGAS metrics.

Reads  evaluation/results/RAGAS/{topic}/{topic}_with_answers.json
Writes evaluation/results/RAGAS/{topic}/
  scores.json        per-row metric values + question_id
  scores.csv         same as CSV
  summary.json       mean per metric
  plot_ragas.png     bar chart

Judge LLM:   Azure OpenAI gpt-4o-mini  (no extra keys needed)
Embeddings:  project BGE-M3            (answer_relevancy uses cosine sim)

Standalone usage (from project root):
  poetry run python -m evaluation.available.generation.ragas_eval
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

TOPICS = ["grownfield", "hydraulics"]
RAGAS_DIR = Path("evaluation/results/RAGAS")
JUDGE_MODEL = "gpt-4o-mini"
METRIC_NAMES = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
METRIC_LABELS = ["Faithfulness", "Answer\nRelevancy", "Context\nPrecision", "Context\nRecall"]
# Wong colour-blind-safe palette (canonical thesis style — see EVALUATION_PLAN.md).
COLORS = ["#0072B2", "#009E73", "#E69F00", "#CC79A7"]


def _build_ragas_llm_emb():
    from dc_available.config.llm_config import get_langchain_llm
    from dc_available.RAG.retrieval_helpers import get_embeddings
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    llm = LangchainLLMWrapper(get_langchain_llm(model_name=JUDGE_MODEL))
    emb = LangchainEmbeddingsWrapper(get_embeddings())
    return llm, emb


def _load_rows(topic: str, ragas_dir: Path, in_file: Path | None = None) -> list[dict]:
    path = in_file or (ragas_dir / topic / f"{topic}_with_answers.json")
    if not path.exists():
        sys.exit(f"Missing {path}  —  run generate.py first.")
    return json.loads(path.read_text(encoding="utf-8"))


def _build_hf_dataset(rows: list[dict]):
    from datasets import Dataset as HFDataset
    items, idx = [], []
    for i, r in enumerate(rows):
        ans = (r.get("generated_answer") or "").strip()
        ctxs = [c for c in (r.get("retrieved_contexts") or []) if c]
        if not ans or not ctxs:
            continue
        gt = (r.get("golden_answer") or "").strip()
        items.append({
            "question": r["question"],
            "answer": ans,
            "contexts": ctxs,
            "ground_truths": [gt] if gt else [""],
        })
        idx.append(i)
    return HFDataset.from_list(items), idx


def _plot(summary: dict[str, float], out: Path, topic: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # Canonical thesis style (see EVALUATION_PLAN.md / plot_results.py): sans-serif,
    # no top/right spines, light y-grid, PNG @300dpi.
    plt.rcParams.update({
        "figure.dpi": 100, "savefig.dpi": 300, "savefig.bbox": "tight",
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 12,
        "xtick.labelsize": 11, "ytick.labelsize": 11,
        "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
        "figure.facecolor": "white", "axes.facecolor": "white",
    })
    vals = [summary.get(m, 0.0) for m in METRIC_NAMES]
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    bars = ax.bar(METRIC_LABELS, vals, color=COLORS, width=0.6)
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score (0–1)")
    ax.set_title(f"RAGAS generation quality — {topic.capitalize()} (hybrid leg)")
    ax.grid(axis="y", color="#999999", alpha=0.3, linewidth=0.6)
    fig.savefig(out)
    plt.close(fig)


def evaluate_topic(
    topic: str,
    ragas_llm,
    ragas_emb,
    ragas_dir: Path = RAGAS_DIR,
    in_file: Path | None = None,
    out_dir: Path | None = None,
) -> dict[str, float]:
    import pandas as pd
    from ragas import evaluate as ragas_evaluate
    from ragas.metrics import (
        faithfulness, answer_relevancy,
        context_precision, context_recall,
    )

    print(f"  loading rows …")
    rows = _load_rows(topic, ragas_dir, in_file=in_file)
    dataset, indices = _build_hf_dataset(rows)
    n_scored = len(dataset)
    print(f"  {n_scored} rows to score (of {len(rows)} hybrid rows)")
    if n_scored == 0:
        print("  nothing to score — skipping")
        return {}

    print(f"  running ragas.evaluate() …")
    result = ragas_evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=ragas_llm,
        embeddings=ragas_emb,
    )
    df = result.to_pandas()

    out_dir = out_dir or (ragas_dir / topic)
    out_dir.mkdir(parents=True, exist_ok=True)

    qids = [rows[i]["question_id"] for i in indices[: len(df)]]
    df.insert(0, "question_id", qids)
    df.to_csv(out_dir / "scores.csv", index=False)
    df.to_json(out_dir / "scores.json", orient="records", indent=2, force_ascii=False)

    summary = {
        m: round(float(df[m].mean(skipna=True)), 4)
        for m in METRIC_NAMES
        if m in df.columns
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _plot(summary, out_dir / "plot_ragas.png", topic)

    print(f"  scores → {out_dir}")
    return summary


def main(
    topics: list[str] | None = None,
    ragas_dir: Path = RAGAS_DIR,
) -> dict[str, dict[str, float]]:
    topics = topics or TOPICS
    ragas_llm, ragas_emb = _build_ragas_llm_emb()
    summaries: dict[str, dict[str, float]] = {}
    for topic in topics:
        print(f"\n=== [ragas_eval] {topic} ===")
        summaries[topic] = evaluate_topic(topic, ragas_llm, ragas_emb, ragas_dir=ragas_dir)
        for m, v in summaries[topic].items():
            print(f"    {m}: {v:.4f}")
    return summaries


if __name__ == "__main__":
    # Drop the eval dirs (so local evaluation/datasets can't shadow HF "datasets"),
    # but KEEP the project root (dc-available) on sys.path so dc_available imports.
    _here = Path(__file__).resolve().parent
    _root = _here.parents[2]                                   # .../dc-available
    _bad = {str(_here), str(_here.parent), str(_here.parents[1])}  # generation, available, evaluation
    sys.path[:] = [p for p in sys.path if str(Path(p or ".").resolve()) not in _bad]
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

    # CLI: ragas_eval <topic> [in_file] [out_dir]
    #   no args -> default two-topic run under results/RAGAS
    args = sys.argv[1:]
    if args:
        topic = args[0]
        in_file = Path(args[1]) if len(args) > 1 else None
        out_dir = Path(args[2]) if len(args) > 2 else None
        llm, emb = _build_ragas_llm_emb()
        print(f"=== [ragas_eval] {topic} ===")
        summary = evaluate_topic(topic, llm, emb, in_file=in_file, out_dir=out_dir)
        for m, v in summary.items():
            print(f"    {m}: {v:.4f}")
    else:
        main()
