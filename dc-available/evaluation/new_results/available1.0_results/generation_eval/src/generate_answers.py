"""
Generate answers for both topics using Azure gpt-4o (headless, no Streamlit).

Reads:  evaluation/retrieval_eval/results/{grownfield,hydraulics}_results.json
Writes: evaluation/generation_eval/results/{grownfield,hydraulics}_gen_results.json
        (same records with generated_answer filled in)

Run:
    .venv/Scripts/python.exe evaluation/generation_eval/src/generate_answers.py
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import gen_paths_config as cfg

from dc_available.config.llm_config import get_langchain_llm

_SYSTEM = (
    "You are a knowledgeable assistant. "
    "Answer the question using ONLY the provided context. "
    "Be concise and factual. Respond in the same language as the question."
)

_USER_TEMPLATE = """\
Context:
{context}

Question: {question}
"""


def _build_context(contexts: List[str]) -> str:
    return "\n\n---\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(contexts))


def generate_answer(llm, question: str, contexts: List[str]) -> str:
    from langchain_core.messages import HumanMessage, SystemMessage

    messages = [
        SystemMessage(content=_SYSTEM),
        HumanMessage(content=_USER_TEMPLATE.format(
            context=_build_context(contexts),
            question=question,
        )),
    ]
    response = llm.invoke(messages)
    return response.content.strip()


def run_topic(
    name: str,
    input_path: Path,
    output_path: Path,
    llm,
) -> List[Dict[str, Any]]:
    if not input_path.exists():
        print(f"[{name}] Input not found: {input_path}  (run retrieval eval first)")
        return []

    with open(input_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    print(f"[{name}] Generating answers for {len(records)} questions …")
    for i, rec in enumerate(records, start=1):
        rec["generated_answer"] = generate_answer(llm, rec["question"], rec["retrieved_contexts"])
        print(f"[{name}] {i:>3}/{len(records)}  {rec['question_id']}  done")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"[{name}] Saved -> {output_path}")
    return records


# topic key -> (display name, retrieval input JSON, generation output JSON)
TOPICS = {
    "grownfield": ("Grownfield", cfg.GROWNFIELD_RETRIEVAL, cfg.GROWNFIELD_GEN_RESULTS),
    "hydraulics": ("Hydraulics", cfg.HYDRAULICS_RETRIEVAL, cfg.HYDRAULICS_GEN_RESULTS),
    "synthetic":  ("Synthetic",  cfg.SYNTHETIC_RETRIEVAL,  cfg.SYNTHETIC_GEN_RESULTS),
}


def main(topics: List[str] | None = None) -> None:
    cfg.ensure_dirs()
    keys = topics or list(TOPICS)
    llm = get_langchain_llm()

    for n, key in enumerate(keys, start=1):
        name, input_path, output_path = TOPICS[key]
        print(f"\n========== {n}/{len(keys)}  {name.upper()} ==========")
        run_topic(name, input_path, output_path, llm)

    print("\n[Done] Answer generation complete.")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", choices=[*TOPICS, "all"], default="all")
    args = ap.parse_args()
    main(None if args.topic == "all" else [args.topic])
