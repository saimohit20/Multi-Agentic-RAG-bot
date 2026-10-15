"""
Main CLI for generating evaluation dataset from ingestion artifacts.

Usage:
    python generate_dataset.py --output dataset.xlsx
    python generate_dataset.py --input-root /path/to/ingestion --output dataset.xlsx --max-files 5
    python generate_dataset.py --max-files 2 --dry-run
"""

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
from langchain.prompts import PromptTemplate
from dc_available.config.llm_config import get_langchain_llm

from evaluation.available.common.dataset_generation.prompts import QA_GENERATION_PROMPT
from evaluation.available.common.dataset_generation.utils import (
    discover_jsonl_files,
    parse_jsonl_file,
    extract_metadata,
    extract_page_content,
    validate_row,
)

# Suppress noisy third-party loggers; only show our own errors
logging.basicConfig(level=logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("azure").setLevel(logging.WARNING)
logging.getLogger("dc_available").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# Default paths
DEFAULT_INPUT_ROOT = Path(__file__).parent.parent.parent / "dc_available" / "data" / "Artifacts" / "resources" / "ingestion"
DEFAULT_OUTPUT = Path(__file__).parent / "generated_dataset.xlsx"


def generate_questions_for_file(file_path: Path, metadata: dict, records: list) -> list[dict]:
    """
    Generate 5 QA pairs for a single source file using LLM.
    
    Args:
        file_path: Path to the source JSONL file
        metadata: Extracted metadata (topic, level, training, source_document)
        records: List of parsed JSONL records
    
    Returns:
        List of generated QA dicts
    """
    # Combine content from records, prioritizing richer records
    content_parts = []
    for record in records:
        # Skip meta summaries and empty records
        if record.get("metadata", {}).get("page_id") == -1:
            continue
        
        text = extract_page_content(record)
        if text and len(text) > 50:
            content_parts.append(text)
    
    if not content_parts:
        return []

    combined_content = " ".join(content_parts)[:4000]

    try:
        llm = get_langchain_llm()
        prompt = PromptTemplate(input_variables=["content"], template=QA_GENERATION_PROMPT)
        chain = prompt | llm
        response = chain.invoke({"content": combined_content})
        response_text = response.content if hasattr(response, "content") else str(response)

        start = response_text.find("[")
        end = response_text.rfind("]") + 1
        if start < 0 or end <= start:
            return []
        qa_pairs = json.loads(response_text[start:end])

        results = []
        for qa in qa_pairs:
            if not isinstance(qa, dict):
                continue
            qa["topic"] = metadata["topic"]
            qa["level"] = metadata["level"]
            qa["source_document"] = metadata["source_document"]
            is_valid, _ = validate_row(qa)
            if is_valid:
                results.append(qa)

        return results

    except Exception as e:
        logger.error(f"LLM error for {metadata['source_document']}: {e}")
        return []


def main():
    parser = argparse.ArgumentParser(
        description="Generate evaluation dataset from ingestion artifacts"
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help=f"Root path to ingestion artifacts (default: {DEFAULT_INPUT_ROOT})"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output Excel file path (default: {DEFAULT_OUTPUT})"
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Limit number of files to process (for testing)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run generation without writing the Excel file"
    )
    
    args = parser.parse_args()
    
    print(f"Input  : {args.input_root}")
    print(f"Output : {args.output}")
    if args.dry_run:
        print("Mode   : dry-run (LLM runs, Excel not written)")
    print()

    files = discover_jsonl_files(args.input_root, max_files=args.max_files)
    if not files:
        print("ERROR  : No JSONL files found")
        return

    print(f"Found {len(files)} JSONL files")
    print()

    all_rows = []
    processed, errors = 0, 0

    for file_path in files:
        records = parse_jsonl_file(file_path)
        if not records:
            print(f"  Error : {file_path.name}  (no records)")
            errors += 1
            continue

        metadata = extract_metadata(file_path)
        qa_pairs = generate_questions_for_file(file_path, metadata, records)

        if qa_pairs:
            all_rows.extend(qa_pairs)
            processed += 1
            print(f"  OK    : {file_path.name}  ({len(qa_pairs)} questions)")
        else:
            print(f"  Error : {file_path.name}  (generation failed)")
            errors += 1

    total_rows = len(all_rows)

    if all_rows:
        df = pd.DataFrame(all_rows)
        columns = ["question", "ground_truth", "keywords", "answer_type", "subtopic", "topic", "level", "source_document"]
        df = df[[col for col in columns if col in df.columns]]
        output_path = args.output if not args.dry_run else args.output.with_stem(args.output.stem + "_dryrun")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_excel(output_path, index=False, engine="openpyxl")
        print(f"\nSaved  : {output_path}")

    print(f"""
{'='*50}
Files scanned  : {len(files)}
Files processed: {processed}
Total rows     : {total_rows}
Errors         : {errors}
{'='*50}""")


if __name__ == "__main__":
    main()
