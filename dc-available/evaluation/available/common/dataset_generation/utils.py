"""Utility functions for dataset generation."""

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _flatten_text(value: object) -> str:
    """Convert nested JSON values into a single plain-text string."""
    if value is None:
        return ""

    if isinstance(value, str):
        return value.strip()

    if isinstance(value, (int, float, bool)):
        return str(value)

    if isinstance(value, list):
        parts = [_flatten_text(item) for item in value]
        return " ".join(part for part in parts if part)

    if isinstance(value, dict):
        preferred_keys = [
            "title",
            "description",
            "main idea",
            "main_topic",
            "overview",
            "text",
            "content",
            "quote",
        ]
        parts = []

        for key in preferred_keys:
            if key in value:
                text = _flatten_text(value[key])
                if text:
                    parts.append(text)

        for key, item in value.items():
            if key in preferred_keys:
                continue
            text = _flatten_text(item)
            if text:
                parts.append(text)

        return " ".join(parts)

    return str(value)


def discover_jsonl_files(root_path: Path, max_files: Optional[int] = None) -> list[Path]:
    """
    Recursively find all __summary_docs.jsonl files under root_path.
    
    Args:
        root_path: Root directory to search
        max_files: Limit number of files (for testing)
    
    Returns:
        List of Path objects to JSONL files
    """
    if not root_path.exists():
        logger.error(f"Root path does not exist: {root_path}")
        return []
    
    files = list(root_path.rglob("*__summary_docs.jsonl"))
    
    if max_files:
        files = files[:max_files]
    
    logger.info(f"Found {len(files)} JSONL files")
    return files


def parse_jsonl_file(file_path: Path) -> list[dict]:
    """
    Parse a JSONL file and return list of records.
    Skip empty lines and log parsing errors.
    
    Args:
        file_path: Path to JSONL file
    
    Returns:
        List of parsed JSON objects
    """
    records = []
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    record = json.loads(line)
                    records.append(record)
                except json.JSONDecodeError as e:
                    logger.warning(f"{file_path}:{line_num} - JSON parse error: {e}")
                    continue
    
    except Exception as e:
        logger.error(f"Failed to read {file_path}: {e}")
        return []
    
    return records


def extract_metadata(file_path: Path) -> dict:
    """
    Extract topic, level, training, and source_document from file path.
    
    Path structure: root/Topic/Level/Training/SourceFile__summary_docs.jsonl
    Example: ingestion/DigitalTransformation/Basic/Training1/Kapitel2.mp4__summary_docs.jsonl
    
    Args:
        file_path: Full path to JSONL file
    
    Returns:
        Dict with keys: topic, level, training, source_document
    """
    parts = file_path.parts
    
    # Find the position to extract from (look for a common root pattern)
    metadata = {
        "topic": "unknown",
        "level": "unknown",
        "training": "unknown",
        "source_document": file_path.stem.replace("__summary_docs", ""),
    }
    
    # Try to extract from path structure (assuming /Topic/Level/Training/ pattern)
    if len(parts) >= 3:
        # Work backwards from the file
        try:
            # source_document = filename without __summary_docs.jsonl
            source = file_path.name.replace("__summary_docs.jsonl", "")
            metadata["source_document"] = source
            
            # training = parent folder (e.g., Training1, training)
            training = parts[-2]
            metadata["training"] = training
            
            # level = 2 levels up (e.g., Basic, Expert)
            level = parts[-3]
            metadata["level"] = level
            
            # topic = 3 levels up
            topic = parts[-4]
            metadata["topic"] = topic
        except Exception as e:
            logger.warning(f"Could not extract full path metadata from {file_path}: {e}")
    
    return metadata


def extract_page_content(record: dict) -> str:
    """
    Extract and normalize page_content from a JSONL record.
    Unwraps JSON code blocks if needed.
    
    Args:
        record: Parsed JSON record from JSONL
    
    Returns:
        Clean text content
    """
    page_content = record.get("page_content", "")
    metadata_text = _flatten_text(record.get("metadata", {}).get("text", ""))
    
    # Unwrap markdown code blocks if present
    if "```json" in page_content:
        try:
            start = page_content.find("```json") + 7
            end = page_content.find("```", start)
            if end > start:
                inner = page_content[start:end].strip()
                data = json.loads(inner)
                if isinstance(data, dict):
                    text = _flatten_text(data)
                    if metadata_text:
                        return f"{text}\n\n{metadata_text}".strip()
                    return text
        except Exception as e:
            logger.debug(f"Could not unwrap code block: {e}")
    
    text = _flatten_text(page_content)
    if metadata_text and metadata_text not in text:
        return f"{text}\n\n{metadata_text}".strip()
    return text


def validate_row(row: dict) -> tuple[bool, Optional[str]]:
    """
    Validate a generated QA row.
    
    Args:
        row: Dict with keys: question, ground_truth, keywords, answer_type, subtopic, etc.
    
    Returns:
        (is_valid, error_message)
    """
    required_fields = ["question", "ground_truth", "keywords", "answer_type", "subtopic"]
    
    for field in required_fields:
        if field not in row:
            return False, f"Missing field: {field}"
        
        if not row[field] or (isinstance(row[field], str) and not row[field].strip()):
            return False, f"Empty field: {field}"
    
    if not isinstance(row.get("keywords"), list) or len(row["keywords"]) == 0:
        return False, "Keywords must be a non-empty list"
    
    valid_types = ["factual", "conceptual", "procedural", "application", "summary"]  # noqa: keep in sync with prompts.py
    if row.get("answer_type") not in valid_types:
        return False, f"Invalid answer_type: {row.get('answer_type')}"
    
    return True, None


def log_summary(summary: dict) -> None:
    """
    Log a summary of the generation run.
    
    Args:
        summary: Dict with keys: total_files, processed_files, total_rows, errors
    """
    print("\n" + "=" * 60)
    print("DATASET GENERATION SUMMARY")
    print("=" * 60)
    print(f"Files scanned:     {summary.get('total_files', 0)}")
    print(f"Files processed:   {summary.get('processed_files', 0)}")
    print(f"Total rows:        {summary.get('total_rows', 0)}")
    print(f"Errors:            {summary.get('errors', 0)}")
    if summary.get('output_file'):
        print(f"Output file:       {summary['output_file']}")
    print("=" * 60 + "\n")
