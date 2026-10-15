# Dataset Generation

This module generates evaluation datasets from ingestion artifacts using LLM-based question generation.

## Quick Start

### Basic usage (all files, default output)
```bash
python -m evaluation.available.common.dataset_generation.generate_dataset
```

### Test on subset (2 files, no LLM calls)
```bash
python -m evaluation.available.common.dataset_generation.generate_dataset --max-files 2 --dry-run
```

### Custom paths
```bash
python -m evaluation.available.common.dataset_generation.generate_dataset \
  --input-root /path/to/ingestion \
  --output /path/to/output.xlsx \
  --max-files 10
```

## Output Schema

Excel file with columns:
- **question**: The generated question
- **ground_truth**: The correct answer (1-3 sentences)
- **keywords**: Comma-separated list of 2-3 keywords
- **answer_type**: Type of question (factual, conceptual, procedural, application, summary)
- **subtopic**: Specific concept being tested
- **topic**: Training topic (e.g., DigitalTransformation)
- **level**: Training level (e.g., Basic, Expert)
- **source_document**: Source file name

## Options

- `--input-root`: Path to ingestion artifacts (default: `dc_available/data/Artifacts/resources/ingestion`)
- `--output`: Output Excel file path (default: `evaluation/datasets/generated_dataset.xlsx`)
- `--max-files`: Limit number of source files to process (for testing)
- `--dry-run`: Preview without calling LLM (shows file discovery only)

## Generation Process

For each source file:
1. Discover all `__summary_docs.jsonl` files
2. Parse JSONL records
3. Extract metadata from path (topic, level, training, source_document)
4. Combine content from non-summary records (up to 4000 tokens)
5. Call LLM to generate exactly 5 questions of different types
6. Validate each question (required fields, answer_type, keywords)
7. Add metadata and export to Excel

## Question Types

The LLM generates a balanced mix:
- **Factual**: Direct recall of facts and definitions
- **Conceptual**: Understanding of concepts and relationships
- **Procedural**: How-to questions about processes
- **Application**: Applying knowledge to new scenarios
- **Summary**: Synthesis of multiple ideas

## Logging

All operations are logged with timestamps. Check console output for:
- Files discovered
- Files processed
- Generated questions per file
- Validation errors
- Final summary (total rows, output location)

## Integration with Evaluation

The generated Excel can be used directly with the evaluation pipeline:
```bash
poetry run python -m evaluation.available.run_pipeline --qa-dataset generated_dataset.xlsx
```

## Notes

- Large files are truncated at 4000 tokens to avoid LLM timeouts
- Meta-summary records (page_id == -1) are skipped during content extraction
- Invalid or unparseable records are logged and skipped
- Use `--dry-run --max-files 2` first to verify configuration
