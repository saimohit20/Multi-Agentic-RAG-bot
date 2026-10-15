# Pre-process — Technical Deep Dive

*Companion to [`pre-process.md`](pre-process.md). That doc is the high-level overview; this one is the
code-level walkthrough: which functions run, in what order, and exactly how each `.jsonl` row is built.*

*Pipeline STEP: `pipeline/preprocess.py` → `dc_available/summarization/**` → `dc_available/common/VideoIE/**`*

---

## 0. The big picture

A raw training file (`.mp4`, `.pdf`, `.pptx`, …) is turned into two artifacts:

- **`<file>__summary_docs.jsonl`** — every extracted piece as a `Document`, **one per line**.
- **`<file>__summary.txt`** — only the overall summary, rendered as human-readable markdown.

A `Document` is always two parts:

```json
{"page_content": "the text that gets EMBEDDED", "metadata": { "...stored, NOT embedded..." }}
```

> **page_content vs metadata.text** — `page_content` is what later gets embedded and searched.
> `metadata.text` is a stored raw copy (e.g. the transcript) kept for reference; it is **never embedded**.

Entry point: `summarize_single_file()` in `dc_available/summarization/trainings.py` branches on the file
suffix, then every branch ends the same way:

```python
# trainings.py (end of summarize_single_file)
output_documents[0]                      # the overall meta-summary → __summary.txt
save_docs_to_jsonl(output_documents, …)  # every Document → one .jsonl line
# each doc also gets: source, part_of_document, part_of_training, trust_score
```

---

## 1. MP4 (video) — the most complex path

Handled by `dc_available/summarization/videos.py` + `dc_available/common/VideoIE/`.

### Step-by-step

1. **Extract audio** — `video_to_audio()` strips the audio track → `<name>__mp4.mp3`.
   *(trainings.py:218)*

2. **Transcribe with Whisper** — the `.mp3` goes to `generate_speech_transcription()`.
   - If the mp3 is > 10 MB it is split into ~5-minute **size-based chunks** (`chunk_000.mp3`, …), each
     transcribed separately (with ~20 s rate-limit waits), then stitched back together.
   - Whisper returns **plain text** (the spoken words). Saved to `<name>__mp3_transcript.txt`.
   *(Speech2Text.py, VideoIE.py:14)*

3. **Extract distinct slides** — `extract_distince_slides_from_video()` (OpenCV):
   - Grab **one frame every 60 s** (`slide_sampling_rate`), save as `slide_0000.jpg`, …
   - **De-duplicate**: each new frame is compared to the previously kept slide by a **vision LLM**
     (`identical_slide_detection_with_llm`, GPT-4o). Prompt forces the answer to start with
     `Yes.` / `No.` / `Mostly similar…`; `No` = keep as new slide, otherwise **delete** the frame and
     append its timestamp to the existing slide.
   - LLM dedup calls ≈ (sampled frames − 1), sequential (each vs previous), not all-vs-all.
   *(ExtractScreenShotsFromVideo.py:81, SlideSimilarity.py:85)*

4. **Align transcript to slides** — `get_audio_transcript_per_slide()` cuts the transcript into
   per-slide sections using each slide's time range. *(VideoIE.py:34)*

5. **Understand + consolidate each slide** — for every distinct slide:
   - `generate_slide_understanding_result()` — vision LLM describes the slide image.
   - `generate_consolidation_response()` — GPT-4o merges *slide description* + *speech* into one
     coherent explanation.
   - Written to `__mp4_understanding/slide_XXXX.txt` with three parts (image description / speech /
     consolidation). *(VideoIE.py:51-79)*

6. **Slide `.txt` → Documents** — `_load_video_parts()` regex-parses each `.txt`; then per slide
   `_split_and_summarize_video_parts()` emits **3 Documents**:

   | Document | `page_content` = | `document_type` |
   |---|---|---|
   | video | `consolidated_understanding` | `video` |
   | image | `slide_image_understanding`  | `image` |
   | audio | `sentences` (speech)         | `audio` |

   Shared per-slide metadata: `page_id`, `start_time`, `end_time`, `image_path`, `audio_path`,
   `video_path`, `id`. *(videos.py:100-145)*

7. **Add summary Documents** — `generate_meta_summary()` (LLM) makes one summary per modality
   (`video_understanding`, `video_image`, `video_audio`) plus one overall `video_summary`
   (= `output_documents[0]` → `__summary.txt`). *(videos.py:186-266)*

### Picture

```
Kapitel1.mp4
  │ video_to_audio()
  ▼
__mp4.mp3 ──► Whisper ──► __mp3_transcript.txt
  │                              │ (align by time)
  │ every 60s frame + LLM dedup  ▼
  ▼                         per-slide transcript
slide_XXXX.jpg ──► vision LLM ─┐
                               ▼
              __mp4_understanding/slide_XXXX.txt  (image desc + speech + consolidation)
                               │ _load_video_parts (regex)
                               ▼
                  per slide → 3 Documents (video / image / audio)
                               + 4 summary Documents (LLM)
                               ▼
        save_docs_to_jsonl ──► __summary_docs.jsonl   (one Document = one line)
        doc_to_markdown(meta) ─► __summary.txt
```

**Rows for N distinct slides ≈ 3·N + 4** (3 per slide + overall + 3 modality metas).

---

## 2. PDF

Handled by `dc_available/summarization/texts.py` (`generate_pdf_summaries_for_retrieval`).

1. **Read each page** — `load_pdf_pages_for_available()` → one **text Document per page**
   (`page_content` = page text) + a rendered page image in `__pdf_images/`.
2. **Summarize each page (LLM, precise mode)** — `generate_sequential_document_summary_for_retrieval()`
   → a **summary Document per page** + an overall meta-summary.
3. **Combine** — `output_documents = pdf_texts + pdf_text_summaries`; `pdf_text_summaries[0]` is the
   meta-summary → `__summary.txt`. *(trainings.py:160-179)*

```
page 1..N ─► load_pdf_pages ─► raw text Document (+ __pdf_images/*.jpg)
                 │ generate_sequential_document_summary (LLM)
                 ▼
        per-page summary Documents + overall meta-summary
                 ▼
     save_docs_to_jsonl → __summary_docs.jsonl ;  doc_to_markdown(meta) → __summary.txt
```

**Rows for an N-page PDF ≈ 2·N + 1** (raw text row + summary row per page, + 1 meta).
No audio/slide-dedup — the **page** is the unit.

---

## 3. PPTX / Office (`.pptx`, `.pptm`, `.docx`, `.xlsx`)

Handled by `dc_available/summarization/slides.py` (`generate_powerpoint_summary_for_retrieval`).

1. **Convert to PDF first** — `ppt_to_pdf()` → `<name>__pptx.pdf` (everything normalized to PDF pages;
   Excel/Word capped at 10 pages to save cost). *(slides.py:50, trainings.py:185)*
2. **Read each page** — `load_pdf_pages_for_available()` → per slide: text Document + image in
   `__pptx_images/`.
3. **Understand each slide with VISION** — `generate_sequential_image_summary_for_retrieval()` sends
   **slide image + text to a vision LLM (GPT-4o)** → an image-understanding Document per slide.
   *(This is the key difference from plain PDF.)* *(slides.py:72)*
4. **Combine** — `output_documents = ppt_texts + ppt_slide_summaries`; `[0]` → `__summary.txt`.

```
file.pptx ─► ppt_to_pdf ─► __pptx.pdf ─► load pages ─► raw text Document + slide image
                                              │ per slide image + text → VISION LLM
                                              ▼
                                   image-understanding Document
                                              ▼
     save_docs_to_jsonl → __summary_docs.jsonl ;  doc_to_markdown(meta) → __summary.txt
```

**Rows for an N-slide deck ≈ 2·N + 1.**

| | PDF | PPTX/Office |
|---|---|---|
| Convert step | none | **→ PDF first** |
| Per-page understanding | text summary (LLM reads text) | **vision** (LLM looks at the slide image) |

---

## 4. MSG (email) & HTML — the simplest text paths

Handled by `texts.py` (`generate_msg_summary_for_retrieval`, `generate_html_summary_for_retrieval`).

- **MSG** — `load_msg_file()` reads the email body into one **text Document**; attachments are queued
  and processed as their own files. Then `generate_text_summary()` (LLM) makes a summary Document.
  → 2 Documents (text + summary) per email. *(texts.py:251)*
- **HTML** — the page is converted to Markdown via `markdownify()` (so tables/headings survive), wrapped
  as one text Document, then summarized. → 2 Documents. *(texts.py:275)*

```
.msg / .html ─► extract text (body / markdown) ─► text Document
                     │ generate_text_summary (LLM)
                     ▼
               summary Document (is_meta_summary=True) → __summary.txt
                     ▼
        save_docs_to_jsonl → __summary_docs.jsonl
```

---

## 5. Common tail (all formats)

Once a format branch has produced `output_documents`:

1. **Stamp shared metadata** — every Document gets `source`, `part_of_document`,
   `part_of_training`, and a computed **`trust_score`**. *(trainings.py:298-328)*
2. **Write `__summary.txt`** — `doc_to_markdown(output_documents[0])` (the overall meta-summary).
3. **Write `__summary_docs.jsonl`** — `save_docs_to_jsonl(output_documents, …)`, one Document per line.
4. **Keep raw data clean** — raw media is processed in a scratch dir; only derived artifacts
   (images, converted PDFs, slide screenshots) are promoted to `Extracted_resources`; the raw file
   (and, for video, the `.mp3`) is discarded. *(trainings.py:_promote_derived_to_extracted)*

Later, per training, all `__summary_docs.jsonl` are aggregated into one **`index.jsonl`** — the single
input the **index** step reads to build the vector store and graph.

---

## Quick reference — key functions

| Concern | Function | File |
|---|---|---|
| Branch by format | `summarize_single_file` | `summarization/trainings.py` |
| Video → parts | `generate_video_summaries_for_retrieval`, `_split_and_summarize_video_parts` | `summarization/videos.py` |
| Whisper | `generate_speech_transcription`, `process_large_audio_file` | `common/VideoIE/Speech2Text.py` |
| Slide extract + dedup | `extract_distince_slides_from_video`, `identical_slide_detection_with_llm` | `common/VideoIE/ExtractScreenShotsFromVideo.py`, `SlideSimilarity.py` |
| Slide understand/merge | `generate_slide_understanding_result`, `generate_consolidation_response` | `common/VideoIE/ImageUnderstanding.py`, `MergeSlideSpeechInfo.py` |
| PDF | `generate_pdf_summaries_for_retrieval` | `summarization/texts.py` |
| PPTX/Office | `generate_powerpoint_summary_for_retrieval` | `summarization/slides.py` |
| MSG / HTML | `generate_msg_summary_for_retrieval`, `generate_html_summary_for_retrieval` | `summarization/texts.py` |
| Meta-summary (LLM) | `generate_meta_summary` | `summarization/texts.py` |
| Write jsonl | `save_docs_to_jsonl` | `common/data_loading.py` |
