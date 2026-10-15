# Observability with Opik

## 1. Overview

Opik is an open-source observability and tracing platform for applications built
on Large Language Models (LLMs), developed by Comet. In this project it is used
to record, inspect, and evaluate the behaviour of the Retrieval-Augmented
Generation (RAG) pipeline and the agent workflows.

For every request that the system handles, Opik captures:

- the input and output of each processing step,
- the execution time of each step,
- token usage and estimated cost of LLM calls,
- automatically computed quality scores (e.g. hallucination, answer relevance,
  context precision).

This information is collected while the application runs and is displayed in a
web-based dashboard, allowing the pipeline to be analysed after execution
without adding manual logging throughout the codebase.

The repository contains two distinct components: a local deployment of the Opik
platform itself (backend, frontend, and server), and the application code that
uses Opik as a client library. The application sends all recorded data to the
locally running Opik server.

## 2. Core Concepts

Opik organises recorded data in a three-level hierarchy.

| Concept | Definition | Scope in this project |
|---|---|---|
| **Span** | A record of a single step (one function call), including its inputs, outputs, and duration. | e.g. `vector_retrieval`, `reranker`, `graph_retrieval` |
| **Trace** | A container for one complete request, composed of nested spans arranged in a tree. | One user question (`rag_pipeline`) |
| **Thread** | A group of related traces that belong to the same conversation. | One user session (all questions asked) |

The resulting relationship is:

```
Session  ->  Thread          (one conversation)
   Question  ->  Trace        (one request: question to answer)
      Step   ->  Span         (one function call inside the request)
```

**Feedback scores** are numeric quality metrics (0.0-1.0) attached to a trace.
They are produced by a separate evaluation step rather than by the pipeline
itself, and are used to assess answer quality directly within the dashboard.

## 3. Integration Mechanisms

The application interacts with Opik through three mechanisms.

### 3.1 The `@opik.track` decorator

A decorator is a Python construct that adds behaviour to a function without
modifying its internal code. The `@opik.track` decorator, placed above a
function, instructs Opik to automatically record that function as a span each
time it is called: it captures the inputs before execution, records the return
value and duration after execution, and transmits the result to the Opik server.

Crucially, when one decorated function calls another, Opik nests the inner span
inside the outer one based on the call order. This automatically reconstructs
the pipeline's structure as a tree, without any manual bookkeeping.

```python
@opik.track(name="rag_pipeline")
def ask_question(...):
    ...
    search_question(...)      # becomes a nested span
```

### 3.2 Enriching spans and traces

The decorator captures inputs and outputs automatically, but additional
information is often required. Two functions attach extra data to the currently
active span or trace:

- `opik.update_current_span(...)` - adds information to the current step
  (e.g. token usage, cost, and step-specific metadata).
- `opik.update_current_trace(...)` - adds information to the overall request
  (e.g. the final answer, the thread identifier, and feedback scores).

The word *current* refers to the span or trace that is active at that point in
execution; Opik tracks this internally, so it does not need to be referenced
explicitly.

### 3.3 Feedback scores via an LLM judge

After each answer is generated, a secondary, lower-cost LLM evaluates it against
the question and the retrieved context. It returns three scores - hallucination,
answer relevance, and context precision - which are attached to the trace as
feedback scores. This provides an automated, per-request measure of answer
quality.

## 4. Configuration

Opik is configured through environment variables set at the top of the relevant
modules:

```python
os.environ.setdefault("OPIK_URL_OVERRIDE", "http://localhost:5173/api")
os.environ.setdefault("OPIK_PROJECT_NAME", "dc-available-rag")
os.environ.setdefault("OPIK_WORKSPACE", "default")
```

- `OPIK_URL_OVERRIDE` directs the client to the locally running Opik server.
- `OPIK_PROJECT_NAME` groups all traces under a named project
  (`dc-available-rag`).
- `OPIK_WORKSPACE` specifies the workspace on the server.

## 5. End-to-End Flow (QA Pipeline)

The following steps describe the lifecycle of a single question in the QA
pipeline.

1. **Session start.** When a user opens the interface, a unique thread
   identifier is generated once using `uuid.uuid4()` and stored in the session
   state. All questions asked during this session reuse this identifier,
   grouping them into one conversation. The thread persists until the session
   ends.

2. **Question submission.** The user enters a question, which triggers a call to
   `ask_question(...)`, passing the thread identifier.

3. **Trace creation.** Because `ask_question` is decorated with
   `@opik.track(name="rag_pipeline")`, Opik opens a new trace, records the
   question as input, and begins timing.

4. **Thread association.** The trace is stamped with the session's thread
   identifier via `update_current_trace(thread_id=...)`, filing it under the
   correct conversation.

5. **Retrieval.** The pipeline performs vector retrieval, optional graph-based
   expansion, and reranking. Each of these is a decorated function and therefore
   appears as a nested span (`vector_retrieval`, `graph_retrieval`, `reranker`),
   with step-specific metadata attached where relevant.

6. **Answer generation.** The retrieved context and the question are passed to
   the LLM chain. Token usage and estimated cost are measured and attached to
   the span with `update_current_span(...)`.

7. **Result recording.** The final answer is attached to the trace via
   `update_current_trace(output=...)`.

8. **Quality evaluation.** The LLM judge scores the answer, and the three
   feedback scores are attached to the trace.

9. **Trace completion.** When `ask_question` returns, the decorator finalises
   the trace and sends it, with all nested spans, cost, output, and scores, to
   the Opik server. The answer is displayed to the user.

10. **Subsequent questions.** Each further question creates a new trace stamped
    with the same thread identifier, so it appears alongside the previous traces
    within the same conversation.

The resulting structure in the dashboard is:

```
Thread (session)
 |__ Trace: rag_pipeline        [input, output, feedback scores, cost]
      |__ search_question
           |__ vector_retrieval
           |__ graph_retrieval
           |__ reranker
```

## 6. Agent Workflow

The agent workflow (curriculum assessment) applies the same core concepts as the
QA pipeline, but its execution model requires additional handling. It is built
on the AutoGen framework and consists of three sequential assessment stages:

1. topic assessment,
2. knowledge assessment,
3. teaching-material assessment.

Each stage is an independent multi-turn conversation between agents that
concludes when the agent emits a `SUMMARY`. The output of each stage feeds into
the next.

### 6.1 One trace per stage

Each stage is recorded as a separate trace, and all three traces are grouped
under the same thread as any QA activity from the same session. To achieve this,
the agent workflow reuses the identical `opik_thread_id` from the session state
that the QA pipeline uses. The resulting structure is:

```
Thread (session)
 |__ Trace: topic_assessor                [input -> topic_summary]
 |__ Trace: knowledge_assessor            [topic_summary -> knowledge_summary]
 |__ Trace: teaching_material_assessor    [summaries -> teaching_material_summary]
```

### 6.2 The Streamlit re-run challenge

The framework used for the user interface (Streamlit) re-runs the entire script
from the top on every user message. A single agent conversation, however, spans
several such re-runs before it produces its `SUMMARY`. If the trace were created
with the standard decorator, it would be discarded and recreated on every
re-run, producing many fragmented, mostly empty traces instead of one coherent
trace per stage.

To solve this, the workflow does not rely on the `@opik.track` decorator for the
stage trace. Instead, it manages the trace lifecycle manually through helper
functions (defined in `agents/agent_tracing.py`) that keep a single trace alive
across re-runs:

- **`activate_stage_trace(...)`** - On the first turn of a stage, it creates the
  trace and stores its state in the session state. On every turn (including the
  first), it re-activates this stored trace as Opik's *current* trace, so that
  any spans produced during the run nest underneath it.
- **`finish_stage_trace(...)`** - When the stage produces its `SUMMARY`, this
  records the stage output and metadata, closes the trace, sends it to the
  server, and removes it from the session state.
- **`detach_stage_trace()`** - When a run ends without a `SUMMARY` (the
  conversation is not finished yet), this removes the trace from the current-run
  context but keeps it in the session state, so the next re-run resumes the same
  trace rather than starting a new one.

Because `activate_stage_trace` sets the stage trace as the current trace, the
retrieval spans executed during that stage (`vector_retrieval`,
`graph_retrieval`, `reranker`) nest underneath it automatically - the same tree
structure as the QA pipeline. In the agent workflow these retrieval spans are
grouped under a `search_question` span that is opened explicitly with
`opik.start_as_current_span(name="search_question")`, mirroring the QA layout.

### 6.3 Relationship to the QA pipeline

The agent workflow reuses the same decorated retrieval functions
(`vector_retrieval`, `reranker`) and the same graph-expansion logic as the QA
pipeline, and both write to the same Opik project (`dc-available-rag`) under a
shared thread. The essential difference is only in how the top-level trace is
managed: the QA pipeline uses the automatic `@opik.track` decorator for a
single-call request, whereas the agent workflow manages the trace manually so
that one trace can survive the multiple Streamlit re-runs that make up one
assessment stage.

## 7. Summary

Opik provides end-to-end observability for the system by recording each request
as a structured trace of nested spans. The `@opik.track` decorator captures
pipeline steps automatically, `update_current_span` and `update_current_trace`
enrich these records with cost, token usage, and results, and thread identifiers
group individual requests into coherent conversations. The QA pipeline uses the
decorator directly for single-call requests, while the agent workflow manages
its stage traces manually to remain coherent across the user interface's re-run
model. Combined with automated LLM-based quality scoring, this makes both the
performance and the answer quality of the system transparent and measurable.
