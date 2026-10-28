# User Guide

AVAILABLE helps you find and understand material in the indexed training corpus.
It does not search every company document or the public web. Available topics
depend on what the operator has processed and indexed.

## Assistant: Find a Learning Path

Use the Assistant when you need guidance rather than a single factual answer.
Describe the topic, your current knowledge, and what you want to achieve. The
guided assessment establishes the topic, knowledge level, and suitable teaching
material. Answer clarification questions with concrete goals and experience.

For example: "I know the basics of hydraulics and want training on pressure
control." Review the recommended material and check its applicability to your
equipment and role. A recommendation is not a certification of competence.

## Q&A: Ask a Specific Question

1. Open Q&A and enter a focused question, or select an example.
2. Choose a retrieval mode and select **Send**.
3. Read the answer and inspect **Sources**.
4. Open **Retrieved chunks** to compare the answer with the evidence.
5. Use **Related documents** to continue reading.

| Mode | Behavior | Use when |
| --- | --- | --- |
| Thinking | Vector and knowledge-graph retrieval. | You want both retrieval paths to contribute evidence. |
| Fast | Vector retrieval only; no graph traversal view. | You want a simpler retrieval path or are investigating graph-related latency. |

These names select retrieval strategies, not a guarantee of deeper reasoning or
a fixed response time. The Q&A page currently handles each submitted question
without passing previous Q&A turns as conversation history. Include the relevant
context in each question rather than relying on "it" or "the previous answer."

## Understand the Result

| Result area | Interpretation |
| --- | --- |
| Sources | Documents cited in the generated answer. Check the original where possible. |
| Related documents | Additional retrieval recommendations; not necessarily used in the answer. |
| Retrieved chunks | Evidence supplied to the language model. |
| Retrieval graph | Available in Thinking mode when graph data is returned; Graph and Tree layouts expose connections. |
| Logs | Link to the corresponding Opik trace when observability is available. |
| Trust value | A ranking signal or fallback value, not a probability that the answer is correct. |

An absent citation, an unrelated chunk, or a contradiction with the original
document is a reason to investigate, not to accept a fluent answer. Try a more
specific question with a training name, technical term, or procedure. Report a
problem with the question, selected mode, source titles, time, and trace link
where available. Do not include restricted content in an unapproved ticket.

## Handbook and Other Pages

The Handbook provides corpus summaries generated during preprocessing. These are
derived material and may omit detail from the originals.

Upload, Review, and RICH pages also exist in the frontend. Their availability
depends on the deployment. Upload and Review currently use legacy staging paths
that are not consistently aligned with the main `available` corpus. An upload or
approval must not be assumed to update the production index automatically.
Coordinate with the operator before using these pages for corpus changes.

RICH is a separate interview-oriented feature with its own data and collections;
it is not automatically included in the standard training-corpus Q&A build.

## Responsible Use

Use only authorized material. Follow source access restrictions, check dates and
versions, and verify important statements against the original. Do not use the
assistant as the sole authority for safety-critical or operational decisions.