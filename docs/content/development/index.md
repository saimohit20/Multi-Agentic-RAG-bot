# Developer Guide

Begin with [Getting Started](../getting-started/index.md) and the
[system architecture](../architecture/index.md). This guide identifies ownership
boundaries and development checks without explaining individual functions.

## Technical Design Guides

| Guide | Technical scope |
| --- | --- |
| [Graph Retrieval Design](graph-retrieval.md) | Construction artifacts, entity matching, topic routing, and the engine contract. |
| [Personalized PageRank](ppr.md) | Weighted adjacency, personalization, restart, convergence, and passage readout. |
| [Entity-Hop Approaches](entity-hop.md) | Direct scoring, bounded multi-hop expansion, and module-local alternatives. |
| [Vector Retrieval Design](vector-retrieval.md) | Dense/sparse representations, document cards, identity, and index compatibility. |
| [Hybrid Retrieval Design](hybrid-retrieval.md) | Candidate fusion, reranking, trust interaction, and failure behavior. |
| [Trust Scoring Design](trust-scoring.md) | Metadata/content priors, document TrustRank, publication, lookup, and ranking limits. |

## Working Environment

Use Python 3.10 or 3.11 and install the development dependency group with
`poetry install --with dev`. Run commands from `dc-available`. Keep credentials,
training data, model caches, and generated artifacts out of commits. Inspect the
worktree before starting; do not overwrite another contributor's local changes.

## Where Changes Belong

| Area | Responsibility |
| --- | --- |
| `pipeline/` | Stage commands, orchestration, and operational sequencing. |
| `dc_available/summarization/` | Document extraction and derived summaries. |
| `dc_available/RAG/` | Vector retrieval, fusion, ranking, generation, and RICH retrieval. |
| `dc_available/graph/` | Graph construction and graph-based retrieval. |
| `dc_available/trust_score/` | Metadata collection, prior, propagation, and published trust. |
| `dc_available/agents/` | Curriculum and learning-assistant orchestration. |
| `dc_available/frontend/` | Streamlit pages and user-facing evidence presentation. |
| `dc_available/config/` | Source, retrieval, model, and endpoint configuration. |
| `dc_available/common/` | Shared paths, integrations, and helpers. |
| `evaluation/` | Datasets, retrieval/generation evaluation, and experiment outputs. |
| `deployment/` | Container and Kubernetes operational assets. |
| Workspace `docs/content/` | Authoritative user, developer, and operations guides for the curated site. |

## Preserve the Data Contracts

Chunk identifiers connect ingestion, Qdrant, the graph, and parent context.
Changing identity or metadata rules can break evidence reconstruction without
producing a simple startup error. Treat a source-layout or embedding change as
a migration requiring coordinated artifacts and evaluation.

Do not mix RICH interview collections or trust outputs with the main training
corpus. The standard pipeline and RICH pipeline have different build behavior;
see [RICH operations](../operations/rich.md).

## Suggested Contribution Workflow

1. Define the user-visible or operational behavior being changed.
2. Identify the owning subsystem and a small representative corpus/query set.
3. Isolate data, collections, and output directories from shared environments.
4. Implement and run focused checks appropriate to that subsystem.
5. Compare quality, latency, cost, and citations before and after the change.
6. Update configuration references and guides alongside changed behavior.
7. Review the diff for secrets, generated files, and unrelated changes.

The project declares pytest and pre-commit as development tools; their presence
does not imply a comprehensive runnable test suite or configured hook policy.
Use existing focused tests where available and record integration prerequisites.
Do not report an evaluation run as a unit-test suite.

## Evaluations

| Guide | Scope |
| --- | --- |
| [Retrieval Evaluation](retrieval-evaluation.md) | Experimental controls, source matching, document ranking, and metric definitions. |
| [Generator Evaluation](generator-evaluation.md) | Context reuse, answer generation, LLM judging, RAGAS, and human-review procedure. |
| [Trust Score Experiments](trust-score-experiments.md) | Controlled score sensitivity, paired comparisons, and isolated RAG experiments. |

The evaluation workspace includes real Grownfield and Hydraulics questions,
synthetic datasets, component studies, and baseline comparisons. The evaluation
plan distinguishes development selection from held-out assessment. Preserve that
separation when adding experiments.

For the existing retrieval evaluation entry point:

```powershell
poetry run python -m evaluation.available.run_pipeline --topics grownfield hydraulics --only retrieval --skip-synthetic
```

This is an integration experiment, not an offline docs check. It requires the
appropriate datasets, indexes, model access, and dependencies and may incur costs.
Read `evaluation/EVALUATION_PLAN.md` and the selected runner options before
launching a larger study. Do not overwrite published experiment outputs.
The consolidated `--limit` flag does not cap retrieval questions.

| Dimension | Report |
| --- | --- |
| Retrieval | Hit@k, MRR, corpus, and document-matching policy. |
| Generation | Judge/human assessments and applicable RAGAS metrics. |
| Performance | End-to-end and stage latency; cold versus warm model/cache state. |
| Reproducibility | Commit, dataset, model, settings, collection build, and run time. |

An automated judge can disagree with human reviewers. Metric definitions and
sample sizes belong beside results; a dashboard screenshot is not a benchmark.

## Handover

Record known limitations, rebuild/rollback requirements, configuration changes,
and checks not run. Follow [documentation maintenance](documentation.md) when
adding or moving guides.