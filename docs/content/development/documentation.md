# Documentation Maintenance

The `docs/` folder is the self-contained documentation package. Its `content/`
directory is the authoritative source for the curated website, and
`docs/mkdocs.yml` configures the build. Documentation dependencies are listed in
`docs/requirements.txt`, independently of the application. Source pages remain
readable as Markdown; the local site adds navigation, search, and section tables
of contents.

## Preview and Validate Locally

From `docs`, with the documentation dependencies installed in your selected
Python environment:

```powershell
python -m mkdocs build --strict
python -m mkdocs serve --dev-addr 127.0.0.1:8000
```

Open <http://127.0.0.1:8000>. If that port is in use, choose another, such as
`--dev-addr 127.0.0.1:8001`. The site output is generated under `docs/site/`,
which is ignored by `docs/.gitignore`; it is not the authoritative source.
Do not commit generated HTML. Preview and build do not require application
credentials or a training corpus. The package README describes an isolated
documentation environment. No hosting or publishing pipeline is configured by
these documentation changes.

## Page Conventions

Use task-oriented titles, a concise scope statement, prerequisites, ordered
procedures, expected outcomes, and troubleshooting where relevant. Describe
concepts and responsibilities rather than individual functions. Include commands
and configuration examples only when they help someone complete a task.

Distinguish current defaults, optional capabilities, historical decisions, and
unverified environment state. Use placeholders instead of private environment
values. If a command rebuilds collections, deletes files, costs money, or writes
remotely, warn before it. Verify a claim against the owning configuration or
entry point before presenting it as supported behavior.

Add new pages to `nav` in the MkDocs configuration and use relative Markdown links
between published pages. Do not link site pages directly to source files outside
the docs tree: those files are not served by MkDocs. Refer to repository paths
as text when describing ownership.

## Figures

Keep supplied figures under `content/assets/figures/`, grouped by architecture, pipeline,
trust, graph, observability, and deployment. Use descriptive alt text, a caption,
and a full-size link for dense diagrams. Prefer editable originals when available.
The existing PNGs are preserved without changing the supplied artwork.

Known figure caveats are documented beside the relevant images: the overview and
hybrid diagrams include optional reranking; the graph drawing abbreviates Level
and mislabels one mentions relationship; the AKS drawing simplifies identity.
The Opik screenshot contains example corpus questions and answer text. Obtain
content-owner approval before sharing the site outside its intended audience.

## Legacy Material and Migration Map

Original documents are retained at their existing paths in the development
workspace to preserve research, history, and local changes. They are not copied
into this package and are not dependencies of its build. The old code
walkthroughs are not part of the navigation or search, and the historical notes
are not current runbooks. Use `docs/content/` for ongoing site maintenance rather
than editing the earlier copies in `documentation/`.

| Original workspace-relative source | Current destination or status |
| --- | --- |
| `documentation/README.md` | New landing page and architecture overview; original retained. |
| `documentation/understand_project_readme.md` | Architecture and developer ownership map; detailed code walkthrough retained outside site. |
| `documentation/readmeUML.md` | Historical diagrams; supplied figures are used in current guides. |
| `documentation/sharepoint-integration.md` | Operations / SharePoint. |
| `documentation/sync.md` | Operations / SharePoint and Pipeline, with verified known-file limitations. |
| `documentation/pre-process.md`, `pre-process-deep-dive.md` | Operations / Pipeline; extractor internals retained outside site. |
| `documentation/vector.md`, `hybrid.md` | Architecture / Hybrid Retrieval. |
| `documentation/graph.md` | Architecture / Knowledge Graph. |
| `documentation/trust-score.md` | Architecture / Document Trust, including current build scope and fallbacks. |
| `documentation/opik-observability.md`, `observability-and-evaluation.md` | Operations / Observability and Developer / Evaluation guidance. |
| `documentation/deployment_aks.md` | Deployment / Azure AKS and Local Docker; original deployment log left untouched. |
| `documentation/Entity_Hop_Note.md`, `GRAPH_APPROACHES.md`, `GRAPH_LEG_PLAN.md` | Research/experimental design history; retained outside site. |
| `documentation/graph-retrieval-latency-optimization.md`, `reranker-ablation-study.md` | Performance/research notes; retained outside site. |
| `documentation/trust_evaluation.md`, `EXPERIMENTS_RESULTS_CHAPTER_DRAFT.md` | Evaluation research and draft results; retained outside site, not release claims. |
| `dc-available/docs/qdrant.md` | Operations / Qdrant Setup, Getting Started, and Configuration. |
| `dc-available/docs/opik.md` | Operations / Opik Setup and Observability; original workarounds are labeled environment-specific. |
| `dc-available/deployment/docker/README.md`, `deployment/kubernetes/README.md` | Deployment guides; original operational references retained. |
| `dc-available/evaluation/EVALUATION_PLAN.md` | Remains the detailed evaluation-plan reference in the repository. |

If old links or external consumers need relocation later, add explicit pointers
before removing old paths. Do not delete research drafts or rewrite deployment
history solely to make the navigation cleaner.

## Release Review

1. Run the strict build and resolve broken links and missing images.
2. Check search, navigation, code blocks, and full-size images in the browser.
3. Inspect desktop and narrow/mobile layouts, especially tables and diagrams.
4. Validate changed commands in an isolated environment when access permits.
5. Review documents and figures for secrets, personal data, and restricted content.
6. Confirm the diff contains only intended documentation/site changes.

A passing docs build does not prove that the application installs, Azure roles
are assigned, external services are healthy, or a pipeline command succeeds.
Record checks not performed and unresolved access constraints in the handover.