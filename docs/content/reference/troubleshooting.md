# Troubleshooting

Start with the command, selected source, collection names, and the first relevant
error in the logs. Avoid deleting artifacts or rebuilding everything before you
identify which system boundary failed.

| Symptom | Check | Next action |
| --- | --- | --- |
| Poetry rejects Python | Interpreter version. | Use Python 3.10 or 3.11, then reinstall in the selected environment. |
| Qdrant unreachable | Container status and `QDRANT_URL`. | Start the intended instance; account for container-versus-host networking. |
| Docstore not found | Source and `Artifacts/resources/index/index_ret.pkl`. | Restore matching trusted artifacts or complete indexing after preprocessing. |
| Collection not found | `QDRANT_COLLECTION` and Qdrant dashboard. | Select the intended collection or build it; do not create an empty stand-in. |
| Graph missing or empty | Structural/semantic outputs and entity collection. | Complete graph then semantic indexing for the same corpus. Fast Q&A can help isolate graph behavior. |
| Azure unauthorized | Identity, endpoint, deployment, and role. | Ask the owner to verify access; never paste credentials into an issue. |
| First request is slow | Model downloads, CPU/memory, and cold caches. | Distinguish cold-start work from steady-state retrieval using logs/traces. |
| Docker APT cannot find LibreOffice | Earlier proxy/DNS or package-index errors. | Fix repository reachability before changing dependencies. |
| Sync says no changes after editing a remote file | Existing manifest entry. | Known-file updates are metadata-only in current sync; plan controlled recovery. |
| Pipeline stops after sync | Additions/deletions reported. | Use explicit local stages for local files; the orchestrator is not a forced rebuild. |
| No trust build from `pipeline.trust_score` | Entry point. | Use the supported GROWNFIELD trust commands, with integration access. |
| All displayed trust values are `1.00` | Trust table presence and coverage. | This may be a fallback, not computed high reliability. |
| No Opik feedback scores | Judge setting and judge model access. | Enable only when needed; inspect errors separately from answer generation. |
| Upload does not affect Q&A | Upload/review staging versus active corpus. | Coordinate publishing, preprocessing, and indexing with the operator. |
| Old behavior after replacing artifacts | In-process caches and artifact compatibility. | Restart after a controlled replacement; verify the full build set. |

## Poor Answers

Inspect the retrieved passages before changing prompts. If the correct evidence
is missing, investigate corpus coverage, extraction quality, retrieval mode, and
index consistency. If the evidence is present but the answer is wrong, investigate
generation and source attribution. More graph hops or a higher trust boost is
not a universal fix.

## Before Reporting a Problem

Include the command or UI mode, timestamp, source key, model/deployment,
collection names, sanitized error, and trace identifier if available. State
whether the issue reproduces on a known small example. Remove credentials,
private endpoints where required, personal data, and restricted passages.

## Known Limitations

- Remote edits to known paths do not refresh local file content in current sync.
- Automatic refresh rebuilds the index; it is not incremental vector indexing.
- Trust build scope is corpus-specific, and missing matches use fallback values.
- Optional reranking and judge stages shown in diagrams may be disabled.
- Upload/review paths and the existing Compose layout need separate integration work.
- Generated summaries, entities, scores, and answers require source verification.
- Historical deployment status and screenshot metrics are not live health checks.

See [pipeline operations](../operations/pipeline.md) and
[configuration](configuration.md) before attempting recovery.