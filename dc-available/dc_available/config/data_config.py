import os
from pathlib import Path

from dc_available import DATA_DIR

DEFAULT_DATA_SOURCE: str = "available"  # pipeline default: reads dc_available/data/local_sharepoint


DEFAULT_EMBEDDING_MODEL: str = "BAAI/bge-m3"  # Hybrid model with dense + sparse vectors
DEFAULT_COLOR: str = "blue-green-70"

APP_ID_ENV_VAR = "AVAILABLE_APP_SOURCE"

# Qdrant vector store (replaces the previous on-disk vector index). One collection is
# active per run; env vars let Docker point at the qdrant service later.
QDRANT_URL: str = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION: str = os.environ.get("QDRANT_COLLECTION", "available2.0")
# Stage B: second collection holding one point per graph entity (per topic).
QDRANT_ENTITY_COLLECTION: str = os.environ.get("QDRANT_ENTITY_COLLECTION", "available_entities")
# RICH feature: its own separate collection (plain vector RAG over the RICH interviews).
# Independent of the QA collections above; built by the self-contained RICH indexer.
QDRANT_RICH_COLLECTION: str = os.environ.get("QDRANT_RICH_COLLECTION", "RICH")
# RICH graph leg: its own entity collection, kept separate from QDRANT_ENTITY_COLLECTION so
# (re)building RICH's entity index never touches (and recreate-wipes) the QA entities.
QDRANT_RICH_ENTITY_COLLECTION: str = os.environ.get("QDRANT_RICH_ENTITY_COLLECTION", "available_rich_entities")
ENTITY_INDEX_BATCH_SIZE: int = 256  # embed/upsert batch size for the entity index

# ---------------------------------------------------------------------------
# Stage C: query-seeded graph retrieval (graph_retriever.retrieve).
# Independent of the vector retriever. All tunables live here for ablation.
# NOTE: the two tier floors and the confidence-signal choice are PLACEHOLDERS
# pending the C5 distribution study (fused vs. raw dense score); do not treat
# them as final.
# ---------------------------------------------------------------------------
GRAPH_QUERY_PHRASE_MODE: str = "raw"        # "raw" | "spacy" | "llm" (only raw implemented)
GRAPH_ENTITY_SEARCH_K: int = 50             # entities fetched from the index before filtering
GRAPH_MIN_FLOOR: float = 0.5                # confidence < this -> abstain (raised for max/dense band)
GRAPH_ENTITY_MATCH_FLOOR: float = 0.6       # confidence < this -> fallback mode (PLACEHOLDER, C5)
GRAPH_CONFIDENCE_SIGNAL: str = "max"        # "fused" | "dense" | "max" -> floor + walk weight + tier
GRAPH_MAX_SEED_ENTITIES: int = 8            # cap on seed entities (ordering by fused score)
GRAPH_FALLBACK_K: int = 3                   # chunks kept in fallback mode
GRAPH_CANDIDATE_K: int = 30                 # chunks kept in normal mode
GRAPH_SCORE_FLOOR_RATIO: float = 0.1        # drop chunks below this fraction of the top score
GRAPH_DOC_PRIOR_BETA: float = 0.3           # MaxP document prior strength; 0.0 disables (ablation)
GRAPH_LIST_WEIGHT_MIN: float = 0.4          # floor for the list's RRF weight in Stage D
GRAPH_TOPIC_CONCENTRATION_MIN: float = 0.5  # log-only for now (topic gate)
GRAPH_TOPIC_GATE_ENABLED: bool = False      # log-only; do not act on concentration yet
GRAPH_IDF_MISSING_MAX_RATIO: float = 0.01   # C7: raise if more than this fraction lack idf
GRAPH_DOC_TOPN: int = 10                     # recommended documents returned (MaxP ranking)
GRAPH_TRACE_TOPN: int = 5                    # rows shown per table in the debug trace
# Hierarchical graph debug/switch knobs (no behavior change until explicitly used).
GRAPH_HIERARCHICAL_ENABLED: bool = False           # OFF: keep entity_1hop a pure 1 hop (no module expansion)
GRAPH_HIERARCHICAL_MODULE_RATIO_MIN: float = 1.15  # top1/top2 module score ratio
GRAPH_HIERARCHICAL_MODULE_SCORE_MIN: float = 3.0   # minimum absolute score for top module
GRAPH_HIERARCHICAL_TOP_MODULES: int = 2            # modules to keep when hierarchy is used
GRAPH_HIERARCHICAL_TOP_DOCS: int = 5               # documents to keep inside selected modules
GRAPH_HIERARCHICAL_DOC_QUERY_BOOST: float = 0.8    # extra score from query-title token overlap in module doc ranking
GRAPH_MAX_CHUNKS_PER_DOC: int = 2                  # diversity cap for final Stage-C chunk list per document
# Stage C engine selector: "entity_1hop" (legacy 1-hop entity->chunk) or "ppr"
# (Personalized PageRank over the semantic + structural graph; see
# dc_available/graph/ppr_retriever.py). PPR tunables are in the PPR block below.
GRAPH_RETRIEVAL_MODE: str = "ppr"

# --- PPR graph leg (used only when GRAPH_RETRIEVAL_MODE="ppr") -------------
# Personalized PageRank over the combined semantic (mentions + entity relations)
# and structural (part_of / contains / related_to) graph. Entity seeds are
# weighted by signal x idf (node specificity); optional dense passage seeds come
# from the vector leg. See dc_available/graph/ppr_retriever.py.
PPR_ALPHA: float = 0.5                    # walk damping; lower = stay nearer seeds = more precision
PPR_MAX_ITER: int = 100                   # power-iteration cap
PPR_TOL: float = 1e-6                     # L1 convergence tolerance
PPR_PASSAGE_SEED_WEIGHT: float = 0.2      # seed-mass fraction for vector-leg chunks (0 = entity-only)
# Per-family edge weights in the PPR adjacency.
PPR_MENTIONS_EDGE_WEIGHT: float = 1.0     # Chunk <-> Entity
PPR_SEMANTIC_EDGE_WEIGHT: float = 0.8     # Entity <-> Entity relations (scaled by confidence)
PPR_RELATED_EDGE_WEIGHT: float = 0.5      # Chunk <-> Chunk sequence links
PPR_STRUCTURAL_EDGE_WEIGHT: float = 0.3   # part_of / contains hierarchy (pool inside a doc/module)

# --- PPR seed selection v2 (threshold + boilerplate filter + concept merge) --
# New seed pipeline for the PPR leg, gated by PPR_SEED_MODE. "legacy" keeps the
# top-GRAPH_MAX_SEED_ENTITIES-by-fused behaviour; "merge" casts a wide net by a
# max(fused,dense) threshold, drops boilerplate by chunk_frequency, then merges
# same-concept entity fragments into one pooled seed (weight = strongest member).
# See dc_available/graph/ppr_retriever.py (_select_seeds). Query-time only; no
# graph rebuild. Flip PPR_SEED_MODE back to "legacy" to revert instantly.
PPR_SEED_MODE: str = "merge"              # "legacy" | "merge"
PPR_SEED_FETCH_K: int = 50                # entities fetched from the index before filtering
PPR_SEED_DENSE_THRESHOLD: float = 0.6     # keep entities with max(fused,dense) >= this
PPR_SEED_TOPN_BACKSTOP: int = 5           # if fewer survive the threshold, keep this many best (never empty)
PPR_SEED_MAX_CF: int = 150                # boilerplate filter: drop entities in > this many chunks
PPR_SEED_MERGE: bool = True               # merge same-concept fragments (False = threshold+filter only)
PPR_SEED_DEBUG: bool = False              # print per-stage seed-selection diagnostics to stdout

# entity_1hop can reuse the SAME seed pipeline as ppr_merge (threshold +
# boilerplate filter + concept merge + topic isolation) so the two engines are
# compared on identical seeds (fair best-vs-best). "legacy" = the old top-8-by-
# fused gate. Affects only the entity_1hop engine (GRAPH_RETRIEVAL_MODE="entity_1hop").
GRAPH_ENTITY_HOP_SEED_MODE: str = "merge"   # "legacy" | "merge"

# --- entity_1hop walk depth (k-hop) ----------------------------------------
# Number of entity->chunk hops for the entity engine. 1 = the classic single
# hop (byte-identical to before). k>1 walks further: chunk -> its entities ->
# more chunks, repeated, with each extra hop's contribution scaled by
# GRAPH_ENTITY_HOP_DECAY (so far-away chunks count less). Safeguards: each new
# chunk is scored once (dedup), the walk stops as soon as a hop finds nothing
# new, and entities touching more than GRAPH_ENTITY_HOP_HUB_CAP chunks are
# skipped on the extra hops (generic "hub" entities = noise; 0 = off). Affects
# only GRAPH_RETRIEVAL_MODE="entity_1hop".
GRAPH_ENTITY_HOP_K: int = 1                 # 1 = single hop (unchanged); 2, 3, ...
GRAPH_ENTITY_HOP_DECAY: float = 0.5         # weight multiplier per extra hop
GRAPH_ENTITY_HOP_HUB_CAP: int = 0           # skip entities in > N chunks on extra hops (0 = off)

# --- PPR topic isolation (Stage 6) -----------------------------------------
# Sum the seed weight (max(fused,dense) x idf) per topic and run PPR only in the
# dominant topic(s) — so a same-named entity (e.g. "SAP S/4HANA") in a foreign
# topic can't pull the walk there. "Soft": keep every topic whose seed mass is
# within PPR_TOPIC_KEEP_RATIO of the top topic (so genuine cross-topic queries
# keep both). Set PPR_TOPIC_ISOLATION=False to revert to all-matched-topics.
PPR_TOPIC_ISOLATION: bool = True          # restrict PPR to the dominant seed topic(s)
PPR_TOPIC_KEEP_RATIO: float = 0.5         # keep topics with seed mass >= this fraction of the top

# ---------------------------------------------------------------------------
# Stage D: parallel hybrid retrieval (RAG/fusion.py, qa_functions).
# Two independent legs (vector + Stage-C graph) each put their top chunks into one
# candidate pool, which a cross-encoder reranker (Option 1) reorders. The final
# top chunks go to the LLM; the recommended-documents list is just the unique
# documents behind those chunks. FUSION_STRATEGY selects how the pool is built:
#   "union" — de-duplicated union of both legs' top chunks (default). Neither leg
#             can bury the other; the reranker does the precision filtering.
#   "rrf"   — legacy weighted Reciprocal Rank Fusion (kept for ablation/comparison).
# ---------------------------------------------------------------------------
FUSION_STRATEGY: str = "union"             # "union" (default) | "rrf" (ablation/comparison)
FUSION_LEGS: str = "both"                   # "both" (default) | "vector" | "graph" (ablation isolation)
FUSION_VECTOR_K: int = 10                   # top chunks the vector leg puts into the pool
FUSION_GRAPH_K: int = 10                    # top chunks the graph leg puts into the pool
FUSION_OUTPUT_K: int = 10                   # chunks sent to the LLM after reranking
FUSION_DOC_TOPN: int = 5                    # recommended documents (unique docs of the final chunks)
FUSION_GRAPH_VIZ_ENTITIES_PER_CHUNK: int = 5  # entities shown per chunk in the QA retrieval graph
# --- RRF-only knobs (used when FUSION_STRATEGY="rrf") ---
FUSION_RRF_K: int = 60                      # RRF damping constant (standard)
FUSION_VECTOR_WEIGHT: float = 1.0           # RRF weight for the vector leg
FUSION_GRAPH_WEIGHT_MODE: str = "fixed"    # "fixed" | "list_weight" (ablation)
FUSION_GRAPH_WEIGHT_FIXED: float = 1.0      # RRF weight for the graph leg when mode="fixed"
FUSION_CANDIDATE_K: int = 20               # fused chunks kept as reranker candidates

# Cross-encoder reranker (Stage D, Option 1): reorders the fused candidates by
# reading query+chunk text together. Disable to keep the plain fusion order.
RERANKER_ENABLED: bool = False
RERANKER_MODEL: str = "BAAI/bge-reranker-v2-m3"

# Document trust (ingestion-time final_trust, per document). Applied as a final
# multiplier on the reranker score: final = rerank_score * (1 + TRUST_BETA * final_trust).
# Query-independent — it only reorders the already-relevant chunks. Tune TRUST_BETA
# on the evaluation set; 0 disables the effect. See dc_available/trust_score/.
TRUST_ENABLED: bool = True
TRUST_BETA: float = 0.5

MAIN_FOLDER: str = "Please_set_me"  # Will be changed after initial setup
DEFAULT_DATA_PATH: Path = DATA_DIR / MAIN_FOLDER
DEFAULT_CACHE_PATH: Path = DATA_DIR / MAIN_FOLDER / "vector_index"
DEFAULT_IMAGE_PATH: Path = DATA_DIR / MAIN_FOLDER / "images"
DEFAULT_VIDEO_PATH: Path = DATA_DIR / MAIN_FOLDER / "video"
DEFAULT_PPTX_PATH: Path = DATA_DIR / MAIN_FOLDER / "pptx"
DEFAULT_PDF_PATH: Path = DATA_DIR / MAIN_FOLDER / "pdf"
DEFAULT_ARTIFACTS_ROOT: Path = DATA_DIR / "artifacts" / MAIN_FOLDER
DEFAULT_INGESTION_ARTIFACTS_PATH: Path = DEFAULT_ARTIFACTS_ROOT / "ingestion"
DEFAULT_INDEX_ARTIFACTS_PATH: Path = DEFAULT_ARTIFACTS_ROOT / "index"
DEFAULT_IMAGE_ARTIFACTS_PATH: Path = DEFAULT_ARTIFACTS_ROOT / "images"
DEFAULT_GRAPH_ARTIFACTS_PATH: Path = DEFAULT_ARTIFACTS_ROOT / "graph"
DEFAULT_HANDBOOK_ARTIFACTS_PATH: Path = DATA_DIR / "artifacts" / "handbook" / MAIN_FOLDER
DEFAULT_HANDBOOK_PATH: Path = DEFAULT_HANDBOOK_ARTIFACTS_PATH
DEFAULT_SEMANTIC_BATCH_SIZE: int = 5
DEFAULT_SEMANTIC_BATCH_OVERLAP: int = 1
DEFAULT_SEMANTIC_MIN_CHARS: int = 120


_data_config = {
    "dc_available": {
        "main_folder": "DC_data",
        "handbook_folder": "available_handbook",
    },
    "test": {
        "main_folder": "TEST_data",
        "handbook_folder": "available_handbook_test",
    },
    "other": {
        "main_folder": "MY_data",
        "handbook_folder": "available_handbook_other",
    },
    # Data pulled from the AVAILABLE SharePoint. The root is the whole
    # available_sharepoint/resources folder; training folders are found anywhere
    # under it (see _discover_trainings), so wrapper folders are optional.
    # base_dir overrides the default data root (DATA_DIR = dc_available/resources).
    "available": {
        # New pipeline source: raw mirror produced by pipeline/sync.py, with a
        # clean output layout that keeps generated files OUT of the raw folder.
        "main_folder": "local_sharepoint",
        "artifact_folder": "resources",   # marks the Artifacts/ + Extracted_resources/ layout
        "base_dir": DATA_DIR.parent / "dc_available" / "data",
        # sync.py: which SharePoint site/folder to mirror (env var names in pipeline/.env).
        "site_env": "AVAILABLE_SITE_ID",
        "remote_folder_env": "AVAILABLE_FOLDER_PATH",
    },
    # RICH feature source: interviews living in a sibling `RICH` folder of the SAME
    # AVAILABLE SharePoint. Mirrored into local_sharepoint/rich (its own manifest),
    # and its artifacts/collection are fully separate from the QA data above.
    "rich": {
        "main_folder": "local_sharepoint/rich",
        "artifact_folder": "rich",        # -> Artifacts/rich/ + Extracted_resources/rich/
        "base_dir": DATA_DIR.parent / "dc_available" / "data",
        "site_env": "AVAILABLE_SITE_ID",  # same site as `available`
        "remote_folder_env": "RICH_FOLDER_PATH",  # SharePoint path to the RICH folder
    },
}


def source_artifact_paths(source: str) -> dict:
    """Resolve a source's key paths WITHOUT mutating the global config.

    The RICH feature runs in the same process as QA, so it must not call
    update_data_config() (that would repoint the globals QA relies on). This returns
    just the paths RICH needs, using the new Artifacts/<artifact_folder>/ layout.
    """
    cfg = _data_config[source]
    base = cfg.get("base_dir", DATA_DIR)
    artifact_folder = cfg["artifact_folder"]
    artifacts_root = base / "Artifacts" / artifact_folder
    return {
        "data_path": base / cfg["main_folder"],
        "ingestion": artifacts_root / "ingestion",
        "index": artifacts_root / "index",
    }


def update_data_config(app_name: str = DEFAULT_DATA_SOURCE) -> None:
    global _data_config
    global MAIN_FOLDER, DEFAULT_DATA_PATH, DEFAULT_CACHE_PATH, DEFAULT_IMAGE_PATH, DEFAULT_VIDEO_PATH, DEFAULT_PPTX_PATH, DEFAULT_PDF_PATH, DEFAULT_HANDBOOK_PATH
    global DEFAULT_ARTIFACTS_ROOT, DEFAULT_INGESTION_ARTIFACTS_PATH, DEFAULT_INDEX_ARTIFACTS_PATH, DEFAULT_IMAGE_ARTIFACTS_PATH, DEFAULT_GRAPH_ARTIFACTS_PATH, DEFAULT_HANDBOOK_ARTIFACTS_PATH

    app_name = os.environ.get(APP_ID_ENV_VAR, app_name)
    source = _data_config[app_name]
    base = source.get("base_dir", DATA_DIR)   # per-source data root (defaults to DATA_DIR)
    MAIN_FOLDER = source["main_folder"]
    DEFAULT_DATA_PATH = base / MAIN_FOLDER

    artifact_folder = source.get("artifact_folder")
    if artifact_folder:
        # New pipeline layout — generated files NEVER go inside the raw data folder:
        #   <base>/<main_folder>/                 raw mirror (read-only for the pipeline)
        #   <base>/Artifacts/handbook/            handbook
        #   <base>/Artifacts/<artifact_folder>/{ingestion,index,graph}
        #   <base>/Extracted_resources/<artifact_folder>/{images,video,pptx,pdf}
        artifacts_root = base / "Artifacts" / artifact_folder
        extracted_root = base / "Extracted_resources" / artifact_folder
        DEFAULT_ARTIFACTS_ROOT = artifacts_root
        DEFAULT_INGESTION_ARTIFACTS_PATH = artifacts_root / "ingestion"
        DEFAULT_INDEX_ARTIFACTS_PATH = artifacts_root / "index"
        DEFAULT_GRAPH_ARTIFACTS_PATH = artifacts_root / "graph"
        DEFAULT_HANDBOOK_ARTIFACTS_PATH = base / "Artifacts" / "handbook"
        DEFAULT_HANDBOOK_PATH = DEFAULT_HANDBOOK_ARTIFACTS_PATH
        DEFAULT_IMAGE_ARTIFACTS_PATH = extracted_root / "images"
        # caches / extracted media live OUTSIDE the raw data folder
        DEFAULT_CACHE_PATH = artifacts_root / "index"
        DEFAULT_IMAGE_PATH = extracted_root / "images"
        DEFAULT_VIDEO_PATH = extracted_root / "video"
        DEFAULT_PPTX_PATH = extracted_root / "pptx"
        DEFAULT_PDF_PATH = extracted_root / "pdf"
    else:
        # Legacy layout (unchanged) — dc_available / test / other.
        DEFAULT_CACHE_PATH = base / MAIN_FOLDER / "vector_index"
        DEFAULT_IMAGE_PATH = base / MAIN_FOLDER / "images"
        DEFAULT_VIDEO_PATH = base / MAIN_FOLDER / "video"
        DEFAULT_PPTX_PATH = base / MAIN_FOLDER / "pptx"
        DEFAULT_PDF_PATH = base / MAIN_FOLDER / "pdf"
        DEFAULT_HANDBOOK_PATH = base / source["handbook_folder"]
        DEFAULT_ARTIFACTS_ROOT = base / "artifacts" / MAIN_FOLDER
        DEFAULT_INGESTION_ARTIFACTS_PATH = DEFAULT_ARTIFACTS_ROOT / "ingestion"
        DEFAULT_INDEX_ARTIFACTS_PATH = DEFAULT_ARTIFACTS_ROOT / "index"
        DEFAULT_IMAGE_ARTIFACTS_PATH = DEFAULT_ARTIFACTS_ROOT / "images"
        DEFAULT_GRAPH_ARTIFACTS_PATH = DEFAULT_ARTIFACTS_ROOT / "graph"
        DEFAULT_HANDBOOK_ARTIFACTS_PATH = base / "artifacts" / "handbook" / MAIN_FOLDER
        DEFAULT_HANDBOOK_PATH = DEFAULT_HANDBOOK_ARTIFACTS_PATH


update_data_config()
