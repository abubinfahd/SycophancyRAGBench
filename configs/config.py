"""
Central configuration for SycophancyRAGBench.
Supports dynamic environment adaptation across Google Colab, Kaggle, and Local/Server runtimes.
"""

import os
import sys

# ==============================================================================
# 1. Environment Detection & Directory Layout
# ==============================================================================
def detect_runtime():
    """Detect whether running in Google Colab, Kaggle Notebooks, or a Local/Server environment."""
    if "KAGGLE_KERNEL_RUN_TYPE" in os.environ or os.path.exists("/kaggle/working"):
        return "kaggle"
    if "google.colab" in sys.modules or os.path.exists("/content"):
        return "colab"
    return "local"

RUNTIME = detect_runtime()

# Base project root depends on platform
if RUNTIME == "colab":
    # On Google Colab, persist to Google Drive if mounted, else use /content/SycophancyRAGBench
    DRIVE_ROOT = "/content/drive/MyDrive/SycophancyRAGBench"
    PROJECT_DIR = DRIVE_ROOT
    OUTPUT_ZIP_PATH = "/content/sycophancy_rag_bench_outputs.zip"
elif RUNTIME == "kaggle":
    # On Kaggle, write to working directory and bundle to downloadable zip
    PROJECT_DIR = "/kaggle/working/SycophancyRAGBench"
    OUTPUT_ZIP_PATH = "/kaggle/working/sycophancy_rag_bench_outputs.zip"
else:
    # Local workspace or server
    _this_dir = os.path.dirname(os.path.abspath(__file__))
    _pkg_root = os.path.abspath(os.path.join(_this_dir, ".."))
    PROJECT_DIR = os.path.join(_pkg_root, "artifacts")
    OUTPUT_ZIP_PATH = os.path.join(_pkg_root, "sycophancy_rag_bench_outputs.zip")

# Subdirectories
DATA_DIR       = os.path.join(PROJECT_DIR, "data")
CORPUS_DIR     = os.path.join(PROJECT_DIR, "corpus")
COLLECTION_DIR = os.path.join(CORPUS_DIR, "collection")
INDEX_DIR      = os.path.join(CORPUS_DIR, "index")
RETRIEVAL_DIR  = os.path.join(PROJECT_DIR, "retrieval")
GEN_DIR        = os.path.join(PROJECT_DIR, "generations")
EVAL_DIR       = os.path.join(PROJECT_DIR, "eval")
STATS_DIR      = os.path.join(PROJECT_DIR, "stats")
FIG_DIR        = os.path.join(STATS_DIR, "figures")
TABLE_DIR      = os.path.join(STATS_DIR, "tables")
RELEASE_DIR    = os.path.join(PROJECT_DIR, "release")

ALL_DIRS = [
    PROJECT_DIR, DATA_DIR, CORPUS_DIR, COLLECTION_DIR, INDEX_DIR,
    RETRIEVAL_DIR, GEN_DIR, EVAL_DIR, STATS_DIR, FIG_DIR, TABLE_DIR, RELEASE_DIR
]

# Source data bundled with package
_this_dir = os.path.dirname(os.path.abspath(__file__))
BUNDLED_DATA_DIR = os.path.abspath(os.path.join(_this_dir, "..", "data"))
BUNDLED_CLAIMS = os.path.join(BUNDLED_DATA_DIR, "claims.jsonl")
BUNDLED_PROMPTS = os.path.join(BUNDLED_DATA_DIR, "prompts.jsonl")
BUNDLED_GOLD = os.path.join(BUNDLED_DATA_DIR, "gold_passages.jsonl")
BUNDLED_DISTRACTORS = os.path.join(BUNDLED_DATA_DIR, "distractors_sample.jsonl")
BUNDLED_HUMAN_LABELS = os.path.join(BUNDLED_DATA_DIR, "manual_validation_labeled.csv")

# Key artifact filepaths
CLAIMS_PATH       = os.path.join(DATA_DIR, "claims.jsonl")
PROMPTS_PATH      = os.path.join(DATA_DIR, "prompts.jsonl")
GOLD_PASSAGES_PATH = os.path.join(COLLECTION_DIR, "gold.jsonl")
DISTRACTORS_PATH  = os.path.join(COLLECTION_DIR, "distractors.jsonl")
CORPUS_STATE      = os.path.join(CORPUS_DIR, "build_state.json")

RETRIEVAL_RESULTS = {
    "bm25":       os.path.join(RETRIEVAL_DIR, "retrieval_bm25.jsonl"),
    "dense":      os.path.join(RETRIEVAL_DIR, "retrieval_dense.jsonl"),
    "hybrid":     os.path.join(RETRIEVAL_DIR, "retrieval_hybrid.jsonl"),
    "oracle":     os.path.join(RETRIEVAL_DIR, "retrieval_oracle.jsonl"),
    "irrelevant": os.path.join(RETRIEVAL_DIR, "retrieval_irrelevant.jsonl"),
}
RETRIEVAL_METRICS = os.path.join(RETRIEVAL_DIR, "retrieval_metrics.json")

JUDGED_PATH       = os.path.join(EVAL_DIR, "judged.jsonl")
METRICS_PATH      = os.path.join(EVAL_DIR, "metrics.json")
CAUSAL_METRICS_PATH = os.path.join(EVAL_DIR, "causal_conflict_metrics.json")
MANUAL_SAMPLE     = os.path.join(EVAL_DIR, "manual_validation_sample.csv")
MANUAL_LABELS     = os.path.join(EVAL_DIR, "manual_validation_labeled.csv")

# ==============================================================================
# 2. Benchmark Design & Dataset Parameters
# ==============================================================================
SEED = 42

# Dataset Scale Configuration:
# Set TAKE_ALL_DATA = True to extract 100% of all available data
# (All available FEVER refuted claims + All 817 TruthfulQA misconceptions)
TAKE_ALL_DATA = False

N_FEVER = None if TAKE_ALL_DATA else 200
N_TRUTHFULQA = None if TAKE_ALL_DATA else 100
BELIEF_STYLES = ["neutral", "weak", "strong", "authority", "emotional", "social"]

# Domains
DOMAINS = ["medical", "scientific", "historical", "political", "technology", "climate", "general"]

# ==============================================================================
# 3. Model Suite (Addressing Recommendation 1: Multi-family & SOTA Baselines)
# ==============================================================================
MODELS = [
    {
        "name": "qwen2.5-3b",
        "hf_id": "Qwen/Qwen2.5-3B-Instruct",
        "family": "causal",
        "gated": False,
        "description": "Alibaba Qwen 2.5 3B (Dense Causal LM)"
    },
    {
        "name": "gemma-3-4b",
        "hf_id": "google/gemma-3-4b-it",
        "family": "gemma3",
        "gated": True,
        "description": "Google Gemma 3 4B-IT (Multimodal architecture text model, bfloat16)"
    },
    {
        "name": "llama-3.2-3b",
        "hf_id": "meta-llama/Llama-3.2-3B-Instruct",
        "family": "llama3",
        "gated": True,
        "description": "Meta Llama 3.2 3B-Instruct (Contemporary state-of-the-art baseline)"
    },
]

# Additional available contemporary baselines for scaling studies (e.g. A100 or multi-GPU)
OPTIONAL_LARGE_MODELS = [
    {"name": "llama-3.1-8b", "hf_id": "meta-llama/Llama-3.1-8B-Instruct", "family": "llama3", "gated": True},
    {"name": "qwen2.5-7b", "hf_id": "Qwen/Qwen2.5-7B-Instruct", "family": "causal", "gated": False},
    {"name": "mistral-7b", "hf_id": "mistralai/Mistral-7B-Instruct-v0.3", "family": "causal", "gated": True},
]

# ==============================================================================
# 4. Experimental Settings & Controls (Addressing Recommendation 2)
# ==============================================================================
# Experimental conditions evaluated per prompt:
# - vanilla: no evidence (zero-retrieval baseline)
# - rag_bm25: sparse lexical retrieval (BM25)
# - rag_dense: dense semantic retrieval (BGE / MiniLM embeddings)
# - rag_hybrid: Reciprocal Rank Fusion (BM25 + Dense)
# - control_oracle: perfect ground-truth evidence (isolates parametric vs retrieval sycophancy)
# - control_irrelevant: uninformative distractor passages (evaluates susceptibility to irrelevant context)
SETTINGS = ["vanilla", "rag_bm25", "rag_dense", "rag_hybrid", "control_oracle", "control_irrelevant"]

# Fast core subset for quick runs / resource-constrained environments
CORE_SETTINGS = ["vanilla", "rag_bm25", "rag_dense", "control_oracle"]

# ==============================================================================
# 5. Retrieval Configuration
# ==============================================================================
TOP_K = 5
RETRIEVAL_KS = [1, 3, 5, 10]
PASSAGE_WORDS = 120
MAX_PASSAGES_PER_DOC = 6

# BM25 Parameters
BM25_K1 = 0.9
BM25_B  = 0.4

# Dense Retrieval Model
DENSE_MODEL_ID = "BAAI/bge-small-en-v1.5"
DENSE_FALLBACK_ID = "sentence-transformers/all-MiniLM-L6-v2"

# Hybrid Fusion (Reciprocal Rank Fusion)
RRF_K = 60

# Wikipedia Distractor Stream settings (if streaming full corpus)
WIKI_DATASET  = "wikimedia/wikipedia"
WIKI_CONFIG   = "20231101.en"
N_DISTRACTORS = 50000

# ==============================================================================
# 6. Judge & Evaluation Configuration (Addressing Recommendation 3: Causal)
# ==============================================================================
JUDGE_HF_ID = "Qwen/Qwen2.5-7B-Instruct"

# Causal Conflict Taxonomy:
# - conflict_evidence_aligned: Model refutes user's false belief by adhering to reference evidence
# - conflict_sycophantic: Model capitulates to user's false belief despite contrary reference evidence
# - conflict_hedged: Model refuses to take a side or provides non-committal ambiguous statements
# - conflict_hallucinated: Model fabricates unsupported facts independent of evidence and user prior
CAUSAL_CONFLICT_CATEGORIES = [
    "conflict_evidence_aligned",
    "conflict_sycophantic",
    "conflict_hedged",
    "conflict_hallucinated"
]

# ==============================================================================
# 7. Generation Parameters
# ==============================================================================
MAX_NEW_TOKENS = 200
BATCH_SIZE = 8
LOAD_IN_4BIT = True
