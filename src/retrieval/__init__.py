"""
Retrieval subpackage for SycophancyRAGBench.
"""

from .base import BaseRetriever
from .bm25 import PureBM25, build_bm25_retriever
from .controls import OracleRetriever, IrrelevantRetriever

try:
    from .dense import DenseRetriever, build_dense_retriever
    from .hybrid import HybridRetriever
except (ImportError, ModuleNotFoundError):
    DenseRetriever = None
    build_dense_retriever = None
    HybridRetriever = None

from .manager import run_retrieval_pipeline, evaluate_retrieval_quality

__all__ = [
    "BaseRetriever",
    "PureBM25",
    "build_bm25_retriever",
    "DenseRetriever",
    "build_dense_retriever",
    "HybridRetriever",
    "OracleRetriever",
    "IrrelevantRetriever",
    "run_retrieval_pipeline",
    "evaluate_retrieval_quality",
]
