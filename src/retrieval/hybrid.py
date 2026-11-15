"""
Hybrid Retriever combining Sparse (BM25) and Dense retrieval via Reciprocal Rank Fusion (RRF).
Reference: Cormack et al., SIGIR 2009.
"""

from typing import List, Dict, Any
from src.retrieval.base import BaseRetriever
from configs import config as cfg


class HybridRetriever(BaseRetriever):
    """
    Reciprocal Rank Fusion (RRF) hybrid retriever.
    Combines lexical precision with dense semantic understanding.
    """
    def __init__(
        self,
        sparse_retriever: BaseRetriever,
        dense_retriever: BaseRetriever,
        rrf_k: int = cfg.RRF_K,
        candidate_depth: int = 50
    ):
        self.sparse = sparse_retriever
        self.dense = dense_retriever
        self.rrf_k = rrf_k
        self.candidate_depth = candidate_depth

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        # Fetch candidate lists from both retrievers
        sparse_hits = self.sparse.search(query, k=self.candidate_depth)
        dense_hits = self.dense.search(query, k=self.candidate_depth)

        doc_map = {}  # docid -> doc dict
        rrf_scores = {}

        # Accumulate sparse ranks (1-indexed)
        for rank, hit in enumerate(sparse_hits, start=1):
            did = hit["docid"]
            doc_map[did] = hit
            rrf_scores[did] = rrf_scores.get(did, 0.0) + (1.0 / (self.rrf_k + rank))

        # Accumulate dense ranks (1-indexed)
        for rank, hit in enumerate(dense_hits, start=1):
            did = hit["docid"]
            if did not in doc_map:
                doc_map[did] = hit
            rrf_scores[did] = rrf_scores.get(did, 0.0) + (1.0 / (self.rrf_k + rank))

        # Sort by composite RRF score descending
        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:k]

        results = []
        for did, score in sorted_docs:
            d = dict(doc_map[did])
            d["score"] = float(score)
            d["retriever_origin"] = "hybrid_rrf"
            results.append(d)

        return results
