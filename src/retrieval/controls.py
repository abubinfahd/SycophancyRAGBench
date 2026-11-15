"""
Experimental Control Retrievers:
1. Oracle Evidence Control: Direct ground-truth passage injection (upper bound).
2. Irrelevant Evidence Control: Random non-relevant distractor injection (noise floor).
"""

import random
from typing import List, Dict, Any, Optional
from configs import config as cfg
from src.retrieval.base import BaseRetriever
from src import utils as U


class OracleRetriever(BaseRetriever):
    """
    Oracle Evidence Control:
    Bypasses retriever search and directly yields verified gold Wikipedia passages.
    Isolates parametric alignment sycophancy from retrieval recall failures.
    """
    def __init__(self, gold_passages: List[Dict[str, Any]]):
        self.gold_passages = gold_passages
        # Title index: norm_title -> list of passages
        self.by_title = {}
        for p in gold_passages:
            nt = U.norm_title(p.get("title", ""))
            if nt not in self.by_title:
                self.by_title[nt] = []
            self.by_title[nt].append(p)

    def search_for_claim(self, claim: Dict[str, Any], k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve gold passages explicitly matched to claim's gold titles."""
        gold_titles = [U.norm_title(t) for t in claim.get("gold_titles", [])]
        hits = []

        for gt in gold_titles:
            if gt in self.by_title:
                for p in self.by_title[gt]:
                    hits.append({
                        "docid": p.get("id", f"gold_{len(hits)}"),
                        "title": p.get("title", ""),
                        "contents": p.get("contents", ""),
                        "score": 1.0,
                        "control_type": "oracle"
                    })
                    if len(hits) >= k:
                        break
            if len(hits) >= k:
                break

        # Fallback for TruthfulQA (which may not have a Wikipedia article title)
        if not hits:
            # Provide the verified true refutation as a reference passage
            hits.append({
                "docid": f"oracle_{claim.get('claim_id')}",
                "title": "Verified Fact",
                "contents": claim.get("true_answer", ""),
                "score": 1.0,
                "control_type": "oracle"
            })

        return hits

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        # Query-based search fallback (first k gold passages)
        return [{
            "docid": p.get("id", f"gold_{i}"),
            "title": p.get("title", ""),
            "contents": p.get("contents", ""),
            "score": 1.0,
            "control_type": "oracle"
        } for i, p in enumerate(self.gold_passages[:k])]


class IrrelevantRetriever(BaseRetriever):
    """
    Irrelevant Evidence Control:
    Supplies random off-topic distractor passages unlinked to the query.
    Evaluates whether unhelpful or unrelated context induces model confusion or sycophancy.
    """
    def __init__(self, distractor_passages: List[Dict[str, Any]], seed: int = 42):
        self.distractors = distractor_passages
        self.rng = random.Random(seed)

    def search_for_claim(self, claim: Dict[str, Any], k: int = 5) -> List[Dict[str, Any]]:
        return self.search(claim.get("retrieval_query", ""), k=k)

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        if not self.distractors:
            return []
        sampled = self.rng.sample(self.distractors, min(k, len(self.distractors)))
        return [{
            "docid": d.get("id", f"irr_{i}"),
            "title": d.get("title", "Unrelated Context"),
            "contents": d.get("contents", ""),
            "score": 0.0,
            "control_type": "irrelevant"
        } for i, d in enumerate(sampled)]
