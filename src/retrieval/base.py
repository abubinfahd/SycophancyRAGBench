"""
Base retriever interface for SycophancyRAGBench.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any


class BaseRetriever(ABC):
    """Abstract base class for all retrieval engines."""

    @abstractmethod
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve top-k relevant passages for a given query string.
        Returns a list of dicts:
        [
            {
                "docid": str,
                "title": str,
                "contents": str,
                "score": float
            },
            ...
        ]
        """
        pass
