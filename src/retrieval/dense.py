"""
Dense Semantic Retriever.
Embeds passages and queries using lightweight dense embedding models
(BAAI/bge-small-en-v1.5 or sentence-transformers/all-MiniLM-L6-v2),
computing cosine similarities via PyTorch matrix operations.
"""

import os
import json
from typing import List, Dict, Any, Optional
import numpy as np

try:
    import torch
except (ImportError, ModuleNotFoundError):
    torch = None

from configs import config as cfg
from src.retrieval.base import BaseRetriever
from src import utils as U


class DenseRetriever(BaseRetriever):
    """
    Dense semantic retriever supporting GPU acceleration and embedding caching.
    """
    def __init__(
        self,
        corpus: List[Dict[str, Any]],
        model_name: str = cfg.DENSE_MODEL_ID,
        cache_path: Optional[str] = None,
        batch_size: int = 64
    ):
        if torch is None:
            raise ImportError("PyTorch is required for DenseRetriever. Please install torch.")

        self.corpus = corpus
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.cache_path = cache_path or os.path.join(cfg.CORPUS_DIR, "dense_embeddings.pt")

        self.model = None
        self.embeddings = None

        self._init_model()
        self._build_or_load_embeddings()

    def _init_model(self):
        print(f"[Dense] Loading embedding model: {self.model_name} on {self.device}...")
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name, device=self.device)
        except Exception as e:
            print(f"[Dense] sentence-transformers load failed ({e}), attempting fallback to {cfg.DENSE_FALLBACK_ID}...")
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(cfg.DENSE_FALLBACK_ID, device=self.device)

    def _build_or_load_embeddings(self):
        if os.path.exists(self.cache_path):
            try:
                cached = torch.load(self.cache_path, map_location=self.device)
                if cached.shape[0] == len(self.corpus):
                    print(f"[Dense] Loaded cached embeddings ({cached.shape}) from {self.cache_path}")
                    self.embeddings = cached
                    return
            except Exception as e:
                print(f"[Dense] Failed loading cache: {e}. Recomputing...")

        print(f"[Dense] Encoding {len(self.corpus)} passages (batch_size={self.batch_size})...")
        texts = [
            f"{doc.get('title', '')}: {doc.get('contents', '')}".strip()
            for doc in self.corpus
        ]

        with torch.no_grad():
            emb = self.model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=True,
                convert_to_tensor=True,
                normalize_embeddings=True,
                device=self.device
            )

        self.embeddings = emb
        try:
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            torch.save(emb.cpu(), self.cache_path)
            print(f"[Dense] Cached embeddings -> {self.cache_path}")
        except Exception as e:
            print(f"[Dense] Warning: Could not cache embeddings to disk: {e}")

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        with torch.no_grad():
            q_emb = self.model.encode(
                query,
                convert_to_tensor=True,
                normalize_embeddings=True,
                device=self.device
            )
            scores = torch.mv(self.embeddings, q_emb)
            topk_scores, topk_indices = torch.topk(scores, k=min(k, len(self.corpus)))

        topk_scores = topk_scores.cpu().tolist()
        topk_indices = topk_indices.cpu().tolist()

        results = []
        for idx, score in zip(topk_indices, topk_scores):
            doc = self.corpus[idx]
            results.append({
                "docid": doc.get("id", f"doc_{idx}"),
                "title": doc.get("title", ""),
                "contents": doc.get("contents", ""),
                "score": float(score)
            })
        return results


def build_dense_retriever(
    gold_path: str = None,
    distractors_path: str = None
) -> BaseRetriever:
    """Build and return an initialized DenseRetriever."""
    gold_p = gold_path or cfg.GOLD_PASSAGES_PATH
    dis_p = distractors_path or cfg.DISTRACTORS_PATH

    corpus = []
    if os.path.exists(gold_p):
        corpus.extend(U.read_jsonl(gold_p))
    if os.path.exists(dis_p):
        corpus.extend(U.read_jsonl(dis_p))

    if not corpus:
        if os.path.exists(cfg.BUNDLED_GOLD):
            corpus.extend(U.read_jsonl(cfg.BUNDLED_GOLD))
        if os.path.exists(cfg.BUNDLED_DISTRACTORS):
            corpus.extend(U.read_jsonl(cfg.BUNDLED_DISTRACTORS))

    return DenseRetriever(corpus)
