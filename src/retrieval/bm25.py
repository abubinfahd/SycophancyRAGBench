"""
Sparse Lexical BM25 Retriever.
Provides a pure Python/NumPy BM25Okapi engine with zero external Java dependencies,
and optional Pyserini Lucene backend if Java 21 is available.
"""

import os
import re
import math
import json
from typing import List, Dict, Any, Optional
from collections import Counter
import numpy as np

from configs import config as cfg
from src.retrieval.base import BaseRetriever
from src import utils as U


def _tokenize(text: str) -> List[str]:
    """Simple, fast lowercase alphanumeric tokenizer for BM25."""
    return re.findall(r"\b[a-z0-9]+\b", text.lower())


class PureBM25(BaseRetriever):
    """
    High-performance in-memory BM25Okapi implementation.
    Operates without requiring JVM/Java or C++ compilation.
    """
    def __init__(self, corpus: List[Dict[str, Any]], k1: float = 0.9, b: float = 0.4):
        self.k1 = k1
        self.b = b
        self.corpus = corpus
        self.doc_lens = []
        self.doc_freqs = Counter()
        self.inverted_index = {}  # term -> list of (doc_idx, term_freq)
        self.N = len(corpus)
        self.avg_dl = 0.0

        self._build_index()

    def _build_index(self):
        print(f"[BM25] Indexing {self.N} documents into in-memory BM25...")
        total_len = 0
        for idx, doc in enumerate(self.corpus):
            tokens = _tokenize(f"{doc.get('title', '')} {doc.get('contents', '')}")
            doc_len = len(tokens)
            self.doc_lens.append(doc_len)
            total_len += doc_len

            tf = Counter(tokens)
            for term, count in tf.items():
                self.doc_freqs[term] += 1
                if term not in self.inverted_index:
                    self.inverted_index[term] = []
                self.inverted_index[term].append((idx, count))

        self.avg_dl = (total_len / self.N) if self.N > 0 else 1.0

        # Precompute IDF for all terms: ln( (N - df + 0.5) / (df + 0.5) + 1 )
        self.idf = {}
        for term, df in self.doc_freqs.items():
            self.idf[term] = math.log(((self.N - df + 0.5) / (df + 0.5)) + 1.0)
        print(f"[BM25] Index built. Vocab size: {len(self.idf):,} terms, Avg doc length: {self.avg_dl:.1f}")

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        tokens = _tokenize(query)
        scores = {}

        for term in tokens:
            if term not in self.inverted_index:
                continue
            term_idf = self.idf[term]
            postings = self.inverted_index[term]
            for doc_idx, tf in postings:
                dl = self.doc_lens[doc_idx]
                denom = tf + self.k1 * (1.0 - self.b + self.b * (dl / self.avg_dl))
                score = term_idf * ((tf * (self.k1 + 1.0)) / denom)
                scores[doc_idx] = scores.get(doc_idx, 0.0) + score

        if not scores:
            # Fallback if no lexical overlap
            top_indices = list(range(min(k, self.N)))
            return [{
                "docid": self.corpus[i].get("id", f"doc_{i}"),
                "title": self.corpus[i].get("title", ""),
                "contents": self.corpus[i].get("contents", ""),
                "score": 0.0
            } for i in top_indices]

        # Get top-k highest scores
        top_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
        results = []
        for doc_idx, score in top_items:
            doc = self.corpus[doc_idx]
            results.append({
                "docid": doc.get("id", f"doc_{doc_idx}"),
                "title": doc.get("title", ""),
                "contents": doc.get("contents", ""),
                "score": float(score)
            })
        return results


class PyseriniLuceneBM25(BaseRetriever):
    """Pyserini Lucene BM25 backend for environments with Java 21 configured."""
    def __init__(self, index_dir: str, k1: float = 0.9, b: float = 0.4):
        from pyserini.search.lucene import LuceneSearcher
        self.searcher = LuceneSearcher(index_dir)
        self.searcher.set_bm25(k1, b)

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        hits = self.searcher.search(query, k=k)
        results = []
        for h in hits:
            raw = getattr(h, "raw", None)
            if raw is None:
                doc = self.searcher.doc(h.docid)
                raw = doc.raw() if doc else None
            data = json.loads(raw) if raw else {}
            results.append({
                "docid": h.docid,
                "title": data.get("title", ""),
                "contents": data.get("contents", ""),
                "score": float(h.score)
            })
        return results


def build_bm25_retriever(
    gold_path: str = None,
    distractors_path: str = None,
    prefer_lucene: bool = False
) -> BaseRetriever:
    """Build and return an initialized BM25 retriever."""
    gold_p = gold_path or cfg.GOLD_PASSAGES_PATH
    dis_p = distractors_path or cfg.DISTRACTORS_PATH

    # Check for Pyserini Lucene index if requested
    if prefer_lucene and os.path.exists(cfg.INDEX_DIR):
        try:
            return PyseriniLuceneBM25(cfg.INDEX_DIR, cfg.BM25_K1, cfg.BM25_B)
        except Exception as e:
            print(f"[BM25] Could not load Lucene index ({e}). Falling back to in-memory PureBM25.")

    # In-memory BM25 corpus loader
    corpus = []
    if os.path.exists(gold_p):
        corpus.extend(U.read_jsonl(gold_p))
    if os.path.exists(dis_p):
        corpus.extend(U.read_jsonl(dis_p))

    if not corpus:
        # Fallback to bundled files if empty
        if os.path.exists(cfg.BUNDLED_GOLD):
            corpus.extend(U.read_jsonl(cfg.BUNDLED_GOLD))
        if os.path.exists(cfg.BUNDLED_DISTRACTORS):
            corpus.extend(U.read_jsonl(cfg.BUNDLED_DISTRACTORS))

    return PureBM25(corpus, k1=cfg.BM25_K1, b=cfg.BM25_B)
