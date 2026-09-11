"""
retriever.py
------------
RAG retriever: given an incoming customer message, finds the top-k most
similar historical Spotify conversations and returns them as context for
the reply drafter.

Uses FAISS flat inner-product search (cosine similarity on normalized vectors).
Falls back to TF-IDF cosine retrieval if FAISS index is not built.
"""

from __future__ import annotations

import json
import numpy as np
from pathlib import Path
from typing import NamedTuple

DATA_DIR = Path(__file__).parent.parent / "data"
VECTORSTORE_DIR = DATA_DIR / "vectorstore"


class RetrievedExample(NamedTuple):
    customer_text: str
    brand_reply: str
    score: float


class Retriever:
    """
    Semantic retriever backed by FAISS + sentence-transformers.

    If the vector store has not been built, falls back to TF-IDF cosine
    similarity over the raw dataset (slower but always works).
    """

    def __init__(self):
        self._model = None
        self._index = None
        self._metadata: list[dict] = []
        self._tfidf_fallback = None
        self._tfidf_texts: list[str] = []
        self._tfidf_replies: list[str] = []

        self._load()

    def _load(self):
        index_path = VECTORSTORE_DIR / "index.faiss"
        meta_path = VECTORSTORE_DIR / "metadata.json"

        if index_path.exists() and meta_path.exists():
            self._load_faiss(index_path, meta_path)
        else:
            print("  Vector store not found — using TF-IDF retrieval fallback.")
            self._load_tfidf_fallback()

    def _load_faiss(self, index_path: Path, meta_path: Path):
        import faiss
        from sentence_transformers import SentenceTransformer

        print("Loading FAISS index...")
        self._index = faiss.read_index(str(index_path))

        with open(meta_path) as f:
            self._metadata = json.load(f)

        print("Loading embedding model for retrieval...")
        self._model = SentenceTransformer("all-MiniLM-L6-v2")
        print(f"  Retriever ready: {self._index.ntotal:,} indexed entries.")

    def _load_tfidf_fallback(self):
        from sklearn.feature_extraction.text import TfidfVectorizer

        csv_path = DATA_DIR / "spotify_subset.csv"
        if not csv_path.exists():
            # Use hard-coded mini corpus as absolute last resort
            self._tfidf_texts = [
                "songs keep pausing randomly",
                "can't log in to my account",
                "charged twice this month",
                "app keeps crashing",
                "downloaded songs disappeared",
                "album was removed from spotify",
                "not working on my ps5",
                "how do i share a playlist",
            ]
            self._tfidf_replies = [
                "Try clearing your cache and restarting the app! DM us if it continues.",
                "Try resetting your password at spoti.fi/reset",
                "So sorry! DM us your account email and charge details.",
                "Please reinstall the app and let us know your device + OS version.",
                "Connect to internet once to refresh offline licenses, then try again.",
                "Labels sometimes remove content — this is outside our control.",
                "Try uninstalling and reinstalling from the PlayStation Store.",
                "Tap the three dots > Share > Copy Link!",
            ]
        else:
            import pandas as pd
            df = pd.read_csv(csv_path).dropna(subset=["customer_text", "brand_reply"])
            self._tfidf_texts = df["customer_text"].astype(str).tolist()
            self._tfidf_replies = df["brand_reply"].astype(str).tolist()

        self._tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        self._tfidf_matrix = self._tfidf_vectorizer.fit_transform(self._tfidf_texts)
        print(f"  TF-IDF fallback retriever ready: {len(self._tfidf_texts)} entries.")

    def retrieve(self, query: str, k: int = 3) -> list[RetrievedExample]:
        """Return top-k most similar historical examples."""
        if self._index is not None:
            return self._retrieve_faiss(query, k)
        return self._retrieve_tfidf(query, k)

    def _retrieve_faiss(self, query: str, k: int) -> list[RetrievedExample]:
        vec = self._model.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        ).astype("float32")
        k = min(k, self._index.ntotal)
        scores, indices = self._index.search(vec, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            meta = self._metadata[idx]
            results.append(RetrievedExample(
                customer_text=meta["customer_text"],
                brand_reply=meta["brand_reply"],
                score=float(score),
            ))
        return results

    def _retrieve_tfidf(self, query: str, k: int) -> list[RetrievedExample]:
        from sklearn.metrics.pairwise import cosine_similarity
        qvec = self._tfidf_vectorizer.transform([query])
        sims = cosine_similarity(qvec, self._tfidf_matrix)[0]
        top_k = np.argsort(sims)[::-1][:k]
        return [
            RetrievedExample(
                customer_text=self._tfidf_texts[i],
                brand_reply=self._tfidf_replies[i],
                score=float(sims[i]),
            )
            for i in top_k
        ]


class TrivialRetriever:
    """Returns a fixed canned reply regardless of query (trivial baseline)."""

    def retrieve(self, query: str, k: int = 3) -> list[RetrievedExample]:
        return [RetrievedExample(
            customer_text="any customer message",
            brand_reply="Please DM us so we can help you further.",
            score=1.0,
        )]


if __name__ == "__main__":
    ret = Retriever()
    results = ret.retrieve("my songs keep stopping randomly")
    print("Query: 'my songs keep stopping randomly'")
    for r in results:
        print(f"  [{r.score:.3f}] Q: {r.customer_text[:60]}")
        print(f"          A: {r.brand_reply[:80]}")
