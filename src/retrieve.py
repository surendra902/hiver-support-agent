import os
import pickle
import logging
from typing import List, Optional
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.schemas import HistoricalResolution

logger = logging.getLogger(__name__)


class RetrievalIndex:
    """
    Search index over historically resolved customer support (query, reply) pairs.
    Uses TF-IDF with sublinear term-frequency scaling and word n-grams for fast,
    deterministic, zero-cost CPU retrieval.
    """

    def __init__(self, top_k: int = 5, min_threshold: float = 0.25):
        self.top_k = top_k
        self.min_threshold = min_threshold
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=25000,
            sublinear_tf=True,
            stop_words="english"
        )
        self.fitted = False
        self.corpus_df: Optional[pd.DataFrame] = None
        self.tfidf_matrix = None

    def build_index(self, df: pd.DataFrame) -> None:
        """Fit index on customer queries from resolved historical pairs."""
        logger.info(f"Building retrieval index on {len(df)} historical pairs...")
        self.corpus_df = df.reset_index(drop=True).copy()
        queries = self.corpus_df["customer_text"].fillna("").tolist()
        self.tfidf_matrix = self.vectorizer.fit_transform(queries)
        self.fitted = True
        logger.info("Retrieval index successfully built.")

    def search(self, query: str, k: Optional[int] = None, exclude_deflections: bool = False) -> List[HistoricalResolution]:
        """Query index and return top-k most similar historical resolutions."""
        if not self.fitted or self.corpus_df is None:
            raise ValueError("Index has not been built. Call build_index first.")

        k = k or self.top_k
        query_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self.tfidf_matrix)[0]

        # Get sorted candidate indices
        sorted_indices = np.argsort(sims)[::-1]

        results = []
        for idx in sorted_indices:
            score = float(sims[idx])
            if score < self.min_threshold and len(results) > 0:
                break

            row = self.corpus_df.iloc[idx]
            is_deflection = bool(row.get("is_deflection", False))
            
            if exclude_deflections and is_deflection:
                continue

            resolution = HistoricalResolution(
                tweet_id=str(row["customer_tweet_id"]),
                customer_query=str(row["customer_text"]),
                brand_reply=str(row["brand_reply"]),
                similarity=round(score, 4),
                is_deflection=is_deflection
            )
            results.append(resolution)

            if len(results) >= k:
                break

        return results

    def save(self, filepath: str = "data/index/retrieval_index.pkl") -> None:
        """Persist index to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"Retrieval index saved to {filepath}")

    @classmethod
    def load(cls, filepath: str = "data/index/retrieval_index.pkl") -> "RetrievalIndex":
        """Load persisted index from disk."""
        with open(filepath, "rb") as f:
            index = pickle.load(f)
        logger.info(f"Retrieval index loaded from {filepath}")
        return index
