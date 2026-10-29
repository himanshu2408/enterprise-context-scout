"""Pure-Python Okapi BM25 implementation for in-memory lexical retrieval (M3).

Provides deterministic lexical ranking without external untyped packages.
"""

import math
import re
from collections import Counter
from typing import Final

_TOKEN_PATTERN: Final[re.Pattern[str]] = re.compile(r"\w+", re.UNICODE)


def tokenize_bm25(text: str) -> list[str]:
    """Extracts lowercase word tokens for BM25 indexing and scoring."""
    return _TOKEN_PATTERN.findall(text.lower())


class BM25Okapi:
    """Okapi BM25 scoring algorithm with Lucene-style non-negative IDF."""

    def __init__(
        self,
        corpus: list[str],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if k1 < 0.0 or b < 0.0 or b > 1.0:
            raise ValueError(f"Invalid BM25 hyperparameters: k1={k1}, b={b}")

        self.k1 = k1
        self.b = b
        self.doc_count = len(corpus)

        self._tokenized_corpus: list[list[str]] = [tokenize_bm25(doc) for doc in corpus]
        self._doc_lengths: list[int] = [len(doc) for doc in self._tokenized_corpus]
        self.avg_doc_length: float = (
            sum(self._doc_lengths) / self.doc_count if self.doc_count > 0 else 0.0
        )

        self._term_frequencies: list[Counter[str]] = [
            Counter(doc) for doc in self._tokenized_corpus
        ]

        # Document frequencies
        df: Counter[str] = Counter()
        for doc_terms in self._term_frequencies:
            for term in doc_terms:
                df[term] += 1

        # Lucene-style smoothed non-negative IDF: ln(1 + (N - df + 0.5) / (df + 0.5))
        self.idf: dict[str, float] = {}
        for term, freq in df.items():
            self.idf[term] = math.log(1.0 + (self.doc_count - freq + 0.5) / (freq + 0.5))

    def get_scores(self, query: str) -> list[float]:
        """Calculates BM25 scores for all corpus documents against query."""
        if self.doc_count == 0:
            return []

        query_tokens = tokenize_bm25(query)
        if not query_tokens:
            return [0.0] * self.doc_count

        scores: list[float] = [0.0] * self.doc_count
        for token in query_tokens:
            token_idf = self.idf.get(token)
            if token_idf is None or token_idf <= 0.0:
                continue

            for doc_idx, term_counts in enumerate(self._term_frequencies):
                tf = term_counts.get(token, 0)
                if tf == 0:
                    continue

                doc_len = self._doc_lengths[doc_idx]
                denom = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avg_doc_length))
                scores[doc_idx] += token_idf * ((tf * (self.k1 + 1.0)) / denom)

        return [round(s, 6) for s in scores]

    def rank_indices(self, query: str) -> list[int]:
        """Returns document indices sorted by descending BM25 score."""
        scores = self.get_scores(query)
        # Stable sort preserving original order for ties
        return sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)


__all__ = ["BM25Okapi", "tokenize_bm25"]
