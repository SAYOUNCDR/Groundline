from __future__ import annotations

from collections import defaultdict

from rank_bm25 import BM25Okapi

from .ingest import CorpusChunk, load_corpus
from .schemas import Evidence
from .text_utils import tokens


class BM25Retriever:
    def __init__(self, chunks: list[CorpusChunk]) -> None:
        if not chunks:
            raise ValueError("Cannot build retriever with an empty corpus.")
        self.chunks = chunks
        self._tokenized = [tokens(chunk.text + " " + chunk.title + " " + chunk.heading) for chunk in chunks]
        self._bm25 = BM25Okapi(self._tokenized)

    @classmethod
    def from_data_dir(cls, data_dir):
        return cls(load_corpus(data_dir))

    def search(
        self,
        query: str,
        company: str | None = None,
        product_area: str | None = None,
        top_k: int = 6,
    ) -> list[Evidence]:
        query_tokens = tokens(query)
        if not query_tokens:
            return []

        scores = self._bm25.get_scores(query_tokens)
        ranked: list[tuple[float, int]] = []
        for idx, score in enumerate(scores):
            chunk = self.chunks[idx]
            boosted = float(score)
            if company and company != "None":
                boosted *= 1.35 if chunk.company.lower() == company.lower() else 0.45
            if product_area and chunk.product_area == product_area:
                boosted *= 1.2
            if boosted > 0:
                ranked.append((boosted, idx))

        ranked.sort(reverse=True, key=lambda item: item[0])
        return [
            self.chunks[idx].as_evidence(score=score, method="bm25")
            for score, idx in ranked[:top_k]
        ]


class CitationStore:
    def __init__(self) -> None:
        self._records: dict[int, list[Evidence]] = defaultdict(list)

    def add(self, row_id: int | None, evidence: list[Evidence]) -> None:
        if row_id is not None:
            self._records[row_id] = evidence

    def get(self, row_id: int) -> list[Evidence]:
        return self._records.get(row_id, [])
