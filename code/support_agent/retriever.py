from __future__ import annotations

from collections import Counter
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from fastembed import TextEmbedding
from qdrant_client import QdrantClient, models
from rank_bm25 import BM25Okapi

from .config import Settings
from .ingest import CorpusChunk, load_corpus
from .schemas import Evidence
from .text_utils import contains_any, tokens


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


class QdrantSemanticRetriever:
    def __init__(
        self,
        chunks: list[CorpusChunk],
        url: str,
        collection_name: str,
        model_name: str,
    ) -> None:
        self.chunks = chunks
        self.collection_name = collection_name
        self.model_name = model_name
        self.client = QdrantClient(url=url, timeout=60)
        self.embedding_model: TextEmbedding | None = None
        self.chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}

    def is_ready(self) -> bool:
        try:
            if not self.client.collection_exists(self.collection_name):
                return False
            count = self.client.count(self.collection_name, exact=True).count
            return count >= len(self.chunks)
        except Exception:
            return False

    def build_index(self, recreate: bool = False, batch_size: int = 64) -> int:
        model = self._model()
        sample_vector = next(model.embed([self._document_text(self.chunks[0])]))
        vector_size = int(len(sample_vector))

        if recreate and self.client.collection_exists(self.collection_name):
            self.client.delete_collection(self.collection_name)

        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
            )
        elif self.is_ready() and not recreate:
            return len(self.chunks)

        for start in range(0, len(self.chunks), batch_size):
            batch = self.chunks[start : start + batch_size]
            vectors = list(model.embed([self._document_text(chunk) for chunk in batch]))
            points = [
                models.PointStruct(
                    id=start + offset,
                    vector=np.asarray(vector, dtype=np.float32).tolist(),
                    payload=self._payload(chunk),
                )
                for offset, (chunk, vector) in enumerate(zip(batch, vectors, strict=True))
            ]
            self.client.upsert(collection_name=self.collection_name, points=points, wait=True)
        return len(self.chunks)

    def search(
        self,
        query: str,
        company: str | None = None,
        product_area: str | None = None,
        top_k: int = 12,
    ) -> list[Evidence]:
        if not self.is_ready():
            self.build_index(recreate=False)

        vector = next(self._model().embed([self._query_text(query)]))
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=np.asarray(vector, dtype=np.float32).tolist(),
            query_filter=self._filter(company, product_area),
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )

        evidence: list[Evidence] = []
        for point in response.points:
            payload = point.payload or {}
            chunk = self.chunk_by_id.get(str(payload.get("chunk_id", "")))
            if chunk is None:
                chunk = self._chunk_from_payload(payload)
            evidence.append(chunk.as_evidence(score=float(point.score), method="qdrant"))
        return evidence

    def _model(self) -> TextEmbedding:
        if self.embedding_model is None:
            self.embedding_model = TextEmbedding(model_name=self.model_name)
        return self.embedding_model

    def _document_text(self, chunk: CorpusChunk) -> str:
        text = f"{chunk.title}\n{chunk.heading}\n{chunk.text}"
        if "bge" in self.model_name.lower():
            return f"passage: {text}"
        return text

    def _query_text(self, query: str) -> str:
        if "bge" in self.model_name.lower():
            return f"query: {query}"
        return query

    def _filter(self, company: str | None, product_area: str | None) -> models.Filter | None:
        must: list[models.FieldCondition] = []
        if company and company != "None":
            must.append(models.FieldCondition(key="company", match=models.MatchValue(value=company)))
        return models.Filter(must=must) if must else None

    def _payload(self, chunk: CorpusChunk) -> dict[str, Any]:
        return {
            "chunk_id": chunk.chunk_id,
            "company": chunk.company,
            "product_area": chunk.product_area,
            "title": chunk.title,
            "heading": chunk.heading,
            "source_path": str(chunk.source_path),
            "text": chunk.text,
        }

    def _chunk_from_payload(self, payload: dict[str, Any]) -> CorpusChunk:
        return CorpusChunk(
            chunk_id=str(payload.get("chunk_id", "")),
            company=str(payload.get("company", "")),
            product_area=str(payload.get("product_area", "")),
            title=str(payload.get("title", "")),
            heading=str(payload.get("heading", "")),
            source_path=Path(str(payload.get("source_path", ""))),
            text=str(payload.get("text", "")),
        )


class HybridRetriever:
    def __init__(
        self,
        chunks: list[CorpusChunk],
        semantic: QdrantSemanticRetriever | None,
    ) -> None:
        self.chunks = chunks
        self.bm25 = BM25Retriever(chunks)
        self.semantic = semantic

    @classmethod
    def from_settings(cls, settings: Settings) -> "HybridRetriever":
        chunks = load_corpus(settings.data_dir)
        semantic = None
        if settings.vector_backend.lower() == "qdrant":
            try:
                semantic = QdrantSemanticRetriever(
                    chunks=chunks,
                    url=settings.qdrant_url,
                    collection_name=settings.qdrant_collection,
                    model_name=settings.embedding_model,
                )
            except Exception:
                semantic = None
        return cls(chunks=chunks, semantic=semantic)

    def build_index(self, recreate: bool = False) -> int:
        if self.semantic is None:
            return 0
        return self.semantic.build_index(recreate=recreate)

    def search(
        self,
        query: str,
        company: str | None = None,
        product_area: str | None = None,
        top_k: int = 6,
    ) -> list[Evidence]:
        expanded_query = expand_query(query)
        bm25_hits = self.bm25.search(expanded_query, company=company, product_area=product_area, top_k=16)
        semantic_hits: list[Evidence] = []
        if self.semantic is not None:
            try:
                semantic_hits = self.semantic.search(expanded_query, company=company, product_area=product_area, top_k=16)
            except Exception:
                semantic_hits = []
        return reciprocal_rank_fusion(bm25_hits, semantic_hits, company=company, product_area=product_area)[:top_k]


def reciprocal_rank_fusion(
    bm25_hits: list[Evidence],
    semantic_hits: list[Evidence],
    company: str | None = None,
    product_area: str | None = None,
    k: int = 60,
) -> list[Evidence]:
    scores: dict[str, float] = defaultdict(float)
    methods: dict[str, set[str]] = defaultdict(set)
    records: dict[str, Evidence] = {}

    for weight, hits in [(1.0, bm25_hits), (1.15, semantic_hits)]:
        for rank, evidence in enumerate(hits, start=1):
            key = evidence.chunk_id
            scores[key] += weight / (k + rank)
            methods[key].add(evidence.method)
            records.setdefault(key, evidence)

    for key, evidence in records.items():
        if company and company != "None" and evidence.company.lower() == company.lower():
            scores[key] *= 1.2
        if product_area and evidence.product_area == product_area:
            scores[key] *= 1.15

    ranked = sorted(records.values(), key=lambda item: scores[item.chunk_id], reverse=True)
    product_counts = Counter(item.product_area for item in ranked[:8] if item.product_area)
    for item in ranked:
        item.score = scores[item.chunk_id]
        item.method = "+".join(sorted(methods[item.chunk_id])) or item.method
        if product_counts and item.product_area == product_counts.most_common(1)[0][0]:
            item.score *= 1.05
    return sorted(ranked, key=lambda item: item.score, reverse=True)


def expand_query(query: str) -> str:
    lowered = query.lower()
    expansions: list[str] = []
    if contains_any(lowered, ["remove an interviewer", "employee has left", "remove them from", "remove a user"]):
        expansions.append("manage team members deactivate user lock user access remove user team member status")
    if contains_any(lowered, ["compatibility check", "zoom connectivity", "unable to take the test", "criteria"]):
        expansions.append("system compatibility check websocket connectivity zoom audio video calls proctor mode")
    if contains_any(lowered, ["infosec", "security process", "security forms"]):
        expansions.append("security compliance trust data protection audit support")
    if contains_any(lowered, ["model improvement", "improve the models", "how long will the data be used"]):
        expansions.append("model training data retention privacy settings data used improve Claude")
    if contains_any(lowered, ["bedrock", "aws", "all requests"]):
        expansions.append("Amazon Bedrock AWS support customer support inquiries requests failing")
    return " ".join([query, *expansions]).strip()


class CitationStore:
    def __init__(self) -> None:
        self._records: dict[int, list[Evidence]] = defaultdict(list)

    def add(self, row_id: int | None, evidence: list[Evidence]) -> None:
        if row_id is not None:
            self._records[row_id] = evidence

    def get(self, row_id: int) -> list[Evidence]:
        return self._records.get(row_id, [])
