from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


class EmbeddingIndex:
    """Local Chroma persistent collection using cosine space."""

    def __init__(self, persist_dir: str | Path, collection_name: str = "face_embeddings") -> None:
        import chromadb

        self._client = chromadb.PersistentClient(path=str(persist_dir))
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert(self, identity: str, embedding: np.ndarray, sample_id: str | None = None) -> None:
        emb = np.asarray(embedding, dtype=np.float32).reshape(-1).tolist()
        sid = sample_id or f"{identity}:{abs(hash(embedding.tobytes())) % 10**9}"
        self._collection.upsert(ids=[sid], embeddings=[emb], metadatas=[{"identity": identity}])

    def query(self, embedding: np.ndarray, top_k: int = 1) -> dict[str, Any]:
        if self._collection.count() == 0:
            return {"identity": None, "similarity": None}
        emb = np.asarray(embedding, dtype=np.float32).reshape(1, -1).tolist()
        res = self._collection.query(query_embeddings=emb, n_results=max(1, top_k), include=["distances", "metadatas"])
        best_identity = None
        best_sim = None
        if res["metadatas"] and res["metadatas"][0]:
            md = res["metadatas"][0][0]
            dist = res["distances"][0][0] if res["distances"] else None
            best_identity = md.get("identity") if md else None
            best_sim = None if dist is None else float(1.0 - dist)
        return {"identity": best_identity, "similarity": best_sim}
