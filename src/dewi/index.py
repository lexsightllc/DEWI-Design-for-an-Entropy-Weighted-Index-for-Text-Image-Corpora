import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union, List

import numpy as np
import logging

from .types import Payload
from .backends import (
    BaseIndex,
    ExactIndex,
    HNSWIndex,
    FAISSIndex,
    IndexBackend,
    _HAS_FAISS,
    _HAS_HNSW,
)

logger = logging.getLogger(__name__)


class DewiIndex(BaseIndex):
    def __init__(
        self,
        dim: int,
        space: str = "cosine",
        backend: Union[str, IndexBackend] = "auto",
        ef: int = 200,
        M: int = 32,
        use_ann: bool = True,
        ef_query: int = 200,
        rerank_eta: float = 0.25,
        entropy_pref: float = 0.0,
        **kwargs: Any,
    ):
        super().__init__(dim, space)
        self._meta: Dict[str, Dict[str, Any]] = {}
        self.ef_query = ef_query
        self.rerank_eta = float(rerank_eta)
        self.entropy_pref = float(entropy_pref)
        self._built = False
        self._use_ann = bool(use_ann)

        if isinstance(backend, str):
            try:
                backend = IndexBackend.from_str(backend)
            except KeyError:
                backend = IndexBackend.EXACT

        if not self._use_ann:
            self._backend: BaseIndex = ExactIndex(dim, space, **kwargs)
        else:
            if backend in (IndexBackend.FAISS_IVFFLAT, IndexBackend.FAISS_HNSW) and _HAS_FAISS:
                index_type = "IVFFlat" if backend == IndexBackend.FAISS_IVFFLAT else "HNSW"
                self._backend = FAISSIndex(dim, space, index_type=index_type, ef=ef, M=M, **kwargs)
            elif backend == IndexBackend.HNSW and _HAS_HNSW:
                self._backend = HNSWIndex(dim, space, M=M, ef_construction=ef, ef_query=ef_query, **kwargs)
            else:
                logger.warning("ANN backend unavailable; falling back to ExactIndex.")
                self._backend = ExactIndex(dim, space, **kwargs)

    def add(
        self,
        doc_id: str,
        embedding: np.ndarray,
        payload: Payload,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        if meta is not None:
            self._meta[doc_id] = meta
        self._backend.add(doc_id, np.asarray(embedding, dtype=np.float32), payload)

    def build(self) -> None:
        self._backend.build()
        self._built = True

    def search(
        self,
        query: np.ndarray,
        k: int = 10,
        eta: Optional[float] = None,
        entropy_pref: Optional[float] = None,
    ) -> List[Tuple[str, float, Payload]]:
        if not self._built:
            self.build()
        if eta is None:
            eta = self.rerank_eta
        if entropy_pref is None:
            entropy_pref = self.entropy_pref
        q = np.asarray(query, dtype=np.float32)
        if q.shape != (self.dim,):
            raise ValueError(f"Expected query shape ({self.dim},), got {q.shape}")
        return self._backend.search(q, k, eta, entropy_pref)

    def __len__(self) -> int:
        return len(self._backend._doc_ids)

    def get_payload(self, doc_id: str) -> Optional[Payload]:
        return self._backend._payloads.get(doc_id)

    def get_embedding(self, doc_id: str) -> Optional[np.ndarray]:
        emb_store = getattr(self._backend, "_embeddings", None)
        if emb_store is None:
            return None
        try:
            idx = self._backend._doc_ids.index(doc_id)
        except ValueError:
            return None
        try:
            if isinstance(emb_store, list):
                return emb_store[idx]
            if isinstance(emb_store, np.ndarray):
                return emb_store[idx]
        except Exception:
            return None
        return None

    def get_metadata(self, doc_id: str) -> Optional[Dict[str, Any]]:
        return self._meta.get(doc_id)

    def save(self, path: Union[str, Path]) -> None:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        self._backend.save(p / "ann_index")
        with open(p / "config.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "dim": self.dim,
                    "space": self.space,
                    "use_ann": self._use_ann,
                    "ef_query": self.ef_query,
                    "rerank_eta": self.rerank_eta,
                    "entropy_pref": self.entropy_pref,
                    "built": self._built,
                    "backend_type": self._backend.__class__.__name__,
                },
                f,
            )
        if self._meta:
            with open(p / "meta.json", "w", encoding="utf-8") as f:
                json.dump(self._meta, f)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "DewiIndex":
        p = Path(path)
        with open(p / "config.json", "r", encoding="utf-8") as f:
            cfg = json.load(f)
        backend_type = cfg.get("backend_type", "ExactIndex")
        ann_cls = getattr(__import__("dewi.backends", fromlist=[backend_type]), backend_type, ExactIndex)
        ann = ann_cls.load(p / "ann_index")
        inst = cls(
            dim=cfg["dim"],
            space=cfg["space"],
            backend=backend_type,
            use_ann=cfg.get("use_ann", True),
            ef_query=cfg.get("ef_query", 200),
            rerank_eta=cfg.get("rerank_eta", 0.25),
            entropy_pref=cfg.get("entropy_pref", 0.0),
        )
        inst._backend = ann
        inst._built = cfg.get("built", False)
        meta_path = p / "meta.json"
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                inst._meta = json.load(f)
        return inst


__all__ = [
    "DewiIndex",
    "BaseIndex",
    "ExactIndex",
    "HNSWIndex",
    "FAISSIndex",
    "IndexBackend",
    "_HAS_FAISS",
    "_HAS_HNSW",
    "Payload",
]
