"""Indexing and retrieval for DEWI with support for multiple backends (FAISS/HNSW)."""

from __future__ import annotations
import json
import os
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Type, TypeVar
import numpy as np
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Try to import backends
try:
    import hnswlib
    _HAS_HNSW = True
except ImportError:
    _HAS_HNSW = False
    logger.warning("HNSW not available. Install with 'pip install hnswlib' for HNSW support.")

try:
    import faiss
    _HAS_FAISS = True
except ImportError:
    _HAS_FAISS = False
    logger.warning("FAISS not available. Install with 'pip install faiss-cpu' or 'faiss-gpu'.")

class IndexBackend(Enum):
    """Supported index backends."""
    HNSW = auto()
    FAISS_IVFFLAT = auto()
    FAISS_HNSW = auto()
    EXACT = auto()

    @classmethod
    def from_str(cls, name: str) -> 'IndexBackend':
        """Create from string representation."""
        name = name.upper()
        if name == 'AUTO':
            if _HAS_FAISS:
                return cls.FAISS_IVFFLAT
            elif _HAS_HNSW:
                return cls.HNSW
            return cls.EXACT
        return cls[name]

T = TypeVar('T', bound='BaseIndex')

@dataclass
class Payload:
    """Container for document metadata and scores."""
    dewi: float = 0.0
    ht_mean: float = 0.0
    ht_q90: float = 0.0
    hi_mean: float = 0.0
    hi_q90: float = 0.0
    I_hat: float = 0.0
    redundancy: float = 0.0
    noise: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        """Convert payload to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> 'Payload':
        """Create Payload from dictionary."""
        return cls(**{k: float(v) for k, v in data.items() if hasattr(cls, k)})
    
    def to_bytes(self) -> bytes:
        """Serialize payload to bytes for efficient storage."""
        return json.dumps(self.to_dict()).encode('utf-8')
    
    @classmethod
    def from_bytes(cls, data: bytes) -> 'Payload':
        """Deserialize payload from bytes."""
        return cls.from_dict(json.loads(data.decode('utf-8')))

class BaseIndex:
    """Base class for index implementations."""
    
    def __init__(self, dim: int, space: str = "cosine", **kwargs):
        """Initialize the index.
        
        Args:
            dim: Dimensionality of the embeddings
            space: Distance metric ('cosine' or 'l2')
            **kwargs: Additional backend-specific parameters
        """
        self.dim = dim
        self.space = space
        self._index = None
        self._doc_ids = []
        self._payloads = {}
        self._is_trained = False
        
    def add(self, doc_id: str, embedding: np.ndarray, payload: Payload) -> None:
        """Add a document to the index.
        
        Args:
            doc_id: Unique document identifier
            embedding: Document embedding vector
            payload: Document metadata and scores
        """
        raise NotImplementedError
        
    def build(self, **kwargs) -> None:
        """Build the index after all documents are added."""
        raise NotImplementedError
        
    def search(self, 
              query: np.ndarray, 
              k: int = 10, 
              eta: float = 0.5,
              entropy_pref: float = 0.0) -> List[Tuple[str, float, Payload]]:
        """Search the index with DEWI re-ranking.
        
        Args:
            query: Query embedding
            k: Number of results to return
            eta: Weight of DEWI score in re-ranking (0.0-1.0)
            entropy_pref: Preference for high/low entropy (-1.0 to 1.0)
            
        Returns:
            List of (doc_id, score, payload) tuples
        """
        raise NotImplementedError
    
    def save(self, path: Union[str, Path]) -> None:
        """Save the index to disk.
        
        Args:
            path: Directory to save index files
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save metadata
        metadata = {
            'dim': self.dim,
            'space': self.space,
            'doc_ids': self._doc_ids,
            'is_trained': self._is_trained,
            'type': self.__class__.__name__
        }
        
        # Save payloads
        payload_path = path / 'payloads.jsonl'
        with open(payload_path, 'w') as f:
            for doc_id in self._doc_ids:
                payload = self._payloads[doc_id]
                f.write(json.dumps({'id': doc_id, 'payload': payload.to_dict()}) + '\n')
        
        # Save metadata
        with open(path / 'metadata.json', 'w') as f:
            json.dump(metadata, f)
    
    @classmethod
    def load(cls, path: Union[str, Path], **kwargs) -> 'BaseIndex':
        """Load an index from disk.
        
        Args:
            path: Directory containing index files
            **kwargs: Additional loading parameters
            
        Returns:
            Loaded index instance
        """
        path = Path(path)
        
        # Load metadata
        with open(path / 'metadata.json') as f:
            metadata = json.load(f)
        
        # Create appropriate index type
        index_cls = globals().get(metadata['type'], cls)
        index = index_cls(dim=metadata['dim'], space=metadata['space'], **kwargs)
        
        # Load payloads
        index._doc_ids = metadata['doc_ids']
        index._is_trained = metadata['is_trained']
        
        with open(path / 'payloads.jsonl') as f:
            for line in f:
                data = json.loads(line)
                index._payloads[data['id']] = Payload.from_dict(data['payload'])
        
        return index


class HNSWIndex(BaseIndex):
    """HNSW-based approximate nearest neighbor search index."""
    
    def __init__(self, dim: int, space: str = "cosine", M: int = 16, ef_construction: int = 200, **kwargs):
        super().__init__(dim, space, **kwargs)
        if not _HAS_HNSW:
            raise ImportError("HNSW not available. Install with 'pip install hnswlib'")
            
        self.M = M
        self.ef_construction = ef_construction
        self._index = hnswlib.Index(space=space, dim=dim)
        self._index.init_index(
            max_elements=10000,  # Will be resized as needed
            ef_construction=ef_construction,
            M=M
        )
    
    def add(self, doc_id: str, embedding: np.ndarray, payload: Payload) -> None:
        if embedding.shape != (self.dim,):
            raise ValueError(f"Expected embedding of shape {(self.dim,)}, got {embedding.shape}")
            
        self._doc_ids.append(doc_id)
        self._payloads[doc_id] = payload
        
        # Convert to numpy array and reshape for HNSW
        emb_array = np.array([embedding], dtype=np.float32)
        
        # Resize index if needed
        if len(self._doc_ids) > self._index.max_elements:
            new_size = int(self._index.max_elements * 1.5)
            self._index.resize_index(new_size)
        
        # Add to index
        self._index.add_items(emb_array, [len(self._doc_ids) - 1])
    
    def build(self, **kwargs) -> None:
        self._is_trained = True
    
    def search(self, query: np.ndarray, k: int = 10, eta: float = 0.5, entropy_pref: float = 0.0) -> List[Tuple[str, float, Payload]]:
        if not self._is_trained:
            self.build()
            
        # Convert query to numpy array if needed
        if not isinstance(query, np.ndarray):
            query = np.array(query, dtype=np.float32)
        
        # Reshape for HNSW
        if query.ndim == 1:
            query = query.reshape(1, -1)
        
        # Search for approximate nearest neighbors
        indices, distances = self._index.knn_query(query, k=k)
        
        # Convert to list of results
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx >= len(self._doc_ids):
                continue
                
            doc_id = self._doc_ids[idx]
            payload = self._payloads[doc_id]
            
            # Apply DEWI re-ranking
            dewi_score = payload.dewi
            adjusted_score = (1 - eta) * (1 - dist) + eta * dewi_score
            
            # Apply entropy preference if specified
            if entropy_pref != 0:
                entropy = (payload.ht_mean + payload.hi_mean) / 2
                adjusted_score += entropy_pref * entropy
            
            results.append((doc_id, float(adjusted_score), payload))
        
        # Sort by adjusted score (descending)
        results.sort(key=lambda x: x[1], reverse=True)
        return results


class FAISSIndex(BaseIndex):
    """FAISS-based index with support for multiple index types."""
    
    def __init__(self, dim: int, space: str = "cosine", index_type: str = "IVFFlat", nlist: int = 100, **kwargs):
        super().__init__(dim, space, **kwargs)
        if not _HAS_FAISS:
            raise ImportError("FAISS not available. Install with 'pip install faiss-cpu' or 'faiss-gpu'")
            
        self.index_type = index_type
        self.nlist = nlist
        self._index = None
        self._embeddings = []
        
        # Initialize FAISS index
        if space == "cosine":
            self.metric = faiss.METRIC_INNER_PRODUCT
            self.normalize = True
        else:  # L2
            self.metric = faiss.METRIC_L2
            self.normalize = False
    
    def add(self, doc_id: str, embedding: np.ndarray, payload: Payload) -> None:
        if embedding.shape != (self.dim,):
            raise ValueError(f"Expected embedding of shape {(self.dim,)}, got {embedding.shape}")
            
        # Store embedding and metadata
        self._doc_ids.append(doc_id)
        self._payloads[doc_id] = payload
        
        # Store embedding (will be added to index during build)
        emb = embedding.astype(np.float32)
        if self.normalize:
            emb = emb / np.linalg.norm(emb)
        self._embeddings.append(emb)
    
    def build(self, **kwargs) -> None:
        if not self._embeddings:
            raise ValueError("No embeddings to build index from")
            
        # Convert embeddings to numpy array
        embeddings = np.stack(self._embeddings)
        
        # Create and train index
        if self.index_type == "IVFFlat":
            quantizer = faiss.IndexFlatIP(self.dim) if self.metric == faiss.METRIC_INNER_PRODUCT else faiss.IndexFlatL2(self.dim)
            self._index = faiss.IndexIVFFlat(
                quantizer, self.dim, min(self.nlist, len(embeddings)), self.metric
            )
            self._index.train(embeddings)
            self._index.add(embeddings)
            
        elif self.index_type == "HNSW":
            self._index = faiss.IndexHNSWFlat(self.dim, 32, self.metric)
            self._index.hnsw.efConstruction = 200
            self._index.add(embeddings)
            
        else:  # Flat index for exact search
            self._index = faiss.IndexFlatIP(self.dim) if self.metric == faiss.METRIC_INNER_PRODUCT else faiss.IndexFlatL2(self.dim)
            self._index.add(embeddings)
        
        self._is_trained = True
        
        # Clear stored embeddings to save memory
        self._embeddings = []
    
    def search(self, query: np.ndarray, k: int = 10, eta: float = 0.5, entropy_pref: float = 0.0) -> List[Tuple[str, float, Payload]]:
        if not self._is_trained and self._index is None:
            self.build()
        
        # Convert query to numpy array if needed
        if not isinstance(query, np.ndarray):
            query = np.array(query, dtype=np.float32)
        
        # Normalize query for cosine similarity
        if self.normalize:
            query = query / np.linalg.norm(query)
        
        # Reshape for FAISS
        if query.ndim == 1:
            query = query.reshape(1, -1)
        
        # Search
        distances, indices = self._index.search(query, k)
        
        # Convert to list of results
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx >= len(self._doc_ids) or idx < 0:
                continue
                
            doc_id = self._doc_ids[idx]
            payload = self._payloads[doc_id]
            
            # For cosine similarity, convert distance to similarity score
            if self.metric == faiss.METRIC_INNER_PRODUCT:
                score = dist
            else:  # L2 distance
                score = 1.0 / (1.0 + dist)  # Convert distance to similarity
            
            # Apply DEWI re-ranking
            dewi_score = payload.dewi
            adjusted_score = (1 - eta) * score + eta * dewi_score
            
            # Apply entropy preference if specified
            if entropy_pref != 0:
                entropy = (payload.ht_mean + payload.hi_mean) / 2
                adjusted_score += entropy_pref * entropy
            
            results.append((doc_id, float(adjusted_score), payload))
        
        # Sort by adjusted score (descending)
        results.sort(key=lambda x: x[1], reverse=True)
        return results
    
    def save(self, path: Union[str, Path]) -> None:
        """Save FAISS index to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        if self._index is not None:
            faiss.write_index(self._index, str(path / 'index.faiss'))
        
        # Save metadata and payloads
        super().save(path)
    
    @classmethod
    def load(cls, path: Union[str, Path], **kwargs) -> 'FAISSIndex':
        """Load FAISS index from disk."""
        path = Path(path)
        
        # Load metadata and payloads
        index = super().load(path, **kwargs)
        
        # Load FAISS index if it exists
        index_file = path / 'index.faiss'
        if index_file.exists():
            index._index = faiss.read_index(str(index_file))
        
        return index


class ExactIndex(BaseIndex):
    """Exact nearest neighbor search using brute force."""
    
    def __init__(self, dim: int, space: str = "cosine", **kwargs):
        super().__init__(dim, space, **kwargs)
        self._embeddings = []
        self._normalize = (space == "cosine")
    
    def add(self, doc_id: str, embedding: np.ndarray, payload: Payload) -> None:
        if embedding.shape != (self.dim,):
            raise ValueError(f"Expected embedding of shape {(self.dim,)}, got {embedding.shape}")
            
        # Store embedding and metadata
        self._doc_ids.append(doc_id)
        self._payloads[doc_id] = payload
        
        # Store normalized embedding for cosine similarity
        emb = embedding.astype(np.float32)
        if self._normalize:
            emb = emb / np.linalg.norm(emb)
        self._embeddings.append(emb)
    
    def build(self, **kwargs) -> None:
        if not self._embeddings:
            raise ValueError("No embeddings to build index from")
        self._embeddings = np.stack(self._embeddings)
        self._is_trained = True
    
    def search(self, query: np.ndarray, k: int = 10, eta: float = 0.5, entropy_pref: float = 0.0) -> List[Tuple[str, float, Payload]]:
        """Search the index with DEWI re-ranking.
        
        Optimized version that minimizes memory allocations and uses vectorized operations.
        """
        # Normalize query for cosine similarity if needed
        query = np.asarray(query, dtype=np.float32)
        if self._normalize:
            query_norm = np.linalg.norm(query)
            if query_norm > 0:
                query = query / query_norm
        
        # Reshape query for matrix operations
        if query.ndim == 1:
            query = query.reshape(1, -1)
        
        # Compute similarity scores (vectorized)
        if self._normalize:
            # Cosine similarity: dot product of normalized vectors
            scores = np.dot(self._embeddings, query.T).flatten()
        else:
            # Negative L2 distance
            scores = -np.sum((self._embeddings - query) ** 2, axis=1)
        
        # Get top-2k candidates for re-ranking (or all if fewer than 2k)
        candidate_count = min(2 * k, len(scores))
        if candidate_count <= 0:
            return []
            
        # Get top candidate indices in one pass
        top_indices = np.argpartition(scores, -candidate_count)[-candidate_count:]
        
        # Extract scores and payloads for candidates
        candidate_scores = scores[top_indices]
        
        # Pre-allocate arrays for DEWI scores and entropy
        dewi_scores = np.zeros(candidate_count, dtype=np.float32)
        entropies = np.zeros(candidate_count, dtype=np.float32)
        
        # Extract payload data in a single pass
        for i, idx in enumerate(top_indices):
            doc_id = self._doc_ids[idx]
            payload = self._payloads[doc_id]
            dewi_scores[i] = payload.dewi
            entropies[i] = (payload.ht_mean + payload.hi_mean) * 0.5  # Mean entropy
        
        # Apply DEWI re-ranking (vectorized)
        adjusted_scores = (1 - eta) * candidate_scores + eta * dewi_scores
        
        # Apply entropy preference if needed (vectorized)
        if entropy_pref != 0:
            adjusted_scores += entropy_pref * entropies
        
        # Get top-k indices after re-ranking
        top_k_indices = np.argpartition(adjusted_scores, -k)[-k:]
        
        # Sort top-k results by adjusted score (descending)
        sorted_indices = top_k_indices[np.argsort(-adjusted_scores[top_k_indices])]
        
        # Prepare final results
        results = []
        for idx in sorted_indices:
            doc_idx = top_indices[idx]
            doc_id = self._doc_ids[doc_idx]
            payload = self._payloads[doc_id]
            results.append((doc_id, float(adjusted_scores[idx]), payload))
        
        return results
        
    def save(self, path: Union[str, Path]) -> None:
        """Save the index to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Convert embeddings to numpy array if needed
        if len(self._embeddings) > 0 and not isinstance(self._embeddings, np.ndarray):
            embeddings_array = np.array(self._embeddings)
        else:
            embeddings_array = self._embeddings
        
        # Save metadata
        metadata = {
            "dim": self.dim,
            "space": self.space,
            "doc_ids": self._doc_ids,
            "normalize": self._normalize,
            "is_trained": self._is_trained,
            "num_embeddings": len(embeddings_array) if embeddings_array is not None else 0
        }
        
        with open(path / "metadata.json", "w") as f:
            json.dump(metadata, f)
            
        # Save payloads
        with open(path / "payloads.jsonl", "w") as f:
            for doc_id in self._doc_ids:
                payload = self._payloads[doc_id]
                f.write(json.dumps({"doc_id": doc_id, "payload": payload.to_dict()}) + "\n")
        
        # Save embeddings as numpy array if we have any
        if embeddings_array is not None and len(embeddings_array) > 0:
            np.save(str(path / "embeddings.npy"), embeddings_array)
    
    @classmethod
    def load(cls, path: Union[str, Path], **kwargs) -> 'ExactIndex':
        """Load an index from disk."""
        path = Path(path)
        
        # Load metadata
        with open(path / "metadata.json", "r") as f:
            metadata = json.load(f)
        
        # Create index
        index = cls(dim=metadata["dim"], space=metadata["space"])
        index._doc_ids = metadata["doc_ids"]
        index._normalize = metadata["normalize"]
        index._is_trained = metadata["is_trained"]
        
        # Load payloads
        index._payloads = {}
        with open(path / "payloads.jsonl", "r") as f:
            for line in f:
                data = json.loads(line)
                index._payloads[data["doc_id"]] = Payload.from_dict(data["payload"])
        
        # Load embeddings if they exist and we expect them to be there
        embeddings_path = path / "embeddings.npy"
        if embeddings_path.exists() and metadata.get("num_embeddings", 0) > 0:
            loaded_embeddings = np.load(str(embeddings_path))
            # Convert to list of arrays if needed
            if isinstance(loaded_embeddings, np.ndarray):
                index._embeddings = loaded_embeddings
            else:
                index._embeddings = list(loaded_embeddings)
        else:
            index._embeddings = []
        
        # If we have doc_ids but no embeddings, we need to rebuild them
        if index._doc_ids and (isinstance(index._embeddings, list) and not index._embeddings or 
                              isinstance(index._embeddings, np.ndarray) and index._embeddings.size == 0):
            logger.warning("No embeddings found during load, index will need to be rebuilt")
        
        return index


class DewiIndex(BaseIndex):
    """DEWI Index with support for multiple backends and entropy-weighted scoring.
    
    This class provides a unified interface for approximate nearest neighbor search
    with support for multiple backends (FAISS, HNSW, or exact search). It includes
    functionality for entropy-weighted scoring and re-ranking of results.
    
    Args:
        dim: Dimensionality of the embedding space
        space: Distance metric ('cosine' or 'l2')
        backend: Backend to use (FAISS, HNSW, or exact)
        ef: Size of the dynamic list for HNSW index building
        M: Number of bi-directional links for HNSW
        use_ann: Whether to use approximate nearest neighbor search
        ef_query: Size of the dynamic list for search
        rerank_eta: Weight of DEWI score in re-ranking (0.0 to 1.0)
        entropy_pref: Entropy preference (-1.0 to 1.0, higher favors more surprising content)
        **kwargs: Additional backend-specific parameters
    """
    
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
        **kwargs
    ):
        """Initialize the DEWI index."""
        super().__init__(dim, space)
        self._ids: List[str] = []
        self._emb: List[np.ndarray] = []
        self._payloads: Dict[str, Payload] = {}
        self._meta: Dict[str, Dict[str, Any]] = {}
        self.ef_query = ef_query
        self.rerank_eta = rerank_eta
        self.entropy_pref = entropy_pref
        self._built = False
        
        # Determine backend
        if isinstance(backend, str):
            backend = IndexBackend.from_str(backend)
        
        # Initialize the appropriate backend
        self._use_ann = use_ann
        self._backend = None
        
        if use_ann:
            if backend == IndexBackend.FAISS_IVFFLAT and _HAS_FAISS:
                self._backend = FAISSIndex(dim, space, index_type="IVFFlat", **kwargs)
            elif backend == IndexBackend.FAISS_HNSW and _HAS_FAISS:
                self._backend = FAISSIndex(dim, space, index_type="HNSW", **kwargs)
            elif backend == IndexBackend.HNSW and _HAS_HNSW:
                self._backend = HNSWIndex(dim, space, ef_construction=ef, M=M, **kwargs)
        
        # Fall back to exact search if no ANN backend is available or requested
        if self._backend is None:
            if use_ann:
                logger.warning("Falling back to exact search. For better performance, install FAISS or HNSW.")
            self._backend = ExactIndex(dim, space, **kwargs)
    
    @classmethod
    def create(
        cls,
        dim: int,
        space: str = "cosine",
        backend: Union[str, IndexBackend] = "auto",
        **kwargs
    ) -> 'DewiIndex':
        """Factory method to create a new DewiIndex instance.
        
        This is provided for backward compatibility with code that used the
        DewiIndex factory function.
        
        Args:
            dim: Dimensionality of the embeddings
            space: Distance metric ('cosine' or 'l2')
            backend: Backend to use ('auto', 'faiss', 'hnsw', or 'exact')
            **kwargs: Additional backend-specific parameters
            
        Returns:
            A new DewiIndex instance
        """
        return cls(dim=dim, space=space, backend=backend, **kwargs)
    
    def add(
        self, 
        doc_id: str, 
        embedding: np.ndarray, 
        payload: Payload, 
        meta: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add a document to the index.
        
        Args:
            doc_id: Unique document identifier
            embedding: Document embedding vector
            payload: DEWI scores and signals
            meta: Optional metadata dictionary
        """
        # Validate input
        if not isinstance(doc_id, str):
            raise ValueError(f"doc_id must be a string, got {type(doc_id)}")
            
        if not isinstance(payload, Payload):
            raise ValueError(f"payload must be an instance of Payload, got {type(payload)}")
            
        embedding = np.asarray(embedding, dtype=np.float32)
        if embedding.shape != (self.dim,):
            raise ValueError(f"Expected embedding of shape ({self.dim},), got {embedding.shape}")
            
        # Add to our internal storage
        self._ids.append(doc_id)
        self._emb.append(embedding)
        self._payloads[doc_id] = payload
        if meta is not None:
            self._meta[doc_id] = meta
            
        # Add to the backend index if it's built
        if hasattr(self, '_backend') and self._backend is not None:
            self._backend.add(doc_id, embedding, payload)
    
    def build(self, ef_construction: Optional[int] = None, M: Optional[int] = None) -> None:
        """Build the index after adding documents.
        
        This method must be called after adding documents and before searching
        for optimal performance with ANN backends.
        
        Args:
            ef_construction: Size of the dynamic list for index building (HNSW)
            M: Number of bi-directional links (HNSW)
        """
        if not self._emb:
            logger.warning("No documents to index")
            return
            
        if hasattr(self._backend, 'build'):
            # If the backend supports building, delegate to it
            self._backend.build(ef_construction=ef_construction, M=M)
        
        self._built = True
        logger.info(f"Built index with {len(self._ids)} documents")
    
    def _search_exact(
        self, 
        q: np.ndarray, 
        k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Exact nearest neighbor search using brute force."""
        if not self._emb:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float32)
            
        xb = np.stack(self._emb, axis=0)
        
        if self.space == "cosine":
            # Normalize query and database vectors
            qn = q / (np.linalg.norm(q) + 1e-8)
            xn = xb / (np.linalg.norm(xb, axis=1, keepdims=True) + 1e-8)
            sims = xn @ qn
            idx = np.argsort(-sims)[:k]
            return idx, sims[idx]
        elif self.space == "l2":
            d2 = np.sum((xb - q[None, :]) ** 2, axis=1)
            idx = np.argsort(d2)[:k]
            return idx, -d2[idx]  # Return negative distance for consistency
        else:
            raise ValueError(f"Unsupported space: {self.space}")
    
    def _rerank_with_dewi(
        self,
        query: np.ndarray,
        results: List[Tuple[str, float, Payload]],
        eta: float,
        entropy_pref: float
    ) -> List[Tuple[str, float, Payload]]:
        """Re-rank search results using DEWI scoring.
        
        Args:
            query: The query embedding
            results: Initial search results (doc_id, score, payload)
            eta: Weight of DEWI score in re-ranking
            entropy_pref: Entropy preference (-1.0 to 1.0)
            
        Returns:
            Re-ranked results with updated scores
        """
        if not results:
            return []
            
        # Extract components for re-ranking
        doc_ids, scores, payloads = zip(*results)
        doc_ids = list(doc_ids)
        scores = np.array(scores, dtype=np.float32)
        
        # Get text and image entropy scores
        ht_means = np.array([p.ht_mean for p in payloads], dtype=np.float32)
        hi_means = np.array([p.hi_mean for p in payloads], dtype=np.float32)
        
        # Combine entropy scores based on preference
        # When entropy_pref > 0, higher entropy is preferred (more surprising)
        # When entropy_pref < 0, lower entropy is preferred (more typical)
        entropy_scores = 0.5 * (ht_means + hi_means)  # Average of text and image entropy
        
        # Normalize scores to [0, 1]
        if len(scores) > 1:
            # Avoid division by zero
            score_min, score_max = scores.min(), scores.max()
            if score_max > score_min:
                norm_scores = (scores - score_min) / (score_max - score_min)
            else:
                norm_scores = np.ones_like(scores)
        else:
            norm_scores = np.ones_like(scores)
            
        # Normalize entropy scores to [0, 1]
        if len(entropy_scores) > 1:
            entropy_min, entropy_max = entropy_scores.min(), entropy_scores.max()
            if entropy_max > entropy_min:
                norm_entropy = (entropy_scores - entropy_min) / (entropy_max - entropy_min)
            else:
                norm_entropy = np.ones_like(entropy_scores)
        else:
            norm_entropy = np.ones_like(entropy_scores)
            
        # Apply entropy preference
        if entropy_pref > 0:
            # Higher entropy is better (more surprising)
            entropy_weights = norm_entropy * entropy_pref
        elif entropy_pref < 0:
            # Lower entropy is better (more typical)
            entropy_weights = (1.0 - norm_entropy) * abs(entropy_pref)
        else:
            entropy_weights = np.zeros_like(norm_entropy)
            
        # Combine scores with entropy weighting
        combined_scores = (1.0 - eta) * norm_scores + eta * entropy_weights
        
        # Sort by combined score
        sorted_indices = np.argsort(-combined_scores)  # Descending order
        
        # Return re-ranked results
        return [
            (doc_ids[i], float(combined_scores[i]), payloads[i])
            for i in sorted_indices
        ]
    
    def search(
        self, 
        query: np.ndarray, 
        k: int = 10, 
        eta: Optional[float] = None,
        entropy_pref: Optional[float] = None
    ) -> List[Tuple[str, float, Payload]]:
        """Search for similar documents with DEWI re-ranking.
        
        Args:
            query: Query embedding vector
            k: Number of results to return
            eta: Weight of DEWI score in re-ranking (0.0 to 1.0)
            entropy_pref: Entropy preference (-1.0 to 1.0, higher favors more surprising content)
                         where +1 favors higher DEWI scores. If None, uses instance default.
            eta: Optional override for DEWI score weight in re-ranking (0.0 to 1.0).
                 If None, uses instance default.
            
        Returns:
            List of (doc_id, score, payload) tuples, sorted by score
        """
        if not self._built:
            self.build()
            
        # Use instance defaults if parameters not provided
        if entropy_pref is None:
            entropy_pref = self.entropy_pref
        if eta is None:
            eta = self.rerank_eta
            
        # Validate parameters
        if not (-1.0 <= entropy_pref <= 1.0):
            raise ValueError("entropy_pref must be between -1.0 and 1.0")
        if not (0.0 <= eta <= 1.0):
            raise ValueError("eta must be between 0.0 and 1.0")
            
        query = np.asarray(query, dtype=np.float32)
        if query.shape != (self.dim,):
            raise ValueError(f"Expected query of shape ({self.dim},), got {query.shape}")
        
        # First-stage retrieval
        if self._use_ann and self._emb:
            # Use approximate nearest neighbor search
            ids, dists = self._index.knn_query(query, k=min(k*5, len(self._emb)))
            cand_idx = ids[0]
            sims = -dists[0] if self.space != "cosine" else dists[0]
        else:
            # Fall back to exact search
            cand_idx, sims = self._search_exact(query, k=min(k*5, len(self._emb)))
        
        # Re-rank candidates with DEWI score
        items = []
        for idx in cand_idx:
            doc_id = self._ids[int(idx)]
            sim = float(sims[np.where(cand_idx == idx)[0][0]])
            pl = self._payloads[doc_id]
            
            # Combine similarity and DEWI score
            rerank = sim + eta * float(entropy_pref) * float(pl.dewi)
            items.append((doc_id, rerank, pl))
        
        # Sort by re-ranking score
        items.sort(key=lambda x: x[1], reverse=True)
        return items[:k]
        
    def save(self, path: Union[str, Path]) -> None:
        """Save the index to disk.
        
        Args:
            path: Directory path where to save the index files
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save HNSW index if using ANN
        if self._use_ann and self._index is not None:
            self._index.save_index(str(path / "hnsw.bin"))
        
        # Save document IDs and embeddings
        if self._ids:
            np.save(path / "ids.npy", np.array(self._ids))
            np.save(path / "emb.npy", np.stack(self._emb))
        
        # Save payloads and metadata
        if self._payloads:
            with open(path / "payloads.jsonl", 'w') as f:
                for doc_id, payload in self._payloads.items():
                    payload_dict = {
                        'id': doc_id,
                        'dewi': payload.dewi,
                        'ht_mean': payload.ht_mean,
                        'ht_q90': payload.ht_q90,
                        'hi_mean': payload.hi_mean,
                        'hi_q90': payload.hi_q90,
                        'I_hat': payload.I_hat,
                        'redundancy': payload.redundancy,
                        'noise': payload.noise
                    }
                    f.write(json.dumps(payload_dict) + '\n')
        
        # Save metadata if present
        if any(self._meta.values()):
            with open(path / "meta.json", 'w') as f:
                json.dump(self._meta, f)
        
        # Save index configuration
        config = {
            'dim': self.dim,
            'space': self.space,
            'use_ann': self._use_ann,
            'ef_query': self.ef_query,
            'rerank_eta': self.rerank_eta,
            'entropy_pref': self.entropy_pref,
            'built': self._built
        }
        with open(path / "config.json", 'w') as f:
            json.dump(config, f)
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'DewiIndex':
        """Load an index from disk.
        
        Args:
            path: Directory path where the index is saved
            
        Returns:
            Loaded DewiIndex instance
        """
        path = Path(path)
        
        # Load configuration
        with open(path / "config.json", 'r') as f:
            config = json.load(f)
        
        # Create index instance
        index = cls(
            dim=config['dim'],
            space=config['space'],
            use_ann=config['use_ann'],
            ef_query=config.get('ef_query', 200),
            rerank_eta=config.get('rerank_eta', 0.25),
            entropy_pref=config.get('entropy_pref', 0.0)
        )
        
        # Load document IDs and embeddings
        if (path / "ids.npy").exists() and (path / "emb.npy").exists():
            index._ids = list(np.load(path / "ids.npy", allow_pickle=True))
            embeddings = np.load(path / "emb.npy")
            index._emb = [emb for emb in embeddings]
        
        # Load payloads
        if (path / "payloads.jsonl").exists():
            with open(path / "payloads.jsonl", 'r') as f:
                for line in f:
                    data = json.loads(line)
                    doc_id = data.pop('id')
                    index._payloads[doc_id] = Payload(**data)
        
        # Load metadata
        if (path / "meta.json").exists():
            with open(path / "meta.json", 'r') as f:
                index._meta = json.load(f)
        
        # Load HNSW index if using ANN
        if index._use_ann and (path / "hnsw.bin").exists():
            index._index = hnswlib.Index(space=index.space, dim=index.dim)
            index._index.load_index(str(path / "hnsw.bin"), max_elements=len(index._ids))
            index._index.set_ef(index.ef_query)
        
        index._built = config.get('built', False)
        return index
    
    def __len__(self) -> int:
        """Number of documents in the index."""
        return len(self._ids)
    
    def get_payload(self, doc_id: str) -> Optional[Payload]:
        """Get payload for a document."""
        return self._payloads.get(doc_id)
    
    def get_metadata(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a document."""
        return self._meta.get(doc_id)
    
    def get_embedding(self, doc_id: str) -> Optional[np.ndarray]:
        """Get embedding for a document."""
        try:
            idx = self._ids.index(doc_id)
            return self._emb[idx]
        except ValueError:
            return None


def create_dewi_index(dim: int, space: str = "cosine", backend: Union[str, IndexBackend] = "auto", **kwargs) -> DewiIndex:
    """Backward-compatible factory for DewiIndex."""
    return DewiIndex(dim=dim, space=space, backend=backend, **kwargs)
