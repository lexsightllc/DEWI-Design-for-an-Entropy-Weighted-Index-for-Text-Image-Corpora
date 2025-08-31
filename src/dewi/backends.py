"""Indexing and retrieval for DEWI with support for multiple backends (FAISS/HNSW)."""

from __future__ import annotations
import json
import os
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Type, TypeVar
import numpy as np
import logging

from .types import Payload

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


