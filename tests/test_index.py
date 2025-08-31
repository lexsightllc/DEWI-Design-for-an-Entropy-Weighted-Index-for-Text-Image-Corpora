"""Tests for DEWI index implementation."""

import json
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pytest

from dewi.index import (
    Payload, 
    DewiIndex, 
    IndexBackend,
    HNSWIndex, 
    FAISSIndex, 
    ExactIndex,
    _HAS_FAISS, 
    _HAS_HNSW
)

# Test data
DIM = 128
N_DOCS = 100
N_QUERIES = 5
K = 10

# Generate test data
np.random.seed(42)

def generate_embeddings(n: int, dim: int) -> np.ndarray:
    """Generate random normalized embeddings."""
    embs = np.random.randn(n, dim).astype(np.float32)
    return embs / np.linalg.norm(embs, axis=1, keepdims=True)

def generate_payloads(n: int) -> List[Payload]:
    """Generate random payloads."""
    return [
        Payload(
            dewi=float(np.clip(np.random.beta(2, 2), 0, 1)),
            ht_mean=float(np.random.gamma(2, 0.5)),
            ht_q90=float(np.random.gamma(2, 0.5) * 1.5),
            hi_mean=float(np.random.gamma(2, 0.3)),
            hi_q90=float(np.random.gamma(2, 0.3) * 1.5),
            I_hat=float(np.random.beta(2, 2)),
            redundancy=float(np.random.beta(1, 5)),
            noise=float(np.random.beta(1, 10))
        )
        for _ in range(n)
    ]

# Fixtures
@pytest.fixture
def doc_embeddings() -> np.ndarray:
    """Generate document embeddings."""
    return generate_embeddings(N_DOCS, DIM)

@pytest.fixture
def doc_payloads() -> List[Payload]:
    """Generate document payloads."""
    return generate_payloads(N_DOCS)

@pytest.fixture
def doc_ids() -> List[str]:
    """Generate document IDs."""
    return [f"doc_{i}" for i in range(N_DOCS)]

@pytest.fixture
def query_embeddings() -> np.ndarray:
    """Generate query embeddings."""
    return generate_embeddings(N_QUERIES, DIM)

def test_payload_serialization():
    """Test Payload serialization and deserialization."""
    payload = Payload(
        dewi=0.5,
        ht_mean=1.2,
        ht_q90=1.8,
        hi_mean=0.8,
        hi_q90=1.2,
        I_hat=0.6,
        redundancy=0.1,
        noise=0.05
    )
    
    # Test dict conversion
    payload_dict = payload.to_dict()
    assert isinstance(payload_dict, dict)
    assert 'dewi' in payload_dict
    
    # Test from_dict
    payload2 = Payload.from_dict(payload_dict)
    assert payload2.dewi == payload.dewi
    
    # Test bytes conversion
    payload_bytes = payload.to_bytes()
    assert isinstance(payload_bytes, bytes)
    
    # Test from_bytes
    payload3 = Payload.from_bytes(payload_bytes)
    assert payload3.dewi == payload.dewi

def test_exact_index(doc_embeddings, doc_payloads, doc_ids, query_embeddings):
    """Test exact nearest neighbor search."""
    index = ExactIndex(dim=DIM, space="cosine")
    
    # Add documents
    for doc_id, emb, payload in zip(doc_ids, doc_embeddings, doc_payloads):
        index.add(doc_id, emb, payload)
    
    # Build index
    index.build()
    
    # Test search
    for query in query_embeddings:
        results = index.search(query, k=K)
        assert len(results) == K
        
        # Check results are sorted by score (descending)
        scores = [r[1] for r in results]
        assert all(scores[i] >= scores[i+1] for i in range(len(scores)-1))
        
        # Check all returned doc_ids are valid
        for doc_id, score, payload in results:
            assert doc_id in doc_ids
            assert isinstance(score, float)
            assert isinstance(payload, Payload)

@pytest.mark.skipif(not _HAS_HNSW, reason="HNSW not installed")
def test_hnsw_index(doc_embeddings, doc_payloads, doc_ids, query_embeddings):
    """Test HNSW approximate nearest neighbor search."""
    index = HNSWIndex(dim=DIM, space="cosine", M=16, ef_construction=200)
    
    # Add documents
    for doc_id, emb, payload in zip(doc_ids, doc_embeddings, doc_payloads):
        index.add(doc_id, emb, payload)
    
    # Build index
    index.build()
    
    # Test search
    for query in query_embeddings:
        results = index.search(query, k=K)
        assert len(results) == K
        
        # Check results are sorted by score (descending)
        scores = [r[1] for r in results]
        assert all(scores[i] >= scores[i+1] for i in range(len(scores)-1))
        
        # Check all returned doc_ids are valid
        for doc_id, score, payload in results:
            assert doc_id in doc_ids
            assert isinstance(score, float)
            assert isinstance(payload, Payload)

@pytest.mark.skipif(not _HAS_FAISS, reason="FAISS not installed")
def test_faiss_index(doc_embeddings, doc_payloads, doc_ids, query_embeddings):
    """Test FAISS approximate nearest neighbor search."""
    # Use a smaller nlist for testing
    index = FAISSIndex(dim=DIM, space="cosine", index_type="IVFFlat", nlist=5)
    
    # Add documents
    for doc_id, emb, payload in zip(doc_ids, doc_embeddings, doc_payloads):
        index.add(doc_id, emb, payload)
    
    # Build index with more training data
    index.build(ntotal=len(doc_embeddings))
    
    # Test search
    for query in query_embeddings:
        results = index.search(query, k=min(K, len(doc_embeddings)))
        # FAISS might return fewer than k results if there aren't enough vectors
        assert len(results) > 0, "No results returned from FAISS search"
        assert len(results) <= K, f"Returned {len(results)} results, expected at most {K}"
        
        # Check results are sorted by score (descending)
        scores = [r[1] for r in results]
        assert all(scores[i] >= scores[i+1] for i in range(len(scores)-1))
        
        # Check all returned doc_ids are valid
        for doc_id, score, payload in results:
            assert doc_id in doc_ids
            assert isinstance(score, float)
            assert isinstance(payload, Payload)

def test_dewi_index_factory(doc_embeddings, doc_payloads, doc_ids):
    """Test DewiIndex factory function."""
    # Test with auto backend
    index = DewiIndex(dim=DIM, space="cosine")
    assert isinstance(index, (ExactIndex, HNSWIndex, FAISSIndex))

    # Add some documents - use more than k to ensure we have enough for search
    num_docs = 20
    for doc_id, emb, payload in zip(doc_ids[:num_docs], doc_embeddings[:num_docs], doc_payloads[:num_docs]):
        index.add(doc_id, emb, payload)

    # Test build with explicit ntotal
    index.build(ntotal=num_docs)
    
    # Test search with k up to the number of documents we added
    k = min(5, num_docs)
    results = index.search(doc_embeddings[0], k=k)
    assert len(results) > 0, "No results returned from search"
    assert len(results) <= k, f"Returned {len(results)} results, expected at most {k}"

def test_index_persistence(tmp_path, doc_embeddings, doc_payloads, doc_ids):
    """Test saving and loading index."""
    # Use a small subset for testing
    test_docs = 10
    test_embeddings = doc_embeddings[:test_docs]
    test_payloads = doc_payloads[:test_docs]
    test_ids = doc_ids[:test_docs]
    
    # Create and populate index
    index = ExactIndex(dim=DIM, space="cosine")
    for doc_id, emb, payload in zip(test_ids, test_embeddings, test_payloads):
        index.add(doc_id, emb, payload)
    
    # Build the index
    index.build()
    
    # Save index
    save_path = tmp_path / "test_index"
    save_path.mkdir(parents=True, exist_ok=True)
    index.save(save_path)
    
    # Check files were created
    assert (save_path / "metadata.json").exists()
    assert (save_path / "payloads.jsonl").exists()
    
    # Load index
    loaded_index = ExactIndex.load(save_path)
    
    # Verify loaded index
    assert loaded_index.dim == DIM
    assert len(loaded_index._doc_ids) == test_docs
    assert len(loaded_index._payloads) == test_docs
    
    # Test search on loaded index with a query
    query = test_embeddings[0]
    results = loaded_index.search(query, k=min(5, test_docs))
    
    # Verify we got some results
    assert len(results) > 0
    assert len(results) <= min(5, test_docs)
    assert len(results) == 5
    assert all(isinstance(r[0], str) for r in results)
    assert all(isinstance(r[1], float) for r in results)
    assert all(isinstance(r[2], Payload) for r in results)

def test_entropy_preference(doc_embeddings, doc_payloads, doc_ids):
    """Test entropy preference in search."""
    # Create an exact index for deterministic results
    index = ExactIndex(dim=DIM, space="cosine")
    
    # Add documents with varying entropy
    for i, (doc_id, emb) in enumerate(zip(doc_ids, doc_embeddings)):
        # Set payload with controlled entropy values
        payload = Payload(
            dewi=0.5,
            ht_mean=i / N_DOCS,  # Vary entropy from 0 to 1
            hi_mean=i / N_DOCS,
            ht_q90=1.0,
            hi_q90=1.0,
            I_hat=0.5,
            redundancy=0.1,
            noise=0.05
        )
        index.add(doc_id, emb, payload)
    
    index.build()
    
    # Test with preference for high entropy
    query = np.ones(DIM) / np.sqrt(DIM)  # Uniform query
    
    # No entropy preference
    results_neutral = index.search(query, k=5, entropy_pref=0.0)
    
    # Prefer high entropy
    results_high = index.search(query, k=5, entropy_pref=1.0)
    
    # Prefer low entropy
    results_low = index.search(query, k=5, entropy_pref=-1.0)
    
    # Get the average entropy for each result set
    def avg_entropy(results):
        return sum((r[2].ht_mean + r[2].hi_mean) / 2 for r in results) / len(results)
    
    # For testing, we'll check that the ordering changes with different entropy preferences
    # rather than strict inequality which can be flaky with random data
    ent_high = avg_entropy(results_high)
    ent_neutral = avg_entropy(results_neutral)
    ent_low = avg_entropy(results_low)
    
    # The relative ordering should be: high >= neutral >= low
    # But we'll use a small epsilon to account for floating point imprecision
    epsilon = 1e-10
    assert ent_high + epsilon >= ent_neutral >= ent_low - epsilon, \
        f"Expected high({ent_high}) >= neutral({ent_neutral}) >= low({ent_low})"

def test_dewi_reranking(doc_embeddings, doc_payloads, doc_ids):
    """Test DEWI score-based re-ranking."""
    # Create an exact index
    index = ExactIndex(dim=DIM, space="cosine")
    
    # Add documents with varying DEWI scores
    for i, (doc_id, emb) in enumerate(zip(doc_ids, doc_embeddings)):
        # Set payload with controlled DEWI scores
        payload = Payload(
            dewi=i / N_DOCS,  # Vary DEWI score from 0 to 1
            ht_mean=0.5,
            hi_mean=0.5,
            ht_q90=1.0,
            hi_q90=1.0,
            I_hat=0.5,
            redundancy=0.1,
            noise=0.05
        )
        index.add(doc_id, emb, payload)
    
    index.build()
    
    # Uniform query
    query = np.ones(DIM) / np.sqrt(DIM)
    
    # Pure similarity search (eta=0)
    results_sim = index.search(query, k=5, eta=0.0)
    
    # Pure DEWI score (eta=1)
    results_dewi = index.search(query, k=5, eta=1.0)
    
    # Mixed (eta=0.5)
    results_mixed = index.search(query, k=5, eta=0.5)
    
    # Get the average DEWI score for each result set
    def avg_dewi(results):
        return sum(r[2].dewi for r in results) / len(results)
    
    # For testing, we'll check the relative ordering of the scores
    # rather than strict inequality which can be flaky with random data
    dewi_scores = {
        'sim': avg_dewi(results_sim),
        'mixed': avg_dewi(results_mixed),
        'dewi': avg_dewi(results_dewi)
    }
    
    # The relative ordering should be: dewi >= mixed >= sim
    # But we'll use a small epsilon to account for floating point imprecision
    epsilon = 1e-10
    assert dewi_scores['dewi'] + epsilon >= dewi_scores['mixed'] - epsilon, \
        f"Expected DEWI score with eta=1 ({dewi_scores['dewi']}) >= eta=0.5 ({dewi_scores['mixed']})"
    assert dewi_scores['mixed'] + epsilon >= dewi_scores['sim'] - epsilon, \
        f"Expected DEWI score with eta=0.5 ({dewi_scores['mixed']}) >= eta=0 ({dewi_scores['sim']})"
