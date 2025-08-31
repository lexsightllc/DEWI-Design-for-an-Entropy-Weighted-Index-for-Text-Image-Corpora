#!/usr/bin/env python3
"""
Profile the DEWI index for performance bottlenecks.

This script helps identify performance bottlenecks in the DEWI index by:
1. Creating a synthetic dataset
2. Profiling index construction
3. Profiling search operations
4. Generating detailed performance reports

Example usage:
    python scripts/profile_index.py --n-docs 100000 --dim 256 --n-queries 1000 --output profile_results/
"""

import argparse
import cProfile
import io
import os
import pstats
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm

# Add parent directory to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from dewi.index import DewiIndex, Payload


def generate_synthetic_data(
    n_docs: int = 10000,
    dim: int = 256,
    seed: int = 42
) -> Tuple[List[str], np.ndarray, List[Payload]]:
    """Generate synthetic data for profiling.
    
    Args:
        n_docs: Number of documents to generate
        dim: Dimensionality of embeddings
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (doc_ids, embeddings, payloads)
    """
    rng = np.random.RandomState(seed)
    
    # Generate document IDs
    doc_ids = [f"doc_{i:08d}" for i in range(n_docs)]
    
    # Generate random embeddings (normalized to unit length)
    embeddings = rng.randn(n_docs, dim).astype(np.float32)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    # Generate random payloads with realistic distributions
    payloads = []
    for _ in range(n_docs):
        payloads.append(Payload(
            dewi=float(np.clip(rng.beta(2, 2), 0, 1)),
            ht_mean=float(rng.gamma(2, 0.5)),
            ht_q90=float(rng.gamma(2, 0.5) * 1.5),
            hi_mean=float(rng.gamma(2, 0.3)),
            hi_q90=float(rng.gamma(2, 0.3) * 1.5),
            I_hat=float(rng.beta(2, 2)),
            redundancy=float(rng.beta(1, 5)),
            noise=float(rng.beta(1, 10))
        ))
    
    return doc_ids, embeddings, payloads


def build_index(
    doc_ids: List[str],
    embeddings: np.ndarray,
    payloads: List[Payload],
    space: str = "cosine",
    m: int = 16,
    ef_construction: int = 200
) -> DewiIndex:
    """Build and return a DEWI index.
    
    Args:
        doc_ids: List of document IDs
        embeddings: Embedding matrix (n_docs x dim)
        payloads: List of Payload objects
        space: Distance metric ('cosine' or 'l2')
        m: HNSW M parameter (number of bi-directional links)
        ef_construction: HNSW ef_construction parameter
        
    Returns:
        Populated DEWI index
    """
    assert len(doc_ids) == len(embeddings) == len(payloads)
    dim = embeddings.shape[1]
    
    index = DewiIndex(
        dim=dim,
        space=space,
        M=m,
        ef_construction=ef_construction,
        use_ann=True
    )
    
    # Add documents in batches
    batch_size = 10000
    for i in tqdm(range(0, len(doc_ids), batch_size), desc="Indexing"):
        batch_ids = doc_ids[i:i+batch_size]
        batch_embs = embeddings[i:i+batch_size]
        batch_payloads = payloads[i:i+batch_size]
        
        for doc_id, emb, payload in zip(batch_ids, batch_embs, batch_payloads):
            index.add(doc_id, emb, payload)
    
    # Build the index
    index.build()
    return index


def profile_index_construction(
    n_docs: int = 100000,
    dim: int = 256,
    output_dir: Path = Path("profile_results")
) -> Dict[str, float]:
    """Profile index construction.
    
    Returns:
        Dictionary of timing metrics
    """
    print(f"\n{'='*80}")
    print(f"Profiling index construction with {n_docs:,} documents (dim={dim})")
    print(f"{'='*80}")
    
    # Generate data
    start = time.time()
    doc_ids, embeddings, payloads = generate_synthetic_data(n_docs, dim)
    data_gen_time = time.time() - start
    print(f"Generated {n_docs} documents in {data_gen_time:.2f}s")
    
    # Profile index construction
    profiler = cProfile.Profile()
    profiler.enable()
    
    build_start = time.time()
    index = build_index(doc_ids, embeddings, payloads)
    build_time = time.time() - build_start
    
    profiler.disable()
    
    # Save profile results
    output_dir.mkdir(parents=True, exist_ok=True)
    profile_file = output_dir / f"build_n{n_docs}_d{dim}.prof"
    profiler.dump_stats(str(profile_file))
    
    # Print top functions by time
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(20)
    print("\nTop functions by cumulative time:")
    print(s.getvalue())
    
    # Save to file
    with open(output_dir / f"build_n{n_docs}_d{dim}_top.txt", "w") as f:
        f.write(s.getvalue())
    
    return {
        "n_docs": n_docs,
        "dim": dim,
        "data_generation_time": data_gen_time,
        "index_construction_time": build_time,
        "docs_per_second": n_docs / build_time,
        "profile_file": str(profile_file)
    }


def profile_search(
    index: DewiIndex,
    n_queries: int = 1000,
    k: int = 10,
    output_dir: Path = Path("profile_results")
) -> Dict[str, float]:
    """Profile search operations.
    
    Returns:
        Dictionary of timing metrics
    """
    print(f"\n{'='*80}")
    print(f"Profiling search with {n_queries:,} queries (k={k})")
    print(f"{'='*80}")
    
    # Generate random queries
    dim = index.dim
    queries = np.random.randn(n_queries, dim).astype(np.float32)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)
    
    # Warm-up
    for q in queries[:10]:
        _ = index.search(q, k=k)
    
    # Profile search
    profiler = cProfile.Profile()
    profiler.enable()
    
    search_start = time.time()
    for q in tqdm(queries, desc="Searching"):
        _ = index.search(q, k=k)
    search_time = time.time() - search_start
    
    profiler.disable()
    
    # Save profile results
    output_dir.mkdir(parents=True, exist_ok=True)
    profile_file = output_dir / f"search_q{n_queries}_k{k}.prof"
    profiler.dump_stats(str(profile_file))
    
    # Print top functions by time
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(20)
    print("\nTop functions by cumulative time:")
    print(s.getvalue())
    
    # Save to file
    with open(output_dir / f"search_q{n_queries}_k{k}_top.txt", "w") as f:
        f.write(s.getvalue())
    
    return {
        "n_queries": n_queries,
        "k": k,
        "total_search_time": search_time,
        "queries_per_second": n_queries / search_time,
        "latency_ms": (search_time / n_queries) * 1000,
        "profile_file": str(profile_file)
    }


def main():
    parser = argparse.ArgumentParser(description="Profile DEWI index performance")
    parser.add_argument("--n-docs", type=int, default=100000,
                        help="Number of documents to index")
    parser.add_argument("--dim", type=int, default=256,
                        help="Dimensionality of embeddings")
    parser.add_argument("--n-queries", type=int, default=1000,
                        help="Number of search queries to profile")
    parser.add_argument("--k", type=int, default=10,
                        help="Number of results to return per query")
    parser.add_argument("--output", type=str, default="profile_results",
                        help="Output directory for profile results")
    parser.add_argument("--skip-build", action="store_true",
                        help="Skip index construction profiling")
    parser.add_argument("--skip-search", action="store_true",
                        help="Skip search profiling")
    
    args = parser.parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Track all metrics
    metrics = {}
    index = None
    
    try:
        # Profile index construction
        if not args.skip_build:
            build_metrics = profile_index_construction(
                n_docs=args.n_docs,
                dim=args.dim,
                output_dir=output_dir
            )
            metrics["build"] = build_metrics
            
            # Load the built index for search profiling
            doc_ids, embeddings, payloads = generate_synthetic_data(args.n_docs, args.dim)
            index = build_index(doc_ids, embeddings, payloads)
        
        # Profile search
        if not args.skip_search and index is not None:
            search_metrics = profile_search(
                index=index,
                n_queries=args.n_queries,
                k=args.k,
                output_dir=output_dir
            )
            metrics["search"] = search_metrics
        
        # Save combined metrics
        metrics_file = output_dir / "metrics.json"
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)
        
        print(f"\nProfile results saved to: {output_dir.absolute()}")
        print(f"To view profile results, run: snakeviz {output_dir}/*.prof")
        
    except Exception as e:
        print(f"Error during profiling: {e}", file=sys.stderr)
        raise


if __name__ == "__main__":
    import json
    from pathlib import Path
    
    main()
