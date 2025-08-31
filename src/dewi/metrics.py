"""Evaluation metrics for DEWI retrieval and analysis."""

from __future__ import annotations
from typing import Dict, List, Sequence, Tuple, Union
import numpy as np

# === Ranking Metrics ===

def recall_at_k(
    ground_truth: Dict[str, Sequence[str]], 
    rankings: Dict[str, Sequence[str]], 
    k: int = 10
) -> float:
    """Compute Recall@k for a set of queries.
    
    Args:
        ground_truth: Dict mapping query_id -> list of relevant doc_ids
        rankings: Dict mapping query_id -> ranked list of retrieved doc_ids
        k: Number of top results to consider
        
    Returns:
        Average recall@k across all queries
    """
    hits = 0
    total = len(ground_truth)
    
    for q, relevant_docs in ground_truth.items():
        if q not in rankings:
            continue
            
        # Get top-k retrieved documents
        retrieved = set(rankings[q][:k])
        # Count how many relevant docs were retrieved
        hits += len(set(relevant_docs) & retrieved)
    
    return hits / max(sum(len(docs) for docs in ground_truth.values()), 1)

def dcg_at_k(relevance_scores: Sequence[float]) -> float:
    """Compute Discounted Cumulative Gain at position k.
    
    Args:
        relevance_scores: List of relevance scores in rank order
        
    Returns:
        DCG score
    """
    relevance = np.asarray(relevance_scores, dtype=np.float32)
    # DCG = sum(rel_i / log2(i + 1))
    gains = (2 ** relevance - 1) / np.log2(np.arange(2, len(relevance) + 2))
    return float(np.sum(gains))

def ndcg_at_k(
    ground_truth: Dict[str, Dict[str, int]], 
    rankings: Dict[str, Sequence[str]], 
    k: int = 10
) -> float:
    """Compute Normalized Discounted Cumulative Gain at position k.
    
    Args:
        ground_truth: Dict mapping query_id -> {doc_id: relevance_score}
        rankings: Dict mapping query_id -> ranked list of retrieved doc_ids
        k: Number of top results to consider
        
    Returns:
        Average nDCG@k across all queries
    """
    scores = []
    
    for q, rel_map in ground_truth.items():
        if q not in rankings:
            continue
            
        # Get relevance scores for retrieved documents
        retrieved = rankings[q][:k]
        rel = [rel_map.get(doc_id, 0) for doc_id in retrieved]
        
        # Compute DCG for retrieved ranking
        dcg = dcg_at_k(rel)
        
        # Compute ideal DCG
        ideal = sorted(rel_map.values(), reverse=True)[:k]
        idcg = dcg_at_k(ideal) if ideal else 1.0
        
        # Avoid division by zero
        scores.append(dcg / (idcg + 1e-8))
    
    return float(np.mean(scores)) if scores else 0.0

def mrr(
    ground_truth: Dict[str, Sequence[str]], 
    rankings: Dict[str, Sequence[str]]
) -> float:
    """Compute Mean Reciprocal Rank.
    
    Args:
        ground_truth: Dict mapping query_id -> list of relevant doc_ids
        rankings: Dict mapping query_id -> ranked list of retrieved doc_ids
        
    Returns:
        MRR score
    """
    reciprocal_ranks = []
    
    for q, relevant_docs in ground_truth.items():
        if q not in rankings:
            continue
            
        # Find position of first relevant document
        for i, doc_id in enumerate(rankings[q], 1):
            if doc_id in relevant_docs:
                reciprocal_ranks.append(1.0 / i)
                break
        else:
            # No relevant document found
            reciprocal_ranks.append(0.0)
    
    return float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0

# === Entropy Analysis ===

def stratify_by_dewi(
    bins: Sequence[float], 
    doc_dewi: Dict[str, float], 
    rankings: Dict[str, Sequence[str]]
) -> Dict[Tuple[float, float], float]:
    """Analyze distribution of DEWI scores in retrieval results.
    
    Args:
        bins: Sorted DEWI score boundaries (e.g., [0.0, 0.33, 0.66, 1.0])
        doc_dewi: Dict mapping doc_id -> DEWI score
        rankings: Dict mapping query_id -> ranked list of retrieved doc_ids
        
    Returns:
        Dict mapping (lower, upper) bin boundaries -> proportion of results in that bin
    """
    if not bins or len(bins) < 2:
        raise ValueError("At least two bin boundaries required")
        
    # Initialize bin counts
    bin_edges = list(bins)
    bin_ranges = [(bin_edges[i], bin_edges[i+1]) for i in range(len(bin_edges)-1)]
    bin_counts = {r: 0 for r in bin_ranges}
    total = 0
    
    # Count documents in each bin
    for docs in rankings.values():
        for doc_id in docs:
            dewi = doc_dewi.get(doc_id, 0.0)
            total += 1
            
            # Find the appropriate bin
            for i in range(len(bin_edges)-1):
                lower, upper = bin_edges[i], bin_edges[i+1]
                if i == len(bin_edges) - 2:  # Last bin is inclusive on both sides
                    if lower <= dewi <= upper:
                        bin_counts[(lower, upper)] += 1
                        break
                else:
                    if lower <= dewi < upper:
                        bin_counts[(lower, upper)] += 1
                        break
    
    # Convert counts to proportions
    proportions = {
        k: (v / total) if total > 0 else 0.0 
        for k, v in bin_counts.items()
    }
    
    return proportions

# === Redundancy & Diversity ===

def duplicate_rate(clusters: List[Sequence[str]]) -> float:
    """Compute the duplicate rate in a set of near-duplicate clusters.
    
    Args:
        clusters: List of clusters, where each cluster is a sequence of doc_ids
        
    Returns:
        Proportion of documents that are duplicates (1.0 - singleton_rate)
    """
    if not clusters:
        return 0.0
        
    total_docs = sum(len(cluster) for cluster in clusters)
    singleton_count = sum(1 for c in clusters if len(c) == 1)
    
    if total_docs == 0:
        return 0.0
        
    return 1.0 - (singleton_count / len(clusters))

def cluster_coverage(
    selected: Sequence[str], 
    clusters: List[Sequence[str]]
) -> float:
    """Compute the proportion of clusters covered by the selected documents.
    
    Args:
        selected: List of selected document IDs
        clusters: List of clusters, where each cluster is a sequence of doc_ids
        
    Returns:
        Proportion of clusters with at least one selected document
    """
    if not clusters:
        return 0.0
        
    selected_set = set(selected)
    covered = sum(1 for cluster in clusters if any(doc in selected_set for doc in cluster))
    
    return covered / len(clusters)
