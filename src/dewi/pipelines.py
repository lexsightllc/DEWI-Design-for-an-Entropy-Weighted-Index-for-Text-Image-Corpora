"""Signal computation pipelines for DEWI."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from tqdm import tqdm

from dewi.config import DewiConfig
from dewi.types import Weights
from dewi.signals.text_entropy import TextEntropyEstimator
from dewi.signals.image_entropy import ImageEntropyEstimator
from dewi.signals.cross_modal import CrossModalDependency
from dewi.signals.redundancy import RedundancyEstimator
from dewi.signals.noise import NoiseEstimator

@dataclass
class Document:
    """Container for document data and signals."""
    doc_id: str
    text: Optional[str] = None
    image_path: Optional[Union[str, Path]] = None
    embedding: Optional[np.ndarray] = None
    
    # Signal values
    ht_mean: Optional[float] = None
    ht_q90: Optional[float] = None
    hi_mean: Optional[float] = None
    hi_q90: Optional[float] = None
    I_hat: Optional[float] = None
    redundancy: Optional[float] = None
    noise: Optional[float] = None
    dewi_score: Optional[float] = None

class DewiPipeline:
    """Pipeline for computing DEWI signals and scores."""
    
    def __init__(self, config: Optional[DewiConfig] = None):
        """Initialize the pipeline with configuration."""
        self.config = config or DewiConfig()
        self._init_components()
    
    def _init_components(self):
        """Initialize signal computation components."""
        # Text entropy estimator
        self.text_entropy = TextEntropyEstimator(
            model_name=self.config.text.model,
            quantiles=self.config.text.quantiles,
            batch_size=self.config.text.batch_size,
            device=self._get_device()
        )
        
        # Image entropy estimator
        self.image_entropy = ImageEntropyEstimator(
            model_name=self.config.image.model,
            patch_size=self.config.image.patch_size,
            batch_size=self.config.image.batch_size,
            device=self._get_device()
        )
        
        # Cross-modal dependency
        self.cross_modal = CrossModalDependency(
            model_name=self.config.cross_modal.model,
            batch_size=self.config.cross_modal.batch_size,
            device=self._get_device()
        )
        
        # Redundancy estimator
        self.redundancy = RedundancyEstimator(
            text_sim=self.config.redundancy.text_sim,
            image_sim=self.config.redundancy.image_sim,
            cross_modal_density=self.config.redundancy.cross_modal_density
        )
        
        # Noise estimator
        self.noise = NoiseEstimator(
            blur_threshold=self.config.noise.blur_threshold,
            ocr_min_coverage=self.config.noise.ocr_min_coverage,
            nsfw_filter=self.config.noise.nsfw_filter
        )
    
    def _get_device(self) -> torch.device:
        """Get the appropriate device (GPU if available)."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def compute_signals(
        self, 
        documents: List[Document],
        progress: bool = True
    ) -> List[Document]:
        """Compute all signals for a batch of documents.
        
        Args:
            documents: List of documents to process
            progress: Whether to show progress bars
            
        Returns:
            List of processed documents with computed signals
        """
        if not documents:
            return []
        
        # Filter documents with text for text processing
        text_docs = [d for d in documents if d.text]
        if text_docs and progress:
            print("Computing text entropy...")
        for i in tqdm(range(0, len(text_docs), self.text_entropy.batch_size), disable=not progress):
            batch = text_docs[i:i+self.text_entropy.batch_size]
            texts = [d.text for d in batch]
            results = self.text_entropy.batch_compute(texts)
            
            for doc, res in zip(batch, results):
                doc.ht_mean = res["mean"]
                doc.ht_q90 = res.get("quantiles", {}).get(0.9)
        
        # Filter documents with images for image processing
        image_docs = [d for d in documents if d.image_path]
        if image_docs and progress:
            print("Computing image entropy...")
        for i in tqdm(range(0, len(image_docs), self.image_entropy.batch_size), disable=not progress):
            batch = image_docs[i:i+self.image_entropy.batch_size]
            image_paths = [str(d.image_path) for d in batch]
            results = self.image_entropy.batch_compute(image_paths)
            
            for doc, res in zip(batch, results):
                doc.hi_mean = res["mean"]
                doc.hi_q90 = res.get("quantiles", {}).get(0.9)
        
        # Compute cross-modal dependencies for documents with both text and image
        multimodal_docs = [d for d in documents if d.text and d.image_path]
        if multimodal_docs and progress:
            print("Computing cross-modal dependencies...")
        for i in tqdm(range(0, len(multimodal_docs), self.cross_modal.batch_size), disable=not progress):
            batch = multimodal_docs[i:i+self.cross_modal.batch_size]
            texts = [d.text for d in batch]
            image_paths = [str(d.image_path) for d in batch]
            
            # Get embeddings and compute MI
            _, _, mi_scores = self.cross_modal.batch_compute(texts, image_paths)
            
            for doc, mi in zip(batch, mi_scores):
                doc.I_hat = mi
        
        # Compute redundancy scores
        if progress:
            print("Computing redundancy...")
        self.redundancy.fit(documents)
        redundancy_scores = self.redundancy.score(documents)
        for doc, score in zip(documents, redundancy_scores):
            doc.redundancy = score
        
        # Compute noise scores
        if progress:
            print("Computing noise...")
        noise_scores = self.noise.score([d.image_path for d in documents])
        for doc, score in zip(documents, noise_scores):
            doc.noise = score
        
        return documents
    
    def compute_dewi_scores(
        self, 
        documents: List[Document],
        weights: Optional[Dict[str, float]] = None,
        delta: Optional[float] = None,
        mode: Optional[str] = None
    ) -> List[Document]:
        """Compute DEWI scores for documents with computed signals.
        
        Args:
            documents: List of documents with computed signals
            weights: Optional override for scoring weights
            delta: Optional override for delta parameter
            mode: Scoring mode ('standard' or 'conditional')
            
        Returns:
            List of documents with computed DEWI scores
        """
        from dewi.scorer import DewiScorer
        
        # Use provided weights or config defaults
        if weights is None:
            weights = {
                'alpha_t': self.config.scoring.weights.alpha_t,
                'alpha_i': self.config.scoring.weights.alpha_i,
                'alpha_m': self.config.scoring.weights.alpha_m,
                'alpha_r': self.config.scoring.weights.alpha_r,
                'alpha_n': self.config.scoring.weights.alpha_n
            }
        
        # Initialize scorer
        scorer = DewiScorer(
            weights=Weights(**weights) if isinstance(weights, dict) else weights,
            delta=delta or self.config.scoring.delta,
        )
        scoring_mode = mode or self.config.scoring.mode
        
        # Collect signals
        signals = []
        for doc in documents:
            signals.append({
                'ht_mean': doc.ht_mean or 0.0,
                'ht_q90': doc.ht_q90 or 0.0,
                'hi_mean': doc.hi_mean or 0.0,
                'hi_q90': doc.hi_q90 or 0.0,
                'I_hat': doc.I_hat or 0.0,
                'redundancy': doc.redundancy or 0.0,
                'noise': doc.noise or 0.0
            })
        
        # Fit scorer statistics if not already fitted
        if not scorer.is_fitted():
            scorer.fit_stats(signals)
        
        # Compute scores based on mode
        for doc, sig in zip(documents, signals):
            if scoring_mode == "conditional":
                doc.dewi_score = scorer.score_conditional(sig)
            else:
                doc.dewi_score = scorer.score(sig)
        
        return documents

def create_document(
    doc_id: str,
    text: Optional[str] = None,
    image_path: Optional[Union[str, Path]] = None,
    embedding: Optional[np.ndarray] = None
) -> Document:
    """Create a document with the given data."""
    return Document(
        doc_id=doc_id,
        text=text,
        image_path=str(image_path) if image_path else None,
        embedding=embedding
    )
