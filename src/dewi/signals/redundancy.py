import numpy as np
import torch
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Union
from datasketch import MinHash, MinHashLSH
from PIL import Image
from transformers import AutoFeatureExtractor, AutoModel
import torch.nn.functional as F

@dataclass
class RedundancyResult:
    text_duplicates: List[int]  # Indices of duplicate texts
    image_duplicates: List[int]  # Indices of duplicate images
    text_similarity: np.ndarray  # Pairwise text similarity matrix
    image_similarity: np.ndarray  # Pairwise image similarity matrix
    cross_modal_similarity: np.ndarray  # Text-to-image similarity matrix

class RedundancyEstimator:
    """Estimates redundancy within and across text and image modalities."""
    
    def __init__(
        self,
        text_threshold: float = 0.9,
        image_threshold: float = 0.9,
        cross_modal_threshold: float = 0.8,
        device: Optional[str] = None,
    ):
        """Initialize the redundancy estimator.
        
        Args:
            text_threshold: Jaccard similarity threshold for text deduplication.
            image_threshold: Similarity threshold for image deduplication.
            cross_modal_threshold: Threshold for cross-modal near-duplicates.
            device: Device to run the model on ('cuda' or 'cpu').
        """
        self.text_threshold = text_threshold
        self.image_threshold = image_threshold
        self.cross_modal_threshold = cross_modal_threshold
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize LSH for efficient near-duplicate detection
        self.text_lsh = MinHashLSH(threshold=text_threshold, num_perm=128)
        self.image_lsh = MinHashLSH(threshold=image_threshold, num_perm=128)
        
        # Model for image feature extraction
        self.image_model = None
        self.image_processor = None
    
    def _init_image_model(self):
        """Lazy initialization of image model."""
        if self.image_model is None:
            model_name = "google/vit-base-patch16-224"
            self.image_processor = AutoFeatureExtractor.from_pretrained(model_name)
            self.image_model = AutoModel.from_pretrained(model_name).to(self.device)
            self.image_model.eval()
    
    def compute_text_similarity(
        self,
        texts: List[str],
        n_grams: int = 3,
    ) -> Tuple[np.ndarray, Dict[int, List[int]]]:
        """Compute text similarity using MinHash and LSH.
        
        Args:
            texts: List of input texts.
            n_grams: Size of n-grams for MinHash.
            
        Returns:
            Tuple of (similarity_matrix, duplicate_groups)
        """
        # Create MinHash for each text
        minhashes = []
        for i, text in enumerate(texts):
            mh = MinHash(num_perm=128)
            # Split into n-grams
            words = text.lower().split()
            for j in range(len(words) - n_grams + 1):
                ngram = ' '.join(words[j:j + n_grams])
                mh.update(ngram.encode('utf-8'))
            minhashes.append((i, mh))
        
        # Build LSH index
        lsh = MinHashLSH(threshold=self.text_threshold, num_perm=128)
        for idx, mh in minhashes:
            lsh.insert(str(idx), mh)
        
        # Find duplicates
        duplicate_groups = {}
        for idx, mh in minhashes:
            duplicates = lsh.query(mh)
            if len(duplicates) > 1:  # At least one other duplicate
                group = sorted([int(d) for d in duplicates])
                group_key = tuple(group)
                if group_key not in duplicate_groups:
                    duplicate_groups[group_key] = group
        
        # Compute full similarity matrix
        n = len(texts)
        similarity_matrix = np.eye(n)  # Diagonal is 1 (self-similarity)
        
        for i in range(n):
            for j in range(i + 1, n):
                similarity = minhashes[i][1].jaccard(minhashes[j][1])
                similarity_matrix[i, j] = similarity
                similarity_matrix[j, i] = similarity
        
        return similarity_matrix, list(duplicate_groups.values())
    
    def compute_image_similarity(
        self,
        images: List[Image.Image],
    ) -> Tuple[np.ndarray, Dict[int, List[int]]]:
        """Compute image similarity using deep features.
        
        Args:
            images: List of input images.
            
        Returns:
            Tuple of (similarity_matrix, duplicate_groups)
        """
        self._init_image_model()
        
        # Preprocess images
        inputs = self.image_processor(images=images, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Extract features
        with torch.no_grad():
            outputs = self.image_model(**inputs)
            features = outputs.last_hidden_state.mean(dim=1)  # [batch_size, hidden_size]
            features = F.normalize(features, p=2, dim=1)
        
        # Compute cosine similarity
        similarity_matrix = torch.mm(features, features.T).cpu().numpy()
        
        # Find duplicates
        duplicate_groups = {}
        n = len(images)
        
        for i in range(n):
            duplicates = [j for j in range(n) 
                        if i != j and similarity_matrix[i, j] >= self.image_threshold]
            if duplicates:
                group = sorted([i] + duplicates)
                group_key = tuple(group)
                if group_key not in duplicate_groups:
                    duplicate_groups[group_key] = group
        
        return similarity_matrix, list(duplicate_groups.values())
    
    def compute_cross_modal_similarity(
        self,
        texts: List[str],
        images: List[Image.Image],
    ) -> np.ndarray:
        """Compute cross-modal similarity between texts and images.
        
        Args:
            texts: List of input texts.
            images: List of input images.
            
        Returns:
            Cross-modal similarity matrix of shape (n_texts, n_images)
        """
        self._init_image_model()
        
        # Get text features (using the same model for simplicity)
        text_inputs = self.image_processor(
            text=texts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=77,
        )
        text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
        
        # Get image features
        image_inputs = self.image_processor(
            images=images, 
            return_tensors="pt", 
            padding=True
        )
        image_inputs = {k: v.to(self.device) for k, v in image_inputs.items()}
        
        with torch.no_grad():
            # Get text features
            text_outputs = self.image_model.get_text_features(**text_inputs)
            text_features = F.normalize(text_outputs, p=2, dim=1)
            
            # Get image features
            image_outputs = self.image_model.get_image_features(**image_inputs)
            image_features = F.normalize(image_outputs, p=2, dim=1)
            
            # Compute similarity
            similarity = torch.mm(text_features, image_features.T)
        
        return similarity.cpu().numpy()
    
    def compute_redundancy(
        self,
        texts: List[str],
        images: List[Image.Image],
    ) -> RedundancyResult:
        """Compute redundancy metrics for a batch of text-image pairs.
        
        Args:
            texts: List of input texts.
            images: List of input images.
            
        Returns:
            RedundancyResult containing redundancy information.
        """
        assert len(texts) == len(images), "Number of texts and images must match"
        
        # Compute within-modality similarities
        text_sim, text_duplicates = self.compute_text_similarity(texts)
        image_sim, image_duplicates = self.compute_image_similarity(images)
        
        # Compute cross-modal similarity
        cross_modal_sim = self.compute_cross_modal_similarity(texts, images)
        
        return RedundancyResult(
            text_duplicates=text_duplicates,
            image_duplicates=image_duplicates,
            text_similarity=text_sim,
            image_similarity=image_sim,
            cross_modal_similarity=cross_modal_sim,
        )
    
    def __call__(
        self,
        texts: List[str],
        images: List[Image.Image],
    ) -> RedundancyResult:
        """Alias for compute_redundancy for easier function-like usage."""
        return self.compute_redundancy(texts, images)
