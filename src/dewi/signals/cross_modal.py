import torch
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
from transformers import CLIPModel, CLIPProcessor
from PIL import Image

@dataclass
class CrossModalResult:
    info_nce_score: float  # Estimated mutual information lower bound
    text_to_image_sim: float  # Text-to-image similarity
    image_to_text_sim: float  # Image-to-text similarity
    
class CrossModalDependency:
    """Estimates cross-modal dependency between text and image using CLIP."""
    
    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: Optional[str] = None,
    ):
        """Initialize the cross-modal dependency estimator.
        
        Args:
            model_name: Name of the CLIP model to use.
            device: Device to run the model on ('cuda' or 'cpu').
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()
        
    def compute_similarity(
        self,
        text: str,
        image: Image.Image,
    ) -> CrossModalResult:
        """Compute cross-modal similarity between text and image.
        
        Args:
            text: Input text.
            image: Input image (PIL Image).
            
        Returns:
            CrossModalResult containing similarity metrics.
        """
        # Process inputs
        inputs = self.processor(
            text=[text],
            images=image,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,  # CLIP's max length
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            
            # Get normalized embeddings
            text_embeds = outputs.text_embeds
            image_embeds = outputs.image_embeds
            
            # Compute cosine similarities
            sim = F.cosine_similarity(text_embeds, image_embeds, dim=1)
            
            # For InfoNCE-style MI lower bound, we'd normally need negative samples
            # Here we just return the similarity as a proxy
            info_nce_score = sim.item()
            
            # Also compute text-to-image and image-to-text similarities
            # In this simple case, they're the same since we're using dot product
            text_to_image_sim = sim.item()
            image_to_text_sim = sim.item()
        
        return CrossModalResult(
            info_nce_score=info_nce_score,
            text_to_image_sim=text_to_image_sim,
            image_to_text_sim=image_to_text_sim,
        )
    
    def batch_compute(
        self,
        texts: List[str],
        images: List[Image.Image],
        batch_size: int = 8,
    ) -> List[CrossModalResult]:
        """Compute cross-modal dependencies for a batch of text-image pairs.
        
        Args:
            texts: List of input texts.
            images: List of input images (PIL Images).
            batch_size: Batch size for processing.
            
        Returns:
            List of CrossModalResult objects, one per input pair.
        """
        assert len(texts) == len(images), "Number of texts and images must match"
        
        results = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_images = images[i:i + batch_size]
            
            # Process batch
            inputs = self.processor(
                text=batch_texts,
                images=batch_images,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77,
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                
                # Get normalized embeddings
                text_embeds = outputs.text_embeds
                image_embeds = outputs.image_embeds
                
                # Compute cosine similarities
                sim = F.cosine_similarity(
                    text_embeds.unsqueeze(1),
                    image_embeds.unsqueeze(0),
                    dim=2
                )
                
                # For each item, get the diagonal (matching pairs)
                batch_sims = torch.diag(sim).cpu().numpy()
                
                # Create results
                for sim_score in batch_sims:
                    results.append(CrossModalResult(
                        info_nce_score=float(sim_score),
                        text_to_image_sim=float(sim_score),
                        image_to_text_sim=float(sim_score),
                    ))
        
        return results
    
    def __call__(
        self,
        text: str,
        image: Image.Image,
    ) -> CrossModalResult:
        """Alias for compute_similarity for easier function-like usage."""
        return self.compute_similarity(text, image)
