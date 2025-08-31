import torch
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Union, Tuple
from PIL import Image
import torchvision.transforms as T
from transformers import AutoModel, AutoFeatureExtractor

@dataclass
class ImageEntropyResult:
    patch_entropies: np.ndarray  # 2D array of patch entropies
    mean_entropy: float
    q90_entropy: float
    q99_entropy: float

class ImageEntropyEstimator:
    """Estimates image entropy using various proxy methods.
    
    Currently implements:
    - MAE (Masked Autoencoder) reconstruction error
    - Can be extended with other methods like VAE or flow-based models
    """
    
    def __init__(
        self,
        model_name: str = "facebook/vit-mae-base",
        device: Optional[str] = None,
        patch_size: int = 16,
    ):
        """Initialize the image entropy estimator.
        
        Args:
            model_name: Pretrained model to use for reconstruction.
            device: Device to run the model on ('cuda' or 'cpu').
            patch_size: Size of patches for entropy computation.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.patch_size = patch_size
        
        # Load model and feature extractor
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        self.model.eval()
        
        # Image transformations
        self.transform = T.Compose([
            T.Resize((224, 224)),  # Standard size for most ViT models
            T.ToTensor(),
            T.Normalize(mean=self.feature_extractor.image_mean,
                       std=self.feature_extractor.image_std),
        ])
    
    def preprocess_image(self, image: Union[str, Image.Image]) -> torch.Tensor:
        """Preprocess an image for the model."""
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        return self.transform(image).unsqueeze(0).to(self.device)
    
    def compute_mae_reconstruction_error(
        self,
        image: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute MAE reconstruction error.
        
        Returns:
            Tuple of (reconstruction_error, patch_errors)
        """
        with torch.no_grad():
            # Forward pass through MAE
            outputs = self.model(
                pixel_values=image,
                output_hidden_states=True,
                return_dict=True
            )
            
            # Get the reconstructed pixel values
            last_hidden = outputs.last_hidden_state
            
            # Project back to pixel space
            # This is a simplified version - actual implementation depends on MAE architecture
            # You may need to adjust this based on the specific MAE variant
            patch_embeddings = last_hidden[:, 1:, :]  # Remove [CLS] token
            batch_size, num_patches, _ = patch_embeddings.shape
            patch_size = self.model.config.patch_size
            
            # Reshape to image patches
            h = w = int(np.sqrt(num_patches))
            patch_embeddings = patch_embeddings.reshape(batch_size, h, w, -1)
            
            # Compute reconstruction error per patch
            # This is a placeholder - actual implementation would compare with original patches
            patch_errors = torch.mean(torch.square(patch_embeddings), dim=-1)
            
            # Compute overall reconstruction error
            reconstruction_error = torch.mean(patch_errors)
            
            return reconstruction_error, patch_errors
    
    def compute_entropy(
        self,
        image: Union[str, Image.Image, torch.Tensor],
        method: str = "mae",
    ) -> ImageEntropyResult:
        """Compute image entropy using the specified method.
        
        Args:
            image: Input image (path, PIL Image, or tensor).
            method: Method to use for entropy estimation ('mae' or 'kde').
            
        Returns:
            ImageEntropyResult containing patch and aggregate entropy metrics.
        """
        if not isinstance(image, torch.Tensor):
            image = self.preprocess_image(image)
        
        if method == "mae":
            _, patch_errors = self.compute_mae_reconstruction_error(image)
            patch_entropies = patch_errors.squeeze().cpu().numpy()
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        # Compute aggregate statistics
        mean_entropy = float(np.mean(patch_entropies))
        q90_entropy = float(np.quantile(patch_entropies, 0.9))
        q99_entropy = float(np.quantile(patch_entropies, 0.99))
        
        return ImageEntropyResult(
            patch_entropies=patch_entropies,
            mean_entropy=mean_entropy,
            q90_entropy=q90_entropy,
            q99_entropy=q99_entropy,
        )
    
    def batch_compute(
        self,
        images: List[Union[str, Image.Image, torch.Tensor]],
        method: str = "mae",
        batch_size: int = 8,
    ) -> List[ImageEntropyResult]:
        """Compute entropy for a batch of images.
        
        Args:
            images: List of input images.
            method: Method to use for entropy estimation.
            batch_size: Batch size for processing.
            
        Returns:
            List of ImageEntropyResult objects, one per input image.
        """
        results = []
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            batch_tensors = [self.preprocess_image(img) if not isinstance(img, torch.Tensor) 
                           else img for img in batch]
            batch_tensors = torch.cat(batch_tensors, dim=0)
            
            if method == "mae":
                _, patch_errors = self.compute_mae_reconstruction_error(batch_tensors)
                for j in range(len(batch)):
                    patch_entropies = patch_errors[j].squeeze().cpu().numpy()
                    
                    mean_entropy = float(np.mean(patch_entropies))
                    q90_entropy = float(np.quantile(patch_entropies, 0.9))
                    q99_entropy = float(np.quantile(patch_entropies, 0.99))
                    
                    results.append(ImageEntropyResult(
                        patch_entropies=patch_entropies,
                        mean_entropy=mean_entropy,
                        q90_entropy=q90_entropy,
                        q99_entropy=q99_entropy,
                    ))
        
        return results
    
    def __call__(
        self,
        image: Union[str, Image.Image, torch.Tensor],
        **kwargs
    ) -> ImageEntropyResult:
        """Alias for compute_entropy for easier function-like usage."""
        return self.compute_entropy(image, **kwargs)
