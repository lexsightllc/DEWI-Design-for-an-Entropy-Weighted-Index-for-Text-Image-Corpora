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
    
    def _patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        """Convert images to patches.
        
        Args:
            imgs: (N, 3, H, W) tensor of images
            
        Returns:
            (N, L, patch_size**2 * 3) tensor of patches
        """
        p = self.model.config.patch_size
        assert imgs.shape[2] % p == 0 and imgs.shape[3] % p == 0, 'Image dimensions must be divisible by patch size.'
        
        h = imgs.shape[2] // p
        w = imgs.shape[3] // p
        
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        patches = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return patches
        
    def _unpatchify(self, patches: torch.Tensor) -> torch.Tensor:
        """Convert patches back to images.
        
        Args:
            patches: (N, L, patch_size**2 * 3) tensor of patches
            
        Returns:
            (N, 3, H, W) tensor of images
        """
        p = self.model.config.patch_size
        h = w = int(patches.shape[1] ** 0.5)
        assert h * w == patches.shape[1], 'Number of patches must be a perfect square.'
        
        x = patches.reshape(shape=(patches.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, w * p))
        return imgs
    
    def compute_mae_reconstruction_error(
        self,
        image: Union[torch.Tensor, List[torch.Tensor]],
        return_patch_errors: bool = True,
        reduction: str = 'mean',
        metric: str = 'l1'  # 'l1', 'l2', or 'ssim'
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Compute MAE reconstruction error with multiple error metrics.
        
        Args:
            image: Input image tensor (C, H, W) or batch of images (N, C, H, W)
            return_patch_errors: Whether to return per-patch errors
            reduction: How to reduce the error across patches: 'mean', 'sum', or 'none'
            metric: Error metric to use: 'l1', 'l2', or 'ssim'
            
        Returns:
            If return_patch_errors is False, returns a scalar or batch of scalars
            If return_patch_errors is True, returns a tuple of (error, patch_errors)
        """
        from torchmetrics.functional import structural_similarity_index_measure as ssim
        
        if isinstance(image, list):
            image = torch.stack(image)
            
        if image.dim() == 3:
            image = image.unsqueeze(0)
            
        assert image.dim() == 4, "Input must be a 4D tensor (N, C, H, W)"
        assert reduction in ['mean', 'sum', 'none'], "reduction must be 'mean', 'sum', or 'none'"
        assert metric in ['l1', 'l2', 'ssim'], "metric must be 'l1', 'l2', or 'ssim'"
        
        with torch.no_grad():
            # Store original image for later comparison
            original_image = image.to(self.device)
            
            # Forward pass through MAE with masking
            outputs = self.model(
                pixel_values=original_image,
                output_hidden_states=True,
                return_dict=True
            )
            
            # Get the reconstructed pixel values
            last_hidden = outputs.last_hidden_state
            
            # Get the decoder output and normalize if needed
            reconstructed = self.model.vit.layernorm(self.model.decoder(last_hidden))
            
            # Predict pixels with decoder
            pred = self.model.decoder.decoder_pred(reconstructed)
            
            # Remove [CLS] token and reshape to patches
            pred = pred[:, 1:, :]
            
            # Reconstruct the image from patches
            reconstructed_patches = pred
            reconstructed_image = self._unpatchify(reconstructed_patches)
            
            # Calculate error based on the specified metric
            if metric == 'l1':
                patch_errors = F.l1_loss(
                    self._patchify(original_image),
                    reconstructed_patches,
                    reduction='none'
                ).mean(dim=-1)  # Mean over channels and pixels in patch
                
            elif metric == 'l2':
                patch_errors = F.mse_loss(
                    self._patchify(original_image),
                    reconstructed_patches,
                    reduction='none'
                ).mean(dim=-1)  # Mean over channels and pixels in patch
                
            elif metric == 'ssim':
                # SSIM is computed on the full image, not per-patch
                # We'll compute it per-patch by extracting patches
                ssim_errors = []
                p = self.model.config.patch_size
                
                # Process each image in the batch
                for i in range(original_image.shape[0]):
                    # Extract patches
                    patches_original = F.unfold(
                        original_image[i:i+1], 
                        kernel_size=p, 
                        stride=p
                    ).transpose(1, 2)
                    
                    patches_recon = F.unfold(
                        reconstructed_image[i:i+1], 
                        kernel_size=p, 
                        stride=p
                    ).transpose(1, 2)
                    
                    # Reshape to (num_patches, 3, p, p)
                    patches_original = patches_original.reshape(-1, 3, p, p)
                    patches_recon = patches_recon.reshape(-1, 3, p, p)
                    
                    # Compute SSIM for each patch
                    ssim_scores = ssim(
                        patches_original, 
                        patches_recon,
                        data_range=1.0,
                        reduction='none'
                    )
                    ssim_errors.append(1.0 - ssim_scores)  # Convert to error (lower is worse)
                
                patch_errors = torch.stack(ssim_errors).view(original_image.shape[0], -1)
            
            # Apply reduction
            if reduction == 'mean':
                error = patch_errors.mean(dim=1)
            elif reduction == 'sum':
                error = patch_errors.sum(dim=1)
            else:  # 'none'
                error = patch_errors
            
            # Squeeze if single image
            if error.shape[0] == 1 and not return_patch_errors:
                error = error.squeeze(0)
            
            if return_patch_errors:
                return error, patch_errors
            return error
    
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
