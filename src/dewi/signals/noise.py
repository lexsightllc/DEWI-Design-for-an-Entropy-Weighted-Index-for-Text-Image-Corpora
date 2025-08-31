import torch
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Union, Tuple
from PIL import Image, ImageFilter, ImageEnhance
import cv2
import re

@dataclass
class NoiseScores:
    blur_score: float  # Higher means more blurry
    jpeg_artifacts: float  # Higher means more compression artifacts
    nsfw_score: float  # Higher means more likely to be NSFW
    watermark_score: float  # Higher means more likely to have watermarks
    text_quality: float  # Higher means better text quality (OCR confidence, etc.)
    language_match: float  # 1.0 if language matches expected, 0.0 otherwise
    
class NoiseEstimator:
    """Estimates noise and quality issues in text and images."""
    
    def __init__(
        self,
        expected_language: str = "en",
        device: Optional[str] = None,
    ):
        """Initialize the noise estimator.
        
        Args:
            expected_language: Expected language code for text content.
            device: Device to run the model on ('cuda' or 'cpu').
        """
        self.expected_language = expected_language
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize NSFW detector (placeholder - in practice, use a proper model)
        self.nsfw_detector = None
        
        # Initialize OCR (placeholder)
        self.ocr = None
    
    def detect_blur(self, image: Image.Image) -> float:
        """Estimate blurriness of an image using Laplacian variance."""
        # Convert to grayscale
        if image.mode != 'L':
            gray = image.convert('L')
        else:
            gray = image
            
        # Convert to numpy array
        img_array = np.array(gray)
        
        # Compute Laplacian
        laplacian = cv2.Laplacian(img_array, cv2.CV_64F)
        variance = laplacian.var()
        
        # Normalize to [0, 1], higher means more blurry
        # These thresholds are approximate and may need adjustment
        if variance < 0:
            return 1.0
        elif variance > 1000:
            return 0.0
        else:
            return 1.0 - (variance / 1000.0)
    
    def detect_jpeg_artifacts(self, image: Image.Image) -> float:
        """Detect JPEG compression artifacts."""
        # Convert to numpy array
        img_array = np.array(image)
        
        # Convert to grayscale if needed
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        # Compute DCT
        dct = cv2.dct(np.float32(gray) / 255.0)
        
        # Calculate the variance of high-frequency components
        # (upper-left corner has low frequencies, bottom-right has high frequencies)
        h, w = dct.shape
        h_cutoff = h // 4
        w_cutoff = w // 4
        
        # Extract high-frequency components
        hf = dct[h_cutoff:, w_cutoff:]
        
        # Calculate normalized energy in high frequencies
        hf_energy = np.sum(hf ** 2) / (h * w)
        
        # Higher energy in high frequencies suggests fewer compression artifacts
        # Normalize to [0, 1], higher means more artifacts
        return 1.0 - min(max(hf_energy * 100, 0.0), 1.0)
    
    def detect_watermark(self, image: Image.Image) -> float:
        """Detect watermarks using edge detection and texture analysis."""
        # Convert to grayscale
        if image.mode != 'L':
            gray = image.convert('L')
        else:
            gray = image
        
        img_array = np.array(gray)
        
        # Edge detection
        edges = cv2.Canny(img_array, 100, 200)
        
        # Calculate edge density
        edge_density = np.mean(edges > 0)
        
        # Simple heuristic: watermarks often have high edge density in a small region
        # This is a very basic approach and would need refinement
        return min(edge_density * 10, 1.0)
    
    def check_text_quality(self, text: str) -> float:
        """Check text quality based on various heuristics."""
        if not text.strip():
            return 0.0
        
        score = 1.0
        
        # Penalize very short texts
        if len(text) < 10:
            score *= 0.5
        
        # Check for excessive capitalization (common in OCR errors)
        upper_ratio = sum(1 for c in text if c.isupper()) / len(text)
        if upper_ratio > 0.5:
            score *= 0.7
        
        # Check for repeated characters (common in OCR errors)
        if re.search(r'(.)\1{3,}', text):
            score *= 0.8
        
        # Check for common OCR artifacts
        ocr_artifacts = [
            r'\|',  # Vertical bars
            r'[l1][|]',  # Misrecognized characters
            r'[o0O]',  # Confusable characters
        ]
        
        for pattern in ocr_artifacts:
            if re.search(pattern, text):
                score *= 0.9
        
        return score
    
    def check_language(self, text: str) -> float:
        """Check if text matches expected language (placeholder)."""
        # In practice, use a language detection library like langdetect
        # This is a simplified placeholder
        if not text.strip():
            return 0.0
            
        # Check for non-ASCII characters (very rough approximation)
        non_ascii = sum(1 for c in text if ord(c) > 127) / len(text)
        
        # If we expect English and see many non-ASCII, it might be a different language
        if self.expected_language == "en" and non_ascii > 0.3:
            return 0.0
            
        return 1.0
    
    def detect_nsfw(self, image: Image.Image) -> float:
        """Detect NSFW content (placeholder implementation)."""
        # In practice, use a proper NSFW detection model
        # This is a placeholder that always returns 0 (not NSFW)
        return 0.0
    
    def compute_noise_scores(
        self,
        text: Optional[str] = None,
        image: Optional[Image.Image] = None,
    ) -> NoiseScores:
        """Compute noise and quality scores for text and/or image.
        
        Args:
            text: Optional input text.
            image: Optional input image.
            
        Returns:
            NoiseScores object with various quality metrics.
        """
        scores = {
            'blur_score': 0.0,
            'jpeg_artifacts': 0.0,
            'nsfw_score': 0.0,
            'watermark_score': 0.0,
            'text_quality': 1.0,  # Default to high quality
            'language_match': 1.0,  # Default to matching
        }
        
        # Process image if provided
        if image is not None:
            scores['blur_score'] = self.detect_blur(image)
            scores['jpeg_artifacts'] = self.detect_jpeg_artifacts(image)
            scores['watermark_score'] = self.detect_watermark(image)
            scores['nsfw_score'] = self.detect_nsfw(image)
        
        # Process text if provided
        if text is not None:
            scores['text_quality'] = self.check_text_quality(text)
            scores['language_match'] = self.check_language(text)
        
        return NoiseScores(**scores)
    
    def batch_compute(
        self,
        texts: Optional[List[str]] = None,
        images: Optional[List[Image.Image]] = None,
    ) -> List[NoiseScores]:
        """Compute noise scores for a batch of texts and/or images.
        
        Args:
            texts: Optional list of input texts.
            images: Optional list of input images.
            
        Returns:
            List of NoiseScores objects.
        """
        if texts is None and images is None:
            raise ValueError("At least one of texts or images must be provided")
            
        if texts is not None and images is not None:
            if len(texts) != len(images):
                raise ValueError("texts and images must have the same length")
            return [
                self.compute_noise_scores(text=t, image=i)
                for t, i in zip(texts, images)
            ]
        elif texts is not None:
            return [self.compute_noise_scores(text=t) for t in texts]
        else:  # images is not None
            return [self.compute_noise_scores(image=i) for i in images]
    
    def __call__(
        self,
        text: Optional[str] = None,
        image: Optional[Image.Image] = None,
    ) -> NoiseScores:
        """Alias for compute_noise_scores for easier function-like usage."""
        return self.compute_noise_scores(text, image)
