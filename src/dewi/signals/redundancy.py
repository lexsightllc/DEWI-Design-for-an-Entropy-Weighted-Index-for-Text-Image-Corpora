"""Cross-modal redundancy estimation using CLIP embeddings."""

from typing import List

try:  # Optional dependency
    import torch
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    F = None  # type: ignore
from PIL import Image

try:  # Optional dependency
    from transformers import CLIPModel, CLIPProcessor
except Exception:  # pragma: no cover
    CLIPModel = CLIPProcessor = None


class RedundancyEstimator:
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: str = "cpu"):
        if CLIPModel is None or CLIPProcessor is None or torch is None or F is None:
            raise ImportError("transformers and torch with CLIP are required for RedundancyEstimator")
        self.device = device if device in ("cpu", "cuda") else "cpu"
        self._clip = CLIPModel.from_pretrained(model_name).to(self.device)
        self._proc = CLIPProcessor.from_pretrained(model_name)
        self._clip.eval()

    def compute_cross_modal_similarity(self, texts: List[str], images: List[Image.Image]):
        inputs = self._proc(text=texts, images=images, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            tfeat = self._clip.get_text_features(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
            if "pixel_values" not in inputs:
                raise ValueError("No image pixel_values provided.")
            ifeat = self._clip.get_image_features(pixel_values=inputs["pixel_values"])
            tfeat = F.normalize(tfeat, p=2, dim=1)
            ifeat = F.normalize(ifeat, p=2, dim=1)
            sim = (tfeat @ ifeat.T).cpu().numpy()
        return sim

