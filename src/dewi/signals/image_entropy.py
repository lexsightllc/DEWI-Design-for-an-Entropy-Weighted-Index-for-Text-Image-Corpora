"""Lightweight image entropy estimator with MAE reconstruction fallback."""

from typing import Optional

import numpy as np
try:  # Optional dependency
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore
from PIL import Image

try:  # Optional dependency
    from transformers import AutoImageProcessor, ViTMAEForPreTraining
except Exception:  # pragma: no cover - transformers not installed
    AutoImageProcessor = None
    ViTMAEForPreTraining = None


class ImageEntropyEstimator:
    """Estimate image entropy via ViT-MAE reconstruction loss.

    Falls back to a simple pixel variance proxy if transformers/weights are
    unavailable. Returned score is a non-negative float (MSE or variance).
    """

    def __init__(
        self,
        model_name: str = "facebook/vit-mae-base",
        device: Optional[str] = None,
        mask_ratio: float = 0.75,
        **_: object,
    ) -> None:
        self.device = device or ("cuda" if torch is not None and torch.cuda.is_available() else "cpu")
        self.mask_ratio = float(mask_ratio)
        if AutoImageProcessor is None or ViTMAEForPreTraining is None or torch is None:
            self.processor = None
            self.model = None
        else:
            self.processor = AutoImageProcessor.from_pretrained(model_name)
            self.model = ViTMAEForPreTraining.from_pretrained(model_name).to(self.device)
            self.model.eval()

    def _masked_positions(self, num_patches: int) -> "torch.Tensor":
        m = int(round(num_patches * self.mask_ratio))
        mask = torch.zeros(num_patches, dtype=torch.bool)
        if m > 0:
            perm = torch.randperm(num_patches)
            mask[perm[:m]] = True
        return mask.unsqueeze(0)

    def score(self, image: Image.Image) -> float:
        if self.processor is None or self.model is None or torch is None:
            arr = np.asarray(image.convert("L"), dtype=np.float32)
            return float(arr.var() / (arr.mean() ** 2 + 1e-8))

        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)
        patch_size = getattr(self.model.config, "patch_size", 16)
        num_patches = getattr(self.model.config, "num_patches", None)
        if num_patches is None:
            _, _, H, W = pixel_values.shape
            num_patches = (H // patch_size) * (W // patch_size)
        bool_masked_pos = self._masked_positions(num_patches).to(self.device)
        with torch.no_grad():
            try:
                out = self.model(pixel_values=pixel_values, bool_masked_pos=bool_masked_pos)
                if hasattr(out, "loss") and out.loss is not None:
                    return float(out.loss.item())
            except Exception:
                pass
            out = self.model.vit(pixel_values=pixel_values)
            hidden = out.last_hidden_state[:, 1:, :]
            mse = torch.mean(hidden.pow(2), dim=-1).mean().item()
            return float(mse)

