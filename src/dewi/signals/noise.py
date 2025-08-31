"""Lightweight noise and quality estimators with optional dependencies."""

from typing import Optional

from PIL import Image
import logging

logger = logging.getLogger(__name__)


def _try_nsfw_pipeline(device: Optional[str]):
    try:  # pragma: no cover - optional
        from transformers import pipeline

        dev = 0 if (device or "").startswith("cuda") else -1
        return pipeline("image-classification", model="Falconsai/nsfw_image_detection", device=dev)
    except Exception:
        return None


def _try_langdetect():
    try:  # pragma: no cover - optional
        from langdetect import detect

        return detect
    except Exception:
        return None


def _try_tesseract():
    try:  # pragma: no cover - optional
        import pytesseract

        return pytesseract
    except Exception:
        return None


class NoiseEstimator:
    def __init__(self, expected_language: str = "en", device: Optional[str] = None, enable_nsfw: bool = True):
        self.expected_language = expected_language
        self.device = device
        self.nsfw_detector = _try_nsfw_pipeline(device) if enable_nsfw else None
        self.ocr = _try_tesseract()
        self._lang_detect = _try_langdetect()

    def detect_nsfw(self, image: Image.Image) -> float:
        if self.nsfw_detector is None:
            return 0.0
        try:  # pragma: no cover - model dependent
            out = self.nsfw_detector(image)
            if not out:
                return 0.0
            risk = 0.0
            for item in out:
                label = item.get("label", "").lower()
                score = float(item.get("score", 0.0))
                if any(t in label for t in ("nsfw", "explicit", "porn", "sexual")):
                    risk = max(risk, score)
            return risk
        except Exception as e:
            logger.debug(f"NSFW detection failed: {e}")
            return 0.0

    def check_text_quality(self, text: str) -> float:
        if not text:
            return 0.0
        total = max(len(text), 1)
        alnum = sum(c.isalnum() for c in text)
        symbol = sum((not c.isalnum()) and c not in " \n\t\r" for c in text)
        alnum_ratio = alnum / total
        symbol_ratio = symbol / total
        score = 1.0
        if alnum_ratio < 0.3:
            score -= 0.4
        if symbol_ratio > 0.5:
            score -= 0.2
        return float(max(0.0, min(1.0, score)))

    def check_language(self, text: str) -> float:
        if not text or len(text.strip()) < 2:
            return 0.5
        if self._lang_detect is None:
            return 0.5
        try:  # pragma: no cover - language model
            lang = self._lang_detect(text)
            return 1.0 if lang == self.expected_language else 0.0
        except Exception:
            return 0.5

    def ocr_text(self, image: Image.Image) -> str:
        if self.ocr is None:
            return ""
        try:  # pragma: no cover - depends on tesseract binary
            return self.ocr.image_to_string(image) or ""
        except Exception:
            return ""

