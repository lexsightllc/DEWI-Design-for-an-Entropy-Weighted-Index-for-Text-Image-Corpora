import pytest
from PIL import Image
import numpy as np


def _img(val=128):
    return Image.fromarray(np.full((224, 224, 3), val, dtype=np.uint8))


def test_clip_similarity_runs():
    try:
        from dewi.signals.redundancy import RedundancyEstimator
        est = RedundancyEstimator()
    except Exception:
        pytest.skip("transformers/CLIP not available in env")
    texts = ["a photo of a dog", "a photo of a cat"]
    imgs = [_img(100), _img(200)]
    sim = est.compute_cross_modal_similarity(texts, imgs)
    assert sim.shape == (len(texts), len(imgs))
