import numpy as np
import pytest
from PIL import Image


def _rgb(w=64, h=64, val=128):
    arr = np.full((h, w, 3), val, dtype=np.uint8)
    return Image.fromarray(arr)


def test_image_entropy_runs():
    from dewi.signals.image_entropy import ImageEntropyEstimator

    img = _rgb()
    est = ImageEntropyEstimator()
    s = est.score(img)
    assert isinstance(s, float)


def test_redundancy_clip_similarity_runs():
    from dewi.signals.redundancy import RedundancyEstimator

    txts = ["a photo of a dog", "a photo of a cat"]
    imgs = [_rgb(), _rgb(val=200)]
    try:
        est = RedundancyEstimator()
    except Exception:
        pytest.skip("CLIP not available")
    sim = est.compute_cross_modal_similarity(txts, imgs)
    assert sim.shape == (len(txts), len(imgs))


def test_noise_estimator_basic():
    from dewi.signals.noise import NoiseEstimator

    ne = NoiseEstimator()
    q = ne.check_text_quality("hello world")
    assert 0.0 <= q <= 1.0

