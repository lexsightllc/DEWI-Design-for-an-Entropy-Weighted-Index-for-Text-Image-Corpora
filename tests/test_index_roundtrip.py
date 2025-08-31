from pathlib import Path

import numpy as np

from dewi.index import DewiIndex, Payload


def test_index_roundtrip(tmp_path: Path):
    dim = 8
    idx = DewiIndex(dim=dim, backend="auto", use_ann=False)
    for i in range(5):
        v = np.random.RandomState(42 + i).randn(dim).astype(np.float32)
        idx.add(f"id-{i}", v, payload=Payload())
    idx.build()
    q = np.zeros(dim, dtype=np.float32)
    q[0] = 1.0
    res = idx.search(q, k=3)
    assert len(res) == 3
    save_dir = tmp_path / "idx"
    idx.save(save_dir)
    re = DewiIndex.load(save_dir)
    res2 = re.search(q, k=3)
    assert len(res2) == 3

