import numpy as np
from dewi.index import DewiIndex
from dewi.types import Payload


def test_index_delegation_shapes_and_no_error(tmp_path):
    dim = 16
    idx = DewiIndex(dim=dim, backend="auto", use_ann=False)
    for j in range(10):
        v = np.random.RandomState(100 + j).randn(dim).astype(np.float32)
        idx.add(f"id-{j}", v, Payload(dewi=0.5))
    idx.build()
    q = np.zeros(dim, dtype=np.float32)
    q[0] = 1.0
    res = idx.search(q, k=5, eta=0.0, entropy_pref=0.0)
    assert len(res) == 5
