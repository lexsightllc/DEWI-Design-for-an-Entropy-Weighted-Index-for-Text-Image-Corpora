from dewi.scorer import DewiScorer
from dewi.types import Weights, Payload


def test_scorer_weights_only():
    w = Weights(alpha_t=0.6, alpha_i=0.2, alpha_r=0.2, alpha_n=0.1)
    s = DewiScorer(weights=w)
    p = Payload(ht_mean=1.0, hi_mean=0.5, redundancy=0.2, noise=0.1, ht_q90=1.2, hi_q90=0.7)
    sig = p.to_dict()
    sig["I_hat"] = 0.0
    s.fit_stats([sig])
    v = s.score(sig)
    c = s.score_conditional(sig)
    assert isinstance(v, float) and isinstance(c, float)
