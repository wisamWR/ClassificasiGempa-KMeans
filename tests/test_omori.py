import numpy as np
import pytest

from gempa.stats import omori_mle


def simulate_omori(k, c, p, end, rng):
    """Waktu gempa susulan dari proses Poisson dengan laju K / (t + c)^p pada [0, end]."""
    total = ((end + c) ** (1 - p) - c ** (1 - p)) / (1 - p)
    n = rng.poisson(k * total)
    u = rng.uniform(size=n)
    return (c ** (1 - p) + u * (1 - p) * total) ** (1 / (1 - p)) - c


def test_omori_mle_recovers_parameters():
    rng = np.random.default_rng(3)
    times = simulate_omori(k=80, c=0.05, p=1.1, end=365, rng=rng)
    k, c, p, n = omori_mle(times, 0, 365)
    assert n == len(times) > 500
    assert p == pytest.approx(1.1, abs=0.06)
    assert c == pytest.approx(0.05, rel=0.6)
    assert k == pytest.approx(80, rel=0.25)


def test_omori_mle_needs_enough_events():
    k, c, p, n = omori_mle([0.1, 0.5, 2.0], 0, 30)
    assert n == 3 and np.isnan(p)
