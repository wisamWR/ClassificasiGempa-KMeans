import numpy as np
import pytest

from gempa.intensity import epicentral_intensity


def test_known_value():
    # I0 = 1.5*5 - 3.5*log10(10) + 3 = 7.0
    assert epicentral_intensity(5.0, 10.0) == pytest.approx(7.0)


def test_depth_clamped_to_minimum():
    # h < 5 km dihitung sebagai 5 km (hindari log10(0))
    assert epicentral_intensity(5.0, 0.0) == pytest.approx(epicentral_intensity(5.0, 5.0))


def test_deeper_and_smaller_is_weaker():
    assert epicentral_intensity(6.0, 300.0) < epicentral_intensity(6.0, 10.0)
    assert epicentral_intensity(4.0, 10.0) < epicentral_intensity(6.0, 10.0)


def test_vectorized():
    result = epicentral_intensity(np.array([5.0, 5.0]), np.array([10.0, 100.0]))
    np.testing.assert_allclose(result, [7.0, 3.5])
