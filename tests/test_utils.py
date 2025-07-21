import numpy as np
import pytest
from rul_timewarping.utils import ecdf, empirical_reliability, estimate_k, compute_g_non_parametric, compute_mrl

def test_ecdf_basic():
    data = np.array([3, 1, 2])
    x, y = ecdf(data)
    assert np.all(np.diff(x) >= 0)  # x sorted
    assert y[0] == 1/3 and y[-1] == 1.0  # y runs from 1/n to 1

def test_empirical_reliability_values():
    data = np.array([10, 20, 30])
    assert np.isclose(empirical_reliability(data, 5), 1.0)
    assert np.isclose(empirical_reliability(data, 20), 2/3)
    assert np.isclose(empirical_reliability(data, 40), 0.001, atol=1e-3)  # clipped min

def test_estimate_k_values():
    data = np.array([1, 2, 3, 4, 5])
    k, mu = estimate_k(data)
    assert mu == np.mean(data)
    assert 0 <= k <= 1

def test_compute_g_non_parametric_shape_and_types():
    data = np.linspace(1, 10, 20)
    k, mu, x, g, R, kde = compute_g_non_parametric(data)
    assert isinstance(k, float)
    assert isinstance(mu, float)
    assert isinstance(x, np.ndarray)
    assert isinstance(g, np.ndarray)
    assert isinstance(R, np.ndarray)
    assert hasattr(kde, 'evaluate')
    assert x.shape == g.shape == R.shape

def test_compute_mrl_non_negative_and_shape():
    x_vals = np.linspace(0, 10, 100)
    R_vals = np.linspace(1, 0, 100)
    mrl = compute_mrl(x_vals, R_vals)
    assert isinstance(mrl, np.ndarray)
    assert mrl.shape == x_vals.shape
    assert np.all(mrl >= 0)