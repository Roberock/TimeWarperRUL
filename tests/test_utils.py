import pytest
import numpy as np
from rul_timewarping.utils import (
    ecdf, empirical_reliability, get_non_param_reliability,
    estimate_k, compute_g_non_parametric, compute_mrl,
    get_g_from_lifelines
)
from lifelines import KaplanMeierFitter
import warnings
from rul_timewarping.timewarping import TimeWarping


class TestUtilsExtended:
    """Extended tests for utils module to cover missing lines."""

    def setup_method(self):
        """Set up test data for each test method."""
        np.random.seed(42)
        self.ttf_data = np.random.weibull(a=2.5, size=100) * 1000 + 200

    def test_ecdf_edge_cases(self):
        """Test ECDF with edge cases."""
        single_data = np.array([100])
        x, y = ecdf(single_data)
        assert isinstance(x, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert len(x) == 1
        assert len(y) == 1
        assert x[0] == 100
        assert y[0] == 1.0
        duplicate_data = np.array([100, 100, 100])
        x, y = ecdf(duplicate_data)
        assert len(x) == 3  # All sorted values
        assert len(y) == 3
        assert np.all(x == 100)
        assert np.allclose(y, [1 / 3, 2 / 3, 1.0])
        empty_data = np.array([])
        x, y = ecdf(empty_data)
        assert len(x) == 0
        assert len(y) == 0

    def test_empirical_reliability_edge_cases(self):
        data = np.array([100, 200, 300, 400, 500])
        R_below = empirical_reliability(data, 50)
        assert isinstance(R_below, float)
        assert R_below == 1.0
        R_min = empirical_reliability(data, 100)
        assert isinstance(R_min, float)
        assert 0 < R_min < 1
        R_max = empirical_reliability(data, 500)
        assert isinstance(R_max, float)
        assert 0 < R_max < 1
        R_above = empirical_reliability(data, 600)
        assert isinstance(R_above, float)
        assert 0 < R_above <= 1
        R_middle = empirical_reliability(data, 250)
        assert isinstance(R_middle, float)
        assert 0 < R_middle < 1
        empty_data = np.array([])
        R_empty = empirical_reliability(empty_data, 100)
        assert isinstance(R_empty, float)
        assert np.isnan(R_empty)

    def test_get_non_param_reliability_edge_cases(self):
        single_data = np.array([100])
        x_vals = np.linspace(0, 200, 10)
        with pytest.raises(ValueError):
            get_non_param_reliability(x_vals, single_data)
        identical_data = np.array([100, 100, 100])
        R_vals = get_non_param_reliability(x_vals, identical_data)
        assert isinstance(R_vals, np.ndarray)
        assert R_vals.shape == (10,)
        assert np.all(R_vals >= 0)
        assert np.all(R_vals <= 1)
        x_vals_outside = np.linspace(1000, 2000, 10)
        R_vals = get_non_param_reliability(x_vals_outside, self.ttf_data)
        assert isinstance(R_vals, np.ndarray)
        assert R_vals.shape == (10,)
        assert np.all(R_vals >= 0)
        assert np.all(R_vals <= 1)

    def test_estimate_k_edge_cases(self):
        single_data = np.array([100])
        k, mu = estimate_k(single_data)
        assert isinstance(k, float)
        assert isinstance(mu, float)
        assert mu == 100
        assert k == 1.0
        identical_data = np.array([100, 100, 100])
        k, mu = estimate_k(identical_data)
        assert isinstance(k, float)
        assert isinstance(mu, float)
        assert mu == 100
        assert k == 1.0
        small_data = np.array([0.001, 0.002, 0.003])
        k, mu = estimate_k(small_data)
        assert isinstance(k, float)
        assert isinstance(mu, float)
        assert mu > 0
        assert 0 <= k <= 1

    def test_compute_g_non_parametric_edge_cases(self):
        single_data = np.array([100])
        with pytest.raises(ValueError):
            compute_g_non_parametric(single_data)
        identical_data = np.array([100, 100, 100])
        k, mu, x, g, R, kde = compute_g_non_parametric(identical_data)
        assert isinstance(k, float)
        assert isinstance(mu, float)
        assert isinstance(x, np.ndarray)
        assert isinstance(g, np.ndarray)
        assert isinstance(R, np.ndarray)
        assert kde is not None
        k, mu, x, g, R, kde = compute_g_non_parametric(self.ttf_data, bw_method='silverman')
        assert isinstance(k, float)
        assert isinstance(mu, float)
        assert isinstance(x, np.ndarray)
        assert isinstance(g, np.ndarray)
        assert isinstance(R, np.ndarray)
        assert kde is not None

    def test_compute_mrl_edge_cases(self):
        """Test MRL computation with edge cases."""
        x_vals = np.linspace(0, 1000, 100)
        R_vals = np.exp(-x_vals / 500)  # Exponential reliability

        # Test with normal case
        mrl = compute_mrl(x_vals, R_vals)

        assert isinstance(mrl, np.ndarray)
        assert mrl.shape == (100,)
        assert np.all(mrl >= 0)

        # Test with R_vals that go to zero at the end
        R_vals_zero = R_vals.copy()
        R_vals_zero[-1] = 0
        mrl = compute_mrl(x_vals, R_vals_zero)

        assert isinstance(mrl, np.ndarray)
        assert mrl.shape == (100,)
        assert np.all(mrl >= 0)

        # Test with very small R_vals
        R_vals_small = np.full_like(R_vals, 1e-10)
        mrl = compute_mrl(x_vals, R_vals_small)

        assert isinstance(mrl, np.ndarray)
        assert mrl.shape == (100,)
        assert np.all(mrl >= 0)

        # Test with single point
        x_single = np.array([500])
        R_single = np.array([0.5])
        mrl = compute_mrl(x_single, R_single)

        assert isinstance(mrl, np.ndarray)
        assert mrl.shape == (1,)
        assert mrl[0] >= 0

    def test_get_g_from_lifelines(self):
        """Test g function computation from lifelines model."""
        kmf = KaplanMeierFitter()
        kmf.fit(self.ttf_data)
        g_vals = get_g_from_lifelines(kmf, TimeWarping, self.ttf_data)
        assert isinstance(g_vals, np.ndarray)
        assert len(g_vals) > 0
        assert np.all(g_vals >= 0)

    def test_get_g_from_lifelines_edge_cases(self):
        """Test g function from lifelines with edge cases."""
        single_data = np.array([100])
        kmf = KaplanMeierFitter()
        kmf.fit(single_data)
        g_vals = get_g_from_lifelines(kmf, TimeWarping, single_data)
        assert isinstance(g_vals, np.ndarray)
        assert len(g_vals) > 0
        assert np.all(g_vals >= 0)
        identical_data = np.array([100, 100, 100])
        kmf = KaplanMeierFitter()
        kmf.fit(identical_data)
        g_vals = get_g_from_lifelines(kmf, TimeWarping, identical_data)
        assert isinstance(g_vals, np.ndarray)
        assert len(g_vals) > 0
        assert np.all(g_vals >= 0)

    def test_utils_imports(self):
        """Test that all utils functions can be imported and called."""
        # Test that restricted_mean_survival_time can be imported
        try:
            from lifelines.utils import restricted_mean_survival_time
            assert callable(restricted_mean_survival_time)
        except ImportError:
            # This might not be available in all lifelines versions
            pass

        # Test that all our functions are callable
        assert callable(ecdf)
        assert callable(empirical_reliability)
        assert callable(get_non_param_reliability)
        assert callable(estimate_k)
        assert callable(compute_g_non_parametric)
        assert callable(compute_mrl)
        assert callable(get_g_from_lifelines)


def test_ecdf_basic():
    data = np.array([3, 1, 2])
    x, y = ecdf(data)
    assert np.all(np.diff(x) >= 0)  # x sorted
    assert y[0] == 1/3 and y[-1] == 1.0  # y runs from 1/n to 1

def test_empirical_reliability_values():
    data = np.array([10, 20, 30])
    assert np.isclose(empirical_reliability(data, 5), 1.0)
    assert  np.isclose(empirical_reliability(data, 20), 1/3)
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