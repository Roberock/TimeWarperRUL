import pytest
import numpy as np
from rul_timewarping.timewarping import TimeWarping
import warnings

class TestTimeWarpingExtended:
    """Extended tests for TimeWarping class to cover missing lines."""

    def setup_method(self):
        """Set up test data for each test method."""
        np.random.seed(42)
        self.ttf_data = np.random.weibull(a=2.5, size=100) * 1000 + 200

    def test_initialization_with_empty_data(self):
        """Test initialization with empty data."""
        tw = TimeWarping(ttf_data=None)
        assert tw.ttf_data is None
        assert tw.N == 0
        assert tw.mu is None
        assert tw.cv is None
        assert tw.k is None

    def test_initialization_with_single_data_point(self):
        """Test initialization with insufficient data (< 2 points)."""
        single_data = np.array([100])
        with pytest.raises(ValueError):
            TimeWarping(ttf_data=single_data)

    def test_initialization_with_two_data_points(self):
        """Test initialization with exactly 2 data points."""
        two_data = np.array([100, 200])
        tw = TimeWarping(ttf_data=two_data)

        assert tw.N == 2
        assert tw.mu == 150
        assert tw.std > 0

    def test_g_vals_computation_failure(self):
        """Test behavior when g_vals computation fails."""
        # Create data that might cause issues with KDE
        problematic_data = np.array([1, 1, 1, 1, 1])  # All identical values

        with pytest.raises(Exception):
            TimeWarping(ttf_data=problematic_data)

    def test_non_monotonic_timeline(self):
        """Test behavior with non-monotonic timeline (edge case)."""
        # This test covers the case where timeline is not monotonically increasing
        # This would be an edge case in the inflection points computation

        tw = TimeWarping(ttf_data=self.ttf_data)

        # The current implementation should handle this gracefully
        # but we can test the inflection points computation
        inflection_x, inflection_g = tw.estimate_inflection_points()

        assert isinstance(inflection_x, np.ndarray)
        assert isinstance(inflection_g, np.ndarray)

    def test_compute_rul_interval_edge_cases(self):
        """Test RUL interval computation with edge cases."""
        tw = TimeWarping(ttf_data=self.ttf_data)

        # Test with very small alpha
        t = np.array([100, 500, 1000])
        s_plus, s_minus = tw.compute_rul_interval(t, alpha=1e-10)

        assert isinstance(s_plus, np.ndarray)
        assert isinstance(s_minus, np.ndarray)
        assert s_plus.shape == t.shape
        assert s_minus.shape == t.shape

        # Test with alpha = 1
        s_plus, s_minus = tw.compute_rul_interval(t, alpha=1.0)

        assert isinstance(s_plus, np.ndarray)
        assert isinstance(s_minus, np.ndarray)

    def test_compute_rul_interval_original_time_edge_cases(self):
        """Test RUL interval in original time with edge cases."""
        tw = TimeWarping(ttf_data=self.ttf_data)

        # Test with different alpha values
        alpha_values = [0.01, 0.05, 0.1, 0.5]
        for alpha in alpha_values:
            L_alpha, U_alpha = tw._get_rul_interval_original_time(alpha=alpha)

            assert isinstance(L_alpha, np.ndarray)
            assert isinstance(U_alpha, np.ndarray)
            assert len(L_alpha) > 0
            assert len(U_alpha) > 0
            assert np.all(L_alpha >= 0)
            assert np.all(U_alpha >= 0)

    def test_compute_g_fun_static_method(self):
        """Test the static compute_g_fun method."""
        R = np.array([1.0, 0.8, 0.5, 0.2, 0.1])
        k = 0.5
        mu = 1000

        g_vals = TimeWarping.compute_g_fun(R, k, mu)

        assert isinstance(g_vals, np.ndarray)
        assert g_vals.shape == R.shape
        assert np.all(g_vals >= 0)

    def test_compute_g_fun_edge_cases(self):
        """Test compute_g_fun with edge cases."""
        R = np.array([1.0, 0.8, 0.5])
        k = 0.0
        mu = 1000
        with pytest.raises(ZeroDivisionError):
            TimeWarping.compute_g_fun(R, k, mu)
        k = 1.0
        with pytest.raises(ZeroDivisionError):
            TimeWarping.compute_g_fun(R, k, mu)

    def test_estimate_inflection_points_edge_cases(self):
        """Test inflection points estimation with edge cases."""
        tw = TimeWarping(ttf_data=self.ttf_data)

        # Test with different smoothing parameters
        smooth_sigmas = [0.1, 1.0, 5.0, 10.0]
        for sigma in smooth_sigmas:
            inflection_x, inflection_g = tw.estimate_inflection_points(smooth_sigma=sigma)

            assert isinstance(inflection_x, np.ndarray)
            assert isinstance(inflection_g, np.ndarray)

        # Test with different tolerance values
        tols = [1e-6, 1e-4, 1e-2]
        for tol in tols:
            inflection_x, inflection_g = tw.estimate_inflection_points(tol=tol)

            assert isinstance(inflection_x, np.ndarray)
            assert isinstance(inflection_g, np.ndarray)

    def test_compute_mrl_edge_cases(self):
        """Test MRL computation with edge cases."""
        tw = TimeWarping(ttf_data=self.ttf_data)

        # Test with single time point
        t_single = np.array([500])
        mrl_single = tw.compute_mrl(t_single)

        assert isinstance(mrl_single, np.ndarray)
        assert mrl_single.shape == (1,)
        assert mrl_single[0] >= 0

        # Test with multiple time points
        t_multiple = np.array([100, 500, 1000, 1500])
        mrl_multiple = tw.compute_mrl(t_multiple)

        assert isinstance(mrl_multiple, np.ndarray)
        assert mrl_multiple.shape == (4,)
        assert np.all(mrl_multiple >= 0)

        # Test with time points outside the range
        t_outside = np.array([0, 10000])
        mrl_outside = tw.compute_mrl(t_outside)

        assert isinstance(mrl_outside, np.ndarray)
        assert mrl_outside.shape == (2,)

    def test_empirical_reliability_edge_cases(self):
        """Test empirical reliability with edge cases."""
        tw = TimeWarping(ttf_data=self.ttf_data)

        # Test with t = 0
        R_zero = tw._empirical_reliability(0)
        assert isinstance(R_zero, float)
        assert 0 <= R_zero <= 1

        # Test with t = max(ttf_data)
        R_max = tw._empirical_reliability(np.max(self.ttf_data))
        assert isinstance(R_max, float)
        assert 0 <= R_max <= 1

        # Test with t > max(ttf_data)
        R_large = tw._empirical_reliability(np.max(self.ttf_data) + 1000)
        assert isinstance(R_large, float)
        assert 0 <= R_large <= 1

        # Test with t < 0
        R_negative = tw._empirical_reliability(-100)
        assert isinstance(R_negative, float)
        assert 0 <= R_negative <= 1

    def test_make_grid(self):
        """Test the _make_grid method."""
        tw = TimeWarping(ttf_data=self.ttf_data)

        grid = tw._make_grid()

        assert isinstance(grid, np.ndarray)
        assert len(grid) > 0
        assert np.all(np.diff(grid) >= 0)  # Should be monotonically increasing
        assert grid[0] >= 0  # Should start at 0 or positive value



def test_initialization():
    ttf = np.array([10, 20, 30, 40, 50])
    tw = TimeWarping(ttf)
    assert tw.mu > 0
    assert tw.k > 0

def test_estimate_inflection_points():
    ttf = np.linspace(1, 100, 100)
    tw = TimeWarping(ttf)
    inflection_x, inflection_g = tw.estimate_inflection_points()
    assert isinstance(inflection_x, np.ndarray)
    assert isinstance(inflection_g, np.ndarray)

def test_compute_rul_interval():
    ttf = np.linspace(1, 100, 100)
    tw = TimeWarping(ttf)
    t = np.array([10, 20, 30])
    s_plus, s_minus = tw.compute_rul_interval(t)
    assert s_plus.shape == t.shape
    assert s_minus.shape == t.shape

def test_compute_rul_interval_original_time():
    ttf = np.linspace(1, 100, 100)
    tw = TimeWarping(ttf)
    L_alpha, U_alpha = tw._get_rul_interval_original_time()
    assert isinstance(L_alpha, np.ndarray)
    assert isinstance(U_alpha, np.ndarray)
    assert L_alpha.shape == U_alpha.shape

def test_empty_initialization():
    tw = TimeWarping()
    assert tw.mu is None
    assert tw.k is None

def test_compute_rul_interval_alpha_edges():
    ttf = np.linspace(1, 100, 100)
    tw = TimeWarping(ttf)
    t = np.array([10, 20, 30])

    s_plus_0, s_minus_0 = tw.compute_rul_interval(t, alpha=0)
    s_plus_1, s_minus_1 = tw.compute_rul_interval(t, alpha=1)

    # alpha=0 might yield max intervals, alpha=1 minimal
    assert np.all(s_plus_0 >= s_minus_0)
    assert np.all(s_plus_1 >= 0)
    assert np.all(s_minus_1 >= 0)


def test_compute_mrl():
    ttf = np.linspace(1, 100, 100)
    tw = TimeWarping(ttf)
    t = np.array([10, 20, 30])
    mrl = tw.compute_mrl(t)

    # Basic shape check
    assert mrl.shape == t.shape

    # MRL should be decreasing (monotonic hazard)
    assert np.all(np.diff(mrl) <= 0)

    # MRL must be positive
    assert np.all(mrl > 0)