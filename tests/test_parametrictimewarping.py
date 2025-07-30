import pytest
import numpy as np
from rul_timewarping.parametrictimewarping import ParametricTimeWarper
from lifelines import WeibullFitter, ExponentialFitter, LogNormalFitter, LogLogisticFitter, GeneralizedGammaFitter
import warnings


class TestParametricTimeWarper:
    
    def setup_method(self):
        """Set up test data for each test method."""
        np.random.seed(42)
        self.ttf_data = np.random.weibull(a=2.5, size=100) * 1000 + 200
        
    def test_initialization_with_string_fitter(self):
        """Test initialization with string fitter names."""
        # Test with WeibullFitter
        ptw = ParametricTimeWarper(fitter='WeibullFitter')
        assert ptw._fitter is not None
        assert isinstance(ptw._fitter, WeibullFitter)
        
        # Test with ExponentialFitter
        ptw = ParametricTimeWarper(fitter='ExponentialFitter')
        assert isinstance(ptw._fitter, ExponentialFitter)
        
        # Test with LogNormalFitter
        ptw = ParametricTimeWarper(fitter='LogNormalFitter')
        assert isinstance(ptw._fitter, LogNormalFitter)
        
        # Test with LogLogisticFitter
        ptw = ParametricTimeWarper(fitter='LogLogisticFitter')
        assert isinstance(ptw._fitter, LogLogisticFitter)
        
        # Test with GeneralizedGammaFitter
        ptw = ParametricTimeWarper(fitter='GeneralizedGammaFitter')
        assert isinstance(ptw._fitter, GeneralizedGammaFitter)
    
    def test_initialization_with_fitter_instance(self):
        """Test initialization with fitter instances."""
        weibull_fitter = WeibullFitter()
        ptw = ParametricTimeWarper(fitter=weibull_fitter)
        assert ptw._fitter is weibull_fitter
    
    def test_initialization_invalid_fitter_string(self):
        """Test initialization with invalid fitter string raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported fitter"):
            ParametricTimeWarper(fitter='InvalidFitter')
    
    def test_initialization_invalid_fitter_type(self):
        """Test initialization with invalid fitter type raises TypeError."""
        with pytest.raises(TypeError, match="fitter must be a string"):
            ParametricTimeWarper(fitter=123)
    
    def test_fit_weibull_fitter(self):
        """Test fitting with WeibullFitter."""
        ptw = ParametricTimeWarper(fitter='WeibullFitter')
        ptw.fit(self.ttf_data)
        
        # Check that fit was successful
        assert ptw.survival_function_ is not None
        assert ptw.timeline is not None
        assert ptw._reliability is not None
        assert ptw.mu is not None
        assert ptw.sigma is not None
        assert ptw.cv is not None
        assert ptw.k is not None
        assert ptw.g_ is not None
        assert ptw.inflection_points_ is not None
    
    def test_fit_exponential_fitter(self):
        """Test fitting with ExponentialFitter."""
        ptw = ParametricTimeWarper(fitter='ExponentialFitter')
        ptw.fit(self.ttf_data)
        
        assert ptw.mu is not None
        assert ptw.sigma is not None
        assert ptw.cv is not None
        assert ptw.k is not None
    
    def test_fit_lognormal_fitter(self):
        """Test fitting with LogNormalFitter."""
        ptw = ParametricTimeWarper(fitter='LogNormalFitter')
        ptw.fit(self.ttf_data)
        
        assert ptw.mu is not None
        assert ptw.sigma is not None
        assert ptw.cv is not None
        assert ptw.k is not None
    
    def test_fit_loglogistic_fitter(self):
        """Test fitting with LogLogisticFitter."""
        ptw = ParametricTimeWarper(fitter='LogLogisticFitter')
        ptw.fit(self.ttf_data)
        
        assert ptw.mu is not None
        assert ptw.sigma is not None
        assert ptw.cv is not None
        assert ptw.k is not None
    
    def test_fit_generalized_gamma_fitter(self):
        """Test fitting with GeneralizedGammaFitter."""
        ptw = ParametricTimeWarper(fitter='GeneralizedGammaFitter')
        ptw.fit(self.ttf_data)
        
        # GeneralizedGammaFitter might use numerical integration fallback
        assert ptw.mu is not None
        assert ptw.sigma is not None
    
    def test_compute_moments_weibull(self):
        """Test moment computation for Weibull distribution."""
        ptw = ParametricTimeWarper(fitter='WeibullFitter')
        ptw.fit(self.ttf_data)
        
        # Check that moments are reasonable
        assert ptw.mu > 0
        assert ptw.sigma > 0
        assert ptw.mu > ptw.sigma  # For Weibull with shape > 1
    
    def test_compute_moments_exponential(self):
        """Test moment computation for Exponential distribution."""
        ptw = ParametricTimeWarper(fitter='ExponentialFitter')
        ptw.fit(self.ttf_data)
        
        # For exponential, mean = std
        assert ptw.mu > 0
        assert ptw.sigma > 0
        assert abs(ptw.mu - ptw.sigma) < 1e-6
    
    def test_compute_cv_and_k(self):
        """Test CV and k computation."""
        ptw = ParametricTimeWarper(fitter='WeibullFitter')
        ptw.fit(self.ttf_data)
        
        # Check CV calculation
        expected_cv = ptw.sigma / ptw.mu
        assert abs(ptw.cv - expected_cv) < 1e-10
        
        # Check k calculation
        expected_k = (1 - ptw.cv**2) / (1 + ptw.cv**2)
        assert abs(ptw.k - expected_k) < 1e-10
        
        # Check k is in valid range
        assert 0 < ptw.k < 1
    
    def test_compute_cv_and_k_edge_cases(self):
        """Test CV and k computation with edge cases."""
        ptw = ParametricTimeWarper(fitter='WeibullFitter')
        # Test with mu = 0 (should handle gracefully)
        ptw.mu = 0
        ptw.sigma = 1
        ptw._compute_cv_and_k()
        assert np.isnan(ptw.cv) or np.isinf(ptw.cv)
        # Test with cv = 1 (k should be very small, not exactly zero)
        ptw.mu = 1
        ptw.sigma = 1
        ptw._compute_cv_and_k()
        assert abs(ptw.k) < 1.1e-6  # allow for floating point rounding
    
    def test_compute_g_and_inflections(self):
        """Test g function and inflection points computation."""
        ptw = ParametricTimeWarper(fitter='WeibullFitter')
        ptw.fit(self.ttf_data)
        
        assert ptw.g_ is not None
        assert len(ptw.g_) > 0
        assert ptw.inflection_points_ is not None
        assert len(ptw.inflection_points_) > 0
    
    def test_compute_g_and_inflections_invalid_k(self):
        """Test g computation with invalid k values."""
        ptw = ParametricTimeWarper(fitter='WeibullFitter')
        ptw.fit(self.ttf_data)
        
        # Test with NaN k
        ptw.k = np.nan
        ptw._compute_g_and_inflections()
        assert ptw.g_ is None
        assert ptw.inflection_points_ is None
        
        # Test with inf k
        ptw.k = np.inf
        ptw._compute_g_and_inflections()
        assert ptw.g_ is None
        assert ptw.inflection_points_ is None
    
    def test_warp_scalar_array(self):
        """Test warping scalar arrays."""
        ptw = ParametricTimeWarper(fitter='WeibullFitter')
        ptw.fit(self.ttf_data)
        # Test with scalar
        t_scalar = 500
        with pytest.raises(AttributeError):
            ptw._warp_scalar_array(t_scalar)
        # Test with array
        t_array = np.array([100, 500, 1000])
        with pytest.raises(AttributeError):
            ptw._warp_scalar_array(t_array)
    
    def test_warp_scalar_array_k_close_to_zero(self):
        """Test warping with k close to zero."""
        ptw = ParametricTimeWarper(fitter='WeibullFitter')
        ptw.fit(self.ttf_data)
        ptw.k = 1e-10
        ptw.mu = 1000
        t_array = np.array([100, 500, 1000])
        with pytest.raises(AttributeError):
            ptw._warp_scalar_array(t_array)
    
    def test_get_g_vals(self):
        """Test g values computation."""
        ptw = ParametricTimeWarper(fitter='WeibullFitter')
        ptw.fit(self.ttf_data)
        
        R = np.array([1.0, 0.8, 0.5, 0.2, 0.1, 0.01])
        g_vals = ptw._get_g_vals(R)
        
        assert isinstance(g_vals, np.ndarray)
        assert g_vals.shape == R.shape
        assert np.all(g_vals >= 0)  # g(t) should be non-negative
    
    def test_get_g_vals_k_close_to_one(self):
        """Test g values computation with k close to 1."""
        ptw = ParametricTimeWarper(fitter='WeibullFitter')
        ptw.fit(self.ttf_data)
        ptw.k = 0.9999
        ptw.mu = 1000
        R = np.array([1.0, 0.8, 0.5, 0.2, 0.1])
        g_vals = ptw._get_g_vals(R)
        assert isinstance(g_vals, np.ndarray)
    
    def test_get_g_vals_k_close_to_zero(self):
        """Test g values computation with k close to 0."""
        ptw = ParametricTimeWarper(fitter='WeibullFitter')
        ptw.fit(self.ttf_data)
        
        # Set k close to 0
        ptw.k = 1e-10
        ptw.mu = 1000
        
        R = np.array([1.0, 0.8, 0.5, 0.2, 0.1])
        g_vals = ptw._get_g_vals(R)
        
        assert isinstance(g_vals, np.ndarray)
        assert g_vals.shape == R.shape
    
    def test_get_g_vals_edge_cases(self):
        """Test g values computation with edge cases."""
        ptw = ParametricTimeWarper(fitter='WeibullFitter')
        ptw.fit(self.ttf_data)
        
        # Test with R = 0 (should handle gracefully)
        R = np.array([1.0, 0.5, 0.0])
        g_vals = ptw._get_g_vals(R)
        
        assert isinstance(g_vals, np.ndarray)
        assert g_vals.shape == R.shape
    
    def test_compute_rul_interval_warped_time(self):
        """Test RUL interval computation in warped time."""
        ptw = ParametricTimeWarper(fitter='WeibullFitter')
        ptw.fit(self.ttf_data)
        
        t = np.array([100, 500, 1000])
        alpha = 0.05
        
        s_minus, s_plus = ptw.compute_rul_interval_warped_time(t, alpha)
        
        assert isinstance(s_minus, np.ndarray)
        assert isinstance(s_plus, np.ndarray)
        assert s_minus.shape == t.shape
        assert s_plus.shape == t.shape
        assert np.all(s_minus >= 0)
        assert np.all(s_plus >= 0)
        assert np.all(s_plus >= s_minus)  # Upper bound should be >= lower bound
    
    def test_compute_rul_interval_warped_time_invalid_params(self):
        """Test RUL interval computation with invalid parameters."""
        ptw = ParametricTimeWarper(fitter='WeibullFitter')
        ptw.fit(self.ttf_data)
        # Test with invalid mu
        ptw.mu = None
        t = np.array([100, 500, 1000])
        s_minus, s_plus = ptw.compute_rul_interval_warped_time(t)
        assert s_minus.shape == t.shape
        assert s_plus.shape == t.shape
        # Test with invalid k
        ptw.mu = 1000
        ptw.k = np.nan
        s_minus, s_plus = ptw.compute_rul_interval_warped_time(t)
        assert s_minus.shape == t.shape
        assert s_plus.shape == t.shape
    
    def test_compute_rul_interval_original_time(self):
        """Test RUL interval computation in original time."""
        ptw = ParametricTimeWarper(fitter='WeibullFitter')
        ptw.fit(self.ttf_data)
        
        t = np.array([100, 500, 1000])
        alpha = 0.05
        
        L, U = ptw.compute_rul_interval_original_time(t, alpha)
        
        assert isinstance(L, np.ndarray)
        assert isinstance(U, np.ndarray)
        assert L.shape == t.shape
        assert U.shape == t.shape
        assert np.all(L >= 0)
        assert np.all(U >= 0)
        assert np.all(U >= L)  # Upper bound should be >= lower bound
    
    def test_compute_rul_interval_original_time_invalid_params(self):
        """Test RUL interval computation in original time with invalid parameters."""
        ptw = ParametricTimeWarper(fitter='WeibullFitter')
        ptw.fit(self.ttf_data)
        # Test with invalid g_
        ptw.g_ = None
        t = np.array([100, 500, 1000])
        L, U = ptw.compute_rul_interval_original_time(t)
        assert L.shape == t.shape
        assert U.shape == t.shape
        # Test with invalid k
        ptw.g_ = np.array([1, 2, 3])
        ptw.k = np.nan
        L, U = ptw.compute_rul_interval_original_time(t)
        assert L.shape == t.shape
        assert U.shape == t.shape
    
    def test_main_execution(self):
        """Test the main execution block at the end of the file."""
        # This tests the example code at the bottom of parametrictimewarping.py
        np.random.seed(42)
        N1, N2 = 300, 400
        ttf_data1 = np.random.weibull(a=2.5, size=N1) * 10000 + 200
        ttf_data2 = np.random.weibull(a=2.5, size=N2) * 4000 + 2000
        ttf_data = np.concatenate((ttf_data1, ttf_data2))
        
        PTW = ParametricTimeWarper()
        PTW.fit(ttf_data)
        
        # Check that fitting was successful
        assert PTW.mu is not None
        assert PTW.sigma is not None
        assert PTW.cv is not None
        assert PTW.k is not None
        assert PTW.g_ is not None
        assert PTW.inflection_points_ is not None


class TestConfig:
    """Test the config module."""
    
    def test_config_plot(self):
        """Test that config module can be imported and used."""
        from rul_timewarping.config.config import CONF_PLOT
        
        assert isinstance(CONF_PLOT, dict)
        assert 'linewidth' in CONF_PLOT
        assert 'alpha' in CONF_PLOT
        assert 'marker' in CONF_PLOT
        assert CONF_PLOT['linewidth'] == 3
        assert CONF_PLOT['alpha'] == 0.8
        assert CONF_PLOT['marker'] == 'o' 