import logging

import numpy as np
from scipy.ndimage import gaussian_filter1d
from typing import Literal
from rul_timewarping.utils import compute_g_non_parametric, get_non_param_reliability
from scipy.stats import gaussian_kde
from scipy.interpolate import interp1d

class TimeWarping:
    """
    Non-parametric time warping transformation for RUL analysis.

    Attributes:
        ttf_data : Sorted TTF samples (positive values only)
        mu       : Mean time to failure
        k        : Degradation slope estimated from coefficient of variation
        t_grid   : Time grid used for evaluation
        g_vals   : Time-transformed values g(t) on the grid
    """

    def __init__(
        self,
        ttf_data: np.ndarray
    ):

        self.ttf_data = ttf_data[ttf_data > 0]
        self.N = len(ttf_data)
        self.mu = np.mean(self.ttf_data)
        self.cv = np.std(self.ttf_data)  / (self.mu + 1e-10)

        # Prepare data
        if len(ttf_data) < 2:
            self.kde = None
            self.k = 0
            self.t_grid = np.linspace(0, np.max(ttf_data), 5000)
            self._reliability = None
            self._kde_cdf = None
            self.g_vals = None
            self.g_inv =None
        else:
            self.kde = gaussian_kde(ttf_data)
            k_est, mu_est, x_vals, g_vals, reliability = compute_g_non_parametric(ttf_data)
            self.k = k_est
            self.t_grid = x_vals
            self._reliability = reliability
            self._kde_cdf = 1 - reliability
            # Compute g(t)
            self.g_vals = g_vals
            self.g_inv = self._get_g_inverse()

    def _make_grid(self) -> np.ndarray:
        """Create an evaluation grid combining percentiles and uniform spacing."""
        pcts = np.percentile(self.ttf_data, np.linspace(0, 100, 300))
        base = np.linspace(0, np.max(self.ttf_data), 200)
        return np.unique(np.concatenate([pcts, base]))


    def _empirical_reliability(self, t: float) -> float:
        """Empirical reliability R(t) from sorted samples."""

        idx = np.searchsorted(self.ttf_data, t, side='right')
        return float(np.clip(1 - idx / self.N, 1e-6, 1.0))


    def compute_g_vals(self) -> np.ndarray:
        """Compute g(t) = (mu/k) [1 - R(t)^(k/(1-k))] over the time grid."""
        R_vals = np.array([self._reliability(t) for t in self.t_grid])
        exponent = self.k / (1 - self.k)
        return (self.mu / self.k) * (1 - np.power(R_vals, exponent))

    def _get_g_inverse(self):
        # Construct numerical inverse of g(t)
        return interp1d(self.g_vals, self.t_grid, bounds_error=False, fill_value="extrapolate")


    def estimate_inflection_points(self):
        """
        Estimate inflection points where g''(t) crosses zero.

        Args:
            smooth_sigma: Gaussian smoothing sigma for g(t)
            tol: Tolerance to consider near-zero second derivative
            mode: 'all' returns all points; 'first' returns the first occurrence

        Returns:
            Inflection times on the grid.
        """
        # Smooth g(t)
        # Compute derivatives of g(t)
        dg_dt = np.gradient(self.g_vals, self.t_grid)
        d2g_dt2 = np.gradient(dg_dt, self.t_grid)
        sign_change = np.where(np.diff(np.sign(d2g_dt2)) != 0)[0]
        inflection_x = self.t_grid[sign_change]
        inflection_g = self.g_vals[sign_change]
        return inflection_x, inflection_g


    def compute_rul_interval(self, t: np.ndarray, alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute upper and lower RUL interval bounds at time t.

        Args:
            t     : Array of time points
            alpha : Significance level (default 0.05 for 95% interval)

        Returns:
            Tuple of arrays (s_plus, s_minus) representing upper and lower bounds
        """
        factor = self.mu / self.k - t
        exponent = self.k / (1 - self.k)
        s_plus = factor * (1 - (alpha / 2) ** exponent)
        s_minus = factor * (1 - (1 - alpha / 2) ** exponent)

        return s_plus, s_minus

    def compute_rul_interval_original_time(self, alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute upper and lower RUL interval bounds at time t.

        Args:
            t     : Array of time points
            alpha : Significance level (default 0.05 for 95% interval)

        Returns:
            Tuple of arrays (s_plus, s_minus) representing upper and lower bounds
        """

        # Compute RUL intervals
        s_plus, s_minus = self.compute_rul_interval(self.g_vals, alpha=alpha)
        idx_valid = s_plus > s_minus

        # Compute lower and upper bounds in time domain
        g_t = self.g_vals[idx_valid]
        s_minus_g = s_minus[idx_valid]
        s_plus_g = s_plus[idx_valid]

        L_alpha = self.g_inv(g_t + s_minus_g) - self.t_grid[idx_valid]
        U_alpha = self.g_inv(g_t + s_plus_g) - self.t_grid[idx_valid]

        return L_alpha, U_alpha