import numpy as np
from scipy.ndimage import gaussian_filter1d
from typing import Literal
from rul_timewarping.utils import compute_g_non_parametric, get_non_param_reliability
from scipy.stats import gaussian_kde

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
        # Prepare data
        self.ttf_data = ttf_data[ttf_data > 0]
        self.N = len(ttf_data)
        self.mu = np.mean(self.ttf_data)
        self.cv = np.std(self.ttf_data)  / (self.mu + 1e-10)
        self.kde = gaussian_kde(ttf_data)
        k_est, mu_est, x_vals, g_vals, reliability = compute_g_non_parametric(ttf_data)
        self.k = k_est
        self.t_grid = x_vals
        self._reliability = reliability
        self._kde_cdf = 1 - reliability

        # Compute g(t)
        self.g_vals = g_vals


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

    def estimate_inflection_points(
        self, ):
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