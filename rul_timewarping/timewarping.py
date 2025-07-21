import logging
import numpy as np
from scipy.ndimage import gaussian_filter1d
from typing import Literal
from rul_timewarping.utils import compute_g_non_parametric, get_non_param_reliability, compute_mrl
from scipy.stats import gaussian_kde
from scipy.interpolate import interp1d
from typing import Optional, Callable, Union
from scipy.signal import find_peaks



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
        ttf_data: np.ndarray,
        bw_method: Optional[Union[str, float, Callable]] = None
    ):
        self.ttf_data = ttf_data[ttf_data > 0]
        self.ttf_data = ttf_data[~np.isnan(ttf_data)]
        self.N = len(self.ttf_data)
        self.mu = np.mean(self.ttf_data)
        self.cv = np.std(self.ttf_data) / (self.mu + 1e-6)

        if self.N >= 2:
            self._initialize_g_transform(bw_method)
        else:
            self.kde = None
            self.k = 0
            self.t_grid = np.linspace(0, np.max(self.ttf_data) + 100, 3000)
            self._reliability = None
            self._kde_cdf = None
            self.g_vals = None
            self.g_inv = None
            self.g_fun = None

    def _initialize_g_transform(self, bw_method):
        k_est, mu_est, x_vals, g_vals, reliability, gauss_kde = compute_g_non_parametric(self.ttf_data, bw_method)

        if g_vals is None or len(g_vals) == 0:
            logging.warning("g_vals computation failed.")
            self.g_vals, self.g_inv, self.g_fun = None, None, None
            return

        self.kde = gauss_kde
        self.k = np.clip(k_est, 1e-3, 0.999)
        self.t_grid = x_vals
        self.g_vals = g_vals
        self._reliability = reliability
        self._kde_cdf = 1 - reliability
        self.g_inv = interp1d(g_vals, x_vals, bounds_error=False, fill_value="extrapolate")
        self.g_fun = interp1d(x_vals, g_vals, bounds_error=False, fill_value="extrapolate")

    def _make_grid(self) -> np.ndarray:
        """Create an evaluation grid combining percentiles and uniform spacing."""
        pcts = np.percentile(self.ttf_data, np.linspace(0, 100, 300))
        base = np.linspace(0, np.max(self.ttf_data), 200)
        return np.unique(np.concatenate([pcts, base]))

    def _empirical_reliability(self, t: float) -> float:
        """Empirical reliability R(t) from sorted samples."""
        idx = np.searchsorted(self.ttf_data, t, side='right')
        return float(np.clip(1 - idx / self.N, 1e-3, 1.0))

    def compute_g_vals(self) -> np.ndarray:
        """Compute g(t) = (mu/k) [1 - R(t)^(k/(1-k))] over the time grid."""
        R_vals = np.array([self._reliability(t) for t in self.t_grid])
        exponent = self.k / (1 - self.k)
        return (self.mu / self.k) * (1 - np.power(R_vals, exponent))

    def estimate_inflection_points(self, smooth_sigma: float = 3.0, tol: float = 1e-4) -> tuple[np.ndarray, np.ndarray]:
        """
        Estimate inflection points where g''(t) changes sign (true curvature change).
        """
        g_smooth = gaussian_filter1d(self.g_vals, sigma=smooth_sigma)
        dg_dt = np.gradient(g_smooth, self.t_grid)

        idx_inflection, _ = find_peaks(dg_dt)

        """d2g_dt2 = np.gradient(dg_dt, self.t_grid) 
        signs = np.sign(d2g_dt2) 
        signs[np.abs(d2g_dt2) < tol] = 0  # ignore flat regions 
        sign_change = np.where(np.diff(signs) != 0)[0]
        curvature_magnitude = np.abs(d2g_dt2[sign_change])
        threshold = np.percentile(curvature_magnitude, 50)
        idx_inflection = sign_change[curvature_magnitude > threshold]
        """
        inflection_x = self.t_grid[idx_inflection]
        inflection_g = self.g_vals[idx_inflection]

        return inflection_x, inflection_g

    def compute_rul_interval(self, t: np.ndarray, alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute upper and lower RUL interval bounds at time t.
        """
        factor = self.mu / self.k - t
        exponent = self.k / (1 - self.k)
        s_plus = factor * (1 - (alpha / 2) ** exponent)
        s_minus = factor * (1 - (1 - alpha / 2) ** exponent)

        return np.maximum(s_plus, 0.0), np.maximum(s_minus, 0.0)

    def compute_rul_interval_original_time(self,  alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute upper and lower RUL interval bounds mapped back to original time axis.
        """
        s_plus, s_minus = self.compute_rul_interval(self.g_vals, alpha=alpha)
        L_alpha = self.g_inv(self.g_vals + s_minus) - self.t_grid
        U_alpha = self.g_inv(self.g_vals + s_plus) - self.t_grid
        return L_alpha, U_alpha
