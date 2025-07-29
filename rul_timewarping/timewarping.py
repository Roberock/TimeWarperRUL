import logging
import numpy as np
from scipy.ndimage import gaussian_filter1d 
from scipy.stats import gaussian_kde
from scipy.interpolate import interp1d
from typing import Optional, Callable, Union
from scipy.signal import find_peaks
from scipy.integrate import cumulative_trapezoid as cumtrapz
import matplotlib.pyplot as plt
from lifelines import *

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
        ttf_data: Optional[np.ndarray] = None,
        bw_method: Optional[Union[str, float, Callable]] = None
    ):

        self._initialize_empty_class()

        if ttf_data is None:
            return

        # Clean input TTF data
        ttf_data = np.asarray(ttf_data)
        ttf_data = ttf_data[(ttf_data > 0) & (~np.isnan(ttf_data))]  # at the moment assume clean data and non-censored

        self.ttf_data = ttf_data
        self.N = len(ttf_data)
        self._get_k_and_stats()  # Compute basic stats
        self._initialize_g_transform(bw_method)  # Initialize KDE and compute g(t)

        if self.g_vals is None or len(self.g_vals) == 0:
            logging.warning("g_vals computation failed.")
            self.g_inv, self.g_fun = None, None
        else:
            self.g_inv = interp1d(self.g_vals, self.t_grid, bounds_error=False, fill_value="extrapolate")
            self.g_fun = interp1d(self.t_grid, self.g_vals, bounds_error=False, fill_value="extrapolate")

    def _initialize_empty_class(self):
        attrs = {  'kde': None, 't_grid': None, '_reliability': None, '_kde_cdf': None, 'g_vals': None,
                   'g_inv': None, 'g_fun': None, 'ttf_data': None, 'N': 0, 'mu': None,
                   'cv': None, 'k': None, 'g_max': None, 't_max': None }
        for key, val in attrs.items():
            setattr(self, key, val)

    def _get_k_and_stats(self):
        self.mu = np.mean(self.ttf_data)
        self.std = np.std(self.ttf_data)
        self.cv = self.std / self.mu
        self.k = np.clip((1 - self.cv ** 2) / (1 + self.cv ** 2), 1e-6, 1 - 1e-6)
        self.g_max = self.mu / (self.k + 1e-6)
        self.t_max = np.max(self.ttf_data)

    def _initialize_g_transform(self, bw_method):
        self.kde = gaussian_kde(self.ttf_data, bw_method)  # Fit KDE
        self.t_grid = np.linspace(0, self.t_max, 5000)  # grid points in time
        pdf_on_grid = self.kde(self.t_grid)   # Compute CDF of TTF and the Reliability function R(t)
        pdf_on_grid[pdf_on_grid < 1e-6] = 0  # Remove negligible tails
        self._kde_cdf = cumtrapz(pdf_on_grid, self.t_grid, initial=0)  # get cdf of the LDE as integral of the pdf---?
        self._reliability = 1 - self._kde_cdf  #get reliability function
        self.g_vals = self._get_g_vals()


    def compute_mrl(self, t_eval: np.ndarray) -> np.ndarray:
        R = interp1d(self.t_grid, self._reliability, fill_value="extrapolate")(t_eval)
        expected = cumtrapz(self._reliability, self.t_grid, initial=0)
        interp_E = interp1d(self.t_grid, expected, fill_value="extrapolate")
        return (interp_E(np.max(self.t_grid)) - interp_E(t_eval)) / R


    def _make_grid(self) -> np.ndarray:
        """Create an evaluation grid combining percentiles and uniform spacing."""
        pcts = np.percentile(self.ttf_data, np.linspace(0, 100, 300))
        base = np.linspace(0, np.max(self.ttf_data), 200)
        return np.unique(np.concatenate([pcts, base]))


    def _empirical_reliability(self, t: float) -> float:
        """Empirical reliability R(t) from sorted samples."""
        idx = np.searchsorted(self.ttf_data, t, side='right')
        return float(np.clip(1 - idx / self.N, 1e-4, 1.0))


    def _get_g_vals(self) -> np.ndarray:
        """Compute g(t) = (mu/k) [1 - R(t)^(k/(1-k))] over the time grid."""
        exponent = self.k / (1 - self.k)
        return (self.mu / self.k) * (1 - np.power(self._reliability, exponent))


    def estimate_inflection_points(self, smooth_sigma: float = 3.0, tol: float = 1e-4) -> tuple[np.ndarray, np.ndarray]:
        """
        Estimate inflection points where g''(t) changes sign (true curvature change).
        """
        g_smooth = gaussian_filter1d(self.g_vals, sigma=smooth_sigma)
        dg_dt = np.gradient(g_smooth, self.t_grid)
        threshold = np.quantile(dg_dt, 0.5)  # 10th percentile as a peak height cutoff
        idx_inflection, _ = find_peaks(dg_dt, height=threshold, prominence=threshold)

        valid_idx_inflection = idx_inflection[self.t_grid[idx_inflection] <= np.max(self.ttf_data)]
        """d2g_dt2 = np.gradient(dg_dt, self.t_grid) 
        signs = np.sign(d2g_dt2) 
        signs[np.abs(d2g_dt2) < tol] = 0  # ignore flat regions 
        sign_change = np.where(np.diff(signs) != 0)[0]
        curvature_magnitude = np.abs(d2g_dt2[sign_change])
        threshold = np.percentile(curvature_magnitude, 50)
        idx_inflection = sign_change[curvature_magnitude > threshold]
        """
        inflection_x = self.t_grid[valid_idx_inflection]
        inflection_g = self.g_vals[valid_idx_inflection]
        return inflection_x, inflection_g

    def compute_rul_interval(self, t: np.ndarray, alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
        """ Compute upper and lower RUL interval bounds at time t. """
        factor = self.mu / self.k - t
        exponent = self.k / (1 - self.k)
        s_plus = factor * (1 - (alpha / 2) ** exponent)
        s_minus = factor * (1 - (1 - alpha / 2) ** exponent)
        idx_valid = s_plus > s_minus
        s_plus[~idx_valid] = s_minus[~idx_valid]
        s_plus, s_minus = np.maximum(s_plus, 0.0), np.maximum(s_minus, 0.0)
        return s_plus, s_minus

    def _get_rul_interval_original_time(self, alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute upper and lower RUL interval bounds mapped back to original time axis.
        Avoids extrapolation by clipping values to the interpolation domain.
        """
        s_plus, s_minus = self.compute_rul_interval(self.g_vals[:len(self.g_vals)], alpha=alpha)

        # Compute input bounds for inverse interpolation
        g_lower = np.clip(self.g_vals + s_minus, 0, self.g_max)
        g_upper = np.clip(self.g_vals + s_plus, 0, self.g_max)

        g_inv_lower =  self.g_inv(g_lower)
        g_inv_upper = self.g_inv(g_upper)



        # Evaluate inverse only within the valid domain
        L_alpha = np.maximum(0, g_inv_lower - self.t_grid)
        U_alpha = np.maximum(0, g_inv_upper - self.t_grid)

        U_alpha[np.isinf(g_inv_upper)] = np.min(U_alpha)  # fixme: dirty fix on upper bound

        return L_alpha, U_alpha


    @staticmethod
    def compute_g_fun(R, k, mu):
        """Compute g(t) = (mu/k) [1 - R(t)^(k/(1-k))] over the time grid."""
        exponent = k / (1 - k)
        return (mu / k) * (1 - np.power(R, exponent))


