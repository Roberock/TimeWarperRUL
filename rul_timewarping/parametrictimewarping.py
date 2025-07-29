# -*- coding: utf-8 -*-
import autograd.numpy as np
from lifelines.fitters import KnownModelParametricUnivariateFitter
from lifelines.utils.safe_exp import safe_exp
import lifelines # Import the top-level lifelines module
from lifelines import utils, WeibullFitter, ExponentialFitter, LogNormalFitter, LogLogisticFitter, PiecewiseExponentialFitter, GeneralizedGammaFitter
from typing import Optional, Callable, Union
from scipy.signal import find_peaks
import scipy.special as sc
from scipy.integrate import trapezoid
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

Allowed_Fitters = {'WeibullFitter', 'ExponentialFitter', 'LogNormalFitter',
                   'LogLogisticFitter', 'GeneralizedGammaFitter'}  # , 'PiecewiseExponentialFitter'


class ParametricTimeWarper(KnownModelParametricUnivariateFitter):

    def __init__(self, fitter: Union[str, KnownModelParametricUnivariateFitter] = 'WeibullFitter',
                 breakpoints: Optional[list[float]] = None):
        """
        A wrapper for parametric lifelines fitters to be used in time warping routines.

        Parameters:
        -----------
        fitter : str or lifelines fitter instance, optional
            Name of a supported lifelines parametric fitter (as string), or a pre-initialized instance.
            Defaults to 'WeibullFitter'.
        """
        if isinstance(fitter, str):
            if fitter not in Allowed_Fitters:
                raise ValueError(f"Unsupported fitter '{fitter}'. Must be one of {Allowed_Fitters}.")
            # Dynamically get the fitter class from lifelines (top-level module)
            fitter_class = getattr(lifelines, fitter)

            # Handle special arguments
            if fitter_class == 'PiecewiseExponentialFitter':
                if breakpoints is None:
                    print("You must supply `breakpoints` for PiecewiseExponentialFitter.....fixing two arbitrarily")
                    breakpoints = [10, 20]
                self._fitter = fitter_class(breakpoints=breakpoints)
            else:
                self._fitter = fitter_class()

        elif isinstance(fitter, KnownModelParametricUnivariateFitter):
            self._fitter = fitter
        else:
            raise TypeError("fitter must be a string or a KnownModelParametricUnivariateFitter instance.")

        # Set the _fitted_parameter_names attribute after base class initialization
        self._fitted_parameter_names = self._fitter._fitted_parameter_names
        super().__init__()

    def fit(self, *args, **kwargs):
        """
        Fits the underlying lifelines fitter and computes derived quantities for time warping.
        """
        # 1. Delegate to the chosen lifelines fitter
        self._fitter.fit(*args, **kwargs)
        self.survival_function_ = self._fitter.survival_function_
        self.timeline = self._fitter.timeline
        self._reliability = self.survival_function_.iloc[:,
                            0].values  # Assuming the first column is the survival function
        self._compute_moments()  # 2. Compute μ and σ (moments)
        self._compute_cv_and_k()  # 3. Compute CV and k
        self._compute_g_and_inflections()  # 4. Compute warping g(t) and inflection points
        return self

    def _compute_moments(self):
        """ Computes the mean (mu) and standard deviation (sigma) of the fitted distribution. """
        cls = self._fitter.__class__.__name__
        mu = None
        sigma = None

        if cls == 'WeibullFitter':
            lam, rho = self._fitter.lambda_, self._fitter.rho_
            mu = lam * sc.gamma(1 + 1 / rho)
            var = lam ** 2 * (sc.gamma(1 + 2 / rho) - sc.gamma(1 + 1 / rho) ** 2)
            sigma = np.sqrt(var)
        elif cls == 'ExponentialFitter':
            lam = self._fitter.lambda_
            mu = 1 / lam
            sigma = mu
        elif cls == 'LogNormalFitter':
            mu_log, sigma_log = self._fitter.mu_, self._fitter.sigma_
            mu = np.exp(mu_log + 0.5 * sigma_log ** 2)
            var = (np.exp(sigma_log ** 2) - 1) * np.exp(2 * mu_log + sigma_log ** 2)
            sigma = np.sqrt(var)
        elif cls == 'GeneralizedGammaFitter':
            # Assuming the parameter names are consistent with lifelines
            try:
                mu_ = self._fitter.mu_
                sigma_ = np.exp(self._fitter.ln_sigma_)
                lambda_ = self._fitter.lambda_

                # This part is based on a common parametrization of Generalized Gamma,
                # verify if it matches lifelines' specific implementation.
                # It seems lifelines uses a different parameterization.
                # Let's fall back to numerical integration for now.
                pass  # Fallback to numerical integration

            except AttributeError:
                # Fallback to numerical integration if parameters are not accessible
                pass

        # Fallback: numerical integration using survival function R(t)
        if mu is None or sigma is None:
            timeline = self._fitter.timeline
            R = self.survival_function_.iloc[:, 0].values
            # E[T]   = ∫₀^∞ R(t) dt
            mu = trapezoid(R, timeline)
            # E[T²] = 2 ∫₀^∞ t R(t) dt
            m2 = 2 * trapezoid(timeline * R, timeline)
            sigma = np.sqrt(m2 - mu ** 2)

        # assign mean and standard deviation
        self.mu = mu
        self.sigma = sigma

    def _compute_cv_and_k(self):
        """
        After self.mu and self.sigma are set, compute:
          CV = σ / μ
          k  = (1 – CV²) / (1 + CV²)
        """
        if self.mu is None or self.sigma is None:
            self.cv = np.nan  # Cannot compute CV if moments are not available
        elif self.mu == 0:
            self.cv = np.inf  # Or handle as appropriate
        else:
            self.cv = self.sigma / self.mu

        # Avoid division by zero or near-zero for k calculation if CV is close to 1
        if np.isnan(self.cv) or np.isinf(self.cv) or self.cv ** 2 == -1:
            self.k = np.nan
        elif self.cv ** 2 + 1 == 0:  # Should not happen for real sigma and mu
            self.k = np.nan  # Or handle as appropriate
        else:
            self.k = np.clip((1.0 - self.cv ** 2) / (1.0 + self.cv ** 2), 1e-6, 1 - 1e-6)

    def _compute_g_and_inflections(self):
        """ Computes the time warping function g(t) and its inflection points. """
        # Use timeline and reliability computed in fit
        timeline = self.timeline
        R = self._reliability

        # Only compute g_ if mu and k are available
        if self.mu is not None and not np.isnan(self.k) and not np.isinf(self.k):
            self.g_ = self._get_g_vals(R)

            # Calculate gradient of g and find peaks (inflection points)
            # Ensure timeline is monotonically increasing for gradient calculation
            if not np.all(np.diff(timeline) >= 0):
                # Handle non-monotonic timeline if necessary, or raise an error
                raise ValueError("Timeline is not monotonically increasing, cannot compute gradient.")

            # Pass the timeline directly to np.gradient for correct spacing calculation
            dg = np.gradient(self.g_, timeline)

            # Find peaks in the gradient to identify inflection points
            peaks, props = find_peaks(dg, prominence=0.05 * dg.max())
            if len(peaks):
                best = peaks[np.argmax(props['prominences'])]
            else:
                best = np.argmax(dg)
            self.inflection_points_ = np.array([timeline[best]])
        else:
            self.g_ = None
            self.inflection_points_ = None

    def _warp_scalar_array(self, t_array: np.ndarray) -> np.ndarray:
        """
        Compute g(t) for arbitrary t by:
         1) interpolating the survival function R(t),
         2) applying either the parametric formula or the log-limit fallback.

        Args:
            t_array: scalar or 1D array of time(s) at which to compute g(t).
        Returns:
            1D array of g(t) values.
        """
        t_arr = np.atleast_1d(t_array)
        # 1) Interpolate R(t)
        R_interp = interp1d(
            self.timeline,
            self.survival_function_.iloc[:, 0].values,
            bounds_error=False,
            fill_value=(1.0, 0.0)
        )
        R_vals = R_interp(t_arr)

        # 2) Fallback if k ≈ 0
        if abs(self.k) < self._K_EPS:
            # g(t) ≈ -μ ln R(t)
            return -self.mu * np.log(R_vals + np.finfo(float).eps)

        # 3) General parametric warp
        exponent = self.k / (1.0 - self.k)
        return (self.mu / self.k) * (1.0 - R_vals ** exponent)

    def _get_g_vals(self, R: np.ndarray) -> np.ndarray:
        """Compute g(t) = (mu/k) [1 - R(t)^(k/(1‑k))] over the time grid."""
        # Handle case where k is close to 1 to avoid division by zero in exponent
        if np.isclose(self.k, 1.0):
            # As k approaches 1, k/(1-k) approaches infinity.
            # The term R(t)^(k/(1-k)) will approach 0 for R(t) < 1 and 1 for R(t) = 1.
            # This needs careful consideration based on the limit as k -> 1.
            # For now, let's return a placeholder or handle the limit case if possible.
            # A simplified approach for k close to 1 might be needed based on the theory.
            # For now, let's return None to indicate it cannot be computed.
            return None

        # Handle case where k is close to 0
        if np.isclose(self.k, 0.0):
            # As k approaches 0, the expression approaches mu * log(R(t)).
            return self.mu * np.log(R)

        exponent = self.k / (1 - self.k)
        # Ensure R is within [0, 1] before applying power
        R_clipped = np.clip(R, 0, 1)
        # Handle potential issues with R_clipped being 0 and exponent being negative infinity
        with np.errstate(divide='ignore', invalid='ignore'):
            term = np.power(R_clipped, exponent)
            term[(R_clipped == 0) & (exponent < 0)] = np.inf  # Handle 0 to negative power
            term[(R_clipped == 1) & (exponent > 0)] = 1  # Handle 1 to positive power

        # Handle potential division by zero if k is close to 0
        if np.isclose(self.k, 0):
            # This case is handled above, but as a fallback
            return self.mu * np.log(R_clipped)
        else:
            return (self.mu / self.k) * (1 - term)

    def compute_rul_interval_warped_time(self, t: np.ndarray, alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute upper and lower RUL interval bounds at time t based on the provided formulas.

        Parameters:
        -----------
        t : np.ndarray
            Time points at which to compute RUL intervals.
        alpha : float, optional
            Significance level for the confidence interval (default is 0.05).

        Returns:
        --------
        tuple[np.ndarray, np.ndarray]
            A tuple containing two arrays: the lower bounds and the upper bounds of the RUL interval.
        """
        if self.mu is None or np.isnan(self.k) or np.isinf(self.k) or np.isclose(self.k, 1.0):
            # Cannot compute RUL interval if mu or k are not valid
            return np.full_like(t, np.nan), np.full_like(t, np.nan)

        factor = (self.mu / self.k) - t
        exponent = self.k / (1 - self.k)
        s_plus = factor * (1 - (alpha / 2) ** exponent)
        s_minus = factor * (1 - (1 - alpha / 2) ** exponent)
        idx_valid = s_plus > s_minus
        s_plus[~idx_valid] = s_minus[~idx_valid]

        s_plus, s_minus = np.maximum(s_plus, 0.0), np.maximum(s_minus, 0.0)
        return s_minus, s_plus

    def compute_rul_interval_original_time(self, t: np.ndarray, alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
        """
        Computes the (1 - alpha)-level RUL confidence interval mapped to the original time domain.

        Implements Theorem 1 using the inverse of the warping function g⁻¹.
        Requires self.g_ and self.timeline to be computed from the fitted model.

        Parameters:
        -----------
        t : np.ndarray
            Array of time points at which to evaluate the RUL confidence interval.
        alpha : float, optional
            Significance level (default is 0.05 for 95% confidence interval).

        Returns:
        --------
        tuple[np.ndarray, np.ndarray]
            Lower and upper bounds of RUL(t) in original time.
        """
        if self.g_ is None or self.mu is None or np.isnan(self.k) or np.isinf(self.k) or np.isclose(self.k, 1.0):
            return np.full_like(t, np.nan), np.full_like(t, np.nan)

        # Define the inverse warping function g⁻¹ using interpolation
        g_inv = interp1d(self.g_, self.timeline, bounds_error=False, fill_value="extrapolate")

        # Compute transformed time t' = g(t)
        g_t = interp1d(self.timeline, self.g_, bounds_error=False, fill_value="extrapolate")(t)

        # Compute fL and fU
        exponent = self.k / (1 - self.k)
        fL = (1 - alpha / 2) ** exponent
        fU = (alpha / 2) ** exponent

        # Compute Lα(t) and Uα(t)
        center = (self.mu / self.k) * (1 - fL) + g_t * fL
        upper = (self.mu / self.k) * (1 - fU) + g_t * fU

        L = g_inv(center) - t
        U = g_inv(upper) - t

        # Ensure non-negative RUL
        return np.maximum(L, 0.0), np.maximum(U, 0.0)


if __name__ == '__main__':
    import numpy as np
    np.random.seed(42)
    N1, N2 = 300, 400
    ttf_data1 = np.random.weibull(a=2.5, size=N1) * 10000+200
    ttf_data2 = np.random.weibull(a=2.5, size=N2) * 4000+2000
    ttf_data = np.concatenate((ttf_data1, ttf_data2))


    PTW = ParametricTimeWarper()
    PTW.fit(ttf_data)