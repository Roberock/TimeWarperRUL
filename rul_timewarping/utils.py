import numpy as np
from scipy.stats import gaussian_kde
from scipy.integrate import cumulative_trapezoid as cumtrapz
from scipy.integrate import simpson as simps

def ecdf(data):
    """Empirical cumulative distribution function."""
    x = np.sort(data)
    y = np.arange(1, len(x)+1) / len(x)
    return x, y

def empirical_reliability(data, t):
    """Empirical reliability function R(t)."""
    data = np.sort(data)
    n = len(data)
    idx = np.searchsorted(data, t, side='right')
    return float(np.clip(1 - idx / n, 1e-6, 1))

def get_non_param_reliability(x_vals, ttf_data):
    kde_combined = gaussian_kde(ttf_data)
    pdf = kde_combined(x_vals)
    cdf = cumtrapz(pdf, x_vals, initial=0)
    return 1 - cdf



# Function to estimate degradation slope k from TTF data
def estimate_k(ttf_data):
    mu = np.mean(ttf_data)
    sigma = np.std(ttf_data)
    cv = sigma / mu
    k = (1 - cv ** 2) / (1 + cv ** 2) # if cv < 1 else 1e-3
    return k, mu


def compute_g_non_parametric(ttf_data):
    # Estimate k and mu
    k, mu = estimate_k(ttf_data)
    # Fit KDE
    kde = gaussian_kde(ttf_data)
    x = np.linspace(0, np.max(ttf_data), 2000)
    pdf = kde(x)
    # Compute CDF and Reliability R(t)
    cdf = cumtrapz(pdf, x, initial=0)
    R = 1 - cdf
    # Compute time transformation g(t)
    g = (mu / k) * (1 - R ** (k / (1 - k)))
    return k, mu, x, g

# MRL computation
def compute_mrl(x_vals, R_vals):
    """ compute MRL(t) by tail integration """
    mrl = []
    for i, t in enumerate(x_vals):
        tail = x_vals[i:]
        R_tail = R_vals[i:]
        integral = simps(R_tail, tail)
        mrl.append(integral / R_vals[i] if R_vals[i] > 1e-6 else 0)
    return np.array(mrl)
