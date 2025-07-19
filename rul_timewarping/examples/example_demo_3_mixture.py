import numpy as np

from rul_timewarping.plotting import plot_mixture_example
from rul_timewarping.timewarping import TimeWarping
from scipy.stats import gaussian_kde
from scipy.integrate import cumulative_trapezoid as cumtrapz
from scipy.integrate import simpson as simps
from scipy.signal import find_peaks
from rul_timewarping.utils import compute_mrl, compute_g_non_parametric

def run_main_mixture_example(CASE:int = 1):
    size = 10000
    np.random.seed(42)

    if CASE == 1: # CASE 1 Simulated TTF mixture data
        ttf_data1 = np.random.weibull(a=5.5, size=size) * 3500
        ttf_data2 = np.random.beta(a=1.5, b=2, size=size) * 3500 + 3000
        ttf_data3 = np.random.normal(loc=5000, scale=2500, size=size) + 5000
        ttf_data = np.concatenate((ttf_data1, ttf_data2, ttf_data3))
    else:
        ttf_data1 = np.random.weibull(a=5.5, size=size) * 3500
        ttf_data2 = np.random.weibull(a=2, size=size) * 1500 + 2000
        ttf_data3 = np.random.weibull(a=6, size=size) * 500
        ttf_data = np.concatenate((ttf_data1, ttf_data2, ttf_data3))

    # Estimate g(t)
    k_est, mu_est, x_vals, g_vals , R = compute_g_non_parametric(ttf_data)

    # Fit KDE and get R(t)
    kde_combined = gaussian_kde(ttf_data)
    pdf = kde_combined(x_vals)
    cdf = cumtrapz(pdf, x_vals, initial=0)
    Reliability_values = 1 - cdf

    # Compute MRL
    mrl_physical = compute_mrl(x_vals, Reliability_values)
    mrl_transformed = compute_mrl(g_vals, Reliability_values)


    # --- Inflection Points Plot ---
    # Compute derivatives of g(t)
    dg_dt = np.gradient(g_vals, x_vals)
    d2g_dt2 = np.gradient(dg_dt, x_vals)
    sign_change = np.where(np.diff(np.sign(d2g_dt2)) != 0)[0]  # otherwise find the maximum value(s) of dg_dt
    inflection_x = x_vals[sign_change]
    inflection_g = g_vals[sign_change]

    # Compute RUL interval
    TW = TimeWarping(ttf_data)
    alpha=0.1
    s_plus, s_minus = TW.compute_rul_interval(TW.g_vals, alpha=alpha)
    L_alpha, U_alpha = TW.compute_rul_interval_original_time(alpha=alpha)

    plot_mixture_example(ttf_data1, ttf_data2, ttf_data3, ttf_data, x_vals, cdf,
                         mrl_physical, g_vals, mrl_transformed,
                         inflection_x, inflection_g,
                         pdf, sign_change, s_plus, s_minus, L_alpha, U_alpha)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    run_main_mixture_example()











