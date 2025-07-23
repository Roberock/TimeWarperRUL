import numpy as np
from rul_timewarping.plotting import plot_mixture_example, plot_envelope_bounds
from rul_timewarping.timewarping import TimeWarping
from scipy.stats import gaussian_kde
from scipy.integrate import cumulative_trapezoid as cumtrapz
from rul_timewarping.utils import compute_mrl, compute_g_non_parametric

from scipy.signal import find_peaks


def run_main_mixture_example_3(CASE:int = 1):
    size = 2000
    np.random.seed(42)
    if CASE == 1: # CASE 1 Simulated TTF mixture data
        ttf_data1 = np.random.weibull(a=5.5, size=size) * 2500
        ttf_data2 = np.random.beta(a=1.5, b=2, size=size) * 3500 + 3000
        ttf_data3 = np.random.normal(loc=5000, scale=2500, size=size) + 8000
        ttf_data = np.concatenate((ttf_data1, ttf_data2, ttf_data3))
    else:
        ttf_data1 = np.random.weibull(a=2, size=size) * 3500 + 2000
        ttf_data2 = np.random.weibull(a=3, size=size) * 2500 + 2000
        ttf_data3 = np.random.weibull(a=6, size=size) * 1500 + 2000
        ttf_data = np.concatenate((ttf_data1, ttf_data2, ttf_data3))

    TW = TimeWarping(ttf_data)

    # Estimate g(t)

    # Compute MRL
    mrl_physical = compute_mrl(TW.t_grid, TW._reliability)
    mrl_transformed = compute_mrl(TW.g_vals, TW._reliability)

    # --- Inflection Points Plot ---
    # Compute derivatives of g(t)
    dg_dt = np.gradient(TW.g_vals, TW.t_grid)

    inflection_idx, _ = find_peaks(dg_dt)
    inflection_x, inflection_g = TW.estimate_inflection_points()
    inflection_x = TW.t_grid[inflection_idx]
    inflection_g = TW.g_vals[inflection_idx]
    plot_envelope_bounds(TW)

    alpha=0.1
    s_plus, s_minus = TW.compute_rul_interval(TW.g_vals, alpha=alpha)
    L_alpha, U_alpha = TW._get_rul_interval_original_time(alpha=alpha)

    plot_mixture_example(ttf_data1, ttf_data2, ttf_data3, ttf_data,
                         TW.t_grid, 1 - TW._reliability,
                         mrl_physical, TW.g_vals, mrl_transformed,
                         inflection_x, inflection_g,
                         TW.kde(TW.t_grid), inflection_idx,
                         s_plus, s_minus, L_alpha, U_alpha)



if __name__ == '__main__':
    # Press the green button in the gutter to run the script.
    run_main_mixture_example_3(CASE=1)
    run_main_mixture_example_3(CASE=2)











