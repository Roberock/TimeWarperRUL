import numpy as np
from rul_timewarping.plotting import plot_mixture_example, plot_envelope_bounds
from rul_timewarping.timewarping import TimeWarping
from scipy.stats import gaussian_kde
from scipy.integrate import cumulative_trapezoid as cumtrapz
from rul_timewarping.utils import compute_mrl, compute_g_non_parametric

from scipy.signal import find_peaks

def check_inflection_ok(TW, inflection_idx):
    # 1. Estimate PDF f(t) via your KDE object, and survival R(t):
    pdf_vals = TW.kde(TW.t_grid)  # f(t)
    R_vals = TW._reliability  # R(t)

    # 2. Compute hazard lambda(t) = f/R
    lambda_vals = pdf_vals / R_vals

    # 3. Compute derivatives dλ/dt and d(log λ)/dt by finite differences
    d_lambda_dt = np.gradient(lambda_vals, TW.t_grid)
    d_loglambda_dt = np.gradient(np.log(lambda_vals), TW.t_grid)

    # 4. Grab your slope k
    k = TW.k

    # 5. For each inflection index, compute residuals for (A0) and (A1)
    t_star_list = TW.t_grid[inflection_idx]
    lam_star = lambda_vals[inflection_idx]
    dlam_star = d_lambda_dt[inflection_idx]
    dlogl_star = d_loglambda_dt[inflection_idx]

    # 6. Evaluate residuals
    r0 = dlam_star - (k / (1 - k)) * lam_star ** 2
    r1 = dlogl_star - (k / (1 - k)) * lam_star

    # 7. Report
    for i, t_star in enumerate(t_star_list):
        print(f"t* = {t_star:.3f}")
        print(f"  (A0) residual: {r0[i]:.3e}")
        print(f"  (A1) residual: {r1[i]:.3e}")
        print("  passes?", np.allclose([r0[i], r1[i]], [0, 0], atol=1e-2))
        print("-" * 40)


def run_main_mixture_example_3(CASE:int = 1):
    size = 20_000
    np.random.seed(42)
    if CASE == 1: # CASE 1 Simulated TTF mixture data
        ttf_data1 = np.random.weibull(a=5.5, size=size) * 2500
        ttf_data2 = np.random.beta(a=1.5, b=2, size=size) * 3500 + 3000
        ttf_data3 = np.random.normal(loc=5000, scale=2500, size=size) + 8000
        ttf_data = np.concatenate((ttf_data1, ttf_data2, ttf_data3))
    else:
        ttf_data1 = np.random.weibull(a=2, size=size) * 3500 + 2000
        ttf_data2 = np.random.weibull(a=3, size=size) * 2500 + 2000
        ttf_data3 = np.random.weibull(a=6, size=size) * 1500 + 100
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
    # inflection_x, inflection_g = TW.estimate_inflection_points()
    inflection_x = TW.t_grid[inflection_idx]
    inflection_g = TW.g_vals[inflection_idx]
    plot_envelope_bounds(TW, case_number=CASE)

    alpha = 0.05
    s_plus, s_minus = TW.compute_rul_interval(TW.g_vals, alpha=alpha)
    L_alpha, U_alpha = TW._get_rul_interval_original_time(alpha=alpha)

    plot_mixture_example(ttf_data1, ttf_data2, ttf_data3, ttf_data,
                         TW.t_grid, 1 - TW._reliability,
                         mrl_physical, TW.g_vals, mrl_transformed,
                         inflection_x, inflection_g,
                         TW.kde(TW.t_grid), inflection_idx,
                         s_plus, s_minus, L_alpha, U_alpha, case_number=CASE)

    check_inflection_ok(TW, inflection_idx)



if __name__ == '__main__':
    # Press the green button in the gutter to run the script.
    run_main_mixture_example_3(CASE=1)
    # run_main_mixture_example_3(CASE=2)











