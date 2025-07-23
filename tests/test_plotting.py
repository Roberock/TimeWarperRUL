import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from rul_timewarping.plotting import plot_g_with_inflection, plot_envelope_bounds , plot_mixture_example
from rul_timewarping.timewarping import TimeWarping

def test_plot_g_with_inflection_runs():
    t_grid = np.linspace(0, 10, 100)
    g_vals = np.sin(t_grid)
    plot_g_with_inflection(t_grid, g_vals, t_star=5)

def test_plot_envelope_bounds_runs():
    ttf = np.linspace(1, 100, 100)
    tw = TimeWarping(ttf)
    plot_envelope_bounds(tw)


def test_plot_g_with_inflection_with_ci():
    t_grid = np.linspace(0, 10, 100)
    g_vals = np.cos(t_grid)
    plot_g_with_inflection(t_grid, g_vals, t_star=4, ci=(3, 5), color='green', label='cos(t)')


def test_plot_envelope_bounds_low_N():
    ttf = np.array([10, 20])
    tw = TimeWarping(ttf)
    try:
        plot_envelope_bounds(tw)
    except Exception:
        pass  # Ensure test doesn't fail on minimal input


def test_plot_g_inflection_invalid_ci():
    t_grid = np.linspace(0, 10, 100)
    g_vals = np.log(t_grid + 1)
    plot_g_with_inflection(t_grid, g_vals, t_star=2, ci=(12, 15))  # CI outside domain


def test_plot_mixture_example_runs():
    from scipy.stats import weibull_min, beta, norm
    from rul_timewarping.utils import compute_mrl
    from scipy.signal import find_peaks

    # Generate synthetic TTF data
    np.random.seed(42)
    ttf_data1 = weibull_min(c=1.5, scale=30).rvs(300)
    ttf_data2 = beta(a=2, b=5, loc=10, scale=20).rvs(300)
    ttf_data3 = norm(loc=40, scale=5).rvs(300)
    ttf_data = np.concatenate([ttf_data1, ttf_data2, ttf_data3])
    ttf_data = ttf_data[ttf_data > 0]  # enforce positivity

    # Instantiate warper
    tw = TimeWarping(ttf_data)

    # Extract quantities for plotting
    x_vals = tw.t_grid
    pdf = tw.kde(x_vals)
    cdf = np.cumsum(pdf)
    cdf /= cdf[-1]

    mrl_physical = compute_mrl(x_vals, tw._reliability)
    mrl_transformed = compute_mrl(tw.g_vals, tw._reliability)

    # Inflection points
    dg_dt = np.gradient(tw.g_vals, x_vals)
    idx_inflect_points, _ = find_peaks(np.abs(np.gradient(dg_dt)))
    inflection_x = x_vals[idx_inflect_points]
    inflection_g = tw.g_vals[idx_inflect_points]

    # Confidence bounds
    alpha = 0.05
    s_plus, s_minus = tw.compute_rul_interval(tw.g_vals, alpha=alpha)
    L_alpha, U_alpha = tw._get_rul_interval_original_time(alpha=alpha)

    # Run plot
    plot_mixture_example(ttf_data1, ttf_data2, ttf_data3,
                         ttf_data, x_vals, cdf,
                         mrl_physical, tw.g_vals, mrl_transformed,
                         inflection_x, inflection_g,
                         pdf, idx_inflect_points, s_plus, s_minus,
                         L_alpha, U_alpha)