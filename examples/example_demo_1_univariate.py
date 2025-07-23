from rul_timewarping.config.config import *
import numpy as np
import matplotlib.pyplot as plt
from rul_timewarping.timewarping import TimeWarping
from rul_timewarping.utils import compute_mrl


def run_univariate_example_1():
    """
    Demonstrates univariate time warping on Weibull-distributed time-to-failure (TTF) data:
    - Transforms time using a learned warping function g(t)
    - Estimates nonparametric reliability and inflection points
    - Computes Mean Residual Life (MRL) and RUL intervals in both original and warped time
    - Visualizes MRL and RUL bounds in both domains
    """
    ttf_data = np.random.weibull(a=2.5, size=200) * 1000

    # Initialize time warping
    TW = TimeWarping(ttf_data)

    # Extract time grid and transformed time
    g, t = TW.g_vals, TW.t_grid
    Reliability_values = TW._reliability

    # Estimate inflection points
    inflection_points_t, inflection_points_g = TW.estimate_inflection_points()
    print('inflection_points_t = ', inflection_points_t)
    print('inflection_points_g = ', inflection_points_g)

    # Compute Mean Residual Life
    alpha = 0.05
    mrl_physical = compute_mrl(t, Reliability_values)
    mrl_transformed = compute_mrl(g, Reliability_values)

    # Compute RUL intervals
    s_plus, s_minus = TW.compute_rul_interval(g, alpha=alpha)
    L_alpha, U_alpha = TW._get_rul_interval_original_time(alpha=alpha)


    # Print some results
    print(f"{'Index':>5} | {'g(t)':>10} | {'RUL in g-space':>20} | {'t':>10} | {'RUL in t-space':>20}")
    print("-" * 85)
    n_samples = 10
    indices = np.linspace(0, len(g) /2, n_samples, dtype=int)
    for i in indices:
        print(f"{i:5d} | {g[i]:10.2f} | [ {s_minus[i]:7.2f}, {s_plus[i]:7.2f} ] |"
              f" {t[i]:10.2f} | [ {L_alpha[i]:7.2f}, {U_alpha[i]:7.2f} ]")


    ## VISUALIZE ------------------------
    # Show g(t) with inflection points, coordinates, and data distribution
    fontsize = 18
    fig, axs = plt.subplots(1, 2, figsize=(18, 5))

    # 1. Left: g(t) with inflection points
    axs[0].plot(t, g, label='g(t)', linewidth=2)
    axs[0].scatter(inflection_points_t, inflection_points_g, color='r', label='Inflection points')
    axs[0].set_xlabel('Time t', fontsize=fontsize)
    axs[0].set_ylabel('g(t)', fontsize=fontsize)
    axs[0].set_title('Time Warping Function', fontsize=fontsize)
    axs[0].legend()
    axs[0].grid(True)

    axs[1].hist(ttf_data, bins=30, color='steelblue', edgecolor='k', alpha=0.7, density=True)
    axs[1].set_xlabel('TTF (time-to-failure)', fontsize=fontsize)
    axs[1].set_ylabel('Density', fontsize=fontsize)
    axs[1].set_title('Distribution of TTF Data', fontsize=fontsize)
    axs[1].grid(True)
    plt.tight_layout()
    plt.show()



    # Plot RUL bounds in both g(t) and t
    fig, axs = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    axs[0].plot(g, s_plus, 'g--', label='s⁺(g)', linewidth=2)
    axs[0].plot(g, mrl_transformed, 'r', label='MRL(g)', linewidth=2)
    axs[0].plot(g, s_minus, 'b--', label='s⁻(g)', linewidth=2)
    axs[0].set_xlabel('Transformed time g(t)', fontsize=fontsize)
    axs[0].set_ylabel('Remaining Useful Life', fontsize=fontsize)
    axs[0].set_title('RUL vs g(t)', fontsize=fontsize)
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(t, U_alpha, 'g--', label='U(t)', linewidth=2)
    axs[1].plot(t, mrl_physical, 'r', label='MRL(t)', linewidth=2)
    axs[1].plot(t, L_alpha, 'b--', label='L(t)', linewidth=2)
    axs[1].set_xlabel('Time t', fontsize=fontsize)
    axs[1].set_title('RUL vs Time t', fontsize=fontsize)
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # Press the green button in the gutter to run the script.
    run_univariate_example_1()