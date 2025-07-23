from rul_timewarping.config.config import *
import numpy as np
import matplotlib.pyplot as plt
from rul_timewarping.timewarping import TimeWarping
from rul_timewarping.utils import compute_mrl, compute_g_non_parametric
from rul_timewarping.plotting import plot_envelope_bounds

def run_bivariate_example_2():

    np.random.seed(42)
    N1, N2 = 5_000, 3_000
    ttf_data1 = np.random.weibull(a=2.5, size=N1) * 10000+200
    ttf_data2 = np.random.weibull(a=2.5, size=N2) * 2000+2000
    ttf_data = np.concatenate((ttf_data1, ttf_data2))

    # Initialize
    TW = TimeWarping(ttf_data)

    # Get inflection points
    inflection_points_t, inflection_points_g = TW.estimate_inflection_points()

    print('inflection_points_t = ', inflection_points_t)
    print('inflection_points_g = ', inflection_points_g)

    time_g, time_t = TW.g_vals, TW.t_grid
    # Compute MRL
    Reliability_values = TW._reliability  # get non parametric reliability estimate

    alpha = 0.1
    mrl_physical = compute_mrl(time_t, Reliability_values)
    mrl_transformed = compute_mrl(time_g, Reliability_values)

    # Compute RUL intervals
    s_plus, s_minus = TW.compute_rul_interval(time_g, alpha=alpha)
    L_alpha, U_alpha = TW._get_rul_interval_original_time(alpha=alpha)

    # Create subplots
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    # Left subplot: Mixture PDF with inflection points
    bins = 100
    ax[0].hist(ttf_data1, bins=bins, density=True, alpha=0.5, color='red', label='Component 1')
    ax[0].hist(ttf_data2, bins=bins, density=True, alpha=0.5, color='blue', label='Component 2')
    ax[0].hist(ttf_data, bins=bins, density=True, alpha=0.3, color='black', label='Mixture')
    ax[0].plot(time_t, TW.kde(time_t), color='navy', label='Mixture PDF (KDE)')
    ax[0].plot(time_t, TW.kde(time_t), color='navy', label='Mixture PDF (KDE)')
    ax[0].scatter(inflection_points_t, TW.kde(inflection_points_t), color='red', zorder=5, label='Inflection Times')
    ax[0].set_title('Mixture PDF with Inflection Points')
    ax[0].set_xlabel('Time to Failure')
    ax[0].set_ylabel('Density')
    ax[0].legend()
    ax[0].grid(True)

    # Right subplot: Time transformation g(t) with inflection points
    ax[1].plot(time_t, time_g, color='black', label=r'$g(t)$')
    ax[1].scatter(inflection_points_t, inflection_points_g, color='red', zorder=5, label='Inflection Points')
    ax[1].set_title('Time Transformation $g(t)$')
    ax[1].set_xlabel('Time to Failure')
    ax[1].set_ylabel('Transformed Time $g(t)$')
    ax[1].legend()
    ax[1].grid(True)

    plt.tight_layout()
    plt.show()

    fig, axs = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    # Plot vs g(t)
    axs[0].plot(time_g, s_plus, 'g--', label='s⁺(g)', linewidth=CONF_PLOT['linewidth'])
    axs[0].plot(time_g, mrl_transformed , label="MRL (transformed time)", color="red")
    axs[0].plot(time_g, s_minus , 'b--', label='s⁻(g)', linewidth=CONF_PLOT['linewidth'])
    axs[0].set_xlabel('g(t)')
    axs[0].set_ylabel('RUL bounds')
    axs[0].set_title('RUL vs g(t)')
    axs[0].grid(True)
    axs[0].legend()

    # Plot vs time t
    axs[1].plot(time_t, U_alpha, 'g--', label='U(t)', linewidth=CONF_PLOT['linewidth'])
    axs[1].plot(time_t, mrl_physical, label="MRL (Physical time)", color="red")
    axs[1].plot(time_t, L_alpha, 'b--', label='L(t)', linewidth=CONF_PLOT['linewidth'])
    axs[1].set_xlabel('Time t')
    axs[1].set_title('RUL vs time t')
    axs[1].grid(True)
    axs[1].legend()
    plt.tight_layout()
    plt.show()

    plot_envelope_bounds(TW)



if __name__ == '__main__':
    # Press the green button in the gutter to run the script.
    run_bivariate_example_2()