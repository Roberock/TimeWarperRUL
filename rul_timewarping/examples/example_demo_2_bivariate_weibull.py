from rul_timewarping.config.config import *
import numpy as np
import matplotlib.pyplot as plt
from rul_timewarping.timewarping import TimeWarping
from rul_timewarping.utils import compute_mrl, compute_g_non_parametric

def run_bivariate_example():

    np.random.seed(42)
    N1, N2 = 1_000, 2_000
    ttf_data1 = np.random.weibull(a=2.5, size=N1) * 10000+1000
    ttf_data2 = np.random.weibull(a=2.5, size=N2) * 2000+2000

    ttf_data = np.concatenate((ttf_data1, ttf_data2))

    # Initialize
    tw = TimeWarping(ttf_data)

    # Get inflection points
    inflection_points_t, inflection_points_g = tw.estimate_inflection_points()

    print('inflection_points_t = ', inflection_points_t)
    print('inflection_points_g = ', inflection_points_g)

    t_grid = tw.t_grid

    # Create subplots
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))

    # Left subplot: Mixture PDF with inflection points
    bins = 100
    ax[0].hist(ttf_data1, bins=bins, density=True, alpha=0.5, color='red', label='Component 1')
    ax[0].hist(ttf_data2, bins=bins, density=True, alpha=0.5, color='blue', label='Component 2')
    ax[0].hist(ttf_data, bins=bins, density=True, alpha=0.3, color='black', label='Mixture')
    ax[0].plot(t_grid, tw.kde(t_grid), color='navy', label='Mixture PDF (KDE)')
    ax[0].plot(t_grid, tw.kde(t_grid), color='navy', label='Mixture PDF (KDE)')
    ax[0].scatter(inflection_points_t, tw.kde(inflection_points_t), color='red', zorder=5, label='Inflection Times')
    ax[0].set_title('Mixture PDF with Inflection Points')
    ax[0].set_xlabel('Time to Failure')
    ax[0].set_ylabel('Density')
    ax[0].legend()
    ax[0].grid(True)

    # Right subplot: Time transformation g(t) with inflection points
    ax[1].plot(t_grid, tw.g_vals, color='black', label=r'$g(t)$')
    ax[1].scatter(inflection_points_t, inflection_points_g, color='red', zorder=5, label='Inflection Points')
    ax[1].set_title('Time Transformation $g(t)$')
    ax[1].set_xlabel('Time to Failure')
    ax[1].set_ylabel('Transformed Time $g(t)$')
    ax[1].legend()
    ax[1].grid(True)

    plt.tight_layout()
    plt.show()


    # Compute MRL
    Reliability_values = tw._reliability  # get non parametric reliability estimate

    alpha = 0.1
    mrl_physical = compute_mrl(tw.t_grid, Reliability_values)
    mrl_transformed = compute_mrl(tw.g_vals, Reliability_values)

    # Compute RUL intervals
    s_plus, s_minus = tw.compute_rul_interval(tw.g_vals, alpha=alpha)
    idx_valid = s_plus > s_minus

    L_alpha, U_alpha = tw.compute_rul_interval_original_time(alpha=alpha)

    fig, axs = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    # Plot vs g(t)
    axs[0].plot(tw.g_vals[idx_valid], s_plus[idx_valid], 'g--', label='s⁺(g)', linewidth=CONF_PLOT['linewidth'])
    axs[0].plot(tw.g_vals[idx_valid], mrl_transformed[idx_valid], label="MRL (transformed time)", color="red")
    axs[0].plot(tw.g_vals[idx_valid], s_minus[idx_valid], 'b--', label='s⁻(g)', linewidth=CONF_PLOT['linewidth'])
    axs[0].set_xlabel('g(t)')
    axs[0].set_ylabel('RUL bounds')
    axs[0].set_title('RUL vs g(t)')
    axs[0].grid(True)
    axs[0].legend()

    # Plot vs time t
    axs[1].plot(tw.t_grid[idx_valid], U_alpha, 'g--', label='U(t)', linewidth=CONF_PLOT['linewidth'])
    axs[1].plot(tw.t_grid[idx_valid], mrl_physical[idx_valid], label="MRL (Physical time)", color="red")
    axs[1].plot(tw.t_grid[idx_valid], L_alpha, 'b--', label='L(t)', linewidth=CONF_PLOT['linewidth'])
    axs[1].set_xlabel('Time t')
    axs[1].set_title('RUL vs time t')
    axs[1].grid(True)
    axs[1].legend()
    plt.tight_layout()
    plt.show()


    # Loop over different alpha values
    # Loop over different alpha values
    fig, axs = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    colors = ['blue', 'green', 'purple']
    alphas = [0.05, 0.02, 0.001]

    # Precompute MRL curves (independent of alpha)
    mrl_physical = compute_mrl(tw.t_grid, Reliability_values)
    mrl_transformed = compute_mrl(tw.g_vals, Reliability_values)

    # Plot MRL once
    axs[0].plot(tw.g_vals, mrl_transformed, color="black", label="MRL (g-space)", linewidth=2)
    axs[1].plot(tw.t_grid, mrl_physical, color="black", label="MRL (t-space)", linewidth=2)

    for color, alpha in zip(colors, alphas):
        s_plus, s_minus = tw.compute_rul_interval(tw.g_vals, alpha=alpha)
        idx_valid = s_plus > s_minus
        L_alpha, U_alpha = tw.compute_rul_interval_original_time(alpha=alpha)

        label_g_plus = rf"$s^+$ (α={alpha})"
        label_g_minus = rf"$s^-$ (α={alpha})"
        label_U = rf"$U(t)$ (α={alpha})"
        label_L = rf"$L(t)$ (α={alpha})"

        axs[0].plot(tw.g_vals[idx_valid], s_plus[idx_valid], '--', color=color, label=label_g_plus,
                    linewidth=CONF_PLOT['linewidth'])
        axs[0].plot(tw.g_vals[idx_valid], s_minus[idx_valid], ':', color=color, label=label_g_minus,
                    linewidth=CONF_PLOT['linewidth'])

        axs[1].plot(tw.t_grid[idx_valid], U_alpha, '--', color=color, label=label_U, linewidth=CONF_PLOT['linewidth'])
        axs[1].plot(tw.t_grid[idx_valid], L_alpha, ':', color=color, label=label_L, linewidth=CONF_PLOT['linewidth'])

    # Final plot settings
    axs[0].set_xlabel('Transformed time g(t)')
    axs[0].set_ylabel('RUL bounds')
    axs[0].set_title('RUL vs g(t)')
    axs[0].grid(True)
    axs[0].legend()

    axs[1].set_xlabel('Original time t')
    axs[1].set_title('RUL vs time t')
    axs[1].grid(True)
    axs[1].legend()

    plt.tight_layout()
    plt.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    run_bivariate_example()