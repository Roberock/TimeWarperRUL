import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from rul_timewarping.timewarping import TimeWarping
from rul_timewarping.utils import *

def plot_g_with_inflection(t_grid, g_vals, t_star, ci=None, color='blue', label='g(t)'):
    """
    Plot g(t) with inflection point and optional confidence interval.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(t_grid, g_vals, color=color, label=label)
    plt.axvline(t_star, color='red', linestyle='--', label='Inflection t*')
    if ci is not None:
        plt.axvspan(ci[0], ci[1], color='gray', alpha=0.2, label='CI')
    plt.xlabel('Time')
    plt.ylabel('g(t)')
    plt.title('g(t) and Inflection Point')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_envelope_bounds(TimeWarper: TimeWarping):
    """
    Plot RUL envelopes at multiple alpha levels in original time and warped time domains.
    """
    fig, axs = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    color = 'blue'
    alphas = [0.5, 0.3, 0.20, 0.15, 0.1, 0.05, 0.02, 0.01, 0.001]
    fontsize = 14
    linewidth = 2

    # Precompute MRL curves
    mrl_physical = compute_mrl(TimeWarper.t_grid, TimeWarper._reliability)
    mrl_transformed = compute_mrl(TimeWarper.g_vals, TimeWarper._reliability)


    # Loop over alphas
    for alpha in alphas:
        s_plus, s_minus = TimeWarper.compute_rul_interval(TimeWarper.g_vals, alpha=alpha)
        L_alpha, U_alpha = TimeWarper._get_rul_interval_original_time(alpha=alpha)

        time_g, time_t = TimeWarper.g_vals, TimeWarper.t_grid

        index_2_keep = s_plus > 0
        time_g = time_g[index_2_keep]
        time_t = time_t[index_2_keep]
        s_minus = s_minus[index_2_keep]
        s_plus = s_plus[index_2_keep]
        L_alpha = L_alpha[index_2_keep]
        U_alpha = U_alpha[index_2_keep]

        if alpha == alphas[0]:
            # Plot MRL
            axs[0].plot(time_g, mrl_transformed[index_2_keep], color="red", label="MRL (g-space)", linewidth=linewidth)
            axs[1].plot(time_t, mrl_physical[index_2_keep], color="red", label="MRL (t-space)", linewidth=linewidth)

        label_fill_g = rf"$s^{{\pm}}$ (α={alpha})"
        label_fill_t = rf"$L,U$ (α={alpha})"

        # g(t) domain: fill + edge
        axs[0].fill_between(
            time_g,
            s_minus,
            s_plus,
            color=color,
            alpha=0.05,
            label=label_fill_g,
        )
        axs[0].plot(time_g, s_plus, color='black', linewidth=0.6)
        axs[0].plot(time_g, s_minus, color='black', linewidth=0.6)

        # t domain: fill + edge
        axs[1].fill_between(
            time_t,
            L_alpha,
            U_alpha,
            color=color,
            alpha=0.05,
            label=label_fill_t,
        )
        axs[1].plot(time_t, U_alpha, color='black', linewidth=0.6)
        axs[1].plot(time_t, L_alpha, color='black', linewidth=0.6)

    # Formatting
    axs[0].set_xlabel('Transformed time $g(t)$', fontsize=fontsize)
    axs[0].set_ylabel('Remaining Useful Life', fontsize=fontsize)
    axs[0].set_title('RUL vs $g(t)$', fontsize=fontsize)
    axs[0].grid(True)
    axs[0].legend(fontsize=fontsize - 2, loc='upper right')

    axs[1].set_xlabel('Original time $t$', fontsize=fontsize)
    axs[1].set_title('RUL vs $t$', fontsize=fontsize)
    axs[1].grid(True)
    axs[1].legend(fontsize=fontsize - 2, loc='upper right')

    plt.tight_layout()
    plt.show()


def plot_mixture_example(ttf_data1, ttf_data2, ttf_data3,
                         ttf_data, x_vals, cdf,
                         mrl_physical, g_vals, mrl_transformed,
                         inflection_x, inflection_g,
                         pdf, idx_inflect_points, s_plus, s_minus,
                         L_alpha, U_alpha):


    # Create 3x2 subplot
    fig, ax = plt.subplots(3, 2, figsize=(14, 10))

    # Top left: PDF (via histogram)
    ax[0, 0].hist(ttf_data1, color='r', alpha=0.4, bins=100, label='Weibull')
    ax[0, 0].hist(ttf_data2, color='b', alpha=0.4, bins=100, label='Beta')
    ax[0, 0].hist(ttf_data3, color='g', alpha=0.4, bins=100, label='Normal')
    ax[0, 0].hist(ttf_data,  color='k', alpha=0.1, bins=100, label='Mixture')
    ax[0, 0].set_title("TTF Histograms (PDF approx.)")
    ax[0, 0].set_xlabel("Time")
    ax[0, 0].set_ylabel("Count")
    ax[0, 0].legend()
    ax[0, 0].grid(True)

    # Top right: CDF
    ax[0, 1].plot(x_vals, cdf, label='CDF (KDE)', color='purple')
    # ax[0, 1].plot(g_vals, cdf, label='CDF (KDE)', color='red')
    ax[0, 1].set_title("CDF of Mixture Data")
    ax[0, 1].set_xlabel("Time")
    ax[0, 1].set_ylabel("CDF")
    ax[0, 1].grid(True)
    ax[0, 1].legend()

    # Bottom left: MRL in physical time
    ax[1, 0].plot(x_vals, mrl_physical, label="MRL (physical time)", color="blue")
    ax[1, 0].plot(x_vals, U_alpha, label="Upper", color="red")
    ax[1, 0].plot(x_vals, L_alpha, label="Lower", color="red")
    ax[1, 0].set_xlabel("t")
    ax[1, 0].set_ylabel("MRL(t)")
    ax[1, 0].set_title("MRL in Physical Time")
    ax[1, 0].grid(True)
    ax[1, 0].legend()

    # Bottom right: MRL in transformed time
    ax[1, 1].plot(g_vals, mrl_transformed, label="MRL (transformed time)", color="green")
    ax[1, 1].plot(g_vals, s_minus, label="Lower", color="red")
    ax[1, 1].plot(g_vals, s_plus, label="Upper", color="red")
    ax[1, 1].set_xlabel("g(t)")
    ax[1, 1].set_ylabel("MRL(g)")
    ax[1, 1].set_title("MRL in Transformed Time")
    ax[1, 1].grid(True)
    ax[1, 1].legend()


    # Left: g(t) with inflections
    ax[2, 0].plot(x_vals, g_vals, label=r'$g(t)$', color='black')
    ax[2, 0].scatter(inflection_x, inflection_g, color='red', zorder=5, label='Inflection Points')
    ax[2, 0].set_xlabel("Time t")
    ax[2, 0].set_ylabel(r"$g(t)$")
    ax[2, 0].set_title("Inflection Points in $g(t)$")
    ax[2, 0].grid(True)
    ax[2, 0].legend()

    # Right: PDF with inflection markers
    ax[2, 1].plot(x_vals, pdf, color='navy', label="KDE PDF")
    ax[2, 1].scatter(inflection_x, pdf[idx_inflect_points], color='red', zorder=5, label='Inflection Times')
    ax[2, 1].set_xlabel("g(t)")
    ax[2, 1].set_ylabel("PDF")
    ax[2, 1].set_title("Inflection Points Mapped on KDE PDF")
    ax[2, 1].grid(True)
    ax[2, 1].legend()

    plt.tight_layout()
    plt.show()
