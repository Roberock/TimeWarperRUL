import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

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




def plot_mixture_example(ttf_data1, ttf_data2, ttf_data3,
                         ttf_data, x_vals, cdf,
                         mrl_physical, g_vals, mrl_transformed,
                         inflection_x, inflection_g,
                         pdf, sign_change, s_plus, s_minus, L_alpha, U_alpha):

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

    idx_valid = s_plus > s_minus
    ax[1, 1].plot(g_vals, mrl_transformed, label="MRL (transformed time)", color="green")
    ax[1, 1].plot(g_vals[idx_valid], s_minus[idx_valid], label="Lower", color="red")
    ax[1, 1].plot(g_vals[idx_valid], s_plus[idx_valid], label="Upper", color="red")
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
    ax[2, 1].scatter(inflection_x, pdf[sign_change], color='red', zorder=5, label='Inflection Times')
    ax[2, 1].set_xlabel("g(t)")
    ax[2, 1].set_ylabel("PDF")
    ax[2, 1].set_title("Inflection Points Mapped on KDE PDF")
    ax[2, 1].grid(True)
    ax[2, 1].legend()

    plt.tight_layout()
    plt.show()
