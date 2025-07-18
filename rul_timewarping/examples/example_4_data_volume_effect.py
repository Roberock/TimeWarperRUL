from rul_timewarping.config.config import *
import numpy as np
import matplotlib.pyplot as plt
from rul_timewarping.timewarping import TimeWarping
from mpmath import *
from scipy.stats import gamma, gaussian_kde
from scipy.spatial import ConvexHull
from matplotlib.cm import ScalarMappable


# Bootstrap function to estimate uncertainty in bounds
def bootstrap_s_bounds(ttf_data, N=100, alpha=0.05, alpha_bstp=0.05):
    s_plus_samples, s_minus_samples, k_samples = [], [], []

    for _ in range(N):
        subset_samples = np.random.choice(ttf_data, size=len(ttf_data), replace=True)
        tw = TimeWarping(subset_samples)
        k_samples.append(tw.k)

        if not (0 < tw.k < 1):
            print('k is not in (0,1)...skipping this sample.')
            continue  # avoid unstable k

        s_minus, s_plus = tw.compute_rul_interval(tw.g_vals, alpha=alpha)
        s_minus_samples.append(s_minus)
        s_plus_samples.append(s_plus)

    Q_Levels = [(alpha_bstp / 2) * 100, (1 - alpha_bstp / 2) * 100]
    s_lower_bounds = np.percentile(s_minus_samples, Q_Levels)
    s_upper_bounds = np.percentile(s_plus_samples, Q_Levels)
    k_bounds = np.percentile(k_samples, Q_Levels)

    return s_lower_bounds, s_upper_bounds, k_bounds


def run_data_volume_example():
    # Generate synthetic TTF data (Weibull-distributed)
    ttf_data = np.random.weibull(a=2.5, size=200) * 1000

    # Initialize TimeWarping model
    tw = TimeWarping(ttf_data)

    # Plot original g_vals and time-warped cdf
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(tw.g_vals, tw.t_grid, label='Time-Warped', color='blue')
    plt.title('Time-Warping Mapping')
    plt.xlabel('g')
    plt.ylabel('t')
    plt.grid(True)
    plt.legend()

    # Compute and display bootstrap confidence bounds
    s_lb, s_ub, k_bounds = bootstrap_s_bounds(ttf_data, N=200, alpha=0.05, alpha_bstp=0.05)
    print(f"s_minus bounds: {s_lb}")
    print(f"s_plus bounds: {s_ub}")
    print(f"k bounds: {k_bounds}")

    # Plot bootstrap intervals
    plt.subplot(1, 2, 2)
    s_minus, s_plus = tw.compute_rul_interval(tw.g_vals, alpha=0.05)

    plt.errorbar([0], [s_minus], yerr=[[s_minus - s_lb[0]], [s_lb[1] - s_minus]], fmt='o', label='s_minus')
    plt.errorbar([1], [s_plus], yerr=[[s_plus - s_ub[0]], [s_ub[1] - s_plus]], fmt='o', label='s_plus')

    plt.title('Bootstrap Interval Estimates')
    plt.xticks([0, 1], ['s_minus', 's_plus'])
    plt.ylabel('Value')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    run_data_volume_example()
