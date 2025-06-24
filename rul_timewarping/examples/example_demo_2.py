from importlib.metadata import distribution

import numpy as np
import matplotlib.pyplot as plt
from rul_timewarping.timewarping import TimeWarping


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    np.random.seed(42)
    N1, N2 = 1_000, 20_000
    ttf_data1 = np.random.weibull(a=2.5, size=N1) * 3000
    ttf_data2 = np.random.weibull(a=2.5, size=N2) * 1000

    ttf_data = np.concatenate((ttf_data1, ttf_data2))

    # Initialize
    tw = TimeWarping(ttf_data)

    # Access g(t) values
    g = tw.g_vals
    t = tw.t_grid

    # Get inflection points
    inflection_points_t, inflection_points_g = tw.estimate_inflection_points()

    print('inflection_points_t = ', inflection_points_t)
    print('inflection_points_g = ', inflection_points_g)

    t_grid = tw.t_grid

    # Create subplots
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))

    # Left subplot: Mixture PDF with inflection points
    bins = 50
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