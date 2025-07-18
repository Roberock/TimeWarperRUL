from rul_timewarping.config.config import *
import numpy as np
import matplotlib.pyplot as plt
from rul_timewarping.timewarping import TimeWarping

def run_univariate_example():

    ttf_data = np.random.weibull(a=2.5, size=200) * 1000

    # Initialize
    tw = TimeWarping(ttf_data)

    # Access g(t) values
    g = tw.g_vals
    t = tw.t_grid


    # Get inflection points
    inflection_points_t, inflection_points_g = tw.estimate_inflection_points()

    print('inflection_points_t = ', inflection_points_t)
    print('inflection_points_g = ', inflection_points_g)

    plt.plot(t, g)
    plt.scatter(inflection_points_t, inflection_points_g, color='r')
    plt.grid(True)
    plt.show()


    # Compute RUL intervals
    alpha = 0.05
    s_plus, s_minus = tw.compute_rul_interval(tw.g_vals, alpha=alpha)
    idx_valid = s_plus > s_minus
    L_alpha, U_alpha = tw.compute_rul_interval_original_time(alpha=alpha)
    # Create two subplots: one vs g(t), one vs time t
    fig, axs = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    # Plot g(t) vs [s- s+]
    axs[0].plot(tw.g_vals[idx_valid], s_plus[idx_valid], 'g--', label='s⁺(g)', linewidth=CONF_PLOT['linewidth'])
    axs[0].plot(tw.g_vals[idx_valid], s_minus[idx_valid], 'b--', label='s⁻(g)', linewidth=CONF_PLOT['linewidth'])
    axs[0].set_ylabel('RUL bounds')
    axs[0].set_title('RUL vs g(t)')
    axs[0].grid(True)
    axs[0].legend()

    # Plot  time t vs [s- s+]
    axs[1].plot(tw.t_grid[idx_valid], U_alpha[idx_valid], 'g--', label='U(t)', linewidth=CONF_PLOT['linewidth'])
    axs[1].plot(tw.t_grid[idx_valid], L_alpha[idx_valid], 'b--', label='L(t)', linewidth=CONF_PLOT['linewidth'])
    axs[1].set_xlabel('Time t')
    axs[1].set_title('RUL vs time t')
    axs[1].grid(True)
    axs[1].legend()
    plt.tight_layout()
    plt.show()

    # print some of the results
    for i in range(10):
        print(f'RUL(g(t) = {tw.g_vals[i]:.2f}) = [{s_minus[i]:.2f}, {s_plus[i]:.2f}]')


# Press the green button in the gutter to run the script.
if __name__ == '__main__':


    run_univariate_example()