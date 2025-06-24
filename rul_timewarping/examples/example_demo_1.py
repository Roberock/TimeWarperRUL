
import numpy as np
import matplotlib.pyplot as plt
from rul_timewarping.timewarping import TimeWarping

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

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