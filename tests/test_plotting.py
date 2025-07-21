import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from rul_timewarping.plotting import plot_g_with_inflection, plot_envelope_bounds

def test_plot_g_with_inflection_runs():
    t_grid = np.linspace(0, 10, 100)
    g_vals = np.sin(t_grid)
    plot_g_with_inflection(t_grid, g_vals, t_star=5)

def test_plot_envelope_bounds_runs():
    from rul_timewarping.timewarping import TimeWarping
    ttf = np.linspace(1, 100, 100)
    tw = TimeWarping(ttf)
    plot_envelope_bounds(tw)