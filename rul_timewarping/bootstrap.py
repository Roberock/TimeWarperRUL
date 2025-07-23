import numpy as np
from rul_timewarping.timewarping import *


def bootstrap_inflection(ttf_data: np.ndarray, B: int=1000, alpha: float=0.1, smooth_sigma: float=2):
    """
    Bootstrap estimation of the inflection point of g(t).
    Returns: dict with mean, lower, upper bounds, sample list, g_samples, and t_grid.
    """
    t_star, g_samps = [], []
    base_tw = TimeWarping(ttf_data)
    fixed_grid = base_tw.t_grid
    for i in range(B):
        sample = np.random.choice(ttf_data, size=len(ttf_data), replace=True)
        tw = TimeWarping(sample)
        t0, g0 = tw.estimate_inflection_points(smooth_sigma)
        if t0 <= np.max(ttf_data):
            t_star.append(t0)
            if len(g_samps) < max(1, B//10):
                g_samps.append(g0)
    arr = np.array(t_star)
    lower, upper = np.percentile(arr, [100*alpha/2, 100*(1-alpha/2)])
    return {
        'mean': float(arr.mean()),
        'lower': lower,
        'upper': upper,
        'samples': arr,
        'g_samples': np.vstack(g_samps),
        't_grid': fixed_grid
    }

