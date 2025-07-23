from rul_timewarping.timewarping import TimeWarping
from rul_timewarping.utils import *


def run_bootstrap_inflection_points(ttf_data: np.ndarray,
                                    B: int=200,
                                    alpha: float=0.1,
                                    smooth_sigma: float=3):
    """
    Bootstrap estimation of the inflection point of g(t).
    Returns: dict with mean, lower, upper bounds, sample list, g_samples, and t_grid.
    """
    t_inflection_samples, g_inflection_samples = [], []

    for i in range(B):
        sample = np.random.choice(ttf_data, size=int(len(ttf_data)*0.95), replace=True)
        TW = TimeWarping(sample)
        t_star_i, g_star_i = TW.estimate_inflection_points(smooth_sigma)

        if t_star_i <= np.max(ttf_data):
            t_inflection_samples.append(t_star_i)
            g_inflection_samples.append(g_star_i)

    sam_t = np.array(t_inflection_samples)
    sam_g = np.array(g_inflection_samples)

    t_lower, t_upper = np.percentile(sam_t, [100*alpha/2, 100*(1-alpha/2)])
    g_lower, g_upper = np.percentile(sam_g, [100*alpha/2, 100*(1-alpha/2)])

    return {
        't_mean': float(sam_t.mean()),
        'g_mean': float(sam_g.mean()),
        'g_lower': g_lower,
        'g_upper': g_upper,
        't_lower': t_lower,
        't_upper': t_upper,
        't_inflection_samples': sam_t,
        'g_inflection_samples': sam_g,
    }

