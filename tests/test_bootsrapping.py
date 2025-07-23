import numpy as np
from rul_timewarping.bootsrapping import run_bootstrap_inflection_points as bt


def test_run_bootstrap_inflection_points():
    np.random.seed(0)
    ttf_data = np.random.exponential(scale=5.0, size=200)

    B = 10
    result = bt(ttf_data, B=B, alpha=0.1, smooth_sigma=2)

    # Basic key checks
    assert 't_mean' in result and isinstance(result['t_mean'], float)
    assert 't_lower' in result and isinstance(result['t_lower'], float)
    assert 't_upper' in result and isinstance(result['t_upper'], float)
    assert 'g_mean' in result and isinstance(result['g_mean'], float)
    assert 'g_lower' in result and isinstance(result['g_lower'], float)
    assert 'g_upper' in result and isinstance(result['g_upper'], float)

    # Sample arrays
    assert isinstance(result['t_inflection_samples'], np.ndarray)
    assert isinstance(result['g_inflection_samples'], np.ndarray)
    assert len(result['t_inflection_samples']) == B
    assert len(result['g_inflection_samples']) == B

    # Relationship checks
    assert result['t_lower'] <= result['t_mean'] <= result['t_upper']
    assert result['g_lower'] <= result['g_mean'] <= result['g_upper']