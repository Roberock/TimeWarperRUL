import numpy as np
import pytest
from rul_timewarping.timewarping import TimeWarping

def test_initialization():
    ttf = np.array([10, 20, 30, 40, 50])
    tw = TimeWarping(ttf)
    assert tw.mu > 0
    assert tw.k > 0

def test_estimate_inflection_points():
    ttf = np.linspace(1, 100, 100)
    tw = TimeWarping(ttf)
    inflection_x, inflection_g = tw.estimate_inflection_points()
    assert isinstance(inflection_x, np.ndarray)
    assert isinstance(inflection_g, np.ndarray)

def test_compute_rul_interval():
    ttf = np.linspace(1, 100, 100)
    tw = TimeWarping(ttf)
    t = np.array([10, 20, 30])
    s_plus, s_minus = tw.compute_rul_interval(t)
    assert s_plus.shape == t.shape
    assert s_minus.shape == t.shape

def test_compute_rul_interval_original_time():
    ttf = np.linspace(1, 100, 100)
    tw = TimeWarping(ttf)
    L_alpha, U_alpha = tw.compute_rul_interval_original_time()
    assert isinstance(L_alpha, np.ndarray)
    assert isinstance(U_alpha, np.ndarray)
    assert L_alpha.shape == U_alpha.shape



def test_single_value():
    ttf = np.array([42])
    tw = TimeWarping(ttf)
    assert tw.mu == 42
    assert tw.k == 0  # std = 0, so k = 0


def test_compute_rul_interval_alpha_edges():
    ttf = np.linspace(1, 100, 100)
    tw = TimeWarping(ttf)
    t = np.array([10, 20, 30])

    s_plus_0, s_minus_0 = tw.compute_rul_interval(t, alpha=0)
    s_plus_1, s_minus_1 = tw.compute_rul_interval(t, alpha=1)

    # alpha=0 might yield max intervals, alpha=1 minimal
    assert np.all(s_plus_0 >= s_minus_0)
    assert np.all(s_plus_1 >= 0)
    assert np.all(s_minus_1 >= 0)