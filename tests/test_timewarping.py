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


def test_negative_input():
    ttf = np.array([-10, -5, 0, 5, 10])
    tw = TimeWarping(ttf)
    # Only positive values should be used internally
    assert np.all(tw.ttf_data > 0)


def test_single_value():
    ttf = np.array([42])
    tw = TimeWarping(ttf)
    assert tw.mu == 42
    assert tw.k == 0  # std = 0, so k = 0


def test_nan_input():
    ttf = np.array([10, 20, np.nan, 40, 50])
    with pytest.raises(Exception):
        TimeWarping(ttf)

def test_empty_input():
    ttf = np.array([])
    with pytest.raises(Exception):
        TimeWarping(ttf)