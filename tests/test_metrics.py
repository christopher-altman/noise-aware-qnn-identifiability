import numpy as np
from src.metrics import accuracy, l2_param_error, identifiability_proxy

def test_accuracy():
    y = np.array([0,1,1,0])
    yhat = np.array([0,1,0,0])
    assert abs(accuracy(y, yhat) - 0.75) < 1e-12

def test_param_error():
    a = np.array([1.0, 2.0])
    b = np.array([1.0, 1.0])
    assert abs(l2_param_error(a, b) - 1.0) < 1e-12

def test_ident_proxy_bounds():
    h = np.array([1.0, 2.0, 4.0])
    v = identifiability_proxy(h)
    assert 0.0 <= v <= 1.0
