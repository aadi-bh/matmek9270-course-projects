import numpy as np
from collections.abc import Callable


def mesh_function(f: Callable[[float], float], t: float) -> np.ndarray:
    return f(t)


def func(t: float) -> float:
    # What function goes 1 -> 1, 2 -> 2, 3 -> 3, but 4 -> 12?
    s = np.where(t < 4, t, 12)
    return np.exp(-s)


def test_mesh_function():
    t = np.array([1, 2, 3, 4])
    f = np.array([np.exp(-1), np.exp(-2), np.exp(-3), np.exp(-12)])
    fun = mesh_function(func, t)
    assert np.allclose(fun, f)

if __name__ == "__main__":
    test_mesh_function()
