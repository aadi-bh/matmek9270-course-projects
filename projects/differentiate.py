import numpy as np


def differentiate(u: np.ndarray, dt: float) -> np.ndarray:
    du = np.empty_like(u)
    for i in range(len(u)-1):
        du[i] = u[i+1] - u[i]
    du[-1] = 0.0
    return du/dt

def differentiate_vector(u: np.ndarray, dt: float) -> np.ndarray:
    du = np.empty_like(u)
    du[:-1] = u[1:] - u[:-1]
    du[-1] = 0.0
    return du/dt

def test_differentiate():
    t = np.linspace(0, 1, 10)
    dt = t[1] - t[0]
    u = t**2
    du1 = differentiate(u, dt)
    du2 = differentiate_vector(u, dt)
    assert np.allclose(du1, du2)

if __name__ == '__main__':
    test_differentiate()
    
