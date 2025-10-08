import numpy as np
import sympy as sp
import scipy

x,y = sympy.symbols('x y')

def test_five(evaluation_fn=):
    fns = [(abs(x), (-1,1)),
           (sp.exp(sp.sin(x)), (0,2)),
           (x**10, (0,1)),
           (sp.exp(-(x-0.5)**2) - sp.exp(-0.25), (0,100))]
    [evaluation_fn(fn) for fn in fns]

def inner(u, v, domain=(-1, 1), x=x):
    return sp.integrate(u * v, (x, domain[0], domain[-1]))

if __name__ == "__main__":

