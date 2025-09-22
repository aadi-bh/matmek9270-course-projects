import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import scipy.sparse as sparse
from sympy.utilities.lambdify import implemented_function
from poisson import Poisson

x, y = sp.symbols('x,y')

class Poisson2D:
    r"""Solve Poisson's equation in 2D::

        \nabla^2 u(x, y) = f(x, y), x, y in [0, Lx] x [0, Ly]

    with homogeneous Dirichlet boundary conditions.
    """

    def __init__(self, Lx, Ly, Nx, Ny):
        self.px = Poisson(Lx, Nx) # we can reuse some of the code from the 1D case
        self.py = Poisson(Ly, Ny)

    def create_mesh(self):
        """Return a 2D Cartesian mesh
        """
        self.x = self.px.create_mesh(self.px.N)
        self.y = self.py.create_mesh(self.py.N)
        self.sxx, self.syy = np.meshgrid(self.x, self.y, sparse=False)
        return self.sxx, self.syy
        raise NotImplementedError

    def laplace(self):
        """Return a vectorized Laplace operator"""
        Dx = self.px.D2()
        Dy = self.py.D2()
        D2X = sparse.kron(Dx, sparse.eye(self.px.N+1))
        D2Y = sparse.kron(sparse.eye(self.py.N+1), Dy)
        return D2X + D2Y
        raise NotImplementedError

    def assemble(self, bc=None, f=implemented_function('f', lambda x, y: 4)(x,y)):
        """Return assembled coefficient matrix A and right hand side vector b"""
        self.create_mesh()
        A = self.laplace()
        F = np.empty((self.px.N+1, self.py.N+1))
        F[:,:] = sp.lambdify((x, y), f)(self.sxx, self.syy)

        # Dirichlet boundary
        B = np.ones((self.px.N+1, self.py.N+1), dtype=bool)
        B[1:-1, 1:-1] = 0
        bnds = np.where(B.ravel() == 1)[0]
        A = A.tolil()
        for i in bnds:
            A[i] = 0.0
            A[i, i] = 1.0
        A = A.tocsr()
        b = F.ravel()
        b[bnds] = 0.0
        return A, b
        raise NotImplementedError

    def l2_error(self, u, ue):
        """Return l2-error

        Parameters
        ----------
        u : array
            The numerical solution (mesh function)
        ue : Sympy function
            The analytical solution
        """
        print(self.sxx, self.syy)
        uj = sp.lambdify((x, y), ue)(self.sxx, self.syy)
        print(uj, '\n', u)
        return np.sqrt(self.px.dx * self.py.dx * np.sum((uj - u)**2))
        raise NotImplementedError

    def __call__(self, f=implemented_function('f', lambda x, y: 2)(x, y)):
        """Solve Poisson's equation with a given right hand side function

        Parameters
        ----------
        f : Sympy function
            The right hand side function f(x, y)

        Returns
        -------
        The solution as a Numpy array

        """
        A, b = self.assemble(f=f)
        return sparse.linalg.spsolve(A, b.ravel()).reshape((self.px.N+1, self.py.N+1))

def test_poisson2d():
    Lx = 2
    Ly = 2
    sol = Poisson2D(Lx, Ly, 5, 5)
    ue = x*(x-Lx) *  y*(y-Ly)
    ue = x*(x-Lx) *  y*(y-Ly)
    B = np.ones((sol.px.N+1, sol.py.N+1)) * np.nan
    u = sol(f=ue.diff(x,2) + ue.diff(y,2))
    print(f'L2-error {sol.l2_error(u, ue)}')



if __name__ == "__main__":
    test_poisson2d()
