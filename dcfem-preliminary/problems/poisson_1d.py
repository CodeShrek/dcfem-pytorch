import torch
from fem.mesh import FEMMesh1D
from fem.assembly import assemble_stiffness
from solvers.cg import conjugate_gradient

def solve_poisson(n, D):
    mesh = FEMMesh1D(n)

    A = assemble_stiffness(mesh, D)

    f = torch.ones(n)

    # Dirichlet BC
    A[0,:] = 0; A[0,0] = 1
    A[-1,:] = 0; A[-1,-1] = 1
    f[0] = 0; f[-1] = 0

    u = conjugate_gradient(A, f)
    return u