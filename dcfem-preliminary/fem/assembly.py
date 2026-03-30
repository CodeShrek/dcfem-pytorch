import torch

def assemble_stiffness(mesh, D):
    n = mesh.n_nodes
    A = torch.zeros((n, n), dtype=torch.float32)

    lengths = mesh.get_element_lengths()

    for e, (i, j) in enumerate(mesh.elements):
        h = lengths[e]

        k_local = D / h * torch.tensor([[1, -1], [-1, 1]])

        A[i, i] += k_local[0, 0]
        A[i, j] += k_local[0, 1]
        A[j, i] += k_local[1, 0]
        A[j, j] += k_local[1, 1]

    return A