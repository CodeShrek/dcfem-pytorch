import torch

def get_shape_grad_1d(mesh):
    """Gradients of 1D linear shape functions: [-1/h, 1/h]."""
    h = mesh.get_element_lengths()
    inv_h = 1.0 / h
    # Shape: (n_elements, 2_nodes, 1_dim)
    grad_N1 = -inv_h.unsqueeze(-1)
    grad_N2 = inv_h.unsqueeze(-1)
    return torch.stack([grad_N1, grad_N2], dim=1)

def get_shape_grad_2d(mesh):
    """Vectorized gradients for 2D P1 triangles."""
    p1, p2, p3 = mesh.coords[mesh.elements[:, 0]], mesh.coords[mesh.elements[:, 1]], mesh.coords[mesh.elements[:, 2]]
    
    # Jacobian terms
    a, b = p2[:, 0] - p1[:, 0], p3[:, 0] - p1[:, 0]
    c, d = p2[:, 1] - p1[:, 1], p3[:, 1] - p1[:, 1]
    inv_det = 1.0 / (a * d - b * c)

    # Global gradients dNi/dx, dNi/dy
    grad_N1 = torch.stack([(c - d), (b - a)], dim=1) * inv_det.unsqueeze(-1)
    grad_N2 = torch.stack([d, -b], dim=1) * inv_det.unsqueeze(-1)
    grad_N3 = torch.stack([-c, a], dim=1) * inv_det.unsqueeze(-1)
    return torch.stack([grad_N1, grad_N2, grad_N3], dim=1)