import torch

def conjugate_gradient(A, b, max_iter=100, tol=1e-6):
    """
    Autograd-compatible CG solver. 
    Compatible with both dense and sparse PyTorch tensors.
    """
    x = torch.zeros_like(b)
    r = b - torch.matmul(A, x)
    p = r.clone()
    rs_old = torch.dot(r, r)

    for _ in range(max_iter):
        Ap = torch.matmul(A, p)
        alpha = rs_old / (torch.dot(p, Ap) + 1e-12)
        
        x = x + alpha * p
        r = r - alpha * Ap
        
        rs_new = torch.dot(r, r)
        if torch.sqrt(rs_new) < tol:
            break
            
        p = r + (rs_new / rs_old) * p
        rs_old = rs_new
        
    return x