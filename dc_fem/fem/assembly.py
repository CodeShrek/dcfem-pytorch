import torch

def assemble_stiffness_1d_sparse(mesh, D):
    """
    Assembles a sparse stiffness matrix for 1D Poisson.
    Uses torch.sparse_coo_tensor for memory efficiency and Autograd compatibility.
    """
    n = mesh.n_nodes
    elements = mesh.elements
    
    # get_element_lengths() now returns a flat 1D tensor
    inv_h = 1.0 / mesh.get_element_lengths()
    
    # Indices for the sparse matrix: [2, 4 * n_elements]
    rows = torch.cat([elements[:, 0], elements[:, 0], elements[:, 1], elements[:, 1]])
    cols = torch.cat([elements[:, 0], elements[:, 1], elements[:, 0], elements[:, 1]])
    indices = torch.stack([rows, cols])
    
    # Values for the entries: (D/h) * [[1, -1], [-1, 1]]
    # Flattening here prevents the 'len(size) == sparse_dim + dense_dim' error
    vals = torch.cat([
        D * inv_h,    # (i,i)
        -D * inv_h,   # (i,j)
        -D * inv_h,   # (j,i)
        D * inv_h     # (j,j)
    ]).reshape(-1) 
    
    # Assemble into a sparse COO tensor and coalesce to sum duplicate entries 
    K_sparse = torch.sparse_coo_tensor(indices, vals, (n, n)).coalesce()
    return K_sparse