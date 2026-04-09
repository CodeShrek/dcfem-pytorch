import torch
from dc_fem.fem.mesh import FEMMesh1D
from dc_fem.fem.assembly import assemble_stiffness_1d_sparse

def run_assembly_test():
    print("--- Verifying Sparse Assembly ---")
    
    # 1. Initialize Mesh
    mesh = FEMMesh1D(n_nodes=5)
    
    # 2. Define learnable Diffusion Coefficient D
    D = torch.tensor(2.0, requires_grad=True)
    
    # 3. Assemble Sparse Stiffness Matrix K
    K = assemble_stiffness_1d_sparse(mesh, D)
    
    print(f"Matrix Type: {K.layout}")
    print(f"Assembled Sparse Matrix:\n{K.to_dense()}")
    
    # 4. Verify Differentiability (Section 3.2 of Proposal)
    # We check if gradients flow from the matrix back to D
    loss = K.values().sum() 
    loss.backward()
    
    print(f"\nGradient of D: {D.grad}")
    if D.grad is not None:
        print("SUCCESS: Assembly is fully differentiable.")

if __name__ == "__main__":
    run_assembly_test()