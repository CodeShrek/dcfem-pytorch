import torch
from dc_fem.fem.mesh import FEMMesh1D, FEMMesh2D
from dc_fem.utils.graph_utils import to_deepchem_graph

def verify_1d():
    print("--- Verifying FEMMesh1D ---")
    mesh = FEMMesh1D(n_nodes=10)
    lengths = mesh.get_element_lengths()
    
    # Test Differentiability: Sum of lengths should be 1.0 (domain size)
    total_length = lengths.sum()
    total_length.backward()
    
    print(f"Total Length: {total_length.item()}")
    print(f"Gradient of coords: {mesh.coords.grad is not None}")
    
    # Verify DeepChem Integration
    graph_data = to_deepchem_graph(mesh)
    print(f"Graph Node Features Shape: {graph_data['node_features'].shape}\n")

def verify_2d():
    print("--- Verifying FEMMesh2D ---")
    # Simple triangle: (0,0), (1,0), (0,1)
    nodes = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    elements = torch.tensor([[0, 1, 2]])
    
    mesh = FEMMesh2D(nodes, elements)
    areas = mesh.get_element_areas()
    
    # Area should be 0.5
    print(f"Element Area: {areas.item()}")
    
    areas.sum().backward()
    print(f"Gradient of 2D coords: {mesh.coords.grad is not None}\n")

if __name__ == "__main__":
    verify_1d()
    verify_2d()