import torch
from dc_fem.fem.mesh import FEMMesh1D, FEMMesh2D
from dc_fem.utils.visualization import plot_mesh, benchmark_mesh_efficiency

if __name__ == "__main__":
    print("--- Milestone 1: Visualizing Working & Efficiency ---")
    
    # 1. Visualize 1D Mesh "Working"
    # Confirms nodes are correctly spaced across the domain
    mesh1d = FEMMesh1D(n_nodes=15)
    plot_mesh(mesh1d, "1D Discretization")
    
    # 2. Visualize 2D Mesh "Working"
    # Confirms the unit square is correctly triangulated for FEM
    mesh2d = FEMMesh2D.generate_unit_square(nx=10, ny=10)
    plot_mesh(mesh2d, "2D Unit Square")
    
    # 3. Efficiency Benchmark
    # Demonstrates the scalability required for production-ready utility
    benchmark_mesh_efficiency(max_n=150)