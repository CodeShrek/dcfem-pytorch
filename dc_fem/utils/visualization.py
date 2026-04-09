import torch
import matplotlib.pyplot as plt
import time
from dc_fem.fem.mesh import FEMMesh2D

def plot_mesh(mesh, title="dcFEM Mesh"):
    """Visualizes the 1D or 2D mesh structure."""
    plt.figure(figsize=(6, 5))
    if mesh.coords.shape[1] == 1: # 1D
        x = mesh.coords.detach().numpy().flatten()
        plt.scatter(x, torch.zeros_like(torch.tensor(x)).numpy(), color='blue', label='Nodes')
        plt.title(f"{title} (1D)")
    else: # 2D
        pts = mesh.coords.detach().numpy()
        tri = mesh.elements.numpy()
        plt.triplot(pts[:,0], pts[:,1], tri, color='teal', lw=0.5)
        plt.scatter(pts[:,0], pts[:,1], s=10, color='red')
        plt.title(f"{title} (2D Triangulation)")
    plt.xlabel("X"); plt.ylabel("Y"); plt.grid(True); plt.show()

def benchmark_mesh_efficiency(max_n=100):
    """Visualizes efficiency by plotting grid resolution vs generation time."""
    ns = range(10, max_n, 20)
    times = []
    print("Benchmarking...")
    for n in ns:
        start = time.time()
        _ = FEMMesh2D.generate_unit_square(n, n)
        times.append(time.time() - start)
    
    plt.figure(figsize=(6, 4))
    plt.plot(ns, times, marker='o', linestyle='--', color='darkorange')
    plt.title("Mesh Generation Efficiency (Scalability)")
    plt.xlabel("Grid Resolution (nx=ny)"); plt.ylabel("Time (seconds)")
    plt.grid(True); plt.show()