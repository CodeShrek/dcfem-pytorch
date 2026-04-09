import torch

class FEMMesh:
    """Base Differentiable Mesh class mapping to DeepChem GraphData."""
    def __init__(self, coords: torch.Tensor, elements: torch.Tensor):
        self.coords = coords.clone().detach().requires_grad_(True)
        self.elements = elements.to(torch.long)
        self.n_nodes = coords.shape[0]

class FEMMesh1D(FEMMesh):
    def __init__(self, n_nodes: int, domain: tuple = (0.0, 1.0)):
        coords = torch.linspace(domain[0], domain[1], n_nodes).unsqueeze(-1)
        elements = torch.stack([torch.arange(0, n_nodes - 1), torch.arange(1, n_nodes)], dim=1)
        super().__init__(coords, elements)

    def get_element_lengths(self):
        diff = self.coords[self.elements[:, 1]] - self.coords[self.elements[:, 0]]
        return diff.abs().reshape(-1)

class FEMMesh2D(FEMMesh):
    """2D Mesh with automated grid generation for unit squares."""
    @classmethod
    def generate_unit_square(cls, nx, ny):
        x = torch.linspace(0, 1, nx)
        y = torch.linspace(0, 1, ny)
        grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
        coords = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)

        # Vectorized element generation for nx*ny grid
        i, j = torch.meshgrid(torch.arange(nx - 1), torch.arange(ny - 1), indexing='ij')
        n0 = i * ny + j
        n1 = n0 + 1
        n2 = (i + 1) * ny + j
        n3 = n2 + 1
        
        # Split each cell into two triangles
        t1 = torch.stack([n0.flatten(), n1.flatten(), n2.flatten()], dim=1)
        t2 = torch.stack([n1.flatten(), n3.flatten(), n2.flatten()], dim=1)
        elements = torch.cat([t1, t2], dim=0)
        
        return cls(coords, elements)

    def get_element_areas(self):
        p1, p2, p3 = self.coords[self.elements[:, 0]], self.coords[self.elements[:, 1]], self.coords[self.elements[:, 2]]
        # 0.5 * |x1(y2-y3) + x2(y3-y1) + x3(y1-y2)|
        areas = 0.5 * torch.abs(p1[:,0]*(p2[:,1]-p3[:,1]) + p2[:,0]*(p3[:,1]-p1[:,1]) + p3[:,0]*(p1[:,1]-p2[:,1]))
        return areas.reshape(-1)