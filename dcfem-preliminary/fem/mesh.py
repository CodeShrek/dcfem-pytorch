import torch

class FEMMesh1D:
    def __init__(self, n_nodes):
        self.n_nodes = n_nodes
        self.x = torch.linspace(0, 1, n_nodes)

        # elements: pairs of nodes
        self.elements = torch.stack([
            torch.arange(0, n_nodes-1),
            torch.arange(1, n_nodes)
        ], dim=1)

    def get_element_lengths(self):
        return self.x[self.elements[:,1]] - self.x[self.elements[:,0]]