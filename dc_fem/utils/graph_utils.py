def to_deepchem_graph(mesh):
    """
    Utility to interface with DeepChem's GraphData.
    - node_features: Stores coordinates[cite: 156].
    - edge_index: Defines sparsity patterns[cite: 156].
    """
    return {
        "node_features": mesh.coords,
        "edge_index": mesh.elements.t()
    }