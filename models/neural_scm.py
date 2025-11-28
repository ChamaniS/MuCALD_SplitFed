# models/neural_scm.py
import torch
import torch.nn as nn
from typing import List, Dict, Tuple

def topological_sort(parents: List[List[int]]):
    N = len(parents)
    indeg = [0]*N
    for i in range(N):
        for p in parents[i]:
            indeg[i] += 1
    order = []
    used = [False]*N
    for _ in range(N):
        found = False
        for i in range(N):
            if not used[i] and indeg[i] == 0:
                order.append(i)
                used[i] = True
                for j in range(N):
                    if i in parents[j]:
                        indeg[j] -= 1
                found = True
                break
        if not found:
            raise ValueError("Graph is not a DAG (cycle detected).")
    return order

class NodeMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 128), nn.ReLU(inplace=True),
            nn.Linear(128, out_dim)
        )
    def forward(self, x):
        return self.net(x)

class NeuralSCM(nn.Module):
    """
    z_i = f_i( [z_parents], u_i )
    parents: list of lists; parents[i] are indices of parents of node i
    node_dims: dimensionality for each z_i (usually small, e.g., 4~8)
    """
    def __init__(self, parents: List[List[int]], node_dims: List[int]):
        super().__init__()
        self.parents = parents
        self.node_dims = node_dims
        self.N = len(node_dims)
        self.order = topological_sort(parents)

        self.mlps = nn.ModuleList()
        for i in range(self.N):
            in_dim = node_dims[i]  # u_i
            for p in parents[i]:
                in_dim += node_dims[p]
            self.mlps.append(NodeMLP(in_dim, node_dims[i]))

    def forward(self, u_list: List[torch.Tensor]):
        # u_list: list of (B, d_i)
        z = [None]*self.N
        for i in self.order:
            parts = [u_list[i]]
            for p in self.parents[i]:
                parts.append(z[p])
            x = torch.cat(parts, dim=1)
            z[i] = self.mlps[i](x)
        return z  # list of z_i tensors

    def as_vector(self, z_list: List[torch.Tensor]) -> torch.Tensor:
        return torch.cat(z_list, dim=1)

