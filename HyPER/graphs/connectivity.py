import torch
import warnings

from torch import Tensor, concatenate, long
from torch_geometric.utils import degree
from itertools import permutations
from math import factorial


def connect_vertices(x, batch: Tensor=None) -> Tensor:
    r"""Computes edges of a fully connected graph.

    Args:
        x (torch.Tensor): Node feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{N \times F}`.
        batch (torch.Tensor, optional): Batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            node to a specific example. (default: :obj:`None`)

    :rtype: :class:`torch.Tensor`
    """
    if batch is not None:
        if x.device != batch.device:
            warnings.warn("Input tensor 'x' and 'batch' are on different devices "
                      "Performing blocking device transfer")
            batch = batch.to(x.device)
        else:
            pass

        num_nodes = degree(batch, dtype=long).detach().cpu().tolist()

        edge_index = torch.tensor([], dtype=long, device=x.device)
        edge_batch = torch.tensor([], dtype=long, device=x.device)

        ptr = 0; idx = 0
        for i in num_nodes:
            edge_index = concatenate([edge_index, torch.tensor(list(permutations(range(ptr,ptr+i),2)),device=x.device).permute(dims=(1,0))],dim=1)
            edge_batch = concatenate([edge_batch, torch.full((1,int(factorial(i)/factorial(i-2))), idx, device=x.device).squeeze(0)])
            ptr += i
            idx += 1

        return edge_index, edge_batch

    else:
        num_nodes = x.size(dim=0)

        edge_index = torch.tensor(list(permutations(range(num_nodes),2))).permute(dims=(1,0))

        return edge_index


def getUndirectedEdges(edge_index: Tensor, edge_attr: Tensor, reduce: str = 'mean') -> Tensor:
    r"""Contracting double edges between two nodes into one using 
    `softmax` method.

    Args:
        edge_index (Tensor): `edge_index` tensor contains directed edges.
        edge_attr (Tensor): `edge_attr` tensor contains features corresponding to directed edges.
        reduce (optional: str): reduction method used to convert directed edges to undirected edges (default: 'mean').

    :rtype: :class:`Tensor`
    """
    edge_index = edge_index.transpose(1,0)
    attr_out_edge = edge_attr[torch.max(torch.sort(edge_index, dim=1)[1]==torch.tensor([0,1]), dim=1)[0]]
    attr_in_edge  = edge_attr[torch.max(torch.sort(edge_index, dim=1)[1]==torch.tensor([1,0]), dim=1)[0]]
    in_edge_matching_order = torch.sort(edge_index[torch.max(torch.sort(edge_index, dim=1, stable=True)[1]==torch.tensor([1,0]), dim=1)[0]], dim=0, stable=True)[1][:,1]

    directed_edges = torch.concat((attr_out_edge, attr_in_edge.take_along_dim(in_edge_matching_order.reshape(-1,1),dim=0)),dim=1)
    
    if reduce == 'mean':
        return torch.mean(directed_edges, dim=1)
    elif reduce == 'sum':
        return torch.sum(directed_edges, dim=1)
    elif reduce == 'max':
        return torch.max(directed_edges, dim=1)[0]
    elif reduce == 'min':
        return torch.min(directed_edges, dim=1)[0]