import torch

from torch import Tensor


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