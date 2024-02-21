import torch
import math

from torch.nn import Module, Sequential as Seq, Linear, ReLU, Dropout, Sigmoid, Parameter, init
from torch.nn.functional import relu
from torch_geometric.utils import degree, softmax

from itertools import combinations


class HyperedgeModel(Module):
    r"""Hyperedge Model.

    Args:
        n_node_feats (int): number of node features of input graph.
        n_node_feats_out (int): number of node features of the output graph.
        message_feats (int, optional): number of intermediate features. (default :obj:`int`=32)
        dropout (float, optional): probability of an element to be zeroed. (default :obj:`float`=0.01)

    :rtype: :class:`Tuple[Tensor,Tensor]`
    """
    def __init__(self, node_in_channels, node_out_channels, global_in_channels, message_feats: int=32, dropout=0.01):
        super().__init__()
        self.node_in_channels = node_in_channels
        self.message_feats    = message_feats
        self.mlp_x  = Seq(Linear(node_in_channels+global_in_channels, message_feats),
                          ReLU(),
                          Dropout(p=dropout),
                          Linear(message_feats, message_feats),
                          ReLU(),
                          Dropout(p=dropout),
                          Linear(message_feats, message_feats))
        self.weight = Parameter(torch.empty((message_feats, message_feats)))
        self.x_hat  = Seq(Linear(message_feats*2, message_feats),
                          ReLU(),
                          Dropout(p=dropout),
                          Linear(message_feats, message_feats),
                          ReLU(),
                          Dropout(p=dropout),
                          Linear(message_feats, node_out_channels),
                          Sigmoid())
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.mlp_x.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        for layer in self.x_hat.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def __hyperedge_finding__(self, x, batch, r):
        device = x.device

        d = degree(batch, dtype=torch.long).detach().cpu().tolist()

        ranges = []
        index_start = 0
        for i in range(0,len(d)):
            ranges.append((index_start, index_start+d[i]))
            index_start += d[i]

        hyperedges = torch.tensor(
            [comb for index in ranges for comb in list(combinations(range(index[0],index[1]),r))],
            dtype=torch.long,
            device=device
        )

        d_new = degree(batch, dtype=torch.long).detach().cpu().apply_(lambda x: math.comb(x,r)).tolist()
        batch_hyper = torch.tensor(
            [x for (a, b) in zip(d_new,range(len(d_new))) for x in a*[b]],
            dtype=torch.long,
            device=device
        )

        x_hyper = torch.stack([torch.index_select(x, 0, hyperedges.select(-1,j)) for j in range(0,r)],dim=2)

        return x_hyper.sum(2), batch_hyper
    
    def weighting(self, x_hyper, batch_hyper):
        coefficient = softmax(x_hyper, index=batch_hyper)
        return coefficient * relu(torch.mm(x_hyper, self.weight), inplace=True)

    def forward(self, x, u, batch, r):
        x_hyper = self.mlp_x(torch.cat([x, u[batch]], dim=1).float())
        x_hyper, batch_hyper = self.__hyperedge_finding__(x_hyper, batch, r)
        x_hyper_hat = self.weighting(x_hyper, batch_hyper)
        out = torch.cat([x_hyper, x_hyper_hat], dim=1).float()
        return self.x_hat(out), batch_hyper