import torch
import math

from torch.nn import Module, Sequential as Seq, Linear, ReLU, Dropout, Sigmoid, Parameter, init
from torch.nn.functional import relu

from HyPER.utils import softmax


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

    def __hyperedge_finding__(self, x, hyperedge_index, r):
        hyperedge_index = hyperedge_index.permute(dims=(1,0))
        x_hyper = torch.gather(
            x.unsqueeze(1).expand([-1,r,-1]),
            0,
            hyperedge_index.unsqueeze(2).expand([-1,-1,self.message_feats])
        ).transpose(1,2)
        return x_hyper.sum(2)

    def weighting(self, x_hyper, batch_hyper):
        coefficient = softmax(x_hyper, index=batch_hyper, dim_size=x_hyper.size(0))
        return coefficient * relu(torch.mm(x_hyper, self.weight), inplace=True)

    def forward(self, x, u, batch, hyperedge_index, batch_hyper, r):
        x_hyper = self.mlp_x(torch.cat([x, u[batch]], dim=1).float())
        x_hyper = self.__hyperedge_finding__(x_hyper, hyperedge_index, r)
        x_hyper_hat = self.weighting(x_hyper, batch_hyper)
        out = torch.cat([x_hyper, x_hyper_hat], dim=1).float()
        return self.x_hat(out), batch_hyper