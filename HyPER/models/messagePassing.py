import torch

from torch.nn import Module, Sequential as Seq, Linear, ReLU, Dropout
from torch_geometric.utils import scatter

from ..utils import custom_scatter


class EdgeModel(Module):
    r"""A callable which updates a graph's edge features based
    on its source and target node features, its current edge
    features and its global features.

    Args:
        node_in_channels (int): number of node features of input graph.
        edge_in_channels (int): number of edge features of input graph.
        global_in_channels (int): number of global features of input graph.
        edge_out_channels (int): number of edge features after updates.
        message_feats (int, optional): number of intermediate features. (default :obj:`int`=32)
        dropout (float, optional): probability of an element to be zeroed. (default :obj:`float`=0.01)
    """
    def __init__(self, node_in_channels, edge_in_channels, global_in_channels, edge_out_channels, message_feats: int=32, dropout: float=0.01):
        super(EdgeModel, self).__init__()
        self.edge_mlp = Seq(Linear(2*node_in_channels+edge_in_channels+global_in_channels, message_feats),
                            ReLU(),
                            Dropout(p=dropout),
                            Linear(message_feats, message_feats),
                            ReLU(),
                            Dropout(p=dropout),
                            Linear(message_feats, edge_out_channels))

    def forward(self, src, dest, edge_attr, u, batch):
        # source, target: [E, F_x], where E is the number of edges.
        # edge_attr: [E, F_e]
        # u: [B, F_u], where B is the number of graphs.
        # batch: [E] with max entry B - 1.
        out = torch.cat([src, dest, edge_attr, u[batch]], 1).float()
        return self.edge_mlp(out)


class NodeModel(Module):
    r"""A callable which updates a graph's node features based
    on its current node features, its graph connectivity, its edge
    features and its global features.

    Args:
        node_in_channels (int): number of node features of input graph.
        edge_in_channels (int): number of edge features of input graph.
        global_in_channels (int): number of global features of input graph.
        node_out_channels (int): number of node features after updates.
        message_feats (int, optional): number of intermediate features. (default :obj:`int`=32)
        dropout (float, optional): probability of an element to be zeroed. (default :obj:`float`=0.01)
    """
    def __init__(self, node_in_channels, edge_in_channels, global_in_channels, node_out_channels, message_feats: int=32, dropout: float=0.01):
        super(NodeModel, self).__init__()
        self.node_mlp_1 = Seq(Linear(2*node_in_channels+edge_in_channels, message_feats),
                              ReLU(),
                              Dropout(p=dropout),
                              Linear(message_feats, message_feats),
                              ReLU(),
                              Dropout(p=dropout),
                              Linear(message_feats, message_feats))
        self.node_mlp_2 = Seq(Linear(2*message_feats+node_in_channels+global_in_channels, message_feats),
                              ReLU(),
                              Dropout(p=dropout),
                              Linear(message_feats, message_feats),
                              ReLU(),
                              Dropout(p=dropout),
                              Linear(message_feats, node_out_channels))

    def forward(self, x, edge_index, edge_attr, u, batch):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        row, col = edge_index
        out = torch.cat([x[row], x[col], edge_attr], 1).float()
        out = self.node_mlp_1(out) # message
        agg_mean = scatter(out, col, dim=0, dim_size=x.size(0), reduce='mean')
        agg_max = custom_scatter(out, col, dim=0, dim_size=x.size(0), reduce='amax')
        out = torch.cat([x, agg_mean, agg_max, u[batch]], dim=1).float()
        return self.node_mlp_2(out) # update node with message


class GlobalModel(Module):
    r"""A callable which updates a graph's global features based
    on its node features, its graph connectivity, its edge features
    and its current global features.

    Args:
        node_in_channels (int): number of node features of input graph.
        global_in_channels (int): number of global features of input graph.
        global_out_channels (int): number of global features after updates.
        message_feats (int, optional): number of intermediate features. (default :obj:`int`=32)
        dropout (float, optional): probability of an element to be zeroed. (default :obj:`float`=0.01)
    """
    def __init__(self, node_in_channels, global_in_channels, global_out_channels, message_feats: int=32, dropout: float=0.01):
        super(GlobalModel, self).__init__()
        self.global_mlp = Seq(Linear(2*node_in_channels+global_in_channels, message_feats),
                              ReLU(),
                              Dropout(p=dropout),
                              Linear(message_feats, message_feats),
                              ReLU(),
                              Dropout(p=dropout),
                              Linear(message_feats, global_out_channels))

    def forward(self, x, edge_index, edge_attr, u, batch):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        out = torch.cat([u, scatter(x, batch, dim=0, dim_size=u.size(0), reduce='mean'), custom_scatter(x, batch, dim=0, dim_size=u.size(0), reduce='amax')], dim=1).float()
        return self.global_mlp(out)
