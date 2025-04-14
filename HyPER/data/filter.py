import torch

from torch_geometric.data import Data, HeteroData
from typing import List, Union

class TargetConnectivityFilter(object):
    r"""Filter events according to the connectivity
    of the target graph.

    Args:
        num_edge_targets (int): Number of required target
            edges.
        num_hyperedge_targets (int): Number of required
            target hyperedges. 
    """
    def __init__(
        self,
        num_edge_targets: int,
        num_hyperedge_targets: int
    ):
        self.num_edge_targets = num_edge_targets
        self.num_hyperedge_targets = num_hyperedge_targets

    def __call__(
        self,
        data: Union[Data, HeteroData],
    ) -> Union[Data, HeteroData]:
        pass_filter = False
        for store in data.stores:
            if (torch.sum(store["edge_attr_t"]) == self.num_edge_targets and
                torch.sum(store["hyperedge_attr_t"]) == self.num_hyperedge_targets):
                pass_filter = True

        return pass_filter