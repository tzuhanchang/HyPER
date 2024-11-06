import h5py
import torch
import warnings

import numpy as np
import pandas as pd

from torch import Tensor
from torch_geometric.data import Data
from torch_hep.lorentz import MomentumTensor
from itertools import permutations, combinations
from typing import List, Tuple, Optional
from omegaconf import DictConfig

from HyPER.data import _check_dataset


class EdgeEmbedding(object):
    def __init__(self, u: MomentumTensor, v: MomentumTensor):
        r"""Calculate edge embedding.

        Input tensors should store node kinematics in EEtaPhiPt or MEtaPhiPt order.
        Args:
            u (torch.Tensor): input tensor with size torch.Size([N,4]).
            v (torch.Tensor): input tensor with size torch.Size([N,4]).
        """
        self.u = u
        self.v = v

    def dEta(self):
        return self.u.eta - self.v.eta

    def dPhi(self):
        return torch.arctan2(torch.sin(self.u.phi-self.v.phi),torch.cos(self.u.phi-self.v.phi))

    def dR(self):
        return torch.sqrt((self.dEta())**2+(self.dPhi())**2)

    def M(self):
        p4 = self.u + self.v
        return p4.m


class GraphDataset(torch.utils.data.Dataset):
    r"""HyPER Graph Dataset interfaced with HDF5 file format. `GraphDataset`
    embeds data into the graph structure when required.

    Args:
        path (str): path to the dataset.
        config (str): path to the network configuration file.
        mode (optional, str): dataset mode, `train` or `eval`. (default: :obj:`train`)
    """
    def __init__(
        self,
        path: str,
        config: DictConfig,
        mode: Optional[str] = 'train',
        _params: Optional[dict] = None
    ):
        super(GraphDataset).__init__()

        self.file_path = path

        if _params is None:
            _params = _check_dataset(path, config, mode)
        for key, value in _params.items():
            setattr(self, key, value)

    def normalization(self, src: Tensor, methods: List, obj: str) -> Tensor:
        r"""Normalise input features.

        Args:
            src (torch.tensor): input tensor.
            obj (str): the graph object to be normalised.

        Note:
            For edge, only the 4th feature (M: invariant mass) is normalised.
        """
        _norm_fns = {"log": lambda x, mean, std: torch.log(x), "z-score": lambda x, mean, std: (x - mean) / std, "non": lambda x, mean, std: x}
        if obj.lower() == "edge":
            src[:,3] = _norm_fns["log"](src[:,3], None, None)
        else:
            for feature_idx in range(len(methods)):
                if obj.lower() == "node":
                    try:
                        src[:,feature_idx] = _norm_fns[methods[feature_idx].lower()](src[:,feature_idx], self._mean_nodes[feature_idx], self._std_nodes[feature_idx])
                    except KeyError:
                        raise ValueError("Available normalisation methods are: 'log', 'z-score' and 'non'.")
                if obj.lower() == "global":
                    try:
                        src[:,feature_idx] = _norm_fns[methods[feature_idx].lower()](src[:,feature_idx], self._mean_glob[feature_idx], self._std_glob[feature_idx])
                    except KeyError:
                        raise ValueError("Available normalisation methods are: 'log', 'z-score' and 'non'.")
        return src

    def get_node_feats(self, inputs_db, index):
        r"""Get node features.

        Args:
            inputs_db: `INPUTS` data group in the h5 file.
            index (int): event index.

        :rtype: :class:`torch.tensor`
        """
        x = torch.concatenate(
            [ 
                torch.cat([torch.tensor(np.array(inputs_db[obj][index].tolist()),dtype=torch.float32), torch.full((self._objPadding[obj],1),self._objLabel[obj],dtype=torch.float32)],dim=1) 
                for obj in self.node_inputs
            ],
            dim=0
        )
        return x[torch.any(x.isnan(),dim=1)==False]

    def get_edge_feats(self, x: Tensor) -> Tuple[Tensor,Tensor]:
        r"""Build a fully connected graph, and get edge feature and index tensors.

        Args:
            x (torch.tensor): node feature tensor.

        :rtype: :class:`Tuple[torch.tensor,torch.tensor]`
        """
        num_nodes = x.size(dim=0)
        edge_index = torch.tensor(list(permutations(range(num_nodes),r=2)),dtype=torch.int64).permute(dims=(1,0))

        if self._use_EEtaPhiPt:
            edge_i = MomentumTensor.EEtaPhiPt(x[:,0:4].index_select(0, index=edge_index[0]))
            edge_j = MomentumTensor.EEtaPhiPt(x[:,0:4].index_select(0, index=edge_index[1]))
        if self._use_EPxPyPz:
            edge_i = MomentumTensor(x[:,0:4].index_select(0, index=edge_index[0]))
            edge_j = MomentumTensor(x[:,0:4].index_select(0, index=edge_index[1]))

        # Get edge embedding
        embedding = EdgeEmbedding(edge_i, edge_j)
        edge_attr = torch.concatenate(
            [ embedding.dEta(), embedding.dPhi(), embedding.dR(), embedding.M() ],
            dim=1
        )
        return edge_attr, edge_index

    def get_global_feats(self, inputs_db, index: int) -> Tensor:
        r"""Get global features.

        Args:
            inputs_db: `INPUTS` data group in the h5 file.
            index (int): event index.

        :rtype: :class:`torch.tensor`
        """
        return torch.tensor(np.array(inputs_db['GLOBAL'][index].tolist()),dtype=torch.float32)

    def get_node_ID(self, labels_db, index: int) -> Tensor:
        r"""Get node ID.

        Args:
            labels_db: `LABELS` data group in the h5 file.
            index (int): event index.

        :rtype: :class:`torch.tensor`
        """
        ids = torch.tensor(np.array(labels_db['ID'][index]), dtype=torch.float32)
        return ids[ids.isnan()==False].reshape(-1,1)

    def get_edge_labels(
        self,
        edge_index: Tensor,
        NodeID: Tensor
    ) -> Tensor:
        r"""Get edge labels.

        Args:
            edge_index (torch.tensor): edge indices.
            NodeID (torch.tensor): node types.

        :rtype: :class:`torch.tensor`
        """
        endpoints_ids = torch.concatenate([2**NodeID.index_select(0,index=edge_index[0]), 2**NodeID.index_select(0,index=edge_index[1])],dim=1).sum(dim=1)
        target_idx = torch.concatenate(
            [ torch.argwhere(endpoints_ids==id) for id in self.edge_identifiers ]
        ).squeeze()
        # Scatter the edge labels in a empty tensor
        return torch.zeros(endpoints_ids.size(), dtype=torch.float32).scatter(dim=0, index=target_idx, src=torch.full(target_idx.size(), 1, dtype=torch.float32)).reshape(-1,1)

    def get_hyperedges(
        self,
        num_nodes: int
    ) -> Tensor:
        r"""Get hyperedges.

        Args:
            num_nodes (int): number of nodes.

        :rtype: :class:`torch.tensor`
        """
        return torch.tensor(list(combinations(range(num_nodes),r=self.hyperedge_order)),dtype=torch.int64).permute(dims=(1,0))

    def get_hyperedge_labels(
        self,
        NodeID: Tensor,
        hyperedge_index: Tensor
    ) -> Tensor:
        r"""Get hyperedge labels.

        Args:
            NodeID (torch.tensor): node types.
            identifiers (List): integer hyperedge identification value.

        :rtype: :class:`torch.tensor`
        """
        endpoints_ids = torch.concatenate(
            [ 2**NodeID.index_select(0, index=hyperedge_index[row]) for row in range(self.hyperedge_order) ],
            dim=1
        ).sum(dim=1)
        target_idx = torch.concatenate(
            [ torch.argwhere(endpoints_ids==id) for id in self.hyperedge_identifiers ]
        ).squeeze()

        hyperegde_t = torch.zeros(endpoints_ids.size(), dtype=torch.float32).scatter(dim=0, index=target_idx, src=torch.full(target_idx.size(), 1, dtype=torch.float32)).reshape(-1,1)
        return hyperegde_t

    def __getitem__(self, index) -> Tensor:
        with h5py.File(self.file_path, 'r') as file:
            if self._use_index_select:
                index = self.index_select[index]

            x = self.get_node_feats(file['INPUTS'], index)
            u = self.get_global_feats(file['INPUTS'], index)
            if self._train_mode:
                NodeID = self.get_node_ID(file['LABELS'], index)

        edge_attr, edge_index = self.get_edge_feats(x)
        hyperedge_index = self.get_hyperedges(x.size(0))

        x = self.normalization(x, self._node_norms, obj='node')
        u = self.normalization(u, self._global_norms, obj='global')
        edge_attr = self.normalization(edge_attr, None, obj='edge')

        if self._train_mode:
            edge_attr_t = self.get_edge_labels(edge_index, NodeID)
            x_t = self.get_hyperedge_labels(NodeID, hyperedge_index)
            return Data(x_s=x, num_nodes=x.size(dim=0), edge_attr_s=edge_attr, edge_index=edge_index, edge_index_h=hyperedge_index, u_s=u, edge_attr_t=edge_attr_t, x_t=x_t)
        else:
            return Data(x_s=x, num_nodes=x.size(dim=0), edge_attr_s=edge_attr, edge_index=edge_index, edge_index_h=hyperedge_index, u_s=u)

    def __len__(self):
        return self.dataset_size