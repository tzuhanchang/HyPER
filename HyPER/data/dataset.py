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
        mode: Optional[str] = 'train'
    ):
        super(GraphDataset).__init__()
        

        self.file_path = path
        self.cfg = config['Dataset']
        self._objLabel = {}
        self._objPadding = {}
        self._train_mode = True if mode.lower() == 'train' else False
        self._use_index_select = False
        self._use_kfold = False
        self._use_EEtaPhiPt = False
        self._use_EPxPyPz = False
        self._node_norms   = self.cfg['node_norms']
        self._global_norms = self.cfg['global_norms']

        with h5py.File(self.file_path, 'r') as file:
            # Check dataset
            assert 'INPUTS' in file.keys(), "Data group `INPUTS` must be provided."
            assert 'GLOBAL' in list(file['INPUTS'].keys()), "`INPUTS/GLOBAL` is not found."
            self.dataset_size = len(file['INPUTS']['GLOBAL'])
            obj_count = 1
            for node_input in self.cfg['node_inputs']:
                assert node_input in list(file['INPUTS'].keys()), f"`INPUTS/{node_input}` is not found."
                assert self.dataset_size == len(file['INPUTS'][node_input]), f"INPUTS/{node_input} does not match with total number of events {self.dataset_size}."
                self._objPadding.update({node_input: len(file['INPUTS'][node_input][0])})
                self._objLabel.update({node_input: obj_count})  # Assign an unique ID for different types of objects
                obj_count += 1

            if self._train_mode:
                assert 'LABELS' in file.keys(), "Data group `LABELS` must be provided in `train` mode."
                assert 'ID' in list(file['LABELS'].keys()), "`LABELS/ID` is not found."
                assert self.dataset_size == len(file['LABELS']['ID']), f"`LABELS/ID` does not match with total number of events {self.dataset_size}."

            if self.cfg['boolean_filter'] is not None:
                assert self.dataset_size == len(file[self.cfg['boolean_filter']]), f"`{self.cfg['boolean_filter']}` does not match with total number of events {self.dataset_size}."
                self.index_select = np.array(range(self.dataset_size))[np.array(file[self.cfg['boolean_filter']],dtype=np.int64)==1]
                self._use_index_select = True
                self.dataset_size = len(self.index_select)

            self.node_features = self.cfg['node_features']
            assert self.node_features == list(file['INPUTS'][self.cfg['node_inputs'][0]].dtype.fields.keys()), \
                f"`node_features` defined in the configuration file ({self.node_features}) do not matches with the ones in the dataset: {list(file['INPUTS'][self.cfg['node_inputs'][0]].dtype.fields.keys())}"

            self.global_features = self.cfg['global_features']
            assert self.global_features == list(file['INPUTS']['GLOBAL'].dtype.fields.keys()), \
                f"`global_features` defined in the configuration file ({self.global_features}) do not matches with the ones in the dataset: {list(file['INPUTS']['GLOBAL'].dtype.fields.keys())}"

            if self.node_features[:4] == ['e','eta','phi','pt']:
                self._use_EEtaPhiPt = True
            elif self.node_features[:4] == ['e','px','py','pz']:
                self._use_EPxPyPz = True
            else:
                warnings.warn("You are not using the standard feature ordering or the naming scheme: ['e', 'eta', 'phi', 'pt'] or ['e', 'px', 'py', 'pz'] (for the first 4 features). This might cause problems in the edge construction stage.", UserWarning)
                self._use_EPxPyPz = True

            self.edge_identifiers = np.apply_along_axis(
                lambda x: 2**x, axis=0, arr=list(self.cfg['edge_target'])
            ).sum(axis=1)
            self.hyperedge_identifiers = np.apply_along_axis(
                lambda x: 2**x, axis=0, arr=list(self.cfg['hyperedge_target'])
            ).sum(axis=1)

            try:
                HE_ids = np.array(list(self.cfg['hyperedge_target']))
                self.hyperedge_order = HE_ids.shape[1]
            except ValueError:
                print("HyPER currently only support hyperedges with the same order.")

            self.n_node_features = len(self.node_features) + 1
            self.n_edge_features = 4
            self.n_global_features = len(file['INPUTS']["GLOBAL"][0][0])

            if self._node_norms is not None:
                assert len(self._node_norms) == len(self.node_features), "For each node features, a normalisation method must be provided in `node_norms`."
            else:
                self._node_norms = ['non']*len(self.node_features)

            if self._global_norms is not None:
                assert len(self._global_norms) == self.n_global_features, "For each global features, a normalisation method must be provided in `global_norms`."
            else:
                self._global_norms = ['non']*self.n_global_features

        self.get_std_scales()

    def get_std_scales(self):
        with h5py.File(self.file_path, 'r') as file:
            nodes = pd.DataFrame(np.concatenate([np.concatenate(np.array(file['INPUTS'][obj])) for obj in self.cfg['node_inputs']]))
            self._mean_nodes = nodes.mean().values
            self._std_nodes  = nodes.std().values

            glob = pd.DataFrame(np.concatenate(np.array(file['INPUTS']["GLOBAL"])))
            self._mean_glob = glob.mean().values
            self._std_glob  = glob.std().values

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
                for obj in self.cfg['node_inputs']
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