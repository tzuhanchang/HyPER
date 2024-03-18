import re
import h5py
import yaml
import torch
import warnings

import numpy as np

from torch import Tensor
from torch_geometric.data import Data
from torch_hep.lorentz import MomentumTensor
from itertools import permutations, groupby, combinations
from typing import List, Tuple, Optional


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
        configs (str): a `yaml` file saves dataset configurations.
    """
    def __init__(self, path: str, configs: str, use_index_select: Optional[bool] = False):
        super(GraphDataset).__init__()

        with open(configs, 'r') as config_file:
            cf = yaml.safe_load(config_file)
            assert 'INPUTS' in cf.keys(), "Data group `INPUTS` must be provided."
            assert 'LABELS' in cf.keys(), "Data group `LABELS` must be provided."

            assert 'Objects' in cf['INPUTS'].keys(), "A list of `Objects` must be provided."
            assert 'Features' in cf['INPUTS'].keys(), "A list of `Features` must be provided."
            assert 'global' in cf['INPUTS'].keys(), "Event `global` must be provided."

            self.objects = cf['INPUTS']['Objects']
            self.node_features = list(cf['INPUTS']['Features'].keys())
            self.node_scalings = list(cf['INPUTS']['Features'].values())
            self.global_features = list(cf['INPUTS']['global'].keys())
            self.global_scalings = list(cf['INPUTS']['global'].values())

            self.edge_identifiers = np.apply_along_axis(
                lambda x: 2**x, axis=0, arr=list(cf['LABELS']['Edges'].values())
            ).sum(axis=1)
            self.hyperedge_identifiers = np.apply_along_axis(
                lambda x: 2**x, axis=0, arr=list(cf['LABELS']['Hyperedges'].values())
            ).sum(axis=1)

            try:
                HE_ids = np.array(list(cf['LABELS']['Hyperedges'].values()))
                self.hyperedge_order = HE_ids.shape[1]
            except ValueError:
                print("HyPER currently only support hyperedges with the same order.")

        self.file_path = path
        self.use_index_select = use_index_select

        with h5py.File(self.file_path, 'r') as file:
            # Check dataset
            assert 'INPUTS' in file.keys(), "Data group `INPUTS` must be provided."
            assert 'LABELS' in file.keys(), "Data group `LABELS` must be provided."

            self.inputs = list(file['INPUTS'].keys())
            self.labels = list(file['LABELS'].keys())

            assert 'VertexID' in self.labels, "`VertexID` must be provided in the dataset."

            assert set(self.objects).issubset(set(self.inputs)), "One or more `Objects` provided not found in the dataset."
            g = groupby([file['INPUTS'][obj].dtype.fields.keys() for obj in self.objects])
            assert next(g, True) and not next(g, False), "All provided node objects (`jet`, `electron` etc.) must have the same `numpy.dtype`, including the name of the features."
            assert self.node_features == list(file['INPUTS'][self.objects[0]].dtype.fields.keys()), "Defined `Features` do not match the ones found in dataset, they must also be ordered."

            assert 'global' in self.inputs, "`global` variables must be provided."
            assert self.global_features == list(file['INPUTS']['global'].dtype.fields.keys()), "Defined `global` variables do not match the ones found in dataset, they must also be ordered."

            if self.node_features[:4] != ['e', 'eta', 'phi', 'pt']:
                warnings.warn("You are not using the standard feature ordering or the naming scheme: ['e', 'eta', 'phi', 'pt'] (for the first 4). This might cause problem in the edge computing stage.", UserWarning)

            self.size = len(file['INPUTS'][self.objects[0]])

            self.index_select = None
            if 'IndexSelect' in file['LABELS']:
                self.index_select = np.array(range(self.size))[np.array(file['LABELS']['IndexSelect'],dtype=np.int64)==1]
                if self.use_index_select:
                    self.size = len(self.index_select)
            else:
                if self.use_index_select:
                    warnings.warn("`IndexSelect` not found in `LABELS`. No index selection will be made.")
                    self.use_index_select = False


    def get_node_feats(self, inputs_db, index):
        r"""Get node features.

        Args:
            inputs_db: `INPUTS` data group in the h5 file.
            index (int): event index.

        :rtype: :class:`torch.tensor`
        """
        x = torch.concatenate(
            [ torch.tensor(np.array(inputs_db[obj][index].tolist()),dtype=torch.float32) for obj in self.objects ],
            dim=0
        )
        return x[x[:,-1]!=0]

    def get_edge_feats(self, x: Tensor) -> Tuple[Tensor,Tensor]:
        r"""Build a fully connected graph, and get edge feature and index tensors.

        Args:
            x (torch.tensor): node feature tensor.

        :rtype: :class:`Tuple[torch.tensor,torch.tensor]`
        """
        num_nodes = x.size(dim=0)
        edge_index = torch.tensor(list(permutations(range(num_nodes),r=2))).permute(dims=(1,0))

        edge_i = MomentumTensor.EEtaPhiPt(x[:,0:4].index_select(0, index=edge_index[0]))
        edge_j = MomentumTensor.EEtaPhiPt(x[:,0:4].index_select(0, index=edge_index[1]))

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
        return torch.tensor(np.array(inputs_db['global'][index].tolist()),dtype=torch.float32)

    def get_VertexID(self, labels_db, index: int) -> Tensor:
        r"""Get object IDs.

        Args:
            labels_db: `LABELS` data group in the h5 file.
            index (int): event index.

        :rtype: :class:`torch.tensor`
        """
        ids = torch.tensor(np.array(labels_db['VertexID'][index]), dtype=torch.float32)
        return ids[ids!=-9].reshape(-1,1)

    def get_edge_labels(
        self,
        edge_index: Tensor,
        VertexID: Tensor
    ) -> Tensor:
        r"""Get edge labels.

        Note:
            `identifiers` contains unique integer values corresponding to the edges one want to label.
            It is defined according to vertex ids (`VertexID`) using:
                    :math:`2^u + 2^v`
            where u and v are labels of the edge's end points.

        Args:
            edge_index (torch.tensor): edge indices.
            VertexID (torch.tensor): node types.

        :rtype: :class:`torch.tensor`
        """
        endpoints_ids = torch.concatenate([2**VertexID.index_select(0,index=edge_index[0]), 2**VertexID.index_select(0,index=edge_index[1])],dim=1).sum(dim=1)
        target_idx = torch.concatenate(
            [ torch.argwhere(endpoints_ids==id) for id in self.edge_identifiers ]
        ).squeeze()
        # Scatter the edge labels in a empty tensor
        return torch.zeros(endpoints_ids.size(), dtype=torch.float32).scatter(dim=0, index=target_idx, src=torch.full(target_idx.size(), 1, dtype=torch.float32)).reshape(-1,1)

    def get_hyperedge_labels(
        self,
        VertexID: Tensor
    ) -> Tensor:
        r"""Get hyperedge labels.

        Note:
            `identifiers` contains unique integer values corresponding to the edges one want to label.
            It is defined according to vertex ids (`VertexID`) using:
                    :math:`\sum_i 2^i \quad \forall i\in\{0,1,\dots, N\}`
            where N is the hyperedge order.

        Args:
            VertexID (torch.tensor): node types.
            identifiers (List): integer hyperedge identification value.

        :rtype: :class:`torch.tensor`
        """
        num_nodes = VertexID.size(dim=0)
        hyperedge_index = torch.tensor(list(combinations(range(num_nodes),r=self.hyperedge_order))).permute(dims=(1,0))

        endpoints_ids = torch.concatenate(
            [ 2**VertexID.index_select(0, index=hyperedge_index[row]) for row in range(self.hyperedge_order) ],
            dim=1
        ).sum(dim=1)
        target_idx = torch.concatenate(
            [ torch.argwhere(endpoints_ids==id) for id in self.hyperedge_identifiers ]
        ).squeeze()
        return torch.zeros(endpoints_ids.size(), dtype=torch.float32).scatter(dim=0, index=target_idx, src=torch.full(target_idx.size(), 1, dtype=torch.float32)).reshape(-1,1)

    def scale_features(self, src: Tensor, scaling_methods: List):
        for feature_idx in range(len(scaling_methods)):
            if scaling_methods[feature_idx].lower() == "log":
                src[:,feature_idx] = torch.log(src[:,feature_idx])
            elif scaling_methods[feature_idx].lower() == "pi":
                src[:,feature_idx] = src[:,feature_idx]/torch.pi
            elif re.search(r'\d+', scaling_methods[feature_idx].lower()):
                src[:,feature_idx] = src[:,feature_idx]/int(re.search(r'\d+', scaling_methods[feature_idx].lower()).group())
            elif scaling_methods[feature_idx].lower() == "none":
                pass
            else:
                warnings.warn("%s is not available, scale is not applied.. Available methods are: `log`, `pi` (divide by pi), `dN` (divide by N) and `none`."%(scaling_methods[feature_idx]))
        return src

    def __getitem__(self, index) -> Tensor:
        with h5py.File(self.file_path, 'r') as file:
            if self.use_index_select:
                index = self.index_select[index]

            x = self.get_node_feats(file['INPUTS'], index)
            u = self.get_global_feats(file['INPUTS'], index)
            VertexID = self.get_VertexID(file['LABELS'], index)

        edge_attr, edge_index = self.get_edge_feats(x)
        edge_attr_t = self.get_edge_labels(edge_index, VertexID)
        x_t = self.get_hyperedge_labels(VertexID)

        x = self.scale_features(x, scaling_methods=self.node_scalings)
        u = self.scale_features(u, scaling_methods=self.global_scalings)
        edge_attr = self.scale_features(edge_attr, scaling_methods=['none','none','none','log'])

        return Data(x_s=x, num_nodes=x.size(dim=0), edge_attr_s=edge_attr, edge_index=edge_index, u_s=u, edge_attr_t=edge_attr_t, x_t=x_t)

    def __len__(self):
        return self.size