import h5py
import yaml
import torch
import numpy as np

from os import listdir, path as osp
from torch import Tensor
from torch_geometric.data import Data, InMemoryDataset
from torch_hep.lorentz import MomentumTensor
from itertools import permutations, combinations
from tqdm import tqdm
from typing import Callable, List, Optional, Tuple
from warnings import warn

from .transform import TransformFeatures
from .filter import TargetConnectivityFilter


class HyPERDataset(InMemoryDataset):
    
    """
    HyPERDataset is loaded into memory
    """
    def __init__(
        self,
        root: str,
        name: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        
        self.root = root
        self.name = name
        self.names = [
            osp.splitext(file)[0] for file in 
            listdir(osp.join(self.root, "raw"))]
        file_index = [i for i in range(len(self.names))
                            if self.names[i] == self.name]
        assert len(file_index) == 1
        self.file_index = file_index[0]

        # Parse database config file, setting instance attributes
        parsed_inputs = HyPERDataset.parse_config_file(f"{self.root}/config.yaml")
        
        self.node_input_names   = list(parsed_inputs['input']['nodes'].keys())
        self.input_id           = parsed_inputs['input']['nodes']
        self.input_pad_size     = parsed_inputs['input']['padding']
        self.edge_targets       = list(parsed_inputs['target']['edge'].values())
        self.hyperedge_targets  = list(parsed_inputs['target']['hyperedge'].values())
        self.hyperedge_order = len(self.hyperedge_targets[0])
        self.target_edge_ids, self.target_hyperedge_ids = self.target_ids()

        # Check 4-momentum inputs
        self._use_EEtaPhiPt = False
        self._use_EPxPyPz = False
        if parsed_inputs['input']['node_features'][:4] == ['e','eta','phi','pt']:
            self._use_EEtaPhiPt = True
        elif parsed_inputs['input']['node_features'][:4] == ['e','px','py','pz']:
            self._use_EPxPyPz = True
        else:
            warn("You are not using the standard feature ordering " \
                "or the naming scheme: ['e', 'eta', 'phi', 'pt'] or "\
                "['e', 'px', 'py', 'pz'] (for the first 4 features). "\
                "This might cause problems in the edge construction stage.", 
                UserWarning)
            self._use_EPxPyPz = True


        # Parse edge features
        self.edge_features_to_use, edge_transforms = self.parse_edge_features(parsed_inputs)
        
        node_transforms   = [eval(f"lambda x: {f}") for f in parsed_inputs['input']['node_transforms']]
        global_transforms = [eval(f"lambda x: {f}") for f in parsed_inputs['input']['global_transforms']]

        transforms = TransformFeatures(["x", "u", "edge_attr"],
            transforms=[
                node_transforms,
                global_transforms,
                edge_transforms
            ])
        
        if parsed_inputs['input']['pre_transform']:
            print("`pre_transform` is turned on.")
            pre_transform = transforms
        else:
            transform = transforms 

        # Check if filter is requested
        if 'filter' in parsed_inputs.keys():
            print("`pre_filter` is turned on.")
            pre_filter = TargetConnectivityFilter(
                num_edge_targets=parsed_inputs['filter']['num_edges'],
                num_hyperedge_targets=parsed_inputs['filter']['num_hyperedge'])

        super().__init__(root, transform, pre_transform, pre_filter,
                         force_reload=force_reload)
        self.load(self.processed_paths[self.file_index])
        
        
    def parse_edge_features(self,parsed_inputs):
        
        """
        User selects from set of pre-defined edge features
        Must select the appropriate transforms too
        """
        all_edge_feature_names = {"delta_eta": lambda x: x,
                                  "delta_phi": lambda x: x,
                                  "delta_R"  : lambda x: x,
                                  "kT"       : lambda x: torch.log(x),
                                  "Z"        : lambda x: torch.log(x),
                                  "M2"       : lambda x: torch.log(x)}
        
        if "edge_features" in parsed_inputs["input"]:
            requested_features  = set(parsed_inputs["input"]["edge_features"])
            HyPER_edge_features = set(all_edge_feature_names.keys())
            
            edge_features_to_use = list(HyPER_edge_features & requested_features)
            if len(requested_features - HyPER_edge_features) !=0:
                warn("Edge feature specified which is not in known HyPER edge features" , UserWarning)
            edge_transforms = [all_edge_feature_names[k] for k in edge_features_to_use]
        else:
            edge_features_to_use = list(all_edge_feature_names.keys())
            edge_transforms = list(all_edge_feature_names.values())    
            
        return edge_features_to_use , edge_transforms
    
    @staticmethod
    def parse_config_file(filename):
        with open(filename) as stream:
            try:
                return yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, 'processed')

    @property
    def raw_file_names(self) -> List[str]:
        return [f'{name}.h5' for name in self.names]

    @property
    def processed_file_names(self) -> List[str]:
        return [f'{name}.pt' for name in self.names]

    def node_attributes(
        self,
        INPUTS: h5py._hl.group.Group,
        index: int
    ) -> Tensor:
        """
        Construct node input tensor at :obj:`index` in the
        :obj:`INPUTS` HDF5 data group.
        """
        x = torch.cat([
                torch.cat([
                    # Input Features
                    torch.tensor(
                        np.array(INPUTS[obj][index].tolist()),
                        dtype=torch.float32
                    ),
                    # Assign an unique ID for each input object
                    torch.full(
                        (self.input_pad_size[obj], 1),
                        self.input_id[obj], 
                        dtype=torch.float32
                    )
                ], dim=1)
            for obj in self.node_input_names],
            dim=0
        )
        
        print(x)
        input()
        return x[torch.any(x.isnan(),dim=1)==False]

    def edge_attributes(
        self,
        x: Tensor
    ) -> Tuple[Tensor,Tensor]:
        """
        Construct edge input tensor assuming a fully connected
        di-graph using pre-existing node input tensor :obj:`x`.

        Note:
            `HyPERDataset` assumes the first four features describles
            four-momentum. This means the first four features are 
            expected to be ordered, either:
                (E, \eta, \phi, p_T) or (E, p_x, p_y, p_z).
            To learn more how four-momentum are calculated, see
            https://github.com/tzuhanchang/pytorch_hep
        """
        num_nodes = x.size(dim=0)
        edge_index = torch.tensor(
            list(permutations(range(num_nodes), r=2)),
            dtype=torch.int64
        ).permute(dims=(1,0))

        if self._use_EEtaPhiPt:
            node_i = MomentumTensor.EEtaPhiPt(
                x[:,0:4].index_select(0, index=edge_index[0]))
            node_j = MomentumTensor.EEtaPhiPt(
                x[:,0:4].index_select(0, index=edge_index[1]))
        if self._use_EPxPyPz:
            node_i = MomentumTensor(
                x[:,0:4].index_select(0, index=edge_index[0]))
            node_j = MomentumTensor(
                x[:,0:4].index_select(0, index=edge_index[1]))

        # Compute all 
        dEta = node_i.eta - node_j.eta
        dPhi = torch.arctan2(
            torch.sin(node_i.phi-node_j.phi),
            torch.cos(node_i.phi-node_j.phi))
        dR  = torch.sqrt((dEta)**2+(dPhi)**2)
        kT  = torch.min(node_i.pt,node_j.pt)*dR
        Z_edge   = torch.min(node_i.pt,node_j.pt)/(node_i.pt + node_j.pt)
        M2  = (node_i + node_j).m

        computed_edge_features = {
            "dEta": dEta,
            "dPhi": dPhi,
            "dR": dR,
            "kT": kT,
            "Z": Z_edge,
            "M2": M2
        }
        
        required_edge_features = [computed_edge_features[k] for k in self.edge_features_to_use]
        edge_attr = torch.cat([required_edge_features],dim=1)
        
        return edge_index, edge_attr

    def global_attributes(
        self,
        INPUTS: h5py._hl.group.Group,
        index: int
    ) -> Tensor:
        """
        Construct global input tensor at :obj:`index` in the
        :obj:`INPUTS` HDF5 data group.
        """
        return torch.tensor(np.array(INPUTS['GLOBAL'][index].tolist()),
                    dtype=torch.float32)

    def hyperedge_index(self, x: Tensor) -> Tensor:
        """
        Construct hyperedge index tensor.
        """
        num_nodes = x.size(0)
        return torch.tensor(list(combinations(range(num_nodes),
            r=self.hyperedge_order)),dtype=torch.int64).permute(dims=(1,0))

    def node_ids(
        self,
        LABELS: h5py._hl.group.Group,
        index: int
    ) -> Tensor:
        """
        Assign an unique ID for all nodes.

        Note:
            The unique ID is created using the Cantor pairing 
            function:
            .. math::
                \pi(k_1,k_2) = \frac{1}{2}(k_1+k_2)(k_1+k_2+1)+k_2
            where k_1 is the `input_id` and k_2 is the `node_id`.
        """
        k1 = torch.cat([
            # Input IDs
            torch.full(
                (self.input_pad_size[obj], 1),
                self.input_id[obj], 
                dtype=torch.float32
            )
            for obj in self.node_input_names],
            dim=0
        )
        k2 = torch.cat([
            # Node labels
            torch.tensor(
                np.array(LABELS[obj][index].tolist()),
                dtype=torch.float32
            )
            for obj in self.node_input_names],
            dim=0
        )
        pi = (k1+k2)*(k1+k2+1)/2 + k2
        return pi[pi.isnan()==False]

    def edge_labels(
        self,
        edge_index: Tensor,
        node_ids: Tensor
    ) -> Tensor:
        """
        Construct edge labels based on the :obj:`self.target_edge_ids` uniquely
        assigned to each node.
        """
        edge_attr_t = torch.zeros((edge_index.size(1),1), dtype=torch.float32)

        edge_index_id_filled = torch.cat([
            node_ids.index_select(0,edge_index[i]).unsqueeze(0) 
            for i in range(edge_index.size(0))],dim=0)

        for targets in self.target_edge_ids:
            edge_index_decision = torch.full(edge_index_id_filled.shape,
                False, dtype=torch.bool)
            for endpoint in targets:
                edge_index_decision += (edge_index_id_filled == endpoint)
            edge_attr_t[torch.argwhere(torch.all(
                edge_index_decision==True, dim=0)==True)] = 1
        return edge_attr_t

    def hyperedge_labels(
        self,
        hyperedge_index: Tensor,
        node_ids: Tensor
    ) -> Tensor:
        """
        Construct edge labels based on the :obj:`self.target_hyperedge_ids` uniquely
        assigned to each node.
        """
        hyperedge_attr_t = torch.zeros((hyperedge_index.size(1),1), dtype=torch.float32)

        hyperedge_index_id_filled = torch.cat([
            node_ids.index_select(0,hyperedge_index[i]).unsqueeze(0) 
            for i in range(hyperedge_index.size(0))],dim=0)

        for targets in self.target_hyperedge_ids:
            hyperedge_index_decision = torch.full(hyperedge_index_id_filled.shape,
                False, dtype=torch.bool)
            for endpoint in targets:
                hyperedge_index_decision += (hyperedge_index_id_filled == endpoint)
            hyperedge_attr_t[torch.argwhere(torch.all(
                hyperedge_index_decision==True, dim=0)==True)] = 1
        return hyperedge_attr_t

    def target_ids(self) -> Tuple[List,List]:
        """
        Assign each edge/hyperedge target with an unique ID.
        """
        target_edge_ids = []
        if len(self.edge_targets) != 0:
            for target in self.edge_targets:
                tmp = []
                for label in target:
                    k1, k2 = label.split('-')
                    k1, k2 = int(k1), int(k2)
                    tmp.append((k1+k2)*(k1+k2+1)/2 + k2)
                target_edge_ids.append(tmp)
        target_hyperedge_ids = []
        if len(self.hyperedge_targets) != 0:
            for target in self.hyperedge_targets:
                tmp = []
                for label in target:
                    k1, k2 = label.split('-')
                    k1, k2 = int(k1), int(k2)
                    tmp.append((k1+k2)*(k1+k2+1)/2 + k2)
                target_hyperedge_ids.append(tmp)
        return target_edge_ids, target_hyperedge_ids

    def process(self) -> None:
        """
        Process all raw files in the `raw_dir`. This is only done 
        once unless `force_reload=True` is set.
        """
        # Loop through files
        for i in range(len(self.raw_file_names)):
            raw_file_name = self.raw_file_names[i]
            data_list = []
            # Open the HDF5 file
            with h5py.File(osp.join(self.raw_dir, raw_file_name),
                'r') as file:
                num_entries = len(file['INPUTS/GLOBAL'])
                # Loop through events
                for j in tqdm(range(num_entries),
                              desc=f"File {raw_file_name}",
                              unit="events"):
                    # Constructing node input tensor
                    x = self.node_attributes(file['INPUTS'], j)
                    # Constructing edge input tensor
                    edge_index, edge_attr = self.edge_attributes(x)
                    # Constructing global input tensor
                    u = self.global_attributes(file['INPUTS'], j)
                    # Constructing hyperedge index tensor
                    hyperedge_index = self.hyperedge_index(x)
                    # Assign unique target ids
                    ids = self.node_ids(file['LABELS'], j)
                    # Constructing edge label tensor
                    edge_attr_t = self.edge_labels(edge_index, ids)
                    # Constructing hyperedge label tensor
                    hyperedge_attr_t = self.hyperedge_labels(hyperedge_index, ids)

                    data_list.append(Data(x=x,
                                          edge_index=edge_index,
                                          edge_attr=edge_attr,
                                          u=u,
                                          hyperedge_index=hyperedge_index,
                                          edge_attr_t=edge_attr_t,
                                          hyperedge_attr_t=hyperedge_attr_t))

            if self.pre_filter is not None:
                data_list = [data for data in data_list if self.pre_filter(data)]

            if self.pre_transform is not None:
                data_list = [self.pre_transform(data) for data in data_list]

            self.save(data_list, self.processed_paths[i])