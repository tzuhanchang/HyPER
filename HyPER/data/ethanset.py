from typing import Callable, List, Optional, Tuple
from warnings import warn
from os import listdir, path as osp
from itertools import permutations,combinations
from tqdm import tqdm 

import h5py
import yaml
import torch
import numpy as np
import awkward as ak
import vector
import numpy.lib.recfunctions as rf 

from torch import Tensor
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.io import fs

from .transform import TransformFeatures
from .filter import TargetConnectivityFilter

class EthanDataset(InMemoryDataset):
    
    """
    Builds the graph dataset. Inherits from the PyTorchGeometric InMemoryDataset,
        which comes with __init__ and process methods.
        
    Process is called when the data is loaded. It calls the various methods for building the nodes, edge and global parameters
    
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
        self.name = name # This is the name of the file to be loaded
        # self.names is a list of all names in the directory
        
        self.names = [
            osp.splitext(file)[0] for file in 
            listdir(osp.join(self.root, "raw"))]
        # Filter self.names to only include the files that match input_name
        self.names = [name for name in self.names if name == self.name]
        # Throw an error if no file matching input_name exists
        if len(self.names) == 0:
            raise FileNotFoundError(f"No file matching '{self.input_name}' found in the 'raw' directory.")
        
        file_index = [i for i in range(len(self.names))
                            if self.names[i] == self.name]
        assert len(file_index) == 1
        self.file_index = file_index[0]

        # Parse database config file, setting instance attributes
        parsed_inputs = EthanDataset.parse_config_file(f"{self.root}/config.yaml")
        
        self.node_input_names   = list(parsed_inputs['input']['nodes'].keys())
        self.input_id           = parsed_inputs['input']['nodes']
        self.input_pad_size     = parsed_inputs['input']['padding']
        
        self.edge_targets       = list(parsed_inputs['target']['edge'].values())
        self.hyperedge_targets  = list(parsed_inputs['target']['hyperedge'].values())
        self.hyperedge_order = len(self.hyperedge_targets[0])
        self.target_edge_ids, self.target_hyperedge_ids = self.assign_target_ids()
    
        # Check 4-momentum inputs
        self._use_EEtaPhiPt = False
        self._use_EPxPyPz = False
        if parsed_inputs['input']['node_features'][:4] == ['e','eta','phi','pt']: # Make this more flexible
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

        # Parse edge features, returning dict
        self.edge_features_to_use = self.parse_edge_features(parsed_inputs)
        
        node_transforms   = [eval(f"lambda x: {f}") for f in parsed_inputs['input']['node_transforms']]
        global_transforms = [eval(f"lambda x: {f}") for f in parsed_inputs['input']['global_transforms']]

        transforms = TransformFeatures(["x", "u", "edge_attr"],
            transforms=[
                node_transforms,
                global_transforms,
                list(self.edge_features_to_use.values())
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
        all_edge_feature_names = {
            "delta_eta": lambda x: x,
            "delta_phi": lambda x: x,
            "delta_R"  : lambda x: x,
            "kT"       : lambda x: torch.log(x),
            "Z"        : lambda x: torch.log(x),
            "M2"       : lambda x: torch.log(x)
        }

        if "edge_features" in parsed_inputs["input"]:
            requested_features  = parsed_inputs["input"]["edge_features"]
            HyPER_edge_features = list(all_edge_feature_names.keys())
            
            # Check if there are any requested features not in the known edge features
            if len(set(requested_features) - set(HyPER_edge_features)) != 0:
                # warn("Edge feature specified which is not in known HyPER edge features", UserWarning)
                raise KeyError("Edge features have been specified which do not match the pre-defined attributes")
            # Create a dictionary containing only the requested edge features
            edge_features_to_use = {k: all_edge_feature_names[k] for k in requested_features}
        else:
            # If no specific edge features are requested, use all available edge features
            edge_features_to_use = all_edge_feature_names
            
        return edge_features_to_use 
    
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

    
    def assign_target_ids(self) -> Tuple[List, List]:
        """
        Assign each edge/hyperedge target with a unique ID.
        
        Uses the Cantor function to define each ID, 
        then parses the input edge and hyperedge targets to define target_edge_ids and target_hyperedge_ids
        """
        def compute_target_id(label: str) -> int:
            k1, k2 = map(int, label.split('-'))
            return (k1 + k2) * (k1 + k2 + 1) // 2 + k2

        # Compute target edge IDs
        target_edge_ids = [
            [compute_target_id(label) for label in target]
            for target in self.edge_targets
        ] if self.edge_targets else []

        # Compute target hyperedge IDs
        target_hyperedge_ids = [
            [compute_target_id(label) for label in target]
            for target in self.hyperedge_targets
        ] if self.hyperedge_targets else []

        return target_edge_ids, target_hyperedge_ids

    
    def build_node_attributes(self,input_h5:h5py._hl.group.Group) -> List:
        
        """
        Constructs node attribute tensor 'x' from input h5 group

        Parameters:
        - input_h5 (h5py._hl.group.Group): h5_file["INPUTS"].

        Returns:
        int or float: The maximum value in the list.
        """
        
        object_arrays = []
        for name,uid in self.input_id.items():
            t = torch.Tensor(rf.structured_to_unstructured(input_h5[name][:]))
            uids = torch.full((t.shape[0],t.shape[1],1),uid)
            object_arrays.append(torch.cat((t,uids),dim=2))
            
        combined = torch.cat(object_arrays,dim=1)
        
        remove_nan_mask = ~torch.any(combined.isnan(),dim=2) # Remove the padded entries in each event
        Nobjects = torch.count_nonzero(remove_nan_mask,dim=1)
        return combined[remove_nan_mask], Nobjects
    
    def build_edge_attributes(self,input_h5:h5py._hl.group.Group) -> torch.Tensor:
        
        """
        Constructs node attribute tensor 'x' from input h5 group

        Parameters:
        - input_h5 (h5py._hl.group.Group): h5_file["INPUTS"].

        Returns:
        int or float: The maximum value in the list.
        """
    
        object_vectors_list = []
        
        for obj in self.input_id.keys():
        
            # Here we should declare which type it is
            if self._use_EEtaPhiPt:
                obj_e   = ak.drop_none(ak.nan_to_none(ak.Array(input_h5[obj][:]["e"])))
                obj_pt  = ak.drop_none(ak.nan_to_none(ak.Array(input_h5[obj][:]["pt"])))
                obj_eta = ak.drop_none(ak.nan_to_none(ak.Array(input_h5[obj][:]["eta"])))
                obj_phi = ak.drop_none(ak.nan_to_none(ak.Array(input_h5[obj][:]["phi"])))    
                obj_vectors = vector.zip({"pt":obj_pt , "eta":obj_eta , "phi": obj_phi , "e": obj_e})
            elif self._use_EPxPyPz:
                obj_e   = ak.drop_none(ak.nan_to_none(ak.Array(input_h5[obj][:]["e"])))
                obj_px  = ak.drop_none(ak.nan_to_none(ak.Array(input_h5[obj][:]["px"])))
                obj_py  = ak.drop_none(ak.nan_to_none(ak.Array(input_h5[obj][:]["py"])))
                obj_pz  = ak.drop_none(ak.nan_to_none(ak.Array(input_h5[obj][:]["pz"])))    
                obj_vectors = vector.zip({"px":obj_px , "py":obj_py , "pz": obj_pz , "e": obj_e})
            
            object_vectors_list.append(obj_vectors)
            
        vectors = ak.concatenate(object_vectors_list,axis=1)
        
        remove_self_pairing = lambda arr,mask: ak.drop_none(ak.nan_to_none(ak.where(mask, np.nan, arr)))
        map_awk_to_torch    = lambda arr: torch.tensor(ak.flatten(ak.flatten(arr)).to_numpy(),dtype=torch.float32).reshape(-1,1)
        
        DR   = vectors[:, None].deltaR(vectors)
        Deta = vectors[:, None].deltaeta(vectors)
        Dphi = vectors[:,None].deltaphi(vectors)
        M    = (vectors[:,None] + vectors).m
        # kT  = torch.min(node_i.pt,node_j.pt)*dR
        # Z_edge   = torch.min(node_i.pt,node_j.pt)/(node_i.pt + node_j.pt)
        
        itself_mask = DR == 0.0 #This is a mask to de-select the self-pairings i.e. jet1 with jet1
        
        torch_DR    = map_awk_to_torch(remove_self_pairing(DR,itself_mask)) 
        torch_Deta  = map_awk_to_torch(remove_self_pairing(Deta,itself_mask))
        torch_Dphi  = map_awk_to_torch(remove_self_pairing(Dphi,itself_mask))
        torch_M     = map_awk_to_torch(remove_self_pairing(M,itself_mask))
        
        torch_M = torch.clamp(torch_M,0.001) # Make sure that zeros are not included
        
        return torch.cat((torch_Deta,torch_Dphi,torch_DR,torch_M),dim=1)
    
    
    def build_global_attributes(self, input_h5: h5py._hl.group.Group) -> torch.Tensor:
        
        """
        
        """
        return torch.tensor(rf.structured_to_unstructured(input_h5["GLOBAL"][:]))[:].squeeze(1)
    
    
    def build_edge_indices(self) -> torch.tensor:
        
        """
        The combination of nodes corresponding to each possible edge
        
        Returns:
        torch.Tensor of shape (E,2) where E is the sum of N-choose-2 for N in self.Nobjects
        """

        edge_list = []
        for mon in tqdm(self.Nobjects.tolist()):
            edge_index = torch.tensor(
                list(permutations(range(mon), r=2)),
                dtype=torch.int64
            ).permute(dims=(1,0))
            edge_list.append(edge_index)
            
        return torch.cat(edge_list,dim=1)
    
    
    def build_hyperedge_indices(self, hyperedge_cardinality: int) -> torch.tensor:
        
        """
        The combination of nodes corresponding to each possible hyperedge
                
        Parameters:
        - hyperedge_cardinality (int): number of nodes in a hyperedge
        
        Returns:
        torch.Tensor of shape (N,1) for N nodes
        """

        hyperedge_list = []
        # Loop over 
        for mon in tqdm(self.Nobjects.tolist()):
            hyperedge_index = torch.tensor(
                list(combinations(range(mon), r=hyperedge_cardinality)),
                dtype=torch.int64
            ).permute(dims=(1,0))
            hyperedge_list.append(hyperedge_index)
            
        return torch.cat(hyperedge_list,dim=1)
    
    def assign_node_ids(self,node_feature_array: torch.Tensor, labels_h5: h5py._hl.group.Group) -> torch.tensor:
        
        """
        Assign target 1 or 0 to all connection objects (edges / hyperedges separately)
        
        Parameters:
        - node_feature_array (torch.Tensor): x
        - labels_h5 (torch.Tensor): h5file["INPUTS"]
        
        Returns:
        torch.Tensor of shape (N,1) for N nodes
        """
    
        # The first number used is the obj type
        k1 = node_feature_array[:,-1]
        
        # The second number used is the matching number
        truth_label_imported = [labels_h5[obj] for obj in ["JET","LEPTON"]]
        truthlabels = np.concatenate(truth_label_imported,axis=1)
        k2 = torch.tensor(truthlabels[truthlabels!=-99]) #Right now this is hard-coded
        
        cantor_pairing = lambda k1,k2: (k1+k2)*(k1+k2+1)/2 + k2
        
        return cantor_pairing(k1,k2)
    
    def find_matched_edges(self,
                           node_ids: torch.Tensor,
                           connection_indices: torch.Tensor,
                           target_connection_ids: torch.Tensor) -> torch.Tensor:
        
        """
        Assign target 1 or 0 to all connection objects (edges / hyperedges separately)
        
        Parameters:
        - node_ids (torch.Tensor): the IDs of each node in the dataset
        - connection_indices (torch.Tensor): edge_index / hyperedge_index
        - target_connection_ids (torch.Tensor): target_edge_ids / target_hyperedge_ids

        Returns:
        torch.Tensor of shape (N,1) for N connections (edges/hyperedges), with each element 1 or 0
        """
        
        # Broadcasts the node ids into the shape of the connection_indices
        connection_index_id_filled = node_ids[connection_indices]

        # Initialise the output edge labels as all zeros
        output_labels = torch.zeros(connection_index_id_filled.shape[1], 1, dtype=torch.float32)
        
        # Loops over all set of targets, which corresponds to all candidate edges
        for target in target_connection_ids:
            # Compute whether the target indices are in connection_index_id_filled
            eid = torch.isin(connection_index_id_filled,torch.tensor(target))
            # If both feature, it's true
            output_labels += 1.0*torch.all(eid,dim=0).reshape(-1,1)
            
        return output_labels
            
    
    def generate_slices(self):
        
        """
        Computes the index which demarcates the separate events
        (This differs for the different categories)
        
        Acts on self.Nobjects.

        Returns:
        Dictionary of the slices for each category
        """
        
        choose_2 = lambda t: t * (t - 1) // 2
        choose_3 = lambda t: t * (t - 1) * (t - 2) // 6

        slice_x_index         = torch.cat((torch.tensor([0]),torch.cumsum(self.Nobjects, dim=0)))
        slice_u_index         = torch.arange(0,len(self.Nobjects+1))
        slice_edge_index      = torch.cat((torch.tensor([0]),torch.cumsum(2*choose_2(self.Nobjects), dim=0)))
        slice_hyperedge_index = torch.cat((torch.tensor([0]),torch.cumsum(choose_3(self.Nobjects), dim=0)))
        
        return {'x'                 : slice_x_index, 
                'edge_index'        : slice_edge_index, 
                'edge_attr'         : slice_edge_index, 
                'edge_attr_t'       : slice_edge_index,
                'u'                 : slice_u_index ,
                'hyperedge_index'   : slice_hyperedge_index, 
                'hyperedge_attr_t'  : slice_hyperedge_index}
           
        
    def process(self) -> None:
        
        """
        Built-in PyG-InMemoryDataset method
        
        """
        
        # Load the file
        filename = osp.join(self.raw_dir, self.raw_file_names[0])
        print(f"Parsing {filename}")
        with h5py.File(filename,'r') as file:
            num_events = len(file['INPUTS/GLOBAL'])
            print(f"Building HyPERDataset with {num_events} events")
            # Constructing node input tensor
            print("Building node attributes")
            x, self.Nobjects = self.build_node_attributes(file['INPUTS'])
            # Constructing edge input tensor
            print("Building edge attributes")
            edge_attr = self.build_edge_attributes(file['INPUTS'])
            # Constructing global input tensor
            print("Building global attributes")
            u = self.build_global_attributes(file['INPUTS'])
            # Construcrting edge index tensor
            print("Building edge indices")            
            edge_index = self.build_edge_indices()
            # Constructing hyperedge index tensor
            print("Building hyperedge indices")    
            hyperedge_index = self.build_hyperedge_indices(3)
            # Assign unique target ids
            node_ids = self.assign_node_ids(x,file['LABELS'])
            # Constructing edge label tensor
            print("Building edge target labels")    
            edge_attr_t = self.find_matched_edges(node_ids,edge_index,self.target_edge_ids)
            # Constructing hyperedge label tensor
            print("Building hyperedge target labels")    
            hyperedge_attr_t = self.find_matched_edges(node_ids,hyperedge_index,self.target_hyperedge_ids)
            
            slices = self.generate_slices()   
            
        # Create data_dict
        data_dict = {}
        data_dict['edge_attr_t']        = edge_attr_t
        data_dict['hyperedge_attr_t']   = hyperedge_attr_t
        data_dict['x']                  = x 
        data_dict['edge_attr']          = edge_attr
        data_dict['u']                  = u
        data_dict['edge_index']         = edge_index
        data_dict['hyperedge_index']    = hyperedge_index
            
        fs.torch_save((data_dict, slices, Data), self.processed_paths[0])

