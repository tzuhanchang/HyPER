from typing import Callable, List, Optional, Tuple
from warnings import warn
from os import listdir, path as osp

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

class HyPERDataset(InMemoryDataset):
    
    """
    Builds the graph dataset. Inherits from the PyTorchGeometric InMemoryDataset,
        which comes with __init__ and process methods.
        
    Process is called when the data is loaded. 
    It calls the various methods for building the nodes, edge and global parameters
    
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
        # Throw an error if no file matching name exists
        if len(self.names) == 0:
            raise FileNotFoundError(f"No file matching '{self.name}' found in the 'raw' directory.")
        
        file_index = [i for i in range(len(self.names))
                            if self.names[i] == self.name]
        assert len(file_index) == 1
        self.file_index = file_index[0]

        # Parse database config file, setting instance attributes
        parsed_inputs = HyPERDataset._parse_config_file(f"{self.root}/config.yaml")
        
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
    
    
    @staticmethod
    def _parse_config_file(filename):
        
        """Parses YAML config"""
        with open(filename) as stream:
            try:
                return yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
    
    @staticmethod
    def _awkward_nondiag_cartesian(arr: ak.Array) -> ak.Array:
        
        """
        Performs cartesian product on an awkward.Array with itself (arr x arr), 
            dropping the diagonal components
        Parameters:
        - ak.Array
        Returns:
        - ak.Array
        """
        tmp       = ak.cartesian((arr,arr),nested=True)     # Compute the standard cartesian product
        tmp_index = ak.argcartesian((arr,arr),nested=True)  # Compute the cartesian product of the indices
        return tmp[tmp_index["0"]!=tmp_index["1"]]          # Return a filtered object where matching indices are dropped
    
    @staticmethod
    def _map_nested_awkward_to_torch(arr:ak.Array) -> torch.tensor:
        
        """
        Takes an ak.Array which is singly-ragged and converts to torch.tensor column required
        [Four instances of this use within the build_edge_indices and build_hyperedge_indices methods]
        """
        # Flatten the array into 1D
        flat = ak.flatten(arr)
        # Convert the result to an unstructured numpy array of shape 
        unstruct_numpy_array = rf.structured_to_unstructured(flat.to_numpy())
        # Convert the numpt array to a tensor of shape Nx1, where .t() takes the transpose
        return torch.as_tensor(unstruct_numpy_array).t()      
    
    
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
    
    def assign_node_ids(self,node_feature_array: torch.Tensor, labels_h5: h5py._hl.group.Group):
        
        """
        Creates awkward arrays storing to the local index and cantor index of the nodes, split by event e.g.
            self.local_node_ids  = ak.Array[[0,1,2,3],[0,1]] - This is just a ragged array of integers from 0
            self.cantor_node_ids = ak.Array[[4,8,13,26],[8,13]] - Ragged array of cantor IDs
            
        local_node_ids are used for building the (hyper)edge indices 
        cantor_node_ids are used for building the (hyper)edge targets
                    
        Parameters:
        - node_feature_array (torch.Tensor): x
        - labels_h5 (torch.Tensor): h5file["LABELS"]
        
        Comments:
        - This could probably be streamlined by using numpy intermediary
        """
    
        # The first number used is the obj type e.g. 1 for jet, 2 for lepton, etc.
        k1 = node_feature_array[:,-1]
        
        # The second number used is the matching number
        # Extract the matching index for each object
        truth_label_imported = [labels_h5[obj] for obj in self.input_id.keys()]
        # Combine into one numpy array
        truthlabels_np = np.concatenate(truth_label_imported,axis=1)
        # Convert to torch tensor
        truthlabels = torch.tensor(truthlabels_np)

        # Remove padded events (padded events have to be hard-coded with np.nan)
        remove_nan_mask = ~torch.isnan(truthlabels)
        # Apply the mask - 
        # this changes the shape from [Nevents,Nnodes] -> [total # nodes in dataset]
        k2 = truthlabels[remove_nan_mask]
        
        # Use the cantor pairing function to assign a unique ID as a tensor
        cantor_pairing = lambda k1,k2: (k1+k2)*(k1+k2+1)/2 + k2
        cantor_node_ids_tensor = cantor_pairing(k1,k2) 
        
        # Re-cast this as an awkward array split by event
        self.cantor_node_ids   = ak.unflatten(
            ak.Array(np.asarray(cantor_node_ids_tensor)),
            ak.Array(np.asarray(self.Nobjects))
        )
        
        # Equivalent integer index for each node in each event
        self.local_node_ids  =   ak.local_index(self.cantor_node_ids)  
    
    def build_edge_indices(self) -> torch.tensor:
        
        """
        The combination of nodes corresponding to each possible edge
        
        Returns:
        torch.Tensor of shape (E,2) where E is the sum of N-choose-2 for N in self.Nobjects
        """
        # Compute all edge_pairs using awkward cartesian operation
        edge_pairs = HyPERDataset._awkward_nondiag_cartesian(self.local_node_ids)
        # Convert to singly-ragged ak.Array 
        edge_pairs_1flat = ak.flatten(edge_pairs)
        # Convert to required torch.tensor
        self.edge_index = HyPERDataset._map_nested_awkward_to_torch(edge_pairs_1flat)
        
        # Compute all cantor ID edge pair combinations using awkward cartesian operation
        cantor_edge_pairs = HyPERDataset._awkward_nondiag_cartesian(self.cantor_node_ids)
        # Convert to singly-ragged ak.Array
        cantor_edge_pairs_1flat = ak.flatten(cantor_edge_pairs)
        # Convert to required torch.tensor
        self.cantor_edge_index = HyPERDataset._map_nested_awkward_to_torch(cantor_edge_pairs_1flat)

    def build_hyperedge_indices(self, hyperedge_cardinality: int) -> torch.tensor:
        
        """
        Computes the combination of nodes corresponding to each possible hyperedge, per event.
        Uses awkward operatiosn to perform N-choose-H, with:
        - N = multiplcity of each event
        - H = hyperedge_cardinality
                
        Parameters:
        - hyperedge_cardinality (int): number of nodes in a hyperedge
        
        Returns:
        torch.Tensor of shape (N,1) for N nodes
        """
        # Perform N-choose-H for local node IDs
        hyperedge_node_index_combinations = ak.combinations(self.local_node_ids, hyperedge_cardinality)
        # Convert to torch.tensor
        self.hyperedge_index = HyPERDataset._map_nested_awkward_to_torch(hyperedge_node_index_combinations)
    
        # Perform N-choose-H for Cantor node IDs
        hyperedge_cantor_index_combinations = ak.combinations(self.cantor_node_ids, hyperedge_cardinality)
        # Convert to torch.tensor
        self.cantor_hyperedge_index = HyPERDataset._map_nested_awkward_to_torch(hyperedge_cantor_index_combinations)
        
    def find_matched_connections(self,
                                 connection_input_cantor_tensor: torch.Tensor,
                                 target_connection_ids: torch.Tensor) -> torch.Tensor:
        
        """
        Assign target 1 or 0 to all connection objects (edges / hyperedges separately)
        
        Parameters:
        - connection_input_cantor_tensor (torch.Tensor): tensor containing the cantor indices of all edges or hyperedges
        - target_connection_ids (torch.Tensor): target_edge_ids / target_hyperedge_ids

        Returns:
        torch.Tensor of shape (N,1) for N connections (edges/hyperedges), with each element 1 or 0
        """

        # Initialise the output edge labels as all zeros
        output_labels = torch.zeros(connection_input_cantor_tensor.shape[1], 1, dtype=torch.float32)
        
        # Loops over all set of targets, which corresponds to all candidate edges
        for target in target_connection_ids:
            # Compute whether the target indices are in connection_input_cantor_tensor
            eid = torch.isin(connection_input_cantor_tensor,torch.tensor(target))
            # If both feature, it's true
            output_labels += 1.0*torch.all(eid,dim=0).reshape(-1,1)
            
        return output_labels
            
    def generate_slices(self):
        
        """
        Computes the index which demarcates separate events
        (This differs for the different categories)
        
        Acts on self.Nobjects.

        Returns:
        Dictionary of the slices for each category
        """
        
        choose_2 = lambda t: t * (t - 1) // 2
        choose_3 = lambda t: t * (t - 1) * (t - 2) // 6
        choose_4 = lambda t: t * (t - 1) * (t - 2) * (t - 3) // 24

        slice_x_index         = torch.cat((torch.tensor([0]),torch.cumsum(self.Nobjects, dim=0)))
        slice_u_index         = torch.arange(0,len(self.Nobjects)+1)
        slice_edge_index      = torch.cat((torch.tensor([0]),torch.cumsum(2*choose_2(self.Nobjects), dim=0)))
        if self.hyperedge_order == 3:
            slice_hyperedge_index = torch.cat((torch.tensor([0]),torch.cumsum(choose_3(self.Nobjects), dim=0)))
        elif self.hyperedge_order == 4:
            slice_hyperedge_index = torch.cat((torch.tensor([0]),torch.cumsum(choose_4(self.Nobjects), dim=0)))
        
        return {'x'                 : slice_x_index, 
                'edge_index'        : slice_edge_index, 
                'edge_attr'         : slice_edge_index, 
                'edge_attr_t'       : slice_edge_index,
                'u'                 : slice_u_index ,
                'hyperedge_index'   : slice_hyperedge_index, 
                'hyperedge_attr_t'  : slice_hyperedge_index}
           
        
    def process(self) -> None:
        
        """
        Built-in PyG-InMemoryDataset method to generate dataset        
        """
        
        # Load the file
        filename = osp.join(self.raw_dir, self.raw_file_names[0])
        print(f"Parsing {filename}")
        with h5py.File(filename,'r') as file:

            num_events = len(file['INPUTS']["GLOBAL"])
            print(f"Building HyPERDataset with {num_events} events")
            
            # Node,edge,global attribute tensor creation
            # Constructing node input tensor
            print("Building node attributes")
            x, self.Nobjects = self.build_node_attributes(file['INPUTS'])
            # Constructing edge input tensor
            print("Building edge attributes")
            edge_attr = self.build_edge_attributes(file['INPUTS'])
            # Constructing global input tensor
            print("Building global attributes")
            u = self.build_global_attributes(file['INPUTS'])
            
            # Assign the local and Cantor node IDs
            self.node_ids = self.assign_node_ids(x,file['LABELS'])
            
        # Construcrting edge index tensor
        print("Building edge indices")            
        self.build_edge_indices()
        # Constructing hyperedge index tensor
        print("Building hyperedge indices")    
        self.build_hyperedge_indices(hyperedge_cardinality=self.hyperedge_order)
        # Assign unique target ids
        # Constructing edge label tensor
        print("Building edge target labels")    
        edge_attr_t = self.find_matched_connections(self.cantor_edge_index,self.target_edge_ids)
        # Constructing hyperedge label tensor
        print("Building hyperedge target labels")    
        hyperedge_attr_t = self.find_matched_connections(self.cantor_hyperedge_index,self.target_hyperedge_ids)
        
        slices = self.generate_slices()   
            
        # Create data_dict
        PyGDataObject = Data(x              = x,
                            edge_attr       = edge_attr,
                            u               = u,
                            edge_index      = self.edge_index,
                            hyperedge_index = self.hyperedge_index,
                            edge_attr_t     = edge_attr_t,
                            hyperedge_attr_t= hyperedge_attr_t)
    
        # Apply the transforms if required
        if self.pre_transform is not None:
            print("Transforming inputs")
            PyGDataObject = self.pre_transform(PyGDataObject)
            
        fs.torch_save((PyGDataObject.to_dict(), slices, Data), self.processed_paths[0])

