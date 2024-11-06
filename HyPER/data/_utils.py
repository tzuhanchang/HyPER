import h5py
import warnings
import numpy as np
import pandas as pd

from typing import Optional
from omegaconf import DictConfig


def _get_standard_scales(path, node_inputs) -> dict:
    standard_scales = {
        "_mean_nodes": None,
        "_std_nodes": None,
        "_mean_glob": None,
        "_std_glob": None
    }
    with h5py.File(path, 'r') as file:
        nodes = pd.DataFrame(np.concatenate([np.concatenate(np.array(file['INPUTS'][obj])) for obj in node_inputs]))
        standard_scales['_mean_nodes'] = nodes.mean().values
        standard_scales['_std_nodes']  = nodes.std().values

        glob = pd.DataFrame(np.concatenate(np.array(file['INPUTS']["GLOBAL"])))
        standard_scales['_mean_glob'] = glob.mean().values
        standard_scales['_std_glob']  = glob.std().values
    return standard_scales


def _check_dataset(path: str, config: DictConfig, mode: Optional[str] = 'train') -> dict:
    r"""Run basic checks on the provided dataset, and return dataset parameters.

    Args:
        path (str): path to the dataset.
    """
    dataset_parm = {
        "dataset_size": None,
        "node_inputs": config['Dataset']['node_inputs'],
        "node_features": config['Dataset']['node_features'],
        "global_features": config['Dataset']['global_features'],
        "_objPadding": {},
        "_objLabel": {},
        "_train_mode": True if mode.lower() == 'train' else False,
        "boolean_filter": config['Dataset']['boolean_filter'],
        "index_select": None,
        "_use_index_select": False,
        "_use_EEtaPhiPt": False,
        "_use_EPxPyPz": False,
        "edge_identifiers": None,
        "hyperedge_identifiers": None,
        "hyperedge_order": None,
        "n_node_features": None,
        "n_edge_features": None,
        "n_global_features": None,
        "_node_norms": config['Dataset']['node_norms'],
        "_global_norms": config['Dataset']['global_norms']
    }

    with h5py.File(path, 'r') as file:
        assert 'INPUTS' in file.keys(), "Data group `INPUTS` must be provided."
        assert 'GLOBAL' in list(file['INPUTS'].keys()), "`INPUTS/GLOBAL` is not found."
        dataset_parm['dataset_size'] = len(file['INPUTS']['GLOBAL'])
        obj_count = 1
        for node_input in dataset_parm['node_inputs']:
            assert node_input in list(file['INPUTS'].keys()), f"`INPUTS/{node_input}` is not found."
            assert dataset_parm['dataset_size'] == len(file['INPUTS'][node_input]), f"INPUTS/{node_input} does not match with total number of events {dataset_parm['dataset_size']}."
            dataset_parm['_objPadding'].update({node_input: len(file['INPUTS'][node_input][0])})
            dataset_parm['_objLabel'].update({node_input: obj_count})  # Assign an unique ID for different types of objects
            obj_count += 1

        if dataset_parm['_train_mode']:
            assert 'LABELS' in file.keys(), "Data group `LABELS` must be provided in `train` mode."
            assert 'ID' in list(file['LABELS'].keys()), "`LABELS/ID` is not found."
            assert dataset_parm['dataset_size'] == len(file['LABELS']['ID']), f"`LABELS/ID` does not match with total number of events {dataset_parm['dataset_size']}."

        if dataset_parm['boolean_filter'] is not None:
            assert dataset_parm['dataset_size'] == len(file[dataset_parm['boolean_filter']]), f"`{dataset_parm['boolean_filter']}` does not match with total number of events {dataset_parm['dataset_size']}."
            dataset_parm['index_select'] = np.array(range(dataset_parm['dataset_size']))[np.array(file[dataset_parm['boolean_filter']],dtype=np.int64)==1]
            dataset_parm['_use_index_select'] = True
            dataset_parm['dataset_size'] = len(dataset_parm['index_select'])

        assert dataset_parm['node_features'] == list(file['INPUTS'][dataset_parm['node_inputs'][0]].dtype.fields.keys()), \
            f"`node_features` defined in the configuration file ({dataset_parm['node_features']}) do not matches with the ones in the dataset: {list(file['INPUTS'][dataset_parm['node_inputs'][0]].dtype.fields.keys())}"

        assert dataset_parm['global_features'] == list(file['INPUTS']['GLOBAL'].dtype.fields.keys()), \
            f"`global_features` defined in the configuration file ({dataset_parm['global_features']}) do not matches with the ones in the dataset: {list(file['INPUTS']['GLOBAL'].dtype.fields.keys())}"

        if dataset_parm['node_features'][:4] == ['e','eta','phi','pt']:
            dataset_parm['_use_EEtaPhiPt'] = True
        elif dataset_parm['node_features'][:4] == ['e','px','py','pz']:
            dataset_parm['_use_EPxPyPz'] = True
        else:
            warnings.warn("You are not using the standard feature ordering or the naming scheme: ['e', 'eta', 'phi', 'pt'] or ['e', 'px', 'py', 'pz'] (for the first 4 features). This might cause problems in the edge construction stage.", UserWarning)
            dataset_parm['_use_EPxPyPz'] = True

        dataset_parm['edge_identifiers'] = np.apply_along_axis(
            lambda x: 2**x, axis=0, arr=list(config['Dataset']['edge_target'])
        ).sum(axis=1)
        dataset_parm['hyperedge_identifiers'] = np.apply_along_axis(
            lambda x: 2**x, axis=0, arr=list(config['Dataset']['hyperedge_target'])
        ).sum(axis=1)

        try:
            HE_ids = np.array(list(config['Dataset']['hyperedge_target']))
            dataset_parm['hyperedge_order'] = HE_ids.shape[1]
        except ValueError:
            print("HyPER currently only support hyperedges with the same order.")

        dataset_parm['n_node_features'] = len(dataset_parm['node_features']) + 1
        dataset_parm['n_edge_features'] = 4
        dataset_parm['n_global_features'] = len(file['INPUTS']["GLOBAL"][0][0])

        if dataset_parm['_node_norms'] is not None:
            assert len(dataset_parm['_node_norms']) == len(dataset_parm['node_features']), "For each node features, a normalisation method must be provided in `node_norms`."
        else:
            dataset_parm['_node_norms'] = ['non']*len(dataset_parm['node_features'])

        if dataset_parm['_global_norms'] is not None:
            assert len(dataset_parm['_global_norms']) == dataset_parm['n_global_features'], "For each global features, a normalisation method must be provided in `global_norms`."
        else:
            dataset_parm['_global_norms'] = ['non']*dataset_parm['n_global_features']

    # Get standard scales
    dataset_parm.update(_get_standard_scales(path, dataset_parm['node_inputs']))
    
    return dataset_parm