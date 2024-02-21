import torch
import warnings

import numpy as np
import awkward as ak

from torch_geometric.data import Data
from torch_hep.graph import GraphBuilder
from itertools import permutations
from pylorentz import Momentum4
from tqdm import tqdm
from HyPER.data import GraphDataset
from HyPER.graphs import BuildHyperedgeTarget, BuildEdgeTarget
from HyPER.utils import Settings
from typing import List


def feature_scaling(src: dict, functions: List[callable]=None) -> dict:
    r"""Implement feature scaling methods to ensure that features are in a suitable range
    for machine learning algorithms, improve model performance, and avoid bias caused by
    different scales of features.

    Examples:
        functions = [lambda x: x/100, lambda x: x/np.pi]

    Args:
        src (dict): a `Python` :obj:`dict` stores feature vectors.
        functions (List[callable]): custom scales set by user. (default: :obj:`None`)

    :rtype: :class:`dict`
    """
    if functions == None:
        warnings.warn("No custom scaling functions provided, feature scaling will not be performed.",UserWarning)
        return src
    else:
        if len(functions) != len(src):
            raise ValueError("Length of the `functions` must equal to the total number of features in `src`.")
        
        custom_idx = 0
        for feat, vec in src.items():
            src[feat] = functions[custom_idx](vec)
            custom_idx += 1
        return src


def cal_inv_mass(node_index_1: int, node_index_2: int,
                 e: np.asarray, eta: np.asarray, phi: np.asarray, pt: np.asarray) -> float:
    r"""Calculate invariant mass of two nodes combine which connected
    by a given edge.

    Args:
        node_index_1 (int): first end of a connecting edge.
        node_index_2 (int): second end of a connecting edge.
        e (List): energy list of nodes.
        eta (List): eta list of nodes.
        phi (List): phi list of nodes.
        pt (List): pt list of nodes.

    :rtype: :class:`float`
    """
    node_1 = Momentum4.e_eta_phi_pt(e[node_index_1],eta[node_index_1],phi[node_index_1],pt[node_index_1])
    node_2 = Momentum4.e_eta_phi_pt(e[node_index_2],eta[node_index_2],phi[node_index_2],pt[node_index_2])
    return (node_1+node_2).m


def BuildGraph(event: ak.highlevel.Record, objects: List[str],
               node_feat: List[str], global_feat: List[str]=None,
               node_scales: List=None, edge_scales: List=None, global_scales: List=None,
               build_target: bool=False, head: str=None, patterns: List=None) -> Data:
    r"""Construct a graph :obj:`torch_geometric.data.Data` using given
    `event` kinematics.

    Note:
        Known physical :obj:`objects` are `jet` (jets), `mu` (muons), `el` (electron)
        and `met` (missing energy). Each object is assigned with a unique encoding.
        :obj:`node_feat`, :obj:`edge_feat` and :obj:`global_feat` are kinematics features associated
        with the physical objects. In the case of `met`, variable $\eta$ is set to 0 and met_pt=met_met.

    Args:
        event (ak.highlevel.Record): input event record.
        objects (List[str]): list of physical objects exist in the graph.
        node_feat (List[str]): list of kinematics associated with the objects
        edge_feat (List[str]): list of variables represents the connections between two objects.
        global_feat (List[str]): list of a graph-wise variables. (default: :obj:`None`)
        scaling_method (str): scaling methods. (default: zscore)
        node_scales (List[callable]): custom scale functions for each node feature. (default: :obj:`None`)
        edge_scales (List[callable]): custom scale functions for each edge feature. (default: :obj:`None`)

    :rtype: :class:`torch_geometric.data.Data`
    """

    nElectrons = 0; nMuons = 0; nJets = 0; has_met = False

    nodes = { feat : np.array([]) for feat in node_feat }

    # check input
    if len(objects) == 0:
        raise ValueError('Least one object must be provided in `objects`. \nKnown physical objects are: `jet`, `el`, `mu`, `met`.')
    if len(node_feat) == 0:
        raise ValueError('Least one features must be provided for `node_feat` and `edge_feat`.')

    if len(set(['pt','eta','e','phi']).intersection(set(node_feat))) == 4:
        pass
    else:
        raise ValueError('No four momentum information provided in the `node_feat`, edges will not be constructed!')

    # get object features
    if 'el' in objects:
        nElectrons = len(event['el_'+node_feat[0]])

        for feat in node_feat:
            try:
                nodes[feat] = np.concatenate((nodes[feat], event['el_'+feat].to_numpy()))
            except:
                nodes[feat] = np.concatenate((nodes[feat], np.full(nElectrons,0)))


    if 'mu' in objects:
        nMuons = len(event['mu_'+node_feat[0]])

        for feat in node_feat:
            try:
                nodes[feat] = np.concatenate((nodes[feat], event['mu_'+feat].to_numpy()))
            except:
                nodes[feat] = np.concatenate((nodes[feat], np.full(nMuons,0)))


    if 'jet' in objects:
        nJets = len(event['jet_'+node_feat[0]])

        for feat in node_feat:
            try:
                nodes[feat] = np.concatenate((nodes[feat], event['jet_'+feat].to_numpy()))
            except:
                nodes[feat] = np.concatenate((nodes[feat], np.full(nJets,0)))


    if 'met' in objects:
        has_met = True

        for feat in node_feat:
            try:
                nodes[feat] = np.concatenate((nodes[feat], event['met_'+feat].to_numpy()))
            except:
                # check if missing energy is asked
                if feat == 'met_e' | feat == 'met_pt':
                    try:
                        nodes[feat] = np.concatenate((nodes[feat], event['met_met'].to_numpy()))
                    except:
                        warnings.warn('`met` is a input object, however, no missing energy variable has been detected, make sure you know what you are doing.', UserWarning)
                        nodes[feat] = np.concatenate((nodes[feat],np.asarray([0])))
                else:
                    nodes[feat] = np.concatenate((nodes[feat],np.asarray([0])))


    num_nodes = (nElectrons + nMuons + nJets) + 1 if has_met==True else (nElectrons + nMuons + nJets)
    encoding = np.concatenate((np.full(nElectrons,-1), np.full(nMuons,-2), np.full(nJets,1), np.asarray([0]))) if has_met==True else np.concatenate((np.full(nElectrons,-1), np.full(nMuons,-2), np.full(nJets,1)))

    if num_nodes < 2:
        raise ValueError('Graph has less than two nodes, check the dataset.')


    edge_index = list(permutations(range(0,num_nodes),2))
    edges = { feat : np.array([]) for feat in ['dPhi','dEta','dR','inv_mass'] }
    for u,v in edge_index:
        d_phi = np.arctan2(np.sin((nodes['phi'][u]-nodes['phi'][v])),
                           np.cos((nodes['phi'][u]-nodes['phi'][v])))
        d_eta = (nodes['eta'][u]-nodes['eta'][v])
        d_r = np.sqrt((d_phi)**2+(d_eta)**2)

        edges['dPhi'] = np.append(edges['dPhi'], d_phi)
        edges['dEta'] = np.append(edges['dEta'], d_eta)
        edges['dR'] = np.append(edges['dR'], d_r)
        edges['inv_mass'] = np.append(edges['inv_mass'], cal_inv_mass(u,v,nodes['e'],nodes['eta'],nodes['phi'],nodes['pt']))


    nodes = feature_scaling(nodes, functions=node_scales)
    if edge_scales is None:
        # use default edge scaling
        edge_scales = feature_scaling(edges, functions=[lambda x: x/np.pi, lambda x: x/np.pi, lambda x: x/np.pi, lambda x: np.log(x)])
    else:
        edge_scales = feature_scaling(edges, functions=edge_scales)

    nodes.update({'encoding': encoding})

    G = GraphBuilder()
    G.add_asNode(key='x_s', **nodes, dtype=torch.float32)
    G.add_asEdge(key='edge_attr_s', index=edge_index, **edges, dtype=torch.float32)
    G.add_asGlobal(key='num_nodes',num_nodes=int(num_nodes),dtype=torch.int64)
    if global_feat != None:
        G.add_asGlobal(key='u_s', **feature_scaling(event[global_feat].tolist(),functions=global_scales),dtype=torch.float32)
    if build_target is True:
        G.add_asNode(key='x_t',targets=BuildHyperedgeTarget(event=event, num_nodes=num_nodes, head=head, patterns=patterns),dtype=torch.float32)
        G.add_asEdge(key='edge_attr_t', edge_target=BuildEdgeTarget(event=event, num_nodes=num_nodes, head=head, patterns=patterns),dtype=torch.float32)
    
    return G.to_Data()


def BuildTorchGraphs(data: ak.highlevel.Array, building_config: Settings, build_target: bool=False) -> GraphDataset:
    r"""Construct graphs and save to a :obj:`GraphDataset`.

    Args:
        data (ak.highlevel.Array): input data, each entry is a single event.
        objects (List[str]): list of physical objects exist in the graph.
        node_feat (List[str]): list of kinematics associated with the objects
        edge_feat (List[str]): list of variables represents the connections between two objects.
        global_feat (List[str]): list of a graph-wise variables. (default: :obj:`None`)
        scaling_method (str): scaling methods. (default: zscale)

    :rtype: :class:`data.GraphDataset`
    """
    graphs = []; indices_allMatched = []
    for idx in tqdm(range(len(data)),desc='Constructing graphs', unit='graph'):

        required_labels = np.unique(ak.flatten(building_config.truth_patterns))
        if len(set(data[idx][building_config.truth_ID]).intersection(required_labels)) == len(required_labels):
            indices_allMatched.append(idx)

        graphs.append(
            BuildGraph( event = data[idx],
                        objects = building_config.objects,
                        node_feat = building_config.node_features,
                        global_feat = building_config.global_features,
                        global_scales = building_config.global_scales,
                        node_scales = building_config.node_scales,
                        edge_scales = None,
                        build_target = build_target,
                        head = building_config.truth_ID,
                        patterns = building_config.truth_patterns )
        )

    return GraphDataset(graphs=graphs, indices_allMatched=indices_allMatched)