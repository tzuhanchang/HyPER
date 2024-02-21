import numpy as np
import awkward as ak

from itertools import combinations, permutations
from typing import List


def BuildHyperedgeTarget(event: ak.highlevel.Record, num_nodes: int, head: str, patterns: List) -> List:
    r"""Construct a target graph according to `patterns` appear in the `head`.

    Examples:
        In the case of all hadronic ttbar, `patterns` are the truth matching results
        which defines the jet-triplets orginated from the two top-quarks. `head` is the
        variable jet_truthmatch commonly found in ATLAS datasets.

        `patterns` is a nested :obj:`List`:
            [ [top1_jet1, top1_jet2, top1_jet3], [top2_jet1, top2_jet2, top2_jet3] ]
        sub-lists should have the same size.

    Args:
        event (ak.highlevel.Record): input event record.
        head (str): name of the variable provides the truth patterns.
        patterns (List): truth matching partterns.

    :rtype: :class:`List`    
    """
    # Only Hyperedge size large than two is considered, otherwise use edge
    patterns = [pattern for pattern in patterns if len(pattern) > 2]

    k = len(patterns[0])

    # searching for `patterns` from the `head`
    target_multiplets = []
    for i in range(len(patterns)):
        tmp = []
        for j in range(len(patterns[i])):
            if k != len(patterns[i]):
                raise ValueError("Contraction requires all hyperedges to have the same number of nodes. Check the `patterns` provided.")
            try:
                tmp.append(np.argwhere(np.array(event[head])==patterns[i][j]).flatten()[0])
            except:
                tmp.append(-9)

        target_multiplets.append(tmp)

    # finding target Hyperedges
    node_combinations = list(combinations(range(num_nodes),k))

    targets = []
    for combination in node_combinations:
        found = False
        for multiplet in target_multiplets:
            if len(set(combination).intersection(set(multiplet))) == k:
                found = True
                break
            else:
                continue

        if found is True:
            targets.append(1)
        else:
            targets.append(0)

    return targets


def BuildEdgeTarget(event: ak.highlevel.Record, num_nodes: int, head: str, patterns: List) -> List:
    r"""Construct target edges according given patterns.

    Args:
        event (ak.highlevel.Record): input event record.
        head (str): name of the variable provides the truth patterns.
        patterns (List): truth matching partterns.

    :rtype: :class:`List`   
    """
    # check pattern shape
    patterns = [pattern for pattern in patterns if len(pattern)==2]
    if len(patterns) == 0:
        raise ValueError("No edge patterns are provided!")
    
    # searching for `patterns` from the `head`
    target_edges = []
    for i in range(len(patterns)):
        tmp = []
        for j in range(2):
            try:
                tmp.append(np.argwhere(np.array(event[head])==patterns[i][j]).flatten()[0])
            except:
                tmp.append(-9)

        target_edges.append(tmp)

    # finding target edges
    edge_index = list(permutations(range(0,num_nodes),2))
    
    targets = np.zeros(len(edge_index))
    for i in range(len(edge_index)):
        for target in target_edges:
            if len(set(edge_index[i]).intersection(set(target))) == 2:
                targets[i] = 1
            else:
                continue

    return targets.tolist()