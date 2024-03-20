import sys
sys.path.insert(0, "../../")

import argparse
import numpy as np
import pandas as pd

from itertools import combinations
from tqdm.rich import tqdm

from HyPER.data import GraphDataset
from HyPER.evaluation import Evaluate


def argparser():
    parser = argparse.ArgumentParser(description='Reconstruct all hadronic ttbar events from HyPER outputs.')

    parser.add_argument('-o', '--output', type=str, required=True,  help='Output .pkl file.')
    parser.add_argument('-c', '--config', type=str, required=False, help='Configuration/settings file.',
                        default='ttbar_allhad.json')
    parser.add_argument('-d', '--dbconf', type=str, required=False, help='Dataset configuration file.',
                        default='db.yaml')
    parser.add_argument('--log_dir', type=str, required=True, help='Model log directory.')
    parser.add_argument('--test', type=str, required=True, help='Path to the testing dataset.')

    return parser.parse_args()


def Reconstruct_ttbar(HyPER_outputs: str | pd.DataFrame):
    r"""
    `reco_patterns`:
        example: [[[1,1,1],[1,1]], [[1,1,1],[1,1]]]

    """
    if   type(HyPER_outputs) is pd.DataFrame:
        results = HyPER_outputs
    elif type(HyPER_outputs) is str:
        results = pd.read_pickle(HyPER_outputs)
    else:
        raise ValueError(f"Unrecognised HyPER output type {type(HyPER_outputs)}, it must be `str` or `pandas.DataFrame`.")
    
    HyPER_best_top1 = []
    HyPER_best_top2 = []
    HyPER_best_w1 = []
    HyPER_best_w2 = []

    for i in tqdm(range(len(results)), desc="Reconstructing", unit='event'):
        HE_IDX = results['HyPER_HE_IDX'][i]
        HE_RAW = results['HyPER_HE_RAW'][i]
        GE_IDX = results['HyPER_GE_IDX'][i]
        GE_RAW = results['HyPER_GE_RAW'][i]

        selected_HE = []    # Selected HyperEdge
        softProb_HE = []    # Soft probability of the selected HyperEdge
        selected_GE = []    # Selected GraphEdge
        softProb_GE = []    # Soft probability of the selected GraphEdge

        completed_patterns = 0
        rank = np.argsort(HE_RAW)
        p    = -1           # current position

        # We need 2 top quarks
        while completed_patterns < 2:
            if completed_patterns == 0:
                pass
            
            if completed_patterns > 0:
                for pattern in selected_HE:
                    while len(set(pattern).intersection(set(HE_IDX[rank[p]]))) != 0:
                        p -= 1

            selected_HE.append(HE_IDX[rank[p]])
            softProb_HE.append(HE_RAW[rank[p]])
            
            # Find best edge in pattern
            best_edge_score_in_pattern = 0
            for possible_edge in list(combinations(HE_IDX[rank[p]],r=2)):
                # this looks complated but it is faster during computing
                edge_in_pattern = np.argwhere(np.sum(np.where(np.array(GE_IDX)==possible_edge[0],1,0) + np.where(np.array(GE_IDX)==possible_edge[1],1,0),axis=1)==2).flatten()[0]

                if GE_RAW[edge_in_pattern] > best_edge_score_in_pattern:
                    best_edge_score_in_pattern = GE_RAW[edge_in_pattern]
                    best_edge_in_pattern = list(possible_edge)

            selected_GE.append(best_edge_in_pattern)
            softProb_GE.append(best_edge_score_in_pattern)

            p -= 1
            completed_patterns += 1

        HyPER_best_top1.append(selected_HE[0])
        HyPER_best_top2.append(selected_HE[1])
        HyPER_best_w1.append(selected_GE[0])
        HyPER_best_w2.append(selected_GE[1])

    results['HyPER_best_top1'] = HyPER_best_top1
    results['HyPER_best_top2'] = HyPER_best_top2
    results['HyPER_best_w1'] = HyPER_best_w1
    results['HyPER_best_w2'] = HyPER_best_w2

    return results


if __name__ == "__main__":
    args = argparser()

    dataset = GraphDataset(
        path=args.test,
        configs=args.dbconf
    )

    evaluated = Evaluate(
        log_dir=args.log_dir,
        dataset=dataset,
        option_file=args.config
    )

    results = Reconstruct_ttbar(evaluated)

    results.to_pickle(args.output)