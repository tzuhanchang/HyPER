import numpy as np
import pandas as pd

from itertools import combinations
from tqdm.rich import tqdm

def ttbar_single_lep(HyPER_outputs: str | pd.DataFrame):
    r"""Reconstruct ttbar events with lepton+jets final states.

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

    HyPER_best_top1_prob = []
    HyPER_best_top2_prob = []
    HyPER_best_w1_prob = []
    HyPER_best_w2_prob = []
    

    for i in tqdm(range(len(results)), desc="Reconstructing", unit='event'):
        HE_IDX = results['HyPER_HE_IDX'][i]
        HE_RAW = results['HyPER_HE_RAW'][i]
        GE_IDX = results['HyPER_GE_IDX'][i]
        GE_RAW = results['HyPER_GE_RAW'][i]

        HE_VCT = results['HyPER_HE_VCT'][i]
        GE_VCT = results['HyPER_GE_VCT'][i]

        selected_HE = []    # Selected HyperEdge
        softProb_HE = []    # Soft probability of the selected HyperEdge
        selected_GE = []    # Selected GraphEdge
        softProb_GE = []    # Soft probability of the selected GraphEdge

        completed_patterns = 0
        tlep_position = 0 #was leptonic top reconstructed first (0) or not (1)
        rank = np.argsort(HE_RAW) #array of sorted indices
        p    = -1           # current position

        # We need 2 top quarks
        while completed_patterns < 2:
            if completed_patterns == 0: 
                #Make sure its not reconstructing MET and lepton in different tops
                while (len(set(HE_VCT[rank[p]]).intersection(set([2,3]))) == 1): 
                    p -=1 

                #check if best top is tlep (!!! Here we assume there is only one charged lepton per event in the tuples !!!)
                if (len(set(HE_VCT[rank[p]]).intersection(set([2,3]))) != 2):
                    tlep_position = 1
            
            if completed_patterns > 0:
                for pattern in selected_HE:
                    #Check that the other reconstructed top does not contain the same nodes and is/isnt leptonic
                    while (len(set(pattern).intersection(set(HE_IDX[rank[p]]))) != 0) or ((tlep_position == 1) and (len(set(HE_VCT[rank[p]]).intersection(set([2,3]))) != 2)): 
                        p -= 1

            if HE_RAW[rank[p]] != 0:
                selected_HE.append(HE_IDX[rank[p]])
                softProb_HE.append(HE_RAW[rank[p]])
            else:
                selected_HE.append([-1, -1, -1])
                softProb_HE.append(0)
            
            # Find best edge in pattern
            best_edge_score_in_pattern = 0
            for possible_edge in list(combinations(HE_IDX[rank[p]],r=2)):
                # this looks complicated but it is faster during computing
                edge_in_pattern = np.argwhere(np.sum(np.where(np.array(GE_IDX)==possible_edge[0],1,0) + np.where(np.array(GE_IDX)==possible_edge[1],1,0),axis=1)==2).flatten()[0]

                if GE_RAW[edge_in_pattern] > best_edge_score_in_pattern:
                    if (completed_patterns != tlep_position) or ((completed_patterns == tlep_position) and (len(set(GE_VCT[edge_in_pattern]).intersection(set([2,3]))) == 2)): #Wlep automatically found as constituted of lepton+MET
                        best_edge_score_in_pattern = GE_RAW[edge_in_pattern]
                        best_edge_in_pattern = list(possible_edge)


            if best_edge_score_in_pattern != 0:
                selected_GE.append(best_edge_in_pattern)
                softProb_GE.append(best_edge_score_in_pattern)
            else:
                selected_GE.append([-1,-1])
                softProb_GE.append(0)

            p -= 1
            completed_patterns += 1

        if tlep_position == 0:
            HyPER_best_top1.append(selected_HE[0])
            HyPER_best_top2.append(selected_HE[1])
            HyPER_best_w1.append(selected_GE[0])
            HyPER_best_w2.append(selected_GE[1])

            HyPER_best_top1_prob.append(softProb_HE[0])
            HyPER_best_top2_prob.append(softProb_HE[1])
            HyPER_best_w1_prob.append(softProb_GE[0])
            HyPER_best_w2_prob.append(softProb_GE[1])
        else:
            HyPER_best_top1.append(selected_HE[1])
            HyPER_best_top2.append(selected_HE[0])
            HyPER_best_w1.append(selected_GE[1])
            HyPER_best_w2.append(selected_GE[0])

            HyPER_best_top1_prob.append(softProb_HE[1])
            HyPER_best_top2_prob.append(softProb_HE[0])
            HyPER_best_w1_prob.append(softProb_GE[1])
            HyPER_best_w2_prob.append(softProb_GE[0])

    results['HyPER_best_top1'] = HyPER_best_top1
    results['HyPER_best_top2'] = HyPER_best_top2
    results['HyPER_best_w1'] = HyPER_best_w1
    results['HyPER_best_w2'] = HyPER_best_w2

    results['HyPER_best_top1_prob'] = HyPER_best_top1_prob
    results['HyPER_best_top2_prob'] = HyPER_best_top2_prob
    results['HyPER_best_w1_prob'] = HyPER_best_w1_prob
    results['HyPER_best_w2_prob'] = HyPER_best_w2_prob

    return results

def ttbar_allhad(HyPER_outputs: str | pd.DataFrame):
    r"""Reconstruct ttbar events with all-hadronic final states.

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
                # this looks complicated but it is faster during computing
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

def ttbar_single_lep(HyPER_outputs: str | pd.DataFrame):
    r"""Reconstruct ttbar events with single lepton + jets final state.

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
    thad_first  = []    #to remember if thad or tlep was reconstructed first (i.e. had the best score)


    for i in tqdm(range(len(results)), desc="Reconstructing", unit='event'):
        HE_IDX = results['HyPER_HE_IDX'][i]
        HE_RAW = results['HyPER_HE_RAW'][i]
        GE_IDX = results['HyPER_GE_IDX'][i]
        GE_RAW = results['HyPER_GE_RAW'][i]

        HE_VCT = results['HyPER_HE_VCT'][i]
        GE_VCT = results['HyPER_GE_VCT'][i]

        selected_HE = []    # Selected HyperEdge
        softProb_HE = []    # Soft probability of the selected HyperEdge
        selected_GE = []    # Selected GraphEdge
        softProb_GE = []    # Soft probability of the selected GraphEdge

        completed_patterns = 0
        tlep_position = 0 # was leptonic top reconstructed first (0) or not (1)
        rank = np.argsort(HE_RAW) #array of sorted indices
        p    = -1           # current position

        # We need 2 top quarks
        while completed_patterns < 2:
            if completed_patterns == 0: 
                # Make sure its not reconstructing MET and lepton in different tops
                while (len(set(HE_VCT[rank[p]]).intersection(set([2,3,4]))) == 1): 
                    p -=1 

                # Check if best top is tlep (!!! Here we assume there is only one charged lepton per event in the input data !!!)
                if (len(set(HE_VCT[rank[p]]).intersection(set([2,4]))) != 2) and (len(set(HE_VCT[rank[p]]).intersection(set([3,4]))) != 2):
                    tlep_position = 1
            
            if completed_patterns > 0:
                for pattern in selected_HE:
                    # Check that the other reconstructed top does not contain the same nodes and is/isnt leptonic
                    while (len(set(pattern).intersection(set(HE_IDX[rank[p]]))) != 0) or ((tlep_position == 1) and (len(set(HE_VCT[rank[p]]).intersection(set([2,4]))) != 2) and (len(set(HE_VCT[rank[p]]).intersection(set([3,4]))) != 2)): 
                        p -= 1

            selected_HE.append(HE_IDX[rank[p]])
            softProb_HE.append(HE_RAW[rank[p]])
            
            # Find best edge in pattern
            best_edge_score_in_pattern = 0
            for possible_edge in list(combinations(HE_IDX[rank[p]],r=2)):
                # this looks complated but it is faster during computing
                edge_in_pattern = np.argwhere(np.sum(np.where(np.array(GE_IDX)==possible_edge[0],1,0) + np.where(np.array(GE_IDX)==possible_edge[1],1,0),axis=1)==2).flatten()[0]

                if GE_RAW[edge_in_pattern] > best_edge_score_in_pattern:
                    if (completed_patterns != tlep_position) or ((completed_patterns == tlep_position) and (len(set(GE_VCT[edge_in_pattern]).intersection(set([2,3,4]))) == 2)): #Wlep automatically found as constituted of lepton+MET
                        best_edge_score_in_pattern = GE_RAW[edge_in_pattern]
                        best_edge_in_pattern = list(possible_edge)

            selected_GE.append(best_edge_in_pattern)
            softProb_GE.append(best_edge_score_in_pattern)

            p -= 1
            completed_patterns += 1

        if tlep_position == 0:
            HyPER_best_top1.append(selected_HE[0])
            HyPER_best_top2.append(selected_HE[1])
            HyPER_best_w1.append(selected_GE[0])
            HyPER_best_w2.append(selected_GE[1])
        else:
            HyPER_best_top1.append(selected_HE[1])
            HyPER_best_top2.append(selected_HE[0])
            HyPER_best_w1.append(selected_GE[1])
            HyPER_best_w2.append(selected_GE[0])

        thad_first.append(tlep_position)

    results['thad_first'] = thad_first

    results['HyPER_best_top1'] = HyPER_best_top1
    results['HyPER_best_top2'] = HyPER_best_top2
    results['HyPER_best_w1'] = HyPER_best_w1
    results['HyPER_best_w2'] = HyPER_best_w2

    return results
