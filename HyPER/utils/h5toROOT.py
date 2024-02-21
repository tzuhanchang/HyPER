import h5py
import torch
import uproot 

import numpy as np
import awkward as ak

from torch_geometric.utils import unbatch


def _extract(values: np.ndarray, paddings: np.ndarray):
    return [ x.tolist() for x in unbatch(torch.tensor(values[paddings]),torch.tensor(np.transpose(paddings)[:,0]))]


def h5toROOT(input: str, output: str):
    r"""Convert a generic .h5 file to ROOT. This is built to use the datasets 
    provided by the topograh network presented in 'Topological Reconstruction 
    of Particle Physics Processes using Graph Neural Networks' (arXiv:2303.13937).

    Dataset structure:
        -'delphes'
        --> 'jets', 'jets_indices', 'matchability', 'nbjets', 'njets', 'partons'

    'jets': pt, eta, phi, energy, is_tagged.
    'jets_indices': from 0 to 5: b1 W1j1 W1j2 b2 W2j1 W2j2 (1= from top, 2=from antitop).

    Args:
        file (str): .h5 file.
    """
    with h5py.File(input, mode='r') as f:
        jets = f['delphes']['jets'][()]
        nonpadded = np.nonzero(jets['pt'])

        pt  = _extract(jets['pt'], nonpadded)
        eta = _extract(jets['eta'], nonpadded)
        phi = _extract(jets['phi'], nonpadded)
        e   = _extract(jets['energy'], nonpadded)
        bTag = [[int(s) for s in sublist] for sublist in _extract(jets['is_tagged'], nonpadded)]
        truthmatch = _extract(f['delphes']['jets_indices'][()], nonpadded)

        with uproot.recreate(output) as r:
            r['delphes'] = {"jet": ak.zip({"pt": pt, "eta": eta, "phi": phi, "e": e, "bTag": bTag, "truthmatch": truthmatch}),
                            "nbTagged": f['delphes']['nbjets'][()]}
