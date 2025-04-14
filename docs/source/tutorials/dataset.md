Build Your Own Dataset
=======================

High energy physics data is commonly stored in [ROOT](https://root.cern) files, which extension is `.root`. In this tutorial, you will learn the dataset structure of HyPER and we will show you how to build your own `GraphDataset` from ROOT files. We are going to use semi-leptonic ttbar events as an example for this tutorial. In ttbar semileptonic events one top decays hadronically, producing one b-jet and two ligh/charm jets; and the other top decays leptonically, producing one b-jet, one lepton and one neutrino.

Dataset Structure
-----------

HyPER uses `HDF5` to store data, it contains two data groups: `INPUTS` and `LABELS`. The structure of our dataset is as follows:
``` yaml
|- INPUTS           |- LABELS
|--- jet            |--- VertexID
|--- electron       |--- IndexSelect
|--- muon
|--- global
|--- ...
```
Datasets: The `jet`, `electron` and `muon` datasets have the same size along `dim=1`, which are the "features" of these objects.
The `global` dataset stores event-wise information.
Each final state is assigned with a unique integer that is used for the identification of target edges and hyperedges, they are stored in the `VertexID` dataset.
An optional `IndexSelect` dataset can be added, it contains boolean values representing if an event is fully matched.

!!! Note

    Some features are only available for certain objects, such as "charge" for leptons (but not for jets), "btag" for jets (but not for leptons). We recommend using `0` as a placeholder for the missing features in such case.


Each dataset has a corresponding configuration file `db.yaml`, it tells HyPER how to interpertate the data:
```yaml
INPUTS:
  Objects:  ['jet', 'electron', 'muon']

  Features:
    e:      log
    eta:    none
    phi:    none
    pt:     log
    btag:   none
    charge: none
    id:     none    # Do NOT use 0 for object ID

  global:
    njet:      none
    nbTagged:  none


LABELS:
  # These integers are `VertexID` defined in the dataset
  Edges:
    w1: [2,3]
    w2: [5,6]

  Hyperedges:
    t1: [1,2,3]
    t2: [4,5,6]
```


Example Script
-----------

The following is an example script for building a semi-leptonic ttbar dataset.

``` python
import h5py
import uproot
import argparse
import numpy as np
from tqdm import tqdm


def argparser():
    parser = argparse.ArgumentParser(description='Make a HyPER dataset')
    parser.add_argument('-f', '--input',  type=str, required=True, help='ROOT input file.')
    parser.add_argument('-o', '--output', type=str, required=True, help='Output .h5 file')
    return parser.parse_args()


def MakeDataset(input: str, output: str):
    r"""A example for constructing `HyPER` dataset.

    Args:
        input (str): input ROOT file.
        output (str): output HyPER dataset in .h5 format.
    """
    # Max number of objects (padding)
    pad_to_jet = 20
    pad_to_el  = 2
    pad_to_mu  = 2

    # Load the ROOT file
    root_file = uproot.open(input)
    tree = root_file['Delphes']
    num_entries = tree.num_entries

    # ---------------------------------------------------
    #                 Load data from ROOT
    # ---------------------------------------------------
    jet_bTag  = tree["jet_bTag"].array(library='np')
    jet_eta   = tree["jet_eta"].array(library='np')
    jet_phi   = tree["jet_phi"].array(library='np')
    jet_e     = tree["jet_e"].array(library='np')
    jet_pt    = tree["jet_pt"].array(library='np')
    jet_match = tree["jet_truthmatch"].array(library='np')
    el_eta    = tree["electron_eta"].array(library='np')
    el_phi    = tree["electron_phi"].array(library='np')
    el_e      = tree["electron_e"].array(library='np')
    el_pt     = tree["electron_pt"].array(library='np')
    el_charge = tree["electron_charge"].array(library='np')
    el_match  = tree["el_truthmatch"].array(library='np')
    mu_eta    = tree["muon_eta"].array(library='np')
    mu_phi    = tree["muon_phi"].array(library='np')
    mu_e      = tree["muon_e"].array(library='np')
    mu_pt     = tree["muon_pt"].array(library='np')
    mu_charge = tree["muon_charge"].array(library='np')
    mu_match  = tree["muon_truthmatch"].array(library='np')
    # ......
    njet      = tree["njet"].array(library='np')
    nbTagged  = tree["nbTagged"].array(library='np')
    # ......
    # ---------------------------------------------------


    # ---------------------------------------------------
    #               Define Input Datatype
    # ---------------------------------------------------
    node_dt = np.dtype([('e', np.float32), ('eta', np.float32), ('phi', np.float32), ('pt', np.float32), ('btag', np.float32), ('id', np.float32)])
    jet_data = np.zeros((num_entries, pad_to_jet), dtype=node_dt)
    el_data = np.zeros((num_entries, pad_to_el), dtype=node_dt)
    mu_data = np.zeros((num_entries, pad_to_mu), dtype=node_dt)
    # ......
    global_dt = np.dtype([('njet', np.float32), ('nbTagged', np.float32)])
    global_data = np.zeros((num_entries, 1), dtype=global_dt)

    # We need `VertexID` to label truth egdes and hyperedges
    VertexID_data = np.full((num_entries, pad_to_jet+pad_to_el+pad_to_mu), -9) # use -9 for none filled values

    # ---------------------------------------------------
    #               Define HDF5 Structure
    # ---------------------------------------------------
    h5_file = h5py.File(output, 'w')
    inputs_group = h5_file.create_group('INPUTS')
    labels_group = h5_file.create_group('LABELS')

    for i in tqdm(range(num_entries), desc='Creating HDF5 dataset', unit='event'):
        num_jets = njet[i]
        for j in range(num_jets):
            jet_data[i][j] = (
                jet_e[i][j], jet_eta[i][j], jet_phi[i][j], jet_pt[i][j],
                jet_bTag[i][j], 0, 1      # Filling in 0 for charge
            )
        num_el = nel[i]
        for j in range(num_el):
            el_data[i][j] = (
                el_e[i][j], el_eta[i][j], el_phi[i][j], el_pt[i][j],
                0, el_charge[i][j], -1    # Filling in 0 for btag
            )
        num_mu = nmu[i]
        for j in range(num_mu):
            mu_data[i][j] = (
                mu_e[i][j], mu_eta[i][j], mu_phi[i][j], mu_pt[i][j],
                0, mu_charge[i][j], -2     # Filling in 0 for btag
            )
        global_data[i] = (njet[i], nbTagged[i])

        VertexID_data[i, :num_jets+num_el+num_mu] = np.concatenate((jet_match[i],el_match[i],mu_match[i]))

    inputs_group.create_dataset("jet", data=jet_data)
    inputs_group.create_dataset("electron", data=el_data)
    inputs_group.create_dataset("muon", data=mu_data)
    inputs_group.create_dataset("global", data=global_data)
    labels_group.create_dataset("VertexID", data=VertexID_data)

    # Close the HDF5 file
    h5_file.close()
    root_file.close()


if __name__ == "__main__":
    args = argparser()
    MakeDataset(args.input, args.output)
```