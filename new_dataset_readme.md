# Dataset 2 README

The `HyPER.train` and `HyPER.predict` scripts now point to a directory which contains the input data (test and train h5 files) as well as the config.

This directory, in this instance called `HyPER_run1` should have the following structure:
```
HyPER_run1
├── config.yaml
├── processed
└── raw
    ├── test.h5
    └── train.h5
```
The `raw` directory contians the input h5 files. 

## Input h5 file structure

An example dataset structure is as follows:
```
file.h5
├── INPUTS
│   ├── JET
│   ├── LEPTON
│   ├── GLOBAL
│   ├── ...
├── LABELS
│   ├── JET
│   ├── LEPTON
│   ├── ...
└── METADATA
    ├── FullyMatched
```

### Inputs 
The `INPUTS` group contains the feature the model will use to build the event graphs. This data is used in both training and testing phases. Each dataset (`JET` , `LEPTON`, ...) is a structured numpy array with fields which correspond to node features, plus the `GLOBAL` dataset which contains the graph-level (event-level) features.

The list of node features which can be used is abitrary, as is the list of global features. Padded entries must be padded with **np.nan**. Edge features are not included in the dataset, but are constructed when HyPER builds the dataset (the primary motivation for this is size of input files).


### Labels 
The `LABELS` group contains the corresponding "truth-labels" for each dataset in INPUTS. This is used to define the targets for the model to train against. Each dataset (`JET` , `LEPTON`, ...) is a single array of integers.

A comment on the truth-matching labels which we use to build the edge and hyperedge targets. You must use **positive integers** to define all non-padded objects in an event, and you must use **np.nan** as the padding placeholder. [HyPER uses the Cantor pairing function to define unique node indices, and removes padded entries specifically by looking for NaN.]

The integer labels used match to known parton in the simulation truth-record. For example, the all-hadronic decay of a ttbar pair to six partons could use label convention:

| Truth parton | Integer label |
|--------------|---------------|
| b1           | 1             |
| W1j1         | 2             |
| W1j2         | 3             |
| b2           | 4             |
| W2j1         | 5             |
| W2j2         | 6             |


The field `LABELS.JET` could then look like 
```
[ 0 , 4 , 2 , 0 , 1 , 5 , 0 , 0 , 6 , NaN , NaN , NaN ]
```
for an event with 9 final-state jets, padded to an array of size 12. In this instance, the 0 indicates jets which are not matched to truth-partons. Note that not all jets may match to a parton: this is dependent on one's truth-matching scheme, hence 3 being absent in the example. 

These truth-labels should then correspond to the what is defined in the `target` field in the dataset config below. Note that the unmatched jets can be labelled with any positive integer, so long as this is not then stipulated in the config `target` field.

## Dataset config (`config.yaml`)

The `config.yaml` file is the configuration for the dataset. This file name is fixed. It should contain an `input` block and a `target` block e.g:

```yaml 
input:
  nodes: # of form {Node type: enum}. Lists each object as defined in the "INPUTS" group of the h5 files, with an enum which the user can define, and which HyPER
    JET: 1
    LEPTON: 2
  node_features: # Ordered list of node features - although should change such that doesn't need to be in this order
    - e
    - eta
    - phi 
    - pt 
    - btag
    - charge
  node_transforms: # Corresponding list of transformations to the node features
    - torch.log(x)
    - x
    - x
    - torch.log(x)
    - x
    - x
  edge_features: # List of edge features from a pre-defined list of edge features which HyPER can construct
    - delta_eta
    - delta_phi
    - delta_R
    - M2
  global_features: # Global feature list
    - njet
    - nbTagged
  global_transforms: # Corresponding list of transformations to the global data
    - x/6
    - x/2
  pre_transform: True # Boolean whether to transform the features before saving the dataset PyG object to file

target: 
# Nodes are of the form [particle_enum - matching index]. particle_enum corresponds to the values of the nodes field above. 
# This example looks combines nodes which are of type jet
    w1: ['1-2','1-3']
    w2: ['1-5','1-6']
  hyperedge: 
    t1: ['1-1','1-2','1-3']
    t2: ['1-4','1-5','1-6']
process: # We don't use this yet
  4tops: 1
  3tops: 2
```

The dataset structure used saves the processed data (pre-training) to the `processed` directory.