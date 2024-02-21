# HyPER

**HyPER** (_Hypergraph for Particle Event Reconstruction_) is Graph Neural Network (GNN) utilising **_blended graph-hypergraph representation learning_** to reconstruct short-lived particles from their complex final states.

- [Quick Start](#quick-start)
- [Enviroment](#environment)

## Quick Start

Basic graph building and training parameters are defined in a `.json` configuration file, including graph topologies, loss function and optimiser. A set of preset configurations can be found in [presets](./presets).


### Graph building
> We are aiming to remove graph building precedure in a future release, directly computing graphs using [torch_hep](https://github.com/tzuhanchang/pytorch_hep) library while training.

Graph building is performed with [torch_hep](https://github.com/tzuhanchang/pytorch_hep) and [PyG](https://github.com/pyg-team/pytorch_geometric) library. The geometry of the source is defined in the configuration file, such as nodes (physical objects), edges (connections) and globals (event-wise information). For each source graph, a target graph is constructed for training use, it contains targeting hyperedges and graph edges with binary labels 1 and 0.

Prior to the training (or testing), [PyG](https://github.com/pyg-team/pytorch_geometric) graph datasets corresponding to training and validation (or testing) sets must be build:
```
python BuildGraphs.py -f ttbar_train.root -t Delphes -c ./presets/ttbar_allhad.json -o ./datasets/ttbar_train
```
This command build a graph dataset `ttbar_train` using the [TTree](https://root.cern.ch/doc/master/classTTree.html) `Delphes` in a [ROOT](https://root.cern) file: `ttbar_train.root`. Graphs in the constructed dataset have the structure defined in the configuration file `ttbar_allhad.json`.
The function `BuildGraph` uses [ATLAS variable naming scheme](http://opendata.atlas.cern/books/current/openatlasdatatools/_book/variable_names.html) by default.


### Network training
HyPER model is built upon [pytorch_lightning](https://lightning.ai/docs/pytorch/stable/), which uses [tensorboard](https://www.tensorflow.org/tensorboard) logger to save trained models.

To train HyPER:
```
python Train.py -c ./presets/ttbar_allhad.json
```
Make sure appropriate parameters are defined in your configuration file. If argument `-c` is not passed, the programme uses a set of default parameters and a `UserWarning` will be raised. 


### Evaluation
To evaluate trained HyPER model on a dataset:
```
from HyPER.data import GraphDataset
from HyPER.evaluation import Evaluate

test_dataset = GraphDataset(root="./datasets/ttbar_test")

results = Evaluate(log_dir="./HyPER_logs/version_0", dataset=test_dataset, option_file="./presets/ttbar_allhad.json")
```
`results` is a [pandas.DataFrame](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html), in which, four output variables are saved:

| Variables | Description |
| ------------- | ------------- |
| `HyPER_HE_IDX` | Indices of the nodes enclosed by a hyperedge  |
| `HyPER_HE_RAW` | Soft probability of a hyperedge |
| `HyPER_GE_IDX` | Indices of the nodes connected by a graph edge |
| `HyPER_GE_RAW` | Soft probability of a graph edge |

Based on these RAW outputs, events can then be reconstructed using users desired methods. We provide a example code for the all-hadronic reconstruction in [examples/ttbar_allhad](examples/ttbar_allhad).


## Enviroment
Currenlty, `conda` auto environment solving is only supported by Linux/amd64 machines, due to some ongoing issues with `torch_geometric` in MacOS. A conda environment file [`environment.yml`](environment.yml) is provided. We recommand use [miniforge](https://github.com/conda-forge/miniforge) as your `conda` package manager due to its lightweightness.

To create a `conda` environment named _"HyPER"_:
```
conda env create -f environment.yml
```

We have tested the code with `CUDA=11.8`, HyPER should work with any `CUDA` versions above.