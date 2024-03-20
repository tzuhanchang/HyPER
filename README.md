# HyPER

**HyPER** (_Hypergraph for Particle Event Reconstruction_) is Graph Neural Network (GNN) utilising **_blended graph-hypergraph representation learning_** to reconstruct short-lived particles from their complex final states.

- [Quick Start](#quick-start)
- [Enviroment](#environment)

## Quick Start

> We have removed graph building precedure following [#4](https://github.com/tzuhanchang/HyPER/pull/4). `GraphDataset` now loads flat data from a `HDF5` file and computes graph structure on the fly. We are no longer recommand and provide support to [release v0.1](https://github.com/tzuhanchang/HyPER/releases/tag/v0.1).


### Network training
HyPER model is built upon [pytorch_lightning](https://lightning.ai/docs/pytorch/stable/), which uses [tensorboard](https://www.tensorflow.org/tensorboard) logger to save trained models.

To train HyPER:
```
python Train.py -c ./examples/ttbar_allhad/ttbar_allhad.json
```
Make sure appropriate parameters are defined in your configuration file. If argument `-c` is not passed, the programme uses a set of default parameters and a `UserWarning` will be raised. 


### Evaluation
To evaluate trained HyPER model on a dataset:
```
from HyPER.data import GraphDataset
from HyPER.evaluation import Evaluate

test_dataset = GraphDataset(
    path="./datasets/ttbar_test.h5",
    configs="./examples/ttbar_allhad/db.yaml"
)

results = Evaluate(
    log_dir="./HyPER_logs/version_0",
    dataset=test_dataset,
    option_file="./examples/ttbar_allhad/ttbar_allhad.json",
    save_to="output.pkl"
)
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