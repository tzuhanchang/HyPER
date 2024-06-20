# HyPER

[![Documentation Status](https://readthedocs.org/projects/hyper-hep/badge/?version=latest)](https://hyper-hep.readthedocs.io/en/latest/?badge=latest)
[![arXiv](https://img.shields.io/badge/arXiv-2402.10149-b31b1b.svg)](https://arxiv.org/abs/2402.10149)

**[Documentation](https://hyper-hep.readthedocs.io)**

**HyPER** (_Hypergraph for Particle Event Reconstruction_) is Graph Neural Network (GNN) utilising **_blended graph-hypergraph representation learning_** to reconstruct short-lived particles from their complex final states.

- [Quick Start](#quick-start)
- [Enviroment](#environment)

## Quick Start

> We have removed graph building precedure following [#4](https://github.com/tzuhanchang/HyPER/pull/4). `GraphDataset` now loads flat data from a `HDF5` file and computes graph structure on the fly. We are no longer recommand and provide support to [release v0.1](https://github.com/tzuhanchang/HyPER/releases/tag/v0.1).


### Network training
HyPER model is built upon [pytorch_lightning](https://lightning.ai/docs/pytorch/stable/), which uses [tensorboard](https://www.tensorflow.org/tensorboard) logger to save trained models.

To train HyPER:
```
python -m HyPER.train --config-name=default [options]
```
HyPER uses [hydra](https://hydra.cc/) for configuring run ([#10](https://github.com/tzuhanchang/HyPER/pull/10)). You can overwrite any option using, for example, `all_matched=False` at the end, it overwrites the `all_matched` option provided in your configuration file.
> Configuration files must be placed in the `configs` folder. Provide the file name without `.yaml` extension to `--config-name`.


### Evaluation
To evaluate trained HyPER model on a dataset:
```
python -m HyPER.predict --config-name=default [options]
```
Four output variables are saved:

| Variables | Description |
| ------------- | ------------- |
| `HyPER_HE_IDX` | Indices of the nodes enclosed by a hyperedge  |
| `HyPER_HE_RAW` | Soft probability of a hyperedge |
| `HyPER_GE_IDX` | Indices of the nodes connected by a graph edge |
| `HyPER_GE_RAW` | Soft probability of a graph edge |

Based on these RAW outputs, events can then be reconstructed by stating the correct `topology` in the configuration file. We currently have limited event topologies available in HyPER, see [`HyPER/topology`](https://github.com/tzuhanchang/HyPER/tree/main/HyPER/topology). If you wish additional ones to be added, you can create a issue [here](https://github.com/tzuhanchang/HyPER/issues).


## Enviroment
Currenlty, `conda` auto environment solving is only supported by Linux/amd64 machines, due to some ongoing issues with `torch_geometric` in MacOS. A conda environment file [`environment_linux.yml`](environment_linux.yml) is provided. We recommend using [miniforge](https://github.com/conda-forge/miniforge) as your `conda` package manager due to its lightweightness.

To create a `conda` environment named _"HyPER"_:
```
conda env create -f environment_linux.yml
```

We have tested the code with `CUDA=11.8`, HyPER should work with any `CUDA` versions above.