Installation
=======================

!!! Note

    HyPER has been tested to be compatible with CUDA=11.8+ and Python=3.9+. For older versions, the stability is not guaranteed.


Quick Start
-----------

Get the latest HyPER release:
```
git clone https://github.com/tzuhanchang/HyPER.git
```

Setup Environment
-----------

!!! Note

    We recommend setting up environemnt using [Anaconda or Miniconda](https://conda.io/projects/conda/en/latest/user-guide/install) with our provided `environment_linux.yml` file.

After installing `conda`, simply run:
```
conda env create -f environment_linux.yaml
```

To setup environment manually, HyPER requires the following dependencies:

 * [PyTorch](https://pytorch.org):`pytorch >= 2.0`, `torchvision`, `torchaudio` and `pytorch-cuda` (for running on GPUs).
 * [PyG](https://pyg.org): `torch_geometric >= 2.0` and `torch_scatter`
 * [PyTorch HEP](https://github.com/tzuhanchang/pytorch_hep):`torch_hep == 0.0.4`
 * [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/): `lightning`
 * [TensorBoard](https://www.tensorflow.org/tensorboard): `tensorboard`
 * [HDF5 for Python](https://www.h5py.org): `h5py`
 * Other: `numpy`, `pandas`, `tqdm`, `rich`

There are some on going issues with PyG (and additional PyG libraries) installation on MacOS, try install it from [wheels](https://data.pyg.org/whl/).