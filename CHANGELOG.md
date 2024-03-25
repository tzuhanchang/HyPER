# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/).

## \[1.0.0\] - 2024-MM-DD

### Added
 - Integrate [`hydra`](https://hydra.cc/) configuration framework ([#10](https://github.com/tzuhanchang/HyPER/pull/10))
 - Add documentation on dataset building ([#8](https://github.com/tzuhanchang/HyPER/pull/8))
 - Add `readthedocs` support ([#8](https://github.com/tzuhanchang/HyPER/pull/8))
 - Add `CHANGELOG.md` ([#5](https://github.com/tzuhanchang/HyPER/pull/5))
 - Add dataset configuration example [`db.yaml`](examples/ttbar_allhad/db.yaml) ([#4](https://github.com/tzuhanchang/HyPER/pull/4))

### Changed
 - Update example all-hadronic ttbar [reconstruction script](examples/ttbar_allhad/reconstruct_ttbar_allhad.py) ([#4](https://github.com/tzuhanchang/HyPER/pull/4))
 - Move function `getUndirectedEdges` to `HyPER.utils` ([#4](https://github.com/tzuhanchang/HyPER/pull/4))
 - Improvements to `GraphDataset`, using `HDF5` file to store and load data, compute graphs on the fly ([#4](https://github.com/tzuhanchang/HyPER/pull/4))

### Deprecated

### Fixed

### Removed
 - Removed legacy configuration class `HyPER.utils.Settings` ([#10](https://github.com/tzuhanchang/HyPER/pull/10))
 - Removed legacy training executable `Train.py` ([#10](https://github.com/tzuhanchang/HyPER/pull/10))
 - Removed graph building precedure: `BuildGraphs` ([#4](https://github.com/tzuhanchang/HyPER/pull/4))
 - Removed unused legacy functions ([#4](https://github.com/tzuhanchang/HyPER/pull/4))
 - Removed `presets`, moved `ttbar_allhad.json` to [`examples/ttbar_allhad`](examples/ttbar_allhad) ([#4](https://github.com/tzuhanchang/HyPER/pull/4))