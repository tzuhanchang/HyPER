Event Reconstruction
=======================

In this tutorial, we will demonstrate how to reconstruct events with a HyPER model.

Reconstruction
-----------

You might have noticed that in the configuration file shown [here](train.md), there exit few prediction related settings:
```yaml
predict_set:            null
predict_with:           "cpu"
predict_model:          null
predict_output:         null
topology:               ttbar_allhad
```
where you can provide the dataset and path to the HyPER model you want to use for the prediction. By default, the frist model you trained is saved to `HyPER_logs/version_0`. The output of the prediction is a `.pkl` file.

With configuration ready, running event reconstruction via:
```bash
python -m HyPER.predict --config-name=default [options]
```
Similar to `Hyper.train`, you can overwrite any options by providing them at the end.


Topology
-----------

`topology` defines which topology you wish to reconstruct your events into. These are functions stored in `HyPER/topology`.

!!! Note

    We currently have limited event topologies available, see [here](../topologies.md).

If you wish additional ones to be included, you can create an issue [here](https://github.com/tzuhanchang/HyPER/issues).
Or if you wish to contribute, you can create a pull request [here](https://github.com/tzuhanchang/HyPER/pulls).