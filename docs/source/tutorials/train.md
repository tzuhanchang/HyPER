Network Training
=======================

Once dataset(s) is constructed following [this instruction](dataset.md), network training can be carried out. In this tutorial, we will show you how to configure and train a HyPER model.

Configuration
-----------

We provide a default configuration file, `config/default.yaml`:
```yaml
dataset:          null
train_set:        null
val_set:          null
train_val_split:  0.9
max_n_events:     -1
drop_last:        False

num_message_layers:   3
hyperedge_order:      3
message_feats:        64
hyperedge_feats:      128
dropout:              0.01

optimizer:            "Adam"
learning_rate:        0.0003
criterion_edge:       "BCE"
criterion_hyperedge:  "BCE"
loss_reduction:       "mean"
alpha:                0.8

epochs:       500
patience:     50
batch_size:   4096

device:       "gpu"
num_devices:   1

savedir: "HyPER_logs"
continue_from_ckpt: null

predict_set:    null
predict_with:   "cpu"
predict_model:  null
predict_output: null
topology: ttbar_allhad

onnx_output:   "HyPER.onnx"
convert_model: null
```
You can overwrite any option in the configuration file to maximise HyPER's performance on your analysis.

!!! Note

    The `PyTorch` to `ONNX` model conversion script does not currently work on the main branch. If you need to do such a conversion, please use the following [HyPER repository](https://github.com/diegobaronm/HyPER_ONNX). It is meant only for exporting, not training or predicting HyPER scores.

It's worth noting that in the case of top quark, the `hyperedge_order` is set to 3 because of the number of final states associated with it. But it could be more than 3 in cases of some BSM processes, such as RPV STop decays.


Training
-----------

To train HyPER, run

```bash
python -m HyPER.train --config-name=default [options]
```

HyPER uses [Hydra](https://hydra.cc/) for configuring run [#10](https://github.com/tzuhanchang/HyPER/pull/10), so, you can overwrite any option using, for example, `max_n_events=100000` at the end. It overwrites the `max_n_events` value provided in the `default.yaml` file.

!!! Warning

    Configuration files must be placed in the `configs` folder.
    Provide the file name **without** `.yaml` extension to `--config-name`.