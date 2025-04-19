import os
import hydra
import torch
import warnings

import pandas as pd
import lightning.pytorch as pl

from tqdm import tqdm
from itertools import permutations, combinations
from omegaconf import DictConfig, OmegaConf

from HyPER.data import HyPERDataModule
from HyPER.models import HyPERModel
from HyPER.utils import getUndirectedEdges, ResultWriter
from HyPER.topology import *


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def Predict(cfg : DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    datamodule = HyPERDataModule(
        root = cfg['dataset'],
        train_set = None,
        val_set = None,
        predict_set = cfg['predict_set'],
        batch_size = cfg['batch_size'],
        percent_valid_samples = 1 - float(cfg['train_val_split']),
        pin_memory = True if cfg['device'] == "gpu" else False,
        drop_last = False
    )

    # Map location
    map_location = torch.device('cuda') if cfg['predict_with'].lower() == "gpu" else torch.device('cpu')

    # Load checkpoints
    assert cfg['predict_model'] is not None, "No model directory is provided in `predict_model`. Abort!"
    ckpt_file = [filename for filename in os.listdir(os.path.join(cfg['predict_model'], "checkpoints")) if filename.startswith("epoch")]
    if len(ckpt_file) > 1:
        warnings.warn(f"There are multiple .ckpt files listed in {cfg['predict_model']}, using the last checkpoint.")
        ckpt_file = os.path.join(cfg['predict_model'], "checkpoints", ckpt_file[-1])
    if len(ckpt_file) == 0:
        raise RuntimeError(f"No checkpoint files have been found in {cfg['predict_model']}.")
    ckpt_file = os.path.join(cfg['predict_model'], "checkpoints", ckpt_file[0])

    # Load hyperparameters
    hparams_file = os.path.join(cfg['predict_model'], "hparams.yaml")
    assert os.path.isfile(hparams_file), f"`hparams.ymal` is not found in {cfg['predict_model']}."

    model = HyPERModel.load_from_checkpoint(
        checkpoint_path = ckpt_file,
        hparams_file = hparams_file,
        map_location = map_location,
    )

    trainer = pl.Trainer(
        accelerator = cfg['predict_with'],
        devices = cfg['num_devices']
    )

    out = trainer.predict(model, datamodule=datamodule)
    
    print(out)

    hyperedge_out = []
    graphedge_out = []
    hyperedge_vct = []
    graphedge_vct = []
    hyperedges = []
    graphedges = []
    
    for i in tqdm(range(len(out)), desc="Evaluating", unit='batch'):
        x_out, edge_attr_out, N_nodes, encodings = out[i]

        for j in range(len(x_out)):
            hyperedges.append([list(x) for x in combinations(range(int(N_nodes[j])),r=cfg['hyperedge_order'])])
            hyperedge_out.append(x_out[j].cpu().flatten().tolist())
            hyperedge_vct.append([list(x) for x in combinations(encodings[j].cpu().flatten().tolist(),r=cfg['hyperedge_order'])])

            # Graph edge directionality removal
            edge_index = list(permutations(range(0,int(N_nodes[j])),2))
            edge_index = torch.tensor([[int(x[0]) for x in edge_index], [int(x[1]) for x in edge_index]])
            edge_attrs = getUndirectedEdges(edge_index, edge_attr_out[j].cpu(), reduce='mean')
            graphedges.append([list(x) for x in combinations(range(int(N_nodes[j])),r=2)])
            graphedge_out.append(edge_attrs.flatten().tolist())
            graphedge_vct.append([list(x) for x in combinations(encodings[j].cpu().flatten().tolist(),r=2)])

    results = pd.DataFrame(
        {
            "HyPER_HE_RAW": hyperedge_out,
            "HyPER_GE_RAW": graphedge_out,
            "HyPER_HE_VCT": hyperedge_vct,
            "HyPER_GE_VCT": graphedge_vct,
            "HyPER_HE_IDX": hyperedges,
            "HyPER_GE_IDX": graphedges   
        }
    )

    results = eval(cfg['topology'])(results)
    
    if cfg['predict_output'] is None:
        warnings.warn("No output path is provided in `predict_output`, use default: `output.h5`.")
        ResultWriter(results, "output.h5")
    else:
        if str(cfg['predict_output'])[-3:] == '.h5':
            warnings.warn("Saving results to a `.h5` file, RAW outputs will not be saved. If you want to save all output, use `.pkl` extension.", UserWarning)
            ResultWriter(results, str(cfg['predict_output']))
        elif str(cfg['predict_output'])[-4:] == '.pkl':
            warnings.warn("Pickling all results (including RAW network outputs), your performance may suffer.", UserWarning)
            results.to_pickle(str(cfg['predict_output']))
        else:
            raise ValueError("You must provide a file extension: `.h5` or `.pkl`.")

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    Predict()
