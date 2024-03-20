import os
import torch
import warnings
import HyPER

import pandas as pd

from torch import Tensor
from torch_geometric.loader import DataLoader
from torch_geometric.utils import unbatch, degree
from tqdm.rich import tqdm
from itertools import permutations, combinations
from typing import Optional

from HyPER.data import GraphDataset
from HyPER.utils import Settings, getUndirectedEdges


def Evaluate(log_dir: str, dataset: GraphDataset, option_file: Optional[str] = None, save_to: Optional[str] = None) -> pd.DataFrame:
    r"""Evaluate HyPER results using a trained model that saved in `log_dir`.

    Information saved:
        `HyPER_HE_RAW`: RAW hyperedge probability.
        `HyPER_GE_RAW`: RAW graphedge probability.
        `HyPER_HE_IDX`: event hyperedge set.
        `HyPER_GE_IDX`: event graphedge set.
        `HyPER_VT_COD`: vertices encoding.
    
    Args:
        log_dir (str): Tensorboard save dir.
        dataset (GraphDataset): GraphDataset to evaluate.
        option_file (str, optional): training configuration file. (default :obj:`str`=None)
        save_to (str, optional): save results into a .pkl file. (default :obj:`str`=None)
    
    :rtype: :class:`pandas.DataFrame`
    """
    # Load settings
    settings = Settings()
    if option_file is None:
        warnings.warn("Using default settings, make sure they are what you wanted.", UserWarning)
    else:
        settings.load(option_file)
    settings.show()

    # Load checkpoints
    ckpt_file = [filename for filename in os.listdir(os.path.join(log_dir, "checkpoints")) if filename.startswith("epoch")]
    if len(ckpt_file) > 1:
        warnings.warn(f"There are multiple .ckpt files listed in {log_dir}, using the last checkpoint.")
        ckpt_file = os.path.join(log_dir, "checkpoints", ckpt_file[-1])
    if len(ckpt_file) == 0:
        raise RuntimeError(f"No checkpoint files have been found in {log_dir}.")
    ckpt_file = os.path.join(log_dir, "checkpoints", ckpt_file[0])
    
    # Map model onto evaluation device
    map_location = torch.device('cuda') if settings.device == "gpu" else torch.device('cpu')

    # Load data
    dataset = DataLoader(dataset=dataset, batch_size=settings.batch_size, follow_batch=['edge_attr_s'], shuffle=False)

    model = HyPER.models.HyPERModel.load_from_checkpoint(ckpt_file, map_location=map_location)
    model.eval()

    # Things we want to save:
    hyperedge_out = []
    graphedge_out = []
    hyperedge_vct = []
    graphedge_vct = []
    hyperedges = []
    graphedges = []
    
    # Evaluate
    for data in tqdm(dataset, desc="Evaluating", unit='batch'):
        data.to(map_location)

        # Message Passing Step
        for i in range(model.hparams.message_passing_recurrent):
            if i == 0:
                x_prime, edge_attr_prime, u_prime = getattr(model, 'MessagePassing' + str(i))(
                    data.x_s, data.edge_index, data.edge_attr_s, data.u_s, data.batch
                )
            else:
                x_prime, edge_attr_prime, u_prime = getattr(model, 'MessagePassing' + str(i))(
                    x_prime, data.edge_index, edge_attr_prime, u_prime, data.batch
                )

        # Hyperedge attention
        x_hat, batch_hyperedge = model.Hyperedge(x_prime, u_prime, data.batch, model.hparams.hyperedge_order)

        # Unbatch results
        x_out         = unbatch(x_hat, batch_hyperedge.type(torch.int64), 0)
        edge_attr_out = unbatch(edge_attr_prime, data.edge_attr_s_batch, 0)
        N_nodes       = degree(data.batch).cpu().flatten().tolist()
        encodings     = unbatch(data.x_s[:,-1].reshape(-1,1),data.batch, 0)

        # Save unbatched results
        for i in range(0,len(x_out)):
            hyperedges.append([list(x) for x in combinations(range(int(N_nodes[i])),r=settings.hyperedge_order)])
            hyperedge_out.append(x_out[i].cpu().flatten().tolist())
            hyperedge_vct.append([list(x) for x in combinations(encodings[i].cpu().flatten().tolist(),r=settings.hyperedge_order)])

            # Graph edge directionality removal
            edge_index = list(permutations(range(0,int(N_nodes[i])),2))
            edge_index = Tensor([[int(x[0]) for x in edge_index], [int(x[1]) for x in edge_index]])
            edge_attrs = getUndirectedEdges(edge_index, edge_attr_out[i].cpu(), reduce='mean')
            graphedges.append([list(x) for x in combinations(range(int(N_nodes[i])),r=2)])
            graphedge_out.append(edge_attrs.flatten().tolist())
            graphedge_vct.append([list(x) for x in combinations(encodings[i].cpu().flatten().tolist(),r=2)])

    outputs = pd.DataFrame(
        {
            "HyPER_HE_RAW": hyperedge_out,
            "HyPER_GE_RAW": graphedge_out,
            "HyPER_HE_VCT": hyperedge_vct,
            "HyPER_GE_VCT": graphedge_vct,
            "HyPER_HE_IDX": hyperedges,
            "HyPER_GE_IDX": graphedges   
        }
    )

    if save_to is not None:
        outputs.to_pickle("save_to")

    return outputs