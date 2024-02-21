import torch

from torch import Tensor
from torch_scatter import scatter_max
from torchmetrics.functional.classification import binary_accuracy


def _index_delete_(src: Tensor, idx: Tensor) -> Tensor:
    mask = torch.ones(src.numel(), dtype=torch.bool, device=src.device)
    mask[idx] = False
    return src[mask]


def Accuracy(preds: Tensor, target: Tensor, batch: Tensor, num_patterns: int) -> Tensor:
    r"""Compute model accuracy for the reconstruction task.

    Args:
        preds (Tensor): a float tensor with shape (N,...), contains 
                        outputs of the reconstruction network.
        target (Tensor): a float tensor with shape (N,...), contains
                         truth hyperedges.
        batch (Tensor): an int tensor with shape (N,...), contains
                        batched indicies.
        num_patterns (int): number of truth patterns available in a graph. 

    :rtype: :class:`torch.Tensor`
    """
    picked = 0
    idx_preds = 0; src_preds = 0
    
    device = preds.device

    preds = preds.flatten()

    while picked < num_patterns:
        # remove the last highest
        if picked >= 1:
            preds = torch.scatter(preds,dim=0,index=idx_preds,src=torch.zeros(idx_preds.size(),device=device))
        
        src_preds, idx_preds = scatter_max(preds,batch)

        if picked == 0:
            evaluable = torch.scatter(torch.zeros(preds.size(),device=device),dim=0,index=idx_preds,src=src_preds)
        else:
            evaluable = torch.scatter(evaluable,dim=0,index=idx_preds,src=src_preds)

        picked += 1

    accuracy = binary_accuracy(evaluable,target.flatten(),ignore_index=0,threshold=0.00)

    return accuracy

