import torch

from torch import Tensor
from torch.nn import BCELoss
from torch_scatter import scatter

from typing import Optional


def EdgeLoss(edge_attr_out: Tensor, edge_attr_t: Tensor, edge_attr_batch: Tensor,
             criterion: Optional[callable] = BCELoss(reduction='none'), reduction: Optional[str] = 'mean') -> Tensor:
    r"""Calculate per graph edge loss.

    Args:
        edge_attr_out (Tensor): output edge features.
        edge_attr_t (Tensor): edge targets.
        edge_attr_batch (Tensor): edge batch.
        criterion (optional: callable): loss/cost function (default: BCELoss).
        reduction (optional: str): the reduce operation (default: 'mean').

    :rtype: :class:`Tensor`
    """
    l = criterion(edge_attr_out, edge_attr_t.float())
    return scatter(l.flatten(), edge_attr_batch, reduce=reduction)


def HyperedgeLoss(x_out: Tensor, x_t: Tensor, x_t_batch: Tensor,
                 criterion: Optional[callable] = BCELoss(reduction='none'), reduction: Optional[str] = 'mean') -> Tensor:
    r"""Calculate per hyperedge loss.

    Args:
        x_out (Tensor): output hyperedge features.
        x_t (Tensor): hyperedge targets.
        x_t_batch (Tensor): hyperedge batch.
        criterion (optional: callable): loss/cost function (default: BCELoss).
        reduction (optional: str): the reduce operation (default: 'mean').
    
    :rtype: :class:`Tensor`
    """
    l = criterion(x_out, x_t.float())
    l = scatter(l.flatten(), x_t_batch, reduce=reduction)
    loss_masks = scatter(x_t.flatten(), x_t_batch, reduce='sum') > 0
    return l, loss_masks


def CombinedLoss(loss_hyperedge: Tensor, loss_edge: Tensor, alpha: Optional[float] = 0.5, reduction: Optional[str] = 'mean',
                 loss_hyperedge_masks: Optional[Tensor] = None) -> Tensor:
    r"""Get combined loss.

    Args:
        loss_hyperedge (Tensor): hyperedge loss.
        loss_edge (Tensor): edge loss.
        alpha (optional: Tensor): hyperedge loss weight (default: 0.5)
        reduction (optional: str): the reduce operation (default: 'mean').

    :rtype: :class:`Tensor`
    """
    if loss_hyperedge_masks is not None:
         # Ignore `loss_hyperedge` for an event that has no labelled hyperedges:
        with torch.no_grad():
            device = loss_hyperedge.device
            loss_shape = loss_hyperedge.shape

            alpha = torch.full(loss_shape, alpha, device=device)
            alpha = torch.scatter(torch.zeros(loss_shape, device=device), 0, loss_hyperedge_masks.nonzero().flatten(), alpha)

    l = ( alpha * loss_hyperedge ) + ( (1-alpha) * loss_edge )

    if reduction == 'mean':
        rd = torch.mean
    elif reduction == 'max':
        rd = torch.max
    elif reduction == 'min':
        rd = torch.min
    # ------- custom reduction func -------
    # elif
    # -------------------------------------
    return rd(l)