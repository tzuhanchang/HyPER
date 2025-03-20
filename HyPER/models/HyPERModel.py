import torch

from torch.nn import BCELoss, Sigmoid, CrossEntropyLoss
from torch.optim import lr_scheduler, Adam
from torch_geometric.utils import unbatch, degree
from lightning import LightningModule
from HyPER.models import MPNNs, HyperedgeModel, HyperedgeLoss, EdgeLoss, CombinedLoss
from HyPER.evaluation import Accuracy
from torchmetrics.classification import BinaryAccuracy
from typing import Optional


class HyPERModel(LightningModule):
    r"""HyPER model.
    :obj:`HyPERModel` is built using the message passing modules :obj:`MPGNN`
    and the hyperedge module :obj:`HyperedgeModel`.

    Args:
        node_in_channels (int): number of node features of input graph.
        edge_in_channels (int): number of edge features of input graph.
        global_in_channels (int): number of global features of input graph.
        message_feats (int, optional): number of intermediate features. (default :obj:`int`=32)
        dropout (float, optional): probability of an element to be zeroed. (default :obj:`float`=0.01)
        message_passing_recurrent (int, optional): number of message passing steps. (default :obj:`int`=3)
        criterion_edge (callable, optional): edge loss function. (default: :obj:`callable`=torch.nn.BCELoss() )
        criterion_hyperedge (callable, optional): hyperedge loss function. (default: :obj:`callable`=torch.nn.BCELoss() )
        optimizer (callable, optional): optimizer. (default: :obj:`callable`=torch.optim.Adam)
        lr (float, optional): learning rate. (default: :obj:`float`=1e-3)
        alpha (float, optional): edge/hyperedge loss balance. (default: :obj:`float`=0.5)
        reduction (str, optional): specifies the reduction to apply to the output loss. (default: 'mean')

    :rtype: :class:`Tuple[torch.Tensor,torch.Tensor,torch.Tensor]
    """
    def __init__(
            self,
            node_in_channels,
            edge_in_channels,
            global_in_channels,
            final_edge_out_channels:        Optional[int] = 2,
            final_hyperedge_out_channels:   Optional[int] = 3,
            message_feats: Optional[int] = 32,
            dropout: Optional[float] = 0.01,
            message_passing_recurrent: Optional[int] = 3,
            contraction_feats: Optional[int] = 32,
            hyperedge_order: Optional[int] = 3,
            criterion_edge: Optional[str] = "BCE",
            criterion_hyperedge: Optional[str] = "BCE",
            optimizer: Optional[str] = "Adam",
            lr: Optional[float] = 1e-3,
            alpha: Optional[float] = 0.5,
            reduction: Optional[str] = 'mean'
        ):
        super().__init__()

        self.save_hyperparameters()

        for i in range(self.hparams.message_passing_recurrent):
            if i == 0:
                setattr(self, 'MessagePassing' + str(i),
                        MPNNs(
                            node_in_channels, edge_in_channels, global_in_channels,
                            node_out_channels = self.hparams.message_feats, edge_out_channels = self.hparams.message_feats, global_out_channels = self.hparams.message_feats,
                            message_feats = self.hparams.message_feats, dropout = self.hparams.dropout
                        )
                )

            elif i == self.hparams.message_passing_recurrent-1:
                setattr(self, 'MessagePassing' + str(i),
                        MPNNs(
                            self.hparams.message_feats, self.hparams.message_feats, self.hparams.message_feats,
                            node_out_channels = self.hparams.message_feats, edge_out_channels = final_edge_out_channels, global_out_channels = self.hparams.message_feats,
                            message_feats = self.hparams.message_feats, dropout = self.hparams.dropout, activation = Sigmoid(), p_out = 'edge'
                        )
                )

            else:
                setattr(self, 'MessagePassing' + str(i),
                        MPNNs(
                            self.hparams.message_feats, self.hparams.message_feats, self.hparams.message_feats,
                            node_out_channels = self.hparams.message_feats, edge_out_channels = self.hparams.message_feats, global_out_channels = self.hparams.message_feats,
                            message_feats = self.hparams.message_feats, dropout = self.hparams.dropout
                        )
                )

        self.Hyperedge = HyperedgeModel(
            node_in_channels = self.hparams.message_feats, node_out_channels = 1, global_in_channels = self.hparams.message_feats,
            message_feats = self.hparams.contraction_feats, dropout = self.hparams.dropout
        )

    def forward(self, x, edge_index, edge_attr, u, batch, hyperedge_index, hyperedge_index_batch):
        # Message Passing Step
        for i in range(self.hparams.message_passing_recurrent):
            if i == 0:
                x_prime, edge_attr_prime, u_prime = getattr(self, 'MessagePassing' + str(i))(
                    x, edge_index, edge_attr, u, batch
                )
            else:
                x_prime, edge_attr_prime, u_prime = getattr(self, 'MessagePassing' + str(i))(
                    x_prime, edge_index, edge_attr_prime, u_prime, batch
                )

        # Hyperedge Finding Step
        x_hat, batch_hyperedge  = self.Hyperedge(x_prime, u_prime, batch, hyperedge_index, hyperedge_index_batch, self.hparams.hyperedge_order)
        return x_hat, batch_hyperedge, edge_attr_prime

    def _shared_step(self, data):
        return self.forward(data.x, data.edge_index, data.edge_attr, data.u, data.batch, data.hyperedge_index, data.hyperedge_index_batch)

    def configure_optimizers(self):
        if str(self.hparams.optimizer).lower() == 'adam':
            optimizer = Adam(self.parameters(), lr=self.hparams.lr)
        # --------- custom optimizers ---------
        # elif
        # -------------------------------------
        else:
            raise ValueError("Supported optimizers are: `torch.Adam`.")
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.8, patience=10),
                "interval": "epoch",
                "monitor": "fuzzy_accuracy/validation_accuracy_hyperedge",
                "frequency": 1,
                "strict": True
            },
        }

    def training_step(self, train_batch, batch_idx):
        x_hat, batch_hyperedge, edge_attr_prime = self._shared_step(train_batch)

        # Train Loss Calculation
        if str(self.hparams.criterion_edge).lower() == 'bce':
            # criterion_edge = BCELoss(reduction='none')
            criterion_edge = CrossEntropyLoss(reduction='none')
        # ------- custom loss functions -------
        # elif
        # -------------------------------------
        else:
            raise ValueError("Supported edge loss functions are: `torch.BCELoss`.")

        if str(self.hparams.criterion_hyperedge).lower() == 'bce':
            criterion_hyperedge = BCELoss(reduction='none')
        # ------- custom loss functions -------
        # elif
        # -------------------------------------
        else:
            raise ValueError("Supported edge loss functions are: `torch.BCELoss`.")

        loss_edge = EdgeLoss(edge_attr_prime, train_batch.edge_attr_t, train_batch.edge_attr_batch, criterion=criterion_edge, reduction='mean')
        loss_hyperedge, loss_hyperedge_masks = HyperedgeLoss(x_hat, train_batch.hyperedge_attr_t.float(), batch_hyperedge, criterion_hyperedge, reduction='mean')
        loss = CombinedLoss(loss_edge, loss_hyperedge, alpha=self.hparams.alpha, reduction=self.hparams.reduction, loss_hyperedge_masks=loss_hyperedge_masks)

        # Logging
        self.log('loss/train_loss', loss, batch_size=len(train_batch), on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return loss

    def validation_step(self, val_batch, batch_idx):
        x_hat, batch_hyperedge, edge_attr_prime = self._shared_step(val_batch)

        # Validation Loss Calculation
        if str(self.hparams.criterion_edge).lower() == 'bce':
            # criterion_edge = BCELoss(reduction='none')
            criterion_edge = CrossEntropyLoss(reduction='none')
        # ------- custom loss functions -------
        # elif
        # -------------------------------------
        else:
            raise ValueError("Supported edge loss functions are: `torch.BCELoss`.")

        if str(self.hparams.criterion_hyperedge).lower() == 'bce':
            criterion_hyperedge = BCELoss(reduction='none')
        # ------- custom loss functions -------
        # elif
        # -------------------------------------
        else:
            raise ValueError("Supported edge loss functions are: `torch.BCELoss`.")

        loss_edge = EdgeLoss(edge_attr_prime, val_batch.edge_attr_t, val_batch.edge_attr_batch, criterion=criterion_edge, reduction='mean')
        loss_hyperedge, loss_hyperedge_masks = HyperedgeLoss(x_hat, val_batch.hyperedge_attr_t.float(), batch_hyperedge, criterion_hyperedge, reduction='mean')
        loss = CombinedLoss(loss_edge, loss_hyperedge, alpha=self.hparams.alpha, reduction=self.hparams.reduction, loss_hyperedge_masks=loss_hyperedge_masks)

        # Validation Accuracy Calculation
        accuracy_edge = BinaryAccuracy(ignore_index=0).to(edge_attr_prime)
        accuracy_hyperedge = Accuracy(x_hat, val_batch.hyperedge_attr_t.float(), batch_hyperedge, num_patterns=2)

        # Logging
        self.log('loss/validation_loss', loss, batch_size=len(val_batch), on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log('fuzzy_accuracy/validation_accuracy_edge', accuracy_edge(edge_attr_prime.flatten(), val_batch.edge_attr_t.float().flatten()),
                 batch_size=len(val_batch), on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log('fuzzy_accuracy/validation_accuracy_hyperedge', accuracy_hyperedge, batch_size=len(val_batch), on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x_hat, batch_hyperedge, edge_attr_prime = self._shared_step(batch)

        # Unbatch results
        x_out         = unbatch(x_hat, batch_hyperedge.type(torch.int64), 0)
        edge_attr_out = unbatch(edge_attr_prime, batch.edge_attr_batch, 0)
        N_nodes       = degree(batch.batch).cpu().flatten().tolist()
        encodings     = unbatch(batch.x[:,-1].reshape(-1,1),batch.batch, 0)

        return x_out, edge_attr_out, N_nodes, encodings