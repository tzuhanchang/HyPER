import pytorch_lightning as L

from pytorch_lightning.loggers import TensorBoardLogger
from lightning_utilities.core.imports import RequirementCache
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
    RichModelSummary,
    DeviceStatsMonitor,
    ModelSummary,
    TQDMProgressBar,
    EarlyStopping
)

from HyPER.data import HyPERDataModule
from HyPER.models import HyPERModel
from HyPER.utils import Settings
from typing import Optional


_RICH_AVAILABLE = RequirementCache("rich>=10.2.2")

callbacks = [ModelCheckpoint(verbose=True,
                             monitor="fuzzy_accuracy/validation_accuracy_hyperedge",
                             save_top_k=1,
                             mode="max",
                             save_last=True),
             EarlyStopping(monitor="fuzzy_accuracy/validation_accuracy_hyperedge",
                           mode="max",
                           min_delta=0.00,
                           patience=20,
                           verbose=False),
             LearningRateMonitor(),
             DeviceStatsMonitor(),
             RichProgressBar() if _RICH_AVAILABLE else TQDMProgressBar(),
             RichModelSummary(max_depth=1) if _RICH_AVAILABLE else ModelSummary(max_depth=1)
             ]


def Train(settings: Settings, ckpt_path: Optional[str] = None):
    r"""Perform network training using parameters defined in 
    `option_file`.

    Args:
        option_file (str, optional): `.json` file, stores training related parameters. (default: :obj:`str`=None)
        ckpt_path (str, optional): restore the full training from the given path. (default: :obj:`str`=None)
    """
    datamodule = HyPERDataModule(
        train_dir = settings.train_data,
        val_dir = settings.valid_data,
        batch_size = settings.batch_size,
        percent_valid_samples = 1 - settings.train_val_split,
        in_memory = settings.in_memory_dataset,
        num_workers = settings.num_dataloader_workers,
        pin_memory = True if settings.device == "gpu" else False,
        all_matched = settings.all_matched
    )

    model = HyPERModel(
        node_in_channels = len(settings.node_features)+1,
        edge_in_channels = 4,
        global_in_channels = len(settings.global_features),
        message_feats = settings.message_feats,
        dropout = settings.dropout,
        message_passing_recurrent = settings.message_passing_recurrent,
        contraction_feats = settings.contraction_feats,
        hyperedge_order = settings.hyperedge_order,
        criterion_edge = settings.criterion_edge,
        criterion_hyperedge = settings.criterion_hyperedge,
        optimizer = settings.optimizer,
        lr = settings.learning_rate,
        alpha = settings.alpha,
        reduction = settings.loss_reduction
    )

    trainer = L.Trainer(
        accelerator = settings.device,
        devices = settings.num_device,
        max_epochs = settings.epochs,
        callbacks = callbacks,
        logger = TensorBoardLogger(save_dir=settings.savedir, name="", log_graph=True)
    )

    if ckpt_path is not None:
        print("Resume training state from %s"%(ckpt_path))

    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)