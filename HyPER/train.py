import hydra
import torch
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
from omegaconf import DictConfig, OmegaConf

_RICH_AVAILABLE = RequirementCache("rich>=10.2.2")


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def Train(cfg : DictConfig) -> None:
    r"""Perform network training using parameters defined in 
    `option_file`.

    Args:
        cfg (str): a `.yaml` file, stores training related parameters. (default: :obj:`str`=None).
    """
    print(OmegaConf.to_yaml(cfg))

    datamodule = HyPERDataModule(
        db_config = cfg['db_config'],
        train_set = cfg['train_set'],
        val_set = cfg['val_set'],
        batch_size = cfg['batch_size'],
        percent_valid_samples = 1 - float(cfg['train_val_split']),
        num_workers = cfg['num_workers'],
        pin_memory = True if cfg['device'] == "gpu" else False,
        all_matched = cfg['all_matched']
    )

    model = HyPERModel(
        node_in_channels = datamodule.node_in_channels,
        edge_in_channels = datamodule.edge_in_channels,
        global_in_channels = datamodule.global_in_channels,
        message_feats = cfg['message_feats'],
        dropout = cfg['dropout'],
        message_passing_recurrent = cfg['num_message_layers'],
        contraction_feats = cfg['hyperedge_feats'],
        hyperedge_order = cfg['hyperedge_order'],
        criterion_edge = cfg['criterion_edge'],
        criterion_hyperedge = cfg['criterion_hyperedge'],
        optimizer = cfg['optimizer'],
        lr = cfg['learning_rate'],
        alpha = cfg['alpha'],
        reduction = cfg['loss_reduction']
    )

    callbacks = [
        ModelCheckpoint(
            verbose=True,
            monitor="fuzzy_accuracy/validation_accuracy_hyperedge",
            save_top_k=1,
            mode="max",
            save_last=True
        ),
        EarlyStopping(
            monitor="fuzzy_accuracy/validation_accuracy_hyperedge",
            mode="max",
            min_delta=0.00,
            patience=20,
            verbose=False
        ),
        LearningRateMonitor(),
        DeviceStatsMonitor(),
        RichProgressBar() if _RICH_AVAILABLE else TQDMProgressBar(),
        RichModelSummary(max_depth=1) if _RICH_AVAILABLE else ModelSummary(max_depth=1)
    ]

    trainer = L.Trainer(
        accelerator = cfg['device'],
        devices = cfg['num_devices'],
        max_epochs = cfg['epochs'],
        callbacks = callbacks,
        logger = TensorBoardLogger(save_dir=cfg['savedir'], name="", log_graph=True)
    )

    if cfg['continue_from_ckpt'] is not None:
        print("Resume training state from %s"%(cfg['continue_from_ckpt']))

    trainer.fit(model, datamodule=datamodule, ckpt_path=cfg['continue_from_ckpt'])


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')

    Train()