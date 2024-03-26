import hydra
import optuna

import pytorch_lightning as L

from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning.loggers import TensorBoardLogger
from omegaconf import DictConfig

from HyPER.data import HyPERDataModule
from HyPER.models import HyPERModel


NUM_TRIALS = 100
NUM_EPOCHS = 10
CONFIGS = None


def objective(trial: optuna.trial.Trial) -> float:
    r"""An `objective` function to be optimised.

    Args:
        trial (optuna.trial.Trial): is used to suggest hyperparameter values.
        option_file (str, optional): `.json` file, stores training related parameters. (default: :obj:`str`=None)
    """
    # We optimise these hyperparameters:
    num_message_layers = trial.suggest_int("num_message_layers", 3, 8, step=1)
    message_feats      = trial.suggest_int("message_feats", 32, 256, step=16)
    hyperedge_feats    = trial.suggest_int("hyperedge_feats", 32, 256, step=16)
    dropout            = trial.suggest_float("dropout", 0.001, 0.1)
    learning_rate      = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    alpha              = trial.suggest_float("alpha", 0.0, 1.0)

    datamodule = HyPERDataModule(
        db_config = CONFIGS['db_config'],
        train_set = CONFIGS['train_set'],
        val_set = CONFIGS['val_set'],
        batch_size = CONFIGS['batch_size'],
        percent_valid_samples = 1 - float(CONFIGS['train_val_split']),
        num_workers = CONFIGS['num_workers'],
        pin_memory = True if CONFIGS['device'] == "gpu" else False,
        all_matched = CONFIGS['all_matched']
    )

    try:
        model = HyPERModel(
            node_in_channels = datamodule.node_in_channels,
            edge_in_channels = datamodule.edge_in_channels,
            global_in_channels = datamodule.global_in_channels,
            message_feats = message_feats,
            dropout = dropout,
            message_passing_recurrent = num_message_layers,
            contraction_feats = hyperedge_feats,
            hyperedge_order = CONFIGS['hyperedge_order'],
            criterion_edge = CONFIGS['criterion_edge'],
            criterion_hyperedge = CONFIGS['criterion_hyperedge'],
            optimizer = CONFIGS['optimizer'],
            lr = learning_rate,
            alpha = alpha,
            reduction = CONFIGS['loss_reduction']
        )

        trainer = L.Trainer(
            accelerator = CONFIGS['device'],
            devices = CONFIGS['num_devices'],
            max_epochs = NUM_EPOCHS,
            enable_checkpointing=False,
            logger = TensorBoardLogger(save_dir="tuner_output", name="", log_graph=True),
            callbacks=[PyTorchLightningPruningCallback(trial, monitor="fuzzy_accuracy/validation_accuracy")]
        )

        trainer.fit(model, datamodule=datamodule)

    except Exception as e:
        print(f"Trial {trial.number} failed: {str(e)}")
        trial.set_user_attr("failed", True)
        trial.set_user_attr("error_message", str(e))
        raise optuna.TrialPruned()

    return trainer.callback_metrics["fuzzy_accuracy/validation_accuracy"].item()


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def Tune(cfg : DictConfig) -> None:
    global CONFIGS
    CONFIGS = cfg

    pruner = optuna.pruners.MedianPruner()

    study = optuna.create_study(
        storage        = "sqlite:///db.sqlite3",
        study_name     = "HyPER-tune",
        direction      = "maximize",
        load_if_exists = True,
        pruner         = pruner
    )

    study.optimize(objective, n_trials = NUM_TRIALS)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))


if __name__ == "__main__":
    Tune()