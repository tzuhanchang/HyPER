import json
import warnings

from argparse import Namespace
from typing import Optional


class Settings(Namespace):
    def __init__(self):
        super(Settings, self).__init__()

        # Datasets
        self.db_config: str = ""
        self.train_data = ""
        self.valid_data   = ""
        self.all_matched: bool = False

        # Message Passing
        self.message_passing_recurrent: int = 3
        self.message_feats: int = 64
        self.contraction_feats: int = 64
        self.hyperedge_order: int = 3
        self.dropout: float = 0.01

        # Optimizer & loss
        self.optimizer: str = "Adam"
        self.learning_rate: float = 1e-3
        self.criterion_edge: str = "BCE"
        self.criterion_hyperedge: str = "BCE"
        self.alpha: int = 0.5
        self.loss_reduction: str = 'mean'
        self.loss_scale: float = 1.0

        # Training
        self.epochs: int = 500
        self.batch_size: int = 5000
        self.train_val_split: float = 0.9

        # Device
        self.device: str = "gpu"
        self.num_device: int = 1
        self.num_dataloader_workers: int = 1
        self.savedir: str = "./subgraph_logs"

    def update_settings(self, updated):
        for key, value in updated.items():
            if key in self.__dict__:
                setattr(self, key, type(self.__dict__[key])(value))

    def load(self, file: str):
        with open(file, 'r') as f:
            self.update_settings(json.load(f))

    def save(self, file: str):
        to_dump = {}
        for key, value in self.__dict__.items():
            to_dump.update({key: value})

        with open(file, 'w') as f:
            json.dump(to_dump, f, indent=4)


    def show(self, tune: Optional[bool] = False):
        # dismiss training related variables
        # and only keep graph related variable during graph building.
        tune_vars    = ['message_passing_recurrent', 'message_feats', 'contraction_feats', 'dropout',
                        'learning_rate', 'alpha']

        try:
            from rich import get_console
            from rich.table import Table

            default_options = self.__class__().__dict__
            console = get_console()

            rows = {}
            for key, value in self.__dict__.items():
                rows.update({key: (str(value), "green" if value != default_options[key] else None)})

            if tune is True:
                table = Table(title="Hyperparameters to Tune",header_style="orange1")
                table.add_column("Settings", justify="left")
                table.add_column("Default Value", justify="left")
                for key, value in rows.items():
                    if key in tune_vars:
                        table.add_row(key, value[0], style=value[1])
                    else:
                        continue
            else:
                table = Table(header_style="orange1")
                table.add_column("Settings", justify="left")
                table.add_column("Value", justify="left")
                for key, value in rows.items():
                    table.add_row(key, value[0], style=value[1])


            console.print(table)

        except ImportError:
            warnings.warn("Current console does not support `rich`. Check configure file for setttings.")