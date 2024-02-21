import json
import warnings

import numpy as np

from argparse import Namespace
from typing import Optional


class Settings(Namespace):
    def __init__(self):
        super(Settings, self).__init__()

        # Datasets
        self.train_data = ""
        self.valid_data   = ""

        # Graphs
        self._objects:         str = "['jet']"
        self._node_features:   str = "['pt','eta','phi','e','bTag']"
        self._node_scales:     str = "[lambda x: np.log(x), lambda x: x/np.pi, lambda x: x/np.pi, lambda x: np.log(x), lambda x: x]"
        self._global_features: str = "['njet','nbTagged']"
        self._global_scales:   str = "[lambda x: x/6, lambda x: x/2]"
        self.truth_ID:         str = 'jet_truthmatch'
        self._truth_patterns:  str = "[[1,2,3],[4,5,6],[2,3],[5,6]]"

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
        self.in_memory_dataset: bool = False
        self.savedir: str = "./subgraph_logs"

        self.special_vars = ['objects', 'node_features', 'node_scales', 'global_features', 'global_scales',
                             'truth_patterns', 'special_vars']

        self.get_value()

    def get_value(self):
        r"""These variable types cannot be stored in a .json file.
        This function recovers their real values from strings.
        """
        self.objects = eval(self._objects)
        self.node_features = eval(self._node_features)
        self.node_scales = eval(self._node_scales)
        self.global_features = eval(self._global_features)
        self.global_scales = eval(self._global_scales)
        self.truth_patterns = eval(self._truth_patterns)

    def update_settings(self, updated):
        for key, value in updated.items():
            if key in self.__dict__:
                if key in self.special_vars:
                    setattr(self, '_'+key, str(value))
                else:
                    setattr(self, key, type(self.__dict__[key])(value))

        self.get_value()

    def load(self, file: str):
        with open(file, 'r') as f:
            self.update_settings(json.load(f))

    def save(self, file: str):
        to_dump = {}
        for key, value in self.__dict__.items():
            if key not in self.special_vars:
                if key[0] == '_':
                    to_dump.update({key[1:]: str(value)})
                else:
                    to_dump.update({key: value})
            else:
                continue

        with open(file, 'w') as f:
            json.dump(to_dump, f, indent=4)


    def show(self, graphs: Optional[bool] = False, tune: Optional[bool] = False):
        # dismiss training related variables
        # and only keep graph related variable during graph building.
        graphs_vars  = ['objects', 'node_features', 'node_scales', 'global_features', 'global_scales',
                        'truth_ID', 'truth_patterns']
        tune_vars    = ['message_passing_recurrent', 'message_feats', 'contraction_feats', 'dropout',
                        'learning_rate', 'alpha']

        try:
            from rich import get_console
            from rich.table import Table

            default_options = self.__class__().__dict__
            console = get_console()

            rows = {}
            for key, value in self.__dict__.items():
                if key not in self.special_vars:
                    if key[0] == '_':
                        rows.update({key[1:]: (str(value)[1:-1], "green" if value != default_options[key] else None)})
                    else:
                        rows.update({key: (str(value), "green" if value != default_options[key] else None)})
                else:
                    continue

            if graphs is True:
                table = Table(title="Hyperedge Definations",header_style="orange1")
                table.add_column("Settings", justify="left")
                table.add_column("Value", justify="left")
                for key, value in rows.items():
                    if key in graphs_vars:
                        table.add_row(key, value[0], style=value[1])
                    else:
                        continue
            elif tune is True:
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
