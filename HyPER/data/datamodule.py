import warnings

from lightning import LightningDataModule
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from typing import Optional

from HyPER.data import GraphDataset, EventSampler, _check_dataset


class HyPERDataModule(LightningDataModule):
    r"""HyPER Data Module encapsulates all the steps needed to
    process data.

    Args:
        config (str): path to the network configuration file.
        train_set (optional, str): training dataset path.(default :obj:`None`)
        val_set (optional, str): validation dataset path. (default :obj:`None`)
        predict_set (optional, str): predict dataset path. (default :obj:`None`)
        batch_size (optional, int): number of samples per batch to load. (default :obj:`128`)
        max_n_events (optional, int): maximum number of events used in training. (default :obj:`-1`)
        percent_valid_samples (optional, float): fraction of dataset to use as validation samples. (default :obj:`0.005`)
        drop_last (optional, bool): drop the last incompleted batch. (default :obj:`False`)
        num_workers (optional, int): loading data into memory with number of cpu workers. (default :obj:`0`)
        pin_memory (optional, bool): use memory pinning. (default :obj:`False`)
        persistent_workers (optional, bool): use the pervious workers. (default :obj:`True`)
    """
    def __init__(
        self,
        config: str,
        batch_size: Optional[int] = 128,
        num_workers: Optional[int] = 0,
        pin_memory: Optional[bool] = True,
        persistent_workers: Optional[bool] = True
    ):
        super().__init__()

        self.cfg         = config
        self.train_set   = self.cfg['Dataset']['train_set']
        self.val_set     = self.cfg['Dataset']['val_set']
        self.predict_set = self.cfg['Dataset']['predict_set']
        self.batch_size  = batch_size
        self.num_workers = num_workers
        self.pin_memory  = pin_memory
        self.drop_last   = self.cfg['Dataset']['drop_last']
        self.max_n_events = self.cfg['Dataset']['max_n_events']
        self.persistent_workers    = persistent_workers
        self.percent_valid_samples = 1 - float(self.cfg['Dataset']['train_val_split'])

        self.train_set_params = _check_dataset(self.train_set, self.cfg, 'train') if self.train_set is not None else None
        self.val_set_params = _check_dataset(self.val_set, self.cfg, 'train') if self.val_set is not None else None
        self.predict_set_params = _check_dataset(self.predict_set, self.cfg, 'train') if self.predict_set is not None else None

        self.node_in_channels   = len(self.cfg['Dataset']['node_features'])+1
        self.edge_in_channels   = 4
        self.global_in_channels = len(self.cfg['Dataset']['global_features'])

        self.index_range = None

    def setup(self, stage: str):
        self.train_data   = None
        self.val_data     = None
        self.predict_data = None

        if self.train_set is not None:
            if self.val_set is None or self.val_set == "" or self.train_set == self.val_set:
                print(f"Creating validation set using {round(self.percent_valid_samples*100,2)}% of the file.")
                data = GraphDataset(
                    path=self.train_set,
                    config=self.cfg,
                    mode='train',
                    _params=self.train_set_params
                )
                self.train_data, self.val_data = random_split(data, [1-self.percent_valid_samples, self.percent_valid_samples])
            else:
                self.train_data = GraphDataset(
                    path=self.train_set,
                    config=self.cfg,
                    mode='train',
                    _params=self.train_set_params
                )
                self.val_data = GraphDataset(
                    path=self.val_set,
                    config=self.cfg,
                    mode='train',
                    _params=self.val_set_params
                )

            # Limit training dataset size to self.max_n_events
            if self.max_n_events == -1:
                pass
            elif self.max_n_events > len(self.train_data):
                warnings.warn("`max_n_events` large than the dataset, use all events in the dataset.")
                pass
            elif self.max_n_events > 0 and self.max_n_events <= len(self.train_data):
                self.index_range = list(range(self.max_n_events))
            else:
                pass

        if self.predict_set is not None:
            self.predict_data = GraphDataset(
                path=self.predict_set,
                config=self.cfg,
                mode='test',
                _params=self.predict_set_params
            )

        if self.train_data is None and self.val_data is None and self.predict_data is None:
            raise ValueError("No datasets have been provided. Abort!")

        try:
            from rich import get_console
            from rich.table import Table

            console = get_console()
            table = Table(title="Dataset Status",header_style="orange1")
            table.add_column("Name", justify="left")
            table.add_column("Value", justify="left")
            table.add_row("Drop last batch", str(self.drop_last))
            if self.train_data is not None:
                if self.index_range is not None:
                    table.add_row("Training samples", str(len(self.index_range)))
                else:
                    table.add_row("Training samples", str(len(self.train_data)))
            if self.val_data is not None:
                table.add_row("Validation samples", str(len(self.val_data)))
            if self.predict_data is not None:
                table.add_row("Prediction samples", str(len(self.predict_data)))
            table.add_row("N node attributes", str(self.node_in_channels))
            table.add_row("N edge attributes", str(self.edge_in_channels))
            table.add_row("N glob attributes", str(self.global_in_channels))
            console.print(table)

        except ImportError:
            pass

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            follow_batch=['edge_attr_s', 'edge_index_h'],
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            drop_last=self.drop_last,
            sampler=EventSampler(self.index_range) if self.index_range is not None else None
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            follow_batch=['edge_attr_s', 'edge_index_h'],
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            drop_last=self.drop_last,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_data,
            batch_size=self.batch_size,
            follow_batch=['edge_attr_s', 'edge_index_h'],
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers
        )