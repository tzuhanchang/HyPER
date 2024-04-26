import yaml
import warnings

from pytorch_lightning import LightningDataModule
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from typing import Optional

from HyPER.data import GraphDataset, EventSampler


class HyPERDataModule(LightningDataModule):
    r"""HyPER Data Module encapsulates all the steps needed to
    process data.

    Args:
        db_config (str): dataset configuration file.
        train_set (optional, str): training dataset path.(default :obj:`None`)
        val_set (optional, str): validation dataset path. (default :obj:`None`)
        predict_set (optional, str): predict dataset path. (default :obj:`None`)
        batch_size (optional, int): number of samples per batch to load. (default :obj:`128`)
        max_n_events (optional, int): maximum number of events used in training. (default :obj:`-1`)
        percent_valid_samples (optional, float): fraction of dataset to use as validation samples. (default :obj:`0.005`)
        all_matched (optional, bool): only select fully matched events. (default :obj:`False`)
        drop_last (optional, bool): drop the last incompleted batch. (default :obj:`False`)
        num_workers (optional, int): loading data into memory with number of cpu workers. (default :obj:`0`)
        pin_memory (optional, bool): use memory pinning. (default :obj:`False`)
        persistent_workers (optional, bool): use the pervious workers. (default :obj:`True`)
    """
    def __init__(
        self,
        db_config: str,
        train_set: Optional[str] = None,
        val_set: Optional[str] = None,
        predict_set: Optional[str] = None,
        batch_size: Optional[int] = 128,
        max_n_events: Optional[int] = -1,
        percent_valid_samples: Optional[float] = 0.05,
        all_matched: Optional[bool] = False,
        drop_last: Optional[bool] = False,
        num_workers: Optional[int] = 0,
        pin_memory: Optional[bool] = True,
        persistent_workers: Optional[bool] = True
    ):
        super().__init__()

        self.db_config   = db_config
        self.train_set   = train_set
        self.val_set     = val_set
        self.predict_set = predict_set
        self.all_matched = all_matched
        self.batch_size  = batch_size
        self.num_workers = num_workers
        self.pin_memory  = pin_memory
        self.drop_last   = drop_last
        self.max_n_events = max_n_events
        self.persistent_workers    = persistent_workers
        self.percent_valid_samples = percent_valid_samples

        with open(db_config, 'r') as db_cfg:
            cf = yaml.safe_load(db_cfg)

            self.node_in_channels = len(list(cf['INPUTS']['Features'].keys()))
            self.edge_in_channels = 4
            self.global_in_channels  = len(list(cf['INPUTS']['global'].keys()))

        self.index_range = None

    def setup(self, stage: str):
        self.train_data = None
        self.val_data   = None
        if self.train_set is not None:
            if self.val_set is None or self.val_set == "" or self.train_set == self.val_set:
                print(f"Creating validation set using {round(self.percent_valid_samples*100,2)}% of the file.")

                if self.all_matched:
                    data = GraphDataset(path=self.train_set, configs=self.db_config, use_index_select=True)
                else:
                    data = GraphDataset(path=self.train_set, configs=self.db_config)

                self.train_data, self.val_data = random_split(data, [1-self.percent_valid_samples, self.percent_valid_samples])

            else:
                if self.all_matched is True:
                    train_data = GraphDataset(path=self.train_set, configs=self.db_config, use_index_select=True)
                    val_data = GraphDataset(path=self.val_set, configs=self.db_config, use_index_select=True)
                else:
                    train_data = GraphDataset(path=self.train_set, configs=self.db_config)
                    val_data = GraphDataset(path=self.val_set, configs=self.db_config)

                self.train_data = train_data
                self.val_data   = val_data

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

        self.predict_data = None
        if self.predict_set is not None:
            if self.all_matched:
                self.predict_data = GraphDataset(path=self.predict_set, configs=self.db_config, use_index_select=True)
            else:
                self.predict_data = GraphDataset(path=self.predict_set, configs=self.db_config)

        if self.train_data is None and self.val_data is None and self.predict_data is None:
            raise ValueError("No datasets have been provided. Abort!")

        try:
            from rich import get_console
            from rich.table import Table

            console = get_console()
            table = Table(title="Dataset Status",header_style="orange1")
            table.add_column("Name", justify="left")
            table.add_column("Value", justify="left")
            table.add_row("All matched only", str(self.all_matched))
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
        return DataLoader(self.predict_data, batch_size=self.batch_size, follow_batch=['edge_attr_s', 'edge_index_h'],
                          pin_memory=self.pin_memory, num_workers=self.num_workers, persistent_workers=self.persistent_workers)