import yaml

from pytorch_lightning import LightningDataModule
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from typing import Optional

from HyPER.data import GraphDataset


class HyPERDataModule(LightningDataModule):
    r"""HyPER Data Module encapsulates all the steps needed to
    process data.

    Args:
        db_config (str): dataset configuration file.
        train_set (str): training dataset path.
        val_set (optional, str): validation dataset path. (default :obj:`None`)
        batch_size (optional, int): number of samples per batch to load. (default :obj:`int`=128)
        percent_valid_samples (optional, float): fraction of dataset to use as validation samples. (default :obj:`float`=0.005)
        mask (optional, Sequence[int]): indices to mask over. (default :obj:`None`)
        num_workers (optional, int): loading data into memory with number of cpu workers. (default :obj:`int`=0)
        pin_memory (optional, bool): use memory pinning. (default :obj:`bool`=False)
        persistent_workers (optional, bool): use the pervious workers. (default :obj:`bool`=True)
    """
    def __init__(
        self,
        db_config: str,
        train_set: Optional[str] = None,
        val_set: Optional[str] = None,
        batch_size: Optional[int] = 128,
        percent_valid_samples: Optional[float] = 0.05,
        all_matched: Optional[bool] = False,
        num_workers: Optional[int] = 0,
        pin_memory: Optional[bool] = True,
        persistent_workers: Optional[bool] = True
    ):
        super().__init__()

        self.db_config   = db_config
        self.train_set   = train_set
        self.val_set     = val_set
        self.all_matched = all_matched
        self.batch_size  = batch_size
        self.num_workers = num_workers
        self.pin_memory  = pin_memory
        self.persistent_workers    = persistent_workers
        self.percent_valid_samples = percent_valid_samples

        with open(db_config, 'r') as db_cfg:
            cf = yaml.safe_load(db_cfg)

            self.node_in_channels = len(list(cf['INPUTS']['Features'].keys()))
            self.edge_in_channels = 4
            self.global_in_channels  = len(list(cf['INPUTS']['global'].keys()))

    def setup(self, stage: str):
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
                val_data   = GraphDataset(path=self.val_set, configs=self.db_config)

            self.train_data = train_data
            self.val_data   = val_data

        try:
            from rich import get_console
            from rich.table import Table

            console = get_console()
            table = Table(title="Dataset Status",header_style="orange1")
            table.add_column("", justify="left")
            table.add_column("", justify="left")
            table.add_row("All matched only", str(self.all_matched))
            table.add_row("Training samples", str(len(self.train_data)))
            table.add_row("Validation samples", str(len(self.val_data)))
            table.add_row("N node attributes", str(self.node_in_channels))
            table.add_row("N edge attributes", str(self.edge_in_channels))
            table.add_row("N glob attributes", str(self.global_in_channels))
            console.print(table)

        except ImportError:
            pass

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, follow_batch=['edge_attr_s'],
                          pin_memory=self.pin_memory, num_workers=self.num_workers, persistent_workers=self.persistent_workers)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, follow_batch=['edge_attr_s'],
                          pin_memory=self.pin_memory, num_workers=self.num_workers, persistent_workers=self.persistent_workers)