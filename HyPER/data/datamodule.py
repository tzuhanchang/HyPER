from pytorch_lightning import LightningDataModule
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from typing import Optional

from HyPER.data import GraphDataset


class HyPERDataModule(LightningDataModule):
    r"""HyPER Data Module encapsulates all the steps needed to 
    process data.

    Args:
        train_dir (str): training dataset path.
        val_dir (optional, str): validation dataset path. (default :obj:`None`)
        batch_size (optional, int): number of samples per batch to load. (default :obj:`int`=128)
        percent_valid_samples (optional, float): fraction of dataset to use as validation samples. (default :obj:`float`=0.005)
        num_workers (optional, int): loading data into memory with number of cpu workers. (default :obj:`int`=0)
        pin_memory (optional, bool): use memory pinning. (default :obj:`bool`=False)
        all_matched (optional, bool): only load in fully matched events. (default :obj:`bool`=False)
        num_samples (optional, int): maximum number of samples to load. (default :obj:`int`=None)
    """
    def __init__(self, train_dir: str, val_dir: Optional[str] = None,
                 batch_size: Optional[int] = 128, 
                 percent_valid_samples: Optional[float] = 0.05,
                 in_memory: Optional[bool] = False,
                 num_workers: Optional[int] = 0,
                 pin_memory: Optional[bool] = True,
                 all_matched: Optional[bool] = False,
                 num_samples: Optional[int] = None):
        super().__init__()

        self.train_dir   = train_dir
        self.val_dir     = val_dir
        self.batch_size  = batch_size
        self.in_memory   = in_memory
        self.num_workers = 0 if self.in_memory else num_workers
        self.pin_memory  = pin_memory
        self.all_matched = all_matched
        self.num_samples = num_samples
        self.percent_valid_samples = percent_valid_samples

    def setup(self, stage: str):
        if self.val_dir is None or self.val_dir == "" or self.train_dir == self.val_dir:
            print(f"Creating validation set using {round(self.percent_valid_samples*100,2)}% of the file.")
            data = GraphDataset(root=self.train_dir)

            if self.all_matched is True:
                data = data[data.metadata['indices_allMatched']]

            if self.num_samples is not None and self.num_samples < len(data):
                data = data[0:self.num_samples]
            
            if self.in_memory is True:
                data = GraphDataset(graphs=data).load_into_memory()

            self.node_in_size = data[0].x_s.shape[1]
            self.edge_in_size = data[0].edge_attr_s.shape[1]
            self.global_in_size = data[0].u_s.shape[1]
            self.train_data, self.val_data = random_split(data, [1-self.percent_valid_samples, self.percent_valid_samples])

        else:
            train_data = GraphDataset(root=self.train_dir)
            val_data   = GraphDataset(root=self.val_dir)

            if self.all_matched is True:
                train_data = train_data[train_data.metadata['indices_allMatched']]
                val_data = val_data[val_data.metadata['indices_allMatched']]

            if self.in_memory is True:
                train_data = GraphDataset(graphs=train_data).load_into_memory()
                val_data = GraphDataset(graphs=val_data).load_into_memory()

            self.node_in_size = train_data[0].x_s.shape[1]
            self.edge_in_size = train_data[0].edge_attr_s.shape[1]
            self.global_in_size = train_data[0].u_s.shape[1]
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
            table.add_row("Inmemory dataset", str(self.in_memory))
            table.add_row("Training samples", str(len(self.train_data)))
            table.add_row("Validation samples", str(len(self.val_data)))
            table.add_row("N node attributes", str(self.node_in_size))
            table.add_row("N edge attributes", str(self.edge_in_size))
            table.add_row("N glob attributes", str(self.global_in_size))
            console.print(table)

        except ImportError:
            pass

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, follow_batch=['edge_attr_s'], 
                          pin_memory=self.pin_memory, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, follow_batch=['edge_attr_s'],
                          pin_memory=self.pin_memory, num_workers=self.num_workers)