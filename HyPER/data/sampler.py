from torch.utils.data import Sampler
from typing import List

class EventSampler(Sampler[int]):
    r"""A simple dataset sampler. Selecting events according to
    the indices provided in :obj:`data`.

    Args:
        data (List[int]): list of indices.
    """
    def __init__(self, data: List[int]) -> None:
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __iter__(self):
        yield from self.data