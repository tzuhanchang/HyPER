from torch_geometric.data import Data, HeteroData
from torch_geometric.transforms import BaseTransform
from typing import List, Union


class TransformFeatures(BaseTransform):
    r"""Transform features according to a list of functions.

    Args:
        attrs (List[str]): The names of attributes to transform.
        transforms (List[List[callable]]): A list of functions 
            used to transform features.
    """
    def __init__(
        self,
        attrs: List[str],
        transforms: List[List[callable]]
    ):
        self.attrs = attrs
        self.transforms = transforms
        assert len(self.attrs) == len(self.transforms)

    def forward(
        self,
        data: Union[Data, HeteroData],
    ) -> Union[Data, HeteroData]:
        for store in data.stores:
            attr_idx = 0
            for key, value in store.items(*self.attrs):
                if value.numel() > 0:
                    # Check the length of the method vector is smaller than the
                    # attribute tensor at `dim=1`.
                    assert len(self.transforms[attr_idx]) <= value.size(1)
                    for i in range(len(self.transforms[attr_idx])):
                        value[:,i] = self.transforms[attr_idx][i](value[:,i])
                    store[key] = value
                    attr_idx += 1
        return data