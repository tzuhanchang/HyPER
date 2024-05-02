from torch import Tensor
from torch_geometric.utils import scatter


def softmax(
    src: Tensor,
    index: Tensor,
    dim_size: int,
    dim: int = 0,
) -> Tensor:
    r"""This function is a modified version of `torch_geometric.utils.softmax`,
    which solves value unconstrain error raised by `torch.onnx.dynamo_export`.

    Args:
        src (Tensor): The source tensor.
        index (LongTensor): The indices of elements for applying the
            softmax.
        dim_size (int): Automatically create output tensor with size
            :attr:`dim_size` in the first dimension. If set to :attr:`None`, a
            minimal sized output tensor is returned.
        dim (int, optional): The dimension in which to normalize.
            (default: :obj:`0`)

    :rtype: :class:`Tensor`
    """
    src_max = scatter(src.detach(), index, dim, dim_size=dim_size, reduce='max')
    out = src - src_max.index_select(dim, index)
    out = out.exp()
    out_sum = scatter(out, index, dim, dim_size=dim_size, reduce='sum') + 1e-16
    out_sum = out_sum.index_select(dim, index)

    return out / out_sum