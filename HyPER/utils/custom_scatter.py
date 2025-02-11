import torch
from torch import Tensor
from torch_scatter import scatter

'''
This function achieves the same behavior as torch_scatter.scatter.
Only 2D tensors are supported!
The parameters are the same as torch_scatter.scatter
The way this function works is:
1. The shape of the output tensor is determined by:
    One dimention size is extracted from:
    - The dimention the reduction is taking place over.
        - If dim = 0, the number of columns in the output is equal to n_columns(src).
        - If dim = 1, the number of rows in the output is equal to n_rows(src).
    The other dimention size s extracted from:
    - Whatever is greater between dim_size or the maximum index in the index tensor.
        - If dim_size is greater than the maximum index, the output tensor will be padded.
    
2. Once the shape of the output tensor is determined, a tensor (base_tensor) is created with all elements set to the return value of fn(create_include_self).
    - This trick is needed to be able to use max and min reduction properly.
3. The scatter_reduce_ function is called on the created tensor.
4. Finally, we have to deal with the values that could be still inf if the ouput tensor was padded by dim_size.
'''
def true_fn(dim_size, max_index):
    return dim_size.clone()

def false_fn(dim_size, max_index):
    return max_index.clone() + 1

def create_include_self(reduce: str):
    if reduce == 'amax':
        include_self_tensor = -torch.inf
    elif reduce == 'amin':
        include_self_tensor = torch.inf
    else:
        include_self_tensor = 0
    return include_self_tensor

def custom_scatter(src: Tensor,
               index: Tensor,
                dim: int ,
                dim_size: int,
                reduce: str ='sum') -> Tensor:
    
    if reduce not in ['sum', 'mean', 'amax', 'amin']:
        raise ValueError(f"Unsupported reduce operation: {reduce}")

    if dim != 0 and dim != 1:
        raise ValueError(f"Unsupported dim: {dim}")
    
    # Step 1. Determine the shape of the output tensor.
    # One of this two will be the input for one dimention size.
    n_rows = src.shape[0]
    n_cols = src.shape[1]

    max_index = index.max().item()
    torch._check_is_size(max_index) # This check is necessary to avoid Dynamo export errors.
    dim_siz_t = torch.tensor(dim_size) # Convert dim_size to a tensor, needed because torch.cond requires tensors.
    max_index_t = index.max() # Same conversion as above.
    # Determine the other dimention size.
    true_size_t = torch.cond(dim_size > max_index+1, true_fn, false_fn, (dim_siz_t, max_index_t))
    true_size = true_size_t.item() # Convert from tensor to int.
    torch._check_is_size(true_size) # This check is necessary to avoid Dynamo export errors.

    # Step 2. Create the base tensor. Split in two cases of reduction.
    if dim == 0:
        assert index.shape[0] == n_rows # This always needs to be true for the reduction to make sense.
        index_tensor = index.repeat(n_cols,1)
        base_tensor = torch.full((true_size, n_cols), create_include_self(reduce))
        # Step 3.
        output = base_tensor.scatter_reduce_(0, index_tensor.T, src,reduce=reduce)
            
    if dim == 1:
        assert index.shape[0] == n_cols # This always needs to be true for the reduction to make sense.
        index_tensor = index.repeat(n_rows,1)
        base_tensor = torch.full((n_rows, true_size), create_include_self(reduce))
        # Step 3.
        output = base_tensor.scatter_reduce_(1, index_tensor, src,reduce=reduce)

    # Step 4.
    output.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
    return output

if __name__ == "__main__":
    print("Testing custom scatter")
    print("Test reduction on dim=0")
    print("Input tensor:")
    a = torch.randn(6,7)
    print(a)
    print("Index tensor:") 
    index = torch.tensor([0,0,1,0,1,2])
    print(index)
    print("Custom scatter result:")
    out = custom_scatter(a,index,dim_size=11,dim=0,reduce='amax')
    print(out)
    print("torch_scatter result:")
    out_two = scatter(a, index, dim_size=11, dim=0, reduce="max")
    print(out_two)
    assert torch.allclose(out, out_two)

    print("")
    print("Test reduction on dim=1")
    print("Input tensor:")
    print(a)
    print("Index tensor:") 
    index = torch.tensor([0,0,1,0,1,2,2])
    print(index)
    print("Custom scatter result:")
    out = custom_scatter(a,index,dim_size=4,dim=1,reduce='amax')
    print(out)
    print("torch_scatter result:")
    out_two = scatter(a, index, dim_size=4, dim=1, reduce="max")
    print(out_two)
    assert torch.allclose(out, out_two)