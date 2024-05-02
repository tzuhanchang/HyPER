import h5py

import numpy as np
import pandas as pd


def ResultWriter(result: pd.DataFrame, dest: str, mode: str = 'a') -> None:
    r"""Write :obj:`pandas.DataFrame` to a HDF5 file :obj:`dest`.

    Args:
        results (pandas.DataFrame): a dataframe containing unbatched 
            network predictions.
        dest (str): destination HDF5 file.
        mode (optional, str): writting mode (default: :obj:`str`='a')
    """
    f = h5py.File(dest, mode)

    out = f.create_group('OUTPUT')

    keys = [ x for x in list(result.keys()) if 'HyPER_best_' in x]  # Ignore RAW outputs
    num_entries = len(result)

    for key in keys:
        dt = np.dtype([ (f'constituent{idx}', np.int64) for idx in range(len(result[key][0])) ])
        data = np.full((num_entries, 1), -9, dtype=dt)
        for i in range(num_entries):
            data[i] = tuple(result[key][i])

        out.create_dataset(key, data=data)

    f.close()