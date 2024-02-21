import uproot
import pandas as pd
import warnings

from typing import List


def root2dataframe(file: str, tree: str, branches: List=None) -> pd.DataFrame:
    r"""Converting a :obj:`ROOT.TTree` to a :obj:`pandas.DataFrame`. This function
    is obsolete due to significant computing time.

    Args:
        file (str): input file path.
        tree (str): name of the :obj:`ROOT.TTree` to convert.
        branches (List): a list of branches needed. (default: :obj:`None`)

    :rtype: :class:`pandas.DataFrame`
    """

    warnings.warn("This function is obsolete due to significant computing time.", DeprecationWarning)

    df = pd.DataFrame()

    with uproot.open(file)[tree] as T:
        if branches == None:
            branches = T.keys()

        for branch in branches:
            df.insert(0, branch, T[branch].array(library="np"))

    return df

