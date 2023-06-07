import pandas as pd
from pandas import DataFrame


def get_feature_block(data: DataFrame, index='0'):
    """
    Get static or dynamic feature names from a multi-index DataFrame
    """
    idx = pd.IndexSlice
    new_df = data.loc[:, idx[index, :]]
    new_df.columns = new_df.columns.droplevel()

    return new_df
