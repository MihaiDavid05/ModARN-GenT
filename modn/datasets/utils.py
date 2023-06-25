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


def get_default_static_data(data):
    """ Get static features information for all patients"""

    data_df = data._data.features.iloc[data._indices]
    static_df = get_feature_block(data_df, index='-1')
    default_info = list(static_df.itertuples(index=False))

    return default_info

