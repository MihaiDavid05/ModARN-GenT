import os
import pandas as pd
import numpy as np
from modn_data import DATA_ABS_PATH
from modn.datasets.utils import get_feature_block
import argparse


def get_cli_args(parser):
    parser.add_argument('--dataset_type', type=str, required=True, help='Dataset type: small or toy')
    parser.add_argument('--reset_state', action='store_true',
                        help='Whether to reset state at each timestep at validation time')
    parser.add_argument('--clean_rows', action='store_true',
                        help='Clean possible abnormal rows that are full of zeros or not')
    parser.add_argument('--output_columns', action='store_true',
                        help='Write all dynamic features to a txt file or not.'
                             'Useful when defining features in mimic.py')
    parser.add_argument('--output_unique_static', action='store_true',
                        help='Print lists of unique values for each static features or not.')

    return parser.parse_args()


if __name__ == "__main__":

    args = get_cli_args(argparse.ArgumentParser())
    dataset_type = args.dataset_type
    reset_state = args.reset_state
    clean_rows = args.clean_rows
    output_columns = args.output_columns
    output_unique_static = args.output_unique_static

    filename = "MIMIC_data_labels_{}".format(dataset_type)
    data_path = os.path.join(DATA_ABS_PATH, f"{filename}.csv")
    save_path = os.path.join(DATA_ABS_PATH, f"new_{filename}.csv")

    # Read dataframe with multiindex columns
    df = pd.read_csv(data_path, header=[0, 1])

    if output_unique_static:
        # Get unique values for the static variables, which are under -1 level
        new_df_stat = get_feature_block(df, index='-1')

        # These unique values are used in the coarse_cleaning function in mimic.py
        print("All unique insurance values are {}".format(new_df_stat['insurance'].dropna().unique()))
        print("All unique gender values are {}".format(new_df_stat['gender'].dropna().unique()))
        print("All unique age values are {}".format(sorted(new_df_stat['Age'].dropna().unique())))
        if dataset_type == 'small':
            print("All unique ethnicity values are {}".format(new_df_stat['ethnicity'].dropna().unique()))

    # Write all dynamic features names to a file (for further defining them into mimic.py)
    if output_columns:
        # Get all features for one timestamp (here 0)
        new_df_dyn = get_feature_block(df, index='0')

        output_file = os.path.join(DATA_ABS_PATH, 'cont_columns_{}.txt'.format(dataset_type))
        with open(output_file, 'w') as f:
            np.savetxt(f, new_df_dyn.columns.values, fmt='\'%s\'', newline=',\n')

    if clean_rows:
        # Select all continuous features
        new_cont_df = get_feature_block(df, list(map(str, list(range(48)))))
        # Check if row full of zeros and drop that rows
        df = df.loc[~(new_cont_df == 0).all(axis=1)]
        df.to_csv(data_path, index=False)
        # Save dataframe
        print("Data cleaned!")
