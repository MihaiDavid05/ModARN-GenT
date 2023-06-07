import os
import pandas as pd
import numpy as np
from modn_data import DATA_ABS_PATH
from modn.datasets.utils import get_feature_block

# Choose type of MIMIC dataset: 'small' or 'big'
MIMIC_TYPE = 'small'
# Set this to True if you want to create a txt file with a List[str] of the continuous features names
OUTPUT_COLUMNS = False
# Set this to True if you want to print a List[str] of the unique values for each static feature
OUTPUT_UNIQUE_STATIC = False
# Clean abnormal rows
CLEAN_ROWS = True

if __name__ == "__main__":

    filename = "MIMIC_data_labels_{}".format(MIMIC_TYPE)
    data_path = os.path.join(DATA_ABS_PATH, f"{filename}.csv")
    save_path = os.path.join(DATA_ABS_PATH, f"new_{filename}.csv")

    # Read dataframe with multiindex columns
    df = pd.read_csv(data_path, header=[0, 1])

    if OUTPUT_UNIQUE_STATIC:
        # Get unique values for the static variables, which are under -1 level
        new_df_stat = get_feature_block(df, index='-1')

        # These unique values are used in the coarse_cleaning function in mimic.py
        print("All unique insurance values are {}".format(new_df_stat['insurance'].dropna().unique()))
        print("All unique gender values are {}".format(new_df_stat['gender'].dropna().unique()))
        print("All unique ethnicity values are {}".format(new_df_stat['ethnicity'].dropna().unique()))
        print("All unique age values are {}".format(sorted(new_df_stat['Age'].dropna().unique())))

    # Write all dynamic features names to a file (for further defining them into mimic.py)
    if OUTPUT_COLUMNS:
        # Get all features for one timestamp (here 0)
        new_df_dyn = get_feature_block(df, index='0')

        output_file = os.path.join(DATA_ABS_PATH, 'cont_columns_{}.txt'.format(MIMIC_TYPE))
        with open(output_file, 'w') as f:
            np.savetxt(f, new_df_dyn.columns.values, fmt='\'%s\'', newline=',\n')

    if CLEAN_ROWS:
        # Select all continuous features
        new_cont_df = get_feature_block(df, list(map(str, list(range(48)))))
        # Check if row full of zeros and drop that rows
        df = df.loc[~(new_cont_df == 0).all(axis=1)]
        df.to_csv(data_path, index=False)
        # Save dataframe
        print("Data cleaned!")


