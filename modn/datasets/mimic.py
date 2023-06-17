from typing import List
from typing import Tuple
from typing import Union

import numpy as np
import pandas as pd
import scipy.stats as ss
import torch
from pandas import DataFrame

from modn.datasets import DataPointMIMIC, ConsultationMIMIC, ObservationMIMIC, FeatureInfo, PatientDataset


class MIMICDataset(PatientDataset):

    # Define toy dataset features
    # feature_toy_static = ['Age', 'gender', 'insurance']
    # feature_toy = ['F1_constant', 'F2_early', 'F3_late', 'F4_narrow', 'F5_wide'] + feature_toy_static
    # target_toy = ["label"]
    # # feature_toy_cont = ['Age', 'F1_constant', 'F2_early', 'F3_late', 'F4_narrow', 'F5_wide']
    # feature_toy_cont = ['F1_constant']
    # feature_toy_cat = ['gender', 'insurance']

    feature_toy_static = ['Age', 'gender', 'insurance', 'label']
    feature_toy = ['F1_constant', 'F2_early', 'F3_late', 'F4_narrow', 'F5_wide'] + feature_toy_static
    target_toy = ["label"]
    feature_toy_cont = ['Age', 'F1_constant', 'F2_early', 'F3_late', 'F4_narrow', 'F5_wide']
    feature_toy_cat = ['gender', 'insurance', 'label']

    # Define small dataset features
    feature_small_static = ['Age', 'gender', 'ethnicity', 'insurance']
    feature_small = ['WBC', 'Chloride (serum)', 'Glucose (serum)', 'Magnesium', 'Sodium (serum)', 'BUN', 'Phosphorous',
                     'Anion gap', 'Potassium (serum)', 'HCO3 (serum)', 'Platelet Count', 'Prothrombin time', 'PTT',
                     'Lactic Acid'] + feature_small_static
    target_small = ["label"]
    feature_small_cat = ['ethnicity', 'gender', 'insurance']
    feature_small_cont = feature_small[:15]

    def __init__(
            self,
            csv_file: str,
            data_type: str,
            global_question_block: bool = False,
            remove_correlated_features: bool = False,
            use_feats_decoders: bool = False,
            normalise: bool = True
    ):
        self.global_question_block = global_question_block
        self.data = self._load_data(csv_file)
        self.data_type = data_type
        self._metadata = None
        self.use_feats_decoders = use_feats_decoders
        self.timestamps = len(self.data.columns.levels[0]) - 1
        self.nr_static_feats = len(getattr(self, 'feature_{}_static'.format(self.data_type)))
        self.normalise = normalise

        if self.use_feats_decoders:
            self.feature_features, self.target_features, self.feature_features_cat, self.feature_features_cont = \
                self.get_feature_names(use_feat_decoders=self.use_feats_decoders)
            self.features = self.data[self.feature_features]
            self.targets = self.data[self.target_features]
            self.features_cont = self.data[self.feature_features_cont]
            self.features_cat = self.data[self.feature_features_cat]
        else:
            self.feature_features_cat, self.feature_features_cont = [], []
            self.feature_features, self.target_features,\
                _, _ = self.get_feature_names(use_feat_decoders=self.use_feats_decoders)
            self.features = self.data[self.feature_features]
            self.targets = self.data[self.target_features]
            self.features_cont, self.features_cat = pd.DataFrame(), pd.DataFrame()

        print("Initial number of features: {}".format(len(self.feature_features)))
        if remove_correlated_features:
            self.feature_features = self._remove_correlated_features(
                self.feature_features
            )
        print("Features after correlation removal: {}".format(len(self.feature_features)))
        self.feature_info = self._init_feature_info(self.data)

        if self.use_feats_decoders:
            self.targets_cont = self.get_cont_targets()
            self.targets_cat = self.get_cat_targets()
        else:
            self.targets_cont, self.targets_cat = pd.DataFrame(), pd.DataFrame()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        features = self.features.iloc[idx]
        targets = self.targets.iloc[idx]

        if self.use_feats_decoders:
            targets_cont = self.targets_cont.iloc[idx]
            targets_cat = self.targets_cat.iloc[idx]
            targets = [targets.to_dict(), targets_cont.to_dict(), targets_cat.to_dict()]
        else:
            targets = targets.to_dict()

        if self.global_question_block:
            dp = DataPointMIMIC(
                consultation=ConsultationMIMIC(
                    question_blocks=[
                        [ObservationMIMIC(question, features[question])
                         for question in self._answered_questions(features)]
                    ]
                ),
                targets=targets,
            )
            return dp
        else:

            dp = DataPointMIMIC(
                consultation=ConsultationMIMIC(
                    question_blocks=[
                        [ObservationMIMIC(question, features[question]) for question in block]
                        for block in self._group_simultaneous_questions(self._answered_questions(features))
                    ]
                ),
                targets=targets,
            )

            return dp

    def get_cont_targets(self):
        """Build the continuous features target dataframe"""
        targets_cont = pd.DataFrame()
        if len(self.features_cont) > 0:
            targets_cont = self.features_cont.copy()
            for f in self.unique_features_cont:
                targets_cont.loc[:, (slice(None), f)] = targets_cont.loc[:, (slice(None), f)].bfill(axis=1)

        return targets_cont

    def get_cat_targets(self):
        """Build the categorical features target dataframe"""
        targets_cat = pd.DataFrame()
        if len(self.features_cat) > 0:
            targets_cat = self.features_cat.copy()
            for f in self.unique_features_cat:
                targets_cat.loc[:, (slice(None), f)] = targets_cat.loc[:, (slice(None), f)].bfill(axis=1)

        return targets_cat

    @staticmethod
    def _group_simultaneous_questions(questions: List[Tuple[str, str]]) -> List[List[Tuple[str, str]]]:
        """
        Takes an ordered list of questions asked. Groups those questions into time blocks.
        """
        groups_dict = {}
        groups = []
        timesteps = set()

        # Compute a dictionary where key is a timestamp and the value is a list of features from that timestamp
        for ts, name in questions:
            timesteps.add(int(ts))
            if ts not in groups_dict:
                groups_dict[ts] = [(ts, name)]
            else:
                groups_dict[ts].append((ts, name))

        # Form groups of features ordered in time, but random within groups.
        # Note: Cannot have groups with static features only!
        for i in list(sorted(timesteps)):
            # Add the static variables to each group
            if str(i) != '-1':
                groups.append(groups_dict[str(i)] + groups_dict['-1'])
        return groups

    @staticmethod
    def _answered_questions(feature_column: pd.Series) -> List[Tuple[str, str]]:
        return feature_column.dropna().index.to_list()

    def features_for_patient(self, patient_idx: Union[int, List[int]]) -> pd.DataFrame:
        return self.features.loc[patient_idx]

    def targets_for_patient(self, patient_idx: Union[int, List[int]]) -> pd.DataFrame:
        return self.targets.loc[patient_idx]

    def get_feature_names(self, use_feat_decoders: bool = False) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]],
                                                                          List[Tuple[str, str]], List[Tuple[str, str]]]:
        """Get names format of all types of features, suitable for the network"""

        self.unique_features = getattr(self, 'feature_{}'.format(self.data_type))
        if use_feat_decoders:
            self.unique_features_cat = getattr(self, 'feature_{}_cat'.format(self.data_type))
            self.unique_features_cont = getattr(self, 'feature_{}_cont'.format(self.data_type))
        else:
            self.unique_features_cat = []
            self.unique_features_cont = []
        new_feature_features = []
        new_feature_features_cont = []
        new_feature_features_cat = []

        # Features will be created in order of their timestamp/consultation
        for tmp in range(-1, self.timestamps, 1):
            if tmp == -1:
                for f in self.unique_features[-self.nr_static_feats:]:
                    new_feature_features.append((str(tmp), f))
                    if use_feat_decoders:
                        if self.data.dtypes[('-1', f)] == "category" and f in self.unique_features_cat:
                            new_feature_features_cat.append((str(tmp), f))
                        elif self.data.dtypes[('-1', f)] != "category" and f in self.unique_features_cont:
                            new_feature_features_cont.append((str(tmp), f))
            else:
                for f in self.unique_features[:-self.nr_static_feats]:
                    new_feature_features.append((str(tmp), f))
                    if use_feat_decoders:
                        if f in self.unique_features_cont:
                            new_feature_features_cont.append((str(tmp), f))

        self.unique_targets = getattr(self, 'target_{}'.format(self.data_type))
        new_target_features = [('-1', t) for t in self.unique_targets]

        return new_feature_features, new_target_features, new_feature_features_cat, new_feature_features_cont

    @classmethod
    def _load_data(cls, csv_file: str) -> pd.DataFrame:
        """Load dataframe into memory"""
        df = pd.read_csv(csv_file, header=[0, 1])

        # Perform categorical labels encoding here
        df = coarse_cleaning(df)

        return df

    def _init_feature_info(self, data: pd.DataFrame):
        """Build feature information dictionary"""

        info = {}
        feat_means = {}
        feat_stds = {}
        # Compute mean and std across all timestamps and all patients
        for f in self.unique_features[:-self.nr_static_feats]:
            col_idx_list = [(str(i), f) for i in range(0, self.timestamps)]
            feat_df = self.data[col_idx_list]
            feat_stds[f] = np.nanstd(feat_df)
            feat_means[f] = np.nanmean(feat_df)
        for f in self.unique_features[-self.nr_static_feats:]:
            col_data = self.data[('-1', f)]
            if col_data.dtype != "category":
                feat_stds[f] = np.nanstd(col_data)
                feat_means[f] = np.nanmean(col_data)

        types = data.dtypes

        # Build feature information depending on the dtype of the column
        for feature in self.feature_features + self.target_features:
            nice_name = (self._metadata.loc[feature].varlab if self._metadata is not None else feature)
            if types[feature] == "category":
                info[feature] = FeatureInfo(
                    type="categorical",
                    possible_values=list(data[feature].cat.categories),
                    mean_std_values=None,
                    encoding_dict={
                        val: torch.tensor(idx) for idx, val in enumerate(list(data[feature].cat.categories))
                    },
                    nice_name=nice_name,
                )
            else:
                info[feature] = FeatureInfo(
                    type="continuous",
                    possible_values=None,
                    mean_std_values=(feat_means[feature[1]], feat_stds[feature[1]]),
                    encoding_dict=None,
                    nice_name=nice_name,
                )
        if self.normalise:
            for feature in self.feature_features + self.target_features:
                if info[feature].type != 'categorical':
                    normalized_values = (self.features.loc[:, feature].values - info[feature].mean_std_values[0]) / \
                                        info[feature].mean_std_values[1]
                    self.features.loc[:, feature] = normalized_values
                    if len(self.features_cont) > 0:
                        self.features_cont.loc[:, feature] = normalized_values

        return info

    def observed_values_for_feature(self, feature: str) -> pd.Series:
        return self.data[feature].dropna()

    def _remove_correlated_features(
            self, feature_features: List[Tuple[str, str]], threshold: float = 0.9
    ) -> List[Tuple[str, str]]:
        """Remove features that are highly correlated with one of the target"""

        def cramers_corrected_stat(x, y):
            """Calculate Cramers V statistic for categorial-categorial (x-y) association"""
            confusion_matrix = pd.crosstab(x, y)
            chi2 = ss.chi2_contingency(confusion_matrix)[0]
            n = confusion_matrix.sum().sum()
            phi2 = chi2 / n
            r, k = confusion_matrix.shape
            phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
            rcorr = r - ((r - 1) ** 2) / (n - 1)
            kcorr = k - ((k - 1) ** 2) / (n - 1)
            return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))

        filtered_features = []
        for feature in feature_features:
            filtered_features.append(feature)
            if self.data.dtypes[feature] == "category":
                cor_method = cramers_corrected_stat
            else:
                cor_method = "pearson"
            for target in self.target_features:
                cor = self.data[feature].corr(
                    self.data[target].cat.codes, method=cor_method
                )
                if cor > threshold:
                    filtered_features.pop()
                    break
        return filtered_features


def coarse_cleaning(df: DataFrame):
    """
    Takes a raw dataframe and turns it into a data frame with better encoding for categorical columns.
    """

    for column in df:
        level0, level1 = column
        # Check only into static variables
        if level0 == '-1':
            data = df[column]
            if data.dtype == "object":
                options = set(data.dropna().unique())
                if options.issubset({'Other', 'Medicare', 'Medicaid'}):
                    df[column] = pd.Categorical(
                        data, categories={'Other', 'Medicare', 'Medicaid'}
                    )
                elif options.issubset({'M', 'F'}):
                    df[column] = pd.Categorical(
                        data, categories={'M', 'F'}
                    )
                elif options.issubset({'WHITE', 'BLACK/AFRICAN AMERICAN', 'HISPANIC/LATINO', 'UNKNOWN', 'OTHER',
                                       'UNABLE TO OBTAIN', 'ASIAN', 'AMERICAN INDIAN/ALASKA NATIVE'}):
                    df[column] = pd.Categorical(
                        data.replace(
                            {
                                "UNKNOWN": None,
                                "UNABLE TO OBTAIN": None
                            }
                        ),
                        categories=['WHITE', 'BLACK/AFRICAN AMERICAN', 'HISPANIC/LATINO', 'OTHER', 'ASIAN',
                                    'AMERICAN INDIAN/ALASKA NATIVE'],
                    )
            elif data.dtype == "int64":
                options = set(data.dropna().unique())
                if options.issubset({0, 1}):
                    df[column] = pd.Categorical(
                        data.replace({1: "Yes", 0: "No"}), categories=["No", "Yes"]
                    )
    return df
