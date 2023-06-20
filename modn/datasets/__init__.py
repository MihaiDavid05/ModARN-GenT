import random
from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import Dict
from typing import List
from typing import Literal
from typing import NamedTuple
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import pandas as pd
import torch


class ObservationMIMIC(NamedTuple):
    """A pair of question and answer"""

    question: Tuple[str, str]
    answer: Any

    def __str__(self) -> str:
        if isinstance(self.answer, str):
            astr = self.answer
        else:
            astr = f"{self.answer:.1f}"
        return f"({self.question[0]}, {self.question[1]}) = {astr}"


# a couple of questions that were asked together
ObservationBlockMIMIC = List[ObservationMIMIC]


class ConsultationMIMIC(NamedTuple):
    """Ordered list of questions/answers, asked in blocks"""

    question_blocks: List[ObservationBlockMIMIC]

    def observations(self, shuffle_within_blocks=False, question_blocks=None, feature_decode=False):
        """Return observations one by one"""
        if question_blocks is not None:
            question_block_list = question_blocks
        else:
            question_block_list = self.question_blocks
        for block in question_block_list:
            len_block = len(block)
            block_timestamp = block[0].question[0]
            if shuffle_within_blocks:
                block = block.copy()
                random.shuffle(block)
            for obs in block:
                len_block = len_block - 1
                if feature_decode:
                    yield obs, len_block, block_timestamp
                else:
                    yield obs, len_block

    def get_decoders_targets(self, index_targets, question_blocks=None):
        """Return observations from next time_block, if available"""
        if question_blocks is not None:
            question_block_list = question_blocks
        else:
            question_block_list = self.question_blocks
        try:
            block2 = question_block_list[index_targets]
        except IndexError:
            return []
        return block2


class DataPointMIMIC(NamedTuple):
    """Entry point in the dataset"""

    consultation: ConsultationMIMIC
    targets: Union[Dict[Tuple[str, str], int], List[Dict[Tuple[str, str], int]]]

    def __str__(self):
        return (
                "Patient:\n"
                + "  Questions:\n"
                + "".join(self._format_block(block) for block in self.consultation.question_blocks)
                + f"  Targets:\n{self._format_targets()}"
        )

    @staticmethod
    def _format_block(block: ObservationBlockMIMIC) -> str:
        return "    Block:\n" + "".join(
            f"      ({obs.question[0]}, {obs.question[1]}) = {obs.answer}\n" for obs in block
        )

    def _format_targets(self) -> str:
        return "".join(f"    ({k[0]}, {k[1]}) = {v}\n" for k, v in self.targets.items())


class FeatureInfo(NamedTuple):
    type: Literal["categorical", "continuous"]
    possible_values: Optional[List[Any]]
    mean_std_values: Optional[Tuple[np.float64, np.float64]]
    encoding_dict: Optional[Dict[str, torch.Tensor]]
    nice_name: str


class PatientDataset(ABC):

    feature_info: Dict[Tuple[str, str], FeatureInfo]
    feature_features: List[Tuple[str, str]]
    feature_features_cont: List[Tuple[str, str]]
    feature_features_cat: List[Tuple[str, str]]
    target_features: List[Tuple[str, str]]
    unique_features: List[str]
    unique_features_cont: List[str]
    unique_features_cat: List[str]
    unique_targets: List[str]
    nr_static_feats: int
    timestamps: int

    @abstractmethod
    def __len__(self) -> int:  # pragma: no cover
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> DataPointMIMIC:  # pragma: no cover
        pass

    @abstractmethod
    def observed_values_for_feature(
        self, feature: str
    ) -> pd.Series:  # pragma: no cover
        pass

    @abstractmethod
    def features_for_patient(
        self, patient_idx: Union[int, List[int]]
    ) -> pd.DataFrame:  # pragma: no cover
        pass

    @abstractmethod
    def targets_for_patient(
        self, patient_idx: Union[int, List[int]]
    ) -> pd.DataFrame:  # pragma: no cover
        pass

    def random_split(
            self,
            probabilities: List[float],
            generator: torch.Generator = torch.default_generator,
    ) -> List["PatientDataset"]:
        sum_p = sum(probabilities)
        lengths = [int(len(self) * p / sum_p) for p in probabilities]

        # to include the left out element when (len(self) * p / sum_p) is not an integer
        lengths[-1] += len(self) - sum(lengths)
        indices = torch.randperm(sum(lengths), generator=generator).tolist()
        return [
            PatientSubsetDataset(self, indices[offset - length: offset])
            for offset, length in zip(torch._utils._accumulate(lengths), lengths)
        ]


class PatientSubsetDataset(PatientDataset):
    def __init__(self, dataset: PatientDataset, subset_indices: List[int]):
        self._data = dataset
        self.feature_features = dataset.feature_features
        self.target_features = dataset.target_features
        self.feature_features_cont = dataset.feature_features_cont
        self.feature_features_cat = dataset.feature_features_cat
        self.unique_features = dataset.unique_features
        self.unique_features_cont = dataset.unique_features_cont
        self.unique_features_cat = dataset.unique_features_cat
        self.unique_targets = dataset.unique_targets
        self.nr_static_feats = dataset.nr_static_feats
        self.timestamps = dataset.timestamps
        self._indices = subset_indices
        self._indices_set = set(self._indices)

        # NOTE: Mean/std are computed on whole data (both for train/test split)
        self.feature_info = self._data.feature_info

        # NOTE: _update_feature_info() does not do anything actually
        # self.feature_info = self._update_feature_info()

    def __len__(self):
        return len(self._indices)

    def observed_values_for_feature(self, feature: str) -> pd.Series:
        observations = self._data.observed_values_for_feature(feature)
        return observations[observations.index.isin(self._indices_set)]

    def __getitem__(self, idx: int):
        idx = self._indices[idx]
        return self._data[idx]

    def features_for_patient(self, patient_idx: Union[int, List[int]]) -> pd.DataFrame:
        return self._data.features_for_patient(patient_idx)

    def targets_for_patient(self, patient_idx: Union[int, List[int]]) -> pd.DataFrame:
        return self._data.targets_for_patient(patient_idx)

    def _update_feature_info(self) -> Dict[Tuple[str, str], FeatureInfo]:
        feature_info = self._data.feature_info
        for (key, value) in feature_info.items():
            if value.type == "continuous":
                value = FeatureInfo(
                    type="continuous",
                    possible_values=None,
                    mean_std_values=(
                        self.observed_values_for_feature(key).mean(),
                        self.observed_values_for_feature(key).std(),
                    ),
                    encoding_dict=None,
                    nice_name=value.nice_name,
                )
        return feature_info
