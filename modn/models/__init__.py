from abc import ABC
from abc import abstractmethod
from typing import Dict
from typing import List
from typing import Tuple

import pandas as pd
from modn.datasets import ConsultationMIMIC
from modn.datasets import PatientDataset

Prediction = pd.DataFrame  # vector of (class) -> probability
PredictionEvolution = pd.DataFrame  # matrix (time, class) -> probability
FeatureNameMIMIC = Tuple[str, str]


class PatientModel(ABC):
    @abstractmethod
    def fit(
        self,
        train_data: PatientDataset,
        val_data: PatientDataset,
        test_data: PatientDataset,
        early_stopping: bool,
        wandb_log: bool,
        saved_model_name: str,
        pretrained: bool,
    ):
        pass

    @abstractmethod
    def predict(
        self, consultation: ConsultationMIMIC, targets: List[FeatureNameMIMIC], reset_state: bool
    ) -> Dict[FeatureNameMIMIC, Prediction]:
        """Final prediction after the consultation is finished"""
        pass

    @abstractmethod
    def predict_evolution(
        self, consultation: ConsultationMIMIC, targets: List[FeatureNameMIMIC], reset_state: bool,
            test_set: PatientDataset = None
    ) -> Dict[FeatureNameMIMIC, PredictionEvolution]:
        """Give a prediction after each new question"""
        pass
