import os
from pathlib import Path
from typing import Dict, List, NamedTuple, Union, Optional
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import wandb
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from tqdm.auto import tqdm

from modn.datasets import ConsultationMIMIC, PatientDataset
from modn.models import FeatureNameMIMIC, PatientModel, Prediction, PredictionEvolution
from modn.models.modules import EpoctBinaryDecoder, EpoctEncoder, InitState

MetricName = str
Stage = float


class EarlyStopper:
    def __init__(self, model, wandb_log):
        self.counter_f1 = 0
        self.counter_loss = 0
        self.min_validation_loss = np.inf
        self.model = model
        self.wandb_log = wandb_log
        self.max_f1_score = 0.0

    def early_stop_loss(self, validation_loss, model_name, patience):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter_loss = 0
            print("Lower loss detected. Saving model.")
            self.model.save_and_store(self.wandb_log, model_name)
        elif validation_loss > self.min_validation_loss:
            self.counter_loss += 1
            print("Higher loss detected. Counter {}".format(self.counter_loss))
            if self.counter_loss >= patience:
                print("Model reached maximum patience of {}. Stopping!".format(patience))
                return True
        return False

    def early_stop_f1(self, val_f1_score, model_name, patience):
        if val_f1_score > self.max_f1_score:
            self.max_f1_score = val_f1_score
            self.counter_f1 = 0
            print("Higher f1 score detected. Saving model.")
            self.model.save_and_store(self.wandb_log, model_name)
        elif val_f1_score < self.max_f1_score:
            self.counter_f1 += 1
            print("Lower f1 score detected. Counter {}".format(self.counter_loss))
            if self.counter_f1 >= patience:
                print("Model reached maximum patience of {}. Stopping!".format(patience))
                return True
        return False


class MoDNMIMICHyperparameters(NamedTuple):
    num_epochs: int = 250
    state_size: int = 50
    lr_encoders: float = 1e-2
    lr_decoders: float = 1e-2
    lr: float = 2e-3
    learning_rate_decay_factor: float = 0.9
    step_size: int = 150
    gradient_clipping: Union[float, int] = 10000
    diseases_loss_weight: Union[float, int] = 1
    state_changes_loss_weight: Union[float, int] = 1
    shuffle_patients: bool = True
    shuffle_within_blocks: bool = True
    add_state: bool = True
    negative_slope: int = 5
    patience: int = 5


class MoDNModelMIMIC(PatientModel):
    """The Modular Decision support Networks model"""

    def __init__(self, reset_state: bool = False, parameters: Optional[MoDNMIMICHyperparameters] = None):

        self.reset_state_eval = reset_state
        self.hyper_parameters = parameters
        if self.hyper_parameters is not None:
            self._load_hyperparameters(self.hyper_parameters)

        self.init_state = None
        self.encoders = None
        self.decoders = None
        self.feature_info = None

    def _load_hyperparameters(self, hyper_parameters: MoDNMIMICHyperparameters):
        (
            self.num_epochs,
            self.state_size,
            self.lr_encoders,
            self.lr_decoders,
            self.lr,
            self.learning_rate_decay_factor,
            self.step_size,
            self.gradient_clipping,
            self.diseases_loss_weight,
            self.state_changes_loss_weight,
            self.shuffle_patients,
            self.shuffle_within_blocks,
            self.add_state,
            self.negative_slope,
            self.patience,
        ) = hyper_parameters

    def _compute_loss(self, data: list, train: bool = True):

        if train:
            shuffle_patients = self.shuffle_patients
            shuffle_within_blocks = self.shuffle_within_blocks
        else:
            shuffle_patients = False
            shuffle_within_blocks = False

        criterion = nn.CrossEntropyLoss()
        disease_loss = state_changes_loss = 0

        training_indices = (
            np.random.permutation(len(data))
            if shuffle_patients
            else range(len(data))
        )

        nr_blocks = 0
        not_good_patient = 0
        for idx in tqdm(training_indices):
            consultation, targets = data[idx]
            selected_question_blocks = None
            if len(consultation.question_blocks) == 0:
                not_good_patient += 1
                continue
            elif len(consultation.question_blocks) > 24:
                random_idx = sorted(np.random.choice(len(consultation.question_blocks), size=24, replace=False))
                selected_question_blocks = (np.array(consultation.question_blocks)[random_idx]).tolist()
            state = self.init_state(1)
            # for t = 0, with just the initial state
            for idx1, (target_name, target) in enumerate(targets.items()):
                enc_dict = self.feature_info[target_name].encoding_dict
                target = enc_dict[target].view(1)
                logits1 = self.decoders[target_name](state)
                disease_loss += criterion(logits1, target)
            # for t > 0, after encoding each question / answer pair
            for (question, answer), nr_obs in consultation.observations(
                    shuffle_within_blocks=shuffle_within_blocks, question_blocks=selected_question_blocks
            ):
                state_before = state.clone()
                question = question[1]
                state = self.encoders[question](state, answer)
                lss1 = torch.mean((state - state_before) ** 2)
                state_changes_loss += lss1
                if nr_obs == 0:
                    for idx2, (target_name, target) in enumerate(targets.items()):
                        enc_dict = self.feature_info[target_name].encoding_dict
                        target = enc_dict[target].view(1)
                        logits2 = self.decoders[target_name](state)
                        lss2 = criterion(logits2, target)
                        disease_loss += lss2
                    nr_blocks += 1

        loss = (
                disease_loss * self.diseases_loss_weight
                + state_changes_loss * self.state_changes_loss_weight
        )

        losses = {
            "loss": loss / nr_blocks,
            "disease_loss": disease_loss / nr_blocks,
            "state_changes_loss": state_changes_loss / nr_blocks,
        }

        return losses

    def fit(
            self,
            train_data: PatientDataset,
            val_data: PatientDataset,
            test_data: PatientDataset,
            early_stopping: bool = True,
            wandb_log: bool = False,
            saved_model_name: str = "modn_plus_model",
            pretrained: bool = False
    ):
        early_stopper = EarlyStopper(self, wandb_log)

        unique_features = train_data.unique_features
        nr_unique_features = len(unique_features)
        self.feature_info = train_data.feature_info
        encoders = {}
        if pretrained:
            assert self.init_state is not None
            assert self.encoders is not None
            assert self.decoders is not None
            assert self.feature_info is not None

            unique_features = set(unique_features).difference(set(self.encoders.keys()))
            encoders = self.encoders

        # initiate the initial state, encoders and decoders for features and targets
        self.init_state = InitState(self.state_size)

        for i, name in enumerate(unique_features):
            if i < nr_unique_features - train_data.nr_static_feats:
                # NOTE: mean and std are the same for all timestamps, therefore choosing any timestamp is the same here
                feat_type = str(train_data.timestamps - 1)
            else:
                feat_type = str(-1)
            encoders[name] = EpoctEncoder(
                self.state_size, self.feature_info[(feat_type, name)], add_state=self.add_state,
                negative_slope=self.negative_slope
            )

        self.encoders = encoders
        self.decoders = {
            name: EpoctBinaryDecoder(self.state_size, negative_slope=self.negative_slope)
            for name in train_data.target_features
        }

        # setup optimizers, the scheduler, and the loss
        parameter_groups = [{"params": [self.init_state.state_value], "lr": self.lr}]
        parameter_groups.extend(
            {"params": list(encoder.parameters()), "lr": self.lr_encoders}
            for encoder in self.encoders.values()
        )
        parameter_groups.extend(
            {"params": list(decoder.parameters()), "lr": self.lr_decoders}
            for decoder in self.decoders.values()
        )
        optimizer = torch.optim.Adam(parameter_groups, lr=self.lr)

        scheduler = StepLR(
            optimizer, step_size=self.step_size, gamma=self.learning_rate_decay_factor
        )

        print("Saving consultation and targets in a list to speed up training")
        train_data_list = []
        for idx in tqdm(range(len(train_data))):
            consultation, targets = train_data[idx]
            train_data_list.append((consultation, targets))
        val_data_list = []
        for idx in tqdm(range(len(val_data))):
            consultation, targets = val_data[idx]
            val_data_list.append((consultation, targets))
        test_data_list = []
        for idx in tqdm(range(len(test_data))):
            consultation, targets = test_data[idx]
            test_data_list.append((consultation, targets))

        # feed data into the modules and train them
        # with one question / answer pair at a time
        print("\nTraining the model")
        for epoch in range(self.num_epochs):
            optimizer.zero_grad()
            train_losses = self._compute_loss(train_data_list)
            train_losses["loss"].backward()

            for param_dict in optimizer.param_groups:
                for p in param_dict["params"]:
                    torch.nn.utils.clip_grad_norm_(p, max_norm=self.gradient_clipping)

            # compute test losses
            with torch.no_grad():
                val_losses = self._compute_loss(val_data_list, train=False)
                test_losses = val_losses
                val_f1_scores = self.evaluate_model_at_multiple_stages(test_set=val_data,
                                                                       targets=val_data.target_features,
                                                                       stages=list(range(-1, val_data.timestamps)),
                                                                       reset_state=self.reset_state_eval)
                macro_f1s = [val_f1_scores[stage]["macro_f1"] for stage in list(range(-1, val_data.timestamps))]
                val_f1_scores = np.array(macro_f1s).mean()

            optimizer.step()
            scheduler.step()

            print(
                f"Epoch: {epoch + 1}/{self.num_epochs}\n"
                + f"train_loss: {train_losses['loss']:.4f}; "
                + f"test_loss: {test_losses['loss']:.4f}\n"
                + f"val_f1_score: {val_f1_scores:.4f}"
            )

            if wandb_log:
                wandb.log(
                    {
                        "train_loss": train_losses["loss"],
                        "disease_loss": train_losses["disease_loss"],
                        "state_changes_loss": train_losses["state_changes_loss"],
                        "val_loss": val_losses["loss"],
                        "val_disease_loss": val_losses["disease_loss"],
                        "val_state_changes_loss": val_losses["state_changes_loss"],
                        "val_f1_score": val_f1_scores
                    }
                )

            if early_stopping:
                save_path_loss = os.path.join('saved_models', saved_model_name + '_best_loss.pt')
                if early_stopper.early_stop_loss(val_losses["loss"], save_path_loss, self.patience):
                    break
                save_path_f1 = os.path.join('saved_models', saved_model_name + '_best_f1.pt')
                if early_stopper.early_stop_f1(val_f1_scores, save_path_f1, self.num_epochs):
                    break
            else:
                save_path = os.path.join('saved_models', saved_model_name + str(epoch + 1) + '_best.pt')
                self.save_and_store(wandb_log, save_path)

    def save_and_store(self, wandb_log, model_name):
        self.save_model(model_name)
        # if wandb_log:
        #     wandb.log_artifact(model_name, name=model_name, type="model")

    def evaluate_model_at_multiple_stages(self,
                                          test_set: PatientDataset,
                                          targets: List[FeatureNameMIMIC],
                                          stages: List[Stage],  # fraction of the consultation
                                          reset_state: bool
                                          ) -> Dict[Stage, Dict[MetricName, float]]:
        counts = defaultdict(int)  # (stage, feature, gt value, correct/wrong) -> count
        correct_predictions = defaultdict(int)  # (stage, feature) -> count
        for patient in tqdm(test_set):
            predictions = self.predict_evolution(
                patient.consultation, targets, reset_state
            )
            true_values = patient.targets
            for target, preds in predictions.items():
                # preds is a matrix (num questions asked, possible value) -> probability
                for stage in stages:
                    if stage == -1:
                        position = "No information"
                    else:
                        position = str(stage)
                    if position not in preds.index.values:
                        continue
                    p = preds.loc[position]
                    predicted_value = p.index[p.argmax()]
                    gt = true_values[target]
                    counts[stage, target, gt, "correct"] += (
                        1 if gt == predicted_value else 0
                    )
                    counts[stage, target, gt, "wrong"] += 1 if gt != predicted_value else 0
                    correct_predictions[stage, target] += 1 if gt == predicted_value else 0

        metrics = {}
        for stage in stages:
            metrics[stage] = {}
            # Accuracy
            for target in targets:
                choices = test_set.feature_info[target].possible_values
                assert choices is not None
                metrics[stage][f"{target}_accuracy"] = sum(
                    counts[stage, target, choice, "correct"] for choice in choices
                ) / len(test_set)

            # F1
            for target in targets:
                choices = test_set.feature_info[target].possible_values
                assert choices is not None
                for choice in choices:
                    assert choices is not None
                    if counts[stage, target, choice, "correct"] == 0:
                        metrics[stage][f"{target}_{choice}_f1"] = 0
                    else:
                        metrics[stage][f"{target}_{choice}_f1"] = counts[
                                                                      stage, target, choice, "correct"
                                                                  ] / (
                                                                          counts[stage, target, choice, "correct"]
                                                                          + 0.5
                                                                          * sum(
                                                                      counts[stage, target, choice, "wrong"] for choice
                                                                      in choices
                                                                  )
                                                                  )

                metrics[stage][f"{target}_f1"] = sum(
                    metrics[stage][f"{target}_{choice}_f1"] for choice in choices
                ) / len(choices)

            metrics[stage][f"accuracy"] = sum(
                correct_predictions[stage, target] for target in targets
            ) / (len(test_set) * len(targets))

            metrics[stage][f"macro_f1"] = sum(
                metrics[stage][f"{target}_f1"] for target in targets
            ) / len(targets)

        return metrics

    def predict_evolution(
            self, consultation: ConsultationMIMIC, targets: List[FeatureNameMIMIC], reset_state: bool,
            test_set: PatientDataset = None
    ) -> Dict[FeatureNameMIMIC, PredictionEvolution]:
        """Give a prediction after each new question"""

        # initialize the data structure with zeros
        predictions = {}
        time_indexes = [q_block[0].question[0] for q_block in consultation.question_blocks]

        for target_feature in targets:
            info = self.feature_info[target_feature]
            assert info.type == "categorical"
            assert info.possible_values is not None
            matrix = (
                    np.zeros([len(consultation.question_blocks) + 1, len(info.possible_values)]) + 0.5
            )
            predictions[target_feature] = pd.DataFrame(
                matrix,
                columns=info.possible_values,
                index=[
                    "No information",
                    *time_indexes,
                ],
            )
        with torch.no_grad():
            state = self.init_state(1)
            for timestep in range(0, len(time_indexes) + 1):
                if reset_state:
                    state = self.init_state(1)
                if timestep != 0:
                    observations = consultation.question_blocks[timestep - 1]
                    questions_asked = [obs.question[1] for obs in observations]
                    # ts = observations[0].question[0]
                    for i, q in enumerate(questions_asked):
                        state = self.encoders[q](
                            state, observations[i].answer
                        )
                for target_feature in targets:
                    info = self.feature_info[target_feature]
                    assert info.possible_values is not None
                    # exp(log_softmax) --> softmax (probabilities)
                    probs = (
                        self.decoders[target_feature](state)
                        .softmax(1)
                        .detach()
                        .numpy()[0]
                    )
                    for i in range(len(info.possible_values)):
                        predictions[target_feature].iloc[timestep, i] = probs[i]

        return predictions

    def save_model(self, model_path: str):
        model_dict = {
            "init_state": self.init_state,
            "encoders": self.encoders,
            "decoders": self.decoders,
            "feature_info": self.feature_info,
            "hyperparameters": self.hyper_parameters,
            "reset_state_eval": self.reset_state_eval
        }
        torch.save(model_dict, Path(model_path))

    def load_model(self, model_path: str):
        model_dict = torch.load(Path(model_path))
        self.init_state = model_dict["init_state"]
        self.encoders = model_dict["encoders"]
        self.decoders = model_dict["decoders"]
        self.feature_info = model_dict["feature_info"]
        self.hyper_parameters = model_dict["hyperparameters"]
        self.reset_state_eval = model_dict["reset_state_eval"]
        self._load_hyperparameters(self.hyper_parameters)
