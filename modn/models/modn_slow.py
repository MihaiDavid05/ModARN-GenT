from collections import defaultdict
from pathlib import Path
from typing import Dict, List, NamedTuple, Union, Optional

import os
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

# Parameters tried

# negative_slope = 50
# add_state = True
# step_size = 120
# lr = 2e-3
# lr_encoders = 1e-6
# lr_decoders = 1e-8
# gradient_clipping = 1
# nr_epochs = 150
# state_size = 30
# patience = 20
# learning_rate_decay_factor = 0.9


class EarlyStopper:
    def __init__(self, model, wandb_log, min_delta=0):
        self.min_delta = min_delta
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
        elif validation_loss > (self.min_validation_loss + self.min_delta):
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
        elif val_f1_score < (self.max_f1_score - self.min_delta):
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
    reset_state_eval: bool = True


class MoDNModelMIMIC(PatientModel):
    """The Modular Decision support Networks model"""

    def __init__(self, parameters: Optional[MoDNMIMICHyperparameters] = None):

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
            self.reset_state_eval
        ) = hyper_parameters

    def compute_loss(self, data, criterion, train: bool = True):

        if train:
            shuffle_patients = self.shuffle_patients
            shuffle_within_blocks = self.shuffle_within_blocks
        else:
            shuffle_patients = False
            shuffle_within_blocks = False

        # Patients that do not have continuous data
        bad_patient = 0
        # Total loss initialization
        total_loss = 0
        total_disease_loss = 0
        total_state_loss = 0

        indices = (
            np.random.permutation(len(data))
            if shuffle_patients
            else range(len(data))
        )
        for idx in tqdm(indices):
            consultation, targets = data[idx]
            # If a patient does not have dynamic data, therefore, no blocks, ignore it
            if len(consultation.question_blocks) == 0:
                bad_patient += 1
                continue
            # Initialize number of blocks per consultation
            nr_blocks = 0
            # Initialize losses per consultation
            disease_loss = state_changes_loss = 0

            state = self.init_state(1)
            # for t = 0, with just the initial state
            for idx1, (target_name, target) in enumerate(targets.items()):
                enc_dict = self.feature_info[target_name].encoding_dict
                target = enc_dict[target].view(1)
                logits1 = self.decoders[target_name](state)
                disease_loss += criterion(logits1, target)
            # for t > 0, after encoding each question / answer pair
            for (question, answer), nr_obs in consultation.observations(shuffle_within_blocks=shuffle_within_blocks):
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
            total_loss += loss / nr_blocks
            total_disease_loss += disease_loss / nr_blocks
            total_state_loss += state_changes_loss / nr_blocks

        return total_loss, total_disease_loss, total_state_loss

    def fit(
            self,
            train_data: PatientDataset,
            val_data: PatientDataset,
            test_data: PatientDataset,
            early_stopping: bool = True,
            wandb_log: bool = False,
            saved_model_name: str = "modn_plus_model",
            pretrained: bool = False,
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
                self.state_size, self.feature_info[(feat_type, name)])

        self.encoders = encoders
        self.decoders = {
            name: EpoctBinaryDecoder(self.state_size)
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
        optimizer = torch.optim.SGD(parameter_groups, momentum=0.9, nesterov=True, lr=self.lr)
        # optimizer = torch.optim.Adam(parameter_groups, lr=self.lr)

        step_size = self.step_size * len(train_data)
        scheduler = StepLR(
            optimizer, step_size=step_size, gamma=self.learning_rate_decay_factor
        )

        # feed data into the modules and train them
        # with one question / answer pair at a time
        print("\nTraining the model")
        criterion = nn.CrossEntropyLoss()
        iters = 0

        for epoch in range(self.num_epochs):
            # Patients that do not have continuous data
            bad_patient = 0

            training_indices = (
                np.random.permutation(len(train_data))
                if self.shuffle_patients
                else range(len(train_data))
            )
            for idx in tqdm(training_indices):
                consultation, targets = train_data[idx]
                # If a patient does not have dynamic data, therefore, no blocks, ignore it
                if len(consultation.question_blocks) == 0:
                    iters += 1
                    bad_patient += 1
                    continue
                # Initialize number of blocks per consultation
                nr_blocks = 0
                # Initialize losses per consultation
                disease_loss = state_changes_loss = 0
                # Set gradients to zero
                optimizer.zero_grad()
                state = self.init_state(1)
                # for t = 0, with just the initial state
                for idx1, (target_name, target) in enumerate(targets.items()):
                    enc_dict = self.feature_info[target_name].encoding_dict
                    target = enc_dict[target].view(1)
                    logits1 = self.decoders[target_name](state)
                    disease_loss += criterion(logits1, target)
                # for t > 0, after encoding each question / answer pair
                for (question, answer), nr_obs in consultation.observations(
                        shuffle_within_blocks=self.shuffle_within_blocks
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

                iters += 1

                train_losses = {
                    "loss": loss / nr_blocks,
                    "disease_loss": disease_loss / nr_blocks,
                    "state_changes_loss": state_changes_loss / nr_blocks,
                }

                train_losses["loss"].backward()

                for param_dict in optimizer.param_groups:
                    for p in param_dict["params"]:
                        torch.nn.utils.clip_grad_norm_(p, max_norm=self.gradient_clipping)

                optimizer.step()
                scheduler.step()

            with torch.no_grad():
                avg_epoch_loss_val, avg_disease_loss_val, avg_state_loss_val = self.compute_loss(val_data, criterion)
                avg_epoch_loss_train, avg_disease_loss_train, avg_state_loss_train = \
                    self.compute_loss(train_data, criterion)

            print("Epoch average val loss is: {}".format(avg_epoch_loss_val))
            print("Epoch average train loss is: {}".format(avg_epoch_loss_train))

            print("Doing validation")
            with torch.no_grad():
                results_mean = []
                for data in [val_data]:
                    f1_score_iteration = \
                        self.evaluate_model_at_multiple_stages(test_set=data,
                                                               targets=data.target_features,
                                                               stages=list(range(-1, data.timestamps)),
                                                               reset_state=self.reset_state_eval)
                    macro_f1s = [f1_score_iteration[stage]["macro_f1"]
                                 for stage in list(range(-1, data.timestamps))]
                    f1_score_iteration_mean = np.array(macro_f1s).mean()
                    results_mean.append(f1_score_iteration_mean)

            print(
                f"Epoch: {epoch + 1}/{self.num_epochs}, iteration: {iters}\n"
                + f"val_f1_score_mean_by_X_iters: {results_mean[0]:.4f}\n"
                # + f"train_f1_score_mean_by_X_iters: {results_mean[0]:.4f}"
            )
            if wandb_log:
                wandb.log(
                    {
                        "val_f1_score_mean": results_mean[0],
                        # "train_f1_score_mean": results_mean[0],
                        "lr": optimizer.param_groups[0]['lr'],
                        "val_loss": avg_epoch_loss_val,
                        "val_loss_state": avg_state_loss_val,
                        "val_loss_disease": avg_disease_loss_val,
                        "train_loss": avg_epoch_loss_train,
                        "train_loss_state": avg_state_loss_train,
                        "train_loss_disease": avg_disease_loss_train
                    }
                )

            if early_stopping:
                save_path_loss = os.path.join('../../saved_models', saved_model_name + '_best_loss.pt')
                if early_stopper.early_stop_loss(avg_epoch_loss_val, save_path_loss, self.patience):
                    break

                save_path_f1 = os.path.join('../../saved_models', saved_model_name + '_best_f1_mean.pt')
                if early_stopper.early_stop_f1(results_mean[0], save_path_f1, self.patience):
                    break
            else:
                save_path = os.path.join('../../saved_models', saved_model_name + str(epoch + 1) + '_best.pt')
                self.save_and_store(wandb_log, save_path)

    def save_and_store(self, wandb_log, model_name):
        self.save_model(model_name)
        # if wandb_log:
        #     wandb.log_artifact(model_name, name=model_name, type="model")

    def evaluate_model_at_multiple_stages(self,
                                          test_set: PatientDataset,
                                          targets: List[FeatureNameMIMIC],
                                          stages: List[Stage],
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

    def predict(
            self,
            consultation: ConsultationMIMIC,
            targets: List[FeatureNameMIMIC],
            reset_state: bool
    ) -> Dict[FeatureNameMIMIC, Prediction]:
        """Final prediction after the consultation is finished"""
        results = self.predict_evolution(consultation=consultation, targets=targets, reset_state=reset_state)
        return {f: d.iloc[-1, :] for f, d in results.items()}

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
        }
        torch.save(model_dict, Path(model_path))

    def load_model(self, model_path: str):
        model_dict = torch.load(Path(model_path))
        self.init_state = model_dict["init_state"]
        self.encoders = model_dict["encoders"]
        self.decoders = model_dict["decoders"]
        self.feature_info = model_dict["feature_info"]
        self.hyper_parameters = model_dict["hyperparameters"]
        self._load_hyperparameters(self.hyper_parameters)
