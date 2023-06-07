import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, NamedTuple, Union, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wandb
from torch.optim.lr_scheduler import StepLR
from tqdm.auto import tqdm

from modn.datasets import ConsultationMIMIC, PatientDataset
from modn.models import FeatureNameMIMIC, PatientModel, Prediction
from modn.models.modules import EpoctBinaryDecoder, EpoctEncoder, InitState, EpoctCategoricalDecoder
from modn.models.modules import EpoctDistributionDecoder
from modn.models.utils import neg_log_likelihood_1d

MetricName = str
Stage = float


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


class MoDNMIMICHyperparametersDecode(NamedTuple):
    num_epochs: int = 250
    state_size: int = 50
    lr_encoders: dict = None
    lr_feature_decoders: dict = None
    lr_decoders: float = 1e-2
    lr: float = 2e-3
    learning_rate_decay_factor: float = 0.9
    step_size: int = 150
    gradient_clipping: Union[float, int] = 10000
    aux_loss_encoded_weight: Union[float, int] = 1,
    diseases_loss_weight: Union[float, int] = 1
    state_changes_loss_weight: Union[float, int] = 1
    shuffle_patients: bool = True
    shuffle_within_blocks: bool = True
    add_state: bool = True
    negative_slope: int = 5
    patience: int = 5


class MoDNModelMIMICDecode(PatientModel):
    """The Modular Decision support Networks model"""

    def __init__(self, reset_state: bool, parameters: Optional[MoDNMIMICHyperparametersDecode] = None):

        self.reset_state_eval = reset_state
        self.hyper_parameters = parameters
        if self.hyper_parameters is not None:
            self._load_hyperparameters(self.hyper_parameters)

        self.init_state = None
        self.encoders = None
        self.decoders = None
        self.feature_decoders_cat = None
        self.feature_decoders_cont = None
        self.feature_info = None

    def _load_hyperparameters(self, hyper_parameters: MoDNMIMICHyperparametersDecode):
        (
            self.num_epochs,
            self.state_size,
            self.lr_encoders,
            self.lr_feature_decoders,
            self.lr_decoders,
            self.lr,
            self.learning_rate_decay_factor,
            self.step_size,
            self.gradient_clipping,
            self.aux_loss_encoded_weight,
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
        # feature encoding loss for encoded continuous features
        aux_loss_cont_encoded = 0
        # feature encoding loss for encoded categorical features
        aux_loss_cat_encoded = 0
        # to store the sum of all the disease decoding losses after each encoding step
        disease_loss = 0
        # state_changes_loss
        state_changes_loss = 0

        decoder_loss_dict = dict.fromkeys(list(self.feature_decoders_cont.keys()), 0)
        decoder_loss_dict.update(dict.fromkeys(list(self.feature_decoders_cat.keys()), 0))

        training_indices = (
            np.random.permutation(len(data))
            if shuffle_patients
            else range(len(data))
        )

        nr_blocks = 0
        nr_blocks_features_cont = 0
        nr_blocks_features_cat = 0
        not_good_patient = 0

        for idx in tqdm(training_indices):
            block_per_consultation = 0
            consultation, targets = data[idx]
            targets, targets_cont, targets_cat = targets

            # Ignore patient without continuous data and select at most 24 random but ordered time-blocks
            selected_question_blocks = None
            if len(consultation.question_blocks) == 0:
                not_good_patient += 1
                continue
            elif len(consultation.question_blocks) > 24:
                random_idx = sorted(np.random.choice(len(consultation.question_blocks), size=24, replace=False))
                selected_question_blocks = (np.array(consultation.question_blocks)[random_idx]).tolist()

            if selected_question_blocks:
                block_timesteps = [int(block[0].question[0]) for block in selected_question_blocks]
            else:
                block_timesteps = [int(block[0].question[0]) for block in consultation.question_blocks]

            # Initialize state for 1 patient
            state = self.init_state(1)
            # for t = 0, with just the initial state
            for idx1, (target_name, target) in enumerate(targets.items()):
                enc_dict = self.feature_info[target_name].encoding_dict
                target = enc_dict[target].view(1)
                logits1 = self.decoders[target_name](state)
                disease_loss += criterion(logits1, target)
            # for t > 0
            for (question, answer), nr_obs, timestamp in consultation.observations(
                    shuffle_within_blocks=shuffle_within_blocks, question_blocks=selected_question_blocks,
                    feature_decode=True
            ):
                state_before = state.clone()
                question = question[1]

                state = self.encoders[question](state, answer)
                lss1 = torch.mean((state - state_before) ** 2)
                state_changes_loss += lss1
                # print("After 1 encoder state loss is {}:".format(lss1))
                # if lss1 > 10:
                #     print(question)
                #     print("Big state loss {}".format(lss1))

                # After having encoded a time-block, apply a disease decoder
                if nr_obs == 0:
                    for idx2, (target_name, target) in enumerate(targets.items()):
                        enc_dict = self.feature_info[target_name].encoding_dict
                        target = enc_dict[target].view(1)
                        logits2 = self.decoders[target_name](state)
                        lss2 = criterion(logits2, target)
                        disease_loss += lss2

                        # print("After 1 decoder loss is {}:".format(lss2))
                        # if lss2 > 50:
                        #     print(question)
                        #     print("Big decode loss {}".format(lss2))

                    nr_blocks += 1
                    block_per_consultation += 1

                    target_obs = consultation.get_decoders_targets(block_per_consultation,
                                                                   question_blocks=selected_question_blocks)
                    if len(target_obs) > 0:
                        add_cont_loss = False
                        add_cat_loss = False
                        for decoder_cont_question in self.feature_decoders_cont.keys():
                            target_val = None
                            for t in block_timesteps[block_per_consultation:]:
                                if decoder_cont_question == 'Age':
                                    time = '-1'
                                else:
                                    time = str(t)
                                if not math.isnan(targets_cont[(time, decoder_cont_question)]):
                                    target_val = targets_cont[(time, decoder_cont_question)]
                                    break
                            if target_val:
                                lss_cont = neg_log_likelihood_1d(torch.tensor(target_val).view(-1, 1),
                                                                 *self.feature_decoders_cont[decoder_cont_question](
                                                                     state))
                                aux_loss_cont_encoded += lss_cont
                                decoder_loss_dict[decoder_cont_question] += lss_cont
                                add_cont_loss = True

                        for decoder_cat_question in self.feature_decoders_cat.keys():
                            target_val = targets_cat[('-1', decoder_cat_question)]
                            if isinstance(target_val, str):
                                target_decoder_cat = self.feature_info[('-1', decoder_cat_question)].encoding_dict[
                                    target_val].view(1)
                                lss_cat = criterion(self.feature_decoders_cat[decoder_cat_question](state),
                                                    target_decoder_cat)
                                aux_loss_cat_encoded += lss_cat
                                decoder_loss_dict[decoder_cat_question] += lss_cat
                                add_cat_loss = True

                        if add_cont_loss:
                            nr_blocks_features_cont += 1
                        if add_cat_loss:
                            nr_blocks_features_cat += 1

        aux_loss_cont_encoded /= nr_blocks_features_cont
        aux_loss_cat_encoded /= nr_blocks_features_cat
        disease_loss /= nr_blocks
        state_changes_loss /= nr_blocks

        loss = (
                (aux_loss_cont_encoded + aux_loss_cat_encoded) * self.aux_loss_encoded_weight
                + disease_loss * self.diseases_loss_weight
                + state_changes_loss * self.state_changes_loss_weight
        )
        losses = {
            "loss": loss,
            "disease_loss": disease_loss,
            "state_changes_loss": state_changes_loss,
            "aux_loss_cont_encoded": aux_loss_cont_encoded,
            "aux_loss_cat_encoded": aux_loss_cat_encoded,
        }
        losses.update(decoder_loss_dict)

        return losses, decoder_loss_dict

    def get_feature_encoders(self, train_data):
        encoders = {}
        for i, name in enumerate(train_data.unique_features):
            if i < len(train_data.unique_features) - train_data.nr_static_feats:
                # Timestamp does not matter, we only need the possible values and the type of feature in the encoder
                feat_type = str(train_data.timestamps - 1)
            else:
                feat_type = str(-1)
            encoders[name] = EpoctEncoder(
                self.state_size, self.feature_info[(feat_type, name)], add_state=self.add_state,
                negative_slope=self.negative_slope)
        return encoders

    def get_feature_decoders(self, train_data):

        cont_feature_decoders = {
            name: EpoctDistributionDecoder(self.state_size, negative_slope=self.negative_slope)
            for name in train_data.unique_features_cont
        }

        cat_feature_decoders = {
            name: EpoctCategoricalDecoder(
                self.state_size, len(self.feature_info[('-1', name)].possible_values),
                negative_slope=self.negative_slope)
            for name in train_data.unique_features_cat
        }

        return cat_feature_decoders, cont_feature_decoders

    def get_optimizer_decoding(self, train_data):

        # setup optimizers, the scheduler, and the loss
        parameter_groups = [{"params": [self.init_state.state_value], "lr": self.lr}]
        parameter_groups.extend(
            {"params": list(self.encoders[encoder_name].parameters()), "lr": self.lr_encoders[encoder_name]}
            for encoder_name in self.encoders
        )
        parameter_groups.extend(
            {"params": list(decoder.parameters()), "lr": self.lr_decoders}
            for decoder in self.decoders.values()
        )

        parameter_groups.extend(
            [
                {
                    "params": list(self.feature_decoders_cont[feature_name].parameters()),
                    "lr": self.lr_feature_decoders[feature_name],
                }
                for feature_name in train_data.unique_features_cont
            ]
        )
        parameter_groups.extend(
            [
                {
                    "params": list(self.feature_decoders_cat[feature_name].parameters()),
                    "lr": self.lr_feature_decoders[feature_name],
                }
                for feature_name in train_data.unique_features_cat
            ]
        )

        optimizer = torch.optim.Adam(parameter_groups, lr=self.lr)

        return optimizer

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
        early_stopper = EarlyStopper(self, wandb_log, self.patience)

        self.feature_info = train_data.feature_info

        if pretrained:
            # TODO: Build and check code for pretrianing
            # assert self.init_state is not None
            # assert self.encoders is not None
            # assert self.decoders is not None
            # assert self.feature_decoders_cat is not None
            # assert self.feature_decoders_cont is not  None
            # assert self.feature_info is not None
            #
            # unique_features = set(train_data.unique_features).difference(set(self.encoders.keys()))
            # encoders = self.encoders
            pass

        # initiate the initial state, encoders and decoders for features and targets
        self.init_state = InitState(self.state_size)

        self.encoders = self.get_feature_encoders(train_data)
        self.decoders = {
            name: EpoctBinaryDecoder(self.state_size, negative_slope=self.negative_slope)
            for name in train_data.target_features
        }
        self.feature_decoders_cat, self.feature_decoders_cont = self.get_feature_decoders(train_data)

        # initialize learning rates, optimizer and scheduler
        optimizer = self.get_optimizer_decoding(train_data)
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

        # feed data into the modules and train them with one question / answer pair at a time
        print("\nTraining the model")
        for epoch in range(self.num_epochs):
            optimizer.zero_grad()
            train_losses, train_decoder_losses = self._compute_loss(train_data_list)

            # for enc_name, enc in self.encoders.items():
            #     print(enc_name)
            #     print(enc.fc1.weight.norm())
            #     print(enc.fc2.weight.norm())
            # for dec_name, dec in self.decoders.items():
            #     print(dec_name)
            #     print(dec.fc1.weight.norm())

            train_losses["loss"].backward()

            # for enc_name, module in self.encoders.items():
            #     print(enc_name)
            #     for name, p in module.named_parameters():
            #         print(name)
            #         print(p.grad.norm())

            for param_dict in optimizer.param_groups:
                for p in param_dict["params"]:
                    torch.nn.utils.clip_grad_norm_(p, max_norm=self.gradient_clipping)

            # for enc_name, module in self.encoders.items():
            #     print(enc_name)
            #     for name, p in module.named_parameters():
            #         print(name)
            #         print(p.grad.norm())

            # compute test losses and metrics
            with torch.no_grad():

                val_losses, val_decoder_losses = self._compute_loss(val_data_list, train=False)
                test_losses = val_losses
                targets = val_data.unique_targets + val_data.unique_features_cat + val_data.unique_features_cont
                val_scores = self.evaluate_model_at_multiple_stages(test_set=val_data,
                                                                    targets=targets,
                                                                    stages=list(range(-1, val_data.timestamps)),
                                                                    reset_state=self.reset_state_eval)

                logs = {}
                for d in self.decoders:
                    f1s = [val_scores[stage][f"{d[1]}_f1"] for stage in list(range(-1, val_data.timestamps))]
                    logs[f"val_f1_score_{d}"] = np.array(f1s).mean()

                rmse = []
                for d in self.feature_decoders_cont:
                    for stage in list(range(0, val_data.timestamps - 1)):
                        if val_scores[stage].get(f"{d}_rmse"):
                            rmse.append(val_scores[stage][f"{d}_rmse"])
                    logs[f"val_rmse_score_{d}"] = np.array(rmse).mean()

                for d in self.feature_decoders_cat:
                    f1s = [val_scores[stage][f"{d}_f1"] for stage in list(range(0, val_data.timestamps))]
                    logs[f"val_f1_score_{d}"] = np.array(f1s).mean()

                macro_f1s_disease = [val_scores[stage]["macro_f1_targets"] for stage in
                                     list(range(-1, val_data.timestamps))]
                val_f1_scores_disease = np.array(macro_f1s_disease).mean()

            optimizer.step()
            scheduler.step()

            # for enc_name, enc in self.encoders.items():
            #     print(enc_name)
            #     print(enc.fc1.weight.norm())
            #     print(enc.fc2.weight.norm())
            # for dec_name, dec in self.decoders.items():
            #     print(dec_name)
            #     print(dec.fc1.weight.norm())

            print(
                f"Epoch: {epoch + 1}/{self.num_epochs}\n"
                + f"train_loss: {train_losses['loss']:.4f}\n"
                + f"disease_loss: {train_losses['disease_loss']:.4f}\n"
                + f"state_changes_loss: {train_losses['state_changes_loss']:.4f}\n"
                + f"aux_loss_cont_encoded: {train_losses['aux_loss_cont_encoded']:.4f}\n"
                + f"aux_loss_cat_encoded: {train_losses['aux_loss_cat_encoded']:.4f}\n"
                + f"test_loss: {test_losses['loss']:.4f}\n"
                + f"test_disease_loss: {test_losses['disease_loss']:.4f}\n"
                + f"test_state_changes_loss: {test_losses['state_changes_loss']:.4f}\n"
                + f"test_aux_loss_cont_encoded: {test_losses['aux_loss_cont_encoded']:.4f}\n"
                + f"test_aux_loss_cat_encoded: {test_losses['aux_loss_cat_encoded']:.4f}\n"
                + f"val_f1_score_disease: {val_f1_scores_disease:.4f}"
            )

            if math.isnan(train_losses["loss"]):
                raise ValueError("NaN loss encountered")

            if wandb_log:
                decoder_losses = {f"val_{key}_loss": value for key, value in
                                  val_decoder_losses.items()}
                decoder_losses.update({f"{key}_loss": value for key, value in
                                       train_decoder_losses.items()})

                results = {
                    "train_loss": train_losses["loss"],
                    "disease_loss": train_losses["disease_loss"],
                    "state_changes_loss": train_losses["state_changes_loss"],
                    "aux_loss_cont_encoded": train_losses['aux_loss_cont_encoded'],
                    "aux_loss_cat_encoded": train_losses['aux_loss_cat_encoded'],
                    "val_loss": val_losses["loss"],
                    "val_disease_loss": val_losses["disease_loss"],
                    "val_state_changes_loss": val_losses["state_changes_loss"],
                    "val_aux_loss_cont_encoded": val_losses["aux_loss_cont_encoded"],
                    "val_aux_loss_cat_encoded": val_losses["aux_loss_cat_encoded"],
                    "val_f1_score_disease": val_f1_scores_disease
                }
                results.update(logs)
                results.update(decoder_losses)

                wandb.log(results)

            if early_stopping:
                save_path = os.path.join('saved_models', saved_model_name + '_best_loss.pt')
                if early_stopper.early_stop_loss(val_losses["loss"], save_path, self.patience):
                    break
            else:
                save_path = os.path.join('saved_models', saved_model_name + str(epoch + 1) + '_best.pt')
                self.save_and_store(wandb_log, save_path)

    def save_and_store(self, wandb_log, model_name):
        # if wandb_log:
        #     wandb.log_artifact(model_name, name=model_name, type="model")
        self.save_model(model_name)

    def evaluate_model_at_multiple_stages(self,
                                          test_set: PatientDataset,
                                          targets,
                                          stages: List[Stage],
                                          reset_state: bool
                                          ) -> Dict[Stage, Dict[MetricName, float]]:
        counts = defaultdict(int)  # (stage, feature, gt value, correct/wrong) -> count
        correct_predictions = defaultdict(int)  # (stage, feature) -> count
        cont_predictions = defaultdict(lambda: [0.0, 0])
        patients_with_missing_val = {k: 0 for k in test_set.unique_features_cat}

        for patient in tqdm(test_set):

            predictions, predictions_cont, predictions_cat = self.predict_evolution(
                patient.consultation, targets, reset_state, test_set
            )
            true_values, true_values_cont, true_values_cat = patient.targets

            for target, preds in predictions_cat.items():
                missing_val = False
                for i, position in enumerate(preds.index.values[:-1]):
                    gt = true_values_cat[('-1', target)]
                    if isinstance(gt, str):
                        p = preds.loc[position]
                        predicted_value = p.index[p.argmax()]
                        counts[int(position), target, gt, "correct"] += (
                            1 if gt == predicted_value else 0
                        )
                        counts[int(position), target, gt, "wrong"] += 1 if gt != predicted_value else 0
                        correct_predictions[int(position), target] += 1 if gt == predicted_value else 0
                    else:
                        missing_val = True
                if missing_val:
                    patients_with_missing_val[target] += 1

            for target, preds in predictions_cont.items():
                for i, position in enumerate(preds.index.values[:-1]):
                    if target == ('-1', 'Age'):
                        gt = true_values_cont[target]
                        pred_name = 'Age'
                    else:
                        gt = true_values_cont[(str(preds.index.values[i + 1]), target)]
                        pred_name = target
                    if not math.isnan(gt):
                        p = preds.loc[position].values[0]
                        se = np.square(np.subtract(gt, p))
                        cont_predictions[position, pred_name][0] += se
                        cont_predictions[position, pred_name][1] += 1

            for target, preds in predictions.items():
                for stage in stages:
                    if stage == -1:
                        position = "No information"
                    else:
                        position = str(stage)
                    if position not in preds.index.values:
                        continue
                    p = preds.loc[position]
                    predicted_value = p.index[p.argmax()]
                    gt = true_values[('-1', target)]
                    counts[stage, target, gt, "correct"] += (
                        1 if gt == predicted_value else 0
                    )
                    counts[stage, target, gt, "wrong"] += 1 if gt != predicted_value else 0
                    correct_predictions[stage, target] += 1 if gt == predicted_value else 0

        cont_predictions_rmse = defaultdict(float)
        for key, value in cont_predictions.items():
            cont_predictions_rmse[key] = math.sqrt(value[0] / value[1])

        metrics = {}
        for stage in stages:
            metrics[stage] = {}

            # RMSE
            for target in test_set.unique_features_cont:
                if cont_predictions_rmse.get((str(stage), target), None):
                    metrics[stage][f"{target}_rmse"] = cont_predictions_rmse[(str(stage), target)]

            # Accuracy
            for target in test_set.unique_targets:
                choices = test_set.feature_info[('-1', target)].possible_values
                assert choices is not None
                metrics[stage][f"{target}_accuracy"] = sum(
                    counts[stage, target, choice, "correct"] for choice in choices
                ) / len(test_set)

            for target in test_set.unique_features_cat:
                choices = test_set.feature_info[('-1', target)].possible_values
                assert choices is not None
                metrics[stage][f"{target}_accuracy"] = sum(
                    counts[stage, target, choice, "correct"] for choice in choices
                ) / (len(test_set) - patients_with_missing_val[target])

            # F1
            for target in test_set.unique_targets + test_set.unique_features_cat:
                choices = test_set.feature_info[('-1', target)].possible_values
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

            metrics[stage][f"accuracy_targets"] = sum(
                correct_predictions[stage, target] for target in test_set.unique_targets
            ) / (len(test_set) * len(test_set.unique_targets))

            metrics[stage][f"macro_f1_targets"] = sum(
                metrics[stage][f"{target}_f1"] for target in test_set.unique_targets
            ) / len(test_set.unique_targets)

            if stage != -1:
                metrics[stage][f"accuracy_targets_cat"] = sum(
                    correct_predictions[stage, target] for target in test_set.unique_features_cat
                ) / (len(test_set) * len(test_set.unique_features_cat) - sum(patients_with_missing_val.values()))

                metrics[stage][f"macro_f1_targets_cat"] = sum(
                    metrics[stage][f"{target}_f1"] for target in test_set.unique_features_cat
                ) / len(test_set.unique_features_cat)

            # TODO: RMSE
            # metrics[stage]['rmse'] = sum(
            #     metrics[stage].get([f"{target}_rmse"], 0) for target in test_set.unique_features_cont
            # )

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
    ):
        """Give a prediction after each time-block"""

        # initialize the data structure with zeros
        predictions = {}
        predictions_cat = {}
        predictions_cont = {}
        # if len(consultation.question_blocks) == 0:
        #     print("No continuous obs")
        time_indexes = [q_block[0].question[0] for q_block in consultation.question_blocks]

        for target_feature in targets:
            if target_feature in test_set.unique_targets or target_feature in test_set.unique_features_cat:
                info = self.feature_info[('-1', target_feature)]
                assert info.type == "categorical"
                assert info.possible_values is not None
                if target_feature in test_set.unique_targets:
                    matrix = np.zeros([len(consultation.question_blocks) + 1, len(info.possible_values)]) + 0.5
                    predictions[target_feature] = pd.DataFrame(
                        matrix,
                        columns=info.possible_values,
                        index=["No information", *time_indexes],
                    )
                else:
                    matrix = np.zeros([len(consultation.question_blocks), len(info.possible_values)]) + 0.5
                    predictions_cat[target_feature] = pd.DataFrame(
                        matrix,
                        columns=info.possible_values,
                        index=[*time_indexes],
                    )
            else:
                # target is continuous
                if target_feature == 'Age':
                    target_feature = ('-1', 'Age')
                matrix = (np.zeros([len(consultation.question_blocks), 1]))
                predictions_cont[target_feature] = pd.DataFrame(
                    matrix,
                    columns=['mu'],
                    index=[*time_indexes],
                )

        with torch.no_grad():
            state = self.init_state(1)
            for timestep in range(0, len(time_indexes) + 1):
                if reset_state:
                    state = self.init_state(1)
                if timestep == 0:
                    for target_feature in test_set.unique_targets:
                        info = self.feature_info[('-1', target_feature)]
                        assert info.possible_values is not None
                        probs = (
                            self.decoders[('-1', target_feature)](state)
                            .softmax(1)
                            .detach()
                            .numpy()[0]
                        )
                        for k in range(len(info.possible_values)):
                            predictions[target_feature].iloc[timestep, k] = probs[k]

                else:
                    observations = consultation.question_blocks[timestep - 1]
                    questions_asked = [obs.question[1] for obs in observations]
                    # ts = observations[0].question[0]
                    for j, q in enumerate(questions_asked):
                        state = self.encoders[q](
                            state, observations[j].answer
                        )
                    for target_feature in test_set.unique_targets:
                        info = self.feature_info[('-1', target_feature)]
                        assert info.possible_values is not None
                        probs = self.decoders[('-1', target_feature)](state).softmax(1).detach().numpy()[0]
                        for k in range(len(info.possible_values)):
                            predictions[target_feature].iloc[timestep, k] = probs[k]
                    if timestep < len(time_indexes):
                        for target_feature in test_set.unique_features_cat:
                            info = self.feature_info[('-1', target_feature)]
                            probs = self.feature_decoders_cat[target_feature](state).softmax(1).detach().numpy()[0]
                            for k in range(len(info.possible_values)):
                                predictions_cat[target_feature].iloc[timestep - 1, k] = probs[k]
                        for target_feature in test_set.unique_features_cont:
                            prediction_target_feature = target_feature
                            if target_feature == 'Age':
                                prediction_target_feature = ('-1', 'Age')
                            mu = self.feature_decoders_cont[target_feature](state)[0].detach().numpy()[0]
                            predictions_cont[prediction_target_feature].iloc[timestep - 1, 0] = mu

        return predictions, predictions_cont, predictions_cat

    def save_model(self, model_path: str):
        model_dict = {
            "init_state": self.init_state,
            "encoders": self.encoders,
            "decoders": self.decoders,
            "feature_decoders_cat": self.feature_decoders_cat,
            "feature_decoders_cont": self.feature_decoders_cont,
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
        self.feature_decoders_cat = model_dict["feature_decoders_cat"]
        self.feature_decoders_cont = model_dict["feature_decoders_cont"]
        self.feature_info = model_dict["feature_info"]
        self.hyper_parameters = model_dict["hyperparameters"]
        self.reset_state_eval = model_dict["reset_state_eval"]
        self._load_hyperparameters(self.hyper_parameters)
