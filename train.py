import os

import seaborn as sns
import torch
from modn_data import DATA_ABS_PATH

import wandb
from modn.datasets.mimic import MIMICDataset
from modn.models.modn import MoDNMIMICHyperparameters
from modn.models.modn_decode import MoDNModelMIMICDecode, MoDNMIMICHyperparametersDecode
from modn.models.modn import MoDNModelMIMIC


PROJECT_NAME = "modn-on-mimic"
DATASET_TYPE = "toy"
EXP_ID = 10

FEATURE_DECODING = True
WANDB_LOG = False

PER_PATIENT = False
EARLY_STOPPING = True
RESET_STATE = False


def main():
    sns.set_style("whitegrid")

    # Initialize dataset
    data = MIMICDataset(os.path.join(DATA_ABS_PATH, "MIMIC_data_labels_{}.csv".format(DATASET_TYPE)),
                        data_type=DATASET_TYPE, global_question_block=False,
                        remove_correlated_features=not FEATURE_DECODING, use_feats_decoders=FEATURE_DECODING)

    # Define train/val/test splits
    train, val = data.random_split([0.8, 0.2], generator=torch.Generator().manual_seed(0))
    test = val

    if FEATURE_DECODING:
        lr_feature_decoders = {
            feature_name: 1e-2 for feature_name in data.unique_features_cat
        }
        if DATASET_TYPE == 'toy':
            lr_feature_decoders.update({
                'F1_constant': 6e-5,
                'F2_early': 1e-4,
                'F3_late': 4e-5,
                'F4_narrow': 2e-4,
                'F5_wide': 2e-4,
                'Age': 8e-5,
            })
            lr_encoders_val = 1e-2
            step_size_val = 150
            learning_rate_decay_factor = 0.9
            nr_epochs = 650

        else:
            lr_feature_decoders.update({
                'WBC': 1e-8,
                'Chloride (serum)': 1e-7,
                'Glucose (serum)': 5e-8,
                'Magnesium': 1e-8,
                'Sodium (serum)': 1e-6,
                'BUN': 1e-7,
                'Phosphorous': 1e-7,
                'Anion gap': 1e-7,
                'Potassium (serum)': 1e-7,
                'HCO3 (serum)': 1e-7,
                'Platelet Count': 1e-7,
                'Prothrombin time': 1e-7,
                'PTT': 5e-8,
                'Lactic Acid': 1e-7,
                'Age': 1e-7
            })
            lr_encoders_val = 1e-3
            step_size_val = 150
            learning_rate_decay_factor = 0.9
            nr_epochs = 250

        lr_encoders = {feature_name: lr_encoders_val for feature_name in data.unique_features}

        model_name = "Exp_{}_MaxEpochs_{}_{}{}".format(EXP_ID, nr_epochs, DATASET_TYPE,
                                                       '_feat_decode' if FEATURE_DECODING else '')

        # Define MoDN hyper parameters
        hyper_parameters = MoDNMIMICHyperparametersDecode(
            num_epochs=nr_epochs,
            state_size=60,
            lr_encoders=lr_encoders,
            lr_feature_decoders=lr_feature_decoders,
            lr_decoders=1e-2,
            lr=2e-3,
            learning_rate_decay_factor=learning_rate_decay_factor,
            step_size=step_size_val,
            gradient_clipping=1,
            aux_loss_encoded_weight=1,
            diseases_loss_weight=1,
            state_changes_loss_weight=1,
            shuffle_patients=True,
            shuffle_within_blocks=True,
            add_state=True,
            negative_slope=55,
            patience=50
        )
        # Define model
        model = MoDNModelMIMICDecode(RESET_STATE, parameters=hyper_parameters)
    else:
        negative_slope = 25
        add_state = True
        step_size = 120
        lr = 2e-3
        lr_encoders = 1e-2
        lr_decoders = 1e-2
        gradient_clipping = 1
        nr_epochs = 150
        state_size = 30
        patience = 30
        learning_rate_decay_factor = 0.9

        model_name = "Exp_{}_MaxEpochs_{}_{}{}{}".format(EXP_ID, nr_epochs, DATASET_TYPE,
                                                         'feat_decode' if FEATURE_DECODING else '',
                                                         '_per_patient' if PER_PATIENT else '')
        # Define MoDN hyper parameters
        hyper_parameters = MoDNMIMICHyperparameters(
            num_epochs=nr_epochs,
            state_size=state_size,
            lr_encoders=lr_encoders,
            lr_decoders=lr_decoders,
            lr=lr,
            learning_rate_decay_factor=learning_rate_decay_factor,
            step_size=step_size,
            gradient_clipping=gradient_clipping,
            diseases_loss_weight=1,
            state_changes_loss_weight=1,
            shuffle_patients=True,
            shuffle_within_blocks=True,
            add_state=add_state,
            negative_slope=negative_slope,
            patience=patience
        )
        # Define model
        model = MoDNModelMIMIC(RESET_STATE, parameters=hyper_parameters)

    if WANDB_LOG:
        wandb.init(project=PROJECT_NAME, name=model_name, config=hyper_parameters._asdict())

    # Fit model
    saved_model_name = f"{model_name}_model"

    model.fit(train, val, test, early_stopping=EARLY_STOPPING, wandb_log=WANDB_LOG, saved_model_name=saved_model_name)


if __name__ == "__main__":
    main()
