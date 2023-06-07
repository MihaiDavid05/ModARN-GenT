import os

import seaborn as sns
import torch
from modn_data import DATA_ABS_PATH
import argparse
import wandb
from modn.datasets.mimic import MIMICDataset
from modn.models.modn import MoDNMIMICHyperparameters
from modn.models.modn_decode import MoDNModelMIMICDecode, MoDNMIMICHyperparametersDecode
from modn.models.modn import MoDNModelMIMIC

# Project name for wandb
PROJECT_NAME = "modn-on-mimic"


def get_cli_args(parser):
    parser.add_argument('--dataset_type', type=str, required=True, help='Dataset type: small or toy')
    parser.add_argument('--exp_id', type=int, required=True, help='Experiment id (in case multiple configs)')
    parser.add_argument('--feature_decoding', action='store_true', help='Whether to use feature decoding or not')
    parser.add_argument('--reset_state', action='store_true',
                        help='Whether to reset state at each timestep at validation time')
    parser.add_argument('--wandb_log', action='store_true', help='Log results in wandb or not')
    parser.add_argument('--early_stopping', action='store_true', help='Use early stopping or not')

    return parser.parse_args()


def main():
    sns.set_style("whitegrid")

    args = get_cli_args(argparse.ArgumentParser())
    dataset_type = args.dataset_type
    exp_id = args.exp_id
    feature_decoding = args.feature_decoding
    reset_state = args.reset_state
    wandb_log = args.wandb_log
    early_stopping = args.early_stopping

    # Note: Deprecated parameter in case using modn_slow.py script. Leave this as False.
    per_patient = False

    # Initialize dataset
    data = MIMICDataset(os.path.join(DATA_ABS_PATH, "MIMIC_data_labels_{}.csv".format(dataset_type)),
                        data_type=dataset_type, global_question_block=False,
                        remove_correlated_features=not feature_decoding, use_feats_decoders=feature_decoding)

    # Define train/val/test splits
    train, val = data.random_split([0.8, 0.2], generator=torch.Generator().manual_seed(0))
    test = val

    if feature_decoding:
        lr_feature_decoders = {
            feature_name: 1e-2 for feature_name in data.unique_features_cat
        }
        if dataset_type == 'toy':
            lr_feature_decoders.update({
                'F1_constant': 6e-5,
                'F2_early': 1e-4,
                'F3_late': 4e-5,
                'F4_narrow': 2e-4,
                'F5_wide': 2e-4,
                'Age': 8e-5,
            })
            lr_encoders_val = 1e-2
            step_size_val = 300
            learning_rate_decay_factor = 0.9
            nr_epochs = 800

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
            lr_encoders_val = 1e-2
            step_size_val = 200
            learning_rate_decay_factor = 0.9
            nr_epochs = 500

        lr_encoders = {feature_name: lr_encoders_val for feature_name in data.unique_features}

        model_name = "Exp_{}_MaxEpochs_{}_{}{}".format(exp_id, nr_epochs, dataset_type,
                                                       '_feat_decode' if feature_decoding else '')

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
        model = MoDNModelMIMICDecode(reset_state, parameters=hyper_parameters)
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

        model_name = "Exp_{}_MaxEpochs_{}_{}{}{}".format(exp_id, nr_epochs, dataset_type,
                                                         'feat_decode' if feature_decoding else '',
                                                         '_per_patient' if per_patient else '')
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
        model = MoDNModelMIMIC(reset_state, parameters=hyper_parameters)

    if wandb_log:
        wandb.init(project=PROJECT_NAME, name=model_name, config=hyper_parameters._asdict())

    # Fit model
    saved_model_name = f"{model_name}_model"

    model.fit(train, val, test, early_stopping=early_stopping, wandb_log=wandb_log, saved_model_name=saved_model_name)


if __name__ == "__main__":
    main()
