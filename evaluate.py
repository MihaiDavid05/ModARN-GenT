import os
import seaborn as sns
import torch
from modn.datasets.mimic import MIMICDataset
from modn.models.modn import MoDNModelMIMIC
from modn.models.modn_decode import MoDNModelMIMICDecode
from modn.scripts.evaluation_utils import get_plots_path, generate_plots
from modn_data import DATA_ABS_PATH
import argparse

# feat decode small
# MODEL_PATH = 'Max_Epoch_150_Mortality_2h_frame_48h_window_1h_small_feat_decode_second_model_best_loss.pt'

# just disease small
# MODEL_PATH = 'Max_Epoch_150_Mortality_2h_frame_48h_window_1h_small_sixth_model_best_f1.pt'
# MODEL_PATH = 'Max_Epoch_150_Mortality_2h_frame_48h_window_1h_small_sixth_model_best_loss.pt'


def get_cli_args(parser):
    parser.add_argument('--model_file', type=str, required=True, help='Checkpoint you want to use for generation')
    parser.add_argument('--reset_state', action='store_true',
                        help='Whether to reset state at each timestep at validation time')

    return parser.parse_args()


def main():
    sns.set_style("whitegrid")

    args = get_cli_args(argparse.ArgumentParser())
    model_name = args.model_file
    reset_state = args.reset_state

    model_path = os.path.join('saved_models', model_name)
    feature_decoding = True if 'feat_decode' in model_name else False
    dataset_type = 'toy' if 'toy' in model_name else 'small'

    if 'loss' in model_name:
        loss_f1 = '_loss'
    elif 'f1' in model_name:
        loss_f1 = '_f1'
    else:
        loss_f1 = ''

    data = MIMICDataset(
        os.path.join(DATA_ABS_PATH, "MIMIC_data_labels_{}.csv".format(dataset_type)), data_type=dataset_type,
        global_question_block=False, remove_correlated_features=False, use_feats_decoders=feature_decoding
    )

    # Define test split
    _, test = data.random_split([0.8, 0.2], generator=torch.Generator().manual_seed(0))

    # Define model
    if feature_decoding:
        m = MoDNModelMIMICDecode(reset_state=reset_state)
    else:
        m = MoDNModelMIMIC(reset_state=reset_state)

    # Load model
    m.load_model(model_path)

    # Define timespan
    stages = list(range(-1, data.timestamps))

    if feature_decoding:
        targets = test.unique_targets + test.unique_features_cat + test.unique_features_cont
    else:
        targets = test.target_features

    results = m.evaluate_model_at_multiple_stages(test_set=test, targets=targets,
                                                  stages=stages, reset_state=reset_state)

    plots_path = get_plots_path(model_name)
    generate_plots(m, plots_path, feature_decoding, reset_state, loss_f1, stages, results)


if __name__ == "__main__":
    main()
