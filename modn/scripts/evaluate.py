import os

import numpy as np
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from modn.datasets.mimic import MIMICDataset
from modn.models.modn import MoDNModelMIMIC
from modn.models.modn_decode import MoDNModelMIMICDecode
from modn_data import DATA_ABS_PATH
import matplotlib.lines as mlines

RESET_STATE = False

# feat decode small
# MODEL_PATH = 'Max_Epoch_150_Mortality_2h_frame_48h_window_1h_small_feat_decode_second_model_best_loss.pt'

# feat decode toy
MODEL_NAME = 'Exp_10_MaxEpochs_650_toy_feat_decode_model_best_loss.pt'

# just disease small
# MODEL_PATH = 'Max_Epoch_150_Mortality_2h_frame_48h_window_1h_small_sixth_model_best_f1.pt'
# MODEL_PATH = 'Max_Epoch_150_Mortality_2h_frame_48h_window_1h_small_sixth_model_best_loss.pt'


def get_plots_name():
    pass


def generate_plots(model_name, feature_decoding, stages, results):
    pass
    # macro_f1s = [results[stage]["macro_f1"] for stage in stages]

    # plots_name = "featdecode_second{}{}.pdf".format('_reset_state' if RESET_STATE else '', loss_f1)
    # plots_path = os.path.join('../../plots', plots_name)
    #
    # fig, ax = plt.subplots(figsize=(14, 5))
    # ax.plot(stages, macro_f1s)

    # max_idx_score = np.argmax(macro_f1s)
    # max_score = np.max(macro_f1s)

    # ax.plot(max_idx_score - 1, np.max(max_score), 'o',
    #         label='max: {}, timestep: {}'.format(np.max(max_score), max_idx_score - 1),
    #         markersize=3, color='orange')
    # point = mlines.Line2D([], [], color='orange', marker='o', linestyle='None', markersize=5, label='max')
    # ax.legend(handles=[point])
    # ax.set(xlabel=r"Timestep", ylabel="Macro F1 score")
    # ax.set_xlim(-2, len(stages) + 1)
    # ax.set_ylim(0, 1)
    # ax.legend()
    # plt.tight_layout()
    # fig.savefig(plots_path)


def main():
    sns.set_style("whitegrid")

    model_path = os.path.join('../../saved_models', MODEL_NAME)
    feature_decoding = True if 'feat_decode' in MODEL_NAME else False
    dataset_type = 'toy' if 'toy' in MODEL_NAME else 'small'

    if 'loss' in MODEL_NAME:
        loss_f1 = '_loss'
    elif 'f1' in MODEL_NAME:
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
        m = MoDNModelMIMICDecode(reset_state=RESET_STATE)
    else:
        m = MoDNModelMIMIC(reset_state=RESET_STATE)

    # Load model
    m.load_model(model_path)

    # Define timespan
    stages = list(range(-1, data.timestamps))

    if feature_decoding:
        targets = test.unique_targets + test.unique_features_cat + test.unique_features_cont
        results = m.evaluate_model_at_multiple_stages(test_set=test, targets=targets,
                                                      stages=stages, reset_state=RESET_STATE)
        print("OK")
    else:
        results = m.evaluate_model_at_multiple_stages(test_set=test, targets=test.target_features,
                                                      stages=stages, reset_state=RESET_STATE)
        print("OK")

    generate_plots(MODEL_NAME, feature_decoding, stages, results)


if __name__ == "__main__":
    main()
