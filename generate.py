import os
import torch
import argparse
from modn_data import DATA_ABS_PATH
from modn.datasets.mimic import MIMICDataset
from modn.models.modn import MoDNModelMIMIC
from modn.models.modn_decode import MoDNModelMIMICDecode


def get_cli_args(parser):
    parser.add_argument('--model_file', type=str, required=True, help='Checkpoint you want to use for generation')

    return parser.parse_args()


def main():

    args = get_cli_args(argparse.ArgumentParser())
    model_name = args.model_file

    model_path = os.path.join('saved_models', model_name)

    # Note: Only for models with feature decoding
    assert 'feat_decode' in model_name, "Model chosen does not have feature decoders"
    feature_decoding = True

    dataset_type = 'toy' if 'toy' in model_name else 'small'

    data = MIMICDataset(
        os.path.join(DATA_ABS_PATH, "MIMIC_data_labels_{}.csv".format(dataset_type)), data_type=dataset_type,
        global_question_block=False, remove_correlated_features=False, use_feats_decoders=feature_decoding
    )

    # Define test split
    _, test = data.random_split([0.8, 0.2], generator=torch.Generator().manual_seed(0))

    # Define model
    if feature_decoding:
        m = MoDNModelMIMICDecode()
    else:
        m = MoDNModelMIMIC()

    # Load model
    m.load_model(model_path)

    # Generate dataset
    m.generate()


if __name__ == "__main__":
    main()
