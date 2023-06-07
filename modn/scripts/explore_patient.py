import os
import torch
from modn.datasets.mimic import MIMICDataset
from modn_data import DATA_ABS_PATH
import argparse


def get_cli_args(parser):
    parser.add_argument('--dataset_type', type=str, required=True, help='Dataset type: small or toy')
    parser.add_argument('--patient_id', type=int, default=10, help='Patient id for inspection')

    return parser.parse_args()


def main():
    args = get_cli_args(argparse.ArgumentParser())
    dataset_type = args.dataset_type
    patient_id = args.patient_id

    data = MIMICDataset(csv_file=os.path.join(DATA_ABS_PATH, f"MIMIC_data_labels_{dataset_type}.csv"),
                        data_type=dataset_type, remove_correlated_features=False, global_question_block=False,
                        use_feats_decoders=True)

    # Split dataset into train and test
    train, test = data.random_split([0.8, 0.2], generator=torch.Generator().manual_seed(0))

    # Choose one patient
    patient = test[patient_id]

    # Print its consultation results
    for q in patient.consultation.observations(shuffle_within_blocks=True):
        print(q.question, q.answer)


if __name__ == "__main__":
    main()
