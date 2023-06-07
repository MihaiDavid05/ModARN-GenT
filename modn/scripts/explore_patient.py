import os
import torch
from modn.datasets.mimic import MIMICDataset
from modn_data import DATA_ABS_PATH


PATIENT_ID = 10
DATASET_TYPE = 'small'


def main():
    data = MIMICDataset(csv_file=os.path.join(DATA_ABS_PATH, f"MIMIC_data_labels_{DATASET_TYPE}.csv"),
                        data_type=DATASET_TYPE, remove_correlated_features=False, global_question_block=False,
                        use_feats_decoders=True)

    # Split dataset into train and test
    train, test = data.random_split([0.8, 0.2], generator=torch.Generator().manual_seed(0))

    # Choose one patient
    patient = test[PATIENT_ID]

    # Print its consultation results
    for q in patient.consultation.observations(shuffle_within_blocks=True):
        print(q.question, q.answer)


if __name__ == "__main__":
    main()
