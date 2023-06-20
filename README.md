# ModARN-GenT
Generative Modular Autoregressive Networks for Medical Time-Series

# Abstract
Bla bla 

# Installation
The code runs completely on CPU, therefore there are no CUDA dependencies and the installation is very simple.
Please use the following command to install the required libraries:
```bash
 pip install -r requirements.txt
```

# Data
Entire data should be stored under `modn_data` folder. To generate the toy dataset you can use the functions from
the `Data_Generation.ipynb` notebook. To generate the MIMIC sub-dataset you can use the `MIMIC_Data_Pipeline.ipynb` notebook,
after you download the MIMIC-IV dataset by following the steps from [here](https://github.com/healthylaife/MIMIC-IV-Data-Pipeline#Steps-to-download-MIMIC-IV-dataset-for-the-pipeline).
Otherwise, please ask the previous student responsible for the project.

# Code structure
```
.
├───modn
│   ├───datasets            ---> dataset manipulation
│   │   ├─── mimic.py
│   │   └─── utils.py
│   ├───models              ---> network building and training
│   │   ├─── modn.py
│   │   ├─── modn_decode.py
│   │   ├─── modules.py
│   │   └─── utils.py
│   ├───notebooks                   
│   │   └─── Data_Generation.ipynb
│   └───scripts                  ---> helpers
│       ├─── data_inspect.py
│       ├─── explore_patient.py
│       └─── evaluation_utils.py
├───modn_data                    ---> data folder
├───plots                        ---> plots folder
├───saved models                  ---> models folder
├───requirements.txt                
├───evaluate.py                  ---> evaluation
├───generate_compare.py          ---> data generation and prediction
└───train.py                     ---> training
```

# Usage

### Training
Use the `train.py` script for training. We provide CLI options that are further explained in the training script.
Example command:
```bash
python train.py --dataset_type toy
                --exp_id <experiment_name_string>
                --feature_decoding
                --early_stopping
                --wandb_log
                --random_init_state
```
In the above command the training is done on the toy dataset, using feature decoders and a random initial state.
Early stopping is used and the experiments progress are tracked using wandb.

All the experiments are saved under `saved_models` folder.

The hyperparameters for each type of network are defined at the script level.

### Evaluation

Use the `evaluation.py` script for evaluation. 

Example command:
```bash
python evaluate.py --model_file <checkpoint_name>.pt
```
You can choose to reset (re-initialize) the state after each timestep with the `--reset_state` flag.

Plots with metrics will be created under the `plots` folder.

### Generation or prediction

Use the `generate_compare.py` script to either predict or generate data under a dataframe format.

```bash
python generate.py --model_file <checkpoint_name>.pt
                   --output_path <output_dataframe_name>.csv
```
With the above command a dataframe will be predicted using the given model.
If you want to generate data with a model, you need to add the `--generate` flag to the above command and TODO.
