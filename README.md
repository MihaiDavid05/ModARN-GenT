# :hospital: ModARN-GenT
Generative Modular Autoregressive Networks for Medical Time-Series

# :bookmark_tabs: Abstract
*BACKGROUND.* In medicine, patients are frequently observed over fine-grained time frames, and
variations in repeated measures are crucial to determining how well the patient is responding to
therapy. However, it might be difficult to gather and manage granular time series data representing a
patient stay in the hospital, not only because confidentiality limits data exchange between institutions
or third parties, but also because of the data’s persistent missingness. Targeted monitoring, in which
the presence or intensity of measurements is predictive of specific outcomes like severity, produces
this missingness known as MNAR (missingness-not-at-random). Although various techniques have
been created to create synthetic times series, many of them fail to get beyond this crucial limitation.

*AIM.* This work proposes MoDNARN-GenT, a modular neural network framework able to learn
time series that suffer from systematic missingness with the aim of generating simulated time series
data in the form of EHRs (electronic health records).

*METHODS/FINDINGS.* We use both a custom toy dataset with 400 records and 5 time series
features and a subset of the most popular publicly available EHRs (electronic health records) dataset,
subset comprising of 6926 records and 14 time series features. We redesign the existing MoDN[1]
(Modular Clinical Decision Support Networks) architecture to take as input and generate time series
data. Through multiple versions of the toy dataset we investigate how different levels of missingness
may hinder the network from learning and generating data and present its current limitations for
working with a particularly complex real dataset.

*CONCLUSION.* We show the power of MoDARN-GenT to work with time series data as it yields
good results in both learning and generating time series from data without missingness. Even though
the modularization allows the model to learn from a flexible number and combination of inputs
without being biased by the systematic missingness, we conclude that the amount of missing data has
a negative impact on the overall learning performance.

# :gear: Installation
The code runs completely on CPU, therefore there are no CUDA dependencies and the installation is very simple.
Please use the following command to install the required libraries:
```bash
 pip install -r requirements.txt
```

# :books: Data
All the `.csv` files related to the data should be stored under `modn_data` folder.

To generate the toy dataset you can use the functions from
the `Data_Generation.ipynb` notebook under `notebooks` folder.

To generate the MIMIC subset you can use the `mainPipeline.ipynb` notebook under `MIMIC-IV-Data-Pipeline` folder. 

**IMPORTANT NOTE**: The `MIMIC-IV-Data-Pipeline` directory (submodule) was not part of this repo in the beginning, as it has its own requirements and setup (which can be installed in the same manner as in `Installation` section), but for the purpose of keeping all the necessary files under the same repo,
we included this folder here. However, we **recommend you** to separate this folder from the rest of the project tree, as the environment was not tested with both projects running together.
In fact, the `MIMIC-IV-Data-Pipeline` directory actually corresponds to a custom version of [this](https://github.com/healthylaife/MIMIC-IV-Data-Pipeline) repo,
in which several files and the main notebook, `mainPipeline.ipynb`, were modified.
Therefore, for creating the MIMIC subset you need to do the setup and installation steps and
download the MIMIC-IV dataset by following the steps in the `README.md` under `MIMIC-IV-Data-Pipeline` directory.
Otherwise, for obtaining the MIMIC subset used in experiments, please ask the previous student responsible for the project.

If you want a list of all time dependent features in the dataset (which will be saved in a text file) or all possible values for each static feature in the dataset (which will be printed in the standard output)
you can run the `data_inspect.py` script under `scripts` folder. This script also has an optional flag for cleaning possible abnormal rows.
# :deciduous_tree: Code structure
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
│   │   └─── Data_Generation.ipynb ---> notebook for generating the toy dataset and other functionalities
│   └───scripts                    ---> helpers
│       ├─── data_inspect.py
│       ├─── explore_patient.py
│       └─── evaluation_utils.py
├───modn_data                      ---> data folder
│   ├─── MIMIC-IV-Data-Pipeline    ---> data pipeline for creating MIMIC subset folder
│   │    ├─── mainPipeline.ipynb   ---> main notebook for creating the MIMIC subset
│   │    └─── ......
│   └───  <all_CSV_files>.csv
├───plots                          ---> plots folder
├───saved models                   ---> models folder
├───requirements.txt                
├───evaluate.py                    ---> evaluation
├───generate_compare.py            ---> data generation and prediction
└───train.py                       ---> training
```

# :computer: Usage

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

If you do not want to use feature decoders and only train the network to predict the label, you must remove the `--feature_decoding` flag.
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
python generate_compare.py --model_file <checkpoint_name>.pt
                   --output_path <output_dataframe_name>.csv
```
With the above command a dataframe will be predicted using the given model.

If you want to generate data with a model, you need to add the `--generate` flag to the above command. The default data for generation will include all the static variables for each patient, depending on the dataset.
