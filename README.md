# MACRO-Towards-Classification-of-Co-Occurring-Diseases-in-12-Lead-ECGs

Official PyTorch implementation of the paper "MACRO: A Multi-Head Attentional Convolutional Recurrent Network for 
Multi-Label Classification of Co-Occurring Diseases in 12-Lead ECGs"

## Requirements
This project was developed using Python 3.10.12 and PyTorch 2.0.1.
The `requirements.txt` contains all Python libraries that the project depends on, and can be installed using:
```console
pip install -r requirements.txt
```

**Important**: The `PYTHONPATH` variable should be adapted to point to the project root directory to ensure that the
modules can be imported correctly:
```console
export PYTHONPATH=$PYTHONPATH:<root project path>
```

## Model Architectures

### MACRO

![images/MACRO_detailed.png](images/MACRO_detailed.png)

### Multi-Branch MACRO (MB-M)

![images/multibranch_MACRO_detailed_v4.png](images/multibranch_MACRO_detailed_v4.png)

## Dataset  
The dataset used in this study is the CPSC 2018 dataset, which contains 6877 ECG recordings.
We preprocessed the dataset by resampling the ECG signals to 250 Hz and equalizing the ECG signal length to 60 seconds, 
yielding a signal length of T=15,000 data points per recording. 
For the hyperparameter study, we employed a fixed train-valid-test split with ratio 60-20-20,
while for the final evaluations, including the comparison with the state-of-the-art methods and ablation studies, 
we used a 10-fold cross-validation strategy.

Our preprocessed data can be downloaded from [FigShare](https://figshare.com/s/c8d7d9641b283e4ae75b) 
and should be placed in the `data` directory at project root level while maintaining the folder structure. 
In case of using the fixed train-valid-test split, 
the validation set is automatically created from the training set during the training process.

Alternatively, the raw CPSC 2018 dataset can be downloaded from the website of the 
[PhysioNet/Computing in Cardiology Challenge 2020](https://physionet.org/content/challenge-2020/1.0.2/training/cpsc_2018/#files-panel). 
The downloaded `.mat` and `.hea` files can be merged into a single folder and copied to `data/CinC_CPSC/raw`. 
Then, our preprocessing pipeline can be executed manually by running the `preprocesssing/preprocess_cpsc.py` script
to get our preprocessed data and folder structure, including the data split and cross-validation folder.

## Device Preparation
To train the models, we recommend using a GPU. 
The GPU ID (device UUID) can be set in the configuration file `global_config.py` by adapting the 
CUDA_VISIBLE_DEVICES variable. For Nvidia GPUs, the UUID can be retrieved by running the following command:
```console
nvidia-smi -q | grep UUID
```

Moreover, a path for creating the RayTune temporary directory should be set in the `global_config.py` file by adjusting the
TUNE_TEMP_DIR variable. 

## Training and Testing with Fixed Data Split
To train one of the models with a fixed 60-20-20 data split, run the following command with the path to
the corresponding configuration file of interest:
```console
python train.py -c configs/single_run_examples/config_baseline.json
```

To evaluate a trained model on the test set, run the following command:
```console
python test.py --resume <path_to_model_checkpoint> 
```
Example resume path: 
<project_path>/savedVM/models/BaselineModel_SingleRun/<run_id>/model_best.pth

The path to the test set is automatically retrieved from the corresponding config file, which is saved to the model 
path during training. Under this path, the evaluation results are saved as well into a `test_output` folder.



## Training with Hyperparameter Tuning
To train the model with hyperparameter tuning via RayTune, run the following command with the path to 
the corresponding configuration file of interest and the `--tune` flag:
```console
python train.py -c configs/param_study_examples/config.json --tune
```

**Important**: The hyperparameter search space can be defined in the `tuning_params` method of the `train.py` file. 
The metrics configured for the CLI Reporter in the `train_fn` method should match those defined in the config file.
Depending on the GPU size and architecture, the number of parallel trials can be adjusted by setting the `num_gpu` 
variable to the desired value in the `train_fn` method.

## Training with 10-Fold Cross-Validation
To train the model with 10-fold cross-validation, run the following command with the path to the
corresponding configuration file of interest:

```console
python train_with_cv.py -c configs/CV/config.json
```


## Machine Learning Classifiers
To train and evaluate machine learning classifiers, the input and output tensors across all folds need to be retrieved,
while maintaining the fold structure and data split.
For this, run the following command to save the detached tensors to an `output_logging` subdirectory:
```console
python ML_ensemble/retrieve_detached_cross_fold_tensors.py -p <path_to_trained_model>
```
Example path: <project_path>/savedVM/models/Multibranch_MACRO_CV/<run_id>/ 

Afterwards, the ML classifiers can be trained and evaluated based on these tensors by running the following command:
```console
python ML_ensemble/train_ML_models_with_detached_tensors.py
 -p <path_to_trained_model>
 --strategy "gradient_boosting"
```
Example path: <project_path>/savedVM/models/Multibranch_MACRO_CV/<run_id>/ 

By default, all features from the Multi-Branch MACRO (MB-M) and all BranchNets are used (i.e., 117 features). 
For a reduced feature set, the following flags can be passed: 
- `--individual_features` : Predicted probabilities only for the class of interest (13 features)
- `--reduced_individual_features` : Predicted probabilities only for the class of interest w/o MB-M (12 features)

**Important**: Since a parameter study is performed for each class' binary classifier per fold, the whole process can 
take a while, depending on the number of features used. After completion, the results are saved in a 
`ML models/<ML setting>` subdirectory. In order to get an improved, human-readable summary, you can use the 
`summarize_cross_valid_batch.py` script from the `utils` directory described below: 
```console
python utils/summarize_cross_valid_batch.py -p "savedVM/models/Multibranch_MACRO_CV/<run_id>/ML models"
```
For each ML setting, a summary file called `cross_valid_runs_summary.tex` is created in the corresponding subdirectory.

## Script Utils
The `utils` directory contains scripts for the following tasks:
- `summarize_cross_valid_batch.py`: Summarizes the results of multiple cross-validation runs within a folder, one run at a time. 
A path should be specified to the folder containing the cross-validation runs. Example usage:
    ```console
    python utils/summarize_cross_valid_batch.py -p "savedVM/models/BaselineWithMultiHeadAttention_CV"
    ```
- `summarize_cross_valid_for_paper.py`: Fuses the results of the specified cross-validation runs in a suitable format 
for the publication. The paths can be specified at the beginning of the script. 
- `summarize_tune_runs.py`: Can be used to summarize the results of a hyperparameter tuning run within a folder. 
Details need to be configured at the beginning of the script. 
- `tracker.py` and `util.py` : Helper functions (used internally by other code segments)


The `utils_bash` directory contains scripts for the following tasks:
- `clear_all_ckpt_from_folder.sh` : Removes all checkpoint files except `model_best.pth` from the folders specified in RELATIVE_PATHS_TO_FOLDERS, including single training runs, tune runs, and CV runs.
- `run-eval.sh` : Evaluates all models in the specified folder on the test set. Should only be used with single training runs.
- `run-eval-tune.sh` : Evaluates all models in the specified folder on the test set. Should only be used with RayTune runs.

## Acknowledgements and License
This project is inspired by the [PyTorch Template Project](https://github.com/victoresque/pytorch-template) by 
[Victor Huang](https://github.com/victoresque).
It is licensed under the MIT License (see LICENSE for more details).