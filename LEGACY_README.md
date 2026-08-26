# BERNN-MSMS: Batch Effect Removal Neural Networks for Tandem Mass Spectrometry. 

> Legacy note
>
> This README is kept mainly as historical/legacy CLI guidance.
> For current usage, prefer the Python API pattern shown below and use
> `TRAINING_PARAMETERS.md` for the complete parameter reference.

## Author

* Simon Pelletier

# Quickstart
The package was tested on Windows 10 and Ubuntu 20.04.4 LTS with Python 3.10.11 and R 4.2.2. 

Due to BERNN has a lot of dependencies, so the easiest way to use it with your data is to use the Docker image using `singularity`. To do so, follow these steps:

First, get the Docker image (should take ~5 to 10 minutes)
`docker pull spel00/bernn:latest`

Finally, use the Docker image with singularity. For example, 
`singularity exec docker://spel00/bernn:latest python bernn/dl/train/train_ae_classifier_holdout.py --device=cpu --dataset=custom --n_trials=20 --n_repeats=5 --exp_id=benchmark --path=data/benchmark --csv_file=intensities.csv --pool=0`

or

`singularity exec docker://spel00/bernn:latest python bernn/dl/train/train_ae_then_classifier_holdout.py --device=cpu --dataset=custom --n_trials=20 --n_repeats=5 --exp_id=benchmark --path=data/benchmark --csv_file=intensities.csv --pool=0`

To run the examples above on GPU, you need to install the `nvidia-container-toolkit` on your machine. On Debian/Ubuntu, install using the command `apt-get install -y nvidia-container-toolkit`. You also need to add the option `--nv` the command to the above commands. For example, the first `singularity` command becomes:

`singularity exec --nv docker://spel00/bernn:latest python bernn/dl/train/train_ae_classifier_holdout.py --device=cpu --dataset=custom --n_trials=20 --n_repeats=5 --exp_id=benchmark --path=data/benchmark --csv_file=intensities.csv --pool=0`

The example above should take ~20 minutes per repetition to run on an Nvidia A100. Here their is 5 repeats per trial, so ~1h30 hour per trial (each trial is to test a new hyperparameters combination). Thus, for 20 trials it should take at least 1 day. The training time is highly dependent on the size of the dataset.

To use BERNN with your own dataset, replace `--path=data` with the path to the data directory that contains the data, replace `--csv_name=intensities.csv` with the name of the csv containing the data and replace `--exp_id=benchmark` with a name for the experiment.

The csv format is specified in the section `Custom experiments` below.

# Install
All install steps should be done in the root directory of the project. <br/>
Everything should take only a few minutes to install,
though it could be longer depending on your internet connection. <br/>
The package was tested on Windows 10 and Ubuntu 20.04.4 LTS with Python 3.10.11 and R 4.2.2. 

## Install python dependencies
`pip install -r requirements.txt`

## Install R dependencies
`install.packages("harmony")`
`install.packages("sva")`
`devtools::install_github("immunogenomics/lisi")`

## Install package
`pip install -e .`

## Other requirements
The scripts should be run on a machine with a GPU that supports CUDA (which should be installed). <br/>
To verify that CUDA is installed, run the command `nvidia-smi` on a terminal. If it is installed, the CUDA version shoud appear. </br>
To verify that pyTorch is properly installed with CUDA support, run the 

## Training scripts
The main scripts for training models are located in src/dl/train. <br/>
Use `train_ae_then_classifier_holdout.py` 
to train a model that freezes the autoencoder and DANN/revTriplet/invTriplet layers of the network after the warmup. 
The labels classifier is then trained alone after the warmup. The models for the alzheimer dataset reach better
scores using this file. <br/>

Use `train_ae_classifier_holdout.py` to keep training the autoencoder and 
DANN/revTriplet/invTriplet layers of the network after the warmup.
The models for the datasets amide (adenocarcinoma) and mice 
(AgingMice) reach better classification scores using this file.

## Demo run
Here are some commands to try the training scripts with only a few epochs. The number of warmup epochs is a tunable 
hyperparameter, thus for a demo run it can be lowered directly in the training scripts, close to the end of the script.
Look for the where the class TrainAE is instantiated, the parameters are right after. By default, the number of warmup
epochs is between 10 and 250. For a demo run, it can be lowered to 1 and 10.
<br/>
Each command runs 20 trials of 5 different splits of the data.
<br/>

These are minimal examples. For more complete descriptions of the available arguments, see the section [Arguments](##Arguments) below.


### Alzheimer dataset
In the root directory of the project, run the following commands:<br/>

`python src\dl\train\train_ae_then_classifier_holdout.py --groupkfold=1 --embeddings_meta=2 --device=cuda:0 --n_epochs=10 --dataset=alzheimer --n_trials=20 --n_repeats=5 --exp_id=test_alzheimer1 --path=data/Alzheimer/`

`python src\dl\train\train_ae_classifier_holdout.py --groupkfold=1 --embeddings_meta=2 --device=cuda:0 --n_epochs=10 --dataset=alzheimer --n_trials=20 --n_repeats=5 --exp_id=test_alzheimer2 --path=data/Alzheimer/`

### Adenocarcinoma dataset
In the root directory of the project, run the following command:<br/>

`python src\dl\train\train_ae_then_classifier_holdout.py --groupkfold=1 --device=cuda:0 --dataset=amide --n_trials=20 --n_repeats=5 --exp_id=test_amide1 --path=data/`

`python src\dl\train\train_ae_classifier_holdout.py --groupkfold=1 --device=cuda:0 --dataset=amide --n_trials=20 --n_repeats=5 --exp_id=test_amide2 --path=data/`

### AgingMice dataset
In the root directory of the project, run the following command:<br/>

`python src\dl\train\train_ae_then_classifier_holdout.py --groupkfold=1 --device=cuda:0 --dataset=mice --n_trials=20 --n_repeats=5 --exp_id=test_mice1 --path=data/`

`python src\dl\train\train_ae_classifier_holdout.py --groupkfold=1 --device=cuda:0 --dataset=mice --n_trials=20 --n_repeats=5 --exp_id=test_mice2 --path=data/`

### Custom dataset
In the root directory of the project, run the following command:<br/>

`python src\dl\train\train_ae_then_classifier_holdout.py --groupkfold=1 --device=cuda:0 --dataset=custom --n_trials=20 --n_repeats=5 --exp_id=<NameOfExperiment> --path=<path/to/folderContainingCsvFile> --csv_name<csvFileName>`

`python src\dl\train\train_ae_classifier_holdout.py --groupkfold=1 --device=cuda:0 --dataset=custom --n_trials=20 --n_repeats=5 --exp_id=<NameOfExperiment> --path=<path/to/folderContainingCsvFile> --csv_name<csvFileName>`


# Run experiments
To reproduce the 3 experiments from the [BERNN paper](https://pubmed.ncbi.nlm.nih.gov/37461653/), use the two bash files (`launch_train_ae_then_classifier_holdout_experiment.sh` or 
`launch_train_ae_classifier_holdout_experiment.sh`). These files can only be run on Linux. The first script was used to train the models for the Alzheimer dataset and the second for the adenocarcinoma and AgingMice datasets. <br/>

These files can also be modified to run on any other csv files, given they follow the structure described in the next subsection () of this README file.

It was designed to run on two NVIDIA RTX 3090 GPUs of 24 GB each. If not enough GPU RAM available, modify the file to reduce the
number of models trained simultaneously. The number of GPU can also be modified.

Each experiment might take a few days to run, depending on the size of the dataset. The results will be saved in the `mlflow` folder.

## Custom experiments
To run with custom data, you need to change the parameter `--dataset` to `custom` and change `--path` to the path 
relative to this `README` file. Alternatively, you can set it to the absolute path where the data is. 

Your dataset must be:
- comma seperated (csv)
- Rows are samples
- Columns are features
- first column must be the sample IDs
- second column must be the labels
- third column must be the batch IDs

# Train scripts
The main scripts for training models are located in src/dl/train. 

# Observe results
## Observe results from a local machine 
On local machine terminal:<br/>
`mlflow ui`

Open in browser:<br/>
`http://127.0.0.1:5000/`

## Observe results from a server on a local machine 
On server, use the command:<br/>
`mlflow server --host=0.0.0.0`

Open in local browser:<br/>
`http://<server-ip-adress>:5000/`

Example of a mlflow user interface: <br/>

![interface2](images/mlflow_ex2.png)

When clicking on the experiment's run name, a menu where the run id, parameters, metrics and images from multiple analysis can be found.

![interface2](images/PCA_batches.png)

## Logs
All results are also logged in the `logs` folder, which is automatically created when the first results are generated. All results are stored either in `logs/ae_classifier_holdout` or `logs/ae_then_classifier_holdout`, depending on the training scenario. The ids of the files in which each run is stored can be found in `mlflow`. To find it, first click on the experiment's name , then on the run's name (see 1st image of the example of the mlflow user interface). The run id can be found on the top of the page (see example 2nd image).

All runs logs can be accessed this way, be more conveniently, the results of the best run of each of BERNN's models are found in the `logs/best_models` folder (see the image below).

![best_logs](images/best_logs.png)

### shap folder

#### Beeswarm Deep Explainer

#### summary_DeepExplainer

### CSV files

#### Encoded representations
The `encs.csv` file contains the encoded values given by the encoder of the autoencoder. They are the values that are used for the classification neural network. 

The rows are the samples and the columns are the learned features. Each row contains the results of a sample and the columns are the features, except for the first column which contains the samples names, the second column contains the batch numbers and the third contains the labels.

#### Reconstructed representations
The `recs.csv` file contains the reconstructed features from the trained model. The rows are the samples and the columns are the initial features. Please note that the features are corrected for batch effects, but might not be as accurate as the encoded values to train a classifier. They could be used for downstream analyses (e.g. for differential analysis), but some biological signal could be lost compared to the encoded features. Rather than using this approach, we recommend using the shap values to get the features importance for the classification task, even if classification is not the end goal of the experiment. 

The rows are the samples and the columns are the learned features. Each row contains the results of a sample and the columns are the features, except for the first column which contains the samples names, the second column contains the batch numbers and the third contains the labels.


#### shap

The `shap` folder contains the values for the shap analysis. `beeswarmp` contains contains the values used to get the `beeswarm` plot, which contains the importance of each feature for the decision of each individual samples. `OTHER ONE` contains the values for the overall importance of each features.

#### predictions
train_predictions.csv, etc
contains the prediction scores for each of the samples. (TODO: Columns need to be defined)

### Models weights

the `.pth` files contain the weights of each of the models trained. `model_i` (where `i` is the i'th model) are each of the final models. `warmup.pth` is the unsupervised model learned during the warmup.

## Get best results and all batch correction metrics
To make a summary of the results obtained in an experiment, use the command: `python3 mlflow_eval_runs.py --exp_name <NameOfExperiment>`<br\>


## Parameters and API reference

The complete, exhaustive parameter reference is documented in:

- [TRAINING_PARAMETERS.md](TRAINING_PARAMETERS.md)

This includes all parameter families:

- Every field in `TrainingConfig` (defaults, expected type, meaning).
- Direct constructor overrides supported by both holdout trainers.
- Non-config constructor flags (logging, pooling, split behavior, etc.).
- Runtime `fit` and `fit_predict` arguments and data contracts.
- Optimization parameter keys consumed during training/Ax runs.

### Most common parameters (quick list)

- `n_epochs`: total epochs.
- `dloss`: domain loss mode (`revTriplet`, `revDANN`, `DANN`, `inverseTriplet`, `normae`, `no`).
- `variational`: enables VAE behavior.
- `n_layers`: number of classifier hidden layers.
- `layer1`: first hidden width seed (deeper defaults are auto-derived).
- `n_repeats`: number of holdout repeats.
- `warmup`: warmup epochs.
- `device`: `cpu` or `cuda*`.
- `kan`: enable KAN behavior.
- `scaler`: feature scaling strategy.
- `bs`: batch size.
- `n_trials`: number of optimization trials.

### Recommended Python API usage

```python
from bernn import TrainAEClassifierHoldout

trainer_cls = TrainAEClassifierHoldout
trainer = trainer_cls(config=bernn_config, log_metrics=True, keep_models=False)

# Train and predict in one call
preds_encoded = trainer.fit_predict(
    X_train,
    y_train,
    X_test=X_test,
    y_test=y_test,
    groups_train=batches_train,
    groups_test=batches_test,
    cross_validation=False,
    cross_test=False,
)

# Decode predictions back to original labels
preds = trainer.predict(X_test)
```

### Constructor precedence

When using constructor overrides and config together, precedence is:

1. Direct constructor arguments (when non-`None`).
2. Values from `config`.
3. `TrainingConfig` defaults.

### Mandatory batch-ID contract

- `groups_train` is mandatory for training.
- If `X_test` is provided, `groups_test` is mandatory.

This contract is enforced to avoid ambiguous domain/batch behavior.

