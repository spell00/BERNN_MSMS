# BERNN-MSMS: Batch Effect Removal Neural Networks for Tandem Mass Spectrometry

## Author

* Simon Pelletier

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
Each command runs 20 trials of 5 different splits of the data. It might take over an hour to run each demo run.
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

# Run experiments
To launch experiments, use the two bash files (launch_train_ae_then_classifier_holdout_experiment.sh or 
launch_train_ae_classifier_holdout_experiment.sh).

It was designed to run on two GPUs of 24 GB each. If not enough GPU RAM available, modify the file to reduce the
number of models trained simultaneously. The number of GPU can also be modified.

Each experiment might take a few days to run. The results will be saved in the mlflow folder.

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


## Parameters
    --dataset (str): ['alzheimer', 'amide', 'mice']
    --n_trials (int): Number of trials for the baeysian optimization of hyperparameters
    --n_repeats (int): Number of repeats in the repetitive holdout
    --exp_id (str): Name of the mlflow experiment
    --device (str): Name of the device to use ['cuda:0', 'cuda:1', ...]
    --use_mapping (bool): Use mapping of the batch ID into the decoder
    --rec_loss (str): Reconstruction loss type ['l1', 'mse']
    --variational (boolean): Use a variational autoencoder?
    --groupkfold (boolean): Use group k-fold? With this command, all the samples from the same batch will be in the same set. E.g. All samples from batch 1 will all be either in the training, validation or test set (https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html)
    --tied_weights (boolean): Use Autoencoders with tied weights?
    --train_after_warmup (boolean): Train the autoencoder after warmup? ()
    --dloss (str): Domain loss ['no', 'revTriplet', 'invTriplet', 'DANN', 'normae']

## Hyperparameters

The hyperparameters are optimized using Bayesian optimization. They are defined at the end of each train script, which 
are located in src/dl/train.
The parameters are the following:

    dropout (float): Number of neurons that are randomly dropped out. 
                     0.0 <= thres < 1.0
    smoothing (float): Label smoothing replaces one-hot encoded label vector 
                       y_hot with a mixture of y_hot and the uniform distribution:
                       y_ls = (1 - α) * y_hot + α / K
    margin (float): Margin for the triplet loss 
                    (https://towardsdatascience.com/triplet-loss-advanced-intro-49a07b7d8905)
    gamma (float): Controls the importance given to the batches adversarial loss
    beta (float): Controls the importance given to the Kullback-Leibler loss
    nu (float): Controls the importance given to the classification loss
    layer1 (int): The number of neurons the the first hidden layer of the encoder and the
                  last hidden layer of the decoder
    layer2 (int): The number of neurons the the second hidden layer (Bottleneck)
    ncols (int): Number of features to keep
    lr (float): Model's optimization learning rate
    wd (float): Weight decay value
    scale (categorical): Choose between ['none', 'minmax', 'robust', 'standard', 'minmax_per_batch', 'robust_per_batch', 'standard_per_batch']

