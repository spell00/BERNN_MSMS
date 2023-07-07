# BERNN-MSMS: Batch Effect Removal Neural Networks for Tandem Mass Spectrometry

## Author

* Simon Pelletier

# Install
All install steps should be done in the root directory of the project. <br/>
Everything should take only a few minutes to install,
though it could be longer depending on your internet connection. <br/>
The package 

## Install python dependencies
`pip install -r requirements.txt`

## Install R dependencies
`install.packages("harmony")`
`install.packages("sva")`
`devtools::install_github("immunogenomics/lisi")`

## Install package
`pip install -e .`

# Run experiments
To launch experiments, use the two bash files (launch_train_ae_then_classifier_holdout_experiment.sh or 
launch_train_ae_classifier_holdout_experiment.sh).

It was designed to run on two GPUs of 24 GB each. If not enough GPU RAM available, modify the file to reduce the
number of models trained simultaneously. The number of GPU can also be modified.

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

Use `train_ae_then_classifier_holdout.py` 
to train a model that freezes the autoencoder and DANN/revTriplet/invTriplet layers of the network after the warmup. 
The labels classifier is then trained alone after the warmup. The models for the alzheimer dataset are trianed
using this file.

Use `train_ae_classifier_holdout.py` to keep the autoencoder and 
DANN/revTriplet/invTriplet layers of the network after the warmup. The models for the datasets amide (adenocarcinoma) amd mice 
(AgingMice) are trained using this file.

## Observe results from a server on a local machine 
On local machine terminal:<br/>
`mlflow ui`

Open in browser:<br/>
`http://127.0.0.1:5000/`

On server:<br/>
`mlflow server --host=0.0.0.0`

Open in local browser:<br/>
`http://<ip-adress>:5000/`


## Parameters
    --dataset (str): ['alzheimer', 'amide', 'mice']
    --n_trials (int): Number of trials for the baeysian optimization of hyperparameters
    --n_repeats (int): Number of repeats in the repetitive holdout
    --exp_id (str): Name of the mlflow experiment
    --device (str): Name of the device to use ['cuda:0', 'cuda:1', ...]
    --use_mapping (bool): Use mapping of the batch ID into the decoder
    --rec_loss (str): Reconstruction loss type ['l1', 'mse']
    --variational (boolean): Use a variational autoencoder?
    --tied_weights (boolean): Use Autoencoders with tied weights?
    --train_after_warmup (boolean): Train the autoencoder after warmup?
    --dloss (str): Domain loss ['no', 'revTriplet', 'invTriplet', 'DANN', 'normae']

## Hyperparameters
    dropout (float): Number of neurons that are randomly dropped out. 
                     0.0 <= thres < 1.0
    smoothing (float): Label smoothing replaces one-hot encoded label vector 
                       y_hot with a mixture of y_hot and the uniform distribution:
                       y_ls = (1 - α) * y_hot + α / K
    margin (float): Margin for the triplet loss 
                    (https://towardsdatascience.com/triplet-loss-advanced-intro-49a07b7d8905)
    gamma (float): Controls the importance given to the batches adversarial loss
    beta (float): Controls the importance given to the Kullback-Leibler loss
    zeta (float): Controls the importance given to the ZINB loss
    nu (float): Controls the importance given to the classification loss
    layer1 (int): The number of neurons the the first hidden layer of the encoder and the
                  last hidden layer of the decoder
    layer2 (int): The number of neurons the the second hidden layer (Bottleneck)
    ncols (int): Number of features to keep
    lr (float): Model's optimization learning rate
    wd (float): Weight decay value
    scale (categorical): Choose between ['none', 'minmax', 'robust', 'standard', 'minmax_per_batch', 'robust_per_batch', 'standard_per_batch']

