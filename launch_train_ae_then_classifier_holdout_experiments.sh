#!/bin/bash

n_trials=30  # The number of hyperparameter configurations to try
n_repeats=5  # The number of times to repeat the experiment for each hyperparameter configuration
n_epochs=1000  # The number of epochs to train for.
early_stop=10  # The number of epochs to wait before stopping training if the validation loss does not improve.

dataset=alzheimer
exp_id=alzheimer_ae_then_classifier_20  # The name of the experiment. The experiment will have this name
path='data/Alzheimer'
csv_file='unique_genes.csv'
n_emb=0 # Number of embeddings to use for the alzheimer dataset. Should be 0 for any other dataset.
strategy='CU_DEM-AD'  # This variable is only used for the alzheimer dataset. For any other dataset, it will have no effect.
i=0
for variational in 0 1
do
	for zinb in 1
	do
		for dloss in no revTriplet inverseTriplet DANN normae
		do
      	cuda=$((i%2)) # Divide by the number of gpus available
		python3 src/dl/train/train_ae_then_classifier_holdout.py --n_epochs=$n_epochs --pool=1 --path=$path \
			--zinb=$zinb --variational=$variational --train_after_warmup=0 --bdisc=1 --rec_loss=l1 --dloss=$dloss \
			--use_mapping=1 --csv_file=$csv_file --strategy=$strategy --remove_zeros=0 --n_meta=0 --groupkfold=1 \
			--embeddings_meta=$n_emb --device=cuda:$cuda --dataset=$dataset --n_trials=$n_trials --n_repeats=$n_repeats \
			--exp_id=$exp_id --early_stop=$early_stop &
		i=$((i+1))
    done
	done
done

