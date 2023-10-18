#!/bin/bash

n_trials=30  # The number of hyperparameter configurations to try
n_repeats=5  # The number of times to repeat the experiment for each hyperparameter configuration
n_epochs=1000  # The number of epochs to train for.
early_stop=10  # The number of epochs to wait before stopping training if the validation loss does not improve.

dataset=custom
n_trials=30
n_repeats=3
exp_id=adenocarcinoma_ae_classifier_holdout_20
early_stop=10
csv_file=adenocarcinoma_data.csv

i=0
for variational in 0 1
do
	for zinb in 0
	do
		for dloss in no revTriplet inverseTriplet DANN normae
		do
      	cuda=$((i%2)) # Divide by the number of gpus available
		python3 src/dl/train/train_ae_classifier_holdout.py --early_stop=$early_stop --n_epochs=$n_epochs \
			--zinb=$zinb --variational=$variational --train_after_warmup=1  --tied_weights=0 --bdisc=1 \
			--rec_loss=l1 --dloss=$dloss --use_mapping=1 --csv_file=$csv_file --remove_zeros=0 --n_meta=0 \
			--groupkfold=1 --embeddings_meta=0 --device=cuda:$cuda --dataset=$dataset --n_trials=$n_trials \
			--n_repeats=$n_repeats --exp_id=$exp_id &
		i=$((i+1))
    done
	done
done
