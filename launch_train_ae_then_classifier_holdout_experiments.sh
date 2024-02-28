#!/bin/bash

n_trials=20  # The number of hyperparameter configurations to try
n_repeats=5  # The number of times to repeat the experiment for each hyperparameter configuration
n_epochs=1000  # The number of epochs to train for.
early_stop=10  # The number of epochs to wait before stopping training if the validation loss does not improve.
groupkfold=0

dataset=cancers
exp_id=tissues_train_after_warmup
csv_file=proteome_log_cancers_tissues.csv
path=data/PanCancer

i=0
for variational in 0 1
do
	for zinb in 1
	do
		for dloss in no revTriplet inverseTriplet DANN normae
		do
      	cuda=$((i%1)) # Divide by the number of gpus available
		python3 src/dl/train/train_ae_then_classifier_holdout.py --n_epochs=$n_epochs --pool=0 --path=$path \
			--zinb=$zinb --variational=$variational --train_after_warmup=0 --bdisc=1 --rec_loss=l1 --dloss=$dloss \
			--use_mapping=1 --csv_file=$csv_file --strategy=$strategy --remove_zeros=0 --n_meta=0 --groupkfold=$groupkfold \
			--embeddings_meta=$n_emb --device=cuda:$cuda --dataset=$dataset --n_trials=$n_trials --n_repeats=$n_repeats \
			--exp_id=$exp_id --early_stop=$early_stop &
		i=$((i+1))
    done
	done
done

