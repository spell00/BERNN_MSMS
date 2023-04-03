#!/bin/bash
i=0
n_emb=2
dataset=alzheimer
n_trials=20
n_repeats=5
exp_id=alzheimer_ae_classifier_20
for variational in 0 1
do
	for zinb in 0 1
	do
		for dloss in no revTriplet inverseTriplet DANN normae
		do
      cuda=$((i%2)) # Divide by the number of gpus available
			python3 src/dl/train/mlp/train_ae_classifier_holdout.py --zinb=$zinb --variational=$variational --train_after_warmup=1  --tied_weights=0 --bdisc=1 --rec_loss=l1 --dloss=$dloss --use_mapping=1 --csv_file=unique_genes.csv --remove_zeros=0 --n_meta=0 --groupkfold=1 --embeddings_meta=$n_emb --device=cuda:$cuda --dataset=$dataset --n_trials=$n_trials --n_repeats=$n_repeats --exp_id=$exp_id &
		  i=$((i+1))
    done
	done
done
