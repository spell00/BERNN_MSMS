#!/bin/bash
n_epochs=1000
i=0
n_emb=2
dataset=adenocarcinoma
n_trials=30
n_repeats=5
exp_id=adenocarcinoma_ae_classifier_holdout_20
early_stop=100
csv_file=adenocarcinoma_data.csv
for variational in 0 1
do
	for zinb in 0
	do
		for dloss in no revTriplet inverseTriplet DANN normae
		do
      cuda=$((i%2)) # Divide by the number of gpus available
			python3 src/dl/train/train_ae_classifier_holdout.py --early_stop=$early_stop --n_epochs=$n_epochs --zinb=$zinb --variational=$variational --train_after_warmup=1  --tied_weights=0 --bdisc=1 --rec_loss=l1 --dloss=$dloss --use_mapping=1 --csv_file=unique_genes.csv --remove_zeros=0 --n_meta=0 --groupkfold=1 --embeddings_meta=$n_emb --device=cuda:$cuda --dataset=$dataset --n_trials=$n_trials --n_repeats=$n_repeats --exp_id=$exp_id &
		  i=$((i+1))
    done
	done
done
