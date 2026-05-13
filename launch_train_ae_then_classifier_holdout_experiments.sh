#!/usr/bin/env bash

set -euo pipefail

python_bin=${PYTHON_BIN:-python}
sleep_seconds=${SLEEP_SECONDS:-60}
dry_run=${DRY_RUN:-0}
max_jobs=${MAX_JOBS:-1}
current_jobs=0
log_mlflow=${LOG_MLFLOW:-1}
log_tb=${LOG_TB:-0}
log_dvclive=${LOG_DVCLIVE:-0}

n_trials=30  # The number of hyperparameter configurations to try
n_repeats=5  # The number of times to repeat the experiment for each hyperparameter configuration
n_epochs=1000  # The number of epochs to train for.
early_stop=100  # The number of epochs to wait before stopping training if the validation loss does not improve.
groupkfold=1
train_after_warmup=0

dataset=alzheimer
exp_id=alzheimer_04_16_2025
csv_file=unique_genes.csv
path=data/Alzheimer
update_grid=1
use_l1=1
n_emb=2
prune_network=0

i=0
for variational in 1 0
do
	for kan in 1
	do
		for dloss in inverseTriplet DANN revTriplet normae no 
		do
			current_jobs=$((current_jobs + 1))
		  	cuda=$((i%1)) # Divide by the number of gpus available
			cmd=(
				"$python_bin" bernn/dl/train/train_ae_then_classifier_holdout.py --early_stop=$early_stop --n_epochs=$n_epochs
			--kan=$kan --variational=$variational --train_after_warmup=$train_after_warmup  --tied_weights=0 --bdisc=1 \
			--rec_loss=l1 --dloss=$dloss --csv_file=$csv_file --remove_zeros=0 --n_meta=$n_emb \
			--groupkfold=$groupkfold --embeddings_meta=$n_emb --device=cuda:$cuda --dataset=$dataset --n_trials=$n_trials \
			--n_repeats=$n_repeats --exp_id=$exp_id --path=$path --pool=0 --log_metrics=1 \
			--update_grid=$update_grid --use_l1=$use_l1 --prune_network=$prune_network \
			--log_mlflow=$log_mlflow --log_tb=$log_tb
			)
			if [ "$dry_run" = "1" ]; then
				printf '%s\n' "${cmd[*]}"
			current_jobs=$((current_jobs - 1))
			else
				"${cmd[@]}" &
			fi
		i=$((i+1))
			if [ "$dry_run" != "1" ] && [ $current_jobs -ge $max_jobs ]; then
				wait -n || true
			current_jobs=$((current_jobs - 1))
			fi
			if [ "$sleep_seconds" -gt 0 ]; then
				sleep "$sleep_seconds"
			fi
    	done
	done
done

	if [ "$dry_run" != "1" ]; then
		wait
	fi
