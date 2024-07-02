#!/usr/bin/env python3

n_trials=30  # The number of hyperparameter configurations to try
n_repeats=5  # The number of times to repeat the experiment for each hyperparameter configuration
n_epochs=1000  # The number of epochs to train for.
early_stop=100  # The number of epochs to wait before stopping training if the validation loss does not improve.
groupkfold=1
train_after_warmup=1

dataset=alzheimer
exp_id=alzheimer_06_27_2024
csv_file=unique_genes.csv
path=data/Alzheimer
best_features_file=''
update_grid=1
use_l1=1

i=0
for variational in 0 1
do
	for kan in 1
	do
		for dloss in revTriplet normae no  inverseTriplet DANN
		do
      	cuda=$((i%1)) # Divide by the number of gpus available
		.conda/bin/python bernn/dl/train/train_ae_classifier_holdout.py --early_stop=$early_stop --n_epochs=$n_epochs \
			--zinb=0 --kan=$kan --variational=$variational --train_after_warmup=$train_after_warmup  --tied_weights=0 --bdisc=1 \
			--rec_loss=l1 --dloss=$dloss --csv_file=$csv_file --remove_zeros=0 --n_meta=0 \
			--groupkfold=$groupkfold --embeddings_meta=0 --device=cuda:$cuda --dataset=$dataset --n_trials=$n_trials \
			--n_repeats=$n_repeats --exp_id=$exp_id --path=$path --pool=0 --log_metrics=1 \
			--best_features_file=$best_features_file --update_grid=$update_grid --use_l1=$use_l1 &
		i=$((i+1))
    	done
	done
	wait
done
