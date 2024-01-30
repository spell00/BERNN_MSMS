#!/usr/bin/env python3

n_trials=50  # The number of hyperparameter configurations to try
n_repeats=5  # The number of times to repeat the experiment for each hyperparameter configuration
n_epochs=20  # The number of epochs to train for.
early_stop=10  # The number of epochs to wait before stopping training if the validation loss does not improve.

dataset=multi
n_trials=30
n_repeats=5
exp_id=reviewer_response
early_stop=10
csv_file=matrix.csv
path=data/PXD015912

i=0
for variational in 0 1
do
	for zinb in 0
	do
		for dloss in no revTriplet inverseTriplet DANN normae
		do
      	cuda=$((i%1)) # Divide by the number of gpus available
		.conda/bin/python bernn/dl/train/train_ae_classifier_holdout.py --early_stop=$early_stop --n_epochs=$n_epochs \
			--zinb=$zinb --variational=$variational --train_after_warmup=1  --tied_weights=0 --bdisc=1 \
			--rec_loss=l1 --dloss=$dloss --csv_file=$csv_file --remove_zeros=0 --n_meta=0 \
			--groupkfold=1 --embeddings_meta=0 --device=cuda:$cuda --dataset=$dataset --n_trials=$n_trials \
			--n_repeats=$n_repeats --exp_id=$exp_id --path=data/PXD015912 --pool=0 --log_metrics=1&
		i=$((i+1))
    done
	done
done
