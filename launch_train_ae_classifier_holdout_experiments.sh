#!/usr/bin/env bash

n_trials=30  # The number of hyperparameter configurations to try
n_repeats=5  # The number of times to repeat the experiment for each hyperparameter configuration
n_epochs=1000  # The number of epochs to train for.
early_stop=100  # The number of epochs to wait before stopping training if the validation loss does not improve.
groupkfold=1

dataset=adenocarcinoma
exp_id=adenocarcinoma_08_20_2024
csv_file=adenocarcinoma_data.csv
path=data
best_features_file=''
update_grid=1
use_l1=1
n_emb=0
prune_network=0
i=0
max_jobs=10
current_jobs=0

for train_after_warmup in 1 0
do
    for warmup_after_warmup in 1 0
    do
        for prune_threshold in 0.0001  
        do
            for variational in 0 1
            do
                for kan in 0
                do
                    for dloss in revTriplet normae no inverseTriplet DANN
                    do
                        ((current_jobs++))
                        cuda=$((i%1)) # Divide by the number of gpus available
                        .conda/bin/python bernn/dl/train/train_ae_classifier_holdout.py --early_stop=$early_stop --n_epochs=$n_epochs \
                            --zinb=0 --kan=$kan --variational=$variational --train_after_warmup=$train_after_warmup --tied_weights=0 --bdisc=1 \
                            --rec_loss=l1 --dloss=$dloss --csv_file=$csv_file --remove_zeros=0 --n_meta=$n_emb \
                            --groupkfold=$groupkfold --embeddings_meta=$n_emb --device=cuda:$cuda --dataset=$dataset --n_trials=$n_trials \
                            --n_repeats=$n_repeats --exp_id=$exp_id --path=$path --pool=0 --log_metrics=1 \
                            --best_features_file=$best_features_file --update_grid=$update_grid --use_l1=$use_l1 --prune_threshold=$prune_threshold --warmup_after_warmup=$warmup_after_warmup &
                        
                        i=$((i+1))
                        
                        if [ $current_jobs -ge $max_jobs ]; then
                            wait -n
                            ((current_jobs--))
                        fi
						sleep 60
                    done
                done
            done
        done
    done
done

wait

# exp_id=testbenchmark_08_15_2024
# train_after_warmup=1
# warmup_after_warmup=1
# prune_threshold=0.0001
# variational=0
# kan=1
# dloss='inverseTriplet'
# cuda=0
# .conda/bin/python bernn/dl/train/train_ae_classifier_holdout.py --early_stop=$early_stop --n_epochs=$n_epochs \
#     --zinb=0 --kan=$kan --variational=$variational --train_after_warmup=$train_after_warmup --tied_weights=0 --bdisc=1 \
#     --rec_loss=l1 --dloss=$dloss --csv_file=$csv_file --remove_zeros=0 --n_meta=$n_emb \
#     --groupkfold=$groupkfold --embeddings_meta=$n_emb --device=cuda:$cuda --dataset=$dataset --n_trials=$n_trials \
#     --n_repeats=$n_repeats --exp_id=$exp_id --path=$path --pool=0 --log_metrics=1 \
#     --best_features_file=$best_features_file --update_grid=$update_grid --use_l1=$use_l1 --prune_threshold=$prune_threshold --warmup_after_warmup=$warmup_after_warmup &
