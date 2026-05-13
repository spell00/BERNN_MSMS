#!/usr/bin/env bash

set -euo pipefail


python_bin=${PYTHON_BIN:-python3}
sleep_seconds=${SLEEP_SECONDS:-60}
dry_run=${DRY_RUN:-0}
log_mlflow=${LOG_MLFLOW:-1}
log_tb=${LOG_TB:-0}
log_dvclive=${LOG_DVCLIVE:-0}
device_mode=${DEVICE_MODE:-cuda}
gpu_count=${GPU_COUNT:-1}
cpu_threads=${CPU_THREADS:-1}
# Add cross-validation and cross-test flags, defaulting to 0, overridable by env
cross_validation=${CROSS_VALIDATION:-1}
cross_test=${CROSS_TEST:-1}

# If not using cross-validation or cross-test, force n_repeats=1
if [ "$cross_validation" -eq 0 ] && [ "$cross_test" -eq 0 ]; then
    n_repeats=1
fi


n_trials=30  # The number of hyperparameter configurations to try
n_repeats=3  # The number of times to repeat the experiment for each hyperparameter configuration
n_epochs=1000  # The number of epochs to train for.
early_stop=100  # The number of epochs to wait before stopping training if the validation loss does not improve.
groupkfold=1

dataset=amide
exp_id=amide_05_05_2026
csv_file=adenocarcinoma_data.csv
path=data
best_features_file=''
update_grid=0
use_l1=1
n_emb=0
prune_network=0
i=0
max_jobs=${MAX_JOBS:-1}
current_jobs=0

if [ "$max_jobs" -lt 1 ]; then
    echo "MAX_JOBS must be >= 1 (got: $max_jobs)"
    exit 1
fi

if [ "$gpu_count" -lt 1 ]; then
    echo "GPU_COUNT must be >= 1 (got: $gpu_count)"
    exit 1
fi

if [ "$cpu_threads" -lt 1 ]; then
    echo "CPU_THREADS must be >= 1 (got: $cpu_threads)"
    exit 1
fi

echo "Launching with DEVICE_MODE=$device_mode MAX_JOBS=$max_jobs CPU_THREADS=$cpu_threads"

for train_after_warmup in 0 1
do
    for warmup_after_warmup in 1 0
    do
        for prune_threshold in 0
        do
            for variational in 0 1
            do
                for kan in 0
                do
                    for dloss in inverseTriplet DANN revTriplet normae no
                    do
                        current_jobs=$((current_jobs + 1))
                        if [ "$device_mode" = "cpu" ]; then
                            device=cpu
                        else
                            cuda=$((i%gpu_count)) # Divide by the number of gpus available
                            device=cuda:$cuda
                        fi

                        cmd=(
                            env
                            PYTHONPATH=$PWD
                            OMP_NUM_THREADS=$cpu_threads
                            MKL_NUM_THREADS=$cpu_threads
                            OPENBLAS_NUM_THREADS=$cpu_threads
                            NUMEXPR_NUM_THREADS=$cpu_threads
                            VECLIB_MAXIMUM_THREADS=$cpu_threads
                            TF_NUM_INTRAOP_THREADS=$cpu_threads
                            TF_NUM_INTEROP_THREADS=1
                            "$python_bin" -m bernn.dl.train.train_ae_classifier_holdout --early_stop=$early_stop --n_epochs=$n_epochs \
                            --kan=$kan --variational=$variational --train_after_warmup=$train_after_warmup --tied_weights=0 --bdisc=1 \
                            --rec_loss=l1 --dloss=$dloss --csv_file=$csv_file --remove_zeros=0 \
                            --groupkfold=$groupkfold --device=$device --dataset=$dataset --n_trials=$n_trials \
                            --n_repeats=$n_repeats --exp_id=$exp_id --path=$path --pool=0 --log_metrics=1 \
                            --best_features_file=$best_features_file --update_grid=$update_grid --use_l1=$use_l1 \
                            --prune_threshold=$prune_threshold --warmup_after_warmup=$warmup_after_warmup \
                            --prune_network=$prune_network --log_mlflow=$log_mlflow \
                            --log_tb=$log_tb --log_dvclive=$log_dvclive \
                            --cross_validation=$cross_validation --cross_test=$cross_test
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
        done
    done
done

if [ "$dry_run" != "1" ]; then
    wait
fi

