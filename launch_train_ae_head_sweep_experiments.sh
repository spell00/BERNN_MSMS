#!/usr/bin/env bash
# launch_train_ae_head_sweep_experiments.sh
# ==========================================
# Two-stage Optuna sweep: jointly optimise AE encoder hyperparameters AND
# the classification head type/hyperparameters.
#
# Stage 1 — sweeps head_type + AE params to find the best head family.
# Stage 2 — fixes head_type, sweeps head hyperparameters + AE params.
#
# New in bernn-msms 0.6.3:
#   - Interchangeable sklearn/XGBoost classification heads
#     (xgboost, random_forest, linear_svc, svc_rbf, logistic_regression,
#      knn, gradient_boosting, prototype_mean, prototype_kmeans)
#   - class_triplet loss: supervised triplet margin loss applied on
#     class-labelled embeddings (on top of domain-alignment dloss)
#   - Head is fit every epoch on frozen encoder embeddings; no nn.Linear
#     classification head is used.
#
# Usage:
#   # Dry-run to preview commands:
#   DRY_RUN=1 bash launch_train_ae_head_sweep_experiments.sh
#
#   # Run with 4 parallel jobs on 2 GPUs:
#   MAX_JOBS=4 GPU_COUNT=2 bash launch_train_ae_head_sweep_experiments.sh
#
#   # CPU-only:
#   DEVICE_MODE=cpu bash launch_train_ae_head_sweep_experiments.sh

set -euo pipefail

python_bin=${PYTHON_BIN:-python3}
sleep_seconds=${SLEEP_SECONDS:-30}
dry_run=${DRY_RUN:-0}
log_mlflow=${LOG_MLFLOW:-1}
log_tb=${LOG_TB:-0}
device_mode=${DEVICE_MODE:-cuda}
gpu_count=${GPU_COUNT:-1}
cpu_threads=${CPU_THREADS:-1}
max_jobs=${MAX_JOBS:-1}

# ---- Sweep hyperparameters ----
n_trials=100       # Optuna trials per run
n_epochs=300       # Max epochs per AE training
early_stop=30      # Early-stop patience (epochs)
n_cv=3             # StratifiedKFold folds for head evaluation
groupkfold=1
bs=32

# ---- Dataset / experiment ----
dataset=amide
exp_id=amide_head_sweep_0.6.3
csv_file=adenocarcinoma_data.csv
path=data
remove_zeros=1
log1p=1

if [ "$max_jobs" -lt 1 ]; then
    echo "MAX_JOBS must be >= 1 (got: $max_jobs)"; exit 1
fi
if [ "$gpu_count" -lt 1 ]; then
    echo "GPU_COUNT must be >= 1 (got: $gpu_count)"; exit 1
fi

echo "Launching AE head sweep — DEVICE_MODE=$device_mode MAX_JOBS=$max_jobs CPU_THREADS=$cpu_threads"

i=0
current_jobs=0

# Sweep: variational x domain-loss x class_triplet
# class_triplet adds a supervised triplet margin loss on class-labelled
# embeddings during AE training (new in 0.6.3).
for variational in 0 1
do
    for dloss in inverseTriplet DANN revTriplet normae no
    do
        for class_triplet in 0 1
        do
            current_jobs=$((current_jobs + 1))

            if [ "$device_mode" = "cpu" ]; then
                device=cpu
            else
                cuda=$((i % gpu_count))
                device=cuda:$cuda
            fi

            # Unique study name per combo for safe parallel Optuna workers
            ct_tag=$([ "$class_triplet" -eq 1 ] && echo "ct" || echo "noct")
            study_name="${exp_id}_v${variational}_${dloss}_${ct_tag}"

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
                "$python_bin" -m bernn.dl.train.train_ae_head_sweep \
                    --dataset="$dataset" \
                    --path="$path" \
                    --csv_file="$csv_file" \
                    --remove_zeros=$remove_zeros \
                    --log1p=$log1p \
                    --groupkfold=$groupkfold \
                    --variational=$variational \
                    --dloss="$dloss" \
                    --class_triplet=$class_triplet \
                    --n_trials=$n_trials \
                    --n_epochs=$n_epochs \
                    --early_stop=$early_stop \
                    --n_cv=$n_cv \
                    --bs=$bs \
                    --device="$device" \
                    --exp_id="$exp_id" \
                    --study_name="$study_name" \
                    --log_mlflow=$log_mlflow \
                    --log_tb=$log_tb
            )

            if [ "$dry_run" = "1" ]; then
                printf '%s\n' "${cmd[*]}"
                current_jobs=$((current_jobs - 1))
            else
                "${cmd[@]}" &
            fi

            i=$((i + 1))

            if [ "$dry_run" != "1" ] && [ $current_jobs -ge $max_jobs ]; then
                wait -n || true
                current_jobs=$((current_jobs - 1))
            fi

            if [ "$sleep_seconds" -gt 0 ]; then
                sleep "$sleep_seconds"
            fi
        done  # class_triplet
    done  # dloss
done  # variational

if [ "$dry_run" != "1" ]; then
    wait
fi

echo "All AE head sweep jobs complete."
