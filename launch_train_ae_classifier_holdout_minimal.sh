#!/usr/bin/env bash

set -euo pipefail

python_bin=${PYTHON_BIN:-python}
device_mode=${DEVICE_MODE:-cpu}
gpu_count=${GPU_COUNT:-1}
cpu_threads=${CPU_THREADS:-1}

echo "Launching minimal test: DEVICE_MODE=$device_mode CPU_THREADS=$cpu_threads"

n_trials=1
n_repeats=1
n_epochs=1
early_stop=1
groupkfold=1

dataset=amide
exp_id=minimal_test
csv_file=adenocarcinoma_data.csv
path=data
best_features_file=''
update_grid=0
use_l1=1
n_emb=0
prune_network=0
max_jobs=1
current_jobs=0

device=cpu
if [ "$device_mode" != "cpu" ]; then
    cuda=0
    device=cuda:$cuda
fi

cmd=(
    env
    OMP_NUM_THREADS=$cpu_threads
    MKL_NUM_THREADS=$cpu_threads
    OPENBLAS_NUM_THREADS=$cpu_threads
    NUMEXPR_NUM_THREADS=$cpu_threads
    VECLIB_MAXIMUM_THREADS=$cpu_threads
    TF_NUM_INTRAOP_THREADS=$cpu_threads
    TF_NUM_INTEROP_THREADS=1
    "$python_bin" bernn/dl/train/train_ae_classifier_holdout.py --early_stop=$early_stop --n_epochs=$n_epochs \
    --kan=0 --variational=0 --train_after_warmup=1 --tied_weights=0 --bdisc=1 \
    --rec_loss=l1 --dloss=DANN --csv_file=$csv_file --remove_zeros=0 --n_meta=$n_emb \
    --groupkfold=$groupkfold --embeddings_meta=$n_emb --device=$device --dataset=$dataset --n_trials=$n_trials \
    --n_repeats=$n_repeats --exp_id=$exp_id --path=$path --pool=0 --log_metrics=1 \
    --best_features_file=$best_features_file --update_grid=$update_grid --use_l1=$use_l1 \
    --prune_threshold=0 --warmup_after_warmup=0 --prune_network=$prune_network \
    --log_mlflow=1 --log_tb=0 --log_dvclive=0
)

printf '%s\n' "${cmd[@]}"
"${cmd[@]}"
