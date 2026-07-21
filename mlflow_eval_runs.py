import csv
import json
import mlflow
import numpy as np
import itertools
import argparse
from pathlib import Path


# ---------------------------------------------------------------------------
# Head-sweep experiments (bernn-msms 0.6.3+) log a params.head_type field.
# Detect them so we can extend the grouping key and model-name label.
# ---------------------------------------------------------------------------

HEAD_TYPES = [
    "xgboost", "random_forest", "linear_svc", "svc_rbf",
    "logistic_regression", "knn", "gradient_boosting",
    "prototype_mean", "prototype_kmeans",
]


def setup_results_dir(exp_name, exp_id):
    """Create and return path to results directory for this experiment."""
    results_dir = Path('results') / f"{exp_name}_{exp_id}"
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def _is_head_sweep_experiment(runs):
    """Return True when the MLflow runs contain head-sweep params (0.6.3+)."""
    return 'params.head_type' in runs.columns and runs['params.head_type'].notna().any()


def _group_label(gg, params_varied):
    """
    Build the model-name string used as dict key / CSV label.

    Legacy format (dloss, variational, kan):
        '<dloss><vae><kan>'  e.g. 'inverseTripletvaekán'

    Extended format (adds class_triplet and/or head_type):
        'dloss=X | variational=Y | class_triplet=Z | head_type=W'
    """
    if isinstance(gg, str):
        return gg

    parts = list(gg)
    varied = list(params_varied)

    extended_fields = {'params.class_triplet', 'params.head_type'}
    if not extended_fields.intersection(set(varied)):
        # Legacy: '<dloss><vae><kan>'
        prefixes = ['', 'vae', 'kan']
        return '_'.join([''.join([s, str(x)]) for s, x in zip(prefixes, parts)])

    label_parts = []
    for col, val in zip(varied, parts):
        short = col.replace('params.', '')
        label_parts.append(f"{short}={val}")
    return ' | '.join(label_parts)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--exp_name', type=str, default='testbenchmark_08_15_2024',
        help='Name of the experiment to evaluate',
    )
    parser.add_argument(
        '--head_sweep', action='store_true', default=False,
        help='Force head-sweep grouping (params.head_type included). '
             'Auto-detected when params.head_type exists in runs.',
    )
    args = parser.parse_args()

    exp_name = args.exp_name

    # the adenocarcinoma dataset (amide) has only 3 batches = 3 holdout splits
    if 'amide' in exp_name or 'bactTest' in exp_name or 'Adeno' in exp_name or 'adeno' in exp_name:
        n_per_run = 3
    else:
        n_per_run = 5

    exp_id = mlflow.get_experiment_by_name(exp_name).experiment_id

    results_dir = setup_results_dir(exp_name, exp_id)

    print(n_per_run, exp_id)

    runs = mlflow.search_runs(exp_id)
    runs.index = runs['run_id']

    # -----------------------------------------------------------------------
    # Determine grouping params:
    #   Legacy (pre-0.6.3):  dloss x variational x kan
    #   0.6.3+ head sweep:   dloss x variational x class_triplet x head_type
    #   0.6.3+ holdout:      dloss x variational x kan x class_triplet
    # -----------------------------------------------------------------------
    use_head_sweep = args.head_sweep or _is_head_sweep_experiment(runs)

    has_class_triplet_col = (
        'params.class_triplet' in runs.columns
        and runs['params.class_triplet'].notna().any()
    )

    if use_head_sweep:
        params_varied = ['params.dloss', 'params.variational', 'params.class_triplet', 'params.head_type']
        params_varied = [p for p in params_varied if p in runs.columns]
    else:
        params_varied = ['params.dloss', 'params.variational', 'params.kan']
        if has_class_triplet_col:
            params_varied.append('params.class_triplet')

    print(f"Grouping by: {params_varied}")

    groups = list(runs.groupby(params_varied).run_id)

    client = mlflow.tracking.MlflowClient()
    param_cols = [c for c in runs.columns if c.startswith("params")]

    metrics = {_group_label(g[0], params_varied): {} for g in groups}
    best_metrics = {g: None for g in metrics}
    best_mccs = {g: -np.inf for g in metrics}

    c = 0
    for gg, run_ids in groups:
        g = _group_label(gg, params_varied)
        print(f"Group: {g}")
        for r in run_ids:
            metrics[g][r] = {col[8:]: -np.inf for col in runs.columns if col.startswith("metrics")}
            c += 1

            # Skip runs that haven't completed enough holdout repeats
            if len([s.value for s in client.get_metric_history(r, 'valid/mcc')]) < n_per_run:
                continue

            for metric in list(metrics[g][r]):
                try:
                    values = [s.value for s in client.get_metric_history(r, metric)]
                except mlflow.exceptions.MlflowException:
                    continue
                if values:
                    try:
                        metrics[g][r][metric] = {
                            'mean': float(np.mean(values)),
                            'std':  float(np.std(values)),
                            'values': values,
                        }
                    except Exception as e:
                        print(e)

            # Preserve best_head_type for head-sweep runs (0.6.3+)
            best_head = runs.loc[r].get('params.head_type')
            if best_head is not None:
                metrics[g][r]['params.head_type'] = best_head

            for p in param_cols:
                metrics[g][r][p] = runs.loc[r, p]

            mcc_entry = metrics[g][r].get('valid/mcc')
            if mcc_entry not in (None, -np.inf) and isinstance(mcc_entry, dict):
                if mcc_entry['mean'] > best_mccs[g]:
                    best_metrics[g] = dict(metrics[g][r])
                    best_metrics[g]['run_id'] = r
                    best_mccs[g] = mcc_entry['mean']

            print(c)

    # -----------------------------------------------------------------------
    # JSON dumps
    # -----------------------------------------------------------------------
    with open(results_dir / f"metrics_mlflow_{exp_name}_{exp_id}.json", "w") as fh:
        json.dump(metrics, fh)

    with open(results_dir / f"best_metrics_mlflow_{exp_name}_{exp_id}.json", "w") as fh:
        json.dump(best_metrics, fh)

    some_metric_key = next((k for k in best_metrics if best_metrics[k] is not None), None)

    # -----------------------------------------------------------------------
    # best_metrics CSV — mean ± std
    # -----------------------------------------------------------------------
    count = 0
    with open(results_dir / f'best_metrics_mlflow_{exp_name}_{exp_id}.csv', 'w', newline='') as data_file:
        csv_writer = csv.writer(data_file)
        for model in best_metrics:
            if best_metrics[model] is None:
                continue
            if count == 0 and some_metric_key is not None:
                ref = best_metrics[some_metric_key]
                try:
                    header = (
                        ["model", "run_id"]
                        + [x for x in ref if x.startswith("params") and x != "run_id"]
                        + list(itertools.chain(*[
                            [f"{x}_mean", f"{x}_std"]
                            for x in ref if not x.startswith("params") and x != "run_id"
                        ]))
                    )
                    csv_writer.writerow(header)
                    count += 1
                except Exception as e:
                    print(e)
                    continue
            m = (
                [model, best_metrics[model].get('run_id', '')]
                + [best_metrics[model][k] for k in best_metrics[model]
                   if k.startswith("params") and k != "run_id"]
            )
            for k in best_metrics[model]:
                if k.startswith("params") or k == "run_id":
                    continue
                v = best_metrics[model][k]
                if isinstance(v, dict):
                    m += [v.get('mean', np.nan), v.get('std', np.nan)]
                else:
                    m += [np.nan, np.nan]
            csv_writer.writerow(m)

    # -----------------------------------------------------------------------
    # best_metrics_values CSV — raw value lists
    # -----------------------------------------------------------------------
    count = 0
    with open(results_dir / f'best_metrics_mlflow_{exp_name}_{exp_id}_values.csv', 'w', newline='') as data_file:
        csv_writer = csv.writer(data_file)
        for model in best_metrics:
            if best_metrics[model] is None:
                continue
            if count == 0 and some_metric_key is not None:
                ref = best_metrics[some_metric_key]
                try:
                    header = (
                        ["model", "run_id"]
                        + [x for x in ref if x.startswith("params") and x != "run_id"]
                        + [x for x in ref if not x.startswith("params") and x != "run_id"]
                    )
                    csv_writer.writerow(header)
                    count += 1
                except Exception as e:
                    print(e)
                    continue
            m = (
                [model, best_metrics[model].get('run_id', '')]
                + [best_metrics[model][k] for k in best_metrics[model]
                   if k.startswith("params") and k != "run_id"]
            )
            for k in best_metrics[model]:
                if k.startswith("params") or k == "run_id":
                    continue
                v = best_metrics[model][k]
                m.append(v.get('values', np.nan) if isinstance(v, dict) else np.nan)
            csv_writer.writerow(m)

    # -----------------------------------------------------------------------
    # All runs CSV — mean ± std
    # -----------------------------------------------------------------------
    count = 0
    with open(results_dir / f'metrics_mlflow_{exp_name}_{exp_id}.csv', 'w', newline='') as data_file:
        csv_writer = csv.writer(data_file)
        for model in metrics:
            if metrics[model] is None:
                continue
            if count == 0 and some_metric_key is not None:
                ref = best_metrics[some_metric_key]
                try:
                    header = (
                        ["model", "run_id"]
                        + [x for x in ref if x.startswith("params") and x != "run_id"]
                        + list(itertools.chain(*[
                            [f"{x}_mean", f"{x}_std"]
                            for x in ref if not x.startswith("params") and x != "run_id"
                        ]))
                    )
                    csv_writer.writerow(header)
                    count += 1
                except Exception as e:
                    print(e)
                    continue
            for run_id in metrics[model]:
                run = metrics[model][run_id]
                sentinel = run.get('acc/train/all_concentrations', None)
                if sentinel == -np.inf:
                    continue
                m = (
                    [model, run_id]
                    + [run[k] for k in run if k.startswith("params") and k != "run_id"]
                )
                for k in run:
                    if k.startswith("params") or k == "run_id":
                        continue
                    v = run[k]
                    if isinstance(v, dict):
                        m += [v.get('mean', np.nan), v.get('std', np.nan)]
                    else:
                        m += [v, v]
                csv_writer.writerow(m)

    # -----------------------------------------------------------------------
    # All runs values CSV
    # -----------------------------------------------------------------------
    count = 0
    with open(results_dir / f'metrics_mlflow_{exp_name}_{exp_id}_values.csv', 'w', newline='') as data_file:
        csv_writer = csv.writer(data_file)
        for model in metrics:
            if metrics[model] is None:
                continue
            if count == 0 and some_metric_key is not None:
                ref = best_metrics[some_metric_key]
                try:
                    header = (
                        ["model", "run_id"]
                        + [x for x in ref if x.startswith("params") and x != "run_id"]
                        + [x for x in ref if not x.startswith("params") and x != "run_id"]
                    )
                    csv_writer.writerow(header)
                    count += 1
                except Exception as e:
                    print(e)
                    continue
            for run_id in metrics[model]:
                run = metrics[model][run_id]
                sentinel = run.get('acc/train/all_concentrations', None)
                if sentinel == -np.inf:
                    continue
                m = (
                    [model, run_id]
                    + [run[k] for k in run if k.startswith("params") and k != "run_id"]
                )
                for k in run:
                    if k.startswith("params") or k == "run_id":
                        continue
                    v = run[k]
                    m.append(v.get('values', np.nan) if isinstance(v, dict) else v)
                csv_writer.writerow(m)

    print('DONE')
