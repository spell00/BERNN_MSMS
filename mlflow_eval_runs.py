import csv
import json
import mlflow
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

n_per_run = 3
# get runs
exp_name = 'amide_ae_classifier_20'
exp_id = mlflow.get_experiment_by_name(exp_name).experiment_id
runs = mlflow.search_runs(exp_id)
runs.index = runs['run_id']
# group runs by parameter
params = [p for p in runs.columns if p.startswith("params") if p != 'params.parameters' and p != 'params.foldername']
params_common = [p for p in params if len([x for x in runs[p].unique() if x is not None]) == 1]
# params_varied = [p for p in params if len([x for x in runs[p].unique() if x is not None]) > 1]
params_varied = ['params.dloss', 'params.variational', 'params.zinb']
groups = list(runs.groupby(params_varied).run_id)
# get history for each metric and run and plot mean + std across identical runs
client = mlflow.tracking.MlflowClient()
metrics = {'_'.join([''.join([s, x]) for s, x in zip(['', 'vae', 'zinb'], g[0])]): {} for g in groups}
params = [c for c in runs.columns if c.startswith("params")]

best_metrics = {g: None for g in (metrics.keys())}
c = 0
for gg, run_ids in groups:
    print(gg)
    best_mcc = -np.inf
    for r in run_ids:
        g = '_'.join([''.join([s, x]) for s, x in zip(['', 'vae', 'zinb'], gg)])
        metrics[g][r] = {c[8:]: -np.inf for c in runs.columns if c.startswith("metrics")}
        c += 1
        if len([s.value for s in client.get_metric_history(r, 'valid/mcc')]) < n_per_run:
            continue

        for metric in metrics[g][r]:
            try:
                values = [s.value for s in client.get_metric_history(r, metric)]
            except mlflow.exceptions.MlflowException:
                continue
            try:
                if len(values) > 0:
                    val = np.atleast_2d(values)
                    try:
                        mean = np.mean(values)
                        std = np.std(values)
                        metrics[g][r][metric] = {'mean': mean, 'std': std}
                    except:
                        pass
            except:
                pass
        for p in params:
            metrics[g][r][p] = runs.loc[r, p]
        if metrics[g][r]['valid/mcc'] is not None:
            if metrics[g][r]['valid/mcc']['mean'] > best_mcc:
                best_metrics[g] = metrics[g][r]
                best_metrics[g]['run_id'] = r
        print(c)

with open(f"metrics_mlflow_{exp_name}.json", "w") as outfile:
    json.dump(metrics, outfile)

with open(f"best_metrics_mlflow_{exp_name}.json", "w") as outfile:
    json.dump(best_metrics, outfile)

# now we will open a file for writing
data_file = open(f'best_metrics_mlflow_{exp_name}.csv', 'w')
 
# create the csv writer object
csv_writer = csv.writer(data_file)
 
# Counter variable used for writing
# headers to the CSV file
count = 0
import itertools
for model in best_metrics:
    if best_metrics[model] is not None:
        if count == 0:
            # Writing headers of CSV file
            header = [["model", 'run_id']] + [[x for x in list(best_metrics['DANN_vae0_zinb0'].keys()) if 'param' in x]] +  [[f"{x}_mean", f"{x}_std"] for x in list(best_metrics['DANN_vae0_zinb0'].keys()) if 'param' not in x]
            # header = [["model"]] + [[f"{x}_mean", f"{x}_std"] for x in list(best_metrics['DANN_vae0_zinb0'].keys()) if 'param' not in x]
            header = list(itertools.chain(*header))
            csv_writer.writerow(header)
            count += 1
        m = [model, best_metrics[model]['run_id']]
        for metrics in best_metrics[model]:
            if 'params' in metrics:
                m += [best_metrics[model][metrics]]
        for metrics in best_metrics[model]:
            if 'params' not in metrics:
                m += [best_metrics[model][metrics]['mean']]
                m += [best_metrics[model][metrics]['std']]
            # Writing data of CSV file
        csv_writer.writerow(m)

data_file.close()

