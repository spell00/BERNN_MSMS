import csv
import json
import mlflow
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import itertools
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='testbenchmark_08_15_2024', help='Name of the experiment to evaluate')
    args = parser.parse_args()

    # get runs
    exp_name = args.exp_name

    # the adenocarcinoma dataset (which I also call amide, but should be changed) has only 3 batches, 
    # so there is only 3 splits possible for training
    if 'amide' in exp_name or 'bactTest' in exp_name or 'Adeno' in exp_name or 'adeno' in exp_name:
        n_per_run = 3
    else:
        n_per_run = 5

    exp_id = mlflow.get_experiment_by_name(exp_name).experiment_id

    print(n_per_run, exp_id)

    runs = mlflow.search_runs(exp_id)
    runs.index = runs['run_id']
    # group runs by parameter
    params = [p for p in runs.columns if p.startswith("params") if p != 'params.parameters' and p != 'params.foldername']
    params_common = [p for p in params if len([x for x in runs[p].unique() if x is not None]) == 1]
    # params_varied = [p for p in params if len([x for x in runs[p].unique() if x is not None]) > 1]
    params_varied = ['params.dloss', 'params.variational', 'params.kan']
    groups = list(runs.groupby(params_varied).run_id)
    # get history for each metric and run and plot mean + std across identical runs
    client = mlflow.tracking.MlflowClient()
    metrics = {'_'.join([''.join([s, x]) for s, x in zip(['', 'vae', 'kan'], g[0])]): {} for g in groups}
    params = [c for c in runs.columns if c.startswith("params")]

    best_metrics = {g: None for g in (metrics.keys())}
    best_mccs = {g: -np.inf for g in (metrics.keys())}

    c = 0
    for gg, run_ids in groups:
        print(gg)
        best_mcc = -np.inf
        for r in run_ids:
            g = '_'.join([''.join([s, x]) for s, x in zip(['', 'vae', 'kan'], gg)])
            metrics[g][r] = {c[8:]: -np.inf for c in runs.columns if c.startswith("metrics")}
            c += 1
            
            # If there is less than the n_per_runs, which was the number of repeated holdout, than we skip this run
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
                            metrics[g][r][metric] = {'mean': mean, 'std': std, 'values': values}
                        except:
                            pass
                except:
                    pass
            for p in params:
                metrics[g][r][p] = runs.loc[r, p]
            if metrics[g][r]['valid/mcc'] is not None:
                if metrics[g][r]['valid/mcc']['mean'] > best_mccs[g]:
                    best_metrics[g] = metrics[g][r]
                    best_metrics[g]['run_id'] = r
                    best_mccs[g] = metrics[g][r]['valid/mcc']['mean']

            print(c)

    with open(f"metrics_mlflow_{exp_name}_{exp_id}.json", "w") as outfile:
        json.dump(metrics, outfile)

    with open(f"best_metrics_mlflow_{exp_name}_{exp_id}.json", "w") as outfile:
        json.dump(best_metrics, outfile)

    # now we will open a file for writing
    data_file = open(f'best_metrics_mlflow_{exp_name}_{exp_id}.csv', 'w')
    
    # create the csv writer object
    csv_writer = csv.writer(data_file)
    
    # Counter variable used for writing
    # headers to the CSV file
    some_metric = list(best_metrics.keys())[0]
    count = 0
    for model in best_metrics:
        if best_metrics[model] is not None:
            if count == 0:
                # Writing headers of CSV file
                try:
                    header = [["model", 'run_id']] + [[x for x in list(best_metrics[some_metric].keys()) if 'param' in x and x != 'run_id']] +  [[f"{x}_mean", f"{x}_std"] for x in list(best_metrics[some_metric].keys()) if 'param' not in x and x != 'run_id']
                except:
                    continue
                # header = [["model"]] + [[f"{x}_mean", f"{x}_std"] for x in list(best_metrics[some_metric].keys()) if 'param' not in x]
                header = list(itertools.chain(*header))
                csv_writer.writerow(header)
                count += 1
            m = [model, best_metrics[model]['run_id']]
            for metric_id in best_metrics[model]:
                if 'params' in metric_id and metric_id != 'run_id':
                    m += [best_metrics[model][metric_id]]
            for metric_id in best_metrics[model]:
                if 'params' not in metric_id and metric_id != 'run_id':
                    try:
                        m += [best_metrics[model][metric_id]['mean']]
                        m += [best_metrics[model][metric_id]['std']]
                    except:
                        m += [np.nan]
                        m += [np.nan]
                # Writing data of CSV file
            # if len(m) > 0:
            csv_writer.writerow(m)

    data_file.close()

    with open(f"best_metrics_mlflow_{exp_name}_{exp_id}_values.json", "w") as outfile:
        json.dump(best_metrics, outfile)

    # now we will open a file for writing
    data_file = open(f'best_metrics_mlflow_{exp_name}_{exp_id}_values.csv', 'w')

    # create the csv writer object
    csv_writer = csv.writer(data_file)

    # Counter variable used for writing
    # headers to the CSV file
    count = 0
    for model in best_metrics:
        if best_metrics[model] is not None:
            if count == 0:
                # Writing headers of CSV file
                try:
                    header = [["model", 'run_id']] + [[x for x in list(best_metrics[some_metric].keys()) if 'param' in x and x != 'run_id']] +  [[x] for x in list(best_metrics[some_metric].keys()) if 'param' not in x and x != 'run_id']
                except:
                    continue
                # header = [["model"]] + [[f"{x}_mean", f"{x}_std"] for x in list(best_metrics[some_metric].keys()) if 'param' not in x]
                header = list(itertools.chain(*header))
                csv_writer.writerow(header)
                count += 1
            m = [model, best_metrics[model]['run_id']]
            # try:
            for metric_id in best_metrics[model]:
                if 'params' in metric_id and metric_id != 'run_id':
                    m += [best_metrics[model][metric_id]]
            for metric_id in best_metrics[model]:
                if 'params' not in metric_id and metric_id != 'run_id':
                    try:
                        m += [best_metrics[model][metric_id]['values']]
                    except:
                        m += [np.nan]
                # Writing data of CSV file
            # except:
            #     m = []
            # if len(m) > 0:
            csv_writer.writerow(m)

    data_file.close()

    # now we will open a file for writing
    data_file = open(f'metrics_mlflow_{exp_name}_{exp_id}.csv', 'w')

    # create the csv writer object
    csv_writer = csv.writer(data_file)

    ex_id = list(metrics[some_metric].keys())[0]

    # Counter variable used for writing
    # headers to the CSV file
    count = 0
    for model in metrics:
        if metrics[model] is not None:
            if count == 0:
                # Writing headers of CSV file
                try:
                    header = [["model", 'run_id']] + [[x for x in list(best_metrics[some_metric].keys()) if 'param' in x and x != 'run_id']] +  [[f"{x}_mean", f"{x}_std"] for x in list(best_metrics[some_metric].keys()) if 'param' not in x and x != 'run_id']
                except:
                    continue
                # header = [["model"]] + [[f"{x}_mean", f"{x}_std"] for x in list(metrics[some_metric].keys()) if 'param' not in x]
                header = list(itertools.chain(*header))
                csv_writer.writerow(header)
                count += 1
            for run_id in metrics[model]:
                m = [model, run_id]
                if metrics[model][run_id]['acc/train/all_concentrations'] != -np.inf:
                    for metric_id in metrics[model][run_id]:
                        if 'params' in metric_id and metric_id != 'run_id':
                            m += [metrics[model][run_id][metric_id]]
                    for metric_id in metrics[model][run_id]:
                        if 'params' not in metric_id and metric_id != 'run_id':
                            try:
                                m += [metrics[model][run_id][metric_id]['mean']]
                                m += [metrics[model][run_id][metric_id]['std']]
                            except:
                                m += [metrics[model][run_id][metric_id]]
                                m += [metrics[model][run_id][metric_id]]
                        # Writing data of CSV file
                # if len(m) > 0:
                csv_writer.writerow(m)

    data_file.close()

    # now we will open a file for writing
    data_file = open(f'metrics_mlflow_{exp_name}_{exp_id}_values.csv', 'w')

    # create the csv writer object
    csv_writer = csv.writer(data_file)

    ex_id = list(metrics[some_metric].keys())[0]

    # Counter variable used for writing
    # headers to the CSV file
    count = 0
    for model in metrics:
        if metrics[model] is not None:
            if count == 0:
                # Writing headers of CSV file
                try:
                    header = [["model", 'run_id']] + [[x for x in list(best_metrics[some_metric].keys()) if 'param' in x and x != 'run_id']] +  [[x] for x in list(best_metrics[some_metric].keys()) if 'param' not in x and x != 'run_id']
                except:
                    continue
                # header = [["model"]] + [[f"{x}_mean", f"{x}_std"] for x in list(metrics[some_metric].keys()) if 'param' not in x]
                header = list(itertools.chain(*header))
                csv_writer.writerow(header)
                count += 1
            for run_id in metrics[model]:
                m = [model, run_id]
                if metrics[model][run_id]['acc/train/all_concentrations'] != -np.inf:
                    for metric_id in metrics[model][run_id]:
                        if 'params' in metric_id and metric_id != 'run_id':
                            m += [metrics[model][run_id][metric_id]]
                    for metric_id in metrics[model][run_id]:
                        if 'params' not in metric_id and metric_id != 'run_id':
                            try:
                                m += [metrics[model][run_id][metric_id]['values']]
                            except:
                                m += [metrics[model][run_id][metric_id]]
                        # Writing data of CSV file
                    # if len(m) > 0:
                    csv_writer.writerow(m)

    data_file.close()


    print('DONE')
