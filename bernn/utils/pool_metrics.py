import numpy as np
import mlflow
from scipy.spatial.distance import cdist, pdist
from scipy.stats import norm
from statistics import NormalDist

norm.cdf(1.96)
from scipy.stats import multivariate_normal
from scipy.stats import pearsonr, f_oneway


# https://stackoverflow.com/questions/32551610/overlapping-probability-of-two-normal-distribution-with-scipy
def solve(m1, m2, std1, std2):
    a = 1 / (2 * std1 ** 2) - 1 / (2 * std2 ** 2)
    b = m2 / (std2 ** 2) - m1 / (std1 ** 2)
    c = m1 ** 2 / (2 * std1 ** 2) - m2 ** 2 / (2 * std2 ** 2) - np.log(std2 / std1)

    r = np.roots([a, b, c])[0]

    # integrate
    area = norm.cdf(r, m2, std2) + (1. - norm.cdf(r, m1, std1))
    print(norm.cdf(r, m2, std2), (1. - norm.cdf(r, m1, std1)))
    return area


def batches_overlaps_mean(data, batches):
    means = []
    covs = []
    for b in batches:
        means += [np.mean(data[b], 0)]
        covs += [np.std(data[b], 0)]

    areas = []
    for i in range(len(batches)):
        for j, b in enumerate(batches[i + 1:]):
            # areas += [solve(means[i], means[j+1], covs[i], covs[j+1])]
            normal = multivariate_normal(np.array(means[i]), np.diag(covs[i]))
            for x in data[b]:
                areas += [normal.pdf(x)]

    return np.mean(areas)


def get_pcc(train_b0, train_b1, fun):
    pccs = []
    pvals = []
    for x1 in train_b0:
        for x2 in train_b1:
            pcc, pval = fun(x1, x2)
            pccs += [pcc]
            pvals += [pval]

    return (np.mean(pccs), np.std(pccs)), (np.mean(pvals), np.std(pvals))


def qc_euclidean(data, fun):
    dists = []
    for i, x1 in enumerate(data):
        for x2 in data[i + 1:]:
            dist = fun(x1.reshape(1, -1), x2.reshape(1, -1))
            dists += [dist]

    return np.median(dists)


def get_euclidean(data, labels, unique_labels, fun, group, metric):
    indx = [i for i, x in enumerate(labels[group]) if 'pool' not in x and x in unique_labels]
    metric[group]['euclidean'] = np.median(fun(data[group][indx]))

    return metric


def get_batches_overlap_means(data, batches, metric):
    for group in list(batches.keys()):
        if group not in ['pool', 'all']:
            try:
                metric[group]['overlap'] = batches_overlaps_mean(data[group], batches[group])
            except:
                metric[group]['overlap'] = 'NA'

    return metric


def get_batches_euclidean(data, batches, fun, group, metric):
    metric[group]['b_euclidean'] = batches_euclidean(data[group], batches[group], fun)
    metric[f'{group}_pool']['b_euclidean'] = batches_euclidean(data[f'{group}_pool'], batches[f'{group}_pool'], fun)

    return metric


def batches_euclidean(data, batches, fun):
    dists = []
    for i, x1 in enumerate(batches):
        for x2 in batches[i + 1:]:
            dist = fun(data[x1], data[x2])
            dists += [np.median(dist)]

    return np.median(dists)


def euclidean(data, fun):
    dists = []
    for i, x1 in enumerate(data):
        for x2 in data[i + 1:]:
            dist = fun(x1.reshape(1, -1), x2.reshape(1, -1))
            dists += [dist]

    return np.median(dists)


def get_PCC(data, batches, group, metric):
    metric[group] = {}

    (qc_pcc_mean_train_total, qc_pcc_std_train_total), (qc_pval_mean_train_total, qc_pval_std_train_total) = get_qc_pcc(
        data[f'{group}_pool'], pearsonr)

    metric[group]['qc_aPCC'] = qc_pcc_mean_train_total

    return metric

def get_qc_pcc(data, fun):
    pccs = []
    pvals = []
    for i, x1 in enumerate(data):
        for x2 in data[i + 1:]:
            pcc, pval = fun(x1, x2)
            pccs += [pcc]
            pvals += [pval]

    return (np.mean(pccs), np.std(pccs)), (np.mean(pvals), np.std(pvals))


def get_qc_euclidean(pool_data, group, metric):
    # qc_dist = qc_euclidean(pool_data, pdist)
    metric[group]['qc_dist'] = np.median(pdist(pool_data))

    return metric


def log_pool_metrics(data, batches, labels, unique_labels, logger, epoch, metrics, form, mlops):
    metric = {}

    for group in list(data.keys()):
        if 'pool' not in group:
            try:
                data[group] = data[group].to_numpy()
                data[f'{group}_pool'] = data[f'{group}_pool'].to_numpy()
            except:
                pass

            metric[group] = {}
            metric[f'{group}_pool'] = {}
            batch_train_samples = [[i for i, batch in enumerate(batches[group].tolist()) if batch == b] for b in
                                   np.unique(batches[group])]
            batch_pool_samples = [[i for i, batch in enumerate(batches[f"{group}_pool"].tolist()) if batch == b] for b in
                                  np.unique(batches[f"{group}_pool"])]

            batches_sample_indices = {
                group: batch_train_samples,
                f'{group}_pool': batch_pool_samples,
            }
            # Average Pearson's Correlation Coefficients
            # try:
            metric = get_PCC(data, batches, group, metric)
            # except:
            #     pass

            # QC euclidean distance
            # try:
            metric = get_qc_euclidean(data[f'{group}_pool'], group, metric)
            # except:
            #    pass

            # Batch avg distance
            # try:
            metric = get_batches_euclidean(data, batches_sample_indices, cdist, group, metric)
            # except:
            #     pass

            # avg distance
            # try:
            metric = get_euclidean(data, labels, unique_labels, pdist, group, metric)
            # except:
            #     pass

    for group in metric:
        if 'pool' not in group:
            metric[group]['qc_dist/tot_eucl'] = metric[group]['qc_dist'] / metric[group]['euclidean']
            metric[group]['b_euclidean/tot_eucl'] = metric[group]['b_euclidean'] / metric[group]['euclidean']
        for m in metric[group]:
            if not np.isnan(metric[group][m]) and m not in ['b_euclidean', 'euclidean']:
                if mlops == 'tensorboard':
                    logger.add_scalar(f'pool_metrics_{form}/{m}/{group}', metric[group][m], epoch)
                elif mlops == 'neptune':
                    logger[f'pool_metrics_{form}/{m}/{group}'].log(metric[group][m])
                elif mlops == 'mlflow':
                    m2 = m.replace('[', ' ')
                    m2 = m2.replace(']', ' ')
                    mlflow.log_metric(f'pool_metrics_{form}/{m2}/{group}', metric[group][m])

    metrics[f'pool_metrics_{form}'] = metric

    return metrics
