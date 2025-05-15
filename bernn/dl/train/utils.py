import matplotlib.pyplot as plt
import numpy as np
import itertools

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, Normalizer
from sklearn.pipeline import Pipeline

def keep_top_features(data, path, args):
    """
    Keeps the top features according to the precalculated scores
    Args:
        data: The data to be used to keep the top features

    Returns:
        data: The data with only the top features
    """
    top_features = pd.read_csv(f'{path}/{args.best_features_file}', sep=',')
    for group in ['all', 'train', 'valid', 'test']:
        data['inputs'][group] = data['inputs'][group].loc[:, top_features.iloc[:, 0].values[:args.n_features]]

    return data

def binarize_labels(data, controls):
    """
    Binarizes the labels to be used in the classification loss
    Args:
        labels: The labels to be binarized
        controls: The control labels

    Returns:
        labels: The binarized labels
    """
    for group in ['all', 'train', 'valid', 'test']:
        data['labels'][group] = np.array([1 if x not in controls else 0 for x in data['labels'][group]])
        data['cats'][group] = data['labels'][group]
    return data

def log_num_neurons(run, n_neurons, init_n_neurons):
    """
    Log the number of neurons in the model to Neptune.

    Args:
        run: The Neptune run object.
        n_neurons: Dictionary of current neuron counts per layer (flattened).
        init_n_neurons: Dictionary of initial neuron counts per layer (nested).
    """
    for key, count in n_neurons.items():
        if key in ["total", "total_neurons", "total_remaining"]:
            run["n_neurons/total"].log(count)
            denom = init_n_neurons.get("total") or init_n_neurons.get("total_neurons")
            if denom:
                run["n_neurons/relative_total"].log(count / denom)
            continue

        if '.' not in key:
            continue  # unexpected format, skip

        layer_abbr, sublayer = key.split(".")
        layer_key = {"enc": "encoder2", "dec": "decoder2"}.get(layer_abbr, layer_abbr)

        run[f"n_neurons/{layer_key}/{sublayer}"].log(count)

        try:
            init_count = init_n_neurons[layer_key][sublayer]
            run[f"n_neurons/{layer_key}/relative_{sublayer}"].log(count / init_count)
        except (KeyError, ZeroDivisionError):
            pass


def scale_data(scale, data, device='cpu'):
    unique_batches = np.unique(data['batches']['all'])
    scaler = None
    if scale == 'binarize':
        for group in list(data['inputs'].keys()):
            data['inputs'][group] = data['inputs'][group]
            data['inputs'][group][data['inputs'][group] > 0] = 1
            data['inputs'][group][data['inputs'][group] <= 0] = 0

    elif scale == 'robust_per_batch':
        scalers = {b: RobustScaler() for b in unique_batches}
        for b in unique_batches:
            mask = data['batches']['all'] == b
            scalers[b].fit(data['inputs']['all'][mask])
            for group in list(data['inputs'].keys()):
                if data['inputs'][group].shape[0] > 0:
                    mask = data['batches'][group] == b
                    # columns = data['inputs'][group][mask].columns
                    # indices = data['inputs'][group][mask].index
                    if data['inputs'][group][mask].shape[0] > 0:
                        data['inputs'][group].iloc[mask] = scalers[b].transform(data['inputs'][group][mask].to_numpy())
                    else:
                        data['inputs'][group][mask] = data['inputs'][group][mask]
            # data[data_type][group] = pd.DataFrame(data[data_type][group], columns=columns, index=indices)
        scaler = RobustScaler()
        scaler.fit(data['meta']['all'])
        for group in list(data['meta'].keys()):
            if data['meta'][group].shape[0] > 0:
                columns = data['meta'][group].columns
                indices = data['meta'][group].index
                if data['meta'][group].shape[0] > 0:
                    data['meta'][group] = scaler.transform(data['meta'][group].to_numpy())
                else:
                    data['meta'][group] = data['meta'][group]
                data['meta'][group] = pd.DataFrame(data['meta'][group], columns=columns, index=indices)

    elif scale == 'standard_per_batch':
        scalers = {b: StandardScaler() for b in unique_batches}
        for b in unique_batches:
            mask = data['batches']['all'] == b
            scalers[b].fit(data['inputs']['all'][mask])
            for group in list(data['inputs'].keys()):
                if data['inputs'][group].shape[0] > 0:
                    mask = data['batches'][group] == b
                    # columns = data['inputs'][group][mask].columns
                    # indices = data['inputs'][group][mask].index
                    if data['inputs'][group][mask].shape[0] > 0:
                        data['inputs'][group].iloc[mask] = scalers[b].transform(data['inputs'][group][mask].to_numpy())
                    else:
                        data['inputs'][group][mask] = data['inputs'][group][mask]
            # data[data_type][group] = pd.DataFrame(data[data_type][group], columns=columns, index=indices)
        scaler = StandardScaler()
        scaler.fit(data['meta']['all'])
        for group in list(data['meta'].keys()):
            if data['meta'][group].shape[0] > 0:
                columns = data['meta'][group].columns
                indices = data['meta'][group].index
                if data['meta'][group].shape[0] > 0:
                    data['meta'][group] = scaler.transform(data['meta'][group].to_numpy())
                else:
                    data['meta'][group] = data['meta'][group]
                data['meta'][group] = pd.DataFrame(data['meta'][group], columns=columns, index=indices)

    elif scale == 'minmax_per_batch':
        scalers = {b: MinMaxScaler() for b in unique_batches}
        for b in unique_batches:
            mask = data['batches']['all'] == b
            scalers[b].fit(data['inputs']['all'][mask])
            for group in list(data['inputs'].keys()):
                if data['inputs'][group].shape[0] > 0:
                    mask = data['batches'][group] == b
                    # columns = data['inputs'][group][mask].columns
                    # indices = data['inputs'][group][mask].index
                    if data['inputs'][group][mask].shape[0] > 0:
                        data['inputs'][group].iloc[mask] = scalers[b].transform(data['inputs'][group][mask].to_numpy())
                    else:
                        data['inputs'][group][mask] = data['inputs'][group][mask]
            # data[data_type][group] = pd.DataFrame(data[data_type][group], columns=columns, index=indices)
        scaler = MinMaxScaler()
        scaler.fit(data['meta']['all'])
        for group in list(data['meta'].keys()):
            if data['meta'][group].shape[0] > 0:
                columns = data['meta'][group].columns
                indices = data['meta'][group].index
                if data['meta'][group].shape[0] > 0:
                    data['meta'][group] = scaler.transform(data['meta'][group].to_numpy())
                else:
                    data['meta'][group] = data['meta'][group]
                data['meta'][group] = pd.DataFrame(data['meta'][group], columns=columns, index=indices)

    elif scale == 'robust':
        scaler = RobustScaler()
        for data_type in ['inputs', 'meta']:
            scaler.fit(data[data_type]['all'])
            for group in list(data[data_type].keys()):
                if data[data_type][group].shape[0] > 0:
                    columns = data[data_type][group].columns
                    indices = data[data_type][group].index
                    if data[data_type][group].shape[0] > 0:
                        data[data_type][group] = scaler.transform(data[data_type][group].to_numpy())
                    else:
                        data[data_type][group] = data[data_type][group]
                    data[data_type][group] = pd.DataFrame(data[data_type][group], columns=columns, index=indices)

    elif scale == 'robust_minmax':
        scaler = Pipeline([('robust', RobustScaler()), ('minmax', MinMaxScaler())])
        for data_type in ['inputs', 'meta']:
            scaler.fit(data[data_type]['all'])
            for group in list(data[data_type].keys()):
                if data[data_type][group].shape[0] > 0:
                    columns = data[data_type][group].columns
                    indices = data[data_type][group].index
                    if data[data_type][group].shape[0] > 0:
                        data[data_type][group] = scaler.transform(data[data_type][group].to_numpy())
                    else:
                        data[data_type][group] = data[data_type][group]
                    data[data_type][group] = pd.DataFrame(data[data_type][group], columns=columns, index=indices)

    elif scale == 'standard':
        scaler = StandardScaler()
        for data_type in ['inputs', 'meta']:
            scaler.fit(data[data_type]['all'])
            for group in list(data[data_type].keys()):
                if data[data_type][group].shape[0] > 0:
                    columns = data[data_type][group].columns
                    indices = data[data_type][group].index
                    if data[data_type][group].shape[0] > 0:
                        data[data_type][group] = scaler.transform(data[data_type][group].to_numpy())
                    else:
                        data[data_type][group] = data[data_type][group]
                    data[data_type][group] = pd.DataFrame(data[data_type][group], columns=columns, index=indices)

    elif scale == 'standard_minmax':
        scaler = Pipeline([('standard', StandardScaler()), ('minmax', MinMaxScaler())])
        for data_type in ['inputs', 'meta']:
            scaler.fit(data[data_type]['all'])
            for group in list(data[data_type].keys()):
                if data[data_type][group].shape[0] > 0:
                    columns = data[data_type][group].columns
                    indices = data[data_type][group].index
                    if data[data_type][group].shape[0] > 0:
                        data[data_type][group] = scaler.transform(data[data_type][group].to_numpy())
                    else:
                        data[data_type][group] = data[data_type][group]
                    data[data_type][group] = pd.DataFrame(data[data_type][group], columns=columns, index=indices)

    elif scale == 'minmax':
        scaler = MinMaxScaler()
        for data_type in ['inputs', 'meta']:
            scaler.fit(data[data_type]['all'])
            for group in list(data[data_type].keys()):
                if data[data_type][group].shape[0] > 0:
                    columns = data[data_type][group].columns
                    indices = data[data_type][group].index
                    if data[data_type][group].shape[0] > 0:
                        data[data_type][group] = scaler.transform(data[data_type][group].to_numpy())
                    else:
                        data[data_type][group] = data[data_type][group]
                    data[data_type][group] = pd.DataFrame(data[data_type][group], columns=columns, index=indices)

    elif scale == 'l1_minmax':
        scaler = Pipeline([('l1', Normalizer(norm='l1')), ('minmax', MinMaxScaler())])
        for data_type in ['inputs', 'meta']:
            scaler.fit(data[data_type]['all'])
            for group in list(data[data_type].keys()):
                if data[data_type][group].shape[0] > 0:
                    columns = data[data_type][group].columns
                    indices = data[data_type][group].index
                    if data[data_type][group].shape[0] > 0:
                        data[data_type][group] = scaler.transform(data[data_type][group].to_numpy())
                    else:
                        data[data_type][group] = data[data_type][group]
                    data[data_type][group] = pd.DataFrame(data[data_type][group], columns=columns, index=indices)

    elif scale == 'l2_minmax':
        scaler = Pipeline([('l2', Normalizer(norm='l2')), ('minmax', MinMaxScaler())])
        for data_type in ['inputs', 'meta']:
            scaler.fit(data[data_type]['all'])
            for group in list(data[data_type].keys()):
                if data[data_type][group].shape[0] > 0:
                    columns = data[data_type][group].columns
                    indices = data[data_type][group].index
                    if data[data_type][group].shape[0] > 0:
                        data[data_type][group] = scaler.transform(data[data_type][group].to_numpy())
                    else:
                        data[data_type][group] = data[data_type][group]
                    data[data_type][group] = pd.DataFrame(data[data_type][group], columns=columns, index=indices)

    elif scale == 'l1':
        scaler = Pipeline([('l1', Normalizer(norm='l1'))])
        for data_type in ['inputs', 'meta']:
            scaler.fit(data[data_type]['all'])
            for group in list(data[data_type].keys()):
                if data[data_type][group].shape[0] > 0:
                    columns = data[data_type][group].columns
                    indices = data[data_type][group].index
                    if data[data_type][group].shape[0] > 0:
                        data[data_type][group] = scaler.transform(data[data_type][group].to_numpy())
                    else:
                        data[data_type][group] = data[data_type][group]
                    data[data_type][group] = pd.DataFrame(data[data_type][group], columns=columns, index=indices)

    elif scale == 'l2':
        scaler = Pipeline([('l2', Normalizer(norm='l2'))])
        for data_type in ['inputs', 'meta']:
            scaler.fit(data[data_type]['all'])
            for group in list(data[data_type].keys()):
                if data[data_type][group].shape[0] > 0:
                    columns = data[data_type][group].columns
                    indices = data[data_type][group].index
                    if data[data_type][group].shape[0] > 0:
                        data[data_type][group] = scaler.transform(data[data_type][group].to_numpy())
                    else:
                        data[data_type][group] = data[data_type][group]
                    data[data_type][group] = pd.DataFrame(data[data_type][group], columns=columns, index=indices)

    elif scale == 'none':
        return data, 'none'
    # Values put on [0, 1] interval to facilitate autoencoder reconstruction (enable use of sigmoid as final activation)
    # scaler = MinMaxScaler()
    # scaler.fit(data['inputs']['all'])
    # for group in list(data['inputs'].keys()):
    #     if data['inputs'][group].shape[0] > 0:
    #         columns = data['inputs'][group].columns
    #         indices = data['inputs'][group].index
    #         if data['inputs'][group].shape[0] > 0:
    #             data['inputs'][group] = scaler.transform(data['inputs'][group].to_numpy())
    #         else:
    #             data['inputs'][group] = data['inputs'][group]
    #         data['inputs'][group] = pd.DataFrame(data['inputs'][group], columns=columns, index=indices)

    return data, scaler


def scale_data_images(scale, data, device='cpu'):
    unique_batches = np.unique(data['batches']['all'])
    scaler = None
    if scale == 'binarize':
        for group in list(data['inputs'].keys()):
            data['inputs'][group] = data['inputs'][group]
            data['inputs'][group][data['inputs'][group] > 0] = 1
            data['inputs'][group][data['inputs'][group] <= 0] = 0

    elif scale == 'standard':
        # scaler = StandardScaler()
        for data_type in ['inputs', 'meta']:
            data[data_type]['all'] = data[data_type]['all'] - data[data_type]['all'].mean(axis=(1, 2), keepdims=True)
            data[data_type]['all'] = data[data_type]['all'] / data[data_type]['all'].std(axis=(1, 2), keepdims=True)
            for group in list(data[data_type].keys()):
                if data[data_type][group].shape[0] > 0:
                    data[data_type][group] = data[data_type][group] - data[data_type][group].mean(axis=(1, 2),
                                                                                                  keepdims=True)
                    data[data_type][group] = data[data_type][group] / data[data_type][group].std(axis=(1, 2),
                                                                                                 keepdims=True)

    elif scale == 'none':
        return data, 'none'
    # Values put on [0, 1] interval to facilitate autoencoder reconstruction (enable use of sigmoid as final activation)
    # scaler = MinMaxScaler()
    # scaler.fit(data['inputs']['all'])
    # for group in list(data['inputs'].keys()):
    #     if data['inputs'][group].shape[0] > 0:
    #         columns = data['inputs'][group].columns
    #         indices = data['inputs'][group].index
    #         if data['inputs'][group].shape[0] > 0:
    #             data['inputs'][group] = scaler.transform(data['inputs'][group].to_numpy())
    #         else:
    #             data['inputs'][group] = data['inputs'][group]
    #         data['inputs'][group] = pd.DataFrame(data['inputs'][group], columns=columns, index=indices)

    return data, scaler


def scale_data_per_batch(scale, data, device='cpu'):
    unique_batches = np.unique(data['batches']['all'])
    if scale == 'robust':
        scalers = {b: RobustScaler() for b in unique_batches}
        for b in unique_batches:
            mask = data['batches']['all'] == b
            scalers[b].fit(data['inputs']['all'][mask])
            for group in list(data['inputs'].keys()):
                if data['inputs'][group].shape[0] > 0:
                    mask = data['batches'][group] == b
                    # columns = data['inputs'][group][mask].columns
                    # indices = data['inputs'][group][mask].index
                    if data['inputs'][group][mask].shape[0] > 0:
                        data['inputs'][group].iloc[mask] = scalers[b].transform(data['inputs'][group][mask].to_numpy())
                    else:
                        data['inputs'][group][mask] = data['inputs'][group][mask]
            # data[data_type][group] = pd.DataFrame(data[data_type][group], columns=columns, index=indices)
        scaler = RobustScaler()
        scaler.fit(data['meta']['all'])
        for group in list(data['meta'].keys()):
            if data['meta'][group].shape[0] > 0:
                columns = data['meta'][group].columns
                indices = data['meta'][group].index
                if data['meta'][group].shape[0] > 0:
                    data['meta'][group] = scaler.transform(data['meta'][group].to_numpy())
                else:
                    data['meta'][group] = data['meta'][group]
                data['meta'][group] = pd.DataFrame(data['meta'][group], columns=columns, index=indices)

    elif scale == 'standard':
        scalers = {b: StandardScaler() for b in unique_batches}
        for b in unique_batches:
            mask = data['batches']['all'] == b
            scalers[b].fit(data['inputs']['all'][mask])
            for group in list(data['inputs'].keys()):
                if data['inputs'][group].shape[0] > 0:
                    mask = data['batches'][group] == b
                    # columns = data['inputs'][group][mask].columns
                    # indices = data['inputs'][group][mask].index
                    if data['inputs'][group][mask].shape[0] > 0:
                        data['inputs'][group].iloc[mask] = scalers[b].transform(data['inputs'][group][mask].to_numpy())
                    else:
                        data['inputs'][group][mask] = data['inputs'][group][mask]
            # data[data_type][group] = pd.DataFrame(data[data_type][group], columns=columns, index=indices)
        scaler = StandardScaler()
        scaler.fit(data['meta']['all'])
        for group in list(data['meta'].keys()):
            if data['meta'][group].shape[0] > 0:
                columns = data['meta'][group].columns
                indices = data['meta'][group].index
                if data['meta'][group].shape[0] > 0:
                    data['meta'][group] = scaler.transform(data['meta'][group].to_numpy())
                else:
                    data['meta'][group] = data['meta'][group]
                data['meta'][group] = pd.DataFrame(data['meta'][group], columns=columns, index=indices)

    elif scale == 'minmax':
        scalers = {b: MinMaxScaler() for b in unique_batches}
        for b in unique_batches:
            mask = data['batches']['all'] == b
            scalers[b].fit(data['inputs']['all'][mask])
            for group in list(data['inputs'].keys()):
                if data['inputs'][group].shape[0] > 0:
                    mask = data['batches'][group] == b
                    # columns = data['inputs'][group][mask].columns
                    # indices = data['inputs'][group][mask].index
                    if data['inputs'][group][mask].shape[0] > 0:
                        data['inputs'][group].iloc[mask] = scalers[b].transform(data['inputs'][group][mask].to_numpy())
                    else:
                        data['inputs'][group][mask] = data['inputs'][group][mask]
            # data[data_type][group] = pd.DataFrame(data[data_type][group], columns=columns, index=indices)
        scaler = MinMaxScaler()
        scaler.fit(data['meta']['all'])
        for group in list(data['meta'].keys()):
            if data['meta'][group].shape[0] > 0:
                columns = data['meta'][group].columns
                indices = data['meta'][group].index
                if data['meta'][group].shape[0] > 0:
                    data['meta'][group] = scaler.transform(data['meta'][group].to_numpy())
                else:
                    data['meta'][group] = data['meta'][group]
                data['meta'][group] = pd.DataFrame(data['meta'][group], columns=columns, index=indices)

    elif scale == 'none':
        return data
    # Values put on [0, 1] interval to facilitate autoencoder reconstruction (enable use of sigmoid as final activation)
    # scaler = MinMaxScaler()
    # scaler.fit(data['inputs']['all'])
    # for group in list(data['inputs'].keys()):
    #     if data['inputs'][group].shape[0] > 0:
    #         columns = data['inputs'][group].columns
    #         indices = data['inputs'][group].index
    #         if data['inputs'][group].shape[0] > 0:
    #             data['inputs'][group] = scaler.transform(data['inputs'][group].to_numpy())
    #         else:
    #             data['inputs'][group] = data['inputs'][group]
    #         data['inputs'][group] = pd.DataFrame(data['inputs'][group], columns=columns, index=indices)

    return data, scaler


def plot_confusion_matrix(cm, class_names, acc):
    """
  Returns a matplotlib figure containing the plotted confusion matrix.

  Args:
    cm (array, shape = [n, n]): a confusion matrix of integer classes
    class_names (array, shape = [n]): String names of the integer classes
  """
    figure = plt.figure(figsize=(8, 8))
    cm_normal = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)
    cm_normal[np.isnan(cm_normal)] = 0
    plt.imshow(cm_normal, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f"Confusion matrix (acc: {acc})")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    threshold = 0.5

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm_normal[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    return figure


def get_unique_labels(labels):
    """
    Get unique labels for a set of labels
    :param labels:
    :return:
    """
    unique_labels = []
    for label in labels:
        if label not in unique_labels:
            unique_labels += [label]
    return np.array(unique_labels)


def get_combinations(cm, acc_cutoff=0.6, prop_cutoff=0.8):
    # cm2 = np.zeros(shape=(2, 2))
    to_combine = []
    for i in range(len(list(cm.index)) - 1):
        for j in range(i + 1, len(list(cm.columns))):
            acc = (cm.iloc[i, i] + cm.iloc[j, j]) / (cm.iloc[i, j] + cm.iloc[j, i] + cm.iloc[i, i] + cm.iloc[j, j])
            prop = (cm.iloc[i, j] + cm.iloc[j, i] + cm.iloc[i, i] + cm.iloc[j, j]) / (
                    np.sum(cm.iloc[i, :]) + np.sum(cm.iloc[j, :]))
            if acc < acc_cutoff and prop > prop_cutoff:
                to_combine += [(i, j)]

    # Combine all tuple that have a class in common

    new = True
    while new:
        new = False
        for i in range(len(to_combine) - 1):
            for j in range(i + 1, len(to_combine)):
                if np.sum([1 if x in to_combine[j] else 0 for x in to_combine[i]]) > 0:
                    new_combination = tuple(set(to_combine[i] + to_combine[j]))
                    to_combine = list(
                        set([ele for x, ele in enumerate(to_combine) if x not in [i, j]] + [new_combination]))
                    new = True
                    break

    return to_combine


def to_csv(lists, complete_log_path, columns):
    encoded_data = {}
    encoded_batches = {}
    encoded_cats = {}
    encoded_names = {}
    for group in list(lists.keys()):
        if len(lists[group]['encoded_values']) == 0 and group == 'all':
            continue
        if len(lists[group]['encoded_values']) > 0:
            encoded_data[group] = pd.DataFrame(np.concatenate(lists[group]['encoded_values']),
                                               # index=np.concatenate(lists[group]['labels']),
                                               ).round(4)
            encoded_data[group] = pd.concat((pd.DataFrame(np.concatenate(lists[group]['labels']), columns=['labels']), encoded_data[group]), 1)
            encoded_data[group] = pd.concat((pd.DataFrame(np.concatenate(lists[group]['domains']), columns=['batches']), encoded_data[group]), 1)
            encoded_data[group].index = np.concatenate(lists[group]['names'])
            encoded_batches[group] = np.concatenate(lists[group]['domains'])
            encoded_cats[group] = np.concatenate(lists[group]['classes'])
            encoded_names[group] = np.concatenate(lists[group]['names'])
        else:
            encoded_data[group] = pd.DataFrame(
                np.empty(shape=(0, encoded_data['train'].shape[1]), dtype='float')).round(4)
            encoded_batches[group] = np.array([])
            encoded_cats[group] = np.array([])
            encoded_names[group] = np.array([])

    rec_data = {}
    rec_batches = {}
    rec_cats = {}
    rec_names = {}
    for group in list(lists.keys()):
        if len(lists[group]['rec_values']) == 0 and group == 'all':
            continue
        if len(lists[group]['rec_values']) > 0:
            rec_data[group] = pd.DataFrame(np.concatenate(lists[group]['rec_values']),
                                           # index=np.concatenate(lists[group]['names']),
                                           columns=list(columns)  # + ['gender', 'age']
                                           ).round(4)

            rec_data[group] = pd.concat((pd.DataFrame(np.concatenate(lists[group]['labels']), columns=['labels']), rec_data[group]), 1)
            rec_data[group] = pd.concat((pd.DataFrame(np.concatenate(lists[group]['domains']), columns=['batches']), rec_data[group]), 1)
            rec_data[group].index = np.concatenate(lists[group]['names'])
            rec_batches[group] = np.concatenate(lists[group]['domains'])
            rec_cats[group] = np.concatenate(lists[group]['classes'])
            rec_names[group] = np.concatenate(lists[group]['names'])
        else:
            rec_data[group] = pd.DataFrame(np.empty(shape=(0, rec_data['train'].shape[1]), dtype='float')).round(4)
            rec_batches[group] = np.array([])
            rec_cats[group] = np.array([])
            rec_names[group] = np.array([])

    rec_data = {
        "inputs": rec_data,
        "cats": rec_cats,
        "batches": rec_batches,
    }
    enc_data = {
        "inputs": encoded_data,
        "cats": encoded_cats,
        "batches": encoded_batches,
    }
    try:
        rec_data['inputs']['all'].to_csv(f'{complete_log_path}/recs.csv')
        enc_data['inputs']['all'].to_csv(f'{complete_log_path}/encs.csv')
    except:
        pd.concat((rec_data['inputs']['train'], rec_data['inputs']['valid'], rec_data['inputs']['test'])).to_csv(
            f'{complete_log_path}/recs.csv')
        pd.concat((enc_data['inputs']['train'], enc_data['inputs']['valid'], enc_data['inputs']['test'])).to_csv(
            f'{complete_log_path}/encs.csv')
    return rec_data, enc_data
