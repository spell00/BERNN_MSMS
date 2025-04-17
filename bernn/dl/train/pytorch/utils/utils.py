import os
import math
import torch
import mlflow
import sklearn
import itertools
import numpy as np
import tensorflow as tf
from sklearn import metrics
from itertools import cycle
from matplotlib import pyplot as plt, cm
from sklearn.metrics import roc_auc_score, PrecisionRecallDisplay
from bernn.ml.train.params_gp import *
from bernn.ml.train.sklearn_train_nocv import Train
from skopt import gp_minimize
from sklearn.preprocessing import label_binarize, OneHotEncoder


def get_optimizer(model, learning_rate, weight_decay, optimizer_type, momentum=0.9):
    """
    This function takes a model with learning rate, weight decay and optionally momentum and returns an optimizer object
    Args:
        model: The PyTorch model to optimize
        learning_rate: The optimizer's learning rate
        weight_decay: The optimizer's weight decay
        optimizer_type: The optimizer's type [adam or sgd]
        momentum:

    Returns:
        an optimizer object
    """
    # TODO Add more optimizers
    if optimizer_type == 'adam':
        optimizer = torch.optim.Adam(params=model.parameters(),
                                     lr=learning_rate,
                                     weight_decay=weight_decay
                                     # betas=(0.5, 0.9)
                                     )
    elif optimizer_type == 'radam':
        optimizer = torch.optim.RAdam(params=model.parameters(),
                                     lr=learning_rate,
                                     weight_decay=weight_decay,
                                     )
    elif optimizer_type == 'adamw':
        optimizer = torch.optim.RAdam(params=model.parameters(),
                                      lr=learning_rate,
                                      weight_decay=weight_decay,
                                      )
    elif optimizer_type == 'rmsprop':
        optimizer = torch.optim.RMSprop(params=model.parameters(),
                                        lr=learning_rate,
                                        weight_decay=weight_decay,
                                        momentum=0.9
                                        )
    else:
        optimizer = torch.optim.SGD(params=model.parameters(),
                                    lr=learning_rate,
                                    weight_decay=weight_decay,
                                    momentum=momentum
                                    )
    return optimizer


def to_categorical(y, num_classes):
    """
    1-hot encodes a tensor
    Args:
        y: values to encode
        num_classes: Number of classes. Length of the 1-encoder

    Returns:
        Tensor corresponding to the one-hot encoded classes
    """
    return torch.eye(num_classes, dtype=torch.int)[y]


def plot_confusion_matrix(cm, class_names, acc):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
        cm (array, shape = [n, n]): a confusion matrix of integer classes
        class_names (array, shape = [n]): String names of the integer classes

    Returns:
        The figure of the confusion matrix
    """
    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f"Confusion matrix (Acc: {np.round(np.mean(acc), 2)})")
    plt.colorbar()
    plt.grid(b=None)
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Compute the labels from the normalized confusion matrix.
    # labels = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    # plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    return figure


class LogConfusionMatrix:
    def __init__(self, complete_log_path):
        self.complete_log_path = complete_log_path
        self.preds = {'train': [], 'valid': [], 'test': []}
        self.classes = {'train': [], 'valid': [], 'test': []}
        self.cats = {'train': [], 'valid': [], 'test': []}
        self.encs = {'train': [], 'valid': [], 'test': []}
        self.recs = {'train': [], 'valid': [], 'test': []}
        self.gender = {'train': [], 'valid': [], 'test': []}
        self.age = {'train': [], 'valid': [], 'test': []}
        self.batches = {'train': [], 'valid': [], 'test': []}

    def add(self, lists):
        n_mini_batches = int(1000 / lists['train']['preds'][0].shape[0])
        for group in list(self.preds.keys()):
            # Calculate the confusion matrix.
            if len(lists[group]['preds']) == 0:
                continue
            self.preds[group] += [np.concatenate(lists[group]['preds'][:n_mini_batches]).argmax(1)]
            self.classes[group] += [np.concatenate(lists[group]['classes'][:n_mini_batches])]
            self.encs[group] += [np.concatenate(lists[group]['encoded_values'][:n_mini_batches])]
            try:
                self.recs[group] += [np.concatenate(lists[group]['rec_values'][:n_mini_batches])]
            except:
                pass
            self.cats[group] += [np.concatenate(lists[group]['cats'][:n_mini_batches])]
            self.batches[group] += [np.concatenate(lists[group]['domains'][:n_mini_batches])]

    def plot(self, logger, epoch, unique_labels, mlops):
        for group in ['train', 'valid', 'test']:
            preds = np.concatenate(self.preds[group])
            classes = np.concatenate(self.classes[group])
            cm = sklearn.metrics.confusion_matrix(classes, preds)

            acc = np.mean([0 if pred != c else 1 for pred, c in zip(preds, classes)])

            try:
                figure = plot_confusion_matrix(cm, class_names=unique_labels[:len(np.unique(self.classes['train']))], acc=acc)
            except:
                figure = plot_confusion_matrix(cm, class_names=unique_labels, acc=acc)
            if mlops == "tensorboard":
                logger.add_figure(f"CM_{group}_all", figure, epoch)
            elif mlops == "neptune":
                logger[f"CM_{group}_all"].upload(figure)
            elif mlops == "mlflow":
                mlflow.log_figure(figure, f"CM_{group}_all.png")
                # logger[f"CM_{group}_all"].log(figure)
            plt.close(figure)
        del cm, figure

    def get_rf_results(self, run, args):
        hparams_names = [x.name for x in rfc_space]
        enc_data = {name: {x: None for x in ['train', 'valid', 'test']} for name in ['cats', 'inputs', 'batches']}
        rec_data = {name: {x: None for x in ['train', 'valid', 'test']} for name in ['cats', 'inputs', 'batches']}
        for group in list(self.preds.keys()):
            enc_data['inputs'][group] = self.encs[group]
            enc_data['cats'][group] = self.cats[group]
            enc_data['batches'][group] = self.batches[group]
            rec_data['inputs'][group] = self.encs[group]
            rec_data['cats'][group] = self.cats[group]
            rec_data['batches'][group] = self.batches[group]

        train = Train("Reconstruction", RandomForestClassifier, rec_data, hparams_names,
                      self.complete_log_path,
                      args, run, ovr=0, binary=False, mlops='neptune')
        _ = gp_minimize(train.train, rfc_space, n_calls=20, random_state=1)
        train = Train("Encoded", RandomForestClassifier, enc_data, hparams_names, self.complete_log_path,
                      args, run, ovr=0, binary=False, mlops='neptune')
        _ = gp_minimize(train.train, rfc_space, n_calls=20, random_state=1)


def log_confusion_matrix(logger, epoch, lists, unique_labels, traces, mlops):
    """
    Logs the confusion matrix with tensorboardX (tensorboard for pytorch)
    Args:
        logger: TensorboardX logger
        epoch: The epoch scores getting logged
        lists: Dict containing lists of information on the run getting logged
        unique_labels: list of strings containing the unique labels (classes)
        traces: Dict of Dict of lists of scores

    Returns:
        Nothing. The logger doesn't need to be returned, it saves things on disk
    """
    for values in list(lists.keys())[1:]:
        # Calculate the confusion matrix.
        if len(lists[values]['preds']) == 0:
            continue
        preds = np.concatenate(lists[values]['preds']).argmax(1)
        classes = np.concatenate(lists[values]['classes'])
        cm = sklearn.metrics.confusion_matrix(classes, preds)
        figure = plot_confusion_matrix(cm, class_names=unique_labels[:len(np.unique(classes))],
                                       acc=traces[values]['acc'])
        if mlops == "tensorboard":
            logger.add_figure(f"CM_{values}_all", figure, epoch)
        elif mlops == "neptune":
            logger[f"CM_{values}_all"].upload(figure)
        elif mlops == "mlflow":
            mlflow.log_figure(figure, f"CM_{values}_all")
        plt.close(figure)
        del cm, figure


def save_roc_curve(model, x_test, y_test, unique_labels, name, binary, acc, mlops, epoch=None, logger=None):
    """

    Args:
        model:
        x_test:
        y_test:
        unique_labels:
        name:
        binary: Is it a binary classification?
        acc:
        epoch:
        logger:

    Returns:

    """
    dirs = '/'.join(name.split('/')[:-1])
    name = name.split('/')[-1]
    os.makedirs(f'{dirs}', exist_ok=True)
    if binary:
        y_pred_proba = model.predict_proba(x_test)[::, 1]
        fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
        roc_auc = metrics.auc(fpr, tpr)
        roc_score = roc_auc_score(y_true=y_test, y_score=y_pred_proba)

        # create ROC curve
        plt.figure()
        plt.plot(fpr, tpr, label='ROC curve (AUC = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (1 - Specificity)')
        plt.ylabel('True Positive Rate (Sensitivity)')
        plt.title(f'ROC curve (acc={np.round(acc, 3)})')
        plt.legend(loc="lower right")
        stuck = True
        while stuck:
            try:
                plt.savefig(f'{dirs}/ROC.png')
                stuck = False
            except:
                print('stuck...')
        plt.close()
    else:
        # Compute ROC curve and ROC area for each class
        from sklearn.preprocessing import label_binarize
        try:
            y_pred_proba = model.predict_proba(x_test)
        except:
            y_pred_proba = model.classifier(x_test)

        y_preds = model.predict_proba(x_test)
        n_classes = len(unique_labels) - 1  # -1 to remove QC
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        classes = np.arange(len(unique_labels))

        roc_score = roc_auc_score(y_true=OneHotEncoder().fit_transform(y_test.reshape(-1, 1)).toarray(),
                                  y_score=y_preds,
                                  multi_class='ovr')
        for i in range(n_classes):
            fpr[i], tpr[i], _ = metrics.roc_curve(label_binarize(y_test, classes=classes)[:, i], y_pred_proba[:, i])
            roc_auc[i] = metrics.auc(fpr[i], tpr[i])
        # roc for each class
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        plt.title(f'ROC curve (AUC={np.round(roc_score, 3)}, acc={np.round(acc, 3)})')
        # ax.plot(fpr[0], tpr[0], label=f'AUC = {np.round(roc_score, 3)} (All)', color='k')
        for i in range(n_classes):
            ax.plot(fpr[i], tpr[i], label=f'AUC = {np.round(roc_auc[i], 3)} ({unique_labels[i]})')
        ax.legend(loc="best")
        ax.grid(alpha=.4)
        # sns.despine()
        stuck = True
        while stuck:
            try:
                plt.savefig(f'{dirs}/ROC.png')
                stuck = False
            except:
                print('stuck...')

        if logger is not None:
            if mlops == 'tensorboard':
                logger.add_figure(name, fig, epoch)
            if mlops == 'neptune':
                logger[name].log(fig)
            if mlops == 'mlflow':
                mlflow.log_figure(fig, name)
                
        plt.close(fig)

    return roc_score


def save_precision_recall_curve(model, x_test, y_test, unique_labels, name, binary, acc, mlops, epoch=None,
                                logger=None):
    dirs = '/'.join(name.split('/')[:-1])
    name = name.split('/')[-1]
    os.makedirs(f'{dirs}', exist_ok=True)
    if binary:
        y_pred_proba = model.predict_proba(x_test)[::, 1]
        fpr, tpr, _ = metrics.precision_recall_curve(y_test, y_pred_proba)
        aps = metrics.average_precision_score(y_test, y_pred_proba)
        # aps = metrics.roc_auc_score(y_true=y_test, y_score=y_pred_proba)

        # create ROC curve
        plt.figure()
        plt.plot(fpr, tpr, label='Precision-Recall curve (average precision score = %0.2f)' % aps)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall (True Positive Rate) (TP/P)')
        plt.ylabel('Precision (TP/PP)')
        plt.title(f'Precision-Recall curve (acc={np.round(acc, 3)})')
        plt.legend(loc="lower right")
        stuck = True
        while stuck:
            try:
                plt.savefig(f'{dirs}/Precision-Recall.png')
                stuck = False
            except:
                print('stuck...')
        plt.close()

    else:
        # Compute Precision-Recall curve and Precision-Recall area for each class
        y_pred_proba = model.predict_proba(x_test)
        y_preds = model.predict(x_test)
        n_classes = len(unique_labels) - 1
        precisions = dict()
        recalls = dict()
        average_precision = dict()
        classes = np.arange(n_classes)
        y_test_bin = OneHotEncoder().fit_transform(y_test.reshape(-1, 1)).toarray()
        for i in range(n_classes):
            y_true = y_test_bin[:, i]
            precisions[i], recalls[i], _ = metrics.precision_recall_curve(y_true, y_pred_proba[:, i], pos_label=1)
            average_precision[i] = metrics.average_precision_score(y_true=y_true, y_score=y_pred_proba[:, i])
        
        # roc for each class
        fig, ax = plt.subplots()
        # A "micro-average": quantifying score on all classes jointly
        precisions["micro"], recalls["micro"], _ = metrics.precision_recall_curve(
            y_test_bin.ravel(), y_pred_proba.ravel()
        )
        average_precision["micro"] = metrics.average_precision_score(y_test_bin, y_pred_proba,
                                                                     average="micro")  # sns.despine()
        display = PrecisionRecallDisplay(
            recall=recalls["micro"],
            precision=precisions["micro"],
            average_precision=average_precision["micro"],
        )
        stuck = True
        while stuck:
            try:
                plt.savefig(f'{dirs}/PRC.png')
                stuck = False
            except:
                print('stuck...')
        display.plot()
        fig = display.figure_
        if logger is not None:
            if mlops == 'tensorboard':
                logger.add_figure(name, fig, epoch)
            if mlops == 'neptune':
                logger[name].log(fig)
            if mlops == 'mlops':
                mlflow.log_figure(fig, name)
        # setup plot details
        colors = cycle(["navy", "turquoise", "darkorange", "cornflowerblue", "teal"])
        viridis = cm.get_cmap('viridis', 256)
        _, ax = plt.subplots(figsize=(7, 8))

        f_scores = np.linspace(0.2, 0.8, num=4)
        lines, labels = [], []
        for f_score in f_scores:
            x = np.linspace(0.01, 1)
            y = f_score * x / (2 * x - f_score)
            (l,) = plt.plot(x[y >= 0], y[y >= 0], color="gray", alpha=0.2)
            plt.annotate("f1={0:0.1f}".format(f_score), xy=(0.9, y[45] + 0.02))

        display = PrecisionRecallDisplay(
            recall=recalls["micro"],
            precision=precisions["micro"],
            average_precision=average_precision["micro"],
        )
        display.plot(ax=ax, name="Micro-average precision-recall", color="gold")

        for i, color in zip(range(n_classes), viridis.colors):
            display = PrecisionRecallDisplay(
                recall=recalls[i],
                precision=precisions[i],
                average_precision=average_precision[i],
            )
            display.plot(ax=ax, name=f"Precision-recall for class {i}", color=color)

        # add the legend for the iso-f1 curves
        handles, labels = display.ax_.get_legend_handles_labels()
        handles.extend([l])
        labels.extend(["iso-f1 curves"])
        # set the legend and the axes
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.legend(handles=handles, labels=labels, loc="best")
        ax.set_title("Extension of Precision-Recall curve to multi-class")

        fig = display.figure_
        plt.savefig(f'{dirs}/{name}_multiclass.png')
        if logger is not None:
            if mlops == 'tensorboard':
                logger.add_figure(f'{name}_multiclass', fig, epoch)
            if mlops == 'neptune':
                logger[f'{name}_multiclass'].log(fig)
            if mlops == 'mlflow':
                mlflow.log_figure(fig, f'{name}_multiclass')

        plt.close(fig)

    # return pr_auc


def get_empty_dicts():
    values = {
        "rec_loss": [],
        "dom_loss": [],
        "dom_acc": [],
        "set": {
            "lisi": {'inputs': {'domains': [], 'labels': [], 'set': []},
                     'enc': {'domains': [], 'labels': [], 'set': []}, 'rec': {'domains': [], 'labels': [], 'set': []}},
            "silhouette": {'inputs': {'domains': [], 'labels': [], 'set': []},
                           'enc': {'domains': [], 'labels': [], 'set': []},
                           'rec': {'domains': [], 'labels': [], 'set': []}},
            "kbet": {'inputs': {'domains': [], 'labels': [], 'set': []},
                     'enc': {'domains': [], 'labels': [], 'set': []}, 'rec': {'domains': [], 'labels': [], 'set': []}},
            "adjusted_rand_score": {'inputs': {'domains': [], 'labels': [], 'set': []},
                                    'enc': {'domains': [], 'labels': [], 'set': []},
                                    'rec': {'domains': [], 'labels': [], 'set': []}},
            "batch_entropy": {'inputs': {'domains': [], 'labels': [], 'set': []},
                                    'enc': {'domains': [], 'labels': [], 'set': []},
                                    'rec': {'domains': [], 'labels': [], 'set': []}},
            "adjusted_mutual_info_score": {'inputs': {'domains': [], 'labels': [], 'set': []},
                                           'enc': {'domains': [], 'labels': [], 'set': []},
                                           'rec': {'domains': [], 'labels': [], 'set': []}},
            "F1_lisi": {'inputs': {'domains': [], 'labels': [], 'set': []}, 'enc': {'domains': [], 'labels': []},
                        'rec': {'domains': [], 'labels': []}},
            "F1_silhouette": {'inputs': {'domains': [], 'labels': [], 'set': []}, 'enc': {'domains': [], 'labels': []},
                              'rec': {'domains': [], 'labels': []}},
            "F1_kbet": {'inputs': {'domains': [], 'labels': [], 'set': []}, 'enc': {'domains': [], 'labels': []},
                        'rec': {'domains': [], 'labels': []}},
            "F1_adjusted_rand_score": {'inputs': {'domains': [], 'labels': []}, 'enc': {'domains': [], 'labels': []},
                                       'rec': {'domains': [], 'labels': []}},
            "F1_adjusted_mutual_info_score": {'inputs': {'domains': [], 'labels': []},
                                              'enc': {'domains': [], 'labels': []},
                                              'rec': {'domains': [], 'labels': []}},
        },
    }
    best_values = {
        'rec_loss': 100,
        'dom_loss': 100,
        'dom_acc': 0,
    }
    best_lists = {}
    best_traces = {
        'rec_loss': [1000],
        'dom_loss': [1000],
        'dom_acc': [0],
    }

    for g in ['all', 'train', 'valid', 'test', 'all_pool', 'train_pool', 'valid_pool', 'test_pool', 'all_pool']:
        values[g] = {
            "closs": [],
            "bic": [],
            "aic": [],
            "lisi": {'inputs': {'domains': [], 'labels': [], 'set': []}, 'enc': {'domains': [], 'labels': []},
                     'rec': {'domains': [], 'labels': []}},
            "silhouette": {'inputs': {'domains': [], 'labels': [], 'set': []}, 'enc': {'domains': [], 'labels': []},
                           'rec': {'domains': [], 'labels': []}},
            "kbet": {'inputs': {'domains': [], 'labels': [], 'set': []}, 'enc': {'domains': [], 'labels': []},
                     'rec': {'domains': [], 'labels': []}},
            "adjusted_rand_score": {'inputs': {'domains': [], 'labels': []}, 'enc': {'domains': [], 'labels': []},
                                    'rec': {'domains': [], 'labels': []}},
            "adjusted_mutual_info_score": {'inputs': {'domains': [], 'labels': []},
                                           'enc': {'domains': [], 'labels': []}, 'rec': {'domains': [], 'labels': []}},
            "batch_entropy": {'inputs': {'domains': [], 'labels': []},
                                           'enc': {'domains': [], 'labels': []}, 'rec': {'domains': [], 'labels': []}},
            "F1_lisi": {'inputs': {'domains': [], 'labels': [], 'set': []}, 'enc': {'domains': [], 'labels': []},
                        'rec': {'domains': [], 'labels': []}},
            "F1_silhouette": {'inputs': {'domains': [], 'labels': [], 'set': []}, 'enc': {'domains': [], 'labels': []},
                              'rec': {'domains': [], 'labels': []}},
            "F1_kbet": {'inputs': {'domains': [], 'labels': [], 'set': []}, 'enc': {'domains': [], 'labels': []},
                        'rec': {'domains': [], 'labels': []}},
            "F1_adjusted_rand_score": {'inputs': {'domains': [], 'labels': []}, 'enc': {'domains': [], 'labels': []},
                                       'rec': {'domains': [], 'labels': []}},
            "F1_adjusted_mutual_info_score": {'inputs': {'domains': [], 'labels': []},
                                              'enc': {'domains': [], 'labels': []},
                                              'rec': {'domains': [], 'labels': []}},
            "F1_batch_entropy": {'inputs': {'domains': [], 'labels': []},
                                              'enc': {'domains': [], 'labels': []},
                                              'rec': {'domains': [], 'labels': []}},
            "kld": [],
            "acc": [],
            "top3": [],
            "mcc": [],
        }
        best_traces[g] = {
            'closs': [100],
            'acc': [0],
            'top3': [0],
            'mcc': [0]
        }

        for m in ['acc', 'top3', 'mcc']:
            best_values[f"{g}_{m}"] = 0
        best_values[f"{g}_loss"] = 100
        best_lists[g] = {
            'set': [],
            'preds': [],
            'domain_preds': [],
            'probs': [],
            'lows': [],
            'cats': [],
            'times': [],
            'classes': [],
            'labels': [],
            'domains': [],
            'encoded_values': [],
            'enc_values': [],
            'age': [],
            'gender': [],
            'atn': [],
            'names': [],
            'rec_values': [],
            'inputs': [],
        }

    return values, best_values, best_lists, best_traces


def get_empty_traces():
    traces = {
        'rec_loss': [],
        'dom_loss': [],
        'dom_acc': [],
    }
    lists = {}
    for g in ['all', 'train', 'valid', 'test', 'train_pool', 'valid_pool', 'test_pool', 'all_pool']:
        lists[g] = {
            'set': [],
            'preds': [],
            'domain_preds': [],
            'probs': [],
            'lows': [],
            'cats': [],
            'times': [],
            'classes': [],
            'domains': [],
            'labels': [],
            'encoded_values': [],
            'enc_values': [],
            'age': [],
            'gender': [],
            'atn': [],
            'names': [],
            'inputs': [],
            'rec_values': [],
        }
        traces[g] = {
            'closs': [],
            'acc': [],
            'top3': [],
            'mcc': [],
        }

    return lists, traces


def log_traces(traces, values):
    if len(traces['dom_loss']) > 0:
        values['dom_loss'] += [np.mean(traces['dom_loss'])]
    if len(traces['dom_acc']) > 0:
        values['dom_acc'] += [np.mean(traces['dom_acc'])]
    if len(traces['rec_loss']) > 0:
        values['rec_loss'] += [np.mean(traces['rec_loss'])]

    if len(traces['train']['closs']) > 0:
        for g in list(traces.keys())[3:]:
            if g not in ['all', 'all_pool']:
                values[g]['closs'] += [np.mean(traces[g]['closs'])]
                values[g]['acc'] += [np.mean(traces[g]['acc'])]
                values[g]['top3'] += [np.mean(traces[g]['top3'])]
                values[g]['mcc'] += [traces[g]['mcc']]

    return values


def get_best_loss_from_tb(event_acc):
    values = {}
    # plugin_data = event_acc.summary_metadata['_hparams_/session_start_info'].plugin_data.content
    # plugin_data = plugin_data_pb2.HParamsPluginData.FromString(plugin_data)
    # for name in event_acc.summary_metadata.keys():
    #     if name not in ['_hparams_/experiment', '_hparams_/session_start_info']:
    best_closs = event_acc.Tensors('valid/loss')
    best_closs = tf.make_ndarray(best_closs[0].tensor_proto).item()

    return best_closs


def get_best_acc_from_tb(event_acc):
    values = {}
    # plugin_data = event_acc.summary_metadata['_hparams_/session_start_info'].plugin_data.content
    # plugin_data = plugin_data_pb2.HParamsPluginData.FromString(plugin_data)
    # for name in event_acc.summary_metadata.keys():
    #     if name not in ['_hparams_/experiment', '_hparams_/session_start_info']:
    best_closs = event_acc.Tensors('valid/acc')
    best_closs = tf.make_ndarray(best_closs[0].tensor_proto).item()

    return best_closs


def get_best_values(values, ae_only, n_agg=10):
    """
    As the name implies - this function gets the best values
    Args:
        values: Dict containing the values to be logged
        ae_only: Boolean to return the appropriate output

    Returns:

    """
    best_values = {}
    if ae_only:
        best_values = {
            'rec_loss':  values['rec_loss'][-1],
            'dom_loss': values['dom_loss'][-1],
            'dom_acc': values['rec_loss'][-1],
        }
        for g in ['train', 'valid', 'test']:
            for k in ['acc', 'mcc', 'top3']:
                best_values[f'{g}_{k}'] = math.nan
                best_values[f'{g}_loss'] = math.nan


    else:
        if len(values['dom_loss']) > 0:
            best_values['dom_loss'] = values['dom_loss'][-1]
        if len(values['dom_acc']) > 0:
            best_values['dom_acc'] = values['dom_acc'][-1]
        if len(values['rec_loss']) > 0:
            best_values['rec_loss'] = values['rec_loss'][-1]
        for g in ['train', 'valid', 'test']:
            for k in ['acc', 'mcc', 'top3']:
                if g == 'test':
                    best_values[f'{g}_{k}'] = values[g][k][-1]
                    best_values[f'{g}_loss'] = values[g]['closs'][-1]
                else:
                    best_values[f'{g}_{k}'] = np.mean(values[g][k][-n_agg:])
                    best_values[f'{g}_loss'] = np.mean(values[g]['closs'][-n_agg:])

    return best_values


def add_to_logger(values, logger, epoch):
    """
    Add values to the tensorboarX logger
    Args:
        values: Dict of values to be logged
        logger: Logger for the current experiment
        epoch: Epoch of the values getting logged

    """
    if not np.isnan(values['rec_loss'][-1]):
        logger.add_scalar(f'rec_loss', values['rec_loss'][-1], epoch)
        logger.add_scalar(f'dom_loss', values['dom_loss'][-1], epoch)
        logger.add_scalar(f'dom_acc', values['dom_acc'][-1], epoch)
    for group in list(values.keys())[4:]:
        try:
            if not np.isnan(values[group]['closs'][-1]):
                logger.add_scalar(f'/closs/{group}', values[group]['closs'][-1], epoch)
            if not np.isnan(values[group]['acc'][-1]):
                logger.add_scalar(f'/acc/{group}/all_concentrations', values[group]['acc'][-1], epoch)
            if not np.isnan(values[group]['mcc'][-1]):
                logger.add_scalar(f'/mcc/{group}/all_concentrations', values[group]['mcc'][-1], epoch)
            if not np.isnan(values[group]['top3'][-1]):
                logger.add_scalar(f'/top3/{group}/all_concentrations', values[group]['top3'][-1], epoch)
        except:
            pass


def add_to_neptune(run, values):
    """
    Add values to the neptune run
    Args:
        values: Dict of values to be logged
        run: Logger for the current experiment
        epoch: Epoch of the values getting logged

    """
    if len(values['rec_loss']) > 0:
        if not np.isnan(values['rec_loss'][-1]):
            run["rec_loss"].log(values['rec_loss'][-1])
    if len(values['dom_loss']) > 0:
        if not np.isnan(values['dom_loss'][-1]):
            run["dom_loss"].log(values['dom_loss'][-1])
    if len(values['dom_acc']) > 0:
        if not np.isnan(values['dom_acc'][-1]):
            run["dom_acc"].log(values['dom_acc'][-1])
    for group in list(values.keys())[4:]:
        try:
            if not np.isnan(values[group]['closs'][-1]):
                run[f'/closs/{group}'].log(values[group]['closs'][-1])
            if not np.isnan(values[group]['acc'][-1]):
                run[f'/acc/{group}/all_concentrations'].log(values[group]['acc'][-1])
            if not np.isnan(values[group]['mcc'][-1]):
                run[f'/mcc/{group}/all_concentrations'].log(values[group]['mcc'][-1])
            if not np.isnan(values[group]['top3'][-1]):
                run[f'/top3/{group}/all_concentrations'].log(values[group]['top3'][-1])
        except:
            pass


def add_to_mlflow(values, epoch):
    """
    Add values to the mlflow logger
    Args:
        values: Dict of values to be logged
        logger: Logger for the current experiment
        epoch: Epoch of the values getting logged

    """
    if len(values['rec_loss']) > 0:
        if not np.isnan(values['rec_loss'][-1]):
            mlflow.log_metric("rec_loss", values['rec_loss'][-1], epoch)
    if len(values['dom_loss']) > 0:
        if not np.isnan(values['dom_loss'][-1]):
            mlflow.log_metric("dom_loss", values['dom_loss'][-1], epoch)
    if len(values['dom_acc']) > 0:
        if not np.isnan(values['dom_acc'][-1]):
            mlflow.log_metric("dom_acc", values['dom_acc'][-1], epoch)
    for group in list(values.keys())[4:]:
        try:
            if not np.isnan(values[group]['closs'][-1]):
                mlflow.log_metric(f'closs/{group}', values[group]['closs'][-1], epoch)
            if not np.isnan(values[group]['acc'][-1]):
                mlflow.log_metric(f'acc/{group}/all_concentrations', values[group]['acc'][-1], epoch)
            if not np.isnan(values[group]['mcc'][-1]):
                mlflow.log_metric(f'mcc/{group}/all_concentrations', values[group]['mcc'][-1], epoch)
            if not np.isnan(values[group]['top3'][-1]):
                mlflow.log_metric(f'top3/{group}/all_concentrations', values[group]['top3'][-1], epoch)
        except:
            pass


def count_labels(arr):
    """
    Counts elements in array

    :param arr:
    :return:
    """
    elements_count = {}
    for element in arr:
        if element in elements_count:
            elements_count[element] += 1
        else:
            elements_count[element] = 1
    to_remove = []
    for key, value in elements_count.items():
        print(f"{key}: {value}")
        if value <= 2:
            to_remove += [key]

    return to_remove
