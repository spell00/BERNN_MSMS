#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 28 2021
@author: Simon Pelletier
"""

import warnings
import pickle

import numpy as np
import pandas as pd
import os
import csv
import json
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import matthews_corrcoef as MCC
from sklearn.multiclass import OneVsRestClassifier
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from bernn.utils.utils import plot_confusion_matrix
import mlflow

np.random.seed(42)

warnings.filterwarnings('ignore')

DIR = 'src/models/sklearn/'


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


def get_confusion_matrix(reals, preds, unique_labels):
    acc = np.mean([1 if pred == label else 0 for pred, label in zip(preds, reals)])
    cm = metrics.confusion_matrix(reals, preds)
    figure = plot_confusion_matrix(cm, unique_labels, acc)

    # cm = np.zeros([len(unique_labels), len(unique_labels)])
    # for real, pred in zip(reals, preds):
    #     confusion_matrix[int(real), int(pred)] += 1
    # indices = [f"{lab}" for lab in unique_labels]
    # columns = [f"{lab}" for lab in unique_labels]
    return figure


def save_confusion_matrix(fig, name, acc, mcc, group, rep):
    # sns_plot = sns.heatmap(df, annot=True, square=True, cmap="YlGnBu",
    #                        annot_kws={"size": 35 / np.sqrt(len(df))})
    # fig = sns_plot.get_figure()
    dirs = '/'.join(name.split('/')[:-1])
    name = name.split('/')[-1]
    plt.title(f'Confusion Matrix (acc={np.round(acc, 3)}, mcc={np.round(mcc, 3)})')
    os.makedirs(f'{dirs}/', exist_ok=True)
    stuck = True
    while stuck:
        try:
            fig.savefig(f"{dirs}/cm_{name}_{group}_{rep}.png")
            stuck = False
        except:
            print('stuck...')
    plt.close()


def save_roc_curve(model, x_test, y_test, unique_labels, name, binary, acc):
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
                plt.savefig(f'{dirs}/ROC_{name}.png')
                stuck = False
            except:
                print('stuck...')
        plt.close()
    else:
        # Compute ROC curve and ROC area for each class
        from sklearn.preprocessing import label_binarize
        y_pred_proba = model.predict_proba(x_test)
        y_preds = model.predict(x_test)
        n_classes = len(unique_labels)
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        classes = np.arange(len(unique_labels))

        bin_label = label_binarize(y_test, classes=classes)
        roc_score = roc_auc_score(y_true=label_binarize(y_test, classes=classes[bin_label.sum(0) != 0]),
                                  y_score=label_binarize(y_preds, classes=classes[bin_label.sum(0) != 0]),
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
                plt.savefig(f'{dirs}/ROC_{name}.png')
                stuck = False
            except:
                print('stuck...')

        plt.close()

    return roc_score


class Train:
    def __init__(self, name, model, data, hparams_names, log_path, args, logger, ovr, model_name='RF', binary=True, mlops='None'):
        self.best_roc_score = -1
        self.ovr = ovr
        self.binary = binary
        self.args = args
        self.log_path = log_path
        self.model = model
        self.model_name = model_name
        self.data = data
        self.logger = logger
        self.hparams_names = hparams_names
        # self.train_indices, self.test_indices, _ = split_train_test(self.labels)
        # self.n_splits = args.n_splits
        # self.n_repeats = args.n_repeats
        # self.jackknife = args.jackknife
        self.best_scores_train = None
        self.best_scores_valid = None
        self.best_mccs_train = None
        self.best_mccs_valid = None
        self.scores_train = None
        self.scores_valid = None
        self.mccs_train = None
        self.mccs_valid = None
        self.y_preds = np.array([])
        self.y_valids = np.array([])
        self.top3_valid = None
        self.top3_train = None
        self.iter = 0
        self.model = model
        self.name = name
        self.mlops = mlops
        self.best_params_dict = {}

    def train(self, h_params):
        self.iter += 1
        features_cutoff = None
        param_grid = {}
        for name, param in zip(self.hparams_names, h_params):
            if name == 'features_cutoff':
                features_cutoff = param
            elif name == 'threshold':
                threshold = param
            else:
                param_grid[name] = param
        # try:
        #     assert features_cutoff is not None
        # except AttributeError:
        #     exit('features_cutoff not in the hyperparameters. Leaving')

        train_labels = np.concatenate(self.data['cats']['train']).argmax(1)
        train_data = np.concatenate(self.data['inputs']['train'])
        train_batches = np.concatenate(self.data['batches']['train'])

        valid_labels = np.concatenate(self.data['cats']['valid']).argmax(1)
        valid_data = np.concatenate(self.data['inputs']['valid'])
        valid_batches = np.concatenate(self.data['batches']['valid'])

        test_labels = np.concatenate(self.data['cats']['test']).argmax(1)
        test_data = np.concatenate(self.data['inputs']['test'])
        test_batches = np.concatenate(self.data['batches']['test'])

        unique_labels = []
        for l in train_labels:
            if l not in unique_labels:
                unique_labels += [l]

        unique_labels = np.array(unique_labels)
        train_classes = np.array([np.argwhere(l == unique_labels)[0][0] for l in train_labels])
        valid_classes = np.array([np.argwhere(l == unique_labels)[0][0] for l in valid_labels])
        test_classes = np.array([np.argwhere(l == unique_labels)[0][0] for l in test_labels])

        print(f'Iteration: {self.iter}')

        m = self.model()
        m.set_params(**param_grid)
        if self.ovr:
            m = OneVsRestClassifier(m)
        try:
            m.fit(train_data, train_classes)
        except:
            return 1

        score_valid = m.score(valid_data, valid_classes)
        score_train = m.score(train_data, train_classes)
        score_test = m.score(test_data, test_classes)
        print('valid_score:', score_valid, 'h_params:', param_grid)
        # scores_train = score_train
        # scores_valid = score_valid

        y_pred_train = m.predict(train_data)
        y_pred_valid = m.predict(valid_data)
        y_pred_test = m.predict(test_data)

        mcc_train = MCC(train_classes, y_pred_train)
        mcc_valid = MCC(valid_classes, y_pred_valid)
        mcc_test = MCC(test_classes, y_pred_test)
        # mccs_train = mcc_train
        # mccs_valid = mcc_valid

        # try:
        # y_proba_train = m.predict_proba(train_data)
        # y_proba_valid = m.predict_proba(valid_data)
        # y_proba_test = m.predict_proba(test_data)
        # top3_train = np.mean([1 if label.item() in pred.tolist()[::-1][:3] else 0 for pred, label in
        #                         zip(y_proba_train.argsort(1), train_classes)])
        # top3_valid = np.mean([1 if label.item() in pred.tolist()[::-1][:3] else 0 for pred, label in
        #                         zip(y_proba_valid.argsort(1), valid_classes)])
        # top3_test = np.mean([1 if label.item() in pred.tolist()[::-1][:3] else 0 for pred, label in
        #                         zip(y_proba_test.argsort(1), test_classes)])

        # top3_train = y_top3_train
        # top3_valid = y_top3_valid
        # except:
        #     self.top3_train += [-1]
        #     self.top3_valid += [-1]

        if self.best_scores_valid is None:
            self.best_scores_valid = 0

        # score = np.mean(self.scores_valid)
        if np.isnan(score_valid):
            score_valid = 0

        if score_valid > self.best_scores_valid:
            self.best_scores_train = score_train
            self.best_scores_valid = score_valid
            self.best_scores_test = score_test
            self.best_mccs_train = mcc_train
            self.best_mccs_valid = mcc_valid
            self.best_mccs_test = mcc_test
            # self.best_top3_train = top3_train
            # self.best_top3_valid = top3_valid
            # self.best_top3_test = top3_test
            fig = get_confusion_matrix(train_classes, y_pred_train, unique_labels)
            save_confusion_matrix(fig, f"{self.log_path}/confusion_matrices/{self.name}_{self.model_name}_train",
                                  acc=score_train, mcc=mcc_train, group='train')
            if self.mlops == 'tensorboard':
                self.logger.add_figure(f"cm_{self.name}_{self.model_name}_train", fig, self.iter)
            else:
                self.logger[f"cm_{self.name}_{self.model_name}_train"].upload(fig)

            fig = get_confusion_matrix(valid_classes, y_pred_valid, unique_labels)
            save_confusion_matrix(fig, f"{self.log_path}/confusion_matrices/{self.name}_{self.model_name}_valid",
                                  acc=score_valid, mcc=mcc_valid, group='train')
            if self.mlops == 'tensorboard':
                self.logger.add_figure(f"cm_{self.name}_{self.model_name}_valid", fig, self.iter)
            else:
                self.logger[f"cm_{self.name}_{self.model_name}_valid"].upload(fig)
            fig = get_confusion_matrix(test_classes, y_pred_test, unique_labels)
            save_confusion_matrix(fig, f"{self.log_path}/confusion_matrices/{self.name}_{self.model_name}_test",
                                  acc=score_valid, mcc=mcc_valid, group='train')
            if self.mlops == 'tensorboard':
                self.logger.add_figure(f"cm_{self.name}_{self.model_name}_test", fig, self.iter)
            else:
                self.logger[f"cm_{self.name}_{self.model_name}_test"].upload(fig)
            try:
                self.best_roc_train = save_roc_curve(m, train_data, train_classes, unique_labels,
                                                     f"{self.log_path}/ROC/{self.name}_{self.model_name}_train", binary=0,
                                                     acc=score_train)
            except:
                pass
            try:
                self.best_roc_valid = save_roc_curve(m, valid_data, valid_classes, unique_labels,
                                                     f"{self.log_path}/ROC/{self.name}_{self.model_name}_valid", binary=0,
                                                     acc=score_valid)
            except:
                pass
            try:
                self.best_roc_test = save_roc_curve(m, test_data, test_classes, unique_labels,
                                                     f"{self.log_path}/ROC/{self.name}_{self.model_name}_test", binary=0,
                                                     acc=score_test)
            except:
                pass
            self.save_best_model_hparams(self.hparams_names, param_grid)

        return 1 - score_valid

    def save_best_model_hparams(self, hparams_names, params):
        param_grid = {}
        for name, param in zip(hparams_names, params):
            param_grid[name] = param
        self.best_params_dict = param_grid

        self.best_params_dict['train_acc'] = self.best_scores_train
        self.best_params_dict['valid_acc'] = self.best_scores_valid
        self.best_params_dict['test_acc'] = self.best_scores_test

        self.best_params_dict['train_mcc'] = self.best_mccs_train
        self.best_params_dict['valid_mcc'] = self.best_mccs_valid
        self.best_params_dict['test_mcc'] = self.best_mccs_test

        # self.best_params_dict['train_top3'] = self.best_top3_train
        # self.best_params_dict['valid_top3'] = self.best_top3_valid
        # self.best_params_dict['test_top3'] = self.best_top3_test
        os.makedirs(f'{self.log_path}/saved_models/', exist_ok=True)
        with open(f'{self.log_path}/saved_models/best_params_{self.name}_{self.model_name}.json', "w") as read_file:
            json.dump(self.best_params_dict, read_file)

class Train2:
    def __init__(self, name, model, data, hparams_names, log_path, args, logger, ovr, model_name='RF', 
                 binary=True, mlops='None', rep=0):
        self.best_roc_score = -1
        self.ovr = ovr
        self.rep = rep
        self.binary = binary
        self.args = args
        self.log_path = log_path
        self.model = model
        self.model_name = model_name
        self.data = data
        self.logger = logger
        self.hparams_names = hparams_names
        # self.train_indices, self.test_indices, _ = split_train_test(self.labels)
        # self.n_splits = args.n_splits
        # self.n_repeats = args.n_repeats
        # self.jackknife = args.jackknife
        self.best_scores_train = None
        self.best_scores_valid = None
        self.best_mccs_train = None
        self.best_mccs_valid = None
        self.scores_train = None
        self.scores_valid = None
        self.mccs_train = None
        self.mccs_valid = None
        self.y_preds = np.array([])
        self.y_valids = np.array([])
        self.top3_valid = None
        self.top3_train = None
        self.iter = 0
        self.model = model
        self.name = name
        self.mlops = mlops
        self.best_params_dict = {}

    def train(self, h_params):
        self.iter += 1
        features_cutoff = None
        param_grid = {}
        for name, param in zip(self.hparams_names, h_params):
            if name == 'features_cutoff':
                features_cutoff = param
            elif name == 'threshold':
                threshold = param
            else:
                param_grid[name] = param
        # try:
        #     assert features_cutoff is not None
        # except AttributeError:
        #     exit('features_cutoff not in the hyperparameters. Leaving')

        train_labels = self.data['cats']['train']
        train_data = self.data['inputs']['train'].iloc[:, 2:]
        train_batches = self.data['batches']['train']

        valid_labels = self.data['cats']['valid']
        valid_data = self.data['inputs']['valid'].iloc[:, 2:]
        valid_batches = self.data['batches']['valid']

        test_labels = self.data['cats']['test']
        test_data = self.data['inputs']['test'].iloc[:, 2:]
        test_batches = self.data['batches']['test']

        unique_labels = []
        for l in train_labels:
            if l not in unique_labels:
                unique_labels += [l]

        unique_labels = np.array(unique_labels)
        train_classes = np.array([np.argwhere(l == unique_labels)[0][0] for l in train_labels])
        valid_classes = np.array([np.argwhere(l == unique_labels)[0][0] for l in valid_labels])
        test_classes = np.array([np.argwhere(l == unique_labels)[0][0] for l in test_labels])

        print(f'Iteration: {self.iter}')

        m = self.model()
        m.set_params(**param_grid)
        if self.ovr:
            m = OneVsRestClassifier(m)
        try:
            m.fit(train_data, train_classes)
        except:
            return 1

        score_valid = m.score(valid_data, valid_classes)
        score_train = m.score(train_data, train_classes)
        score_test = m.score(test_data, test_classes)
        print('valid_score:', score_valid, 'h_params:', param_grid)
        # scores_train = score_train
        # scores_valid = score_valid

        y_pred_train = m.predict(train_data)
        y_pred_valid = m.predict(valid_data)
        y_pred_test = m.predict(test_data)

        mcc_train = MCC(train_classes, y_pred_train)
        mcc_valid = MCC(valid_classes, y_pred_valid)
        mcc_test = MCC(test_classes, y_pred_test)
        # mccs_train = mcc_train
        # mccs_valid = mcc_valid

        # try:
        # y_proba_train = m.predict_proba(train_data)
        # y_proba_valid = m.predict_proba(valid_data)
        # y_proba_test = m.predict_proba(test_data)
        # top3_train = np.mean([1 if label.item() in pred.tolist()[::-1][:3] else 0 for pred, label in
        #                         zip(y_proba_train.argsort(1), train_classes)])
        # top3_valid = np.mean([1 if label.item() in pred.tolist()[::-1][:3] else 0 for pred, label in
        #                         zip(y_proba_valid.argsort(1), valid_classes)])
        # top3_test = np.mean([1 if label.item() in pred.tolist()[::-1][:3] else 0 for pred, label in
        #                         zip(y_proba_test.argsort(1), test_classes)])

        # top3_train = y_top3_train
        # top3_valid = y_top3_valid
        # except:
        #     self.top3_train += [-1]
        #     self.top3_valid += [-1]

        if self.best_scores_valid is None:
            self.best_scores_valid = 0

        # score = np.mean(self.scores_valid)
        if np.isnan(score_valid):
            score_valid = 0

        if score_valid > self.best_scores_valid:
            self.best_scores_train = score_train
            self.best_scores_valid = score_valid
            self.best_scores_test = score_test
            self.best_mccs_train = mcc_train
            self.best_mccs_valid = mcc_valid
            self.best_mccs_test = mcc_test
            # self.best_top3_train = top3_train
            # self.best_top3_valid = top3_valid
            # self.best_top3_test = top3_test
            fig = get_confusion_matrix(train_classes, y_pred_train, unique_labels)
            save_confusion_matrix(fig, f"{self.log_path}/confusion_matrices/{self.name}_{self.model_name}_train",
                                  acc=score_train, mcc=mcc_train, group='train', rep=self.rep)
            if self.mlops == 'tensorboard':
                self.logger.add_figure(f"cm_{self.name}_{self.model_name}_train", fig, self.iter)
            if self.mlops == 'neptune':
                self.logger[f"cm_{self.name}_{self.model_name}_train"].upload(fig)
            if self.mlops == 'mlflow':
                mlflow.log_figure(fig, f"{self.log_path}/confusion_matrices/cm_{self.name}_train_{self.rep}.png")

            fig = get_confusion_matrix(valid_classes, y_pred_valid, unique_labels)
            save_confusion_matrix(fig, f"{self.log_path}/confusion_matrices/{self.name}_{self.model_name}_valid",
                                  acc=score_valid, mcc=mcc_valid, group='valid', rep=self.rep)
            if self.mlops == 'tensorboard':
                self.logger.add_figure(f"cm_{self.name}_{self.model_name}_valid", fig, self.iter)
            if self.mlops == 'neptune':
                self.logger[f"cm_{self.name}_{self.model_name}_valid"].upload(fig)
            if self.mlops == 'mlflow':
                mlflow.log_figure(fig, f"{self.log_path}/confusion_matrices/cm_{self.name}_valid_{self.rep}.png")

            fig = get_confusion_matrix(test_classes, y_pred_test, unique_labels)
            save_confusion_matrix(fig, f"{self.log_path}/confusion_matrices/{self.name}_{self.model_name}_test",
                                  acc=score_test, mcc=mcc_test, group='test', rep=self.rep)
            if self.mlops == 'tensorboard':
                self.logger.add_figure(f"cm_{self.name}_{self.model_name}_test", fig, self.iter)
            if self.mlops == 'neptune':
                self.logger[f"cm_{self.name}_{self.model_name}_test"].upload(fig)
            if self.mlops == 'mlflow':
                mlflow.log_figure(fig, f"{self.log_path}/confusion_matrices/cm_{self.name}_test_{self.rep}.png")

            # try:
            #     self.best_roc_train = save_roc_curve(m, train_data, train_classes, unique_labels,
            #                                          f"{self.log_path}/ROC/{self.name}_{self.model_name}_train", binary=0,
            #                                          acc=score_train)
            # except:
            #     pass
            # try:
            #     self.best_roc_valid = save_roc_curve(m, valid_data, valid_classes, unique_labels,
            #                                          f"{self.log_path}/ROC/{self.name}_{self.model_name}_valid", binary=0,
            #                                          acc=score_valid)
            # except:
            #     pass
            # try:
            #     self.best_roc_test = save_roc_curve(m, test_data, test_classes, unique_labels,
            #                                          f"{self.log_path}/ROC/{self.name}_{self.model_name}_test", binary=0,
            #                                          acc=score_test)
            # except:
            #     pass
            self.save_best_model_hparams(self.hparams_names, param_grid)

        return 1 - score_valid

    def save_best_model_hparams(self, hparams_names, params):
        param_grid = {}
        for name, param in zip(hparams_names, params):
            param_grid[name] = param
        self.best_params_dict = param_grid

        self.best_params_dict['train_acc'] = self.best_scores_train
        self.best_params_dict['valid_acc'] = self.best_scores_valid
        self.best_params_dict['test_acc'] = self.best_scores_test

        self.best_params_dict['train_mcc'] = self.best_mccs_train
        self.best_params_dict['valid_mcc'] = self.best_mccs_valid
        self.best_params_dict['test_mcc'] = self.best_mccs_test

        # self.best_params_dict['train_top3'] = self.best_top3_train
        # self.best_params_dict['valid_top3'] = self.best_top3_valid
        # self.best_params_dict['test_top3'] = self.best_top3_test
        os.makedirs(f'{self.log_path}/saved_models/', exist_ok=True)
        with open(f'{self.log_path}/saved_models/best_params_{self.name}_{self.model_name}.json', "w") as read_file:
            json.dump(self.best_params_dict, read_file)

