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
import xgboost
import xgboost as xgb
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from bernn.ml.import_data import get_harvard, get_prostate
from bernn.utils.data_getters import get_amide, get_mice, get_bacteria1, get_cifar10

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import matthews_corrcoef as MCC
from skopt import gp_minimize
from sklearn import metrics
from sklearn.feature_selection import mutual_info_classif
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_auc_score
from bernn.ml.train.params_gp import *
from bernn.utils.utils import plot_confusion_matrix, scale_data, get_unique_labels, to_csv, scale_data_per_batch

np.random.seed(41)

warnings.filterwarnings('ignore')


def get_confusion_matrix(reals, preds, unique_labels):
    acc = np.mean([1 if pred == label else 0 for pred, label in zip(preds, reals)])
    cm = metrics.confusion_matrix(reals, preds)
    figure = plot_confusion_matrix(cm, unique_labels, acc)

    return figure


def save_confusion_matrix(fig, name, acc):
    dirs = '/'.join(name.split('/')[:-1])
    name = name.split('/')[-1]
    plt.title(f'Confusion Matrix (acc={np.round(acc, 3)})')
    os.makedirs(f'{dirs}/', exist_ok=True)
    stuck = True
    while stuck:
        try:
            fig.savefig(f"{dirs}/cm_{name}.png")
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


# TODO put this function in utils
def drop_lows(train_data):
    inds = train_data.index
    for ind in inds:
        if 'l' in ind.split('_')[1]:
            train_data = train_data.drop(ind)
    return train_data


# TODO put this function in utils
def drop_blks(train_data):
    inds = train_data.index
    for ind in inds:
        if 'blk' in ind:
            train_data = train_data.drop(ind)
    return train_data


def sort_by_mi(train_data, train_labels):
    mi = mutual_info_classif(train_data, train_labels)
    order = np.argsort(mi)

    return order


class Train:
    def __init__(self, model, data, unique_labels, hparams_names, args, ovr, binary=True):
        self.best_roc_score = -1
        try:
            with open(f'{args.destination}/saved_models/sklearn/best_params.json', "r") as json_file:
                self.previous_models = json.load(json_file)
        except:
            self.previous_models = {}
        self.ovr = ovr
        self.unique_labels = unique_labels
        self.binary = binary
        self.args = args
        # self.random = args.random
        self.model = model
        self.data = data
        self.hparams_names = hparams_names
        self.h_params = None
        # self.train_indices, self.test_indices, _ = split_train_test(self.labels)
        self.n_splits = args.n_splits

        self.n_repeats = args.n_repeats
        self.jackknife = args.jackknife
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
        try:
            assert features_cutoff is not None
        except AttributeError:
            exit('features_cutoff not in the hyperparameters. Leaving')

        # scaler = get_scaler(self.args.scaler)()

        print(f'Iteration: {self.iter}')

        self.scores_train = []
        self.scores_valid = []
        self.mccs_train = []
        self.mccs_valid = []
        self.top3_train = []
        self.top3_valid = []
        order = sort_by_mi(self.data['inputs']['train'], self.data['labels']['train'])
        x_train, x_valid = self.data['inputs']['train'].iloc[:, order], self.data['inputs']['valid'].iloc[:, order]
        y_train, y_valid = self.data['cats']['train'], self.data['cats']['valid']

        x_train, x_valid = x_train.iloc[:, :features_cutoff], x_valid.iloc[:, :features_cutoff]
        # x_train = scaler.fit_transform(x_train)
        # x_valid = scaler.transform(x_valid)

        m = self.model()
        m.set_params(**param_grid)
        if self.ovr:
            m = OneVsRestClassifier(m)
        m.fit(x_train, y_train)

        score_valid = m.score(x_valid, y_valid)
        score_train = m.score(x_train, y_train)
        print('valid_score:', score_valid, 'h_params:', param_grid)
        self.scores_train += [score_train]
        self.scores_valid += [score_valid]
        y_pred_train = m.predict(x_train)
        y_pred_valid = m.predict(x_valid)
        mcc_train = MCC(y_train, y_pred_train)
        mcc_valid = MCC(y_valid, y_pred_valid)

        self.mccs_train += [mcc_train]
        self.mccs_valid += [mcc_valid]

        try:
            y_proba_train = m.predict_proba(x_train)
            y_proba_valid = m.predict_proba(x_valid)
            y_top3_train = np.mean([1 if label.item() in pred.tolist()[::-1][:3] else 0 for pred, label in
                                    zip(y_proba_train.argsort(1), y_train)])
            y_top3_valid = np.mean([1 if label.item() in pred.tolist()[::-1][:3] else 0 for pred, label in
                                    zip(y_proba_valid.argsort(1), y_valid)])

            self.top3_train += [y_top3_train]
            self.top3_valid += [y_top3_valid]
        except:
            self.top3_train += [-1]
            self.top3_valid += [-1]

        if self.best_scores_valid is None:
            self.best_scores_valid = 0

        score = np.mean(self.scores_valid)
        if np.isnan(score):
            score = 0
        if model not in self.previous_models:
            self.previous_models[model] = {'valid_acc_mean': -1}
        if score > np.mean(self.best_scores_valid) and score > float(self.previous_models[model]['valid_acc_mean']):
            self.best_scores_train = self.scores_train
            self.best_scores_valid = self.scores_valid
            self.best_mccs_train = self.mccs_train
            self.best_mccs_valid = self.mccs_valid
            self.best_top3_train = self.top3_train
            self.best_top3_valid = self.top3_valid
            fig = get_confusion_matrix(y_pred_valid, y_valid, self.unique_labels)
            save_confusion_matrix(fig, f"{args.destination}/confusion_matrices/{model}_valid", acc=score)
            try:
                self.best_roc_score = save_roc_curve(m, x_valid, y_valid, self.unique_labels,
                                                     f"{args.destination}/ROC/{model}_valid", binary=args.binary,
                                                     acc=score)
            except:
                pass
            self.h_params = h_params
        return 1 - score


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--n_calls', type=int, default=10)
    parser.add_argument('--ovr', type=int, default=1, help='OneVsAll strategy')
    parser.add_argument('--correct_batches', type=int, default=0)
    parser.add_argument('--verbose', type=int, default=0)
    # parser.add_argument('--random', type=int, default=1)
    parser.add_argument('--binary', type=int, default=0)
    parser.add_argument('--n_splits', type=int, default=5)
    parser.add_argument('--run_name', type=str, default='4')
    parser.add_argument('--scaler', type=str, default='robust')
    parser.add_argument('--n_repeats', type=int, default=1)
    parser.add_argument('--groupkfold', type=int, default=1)
    parser.add_argument('--jackknife', type=str, default=0)
    parser.add_argument('--strategy', type=str, default='CU_DEM-AD')
    parser.add_argument('--bad_batches', type=str, default='')  # 0;23;22;21;20;19;18;17;16;15
    parser.add_argument('--remove_zeros', type=int, default=1)
    parser.add_argument("--input_dir", type=str, help="Path to intensities csv file")
    parser.add_argument("--output_dir", type=str, default='results', help="Path to intensities csv file")
    parser.add_argument('--path', type=str, default='./data/bacteria/')
    parser.add_argument('--use_valid', type=int, default=0, help='Use if valid data is in a seperate file')
    parser.add_argument('--use_test', type=int, default=0, help='Use if test data is in a seperate file')
    parser.add_argument('--cell_type', type=str, default="HepG2", help='')
    parser.add_argument('--csv_file', type=str, default="unique_genes.csv", help='')
    parser.add_argument('--dataset', type=str, default="bacteria", help='')
    parser.add_argument('--threshold', type=float, default=0, help='')
    parser.add_argument('--ncols', type=int, default=1000, help='')
    args = parser.parse_args()

    args.destination = f"{args.output_dir}/{args.dataset}/corrected{args.correct_batches}/" \
                       f"/binary{args.binary}/boot{args.jackknife}/" \
                       f"{args.scaler}/cv{args.n_splits}/nrep{args.n_repeats}/ovr{args.ovr}/{args.run_name}/"

    models = {
        # "BaggingClassifier": [BaggingClassifier, bag_space],
        # "KNeighbors": [KNeighborsClassifier, kn_space],
        # "xgboost": [xgboost.XGBClassifier, xgb_space],
        "RandomForestClassifier": [RandomForestClassifier, rfc_space],
        "LinearSVC": [LinearSVC, linsvc_space],
        # "LogisticRegression": [LogisticRegression, logreg_space],
        # "Gaussian_Naive_Bayes": [GaussianNB, nb_space],
        # "QDA": [QuadraticDiscriminantAnalysis, qda_space],
        "SGDClassifier": [SGDClassifier, sgd_space],
        "SVCLinear": [SVC, svc_space],
        # "LDA": [LinearDiscriminantAnalysis, lda_space],  # Creates an error...
        # "AdaBoost_Classifier": [AdaBoostClassifier, param_grid_ada],
        # "Voting_Classifier": [VotingClassifier, param_grid_voting],
    }

    try:
        with open(f'{args.destination}/saved_models/sklearn/best_params.json', "r") as json_file:
            previous_models = json.load(json_file)
    except:
        previous_models = {}
    best_params_dict = previous_models
    os.makedirs(f"{args.destination}/saved_models/sklearn/", exist_ok=True)
    if args.dataset == 'alzheimer':
        data, unique_labels, unique_batches = get_harvard(args)
    elif args.dataset == 'bacteria':
        data, unique_labels, unique_batches = get_bacteria1('data/bacteria', args)
    elif args.dataset == 'prostate':
        data, unique_labels, unique_batches = get_prostate(args, path='data')
    else:
        exit('Invalid dataset')
    # data, unique_labels, unique_batches = data['data'], data['unique_labels'], data['unique_batches']
    data, scaler = scale_data_per_batch(args.scaler, data)
    for model in models:
        print(f"Training {model}")
        hparams_names = [x.name for x in models[model][1]]
        train = Train(models[model][0], data, unique_labels, hparams_names, args, ovr=args.ovr, binary=True)
        res = gp_minimize(train.train, models[model][1], n_calls=args.n_calls, random_state=42)

        features_cutoff = None
        param_grid = {}
        for name, param in zip(hparams_names, train.h_params):
            if name == 'features_cutoff':
                features_cutoff = param
            elif name == 'threshold':
                threshold = param
            else:
                param_grid[name] = param
        try:
            assert features_cutoff is not None
        except AttributeError:
            exit('features_cutoff not in the hyperparameters. Leaving')

        all_x_train = data["inputs"]["train"]
        test_data = data["inputs"]["test"]

        m = models[model][0]()
        m.set_params(**param_grid)
        if args.ovr:
            m = OneVsRestClassifier(m)

        m.fit(all_x_train, data['cats']['train'])
        test_score = m.score(test_data, data["cats"]["test"])

        train_score = m.score(all_x_train, data['cats']['train'])
        y_preds_test = m.predict(test_data)
        y_preds_train = m.predict(all_x_train)
        print(y_preds_test.shape)
        mcc_test = MCC(data["cats"]["test"], y_preds_test)
        mcc_train = MCC(data['cats']['train'], y_preds_train)
        try:
            y_proba_train = m.predict_proba(all_x_train)
            y_proba_test = m.predict_proba(test_data)
            y_top3_train = np.mean([1 if label.item() in pred.tolist()[::-1][:3] else 0 for pred, label in
                                    zip(y_proba_train.argsort(1), data['cats']['train'])])
            y_top3_test = np.mean([1 if label.item() in pred.tolist()[::-1][:3] else 0 for pred, label in
                                   zip(y_proba_test.argsort(1), data["cats"]["test"])])
        except:
            y_top3_valid = -1
            y_top3_test = -1
        if train.best_scores_train is not None:
            param_grid = {}
            for name, param in zip(hparams_names, train.h_params):
                param_grid[name] = param
            best_params_dict[model] = param_grid

            best_params_dict[model]['train_acc_mean'] = np.mean(train.best_scores_train)
            best_params_dict[model]['train_acc_std'] = np.std(train.best_scores_train)
            best_params_dict[model]['valid_acc_mean'] = np.mean(train.best_scores_valid)
            best_params_dict[model]['valid_acc_std'] = np.std(train.best_scores_valid)
            best_params_dict[model]['test_acc'] = test_score

            best_params_dict[model]['train_mcc_mean'] = np.mean(train.best_mccs_train)
            best_params_dict[model]['train_mcc_std'] = np.std(train.best_mccs_train)
            best_params_dict[model]['valid_mcc_mean'] = np.mean(train.best_mccs_valid)
            best_params_dict[model]['valid_mcc_std'] = np.std(train.best_mccs_valid)
            best_params_dict[model]['test_mcc'] = mcc_test

            best_params_dict[model]['train_top3_mean'] = np.mean(train.best_top3_train)
            best_params_dict[model]['train_top3_std'] = np.std(train.best_top3_train)
            best_params_dict[model]['valid_top3_mean'] = np.mean(train.best_top3_valid)
            best_params_dict[model]['valid_top3_std'] = np.std(train.best_top3_valid)
            best_params_dict[model]['test_top3'] = y_top3_test

        if model not in previous_models:
            previous_models[model]['valid_acc_mean'] = -1

        print(f'test score: {test_score}')
        if float(best_params_dict[model]['valid_acc_mean']) >= float(previous_models[model]['valid_acc_mean']):
            model_filename = f"{args.destination}/saved_models/sklearn/{model}.sav"
            with open(model_filename, 'wb') as file:
                pickle.dump(m, file)
            scaler_filename = f"{args.destination}/saved_models/sklearn/scaler_{model}.sav"
            with open(scaler_filename, 'wb') as file:
                pickle.dump(scaler, file)

            fig = get_confusion_matrix(y_preds_test, data["cats"]["test"], unique_labels)
            save_confusion_matrix(fig, f"{args.destination}/confusion_matrices/{model}_test", acc=test_score)
            try:
                save_roc_curve(m, test_data, data["cats"]["test"], unique_labels,
                               f"{args.destination}/ROC/{model}_test",
                               binary=args.binary,
                               acc=test_score)
            except:
                print('No proba function, or something else.')

    for name in best_params_dict.keys():
        if name in previous_models.keys():
            prev_valid_acc = float(previous_models[name]['valid_acc_mean'])
        else:
            prev_valid_acc = -1
        if float(best_params_dict[name]['valid_acc_mean']) > prev_valid_acc:
            for param in best_params_dict[name].keys():
                best_params_dict[name][param] = str(best_params_dict[name][param])
        else:
            for param in previous_models[name].keys():
                best_params_dict[name][param] = str(previous_models[name][param])
    for name in previous_models.keys():
        if name not in best_params_dict.keys():
            best_params_dict[name] = {}
            for param in previous_models[name].keys():
                best_params_dict[name][param] = str(previous_models[name][param])

    with open(f'{args.destination}/saved_models/sklearn/best_params.json', "w") as read_file:
        json.dump(best_params_dict, read_file)
