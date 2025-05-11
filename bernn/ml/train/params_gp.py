#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 28 2021
@author: Simon Pelletier
"""

# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LogisticRegression, SGDClassifier
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import LinearSVC # , SVC, NuSVC
from skopt.space import Real, Integer, Categorical

sgd_space = [
    Integer(1, 20000, 'uniform', name='features_cutoff'),
    # Real(0, 1, 'uniform', name='threshold'),
    Integer(1, 1000, 'uniform', name='max_iter'),
    Real(1e-4, 1, 'uniform', name='alpha'),
    Categorical(['log', 'modified_huber'], name='loss'),
    Categorical(['l2', 'l1', 'elasticnet'], name='penalty'),
    Categorical(['balanced'], name='class_weight'),
    Categorical([True, False], name='fit_intercept')
]
xgb_space = [
    Integer(1, 20000, 'uniform', name='features_cutoff'),
    # Real(0, 1, 'uniform', name='threshold'),
    Categorical(['l2', 'gpu_hist', 'hist'], name='tree_method'),
]

rfc_space = [
    Integer(1, 20000, 'uniform', name='features_cutoff'),
    # Real(0, 1, 'uniform', name='threshold'),
    Integer(1, 100, 'uniform', name="max_features"),
    Integer(2, 10, 'uniform', name="min_samples_split"),
    Integer(1, 10, 'uniform', name="min_samples_leaf"),
    Integer(1, 1000, 'uniform', name="n_estimators"),
    Categorical(['gini', 'entropy'], name="criterion"),
    Categorical([True, False], name="oob_score"),
    Categorical(['balanced'], name="class_weight"),
]

lda_space = [
    Integer(1, 20000, 'uniform', name='features_cutoff'),
    # Real(0, 1, 'uniform', name='threshold'),
    Categorical(["svd", "eigen"], name="solver"),
]
qda_space = [
    Integer(1, 20000, 'uniform', name='features_cutoff'),
    # Real(0, 1, 'uniform', name='threshold'),
]
nb_space = [
    Integer(1, 20000, 'uniform', name='features_cutoff'),
    # Real(0, 1, 'uniform', name='threshold'),
]
kn_space = [
    Integer(1, 20000, 'uniform', name='features_cutoff'),
    # Real(0, 1, 'uniform', name='threshold'),
]
logreg_space = [
    Integer(1, 20000, 'uniform', name='features_cutoff'),
    # Real(0, 1, 'uniform', name='threshold'),
    Integer(1, 20000, 'uniform', name='max_iter'),
    Real(1e-3, 20000, 'uniform', name='C'),
    Categorical(['saga'], name='solver'),
    Categorical(['l1', 'l2'], name='penalty'),
    Categorical([True, False], name='fit_intercept'),
    Categorical(['balanced'], name='class_weight'),
]
linsvc_space = [
    Integer(1, 20000, 'uniform', name='features_cutoff'),
    # Real(0, 1, 'uniform', name='threshold'),
    Real(1e-4, 1, 'log-uniform', name='tol'),
    Integer(1, 1000, 'uniform', name='max_iter'),
    Categorical(['l2'], name='penalty'),
    Real(1e-3, 10000, 'uniform', name='C'),
    Categorical(['balanced'], name='class_weight'),

]
nusvc_space = [
    Integer(1, 20000, 'uniform', name='features_cutoff'),
    Real(0, 1, 'uniform', name='nu'),
    Real(1e-4, 1, 'log-uniform', name='tol'),
    Integer(1, 1000, 'uniform', name='max_iter'),
    # Categorical(['l1', 'l2'], name='penalty'),
    Categorical(['linear', 'sigmoid'], name='kernel'),
    Categorical([True], name='probability'),
    Categorical(['balanced'], name='class_weight'),
]
svc_space = [
    Integer(1, 20000, 'uniform', name='features_cutoff'),
    # Real(0, 1, 'uniform', name='threshold'),
    Integer(1, 1000, 'uniform', name='max_iter'),
    Real(1e-3, 1000, 'uniform', name='C'),
    Categorical(['balanced'], name='class_weight'),
    Categorical(['linear'], name='kernel'),
    # Categorical(['linear', 'poly', 'rbf', 'sigmoid'], name='kernel'),
    # Categorical([True, False], name='probability'),
]

svcrbf_space = [
    Integer(1, 20000, 'uniform', name='features_cutoff'),
    # Real(0, 1, 'uniform', name='threshold'),
    Integer(1, 1000, 'uniform', name='max_iter'),
    Real(1e-3, 1000, 'uniform', name='C'),
    Categorical(['balanced'], name='class_weight'),
    Categorical(['rbf'], name='kernel'),
    # Categorical(['linear', 'poly', 'rbf', 'sigmoid'], name='kernel'),
    # Categorical([True, False], name='probability'),
]

rdcs = [
    LinearSVC(max_iter=100)
]

bag_space = [
    Categorical(rdcs, name='base_estimator'),
    # Integer(1, 10000, 'uniform', name='features_cutoff'),
    # Real(0, 1, 'uniform', name='threshold'),
    Integer(10, 100, 'uniform', name='n_estimators'),
]

models = {
    # "KNeighbors": [KNeighborsClassifier, kn_space],
    # "LinearSVC": [LinearSVC, linsvc_space],
    # "SVCLinear": [SVC, svc_space],
    # "LDA": [LinearDiscriminantAnalysis, lda_space],
    # "LogisticRegression": [LogisticRegression, logreg_space],
    "RandomForestClassifier": [RandomForestClassifier, rfc_space],
    # "Gaussian_Naive_Bayes": [GaussianNB, nb_space],
    # "QDA": [QuadraticDiscriminantAnalysis, qda_space],
    # "SGDClassifier": [SGDClassifier, sgd_space],
    # "BaggingClassifier": [BaggingClassifier, bag_space],
    # "AdaBoost_Classifier": [AdaBoostClassifier, param_grid_ada],
    # "Voting_Classifier": [VotingClassifier, param_grid_voting],
}
