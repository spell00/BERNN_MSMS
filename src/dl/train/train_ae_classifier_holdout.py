NEPTUNE_API_TOKEN = "YOUR-API-KEY"
NEPTUNE_PROJECT_NAME = "YOUR-PROJECT-NAME"
NEPTUNE_MODEL_NAME = "YOUR-MODEL-NAME"

import matplotlib
from src.utils.pool_metrics import log_pool_metrics

matplotlib.use('Agg')
CUDA_VISIBLE_DEVICES = ""

import uuid
import shutil

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import json
import copy
import torch
from torch import nn
import os

from sklearn import metrics
from tensorboardX import SummaryWriter
from ax.service.managed_loop import optimize
from sklearn.metrics import matthews_corrcoef as MCC
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from src.ml.train.params_gp import *
from src.utils.data_getters import get_alzheimer, get_amide, get_mice, get_data
from src.dl.models.pytorch.aedann import ReverseLayerF
from src.dl.models.pytorch.aedann import AutoEncoder2 as AutoEncoder
from src.dl.models.pytorch.aedann import SHAPAutoEncoder2 as SHAPAutoEncoder
from src.dl.models.pytorch.utils.loggings import TensorboardLoggingAE, log_metrics, log_input_ordination, \
    LogConfusionMatrix, log_plots, log_neptune, log_shap, log_mlflow
from src.dl.models.pytorch.utils.dataset import get_loaders, get_loaders_no_pool
from src.utils.utils import scale_data, to_csv
from src.dl.models.pytorch.utils.utils import get_optimizer, to_categorical, get_empty_dicts, get_empty_traces, \
    log_traces, get_best_values, add_to_logger, add_to_neptune, \
    add_to_mlflow
from src.dl.models.pytorch.utils.loggings import make_data
import neptune.new as neptune
import mlflow
import warnings
from datetime import datetime


# from fastapi import BackgroundTasks, FastAPI
# from threading import Thread

# app = FastAPI()

warnings.filterwarnings("ignore")

random.seed(1)
torch.manual_seed(1)
np.random.seed(1)

class TrainAE:

    def __init__(self, args, path, fix_thres=-1, load_tb=False, log_metrics=False, keep_models=True, log_inputs=True,
                 log_plots=False, log_tb=False, log_neptune=False, log_mlflow=True, groupkfold=True):
        """

        Args:
            args: contains multiple arguments passed in the command line
            log_path (str): Path where the tensorboard logs are saved
            path (str): Path to the data (in .csv format)
            fix_thres (float): If 1 > fix_thres >= 0 then the threshold is fixed to that value.
                       any other value means the threshold won't be fixed and will be
                       learned as an hyperparameter
            load_tb (bool): If True, loads previous runs already saved
            log_metrics (bool): Whether or not to keep the batch effect metrics
            keep_models (bool): Whether or not to save the models trained
                                (can take a lot of space if training a lot of models)
            log_inputs (bool): Whether or not to log graphs or batch effect metrics
                                of the scaled inputs
            log_plots (bool): For each optimization iteration, on the first iteration, wether or
                              not to plot PCA, UMAP, CCA and LDA of the encoded and reconstructed
                              representations.
            log_tb (bool): Whether or not to use tensorboard.
            log_mlflow (bool): Wether or not to use mlflow.
        """
        self.hparams_names = None
        self.best_acc = 0
        self.best_mcc = -1
        self.best_closs = np.inf
        self.logged_inputs = False
        self.log_tb = log_tb
        self.log_neptune = log_neptune
        self.log_mlflow = log_mlflow
        self.args = args
        self.path = path
        self.log_metrics = log_metrics
        self.log_plots = log_plots
        self.log_inputs = log_inputs
        self.keep_models = keep_models
        self.fix_thres = fix_thres
        self.load_tb = load_tb
        self.groupkfold = groupkfold
        self.foldername = None

        self.verbose = 1

        self.n_cats = None
        self.data = None
        self.unique_labels = None
        self.unique_batches = None

        self.pools = False

    def make_samples_weights(self):
        """
        Makes the samples weights for the training set. The weights are calculated as the inverse of the frequency of
        each class in the training set.
        Returns:

        """
        self.n_batches = len(set(self.data['batches']['all']))
        self.class_weights = {
            label: 1 / (len(np.where(label == self.data['labels']['train'])[0]) /
                        self.data['labels']['train'].shape[0])
            for label in self.unique_labels if
            label in self.data['labels']['train'] and label not in ["MCI-AD", 'MCI-other', 'DEM-other', 'NPH']}
        self.unique_unique_labels = list(self.class_weights.keys())
        for group in ['train', 'valid', 'test']:
            inds_to_keep = np.array([i for i, x in enumerate(self.data['labels'][group]) if x in self.unique_labels])
            self.data['inputs'][group] = self.data['inputs'][group].iloc[inds_to_keep]
            try:
                self.data['names'][group] = self.data['names'][group].iloc[inds_to_keep]
            except:
                self.data['names'][group] = self.data['names'][group][inds_to_keep]

            self.data['labels'][group] = self.data['labels'][group][inds_to_keep]
            self.data['cats'][group] = self.data['cats'][group][inds_to_keep]
            self.data['batches'][group] = self.data['batches'][group][inds_to_keep]

        self.samples_weights = {
            group: [self.class_weights[label] if label not in ["MCI-AD", 'MCI-other', 'DEM-other', 'NPH'] else 0 for
                    name, label in
                    zip(self.data['names'][group],
                        self.data['labels'][group])] if group == 'train' else [
                1 if label not in ["MCI-AD", 'MCI-other', 'DEM-other', 'NPH'] else 0 for name, label in
                zip(self.data['names'][group], self.data['labels'][group])] for group in
            ['train', 'valid', 'test']}
        self.n_cats = len(self.class_weights)  # + 1  # for pool samples
        self.scaler = None

    def train(self, params):
        """
        Args:
            params: Contains the hyperparameters to be optimized

        Returns:
            best_closs: The best classification loss on the valid set

        """
        start_time = datetime.now()
        # Fixing the hyperparameters that are not optimized
        if args.dloss not in ['revTriplet', 'revDANN', 'DANN',
                              'inverseTriplet', 'normae'] or 'gamma' not in params:
            # gamma = 0 will ensure DANN is not learned
            params['gamma'] = 0
        if not self.args.variational or 'beta' not in params:
            # beta = 0 because useless outside a variational autoencoder
            params['beta'] = 0
        if not self.args.zinb or 'zeta' not in params:
            # zeta = 0 because useless outside a zinb autoencoder
            params['zeta'] = 0
        if 1 > self.fix_thres >= 0:
            # fixes the threshold of 0s tolerated for a feature
            params['thres'] = self.fix_thres
        else:
            params['thres'] = 0

        # params['dropout'] = 0
        # params['smoothing'] = 0
        # params['margin'] = 0
        # params['wd'] = 0
        print(params)

        # Assigns the hyperparameters getting optimized
        smooth = params['smoothing']
        layer1 = params['layer1']
        layer2 = params['layer2']
        scale = params['scaler']
        self.gamma = params['gamma']
        self.beta = params['beta']
        self.zeta = params['zeta']
        thres = params['thres']
        wd = params['wd']
        nu = params['nu']
        lr = params['lr']

        dropout = params['dropout']
        margin = params['margin']

        self.args.scaler = scale
        self.args.warmup = params['warmup']
        self.args.disc_b_warmup = params['disc_b_warmup']

        # ncols = params['ncols']
        # if ncols > self.data['inputs']['all'].shape[1]:
        #     ncols = self.data['inputs']['all'].shape[1]

        optimizer_type = 'adam'
        metrics = {'pool_metrics': {}}
        # self.log_path is where tensorboard logs are saved
        self.foldername = str(uuid.uuid4())

        self.complete_log_path = f'logs/ae_classifier_holdout/{self.foldername}'
        loggers = {'cm_logger': LogConfusionMatrix(self.complete_log_path)}
        print(f'See results using: tensorboard --logdir={self.complete_log_path} --port=6006')

        hparams_filepath = self.complete_log_path + '/hp'
        os.makedirs(hparams_filepath, exist_ok=True)
        self.args.model_name = 'ae_classifier_holdout'
        if self.log_tb:
            loggers['tb_logging'] = TensorboardLoggingAE(hparams_filepath, params, variational=self.args.variational,
                                                         zinb=self.args.zinb,
                                                         tw=self.args.tied_weights,
                                                         dloss=self.args.dloss,
                                                         tl=0,  # to remove, useless now
                                                         pseudo=self.args.predict_tests,
                                                         train_after_warmup=self.args.train_after_warmup,
                                                         berm='none',  # to remove, useless now
                                                         args=self.args)
        else:
            model = None
            run = None

        if self.log_neptune:
            # Create a Neptune run object
            run = neptune.init_run(
                project=NEPTUNE_PROJECT_NAME,
                api_token=NEPTUNE_API_TOKEN,
            )  # your credentials
            model = neptune.init_model_version(
                model=NEPTUNE_MODEL_NAME,
                project=NEPTUNE_PROJECT_NAME,
                api_token=NEPTUNE_API_TOKEN,
                # your credentials
            )
            run["dataset"].track_files(f"{self.path}/{self.args.csv_file}")
            run["metadata"].track_files(
                f"{self.path}/subjects_experiment_ATN_verified_diagnosis.csv"
            )
            # Track metadata and hyperparameters by assigning them to the run
            model["inputs_type"] = run["inputs_type"] = args.csv_file.split(".csv")[0]
            model["best_unique"] = run["best_unique"] = args.best_features_file.split(".tsv")[0]
            model["use_valid"] = run["use_valid"] = args.use_valid
            model["use_test"] = run["use_test"] = args.use_test
            model["tied_weights"] = run["tied_weights"] = args.tied_weights
            model["random_recs"] = run["random_recs"] = args.random_recs
            model["train_after_warmup"] = run["train_after_warmup"] = args.train_after_warmup
            model["dloss"] = run["dloss"] = args.dloss
            model["predict_tests"] = run["predict_tests"] = args.predict_tests
            model["variational"] = run["variational"] = args.variational
            model["zinb"] = run["zinb"] = args.zinb
            model["threshold"] = run["threshold"] = args.threshold
            model["rec_loss_type"] = run["rec_loss_type"] = args.rec_loss
            model["strategy"] = run["strategy"] = args.strategy
            model["bad_batches"] = run["bad_batches"] = args.bad_batches
            model["remove_zeros"] = run["remove_zeros"] = args.remove_zeros
            model["parameters"] = run["parameters"] = params
            model["csv_file"] = run["csv_file"] = args.csv_file
            model["model_name"] = run["model_name"] = 'ae_classifier_holdout'
            model["n_meta"] = run["n_meta"] = args.n_meta
            model["n_emb"] = run["n_emb"] = args.embeddings_meta
            model["groupkfold"] = run["groupkfold"] = args.groupkfold
            model["embeddings_meta"] = run["embeddings_meta"] = args.embeddings_meta
            model["foldername"] = run["foldername"] = self.foldername
            model["use_mapping"] = run["use_mapping"] = args.use_mapping
            model["dataset_name"] = run["dataset_name"] = args.dataset
            model["n_agg"] = run["n_agg"] = args.n_agg
        else:
            model = None
            run = None

        if self.log_mlflow:
            mlflow.set_experiment(
                self.args.exp_id,
            )
            # try:
            mlflow.start_run()
            # except:
            #     mlflow.end_run()
            #     mlflow.start_run()
            mlflow.log_params({
                "inputs_type": args.csv_file.split(".csv")[0],
                "best_unique": args.best_features_file.split(".tsv")[0],
                "tied_weights": args.tied_weights,
                "random_recs": args.random_recs,
                "train_after_warmup": args.train_after_warmup,
                "dloss": args.dloss,
                "predict_tests": args.predict_tests,
                "variational": args.variational,
                "zinb": args.zinb,
                "threshold": args.threshold,
                "rec_loss_type": args.rec_loss,
                "bad_batches": args.bad_batches,
                "remove_zeros": args.remove_zeros,
                "parameters": params,
                "scaler": params['scaler'],
                "csv_file": args.csv_file,
                "model_name": args.model_name,
                "n_meta": args.n_meta,
                "n_emb": args.embeddings_meta,
                "groupkfold": args.groupkfold,
                "foldername": self.foldername,
                "use_mapping": args.use_mapping,
                "dataset_name": args.dataset,
                "n_agg": args.n_agg,
            })
        else:
            model = None
            run = None
        seed = 0
        combinations = []
        h = 0
        best_closses = []
        best_mccs = []
        while h < self.args.n_repeats:
            print(f'Rep: {h}')
            epoch = 0
            self.best_loss = np.inf
            self.best_closs = np.inf
            self.best_dom_loss = np.inf
            self.best_dom_acc = np.inf
            self.best_acc = 0
            self.best_mcc = -1
            self.warmup_counter = 0
            self.warmup_b_counter = 0
            self.warmup_disc_b = False
            # best_acc = 0
            # best_mcc = -1
            # warmup_counter = 0
            # warmup_b_counter = 0
            if self.args.warmup > 0:
                warmup = True
            else:
                warmup = False
            if self.args.dataset == 'alzheimer':
                self.data, self.unique_labels, self.unique_batches = get_alzheimer(self.path, args, seed=seed)
                self.pools = True
            elif self.args.dataset in ['amide', 'adenocarcinoma']:
                self.data, self.unique_labels, self.unique_batches = get_amide(self.path, args, seed=seed)
                self.pools = True

            elif self.args.dataset == 'mice':
                # This seed split the data to have n_samples in train: 96, valid:52, test: 23
                self.data, self.unique_labels, self.unique_batches = get_mice(self.path, args, seed=seed)
            else:
                self.data, self.unique_labels, self.unique_batches = get_data(self.path, args, seed=seed)
                self.pools = self.args.pool

            combination = list(np.concatenate((np.unique(self.data['batches']['train']),
                                               np.unique(self.data['batches']['valid']),
                                               np.unique(self.data['batches']['test']))))
            seed += 1
            if combination not in combinations:
                combinations += [combination]
                h += 1
            else:
                continue
            # print(combinations)
            self.columns = self.data['inputs']['all'].columns
            self.make_samples_weights()
            # event_acc is used to verify if the hparams have already been tested. If they were,
            # the best classification loss is retrieved and we go to the next trial
            event_acc = EventAccumulator(hparams_filepath)
            event_acc.Reload()
            if len(event_acc.Tags()['tensors']) > 2 and self.load_tb:
                # try:
                #     best_acc = get_best_acc_from_tb(event_acc)
                # except:
                pass
            else:
                # If thres > 0, features that are 0 for a proportion of samples smaller than thres are removed
                # data = self.keep_good_features(thres)

                # Transform the data with the chosen scaler
                data = copy.deepcopy(self.data)
                data, self.scaler = scale_data(scale, data, self.args.device)

                # feature_selection = get_feature_selection_method('mutual_info_classif')
                # mi = feature_selection(data['inputs']['train'], data['cats']['train'])
                for g in list(data['inputs'].keys()):
                    data['inputs'][g] = data['inputs'][g].round(4)
                # Gets all the pytorch dataloaders to train the models
                if self.pools:
                    loaders = get_loaders(data, self.args.random_recs, self.samples_weights, self.args.dloss, None,
                                        None, bs=64)
                else:
                    loaders = get_loaders_no_pool(data, self.args.random_recs, self.samples_weights, self.args.dloss,
                                                  None, None, bs=self.args.bs)

                ae = AutoEncoder(data['inputs']['all'].shape[1],
                                 n_batches=self.n_batches,
                                 nb_classes=self.n_cats,
                                 mapper=self.args.use_mapping,
                                 layer1=layer1,
                                 layer2=layer2,
                                 n_layers=self.args.n_layers,
                                 n_meta=self.args.n_meta,
                                 n_emb=self.args.embeddings_meta,
                                 dropout=dropout,
                                 variational=self.args.variational, conditional=False,
                                 zinb=self.args.zinb, add_noise=0, tied_weights=self.args.tied_weights,
                                 use_gnn=0,  # TODO to remove
                                 device=self.args.device).to(self.args.device)
                ae.mapper.to(self.args.device)
                ae.dec.to(self.args.device)
                # if self.args.embeddings_meta > 0:
                #     n_meta = self.n_meta
                shap_ae = SHAPAutoEncoder(data['inputs']['all'].shape[1],
                                          n_batches=self.n_batches,
                                          nb_classes=self.n_cats,
                                          mapper=self.args.use_mapping,
                                          layer1=layer1,
                                          layer2=layer2,
                                          n_layers=self.args.n_layers,
                                          n_meta=self.args.n_meta,
                                          n_emb=self.args.embeddings_meta,
                                          dropout=dropout,
                                          variational=self.args.variational, conditional=False,
                                          zinb=self.args.zinb, add_noise=0, tied_weights=self.args.tied_weights,
                                          use_gnn=0,  # TODO remove this
                                          device=self.args.device).to(self.args.device)
                shap_ae.mapper.to(self.args.device)
                shap_ae.dec.to(self.args.device)
                loggers['logger_cm'] = SummaryWriter(f'{self.complete_log_path}/cm')
                loggers['logger'] = SummaryWriter(f'{self.complete_log_path}/traces')
                sceloss, celoss, mseloss, triplet_loss = self.get_losses(scale, smooth, margin, args.dloss)

                optimizer_ae = get_optimizer(ae, lr, wd, optimizer_type)

                # Used only if bdisc==1
                optimizer_b = get_optimizer(ae.dann_discriminator, 1e-2, 0, optimizer_type)

                self.hparams_names = [x.name for x in linsvc_space]
                if self.log_inputs and not self.logged_inputs:
                    data['inputs']['all'].to_csv(
                        f'{self.complete_log_path}/{self.args.berm}_inputs.csv')
                    if self.log_neptune:
                        run[f"inputs.csv"].track_files(f'{self.complete_log_path}/{self.args.berm}_inputs.csv')
                    log_input_ordination(loggers['logger'], data, self.scaler, epoch)
                    if self.pools:
                        metrics = log_pool_metrics(data['inputs'], data['batches'], data['labels'], loggers, epoch,
                                                   metrics, 'inputs')
                    self.logged_inputs = True

                values, best_values, _, best_traces = get_empty_dicts()

                early_stop_counter = 0
                best_vals = values
                if h > 1:  # or warmup_counter == 100:
                    ae.load_state_dict(torch.load(f'{self.complete_log_path}/warmup.pth'))
                    print(f"\n\nNO WARMUP\n\n")
                if h == 1:
                    for epoch in range(0, self.args.warmup):
                        self.warmup_loop(optimizer_ae, ae, celoss, loaders['all'], triplet_loss, mseloss,
                                         self.best_loss, True, epoch,
                                         optimizer_b, values, loggers, loaders, run, self.args.use_mapping)

                for epoch in range(0, self.args.n_epochs):
                    lists, traces = get_empty_traces()
                    if self.args.train_after_warmup:
                        self.warmup_loop(optimizer_ae, ae, celoss, loaders['all'], triplet_loss, mseloss,
                                         self.best_loss, False,
                                         epoch, optimizer_b, values, loggers, loaders, run, self.args.use_mapping)

                    else:
                        ae = self.freeze_clayers(ae)
                    closs, _, _ = self.loop('train', optimizer_ae, ae, sceloss,
                                            loaders['train'], lists, traces, nu=nu)

                    if torch.isnan(closs):
                        if self.log_mlflow:
                            mlflow.log_param('finished', 0)
                            mlflow.end_run()
                        return self.best_loss
                    ae.eval()

                    # Below is the loop for all sets
                    with torch.no_grad():
                        for group in list(data['inputs'].keys()):
                            if group in ['all', 'all_pool']:
                                continue
                            closs, lists, traces = self.loop(group, optimizer_ae, ae, sceloss,
                                                             loaders[group], lists, traces, nu=0)

                    traces = self.get_mccs(lists, traces)
                    values = log_traces(traces, values)
                    if self.log_tb:
                        try:
                            add_to_logger(values, loggers['logger'], epoch)
                        except:
                            print("Problem with add_to_logger!")
                    if self.log_neptune:
                        add_to_neptune(values, run, epoch)
                    if self.log_mlflow:
                        add_to_mlflow(values, epoch)
                    if np.mean(values['valid']['mcc'][-self.args.n_agg:]) > self.best_mcc and len(
                            values['valid']['mcc']) > self.args.n_agg:
                        print(f"Best Classification Mcc Epoch {epoch}, "
                              f"Acc: {values['test']['acc'][-1]}"
                              f"Mcc: {values['test']['mcc'][-1]}"
                              f"Classification train loss: {values['train']['closs'][-1]},"
                              f" valid loss: {values['valid']['closs'][-1]},"
                              f" test loss: {values['test']['closs'][-1]}")
                        self.best_mcc = np.mean(values['valid']['mcc'][-self.args.n_agg:])
                        torch.save(ae.state_dict(), f'{self.complete_log_path}/model_{h}.pth')
                        best_values = get_best_values(values.copy(), ae_only=False, n_agg=self.args.n_agg)
                        best_vals = values.copy()
                        best_vals['rec_loss'] = self.best_loss
                        best_vals['dom_loss'] = self.best_dom_loss
                        best_vals['dom_acc'] = self.best_dom_acc
                        early_stop_counter = 0

                    if values['valid']['acc'][-1] > self.best_acc:
                        print(f"Best Classification Acc Epoch {epoch}, "
                              f"Acc: {values['test']['acc'][-1]}"
                              f"Mcc: {values['test']['mcc'][-1]}"
                              f"Classification train loss: {values['train']['closs'][-1]},"
                              f" valid loss: {values['valid']['closs'][-1]},"
                              f" test loss: {values['test']['closs'][-1]}")

                        self.best_acc = values['valid']['acc'][-1]
                        early_stop_counter = 0

                    if values['valid']['closs'][-1] < self.best_closs:
                        print(f"Best Classification Loss Epoch {epoch}, "
                              f"Acc: {values['test']['acc'][-1]} "
                              f"Mcc: {values['test']['mcc'][-1]} "
                              f"Classification train loss: {values['train']['closs'][-1]}, "
                              f"valid loss: {values['valid']['closs'][-1]}, "
                              f"test loss: {values['test']['closs'][-1]}")
                        self.best_closs = values['valid']['closs'][-1]
                        early_stop_counter = 0
                    else:
                        # if epoch > self.warmup:
                        early_stop_counter += 1

                    if self.args.predict_tests and (epoch % 10 == 0):
                        loaders = get_loaders(self.data, data, self.args.random_recs, self.args.triplet_dloss, ae,
                                              ae.classifier, bs=self.args.bs)

                best_mccs += [self.best_mcc]

                best_lists, traces = get_empty_traces()
                # Loading best model that was saved during training
                ae.load_state_dict(torch.load(f'{self.complete_log_path}/model_{h}.pth'))
                # Need another model because the other cant be use to get shap values
                shap_ae.load_state_dict(torch.load(f'{self.complete_log_path}/model_{h}.pth'))
                # ae.load_state_dict(sd)
                ae.eval()
                shap_ae.eval()
                with torch.no_grad():
                    for group in list(data['inputs'].keys()):
                        # if group in ['all', 'all_pool']:
                        #     continue
                        closs, best_lists, traces = self.loop(group, None, ae, sceloss,
                                                              loaders[group], best_lists, traces, nu=0, mapping=False)
                if self.log_neptune:
                    model["model"].upload(f'{self.complete_log_path}/model_{h}.pth')
                    model["validation/closs"].log(self.best_closs)
                best_closses += [self.best_closs]
                # logs things in the background. This could be problematic if the logging takes more time than each iteration of repetitive holdout
                # daemon = Thread(target=self.log_rep, daemon=True, name='Monitor',
                #                 args=[best_lists, best_vals, best_values, traces, model, metrics, run, cm_logger, ae,
                #                       shap_ae, h,
                #                       epoch])
                # daemon.start()
                self.log_rep(best_lists, best_vals, best_values, traces, model, metrics, run, loggers, ae,
                             shap_ae, h, epoch)
                del ae, shap_ae

        # Logging every model is taking too much resources and it makes it quite complicated to get information when
        # Too many runs have been made. This will make the notebook so much easier to work with
        if np.mean(best_mccs) > self.best_mcc:
            try:
                if os.path.exists(
                        f'logs/best_models/ae_classifier_holdout/{self.args.dataset}/{self.args.dloss}_vae{self.args.variational}'):
                    shutil.rmtree(
                        f'logs/best_models/ae_classifier_holdout/{self.args.dataset}/{self.args.dloss}_vae{self.args.variational}',
                        ignore_errors=True)
                # os.makedirs(f'logs/best_models/ae_classifier_holdout/{self.args.dloss}_vae{self.args.variational}', exist_ok=True)
                shutil.copytree(f'{self.complete_log_path}',
                                f'logs/best_models/ae_classifier_holdout/{self.args.dataset}/{self.args.dloss}_vae{self.args.variational}')
                # print("File copied successfully.")

            # If source and destination are same
            except shutil.SameFileError:
                # print("Source and destination represents the same file.")
                pass
            self.best_mcc = np.mean(best_mccs)

        # Logs confusion matrices in the background. Also runs RandomForestClassifier on encoded and reconstructed
        # representations. This should be shorter than the actual calculation of the model above in the function,
        # otherwise the number of threads will keep increasing.
        # daemon = Thread(target=self.logging, daemon=True, name='Monitor', args=[run, cm_logger])
        # daemon.start()
        if self.log_mlflow:
            mlflow.log_param('finished', 1)
        self.logging(run, loggers['cm_logger'])

        if not self.keep_models:
            # shutil.rmtree(f'{self.complete_log_path}/traces', ignore_errors=True)
            # shutil.rmtree(f'{self.complete_log_path}/cm', ignore_errors=True)
            # shutil.rmtree(f'{self.complete_log_path}/hp', ignore_errors=True)
            shutil.rmtree(f'{self.complete_log_path}', ignore_errors=True)
        print('Duration: {}'.format(datetime.now() - start_time))
        best_closs = np.mean(best_closses)
        if best_closs < self.best_closs:
            self.best_closs = best_closs
            print("Best closs!")

        # It should not be necessary. To remove once certain the "Too many files open" error is no longer a problem
        plt.close('all')

        return self.best_mcc

    def log_rep(self, best_lists, best_vals, best_values, traces, model, metrics, run, loggers, ae, shap_ae, h,
                epoch):
        # if run is None:
        #     return None
        best_traces = self.get_mccs(best_lists, traces)

        self.log_predictions(best_lists, run, h)

        if self.log_metrics:
            if self.log_tb:
                try:
                    # logger, lists, values, model, unique_labels, mlops, epoch, metrics, n_meta_emb=0, device='cuda'
                    metrics = log_metrics(loggers['logger'], best_lists, best_vals, ae,
                                          np.unique(np.concatenate(best_lists['train']['labels'])),
                                          np.unique(self.data['batches']), epoch, mlops="tensorboard",
                                          metrics=metrics, n_meta_emb=self.args.embeddings_meta,
                                          device=self.args.device)
                except BrokenPipeError:
                    print("\n\n\nProblem with logging stuff!\n\n\n")
            if self.log_neptune:
                try:
                    metrics = log_metrics(run, best_lists, best_vals, ae,
                                          np.unique(np.concatenate(best_lists['train']['labels'])),
                                          np.unique(self.data['batches']), epoch=epoch, mlops="neptune",
                                          metrics=metrics, n_meta_emb=self.args.embeddings_meta,
                                          device=self.args.device)
                except BrokenPipeError:
                    print("\n\n\nProblem with logging stuff!\n\n\n")
            if self.log_mlflow:
                try:
                    metrics = log_metrics(None, best_lists, best_vals, ae,
                                          np.unique(np.concatenate(best_lists['train']['labels'])),
                                          np.unique(self.data['batches']), epoch, mlops="mlflow",
                                          metrics=metrics, n_meta_emb=self.args.embeddings_meta,
                                          device=self.args.device)
                except BrokenPipeError:
                    print("\n\n\nProblem with logging stuff!\n\n\n")

        if self.log_metrics and self.pools:
            try:
                if self.log_neptune:
                    enc_data = make_data(best_lists, 'encoded_values')
                    metrics = log_pool_metrics(enc_data['inputs'], enc_data['batches'], enc_data['labels'],
                                               self.unique_unique_labels, run, epoch, metrics, 'enc', 'neptune')
                    rec_data = make_data(best_lists, 'rec_values')
                    metrics = log_pool_metrics(rec_data['inputs'], rec_data['batches'], rec_data['labels'],
                                               self.unique_unique_labels, run, epoch, metrics, 'rec', 'neptune')
                if self.log_mlflow:
                    enc_data = make_data(best_lists, 'encoded_values')
                    metrics = log_pool_metrics(enc_data['inputs'], enc_data['batches'], enc_data['labels'],
                                               self.unique_unique_labels, run, epoch, metrics, 'enc', 'mlflow')
                    rec_data = make_data(best_lists, 'rec_values')
                    metrics = log_pool_metrics(rec_data['inputs'], rec_data['batches'], rec_data['labels'],
                                               self.unique_unique_labels, run, epoch, metrics, 'rec', 'mlflow')
                if self.log_tb:
                    enc_data = make_data(best_lists, 'encoded_values')
                    metrics = log_pool_metrics(enc_data['inputs'], enc_data['batches'], enc_data['labels'],
                                               self.unique_unique_labels, loggers['logger'], epoch, metrics, 'enc',
                                               'tensorboard')
                    rec_data = make_data(best_lists, 'rec_values')
                    metrics = log_pool_metrics(rec_data['inputs'], rec_data['batches'], rec_data['labels'],
                                               self.unique_unique_labels, loggers['logger'], epoch, metrics, 'rec',
                                               'tensorboard')

            except BrokenPipeError:
                print("\n\n\nProblem with logging stuff!\n\n\n")

        loggers['cm_logger'].add(best_lists)
        if h == 1:
            if self.log_plots:
                if self.log_tb:
                    # TODO Add log_shap
                    loggers.add(loggers['logger_cm'], epoch, best_lists,
                               self.unique_labels, best_traces, 'tensorboard')
                    log_plots(loggers['logger_cm'], best_lists, 'tensorboard', epoch)
                    log_shap(loggers['logger_cm'], shap_ae, best_lists, self.columns, self.args.n_meta, 'mlflow',
                             self.complete_log_path,
                             self.args.device)
                if self.log_neptune:
                    log_shap(run, shap_ae, best_lists, self.columns, self.args.embeddings_meta, 'neptune',
                             self.complete_log_path,
                             self.args.device)
                    log_plots(run, best_lists, 'neptune', epoch)
                if self.log_mlflow:
                    log_shap(None, shap_ae, best_lists, self.columns, self.args.embeddings_meta, 'mlflow',
                             self.complete_log_path,
                             self.args.device)
                    log_plots(None, best_lists, 'mlflow', epoch)

        columns = self.data['inputs']['all'].columns
        if self.args.n_meta == 2:
            columns += ['gender', 'age']

        rec_data, enc_data = to_csv(best_lists, self.complete_log_path, self.data['inputs']['all'].columns)

        # from skopt import gp_minimize
        # train = Train2("Reconstruction", sklearn.svm.LinearSVC, rec_data, self.hparams_names,
        #                self.complete_log_path,
        #                args, loggers['logger_cm'], ovr=0, binary=False, mlops='mlflow')
        # res = gp_minimize(train.train, linsvc_space, n_calls=20, random_state=1)
        # train = Train2("Encoded", sklearn.svm.LinearSVC, enc_data, self.hparams_names, self.complete_log_path,
        #                args, loggers['logger_cm'], ovr=0, binary=False, mlops='mlflow')
        # res = gp_minimize(train.train, linsvc_space, n_calls=20, random_state=1)

        if self.log_neptune:
            run["recs"].track_files(f'{self.complete_log_path}/recs.csv')
            run["encs"].track_files(f'{self.complete_log_path}/encs.csv')
            # if self.log_mlflow:
            #     mlflow.log_artifact(f'{self.complete_log_path}/recs.csv')
            #     mlflow.log_artifact(f'{self.complete_log_path}/encs.csv')
            # try:
            #     enc_data = make_data(best_lists, 'encoded_values')
            #     if self.log_tb and self.pools:
            #         metrics = log_pool_metrics(enc_data['inputs'], enc_data['batches'], enc_data['labels'], self.unique_labels, ['logger_cm'], epoch,
            #                                    metrics,
            #                                    'enc', 'tensorboard')
            #     if self.log_neptune and self.pools:
            #         metrics = log_pool_metrics(enc_data['inputs'], enc_data['batches'], enc_data['labels'], run, epoch, metrics, 'enc',
            #                                    'neptune')
            #         metrics = log_pool_metrics(enc_data['inputs'], enc_data['batches'], enc_data['labels'], model, epoch, metrics,
            #                                    'enc',
            #                                    'neptune')
            #     if self.log_mlflow and self.pools:
            #         metrics = log_pool_metrics(enc_data['inputs'], enc_data['batches'], enc_data['labels'], run, epoch, metrics, 'enc',
            #                                    'mlflow')
            #         metrics = log_pool_metrics(enc_data['inputs'], enc_data['batches'], enc_data['labels'], model, epoch, metrics,
            #                                    'enc',
            #                                    'mlflow')
            #
            #     if self.pools:
            #         rec_data = make_data(best_lists, 'rec_values')
            #         if self.log_tb:
            #             metrics = log_pool_metrics(rec_data['inputs'], rec_data['batches'], rec_data['labels'], loggers['logger_cm'], epoch,
            #                                        metrics,
            #                                        'rec', 'tensorboard')
            #         if self.log_neptune:
            #             metrics = log_pool_metrics(rec_data['inputs'], rec_data['batches'], rec_data['labels'], run, epoch, metrics, 'rec',
            #                                        'neptune')
            #             metrics = log_pool_metrics(rec_data['inputs'], rec_data['batches'], rec_data['labels'], model, epoch, metrics,
            #                                        'rec',
            #                                        'neptune')
            #         if self.log_mlflow:
            #             metrics = log_pool_metrics(rec_data['inputs'], rec_data['batches'], rec_data['labels'], run, epoch, metrics, 'rec',
            #                                        'mlflow')
            #             metrics = log_pool_metrics(rec_data['inputs'], rec_data['batches'], rec_data['labels'], model, epoch, metrics,
            #                                        'rec',
            #                                        'mlflow')

            ### TODO once log_tb and log_neptune are repaired, remove the block below

            # best_values['enc b_euclidean/tot_eucl'] = metrics['pool_metrics_enc']['all'][
            #     '[b_euclidean/tot_eucl]']
            # best_values['rec b_euclidean/tot_eucl'] = metrics['pool_metrics_rec']['all'][
            #     '[b_euclidean/tot_eucl]']
            # best_values['enc qc_aPCC'] = metrics['pool_metrics_enc']['all']['qc_aPCC']
            # best_values['rec qc_aPCC'] = metrics['pool_metrics_rec']['all']['qc_aPCC']
            # best_values['enc qc_dist/tot_eucl'] = metrics['pool_metrics_enc']['all']['[qc_dist/tot_eucl]']
            # best_values['rec qc_dist/tot_eucl'] = metrics['pool_metrics_rec']['all']['[qc_dist/tot_eucl]']
            # best_values['enc batch_entropy'] = metrics['pool_metrics_enc']['test']['qc_aPCC']
            # best_values['rec batch_entropy'] = metrics['pool_metrics_rec']['test']['qc_aPCC']

            ### TODO once log_tb and log_neptune are repaired, remove the block above

            best_values['pool_metrics'] = {}
            if self.log_metrics:
                best_values['batches'] = metrics['batches']
                best_values['pool_metrics']['enc'] = metrics['pool_metrics_enc']
                best_values['pool_metrics']['rec'] = metrics['pool_metrics_rec']
        # TODO change logging with tensorboard and neptune. The previous
        if self.log_tb:
            loggers['tb_logging'].logging(best_values, metrics)
        if self.log_neptune:
            log_neptune(run, best_values)
        if self.log_mlflow:
            log_mlflow(best_values, h)

        # except BrokenPipeError:
        #     print("\n\n\nProblem with logging stuff!\n\n\n")

    def logging(self, run, cm_logger):

        if self.log_neptune:
            cm_logger.plot(run, 0, self.unique_labels, 'neptune')
            # cm_logger.get_rf_results(run, self.args)
            run.stop()
        if self.log_mlflow:
            cm_logger.plot(None, 0, self.unique_labels, 'mlflow')
            # cm_logger.get_rf_results(run, self.args)
            mlflow.end_run()
        # cm_logger.close()
        # logger.close()

    def log_predictions(self, best_lists, run, step):
        cats, labels, preds, scores, names = [{'train': [], 'valid': [], 'test': []} for _ in range(5)]
        for group in ['train', 'valid', 'test']:
            cats[group] = np.concatenate(best_lists[group]['cats'])
            labels[group] = np.concatenate(best_lists[group]['labels'])
            scores[group] = torch.softmax(torch.Tensor(np.concatenate(best_lists[group]['preds'])), 1)
            preds[group] = scores[group].argmax(1)
            names[group] = np.concatenate(best_lists[group]['names'])
            pd.DataFrame(np.concatenate((labels[group].reshape(-1, 1), scores[group],
                                         np.array([self.unique_labels[x] for x in preds[group]]).reshape(-1, 1),
                                         names[group].reshape(-1, 1)), 1)).to_csv(
                f'{self.complete_log_path}/{group}_predictions.csv')
            if self.log_neptune:
                run[f"{group}_predictions"].track_files(f'{self.complete_log_path}/{group}_predictions.csv')
                run[f'{group}_AUC'] = metrics.roc_auc_score(y_true=cats[group], y_score=scores[group],
                                                            multi_class='ovr')
            if self.log_mlflow:
                mlflow.log_metric(f'{group}_AUC',
                                  metrics.roc_auc_score(y_true=cats[group], y_score=scores[group], multi_class='ovr'),
                                  step=step)

    def loop(self, group, optimizer_ae, ae, celoss, loader, lists, traces, nu=1, mapping=True):
        """

        Args:
            group: Which set? Train, valid or test
            optimizer_ae: Object that contains the optimizer for the autoencoder
            ae: AutoEncoder (pytorch model, inherits nn.Module)
            celoss: torch.nn.CrossEntropyLoss instance
            triplet_loss: torch.nn.TripletMarginLoss instance
            loader: torch.utils.data.DataLoader
            lists: List keeping informations on the current run
            traces: List keeping scores on the current run
            nu: hyperparameter controlling the importance of the classification loss

        Returns:

        """
        # If group is train and nu = 0, then it is not training. valid can also have sampling = True
        if group in ['train', 'valid'] and nu != 0:
            sampling = True
        else:
            sampling = False
        classif_loss = None
        for i, batch in enumerate(loader):
            if group in ['train'] and nu != 0:
                optimizer_ae.zero_grad()
            data, meta_inputs, names, labels, domain, to_rec, not_to_rec, pos_batch_sample, \
                neg_batch_sample, meta_pos_batch_sample, meta_neg_batch_sample, set = batch
            # data[torch.isnan(data)] = 0
            data = data.to(self.args.device).float()
            meta_inputs = meta_inputs.to(self.args.device).float()
            to_rec = to_rec.to(self.args.device).float()

            # If n_meta > 0, meta data added to inputs
            if self.args.n_meta > 0:
                data = torch.cat((data, meta_inputs), 1)
                to_rec = torch.cat((to_rec, meta_inputs), 1)
            not_to_rec = not_to_rec.to(self.args.device).float()
            enc, rec, _, kld = ae(data, to_rec, domain, sampling=sampling, mapping=mapping)
            rec = rec['mean']

            # If embedding_meta > 0, meta data added to embeddings
            if self.args.embeddings_meta:
                preds = ae.classifier(torch.cat((enc, meta_inputs), 1))
            else:
                preds = ae.classifier(enc)

            domain_preds = ae.dann_discriminator(enc)
            try:
                cats = to_categorical(labels.long(), self.n_cats).to(self.args.device).float()
                classif_loss = celoss(preds, cats)
            except:
                cats = torch.Tensor([self.n_cats + 1 for _ in labels])
                classif_loss = torch.Tensor([0])

            # if self.args.triplet_loss and not self.args.predict_tests:
            #     rec_loss = triplet_loss(rec, to_rec, not_to_rec)
            # else:
            #     rec_loss = torch.zeros(1).to(device)[0]

            if not self.args.zinb:
                if isinstance(rec, list):
                    rec = rec[-1]
                if isinstance(to_rec, list):
                    to_rec = to_rec[-1]
            lists[group]['set'] += [np.array([group for _ in range(len(domain))])]
            lists[group]['domains'] += [np.array([self.unique_batches[d] for d in domain.detach().cpu().numpy()])]
            lists[group]['domain_preds'] += [domain_preds.detach().cpu().numpy()]
            lists[group]['preds'] += [preds.detach().cpu().numpy()]
            lists[group]['classes'] += [labels.detach().cpu().numpy()]
            # lists[group]['encoded_values'] += [enc.view(enc.shape[0], -1).detach().cpu().numpy()]
            lists[group]['names'] += [names]
            lists[group]['cats'] += [cats.detach().cpu().numpy()]
            lists[group]['gender'] += [data.detach().cpu().numpy()[:, -1]]
            lists[group]['age'] += [data.detach().cpu().numpy()[:, -2]]
            lists[group]['atn'] += [str(x) for x in data.detach().cpu().numpy()[:, -5:-2]]
            lists[group]['inputs'] += [data.view(rec.shape[0], -1).detach().cpu().numpy()]
            lists[group]['encoded_values'] += [enc.detach().cpu().numpy()]
            lists[group]['rec_values'] += [rec.detach().cpu().numpy()]
            try:
                lists[group]['labels'] += [np.array(
                    [self.unique_labels[x] for x in labels.detach().cpu().numpy()])]
            except:
                pass
            traces[group]['acc'] += [np.mean([0 if pred != dom else 1 for pred, dom in
                                              zip(preds.detach().cpu().numpy().argmax(1),
                                                  labels.detach().cpu().numpy())])]
            traces[group]['top3'] += [np.mean([1 if label.item() in pred.tolist()[::-1][:3] else 0 for pred, label in
                                               zip(preds.argsort(1), labels)])]

            traces[group]['closs'] += [classif_loss.item()]
            traces[group]['mcc'] += [np.round(
                MCC(labels.detach().cpu().numpy(), preds.detach().cpu().numpy().argmax(1)), 3)
            ]
            if group in ['train'] and nu != 0:
                # w = np.mean([1/self.class_weights[x] for x in lists[group]['labels'][-1]])
                w = 1
                total_loss = w * nu * classif_loss
                # if self.args.train_after_warmup:
                #     total_loss += rec_loss
                try:
                    total_loss.backward()
                except:
                    pass
                nn.utils.clip_grad_norm_(ae.classifier.parameters(), max_norm=1)
                optimizer_ae.step()

        return classif_loss, lists, traces

    def warmup_loop(self, optimizer_ae, ae, celoss, loader, triplet_loss, mseloss, best_loss, warmup, epoch,
                    optimizer_b, values, loggers, loaders, run, mapping=True):
        lists, traces = get_empty_traces()
        ae.train()

        iterator = enumerate(loader)

        # If option train_after_warmup=1, then this loop is only for preprocessing
        for i, all_batch in iterator:
            optimizer_ae.zero_grad()
            inputs, meta_inputs, names, labels, domain, to_rec, not_to_rec, pos_batch_sample, \
                neg_batch_sample, meta_pos_batch_sample, meta_neg_batch_sample, _ = all_batch
            inputs = inputs.to(self.args.device).float()
            meta_inputs = meta_inputs.to(self.args.device).float()
            to_rec = to_rec.to(self.args.device).float()
            if self.args.n_meta > 0:
                inputs = torch.cat((inputs, meta_inputs), 1)
                to_rec = torch.cat((to_rec, meta_inputs), 1)

            enc, rec, zinb_loss, kld = ae(inputs, to_rec, domain, sampling=True, mapping=mapping)
            rec = rec['mean']
            zinb_loss = zinb_loss.to(self.args.device)
            reverse = ReverseLayerF.apply(enc, 1)
            if args.dloss == 'DANN':
                domain_preds = ae.dann_discriminator(reverse)
            else:
                domain_preds = ae.dann_discriminator(enc)
            if args.dloss not in ['revTriplet', 'inverseTriplet']:
                dloss, domain = self.get_dloss(celoss, domain, domain_preds)
            elif args.dloss == 'revTriplet':
                pos_batch_sample = pos_batch_sample.to(self.args.device).float()
                neg_batch_sample = neg_batch_sample.to(self.args.device).float()
                meta_pos_batch_sample = meta_pos_batch_sample.to(self.args.device).float()
                meta_neg_batch_sample = meta_neg_batch_sample.to(self.args.device).float()
                if self.args.n_meta > 0:
                    pos_batch_sample = torch.cat((pos_batch_sample, meta_pos_batch_sample), 1)
                    neg_batch_sample = torch.cat((neg_batch_sample, meta_neg_batch_sample), 1)
                pos_enc, _, _, _ = ae(pos_batch_sample, pos_batch_sample, domain, sampling=True)
                neg_enc, _, _, _ = ae(neg_batch_sample, neg_batch_sample, domain, sampling=True)
                dloss = triplet_loss(reverse,
                                     ReverseLayerF.apply(pos_enc, 1),
                                     ReverseLayerF.apply(neg_enc, 1)
                                     )
            elif args.dloss == 'inverseTriplet':
                pos_batch_sample, neg_batch_sample = neg_batch_sample.to(self.args.device).float(), pos_batch_sample.to(
                    self.args.device).float()
                meta_pos_batch_sample, meta_neg_batch_sample = meta_neg_batch_sample.to(
                    self.args.device).float(), meta_pos_batch_sample.to(self.args.device).float()
                if self.args.n_meta > 0:
                    pos_batch_sample = torch.cat((pos_batch_sample, meta_pos_batch_sample), 1)
                    neg_batch_sample = torch.cat((neg_batch_sample, meta_neg_batch_sample), 1)
                pos_enc, _, _, _ = ae(pos_batch_sample, pos_batch_sample, domain, sampling=True)
                neg_enc, _, _, _ = ae(neg_batch_sample, neg_batch_sample, domain, sampling=True)
                dloss = triplet_loss(enc, pos_enc, neg_enc)
                # domain = domain.argmax(1)

            if torch.isnan(enc[0][0]):
                if self.log_mlflow:
                    mlflow.log_param('finished', 0)
                    mlflow.end_run()
                return None
            # rec_loss = triplet_loss(rec, to_rec, not_to_rec)
            if isinstance(rec, list):
                rec = rec[-1]
            if isinstance(to_rec, list):
                to_rec = to_rec[-1]
            if self.args.scaler == 'binarize':
                rec = torch.sigmoid(rec)
            rec_loss = mseloss(rec, to_rec)
            # else:
            #     rec_loss = zinb_loss
            traces['rec_loss'] += [rec_loss.item()]
            traces['dom_loss'] += [dloss.item()]
            traces['dom_acc'] += [np.mean([0 if pred != dom else 1 for pred, dom in
                                           zip(domain_preds.detach().cpu().numpy().argmax(1),
                                               domain.detach().cpu().numpy())])]
            # lists['all']['set'] += [np.array([group for _ in range(len(domain))])]
            lists['all']['domains'] += [np.array(
                [self.unique_batches[d] for d in domain.detach().cpu().numpy()])]
            lists['all']['domain_preds'] += [domain_preds.detach().cpu().numpy()]
            # lists[group]['preds'] += [preds.detach().cpu().numpy()]
            lists['all']['classes'] += [labels.detach().cpu().numpy()]
            lists['all']['encoded_values'] += [
                enc.detach().cpu().numpy()]
            lists['all']['rec_values'] += [
                rec.detach().cpu().numpy()]
            lists['all']['names'] += [names]
            lists['all']['gender'] += [meta_inputs.detach().cpu().numpy()[:, -1]]
            lists['all']['age'] += [meta_inputs.detach().cpu().numpy()[:, -2]]
            lists['all']['atn'] += [str(x) for x in
                                    meta_inputs.detach().cpu().numpy()[:, -5:-2]]
            lists['all']['inputs'] += [to_rec]
            try:
                lists['all']['labels'] += [np.array(
                    [self.unique_labels[x] for x in labels.detach().cpu().numpy()])]
            except:
                pass
            (rec_loss + self.gamma * dloss + self.beta * kld.mean() + self.zeta * zinb_loss).backward()
            nn.utils.clip_grad_norm_(ae.parameters(), max_norm=1)
            optimizer_ae.step()

        if np.mean(traces['rec_loss']) < self.best_loss:
            # "Every counters go to 0 when a better reconstruction loss is reached"
            print(
                f"Best Loss Epoch {epoch}, Losses: {np.mean(traces['rec_loss'])}, "
                f"Domain Losses: {np.mean(traces['dom_loss'])}, "
                f"Domain Accuracy: {np.mean(traces['dom_acc'])}")
            self.best_loss = np.mean(traces['rec_loss'])
            self.dom_loss = np.mean(traces['dom_loss'])
            self.dom_acc = np.mean(traces['dom_acc'])
            if warmup:
                torch.save(ae.state_dict(), f'{self.complete_log_path}/warmup.pth')

        if (
                self.args.early_warmup_stop != 0 and self.warmup_counter == self.args.early_warmup_stop) and warmup:  # or warmup_counter == 100:
            # When the warnup counter gets to
            values = log_traces(traces, values)
            if self.args.early_warmup_stop != 0:
                try:
                    ae.load_state_dict(torch.load(f'{self.complete_log_path}/model.pth'))
                except:
                    pass
            print(f"\n\nWARMUP FINISHED (early stop). {epoch}\n\n")
            warmup = False
            self.warmup_disc_b = True

        if epoch == self.args.warmup and warmup:  # or warmup_counter == 100:
            # When the warnup counter gets to
            if self.args.early_warmup_stop != 0:
                try:
                    ae.load_state_dict(torch.load(f'{self.complete_log_path}/model.pth'))
                except:
                    pass
            print(f"\n\nWARMUP FINISHED. {epoch}\n\n")
            values = log_traces(traces, values)
            warmup = False
            self.warmup_disc_b = True

        if epoch < self.args.warmup and warmup:  # and np.mean(traces['rec_loss']) >= best_loss:
            values = log_traces(traces, values)
            self.warmup_counter += 1
            # best_values = get_best_values(traces, ae_only=True)
            # TODO change logging with tensorboard and neptune. The previous
            if self.log_tb:
                loggers['tb_logging'].logging(values, metrics)
            if self.log_neptune:
                log_neptune(run, values)
            if self.log_mlflow:
                add_to_mlflow(values, epoch)
        ae.train()

        # If training of the autoencoder is retricted to the warmup, (train_after_warmup=0),
        # all layers except the classification layers are frozen

        if self.args.bdisc:
            self.forward_discriminate(optimizer_b, ae, celoss, loaders['all'])
        if self.warmup_disc_b and self.warmup_b_counter < 0:
            self.warmup_b_counter += 1
        else:
            self.warmup_disc_b = False

    def forward_discriminate(self, optimizer_b, ae, celoss, loader):
        # Freezing the layers so the batch discriminator can get some knowledge independently
        # from the part where the autoencoder is trained. Only for DANN
        self.freeze_dlayers(ae)
        sampling = True
        for i, batch in enumerate(loader):
            optimizer_b.zero_grad()
            data, meta_inputs, names, labels, domain, to_rec, not_to_rec, pos_batch_sample, \
                neg_batch_sample, meta_pos_batch_sample, meta_neg_batch_sample, set = batch
            # data[torch.isnan(data)] = 0
            data = data.to(self.args.device).float()
            to_rec = to_rec.to(self.args.device).float()
            with torch.no_grad():
                enc, rec, _, kld = ae(data, to_rec, domain, sampling=sampling)
            with torch.enable_grad():
                domain_preds = ae.dann_discriminator(enc)
                bclassif_loss = celoss(domain_preds,
                                       to_categorical(domain.long(), self.n_batches).to(self.args.device).float())
                bclassif_loss.backward()
                optimizer_b.step()
        self.unfreeze_layers(ae)
        # return classif_loss, lists, traces

    def get_dloss(self, celoss, domain, domain_preds):
        """
        This function is used to get the domain loss
        Args:
            celoss: PyTorch CrossEntropyLoss instance object
            domain: one-hot encoded domain classes []
            domain_preds: Matrix containing the predicted domains []

        Returns:
            dloss: Domain loss
            domain: True domain (batch) values
        """
        if args.dloss in ['revTriplet', 'revDANN', 'DANN', 'inverseTriplet', 'normae']:
            domain = domain.to(self.args.device).long().to(self.args.device)
            dloss = celoss(domain_preds, domain)
        else:
            dloss = torch.zeros(1)[0].float().to(self.args.device)
        if args.dloss == 'normae':
            dloss = -dloss
        return dloss, domain

    def get_losses(self, scale, smooth, margin, dloss):
        """
        Getter for the losses.
        Args:
            scale: Scaler that was used, e.g. normalizer or binarize
            smooth: Parameter for label_smoothing
            margin: Parameter for the TripletMarginLoss

        Returns:
            sceloss: CrossEntropyLoss (with label smoothing)
            celoss: CrossEntropyLoss object (without label smoothing)
            mseloss: MSELoss object
            triplet_loss: TripletMarginLoss object
        """
        sceloss = nn.CrossEntropyLoss(label_smoothing=smooth)
        celoss = nn.CrossEntropyLoss()
        if self.args.rec_loss == 'mse':
            mseloss = nn.MSELoss()
        elif self.args.rec_loss == 'l1':
            mseloss = nn.L1Loss()
        if scale == "binarize":
            mseloss = nn.BCELoss()
        if dloss == 'revTriplet':
            triplet_loss = nn.TripletMarginLoss(margin, p=2, swap=True)
        else:
            triplet_loss = nn.TripletMarginLoss(0, p=2, swap=False)

        return sceloss, celoss, mseloss, triplet_loss

    def freeze_dlayers(self, ae):
        """
        Freeze all layers except the classifier
        Args:
            ae: AutoEncoder object. It inherits torch.nn.Module

        Returns:
            ae: The same AutoEncoder object, but with all frozen layers. Only the classifier layers are not frozen.

        """
        if not self.args.train_after_warmup:
            for param in ae.dec.parameters():
                param.requires_grad = False
            for param in ae.enc.parameters():
                param.requires_grad = False
            for param in ae.classifier.parameters():
                param.requires_grad = False
            for param in ae.dann_discriminator.parameters():
                param.requires_grad = True
        return ae

    def freeze_ae(self, ae):
        """
        Freeze all layers of ae
        Args:
            ae: AutoEncoder object. It inherits torch.nn.Module

        Returns:
            ae: The same AutoEncoder object, but with all frozen layers. Only the classifier layers are not frozen.

        """
        if not self.args.train_after_warmup:
            for param in ae.dec.parameters():
                param.requires_grad = True
            for param in ae.enc.parameters():
                param.requires_grad = True
            for param in ae.classifier.parameters():
                param.requires_grad = False
            for param in ae.dann_discriminator.parameters():
                param.requires_grad = False
        return ae

    def freeze_clayers(self, ae):
        """
        Freeze all layers except the classifier
        Args:
            ae: AutoEncoder object. It inherits torch.nn.Module

        Returns:
            ae: The same AutoEncoder object, but with all frozen layers. Only the classifier layers are not frozen.

        """
        if not self.args.train_after_warmup:
            ae.enc.eval()
            ae.dec.eval()
            for param in ae.dec.parameters():
                param.requires_grad = False
            for param in ae.enc.parameters():
                param.requires_grad = False
            for param in ae.classifier.parameters():
                param.requires_grad = True
            for param in ae.dann_discriminator.parameters():
                param.requires_grad = False
        return ae

    def unfreeze_layers(self, ae):
        """
        Unfreeze all layers
        Args:
            ae: AutoEncoder object. It inherits torch.nn.Module

        Returns:
            ae: The same AutoEncoder object, but with all frozen layers. Only the classifier layers are not frozen.

        """
        for param in ae.parameters():
            param.requires_grad = True
        return ae

    @staticmethod
    def get_mccs(lists, traces):
        """
        Function that gets the Matthews Correlation Coefficients. MCC is a statistical tool for model evaluation.
        It is a balanced measure which can be used even if the classes are of very different sizes.
        Args:
            lists:
            traces:

        Returns:
            traces: Same list as in the inputs arguments, except in now contains the MCC values
        """
        for group in ['train', 'valid', 'test']:
            try:
                preds, classes = np.concatenate(lists[group]['preds']).argmax(1), np.concatenate(
                    lists[group]['classes'])
            except:
                pass
            traces[group]['mcc'] = MCC(preds, classes)

        return traces


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--random_recs', type=int, default=0)
    parser.add_argument('--predict_tests', type=int, default=0)
    parser.add_argument('--early_stop', type=int, default=100)
    parser.add_argument('--early_warmup_stop', type=int, default=0, help='If 0, then no early warmup stop')
    parser.add_argument('--train_after_warmup', type=int, default=1)
    parser.add_argument('--threshold', type=float, default=0.)
    parser.add_argument('--n_epochs', type=int, default=1000)
    parser.add_argument('--n_trials', type=int, default=100)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--rec_loss', type=str, default='mse')
    parser.add_argument('--tied_weights', type=int, default=0)
    parser.add_argument('--variational', type=int, default=0)
    parser.add_argument('--zinb', type=int, default=0)
    parser.add_argument('--use_valid', type=int, default=0, help='Use if valid data is in a seperate file')
    parser.add_argument('--use_test', type=int, default=0, help='Use if test data is in a seperate file')
    parser.add_argument('--use_mapping', type=int, default=1, help="Use batch mapping for reconstruct")
    parser.add_argument('--freeze_ae', type=int, default=0)
    parser.add_argument('--freeze_c', type=int, default=0)
    parser.add_argument('--bdisc', type=int, default=1)
    parser.add_argument('--n_repeats', type=int, default=5)
    parser.add_argument('--dloss', type=str, default='inverseTriplet')  # one of revDANN, DANN, inverseTriplet, revTriplet
    parser.add_argument('--csv_file', type=str, default='unique_genes.csv')
    parser.add_argument('--best_features_file', type=str, default='')  # best_unique_genes.tsv
    parser.add_argument('--bad_batches', type=str, default='')  # 0;23;22;21;20;19;18;17;16;15
    parser.add_argument('--remove_zeros', type=int, default=1)
    parser.add_argument('--n_meta', type=int, default=0)
    parser.add_argument('--embeddings_meta', type=int, default=0)
    parser.add_argument('--features_to_keep', type=str, default='features_proteins.csv')
    parser.add_argument('--groupkfold', type=int, default=1)
    parser.add_argument('--dataset', type=str, default='prostate')
    parser.add_argument('--path', type=str, default='./data/')
    parser.add_argument('--exp_id', type=str, default='default_ae_classifier')
    parser.add_argument('--bs', type=int, default=32, help='Batch size')
    parser.add_argument('--n_agg', type=int, default=1, help='Number of trailing values to get stable valid values')
    parser.add_argument('--n_layers', type=int, default=1, help='N layers for classifier')
    parser.add_argument('--log1p', type=int, default=1, help='log1p the data? Should be 0 with zinb')
    parser.add_argument('--strategy', type=str, default='CU_DEM', help='only for alzheimer dataset')
    parser.add_argument('--pool', type=int, default=1, help='only for alzheimer dataset')

    args = parser.parse_args()
    try:
        mlflow.create_experiment(
            args.exp_id,
            # artifact_location=Path.cwd().joinpath("mlruns").as_uri(),
            # tags={"version": "v1", "priority": "P1"},
        )
    except:
        print(f"\n\nExperiment {args.exp_id} already exists\n\n")

    train = TrainAE(args, args.path, fix_thres=-1, load_tb=False, log_metrics=True, keep_models=False,
                    log_inputs=False, log_plots=True, log_tb=False, log_neptune=False,
                    log_mlflow=True, groupkfold=args.groupkfold)

    # train.train()
    # List of hyperparameters getting optimized
    parameters = [
        {"name": "nu", "type": "range", "bounds": [1e-4, 1e2], "log_scale": False},
        {"name": "lr", "type": "range", "bounds": [1e-4, 1e-2], "log_scale": True},
        {"name": "wd", "type": "range", "bounds": [1e-8, 1e-5], "log_scale": True},
        {"name": "l1", "type": "range", "bounds": [1e-8, 1e-5], "log_scale": True},
        # {"name": "lr_b", "type": "range", "bounds": [1e-6, 1e-1], "log_scale": True},
        # {"name": "wd_b", "type": "range", "bounds": [1e-8, 1e-5], "log_scale": True},
        {"name": "smoothing", "type": "range", "bounds": [0., 0.2]},
        {"name": "margin", "type": "range", "bounds": [0., 10.]},
        {"name": "warmup", "type": "range", "bounds": [1, 1000]},
        {"name": "disc_b_warmup", "type": "range", "bounds": [1, 2]},

        {"name": "dropout", "type": "range", "bounds": [0.0, 0.5]},
        {"name": "ncols", "type": "range", "bounds": [20, 10000]},
        {"name": "scaler", "type": "choice",
         "values": ['standard_per_batch', 'standard', 'robust', 'robust_per_batch']},  # scaler whould be no for zinb
        # {"name": "layer3", "type": "range", "bounds": [32, 512]},
        {"name": "layer2", "type": "range", "bounds": [32, 1024]},
        {"name": "layer1", "type": "range", "bounds": [512, 2048]},
    ]

    # Some hyperparameters are not always required. They are set to a default value in Train.train()
    if args.dloss in ['revTriplet', 'revDANN', 'DANN', 'inverseTriplet', 'normae']:
        # gamma = 0 will ensure DANN is not learned
        parameters += [{"name": "gamma", "type": "range", "bounds": [1e-2, 1e2], "log_scale": True}]
    if args.variational:
        # beta = 0 because useless outside a variational autoencoder
        parameters += [{"name": "beta", "type": "range", "bounds": [1e-2, 1e2], "log_scale": True}]
    if args.zinb:
        # zeta = 0 because useless outside a zinb autoencoder
        parameters += [{"name": "zeta", "type": "range", "bounds": [1e-2, 1e2], "log_scale": True}]

    best_parameters, values, experiment, model = optimize(
        parameters=parameters,
        evaluation_function=train.train,
        objective_name='closs',
        minimize=True,
        total_trials=args.n_trials,
        random_seed=41,

    )

    # fig = plt.figure()
    # render(plot_contour(model=model, param_x="learning_rate", param_y="weight_decay", metric_name='Loss'))
    # fig.savefig('test.jpg')
    # print('Best Loss:', values[0]['loss'])
    # print('Best Parameters:')
    # print(json.dumps(best_parameters, indent=4))
