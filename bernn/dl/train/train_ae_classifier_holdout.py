#!/usr/bin/python3

# Allow direct execution from the repository root, e.g.
# ``python bernn/dl/train/train_ae_classifier_holdout.py``.
import sys
from pathlib import Path
_repo_root = Path(__file__).resolve().parents[3]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import os
import traceback
import matplotlib
from bernn.utils.pool_metrics import log_pool_metrics
import uuid
import shutil
import matplotlib.pyplot as plt
import numpy as np
import random
# import json
import copy
import torch
from torch import nn
# from sklearn import metrics
from tensorboardX import SummaryWriter
from bernn.utils.ax_compat import optimize
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from bernn.ml.train.params_gp import linsvc_space
from bernn.dl.models.pytorch.utils.loggings import TensorboardLoggingAE, log_input_ordination
from bernn.dl.models.pytorch.utils.utils import LogConfusionMatrix
from bernn.dl.models.pytorch.utils.dataset import get_loaders, get_loaders_no_pool
from bernn.utils.utils import scale_data
from bernn.dl.models.pytorch.utils.utils import get_optimizer, get_empty_dicts, get_empty_traces, \
    log_traces, get_best_values, add_to_logger, add_to_mlflow
from bernn.utils.mlflow_compat import mlflow
import warnings
from datetime import datetime
from bernn.dl.train.train_ae import TrainAE
from typing import Union, Optional, Any
from bernn.config.training_config import TrainingConfig
import pandas as pd

matplotlib.use('Agg')
CUDA_VISIBLE_DEVICES = ""

random.seed(1)
torch.manual_seed(1)
np.random.seed(1)


class TrainAEClassifierHoldout(TrainAE):
    def __init__(self,
                 config: Union[TrainingConfig, object, None] = None,
                 n_epochs: Optional[int] = None,
                 dloss: Optional[str] = None,
                 variational: Optional[bool] = None,
                 n_layers: Optional[int] = None,
                 layer1: Optional[int] = None,
                 n_repeats: Optional[int] = None,
                 warmup: Optional[int] = None,
                 device: Optional[str] = None,
                 kan: Optional[bool] = None,
                 scaler: Optional[str] = None,
                 bs: Optional[int] = None,
                 n_trials: Optional[int] = None,
                 fix_thres: float = -1,
                 load_tb: bool = False,
                 log_metrics: bool = False,
                 keep_models: bool = True,
                 log_inputs: bool = False,
                 log_plots: bool = False,
                 log_tb: bool = False,
                 log_mlflow: bool = False,
                 log_dvclive: bool = False,
                 groupkfold: bool = True,
                 pools: bool = False,
                 **kwargs):
        """
        Args:
            config: TrainingConfig object, legacy args, or None
            n_epochs: Optional direct override for TrainingConfig.n_epochs
            dloss: Optional direct override for TrainingConfig.dloss
            variational: Optional direct override for TrainingConfig.variational
            n_layers: Optional direct override for TrainingConfig.n_layers
            layer1: Optional direct override for TrainingConfig.layer1
            n_repeats: Optional direct override for TrainingConfig.n_repeats
            warmup: Optional direct override for TrainingConfig.warmup
            device: Optional direct override for TrainingConfig.device
            kan: Optional direct override for TrainingConfig.kan
            scaler: Optional direct override for TrainingConfig.scaler
            bs: Optional direct override for TrainingConfig.bs
            n_trials: Optional direct override for TrainingConfig.n_trials
            fix_thres: Fixed zero-threshold for features
            load_tb: Load previous tensorboard runs
            log_metrics: Keep batch effect metrics
            keep_models: Save trained models
            log_inputs: Log input metrics/plots
            log_plots: Plot PCA/UMAP/etc.
            log_tb: Use tensorboard
            log_mlflow: Use mlflow
            log_dvclive: Use dvclive
            groupkfold: Use GroupKFold split
            pools: Use pooled data structures
        """
        direct_overrides: dict[str, Any] = {
            'n_epochs': n_epochs,
            'dloss': dloss,
            'variational': variational,
            'n_layers': n_layers,
            'layer1': layer1,
            'n_repeats': n_repeats,
            'warmup': warmup,
            'device': device,
            'kan': kan,
            'scaler': scaler,
            'bs': bs,
            'n_trials': n_trials,
        }
        direct_overrides = {k: v for k, v in direct_overrides.items() if v is not None}

        if config is None:
            # Filter kwargs to only include valid TrainingConfig fields
            valid_keys = {f.name for f in TrainingConfig.__dataclass_fields__.values()}
            filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_keys}
            filtered_kwargs.update(direct_overrides)
            self.config = TrainingConfig(**filtered_kwargs)
            args = self.config
        elif isinstance(config, TrainingConfig):
            self.config = config
            for key, value in direct_overrides.items():
                setattr(self.config, key, value)
            args = config
        elif isinstance(config, dict):
            merged_config = dict(config)
            merged_config.update(direct_overrides)
            self.config = TrainingConfig.from_dict(merged_config)
            args = self.config
        else:
            try:
                self.config = TrainingConfig.from_args(config)
                for key, value in direct_overrides.items():
                    setattr(self.config, key, value)
                    if hasattr(config, key):
                        setattr(config, key, value)
                args = config
            except Exception:
                args = config
                self.config = TrainingConfig(**direct_overrides)
        super().__init__(args=args, fix_thres=fix_thres, load_tb=load_tb, log_metrics=log_metrics, keep_models=keep_models, log_inputs=log_inputs, log_plots=log_plots, log_tb=log_tb,
                         log_mlflow=log_mlflow, log_dvclive=log_dvclive, groupkfold=groupkfold, pools=pools)

        # Deactivate SHAP by default
        self.use_shap = getattr(args, 'use_shap', False) if hasattr(self, 'args') else False

    # TODO SHOULD BE IN PARENT CLASS
    def launch_mlflow(self, params):
        mlflow.set_experiment(
            self.args.exp_id,
        )
        active_run = mlflow.active_run()
        if active_run is not None:
            mlflow.end_run()
        try:
            mlflow.start_run()
        except Exception as e:
            print(f"Error starting mlflow run: {e}")
            mlflow.end_run()
            mlflow.start_run()
        log_params = {
            "inputs_type": getattr(self.args, 'csv_file', 'unknown').split(".csv")[0],
            "best_unique": getattr(self.args, 'best_features_file', 'none').split(".tsv")[0],
            "tied_weights": getattr(self.args, 'tied_weights', False),
            "random_recs": getattr(self.args, 'random_recs', False),
            "train_after_warmup": getattr(self.args, 'train_after_warmup', False),
            "warmup_after_warmup": getattr(self.args, 'warmup_after_warmup', False),
            "dloss": getattr(self.args, 'dloss', 'unknown'),
            "predict_tests": getattr(self.args, 'predict_tests', False),
            "variational": getattr(self.args, 'variational', False),
            "threshold": getattr(self.args, 'threshold', 0.0),
            "rec_loss_type": getattr(self.args, 'rec_loss', 'l1'),
            "bad_batches": getattr(self.args, 'bad_batches', ''),
            "remove_zeros": getattr(self.args, 'remove_zeros', False),
            "parameters": params,
            "scaler": params['scaler'],
            "csv_file": getattr(self.args, 'csv_file', 'unknown'),
            "model_name": getattr(self.args, 'model_name', 'unknown'),
            "groupkfold": getattr(self.args, 'groupkfold', True),
            "foldername": self.foldername,
            "use_mapping": getattr(self.args, 'use_mapping', True),
            "dataset_name": getattr(self.args, 'dataset', 'unknown'),
            "n_agg": getattr(self.args, 'n_agg', 5),
            "lr": params['lr'],
            "wd": params['wd'],
            "dropout": params['dropout'],
            "margin": params['margin'],
            "smooth": params['smoothing'],
            "gamma": self.gamma,
            "beta": self.beta,
            "thres": params['thres'],
            "nu": params['nu'],
            "kan": self.args.kan,
            "l1": self.l1,
            "reg_entropy": self.reg_entropy,
            "use_l1": self.args.use_l1,
            "clip_val": self.args.clip_val,
            "update_grid": self.args.update_grid,
            "prune_threshold": self.args.prune_threshold,
        }
        for key, value in params.items():
            if key.startswith('layer'):
                log_params[key] = value
        mlflow.log_params(log_params)


    def get_ordered_layers(self, params):
        """Extract layer parameters from params dictionary, order them, and store in a new dictionary.

        Args:
            params (dict): Dictionary containing model parameters including layer sizes

        Returns:
            dict: Ordered dictionary of layer parameters
        """
        # Extract layer parameters and sort them
        layer_params = {k: v for k, v in params.items() if k.startswith('layer')}
        ordered_layers = dict(sorted(layer_params.items(),
                                     key=lambda x: int(x[0].replace('layer', ''))))
        return ordered_layers


    def _train(self, params=None):
        """
        Executes the specific training loop for this class.
        If params is None, uses values from self.args.

        Args:
            params: hyperparameters
        Returns:
            best valid classification loss (float) for Ax (minimize)
        """
        if params is None:
            params = {
                'lr': getattr(self.args, 'lr', 1e-3),
                'dropout': getattr(self.args, 'dropout', 0.1),
                'wd': getattr(self.args, 'wd', 1e-5),
                'margin': getattr(self.args, 'margin', 1.0),
                'smoothing': getattr(self.args, 'smoothing', 0.1),
                'scaler': getattr(self.args, 'scaler', 'standard'),
                'gamma': getattr(self.args, 'gamma', 0.1),
                'beta': getattr(self.args, 'beta', 0.1),
                'nu': getattr(self.args, 'nu', 1.0),
                'thres': getattr(self.args, 'thres', 0.0),
                'prune_threshold': getattr(self.args, 'prune_threshold', 0.0),
                'warmup': getattr(self.args, 'warmup', 100),
                'l1': getattr(self.args, 'l1', 0.0),
                'reg_entropy': getattr(self.args, 'reg_entropy', 0.0),
            }
            params.update(self.get_default_layer_params())

        start_time = datetime.now()
        params = self.make_params(params)
        optimizer_type = 'adam'
        metrics = {'pool_metrics': {}}
        loggers = {'cm_logger': LogConfusionMatrix(self.complete_log_path)}
        print(f'See results using: tensorboard --logdir={self.complete_log_path} --port=6006')

        os.makedirs(self.hparams_filepath, exist_ok=True)
        self.args.model_name = 'ae_classifier_holdout'
        if self.log_tb:
            loggers['tb_logging'] = TensorboardLoggingAE(self.hparams_filepath, params,
                                                         variational=self.args.variational,
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


        if self.log_mlflow:
            self.launch_mlflow(params)
        if not self.log_mlflow:
            model = None
            run = None
            
        best_closses = []
        best_mccs = []
        if self.rep < self.args.n_repeats:
            # prune_threshold = self.args.prune_threshold
            print(f'Rep: {self.rep}, Seed: {self.seed}')
            epoch = 0
            best_loss = np.inf
            best_closs = np.inf
            best_dom_loss = np.inf
            best_dom_acc = np.inf
            best_acc = 0
            best_mcc = -1
            self.warmup_counter = 0
            self.warmup_b_counter = 0
            self.warmup_disc_b = False
            if self.data is None or 'cats' not in self.data or self.data['cats'] is None:
                raise ValueError("Training data (specifically 'cats' labels) is not initialized. Ensure fit() is called correctly with valid labels.")
            self.n_cats = len(np.unique(self.data['cats']['all']))  # + 1  # for pool samples
            if self.args.groupkfold:
                combination = list(np.concatenate((np.unique(self.data['batches']['train']),
                                                   np.unique(self.data['batches']['valid']),
                                                   np.unique(self.data['batches']['test']))))
                self.seed += 1
                if combination not in self.combinations:
                    self.combinations += [combination]
                else:
                    return -1
            # print(combinations)
            self.columns = self.data['inputs']['all'].columns
            self.make_samples_weights()
            # event_acc is used to verify if the hparams have already been tested. If they were,
            # the best classification loss is retrieved and we go to the next trial
            event_acc = EventAccumulator(self.hparams_filepath)
            event_acc.Reload()
            if len(event_acc.Tags()['tensors']) > 2 and self.load_tb:
                pass
            else:
                # Transform the data with the chosen scaler
                data = copy.deepcopy(self.data)
                data, self.scaler = scale_data(params['scaler'], data, self.args.device)

                for g in list(data['inputs'].keys()):
                    data['inputs'][g] = data['inputs'][g].round(4)
                # Gets all the pytorch dataloaders to train the models
                if self.pools:
                    loaders = get_loaders(data, self.args.random_recs, self.samples_weights, self.args.dloss, None,
                                          None, bs=self.args.bs)
                else:
                    loaders = get_loaders_no_pool(data, self.args.random_recs, self.samples_weights, self.args.dloss,
                                                  None, None, bs=self.args.bs)

                if self.rep == 1 or self.args.kan == 1:
                    ae_cls = self.load_autoencoder()
                    self.ae = ae_cls(
                        data['inputs']['all'].shape[1],
                        is_sigmoid=self.args.use_sigmoid,
                        n_batches=self.n_batches,
                        nb_classes=self.n_cats,
                        mapper=self.args.use_mapping,
                        layers=self.get_ordered_layers(params),
                        n_layers=self.args.n_layers,
                        dropout=params['dropout'],
                        variational=self.args.variational,
                        conditional=False,
                        # zinb removed
                        add_noise=0,
                        tied_weights=self.args.tied_weights,
                        device=self.args.device,
                        prune_threshold=params['prune_threshold'],
                        update_grid=self.args.update_grid,
                    ).to(self.args.device)
                    self.ae.mapper.to(self.args.device)
                    self.ae.dec.to(self.args.device)
                    n_neurons = {}
                    init_n_neurons = {}
                    if params['prune_threshold'] > 0:
                        n_neurons = self.ae.prune_model_paperwise(False, False, weight_threshold=params['prune_threshold'])
                        init_n_neurons = self.ae.count_n_neurons()
                    # SHAP model only if enabled
                    if self.use_shap:
                        shap_ae = self.shap_ae(
                            data['inputs']['all'].shape[1],
                            is_sigmoid=self.args.use_sigmoid,
                            n_batches=self.n_batches,
                            nb_classes=self.n_cats,
                            mapper=self.args.use_mapping,
                            layers=self.get_ordered_layers(params),
                            n_layers=self.args.n_layers,
                            dropout=params['dropout'],
                            variational=self.args.variational, conditional=False,
                            add_noise=0, tied_weights=self.args.tied_weights,
                            device=self.args.device,
                            # prune_threshold=params['prune_threshold'],
                            # update_grid=self.args.update_grid
                        ).to(self.args.device)
                        shap_ae.mapper.to(self.args.device)
                        shap_ae.dec.to(self.args.device)
                else:
                    if self.ae is not None:
                        self.ae.random_init(nn.init.xavier_uniform_)
                    else:
                        # Defensive: re-instantiate if missing
                        ae_cls = self.load_autoencoder()
                        self.ae = ae_cls(
                            data['inputs']['all'].shape[1],
                            is_sigmoid=self.args.use_sigmoid,
                            n_batches=self.n_batches,
                            nb_classes=self.n_cats,
                            mapper=self.args.use_mapping,
                            layers=self.get_ordered_layers(params),
                            n_layers=self.args.n_layers,
                            dropout=params['dropout'],
                            variational=self.args.variational,
                            conditional=False,
                            add_noise=0,
                            tied_weights=self.args.tied_weights,
                            device=self.args.device,
                            prune_threshold=params['prune_threshold'],
                            update_grid=self.args.update_grid,
                        ).to(self.args.device)
                loggers['logger_cm'] = SummaryWriter(f'{self.complete_log_path}/cm')
                loggers['logger'] = SummaryWriter(f'{self.complete_log_path}/traces')
                sceloss, celoss, mseloss, triplet_loss = self.get_losses(
                    params['scaler'], params['smoothing'], params['margin'], self.args.dloss
                )

                optimizer_ae = get_optimizer(self.ae, params['lr'], params['wd'], optimizer_type)

                # Used only if bdisc==1
                optimizer_b = get_optimizer(self.ae.dann_discriminator, 1e-2, 0, optimizer_type)

                self.hparams_names = [x.name for x in linsvc_space]
                if self.log_inputs and not self.logged_inputs:
                    data['inputs']['all'].to_csv(
                        f'{self.complete_log_path}/{self.args.berm}_inputs.csv')
                    log_input_ordination(loggers['logger'], data, self.scaler, epoch)
                    if self.pools:
                        metrics = log_pool_metrics(data['inputs'], data['batches'], data['labels'], loggers, epoch,
                                                   metrics, 'inputs')
                    self.logged_inputs = True

                values, best_values, _, best_traces = get_empty_dicts()

                early_stop_counter = 0
                best_vals = values
                if self.rep > 0:
                    self.ae.load_state_dict(torch.load(f'{self.complete_log_path}/warmup.pth', weights_only=True))
                    print("\n\nNO WARMUP\n\n")
                if self.rep == 0:
                    for epoch in range(0, self.args.warmup):
                        no_error, self.ae, warmup = self.warmup_loop(optimizer_ae, None, self.ae, celoss, loaders['all'],
                                                                triplet_loss, mseloss, True, epoch,
                                                                optimizer_b, values, loggers, loaders, run, self.args.use_mapping)

                        if not warmup:
                            break
                        # End-of-epoch update (skip during warmup if set)
                        if self.args.update_grid and epoch >= getattr(self.args, 'update_grid_warmup', 0):
                            # Safely call grid updates on the AE model if available (KAN only)
                            try:
                                updated = self.ae.update_grids()
                                print(f"[epoch {epoch}] Updated {updated} KAN grids")
                            except Exception:
                                pass

                test_labels = np.asarray(data.get('labels', {}).get('test', []))
                has_test_labels = test_labels.size > 0 and not np.all(test_labels == -1)

                for epoch in range(0, self.args.n_epochs):
                    if early_stop_counter == self.args.early_stop:
                        if self.verbose > 0:
                            print('EARLY STOPPING.', epoch)
                        break
                    lists, traces = get_empty_traces()

                    if self.args.warmup_after_warmup:
                        no_error, self.ae, warmup = self.warmup_loop(optimizer_ae, None, self.ae, celoss, loaders['all'], triplet_loss, mseloss,
                                         False, epoch,
                                         optimizer_b, values, loggers, loaders, run, self.args.use_mapping)
                    if not self.args.train_after_warmup:
                        self.ae = self.freeze_all_but_clayers(self.ae)
                    losses = {
                        "mseloss": mseloss,
                        "celoss": sceloss,
                    }
                    # After warmup, run a warmup_loop if requested
                    if self.args.warmup_after_warmup:
                        no_error, self.ae, warmup = self.warmup_loop(
                            optimizer_ae, None, self.ae, celoss, loaders['all'],
                            triplet_loss, mseloss, False, epoch,
                            optimizer_b, values, loggers, loaders, run=None, mapping=self.args.use_mapping)
                    # Use modular training loops: train_classifier and train_bdisc
                    closs = self.train_classifier('train', optimizer_ae, self.ae, None, loaders['train'], nu=params['nu'])
                    if self.args.bdisc:
                        bdisc_loss = self.train_bdisc('train', optimizer_b, self.ae, None, loaders['train'])

                    if torch.isnan(closs):
                        print("NAN LOSS")
                        break
                    self.ae.eval()
                    self.ae.mapper.eval()

                    # Below is the loop for all sets
                    with torch.no_grad():
                        for group in list(data['inputs'].keys()):
                            if group in ['all', 'all_pool']:
                                continue
                            closs, lists, traces = self.loop(group, optimizer_ae, self.ae, sceloss,
                                                             loaders[group], lists, traces, nu=0)

                    traces = self.get_mccs(lists, traces)
                    values = log_traces(traces, values)
                    if self.log_tb:
                        try:
                            add_to_logger(values, loggers['logger'], epoch)
                        except Exception as e:
                            print(f"Problem with add_to_logger: {e}")
                    if self.log_mlflow:
                        add_to_mlflow(values, epoch)
                    valid_window = values['valid']['mcc'][-self.args.n_agg:]
                    has_test_metrics = (
                        has_test_labels
                        and len(values['test']['mcc']) > 0
                        and len(values['test']['acc']) > 0
                        and len(values['test']['closs']) > 0
                    )
                    test_summary = (
                        f", Test Acc: {values['test']['acc'][-1]}"
                        f", Test MCC: {values['test']['mcc'][-1]}"
                        f", Test loss: {values['test']['closs'][-1]}"
                        if has_test_metrics
                        else ", Test metrics unavailable (no y_test)"
                    )

                    valid_mcc_improved = (
                        len(valid_window) > 0
                        and len(values['valid']['mcc']) >= self.args.n_agg
                        and np.mean(valid_window) > best_mcc
                    )
                    if valid_mcc_improved:
                        print(f"Best Classification Epoch {epoch} (best valid MCC), "
                            f"Valid Acc: {values['valid']['acc'][-1]}, "
                            f"Valid MCC: {values['valid']['mcc'][-1]}, "
                            f"Classification train loss: {values['train']['closs'][-1]}, "
                            f"Valid loss: {values['valid']['closs'][-1]}"
                            f"{test_summary}")
                        best_mcc = np.mean(valid_window)
                        torch.save(self.ae.state_dict(), f'{self.complete_log_path}/model_{self.rep}_state.pth')
                        torch.save(self.ae, f'{self.complete_log_path}/model_{self.rep}.pth')
                        self._save_best_model_state(epoch, best_mcc)
                        best_values = get_best_values(values.copy(), ae_only=False, n_agg=self.args.n_agg)
                        best_vals = values.copy()
                        best_vals['rec_loss'] = best_loss
                        best_vals['dom_loss'] = best_dom_loss
                        best_vals['dom_acc'] = best_dom_acc
                        early_stop_counter = 0

                    if values['valid']['acc'][-1] > best_acc:
                        print(f"Best Valid-Acc Diagnostic Epoch {epoch}, "
                              f"Valid Acc: {values['valid']['acc'][-1]}, "
                              f"Valid MCC: {values['valid']['mcc'][-1]}, "
                              f"Classification train loss: {values['train']['closs'][-1]}, "
                              f"Valid loss: {values['valid']['closs'][-1]}"
                              f"{test_summary}")

                        best_acc = values['valid']['acc'][-1]

                    if values['valid']['closs'][-1] < best_closs:
                        print(f"Lowest Valid-Loss Diagnostic Epoch {epoch}, "
                              f"Valid Acc: {values['valid']['acc'][-1]}, "
                              f"Valid MCC: {values['valid']['mcc'][-1]}, "
                              f"Classification train loss: {values['train']['closs'][-1]}, "
                              f"Valid loss: {values['valid']['closs'][-1]}"
                              f"{test_summary}")
                        best_closs = values['valid']['closs'][-1]
                    if not valid_mcc_improved:
                        early_stop_counter += 1

                    if self.args.predict_tests and (epoch % 10 == 0):
                        loaders = get_loaders(self.data, data, self.args.random_recs, self.args.triplet_dloss, ae,
                                              self.ae.classifier, bs=self.args.bs)
                    if params['prune_threshold'] > 0:
                        n_neurons = self.ae.prune_model_paperwise(False, False, weight_threshold=params['prune_threshold'])

                best_mccs += [best_mcc]

                best_lists, traces = get_empty_traces()
                
                # Verify the model exists
                if not os.path.exists(f'{self.complete_log_path}/model_{self.rep}_state.pth'):
                    # Return a large penalty instead of -1 (keeps Ax consistent)
                    return 1e9
                # Loading best model that was saved during training
                self.ae.load_state_dict(torch.load(f'{self.complete_log_path}/model_{self.rep}_state.pth', weights_only=True))
                # Only use shap_ae if enabled
                if self.use_shap:
                    if self.rep == 1:
                        shap_ae.load_state_dict(torch.load(f'{self.complete_log_path}/model_{self.rep}_state.pth', weights_only=True))
                    shap_ae.eval()
                    shap_ae.mapper.eval()
                self.ae.eval()
                self.ae.mapper.eval()
                with torch.no_grad():
                    for group in list(data['inputs'].keys()):
                        closs, best_lists, traces = self.loop(group, None, self.ae, sceloss,
                                                              loaders[group], best_lists, traces, nu=0, mapping=False)
                best_closses += [best_closs]

                self.log_rep(best_lists, best_vals, best_values, traces, metrics, run, loggers, self.ae,
                             shap_ae if self.use_shap else None, self.rep, epoch)

        # Logging every model is taking too much resources and it makes it quite complicated to get information when
        # Too many runs have been made. This will make the notebook so much easier to work with
        if len(best_mccs) > 0 and np.mean(best_mccs) > self.best_mcc:
            try:
                if os.path.exists(
                        f'logs/best_models/ae_classifier_holdout/{self.args.dataset}/'
                        f'{self.args.dloss}_vae{self.args.variational}'
                ):
                    shutil.rmtree(
                        f'logs/best_models/ae_classifier_holdout/{self.args.dataset}/'
                        f'{self.args.dloss}_vae{self.args.variational}',
                        ignore_errors=True)
                shutil.copytree(f'{self.complete_log_path}',
                                f'logs/best_models/ae_classifier_holdout/'
                                f'{self.args.dataset}/{self.args.dloss}_vae{self.args.variational}')
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
        try:
            self.logging(run, loggers['cm_logger'])
        except Exception as e:
            print(f"Logging failed: {e}")

        if not self.keep_models:
            # shutil.rmtree(f'{self.complete_log_path}/traces', ignore_errors=True)
            # shutil.rmtree(f'{self.complete_log_path}/cm', ignore_errors=True)
            # shutil.rmtree(f'{self.complete_log_path}/hp', ignore_errors=True)
            shutil.rmtree(f'{self.complete_log_path}', ignore_errors=True)
        print('\n\nDuration: {}\n\n'.format(datetime.now() - start_time))
        best_closs = np.mean(best_closses) if len(best_closses) > 0 else np.inf
        if best_closs < best_closs:
            best_closs = best_closs
            print("Best closs!")

        # It should not be necessary. To remove once certain the "Too many files open" error is no longer a problem
        plt.close('all')
        
        self.rep += 1

        return np.mean(best_mccs)

    def increase_pruning_threshold(self):
        '''
        increase the pruning threshold
        
        Args:
        -----
            threshold : float
                the amount of increase
        
        Returns:
        --------
            None
        '''
        if self.args.prune_threshold == 0:
            self.args.prune_threshold = 1e-8
        else:
            self.args.prune_threshold *= 10


def main():
    import runpy

    runpy.run_module("bernn.dl.train.train_ae_classifier_holdout", run_name="__main__")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cross_validation', type=int, default=0, help='Enable cross-validation splitting (1/0)')
    parser.add_argument('--cross_test', type=int, default=0, help='Enable cross-test splitting (1/0)')
    parser.add_argument('--random_recs', type=int, default=0)
    parser.add_argument('--predict_tests', type=int, default=0)
    parser.add_argument('--early_stop', type=int, default=100)
    parser.add_argument('--early_warmup_stop', type=int, default=0, help='If 0, then no early warmup stop')
    parser.add_argument('--train_after_warmup', type=int, default=1, help="Train autoencoder after warmup")
    parser.add_argument('--warmup_after_warmup', type=int, default=1, help="Warmup after warmup")
    parser.add_argument('--max_warmup', type=int, default=10, help="Maximum number of warmup iterations")
    parser.add_argument('--threshold', type=float, default=0.)
    parser.add_argument('--n_epochs', type=int, default=1000)
    parser.add_argument('--n_trials', type=int, default=100)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--rec_loss', type=str, default='mse')
    parser.add_argument('--tied_weights', type=int, default=0)
    parser.add_argument('--variational', type=int, default=0)
    parser.add_argument('--use_valid', type=int, default=0, help='Use if valid data is in a seperate file')
    parser.add_argument('--use_test', type=int, default=0, help='Use if test data is in a seperate file')
    parser.add_argument('--use_mapping', type=int, default=1, help="Use batch mapping for reconstruct")
    parser.add_argument('--freeze_ae', type=int, default=0)
    parser.add_argument('--freeze_c', type=int, default=0)
    parser.add_argument('--bdisc', type=int, default=1)
    parser.add_argument('--n_repeats', type=int, default=5)
    parser.add_argument('--dloss', type=str, default='inverseTriplet')
    parser.add_argument('--class_triplet', type=int, default=0,
                        help='Add a class-based triplet loss on embeddings (combinable with dloss)')
    parser.add_argument('--class_triplet_w', type=float, default=1.0,
                        help='Weight of the class-based triplet loss')
    parser.add_argument('--csv_file', type=str, default='matrix.csv')
    parser.add_argument('--best_features_file', type=str, default='')  # best_unique_genes.tsv
    parser.add_argument('--bad_batches', type=str, default='')  # 0;23;22;21;20;19;18;17;16;15
    parser.add_argument('--remove_zeros', type=int, default=1)
    parser.add_argument('--features_to_keep', type=str, default='features_proteins.csv')
    parser.add_argument('--groupkfold', type=int, default=1)
    parser.add_argument('--dataset', type=str, default='custom')
    parser.add_argument('--path', type=str, default='./data/PXD015912/', help='Directory containing the CSV file (CLI only)')
    parser.add_argument('--exp_id', type=str, default='reviewer_exp')
    parser.add_argument('--bs', type=int, default=32, help='Batch size')
    parser.add_argument('--n_agg', type=int, default=1, help='Number of trailing values to get stable valid values')
    parser.add_argument('--n_layers', type=int, default=1, help='N layers for classifier')
    parser.add_argument('--log1p', type=int, default=1, help='log1p the data?')
    parser.add_argument('--strategy', type=str, default='CU_DEM', help='only for alzheimer dataset')
    parser.add_argument('--pool', type=int, default=1, help='only for alzheimer dataset')
    parser.add_argument('--log_plots', type=int, default=1, help='')
    parser.add_argument('--log_metrics', type=int, default=0, help='')
    parser.add_argument('--controls', type=str, default='', help='Which samples are the controls. Empty for not binary')
    parser.add_argument('--n_features', type=int, default=-1, help='')
    parser.add_argument('--kan', type=int, default=1, help='')
    parser.add_argument('--update_grid', type=int, default=1, help='')
    parser.add_argument('--use_l1', type=int, default=1, help='')
    parser.add_argument('--clip_val', type=float, default=1, help='')
    parser.add_argument('--prune_threshold', type=float, default=1e-4, help='')
    parser.add_argument('--prune_neurites_threshold', type=float, default=0.0, help='')
    parser.add_argument('--prune_network', type=float, default=0, help='')
    parser.add_argument('--log_inputs', type=int, default=0, help='')
    parser.add_argument('--log_mlflow', type=int, default=1, help='Enable MLflow logging (recommended).')
    parser.add_argument('--log_dvclive', type=int, default=0, help='')
    parser.add_argument('--log_tb', type=int, default=0, help='')
    parser.add_argument('--keep_models', type=int, default=0, help='')
    parser.add_argument('--update_grid_warmup', type=int, default=0, help='If > 0, then update grid after this many epochs of warmup')
    parser.add_argument('--classif_loss', type=str, default='ce', help='')
    parser.add_argument('--scheduler', type=str, default='ReduceLROnPlateau', help='')
    parser.add_argument('--n_move_valid', type=int, default=0, help='Number of valid samples to move to train')
    parser.add_argument('--n_move_test', type=int, default=0, help='Number of test samples to move to valid')
    args = parser.parse_args()

    if args.kan == 0:
        args.update_grid = 0
        args.update_grid_warmup = 0

    
    # 1. RECOMMENDED: Modern approach with TrainingConfig
    print("=== Modern Configuration Class Approach ===")
    config = TrainingConfig.from_args(args)
    config.exp_id = 'modern_ae_classifier_holdout' # Override if needed

    # Clean, type-safe instantiation
    trainer_modern = TrainAEClassifierHoldout(
        config=config,
        log_metrics=args.log_metrics,
        keep_models=args.keep_models,
        log_inputs=args.log_inputs,
        log_plots=args.log_plots,
        log_tb=args.log_tb,
        log_mlflow=args.log_mlflow,
        log_dvclive=args.log_dvclive,
        pools=False  # TODO redundancy with args.pool
    )

    try:
        mlflow.create_experiment(
            args.exp_id,
            # artifact_location=Path.cwd().joinpath("mlruns").as_uri(),
            # tags={"version": "v1", "priority": "P1"},
        )
    except Exception as e:
        print(f"Error creating experiment: {e}")
        print(f"\n\nExperiment {args.exp_id} already exists\n\n")

    # args.batch_columns = [int(x) for x in args.batch_columns.split(',')]

    train = TrainAEClassifierHoldout(args, fix_thres=-1, load_tb=False,
                                     log_metrics=args.log_metrics,
                                     keep_models=args.keep_models, log_inputs=args.log_inputs,
                                     log_plots=args.log_plots, log_tb=args.log_tb,
                                     log_mlflow=args.log_mlflow, groupkfold=args.groupkfold,
                                     pools=args.pool)

    # 3. CLI Data Loading: Load data from disk and pass to fit()
    csv_path = os.path.join(args.path, args.csv_file)
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path, index_col=0)
    
    # Improved heuristic to find labels and groups
    y = None
    if 'labels' in df.columns:
        y = df['labels'].values
    elif 'group' in df.columns:
        y = df['group'].values
        
    groups = None
    if 'batches' in df.columns:
        groups = df['batches'].values
    elif 'batch' in df.columns:
        groups = df['batch'].values
        
    X = df.drop(['labels', 'batches', 'group', 'batch'], axis=1, errors='ignore')

    from sklearn.model_selection import train_test_split
    
    print("Data loaded and split for optimization.")
    # No need to call fit() here, ax_eval will call it for each trial.


    # List of hyperparameters getting optimized
    parameters = [
        {"name": "n_layers", "type": "choice", "values": [1, 2, 3, 4, 5]},
        {"name": "nu", "type": "range", "bounds": [1e-4, 1e2], "log_scale": False},
        {"name": "lr", "type": "range", "bounds": [1e-4, 1e-2], "log_scale": True},
        {"name": "wd", "type": "range", "bounds": [1e-6, 1e-3], "log_scale": True},
        {"name": "smoothing", "type": "range", "bounds": [0., 0.2]},
        {"name": "margin", "type": "range", "bounds": [0., 10.]},
        {"name": "warmup", "type": "range", "bounds": [1, args.max_warmup]},
        {"name": "dropout", "type": "range", "bounds": [0.0, 0.5]},
        {"name": "scaler", "type": "choice", "values": ['standard', 'robust', 'standard_per_batch', 'robust_per_batch']},
        {"name": "thres", "type": "range", "bounds": [0.0, 0.1]},
        {"name": "layer1", "type": "range", "bounds": [512, 1024]},        
    ]

    # Some hyperparameters are not always required. 
    if args.dloss in ['revTriplet', 'revDANN', 'DANN', 'inverseTriplet', 'normae']:
        # gamma = 0 will ensure DANN is not learned
        parameters += [{"name": "gamma", "type": "range", "bounds": [1e-2, 1e2], "log_scale": True}]
    if args.variational:
        # beta = 0 because useless outside a variational autoencoder
        parameters += [{"name": "beta", "type": "range", "bounds": [1e-2, 1e2], "log_scale": True}]
    if args.kan and args.use_l1:
        parameters += [{"name": "reg_entropy", "type": "range", "bounds": [1e-5, 1e-2], "log_scale": True}]
    if args.use_l1:
        parameters += [{"name": "l1", "type": "range", "bounds": [1e-5, 1e-3], "log_scale": True}]
    if args.prune_network:
        parameters += [{"name": "prune_threshold", "type": "range", "bounds": [1e-3, 3e-3], "log_scale": True}]

    # Wrapper for Ax
    def ax_eval(parameterization):
        try:
            # Ax may give numpy types; ensure plain Python
            param_clean = {k: (int(v) if (k == 'n_layers' or k.startswith('layer')) else float(v) if isinstance(v, (np.floating,)) else v)
                           for k, v in parameterization.items()}
            train.fit(
                X, y, groups_train=groups, params=param_clean,
                cross_validation=bool(args.cross_validation), cross_test=bool(args.cross_test)
            )
            valid_mcc = float(train.best_mcc)
            train._reset_counts()
        except Exception as e:
            print(f"[AX WARN] Trial failed: {e}")
            traceback.print_exc()
            valid_mcc = -1e9
        # Ax optimize() (managed_loop) accepts scalar when objective_name is provided,
        # but explicit dict form is more robust; returning scalar is also fine. Choose one:
        return valid_mcc  # scalar OK because objective_name='mcc'

    best_parameters, values, experiment, model = optimize(
        parameters=parameters,
        evaluation_function=ax_eval,
        objective_name='mcc',
        minimize=False,
        total_trials=args.n_trials,
        random_seed=41
    )
