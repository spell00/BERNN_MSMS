#!/usr/bin/python3

import os
import matplotlib

import uuid
import shutil
from pathlib import Path
from typing import Union
import sys

# Add the project root to the path
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')))

from bernn.config.training_config import TrainingConfig
from bernn.utils.pool_metrics import log_pool_metrics

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import copy
import torch
from torch import nn
from tensorboardX import SummaryWriter

# Handle ax-platform import with graceful fallback
try:
    from ax.service.managed_loop import optimize
    AX_AVAILABLE = True
except ImportError as e:
    print(f"Warning: ax-platform not available or incompatible: {e}")
    print("Hyperparameter optimization features will be disabled.")
    AX_AVAILABLE = False
    def optimize(*args, **kwargs):
        raise ImportError("ax-platform is not available. Please install with: pip install ax-platform==0.3.7")

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from bernn.ml.train.params_gp import linsvc_space
# from bernn.utils.data_getters import get_alzheimer, get_amide, get_mice, get_data
from bernn.dl.models.pytorch.aedann import ReverseLayerF
from bernn.dl.models.pytorch.utils.loggings import TensorboardLoggingAE, log_input_ordination
from bernn.dl.models.pytorch.utils.utils import LogConfusionMatrix
from bernn.dl.models.pytorch.utils.dataset import get_loaders, get_loaders_no_pool
from bernn.utils.utils import scale_data
from bernn.dl.models.pytorch.utils.utils import get_optimizer, get_empty_dicts, get_empty_traces, \
    log_traces, get_best_values, add_to_logger, add_to_mlflow
import mlflow
import warnings
from datetime import datetime
from bernn.dl.train.train_ae import TrainAE

matplotlib.use('Agg')
CUDA_VISIBLE_DEVICES = ""
NEPTUNE_API_TOKEN = os.environ.get("NEPTUNE_API_TOKEN")
NEPTUNE_PROJECT_NAME = "BERNN"
DEPRECATED_NEPTUNE_MESSAGE = (
    "[DEPRECATED] Neptune integration is disabled and no longer supported. "
    "Please use MLflow logging instead (--log_mlflow=1)."
)

warnings.filterwarnings("ignore")

random.seed(1)
torch.manual_seed(1)
np.random.seed(1)


def _disable_deprecated_neptune(log_neptune: bool) -> bool:
    if bool(log_neptune):
        print(DEPRECATED_NEPTUNE_MESSAGE)
    return False


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
        controls: The control labels (string, list, or semicolon-separated string)

    Returns:
        labels: The binarized labels
    """
    if isinstance(controls, str) and ';' in controls:
        controls = controls.split(';')
    elif isinstance(controls, str) and not controls:
        controls = []
    elif isinstance(controls, str):
        controls = [controls]

    for group in ['all', 'train', 'valid', 'test']:
        if group in data['labels']:
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


class TrainAEThenClassifierHoldout(TrainAE):
    """
    This class was previously named TrainAEClassifierHoldout. It is now TrainAEThenClassifierHoldout.

    Modern usage with configuration class (recommended):
        config = TrainingConfig(
            csv_file='my_data.csv',
            dloss='DANN',
            variational=True,
            n_epochs=500
        )
        trainer = TrainAEThenClassifierHoldout(config, path='./data')

    Legacy usage (still supported):
        trainer = TrainAEThenClassifierHoldout(args)
    """

    def __init__(self,
                 config: Union[TrainingConfig, object, None] = None,
                 fix_thres: float = -1,
                 log_metrics: bool = False,
                 keep_models: bool = True,
                 log_inputs: bool = False,
                 log_plots: bool = False,
                 log_tb: bool = False,
                 log_neptune: bool = False,
                 log_mlflow: bool = False,
                 log_dvclive: bool = False,
                 groupkfold: bool = True,
                 pools: bool = False,
                 load_tb: bool = False,
                 **kwargs):
        """
        Initialize the TrainAEThenClassifierHoldout trainer.

        Args:
            config: TrainingConfig object with training parameters, or legacy args object, or None
            fix_thres: If 1 > fix_thres >= 0 then the threshold is fixed to that value.
                       any other value means the threshold won't be fixed and will be
                       learned as an hyperparameter
            log_metrics: Whether or not to keep the batch effect metrics
            keep_models: Whether or not to save the models trained
                         (can take a lot of space if training a lot of models)
            log_inputs: Whether or not to log graphs or batch effect metrics
                        of the scaled inputs
            log_plots: For each optimization iteration, on the first iteration, whether or
                       not to plot PCA, UMAP, CCA and LDA of the encoded and reconstructed
                       representations.
            log_tb: Whether or not to use tensorboard.
            log_neptune: Whether or not to use neptune.
            log_mlflow: Whether or not to use mlflow.
            log_dvclive: Whether or not to use dvclive.
            groupkfold: Whether or not to use GroupKFold cross-validation.
            pools: Whether or not to use pooled samples.
            load_tb: Whether or not to load previous tensorboard runs.
            **kwargs: Additional keyword arguments to pass to the TrainingConfig constructor.

        Examples:
            # Modern approach with configuration class
            config = TrainingConfig(
                csv_file='adenocarcinoma_data.csv',
                dloss='DANN',
                variational=True,
                n_epochs=500,
                device='cuda:0'
            )
            trainer = TrainAEThenClassifierHoldout(config)

            # Quick setup with kwargs
            trainer = TrainAEThenClassifierHoldout(
                None,
                csv_file='my_data.csv',
                dloss='inverseTriplet',
                n_epochs=1000
            )

            # Legacy approach (still supported)
            trainer = TrainAEThenClassifierHoldout(args)
        """

        # Handle different input types
        if config is None:
            # Create config from kwargs
            self.config = TrainingConfig(**kwargs)
            # Convert to args-like object for parent class compatibility
            args = self.config
        elif isinstance(config, TrainingConfig):
            # Already a TrainingConfig
            self.config = config
            args = config
        else:
            # Legacy args object - convert to TrainingConfig
            try:
                self.config = TrainingConfig.from_args(config)
                args = config  # Keep original for parent class
            except Exception:
                # If conversion fails, keep original args and create default config
                args = config
                self.config = TrainingConfig()
        log_neptune = _disable_deprecated_neptune(log_neptune)
        self.log_neptune = log_neptune

        super().__init__(
            args=args,
            fix_thres=fix_thres,
            load_tb=load_tb,
            log_metrics=log_metrics,
            keep_models=keep_models,
            log_inputs=log_inputs,
            log_plots=log_plots,
            log_tb=log_tb,
            log_neptune=log_neptune,
            log_mlflow=log_mlflow,
            log_dvclive=False,
            groupkfold=groupkfold,
            pools=pools,
        )
        self.log_neptune = log_neptune

    def get_ordered_layers(self, params):
        layer_params = {k: v for k, v in params.items() if k.startswith('layer')}
        return dict(sorted(layer_params.items(), key=lambda x: int(x[0].replace('layer', ''))))



    def _train(self, params=None):
        """
        Executes the specific training loop for this class.
        If params is None, uses values from self.args.
        Returns:
            best valid classification loss (float) for Ax (minimize)
        """
        if params is None:
            params = {
                'lr': getattr(self.args, 'lr', 1e-3),
                'layer1': getattr(self.args, 'layer1', 512),
                'layer2': getattr(self.args, 'layer2', 128),
                'layer3': getattr(self.args, 'layer3', 32),
                'dropout': getattr(self.args, 'dropout', 0.1),
                'wd': getattr(self.args, 'wd', 1e-5),
                'margin': getattr(self.args, 'margin', 1.0),
                'smoothing': getattr(self.args, 'smoothing', 0.1),
                'scaler': getattr(self.args, 'scaler', 'standard'),
                'gamma': getattr(self.args, 'gamma', 0.1),
                'beta': getattr(self.args, 'beta', 0.1),
                'zeta': getattr(self.args, 'zeta', 0.1),
                'nu': getattr(self.args, 'nu', 1.0),
                'thres': getattr(self.args, 'thres', 0.0),
                'prune_threshold': getattr(self.args, 'prune_threshold', 0.0),
                'warmup': getattr(self.args, 'warmup', 100),
                'l1': getattr(self.args, 'l1', 0.0),
                'reg_entropy': getattr(self.args, 'reg_entropy', 0.0),
            }

        start_time = datetime.now()
        # Fixing the hyperparameters that are not optimized
        if self.args.dloss not in ['revTriplet', 'revDANN', 'DANN',
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
        if not self.args.prune_network:
            params['prune_threshold'] = 0
        params.setdefault('prune_threshold', 0)

        if not self.args.kan:
            params['reg_entropy'] = 0
        if not self.args.use_l1:
            params['l1'] = 0
        params['smoothing'] = 0
        print(params)
        # Assigns the hyperparameters getting optimized
        smooth = params['smoothing']
        layer1 = params['layer1']
        layer2 = params['layer2']
        scale = params['scaler']
        gamma = params['gamma']
        beta = params['beta']
        zeta = params['zeta']
        wd = params['wd']
        nu = params['nu']
        lr = params['lr']
        self.l1 = params['l1']
        self.reg_entropy = params.get('reg_entropy', 0)

        if params['prune_threshold'] > 0:
            dropout = 0
        else:
            dropout = params['dropout']
        margin = params['margin']

        self.args.scaler = scale
        self.args.warmup = params['warmup']
        # self.args.disc_b_warmup = params['disc_b_warmup']

        optimizer_type = 'adam'
        metrics = {'pool_metrics': {}}
        # self.log_path is where tensorboard logs are saved
        self.foldername = str(uuid.uuid4())

        self.complete_log_path = f'logs/ae_then_classifier_holdout/{self.foldername}'
        loggers = {'cm_logger': LogConfusionMatrix(self.complete_log_path)}
        print(f'See results using: tensorboard --logdir={self.complete_log_path} --port=6006')

        hparams_filepath = self.complete_log_path + '/hp'
        os.makedirs(hparams_filepath, exist_ok=True)
        self.args.model_name = 'ae_then_classifier_holdout'
        if self.log_tb:
            loggers['tb_logging'] = TensorboardLoggingAE(hparams_filepath, params, variational=self.args.variational,
                                                         zinb=self.args.zinb,
                                                         tw=self.args.tied_weights,
                                                         dloss=self.args.dloss,
                                                         tl=0,  # to remove
                                                         pseudo=self.args.predict_tests,
                                                         train_after_warmup=self.args.train_after_warmup,
                                                         berm='no',  # to remove
                                                         args=self.args)
        if self.log_neptune:
            print(DEPRECATED_NEPTUNE_MESSAGE)
            self.log_neptune = False
            model = None
            run = None
        else:
            model = None
            run = None

        if self.log_mlflow:
            mlflow.set_experiment(
                self.args.exp_id,
            )
            try:
                mlflow.start_run()
            except:
                mlflow.end_run()
                mlflow.start_run()
            mlflow.log_params({
                "inputs_type": self.args.csv_file.split(".csv")[0],
                "best_unique": self.args.best_features_file.split(".tsv")[0],
                "tied_weights": self.args.tied_weights,
                "random_recs": self.args.random_recs,
                "train_after_warmup": self.args.train_after_warmup,
                "dloss": self.args.dloss,
                "predict_tests": self.args.predict_tests,
                "variational": self.args.variational,
                "zinb": self.args.zinb,
                "threshold": self.args.threshold,
                "rec_loss_type": self.args.rec_loss,
                "bad_batches": self.args.bad_batches,
                "remove_zeros": self.args.remove_zeros,
                "parameters": params,
                "scaler": params['scaler'],
                "csv_file": self.args.csv_file,
                "model_name": self.args.model_name,
                "n_meta": self.args.n_meta,
                "n_emb": self.args.embeddings_meta,
                "groupkfold": self.args.groupkfold,
                "foldername": self.foldername,
                "use_mapping": self.args.use_mapping,
                "dataset_name": self.args.dataset,
                "n_agg": self.args.n_agg,
                "kan": self.args.kan,
                "l1": self.l1,
                "reg_entropy": self.reg_entropy,
                "use_l1": self.args.use_l1,
                "clip_val": self.args.clip_val,
                "update_grid": self.args.update_grid,
            })
        else:
            model = None
            run = None
        seed = 0
        combinations = []
        h = 0
        best_closses = []
        best_mccs = []

        # warmup is done only once, at first repeat
        warmup_counter = 0
        warmup_b_counter = 0
        if self.args.warmup > 0:
            warmup = True
        else:
            warmup = False
        warmup_disc_b = False

        while h < self.args.n_repeats:
            print(f'Rep: {h}')
            epoch = 0
            best_loss = np.inf
            best_closs = np.inf
            best_dom_loss = np.inf
            best_dom_acc = np.inf
            best_acc = 0
            best_mcc = -np.inf
            if self.data is None or 'batches' not in self.data or self.data['batches'] is None:
                raise ValueError("Training data (specifically 'batches' info) is not initialized. Ensure fit() is called correctly.")
            if self.args.groupkfold:
                combination = list(np.concatenate((np.unique(self.data['batches']['train']),
                                                   np.unique(self.data['batches']['valid']),
                                                   np.unique(self.data['batches']['test']))))
                seed += 1
                if combination not in combinations:
                    combinations += [combination]
                else:
                    continue
            h += 1
            self.columns = self.data['inputs']['all'].columns
            self.make_samples_weights()
            # event_acc is used to verify if the hparams have already been tested. If they were,
            # the best classification loss is retrieved and we go to the next trial
            event_acc = EventAccumulator(hparams_filepath)
            event_acc.Reload()
            # Transform the data with the chosen scaler
            data = copy.deepcopy(self.data)
            data, self.scaler = scale_data(scale, data, self.args.device)

            for g in list(data['inputs'].keys()):
                data['inputs'][g] = data['inputs'][g].round(4)
            # Gets all the pytorch dataloaders to train the models
            if self.pools:
                loaders = get_loaders(data, self.args.random_recs, self.samples_weights, self.args.dloss, None,
                                      None, bs=getattr(self.args, 'bs', 32))
            else:
                loaders = get_loaders_no_pool(data, self.args.random_recs, self.samples_weights, self.args.dloss,
                                              None, None, bs=getattr(self.args, 'bs', 32))

            ae = self.ae(
                data['inputs']['all'].shape[1],
                is_sigmoid=self.args.use_sigmoid,
                n_batches=self.n_batches,
                nb_classes=self.n_cats,
                mapper=self.args.use_mapping,
                layers=self.get_ordered_layers(params),
                n_layers=self.args.n_layers,
                n_meta=self.args.n_meta,
                n_emb=self.args.embeddings_meta,
                dropout=dropout,
                variational=self.args.variational,
                conditional=False,
                zinb=self.args.zinb,
                add_noise=0,
                tied_weights=self.args.tied_weights,
                prune_threshold=params['prune_threshold'],
                device=self.args.device,
                update_grid=self.args.update_grid,
            ).to(self.args.device)
            self.ae_instance = ae
            if self.args.kan:
                self.count_neurons(ae)
            ae.mapper.to(self.args.device)
            ae.dec.to(self.args.device)
            n_neurons = ae.prune_model_paperwise(False, False, weight_threshold=params['prune_threshold'])
            init_n_neurons = ae.count_n_neurons()

            # if self.args.embeddings_meta > 0:
            #     n_meta = self.n_meta
            shap_ae = self.shap_ae(
                data['inputs']['all'].shape[1],
                is_sigmoid=self.args.use_sigmoid,
                n_batches=self.n_batches,
                nb_classes=self.n_cats,
                mapper=self.args.use_mapping,
                layers=self.get_ordered_layers(params),
                n_layers=self.args.n_layers,
                n_meta=self.args.n_meta,
                n_emb=self.args.embeddings_meta,
                dropout=dropout,
                variational=self.args.variational,
                conditional=False,
                zinb=self.args.zinb,
                add_noise=0,
                tied_weights=self.args.tied_weights,
                device=self.args.device,
            ).to(self.args.device)
            shap_ae.mapper.to(self.args.device)
            shap_ae.dec.to(self.args.device)
            loggers['logger_cm'] = SummaryWriter(f'{self.complete_log_path}/cm')
            loggers['logger'] = SummaryWriter(f'{self.complete_log_path}/traces')
            sceloss, celoss, mseloss, triplet_loss = self.get_losses(scale, smooth, margin, self.args.dloss)

            optimizer_ae = get_optimizer(ae, lr, wd, optimizer_type)
            optimizer_c = get_optimizer(ae.classifier, nu * lr, wd, optimizer_type)

            # Used only if bdisc==1
            optimizer_b = get_optimizer(ae.dann_discriminator, 1e-2, 0, optimizer_type)

            self.hparams_names = [x.name for x in linsvc_space]
            if self.log_inputs and not self.logged_inputs:
                data['inputs']['all'].to_csv(
                    f'{self.complete_log_path}/{self.args.berm}_inputs.csv')  # TODO berm (batch effect removal method) has been removed. change this
                if self.log_neptune:
                    run["inputs.csv"].track_files(f'{self.complete_log_path}/{self.args.berm}_inputs.csv')
                log_input_ordination(loggers['logger'], data, self.scaler, epoch)
                if self.pools:
                    metrics = log_pool_metrics(data['inputs'], data['batches'], data['labels'],
                                               self.unique_unique_labels, loggers, epoch, metrics, 'inputs')
                self.logged_inputs = True

            values, best_values, _, best_traces = get_empty_dicts()

            best_vals = values
            if h > 1:  # or warmup_counter == 100:
                ae.load_state_dict(torch.load(f'{self.complete_log_path}/warmup.pth'))
                print(f"\n\nNO WARMUP\n\n")
            # while new_combinations:
            if h == 1:
                for epoch in range(0, self.args.warmup):
                    lists, traces = get_empty_traces()
                    ae.train()

                    iterator = enumerate(loaders['all'])

                    # If option train_after_warmup=1, then this loop is only for preprocessing
                    # TODO MAKE warmup loop like in train_ae_classifier_holdout, or make just 1 file for both (2nd option better)
                    if warmup or self.args.train_after_warmup:
                        for i, all_batch in iterator:
                            if warmup or self.args.train_after_warmup:
                                optimizer_ae.zero_grad()
                            inputs, meta_inputs, names, labels, domain, to_rec, not_to_rec, pos_to_rec, neg_to_rec, \
                                pos_batch_sample, neg_batch_sample, meta_pos_batch_sample, meta_neg_batch_sample, _ = all_batch
                            inputs = inputs.to(self.args.device).float()
                            meta_inputs = meta_inputs.to(self.args.device).float()
                            to_rec = to_rec.to(self.args.device).float()
                            if self.args.n_meta > 0:
                                inputs = torch.cat((inputs, meta_inputs), 1)
                                to_rec = torch.cat((to_rec, meta_inputs), 1)

                            enc, rec, zinb_loss, kld = ae(inputs, to_rec, domain, sampling=True)
                            if enc.abs().sum() == 0 or rec['mean'][0].abs().sum() == 0:
                                return -1
                            rec = rec['mean']
                            zinb_loss = zinb_loss.to(self.args.device)
                            reverse = ReverseLayerF.apply(enc, 1)
                            if self.args.dloss == 'DANN':
                                domain_preds = ae.dann_discriminator(reverse)
                                is_dann = True
                            else:
                                domain_preds = ae.dann_discriminator(enc)
                                is_dann = False
                            if self.args.dloss not in ['revTriplet', 'inverseTriplet']:
                                dloss, domain = self.get_dloss(celoss, domain, domain_preds, 2)
                            elif self.args.dloss == 'revTriplet':
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
                            elif self.args.dloss == 'inverseTriplet':
                                pos_batch_sample, neg_batch_sample = neg_batch_sample.to(
                                    self.args.device).float(), pos_batch_sample.to(self.args.device).float()
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
                                return best_loss

                            if isinstance(rec, list):
                                rec = rec[-1]
                            if isinstance(to_rec, list):
                                to_rec = to_rec[-1]
                            if not self.args.kan and self.l1 > 0:
                                l1_loss = self.l1_regularization(ae, self.l1)
                            elif self.args.kan and self.l1 > 0:
                                l1_loss = self.reg_kan(ae, self.l1, self.reg_entropy)
                            else:
                                l1_loss = torch.zeros(1).to(self.args.device)[0]
                            l1_loss += self.l1 * self.l1_regularization(ae, self.l1)
                            rec_loss = mseloss(rec, to_rec)
                            # if zinb_loss > 0:
                            #     rec_loss = zinb_loss
                            # else:
                            #     rec_loss = zinb_loss
                            traces['rec_loss'] += [rec_loss.item()]
                            traces['dom_loss'] += [dloss.item()]
                            traces['dom_acc'] += [np.mean([0 if pred != dom else 1 for pred, dom in
                                                           zip(domain_preds.detach().float().cpu().numpy().argmax(1),
                                                               domain.detach().int().cpu().numpy())])]
                            # lists['all']['set'] += [np.array([group for _ in range(len(domain))])]
                            lists['all']['domains'] += [np.array(
                                [self.unique_batches[d] for d in domain.detach().int().cpu().numpy()])]
                            lists['all']['domain_preds'] += [domain_preds.detach().float().cpu().numpy()]
                            # lists[group]['preds'] += [preds.detach().float().cpu().numpy()]
                            lists['all']['classes'] += [labels.detach().float().cpu().numpy()]
                            lists['all']['encoded_values'] += [
                                enc.detach().float().cpu().numpy()]
                            lists['all']['rec_values'] += [
                                rec.detach().float().cpu().numpy()]
                            lists['all']['names'] += [names]
                            lists['all']['gender'] += [meta_inputs.detach().float().cpu().numpy()[:, -1]]
                            lists['all']['age'] += [meta_inputs.detach().float().cpu().numpy()[:, -2]]
                            lists['all']['atn'] += [str(x) for x in
                                                    meta_inputs.detach().float().cpu().numpy()[:, -5:-2]]
                            lists['all']['inputs'] += [data['inputs']['all'].to_numpy()]
                            try:
                                lists['all']['labels'] += [np.array(
                                    [self.unique_labels[x] for x in labels.detach().float().cpu().numpy()])]
                            except:
                                pass
                            if warmup or self.args.train_after_warmup and not warmup_disc_b:
                                # (rec_loss + gamma * dloss + beta * kld.mean()).backward()
                                (rec_loss + gamma * dloss + beta * kld.mean() + zeta * zinb_loss + l1_loss).backward()
                                nn.utils.clip_grad_norm_(ae.parameters(), max_norm=1)
                                optimizer_ae.step()
                            # self.prune_neurons(ae, threshold=params['prune_threshold'])
                            # If prune is True, prune the model
                        if params['prune_threshold'] > 0:
                            n_neurons = ae.prune_model_paperwise(False, False, weight_threshold=params['prune_threshold'])
                            # If save neptune is True, save the model
                            if self.log_neptune:
                                log_num_neurons(run, n_neurons, init_n_neurons)

                    else:
                        ae = self.freeze_ae(ae)

                    if np.mean(traces['rec_loss']) < best_loss:
                        # "Every counters go to 0 when a better reconstruction loss is reached"
                        print(
                            f"Best Loss Epoch {epoch}, Losses: {np.mean(traces['rec_loss'])}, "
                            f"Domain Losses: {np.mean(traces['dom_loss'])}, "
                            f"Domain Accuracy: {np.mean(traces['dom_acc'])}")
                        warmup_counter = 0
                        # early_stop_counter = 0
                        best_loss = np.mean(traces['rec_loss'])
                        dom_loss = np.mean(traces['dom_loss'])
                        dom_acc = np.mean(traces['dom_acc'])
                        if warmup:
                            torch.save(ae.state_dict(), f'{self.complete_log_path}/warmup.pth')

                    if (
                            self.args.early_warmup_stop != 0 and warmup_counter == self.args.early_warmup_stop) and warmup:  # or warmup_counter == 100:
                        # When the warnup counter gets to
                        values = log_traces(traces, values)
                        if self.args.early_warmup_stop != 0:
                            try:
                                ae.load_state_dict(torch.load(f'{self.complete_log_path}/model_{h}.pth'))
                            except:
                                pass
                        print(f"\n\nWARMUP FINISHED (early stop). {epoch}\n\n")
                        warmup = False
                        warmup_disc_b = True

                    if epoch == self.args.warmup and warmup:  # or warmup_counter == 100:
                        # When the warnup counter gets to
                        if self.args.early_warmup_stop != 0:
                            try:
                                ae.load_state_dict(torch.load(f'{self.complete_log_path}/model_{h}.pth'))
                            except:
                                pass
                        print(f"\n\nWARMUP FINISHED. {epoch}\n\n")
                        values = log_traces(traces, values)
                        warmup = False
                        warmup_disc_b = True

                    if epoch < self.args.warmup and warmup:  # and np.mean(traces['rec_loss']) >= best_loss:
                        values = log_traces(traces, values)
                        warmup_counter += 1
                        # best_values = get_best_values(traces, ae_only=True)
                        if self.log_tb:
                            loggers['tb_logging'].logging(values, metrics)
                        if self.log_mlflow:
                            add_to_mlflow(values, epoch)
                        continue
                    ae.train()
                    if self.args.bdisc:
                        self.forward_discriminate(optimizer_b, ae, celoss, loaders['all'])
                    if warmup_disc_b and warmup_b_counter < 0:
                        warmup_b_counter += 1
                        continue
                    else:
                        warmup_disc_b = False

                    # End-of-epoch update (skip during warmup if set)
                    if self.args.update_grid and epoch >= self.args.update_grid_warmup:
                        updated = model.update_grids()
                        print(f"[epoch {epoch}] Updated {updated} KAN grids")

                # If training of the autoencoder is retricted to the warmup, (train_after_warmup=0),
                # all layers except the classification layers are frozen
            if self.args.train_after_warmup == 0:

                ae = self.freeze_ae(ae)
                ae.eval()
                ae.classifier.train()
            # ae.classifier.random_init()
            early_stop_counter = 0
            for epoch in range(0, self.args.n_epochs):
                if early_stop_counter == self.args.early_stop:
                    if self.verbose > 0:
                        print('EARLY STOPPING.', epoch)
                    break
                lists, traces = get_empty_traces()
                losses = {
                    "mseloss": mseloss,
                    "celoss": sceloss,
                }
                closs = self.loop_train('train', optimizer_c, ae, None, losses, loaders['train'], lists, traces, nu=nu)

                if torch.isnan(closs):
                    if self.log_mlflow:
                        mlflow.log_param('finished', 0)
                        mlflow.end_run()
                    return best_loss

                # Below is the loop for all sets
                with torch.no_grad():
                    for group in list(data['inputs'].keys()):
                        if group in ['all', 'all_pool']:
                            continue
                        closs, lists, traces = self.loop(group, optimizer_c, ae, sceloss, loaders[group], lists, traces, nu=0)
                    closs, _, _ = self.loop('train', optimizer_ae, ae, sceloss,
                                            loaders['train'], lists, traces, nu=nu)

                traces = self.get_mccs(lists, traces)
                values = log_traces(traces, values)
                if self.log_tb:
                    try:
                        add_to_logger(values, loggers['logger'], epoch)
                    except:
                        print("Problem with add_to_logger!")
                if self.log_mlflow:
                    add_to_mlflow(values, epoch)
                if np.mean(values['valid']['mcc'][-self.args.n_agg:]) > best_mcc and len(
                        values['valid']['mcc']) > self.args.n_agg:
                    print(f"Best Classification Mcc Epoch {epoch}, "
                          f"Acc: {values['test']['acc'][-1]}"
                          f"Mcc: {values['test']['mcc'][-1]}"
                          f"Classification train loss: {values['train']['closs'][-1]},"
                          f" valid loss: {values['valid']['closs'][-1]},"
                          f" test loss: {values['test']['closs'][-1]}")
                    best_mcc = np.mean(values['valid']['mcc'][-self.args.n_agg:])
                    torch.save(ae.state_dict(), f'{self.complete_log_path}/model_{h}.pth')
                    best_values = get_best_values(values.copy(), ae_only=False, n_agg=self.args.n_agg)
                    best_vals = values.copy()
                    best_vals['rec_loss'] = best_loss
                    best_vals['dom_loss'] = best_dom_loss
                    best_vals['dom_acc'] = best_dom_acc
                    early_stop_counter = 0

                if values['valid']['acc'][-1] > best_acc:
                    print(f"Best Classification Acc Epoch {epoch}, "
                          f"Acc: {values['test']['acc'][-1]}"
                          f"Mcc: {values['test']['mcc'][-1]}"
                          f"Classification train loss: {values['train']['closs'][-1]},"
                          f" valid loss: {values['valid']['closs'][-1]},"
                          f" test loss: {values['test']['closs'][-1]}")

                    best_acc = values['valid']['acc'][-1]
                    early_stop_counter = 0

                if values['valid']['closs'][-1] < best_closs:
                    print(f"Best Classification Loss Epoch {epoch}, "
                          f"Acc: {values['test']['acc'][-1]} "
                          f"Mcc: {values['test']['mcc'][-1]} "
                          f"Classification train loss: {values['train']['closs'][-1]}, "
                          f"valid loss: {values['valid']['closs'][-1]}, "
                          f"test loss: {values['test']['closs'][-1]}")
                    best_closs = values['valid']['closs'][-1]
                    early_stop_counter = 0
                else:
                    # if epoch > self.warmup:
                    early_stop_counter += 1

                if self.args.predict_tests and (epoch % 10 == 0):
                    loaders = get_loaders(self.data, data, self.args.random_recs, self.args.triplet_dloss, ae,
                                          ae.classifier)

                if params['prune_threshold'] > 0 and self.args.kan == 1:
                    n_neurons = ae.prune_model_paperwise(True, is_dann, weight_threshold=params['prune_threshold'])
                    # If save neptune is True, save the model
                    if self.log_neptune:
                        log_num_neurons(run, n_neurons, init_n_neurons)
                # End-of-epoch update (skip during warmup if set)
                if self.args.update_grid and epoch >= self.args.update_grid_warmup:
                    updated = ae.update_grids()
                    print(f"[epoch {epoch}] Updated {updated} KAN grids")

            best_mccs += [best_mcc]

            # Running the loop one last time to register the reconstructions without batch effects.
            # In the previous loop, when mapping=True, the reconstructions have batch effects to make
            # The reconstructions more accurate. This is necessary when we want to get batch-free reconstructions
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
                    closs, best_lists, traces = self.loop(group, optimizer_c, ae, sceloss, loaders[group], best_lists, traces, nu=0,
                                                          mapping=False)  # -1
            if self.log_neptune:
                run["model"].upload(f'{self.complete_log_path}/model_{h}.pth')
                run["validation/closs"].log(best_closs)
            best_closses += [best_closs]
            # logs things in the background. This could be problematic if the logging takes more time than each iteration of repetitive holdout
            # daemon = Thread(target=self.log_rep, daemon=True, name='Monitor',
            #                 args=[best_lists, best_vals, best_values, traces, model, metrics, run, cm_logger, ae,
            #                       shap_ae, h,
            #                       epoch])
            # daemon.start()
            self.log_rep(best_lists, best_vals, best_values, traces, metrics, run, loggers, ae,
                         shap_ae, h, epoch)
            self.ae_instance = ae
            del ae, shap_ae

        # Logging every model is taking too much resources and it makes it quite complicated to get information when
        # Too many runs have been made. This will make the notebook so much easier to work with
        if np.mean(best_mccs) > self.best_mcc:
            try:
                if os.path.exists(
                        f'logs/best_models/ae_then_classifier_holdout/{self.args.dataset}/{self.args.dloss}_vae{self.args.variational}'):
                    shutil.rmtree(
                        f'logs/best_models/ae_then_classifier_holdout/{self.args.dataset}/{self.args.dloss}_vae{self.args.variational}',
                        ignore_errors=True)
                # os.makedirs(f'logs/best_models/ae_classifier_holdout/{self.args.dloss}_vae{self.args.variational}', exist_ok=True)
                shutil.copytree(f'{self.complete_log_path}',
                                f'logs/best_models/ae_then_classifier_holdout/{self.args.dataset}/{self.args.dloss}_vae{self.args.variational}')
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


def main():
    import runpy

    runpy.run_module("bernn.dl.train.train_ae_then_classifier_holdout", run_name="__main__")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--random_recs', type=int, default=0)  # TODO to deprecate, no longer used
    parser.add_argument('--predict_tests', type=int, default=0)
    # parser.add_argument('--balanced_rec_loader', type=int, default=0)
    parser.add_argument('--early_stop', type=int, default=50)
    parser.add_argument('--early_warmup_stop', type=int, default=-1)
    parser.add_argument('--train_after_warmup', type=int, default=0)
    parser.add_argument('--threshold', type=float, default=0.)
    parser.add_argument('--n_epochs', type=int, default=1000)
    parser.add_argument('--n_trials', type=int, default=100)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--rec_loss', type=str, default='l1')
    parser.add_argument('--tied_weights', type=int, default=0)
    parser.add_argument('--random', type=int, default=1)
    parser.add_argument('--variational', type=int, default=0)
    parser.add_argument('--zinb', type=int, default=0) # TODO resolve problems, do not use
    parser.add_argument('--use_mapping', type=int, default=1, help="Use batch mapping for reconstruct")
    parser.add_argument('--bdisc', type=int, default=1)
    parser.add_argument('--n_repeats', type=int, default=5)
    parser.add_argument('--dloss', type=str, default='inverseTriplet')  # one of revDANN, DANN, inverseTriplet, revTriplet
    parser.add_argument('--csv_file', type=str, default='unique_genes.csv', help='')
    parser.add_argument('--bad_batches', type=str, default='', help='0;23;22;21;20;19;18;17;16;15')
    parser.add_argument('--remove_zeros', type=int, default=0)
    parser.add_argument('--n_meta', type=int, default=0)
    parser.add_argument('--embeddings_meta', type=int, default=0)
    parser.add_argument('--groupkfold', type=int, default=1)
    parser.add_argument('--dataset', type=str, default='custom')
    parser.add_argument('--bs', type=int, default=32, help='Batch size')
    parser.add_argument('--path', type=str, default='./data/', help='Directory containing the CSV file (CLI only)')
    parser.add_argument('--exp_id', type=str, default='default_ae_then_classifier')
    parser.add_argument('--strategy', type=str, default='CU_DEM', help='only for alzheimer dataset')
    parser.add_argument('--n_agg', type=int, default=5, help='Number of trailing values to get stable valid values')
    parser.add_argument('--n_layers', type=int, default=2, help='N layers for classifier')
    parser.add_argument('--log1p', type=int, default=1, help='log1p the data? Should be 0 with zinb')
    parser.add_argument('--pool', type=int, default=1, help='only for alzheimer dataset')
    parser.add_argument('--kan', type=int, default=1, help='')
    parser.add_argument('--update_grid', type=int, default=1, help='')
    parser.add_argument('--use_l1', type=int, default=1, help='')
    parser.add_argument('--clip_val', type=float, default=1, help='')
    parser.add_argument('--log_metrics', type=int, default=1, help='')
    parser.add_argument('--log_plots', type=int, default=1, help='')
    parser.add_argument('--log_inputs', type=int, default=0, help='')
    parser.add_argument('--prune_network', type=float, default=1, help='')
    parser.add_argument('--log_neptune', type=int, default=0, help='Deprecated and ignored. Use --log_mlflow=1.')
    parser.add_argument('--log_mlflow', type=int, default=1, help='Enable MLflow logging (recommended).')
    parser.add_argument('--log_tb', type=int, default=0, help='')
    parser.add_argument('--keep_models', type=int, default=0, help='')
    parser.add_argument('--update_grid_warmup', type=int, default=5, help='Update grid after warmup?')

    args = parser.parse_args()

    if args.kan == 0:
        args.update_grid = 0
        args.update_grid_warmup = 0

    # Example usage showing different approaches:

    # 1. RECOMMENDED: Modern approach with TrainingConfig
    print("=== Modern Configuration Class Approach ===")
    config = TrainingConfig(
        csv_file=args.csv_file,
        exp_id='modern_ae_then_classifier',
        dloss=args.dloss,
        variational=args.variational,
        n_epochs=args.n_epochs,
        device=args.device,
        groupkfold=args.groupkfold,
        n_meta=args.n_meta,
        embeddings_meta=args.embeddings_meta,
        n_layers=args.n_layers,
        n_agg=args.n_agg,
        bad_batches=args.bad_batches,
        remove_zeros=args.remove_zeros,
        dataset=args.dataset,
        kan=args.kan,
        # path=args.path,
        bs=args.bs,
        strategy=args.strategy,
        log1p=args.log1p,
        pool=args.pool,
        prune_network=args.prune_network,
        update_grid=args.update_grid,
        use_l1=args.use_l1,
        clip_val=args.clip_val,
    )

    try:
        mlflow.create_experiment(config.exp_id)
    except:
        print(f"Experiment {config.exp_id} already exists")

    # Clean, type-safe instantiation
    trainer_modern = TrainAEThenClassifierHoldout(
        config=config,
        log_metrics=args.log_metrics,
        keep_models=args.keep_models,
        log_inputs=args.log_inputs,
        log_plots=args.log_plots,
        log_tb=args.log_tb,
        log_neptune=args.log_neptune,
        log_mlflow=args.log_mlflow,
        pools=False  # TODO redundancy with args.pool
    )

    # Use the modern trainer for the actual experiment
    train = trainer_modern

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
    if groups is not None:
        X_train, X_test, y_train, y_test, g_train, g_test = train_test_split(
            X, y, groups, test_size=0.2, random_state=41, stratify=y
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=41, stratify=y
        )
        g_train, g_test = None, None
    
    print("Data loaded and split for optimization.")
    # No need to call fit() here, ax_eval will call it for each trial.

    # List of hyperparameters getting optimized
    parameters = [
        {"name": "nu", "type": "range", "bounds": [1e-4, 1e2], "log_scale": False},
        {"name": "lr", "type": "range", "bounds": [1e-4, 1e-2], "log_scale": True},
        {"name": "wd", "type": "range", "bounds": [1e-6, 1e-3], "log_scale": True},
        # {"name": "l1", "type": "range", "bounds": [1e-8, 1e-5], "log_scale": True},
        # {"name": "lr_b", "type": "range", "bounds": [1e-6, 1e-1], "log_scale": True},
        # {"name": "wd_b", "type": "range", "bounds": [1e-8, 1e-5], "log_scale": True},
        {"name": "smoothing", "type": "range", "bounds": [0., 0.2]},
        {"name": "margin", "type": "range", "bounds": [0., 10.]},
        {"name": "warmup", "type": "range", "bounds": [100, 1000]},
        # {"name": "disc_b_warmup", "type": "range", "bounds": [1, 2]},

        {"name": "dropout", "type": "range", "bounds": [0.0, 0.5]},
        # {"name": "ncols", "type": "range", "bounds": [20, 10000]},
        {"name": "scaler", "type": "choice",
         "values": ['standard_per_batch', 'standard', 'robust', 'robust_per_batch']},  # scaler whould be no for zinb
        # {"name": "layer3", "type": "range", "bounds": [32, 512]},
        {"name": "layer2", "type": "range", "bounds": [32, 512]},
        {"name": "layer1", "type": "range", "bounds": [512, 1024]},
        # {"name": "layer2", "type": "range", "bounds": [32, 64]},
        # {"name": "layer1", "type": "range", "bounds": [64, 128]},

    ]

    # Some hyperparameters are not always required. They are set to a default value in Train.train()
    if train.config.dloss in ['revTriplet', 'revDANN', 'DANN', 'inverseTriplet', 'normae']:
        # gamma = 0 will ensure DANN is not learned
        parameters += [{"name": "gamma", "type": "range", "bounds": [1e-2, 1e2], "log_scale": True}]
    if train.config.variational:
        # beta = 0 because useless outside a variational autoencoder
        parameters += [{"name": "beta", "type": "range", "bounds": [1e-2, 1e2], "log_scale": True}]
    if train.config.zinb:
        # zeta = 0 because useless outside a zinb autoencoder
        parameters += [{"name": "zeta", "type": "range", "bounds": [1e-2, 1e2], "log_scale": True}]
    if train.config.kan and train.config.use_l1:
        # reg_entropy for KAN regularization
        parameters += [{"name": "reg_entropy", "type": "range", "bounds": [1e-4, 1e-2], "log_scale": True}]
    if train.config.use_l1:
        parameters += [{"name": "l1", "type": "range", "bounds": [1e-4, 1e-2], "log_scale": True}]
    if train.config.prune_network:
        parameters += [{"name": "prune_threshold", "type": "range", "bounds": [1e-3, 3e-3], "log_scale": True}]

    def ax_eval(parameterization):
        return train.fit_transform(X_train, y_train, X_test, y_test, groups_train=g_train, groups_test=g_test, params=parameterization)

    best_parameters, values, experiment, model = optimize(
        parameters=parameters,
        evaluation_function=ax_eval,
        objective_name='mcc',
        minimize=False,
        total_trials=train.config.n_trials,
        random_seed=41,
    )


    # Example of how to access results
    print("=== Optimization Results ===")
    print(f'Best MCC: {values[0]["mcc"]}')
    print('Best Parameters:')
    for param, value in best_parameters.items():
        print(f'  {param}: {value}')

    # fig = plt.figure()
    # render(plot_contour(model=model, param_x="learning_rate", param_y="weight_decay", metric_name='Loss'))
    # fig.savefig('test.jpg')
