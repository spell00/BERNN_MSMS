import matplotlib
from bernn.utils.pool_metrics import log_pool_metrics
import pandas as pd
import numpy as np
import random
import torch
from torch import nn
from sklearn import metrics
import contextlib

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

from sklearn.metrics import matthews_corrcoef as MCC
from sklearn.neighbors import KNeighborsClassifier
# from ...ml.train.params_gp import *
from ..models.pytorch.aedann import ReverseLayerF
from ..models.pytorch.aeekandann import KANAutoEncoder2
from ..models.pytorch.ekan.src.efficient_kan.kan import KANLinear
from ..models.pytorch.utils.loggings import log_metrics, \
    log_plots, log_neptune, log_shap, log_mlflow
from bernn.utils.utils import to_csv
from ..models.pytorch.utils.utils import to_categorical, get_empty_traces, \
    log_traces, add_to_mlflow
from ..models.pytorch.utils.loggings import make_data
import warnings
from bernn.utils.data_getters import get_alzheimer, get_amide, get_mice, get_data, get_dummy
import uuid

matplotlib.use('Agg')
CUDA_VISIBLE_DEVICES = ""

warnings.filterwarnings("ignore")

random.seed(1)
torch.manual_seed(1)
np.random.seed(1)


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


class TrainAE:

    def __init__(self, args, path, fix_thres=-1, load_tb=False, log_metrics=False, keep_models=True, log_inputs=True,
                 log_plots=False, log_tb=False, log_neptune=False, log_mlflow=True, groupkfold=True, pools=True):
        """

        Args:
            args: contains multiple arguments passed in the command line
            log_path (str): Path where the tensorboard logs are saved
            path (str): Path to the data (in .csv format)
            fix_thres (float): If 1 > fix_thres >= 0 then the threshold is fixed to that value.
                       any other value means the threshold won't be fixed and will be
                       learned as an hyperparameter
            load_tb (bool): If True, loads previous runs already saved
            log_metrics (bool): Wether or not to keep the batch effect metrics
            keep_models (bool): Wether or not to save the models trained
                                (can take a lot of space if training a lot of models)
            log_inputs (bool): Wether or not to log graphs or batch effect metrics
                                of the scaled inputs
            log_plots (bool): For each optimization iteration, on the first iteration, wether or
                              not to plot PCA, UMAP, CCA and LDA of the encoded and reconstructed
                              representations.
            log_tb (bool): Wether or not to use tensorboard.
            log_mlflow (bool): Wether or not to use mlflow.

        """
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
        self.pools = pools
        self.default_params()
        self.args = self.fill_missing_params_with_default(args)
        self.load_autoencoder()
        # Persistent KNN for triplet mode
        self._knn_ready = False
        # Initialize KNN with configured number of neighbors
        try:
            n_neighbors = int(getattr(self.args, 'knn_n_neighbors', 5))
        except Exception:
            n_neighbors = 5
        self.knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights='distance')

    # Back-compat wrappers (old names)
    def loop(self, group, optimizer, ae, celoss, loader, 
             lists, traces, nu=1, mapping=True):
        return self.loop_infer(group, optimizer, ae, celoss, loader, lists, traces, nu, mapping)

    def loop2(self, group, optimizer, ae, scheduler, losses, loader, lists, traces, nu=1, mapping=True):
        return self.loop_train(group, optimizer, ae, scheduler, losses, loader, lists, traces, nu, mapping)

    def make_params(self, params):
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
        if not self.args.kan or not self.args.use_l1:
            params['reg_entropy'] = 0
        if not self.args.use_l1:
            params['l1'] = 0
        if not self.args.prune_network:
            params['prune_threshold'] = 0
        if params['prune_threshold'] > 0:
            params['dropout'] = 0

        print(params)

        # Assigns the hyperparameters getting optimized
        # scale = params['scaler']
        self.gamma = params['gamma']
        self.beta = params['beta']
        self.zeta = params['zeta']
        self.l1 = params['l1']
        self.reg_entropy = params['reg_entropy']
        self.args.scaler = params['scaler']
        self.args.warmup = params['warmup']
        self.args.disc_b_warmup = params['disc_b_warmup']
        if 'triplet_margin' in params:
            self.triplet_margin = float(params['triplet_margin'])
        else:
            self.triplet_margin = 0.
        # KNN neighbors (for triplet prediction). Rebuild KNN if provided by HPO
        try:
            if 'knn_n_neighbors' in params:
                self.args.knn_n_neighbors = int(params['knn_n_neighbors'])
                self.knn = KNeighborsClassifier(n_neighbors=self.args.knn_n_neighbors, weights='distance')
        except Exception as e:
            print(f"Warning: couldn't set knn_n_neighbors from params: {e}")
        self.foldername = str(uuid.uuid4())
        self.complete_log_path = f'logs/ae_classifier_holdout/{self.foldername}'
        self.hparams_filepath = self.complete_log_path + '/hp'

        return params

    def get_data(self, seed):
        """
        Splits the data into train, valid and test sets
        """
        if self.args.dataset == 'alzheimer':
            self.data, self.unique_labels, self.unique_batches = get_alzheimer(self.path, self.args, seed=seed)
            self.pools = True
        elif self.args.dataset in ['amide', 'adenocarcinoma']:
            self.data, self.unique_labels, self.unique_batches = get_amide(self.path, self.args, seed=seed)
            self.pools = True

        elif self.args.dataset == 'mice':
            # This seed split the data to have n_samples in train: 96, valid:52, test: 23
            self.data, self.unique_labels, self.unique_batches = get_mice(self.path, self.args, seed=seed)
        elif self.args.dataset == 'dummy':
            self.data, self.unique_labels, self.unique_batches = get_dummy(self.args, seed=seed)
        else:
            self.data, self.unique_labels, self.unique_batches = get_data(self.path, self.args, seed=seed)
            self.pools = self.args.pool
        if self.args.best_features_file != '':
            self.data = keep_top_features(self.data, self.path, self.args)
        if self.args.controls != '':
            self.data = binarize_labels(self.data, self.args.controls)
            self.unique_labels = np.unique(self.data['labels']['all'])

        # Move n_move_test samples from test to train, and n_move_valid from valid to train, randomly
        n_move_test = getattr(self.args, 'n_move_test', 0)
        n_move_valid = getattr(self.args, 'n_move_valid', 0)
        move_keys = ['inputs', 'meta', 'labels', 'names', 'batches', 'cats']
        # Move from test to train
        if n_move_test > 0 and len(self.data['inputs']['test']) > 0:
            idxs = np.random.choice(np.arange(len(self.data['inputs']['test'])), size=min(n_move_test, len(self.data['inputs']['test'])), replace=False)
            for key in move_keys:
                if key in self.data and 'test' in self.data[key] and 'train' in self.data[key]:
                    if hasattr(self.data[key]['test'], 'iloc'):
                        samples = self.data[key]['valid'].iloc[idxs]
                        self.data[key]['train'] = pd.concat([self.data[key]['train'], samples])
                        # Remove by position, not by index value
                        mask = np.ones(len(self.data[key]['test']), dtype=bool)
                        mask[idxs] = False
                        self.data[key]['test'] = self.data[key]['test'].iloc[mask]
                    else:
                        samples = self.data[key]['test'][idxs]
                        self.data[key]['train'] = np.concatenate([self.data[key]['train'], samples])
                        self.data[key]['test'] = np.delete(self.data[key]['test'], idxs, axis=0)

        # Move from valid to train
        if n_move_valid > 0 and len(self.data['inputs']['valid']) > 0:
            idxs = np.random.choice(np.arange(len(self.data['inputs']['valid'])), size=min(n_move_valid, len(self.data['inputs']['valid'])), replace=False)
            for key in move_keys:
                if key in self.data and 'valid' in self.data[key] and 'train' in self.data[key]:
                    if hasattr(self.data[key]['valid'], 'iloc'):
                        samples = self.data[key]['valid'].iloc[idxs]
                        self.data[key]['train'] = pd.concat([self.data[key]['train'], samples])
                        # Remove by position, not by index value
                        mask = np.ones(len(self.data[key]['valid']), dtype=bool)
                        mask[idxs] = False
                        self.data[key]['valid'] = self.data[key]['valid'].iloc[mask]
                    else:
                        samples = self.data[key]['valid'][idxs]
                        self.data[key]['train'] = np.concatenate([self.data[key]['train'], samples])
                        self.data[key]['valid'] = np.delete(self.data[key]['valid'], idxs, axis=0)

    def autocast_context(self):
        """Create autocast context for mixed precision training with bfloat16."""
        try:
            # Try to create autocast context with bfloat16
            return torch.autocast(device_type='cuda', dtype=torch.bfloat16)
        except (AttributeError, RuntimeError):
            # Fallback to null context if autocast or bfloat16 is not supported
            return contextlib.nullcontext()

    def default_params(self):
        """Initialize default parameters for the training process."""
        self.all_params = {
            'controls': '',
            'random_recs': 0,
            'predict_tests': 0,
            'early_stop': 50,
            'early_warmup_stop': -1,
            'train_after_warmup': 0,
            'threshold': 0.,
            'n_epochs': 1000,
            'n_trials': 100,
            'device': 'cuda:0',
            'rec_loss': 'l1',
            # Classification loss selector: 'ce' or 'triplet'
            'classif_loss': 'ce',
            # Margin for TripletMarginLoss when classif_loss='triplet'
            'triplet_margin': 1.0,
            'tied_weights': 0,
            'random': 1,
            'variational': 0,
            'zinb': 0,  # TODO resolve problems, do not use
            'use_mapping': 1,
            'bdisc': 1,
            'n_repeats': 5,
            'dloss': 'inverseTriplet',  # one of revDANN, DANN, inverseTriplet, revTriplet
            'csv_file': 'unique_genes.csv',
            'best_features_file': '',  # best_unique_genes.tsv
            'bad_batches': '',  # 0;23;22;21;20;19;18;17;16;15
            'remove_zeros': 0,
            'n_meta': 0,
            'embeddings_meta': 0,
            'groupkfold': 1,
            'dataset': 'custom',
            'bs': 32,
            'path': './data/',
            'exp_id': 'default_ae_then_classifier',
            'strategy': 'CU_DEM',  # only for alzheimer dataset
            'n_agg': 1,  # Number of trailing values to get stable valid values
            'n_layers': 2,  # N layers for classifier
            'log1p': 1,  # log1p the data? Should be 0 with zinb
            'pool': 1,  # only for alzheimer dataset
            'kan': 1,
            'update_grid': 1,
            'use_l1': 1,
            'clip_val': 1,
            'log_metrics': 1,
            'log_plots': 1,
            'prune_network': 1,
            'prune_threshold': 0,  # Threshold for pruning the network
            'precision': 'bf16',  # Mixed precision training type
            'dropout': 0,  # Dropout rate for the network
            'use_sigmoid': 0,  # Use sigmoid activation in the last layer of the AE
            'scaler': 'standard',  # Set during training
            'warmup': 100,  # Set during training
            'disc_b_warmup': 0,  # Set during training
            'knn_n_neighbors': 5,  # K for persistent KNN used in triplet mode
            'n_move_test': 0,  # Number of test samples to move to train
            'n_move_valid': 0,  # Number of valid samples to move to train
            # 'hparams_filepath': '',  # Path to save hyperparameters
            # 'foldername': '',  # Unique folder name for the run
            # 'complete_log_path': '',  # Complete path for logging 

        }

    def fill_missing_params_with_default(self, params):
        """
        Fill missing parameters with default values.

        Args:
            params: An argparse.Namespace object containing parameters.

        Returns:
            argparse.Namespace: Updated namespace with default values for missing parameters.
        """
        # Convert params to dict if it's a Namespace object
        params_dict = vars(params) if hasattr(params, '__dict__') else params

        # Create a new dict with default values
        updated_params = {}

        # First copy all default values
        for param, default_value in self.all_params.items():
            updated_params[param] = default_value

        # Then override with provided values
        for param, value in params_dict.items():
            if param in self.all_params:
                updated_params[param] = value

        # Convert back to Namespace if input was Namespace
        if hasattr(params, '__dict__'):
            for key, value in updated_params.items():
                setattr(params, key, value)
            return params
        else:
            return updated_params

    def make_samples_weights(self):
        self.n_batches = len(set(np.concatenate([
            v for v in self.data['batches'].values()
            if isinstance(v, (list, np.ndarray))
        ])))

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
            except Exception as e:
                print(f"Error loading names: {e}")
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

    def load_autoencoder(self):
        if not self.args.kan:
            from bernn import AutoEncoder3 as AutoEncoder
            from bernn import SHAPAutoEncoder3 as SHAPAutoEncoder
        elif self.args.kan == 1:
            from bernn import KANAutoEncoder3 as AutoEncoder
            from bernn import SHAPKANAutoEncoder3 as SHAPAutoEncoder
        self.ae = AutoEncoder
        self.shap_ae = SHAPAutoEncoder

    def log_rep(self, best_lists, best_vals, best_values, traces, metrics, run, loggers, ae, shap_ae, h,
                epoch):
        # best_traces = self.get_mccs(best_lists, traces)

        self.log_predictions(best_lists, run, h)

        if self.log_metrics:
            if self.log_tb and self.log_metrics:
                try:
                    # logger, lists, values, model, unique_labels, mlops, epoch, metrics, n_meta_emb=0, device='cuda'
                    metrics = log_metrics(loggers['logger'], best_lists, best_vals, ae,
                                          np.unique(np.concatenate(best_lists['train']['labels'])),
                                          np.unique(self.data['batches']), epoch, mlops="tensorboard",
                                          metrics=metrics, n_meta_emb=self.args.embeddings_meta,
                                          device=self.args.device)
                except BrokenPipeError:
                    print("\n\n\nProblem with logging stuff!\n\n\n")
            if self.log_neptune and self.log_metrics:
                try:
                    metrics = log_metrics(run, best_lists, best_vals, ae,
                                          np.unique(np.concatenate(best_lists['train']['labels'])),
                                          np.unique(self.data['batches']), epoch=epoch, mlops="neptune",
                                          metrics=metrics, n_meta_emb=self.args.embeddings_meta,
                                          device=self.args.device)
                except BrokenPipeError:
                    print("\n\n\nProblem with logging stuff!\n\n\n")
            if self.log_mlflow and self.log_metrics:
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
                    # logger.add(loggers['logger_cm'], epoch, best_lists,
                    #            self.unique_labels, best_traces, 'tensorboard')
                    log_plots(loggers['logger_cm'], best_lists, 'tensorboard', epoch)
                    log_shap(loggers['logger_cm'], shap_ae, best_lists, self.columns, self.args.embeddings_meta, 'tb',
                             self.complete_log_path, self.args.device)
                if self.log_neptune:
                    log_shap(run, shap_ae, best_lists, self.columns, self.args.embeddings_meta, 'neptune',
                             self.complete_log_path, self.args.device)
                    log_plots(run, best_lists, 'neptune', epoch)
                if self.log_mlflow:
                    log_shap(None, shap_ae, best_lists, self.columns, self.args.embeddings_meta, 'mlflow',
                             self.complete_log_path, self.args.device)
                    log_plots(None, best_lists, 'mlflow', epoch)

        columns = list(self.data['inputs']['all'].columns)
        if self.args.n_meta == 2:
            columns += ['gender', 'age']

        rec_data, enc_data = to_csv(best_lists, self.complete_log_path, columns)

        if self.log_neptune:
            run["recs"].track_files(f'{self.complete_log_path}/recs.csv')
            run["encs"].track_files(f'{self.complete_log_path}/encs.csv')

        best_values['pool_metrics'] = {}
        try:
            best_values['batches'] = metrics['batches']
        except Exception as e:
            print(f"Error in batches: {e}")
            pass
        try:
            best_values['pool_metrics']['enc'] = metrics['pool_metrics_enc']
        except Exception as e:
            print(f"Error in pool_metrics_enc: {e}")
            pass
        try:
            best_values['pool_metrics']['rec'] = metrics['pool_metrics_rec']
        except Exception as e:
            print(f"Error in pool_metrics_rec: {e}")
            pass

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
            cm_logger.plot(run, 0, self.unique_unique_labels, 'neptune')
            # cm_logger.get_rf_results(run, self.args)
            run.stop()
        if self.log_mlflow:
            cm_logger.plot(None, 0, self.unique_unique_labels, 'mlflow')
            # cm_logger.get_rf_results(run, self.args)
            # mlflow.end_run()
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
                try:
                    run[f'{group}_AUC'] = metrics.roc_auc_score(y_true=cats[group], y_score=scores[group],
                                                            multi_class='ovr')
                except Exception as e:
                    print(f"Error in {group} AUC: {e}")
            if self.log_mlflow:
                try:
                    mlflow.log_metric(f'{group}_AUC',
                                      metrics.roc_auc_score(y_true=cats[group], y_score=scores[group], multi_class='ovr'),
                                      step=step)
                except Exception as e:
                    print(f"Error in {group} AUC: {e}")


    def loop_infer(self, group, optimizer, ae, celoss, loader, 
                   lists, traces, nu=1, mapping=True):
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
                optimizer.zero_grad()
            data, meta_inputs, names, labels, domain, to_rec, not_to_rec, pos_to_rec, neg_to_rec, \
                pos_batch_sample, neg_batch_sample, meta_pos_batch_sample, meta_neg_batch_sample, set = batch
            data = data.to(self.args.device).float()
            meta_inputs = meta_inputs.to(self.args.device).float()
            to_rec = to_rec.to(self.args.device).float()

            # If n_meta > 0, meta data added to inputs
            if self.args.n_meta > 0:
                data = torch.cat((data, meta_inputs), 1)
                to_rec = torch.cat((to_rec, meta_inputs), 1)
            not_to_rec = not_to_rec.to(self.args.device).float()
            
            # Use autocast for mixed precision training
            with self.autocast_context():
                enc, rec, _, kld = ae(data, to_rec, domain, sampling=sampling, mapping=mapping)
                rec = rec['mean']

                if getattr(self.args, 'classif_loss', 'ce') == 'triplet':
                    feats = torch.cat((ae.classifier.net[0](enc), meta_inputs), 1) if self.args.embeddings_meta else enc
                    # Prefer persistent KNN if available, fallback to in-batch KNN
                    X = feats.detach().float().cpu().numpy()
                    try:
                        from sklearn.exceptions import NotFittedError
                        if getattr(self, "_knn_ready", False):
                            proba = self.knn.predict_proba(X)
                            # Map to full number of classes in correct order
                            proba_full = np.zeros((X.shape[0], self.n_cats), dtype=np.float32)
                            cls_idx = np.array(self.knn.classes_, dtype=int)
                            proba_full[:, cls_idx] = proba.astype(np.float32)
                        else:
                            raise NotFittedError("Persistent KNN not ready")
                    except Exception as e:
                        # Fallback: in-batch KNN using neighbors within the current batch
                        from sklearn.neighbors import NearestNeighbors
                        k = int(getattr(self.args, 'knn_n_neighbors', 5))
                        y_np = labels.detach().int().cpu().numpy()
                        nns = NearestNeighbors(n_neighbors=min(k, len(X)), metric='minkowski')
                        nns.fit(X)
                        idx = nns.kneighbors(X, return_distance=False)
                        proba_full = np.zeros((X.shape[0], self.n_cats), dtype=np.float32)
                        for i in range(X.shape[0]):
                            counts = np.bincount(y_np[idx[i]], minlength=self.n_cats).astype(np.float32)
                            s = counts.sum()
                            proba_full[i] = counts / s if s > 0 else np.full(self.n_cats, 1.0 / self.n_cats, dtype=np.float32)
                    preds = torch.from_numpy(proba_full).to(self.args.device)
                else:
                    if self.args.embeddings_meta:
                        preds = ae.classifier(torch.cat((enc, meta_inputs), 1))
                    else:
                        preds = ae.classifier(enc)

                domain_preds = ae.dann_discriminator(enc)
            # Build one-hot labels for metrics and CE mode
            if torch.all(labels < self.n_cats):
                cats = to_categorical(labels.long(), self.n_cats).to(self.args.device).float()
            else:
                # Fallback if labels out of bounds
                cats = torch.zeros((labels.shape[0], self.n_cats), device=self.args.device)
                cats[:, 0] = 1

            # Select classification loss
            if getattr(self.args, 'classif_loss', 'ce') == 'triplet':
                # Compute embeddings for positive/negative samples and apply TripletMarginLoss on enc
                class_triplet = nn.TripletMarginLoss(getattr(self.args, 'triplet_margin', self.triplet_margin), p=2, swap=True)
                pos_to_rec = pos_to_rec.to(self.args.device).float()
                neg_to_rec = neg_to_rec.to(self.args.device).float()
                mpos_bs = meta_pos_batch_sample.to(self.args.device).float()
                mneg_bs = meta_neg_batch_sample.to(self.args.device).float()
                if self.args.n_meta > 0:
                    pos_to_rec = torch.cat((pos_to_rec, mpos_bs), 1)
                    neg_to_rec = torch.cat((neg_to_rec, mneg_bs), 1)
                pos_enc, _, _, _ = ae(pos_to_rec, pos_to_rec, domain, sampling=True, mapping=mapping)
                neg_enc, _, _, _ = ae(neg_to_rec, neg_to_rec, domain, sampling=True, mapping=mapping)
                if not self.args.train_after_warmup:
                    enc = ae.classifier.net[0](enc)
                    pos_enc = ae.classifier.net[0](pos_enc)
                    neg_enc = ae.classifier.net[0](neg_enc)
                classif_loss = class_triplet(enc, pos_enc, neg_enc)
            else:
                classif_loss = celoss(preds, cats)

            if not self.args.zinb:
                if isinstance(rec, list):
                    rec = rec[-1]
                if isinstance(to_rec, list):
                    to_rec = to_rec[-1]
            lists[group]['set'] += [np.array([group for _ in range(len(domain))])]
            lists[group]['domains'] += [
                np.array([self.unique_batches[d] for d in domain.detach().int().cpu().numpy()])
            ]
            lists[group]['domain_preds'] += [domain_preds.detach().float().cpu().numpy()]
            lists[group]['preds'] += [preds.detach().float().cpu().numpy()]
            lists[group]['classes'] += [labels.detach().int().cpu().numpy()]
            # lists[group]['encoded_values'] += [enc.view(enc.shape[0], -1).detach().float().cpu().numpy()]
            lists[group]['names'] += [names]
            lists[group]['cats'] += [cats.detach().float().cpu().numpy()]
            lists[group]['gender'] += [data.detach().float().cpu().numpy()[:, -1]]
            lists[group]['age'] += [data.detach().float().cpu().numpy()[:, -2]]
            lists[group]['atn'] += [str(x) for x in data.detach().float().cpu().numpy()[:, -5:-2]]
            lists[group]['inputs'] += [data.view(rec.shape[0], -1).detach().float().cpu().numpy()]
            lists[group]['encoded_values'] += [enc.detach().float().cpu().numpy()]
            lists[group]['rec_values'] += [rec.detach().float().cpu().numpy()]
            try:
                lists[group]['labels'] += [np.array(
                    [self.unique_labels[x] for x in labels.detach().int().cpu().numpy()])]
            except Exception as e:
                print(f"Error in labels: {e}")
                pass
            traces[group]['acc'] += [np.mean([0 if pred != dom else 1 for pred, dom in
                                              zip(preds.detach().float().cpu().numpy().argmax(1),
                                                  labels.detach().int().cpu().numpy())])]
            traces[group]['top3'] += [np.mean(
                [1 if label.item() in pred.tolist()[::-1][:3] else 0 for pred, label in
                 zip(preds.argsort(1), labels)])]

            traces[group]['closs'] += [classif_loss.item()]
            try:
                traces[group]['mcc'] += [np.round(
                    MCC(labels.detach().int().cpu().numpy(), preds.detach().float().cpu().numpy().argmax(1)), 3)
                ]
            except Exception as e:
                print(f"Error in mcc: {e}")
                traces[group]['mcc'] = []
                traces[group]['mcc'] += [np.round(
                    MCC(labels.detach().int().cpu().numpy(), preds.detach().float().cpu().numpy().argmax(1)), 3)
                ]

            if group in ['train'] and nu != 0:
                # w = np.mean([1/self.class_weights[x] for x in lists[group]['labels'][-1]])
                w = 1
                total_loss = w * nu * classif_loss
                # if self.args.train_after_warmup:
                #     total_loss += rec_loss
                try:
                    total_loss.backward()
                except Exception as e:
                    print(f"Error in total_loss: {e}")
                # nn.utils.clip_grad_norm_(ae.classifier.parameters(), max_norm=1)
                optimizer.step()

        return classif_loss, lists, traces

    def loop_train(self, group, optimizer, ae, scheduler, losses, loader, lists, traces, nu=1, mapping=True):
        """
        Joint training/eval step: classification + reconstruction.

        Args:
            group: 'train' | 'valid' | 'test'
            optimizer: optimizer for classifier (or ae if applicable)
            ae: autoencoder model
            scheduler: LR scheduler (can be None or ReduceLROnPlateau/others)
            losses: dict with {'mseloss': ..., 'celoss': ...}
            loader: DataLoader
            lists: accumulators dict
            traces: metrics traces dict
            nu: weight for classification loss
            mapping: pass-through to AE forward

        Returns:
            classif_loss, lists, traces
        """
        sampling = True if (group in ['train', 'valid'] and nu != 0) else False
        classif_loss = None
        # Collect features/labels to fit persistent KNN after the loop when using triplet
        knn_feats, knn_labels = [], []

        for i, batch in enumerate(loader):
            if group == 'train' and nu != 0:
                optimizer.zero_grad()

            data, meta_inputs, names, labels, domain, to_rec, not_to_rec, pos_to_rec, neg_to_rec, \
                pos_batch_sample, neg_batch_sample, meta_pos_batch_sample, \
                meta_neg_batch_sample, set_name = batch

            data = data.to(self.args.device).float()
            meta_inputs = meta_inputs.to(self.args.device).float()
            to_rec = to_rec.to(self.args.device).float()

            # Concatenate meta to inputs if configured
            if self.args.n_meta > 0:
                data = torch.cat((data, meta_inputs), 1)
                to_rec = torch.cat((to_rec, meta_inputs), 1)

            not_to_rec = not_to_rec.to(self.args.device).float()

            enc, rec, _, kld = ae(data, to_rec, domain, sampling=sampling, mapping=mapping)
            rec = rec['mean']
            if self.args.train_after_warmup:
                rec_loss = losses['mseloss'](rec, to_rec)
            else:
                rec_loss = torch.tensor(0.0, device=self.args.device)

            # Classifier head; for triplet we only collect train features to fit KNN later
            if getattr(self.args, 'classif_loss', 'ce') == 'triplet':
                feats = torch.cat((enc, meta_inputs), 1) if self.args.embeddings_meta else enc
                if group == 'train' and nu != 0:
                    knn_feats.append(feats.detach().float().cpu().numpy())
                    knn_labels.append(labels.detach().int().cpu().numpy())
                preds = None
            else:
                if self.args.embeddings_meta:
                    preds = ae.classifier(torch.cat((enc, meta_inputs), 1))
                else:
                    preds = ae.classifier(enc)

            domain_preds = ae.dann_discriminator(enc)

            # One-hot targets (needed only for CE)
            if getattr(self.args, 'classif_loss', 'ce') != 'triplet':
                if torch.all(labels < self.n_cats):
                    cats = to_categorical(labels.long(), self.n_cats).to(self.args.device).float()
                else:
                    cats = torch.zeros((labels.shape[0], self.n_cats), device=self.args.device)
                    cats[:, 0] = 1
            else:
                cats = None

            # Select classification loss
            if getattr(self.args, 'classif_loss', 'ce') == 'triplet':
                class_triplet = nn.TripletMarginLoss(getattr(self.args, 'triplet_margin', self.triplet_margin), p=2, swap=True)
                pos_to_rec = pos_to_rec.to(self.args.device).float()
                neg_to_rec = neg_to_rec.to(self.args.device).float()
                mpos_bs = meta_pos_batch_sample.to(self.args.device).float()
                mneg_bs = meta_neg_batch_sample.to(self.args.device).float()
                if self.args.n_meta > 0:
                    pos_to_rec = torch.cat((pos_to_rec, mpos_bs), 1)
                    neg_to_rec = torch.cat((neg_to_rec, mneg_bs), 1)
                pos_enc, _, _, _ = ae(pos_to_rec, pos_to_rec, domain, sampling=True, mapping=mapping)
                neg_enc, _, _, _ = ae(neg_to_rec, neg_to_rec, domain, sampling=True, mapping=mapping)
                if not self.args.train_after_warmup:
                    enc = ae.classifier.net[0](enc)
                    pos_enc = ae.classifier.net[0](pos_enc)
                    neg_enc = ae.classifier.net[0](neg_enc)
                classif_loss = class_triplet(enc, pos_enc, neg_enc)
            else:
                classif_loss = losses['celoss'](preds, cats)

            # Handle possible list outputs
            if not self.args.zinb:
                if isinstance(rec, list):
                    rec = rec[-1]
                if isinstance(to_rec, list):
                    to_rec = to_rec[-1]

            # Backprop when training
            if group == 'train' and nu != 0:
                w = 1.0
                total_loss = w * nu * classif_loss + rec_loss
                try:
                    total_loss.backward()
                except Exception as e:
                    print(f"Error in total_loss: {e}")
                optimizer.step()
                if self.args.scheduler is not None and self.args.scheduler != 'ReduceLROnPlateau':
                    scheduler.step()

        # Fit persistent KNN at the end of training loop when using triplet
        if getattr(self.args, 'classif_loss', 'ce') == 'triplet' and group == 'train' and len(knn_feats) > 0:
            try:
                X_all = np.concatenate(knn_feats, axis=0)
                y_all = np.concatenate(knn_labels, axis=0)
                self.knn.fit(X_all, y_all)
                self._knn_ready = True
            except Exception as e:
                print(f"KNN fit failed, falling back to batch KNN: {e}")
                self._knn_ready = False

        return classif_loss

    def forward_discriminate(self, optimizer_b, ae, celoss, loader):
        # Freezing the layers so the batch discriminator can get some knowledge independently
        # from the part where the autoencoder is trained. Only for DANN
        self.freeze_dlayers(ae)
        sampling = True
        for i, batch in enumerate(loader):
            optimizer_b.zero_grad()
            data, meta_inputs, names, labels, domain, to_rec, not_to_rec, pos_to_rec, neg_to_rec, \
                pos_batch_sample, neg_batch_sample, meta_pos_batch_sample, meta_neg_batch_sample, set = batch
            # data[torch.isnan(data)] = 0
            data = data.to(self.args.device).float()
            to_rec = to_rec.to(self.args.device).float()
            meta_inputs = meta_inputs.to(self.args.device).float()
            if self.args.n_meta > 0:
                data = torch.cat((data, meta_inputs), 1)
                to_rec = torch.cat((to_rec, meta_inputs), 1)
            with torch.no_grad():
                enc, rec, _, kld = ae(data, to_rec, domain, sampling=sampling)
            with torch.enable_grad():
                domain_preds = ae.dann_discriminator(enc)

                bclassif_loss = celoss(domain_preds,
                                       to_categorical(domain.long(), self.n_batches).to(self.args.device).float())
                if torch.isnan(bclassif_loss):
                    print("NAN in batch discriminator loss!")
                bclassif_loss.backward()
                # nn.utils.clip_grad_norm_(ae.dann_discriminator.parameters(), max_norm=1)
                optimizer_b.step()
        self.unfreeze_layers(ae)

    def get_dloss(self, celoss, domain, domain_preds, set_num=None):
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
        if self.args.dloss in ['revTriplet', 'revDANN', 'DANN', 'inverseTriplet', 'normae']:
            domain = domain.to(self.args.device).long().to(self.args.device)
            dloss = celoss(domain_preds, domain)
        else:
            dloss = torch.zeros(1)[0].float().to(self.args.device)
        if self.args.dloss == 'normae':
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
        if dloss == 'revTriplet' or dloss == 'inverseTriplet':
            triplet_loss = nn.TripletMarginLoss(margin, p=2, swap=True)
        else:
            triplet_loss = None

        return sceloss, celoss, mseloss, triplet_loss

    def compute_classif_loss(self, enc, preds, labels, celoss, triplet_margin):
            """Compute classification loss as CE or TripletMarginLoss based on args.classif_loss.

            - For 'ce': expects labels as class indices; will build one-hot cats if needed
            - For 'triplet': expects batch to include positive/negative samples; we derive
                triplet from enc (anchor) and encodings of pos/neg built in calling scope.
            """
            if self.args.classif_loss == 'triplet':
                    # Triplet handled in calling scope where pos/neg enc are available
                    # Return None here; caller must pass actual triplet value
                    return None
            # Default to CrossEntropy-style loss (the code uses one-hot 'cats')
            return celoss(preds, labels)

    def freeze_dlayers(self, ae):
        """
        Freeze all layers except the dann classifier
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
            except Exception as e:
                print(f"Error loading preds and classes: {e}")
                pass
            traces[group]['mcc'] = MCC(preds, classes)

        return traces

    def l1_regularization(self, model, lambda_l1):
        l1 = 0
        for p in model.parameters():
            l1 = l1 + p.abs().sum()
        return lambda_l1 * l1

    def reg_kan(self, model, l1, reg_entropy):
        """
        Regularization for KAN
        Args:
            model: Autoencoder model
            l1: L1 regularization
            reg_entropy: Entropy regularization

        Returns:
            l1_loss: Regularization loss
        """
        # Collect all layers dynamically
        layers = []

        # Add encoder layers dynamically
        if hasattr(model.enc, 'kan_layers'):
            layers.extend(model.enc.kan_layers)

        # Add decoder layers dynamically
        if hasattr(model.dec, 'kan_layers'):
            layers.extend(model.dec.kan_layers)

        # Add classifier layers dynamically
        if hasattr(model.classifier, 'linear1'):
            layers.extend([layer for layer in model.classifier.modules() if isinstance(layer, KANLinear)])

        # Add discriminator layers dynamically
        if hasattr(model.dann_discriminator, 'linear1'):
            layers.extend([layer for layer in model.dann_discriminator.modules() if isinstance(layer, KANLinear)])

        # Compute regularization loss for all layers
        l1_loss = sum(layer.regularization_loss(l1, reg_entropy) for layer in layers)

        # Handle NaN values in the regularization loss
        if torch.isnan(l1_loss):
            l1_loss = torch.zeros(1).to(self.args.device)[0]

        return l1_loss

    def warmup_loop(self, optimizer_ae, scheduler, ae, celoss, loader, triplet_loss, mseloss, warmup, epoch,
                    optimizer_b, values, loggers, loaders, run, mapping=True):
        lists, traces = get_empty_traces()
        ae.train()
        ae.mapper.train()

        iterator = enumerate(loader)

        # If option train_after_warmup=1, then this loop is only for preprocessing
        for i, all_batch in iterator:
            # print(i)
            optimizer_ae.zero_grad()
            inputs, meta_inputs, names, labels, domain, to_rec, not_to_rec, pos_to_rec, neg_to_rec, \
                pos_batch_sample, neg_batch_sample, meta_pos_batch_sample, meta_neg_batch_sample, _ = all_batch
            inputs = inputs.to(self.args.device).float()
            meta_inputs = meta_inputs.to(self.args.device).float()
            to_rec = to_rec.to(self.args.device).float()
            # verify if domain is str
            if isinstance(domain, str):
                domain = torch.Tensor([[int(y) for y in x.split("_")] for x in domain])
            if self.args.n_meta > 0:
                inputs = torch.cat((inputs, meta_inputs), 1)
                to_rec = torch.cat((to_rec, meta_inputs), 1)

            enc, rec, zinb_loss, kld = ae(inputs, to_rec, domain, sampling=True, mapping=mapping)
            rec = rec['mean']
            zinb_loss = zinb_loss.to(self.args.device)
            reverse = ReverseLayerF.apply(enc, 1)
            if self.args.dloss == 'DANN':
                domain_preds = ae.dann_discriminator(reverse)
            else:
                domain_preds = ae.dann_discriminator(enc)
            if self.args.dloss not in ['revTriplet', 'inverseTriplet']:
                dloss, domain = self.get_dloss(celoss, domain, domain_preds)
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
                return 0, ae, 0
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
                                           zip(domain_preds.detach().float().cpu().numpy().argmax(1),
                                               domain.detach().int().cpu().numpy())])]
            # lists['all']['set'] += [np.array([group for _ in range(len(domain))])]
            lists['all']['domains'] += [np.array(
                [self.unique_batches[d] for d in domain.detach().int().cpu().numpy()])]
            lists['all']['domain_preds'] += [domain_preds.detach().float().cpu().numpy()]
            # lists[group]['preds'] += [preds.detach().float().cpu().numpy()]
            lists['all']['classes'] += [labels.detach().int().cpu().numpy()]
            lists['all']['encoded_values'] += [
                enc.detach().float().cpu().numpy()]
            lists['all']['rec_values'] += [
                rec.detach().float().cpu().numpy()]
            lists['all']['names'] += [names]
            lists['all']['gender'] += [meta_inputs.detach().float().cpu().numpy()[:, -1]]
            lists['all']['age'] += [meta_inputs.detach().float().cpu().numpy()[:, -2]]
            lists['all']['atn'] += [str(x) for x in
                                    meta_inputs.detach().float().cpu().numpy()[:, -5:-2]]
            lists['all']['inputs'] += [to_rec]
            try:
                lists['all']['labels'] += [np.array(
                    [self.unique_labels[x] for x in labels.detach().int().cpu().numpy()])]
            except Exception as e:
                print(f"Error loading labels: {e}")
                pass
            if not self.args.kan and self.l1 > 0:
                l1_loss = self.l1_regularization(ae, self.l1)
            elif self.args.kan and self.l1 > 0:
                l1_loss = self.reg_kan(ae, self.l1, self.reg_entropy)
            else:
                l1_loss = torch.zeros(1).to(self.args.device)[0]
            loss = rec_loss + self.gamma * dloss + self.beta * kld.mean() + self.zeta * zinb_loss + l1_loss
            if torch.isnan(loss):
                print("NAN in loss!")
                return 0, ae, warmup
            loss.backward()
            # Clip gradients if requested
            if hasattr(self.args, 'clip_val') and self.args.clip_val and self.args.clip_val > 0:
                nn.utils.clip_grad_norm_(ae.parameters(), max_norm=self.args.clip_val)
            optimizer_ae.step()
            # Step scheduler if configured and not ReduceLROnPlateau
            if self.args.scheduler is not None and self.args.scheduler != 'ReduceLROnPlateau' and scheduler is not None:
                scheduler.step()

        if np.mean(traces['rec_loss']) < self.best_loss:
            # "Every counters go to 0 when a better reconstruction loss is reached"
            print(
                f"Best Loss Epoch {epoch}, Losses: {np.mean(traces['rec_loss'])}, "
                f"Domain Losses: {np.mean(traces['dom_loss'])}, "
                f"Domain Accuracy: {np.mean(traces['dom_acc'])}"
            )
            self.best_loss = np.mean(traces['rec_loss'])
            self.dom_loss = np.mean(traces['dom_loss'])
            self.dom_acc = np.mean(traces['dom_acc'])
            self.warmup_counter = 0
            if warmup:
                torch.save(ae.state_dict(), f'{self.complete_log_path}/warmup.pth')

        # Handle early stop for warmup
        if (self.args.early_warmup_stop != 0 and self.warmup_counter == self.args.early_warmup_stop) and warmup:
            # When the warmup counter reaches limit
            values = log_traces(traces, values)
            if self.args.early_warmup_stop != 0:
                try:
                    ae.load_state_dict(torch.load(f'{self.complete_log_path}/warmup.pth'))
                except Exception as e:
                    print(f"Error loading model: {e}")
            print(f"\n\nWARMUP FINISHED (early stop). {epoch}\n\n")
            warmup = False
            self.warmup_disc_b = True

        # Finish warmup at specified epoch
        if epoch == self.args.warmup and warmup:  # or warmup_counter == 100:
            if self.args.early_warmup_stop != 0:
                try:
                    ae.load_state_dict(torch.load(f'{self.complete_log_path}/warmup.pth'))
                except Exception as e:
                    print(f"Error loading model: {e}")
            print(f"\n\nWARMUP FINISHED. {epoch}\n\n")
            values = log_traces(traces, values)
            warmup = False
            self.warmup_disc_b = True

        # Regular logging during warmup
        if epoch < self.args.warmup and warmup:
            values = log_traces(traces, values)
            self.warmup_counter += 1
            # TODO change logging with tensorboard and neptune. The previous
            if self.log_tb:
                loggers['tb_logging'].logging(values, metrics)
            if self.log_neptune:
                log_neptune(run, values)
            if self.log_mlflow:
                add_to_mlflow(values, epoch)

        ae.train()
        ae.mapper.train()

        # If training of the autoencoder is restricted to the warmup (train_after_warmup=0),
        # all layers except the classification layers are frozen
        if self.args.bdisc:
            self.forward_discriminate(optimizer_b, ae, celoss, loaders['all'])
        if self.warmup_disc_b and self.warmup_b_counter < 0:
            self.warmup_b_counter += 1
        else:
            self.warmup_disc_b = False

        # Step ReduceLROnPlateau after epoch based on reconstruction loss
        if self.args.scheduler == 'ReduceLROnPlateau' and scheduler is not None and len(traces['rec_loss']) > 0:
            scheduler.step(np.mean(traces['rec_loss']))

        return 1, ae, warmup

    def freeze_all_but_clayers(self, ae):
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
            ae.mapper.eval()
            for param in ae.dec.parameters():
                param.requires_grad = False
            for param in ae.enc.parameters():
                param.requires_grad = False
            for param in ae.classifier.parameters():
                param.requires_grad = True
            for param in ae.dann_discriminator.parameters():
                param.requires_grad = False
        return ae

    # def prune_neurons(self, ae, threshold):
    #     """
    #     Prune neurons in the autoencoder
    #     Args:
    #         ae: AutoEncoder object
    #     Returns:
    #         ae: AutoEncoder object
    #     """
    #     for m in ae.modules():
    #         if isinstance(m, KANAutoEncoder2):
    #             for n in m.modules():
    #                 for i in n.modules():
    #                     if isinstance(i, KANLinear):
    #                         i.prune_neurons(threshold)

    def count_neurons(self, ae):
        """
        Count the number of neurons in the autoencoder
        Args:
            ae: AutoEncoder object

        Returns:
            neurons: Number of neurons in the autoencoder
        """
        neurons = 0
        for m in ae.modules():
            if isinstance(m, KANAutoEncoder2):
                for n in m.modules():
                    for i in n.modules():
                        if isinstance(i, KANLinear):
                            i.count_active_neurons()
        return neurons


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--random_recs', type=int, default=0)  # TODO to deprecate, no longer used
    parser.add_argument('--predict_tests', type=int, default=0)
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
    parser.add_argument('--zinb', type=int, default=0)  # TODO resolve problems, do not use
    parser.add_argument('--use_mapping', type=int, default=1, help="Use batch mapping for reconstruct")
    parser.add_argument('--bdisc', type=int, default=1)
    parser.add_argument('--n_repeats', type=int, default=5)
    parser.add_argument('--dloss', type=str, default='inverseTriplet')
    parser.add_argument('--knn_n_neighbors', type=int, default=5, help='Number of neighbors for persistent KNN (triplet mode)')
    parser.add_argument('--csv_file', type=str, default='unique_genes.csv')
    parser.add_argument('--bad_batches', type=str, default='')  # 0;23;22;21;20;19;18;17;16;15
    parser.add_argument('--remove_zeros', type=int, default=0)
    parser.add_argument('--n_meta', type=int, default=0)
    parser.add_argument('--embeddings_meta', type=int, default=0)
    parser.add_argument('--groupkfold', type=int, default=1)
    parser.add_argument('--dataset', type=str, default='custom')
    parser.add_argument('--bs', type=int, default=32, help='Batch size')
    parser.add_argument('--path', type=str, default='./data/')
    parser.add_argument('--exp_id', type=str, default='default_ae_then_classifier')
    parser.add_argument('--strategy', type=str, default='CU_DEM', help='only for alzheimer dataset')
    parser.add_argument('--n_agg', type=int, default=5, help='Number of trailing values to get stable valid values')
    parser.add_argument('--n_layers', type=int, default=2, help='N layers for classifier')
    parser.add_argument('--log1p', type=int, default=1, help='log1p the data? Should be 0 with zinb')
    parser.add_argument('--pool', type=int, default=1, help='only for alzheimer dataset')

    args = parser.parse_args()

    try:
        import mlflow
        mlflow.create_experiment(
            args.exp_id,
            # artifact_location=Path.cwd().joinpath("mlruns").as_uri(),
            # tags={"version": "v1", "priority": "P1"},
        )
    except Exception as e:
        print(f"Error creating experiment: {e}")
        print(f"\n\nExperiment {args.exp_id} already exists\n\n")
    train = TrainAE(args, args.path, fix_thres=-1, load_tb=False, log_metrics=True, keep_models=False,
                    log_inputs=False, log_plots=True, log_tb=False, log_neptune=False,
                    log_mlflow=True, groupkfold=args.groupkfold, pools=True)

    # train.train()
    # List of hyperparameters getting optimized
    parameters = [
        {"name": "nu", "type": "range", "bounds": [1e-4, 1e2], "log_scale": False},
        {"name": "lr", "type": "range", "bounds": [1e-4, 1e-2], "log_scale": True},
        {"name": "wd", "type": "range", "bounds": [1e-8, 1e-5], "log_scale": True},
        {"name": "smoothing", "type": "range", "bounds": [0., 0.2]},
        {"name": "margin", "type": "range", "bounds": [0., 10.]},
        {"name": "triplet_margin", "type": "range", "bounds": [0., 10.]},
        {"name": "knn_n_neighbors", "type": "choice", "values": [1, 3, 5, 7, 9, 11]},
        {"name": "warmup", "type": "range", "bounds": [10, 1000]},
        {"name": "dropout", "type": "range", "bounds": [0.0, 0.5]},
        {"name": "scaler", "type": "choice",
         "values": ['l1', 'minmax', "l2"]},  # scaler whould be no for zinb
        {"name": "layer2", "type": "range", "bounds": [32, 512]},
        {"name": "layer1", "type": "range", "bounds": [512, 1024]},
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
        objective_name='mcc',
        minimize=False,
        total_trials=args.n_trials,
        random_seed=41,
    )
