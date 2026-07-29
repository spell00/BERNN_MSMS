import matplotlib
from tqdm import trange
from bernn.utils.pool_metrics import log_pool_metrics
import pandas as pd
import numpy as np
import random
import torch
from torch import nn
from torch import nn
from sklearn import metrics
import contextlib
import copy
from bernn.utils.ax_compat import optimize, AX_AVAILABLE

from sklearn.metrics import matthews_corrcoef as MCC
from sklearn.neighbors import KNeighborsClassifier
# from ...ml.train.params_gp import *
from ..models.pytorch.aedann import ReverseLayerF
from ..models.pytorch.ekan.src.efficient_kan.kan import KANLinear
from ..models.pytorch.utils.loggings import log_metrics, \
    log_plots, log_shap, log_mlflow, log_dvclive
from bernn.utils.mlflow_compat import mlflow
from bernn.utils.utils import to_csv
from ..models.pytorch.utils.utils import to_categorical, get_empty_traces, \
    log_traces, add_to_mlflow, compute_class_triplet
from ..models.pytorch.utils.loggings import make_data
import warnings
from bernn.utils.data_getters import get_alzheimer, get_amide, get_mice, get_data, get_dummy
import uuid
import os
from sklearn.preprocessing import LabelEncoder

matplotlib.use('Agg')
CUDA_VISIBLE_DEVICES = ""

random.seed(1)
torch.manual_seed(1)
np.random.seed(1)


def keep_top_features(X, features):
    """
    Keeps the top features according to the provided list
    Args:
        X: The data DataFrame to be used to keep the top features
        features: A list or array of features to keep

    Returns:
        X: The data with only the top features
    """
    return X.loc[:, features]


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


class TrainAE:

    @staticmethod
    def _normalize_labels_for_encoding(values):
        """Canonicalize labels before LabelEncoder fit/transform.

        This avoids equivalent numeric strings being treated as different labels,
        e.g. ``"0"`` vs ``"0.0"``.
        """
        series = pd.Series(np.asarray(values).reshape(-1)).astype(str).str.strip()
        numeric = pd.to_numeric(series, errors='coerce')
        normalized = series.to_numpy(dtype=object)

        # Normalize every numeric-like value independently so mixed label sets
        # (e.g. ["QC", "0.0"]) still align with ["QC", "0"].
        valid_numeric = numeric.notna().to_numpy()
        if np.any(valid_numeric):
            numeric_vals = numeric.to_numpy(dtype=np.float64)
            valid_vals = numeric_vals[valid_numeric]
            rounded = np.rint(valid_vals).astype(np.int64)
            is_integer = np.isclose(valid_vals, rounded.astype(np.float64))
            normalized_numeric = np.where(
                is_integer,
                rounded.astype(str),
                np.array([format(v, '.15g') for v in valid_vals], dtype=object),
            )
            normalized[valid_numeric] = normalized_numeric

        return normalized.astype(str)

    def __init__(self, args=None, fix_thres=-1, load_tb=False, log_metrics=False, keep_models=True, log_inputs=True,
                 log_plots=False, log_tb=False, log_mlflow=True, log_dvclive=False, groupkfold=True, pools=True, **kwargs):
        """

        Args:
            args: contains multiple arguments passed in the command line
            log_path (str): Path where the tensorboard logs are saved
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
        self.best_loss = np.inf
        self.best_closs = np.inf
        self.logged_inputs = False
        self.log_tb = log_tb
        self.log_mlflow = log_mlflow
        self.best_epoch = -1
        self.best_valid_mcc = float("-inf")
        self.best_state_dicts = None
                
        if args is None:
            class Args: pass
            args = Args()
            
        for k, v in kwargs.items():
            setattr(args, k, v)
            
        self.args = args
        self.log_metrics = log_metrics
        self.log_plots = log_plots
        self.log_inputs = log_inputs
        self.log_dvclive = log_dvclive
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
        self.ae = None
        self.best_checkpoint_path = None  # Path to best model checkpoint
        self.best_model_state = None      # Optional: in-memory best state for fast loading
        self._label_encoder = None        # Optional encoder for non-numeric labels
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
        self.rep = 0
        self.seed = 0
        self.combinations = []


    # Back-compat wrappers (old names)
    def loop(self, group, optimizer, ae, celoss, loader, 
             lists, traces, nu=1, mapping=True):
        return self.loop_infer(group, optimizer, ae, celoss, loader, lists, traces, nu, mapping)

    def loop2(self, group, optimizer, ae, scheduler, losses, loader, lists, traces, nu=1, mapping=True):
        return self.loop_train(group, optimizer, ae, scheduler, losses, loader, lists, traces, nu, mapping)

    def train(self, params=None):
        """Deprecated alias for _train."""
        return self._train(params)

    def make_params(self, params):
        # Normalize depth first so any Ax-provided n_layers is honored.
        try:
            n_layers = int(params.get('n_layers', getattr(self.args, 'n_layers', 1)))
        except Exception:
            n_layers = 1
        n_layers = max(1, n_layers)
        self.args.n_layers = n_layers
        params['n_layers'] = n_layers

        # Ensure layer1..layerN are always available, using halving defaults
        # (floor at 16) for missing deeper layers.
        layer_params = self.get_default_layer_params()
        for key, value in layer_params.items():
            params.setdefault(key, value)
            setattr(self.args, key, int(params[key]))
        # Drop any stale layer keys beyond current n_layers.
        for key in [k for k in list(params.keys()) if k.startswith('layer')]:
            try:
                idx = int(key.replace('layer', ''))
            except Exception:
                continue
            if idx > n_layers:
                params.pop(key, None)

        # Fixing the hyperparameters that are not optimized
        if self.args.dloss not in ['revTriplet', 'revDANN', 'DANN',
                                   'inverseTriplet', 'normae'] or 'gamma' not in params:
            # gamma = 0 will ensure DANN is not learned
            params['gamma'] = 0
        if not self.args.variational or 'beta' not in params:
            # beta = 0 because useless outside a variational autoencoder
            params['beta'] = 0
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
        self.l1 = params.get('l1', 0)
        self.reg_entropy = params.get('reg_entropy', 0)
        self.args.scaler = params['scaler']
        self.args.warmup = params['warmup']
        self.args.disc_b_warmup = params.get('disc_b_warmup', 0)
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

    def get_default_layer_params(self):
        """Build default layer parameters for any classifier depth.

        Rules:
        - Start from ``layer1`` (default 512, minimum 16)
        - For each next layer, default to half of previous
        - Never go below 16
        - Explicitly provided ``layer{i}`` values still override defaults

        Returns:
            Dict like ``{'layer1': ..., 'layer2': ...}`` containing exactly
            ``n_layers`` entries.
        """
        try:
            n_layers = int(getattr(self.args, 'n_layers', 1))
        except Exception:
            n_layers = 1
        n_layers = max(1, n_layers)

        try:
            first = int(getattr(self.args, 'layer1', 512))
        except Exception:
            first = 512
        first = max(16, first)

        layer_params = {'layer1': first}
        prev = first

        for i in range(2, n_layers + 1):
            default_i = max(prev // 2, 16)
            raw = getattr(self.args, f'layer{i}', default_i)
            try:
                value_i = int(raw)
            except Exception:
                value_i = default_i
            value_i = max(16, value_i)
            layer_params[f'layer{i}'] = value_i
            prev = value_i

        return layer_params

    def _prepare_data(self, X, y=None, groups=None, X_valid=None, y_valid=None,
                      groups_valid=None, X_test=None, y_test=None, groups_test=None,
                      cross_validation=False, cross_test=False, val_size=0.2,
                      internal_validation=False):
        """
        Internal method to prepare the in-memory data dictionary (self.data) from inputs.

        Important contract:
        - Batch IDs are mandatory.
        - ``groups`` must be provided for training samples.
        - ``groups_test`` must be provided when ``X_test`` is provided.

        This is required to keep BERNN behavior explicit and avoid silently
        assigning synthetic batch IDs that would make domain-adaptation metrics
        ambiguous.
        """
        # cross-test can't be done without cross-validation logic
        if cross_test:
            cross_validation = True
        self._cross_test_active = cross_test

        # Ensure inputs are in the right format
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if y is None:
            y = np.zeros(len(X))
        if not isinstance(y, np.ndarray):
            y = np.array(y)

        # Normalize all labels (train, test, valid) upfront to ensure consistency.
        y = self._normalize_labels_for_encoding(y)
        test_labels_for_union = None
        if y_test is not None:
            y_test = self._normalize_labels_for_encoding(y_test)
            test_labels_for_union = y_test
        if y_valid is not None:
            y_valid = self._normalize_labels_for_encoding(y_valid)

        # Build label encoder on the union of train, valid, and test labels (if provided)
        y_is_numeric = np.issubdtype(y.dtype, np.number)
        if not y_is_numeric:
            self._label_encoder = LabelEncoder()
            # Union all available labels for consistent encoding
            label_union = [y]
            if test_labels_for_union is not None:
                label_union.append(test_labels_for_union)
            if y_valid is not None:
                label_union.append(y_valid)
            all_labels_union = np.unique(np.concatenate(label_union))
            self._label_encoder.fit(all_labels_union)
            y = self._label_encoder.transform(y)
            if y_test is not None:
                y_test = self._label_encoder.transform(y_test)
            if y_valid is not None:
                y_valid = self._label_encoder.transform(y_valid)
        else:
            self._label_encoder = None

        if groups is None:
            raise ValueError("Batch IDs are mandatory: provide groups_train/groups for every training sample.")
        if not isinstance(groups, np.ndarray):
            groups = np.array(groups)

        self.data = {
            'inputs': {}, 'names': {}, 'labels': {},
            'cats': {}, 'batches': {}, 'orders': {}, 'sets': {}
        }

        assert len(X) == len(y) == len(groups), "X, y, groups must have same length"

        label_map = {l: i for i, l in enumerate(sorted(np.unique(y), key=str))}
        self.unique_labels = np.array(sorted(label_map.keys(), key=str))
        self.unique_batches = np.unique(groups)

        # --- SPLIT LOGIC ---
        # If X_valid/y_valid are provided, use them for valid
        # If X_test/y_test are provided, use them for test
        # If neither, split into train/valid/test
        # If only test is missing, split once for valid, leave test empty
        # If only valid is missing, raise error

        self._no_internal_validation = not internal_validation

        # Sklearn-style fit: train on every provided row. BERNN's legacy trainer
        # expects valid/test loaders for logging/checkpoint bookkeeping, so use
        # train-resubstitution monitors instead of splitting or accepting
        # external evaluation rows during fit.
        if not internal_validation and X_valid is None and y_valid is None and X_test is None and y_test is None:
            batch_map = {b: i for i, b in enumerate(self.unique_batches)}
            mapped_groups = np.array([batch_map[b] for b in groups])
            self.data['inputs']['train'] = X
            self.data['labels']['train'] = y
            self.data['batches']['train'] = mapped_groups
            self.data['inputs']['valid'] = X.copy()
            self.data['labels']['valid'] = y.copy()
            self.data['batches']['valid'] = mapped_groups.copy()
            self.data['inputs']['test'] = X.copy()
            self.data['labels']['test'] = y.copy()
            self.data['batches']['test'] = mapped_groups.copy()
            self.unique_batches = np.array(list(batch_map.keys()))
        # If external validation is provided, use it for monitor/early stopping.
        # X_test may be unlabeled (hidden benchmark); use sentinel labels in that case.
        elif X_valid is not None and y_valid is not None:
            if not isinstance(X_valid, pd.DataFrame):
                X_valid = pd.DataFrame(X_valid)
            if groups_valid is None:
                groups_valid = np.zeros(len(X_valid))
            if not isinstance(groups_valid, np.ndarray):
                groups_valid = np.array(groups_valid)
            assert len(X_valid) == len(y_valid) == len(groups_valid), (
                "X_valid, y_valid, groups_valid must have same length"
            )

            if X_test is None:
                X_test = pd.DataFrame(columns=X.columns)
                y_test_for_data = np.array([], dtype=int)
                groups_test = np.array([])
            else:
                if not isinstance(X_test, pd.DataFrame):
                    X_test = pd.DataFrame(X_test)
                if groups_test is None:
                    groups_test = np.zeros(len(X_test))
                if not isinstance(groups_test, np.ndarray):
                    groups_test = np.array(groups_test)
                assert len(X_test) == len(groups_test), "X_test and groups_test must have same length"
                if y_test is None:
                    y_test_for_data = np.full(len(X_test), -1, dtype=int)
                else:
                    y_test_for_data = y_test
                    assert len(X_test) == len(y_test_for_data), "X_test and y_test must have same length"

            batch_values = [groups, groups_valid]
            if len(groups_test) > 0:
                batch_values.append(groups_test)
            batch_map = {b: i for i, b in enumerate(np.unique(np.concatenate(batch_values)))}

            self.data['inputs']['train'] = X
            self.data['labels']['train'] = y
            self.data['batches']['train'] = np.array([batch_map[b] for b in groups])
            self.data['inputs']['valid'] = X_valid
            self.data['labels']['valid'] = y_valid
            self.data['batches']['valid'] = np.array([batch_map[b] for b in groups_valid])
            self.data['inputs']['test'] = X_test
            self.data['labels']['test'] = y_test_for_data
            self.data['batches']['test'] = np.array([batch_map[b] for b in groups_test])
            self.unique_batches = np.array(list(batch_map.keys()))
        # If only test is provided, assign test, split for valid
        elif X_test is not None and y_test is not None:
            # Split X into train/valid
            n_splits = int(1.0 / val_size) if val_size > 0 else 5
            if len(self.unique_batches) > 1:
                from sklearn.model_selection import StratifiedGroupKFold
                skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=self.seed)
                split_iter = list(skf.split(X, y, groups))
            else:
                from sklearn.model_selection import StratifiedKFold
                skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.seed)
                split_iter = list(skf.split(X, y))
            required_labels = set(np.unique(y))
            valid_fold = 0
            train_inds, valid_inds = split_iter[valid_fold]
            for candidate_fold, (candidate_train, candidate_valid) in enumerate(split_iter):
                if required_labels.issubset(set(np.unique(y[candidate_train]))):
                    valid_fold = candidate_fold
                    train_inds, valid_inds = candidate_train, candidate_valid
                    break
            self.data['inputs']['train'] = X.iloc[train_inds]
            self.data['labels']['train'] = y[train_inds]
            self.data['batches']['train'] = groups[train_inds]
            self.data['inputs']['valid'] = X.iloc[valid_inds]
            self.data['labels']['valid'] = y[valid_inds]
            self.data['batches']['valid'] = groups[valid_inds]
            batch_map = {b: i for i, b in enumerate(self.unique_batches)}
            self.data['batches']['train'] = np.array([batch_map[b] for b in self.data['batches']['train']])
            self.data['batches']['valid'] = np.array([batch_map[b] for b in self.data['batches']['valid']])
            # Assign test
            if not isinstance(groups_test, np.ndarray):
                groups_test = np.array(groups_test)
            if not isinstance(X_test, pd.DataFrame):
                X_test = pd.DataFrame(X_test)
            self.data['inputs']['test'] = X_test
            self.data['labels']['test'] = y_test
            # Map test batches through the same batch_map so every split uses
            # consistent integer batch ids. Leaving test as raw values mixed
            # str/int in data['batches']['all'] and broke np.unique in scale_data.
            # Batches present only in test get new indices appended.
            mapped_test = []
            for b in groups_test:
                if b not in batch_map:
                    batch_map[b] = len(batch_map)
                mapped_test.append(batch_map[b])
            self.data['batches']['test'] = np.array(mapped_test)
            self.unique_batches = np.array(list(batch_map.keys()))
        # If neither valid nor test is provided, split into train/valid/test
        else:
            n_splits = max(3, int(getattr(self.args, 'n_repeats', 3)))
            if len(self.unique_batches) > 1:
                from sklearn.model_selection import StratifiedGroupKFold
                skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=self.seed)
                split_iter = list(skf.split(X, y, groups))
            else:
                from sklearn.model_selection import StratifiedKFold
                skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.seed)
                split_iter = list(skf.split(X, y))
            if (self.seed - self.rep) > 10:
                valid_fold = 1
            else:
                valid_fold = 0
            test_fold = (valid_fold + 1) % n_splits
            train_inds_v, valid_inds = split_iter[valid_fold]
            train_inds_t, test_inds = split_iter[test_fold]
            train_inds = np.array([x for x in range(len(X)) if x not in np.concatenate((valid_inds, test_inds))])
            self.data['inputs']['train'] = X.iloc[train_inds]
            self.data['labels']['train'] = y[train_inds]
            self.data['batches']['train'] = groups[train_inds]
            self.data['inputs']['valid'] = X.iloc[valid_inds]
            self.data['labels']['valid'] = y[valid_inds]
            self.data['batches']['valid'] = groups[valid_inds]
            self.data['inputs']['test'] = X.iloc[test_inds]
            self.data['labels']['test'] = y[test_inds]
            self.data['batches']['test'] = groups[test_inds]        
            
            batch_map = {b: i for i, b in enumerate(self.unique_batches)}
            self.data['batches']['train'] = np.array([batch_map[b] for b in self.data['batches']['train']])
            self.data['batches']['valid'] = np.array([batch_map[b] for b in self.data['batches']['valid']])
            self.data['batches']['test'] = np.array([batch_map[b] for b in self.data['batches']['test']])
            

        # Ensure all keys are set for downstream code
        for group in ['train', 'valid', 'test']:
            if group not in self.data['inputs']:
                self.data['inputs'][group] = pd.DataFrame()
            if group not in self.data['labels']:
                self.data['labels'][group] = np.array([])
            if group not in self.data['batches']:
                self.data['batches'][group] = np.array([])

        # Invariant: all known holdout labels must already be represented in train labels.
        # Unknown labels use sentinel -1 when y_test is omitted and are excluded.
        all_labels = np.concatenate([self.data['labels']['train'], self.data['labels']['valid'], self.data['labels']['test']])
        all_labels_set = set(all_labels[all_labels != -1])
        train_label_set = set(np.unique(self.data['labels']['train']))
        # holdout_labels = np.concatenate((self.data['labels']['valid'], self.data['labels']['test']))
        # known_holdout_labels = holdout_labels[holdout_labels != -1]
        missing_train_labels = all_labels_set - train_label_set
        assert not missing_train_labels, (
            "After split, holdout labels must be present in train labels. "
            f"Missing in train: {sorted(missing_train_labels)}"
        )
            
        for group in ['train', 'valid', 'test']:
            n_samples = len(self.data['inputs'][group])
            self.data['names'][group] = pd.Series([f"{group}_{i}" for i in range(n_samples)])
            self.data['orders'][group] = np.arange(n_samples)
            self.data['sets'][group] = np.array([group] * n_samples)
            self.data['cats'][group] = np.array([label_map[l] if l in label_map else -1 for l in self.data['labels'][group]])
            
        for key in ['inputs']:
            if len(self.data['inputs']['test']) > 0:
                self.data[key]['all'] = pd.concat([self.data[key]['train'], self.data[key]['valid'], self.data[key]['test']])
            else:
                self.data[key]['all'] = pd.concat([self.data[key]['train'], self.data[key]['valid']])
                
        for key in ['names', 'labels', 'cats', 'batches', 'orders', 'sets']:
            if len(self.data['labels']['test']) > 0:
                self.data[key]['all'] = np.concatenate([self.data[key]['train'], self.data[key]['valid'], self.data[key]['test']])
            else:
                self.data[key]['all'] = np.concatenate([self.data[key]['train'], self.data[key]['valid']])
                
        if self.args.controls != '':
            self.data = binarize_labels(self.data, self.args.controls)

        if self.pools:
            for key in ['inputs', 'names', 'labels', 'cats', 'batches', 'orders', 'sets']:
                self.data[key]['train_pool'] = self.data[key]['train']
                self.data[key]['valid_pool'] = self.data[key]['valid']
                self.data[key]['test_pool'] = self.data[key]['test']
                self.data[key]['all_pool'] = self.data[key]['all']

        print('Data loaded')
        # self.make_samples_weights()

    def fit(self, X_train, y_train, *, X_valid=None, y_valid=None, groups_valid=None,
            X_test=None, y_test=None, groups_test=None, groups_train=None, params=None,
            cross_validation=False, cross_test=False, val_size=0.2,
            internal_validation=False, **kwargs):
        """Fit BERNN on the provided training rows, sklearn-style.

        External validation data can be provided with X_valid/y_valid/groups_valid
        and is used by BERNN's monitor/early-stopping path. X_test can be provided
        for test-set monitoring; y_test may be omitted for hidden/unlabeled test
        matrices, in which case test labels use the sentinel ``-1`` internally.
        Set internal_validation=True only for legacy BERNN experiments that
        intentionally want BERNN to split the training data internally.
        """
        if groups_train is None:
            groups_train = kwargs.pop('batches_train', None)
        if groups_valid is None:
            groups_valid = kwargs.pop('batches_valid', None)
        if groups_test is None:
            groups_test = kwargs.pop('batches_test', None)
        if kwargs:
            raise TypeError(f"Unexpected BERNN.fit arguments: {sorted(kwargs)}")
        if (X_valid is None) != (y_valid is None):
            raise TypeError("BERNN.fit requires X_valid and y_valid to be provided together.")
        if X_test is not None and y_test is None and (X_valid is None or y_valid is None):
            raise TypeError(
                "BERNN.fit requires external X_valid/y_valid when X_test is provided without y_test."
            )
        response = -1
        flag = True
        max_repeats = max(1, int(getattr(self.args, 'n_repeats', 1)))
        while self.rep < max_repeats and flag:
            self._prepare_data(X=X_train, y=y_train, groups=groups_train,
                               X_valid=X_valid, y_valid=y_valid, groups_valid=groups_valid,
                               X_test=X_test, y_test=y_test, groups_test=groups_test,
                               cross_validation=cross_validation, cross_test=cross_test,
                               val_size=val_size, internal_validation=internal_validation)
            response = self._train(params)
            if response == -1 and self.seed > self.args.n_repeats * 100:
                print("Warning: multiple training attempts failed, stopping early.")
                print('Setting n_repeats to current n rep:', self.rep + 1)
                self.args.n_repeats = self.rep
                break
            if not cross_validation and not cross_test:
                flag = False
        if self.best_state_dicts is not None:
            self.restore_best_model_state()
        else:
            raise RuntimeError(
                "BERNN fit finished but no model state was saved. "
                "This likely means no training epoch completed."
            )
        return self

    def fit_predict(self, X_train, y_train, *args, groups_train=None, params=None,
                    cross_validation=False, cross_test=False, val_size=0.2,
                    internal_validation=False, **kwargs):
        """Fit on X_train/y_train and return in-sample predictions.

        This method deliberately does not accept X_test/y_test. For holdout
        prediction use fit(...); predict(X_test).
        """
        if args:
            raise TypeError(
                "BERNN.fit_predict no longer accepts a test matrix. "
                "Use fit(X_train, y_train, groups_train=...) then predict(X_test)."
            )
        self.fit(X_train, y_train, groups_train=groups_train, params=params,
                 cross_validation=cross_validation, cross_test=cross_test,
                 val_size=val_size, internal_validation=internal_validation,
                 **kwargs)
        return self.predict(X_train)

    def fit_transform(self, *args, **kwargs):
        """Deprecated alias for fit_predict."""
        warnings.warn("fit_transform is deprecated, use fit_predict instead", DeprecationWarning)
        return self.fit_predict(*args, **kwargs)

    def _validate_and_save_best_checkpoint(self, ae_model, epoch, valid_mcc):
        """
        Validate model on validation set and save checkpoint if it's the best so far (based on MCC).
        
        Args:
            ae_model: The autoencoder model to evaluate and potentially save
            epoch: Current epoch number
            valid_mcc: Validation MCC score
            
        Returns:
            bool: True if this is a new best model, False otherwise
        """
        is_best = valid_mcc > self.best_mcc
        
        if is_best:
            self.best_mcc = valid_mcc
            
            # Create checkpoint directory if it doesn't exist
            if self.complete_log_path:
                os.makedirs(self.complete_log_path, exist_ok=True)
                checkpoint_path = os.path.join(self.complete_log_path, 'best_model.pth')
                
                # Save checkpoint to disk
                try:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': ae_model.state_dict(),
                        'best_mcc': self.best_mcc,
                    }, checkpoint_path)
                    self.best_checkpoint_path = checkpoint_path
                    print(f"✓ Saved best model checkpoint at epoch {epoch} (valid_mcc={valid_mcc:.4f})")
                except Exception as e:
                    print(f"Warning: Failed to save checkpoint: {e}")
            
            return True
        
        return False

    def _reset_counts(self):
        """
        Resets any internal counts or state that should be cleared between training runs.
        This can be useful if the same TrainAE instance is reused for multiple fits.
        """
        self.rep = 0
        self.seed = 0
        self.combinations = []
        self._knn_ready = False
        # Clear best-model tracking so each fit/trial is independent. This is
        # required because Ax trials reuse the same trainer but build models of
        # different widths (layer1 is searched); without resetting, a trial that
        # fails to beat the previous best valid MCC would try to restore the
        # prior trial's differently-shaped state_dict and raise a size mismatch.
        self.best_mcc = -1
        self.best_valid_mcc = float("-inf")
        self.best_state_dicts = None
        self.best_epoch = None

    def _train(self, params=None):
        """
        Master training loop that executes after data is loaded.
        """
        from bernn.dl.models.pytorch.utils.dataset import get_loaders, get_loaders_no_pool
        from bernn.dl.models.pytorch.utils.utils import get_optimizer, get_empty_traces
        import inspect
        import torch.nn as nn
        
        self.n_cats = len(self.unique_labels)
        self.n_batches = len(set(self.data['batches']['all']))
        
        if getattr(self, 'pools', False):
            loaders = get_loaders(self.data, getattr(self.args, 'random_recs', 0), self.samples_weights, getattr(self.args, 'dloss', 'revTriplet'), None, None, bs=getattr(self.args, 'bs', 32))
        else:
            loaders = get_loaders_no_pool(self.data, getattr(self.args, 'random_recs', 0), self.samples_weights, getattr(self.args, 'dloss', 'revTriplet'), None, None, bs=getattr(self.args, 'bs', 32))

        device = getattr(self.args, 'device', 'cpu')
        
        if isinstance(self.ae, nn.Module):
            print("Pre-loaded autoencoder instance detected. Skipping warmup phase.")
            warmup_epochs = 0
            ae_model = self.ae
        else:
            ae_cls = self.load_autoencoder()
            warmup_epochs = getattr(self.args, 'warmup', 100)
            print(f"Instantiating new autoencoder. Warmup epochs: {warmup_epochs}")
            model_kwargs = {
                'n_features': self.data['inputs']['all'].shape[1],
                'n_batches': self.n_batches,
                'nb_classes': self.n_cats,
                'mapper': getattr(self.args, 'use_mapping', 1),
                'layer1': getattr(self.args, 'layer1', 512),
                'layer2': getattr(self.args, 'layer2', 128),
                'n_layers': getattr(self.args, 'n_layers', 2),
                'dropout': getattr(self.args, 'dropout', 0.1),
                'variational': getattr(self.args, 'variational', 0),
                'conditional': getattr(self.args, 'conditional', False),
                'add_noise': getattr(self.args, 'add_noise', 0),
                'tied_weights': getattr(self.args, 'tied_weights', 0),
                'device': device
            }
            try:
                ae_model = ae_cls(**model_kwargs).to(device)
            except TypeError:
                ae_model = ae_cls(self.data['inputs']['all'].shape[1], self.n_cats, self.n_batches, 0, self.args, None).to(device)
                
        self.ae = ae_model

        if hasattr(self.ae, 'mapper'):
            self.ae.mapper.to(device)
        if hasattr(self.ae, 'dec'):
            self.ae.dec.to(device)
            
        lr = getattr(self.args, 'lr', 1e-3)
        wd = getattr(self.args, 'wd', 1e-5)
        optimizer_type = getattr(self.args, 'optimizer_type', 'adam')
        
        optimizer_ae = get_optimizer(self.ae, lr, wd, optimizer_type)
        if hasattr(self.ae, 'classifier'):
            self.optimizer_c = get_optimizer(self.ae.classifier, getattr(self.args, 'nu', 1.0) * lr, wd, optimizer_type)
        else:
            self.optimizer_c = None

        sceloss, celoss, mseloss, triplet_loss = self.get_losses(
            getattr(self.args, 'scaler', 'l1'), 
            getattr(self.args, 'smoothing', 0), 
            getattr(self.args, 'margin', 1.0), 
            getattr(self.args, 'dloss', 'revTriplet')
        )
        
        n_epochs = getattr(self.args, 'n_epochs', 100)
        
        values = {}
        loggers = {}
        
        # Initialize best_mcc for tracking (will be updated when checkpoints are saved)
        mcc = -1
        
        print(f"Starting training loop for {n_epochs} epochs...")
        from tqdm import tqdm
        if warmup_epochs > 0:
            warmup_pbar = tqdm(range(warmup_epochs), desc="Warmup Epochs", unit="epoch")
            for warmup_epoch in warmup_pbar:
                lists, traces = get_empty_traces()
                self.ae.train()
                self.warmup_loop(optimizer_ae, None, self.ae, celoss, loaders['all'], triplet_loss, mseloss, True, warmup_epoch, values, loggers, {}, traces, None)

        pbar = tqdm(range(warmup_epochs, n_epochs), desc="Epochs", unit="epoch")
        for epoch in pbar:
            lists, traces = get_empty_traces()
            self.ae.train()
            pbar.set_postfix_str("Training")
            self.loop_train('train', optimizer_ae, self.ae, None,
                            {'celoss': celoss, 'sceloss': sceloss, 'mseloss': mseloss, 'triplet_loss': triplet_loss},
                            loaders['train'], lists, traces, nu=getattr(self.args, 'nu', 1), mapping=getattr(self.args, 'use_mapping', True))

            # ===== VALIDATION & CHECKPOINT SAVING =====
            # After training epoch, validate on validation set
            if 'valid' in loaders and len(loaders['valid'].dataset) > 0:
                valid_lists, valid_traces = get_empty_traces()
                self.ae.eval()

                with torch.no_grad():
                    self.loop_train('valid', optimizer_ae, self.ae, None,
                                   {'celoss': celoss, 'sceloss': sceloss, 'mseloss': mseloss, 'triplet_loss': triplet_loss},
                                   loaders['valid'], valid_lists, valid_traces, nu=getattr(self.args, 'nu', 1), mapping=getattr(self.args, 'use_mapping', True))

                # Compute validation MCC
                try:
                    valid_preds = np.concatenate(valid_lists['valid']['preds']).argmax(1)
                    valid_labels = np.concatenate(valid_lists['valid']['classes'])
                    valid_mcc = MCC(valid_labels, valid_preds)

                    # Save checkpoint if this is the best MCC so far
                    self._validate_and_save_best_checkpoint(self.ae, epoch, valid_mcc)

                    if epoch % 10 == 0:
                            print(f"Epoch {epoch}: Valid MCC = {valid_mcc:.4f}, Best MCC = {self.best_mcc}")
                except Exception as e:
                    print(f"Warning: Could not compute validation MCC: {e}")

        # ===== LOAD BEST CHECKPOINT =====
        # After training completes, load the best model checkpoint
        if self.best_checkpoint_path and os.path.exists(self.best_checkpoint_path):
            try:
                checkpoint = torch.load(self.best_checkpoint_path, map_location=device)
                self.ae.load_state_dict(checkpoint['model_state_dict'])
                print(f"✓ Loaded best model from checkpoint (best_mcc={checkpoint['best_mcc']:.4f})")
            except Exception as e:
                print(f"Warning: Could not load best checkpoint: {e}")
        
        print("Training completed.")
        return self

    def transform(self, X):
        """
        Transform X into the latent space of the autoencoder.
        """
        if not isinstance(self.ae, nn.Module):
            raise ValueError("AutoEncoder is not initialized. Please run training first.")
        
        self.ae.enc.eval()
        self.ae.classifier.eval()
        
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        # We need a dataloader to transform X
        from torch.utils.data import DataLoader, TensorDataset
        dataset = TensorDataset(torch.tensor(X.values, dtype=torch.float32))
        loader = DataLoader(dataset, batch_size=getattr(self.args, 'bs', 32), shuffle=False)
        
        from tqdm import tqdm
        encoded_list = []
        with torch.no_grad():
            for batch in tqdm(loader, desc="Transforming", leave=False):
                data = batch[0].to(self.args.device)
                
                # Mock domain as all zeros
                domain = torch.zeros(data.shape[0], dtype=torch.long, device=self.args.device)
                to_rec = data.clone()
                
                enc, _, _, _ = self.ae(data, to_rec, domain, sampling=False)
                encoded_list.append(enc.detach().cpu().numpy())
                
        return np.concatenate(encoded_list, axis=0)

    def predict(self, X):
        """
        Predict numeric class ids for X using the best trained autoencoder (loaded from checkpoint).
        
        Architecture Note:
        - During training: best model weights are saved to disk when validation MCC improves
        - At end of training: best checkpoint is loaded into self.ae
        - During inference: self.ae contains the best model, not the last epoch
        - This hybrid approach gives fast inference (no disk I/O) with checkpoint persistence
        """
        
        if not isinstance(self.ae, nn.Module):
            raise ValueError("AutoEncoder is not initialized. Please run training first.")
        
        self.ae.enc.eval()
        self.ae.classifier.eval()

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        from torch.utils.data import DataLoader, TensorDataset
        device = getattr(self.args, 'device', 'cpu')
        dataset = TensorDataset(torch.tensor(X.values, dtype=torch.float32))
        loader = DataLoader(dataset, batch_size=getattr(self.args, 'bs', 32), shuffle=False)

        from tqdm import tqdm
        preds_list = []
        with torch.no_grad():
            for batch in tqdm(loader, desc="Predicting", leave=False):
                data = batch[0].to(device)
                preds = np.asarray(self.ae.predict(data)).reshape(-1)
                preds_list.append(np.asarray(preds).reshape(-1))

        preds_numeric = np.concatenate(preds_list, axis=0).astype(np.int64)
        if self._label_encoder is not None:
            return self._label_encoder.inverse_transform(preds_numeric)
        return preds_numeric

    def predict_proba(self, X):
        """Predict class probabilities for X using the best trained autoencoder.

        This mirrors ``predict`` but returns probability estimates from the
        underlying model when available.
        """
        if not isinstance(self.ae, nn.Module):
            raise ValueError("AutoEncoder is not initialized. Please run training first.")

        if not hasattr(self.ae, 'predict_proba'):
            raise AttributeError("Underlying autoencoder does not expose predict_proba.")

        self.ae.eval()

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        from torch.utils.data import DataLoader, TensorDataset
        device = getattr(self.args, 'device', 'cpu')
        dataset = TensorDataset(torch.tensor(X.values, dtype=torch.float32))
        loader = DataLoader(dataset, batch_size=getattr(self.args, 'bs', 32), shuffle=False)

        from tqdm import tqdm
        proba_list = []
        with torch.no_grad():
            for batch in tqdm(loader, desc="Predicting probabilities", leave=False):
                data = batch[0].to(device)
                probs = np.asarray(self.ae.predict_proba(data))
                proba_list.append(probs)

        probs = np.concatenate(proba_list, axis=0)
        if probs.ndim == 1:
            probs = probs.reshape(-1, 1)
        return probs



    def autocast_context(self):
        """Create autocast context for mixed precision training with bfloat16."""
        device = str(getattr(self.args, 'device', 'cpu')).lower()

        # Never request CUDA autocast on CPU runs.
        if device == 'cpu':
            return contextlib.nullcontext()

        # Use CUDA autocast only when CUDA is both requested and available.
        if device.startswith('cuda') and torch.cuda.is_available():
            try:
                return torch.autocast(device_type='cuda', dtype=torch.bfloat16)
            except (AttributeError, RuntimeError):
                return contextlib.nullcontext()

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
            'use_mapping': 1,
            'bdisc': 1,
            'n_repeats': 5,
            'dloss': 'inverseTriplet',  # one of revDANN, DANN, inverseTriplet, revTriplet
            'bad_batches': '',  # 0;23;22;21;20;19;18;17;16;15
            'groupkfold': 1,
            'bs': 32,
            'exp_id': 'default_ae_then_classifier',
            'n_agg': 1,  # Number of trailing values to get stable valid values
            'n_layers': 2,  # N layers for classifier
            'log1p': 1,  # log1p the data?
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
            # In cross_test mode or for the test set, we keep all samples even if they have 
            # unknown labels (e.g. dummy zeros) to ensure we get predictions for everything.
            if group == 'test' or (group == 'valid' and getattr(self, '_cross_test_active', False)):
                inds_to_keep = np.arange(len(self.data['labels'][group]), dtype=int)
            else:
                inds_to_keep = np.array([i for i, x in enumerate(self.data['labels'][group]) if x in self.unique_labels], dtype=int)
            
            display_group = group
            if getattr(self, '_no_internal_validation', False) and group in ['valid', 'test']:
                display_group = f"{group} (train monitor; no internal holdout)"
            print(f"[make_samples_weights] Group '{display_group}': {len(inds_to_keep)} samples to keep out of {len(self.data['labels'][group])}")
            if len(inds_to_keep) == 0:
                if group == 'test':
                     continue # Test set might be empty if not provided
                raise ValueError(f"[make_samples_weights] After filtering, no samples remain for group '{group}'. Check your CSV and label filtering criteria.")
            # Defensive: ensure indices are valid
            if inds_to_keep.max(initial=-1) >= len(self.data['labels'][group]) or inds_to_keep.min(initial=0) < 0:
                raise IndexError(f"[make_samples_weights] Computed indices for group '{group}' are out of bounds.")
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
        self.shap_ae = SHAPAutoEncoder
        return AutoEncoder

    def log_rep(self, best_lists, best_vals, best_values, traces, metrics, run, loggers, ae, shap_ae, h,
                epoch):
        # best_traces = self.get_mccs(best_lists, traces)

        self.log_predictions(best_lists, run, h)

        if self.log_metrics:
            if self.log_tb and self.log_metrics:
                try:
                    # logger, lists, values, model, unique_labels, mlops, epoch, metrics, device='cuda'
                    metrics = log_metrics(loggers['logger'], best_lists, best_vals, ae,
                                          np.unique(np.concatenate(best_lists['train']['labels'])),
                                          np.unique(self.data['batches']), epoch, mlops="tensorboard",
                                          metrics=metrics, device=self.args.device)
                except BrokenPipeError:
                    print("\n\n\nProblem with logging stuff!\n\n\n")
            if self.log_mlflow and self.log_metrics:
                try:
                    metrics = log_metrics(None, best_lists, best_vals, ae,
                                          np.unique(np.concatenate(best_lists['train']['labels'])),
                                          np.unique(self.data['batches']), epoch, mlops="mlflow",
                                          metrics=metrics,
                                          device=self.args.device)
                except BrokenPipeError:
                    print("\n\n\nProblem with logging stuff!\n\n\n")
            if self.log_dvclive:
                try:
                    metrics = log_metrics(None, best_lists, best_vals, ae,
                                          np.unique(np.concatenate(best_lists['train']['labels'])),
                                          np.unique(self.data['batches']), epoch, mlops="dvclive",
                                          metrics=metrics,
                                          device=self.args.device)
                except BrokenPipeError:
                    print("\n\n\nProblem with logging dvclive!\n\n\n")

        if self.log_metrics and self.pools:
            try:
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
                if self.log_dvclive:
                    print("Logging pool metrics to dvclive: not implemented")

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
                    log_shap(loggers['logger_cm'], shap_ae, best_lists, self.columns, 'tb',
                             self.complete_log_path, self.args.device)
                if self.log_mlflow:
                    log_shap(None, shap_ae, best_lists, self.columns, 'mlflow',
                             self.complete_log_path, self.args.device)
                    log_plots(None, best_lists, 'mlflow', epoch)
                if self.log_dvclive:
                    print("Logging plots to dvclive: not implemented")

        columns = list(self.data['inputs']['all'].columns)

        rec_data, enc_data = to_csv(best_lists, self.complete_log_path, columns)

        # Pool/batch metrics are optional and absent for non-pool datasets.
        best_values['pool_metrics'] = {}
        if 'batches' in metrics:
            best_values['batches'] = metrics['batches']
        if 'pool_metrics_enc' in metrics:
            best_values['pool_metrics']['enc'] = metrics['pool_metrics_enc']
        if 'pool_metrics_rec' in metrics:
            best_values['pool_metrics']['rec'] = metrics['pool_metrics_rec']

        if self.log_tb:
            loggers['tb_logging'].logging(best_values, metrics)
        if self.log_mlflow:
            log_mlflow(best_values, h)
        if self.log_dvclive:
            log_dvclive(self.live, best_values)

        # except BrokenPipeError:
        #     print("\n\n\nProblem with logging stuff!\n\n\n")

    def logging(self, run, cm_logger):
        if self.log_dvclive:
            cm_logger.plot(run, 0, self.unique_unique_labels, 'dvclive')
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
            if self.log_mlflow:
                try:
                    mlflow.log_metric(f'{group}_AUC',
                                      metrics.roc_auc_score(y_true=cats[group], y_score=scores[group], multi_class='ovr'),
                                      step=step)
                except Exception as e:
                    print(f"Error in {group} AUC: {e}")
            if self.log_dvclive:
                # track files
                print(f"Logging {group} predictions to dvclive: not implemented")

    def loop(self, group, optimizer, ae, celoss, loader, lists, traces, nu=1, mapping=True):
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
            data, names, labels, domain, to_rec, not_to_rec, pos_to_rec, neg_to_rec, \
                pos_batch_sample, neg_batch_sample, sets = batch
            data = data.to(self.args.device).float()
            to_rec = to_rec.to(self.args.device).float()

            not_to_rec = not_to_rec.to(self.args.device).float()
            
            # Use autocast for mixed precision training
            with self.autocast_context():
                enc, rec, _, kld = ae(data, to_rec, domain, sampling=sampling, mapping=mapping)
                rec = rec['mean']

                if getattr(self.args, 'classif_loss', 'ce') == 'triplet':
                    feats = enc
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
                pos_enc, _, _, _ = ae(pos_to_rec, pos_to_rec, domain, sampling=True, mapping=mapping)
                neg_enc, _, _, _ = ae(neg_to_rec, neg_to_rec, domain, sampling=True, mapping=mapping)
                if not self.args.train_after_warmup:
                    enc = ae.classifier.net[0](enc)
                    pos_enc = ae.classifier.net[0](pos_enc)
                    neg_enc = ae.classifier.net[0](neg_enc)
                classif_loss = class_triplet(enc, pos_enc, neg_enc)
            else:
                classif_loss = celoss(preds, cats)

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

    def train_bdisc(self, group, optimizer, ae, scheduler, loader):
        """
        Optimize the batch/domain discriminator (DANN discriminator).
        """
        sampling = True if (group in ['train', 'valid']) else False
        celoss = nn.CrossEntropyLoss()
        bclassif_loss = None
        for i, batch in enumerate(loader):
            if group == 'train':
                optimizer.zero_grad()
            data, names, labels, domain, to_rec, *_ = batch
            data = data.to(self.args.device).float()
            to_rec = to_rec.to(self.args.device).float()
            domain = domain.to(self.args.device).long()
            enc, rec, _, kld = ae(data, to_rec, domain, sampling=sampling)
            enc.requires_grad_()
            domain_preds = ae.dann_discriminator(enc)
            bclassif_loss = celoss(domain_preds, domain.long().to(self.args.device))
            if torch.isnan(bclassif_loss):
                print("NAN in batch discriminator loss!")
            bclassif_loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
        return bclassif_loss

    def train_classifier(self, group, optimizer, ae, scheduler, loader, nu=1):
        """
        Optimize only the classifier (AE frozen if train_after_warmup==0).
        """
        sampling = True if (group in ['train', 'valid'] and nu != 0) else False
        celoss = nn.CrossEntropyLoss()
        classif_loss = None
        for i, batch in enumerate(loader):
            if group == 'train' and nu != 0:
                optimizer.zero_grad()
            data, names, labels, domain, to_rec, *_ = batch
            data = data.to(self.args.device).float()
            to_rec = to_rec.to(self.args.device).float()
            if hasattr(self.args, 'train_after_warmup') and self.args.train_after_warmup == 0:
                ae.eval()
                ae.classifier.train()
                for param in ae.parameters():
                    param.requires_grad = False
                for param in ae.classifier.parameters():
                    param.requires_grad = True
            else:
                ae.train()
                for param in ae.parameters():
                    param.requires_grad = True
            enc, rec, _, kld = ae(data, to_rec, domain, sampling=sampling)
            logits = ae.classifier(enc)
            classif_loss = celoss(logits, labels.long().to(self.args.device))
            if torch.isnan(classif_loss):
                print("NAN in classifier loss!")
            classif_loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
        return classif_loss

    def forward_discriminate(self, optimizer_b, ae, celoss, loader):
        # Freezing the layers so the batch discriminator can get some knowledge independently
        # from the part where the autoencoder is trained. Only for DANN
        self.freeze_dlayers(ae)
        sampling = True
        for i, batch in enumerate(loader):
            optimizer_b.zero_grad()
            data, names, labels, domain, to_rec, not_to_rec, pos_to_rec, neg_to_rec, \
                pos_batch_sample, neg_batch_sample, _ = batch
            # data[torch.isnan(data)] = 0
            data = data.to(self.args.device).float()
            to_rec = to_rec.to(self.args.device).float()
            with torch.no_grad():
                enc, rec, _, kld = ae(data, to_rec, domain, sampling=sampling)
            # with torch.enable_grad():
            enc.requires_grad_()
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
        # Build a triplet loss for the batch-effect dloss, or for the additive
        # class-based triplet loss (which reuses the same margin).
        if dloss in ('revTriplet', 'inverseTriplet') or getattr(self.args, 'class_triplet', False):
            triplet_loss = nn.TripletMarginLoss(margin, p=2, swap=True)
        else:
            triplet_loss = None
        # Remember the margin so loops that only have class_triplet can rebuild it.
        self.class_triplet_margin = margin

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
            model: AutoEncoder model
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
            inputs, names, labels, domain, to_rec, not_to_rec, pos_to_rec, neg_to_rec, \
                pos_batch_sample, neg_batch_sample, _ = all_batch
            inputs = inputs.to(self.args.device).float()
            to_rec = to_rec.to(self.args.device).float()
            # verify if domain is str
            if isinstance(domain, str):
                domain = torch.Tensor([[int(y) for y in x.split("_")] for x in domain])

            enc, rec, _, kld = ae(inputs, to_rec, domain, sampling=True, mapping=mapping)
            rec = rec['mean']
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
                pos_enc, _, _, _ = ae(pos_batch_sample, pos_batch_sample, domain, sampling=True)
                neg_enc, _, _, _ = ae(neg_batch_sample, neg_batch_sample, domain, sampling=True)
                dloss = triplet_loss(reverse,
                                     ReverseLayerF.apply(pos_enc, 1),
                                     ReverseLayerF.apply(neg_enc, 1)
                                     )
            elif self.args.dloss == 'inverseTriplet':
                pos_batch_sample, neg_batch_sample = neg_batch_sample.to(self.args.device).float(), pos_batch_sample.to(
                    self.args.device).float()
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
            loss = rec_loss + self.gamma * dloss + self.beta * kld.mean() + l1_loss
            # Additive class-based triplet loss (combinable with any batch dloss)
            if getattr(self.args, 'class_triplet', False):
                ct_loss = compute_class_triplet(
                    ae, enc, pos_to_rec, neg_to_rec, domain, self.args.device,
                    margin=getattr(self, 'class_triplet_margin', 1.0), mapping=mapping,
                )
                loss = loss + getattr(self.args, 'class_triplet_w', 1.0) * ct_loss
            if torch.isnan(loss):
                print("NAN in loss!")
                return 0, ae, warmup
            assert loss.requires_grad, "Total loss does not require grad!"
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
            # self.dom_loss = np.mean(traces['dom_loss'])
            # self.dom_acc = np.mean(traces['dom_acc'])
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
            # TODO change logging with tensorboard. The previous
            if self.log_tb:
                loggers['tb_logging'].logging(values, metrics)
            if self.log_mlflow:
                add_to_mlflow(values, epoch)
            if self.log_dvclive:
                log_dvclive(self.live, values)
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


    def _iter_torch_modules(self):
        """
        Return all torch modules directly attached to the trainer.
        Adjust names if BERNN stores modules in specific attributes.
        """
        for name, obj in self.__dict__.items():
            if isinstance(obj, torch.nn.Module):
                yield name, obj


    def _save_best_model_state(self, epoch, valid_mcc):
        """
        Save a deep copy of all torch module states when validation improves.
        """
        if valid_mcc is None:
            return

        valid_mcc = float(valid_mcc)

        if valid_mcc > self.best_valid_mcc:
            self.best_valid_mcc = valid_mcc
            self.best_epoch = int(epoch)

            self.best_state_dicts = {
                name: copy.deepcopy(module.state_dict())
                for name, module in self._iter_torch_modules()
            }


    def restore_best_model_state(self):
        """
        Restore the best validation model before prediction.
        """
        if not self.best_state_dicts:
            raise RuntimeError(
                "No best model state was saved. Training may have failed before a valid epoch."
            )

        modules = dict(self._iter_torch_modules())

        for name, state in self.best_state_dicts.items():
            if name not in modules:
                raise RuntimeError(f"Cannot restore best model: missing module '{name}'")
            modules[name].load_state_dict(state)

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
            if isinstance(m, KANLinear):
                try:
                    n_active = int(m.count_active_neurons())
                except Exception:
                    n_active = 0
                neurons += n_active
        return neurons


def main():
    import runpy

    runpy.run_module("bernn.dl.train.train_ae", run_name="__main__")


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
    parser.add_argument('--use_mapping', type=int, default=1, help="Use batch mapping for reconstruct")
    parser.add_argument('--bdisc', type=int, default=1)
    parser.add_argument('--n_repeats', type=int, default=5)
    parser.add_argument('--dloss', type=str, default='inverseTriplet')
    parser.add_argument('--class_triplet', type=int, default=0,
                        help='Add a class-based triplet loss on embeddings (combinable with dloss)')
    parser.add_argument('--class_triplet_w', type=float, default=1.0,
                        help='Weight of the class-based triplet loss')
    parser.add_argument('--knn_n_neighbors', type=int, default=5, help='Number of neighbors for persistent KNN (triplet mode)')
    parser.add_argument('--csv_file', type=str, default='unique_genes.csv')
    parser.add_argument('--bad_batches', type=str, default='')  # 0;23;22;21;20;19;18;17;16;15
    parser.add_argument('--remove_zeros', type=int, default=0)
    parser.add_argument('--groupkfold', type=int, default=1)
    parser.add_argument('--dataset', type=str, default='custom')
    parser.add_argument('--bs', type=int, default=32, help='Batch size')
    parser.add_argument('--path', type=str, default='./data/')
    parser.add_argument('--exp_id', type=str, default='default_ae_then_classifier')
    parser.add_argument('--strategy', type=str, default='CU_DEM', help='only for alzheimer dataset')
    parser.add_argument('--n_agg', type=int, default=5, help='Number of trailing values to get stable valid values')
    parser.add_argument('--n_layers', type=int, default=2, help='N layers for classifier')
    parser.add_argument('--log1p', type=int, default=1, help='log1p the data?')
    parser.add_argument('--pool', type=int, default=1, help='only for alzheimer dataset')

    args = parser.parse_args()

    try:
        from bernn.utils.mlflow_compat import mlflow
        mlflow.create_experiment(
            args.exp_id,
            # artifact_location=Path.cwd().joinpath("mlruns").as_uri(),
            # tags={"version": "v1", "priority": "P1"},
        )
    except Exception as e:
        print(f"Error creating experiment: {e}")
        print(f"\n\nExperiment {args.exp_id} already exists\n\n")
    train = TrainAE(args, fix_thres=-1, load_tb=False, log_metrics=True, keep_models=False,
                    log_inputs=False, log_plots=True, log_tb=False,
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
         "values": ['l1', 'minmax', "l2"]},
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

    best_parameters, values, experiment, model = optimize(
        parameters=parameters,
        evaluation_function=train.train,
        objective_name='mcc',
        minimize=False,
        total_trials=args.n_trials,
        random_seed=41,
    )
