# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# CUDA_VISIBLE_DEVICES = ""
import os

from tensorboard.plugins.hparams import api as hp
import tensorflow as tf
from sklearn.metrics import silhouette_score, adjusted_rand_score, adjusted_mutual_info_score
from bernn.dl.models.pytorch.utils.metrics import rKBET, rLISI
import numpy as np
import mlflow
import pandas as pd
import matplotlib
import shap

from bernn.utils.utils import get_unique_labels

matplotlib.use('Agg')
import torch
import matplotlib.pyplot as plt
from matplotlib import rcParams, cycler
from matplotlib.lines import Line2D
from bernn.dl.models.pytorch.utils.plotting import confidence_ellipse
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
from bernn.dl.models.pytorch.utils.metrics import batch_f1_score
from bernn.dl.models.pytorch.utils.utils import log_confusion_matrix, save_roc_curve, save_precision_recall_curve, \
    LogConfusionMatrix
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA

from umap.umap_ import UMAP

from sklearn.manifold import TSNE
from bernn.utils.metrics import calculate_aic, calculate_bic
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# It is useless to run tensorflow on GPU and it takes a lot of GPU RAM for nothing
physical_devices = tf.config.list_physical_devices('CPU')
tf.config.set_visible_devices(physical_devices)


class TensorboardLoggingAE:
    def __init__(self, hparams_filepath, params, variational, zinb, tw, tl, dloss, pseudo,
                 train_after_warmup, berm, args):
        self.params = params
        self.train_after_warmup = train_after_warmup
        self.tw = tw
        self.berm = berm
        self.tl = tl
        self.dloss = dloss
        self.pseudo = pseudo
        self.variational = variational
        self.zinb = zinb
        # self.dann_sets = dann_sets
        # self.dann_plates = dann_batches
        self.hparams_filepath = hparams_filepath
        # self.mz_bin = args.mz_bin
        # self.rt_bin = args.rt_bin
        # self.mzp_bin = args.mz_bin_post
        # self.rtp_bin = args.rt_bin_post
        # self.spd = args.spd
        # self.ms = args.ms_level
        # self.shift = args.shift
        self.berm = args.berm
        # self.log = args.log2
        # self.featselect = args.feature_selection
        self.rec_loss = args.rec_loss
        self.dataset_name = args.dataset
        self.model_name = args.model_name
        self.groupkfold = args.groupkfold
        self.strategy = args.strategy
        self.bad_batches = args.bad_batches
        self.remove_zeros = args.remove_zeros
        self.use_mapping = args.use_mapping
        self.n_emb = args.embeddings_meta
        self.n_meta = args.n_meta
        self.csv_file = args.csv_file
        self.tied_weights = args.tied_weights
        HPARAMS = [
            # hp.HParam('mz_bin', hp.RealInterval(0.0, 100.0)),
            # hp.HParam('rt_bin', hp.RealInterval(0.0, 100.0)),
            # hp.HParam('mzp_bin', hp.RealInterval(0.0, 10000.0)),
            # hp.HParam('rtp_bin', hp.RealInterval(0.0, 10000.0)),
            # hp.HParam('spd', hp.IntInterval(0, 1000)),
            # hp.HParam('ms', hp.Discrete([0, 1])),
            # hp.HParam('shift', hp.Discrete([0, 1])),
            hp.HParam('berm', hp.Discrete(['none', 'combat', 'harmony'])),
            # hp.HParam('log', hp.Discrete(['inloop', 'after', 'none'])),
            hp.HParam('rec', hp.Discrete(['mse', 'l1', 'bce'])),
            # hp.HParam('featselect', hp.Discrete(['mutual_info_classif', 'f_classif'])),

            # hp.HParam('thres', hp.RealInterval(0.0, 1.0)),
            hp.HParam('dropout', hp.RealInterval(0.0, 1.0)),
            hp.HParam('smoothing', hp.RealInterval(0.0, 1.0)),
            hp.HParam('margin', hp.RealInterval(0.0, 10.0)),
            hp.HParam('gamma', hp.RealInterval(0., 100.0)),
            hp.HParam('beta', hp.RealInterval(0., 100.0)),
            hp.HParam('zeta', hp.RealInterval(0., 100.0)),
            hp.HParam('nu', hp.RealInterval(0., 100.0)),
            hp.HParam('layer1', hp.IntInterval(0, 256)),
            hp.HParam('layer2', hp.IntInterval(0, 1024)),
            # hp.HParam('ncols', hp.IntInterval(0, 10000)),
            hp.HParam('lr', hp.RealInterval(1e-8, 1e-2)),
            hp.HParam('wd', hp.RealInterval(1e-15, 1e-1)),
            hp.HParam('scale', hp.Discrete(['binarize', 'minmax', 'standard', 'robust'])),
            hp.HParam('dloss', hp.Discrete(['normae', 'DANN', 'revTriplet', 'inverseTriplet'])),
            # hp.HParam('dann_sets', hp.Discrete([0, 1])),
            # hp.HParam('dann_plates', hp.Discrete([0, 1])),
            hp.HParam('zinb', hp.Discrete([0, 1])),
            hp.HParam('variational', hp.Discrete([0, 1])),
            hp.HParam('tied_w', hp.Discrete([0, 1])),
            hp.HParam('pseudo', hp.Discrete([0, 1])),
            hp.HParam('tripletloss', hp.Discrete([0, 1])),
            hp.HParam('tripletdloss', hp.Discrete([0, 1])),
            hp.HParam('train_after_warmup', hp.Discrete([0, 1]))
        ]

        metrics = [
            hp.Metric('rec_loss', display_name='Rec Loss'),
            hp.Metric('dom_loss', display_name='Domain Loss'),
            hp.Metric('dom_acc', display_name='Domain Accuracy'),
            hp.Metric('train/loss', display_name='Train Loss'),
            hp.Metric('valid/loss', display_name='Valid Loss'),
            hp.Metric('test/loss', display_name='Test Loss'),
            hp.Metric('train/acc', display_name='Train Accuracy'),
            hp.Metric('valid/acc', display_name='Valid Accuracy'),
            hp.Metric('test/acc', display_name='Test Accuracy'),
            hp.Metric('train/top3', display_name='Train top3'),
            hp.Metric('valid/top3', display_name='Valid top3'),
            hp.Metric('test/top3', display_name='Test top3'),
            hp.Metric('train/mcc', display_name='Train MCC'),
            hp.Metric('valid/mcc', display_name='Valid MCC'),
            hp.Metric('test/mcc', display_name='Test MCC'),
            hp.Metric('enc b_euclidean/tot_eucl', display_name='enc b_euclidean/tot_eucl'),
            hp.Metric('enc qc_dist/tot_eucl', display_name='enc qc_dist/tot_eucl'),
            hp.Metric('enc qc_aPCC', display_name='enc qc_aPCC'),
            hp.Metric('enc batch_entropy', display_name='enc qc_aPCC'),
            hp.Metric('rec b_euclidean/tot_eucl', display_name='rec b_euclidean/tot_eucl'),
            hp.Metric('rec qc_dist/tot_eucl', display_name='rec qc_dist/tot_eucl'),
            hp.Metric('rec qc_aPCC', display_name='rec qc_aPCC'),
            hp.Metric('rec batch_entropy', display_name='enc qc_aPCC'),
        ]

        self.writer = tf.summary.create_file_writer(hparams_filepath)
        with self.writer.as_default():
            hp.hparams_config(
                hparams=HPARAMS,
                metrics=metrics,
            )
        self.writer.flush()

    def logging(self, traces, metrics):
        with self.writer.as_default():
            hp.hparams({
                'berm': self.berm,
                'rec': self.rec_loss,
                'gamma': self.params['gamma'],
                'beta': self.params['beta'],
                'zeta': self.params['zeta'],
                'nu': self.params['nu'],
                'dropout': self.params['dropout'],
                'smoothing': self.params['smoothing'],
                'lr': self.params['lr'],
                'margin': self.params['margin'],
                'wd': self.params['wd'],
                'layer1': self.params['layer1'],
                'layer2': self.params['layer2'],
                'scale': self.params['scaler'],
                'tied_w': self.tw,
                'tripletloss': self.tl,
                'dloss': self.dloss,
                'variational': self.variational,
                'zinb': self.zinb,
                'pseudo': self.pseudo,
                'train_after_warmup': self.train_after_warmup,
                'rec_loss': self.rec_loss,
                'dataset_name': self.dataset_name,
                'model_name': self.model_name,
                'groupkfold': self.groupkfold,
                'strategy': self.strategy,
                'bad_batches': self.bad_batches,
                'remove_zeros': self.remove_zeros,
                'use_mapping': self.use_mapping,
                'n_emb': self.n_emb,
                'n_meta': self.n_meta,
                'csv_file': self.csv_file,
                'tied_weights': self.tied_weights
            })  # record the values used in this trial
            tf.summary.scalar('rec_loss', traces['rec_loss'], step=1)
            tf.summary.scalar('dom_loss', traces['dom_loss'], step=1)
            try:
                tf.summary.scalar('dom_acc', traces['dom_acc'], step=1)
            except:
                tf.summary.scalar('dom_acc', traces['dom_acc'][0], step=1)

            # tf.summary.scalar('qc_aPCC', metrics['pool_metrics']['encoded']['all']['qc_aPCC'], step=1)
            # tf.summary.scalar('qc_aPCC_input', metrics['pool_metrics']['inputs']['all']['qc_aPCC'], step=1)
            # tf.summary.scalar('qc_aPCC_rec', metrics['pool_metrics']['recs']['all']['qc_aPCC'], step=1)
            # tf.summary.scalar('qc_dist', metrics['pool_metrics']['encoded']['all']['qc_dist'], step=1)
            # tf.summary.scalar('qc_dist_input', metrics['pool_metrics']['inputs']['all']['qc_dist'], step=1)
            # tf.summary.scalar('qc_dist_rec', metrics['pool_metrics']['recs']['all']['qc_dist'], step=1)

            for g in ['train', 'valid', 'test']:
                tf.summary.scalar(f'{g}/loss', traces[f'{g}_loss'], step=1)
                tf.summary.scalar(f'{g}/acc', traces[f'{g}_acc'], step=1)
                tf.summary.scalar(f'{g}/top3', traces[f'{g}_top3'], step=1)
                tf.summary.scalar(f'{g}/mcc', traces[f'{g}_mcc'], step=1)
            tf.summary.scalar('enc batch_entropy', metrics['batches']['all']['batch_entropy']['enc']['domains'][0], step=1),
            tf.summary.scalar('enc b_euclidean/tot_eucl', traces['enc b_euclidean/tot_eucl'], step=1),
            tf.summary.scalar('enc qc_dist/tot_eucl', traces['enc qc_dist/tot_eucl'], step=1),
            tf.summary.scalar('enc qc_aPCC', traces['enc qc_aPCC'], step=1),
            tf.summary.scalar('rec batch_entropy', metrics['batches']['all']['batch_entropy']['rec']['domains'][0], step=1),
            tf.summary.scalar('rec b_euclidean/tot_eucl', traces['rec b_euclidean/tot_eucl'], step=1),
            tf.summary.scalar('rec qc_dist/tot_eucl', traces['rec qc_dist/tot_eucl'], step=1),
            tf.summary.scalar('rec qc_aPCC', traces['rec qc_aPCC'], step=1),
            
            # if 'batches' in metrics:
            #     for metric in list(metrics['batches']['set'].keys()):
            #         if 'F1' in metric:
            #             for repres in ['enc', 'rec', 'inputs']:
            #                 tf.summary.scalar(f'{metric} {repres} set', metrics['batches']['set'][metric][repres]['set'], step=1)
            #                 tf.summary.scalar(f'{metric} {repres} domains', metrics['batches']['set'][metric][repres]['domains'], step=1)
        self.writer.flush()


def batch_entropy(proba):
    prob_list = []
    for prob in proba:
        loc = 0
        for p in prob:
            loc -= p * np.log(p + 1e-8)
        prob_list += [loc]
    return np.mean(prob_list)


def interactions_mean_matrix(shap_interactions, run, group):
    # Get absolute mean of matrices
    mean_shap = np.abs(shap_interactions).mean(0)
    df = pd.DataFrame(mean_shap, index=X.columns, columns=X.columns)

    # times off diagonal by 2
    df.where(df.values == np.diagonal(df), df.values * 2, inplace=True)

    # display
    plt.figure(figsize=(10, 10), facecolor='w', edgecolor='k')
    sns.set(font_scale=1.5)
    sns.heatmap(df, cmap='coolwarm', annot=True, fmt='.3g', cbar=False)
    plt.yticks(rotation=0)
    f = plt.gcf()
    run[f'shap/interactions_matrix/{group}_values'].upload(f)
    plt.close(f)


def make_summary_plot(df, values, group, run, log_path, category='explainer', mlops='mlflow'):
    shap.summary_plot(values, df, show=False)
    f = plt.gcf()
    if mlops == 'neptune':
        run[f'shap/summary_{category}/{group}_values'].upload(f)
    if mlops == 'mlflow':
        os.makedirs(f'{log_path}/shap/summary_{category}', exist_ok=True)
        plt.savefig(f'{log_path}/shap/summary_{category}/{group}_values.png')
        mlflow.log_figure(f, f'{log_path}/shap/summary_{category}/{group}_values.png')

    plt.close(f)


def make_force_plot(df, values, features, group, run, log_path, category='explainer', mlops='mlflow'):
    shap.force_plot(df, values, features=features, show=False)
    f = plt.gcf()
    if mlops == 'neptune':
        run[f'shap/force_{category}/{group}_values'].upload(f)
    if mlops == 'mlflow':
        os.makedirs(f'{log_path}/shap/force_{category}', exist_ok=True)
        plt.savefig(f'{log_path}/shap/force_{category}/{group}_values.png')
        mlflow.log_figure(f, f'{log_path}/shap/force_{category}/{group}_values.png')

    plt.close(f)


def make_deep_beeswarm(df, values, group, run, log_path, category='explainer', mlops='mlflow'):
    shap.summary_plot(values, feature_names=df.columns, features=df, show=False)
    f = plt.gcf()
    if mlops == 'neptune':
        run[f'shap/beeswarm_{category}/{group}_values'].upload(f)
    if mlops == 'mlflow':
        os.makedirs(f'{log_path}/shap/beeswarm_{category}', exist_ok=True)
        plt.savefig(f'{log_path}/shap/beeswarm_{category}/{group}_values.png')
        mlflow.log_figure(f, f'{log_path}/shap/beeswarm_{category}/{group}_values.png')

    plt.close(f)


def make_decision_plot(df, values, misclassified, feature_names, group, run, log_path, category='explainer', mlops='mlflow'):
    shap.decision_plot(df, values, feature_names=list(feature_names), show=False, link='logit', highlight=misclassified)
    f = plt.gcf()
    if mlops == 'neptune':
        run[f'shap/decision_{category}/{group}_values'].upload(f)
        run[f'shap/decision_{category}/{group}_values'].upload(f)
    if mlops == 'mlflow':
        os.makedirs(f'{log_path}/shap/decision_{category}', exist_ok=True)
        plt.savefig(f'{log_path}/shap/decision_{category}/{group}_values.png')
        mlflow.log_figure(f, f'{log_path}/shap/decision_{category}/{group}_values.png')
    plt.close(f)


def make_decision_deep(df, values, misclassified, feature_names, group, run, log_path, category='explainer', mlops='mlflow'):
    shap.decision_plot(df, values, feature_names=list(feature_names), show=False, link='logit', highlight=misclassified)
    f = plt.gcf()
    if mlops == 'neptune':
        run[f'shap/decision_{category}/{group}_values'].upload(f)
        run[f'shap/decision_{category}/{group}_values'].upload(f)
    if mlops == 'mlflow':
        os.makedirs(f'{log_path}/shap/decision_{category}', exist_ok=True)
        plt.savefig(f'{log_path}/shap/decision_{category}/{group}_values.png')
        mlflow.log_figure(f, f'{log_path}/shap/decision_{category}/{group}_values.png')
    plt.close(f)


def make_multioutput_decision_plot(df, values, group, run, log_path, category='explainer', mlops='mlflow'):
    shap.multioutput_decision_plot(values, df, show=False)
    f = plt.gcf()
    if mlops == 'neptune':
        run[f'shap/multioutput_decision_{category}/{group}_values'].upload(f)
    if mlops == 'mlflow':
        os.makedirs(f'{log_path}/shap/multioutput_decision_{category}', exist_ok=True)
        plt.savefig(f'{log_path}/shap/multioutput_decision_{category}/{group}_values.png')
        mlflow.log_figure(f, f'{log_path}/shap/multioutput_decision_{category}/{group}_values.png')
    plt.close(f)


def make_group_difference_plot(values, mask, group, run, log_path, category='explainer', mlops='mlflow'):
    shap.group_difference_plot(values, mask, show=False)
    f = plt.gcf()
    if mlops == 'neptune':
        # run[f'shap/gdiff_{category}/{group}'].upload(f)
        run[f'shap/gdiff_{category}/{group}'].upload(f)
    if mlops == 'mlflow':
        os.makedirs(f'{log_path}/shap/gdiff_{category}', exist_ok=True)
        plt.savefig(f'{log_path}/shap/gdiff_{category}/{group}_values.png')
        mlflow.log_figure(f, f'{log_path}/shap/gdiff_{category}/{group}_values.png')
    plt.close(f)


def make_beeswarm_plot(values, group, run, log_path, category='explainer', mlops='mlflow'):
    shap.plots.beeswarm(values, max_display=20, show=False)
    f = plt.gcf()
    if mlops == 'neptune':
        # run[f'shap/beeswarm_{category}/{group}'].upload(f)
        run[f'shap/beeswarm_{category}/{group}'].upload(f)
    if mlops == 'mlflow':
        os.makedirs(f'{log_path}/shap/beeswarm_{category}', exist_ok=True)
        plt.savefig(f'{log_path}/shap/beeswarm_{category}/{group}_values.png')
        mlflow.log_figure(f, f'{log_path}/shap/beeswarm_{category}/{group}_values.png')
    plt.close(f)


def make_heatmap(values, group, run, log_path, category='explainer', mlops='mlflow'):
    shap.plots.heatmap(values, instance_order=values.values.sum(1).argsort(), max_display=20, show=False)
    f = plt.gcf()
    if mlops == 'neptune':
        # run[f'shap/heatmap_{category}/{group}'].upload(f)
        run[f'shap/heatmap_{category}/{group}'].upload(f)
    if mlops == 'mlflow':
        os.makedirs(f'{log_path}/shap/heatmap_{category}', exist_ok=True)
        plt.savefig(f'{log_path}/shap/heatmap_{category}/{group}_values.png')
        mlflow.log_figure(f, f'{log_path}/shap/heatmap_{category}/{group}_values.png')
    plt.close(f)


def make_heatmap_deep(values, group, run, log_path, category='explainer', mlops='mlflow'):

    shap.plots.heatmap(pd.DataFrame(values), instance_order=values.sum(1).argsort(), max_display=20, show=False)
    f = plt.gcf()
    if mlops == 'neptune':
        # run[f'shap/heatmap_{category}/{group}'].upload(f)
        run[f'shap/heatmap_{category}/{group}'].upload(f)
    if mlops == 'mlflow':
        os.makedirs(f'{log_path}/shap/heatmap_{category}', exist_ok=True)
        plt.savefig(f'{log_path}/shap/heatmap_{category}/{group}_values.png')
        mlflow.log_figure(f, f'{log_path}/shap/heatmap_{category}/{group}_values.png')
    plt.close(f)


def make_barplot(df, y, values, group, run, log_path, category='explainer', mlops='mlflow'):
    clustering = shap.utils.hclust(df, y, metric='correlation')  # cluster_threshold=0.9
    # shap.plots.bar(values, max_display=20, show=False, clustering=clustering)
    shap.plots.bar(values, max_display=20, show=False, clustering=clustering, clustering_cutoff=0.5)
    f = plt.gcf()
    if mlops == 'neptune':
        run[f'shap/bar_{category}/{group}'].upload(f)
    if mlops == 'mlflow':
        os.makedirs(f'{log_path}/shap/bar_{category}', exist_ok=True)
        plt.savefig(f'{log_path}/shap/bar_{category}/{group}_values.png')
        mlflow.log_figure(f, f'{log_path}/shap/bar_{category}/{group}_values.png')
    plt.close(f)


def make_bar_plot(df, values, group, run, log_path, category='explainer', mlops='mlflow'):
    shap.bar_plot(values, max_display=40, feature_names=df.columns, show=False)
    f = plt.gcf()
    if mlops == 'neptune':
        # run[f'shap/barold_{category}/{group}'].upload(f)
        run[f'shap/barold_{category}/{group}'].upload(f)
    if mlops == 'mlflow':
        os.makedirs(f'{log_path}/shap/barold_{category}', exist_ok=True)
        plt.savefig(f'{log_path}/shap/barold_{category}/{group}_values.png')
        mlflow.log_figure(f, f'{log_path}/shap/barold_{category}/{group}_values.png')
    plt.close(f)


def make_dependence_plot(df, values, var, group, run, log_path, category='explainer', mlops='mlflow'):
    shap.dependence_plot(var, values[1], df, show=False)
    f = plt.gcf()
    if mlops == 'neptune':
        run[f'shap/dependence_{category}/{group}'].upload(f)
    if mlops == 'mlflow':
        os.makedirs(f'{log_path}/shap/dependence_{category}', exist_ok=True)
        plt.savefig(f'{log_path}/shap/dependence_{category}/{group}_values.png')
        mlflow.log_figure(f, f'{log_path}/shap/dependence_{category}/{group}_values.png')
    plt.close(f)


def log_explainer(model, x_df, labels, group, run, cats, log_path, device):
    unique_labels = np.unique(labels)
    # The explainer doesn't like tensors, hence the f function
    explainer = shap.Explainer(model.to(device), x_df, max_evals=2 * x_df.shape[1] + 1)

    # Get the shap values from my test data
    shap_values = explainer(x_df)

    for i, label in enumerate(unique_labels):
        if i == len(shap_values):
            break
        shap_values_df = pd.DataFrame(shap_values[i].data.reshape(1, -1), columns=x_df.columns)
        try:
            shap_values_df.to_csv(f"{log_path}/{group}_permutation_shap_{label}.csv")
        except:
            pass

    make_barplot(x_df, labels, shap_values[:, :, 0], group, run, 'Kernel')

    # Summary plot
    make_summary_plot(x_df, shap_values[:, :, 0], group, run, 'PermutExplainer')

    make_beeswarm_plot(shap_values[:, :, 0], group, run, 'PermutExplainer')
    make_heatmap(shap_values[:, :, 0], group, run, 'PermutExplainer')

    mask = np.array([np.argwhere(x[0] == 1)[0][0] for x in cats])
    make_group_difference_plot(x_df.sum(1).to_numpy(), mask, group, run, 'PermutExplainer')


def log_deep_explainer(model, x_df, misclassified, labels, group, run, cats, log_path, mlops, device):
    unique_labels = np.unique(labels)
    # The explainer doesn't like tensors, hence the f function
    explainer = shap.DeepExplainer(model.to(device), torch.Tensor(x_df.values).to(device))

    # Get the shap values from my test data
    shap_values = explainer.shap_values(torch.Tensor(x_df.values).to(device))

    # Summary plot
    make_summary_plot(x_df, shap_values, group, run, log_path, 'DeepExplainer', mlops)
    make_force_plot(explainer.expected_value[0], shap_values[0][0], x_df.columns, group, run, log_path, 'DeepExplainer', mlops)
    make_deep_beeswarm(x_df, shap_values[0], group, run, log_path, 'DeepExplainer', mlops)
    make_decision_deep(explainer.expected_value[0], shap_values[0], misclassified, x_df.columns, group, run, 'DeepExplainer')

    for i, label in enumerate(unique_labels):
        if i == len(shap_values):
            break
        shap_values_df = pd.DataFrame(shap_values[i], columns=x_df.columns, index=x_df.index)
        try:
            shap_values_df.to_csv(f"{log_path}/{group}_deep_shap_{label}.csv")
        except:
            pass

    try:
        make_dependence_plot(x_df, shap_values, 'APOE', group, run, log_path, 'DeepExplainer', mlops)
    except:
        pass

    # mask = np.array([np.argwhere(x[0] == 1)[0][0] for x in cats])
    # make_group_difference_plot(x_df, mask, group, run, 'DeepExplainer')


def log_kernel_explainer(model, x_df, misclassified, labels, group, run, cats, log_path):
    unique_labels = np.unique(labels)

    f = lambda x: model.to('cpu')(torch.from_numpy(x)).detach().cpu().numpy()

    # Convert my pandas dataframe to numpy
    data = x_df.to_numpy(dtype=np.float32)

    # The explainer doesn't like tensors, hence the f function
    explainer = shap.KernelExplainer(f, data)

    # Get the shap values from my test data
    df = pd.DataFrame(data, columns=x_df.columns)
    shap_values = explainer.shap_values(df)
    # shap_interaction = explainer.shap_interaction_values(X_test)
    shap_values_df = pd.DataFrame(np.concatenate(shap_values), columns=x_df.columns)
    for i, label in enumerate(unique_labels):
        if i == len(shap_values):
            break
        shap_values_df.iloc[i].to_csv(f"{log_path}/{group}_kernel_shap_{label}.csv")
    # shap_values = pd.DataFrame(np.concatenate(s))
    # Summary plot
    make_summary_plot(x_df, shap_values, group, run, 'Kernel')

    make_bar_plot(x_df, shap_values_df.iloc[1], group, run, 'localKernel')

    make_decision_plot(explainer.expected_value[0], shap_values[0], misclassified, x_df.columns, group, run, 'Kernel')

    mask = np.array([np.argwhere(x[0] == 1)[0][0] for x in cats])
    make_group_difference_plot(x_df.sum(1).to_numpy(), mask, group, run, 'Kernel')


def log_shap(run, ae, best_lists, cols, n_meta, mlops, log_path, device, log_deep_only=True):
    # explain all the predictions in the test set
    # explainer = shap.KernelExplainer(svc_linear.predict_proba, X_train[:100])
    os.makedirs(log_path, exist_ok=True)
    for group in ['valid', 'test']:
        if n_meta > 0:
            X = np.concatenate((
                np.concatenate(best_lists[group]['inputs']),
                # np.concatenate(best_lists[group]['age']).reshape(-1, 1),
                # np.concatenate(best_lists[group]['gender']).reshape(-1, 1),
            ), 1)
            X_test = torch.Tensor(X).to(device)
            X_test_df = pd.DataFrame(X, columns=list(cols) + ['age', 'sex'])
        else:
            X = np.concatenate(best_lists[group]['inputs'])
            X_test = torch.Tensor(X).to(device)
            X_test_df = pd.DataFrame(X, columns=list(cols))

        # explainer = shap.DeepExplainer(ae, X_test)
        # explanation = shap.Explanation(X_test, feature_names=X_test_df.columns)
        # explanation.values = explanation.values.detach().cpu().numpy()
        misclassified = [pred != label for pred, label in zip(np.concatenate(best_lists[group]['preds']).argmax(1),
                                                              np.concatenate(best_lists[group]['cats']).argmax(1))]
        try:
            log_deep_explainer(ae, X_test_df, misclassified, np.concatenate(best_lists[group]['labels']),
                           group, run, best_lists[group]['cats'], log_path, mlops, device
                           )
        except:
            pass
        if not log_deep_only:
            # TODO Problem with not enough memory...
            try:
                log_explainer(ae, X_test_df, np.concatenate(best_lists[group]['labels']),
                          group, run, best_lists[group]['cats'], log_path, device
                          )
            except:
                pass
            try:
                log_kernel_explainer(ae, X_test_df,
                                 misclassified,
                                 np.concatenate(best_lists[group]['labels']),
                                 group, run, best_lists[group]['cats'], log_path
                                 )
            except:
                pass

def log_neptune(run, traces):
    if 'rec_loss' in traces:
        if not np.isnan(traces['rec_loss']):
            try:
                run["rec_loss"].log(traces['rec_loss'])
            except:
                print(f"\n\n\nPROBLEM HERE:::: {traces['rec_loss']}\n\n\n")
        if not np.isnan(traces['dom_loss']):
            run["dom_loss"].log(traces['dom_loss'])
        if not np.isnan(traces['dom_acc']):
            try:
                run["dom_acc"].log(traces['dom_acc'])
            except:
                run["dom_acc"].log(traces['dom_acc'][0])

    # tf.summary.scalar('qc_aPCC', metrics['pool_metrics']['encoded']['all']['qc_aPCC'], step=1)
    # tf.summary.scalar('qc_aPCC_input', metrics['pool_metrics']['inputs']['all']['qc_aPCC'], step=1)
    # tf.summary.scalar('qc_aPCC_rec', metrics['pool_metrics']['recs']['all']['qc_aPCC'], step=1)
    # tf.summary.scalar('qc_dist', metrics['pool_metrics']['encoded']['all']['qc_dist'], step=1)
    # tf.summary.scalar('qc_dist_input', metrics['pool_metrics']['inputs']['all']['qc_dist'], step=1)
    # tf.summary.scalar('qc_dist_rec', metrics['pool_metrics']['recs']['all']['qc_dist'], step=1)

    for g in ['train', 'valid', 'test']:
        run[f'{g}/loss'].log(traces[f'{g}_loss'])
        run[f'{g}/acc'].log(traces[f'{g}_acc'])
        run[f'{g}/top3'].log(traces[f'{g}_top3'])
        run[f'{g}/mcc'].log(traces[f'{g}_mcc'])
    for rep in ['enc', 'rec']:
        for g in ['all', 'train', 'valid', 'test']:
            try:
                run['enc b_euclidean/tot_eucl'].log(traces['enc b_euclidean/tot_eucl']),
                run['enc qc_dist/tot_eucl'].log(traces['enc qc_dist/tot_eucl']),
                run['enc qc_aPCC'].log(traces['enc qc_aPCC']),
                run['enc batch_entropy'].log(traces['enc batch_entropy']),
                run['rec b_euclidean/tot_eucl'].log(traces['rec b_euclidean/tot_eucl']),
                run['rec qc_dist/tot_eucl'].log(traces['rec qc_dist/tot_eucl']),
                run['rec qc_aPCC'].log(traces['rec qc_aPCC']),
                run['rec batch_entropy'].log(traces['rec batch_entropy']),
            except:
                pass


def log_mlflow(traces, step):
    if 'rec_loss' in traces:
        if not np.isnan(traces['rec_loss']):
            try:
                mlflow.log_metric("rec_loss", traces['rec_loss'], step)
            except:
                print(f"\n\n\nPROBLEM HERE:::: {traces['rec_loss']}\n\n\n")
        if not np.isnan(traces['dom_loss']):
            mlflow.log_metric("dom_loss", traces['dom_loss'], step)
        if not np.isnan(traces['dom_acc']):
            try:
                mlflow.log_metric("dom_acc", traces['dom_acc'], step)
            except:
                mlflow.log_metric("dom_acc", traces['dom_acc'], step)

    # tf.summary.scalar('qc_aPCC', metrics['pool_metrics']['encoded']['all']['qc_aPCC'], step=1)
    # tf.summary.scalar('qc_aPCC_input', metrics['pool_metrics']['inputs']['all']['qc_aPCC'], step=1)
    # tf.summary.scalar('qc_aPCC_rec', metrics['pool_metrics']['recs']['all']['qc_aPCC'], step=1)
    # tf.summary.scalar('qc_dist', metrics['pool_metrics']['encoded']['all']['qc_dist'], step=1)
    # tf.summary.scalar('qc_dist_input', metrics['pool_metrics']['inputs']['all']['qc_dist'], step=1)
    # tf.summary.scalar('qc_dist_rec', metrics['pool_metrics']['recs']['all']['qc_dist'], step=1)

    for g in ['train', 'valid', 'test']:
        mlflow.log_metric(f'{g}/loss', traces[f'{g}_loss'], step)
        mlflow.log_metric(f'{g}/acc', traces[f'{g}_acc'], step)
        mlflow.log_metric(f'{g}/top3', traces[f'{g}_top3'], step)
        mlflow.log_metric(f'{g}/mcc', traces[f'{g}_mcc'], step)
    for rep in ['enc', 'rec']:
        for g in ['all', 'train', 'valid', 'test']:
            try:
                mlflow.log_metric(f"{rep} {g} batch_entropy", traces['batches'][g]['batch_entropy'][rep]['domains'][-1], step)
                mlflow.log_metric(f"{rep} {g} adjusted_rand_score", traces['batches'][g]['adjusted_rand_score'][rep]['domains'][-1], step)
                mlflow.log_metric(f"{rep} {g} adjusted_mutual_info_score", traces['batches'][g]['adjusted_mutual_info_score'][rep]['domains'][-1], step)
                mlflow.log_metric(f"{rep} {g} b_euclidean/tot_eucl", traces['pool_metrics'][rep][g]['[b_euclidean/tot_eucl]'], step)
                mlflow.log_metric(f"{rep} {g} qc_dist/tot_eucl", traces['pool_metrics'][rep][g]['[b_euclidean/tot_eucl]'], step)
                mlflow.log_metric(f"{rep} {g} qc_aPCC", traces['pool_metrics'][rep][g]['[b_euclidean/tot_eucl]'], step)
                mlflow.log_metric(f"{rep} {g} silhouette", traces['batches'][g]['silhouette'][rep]['domains'][-1], step)
                mlflow.log_metric(f"{rep} {g} lisi", traces['batches'][g]['lisi'][rep]['domains'][-1], step)
                mlflow.log_metric(f"{rep} {g} kbet", traces['batches'][g]['kbet'][rep]['domains'][-1], step)
            except:
                pass


def log_PAD(lists, values, model):
    pass


def get_metrics(lists, values, model):
    # sets are grouped togheter for a single metric
    from sklearn.metrics import silhouette_score, adjusted_rand_score, adjusted_mutual_info_score
    from bernn.dl.models.pytorch.utils.metrics import rKBET, rLISI
    from sklearn.neighbors import KNeighborsClassifier
    
    knns = {info: {repres: KNeighborsClassifier(n_neighbors=20) for repres in ['set', 'domains', 'labels', 'times']} for
            info in ['enc', 'rec', 'inputs']}
    # classes = []
    # domains = []
    # sets = []
    # encoded_values = []
    # rec_values = []
    # inputs = []

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    for group in ['all', 'train', 'valid', 'test']:
        if len(lists[group]['inputs']) > 0:
            try:
                classes = np.array(np.concatenate(lists[group]['classes']), np.int)
            except:
                pass

            if group == 'all':
                knns['enc']['domains'].fit(
                    np.concatenate(lists[group]['encoded_values']),
                    np.concatenate(lists[group]['domains'])
                )
                if len(lists[group]['rec_values']) > 0:
                    knns['rec']['domains'].fit(
                        np.concatenate(lists[group]['rec_values']),
                        np.concatenate(lists[group]['domains'])
                    )
                knns['inputs']['domains'].fit(
                    np.concatenate(lists[group]['inputs']),
                    np.concatenate(lists[group]['domains'])
                )

                try:
                    knns['enc']['labels'].fit(
                        np.concatenate(lists[group]['encoded_values']),
                        np.concatenate(lists[group]['classes'])
                    )
                    if len(lists[group]['rec_values']) > 0:
                        knns['rec']['labels'].fit(
                            np.concatenate(lists[group]['rec_values']),
                            np.concatenate(lists[group]['classes'])
                        )
                    knns['inputs']['labels'].fit(
                        np.concatenate(lists[group]['inputs']),
                        np.concatenate(lists[group]['classes'])
                    )
                except:
                    pass

            lists[group]['domain_proba'] = knns['inputs']['domains'].predict_proba(
                np.concatenate(lists[group]['inputs']),
            )
            lists[group]['domain_preds'] = knns['inputs']['domains'].predict(
                np.concatenate(lists[group]['inputs']),
            )
            values[group]['batch_entropy']['inputs']['domains'] = [batch_entropy(lists[group]['domain_proba'])]

            lists[group]['domain_proba'] = knns['enc']['domains'].predict_proba(
                np.concatenate(lists[group]['encoded_values']),
            )
            lists[group]['domain_preds'] = knns['enc']['domains'].predict(
                np.concatenate(lists[group]['encoded_values']),
            )
            values[group]['batch_entropy']['enc']['domains'] = [batch_entropy(lists[group]['domain_proba'])]

            if len(lists[group]['rec_values']) > 0:
                lists[group]['domain_proba'] = knns['rec']['domains'].predict_proba(
                    np.concatenate(lists[group]['rec_values']),
                )
                lists[group]['domain_preds'] = knns['rec']['domains'].predict(
                    np.concatenate(lists[group]['rec_values']),
                )
            values[group]['batch_entropy']['rec']['domains'] = [batch_entropy(lists[group]['domain_proba'])]

            # for metric, funct in zip(['lisi', 'silhouette', 'kbet'], [rLISI, silhouette_score, rKBET]):
            for metric, funct in zip(['lisi', 'silhouette'], [rLISI, silhouette_score]):
                try:
                    values[group][metric]['enc']['labels'] = [funct(np.concatenate(lists[group]['encoded_values']),
                                                                np.concatenate(lists[group]['classes']))]
                except:
                    values[group][metric]['enc']['labels'] = [-1]
                try:
                    values[group][metric]['enc']['domains'] = [funct(np.concatenate(lists[group]['encoded_values']),
                                                                     np.concatenate(lists[group]['domains']))]
                except:
                    values[group][metric]['enc']['domains'] = [-1]

                try:
                    values[group][metric]['rec']['labels'] = [funct(np.concatenate(lists[group]['rec_values']),
                                                                    np.concatenate(lists[group]['classes']))]
                except:
                    values[group][metric]['rec']['labels'] = [-1]

                try:
                    values[group][metric]['rec']['domains'] = [funct(np.concatenate(lists[group]['rec_values']),
                                                                     np.concatenate(lists[group]['domains']))]
                except:
                    values[group][metric]['rec']['domains'] = [-1]

                try:
                    values[group][metric]['inputs']['labels'] = [funct(np.concatenate(lists[group]['inputs']),
                                                                       np.concatenate(lists[group]['classes']))]
                except:
                    values[group][metric]['inputs']['labels'] = [-1]

                try:
                    values[group][metric]['inputs']['domains'] = [funct(np.concatenate(lists[group]['inputs']),
                                                                        np.concatenate(lists[group]['domains']))]
                except:
                    values[group][metric]['inputs']['domains'] = [-1]

            for metric, funct in zip(
                    ['adjusted_rand_score', 'adjusted_mutual_info_score'],
                    [adjusted_rand_score, adjusted_mutual_info_score]):


                try:
                    lists[group]['domain_preds'] = knns['enc']['domains'].predict(
                        np.concatenate(lists[group]['encoded_values']),
                    )
                except:
                    pass

                try:
                    values[group][metric]['enc']['domains'] = [funct(np.concatenate(lists[group]['domains']),
                                                                     lists[group]['domain_preds'])]
                except:
                    pass


                try:
                    values[group][metric]['enc']['labels'] = [funct(np.concatenate(lists[group]['domains']),
                                                                    lists[group]['domain_preds'])]
                except:
                    pass

                # lists[group]['domain_preds'] = knns['enc']['labels'].predict(
                #     np.concatenate(lists[group]['encoded_values']),
                # )

                #################


                try:
                    lists[group]['domain_preds'] = knns['rec']['labels'].predict(
                        np.concatenate(lists[group]['rec_values']),
                    )
                except:
                    pass
                try:
                    values[group][metric]['rec']['domains'] = [funct(np.concatenate(lists[group]['domains']),
                                                                     lists[group]['domain_preds'])]
                except:
                    pass

                try:
                    values[group][metric]['rec']['labels'] = [funct(np.concatenate(lists[group]['labels']),
                                                                    lists[group]['domain_preds'])]
                except:
                    pass

                # lists[group]['domain_preds'] = knns['rec']['labels'].predict(
                #     np.concatenate(lists[group]['rec']),
                # )

                #################

                try:
                    lists[group]['domain_preds'] = knns['inputs']['domains'].predict(
                        np.concatenate(lists[group]['inputs']),
                    )
                except:
                    pass

                try:
                    values[group][metric]['inputs']['domains'] = [funct(np.concatenate(lists[group]['domains']),
                                                                        lists[group]['domain_preds'])]
                except:
                    pass

                try:
                    lists[group]['domain_preds'] = knns['inputs']['labels'].predict(
                        np.concatenate(lists[group]['inputs']),
                    )
                except:
                    pass

                try:
                    values[group][metric]['inputs']['labels'] = [funct(np.concatenate(lists[group]['labels']),
                                                                       lists[group]['domain_preds'])]
                except:
                    pass

    return values


def log_ORD(ordin, logger, data, uniques, mlops, epoch, transductive=False):
    model = ordin['model']
    for f in ['inputs', 'batches', 'labels']:
        for g in ['train', 'valid', 'test']:
            # print(f"{f} {g}")
            try:
                data[f][g] = np.concatenate((data[f][g], data[f][f'{g}_pool']))
            except:
                pass
    if transductive:
        model.fit(np.concatenate((data['inputs']['train'], data['inputs']['valid'], data['inputs']['test'])))
        pcs_train = model.transform(data['inputs']['train'])
        if "transductive" not in ordin['name']:
            ordin['name'] += "_transductive"
    else:
        pcs_train = model.fit_transform(data['inputs']['train'])
    if data['inputs']['valid'] is not None:
        pcs_valid = model.transform(data['inputs']['valid'])
    else:
        pcs_valid = np.array([])
    if data['inputs']['test'] is not None:
        pcs_test = model.transform(data['inputs']['test'])
    else:
        pcs_test = np.array([])
        test_labels = np.array([])

    pcs_train_df = pd.DataFrame(data=pcs_train, columns=['PC 1', 'PC 2'])
    pcs_valid_df = pd.DataFrame(data=pcs_valid, columns=['PC 1', 'PC 2'])
    pcs_test_df = pd.DataFrame(data=pcs_test, columns=['PC 1', 'PC 2'])
    for name in list(uniques.keys()):
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111)

        try:
            ev = model.explained_variance_ratio_
            pc1 = 'Component_1 : ' + str(np.round(ev[0] * 100, decimals=2)) + "%"
            pc2 = 'Component_2 : ' + str(np.round(ev[1] * 100, decimals=2)) + "%"
        except:
            pc1 = 'Component_1'
            pc2 = 'Component_2'

        ax.set_xlabel(pc1, fontsize=15)
        ax.set_ylabel(pc2, fontsize=15)
        ax.set_title(f"2 component {ordin['name']}", fontsize=20)

        num_targets = len(uniques[name])
        cmap = plt.cm.tab20

        cols = cmap(np.linspace(0, 1, len(uniques[name]) + 1))
        colors = rcParams['axes.prop_cycle'] = cycler(color=cols)
        colors_list = {name: [] for name in ['train', 'valid', 'test']}
        data1_list = {name: [] for name in ['train', 'valid', 'test']}
        data2_list = {name: [] for name in ['train', 'valid', 'test']}
        new_labels = {name: [] for name in ['train', 'valid', 'test']}
        new_cats = {name: [] for name in ['train', 'valid', 'test']}

        ellipses = []
        unique_cats_train = np.array([])
        for df_name, df, labels in zip(['train', 'valid', 'test'],
                                       [pcs_train_df, pcs_valid_df, pcs_test_df],
                                       [data[name]['train'], data[name]['valid'], data[name]['test']]):
            for t, target in enumerate(uniques[name]):
                indices_to_keep = [True if x == target else False for x in list(labels)]
                data1 = list(df.loc[indices_to_keep, 'PC 1'])
                new_labels[df_name] += [target for _ in range(len(data1))]
                new_cats[df_name] += [target for _ in range(len(data1))]

                data2 = list(df.loc[indices_to_keep, 'PC 2'])
                data1_list[df_name] += [data1]
                data2_list[df_name] += [data2]
                colors_list[df_name] += [np.array([[cols[t]] * len(data1)])]
                if len(indices_to_keep) > 1 and df_name == 'train_data' or target not in unique_cats_train:
                    unique_cats_train = np.unique(np.concatenate((new_labels[df_name], unique_cats_train)))
                    try:
                        confidence_ellipses = confidence_ellipse(np.array(data1), np.array(data2), ax, 1.5,
                                                                 edgecolor=cols[t],
                                                                 train_set=True)
                        ellipses += [confidence_ellipses[1]]
                    except:
                        pass

        for df_name, marker in zip(list(data1_list.keys()), ['o', 'x', '*']):
            data1_vector = np.hstack([d for d in data1_list[df_name] if len(d) > 0]).reshape(-1, 1)
            colors_vector = np.hstack([d for d in colors_list[df_name] if d.shape[1] > 0]).squeeze()
            data2_vector = np.hstack(data2_list[df_name]).reshape(-1, 1)
            data_colors_vector = np.concatenate((data1_vector, data2_vector, colors_vector), axis=1)
            data2 = data_colors_vector[:, 1]
            col = data_colors_vector[:, 2:]
            data1 = data_colors_vector[:, 0]

            ax.scatter(data1, data2, s=50, alpha=1.0, c=col, label=new_labels[df_name], marker=marker)
            custom_lines = [Line2D([0], [0], color=cmap(x), lw=4) for x in np.linspace(0, 1, len(uniques[name]) + 1)]
            ax.legend(custom_lines, uniques[name].tolist())

        if mlops == "tensorboard":
            logger.add_figure(f'{ordin["name"]}_{name}', fig, epoch)
        elif mlops == "neptune":
            logger[f'{ordin["name"]}/{name}'].upload(fig)
        elif mlops == "mlflow":
            mlflow.log_figure(fig, f'{ordin["name"]}/{name}.png')
        else:
            plt.show()

        plt.close(fig)


def log_LDA(ordin, logger, data, uniques, mlops, epoch):
    for name in list(uniques.keys()):
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111)

        if len(uniques[name]) > 2:
            n_comp = 2
        else:
            n_comp = 1
        model = ordin['model'](n_components=n_comp)

        pcs_train = model.fit_transform(data['inputs']['train'], data[name]['train'])
        pcs_valid = model.transform(data['inputs']['valid'])
        pcs_test = model.transform(data['inputs']['test'])

        if n_comp > 1:
            pcs_train_df = pd.DataFrame(data=pcs_train, columns=['LD1', 'LD2'])
            pcs_valid_df = pd.DataFrame(data=pcs_valid, columns=['LD1', 'LD2'])
            pcs_test_df = pd.DataFrame(data=pcs_test, columns=['LD1', 'LD2'])
        else:
            pcs_train_df = pd.DataFrame(data=pcs_train, columns=['LD1'])
            pcs_valid_df = pd.DataFrame(data=pcs_valid, columns=['LD1'])
            pcs_test_df = pd.DataFrame(data=pcs_test, columns=['LD1'])

        try:
            ev = model.explained_variance_ratio_
            pc1 = 'Component_1 : ' + str(np.round(ev[0] * 100, decimals=2)) + "%"
            pc2 = 'Component_2 : ' + str(np.round(ev[1] * 100, decimals=2)) + "%"
        except:
            pc1 = 'Component_1'
            pc2 = 'Component_2'

        ax.set_xlabel(pc1, fontsize=15)
        ax.set_ylabel(pc2, fontsize=15)
        ax.set_title(f"2 component {ordin['name']}", fontsize=20)

        num_targets = len(uniques[name])
        cmap = plt.cm.tab20

        cols = cmap(np.linspace(0, 1, len(uniques[name]) + 1))
        colors = rcParams['axes.prop_cycle'] = cycler(color=cols)
        colors_list = {name: [] for name in ['train_data', 'valid_data', 'test_data']}
        data1_list = {name: [] for name in ['train_data', 'valid_data', 'test_data']}
        data2_list = {name: [] for name in ['train_data', 'valid_data', 'test_data']}
        new_labels = {name: [] for name in ['train_data', 'valid_data', 'test_data']}
        new_cats = {name: [] for name in ['train_data', 'valid_data', 'test_data']}

        ellipses = []
        unique_cats_train = np.array([])
        for df_name, df, labels in zip(['train_data', 'valid_data', 'test_data'],
                                       [pcs_train_df, pcs_valid_df, pcs_test_df],
                                       [data[name]['train'], data[name]['valid'], data[name]['test']]):
            for t, target in enumerate(uniques[name]):
                indices_to_keep = [True if x == target else False for x in
                                   list(labels)]  # 0 is the name of the column with target values
                data1 = list(df.loc[indices_to_keep, 'LD1'])
                new_labels[df_name] += [target for _ in range(len(data1))]
                new_cats[df_name] += [target for _ in range(len(data1))]

                data1_list[df_name] += [data1]
                colors_list[df_name] += [np.array([[cols[t]] * len(data1)])]
                if n_comp > 1:
                    data2 = list(df.loc[indices_to_keep, 'LD2'])
                    data2_list[df_name] += [data2]
                if len(indices_to_keep) > 1 and df_name == 'train_data' or target not in unique_cats_train:
                    unique_cats_train = np.unique(np.concatenate((new_labels[df_name], unique_cats_train)))
                    if n_comp > 1:
                        try:
                            confidence_ellipses = confidence_ellipse(np.array(data1), np.array(data2), ax, 1.5,
                                                                     edgecolor=cols[t],
                                                                     train_set=True)
                            ellipses += [confidence_ellipses[1]]
                        except:
                            pass

        for df_name, marker in zip(list(data1_list.keys()), ['o', 'x', '*']):
            data1_vector = np.hstack([d for d in data1_list[df_name] if len(d) > 0]).reshape(-1, 1)
            colors_vector = np.hstack([d for d in colors_list[df_name] if d.shape[1] > 0]).squeeze()
            if n_comp > 1:
                data2_vector = np.hstack(data2_list[df_name]).reshape(-1, 1)
                data_colors_vector = np.concatenate((data1_vector, data2_vector, colors_vector), axis=1)
                data1 = data_colors_vector[:, 0]
                data2 = data_colors_vector[:, 1]
                col = data_colors_vector[:, 2:]
                ax.scatter(data1, data2, s=50, alpha=1.0, c=col, label=new_labels[df_name], marker=marker)
            else:
                data_colors_vector = np.concatenate((data1_vector, colors_vector), axis=1)
                data1 = data_colors_vector[:, 0]
                col = data_colors_vector[:, 1:]
                ax.scatter(data1, np.random.random(len(data1)), s=50, alpha=1.0, c=col, label=new_labels[df_name],
                           marker=marker)

            custom_lines = [Line2D([0], [0], color=cmap(x), lw=4) for x in np.linspace(0, 1, len(uniques[name]) + 1)]
            ax.legend(custom_lines, uniques[name].tolist())

        if mlops == "tensorboard":
            logger.add_figure(f'{ordin["name"]}_{name}', fig, epoch)
        elif mlops == "neptune":
            logger[f'{ordin["name"]}/{name}'].upload(fig)
        elif mlops == "mlflow":
            mlflow.log_figure(fig, f'{ordin["name"]}/{name}.png')
        plt.close(fig)
        plt.close()


def log_CCA(ordin, logger, data, uniques, mlops, epoch):
    for name in list(uniques.keys()):
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111)

        model = ordin['model']

        train_cats = OneHotEncoder().fit_transform(
            np.stack([np.argwhere(uniques[name] == x) for x in data[name]['train']]).reshape(-1, 1)
        ).toarray()

        pcs_train, _ = model.fit_transform(data['inputs']['train'], train_cats)
        pcs_valid = model.transform(data['inputs']['valid'])
        pcs_test = model.transform(data['inputs']['test'])

        pcs_train_df = pd.DataFrame(data=pcs_train, columns=['PC 1', 'PC 2'])
        pcs_valid_df = pd.DataFrame(data=pcs_valid, columns=['PC 1', 'PC 2'])
        pcs_test_df = pd.DataFrame(data=pcs_test, columns=['PC 1', 'PC 2'])
        try:
            ev = model.explained_variance_ratio_
            pc1 = 'Component_1 : ' + str(np.round(ev[0] * 100, decimals=2)) + "%"
            pc2 = 'Component_2 : ' + str(np.round(ev[1] * 100, decimals=2)) + "%"
        except:
            pc1 = 'Component_1'
            pc2 = 'Component_2'

        ax.set_xlabel(pc1, fontsize=15)
        ax.set_ylabel(pc2, fontsize=15)
        ax.set_title(f"2 component {ordin['name']}", fontsize=20)

        num_targets = len(uniques[name])
        cmap = plt.cm.tab20

        cols = cmap(np.linspace(0, 1, len(uniques[name]) + 1))
        colors = rcParams['axes.prop_cycle'] = cycler(color=cols)
        colors_list = {name: [] for name in ['train_data', 'valid_data', 'test_data']}
        data1_list = {name: [] for name in ['train_data', 'valid_data', 'test_data']}
        data2_list = {name: [] for name in ['train_data', 'valid_data', 'test_data']}
        new_labels = {name: [] for name in ['train_data', 'valid_data', 'test_data']}
        new_cats = {name: [] for name in ['train_data', 'valid_data', 'test_data']}

        ellipses = []
        unique_cats_train = np.array([])
        for df_name, df, labels in zip(['train_data', 'valid_data', 'test_data'],
                                       [pcs_train_df, pcs_valid_df, pcs_test_df],
                                       [data[name]['train'], data[name]['valid'], data[name]['test']]):
            for t, target in enumerate(uniques[name]):
                indices_to_keep = [True if x == target else False for x in
                                   list(labels)]  # 0 is the name of the column with target values
                data1 = list(df.loc[indices_to_keep, 'PC 1'])
                new_labels[df_name] += [target for _ in range(len(data1))]
                new_cats[df_name] += [target for _ in range(len(data1))]

                data2 = list(df.loc[indices_to_keep, 'PC 2'])
                data1_list[df_name] += [data1]
                data2_list[df_name] += [data2]
                colors_list[df_name] += [np.array([[cols[t]] * len(data1)])]
                if len(indices_to_keep) > 1 and df_name == 'train_data' or target not in unique_cats_train:
                    unique_cats_train = np.unique(np.concatenate((new_labels[df_name], unique_cats_train)))
                    try:
                        confidence_ellipses = confidence_ellipse(np.array(data1), np.array(data2), ax, 1.5,
                                                                 edgecolor=cols[t],
                                                                 train_set=True)
                        ellipses += [confidence_ellipses[1]]
                    except:
                        pass

        for df_name, marker in zip(list(data1_list.keys()), ['o', 'x', '*']):
            data1_vector = np.hstack([d for d in data1_list[df_name] if len(d) > 0]).reshape(-1, 1)
            colors_vector = np.hstack([d for d in colors_list[df_name] if d.shape[1] > 0]).squeeze()
            data2_vector = np.hstack(data2_list[df_name]).reshape(-1, 1)
            data_colors_vector = np.concatenate((data1_vector, data2_vector, colors_vector), axis=1)
            data2 = data_colors_vector[:, 1]
            col = data_colors_vector[:, 2:]
            data1 = data_colors_vector[:, 0]

            ax.scatter(data1, data2, s=50, alpha=1.0, c=col, label=new_labels[df_name], marker=marker)
            custom_lines = [Line2D([0], [0], color=cmap(x), lw=4) for x in np.linspace(0, 1, len(uniques[name]) + 1)]
            ax.legend(custom_lines, uniques[name].tolist())

        if mlops == "tensorboard":
            logger.add_figure(f'{ordin["name"]}_{name}', fig, epoch)
        elif mlops == "neptune":
            logger[f'{ordin["name"]}/{name}'].upload(fig)
        elif mlops == "mlflow":
            mlflow.log_figure(fig, f'{ordin["name"]}/{name}.png')
        plt.close(fig)


def log_metrics(logger, lists, values, model, unique_labels, unique_batches, epoch, mlops, metrics, n_meta_emb=0, device='cuda'):
    from bernn.dl.models.pytorch.utils.metrics import batch_f1_score
    
    if len(unique_labels) > 2:
        bout = 0
    else:
        bout = 1
    try:
        values = get_metrics(lists, values, model)
    except:
        print("\n\n\nProblem with logging metrics\n\n\n")
    # print('pool metrics logged')
    try:
        for repres in ['enc', 'rec', 'inputs']:
            for metric in ['silhouette', 'lisi']:  # 'kbet', 
                for info in ['labels', 'domains']:
                    for group in ['train', 'valid', 'test']:
                        if metric == 'lisi':
                            if mlops == "tensorboard":
                                try:
                                    logger.add_scalar(f'{metric}/{group}/{repres}/{info}',
                                                      values[group][metric][repres][info][0][0], epoch)
                                except:
                                    logger.add_scalar(f'{metric}/{group}/{repres}/{info}',
                                                      values[group][metric][repres][info][0], epoch)
                            elif mlops == "neptune":
                                try:
                                    logger[f'{metric}/{group}/{repres}/{info}'].log(
                                        values[group][metric][repres][info][0])
                                except:
                                    logger[f'{metric}/{group}/{repres}/{info}'].log(
                                        values[group][metric][repres][info][0][0])
                            elif mlops == "mlflow":
                                try:
                                    mlflow.log_metric(
                                        f'{metric}/{group}/{repres}/{info}',
                                        values[group][metric][repres][info][0], epoch
                                    )
                                except:
                                    mlflow.log_metric(f'{metric}/{group}/{repres}/{info}',
                                        values[group][metric][repres][info][0][0], epoch)

                        elif metric == 'silhouette':
                            if mlops == "tensorboard":
                                try:
                                    logger.add_scalar(f'{metric}/{group}/{repres}/{info}',
                                                      values[group][metric][repres][info][0], epoch)
                                except:
                                    pass
                            elif mlops == "neptune":
                                try:
                                    logger[f'{metric}/{group}/{repres}/{info}'].log(
                                        values[group][metric][repres][info][0])
                                except:
                                    pass
                            elif mlops == "mlflow":
                                try:
                                    mlflow.log_metric(f'{metric}/{group}/{repres}/{info}',
                                        values[group][metric][repres][info][0], epoch)
                                except:
                                    pass

                        elif metric == 'kbet':
                            if mlops == "tensorboard":
                                try:
                                    logger.add_scalar(f'{metric}/{group}/{repres}/{info}',
                                                      values[group][metric][repres][info][0], epoch)
                                except:
                                    pass
                            elif mlops == "neptune":
                                try:
                                    logger[f'{metric}/{group}/{repres}/{info}'].log(
                                        values[group][metric][repres][info][0])
                                except:
                                    pass
                            elif mlops == "mlflow":
                                try:
                                    mlflow.log_metric(f'{metric}/{group}/{repres}/{info}',
                                        values[group][metric][repres][info][0], epoch)
                                except:
                                    pass

                for group in ['train', 'valid', 'test']:
                    if metric == 'lisi':
                        try:
                            values[group][f"F1_{metric}"][repres]['domains'] = batch_f1_score(
                                batch_score=values[group][metric][repres]['domains'][0][0] / len(unique_batches),
                                class_score=values[group][metric][repres]['labels'][0][0] / len(unique_labels),
                            )
                        except:
                            values[group][f"F1_{metric}"][repres]['domains'] = batch_f1_score(
                                batch_score=values[group][metric][repres]['domains'][0] / len(unique_batches),
                                class_score=values[group][metric][repres]['labels'][0] / len(unique_labels),
                            )

                    elif metric == 'silhouette':
                        values[group][f"F1_{metric}"][repres]['domains'] = batch_f1_score(
                            batch_score=(values[group][metric][repres]['domains'][0] + 1) / 2,
                            class_score=(values[group][metric][repres]['labels'][0] + 1) / 2,
                        )
                    elif metric == 'kbet':
                        try:
                            values[group][f"F1_{metric}"][repres]['domains'] = batch_f1_score(
                                batch_score=values[group][metric][repres]['domains'][0] / len(unique_batches),
                                class_score=values[group][metric][repres]['labels'][0] / len(unique_labels),
                            )
                        except:
                            pass
                    if mlops == "tensorboard":
                        logger.add_scalar(f'F1_{metric}/{group}/{repres}/domains',
                                          values[group][f"F1_{metric}"][repres]['domains'],
                                          epoch)
                    elif mlops == "neptune":
                        logger[f'F1_{metric}/{group}/{repres}/domains'].log(
                            values[group][f"F1_{metric}"][repres]['domains'],
                        )
                    elif mlops == "mlflow":
                        mlflow.log_metric(f'F1_{metric}/{group}/{repres}/domains',
                            values[group][f"F1_{metric}"][repres]['domains'], epoch
                        )

    except:
        return metrics

    try:
        for repres in ['enc', 'rec', 'inputs']:
            for metric in ['adjusted_rand_score', 'adjusted_mutual_info_score', 'batch_entropy']:
                for group in ['all', 'train', 'valid', 'test']:  # , 'set'
                    for info in ['labels', 'domains']:
                        if metric == 'batch_entropy' and info == 'labels':
                            continue
                        if mlops == "tensorboard":
                            logger.add_scalar(f'{metric}/{group}/{repres}/{info}',
                                              values[group][metric][repres][info][0],
                                              epoch)
                        elif mlops == "neptune":
                            logger[f'{metric}/{group}/{repres}/{info}'].log(
                                values[group][metric][repres][info][0])
                        elif mlops == "mlflow":
                            mlflow.log_metric(f'{metric}/{group}/{repres}/{info}',
                                values[group][metric][repres][info][0], epoch)

                    if metric != 'batch_entropy':
                        values[group][f"F1_{metric}"][repres]['domains'] = batch_f1_score(
                            batch_score=values[group][metric][repres]['domains'][0],
                            class_score=values[group][metric][repres]['labels'][0],
                        )
                        if mlops == "tensorboard":
                            logger.add_scalar(f'F1_{metric}/{group}/{repres}/domains',
                                              values[group][f"F1_{metric}"][repres]['domains'],
                                              epoch)
                        elif mlops == "neptune":
                            logger[f'F1_{metric}/{group}/{repres}/domains'].log(
                                values[group][f"F1_{metric}"][repres]['domains'])
                        elif mlops == "mlflow":
                            mlflow.log_metric(f'F1_{metric}/{group}/{repres}/domains',
                                values[group][f"F1_{metric}"][repres]['domains'], epoch)
    except:
        print("\n\n\nProblem with Adjusted Rand Index or Adjusted Mutual Info\n\n\n")

    # print('batch mixing metrics logged')
    metrics['batches'] = values
    try:
        clusters = ['labels', 'domains']
        reps = ['enc', 'rec']
        train_lisi_enc = [values['train']['lisi']['enc'][c][0].reshape(-1) for c in clusters]
        train_lisi_rec = [values['train']['lisi']['rec'][c][0].reshape(-1) for c in clusters]
        valid_lisi_enc = [values['valid']['lisi']['enc'][c][0].reshape(-1) for c in clusters]
        valid_lisi_rec = [values['valid']['lisi']['rec'][c][0].reshape(-1) for c in clusters]
        test_lisi_enc = [values['test']['lisi']['enc'][c][0].reshape(-1) for c in clusters]
        test_lisi_rec = [values['test']['lisi']['rec'][c][0].reshape(-1) for c in clusters]

        # set_lisi_enc = [values['set']['lisi']['enc'][c][0].reshape(-1) for c in ['labels', 'set']]
        # set_lisi_rec = [values['set']['lisi']['rec'][c][0].reshape(-1) for c in ['labels', 'set']]
    except:
        print("\n\n\nProblem with preparing LISI\n\n\n")
    try:
        lisi_train_reps = np.concatenate(
            [np.array([r for _ in np.concatenate(values)]) for r, values in zip(reps, [train_lisi_enc, train_lisi_rec])]
        )
        lisi_valid_reps = np.concatenate(
            [np.array([r for _ in np.concatenate(values)]) for r, values in zip(reps, [valid_lisi_enc, valid_lisi_rec])]
        )
        lisi_test_reps = np.concatenate(
            [np.array([r for _ in np.concatenate(values)]) for r, values in zip(reps, [test_lisi_enc, test_lisi_rec])]
        )
    except:
        print("\n\n\nProblem with concat LISI\n\n\n")
    try:
        lisi_df_train = pd.DataFrame(np.concatenate((
            np.concatenate((np.concatenate(train_lisi_enc), np.concatenate(train_lisi_rec))).reshape(1, -1),
            lisi_train_reps.reshape(1, -1),
            # np.concatenate((lisi_set_train_enc, lisi_set_train_rec)).reshape(1, -1),
        ), 0).T, columns=['lisi', 'representation'])
        lisi_df_train['lisi'] = pd.to_numeric(lisi_df_train['lisi'])
        lisi_df_valid = pd.DataFrame(np.concatenate((
            np.concatenate((valid_lisi_enc, valid_lisi_rec)).reshape(1, -1),
            lisi_valid_reps.reshape(1, -1),
            # np.concatenate((lisi_set_valid_enc, lisi_set_valid_rec)).reshape(1, -1),
        ), 0).T, columns=['lisi', 'representation'])
        lisi_df_valid['lisi'] = pd.to_numeric(lisi_df_valid['lisi'])
        lisi_df_test = pd.DataFrame(np.concatenate((
            np.concatenate((test_lisi_enc, test_lisi_rec)).reshape(1, -1),
            lisi_test_reps.reshape(1, -1),
            # np.concatenate((lisi_set_test_enc, lisi_set_test_rec)).reshape(1, -1),
        ), 0).T, columns=['lisi', 'representation'])
        lisi_df_test['lisi'] = pd.to_numeric(lisi_df_test['lisi'])
        # lisi_means = [values[s]['lisi'][0] for s in ['train', 'valid', 'test']]
    except:
        print("\n\n\nProblem with plotting LISI\n\n\n")

    try:
        sns.set_theme(style="whitegrid")

        if lisi_df_train.shape[0] > 0:
            figure = plt.figure(figsize=(8, 8))
            ax = sns.boxplot(x="representation", y="lisi", data=lisi_df_train)
            # logger.add_figure(f"LISI_train", figure, epoch)
            if mlops == "tensorboard":
                logger.add_figure(f"LISI_train", figure, epoch)
            elif mlops == "neptune":
                logger[f'LISI_train'].log(figure)
            elif mlops == "mlflow":
                mlflow.log_figure(figure, f'LISI_train.png')
            plt.close(figure)

        else:
            print('LISI_train Problem')

        if lisi_df_valid.shape[0] > 0:
            figure = plt.figure(figsize=(8, 8))
            ax = sns.boxplot(x="representation", y="lisi", data=lisi_df_valid)
            if mlops == "tensorboard":
                logger.add_figure(f"LISI_valid", figure, epoch)
            elif mlops == "neptune":
                logger[f'LISI_valid'].log(figure)
            elif mlops == "mlflow":
                mlflow.log_figure(figure, f'LISI_valid.png')
            plt.close(figure)

        else:
            print('LISI_valid Problem')
        if lisi_df_test.shape[0] > 0:
            figure = plt.figure(figsize=(8, 8))
            ax = sns.boxplot(x="representation", y="lisi", data=lisi_df_test)
            sns.set_theme(style="white")
            if mlops == "tensorboard":
                logger.add_figure(f"LISI_test", figure, epoch)
            elif mlops == "neptune":
                logger[f'LISI_test'].log(figure)
            elif mlops == "mlflow":
                mlflow.log_figure(figure, f'LISI_test.png')
            plt.close(figure)

        else:
            print('LISI_test Problem')
    except:
        print("\n\n\nProblem with plotting LISI\n\n\n")

    train_enc = torch.Tensor(np.concatenate(lists['train']['encoded_values']))
    valid_enc = torch.Tensor(np.concatenate(lists['valid']['encoded_values']))
    test_enc = torch.Tensor(np.concatenate(lists['test']['encoded_values']))
    if n_meta_emb > 0:
        train_enc = torch.cat((train_enc,
                               torch.Tensor(np.concatenate(lists['train']['age'])).view(-1, 1),
                               torch.Tensor(np.concatenate(lists['train']['gender'])).view(-1, 1),
                               ), 1)
        valid_enc = torch.cat((valid_enc,
                               torch.Tensor(np.concatenate(lists['valid']['age'])).view(-1, 1),
                               torch.Tensor(np.concatenate(lists['valid']['gender'])).view(-1, 1),
                               ), 1)
        test_enc = torch.cat((test_enc,
                              torch.Tensor(np.concatenate(lists['test']['age'])).view(-1, 1),
                              torch.Tensor(np.concatenate(lists['test']['gender'])).view(-1, 1),
                              ), 1)
    try:
        save_roc_curve(model,
                       train_enc.to(device),
                       np.concatenate(lists['train']['classes']),
                       unique_labels, name='./roc_train', binary=bout, epoch=epoch,
                       acc=values['train']['acc'][-1], logger=logger, mlops=mlops)
        save_roc_curve(model,
                       valid_enc.to(device),
                       np.concatenate(lists['valid']['classes']),
                       unique_labels, name='./roc_valid', binary=bout, epoch=epoch,
                       acc=values['valid']['acc'][-1], logger=logger, mlops=mlops)
        save_roc_curve(model,
                       test_enc.to(device),
                       np.concatenate(lists['test']['classes']),
                       unique_labels, name='./roc_test', binary=bout, epoch=epoch,
                       acc=values['test']['acc'][-1], logger=logger, mlops=mlops)
    except:
        print("\n\n\nProblem with ROC curves\n\n\n")

    try:
        save_precision_recall_curve(model,
                                    train_enc.to(device),
                                    np.concatenate(lists['train']['classes']),
                                    unique_labels, name='./prc_train', binary=bout, epoch=epoch,
                                    acc=values['train']['acc'][-1], logger=logger, mlops=mlops)
        save_precision_recall_curve(model,
                                    valid_enc.to(device),
                                    np.concatenate(lists['valid']['classes']),
                                    unique_labels, name='./prc_valid', binary=bout, epoch=epoch,
                                    acc=values['valid']['acc'][-1], logger=logger, mlops=mlops)
        save_precision_recall_curve(model,
                                    test_enc.to(device),
                                    np.concatenate(lists['test']['classes']),
                                    unique_labels, name='./prc_test', binary=bout, epoch=epoch,
                                    acc=values['test']['acc'][-1], logger=logger, mlops=mlops)
    except:
        print("\n\n\nProblem with precision/recall curves\n\n\n")

    return metrics


def make_data(lists, values):
    n_mini_batches = int(1000/lists['train'][values][0].shape[0])
    try:
        data = {
            'inputs': {group: np.concatenate(lists[group][values][:n_mini_batches]) for group in list(lists.keys()) if
                       len(lists[group][values]) > 0},
            'labels': {group: np.concatenate(lists[group]['labels'][:n_mini_batches]) for group in list(lists.keys()) if
                       len(lists[group][values]) > 0},
            'batches': {group: np.concatenate(lists[group]['domains'][:n_mini_batches]) for group in list(lists.keys()) if
                        len(lists[group][values]) > 0},
            # 'age': {group: np.concatenate(lists[group]['age']) for group in list(lists.keys()) if
            #         len(lists[group][values]) > 0},
            # 'gender': {group: np.concatenate(lists[group]['gender']) for group in list(lists.keys()) if
            #            len(lists[group][values]) > 0},
            # 'atn': {group: np.array(lists[group]['atn']) for group in list(lists.keys()) if
            #         len(lists[group][values]) > 0},
        }
        keys = list(data['inputs'].keys())
    except:
        data = {
            'inputs': {group: np.concatenate(lists[group][values][:n_mini_batches]) for group in ['all', 'all_pool']},
            'labels': {group: np.concatenate(lists[group]['labels'][:n_mini_batches]) for group in ['all', 'all_pool']},
            'batches': {group: np.concatenate(lists[group]['domains'][:n_mini_batches]) for group in ['all', 'all_pool']},
            # 'age': {group: np.concatenate(lists[group]['age']) for group in ['all', 'all_pool']},
            # 'gender': {group: np.concatenate(lists[group]['gender']) for group in ['all', 'all_pool']},
            # 'atn': {group: np.array(lists[group]['atn']) for group in ['all', 'all_pool']},
        }
        keys = ['all', 'all_pool']

    # for group in keys:
    #     meta_age = []
    #     for age in data['age'][group]:
    #         if age < 60:
    #             meta_age += ['50s']
    #         elif age < 70:
    #             meta_age += ['60s']
    #         elif age < 80:
    #             meta_age += ['70s']
    #         else:
    #             meta_age += ['80+']
    #     data['age'][group] = np.array(meta_age)

    return data


def log_plots(logger, lists, mlops, epoch):
    try:
        for name, values in zip(['encs', 'inputs', 'recs'], ['encoded_values', 'inputs', 'rec_values']):
            unique_labels = get_unique_labels(np.concatenate(lists['train']['labels']))
            unique_batches = np.unique(
                np.concatenate((
                    np.unique(np.concatenate(lists['train']['domains'])),
                    np.unique(np.concatenate(lists['valid']['domains'])),
                    np.unique(np.concatenate(lists['test']['domains'])),

                ))
            )
            # unique_ages = np.array(['50s', '60s', '70s', '80+'])
            # unique_genders = np.unique(np.concatenate(lists['all']['gender']))
            # unique_atns = np.unique([str(x) for x in np.array(lists['all']['atn'])])
            if len(lists['test'][values]) == 0:
                continue
            uniques = {'batches': unique_batches, 'labels': unique_labels}
            try:
                log_CCA({'model': CCA(n_components=2), 'name': f'CCA_{name}'},
                        logger, make_data(lists, values), uniques, mlops, epoch)
            except:
                pass

            try:
                log_LDA({'model': LDA, 'name': f'LDA_{name}'},
                        logger, make_data(lists, values), uniques, mlops, epoch)
            except:
                pass
            log_ORD({'model': PCA(n_components=2), 'name': f'PCA_{name}'}, logger,
                    make_data(lists, values), uniques, mlops, epoch),
            log_ORD({'model': PCA(n_components=2), 'name': f'PCA_{name}'}, logger,
                    make_data(lists, values), uniques, mlops, epoch, transductive=True),
            log_ORD({'model': UMAP(n_neighbors=5, min_dist=0.3, metric='correlation'), 'name': f'UMAP_{name}'},
                    logger, make_data(lists, values), uniques, mlops, epoch)
            log_ORD({'model': UMAP(n_neighbors=5, min_dist=0.3, metric='correlation'), 'name': f'UMAP_{name}'},
                    logger, make_data(lists, values), uniques, mlops, epoch, transductive=True)
    except:
        print("\n\n\nProblem with logging PCA, TSNE or UMAP\n\n\n")


def log_input_ordination(logger, data, scaler, mlops, epoch=0):
    try:
        data['gender'] = {}
        data['age'] = {}
        data['atn'] = {}
        for group in ['train', 'valid', 'test']:
            tmp = scaler.inverse_transform(data['inputs'][group])
            data['gender'][group] = tmp[:, -1].astype(np.int)
            data['age'][group] = tmp[:, -2]
            data['atn'][group] = tmp[:, -5:-2]
            data['atn'][group] = np.array([str(x) for x in data['atn'][group]])
            meta_age = []
            for age in data['age'][group]:
                if age < 60:
                    meta_age += ['50s']
                elif age < 70:
                    meta_age += ['60s']
                elif age < 80:
                    meta_age += ['70s']
                else:
                    meta_age += ['80+']
            data['age'][group] = np.array(meta_age)

        unique_labels = get_unique_labels(data['labels']['train'])
        unique_batches = np.unique(np.concatenate([
            data['batches']['train'],
            data['batches']['valid'],
            data['batches']['test']
        ]))
        unique_ages = np.array(['50s', '60s', '70s', '80+'])
        unique_genders = np.unique(data['gender']['train'])
        unique_atns = np.unique([str(x) for x in data['atn']['train']])
        uniques = {'batches': unique_batches, 'labels': unique_labels, 'age': unique_ages,
                   'gender': unique_genders, 'atn': unique_atns}
        try:
            log_CCA({'model': CCA(n_components=2), 'name': f'CCA_inputs'},
                    logger, data, uniques, mlops, epoch)
        except:
            pass
        try:
            log_LDA({'model': LDA, 'name': f'LDA_inputs'},
                    logger, data, uniques, mlops, epoch)
        except:
            pass
        log_ORD({'model': PCA(n_components=2), 'name': f'PCA_inputs'}, logger,
                data, uniques, mlops, epoch),
        log_ORD({'model': PCA(n_components=2), 'name': f'PCA_inputs'}, logger,
                data, uniques, mlops, epoch),
        log_ORD({'model': PCA(n_components=2), 'name': f'PCA_inputs'}, logger,
                data, uniques, mlops, epoch, transductive=True),
        log_ORD({'model': UMAP(n_neighbors=5, min_dist=0.3, metric='correlation'), 'name': f'UMAP_inputs'},
                logger, data, mlops, uniques, epoch)
        log_ORD({'model': UMAP(n_neighbors=5, min_dist=0.3, metric='correlation'), 'name': f'UMAP_inputs'},
                logger, data, mlops, uniques, epoch, transductive=True)

    except:
        print("\n\n\nProblem with logging PCA, TSNE or UMAP\n\n\n")
