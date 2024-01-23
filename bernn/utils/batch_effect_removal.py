import pandas as pd
import numpy as np
import json
import os
import torch
import rpy2.robjects as robjects
import rpy2.robjects as ro
import rpy2.robjects.numpy2ri
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
from rpy2.robjects.conversion import localconverter
from rpy2.rinterface import NULL
from time import perf_counter
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
# from src.dl.models.pytorch.normae_no_order import BatchEffectTrainer

minmax_scaler = MinMaxScaler()
standard_scaler = StandardScaler()
robust_scaler = RobustScaler()

base = importr('base')
stats = importr('stats')


def comBatR(data, batches, classes=None, orders=None, par_prior=True, ref_batch=NULL):
    df = pd.DataFrame(data.values.T)
    data_r = robjects.r.matrix(robjects.FloatVector(df.values.reshape(-1)), nrow=df.shape[0])
    batches_r = robjects.IntVector(batches.reshape(-1))
    if classes is not None:
        classes_r = robjects.IntVector(classes.reshape(-1))
    sva = importr('sva')

    if classes is None:
        print("not with classes!")
        newdata = sva.ComBat(data_r, batches_r)
    else:
        print("with classes!")
        ro.numpy2ri.activate()
        R = ro.r
        R.assign('subject', classes_r)
        R('subject <- as.factor(subject)')
        design = R('model.matrix(~subject)')
        newdata = sva.ComBat(data_r, batches_r, design)

    with localconverter(robjects.default_converter + pandas2ri.converter):
        newdata = np.array(robjects.conversion.rpy2py(newdata))
    return newdata.T


def harmonyR(data, batches, orders=None, classes=NULL, par_prior=True, ref_batch=NULL):
    df = pd.DataFrame(data.values)
    data_r = robjects.r.matrix(robjects.FloatVector(df.values.reshape(-1)), nrow=df.shape[0])
    batches_r = robjects.IntVector(batches.reshape(-1))
    harmony = importr('harmony')
    newdata = harmony.HarmonyMatrix(data_r, batches_r, do_pca=True)
    with localconverter(robjects.default_converter + pandas2ri.converter):
        newdata = np.array(robjects.conversion.rpy2py(newdata))
    return newdata


def waveicaR(data, batches, orders=None, classes=NULL, par_prior=True, ref_batch=NULL):
    with localconverter(ro.default_converter + pandas2ri.converter):
        data_r = ro.conversion.py2rpy(data)

    # data_r = robjects.r.matrix(robjects.FloatVector(df.values.reshape(-1)), nrow=df.shape[0])
    batches_r = robjects.IntVector(batches.reshape(-1))
    waveica = importr('WaveICA')
    # data_r.colnames = robjects.StrVector([str(x) for x in range(df.shape[1])])
    newdata = waveica.WaveICA(dat=data_r, batch=batches_r)
    with localconverter(robjects.default_converter + pandas2ri.converter):
        newdata = np.array(robjects.conversion.rpy2py(newdata))
    return newdata


def seuratR(data, batches, orders=None, classes=NULL, par_prior=True, ref_batch=NULL):
    df = pd.DataFrame(data.values)
    data_r = robjects.r.matrix(robjects.FloatVector(df.values.reshape(-1)), nrow=df.shape[0])
    batches_r = robjects.IntVector(batches.reshape(-1))
    waveica = importr('WaveICA')
    data_r.colnames = robjects.StrVector([str(x) for x in range(df.shape[1])])
    newdata = waveica.WaveICA(dat=data_r, batch=batches_r)
    with localconverter(robjects.default_converter + pandas2ri.converter):
        newdata = np.array(robjects.conversion.rpy2py(newdata))
    return newdata


def qcrlscR(data, batches, orders=None, classes=NULL, par_prior=True, ref_batch=NULL):
    """
    Batches need to be qc or not qc
    Args:
        data:
        batches:
        orders:
        classes:
        par_prior:
        ref_batch:

    Returns:

    """
    df = pd.DataFrame(data.values)
    data_r = robjects.r.matrix(robjects.FloatVector(df.values.reshape(-1)), nrow=df.shape[0])
    batches_r = robjects.IntVector(batches.reshape(-1))
    rcpm = importr('Rcpm')
    if orders is None:
        orders = robjects.IntVector(list(range(batches.reshape(-1).shape[0])))
    newdata = rcpm.qc_rlsc(data_r, batches_r, orders)
    with localconverter(robjects.default_converter + pandas2ri.converter):
        newdata = np.array(robjects.conversion.rpy2py(newdata))
    return newdata


def fasticaR(data, batches, orders=None, classes=NULL, par_prior=True, ref_batch=NULL):
    """
    Batches need to be qc or not qc
    Args:
        data:
        batches:
        orders:
        classes:
        par_prior:
        ref_batch:

    Returns:

    """
    df = pd.DataFrame(data.values)
    data_r = robjects.r.matrix(robjects.FloatVector(df.values.reshape(-1)), nrow=df.shape[0])
    ica = importr('ica')
    newdata = ica.icafast(data_r, df.shape[0])
    with localconverter(robjects.default_converter + pandas2ri.converter):
        newdata = np.array(robjects.conversion.rpy2py(newdata))
    return newdata


def zinbWaveR(data, batches, orders=None, classes=NULL, par_prior=True, ref_batch=NULL):
    df = pd.DataFrame(data.values)
    data_r = robjects.r.matrix(robjects.FloatVector(df.values.reshape(-1)), nrow=df.shape[0])
    batches_r = robjects.IntVector(batches.reshape(-1))
    zinbwave = importr('zinbwave')
    granges = importr('GenomicRanges')
    sumexp = importr('SummarizedExperiment')
    iranges = importr('IRanges')
    s4vectors = importr('S4Vectors')

    nrows = 200
    ncols = 6
    counts = robjects.r.matrix(stats.runif(nrows * ncols, 1, 1e4), nrows)
    rowRanges = granges.GRanges(base.rep(base.c("chr1", "chr2"), base.c(50, 150)),
                                iranges.IRanges(base.floor(stats.runif(200, 1e5, 1e6)), width=100),
                                strand=base.sample(base.c("+", "-"), 200, True),
                                )
    colData = s4vectors.DataFrame(Treatment=base.rep(base.c("ChIP", "Input"), 3), row_names=base.LETTERS[0:6])

    exp = sumexp.SummarizedExperiment(assays=base.list(counts=counts),
                                      rowRanges=rowRanges, colData=colData)

    data_assay_r = sumexp.SummarizedExperiment(data_r)
    newdata = zinbwave.zinbwave(data_assay_r, K=2, epsilon=1000)
    with localconverter(robjects.default_converter + pandas2ri.converter):
        newdata = np.array(robjects.conversion.rpy2py(newdata))
    return newdata


def ligerR(data, batches, orders=None, classes=NULL, par_prior=True, ref_batch=NULL):
    df = pd.DataFrame(data.values)
    df = df.iloc[:1000]
    data_r = robjects.r.array(robjects.r.matrix(robjects.FloatVector(df.values.reshape(-1)), nrow=df.shape[0]))
    batches_r = robjects.IntVector(batches.reshape(-1))
    rliger = importr('rliger')
    newdata = rliger.normalize(data_r, batches_r)
    with localconverter(robjects.default_converter + pandas2ri.converter):
        newdata = np.array(robjects.conversion.rpy2py(newdata))
    return newdata


def scMergeR(data, batches, orders=None, classes=NULL, par_prior=True, ref_batch=NULL):
    df = pd.DataFrame(data.values)

    data_r = robjects.r.matrix(robjects.FloatVector(df.values.reshape(-1)), nrow=df.shape[0])
    batches_r = robjects.IntVector(batches.reshape(-1))
    scMerge = importr('scMerge')
    newdata = scMerge.scMerge(data_r, batches_r)
    with localconverter(robjects.default_converter + pandas2ri.converter):
        newdata = np.array(robjects.conversion.rpy2py(newdata))
    return newdata


def remove_batch_effect(berm, data, all_batches):
    """
    All dataframes have a shape of N samples (rows) x M features (columns)

    Args:
        berm: Batch effect removal method
        all_data: Pandas dataframe containing all data (train, valid and test data)
        data: list of Pandas dataframe containing the training data. Used only to get
        valid_data: Pandas dataframe containing the validation data
        test_data: Pandas dataframe containing the test data
        all_batches: A list containing the batch ids corresponding to all_data

    Returns:
        Returns:
        A dictionary of pandas datasets corrected for batch effect with keys:
            'all': Pandas dataframe containing all data (train, valid and test data)
            'train': Pandas dataframe containing the training data
            'valid': Pandas dataframe containing the validation data
            'test: Pandas dataframe containing the test data

    """
    if berm is not None:  # Look if has 20 before and after
        df = pd.concat((data['inputs']['train'].copy(), data['inputs']['valid'].copy(), data['inputs']['test'].copy()))
        # assert np.sum(all_batches != np.concatenate((data['batches']['train'], data['batches']['valid'], data['batches']['test']))) == 0
        tmp = berm(df, all_batches)
        previous_len = 0
        for g in list(data['inputs'].keys())[1:]:
            data['inputs'][g] = pd.DataFrame(
                tmp[previous_len:previous_len + data['inputs'][g].shape[0]],
                index=data['inputs'][g].index)
            previous_len += data['inputs'][g].shape[0]
        try:
            data['inputs']['all'] = pd.DataFrame(tmp, index=df.index, columns=df.columns)
        except:
            data['inputs']['all'] = pd.DataFrame(tmp, index=df.index)

    return data


def remove_batch_effect2(berm, all_data, train_data, valid_data, test_data, train_pool_data, valid_pool_data,
                         test_pool_data, all_batches, orders=None):
    """
    All dataframes have a shape of N samples (rows) x M features (columns)

    Args:
        berm: Batch effect removal method
        all_data:
        train_data:
        valid_data:
        test_data:
        all_batches:
        orders:

    Returns:
        Returns:
        A dictionary of pandas datasets with keys:
            'all': Pandas dataframe containing all data (train, valid and test data),
            'train': Pandas dataframe containing the training data,
            'valid': Pandas dataframe containing the validation data,
            'test: Pandas dataframe containing the test data'

    """
    if berm is not None:
        df = pd.DataFrame(all_data)
        # df[df.isna()] = 0
        all_data = berm(df, all_batches, orders)
        all_data = pd.DataFrame(all_data, index=df.index, columns=df.columns)
        train_data = all_data.iloc[:train_data.shape[0]]
        valid_data = all_data.iloc[train_data.shape[0]:train_data.shape[0] + valid_data.shape[0]]
        test_data = all_data.iloc[train_data.shape[0] + valid_data.shape[0]:train_data.shape[0] + valid_data.shape[0] +
                                                                            test_data.shape[0]]
        train_pool_data = all_data.iloc[
                          train_data.shape[0] + valid_data.shape[0] + test_data.shape[0]:train_data.shape[0] +
                                                                                         valid_data.shape[0] +
                                                                                         test_data.shape[0] +
                                                                                         train_pool_data.shape[0]]
        valid_pool_data = all_data.iloc[
                          train_data.shape[0] + valid_data.shape[0] + test_data.shape[0] + train_pool_data.shape[0]:
                          train_data.shape[0] + valid_data.shape[0] + test_data.shape[0] + train_pool_data.shape[0] +
                          valid_pool_data.shape[0]]
        test_pool_data = all_data.iloc[
                         train_data.shape[0] + valid_data.shape[0] + test_data.shape[0] + train_pool_data.shape[0] +
                         valid_pool_data.shape[0]:]

    return {'all': all_data, 'train': train_data, 'valid': valid_data, 'test': test_data, 'train_pool': train_pool_data,
            'valid_pool': valid_pool_data, 'test_pool': test_pool_data}


def remove_batch_effect_all(berm, all_data, all_batches):
    """
    All dataframes have a shape of N samples (rows) x M features (columns)

    Args:
        berm: Batch effect removal method
        all_data:
        all_batches:

    Returns:
        Returns:
            Pandas dataframe containing all data (train, valid and test data),

    """
    if berm is not None:
        df = pd.DataFrame(all_data)
        # df[df.isna()] = 0
        all_data = berm(df, all_batches)

    return all_data


def get_berm(berm):
    # berm: batch effect removal method
    if berm == 'combat':
        berm = comBatR
    if berm == 'harmony':
        berm = harmonyR
    if berm == 'waveica':
        berm = waveicaR
    if berm == 'qcrlsc':
        berm = qcrlscR
    if berm == 'ica':
        berm = fasticaR
    if berm == 'none':
        berm = None
    return berm


# TODO Add hparam optimization. Make it work without orders
def normae(data, batches, orders=None, classes=NULL, par_prior=True, ref_batch=NULL):
    def save(self, fname):
        self.save_dict = self.args.__dict__
        with open(fname, 'w') as f:
            json.dump(self.save_dict, f)

    import argparse

    opts = argparse.ArgumentParser()
    opts.task = 'train'
    opts.save = './'
    opts.device = 'cuda'
    opts.ae_encoder_units = [1000, 1000]
    opts.ae_decoder_units = [1000, 1000]
    opts.disc_b_units = [250, 250]
    opts.disc_o_units = [250, 250]
    # opts.meta_data = None
    # opts.sample_data = None
    # opts.use_log = None
    # opts.use_batch = None
    # opts.sample_size = None
    # opts.random_seed = None
    opts.load = None
    opts.bottle_num = 500
    opts.dropouts = [0.5, 0.5, 0.5, 0.5]
    opts.use_batch_for_order = 0
    opts.lambda_b = 0.1
    opts.lambda_o = 0.1
    opts.lr_rec = 1e-3
    opts.lr_disc_b = 1e-3
    opts.lr_disc_o = 1e-3
    # opts.epoch = (1000, 100, 700)
    opts.epoch = (500, 10, 300)
    opts.batch_size = 8
    opts.num_workers = 0
    opts.train_data = "all"
    opts.visdom_port = None
    opts.visdom_env = 'main'
    opts.random_pool = False

    subject_dat = minmax_scaler.fit_transform(data.values)
    # qc_dat = minmax_scaler.transform(pool_df.values)
    datas = {'subject': subject_dat, 'qc': None}
    labels = {'subject': data.index, 'qc': None}  # or columns
    batches = {'subject': batches, 'qc': None}
    orders = {'subject': orders, 'qc': None}

    # build estimator
    if opts.device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cuda:0" if opts.device == "GPU" else "cpu")
    norm_ae = BatchEffectTrainer(
        subject_dat.shape[1], 9,
        device, standard_scaler, opts)
    # load models
    if opts.load is not None:
        model_file = os.path.join(opts.load, "models.pth") \
            if os.path.isdir(opts.load) else opts.load
        norm_ae.load_model(model_file)
    if opts.task == "train":
        # ----- training -----
        fit_time1 = perf_counter()
        best_models, hist, early_stop_objs = norm_ae.fit(datas, labels, batches, orders)
        fit_time2 = perf_counter()
        early_stop_objs["fit_duration"] = fit_time2 - fit_time1
        # ----- save models and results -----
        # if os.path.exists(opts.save):
        #     dirname = input("%s has been already exists, please input New: " %
        #                     config.args.save)
        #     os.makedirs(dirname)
        # else:
        #     os.makedirs(opts.save)
        torch.save(best_models, os.path.join(opts.save, 'models.pth'))
        # pd.DataFrame(hist).to_csv(os.path.join(opts.save, 'train.csv'))
        # save(os.path.join(opts.save, 'config.json'))
        # with open(os.path.join(opts.save, 'early_stop_info.json'), 'w') as f:
        #     json.dump(early_stop_objs, f)


def rLISI(data, meta_data, perplexity=10):
    import rpy2.robjects as robjects
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.packages import importr
    from rpy2.robjects.conversion import localconverter
    lisi = importr('lisi')
    # all_batches_r = robjects.IntVector(all_batches[all_ranks])
    # all_data_r.colnames = robjects.StrVector([str(x) for x in range(df.shape[1])])
    # labels = ['label1', 'label2']
    labels = robjects.StrVector(['label1'])
    new_meta_data = robjects.r.matrix(robjects.IntVector(meta_data), nrow=data.shape[0])
    newdata = robjects.r.matrix(robjects.FloatVector(data.values.reshape(-1)), nrow=data.shape[0])

    new_meta_data.colnames = labels
    results = lisi.compute_lisi(newdata, new_meta_data, labels, perplexity)
    with localconverter(robjects.default_converter + pandas2ri.converter):
        results = np.array(robjects.conversion.rpy2py(results))
    mean = np.mean(results)
    return mean  # , np.std(results), results


def rKBET(inputs, cats):
    kbet = importr('kBET')
    # all_batches_r = robjects.IntVector(all_batches[all_ranks])
    # all_data_r.colnames = robjects.StrVector([str(x) for x in range(df.shape[1])])
    # labels = ['label1', 'label2']
    labels = robjects.StrVector(['label1'])
    new_meta_data = robjects.IntVector(cats)
    newdata = robjects.r.matrix(robjects.FloatVector(inputs.values.reshape(-1)), nrow=inputs.shape[0])

    new_meta_data.colnames = labels
    results = kbet.kBET(newdata, new_meta_data, do_pca=False, plot=False)
    with localconverter(robjects.default_converter + pandas2ri.converter):
        results = robjects.conversion.rpy2py(results[0])
    try:
        mean = results['kBET.signif'][0]
    except:
        mean = 0

    return mean
