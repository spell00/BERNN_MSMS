# Libraries
import numpy as np
from tensorflow.keras import backend as K

smooth = 1


def jaccard_distance_loss(y_true, y_pred, smooth=100, backend=K):
    y_true_f = backend.flatten(y_true)
    y_pred_f = backend.flatten(y_pred)
    intersection = backend.sum(backend.abs(y_true_f * y_pred_f))
    sum_ = backend.sum(backend.abs(y_true_f) + backend.abs(y_pred_f))
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth


def mean_length_error(y_true, y_pred, backend=K):
    y_true_f = backend.sum(backend.round(backend.flatten(y_true)))
    y_pred_f = backend.sum(backend.round(backend.flatten(y_pred)))
    delta = (y_pred_f - y_true_f)
    return backend.mean(backend.tanh(delta))


def dice_coef(y_true, y_pred, backend=K):
    y_true_f = backend.flatten(y_true)
    y_pred_f = backend.flatten(y_pred)
    intersection = backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (backend.sum(y_true_f) + backend.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred, backend=K):
    return -dice_coef(y_true, y_pred, backend=backend)


def np_dice_coef(y_true, y_pred):
    tr = y_true.flatten()
    pr = y_pred.flatten()
    return (2. * np.sum(tr * pr) + smooth) / (np.sum(tr) + np.sum(pr) + smooth)


def batch_f1_score(batch_score, class_score):
    return 2 * (1 - batch_score) * (class_score) / (1 - batch_score + class_score + 1e-4)


# matthews_correlation
def matthews_correlation(y_true, y_pred):
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos

    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos

    tp = K.sum(y_pos * y_pred_pos)
    tn = K.sum(y_neg * y_pred_neg)

    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)

    numerator = (tp * tn - fp * fn)
    denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    return numerator / (denominator + K.epsilon())

# Compute LISI for: labels, batches, plates
def rLISI(data, meta_data, perplexity=5):
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
    newdata = robjects.r.matrix(robjects.FloatVector(data.reshape(-1)), nrow=data.shape[0])

    new_meta_data.colnames = labels
    results = lisi.compute_lisi(newdata, new_meta_data, labels, perplexity)
    with localconverter(robjects.default_converter + pandas2ri.converter):
        results = np.array(robjects.conversion.rpy2py(results))
    mean = np.mean(results)
    return mean  # , np.std(results), results


# Compute LISI for: labels, batches, plates
def rKBET(data, meta_data):
    import rpy2.robjects as robjects
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.packages import importr
    from rpy2.robjects.conversion import localconverter
    kbet = importr('kBET')
    # all_batches_r = robjects.IntVector(all_batches[all_ranks])
    # all_data_r.colnames = robjects.StrVector([str(x) for x in range(df.shape[1])])
    # labels = ['label1', 'label2']
    labels = robjects.StrVector(['label1'])
    new_meta_data = robjects.IntVector(meta_data)
    newdata = robjects.r.matrix(robjects.FloatVector(data.reshape(-1)), nrow=data.shape[0])

    new_meta_data.colnames = labels
    results = kbet.kBET(newdata, new_meta_data, plot=False)
    with localconverter(robjects.default_converter + pandas2ri.converter):
        results = np.array(robjects.conversion.rpy2py(results))
    # results[0]: summary - a rejection rate for the data, an expected rejection
    # rate for random labeling and the significance for the observed result
    try:
        mean = results[0]['kBET.observed'][0]
    except:
        mean = 1
    return mean  # , results[0]['kBET.signif'][0]

