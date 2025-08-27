import warnings
warnings.filterwarnings("ignore")

from types import SimpleNamespace
from bernn.dl.train.train_ae_classifier_holdout import TrainAEClassifierHoldout  # [`bernn.dl.train.train_ae_classifier_holdout.TrainAEClassifierHoldout`](bernn/dl/train/train_ae_classifier_holdout.py)

# 1) Minimal args — let TrainAE fill the rest with safe defaults
args = SimpleNamespace(
    device='cpu',
    dataset='dummy',    # use the built-in dummy generator: [`bernn.utils.data_getters.get_dummy`](bernn/utils/data_getters.py)
    bs=16,
    n_epochs=100,         # keep it short
    kan=0,              # use standard AE (lighter)
    bdisc=0,            # disable batch discriminator for simplicity
    log_mlflow=False,
    log_neptune=False,
    log_tb=False,
    log_metrics=False,
    log_plots=False,
    keep_models=False,
    groupkfold=True,
    warmup_after_warmup=False  # TODO add to defaults
)

# 2) Minimal hyperparameters required by TrainAEClassifierHoldout.train
params = {
    'nu': 0.1,
    'lr': 1e-3,
    'wd': 0.0,
    'smoothing': 0.0,
    'margin': 1.0,
    'warmup': 1,
    'disc_b_warmup': 0,
    'dropout': 0.0,
    'scaler': 'standard',
    'layer2': 32,
    'layer1': 64,
    # fixed/disabled knobs
    'gamma': 0.0, 'beta': 0.0, 'zeta': 0.0,
    'thres': 0.0, 'prune_threshold': 0.0,
    'reg_entropy': 0.0, 'l1': 0.0,
}

trainer = TrainAEClassifierHoldout(
    args,
    path='.',                 # not used by dummy, safe to keep '.'
    fix_thres=-1,
    load_tb=False,
    log_metrics=False,
    keep_models=False,
    log_inputs=False,
    log_plots=False,
    log_tb=False,
    log_neptune=False,
    log_mlflow=False,
    groupkfold=True,
    pools=False
)

# Run a tiny training pass
trainer.train(params)
print("Done.")