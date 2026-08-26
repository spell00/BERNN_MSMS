import subprocess
import sys
import os

def test_minimal_pipeline():
    """
    Integration test: run the training pipeline with minimal settings and check for success.
    """
    cmd = [
        sys.executable, 'bernn/dl/train/train_ae_classifier_holdout.py',
        '--early_stop=1', '--n_epochs=1', '--kan=0', '--variational=0',
        '--train_after_warmup=1', '--tied_weights=0', '--bdisc=1', '--rec_loss=l1', '--dloss=DANN',
        '--csv_file=adenocarcinoma_data.csv', '--remove_zeros=0', '--groupkfold=1',
        '--device=cpu', '--dataset=amide', '--n_trials=1', '--n_repeats=1',
        '--exp_id=integration_test', '--path=data', '--pool=0', '--log_metrics=1',
        '--best_features_file=', '--update_grid=0', '--use_l1=1', '--prune_threshold=0',
        '--warmup_after_warmup=0', '--prune_network=0', '--log_mlflow=0',
        '--log_tb=0', '--log_dvclive=0'
    ]
    print('Running:', ' '.join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr, file=sys.stderr)
    assert result.returncode == 0, f"Pipeline failed with code {result.returncode}"
    assert 'Epoch' in result.stdout or 'See results using:' in result.stdout, "No training output detected"

if __name__ == "__main__":
    test_minimal_pipeline()
