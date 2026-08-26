# MassBench Adenocarcinoma Dataset Documentation

## Description
Mass spectrometry dataset for adenocarcinoma classification with batch effects. This dataset contains tandem mass spectrometry (MS/MS) data from adenocarcinoma specimens.

## Task
Multi-class classification of sample types in adenocarcinoma specimens

## Dataset Metrics

| Metric | Value |
|--------|-------|
| Total Samples (All) | 642 |
| Number of Batches | 3 |
| Number of Classes | 3 |
| Domain | Mass Spectrometry |

## Feature Information

| Set | Features |
|-----|----------|
| Training | 6,464 |
| Test | 6,463 |
| Note | Feature count may vary slightly due to preprocessing |

## Class Distribution (Overall)

| Class | Samples | Percentage |
|-------|---------|-----------|
| 1 | 497 | 77.4% |
| QC | 74 | 11.5% |
| 0 | 71 | 11.1% |
| **Total** | **642** | **100%** |

## Batch Distribution (Overall)

| Batch | Samples | Percentage |
|-------|---------|-----------|
| 1 | 217 | 33.8% |
| 2 | 217 | 33.8% |
| 3 | 208 | 32.4% |
| **Total** | **642** | **100%** |

## Batch × Class Cross-tabulation

```
       Class 0  Class 1  Class QC  Total
Batch 1      24      168        25     217
Batch 2      24      168        25     217
Batch 3      23      161        24     208
--------
Total        71      497        74     642
```

## Train/Valid/Test Split Information

### Data Split Strategy
- **Method**: StratifiedGroupKFold (5-fold)
  - Stratifies by class labels
  - Groups by batch to prevent batch leakage
  - First fold: Train/Valid split
  - Second fold: Test split (for cross-validation)

### Typical Split Distribution
Using StratifiedGroupKFold with 5 splits and batches 1-3:

#### Training Set
- **Total Samples**: ~434 samples (67.5%)
- Approximately 80% of each batch combined with stratification

#### Validation Set
- **Total Samples**: ~200 samples (31.2%)
- Approximately 20% of training samples
- Stratified by class and group

#### Test Set
- **Total Samples**: ~208 samples (32.4%)
- May contain full samples from one or more batches (depending on fold iteration)

### Class Distribution Per Set

#### Training Set (approx.)
| Class | Samples | Percentage |
|-------|---------|-----------|
| 1 | 335 | 77% |
| QC | 50 | 11.5% |
| 0 | 49 | 11.5% |

#### Test Set (approx.)
| Class | Samples | Percentage |
|-------|---------|-----------|
| 1 | 162 | 77.8% |
| QC | 24 | 11.5% |
| 0 | 22 | 10.6% |

### Batch Distribution Per Set

#### Training Set (approx. for fold 1)
| Batch | Samples | Percentage |
|-------|---------|-----------|
| 1 | 173 | 39.9% |
| 2 | 173 | 39.9% |
| 3 | 88 | 20.3% |

#### Test Set (when using fold 2)
| Batch | Samples | Percentage |
|-------|---------|-----------|
| 3 (full) | 208 | 100% |

*Note: Actual distributions vary based on which fold is selected for testing*

## Data Characteristics

### Batch Effects
- 3 different measurement batches with potentially different instrumental or sample preparation conditions
- Relatively balanced batch sizes (32-34% each)
- Batch × Class interactions are roughly proportional across batches

### Class Imbalance
- Strong class imbalance with Class 1 being dominant (77.4%)
- QC (Quality Control) samples: 11.5%
- Class 0: 11.1%

## Usage Notes

### Loading Data
```python
from bernn.dl.train.train_ae import TrainAE

trainer = TrainAE(
    dataset='amide',
    path='data/',
    groupkfold=True,  # Important: use GroupKFold to prevent batch leakage
)
```

### Features
- Handled through scikit-learn's StratifiedGroupKFold
- Batch information prevents test leakage during k-fold cross-validation
- Test sets in later folds provide additional validation coverage

## Citation & Reference
- Dataset: MassBench Adenocarcinoma / Amide Dataset
- Associated code: BERNN (Batch Effect Removal Neural Networks)
