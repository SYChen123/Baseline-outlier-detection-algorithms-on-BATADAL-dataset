# Baseline-outlier-detection-algorithms-on-BATADAL-dataset

Apply some baseline outlier detection algorithms on BATADAL dataset. The baseline algorithms include one-class SVM, Isolation Forest, LOF, KNN, XBGOD. They're trained on BATADAL_dataset04.csv and evaluated on BATADAL_test_dataset.csv.

For implementation of these models, the corresponding interfaces in pyod are called. 

## Requirements

1. numpy
2. pandas
3. matplotlib
4. scikit-learn
5. pyod

## Usage

**od_BATADAL.ipynb**  contains the usage and results.

**utils.py**  contains some useful functions.