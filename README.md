To run all analysis, run `make all`.
You might need to install packages via either pip or conda.

# Data
## raw/inspections.csv
Contains inspection results.

## raw/portcalls_df.pkl
Contains information about the portcalls. Used by src/clean.py.

## raw/portcalls_new.csv
Not used. Contains a newer set of portcalls.

## raw/ports_df.pkl
Not used. Contains information about the ports.

# Notebooks
## 04-performance.ipynb
Summarizes information on the performance of learned classifiers.

## 03-sanity_checks.ipynb
In this notebook we check the following:
- whether folds are uniform with PCA or t-SNE.
- whether we can predict (either label or flag) with just PCA or t-SNE
- whether number of portcalls, port stay, or travel times features are sufficient to predict (either label or flag)

## 01-stats_inspections.ipynb
This notebook provides some statistics on the inspections dataset.

## 02-stats_portcalls.ipynb
This notebook features some statistics of the portcalls dataset.
We also test with a simple logistic regression model whether we can reconstruct the baseline classification using some of the information that is used to construct this baseline classification.

