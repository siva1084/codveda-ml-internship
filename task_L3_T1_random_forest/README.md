# Level 3, Task 1: Random Forest Classifier

## Problem
Predict customer churn with a tuned Random Forest, evaluate it with cross-validation and classification metrics, and find the most important features.

## Dataset
Codveda churn data: `data/churn_train.csv` (2,666 customers) for training and tuning, `data/churn_test.csv` (667 customers) as the held-out test set. 14.6% of training customers churn (imbalanced).

## Steps
1. Dropped `State` and the four `Total ... charge` columns (duplicates of the minutes columns); one-hot encoded categories. No scaling needed for trees. 16 features.
2. Tuned `n_estimators` (100, 300), `max_depth` (5, 10, 20, None) and `min_samples_leaf` (1, 3) with `GridSearchCV`, 5-fold stratified CV, scoring on F1 because of the class imbalance.
3. Evaluated with 5-fold cross-validation and a final test-set report.
4. Plotted feature importance.

## Results
Best parameters: `n_estimators=300`, `max_depth=20`, `min_samples_leaf=1`.

| 5-fold CV (train) | Mean | Std |
|---|---|---|
| Accuracy | 0.948 | 0.007 |
| Precision | 0.939 | 0.045 |
| Recall | 0.688 | 0.031 |
| F1 | 0.793 | 0.027 |

**Test set (churn class):** precision 0.95, recall 0.65, F1 0.78; overall accuracy 0.95; ROC AUC 0.925.

For comparison, logistic regression on the same data reached recall 0.18 and AUC 0.825, so the forest catches far more churners and ranks customers better. It still misses about a third of them.

**Top features by importance:** Total day minutes (0.238), Customer service calls (0.123), Total eve minutes (0.100), International plan (0.092), Total intl minutes (0.078). Importance shows what the model relies on, not the direction of the effect.

## Run
`python random_forest.py` (under a minute; saves `figure_1.png` and `figure_2.png`) or open `random_forest.ipynb`.
