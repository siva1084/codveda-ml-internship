# Level 2, Task 1: Logistic Regression for Binary Classification

## Problem
Predict whether a telecom customer will churn (leave), and explain which factors are linked to churning.

## Dataset
Codveda churn data: `data/churn_train.csv` (2,666 customers) for training and `data/churn_test.csv` (667 customers) as the held-out test set. 19 features, target `Churn`. No missing values; 14.6% of training customers churn (imbalanced).

## Steps
1. Dropped `State` (51 sparse categories) and the four `Total ... charge` columns (each is minutes times a fixed rate, so they duplicate the minutes columns).
2. Treated `Area code` as a category, one-hot encoded the yes/no plans, and standardised numeric columns (fitted on train only). Result: 16 features.
3. Trained `LogisticRegression` and computed odds ratios with `exp(coefficient)`.
4. Evaluated on the test set with accuracy, precision, recall, confusion matrix, and the ROC curve.

## Results
| Metric | Test |
|---|---|
| Accuracy | 0.853 |
| Precision | 0.459 |
| Recall | 0.179 |
| ROC AUC | 0.825 |

**Important:** always predicting "no churn" scores 0.858 accuracy, so accuracy alone is misleading. The model ranks customers reasonably well (AUC 0.825) but at the default 0.5 threshold it catches only about 18% of churners.

**Strongest odds ratios:** International plan 7.73 (much higher churn odds), Voice mail plan 0.24 (lower), Total day minutes 1.97 and Customer service calls 1.93 (per +1 standard deviation). These show association, not cause.

## Next steps
Try `class_weight="balanced"`, lower the decision threshold, or compare tree-based models.

## Run
`python logistic_regression.py` (saves `figure_1.png` and `figure_2.png`) or open `logistic_regression.ipynb`.
