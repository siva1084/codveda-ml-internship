
# Level 3, Task 2: Support Vector Machine (SVM) for Classification

## Problem
Classify customers as churn or stay with an SVM, compare linear and RBF kernels, and visualise the decision boundary.

## Dataset
Codveda churn data (binary target): `data/churn_train.csv` for training and `data/churn_test.csv` as the held-out test set. Iris was not used because it has three classes, and this task needs binary classification.

## Steps
1. Dropped `State` and the four `Total ... charge` columns; one-hot encoded categories and standardised numbers (fitted on train only). Scaling is essential for SVMs.
2. Trained a linear SVM and an RBF SVM, plus a small grid search for the RBF (`C` in 0.1/1/10, `gamma` in scale/0.01/0.1, scored on AUC).
3. Evaluated on the test set with accuracy, precision, recall and AUC.
4. Plotted the decision boundary using only 2 features (Total day minutes, Customer service calls), since a boundary can only be drawn in 2D.

## Results
Baseline: always predicting "stay" gives 0.858 accuracy.

| Model | Accuracy | Precision | Recall | AUC |
|---|---|---|---|---|
| Linear | 0.858 | 0.000 | 0.000 | 0.781 |
| RBF | 0.915 | 0.896 | 0.453 | 0.928 |
| RBF (tuned: C=1, gamma=scale) | 0.915 | 0.896 | 0.453 | 0.928 |

- The **linear kernel predicts "stay" for every customer**, so its accuracy equals the baseline and precision and recall are 0. It still ranks customers somewhat (AUC 0.781).
- The **RBF kernel** is clearly better: churn is not separable by a straight line. It catches 45% of churners with 90% precision.
- The grid search chose the default RBF settings, so tuning added nothing here.
- The 2-feature boundary chart is an illustration only; the full models use all 16 features.

For comparison, the Random Forest (Task 1) reaches recall 0.65 and AUC 0.925 on the same data.

## Next steps
Try `class_weight="balanced"` to improve recall, or lower the decision threshold.

## Run
`python svm.py` (saves `figure_1.png` ROC curves and `figure_2.png` decision boundaries) or open `svm.ipynb`.
