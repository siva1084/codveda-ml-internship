# Level 2, Task 2: Decision Trees for Classification

## Problem
Predict the species of an iris flower from its measurements, and show how pruning keeps a tree simple.

## Dataset
Codveda `data/iris.csv`: 150 flowers, 4 measurements (sepal and petal length/width), 3 balanced species (50 each). No missing values (3 duplicate rows are left in as they are part of the standard dataset).

## Steps
1. Stratified 80/20 split with `random_state=42` (120 train, 30 test). No scaling or encoding is needed for trees.
2. Trained an unpruned `DecisionTreeClassifier` and visualised it with `plot_tree`.
3. Pruned it with cost-complexity pruning. `ccp_alpha` was chosen by 5-fold cross-validation on the training data, using the simplest tree within one standard error of the best score.
4. Evaluated with accuracy and F1-score (macro).

## Results
| Tree | Leaves | CV accuracy (train) | Test accuracy | Test F1 (macro) |
|---|---|---|---|---|
| Unpruned | 8 | 0.942 | 0.933 | 0.933 |
| Pruned (`ccp_alpha` 0.0081) | 6 | 0.942 | 0.967 | 0.967 |

The pruned tree is smaller and scores the same in cross-validation. Its test score is one flower better out of 30, which is within noise, so the fair conclusion is "simpler, with no loss in performance". Setosa is separated perfectly; the only mistakes are between versicolor and virginica.

## Run
`python decision_tree.py` (saves `figure_1.png` full tree and `figure_2.png` pruned tree) or open `decision_tree.ipynb`.
