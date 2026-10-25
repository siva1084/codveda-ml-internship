"""Level 2 - Task 2: Decision Tree on Iris (Codveda ML Internship).

Run: python decision_tree.py
Needs: pandas, numpy, scikit-learn, matplotlib
Data: ../data/iris.csv
Saves: figure_1.png (full tree), figure_2.png (pruned tree)
"""

import matplotlib
matplotlib.use("Agg")  # charts are saved as figure_1.png, figure_2.png, ... next to this script
import matplotlib.pyplot as plt
from pathlib import Path
HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent / "data"
_n = [0]
def _save():
    _n[0] += 1
    plt.savefig(HERE / f"figure_{_n[0]}.png", dpi=150, bbox_inches="tight")
    plt.close()
plt.show = _save


# Level 2 - Task 2: Decision Trees for Classification

import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, f1_score, classification_report

df = pd.read_csv(f"{DATA_DIR}/iris.csv")
print("Shape:", df.shape, "| Missing:", int(df.isna().sum().sum()), "| Duplicates:", int(df.duplicated().sum()))
print(df["species"].value_counts().to_string())

X = df.drop(columns="species"); y = df["species"]
# stratify keeps the same species mix in train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
print("Train:", X_train.shape, "| Test:", X_test.shape)


# What this means: The data is clean and perfectly balanced, and a tree does not need scaling or encoding, so preprocessing is minimal. `stratify=y` keeps the species mix equal in train and test.


# Step 1: Train an unpruned tree

full_tree = DecisionTreeClassifier(random_state=42).fit(X_train, y_train)
print("Depth:", full_tree.get_depth(), "| Leaves:", full_tree.get_n_leaves())
print("Train accuracy:", round(full_tree.score(X_train, y_train), 3))
print("Test accuracy :", round(full_tree.score(X_test, y_test), 3))


# What this means: With no limits, a tree keeps splitting until every training flower is classified correctly (train accuracy 1.0). That can mean it memorises quirks of the training set, which is overfitting.


# Step 2: Visualise the tree

plt.figure(figsize=(13, 7))
plot_tree(full_tree, feature_names=X.columns, class_names=full_tree.classes_, filled=True, rounded=True, fontsize=8)
plt.title("Unpruned decision tree"); plt.show()


# What this means: Each box asks a yes/no question about one measurement (for example a petal width threshold). Colour shows the majority species and the `gini` value shows impurity (0 means the box is pure). Petal measurements do most of the work.


# Step 3: Prune the tree

# Cost-complexity pruning: a larger ccp_alpha removes more branches.
# We choose it using 5-fold cross-validation on the TRAINING data only ("one standard error" rule):
# take the most pruned tree whose CV score is within one standard error of the best score.
alphas = np.unique(np.round(full_tree.cost_complexity_pruning_path(X_train, y_train).ccp_alphas, 4))
scores = [cross_val_score(DecisionTreeClassifier(random_state=42, ccp_alpha=a), X_train, y_train, cv=5) for a in alphas]
means = np.array([sc.mean() for sc in scores]); se = scores[int(np.argmax(means))].std() / np.sqrt(5)
best_alpha = alphas[means >= means.max() - se].max()
print("Best CV accuracy: %.3f | chosen ccp_alpha: %s" % (means.max(), best_alpha))

pruned = DecisionTreeClassifier(random_state=42, ccp_alpha=best_alpha).fit(X_train, y_train)
print("Pruned depth:", pruned.get_depth(), "| Leaves:", pruned.get_n_leaves())

plt.figure(figsize=(9, 5))
plot_tree(pruned, feature_names=X.columns, class_names=pruned.classes_, filled=True, rounded=True, fontsize=9)
plt.title("Pruned decision tree"); plt.show()


# What this means: Pruning cuts branches that add little accuracy, giving a smaller tree that is easier to explain and less likely to overfit. We pick the amount of pruning with cross-validation on the training data, so the test set stays untouched. We choose the *simplest* tree whose score is within one standard error of the best, because a smaller tree that performs equally well is preferred.


# Step 4: Evaluate

rows = {}
for name, m in [("Unpruned", full_tree), ("Pruned", pruned)]:
    p = m.predict(X_test)
    rows[name] = {"Leaves": m.get_n_leaves(),
                  "CV accuracy (train)": cross_val_score(m, X_train, y_train, cv=5).mean(),
                  "Test accuracy": accuracy_score(y_test, p),
                  "Test F1 (macro)": f1_score(y_test, p, average="macro")}
print(pd.DataFrame(rows).T.round(3))
print()
print(classification_report(y_test, pruned.predict(X_test)))


# What this means:
