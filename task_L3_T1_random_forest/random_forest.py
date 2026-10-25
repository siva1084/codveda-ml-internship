"""Level 3 - Task 1: Random Forest for churn (Codveda ML Internship).

Run: python random_forest.py   (the grid search takes under a minute)
Needs: pandas, numpy, scikit-learn, matplotlib
Data: ../data/churn_train.csv and churn_test.csv
Saves: figure_1.png (confusion matrix), figure_2.png (feature importance)
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


# Level 3 - Task 1: Random Forest Classifier

import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score

train = pd.read_csv(f"{DATA_DIR}/churn_train.csv"); test = pd.read_csv(f"{DATA_DIR}/churn_test.csv")
print("Train:", train.shape, "| Test:", test.shape, "| Churn rate (train): %.1f%%" % (train["Churn"].mean() * 100))

drop_cols = ["State", "Total day charge", "Total eve charge", "Total night charge", "Total intl charge"]
y_train = train["Churn"].astype(int); y_test = test["Churn"].astype(int)
X_train = train.drop(columns=drop_cols + ["Churn"]); X_test = test.drop(columns=drop_cols + ["Churn"])
X_train["Area code"] = X_train["Area code"].astype(str); X_test["Area code"] = X_test["Area code"].astype(str)
cat_cols = X_train.select_dtypes(exclude="number").columns.tolist()

# Trees do not need scaling, so we only one-hot encode the categories
pre = ColumnTransformer([("cat", OneHotEncoder(drop="if_binary", handle_unknown="ignore", sparse_output=False), cat_cols)],
                        remainder="passthrough", verbose_feature_names_out=False)
Xtr = pre.fit_transform(X_train); Xte = pre.transform(X_test)
names = pre.get_feature_names_out()
print("Features:", Xtr.shape[1])


# Step 1: Preprocess


# Step 2: Tune the hyperparameters

param_grid = {"n_estimators": [100, 300],      # number of trees
              "max_depth": [5, 10, 20, None],  # how deep each tree may grow
              "min_samples_leaf": [1, 3]}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# Churn is imbalanced, so we tune for F1 instead of accuracy
grid = GridSearchCV(RandomForestClassifier(random_state=42, n_jobs=-1), param_grid, scoring="f1", cv=cv, n_jobs=-1)
grid.fit(Xtr, y_train)
print("Best parameters:", grid.best_params_)
print("Best cross-validated F1: %.3f" % grid.best_score_)
best = grid.best_estimator_


# What this means: `GridSearchCV` tries every combination in the grid with 5-fold cross-validation on the training data and keeps the best. More trees make results steadier; `max_depth` and `min_samples_leaf` limit how closely each tree can memorise the training data. We score with F1 because accuracy would reward a model that mostly says "no churn".


# Step 3: Cross-validation and test evaluation

for metric in ["accuracy", "precision", "recall", "f1"]:
    s = cross_val_score(best, Xtr, y_train, cv=cv, scoring=metric)
    print("CV %-9s %.3f (+/- %.3f)" % (metric, s.mean(), s.std()))

pred = best.predict(Xte)
print("\nTest set:")
print(classification_report(y_test, pred, target_names=["Stay", "Churn"]))
print("Test ROC AUC: %.3f" % roc_auc_score(y_test, best.predict_proba(Xte)[:, 1]))
ConfusionMatrixDisplay(confusion_matrix(y_test, pred), display_labels=["Stay", "Churn"]).plot(colorbar=False)
plt.title("Confusion matrix (test)"); plt.show()


# What this means: The cross-validation scores (mean and spread over 5 folds) show how stable the model is on data it was not trained on. The test set is a final, one-time check. Compare recall (churners caught) with the logistic regression model, where it was very low; the confusion matrix shows exactly how many churners are caught and missed.


# Step 4: Feature importance

imp = pd.Series(best.feature_importances_, index=names).sort_values()
print(imp.sort_values(ascending=False).head(8).round(3).to_string())
imp.tail(10).plot.barh(figsize=(7, 4.5), color="tab:blue")
plt.title("Top 10 most important features"); plt.xlabel("Importance (share of impurity reduction)")
plt.tight_layout(); plt.show()


# What this means: Importance shows how much each feature helps the trees make good splits (values add up to 1). It tells you what the model relies on, not the direction of the effect, and it can favour features with many distinct values. Use it as a guide and not as proof of cause.
