"""Level 2 - Task 1: Logistic Regression for churn (Codveda ML Internship).

Run: python logistic_regression.py
Needs: pandas, numpy, scikit-learn, matplotlib
Data: ../data/churn_train.csv and churn_test.csv
Saves: figure_1.png (odds ratios), figure_2.png (confusion matrix + ROC)
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


# Level 2 - Task 1: Logistic Regression for Binary Classification

import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, confusion_matrix,
                             ConfusionMatrixDisplay, roc_curve, roc_auc_score)

train = pd.read_csv(f"{DATA_DIR}/churn_train.csv")
test = pd.read_csv(f"{DATA_DIR}/churn_test.csv")
print("Train:", train.shape, "| Test:", test.shape)
print("Missing values:", int(train.isna().sum().sum()), "| Duplicates:", int(train.duplicated().sum()))
print("Churn rate (train): %.1f%%" % (train["Churn"].mean() * 100))
print(train.head(3).T.head(8))


# Step 1: Preprocess

drop_cols = ["State", "Total day charge", "Total eve charge", "Total night charge", "Total intl charge"]
y_train = train["Churn"].astype(int); y_test = test["Churn"].astype(int)
X_train = train.drop(columns=drop_cols + ["Churn"]); X_test = test.drop(columns=drop_cols + ["Churn"])
X_train["Area code"] = X_train["Area code"].astype(str); X_test["Area code"] = X_test["Area code"].astype(str)

cat_cols = X_train.select_dtypes(exclude="number").columns.tolist()
num_cols = X_train.select_dtypes(include="number").columns.tolist()
pre = ColumnTransformer([("num", StandardScaler(), num_cols),
                         ("cat", OneHotEncoder(drop="if_binary", handle_unknown="ignore", sparse_output=False), cat_cols)])
Xtr = pre.fit_transform(X_train)   # fit on TRAIN only
Xte = pre.transform(X_test)
feature_names = pre.get_feature_names_out()
print("Features:", Xtr.shape[1], "| Categorical:", cat_cols)


# Step 2: Train the model

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(Xtr, y_train)
print("Trained. Intercept:", round(model.intercept_[0], 3))


# Step 3: Interpret coefficients and odds ratios

coef = pd.DataFrame({"feature": [f.split("__")[1] for f in feature_names], "coefficient": model.coef_[0]})
coef["odds_ratio"] = np.exp(coef["coefficient"])
coef = coef.reindex(coef["coefficient"].abs().sort_values(ascending=False).index).reset_index(drop=True)
print(coef.round(3).head(10).to_string(index=False))

top = coef.head(10)[::-1]
plt.figure(figsize=(7, 4.5))
plt.barh(top["feature"], top["odds_ratio"], color=["tab:red" if v > 1 else "tab:green" for v in top["odds_ratio"]])
plt.axvline(1, color="black", linewidth=0.8)
plt.title("Odds ratios (red = more likely to churn)")
plt.tight_layout(); plt.show()


# What this means: A coefficient is the change in the log-odds of churning. The odds ratio (`exp(coefficient)`) is easier to read: above 1 raises the odds of churn, below 1 lowers them, and 1 means no effect.


# Step 4: Evaluate on the test set

pred = model.predict(Xte)
proba = model.predict_proba(Xte)[:, 1]
print("Accuracy :", round(accuracy_score(y_test, pred), 3))
print("Precision:", round(precision_score(y_test, pred), 3))
print("Recall   :", round(recall_score(y_test, pred), 3))
print("ROC AUC  :", round(roc_auc_score(y_test, proba), 3))
print("Baseline accuracy (always predict 'no churn'):", round(1 - y_test.mean(), 3))

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
ConfusionMatrixDisplay(confusion_matrix(y_test, pred), display_labels=["Stay", "Churn"]).plot(ax=ax[0], colorbar=False)
ax[0].set_title("Confusion matrix (test)")
fpr, tpr, _ = roc_curve(y_test, proba)
ax[1].plot(fpr, tpr, label="AUC = %.3f" % roc_auc_score(y_test, proba))
ax[1].plot([0, 1], [0, 1], "k--", label="Random guess")
ax[1].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curve (test)")
ax[1].legend(); plt.tight_layout(); plt.show()


# What this means:
