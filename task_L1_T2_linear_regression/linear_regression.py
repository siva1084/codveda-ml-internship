"""Level 1 - Task 2: Simple Linear Regression (Codveda ML Internship)

Run:  python linear_regression.py
Needs: pandas, numpy, scikit-learn, matplotlib
Data:  ../data/house_prices.csv  (space-separated, no header)
Saves: coefficients.png and predictions.png next to this script.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # save charts to files; remove this line to pop up windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
HERE = Path(__file__).resolve().parent
DATA_PATH = HERE.parent / "data" / "house_prices.csv"

# 1. Load (file has no header, so we name the columns; MEDV is price in $1000s)
cols = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS",
        "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV"]
df = pd.read_csv(DATA_PATH, sep=r"\s+", header=None, names=cols)
print("Shape:", df.shape)
print("Missing values:", int(df.isna().sum().sum()), "| Duplicates:", int(df.duplicated().sum()))

# 2. Preprocess: split, then scale using training data only
X, y = df.drop(columns=["MEDV"]), df["MEDV"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# 3. Train
model = LinearRegression().fit(X_train_s, y_train)
print("\nIntercept:", round(model.intercept_, 2))

# 4. Interpret coefficients (standardised: change in price per +1 std of the feature)
coef = pd.DataFrame({"feature": X.columns, "coefficient": model.coef_})
coef = coef.reindex(coef["coefficient"].abs().sort_values(ascending=False).index)
print("\nCoefficients:\n", coef.round(3).to_string(index=False))

plt.figure(figsize=(7, 4.5))
plt.barh(coef["feature"][::-1], coef["coefficient"][::-1],
         color=["tab:green" if v > 0 else "tab:red" for v in coef["coefficient"][::-1]])
plt.axvline(0, color="black", linewidth=0.8)
plt.title("Effect of each feature on price (standardised)")
plt.tight_layout()
plt.savefig(HERE / "coefficients.png", dpi=150)
plt.close()

# 5. Evaluate with R-squared and MSE
pred_train, pred_test = model.predict(X_train_s), model.predict(X_test_s)
results = pd.DataFrame({
    "R-squared": [r2_score(y_train, pred_train), r2_score(y_test, pred_test)],
    "MSE": [mean_squared_error(y_train, pred_train), mean_squared_error(y_test, pred_test)],
}, index=["Train", "Test"])
results["RMSE ($1000s)"] = np.sqrt(results["MSE"])
print("\nEvaluation:\n", results.round(3))

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
ax[0].scatter(y_test, pred_test, alpha=0.7)
lims = [y_test.min(), y_test.max()]
ax[0].plot(lims, lims, "r--")
ax[0].set(xlabel="Actual price", ylabel="Predicted price", title="Actual vs predicted (test)")
ax[1].scatter(pred_test, y_test - pred_test, alpha=0.7)
ax[1].axhline(0, color="r", linestyle="--")
ax[1].set(xlabel="Predicted price", ylabel="Error", title="Residuals (test)")
plt.tight_layout()
plt.savefig(HERE / "predictions.png", dpi=150)
plt.close()
print("\nSaved coefficients.png and predictions.png")
