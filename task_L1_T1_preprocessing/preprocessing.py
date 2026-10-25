"""Level 1 - Task 1: Data Preprocessing for Machine Learning (Codveda ML Internship)

Run:  python preprocessing.py
Needs: pandas, numpy, scikit-learn
Data:  ../data/stock_prices.csv  (symbol, date, open, high, low, close, volume)
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
DATA_PATH = Path(__file__).resolve().parent.parent/ "data"/ "stock_prices .csv"

# 1. Load and explore
df = pd.read_csv(DATA_PATH)
print("Shape:", df.shape)
print("Duplicate rows:", df.duplicated().sum())

# 2. Missing data
missing = df.isna().sum()
print("\nMissing values per column:\n", missing[missing > 0])
print("Total missing cells:", int(missing.sum()))

# 3. Feature preparation: turn the date into numbers, then drop it
df["date"] = pd.to_datetime(df["date"])
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["day_of_week"] = df["date"].dt.dayofweek
df = df.drop(columns=["date"])

y = df["close"]                 # example target
X = df.drop(columns=["close"])
num_cols = X.select_dtypes(include="number").columns.tolist()
cat_cols = X.select_dtypes(exclude="number").columns.tolist()
print("\nNumeric columns:", num_cols)
print("Categorical columns:", cat_cols)

# 4. Split FIRST so nothing from the test set leaks into training
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE)
print(f"\nTrain rows: {len(X_train)} | Test rows: {len(X_test)}")

# 5. Impute + scale numbers, impute + one-hot encode categories (fit on train only)
preprocessor = ColumnTransformer([
    ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                      ("scale", StandardScaler())]), num_cols),
    ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                      ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
])
X_train_p = preprocessor.fit_transform(X_train)
X_test_p = preprocessor.transform(X_test)
print("Processed train shape:", X_train_p.shape)
print("Processed test shape: ", X_test_p.shape)

# 6. Sanity check: scaled numeric columns should have mean ~0 and std ~1
scaled = X_train_p[:, :len(num_cols)]
scaled = scaled.toarray() if hasattr(scaled, "toarray") else scaled
print("\nScaled train mean (should be ~0):", np.round(scaled.mean(axis=0), 3))
print("Scaled train std  (should be ~1):", np.round(scaled.std(axis=0), 3))
print("Missing numeric cells after processing:", int(np.isnan(scaled).sum()))
print("\nDone: data is ready for a model.")
