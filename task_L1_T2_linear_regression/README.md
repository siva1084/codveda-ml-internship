# Level 1, Task 2: Simple Linear Regression Model

## Problem
Predict house prices (a continuous value) with linear regression and explain what the model learned.

## Dataset
Codveda's house price file (`data/house_prices.csv`): 506 rows, 13 features, target `MEDV` (median home value in $1000s). No missing values and all columns numeric. The file has no header, so column names are set in the notebook.

## Steps
1. Loaded the data, checked for missing values and duplicates (none), and plotted a correlation heatmap.
2. Split 80/20 with `random_state=42` and standardised features (scaler fitted on train only).
3. Trained `LinearRegression` from scikit-learn.
4. Interpreted the coefficients and evaluated with R-squared and MSE.

## Results
| Set | R-squared | MSE | RMSE ($1000s) |
|---|---|---|---|
| Train | 0.751 | 21.64 | 4.65 |
| Test | 0.669 | 24.29 | 4.93 |

The model explains about 67% of price variation on unseen data, with a typical error of roughly $4,900.

**Strongest effects (standardised coefficients):** `LSTAT` -3.61, `RM` +3.15, `DIS` -3.08, `RAD` +2.25, `PTRATIO` -2.04. More rooms raise price; a higher share of lower-income residents, greater distance from job centres, and more crowded schools lower it.

## Notes
The `B` column is derived from racial demographics. It is kept only because it is in Codveda's file and should be excluded in real projects.

## Run
Open `linear_regression.ipynb` and run all cells. Tools: Python, pandas, scikit-learn, matplotlib, seaborn.
