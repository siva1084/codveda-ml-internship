# Level 1, Task 1: Data Preprocessing for Machine Learning

## Problem
Prepare a raw dataset so a machine learning model can use it: handle missing data, encode categories, scale numbers, and split into train/test sets.

## Dataset
Codveda's daily stock prices file (`data/stock_prices.csv`): 497,472 rows, 7 columns (`symbol, date, open, high, low, close, volume`), covering 505 stock symbols. Example target: `close`.

## Steps
1. **Explored** the data: types, summary statistics, duplicates (none).
2. **Found missing values:** 27 cells (`open` 11, `high` 8, `low` 8), under 0.01% of the data.
3. **Imputed** numbers with the median and text with the most frequent value, learned from the training set only.
4. **Engineered features:** converted `date` into year, month, and day of week.
5. **Encoded** `symbol` with one-hot encoding (sparse). Label encoding is explained in the notebook and why it was not used here.
6. **Standardised** numeric columns (mean 0, std 1).
7. **Split** 80/20 with `random_state=42`: 397,977 train rows and 99,495 test rows.

## Results
- Output: 512 model-ready features (7 numeric + 505 symbol columns).
- Missing numeric cells in the training set went from 24 to 0.
- Scaled columns have mean 0 and standard deviation 1.

## Notes
The split is random for learning purposes. For real stock forecasting, split by date instead.

## Run
Open `preprocessing.ipynb` and run all cells. Tools: Python, pandas, scikit-learn.
