# Codveda Machine Learning Internship

**Intern:** [Sivaram S] &nbsp;|&nbsp; **Company:** [Codveda Technology](https://www.codveda.com) &nbsp;|&nbsp; **Domain:** Machine Learning

Six tasks across three levels (two per level), built in Python with pandas and scikit-learn. Each task folder has a Jupyter notebook with explanations and saved outputs, a standalone `.py` script, and its own README with the full results.

## Tasks and key results

| Level | Task | Dataset | Key result | Video |
|---|---|---|---|---|
| 1 | [Data Preprocessing](task_L1_T1_preprocessing) | Stock prices (497,472 rows) | 27 missing values filled, 512 model-ready features, no data leakage | [link](#) |
| 1 | [Linear Regression](task_L1_T2_linear_regression) | House prices (506 rows) | Test R² 0.669, MSE 24.29 | [link](#) |
| 2 | [Logistic Regression](task_L2_T1_logistic_regression) | Telecom churn | Test AUC 0.825; recall only 0.179 | [link](#) |
| 2 | [Decision Tree](task_L2_T2_decision_tree) | Iris (150 rows) | Pruned tree: 6 leaves, test accuracy 0.967 | [link](#) |
| 3 | [Random Forest](task_L3_T1_random_forest) | Telecom churn | Test AUC 0.925; churn precision 0.95, recall 0.65 | [link](#) |
| 3 | [SVM](task_L3_T2_svm) | Telecom churn | RBF kernel accuracy 0.915, AUC 0.928 (linear kernel predicted "stay" for everyone) | [link](#) |

On the imbalanced churn data, always predicting "no churn" scores 0.858 accuracy, so precision, recall and AUC are reported alongside accuracy.

## Project structure

```
codveda_ml/
├── data/                              # Datasets provided by Codveda
├── task_L1_T1_preprocessing/
├── task_L1_T2_linear_regression/
├── task_L2_T1_logistic_regression/
├── task_L2_T2_decision_tree/
├── task_L3_T1_random_forest/
├── task_L3_T2_svm/
└── requirements.txt
```

Each task folder contains `<task>.ipynb`, `<task>.py`, a `README.md`, and chart images (`.png`).

## How to run

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

- **Notebook:** open the `.ipynb` in VS Code or Jupyter, select the `.venv` kernel and click **Run All**.
- **Script:** for example `python task_L1_T1_preprocessing/preprocessing.py`. Charts are saved as image files in the task folder.

Keep the `data/` folder next to the task folders, because the code finds the datasets by that layout.

## Tools

Python, pandas, NumPy, scikit-learn, matplotlib, seaborn, Jupyter.

## Notes

- Every preprocessing step is fitted on training data only to avoid data leakage, and `random_state=42` is used throughout for reproducibility.
- The house-price data includes a column (`B`) derived from racial demographics. It is kept only because it is part of the provided dataset and should be excluded in real projects.
