# Documentation: Cell Tower Forecasting Codebase

This codebase implements **time series forecasting for cell tower capacity planning** using multiple models (Seasonal Naive, Prophet, and Linear Autoregression). It focuses on evaluating model accuracy against test data.

The project currently consists of **three main scripts**:

---

## 1. `decode_as_entries.py`
**Purpose**: Inspect the dataset hosted on Hugging Face to verify accessibility and structure.

### Functions:
- **`create_retry_session(total_retries=5, backoff_factor=0.8)`**
  - Creates an HTTP session with retry logic for robust dataset access.

- **`fetch_first_rows()`**
  - Fetches the first few rows of the dataset.
  - If the request fails, raises an error with diagnostic information.

- **`list_splits_for_debug()`**
  - Lists available dataset splits/configs from Hugging Face for troubleshooting.

### Usage:
Run the script to quickly confirm dataset availability and preview data rows.

---

## 2. `seasonal_naive_forecast.py`
**Purpose**: Core forecasting engine. Implements forecasting models, evaluation metrics, visualization, and a flexible `run()` function.

### Key Features:
- **Data Handling**:
  - Reads train and test splits for supported metrics (`DLPRB`, `DLThpVol`, `DLThpTime`, `MR_number`).
  - Validates beam IDs.

- **Models Supported**:
  - **Seasonal Naive**: Repeats the last seasonal cycle (period = 168 hours).
  - **Prophet**: Decomposable time series model supporting trend + seasonality.
  - **Linear AR**: Custom autoregressive model with lagged values + cyclical features.

- **Metrics**:
  - `MAE` (Mean Absolute Error)
  - `MAPE` (Mean Absolute Percentage Error)
  - `RMSE` (Root Mean Square Error)
  - `sMAPE` (Symmetric MAPE)
  - `Median AE` (Median Absolute Error)

- **Visualization**:
  - Hexbin plots of actual vs predicted values.

- **Main Function**:
  - **`run(...)`**
    - Loads training/testing data.
    - Runs selected model.
    - Computes metrics.
    - Prints results.
    - Returns a results dictionary for programmatic evaluation.

- **CLI**:
  - Script can be run directly with arguments for metric, beam, target week, model choice, Prophet hyperparameters, and visualization options.

### Example:
```bash
python seasonal_naive_forecast.py --metric DLPRB --beam 0_0_1 --week 6 --model prophet \
    --prophet-daily-order 8 --prophet-weekly-order 10 --prophet-changepoint-prior 0.1
```

---

## 3. `evaluate_via_run.py`
**Purpose**: Evaluate and compare all models (Seasonal Naive, Prophet, Linear AR) using the shared `run()` function from `seasonal_naive_forecast.py`.

### Workflow:
1. Imports `run()` from `seasonal_naive_forecast.py`.
2. Runs all three models sequentially on the same dataset split.
3. Collects returned metrics into a results table.
4. Prints comparison results.
5. Optionally saves results to CSV.

### Functions:
- **`evaluate_all(...)`**
  - Calls `run()` for each model.
  - Collects and prints metrics.
  - Saves results to CSV if specified.

- **`parse_args()`**
  - Defines CLI arguments (metric, beam, week, Prophet params, log transform, samples, save_csv).

### Example:
```bash
python evaluate_via_run.py --metric DLPRB --beam 0_0_1 --week 6 --save-csv results.csv
```

### Output:
- Console table of model accuracies.
- Optional CSV file with the same results.

---

## Typical Workflow

1. **Check Dataset**:
   ```bash
   python decode_as_entries.py
   ```
   Confirms dataset accessibility and shows sample rows.

2. **Run Individual Model**:
   ```bash
   python seasonal_naive_forecast.py --metric DLPRB --beam 0_0_1 --week 6 --model seasonal_naive
   ```

3. **Compare All Models**:
   ```bash
   python evaluate_via_run.py --metric DLPRB --beam 0_0_1 --week 6 --save-csv comparison.csv
   ```

---

## Summary
- **`decode_as_entries.py`** → Data inspection.
- **`seasonal_naive_forecast.py`** → Core forecasting + metrics + visualization.
- **`evaluate_via_run.py`** → Unified evaluation/comparison across all models.

This structure ensures **clean separation of concerns**: data validation, forecasting engine, and evaluation wrapper.

