import argparse
import os
import sys
from typing import Tuple, List
import pandas as pd
import numpy as np

try:
	from prophet import Prophet  # type: ignore
except Exception:
	Prophet = None  # Deferred import error until model selection

try:
	import matplotlib.pyplot as plt  # type: ignore
except Exception:
	plt = None  # Optional plotting

DATA_DIR = os.path.join("Beam-Level-Traffic-Timeseries-Dataset", "data")
TRAIN_FILE_TPL = os.path.join(DATA_DIR, "train", "{metric}_train_0w-5w.csv")
TEST_FILE_TPL_6 = os.path.join(DATA_DIR, "test", "{metric}_test_5w-6w.csv")
TEST_FILE_TPL_11 = os.path.join(DATA_DIR, "test", "{metric}_test_10w-11w.csv")

SUPPORTED_METRICS = [
	"DLPRB",
	"DLThpVol",
	"DLThpTime",
	"MR_number",
]

SEASONAL_PERIOD = 168  # hours in a week


def read_split(metric: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
	if metric not in SUPPORTED_METRICS:
		raise ValueError(f"Unsupported metric '{metric}'. Choose from: {', '.join(SUPPORTED_METRICS)}")
	train_path = TRAIN_FILE_TPL.format(metric=metric)
	test6_path = TEST_FILE_TPL_6.format(metric=metric)
	test11_path = TEST_FILE_TPL_11.format(metric=metric)
	for p in [train_path, test6_path, test11_path]:
		if not os.path.exists(p):
			raise FileNotFoundError(f"Missing file: {p}")
	train = pd.read_csv(train_path)
	test6 = pd.read_csv(test6_path)
	test11 = pd.read_csv(test11_path)
	return train, test6, test11


def validate_beam_column(df: pd.DataFrame, beam: str) -> None:
	if beam not in df.columns:
		raise ValueError(
			f"Beam '{beam}' not found. Example beams look like '0_0_1'. "
			f"Available columns (truncated): {df.columns[:10].tolist()} ..."
		)


def seasonal_naive_forecast(history: np.ndarray, horizon: int, period: int = SEASONAL_PERIOD) -> np.ndarray:
	if len(history) < period:
		raise ValueError(f"History length {len(history)} must be >= seasonal period {period}")
	season = history[-period:]
	repeats = int(np.ceil(horizon / period))
	forecast = np.tile(season, repeats)[:horizon]
	return forecast


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float]:
	mae = float(np.mean(np.abs(y_true - y_pred)))
	rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
	# Safe MAPE: ignore zeros in denominator
	mask = y_true != 0
	if np.any(mask):
		mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)
	else:
		mape = float("nan")
	return mae, mape, rmse


def compute_additional_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
	# sMAPE definition with small epsilon to avoid division by zero when both are zero
	eps = 1e-9
	num = np.abs(y_pred - y_true)
	den = (np.abs(y_true) + np.abs(y_pred)).astype(float)
	smape = float(np.mean(2.0 * num / (den + eps)) * 100.0)
	median_ae = float(np.median(np.abs(y_true - y_pred)))
	return smape, median_ae


def build_hourly_time_index(start: str, hours: int) -> pd.DatetimeIndex:
	start_ts = pd.to_datetime(start)
	return pd.date_range(start=start_ts, periods=hours, freq="h")


def prophet_forecast(y_train: np.ndarray, horizon: int, daily_order: int, weekly_order: int, changepoint_prior: float, seasonality_mode: str) -> np.ndarray:
	if Prophet is None:
		raise ImportError(
			"Prophet is not installed. Install with 'pip install prophet'."
		)
	# Create hourly timestamps for training and future horizon
	train_hours = len(y_train)
	train_ds = build_hourly_time_index("2024-01-01", train_hours)
	history_df = pd.DataFrame({"ds": train_ds, "y": y_train.astype(float)})
	model = Prophet(weekly_seasonality=False, daily_seasonality=False, yearly_seasonality=False, changepoint_prior_scale=changepoint_prior, seasonality_mode=seasonality_mode)
	if daily_order > 0:
		model.add_seasonality(name="daily", period=24, fourier_order=daily_order)
	if weekly_order > 0:
		model.add_seasonality(name="weekly", period=168, fourier_order=weekly_order)
	model.fit(history_df)
	future_ds = build_hourly_time_index(train_ds[-1] + pd.Timedelta(hours=1), horizon)
	future_df = pd.DataFrame({"ds": future_ds})
	forecast_df = model.predict(future_df)
	return forecast_df["yhat"].to_numpy()


def build_time_features(total_length: int) -> np.ndarray:
	# Hour-of-week cyclical encoding
	hours = np.arange(total_length)
	phase = 2.0 * np.pi * (hours % 168) / 168.0
	return np.stack([np.sin(phase), np.cos(phase)], axis=1)


def linear_ar_forecast(y_train: np.ndarray, horizon: int) -> np.ndarray:
	# Lags to use (ensure available)
	candidate_lags = [1, 2, 3, 4, 6, 12, 24, 168]
	usable_lags = [lag for lag in candidate_lags if lag < len(y_train)]
	if not usable_lags:
		raise ValueError("Not enough history for linear AR model.")
	max_lag = max(usable_lags)
	n = len(y_train)
	time_feats = build_time_features(n)
	rows = []
	targets = []
	for t in range(max_lag, n):
		lag_values = [y_train[t - lag] for lag in usable_lags]
		feat_row = np.concatenate([np.array(lag_values, dtype=float), time_feats[t]])
		rows.append(feat_row)
		targets.append(y_train[t])
	X = np.vstack(rows)
	y = np.array(targets, dtype=float)
	# Add bias term
	X_design = np.hstack([np.ones((X.shape[0], 1)), X])
	beta, *_ = np.linalg.lstsq(X_design, y, rcond=None)
	# Recursive forecast
	history = y_train.astype(float).tolist()
	preds: List[float] = []
	for h in range(horizon):
		t_idx = len(history)
		lag_values = [history[t_idx - lag] if t_idx - lag >= 0 else history[0] for lag in usable_lags]
		time_row = build_time_features(t_idx + 1)[-1]
		feat_row = np.concatenate([np.array(lag_values, dtype=float), time_row])
		x_vec = np.concatenate([np.array([1.0]), feat_row])
		y_hat = float(x_vec @ beta)
		preds.append(y_hat)
		history.append(y_hat)
	return np.array(preds, dtype=float)


def maybe_transform(series: np.ndarray, log1p: bool) -> np.ndarray:
	if log1p:
		return np.log1p(np.maximum(series, 0.0))
	return series


def maybe_inverse(series: np.ndarray, log1p: bool) -> np.ndarray:
	if log1p:
		return np.expm1(series)
	return series


def plot_hexbin(y_true: np.ndarray, y_pred: np.ndarray, title: str, gridsize: int, show: bool, save_path: str) -> None:
	if plt is None:
		print("matplotlib is not installed. Install with 'pip install matplotlib'.")
		return
	fig, ax = plt.subplots(figsize=(6, 6))
	hb = ax.hexbin(y_true, y_pred, gridsize=gridsize, cmap="viridis", mincnt=1)
	ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], color="red", linestyle="--", linewidth=1, label="y = x")
	ax.set_xlabel("Actual value")
	ax.set_ylabel("Predicted value")
	ax.set_title(title)
	ax.legend(loc="upper left")
	cb = fig.colorbar(hb, ax=ax)
	cb.set_label("Counts per hex")
	ax.set_aspect("equal", adjustable="box")
	fig.tight_layout()
	if save_path:
		fig.savefig(save_path, dpi=150)
	if show:
		plt.show()
	plt.close(fig)


def run(metric: str, beam: str, target_week: int, model_name: str, samples: int, use_log1p: bool, prophet_daily_order: int, prophet_weekly_order: int, prophet_changepoint_prior: float, prophet_seasonality_mode: str, hexbin_show: bool, hexbin_save: str, hexbin_gridsize: int) -> None:
	train, test6, test11 = read_split(metric)
	for df in [train, test6, test11]:
		validate_beam_column(df, beam)
	# Training series (0-839)
	y_train_raw = train[beam].to_numpy()
	# Targets
	if target_week == 6:
		y_true_raw = test6[beam].to_numpy()  # 168 points
	elif target_week == 11:
		y_true_raw = test11[beam].to_numpy()  # 168 points
	else:
		raise ValueError("target_week must be 6 or 11")
	# Optional transform
	y_train = maybe_transform(y_train_raw, use_log1p)
	# Forecast in transformed space if enabled
	if model_name == "seasonal_naive":
		y_pred_t = seasonal_naive_forecast(y_train, horizon=len(y_true_raw), period=SEASONAL_PERIOD)
	elif model_name == "prophet":
		y_pred_t = prophet_forecast(y_train, horizon=len(y_true_raw), daily_order=prophet_daily_order, weekly_order=prophet_weekly_order, changepoint_prior=prophet_changepoint_prior, seasonality_mode=prophet_seasonality_mode)
	elif model_name == "linear_ar":
		y_pred_t = linear_ar_forecast(y_train, horizon=len(y_true_raw))
	else:
		raise ValueError("Unknown model. Use 'seasonal_naive', 'prophet', or 'linear_ar'.")
	# Inverse transform to original space
	y_pred = maybe_inverse(y_pred_t, use_log1p)
	y_true = y_true_raw.astype(float)
	mae, mape, rmse = compute_metrics(y_true, y_pred)
	smape, median_ae = compute_additional_metrics(y_true, y_pred)
	print({
		"metric": metric,
		"beam": beam,
		"target_week": target_week,
		"model": model_name,
		"log1p": use_log1p,
		"horizon": len(y_true),
		"seasonal_period": SEASONAL_PERIOD if model_name == "seasonal_naive" else None,
		"MAE": round(mae, 4),
		"Median_AE": round(median_ae, 4),
		"MAPE_percent": round(mape, 4) if not np.isnan(mape) else None,
		"sMAPE_percent": round(smape, 4),
		"RMSE": round(rmse, 4),
	})
	if samples and samples > 0:
		samples = int(min(samples, len(y_true)))
		comp_df = pd.DataFrame({
			"t": np.arange(samples),
			"y_true": y_true[:samples],
			"y_pred": y_pred[:samples],
		})
		print("Samples (first N):")
		print(comp_df.to_string(index=False))
	# Hexbin visualization
	if hexbin_show or (hexbin_save and len(hexbin_save) > 0):
		title = f"Actual vs Predicted ({model_name}) - {metric} {beam} week {target_week}"
		plot_hexbin(y_true, y_pred, title=title, gridsize=hexbin_gridsize, show=hexbin_show, save_path=hexbin_save)


def parse_args(argv: List[str]) -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Forecast for Beam-Level dataset")
	parser.add_argument("--metric", choices=SUPPORTED_METRICS, default="DLPRB", help="Which KPI to forecast")
	parser.add_argument("--beam", type=str, default="0_0_1", help="Beam column id like '0_0_1'")
	parser.add_argument("--week", type=int, choices=[6, 11], default=6, help="Target week: 6 or 11")
	parser.add_argument("--model", choices=["seasonal_naive", "prophet", "linear_ar"], default="seasonal_naive", help="Which model to use")
	parser.add_argument("--samples", type=int, default=0, help="Print first N predicted vs actual samples")
	parser.add_argument("--log1p", action="store_true", help="Use log1p transform during modeling")
	# Prophet hyperparameters
	parser.add_argument("--prophet-daily-order", type=int, default=10, help="Fourier order for daily seasonality (0 to disable)")
	parser.add_argument("--prophet-weekly-order", type=int, default=10, help="Fourier order for weekly seasonality (0 to disable)")
	parser.add_argument("--prophet-changepoint-prior", type=float, default=0.05, help="Changepoint prior scale")
	parser.add_argument("--prophet-seasonality", choices=["additive", "multiplicative"], default="additive", help="Seasonality mode")
	# Hexbin options
	parser.add_argument("--hexbin-show", action="store_true", help="Display hexbin plot window")
	parser.add_argument("--hexbin-save", type=str, default="", help="Save hexbin plot to this file path (e.g., plot.png)")
	parser.add_argument("--hexbin-gridsize", type=int, default=30, help="Hexbin grid size (higher = finer)")
	return parser.parse_args(argv)


if __name__ == "__main__":
	args = parse_args(sys.argv[1:])
	run(
		metric=args.metric,
		beam=args.beam,
		target_week=args.week,
		model_name=args.model,
		samples=args.samples,
		use_log1p=args.log1p,
		prophet_daily_order=args.prophet_daily_order,
		prophet_weekly_order=args.prophet_weekly_order,
		prophet_changepoint_prior=args.prophet_changepoint_prior,
		prophet_seasonality_mode=args.prophet_seasonality,
		hexbin_show=args.hexbin_show,
		hexbin_save=args.hexbin_save,
		hexbin_gridsize=args.hexbin_gridsize,
	)
