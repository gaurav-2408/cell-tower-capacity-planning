import argparse
import os
from typing import List, Optional
import pandas as pd
from datetime import datetime

from seasonal_naive_forecast import run as run_model, SUPPORTED_METRICS


def discover_beams(metric: str, max_beams: Optional[int]) -> List[str]:
    """Read the training CSV for a metric and return a list of beam column names.

    Excludes the 'Time' column if present. Optionally limits to first N beams.
    """
    # Paths are relative to this file's directory
    here = os.path.dirname(__file__)
    train_path = os.path.join(here, "Beam-Level-Traffic-Timeseries-Dataset", "data", "train", f"{metric}_train_0w-5w.csv")
    df_head = pd.read_csv(train_path, nrows=1)
    # Exclude non-beam columns such as CSV indices and time columns
    cols = [
        c for c in df_head.columns
        if c.lower() != "time" and not c.lower().startswith("unnamed:")
    ]
    if max_beams is not None and max_beams > 0:
        cols = cols[:max_beams]
    return cols


def resolve_output_path(filename: str) -> str:
    """Resolve a filename to the CSV_Results_Analysis directory when no dir is provided."""
    if os.path.isabs(filename) or os.path.dirname(filename):
        return filename
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    out_dir = os.path.join(repo_root, "CSV_Results_Analysis")
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, filename)


def benchmark(metrics: List[str], weeks: List[int], beams: Optional[List[str]], max_beams: Optional[int],
              models: List[str], log1p: bool, samples: int, prophet_daily_order: int, prophet_weekly_order: int,
              prophet_changepoint_prior: float, prophet_seasonality: str, output_csv: str) -> str:
    all_results = []

    # If beams not provided, discover per-metric (may differ) and iterate
    for metric in metrics:
        if beams:
            candidate_beams = beams
        else:
            candidate_beams = discover_beams(metric, max_beams)

        for beam in candidate_beams:
            for week in weeks:
                for model in models:
                    res = run_model(
                        metric=metric,
                        beam=beam,
                        target_week=week,
                        model_name=model,
                        samples=samples,
                        use_log1p=log1p,
                        prophet_daily_order=prophet_daily_order,
                        prophet_weekly_order=prophet_weekly_order,
                        prophet_changepoint_prior=prophet_changepoint_prior,
                        prophet_seasonality_mode=prophet_seasonality,
                        hexbin_show=False,
                        hexbin_save="",
                        hexbin_gridsize=30,
                    )
                    all_results.append(res)

    df = pd.DataFrame(all_results)
    out_path = resolve_output_path(output_csv)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    try:
        df.to_csv(out_path, index=False)
        print(f"Saved benchmark results to {out_path}")
    except PermissionError:
        # Likely open in another program (e.g., Excel). Save to a timestamped file instead.
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        base, ext = os.path.splitext(out_path)
        alt_path = f"{base}_{ts}{ext or '.csv'}"
        df.to_csv(alt_path, index=False)
        print(f"Target file locked; saved benchmark results to {alt_path}")
        out_path = alt_path

    # Also emit aggregated summaries for quick comparison
    try:
        base_no_ext, _ = os.path.splitext(out_path)
        by_model_path = f"{base_no_ext}_summary_by_model.csv"
        by_metric_model_path = f"{base_no_ext}_summary_by_metric_model.csv"

        # Group means (lower is better for these metrics). Prefer robust metrics.
        metric_cols = [
            "MAE",
            "Median_AE",
            "RMSE",
            "WAPE_percent",
            "sWAPE_percent",
            "MASE",
        ]
        # Coerce to numeric in case of mixed types
        for c in metric_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        by_model = (
            df.groupby(["model"]) [metric_cols]
              .mean(numeric_only=True)
              .reset_index()
        )
        by_model.to_csv(by_model_path, index=False)

        if "metric" in df.columns:
            by_metric_model = (
                df.groupby(["metric", "model"]) [metric_cols]
                  .mean(numeric_only=True)
                  .reset_index()
            )
            by_metric_model.to_csv(by_metric_model_path, index=False)

        print(f"Saved summaries: {by_model_path} and {by_metric_model_path}")
    except Exception:
        # Don't fail the run if summary export has issues
        pass

    return out_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark models across metrics, beams, and weeks")
    p.add_argument("--metrics", nargs="*", default=SUPPORTED_METRICS, choices=SUPPORTED_METRICS,
                   help="Metrics to benchmark")
    p.add_argument("--weeks", nargs="*", type=int, default=[6, 11], choices=[6, 11],
                   help="Target weeks to evaluate")
    p.add_argument("--beams", nargs="*", default=None,
                   help="Specific beam IDs to evaluate (default: auto-discover)")
    p.add_argument("--max-beams", type=int, default=5,
                   help="If beams not provided, limit to first N discovered beams per metric (0=all)")
    p.add_argument("--models", nargs="*", default=["seasonal_naive", "prophet", "linear_ar"],
                   choices=["seasonal_naive", "prophet", "linear_ar"],
                   help="Models to benchmark")
    p.add_argument("--log1p", action="store_true", help="Use log1p transform")
    p.add_argument("--samples", type=int, default=0, help="Print first N samples for each run")
    # Prophet params
    p.add_argument("--prophet-daily-order", type=int, default=10)
    p.add_argument("--prophet-weekly-order", type=int, default=10)
    p.add_argument("--prophet-changepoint-prior", type=float, default=0.05)
    p.add_argument("--prophet-seasonality", choices=["additive", "multiplicative"], default="additive")
    p.add_argument("--output-csv", type=str, default="benchmark_results.csv",
                   help="Output CSV filename or path")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    benchmark(
        metrics=args.metrics,
        weeks=args.weeks,
        beams=args.beams,
        max_beams=None if args.max_beams == 0 else args.max_beams,
        models=args.models,
        log1p=args.log1p,
        samples=args.samples,
        prophet_daily_order=args.prophet_daily_order,
        prophet_weekly_order=args.prophet_weekly_order,
        prophet_changepoint_prior=args.prophet_changepoint_prior,
        prophet_seasonality=args.prophet_seasonality,
        output_csv=args.output_csv,
    )


