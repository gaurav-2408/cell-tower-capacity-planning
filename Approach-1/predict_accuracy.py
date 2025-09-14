import argparse
import pandas as pd

# Import the run function from seasonal_naive_forecast
from seasonal_naive_forecast import run


def evaluate_all(metric: str, beam: str, target_week: int, log1p: bool,
                 daily_order: int, weekly_order: int, changepoint_prior: float,
                 seasonality_mode: str, samples: int, save_csv: str):
    models = ["seasonal_naive", "prophet", "linear_ar"]
    results = []

    for model in models:
        print(f"\nRunning model: {model}")
        res = run(
            metric=metric,
            beam=beam,
            target_week=target_week,
            model_name=model,
            samples=samples,
            use_log1p=log1p,
            prophet_daily_order=daily_order,
            prophet_weekly_order=weekly_order,
            prophet_changepoint_prior=changepoint_prior,
            prophet_seasonality_mode=seasonality_mode,
            hexbin_show=False,
            hexbin_save="",
            hexbin_gridsize=30,
        )
        results.append(res)

    df_results = pd.DataFrame(results)
    print("\nModel Accuracy Comparison")
    print("=" * 40)
    print(df_results.to_string(index=False))

    if save_csv:
        df_results.to_csv(save_csv, index=False)
        print(f"\nResults saved to {save_csv}")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate accuracy of all models via run()")
    parser.add_argument("--metric", choices=["DLPRB", "DLThpVol", "DLThpTime", "MR_number"], default="DLPRB")
    parser.add_argument("--beam", type=str, default="0_0_1", help="Beam column id like '0_0_1'")
    parser.add_argument("--week", type=int, choices=[6, 11], default=6, help="Target week: 6 or 11")
    parser.add_argument("--log1p", action="store_true", help="Use log1p transform during modeling")
    parser.add_argument("--prophet-daily-order", type=int, default=10)
    parser.add_argument("--prophet-weekly-order", type=int, default=10)
    parser.add_argument("--prophet-changepoint-prior", type=float, default=0.05)
    parser.add_argument("--prophet-seasonality", choices=["additive", "multiplicative"], default="additive")
    parser.add_argument("--samples", type=int, default=0, help="Print first N predicted vs actual samples for each model")
    parser.add_argument("--save-csv", type=str, default="", help="File path to save results as CSV")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_all(
        metric=args.metric,
        beam=args.beam,
        target_week=args.week,
        log1p=args.log1p,
        daily_order=args.prophet_daily_order,
        weekly_order=args.prophet_weekly_order,
        changepoint_prior=args.prophet_changepoint_prior,
        seasonality_mode=args.prophet_seasonality,
        samples=args.samples,
        save_csv=args.save_csv,
    )