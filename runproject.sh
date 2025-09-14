#!/bin/bash

# FOR seasonal_naive_forecast.py
echo "=== Running seasonal_naive_forecast.py commands ==="
# Navigate to the correct directory
cd Approach-1

# Basic run with default parameters (seasonal_naive model)
python seasonal_naive_forecast.py

# Run different models
python seasonal_naive_forecast.py --model prophet
python seasonal_naive_forecast.py --model linear_ar

# Run with different metrics
python seasonal_naive_forecast.py --metric DLThpVol
python seasonal_naive_forecast.py --metric DLThpTime
python seasonal_naive_forecast.py --metric MR_number

# Show sample predictions
python seasonal_naive_forecast.py --samples 20

# Use log transformation
python seasonal_naive_forecast.py --log1p

# Custom Prophet parameters
python seasonal_naive_forecast.py --model prophet --prophet-daily-order 15 --prophet-weekly-order 5

# Generate hexbin plots
python seasonal_naive_forecast.py --hexbin-show
python seasonal_naive_forecast.py --hexbin-save plot.png

# Full custom run
python seasonal_naive_forecast.py --metric DLThpVol --beam 1_2_3 --week 11 --model prophet --log1p --samples 20 --hexbin-save output.png

echo "=== Running predict_accuracy.py commands ==="
# FOR predict_accuracy.py
# Navigate to the correct directory (already in Approach-1 from above)

# Compare all models with default settings
python predict_accuracy.py

# Compare all models and save results
python predict_accuracy.py --save-csv comparison_results.csv

# Compare all models for different metric
python predict_accuracy.py --metric DLThpVol --save-csv dlthpvol_results.csv

# Additional useful commands for predict_accuracy.py
python predict_accuracy.py --metric DLThpTime --save-csv dlthptime_results.csv
python predict_accuracy.py --metric MR_number --save-csv mr_number_results.csv
python predict_accuracy.py --week 11 --save-csv week11_results.csv
python predict_accuracy.py --log1p --save-csv log_transformed_results.csv