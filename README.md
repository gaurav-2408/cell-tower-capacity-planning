# cell-tower-capacity-planning
Serverless cell tower capacity planning using data mining, forecasting, and geospatial visualization. Predicts usage trends, detects capacity breaches, and displays network load on interactive hexbin maps.

Tasks:
[x] - Initialized Github Projecct to track progress 
[x] - Data Set upload in git

Commands:
To visualize hexbin-
python seasonal_naive_forecast.py --metric DLPRB --beam 0_0_1 --week 6 --model linear_ar --hexbin-show