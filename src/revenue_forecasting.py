
import json
import joblib
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Project paths
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "data" / "processed" / "sales_cleaned.csv"
OUTPUT_DIR = ROOT_DIR / "data" / "processed"
MODEL_DIR = ROOT_DIR / "models"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# 1. Load cleaned sales data
sales = pd.read_csv(DATA_PATH, parse_dates=["OrderDate"])

# 2. Aggregate transaction revenue by month
sales["MonthStart"] = sales["OrderDate"].dt.to_period("M").dt.to_timestamp()

monthly = (
    sales.groupby("MonthStart", as_index=False)["Revenue"]
    .sum()
    .sort_values("MonthStart")
)

# Ensure every month exists in the time series
monthly = (
    monthly.set_index("MonthStart")
    .asfreq("MS", fill_value=0)
    .rename_axis("MonthStart")
    .reset_index()
)

monthly.to_csv(OUTPUT_DIR / "monthly_revenue.csv", index=False)

# 3. Create features using historical data only
monthly["Year"] = monthly["MonthStart"].dt.year
monthly["Month"] = monthly["MonthStart"].dt.month
monthly["MonthSin"] = np.sin(2 * np.pi * monthly["Month"] / 12)
monthly["MonthCos"] = np.cos(2 * np.pi * monthly["Month"] / 12)

monthly["Trend"] = np.arange(len(monthly))
monthly["Lag1"] = monthly["Revenue"].shift(1)
monthly["Lag2"] = monthly["Revenue"].shift(2)
monthly["Lag3"] = monthly["Revenue"].shift(3)
monthly["Lag12"] = monthly["Revenue"].shift(12)

monthly["RollingMean3"] = (
    monthly["Revenue"].shift(1).rolling(window=3).mean()
)

FEATURES = [
    "Year",
    "Month",
    "MonthSin",
    "MonthCos",
    "Trend",
    "Lag1",
    "Lag2",
    "Lag3",
    "Lag12",
    "RollingMean3",
]

model_data = monthly.dropna(subset=FEATURES).copy()

# 4. Split chronologically: earlier months for training,
#    final 12 months for testing
if len(model_data) < 24:
    raise ValueError(
        "Not enough monthly history. At least 24 usable months are required."
    )

split_index = len(model_data) - 12

train = model_data.iloc[:split_index]
test = model_data.iloc[split_index:]

X_train = train[FEATURES]
y_train = train["Revenue"]

X_test = test[FEATURES]
y_test = test["Revenue"]

# 5. Train the Random Forest model
model = RandomForestRegressor(
    n_estimators=300,
    max_depth=6,
    min_samples_leaf=2,
    random_state=42,
)

model.fit(X_train, y_train)

# 6. Predict revenue for the test period
predictions = model.predict(X_test)
predictions = np.maximum(predictions, 0)

# 7. Compare against a seasonal-naive baseline:
#    predict each month using revenue from 12 months earlier
baseline_predictions = test["Lag12"].to_numpy()

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

baseline_mae = mean_absolute_error(y_test, baseline_predictions)
baseline_rmse = np.sqrt(
    mean_squared_error(y_test, baseline_predictions)
)

# 8. Save test results
evaluation = pd.DataFrame({
    "MonthStart": test["MonthStart"].to_numpy(),
    "ActualRevenue": y_test.to_numpy(),
    "PredictedRevenue": predictions,
    "BaselineRevenue": baseline_predictions,
})

evaluation.to_csv(
    OUTPUT_DIR / "monthly_forecast_evaluation.csv",
    index=False,
)

# 9. Save evaluation metrics
metrics = {
    "model": "RandomForestRegressor",
    "test_months": int(len(test)),
    "mae": float(mae),
    "rmse": float(rmse),
    "r2": float(r2),
    "baseline_mae": float(baseline_mae),
    "baseline_rmse": float(baseline_rmse),
}

with open(MODEL_DIR / "forecast_metrics.json", "w") as file:
    json.dump(metrics, file, indent=4)

# 10. Retrain on all available data for future forecasting
final_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=6,
    min_samples_leaf=2,
    random_state=42,
)

final_model.fit(model_data[FEATURES], model_data["Revenue"])

joblib.dump(
    {"model": final_model, "features": FEATURES},
    MODEL_DIR / "revenue_forecast_model.pkl",
)

# 11. Forecast the next 12 months recursively
history = monthly["Revenue"].astype(float).tolist()
last_month = monthly["MonthStart"].max()
future_rows = []

for step in range(1, 13):
    future_date = last_month + pd.DateOffset(months=step)
    month_number = future_date.month
    trend = len(history)

    feature_row = pd.DataFrame([{
        "Year": future_date.year,
        "Month": month_number,
        "MonthSin": np.sin(2 * np.pi * month_number / 12),
        "MonthCos": np.cos(2 * np.pi * month_number / 12),
        "Trend": trend,
        "Lag1": history[-1],
        "Lag2": history[-2],
        "Lag3": history[-3],
        "Lag12": history[-12],
        "RollingMean3": np.mean(history[-3:]),
    }])[FEATURES]

    forecast = max(0.0, float(final_model.predict(feature_row)[0]))

    future_rows.append({
        "MonthStart": future_date,
        "PredictedRevenue": forecast,
    })

    # Use this prediction as history for later forecast months
    history.append(forecast)

future_forecast = pd.DataFrame(future_rows)
future_forecast.to_csv(
    OUTPUT_DIR / "future_revenue_forecast.csv",
    index=False,
)

# 12. Display results
print("\nREVENUE FORECASTING RESULTS")
print("-" * 40)
print(f"Training months: {len(train)}")
print(f"Testing months: {len(test)}")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R2 Score: {r2:.4f}")

print("\nBASELINE COMPARISON")
print(f"Baseline MAE: {baseline_mae:.2f}")
print(f"Baseline RMSE: {baseline_rmse:.2f}")

print("\nNEXT 12 MONTHS FORECAST")
print(future_forecast.to_string(index=False))

print("\nFiles saved successfully.")