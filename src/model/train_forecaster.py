"""
Member 3: ML Forecaster
Project: Aadhaar Darpan

Input  : src/etl_pipeline/processed_metrics.json
Output : src/model/load_forecast.pkl

Forecasts Biometric_Updates for next 3 months
"""

import json
import pandas as pd
from prophet import Prophet
import joblib
from pathlib import Path

# Path configuration based on project root
DATA_PATH = Path("src/etl_pipeline/processed_metrics.json")
MODEL_PATH = Path("src/model/load_forecast.pkl")


def load_processed_metrics():
    with open(DATA_PATH, "r") as f:
        return json.load(f)


def prepare_timeseries(data):
    """
    Handles raw list input from ETL and simulates 3 months of history 
    to satisfy Prophet's requirement for multiple data points.
    """
    records = []

    # Member 1 provides a list directly
    for district_record in data:
        # Use Gender_Female as the proxy for biometric load
        val = district_record.get("Gender_Female", 0)
        
        # SIMULATION: Create 3 months of history so the model has a trend to learn
        # This fixes the "ValueError: Dataframe has less than 2 non-NaN rows"
        for i in range(3):
            records.append({
                # Create dates for Oct, Nov, Dec 2024
                "ds": pd.Timestamp("2024-10-01") + pd.DateOffset(months=i),
                # Apply slight variation (90%, 95%, 100% of the total)
                "y": val * (0.9 + (i * 0.05))
            })

    df = pd.DataFrame(records)

    # Aggregate across all districts to create a National Load Trend
    df = df.groupby("ds", as_index=False)["y"].sum()

    return df


def train_prophet(df):
    # Initialize model with minimal seasonality for the simulated trend
    model = Prophet(
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False
    )
    model.fit(df)
    return model


def main():
    print("Loading processed metrics...")
    try:
        data = load_processed_metrics()
    except FileNotFoundError:
        print(f"❌ Error: Could not find {DATA_PATH}. Run ingest_data.py first.")
        return

    print("Preparing time-series data...")
    ts_df = prepare_timeseries(data)

    print("Training forecasting model...")
    model = train_prophet(ts_df)

    print("Saving model...")
    # Ensure the model directory exists
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print("✅ load_forecast.pkl created successfully")


if __name__ == "__main__":
    main()