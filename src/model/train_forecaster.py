"""
Member 3: ML Specialist
Phase 2 – State-Level Forecasting (ETL v3.0 compatible)
"""

import json
import pandas as pd
import numpy as np
from prophet import Prophet
import joblib
from pathlib import Path

# ---------------- CONFIG ----------------
DATA_PATH = Path("src/etl_pipeline/processed_metrics.json")
OUTPUT_PATH = Path("src/model/load_forecast.pkl")


# ---------------- LOAD DATA ----------------
def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError("processed_metrics.json not found")

    with open(DATA_PATH, "r") as f:
        records = json.load(f)

    return pd.DataFrame(records)


# ---------------- SIMULATE HISTORY ----------------
def simulate_history(base_volume, months=6):
    """
    Simulate historical monthly mobile update volume
    so Prophet has sufficient time points.
    """
    dates = pd.date_range(end=pd.Timestamp.today(), periods=months, freq="M")
    noise = np.random.normal(0, base_volume * 0.05, months)
    values = np.maximum(base_volume + noise, 0)

    return pd.DataFrame({"ds": dates, "y": values})


# ---------------- STATE-LEVEL FORECAST ----------------
def generate_state_forecasts(df):
    forecasts = {}

    for state in sorted(df["State"].unique()):
        state_df = df[df["State"] == state]

        # ETL v3.0 provides this directly
        base_volume = state_df["mobile_update_volume"].sum()

        ts_df = simulate_history(base_volume)

        model = Prophet(
            yearly_seasonality=False,
            weekly_seasonality=False,
            daily_seasonality=False
        )
        model.fit(ts_df)

        future = model.make_future_dataframe(periods=3, freq="M")
        forecast = model.predict(future)

        forecasts[state] = (
            forecast.tail(3)["yhat"]
            .round()
            .astype(int)
            .tolist()
        )

    return forecasts


# ---------------- MAIN ----------------
def main():
    print("Loading normalized metrics (ETL v3.0)...")
    df = load_data()

    print("Generating state-level forecasts...")
    state_forecasts = generate_state_forecasts(df)

    print("Saving forecast dictionary...")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(state_forecasts, OUTPUT_PATH)

    print(f"✅ Forecasting complete for {len(state_forecasts)} states")


if __name__ == "__main__":
    main()
