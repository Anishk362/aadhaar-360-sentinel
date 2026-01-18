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
import os

# ---------------- CONFIG ----------------
# Standardized paths for the Aadhaar Darpan ecosystem
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "etl_pipeline" / "processed_metrics.json"
OUTPUT_PATH = BASE_DIR / "model" / "load_forecast.pkl"

# ---------------- LOAD DATA ----------------
def load_data():
    """
    Loads processed metrics and applies bilingual schema mapping 
    to prevent KeyErrors.
    """
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"CRITICAL: {DATA_PATH} not found. Run ingest_data.py first.")

    with open(DATA_PATH, "r") as f:
        records = json.load(f)

    df = pd.DataFrame(records)
    
    # SCHEMA FIX: Ensures ML model finds volume column regardless of naming
    rename_map = {
        'Mobile_Number_Updates': 'mobile_update_volume'
    }
    df.rename(columns=rename_map, inplace=True)
    
    # Standardize State names for consistent dictionary keys
    df['State'] = df['State'].astype(str).str.strip().str.upper()
    
    return df

# ---------------- SIMULATE HISTORY ----------------
def simulate_history(base_volume, months=6):
    """
    Simulates historical monthly volume so Prophet has sufficient 
    data points for a 3-month forecast.
    """
    dates = pd.date_range(end=pd.Timestamp.today(), periods=months, freq="ME")
    # Add 5% variance to simulate real-world data noise
    noise = np.random.normal(0, base_volume * 0.05, months)
    values = np.maximum(base_volume + noise, 0)

    return pd.DataFrame({"ds": dates, "y": values})

# ---------------- STATE-LEVEL FORECAST ----------------
def generate_state_forecasts(df):
    """
    Generates a 3-month biometric traffic forecast for every 
    unique State found in the dataset.
    """
    forecasts = {}
    unique_states = sorted(df["State"].unique())

    print(f"Analyzing {len(unique_states)} states/UTs...")

    for state in unique_states:
        state_df = df[df["State"] == state]

        # Aggregate total volume for the state
        base_volume = state_df["mobile_update_volume"].sum()

        # Prepare data for Facebook Prophet
        ts_df = simulate_history(base_volume)

        model = Prophet(
            yearly_seasonality=False,
            weekly_seasonality=False,
            daily_seasonality=False,
            interval_width=0.95
        )
        
        # Suppress Prophet logs for cleaner terminal output
        model.fit(ts_df)

        # Predict next 3 months
        future = model.make_future_dataframe(periods=3, freq="ME")
        forecast = model.predict(future)

        # Convert forecast to a list of integers for the API/Mobile UI
        forecast_list = (
            forecast.tail(3)["yhat"]
            .round()
            .astype(int)
            .tolist()
        )
        
        forecasts[state] = forecast_list
        print(f"  > Forecast generated for: {state}")

    return forecasts

# ---------------- MAIN ----------------
def main():
    print("🚀 Initializing ML Forecasting Intelligence...")
    
    try:
        # 1. Load standardized data
        print(f"Loading normalized metrics from {DATA_PATH}...")
        df = load_data()

        # 2. Generate predictions for all 28 states + UTs
        print("Generating state-level forecasts using Prophet...")
        state_forecasts = generate_state_forecasts(df)

        # 3. Save the brain (forecast dictionary)
        print("Saving forecast dictionary to disk...")
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(state_forecasts, OUTPUT_PATH)

        print("-" * 40)
        print(f"✅ SUCCESS: Forecasting complete for {len(state_forecasts)} regions.")
        print(f"📂 Model saved at: {OUTPUT_PATH}")
        print("-" * 40)
        
    except Exception as e:
        print(f"❌ ML ENGINE ERROR: {str(e)}")

if __name__ == "__main__":
    main()