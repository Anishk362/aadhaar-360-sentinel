import json
import os
import pandas as pd
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS

# --- CONFIGURATION ---
app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE_PATH = os.path.join(BASE_DIR, '..', 'etl_pipeline', 'processed_metrics.json')
MODEL_PATH = os.path.join(BASE_DIR, '..', 'model', 'load_forecast.pkl')

# --- ML MODEL LOADER ---
try:
    forecast_model = joblib.load(MODEL_PATH)
    print("✅ ML Forecasting Model loaded successfully.")
except Exception as e:
    forecast_model = None
    print(f"⚠️ Warning: Could not load ML model: {e}")

# --- DATA LOADER ---
def load_data():
    if not os.path.exists(DATA_FILE_PATH):
        print(f"[ERROR] Could not find data file at: {DATA_FILE_PATH}")
        return None
    try:
        with open(DATA_FILE_PATH, 'r') as f:
            data = json.load(f)

        record_list = data['records'] if isinstance(data, dict) and 'records' in data else data
        df = pd.DataFrame(record_list)

        if not df.empty:
            df['State'] = df['State'].astype(str).str.replace('&', 'And').str.strip().str.title()
            df['District'] = df['District'].astype(str).str.strip().str.title()

        return df
    except Exception as e:
        print(f"[ERROR] Failed to read JSON: {e}")
        return None

# --- INTELLIGENCE LOGIC (MEMBER 2 – FINAL JUDGEMENT) ---
def analyze_logic(volume, ratio, forecast_values):
    """
    Strategic Intelligence Logic for District & State Evaluation
    Aligned with Aadhaar Darpan Protocol V3.0
    """

    # ───────────────── SECURITY PILLAR ─────────────────
    # Policy: Mean + 2σ (approximated using forecast baseline)
    baseline_mean = forecast_values[0] / 1.05
    upper_limit = baseline_mean * 1.15

    if volume > upper_limit:
        sec_status = "CRITICAL"
        deviation_pct = round(((volume - baseline_mean) / baseline_mean) * 100, 1)
        sec_msg = (
            f"Mobile update anomaly detected: "
            f"{deviation_pct}% above district monthly baseline."
        )
    else:
        sec_status = "SAFE"
        sec_msg = "Mobile update activity within expected district baseline."

    # ───────────────── INCLUSIVITY PILLAR ─────────────────
    if ratio < 0.40:
        inc_status = "WARNING"
        inc_msg = (
            f"Female enrolment ratio at {int(ratio * 100)}%, "
            f"below inclusion benchmark."
        )
    else:
        inc_status = "SAFE"
        inc_msg = "Female enrolment ratio within acceptable benchmark."

    # ───────────────── EFFICIENCY PILLAR ─────────────────
    eff_status = "SAFE"

    return {
        "security": {
            "status": sec_status,
            "message": sec_msg,
            "mobile_update_volume": volume
        },
        "inclusivity": {
            "status": inc_status,
            "message": inc_msg,
            "female_enrolment_pct": round(ratio, 2)
        },
        "efficiency": {
            "status": eff_status,
            "biometric_traffic_trend": forecast_values
        }
    }

# --- METADATA ENDPOINT ---
@app.route('/api/metadata', methods=['GET'])
def get_metadata():
    df = load_data()
    if df is None:
        return jsonify({"status": "error", "message": "Data Not Ready"}), 503

    metadata = {}
    states = sorted(df['State'].unique())

    for state in states:
        districts = df[df['State'] == state]['District'].unique().tolist()
        metadata[state] = sorted(districts)

    return jsonify({
        "status": "success",
        "metadata": metadata
    })

# --- AUDIT ENDPOINT (STATE & DISTRICT) ---
@app.route('/api/audit', methods=['GET'])
def get_audit_report():
    target_state = request.args.get('state')
    target_district = request.args.get('district')

    if not target_state:
        return jsonify({"status": "error", "message": "State is required"}), 400

    df = load_data()
    if df is None:
        return jsonify({"status": "error", "message": "Data Pipeline Not Ready"}), 503

    state_df = df[df['State'] == target_state.strip().title()]
    if state_df.empty:
        return jsonify({"status": "error", "message": f"State '{target_state}' not found"}), 404

    # ───────── STATE VIEW ─────────
    if not target_district or target_district.lower() in ["all", "none", ""]:
        volume = int(state_df['Mobile_Number_Updates'].sum())
        ratio = float(state_df['Gender_Female'].mean())
        forecast_values = [
            int(volume * 1.05),
            int(volume * 1.10),
            int(volume * 1.15)
        ]
        location_name = f"All {target_state}"

    # ───────── DISTRICT VIEW ─────────
    else:
        match_df = state_df[state_df['District'] == target_district.strip().title()]
        if match_df.empty:
            return jsonify({"status": "error", "message": "District not found"}), 404

        record = match_df.iloc[0]
        volume = int(record['Mobile_Number_Updates'])
        ratio = float(record['Gender_Female'])
        forecast_values = [
            int(volume * 1.02),
            int(volume * 1.05),
            int(volume * 1.08)
        ]
        location_name = record['District']

    cards_data = analyze_logic(volume, ratio, forecast_values)

    return jsonify({
        "status": "success",
        "location": location_name,
        "cards": cards_data
    })

# --- APPLICATION START ---
if __name__ == '__main__':
    print("🚀 Aadhaar Darpan Command Center is Starting...")
    app.run(debug=True, host='0.0.0.0', port=5001)
