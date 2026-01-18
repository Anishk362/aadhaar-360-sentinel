import json, os, pandas as pd, joblib
from flask import Flask, request, jsonify
from flask_cors import CORS

# --- CONFIGURATION ---
app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE_PATH = os.path.join(BASE_DIR, '..', 'etl_pipeline', 'processed_metrics.json')

# --- DATA LOADER ---
def load_data():
    if not os.path.exists(DATA_FILE_PATH):
        return None
    try:
        with open(DATA_FILE_PATH, 'r') as f:
            data = json.load(f)
        record_list = data['records'] if isinstance(data, dict) and 'records' in data else data
        return pd.DataFrame(record_list)
    except Exception:
        return None

# --- INTELLIGENCE LOGIC (V3.0 PROTOCOL) ---
def analyze_logic(volume, ratio, forecast_values):
    baseline_mean = forecast_values[0] / 1.05
    upper_limit = baseline_mean * 1.15
    sec_status = "CRITICAL" if volume > upper_limit else "SAFE"
    return {
        "security": {
            "status": sec_status,
            "message": f"Anomaly: {round(((volume-baseline_mean)/baseline_mean)*100, 1)}% above baseline." if sec_status == "CRITICAL" else "Normal activity.",
            "mobile_update_volume": volume
        },
        "inclusivity": {
            "status": "WARNING" if ratio < 0.40 else "SAFE",
            "female_enrolment_pct": round(ratio, 2)
        },
        "efficiency": {
            "status": "SAFE",
            "biometric_traffic_trend": forecast_values
        }
    }

# --- 1. NEW HOME DASHBOARD (FIXES 404 ON ROOT) ---
@app.route('/')
def home():
    """Provides a status check when clicking the terminal link."""
    return jsonify({
        "status": "online",
        "system": "Aadhaar Darpan API",
        "version": "3.0",
        "active_port": 5001,
        "endpoints": {
            "metadata": "/api/metadata",
            "audit": "/api/audit?state=NAME&district=NAME"
        }
    })

# --- 2. METADATA ENDPOINT ---
@app.route('/api/metadata', methods=['GET'])
def get_metadata():
    df = load_data()
    if df is None: return jsonify({"status": "error", "message": "Data Not Ready"}), 503
    
    metadata = {}
    for state in sorted(df['State'].unique()):
        districts = df[df['State'] == state]['District'].unique().tolist()
        metadata[state] = sorted(districts)
    return jsonify({"status": "success", "metadata": metadata})

# --- 3. AUDIT ENDPOINT ---
@app.route('/api/audit', methods=['GET'])
def get_audit_report():
    target_state = request.args.get('state', '').strip().title().replace('&', 'And')
    target_district = request.args.get('district', '').strip().title().replace('&', 'And')
    
    df = load_data()
    if df is None: return jsonify({"status": "error"}), 503

    state_df = df[df['State'] == target_state]
    if state_df.empty: return jsonify({"status": "error", "message": "State not found"}), 404

    if not target_district or target_district.lower() in ["all", "none", ""]:
        volume = int(state_df['mobile_update_volume'].sum())
        ratio = float(state_df['female_enrolment_pct'].mean())
        forecast_values = [int(volume * 1.05), int(volume * 1.1), int(volume * 1.15)]
        loc = f"All {target_state}"
    else:
        match = state_df[state_df['District'] == target_district]
        if match.empty: return jsonify({"status": "error", "message": "District not found"}), 404
        volume = int(match.iloc[0]['mobile_update_volume'])
        ratio = float(match.iloc[0]['female_enrolment_pct'])
        forecast_values = [int(volume * 1.02), int(volume * 1.05), int(volume * 1.08)]
        loc = target_district

    return jsonify({"status": "success", "location": loc, "cards": analyze_logic(volume, ratio, forecast_values)})

if __name__ == '__main__':
    # Start on Port 5001 as per Member 2's latest logic
    app.run(debug=True, host='0.0.0.0', port=5001)