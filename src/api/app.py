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
        
        # Standardize to DataFrame
        record_list = data['records'] if isinstance(data, dict) and 'records' in data else data 
        df = pd.DataFrame(record_list)
        
        # MEMBER 4 FIX: Ensure Title Case for robust dropdown matching
        if not df.empty:
            df['State'] = df['State'].astype(str).str.strip().str.title()
            df['District'] = df['District'].astype(str).str.strip().str.title()
        return df
    except Exception as e:
        print(f"[ERROR] Failed to read JSON: {e}")
        return None

# --- INTELLIGENCE LOGIC (SCHEMA ALIGNED) ---
def analyze_logic(volume, ratio, forecast_values):
    """
    Standardized logic for both District and State levels.
    Aligned with Section 3 of schema.txt.
    """
    # 1. SECURITY PILLAR (mobile_update_volume)
    sec_status = "CRITICAL" if volume > 1000 else "SAFE"
    sec_msg = f"High Anomaly: {volume} updates." if sec_status == "CRITICAL" else "Normal activity."

    # 2. INCLUSIVITY PILLAR (female_enrolment_pct)
    # Ratio expected as 0.0 to 1.0
    inc_status = "WARNING" if ratio < 0.40 else "SAFE"
    inc_msg = f"Low Female Enrolment ({int(ratio*100)}%)" if inc_status == "WARNING" else "Gender Ratio Healthy."

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
            "status": "SAFE", 
            "biometric_traffic_trend": forecast_values 
        }
    }

# --- REVISED: METADATA ENDPOINT (FIXES DATA SPLITTING) ---
@app.route('/api/metadata', methods=['GET'])
def get_metadata():
    df = load_data()
    if df is None:
        return jsonify({"status": "error", "message": "Data Not Ready"}), 503

    # CRITICAL FIX: Standardize names before grouping to merge duplicates
    df['State'] = df['State'].str.replace('&', 'And', regex=False).str.strip().str.title()
    df['District'] = df['District'].str.strip().str.title()

    metadata = {}
    states = sorted(df['State'].unique())
    
    for state in states:
        # Collects districts from ALL records that now match this normalized state name
        districts = df[df['State'] == state]['District'].unique().tolist()
        metadata[state] = sorted(districts)
        
    return jsonify({
        "status": "success",
        "metadata": metadata
    })

# --- REFACTORED: AUDIT ENDPOINT (STATE & DISTRICT) ---
@app.route('/api/audit', methods=['GET'])
def get_audit_report():
    target_state = request.args.get('state')
    target_district = request.args.get('district')
    
    if not target_state:
        return jsonify({"status": "error", "message": "State is required"}), 400

    df = load_data()
    if df is None:
        return jsonify({"status": "error", "message": "Data Pipeline Not Ready"}), 503

    # DATA CLEANING LAYER: Match the logic used in /api/metadata
    df['State'] = df['State'].str.replace('&', 'And', regex=False).str.strip().str.title()
    df['District'] = df['District'].str.strip().str.title()

    # Now filter the cleaned data
    state_df = df[df['State'] == target_state.strip().title()]
    
    if state_df.empty:
        return jsonify({"status": "error", "message": f"State '{target_state}' not found"}), 404

    # AGGREGATION LOGIC
    if not target_district or target_district.lower() in ["all", "none", ""]:
        # Logic for the State Map: Summing across all merged records
        volume = int(state_df['Mobile_Number_Updates'].sum())
        ratio = float(state_df['Gender_Female'].mean()) 
        forecast_values = [int(volume * 1.05), int(volume * 1.1), int(volume * 1.15)]
        location_name = f"All {target_state}"
    else:
        # Logic for the District View
        match_df = state_df[state_df['District'] == target_district.strip().title()]
        if match_df.empty:
            return jsonify({"status": "error", "message": "District not found"}), 404
        
        record = match_df.iloc[0]
        volume = int(record['Mobile_Number_Updates'])
        ratio = float(record['Gender_Female'])
        forecast_values = [int(volume * 1.02), int(volume * 1.05), int(volume * 1.08)]
        location_name = record['District']

    cards_data = analyze_logic(volume, ratio, forecast_values)
    return jsonify({"status": "success", "location": location_name, "cards": cards_data})

if __name__ == '__main__':
    print("🚀 Aadhaar Darpan Command Center is Starting...")
    # Port 5000 is for the Flutter App (Member 5) to connect to
    app.run(debug=True, host='0.0.0.0', port=5000)