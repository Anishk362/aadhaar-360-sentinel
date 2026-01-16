import json
import os
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

# --- CONFIGURATION ---
app = Flask(__name__)
CORS(app)

# Logic to find the JSON file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE_PATH = os.path.join(BASE_DIR, '..', 'etl_pipeline', 'processed_metrics.json')

# --- DATA LOADER ---
def load_data():
    if not os.path.exists(DATA_FILE_PATH):
        print(f"[ERROR] Could not find data file at: {DATA_FILE_PATH}")
        return None
    try:
        # 1. Read file as a standard JSON dictionary first
        with open(DATA_FILE_PATH, 'r') as f:
            data = json.load(f)
        
        # 2. Extract ONLY the list inside "records"
        # If 'records' key exists, use it. Otherwise assume the file IS the list.
        if isinstance(data, dict) and 'records' in data:
            record_list = data['records']
        else:
            record_list = data # Fallback if structure changes
            
        # 3. Convert that list to a DataFrame
        return pd.DataFrame(record_list)
        
    except Exception as e:
        print(f"[ERROR] Failed to read JSON: {e}")
        return None

# --- INTELLIGENCE LOGIC ---
def analyze_district(record):
    # Member 1 used "Mobile_Number_Updates" (Capitalized), so we match that
    # We use .get() to avoid crashing if a key is missing
    metrics = record  # In Member 1's file, the metrics are at the root level, not inside 'metrics'
    
    # 1. SECURITY LOGIC
    # Notice the Capitals: Mobile_Number_Updates
    mobile_updates = metrics.get('Mobile_Number_Updates', 0)
    
    if mobile_updates > 1000:
        sec_status = "CRITICAL"
        sec_msg = f"High Anomaly: {mobile_updates} updates detected."
    else:
        sec_status = "SAFE"
        sec_msg = "Normal activity."

    # 2. INCLUSIVITY LOGIC
    female = metrics.get('Gender_Female', 0)
    total = metrics.get('Total_Enrolment', 1)
    ratio = female / total if total > 0 else 0
    
    if ratio < 0.40:
        inc_status = "WARNING"
        inc_msg = f"Low Female Enrolment ({int(ratio*100)}%)"
    else:
        inc_status = "SAFE"
        inc_msg = "Gender Ratio Healthy."

    # 3. EFFICIENCY LOGIC
    history = metrics.get('Biometric_Updates', []) # Changed to match JSON key
    eff_status = "SAFE"

    return {
        "security": { "status": sec_status, "message": sec_msg, "value": mobile_updates },
        "inclusivity": { "status": inc_status, "message": inc_msg, "value": ratio },
        "efficiency": { "status": eff_status, "forecast": history }
    }

# --- API ENDPOINT ---
@app.route('/api/audit', methods=['GET'])
def get_audit_report():
    target_district = request.args.get('district')
    
    if not target_district:
        return jsonify({"status": "error", "message": "Please provide a district name"}), 400

    df = load_data()
    if df is None:
        return jsonify({"status": "error", "message": "Data Pipeline Not Ready"}), 503

    # --- THE FIX IS HERE ---
    all_records = df.to_dict('records')
    
    # We look for "District" (Capital D) to match Member 1's file
    # We use .get("District", "") so it doesn't crash if the key is missing
    match = next((item for item in all_records if item.get("District", "").lower() == target_district.lower()), None)

    if not match:
        return jsonify({"status": "error", "message": "District not found"}), 404

    cards_data = analyze_district(match)

    return jsonify({
        "status": "success",
        "district": match.get('District'), # Capital D
        "state": match.get('State'),       # Capital S
        "cards": cards_data
    })

if __name__ == '__main__':
    print("🚀 Aadhaar Darpan Backend is Starting...")
    app.run(debug=True, host='0.0.0.0', port=5000)