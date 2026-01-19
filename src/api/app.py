import json, os, pandas as pd, joblib
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# --- DYNAMIC PATH CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Points to etl_pipeline folder for the Sanitized JSON Source of Truth
DATA_FILE_PATH = os.path.join(BASE_DIR, '..', 'etl_pipeline', 'processed_metrics.json')
# Points to model folder for the Prophet Forecaster
MODEL_PATH = os.path.join(BASE_DIR, '..', 'model', 'load_forecast.pkl')

def load_data():
    """Loads sanitized metrics with case-insensitive standardization."""
    if not os.path.exists(DATA_FILE_PATH): 
        return None
    try:
        with open(DATA_FILE_PATH, 'r') as f: 
            data = json.load(f)
        df = pd.DataFrame(data)
        
        # Standardize for exact matching regardless of JSON case
        df['State'] = df['State'].astype(str).str.strip().str.upper()
        df['District'] = df['District'].astype(str).str.strip().str.upper()
        return df
    except Exception as e:
        print(f"❌ API Load Error: {e}")
        return None

def analyze_logic(volume, ratio, forecast_values):
    """
    Pillar Logic Engine:
    - Inclusivity: Based on Generation Saturation Index
    - Security: Based on Service Access Risk (Update Velocity)
    """
    # Pillar 1: Generation Saturation (Youth vs Adult ratio)
    sat_status = "CRITICAL" if ratio < 0.5 else ("WARNING" if ratio < 0.7 else "SAFE")
    
    # Pillar 2: Service Access Risk (Update velocity relative to baseline)
    avg_forecast = sum(forecast_values) / len(forecast_values) if forecast_values else 0
    total_activity = volume + avg_forecast
    velocity = (volume / total_activity) if total_activity > 0 else 0
    
    # Low velocity implies a 'Digital Desert' where citizens can't access updates
    acc_status = "CRITICAL" if velocity < 0.75 else ("WARNING" if velocity < 0.85 else "SAFE")
    
    return {
        "inclusivity": {
            "status": sat_status, 
            "value": round(ratio, 4)
        },
        "security": {
            "status": acc_status, 
            "value": round(velocity * 100, 2)
        },
        "efficiency": {
            "status": "SAFE", 
            "biometric_traffic_trend": forecast_values
        }
    }

@app.route('/api/heatmap', methods=['GET'])
def get_national_heatmap():
    df = load_data()
    if df is None: return jsonify({"status": "error"}), 503
    
    heatmap_data = {}
    states = df['State'].unique()
    
    for state in states:
        state_df = df[df['State'] == state]
        pop = state_df['total_enrolment'].sum()
        # Calculate Weighted Ratio
        ratio = (state_df['ratio'] * state_df['total_enrolment']).sum() / pop if pop > 0 else 0
        
        # Determine Color Coding
        status = "CRITICAL" if ratio < 0.5 else ("WARNING" if ratio < 0.7 else "SAFE")
        heatmap_data[state] = {
            "ratio": round(ratio, 2),
            "status": status
        }
    
    return jsonify({"status": "success", "data": heatmap_data})

@app.route('/api/metadata', methods=['GET'])
def get_metadata():
    """Returns the dynamic State-District hierarchy for the Flutter dropdowns."""
    df = load_data()
    if df is None: 
        return jsonify({"status": "error", "message": "Data Not Found"}), 503
    
    # Build clean metadata dictionary for the 36 Official Entities
    meta = {
        s: sorted(df[df['State'] == s]['District'].unique().tolist()) 
        for s in sorted(df['State'].unique())
    }
    return jsonify({"status": "success", "metadata": meta})

@app.route('/api/audit', methods=['GET'])
def get_audit_report():
    """
    Core Intelligence Endpoint:
    - If district is empty: Calculates Population-Weighted State Analysis.
    - If district is provided: Performs Precise District Drilldown.
    """
    t_state = request.args.get('state', '').strip().upper()
    t_dist = request.args.get('district', '').strip().upper()
    
    df = load_data()
    if df is None: 
        return jsonify({"status": "error", "message": "Backend Sync Error"}), 503

    # Filter by the canonical state name
    state_df = df[df['State'] == t_state]
    if state_df.empty: 
        return jsonify({"status": "error", "message": "State not found"}), 404

    if not t_dist or t_dist == "":
        # SCENARIO A: State-Level Overview (Logical Combination of all Districts)
        # Logic: We use population-weighted math to ensure statistical honesty
        total_pop = state_df['total_enrolment'].sum()
        
        # Weighted Ratio = Sum(District_Ratio * District_Population) / Total_State_Population
        weighted_youth = (state_df['ratio'] * state_df['total_enrolment']).sum()
        final_ratio = weighted_youth / total_pop if total_pop > 0 else 0
        
        final_volume = int(state_df['mobile_update_volume'].sum())
        location_label = t_state
    else:
        # SCENARIO B: Specific District Drilldown (Readjusts everything to this district)
        dist_df = state_df[state_df['District'] == t_dist]
        if dist_df.empty: 
            return jsonify({"status": "error", "message": f"District {t_dist} not found"}), 404
        
        final_ratio = float(dist_df.iloc[0]['ratio'])
        final_volume = int(dist_df.iloc[0]['mobile_update_volume'])
        location_label = t_dist

    # --- ML Handshake ---
    # Retrieve forecast values. If specific district is chosen, 
    # the values are state-scoped but reflect the specific regional trend.
    try:
        model_data = joblib.load(MODEL_PATH)
        forecast_values = model_data.get(t_state, [int(final_volume * 1.1)] * 3)
    except:
        # Fallback if model is not yet trained
        forecast_values = [int(final_volume * 1.05)] * 3

    return jsonify({
        "status": "success", 
        "location": location_label, 
        "cards": analyze_logic(final_volume, final_ratio, forecast_values)
    })

if __name__ == '__main__':
    # Listen on all interfaces to support Flutter mobile device testing
    app.run(debug=True, host='0.0.0.0', port=5001)