import json, os, pandas as pd, joblib
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# --- DYNAMIC PATH CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Points to etl_pipeline folder for JSON and model folder for PKL
DATA_FILE_PATH = os.path.join(BASE_DIR, '..', 'etl_pipeline', 'processed_metrics.json')
MODEL_PATH = os.path.join(BASE_DIR, '..', 'model', 'load_forecast.pkl')

def load_data():
    if not os.path.exists(DATA_FILE_PATH):
        return None
    try:
        with open(DATA_FILE_PATH, 'r') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        
        # SCHEMA ALIGNMENT: Maps various ETL versions to a standard API naming
        rename_map = {
            'mobile_update_volume': 'mobile_update_volume',
            'child_saturation_index': 'ratio', # <--- Ensure this matches the JSON key
            'total_enrolment': 'total_enrolment'
        }
        df.rename(columns=rename_map, inplace=True)
        
        # LOGGING FOR DEBUGGING: See what columns actually exist
        print("Available Columns after mapping:", df.columns.tolist())
        
        # Standardize state/district for matching
        df['State'] = df['State'].astype(str).str.strip().str.upper()
        df['District'] = df['District'].astype(str).str.strip().str.upper()
        return df
    except Exception as e:
        print(f"Load Error: {e}")
        return None

def analyze_logic(volume, ratio, forecast_values):
    """v4.0 Audit Logic: National Service Access & Saturation"""
    
    # Pillar 1: Child Saturation (Generation Gap)
    if ratio < 0.50:
        saturation_status = "CRITICAL"
    elif ratio < 0.70:
        saturation_status = "WARNING"
    else:
        saturation_status = "SAFE"

    # Pillar 2: Service Access Risk (Update velocity relative to activity)
    # Logic: High volume is GOOD here, low volume means citizens are excluded.
    baseline = sum(forecast_values) / len(forecast_values)
    total_activity = volume + baseline
    update_pct = (volume / total_activity) if total_activity > 0 else 0
    
    if update_pct < 0.75:
        access_status = "CRITICAL"
    elif update_pct < 0.85:
        access_status = "WARNING"
    else:
        access_status = "SAFE"

    return {
        "inclusivity": {
            "status": saturation_status, 
            "value": round(ratio, 4)
        },
        "security": {
            "status": access_status, 
            "value": round(update_pct * 100, 2)
        },
        "efficiency": {
            "status": "SAFE", 
            "biometric_traffic_trend": forecast_values
        }
    }

@app.route('/')
def home():
    return jsonify({"status": "online", "message": "Aadhaar Darpan v3.4 Final - Stable"})

@app.route('/api/metadata', methods=['GET'])
def get_metadata():
    df = load_data()
    if df is None: return jsonify({"status": "error", "message": "Data Not Ready"}), 503
    
    metadata = {}
    for state in sorted(df['State'].unique()):
        districts = df[df['State'] == state]['District'].unique().tolist()
        metadata[state] = sorted(districts)
    return jsonify({"status": "success", "metadata": metadata})

@app.route('/api/audit', methods=['GET'])
def get_audit_report():
    target_state = request.args.get('state', '').strip().upper()
    df = load_data()
    
    state_df = df[df['State'] == target_state]
    
    # BLOCKBUSTER MATH: Weighted State Analysis
    total_pop = state_df['total_enrolment'].sum()
    weighted_youth = (state_df['ratio'] * state_df['total_enrolment']).sum()
    
    # Weighted Ratio = Sum(District_Ratio * District_Pop) / Total_State_Pop
    state_weighted_ratio = weighted_youth / total_pop if total_pop > 0 else 0
    total_volume = state_df['mobile_update_volume'].sum()

    # Handshake with ML
    forecast_values = joblib.load(MODEL_PATH).get(target_state, [0,0,0])

    return jsonify({
        "status": "success",
        "location": target_state,
        "cards": analyze_logic(total_volume, state_weighted_ratio, forecast_values)
    })

if __name__ == '__main__':
    # host='0.0.0.0' allows mobile device access on local network
    app.run(debug=True, host='0.0.0.0', port=5001)