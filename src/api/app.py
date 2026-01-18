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
            'Mobile_Number_Updates': 'mobile_update_volume',
            'Gender_Female': 'female_count', # Common discrepancy
            'female_enrolment_pct': 'ratio', # If Member 2 sent the ratio directly
            'Total_Enrolment': 'total_enrolment'
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
    """Business Logic for Aadhaar Darpan Audit Cards"""
    # Security: Trigger Critical if volume is >15% above forecast baseline
    baseline = forecast_values[0] / 1.05
    status = "CRITICAL" if volume > (baseline * 1.15) else "SAFE"
    
    return {
        "security": {
            "status": status, 
            "mobile_update_volume": int(volume),
            "message": "High anomaly detected" if status == "CRITICAL" else "Normal traffic"
        },
        "inclusivity": {
            "status": "WARNING" if ratio < 0.45 else "SAFE", 
            "female_enrolment_pct": round(ratio, 4)
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
    target_district = request.args.get('district', '').strip().upper()
    
    df = load_data()
    if df is None: return jsonify({"status": "error", "message": "Data Not Ready"}), 503

    state_df = df[df['State'] == target_state]
    if state_df.empty:
        return jsonify({"status": "error", "message": f"State {target_state} not found"}), 404

    # --- INITIALIZE VARIABLES ---
    volume = 0
    ratio = 0.0
    loc = ""

    # EXPERT LOGIC: Aggregation vs Single Point Analysis
    if not target_district or target_district in ["ALL", "NONE", "ENTIRE STATE"]:
        # 1. SUM total volume for the state
        volume = int(state_df['mobile_update_volume'].sum())
        
        # 2. CALCULATE State-Wide Ratio
        # Since 'female_count' is missing, we use a weighted average of the 'ratio' column
        # Formula: Sum(ratio * total_enrolment) / Sum(total_enrolment)
        total_enrolled = state_df['total_enrolment'].sum()
        if total_enrolled > 0:
            # We reconstruct the female count mathematically to get the true weighted ratio
            weighted_females = (state_df['ratio'] * state_df['total_enrolment']).sum()
            ratio = float(weighted_females / total_enrolled)
        else:
            ratio = 0.0
            
        loc = f"ALL {target_state}"
    else:
        # 3. Targeted District Deep-Dive
        match = state_df[state_df['District'] == target_district]
        if match.empty: 
            return jsonify({"status": "error", "message": "District not found"}), 404
            
        volume = int(match.iloc[0]['mobile_update_volume'])
        ratio = float(match.iloc[0]['ratio']) # Use 'ratio' instead of 'female_enrolment_pct'
        loc = target_district

    # --- ML FORECAST HANDSHAKE ---
    try:
        forecast_dict = joblib.load(MODEL_PATH)
        forecast_values = forecast_dict.get(target_state, [int(volume*1.02), int(volume*1.05), int(volume*1.08)])
    except:
        forecast_values = [int(volume*1.02), int(volume*1.05), int(volume*1.08)]

    return jsonify({
        "status": "success", 
        "location": loc, 
        "cards": analyze_logic(volume, ratio, forecast_values)
    })

if __name__ == '__main__':
    # host='0.0.0.0' allows mobile device access on local network
    app.run(debug=True, host='0.0.0.0', port=5001)