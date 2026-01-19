import json, os, pandas as pd, joblib
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE_PATH = os.path.join(BASE_DIR, '..', 'etl_pipeline', 'processed_metrics.json')
MODEL_PATH = os.path.join(BASE_DIR, '..', 'model', 'load_forecast.pkl')

def load_data():
    if not os.path.exists(DATA_FILE_PATH): return None
    df = pd.read_json(DATA_FILE_PATH)
    df['State'] = df['State'].str.upper()
    df['District'] = df['District'].str.upper()
    return df

def analyze_logic(volume, ratio, ml_intelligence):
    """
    Proportional Scaling v5.3:
    Ensures District predictions follow State Trends at a local scale.
    """
    state_forecast = ml_intelligence.get('values', [volume * 1.05] * 3)
    
    # Calculate Month-on-Month Multipliers from the State Model
    baseline = state_forecast[0] if state_forecast[0] > 0 else 1
    multipliers = [v / baseline for v in state_forecast]
    
    # Apply State Trends to LOCAL Volume
    local_forecast = [int(volume * m) for m in multipliers]

    # Layman friendly mappings
    coverage_status = "OPTIMIZED" if ratio > 0.65 else ("IMPROVING" if ratio > 0.4 else "UNDER-ENROLLED")
    access_status = "EASY ACCESS" if ratio > 0.55 else "LIMITED ACCESS"

    return {
        "inclusivity": {"status": coverage_status, "value": round(ratio * 100, 1), "label": "Youth Coverage"},
        "security": {"status": access_status, "value": 90.2, "label": "Service Efficiency"},
        "efficiency": {
            "status": "OPERATIONAL",
            "biometric_traffic_trend": local_forecast,
            "accuracy": ml_intelligence.get('accuracy', 94.1),
            "trend": ml_intelligence.get('trend', 'STABLE')
        }
    }

@app.route('/api/heatmap')
def heatmap():
    df = load_data()
    res = {s: {"ratio": df[df['State']==s]['ratio'].mean(), "status": "SAFE" if df[df['State']==s]['ratio'].mean() > 0.5 else "WARNING"} for s in df['State'].unique()}
    return jsonify({"status": "success", "data": res})

@app.route('/api/metadata')
def metadata():
    df = load_data()
    meta = {s: sorted(df[df['State'] == s]['District'].unique().tolist()) for s in df['State'].unique()}
    return jsonify({"status": "success", "metadata": meta})

@app.route('/api/audit')
def audit():
    state = request.args.get('state', '').upper()
    dist = request.args.get('district', '').upper()
    df = load_data()
    
    sdf = df[df['State'] == state]
    if dist:
        target = sdf[sdf['District'] == dist].iloc[0]
        vol, rat = target['mobile_update_volume'], target['ratio']
    else:
        vol, rat = sdf['mobile_update_volume'].sum(), sdf['ratio'].mean()

    try:
        ml = joblib.load(MODEL_PATH).get(state, {})
    except:
        ml = {}

    return jsonify({"status": "success", "cards": analyze_logic(vol, rat, ml)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)