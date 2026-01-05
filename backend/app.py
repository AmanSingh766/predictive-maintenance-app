from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import os

app = Flask(
    __name__,
    static_folder="frontend",
    static_url_path=""
)
CORS(app)

print("\n" + "=" * 70)
print("🚀 INDUSTRIAL AI PREDICTIVE MAINTENANCE - BACKEND SERVER")
print("=" * 70)

# --------------------------------------------------
# Load ML artifacts
# --------------------------------------------------
print("\n🔄 Loading ML model and preprocessing objects...")
try:
    model = joblib.load('models/predictive_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    label_encoder = joblib.load('models/label_encoder.pkl')
    print("✅ Model, scaler, and label encoder loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model files: {e}")
    exit(1)

# --------------------------------------------------
# Load dataset
# --------------------------------------------------
print("📊 Loading dataset...")
try:
    df = pd.read_csv('../dataset/predictive_maintenance.csv')
    print(f"✅ Dataset loaded: {len(df)} machines")
except Exception as e:
    print(f"❌ Error loading dataset: {e}")
    exit(1)

# --------------------------------------------------
# FRONTEND (UI)
# --------------------------------------------------
@app.route('/')
def serve_frontend():
    return send_from_directory(app.static_folder, 'index.html')

# --------------------------------------------------
# PREDICTION API
# --------------------------------------------------
@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.json

        type_encoded = label_encoder.transform([data['type']])[0]

        features = np.array([[  
            float(data['air_temperature']),
            float(data['process_temperature']),
            float(data['rotational_speed']),
            float(data['torque']),
            float(data['tool_wear']),
            int(type_encoded)
        ]])

        features_scaled = scaler.transform(features)

        probabilities = model.predict_proba(features_scaled)[0]
        risk_score = probabilities[1] * 100

        if risk_score < 30:
            status = "healthy"
            recommendation = (
                "✅ Equipment is operating normally. "
                "Continue routine monitoring. Next inspection in 30 days."
            )
        elif risk_score < 60:
            status = "warning"
            recommendation = (
                "⚠️ Early warning signs detected. "
                "Schedule preventive maintenance within 7 days."
            )
        else:
            status = "critical"
            recommendation = (
                "🚨 High failure risk detected. "
                "Immediate maintenance required!"
            )

        response = {
            "status": status,
            "risk_score": round(float(risk_score), 2),
            "healthy_probability": round(float(probabilities[0]), 3),
            "failure_probability": round(float(probabilities[1]), 3),
            "recommendation": recommendation
        }

        print(f"✅ Prediction served | Risk: {risk_score:.2f}% | Status: {status}")
        return jsonify(response)

    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return jsonify({"error": str(e)}), 400

# --------------------------------------------------
# ANALYTICS API
# --------------------------------------------------
@app.route('/api/analytics', methods=['GET'])
def analytics():
    try:
        response = {
            "statistics": {
                "total_machines": len(df),
                "failure_rate": round(df['Machine_failure'].mean() * 100, 2),
                "avg_temperature": round(df['Air_temperature'].mean(), 2),
                "high_risk_machines": int(df['Machine_failure'].sum())
            },
            "failure_types": {
                "TWF": int(df['TWF'].sum()),
                "HDF": int(df['HDF'].sum()),
                "PWF": int(df['PWF'].sum()),
                "OSF": int(df['OSF'].sum()),
                "RNF": int(df['RNF'].sum())
            }
        }
        return jsonify(response)

    except Exception as e:
        print(f"❌ Analytics error: {e}")
        return jsonify({"error": str(e)}), 400

# --------------------------------------------------
# EQUIPMENT LIST API
# --------------------------------------------------
@app.route('/api/equipment-list', methods=['GET'])
def equipment_list():
    try:
        equipment = []

        df_encoded = df.copy()
        df_encoded['Type_Encoded'] = label_encoder.transform(df['Type'])

        for idx, row in df.iterrows():
            features = np.array([[  
                row['Air_temperature'],
                row['Process_temperature'],
                row['Rotational_speed'],
                row['Torque'],
                row['Tool_wear'],
                df_encoded.loc[idx, 'Type_Encoded']
            ]])

            features_scaled = scaler.transform(features)
            risk = model.predict_proba(features_scaled)[0][1] * 100

            status = "healthy" if risk < 30 else "warning" if risk < 60 else "critical"

            equipment.append({
                "product_id": row['Product_ID'],
                "type": row['Type'],
                "air_temperature": row['Air_temperature'],
                "process_temperature": row['Process_temperature'],
                "rotational_speed": row['Rotational_speed'],
                "torque": row['Torque'],
                "tool_wear": row['Tool_wear'],
                "risk_score": round(float(risk), 2),
                "status": status
            })

        return jsonify(equipment)

    except Exception as e:
        print(f"❌ Equipment list error: {e}")
        return jsonify({"error": str(e)}), 400

# --------------------------------------------------
# SERVER START
# --------------------------------------------------
if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("🌐 SERVER STARTING...")
    print("=" * 70)
    print("📡 Open UI: http://localhost:5000")
    print("📚 APIs: /api/predict | /api/analytics | /api/equipment-list")
    print("=" * 70 + "\n")

    app.run(debug=True, host='0.0.0.0', port=5000)
