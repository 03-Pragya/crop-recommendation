import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# Waitress for Windows, fallback to Flask server otherwise
try:
    from waitress import serve
    WAITRESS_AVAILABLE = True
except ImportError:
    WAITRESS_AVAILABLE = False

app = Flask(__name__)
CORS(app)

# Load model, scaler, and targets
best_model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")
targets_df = pd.read_csv("targets.csv")
targets_dict = dict(zip(targets_df['code'], targets_df['crop']))

feature_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        for feature in feature_cols:
            if feature not in data:
                return jsonify({"success": False, "error": f"Missing feature: {feature}"}), 400
        
        user_data = pd.DataFrame([data], columns=feature_cols)
        for col in feature_cols:
            user_data[col] = pd.to_numeric(user_data[col], errors='coerce')
        
        if user_data.isnull().any().any():
            return jsonify({"success": False, "error": "Invalid numeric values"}), 400
        
        model_name = type(best_model).__name__
        if model_name in ["KNeighborsClassifier", "SVC"]:
            scaled_data = scaler.transform(user_data)
            prediction_code = best_model.predict(scaled_data)[0]
        else:
            prediction_code = best_model.predict(user_data)
        
        predicted_crop = targets_dict.get(prediction_code, "Unknown")
        return jsonify({"success": True, "predicted_crop": predicted_crop, "model_used": model_name})
    
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == '__main__':
    HOST = '0.0.0.0'
    PORT = int(os.environ.get('PORT', 5000))
    
    if os.name == 'nt' and WAITRESS_AVAILABLE:
        # Use Waitress server on Windows
        print("Running with Waitress on Windows")
        serve(app, host=HOST, port=PORT)
    else:
        # Use Flask built-in server otherwise (Linux or if Waitress missing)
        app.run(host=HOST, port=PORT, debug=True)

