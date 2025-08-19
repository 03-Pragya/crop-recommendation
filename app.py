# app.py - Production-ready Flask app for Render.com
import joblib
import pandas as pd
import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configuration
app.config['ENV'] = 'production'
app.config['DEBUG'] = False

# Load models and data
try:
    # Load the best trained model
    best_model = joblib.load("best_model.pkl")
    
    # Load the fitted scaler
    scaler = joblib.load("scaler.pkl")
    
    # Load the mapping of numerical codes back to crop names
    targets_df = pd.read_csv("targets.csv")
    targets_dict = dict(zip(targets_df['code'], targets_df['crop']))
    
    print("✅ Models and files loaded successfully!")
    
except FileNotFoundError as e:
    print(f"❌ Error: Model files not found - {e}")
    print("Please ensure 'best_model.pkl', 'scaler.pkl', and 'targets.csv' are in the same directory")

# Feature columns used by the model
feature_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']

@app.route('/')
def home():
    """Serve the main HTML page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    API endpoint for crop prediction
    Receives JSON data and returns prediction
    """
    try:
        # Get JSON data from request
        data = request.get_json(force=True)
        
        # Validate that all required features are present
        for feature in feature_cols:
            if feature not in data:
                return jsonify({
                    "success": False, 
                    "error": f"Missing required field: {feature}"
                }), 400
        
        # Convert to DataFrame for prediction
        user_data = pd.DataFrame([data], columns=feature_cols)
        
        # Convert string values to float if needed
        for col in feature_cols:
            user_data[col] = pd.to_numeric(user_data[col], errors='coerce')
        
        # Check for any NaN values after conversion
        if user_data.isnull().any().any():
            return jsonify({
                "success": False, 
                "error": "Invalid numeric values provided"
            }), 400
        
        # Determine if the best model needs scaled data
        model_name = type(best_model).__name__
        
        if model_name in ["KNeighborsClassifier", "SVC"]:
            # Scale the data for KNN and SVM
            scaled_data = scaler.transform(user_data)
            prediction_code = best_model.predict(scaled_data)[0]
        else:
            # Tree-based models don't need scaling
            prediction_code = best_model.predict(user_data)
        
        # Get the crop name from the prediction code
        predicted_crop = targets_dict.get(prediction_code, "Unknown")
        
        # Return successful prediction
        return jsonify({
            "success": True,
            "predicted_crop": predicted_crop,
            "model_used": model_name
        })
        
    except Exception as e:
        # Handle any unexpected errors
        return jsonify({
            "success": False, 
            "error": f"Prediction failed: {str(e)}"
        }), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "model_loaded": 'best_model' in globals()})

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({"success": False, "error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"success": False, "error": "Internal server error"}), 500

if __name__ == '__main__':
    # For local development only
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
