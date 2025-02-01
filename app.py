from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import pickle
import joblib
from datetime import datetime

app = Flask(__name__)

# Load the trained model and scaler
def load_model():
    try:
        model = pickle.load(open('saved_models/optimized_xgboost.pkl', 'rb'))
        scaler = joblib.load('saved_models/scaler.pkl')
        return model, scaler
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None, None

model, scaler = load_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from request
        data = request.json
        
        # Create feature vector
        features = []
        # Add V1-V28
        for i in range(1, 29):
            features.append(data[f'V{i}'])
        # Add Amount and Time
        features.extend([data['Amount'], data['Time']])
        
        # Convert to DataFrame and scale
        features_df = pd.DataFrame([features], 
                                 columns=[f'V{i+1}' for i in range(28)] + ['Amount', 'Time'])
        scaled_features = scaler.transform(features_df)
        
        # Make prediction
        prediction = int(model.predict(scaled_features)[0])
        probability = float(model.predict_proba(scaled_features)[0][1])
        
        return jsonify({
            'prediction': prediction,
            'probability': probability,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=False)
