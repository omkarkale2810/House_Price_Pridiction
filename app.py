from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

# Load the model, scaler, and feature names
model = joblib.load('house_price_gbr_model.pkl')
scaler = joblib.load('house_price_scaler.pkl')
feature_names = joblib.load('feature_names.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from request
        if request.is_json:
            # If API call with JSON data
            data = request.get_json()
            input_df = pd.DataFrame(data, index=[0])
        else:
            # If form submission
            input_data = {}
            for feature in request.form:
                value = request.form[feature]
                # Convert numeric values
                try:
                    value = float(value)
                except ValueError:
                    pass
                input_data[feature] = value
            input_df = pd.DataFrame([input_data])
        
        # Create DataFrame with all required features
        full_input = pd.DataFrame(columns=feature_names)
        
        # Fill in provided values
        for feature in feature_names:
            if feature in input_df.columns:
                full_input[feature] = input_df[feature]
            else:
                full_input[feature] = 0  # Default value
        
        # Scale the input data
        input_scaled = scaler.transform(full_input)
        
        # Make prediction
        prediction = model.predict(input_scaled)
        
        # Format the result
        result = {'predicted_price': float(prediction[0])}
        
        # Return based on request type
        if request.is_json:
            return jsonify(result)
        else:
            return render_template('result.html', prediction=result['predicted_price'])
    
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """Endpoint for API calls"""
    try:
        data = request.get_json()
        
        # Convert to DataFrame
        if isinstance(data, list):
            input_df = pd.DataFrame(data)
        else:
            input_df = pd.DataFrame([data])
        
        # Create DataFrame with all required features
        full_input = pd.DataFrame(columns=feature_names)
        
        # Fill in provided values
        for feature in feature_names:
            if feature in input_df.columns:
                full_input[feature] = input_df[feature]
            else:
                full_input[feature] = 0  # Default value
        
        # Scale the data
        input_scaled = scaler.transform(full_input)
        
        # Make prediction
        predictions = model.predict(input_scaled)
        
        # Return results
        results = [{'predicted_price': float(price)} for price in predictions]
        return jsonify(results if len(results) > 1 else results[0])
    
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/features')
def get_features():
    """Get top features for the form"""
    try:
        # Load feature importances and return top features
        importances = pd.DataFrame(
            {'feature': feature_names, 
             'importance': model.feature_importances_}
        ).sort_values('importance', ascending=False)
        
        top_features = importances.head(10)['feature'].tolist()
        return jsonify({'top_features': top_features})
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)