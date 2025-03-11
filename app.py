from flask import Flask, render_template, jsonify
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os
import json

app = Flask(__name__)

# Load the saved model and scaler
MODEL_PATH = r'C:\Users\User\bitcoin_model.keras'
SCALER_PATH = r'C:\Users\User\scaler.json'
DATA_PATH = r'C:\Users\User\DataFiles\btcusd_1min_data.csv'

def load_model_and_scaler(model_path, scaler_path):
    """Load the saved model and scaler"""
    try:
        model = tf.keras.models.load_model(model_path)
        
        with open(scaler_path, 'r') as f:
            scaler_params = json.load(f)
        
        scaler = MinMaxScaler()
        scaler.min_ = np.array(scaler_params['min_'])
        scaler.scale_ = np.array(scaler_params['scale_'])
        scaler.data_min_ = np.array(scaler_params['data_min_'])
        scaler.data_max_ = np.array(scaler_params['data_max_'])
        scaler.data_range_ = np.array(scaler_params['data_range_'])
        scaler.n_features_in_ = scaler_params['n_features_in_']
        scaler.n_samples_seen_ = scaler_params['n_samples_seen_']
        
        return model, scaler
    except Exception as e:
        raise Exception(f"Error loading model or scaler: {e}")

def load_recent_data(file_path, days=30):
    """Load recent data from CSV"""
    data = pd.read_csv(file_path)
    data[data.columns[0]] = pd.to_datetime(pd.to_numeric(data[data.columns[0]], errors='coerce'), unit='s')
    data.set_index(data.columns[0], inplace=True)
    data.dropna(inplace=True)
    end_date = data.index.max()
    start_date = end_date - pd.Timedelta(days=days)
    filtered_data = data.loc[(data.index >= start_date) & (data.index <= end_date)]
    return filtered_data

def create_sequences(data, seq_length=10):
    """Create sequences from data"""
    features = ['Open', 'High', 'Low', 'Close', 'Volume']
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[features])
    
    X = []
    for i in range(len(scaled_data) - seq_length):
        X.append(scaled_data[i:i+seq_length])
    
    return np.array(X), scaler

def predict_prices(model, last_sequence, scaler, steps=5):
    """Predict future prices"""
    current_seq = last_sequence.copy()
    predictions = []
    
    for _ in range(steps):
        current_reshape = current_seq.reshape(1, current_seq.shape[0], current_seq.shape[1])
        next_pred = model.predict(current_reshape, verbose=0)
        
        dummy_row = np.zeros((1, scaler.n_features_in_))
        dummy_row[0, 3] = next_pred[0, 0]  # Close price
        next_pred_actual = scaler.inverse_transform(dummy_row)[0, 3]
        predictions.append(next_pred_actual)
        
        current_seq = np.roll(current_seq, -1, axis=0)
        current_seq[-1, :] = current_seq[-2, :]
        current_seq[-1, 3] = next_pred[0, 0]
    
    return predictions

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/predict', methods=['GET'])
def get_predictions():
    """API endpoint to get price predictions"""
    try:
        # Load model and scaler
        model, scaler = load_model_and_scaler(MODEL_PATH, SCALER_PATH)
        
        # Load data and create sequences
        data = load_recent_data(DATA_PATH)
        X, _ = create_sequences(data)
        last_sequence = X[-1]
        
        # Get predictions
        predictions = predict_prices(model, last_sequence, scaler)
        
        # Format predictions for JSON response
        pred_dict = {f"t+{i+1}": f"${price:.2f}" for i, price in enumerate(predictions)}
        
        return jsonify({
            'status': 'success',
            'predictions': pred_dict,
            'last_date': str(data.index[-1])
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

if __name__ == '__main__':
    app.run(debug=True)