import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Load and preprocess the data
def load_and_preprocess_data(file_path):
    # Read the CSV file - fix the file path syntax
    data = pd.read_csv(file_path)  # Remove the hardcoded path
    
    # Convert timestamps
    data[data.columns[0]] = pd.to_numeric(data[data.columns[0]], errors='coerce')
    data[data.columns[0]] = pd.to_datetime(data[data.columns[0]], unit='s')
    data.set_index(data.columns[0], inplace=True)
    
    # Remove any rows with NaN
    data.dropna(inplace=True)
    
    return data

# Create sequences for time series prediction
def create_sequences(data, seq_length):
    # Select features for prediction (typically Open, High, Low, Close, Volume)
    features = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    # Ensure all required columns exist
    missing_columns = [col for col in features if col not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")
    
    # Normalize the features
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[features])
    
    # Create sequences
    X, y = [], []
    for i in range(len(scaled_data) - seq_length):
        X.append(scaled_data[i:i+seq_length])
        y.append(scaled_data[i+seq_length, 3])  # Predicting next Close price
    
    return np.array(X), np.array(y), scaler

# Build LSTM Model for Price Prediction
def build_lstm_model(seq_length, feature_count):
    model = tf.keras.Sequential([
        # First LSTM layer with return sequences for deeper network
        tf.keras.layers.LSTM(50, activation='relu', input_shape=(seq_length, feature_count), return_sequences=True),
        
        # Dropout to prevent overfitting
        tf.keras.layers.Dropout(0.2),
        
        # Second LSTM layer
        tf.keras.layers.LSTM(50, activation='relu'),
        
        # Dropout layer
        tf.keras.layers.Dropout(0.2),
        
        # Dense layers for final prediction
        tf.keras.layers.Dense(25, activation='relu'),
        tf.keras.layers.Dense(1)  # Output layer for price prediction
    ])
    
    # Compile the model
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    return model

# Main training and evaluation function
def train_bitcoin_prediction_model(file_path, seq_length=10, epochs=50, batch_size=32):
    # Load data
    data = load_and_preprocess_data(file_path)
    
    # Create sequences
    X, y, scaler = create_sequences(data, seq_length)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Build model
    model = build_lstm_model(seq_length, X.shape[2])
    
    # Train model
    history = model.fit(
        X_train, y_train, 
        epochs=epochs, 
        batch_size=batch_size, 
        validation_split=0.2,
        verbose=1
    )
    
    # Evaluate model
    loss, mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {loss}")
    print(f"Test MAE: {mae}")
    
    # Visualize training history
    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1,2,2)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Model Mean Absolute Error')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    return model, scaler

# Prediction function
def predict_next_prices(model, last_sequence, scaler, steps=5):
    current_seq = last_sequence.copy()
    predictions = []
    
    for _ in range(steps):
        # Predict next price
        next_pred = model.predict(current_seq.reshape(1, current_seq.shape[0], current_seq.shape[1]))
        
        # Inverse transform to get actual price
        next_pred_actual = scaler.inverse_transform(
            np.concatenate([
                np.zeros((1, 4)),  # Placeholder for Open, High, Low
                next_pred,  # Predicted Close
                np.zeros((1, 1))  # Placeholder for Volume
            ], axis=1)
        )[0, 3]
        
        predictions.append(next_pred_actual)
        
        # Update sequence
        current_seq = np.roll(current_seq, -1, axis=0)
        current_seq[-1, :, 3] = next_pred  # Update Close price
    
    return predictions

# Main execution
if __name__ == "__main__":
    file_path = r'C:\Users\User\DataFiles\btcusd_1min_data.csv'
    
    try:
        # Train the model
        model, scaler = train_bitcoin_prediction_model(file_path)
        
        # Load data to get the last sequence for prediction
        data = load_and_preprocess_data(file_path)
        
        # Create sequences
        X, _, _ = create_sequences(data, seq_length=10)
        
        # Use the last sequence for prediction
        last_known_sequence = X[-1]
        
        # Predict next prices
        future_prices = predict_next_prices(model, last_known_sequence, scaler)
        print("Predicted Future Prices:", future_prices)
        
    except Exception as e:
        print(f"An error occurred: {e}")