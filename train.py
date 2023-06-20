import sys
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras

def train_model(data):
    # Extract features and target variable
    # data.fillna(data.mean(), inplace=True)

    features = data[['Open', 'High', 'Low', 'Close', 'Volume']].values
    data['Datetime'] = pd.to_datetime(data['Datetime'], utc=True)

    target = data['Adj Close'].values.reshape(-1, 1)

    time_steps = 1  # Number of time steps to consider
    num_features = features.shape[1]
    features = np.reshape(features, (-1, time_steps, num_features))

    # Normalize feature data if necessary
    # features = (features - np.mean(features, axis=0)) / np.std(features, axis=0)

    # Split data into training and testing sets
    train_ratio = 0.8
    train_size = int(train_ratio * len(data))
    features_train, features_test = features[:train_size], features[train_size:]
    target_train, target_test = target[:train_size], target[train_size:]


    # Create the model
    model = keras.models.Sequential()
    model.add(keras.layers.LSTM(64, input_shape=(time_steps, num_features)))
    model.add(keras.layers.Dense(1))

    # Compile the model
    model.compile(optimizer=keras.optimizers.Adam(), loss='mse')

    # Train the model
    batch_size = 32
    epochs = 10
    model.fit(features_train, target_train, batch_size=batch_size, epochs=epochs, validation_data=(features_test, target_test))

    return model

def save_model(model, filename):
    model.save(filename)
    print("Model saved to disk.")

def main():
    if len(sys.argv) < 3:
        print("Usage: python train.py <input_csv_file> <model_keras_file>")
        print("Example: python train.py data/msft.csv models/msft.keras")
        sys.exit(1)

    input_file = sys.argv[1]
    model_file = sys.argv[2]

    if(not os.path.exists(input_file)):
        print("Input - CSV file not found")
        sys.exit(1);
    
    if(not os.path.exists(model_file)):
        print("Model - Keras file not found")
        sys.exit(1);
    
    try:
        data = pd.read_csv(input_file)
        model = train_model(data)
        save_model(model, model_file)
    except Exception as e:
        print("Error: ", e)
        sys.exit(1)

if __name__ == '__main__':
    main()
