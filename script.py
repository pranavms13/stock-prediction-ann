import sys
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM

def train_model(data):
    # Extract time and price columns
    time = data['time'].values.reshape(-1, 1)
    price = data['price'].values.reshape(-1, 1)

    # Normalize price data if necessary
    # price = (price - np.mean(price)) / np.std(price)

    # Split data into training and testing sets
    train_ratio = 0.8
    train_size = int(train_ratio * len(data))
    time_train, time_test = time[:train_size], time[train_size:]
    price_train, price_test = price[:train_size], price[train_size:]

    # Create the model
    model = Sequential()
    model.add(LSTM(64, input_shape=(1,)))
    model.add(Dense(1))

    # Compile the model
    model.compile(optimizer='adam', loss='mse')

    # Train the model
    batch_size = 32
    epochs = 10
    model.fit(time_train, price_train, batch_size=batch_size, epochs=epochs, validation_data=(time_test, price_test))

    return model

def save_model(model, filename):
    model.save(filename)
    print("Model saved to disk.")

def load_saved_model(filename):
    loaded_model = load_model(filename)
    print("Model loaded from disk.")
    return loaded_model

def main():
    if len(sys.argv) < 4:
        print("Usage: python script.py <command> <input_file> <model_file>")
        print("Commands: train, load")
        sys.exit(1)

    command = sys.argv[1]
    input_file = sys.argv[2]
    model_file = sys.argv[3]

    if command == 'train':
        # Read time series data from CSV
        data = pd.read_csv(input_file)
        model = train_model(data)
        save_model(model, model_file)
    elif command == 'load':
        loaded_model = load_saved_model(model_file)
        # Use the loaded model for prediction or other tasks
    else:
        print("Invalid command")
        sys.exit(1)

if __name__ == '__main__':
    main()
