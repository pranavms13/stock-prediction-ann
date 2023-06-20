import os
import sys
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras

def load_model(filename):
    model = keras.models.load_model(filename)
    return model

def predict_price(model, data):
    # Preprocess the data
    features = data[['Open', 'High', 'Low', 'Close', 'Volume']].values
    data['Datetime'] = pd.to_datetime(data['Datetime'], utc=True)

    time_steps = 1
    num_features = features.shape[1]
    features = np.reshape(features, (-1, time_steps, num_features))

    # Perform prediction
    predictions = model.predict(features)

    # Return predicted prices
    return predictions

def main():
    if len(sys.argv) < 3:
        print("Usage: python predict.py <model_keras_file> <input_csv_file>")
        print("Example: python predict.py models/msft.keras data/msft.csv")
        sys.exit(1)

    model_file = sys.argv[1]
    input_file = sys.argv[2]

    if (not os.path.exists(model_file)):
        print("Model - Keras file not found")
        sys.exit(1)

    if (not os.path.exists(input_file)):
        print("Input - CSV file not found")
        sys.exit(1)

    try:
        model = load_model(model_file)
        data = pd.read_csv(input_file)
        predictions = predict_price(model, data)
        print("Predictions:")
        print(predictions)
    except Exception as e:
        print("Error: ", e)
        sys.exit(1)

if __name__ == '__main__':
    main()