import os
import numpy as np
import tensorflow as tf
import tempfile 
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow_model_optimization.quantization.keras import quantize_model
import pandas as pd


# Set the path where the model is saved
path = "saved_models/sign_mnist_cnn_10_epochs.h5"  # 
model = tf.keras.models.load_model(path)

threshold = 0.05  # Define the threshold for zeroing weights

# Iterate over each layer in the model
for layer in model.layers:
    weights = layer.get_weights()  # Get current weights of the layer
    new_weights = []
    for w in weights:
        # Apply thresholding to zero out small weights
        modified_w = np.where(np.abs(w) < threshold, 0, w)
        new_weights.append(modified_w)
    layer.set_weights(new_weights)  # Set the modified weights back to the layer


def quantize_and_save_model(model, optimizations, supported_types, file_path):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = optimizations
    converter.target_spec.supported_types = supported_types  # Set the supported types (e.g., tf.float16)
    tflite_model = converter.convert()

    # Write the TFLite model to a file specified by file_path
    with open(file_path, 'wb') as f:
        f.write(tflite_model)

    # Optionally return the file size in kilobytes
    model_size = os.path.getsize(file_path) / 1024
    return model_size

# Example usage
file_path = 'compressed/test.tflite'
model_size = quantize_and_save_model(model, [tf.lite.Optimize.DEFAULT], [tf.float16], file_path)
print("Model size:", model_size, "KB")


