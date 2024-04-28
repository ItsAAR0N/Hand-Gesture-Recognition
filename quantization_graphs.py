import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow_model_optimization.quantization.keras import quantize_model
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import tempfile 
# from tensorflow.lite.experimental.new_converter import ExperimentalConverterFeatures


# def simulate_quantization(model, factor):
#     for layer in model.layers:
#         if 'conv' in layer.name or 'dense' in layer.name:
#             weights, biases = layer.get_weights()
#             # Simulate quantization by scaling weights
#             quantized_weights = np.round(weights * factor) / factor
#             layer.set_weights([quantized_weights, biases])
#     return model

# def convert_to_tflite_and_save(model, filename):
#     converter = tf.lite.TFLiteConverter.from_keras_model(model)
#     converter.optimizations = [tf.lite.Optimize.DEFAULT]
#     tflite_model = converter.convert()
#     with open(filename, 'wb') as f:
#         f.write(tflite_model)
#     return os.path.getsize(filename)

def quantize_and_get_size(model, optimizations, supported_types):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = optimizations
    converter.target_spec.supported_types = supported_types  # Set the supported types to float16
    tflite_model = converter.convert()
    tflite_model_file = tempfile.NamedTemporaryFile(delete=False)
    tflite_model_file.write(tflite_model)
    tflite_model_file.flush()
    model_size = os.path.getsize(tflite_model_file.name) / 1024  # Get the file size in kilobytes
    tflite_model_file.close()  # Close and delete the temp file
    os.remove(tflite_model_file.name)  # Ensure the file is deleted
    return model_size

def convert_to_tflite_and_save(model, filename):
    # Create a converter object from the model
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Convert the model without any optimizations
    tflite_model = converter.convert()
    
    # Write the converted model to a binary file
    with open(filename, 'wb') as f:
        f.write(tflite_model)
    
    # Return the size of the model file in kilobytes
    return os.path.getsize(filename) / 1024
# Define the number of classes
num_classes = 24  # Adjust as per your dataset

# Load data
train = pd.read_csv('sign_mnist_train/sign_mnist_train.csv')
test = pd.read_csv('sign_mnist_test/sign_mnist_test.csv')
labels = train['label'].values
unique_val = np.array(labels)
np.unique(unique_val)
train.drop('label', axis = 1, inplace = True)
images = train.values
images = np.array([np.reshape(i, (28, 28)) for i in images])
images = np.array([i.flatten() for i in images])

# One hot encode the labels
label_binarizer = LabelBinarizer()
labels = label_binarizer.fit_transform(labels)

# Split data
x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=101)
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# Define and train the original model
original_model = Sequential([
    Conv2D(64, kernel_size=(3,3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2,2)),
    Conv2D(64, kernel_size=(3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Conv2D(64, kernel_size=(3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.20),
    Dense(num_classes, activation='softmax')
])
original_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
original_model.fit(x_train, y_train, batch_size=32, epochs=1, validation_data=(x_test, y_test))
_, baseline_model_accuracy = original_model.evaluate(x_test, y_test, verbose=0)

# Make the model quantization aware, train, and evaluate
quant_aware_model = quantize_model(original_model)
quant_aware_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
quant_aware_model.fit(x_train, y_train, batch_size=32, epochs=1, validation_data=(x_test, y_test))
_, quant_aware_model_accuracy = quant_aware_model.evaluate(x_test, y_test, verbose=0)

# Path to save the TFLite models
path = "tflite_models/"
os.makedirs(path, exist_ok=True)

# Convert and save the original model to TFLite and get its size
original_tflite_size = convert_to_tflite_and_save(original_model, os.path.join(path, 'original_model.tflite'))
float32_quantized_size = quantize_and_get_size(original_model, [tf.lite.Optimize.DEFAULT], [tf.float32])
float16_quantized_size = quantize_and_get_size(original_model, [tf.lite.Optimize.DEFAULT], [tf.float16])
float8_quantized_size = quantize_and_get_size(original_model, [tf.lite.Optimize.DEFAULT], [])
non_quantized_tflite_size = convert_to_tflite_and_save(original_model, os.path.join(path, 'non_quantized_model.tflite'))
reduction_percentage_float16 = 100 * (non_quantized_tflite_size - float16_quantized_size) / non_quantized_tflite_size

print(float32_quantized_size)
print(float16_quantized_size)
print(float8_quantized_size)
print(original_tflite_size)

def reduc(base,quant):
    return 100 * (base - quant) / base
quant = [float8_quantized_size,float16_quantized_size,float32_quantized_size]
for i in range(0,3):
    quant[i] = reduc(original_tflite_size,quant[i])
indice = [8,16,32]

plt.figure(figsize=(8, 6))
plt.plot(indice, quant, marker='o', linestyle='-', color='b')
plt.xlabel('Number of bytes per weight')
plt.ylabel('Compression gain (%)')
plt.title('Compression gain vs. Number of bytes per weight')
plt.grid(True)
plt.show()
# Convert and save the quantization-aware model to TFLite and get its size
# quantized_tflite_size = quantize_and_get_size(quant_aware_model, [tf.lite.Optimize.DEFAULT])

# Factors to simulate different quantizations (not actual quantization levels)
# factors = [256, 128, 64, 32, 16]  # Simulating different levels of quantization

# Dictionary to hold the percentage reduction for each factor
# percentage_reductions = {}

# # Calculate and store sizes for simulated quantization at different levels
# for factor in factors:
#     # Simulate quantization and get the model size
#     simulated_model = simulate_quantization(tf.keras.models.clone_model(original_model), factor)
#     simulated_tflite_size = quantize_and_get_size(simulated_model, [])
#     reduction = 100 * (quantized_tflite_size - simulated_tflite_size) / quantized_tflite_size
#     percentage_reductions[factor] = reduction
#     print(f"Simulated quantization factor {factor}: model size {simulated_tflite_size} KB, reduction {reduction:.2f}%")

# Plot the percentage reductions
# print(f"Float16 quantized model size: {float16_quantized_size} KB")
# print(f"Default quantized model size: {non_quantized_tflite_size} KB")
# print(f"Reduction in model size (float16 compared to default quantization): {reduction_percentage_float16:.2f}%"