# ELEC4342 Hand Gesture Recognition Project
# Baseline CNN model definition
# University of Hong Kong, Author: Aaron Shek
# Last Edited: 05/05/24

# Import Libraries
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns # Visualization Library
from IPython.display import display, Image
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization, Flatten, Activation
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam, AdamW
from sklearn.metrics import accuracy_score
import argparse
import cv2
import time
import os

def parse_arguments() -> argparse.Namespace:
    '''
    Add arguments.
    '''
    parser = argparse.ArgumentParser(
        description='Hand Gesture Recognition using LeNet-5'
    )
    parser.add_argument('--train_path', default='./handgestures_lenet5_cnn/train', 
                        type=str, required=False, help = 'Path to training dataset folder'),
    parser.add_argument('--test_path', default= './handgestures_lenet5_cnn/test', 
                        type=str, required=False, help = 'Path to testing dataset folder')                            
    return parser.parse_args()

def plot_training(history):
    # Plot training & validation accuracy values
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('MODEL ACCURACY\n({})'.format(model_info))
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.grid(True)
    plt.show()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('MODEL LOSS\n({})'.format(model_info))
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    args = parse_arguments()
    start_time = time.time()  # Record start time

    # Number of classes 
    num_classes = 3

    # Image dimensions
    img_rows, img_cols = 32, 32 # LeNet-5 needs 32x32

    # Batch size for training
    batch_size = 16 # Adjust hyper-parameter

    # Directory paths for training and validation data
    train_data_dir = args.train_path
    validation_data_dir = args.test_path  # Corrected the directory path

    ####### Apply Data Augmentation for training data #######
    train_datagen = ImageDataGenerator(
        rescale=1./255,  # Rescale pixel values to [0, 1]
        rotation_range=30,  # Randomly rotate images by 30 degrees
        width_shift_range=0.3,  # Randomly shift images horizontally by 30%
        height_shift_range=0.3,  # Randomly shift images vertically by 30%
        horizontal_flip=True,  # Randomly flip images horizontally
        fill_mode='nearest',  # Strategy for filling in newly created pixels
    )

    # Data augmentation not needed on validation, apart from rescaling
    validation_datagen = ImageDataGenerator(rescale=1./255, # Scale between [0, 1]
                                            ) 

    # Generate batches of augmented data for training
    train_generator = train_datagen.flow_from_directory(
        train_data_dir, 
        target_size=(img_rows, img_cols),  # Resize images to match model input size
        batch_size=batch_size, 
        color_mode='grayscale',  # Convert images to grayscale
        class_mode='categorical' # Categorical labels 
    )

    # Generate batches of augmented data for validation
    validation_generator = validation_datagen.flow_from_directory(
        validation_data_dir, 
        target_size=(img_rows, img_cols), 
        batch_size=batch_size, 
        color_mode='grayscale', 
        class_mode='categorical'
    )

    # Number of images in the training dataset
    num_train_images = train_generator.n
    print("Number of images in the training dataset:", num_train_images)

    # Number of images in the validation dataset
    num_validation_images = validation_generator.n
    print("Number of images in the validation dataset:", num_validation_images)
    
    ####### LeNet-5 Model ########
    model = Sequential([
            Conv2D(6, kernel_size=(5, 5), strides=(1, 1), activation='relu', input_shape=(img_rows, img_cols, 1)),
            MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
            Conv2D(16, kernel_size=(5, 5), strides=(1, 1), activation='relu'),
            MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
            Flatten(),
            Dense(120, activation='relu'),
            Dense(84, activation='relu'),
            Dense(num_classes, activation='softmax')
        ])

    print(model.summary()) # ~ 63,203 parameters
    learning_rate = 0.001
    ######### Training #########
    # Use a very small learning rate
    # Compile the model with binary cross-entropy loss, RMSprop optimizer, and accuracy metrics
    model.compile(loss='categorical_crossentropy',
                optimizer=Adam(learning_rate=learning_rate), # Adjustable hyperparameter # Adam W (weight_decay = 0.005)
                metrics=['accuracy'])

    # Number of training and validation samples
    nb_train_samples = num_train_images
    nb_validation_samples = num_validation_images

    # Number of epochs for training
    epochs = 20

    # Train the model
    history = model.fit(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,  # Number of batches per epoch
        epochs=epochs,  # Number of epochs
        validation_data=validation_generator,
        validation_steps=nb_validation_samples // batch_size  # Number of validation batches per epoch
    )

    # Calculate training time
    training_time = time.time() - start_time
    print("Training time: {:.2f} seconds".format(training_time))

    model_info = "LeNet-5, Loss: C-CE, Adam Optimizer, LR: {}".format(learning_rate)

    plot_training(history)

    # Save our Model
    model.save('saved_models/my_gestures_lenet5_cnn_{0}_epochs.h5'.format(str(epochs)))
    print("Model is saved.")