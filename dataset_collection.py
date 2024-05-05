# ELEC4342 Hand Gesture Recognition Project
# Dataset Collection utilizing CV2
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
from sklearn.metrics import accuracy_score
import cv2
import time
import os

# Create a function to setup the directories to store images
def makedir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        return None 
    else:
        pass 

# Open camera
cap = cv2.VideoCapture(0)

# Number of classes
num_classes = 3  # Change this to the desired number of classes

# Loop over each class to collect data
if os.path.exists('./handgestures/train') and os.path.exists('./handgestures/test'):
    print("Dataset already exists, skipping data collection...")
else:
    for class_index in range(num_classes):
        # Initialize counters
        i = 0
        image_count = 0
        while True:
            print("Press Enter key when ready...")
            if cv2.waitKey(1) == 13:
                break
        # Loop to capture gestures for the current class
        while i < 3:  # Adjust the condition based on your requirement
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)

            # Define Region of Interest (ROI)
            roi = frame[100:400, 320:620]
            cv2.imshow('roi', roi)
            
            # Convert ROI to grayscale and resize
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)

            cv2.imshow('roi scaled and gray', roi)
            copy = frame.copy()
            cv2.rectangle(copy, (320, 100), (620, 400), (255, 0, 0), 5)

            # Define directory based on class index
            gesture_dir = f'./handgestures/train/{class_index}/' if i % 2 == 1 else f'./handgestures/test/{class_index}/'
            makedir(gesture_dir)

            # Increment image count for the current class
            image_count += 1

            # Save the image with appropriate label
            cv2.imwrite(gesture_dir + str(image_count) + ".jpg", roi)

            # Display appropriate message based on the stage of recording
            if i == 0:
                cv2.putText(copy, "Hit enter to record when ready", (100, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
            elif i == 6:
                cv2.putText(copy, "Hit Enter to Exit", (100, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
            else:
                recording_stage = "Train" if i % 2 == 1 else "Test"
                cv2.putText(copy, f"Recording {class_index} gesture - {recording_stage}", (100, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
                cv2.putText(copy, str(image_count), (400, 400), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)

            cv2.imshow('frame', copy)

            # Wait for enter key (13 is the ASCII code for enter)
            if cv2.waitKey(1) == 13:
                image_count = 0
                i += 1


# Release camera and close windows
cap.release()
cv2.destroyAllWindows()

print("Data Collection Complete.")
