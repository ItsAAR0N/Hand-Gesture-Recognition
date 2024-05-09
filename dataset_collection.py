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
import argparse
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

def parse_arguments() -> argparse.Namespace:
    '''
    Add arguments.
    '''
    parser = argparse.ArgumentParser(
        description='Hand Gesture Recognition using Baseline CNN'
    )
    parser.add_argument('--model', default='mobilenetv2', # Adjust file name as necessary
                        type=str, required=False, help = 'Select model: baseline_cnn - mobilenetv2 - '),                      
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    if args.model == 'mobilenetv2':
        file_loc_name = 'handgestures_mnv2'
        image_size = (96,96)
    else:
        image_size = (28,28)
        file_loc_name = 'handgestures_baseline_cnn'

    # Open camera
    cap = cv2.VideoCapture(0)

    # Number of classes
    num_classes = 3  # Change this to the desired number of classes

    # Loop over each class to collect data
    if os.path.exists('./{0}/train'.format(file_loc_name)) and os.path.exists('./{0}/test'.format(file_loc_name)):
        print("Dataset already exists, skipping data collection...")
    else:
        # Open camera
        cap = cv2.VideoCapture(0)

        # Initialize Counters
        i = 0
        image_count = 0

        while i < 9: 
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)

            # Define ROI (Region of Interest)
            roi = frame[100:400, 320:620]
            cv2.imshow('roi', roi)

            # Convert RoI to grayscale and resize
            if args.model == 'baseline_cnn':
                roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            else:
                roi = roi

            roi = cv2.resize(roi, image_size, interpolation=cv2.INTER_AREA) # Baseline CNN (28x28x1), MobileNETV2 (96x96x3)

            cv2.imshow('roi scaled and gray', roi)
            copy = frame.copy()
            cv2.rectangle(copy, (320, 100), (620, 400), (255,0,0), 5)

            if i == 0:
                image_count = 0
                cv2.putText(copy, "Hit enter to record when ready", (100,100), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
            elif i == 1:
                image_count += 1
                cv2.putText(copy, "Recording 1st gesture - Train", (100, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
                cv2.putText(copy, str(image_count), (400, 400), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
                gesture_dir = './{0}/train/0/'.format(file_loc_name)
                makedir(gesture_dir)
                cv2.imwrite(gesture_dir + str(image_count) + ".jpg", roi)
            elif i == 2:
                image_count += 1
                cv2.putText(copy, "Recording 1st gesture - Test", (100, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
                cv2.putText(copy, str(image_count), (400, 400), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
                gesture_dir = './{0}/test/0/'.format(file_loc_name)
                makedir(gesture_dir)
                cv2.imwrite(gesture_dir + str(image_count) + ".jpg", roi)
            elif i == 3:
                cv2.putText(copy, "Hit enter to record when ready to record 2nd gesture", (100,100), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
            elif i == 4:
                image_count += 1
                cv2.putText(copy, "Recording 2nd gesture - Train", (100, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
                cv2.putText(copy, str(image_count), (400, 400), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
                gesture_dir = './{0}/train/1/'.format(file_loc_name)
                makedir(gesture_dir)
                cv2.imwrite(gesture_dir + str(image_count) + ".jpg", roi)
            elif i == 5:
                image_count += 1
                cv2.putText(copy, "Recording 2nd gesture - Test", (100, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
                cv2.putText(copy, str(image_count), (400, 400), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
                gesture_dir = './{0}/test/1/'.format(file_loc_name)
                makedir(gesture_dir)
                cv2.imwrite(gesture_dir + str(image_count) + ".jpg", roi)
            elif i == 6:
                cv2.putText(copy, "Hit enter to record when ready to record 3rd gesture", (100,100), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
            elif i == 7:
                image_count += 1
                cv2.putText(copy, "Recording 3rd gesture - Train", (100, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
                cv2.putText(copy, str(image_count), (400, 400), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
                gesture_dir = './{0}/train/2/'.format(file_loc_name)
                makedir(gesture_dir)
                cv2.imwrite(gesture_dir + str(image_count) + ".jpg", roi)
            elif i == 8:
                image_count += 1
                cv2.putText(copy, "Recording 3rd gesture - Test", (100, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
                cv2.putText(copy, str(image_count), (400, 400), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
                gesture_dir = './{0}/test/2/'.format(file_loc_name)
                makedir(gesture_dir)
                cv2.imwrite(gesture_dir + str(image_count) + ".jpg", roi)
            elif i == 9:
                cv2.putText(copy, "Hit Enter to Exit", (100,100), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)

            cv2.imshow('frame', copy)

            if cv2.waitKey(1) == 13: # 13 is enter key
                image_count = 0
                i += 1

    # Release camera and close windows
    cap.release()
    cv2.destroyAllWindows()

cap.release()
cv2.destroyAllWindows()
print("Data Collection Complete.")
