# ELEC4342 Hand Gesture Recognition Project
# Testing using Live video feed
# University of Hong Kong, Author: Aaron Shek
# Last Edited: 05/05/24

# Import libraries
from tensorflow import keras
from keras.models import load_model
import numpy as np
import argparse
import cv2
import time

def parse_arguments() -> argparse.Namespace:
    '''
    Add arguments.
    '''
    parser = argparse.ArgumentParser(
        description='Hand Gesture Recognition using Baseline CNN'
    )
    parser.add_argument('--model_location', default='saved_models/my_gestures_cnn_20_epochs.h5', 
                        type=str, required=False, help = 'Path to model'),
                       
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    classifier = load_model(args.model_location)
    
    cap = cv2.VideoCapture(0)

    # Variables for FPS calculation
    start_time = time.time()
    frame_count = 0

    while True:
        # Capture frame by frame
        ret, frame = cap.read()
        
        # Increment frame count
        frame_count += 1 
        frame = cv2.flip(frame, 1)

        # Define Region of Interest (RoI)
        RoI = frame[100:400, 320:620]
        cv2.imshow('RoI', RoI)

        # Covert RoI to grayscale
        RoI = cv2.cvtColor(RoI, cv2.COLOR_BGR2GRAY)

        # Resize RoI to match the input size of the model
        RoI = cv2.resize(RoI, (28, 28), interpolation = cv2.INTER_AREA)
        cv2.imshow('RoI scaled and gray', RoI)
        
        # Draw Rectangle around RoI on original frame
        copy = frame.copy()
        cv2.rectangle(copy, (320, 100), (620, 400), (255, 0, 0), 5)

        # Reshape RoI to match input shape of model
        RoI = RoI.reshape(1, 28, 28, 1)
        RoI = RoI/255 # Data scaling

        # Predict the class of RoI then get corresponding letter for predicted class
        
        prediction = classifier.predict(RoI, 1, verbose = 0)[0]
        # result = str(np.argmax(model.predict(RoI)))
        # result = str(classifier.predict_classes(roi, 1, verbose = 0)[0])
        # binary_prediction = 1 if result >= 0.5 else 0 
        # result = "rock" if binary_prediction == 0 else "scissor"

        # Get the index of the class with the highest probability
        predicted_class_index = np.argmax(prediction)

        # Map the class index to the corresponding class label
        class_labels = {0: "rock", 1: "scissor", 2: "paper"}
        result = class_labels[predicted_class_index]

        # Add predicted letter to frame
        cv2.putText(copy, str(result), (300, 100), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 2)
        cv2.imshow('frame', copy)

        # Calculate FPS
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time

        # Add FPS to the frame
        cv2.putText(copy, f"FPS: {round(fps, 2)}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('frame', copy)

        if cv2.waitKey(1) == 13: # Enter key
            break

cap.release()
cv2.destroyAllWindows()
