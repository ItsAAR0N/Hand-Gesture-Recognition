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
import psutil # Memory info

def parse_arguments() -> argparse.Namespace:
    '''
    Add arguments.
    '''
    parser = argparse.ArgumentParser(
        description='Hand Gesture Recognition using Baseline CNN, LeNet-5, or MobileNetV2'
    )
    parser.add_argument('--model', default='mobilenetv2', # Adjust file name as necessary
                        type=str, required=False, help = 'Select model: "baseline_cnn" - "lenet-5" - "mobilenetv2" - '),
    parser.add_argument('--model_location', default='saved_models/my_gestures_mobilenet_25_epochs.h5', # Adjust file name as necessary 
                        type=str, required=False, help = 'Path to model'),
                       
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    classifier = load_model(args.model_location)
    
    if (args.model) == 'mobilenetv2':
        image_size = (96,96)
        input_shape = (96, 96, 3) # RGB 
    elif args.model == 'baseline_cnn':
        image_size = (28,28)
        input_shape = (28, 28, 1) # Grayscale
    else: 
        image_size = (32, 32)
        input_shape = (32, 32, 1) # Gray scale

    cap = cv2.VideoCapture(0)

    # Variables for FPS calculation
    start_time = time.time()
    frame_count = 0

    while True:
        # Get memory usage information
        process = psutil.Process()
        memory_usage = process.memory_info().rss / (1024 * 1024)  # Memory usage in MB


        # Capture frame by frame
        ret, frame = cap.read()
        
        # Increment frame count
        frame_count += 1 
        frame = cv2.flip(frame, 1)

        # Define Region of Interest (RoI)
        RoI = frame[100:400, 320:620]
        cv2.imshow('RoI', RoI)

        # Covert RoI to grayscale
        if args.model == 'baseline_cnn':
            RoI = cv2.cvtColor(RoI, cv2.COLOR_BGR2GRAY) # Adjust as necessary for baseline CNN and MobileNetV2 
        else:
            RoI = RoI

        # Resize RoI to match the input size of the model
        RoI = cv2.resize(RoI, image_size, interpolation = cv2.INTER_AREA) # Baseline CNN (28x28x1), MobileNETV2 (96x96x3)
        cv2.imshow('RoI scaled and gray', RoI)
        
        # Draw Rectangle around RoI on original frame
        copy = frame.copy()
        cv2.rectangle(copy, (320, 100), (620, 400), (255, 0, 0), 5)

        # Reshape RoI to match input shape of model
        RoI = RoI.reshape((1,) + input_shape)
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

        # Get the confidence score of the predicted class
        confidence_score = prediction[predicted_class_index]

        # Add predicted letter and confidence score to frame
        cv2.putText(copy, f"Prediction: {result}, Confidence: {confidence_score:.2f}", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('frame', copy)

        # Calculate FPS
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time

        # Add FPS and memory consumption to the frame
        cv2.putText(copy, f"FPS: {round(fps, 2)}, Memory Usage: {memory_usage:.2f} MB", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('frame', copy)

        if cv2.waitKey(1) == 13: # Enter key
            break

cap.release()
cv2.destroyAllWindows()