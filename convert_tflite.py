# ELEC4342 Hand Gesture Recognition Project
# Conversion to TF Lite model for Raspberry Pi Deployment
# University of Hong Kong, Author: Aaron Shek
# Last Edited: 06/05/24

# Import libraries
import tensorflow as tf
import argparse

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

    # Load the Keras model
    classifier = tf.keras.models.load_model(args.model_location)

    # Convert the model to TFLite format
    converter = tf.lite.TFLiteConverter.from_keras_model(classifier)
    tflite_model = converter.convert()

    # Save the TFLite model to a file
    with open('saved_models/my_gestures_cnn_20_epochs_tflite.tflite', 'wb') as f:
        f.write(tflite_model)