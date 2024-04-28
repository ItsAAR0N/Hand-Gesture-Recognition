# Import Libraries
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns # Visualization Library
from IPython.display import display, Image
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from sklearn.metrics import accuracy_score
import cv2
import time
# Define our CNN model
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import backend as K 
from tensorflow.keras.optimizers import Adam # Adam optimizer



# Obtain our Training and Testing Data
train = pd.read_csv('sign_mnist_train/sign_mnist_train.csv')
test = pd.read_csv('sign_mnist_test/sign_mnist_test.csv')

# Get our Training Labels
labels = train['label'].values

# View the unique labels, 24 in total (no. 9)
unique_val = np.array(labels)
np.unique(unique_val)

train.drop('label', axis = 1, inplace = True)
# Extract the image data from each row in csv, 784 columns
images = train.values
images = np.array([np.reshape(i, (28, 28)) for i in images])
images = np.array([i.flatten() for i in images])

# One hot encode our labels (binary)
label_binrizer = LabelBinarizer() # Pre-processing
labels = label_binrizer.fit_transform(labels)
labels # View labels
# len(labels[0]) # 24

# Inspect a random image
index = random.randint(1,len(labels[0])) 
# print(labels[index])

# Split data into x_train, x_test, y_trian and y_test on totally unseen dataset good practice, 80/20 split
x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size = 0.2, random_state = 101)

# Utilize Tensorflow and define batch size, etc.
batch_size = 128
num_classes = 24
epochs = 1

# Scale images (Pre-processing)
x_train = x_train / 255
x_test = x_test / 255

# Reshape into required TF and Keras size
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)


model = Sequential()
model.add(Conv2D(64, kernel_size = (3,3), activation = 'relu', input_shape = (28, 28, 1)))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Conv2D(64, kernel_size = (3,3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Conv2D(64, kernel_size = (3,3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Flatten())
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.20)) # Add 20% dropout (pruning)

model.add(Dense(num_classes, activation = 'softmax'))

# Compile our model
model.compile(loss = 'categorical_crossentropy',
              optimizer = Adam(), 
              metrics = ['accuracy'])

history = model.fit(x_train, y_train, validation_data = (x_test, y_test), epochs = epochs, batch_size = batch_size)

model.save('saved_models/sign_mnist_cnn_10_epochs.h5')

test_labels = test['label']
test.drop('label', axis = 1, inplace = True) # Drop classes from test dataset

# Convert test images data to numpy array
test_images = test.values
test_images = np.array([np.reshape(i, (28, 28)) for i in test_images])
test_images = np.array([i.flatten() for i in test_images]) # Flatten each reshaped image to 1d array

# One-hot encoding
test_labels = label_binrizer.fit_transform(test_labels) 

test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)

test_images.shape

y_pred = model.predict(test_images) # Generate predictions

accuracy = accuracy_score(test_labels, y_pred.round())
# print(accuracy)
accuracies = []
zero_percentages = []

# Définir les seuils
threshholds = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3,0.35,0.4,0.45]

# Calculer l'accuracy initiale et l'ajouter à la liste
y_pred = model.predict(test_images)  # Generate predictions
initial_accuracy = accuracy_score(test_labels, y_pred.round())
accuracies.append(initial_accuracy)
zero_percentages.append(0)  # Pas de pruning pour le seuil 0

# Parcourir les seuils de pruning
for threshold in threshholds[1:]:
    total_weights = 0
    total_zero_weights = 0

    for layer in model.layers:
        weights = layer.get_weights()  # obtenir les poids actuels
        new_weights = []
        norme = 0
        for w in weights:
            # Compter le total des poids et des poids mis à zéro
            total_weights += np.size(w)
            total_zero_weights += np.sum(np.abs(w) == 0)

            # Appliquer la condition : poids < threshold deviennent 0
            modified_w = np.where(np.abs(w) < threshold, 0, w)
            new_weights.append(modified_w)
            norme+= np.sum(np.square(w))
        norme = (norme)**(1/2)
        print(type(threshold))
        print("norme",norme)
        # if norme < threshold:
        #     for w in weights:
        #         # Compter le total des poids et des poids mis à zéro
        #         # total_weights += np.size(w)
        #         # total_zero_weights += np.sum(np.abs(w) < threshold)

        #         # Appliquer la condition : poids < threshold deviennent 0
        #         modified_w = np.where(np.abs(w) !=0, 0, w)
        #         new_weights.append(modified_w)
        #         norme+= np.sum(np.square(w))
        #         total_zero_weights += len(w)
        # # Mettre à jour les poids de la couche avec les nouveaux poids
        # if new_weights != []:
        #     layer.set_weights(new_weights)
        layer.set_weights(new_weights)
    # Prédire et calculer la nouvelle accuracy
    y_pred = model.predict(test_images)
    new_accuracy = accuracy_score(test_labels, y_pred.round())
    accuracies.append(new_accuracy)

    # Calculer le pourcentage de poids à zéro
    zero_percentage = (total_zero_weights / total_weights) * 100
    
    zero_percentages.append(zero_percentage)
    # zero_percentages.append(0)

# Créer le graphique pour l'accuracy
plt.figure(figsize=(10, 5))
plt.plot(threshholds, accuracies, marker='o', label='Accuracy')
# plt.plot(threshholds, zero_percentages, marker='o', color='red', label='% of Weights Zeroed')
plt.title(' accuracy after structured Pruning Threshold')
plt.xlabel('Pruning Threshold')
plt.ylabel('accuracy')
plt.legend()
plt.grid(True)
plt.show()