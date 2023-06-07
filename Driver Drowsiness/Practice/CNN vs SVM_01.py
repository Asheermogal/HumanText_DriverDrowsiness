import numpy as np
import os
from sklearn.metrics import accuracy_score
import cv2
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

# Define the paths to the image directories
active_subjects_path = "E:/Data Science Project/Driver Drowsiness/Datasets/FaceImages/Active Subjects"
fatigue_subjects_path = "E:/Data Science Project/Driver Drowsiness/Datasets/FaceImages/Fatigue Subjects"

# Define the image dimensions
image_width = 100
image_height = 100
channels = 3

# Function to load images from a directory
def load_images_from_directory(directory_path):
    images = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".jpg"):
            image_path = os.path.join(directory_path, filename)
            image = cv2.imread(image_path)
            image = cv2.resize(image, (image_width, image_height))
            images.append(image)
    return np.array(images)

# Load images from Active Subjects directory
active_subjects_images = load_images_from_directory(active_subjects_path)
# Load images from Fatigue Subjects directory
fatigue_subjects_images = load_images_from_directory(fatigue_subjects_path)

# Create labels for the images
active_subjects_labels = np.zeros(len(active_subjects_images))  # 0 represents active subjects
fatigue_subjects_labels = np.ones(len(fatigue_subjects_images))  # 1 represents fatigue subjects

# Concatenate the images and labels
images = np.concatenate((active_subjects_images, fatigue_subjects_images), axis=0)
labels = np.concatenate((active_subjects_labels, fatigue_subjects_labels), axis=0)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Convert the labels to categorical if needed
y_train_categorical = keras.utils.to_categorical(y_train)
y_test_categorical = keras.utils.to_categorical(y_test)

# # RNN model
# rnn_model = Sequential([
#     layers.Reshape((image_width * image_height * channels,), input_shape=(image_width, image_height, channels)),
#     layers.Dense(64, activation="relu"),
#     layers.RepeatVector(10),  # Repeat the vector for 10 timesteps
#     layers.LSTM(64, return_sequences=True),
#     layers.TimeDistributed(layers.Dense(2, activation="softmax"))  # Assuming 2 classes (active and fatigue)
# ])
#
# rnn_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
# rnn_model.fit(X_train, y_train_categorical, batch_size=32, epochs=10, validation_data=(X_test, y_test_categorical))

# SVM model
svm_model = SVC()
svm_model.fit(X_train, y_train)
svm_predictions = svm_model.predict(X_test)
svm_accuracy = accuracy_score(y_test, svm_predictions)

# MLP model
mlp_model = MLPClassifier(hidden_layer_sizes=(128,), activation="relu", solver="adam")
mlp_model.fit(X_train, y_train)
mlp_predictions = mlp_model.predict(X_test)
mlp_accuracy = accuracy_score(y_test, mlp_predictions)

# Random Forest model
rf_model = RandomForestClassifier(n_estimators=100)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_predictions)

# Gradient Boosting model
gb_model = GradientBoostingClassifier(n_estimators=100)
gb_model.fit(X_train, y_train)
gb_predictions = gb_model.predict(X_test)
gb_accuracy = accuracy_score(y_test, gb_predictions)

# Print the accuracies
# print("RNN Accuracy:", rnn_model.evaluate(X_test, y_test_categorical)[1])
print("SVM Accuracy:", svm_accuracy)
print("MLP Accuracy:", mlp_accuracy)
print("Random Forest Accuracy:", rf_accuracy)
print("Gradient Boosting Accuracy:", gb_accuracy)
