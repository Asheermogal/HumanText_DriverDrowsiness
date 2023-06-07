import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers

# Set the paths to the dataset folders
active_subjects_path = r'E:\Data Science Project\Driver Drowsiness\Datasets\FaceImages\Active Subjects'
fatigue_subjects_path = r'E:\Data Science Project\Driver Drowsiness\Datasets\FaceImages\Fatigue Subjects'

# Function to load and preprocess images
def load_images(folder_path):
    images = []
    labels = []

    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (64, 64))  # Resize images to a common size
        images.append(image)
        labels.append(0 if 'Active' in folder_path else 1)  # Assign labels: 0 for active, 1 for fatigue

    return images, labels

# Load images and labels from the dataset folders
active_images, active_labels = load_images(active_subjects_path)
fatigue_images, fatigue_labels = load_images(fatigue_subjects_path)

# Concatenate the data and labels
all_images = np.concatenate((active_images, fatigue_images), axis=0)
all_labels = np.concatenate((active_labels, fatigue_labels), axis=0)

# Split the data into training and testing sets
train_images, test_images, train_labels, test_labels = train_test_split(
    all_images, all_labels, test_size=0.2, random_state=42
)

# Reshape and normalize the image data
train_images = train_images.reshape(train_images.shape[0], 64, 64, 1)
test_images = test_images.reshape(test_images.shape[0], 64, 64, 1)
train_images = train_images.astype("float32") / 255.0
test_images = test_images.astype("float32") / 255.0

# Define the CNN model
model = keras.Sequential(
    [
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=(64, 64, 1)),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ]
)

# Compile and train the model
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(train_images, train_labels, epochs=10, batch_size=32, verbose=1)

# Evaluate the model on the test data
_, accuracy = model.evaluate(test_images, test_labels)
print("CNN Accuracy:", accuracy)
