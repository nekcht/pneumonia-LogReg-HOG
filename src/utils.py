# utils.py
# Christou Nektarios - Image Processing, 2022-2023 NKUA


import os
import cv2
import random
import zipfile
import requests
import numpy as np
import seaborn as sns
from io import BytesIO
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def plot_random_samples(X, y, num_samples=4):
    # Plot some randomly selected images
    random_indices = random.sample(range(len(X)), k=num_samples)
    fig, axes = plt.subplots(1, 4, figsize=(20, 10))
    for i, idx in enumerate(random_indices):
        image = X[idx]
        label = y[idx]
        axes[i].imshow(image, cmap='gray')
        axes[i].set_title("PNEUMONIA" if label == 1 else "NORMAL")
        axes[i].axis('off')
    plt.show()


def extract_hog_features(image):
    """
    Histogram of Oriented Gradients" (HOG),
    extracts features based on the distribution
    of edge directions in an image, which can be
    useful for differentiating between different
    patterns and structures.
    """
    # Compute HOG features
    hog = cv2.HOGDescriptor()
    features = hog.compute(image)

    # Flatten the feature vector
    features = features.flatten()

    return features


def show_cm(y_test, y_pred_test):
    # Compute the confusion matrix
    cm = confusion_matrix(y_test, y_pred_test)
    # Create a heatmap of the confusion matrix
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")

    # Customize the plot
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")

    # Show the plot
    plt.show()

    return None


def get_dataset():
  """
  This function assumes you have extracted the pneumonia dataset
  in ./dataset dir.
  """
    # Set the path to the main folder containing the subfolders
    main_folder = "dataset"

    # Set the names of the subfolders representing the classes
    class_names = ["normal", "pneumonia"]

    # Initialize lists to store the images and labels
    images = []
    labels = []

    # Read images from each subfolder and assign labels
    for class_index, class_name in enumerate(class_names):
        folder_path = os.path.join(os.getcwd(), main_folder, class_name)

        if not os.path.exists(folder_path):
            print(f"Folder not found: {folder_path}")
            continue

        image_files = os.listdir(folder_path)

        for image_file in image_files:
            image_path = os.path.join(os.getcwd(), folder_path, image_file)

            if not os.path.exists(image_path):
                print(f"File not found: {image_path}")
                continue
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (150, 150))  # you can experiment with different sizes
            images.append(image)
            labels.append(class_index)

    # Convert the lists to numpy arrays
    images = np.array(images)
    labels = np.array(labels)

    # Shuffle the dataset
    shuffle_indices = np.random.permutation(len(images))
    images = images[shuffle_indices]
    labels = labels[shuffle_indices]

    # Split the dataset into train, test set
    train_ratio = 0.85
    test_ratio = 0.15

    train_size = int(len(images) * train_ratio)
    test_size = int(len(images) * test_ratio)

    X_train = images[:train_size]
    y_train = labels[:train_size]

    X_test = images[train_size:]
    y_test = labels[train_size:]

    return X_train, y_train, X_test, y_test
