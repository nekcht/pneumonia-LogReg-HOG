# main.py
# Christou Nektarios - Image Processing, 2022-2023 NKUA


import numpy as np
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from utils import plot_random_samples, extract_hog_features, show_cm, get_dataset


def main():
    # read the dataset
    X_train, y_train, X_test, y_test = get_dataset()

    # plot some random samples
    plot_random_samples(X_train, y_train, num_samples=4)

    """
    Now we'll define the function that extracts HOG features from an image. 
    HOG(Histogram of Gradients) extracts features based on the distribution
    of edge directions in an image, which can be useful for differentiating
    between different patterns and structures.
    """

    # Perform feature (HOG) extraction on the training set
    hog_features_train = np.array([extract_hog_features(image) for image in X_train])

    # Perform feature (HOG) extraction on the test set
    hog_features_test = np.array([extract_hog_features(image) for image in X_test])

    """
    Now that we've got our feature vectors, let's use a simple Logistic Regression model.
    """

    # Calculate the mean and standard deviation across all intensities in the images
    intensity_mean_train = np.mean(hog_features_train)
    intensity_std_train = np.std(hog_features_train)
    intensity_mean_test = np.mean(hog_features_train)
    intensity_std_test = np.std(hog_features_train)

    # Standardize the images by subtracting the mean and dividing by the standard deviation
    standardized_hog_features_train = (hog_features_train - intensity_mean_train) / intensity_std_train
    standardized_hog_features_test = (hog_features_test - intensity_mean_test) / intensity_std_test

    # Train a Logistic Regression classifier
    logistic_regression_classifier = LogisticRegression()
    logistic_regression_classifier.fit(standardized_hog_features_train, y_train)

    # Make predictions on the test set
    y_pred_test = logistic_regression_classifier.predict(standardized_hog_features_test)

    # Evaluate the classifier on the test set
    test_report = classification_report(y_test, y_pred_test)
    print("Test Set Report:")
    print(test_report)

    # Compute the confusion matrix
    show_cm(y_test, y_pred_test)


if __name__ == '__main__':
    main()
