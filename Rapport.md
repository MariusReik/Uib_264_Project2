### INF264 PROJECT 2: Gift Recognizer 4 Santa

## Members:

# Lyder Samnøy
# Marius Reikerås

## Introduction

This Christmas-themed task aims to explore different machine learning classifiers in situations where data is both labled and unlabled. As well as how PCA is used to drastically lower the number of features in a data set without compromising on the accuracy of the models.

## Data Description

We make use of two datasets; dataset.npz for task 1 and 2, and dataset_corrupted.npz for task 3. dataset.npz is a one-demensional array of grayscale values, with 2614 images in total, corresponding to 14 different classes of labled objects. dataset_corrupted.npz is smaller, with only 405 images, these are not labled, and 89 of the images are "corrupted", meaning they should not conform to any of the image classes.

## Task 1: Automatic Gift Recognizer
In this task, we aim to find reliable classifiers to correctly identify the images. We chose two different models,- K nearest neighbors and Random Forest, comparing them and seeing which is more accurate, also comparing other factors like runtime.

K nearest neighbors was chosen because it is a simple and easy to understand model, where noise levels can be easily tweaked by incrementing the n-neighbors parameter. With Random Forest, we wanted a more sophisticated model that utilizes several models (individual decision trees and the ensemble forest) to se if light ensemble learning yields a higher accuracy.

We utilize Sklearn's implementations for both machine learning models (KNeighborsClassifier and RandomForestClassifier). In addition to providing an easy-to-use implementation, Sklearn also provides tuning tools for hyperparameter tuning. We make use of Sklearn's GridSearchCV and cross_val_score for tuning and validation. As well as classification_report and accuracy_score to calculate and present model results.

## Task 2 Dimensionality Reduction
Task 2 aims to explore how Principal Component Analysis (PCA) can be used to reduce dimensionality in data without compromising on accuracy. For our dataset, the default feature number is 400. In this task, we aim to reduce the feature number, and analyze how the accuracy score is impacted.

We used Sklearn's existing PCA implementation with our best parameters from task 1. We found that the accuracy stayed roughly the same at higher k-values, with a double-digit % decrease in accuracy for both models only at k-values 6 and below. This illustrates how PCA can drastically reduce the number of features without affecting the accuracy. PCA sorts the features by variance. So in our case, when f.ex. k = 8, only the 8 pixels that generate the most variance are kept. Often, this is enough to make an equally accurate prediction compared to having the full feature set, as the majority of features are redundant.

## Task 3: Bad Data

In this task, we were given the dataset `corrupted_data.npz`, containing 405 unlabeled images where 89 are known to be corrupted. These corrupted images do not belong to any of the 14 known categories. The goal of this task was to automatically identify the corrupted images.

We approached this as an unsupervised anomaly detection problem, where the goal is to find images that differ from the original dataset categories from Task 1. As the task description suggested, we reused our model from Task 1 the SVM, since it achieved the best performance, and combined it with PCA and KMeans clustering.

The idea was for the SVM model to measure model uncertainty for each image, where higher uncertainty indicates potential anomalies. We then combined this with PCA, which calculated a reconstruction error that measures how well each image fits the general structure of the “normal” data. A higher reconstruction score indicates higher noise or corruption.

After normalizing both values, we used KMeans clustering to group the images into two clusters: one representing normal images and one representing corrupted ones. The cluster with the lowest average model confidence was labeled as “corrupted.”

This approach flagged 88 out of 405 images as corrupted. Since there were no labels to verify the predictions directly, we manually inspected the flagged images. Almost all of them appeared visibly corrupted or noisy, suggesting that the approach worked very well. We therefore estimate the method’s accuracy to be approximately 88/89, matching closely with the known number of corrupted images in the dataset.