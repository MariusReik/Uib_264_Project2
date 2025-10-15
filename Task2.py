import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load dataset
dataset = np.load("dataset.npz")
X, y = dataset["X"], dataset["y"]

# Normalize and standardize
X = X / 255.0
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train_full, X_test_full, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Best parameters from Task 1
rf_best_params = {'n_estimators': 150, 'max_depth': None, 'min_samples_split': 2}
knn_best_params = {'n_neighbors': 5, 'weights': 'distance'}

# Baseline scores (from Task 1)
baseline_rf = 0.745  # Replace with your actual RF baseline
baseline_knn = 0.755  # Replace with your actual KNN baseline

# PCA component values to test
k_values = [3, 4, 6, 8, 10, 14, 20, 30]
rf_scores = []
knn_scores = []

print("Evaluating PCA-reduced datasets...\n")
print(f"{'k':>5} | {'RF Acc':>7} | {'RF Drop':>8} | {'KNN Acc':>7} | {'KNN Drop':>8}")
print("-" * 45)

for k in k_values:
    # Apply PCA
    pca = PCA(n_components=k)
    X_train = pca.fit_transform(X_train_full)
    X_test = pca.transform(X_test_full)

    # Train and evaluate RF
    rf = RandomForestClassifier(**rf_best_params, random_state=42)
    rf.fit(X_train, y_train)
    rf_acc = accuracy_score(y_test, rf.predict(X_test))
    rf_scores.append(rf_acc)

    # Train and evaluate KNN
    knn = KNeighborsClassifier(**knn_best_params)
    knn.fit(X_train, y_train)
    knn_acc = accuracy_score(y_test, knn.predict(X_test))
    knn_scores.append(knn_acc)

    # Print results
    print(f"{k:>5} | {rf_acc:>7.3f} | {baseline_rf - rf_acc:>8.2%} | {knn_acc:>7.3f} | {baseline_knn - knn_acc:>8.2%}")

