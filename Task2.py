import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from joblib import load

# --- Load dataset ---
dataset = np.load("dataset.npz")
X, y = dataset["X"], dataset["y"]

# --- Normalize and standardize ---
X = X / 255.0
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- Train/test split ---
X_train_full, X_test_full, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)

# --- Load best SVM model from Task 1 ---
best_svm = load("best_svm_model.pkl")
best_rf = load("best_rf_model.pkl")
best_knn = load("best_knn_model.pkl")


# --- Baseline scores from Task 1 (full feature set) ---
baseline_rf = 0.725 #accuracy_score(y_test, best_rf.predict(X_test_full)) 
baseline_knn = 0.757 #accuracy_score(y_test, best_knn.predict(X_test_full))
baseline_svm = 0.792 #accuracy_score(y_test, best_svm.predict(X_test_full))

# --- PCA component values to test ---
k_values = [3, 4, 6, 7, 8, 9, 10, 14, 20, 30]
rf_scores, knn_scores, svm_scores = [], [], []

print("Evaluating PCA-reduced datasets...\n")
print(f"{'k':>5} | {'RF Acc':>7} | {'RF Drop':>8} | {'KNN Acc':>7} | {'KNN Drop':>8} | {'SVM Acc':>7} | {'SVM Drop':>8}")
print("-" * 75)

for k in k_values:
    # --- Apply PCA ---
    pca = PCA(n_components=k, random_state=42)
    X_train = pca.fit_transform(X_train_full)
    X_test = pca.transform(X_test_full)

    # --- Train and evaluate RF ---
    rf = RandomForestClassifier(**best_rf.get_params())
    rf.fit(X_train, y_train)
    rf_acc = accuracy_score(y_test, rf.predict(X_test))
    rf_scores.append(rf_acc)

    # --- Train and evaluate KNN ---
    knn = KNeighborsClassifier(**best_knn.get_params())
    knn.fit(X_train, y_train)
    knn_acc = accuracy_score(y_test, knn.predict(X_test))
    knn_scores.append(knn_acc)

    # --- Train and evaluate SVM ---
    svm = SVC(**best_svm.get_params())
    svm.fit(X_train, y_train)
    svm_acc = accuracy_score(y_test, svm.predict(X_test))
    svm_scores.append(svm_acc)

    # --- Print results ---
    print(f"{k:>5} | {rf_acc:>7.3f} | {baseline_rf - rf_acc:>8.2%} | {knn_acc:>7.3f} | {baseline_knn - knn_acc:>8.2%} | {svm_acc:>7.3f} | {baseline_svm - svm_acc:>8.2%}")
