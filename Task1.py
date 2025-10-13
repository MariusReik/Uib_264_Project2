import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler

# Load dataset
dataset = np.load("dataset.npz")
X, y = dataset["X"], dataset["y"]

# Normalize pixel values (0–255 → 0–1)
X = X / 255.0

# Optional: Standardize features (mean=0, std=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize classifiers
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
knn_clf = KNeighborsClassifier(n_neighbors=5)

# Train classifiers
rf_clf.fit(X_train, y_train)
knn_clf.fit(X_train, y_train)

# Evaluate Random Forest
rf_preds = rf_clf.predict(X_test)
print("Random Forest Results")
print("Accuracy:", accuracy_score(y_test, rf_preds))
print(classification_report(y_test, rf_preds))

# Evaluate KNN
knn_preds = knn_clf.predict(X_test)
print("KNN Results")
print("Accuracy:", accuracy_score(y_test, knn_preds))
print(classification_report(y_test, knn_preds))
