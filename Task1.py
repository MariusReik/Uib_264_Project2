import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler

# Load dataset
dataset = np.load("dataset.npz")
X, y = dataset["X"], dataset["y"]

# Normalize pixel values
X = X / 255.0

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# -------------------------------
# 🔍 Grid Search for Random Forest
# -------------------------------
rf_param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 7, 13],
    'min_samples_split': [2, 5, 9]
}

rf_grid = GridSearchCV(RandomForestClassifier(random_state=42), rf_param_grid, cv=3, scoring='accuracy', n_jobs=-1)
rf_grid.fit(X_train, y_train)

print("🎯 Best RF Parameters:", rf_grid.best_params_)
print("🎯 RF Cross-Validated Score:", rf_grid.best_score_)

# -------------------------------
# 🔍 Grid Search for KNN
# -------------------------------
knn_param_grid = {
    'n_neighbors': [3, 5, 7],
    'weights': ['uniform', 'distance']
}

knn_grid = GridSearchCV(KNeighborsClassifier(), knn_param_grid, cv=3, scoring='accuracy', n_jobs=-1)
knn_grid.fit(X_train, y_train)

print("\n🎯 Best KNN Parameters:", knn_grid.best_params_)
print("🎯 KNN Cross-Validated Score:", knn_grid.best_score_)

# -------------------------------
# 🔍 Evaluate Best Models on Test Set
# -------------------------------
rf_preds = rf_grid.predict(X_test)
print("\nRandom Forest Results")
print("Test Accuracy:", accuracy_score(y_test, rf_preds))
print(classification_report(y_test, rf_preds))

knn_preds = knn_grid.predict(X_test)
print("KNN Results")
print("Test Accuracy:", accuracy_score(y_test, knn_preds))
print(classification_report(y_test, knn_preds))
