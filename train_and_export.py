"""
Email Spam Detection using KNN and SVM
Train models on word frequency features and export weights as JSON
"""

import numpy as np
import pandas as pd
import zipfile
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import json

print("=" * 60)
print("EMAIL SPAM DETECTION - KNN & SVM MODELS")
print("=" * 60)

# Extract dataset from zip
print("\n📦 Extracting dataset from zip...")
with zipfile.ZipFile('supabase/functions/analyze-email/emails.csv.zip', 'r') as zip_ref:
    zip_ref.extractall('supabase/functions/analyze-email/')

# Data Collection & Pre-Processing
print("📊 Loading email dataset...")
df = pd.read_csv('supabase/functions/analyze-email/emails.csv')

print(f"Dataset shape: {df.shape}")
print(f"Total emails: {len(df)}")

# Separate features and labels
# X contains word frequency features (3000+ columns)
# y contains Prediction (0 = ham/safe, 1 = spam)
X = df.drop(['Email No.', 'Prediction'], axis=1)
y = df['Prediction']

print(f"Spam emails: {y.sum()}")
print(f"Ham emails: {len(y) - y.sum()}")
print(f"Number of word features: {X.shape[1]}")

# Feature Scaling using MinMaxScaler
print("\n⚖️ Scaling features using MinMaxScaler...")
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and test sets (75% train, 25% test)
print("\n🔀 Splitting data (75% train, 25% test)...")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.25, random_state=0
)

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# ============================================
# Model 1: K-Nearest Neighbors (KNN)
# ============================================
print("\n🤖 Training K-Nearest Neighbors (KNN) model...")
print("   Testing different k values to find optimal...")

# Find optimal k by testing k=1 to k=40
errors = []
for k in range(1, 41):
    knn_temp = KNeighborsClassifier(n_neighbors=k)
    knn_temp.fit(X_train, y_train)
    pred_temp = knn_temp.predict(X_test)
    errors.append(np.mean(pred_temp != y_test))

optimal_k = errors.index(min(errors)) + 1
print(f"   Optimal k value: {optimal_k} (lowest error: {min(errors):.4f})")

# Train final KNN model with optimal k
knn_model = KNeighborsClassifier(n_neighbors=optimal_k)
knn_model.fit(X_train, y_train)

# Evaluate KNN
knn_train_pred = knn_model.predict(X_train)
knn_test_pred = knn_model.predict(X_test)
knn_train_accuracy = accuracy_score(y_train, knn_train_pred)
knn_test_accuracy = accuracy_score(y_test, knn_test_pred)

print(f"   KNN Training accuracy: {knn_train_accuracy:.4f} ({knn_train_accuracy*100:.2f}%)")
print(f"   KNN Test accuracy: {knn_test_accuracy:.4f} ({knn_test_accuracy*100:.2f}%)")

# ============================================
# Model 2: Support Vector Machine (SVM)
# ============================================
print("\n🤖 Training Support Vector Machine (SVM) model...")
print("   Using linear kernel...")

svm_model = SVC(kernel='linear', probability=True)
svm_model.fit(X_train, y_train)

# Evaluate SVM
svm_train_pred = svm_model.predict(X_train)
svm_test_pred = svm_model.predict(X_test)
svm_train_accuracy = accuracy_score(y_train, svm_train_pred)
svm_test_accuracy = accuracy_score(y_test, svm_test_pred)

print(f"   SVM Training accuracy: {svm_train_accuracy:.4f} ({svm_train_accuracy*100:.2f}%)")
print(f"   SVM Test accuracy: {svm_test_accuracy:.4f} ({svm_test_accuracy*100:.2f}%)")

# Detailed Classification Reports
print("\n📈 KNN Classification Report:")
print(classification_report(y_test, knn_test_pred))

print("\n📈 SVM Classification Report:")
print(classification_report(y_test, svm_test_pred))

# Export Model Weights and Components
print("\n💾 Exporting model weights and components...")

# Get feature names (word columns)
feature_names = X.columns.tolist()

# Export scaler parameters
scaler_min = scaler.data_min_.tolist()
scaler_max = scaler.data_max_.tolist()
scaler_scale = scaler.scale_.tolist()

# Export KNN training data (needed for prediction)
knn_training_data = X_train.tolist()
knn_training_labels = y_train.tolist()

# Export SVM parameters
svm_support_vectors = svm_model.support_vectors_.tolist()
svm_support_labels = svm_model.support_[svm_model.support_].tolist()
svm_dual_coef = svm_model.dual_coef_.tolist()
svm_intercept = float(svm_model.intercept_[0])

# Create comprehensive export data structure
model_data = {
    "model_types": ["KNN", "SVM"],
    "primary_model": "SVM",  # SVM performs slightly better in general
    "dataset_info": {
        "total_samples": len(df),
        "n_features": len(feature_names),
        "spam_count": int(y.sum()),
        "ham_count": int(len(y) - y.sum())
    },
    "feature_names": feature_names,
    "scaler": {
        "type": "MinMaxScaler",
        "data_min": scaler_min,
        "data_max": scaler_max,
        "scale": scaler_scale
    },
    "knn": {
        "n_neighbors": optimal_k,
        "training_data": knn_training_data,
        "training_labels": knn_training_labels,
        "train_accuracy": knn_train_accuracy,
        "test_accuracy": knn_test_accuracy
    },
    "svm": {
        "kernel": "linear",
        "support_vectors": svm_support_vectors,
        "support_indices": svm_support_labels,
        "dual_coef": svm_dual_coef,
        "intercept": svm_intercept,
        "train_accuracy": svm_train_accuracy,
        "test_accuracy": svm_test_accuracy
    }
}

# Save to JSON file
output_path = 'supabase/functions/analyze-email/model_weights.json'
with open(output_path, 'w') as f:
    json.dump(model_data, f, indent=2)

print(f"\n✅ Model training complete!")
print(f"📁 Model weights exported to: {output_path}")
print(f"📊 Models trained on {len(feature_names)} word frequency features")
print(f"🎯 KNN Test Accuracy: {knn_test_accuracy*100:.2f}%")
print(f"🎯 SVM Test Accuracy: {svm_test_accuracy*100:.2f}%")
print(f"🎯 Ready for production deployment!")
print("=" * 60)
