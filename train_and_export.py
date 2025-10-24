"""
Spam Mail Prediction using Machine Learning
Train TF-IDF + Logistic Regression model and export weights as JSON
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import json

print("=" * 60)
print("SPAM DETECTION MODEL TRAINING")
print("=" * 60)

# Data Collection & Pre-Processing
print("\n📊 Loading dataset...")
raw_mail_data = pd.read_csv('supabase/functions/analyze-email/mail_data.csv')

# Replace null values with empty string
mail_data = raw_mail_data.where((pd.notnull(raw_mail_data)), '')

print(f"Dataset shape: {mail_data.shape}")
print(f"Spam emails: {len(mail_data[mail_data['Category'] == 'spam'])}")
print(f"Ham emails: {len(mail_data[mail_data['Category'] == 'ham'])}")

# Label encoding: spam = 0, ham = 1
mail_data.loc[mail_data['Category'] == 'spam', 'Category'] = 0
mail_data.loc[mail_data['Category'] == 'ham', 'Category'] = 1

# Separate features and labels
X = mail_data['Message']
Y = mail_data['Category']

# Splitting the data into training & test data
print("\n🔀 Splitting data (80% train, 20% test)...")
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# Feature Extraction
print("\n🔤 Extracting features using TF-IDF...")
feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)

# Convert Y_train and Y_test to integers
Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')

print(f"Feature dimensions: {X_train_features.shape[1]}")

# Training the Model - Logistic Regression
print("\n🤖 Training Logistic Regression model...")
model = LogisticRegression()
model.fit(X_train_features, Y_train)

# Evaluate Model
print("\n📈 Evaluating model accuracy...")
train_prediction = model.predict(X_train_features)
train_accuracy = accuracy_score(Y_train, train_prediction)

test_prediction = model.predict(X_test_features)
test_accuracy = accuracy_score(Y_test, test_prediction)

print(f"Training accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
print(f"Test accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

# Export Model Weights and Vocabulary
print("\n💾 Exporting model weights and vocabulary...")

# Export TF-IDF vocabulary and IDF values
vocabulary = feature_extraction.vocabulary_
idf_values = feature_extraction.idf_.tolist()

# Export Logistic Regression coefficients and intercept
coefficients = model.coef_[0].tolist()
intercept = float(model.intercept_[0])

# Create export data structure
model_data = {
    "model_type": "LogisticRegression",
    "vectorizer_type": "TfidfVectorizer",
    "vocabulary": vocabulary,
    "idf_values": idf_values,
    "coefficients": coefficients,
    "intercept": intercept,
    "n_features": X_train_features.shape[1],
    "train_accuracy": train_accuracy,
    "test_accuracy": test_accuracy
}

# Save to JSON file
output_path = 'supabase/functions/analyze-email/model_weights.json'
with open(output_path, 'w') as f:
    json.dump(model_data, f, indent=2)

print(f"\n✅ Model training complete!")
print(f"📁 Model weights exported to: {output_path}")
print(f"📊 Model contains {len(vocabulary)} vocabulary terms")
print(f"🎯 Ready for production deployment!")
print("=" * 60)
