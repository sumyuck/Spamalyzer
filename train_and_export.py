"""
Train spam detection model and export to ONNX format
Run this script once to generate the ONNX model files
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import StringTensorType
import pickle

print("Loading dataset...")
# Load the data
raw_mail_data = pd.read_csv('supabase/functions/analyze-email/mail_data.csv')

# Replace null values with empty string
mail_data = raw_mail_data.where((pd.notnull(raw_mail_data)), '')

# Label encoding: spam = 0, ham = 1
mail_data.loc[mail_data['Category'] == 'spam', 'Category'] = 0
mail_data.loc[mail_data['Category'] == 'ham', 'Category'] = 1

# Separate features and labels
X = mail_data['Message']
Y = mail_data['Category']

print(f"Dataset loaded: {len(X)} emails")

# Split the data (same random_state as notebook)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)

print("Training TF-IDF vectorizer...")
# Feature extraction (exact same parameters as notebook)
feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)

# Convert labels to integers
Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')

print("Training Logistic Regression model...")
# Train the model
model = LogisticRegression()
model.fit(X_train_features, Y_train)

# Evaluate accuracy
train_accuracy = accuracy_score(Y_train, model.predict(X_train_features))
test_accuracy = accuracy_score(Y_test, model.predict(X_test_features))

print(f"Training accuracy: {train_accuracy:.4f}")
print(f"Test accuracy: {test_accuracy:.4f}")

# Save the vectorizer as pickle (ONNX doesn't support TF-IDF well)
print("Saving TF-IDF vectorizer...")
with open('supabase/functions/analyze-email/tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(feature_extraction, f)

# Convert Logistic Regression model to ONNX
print("Converting model to ONNX...")
initial_type = [('float_input', FloatTensorType([None, X_train_features.shape[1]]))]
onnx_model = convert_sklearn(model, initial_types=initial_type, target_opset=12)

# Save ONNX model
with open('supabase/functions/analyze-email/spam_model.onnx', 'wb') as f:
    f.write(onnx_model.SerializeToString())

print("\n✅ Model training complete!")
print("📁 Generated files:")
print("   - supabase/functions/analyze-email/tfidf_vectorizer.pkl")
print("   - supabase/functions/analyze-email/spam_model.onnx")
print("\nYou can now use these files in your edge function!")
