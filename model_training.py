import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
raw_mail_data = pd.read_csv("mail_data.csv")
mail_data = raw_mail_data.where((pd.notnull(raw_mail_data)), '')

# Encode labels: 'spam' -> 0, 'ham' -> 1
mail_data.loc[mail_data['Category'] == 'spam', 'Category'] = 0
mail_data.loc[mail_data['Category'] == 'ham', 'Category'] = 1

# Features and labels
X = mail_data['Message']
Y = mail_data['Category']

# Split into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)

# TF-IDF vectorization
vectorizer = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
X_train_features = vectorizer.fit_transform(X_train)
X_test_features = vectorizer.transform(X_test)

# Convert labels to integers
Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')

# Train the model
model = LogisticRegression(max_iter=200)
model.fit(X_train_features, Y_train)

# Evaluate the model
accuracy = accuracy_score(Y_test, model.predict(X_test_features))
print(f"Model Accuracy: {accuracy:.2f}")

# Detailed metrics
print("\nClassification Report:")
print(classification_report(Y_test, model.predict(X_test_features), target_names=['Spam', 'Ham']))

# Save the model and vectorizer
with open("model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

with open("vectorizer.pkl", "wb") as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

print("Model and vectorizer saved successfully.")