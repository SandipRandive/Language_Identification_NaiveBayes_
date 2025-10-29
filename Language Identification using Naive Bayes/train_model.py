import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import pickle

# Load your dataset
data = pd.read_csv('Language Detection.csv', encoding='latin1')


# Make sure column names are correct
data.columns = ['Text', 'Language']

# Drop missing values if any
data.dropna(inplace=True)

print("✅ Dataset Loaded Successfully!")
print("Total Samples:", len(data))
print("Languages in dataset:", data['Language'].unique())

# Features and labels
X = data['Text']
y = data['Language']

# Convert text into numerical form (TF-IDF)
vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

# Split data into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

# Train the Naive Bayes model
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred) * 100

print(f"✅ Model trained successfully! Accuracy: {accuracy:.2f}%")

# Save the model and vectorizer
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("✅ Model and vectorizer saved successfully!")
