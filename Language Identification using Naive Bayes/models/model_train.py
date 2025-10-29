import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle
import os

# ✅ Create models folder if not exist
os.makedirs('models', exist_ok=True)

# ✅ Load dataset
data = pd.read_csv('dataset/language_detection.csv', encoding='latin1')
data.columns = ['Text', 'Language']

# ✅ Features & Labels
X = data['Text']
y = data['Language']

# ✅ Convert text to numerical features
vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

# ✅ Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

# ✅ Train Naive Bayes model
model = MultinomialNB()
model.fit(X_train, y_train)

# ✅ Predict & Evaluate
y_pred = model.predict(X_test)
print("✅ Model trained successfully!")
print("✅ Accuracy:", round(accuracy_score(y_test, y_pred)*100, 2), "%")
print("✅ Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# ✅ Save model & vectorizer
with open('models/naive_bayes_model.pkl', 'wb') as f:
    pickle.dump((model, vectorizer), f)

print("\n✅ Model and vectorizer saved successfully at 'models/naive_bayes_model.pkl'")
