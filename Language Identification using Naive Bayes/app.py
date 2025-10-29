from flask import Flask, render_template, request
import pickle
import re

app = Flask(__name__)

# --- Load the trained model and vectorizer ---
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    import numpy as np

    text = request.form['text']
    text = text.strip()

    if not text:
        return render_template('result.html', text="No input provided", lang="N/A", confidence=0)

    # Keep Unicode characters
    text_cleaned = re.sub(r'[^\w\s]', '', text)

    # Transform text for prediction
    text_vector = vectorizer.transform([text_cleaned])

    # Predict language and probabilities
    probs = model.predict_proba(text_vector)[0]
    labels = model.classes_

    # Get top prediction
    top_index = probs.argmax()
    pred_lang = labels[top_index]
    confidence = round(probs[top_index] * 100, 2)

    # Convert to a list for chart
    chart_data = [{'lang': labels[i], 'prob': round(float(probs[i]) * 100, 2)} for i in range(len(labels))]

    return render_template(
        'result.html',
        text=text,
        lang=pred_lang,
        confidence=confidence,
        chart_data=chart_data
    )


if __name__ == '__main__':
    app.run(debug=True)
