# app.py
from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import sklearn
nltk.download('punkt_tab')
nltk.download('stopwords')

# vectorizer
vectorizer_path = 'Vectorizer.pkl'
with open(vectorizer_path, 'rb') as file:
    tf_idf = pickle.load(file)

# Load the trained model
model_path = 'model.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)

ps=PorterStemmer()

def transform_text(text):
    print(text)
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y = []

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y = []

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


@app.route('/', methods=['POST'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])

def predict():
# preprocessing
    input_sms = request.form.get("Email")
    transformed_text=transform_text(input_sms)

# vectorize
    transformed_text=tf_idf.transform([transformed_text])

# predict
    result=model.predict(transformed_text)[0]

# display
    if result==1:
        output="Spam"
    else:
        output="Not Spam" 

    return render_template('index.html', prediction_text='Prediction: {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)
