from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import string
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from wordcloud import WordCloud, STOPWORDS
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

app = Flask(__name__)

# Load the data, TF-IDF vectorizer, and the trained model
data = pd.read_csv("Reviews.csv")

with open("transformer.pkl", "rb") as f:
    tfidf_vectorizer = pickle.load(f)

with open("model.pkl", "rb") as f:
    bmodel = pickle.load(f)

labels = ["Negative", "Neutral", "Positive"]

# Preprocessor function
def preprocessor(review):
    # Remove HTML tags
    review = re.compile('<.*?>').sub(r'', review)
    # Remove punctuation
    review = review.translate(str.maketrans('', '', string.punctuation))
    # Remove digits
    review = review.translate(str.maketrans('', '', string.digits))
    # Lowercase
    review = review.lower()
    # Replace multiple whitespaces with a single space
    review = re.compile(r"\s+").sub(" ", review).strip()

    # Remove stop words
    total_stopwords = set(stopwords.words('english'))
    negative_stop_words = set(word for word in total_stopwords if "n't" in word or 'no' in word)
    final_stopwords = total_stopwords - negative_stop_words
    final_stopwords.add("one")

    review = [word for word in review.split() if word not in final_stopwords]

    # Stemming
    stemmer = PorterStemmer()
    review = ' '.join([stemmer.stem(word) for word in review])

    return review

# Function to get sentiment
def get_sentiment(review):
    x = preprocessor(review)
    x = tfidf_vectorizer.transform([x])
    y = int(bmodel.predict(x.reshape(1, -1)))
    return labels[y]

# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Route for handling the form submission
@app.route('/', methods=['POST'])
def analyze_sentiment():
    if request.method == 'POST':
        user_input = request.form['user_input']
        sentiment = get_sentiment(user_input)
        return render_template('index.html', user_input=user_input, sentiment=sentiment)

if __name__ == '__main__':
    app.run(debug=True)
