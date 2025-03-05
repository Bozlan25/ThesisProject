import streamlit as st
import pandas as pd
import numpy as np
import skfuzzy as fuzz
import skfuzzy.control as ctrl
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from langdetect import detect, DetectorFactory
from textblob import TextBlob
import re
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import torch

# Ensure consistent language detection
DetectorFactory.seed = 0

# Load mBERT for Filipino sentiment analysis
model_name = "bert-base-multilingual-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

label_mapping = {0: "Negative", 1: "Neutral", 2: "Positive"}

# Load predefined Filipino sentiment words dataset
filipino_words_df = pd.read_csv("filtered_words.csv")  # Updated filename

# Convert dataset into a dictionary for faster lookup
filipino_sentiment_words = {
    "Positive": filipino_words_df[filipino_words_df['Sentiment'] == 'positive']['Word'].tolist(),
    "Neutral": filipino_words_df[filipino_words_df['Sentiment'] == 'neutral']['Word'].tolist(),
    "Negative": filipino_words_df[filipino_words_df['Sentiment'] == 'negative']['Word'].tolist()
}

# Load sarcasm words dataset
def load_sarcasm_words():
    df = pd.read_csv("sarcasm.csv")
    return {
        "positive": df[df['Sentiment'] == 'positive']['Word'].tolist(),
        "negative": df[df['Sentiment'] == 'negative']['Word'].tolist()
    }

sarcasm_words = load_sarcasm_words()

def analyze_filipino_sentiment(text):
    words = text.lower().split()
    for sentiment, word_list in filipino_sentiment_words.items():
        if any(word in words for word in word_list):
            return sentiment
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=1).item()
    return label_mapping[prediction]


def detect_language(text):
    """Detects language and classifies it as English, Tagalog, or English-Tagalog."""
    try:
        lang_code = detect(text)

        # Split words and count Tagalog vs English words
        words = re.findall(r'\b\w+\b', text.lower())  # Extract words
        tagalog_count = sum(1 for word in words if detect(word) == 'tl')
        english_count = sum(1 for word in words if detect(word) == 'en')

        if tagalog_count > 0 and english_count > 0:
            return "English-Tagalog"
        elif english_count > 0 and tagalog_count > 0:
            return "English-Tagalog"
        elif lang_code == "en":
            return "English"
        elif lang_code == "tl":
            return "Tagalog"
        else:
            return "Other"
    except:
        return "Unknown"


def analyze_sentiment(text):
    """Analyzes sentiment using TextBlob (supports English)."""
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
    subjectivity = analysis.sentiment.subjectivity

    # Fuzzy system evaluation
    fis = define_fuzzy_system()
    degree_of_polarity = compute_sentiment_fis(fis, polarity, subjectivity)

    # Assign label based on fuzzy output
    if degree_of_polarity > 0.1:
        return "Positive"
    elif degree_of_polarity < -0.1:
        return "Negative"
    else:
        return "Neutral"


def define_fuzzy_system():
    """Defines the fuzzy logic system for sentiment analysis."""
    polarity = ctrl.Antecedent(np.arange(-1, 1.1, 0.1), 'polarity')
    subjectivity = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'subjectivity')
    degree_of_polarity = ctrl.Consequent(np.arange(-1, 1.1, 0.1), 'degree_of_polarity')

    polarity['Negative'] = fuzz.gaussmf(polarity.universe, -1, 0.5)
    polarity['Neutral'] = fuzz.gaussmf(polarity.universe, 0, 0.5)
    polarity['Positive'] = fuzz.gaussmf(polarity.universe, 1, 0.5)

    subjectivity['Low'] = fuzz.gaussmf(subjectivity.universe, 0, 0.3)
    subjectivity['High'] = fuzz.gaussmf(subjectivity.universe, 1, 0.3)

    degree_of_polarity['Negative'] = fuzz.gaussmf(degree_of_polarity.universe, -1, 0.5)
    degree_of_polarity['Neutral'] = fuzz.gaussmf(degree_of_polarity.universe, 0, 0.5)
    degree_of_polarity['Positive'] = fuzz.gaussmf(degree_of_polarity.universe, 1, 0.5)

    rule1 = ctrl.Rule(polarity['Negative'] & subjectivity['High'], degree_of_polarity['Negative'])
    rule2 = ctrl.Rule(polarity['Positive'] & subjectivity['High'], degree_of_polarity['Positive'])
    rule3 = ctrl.Rule(subjectivity['Low'], degree_of_polarity['Neutral'])

    fis_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
    fis = ctrl.ControlSystemSimulation(fis_ctrl)

    return fis


def compute_sentiment_fis(fis, polarity_score, subjectivity_score):
    """Computes sentiment based on fuzzy logic system."""
    fis.input['polarity'] = polarity_score
    fis.input['subjectivity'] = subjectivity_score
    fis.compute()
    return fis.output['degree_of_polarity']


def train_svm_model(df):
    """Trains an SVM model on text data."""
    vectorizer = TfidfVectorizer()
    encoder = LabelEncoder()
    X = vectorizer.fit_transform(df['Text'])
    y = encoder.fit_transform(df['Label'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = SVC(kernel='linear', probability=True)
    model.fit(X_train, y_train)

    return vectorizer, encoder, model


def classify_sentiment(text, vectorizer, encoder, model):
    """Classifies sentiment using a trained SVM model."""
    text_vectorized = vectorizer.transform([text])
    prediction = model.predict(text_vectorized)
    sentiment_label = encoder.inverse_transform(prediction)[0]
    return sentiment_label

def detect_sarcasm_adjusted_sentiment(text):
    detected_language = detect_language(text)
    sentiment = analyze_sentiment(text) if "English" in detected_language else analyze_filipino_sentiment(text)
    words = text.lower().split()
    sarcasm_detected = any(word in sarcasm_words["positive"] or word in sarcasm_words["negative"] for word in words)
    if sarcasm_detected:
        sentiment = "Negative" if sentiment == "Positive" else "Positive" if sentiment == "Negative" else "Neutral"
    return sentiment

# Streamlit UI
st.title("Sentiment Analysis (Enhanced)")

user_input = st.text_area("Enter a sentiment:")
if st.button("Analyze"):
    if user_input.strip():
        detected_language = detect_language(user_input)
        sentiment_label = detect_sarcasm_adjusted_sentiment(user_input)

        st.write(f"**Entered Text:** {user_input}")
        st.write(f"**Detected Language:** {detected_language}")
        st.write(f"**Predicted Sentiment:** {sentiment_label}")
