import streamlit as st
import pandas as pd
import plotly.express as px
from textblob import TextBlob
import nltk
from nltk.corpus import wordnet
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from wordcloud import WordCloud
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import langid

# Download NLTK resources
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('words')

# Define fuzzy linguistic variables
polarity = ctrl.Antecedent(universe=[-1, 1], label='Polarity')
subjectivity = ctrl.Antecedent(universe=[0, 1], label='Subjectivity')

# Define membership functions
sigma = 0.2  # Adjust based on preference

polarity['Negative'] = fuzz.trimf(polarity.universe, [-1, -1, 0])
polarity['Positive'] = fuzz.trimf(polarity.universe, [0, 1, 1])

subjectivity['Low'] = fuzz.trimf(subjectivity.universe, [0, 0, 0.5])
subjectivity['Moderate'] = fuzz.trimf(subjectivity.universe, [0.3, 0.5, 0.7])
subjectivity['High'] = fuzz.trimf(subjectivity.universe, [0.5, 1, 1])


# Function to calculate polarity degree
def calculate_polarity_degree(polarity_score, subjectivity_score):
    if subjectivity_score != 0:
        polarity_degree = polarity_score / subjectivity_score
    else:
        polarity_degree = 0
    return polarity_degree


# Define the sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Language detection function using langid (only English or Unknown)
def detect_language(word):
    lang, _ = langid.classify(word)
    if lang == 'en':
        return "English"
    else:
        return "Unknown"


# Function to get word meaning using WordNet
def get_word_meaning(word):
    # Get synsets for the word
    synsets = wordnet.synsets(word)
    if synsets:
        # Return the definition of the first synset
        return synsets[0].definition()
    else:
        return "Unknown"


# Main function to run sentiment analysis
def run_sentiment_analysis(df):
    sentiment_details = []

    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    for sentiment, manual_label in zip(df['Sentiment'], df['Label']):
        # Perform POS tagging
        word_pos = pos_tag(nltk.word_tokenize(sentiment))

        # Filter out stop words, lemmatize words, and split compound words
        filtered_words = []
        for word, pos in word_pos:
            word = word.lower()
            if word not in stop_words:
                if pos.startswith('V'):  # Verb
                    word = lemmatizer.lemmatize(word, pos='v')
                elif pos.startswith('N'):  # Noun
                    word = lemmatizer.lemmatize(word, pos='n')
                elif pos.startswith('J'):  # Adjective
                    word = lemmatizer.lemmatize(word, pos='a')
                elif pos.startswith('R'):  # Adverb
                    word = lemmatizer.lemmatize(word, pos='r')
                filtered_words.extend(re.split(r'\s|[-_]', word))

        # Reconstruct sentiment
        reconstructed_sentiment = ' '.join(filtered_words)
        # Perform sentiment analysis using TextBlob
        tb_analysis = TextBlob(reconstructed_sentiment)

        # Initialize lists to store word grades
        word_grades = []

        # Get polarity and subjectivity scores for each word
        for word, _ in word_pos:
            # Perform sentiment analysis using TextBlob on each word
            word_tb = TextBlob(word)
            polarity_score = word_tb.sentiment.polarity
            subjectivity_score = word_tb.sentiment.subjectivity

            # Remove symbols from word
            clean_word = re.sub(r'[^\w\s]', '', word)

            # Store word grade (word) and value (polarity score, subjectivity score)
            word_grades.append((lemmatizer.lemmatize(clean_word.lower()), (polarity_score, subjectivity_score)))

        # Calculate average polarity and subjectivity scores for the sentence
        if word_grades:  # Check if word_grades list is not empty
            polarity_score = sum(value[1][0] for value in word_grades) / len(word_grades)
            subjectivity_score = sum(value[1][1] for value in word_grades) / len(word_grades)
        else:
            polarity_score = 0
            subjectivity_score = 0

        # Calculate polarity degree
        polarity_degree = calculate_polarity_degree(polarity_score, subjectivity_score)

        # Normalize polarity degree using sigmoid function
        polarity_degree_normalized = sigmoid(polarity_degree)

        # Determine fuzzy linguistic labels for polarity and subjectivity
        polarity_label = 'Positive' if polarity_score > 0 else 'Negative'
        subjectivity_label = 'Low' if subjectivity_score <= 0.5 else 'Moderate' if subjectivity_score <= 0.7 else 'High'

        # Determine predicted label based on conditional rules
        predicted_label = None
        if polarity_label == 'Positive':
            predicted_label = 'Positive & ' + subjectivity_label
        elif polarity_label == 'Negative':
            predicted_label = 'Negative & ' + subjectivity_label

        # Append sentiment details if predicted label is not None
        if predicted_label:
            sentiment_details.append(
                [sentiment, word_grades, polarity_score, polarity_label,
                 subjectivity_score, subjectivity_label, polarity_degree_normalized, predicted_label])
        else:
            # Append an empty record for each label combination
            for pol in ['Positive', 'Negative']:
                for subj in ['High', 'Moderate', 'Low']:
                    sentiment_details.append(
                        ['', '', '', '', '', '', '', f"{pol} & {subj}"])

    return sentiment_details


# Streamlit app
def main():

    # Simulation section for language and meaning detection
    st.write("### Simulation")
    test_input = st.text_input("Enter a sentence or word:")

    if test_input:
        tokens = test_input.split()
        simulation_results = []
        for token in tokens:
            language = detect_language(token.lower())
            meaning = get_word_meaning(token.lower())
            tb = TextBlob(token)
            simulation_results.append({
                "Word": token,
                "Language": language,
                "Meaning": meaning,
                "Polarity": tb.sentiment.polarity,
                "Subjectivity": tb.sentiment.subjectivity
            })

        # Display results in the UI
        st.subheader("Analysis Results")
        st.write(f"Entered Text: {test_input}")
        st.write(f"Detected Language: {detect_language(test_input)}")  # Show only English or Unknown
        st.write(f"Analysis Details: {simulation_results}")
        st.write(f"Analysis Details: {get_word_meaning(test_input)}")


if __name__ == "__main__":
    main()
