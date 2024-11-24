import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import nltk
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.corpus import sentiwordnet as swn
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Download NLTK resources
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('sentiwordnet')
nltk.download('words')

# Define fuzzy linguistic variables
polarity = ctrl.Antecedent(universe=[-1, 1], label='Polarity')
subjectivity = ctrl.Antecedent(universe=[0, 1], label='Subjectivity')

# Define Gaussian membership functions
polarity['Negative'] = fuzz.gaussmf(polarity.universe, -1, 0.5)
polarity['Neutral'] = fuzz.gaussmf(polarity.universe, 0, 0.5)
polarity['Positive'] = fuzz.gaussmf(polarity.universe, 1, 0.5)
subjectivity['Low'] = fuzz.gaussmf(subjectivity.universe, 0, 0.1)
subjectivity['Moderate'] = fuzz.gaussmf(subjectivity.universe, 0.5, 0.1)
subjectivity['High'] = fuzz.gaussmf(subjectivity.universe, 1, 0.1)

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

# Function to check if a word is a negation word
def is_negation(word):
    return word.startswith("un")

# Function to determine if a word is important based on TF-IDF score or other metrics
def is_important(word, tfidf_scores, threshold=0.5):
    return word in tfidf_scores and tfidf_scores[word] > threshold

# Function to get sentiment polarity from SentiWordNet
def get_sentiment(word):
    pos_score = 0
    neg_score = 0
    synsets = list(swn.senti_synsets(word))
    if synsets:
        for syn in synsets:
            pos_score += syn.pos_score()
            neg_score += syn.neg_score()
        pos_score /= len(synsets)
        neg_score /= len(synsets)
    return pos_score, neg_score

# Sentiment Analysis
def perform_sentiment_analysis(data, tfidf_scores):
    sentiment_details = []

    for sentiment, manual_label in zip(data['Sentiment'], data['Label']):
        word_pos = pos_tag(nltk.word_tokenize(sentiment))
        reconstructed_sentiment = ' '.join(word for word, pos in word_pos)
        tb_analysis = TextBlob(reconstructed_sentiment)
        important_word_grades = {}

        # Default values for polarity_score and subjectivity_score
        polarity_score = 0
        subjectivity_score = 0

        for word in tb_analysis.words:
            word_tb = TextBlob(word)
            polarity_score = word_tb.sentiment.polarity
            subjectivity_score = word_tb.sentiment.subjectivity

            if (is_important(word, tfidf_scores) and word not in stopwords.words('english')
                    and polarity_score != 0 and subjectivity_score != 0):
                important_word_grades[word] = [polarity_score, subjectivity_score]

            if is_negation(word):
                polarity_score *= -1
                swn_pos_score, swn_neg_score = get_sentiment(word)
                if swn_pos_score or swn_neg_score:
                    if (is_important(word, tfidf_scores)
                            and word not in stopwords.words('english')):
                        important_word_grades[word] = [swn_pos_score - swn_neg_score, subjectivity_score]

                if word.lower() in ["neither", "nor"]:
                    polarity_score = 0
                    subjectivity_score = 0.5
                    if polarity_score != 0 or subjectivity_score != 0:
                        if is_important(word, tfidf_scores):
                            important_word_grades[word] = [polarity_score, subjectivity_score]

        if important_word_grades:
            polarity_score = sum(value[0] for value in important_word_grades.values()) / len(important_word_grades)
            subjectivity_score = sum(value[1] for value in important_word_grades.values()) / len(important_word_grades)

        # Ensure that polarity_score and subjectivity_score are assigned properly
        if polarity_score < 0:
            polarity_label = 'Negative'
        elif polarity_score == 0:
            polarity_label = 'Neutral'
        else:
            polarity_label = 'Positive'

        if subjectivity_score <= 0.3:
            subjectivity_label = 'Low'
        elif subjectivity_score <= 0.7:
            subjectivity_label = 'Moderate'
        else:
            subjectivity_label = 'High'

        # Default label for when conditions are not met
        predicted_label = 'Neutral'

        if polarity_label == 'Neutral':
            predicted_label = 'Neutral'
        elif polarity_label == 'Positive':
            if subjectivity_label == 'High' or subjectivity_label == 'Moderate':
                predicted_label = 'Positive'
            elif subjectivity_label == 'Low':
                predicted_label = 'Neutral' if polarity_score >= 0 else 'Negative'
        elif polarity_label == 'Negative':
            if subjectivity_label == 'Low':
                predicted_label = 'Neutral' if polarity_score >= 0 else 'Negative'
            elif subjectivity_label == 'Moderate' or subjectivity_label == 'High':
                predicted_label = 'Negative'

        sentiment_details.append([sentiment, manual_label, predicted_label])
    return sentiment_details

def analyze_text(user_input, tfidf_scores):
    if user_input.strip() == "":
        return "INVALID INPUT!"

    try:
        # Tokenize the input to check if it contains English words
        words = nltk.word_tokenize(user_input)
        english_words = [word for word in words if word.lower() in nltk.corpus.words.words()]

        if not english_words:
            return "INVALID INPUT!"

        user_df = pd.DataFrame({'Sentiment': [user_input], 'Label': ['Unknown']})
        sentiment_details = perform_sentiment_analysis(user_df, tfidf_scores)
        predicted_label = sentiment_details[0][-1]
        return predicted_label
    except:
        return "INVALID INPUT!"

# Main Streamlit app
def main():
    st.title("Sentiment Analysis")

    # Analyze Text section
    st.subheader("Analyze Text")
    user_input = st.text_area("Enter your text here:")
    if st.button("Analyze"):
        if user_input.strip() == "":
            st.error("Please enter some text.")
        else:
            # Train TF-IDF vectorizer and transform training data
            tfidf_vectorizer = TfidfVectorizer()
            X_train_tfidf = tfidf_vectorizer.fit_transform([user_input])
            feature_names = tfidf_vectorizer.get_feature_names_out()
            tfidf_scores = dict(zip(feature_names, tfidf_vectorizer.idf_))
            # Analyze the text
            predicted_label = analyze_text(user_input, tfidf_scores)
            st.success(f"Predicted Label: {predicted_label}")

    # File uploader
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        # Ensure the dataset has the required columns
        if 'Sentiment' in df.columns and 'Label' in df.columns:
            # Train TF-IDF vectorizer and transform training data
            tfidf_vectorizer = TfidfVectorizer()
            X_train_tfidf = tfidf_vectorizer.fit_transform(df['Sentiment'])

            # Get feature names and TF-IDF scores
            feature_names = tfidf_vectorizer.get_feature_names_out()
            tfidf_scores = dict(zip(feature_names, tfidf_vectorizer.idf_))

            # Perform sentiment analysis
            sentiment_details = perform_sentiment_analysis(df, tfidf_scores)

            # Display overall results in a table
            overall_results_df = pd.DataFrame(sentiment_details,
                                              columns=['Sentiment', 'Manual Label', 'Predicted Label'])
            st.subheader("Overall Results")
            st.dataframe(overall_results_df)

            # Pie chart for predicted labels
            predicted_label_counts = overall_results_df['Predicted Label'].value_counts()
            labels = predicted_label_counts.index.tolist()
            sizes = predicted_label_counts.values.tolist()
            colors = ['red' if label == 'Negative' else 'green' if label == 'Positive' else 'orange' for label in
                      labels]
            explode = [0.1] * len(labels)
            fig1, ax1 = plt.subplots()
            wedges, texts, autotexts = ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', colors=colors,
                                               startangle=140)
            ax1.axis('equal')
            for i, a in enumerate(autotexts):
                a.set_text(f'{sizes[i]} ({round(sizes[i] / sum(sizes) * 100, 1)}%)')

            st.title("Predicted Label Distribution")
            st.pyplot(fig1)

            # Bar graph for manual label
            manual_label_counts = df['Label'].value_counts()
            manual_labels = manual_label_counts.index.tolist()
            manual_sizes = manual_label_counts.values.tolist()

            manual_colors = ['green' if label == 'Positive' else 'red' for label in manual_labels]
            manual_percentage = [f'{(size / sum(manual_sizes)) * 100:.1f}%' for size in manual_sizes]
            fig2, ax2 = plt.subplots()
            ax2.bar(manual_labels, manual_sizes, color=manual_colors)
            ax2.set_xlabel('Manual Label')
            ax2.set_ylabel('Total Count')
            ax2.set_title('Manual Label Distribution')

            # Add total count and percentage above the bars
            for i, v in enumerate(manual_sizes):
                ax2.text(i, v + 3, f"{v} ({manual_percentage[i]})", ha='center', va='bottom')

            # Bar graph for predicted label
            fig3, ax3 = plt.subplots()
            ax3.bar(labels, sizes, color=colors)
            ax3.set_xlabel('Predicted Label')
            ax3.set_ylabel('Total Count')
            ax3.set_title('Predicted Label Distribution')

            # Add total count and percentage above the bars
            for i, v in enumerate(sizes):
                ax3.text(i, v + 3, f"{v} ({round(v / sum(sizes) * 100, 1)}%)", ha='center', va='bottom')

            st.subheader("Comparison of Manual Label and Predicted Label")
            col1, col2 = st.columns(2)
            with col1:
                st.pyplot(fig2)
            with col2:
                st.pyplot(fig3)
        else:
            st.error("Uploaded CSV file must contain 'Sentiment' and 'Label' columns.")

if __name__ == "__main__":
    main()
