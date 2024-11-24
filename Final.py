import streamlit as st
import pandas as pd
import plotly.express as px
from transformers import pipeline, BertTokenizer, BertForSequenceClassification
from nltk.tokenize import word_tokenize
from textblob import TextBlob
import nltk
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import numpy as np
import re
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from wordcloud import WordCloud
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import fasttext.util


# Download NLTK resources
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('wordnet')

# Load BERT for recognizing cultural nuances and ambiguous language
m_bert = BertForSequenceClassification.from_pretrained('google/bert_uncased_L-12_H-768_A-12')
tokenizer = BertTokenizer.from_pretrained('google/bert_uncased_L-12_H-768_A-12')
bert_analyzer = pipeline("sentiment-analysis", model=m_bert, tokenizer=tokenizer)

# Load m-BERT for Filipino language
m_bert_fil = BertForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
tokenizer_fil = BertTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
bert_analyzer_fil = pipeline("sentiment-analysis", model=m_bert_fil, tokenizer=tokenizer_fil)

# Load FastText model (for slang and neologisms)
fasttext.util.download_model('en', if_exists='ignore')  # For English
fasttext_model = fasttext.load_model('cc.en.300.bin')  # Use the desired language model


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
    # Check if subjectivity score is not zero to avoid division by zero
    if subjectivity_score != 0:
        polarity_degree = polarity_score / subjectivity_score
    else:
        polarity_degree = 0  # Handle division by zero by setting polarity degree to 0
    return polarity_degree


# Define the sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


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
                # Split compound words
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
            polarity_score = 0  # Set default polarity score to 0 if no words are found
            subjectivity_score = 0  # Set default subjectivity score to 0 if no words are found

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
            if subjectivity_label == 'High' or subjectivity_label == 'Moderate' or subjectivity_label == 'Low':
                predicted_label = 'Positive & ' + subjectivity_label
        elif polarity_label == 'Negative':
            if subjectivity_label == 'High' or subjectivity_label == 'Moderate' or subjectivity_label == 'Low':
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


# SVM and TF-IDF
def svm_and_tfidf(df):
    # Instantiate TF-IDF Vectorizer
    tfidf_vectorizer = TfidfVectorizer(max_features=1000)

    # Fit and transform the data
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['Sentiment']).toarray()

    additional_features = np.array([[TextBlob(sentiment).sentiment.polarity,
                                     TextBlob(sentiment).sentiment.subjectivity]
                                    for sentiment in df['Sentiment']])

    # Concatenate TF-IDF matrix with additional features
    features = np.concatenate((tfidf_matrix, additional_features), axis=1)

    # SVM model
    svm_model = SVC(kernel='linear')
    svm_model.fit(features, df['Label'])

    return svm_model, tfidf_vectorizer


# Streamlit app
def main():
    st.title("Sentiment Analysis")
    st.write("Please upload a CSV file containing two columns: Sentiment and Label (Positive, Negative).")

    uploaded_file = st.file_uploader("Choose a file", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        st.write("### Overall Results")
        st.write("")

        sentiment_details = run_sentiment_analysis(df)
        result_df = pd.DataFrame(sentiment_details,
                                 columns=['Sentiment', 'Word Grades', 'Polarity Score', 'Polarity Label',
                                          'Subjectivity Score', 'Subjectivity Label', 'Normalized Polarity Degree',
                                          'Predicted Label'])

        # Convert tuples in 'Word Grades' column to strings
        result_df['Word Grades'] = result_df['Word Grades'].apply(lambda x: ', '.join([word for word, _ in x]))

        st.dataframe(result_df)

        # Generate word cloud from word grades
        all_words = ' '.join(result_df['Word Grades'])
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_words)

        # Plot word cloud
        st.write("### Word Cloud Based on Word Grades")
        st.image(wordcloud.to_array(), caption="Word Cloud", use_container_width=True)

        # Scatter plot for Polarity Score
        fig_polarity = px.scatter(result_df, x=result_df.index, y='Polarity Score', color='Polarity Label',
                                  hover_data=['Sentiment', 'Polarity Label', 'Subjectivity Label', 'Predicted Label'],
                                  color_discrete_map={'Positive': 'green', 'Negative': 'red'})
        fig_polarity.update_layout(title='Polarity of Sentiments Visualization', xaxis_title='Sentiment Index',
                                   yaxis_title='Polarity Score', xaxis=dict(range=[0, len(result_df)]),
                                   yaxis=dict(range=[-1, 1]))
        fig_polarity.update_xaxes(title_text="Sentiment Index")
        fig_polarity.update_traces(textposition='top center')

        # Background color for y-axis ranges
        fig_polarity.add_shape(type="rect", xref="paper", yref="y",
                               x0=0, y0=-1, x1=1, y1=0,
                               fillcolor="#FFC0CB", opacity=0.5, layer="below", line_width=0)
        fig_polarity.add_shape(type="rect", xref="paper", yref="y",
                               x0=0, y0=0, x1=1, y1=1,
                               fillcolor="#90EE90", opacity=0.5, layer="below", line_width=0)

        # Display the scatter plot with modified settings
        st.plotly_chart(fig_polarity)

        # Scatter plot for Subjectivity Score
        fig_subjectivity = px.scatter(result_df, x=result_df.index, y='Subjectivity Score', color='Subjectivity Label',
                                      hover_data=['Sentiment', 'Polarity Label', 'Subjectivity Label',
                                                  'Predicted Label'])

        # Change color of data points corresponding to "Low" subjectivity to blue
        fig_subjectivity.for_each_trace(lambda t: t.update(marker_color='blue') if t.name == 'Low' else ())

        fig_subjectivity.update_layout(
            title='Subjectivity of Sentiments Visualization',
            xaxis_title='Sentiment Index',
            yaxis_title='Subjectivity Score',
            xaxis=dict(range=[0, len(result_df)]),
            yaxis=dict(
                range=[0, 1],
                tickvals=[0, 0.1, 0.3, 0.5, 0.7, 1],
                ticktext=["0", "0.1", "0.3", "0.5", "0.7", "1"]
            )
        )
        fig_subjectivity.update_xaxes(title_text="Sentiment Index")
        fig_subjectivity.update_traces(textposition='top center')

        # Background color for y-axis ranges
        fig_subjectivity.add_shape(type="rect", xref="paper", yref="y",
                                   x0=0, y0=0, x1=1, y1=0.3,
                                   fillcolor="#ADD8E6", opacity=0.5, layer="below", line_width=0)
        fig_subjectivity.add_shape(type="rect", xref="paper", yref="y",
                                   x0=0, y0=0.1, x1=1, y1=0.7,
                                   fillcolor="#ADD8E6", opacity=0.7, layer="below", line_width=0)
        fig_subjectivity.add_shape(type="rect", xref="paper", yref="y",
                                   x0=0, y0=0.5, x1=1, y1=1,
                                   fillcolor="#FFA07A", opacity=0.5, layer="below", line_width=0)

        # Display the scatter plot with modified settings
        st.plotly_chart(fig_subjectivity)

        # Conditional Statements Visualization
        fig_conditional = px.scatter(result_df, x=result_df.index, y='Predicted Label',
                                     color='Predicted Label',
                                     hover_data=['Sentiment', 'Polarity Label', 'Subjectivity Label',
                                                 'Predicted Label'])

        # Define custom y-axis labels
        custom_y_labels = ['Negative & High', 'Negative & Moderate', 'Negative & Low',
                           'Positive & Low', 'Positive & Moderate', 'Positive & High']

        # Update y-axis labels and background colors
        fig_conditional.update_yaxes(categoryorder='array', categoryarray=custom_y_labels)
        fig_conditional.update_layout(title='Conditional Statements Visualization',
                                      xaxis_title='Sentiment Index', yaxis_title='Predicted Label',
                                      shapes=[dict(type='rect', xref='paper', yref='paper', x0=0, y0=0,
                                                   x1=1, y1=0.5, fillcolor='red', opacity=0.5,
                                                   layer='below', line_width=0),
                                              dict(type='rect', xref='paper', yref='paper', x0=0, y0=0.5,
                                                   x1=1, y1=1, fillcolor='green', opacity=0.5,
                                                   layer='below', line_width=0)])

        # Display the scatter plot with modified settings
        st.plotly_chart(fig_conditional)

        # SVM Performance Metrics
        svm_model, tfidf_vectorizer = svm_and_tfidf(df)

        tfidf_matrix = tfidf_vectorizer.transform(df['Sentiment']).toarray()
        additional_features = np.array([[TextBlob(sentiment).sentiment.polarity,
                                         TextBlob(sentiment).sentiment.subjectivity]
                                        for sentiment in df['Sentiment']])
        features = np.concatenate((tfidf_matrix, additional_features), axis=1)
        predictions = svm_model.predict(features)

        accuracy = accuracy_score(df['Label'], predictions)
        precision = precision_score(df['Label'], predictions, average='weighted')
        recall = recall_score(df['Label'], predictions, average='weighted')
        f1 = f1_score(df['Label'], predictions, average='weighted')

        st.write("### Existing FSVM Model Performance Metrics")
        st.write(f"Accuracy: {accuracy:.2f}")
        st.write(f"Precision: {precision:.2f}")
        st.write(f"Recall: {recall:.2f}")
        st.write(f"F1-score: {f1:.2f}")


if __name__ == "__main__":
    main()
