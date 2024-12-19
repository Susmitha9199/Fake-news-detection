import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import nltk
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from wordcloud import WordCloud
import dask.dataframe as dd  # Dask for memory-efficient handling of large datasets

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('vader_lexicon')

# Set up the Streamlit page
st.set_page_config(page_title="Fake News Detection", layout="wide")

# Sidebar for options
st.sidebar.title("Options")
st.sidebar.write("Choose visualization or load the datasets.")
visualization_choice = st.sidebar.selectbox(
    "Choose Visualization",
    [
        "Bar Plot of Average Article Length by Label",
        "Distribution of Article Lengths by Label",
        "Top 10 Authors by Article Count",
        "Article Length vs Sentiment Score Scatter Plot",
        "Top 10 Most Common Words in Fake Articles",
        "Top 10 Most Common Words in Real Articles",
        "Pie Chart of Articles by Label",
        "Correlation Heatmap"
    ]
)

# Direct paths to the datasets
train_file_path = '/Users/susmitha/Desktop/cpp/train.csv'
test_file_path = '/Users/susmitha/Desktop/cpp/test.csv'

# Load datasets with pandas (change to Dask if working with larger files)
train_df = pd.read_csv(train_file_path)
test_df = pd.read_csv(test_file_path)

# Show basic information about datasets
st.write("### Training Dataset")
st.write(train_df.head())
st.write("### Testing Dataset")
st.write(test_df.head())

# Preprocessing function to clean text
stop_words = set(stopwords.words('english'))
def preprocess_text(text):
    if isinstance(text, str):
        text = text.lower()
        text = ' '.join([word for word in text.split() if word not in stop_words])
        return text
    else:
        return ''

# Apply preprocessing to text columns
train_df['text'] = train_df['text'].apply(preprocess_text)
test_df['text'] = test_df['text'].apply(preprocess_text)
train_df['article_length'] = train_df['text'].apply(lambda x: len(x.split()))
test_df['article_length'] = test_df['text'].apply(lambda x: len(x.split()))

# Add sentiment scores to the DataFrame
sid = SentimentIntensityAnalyzer()
train_df['sentiment'] = train_df['text'].apply(lambda x: sid.polarity_scores(x)['compound'])
test_df['sentiment'] = test_df['text'].apply(lambda x: sid.polarity_scores(x)['compound'])

# Vectorization
tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(train_df['text']).toarray()
X_test_tfidf = tfidf.transform(test_df['text']).toarray()

# Add sentiment to feature vectors
X_train = np.hstack((X_train_tfidf, train_df['sentiment'].values.reshape(-1, 1)))
y_train = train_df['label']
X_test = np.hstack((X_test_tfidf, test_df['sentiment'].values.reshape(-1, 1)))

# Model training and prediction
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Save predictions in test dataset
test_df['predicted_label'] = y_pred
st.write("### Predictions")
st.write(test_df.head())

# Visualizations based on user choice
if visualization_choice == "Bar Plot of Average Article Length by Label":
    st.write("### Bar Plot of Average Article Length by Label")
    plt.figure(figsize=(10, 5))
    sns.barplot(x='label', y='article_length', data=train_df, estimator=np.mean)
    plt.title('Average Article Length by Label')
    plt.xlabel('Label (0: Real, 1: Fake)')
    plt.ylabel('Average Article Length (Number of Words)')
    st.pyplot(plt)

elif visualization_choice == "Distribution of Article Lengths by Label":
    st.write("### Distribution of Article Lengths by Label")
    plt.figure(figsize=(10, 5))
    sns.histplot(data=train_df, x='article_length', hue='label', bins=30, kde=True, element='step')
    plt.title('Distribution of Article Lengths by Label')
    plt.xlabel('Article Length (Number of Words)')
    plt.ylabel('Frequency')
    st.pyplot(plt)

elif visualization_choice == "Top 10 Authors by Article Count":
    st.write("### Top 10 Authors by Article Count")
    top_authors = train_df['author'].value_counts().head(10)
    plt.figure(figsize=(10, 5))
    sns.barplot(x=top_authors.index, y=top_authors.values)
    plt.title('Top 10 Authors by Article Count')
    plt.xlabel('Authors')
    plt.ylabel('Number of Articles')
    plt.xticks(rotation=45)
    st.pyplot(plt)

elif visualization_choice == "Article Length vs Sentiment Score Scatter Plot":
    st.write("### Article Length vs Sentiment Score Scatter Plot")
    plt.figure(figsize=(10, 5))
    sns.scatterplot(x='article_length', y='sentiment', data=train_df, hue='label', alpha=0.5)
    plt.title('Article Length vs Sentiment Score')
    plt.xlabel('Article Length (Number of Words)')
    plt.ylabel('Sentiment Score')
    st.pyplot(plt)

elif visualization_choice == "Top 10 Most Common Words in Fake Articles":
    st.write("### Top 10 Most Common Words in Fake Articles")
    fake_articles = train_df[train_df['label'] == 1]['text']
    fake_words = ' '.join(fake_articles)
    fake_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(fake_words)
    plt.figure(figsize=(10, 5))
    plt.imshow(fake_wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)

elif visualization_choice == "Top 10 Most Common Words in Real Articles":
    st.write("### Top 10 Most Common Words in Real Articles")
    real_articles = train_df[train_df['label'] == 0]['text']
    real_words = ' '.join(real_articles)
    real_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(real_words)
    plt.figure(figsize=(10, 5))
    plt.imshow(real_wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)

elif visualization_choice == "Pie Chart of Articles by Label":
    st.write("### Pie Chart of Articles by Label")
    plt.figure(figsize=(8, 8))
    labels = ['Real', 'Fake']
    sizes = train_df['label'].value_counts()
    colors = ['lightblue', 'lightcoral']
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
    plt.title('Distribution of Articles by Label')
    st.pyplot(plt)

elif visualization_choice == "Correlation Heatmap":
    st.write("### Correlation Heatmap")
    plt.figure(figsize=(10, 8))
    correlation_matrix = train_df.corr()
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', square=True)
    st.pyplot(plt)
