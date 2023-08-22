import streamlit as st
from newspaper import Article
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd

nltk.download('vader_lexicon')
nltk.download('stopwords')

url = st.text_input('Enter the URL of the article:')

if url:
    # Extract article information
    article = Article(url)
    article.download()
    article.parse()
    st.write('Title:', article.title)
    st.write('Text:', article.text)

    # Save article text to file
    file_name = f'{article.title}.txt'
    with open(file_name, 'w') as f:
        f.write(article.text)

    # Perform sentiment analysis
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(article.text)
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [word for word in tokens if not word in stop_words]

    sid = SentimentIntensityAnalyzer()
    sentiment_scores = sid.polarity_scores(' '.join(tokens))
    
    # Create DataFrame
    data = {'URL': [url],
            'Article Title': [article.title],
            'Positive Score': [sentiment_scores['pos']],
            'Negative Score': [sentiment_scores['neg']],
            'Neutral Score': [sentiment_scores['neu']],
            'Subjectivity Score': [sentiment_scores['pos'] + sentiment_scores['neg']]}
    df = pd.DataFrame(data)
    st.write(df)
