import streamlit as st
import nltk
from newspaper import Article
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd
import requests
from streamlit_lottie import st_lottie
import base64
from fpdf import FPDF
import docx
import os

# Create a directory for NLTK data
nltk_data_dir = os.path.join(os.getcwd(), 'nltk_data')
os.makedirs(nltk_data_dir, exist_ok=True)

# Download NLTK data to the specified directory
nltk.download('vader_lexicon', download_dir=nltk_data_dir)
nltk.download('stopwords', download_dir=nltk_data_dir)
nltk.download('punkt', download_dir=nltk_data_dir)

# Add the directory to NLTK's data path
nltk.data.path.append(nltk_data_dir)

# Set up page config
st.set_page_config(
    page_title="Article Sentiment Analyzer",
    page_icon="📰",
    layout="wide"
)

# Function to load Lottie animations
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# App title and description
st.title("📰 Article Sentiment Analyzer")
st.markdown("""
    This app extracts text from online articles and performs sentiment analysis.
    Enter a URL below to get started!
""")

# Display animation
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    lottie_url_hello = "https://lottie.host/0d49d388-9acb-492c-92c2-8dad820db057/R3WmiFyHXU.json"
    lottie_hello = load_lottieurl(lottie_url_hello)
    st_lottie(lottie_hello, height=200)

# URL input
url = st.text_input('Enter the URL of the article:')

# Function to safely create file name
def create_safe_filename(title):
    # Replace invalid characters
    invalid_chars = ['<', '>', ':', '"', '/', '\\', '|', '?', '*']
    for char in invalid_chars:
        title = title.replace(char, '')
    # Limit length
    return title[:50] if title else "article"

# Main process when URL is provided
if url:
    try:
        with st.spinner('Extracting article...'):
            # Extract article information
            article = Article(url)
            article.download()
            article.parse()
            
            # Create safe filename
            safe_title = create_safe_filename(article.title)
            file_name = f'{safe_title}.txt'
            
            # Display article info in expandable sections
            with st.expander("Article Title"):
                st.subheader(article.title)
            
            with st.expander("Article Text"):
                st.write(article.text)
            
            # Save article text to file
            with open(file_name, 'w', encoding='utf-8') as f:
                f.write(article.text)
            
            # Perform sentiment analysis
            with st.spinner('Analyzing sentiment...'):
                stop_words = set(stopwords.words('english'))
                tokens = word_tokenize(article.text.lower())
                tokens = [word for word in tokens if word.isalpha()]
                tokens = [word for word in tokens if word not in stop_words]
                
                sid = SentimentIntensityAnalyzer()
                sentiment_scores = sid.polarity_scores(' '.join(tokens))
                
                # Calculate overall sentiment
                if sentiment_scores['compound'] >= 0.05:
                    overall_sentiment = "Positive"
                elif sentiment_scores['compound'] <= -0.05:
                    overall_sentiment = "Negative"
                else:
                    overall_sentiment = "Neutral"
                
                # Create DataFrame
                data = {
                    'URL': [url],
                    'Article Title': [article.title],
                    'Positive Score': [sentiment_scores['pos']],
                    'Negative Score': [sentiment_scores['neg']],
                    'Neutral Score': [sentiment_scores['neu']],
                    'Compound Score': [sentiment_scores['compound']],
                    'Overall Sentiment': [overall_sentiment]
                }
                df = pd.DataFrame(data)
            
            # Display sentiment analysis results
            st.subheader("Sentiment Analysis Results")
            st.dataframe(df)
            
            # Visualize sentiment scores
            st.subheader("Sentiment Score Distribution")
            sentiment_data = {
                'Sentiment': ['Positive', 'Neutral', 'Negative'],
                'Score': [sentiment_scores['pos'], sentiment_scores['neu'], sentiment_scores['neg']]
            }
            chart_data = pd.DataFrame(sentiment_data)
            st.bar_chart(chart_data.set_index('Sentiment'))
            
            # Download section
            st.subheader("Download Article")
            col1, col2 = st.columns(2)
            
            with col1:
                download_format = st.selectbox("Select download format", ["PDF", "Text", "DOCX"])
            
            with col2:
                download_button = st.button("Generate Download Link")
            
            if download_button:
                with st.spinner('Preparing download...'):
                    with open(file_name, "r", encoding='utf-8') as f:
                        text = f.read()
                        
                        if download_format == "PDF":
                            pdf = FPDF()
                            pdf.add_page()
                            pdf.set_font("Arial", size=12)
                            
                            # Handle encoding issues by splitting text into smaller chunks
                            # and removing non-Latin characters
                            chunks = [text[i:i+100] for i in range(0, len(text), 100)]
                            for chunk in chunks:
                                try:
                                    pdf.multi_cell(0, 10, txt=chunk)
                                except Exception:
                                    # If a chunk causes an error, try to clean it
                                    clean_chunk = ''.join(c if ord(c) < 128 else ' ' for c in chunk)
                                    pdf.multi_cell(0, 10, txt=clean_chunk)
                            
                            pdf_filename = f"{safe_title}.pdf"
                            pdf.output(pdf_filename)
                            
                            with open(pdf_filename, "rb") as f:
                                b64 = base64.b64encode(f.read()).decode()
                            href = f'<a href="data:application/pdf;base64,{b64}" download="{pdf_filename}">Download PDF</a>'
                        
                        elif download_format == "Text":
                            b64 = base64.b64encode(text.encode('utf-8')).decode()
                            href = f'<a href="data:file/txt;base64,{b64}" download="{file_name}">Download Text</a>'
                        
                        elif download_format == "DOCX":
                            doc = docx.Document()
                            doc.add_paragraph(text)
                            docx_filename = f"{safe_title}.docx"
                            doc.save(docx_filename)
                            
                            with open(docx_filename, "rb") as f:
                                b64 = base64.b64encode(f.read()).decode()
                            href = f'<a href="data:application/vnd.openxmlformats-officedocument.wordprocessingml.document;base64,{b64}" download="{docx_filename}">Download DOCX</a>'
                        
                        st.markdown(href, unsafe_allow_html=True)
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.info("Please check if the URL is valid and accessible. If using Streamlit Cloud, make sure all dependencies are properly installed.")

# Add additional information at the bottom
st.markdown("---")
st.markdown("""
    **About this app:**
    
    This application extracts text from online articles using the newspaper3k library and performs sentiment analysis using NLTK's VADER sentiment analyzer.
    
    The sentiment scores range from 0 to 1, where higher numbers indicate stronger sentiment. The compound score is a normalized score that summarizes the overall sentiment.
""")
