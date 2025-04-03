import streamlit as st
from newspaper import Article
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd
import requests
from streamlit_lottie import st_lottie
import base64
from fpdf import FPDF
import docx

# Download necessary NLTK data
nltk.download('vader_lexicon')
nltk.download('stopwords')
nltk.download('punkt')

# Load Lottie Animation
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_url_hello = "https://lottie.host/0d49d388-9acb-492c-92c2-8dad820db057/R3WmiFyHXU.json"
lottie_hello = load_lottieurl(lottie_url_hello)
st_lottie(lottie_hello)

# User Input
url = st.text_input('Enter the URL of the article:')

if url:
    # Extract article information
    article = Article(url)
    article.download()
    article.parse()

    # Handle missing title issue
    safe_title = article.title.replace("/", "_").replace("\\", "_") if article.title else "article"
    file_name = f"{safe_title}.txt"

    st.write(f"**Title:** {article.title}")
    st.write(f"**Text:** {article.text[:1000]}...")  # Show only first 1000 characters for preview

    # Save article text
    with open(file_name, "w", encoding="utf-8") as f:
        f.write(article.text)

    # Perform sentiment analysis
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(article.text)
    tokens = [word for word in tokens if word.isalpha() and word.lower() not in stop_words]

    sid = SentimentIntensityAnalyzer()
    sentiment_scores = sid.polarity_scores(' '.join(tokens))

    # Create DataFrame for Sentiment Analysis
    df = pd.DataFrame({
        'URL': [url],
        'Article Title': [article.title],
        'Positive Score': [sentiment_scores['pos']],
        'Negative Score': [sentiment_scores['neg']],
        'Neutral Score': [sentiment_scores['neu']],
        'Subjectivity Score': [sentiment_scores['pos'] + sentiment_scores['neg']]
    })
    st.write(df)

    # Download options
    download_format = st.selectbox("Select download format", ["PDF", "Text", "DOCX"])

    if st.button("Download Article Text"):
        if download_format == "PDF":
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.multi_cell(0, 10, txt=article.text)
            pdf_file = f"{file_name}.pdf"
            pdf.output(pdf_file)
            
            with open(pdf_file, "rb") as f:
                st.download_button("Download PDF", f, file_name="article.pdf", mime="application/pdf")

        elif download_format == "Text":
            st.download_button("Download Text", article.text, file_name="article.txt", mime="text/plain")

        elif download_format == "DOCX":
            doc = docx.Document()
            doc.add_paragraph(article.text)
            docx_file = f"{file_name}.docx"
            doc.save(docx_file)

            with open(docx_file, "rb") as f:
                st.download_button("Download DOCX", f, file_name="article.docx", mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")
