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
nltk.download('vader_lexicon')
nltk.download('stopwords')
nltk.download('punkt')
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()
lottie_url_hello = "https://lottie.host/0d49d388-9acb-492c-92c2-8dad820db057/R3WmiFyHXU.json"
lottie_hello = load_lottieurl(lottie_url_hello)
st_lottie(lottie_hello)
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
# Download the extracted text of the article
download_format = st.selectbox("Select download format", ["PDF", "Text", "DOCX"])
download_button = st.button("Download Article Text")
if download_button:
    with open(file_name, "r") as f:
        text = f.read()
        if download_format == "PDF":
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.multi_cell(0, 10, txt=text)
            pdf.output(file_name + ".pdf")
            with open(file_name + ".pdf", "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            href = f'<a href="data:application/pdf;base64,{b64}" download="{file_name}.pdf">Download</a>'
        elif download_format == "Text":
            b64 = base64.b64encode(text.encode()).decode()
            href = f'<a href="data:file/txt;base64,{b64}" download="{file_name}.txt">Download</a>'
        elif download_format == "DOCX":
            doc = docx.Document()
            doc.add_paragraph(text)
            doc.save(file_name + ".docx")
            with open(file_name + ".docx", "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            href = f'<a href="data:application/vnd.openxmlformats-officedocument.wordprocessingml.document;base64,{b64}" download="{file_name}.docx">Download</a>'
        st.markdown(href, unsafe_allow_html=True)
