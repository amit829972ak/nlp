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
import os
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
# Set up local nltk_data path
nltk_data_dir = os.path.join(os.path.dirname(__file__), "nltk_data")
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.append(nltk_data_dir)

# Download required NLTK data to local path
def download_nltk_data():
    required_nltk_data = [
        'vader_lexicon',
        'stopwords',
        'punkt',
        'averaged_perceptron_tagger',
        'wordnet'
    ]
    
    for resource in required_nltk_data:
        try:
            nltk.download(resource, download_dir=nltk_data_dir)
        except Exception as e:
            st.error(f"Error downloading NLTK resource {resource}: {str(e)}")
            return False
    return True

# Initialize NLTK data
if not download_nltk_data():
    st.error("Failed to download required NLTK data. Please check your internet connection and try again.")
    st.stop()

# Load Lottie animation from URL
def load_lottieurl(url: str):
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception as e:
        st.error(f"Error loading Lottie animation: {str(e)}")
        return None

# Display animation
lottie_url_hello = "https://lottie.host/0d49d388-9acb-492c-92c2-8dad820db057/R3WmiFyHXU.json"
lottie_hello = load_lottieurl(lottie_url_hello)
if lottie_hello:
    st_lottie(lottie_hello)

# Streamlit UI
st.title("📰 Article Analysis Tool")
url = st.text_input('🔗 Enter the URL of the article:')

if url:
    try:
        # Extract article information
        article = Article(url)
        article.download()
        article.parse()
        
        if not article.text:
            st.error("❌ Could not extract text from the article. Please check the URL and try again.")
        else:
            st.subheader("📄 Article Title")
            st.write(article.title)

            st.subheader("📝 Article Text")
            st.write(article.text)

            # Save article text to file
            file_name = f'{article.title}.txt'
            try:
                with open(file_name, 'w', encoding='utf-8') as f:
                    f.write(article.text)
            except Exception as e:
                st.error(f"⚠️ Error saving article text: {str(e)}")

            # Perform sentiment analysis
            try:
                stop_words = set(stopwords.words('english'))
                tokens = word_tokenize(article.text)
                tokens = [word for word in tokens if word.isalpha()]
                tokens = [word for word in tokens if word.lower() not in stop_words]

                sid = SentimentIntensityAnalyzer()
                sentiment_scores = sid.polarity_scores(' '.join(tokens))

                # Create DataFrame
                data = {
                    'URL': [url],
                    'Article Title': [article.title],
                    'Positive Score': [sentiment_scores['pos']],
                    'Negative Score': [sentiment_scores['neg']],
                    'Neutral Score': [sentiment_scores['neu']],
                    'Subjectivity Score': [sentiment_scores['pos'] + sentiment_scores['neg']]
                }
                df = pd.DataFrame(data)
                st.subheader("📊 Sentiment Analysis")
                st.dataframe(df)

                # Download options
                download_format = st.selectbox("💾 Select download format", ["PDF", "Text", "DOCX"])
                download_button = st.button("⬇️ Download Article Text")

                if download_button:
                    try:
                        with open(file_name, "r", encoding='utf-8') as f:
                            text = f.read()
                            if download_format == "PDF":
                                pdf = FPDF()
                                pdf.add_page()
                                pdf.set_font("Arial", size=12)
                                pdf.multi_cell(0, 10, txt=text)
                                pdf.output(file_name + ".pdf")
                                with open(file_name + ".pdf", "rb") as f:
                                    b64 = base64.b64encode(f.read()).decode()
                                href = f'<a href="data:application/pdf;base64,{b64}" download="{file_name}.pdf">Download PDF</a>'
                            elif download_format == "Text":
                                b64 = base64.b64encode(text.encode()).decode()
                                href = f'<a href="data:file/txt;base64,{b64}" download="{file_name}.txt">Download Text</a>'
                            elif download_format == "DOCX":
                                doc = docx.Document()
                                doc.add_paragraph(text)
                                doc.save(file_name + ".docx")
                                with open(file_name + ".docx", "rb") as f:
                                    b64 = base64.b64encode(f.read()).decode()
                                href = f'<a href="data:application/vnd.openxmlformats-officedocument.wordprocessingml.document;base64,{b64}" download="{file_name}.docx">Download DOCX</a>'
                            st.markdown(href, unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"❌ Error during download: {str(e)}")
            except Exception as e:
                st.error(f"❌ Error during sentiment analysis: {str(e)}")

    except Exception as e:
        st.error(f"❌ Error processing article: {str(e)}")
        st.error("Please check the URL and try again.")
