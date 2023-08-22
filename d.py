import streamlit as st
import pandas as pd
import nltk
import re
import io
from newspaper import Article
import requests
from streamlit_lottie import st_lottie
from streamlit_lottie import st_lottie_spinner
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
buffer = io.BytesIO()


nltk.download('punkt')
nltk.download('stopwords')

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_url_hello = "https://lottie.host/0d49d388-9acb-492c-92c2-8dad820db057/R3WmiFyHXU.json"

lottie_hello = load_lottieurl(lottie_url_hello)
st_lottie(lottie_hello)
def get_article_text(url):
    article = Article(url)
    article.download()
    article.parse()
    return article.title, article.text

def clean_text(text, stopword_file):
    stop_words = set(stopwords.words('english'))
    with open(stopword_file, 'r') as f:
        for line in f:
            stop_words.add(line.strip())
    word_tokens = word_tokenize(text)
    filtered_text = [w for w in word_tokens if not w in stop_words]
    return filtered_text

def is_complex(word):
    if len(word) > 7:
        return True
    else:
        return False

def get_readability(text):
    sentences = sent_tokenize(text)
    words = word_tokenize(text)
    avg_sentence_length = len(words) / len(sentences)
    complex_words = [w for w in words if is_complex(w)]
    percent_complex_words = len(complex_words) / len(words) * 100
    fog_index = 0.4 * (avg_sentence_length + percent_complex_words)
    return avg_sentence_length, percent_complex_words, fog_index, len(complex_words)

def get_word_count(text):
    words = word_tokenize(text)
    words = [w for w in words if w.isalpha()]
    return len(words)

def count_syllables(word):
    vowels = "aeiouy"
    word = word.lower()
    if word[0] in vowels:
        count = 1
    else:
        count = 0
    for i in range(1, len(word)):
        if word[i] in vowels and word[i - 1] not in vowels:
            count += 1
    if word.endswith("e"):
        count -= 1
    if word.endswith("le") and len(word) > 2 and word[-3] not in vowels:
        count += 1
    if count == 0:
        count += 1
    return count

def get_syllable_count(text):
    words = word_tokenize(text)
    syllable_count = [count_syllables(w) for w in words]
    return syllable_count

def get_personal_pronouns(text):
    pattern = r'\b[Ii]|[Ww]e|[Mm]y|[Oo]urs|[Uu]s\b'
    personal_pronouns = re.findall(pattern, text)
    personal_pronouns = [w for w in personal_pronouns if w.lower() != 'us']
    return len(personal_pronouns)

def get_avg_word_length(text):
    words = word_tokenize(text)
    words = [w for w in words if w.isalpha()]
    total_chars = sum([len(w) for w in words])
    avg_word_length = total_chars / len(words)
    return avg_word_length

def get_sentiment(text, pos_dict_file, neg_dict_file, stopword_file):
    text = clean_text(text, stopword_file)
    pos_dict = set()
    neg_dict = set()
    with open(pos_dict_file, 'r') as f:
        for line in f:
            pos_dict.add(line.strip())
    with open(neg_dict_file, 'r') as f:
        for line in f:
            neg_dict.add(line.strip())
    pos_score = sum([1 for word in text if word in pos_dict])
    neg_score = sum([-1 for word in text if word in neg_dict]) * -1
    polarity_score = (pos_score - neg_score) / ((pos_score + neg_score) + 0.000001)
    subjectivity_score = (pos_score + neg_score) / (len(text) + 0.000001)
    return pos_score, neg_score, polarity_score, subjectivity_score


st.title('Sentiment Analysis')

url = st.text_input('Enter URL:')
if url:
    article = Article(url)
    article.download()
    article.parse()
    text = article.text
    title = article.title
    with open("article.txt", "w", encoding="utf-8") as f:
        f.write(title + "\n\n" + text)
     
    st.success('Article text extracted and saved successfully!')
    
    # Assuming that the functions `get_sentiment`, `get_readability`, `get_word_count`, `get_syllable_count`, `get_personal_pronouns`, and `get_avg_word_length` are defined
    pos_score, neg_score, polarity_score, subjectivity_score = get_sentiment(text, 'positive.txt', 'negative.txt', 'stopwords.txt')
    avg_sentence_length, percent_complex_words, fog_index, complex_word_count = get_readability(text)
    word_count = get_word_count(text)
    syllable_count_per_word = get_syllable_count(text)
    personal_pronouns_count = get_personal_pronouns(text)
    avg_word_length = get_avg_word_length(text)
    
    result = {
        'URL': url,
        'POSITIVE SCORE': pos_score,
        'NEGATIVE SCORE': neg_score,
        'POLARITY SCORE': polarity_score,
        'SUBJECTIVITY SCORE': subjectivity_score,
        'AVERAGE SENTENCE LENGTH': avg_sentence_length,
        'PERCENTAGE OF COMPLEX WORDS': percent_complex_words,
        'FOG INDEX': fog_index,
        'AVERAGE NUMBER OF WORDS PER SENTENCE': avg_sentence_length,
        'COMPLEX WORD COUNT': complex_word_count,
        'WORD COUNT': word_count,
        'SYLLABEL COUNT PER WORD': syllable_count_per_word,
        'PERSONAL PRONOUNS COUNT': personal_pronouns_count,
        'AVERAGE WORD LENGTH': avg_word_length
    }
    
    result_df = pd.DataFrame([result])
    st.write(result_df)
    
    with pd.ExcelWriter('result.xlsx', engine='xlsxwriter') as writer:
        result_df.to_excel(writer, sheet_name='Sheet1', index=False)

download_button = st.download_button(
    label="Download data as Excel",
    data='result.xlsx',
    file_name='result.xlsx',
    mime='application/vnd.ms-excel'     
)



                