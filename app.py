import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# -------------------- NLTK Setup --------------------
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# -------------------- Initialize --------------------

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

# -------------------- Text Preprocessing --------------------

def transform_text(text):
    text = text.lower()
    words = nltk.word_tokenize(text)

    filtered_words = []

    for word in words:
        if word.isalnum() and word not in stop_words:
            filtered_words.append(ps.stem(word))

    return " ".join(filtered_words)

# -------------------- Load Model (Cached) --------------------

@st.cache_resource
def load_models():
    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))
    return tfidf, model

tfidf, model = load_models()

# -------------------- UI --------------------

st.title("📩 Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):

    if input_sms.strip() == "":
        st.warning("Please enter a message first.")
    else:
        transformed_sms = transform_text(input_sms)
        vector_input = tfidf.transform([transformed_sms])
        result = model.predict(vector_input)[0]
        probability = model.predict_proba(vector_input)[0]

        if result == 1:
            st.header("🚨 Spam")
        else:
            st.header("✅ Not Spam")

        st.write(f"Confidence: {max(probability)*100:.2f}%")