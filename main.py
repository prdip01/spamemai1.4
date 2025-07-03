import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import streamlit as st
import requests
from streamlit_lottie import st_lottie

@st.cache_data
def load_data():
    data = pd.read_csv("spam.csv")
    data.drop_duplicates(inplace=True)
    data["Category"] = data["Category"].replace(["ham", "spam"], ["Not Spam", "Spam"])
    return data

data = load_data()
X = data["Message"]
y = data["Category"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(stop_words="english")
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

def predict(message):
    transformed = vectorizer.transform([message])
    prediction = model.predict(transformed)[0]
    confidence = model.predict_proba(transformed).max()
    return prediction, round(confidence * 100, 2)

@st.cache_data
def load_lottie_url(url):
    try:
        r = requests.get(url)
        if r.status_code == 200:
            return r.json()
    except:
        return None

spam_lottie = load_lottie_url("https://lottie.host/2f0f9f16-35f0-4c44-92f0-03f39fcb7f2d/knrKDjHp0T.json")
clean_lottie = load_lottie_url("https://lottie.host/8607d8e5-9267-47be-8fd8-56edb5efb3ef/F4Hn9NDuSl.json")

st.set_page_config(page_title="Spam Detector", page_icon="📩", layout="centered")
st.title("📩 Smart Spam Detection App")

user_input = st.text_area("✉️ Enter your message:")

if st.button("🚀 Analyze"):
    if not user_input.strip():
        st.warning("Please enter a message.")
    else:
        prediction, confidence = predict(user_input)
        if prediction == "Spam":
            st.error(f"🚫 SPAM with {confidence}% confidence.")
            if spam_lottie:
                st_lottie(spam_lottie, height=200)
        else:
            st.success(f"✅ Not Spam with {confidence}% confidence.")
            if clean_lottie:
                st_lottie(clean_lottie, height=200)