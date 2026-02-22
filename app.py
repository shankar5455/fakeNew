# =====================================
# 🎨 Fake News Detection Streamlit App
# (FINAL PRO VERSION)
# =====================================

import streamlit as st
import pickle
from textblob import TextBlob
import random
import time
import os

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="wide"
)

# ===============================
# 🎨 MODERN UI CSS
# ===============================
st.markdown("""
<style>
html, body, [class*="css"] { font-family: Arial; }

.app-title {
    text-align:center;
    padding:2rem;
    background:linear-gradient(135deg,#667eea,#764ba2);
    border-radius:20px;
    margin-bottom:2rem;
}
.app-title h1 {color:white;}

.result-card {
    border-radius:15px;
    padding:1.5rem;
    margin:1rem 0;
    color:white;
    text-align:center;
    font-size:1.3rem;
}

.fake-news {background:linear-gradient(135deg,#ff6b6b,#ee5253);}
.real-news {background:linear-gradient(135deg,#51cf66,#37b24d);}

.sentiment-card {
    background:#ffd93d;
    padding:1rem;
    border-radius:15px;
    text-align:center;
    font-weight:bold;
}
</style>
""", unsafe_allow_html=True)

# ===============================
# HEADER
# ===============================
st.markdown("""
<div class="app-title">
<h1>📰 Fake News Detection System</h1>
<p style="color:white;">News Authenticity & Sentiment Analyzer</p>
</div>
""", unsafe_allow_html=True)

# ===============================
# LOAD SAVED MODEL SAFELY
# ===============================
@st.cache_resource
def load_model():

    model_path = "fake_news_model.pkl"
    vectorizer_path = "tfidf_vectorizer.pkl"

    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        st.error("❌ Model files not found!")
        st.warning("👉 Keep fake_news_model.pkl & tfidf_vectorizer.pkl inside project folder.")
        st.stop()

    model = pickle.load(open(model_path, "rb"))
    vectorizer = pickle.load(open(vectorizer_path, "rb"))

    return model, vectorizer

model, vectorizer = load_model()

# ===============================
# SENTIMENT FUNCTION
# ===============================
def get_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity

    if polarity > 0.3:
        return "Very Positive 🌟"
    elif polarity > 0:
        return "Positive 😊"
    elif polarity < -0.3:
        return "Very Negative 💢"
    elif polarity < 0:
        return "Negative 😠"
    else:
        return "Neutral 😐"

# ===============================
# SIDEBAR STATS
# ===============================
with st.sidebar:
    st.title("📊 Stats")

    if "total" not in st.session_state:
        st.session_state.total = 0
        st.session_state.fake = 0
        st.session_state.real = 0

    st.metric("Total Checks", st.session_state.total)
    st.metric("Fake News", st.session_state.fake)
    st.metric("Real News", st.session_state.real)

    if st.button("Reset Stats"):
        st.session_state.total = 0
        st.session_state.fake = 0
        st.session_state.real = 0
        st.rerun()

# ===============================
# MAIN INPUT AREA
# ===============================
col1, col2, col3 = st.columns([1,2,1])

with col2:

    sample_news = [
        "Scientists discover new renewable energy source",
        "Aliens officially landed in New York yesterday",
        "Government announces new education reforms",
        "Miracle drink cures all diseases instantly"
    ]

    if st.button("🎲 Try Sample News"):
        st.session_state.sample = random.choice(sample_news)

    news_input = st.text_area(
        "✏️ Enter News Text",
        value=st.session_state.get("sample",""),
        height=150
    )

    predict_button = st.button("🔍 ANALYZE NEWS", use_container_width=True)

# ===============================
# PREDICTION LOGIC
# ===============================
if predict_button:

    if news_input.strip()=="":
        st.warning("⚠️ Please enter news text.")
    else:
        with st.spinner("🤖 AI is analyzing..."):
            time.sleep(1)

            news_vec = vectorizer.transform([news_input])
            prediction = model.predict(news_vec)[0]

            # REAL confidence score
            confidence = int(model.predict_proba(news_vec)[0].max()*100)

            # Update stats
            st.session_state.total += 1

            res1,res2 = st.columns(2)

            with res1:
                if prediction==1:
                    st.session_state.fake+=1
                    st.markdown(f"""
                    <div class="result-card fake-news">
                    ❌ FAKE NEWS <br> Confidence: {confidence}%
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.session_state.real+=1
                    st.markdown(f"""
                    <div class="result-card real-news">
                    ✅ REAL NEWS <br> Confidence: {confidence}%
                    </div>
                    """, unsafe_allow_html=True)

            with res2:
                sentiment = get_sentiment(news_input)
                st.markdown(f"""
                <div class="sentiment-card">
                💡 Sentiment: {sentiment}
                </div>
                """, unsafe_allow_html=True)

            st.divider()

            c1,c2,c3 = st.columns(3)
            c1.info(f"📝 Words: {len(news_input.split())}")
            c2.info(f"📊 Characters: {len(news_input)}")
            c3.info(f"🤖 Model: Logistic Regression")