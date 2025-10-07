import streamlit as st
import pandas as pd
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="Comments & Insights", layout="wide")
st.title("💬 HELB Service Delivery – Comments & Insights")

# Apply custom color theme
st.markdown(
    """
    <style>
    body {
        background-color: #f7fafc;
    }
    .stApp {
        background-color: white;
    }
    h1, h2, h3 {
        color: #003366; /* Dark Blue */
    }
    .stSuccess {
        color: #006400; /* Green */
    }
    .stError {
        color: #b22222; /* Red */
    }
    .stDownloadButton>button {
        background-color: #006400 !important;
        color: white !important;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------------
# LOAD DATA
# -------------------------------
@st.cache_data
def load_data():
    url = "Issues Influencing Excellent Service Delivery.xlsx"
    return pd.read_excel(url)

try:
    df = load_data()
    st.success("✅ Data loaded successfully!")
except Exception as e:
    st.error(f"❌ Failed to load data: {e}")
    st.stop()

# Rename columns for consistency
df.columns = [
    "Enablers",
    "Internal_Systems",
    "Leadership_Influence",
    "Improvement_Idea",
    "Questions_to_CEO"
]

# -------------------------------
# SENTIMENT ANALYSIS
# -------------------------------
def get_sentiment(text):
    if pd.isna(text) or str(text).strip() == "":
        return "Neutral"
    score = TextBlob(str(text)).sentiment.polarity
    if score > 0.1:
        return "Positive"
    elif score < -0.1:
        return "Negative"
    else:
        return "Neutral"

df["Sentiment"] = df["Improvement_Idea"].apply(get_sentiment)

# -------------------------------
# CLASSIFY QUESTIONS
# -------------------------------
def classify_question(text):
    if pd.isna(text):
        return "General"
    text = str(text).strip().lower()
    if text == "":
        return "General"
    if "ceo" in text or "chief" in text:
        return "CEO"
    elif "management" in text or "leadership" in text:
        return "Management"
    else:
        return "General"

df["Question_Category"] = df["Questions_to_CEO"].apply(classify_question)

# -------------------------------
# CLUSTER THEMES
# -------------------------------
st.subheader("🔍 Theme Clustering (based on Enablers)")

corpus = df["Enablers"].dropna().astype(str).tolist()
if len(corpus) > 0:
    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(corpus)
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X)
    df.loc[df["Enablers"].notna(), "Theme_Cluster"] = clusters

    col1, col2 = st.columns(2)
    with col1:
        st.write("**Cluster Sample:**")
        st.dataframe(df[["Enablers", "Theme_Cluster"]].head(10))
    with col2:
        st.bar_chart(df["Theme_Cluster"].value_counts().sort_index())
else:
    st.info("No enabler text available for clustering.")

# -------------------------------
# WORD CLOUD (SMART PHRASE DETECTION)
# -------------------------------
st.subheader("☁️ Word Cloud – Common Terms & Phrases")

text_series = df["Improvement_Idea"].dropna().astype(str)
if not text_series.empty:
    # Create bigrams and trigrams
    vectorizer = CountVectorizer(ngram_range=(1, 3), stop_words="english").fit(text_series)
    bag_of_words = vectorizer.transform(text_series)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
    sorted_words = sorted(words_freq, key=lambda x: x[1], reverse=True)
    word_freq_dict = dict(sorted_words[:100])  # top 100 phrases

    wc = WordCloud(
        width=900, height=450,
        background_color="white",
        colormap="viridis"
    ).generate_from_frequencies(word_freq_dict)

    fig, ax = plt.subplots()
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)
else:
    st.info("No text available for generating word cloud.")

# -------------------------------
# SENTIMENT SUMMARY
# -------------------------------
st.subheader("📊 Sentiment Distribution")
sent_counts = df["Sentiment"].value_counts()
st.bar_chart(sent_counts)

# -------------------------------
# FILTER & VIEW COMMENTS
# -------------------------------
st.subheader("🗒️ Explore Individual Comments")

sentiment_filter = st.selectbox("Filter by Sentiment", ["All", "Positive", "Neutral", "Negative"])
category_filter = st.selectbox("Filter by Question Category", ["All", "CEO", "Management", "General"])

filtered = df.copy()
if sentiment_filter != "All":
    filtered = filtered[filtered["Sentiment"] == sentiment_filter]
if category_filter != "All":
    filtered = filtered[filtered["Question_Category"] == category_filter]

st.dataframe(filtered[["Improvement_Idea", "Sentiment", "Questions_to_CEO", "Question_Category"]])

# -------------------------------
# DOWNLOAD BUTTON
# -------------------------------
st.download_button(
    label="📥 Download Full Analysis (CSV)",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name="HELB_Service_Delivery_Analysis.csv",
    mime="text/csv"
)

st.caption("Data source: HELB Internal Service Delivery Survey – Processed with Streamlit Cloud")
