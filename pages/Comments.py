import streamlit as st
import pandas as pd
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from wordcloud import WordCloud
import matplotlib.pyplot as plt

st.set_page_config(page_title="Comments & Analysis", layout="wide")

st.title("💬 HELB Service Delivery – Comments & Insights")

# --- Load data ---
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

# --- Rename columns for simplicity ---
df.columns = [
    "Enablers",
    "Internal_Systems",
    "Leadership_Influence",
    "Improvement_Idea",
    "Questions_to_CEO"
]

# --- Sentiment Analysis ---
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

# --- Categorize Questions (CEO / Management / General) ---
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

# --- Text Clustering (Themes) ---
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
        cluster_counts = df["Theme_Cluster"].value_counts().sort_index()
        st.bar_chart(cluster_counts)
else:
    st.info("No enabler text available for clustering.")

# --- Word Cloud for All Comments ---
st.subheader("☁️ Word Cloud – Common Terms")
text_data = " ".join(df["Improvement_Idea"].dropna().astype(str))
if text_data.strip():
    wc = WordCloud(width=800, height=400, background_color="white",
                   colormap="viridis").generate(text_data)
    fig, ax = plt.subplots()
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)
else:
    st.info("No text available for generating word cloud.")

# --- Sentiment Summary ---
st.subheader("📊 Sentiment Distribution")
sent_counts = df["Sentiment"].value_counts()
st.bar_chart(sent_counts)

# --- View and Filter Comments ---
st.subheader("🗒️ Explore Individual Comments")
sentiment_filter = st.selectbox("Filter by Sentiment", ["All", "Positive", "Neutral", "Negative"])
category_filter = st.selectbox("Filter by Question Category", ["All", "CEO", "Management", "General"])

filtered = df.copy()
if sentiment_filter != "All":
    filtered = filtered[filtered["Sentiment"] == sentiment_filter]
if category_filter != "All":
    filtered = filtered[filtered["Question_Category"] == category_filter]

st.dataframe(filtered[["Improvement_Idea", "Sentiment", "Questions_to_CEO", "Question_Category"]])

# --- Download Option ---
st.download_button(
    label="📥 Download Full Analysis (Excel)",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name="HELB_Service_Delivery_Analysis.csv",
    mime="text/csv"
)

st.caption("Data source: HELB Internal Service Delivery Survey – Processed with Streamlit Cloud")
