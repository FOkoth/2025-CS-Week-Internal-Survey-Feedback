import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from textblob import TextBlob

# =============================
# PAGE CONFIGURATION
# =============================
st.set_page_config(page_title="HELB Comments & Sentiment Analysis", layout="wide")

# Apply HELB color theme
st.markdown("""
    <style>
    .main {
        background-color: #FFFFFF;
    }
    h1, h2, h3, h4 {
        color: #002147;
    }
    .stMetric {
        background-color: #FFFFFF !important;
        border-left: 4px solid #FFD700;
        padding: 10px;
        border-radius: 8px;
    }
    </style>
""", unsafe_allow_html=True)

# =============================
# LOAD DATA
# =============================
@st.cache_data
def load_data():
    url = "Issues Influencing Excellent Service Delivery.xlsx"
    df = pd.read_excel(url)
    rename_map = {
        "In your view, what are the top three issues that enable staff to deliver excellent service to both internal and external customers?": "Top_Issues",
        "How well do our current internal processes and systems support timely and effective customer service?": "Processes_Systems",
        "To what extent does the leadership communication and support influence your ability to deliver effective service to the customers?": "Leadership_Influence",
        "If you were to make a change, what is the one thing you would do to ascertain excellent service delivery?": "Change_Idea",
        "If you had an opportunity to ask the CEO and Top Management two major questions, what would they be? Please list the two questions:": "Questions_to_CEO"
    }
    df = df.rename(columns=rename_map)
    return df

df = load_data()

# =============================
# PAGE TITLE
# =============================
st.title("💬 HELB Comments & Analysis")
st.markdown("<h4 style='color:#008000;'>Text Insights, Sentiment & Thematic Clustering</h4>", unsafe_allow_html=True)
st.divider()

# =============================
# COMBINE TEXT COLUMNS
# =============================
text_columns = ["Top_Issues", "Processes_Systems", "Leadership_Influence", "Change_Idea", "Questions_to_CEO"]
df["All_Comments"] = df[text_columns].astype(str).agg(" ".join, axis=1)

# =============================
# SENTIMENT ANALYSIS
# =============================
def get_sentiment(text):
    if not text or text.strip() == "":
        return "Neutral"
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.1:
        return "Positive"
    elif polarity < -0.1:
        return "Negative"
    else:
        return "Neutral"

df["Sentiment"] = df["All_Comments"].apply(get_sentiment)

# Sentiment counts
sentiment_counts = df["Sentiment"].value_counts().reset_index()
sentiment_counts.columns = ["Sentiment", "Count"]

st.subheader("😊 Sentiment Overview")
fig = px.bar(
    sentiment_counts,
    x="Sentiment",
    y="Count",
    color="Sentiment",
    color_discrete_map={"Positive": "#008000", "Neutral": "#FFD700", "Negative": "#002147"},
    title="Overall Sentiment of Survey Responses"
)
st.plotly_chart(fig, use_container_width=True)
st.divider()

# =============================
# QUESTION TYPE CLASSIFICATION
# =============================
st.subheader("🏢 Question Categorization")

def classify_question(text):
    if not text or text.strip() == "":
        return "General"
    text = text.lower()
    if "ceo" in text or "chief" in text:
        return "CEO"
    elif "management" in text or "leadership" in text:
        return "Management"
    else:
        return "General"

df["Question_Category"] = df["Questions_to_CEO"].apply(classify_question)

category_counts = df["Question_Category"].value_counts().reset_index()
category_counts.columns = ["Category", "Count"]

fig2 = px.pie(
    category_counts,
    names="Category",
    values="Count",
    color="Category",
    color_discrete_map={"CEO": "#008000", "Management": "#FFD700", "General": "#002147"},
    title="Classification of Questions by Target Audience"
)
st.plotly_chart(fig2, use_container_width=True)
st.divider()

# =============================
# TEXT CLUSTERING (Themes)
# =============================
st.subheader("📚 Thematic Clustering of Comments")

# Prepare text data
comments = df["All_Comments"].dropna()
if len(comments) > 5:
    vectorizer = TfidfVectorizer(stop_words="english", max_features=1000)
    X = vectorizer.fit_transform(comments)

    # KMeans clustering into 5 themes
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    df_clusters = df.loc[comments.index].copy()
    df_clusters["Theme_Cluster"] = kmeans.fit_predict(X)

    st.write("Top words in each theme:")
    terms = vectorizer.get_feature_names_out()
    order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]

    for i in range(5):
        top_terms = [terms[ind] for ind in order_centroids[i, :10]]
        st.markdown(f"**Theme {i+1}:** {' | '.join(top_terms)}")

    # Display clustered data
    st.dataframe(df_clusters[["All_Comments", "Theme_Cluster"]])
else:
    st.info("Not enough text responses to perform clustering.")
st.divider()

# =============================
# DATA TABLE
# =============================
st.subheader("📄 All Comments with Analysis")
st.dataframe(df[["Top_Issues", "Processes_Systems", "Leadership_Influence", "Change_Idea", "Questions_to_CEO", "Sentiment", "Question_Category"]])

# =============================
# FOOTER
# =============================
st.markdown("---")
st.caption("Automated sentiment & thematic clustering • HELB Service Delivery Survey • Green | Gold | White | Dark Blue")

