import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import plotly.express as px

# =============================
# PAGE CONFIGURATION
# =============================
st.set_page_config(page_title="HELB Service Delivery Dashboard", layout="wide")

# Apply custom color styling
st.markdown("""
    <style>
    .main {
        background-color: #FFFFFF;
    }
    h1, h2, h3, h4 {
        color: #002147;
    }
    .metric-card {
        background-color: #F5F5F5;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        border: 2px solid #008000;
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
    # Change to your raw GitHub Excel URL if not using local file
    url = "Issues Influencing Excellent Service Delivery(1-45).xlsx"
    return pd.read_excel(url)

df = load_data()

# =============================
# PAGE HEADER
# =============================
st.title("📊 HELB Service Delivery Dashboard")
st.markdown(
    "<h4 style='color:#008000;'>Visual Insights from the Internal Service Delivery Survey</h4>",
    unsafe_allow_html=True
)

# =============================
# METRIC CARDS
# =============================
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Responses", len(df))
with col2:
    st.metric("Unique Departments", df["Department"].nunique() if "Department" in df.columns else "N/A")
with col3:
    st.metric("Unique Respondents", df["Respondent ID"].nunique() if "Respondent ID" in df.columns else "N/A")

st.divider()

# =============================
# CHARTS
# =============================

# Example categorical breakdown (department or rating)
if "Department" in df.columns:
    st.subheader("📈 Responses by Department")
    dept_counts = df["Department"].value_counts().reset_index()
    dept_counts.columns = ["Department", "Count"]
    fig = px.bar(
        dept_counts,
        x="Department",
        y="Count",
        color="Department",
        color_discrete_sequence=["#008000", "#FFD700", "#002147", "#006400", "#DAA520"],
        title="Responses by Department"
    )
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# =============================
# SENTIMENT VISUALIZATION
# =============================
if "Sentiment" in df.columns:
    st.subheader("😊 Sentiment Distribution")
    fig2 = px.pie(
        df,
        names="Sentiment",
        color="Sentiment",
        color_discrete_map={
            "Positive": "#008000",
            "Neutral": "#FFD700",
            "Negative": "#002147"
        },
        title="Overall Sentiment from Comments"
    )
    st.plotly_chart(fig2, use_container_width=True)

st.divider()

# =============================
# WORD CLOUD
# =============================
st.subheader("☁️ Commonly Used Words in Comments")

# Determine the column containing text
text_col = None
for col in df.columns:
    if "comment" in col.lower() or "feedback" in col.lower() or "issue" in col.lower():
        text_col = col
        break

if text_col:
    text = " ".join(str(i) for i in df[text_col].dropna())
    wordcloud = WordCloud(width=900, height=400, background_color="white", colormap="Dark2").generate(text)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    plt.title("Most Frequent Words in Survey Comments", color="#002147", fontsize=14)
    st.pyplot(fig)
else:
    st.info("No text column found for word cloud generation.")

# =============================
# FOOTER
# =============================
st.markdown("---")
st.caption("Dashboard colors: Green, Gold, White, and Dark Blue — aligned to HELB branding.")

