import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import plotly.express as px

# =============================
# PAGE CONFIGURATION
# =============================
st.set_page_config(page_title="HELB Service Delivery Dashboard", layout="wide")

# Apply custom styling
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
        border-left: 4px solid #008000;
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

    # Rename columns to shorter logical names
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
# HEADER
# =============================
st.title("📊 HELB Service Delivery Dashboard")
st.markdown(
    "<h4 style='color:#008000;'>Visual Insights from the Internal Service Delivery Survey</h4>",
    unsafe_allow_html=True
)

# =============================
# METRICS
# =============================
col1, col2, col3 = st.columns(3)
st.markdown("### 📈 Survey Overview")

with col1:
    st.metric("Total Responses", len(df))
with col2:
    st.metric("Questions Analyzed", len(df.columns))
with col3:
    st.metric("Unique Text Entries", df.apply(lambda x: x.notna().sum()).sum())

st.divider()

# =============================
# WORD CLOUDS PER QUESTION
# =============================

st.subheader("☁️ Key Themes from Responses")

text_columns = [
    "Top_Issues",
    "Processes_Systems",
    "Leadership_Influence",
    "Change_Idea",
    "Questions_to_CEO"
]

for col in text_columns:
    if col in df.columns:
        st.markdown(f"#### {col.replace('_', ' ')}")
        text = " ".join(str(t) for t in df[col].dropna())
        if len(text.strip()) > 0:
            wordcloud = WordCloud(
                width=900,
                height=400,
                background_color="white",
                colormap="Dark2"
            ).generate(text)
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.imshow(wordcloud, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig)
        else:
            st.info(f"No text available for {col}")

        st.divider()

# =============================
# SENTIMENT PLACEHOLDER
# =============================

st.subheader("😊 Sentiment Overview (Coming Next)")
st.info("Sentiment analysis and clustering will be included in the Comments Analysis page.")

# =============================
# FOOTER
# =============================
st.markdown("---")
st.caption("Dashboard colors: Green, Gold, White, and Dark Blue — aligned to HELB branding.")
