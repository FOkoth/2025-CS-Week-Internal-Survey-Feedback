# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from io import BytesIO

st.set_page_config(page_title="Survey Analysis - Service Delivery", layout="wide")

# ---------- Helper functions ----------
@st.cache_data
def load_excel(file_or_path):
    if hasattr(file_or_path, "read"):  # uploaded file-like
        return pd.read_excel(file_or_path)
    else:
        return pd.read_excel(file_or_path)

@st.cache_data
def detect_text_columns(df):
    # choose object columns (likely text)
    text_cols = df.select_dtypes(include=["object"]).columns.tolist()
    return text_cols

analyzer = SentimentIntensityAnalyzer()

@st.cache_data
def sentiment_label(text):
    if not isinstance(text, str) or text.strip() == "":
        return None, 0.0
    score = analyzer.polarity_scores(text)["compound"]
    if score >= 0.05:
        label = "Positive"
    elif score <= -0.05:
        label = "Negative"
    else:
        label = "Neutral"
    return label, score

def make_wordcloud(text, max_words=150):
    if not text or len(text.strip()) == 0:
        return None
    wc = WordCloud(width=1200, height=500, background_color="white",
                   stopwords=STOPWORDS, max_words=max_words)
    wc.generate(text)
    return wc

def parse_multi_questions(text):
    """Attempt to split multiple questions listed in one cell."""
    if not isinstance(text, str) or text.strip() == "":
        return []
    # split on newlines, semicolons, ' / ', ' and '
    parts = []
    for sep in ["\n", ";", " / ", "/", "  -  ", " - ", " and ", " & "]:
        if sep in text:
            parts = [p.strip() for p in text.split(sep) if p.strip()]
            break
    if not parts:
        # fallback: split into sentences
        parts = [s.strip() for s in text.split(".") if s.strip()]
    return parts

# ---------- UI: Upload or use embedded file ----------
st.sidebar.title("Data / Upload")
use_local = st.sidebar.checkbox("Use static file in repo (data/Issues...xlsx)", value=True)
uploaded_file = None
local_path = "data/Issues Influencing Excellent Service Delivery(1-45).xlsx"

if use_local:
    try:
        df = load_excel(local_path)
        st.sidebar.success(f"Loaded local data: {local_path}")
    except Exception as e:
        st.sidebar.warning("Could not load local file. Please upload using the uploader below.")
        uploaded_file = st.sidebar.file_uploader("Upload Excel file", type=["xlsx"])
        if uploaded_file:
            df = load_excel(uploaded_file)
else:
    uploaded_file = st.sidebar.file_uploader("Upload Excel file", type=["xlsx"])
    if uploaded_file:
        df = load_excel(uploaded_file)
    else:
        st.sidebar.info("Please upload the Excel file (or toggle 'Use static file' if you put the file in the repo).")
        st.stop()

# Quick check
st.sidebar.write(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")

# ---------- Detect text columns and suggest defaults ----------
text_cols = detect_text_columns(df)
# heuristics for defaults
col_name_lower = [c.lower() for c in df.columns.tolist()]

def find_col_by_keywords(keywords):
    for i, name in enumerate(col_name_lower):
        if any(kw in name for kw in keywords):
            return df.columns[i]
    return None

default_issues_col = find_col_by_keywords(["top three", "top 3", "issues", "enable staff"])
default_change_col = find_col_by_keywords(["one thing", "make a change", "if you were to make a change"])
default_ceo_col = find_col_by_keywords(["ceo", "top management", "top management", "questions", "ask the ceo"])
default_free_text = default_issues_col or default_change_col or (text_cols[0] if text_cols else None)

st.sidebar.subheader("Text column selection (auto-detected, change if needed)")
text_col = st.sidebar.selectbox("Main open-ended column (for general analysis / wordcloud):",
                                options=text_cols, index=(text_cols.index(default_free_text) if default_free_text in text_cols else 0))
ceo_col = st.sidebar.selectbox("CEO / Top Management questions column (if present):",
                               options=[None] + text_cols, index=(text_cols.index(default_ceo_col)+1 if default_ceo_col in text_cols else 0))
change_col = st.sidebar.selectbox("If you were to make a change... column (suggestion):",
                                 options=[None] + text_cols, index=(text_cols.index(default_change_col)+1 if default_change_col in text_cols else 0))

# ---------- Precompute sentiment and combined text ----------
# Create a column with the selected text joined if user wants combined analysis
combine_cols = st.sidebar.multiselect("Optional: combine multiple text columns for analysis (recommended: include CEO/questions & change suggestions)",
                                      options=text_cols, default=[text_col] + ([ceo_col] if ceo_col else []) + ([change_col] if change_col else []))

if combine_cols:
    df["_combined_text"] = df[combine_cols].fillna("").agg(" ".join, axis=1)
else:
    df["_combined_text"] = df[text_col].fillna("").astype(str)

# compute sentiment scores
if "_sentiment_compound" not in df.columns:
    # compute row-wise
    labels = []
    scores = []
    for t in df["_combined_text"].astype(str):
        lbl, sc = sentiment_label(t)
        labels.append(lbl)
        scores.append(sc)
    df["_sentiment_label"] = labels
    df["_sentiment_compound"] = scores

# ---------- NAVIGATION ----------
st.title("Service Delivery Survey — Streamlit Analysis")
page = st.sidebar.radio("Go to", ["Dashboard", "Comments & Themes", "Settings & Download"])

# ---------- PAGE: Dashboard ----------
if page == "Dashboard":
    st.header("Dashboard — high level")
    c1, c2, c3 = st.columns([1,1,2])
    # summary cards
    total = df.shape[0]
    pos = df["_sentiment_label"].value_counts().get("Positive", 0)
    neu = df["_sentiment_label"].value_counts().get("Neutral", 0)
    neg = df["_sentiment_label"].value_counts().get("Negative", 0)
    c1.metric("Total responses", total)
    c2.metric("Positive / Neutral / Negative", f"{pos} / {neu} / {neg}")

    # Sentiment distribution chart
    st.subheader("Sentiment distribution")
    sentiment_counts = df["_sentiment_label"].value_counts().reindex(["Positive","Neutral","Negative"]).fillna(0)
    st.bar_chart(sentiment_counts)

    # Word cloud
    st.subheader("Word Cloud (combined text)")
    combined_text_all = " ".join(df["_combined_text"].astype(str).tolist())
    wc = make_wordcloud(combined_text_all)
    if wc:
        fig = plt.figure(figsize=(10,4))
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        st.pyplot(fig)
    else:
        st.info("No text available to build a word cloud.")

    # Top words using TF-IDF
    st.subheader("Top terms (TF-IDF)")
    sample_for_tfidf = df["_combined_text"].astype(str).tolist()
    if any(t.strip() for t in sample_for_tfidf):
        vect = TfidfVectorizer(stop_words="english", max_features=30)
        X = vect.fit_transform(sample_for_tfidf)
        terms = vect.get_feature_names_out()
        # compute mean tfidf for ranking
        mean_tfidf = np.asarray(X.mean(axis=0)).ravel()
        top_idx = mean_tfidf.argsort()[::-1][:20]
        top_terms_df = pd.DataFrame({
            "term": terms[top_idx],
            "mean_tfidf": mean_tfidf[top_idx]
        })
        st.dataframe(top_terms_df)
    else:
        st.info("Not enough textual content for TF-IDF ranking.")

    # Quick look at the selected columns distribution
    st.subheader("Selected text columns preview")
    preview_cols = [c for c in [text_col, ceo_col, change_col] if c]
    st.dataframe(df[preview_cols + ["_sentiment_label", "_sentiment_compound"]].head(100))

# ---------- PAGE: Comments & Themes ----------
elif page == "Comments & Themes":
    st.header("Comments — Tonality, CEO / Management questions, and Clustering")

    # Filter by sentiment and category
    st.subheader("Filters")
    col1, col2, col3 = st.columns(3)
    chosen_sentiments = col1.multiselect("Sentiment", options=["Positive","Neutral","Negative"], default=["Positive","Neutral","Negative"])
    show_ceo_only = col2.checkbox("Show only rows with CEO/Top Management questions", value=False)
    min_len = col3.slider("Minimum text length (characters)", 0, 500, 0)

    filtered = df[df["_combined_text"].astype(str).str.len() >= min_len]
    if chosen_sentiments:
        filtered = filtered[filtered["_sentiment_label"].isin(chosen_sentiments)]
    if show_ceo_only and ceo_col:
        filtered = filtered[filtered[ceo_col].notna() & (filtered[ceo_col].astype(str).str.strip() != "")]
    st.write(f"Filtered rows: {filtered.shape[0]}")

    # Display comments with sentiment chips
    st.subheader("Comments (sample)")
    for i, row in filtered.reset_index(drop=True).iterrows():
        cols = st.columns([1,8])
        label = row["_sentiment_label"]
        emoji = "🟢" if label == "Positive" else ("🔴" if label == "Negative" else "🟡")
        cols[0].markdown(f"**{emoji} {label or ''}**")
        cols[1].write(row["_combined_text"])

    # CEO / Top Management questions (if column present)
    if ceo_col:
        st.subheader("CEO & Top Management — extracted questions")
        qlist = []
        for text in df[ceo_col].dropna().astype(str):
            parts = parse_multi_questions(text)
            qlist.extend(parts)
        if qlist:
            st.write(f"Extracted {len(qlist)} questions from column `{ceo_col}`")
            for q in qlist:
                st.markdown(f"- {q}")
        else:
            st.info("No CEO / Top Management questions found or column empty.")

    # Thematic clustering: TF-IDF + KMeans
    st.subheader("Thematic clustering (auto-generated)")
    n_clusters = st.slider("Number of clusters (themes):", 2, 10, 4)
    # use combined text (filtered)
    texts_for_clustering = filtered["_combined_text"].astype(str).tolist()
    if len(texts_for_clustering) < 2:
        st.warning("Not enough comments to cluster.")
    else:
        tfidf = TfidfVectorizer(stop_words="english", max_features=2000)
        X = tfidf.fit_transform(texts_for_clustering)
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = km.fit_predict(X)
        # attach and display sample per cluster
        filtered = filtered.reset_index(drop=True)
        filtered["cluster"] = clusters
        for c in range(n_clusters):
            st.markdown(f"### Theme {c+1} (Cluster {c}) — sample comments")
            sample = filtered[filtered["cluster"]==c]["_combined_text"].head(6).tolist()
            if sample:
                for s in sample:
                    st.write(f"- {s}")
            else:
                st.write("No comments in this cluster.")

# ---------- PAGE: Settings & Download ----------
else:
    st.header("Settings, Export & Next steps")
    st.subheader("Download filtered data")
    if st.button("Download current dataframe (CSV)"):
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Click to download CSV", data=csv, file_name="survey_analysis.csv", mime="text/csv")

    st.markdown("**Deployment tips**")
    st.markdown("""
    - If you want the app to load the Excel file automatically on Streamlit Cloud, add the Excel under `/data/` in your repository (path used by default: `data/Issues Influencing Excellent Service Delivery(1-45).xlsx`).
    - Alternatively, leave `Use static file in repo` unchecked and upload from the Streamlit UI each time.
    - To share with management, deploy to Streamlit Cloud (instructions in the README).  
    """)

st.sidebar.markdown("---")
st.sidebar.caption("App generated for: Issues Influencing Excellent Service Delivery (uploaded file).")
