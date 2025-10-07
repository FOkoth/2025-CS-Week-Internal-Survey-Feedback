import streamlit as st
import pandas as pd

st.set_page_config(page_title="Service Delivery Survey", layout="wide")

st.title("📊 HELB Service Delivery Analysis")

# Option 1: Load file directly from GitHub (static)
@st.cache_data
def load_data():
    url = "Issues Influencing Excellent Service Delivery.xlsx"
    return pd.read_excel(url)

try:
    df = load_data()
    st.success("✅ Data loaded successfully from GitHub!")
    st.dataframe(df.head())
except Exception as e:
    st.error(f"❌ Failed to load data: {e}")

st.caption("Data source: HELB Internal Service Delivery Survey")

