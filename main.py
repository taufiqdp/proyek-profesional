import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from transformers import pipeline
import re
from huggingface_hub import login

# Login to Hugging Face
login("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")

# Set page configuration
st.set_page_config(page_title="Analisis Sentimen IKN", layout="wide", page_icon="📊")

# Hide Streamlit menu
hide_streamlit_style = """
                <style>
                div[data-testid="stToolbar"] {
                visibility: hidden;
                height: 0%;
                position: fixed;
                }
                div[data-testid="stDecoration"] {
                visibility: hidden;
                height: 0%;
                position: fixed;
                }
                div[data-testid="stStatusWidget"] {
                visibility: hidden;
                height: 0%;
                position: fixed;
                }
                #MainMenu {
                visibility: hidden;
                height: 0%;
                }
                header {
                visibility: hidden;
                height: 0%;
                }
                footer {
                visibility: hidden;
                height: 0%;
                }
                </style>
                """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)


# About
st.sidebar.markdown("---")


MODEL_PATHS = {
    "NusaBERT-base": "runs/NusaBERT-base",
    "NusaBERT-large": "runs/NusaBERT-large",
    "bert-base-multilingual-uncased": "runs/bert-base-multilingual-uncased",
    "indobert-base-p1": "runs/indobert-base-p1",
    "indobert-base-uncased": "runs/indobert-base-uncased",
    "indobert-large-p1": "runs/indobert-large-p1",
    "indobert-lite-base-p1": "runs/indobert-lite-base-p1",
    "indobert-lite-large-p1": "runs/indobert-lite-large-p1"
}

# Select model
model_name = st.sidebar.selectbox("Pilih model", list(MODEL_PATHS.keys()))


# Function to load huggingface model
@st.cache_resource(show_spinner=False)
def load_model(model_name):
    return pipeline("text-classification", model=model_name)


# Load sentiment analysis model
with st.spinner("Memuat model..."):
    model = load_model(MODEL_PATHS[model_name])


def clean_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = " ".join(text.split())
    return text.lower()


# Function to load data from CSV and run model
@st.cache_data(show_spinner=False)
def load_and_analyze_data(uploaded_file, _model_name):
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)

            required_columns = ["created_at", "full_text"]
            if not all(col in df.columns for col in required_columns):
                st.error(
                    "File CSV harus mengandung kolom 'created_at', 'sentiment', dan 'full_text'."
                )
                return None

            df["created_at"] = pd.to_datetime(df["created_at"])

            # Clean text
            df["full_text"] = df["full_text"].apply(clean_text)

            # Remove duplicates
            df = df.drop_duplicates(subset=["full_text"], keep='first')

            # Perform sentiment analysis and extract labels
            sentiments = model(
                df["full_text"].tolist(), truncation=True, padding=True, batch_size=16
            )
            df["sentiment"] = [s["label"] for s in sentiments]

            return df
        except Exception as e:
            st.error(f"Kesalahan saat memuat file CSV: {e}")
            return None
    else:
        return None

# Title
st.title("Analisis Sentimen Ibu Kota Nusantara (IKN)")

# File uploader
uploaded_file = st.file_uploader("Unggah file CSV", type="csv")

# Load data and run model with caching
with st.spinner("Memuat dan menganalisis data..."):
    df = load_and_analyze_data(uploaded_file, model_name)


if df is not None:
    # Create tabs
    tab1, tab2 = st.tabs(["Hasil", "Tabel Hasil"])

    with tab1:
        # Sidebar
        st.sidebar.header("Filter")
        date_range = st.sidebar.date_input(
            "Pilih Rentang Tanggal",
            [df["created_at"].min().date(), df["created_at"].max().date()],
        )

        # Filter data based on date range
        mask = (
            df["created_at"] >= pd.to_datetime(date_range[0]).tz_localize("UTC")
        ) & (df["created_at"] <= pd.to_datetime(date_range[1]).tz_localize("UTC"))
        filtered_df = df.loc[mask]

        # Main content
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Distribusi Sentimen")
            sentiment_counts = filtered_df["sentiment"].value_counts()
            fig = px.pie(
                values=sentiment_counts.values,
                names=sentiment_counts.index,
                title="Distribusi Sentimen",
            )
            st.plotly_chart(fig)

        with col2:
            st.subheader("Sentimen Seiring Waktu")
            sentiment_over_time = (
                filtered_df.groupby([filtered_df["created_at"].dt.date, "sentiment"])
                .size()
                .unstack(fill_value=0)
            )
            fig = px.line(
                sentiment_over_time,
                x=sentiment_over_time.index,
                y=sentiment_over_time.columns,
                title="Tren Sentimen",
            )
            st.plotly_chart(fig)

        # Word Cloud
        st.subheader("Kata Kunci dari Tweet")
        text = " ".join(tweet for tweet in filtered_df["full_text"])
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate(
            text
        )
        fig, ax = plt.subplots()
        ax.imshow(wordcloud, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)

        # Metrics
        st.subheader("Metrik Utama")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Tweet yang Dianalisis", len(filtered_df))
        col2.metric(
            "Sentimen Positif",
            f"{(filtered_df['sentiment'] == 'positif').mean():.2%}",
        )
        col3.metric(
            "Sentimen Negatif",
            f"{(filtered_df['sentiment'] == 'negatif').mean():.2%}",
        )
        col4.metric(
            "Sentimen Netral", f"{(filtered_df['sentiment'] == 'netral').mean():.2%}"
        )

        # Sample Tweets
        st.subheader("Contoh Tweet")
        sample_tweets = filtered_df.sample(min(5, len(filtered_df)))
        for _, tweet in sample_tweets.iterrows():
            st.text_area(
                f"Tweet ({tweet['sentiment']})", tweet["full_text"], height=100
            )

    with tab2:
        st.subheader("Tabel Hasil Lengkap")
        # Add search and filter functionality to the full results table
        search_term = st.text_input("Cari tweet")

        # Filter options
        sentiment_filter = st.multiselect(
            "Filter berdasarkan Sentimen",
            options=df["sentiment"].unique(),
            default=df["sentiment"].unique(),
        )

        # Apply filters
        filtered_table = filtered_df[
            (filtered_df["sentiment"].isin(sentiment_filter))
            & (filtered_df["full_text"].str.contains(search_term, case=False, na=False))
        ]

        # Display table with pagination
        st.dataframe(
            filtered_table[["created_at", "sentiment", "full_text"]],
            column_config={
                "created_at": "Tanggal",
                "sentiment": "Sentimen",
                "full_text": "Konten Tweet",
            },
            hide_index=True,
            use_container_width=True,
        )

        # Download filtered data
        csv = filtered_table.to_csv(index=False)
        st.download_button(
            label="Unduh data sebagai CSV",
            data=csv,
            file_name="filtered_ikn_sentiment_data.csv",
            mime="text/csv",
        )


# Instructions for CSV format
st.sidebar.subheader("Format CSV")
st.sidebar.info(
    """
File CSV Anda harus mengandung kolom berikut:
- created_at: Tanggal tweet (format YYYY-MM-DD)
- full_text: Teks lengkap dari tweet
"""
)