import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from transformers import pipeline
import io

# Set page configuration
st.set_page_config(page_title="Analisis Sentimen IKN", layout="wide", page_icon="📊")

# Function to load huggingface model
@st.cache_resource
def load_model(model_name):
    with st.spinner("Loading model..."):
        return pipeline("text-classification", model=model_name)

# Load sentiment analysis model
model = load_model("taufiqdp/indonesian-sentiment")

# Function to load data from CSV
@st.cache_data
def load_data(uploaded_file):
    if uploaded_file is not None:
        try:
            with st.spinner("Loading data..."):
                df = pd.read_csv(uploaded_file)

                required_columns = ['created_at', 'full_text']
                if not all(col in df.columns for col in required_columns):
                    st.error("CSV file must contain 'created_at', 'sentiment', and 'full_text' columns.")
                    return None

                df['created_at'] = pd.to_datetime(df['created_at'])

                # Perform sentiment analysis
                df['sentiment'] = model(df['full_text'].tolist(), truncation=True, padding=True, batch_size=64)

                return df
        except Exception as e:
            st.error(f"Error loading CSV file: {e}")
            return None
    else:
        st.info("Upload a CSV file to get started.")
        return None


# Title
st.title("Analisis Sentimen Ibu Kota Nusantara (IKN)")

# File uploader
uploaded_file = st.file_uploader("Upload CSV file", type="csv")

# Load data
df = load_data(uploaded_file)


if df is not None:
    # Sidebar
    st.sidebar.header("Filters")
    date_range = st.sidebar.date_input("Select Date Range", [df['created_at'].min().date(), df['created_at'].max().date()])

    # Filter data based on date range
    mask = (df['created_at'] >= pd.to_datetime(date_range[0]).tz_localize('UTC')) & \
           (df['created_at'] <= pd.to_datetime(date_range[1]).tz_localize('UTC'))
    filtered_df = df.loc[mask]

    # Main content
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Sentiment Distribution")
        sentiment_counts = filtered_df['sentiment'].value_counts()
        fig = px.pie(values=sentiment_counts.values, names=sentiment_counts.index, title="Sentiment Distribution")
        st.plotly_chart(fig)

    with col2:
        st.subheader("Sentiment Over Time")
        sentiment_over_time = filtered_df.groupby([filtered_df['created_at'].dt.date, 'sentiment']).size().unstack(fill_value=0)
        fig = px.line(sentiment_over_time, x=sentiment_over_time.index, y=sentiment_over_time.columns, title="Sentiment Trends")
        st.plotly_chart(fig)

    # Word Cloud
    st.subheader("Word Cloud of Tweets")
    text = " ".join(tweet for tweet in filtered_df['full_text'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)

    # Sample Tweets
    st.subheader("Sample Tweets")
    sample_tweets = filtered_df.sample(min(5, len(filtered_df)))
    for _, tweet in sample_tweets.iterrows():
        st.text_area(f"Tweet ({tweet['sentiment']})", tweet['full_text'], height=100)

    # Metrics
    st.subheader("Key Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Tweets Analyzed", len(filtered_df))
    col2.metric("Positive Sentiment", f"{(filtered_df['sentiment'] == 'positive').mean():.2%}")
    col3.metric("Negative Sentiment", f"{(filtered_df['sentiment'] == 'negative').mean():.2%}")

    # Download filtered data
    csv = filtered_df.to_csv(index=False)
    st.download_button(
        label="Download filtered data as CSV",
        data=csv,
        file_name="filtered_ikn_sentiment_data.csv",
        mime="text/csv",
    )

# About
st.sidebar.markdown("---")
st.sidebar.subheader("About")
st.sidebar.info("This app presents a sentiment analysis of tweets about Ibu Kota Nusantara (IKN), Indonesia's planned new capital city. Upload a CSV file containing tweet data to analyze.")

# Select Model
st.sidebar.subheader("Select Model")
# model = st.sidebar.selectbox("Select Sentiment Analysis Model", ["taufiqdp/indonesian-sentiment", "indobenchmark/indobert-base-p1"])

# Instructions for CSV format
st.sidebar.subheader("CSV Format")
st.sidebar.info("""
Your CSV file should contain the following columns:
- date: The date of the tweet (YYYY-MM-DD format)
- sentiment: The sentiment of the tweet (positive, negative, or neutral)
- full_text: The full text of the tweet
""")