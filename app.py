# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import ast

# Page configuration
st.set_page_config(page_title="WhatsApp Chat Dashboard", layout="wide")
plt.style.use("dark_background")
sns.set_style("darkgrid")

# Loading cleaned dataframe
df_clean = pd.read_csv("cleaned_chat.csv", parse_dates=["Timestamp"])
# making a copy to avoid SettingWithCopy warnings
df_clean = df_clean.copy()


if "Emojis" in df_clean.columns:
    if df_clean["Emojis"].dtype == object:
        def try_parse(x):
            try:
                return ast.literal_eval(x) if pd.notna(x) and x.strip().startswith(("[", "(")) else x
            except Exception:
                return x
        df_clean["Emojis"] = df_clean["Emojis"].apply(try_parse)

# Ensuring Message is string
df_clean["Message"] = df_clean["Message"].astype(str)

# Defining tabs
tab_home, tab_stats, tab_words, tab_sentiment, tab_users = st.tabs(
    ["🏠 Home", "📊 Message Statistics", "📝 Word Analysis", "😊 Sentiment Analysis", "👥 User Comparison"]
)


# Home

with tab_home:
    st.header("Home — Cleaned Data (sample)")
    st.write("Cleaned dataset used for the dashboard.")
    cols_to_show = [c for c in df_clean.columns if c != "Cleaned" and c!='Emojis']
    st.dataframe(df_clean[cols_to_show].head(5))
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Messages", len(df_clean))
    col2.metric("Unique Senders", df_clean["Sender"].nunique())
    col3.metric("Time Range", f"{df_clean['Timestamp'].min().date()} → {df_clean['Timestamp'].max().date()}")


# Message Statistics

with tab_stats:
    st.header("📊 Message Statistics")
    st.write("Summary stats and trends.")

    # Daily / Monthly trends
    df_time = df_clean.set_index('Timestamp').sort_index()
    daily = df_time['Message'].resample('D').count()
    monthly = df_time['Message'].resample('M').count()

    # Day of week bar
    df_clean['DayOfWeek'] = df_clean['Timestamp'].dt.day_name()
    dow = df_clean.groupby('DayOfWeek').size().reindex(
        ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    ).fillna(0)
    st.subheader("Messages by Day of the Week")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x=dow.index, y=dow.values, color="orange", ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    st.pyplot(fig)

    # Heatmap (Day vs Hour)
    df_clean['Hour'] = df_clean['Timestamp'].dt.hour
    heatmap_data = df_clean.groupby(['DayOfWeek', 'Hour']).size().unstack(fill_value=0)
    heatmap_data = heatmap_data.reindex(
        ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    ).fillna(0)

    st.subheader("Activity Heatmap (Day vs Hour)")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(heatmap_data, cmap="magma", linewidths=.5, ax=ax)
    ax.set_xlabel("Hour of the day")
    ax.set_ylabel("Day of week")
    st.pyplot(fig)


# Word Analysis (WordCloud)

with tab_words:
    st.header("📝 Word Analysis")
    st.write("Overall wordcloud")
    df_clean = df_clean.copy()
    df_clean['Cleaned'] = df_clean['Message'].str.replace(r'http\S+|www\S+', '', regex=True)

    all_text = " ".join(df_clean['Cleaned'].dropna().astype(str))
    if len(all_text.strip()) == 0:
        st.info("No text available to build word cloud.")
    else:
        wc = WordCloud(width=900, height=450, background_color="black", colormap="plasma").generate(all_text)
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title("Word Cloud - Overall Chat")
        st.pyplot(fig)


# Sentiment Analysis

with tab_sentiment:
    st.header("😊 Sentiment Analysis")
    st.write("Sentiment charts will appear here.")
    if "Sentiment" in df_clean.columns:
        sent_counts = df_clean["Sentiment"].value_counts(normalize=True) * 100
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.pie(sent_counts, labels=sent_counts.index, autopct="%1.1f%%", startangle=90, radius=0.6, textprops={"fontsize":8})
        ax.set_title("Sentiment Distribution", fontsize=10)
        st.pyplot(fig)
    else:
        st.info("No 'Sentiment' column found.")


# User Comparison

with tab_users:
    st.header("👥 User Comparison")
    st.write("User-wise tables/plots")
    if "Sender" in df_clean.columns:
        st.dataframe(df_clean["Sender"].value_counts().head(50).rename_axis("Sender").reset_index(name="Messages"))
    else:
        st.info("No 'Sender' column present.")
