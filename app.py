# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import ast

# Page config
st.set_page_config(page_title="WhatsApp Chat Dashboard", layout="wide")
plt.style.use("dark_background")
sns.set_style("darkgrid")

# Load cleaned dataframe
df_clean = pd.read_csv("cleaned_chat2.csv", parse_dates=["Timestamp"])
# make a copy to avoid SettingWithCopy warnings
df_clean = df_clean.copy()

# If Emojis column was saved as a string representation of a list, try to convert it back
if "Emojis" in df_clean.columns:
    if df_clean["Emojis"].dtype == object:
        def try_parse(x):
            try:
                # common format: "['😂', '🤣']"
                return ast.literal_eval(x) if pd.notna(x) and x.strip().startswith(("[", "(")) else x
            except Exception:
                return x
        df_clean["Emojis"] = df_clean["Emojis"].apply(try_parse)

# Ensure Message is string
df_clean["Message"] = df_clean["Message"].astype(str)

# Define tabs (categories)
tab_home, tab_stats, tab_words, tab_emojis, tab_sentiment, tab_users = st.tabs(
    ["🏠 Home", "📊 Message Statistics", "📝 Word Analysis", "😂 Emoji Analysis", "😊 Sentiment Analysis", "👥 User Comparison"]
)

# -------------------------
# Home (cleaned data sample)
# -------------------------
with tab_home:
    st.header("Home — Cleaned Data (sample)")
    st.write("This is the cleaned dataset used for the dashboard. You can edit cleaning steps later in your notebook.")
    st.dataframe(df_clean.head(10))
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Messages", len(df_clean))
    col2.metric("Unique Senders", df_clean["Sender"].nunique())
    col3.metric("Time Range", f"{df_clean['Timestamp'].min().date()} → {df_clean['Timestamp'].max().date()}")

# -----------------------------------
# Message Statistics (paste / extend)
# -----------------------------------
with tab_stats:
    st.header("📊 Message Statistics")
    st.write("Summary stats and trends (daily/monthly/DOW/heatmap).")
    # Example: top senders (you can expand with your full code)
    msg_counts = df_clean['Sender'].value_counts()
    top_n = 10
    top_senders = msg_counts.head(top_n).reset_index()
    top_senders.columns = ['Sender', 'Messages']

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=top_senders, x='Sender', y='Messages', palette="viridis", ax=ax)
    for i, v in enumerate(top_senders['Messages']):
        ax.text(i, v + max(top_senders['Messages']) * 0.01, str(v), color='white', ha='center')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right')
    ax.set_title(f"Top {top_n} Senders by Number of Messages")
    st.pyplot(fig)

    # Daily / Monthly trends (example)
    df_time = df_clean.set_index('Timestamp').sort_index()
    daily = df_time['Message'].resample('D').count()
    monthly = df_time['Message'].resample('M').count()

    st.subheader("Messages per Day (with 7-day rolling avg)")
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(daily.index, daily.values, alpha=0.5, label='Daily count', color='cyan')
    ax.plot(daily.index, daily.rolling(7).mean(), label='7-day rolling avg', color='magenta')
    ax.legend()
    st.pyplot(fig)

    st.subheader("Messages per Month")
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(monthly.index, monthly.values, marker='o', color='orange')
    for x, y in zip(monthly.index, monthly.values):
        ax.text(x, y + max(monthly.values) * 0.01, str(y), ha='center', color='white')
    st.pyplot(fig)

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

# -------------------------
# Word Analysis (WordCloud)
# -------------------------
with tab_words:
    st.header("📝 Word Analysis")
    st.write("Overall wordcloud (links removed). Per-user top words were skipped as requested.")
    # Clean links
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

# -------------------------
# Emoji Analysis (placeholder)
# -------------------------
with tab_emojis:
    st.header("😂 Emoji Analysis")
    st.write("Top emojis and counts (you can replace this with your full emoji plotting code).")
    if "Emojis" in df_clean.columns:
        # flatten if list-like; handle single strings etc.
        flat = []
        for item in df_clean["Emojis"].dropna():
            if isinstance(item, (list, tuple)):
                flat.extend(item)
            else:
                # if it's a single emoji string, try to split into emojis conservatively
                flat.append(item)
        emoji_counts = Counter(flat).most_common(20)
        if emoji_counts:
            emojis, counts = zip(*emoji_counts)
            fig, ax = plt.subplots(figsize=(10,4))
            sns.barplot(x=list(emojis), y=list(counts), palette="magma", ax=ax)
            ax.set_title("Top Emojis")
            st.pyplot(fig)
        else:
            st.info("No emojis found.")
    else:
        st.info("No 'Emojis' column in dataset.")

# -------------------------
# Sentiment Analysis (placeholder)
# -------------------------
with tab_sentiment:
    st.header("😊 Sentiment Analysis")
    st.write("Sentiment charts will appear here (ensure 'Sentiment' column exists in df_clean).")
    if "Sentiment" in df_clean.columns:
        sent_counts = df_clean["Sentiment"].value_counts(normalize=True) * 100
        fig, ax = plt.subplots()
        ax.pie(sent_counts, labels=sent_counts.index, autopct="%1.1f%%", startangle=90)
        ax.set_title("Sentiment Distribution")
        st.pyplot(fig)

        trend = df_clean.groupby([pd.Grouper(key="Timestamp", freq="W"), "Sentiment"]).size().unstack().fillna(0)
        fig, ax = plt.subplots(figsize=(10,4))
        trend.plot(ax=ax)
        ax.set_title("Sentiment Trend (Weekly)")
        st.pyplot(fig)
    else:
        st.info("No 'Sentiment' column found. Run sentiment code in notebook and add it to cleaned CSV.")

# -------------------------
# User Comparison (placeholder)
# -------------------------
with tab_users:
    st.header("👥 User Comparison")
    st.write("User-wise tables/plots will appear here.")
    if "Sender" in df_clean.columns:
        st.dataframe(df_clean["Sender"].value_counts().head(50).rename_axis("Sender").reset_index(name="Messages"))
    else:
        st.info("No 'Sender' column present.")
