import streamlit as st
import pandas as pd
import altair as alt
from textblob import TextBlob

# Load and cache the dataset
@st.cache
def load_data():
    data = pd.read_csv('english_tweets.csv')
    data['created_at'] = pd.to_datetime(data['created_at'])  # Ensure datetime format
    return data

# Sentiment Analysis Function
def get_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0:
        return 'Positive'
    elif polarity < 0:
        return 'Negative'
    else:
        return 'Neutral'

# Load the data
tweets_df = load_data()

# Title and Introduction
st.title("Twitter Analysis of Tweets about Biden and Trump")
st.write("""
    This app analyzes a dataset of tweets about Biden and Trump, showing trends over time, geographic distribution, 
    engagement metrics, and sentiment analysis.
""")

# Total Tweet Count per Candidate
st.header("Total Tweet Count for Biden and Trump")
tweet_count = tweets_df['candidate'].value_counts()
st.bar_chart(tweet_count)

# Daily Tweet Count Timeline
st.header("Daily Tweet Count Timeline for Biden and Trump")
tweets_df['date'] = tweets_df['created_at'].dt.date  # Extract the date
daily_tweet_count = tweets_df.groupby(['date', 'candidate']).size().reset_index(name='tweet_count')



# Top 10 Countries by Tweet Count
st.header("Top 10 Countries Tweeting About Biden and Trump")
country_tweet_counts = tweets_df.groupby(['candidate', 'country']).size().reset_index(name='tweet_count')
top_10_countries = country_tweet_counts.groupby('candidate').apply(lambda x: x.nlargest(10, 'tweet_count')).reset_index(drop=True)

country_chart = alt.Chart(top_10_countries).mark_bar().encode(
    x='tweet_count:Q',
    y=alt.Y('country:N', sort='-x'),
    color='candidate:N',
    tooltip=['candidate', 'country', 'tweet_count']
).properties(
    width=600,
    height=400
).interactive()

st.altair_chart(country_chart)

# Top 10 States by Tweet Count
st.header("Top 10 States Tweeting About Biden and Trump")
state_tweet_counts = tweets_df.groupby(['candidate', 'state']).size().reset_index(name='tweet_count')
top_10_states = state_tweet_counts.groupby('candidate').apply(lambda x: x.nlargest(10, 'tweet_count')).reset_index(drop=True)

state_chart = alt.Chart(top_10_states).mark_bar().encode(
    x='tweet_count:Q',
    y=alt.Y('state:N', sort='-x'),
    color='candidate:N',
    tooltip=['candidate', 'state', 'tweet_count']
).properties(
    width=600,
    height=400
).interactive()



# Sentiment Analysis Section
st.header("Sentiment Analysis of Tweets")

# Apply sentiment analysis
tweets_df['sentiment'] = tweets_df['tweet'].apply(get_sentiment)

# Sentiment Distribution per Candidate
sentiment_counts = tweets_df.groupby(['candidate', 'sentiment']).size().reset_index(name='count')

# Bar chart for sentiment distribution
sentiment_chart = alt.Chart(sentiment_counts).mark_bar().encode(
    x='sentiment:N',
    y='count:Q',
    color='candidate:N',
    column='candidate:N',
    tooltip=['candidate', 'sentiment', 'count']
).properties(
    width=200,
    height=300
)

st.altair_chart(sentiment_chart)

# Daily Sentiment Timeline
st.header("Daily Sentiment Timeline for Biden and Trump")
daily_sentiment = tweets_df.groupby(['date', 'candidate', 'sentiment']).size().reset_index(name='count')

# Line chart for daily sentiment counts
sentiment_timeline = alt.Chart(daily_sentiment).mark_line().encode(
    x='date:T',
    y='count:Q',
    color='sentiment:N',
    tooltip=['date', 'candidate', 'sentiment', 'count'],
    facet=alt.Facet('candidate:N', columns=1)
).properties(
    width=700,
    height=300
).interactive()

st.altair_chart(sentiment_timeline)
