#pip install streamlit pandas altair plotly



import streamlit as st
import pandas as pd
import altair as alt



# Load the dataset
@st.cache
def load_data():
    data = pd.read_csv('english_tweets.csv')
    data['created_at'] = pd.to_datetime(data['created_at'])  # Ensure datetime format
    return data

# Load the data
tweets_df = load_data()

# Title and Introduction
st.title("Twitter Analysis of Tweets about Biden and Trump")
st.write("""
    This app analyzes a dataset of tweets about Biden and Trump, showing trends over time, geographic distribution, 
    and engagement metrics such as likes and retweets.
""")






# Total Tweet Count per Candidate
st.header("Total Tweet Count for Biden and Trump")
tweet_count = tweets_df['candidate'].value_counts()
st.bar_chart(tweet_count)

# Daily Tweet Count Timeline
st.header("Daily Tweet Count Timeline for Biden and Trump")
tweets_df['date'] = tweets_df['created_at'].dt.date  # Extract the date
daily_tweet_count = tweets_df.groupby(['date', 'candidate']).size().reset_index(name='tweet_count')

# Line chart for daily tweet count
line_chart = alt.Chart(daily_tweet_count).mark_line().encode(
    x='date:T',
    y='tweet_count:Q',
    color='candidate:N',
    tooltip=['date', 'candidate', 'tweet_count']
).properties(
    width=700,
    height=400
).interactive()

st.altair_chart(line_chart)

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

st.altair_chart(state_chart)

# Distribution of Likes and Retweets
st.header("Distribution of Likes and Retweets")

# Histogram for likes
st.subheader("Likes Distribution")
likes_hist = alt.Chart(tweets_df).mark_bar().encode(
    alt.X("likes:Q", bin=alt.Bin(maxbins=50), title='Likes Count'),
    y='count()',
    color='candidate:N'
).properties(
    width=600,
    height=300
)
st.altair_chart(likes_hist)

# Histogram for retweets
st.subheader("Retweet Distribution")
retweets_hist = alt.Chart(tweets_df).mark_bar().encode(
    alt.X("retweet_count:Q", bin=alt.Bin(maxbins=50), title='Retweet Count'),
    y='count()',
    color='candidate:N'
).properties(
    width=600,
    height=300
)
st.altair_chart(retweets_hist)





import streamlit as st
import pandas as pd
import altair as alt
import plotly.express as px

# Set page title
st.title('Twitter Data Analysis Dashboard')

# Load and prepare data
def load_data():
    # Adjust the file path to match the uploaded file name
    df = pd.read_csv('english_tweets.csv')  # Replace with the actual file path if necessary
    df['created_at'] = pd.to_datetime(df['created_at'])  # Ensure datetime format for time-based analysis
    return df

df = load_data()

# Sidebar
st.sidebar.header('Dashboard')
st.sidebar.markdown('Analysis of Twitter data for Trump and Biden')

# Main content
st.header('Overview Statistics')

# Total tweets count for Trump and Biden
col1, col2 = st.columns(2)

with col1:
    trump_tweets = len(df[df['candidate'] == 'trump_df'])
    st.metric("Trump Tweets", trump_tweets)

with col2:
    biden_tweets = len(df[df['candidate'] == 'biden_df'])
    st.metric("Biden Tweets", biden_tweets)

# Top 10 Countries Bar Chart
st.header('Top 10 Countries by Tweet Count')
country_counts = df['country'].value_counts().head(10).reset_index()
country_counts.columns = ['Country', 'Count']

country_chart = alt.Chart(country_counts).mark_bar().encode(
    x=alt.X('Country:N', sort='-y'),
    y='Count:Q',
    color=alt.Color('Country:N', legend=None)
).properties(
    width=600,
    height=400
)
st.altair_chart(country_chart, use_container_width=True)

# Tweet Distribution by Continent
st.header('Tweet Distribution by Continent')
continent_data = df['continent'].value_counts().reset_index()
continent_data.columns = ['Continent', 'Count']

fig_continent = px.pie(continent_data, 
                      values='Count', 
                      names='Continent',
                      title='Tweet Distribution by Continent')
st.plotly_chart(fig_continent)

# Likes Distribution
st.header('Average Likes by Candidate')
likes_data = df.groupby('candidate')['likes'].mean().reset_index()
likes_data.columns = ['Candidate', 'Average Likes']

likes_chart = alt.Chart(likes_data).mark_bar().encode(
    x='Candidate:N',
    y='Average Likes:Q',
    color='Candidate:N'
).properties(
    width=400,
    height=300
)
st.altair_chart(likes_chart, use_container_width=True)

# Time Series Analysis
st.header('Tweets Over Time')
time_data = df.groupby([df['created_at'].dt.date, 'candidate']).size().reset_index(name='count')

time_chart = alt.Chart(time_data).mark_line().encode(
    x='created_at:T',
    y='count:Q',
    color='candidate:N'
).properties(
    width=700,
    height=400
)
st.altair_chart(time_chart, use_container_width=True)

# Top States (US only)
st.header('Top US States by Tweet Count')
state_data = df[df['country'] == 'US']['state'].value_counts().head(10).reset_index()
state_data.columns = ['State', 'Count']

state_chart = alt.Chart(state_data).mark_bar().encode(
    x=alt.X('State:N', sort='-y'),
    y='Count:Q',
    color=alt.Color('State:N', legend=None)
).properties(
    width=600,
    height=400
)
st.altair_chart(state_chart, use_container_width=True)

# Language Distribution
st.header('Tweet Language Distribution')
lang_data = df['detected_language'].value_counts().head(5).reset_index()
lang_data.columns = ['Language', 'Count']

fig_lang = px.pie(lang_data, 
                 values='Count', 
                 names='Language',
                 title='Top 5 Languages Used in Tweets')
st.plotly_chart(fig_lang)

# Add filters in sidebar
st.sidebar.header('Filters')
selected_continent = st.sidebar.multiselect(
    'Select Continent',
    options=df['continent'].unique(),
    default=df['continent'].unique()
)

selected_candidate = st.sidebar.selectbox(
    'Select Candidate',
    options=['Both', 'trump_df', 'biden_df']
)

# Conditional filtering based on selected filters (optional feature you can add)
filtered_data = df[(df['continent'].isin(selected_continent))]

if selected_candidate != 'Both':
    filtered_data = filtered_data[filtered_data['candidate'] == selected_candidate]

# Display filtered data or use it in additional visualizations
st.write(filtered_data.head())

# Add some explanatory text
st.markdown("""
### About this dashboard
This dashboard provides analysis of Twitter data related to Trump and Biden.
Key metrics include:
- Total tweet counts
- Geographical distribution
- Engagement metrics (likes)
- Temporal patterns
- Language distribution
""")

# Footer
st.sidebar.markdown('---')
st.sidebar.markdown('Created with Streamlit')



import streamlit as st
import pandas as pd
import altair as alt

@st.cache_data
def load_data():
    df = pd.read_csv('english_tweets.csv')  
    return df

df = load_data()

# Example for processing country tweet counts
country_tweet_counts = df.groupby(['country', 'candidate']).size().reset_index(name='tweet_count')
top_10_countries = country_tweet_counts.groupby('candidate')['tweet_count'].apply(lambda x: x.nlargest(10)).reset_index()

# Ensure you have this variable defined before using it
data = df  # or however you want to define your data

# Example of creating an Altair chart
country_chart = alt.Chart(data).mark_arc().encode(
    theta='tweet_count:Q',
    color='country:N'
).properties(
    width=600,
    height=400
)

st.altair_chart(country_chart, use_container_width=True)


