import streamlit as st
import pandas as pd
import altair as alt
import plotly.express as px

#st.write("hello")
#st.write("WElcome")

#st.image("Figure/2_likes_trump.png")

# cd /d E:\ReDi_School\Data_Circle\Project\0_Github\Mir
# activate my_env
# streamlit run streamlit.py

#We can choose my_env select interpreter here and run "streamlit run streamlit.py" directly.


#To convert from ipynb to python

#pip install ipynb-py-convert
#ipynb-py-convert .\twitter_updated.ipynb .\twitter_updated.py




# Load the dataset
@st.cache_data
def load_data():
    data = pd.read_csv('english_tweets.csv')
    data['created_at'] = pd.to_datetime(data['created_at'])  # Ensure datetime format
    return data

# Load the data
tweets_df = load_data()

# Title and Introduction
st.title('Dashboard: Twitter Data Analysis for US Election 2020')
st.write("""
    This app analyzes a dataset of tweets about Biden and Trump, showing trends over time, geographic distribution, 
    and engagement metrics such as likes and retweets.
""")

# Main content
st.header('Overview Statistics')

# Total tweets count for Trump and Biden
col1, col2 = st.columns(2)

with col1:
    trump_tweets = len(tweets_df[tweets_df['candidate'] == 'trump_df'])
    st.metric("Tweets for Trump", trump_tweets)

with col2:
    biden_tweets = len(tweets_df[tweets_df['candidate'] == 'biden_df'])
    st.metric("Tweets for Biden", biden_tweets)

# Total Tweet Count per Candidate
st.header("Total Tweet Count for Biden and Trump")

# Count tweets per candidate and rename the labels
tweet_count = tweets_df['candidate'].value_counts()
tweet_count = tweet_count.rename(index={"biden_df": "Biden", "trump_df": "Trump"}).reset_index()
tweet_count.columns = ["candidate", "count"]

# Define the colors
color_scale = alt.Scale(domain=["Biden", "Trump"], range=["blue", "yellow"])

# Create the bar chart with Altair
bar_chart = alt.Chart(tweet_count).mark_bar().encode(
    x=alt.X("candidate", title="Candidate"),
    y=alt.Y("count", title="Tweet Count"),
    color=alt.Color("candidate", scale=color_scale, legend=None)
).properties(
    width=200,
    height=400
)
# Display the chart in Streamlit
st.altair_chart(bar_chart, use_container_width=True)



# Daily Tweet Count Timeline
st.header("Daily Tweet Count for Biden and Trump")

# Extract the date and calculate daily tweet counts
tweets_df['date'] = tweets_df['created_at'].dt.date
# Map candidate labels to desired legend names
tweets_df['candidate_label'] = tweets_df['candidate'].map({"biden_df": "Biden", "trump_df": "Trump"})
daily_tweet_count = tweets_df.groupby(['date', 'candidate_label']).size().reset_index(name='tweet_count')

# Define the colors for each candidate
color_scale = alt.Scale(domain=["Biden", "Trump"], range=["blue", "yellow"])

# Line chart for daily tweet count
line_chart = alt.Chart(daily_tweet_count).mark_line().encode(
    x=alt.X('date:T', title='Date'),
    y=alt.Y('tweet_count:Q', title='Tweet Count'),
    color=alt.Color('candidate_label:N', scale=color_scale, legend=alt.Legend(title="Candidate")),
    tooltip=['date', 'candidate_label', 'tweet_count']
).properties(
    width=700,
    height=400
).interactive()

st.altair_chart(line_chart)




# Top States (US only) according to tweet counts
st.header('Top US States by Tweet Count')
state_data = tweets_df[tweets_df['country'] == 'US']['state'].value_counts().head(10).reset_index()
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


# Top 10 States by comparing Tweet Counts of Biden and Trump
st.header("Comaring the Tweets between the Top Counting States")
state_tweet_counts = tweets_df.groupby(['candidate', 'state']).size().reset_index(name='tweet_count')

state_tweet_counts['candidate_label'] = state_tweet_counts['candidate'].map({"biden_df": "Biden", "trump_df": "Trump"})
top_10_states = state_tweet_counts.groupby('candidate_label').apply(lambda x: x.nlargest(10, 'tweet_count')).reset_index(drop=True)

# Define the colors for each candidate
color_scale = alt.Scale(domain=["Biden", "Trump"], range=["blue", "yellow"])

state_chart = alt.Chart(top_10_states).mark_bar().encode(
    x='tweet_count:Q',
    y=alt.Y('state:N', sort='-x'),
    color=alt.Color('candidate_label:N', scale=color_scale, legend=alt.Legend(title="Candidate")),
    tooltip=['candidate_label', 'state', 'tweet_count']
).properties(
    width=400,
    height=600
).facet(column=alt.Column('candidate_label:N', header=alt.Header(title='Candidate'))

).interactive()

st.altair_chart(state_chart)





# Load and prepare data
def load_data():
    # Adjust the file path to match the uploaded file name
    df = pd.read_csv('english_tweets.csv')  # Replace with the actual file path if necessary
    df['created_at'] = pd.to_datetime(df['created_at'])  # Ensure datetime format for time-based analysis
    return df

df = load_data()




# Likes Distribution
st.header('Total Likes by Candidate')

# Calculate total likes by candidate
likes_data = df.groupby('candidate')['likes'].sum().reset_index()
likes_data.columns = ['Candidate', 'Total Likes']

# Map candidate names to desired labels
likes_data['Candidate'] = likes_data['Candidate'].replace({'biden_df': 'Biden', 'trump_df': 'Trump'})

# Define the colors
color_scale = alt.Scale(domain=["Biden", "Trump"], range=["blue", "yellow"])

# Create the bar chart
likes_chart = alt.Chart(likes_data).mark_bar().encode(
    x=alt.X('Candidate:N', title='Candidate'),  # Ensure the x-axis is labeled
    y=alt.Y('Total Likes:Q', title='Total Likes'),  # Use the correct column for y-axis
    color=alt.Color('Candidate:N', scale=color_scale, legend=alt.Legend(title="Candidate")),
).properties(
    width=400,
    height=300
)

st.altair_chart(likes_chart, use_container_width=True)




# Add filters in sidebar
st.sidebar.header('Filters')
selected_continent = st.sidebar.multiselect(
    'Select Continent',
    options=df['continent'].unique(),
    #default=df['continent'].unique()
)

selected_candidate = st.sidebar.selectbox(
    'Select Candidate',
    options=['Both', 'trump_df', 'biden_df']
)


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





