# %%
"""
Explanation of the Columns in the Datate
sets
created_at: Date and time of tweet creation
tweet_id: Unique ID of the tweet
tweet: Full tweet text
likes: Number of likes
retweet_count: Number of retweets
source: Utility used to post tweet
user_id: User ID of tweet creator
user_name: Username of tweet creator
user_screen_name: Screen name of tweet creator
user_description: Description of self by tweet creator
user_join_date: Join date of tweet creator
user_followers_count: Followers count on tweet creator
user_location: Location given on tweet creator's profile
lat: Latitude parsed from user_location
long: Longitude parsed from user_location
city: City parsed from user_location
country: Country parsed from user_location
state: State parsed from user_location
state_code: State code parsed from user_location
collected_at: Date and time tweet data was mined from twitter
"""

# %%
import sys
print(sys.executable)


# %%
pip install plotly

# %%
import pandas as pd

import numpy as np 
import matplotlib.pyplot as plt 
import plotly.express as px 
print(pd.__version__)

# %%
import os
os.chdir("E:/ReDi_School/Data_Circle/Project/dataset")

# %%
trump_df = pd.read_csv("trump.csv",lineterminator='\n')


# %%
biden_df = pd.read_csv("biden.csv",lineterminator='\n')


# %%
print(trump_df.shape)
print(biden_df.shape)

# %%
trump_df.info()

# %%
biden_df.info()

# %%
trump_df.head()

# %%
biden_df.head()

# %%
trump_df.columns

# %%
trump_df.describe()

# %%
biden_df.describe()

# %%
trump_df['candidate'] = 'trump_df'

biden_df['candidate'] = 'biden_df'

# combining the dataframes 
both_data = pd.concat([trump_df, biden_df]) 

# FInal data shape 
print('Final Data Shape :', both_data.shape) 

 
both_data.head(3)

# %%
both_data.drop(['source', 'city',
       'user_id', 'user_name', 'user_screen_name', 'user_description',
       'user_join_date','user_followers_count','long','lat','state_code','user_location','state','collected_at'], axis=1,inplace=True)

# %%
both_data.duplicated().sum()
#no duplicate rows in DataFrame trump_df

# %%
both_data.drop_duplicates().inplace=True

# %%
both_data.isnull().sum()
#NaN values for each column

# %%
both_data['continent'].unique()

# %%
both_data['country'] = both_data['country'].replace({'United States of America': "US",'United States': "US"}) 

# %%
both_data['country'] = both_data['country'].replace({'The Netherlands': "Netherlands",'Netherlands': "Netherlands"}) 

# %%
top_countries = both_data['country'].value_counts().head(10)

# Plot using matplotlib
plt.figure(figsize=(10,6))
top_countries.plot(kind='bar', color='green')
plt.xlabel('Country')
plt.ylabel('Number of Tweets')
plt.title('Top 10 Countries by Tweet Count')
plt.show()

# %%
top_10_countries = both_data['country'].value_counts().head(10).index

# Filter the data to include only the top 10 countries
top_country_data = both_data[both_data['country'].isin(top_10_countries)]

# Group by country and candidate to count tweets
tweets_by_candidate_country = top_country_data.groupby(['country', 'candidate'])['tweet'].count().unstack()

# Plot using matplotlib
tweets_by_candidate_country.plot(kind='bar', stacked=True, figsize=(10,6), color=['blue', 'pink'])
plt.xlabel('Country')
plt.ylabel('Number of Tweets')
plt.title('Tweet Counts for Each Candidate in the Top 10 Countries')
plt.legend(title='Candidate')
plt.show()

# %%
top10countries = both_data.groupby('country')['tweet'].count().sort_values(ascending=False).head(10).index.tolist()
tweet_df = both_data.groupby(['country','candidate'])['tweet'].count().reset_index()
tweet_df = tweet_df[tweet_df['country'].isin(top10countries)]
tweet_df

# %%
both_data['country'].unique()

# %%
#continents_to_remove = ['Europe', 'Oceania', 'Africa','South America', 'Asia', 'Antarctica']

#both_data.drop(both_data[both_data['continent'].isin(continents_to_remove)].index,inplace=True)

# %%
both_data['country'].unique()

# %%
both_data.dropna(inplace=True) 

# %%
both_data.info()

# %%
both_data.apply(lambda x: x.isna().sum()/len(x), axis=0)

# %%
both_data['country'] = both_data['country'].replace({'United States of America': "US",'United States': "US"}) 

# %%
both_data['country'].value_counts()

# %%
both_data['tweet_id'].duplicated().sum()

# %%
both_data.drop_duplicates().inplace=True

# %%
#tweets_count = trump_df.groupby('country)['tweet'].count().reset_index() 

#maybe we should do it at the beginning before focusing on only US 

# %%
both_data['candidate'].value_counts().plot(kind='bar',color=['blue', 'pink'])
plt.xlabel('Candidates')
plt.ylabel('Number of Tweets')
plt.title('Tweets per Candidate')
plt.show()

# %%
# Likes comparison between candidates
likes_comparison = both_data.groupby('candidate')['likes'].sum().reset_index()

# Plot using matplotlib
plt.bar(likes_comparison['candidate'], likes_comparison['likes'], color=['blue', 'pink'])
plt.xlabel('Candidates')
plt.ylabel('Total Likes')
plt.title('Total Likes for Each Candidate')
plt.show()

# %%
both_data['created_at'] = pd.to_datetime(both_data['created_at'])


# %%
both_data['created_at'].unique()

# %%
likes_retweets = both_data.groupby('candidate')[['likes', 'retweet_count']].sum().reset_index()

# Plot
import matplotlib.pyplot as plt

# Bar chart comparison
likes_retweets.plot(x='candidate', y=['likes', 'retweet_count'], kind='bar', color=['blue', 'green'])
plt.title('Total Likes and Retweets per Candidate')
plt.ylabel('Count')
plt.show()

# %%
#time series analysis

both_data['date'] = both_data['created_at'].dt.date
tweets_over_time = both_data.groupby(['date', 'candidate'])['tweet'].count().unstack()

# Plot
tweets_over_time.plot(figsize=(12, 6))
plt.title('Tweets Over Time by Candidate')
plt.ylabel('Number of Tweets')
plt.xlabel('Date')
plt.show()

# %%
import re

def clean_tweet(tweet):
    # Convert to lowercase
    tweet = tweet.lower()
    
    # Remove URLs
    tweet = re.sub(r'http\S+|www\S+|https\S+', '', tweet, flags=re.MULTILINE)
    
    # Remove @mentions
    tweet = re.sub(r'@\w+', '', tweet)
    
    # Remove hashtags (keep the word, remove the #)
    tweet = re.sub(r'#\w+', lambda x: x.group()[1:], tweet)
    
    # Remove non-word characters (anything other than a-z, A-Z, 0-9, and underscore)
    tweet = re.sub(r'[^\w\s]', '', tweet)
    
    # Remove extra whitespace
    tweet = re.sub(r'\s+', ' ', tweet).strip()
    
    return tweet

both_data['cleaned_tweet'] = both_data['tweet'].apply(clean_tweet)

# Preview the first few rows of the dataset to verify
print(both_data[['tweet', 'cleaned_tweet']].head())

# %%
pip install langdetect

# %%
from langdetect import detect

# %%
def detect_language(tweet):
    try:
        return detect(tweet)
    except:
        return None 

# %%
both_data['detected_language'] = np.where(
    both_data['country'] == 'US',           # Condition: country is 'US'
    both_data['tweet'].apply(detect_language),  # Apply detect_language function if true
    np.nan                                  # Set to NaN or some other value if false
)

# %%
'''


both_data['detected_language'] = both_data.apply(
    lambda row: detect_language(row['tweet']) if row['country'] == 'US' else np.nan, axis=1
)
'''

# %%
# Filter for English tweets
english_tweets = both_data[both_data['detected_language'] == 'en']

# Check the number of English tweets
print(f'Total English Tweets: {len(english_tweets)}')

# %%
english_tweets

# %%
english_tweets.to_csv("E:/ReDi_School/Data_Circle/Project/0_Github/Mir/english_tweets.csv")

# %%
