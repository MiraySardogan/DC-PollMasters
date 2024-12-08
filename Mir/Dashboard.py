import streamlit as st
import numpy as np
import pandas as pd
import geopandas as gpd
from textblob import TextBlob
from wordcloud import WordCloud
import re
import altair as alt
import plotly.express as px
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, DayLocator
from matplotlib.ticker import MaxNLocator
import matplotlib.colors as mcolors
import leafmap.foliumap as leafmap
import seaborn as sns
import nltk

nltk.download("punkt")
nltk.download("stopwords")

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize



# Load the dataset
@st.cache_data
def load_data():
    data = pd.read_csv('english_tweets.csv')
    data['created_at'] = pd.to_datetime(data['created_at'])  # Ensure datetime format
    return data

# Load the data
tweets_df = load_data()

# Sidebar for selecting analysis
selected_candidate = st.sidebar.selectbox(
    'Select Analysis',
    options=['None', 'Exploratory Analysis', 'Sentiment Analysis']
)

# Conditional display of sections
if selected_candidate == 'Exploratory Analysis':
    # Title and Introduction
    st.title('Dashboard: Twitter Data Analysis for US Election 2020')
    st.write("""
        This app analyzes a dataset of tweets about Biden and Trump, showing trends over time, 
        geographic distribution, and engagement metrics such as likes and retweets.
    """)

    # Overview Statistics
    st.header('Overview Statistics')
    col1, col2 = st.columns(2)

    with col1:
        trump_tweets = len(tweets_df[tweets_df['candidate'] == 'trump_df'])
        st.metric("Tweets for Trump", trump_tweets)

    with col2:
        biden_tweets = len(tweets_df[tweets_df['candidate'] == 'biden_df'])
        st.metric("Tweets for Biden", biden_tweets)

    # Total Tweet Count per Candidate
    st.header("Total Tweet Count for Biden and Trump")
    tweet_count = tweets_df['candidate'].value_counts()
    tweet_count = tweet_count.rename(index={"biden_df": "Biden", "trump_df": "Trump"}).reset_index()
    tweet_count.columns = ["candidate", "count"]

    color_scale = alt.Scale(domain=["Biden", "Trump"], range=["blue", "red"])

    bar_chart = alt.Chart(tweet_count).mark_bar().encode(
        x=alt.X("candidate", title="Candidate"),
        y=alt.Y("count", title="Tweet Count"),
        color=alt.Color("candidate", scale=color_scale, legend=None)
    ).properties(
        width=200,
        height=400
    )
    st.altair_chart(bar_chart, use_container_width=True)

    # Daily Tweet Count Timeline
    st.header("Daily Tweet Count for Biden and Trump")
    tweets_df['date'] = tweets_df['created_at'].dt.date
    tweets_df['candidate_label'] = tweets_df['candidate'].map({"biden_df": "Biden", "trump_df": "Trump"})
    daily_tweet_count = tweets_df.groupby(['date', 'candidate_label']).size().reset_index(name='tweet_count')

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
    color_scale = alt.Scale(domain=["Biden", "Trump"], range=["blue", "red"])

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




    # Likes Distribution
    st.header('Total Likes by Candidate')

    # Calculate total likes by candidate
    likes_data = tweets_df.groupby('candidate')['likes'].sum().reset_index()
    likes_data.columns = ['Candidate', 'Total Likes']

    # Map candidate names to desired labels
    likes_data['Candidate'] = likes_data['Candidate'].replace({'biden_df': 'Biden', 'trump_df': 'Trump'})

    # Define the colors
    color_scale = alt.Scale(domain=["Biden", "Trump"], range=["blue", "red"])

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




    # Geographical Distribution of Likes
    #show the distribution of map with likes
    st.header("Geographical Distribution of Likes")
    trump_likes_path= "USA_shp_file/trump_likes.gpkg"
    biden_likes_path= "USA_shp_file/biden_likes.gpkg"

    @st.cache_data
    def read_gdf(file_path):
        gdf = gpd.read_file(file_path)
        return gdf

    #now read the files
    trump_likes= read_gdf(trump_likes_path)
    biden_likes= read_gdf(biden_likes_path)


    st.header("Geographical Distribution of Likes")
    basemap = st.selectbox(
        'Select a Basemap',
        ['CartoDB.DarkMatter', 'OpenStreetMap', 'CartoDB.Positron'],
        key="basemap_selectbox"
    )
    dataset_choice = st.selectbox(
        "Select Candidate",
        ["Trump Likes", "Biden Likes"],
        key="dataset_selectbox"
    )
    if dataset_choice == "Trump Likes":
        selected_gdf = trump_likes
        cmap = plt.cm.Reds
        title = "Likes for Trump"
    else:
        selected_gdf = biden_likes
        cmap = plt.cm.Blues
        title = "Likes for Biden"

    values = selected_gdf['likes']
    norm = mcolors.Normalize(vmin=values.min(), vmax=values.max())
    selected_gdf['color'] = selected_gdf['likes'].apply(lambda x: mcolors.to_hex(cmap(norm(x))))

    m = leafmap.Map(
        layers_control=False,
        draw_control=False,
        measure_control=False,
        fullscreen_control=False,
    )
    m.add_basemap(basemap)
    m.add_gdf(
        gdf=selected_gdf,
        zoom_to_layer=True,
        style_function=lambda feature: {
            'fillColor': feature['properties']['color'],
            'color': 'black',
            'weight': 0.5,
            'fillOpacity': 0.7,
        }
    )
    st.write(f"Map showing {title}")
    m.to_streamlit(900, 650)

    # Add a colorbar
    fig, ax = plt.subplots(figsize=(6, 1))
    fig.subplots_adjust(bottom=0.5)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm._A = []
    cbar = fig.colorbar(sm, cax=ax, orientation='horizontal')
    cbar.set_label("Likes")
    st.pyplot(fig)




elif selected_candidate == 'Sentiment Analysis':
    st.title("Sentiment Analysis")
    st.write("Sentiment Analysis of the tweets is done using Natural Language Processing (NLP).")
    st.header("Wordcloud for the tweets")

    # Load the dataset
    @st.cache_data
    def load_data():
        data = pd.read_csv('trump_tweets_sentiment.csv')
        data['created_at'] = pd.to_datetime(data['created_at'])  # Ensure datetime format
        return data

    # Load the data
    sentiment_trump = load_data()
    #Pivot table for Trump
    #trump_pivot= sentiment_trump.pivot_table(index=['segmentation'], aggfunc={'segmentation': 'count'})
    #trump_pivot

    def load_data():
        data = pd.read_csv('biden_tweets_sentiment.csv')
        data['created_at'] = pd.to_datetime(data['created_at'])  # Ensure datetime format
        return data

    # Load the data
    sentiment_biden = load_data()
    #Pivot table for Biden
    #biden_pivot= sentiment_biden.pivot_table(index=['segmentation'], aggfunc={'segmentation': 'count'})
    #biden_pivot

    dataset_choice = st.selectbox(
            "Select Candidate",
            ["Wordcloud for Trump", "Wordcloud for Biden"],
            key="dataset_selectbox_wordCloud"
        )
    if dataset_choice == "Wordcloud for Trump":
        selected_df = sentiment_trump
            
    else:
            selected_df = sentiment_biden
                       

    consolidated= ' '. join(word for word in selected_df["cleaned_tweet"])

    worCloud= WordCloud(width=400, height=200, random_state=20, max_font_size=120).generate(consolidated)

    plt.imshow(worCloud, interpolation='bilinear')
    plt.title(dataset_choice)
    plt.axis('off')
    plt.tight_layout()
    st.pyplot(plt)
    plt.show()




    # Plot the time-series polarity for both Biden and Trump
    st.header("Comparison time series polarity")

    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(12, 7)  # Adjust figure size

    # Load the dataset
    @st.cache_data
    def load_data():
        data = pd.read_csv('trump_tweets_sentiment.csv')
        data['created_at'] = pd.to_datetime(data['created_at'])  # Ensure datetime format
        data["Date"]= data["created_at"].dt.date #sepearate the Date only as a new column
        data= data.set_index("Date")
        return data

    # Load the data
    sentiment_trump = load_data()
    # Group by date and calculate the mean polarity for each day
    trump_polarity = sentiment_trump.groupby(sentiment_trump.index)["tPolarity"].mean()
    trump_polarity = pd.DataFrame(trump_polarity)

    #print(trump_polarity.head())

    def load_data():
        data = pd.read_csv('biden_tweets_sentiment.csv')
        data['created_at'] = pd.to_datetime(data['created_at'])  # Ensure datetime format
        data["Date"]= data["created_at"].dt.date #sepearate the Date only as a new column
        data= data.set_index("Date")
        return data

    # Load the data
    sentiment_biden = load_data()
    #print(sentiment_biden.head())
    #Pivot table for Biden
    #biden_pivot= sentiment_biden.pivot_table(index=['segmentation'], aggfunc={'segmentation': 'count'})
    #biden_pivot
    

    biden_polarity = sentiment_biden.groupby(sentiment_biden.index)["tPolarity"].mean()
    biden_polarity = pd.DataFrame(biden_polarity)

    # Plot the data for both Biden and Trump
    biden_polarity.plot(ax=ax, color="blue", )
    trump_polarity.plot(ax=ax, color="red",)

    # Set x-axis locator for everyday but limit the number of ticks
    #ax.xaxis.set_major_locator(MaxNLocator(integer=True, prune='both', nbins=10))  # Limit the number of ticks
    #ax.xaxis.set_major_formatter(DateFormatter('%d-%m'))  # Set the date format

    # Add vertical lines for Election Day and Last Debate
    election_date = pd.Timestamp("2020-11-03")
    ax.axvline(election_date, color='black', linestyle='--', )
    ax.text(election_date, 0.11, "Election Day", rotation=45)

    last_debate = pd.Timestamp("2020-10-22")
    ax.axvline(last_debate, color='black', linestyle='--',)
    ax.text(last_debate, 0.11, 'Last Debate', rotation=45)

    # Rotate the x-axis labels for better readability
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30)

    # Add grid and legend
    ax.yaxis.grid(True, linewidth=0.4)
    ax.legend(['tPolarity (Biden)', 'tPolarity (Trump)'])

    plt.title("Time-Series tPolarity")
    fig.tight_layout()

    # Display the plot in Streamlit
    st.pyplot(fig)

    
    
    #plot the win/lose prediction
    
    st.header("Prediction of Wind  / Loose")
    positive_trump= round(len(sentiment_trump[sentiment_trump["segmentation"]=="positive"]) / len(sentiment_trump)*100, 1)
    neutral_trump= round(len(sentiment_trump[sentiment_trump["segmentation"]=="neutral"]) / len(sentiment_trump)*100, 1)
    negative_trump= round(len(sentiment_trump[sentiment_trump["segmentation"]=="negative"]) / len(sentiment_trump)*100, 1)
    
    #create a list for these values
    responses_trump_list= [positive_trump, neutral_trump, negative_trump]

    #create a dataframe for Trump
    response_trump= {'response': ['mayWin', 'mayLoose', 'notSure'], 'percentage': [positive_trump, negative_trump, neutral_trump] }
    response_trump= pd.DataFrame(response_trump)

    positive_biden= round(len(sentiment_biden[sentiment_biden["segmentation"]=="positive"]) / len(sentiment_biden)*100, 1)
    neutral_biden= round(len(sentiment_biden[sentiment_biden["segmentation"]=="neutral"]) / len(sentiment_biden)*100, 1)
    negative_biden= round(len(sentiment_biden[sentiment_biden["segmentation"]=="negative"]) / len(sentiment_biden)*100, 1)
    
    #create a list for these values
    responses_biden_list= [positive_biden, neutral_biden, negative_biden]

    #create a dataframe for Trump
    response_biden= {'response': ['mayWin', 'mayLoose', 'notSure'], 'percentage': [positive_biden, negative_biden, neutral_biden] }
    response_biden= pd.DataFrame(response_biden)    


    candidate_choice = st.selectbox(
            "Select Candidate",
            ["Trump", "Biden"],
            key="dataset_selectbox_winPred"
        )
    if candidate_choice == "Trump":
        selected_df = response_trump
            
    else:
        selected_df = response_biden
                       

    # Plot using Seaborn
    fig, ax = plt.subplots()
    sns.barplot(
        data=selected_df,
        x="response",
        y="percentage",
        palette=["red", "blue", "green"],
        ax=ax  # Pass the axes object explicitly
    )

    plt.tight_layout()
    st.pyplot(plt)
    plt.show()

else:
    st.title("Welcome!")
    st.write("Please select an analysis type from the sidebar.")


