# DC-PollMasters
US Election 2020 Tweets

Project Proposal: Twitter Election Data Analysis
Data source: Link
Paper further reading: Link1, Link2
Project Overview 
The objective of this project is to analyze Twitter data related to the 2020 U.S. election using Natural Language Processing (NLP) and time series analysis to uncover patterns in public sentiment, trending topics, and engagement over time and across regions. The project will involve sentiment analysis, trend detection, and predictive modeling to understand how discussions evolved during the election period. The final deliverable will be an interactive dashboard that visualizes these insights, allowing users to explore the data and gain actionable information about public opinion and the impact of key events. 

Tasks 
Time Series Analysis
Level 1:
Geospatial Time Series Analysis
Objective: Analyze the time series data to identify trends in tweet activity, sentiment, and engagement metrics (likes, retweets).
Approach: Apply moving averages, seasonal decomposition, and trend detection methods. Hint: Prophet
Outcome: Visualization of trends over time and identification of peak periods.
Trend Analysis
Objective: Predict future engagement metrics (likes, retweets) based on historical data.
Approach: Use basic time series forecasting models like ARIMA or exponential smoothing.
Outcome: Predictive models that can estimate future engagement levels.
Engagement Prediction
Objective: Predict future engagement metrics (likes, retweets) based on historical data.
Approach: Use basic time series forecasting models like ARIMA, exponential smoothing.
Outcome: Predictive models that can estimate future engagement levels.

Level 2:
Anomaly Detection
Objective: Detect unusual spikes or drops in tweet activity, sentiment, or engagement.
Approach: Implement advanced techniques like Seasonal Hybrid Extreme Studentized Deviate (S-H-ESD) or deep learning models for anomaly detection.
Outcome: Identification of events or actions that cause anomalies in the data.
Causal Impact Analysis
Objective: Assess the impact of specific events (e.g., debates, policy announcements) on tweet activity and sentiment.
Approach: Use models like Bayesian structural time series (BSTS) to quantify the effect of events.
Outcome: Quantifiable impact analysis that can be correlated with key events.

Natural Language Processing Tasks
Level 1
Sentiment Analysis
Objective: Classify the sentiment of tweets (positive, negative, neutral) to understand public opinion on various topics like the 2020 U.S. election, candidates, or specific issues.
Approach: Use pre-trained models like VADER, Pysentimiento for sentiment analysis.
Outcome: A sentiment score that can be aggregated over time or by location.
Hashtag Analysis
Objective: Identify and analyze the most frequently used hashtags.
Approach: Use frequency analysis and clustering to group related hashtags.
Outcome: Insights into trending topics and their geographical distribution.
Level 2 (for those high achievers)
Stance Detection
Objective: Identify and categorize entities such as people, organizations, and locations mentioned in tweets.
Approach: Use advanced NLP libraries like SpaCy or fine-tuned transformers.
Outcome: A detailed analysis of which entities are being discussed and how they are connected.
Name Entity Recognition (NER)
Objective: Determine the stance (supportive, opposing, neutral) of tweets towards specific entities (e.g., political candidates).
Approach: Develop or fine-tune models specifically designed for stance detection.
Outcome: Insights into public opinion polarization.

Dashboard development
Use a Data Science Visualization tool such as Streamlit or Taipy. We will select one to stay consistent for all students.

Data Visualization
Create interactive visualizations using libraries such as Bokeh, Seaborn 
User Interface Design
Develop a user friendly UI dashboard that allows users of the tool to manipulate variables such as time or categories such as location, candidate… 
