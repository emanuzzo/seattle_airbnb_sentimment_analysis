# -*- coding: utf-8 -*-
"""
Created on Sun 2024 Nov 10

@author: e.nuzzo
"""

import kagglehub
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

## nltk stays for "natual language toolkit"
###  it is a well known package
##  when it comes to deal with natural la language in python
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
# Download latest version of the dataset I am using from kaggle
path = kagglehub.dataset_download("swsw1717/seatle-airbnb-open-data-sql-project")


calendar_path = os.path.join(path, 'calendar.csv') 
listings_path = os.path.join(path, 'listings.csv') 
reviews_path = os.path.join(path, 'reviews.csv') 
 
# import csv files into Pandas Dataframes
calendar_df = pd.read_csv(calendar_path)
listings_df = pd.read_csv(listings_path)
reviews_df = pd.read_csv(reviews_path)

 
### see https://www.nltk.org/
## if you want to know more about, you can consider this book:
"""
Bird, Steven, Edward Loper and Ewan Klein (2009), Natural Language Processing with Python. O’Reilly Media Inc.
"""   
 

# Download VADER lexicon for sentiment analysis
# The VADER lexicon is a pre-built sentiment dictionary specifically designed for analyzing 
# social media content and short text, which includes informal language, emoticons, and slang. 
# It calculates sentiment scores based on the words' polarity and context (e.g., negations or intensifiers).
nltk.download('vader_lexicon')
sid = SentimentIntensityAnalyzer()
nltk.download('vader_lexicon')
sid = SentimentIntensityAnalyzer()
 

# Apply VADER to each comment
# Fill NaN values in the comments column with an empty string
## if we do not, we will get an error when we are going to validate 

reviews_df['comments'] = reviews_df['comments'].fillna('')

# Apply VADER sentiment analysis
reviews_df['sentiment_score'] = reviews_df['comments'].apply(lambda x: sid.polarity_scores(x)['compound'])


reviews_df['sentiment_score'][0:10]

# Classify sentiment based on the compound score
reviews_df['sentiment_category'] = reviews_df['sentiment_score'].apply(lambda score: 'Positive' if score > 0.05 else ('Negative' if score < -0.05 else 'Neutral'))

 

# Histogram of sentiment scores
plt.figure(figsize=(10,6))
sns.histplot(reviews_df['sentiment_score'], bins=50, kde=True, color='blue')
plt.title('Distribution of Sentiment Scores')
plt.xlabel('Sentiment Score')
plt.ylabel('Frequency')
plt.show()
