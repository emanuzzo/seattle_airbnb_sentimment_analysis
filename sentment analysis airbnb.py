# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 07:55:20 2024

@author: e.nuzzo
"""

import kagglehub
import os
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
# Download latest version
path = kagglehub.dataset_download("swsw1717/seatle-airbnb-open-data-sql-project")
type(path)
print("Path to dataset files:", path)

# Verifica i file presenti nella cartella scaricata
print(os.listdir(path))

calendar_path = os.path.join(path, 'calendar.csv') 
listings_path = os.path.join(path, 'listings.csv') 
reviews_path = os.path.join(path, 'reviews.csv') 
 
# Carica il CSV in un DataFrame Pandas
calendar_df = pd.read_csv(calendar_path)
listings_df = pd.read_csv(listings_path)
reviews_df = pd.read_csv(reviews_path)

 
### see https://www.nltk.org/
## if you want to know more about, you can consider this book:
"""
Bird, Steven, Edward Loper and Ewan Klein (2009), Natural Language Processing with Python. O’Reilly Media Inc.
"""   
 

# Download VADER lexicon
nltk.download('vader_lexicon')
sid = SentimentIntensityAnalyzer()
 

# Apply VADER to each comment
# Fill NaN values in the comments column with an empty string
reviews_df['comments'] = reviews_df['comments'].fillna('')

# Apply VADER sentiment analysis
reviews_df['sentiment_score'] = reviews_df['comments'].apply(lambda x: sid.polarity_scores(x)['compound'])


reviews_df['sentiment_score'][0:10]

# Classify sentiment based on the compound score
reviews_df['sentiment_category'] = reviews_df['sentiment_score'].apply(lambda score: 'Positive' if score > 0.05 else ('Negative' if score < -0.05 else 'Neutral'))

reviews_df[reviews_df['sentiment_category'] =='Negative'].head(5)
