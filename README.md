# Seattle Airbnb Sentiment Analysis
# Seattle Airbnb Sentiment Analysis

This project analyzes the sentiment of guest reviews for Airbnb listings in Seattle using sentiment analysis techniques with the `VADER` library from `NLTK` and data analysis tools with `Pandas`.

## Project Objectives
The goal is to explore and classify the sentiment of guest reviews to better understand how Airbnb properties are perceived. Sentiment classification is divided into three categories: **Positive**, **Neutral**, and **Negative**.

## Dataset
The dataset used is Seattle Airbnb's, downloaded from Kaggle via the `kagglehub` API. The dataset includes the following files:
- `calendar.csv`: Availability information for the listings.
- `listings.csv`: Property details.
- `reviews.csv`: Reviews left by guests.

## Requirements
- Python 3.6 or above
- Libraries: `pandas`, `nltk`, `kagglehub`
- The `nltk` library requires downloading the `vader_lexicon`.

## Code Structure
- **Importing and Downloading the Dataset**: Uses the `kagglehub` API to download the dataset and load it into `Pandas DataFrame`.
- **Data Cleaning**: Handles missing values in the `comments` column.
- **Sentiment Analysis**: Applies `VADER` from `nltk` to calculate a sentiment score for each comment.
- **Sentiment Classification**: Based on the `compound` score, each comment is categorized as `Positive`, `Negative`, or `Neutral`.

## Results
The output includes sentiment scores and a sentiment category for each comment. Sample results:

| listing_id | comments               | sentiment_score | sentiment_category |
|------------|-------------------------|-----------------|--------------------|
| 6606       | "Great place..."        | 0.8             | Positive          |
| 6607       | "Negative experience..."| -0.6            | Negative          |

## Execution Instructions
1. Ensure you have a Kaggle API key configured for `kagglehub`.
2. Run the script `sentiment_analysis_airbnb.py` to download the dataset, calculate sentiment scores, and display the results.

## Additional Resources
- [NLTK and VADER Documentation](https://www.nltk.org/)
- Bird, Steven, Edward Loper, and Ewan Klein (2009), *Natural Language Processing with Python*. O’Reilly Media Inc.
