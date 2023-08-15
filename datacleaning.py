import pandas as pd
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
import re
from tqdm import tqdm

# Create an instance of SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

# Load the DataFrame from the CSV file
df = pd.read_csv(r'C:\Projects\Instagram Reviews\instagram.csv')

# Function to calculate sentiment scores
def calculate_sentiment_scores(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in text.split() if word not in stop_words]
    filtered_text = ' '.join(filtered_words)
    sentiment_scores = sia.polarity_scores(filtered_text)
    return sentiment_scores

# Add sentiment analysis scores to the DataFrame
tqdm.pandas()  # Use tqdm with pandas
df['sentiment_scores'] = df['review_description'].progress_apply(calculate_sentiment_scores)

# Extract sentiment scores and add as separate columns
df['positive_score'] = df['sentiment_scores'].apply(lambda x: x['pos'])
df['negative_score'] = df['sentiment_scores'].apply(lambda x: x['neg'])
df['neutral_score'] = df['sentiment_scores'].apply(lambda x: x['neu'])
df['compound_score'] = df['sentiment_scores'].apply(lambda x: x['compound'])

print(df.head(50))
