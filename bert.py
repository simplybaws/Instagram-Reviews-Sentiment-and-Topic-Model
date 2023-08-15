import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load the DataFrame from the CSV file
df = pd.read_csv(r'C:\Projects\Instagram Reviews\instagram.csv')

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")

# Set up device (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Function to calculate sentiment scores using BERT
def calculate_sentiment_bert(text):
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        inputs.to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        softmax_scores = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
        return {
            "positive_score": softmax_scores[2],  # Positive class score
            "negative_score": softmax_scores[0],  # Negative class score
            "neutral_score": softmax_scores[1],   # Neutral class score
        }
    except RuntimeError as e:
        print(f"Skipped line due to RuntimeError: {e}")
        return None

# Add sentiment analysis scores using BERT to the DataFrame
tqdm.pandas()  # Use tqdm with pandas
sentiment_scores = df['review_description'].progress_apply(calculate_sentiment_bert)

# Merge sentiment scores into the DataFrame
df = pd.concat([df, pd.DataFrame(sentiment_scores)], axis=1)

df.to_csv(r'C:\Projects\Instagram Reviews\postbertmodel.csv')  # to be used for visualizations to see sentiment versus star rating
 

