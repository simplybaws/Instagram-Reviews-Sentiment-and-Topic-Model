import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import ast


file_path = r'C:\Projects\Instagram Reviews\postbertmodel.csv'
df = pd.read_csv(file_path)

def extract_sentiment_scores(row):
    sent_dict = ast.literal_eval(row)
    return sent_dict

df['sentiment_scores'] = df['sentvals'].apply(extract_sentiment_scores)

sent_columns = ['positive_score', 'negative_score', 'neutral_score']

for col in sent_columns:
    df[col] = df['sentiment_scores'].apply(lambda x: x[col])


#visualization
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))

# Group by star rating and calculate mean sentiment scores
grouped_df = df.groupby('rating')[sent_columns].mean().reset_index()

# Melt the DataFrame for plotting
melted_df = pd.melt(grouped_df, id_vars='rating', value_vars=sent_columns)

sns.barplot(data=melted_df, x='rating', y='value', hue='variable', palette=["green", "red", "gray"])
plt.xlabel('rating')
plt.ylabel('Mean Sentiment Score')
plt.title('Sentiment Distribution for Each Star Rating')
plt.legend(title='Sentiment')
plt.show()