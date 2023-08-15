import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from gensim import corpora, models
from tqdm import tqdm

# Load the DataFrame from the CSV file
df = pd.read_csv(r'C:\Projects\Instagram Reviews\instagram.csv')

# Separate 1-star and 5-star reviews
one_star_reviews = df[df['rating'] == 1]['review_description']
five_star_reviews = df[df['rating'] == 5]['review_description']

# Preprocessing function
def preprocess(text):
    # Tokenization
    tokens = word_tokenize(text.lower())

    # Remove stopwords and punctuation
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word.isalnum() and word not in stop_words]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return tokens

# Initialize tqdm for preprocessing
with tqdm(total=len(one_star_reviews), desc="Preprocessing 1-Star Reviews") as pbar:
    # Preprocess 1-star reviews
    one_star_tokens = [preprocess(review) for review in one_star_reviews]
    pbar.update(len(one_star_reviews))

# Initialize tqdm for preprocessing
with tqdm(total=len(five_star_reviews), desc="Preprocessing 5-Star Reviews") as pbar:
    # Preprocess 5-star reviews
    five_star_tokens = [preprocess(review) for review in five_star_reviews]
    pbar.update(len(five_star_reviews))

# Create dictionary and corpus for 1-star reviews
one_star_dictionary = corpora.Dictionary(one_star_tokens)
one_star_corpus = [one_star_dictionary.doc2bow(tokens) for tokens in one_star_tokens]

# Create dictionary and corpus for 5-star reviews
five_star_dictionary = corpora.Dictionary(five_star_tokens)
five_star_corpus = [five_star_dictionary.doc2bow(tokens) for tokens in five_star_tokens]

# Initialize tqdm for topic modeling
with tqdm(total=len(one_star_corpus), desc="Topic Modeling for 1-Star Reviews") as pbar:
    # Perform topic modeling using LDA for 1-star reviews
    one_star_lda_model = models.LdaModel(one_star_corpus, num_topics=5, id2word=one_star_dictionary, passes=15)
    pbar.update(len(one_star_corpus))

# Initialize tqdm for topic modeling
with tqdm(total=len(five_star_corpus), desc="Topic Modeling for 5-Star Reviews") as pbar:
    # Perform topic modeling using LDA for 5-star reviews
    five_star_lda_model = models.LdaModel(five_star_corpus, num_topics=5, id2word=five_star_dictionary, passes=15)
    pbar.update(len(five_star_corpus))

# Print topics for 1-star reviews
print("Topics for 1-star reviews:")
for topic_id, topic_words in one_star_lda_model.print_topics(num_topics=-1, num_words=5):
    print(f"Topic {topic_id}: {topic_words}")

# Print topics for 5-star reviews
print("\nTopics for 5-star reviews:")
for topic_id, topic_words in five_star_lda_model.print_topics(num_topics=-1, num_words=5):
    print(f"Topic {topic_id}: {topic_words}")

