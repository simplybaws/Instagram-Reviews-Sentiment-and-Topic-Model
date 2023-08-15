# Instagram-Reviews-Sentiment-and-Topic-Model
 Performing sentiment analysis using BERT model, and topic modeling


 Python Files -- 
 -  datacleaning.py - File that cleans the data, lowercase, stopwords, tokens, lemmatization, etc.
 -  NLTK Sentiment.py - uses NLTK sentiment analyzer, did not get very great results so switched to new model
 -  bert.py - uses a huggingface bert model to perform sentiment analysis
 -  sentimentvisualizations.py - uses the bert model results to visualize the average sentiment scores for each star rating
 -  topicmodel.py  -  Performs a topic model on 1 and 5 star reviews to see which topics are most prevalent in each, to try and gain understanding into what makes a 5 star and 1 star review.

 CSV Files --
 -  instagram.csv - Original csv dataset file of instagram reviews pulled directly from the play store
 -  postlemmreviews.csv - Instagram reviews after performing text cleaning techniques
 -  postbertmodel.csv - Instagram reviews along with sentiment scores provided by bert, the scores are saved into a dict and later used.
