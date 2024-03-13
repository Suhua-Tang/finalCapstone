# install virtual environment and packages
import spacy
from textblob import TextBlob
import pandas as pd

nlp = spacy.load("en_core_web_sm")
df = pd.read_csv("amazon_product_reviews.csv", low_memory= False)

# a look through the data, I think the reviews.text and reviews.titles are revelant
reviews_data = df[["reviews.text", "reviews.title"]]

# a peek into the data and find out the number of null data 
print(reviews_data.isnull().sum())

# there is one null entry in reviews.text and 6 null entries in reviews.titles
# clear the null entry
reviews_data = reviews_data.dropna()

# function to tokenize texts, which exclude function workds and punctuation
def preprocess(text):
    doc = nlp(text)
    return ' '.join([token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct])

# preprocossing the reviews
reviews_data["preprocessed_reviews"] = reviews_data["reviews.text"].apply(preprocess)

# function to analyse the sentiments of the reviews with TextBlob
def analyze_polarity(text):
    blob = TextBlob(text)
    polarity = blob.polarity
    return polarity

# apply the function to new column preprocessed review.
polarity_score = reviews_data["preprocessed_reviews"].apply(analyze_polarity)

# assess the sentiments according to the polarity_score, return the sentiment result
def analyze_sentiment (text):
    polarity_score = analyze_polarity(text)
    if polarity_score > 0:
        sentiment = 'positive'
    elif polarity_score < 0:
        sentiment = 'negative'
    else:
        sentiment = 'neutral'
    return sentiment 

sentiment = reviews_data["preprocessed_reviews"].apply(analyze_sentiment)

# loop through a sample of 10 preprocessed reviews and return its polarity scores and sentiments

for i in range(15):
    print(f"Review: {reviews_data.iloc[i]}\nPolarity score: {polarity_score.iloc[i]}\nSentiment: {sentiment.iloc[i]}")


my_review_of_choice_1 = nlp(reviews_data['reviews.text'][0])

my_review_of_choice_2 = nlp(reviews_data['reviews.text'][14])

similarity_score = my_review_of_choice_1.similarity(my_review_of_choice_2)

print("review 1 :", my_review_of_choice_1, "\n review 2: ", my_review_of_choice_2, "\nThe similarity score : ", similarity_score)