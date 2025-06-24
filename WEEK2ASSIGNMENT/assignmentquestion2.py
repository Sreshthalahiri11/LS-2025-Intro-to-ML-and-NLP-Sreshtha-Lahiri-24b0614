import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.linear_model import LogisticRegression
import re
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
nltk.download('wordnet')
nltk.download('omw-1.4')
lemmatizer = WordNetLemmatizer()
nltk.download('stopwords')
from gensim.models import KeyedVectors

w2v = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def clean_tweet(text):
    text = text.lower()
    text = re.sub(r"http\S+|@\S+|#\S+|[^a-z\s]", '', text)
    words = word_tokenize(text)
    lemmatized = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return lemmatized

def vectorize(tokens):
    valid_vecs = [w2v[word] for word in tokens if word in w2v]
    return np.mean(valid_vecs, axis=0) if valid_vecs else np.zeros(300)
df2 = pd.read_csv('Tweets.csv')[['airline_sentiment', 'text']]
df2['Tokens'] = df2['text'].apply(clean_tweet)


df2['Vector'] = df2['Tokens'].apply(vectorize)


X2 = np.vstack(df2['Vector'].values)
y2 = df2['airline_sentiment']
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=42)

model2 = LogisticRegression(max_iter=1000)
model2.fit(X2_train, y2_train)


preds2 = model2.predict(X2_test)
print("Accuracy:", accuracy_score(y2_test, preds2))


def predict_tweet_sentiment(model, w2v_model, tweet):
    tokens = clean_tweet(tweet)
    vec = vectorize(tokens)
    return model.predict([vec])[0]