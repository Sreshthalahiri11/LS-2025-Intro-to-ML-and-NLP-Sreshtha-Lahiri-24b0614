import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from gensim.models import KeyedVectors

nltk.download('punkt')
nltk.download('stopwords')

# Load dataset
df = pd.read_csv('spam.csv', encoding='latin-1')[['v1', 'v2']]
df.columns = ['Label', 'Message']

# Preprocessing
stop_words = set(stopwords.words('english'))
def preprocess(text):
    tokens = word_tokenize(text.lower())
    return [word for word in tokens if word.isalpha() and word not in stop_words]

df['Tokens'] = df['Message'].apply(preprocess)

# Load Google News Word2Vec (binary=True for .bin format)
w2v = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)

# Averaging Word Vectors
def vectorize(tokens):
    valid_vecs = [w2v[word] for word in tokens if word in w2v]
    return np.mean(valid_vecs, axis=0) if valid_vecs else np.zeros(300)

df['Vector'] = df['Tokens'].apply(vectorize)

# Split data
X = np.vstack(df['Vector'].values)
y = df['Label'].map({'ham': 0, 'spam': 1}).values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train classifier
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate
preds = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, preds))

# Prediction function
def predict_message_class(model, w2v_model, message):
    tokens = preprocess(message)
    vec = vectorize(tokens)
    pred = model.predict([vec])[0]
    return 'spam' if pred == 1 else 'ham'