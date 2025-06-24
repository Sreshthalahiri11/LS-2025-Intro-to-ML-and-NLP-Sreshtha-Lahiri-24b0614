corpus = [
    'the sun is a star',
    'the moon is a satellite',
    'the sun and moon are celestial bodies'
]

documents = [doc.lower().split() for doc in corpus]
from collections import defaultdict

def compute_tf(doc):
    tf_scores = []
    for words in doc:
        tf = defaultdict(int)
        total = len(words)
        for word in words:
            tf[word] += 1
        tf_scores.append({word: count/total for word, count in tf.items()})
    return tf_scores

tf_scores = compute_tf(documents)
import math

def compute_idf(doc):
    N = len(doc)
    df = defaultdict(int)
    for words in doc:
        unique = set(words)
        for word in unique:
            df[word] += 1
    idf = {word: math.log(N / (1 + freq)) for word, freq in df.items()}
    return idf

idf_scores = compute_idf(documents)
def compute_tfidf(tf_scores, idf_scores):
    tfidf = []
    for tf in tf_scores:
        doc_tfidf = {}
        for word, val in tf.items():
            doc_tfidf[word] = val * idf_scores.get(word, 0.0)
        tfidf.append(doc_tfidf)
    return tfidf

tfidf_manual = compute_tfidf(tf_scores, idf_scores)
for i, doc in enumerate(tfidf_manual):
    print(f"TF-IDF for Document {i+1}: {doc}")
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# CountVectorizer
cv = CountVectorizer()
cv_matrix = cv.fit_transform(corpus)
print("CountVectorizer output:\n", cv_matrix.toarray())

# TfidfVectorizer
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(corpus)
print("TfidfVectorizer output:\n", tfidf_matrix.toarray())