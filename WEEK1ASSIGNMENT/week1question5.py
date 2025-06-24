import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


texts = ["Good feedback " + str(i) for i in range(50)] + ["Bad feedback " + str(i) for i in range(50)]
labels = ["good"] * 50 + ["bad"] * 50
df = pd.DataFrame({"Text": texts, "Label": labels})


vectorizer = TfidfVectorizer(max_features=300, lowercase=True, stop_words="english")
X = vectorizer.fit_transform(df["Text"])
y = df["Label"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)
y_pred = lr_model.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))


def text_preprocess_vectorize(texts, vectorizer):
    return vectorizer.transform(texts)

example_texts = ["Amazing product!", "Terrible experience."]
example_vectors = text_preprocess_vectorize(example_texts, vectorizer)
print("Example Text Vectors:\n", example_vectors)
