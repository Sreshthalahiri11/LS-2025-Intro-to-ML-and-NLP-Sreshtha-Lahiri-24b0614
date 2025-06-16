
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import pandas as pd


Reviews = ["Positive review " + str(i) for i in range(50)] + ["Negative review " + str(i) for i in range(50)]
Sentiments = ["positive"] * 50 + ["negative"] * 50
df = pd.DataFrame({"Review": Reviews, "Sentiment": Sentiments})


vectorizer = CountVectorizer(max_features=500, stop_words="english")
X = vectorizer.fit_transform(df["Review"])
y = df["Sentiment"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = MultinomialNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))


def predict_review_sentiment(model, vectorizer, Review):
    review_vector = vectorizer.transform([Review])
    return model.predict(review_vector)[0]

example_review = "Positive review example"
print("Predicted Sentiment:", predict_review_sentiment(model, vectorizer, example_review))
