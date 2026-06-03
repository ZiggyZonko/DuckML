import pandas as pd
import math

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB # Naive Bayes Classifier, probability algorithm for classification.

df = pd.read_csv("dataset.csv") # load dataset

X = df["text"] # input
y = df["label"] # label, happy, sad etc.

vectorizer = CountVectorizer() # learn phrases and words, one or two gram words.

X_vectors = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_vectors,
    y,
    test_size=0.1,
    random_state=42
)

model = MultinomialNB()
model.fit(X_train, y_train)

print("Accuracy:", math.ceil(model.score(X_test, y_test)*100), "%")

predictions = model.predict(X_test)

""" for actual, predicted in zip(y_test, predictions):
    print(f"Actual: {actual:8} | Predicted: {predicted}") """

while True:
    message = input("\nEnter a message: ")

    message_vector = vectorizer.transform([message])

    prediction = model.predict(message_vector)

    probabilities = model.predict_proba(message_vector)[0]

    print("\n=== QuackJudge ===")
    print("Verdict:", prediction[0] + " Duck 🦆")
    print("\nProbability Breakdown")

    for label, probability in sorted(
        zip(model.classes_, probabilities),
        key=lambda x: x[1],
        reverse=True
    ):
        bar = "█" * int(probability * 20)
        print(f"{label:8} {probability * 100:6.2f}% {bar}")

    print("\nVectorized Message:",
        vectorizer.transform(
        [message]
    ).toarray()
)