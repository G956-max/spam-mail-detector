import pandas as pd

data = pd.read_csv("SMSSpamCollection",sep="\t",header=None,names=["label", "message"])

data["message"] = data["message"].str.lower()

data["label"] = data["label"].map({"ham": 0,"spam": 1})

print("\nLabels:")
print(data["label"].value_counts())

from sklearn.model_selection import train_test_split
X = data["message"]
Y = data["label"]
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,random_state=42,stratify=Y)

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words="english")

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("\nTF-IDF Training Shape:", X_train_tfidf.shape)

from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(X_train_tfidf, Y_train)

Y_pred = model.predict(X_test_tfidf)

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score

accuracy = accuracy_score(Y_test, Y_pred)
precision = precision_score(Y_test, Y_pred)
f1 = f1_score(Y_test, Y_pred)

print("\nAccuracy :", accuracy * 100, "%")
print("Precision:", precision * 100, "%")
print("F1 Score :", f1 * 100, "%")

messages = ["Congratulations! You won a free lottery ticket.","Hey, are you coming to college today?"]

messages_tfidf = vectorizer.transform(messages)
predictions = model.predict(messages_tfidf)
for message, prediction in zip(messages, predictions):
    if prediction == 1:
        print("\nMessage:", message)
        print("Prediction: SPAM")
    else:
        print("\nMessage:", message)
        print("Prediction: HAM")
