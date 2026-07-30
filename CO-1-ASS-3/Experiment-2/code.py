import os
import urllib.request
import zipfile
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Download dataset if not already present
if not os.path.exists("SMSSpamCollection"):
    print("Downloading dataset...")
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
    urllib.request.urlretrieve(url, "smsspamcollection.zip")

    with zipfile.ZipFile("smsspamcollection.zip", "r") as zip_ref:
        zip_ref.extractall()

print("Dataset loaded!")

# Load dataset
df = pd.read_csv(
    "SMSSpamCollection",
    sep="\t",
    names=["label", "message"]
)

# Features and Labels
X = df["message"]
y = df["label"]

# Convert text to vectors
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Test custom message
msg = ["Congratulations! You won a free iPhone. Click here to claim."]
msg_vec = vectorizer.transform(msg)

print("\nPrediction:", model.predict(msg_vec)[0])
