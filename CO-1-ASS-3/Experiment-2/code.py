import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Fetch SMS Spam Collection Dataset
sms_spam = fetch_ucirepo(id=228)

# Features and Target
X = sms_spam.data.features
y = sms_spam.data.targets

# Convert target to Series
if isinstance(y, pd.DataFrame):
    y = y.iloc[:, 0]

# Display first 5 rows
print("Dataset Preview:")
print(pd.concat([X, y], axis=1).head())

# Convert text into numerical vectors
vectorizer = CountVectorizer()

# The dataset has one text column
X_vector = vectorizer.fit_transform(X.iloc[:, 0])

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_vector,
    y,
    test_size=0.2,
    random_state=42
)

# Train Naive Bayes Model
model = MultinomialNB()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
print("\nAccuracy:", accuracy_score(y_test, y_pred))

# Classification Report
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Actual vs Predicted
result = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})

print("\nActual vs Predicted (First 10 Rows):")
print(result.head(10))

# Test with Custom Messages
messages = [
    "Congratulations! You have won a free iPhone. Click here to claim your prize.",
    "Hi, are we meeting at 6 PM today?"
]

messages_vector = vectorizer.transform(messages)

predictions = model.predict(messages_vector)

print("\nCustom Message Predictions:")
for msg, pred in zip(messages, predictions):
    print(f"\nMessage: {msg}")
    print(f"Prediction: {pred}")
