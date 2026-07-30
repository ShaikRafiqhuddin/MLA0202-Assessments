from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
sms = fetch_openml(name="sms_spam", version=1, as_frame=True)

X = sms.data.iloc[:, 0]
y = sms.target

# Convert text into numerical features
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train Naive Bayes model
model = MultinomialNB()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Classification Report
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Test a custom message
message = ["Congratulations! You have won a free iPhone. Click here to claim your prize."]

message_vector = vectorizer.transform(message)

prediction = model.predict(message_vector)

print("\nMessage:", message[0])
print("Prediction:", prediction[0])
