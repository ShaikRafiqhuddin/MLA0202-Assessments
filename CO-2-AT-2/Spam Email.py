import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("spam.csv", encoding="latin-1")

df = df[['v1', 'v2']]
df.columns = ['label', 'message']

df['label'] = df['label'].map({'ham': 0, 'spam': 1})

X = df['message']
y = df['label']

vectorizer = TfidfVectorizer(stop_words='english')

X = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

email = ["URGENT! Congratulations! You have won a FREE iPhone. Click here now to claim your prize."]

email_vector = vectorizer.transform(email)

prediction = model.predict(email_vector)
probability = model.predict_proba(email_vector)

print("Ham Probability :", probability[0][0])
print("Spam Probability:", probability[0][1])

if prediction[0] == 1:
    print("Spam Email")
else:
    print("Non-Spam Email")
