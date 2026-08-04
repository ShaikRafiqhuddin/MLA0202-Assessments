import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay

df = pd.read_csv("spam.csv", encoding="latin-1")

df = df[['v1', 'v2']]
df.columns = ['label', 'message']

df['label'] = df['label'].map({'ham': 0, 'spam': 1})

X = df['message']
y = df['label']

vectorizer = TfidfVectorizer(stop_words='english')

X = vectorizer.fit_transform(X)

model = LogisticRegression(max_iter=1000)

model.fit(X, y)

y_pred = model.predict(X)

print("Accuracy:", round(accuracy_score(y, y_pred), 4))
print("\nClassification Report")
print(classification_report(y, y_pred))

comparison = pd.DataFrame({
    "Actual": y,
    "Predicted": y_pred
})

print("\nActual vs Predicted")
print(comparison.head(20))

email = ["Congratulations! You have won a FREE iPhone. Click here to claim your prize now."]

email_vector = vectorizer.transform(email)

prediction = model.predict(email_vector)

if prediction[0] == 1:
    print("\nPrediction: Spam Email")
else:
    print("\nPrediction: Non-Spam Email")

ConfusionMatrixDisplay.from_predictions(
    y,
    y_pred,
    display_labels=["Ham", "Spam"],
    cmap="Blues"
)

plt.title("Confusion Matrix")
plt.show()
