import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv("spam.csv")

print(df.head())
print(df["Prediction"].value_counts())

X = df.drop(columns=["Email No.", "Prediction"])
y = df["Prediction"]

X = X.fillna(0)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model_without_smoothing = MultinomialNB(alpha=0)

model_without_smoothing.fit(X_train, y_train)

y_pred = model_without_smoothing.predict(X_test)

print("\nWithout Laplace Smoothing")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(cm)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Not Spam", "Spam"],
    yticklabels=["Not Spam", "Spam"]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Without Laplace Smoothing")
plt.savefig(
    "q3_confusion_matrix_without_smoothing.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()

model_with_smoothing = MultinomialNB(alpha=1.0)

model_with_smoothing.fit(X_train, y_train)

y_pred_smooth = model_with_smoothing.predict(X_test)

print("\nWith Laplace Smoothing")
print("Accuracy:", accuracy_score(y_test, y_pred_smooth))
print(classification_report(y_test, y_pred_smooth))

cm_smooth = confusion_matrix(y_test, y_pred_smooth)

print("Confusion Matrix:")
print(cm_smooth)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm_smooth,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Not Spam", "Spam"],
    yticklabels=["Not Spam", "Spam"]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("With Laplace Smoothing")

plt.savefig(
    "q3_confusion_matrix_with_smoothing.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()