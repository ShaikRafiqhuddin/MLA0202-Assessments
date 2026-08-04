import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay

data = {
    'Fever': ['Yes','Yes','No','Yes','No','Yes','No','Yes'],
    'Cough': ['Yes','No','Yes','Yes','No','Yes','No','No'],
    'Headache': ['Yes','Yes','No','Yes','No','No','No','Yes'],
    'Flu': ['Yes','Yes','No','Yes','No','Yes','No','No']
}

df = pd.DataFrame(data)

print("Dataset:")
print(df)

encoder = LabelEncoder()

for column in df.columns:
    df[column] = encoder.fit_transform(df[column])

X = df[['Fever', 'Cough', 'Headache']]
y = df['Flu']

model = GaussianNB()

model.fit(X, y)

y_pred = model.predict(X)

print("\nModel Performance")
print("--------------------------")
print("Accuracy:", round(accuracy_score(y, y_pred), 4))

print("\nClassification Report")
print(classification_report(y, y_pred))

comparison = pd.DataFrame({
    "Actual": y,
    "Predicted": y_pred
})

print("\nActual vs Predicted")
print(comparison)

new_patient = [[1, 1, 1]]

prediction = model.predict(new_patient)

if prediction[0] == 1:
    print("\nPrediction: Patient has Flu")
else:
    print("\nPrediction: Patient does not have Flu")

ConfusionMatrixDisplay.from_predictions(
    y,
    y_pred,
    display_labels=["No Flu", "Flu"],
    cmap="Blues"
)

plt.title("Naive Bayes - Confusion Matrix")
plt.show()
