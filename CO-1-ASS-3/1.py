from sklearn.datasets import load_breast_cancer
import pandas as pd

data = load_breast_cancer()

df = pd.DataFrame(data.data, columns=data.feature_names)

df["Diagnosis"] = data.target

df["Diagnosis"] = df["Diagnosis"].map({
    0: "Malignant",
    1: "Benign"
})
print("Counts:")
print(df["Diagnosis"].value_counts())
print("\nProbabilities:")
print(df["Diagnosis"].value_counts(normalize=True))
