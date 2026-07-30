from sklearn.datasets import load_breast_cancer
import pandas as pd

# Load dataset
data = load_breast_cancer()

# Create DataFrame
df = pd.DataFrame(data.data, columns=data.feature_names)

# Add target column
df["Diagnosis"] = data.target

# Convert 0 and 1 into labels
df["Diagnosis"] = df["Diagnosis"].map({
    0: "Malignant",
    1: "Benign"
})

# Count samples
print("Counts:")
print(df["Diagnosis"].value_counts())

# Calculate probabilities
print("\nProbabilities:")
print(df["Diagnosis"].value_counts(normalize=True))
