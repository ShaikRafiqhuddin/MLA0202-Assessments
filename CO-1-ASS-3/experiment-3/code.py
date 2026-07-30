import pandas as pd
from math import log2

# Play Tennis Dataset
data = {
    "Outlook": ["Sunny","Sunny","Overcast","Rain","Rain","Rain","Overcast",
                "Sunny","Sunny","Rain","Sunny","Overcast","Overcast","Rain"],
    "Temperature": ["Hot","Hot","Hot","Mild","Cool","Cool","Cool",
                    "Mild","Cool","Mild","Mild","Mild","Hot","Mild"],
    "Humidity": ["High","High","High","High","Normal","Normal","Normal",
                 "High","Normal","Normal","Normal","High","Normal","High"],
    "Wind": ["Weak","Strong","Weak","Weak","Weak","Strong","Strong",
             "Weak","Weak","Weak","Strong","Strong","Weak","Strong"],
    "PlayTennis": ["No","No","Yes","Yes","Yes","No","Yes",
                   "No","Yes","Yes","Yes","Yes","Yes","No"]
}

df = pd.DataFrame(data)

# Function to calculate Entropy
def entropy(target):
    values = target.value_counts(normalize=True)
    return -sum(p * log2(p) for p in values)

# Function to calculate Information Gain
def information_gain(df, attribute, target):
    total_entropy = entropy(df[target])

    weighted_entropy = 0

    for value in df[attribute].unique():
        subset = df[df[attribute] == value]
        weighted_entropy += (len(subset) / len(df)) * entropy(subset[target])

    return total_entropy - weighted_entropy

# Target Entropy
target_entropy = entropy(df["PlayTennis"])

print("Entropy of PlayTennis:", round(target_entropy, 4))

print("\nInformation Gain:")

gains = {}

for column in df.columns[:-1]:
    gain = information_gain(df, column, "PlayTennis")
    gains[column] = gain
    print(f"{column}: {gain:.4f}")

best_attribute = max(gains, key=gains.get)

print("\nBest Attribute (Root Node):", best_attribute)
