print("SHAIK RAFIQHUDDIN - 192525129")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn_crfsuite

def word_features(sentence, index):

    word = sentence[index]

    features = {
        "word.lower()": word.lower(),
        "word.isupper()": word.isupper(),
        "word.istitle()": word.istitle(),
        "word.isdigit()": word.isdigit(),
        "word.isalnum()": word.isalnum(),
        "word.has_digit": any(char.isdigit() for char in word),
        "word.prefix2": word[:2],
        "word.prefix3": word[:3],
        "word.suffix2": word[-2:],
        "word.suffix3": word[-3:],
        "word.length": len(word)
    }

    if index > 0:

        previous_word = sentence[index - 1]

        features.update({
            "-1:word.lower()": previous_word.lower(),
            "-1:word.istitle()": previous_word.istitle(),
            "-1:word.isdigit()": previous_word.isdigit()
        })

    else:
        features["BOS"] = True

    if index < len(sentence) - 1:

        next_word = sentence[index + 1]

        features.update({
            "+1:word.lower()": next_word.lower(),
            "+1:word.istitle()": next_word.istitle(),
            "+1:word.isdigit()": next_word.isdigit()
        })

    else:
        features["EOS"] = True

    return features


def sentence_features(sentence):
    return [
        word_features(sentence, i)
        for i in range(len(sentence))
    ]


training_sentences = [
    ["Ravi", "ordered", "Nike", "Shoes", "with", "order", "ORD12345"],
    ["Aisha", "bought", "Apple", "iPhone", "under", "order", "ORD23456"],
    ["Rahul", "requested", "Samsung", "Galaxy", "for", "order", "ORD34567"],
    ["Priya", "purchased", "Dell", "Laptop", "order", "ORD45678"],
    ["Arjun", "ordered", "Sony", "Headphones", "with", "order", "ORD56789"],
    ["Meera", "bought", "Nike", "Running", "Shoes", "order", "ORD67890"],
    ["Kiran", "ordered", "Apple", "MacBook", "under", "order", "ORD78901"],
    ["Neha", "purchased", "Samsung", "Tablet", "order", "ORD89012"],
    ["Vikram", "ordered", "Dell", "Monitor", "with", "order", "ORD90123"],
    ["Anjali", "bought", "Sony", "Camera", "under", "order", "ORD11223"]
]

training_labels = [
    ["B-CUST", "O", "B-PROD", "I-PROD", "O", "O", "B-ORD"],
    ["B-CUST", "O", "B-PROD", "I-PROD", "O", "O", "B-ORD"],
    ["B-CUST", "O", "B-PROD", "I-PROD", "O", "O", "B-ORD"],
    ["B-CUST", "O", "B-PROD", "I-PROD", "O", "B-ORD"],
    ["B-CUST", "O", "B-PROD", "I-PROD", "O", "O", "B-ORD"],
    ["B-CUST", "O", "B-PROD", "I-PROD", "I-PROD", "O", "B-ORD"],
    ["B-CUST", "O", "B-PROD", "I-PROD", "O", "O", "B-ORD"],
    ["B-CUST", "O", "B-PROD", "I-PROD", "O", "B-ORD"],
    ["B-CUST", "O", "B-PROD", "I-PROD", "O", "O", "B-ORD"],
    ["B-CUST", "O", "B-PROD", "I-PROD", "O", "O", "B-ORD"]
]

X_train = [
    sentence_features(sentence)
    for sentence in training_sentences
]

y_train = training_labels

crf = sklearn_crfsuite.CRF(
    algorithm="lbfgs",
    c1=0.1,
    c2=0.1,
    max_iterations=100,
    all_possible_transitions=True
)

crf.fit(X_train, y_train)

new_sentence = [
    "Customer",
    "Sameer",
    "ordered",
    "Nike",
    "Shoes",
    "with",
    "order",
    "ORD99887"
]

X_test = [
    sentence_features(new_sentence)
]

prediction = crf.predict(X_test)[0]

print("\nPredicted Named Entities:")

for word, label in zip(new_sentence, prediction):
    print(f"{word:12s} -> {label}")

result_df = pd.DataFrame({
    "Word": new_sentence,
    "Predicted_Label": prediction
})

print("\nPrediction Data:")
print(result_df)

label_mapping = {
    "O": 0,
    "B-CUST": 1,
    "I-CUST": 2,
    "B-ORD": 3,
    "I-ORD": 4,
    "B-PROD": 5,
    "I-PROD": 6
}

numeric_labels = [
    label_mapping[label]
    for label in prediction
]

plt.figure(figsize=(12, 4))
sns.heatmap(
    [numeric_labels],
    annot=True,
    fmt="d",
    cmap="viridis",
    xticklabels=new_sentence,
    yticklabels=["Predicted Label"]
)
plt.title("CRF Named Entity Recognition")
plt.xlabel("Words")
plt.ylabel("Sequence")
plt.tight_layout()
plt.show()

print("\nCRF Labels:")
print(crf.classes_)