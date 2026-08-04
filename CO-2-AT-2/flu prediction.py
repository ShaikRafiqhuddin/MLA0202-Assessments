import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB

data = {
    'Fever': ['Yes','Yes','No','Yes','No','Yes','No','Yes'],
    'Cough': ['Yes','No','Yes','Yes','No','Yes','No','No'],
    'Headache': ['Yes','Yes','No','Yes','No','No','No','Yes'],
    'Flu': ['Yes','Yes','No','Yes','No','Yes','No','No']
}

df = pd.DataFrame(data)

encoder = LabelEncoder()

for column in df.columns:
    df[column] = encoder.fit_transform(df[column])

X = df[['Fever','Cough','Headache']]
y = df['Flu']

model = GaussianNB()
model.fit(X, y)

new_patient = [[1, 1, 1]]

prediction = model.predict(new_patient)

if prediction[0] == 1:
    print("Patient has Flu")
else:
    print("Patient does not have Flu")
