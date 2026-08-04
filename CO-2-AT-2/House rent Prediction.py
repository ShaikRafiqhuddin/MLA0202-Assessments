import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("House_Rent_Dataset.csv")

X = df[['Size', 'BHK', 'City']]
y = df['Rent']

X = pd.get_dummies(X, columns=['City'], drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("R² Score:", round(r2_score(y_test, y_pred), 4))
print("MAE:", round(mean_absolute_error(y_test, y_pred), 2))
print("RMSE:", round(mean_squared_error(y_test, y_pred) ** 0.5, 2))

new_apartment = pd.DataFrame({
    'Size': [1200],
    'BHK': [3],
    'City': ['Mumbai']
})

new_apartment = pd.get_dummies(new_apartment)
new_apartment = new_apartment.reindex(columns=X_train.columns, fill_value=0)

predicted_rent = model.predict(new_apartment)

print("Predicted Rent: ₹{:.2f}".format(predicted_rent[0]))
