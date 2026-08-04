import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

df = pd.read_csv("rent_dataset.csv")

print("First 5 Rows:")
print(df.head())

print("\nColumns:")
print(df.columns)

X = df[['Size', 'BHK', 'City']]
y = df['Rent']

X = pd.get_dummies(X, columns=['City'], drop_first=True)

model = LinearRegression()
model.fit(X, y)

y_pred = model.predict(X)

print("\nModel Performance")
print("--------------------------")
print("R² Score :", round(r2_score(y, y_pred), 4))
print("MAE      :", round(mean_absolute_error(y, y_pred), 2))
print("RMSE     :", round(mean_squared_error(y, y_pred) ** 0.5, 2))

comparison = pd.DataFrame({
    "Actual Rent": y,
    "Predicted Rent": y_pred.round(2)
})

print("\nActual vs Predicted Rent")
print(comparison)

new_apartment = pd.DataFrame({
    "Size": [1250],
    "BHK": [3],
    "City": ["Mumbai"]
})

new_apartment = pd.get_dummies(new_apartment)
new_apartment = new_apartment.reindex(columns=X.columns, fill_value=0)

predicted_rent = model.predict(new_apartment)

print("\nPredicted Rent for New Apartment")
print("--------------------------------")
print(f"₹ {predicted_rent[0]:.2f}")

plt.figure(figsize=(8,6))
plt.scatter(y, y_pred, color="blue", s=80)

plt.plot(
    [y.min(), y.max()],
    [y.min(), y.max()],
    color="red",
    linewidth=2
)

plt.xlabel("Actual Rent")
plt.ylabel("Predicted Rent")
plt.title("Actual vs Predicted Rent")
plt.grid(True)
plt.show()
