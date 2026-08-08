import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("Housing.csv")

print(df.shape)
print(df.head())
print(df.isnull().sum())

numeric_df = df.select_dtypes(include=np.number)

plt.figure(figsize=(10, 7))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.show()

numeric_df.plot(
    kind="box",
    subplots=True,
    layout=(3, 3),
    figsize=(15, 10),
    sharex=False
)
plt.show()

X = df[["area", "bedrooms", "bathrooms", "stories", "parking"]]
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("Original Model")
print("MAE:", mae)
print("RMSE:", rmse)
print("R2:", r2)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

scaled_model = LinearRegression()
scaled_model.fit(X_train_scaled, y_train)

y_pred_scaled = scaled_model.predict(X_test_scaled)

scaled_mae = mean_absolute_error(y_test, y_pred_scaled)
scaled_rmse = np.sqrt(mean_squared_error(y_test, y_pred_scaled))
scaled_r2 = r2_score(y_test, y_pred_scaled)

print("Scaled Model")
print("MAE:", scaled_mae)
print("RMSE:", scaled_rmse)
print("R2:", scaled_r2)

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual House Price")
plt.ylabel("Predicted House Price")
plt.title("Actual vs Predicted House Prices")
plt.show()