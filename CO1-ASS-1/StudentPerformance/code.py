import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
student_performance = fetch_ucirepo(id=320)
X = student_performance.data.features
y = student_performance.data.targets

X = pd.get_dummies(X, drop_first=True)

if isinstance(y, pd.DataFrame):
    y = y.iloc[:, 0]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("R² Score:", r2_score(y_test, y_pred))

result = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})
print(result.head(10))
