import numpy as np

X = np.array([
    [1000, 2],
    [1500, 3],
    [800, 2],
    [1200, 3],
    [2000, 4]
], dtype=float)

y = np.array([50, 75, 40, 60, 90], dtype=float)

X_aug = np.column_stack([np.ones(len(X)), X])
beta = np.linalg.lstsq(X_aug, y, rcond=None)[0]

print("Q1: Multiple Linear Regression")
print(f"Model: Price = {beta[0]:.4f} + {beta[1]:.4f}*Area + {beta[2]:.4f}*Bedrooms")
print(f"Intercept = {beta[0]:.4f} lakh")
print(f"Area coefficient = {beta[1]:.4f} lakh/sq.ft")
print(f"Bedroom coefficient = {beta[2]:.4f} lakh/bedroom")

predicted = X_aug @ beta
print("\nPredicted prices:", np.round(predicted, 4))
