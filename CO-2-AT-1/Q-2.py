import numpy as np

X = np.array([
    [3, 2],
    [2, 1],
    [1, 0],
    [3, 3],
    [0, 1]
], dtype=float)

y = np.array([1, 1, 0, 1, 0], dtype=float)

X_aug = np.column_stack([np.ones(len(X)), X])
beta = np.linalg.lstsq(X_aug, y, rcond=None)[0]

print("Q2: Linear Classification Model")
print(f"Model: Score = {beta[0]:.4f} + {beta[1]:.4f}*Offer + {beta[2]:.4f}*Win")
print("Decision rule: Spam if Score >= 0.5, otherwise Not Spam")
print(f"Decision boundary: {beta[1]:.4f}*Offer + {beta[2]:.4f}*Win = {0.5-beta[0]:.4f}")

new_email = np.array([1, 2, 1], dtype=float)
score = new_email @ beta
print(f"\nNew email: Offer=2, Win=1")
print(f"Score = {score:.4f}")
print("Classification:", "Spam (1)" if score >= 0.5 else "Not Spam (0)")
