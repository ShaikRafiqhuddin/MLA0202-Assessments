import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.decomposition import FactorAnalysis
from sklearn.decomposition import FastICA
wine = load_wine()
X = wine.data
y = wine.target
print("Dataset Shape:", X.shape)
print("Number of Features:", X.shape[1])
print("\nFeature Names:")
print(wine.feature_names)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)

fa = FactorAnalysis(
    n_components=2,
    random_state=42
)

X_fa = fa.fit_transform(X_scaled)

ica = FastICA(
    n_components=2,
    random_state=42,
    max_iter=2000,
    whiten="unit-variance"
)

X_ica = ica.fit_transform(X_scaled)

print("\nOriginal Shape:", X_scaled.shape)
print("PCA Shape:", X_pca.shape)
print("FA Shape:", X_fa.shape)
print("ICA Shape:", X_ica.shape)

print("\nPCA Explained Variance Ratio:")
print(
    np.round(
        pca.explained_variance_ratio_,
        4
    )
)

total_variance = pca.explained_variance_ratio_.sum()

print(
    "Total Variance:",
    round(total_variance, 4)
)

print(
    "Percentage:",
    round(total_variance * 100, 2),
    "%"
)

pca_loadings = pd.DataFrame(
    pca.components_.T,
    index=wine.feature_names,
    columns=["PC1", "PC2"]
)

print("\nPCA Loadings:")
print(pca_loadings.round(4))

fa_loadings = pd.DataFrame(
    fa.components_.T,
    index=wine.feature_names,
    columns=["Factor1", "Factor2"]
)

print("\nFactor Analysis Loadings:")
print(fa_loadings.round(4))

ica_components = pd.DataFrame(
    ica.components_,
    columns=wine.feature_names,
    index=["IC1", "IC2"]
)

print("\nICA Components:")
print(ica_components.round(4))

plt.figure(figsize=(8, 6))

scatter = plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=y,
    cmap="viridis",
    s=50
)

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Wine Dataset - PCA")
plt.colorbar(scatter, label="Wine Class")
plt.show()

plt.figure(figsize=(8, 6))

scatter = plt.scatter(
    X_fa[:, 0],
    X_fa[:, 1],
    c=y,
    cmap="viridis",
    s=50
)

plt.xlabel("Factor 1")
plt.ylabel("Factor 2")
plt.title("Wine Dataset - Factor Analysis")
plt.colorbar(scatter, label="Wine Class")
plt.show()

plt.figure(figsize=(8, 6))

scatter = plt.scatter(
    X_ica[:, 0],
    X_ica[:, 1],
    c=y,
    cmap="viridis",
    s=50
)

plt.xlabel("Independent Component 1")
plt.ylabel("Independent Component 2")
plt.title("Wine Dataset - ICA")
plt.colorbar(scatter, label="Wine Class")
plt.show()