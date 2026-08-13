import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FactorAnalysis, FastICA
from sklearn.mixture import GaussianMixture

print("Shaik Rafiqhuddin-192525129")
df = pd.read_csv("wine_samples.csv")

print(df.head())
print(df.isnull().sum())

X = df[
    [
        "Alcohol",
        "MalicAcid",
        "Ash",
        "Alcalinity",
        "Magnesium",
        "Phenols"
    ]
]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("PCA Explained Variance:")
print(pca.explained_variance_ratio_)

fa = FactorAnalysis(n_components=2, random_state=42)
X_fa = fa.fit_transform(X_scaled)

ica = FastICA(
    n_components=2,
    random_state=42,
    max_iter=1000
)

X_ica = ica.fit_transform(X_scaled)

gmm = GaussianMixture(
    n_components=3,
    random_state=42
)

gmm_labels = gmm.fit_predict(X_pca)

plt.figure(figsize=(8, 6))
plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=gmm_labels,
    s=100
)

plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("GMM Clustering after PCA")
plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(
    X_fa[:, 0],
    X_fa[:, 1],
    s=100
)

plt.xlabel("Factor 1")
plt.ylabel("Factor 2")
plt.title("Factor Analysis")
plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(
    X_ica[:, 0],
    X_ica[:, 1],
    s=100
)

plt.xlabel("Independent Component 1")
plt.ylabel("Independent Component 2")
plt.title("Independent Component Analysis")
plt.show()

print("PCA Shape:", X_pca.shape)
print("Factor Analysis Shape:", X_fa.shape)
print("ICA Shape:", X_ica.shape)
print("GMM Labels:", gmm_labels)