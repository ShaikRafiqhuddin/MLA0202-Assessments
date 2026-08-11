import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import normalized_mutual_info_score

digits = load_digits()

X = digits.data
y = digits.target

print("Dataset Shape:", X.shape)
print("Number of Classes:", len(np.unique(y)))

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

k = 10

kmeans = KMeans(
    n_clusters=k,
    random_state=42,
    n_init=10
)

kmeans_labels = kmeans.fit_predict(X_scaled)

gmm = GaussianMixture(
    n_components=k,
    covariance_type="diag",
    random_state=42
)

gmm_labels = gmm.fit_predict(X_scaled)

kmeans_silhouette = silhouette_score(
    X_scaled,
    kmeans_labels
)

gmm_silhouette = silhouette_score(
    X_scaled,
    gmm_labels
)

kmeans_ari = adjusted_rand_score(
    y,
    kmeans_labels
)

gmm_ari = adjusted_rand_score(
    y,
    gmm_labels
)

kmeans_nmi = normalized_mutual_info_score(
    y,
    kmeans_labels
)

gmm_nmi = normalized_mutual_info_score(
    y,
    gmm_labels
)

print("\nK-Means")
print("Silhouette Score:", round(kmeans_silhouette, 4))
print("ARI:", round(kmeans_ari, 4))
print("NMI:", round(kmeans_nmi, 4))

print("\nGMM / EM")
print("Silhouette Score:", round(gmm_silhouette, 4))
print("ARI:", round(gmm_ari, 4))
print("NMI:", round(gmm_nmi, 4))

k_values = range(2, 11)

kmeans_scores = []
gmm_scores = []

for k in k_values:

    km = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    km_labels = km.fit_predict(X_scaled)

    kmeans_scores.append(
        silhouette_score(X_scaled, km_labels)
    )

    gm = GaussianMixture(
        n_components=k,
        covariance_type="diag",
        random_state=42
    )

    gm_labels = gm.fit_predict(X_scaled)

    gmm_scores.append(
        silhouette_score(X_scaled, gm_labels)
    )

plt.figure(figsize=(8, 5))

plt.plot(
    k_values,
    kmeans_scores,
    marker="o",
    label="K-Means"
)

plt.plot(
    k_values,
    gmm_scores,
    marker="s",
    label="GMM / EM"
)

plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")
plt.title("K-Means vs GMM")
plt.legend()
plt.grid(True)
plt.show()

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)

print("\nPCA Explained Variance:")
print(np.round(pca.explained_variance_ratio_, 4))

plt.figure(figsize=(8, 6))

scatter = plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=gmm_labels,
    cmap="tab10",
    s=15
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("Digits Clustering using GMM and PCA")
plt.colorbar(scatter, label="GMM Cluster")
plt.show()

plt.figure(figsize=(8, 6))

scatter = plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=kmeans_labels,
    cmap="tab10",
    s=15
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("Digits Clustering using K-Means and PCA")
plt.colorbar(scatter, label="K-Means Cluster")
plt.show()