print("SHAIK RAFIQHUDDIN - 192525129")

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

image = np.array([
    [20, 25, 30, 35, 40],
    [22, 120, 125, 130, 42],
    [25, 118, 200, 135, 45],
    [28, 115, 140, 138, 48],
    [30, 35, 40, 45, 50]
], dtype=float)

print("Original Image:")
print(image)

rows, cols = image.shape

graph = nx.Graph()

for i in range(rows):
    for j in range(cols):
        graph.add_node((i, j))

        if j + 1 < cols:
            graph.add_edge((i, j), (i, j + 1))

        if i + 1 < rows:
            graph.add_edge((i, j), (i + 1, j))

updated_image = image.copy()

for i in range(rows):
    for j in range(cols):

        neighbors = []

        if i > 0:
            neighbors.append(image[i - 1, j])

        if i < rows - 1:
            neighbors.append(image[i + 1, j])

        if j > 0:
            neighbors.append(image[i, j - 1])

        if j < cols - 1:
            neighbors.append(image[i, j + 1])

        values = [image[i, j]] + neighbors

        updated_image[i, j] = np.mean(values)

print("\nUpdated Image:")
print(np.round(updated_image, 2))

print("\nNumber of Pixel Nodes:", graph.number_of_nodes())
print("Number of Neighbor Edges:", graph.number_of_edges())

plt.figure(figsize=(6, 5))
sns.heatmap(
    image,
    annot=True,
    fmt=".0f",
    cmap="gray",
    cbar=True
)
plt.title("Original Grayscale Image")
plt.xlabel("Column")
plt.ylabel("Row")
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 5))
sns.heatmap(
    updated_image,
    annot=True,
    fmt=".1f",
    cmap="gray",
    cbar=True
)
plt.title("Updated Image After MRF Smoothing")
plt.xlabel("Column")
plt.ylabel("Row")
plt.tight_layout()
plt.show()

positions = {
    (i, j): (j, -i)
    for i in range(rows)
    for j in range(cols)
}

plt.figure(figsize=(8, 7))
nx.draw(
    graph,
    positions,
    with_labels=True,
    node_size=900,
    font_size=8
)
plt.title("Markov Random Field Pixel Graph")
plt.axis("off")
plt.show()