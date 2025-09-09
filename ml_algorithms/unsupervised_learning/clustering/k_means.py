"""
K-Means Clustering

Core Idea: Partitions data into 'k' distinct clusters based on distance to centroid.
Each data point belongs to the cluster with the nearest centroid.

Best For: Spherical clusters, when k is known, large datasets.

Pros:
- Simple and fast
- Scales well to large datasets
- Easy to interpret

Cons:
- Sensitive to initial centroids
- Assumes spherical clusters
- Requires specifying k
- Sensitive to outliers

Key Hyperparameters:
- k: Number of clusters
- max_iter: Maximum iterations
- tol: Tolerance for convergence
- init: Initialization method ('random' or 'k-means++')
"""

import numpy as np
import matplotlib.pyplot as plt

class KMeans:
    def __init__(self, k=3, max_iter=100, tol=1e-4, init='random'):
        self.k = k
        self.max_iter = max_iter
        self.tol = tol
        self.init = init
        self.centroids = None
        self.labels_ = None

    def _initialize_centroids(self, X):
        if self.init == 'random':
            indices = np.random.choice(X.shape[0], self.k, replace=False)
            self.centroids = X[indices]
        elif self.init == 'k-means++':
            self.centroids = [X[np.random.randint(X.shape[0])]]
            for _ in range(1, self.k):
                distances = np.min([np.sum((X - c)**2, axis=1) for c in self.centroids], axis=0)
                probs = distances / np.sum(distances)
                next_centroid = X[np.random.choice(X.shape[0], p=probs)]
                self.centroids.append(next_centroid)
            self.centroids = np.array(self.centroids)

    def _assign_clusters(self, X):
        distances = np.array([np.sum((X - c)**2, axis=1) for c in self.centroids])
        return np.argmin(distances, axis=0)

    def _update_centroids(self, X, labels):
        new_centroids = []
        for i in range(self.k):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                new_centroids.append(np.mean(cluster_points, axis=0))
            else:
                new_centroids.append(self.centroids[i])  # Keep old if empty
        return np.array(new_centroids)

    def fit(self, X):
        self._initialize_centroids(X)
        for _ in range(self.max_iter):
            labels = self._assign_clusters(X)
            new_centroids = self._update_centroids(X, labels)
            if np.all(np.abs(new_centroids - self.centroids) < self.tol):
                break
            self.centroids = new_centroids
        self.labels_ = labels
        return self

    def predict(self, X):
        return self._assign_clusters(X)

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_

# Example usage
if __name__ == "__main__":
    # Generate synthetic data
    np.random.seed(42)
    X1 = np.random.randn(100, 2) + [2, 2]
    X2 = np.random.randn(100, 2) + [-2, -2]
    X3 = np.random.randn(100, 2) + [2, -2]
    X = np.vstack([X1, X2, X3])

    # Fit K-Means
    kmeans = KMeans(k=3, init='k-means++')
    labels = kmeans.fit_predict(X)

    # Plot results
    plt.figure(figsize=(8, 6))
    colors = ['red', 'blue', 'green']
    for i in range(3):
        cluster = X[labels == i]
        plt.scatter(cluster[:, 0], cluster[:, 1], c=colors[i], label=f'Cluster {i+1}')
    plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], c='black', marker='x', s=100, label='Centroids')
    plt.title('K-Means Clustering')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()
