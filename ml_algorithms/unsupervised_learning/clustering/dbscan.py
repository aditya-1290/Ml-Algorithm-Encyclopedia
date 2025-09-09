"""
DBSCAN (Density-Based Spatial Clustering of Applications with Noise)

Description:
DBSCAN is a density-based clustering algorithm that groups together points that are closely packed
and marks points in low-density regions as outliers.

Core Intuition:
The algorithm defines clusters as areas of high density separated by areas of low density.
It identifies core points (points with at least min_samples neighbors within eps distance),
border points (points within eps of a core point but not core themselves), and noise points.

Best For:
- Arbitrarily shaped clusters
- Datasets with noise/outliers
- When cluster structure is not known

Pros:
- Can find arbitrarily shaped clusters
- Robust to outliers
- No need to specify number of clusters
- Works well with varying densities

Cons:
- Sensitive to parameters eps and min_samples
- Struggles with varying densities
- Cannot cluster datasets with large differences in densities
- Computationally expensive for high-dimensional data

Key Hyperparameters:
- eps: Maximum distance between two samples for one to be considered neighbor of the other
- min_samples: Number of samples in a neighborhood for a point to be considered a core point
"""

import numpy as np
import matplotlib.pyplot as plt

class DBSCAN:
    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
        self.labels = None

    def fit(self, X):
        """
        Fit the DBSCAN model to the data.

        Parameters:
        X: array-like, shape (n_samples, n_features)
        """
        X = np.asarray(X)
        n_samples = X.shape[0]

        # Initialize labels: -1 for unvisited, 0 for noise, positive for clusters
        self.labels = np.full(n_samples, -1)
        cluster_id = 0

        for i in range(n_samples):
            if self.labels[i] != -1:
                continue  # Already visited

            neighbors = self._region_query(X, i)

            if len(neighbors) < self.min_samples:
                self.labels[i] = 0  # Mark as noise (for now)
            else:
                cluster_id += 1
                self._expand_cluster(X, i, neighbors, cluster_id)

    def _region_query(self, X, point_idx):
        """Find all points within eps distance of the given point."""
        distances = np.sqrt(np.sum((X - X[point_idx])**2, axis=1))
        neighbors = np.where(distances <= self.eps)[0]
        return neighbors

    def _expand_cluster(self, X, point_idx, neighbors, cluster_id):
        """Expand the cluster from the core point."""
        self.labels[point_idx] = cluster_id

        i = 0
        while i < len(neighbors):
            neighbor_idx = neighbors[i]

            if self.labels[neighbor_idx] == 0:
                # Previously marked as noise, but now part of cluster
                self.labels[neighbor_idx] = cluster_id

            if self.labels[neighbor_idx] == -1:
                # Not visited yet
                self.labels[neighbor_idx] = cluster_id
                new_neighbors = self._region_query(X, neighbor_idx)

                if len(new_neighbors) >= self.min_samples:
                    neighbors = np.concatenate([neighbors, new_neighbors])

            i += 1

    def fit_predict(self, X):
        """
        Fit the model and return cluster labels.

        Parameters:
        X: array-like, shape (n_samples, n_features)

        Returns:
        labels: array-like, shape (n_samples,)
        """
        self.fit(X)
        return self.labels

# Example usage
if __name__ == "__main__":
    # Generate synthetic data with noise
    np.random.seed(42)
    n_samples = 300

    # Create three clusters
    centers = [[1, 1], [-1, -1], [1, -1]]
    X = []
    for center in centers:
        cluster = np.random.randn(n_samples//3, 2) * 0.3 + center
        X.append(cluster)
    X = np.vstack(X)

    # Add some noise
    noise = np.random.uniform(-2, 2, (50, 2))
    X = np.vstack([X, noise])

    # Fit DBSCAN
    dbscan = DBSCAN(eps=0.3, min_samples=10)
    labels = dbscan.fit_predict(X)

    # Plot results
    unique_labels = set(labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

    for k, col in zip(unique_labels, colors):
        if k == 0:
            # Black for noise
            col = 'black'

        class_member_mask = (labels == k)
        xy = X[class_member_mask]
        plt.scatter(xy[:, 0], xy[:, 1], c=[col], alpha=0.6, s=50)

    plt.title('DBSCAN Clustering')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

    n_clusters = len(set(labels)) - (1 if 0 in labels else 0)
    n_noise = list(labels).count(0)
    print(f"Estimated number of clusters: {n_clusters}")
    print(f"Estimated number of noise points: {n_noise}")
