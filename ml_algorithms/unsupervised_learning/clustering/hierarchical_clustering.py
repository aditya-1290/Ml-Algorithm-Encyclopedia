"""
Hierarchical Clustering

Description:
Hierarchical clustering builds a tree of clusters by either merging smaller clusters (agglomerative)
or splitting larger clusters (divisive). This implementation uses the agglomerative approach.

Core Intuition:
The algorithm starts with each data point as its own cluster and iteratively merges the closest
clusters until only one cluster remains, creating a hierarchy represented by a dendrogram.

Best For:
- Small to medium datasets
- When hierarchical relationships are important
- When the number of clusters is not known

Pros:
- No need to specify number of clusters
- Provides hierarchical structure
- Can be visualized with dendrograms

Cons:
- Computationally expensive O(n^3)
- Cannot handle large datasets
- Sensitive to noise and outliers
- Once merged, clusters cannot be split

Key Hyperparameters:
- linkage: Method for calculating distance between clusters ('single', 'complete', 'average')
- n_clusters: Number of clusters to form (optional, can be determined from dendrogram)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

class HierarchicalClustering:
    def __init__(self, linkage='single'):
        self.linkage = linkage
        self.labels = None
        self.n_clusters = None

    def fit(self, X):
        """
        Fit the hierarchical clustering model.

        Parameters:
        X: array-like, shape (n_samples, n_features)
        """
        X = np.asarray(X)
        n_samples = X.shape[0]

        # Initialize each point as its own cluster
        clusters = [[i] for i in range(n_samples)]
        distances = self._compute_distance_matrix(X)

        # Keep track of merge history for dendrogram
        self.merge_history = []

        while len(clusters) > 1:
            # Find closest pair of clusters
            min_dist = float('inf')
            merge_i, merge_j = -1, -1

            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    dist = self._cluster_distance(clusters[i], clusters[j], distances, X)
                    if dist < min_dist:
                        min_dist = dist
                        merge_i, merge_j = i, j

            # Merge the closest clusters
            new_cluster = clusters[merge_i] + clusters[merge_j]
            self.merge_history.append((clusters[merge_i], clusters[merge_j], min_dist, len(new_cluster)))

            # Remove old clusters and add new one
            clusters.pop(max(merge_i, merge_j))
            clusters.pop(min(merge_i, merge_j))
            clusters.append(new_cluster)

    def _compute_distance_matrix(self, X):
        """Compute pairwise distances between all points."""
        n_samples = X.shape[0]
        distances = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                distances[i, j] = np.sqrt(np.sum((X[i] - X[j])**2))
                distances[j, i] = distances[i, j]
        return distances

    def _cluster_distance(self, cluster1, cluster2, distances, X):
        """Calculate distance between two clusters based on linkage method."""
        if self.linkage == 'single':
            # Minimum distance between any points in the two clusters
            min_dist = float('inf')
            for i in cluster1:
                for j in cluster2:
                    min_dist = min(min_dist, distances[i, j])
            return min_dist
        elif self.linkage == 'complete':
            # Maximum distance between any points in the two clusters
            max_dist = 0
            for i in cluster1:
                for j in cluster2:
                    max_dist = max(max_dist, distances[i, j])
            return max_dist
        elif self.linkage == 'average':
            # Average distance between all pairs of points
            total_dist = 0
            count = 0
            for i in cluster1:
                for j in cluster2:
                    total_dist += distances[i, j]
                    count += 1
            return total_dist / count
        else:
            raise ValueError(f"Unsupported linkage method: {self.linkage}")

    def fit_predict(self, X, n_clusters=None):
        """
        Fit the model and return cluster labels.

        Parameters:
        X: array-like, shape (n_samples, n_features)
        n_clusters: int, number of clusters to form

        Returns:
        labels: array-like, shape (n_samples,)
        """
        self.fit(X)

        if n_clusters is None:
            # Return labels for 2 clusters by default
            n_clusters = 2

        # Cut the dendrogram at n_clusters
        n_samples = X.shape[0]
        labels = np.zeros(n_samples, dtype=int)

        # Start with each point as its own cluster
        clusters = [[i] for i in range(n_samples)]
        cluster_id = 0

        # Reverse the merge history to build clusters
        merge_steps = self.merge_history[::-1]

        for step in range(len(merge_steps) - n_clusters + 1):
            cluster1, cluster2, _, _ = merge_steps[step]
            # Assign cluster id to points in cluster1 and cluster2
            for point in cluster1 + cluster2:
                labels[point] = cluster_id
            cluster_id += 1

        # Handle remaining points
        remaining_points = set(range(n_samples)) - set(labels.nonzero()[0])
        for point in remaining_points:
            labels[point] = cluster_id
            cluster_id += 1

        self.labels = labels
        return labels

    def plot_dendrogram(self, X):
        """
        Plot the dendrogram.

        Parameters:
        X: array-like, shape (n_samples, n_features)
        """
        # Use scipy for dendrogram plotting
        Z = linkage(X, method=self.linkage)
        plt.figure(figsize=(10, 7))
        dendrogram(Z)
        plt.title('Hierarchical Clustering Dendrogram')
        plt.xlabel('Sample index')
        plt.ylabel('Distance')
        plt.show()

# Example usage
if __name__ == "__main__":
    # Generate synthetic clustered data
    np.random.seed(42)
    n_samples = 20
    centers = [[1, 1], [-1, -1], [1, -1]]
    X = []
    for center in centers:
        cluster = np.random.randn(n_samples//3, 2) * 0.5 + center
        X.append(cluster)
    X = np.vstack(X)

    # Fit hierarchical clustering
    hc = HierarchicalClustering(linkage='single')
    labels = hc.fit_predict(X, n_clusters=3)

    # Plot results
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.6)
    plt.title('Hierarchical Clustering')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

    # Plot dendrogram
    hc.plot_dendrogram(X)
