"""
K-Nearest Neighbors Classifier

Description:
KNN Classifier assigns a class label to a new data point based on the majority class of its k nearest neighbors
in the feature space.

Core Intuition:
The algorithm assumes that similar data points (close in feature space) belong to the same class.
It finds the k most similar training examples and assigns the most common class among them.

Best For:
- Non-linear decision boundaries
- Multi-class classification
- When the decision boundary is irregular

Pros:
- Simple and intuitive
- No training phase (lazy learning)
- Can capture complex patterns
- Robust to noisy training data

Cons:
- Computationally expensive for large datasets
- Sensitive to choice of k and distance metric
- Doesn't handle high-dimensional data well
- Requires feature scaling

Key Hyperparameters:
- n_neighbors (k): Number of neighbors to consider
- metric: Distance metric ('euclidean', 'manhattan', etc.)
"""

import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

class KNeighborsClassifier:
    def __init__(self, n_neighbors=5, metric='euclidean'):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        """
        Store the training data (no actual training for KNN).

        Parameters:
        X: array-like, shape (n_samples, n_features)
        y: array-like, shape (n_samples,)
        """
        self.X_train = np.asarray(X)
        self.y_train = np.asarray(y)

    def _distance(self, x1, x2):
        """Calculate distance between two points."""
        if self.metric == 'euclidean':
            return np.sqrt(np.sum((x1 - x2) ** 2))
        elif self.metric == 'manhattan':
            return np.sum(np.abs(x1 - x2))
        else:
            raise ValueError(f"Unsupported metric: {self.metric}")

    def predict(self, X):
        """
        Predict class labels for given features.

        Parameters:
        X: array-like, shape (n_samples, n_features)

        Returns:
        y_pred: array-like, shape (n_samples,)
        """
        if self.X_train is None:
            raise ValueError("Model not fitted. Call fit() first.")

        X = np.asarray(X)
        y_pred = []

        for x in X:
            # Calculate distances to all training points
            distances = [self._distance(x, x_train) for x_train in self.X_train]

            # Find k nearest neighbors
            k_indices = np.argsort(distances)[:self.n_neighbors]
            k_nearest_labels = self.y_train[k_indices]

            # Majority vote
            most_common = Counter(k_nearest_labels).most_common(1)[0][0]
            y_pred.append(most_common)

        return np.array(y_pred)

    def predict_proba(self, X):
        """
        Predict class probabilities for given features.

        Parameters:
        X: array-like, shape (n_samples, n_features)

        Returns:
        probabilities: array-like, shape (n_samples, n_classes)
        """
        if self.X_train is None:
            raise ValueError("Model not fitted. Call fit() first.")

        X = np.asarray(X)
        unique_classes = np.unique(self.y_train)
        n_classes = len(unique_classes)
        probabilities = []

        for x in X:
            distances = [self._distance(x, x_train) for x_train in self.X_train]
            k_indices = np.argsort(distances)[:self.n_neighbors]
            k_nearest_labels = self.y_train[k_indices]

            # Count occurrences of each class
            class_counts = Counter(k_nearest_labels)
            proba = [class_counts.get(cls, 0) / self.n_neighbors for cls in unique_classes]
            probabilities.append(proba)

        return np.array(probabilities)

    def score(self, X, y):
        """
        Calculate accuracy score.

        Parameters:
        X: array-like, shape (n_samples, n_features)
        y: array-like, shape (n_samples,)

        Returns:
        accuracy: float
        """
        y_pred = self.predict(X)
        return np.mean(y_pred == y)

# Example usage
if __name__ == "__main__":
    # Generate synthetic classification data
    np.random.seed(42)
    X = np.random.randn(200, 2)
    y = (X[:, 0]**2 + X[:, 1]**2 < 1).astype(int)  # Circular decision boundary

    # Test different k values
    k_values = [1, 3, 5, 10]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.ravel()

    for i, k in enumerate(k_values):
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X, y)
        accuracy = model.score(X, y)

        # Plot decision boundary
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                             np.arange(y_min, y_max, 0.1))
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        axes[i].contourf(xx, yy, Z, alpha=0.4)
        axes[i].scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
        axes[i].set_title(f'KNN (k={k}), Accuracy: {accuracy:.3f}')
        axes[i].set_xlabel('Feature 1')
        axes[i].set_ylabel('Feature 2')

    plt.tight_layout()
    plt.show()
