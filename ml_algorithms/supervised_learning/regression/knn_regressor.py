"""
K-Nearest Neighbors Regressor

Description:
KNN Regressor predicts the target value for a new data point by averaging the target values
of its k nearest neighbors in the feature space.

Core Intuition:
The algorithm assumes that similar data points (close in feature space) have similar target values.
It finds the k most similar training examples and averages their targets.

Best For:
- Non-linear relationships
- When the decision boundary is irregular
- Small to medium datasets

Pros:
- Simple and intuitive
- No training phase (lazy learning)
- Can capture complex patterns
- Robust to outliers in predictions

Cons:
- Computationally expensive for large datasets
- Sensitive to choice of k and distance metric
- Doesn't handle high-dimensional data well (curse of dimensionality)
- Requires feature scaling

Key Hyperparameters:
- n_neighbors (k): Number of neighbors to consider (default: 5)
- metric: Distance metric ('euclidean', 'manhattan', etc.)
"""

import numpy as np
import matplotlib.pyplot as plt

class KNNRegressor:
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
        Predict target values for given features.

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
            k_nearest_targets = self.y_train[k_indices]

            # Average the targets
            prediction = np.mean(k_nearest_targets)
            y_pred.append(prediction)

        return np.array(y_pred)

    def score(self, X, y):
        """
        Calculate R-squared score.

        Parameters:
        X: array-like, shape (n_samples, n_features)
        y: array-like, shape (n_samples,)

        Returns:
        r2: float
        """
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

# Example usage
if __name__ == "__main__":
    # Generate synthetic data
    np.random.seed(42)
    X = np.random.rand(100, 1) * 10
    y = np.sin(X).ravel() + np.random.randn(100) * 0.1

    # Test different k values
    k_values = [1, 3, 5, 10]
    colors = ['red', 'blue', 'green', 'orange']

    plt.scatter(X, y, color='black', alpha=0.5, label='Data points')

    X_test = np.linspace(0, 10, 100).reshape(-1, 1)

    for k, color in zip(k_values, colors):
        model = KNNRegressor(n_neighbors=k)
        model.fit(X, y)
        y_pred = model.predict(X_test)
        plt.plot(X_test, y_pred, color=color, label=f'k={k}')
        print(f"k={k}, R-squared: {model.score(X, y):.4f}")

    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('KNN Regressor with Different k Values')
    plt.legend()
    plt.show()
