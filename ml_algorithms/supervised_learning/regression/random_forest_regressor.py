"""
Random Forest Regressor

Description:
Random Forest is an ensemble learning method that builds multiple decision trees and merges their
predictions to improve accuracy and control overfitting.

Core Intuition:
By averaging the predictions of many decorrelated trees, the model reduces variance and improves
generalization compared to a single decision tree.

Best For:
- Non-linear relationships
- Handling high-dimensional data
- Robustness to noise and overfitting

Pros:
- High accuracy
- Handles large datasets well
- Can model complex relationships
- Provides feature importance

Cons:
- Less interpretable than single trees
- Can be computationally expensive
- Requires tuning of many hyperparameters

Key Hyperparameters:
- n_estimators: Number of trees in the forest
- max_depth: Maximum depth of each tree
- min_samples_split: Minimum samples to split a node
- min_samples_leaf: Minimum samples in a leaf node
"""

import numpy as np
from supervised_learning.regression.decision_tree_regressor import DecisionTreeRegressor

class RandomForestRegressor:
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        n_samples = X.shape[0]

        for _ in range(self.n_estimators):
            # Bootstrap sampling
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_sample = X[indices]
            y_sample = y[indices]

            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf
            )
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        # Aggregate predictions from all trees
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        return np.mean(tree_preds, axis=0)

# Example usage
if __name__ == "__main__":
    np.random.seed(42)
    X = np.random.rand(100, 1) * 10
    y = np.sin(X).ravel() + np.random.randn(100) * 0.1

    model = RandomForestRegressor(n_estimators=10, max_depth=5)
    model.fit(X, y)
    y_pred = model.predict(X)

    import matplotlib.pyplot as plt
    plt.scatter(X, y, label="Data")
    plt.scatter(X, y_pred, label="Predictions")
    plt.legend()
    plt.title("Random Forest Regressor")
    plt.show()
