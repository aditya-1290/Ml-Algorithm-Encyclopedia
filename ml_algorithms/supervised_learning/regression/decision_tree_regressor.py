"""
Decision Tree Regressor

Description:
A decision tree regressor predicts continuous values by learning decision rules inferred from the features.
It recursively splits the data into subsets based on feature values to minimize variance in the target variable.

Core Intuition:
The tree partitions the feature space into regions with similar target values. Predictions are made by averaging
the target values in the leaf nodes.

Best For:
- Non-linear relationships
- Handling mixed data types
- Interpretability

Pros:
- Captures non-linear patterns
- Handles categorical and numerical data
- Easy to visualize and interpret

Cons:
- Prone to overfitting
- Unstable to small data changes
- Can create biased trees if some classes dominate

Key Hyperparameters:
- max_depth: Maximum depth of the tree
- min_samples_split: Minimum samples to split a node
- min_samples_leaf: Minimum samples in a leaf node
"""

import numpy as np

class DecisionTreeRegressor:
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.tree = None

    class Node:
        def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
            self.feature_index = feature_index
            self.threshold = threshold
            self.left = left
            self.right = right
            self.value = value

    def fit(self, X, y):
        self.tree = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        n_samples, n_features = X.shape
        if n_samples < self.min_samples_split or (self.max_depth is not None and depth >= self.max_depth):
            leaf_value = self._calculate_leaf_value(y)
            return self.Node(value=leaf_value)

        best_feature, best_threshold = self._best_split(X, y, n_features)
        if best_feature is None:
            leaf_value = self._calculate_leaf_value(y)
            return self.Node(value=leaf_value)

        left_indices = X[:, best_feature] <= best_threshold
        right_indices = X[:, best_feature] > best_threshold

        left = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right = self._build_tree(X[right_indices], y[right_indices], depth + 1)
        return self.Node(best_feature, best_threshold, left, right)

    def _best_split(self, X, y, n_features):
        best_feature, best_threshold = None, None
        best_mse = float('inf')

        for feature_index in range(n_features):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                left_indices = X[:, feature_index] <= threshold
                right_indices = X[:, feature_index] > threshold

                if len(y[left_indices]) < self.min_samples_leaf or len(y[right_indices]) < self.min_samples_leaf:
                    continue

                mse = self._calculate_mse(y[left_indices], y[right_indices])
                if mse < best_mse:
                    best_mse = mse
                    best_feature = feature_index
                    best_threshold = threshold

        return best_feature, best_threshold

    def _calculate_mse(self, left_y, right_y):
        left_var = np.var(left_y) if len(left_y) > 0 else 0
        right_var = np.var(right_y) if len(right_y) > 0 else 0
        total_len = len(left_y) + len(right_y)
        weighted_mse = (len(left_y) * left_var + len(right_y) * right_var) / total_len
        return weighted_mse

    def _calculate_leaf_value(self, y):
        return np.mean(y)

    def predict(self, X):
        return np.array([self._predict_sample(x, self.tree) for x in X])

    def _predict_sample(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature_index] <= node.threshold:
            return self._predict_sample(x, node.left)
        else:
            return self._predict_sample(x, node.right)

# Example usage
if __name__ == "__main__":
    np.random.seed(42)
    X = np.random.rand(100, 1) * 10
    y = np.sin(X).ravel() + np.random.randn(100) * 0.1

    model = DecisionTreeRegressor(max_depth=5)
    model.fit(X, y)
    y_pred = model.predict(X)

    import matplotlib.pyplot as plt
    plt.scatter(X, y, label="Data")
    plt.scatter(X, y_pred, label="Predictions")
    plt.legend()
    plt.title("Decision Tree Regressor")
    plt.show()
