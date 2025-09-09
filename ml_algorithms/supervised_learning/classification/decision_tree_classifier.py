"""
Decision Tree Classifier

Description:
A decision tree classifier predicts categorical class labels by learning decision rules inferred from the features.
It recursively splits the data into subsets based on feature values to maximize information gain.

Core Intuition:
The tree partitions the feature space into regions with similar class distributions. Predictions are made by
assigning the majority class in the leaf nodes.

Best For:
- Non-linear decision boundaries
- Handling mixed data types
- Interpretability

Pros:
- Captures non-linear patterns
- Handles categorical and numerical data
- Easy to visualize and interpret
- No need for feature scaling

Cons:
- Prone to overfitting
- Unstable to small data changes
- Can create biased trees if some classes dominate

Key Hyperparameters:
- max_depth: Maximum depth of the tree
- min_samples_split: Minimum samples to split a node
- min_samples_leaf: Minimum samples in a leaf node
- criterion: Splitting criterion ('gini' or 'entropy')
"""

import numpy as np
from collections import Counter

class DecisionTreeClassifier:
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1, criterion='gini'):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
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
        best_score = float('inf') if self.criterion == 'gini' else float('-inf')

        for feature_index in range(n_features):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                left_indices = X[:, feature_index] <= threshold
                right_indices = X[:, feature_index] > threshold

                if len(y[left_indices]) < self.min_samples_leaf or len(y[right_indices]) < self.min_samples_leaf:
                    continue

                score = self._calculate_score(y[left_indices], y[right_indices])
                if (self.criterion == 'gini' and score < best_score) or (self.criterion == 'entropy' and score > best_score):
                    best_score = score
                    best_feature = feature_index
                    best_threshold = threshold

        return best_feature, best_threshold

    def _calculate_score(self, left_y, right_y):
        if self.criterion == 'gini':
            return self._gini_impurity(left_y, right_y)
        elif self.criterion == 'entropy':
            return self._information_gain(left_y, right_y)
        else:
            raise ValueError(f"Unsupported criterion: {self.criterion}")

    def _gini_impurity(self, left_y, right_y):
        def gini(y):
            if len(y) == 0:
                return 0
            proportions = np.array(list(Counter(y).values())) / len(y)
            return 1 - np.sum(proportions ** 2)

        n_left, n_right = len(left_y), len(right_y)
        n_total = n_left + n_right
        return (n_left / n_total) * gini(left_y) + (n_right / n_total) * gini(right_y)

    def _information_gain(self, left_y, right_y):
        def entropy(y):
            if len(y) == 0:
                return 0
            proportions = np.array(list(Counter(y).values())) / len(y)
            return -np.sum(proportions * np.log2(proportions + 1e-10))

        n_left, n_right = len(left_y), len(right_y)
        n_total = n_left + n_right
        parent_entropy = entropy(np.concatenate([left_y, right_y]))
        child_entropy = (n_left / n_total) * entropy(left_y) + (n_right / n_total) * entropy(right_y)
        return parent_entropy - child_entropy

    def _calculate_leaf_value(self, y):
        return Counter(y).most_common(1)[0][0]

    def predict(self, X):
        return np.array([self._predict_sample(x, self.tree) for x in X])

    def _predict_sample(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature_index] <= node.threshold:
            return self._predict_sample(x, node.left)
        else:
            return self._predict_sample(x, node.right)

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)

# Example usage
if __name__ == "__main__":
    np.random.seed(42)
    X = np.random.rand(200, 2)
    y = (X[:, 0] + X[:, 1] > 1).astype(int)

    model = DecisionTreeClassifier(max_depth=5, criterion='gini')
    model.fit(X, y)
    accuracy = model.score(X, y)

    print(f"Decision Tree Accuracy: {accuracy:.4f}")

    # Simple visualization (text-based tree representation would be complex)
    print("Decision tree fitted successfully.")
