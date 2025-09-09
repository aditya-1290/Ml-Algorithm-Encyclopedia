"""
Random Forest Classifier

Description:
Random Forest is an ensemble learning method that builds multiple decision trees and merges their predictions
to improve accuracy and control overfitting.

Core Intuition:
By averaging the predictions of many decorrelated trees, the model reduces variance and improves
generalization compared to a single decision tree.

Best For:
- Non-linear decision boundaries
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
from supervised_learning.classification.decision_tree_classifier import DecisionTreeClassifier

class RandomForestClassifier:
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

            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf
            )
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        # Aggregate predictions from all trees (majority vote)
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        # For each sample, find the most common prediction
        y_pred = []
        for i in range(X.shape[0]):
            preds = tree_preds[:, i]
            unique, counts = np.unique(preds, return_counts=True)
            y_pred.append(unique[np.argmax(counts)])
        return np.array(y_pred)

    def predict_proba(self, X):
        # Average probabilities from all trees
        tree_probas = np.array([tree.predict_proba(X) for tree in self.trees])
        return np.mean(tree_probas, axis=0)

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)

# Example usage
if __name__ == "__main__":
    np.random.seed(42)
    X = np.random.rand(200, 2)
    y = (X[:, 0] + X[:, 1] > 1).astype(int)

    model = RandomForestClassifier(n_estimators=10, max_depth=5)
    model.fit(X, y)
    accuracy = model.score(X, y)

    print(f"Random Forest Accuracy: {accuracy:.4f}")
