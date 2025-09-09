"""
Gradient Boosting Classifier

Description:
Gradient Boosting builds an ensemble of weak learners (usually decision trees) sequentially,
where each new tree tries to correct the errors of the previous ensemble by fitting to the residuals.

Core Intuition:
By iteratively minimizing the loss function using gradient descent in function space, the model
improves predictions step-by-step.

Best For:
- High predictive accuracy
- Handling complex non-linear relationships
- When interpretability is less critical

Pros:
- High accuracy and flexibility
- Can handle various loss functions
- Robust to overfitting with proper tuning

Cons:
- Computationally intensive
- Sensitive to hyperparameters
- Less interpretable than single trees

Key Hyperparameters:
- n_estimators: Number of boosting stages
- learning_rate: Shrinks contribution of each tree
- max_depth: Maximum depth of individual trees
- min_samples_split: Minimum samples to split a node
- min_samples_leaf: Minimum samples in a leaf node
"""

import numpy as np
from supervised_learning.classification.decision_tree_classifier import DecisionTreeClassifier

class GradientBoostingClassifier:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3,
                 min_samples_split=2, min_samples_leaf=1):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.trees = []
        self.initial_prediction = None

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def fit(self, X, y):
        n_samples = X.shape[0]
        # Initialize model with log odds
        pos_ratio = np.mean(y)
        self.initial_prediction = np.log(pos_ratio / (1 - pos_ratio))
        residuals = y - self._sigmoid(self.initial_prediction)
        self.trees = []

        for _ in range(self.n_estimators):
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf
            )
            tree.fit(X, residuals)
            prediction = tree.predict(X)
            residuals -= self.learning_rate * prediction
            self.trees.append(tree)

    def predict_proba(self, X):
        y_pred = np.full(X.shape[0], self.initial_prediction)
        for tree in self.trees:
            y_pred += self.learning_rate * tree.predict(X)
        proba = self._sigmoid(y_pred)
        return np.column_stack([1 - proba, proba])

    def predict(self, X, threshold=0.5):
        proba = self.predict_proba(X)[:, 1]
        return (proba >= threshold).astype(int)

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)

# Example usage
if __name__ == "__main__":
    np.random.seed(42)
    X = np.random.rand(200, 2)
    y = (X[:, 0] + X[:, 1] > 1).astype(int)

    model = GradientBoostingClassifier(n_estimators=10, learning_rate=0.1, max_depth=3)
    model.fit(X, y)
    accuracy = model.score(X, y)

    print(f"Gradient Boosting Classifier Accuracy: {accuracy:.4f}")
