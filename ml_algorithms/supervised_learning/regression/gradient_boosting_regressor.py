"""
Gradient Boosting Regressor

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
from supervised_learning.regression.decision_tree_regressor import DecisionTreeRegressor

class GradientBoostingRegressor:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3,
                 min_samples_split=2, min_samples_leaf=1):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.trees = []
        self.initial_prediction = None

    def fit(self, X, y):
        n_samples = X.shape[0]
        # Initialize model with mean prediction
        self.initial_prediction = np.mean(y)
        residuals = y - self.initial_prediction
        self.trees = []

        for _ in range(self.n_estimators):
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf
            )
            tree.fit(X, residuals)
            prediction = tree.predict(X)
            residuals -= self.learning_rate * prediction
            self.trees.append(tree)

    def predict(self, X):
        y_pred = np.full(X.shape[0], self.initial_prediction)
        for tree in self.trees:
            y_pred += self.learning_rate * tree.predict(X)
        return y_pred

# Example usage
if __name__ == "__main__":
    np.random.seed(42)
    X = np.random.rand(100, 1) * 10
    y = np.sin(X).ravel() + np.random.randn(100) * 0.1

    model = GradientBoostingRegressor(n_estimators=10, learning_rate=0.1, max_depth=3)
    model.fit(X, y)
    y_pred = model.predict(X)

    import matplotlib.pyplot as plt
    plt.scatter(X, y, label="Data")
    plt.scatter(X, y_pred, label="Predictions")
    plt.legend()
    plt.title("Gradient Boosting Regressor")
    plt.show()
