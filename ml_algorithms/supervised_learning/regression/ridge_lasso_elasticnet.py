"""
Ridge, Lasso, and ElasticNet Regression

Description:
These are regularized versions of linear regression that add penalty terms to the loss function
to prevent overfitting and improve generalization.

- Ridge Regression (L2): Adds a penalty proportional to the square of the magnitude of coefficients
- Lasso Regression (L1): Adds a penalty proportional to the absolute value of coefficients
- ElasticNet: Combines L1 and L2 penalties

Core Intuition:
Regularization helps prevent overfitting by shrinking coefficients towards zero. L2 (Ridge) shrinks
all coefficients equally, while L1 (Lasso) can drive some coefficients to exactly zero for feature
selection. ElasticNet combines both benefits.

Best For:
- High-dimensional datasets
- When linear regression overfits
- Feature selection (Lasso/ElasticNet)

Pros:
- Reduce overfitting
- Handle multicollinearity
- Lasso performs automatic feature selection

Cons:
- Ridge: Doesn't perform feature selection
- Lasso: Can be unstable with correlated features
- ElasticNet: More hyperparameters to tune
- All: May underfit if regularization is too strong

Key Hyperparameters:
- alpha (lambda): Regularization strength (default: 1.0)
- l1_ratio (ElasticNet only): Balance between L1 and L2 (0=Ridge, 1=Lasso, default: 0.5)
"""

import numpy as np
import matplotlib.pyplot as plt

class RidgeRegression:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        n_features = X_b.shape[1]

        # Ridge regression: w = (X^T X + alpha I)^-1 X^T y
        I = np.eye(n_features)
        I[0, 0] = 0  # Don't regularize bias term

        try:
            self.weights = np.linalg.inv(X_b.T.dot(X_b) + self.alpha * I).dot(X_b.T).dot(y)
        except np.linalg.LinAlgError:
            self.weights = np.linalg.pinv(X_b.T.dot(X_b) + self.alpha * I).dot(X_b.T).dot(y)

        self.bias = self.weights[0]
        self.weights = self.weights[1:]

    def predict(self, X):
        if self.weights is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return X.dot(self.weights) + self.bias

class LassoRegression:
    def __init__(self, alpha=1.0, max_iter=1000, tol=1e-4):
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.weights = None
        self.bias = None

    def _soft_threshold(self, x, threshold):
        """Soft thresholding operator for L1 regularization"""
        return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = np.mean(y)  # Initialize bias

        for _ in range(self.max_iter):
            weights_old = self.weights.copy()

            for j in range(n_features):
                # Compute residual without feature j
                X_j = X[:, j]
                y_pred = X.dot(self.weights) + self.bias
                residual = y - y_pred + self.weights[j] * X_j

                # Update weight j using soft thresholding
                rho = X_j.dot(residual)
                if rho < -self.alpha * n_samples / 2:
                    self.weights[j] = (rho + self.alpha * n_samples / 2) / (X_j.dot(X_j))
                elif rho > self.alpha * n_samples / 2:
                    self.weights[j] = (rho - self.alpha * n_samples / 2) / (X_j.dot(X_j))
                else:
                    self.weights[j] = 0

            # Update bias
            self.bias = np.mean(y - X.dot(self.weights))

            # Check convergence
            if np.linalg.norm(self.weights - weights_old) < self.tol:
                break

    def predict(self, X):
        if self.weights is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return X.dot(self.weights) + self.bias

class ElasticNetRegression:
    def __init__(self, alpha=1.0, l1_ratio=0.5, max_iter=1000, tol=1e-4):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.tol = tol
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = np.mean(y)

        for _ in range(self.max_iter):
            weights_old = self.weights.copy()

            for j in range(n_features):
                X_j = X[:, j]
                y_pred = X.dot(self.weights) + self.bias
                residual = y - y_pred + self.weights[j] * X_j

                rho = X_j.dot(residual)
                threshold = self.alpha * self.l1_ratio * n_samples / 2

                if rho < -threshold:
                    self.weights[j] = (rho + threshold) / (X_j.dot(X_j) + self.alpha * (1 - self.l1_ratio) * n_samples)
                elif rho > threshold:
                    self.weights[j] = (rho - threshold) / (X_j.dot(X_j) + self.alpha * (1 - self.l1_ratio) * n_samples)
                else:
                    self.weights[j] = 0

            self.bias = np.mean(y - X.dot(self.weights))

            if np.linalg.norm(self.weights - weights_old) < self.tol:
                break

    def predict(self, X):
        if self.weights is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return X.dot(self.weights) + self.bias

# Example usage
if __name__ == "__main__":
    # Generate synthetic data with some irrelevant features
    np.random.seed(42)
    X = np.random.randn(100, 5)
    true_weights = np.array([3, 2, 0, 0, 1])  # Last two features are irrelevant
    y = X.dot(true_weights) + 0.1 * np.random.randn(100)

    # Compare models
    models = [
        ("Ridge", RidgeRegression(alpha=0.1)),
        ("Lasso", LassoRegression(alpha=0.1)),
        ("ElasticNet", ElasticNetRegression(alpha=0.1, l1_ratio=0.5))
    ]

    print("True weights:", true_weights)
    print()

    for name, model in models:
        model.fit(X, y)
        print(f"{name} weights: {model.weights}")
        print(f"{name} bias: {model.bias}")
        print()
