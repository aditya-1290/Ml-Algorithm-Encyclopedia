"""
Support Vector Regression (SVR)

Description:
SVR finds a function that has at most epsilon deviation from the actual target values for all training data,
and is as flat as possible. It's an extension of SVM for regression problems.

Core Intuition:
The algorithm tries to fit the best line (or hyperplane) within an epsilon-wide tube around the data points,
while minimizing the complexity of the model. Points outside the tube contribute to the loss.

Best For:
- Regression with margin of tolerance
- When outliers should be ignored within epsilon
- High-dimensional data

Pros:
- Effective in high-dimensional spaces
- Robust to outliers within epsilon
- Can use kernel trick for non-linear relationships

Cons:
- Sensitive to choice of C and epsilon
- Computationally intensive for large datasets
- Requires feature scaling

Key Hyperparameters:
- C: Regularization parameter (penalty for points outside epsilon tube)
- epsilon: Width of the epsilon-insensitive zone
"""

import numpy as np
import matplotlib.pyplot as plt

class SVR:
    def __init__(self, C=1.0, epsilon=0.1, learning_rate=0.01, max_iter=1000, tol=1e-4):
        self.C = C
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.w = None
        self.b = 0

    def fit(self, X, y):
        """
        Fit the SVR model using gradient descent.

        Parameters:
        X: array-like, shape (n_samples, n_features)
        y: array-like, shape (n_samples,)
        """
        X = np.asarray(X)
        y = np.asarray(y)
        n_samples, n_features = X.shape

        # Initialize weights
        self.w = np.zeros(n_features)

        for _ in range(self.max_iter):
            w_old = self.w.copy()
            b_old = self.b

            # Compute predictions
            y_pred = X.dot(self.w) + self.b

            # Compute errors
            errors = y - y_pred

            # Compute gradients
            dw = np.zeros(n_features)
            db = 0

            for i in range(n_samples):
                if errors[i] > self.epsilon:
                    dw -= self.C * X[i]
                    db -= self.C
                elif errors[i] < -self.epsilon:
                    dw += self.C * X[i]
                    db += self.C

            # Update weights
            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db

            # Check convergence
            if np.linalg.norm(self.w - w_old) < self.tol and abs(self.b - b_old) < self.tol:
                break

    def predict(self, X):
        """
        Predict target values for given features.

        Parameters:
        X: array-like, shape (n_samples, n_features)

        Returns:
        y_pred: array-like, shape (n_samples,)
        """
        if self.w is None:
            raise ValueError("Model not fitted. Call fit() first.")
        X = np.asarray(X)
        return X.dot(self.w) + self.b

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
    y = 2 * X.ravel() + 1 + np.random.randn(100) * 0.5

    # Fit SVR
    model = SVR(C=1.0, epsilon=0.1, learning_rate=0.01, max_iter=1000)
    model.fit(X, y)
    y_pred = model.predict(X)

    print(f"SVR weights: {model.w}")
    print(f"SVR bias: {model.b}")
    print(f"R-squared: {model.score(X, y):.4f}")

    # Plot results
    plt.scatter(X, y, color='blue', label='Data points')
    plt.plot(X, y_pred, color='red', label='SVR prediction')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Support Vector Regression')
    plt.legend()
    plt.show()
