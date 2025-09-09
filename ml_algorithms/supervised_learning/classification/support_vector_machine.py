"""
Support Vector Machine (SVM)

Description:
SVM finds the optimal hyperplane that best separates classes with the maximum margin.
It can handle non-linear classification using the kernel trick.

Core Intuition:
The algorithm maximizes the margin between classes while minimizing classification errors.
Support vectors are the data points closest to the decision boundary.

Best For:
- Binary classification
- High-dimensional data
- When clear margin of separation exists

Pros:
- Effective in high-dimensional spaces
- Robust to overfitting
- Can use kernel trick for non-linear boundaries

Cons:
- Sensitive to choice of C and kernel
- Computationally intensive for large datasets
- Requires feature scaling
- Binary classification primarily

Key Hyperparameters:
- C: Regularization parameter (penalty for misclassification)
"""

import numpy as np
import matplotlib.pyplot as plt

class SVM:
    def __init__(self, C=1.0, learning_rate=0.001, max_iter=1000, tol=1e-4):
        self.C = C
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.w = None
        self.b = 0

    def fit(self, X, y):
        """
        Fit the SVM model using gradient descent on hinge loss.

        Parameters:
        X: array-like, shape (n_samples, n_features)
        y: array-like, shape (n_samples,) - labels should be -1 or 1
        """
        X = np.asarray(X)
        y = np.asarray(y)
        # Convert labels to -1, 1 if needed
        y = np.where(y == 0, -1, 1)
        n_samples, n_features = X.shape

        # Initialize weights
        self.w = np.zeros(n_features)

        for _ in range(self.max_iter):
            w_old = self.w.copy()
            b_old = self.b

            # Compute predictions
            decision = y * (X.dot(self.w) + self.b)

            # Find misclassified points
            misclassified = decision < 1

            # Compute gradients
            dw = self.w - self.C * np.sum((y[misclassified, np.newaxis] * X[misclassified]), axis=0)
            db = -self.C * np.sum(y[misclassified])

            # Update weights
            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db

            # Check convergence
            if np.linalg.norm(self.w - w_old) < self.tol and abs(self.b - b_old) < self.tol:
                break

    def predict(self, X):
        """
        Predict class labels for given features.

        Parameters:
        X: array-like, shape (n_samples, n_features)

        Returns:
        y_pred: array-like, shape (n_samples,) - labels are -1 or 1
        """
        if self.w is None:
            raise ValueError("Model not fitted. Call fit() first.")
        X = np.asarray(X)
        decision = X.dot(self.w) + self.b
        return np.sign(decision)

    def score(self, X, y):
        """
        Calculate accuracy score.

        Parameters:
        X: array-like, shape (n_samples, n_features)
        y: array-like, shape (n_samples,) - original labels (0 or 1)

        Returns:
        accuracy: float
        """
        y_pred = self.predict(X)
        # Convert predictions back to 0, 1
        y_pred = np.where(y_pred == -1, 0, 1)
        return np.mean(y_pred == y)

# Example usage
if __name__ == "__main__":
    # Generate synthetic linearly separable data
    np.random.seed(42)
    X = np.random.randn(200, 2)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    # Fit the model
    model = SVM(C=1.0, learning_rate=0.01, max_iter=1000)
    model.fit(X, y)

    # Make predictions
    y_pred = model.predict(X)
    accuracy = model.score(X, y)

    print(f"SVM weights: {model.w}")
    print(f"SVM bias: {model.b}")
    print(f"Accuracy: {accuracy:.4f}")

    # Plot decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
    # Plot support vectors (approximate)
    margin = 1 / np.linalg.norm(model.w)
    plt.plot([x_min, x_max], [-(model.w[0]*x_min + model.b)/model.w[1], -(model.w[0]*x_max + model.b)/model.w[1]], 'k-')
    plt.plot([x_min, x_max], [-(model.w[0]*x_min + model.b - 1)/model.w[1], -(model.w[0]*x_max + model.b - 1)/model.w[1]], 'k--')
    plt.plot([x_min, x_max], [-(model.w[0]*x_min + model.b + 1)/model.w[1], -(model.w[0]*x_max + model.b + 1)/model.w[1]], 'k--')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('SVM Decision Boundary')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.show()
