"""
Logistic Regression

Description:
Logistic Regression is a classification algorithm that uses the logistic function to model the probability
of a binary outcome. It predicts the probability that a given input belongs to a particular class.

Core Intuition:
The algorithm models the log-odds of the probability as a linear combination of the features.
The logistic function (sigmoid) transforms this to a probability between 0 and 1.

Best For:
- Binary classification problems
- When probabilistic predictions are needed
- Linear decision boundaries

Pros:
- Simple and interpretable
- Provides probabilistic predictions
- Efficient training
- Works well with small datasets

Cons:
- Assumes linear relationship between features and log-odds
- Sensitive to outliers
- May underfit complex decision boundaries
- Requires feature scaling

Key Hyperparameters:
- None (but regularization can be added)
"""

import numpy as np
import matplotlib.pyplot as plt

class LogisticRegression:
    def __init__(self, learning_rate=0.01, max_iter=1000, tol=1e-4):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.weights = None
        self.bias = 0

    def _sigmoid(self, z):
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        """
        Fit the logistic regression model using gradient descent.

        Parameters:
        X: array-like, shape (n_samples, n_features)
        y: array-like, shape (n_samples,) - binary labels (0 or 1)
        """
        X = np.asarray(X)
        y = np.asarray(y)
        n_samples, n_features = X.shape

        # Initialize weights
        self.weights = np.zeros(n_features)

        for _ in range(self.max_iter):
            # Compute predictions
            z = X.dot(self.weights) + self.bias
            y_pred = self._sigmoid(z)

            # Compute gradients
            dw = (1 / n_samples) * X.T.dot(y_pred - y)
            db = (1 / n_samples) * np.sum(y_pred - y)

            # Update weights
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # Check convergence (simplified)
            if np.linalg.norm(dw) < self.tol and abs(db) < self.tol:
                break

    def predict_proba(self, X):
        """
        Predict class probabilities.

        Parameters:
        X: array-like, shape (n_samples, n_features)

        Returns:
        probabilities: array-like, shape (n_samples, 2)
        """
        if self.weights is None:
            raise ValueError("Model not fitted. Call fit() first.")
        X = np.asarray(X)
        z = X.dot(self.weights) + self.bias
        proba_class_1 = self._sigmoid(z)
        proba_class_0 = 1 - proba_class_1
        return np.column_stack([proba_class_0, proba_class_1])

    def predict(self, X, threshold=0.5):
        """
        Predict class labels.

        Parameters:
        X: array-like, shape (n_samples, n_features)
        threshold: float, decision threshold

        Returns:
        y_pred: array-like, shape (n_samples,)
        """
        proba = self.predict_proba(X)
        return (proba[:, 1] >= threshold).astype(int)

    def score(self, X, y):
        """
        Calculate accuracy score.

        Parameters:
        X: array-like, shape (n_samples, n_features)
        y: array-like, shape (n_samples,)

        Returns:
        accuracy: float
        """
        y_pred = self.predict(X)
        return np.mean(y_pred == y)

# Example usage
if __name__ == "__main__":
    # Generate synthetic binary classification data
    np.random.seed(42)
    X = np.random.randn(200, 2)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    # Fit the model
    model = LogisticRegression(learning_rate=0.1, max_iter=1000)
    model.fit(X, y)

    # Make predictions
    y_pred = model.predict(X)
    accuracy = model.score(X, y)

    print(f"Model weights: {model.weights}")
    print(f"Model bias: {model.bias}")
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
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Logistic Regression Decision Boundary')
    plt.show()
