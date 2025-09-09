"""
Perceptron

Description:
The Perceptron is the simplest type of artificial neural network, consisting of a single layer
of neurons. It is a linear binary classifier that learns weights using the perceptron learning rule.

Core Intuition:
The algorithm iteratively updates the weights when a misclassification occurs, moving the decision
boundary to correctly classify the misclassified point.

Best For:
- Linearly separable binary classification
- Simple and fast learning
- Online learning scenarios

Pros:
- Simple and computationally efficient
- Converges for linearly separable data
- Easy to implement and understand

Cons:
- Only works for linearly separable data
- Sensitive to feature scaling
- No probabilistic outputs
- Can oscillate if data is not separable

Key Hyperparameters:
- learning_rate: Step size for weight updates
- max_iter: Maximum number of iterations
"""

import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, learning_rate=0.01, max_iter=1000, random_state=None):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.random_state = random_state
        self.weights = None
        self.bias = 0

    def fit(self, X, y):
        """
        Fit the perceptron model using the perceptron learning rule.

        Parameters:
        X: array-like, shape (n_samples, n_features)
        y: array-like, shape (n_samples,) - labels should be -1 or 1
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)

        X = np.asarray(X)
        y = np.asarray(y)
        # Convert labels to -1, 1 if needed
        y = np.where(y == 0, -1, 1)
        n_samples, n_features = X.shape

        # Initialize weights
        self.weights = np.zeros(n_features)

        for _ in range(self.max_iter):
            errors = 0
            for xi, target in zip(X, y):
                prediction = self._predict_single(xi)
                update = self.learning_rate * (target - prediction)
                self.weights += update * xi
                self.bias += update
                if update != 0:
                    errors += 1

            # Early stopping if no errors
            if errors == 0:
                break

    def _predict_single(self, x):
        """Predict for a single sample."""
        return np.sign(np.dot(x, self.weights) + self.bias)

    def predict(self, X):
        """
        Predict class labels for given features.

        Parameters:
        X: array-like, shape (n_samples, n_features)

        Returns:
        y_pred: array-like, shape (n_samples,) - labels are -1 or 1
        """
        if self.weights is None:
            raise ValueError("Model not fitted. Call fit() first.")
        X = np.asarray(X)
        return np.sign(X.dot(self.weights) + self.bias)

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
    # Generate linearly separable data
    np.random.seed(42)
    X = np.random.randn(200, 2)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    # Fit the model
    model = Perceptron(learning_rate=0.01, max_iter=1000)
    model.fit(X, y)

    # Make predictions
    y_pred = model.predict(X)
    accuracy = model.score(X, y)

    print(f"Perceptron weights: {model.weights}")
    print(f"Perceptron bias: {model.bias}")
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
    # Plot decision boundary
    if model.weights[1] != 0:
        plt.plot([x_min, x_max], [-(model.weights[0]*x_min + model.bias)/model.weights[1], -(model.weights[0]*x_max + model.bias)/model.weights[1]], 'k-')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Perceptron Decision Boundary')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.show()
