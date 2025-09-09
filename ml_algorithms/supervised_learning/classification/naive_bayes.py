"""
Naive Bayes Classifier

Description:
Naive Bayes is a probabilistic classifier based on Bayes' theorem with the assumption that
features are conditionally independent given the class label.

Core Intuition:
The algorithm calculates the probability of each class given the features by multiplying
the prior probability of the class with the likelihood of each feature given the class,
then normalizes to get posterior probabilities.

Best For:
- Text classification
- Spam filtering
- When features are conditionally independent
- Small to medium datasets

Pros:
- Simple and fast
- Works well with high-dimensional data
- Handles missing values
- Requires little training data

Cons:
- Assumes feature independence (often violated)
- Can be dominated by features with many categories
- Not suitable for regression
- Sensitive to irrelevant features

Key Hyperparameters:
- None (but smoothing can be added for categorical features)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

class GaussianNaiveBayes:
    def __init__(self):
        self.classes = None
        self.class_priors = {}
        self.class_means = {}
        self.class_variances = {}

    def fit(self, X, y):
        """
        Fit the Gaussian Naive Bayes model.

        Parameters:
        X: array-like, shape (n_samples, n_features)
        y: array-like, shape (n_samples,)
        """
        X = np.asarray(X)
        y = np.asarray(y)
        self.classes = np.unique(y)

        for cls in self.classes:
            X_cls = X[y == cls]
            self.class_priors[cls] = len(X_cls) / len(X)
            self.class_means[cls] = np.mean(X_cls, axis=0)
            self.class_variances[cls] = np.var(X_cls, axis=0)

    def _calculate_likelihood(self, x, cls):
        """Calculate likelihood of feature vector given class."""
        mean = self.class_means[cls]
        var = self.class_variances[cls]
        # Avoid division by zero
        var = np.where(var == 0, 1e-6, var)
        # Product of Gaussian probabilities
        likelihood = 1.0
        for i in range(len(x)):
            likelihood *= norm.pdf(x[i], mean[i], np.sqrt(var[i]))
        return likelihood

    def predict_proba(self, X):
        """
        Predict class probabilities for given features.

        Parameters:
        X: array-like, shape (n_samples, n_features)

        Returns:
        probabilities: array-like, shape (n_samples, n_classes)
        """
        if self.classes is None:
            raise ValueError("Model not fitted. Call fit() first.")
        X = np.asarray(X)
        probabilities = []

        for x in X:
            class_probs = {}
            for cls in self.classes:
                prior = self.class_priors[cls]
                likelihood = self._calculate_likelihood(x, cls)
                class_probs[cls] = prior * likelihood

            # Normalize to get posterior probabilities
            total = sum(class_probs.values())
            if total == 0:
                # Handle case where all likelihoods are zero
                posterior = [1.0 / len(self.classes)] * len(self.classes)
            else:
                posterior = [class_probs[cls] / total for cls in self.classes]
            probabilities.append(posterior)

        return np.array(probabilities)

    def predict(self, X):
        """
        Predict class labels for given features.

        Parameters:
        X: array-like, shape (n_samples, n_features)

        Returns:
        y_pred: array-like, shape (n_samples,)
        """
        proba = self.predict_proba(X)
        return self.classes[np.argmax(proba, axis=1)]

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
    # Generate synthetic data with Gaussian distributions
    np.random.seed(42)
    n_samples = 300
    # Class 0: mean [2, 2], variance [1, 1]
    X0 = np.random.multivariate_normal([2, 2], [[1, 0], [0, 1]], n_samples//2)
    y0 = np.zeros(n_samples//2)
    # Class 1: mean [-2, -2], variance [1, 1]
    X1 = np.random.multivariate_normal([-2, -2], [[1, 0], [0, 1]], n_samples//2)
    y1 = np.ones(n_samples//2)

    X = np.vstack([X0, X1])
    y = np.hstack([y0, y1])

    # Fit the model
    model = GaussianNaiveBayes()
    model.fit(X, y)
    accuracy = model.score(X, y)

    print(f"Class priors: {model.class_priors}")
    print(f"Class means: {model.class_means}")
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
    plt.title('Gaussian Naive Bayes Decision Boundary')
    plt.show()
