"""
Gaussian Mixture Models (GMM)

Description:
GMM is a probabilistic model that assumes the data is generated from a mixture of several
Gaussian distributions with unknown parameters.

Core Intuition:
The algorithm uses the Expectation-Maximization (EM) algorithm to estimate the parameters
of the Gaussian components (means, covariances, and mixing coefficients) that best fit the data.

Best For:
- Soft clustering (probabilistic cluster assignment)
- When clusters have different sizes and shapes
- Density estimation

Pros:
- Provides probabilistic cluster assignments
- Can model clusters with different shapes and sizes
- Can be used for density estimation
- Handles overlapping clusters

Cons:
- Sensitive to initialization
- Assumes Gaussian distributions
- Can converge to local optima
- Requires specifying number of components

Key Hyperparameters:
- n_components: Number of mixture components
- max_iter: Maximum number of EM iterations
- tol: Tolerance for convergence
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

class GaussianMixture:
    def __init__(self, n_components=3, max_iter=100, tol=1e-3, random_state=None):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.weights = None
        self.means = None
        self.covariances = None
        self.labels = None

    def _initialize_parameters(self, X):
        if self.random_state is not None:
            np.random.seed(self.random_state)

        n_samples, n_features = X.shape

        # Initialize weights uniformly
        self.weights = np.ones(self.n_components) / self.n_components

        # Initialize means randomly from data
        indices = np.random.choice(n_samples, self.n_components, replace=False)
        self.means = X[indices].copy()

        # Initialize covariances as identity matrices
        self.covariances = np.array([np.eye(n_features)] * self.n_components)

    def _e_step(self, X):
        """Expectation step: compute responsibilities."""
        n_samples = X.shape[0]
        responsibilities = np.zeros((n_samples, self.n_components))

        for i in range(n_samples):
            for k in range(self.n_components):
                try:
                    responsibilities[i, k] = self.weights[k] * multivariate_normal.pdf(
                        X[i], self.means[k], self.covariances[k]
                    )
                except np.linalg.LinAlgError:
                    # Handle singular covariance matrix
                    responsibilities[i, k] = self.weights[k] * 1e-10

            # Normalize responsibilities
            total = np.sum(responsibilities[i])
            if total > 0:
                responsibilities[i] /= total
            else:
                # If all probabilities are zero, assign equal responsibility
                responsibilities[i] = 1.0 / self.n_components

        return responsibilities

    def _m_step(self, X, responsibilities):
        """Maximization step: update parameters."""
        n_samples = X.shape[0]
        n_features = X.shape[1]

        # Update weights
        N_k = np.sum(responsibilities, axis=0)
        self.weights = N_k / n_samples

        # Update means
        for k in range(self.n_components):
            if N_k[k] > 0:
                self.means[k] = np.sum(responsibilities[:, k, np.newaxis] * X, axis=0) / N_k[k]
            else:
                # If no responsibility, reinitialize
                self.means[k] = X[np.random.randint(n_samples)]

        # Update covariances
        for k in range(self.n_components):
            if N_k[k] > 0:
                diff = X - self.means[k]
                self.covariances[k] = np.sum(
                    responsibilities[:, k, np.newaxis, np.newaxis] * 
                    np.einsum('ij,ik->ijk', diff, diff), axis=0
                ) / N_k[k]
                
                # Regularize covariance matrix to prevent singularity
                self.covariances[k] += np.eye(n_features) * 1e-6
            else:
                self.covariances[k] = np.eye(n_features)

    def fit(self, X):
        """
        Fit the Gaussian Mixture Model to the data.

        Parameters:
        X: array-like, shape (n_samples, n_features)
        """
        X = np.asarray(X)
        self._initialize_parameters(X)

        for _ in range(self.max_iter):
            old_means = self.means.copy()
            responsibilities = self._e_step(X)
            self._m_step(X, responsibilities)

            # Check for convergence
            if np.sum((self.means - old_means)**2) < self.tol:
                break

    def predict(self, X):
        """
        Predict cluster labels for new data.

        Parameters:
        X: array-like, shape (n_samples, n_features)

        Returns:
        labels: array-like, shape (n_samples,)
        """
        if self.means is None:
            raise ValueError("Model not fitted. Call fit() first.")
        responsibilities = self._e_step(X)
        return np.argmax(responsibilities, axis=1)

    def predict_proba(self, X):
        """
        Predict cluster probabilities for new data.

        Parameters:
        X: array-like, shape (n_samples, n_features)

        Returns:
        probabilities: array-like, shape (n_samples, n_components)
        """
        if self.means is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self._e_step(X)

    def fit_predict(self, X):
        """
        Fit the model and return cluster labels.

        Parameters:
        X: array-like, shape (n_samples, n_features)

        Returns:
        labels: array-like, shape (n_samples,)
        """
        self.fit(X)
        return self.predict(X)

# Example usage
if __name__ == "__main__":
    # Generate synthetic data from mixture of Gaussians
    np.random.seed(42)
    n_samples = 300

    # Component 1
    mean1 = [2, 2]
    cov1 = [[1, 0.5], [0.5, 1]]
    X1 = np.random.multivariate_normal(mean1, cov1, n_samples//3)

    # Component 2
    mean2 = [-2, -2]
    cov2 = [[1, -0.5], [-0.5, 1]]
    X2 = np.random.multivariate_normal(mean2, cov2, n_samples//3)

    # Component 3
    mean3 = [0, 0]
    cov3 = [[0.5, 0], [0, 0.5]]
    X3 = np.random.multivariate_normal(mean3, cov3, n_samples//3)

    X = np.vstack([X1, X2, X3])

    # Fit GMM
    gmm = GaussianMixture(n_components=3, random_state=42)
    labels = gmm.fit_predict(X)

    # Plot results
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.6)
    plt.scatter(gmm.means[:, 0], gmm.means[:, 1], c='red', marker='x', s=200, linewidth=3)
    plt.title('Gaussian Mixture Model Clustering')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

    print(f"Estimated means: {gmm.means}")
    print(f"Estimated weights: {gmm.weights}")
