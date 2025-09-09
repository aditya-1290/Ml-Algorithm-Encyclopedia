"""
Kernel Principal Component Analysis (Kernel PCA)

Description:
Kernel PCA is an extension of PCA that uses the kernel trick to perform non-linear dimensionality
reduction by implicitly mapping the data to a higher-dimensional feature space.

Core Intuition:
By using kernel functions, the algorithm can find non-linear principal components without
explicitly computing the mapping to the higher-dimensional space.

Best For:
- Non-linear dimensionality reduction
- When data has non-linear structure
- Feature extraction for non-linear problems

Pros:
- Can capture non-linear relationships
- Uses kernel trick for computational efficiency
- Flexible with different kernel functions

Cons:
- More computationally expensive than linear PCA
- Requires choosing appropriate kernel and parameters
- Harder to interpret the components
- Memory intensive for large datasets

Key Hyperparameters:
- n_components: Number of principal components to keep
- kernel: Kernel function ('rbf', 'poly', 'sigmoid')
- gamma: Kernel coefficient for rbf/poly kernels
- degree: Degree for polynomial kernel
"""

import numpy as np
import matplotlib.pyplot as plt

class KernelPCA:
    def __init__(self, n_components=None, kernel='rbf', gamma=None, degree=3):
        self.n_components = n_components
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.alphas = None
        self.lambdas = None
        self.X_fit = None

    def _kernel_function(self, X, Y=None):
        """Compute kernel matrix."""
        if Y is None:
            Y = X

        if self.kernel == 'rbf':
            if self.gamma is None:
                self.gamma = 1.0 / X.shape[1]
            K = np.exp(-self.gamma * np.sum((X[:, np.newaxis] - Y[np.newaxis, :])**2, axis=2))
        elif self.kernel == 'poly':
            if self.gamma is None:
                self.gamma = 1.0
            K = (self.gamma * np.dot(X, Y.T) + 1) ** self.degree
        elif self.kernel == 'sigmoid':
            if self.gamma is None:
                self.gamma = 1.0
            K = np.tanh(self.gamma * np.dot(X, Y.T))
        else:
            raise ValueError(f"Unsupported kernel: {self.kernel}")

        return K

    def fit(self, X):
        """
        Fit the Kernel PCA model to the data.

        Parameters:
        X: array-like, shape (n_samples, n_features)
        """
        X = np.asarray(X)
        n_samples = X.shape[0]

        # Compute kernel matrix
        K = self._kernel_function(X)

        # Center the kernel matrix
        one_n = np.ones((n_samples, n_samples)) / n_samples
        K_centered = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)

        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(K_centered)

        # Sort in descending order
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]

        # Determine number of components
        if self.n_components is None:
            self.n_components = n_samples
        elif self.n_components > n_samples:
            self.n_components = n_samples

        # Store the first n_components eigenvectors and eigenvalues
        self.lambdas = eigenvalues[:self.n_components]
        self.alphas = eigenvectors[:, :self.n_components]

        # Normalize alphas
        for i in range(self.n_components):
            if self.lambdas[i] > 0:
                self.alphas[:, i] /= np.sqrt(self.lambdas[i])

        self.X_fit = X

    def transform(self, X):
        """
        Transform the data to the kernel PCA space.

        Parameters:
        X: array-like, shape (n_samples, n_features)

        Returns:
        X_transformed: array-like, shape (n_samples, n_components)
        """
        if self.alphas is None:
            raise ValueError("Model not fitted. Call fit() first.")
        X = np.asarray(X)

        # Compute kernel matrix between X and X_fit
        K = self._kernel_function(X, self.X_fit)

        # Center the kernel matrix
        n_samples = X.shape[0]
        n_fit = self.X_fit.shape[0]
        one_n = np.ones((n_samples, n_fit)) / n_fit
        one_m = np.ones((n_fit, n_fit)) / n_fit

        K_centered = K - one_n.dot(self._kernel_function(self.X_fit)) - \
                     (self._kernel_function(X, self.X_fit) - one_n.dot(self._kernel_function(self.X_fit))).dot(one_m) + \
                     one_n.dot(self._kernel_function(self.X_fit)).dot(one_m)

        # Project onto principal components
        return np.dot(K_centered, self.alphas)

    def fit_transform(self, X):
        """
        Fit the model and transform the data.

        Parameters:
        X: array-like, shape (n_samples, n_features)

        Returns:
        X_transformed: array-like, shape (n_samples, n_components)
        """
        self.fit(X)
        return self.transform(X)

# Example usage
if __name__ == "__main__":
    # Generate non-linear data (concentric circles)
    np.random.seed(42)
    n_samples = 200

    # Outer circle
    theta = np.random.uniform(0, 2*np.pi, n_samples//2)
    r = 2 + np.random.normal(0, 0.1, n_samples//2)
    X1 = np.column_stack([r * np.cos(theta), r * np.sin(theta)])

    # Inner circle
    theta = np.random.uniform(0, 2*np.pi, n_samples//2)
    r = 1 + np.random.normal(0, 0.1, n_samples//2)
    X2 = np.column_stack([r * np.cos(theta), r * np.sin(theta)])

    X = np.vstack([X1, X2])
    y = np.hstack([np.zeros(n_samples//2), np.ones(n_samples//2)])

    # Fit Kernel PCA
    kpca = KernelPCA(n_components=2, kernel='rbf', gamma=1.0)
    X_kpca = kpca.fit_transform(X)

    print(f"Original shape: {X.shape}")
    print(f"Transformed shape: {X_kpca.shape}")

    # Plot original data
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', alpha=0.6)
    plt.title('Original Data')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')

    # Plot Kernel PCA transformed data
    plt.subplot(1, 2, 2)
    plt.scatter(X_kpca[:, 0], X_kpca[:, 1], c=y, cmap='viridis', alpha=0.6)
    plt.title('Kernel PCA (RBF kernel)')
    plt.xlabel('PC1')
    plt.ylabel('PC2')

    plt.tight_layout()
    plt.show()
