"""
Principal Component Analysis (PCA)

Description:
PCA is a dimensionality reduction technique that transforms the data to a new coordinate system
where the greatest variances lie on the first coordinates (principal components).

Core Intuition:
The algorithm finds the directions (principal components) that maximize the variance in the data,
allowing for dimensionality reduction while preserving as much information as possible.

Best For:
- High-dimensional datasets
- Feature extraction
- Data visualization (2D/3D)
- Noise reduction

Pros:
- Unsupervised and deterministic
- Reduces dimensionality while preserving variance
- Can be used for feature extraction
- Helps with multicollinearity

Cons:
- Linear transformation only
- Sensitive to feature scaling
- Interpretability of components can be difficult
- Assumes linear relationships

Key Hyperparameters:
- n_components: Number of principal components to keep
"""

import numpy as np
import matplotlib.pyplot as plt

class PCA:
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.components = None
        self.mean = None
        self.explained_variance = None
        self.explained_variance_ratio = None

    def fit(self, X):
        """
        Fit the PCA model to the data.

        Parameters:
        X: array-like, shape (n_samples, n_features)
        """
        X = np.asarray(X)
        n_samples, n_features = X.shape

        # Center the data
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        # Compute covariance matrix
        covariance_matrix = np.cov(X_centered.T)

        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

        # Sort eigenvalues and eigenvectors in descending order
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]

        # Determine number of components
        if self.n_components is None:
            self.n_components = n_features
        elif self.n_components > n_features:
            self.n_components = n_features

        # Store the principal components
        self.components = eigenvectors[:, :self.n_components]

        # Calculate explained variance
        total_variance = np.sum(eigenvalues)
        self.explained_variance = eigenvalues[:self.n_components]
        self.explained_variance_ratio = self.explained_variance / total_variance

    def transform(self, X):
        """
        Transform the data to the new coordinate system.

        Parameters:
        X: array-like, shape (n_samples, n_features)

        Returns:
        X_transformed: array-like, shape (n_samples, n_components)
        """
        if self.components is None:
            raise ValueError("Model not fitted. Call fit() first.")
        X = np.asarray(X)
        X_centered = X - self.mean
        return np.dot(X_centered, self.components)

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

    def inverse_transform(self, X_transformed):
        """
        Transform data back to original space.

        Parameters:
        X_transformed: array-like, shape (n_samples, n_components)

        Returns:
        X_reconstructed: array-like, shape (n_samples, n_features)
        """
        if self.components is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return np.dot(X_transformed, self.components.T) + self.mean

# Example usage
if __name__ == "__main__":
    # Generate synthetic high-dimensional data
    np.random.seed(42)
    n_samples = 200
    n_features = 10

    # Create correlated features
    X = np.random.randn(n_samples, n_features)
    # Add correlation
    X[:, 1] = X[:, 0] + 0.5 * np.random.randn(n_samples)
    X[:, 2] = 2 * X[:, 0] + 0.1 * np.random.randn(n_samples)

    # Fit PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    print(f"Original shape: {X.shape}")
    print(f"Transformed shape: {X_pca.shape}")
    print(f"Explained variance ratio: {pca.explained_variance_ratio}")
    print(f"Cumulative explained variance: {np.cumsum(pca.explained_variance_ratio)}")

    # Plot the first two principal components
    plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('PCA: First Two Principal Components')
    plt.show()

    # Plot explained variance
    plt.bar(range(1, len(pca.explained_variance_ratio) + 1), pca.explained_variance_ratio)
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Explained Variance by Principal Components')
    plt.show()
