"""
t-Distributed Stochastic Neighbor Embedding (t-SNE)

Description:
t-SNE is a non-linear dimensionality reduction technique particularly well-suited for
visualizing high-dimensional data in 2D or 3D by preserving local structure.

Core Intuition:
The algorithm converts similarities between data points to joint probabilities and tries
to minimize the Kullback-Leibler divergence between the joint probabilities of the low-dimensional
and high-dimensional representations.

Best For:
- Data visualization (2D/3D)
- Preserving local structure
- Exploratory data analysis

Pros:
- Excellent for visualizing high-dimensional data
- Preserves local structure very well
- Can reveal clusters and patterns

Cons:
- Computationally expensive
- Stochastic (results can vary)
- Not suitable for transforming new data
- Hyperparameter sensitive

Key Hyperparameters:
- n_components: Number of dimensions in the embedded space (usually 2 or 3)
- perplexity: Related to the number of nearest neighbors
- learning_rate: Learning rate for optimization
- max_iter: Maximum number of iterations
"""

import numpy as np
import matplotlib.pyplot as plt

class TSNE:
    def __init__(self, n_components=2, perplexity=30.0, learning_rate=200.0, max_iter=1000, random_state=None):
        self.n_components = n_components
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.random_state = random_state
        self.embedding = None

    def _compute_pairwise_distances(self, X):
        """Compute pairwise Euclidean distances."""
        sum_X = np.sum(X**2, axis=1)
        return np.sqrt(np.maximum(sum_X[:, np.newaxis] + sum_X[np.newaxis, :] - 2 * np.dot(X, X.T), 0))

    def _compute_p_conditional(self, distances, sigmas):
        """Compute conditional probabilities P_{j|i}."""
        n_samples = distances.shape[0]
        P = np.zeros((n_samples, n_samples))

        for i in range(n_samples):
            # Compute P_{j|i} for all j
            P[i, :] = np.exp(-distances[i, :] ** 2 / (2 * sigmas[i] ** 2))
            P[i, i] = 0  # Set diagonal to 0
            P[i, :] /= np.sum(P[i, :])  # Normalize

        return P

    def _compute_p_joint(self, P_conditional):
        """Compute joint probabilities P."""
        return (P_conditional + P_conditional.T) / (2 * P_conditional.shape[0])

    def _compute_q_joint(self, Y):
        """Compute joint probabilities Q in low-dimensional space."""
        distances = self._compute_pairwise_distances(Y)
        n_samples = distances.shape[0]

        # Student's t-distribution (degrees of freedom = 1)
        Q = 1 / (1 + distances**2)
        Q = Q / np.sum(Q)  # Normalize

        return Q

    def _compute_gradients(self, P, Q, Y):
        """Compute gradients for optimization."""
        n_samples = P.shape[0]
        gradients = np.zeros_like(Y)

        for i in range(n_samples):
            for j in range(n_samples):
                if i != j:
                    diff = Y[i] - Y[j]
                    gradients[i] += (P[i, j] - Q[i, j]) * diff * (1 / (1 + np.sum(diff**2)))

        return 4 * gradients

    def _optimize(self, P, Y):
        """Optimize the embedding using gradient descent."""
        for iteration in range(self.max_iter):
            Q = self._compute_q_joint(Y)
            gradients = self._compute_gradients(P, Q, Y)

            # Update Y
            Y -= self.learning_rate * gradients

            # Center the embedding
            Y -= np.mean(Y, axis=0)

            # Print progress
            if (iteration + 1) % 100 == 0:
                kl_divergence = np.sum(P * np.log(np.maximum(P, 1e-12) / np.maximum(Q, 1e-12)))
                print(f"Iteration {iteration + 1}, KL divergence: {kl_divergence:.4f}")

    def fit_transform(self, X):
        """
        Fit the t-SNE model and transform the data.

        Parameters:
        X: array-like, shape (n_samples, n_features)

        Returns:
        embedding: array-like, shape (n_samples, n_components)
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)

        X = np.asarray(X)
        n_samples, n_features = X.shape

        # Compute pairwise distances
        distances = self._compute_pairwise_distances(X)

        # Find optimal sigmas for each point
        sigmas = self._find_optimal_sigmas(distances)

        # Compute conditional probabilities
        P_conditional = self._compute_p_conditional(distances, sigmas)

        # Compute joint probabilities
        P = self._compute_p_joint(P_conditional)

        # Initialize embedding randomly
        Y = np.random.randn(n_samples, self.n_components)

        # Optimize
        self._optimize(P, Y)

        self.embedding = Y
        return Y

    def _find_optimal_sigmas(self, distances):
        """Find optimal sigma for each point using binary search."""
        n_samples = distances.shape[0]
        sigmas = np.zeros(n_samples)

        for i in range(n_samples):
            # Binary search for sigma that gives perplexity closest to target
            sigma_min = 1e-10
            sigma_max = np.max(distances[i, :])

            for _ in range(50):  # Binary search iterations
                sigma = (sigma_min + sigma_max) / 2

                # Compute conditional probabilities
                P_i = np.exp(-distances[i, :] ** 2 / (2 * sigma ** 2))
                P_i[i] = 0  # Exclude self
                P_i /= np.sum(P_i)

                # Compute perplexity
                entropy = -np.sum(P_i * np.log2(P_i + 1e-12))
                perplexity = 2 ** entropy

                if perplexity < self.perplexity:
                    sigma_min = sigma
                else:
                    sigma_max = sigma

            sigmas[i] = sigma

        return sigmas

# Example usage
if __name__ == "__main__":
    # Generate synthetic data with clusters
    np.random.seed(42)
    n_samples = 100

    # Cluster 1
    X1 = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], n_samples//4)

    # Cluster 2
    X2 = np.random.multivariate_normal([5, 5], [[1, 0], [0, 1]], n_samples//4)

    # Cluster 3
    X3 = np.random.multivariate_normal([0, 5], [[1, 0], [0, 1]], n_samples//4)

    # Cluster 4
    X4 = np.random.multivariate_normal([5, 0], [[1, 0], [0, 1]], n_samples//4)

    X = np.vstack([X1, X2, X3, X4])
    y = np.repeat(range(4), n_samples//4)

    # Fit t-SNE
    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, max_iter=300, random_state=42)
    X_tsne = tsne.fit_transform(X)

    print(f"Original shape: {X.shape}")
    print(f"t-SNE shape: {X_tsne.shape}")

    # Plot results
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis', alpha=0.6)
    plt.title('t-SNE Visualization')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.show()
