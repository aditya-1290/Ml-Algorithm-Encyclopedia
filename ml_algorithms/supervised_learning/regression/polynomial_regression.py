"""
Polynomial Regression

Description:
Polynomial regression extends linear regression by adding polynomial terms (powers of the features)
to model non-linear relationships between the dependent and independent variables.

Core Intuition:
By transforming the features to include higher-order terms (x^2, x^3, etc.), the algorithm can
capture curved relationships that linear regression cannot. The model still uses linear regression
on the transformed features.

Best For:
- Modeling non-linear relationships
- Capturing curvature in data
- When linear regression underfits the data

Pros:
- Can model complex non-linear relationships
- Still uses efficient linear regression under the hood
- Interpretable when polynomial degree is low

Cons:
- Prone to overfitting with high polynomial degrees
- Can be computationally expensive for high degrees
- Extrapolation beyond data range can be unreliable
- Sensitive to outliers

Key Hyperparameters:
- degree: The degree of the polynomial (default: 2)
"""

import numpy as np
import matplotlib.pyplot as plt

class PolynomialFeatures:
    def __init__(self, degree=2):
        self.degree = degree

    def transform(self, X):
        """
        Transform input features to polynomial features.

        Parameters:
        X: array-like, shape (n_samples, n_features)

        Returns:
        X_poly: array-like, shape (n_samples, n_output_features)
        """
        X = np.asarray(X)
        n_samples, n_features = X.shape
        X_poly = np.ones((n_samples, 1))  # Start with bias term

        for d in range(1, self.degree + 1):
            for i in range(n_features):
                X_poly = np.c_[X_poly, X[:, i]**d]

        return X_poly

class PolynomialRegression:
    def __init__(self, degree=2):
        self.degree = degree
        self.poly_features = PolynomialFeatures(degree)
        self.weights = None

    def fit(self, X, y):
        """
        Fit the polynomial regression model.

        Parameters:
        X: array-like, shape (n_samples, n_features)
        y: array-like, shape (n_samples,)
        """
        X_poly = self.poly_features.transform(X)

        # Use normal equation for linear regression on polynomial features
        try:
            self.weights = np.linalg.inv(X_poly.T.dot(X_poly)).dot(X_poly.T).dot(y)
        except np.linalg.LinAlgError:
            self.weights = np.linalg.pinv(X_poly.T.dot(X_poly)).dot(X_poly.T).dot(y)

    def predict(self, X):
        """
        Predict target values for given features.

        Parameters:
        X: array-like, shape (n_samples, n_features)

        Returns:
        y_pred: array-like, shape (n_samples,)
        """
        if self.weights is None:
            raise ValueError("Model not fitted. Call fit() first.")
        X_poly = self.poly_features.transform(X)
        return X_poly.dot(self.weights)

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
    # Generate synthetic non-linear data
    np.random.seed(42)
    X = 6 * np.random.rand(100, 1) - 3
    y = 0.5 * X**2 + X + 2 + np.random.randn(100, 1).ravel()

    # Fit polynomial regression models of different degrees
    degrees = [1, 2, 3, 5]
    colors = ['red', 'blue', 'green', 'orange']

    plt.scatter(X, y, color='black', alpha=0.5, label='Data points')

    X_plot = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)

    for degree, color in zip(degrees, colors):
        model = PolynomialRegression(degree=degree)
        model.fit(X, y)
        y_plot = model.predict(X_plot)
        plt.plot(X_plot, y_plot, color=color, label=f'Degree {degree}')
        print(f"Degree {degree} R-squared: {model.score(X, y):.4f}")

    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Polynomial Regression with Different Degrees')
    plt.legend()
    plt.show()
