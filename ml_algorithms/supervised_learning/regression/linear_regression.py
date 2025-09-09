"""
Linear Regression

Description:
Linear regression is a supervised learning algorithm that finds the best-fitting straight line
through the data points. It models the relationship between a dependent variable (target) and
one or more independent variables (features) by fitting a linear equation.

Core Intuition:
The algorithm minimizes the sum of squared differences between the predicted values and the
actual values (least squares method). This finds the line that best represents the data.

Best For:
- Predicting continuous numerical values
- Understanding linear relationships between variables
- Baseline model for regression tasks

Pros:
- Simple and interpretable
- Fast training and prediction
- No hyperparameters to tune
- Works well with small datasets

Cons:
- Assumes linear relationship between features and target
- Sensitive to outliers
- Can overfit with too many features (multicollinearity)
- Not suitable for non-linear relationships

Key Hyperparameters:
- None (intercept is automatically included)

Implementation from scratch using numpy.
"""

import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self):
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        """
        Fit the linear regression model using normal equation.

        Parameters:
        X: array-like, shape (n_samples, n_features)
        y: array-like, shape (n_samples,)
        """
        # Add bias term (intercept) by adding a column of ones
        X_b = np.c_[np.ones((X.shape[0], 1)), X]

        # Normal equation: w = (X^T * X)^-1 * X^T * y
        try:
            self.weights = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
            self.bias = self.weights[0]
            self.weights = self.weights[1:]
        except np.linalg.LinAlgError:
            # Handle singular matrix (multicollinearity)
            print("Warning: Matrix is singular. Using pseudoinverse.")
            self.weights = np.linalg.pinv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
            self.bias = self.weights[0]
            self.weights = self.weights[1:]

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
        return X.dot(self.weights) + self.bias

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
    X = 2 * np.random.rand(100, 1)
    y = 4 + 3 * X + np.random.randn(100, 1).ravel()  # y = 4 + 3x + noise

    # Fit the model
    model = LinearRegression()
    model.fit(X, y)

    # Make predictions
    X_new = np.array([[0], [2]])
    y_pred = model.predict(X_new)

    print(f"Model weights: {model.weights}")
    print(f"Model bias: {model.bias}")
    print(f"Predictions for X=[0, 2]: {y_pred}")
    print(f"R-squared score: {model.score(X, y):.4f}")

    # Plot the results
    plt.scatter(X, y, color='blue', label='Data points')
    plt.plot(X, model.predict(X), color='red', label='Regression line')
    plt.scatter(X_new, y_pred, color='green', marker='x', s=100, label='Predictions')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Linear Regression Example')
    plt.legend()
    plt.show()
