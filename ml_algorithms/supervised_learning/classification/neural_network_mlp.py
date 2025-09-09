"""
Multi-Layer Perceptron (MLP) Classifier

Description:
MLP is a feedforward artificial neural network with one or more hidden layers.
It can model complex non-linear decision boundaries by learning hierarchical feature representations.

Core Intuition:
The network learns weights through backpropagation to minimize classification error.
Activation functions introduce non-linearity, enabling the network to learn complex patterns.

Best For:
- Complex classification tasks
- Non-linear decision boundaries
- Multi-class classification

Pros:
- Can model complex relationships
- Flexible architecture
- Supports multi-class problems

Cons:
- Requires tuning many hyperparameters
- Computationally intensive
- Prone to overfitting without regularization
- Requires large datasets for best performance

Key Hyperparameters:
- hidden_layer_sizes: Tuple specifying number of neurons per hidden layer
- learning_rate: Step size for weight updates
- max_iter: Maximum number of training iterations
- activation: Activation function ('relu', 'sigmoid', 'tanh')
"""

import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

class MLPClassifier:
    def __init__(self, hidden_layer_sizes=(10,), learning_rate=0.01, max_iter=1000, activation='relu'):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.activation_name = activation
        self.weights = []
        self.biases = []
        self.activation = relu if activation == 'relu' else sigmoid
        self.activation_derivative = relu_derivative if activation == 'relu' else sigmoid_derivative

    def _initialize_weights(self, n_features, n_classes):
        layer_sizes = [n_features] + list(self.hidden_layer_sizes) + [n_classes]
        self.weights = []
        self.biases = []
        for i in range(len(layer_sizes) - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2. / layer_sizes[i])
            b = np.zeros(layer_sizes[i+1])
            self.weights.append(w)
            self.biases.append(b)

    def _forward(self, X):
        activations = [X]
        zs = []
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            z = activations[-1].dot(w) + b
            zs.append(z)
            a = self.activation(z)
            activations.append(a)
        # Output layer with softmax
        z = activations[-1].dot(self.weights[-1]) + self.biases[-1]
        zs.append(z)
        exp_scores = np.exp(z - np.max(z, axis=1, keepdims=True))
        a = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        activations.append(a)
        return activations, zs

    def _backward(self, X, y, activations, zs):
        grads_w = [np.zeros_like(w) for w in self.weights]
        grads_b = [np.zeros_like(b) for b in self.biases]

        # One-hot encode y
        y_onehot = np.zeros_like(activations[-1])
        y_onehot[np.arange(len(y)), y] = 1

        delta = activations[-1] - y_onehot  # Cross-entropy loss gradient
        grads_w[-1] = activations[-2].T.dot(delta)
        grads_b[-1] = np.sum(delta, axis=0)

        for l in range(len(self.weights) - 2, -1, -1):
            delta = delta.dot(self.weights[l+1].T) * self.activation_derivative(zs[l])
            grads_w[l] = activations[l].T.dot(delta)
            grads_b[l] = np.sum(delta, axis=0)

        return grads_w, grads_b

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))

        self._initialize_weights(n_features, n_classes)

        for _ in range(self.max_iter):
            activations, zs = self._forward(X)
            grads_w, grads_b = self._backward(X, y, activations, zs)

            for i in range(len(self.weights)):
                self.weights[i] -= self.learning_rate * grads_w[i]
                self.biases[i] -= self.learning_rate * grads_b[i]

    def predict(self, X):
        X = np.asarray(X)
        activations, _ = self._forward(X)
        return np.argmax(activations[-1], axis=1)

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)

# Example usage
if __name__ == "__main__":
    np.random.seed(42)
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=200, n_features=4, n_classes=3, n_informative=3, random_state=42)

    model = MLPClassifier(hidden_layer_sizes=(10, 5), learning_rate=0.01, max_iter=500, activation='relu')
    model.fit(X, y)
    accuracy = model.score(X, y)

    print(f"MLP Classifier Accuracy: {accuracy:.4f}")
