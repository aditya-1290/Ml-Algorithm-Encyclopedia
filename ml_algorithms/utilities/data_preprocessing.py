"""
Data Preprocessing Utilities

Includes standard preprocessing techniques for machine learning data.
"""

import numpy as np

class StandardScaler:
    """
    Standardize features by removing the mean and scaling to unit variance.
    """
    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X):
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)
        return self

    def transform(self, X):
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, X):
        return X * self.scale_ + self.mean_

class MinMaxScaler:
    """
    Scale features to a given range, usually [0, 1].
    """
    def __init__(self, feature_range=(0, 1)):
        self.feature_range = feature_range
        self.min_ = None
        self.scale_ = None

    def fit(self, X):
        self.min_ = np.min(X, axis=0)
        self.scale_ = np.max(X, axis=0) - self.min_
        return self

    def transform(self, X):
        return (X - self.min_) / self.scale_ * (self.feature_range[1] - self.feature_range[0]) + self.feature_range[0]

    def fit_transform(self, X):
        return self.fit(X).transform(X)

class LabelEncoder:
    """
    Encode categorical labels with value between 0 and n_classes-1.
    """
    def __init__(self):
        self.classes_ = None

    def fit(self, y):
        self.classes_ = np.unique(y)
        return self

    def transform(self, y):
        return np.searchsorted(self.classes_, y)

    def fit_transform(self, y):
        return self.fit(y).transform(y)

    def inverse_transform(self, y):
        return self.classes_[y]

class OneHotEncoder:
    """
    Encode categorical features as one-hot vectors.
    """
    def __init__(self):
        self.categories_ = None

    def fit(self, X):
        self.categories_ = [np.unique(X[:, i]) for i in range(X.shape[1])]
        return self

    def transform(self, X):
        encoded = []
        for i in range(X.shape[1]):
            one_hot = np.zeros((X.shape[0], len(self.categories_[i])))
            for j, cat in enumerate(self.categories_[i]):
                one_hot[X[:, i] == cat, j] = 1
            encoded.append(one_hot)
        return np.hstack(encoded)

    def fit_transform(self, X):
        return self.fit(X).transform(X)

# Example usage
if __name__ == "__main__":
    # StandardScaler
    X = np.array([[1, 2], [3, 4], [5, 6]])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("Standardized X:\n", X_scaled)

    # LabelEncoder
    y = np.array(['cat', 'dog', 'cat', 'bird'])
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    print("Encoded y:", y_encoded)
    print("Inverse:", encoder.inverse_transform(y_encoded))
