"""
Model Evaluation Utilities

Functions for evaluating machine learning models.
"""

import numpy as np

# Classification Metrics
def accuracy_score(y_true, y_pred):
    return np.mean(y_true == y_pred)

def precision_score(y_true, y_pred, pos_label=1):
    tp = np.sum((y_true == pos_label) & (y_pred == pos_label))
    fp = np.sum((y_true != pos_label) & (y_pred == pos_label))
    return tp / (tp + fp) if (tp + fp) > 0 else 0

def recall_score(y_true, y_pred, pos_label=1):
    tp = np.sum((y_true == pos_label) & (y_pred == pos_label))
    fn = np.sum((y_true == pos_label) & (y_pred != pos_label))
    return tp / (tp + fn) if (tp + fn) > 0 else 0

def f1_score(y_true, y_pred, pos_label=1):
    p = precision_score(y_true, y_pred, pos_label)
    r = recall_score(y_true, y_pred, pos_label)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0

def confusion_matrix(y_true, y_pred):
    classes = np.unique(np.concatenate([y_true, y_pred]))
    n_classes = len(classes)
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for i, true_cls in enumerate(classes):
        for j, pred_cls in enumerate(classes):
            cm[i, j] = np.sum((y_true == true_cls) & (y_pred == pred_cls))
    return cm

# Regression Metrics
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

# Cross-validation
def train_test_split(X, y, test_size=0.2, random_state=None):
    if random_state:
        np.random.seed(random_state)
    n_samples = X.shape[0]
    indices = np.random.permutation(n_samples)
    test_size = int(n_samples * test_size)
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]

# Example usage
if __name__ == "__main__":
    # Classification
    y_true_cls = np.array([0, 1, 1, 0, 1])
    y_pred_cls = np.array([0, 1, 0, 0, 1])
    print("Accuracy:", accuracy_score(y_true_cls, y_pred_cls))
    print("Precision:", precision_score(y_true_cls, y_pred_cls))
    print("Recall:", recall_score(y_true_cls, y_pred_cls))
    print("F1 Score:", f1_score(y_true_cls, y_pred_cls))
    print("Confusion Matrix:\n", confusion_matrix(y_true_cls, y_pred_cls))

    # Regression
    y_true_reg = np.array([1, 2, 3, 4, 5])
    y_pred_reg = np.array([1.1, 1.9, 3.2, 3.8, 4.9])
    print("MSE:", mean_squared_error(y_true_reg, y_pred_reg))
    print("MAE:", mean_absolute_error(y_true_reg, y_pred_reg))
    print("R2 Score:", r2_score(y_true_reg, y_pred_reg))
