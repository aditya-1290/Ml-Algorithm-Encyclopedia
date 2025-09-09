"""
Hyperparameter Tuning Utilities

GridSearchCV and RandomizedSearchCV implementations from scratch.
"""

import numpy as np
from itertools import product
import random

def cross_val_score(estimator, X, y, cv=5, scoring='accuracy'):
    n_samples = X.shape[0]
    fold_size = n_samples // cv
    scores = []
    for i in range(cv):
        start = i * fold_size
        end = (i + 1) * fold_size if i < cv - 1 else n_samples
        X_test = X[start:end]
        y_test = y[start:end]
        X_train = np.concatenate([X[:start], X[end:]])
        y_train = np.concatenate([y[:start], y[end:]])
        
        estimator.fit(X_train, y_train)
        y_pred = estimator.predict(X_test)
        
        if scoring == 'accuracy':
            score = np.mean(y_test == y_pred)
        elif scoring == 'mse':
            score = -np.mean((y_test - y_pred) ** 2)  # Negative for consistency
        else:
            score = np.mean(y_test == y_pred)
        scores.append(score)
    return np.mean(scores)

class GridSearchCV:
    def __init__(self, estimator, param_grid, cv=5, scoring='accuracy'):
        self.estimator = estimator
        self.param_grid = param_grid
        self.cv = cv
        self.scoring = scoring
        self.best_params_ = None
        self.best_score_ = None
        self.cv_results_ = []

    def fit(self, X, y):
        param_combinations = list(product(*self.param_grid.values()))
        param_names = list(self.param_grid.keys())
        
        best_score = -np.inf
        best_params = None
        results = []
        
        for combo in param_combinations:
            params = dict(zip(param_names, combo))
            estimator_copy = self.estimator.__class__(**params)
            score = cross_val_score(estimator_copy, X, y, self.cv, self.scoring)
            results.append({'params': params, 'mean_test_score': score})
            
            if score > best_score:
                best_score = score
                best_params = params
        
        self.best_params_ = best_params
        self.best_score_ = best_score
        self.cv_results_ = results
        return self

class RandomizedSearchCV:
    def __init__(self, estimator, param_distributions, n_iter=10, cv=5, scoring='accuracy', random_state=None):
        self.estimator = estimator
        self.param_distributions = param_distributions
        self.n_iter = n_iter
        self.cv = cv
        self.scoring = scoring
        self.random_state = random_state
        self.best_params_ = None
        self.best_score_ = None
        self.cv_results_ = []

    def fit(self, X, y):
        if self.random_state:
            random.seed(self.random_state)
            np.random.seed(self.random_state)
        
        param_names = list(self.param_distributions.keys())
        best_score = -np.inf
        best_params = None
        results = []
        
        for _ in range(self.n_iter):
            params = {}
            for name, dist in self.param_distributions.items():
                if isinstance(dist, list):
                    params[name] = random.choice(dist)
                elif isinstance(dist, tuple) and len(dist) == 2:
                    params[name] = random.uniform(dist[0], dist[1])
                else:
                    params[name] = dist
            
            estimator_copy = self.estimator.__class__(**params)
            score = cross_val_score(estimator_copy, X, y, self.cv, self.scoring)
            results.append({'params': params, 'mean_test_score': score})
            
            if score > best_score:
                best_score = score
                best_params = params
        
        self.best_params_ = best_params
        self.best_score_ = best_score
        self.cv_results_ = results
        return self

# Example usage (assuming a simple classifier)
class SimpleClassifier:
    def __init__(self, param1=1, param2=0.1):
        self.param1 = param1
        self.param2 = param2
    
    def fit(self, X, y):
        pass
    
    def predict(self, X):
        return np.random.randint(0, 2, X.shape[0])

if __name__ == "__main__":
    # Dummy data
    X = np.random.randn(100, 2)
    y = np.random.randint(0, 2, 100)
    
    # Grid Search
    param_grid = {'param1': [1, 2, 3], 'param2': [0.1, 0.2]}
    grid_search = GridSearchCV(SimpleClassifier(), param_grid)
    grid_search.fit(X, y)
    print("Best params (Grid):", grid_search.best_params_)
    print("Best score (Grid):", grid_search.best_score_)
    
    # Random Search
    param_dist = {'param1': [1, 2, 3, 4, 5], 'param2': (0.1, 0.5)}
    random_search = RandomizedSearchCV(SimpleClassifier(), param_dist, n_iter=5)
    random_search.fit(X, y)
    print("Best params (Random):", random_search.best_params_)
    print("Best score (Random):", random_search.best_score_)
