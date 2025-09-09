"""
Sentiment Analysis

Core Idea: Identifying and categorizing opinions in text (e.g., positive/negative).
Uses machine learning to classify text sentiment.

Best For: Customer feedback, social media monitoring, review analysis.

Pros:
- Automates sentiment detection
- Scalable to large datasets
- Can be fine-tuned for domains

Cons:
- Requires labeled training data
- Struggles with sarcasm and context
- Accuracy depends on data quality

Key Hyperparameters:
- vectorizer: Type of text representation (BoW, TF-IDF)
- classifier: ML algorithm for classification
- n_features: Number of features
"""

import numpy as np
from collections import Counter
import re

# Simple Logistic Regression for classification
class SimpleLogisticRegression:
    def __init__(self, learning_rate=0.01, epochs=100):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = 0

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        for _ in range(self.epochs):
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(linear_model)
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict_proba(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        return self.sigmoid(linear_model)

    def predict(self, X):
        return (self.predict_proba(X) > 0.5).astype(int)

# Simple CountVectorizer
class SimpleCountVectorizer:
    def __init__(self, max_features=1000):
        self.max_features = max_features
        self.vocabulary_ = {}

    def _tokenize(self, text):
        text = re.sub(r'[^\w\s]', '', text.lower())
        return text.split()

    def fit(self, documents):
        word_counts = Counter()
        for doc in documents:
            tokens = self._tokenize(doc)
            word_counts.update(tokens)
        most_common = word_counts.most_common(self.max_features)
        self.vocabulary_ = {word: i for i, (word, _) in enumerate(most_common)}
        return self

    def transform(self, documents):
        X = np.zeros((len(documents), len(self.vocabulary_)))
        for i, doc in enumerate(documents):
            tokens = self._tokenize(doc)
            for token in tokens:
                if token in self.vocabulary_:
                    X[i, self.vocabulary_[token]] += 1
        return X

    def fit_transform(self, documents):
        return self.fit(documents).transform(documents)

class SentimentAnalyzer:
    def __init__(self):
        self.vectorizer = SimpleCountVectorizer()
        self.classifier = SimpleLogisticRegression()

    def fit(self, texts, labels):
        X = self.vectorizer.fit_transform(texts)
        self.classifier.fit(X, labels)

    def predict(self, texts):
        X = self.vectorizer.transform(texts)
        return self.classifier.predict(X)

    def predict_sentiment(self, texts):
        predictions = self.predict(texts)
        sentiments = ['Negative' if pred == 0 else 'Positive' for pred in predictions]
        return sentiments

# Example usage
if __name__ == "__main__":
    # Sample training data
    texts = [
        "I love this product",
        "This is amazing",
        "Great quality",
        "Terrible experience",
        "Worst purchase ever",
        "Not satisfied"
    ]
    labels = [1, 1, 1, 0, 0, 0]  # 1: positive, 0: negative

    analyzer = SentimentAnalyzer()
    analyzer.fit(texts, labels)

    test_texts = ["I really like it", "This is bad"]
    sentiments = analyzer.predict_sentiment(test_texts)
    print("Sentiments:", sentiments)
