"""
Bag-of-Words (BoW)

Core Idea: Represents text as a multiset (bag) of words, ignoring grammar and word order.
Each document is a vector of word counts.

Best For: Text classification, sentiment analysis, topic modeling.

Pros:
- Simple and intuitive
- Works well with traditional ML algorithms
- Fast to compute

Cons:
- Ignores word order and context
- High dimensionality (vocabulary size)
- Sparse representations

Key Hyperparameters:
- max_features: Maximum number of features (words)
- ngram_range: Range of n-grams to include
- stop_words: Words to ignore
"""

import numpy as np
from collections import Counter
import re

class CountVectorizer:
    def __init__(self, max_features=None, ngram_range=(1, 1), stop_words=None):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.stop_words = set(stop_words) if stop_words else set()
        self.vocabulary_ = {}
        self.feature_names_ = []

    def _tokenize(self, text):
        text = re.sub(r'[^\w\s]', '', text.lower())
        tokens = text.split()
        if self.stop_words:
            tokens = [t for t in tokens if t not in self.stop_words]
        return tokens

    def _get_ngrams(self, tokens):
        ngrams = []
        for n in range(self.ngram_range[0], self.ngram_range[1] + 1):
            for i in range(len(tokens) - n + 1):
                ngrams.append(' '.join(tokens[i:i+n]))
        return ngrams

    def fit(self, documents):
        word_counts = Counter()
        for doc in documents:
            tokens = self._tokenize(doc)
            ngrams = self._get_ngrams(tokens)
            word_counts.update(ngrams)

        if self.max_features:
            most_common = word_counts.most_common(self.max_features)
        else:
            most_common = word_counts.most_common()

        self.vocabulary_ = {word: i for i, (word, _) in enumerate(most_common)}
        self.feature_names_ = list(self.vocabulary_.keys())
        return self

    def transform(self, documents):
        X = np.zeros((len(documents), len(self.vocabulary_)))
        for i, doc in enumerate(documents):
            tokens = self._tokenize(doc)
            ngrams = self._get_ngrams(tokens)
            for ngram in ngrams:
                if ngram in self.vocabulary_:
                    X[i, self.vocabulary_[ngram]] += 1
        return X

    def fit_transform(self, documents):
        return self.fit(documents).transform(documents)

# Example usage
if __name__ == "__main__":
    documents = [
        "This is a sample document",
        "Another document with different words",
        "Sample document for testing"
    ]

    vectorizer = CountVectorizer(max_features=10)
    X = vectorizer.fit_transform(documents)
    print("Vocabulary:", vectorizer.vocabulary_)
    print("Feature matrix shape:", X.shape)
    print("Feature matrix:\n", X)
