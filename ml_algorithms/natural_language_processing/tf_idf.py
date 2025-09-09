"""
TF-IDF

Core Idea: BoW but weights words by how unique they are to a document in a corpus.
Term Frequency-Inverse Document Frequency highlights important words.

Best For: Information retrieval, text mining, document similarity.

Pros:
- Reduces impact of common words
- Highlights unique terms
- Better than BoW for many tasks

Cons:
- Still ignores word order
- Can be affected by document length
- Requires careful preprocessing

Key Hyperparameters:
- max_features: Maximum number of features
- smooth_idf: Add 1 to document frequencies
- sublinear_tf: Apply log to term frequencies
"""

import numpy as np
from collections import Counter
import re

class TfidfVectorizer:
    def __init__(self, max_features=None, smooth_idf=True, sublinear_tf=False):
        self.max_features = max_features
        self.smooth_idf = smooth_idf
        self.sublinear_tf = sublinear_tf
        self.vocabulary_ = {}
        self.idf_ = None

    def _tokenize(self, text):
        text = re.sub(r'[^\w\s]', '', text.lower())
        return text.split()

    def fit(self, documents):
        word_counts = Counter()
        doc_freq = Counter()
        for doc in documents:
            tokens = self._tokenize(doc)
            unique_tokens = set(tokens)
            word_counts.update(tokens)
            doc_freq.update(unique_tokens)

        if self.max_features:
            most_common = word_counts.most_common(self.max_features)
        else:
            most_common = word_counts.most_common()

        self.vocabulary_ = {word: i for i, (word, _) in enumerate(most_common)}
        n_docs = len(documents)
        self.idf_ = np.zeros(len(self.vocabulary_))
        for word, idx in self.vocabulary_.items():
            df = doc_freq[word]
            if self.smooth_idf:
                df += 1
                n_docs += 1
            self.idf_[idx] = np.log(n_docs / df)
        return self

    def transform(self, documents):
        X = np.zeros((len(documents), len(self.vocabulary_)))
        for i, doc in enumerate(documents):
            tokens = self._tokenize(doc)
            term_freq = Counter(tokens)
            for word, freq in term_freq.items():
                if word in self.vocabulary_:
                    tf = freq
                    if self.sublinear_tf:
                        tf = 1 + np.log(tf)
                    idx = self.vocabulary_[word]
                    X[i, idx] = tf * self.idf_[idx]
        return X

    def fit_transform(self, documents):
        return self.fit(documents).transform(documents)

# Example usage
if __name__ == "__main__":
    documents = [
        "This is a sample document",
        "Another document with different words",
        "Sample document for testing TF-IDF"
    ]

    vectorizer = TfidfVectorizer(max_features=10)
    X = vectorizer.fit_transform(documents)
    print("Vocabulary:", vectorizer.vocabulary_)
    print("IDF values:", vectorizer.idf_)
    print("TF-IDF matrix shape:", X.shape)
    print("TF-IDF matrix:\n", X)
