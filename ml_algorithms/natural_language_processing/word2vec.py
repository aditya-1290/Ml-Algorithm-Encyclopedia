"""
Word2Vec

Core Idea: An algorithm that uses a neural network to learn word embeddings (vector representations).
Captures semantic relationships between words.

Best For: Semantic similarity, analogy tasks, as features for NLP tasks.

Pros:
- Captures word semantics
- Low-dimensional dense vectors
- Transferable to other tasks

Cons:
- Requires large corpora
- Computationally intensive
- Ignores polysemy

Key Hyperparameters:
- embedding_dim: Dimension of word vectors
- window_size: Context window size
- learning_rate: Learning rate
- epochs: Number of training epochs
"""

import numpy as np
from collections import Counter
import re

class Word2Vec:
    def __init__(self, embedding_dim=100, window_size=5, learning_rate=0.01, epochs=5, negative_samples=5):
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.negative_samples = negative_samples
        self.word2idx = {}
        self.idx2word = {}
        self.vocab_size = 0
        self.W_in = None
        self.W_out = None

    def _tokenize(self, text):
        text = re.sub(r'[^\w\s]', '', text.lower())
        return text.split()

    def _build_vocab(self, corpus):
        word_counts = Counter()
        for doc in corpus:
            tokens = self._tokenize(doc)
            word_counts.update(tokens)

        self.word2idx = {word: i for i, (word, _) in enumerate(word_counts.most_common())}
        self.idx2word = {i: word for word, i in self.word2idx.items()}
        self.vocab_size = len(self.word2idx)

    def _generate_training_data(self, corpus):
        training_data = []
        for doc in corpus:
            tokens = self._tokenize(doc)
            for i, target in enumerate(tokens):
                if target not in self.word2idx:
                    continue
                target_idx = self.word2idx[target]
                start = max(0, i - self.window_size)
                end = min(len(tokens), i + self.window_size + 1)
                for j in range(start, end):
                    if i != j:
                        context = tokens[j]
                        if context in self.word2idx:
                            context_idx = self.word2idx[context]
                            training_data.append((target_idx, context_idx))
        return training_data

    def _negative_sampling(self, target_idx):
        negatives = []
        while len(negatives) < self.negative_samples:
            neg = np.random.randint(0, self.vocab_size)
            if neg != target_idx:
                negatives.append(neg)
        return negatives

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def fit(self, corpus):
        self._build_vocab(corpus)
        training_data = self._generate_training_data(corpus)

        self.W_in = np.random.randn(self.vocab_size, self.embedding_dim) * 0.01
        self.W_out = np.random.randn(self.vocab_size, self.embedding_dim) * 0.01

        for epoch in range(self.epochs):
            np.random.shuffle(training_data)
            for target_idx, context_idx in training_data:
                # Positive sample
                v_target = self.W_in[target_idx]
                v_context = self.W_out[context_idx]
                score = self._sigmoid(np.dot(v_target, v_context))
                grad = self.learning_rate * (1 - score)

                # Update
                self.W_in[target_idx] += grad * v_context
                self.W_out[context_idx] += grad * v_target

                # Negative samples
                negatives = self._negative_sampling(target_idx)
                for neg_idx in negatives:
                    v_neg = self.W_out[neg_idx]
                    score_neg = self._sigmoid(np.dot(v_target, v_neg))
                    grad_neg = self.learning_rate * (0 - score_neg)

                    self.W_in[target_idx] += grad_neg * v_neg
                    self.W_out[neg_idx] += grad_neg * v_target

    def get_embedding(self, word):
        if word in self.word2idx:
            return self.W_in[self.word2idx[word]]
        else:
            return None

    def most_similar(self, word, top_n=5):
        if word not in self.word2idx:
            return []
        target_vec = self.get_embedding(word)
        similarities = {}
        for w, idx in self.word2idx.items():
            if w != word:
                vec = self.W_in[idx]
                sim = np.dot(target_vec, vec) / (np.linalg.norm(target_vec) * np.linalg.norm(vec))
                similarities[w] = sim
        sorted_words = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        return sorted_words[:top_n]

# Example usage
if __name__ == "__main__":
    corpus = [
        "the quick brown fox jumps over the lazy dog",
        "the cat sat on the mat",
        "dogs are loyal pets",
        "cats are independent animals"
    ]

    w2v = Word2Vec(embedding_dim=50, epochs=10)
    w2v.fit(corpus)

    print("Embedding for 'dog':", w2v.get_embedding('dog'))
    print("Most similar to 'dog':", w2v.most_similar('dog'))
