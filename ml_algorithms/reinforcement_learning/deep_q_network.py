"""
Deep Q-Network (DQN)

Core Idea: Uses a neural network to approximate the Q-table, enabling complex environments.
Incorporates experience replay and target network for stability.

Best For: Environments with large state spaces, continuous or high-dimensional states.

Pros:
- Handles large state spaces
- Learns complex policies
- Stable learning with replay

Cons:
- Computationally expensive
- Requires careful hyperparameter tuning
- Can be unstable without proper techniques

Key Hyperparameters:
- learning_rate: Learning rate for optimizer
- gamma: Discount factor
- epsilon: Exploration rate
- batch_size: Experience replay batch size
- target_update_freq: Frequency to update target network
"""

import numpy as np
import matplotlib.pyplot as plt

from ml_algorithms.reinforcement_learning.q_learning import GridWorld

class SimpleNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))

    def forward(self, X):
        self.Z1 = X @ self.W1 + self.b1
        self.A1 = np.maximum(0, self.Z1)  # ReLU
        self.Z2 = self.A1 @ self.W2 + self.b2
        return self.Z2

    def backward(self, X, y, learning_rate):
        m = X.shape[0]
        dZ2 = self.Z2 - y
        dW2 = (self.A1.T @ dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m
        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * (self.Z1 > 0)
        dW1 = (X.T @ dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1

class DQN:
    def __init__(self, state_size, action_size, hidden_size=64, learning_rate=0.001, gamma=0.99,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, batch_size=32,
                 memory_size=10000, target_update_freq=100):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.memory = []
        self.memory_size = memory_size
        self.target_update_freq = target_update_freq

        self.model = SimpleNeuralNetwork(state_size, hidden_size, action_size)
        self.target_model = SimpleNeuralNetwork(state_size, hidden_size, action_size)
        self.update_target_model()

    def update_target_model(self):
        self.target_model.W1 = self.model.W1.copy()
        self.target_model.b1 = self.model.b1.copy()
        self.target_model.W2 = self.model.W2.copy()
        self.target_model.b2 = self.model.b2.copy()

    def remember(self, state, action, reward, next_state, done):
        if len(self.memory) > self.memory_size:
            self.memory.pop(0)
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_size)
        q_values = self.model.forward(state.reshape(1, -1))
        return np.argmax(q_values)

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        batch = np.random.choice(len(self.memory), self.batch_size, replace=False)
        states = []
        targets = []
        for i in batch:
            state, action, reward, next_state, done = self.memory[i]
            target = reward
            if not done:
                next_q = self.target_model.forward(next_state.reshape(1, -1))
                target += self.gamma * np.max(next_q)
            target_q = self.model.forward(state.reshape(1, -1))
            target_q[0, action] = target
            states.append(state)
            targets.append(target_q[0])
        states = np.array(states)
        targets = np.array(targets)
        self.model.backward(states, targets, self.learning_rate)

    def train(self, env, episodes):
        rewards = []
        for episode in range(episodes):
            state = env.reset()
            total_reward = 0
            done = False
            while not done:
                action = self.choose_action(state)
                next_state, reward, done = env.step(action)
                self.remember(state, action, reward, next_state, done)
                self.replay()
                state = next_state
                total_reward += reward
            rewards.append(total_reward)
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            if episode % self.target_update_freq == 0:
                self.update_target_model()
        return rewards

# Use the same GridWorld from q_learning.py
# Example usage similar

if __name__ == "__main__":
    env = GridWorld(size=4)
    dqn = DQN(state_size=16, action_size=4)  # Assuming state is one-hot encoded or flattened
    # For simplicity, assume state is integer, need to encode
    # This is a simplified version; in practice, state should be properly encoded
    print("DQN implementation ready. Requires state encoding for full functionality.")
