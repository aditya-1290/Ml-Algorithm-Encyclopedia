"""
Policy Gradients

Core Idea: Directly learns the policy function that maps states to actions by optimizing expected reward.
Uses gradient ascent on policy parameters.

Best For: Continuous action spaces, complex policies.

Pros:
- Can learn stochastic policies
- Suitable for continuous and high-dimensional action spaces
- Model-free

Cons:
- High variance in gradient estimates
- Requires careful tuning
- Sample inefficient

Key Hyperparameters:
- learning_rate: Step size for gradient ascent
- gamma: Discount factor
- episodes: Number of training episodes
"""

import numpy as np

from ml_algorithms.reinforcement_learning.q_learning import GridWorld

class PolicyGradientAgent:
    def __init__(self, state_size, action_size, learning_rate=0.01, gamma=0.99):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.theta = np.random.randn(state_size, action_size) * 0.01

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=1, keepdims=True)

    def get_action(self, state):
        state = state.reshape(1, -1)
        probs = self.softmax(state @ self.theta)
        action = np.random.choice(self.action_size, p=probs.ravel())
        return action, probs

    def discount_rewards(self, rewards):
        discounted = np.zeros_like(rewards, dtype=np.float32)
        running_add = 0
        for t in reversed(range(len(rewards))):
            running_add = running_add * self.gamma + rewards[t]
            discounted[t] = running_add
        return discounted

    def train(self, env, episodes=1000):
        for episode in range(episodes):
            states, actions, rewards = [], [], []
            state = env.reset()
            done = False
            while not done:
                action, probs = self.get_action(state)
                next_state, reward, done = env.step(action)
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                state = next_state

            discounted_rewards = self.discount_rewards(rewards)
            states = np.vstack(states)
            actions = np.array(actions)
            discounted_rewards = (discounted_rewards - np.mean(discounted_rewards)) / (np.std(discounted_rewards) + 1e-8)

            for i in range(len(states)):
                state = states[i].reshape(1, -1)
                probs = self.softmax(state @ self.theta)
                dsoftmax = probs.copy()
                dsoftmax[0, actions[i]] -= 1
                grad = state.T @ dsoftmax
                self.theta -= self.learning_rate * grad * discounted_rewards[i]

# Example usage with GridWorld from q_learning.py
if __name__ == "__main__":
    env = GridWorld(size=4)
    agent = PolicyGradientAgent(state_size=16, action_size=4)
    agent.train(env, episodes=500)
    print("Policy Gradient training complete.")
