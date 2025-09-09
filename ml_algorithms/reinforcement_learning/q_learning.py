"""
Q-Learning

Core Idea: A value-based method that learns a table (Q-table) of action-values.
The agent learns the optimal policy by updating Q-values based on rewards received.

Best For: Discrete state and action spaces, environments with clear rewards.

Pros:
- Simple to implement
- Guaranteed convergence under certain conditions
- Off-policy learning

Cons:
- Requires discretization of continuous spaces
- Can be slow for large state spaces
- Exploration vs exploitation trade-off

Key Hyperparameters:
- alpha: Learning rate
- gamma: Discount factor
- epsilon: Exploration rate
- episodes: Number of training episodes
"""

import numpy as np
import matplotlib.pyplot as plt

class QLearning:
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.9, epsilon=0.1, episodes=1000):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.episodes = episodes
        self.Q = np.zeros((n_states, n_actions))

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            return np.argmax(self.Q[state])

    def update(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.Q[next_state])
        td_target = reward + self.gamma * self.Q[next_state, best_next_action]
        td_error = td_target - self.Q[state, action]
        self.Q[state, action] += self.alpha * td_error

    def train(self, env):
        rewards = []
        for episode in range(self.episodes):
            state = env.reset()
            total_reward = 0
            done = False
            while not done:
                action = self.choose_action(state)
                next_state, reward, done = env.step(action)
                self.update(state, action, reward, next_state)
                state = next_state
                total_reward += reward
            rewards.append(total_reward)
        return rewards

    def get_policy(self):
        return np.argmax(self.Q, axis=1)

# Simple Grid World Environment
class GridWorld:
    def __init__(self, size=4):
        self.size = size
        self.n_states = size * size
        self.n_actions = 4  # up, down, left, right
        self.start = 0
        self.goal = size * size - 1
        self.reset()

    def reset(self):
        self.state = self.start
        return self.state

    def step(self, action):
        row, col = divmod(self.state, self.size)
        if action == 0:  # up
            row = max(0, row - 1)
        elif action == 1:  # down
            row = min(self.size - 1, row + 1)
        elif action == 2:  # left
            col = max(0, col - 1)
        elif action == 3:  # right
            col = min(self.size - 1, col + 1)
        next_state = row * self.size + col
        reward = 1 if next_state == self.goal else -0.1
        done = next_state == self.goal
        self.state = next_state
        return next_state, reward, done

# Example usage
if __name__ == "__main__":
    env = GridWorld(size=4)
    ql = QLearning(n_states=16, n_actions=4, episodes=1000)
    rewards = ql.train(env)

    # Plot learning curve
    plt.plot(rewards)
    plt.title('Q-Learning Learning Curve')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.show()

    # Print learned policy
    policy = ql.get_policy()
    print("Learned Policy (0:up, 1:down, 2:left, 3:right):")
    for i in range(4):
        print(policy[i*4:(i+1)*4])
