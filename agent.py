import numpy as np
import random


def train(env, episodes=15000, alpha=0.15, gamma=0.95,
          eps_start=1.0, eps_min=0.05, eps_decay=0.9998):
    n_states = env.max_stock + 1      # stock can be 0..max_stock
    n_actions = env.max_stock + 1     # you can order 0..max_stock
    Q = np.zeros((n_states, n_actions))
    eps = eps_start
    reward_curve = []

    for episode in range(episodes):
        state = env.reset()
        done = False
        total = 0
        while not done:
            # epsilon-greedy
            if random.random() < eps:
                action = random.randint(0, n_actions - 1)      # explore
            else:
                action = int(np.argmax(Q[state]))              # exploit

            reward, next_state, done, _ = env.step(action)

            # Q-learning update
            best_next = np.max(Q[next_state])
            Q[state, action] += alpha * (reward + gamma * best_next - Q[state, action])

            state = next_state
            total += reward
        reward_curve.append(total)
        eps = max(eps_min, eps * eps_decay)
    return Q, reward_curve


def policy_from_Q(Q):
    return np.argmax(Q, axis=1)
