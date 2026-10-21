"""agent.py - THE BRAIN: a tabular Q-learning agent for the Smart Café."""
import random
import numpy as np


def train(env, episodes=15000, alpha=0.15, gamma=0.95,
          eps_start=1.0, eps_min=0.05, eps_decay=0.9998, progress=None):
    n_stock = env.max_stock + 1
    n_actions = env.max_stock + 1
    # Q[stock, day_of_week, event_flag, action] - starts knowing NOTHING
    Q = np.zeros((n_stock, 7, 2, n_actions))
    eps = eps_start
    reward_curve = []

    for episode in range(episodes):
        state = env.reset()
        done = False
        total = 0.0
        while not done:
            stock, dow, evt = state
            if random.random() < eps:                       # explore
                action = random.randint(0, n_actions - 1)
            else:                                           # exploit
                action = int(np.argmax(Q[stock, dow, evt]))

            reward, next_state, done, _ = env.step(action)
            ns, nd, ne = next_state

            best_next = np.max(Q[ns, nd, ne])               # Q-learning update
            Q[stock, dow, evt, action] += alpha * (reward + gamma * best_next
                                                   - Q[stock, dow, evt, action])
            state = next_state
            total += reward
        reward_curve.append(total)
        eps = max(eps_min, eps * eps_decay)
        if progress and episode % 500 == 0:
            progress(episode / episodes)
    if progress:
        progress(1.0)
    return Q, reward_curve


def get_best_action(Q, stock, dow, event):
    return int(np.argmax(Q[stock, dow, event]))


def policy_from_Q(Q):
    """Learned strategy: array [stock, dow, event] -> best order."""
    return np.argmax(Q, axis=3)


def run_month(env, order_fn):
    """Play one 30-day month with any strategy order_fn(state) -> order. Returns (profit, stats)."""
    s = env.reset()
    done = False
    tot = demand = sold = spoiled = stockout = ordered = 0
    while not done:
        r, s2, done, info = env.step(order_fn(s))
        tot += r; demand += info["demand"]; sold += info["sold"]
        spoiled += info["spoiled"]; stockout += info["stockout"]; ordered += info["order"]
        s = s2
    return tot, dict(demand=demand, sold=sold, spoiled=spoiled, stockout=stockout, ordered=ordered)


def average_profit(env, order_fn, months=300):
    profits, stats = [], dict(demand=0, sold=0, spoiled=0, stockout=0, ordered=0)
    for _ in range(months):
        p, st = run_month(env, order_fn)
        profits.append(p)
        for k in stats:
            stats[k] += st[k]
    return float(np.mean(profits)), float(np.std(profits)), stats
