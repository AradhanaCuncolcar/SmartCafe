import numpy as np
import random

def train(env, episodes=15000, alpha=0.15, gamma=0.95, eps_start=1.0, eps_min=0.05, eps_decay=0.9998):
    # State dimensions
    n_stock = env.max_stock + 1     
    n_dow = 7
    n_events = 2
    n_actions = env.max_stock + 1     
    
    # 4D Q-Table: [Stock, Day_of_Week, Event_Flag, Action]
    Q = np.zeros((n_stock, n_dow, n_events, n_actions)) 
    eps = eps_start
    reward_curve = []

    for episode in range(episodes):
        state = env.reset()
        done = False
        total = 0
        
        while not done:
            stock, dow, evt = state
            
            # Epsilon-greedy exploration
            if random.random() < eps:
                action = random.randint(0, n_actions - 1)
            else:
                action = np.argmax(Q[stock, dow, evt])

            reward, next_state, done, _ = env.step(action)
            next_stock, next_dow, next_evt = next_state

            # Q-learning update
            best_next = np.max(Q[next_stock, next_dow, next_evt])
            Q[stock, dow, evt, action] += alpha * (reward + gamma * best_next - Q[stock, dow, evt, action])

            state = next_state
            total += reward
            
        reward_curve.append(total)
        eps = max(eps_min, eps * eps_decay)
        
    return Q, reward_curve

def get_best_action(Q, stock, dow, event):
    # Retrieve the best action for a specific multidimensional state
    return int(np.argmax(Q[stock, dow, event]))