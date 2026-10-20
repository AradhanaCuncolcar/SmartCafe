# ☕ The Smart Café — Build Guide
## Build the simulator, train the AI, run the dashboard (~3 hours)

> Read `01_Problem_Statement.md` first. This guide is where you build. You will create **two files** — `cafe.py` (the shop) and `agent.py` (the AI) — and then run the provided **`app.py`** dashboard, which imports your code and brings it to life.

**Setup:** `pip install numpy matplotlib streamlit`
You may use an **LLM as a coding copilot** — ask it to help you write and understand the tricky lines. But *you* make the decisions, and you must be able to explain every line.

---

## PART 1 — Frame the problem (no code, ~15 min)

Fill this in **in your own words** before coding. This is the single most important step — get it wrong and nothing else works.

| RL term | In the café, this is… | Your answer |
|---|---|---|
| **State** | the situation you see each morning | ___ |
| **Action** | the decision you make | ___ |
| **Reward** | the feedback you get | ___ |
| **Episode** | one full… | ___ |
| **Policy** | the rule you're trying to learn | ___ |

*(Hint: the state is something that changes day to day and that your decision should depend on.)*

---

## PART 2 — Build the Café Simulator (`cafe.py`) — THE BUSINESS 🏪

This is the part where you build something real: a working simulation of the shop. Implement the daily cycle from the problem statement.

**The business rules (the exact numbers to use):**
| Rule | Value |
|---|---|
| Max stock the fridge holds | 15 units |
| Selling price per unit | AED 5 |
| Cost to order one unit | AED 1 |
| Holding cost per leftover unit | AED 0.3 |
| Spoilage cost per spoiled unit | AED 1.5 |
| Stockout penalty per unhappy customer | AED 6 |
| Fraction of leftover stock that spoils overnight | 25% (the rest carries to tomorrow) |
| Typical daily demand | about 7, but ~20% of days are "busy" (extra customers) |
| Days per month (one episode) | 30 |

**The skeleton — fill in every `# TODO`:**
```python
import numpy as np
rng = np.random.default_rng()

class CafeSim:
    def __init__(self, price=5.0, order_cost=1.0, holding_cost=0.3,
                 spoilage_cost=1.5, stockout_penalty=6.0, spoil_frac=0.25,
                 mean_demand=7, busy_chance=0.20, max_stock=15, days=30):
        # store all the settings on self (so the dashboard can change them)
        self.price=price; self.order_cost=order_cost; self.holding_cost=holding_cost
        self.spoilage_cost=spoilage_cost; self.stockout_penalty=stockout_penalty
        self.spoil_frac=spoil_frac; self.mean_demand=mean_demand
        self.busy_chance=busy_chance; self.max_stock=max_stock; self.days=days

    def reset(self):
        self.stock = 5          # start each month with 5 units on hand
        self.day = 0
        return self.stock       # the STATE is the stock on hand

    def _demand(self):
        # random daily demand; sometimes a busy day adds extra customers
        d = rng.poisson(self.mean_demand)
        if rng.random() < self.busy_chance:
            d += rng.poisson(5)
        return int(d)

    def step(self, order):
        order = int(np.clip(order, 0, self.max_stock))
        # 1. stock available today = what you had + what you ordered (capped by fridge size)
        available = ___                                  # TODO
        # 2. today's demand
        demand = self._demand()
        # 3. you can only sell what you have
        sold = ___                                       # TODO  (hint: min of two things)
        # 4. customers you couldn't serve (stockout)
        stockout = ___                                   # TODO  (hint: demand beyond available, never negative)
        # 5. what's left over after selling
        leftover = available - sold
        # 6. part of the leftover spoils overnight; the rest carries to tomorrow
        spoiled = int(round(self.spoil_frac * leftover)) # given
        carry   = leftover - spoiled
        # 7. the day's PROFIT (the REWARD)
        revenue = sold * self.price
        profit  = revenue - ( ___ )                      # TODO: subtract order cost, holding, spoilage, stockout penalty
        # 8. move to tomorrow
        self.stock = carry
        self.day += 1
        done = self.day >= self.days
        info = dict(order=order, demand=demand, sold=sold, spoiled=spoiled,
                    stockout=stockout, carry=carry, profit=round(profit,2))
        return profit, self.stock, done, info
```

**Test it yourself** before moving on — order a few amounts by hand and check the numbers make sense:
```python
env = CafeSim(); s = env.reset()
for order in [8, 8, 2, 10]:
    r, s, done, info = env.step(order)
    print(info)
```

**Reflection P2:** Order a huge amount (say 15) every day for a few days — what happens to spoilage and profit? Now order 1 every day — what happens to stockouts? Describe the two failure modes you built.

---

## PART 3 — Feel the pain: naive managers (~15 min)

Before any AI, measure how **guessing** performs. Run each strategy over many months and record the **average monthly profit**:
- **Cautious manager:** always order 4.
- **Generous manager:** always order 10.
- **Random manager:** order a random amount each day.

```python
import random
def average_profit(env, policy, months=500):
    totals=[]
    for _ in range(months):
        s=env.reset(); done=False; total=0
        while not done:
            r,s,done,_=env.step(policy(s)); total+=r
        totals.append(total)
    return sum(totals)/len(totals)

env=CafeSim()
print("Cautious :", average_profit(env, lambda s: 4))
print("Generous :", average_profit(env, lambda s: 10))
print("Random   :", average_profit(env, lambda s: random.randint(0, 15)))
```

**Reflection P3:** Which naive manager is best? Is any of them *losing money*? Why would ordering too little actually be *worse* than ordering too much here?

---

## PART 4 — Build the Q-Learning Agent (`agent.py`) — THE BRAIN 🧠

Now the AI that **learns** the best order for each stock level. Fill in every `# TODO`.

```python
import numpy as np, random

def train(env, episodes=15000, alpha=0.15, gamma=0.95,
          eps_start=1.0, eps_min=0.05, eps_decay=0.9998):
    n_states  = env.max_stock + 1     # stock can be 0..max_stock
    n_actions = env.max_stock + 1     # you can order 0..max_stock
    Q = np.zeros((n_states, n_actions))   # the Q-table starts knowing NOTHING
    eps = eps_start
    reward_curve = []

    for episode in range(episodes):
        state = env.reset()
        done = False
        total = 0
        while not done:
            # --- choose an action (epsilon-greedy): mostly best-known, sometimes random ---
            if random.random() < eps:
                action = ___                      # TODO: explore (a random order 0..n_actions-1)
            else:
                action = ___                      # TODO: exploit (the best order for this state -> np.argmax)

            reward, next_state, done, _ = env.step(action)

            # --- the Q-learning update (the heart of it) ---
            best_next = np.max(Q[next_state])
            Q[state, action] += alpha * ( ___ )   # TODO: reward + gamma*best_next - Q[state,action]

            state = next_state
            total += reward
        reward_curve.append(total)
        eps = max(eps_min, eps * eps_decay)       # explore less as it learns
    return Q, reward_curve

def policy_from_Q(Q):
    # the learned strategy: for each stock level, the best order
    return np.argmax(Q, axis=1)
```

**Reflection P4:** In one sentence each — what do `alpha`, `gamma`, and `eps` control? Why does `eps` start high and shrink?

---

## PART 5 — Launch the Dashboard & watch it learn 📊

The provided **`app.py`** imports your `CafeSim` and `train`. Run it:
```bash
streamlit run app.py
```
Press **Train**, then explore:
- 📈 **Learning curve** — profit per month climbing. *This is your proof the agent learned.*
- 🗺️ **Policy heatmap** — the strategy it found (stock level → order amount).
- 📋 **Day-by-day ledger** — step through a month and watch the agent run the shop.
- 💳 **KPI cards** — total profit, spoilage %, stockout %, service level.

**Reflection P5:** Describe the strategy the AI discovered in plain English. Does "order more when stock is low, less when it's high" show up? Nobody programmed that — explain how it learned it.

---

## PART 6 — You vs the AI 🎮 + the Scoreboard 🏆

In the dashboard's **"You vs AI"** tab, **you** play café manager for 10 days — type your order each morning, watch your profit. Then the trained AI plays the **same** 10 days. Compare.

Then look at the **Scoreboard**: RL agent vs Cautious vs Generous vs Random vs **You**.

**Reflection P6:** Did the AI beat you? Where did your human guesses go wrong that the AI avoided? Did the "cautious" strategy surprise you?

---

## PART 7 — Experiments: what you reward is what you get 🎚️

Use the dashboard **sliders** to change the business, **retrain**, and watch the strategy shift:
1. **Make spoilage much more expensive** → does the AI start ordering *less*? Why?
2. **Make the stockout penalty bigger** → does it start ordering *more* (keeping a safety buffer)? Why?
3. **Make customers busier (higher demand)** → how does the policy move?

**Reflection P7:** Pick one change and explain how the AI's *strategy* changed and *why*. This is the big lesson: the AI optimises exactly what you reward — so the rewards you choose quietly decide its behaviour.

---

## Deliverables
1. `cafe.py` and `agent.py` — your working simulator and agent.
2. Your notebook/script output: the naive baselines, the trained agent's profit, the learning curve, the learned policy.
3. Screenshots from the dashboard: the learning curve, the policy heatmap, the scoreboard, and your "You vs AI" result.
4. Your **reflection answers** (P1–P7).
5. A 3–4 sentence **recommendation to the café owner**: which strategy to use, how much more it earns than guessing, and how it would adapt if costs changed.

## Rubric (50)
| # | Criterion | Marks |
|---|---|---|
| 1 | MDP framing correct (Part 1) | 5 |
| 2 | **Café simulator works** (demand, sales, stockout, spoilage, profit, carry-over all correct) | 12 |
| 3 | Naive baselines run and interpreted | 5 |
| 4 | **Q-learning agent works** (ε-greedy + update + loop correct; it actually learns) | 12 |
| 5 | Dashboard runs with your code; learning curve + policy shown | 6 |
| 6 | "You vs AI" + scoreboard completed | 3 |
| 7 | Experiments: policy shifts when a cost changes, explained | 4 |
| 8 | Reflections + owner recommendation (clear, correct, insightful) | 3 |

## Hints
- **Test the simulator by hand first** (Part 2) — if the day's numbers don't make sense to *you*, the AI can't learn from them.
- `sold = min(available, demand)` and `stockout = max(demand - available, 0)` are the two lines people get backwards.
- The Q-update is **one line** — don't overthink it; translate the formula your instructor showed.
- The learning curve is **noisy** and climbs *gradually* — that's normal, not a bug.
- Judge the agent on **average** profit over many months, never one lucky run.

> **Remember:** you never tell the AI the right order — not once. You only let it feel profit and loss, and it works out the strategy itself. That's reinforcement learning. 🚀
