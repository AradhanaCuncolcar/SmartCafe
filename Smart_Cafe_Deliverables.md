# ☕ The Smart Café — Deliverables

All numbers below come from actually running `cafe.py` and `agent.py` (see `run_results.py`).
Averages are over hundreds/thousands of simulated months. Exact values shift by a few AED between runs because demand is random.

---

## Part 1 — MDP framing

| RL term | In the café, this is… |
|---|---|
| **State** | The stock on hand when the shop opens (integer 0–15), i.e. leftover carried over from yesterday. |
| **Action** | How many units to order today (0–15). |
| **Reward** | The day's profit = sales revenue − order cost − holding cost − spoilage cost − stockout penalty. |
| **Episode** | One full 30-day month (starts again with 5 units in stock). |
| **Policy** | The rule "stock level → order amount" that the agent learns. |

---

## Part 2 — Simulator hand test and Reflection P2

Hand test, `for order in [8, 8, 2, 10]`:

| Order | Demand | Sold | Spoiled | Stockout | Carry | Profit |
|---|---|---|---|---|---|---|
| 8 | 5 | 5 | 2 | 0 | 6 | 11.6 |
| 8 | 11 | 11 | 1 | 0 | 2 | 44.6 |
| 2 | 5 | 4 | 0 | 1 | 0 | 12.0 |
| 10 | 8 | 8 | 0 | 0 | 2 | 29.4 |

Check of day 1 by hand: available = 5+8 = 13, sold = 5, leftover = 8, spoiled = round(0.25×8) = 2, carry = 6.
Profit = 5×5 − (8×1 + 8×0.3 + 2×1.5 + 0) = 25 − 13.4 = **11.6** ✔

**P2: Over a 30-day month:**
- **Order 15 every day:** profit ≈ AED 499, about 55 units spoiled, 0 stockouts. The fridge is always full, so nobody is turned away, but money is wasted on spoilage, holding costs and over-buying. **Failure mode 1 is waste.**
- **Order 1 every day:** profit ≈ **−AED 1,187**, about 222 unhappy customers (stockouts), 0 spoilage. The shop sells almost nothing and pays AED 6 for every customer it can't serve. **Failure mode 2 is stockout.**

Over-ordering loses a bit of margin. Under-ordering loses the sale and adds a penalty.

---

## Part 3 — Naive managers and Reflection P3

| Manager | Avg monthly profit (AED) |
|---|---|
| Cautious (always 4) | **≈ −200** (loses money) |
| Random (0–15) | ≈ 350 |
| Generous (always 10) | **≈ 713** (best naive) |

**P3:** The Generous manager is best. The Cautious manager **loses money**.
Under-ordering is worse here because every missed customer costs the AED 5 sale you didn't make plus the AED 6 penalty, about AED 11 per customer. An over-ordered unit costs at most about AED 1 (purchase) + 0.3 (holding) + a share of 1.5 (spoilage), and only 25% of leftovers spoil. So a missed customer costs several times more than a spare unit, and "play safe by ordering little" is the wrong kind of safe.

---

## Part 4 — Q-learning agent and Reflection P4

Training (15,000 months, α=0.15, γ=0.95, ε 1.0→0.05):

- Learning curve: about **AED 355/month at the start → about AED 707/month at the end** (while still exploring 5% of the time).
- Learned policy (stock 0…15 → order):
  `[12, 12, 13, 8, 10, 8, 7, 4, 3, 9, 9, 1, 0, 0, 0, 0]`
- **Trained agent's average monthly profit: ≈ AED 740** (over 2,000 test months; across several training runs it lands between about 727 and 750).

**P4:**
- **alpha (learning rate):** how far each new experience moves the Q-value. High values learn fast but are noisy; low values are stable but slow.
- **gamma (discount factor):** how much the agent cares about future profit versus today's. High gamma makes it plan ahead, e.g. understand that today's leftover is tomorrow's stock.
- **eps (exploration rate):** the chance of trying a random order instead of the best-known one.
- **Why eps starts high and shrinks:** at the start the Q-table knows nothing, so it must explore to discover which orders work. Once it has learned something, it should mostly exploit that knowledge to earn profit, while keeping a small amount of exploration (5%) so it can still find improvements.

---

## Part 5 — Dashboard and Reflection P5

Run `streamlit run app.py` and press **Train the AI**. Screenshots to take from your own run: the Learning tab, the Strategy tab and the KPI cards. Reference charts from my run are in `learning_curve.png`, `policy.png` and `scoreboard.png`.

**P5:** The AI discovered an **"order up to a target level"** strategy. When stock is low (0–3 units) it orders about 12–13 units, filling the fridge to about 12–15. As stock rises it orders less (4–8 units at 5–7 stock), and when stock is 11 or more it orders **nothing**. So yes, "order more when stock is low, less when it's high" appears.
Nobody programmed it. In each state the agent tried many order sizes and saw the resulting profit. Q-learning updates Q(state, action) toward `reward + γ·best future value`, so orders that tended to give high profit (and good stock tomorrow) had their Q-values pushed up, and bad orders (stockout-prone or wasteful) were pushed down. After many months, `argmax` over each row of the Q-table gives this pattern.
Note: a few middle states (e.g. stock 9 or 10) look odd because they are rarely visited, so their values are less reliable. That is normal for tabular Q-learning.

---

## Part 6 — You vs AI, scoreboard and Reflection P6

Scoreboard (avg monthly profit):

| Strategy | AED |
|---|---|
| 🧠 RL agent | **≈ 740** |
| Generous | ≈ 713 |
| Random | ≈ 350 |
| Cautious | ≈ −200 |

For "You vs AI", play the 10-day game in the dashboard and screenshot your result. Your result is personal and isn't something I can generate.

**P6 (template to fill in with your own result, using typical findings):** The AI should beat most human players over 10 days, though with only 10 days luck matters a lot, and you can win. The usual human mistakes are ordering a flat "safe" number, forgetting that carried-over stock reduces how much you need to order, and under-ordering after a wasteful day. The AI avoids these because it orders relative to current stock. The cautious strategy is the surprise: it is not safe at all, it loses money.

---

## Part 7 — Experiments and Reflection P7

Each row is a retrain (15,000 months) with one setting changed. Policies are listed for stock 0…15.

| Change | Learned policy | Avg profit |
|---|---|---|
| Baseline | `[12,15,10,13,10,8,13,3,9,2,2,11,0,0,0,0]` | ≈ 725 |
| Spoilage cost 1.5 → **5.0** | `[13,15,10,8,10,6,5,5,4,1,1,9,0,0,0,0]` | ≈ 616 |
| Stockout penalty 6 → **12** | `[13,13,13,14,13,10,8,10,4,1,8,9,0,0,0,0]` | ≈ 692 |
| Stockout penalty 6 → **0** | `[13,5,9,7,6,3,2,4,2,1,1,11,0,0,0,0]` | ≈ 784* |
| Mean demand 7 → **12** | `[15,14,13,14,12,10,10,12,7,3,9,0,0,0,0,0]` | ≈ 1204 |

*Not comparable: with no penalty, the cost of a stockout is simply removed, so profit looks higher.

**P7 (stockout penalty 6 → 12):** The AI orders more at low stock (13–14 units at stock 0–4, versus a more mixed 10–15 in the baseline) and keeps more safety stock. With a penalty of AED 12, each missed customer is far more expensive than a unit of leftover, so the profit-maximising move is to hold a bigger buffer even though more spoils.
The opposite check supports this: with penalty 0, the agent orders much less (e.g. 5–9 at stock 1–4), because running out no longer costs anything beyond the lost sale.
**Big lesson:** the AI optimises exactly what you reward, so the cost numbers you pick quietly decide its behaviour. (The individual numbers in the middle states are noisy, but the overall direction is consistent.)

---

## Recommendation to the café owner

Use the learned **order-up-to strategy**: top the fridge up to about 12–13 units each morning (order roughly 12–13 minus what's on hand, and order nothing when you already have 11 or more). In simulation this earns about **AED 740 per month, roughly 4% more than the best fixed guess (always order 10, about 713), about twice a random manager (≈350), and about AED 940 more than the cautious manager, who actually loses money**. If costs change (e.g. spoilage becomes pricier or stockouts hurt more), retrain the agent on the new numbers and it will shift its safety stock accordingly, which is something a fixed rule can't do.
