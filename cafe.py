"""cafe.py - THE BUSINESS: a simulator of the Smart Café.

State (what the manager sees each morning): (stock, day_of_week, event_today)
Action (the decision):                       how many units to order (0..max_stock)
Reward (the feedback):                       the day's profit
"""
import numpy as np

rng = np.random.default_rng()

DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


class CafeSim:
    def __init__(self, price=5.0, order_cost=1.0, holding_cost=0.3,
                 spoilage_cost=1.5, stockout_penalty=6.0, spoil_frac=0.25,
                 base_demand=7, max_stock=15, days=30,
                 weekend_boost=1.6, event_boost=2.0, event_chance=0.15):
        self.price = price
        self.order_cost = order_cost
        self.holding_cost = holding_cost
        self.spoilage_cost = spoilage_cost
        self.stockout_penalty = stockout_penalty
        self.spoil_frac = spoil_frac
        self.base_demand = base_demand
        self.max_stock = max_stock
        self.days = days
        # Real-world demand multipliers
        self.weekend_boost = weekend_boost   # Fri, Sat, Sun are busier
        self.event_boost = event_boost       # festivals, local events, high footfall
        self.event_chance = event_chance     # chance that any given day has an event
        self.reset()

    # ---------- helpers ----------
    def _draw_event(self):
        return 1 if rng.random() < self.event_chance else 0

    def _get_state(self):
        return (int(self.stock), int(self.day_of_week), int(self.event_today))

    def expected_demand(self, dow, event):
        """Average customers for a given day type (used for explanations)."""
        mean_d = self.base_demand
        if dow in (4, 5, 6):
            mean_d *= self.weekend_boost
        if event == 1:
            mean_d *= self.event_boost
        return mean_d

    def draw_demand(self, dow, event):
        """Random demand for a given day type."""
        return int(rng.poisson(self.expected_demand(dow, event)))

    def _demand(self):
        return self.draw_demand(self.day_of_week, self.event_today)

    # ---------- the core business rules (pure: does not change the sim) ----------
    def simulate_day(self, stock, order, demand):
        order = int(np.clip(order, 0, self.max_stock))
        available = min(stock + order, self.max_stock)      # fridge limit
        sold = min(available, demand)                        # can only sell what we have
        stockout = max(demand - available, 0)                # unhappy customers
        leftover = available - sold
        spoiled = int(round(self.spoil_frac * leftover))     # part spoils overnight
        carry = leftover - spoiled                           # rest is tomorrow's stock
        revenue = sold * self.price
        costs = (order * self.order_cost + leftover * self.holding_cost
                 + spoiled * self.spoilage_cost + stockout * self.stockout_penalty)
        profit = revenue - costs
        return dict(order=order, demand=int(demand), sold=int(sold), spoiled=int(spoiled),
                    stockout=int(stockout), carry=int(carry), profit=round(float(profit), 2))

    # ---------- RL interface ----------
    def reset(self):
        self.stock = 5
        self.day_of_week = int(rng.integers(0, 7))
        self.event_today = self._draw_event()
        self.day_count = 0
        return self._get_state()

    def step(self, order, demand=None):
        if demand is None:
            demand = self._demand()
        info = self.simulate_day(self.stock, order, demand)
        self.stock = info["carry"]
        self.day_of_week = (self.day_of_week + 1) % 7
        self.event_today = self._draw_event()
        self.day_count += 1
        done = self.day_count >= self.days
        info.update(dow=self.day_of_week, event=self.event_today)
        return info["profit"], self._get_state(), done, info
