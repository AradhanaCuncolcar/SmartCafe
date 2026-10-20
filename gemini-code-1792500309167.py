import numpy as np

rng = np.random.default_rng()

class CafeSim:
    def __init__(self, price=5.0, order_cost=1.0, holding_cost=0.3,
                 spoilage_cost=1.5, stockout_penalty=6.0, spoil_frac=0.25,
                 base_demand=7, max_stock=15, days=30):
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
        self.weekend_boost = 1.6  # Fridays, Saturdays, Sundays are busier
        self.event_boost = 2.0    # Shopping festivals, local events, etc.

    def reset(self):
        self.stock = 5          
        self.day_of_week = rng.integers(0, 7) # 0=Mon, 6=Sun
        self.event_today = rng.choice([0, 1], p=[0.85, 0.15]) # 15% chance of event
        self.day_count = 0
        return self._get_state()

    def _get_state(self):
        # State is now 3-dimensional: (Stock, Day of Week, Event Flag)
        return (int(self.stock), int(self.day_of_week), int(self.event_today))

    def _demand(self):
        mean_d = self.base_demand
        
        # Weekend traffic spike
        if self.day_of_week in [4, 5, 6]: # Fri, Sat, Sun
            mean_d *= self.weekend_boost
            
        # Event traffic spike
        if self.event_today == 1:
            mean_d *= self.event_boost
            
        return int(rng.poisson(mean_d))

    def step(self, order):
        order = int(np.clip(order, 0, self.max_stock))
        
        available = min(self.stock + order, self.max_stock)
        demand = self._demand()
        
        sold = min(available, demand)
        stockout = max(demand - available, 0)
        leftover = available - sold
        
        spoiled = int(round(self.spoil_frac * leftover))
        carry = leftover - spoiled
        
        revenue = sold * self.price
        costs = (order * self.order_cost) + (leftover * self.holding_cost) + \
                (spoiled * self.spoilage_cost) + (stockout * self.stockout_penalty)
        profit = revenue - costs
        
        # Advance environment
        self.stock = carry
        self.day_of_week = (self.day_of_week + 1) % 7
        self.event_today = rng.choice([0, 1], p=[0.85, 0.15])
        self.day_count += 1
        
        done = self.day_count >= self.days
        info = dict(order=order, demand=demand, sold=sold, spoiled=spoiled,
                    stockout=stockout, carry=carry, profit=round(profit, 2),
                    dow=self.day_of_week, event=self.event_today)
                    
        return profit, self._get_state(), done, info