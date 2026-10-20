# app.py — The Smart Café Dashboard (All-in-One)
# ------------------------------------------------------------
# Run:  streamlit run app.py
# ------------------------------------------------------------
import numpy as np
import random
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="The Smart Café", page_icon="☕", layout="wide")

# --- Custom Café UI/UX Theme ---
st.markdown("""
    <style>
    /* Warm Café Color Palette */
    .stApp {
        background-color: #FAF4E8; /* Creamy milk background */
    }
    h1, h2, h3, h4, h5, h6 {
        color: #3E2723 !important; /* Deep espresso */
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    p, span, label, div {
        color: #4E342E; /* Mocha text */
    }
    div.stButton > button:first-child {
        background-color: #D7CCC8; 
        color: #3E2723;
        border: 2px solid #8D6E63;
        border-radius: 8px;
        font-weight: bold;
        transition: all 0.3s;
    }
    div.stButton > button:first-child:hover {
        background-color: #8D6E63;
        color: white;
    }
    .css-1d391kg {
        background-color: #EFEBE9; /* Sidebar subtle grey/brown */
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px 4px 0px 0px;
        gap: 10px;
        padding-top: 10px;
        padding-bottom: 10px;
        color: #5D4037;
        font-weight: bold;
    }
    .stTabs [aria-selected="true"] {
        border-bottom: 3px solid #6D4C41;
        color: #3E2723;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# PART 1: THE SIMULATOR (CafeSim)
# ==========================================
rng = np.random.default_rng()

class CafeSim:
    def __init__(self, price=5.0, order_cost=1.0, holding_cost=0.3,
                 spoilage_cost=1.5, stockout_penalty=6.0, spoil_frac=0.25,
                 mean_demand=7, busy_chance=0.20, max_stock=15, days=30):
        self.price = price
        self.order_cost = order_cost
        self.holding_cost = holding_cost
        self.spoilage_cost = spoilage_cost
        self.stockout_penalty = stockout_penalty
        self.spoil_frac = spoil_frac
        self.mean_demand = mean_demand
        self.busy_chance = busy_chance
        self.max_stock = max_stock
        self.days = days

    def reset(self):
        self.stock = 5          
        self.day = 0
        return self.stock       

    def _demand(self):
        d = rng.poisson(self.mean_demand)
        if rng.random() < self.busy_chance:
            d += rng.poisson(5)
        return int(d)

    def step(self, order):
        order = int(np.clip(order, 0, self.max_stock))
        
        # 1. available today = what you had + what you ordered
        available = min(self.stock + order, self.max_stock)
        
        # 2. today's demand
        demand = self._demand()
        
        # 3. sell what you have
        sold = min(available, demand)
        
        # 4. unmet demand
        stockout = max(demand - available, 0)
        
        # 5. leftovers
        leftover = available - sold
        
        # 6. spoilage & carry over
        spoiled = int(round(self.spoil_frac * leftover))
        carry = leftover - spoiled
        
        # 7. daily profit
        revenue = sold * self.price
        costs = (order * self.order_cost) + (leftover * self.holding_cost) + (spoiled * self.spoilage_cost) + (stockout * self.stockout_penalty)
        profit = revenue - costs
        
        # 8. advance day
        self.stock = carry
        self.day += 1
        done = self.day >= self.days
        info = dict(order=order, demand=demand, sold=sold, spoiled=spoiled,
                    stockout=stockout, carry=carry, profit=round(profit, 2))
        return profit, self.stock, done, info

# ==========================================
# PART 2: THE AI AGENT (Q-Learning)
# ==========================================
def train(env, episodes=15000, alpha=0.15, gamma=0.95, eps_start=1.0, eps_min=0.05, eps_decay=0.9998):
    n_states = env.max_stock + 1     
    n_actions = env.max_stock + 1     
    Q = np.zeros((n_states, n_actions)) 
    eps = eps_start
    reward_curve = []

    for episode in range(episodes):
        state = env.reset()
        done = False
        total = 0
        while not done:
            # Epsilon-greedy action selection
            if random.random() < eps:
                action = random.randint(0, n_actions - 1)  # Explore
            else:
                action = np.argmax(Q[state])               # Exploit

            reward, next_state, done, _ = env.step(action)

            # Q-learning update rule
            best_next = np.max(Q[next_state])
            Q[state, action] += alpha * (reward + gamma * best_next - Q[state, action])

            state = next_state
            total += reward
            
        reward_curve.append(total)
        eps = max(eps_min, eps * eps_decay)
    return Q, reward_curve

def policy_from_Q(Q):
    return np.argmax(Q, axis=1)

# ==========================================
# PART 3: DASHBOARD HELPERS
# ==========================================
def eval_policy(env, order_fn, months=400):
    profits=[]; demand=0; sold=0; spoiled=0; stockout=0; orders=0
    for _ in range(months):
        s=env.reset(); done=False; tot=0
        while not done:
            r,s,done,info=env.step(order_fn(s)); tot+=r
            demand+=info['demand']; sold+=info['sold']; spoiled+=info['spoiled']
            stockout+=info['stockout']; orders+=info['order']
        profits.append(tot)
    service = sold/max(demand,1)
    spoil_rate = spoiled/max(orders,1)
    stockout_rate = stockout/max(demand,1)
    return np.mean(profits), service, spoil_rate, stockout_rate

def simulate_day(p, stock, order, demand):
    order=int(np.clip(order,0,p['max_stock']))
    available=min(stock+order, p['max_stock'])
    sold=min(available, demand); stockout=max(demand-available,0); leftover=available-sold
    spoiled=int(round(p['spoil_frac']*leftover)); carry=leftover-spoiled
    profit=sold*p['price']-(order*p['order_cost']+leftover*p['holding_cost']
                            +spoiled*p['spoilage_cost']+stockout*p['stockout_penalty'])
    return round(profit,2), carry, dict(start=stock,order=order,demand=demand,sold=sold,
                                        spoiled=spoiled,stockout=stockout,carry=carry,profit=round(profit,2))

# ==========================================
# PART 4: STREAMLIT UI
# ==========================================
st.title("☕ The Smart Café")
st.markdown("### *Can an AI learn to run the shop without being told the answers?*")

# --- Sidebar Configuration ---
st.sidebar.header("🏪 Business Settings")
spoilage_cost   = st.sidebar.slider("Spoilage cost / unit (AED)", 0.0, 5.0, 1.5, 0.1)
stockout_pen    = st.sidebar.slider("Stockout penalty / customer (AED)", 0.0, 12.0, 6.0, 0.5)
mean_demand     = st.sidebar.slider("Typical daily demand", 3, 12, 7, 1)
spoil_frac      = st.sidebar.slider("Fraction that spoils overnight", 0.0, 1.0, 0.25, 0.05)

st.sidebar.header("🧠 AI Settings")
episodes        = st.sidebar.select_slider("Training months (episodes)", [2000,5000,8000,12000,15000,20000], value=12000)
gamma           = st.sidebar.slider("γ (Future Valuation)", 0.0, 0.99, 0.95, 0.01)

def make_env():
    return CafeSim(spoilage_cost=spoilage_cost, stockout_penalty=stockout_pen,
                   mean_demand=mean_demand, spoil_frac=spoil_frac)

params = dict(price=5.0, order_cost=1.0, holding_cost=0.3, spoilage_cost=spoilage_cost,
              stockout_penalty=stockout_pen, spoil_frac=spoil_frac, max_stock=make_env().max_stock)

# --- AI Training Trigger ---
if st.sidebar.button("🚀 Train the AI", type="primary"):
    with st.spinner("The AI is practising over thousands of months... Pouring coffee..."):
        env=make_env()
        Q, curve = train(env, episodes=episodes, gamma=gamma)
        st.session_state['Q']=Q
        st.session_state['curve']=curve
        st.session_state['params']=params
    st.sidebar.success("Training complete!")

if 'Q' not in st.session_state:
    st.info("👈 Set your business costs on the left, then press **Train the AI** to open the dashboard.")
    st.stop()

# --- Load State ---
Q=st.session_state['Q']
curve=st.session_state['curve']
policy=policy_from_Q(Q)
MAX=make_env().max_stock
ai_order = lambda s: int(policy[int(np.clip(s, 0, MAX))])
env=make_env()

# --- KPI Metrics ---
st.markdown("---")
rl_profit, service, spoil_rate, stockout_rate = eval_policy(env, ai_order)
c1, c2, c3, c4 = st.columns(4)
c1.metric("💰 AI Avg Monthly Profit", f"AED {rl_profit:,.0f}")
c2.metric("✅ Service Level", f"{service*100:.0f}%")
c3.metric("🗑️ Spoilage Rate", f"{spoil_rate*100:.0f}%")
c4.metric("🚫 Stockout Rate", f"{stockout_rate*100:.0f}%")
st.markdown("---")

# --- Tabs ---
tab1, tab2, tab3 = st.tabs(["📈 AI Insights & Strategy", "🏆 Scoreboard", "🎮 You vs AI"])

with tab1:
    st.subheader("1. The Learning Curve")
    st.write("Watch how the AI's monthly profit improves from clueless guesses to a competent manager.")
    w=max(1, len(curve)//100)
    smooth=np.convolve(curve, np.ones(w)/w, mode='valid')
    fig, ax = plt.subplots(figsize=(10, 3))
    fig.patch.set_facecolor('#FAF4E8')
    ax.set_facecolor('#FAF4E8')
    ax.plot(smooth, color="#6D4C41", linewidth=2)
    ax.set_xlabel("Month (episode)", color="#3E2723")
    ax.set_ylabel("Profit", color="#3E2723")
    ax.tick_params(colors="#3E2723")
    for spine in ax.spines.values():
        spine.set_edgecolor('#8D6E63')
    st.pyplot(fig)

    st.subheader("2. The Learned Strategy (Policy)")
    st.write("The simple rule the AI discovered: for any given stock level, how much should it order? Notice how it naturally learns to balance the inventory.")
    fig2, ax2 = plt.subplots(figsize=(10, 1.5))
    fig2.patch.set_facecolor('#FAF4E8')
    cax = ax2.imshow(policy.reshape(1,-1), cmap="YlOrBr", aspect="auto")
    for i,v in enumerate(policy):
        ax2.text(i, 0, str(int(v)), ha="center", va="center", color="black" if v < (MAX/2) else "white", fontsize=10, fontweight="bold")
    ax2.set_yticks([])
    ax2.set_xticks(range(len(policy)))
    ax2.set_xlabel("Stock on hand  →", color="#3E2723")
    ax2.tick_params(colors="#3E2723")
    st.pyplot(fig2)

with tab2:
    st.subheader("Performance vs. Human Baselines")
    st.write("Comparing the Reinforcement Learning AI against naive managers who guess.")
    cautious, _, _, _ = eval_policy(env, lambda s: 4)
    generous, _, _, _ = eval_policy(env, lambda s: 12)
    rnd, _, _, _      = eval_policy(env, lambda s: random.randint(0, MAX))
    
    names=["Cautious (orders 4)","Random Guesser","Generous (orders 12)","🧠 AI Agent"]
    vals=[cautious, rnd, generous, rl_profit]
    
    fig3, ax3 = plt.subplots(figsize=(8, 3.5))
    fig3.patch.set_facecolor('#FAF4E8')
    ax3.set_facecolor('#FAF4E8')
    colors=["#D7CCC8", "#D7CCC8", "#D7CCC8", "#4CAF50"]
    ax3.barh(names, vals, color=colors)
    for i,v in enumerate(vals): 
        ax3.text(v, i, f" AED {v:,.0f}", va="center", color="#3E2723", fontweight="bold")
    ax3.set_xlabel("Average monthly profit (AED)", color="#3E2723")
    ax3.tick_params(colors="#3E2723")
    for spine in ax3.spines.values():
        spine.set_visible(False)
    st.pyplot(fig3)

with tab3:
    st.subheader("The 10-Day Café Challenge")
    st.write("You play the café manager for 10 days. Then, the AI faces the **exact same customer demand**. Who makes more profit?")
    
    if 'game_demand' not in st.session_state or st.button("🔄 Start New 10-Day Challenge"):
        g = np.random.default_rng()
        seq = []
        for _ in range(10):
            d = g.poisson(mean_demand)
            if g.random() < 0.2: d += g.poisson(5)
            seq.append(int(d))
        st.session_state['game_demand'] = seq
        st.session_state['you_day'] = 0
        st.session_state['you_stock'] = 5
        st.session_state['you_profit'] = 0.0
        st.session_state['you_log'] = []

    seq = st.session_state['game_demand']
    day = st.session_state['you_day']
    
    if day < 10:
        st.markdown(f"#### Day {day+1} of 10")
        st.info(f"You have **{st.session_state['you_stock']}** milks in stock this morning.")
        order = st.number_input("How many do you order today?", 0, MAX, 7, key=f"ord{day}")
        
        if st.button("Open the shop ▶️", key=f"go{day}"):
            profit, carry, info = simulate_day(params, st.session_state['you_stock'], order, seq[day])
            st.session_state['you_profit'] += profit
            st.session_state['you_stock'] = carry
            st.session_state['you_log'].append(info)
            st.session_state['you_day'] += 1
            st.rerun()
            
        if st.session_state['you_log']:
            st.table(st.session_state['you_log'])
    else:
        # AI plays same demand
        ai_stock = 5
        ai_profit = 0.0
        for d in seq:
            o = ai_order(ai_stock)
            p, ai_stock, _ = simulate_day(params, ai_stock, o, d)
            ai_profit += p
            
        you = st.session_state['you_profit']
        cc1, cc2 = st.columns(2)
        cc1.metric("🧍 Your Total Profit", f"AED {you:,.0f}")
        cc2.metric("🧠 AI Total Profit", f"AED {ai_profit:,.0f}", f"{ai_profit-you:+,.0f} vs you")
        
        if ai_profit > you: 
            st.success("The AI ran the more profitable café! 🤖☕")
        else: 
            st.balloons()
            st.success("You beat the AI! Impressive instinct for the business. 🎉")
            
        st.write("### Your Daily Ledger")
        st.table(st.session_state['you_log'])
