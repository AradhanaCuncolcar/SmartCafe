# app.py — The Smart Café Dashboard
# ------------------------------------------------------------
# Run:  streamlit run app.py
# Needs (same folder, written by YOU):  cafe.py  (class CafeSim)  and  agent.py  (train, policy_from_Q)
# ------------------------------------------------------------
import numpy as np
import streamlit as st

st.set_page_config(page_title="The Smart Café", page_icon="☕", layout="wide")

# ---- import the student's code ----
try:
    from cafe import CafeSim
    from agent import train, policy_from_Q
except Exception as e:
    st.error("Could not import your code. Make sure **cafe.py** (with class `CafeSim`) and "
             "**agent.py** (with `train` and `policy_from_Q`) are in this folder and run without errors.")
    st.exception(e); st.stop()

# ---------- helpers ----------
def eval_policy(env, order_fn, months=400):
    """Average monthly profit + operational stats by running the student's simulator."""
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

def one_month_ledger(env, order_fn):
    rows=[]; s=env.reset(); done=False; day=1
    while not done:
        start=s; r,s,done,info=env.step(order_fn(start))
        rows.append(dict(Day=day, Start=start, Order=info['order'], Demand=info['demand'],
                         Sold=info['sold'], Spoiled=info['spoiled'], Stockout=info['stockout'],
                         Carry=info['carry'], Profit=info['profit']))
        day+=1
    return rows

def simulate_day(p, stock, order, demand):
    """Self-contained day economics (for fair You-vs-AI on identical demand)."""
    order=int(np.clip(order,0,p['max_stock']))
    available=min(stock+order, p['max_stock'])
    sold=min(available, demand); stockout=max(demand-available,0); leftover=available-sold
    spoiled=int(round(p['spoil_frac']*leftover)); carry=leftover-spoiled
    profit=sold*p['price']-(order*p['order_cost']+leftover*p['holding_cost']
                            +spoiled*p['spoilage_cost']+stockout*p['stockout_penalty'])
    return round(profit,2), carry, dict(start=stock,order=order,demand=demand,sold=sold,
                                        spoiled=spoiled,stockout=stockout,carry=carry,profit=round(profit,2))

# ---------- sidebar: the business + the AI knobs ----------
st.sidebar.header("🏪 The Business")
spoilage_cost   = st.sidebar.slider("Spoilage cost / unit (AED)", 0.0, 5.0, 1.5, 0.1)
stockout_pen    = st.sidebar.slider("Stockout penalty / customer (AED)", 0.0, 12.0, 6.0, 0.5)
mean_demand     = st.sidebar.slider("Typical daily demand", 3, 12, 7, 1)
spoil_frac      = st.sidebar.slider("Fraction that spoils overnight", 0.0, 1.0, 0.25, 0.05)
st.sidebar.header("🧠 The AI")
episodes        = st.sidebar.select_slider("Training months (episodes)", [2000,5000,8000,12000,15000,20000], value=12000)
gamma           = st.sidebar.slider("γ  (how much it values the future)", 0.0, 0.99, 0.95, 0.01)

def make_env():
    return CafeSim(spoilage_cost=spoilage_cost, stockout_penalty=stockout_pen,
                   mean_demand=mean_demand, spoil_frac=spoil_frac)

params = dict(price=5.0, order_cost=1.0, holding_cost=0.3, spoilage_cost=spoilage_cost,
              stockout_penalty=stockout_pen, spoil_frac=spoil_frac, max_stock=make_env().max_stock)

st.title("☕ The Smart Café — can an AI learn to run the shop?")

if st.sidebar.button("🚀 Train the AI", type="primary"):
    with st.spinner("The AI is practising over thousands of months..."):
        env=make_env()
        Q, curve = train(env, episodes=episodes, gamma=gamma)
        st.session_state['Q']=Q; st.session_state['curve']=curve
        st.session_state['params']=params
    st.success("Training complete! Explore the tabs below.")

if 'Q' not in st.session_state:
    st.info("👈 Set the business and the AI on the left, then press **Train the AI**.")
    st.stop()

Q=st.session_state['Q']; curve=st.session_state['curve']
policy=policy_from_Q(Q); MAX=make_env().max_stock
ai_order = lambda s: int(policy[int(np.clip(s,0,MAX))])
env=make_env()

# ---------- KPI cards ----------
rl_profit, service, spoil_rate, stockout_rate = eval_policy(env, ai_order)
c1,c2,c3,c4 = st.columns(4)
c1.metric("💰 Avg monthly profit", f"AED {rl_profit:,.0f}")
c2.metric("✅ Service level (demand met)", f"{service*100:.0f}%")
c3.metric("🗑️ Spoilage rate", f"{spoil_rate*100:.0f}%")
c4.metric("🚫 Stockout rate", f"{stockout_rate*100:.0f}%")

tab1,tab2,tab3,tab4 = st.tabs(["📈 Learning", "🗺️ Strategy", "🏆 Scoreboard", "🎮 You vs AI"])

with tab1:
    st.subheader("Did it actually learn?")
    import matplotlib.pyplot as plt
    w=max(1,len(curve)//100)
    smooth=np.convolve(curve, np.ones(w)/w, mode='valid')
    fig,ax=plt.subplots(figsize=(8,3.2))
    ax.plot(smooth, color="#2563eb"); ax.set_xlabel("Month (episode)"); ax.set_ylabel("Profit")
    ax.set_title("Average monthly profit while training — it should climb then flatten")
    st.pyplot(fig)
    st.caption(f"Started around AED {np.mean(curve[:200]):.0f}/month → ended around AED {np.mean(curve[-200:]):.0f}/month.")

with tab2:
    st.subheader("The strategy the AI discovered")
    st.write("For each amount of **stock on hand**, how much does it choose to **order**? "
             "Watch for the pattern: *order more when stock is low, little or nothing when stock is high.*")
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,1.6))
    ax.imshow(policy.reshape(1,-1), cmap="YlGnBu", aspect="auto")
    for i,v in enumerate(policy):
        ax.text(i,0,str(int(v)),ha="center",va="center",fontsize=9)
    ax.set_yticks([]); ax.set_xticks(range(len(policy))); ax.set_xlabel("Stock on hand  →")
    ax.set_title("Learned policy:  stock level → order amount")
    st.pyplot(fig)

with tab3:
    st.subheader("Who runs the better café?")
    cautious,_,_,_ = eval_policy(env, lambda s: 4)
    generous,_,_,_ = eval_policy(env, lambda s: 12)
    import random as _r
    rnd,_,_,_      = eval_policy(env, lambda s: _r.randint(0,MAX))
    names=["Cautious (order 4)","Random","Generous (order 12)","🧠 RL Agent"]
    vals=[cautious,rnd,generous,rl_profit]
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,3.2))
    colors=["#94a3b8","#94a3b8","#94a3b8","#16a34a"]
    ax.barh(names, vals, color=colors)
    for i,v in enumerate(vals): ax.text(v,i,f" {v:,.0f}",va="center")
    ax.set_xlabel("Average monthly profit (AED)"); ax.set_title("RL vs the guessing managers")
    st.pyplot(fig)
    st.caption("Notice: a 'cautious' manager who under-orders can actually *lose money* — stockouts cost more than waste.")

with tab4:
    st.subheader("🎮 You play manager for 10 days — then the AI plays the same 10 days")
    if 'game_demand' not in st.session_state or st.button("🔄 New 10-day challenge"):
        g=np.random.default_rng(); seq=[]
        for _ in range(10):
            d=g.poisson(mean_demand)
            if g.random()<0.2: d+=g.poisson(5)
            seq.append(int(d))
        st.session_state['game_demand']=seq
        st.session_state['you_day']=0; st.session_state['you_stock']=5; st.session_state['you_profit']=0.0; st.session_state['you_log']=[]
    seq=st.session_state['game_demand']
    day=st.session_state['you_day']
    if day<10:
        st.write(f"**Day {day+1} of 10** — you have **{st.session_state['you_stock']}** units in stock this morning.")
        order=st.number_input("How many do you order today?", 0, MAX, 7, key=f"ord{day}")
        if st.button("Open the shop ▶️", key=f"go{day}"):
            profit,carry,info=simulate_day(params, st.session_state['you_stock'], order, seq[day])
            st.session_state['you_profit']+=profit; st.session_state['you_stock']=carry
            st.session_state['you_log'].append(info); st.session_state['you_day']+=1
            st.rerun()
        if st.session_state['you_log']:
            st.table(st.session_state['you_log'])
    else:
        # AI plays same demand
        ai_stock=5; ai_profit=0.0
        for d in seq:
            o=ai_order(ai_stock); p,ai_stock,_=simulate_day(params, ai_stock, o, d)
            ai_profit+=p
        you=st.session_state['you_profit']
        cc1,cc2=st.columns(2)
        cc1.metric("🧍 Your total profit", f"AED {you:,.0f}")
        cc2.metric("🧠 AI's total profit", f"AED {ai_profit:,.0f}", f"{ai_profit-you:+,.0f} vs you")
        if ai_profit>you: st.success("The AI ran the more profitable café. It never saw the answer — it learned it. 🤖")
        else: st.balloons(); st.success("You beat the AI! Impressive instinct for the business. 🎉")
        st.table(st.session_state['you_log'])
        if st.button("Play again"): 
            for k in ['game_demand','you_day','you_stock','you_profit','you_log']: st.session_state.pop(k,None)
            st.rerun()
