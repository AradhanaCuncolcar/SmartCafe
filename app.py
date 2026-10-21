# ☕ The Smart Café - all three tools in one app:
#   1) the Café Simulator (cafe.py)   2) the Learning Agent (agent.py)   3) this Dashboard
import random
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

st.set_page_config(page_title="Smart Café AI Manager", page_icon="☕", layout="wide")

try:
    from cafe import CafeSim, DAY_NAMES
    from agent import train, policy_from_Q, get_best_action, average_profit
except Exception as e:  # show the real reason instead of hiding it
    st.error(f"Could not import cafe.py / agent.py: {e}")
    st.stop()

ss = st.session_state

# ------------------------------------------------------------------ theme
ESPRESSO, CARAMEL, LATTE, CREAM = "#3E2723", "#C68B59", "#E8D5C0", "#FFF8F0"
GREEN, RED = "#6B8F71", "#B5543C"
st.markdown(f"""
<style>
.stApp {{ background-color: {CREAM}; }}
h1, h2, h3, h4 {{ color: {ESPRESSO}; }}
section[data-testid="stSidebar"] {{ background-color: {LATTE}; }}
.stButton > button {{ background-color: {ESPRESSO}; color: white; border-radius: 8px; font-weight: 600; border: 0; }}
.stButton > button:hover {{ background-color: {CARAMEL}; color: white; }}
div[data-testid="stMetric"] {{ background: white; border: 1px solid {LATTE}; border-radius: 10px; padding: 10px 14px; }}
.rec-card {{ background: {ESPRESSO}; color: {CREAM}; border-radius: 14px; padding: 18px 22px; margin: 8px 0 14px 0; }}
.rec-card .big {{ font-size: 2.2rem; font-weight: 700; color: {CARAMEL}; }}
.pill {{ display:inline-block; padding:2px 10px; border-radius:12px; background:{CARAMEL}; color:white; font-size:.85rem; margin-right:6px; }}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------ settings
DEFAULTS = dict(base_demand=7, spoilage_cost=1.5, stockout_penalty=6.0, weekend_boost=1.6, event_chance=0.15)
DEFAULT_EPISODES = 15000
GAME_DAYS = 10
CAUTIOUS, GENEROUS = 4, 10


def make_env(cfg):
    return CafeSim(**cfg)


def current_cfg():
    return {k: (int(ss[f"sl_{k}"]) if k == "base_demand" else float(ss[f"sl_{k}"])) for k in DEFAULTS}


for k, v in DEFAULTS.items():
    ss.setdefault(f"sl_{k}", v)
ss.setdefault("sl_episodes", DEFAULT_EPISODES)

PRESETS = {
    "💸 Spoilage costs 3x more (1.5 → 5.0)": dict(spoilage_cost=5.0),
    "😡 Unhappy customers cost 2x more (6 → 12)": dict(stockout_penalty=12.0),
    "🛍️ A much busier café (demand 7 → 12)": dict(base_demand=12),
    "🎉 Events twice as likely (15% → 30%)": dict(event_chance=0.30),
}


def apply_values(values):
    for k, v in {**DEFAULTS, **values}.items():
        ss[f"sl_{k}"] = v
    ss["auto_train"] = True


# ------------------------------------------------------------------ sidebar
st.sidebar.title("☕ Café Controls")
st.sidebar.caption("Change the business, press **Train**, and watch the AI's strategy shift.")
st.sidebar.slider("Typical daily demand (customers)", 3, 12, key="sl_base_demand")
st.sidebar.slider("Spoilage cost per unit (AED)", 0.5, 6.0, step=0.5, key="sl_spoilage_cost")
st.sidebar.slider("Stockout penalty per customer (AED)", 0.0, 15.0, step=1.0, key="sl_stockout_penalty")
st.sidebar.slider("Weekend boost (Fri-Sun)", 1.0, 2.5, step=0.1, key="sl_weekend_boost")
st.sidebar.slider("Chance of a local event", 0.0, 0.5, step=0.05, key="sl_event_chance")
st.sidebar.select_slider("Training months", [5000, 10000, 15000, 30000], key="sl_episodes")
train_clicked = st.sidebar.button("🚀 Train the AI", use_container_width=True)
st.sidebar.button("↩️ Reset to default café", on_click=apply_values, args=({},), use_container_width=True)


# ------------------------------------------------------------------ training
def train_model(cfg, episodes):
    env = make_env(cfg)
    bar = st.progress(0.0, text="☕ The AI is practising in the simulator...")
    Q, curve = train(env, episodes=episodes,
                     progress=lambda f: bar.progress(min(f, 1.0), text=f"☕ The AI is practising... {int(f * 100)}%"))
    bar.empty()
    return dict(cfg=cfg, episodes=episodes, Q=Q, curve=curve)


if "model" not in ss or train_clicked or ss.pop("auto_train", False):
    cfg_now = current_cfg()
    ss["model"] = train_model(cfg_now, int(ss["sl_episodes"]))
    if "baseline" not in ss or cfg_now == DEFAULTS:
        ss["baseline"] = ss["model"]

model, baseline = ss["model"], ss["baseline"]
cfg = model["cfg"]
env = make_env(cfg)
policy = policy_from_Q(model["Q"])   # [stock, day, event] -> order
MAX = env.max_stock

if current_cfg() != cfg:
    st.sidebar.warning("Sliders changed - press **Train the AI** to apply them.")
else:
    st.sidebar.success(f"AI trained on {model['episodes']:,} simulated months.")


# ------------------------------------------------------------------ evaluation helpers
def get_eval(m):
    if "eval" not in m:
        e = make_env(m["cfg"])
        pol = policy_from_Q(m["Q"])
        strategies = {
            "🧠 RL agent": lambda s: int(pol[s]),
            f"Generous (always {GENEROUS})": lambda s: GENEROUS,
            f"Cautious (always {CAUTIOUS})": lambda s: CAUTIOUS,
            "Random": lambda s: random.randint(0, e.max_stock),
        }
        res = {}
        for name, fn in strategies.items():
            mean, sd, stt = average_profit(e, fn, months=300)
            res[name] = dict(mean=mean, std=sd,
                             service=stt["sold"] / max(stt["demand"], 1),
                             spoil=stt["spoiled"] / max(stt["ordered"], 1))
        sweep = [average_profit(e, lambda s, a=a: a, months=100)[0] for a in range(e.max_stock + 1)]
        m["eval"] = dict(res=res, sweep=sweep)
    return m["eval"]


def style_ax(ax):
    ax.set_facecolor(CREAM)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(alpha=0.25)


def new_fig(w=7, h=3.6):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(CREAM)
    style_ax(ax)
    return fig, ax


# ------------------------------------------------------------------ header
st.title("☕ The Smart Café: AI Operations Manager")
st.caption("One decision every morning: *how many units should I order?*  "
           "The simulator is the shop, the agent is the brain, and this dashboard is your window into both.")

tabs = st.tabs(["🌅 Morning Briefing", "📈 Learning", "🗺️ Strategy", "🏆 Scoreboard", "🎮 You vs AI", "🎚️ Experiments"])

# ================================================================== TAB 1: Morning Briefing
ss.setdefault("b_stock", 3)
ss.setdefault("b_day", "Friday")
ss.setdefault("b_event", False)
ss.setdefault("ledger", [])


def submit_order(stock, dow, event, rec, order):
    e = make_env(cfg)
    demand = e.draw_demand(dow, event)           # same customers for you and the AI
    mine = e.simulate_day(stock, order, demand)
    ai = e.simulate_day(stock, rec, demand)
    ss["ledger"].append({
        "Day": DAY_NAMES[dow], "Event": "Yes" if event else "No", "Start stock": stock,
        "AI suggested": rec, "You ordered": mine["order"], "Customers": demand, "Sold": mine["sold"],
        "Spoiled": mine["spoiled"], "Missed customers": mine["stockout"],
        "Your profit": mine["profit"], "AI profit (same day)": ai["profit"],
        "Overrode AI?": "Yes" if rec != mine["order"] else "No"})
    ss["b_stock"] = mine["carry"]                         # today's leftover is tomorrow's stock
    ss["b_day"] = DAY_NAMES[(dow + 1) % 7]
    ss["b_event"] = False


with tabs[0]:
    left, right = st.columns([1, 1.4], gap="large")
    with left:
        st.subheader("Today's situation")
        with st.container(border=True):
            stock_in = st.number_input("Leftover stock (units still fresh)", 0, MAX, key="b_stock")
            day_in = st.selectbox("Day of the week", DAY_NAMES, key="b_day")
            event_in = st.checkbox("🎉 Local event / high footfall expected today", key="b_event")
        dow_i, evt_i = DAY_NAMES.index(day_in), int(event_in)
        rec = get_best_action(model["Q"], int(stock_in), dow_i, evt_i)
        exp_d = env.expected_demand(dow_i, evt_i)
        after = min(int(stock_in) + rec, MAX)
        st.markdown(f"""<div class="rec-card">🤖 AI recommends ordering<br>
            <span class="big">{rec} units</span><br>
            <span class="pill">expect ≈ {exp_d:.0f} customers</span>
            <span class="pill">{after} units on the shelf</span></div>""", unsafe_allow_html=True)
        if exp_d > MAX:
            st.warning(f"Expected demand ({exp_d:.0f}) is more than the fridge holds ({MAX}). "
                       "Even a full fridge may run out - plan a mid-day top-up if you can.")
        qrow = model["Q"][int(stock_in), dow_i, evt_i]
        if np.all(qrow == 0):
            st.info("The AI rarely saw this exact situation in training, so treat its advice with care.")

        fig, ax = new_fig(5.5, 2.6)
        ax.bar(range(MAX + 1), qrow, color=[CARAMEL if a == rec else LATTE for a in range(MAX + 1)])
        ax.set_xlabel("Units ordered"); ax.set_ylabel("AI's score")
        ax.set_title("How the AI rates each order (taller = better)", fontsize=10, color=ESPRESSO)
        st.pyplot(fig); plt.close(fig)

        st.markdown("#### Execute order")
        order_in = st.number_input("Your order (change it to override the AI)", 0, MAX, value=rec)
        st.button("✅ Submit order & close the day", on_click=submit_order,
                  args=(int(stock_in), dow_i, evt_i, rec, int(order_in)), use_container_width=True)
        if ss["ledger"]:
            last = ss["ledger"][-1]
            st.success(f"Last day: AED {last['Your profit']:,.1f} (AI would have made AED {last['AI profit (same day)']:,.1f}). "
                       "Tomorrow's stock and day are filled in for you.")
    with right:
        st.subheader("📒 Executive ledger")
        if ss["ledger"]:
            df = pd.DataFrame(ss["ledger"])
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Your total profit", f"AED {df['Your profit'].sum():,.0f}")
            c2.metric("AI on same days", f"AED {df['AI profit (same day)'].sum():,.0f}",
                      delta=f"{df['Your profit'].sum() - df['AI profit (same day)'].sum():,.0f} you vs AI")
            c3.metric("Spoiled units", int(df["Spoiled"].sum()))
            c4.metric("Missed customers", int(df["Missed customers"].sum()))
            st.dataframe(df.tail(15), hide_index=True)
            st.button("🗑️ Clear ledger", on_click=lambda: ss.update(ledger=[]))
        else:
            st.info("The ledger is empty. Submit your first morning order to start tracking.")

# ================================================================== TAB 2: Learning
with tabs[1]:
    st.subheader("📈 Proof the AI is learning")
    curve = np.array(model["curve"])
    w = max(50, len(curve) // 60)
    ma = np.convolve(curve, np.ones(w) / w, mode="valid")
    fig, ax = new_fig(9, 3.8)
    ax.plot(curve, color=CARAMEL, alpha=0.25, lw=0.6, label="each practice month")
    ax.plot(np.arange(w - 1, len(curve)), ma, color=ESPRESSO, lw=2, label=f"average of {w} months")
    ax.set_xlabel("Training month"); ax.set_ylabel("Profit per month (AED)"); ax.legend(frameon=False)
    st.pyplot(fig); plt.close(fig)
    start_avg, end_avg = curve[:w].mean(), curve[-w:].mean()
    with st.spinner("Measuring the trained agent..."):
        ev = get_eval(model)
    rl = ev["res"]["🧠 RL agent"]
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Profit at the start", f"AED {start_avg:,.0f}")
    k2.metric("Profit at the end", f"AED {end_avg:,.0f}", delta=f"{end_avg - start_avg:,.0f}")
    k3.metric("Trained avg / month", f"AED {rl['mean']:,.0f}")
    k4.metric("Service level", f"{rl['service'] * 100:.0f}%")
    k5.metric("Spoilage rate", f"{rl['spoil'] * 100:.0f}%")
    st.caption("The curve is noisy and climbs gradually - that is normal. Early on the AI explores (random orders); "
               "later it mostly uses what it learned. 'Trained avg' is measured over 300 fresh months with no exploring.")

# ================================================================== TAB 3: Strategy
with tabs[2]:
    st.subheader("🗺️ The strategy the AI discovered")
    st.write("Nobody programmed this. Read it as: *\"it's Friday, I have 3 units left - order this many.\"*")
    view = st.radio("Show the plan for", ["Normal days", "Event days 🎉"], horizontal=True)
    evt_view = 1 if view.startswith("Event") else 0
    grid = policy[:, :, evt_view]
    cmap = LinearSegmentedColormap.from_list("cafe", [CREAM, CARAMEL, ESPRESSO])
    fig, ax = plt.subplots(figsize=(9, 5.6))
    fig.patch.set_facecolor(CREAM)
    ax.imshow(grid, cmap=cmap, aspect="auto", vmin=0, vmax=MAX)
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            ax.text(c, r, int(grid[r, c]), ha="center", va="center", fontsize=9,
                    color="white" if grid[r, c] > MAX * 0.55 else ESPRESSO)
    ax.set_xticks(range(7)); ax.set_xticklabels([d[:3] for d in DAY_NAMES])
    ax.set_yticks(range(MAX + 1)); ax.set_ylabel("Stock on hand in the morning"); ax.set_xlabel("Day of the week")
    ax.set_title("Units to order", color=ESPRESSO)
    st.pyplot(fig); plt.close(fig)

    st.markdown("#### In plain English: how full to fill the fridge")
    rows = []
    for d in range(7):
        row = {"Day": DAY_NAMES[d]}
        for e_, label in ((0, "Normal day"), (1, "Event day 🎉")):
            target = np.mean([min(s + int(policy[s, d, e_]), MAX) for s in range(0, 5)])
            row[label] = f"top up to ≈ {target:.0f} units"
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    st.caption("Pattern to look for: low stock → order a lot, high stock → order little or nothing, and bigger orders on "
               "weekends and event days. Some high-stock cells look odd because the AI rarely visits them, so those "
               "values are less reliable - normal for tabular Q-learning.")

# ================================================================== TAB 4: Scoreboard
with tabs[3]:
    st.subheader("🏆 Who ran the better café?")
    with st.spinner("Running each manager through hundreds of simulated months..."):
        ev = get_eval(model)
    board = {k: v["mean"] for k, v in ev["res"].items()}
    human = ss.get("human")
    if human:
        board["🙋 You (10-day pace x3)"] = human["per_day"] * 30
    names = list(board.keys()); vals = [board[n] for n in names]
    fig, ax = new_fig(9, 3.8)
    bars = ax.barh(names, vals, color=[CARAMEL if n.startswith("🧠") else (RED if v < 0 else LATTE) for n, v in zip(names, vals)])
    ax.axvline(0, color=ESPRESSO, lw=0.8); ax.invert_yaxis(); ax.set_xlabel("Average profit per 30-day month (AED)")
    for b, v in zip(bars, vals):
        ax.text(v + (8 if v >= 0 else -8), b.get_y() + b.get_height() / 2, f"{v:,.0f}",
                va="center", ha="left" if v >= 0 else "right", color=ESPRESSO, fontsize=9)
    st.pyplot(fig); plt.close(fig)
    tbl = pd.DataFrame([{"Manager": n, "Avg monthly profit (AED)": round(v["mean"]), "Month-to-month swing (±)": round(v["std"]),
                         "Service level": f"{v['service'] * 100:.0f}%", "Spoilage rate": f"{v['spoil'] * 100:.0f}%"}
                        for n, v in ev["res"].items()])
    st.dataframe(tbl, hide_index=True)
    if not human:
        st.info("Play the **🎮 You vs AI** tab to add yourself to this scoreboard.")
    with st.expander("Is there a 'best fixed order'? (profit if you always order the same number)"):
        fig, ax = new_fig(8, 3)
        ax.plot(range(MAX + 1), ev["sweep"], marker="o", color=ESPRESSO)
        ax.axhline(rl["mean"], color=CARAMEL, ls="--", label="RL agent")
        ax.set_xlabel("Fixed daily order"); ax.set_ylabel("Avg monthly profit (AED)"); ax.legend(frameon=False)
        st.pyplot(fig); plt.close(fig)
        st.caption("There is a sweet spot for a single fixed number - but nobody knows it in advance, and it can't "
                   "react to weekends, events or leftover stock. The AI finds it, then goes beyond it.")

# ================================================================== TAB 5: You vs AI
ss.setdefault("g_order", 5)


def new_game():
    g_env = make_env(cfg)
    start = random.randint(0, 6)
    script = []
    for i in range(GAME_DAYS):
        d = (start + i) % 7
        e_ = 1 if random.random() < g_env.event_chance else 0
        script.append(dict(dow=d, event=e_, demand=g_env.draw_demand(d, e_)))
    stock, ai_rows = 5, []                       # the AI plays the exact same days (hidden until the end)
    for i, day in enumerate(script):
        o = int(policy[stock, day["dow"], day["event"]])
        r = g_env.simulate_day(stock, o, day["demand"])
        ai_rows.append(dict(**r, start=stock)); stock = r["carry"]
    ss["game"] = dict(cfg=cfg, script=script, ai=ai_rows, you=[], stock=5)
    ss["g_order"] = 5
    ss.pop("human", None)


def place_order():
    g = ss["game"]
    day = g["script"][len(g["you"])]
    r = make_env(g["cfg"]).simulate_day(g["stock"], int(ss["g_order"]), day["demand"])
    g["you"].append(dict(**r, start=g["stock"]))
    g["stock"] = r["carry"]
    if len(g["you"]) == GAME_DAYS:
        ss["human"] = dict(per_day=sum(x["profit"] for x in g["you"]) / GAME_DAYS)


with tabs[4]:
    st.subheader(f"🎮 You vs the AI - {GAME_DAYS} days as café manager")
    st.write("You and the AI face the **same customers** on the **same days**. Order each morning, then see who earns more.")
    g = ss.get("game")
    if g is None or (len(g["you"]) == GAME_DAYS and st.button("🔄 Play again")):
        new_game(); g = ss["game"]
    if len(g["you"]) < GAME_DAYS:
        i = len(g["you"]); day = g["script"][i]
        st.progress(i / GAME_DAYS, text=f"Day {i + 1} of {GAME_DAYS}")
        c1, c2 = st.columns([1, 1.3], gap="large")
        with c1:
            ev_tag = " 🎉 **Local event today!**" if day["event"] else ""
            st.markdown(f"### {DAY_NAMES[day['dow']]}{ev_tag}")
            st.metric("Stock on hand", g["stock"])
            st.number_input("How many units do you order?", 0, MAX, key="g_order")
            st.button("Place order", on_click=place_order, use_container_width=True)
            st.caption("Hint: leftover stock counts, weekends and events are busier, and the fridge holds 15.")
        with c2:
            if g["you"]:
                st.dataframe(pd.DataFrame(g["you"])[["start", "order", "demand", "sold", "spoiled", "stockout", "profit"]]
                             .rename(columns={"start": "Start stock", "order": "You ordered", "demand": "Customers",
                                              "sold": "Sold", "spoiled": "Spoiled", "stockout": "Missed", "profit": "Profit"}),
                             hide_index=True)
    else:
        you_total = sum(x["profit"] for x in g["you"]); ai_total = sum(x["profit"] for x in g["ai"])
        m1, m2, m3 = st.columns(3)
        m1.metric("🙋 You", f"AED {you_total:,.0f}")
        m2.metric("🧠 AI", f"AED {ai_total:,.0f}")
        m3.metric("Difference", f"AED {you_total - ai_total:,.0f}")
        if you_total > ai_total:
            st.success("You beat the AI this round! With only 10 days luck matters a lot - try again.")
        else:
            st.info("The AI won this round. Check the table to see where the orders differed.")
        rows = []
        for i, (y, a, d) in enumerate(zip(g["you"], g["ai"], g["script"])):
            rows.append({"Day": DAY_NAMES[d["dow"]] + (" 🎉" if d["event"] else ""), "Customers": d["demand"],
                         "You: stock": y["start"], "You: order": y["order"], "You: profit": y["profit"],
                         "AI: stock": a["start"], "AI: order": a["order"], "AI: profit": a["profit"]})
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        st.caption("Your result is now on the 🏆 Scoreboard.")

# ================================================================== TAB 6: Experiments
with tabs[5]:
    st.subheader("🎚️ What you reward is what you get")
    st.write("Change the business, retrain, and compare the new strategy with the original café. "
             "Pick a ready-made experiment (it sets the sliders and retrains), or use the sidebar sliders and press **Train**.")
    cols = st.columns(2)
    for i, name in enumerate(PRESETS):
        cols[i % 2].button(name, on_click=apply_values, args=(PRESETS[name],), key=f"preset_{i}", use_container_width=True)

    if cfg == baseline["cfg"]:
        st.info("The AI is currently trained on the default café. Choose an experiment above to see the strategy shift.")
    else:
        changes = [f"**{k.replace('_', ' ')}**: {baseline['cfg'][k]} → {cfg[k]}" for k in DEFAULTS if cfg[k] != baseline["cfg"][k]]
        st.markdown("Changed: " + " · ".join(changes))
        base_pol = policy_from_Q(baseline["Q"])

        def avg_order(pol, p_event):
            wts = np.array([1 - p_event, p_event])
            return np.array([(pol[s].mean(axis=0) * wts).sum() for s in range(MAX + 1)])

        base_curve, new_curve = avg_order(base_pol, baseline["cfg"]["event_chance"]), avg_order(policy, cfg["event_chance"])
        fig, ax = new_fig(8, 3.6)
        ax.plot(range(MAX + 1), base_curve, marker="o", color=LATTE, lw=2.5, label="Original café")
        ax.plot(range(MAX + 1), new_curve, marker="o", color=ESPRESSO, lw=2.5, label="Changed café")
        ax.set_xlabel("Stock on hand in the morning"); ax.set_ylabel("Average units ordered"); ax.legend(frameon=False)
        st.pyplot(fig); plt.close(fig)
        low_b, low_n = base_curve[:4].mean(), new_curve[:4].mean()
        verb = "more" if low_n > low_b else "less"
        st.markdown(f"When stock is low (0-3 units) the AI now orders **{low_n:.1f}** units on average, versus **{low_b:.1f}** "
                    f"in the original café - **{abs(low_n - low_b):.1f} {verb}**.")
        if "base_in_new" not in model:
            model["base_in_new"] = average_profit(env, lambda s: int(base_pol[s]), months=200)[0]
        new_profit = get_eval(model)["res"]["🧠 RL agent"]["mean"]
        a, b = st.columns(2)
        a.metric("Old strategy in the changed café", f"AED {model['base_in_new']:,.0f}")
        b.metric("Retrained strategy", f"AED {new_profit:,.0f}", delta=f"{new_profit - model['base_in_new']:,.0f}")
        st.caption("This is why you retrain when costs change: the AI optimises exactly the rewards you give it, so the "
                   "numbers you choose quietly decide its behaviour. Raise spoilage cost → it orders less. "
                   "Raise the stockout penalty → it keeps a bigger safety buffer.")
