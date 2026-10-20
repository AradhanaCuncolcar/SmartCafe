# app.py — The Smart Café: Simulator · Learning Agent · Dashboard Insights
# ------------------------------------------------------------------------
# Run:  streamlit run app.py
# Needs (same folder):  cafe.py (class CafeSim)  and  agent.py (train, policy_from_Q)
# ------------------------------------------------------------------------
import random
import time

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="The Smart Café", page_icon="☕", layout="wide")

try:
    from cafe import CafeSim
    from agent import train, policy_from_Q
except Exception as e:  # pragma: no cover
    st.error("Could not import your code. Make sure **cafe.py** (class `CafeSim`) and "
             "**agent.py** (`train`, `policy_from_Q`) are in this folder and run without errors.")
    st.exception(e)
    st.stop()

MAX_STOCK = 15
DAYS = 30
BLUE, GREEN, GREY, RED, AMBER = "#2563eb", "#16a34a", "#94a3b8", "#dc2626", "#d97706"
ss = st.session_state


# =====================================================================
# Helpers (all use YOUR simulator; nothing here re-implements the rules
# except simulate_day, which gives a fair You-vs-AI on identical demand)
# =====================================================================
def make_env(settings):
    return CafeSim(**settings)


def settings_key(settings):
    return tuple(sorted(settings.items()))


def order_fn(kind, arg, max_stock=MAX_STOCK):
    if kind == "fixed":
        return lambda s: arg
    if kind == "random":
        return lambda s: random.randint(0, max_stock)
    if kind == "upto":
        return lambda s: max(0, arg - s)
    if kind == "policy":
        return lambda s: int(arg[int(np.clip(s, 0, max_stock))])
    raise ValueError(kind)


@st.cache_data(show_spinner=False)
def evaluate(key, kind, arg, months=400):
    """Run `months` simulated months of a strategy. Returns profits + operating stats."""
    settings = dict(key)
    env = CafeSim(**settings)
    fn = order_fn(kind, arg, settings["max_stock"])
    profits = []
    demand = sold = spoiled = stockout = orders = 0
    visits = np.zeros(settings["max_stock"] + 1)
    target_sum = days_total = 0
    for _ in range(months):
        s = env.reset()
        done = False
        tot = 0
        while not done:
            o = fn(s)
            visits[int(s)] += 1
            target_sum += min(int(s) + int(o), settings["max_stock"])
            days_total += 1
            r, s, done, info = env.step(o)
            tot += r
            demand += info["demand"]; sold += info["sold"]; spoiled += info["spoiled"]
            stockout += info["stockout"]; orders += info["order"]
        profits.append(tot)
    return dict(profits=profits, mean=float(np.mean(profits)),
                service=sold / max(demand, 1),
                spoil_rate=spoiled / max(orders, 1),
                stockout_rate=stockout / max(demand, 1),
                spoiled_pm=spoiled / months, stockout_pm=stockout / months,
                visits=visits / max(visits.sum(), 1),
                avg_target=target_sum / max(days_total, 1))


def run_month(settings, fn):
    env = make_env(settings)
    s = env.reset()
    done = False
    rows = []
    day = 1
    while not done:
        start = s
        r, s, done, info = env.step(fn(start))
        rows.append(dict(Day=day, Start=start, Order=info["order"], Demand=info["demand"],
                         Sold=info["sold"], Spoiled=info["spoiled"], Stockout=info["stockout"],
                         Carry=info["carry"], Profit=info["profit"]))
        day += 1
    return pd.DataFrame(rows)


def simulate_day(p, stock, order, demand):
    """Same economics as CafeSim.step, but with a fixed demand (for fair You-vs-AI)."""
    order = int(np.clip(order, 0, p["max_stock"]))
    available = min(stock + order, p["max_stock"])
    sold = min(available, demand)
    stockout = max(demand - available, 0)
    leftover = available - sold
    spoiled = int(round(p["spoil_frac"] * leftover))
    carry = leftover - spoiled
    profit = sold * p["price"] - (order * p["order_cost"] + leftover * p["holding_cost"]
                                  + spoiled * p["spoilage_cost"] + stockout * p["stockout_penalty"])
    return round(profit, 2), carry, dict(Start=stock, Order=order, Demand=demand, Sold=sold,
                                         Spoiled=spoiled, Stockout=stockout, Carry=carry,
                                         Profit=round(profit, 2))


def explain_day(p, stock, order, demand):
    """Line-by-line worked example for the calculator."""
    order = int(np.clip(order, 0, p["max_stock"]))
    available = min(stock + order, p["max_stock"])
    sold = min(available, demand)
    stockout = max(demand - available, 0)
    leftover = available - sold
    spoiled = int(round(p["spoil_frac"] * leftover))
    carry = leftover - spoiled
    rev = sold * p["price"]
    c_order = order * p["order_cost"]
    c_hold = leftover * p["holding_cost"]
    c_spoil = spoiled * p["spoilage_cost"]
    c_out = stockout * p["stockout_penalty"]
    profit = rev - c_order - c_hold - c_spoil - c_out
    rows = [
        ("Available today", f"min({stock} + {order}, {p['max_stock']})", available),
        ("Sold", f"min({available} available, {demand} demand)", sold),
        ("Stockout (customers turned away)", f"max({demand} − {available}, 0)", stockout),
        ("Leftover after selling", f"{available} − {sold}", leftover),
        ("Spoiled overnight", f"round({p['spoil_frac']} × {leftover})", spoiled),
        ("Carried to tomorrow", f"{leftover} − {spoiled}", carry),
        ("Revenue", f"{sold} × {p['price']}", f"AED {rev:.2f}"),
        ("Order cost", f"{order} × {p['order_cost']}", f"− AED {c_order:.2f}"),
        ("Holding cost (on leftover)", f"{leftover} × {p['holding_cost']}", f"− AED {c_hold:.2f}"),
        ("Spoilage cost", f"{spoiled} × {p['spoilage_cost']}", f"− AED {c_spoil:.2f}"),
        ("Stockout penalty", f"{stockout} × {p['stockout_penalty']}", f"− AED {c_out:.2f}"),
        ("PROFIT (the reward)", "revenue − all costs", f"AED {profit:.2f}"),
    ]
    return pd.DataFrame(rows, columns=["Step", "Calculation", "Result"]).astype(str)


def kpi_row(df):
    c = st.columns(5)
    c[0].metric("💰 Profit", f"AED {df['Profit'].sum():,.0f}")
    c[1].metric("🛒 Sold", f"{int(df['Sold'].sum())}")
    c[2].metric("🗑️ Spoiled", f"{int(df['Spoiled'].sum())}")
    c[3].metric("🚫 Stockouts", f"{int(df['Stockout'].sum())}")
    c[4].metric("📦 Ordered", f"{int(df['Order'].sum())}")


# =====================================================================
# Sidebar: the business + the AI knobs
# =====================================================================
st.sidebar.title("☕ The Smart Café")
st.sidebar.header("🏪 The business")
spoilage_cost = st.sidebar.slider("Spoilage cost / unit (AED)", 0.0, 5.0, 1.5, 0.1)
stockout_pen = st.sidebar.slider("Stockout penalty / customer (AED)", 0.0, 12.0, 6.0, 0.5)
mean_demand = st.sidebar.slider("Typical daily demand", 3, 12, 7, 1)
spoil_frac = st.sidebar.slider("Fraction that spoils overnight", 0.0, 1.0, 0.25, 0.05)
busy_chance = st.sidebar.slider("Chance of a busy day", 0.0, 0.6, 0.20, 0.05)
with st.sidebar.expander("Advanced costs"):
    price = st.slider("Selling price / unit (AED)", 1.0, 10.0, 5.0, 0.5)
    order_cost = st.slider("Order cost / unit (AED)", 0.0, 4.0, 1.0, 0.1)
    holding_cost = st.slider("Holding cost / leftover unit (AED)", 0.0, 2.0, 0.3, 0.1)

st.sidebar.header("🧠 The AI")
episodes = st.sidebar.select_slider("Training months (episodes)",
                                    [2000, 5000, 8000, 12000, 15000, 20000, 30000], value=15000)
alpha = st.sidebar.slider("α  (learning rate)", 0.01, 0.50, 0.15, 0.01)
gamma = st.sidebar.slider("γ  (how much it values the future)", 0.0, 0.99, 0.95, 0.01)
eps_decay = st.sidebar.select_slider("ε decay (exploration fades)",
                                     [0.999, 0.9995, 0.9998, 0.9999], value=0.9998)
train_btn = st.sidebar.button("🚀 Train the AI", type="primary")
st.sidebar.caption("Change the business, press Train again, and compare runs in the "
                   "**Dashboard Insights** tab.")

live_settings = dict(price=price, order_cost=order_cost, holding_cost=holding_cost,
                     spoilage_cost=spoilage_cost, stockout_penalty=stockout_pen,
                     spoil_frac=spoil_frac, mean_demand=mean_demand,
                     busy_chance=busy_chance, max_stock=MAX_STOCK, days=DAYS)

# =====================================================================
# Training
# =====================================================================
if train_btn:
    with st.spinner("The AI is practising over thousands of simulated months..."):
        t0 = time.time()
        env = make_env(live_settings)
        Q, curve = train(env, episodes=episodes, alpha=alpha, gamma=gamma, eps_decay=eps_decay)
        policy = [int(x) for x in policy_from_Q(Q)]
        ss["trained"] = dict(Q=np.array(Q), curve=[float(c) for c in curve], policy=policy,
                             settings=dict(live_settings),
                             hyper=dict(episodes=episodes, alpha=alpha, gamma=gamma,
                                        eps_decay=eps_decay, eps_start=1.0, eps_min=0.05),
                             seconds=time.time() - t0)
        res = evaluate(settings_key(live_settings), "policy", tuple(policy))
        snaps = ss.setdefault("snapshots", [])
        snaps.append(dict(
            Run=f"#{len(snaps) + 1}",
            Setup=f"spoil {spoilage_cost} · stockout {stockout_pen} · demand {mean_demand} · "
                  f"spoils {int(spoil_frac * 100)}%",
            Policy=policy, Profit=res["mean"], Service=res["service"],
            Spoilage=res["spoil_rate"], Stockouts=res["stockout_rate"],
            Target=res["avg_target"]))
        ss["snapshots"] = snaps[-8:]
        ss["ai_month"] = None
        for k in ["g_demand", "g_day", "g_stock", "g_log", "g_result"]:
            ss.pop(k, None)
    st.success(f"Training complete in {ss['trained']['seconds']:.1f}s. Explore the tabs below.")

trained = ss.get("trained")

# =====================================================================
# Page
# =====================================================================
st.title("☕ The Smart Café — can an AI learn to run the shop?")
tab_sim, tab_agent, tab_dash = st.tabs(["🏪 1 · Café Simulator", "🧠 2 · Learning Agent",
                                        "📊 3 · Dashboard Insights"])

# ---------------------------------------------------------------------
# TAB 1 — CAFÉ SIMULATOR
# ---------------------------------------------------------------------
with tab_sim:
    st.header("The Café Simulator")
    st.write("This is the business world in code (your `CafeSim`). Every morning you choose **one number** — "
             "how many units to order — and the simulator plays out the day. It uses the settings in the sidebar.")
    s1, s2, s3, s4 = st.tabs(["📜 Rules & worked example", "▶️ Run a month", "🎮 Manage it yourself",
                              "⚠️ The two failure modes"])

    with s1:
        c1, c2 = st.columns([1, 1])
        with c1:
            st.subheader("Business rules (current settings)")
            rules = pd.DataFrame([
                ("Fridge capacity", f"{MAX_STOCK} units"),
                ("Selling price", f"AED {price}"),
                ("Order cost", f"AED {order_cost} / unit"),
                ("Holding cost", f"AED {holding_cost} / leftover unit"),
                ("Spoilage cost", f"AED {spoilage_cost} / spoiled unit"),
                ("Stockout penalty", f"AED {stockout_pen} / unhappy customer"),
                ("Spoils overnight", f"{int(spoil_frac * 100)}% of leftover"),
                ("Typical demand", f"about {mean_demand} / day (Poisson)"),
                ("Busy days", f"{int(busy_chance * 100)}% of days, extra ~5 customers"),
                ("Month length", f"{DAYS} days"),
            ], columns=["Rule", "Value"])
            st.table(rules)
            st.info("**Profit = sales − order cost − holding cost − spoilage cost − stockout penalty**")
        with c2:
            st.subheader("Try one day by hand")
            a, b, c = st.columns(3)
            w_stock = a.number_input("Stock this morning", 0, MAX_STOCK, 5)
            w_order = b.number_input("You order", 0, MAX_STOCK, 8)
            w_dem = c.number_input("Customers want", 0, 30, 7)
            st.table(explain_day(live_settings, int(w_stock), int(w_order), int(w_dem)))

    with s2:
        st.subheader("Pick a strategy and run a month")
        options = ["Always order a fixed amount", "Order up to a target stock level", "Random order"]
        if trained:
            options.append("🧠 The trained AI")
        strat = st.selectbox("Strategy", options)
        arg = None
        if strat.startswith("Always"):
            arg = st.slider("Order this many every day", 0, MAX_STOCK, 8)
            fn, label = order_fn("fixed", arg), f"Always order {arg}"
        elif strat.startswith("Order up"):
            arg = st.slider("Top the fridge up to", 0, MAX_STOCK, 12)
            fn, label = order_fn("upto", arg), f"Order up to {arg}"
        elif strat.startswith("Random"):
            fn, label = order_fn("random", None), "Random"
        else:
            fn, label = order_fn("policy", tuple(trained["policy"])), "Trained AI"
        if st.button("▶️ Run one month") or "sim_month" not in ss:
            ss["sim_month"] = (label, run_month(live_settings, fn))
        label, df = ss["sim_month"]
        st.markdown(f"**Result: {label}**")
        kpi_row(df)
        ch1, ch2 = st.columns(2)
        with ch1:
            st.caption("Customers wanted vs units sold, per day")
            st.line_chart(df.set_index("Day")[["Demand", "Sold"]])
        with ch2:
            st.caption("Cumulative profit (AED)")
            st.line_chart(pd.DataFrame({"Cumulative profit": df["Profit"].cumsum().values},
                                       index=df["Day"]))
        st.caption("Day-by-day ledger")
        st.dataframe(df, hide_index=True)

    with s3:
        st.subheader("You are the manager")
        st.write("Order each morning and watch the month unfold. Yesterday's unsold stock carries over.")

        def new_play():
            e = make_env(live_settings)
            e.reset()
            ss["play_env"] = e
            ss["play_log"] = []
            ss["play_total"] = 0.0
            ss["play_gen"] = ss.get("play_gen", 0) + 1

        if "play_env" not in ss:
            new_play()
        if st.button("🔄 Start a new month"):
            new_play()
        e = ss["play_env"]
        if e.day < e.days:
            st.write(f"**Day {e.day + 1} of {e.days}** — you have **{e.stock}** units in the fridge this morning. "
                     f"Total so far: **AED {ss['play_total']:,.0f}**")
            o = st.number_input("How many do you order?", 0, MAX_STOCK, 8, key=f"play_o_{ss['play_gen']}_{e.day}")
            if st.button("Open the shop ▶️", key=f"play_go_{ss['play_gen']}_{e.day}"):
                start = e.stock
                r, _, _, info = e.step(o)
                ss["play_total"] += r
                ss["play_log"].append(dict(Day=e.day, Start=start, Order=info["order"], Demand=info["demand"],
                                           Sold=info["sold"], Spoiled=info["spoiled"], Stockout=info["stockout"],
                                           Carry=info["carry"], Profit=info["profit"]))
                st.rerun()
        else:
            st.success(f"Month finished! Your profit: **AED {ss['play_total']:,.0f}**")
            if trained:
                ref = evaluate(settings_key(trained["settings"]), "policy", tuple(trained["policy"]))["mean"]
                st.caption(f"For comparison, the trained AI averages about AED {ref:,.0f} per month "
                           f"(under the settings it trained on).")
        if ss["play_log"]:
            st.dataframe(pd.DataFrame(ss["play_log"]), hide_index=True)

    with s4:
        st.subheader("Two ways to lose money")
        key = settings_key(live_settings)
        hi = evaluate(key, "fixed", MAX_STOCK, 300)
        lo = evaluate(key, "fixed", 1, 300)
        mid = evaluate(key, "fixed", 10, 300)
        cc = st.columns(3)
        for col, title, res, note in [
            (cc[0], f"Order {MAX_STOCK} every day", hi, "Over-ordering → waste 🗑️"),
            (cc[1], "Order 1 every day", lo, "Under-ordering → stockouts 😟"),
            (cc[2], "Order 10 every day", mid, "A decent fixed guess"),
        ]:
            with col:
                st.markdown(f"**{title}**")
                st.caption(note)
                st.metric("Avg monthly profit", f"AED {res['mean']:,.0f}")
                st.metric("Units spoiled / month", f"{res['spoiled_pm']:.0f}")
                st.metric("Customers turned away / month", f"{res['stockout_pm']:.0f}")
        st.info("A turned-away customer costs the AED sale you missed **plus** the stockout penalty. "
                "A spare unit costs far less, which is why under-ordering hurts more than over-ordering.")

# ---------------------------------------------------------------------
# TAB 2 — LEARNING AGENT
# ---------------------------------------------------------------------
with tab_agent:
    st.header("The Learning Agent")
    with st.expander("How the agent learns (Q-learning in 30 seconds)", expanded=not trained):
        st.markdown(
            "- The agent keeps a **Q-table**: one row per stock level (0–15), one column per possible order (0–15).\n"
            "- Each cell holds its current estimate of *how good it is to order that amount at that stock level*, "
            "counting future profit too.\n"
            "- **Explore / exploit (ε-greedy):** with probability ε it tries a random order; otherwise it picks the "
            "best-known order (`argmax`).\n"
            "- **Update:** `Q[s,a] += α · (reward + γ · max Q[next] − Q[s,a])`.\n"
            "- After thousands of simulated months, the best column in each row is the **policy**."
        )
    if not trained:
        st.info("👈 Set the business and the AI knobs in the sidebar, then press **🚀 Train the AI**.")
    else:
        Q, curve, policy, h = trained["Q"], trained["curve"], trained["policy"], trained["hyper"]
        st.caption(f"Trained for {h['episodes']:,} months (α={h['alpha']}, γ={h['gamma']}, "
                   f"ε decay={h['eps_decay']}) in {trained['seconds']:.1f}s.")

        st.subheader("1 · Did it actually learn?")
        w = max(1, len(curve) // 100)
        smooth = np.convolve(curve, np.ones(w) / w, mode="valid")
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.3), gridspec_kw=dict(width_ratios=[2, 1]))
        axes[0].plot(curve, color=BLUE, alpha=0.15, lw=0.6, label="each month")
        axes[0].plot(np.arange(len(smooth)) + w // 2, smooth, color=BLUE, lw=2, label=f"{w}-month average")
        axes[0].set_xlabel("Training month (episode)"); axes[0].set_ylabel("Profit (AED)")
        axes[0].set_title("Learning curve: should climb, then flatten"); axes[0].legend(loc="lower right")
        t = np.arange(len(curve))
        eps_curve = np.maximum(h["eps_min"], h["eps_start"] * h["eps_decay"] ** t)
        axes[1].plot(eps_curve, color=AMBER, lw=2)
        axes[1].set_xlabel("Training month"); axes[1].set_ylabel("ε (chance of exploring)")
        axes[1].set_title("Exploration fades as it learns")
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)
        m1, m2, m3 = st.columns(3)
        n0 = min(200, len(curve))
        m1.metric("Start (first 200 months)", f"AED {np.mean(curve[:n0]):,.0f}")
        m2.metric("End (last 200 months)", f"AED {np.mean(curve[-n0:]):,.0f}",
                  f"{np.mean(curve[-n0:]) - np.mean(curve[:n0]):+,.0f}")
        m3.metric("Policy states explored", f"{int((np.abs(Q).sum(axis=1) > 0).sum())} / {Q.shape[0]}")
        st.caption("Training profit includes the random exploration, so it understates what the finished "
                   "policy earns. See the Dashboard Insights tab for the true test.")

        st.subheader("2 · The Q-table and the strategy it discovered")
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        im = axes[0].imshow(Q, cmap="viridis", aspect="auto", origin="lower")
        axes[0].scatter(policy, np.arange(len(policy)), color="white", edgecolor="red", s=40, zorder=3,
                        label="best order")
        axes[0].set_xlabel("Order amount"); axes[0].set_ylabel("Stock on hand")
        axes[0].set_title("Q-table (brighter = better)"); axes[0].legend(loc="upper right")
        fig.colorbar(im, ax=axes[0], fraction=0.046)
        stocks = np.arange(len(policy))
        axes[1].bar(stocks, policy, color=BLUE, label="order")
        axes[1].plot(stocks, np.minimum(stocks + np.array(policy), MAX_STOCK), color=GREEN, marker="o",
                     label="stock after order")
        axes[1].set_xlabel("Stock on hand this morning"); axes[1].set_ylabel("Units")
        axes[1].set_title("Learned policy: stock → order"); axes[1].legend()
        axes[1].set_xticks(stocks)
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)
        st.caption("Look for the pattern: order a lot when stock is low, little or nothing when stock is high. "
                   "The green line shows the stock level the AI is effectively aiming for.")

        res = evaluate(settings_key(trained["settings"]), "policy", tuple(policy))
        pol_df = pd.DataFrame({"Stock": stocks, "AI orders": policy,
                               "Stock after order": np.minimum(stocks + np.array(policy), MAX_STOCK),
                               "How often this state occurs": [f"{v * 100:.1f}%" for v in res["visits"]]})
        with st.expander("Policy table (rarely-visited states are less reliable)"):
            st.dataframe(pol_df, hide_index=True)

        st.subheader("3 · Inspect one row of the brain")
        sel = st.slider("Stock on hand", 0, MAX_STOCK, 3)
        qrow = pd.DataFrame({"Q-value": Q[sel]}, index=pd.Index(range(Q.shape[1]), name="Order amount"))
        st.bar_chart(qrow)
        st.caption(f"With {sel} units on hand, the AI chooses to order **{policy[sel]}** (the tallest bar).")

# ---------------------------------------------------------------------
# TAB 3 — DASHBOARD INSIGHTS
# ---------------------------------------------------------------------
with tab_dash:
    st.header("Dashboard Insights")
    if not trained:
        st.info("👈 Train the AI first (sidebar) to unlock the insights.")
    else:
        tset = trained["settings"]
        tkey = settings_key(tset)
        policy = trained["policy"]
        if tset != live_settings:
            st.warning("The sidebar business settings differ from the ones the AI was trained on. "
                       "These insights use the **trained** settings. Press **Train the AI** to update.")

        # ---- KPIs
        rl = evaluate(tkey, "policy", tuple(policy), 600)
        base = {
            "Cautious (order 4)": evaluate(tkey, "fixed", 4),
            "Random": evaluate(tkey, "random", None),
            "Generous (order 10)": evaluate(tkey, "fixed", 10),
            "Rule of thumb (order up to 12)": evaluate(tkey, "upto", 12),
        }
        best_name = max(base, key=lambda k: base[k]["mean"])
        best_val = base[best_name]["mean"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("💰 Avg monthly profit", f"AED {rl['mean']:,.0f}",
                  f"{rl['mean'] - best_val:+,.0f} vs best baseline")
        c2.metric("✅ Service level (demand met)", f"{rl['service'] * 100:.0f}%")
        c3.metric("🗑️ Spoilage (% of units ordered)", f"{rl['spoil_rate'] * 100:.0f}%")
        c4.metric("🚫 Stockout rate (% of demand)", f"{rl['stockout_rate'] * 100:.0f}%")
        st.caption(f"Best baseline here: **{best_name}**. Averages over {len(rl['profits'])} simulated months.")

        i1, i2, i3, i4 = st.tabs(["🏆 Scoreboard", "📋 A month, day by day", "🎮 You vs AI",
                                  "🎚️ Experiments"])

        # ---- Scoreboard
        with i1:
            names = list(base.keys()) + ["RL Agent"]
            vals = [base[k]["mean"] for k in base] + [rl["mean"]]
            colors = [GREY] * len(base) + [GREEN]
            gr = ss.get("g_result")
            if gr:
                names.append("You (10-day game x 3)")
                vals.append(gr["you"] * 3)
                colors.append(AMBER)
            fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw=dict(width_ratios=[1.1, 1]))
            axes[0].barh(names, vals, color=colors)
            lo, hi = min(vals + [0]), max(vals)
            axes[0].set_xlim(lo - 0.1 * (hi - lo), hi + 0.18 * (hi - lo))
            for i, v in enumerate(vals):
                axes[0].text(v, i, f" {v:,.0f}", va="center")
            axes[0].axvline(0, color="black", lw=0.6)
            axes[0].set_xlabel("Average monthly profit (AED)"); axes[0].set_title("Who runs the better café?")
            bins = np.linspace(min(min(rl["profits"]), min(base["Cautious (order 4)"]["profits"])),
                               max(rl["profits"]) + 50, 40)
            axes[1].hist(base["Cautious (order 4)"]["profits"], bins=bins, color=RED, alpha=0.5, label="Cautious")
            axes[1].hist(base["Generous (order 10)"]["profits"], bins=bins, color=GREY, alpha=0.6, label="Generous")
            axes[1].hist(rl["profits"], bins=bins, color=GREEN, alpha=0.6, label="RL agent")
            axes[1].set_xlabel("Monthly profit (AED)"); axes[1].set_ylabel("Months")
            axes[1].set_title("Spread of outcomes"); axes[1].legend()
            fig.tight_layout(); st.pyplot(fig); plt.close(fig)
            if gr:
                st.caption("Your bar is your 10-day profit multiplied by 3 to approximate a 30-day month, "
                           "so treat it as a rough guide only.")
            table = pd.DataFrame([
                dict(Strategy=k, **{"Avg profit": f"{v['mean']:,.0f}", "Service": f"{v['service'] * 100:.0f}%",
                                    "Spoilage": f"{v['spoil_rate'] * 100:.0f}%",
                                    "Stockouts": f"{v['stockout_rate'] * 100:.0f}%"})
                for k, v in list(base.items()) + [("🧠 RL Agent", rl)]])
            st.dataframe(table, hide_index=True)
            st.caption("A 'cautious' manager who under-orders can actually lose money: stockouts cost more than waste.")

        # ---- Ledger
        with i2:
            if st.button("🎲 Simulate another month") or ss.get("ai_month") is None:
                ss["ai_month"] = run_month(tset, order_fn("policy", tuple(policy)))
            df = ss["ai_month"]
            kpi_row(df)
            fig, ax = plt.subplots(figsize=(11, 3.2))
            ax.bar(df["Day"] - 0.2, df["Order"], width=0.4, color=BLUE, label="Ordered")
            ax.bar(df["Day"] + 0.2, df["Demand"], width=0.4, color=GREY, label="Customers wanted")
            ax.plot(df["Day"], df["Start"], color=GREEN, marker="o", ms=3, label="Stock at open")
            ax.set_xlabel("Day"); ax.set_ylabel("Units"); ax.legend(ncol=3, loc="upper right")
            ax.set_title("The AI running the shop for one month")
            fig.tight_layout(); st.pyplot(fig); plt.close(fig)
            st.dataframe(df, hide_index=True)

        # ---- You vs AI
        with i3:
            st.write("Play 10 days as manager. Then the AI plays the **same** 10 days of demand.")
            if "g_demand" not in ss or st.button("🔄 New 10-day challenge"):
                g = np.random.default_rng()
                seq = []
                for _ in range(10):
                    d = g.poisson(tset["mean_demand"])
                    if g.random() < tset["busy_chance"]:
                        d += g.poisson(5)
                    seq.append(int(d))
                ss.update(g_demand=seq, g_day=0, g_stock=5, g_profit=0.0, g_log=[])
                ss.pop("g_result", None)
            seq, day = ss["g_demand"], ss["g_day"]
            if day < 10:
                st.write(f"**Day {day + 1} of 10** — you have **{ss['g_stock']}** units in stock this morning.")
                o = st.number_input("How many do you order today?", 0, MAX_STOCK, 7, key=f"g_ord{day}")
                if st.button("Open the shop ▶️", key=f"g_go{day}"):
                    p, carry, info = simulate_day(tset, ss["g_stock"], o, seq[day])
                    ss["g_profit"] += p
                    ss["g_stock"] = carry
                    ss["g_log"].append(dict(Day=day + 1, **info))
                    ss["g_day"] += 1
                    st.rerun()
                if ss["g_log"]:
                    st.dataframe(pd.DataFrame(ss["g_log"]), hide_index=True)
            else:
                ai_stock, ai_profit, ai_log = 5, 0.0, []
                for i, d in enumerate(seq):
                    p, ai_stock_new, info = simulate_day(tset, ai_stock, policy[int(np.clip(ai_stock, 0, MAX_STOCK))], d)
                    ai_log.append(dict(Day=i + 1, **info))
                    ai_profit += p
                    ai_stock = ai_stock_new
                you = ss["g_profit"]
                ss["g_result"] = dict(you=you, ai=ai_profit)
                a, b = st.columns(2)
                a.metric("🧍 Your total profit", f"AED {you:,.0f}")
                b.metric("🧠 AI's total profit", f"AED {ai_profit:,.0f}", f"{ai_profit - you:+,.0f} vs you")
                if ai_profit > you:
                    st.success("The AI ran the more profitable café. It never saw the answer, it learned it. 🤖")
                else:
                    st.balloons()
                    st.success("You beat the AI! Impressive instinct for the business. 🎉")
                la, lb = st.columns(2)
                la.markdown("**Your days**"); la.dataframe(pd.DataFrame(ss["g_log"]), hide_index=True)
                lb.markdown("**AI's days (same demand)**"); lb.dataframe(pd.DataFrame(ai_log), hide_index=True)
                st.caption("Ten days is short, so luck matters. The scoreboard shows averages over hundreds of months.")

        # ---- Experiments
        with i4:
            st.write("Every time you press **Train the AI**, the run is saved here. Change a slider "
                     "(spoilage cost, stockout penalty, demand…), retrain, and compare the strategies.")
            snaps = ss.get("snapshots", [])
            if snaps:
                tbl = pd.DataFrame([dict(Run=s["Run"], Setup=s["Setup"],
                                         **{"Avg profit": round(s["Profit"]),
                                            "Aims for stock of": round(s["Target"], 1),
                                            "Service": f"{s['Service'] * 100:.0f}%",
                                            "Spoilage": f"{s['Spoilage'] * 100:.0f}%",
                                            "Stockouts": f"{s['Stockouts'] * 100:.0f}%"}) for s in snaps])
                st.dataframe(tbl, hide_index=True)
                fig, ax = plt.subplots(figsize=(10, 3.6))
                cmap = plt.get_cmap("tab10")
                for i, s in enumerate(snaps):
                    ax.plot(range(len(s["Policy"])), s["Policy"], marker="o", ms=3, color=cmap(i % 10),
                            label=f"{s['Run']}  {s['Setup']}")
                ax.set_xlabel("Stock on hand"); ax.set_ylabel("Order amount")
                ax.set_title("How the strategy shifts when the business changes")
                ax.set_xticks(range(MAX_STOCK + 1))
                ax.legend(fontsize=7, loc="upper right")
                fig.tight_layout(); st.pyplot(fig); plt.close(fig)
                st.caption("Profits across different cost settings are not directly comparable "
                           "(a higher penalty makes the same café less profitable). Compare the **shape** of the policy.")
                if st.button("🧹 Clear saved runs"):
                    ss["snapshots"] = []
                    st.rerun()
            else:
                st.info("No runs saved yet.")
            st.markdown("**Things to try:** make spoilage expensive (does it order less?), raise the stockout "
                        "penalty (does it keep a bigger buffer?), raise demand (does the whole policy move up?). "
                        "The AI optimises exactly what you reward.")
