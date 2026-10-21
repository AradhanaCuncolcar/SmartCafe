# The Smart Café - Menu Book edition.
# Three tools in one app: the Café Simulator (cafe.py), the Learning Agent (agent.py)
# and this Dashboard, laid out as a menu book: one spread (two pages) per chapter, Back / Next to turn pages.
import random
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

st.set_page_config(page_title="The Smart Café - Menu Book", page_icon="☕", layout="wide")

try:
    from cafe import CafeSim, DAY_NAMES
    from agent import train, policy_from_Q, get_best_action, average_profit
except Exception as e:
    st.error(f"Could not import cafe.py / agent.py: {e}")
    st.stop()

ss = st.session_state

# ------------------------------------------------------------------ theme
INK, PAPER, GOLD, RED = "#3A2417", "#F6ECDA", "#7A4A18", "#8C2F1F"
ESPRESSO, CARAMEL, LATTE, CREAM = "#3E2723", "#C68B59", "#E8D5C0", PAPER

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@500;700&family=Lora:wght@400;600&display=swap');
.stApp { background: radial-gradient(ellipse at 50% 15%, #4A2E20 0%, #241109 85%); }
[data-testid="stHeader"] { background: transparent; }
section[data-testid="stSidebar"] { background: #E8D5C0; }
.cover { text-align:center; font-family:Lora,serif; color:#F3E3C8; letter-spacing:6px; text-transform:uppercase; font-size:15px; margin:4px 0 14px 0; }
.st-key-book { position:relative; border-radius:10px; box-shadow:0 30px 60px rgba(0,0,0,.55); padding:34px 44px 40px 44px;
  background: linear-gradient(to right,#E4D3B5 0%,#F6ECDA 5%,#F6ECDA 47.5%,#C9B48F 49.6%,#9C8560 50%,#C9B48F 50.4%,#F6ECDA 52.5%,#F6ECDA 95%,#E4D3B5 100%); }
.st-key-book::before { content:""; position:absolute; right:70px; top:-6px; width:30px; height:84px; background:#8C2F1F;
  clip-path:polygon(0 0,100% 0,100% 100%,50% 82%,0 100%); }
.st-key-book, .st-key-book p, .st-key-book label, .st-key-book li, .st-key-book span, .st-key-book div { font-family:Lora,serif; color:#3A2417; }
.st-key-book [data-testid="stMetricValue"] { font-family:'Playfair Display',serif; font-weight:700; }
.st-key-book [data-testid="stMetric"] { background:transparent; border:1.5px solid #7A4A18; border-radius:10px; padding:8px 14px; }
.st-key-book .stButton button { background:#3E2723; border:0; border-radius:22px; min-height:44px; font-weight:600; }
.st-key-book .stButton button p { color:#F6ECDA; }
.st-key-book .stButton button:hover { background:#8C2F1F; }
.ch { font-size:13px; letter-spacing:3px; text-transform:uppercase; color:#7A4A18 !important; margin-top:4px; }
.pt { font-family:'Playfair Display',serif !important; font-size:36px; margin:4px 0 2px 0; padding:0; color:#3A2417; }
.rule { height:2px; background:#7A4A18; width:90px; margin:10px 0 16px 0; }
.sub { font-style:italic; margin-bottom:10px; }
.folio { font-size:14px; color:#7A4A18 !important; margin-top:14px; }
.rec-card { background:#3E2723; border-radius:12px; padding:16px 22px; margin:10px 0 12px 0; }
.rec-card, .rec-card div, .rec-card span { color:#F6ECDA !important; }
.rec-card .big { font-family:'Playfair Display',serif; font-size:2.3rem; font-weight:700; color:#E2A96B !important; }
.rec-card .lbl { font-size:13px; letter-spacing:2px; text-transform:uppercase; }
.pill { display:inline-block; padding:2px 10px; border-radius:12px; background:#7A4A18; font-size:.85rem; margin-right:6px; }
.st-key-trainbox { background:#F6ECDA; border-radius:10px; padding:12px 18px; margin-bottom:12px; }
.st-key-nav_back button, .st-key-nav_next button { background:transparent !important; border:1.5px solid #D9A55E !important; border-radius:24px; min-height:48px; }
.st-key-nav_back button p, .st-key-nav_next button p { color:#F3E3C8 !important; font-family:Lora,serif; font-weight:600; }
.st-key-nav_back button:hover, .st-key-nav_next button:hover { background:#D9A55E !important; }
.st-key-nav_back button:hover p, .st-key-nav_next button:hover p { color:#2B1A12 !important; }
.st-key-nav_back button:disabled, .st-key-nav_next button:disabled { opacity:.35; }
.navmid { text-align:center; font-family:Lora,serif; color:#F3E3C8; padding-top:2px; }
.navmid .dots span { display:inline-block; height:9px; border-radius:5px; margin:0 4px; }
@media (max-width: 800px) { .st-key-book { background:#F6ECDA; padding:24px 18px; } }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ------------------------------------------------------------------ settings
DEFAULTS = dict(base_demand=7, spoilage_cost=1.5, stockout_penalty=6.0, weekend_boost=1.6, event_chance=0.15)
DEFAULT_EPISODES = 15000
GAME_DAYS = 10
CAUTIOUS, GENEROUS = 4, 10
RL = "RL agent"
PAGES = [("Chapter I", "Morning Briefing"), ("Chapter II", "Learning"), ("Chapter III", "The Strategy"),
         ("Chapter IV", "The Scoreboard"), ("Chapter V", "You vs the AI"), ("Chapter VI", "Experiments")]
PRESETS = {
    "Spoilage costs 3x more (1.5 → 5.0)": dict(spoilage_cost=5.0),
    "Unhappy customers cost 2x more (6 → 12)": dict(stockout_penalty=12.0),
    "A much busier café (demand 7 → 12)": dict(base_demand=12),
    "Events twice as likely (15% → 30%)": dict(event_chance=0.30),
}


def make_env(cfg):
    return CafeSim(**cfg)


def current_cfg():
    return {k: (int(ss[f"sl_{k}"]) if k == "base_demand" else float(ss[f"sl_{k}"])) for k in DEFAULTS}


def apply_values(values):
    for k, v in {**DEFAULTS, **values}.items():
        ss[f"sl_{k}"] = v
    ss["auto_train"] = True


def go(delta):
    ss["page"] = max(0, min(len(PAGES) - 1, ss["page"] + delta))


for k, v in DEFAULTS.items():
    ss.setdefault(f"sl_{k}", v)
ss.setdefault("sl_episodes", DEFAULT_EPISODES)
ss.setdefault("page", 0)
ss.setdefault("b_stock", 3); ss.setdefault("b_day", "Friday"); ss.setdefault("b_event", False)
ss.setdefault("g_order", 5); ss.setdefault("ledger", [])
# Only the current page's widgets exist, so re-save their values to keep them when you turn the page.
for _k in ("b_stock", "b_day", "b_event", "g_order"):
    ss[_k] = ss[_k]

# ------------------------------------------------------------------ sidebar: the café controls
st.sidebar.title("Café Controls")
st.sidebar.caption("Change the business, press **Train**, then turn to *Experiments* to see the strategy shift.")
st.sidebar.slider("Typical daily demand (customers)", 3, 12, key="sl_base_demand")
st.sidebar.slider("Spoilage cost per unit (AED)", 0.5, 6.0, step=0.5, key="sl_spoilage_cost")
st.sidebar.slider("Stockout penalty per customer (AED)", 0.0, 15.0, step=1.0, key="sl_stockout_penalty")
st.sidebar.slider("Weekend boost (Fri-Sun)", 1.0, 2.5, step=0.1, key="sl_weekend_boost")
st.sidebar.slider("Chance of a local event", 0.0, 0.5, step=0.05, key="sl_event_chance")
st.sidebar.select_slider("Training months", [5000, 10000, 15000, 30000], key="sl_episodes")
train_clicked = st.sidebar.button("Train the AI", use_container_width=True)
st.sidebar.button("Reset to the default café", on_click=apply_values, args=({},), use_container_width=True)


# ------------------------------------------------------------------ training
def train_model(cfg, episodes):
    env = make_env(cfg)
    bar = st.progress(0.0, text="The AI is practising in the simulator...")
    Q, curve = train(env, episodes=episodes,
                     progress=lambda f: bar.progress(min(f, 1.0), text=f"The AI is practising... {int(f * 100)}%"))
    bar.empty()
    return dict(cfg=cfg, episodes=episodes, Q=Q, curve=curve)


st.markdown('<div class="cover">The Smart Café &middot; Menu Book</div>', unsafe_allow_html=True)

if "model" not in ss or train_clicked or ss.pop("auto_train", False):
    cfg_now = current_cfg()
    with st.container(key="trainbox"):
        ss["model"] = train_model(cfg_now, int(ss["sl_episodes"]))
    if "baseline" not in ss or cfg_now == DEFAULTS:
        ss["baseline"] = ss["model"]

model, baseline = ss["model"], ss["baseline"]
cfg = model["cfg"]
env = make_env(cfg)
policy = policy_from_Q(model["Q"])
MAX = env.max_stock

if current_cfg() != cfg:
    st.sidebar.warning("Sliders changed - press **Train the AI** to apply them.")
else:
    st.sidebar.success(f"AI trained on {model['episodes']:,} simulated months.")


# ------------------------------------------------------------------ helpers
def get_eval(m):
    if "eval" not in m:
        e = make_env(m["cfg"])
        pol = policy_from_Q(m["Q"])
        strategies = {
            RL: lambda s: int(pol[s]),
            f"Generous (always {GENEROUS})": lambda s: GENEROUS,
            f"Cautious (always {CAUTIOUS})": lambda s: CAUTIOUS,
            "Random": lambda s: random.randint(0, e.max_stock),
        }
        res = {}
        for name, fn in strategies.items():
            mean, sd, stt = average_profit(e, fn, months=300)
            res[name] = dict(mean=mean, std=sd, service=stt["sold"] / max(stt["demand"], 1),
                             spoil=stt["spoiled"] / max(stt["ordered"], 1))
        sweep = [average_profit(e, lambda s, a=a: a, months=100)[0] for a in range(e.max_stock + 1)]
        m["eval"] = dict(res=res, sweep=sweep)
    return m["eval"]


def style_ax(ax):
    ax.set_facecolor(PAPER)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(alpha=0.25)


def new_fig(w=5.6, h=3.4):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(PAPER)
    style_ax(ax)
    return fig, ax


def show(fig):
    st.pyplot(fig)
    plt.close(fig)


def head(chapter, title, sub=None):
    html = f'<div class="ch">{chapter}</div><h2 class="pt">{title}</h2><div class="rule"></div>'
    if sub:
        html += f'<div class="sub">{sub}</div>'
    st.markdown(html, unsafe_allow_html=True)


def folio(n):
    st.markdown(f'<div class="folio">{n}</div>', unsafe_allow_html=True)


# ------------------------------------------------------------------ actions (callbacks)
def submit_order(stock, dow, event, rec, order):
    e = make_env(cfg)
    demand = e.draw_demand(dow, event)           # same customers for you and the AI
    mine = e.simulate_day(stock, order, demand)
    ai = e.simulate_day(stock, rec, demand)
    ss["ledger"].append({
        "Day": DAY_NAMES[dow], "Event": "Yes" if event else "No", "Start stock": stock,
        "AI suggested": rec, "You ordered": mine["order"], "Customers": demand, "Sold": mine["sold"],
        "Spoiled": mine["spoiled"], "Missed": mine["stockout"],
        "Your profit": mine["profit"], "AI profit": ai["profit"], "Overrode AI?": "Yes" if rec != mine["order"] else "No"})
    ss["b_stock"] = mine["carry"]                         # today's leftover is tomorrow's stock
    ss["b_day"] = DAY_NAMES[(dow + 1) % 7]
    ss["b_event"] = False


def new_game():
    g_env = make_env(cfg)
    start = random.randint(0, 6)
    script = []
    for i in range(GAME_DAYS):
        d = (start + i) % 7
        e_ = 1 if random.random() < g_env.event_chance else 0
        script.append(dict(dow=d, event=e_, demand=g_env.draw_demand(d, e_)))
    stock, ai_rows = 5, []                       # the AI plays the exact same days, hidden until the end
    for day in script:
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


# ================================================================== CHAPTER I: Morning Briefing
def page_briefing(L, R):
    with L:
        head("Chapter I", "Morning Briefing", "Tell the kitchen what is on the shelf.")
        stock_in = st.number_input("Leftover stock (units still fresh)", 0, MAX, key="b_stock")
        day_in = st.selectbox("Day of the week", DAY_NAMES, key="b_day")
        event_in = st.checkbox("Local event / high footfall expected today", key="b_event")
        dow_i, evt_i = DAY_NAMES.index(day_in), int(event_in)
        rec = get_best_action(model["Q"], int(stock_in), dow_i, evt_i)
        exp_d = env.expected_demand(dow_i, evt_i)
        after = min(int(stock_in) + rec, MAX)
        st.markdown(f"""<div class="rec-card"><div class="lbl">The chef recommends</div>
            <div class="big">Order {rec} units</div>
            <span class="pill">expect about {exp_d:.0f} customers</span><span class="pill">{after} units on the shelf</span></div>""",
                    unsafe_allow_html=True)
        if exp_d > MAX:
            st.warning(f"Expected demand ({exp_d:.0f}) is more than the fridge holds ({MAX}). "
                       "Even a full fridge may run out - plan a mid-day top-up if you can.")
        qrow = model["Q"][int(stock_in), dow_i, evt_i]
        if np.all(qrow == 0):
            st.info("The AI rarely saw this exact situation in training, so treat its advice with care.")
        order_in = st.number_input("Your order (change it to override the AI)", 0, MAX, value=rec)
        st.button("Submit order and close the day", on_click=submit_order,
                  args=(int(stock_in), dow_i, evt_i, rec, int(order_in)), use_container_width=True)
        folio(1)
    with R:
        head("Chapter I &middot; continued", "The Day's Ledger")
        fig, ax = new_fig(5.4, 2.3)
        ax.bar(range(MAX + 1), qrow, color=[CARAMEL if a == rec else LATTE for a in range(MAX + 1)])
        ax.set_xlabel("Units ordered"); ax.set_ylabel("AI's score")
        ax.set_title("How the AI rates each order (taller is better)", fontsize=10, color=INK)
        show(fig)
        if ss["ledger"]:
            df = pd.DataFrame(ss["ledger"])
            a, b = st.columns(2)
            a.metric("Your total profit", f"AED {df['Your profit'].sum():,.0f}")
            b.metric("AI on same days", f"AED {df['AI profit'].sum():,.0f}",
                     delta=f"{df['Your profit'].sum() - df['AI profit'].sum():,.0f} you vs AI")
            c, d = st.columns(2)
            c.metric("Spoiled units", int(df["Spoiled"].sum()))
            d.metric("Missed customers", int(df["Missed"].sum()))
            st.dataframe(df.tail(8), hide_index=True)
            last = ss["ledger"][-1]
            st.caption(f"Last day: AED {last['Your profit']:,.1f} (the AI would have made AED {last['AI profit']:,.1f}). "
                       "Tomorrow's stock and weekday are filled in for you.")
            st.button("Clear the ledger", on_click=lambda: ss.update(ledger=[]))
        else:
            st.info("The ledger is empty. Submit your first morning order to start tracking.")
        folio(2)


# ================================================================== CHAPTER II: Learning
def page_learning(L, R):
    curve = np.array(model["curve"])
    w = max(50, len(curve) // 60)
    ma = np.convolve(curve, np.ones(w) / w, mode="valid")
    with st.spinner("Measuring the trained agent..."):
        ev = get_eval(model)
    rl = ev["res"][RL]
    start_avg, end_avg = curve[:w].mean(), curve[-w:].mean()
    with L:
        head("Chapter II", "Learning", "Proof that the AI is really learning.")
        fig, ax = new_fig(5.6, 3.8)
        ax.plot(curve, color=CARAMEL, alpha=0.25, lw=0.6, label="each practice month")
        ax.plot(np.arange(w - 1, len(curve)), ma, color=ESPRESSO, lw=2, label=f"average of {w} months")
        ax.set_xlabel("Training month"); ax.set_ylabel("Profit per month (AED)"); ax.legend(frameon=False)
        show(fig)
        st.caption("The curve is noisy and climbs gradually - that is normal. Early on the AI explores with random orders; "
                   "later it mostly uses what it learned.")
        folio(3)
    with R:
        head("Chapter II &middot; continued", "How Well It Does")
        a, b = st.columns(2)
        a.metric("Profit at the start", f"AED {start_avg:,.0f}")
        b.metric("Profit at the end", f"AED {end_avg:,.0f}", delta=f"{end_avg - start_avg:,.0f}")
        c, d = st.columns(2)
        c.metric("Trained average / month", f"AED {rl['mean']:,.0f}")
        d.metric("Service level", f"{rl['service'] * 100:.0f}%")
        e_, _ = st.columns(2)
        e_.metric("Spoilage rate", f"{rl['spoil'] * 100:.0f}%")
        st.markdown("**Service level** is the share of customers who got what they wanted. **Spoilage rate** is the share of "
                    "ordered units thrown away. The trained average is measured over 300 fresh months with no exploring.")
        folio(4)


# ================================================================== CHAPTER III: Strategy
def page_strategy(L, R):
    with L:
        head("Chapter III", "The Strategy", "Nobody wrote this recipe - the AI found it by practice.")
        view = st.radio("Show the plan for", ["Normal days", "Event days"], horizontal=True)
        grid = policy[:, :, 1 if view.startswith("Event") else 0]
        cmap = LinearSegmentedColormap.from_list("cafe", [PAPER, CARAMEL, ESPRESSO])
        fig, ax = plt.subplots(figsize=(5.6, 5.2))
        fig.patch.set_facecolor(PAPER)
        ax.imshow(grid, cmap=cmap, aspect="auto", vmin=0, vmax=MAX)
        for r in range(grid.shape[0]):
            for c in range(grid.shape[1]):
                ax.text(c, r, int(grid[r, c]), ha="center", va="center", fontsize=8,
                        color="white" if grid[r, c] > MAX * 0.55 else INK)
        ax.set_xticks(range(7)); ax.set_xticklabels([d[:3] for d in DAY_NAMES])
        ax.set_yticks(range(MAX + 1)); ax.set_ylabel("Stock on hand in the morning"); ax.set_title("Units to order", color=INK)
        show(fig)
        folio(5)
    with R:
        head("Chapter III &middot; continued", "How Full to Fill the Fridge")
        rows = []
        for d in range(7):
            row = {"Day": DAY_NAMES[d]}
            for e_, label in ((0, "Normal day"), (1, "Event day")):
                target = np.mean([min(s + int(policy[s, d, e_]), MAX) for s in range(0, 5)])
                row[label] = f"top up to about {target:.0f}"
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        st.markdown("Read the heatmap as: *it's Friday, I have 3 units left - order this many.* Low stock means order plenty, "
                    "a full shelf means order little or nothing, and weekends and events call for bigger orders.")
        st.caption("Some high-stock cells look odd because the AI rarely visits them in training, so those values are less "
                   "reliable - normal for tabular Q-learning.")
        folio(6)


# ================================================================== CHAPTER IV: Scoreboard
def page_scoreboard(L, R):
    with st.spinner("Running each manager through hundreds of simulated months..."):
        ev = get_eval(model)
    human = ss.get("human")
    board = {k: v["mean"] for k, v in ev["res"].items()}
    if human:
        board["You (10-day pace x3)"] = human["per_day"] * 30
    names = list(board.keys()); vals = [board[n] for n in names]
    with L:
        head("Chapter IV", "The Scoreboard", "Who ran the better café?")
        fig, ax = new_fig(5.6, 3.8)
        bars = ax.barh(names, vals, color=[CARAMEL if n == RL else (RED if v < 0 else LATTE) for n, v in zip(names, vals)])
        ax.axvline(0, color=ESPRESSO, lw=0.8); ax.invert_yaxis(); ax.set_xlabel("Average profit per 30-day month (AED)")
        for b, v in zip(bars, vals):
            ax.text(v + (8 if v >= 0 else -8), b.get_y() + b.get_height() / 2, f"{v:,.0f}",
                    va="center", ha="left" if v >= 0 else "right", color=INK, fontsize=9)
        show(fig)
        if not human:
            st.info("Play the *You vs the AI* chapter to add yourself to this scoreboard.")
        folio(7)
    with R:
        head("Chapter IV &middot; continued", "The Small Print")
        tbl = pd.DataFrame([{"Manager": n, "Avg monthly profit (AED)": round(v["mean"]), "Swing (±)": round(v["std"]),
                             "Service": f"{v['service'] * 100:.0f}%", "Spoilage": f"{v['spoil'] * 100:.0f}%"}
                            for n, v in ev["res"].items()])
        st.dataframe(tbl, hide_index=True)
        with st.expander("Is there a best fixed order?"):
            fig, ax = new_fig(5, 2.8)
            ax.plot(range(MAX + 1), ev["sweep"], marker="o", color=ESPRESSO)
            ax.axhline(ev["res"][RL]["mean"], color=CARAMEL, ls="--", label="RL agent")
            ax.set_xlabel("Fixed daily order"); ax.set_ylabel("Avg monthly profit (AED)"); ax.legend(frameon=False)
            show(fig)
            st.caption("There is a sweet spot for a single fixed number - but nobody knows it in advance, and it can't react "
                       "to weekends, events or leftover stock. The AI finds it, then goes beyond it.")
        folio(8)


# ================================================================== CHAPTER V: You vs the AI
def page_game(L, R):
    if ss.get("game") is None:
        new_game()
    g = ss["game"]
    done = len(g["you"]) == GAME_DAYS
    with L:
        head("Chapter V", "You vs the AI", f"{GAME_DAYS} days as café manager. Same customers, same days - who earns more?")
        if not done:
            i = len(g["you"]); day = g["script"][i]
            st.progress(i / GAME_DAYS, text=f"Day {i + 1} of {GAME_DAYS}")
            st.markdown(f"### {DAY_NAMES[day['dow']]}" + ("  -  local event today!" if day["event"] else ""))
            st.metric("Stock on hand", g["stock"])
            st.number_input("How many units do you order?", 0, MAX, key="g_order")
            st.button("Place order", on_click=place_order, use_container_width=True)
            st.caption("Hint: leftover stock counts, weekends and events are busier, and the fridge holds 15.")
        else:
            you_total = sum(x["profit"] for x in g["you"]); ai_total = sum(x["profit"] for x in g["ai"])
            a, b = st.columns(2)
            a.metric("You", f"AED {you_total:,.0f}")
            b.metric("The AI", f"AED {ai_total:,.0f}", delta=f"{you_total - ai_total:,.0f} you vs AI")
            if you_total > ai_total:
                st.success("You beat the AI this round! With only 10 days luck matters a lot - try again.")
            else:
                st.info("The AI won this round. The table shows where the orders differed.")
            st.caption("Your result is now on the Scoreboard.")
            st.button("Play again", on_click=new_game, use_container_width=True)
        folio(9)
    with R:
        head("Chapter V &middot; continued", "The Order Book")
        if not done:
            if g["you"]:
                st.dataframe(pd.DataFrame(g["you"])[["start", "order", "demand", "sold", "spoiled", "stockout", "profit"]]
                             .rename(columns={"start": "Start stock", "order": "You ordered", "demand": "Customers",
                                              "sold": "Sold", "spoiled": "Spoiled", "stockout": "Missed", "profit": "Profit"}),
                             hide_index=True)
            else:
                st.info("Your orders will be written here, day by day.")
        else:
            rows = []
            for y, a_, d in zip(g["you"], g["ai"], g["script"]):
                rows.append({"Day": DAY_NAMES[d["dow"]] + (" *" if d["event"] else ""), "Customers": d["demand"],
                             "You: stock": y["start"], "You: order": y["order"], "You: profit": y["profit"],
                             "AI: stock": a_["start"], "AI: order": a_["order"], "AI: profit": a_["profit"]})
            st.dataframe(pd.DataFrame(rows), hide_index=True)
            st.caption("* marks a local-event day.")
        folio(10)


# ================================================================== CHAPTER VI: Experiments
def page_experiments(L, R):
    same = cfg == baseline["cfg"]
    with L:
        head("Chapter VI", "Experiments", "What you reward is what you get.")
        st.write("Pick an experiment (it sets the sliders and retrains the AI), or use the sidebar sliders and press **Train**.")
        for i, name in enumerate(PRESETS):
            st.button(name, on_click=apply_values, args=(PRESETS[name],), key=f"preset_{i}", use_container_width=True)
        folio(11)
    with R:
        head("Chapter VI &middot; continued", "The Result")
        if same:
            st.info("The AI is trained on the default café. Choose an experiment to see the strategy shift.")
        else:
            changes = [f"**{k.replace('_', ' ')}**: {baseline['cfg'][k]} → {cfg[k]}" for k in DEFAULTS if cfg[k] != baseline["cfg"][k]]
            st.markdown("Changed: " + " · ".join(changes))
            base_pol = policy_from_Q(baseline["Q"])

            def avg_order(pol, p_event):
                wts = np.array([1 - p_event, p_event])
                return np.array([(pol[s].mean(axis=0) * wts).sum() for s in range(MAX + 1)])

            base_curve = avg_order(base_pol, baseline["cfg"]["event_chance"])
            new_curve = avg_order(policy, cfg["event_chance"])
            fig, ax = new_fig(5.4, 3.0)
            ax.plot(range(MAX + 1), base_curve, marker="o", color=CARAMEL, lw=2.5, label="Original café")
            ax.plot(range(MAX + 1), new_curve, marker="o", color=ESPRESSO, lw=2.5, label="Changed café")
            ax.set_xlabel("Stock on hand in the morning"); ax.set_ylabel("Average units ordered"); ax.legend(frameon=False)
            show(fig)
            low_b, low_n = base_curve[:4].mean(), new_curve[:4].mean()
            st.markdown(f"With low stock (0-3 units) the AI now orders **{low_n:.1f}** units on average, versus **{low_b:.1f}** "
                        f"originally - **{abs(low_n - low_b):.1f} {'more' if low_n > low_b else 'less'}**.")
            if "base_in_new" not in model:
                model["base_in_new"] = average_profit(env, lambda s: int(base_pol[s]), months=200)[0]
            new_profit = get_eval(model)["res"][RL]["mean"]
            a, b = st.columns(2)
            a.metric("Old strategy, changed café", f"AED {model['base_in_new']:,.0f}")
            b.metric("Retrained strategy", f"AED {new_profit:,.0f}", delta=f"{new_profit - model['base_in_new']:,.0f}")
            st.caption("This is why you retrain when costs change: the AI optimises exactly the rewards you give it. "
                       "Raise spoilage cost and it orders less; raise the stockout penalty and it keeps a bigger safety buffer.")
        folio(12)


PAGE_FUNCS = [page_briefing, page_learning, page_strategy, page_scoreboard, page_game, page_experiments]

# ------------------------------------------------------------------ the book + page turning
with st.container(key="book"):
    left_page, right_page = st.columns(2, gap="large")
    PAGE_FUNCS[ss["page"]](left_page, right_page)

n = ss["page"]
dots = "".join(f'<span style="width:{14 if k == n else 9}px;background:{"#D9A55E" if k == n else "#6B4A38"}"></span>'
               for k in range(len(PAGES)))
st.write("")
back_col, mid_col, next_col = st.columns([1, 2, 1], vertical_alignment="center")
back_col.button("← Back", key="nav_back", on_click=go, args=(-1,), disabled=n == 0, use_container_width=True)
mid_col.markdown(f'<div class="navmid"><div class="dots">{dots}</div>'
                 f'<div style="margin-top:8px">Spread {n + 1} of {len(PAGES)} &middot; {PAGES[n][1]}</div></div>',
                 unsafe_allow_html=True)
next_col.button("Next →", key="nav_next", on_click=go, args=(1,), disabled=n == len(PAGES) - 1, use_container_width=True)
