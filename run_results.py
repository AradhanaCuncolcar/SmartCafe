import random, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from cafe import CafeSim
from agent import train, policy_from_Q

def average_profit(env, policy, months=500):
    totals = []
    for _ in range(months):
        s = env.reset(); done = False; total = 0
        while not done:
            r, s, done, _ = env.step(policy(s)); total += r
        totals.append(total)
    return sum(totals) / len(totals)

print("== Part 2 hand test ==")
env = CafeSim(); s = env.reset()
for order in [8, 8, 2, 10]:
    r, s, done, info = env.step(order); print(info)

print("\n== P2: failure modes ==")
for o in (15, 1):
    env = CafeSim(); env.reset(); tot=0; sp=0; so=0
    for _ in range(30):
        r,_,_,i = env.step(o); tot+=r; sp+=i['spoiled']; so+=i['stockout']
    print(f"order {o}/day: profit {tot:.0f}, spoiled {sp}, stockout {so}")

print("\n== Part 3 baselines ==")
env = CafeSim()
base = {"Cautious(4)": average_profit(env, lambda s: 4),
        "Generous(10)": average_profit(env, lambda s: 10),
        "Random": average_profit(env, lambda s: random.randint(0, 15))}
for k, v in base.items(): print(k, round(v, 1))

print("\n== Part 4 training ==")
random.seed(0); np.random.seed(0)
env = CafeSim()
Q, curve = train(env, episodes=15000)
pol = policy_from_Q(Q)
print("policy (stock 0..15 -> order):", list(map(int, pol)))
rl = average_profit(env, lambda s: int(pol[s]), months=2000)
print("RL avg monthly profit:", round(rl, 1))
print("curve start/end:", round(np.mean(curve[:200]),1), round(np.mean(curve[-200:]),1))
best_naive = max(base.values())
print("gain vs best naive: %.1f (%.0f%%)" % (rl-best_naive, 100*(rl-best_naive)/abs(best_naive)))

w = 150
plt.figure(figsize=(8,3.2)); plt.plot(np.convolve(curve, np.ones(w)/w, mode='valid'), color="#2563eb")
plt.xlabel("Month (episode)"); plt.ylabel("Profit (AED)"); plt.title("Learning curve"); plt.tight_layout()
plt.savefig("learning_curve.png", dpi=130)
fig, ax = plt.subplots(figsize=(8,1.6)); ax.imshow(pol.reshape(1,-1), cmap="YlGnBu", aspect="auto")
for i,v in enumerate(pol): ax.text(i,0,str(int(v)),ha="center",va="center")
ax.set_yticks([]); ax.set_xticks(range(16)); ax.set_xlabel("Stock on hand"); ax.set_title("Learned policy"); plt.tight_layout()
plt.savefig("policy.png", dpi=130)
plt.figure(figsize=(8,3.2))
names=list(base)+["RL Agent"]; vals=list(base.values())+[rl]
plt.barh(names, vals, color=["#94a3b8"]*3+["#16a34a"])
for i,v in enumerate(vals): plt.text(v,i,f" {v:,.0f}",va="center")
plt.xlabel("Avg monthly profit (AED)"); plt.title("Scoreboard"); plt.tight_layout(); plt.savefig("scoreboard.png", dpi=130)

print("\n== Part 7 experiments ==")
def exp(label, **kw):
    random.seed(2); e = CafeSim(**kw); Q,_ = train(e, episodes=15000); p = policy_from_Q(Q)
    print(f"{label:28s}", list(map(int,p)), "profit", round(average_profit(e, lambda s:int(p[s]), 1000),1))
exp("baseline")
exp("spoilage cost 5.0", spoilage_cost=5.0)
exp("stockout penalty 12", stockout_penalty=12.0)
exp("stockout penalty 0", stockout_penalty=0.0)
exp("mean demand 12", mean_demand=12)
