# ☕ The Smart Café
## Understand the Problem First — (read this before you write any code)

> This document has **no code**. Its only job is to make sure you *completely understand the business situation and your mission* before you build anything. Read it slowly. If you can explain this café to a friend who knows nothing about AI, you're ready.

---

## 1. The Situation

You run a small **café** that sells one popular **fresh item** — let's say **fresh milk** (it could be pastries, sandwiches, juice… anything perishable). Every single **morning**, before the shop opens, you must make **one decision**:

> **"How many units should I order for today?"**

That's it. One number, every morning. Sounds easy. It isn't — and here's why.

### What makes it hard

**The item is perishable.** Whatever you don't sell starts to go off. Leftover stock loses value overnight — some of it spoils and must be thrown away.

**Customers are unpredictable.** Some days are quiet, some days are busy. You don't know this morning exactly how many customers will walk in today.

**Both mistakes cost you money — in different ways:**
- **Order too much** → customers don't buy it all → leftover stock **spoils** → you paid for milk you threw in the bin. 💸 (waste)
- **Order too little** → you **run out** → customers who wanted milk leave empty-handed → **lost sales and unhappy customers.** 😟 (stockout)

So every morning you're walking a tightrope: **order enough to meet demand, but not so much that it spoils.** And because you never know exactly how busy today will be, there is **no single "correct" number** that's always right.

### One more twist — yesterday affects today

A little bit of yesterday's unsold stock is **still good this morning** (it hasn't fully spoiled yet). So today's leftover becomes part of tomorrow's starting stock. That means **today's decision changes tomorrow's situation.** You're not making one isolated choice — you're making a *chain* of choices over time, each one affecting the next.

---

## 2. How the Shop Works (one day, start to finish)

Picture a single day in the café:

1. **Morning — you have some stock on hand** (leftover from yesterday that's still fresh).
2. **You decide how much to order.** The order arrives and adds to your stock. *(You can't store more than a fixed maximum — the fridge is only so big.)*
3. **The day happens — customers arrive.** How many want the item is **random** (some days more, some days fewer).
4. **You sell** as much as you can — but only up to what you have in stock.
   - If customers wanted *more* than you had → the extra is a **stockout** (lost sales + a penalty for the unhappy customers).
5. **The shop closes.** Whatever is left over:
   - costs you a little to have kept it on the shelf (**holding cost**),
   - and **part of it spoils overnight** (thrown away — a **spoilage cost**),
   - the rest is **still fresh enough to carry into tomorrow.**
6. **You tally the day's profit:**

> **Profit = money from sales − cost of ordering − holding cost − spoilage cost − stockout penalty**

Then tomorrow begins, with your carried-over stock, and you decide again. Over a **month (30 days)**, all those daily profits add up. **Your goal is to make the total monthly profit as high as possible.**

---

## 3. The Real Business Question

Right now, the café manager just **guesses** the order each day. Some managers are **cautious** (order a little, to avoid waste) — but they keep running out and losing sales. Some are **generous** (order a lot, to never run out) — but they throw away spoiled stock every night. Nobody really knows the right amount, because it depends on demand that keeps changing.

**Your mission:** instead of guessing, you will create an **Artificial Intelligence that learns the best ordering strategy on its own** — not by being told the answer, but by **trying different amounts over thousands of simulated days and learning from the profit (or loss) it earns.**

This is **Reinforcement Learning (RL):** an AI that learns by *doing*, guided only by rewards and consequences — exactly like a human learns a new skill through practice and feedback, not from a manual.

---

## 4. What You Are Going to Build

By the end, you will have built **three things** — a complete, working business tool:

1. **The Café Simulator** 🏪 — a program that *is* the shop. It models the demand, the sales, the spoilage, and the profit for any order you make. **This is the business world, in code — and you build it.**

2. **The Learning Agent** 🧠 — the AI "manager" that learns, through trial and error over thousands of simulated days, the best amount to order for each situation.

3. **The Dashboard** 📊 — a visual window where you *watch* the AI learn: its profit climbing over time, the strategy it discovered, and a head-to-head **"You vs the AI"** game where **you** play shop manager for a few days and then see if the AI can beat you.

---

## 5. What Success Will Look Like

When it works, you will be able to **see**, clearly, on your dashboard:

- 📈 **A learning curve that climbs** — the AI's profit rising from "clueless" to "competent" as it practises. *Proof it's actually learning.*
- 🗺️ **The strategy it discovered** — a simple, readable rule: *"when I have little stock, order a lot; when I already have plenty, order little or nothing."* The AI will work this out **by itself** — nobody programs it in.
- 🏆 **A scoreboard** — the AI's monthly profit vs. the cautious manager, the generous manager, the random manager, and **you**. You'll see who ran the better café.
- 🎚️ **Living experiments** — change the business (make spoilage more expensive, make customers busier) and watch the AI's strategy *shift* in response. This reveals the deepest lesson of all.

---

## 6. The Big Ideas You'll Walk Away With

1. **An AI can learn a strategy from nothing but rewards** — no labelled "right answers", just the profit and loss it experiences.
2. **It must balance "trying new things" against "doing what works"** — explore to discover, exploit to profit.
3. **What you reward is what you get** — change the costs of the business, and the AI's strategy changes to match. This is one of the most important (and sometimes dangerous) truths in all of AI.

---

## 7. Check Yourself Before Moving On

Before you open the build guide, make sure you can answer these **in your own words** (no code needed):

1. What is the **one decision** the café manager makes each day?
2. What are the **two different ways** a wrong decision costs money?
3. Why is there **no single correct number** to order every day?
4. How does **today's** leftover stock affect **tomorrow**?
5. In one sentence: **what will the AI be trying to maximise**, and **how will it learn** to do that?

If those five are clear, you understand the problem. **Now go build it.** 🚀

> ➡️ *Next: open `02_Activity_Guide.md` to start building the simulator, the agent, and the dashboard.*
