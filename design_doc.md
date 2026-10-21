# Design Doc — Draw-Along Guide

> This doc is a set of drawing instructions. Follow each step and you'll end up with a full picture of the system. Every shape has a name, a position, and arrows in or out. You don't need to know what any of it means to draw it — the names are just labels.

---

## Setup Before You Draw

- Get a big piece of paper (or a whiteboard tool). Landscape.
- Divide it mentally into **three horizontal bands**:
  - **Bottom band:** where data comes in.
  - **Middle band:** where the agent plays and learns (this is the loop).
  - **Top band:** where results come out.
- Use a **thick pen for shapes**, a **thin pen for arrows**, and **coloured pens for labels** if you have them.

---

## Picture 1 — The Whole Pipeline

This is the "you-are-here" picture. It shows the whole system at a glance. Later pictures zoom in.

### Step 1 — the data source
- **Bottom-left of the page.** Draw a small **cloud shape**. Label it **"Yahoo Finance"**.
- To the right of the cloud, draw two small **stacked rectangles** touching each other. Label the top one **"vix.csv"** and the bottom one **"sp500.csv"**.
- Draw an arrow from the cloud to the rectangles. Label the arrow **"download once"**.

### Step 2 — the loader
- To the right of the two CSVs, draw a **larger rectangle**. Label it **"Loader + Transform"**.
- Draw an arrow from the CSVs into this rectangle.
- On the arrow, write **"reads + aligns dates"**.
- Out of the right side of this rectangle, draw a **thick arrow** pointing right. Label it **"(observations, returns, dates)"**.

### Step 3 — the env
- To the right of the Loader box, draw a **big square**. Make it about twice the size of the Loader. Label it **"Environment (TimeSeries)"**.
- Point the thick arrow from Step 2 into this square.

### Step 4 — the memory wrapper
- Draw a **hollow ring** around the outside of the Env square (like a picture frame around a photo). Label the ring **"Memory Wrapper (Frame Stack OR Adaptive Stack)"**.
- This ring is not a separate box — it *hugs* the Env square. The label goes on the ring itself.

### Step 5 — the agent
- Above the Env+Ring shape, draw a **circle**. Label the circle **"PPO Agent"**.
- Draw a **two-headed arrow** between the Circle and the Ring:
  - Arrow-down (from PPO to Ring): label **"action"**.
  - Arrow-up (from Ring to PPO): label **"observation + reward"**.
- This is the **loop**. It runs many times per second while the agent is learning.

### Step 6 — the results
- Above the PPO circle, draw **three small stacked rectangles**. From top to bottom label them:
  - **"summary.json"**
  - **"windows.csv"**
  - **"steps.csv"**
- Draw one arrow from the PPO circle up into these three rectangles. Label the arrow **"writes after training"**.

### Step 7 — the config
- All the way on the **left side of the page**, draw a **small envelope shape**. Label it **"YAML Config (recipe)"**.
- Draw a **dashed arrow** from the envelope reaching across the whole page to *everything*. Label the arrow **"tells each piece its settings"**.

**You should now have:**
```
                            [ summary.json ]
                            [ windows.csv  ]
                            [ steps.csv    ]
                                  ▲
                            [   PPO Agent   ]  (circle)
                                  ▲  action
                                  │ ↕
                                  │  observation + reward
                            [ Memory Wrapper (ring) ]
                            [ ─────────────────── ]
                            [   Environment       ]  (square inside ring)
                                  ▲
                     (obs, returns, dates)
                                  │
[Yahoo Finance] → [csv][csv] → [Loader + Transform] ──────┘

[ YAML Config ]  ── ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ (dashed, reaches everything)
```

---

## Picture 2 — What Happens Inside the Loop

You already drew the two-headed arrow between the PPO circle and the Ring in Picture 1. Now zoom in on that loop.

### Step 1 — split the day into two sides
- On a fresh part of your paper, draw a **big vertical line** down the middle. Left half is **"PPO's turn"**, right half is **"Environment's turn"**.

### Step 2 — the PPO side (left half)
- At the top of the left half, draw an **arrow pointing in from the right**. Label it **"observation from env"**.
- Below that arrow, draw a **rectangle**. Label it **"Policy Network"**.
- Below the Policy Network, draw *either* one arrow (Frame Stack) *or* two arrows side by side (Adaptive Stack) pointing down:
  - **Frame Stack case:** one arrow, label **"trading action ∈ {short, flat, long}"**.
  - **Adaptive Stack case:** two arrows next to each other:
    - Left arrow label: **"trading action ∈ {short, flat, long}"**
    - Right arrow label: **"memory action ∈ {slot 0, slot 1, ..., slot k−1}"**
- At the very bottom of the left half, draw a small **circle** where the arrow(s) meet the vertical divider. Label it **"action(s) sent to env"**.

### Step 3 — the Env side (right half)
- At the top of the right half, mirror-image of the PPO arrow: draw an **arrow pointing in from the left**. Label it **"action(s)"**.
- Below the arrow, draw a **rectangle**. Label it **"Env computes reward"**.
- Inside that rectangle, in small text, write the formula:
  ```
  reward = position × next-day S&P return
           − cost × |position change|
  ```
- Below the reward rectangle, if the wrapper is Adaptive Stack, draw a **second rectangle** labelled **"Memory ring updates: pop chosen slot, append new obs"**. (Frame Stack skips this — it just auto-drops the oldest.)
- Below that, draw an **arrow pointing up-and-left** back toward the PPO side. Label it **"next observation + reward"**.

### Step 4 — close the loop
- Draw a curved arrow that comes out the bottom-right of the Env side, sweeps around, and comes back up into the top-left of the PPO side. Label it **"one step complete — repeat"**.

---

## Picture 3 — The Walk-Forward Outer Loop

Pictures 1 and 2 show what happens *inside one training window*. This picture shows how many windows sit next to each other over time.

### Step 1 — the timeline
- Draw a **long horizontal line** across a fresh area. Left end label **"2000"**, right end label **"2024"**.

### Step 2 — the windows
- Along the timeline, draw a series of pairs of touching rectangles like this:
  - **Big rectangle** (the train chunk) followed by a **small rectangle** (the test chunk). They touch, no gap.
  - Big = train (label it **"~5 yrs train"**), small = test (label it **"~1 yr test"**).
- Draw about 5 pairs, each pair starting where the previous *test* ended. So the next pair's train starts right after the previous test ended.
- Between each pair, draw a **thin arrow** labelled **"slide forward"**.

### Step 3 — mark what the loop does per pair
- Above the first pair, draw a **balloon** that says **"For each pair: fresh PPO trained on the big rectangle → evaluated on the small rectangle → row written to windows.csv"**.

### Step 4 — the "hard rule" line
- Above the LAST pair on the far right, draw a **red dashed vertical line**. Label it **"final holdout — touched exactly once, at the very end"**.

---

## Picture 4 — What Each PPO Head Looks Like Inside

Zoom into the "Policy Network" rectangle from Picture 2.

### Step 1 — the shared body
- Draw a big **oval** on your page. Label it **"Shared neural net body (2 hidden layers, 64 units each)"**.
- Draw an arrow coming *into* the left side of the oval. Label it **"observation vector of length k"**.

### Step 2 — the head(s) coming out
- Out of the right side of the oval:
  - **Frame Stack case:** draw ONE arrow leading to a small circle. Circle labelled **"trading head: 3 outputs (probabilities over short/flat/long)"**.
  - **Adaptive Stack case:** draw TWO arrows, going to two separate circles:
    - Top circle: **"trading head: 3 outputs"**
    - Bottom circle: **"memory head: k outputs (probabilities over slot 0..k−1)"**

### Step 3 — how a decision gets made
- Below each head, draw a small arrow going down to a **little diamond**.
- Label the diamond **"sample from the probability distribution → one integer"**.
- The integer(s) from the diamond(s) get sent to the env (this is the same arrow you drew in Picture 2).

---

## Picture 5 — What Comes Out and What You Do With It

You already drew three little rectangles at the top of Picture 1 (`summary.json`, `windows.csv`, `steps.csv`). Now show what each is *for*.

### Step 1 — draw three columns below the three rectangles
- Below `summary.json`, draw a **notebook shape**. Label it **"human reads this: what the config was, headline mean±std, position totals"**.
- Below `windows.csv`, draw a **bar-chart shape**. Label it **"pandas loads this: one row per window, easy to concat across many runs and plot"**.
- Below `steps.csv`, draw a **line-chart shape**. Label it **"per-day analysis: equity curve, when did the agent flip, when did it get memory right"**.

### Step 2 — many runs merge into one comparison
- Below all three columns, draw a **big funnel**. Above the funnel, draw arrows from many small `windows.csv` icons (from lots of runs) pouring into the top of the funnel.
- Out of the bottom of the funnel, draw an arrow to a rectangle labelled **"RQ2 comparison plot: Frame Stack vs Adaptive Stack, one line per mechanism, x-axis = k, y-axis = mean Sharpe ± std"**.

---

## Picture 6 — Business-Value Wrapper (Aspirational)

If someone outside the university wanted to *use* this system, what would that look like?

### Step 1 — a big outer box
- On a fresh page, draw a **large rounded rectangle** that fills the whole area. Label it **"HTTP API (future)"**.

### Step 2 — inside the box, put the whole current system
- Inside, draw a smaller box labelled **"Everything from Picture 1"** — this represents the entire research pipeline collapsed into one thing.

### Step 3 — arrows in and out of the API box
- Arrow coming in from the left, going into the box. Label it **"POST /train {config JSON}"**.
- Arrow coming out to the right, exiting the box. Label it **"200 OK {run_id, links to summary.json, csvs}"**.
- A second incoming arrow: **"GET /runs/{run_id}/positions"**.
- A second outgoing arrow: **"per-day position stream (for a live trader)"**.

### Step 4 — the "not yet" stamp
- Write **"NOT BUILT — ROADMAP ONLY"** in big letters diagonally across this whole picture.

*(The academic project doesn't need this. It's here so you can hand the diagram to a stakeholder and show the shape of what could come after the thesis.)*

---

## Picture 7 — Proposal Contract Check

The last picture is a checklist of what the proposal promised, mapped to boxes you already drew.

Draw a **two-column table** somewhere clear on the page:

| Proposal objective (short form)         | Which box on your diagram fills this?             |
|------------------------------------------|---------------------------------------------------|
| Build a financial time-series env       | The **square** in Picture 1                        |
| Supervised LSTM baseline                | (separate file, not in this diagram yet — draw a small tag off to the side labelled **"LSTM baseline runs in parallel, not in the RL loop"**) |
| Frame Stacking baseline (k∈{3,5,10,20}) | The **ring** in Picture 1, Frame Stack case      |
| Adaptive Stacking (κ∈{3,5,10,20})       | The **ring** in Picture 1, Adaptive Stack case    |
| Compare using financial + RL metrics    | The three **output rectangles** at top of Picture 1 |
| Analyse learned memory policy           | The `mem_slot_0..k−1` columns in **windows.csv**  |
| Walk-forward evaluation                 | Picture 3 in its entirety                         |
| Buy-and-hold sanity floor               | Extra column `bh_sharpe/bh_cumret/bh_maxdd` in windows.csv (a small floor-line to add to the funnel plot in Picture 5) |

If every row has a box that fills it, the diagram is proposal-complete.

---

## Summary of Shapes You Drew (and What They Mean)

| Shape        | What it is                                                       |
|--------------|------------------------------------------------------------------|
| Cloud        | Data source outside your control (Yahoo Finance)                 |
| Rectangles (stacked) | Files on disk                                            |
| Big square   | The environment — the "game" the agent plays                     |
| Ring around square | The memory wrapper — hugs the env, changes what the agent sees |
| Circle       | The learning agent (PPO)                                          |
| Oval         | The agent's neural network body                                   |
| Small circles (heads) | The output probability distributions                     |
| Diamond      | Sampling — turning probabilities into one chosen action           |
| Envelope     | The YAML config that controls everything                          |
| Notebook / bar chart / line chart | How outputs are consumed by humans + code    |
| Funnel       | Aggregation across many runs → one comparison plot                |
| Rounded outer rectangle (Picture 6) | The API layer (roadmap-only)              |

---

## Loop You Should Now Be Able to Trace With Your Finger

1. YAML config points at CSVs.
2. Loader turns CSVs into arrays.
3. Walk-forward splits arrays into training windows.
4. Per window: env is built → memory wrapper hugs it → PPO agent plays it in a loop, learning.
5. After training, agent replays the test slice once → per-day outputs recorded.
6. Rows appended to CSVs, summary.json written.
7. Many runs' CSVs merged → RQ2 comparison plot.

That's the whole project on one page.
