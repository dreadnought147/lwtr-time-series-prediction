# Speaker Notes

How to talk through each slide. Written the way you actually speak, so you can glance down and keep going.

---

## Slide 1: Title

**Say:**
- "So the title of my project is Learning What to Remember for Time Series Prediction Using Reinforcement Learning."
- "Basically I'm testing whether an agent that learns what to remember does better than an agent that just remembers the last few things."
- "I'm going to walk you through the three things I've trained and where each one landed."

**If he asks why the weatherman picture:**
- "It's a joke about forecasting. The weatherman points at a map, my agent points at market regimes. Same job, different map."

---

## Slide 2: What we have done

**Say:**
- "There are three things trained and one metrics catalogue."
- "First is the LSTM supervised baseline. That's my control."
- "Second is Frame Stacking reinforcement learning. That's the standard memory approach."
- "Third is Adaptive Stacking reinforcement learning. That's the one your paper introduced, and that's my actual contribution."
- "Then I've got metrics, and then what I'm doing next week."

**Keep it short here.** This slide is just a map of the talk.

---

## Slide 3: Why we needed an RQ1 control

**Say:**
- "Research question one asks: can reinforcement learning with Frame Stacking beat a properly tuned LSTM?"
- "The reason I need this is simple. If my RL agent can't beat a normal supervised model, then all the extra machinery of reinforcement learning isn't buying me anything."
- "And I was careful not to make the LSTM weak on purpose. If I beat a lazy LSTM that proves nothing."

**Concept to explain if he pushes:**
- "Supervised learning just looks at data and makes one guess. Reinforcement learning makes a decision, sees what happens, and the consequences stack up over time. I wanted to check whether that extra structure actually helps here."

---

## Slide 4: LSTM, the data

**Say:**
- "I pulled daily data from Yahoo Finance, about 36 years of it."
- "I'm using two series. One is what the agent looks at, the other is what the agent actually trades on."
- "I inner joined them, meaning I only kept days where both series have data. If I'd done it any other way I'd end up with gaps, and a gap turns into a broken reward that silently poisons training."
- "After the join I've got 9188 usable trading days, roughly 24 years of overlap."
- "Then I converted prices into log returns."

**Why log returns, in simple terms:**
- "Raw prices drift upward forever, so a model trained on old prices sees totally different numbers than new prices. Log returns are just daily percentage changes, so they sit around zero the whole time and the model sees a consistent scale."

**Why these two series work well together:**
- "They move against each other. When one spikes the other usually drops. So the first one is genuinely informative about the second, which is exactly what you want when you're testing whether memory helps."

**Walk forward, in simple terms:**
- "I don't train once and test once. I train on five years, test on the next year, then slide forward and do it again. 31 times total. Each time it's a brand new model."

---

## Slide 5: LSTM, the model and how it decides

**Say:**
- "It's a two layer LSTM, hidden size either 64 or 128 depending on the sweep."
- "Every day it looks at the last 20 days, which is about a trading month."
- "It outputs one number, a guess of tomorrow's return."
- "If that number is positive it goes long. If negative it goes short. If zero it sits flat, but that basically never happens."
- "Then the reward is position times the actual next day return, minus a small cost whenever it changes its mind."

**The important fairness detail:**
- "When I standardise the inputs, I only use statistics from the training slice. Never the test slice. If I used the whole dataset I'd be leaking future information backwards, which would make the results look better than they are."

**If he asks what "fresh model per window" means:**
- "Every one of the 31 windows gets a brand new model with random starting weights. Nothing carries over from the previous window. So window 10 can't accidentally remember anything from window 9."

---

## Slide 6: LSTM, the sweep and fairness

**Say:**
- "I swept 4 hyperparameter combinations, 7 seeds each, across 31 windows. That's 868 LSTMs trained from scratch."
- "I used 7 seeds because the literature says 5 to 10 is the defensible range. 3 is prototyping, which is where I started, but 7 is what I report."

**Why seeds matter, simply:**
- "A seed controls the random starting weights. Run the same model twice with different seeds and you can get noticeably different results. So if I only ran one seed I wouldn't know whether a good result was skill or luck. Running 7 and reporting the average plus the spread tells you which it is."

**Other fairness things I added:**
- "Gradient clipping, which stops the training from blowing up on noisy days."
- "Weight decay, which is mild regularisation."
- "And the champion is picked by averaging across all seeds, not by picking the best run."

---

## Slide 7: LSTM champion

**Say:**
- "The winner was hidden size 128, one layer."
- "Sharpe of plus 0.208, with a spread of 1.249."
- "That spread is wide, which honestly tells you something on its own. The LSTM is not confident."
- "This is the number that Frame Stacking has to beat."

---

## Slide 8: Frame Stacking, why

**Say:**
- "Frame Stacking is the simplest memory you can give an agent. You just show it the last k observations stacked together."
- "The agent has no say in what gets kept. The oldest one falls off automatically, like a queue."
- "This is the same mechanism DQN used for Atari back in 2015, so it's the standard baseline."
- "It plays two roles for me. It's the RL side of research question one, and it's the thing Adaptive Stacking has to beat in research question two."

---

## Slide 9: Frame Stacking, model and sweep

**Say:**
- "The agent is PPO with a plain feedforward network. No memory inside the network itself."
- "All the memory comes from the wrapper. That's deliberate, because if the network had its own memory I wouldn't be able to tell which part is doing the work."
- "Each day it sees k stacked observations and picks short, flat, or long."
- "Reward is exactly the same formula as the LSTM, so the comparison is fair."
- "I swept k across 3, 5, 10 and 20, with 7 seeds each, across 31 windows. 868 agents."

**Critical point to make:**
- "The PPO settings are locked identical between Frame Stacking and Adaptive Stacking. Same learning rate, same batch size, same network widths. The only thing that changes is the memory mechanism. That's what makes the research question two comparison honest."

---

## Slide 10: Frame Stacking champion

**Say:**
- "The winner was k equals 3. Sharpe of plus 0.484."
- "So that answers research question one. Frame Stacking beats the LSTM by about 0.28 Sharpe."
- "Reinforcement learning earns its keep on this task."

**If he asks about buy and hold:**
- "Buy and hold sits at 0.841, so neither of them beat it. But that's expected, this period was mostly a bull market and just holding the index is hard to beat. My research question is about which memory mechanism is better, not about whether I built a profitable trading system."

---

## Slide 11: Adaptive Stacking, what it is

**Say:**
- "This is the main contribution."
- "Same k slots as Frame Stacking, but now the agent chooses which slot to overwrite each step."
- "So memory becomes a decision the agent learns, instead of a fixed rule."
- "I ported this directly from your paper, Beyond Sliding Windows."
- "In the paper it was tested on T Maze, MiniGrid, Rubik's Cube and FetchReach. Those are all clean, episodic, stationary environments."
- "My question is whether it still works when the environment is noisy, non stationary, and the agent's actions don't affect what happens next."

---

## Slide 12: Adaptive Stacking, how the agent chooses

**Say:**
- "The action space grows. Frame Stacking just picks a trade. Adaptive Stacking picks a trade and a memory slot at the same time."
- "PPO handles that natively, so I didn't have to change the algorithm."
- "Both decisions are trained by the same reward signal."

**Why the overwrite is forced, explain it this way:**
- "I didn't give it a skip option. And the reason is that skipping would always be the safe choice. Nothing gets lost, nothing gets risked. So the agent would just learn to always skip and the memory head would never learn anything."
- "By forcing it to throw something away every single step, different choices lead to different outcomes, and that difference is what teaches it."

**The diagnostic:**
- "I log which slot it picks every day. If the distribution comes out flat then it's basically doing the same thing as Frame Stacking. If it's skewed, that means it learned that some observations are worth keeping longer than others."

---

## Slide 13: Adaptive Stacking champion

**Say:**
- "Winner was k equals 5. Sharpe of plus 0.515."
- "So it beats the Frame Stacking champion, but only slightly."
- "The more interesting result is the pattern. At matched memory budget, Adaptive Stacking wins at k equals 5, 10 and 20. Frame Stacking only wins at k equals 3."
- "And the gap grows as the memory budget grows. At k equals 10 the gap is the biggest."

**What that means, in plain terms:**
- "When the agent only has 3 slots there isn't much to be clever about. But once it has 10 or 20 slots, being able to choose what to keep actually starts paying off. That's consistent with the whole idea behind the method."

**Be honest about this:**
- "The spreads are wide, so I'd call this directional rather than conclusive. I'd want a significance test before claiming it hard."

---

## Slide 14: Metrics

**Say:**
- "I'm tracking three main numbers plus two diagnostic ones."

**Sharpe:**
- "Return divided by volatility. It answers how much you got paid for the risk you took. Two agents can make the same money, but if one did it smoothly and one did it with wild swings, Sharpe says the smooth one is better."

**Cumulative return:**
- "Just total profit. No risk adjustment. It keeps me honest, because an agent could get a good Sharpe just by being less bad rather than actually making money."

**Max drawdown:**
- "The worst drop from a previous high. It's the pain metric. If my memory mechanism actually helps the agent see trouble coming, this is where it should show up."

**Position stats:**
- "How often it goes long, short, or flat, and how often it changes its mind. Fewer changes means it's more committed, which usually means it's actually seeing a signal rather than guessing."

**Memory distribution, Adaptive Stacking only:**
- "This is the one unique to my contribution. If it's flat, the agent never learned to be selective. If it's skewed, it did."

---

## Slide 15: Results

**Say:**
- "Research question one: Frame Stacking at 0.484 beats the LSTM at 0.208. Reinforcement learning wins."
- "Research question two: Adaptive Stacking wins at three out of four memory budgets, and the advantage grows with the budget."
- "Neither beats buy and hold, which I'd flag as a limitation rather than a failure, because my question was about mechanisms, not about building a profitable fund."

---

## Slide 16: Next week

**Say:**
- "First thing is the Overleaf project in the IEEE template, and I'll share it with you."
- "Then the poster for Innovation Day."
- "Then writing up method and results, which I can pull straight from these slides and the plots I've already generated."
- "There are two small things I want to flag as well."

**Thing one:**
- "There's a bug in how I compute the champion in the summary file. It only reflects the last resumed session rather than the whole sweep. The CSV has the correct data, so it's a five line fix, but I wanted to mention it rather than hide it."

**Thing two, and this is the one I'd like your opinion on:**
- "Industry practice says you should combine walk forward with a sealed final holdout, meaning you lock away the last two years and never touch them until the very end. I did the walk forward part but not the sealed holdout."
- "My walk forward gives me 217 out of sample measurements per configuration, so I don't think the results are invalid. But I'd rather document it as a limitation than pretend it isn't there. Would you like me to re run with a sealed holdout, or is documenting it enough for this scope?"

---

## General tips

- If you lose your place, go back to the three pillars: LSTM is the control, Frame Stacking is the standard, Adaptive Stacking is the contribution.
- If he asks something you don't know, say "I haven't checked that, let me go look" rather than guessing. That reads better than a wrong answer.
- Numbers worth memorising: LSTM 0.208, Frame Stacking 0.484, Adaptive Stacking 0.515, buy and hold 0.841.
- Total models trained: 2604. That number does a lot of work on its own.
