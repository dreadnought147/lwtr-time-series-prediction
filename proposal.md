| L   |         |              | W     |           |               | R       |             |      |     |
| --- | ------- | ------------ | ----- | --------- | ------------- | ------- | ----------- | ---- | --- |
|     | EARNING |              | HAT   | TO        |               | EMEMBER |             |      | FOR |
|     | T       | -S           |       | P         |               |         |             | U    |     |
|     | IME     |              | ERIES | REDICTION |               |         |             | SING |     |
|     | R       |              |       |           |               | L       |             |      |     |
|     |         | EINFORCEMENT |       |           |               | EARNING |             |      |     |
|     | School  | of Computer  |       | Science   | &             | Applied | Mathematics |      |     |
|     |         | University   |       | of the    | Witwatersrand |         |             |      |     |
Banele Mkhize
|     |     |     | Student    | Number:   | 2663814 |     |       |     |     |
| --- | --- | --- | ---------- | --------- | ------- | --- | ----- | --- | --- |
|     |     |     | Supervised | by Geraud | Nangue  |     | Tasse |     |     |
|     |     |     |            | May 17,   | 2026    |     |       |     |     |
A literature review submitted to the Faculty of Science, University of the Witwatersrand,
Johannesburg, in partial fulfilment of the requirements for the degree of Bachelor of Science
with Honours

Abstract
Reinforcement learning agents that operate in partially observable environments
needmemorytoacteffectively,becauseasingleobservationrarelycontainsenough
informationabouttheunderlyingstate. Thedominantmemorymechanismindeep
reinforcement learning is frame stacking, a fixed-size sliding window in which the
agent has no say in what enters or leaves. Adaptive Stacking, a recently proposed
alternative, gives the agent an explicit memory action and learns what to retain
through the same reward signal that drives its behaviour. So far it has only been
validated on stationary, episodic benchmarks such as T-Maze, MiniGrid-Memory,
Rubik’sCube,andFetchReach,wheretheinformativeobservationisclearlydefined
and the environment does not shift over time. Whether the same active memory
mechanism transfers to noisy, non-stationary data streams is an open empirical
question. This work evaluates Adaptive Stacking on financial time-series, using
the VIX as the agent’s observation stream and the S&P 500 as the underlying for
positioning. The comparison is built as a three-step staircase that changes one
variable at each step: supervised learning against reinforcement learning, rein-
forcement learning without memory against reinforcement learning with Frame
Stacking, and Frame Stacking against Adaptive Stacking at matched memory bud-
get. All four configurations are trained with stack sizes swept over {3,5,10,20},
evaluated using walk-forward windows to handle non-stationarity, and compared
on cumulative return, Sharpe ratio, and maximum drawdown. A negative result,
in which Adaptive Stacking fails to transfer, is itself a useful contribution because
it would say the sparsity assumption underlying the method does not generalise
from controlled benchmarks to diffuse financial signals. This is the a empirical
evaluation of Adaptive Stacking on financial time-series, and more broadly it tests
whether explicit, agent-controlled memory mechanisms generalise from the con-
trolled environments they are typically validated on to the messy, non-stationary
settings real-world reinforcement learning agents have to act in.
i

Declaration
I, Banele Mkhize (Student Number: 2663814), hereby declare the contents of this lit-
erature review to be my own work. This literature review is submitted for the degree
of Bachelor of Science with Honours in Computer Science at the University of the Wit-
watersrand, Johannesburg. This work has not been submitted to any other university,
or for any other degree.
ii

Acknowledgements
| Thank you | to the most | High |
| --------- | ----------- | ---- |
iii

Contents
Preface
Abstract . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . i
Declaration . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . ii
Acknowledgements . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . iii
Table of Contents . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . iv
List of Figures . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . v
List of Tables . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . vi
| 1 Introduction |                | 1   |
| -------------- | -------------- | --- |
| 2 Problem      | Statement      | 2   |
| 3 Research     | Questions      | 3   |
| 4 Aims         | and Objectives | 4   |
4.1 Aim . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 4
4.2 Objectives . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 4
| 5 Literature | Review | 5   |
| ------------ | ------ | --- |
5.1 Reinforcement Learning and Sequential Decision-Making . . . . . . . . . 5
5.2 The Markov Property and Partial Observability . . . . . . . . . . . . . . . 6
5.3 Policies and Value Functions . . . . . . . . . . . . . . . . . . . . . . . . . 7
5.4 Deep Reinforcement Learning . . . . . . . . . . . . . . . . . . . . . . . . 8
5.5 Memory Mechanisms in Neural Networks . . . . . . . . . . . . . . . . . 8
5.6 Frame Stacking . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 9
5.7 Long Short-Term Memory . . . . . . . . . . . . . . . . . . . . . . . . . . 10
5.8 Recurrent Neural Networks and Selective Gating . . . . . . . . . . . . . 10
5.9 Limitations of Fixed Frame Stacking . . . . . . . . . . . . . . . . . . . . 10
5.10 Adaptive Stacking as a Memory Action . . . . . . . . . . . . . . . . . . . 11
5.11 Reinforcement Learning for Time-Series Prediction . . . . . . . . . . . . 12
5.12 Time-Series Prediction in Financial Domains . . . . . . . . . . . . . . . . 13
5.13 Comparison of Memory Approaches . . . . . . . . . . . . . . . . . . . . . 13
5.14 Summary and Gap . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 14
| 6 Methodology |     | 15  |
| ------------- | --- | --- |
6.1 Environment and Data . . . . . . . . . . . . . . . . . . . . . . . . . . . . 15
6.2 Algorithm . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 16
6.3 Baselines . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 16
6.4 Evaluation Metrics . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 16
6.5 Hyperparameters and Sweeps . . . . . . . . . . . . . . . . . . . . . . . . 17
6.6 Expected Outcomes . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 17
| 7 Risks    | and Limitations | 17  |
| ---------- | --------------- | --- |
| 8 Workplan |                 | 18  |
iv

List of Figures
1 The agent-environment interaction loop in reinforcement learning [10]. 5
v

List of Tables
1 Comparison of memory mechanisms. . . . . . . . . . . . . . . . . . . . . 13
vi

1 Introduction
Reinforcement learning (RL) provides a general framework for sequential decision-
making under uncertainty, where an agent learns a policy π(a | s ) that maximises
t t
cumulative reward through interaction with an environment [1]. At each timestep t,
the agent observes a state s , takes an action a , and receives a reward r , with the
t t t+1
environment transitioning according to the dynamics P(s ,r | s ,a ). The standard
t+1 t+1 t t
RL formulation assumes the Markov property which is that that s alone contains all
t
theinformationneededtoactoptimally, meaningthesedynamicsarestationaryanddo
not depend on any observations prior to s .
t
However in real applications, real-world environments tend to violate the Markov as-
sumption. The agent’s immediate or current observation is most often an incomplete
representation of the full underlying state. This leads to creating what are known as
non-Markovian dependencies over time [2]. In such an environment where the agent
needs more context than the current observation, the environment may be k∗-order
Markovian,meaningthetransitiondynamicsonlybecomestationarywhenconditioned
onthelastk∗ observations,i.eP(s ,r | s ,a ). Thisstatesthattheprobabilityof
t+1 t+1 t:t−k∗ t
the next state and reward depends not just on the current observation, but on the most
recent k∗ observations s . So when k∗ = 1, this reduces to the standard Markov
t:t−k∗
case previously discussed. So it is clear that past observations carry critical context for
future decisions, and the agent must maintain some form of memory to act effectively.
The question we want to explore is how to manage that memory, including what to
remember, what to forget, and how much to retain, still is a actively studied area of
research, the most popular way to alleviate this problem is Frame Stacking (FS), which
is a FIFO short-term memory, whereby a fixed stack of the most recent observations
(and actions) are concatenated [2], [3]. This challenge is particularly pronounced in
domains like financial markets, where relevant signals may be separated by long time
intervals, and where the cost of forgetting critical information can directly translate
into poor investment decisions [4].
This work traces the evolution of approaches to this memory problem. It begins by
establishing the foundational concepts of RL, covering the agent-environment interac-
tion loop, value functions, and the Bellman optimality equation [1]. It then examines
how deep reinforcement learning introduced the capacity to learn directly from high-
dimensional observations but in partciular we see how it relies on fixed-size memory
mechanisms such as frame stacking, where the most recent k observations are concate-
nated as input [3]. The review then considers how recurrent architectures like Long
Short-Term Memory (LSTM) networks offered a more flexible approach to sequential
memorythroughlearnedgatingmechanisms[5]. FinallyweturntoAdaptiveStacking,
a recent meta-algorithm that directly addresses the limitations of both frame stacking
and recurrent approaches by learning to selectively manage a bounded memory stack
of size κ ≪ k∗ via reinforcement learning [2]. Here, κ represents the minimal number
of past observations that are actually needed to act optimally, which is typically much
smaller than the full k∗ window required to make the environment fully Markovian.
The central idea is that in many environments with large k∗, only a small subset of
past observations actually carry information relevant to future rewards, so the agent
1

does not need to remember everything, just the right things. Rather than passively
storing the most recent k observations as FS does, Adaptive Stacking allows the agent
to actively decide which observations to keep and which to discard at each timestep.
This is achieved by adding to the standard RL agent’s action with a memory action that
selects which slot in the stack to overwrite, turning memory management itself into a
decision-making problem aligned with maximising reward. The result is an agent that
learns what to remember through the same reward signal that shapes its behaviour in
theenvironment,whichofferssubstantialimprovementsincomputeefficiency,memory
efficiency, and generalisation. Finally, the review situates these developments within
the domain of financial time-series prediction, where non-Markovian dependencies are
prevalent and memory-efficient decision-making is essential [4]. The position of this
review is that Adaptive Stacking represents a well-motivated and promising direction
for RL-based financial prediction. To ground this discussion, the following section es-
tablishes the foundational concepts of reinforcement learning.
2 Problem Statement
Reinforcement learning agents rarely operate in environments where the full state is
handedtothem. Mostreal-worldenvironmentsareinsteadPartiallyObservableMarkov
Decision Processes (POMDPs), in which the agent only receives partial glimpses of the
underlying system state through its observations [6], [7]. These observations arrive
in sequence over time, as a continuous data stream emitted by the environment. Be-
cause no single observation is enough to determine the true state, the agent must rely
on memory of past observations to resolve non-Markovian dependencies and to infer
enough of the underlying state to act on [2].
The dominant way of handling these dependencies in deep reinforcement learning has
been Frame Stacking, a fixed-size sliding window that concatenates the most recent k
observationsintoasingleinput[2],[3]. Thisisapassivemechanism. Theagenthasno
control over what enters or leaves the stack, observations are discarded purely on the
basis of recency, and the size of the stack has to be specified before training begins. As
soonastherelevantinformationliesfurtherbackthank timesteps,theagentphysically
cannotaccessit,andpushingk highertocompensatescalesmemoryandcomputecosts
proportionally [2]. Adaptive Stacking, proposed by Nangue Tasse et al. [2], addresses
this by giving the agent an explicit memory action. Rather than passively retaining the
most recent k observations, the agent learns which slot in a bounded stack of size κ to
overwrite at each timestep, with the memory action shaped by the same reward signal
that drives the policy. This turns memory management itself into a decision-making
problem.
The limitation is that Adaptive Stacking has only been validated on controlled bench-
marks. The experiments in [2] are run on T-Maze, MiniGrid-Memory, Rubik’s Cube,
and FetchReach. These are stationary, episodic environments with sparse rewards and
clearly defined informative observations, where the agent only has to retain one or two
specific past observations to succeed. Real-world data streams behave very differently.
Financial time-series, for example, are noisy, non-stationary, and chaotic, with statisti-
2

cal properties and dependencies that shift over time and informative signals that are
diffuse rather than concentrated in a few well-marked timesteps [8]. The open ques-
tion is whether an active memory mechanism that succeeds in controlled POMDPs can
also filter out the large amount of uninformative noise present in live, non-stationary
time-series streams.
This research therefore aims to investigate how reinforcement learning agents perform
on non-stationary, noisy time-series data streams when given active control over their
memory. Specifically, the goal is to evaluate whether the explicit memory management
offered by Adaptive Stacking lets an agent retain sparse, meaningful signals and dis-
card uninformative observations better than the existing passive alternatives. This is
answeredthrough astructured setof comparisons: firstestablishing whetherreinforce-
ment learning is the right learning paradigm for non-stationary financial time-series at
all, by comparing it against a supervised learning baseline, and then comparing the ac-
tive memory control of Adaptive Stacking against the passive, fixed-window approach
of traditional Frame Stacking.
3 Research Questions
The central question this research investigates is whether active memory management
improvesreinforcementlearningperformanceonnon-stationary,noisytime-seriesstreams.
The comparison is built up in three steps, with each step changing only one variable so
that its contribution can be attributed cleanly.
1. The first research question asks whether, on a non-stationary financial time-series
task, framing the problem as reinforcement learning yields better performance
than a supervised learning baseline [4], [8].
2. The second research question asks whether giving the reinforcement learning
agent a fixed memory window through Frame Stacking improves performance
over a memoryless agent that observes only the current timestep [3].
3. The third research question asks whether Adaptive Stacking outperforms Frame
Stacking at a matched memory budget when the environment is a non-stationary
financial time-series stream, rather than the stationary, episodic benchmarks on
which it has been validated so far [2].
The first two questions establish the conditions under which the third one is asked.
The third research question is the primary contribution of this work. The same com-
parison between Adaptive Stacking and Frame Stacking has already been answered in
stationary, controlled environments [2], but it has not been answered in the noisy, non-
stationary time-series setting. Whether that result extends to this setting is the open
empirical question this research investigates.
3

4 Aims and Objectives
4.1 Aim
The aim of this research is to empirically evaluate whether reinforcement learning
agentsequippedwithAdaptiveStackingcanoutperformbothsupervisedlearningbase-
lines and traditional Frame Stacking baselines on non-stationary, noisy financial time-
series prediction tasks. By extension, this evaluates whether explicit, learned mem-
ory management generalises from the controlled benchmarks used in [2] to a domain
where the relevant past is sparse, diffuse, and noisy.
4.2 Objectives
To meet this aim, the following objectives have been defined. Each objective maps onto
one or more of the research questions stated above.
1. Build a financial time-series environment. Wrap a non-stationary financial
dataset, such as a daily or intraday equity index, into a reinforcement-learning
environment with a well-defined observation space, action space, and reward
function, following the framing established by Moody and Saffell [4].
2. Implement a supervised learning baseline. Train an LSTM forecaster on the
same data, wrapped with a simple trading rule to convert forecasts into positions
[5]. This baseline answers RQ1.
3. Implement a no-memory reinforcement learning baseline. Train a PPO agent
that observes only the current observation o (equivalent to Frame Stacking with
t
k = 1)to establishwhat amemoryless reinforcementlearning agentachieves [9].
4. Implement a Frame Stacking baseline. Train PPO agents with Frame Stacking
at a range of stack sizes k ∈ {3,5,10,20} to establish how passive memory affects
performance and to identify a reasonable matched-budget configuration [2], [3].
5. Implement Adaptive Stacking. Train PPO agents with Adaptive Stacking at
matched memory budgets κ ∈ {3,5,10,20}, mirroring the Frame Stacking sweep
so that the comparison is honest and controlled [2]. Objectives 4 and 5 together
answer RQ2.
6. Compare performance using financial and reinforcement-learning metrics.
Evaluate all agents using financial metrics, including cumulative return, Sharpe
ratio, and maximum drawdown, alongside reinforcement-learning metrics such
as sample efficiency and training-time memory footprint [8].
7. Analyse the learned memory policy. Inspect what Adaptive Stacking actually
retains over time, in order to address RQ3. This means looking at which obser-
vations the agent chooses to keep in its stack and whether those retentions align
with externally identifiable market events such as regime shifts.
8. Report a comparison of memory mechanisms in the financial setting. Present
the empirical results, situate them against the controlled-benchmark results in
4

[2], and identify whether the central assumption that the relevant past is sparse,
κ ≪ k∗, holds in this domain.
5 Literature Review
5.1 Reinforcement Learning and Sequential Decision-Making
Building on the formulation introduced above, the RL interaction loop can be made
precise. Feedback in RL is evaluative rather than instructive: the reward signals the
quality of the chosen action, never which action was correct [1]. At each timestep t,
the agent observes a state s ∈ S, selects an action a ∈ A according to a policy π, and
t t
the environment responds with a reward r ∈ R and transitions to a new state s
t+1 t+1
[1]. This interaction loop is illustrated in Figure 1.
Figure 1: The agent-environment interaction loop in reinforcement learning [10].
The policy can be stochastic, π(a | s), defining a probability distribution over actions
for each state, or deterministic, π(s) = a, mapping each state directly to a single action
[10]. Stochasticpoliciesareusefulforexplorationandforenvironmentswithstochastic
optimal behaviour, while deterministic policies are efficient in well-defined scenarios.
The environment dynamics are characterised by a transition probability P(s′ | s,a)
and reward function R(s,a), which together with the state space, action space, and a
discount factor γ ∈ [0,1] form a Markov Decision Process M = (S,A,P,R,γ) [1]. The
agent’sobjectiveistomaximisetheexpectedreturnG = (cid:80)∞ γkr ,thecumulative
t k=0 t+k+1
discounted sum of future rewards, where γ controls how much weight is placed on
futurerewardsrelativetoimmediateones; γ = 0producesamyopicagentwhichisone
that maximises more current rewards and with values close to 1 produce a far-sighted
agent. [1].
RL methods are commonly split into model-free and model-based approaches, model-
free is when the RL agent learns a policy or value function directly from experience,
and model-based appraoches to RL agents use a learned or given transition model to
plan [10]. In both cases, the quality of the agent’s decisions is fundamentally tied to
the information about its environment available to it at each timestep [2]. This raises
a critical question for environments where the current observation alone may not be
5

sufficient: how should the agent represent and manage the information it needs to act
well?
5.2 The Markov Property and Partial Observability
The central assumption underlying the MDP framework is the Markov property, which
states that the future depends only on the current state and action, not on the full
history of everything that came before [1], [10]. We define a system as Markovian,
when the next state and reward are conditionally independent of all past states and
actions provided it is given the current state and action,
P(S ,R | S ,A ) = P(S ,R | S ,A ,S ,A ,...,S ,A ) (1)
t+1 t+1 t t t+1 t+1 t t t−1 t−1 0 0
which says that knowing the full history of past states and actions adds no additional
information beyond what is already contained in the current state S [1], [10]. Un-
t
der this assumption, the agent’s current observation is enough on its own to act opti-
mally,thankfullythisiswhatmakesthelearningproblemtractable. Theagentcanlearn
a policy π(a | s) and evaluate it using value functions without needing to account for
anything that happened before the current timestep, this saves us alot of computation
and memory.
The problem however is that the Markov property almost never holds in real environ-
ments,theworldisfartoocomplicatedthanthesimplificationinthepropert. Theagent
is usually given an observation that is only a partial view of the underlying state, so a
single observation is not enough to act optimally on its own. A more fitting formalism
for this setting is the Partially Observable Markov Decision Process (POMDP), defined
by the tuple ⟨S,A,T ,R,Ω,O⟩, where S is the state space, A is the action space, T is
the transition function, R is the reward function, Ω is a finite set of observations, and
O is the observation function that specifies the distribution over observations given the
underlying state and action [7]. In a POMDP the agent no longer sees the true state S
t
directly, and must instead make decisions based on observations o ∈ Ω that may only
t
partially reflect the true state.
A central problem that arises in this setting is perceptual aliasing where distinct states
may produce identical observations while requiring different actions [7]. When two
states look the same to the agent but demand different behaviour, a policy that doenst
retain some history of observations cannot distinguish them and therefore cannot re-
cover the optimal policy. In fact, finding the best memoryless policy in a POMDP has
been shown to be NP-hard [7],which gives theoretic motivation behind delevoping
agents with some explicit form of memory. As discussed in the Introduction, Nangue
Tasse et al. [2] formalise this by defining the environment as k∗-order Markovian,
meaning the dynamics only become stationary when conditioned on the last k∗ obser-
vations,
P(S ,R | S ,A ) (2)
t+1 t+1 t:t−k∗ t
6

whereS denotesthe mostrecentk∗ observations. Whenk∗ = 1, thisreducestothe
t:t−k∗
standard Markov case in Equation 1. When k∗ is large, the agent must maintain a large
| history | of passed | observations |     |     | to act | optimally. |     |     |     |     |     |
| ------- | --------- | ------------ | --- | --- | ------ | ---------- | --- | --- | --- | --- | --- |
In a financial time-series domain, for example, a single price reading at a given time
tells the agent very little about the current market regime, trend, recent volatility, or
older structural events that still shape future returns. The Markov assumption would
struggle in this kind of setting, so to equip the agent with some explicit mechanism
for keeping past information, is crucial to battling this memmory problem. This is the
memory problem that the rest of this review is concerned with.
| 5.3 | Policies | and | Value |     | Functions |     |     |     |     |     |     |
| --- | -------- | --- | ----- | --- | --------- | --- | --- | --- | --- | --- | --- |
To evaluate how good a policy is from any given state, RL defines two core functions.
The state-value function v (s) gives the expected return when starting in state s and
π
| following | policy | π   | thereafter, |       |     |          |       |            |          |     |     |
| --------- | ------ | --- | ----------- | ----- | --- | -------- | ----- | ---------- | -------- | --- | --- |
|           |        |     |             |       |     | (cid:34) |       | (cid:12)   | (cid:35) |     |     |
|           |        |     |             |       |     | ∞        |       | (cid:12)   |          |     |     |
|           |        |     |             |       | E   | (cid:88) |       |            |          |     |     |
|           |        |     |             | v (s) | =   | γkR      |       | (cid:12) S | = s      |     | (3) |
|           |        |     |             | π     |     | π        | t+k+1 | (cid:12) t |          |     |     |
(cid:12)
k=0
and the action-value function q (s,a) extends this to the case where the first action is
π
fixed,
|     |     |     |     |     | (cid:34) |     | (cid:12) |     |     | (cid:35) |     |
| --- | --- | --- | --- | --- | -------- | --- | -------- | --- | --- | -------- | --- |
|     |     |     |     |     |          | ∞   | (cid:12) |     |     |          |     |
(cid:88)
|     |     |     | q   | (s,a) = | E   | γkR   | (cid:12) S | = s, | A   | = a | (4) |
| --- | --- | --- | --- | ------- | --- | ----- | ---------- | ---- | --- | --- | --- |
|     |     |     | π   |         | π   | t+k+1 | (cid:12)   | t    | t   |     |     |
(cid:12)
k=0
[1], [10]. Both functions satisfy a recursive relationship known as the Bellman equa-
tion, which expresses the value of a state in terms of the values of its successor states,
|     |     |     |       | (cid:88) |     | (cid:88)    |        | (cid:2) |     | (cid:3) |     |
| --- | --- | --- | ----- | -------- | --- | ----------- | ------ | ------- | --- | ------- | --- |
|     |     |     | v (s) | =        | π(a | | s) p(s′,r | | s,a) | r+γv    |     | (s′)    | (5) |
|     |     |     | π     |          |     |             |        |         |     | π       |     |
s′,r
a
[1]. This recursion is the foundation of nearly all RL algorithms because it allows value
estimatestobeupdatedfromsamplesorfromothervalueestimates,aprocessknownas
bootstrapping. Policy-gradient methods take a different approach: rather than learning
a value function and deriving a policy from it, they parameterise the policy directly
and update its parameters by gradient ascent on the expected return. One widely used
policy-gradientmethodisProximalPolicyOptimisation(PPO).THoughnottehfocusof
the review it is the algorithm used in the Adaptive Stacking experiments revisited later
in this review [2], [9]. At a high level, PPO updates the policy in a way that prevents
eachupdatefrommovingthepolicytoofarfromthepreviousone,whichkeepstraining
stable. The algorithmic details are not central to the argument that follows.
The detail that matters for this review is that v , q , and any policy-gradient update
π π
π
are all expectations over trajectories generated by acting on a state representation.
7

When that representation is the true Markov state, these estimates converge correctly.
However when under partial observability the agent only ever sees an observation o ,
t
andanyvalueorpolicyitlearnsisimplicitlyafunctionofwhateverhistoryithaschosen
to condition on. The same PPO update applied to an enriched history representation
produces a fundamentally different policy than when applied to a single observation as
we would in the true Markov case, not because the algorithm changed, but because the
information the algorithm was given changed. A policy is only as good as the history it
conditions on. Memory mechanisms are therefore not a peripheral engineering detail;
they directly determine whether v and q are even meaningful estimates of long-run
π π
reward [2].
5.4 Deep Reinforcement Learning
Classical RL methods store value functions in lookup tables, which works for small
discrete state spaces but breaks down entirely when the agent must learn from high-
dimensional inputs such as raw pixels [10]. Deep reinforcement learning addresses
this by using neural networks as function approximators, enabling methods such as
DQN and PPO to operate directly on high-dimensional inputs that were previously in-
tractable for tabular RL [3], [10]. However, simply replacing a table with a neural
network was known to be unstable, often causing Q-values to diverge during training.
The Deep Q-Network (DQN) addressed this with two things, experience replay, which
stores past transitions in a buffer and samples them at random to not cause temporal
correlations in the training data to form, and a separate target network that is updated
only periodically to keep learning targets stable [3]. Because of this, DQN managed to
achieved human-level performance across a large set of Atari 2600 games directly from
raw pixel input, the first demonstration that deep RL could scale to high-dimensional
visual environments [3].
DeepRLalgorithmsareagainclassifiedaseitheron-policyoroff-policy[10]. Off-policy
RL methods such as DQN learn the value of an optimal policy from data generated
by a separate behaviour policy, which makes them more sample-efficient because past
experience can be reused through the replay buffer. On-policy methods such as PPO
updatethesamepolicythatisbeingusedtointeractwiththeenvironment, whichgives
more stable learning at some cost in sample efficiency. This distinction matters for
memory because off-policy methods can revisit old transitions from their buffer, while
on-policy methods must learn from whatever history representation is available during
the current rollout.
5.5 Memory Mechanisms in Neural Networks
Both deep RL and deep learning generally have had to confront the same underlying
problem, how should a neural network handle sequential inputs when a single ob-
servation is not enough to decide what to do? The first is to widen the input itself,
concatenating several consecutive observations into one fixed-size input vector before
anything else happens inside the network. The second is to add recurrent structure
to the network so that information can flow from one timestep to the next through
8

an internal hidden state. More broadly, memory-based approaches to partial observ-
ability include fixed-length sliding windows, variable-length tree-structured memories,
finite-state-machine memories, predictive state representations, and recurrent neural
networks [7]. They all share the premise that some function of the past must be re-
tained to give more clarity on the present, but they differ in what information is kept,
how it is updated and kept, and lastly, whether the agent has any control over these
kept histories.
5.6 Frame Stacking
Frame stacking was introduced as the memory appraoch inside DQN , and from there
it became the most prefered approach for handling temporal dependencies or obser-
vations in deep RL [3]. The idea is this, a single Atari frame is a static image (high
dimesionalinput),andastaticimagecarriesnoinformationaboutmotion,sotheagent
cannot tell from one frame alone its trajectory. DQN handles this by concatenating the
fourmostrecentframesintoasingleinputtensor,effectivelytreatingtheresultingstack
as if it were one higher-dimensional state. This is a fixed sliding window of length k,
and it works well when all the information the agent needs falls inside that window.
The combination of its simplicity and its empirical success in DQN is what made frame
stacking the standard pre-processing step across a wide range of deep RL methods [3].
Frame stacking relatively in this field of study is the simplest possible memory mecha-
nism because it uses a first-in-first-out (FIFO) buffer in which the newest observation
displaces the oldest, with the agent having no say in what enters or leaves the window
[7]. This works when the environment’s temporal dependencies are short and local,
as in Atari where four frames are enough to infer object motion. But when k∗ is large,
meaning the agent needs information from maybe far back in time to act optimally,
frame stacking faces three big problems. First, the stack size k must grow toward k∗
to capture the relevant dependencies, and increasing k causes the observation space to
grow proportionally, leading to computational and memory costs that scale linearly for
MLPandLSTMarchitecturesandquadraticallyforTransformers[2]. Second,theagent
requires oracle knowledge of k∗ to set k correctly, which is rarely available in practice.
Third, and most critically, the FIFO mechanism is indiscriminate, theres no conditional
logic, observations are discarded purely on the basis of recency, regardless of whether
they carry information the agent still needs. An observation that is highly informative
for future reward, such as an indicative financial signal observed many timesteps ago,
can be pushed out of the stack simply because it is no longer recentt. The quality of
information an agent conditions its decisions on directly determines the quality of its
learned policy [1]. What is contained in the memory stack should not be random and
passive, it should directly shape how well the agent can act, and a memory mechanism
that indiscriminately discards potentially valuable observations undermines the agent’s
ability to learn effectively.
9

5.7 Long Short-Term Memory
Framestackinghandlesshort-termdependencies,butmanysequentialtasksrequirethe
agent to use information from much further back in time than a fixed window of a few
frames can provide. Standard recurrent neural networks were the first architecture de-
signedtohandlelong-rangeinputs,buttheysufferedfromthevanishingandexploding
gradient problem [5]. When gradients are propagated back through many timesteps,
they tend to either shrink toward zero or grow without bound, making it very challeng-
ing for the network to learn long-range dependencies in practice [5]. Long Short-Term
Memory [5] (LSTM) was introduced to address exactly this problem. Each LSTM cell
contains a memory cell with a constant error carousel (CEC) that allows gradient in-
formation to flow across long time intervals without decaying, along with input and
output gates that learn when to write new information into the cell and when to read
information out of it [5]. With this gating structure, LSTMs were able to learn de-
pendencies over sequences far longer than anything standard recurrent networks could
reliably handle. In the context of RL, LSTMs have been applied as policy networks that
process sequences of observations and maintain a hidden state h that serves as a com-
t
pressedsummaryofthehistoryseensofar,allowingtheagenttoconditionitsdecisions
on more than just the current observation.
5.8 Recurrent Neural Networks and Selective Gating
LSTManditsvariantssuchastheGatedRecurrentUnit(GRU)handlememorythrough
learned gating, and on the surface it can look as though the agent is deciding what to
remember and what to forget. But that decision is implicit. The gates are parame-
ters that get updated through gradient descent, and the agent has no explicit discrete
action that lets it say “keep this observation, discard that one.” The memory that an
LSTM maintains is a byproduct of the model’s architecture and training, not an active
choice the agent makes in pursuit of reward. This matters in an RL setting because
when an LSTM is trained inside an agent, backpropagation through time (BPTT) has to
be truncated to a fixed-length window for computational inhibitions, and that trunca-
tion quietly reintroduces the same fixed-window limitation that frame stacking has [2].
LSTMshaveshownstrongperformanceonpartiallyobservableproblemswithvisualin-
puts, but the gating remains an implicit, environment-specific learned bias rather than
anexplicitdecisiontheagentmakes[2], [7]. Sothememoryselectionproblemhasnot
been solved by recurrent gating rather it has been shifted from the input layer into the
hidden state. The memory exists as a passive product of the model rather than as an
explicit construct the agent actively controls. This distinction between passive and ac-
tive memory management is precisely what motivates the Adaptive Stacking approach
discussed in the following section [2].
5.9 Limitations of Fixed Frame Stacking
Frame stacking, as established in the background, is a fixed-size FIFO buffer that con-
catenates the most recent k observations into a single input [3]. Two structural prob-
lems follow directly from this design. The first is that k must be chosen before training
10

begins, but the true dependency length k∗ is almost never known in advance. Practi-
tioners are left to either over-provision the stack, which wastes compute and memory,
or under-provision it, which means the agent physically cannot access observations it
needs to act optimally. The second problem is that the FIFO eviction rule discards ob-
servations strictly by recency, not by its importance. As k grows toward k∗ to capture
longer dependencies, the computational cost of inference and learning scales linearly
with k for MLP and LSTM architectures, and quadratically for Transformers [2]. So
increasing the stack to accommodate longer dependencies does not come for free, it
comes with a direct cost to the agent’s efficiency.
The T-Maze environment, originally introduced by Bakker [11] and used extensively in
the Adaptive Stacking experiments [2], makes this problem concrete. The agent starts
in a corridor and is shown a colour cue at the very first timestep that determines which
direction it must turn at the end of the corridor. The corridor has length L, so the
agent must retain the cue for L+2 timesteps before it becomes useful. Frame stacking
requires k = L + 2 to keep the cue in the window, but only one observation in that
entire stack, the initial colour cue, is actually important for the decision at the end.
Every other observation in the stack is a grey corridor tile that does not contributes.
This is not a some forced edge case. In a financial setting, the same pattern appears
when a structural signal such as a regime shift or a central bank decision is observed
once and then buried under a long sequence of noisy, uninformative price movements
[4]. Frame stacking has no mechanism for distinguishing between the signal and the
noise in its own buffer. That is the problem that the next subsection addresses.
5.10 Adaptive Stacking as a Memory Action
Adaptive Stacking (AS) addresses the limitations of frame stacking by giving the agent
explicit control over the contents of its memory stack [2]. The agent keeps a stack of
fixed size κ, and at each timestep it does two actions, an environment action a ∈ A
t
and a memory action i ∈ {1,...,κ} that selects which slot in the stack to overwrite
t
with the new observation. The stack update is then
s = push(pop(s , i ), x ) (6)
t+1 t t t+1
where s = [x ,...,x ] is the current memory state containing κ selected past obser-
t i1 iκ
vations [2]. Both actions are learned through the same reward signal, so the agent
learnswhattorememberinserviceofthesameobjectivethatshapesitsbehaviour. The
core argument is that in many environments with large k∗, only κ ≪ k∗ observations
are actually needed to predict the environment dynamics and act optimally [2]. The
large k∗ arises not because there are many relevant observations, but because the few
relevant ones are separated by long stretches of uninformative timesteps [2].
The paper provides both theoretical and empirical support for this approach. On the
theoretical side, Nangue Tasse et al. prove that any RL algorithm that converges un-
der frame stacking with k ≥ k∗ also converges under Adaptive Stacking with k ≥ κ,
provided the algorithm uses unbiased value estimates [2]. This convergence result ex-
tends to TD-learning under a value-consistency assumption that holds for a wide class
11

of goal-reaching and sparse-reward tasks. On the empirical side, the paper reports re-
sults across T-Maze variants, Rubik’s Cube, MiniGrid-Memory, and FetchReach. In all
domains, Adaptive Stacking with a stack of size κ matches or exceeds the performance
of an oracle frame stacking agent that has access to the full k∗ history, while substan-
tially outperforming frame stacking with the same limited budget of κ observations[2].
These results hold across MLP, LSTM, and Transformer architectures, which means the
method is architecture-agnostic and not tied to any particular network design [2]. The
paper is a 2025 preprint and has not yet undergone peer review which would help
quality control. All of the benchmark environments are controlled, episodic settings
where the informative observation is well-defined and the reward structure is sparse.
Non-stationary environments, noisy observations, and dense reward structures are not
tested [2]. The assumption that the relevant past is sparse, meaning κ ≪ k∗, is central
totheefficiencygains,andforthisresearchtopicitisnotclearwhetherthisassumption
holds in domains like financial markets where the relevance of past observations may
be diffuse rather than concentrated in a few critical timesteps. This is precisely the gap
that motivates applying Adaptive Stacking to financial time-series prediction.
5.11 Reinforcement Learning for Time-Series Prediction
Time-series prediction maps naturally onto the RL framework when the problem is
framedassequentialdecisions. Thedatasaresequential, theenvironmentisstochastic,
and the agent only sees a partial view of the underlying process at each step. Framing
the task as a policy that emits actions, whether in our case it be trading positions or
allocation weights, and is scored by an evaluative reward signal is a natural fit for the
RL formalism established in the background [1].
Moody and Saffell [4] were among the people that took this approach seriously for fi-
nancial markets. They relied upon a Recurrent Reinforcement Learning (RRL) system ,
whichparameterisesthetradingpolicydirectlyandoptimisesasaresultthedifferential
Sharpe ratio as its reward signal, bypassing the need for a separate forecasting model
entirely. Thepolicytakesafixedwindowofpastreturnsasinputandoutputsaposition
in {−1,0,+1} representing short, neutral, or long. On S&P 500 and T-Bill data ranging
25 years, RRL outperformed both Q-learning and a buy-and-hold baseline under their
experimental setup. The key observation for this review is that the input to RRL is ex-
actly the frame-stacking pattern discussed in the previous subsections. A fixed sliding
window of the most recent returns, with no mechanism for the agent to control what
is retained. A parallel pattern appears outside the financial domain. Hausknecht and
Stone [6] combined DQN with LSTM to create Deep Recurrent Q-Networks (DRQN),
demonstratingonAtariwithflickeringframesthatrecurrentmemorycouldpartiallyad-
dress the limitations of fixed windows in partially observable environments. However,
even DRQN relies on truncated backpropagation through time, which reintroduces a
fixed-window constraint on the effective memory horizon during training. The pattern
across these works is consistent: the RL framing for sequential, partially observable
problems is well-motivated, but the memory mechanism itself has remained an unex-
amined assumption inherited from early implementations.
12

| 5.12 | Time-Series | Prediction | in Financial | Domains |     |     |     |
| ---- | ----------- | ---------- | ------------ | ------- | --- | --- | --- |
Financialtime-seriespresentexactlytheattributesthataFSwindowhandlespoorly. Re-
turnsinfinanceexhibitlong-rangedependenciesdrivenbyregimeshifts,markettrends
, and structural market events that can influence prices for weeks or months after they
occur [8]. The observations which actually carry predictive information are sparse and
embedded within long sequences of noisy, largely uninformative price movements [8].
Financial markets are also non-stationary: the statistical properties of returns change
over time, so a memory mechanism that worked well in one regime may become irrele-
vant or even misleading in another. Hambly, Xu and Yang [8] provide a comprehensive
survey of RL applications in finance, covering portfolio optimisation, execution, market
making, option pricing, and related decision problems. Notably, the survey’s treatment
does not include a dedicated discussion of how agents should manage their observa-
tion history in these settings, which suggests that memory mechanism design has not
| yet been | a focus | of the financial | RL studies. |     |     |     |     |
| -------- | ------- | ---------------- | ----------- | --- | --- | --- | --- |
Adaptive Stacking as mentioned been validated on controlled benchmarks where the
informative observation is clearly defined, the reward is sparse, and the environment
dynamics are stationary [2]. Financial time-series violate all three of these condi-
tions,so exploring this using time series will evaluate whether AS works well , only
in well-defined environments. Therefore the relevant question is empirical rather than
theoretical, does the same memory-action mechanism that allows an agent to retain a
goalcueinaT-Mazealsoallowittoretainameaningfulfinancialsignalinanoisy, non-
stationary return series? This review does not answer that question, but it identifies it
as the specific open problem that motivates further investigation.
| 5.13 | Comparison | of Memory | Approaches |     |     |     |     |
| ---- | ---------- | --------- | ---------- | --- | --- | --- | --- |
The three memory mechanisms discussed in this review, frame stacking, LSTM, and
Adaptive Stacking, each represent a different point in the trade-off between simplic-
ity, flexibility, and agent control. Table 1 summarises these trade-offs across several
dimensions.
|           |     | Table 1: Comparison | of memory | mechanisms. |     |          |          |
| --------- | --- | ------------------- | --------- | ----------- | --- | -------- | -------- |
| Dimension |     | Frame Stacking      |           | LSTM        |     | Adaptive | Stacking |
Memory type Fixed FIFO window Continuous hidden state Learned stack
| Agent    | control  | None  |     | Implicit (gates) |      | Explicit | (memory action) |
| -------- | -------- | ----- | --- | ---------------- | ---- | -------- | --------------- |
| Scales   | with k∗  | O(k∗) |     | O(k∗) using      | BPTT | O(κ)     |                 |
| Requires | k∗ known | Yes   |     | No               |      | No       |                 |
Evaluated on Atari, benchmarks POMDPs, NLP T-Maze, MiniGrid, Rubik’s
| Financial | data | Yes (Moody | and Saffell) | Limited |     | Not yet |     |
| --------- | ---- | ---------- | ------------ | ------- | --- | ------- | --- |
Each mechanism trades something for something else. Frame stacking trades flexibility
for simplicity: it is trivial to implement and works well for short-term dependencies,
but it cannot scale to large k∗ without paying a proportional computational cost and
it gives the agent no control over what is retained [3]. LSTM trades explicit control
13

for continuous, gradient-trained memory: the gating mechanism can in principle learn
to retain long-range information, but the memory remains an implicit byproduct of
the architecture rather than a deliberate decision by the agent, and truncated BPTT
reintroduces a fixed-window constraint during training [2], [5]. Adaptive Stacking
trades architectural simplicity for an additional learned action: the agent gains explicit
control over its memory contents, but at the cost of an expanded action space and the
assumption that the relevant history is sparse [2]. The final column highliths the key
observation that Adaptive Stacking has not been evaluated on financial data or time-
series data in general. Whether its advantages over frame stacking and LSTM carry
over to a noisy, non-stationary domain remains an open empirical question.
5.14 Summary and Gap
Thisreviewhasexaminedtheevolutionofmemorymechanismsinreinforcementlearn-
ing, from foundational RL concepts through deep RL and fixed memory approaches to
the recent Adaptive Stacking meta-algorithm, with a focus on how these developments
relate to financial time-series prediction. The specific problem addressed is that RL
agents operating in partially observable environments need memory to act effectively,
and the dominant memory mechanism in deep RL, frame stacking, has structural limi-
tations that become costly in domains with long-range dependencies [2], [3].
The literature comes to a clear finding that memory is not a peripheral implementa-
tion detail. The choice of history representation directly determines whether value and
policy estimates are meaningful, and by extension whether the agent can learn an ef-
fective policy at all. Frame stacking is simple and effective for short-term dependencies
but cannot scale tolarge k∗ without proportional computational cost, and its FIFOevic-
tion rule discards observations by recency rather than by relevance. LSTM extends the
effective memory horizon through learned gating but keeps memory control implicit
and inherits a fixed-window limitation through truncated backpropagation. Adaptive
Stacking gives the agent an explicit memory action with a bounded budget κ ≪ k∗ and
provides formal convergence guarantees under standard assumptions [2]. However,
it has been validated only on controlled, episodic benchmarks where the informative
observation is well-defined. The memory action space itself grows with κ, which in-
troduces its own scaling problmes. And the assumption that the relevant past is sparse
may not hold uniformly across all domains.
The gap that remains is specific and testable. Financial time-series prediction is a do-
main where long-range dependencies are prevalent, informative signals are sparse rel-
ative to noise, and the underlying dynamics are non-stationary [4], [8]. If the memory
mechanismistheunexaminedassumptioninpriorfinancialRLwork,thentryingitwith
an explicit memory-action formulation is a prospective but concrete and testable im-
provement. Whether Adaptive Stacking’s advantages carry over from controlled bench-
marks to this setting is the open question this review identifies.
14

6 Methodology
The methodology is structured around the three-step comparison set out by the re-
search questions. Each step is implemented as a controlled experiment that changes a
single variable while holding everything else fixed, so the contribution of each design
choice can be attributed cleanly. All four configurations described below are trained
and evaluated together, rather than running a knockout in which the winner of one
round advances to the next. Reporting all four means each pairwise comparison can be
made directly from the same set of results, and no baseline is discarded before it has
done its work.
6.1 Environment and Data
The task is positioning on the S&P 500 conditional on a stream of VIX observations.
VIX is a natural choice of observation for this work for two reasons. First, it has a clear
regime structure, in that long, mean-reverting low-volatility stretches are punctuated
by sharp spikes that mark stress regimes, and those spikes can sit weeks back in the
data and still matter for what the market is doing now. Second, this is exactly the
sparse-signal-in-noisy-history pattern that Adaptive Stacking is built for [2]. Most daily
VIX readings carry little new information, but a handful of historic spikes carry the
signal that defines the current regime. The setup is therefore structurally honest about
the central assumption Adaptive Stacking depends on, namely that the relevant past is
sparse [2].
AnaturalfirstinstinctistouseVIXasbothobservationandtargetandkeepthetasktoa
single data source. The problem with that is pure prediction (next VIX value or regime
classification on VIX itself) does not benefit from the RL framing in any meaningful
way. There is no sequential consequence to a prediction. A wrong call at time t does
not change the agent’s situation at t + 1, and the supervised baseline would trivially
be at least as good. The reason RL framings have traction on financial time-series
is that positions accumulate over time, transaction costs compound, and drawdowns
matter [4], [8]. Using the S&P 500 as the underlying for positions, with VIX as the
observation, gives the agent a real sequential decision problem while keeping the data
sources to two daily series.
At each timestep the agent receives the current VIX value and selects a position in
{−1,0,+1} on the S&P 500, where −1 is short, 0 is flat, and +1 is long [4]. The
rewardisthechosenpositionmultipliedbythenext-dayS&P500return, minusasmall
fixed transaction cost when the position changes. The supervised baseline uses the
same observation stream to forecast next-day return and converts that forecast to a
position via a fixed threshold rule, so both paradigms produce the same kind of output
(a position) on the same kind of input (a VIX history).
The dataset covers daily VIX and S&P 500 closes over a multi-year period. The data
is split chronologically into train, validation, and test segments, with the test segment
held out until final evaluation. Because financial returns are non-stationary [8], the
test segment is evaluated using walk-forward windows. The model is retrained or fine-
15

tuned at the start of each window, and performance is reported across windows rather
than as a single number on the full test set. This avoids drawing conclusions from a
| single, | possibly | atypical, | period. |     |     |     |     |
| ------- | -------- | --------- | ------- | --- | --- | --- | --- |
6.2 Algorithm
The reinforcement learning configurations all use Proximal Policy Optimisation (PPO)
asthebasealgorithm[9]. PPOisthesamebasealgorithmusedintheAdaptiveStacking
experiments[2],sothethirdresearchquestion’scomparisonsitsonafaithfulreproduc-
tion of that paper’s setup rather than introducing a new algorithmic variable on top of
the memory mechanism. The policy and value networks are multi-layer perceptrons
with the same hidden-layer widths across all RL configurations. The Adaptive Stack-
ing variant adds a memory action head that selects which slot in the stack of size κ
to overwrite at each timestep, alongside the environment action head. Everything else
(optimiser, clip range, batch size, number of update epochs) matches across configura-
tions.
6.3 Baselines
Four configurations are trained and evaluated. Each pairwise comparison answers one
| research | question. |     |     |     |     |     |     |
| -------- | --------- | --- | --- | --- | --- | --- | --- |
1. Supervised learning baseline. An LSTM forecaster [5] trained on the VIX obser-
vation stream to predict the next-day S&P 500 return. The forecast is converted
|     | to a position |     | via a fixed | threshold | rule. |     |     |
| --- | ------------- | --- | ----------- | --------- | ----- | --- | --- |
2. Reinforcement learning, no memory. A PPO agent that observes only the cur-
rent VIX value [9]. This is equivalent to Frame Stacking with k = 1 and is the
|     | floor for | what | an RL | agent achieves | without | memory. |     |
| --- | --------- | ---- | ----- | -------------- | ------- | ------- | --- |
3. Reinforcement learning with Frame Stacking. A PPO agent that observes a
|     | fixed-size | window | of  | the most | recent k VIX | values | [3]. |
| --- | ---------- | ------ | --- | -------- | ------------ | ------ | ---- |
4. Reinforcement learning with Adaptive Stacking. A PPO agent with an addi-
tional memory action that selects which slot in a bounded stack of size κ to over-
|     | write at | each | timestep | [2]. |     |     |     |
| --- | -------- | ---- | -------- | ---- | --- | --- | --- |
Abuy-and-holdpositionontheS&P500overthetestwindowisalsoreportedalongside
these four, as a financial sanity baseline rather than a methodological one. Any agent
that fails to beat buy-and-hold is not earning the right to call itself a useful trader,
| regardless | of         | how | the memory-mechanism |     | comparison |     | turns out. |
| ---------- | ---------- | --- | -------------------- | --- | ---------- | --- | ---------- |
| 6.4        | Evaluation |     | Metrics              |     |            |     |            |
The primary metrics are financial. Cumulative return on the test set is the headline
number. The Sharpe ratio gives a risk-adjusted comparison. Maximum drawdown
captures the worst loss the agent would have inflicted on a real account [8]. The
secondary metrics are RL-side: sample efficiency, measured as return as a function of
16

trainingtimesteps;wall-clocktrainingtime;andmemoryfootprintasafunctionofstack
size. For the third research question specifically, an additional diagnostic looks at what
the Adaptive Stacking agent actually retains over time. If Adaptive Stacking is doing
what the method claims, the retained observations should cluster around identifiable
marketevents,particularlyVIXspikesandregimetransitions,ratherthanbeinguniform
across the trailing history.
6.5 Hyperparameters and Sweeps
Both Frame Stacking and Adaptive Stacking are swept over stack size ∈ {3,5,10,20}.
Thethirdresearchquestionisreportedatmatchedbudget: FrameStackingatk against
Adaptive Stacking at κ = k for each value in the sweep. Each configuration is trained
with multiple random seeds so that variance can be reported alongside means and the
comparison is not riding on a single lucky initialisation. The PPO hyperparameters
(learning rate, clip range, batch size, number of update epochs) follow the defaults
usedintheAdaptiveStackingpaper[2],soanyperformancedifferencebetweenFrame
Stacking and Adaptive Stacking is attributable to the memory mechanism rather than
to unrelated tuning choices.
6.6 Expected Outcomes
If the first research question is answered in favour of reinforcement learning, the
methodology continues into the memory comparisons. If it is answered in favour of su-
pervisedlearning,thatisitselfameaningfulfinding: itwouldsaythatonnon-stationary
financial time-series, the sequential framing of RL adds nothing the supervised fore-
caster does not already have. For the second research question, prior work has shown
thatframestackinghelpsovernomemoryonpartiallyobservabletasks[3], soreplicat-
ing this on financial data is a sanity check more than a contribution. The third research
question is where the contribution lives. A positive result extends the Adaptive Stack-
ing advantage from the stationary, episodic benchmarks in [2] into the non-stationary
time-series setting. A negative result is an honest report that the sparsity assumption
Adaptive Stacking depends on does not transfer to this domain, which is still a useful
finding because it tells the next person what not to try. Either way, this work is the
first empirical evaluation of Adaptive Stacking on financial time-series, and that is the
contribution this proposal commits to.
7 Risks and Limitations
Three risks are worth flagging up front, alongside their contingencies.
The first is that the central assumption Adaptive Stacking depends on, namely κ ≪ k∗,
may not hold in financial data. The benchmark environments Adaptive Stacking was
validatedon[2]haveasmallnumberofclearlyinformativeobservationsburiedinlong
stretches of uninformative ones, such as a T-Maze colour cue or a Rubik’s Cube starting
configuration. Financial signals are often diffuse rather than concentrated, meaning
17

many marginally informative observations rather than a few decisive ones. If that is
the case here, Adaptive Stacking may not show an advantage over Frame Stacking at
any κ, and the contribution becomes a negative empirical result rather than a positive
one. A negative result is still a contribution, but it is worth naming the risk now rather
than pretending it does not exist.
The second is that the convergence guarantee for Adaptive Stacking in [2] is proved
for stationary, k∗-Markov environments. Financial time-series are not stationary. The
theorem does not promise anything about what happens when the environment’s dy-
namics shift mid-episode, and walk-forward retraining is a partial mitigation but not a
fix. Treating Adaptive Stacking as a method with theoretical guarantees in this setting
would overclaim. The empirical comparison reported here stands on its own merits.
The third is compute. A full sweep across four stack sizes, two memory mechanisms,
multiple random seeds, and walk-forward windows is more compute than the rest of
the experiments combined. The contingency is to reduce the seed count from a target
of five down to three first, and then to shrink the sweep grid from four points down to
three ({3,10,20}), in that order. This keeps the four-configuration comparison intact
at the cost of statistical confidence on each individual data point, which is the right
trade-off when the comparison structure is the load-bearing part of the contribution.
8 Workplan
The workplan is structured month by month. The first two months are work that has
already been completed and is included for completeness.
March Annotatedbibliographysubmitted. Thefoundationalpaperscoveringreinforce-
ment learning, deep RL, LSTM, financial RL, and the Adaptive Stacking paper it-
self are read and summarised. The bibliography is the basis for the citations used
throughout the proposal and literature review.
April Literature review drafted and submitted. The review traces the evolution of
memory mechanisms in deep RL from frame stacking through LSTM to Adaptive
Stacking, and situates this within financial time-series prediction.
May Proposal finalised. Data pipeline built: daily VIX and S&P 500 closes pulled,
cleaned, and split into chronological train, validation, and test segments. The
environment is wrapped as a gymnasium-style class with the observation space,
action space, and reward function defined. The supervised learning baseline is
scaffolded, with an LSTM forecaster trained on the train segment and a thresh-
olded trading rule for converting forecasts to positions.
June The no-memory PPO agent is implemented and trained. The first comparison
(supervised baseline versus no-memory RL) is run and the first research question
is reported on. PPO hyperparameters are tuned just enough to make the agent
not obviously broken; deeper tuning is held until the matched-budget sweep.
18

July Frame Stacking is added to the PPO agent. The sweep over k ∈ {3,5,10,20} is
run, and the second comparison (no-memory RL versus Frame Stacking RL) is
reported on, answering the second research question.
August Adaptive Stacking is implemented on top of the same PPO base used the previ-
ousmonth. Thematched-budgetsweepoverκ ∈ {3,5,10,20}isrun,andthethird
comparison is reported on, answering the third research question. The diagnostic
for what Adaptive Stacking actually retains in its stack over time is computed in
this month.
September Analysis and write-up. Walk-forward evaluation is run across the test seg-
ment for the configurations that are still in contention. Final figures and tables
are built. A draft of the full report is shared with the supervisor for feedback.
October Revisions and submission.
19

References
[1] R. S. Sutton and A. G. Barto, Reinforcement Learning: An Introduction, 2nd ed.
| Cambridge, | MA: MIT | Press, 2018. |     |     |
| ---------- | ------- | ------------ | --- | --- |
[2] G. Nangue Tasse, M. Riemer, B. Rosman, and T. Klinger, Beyond sliding win-
dows:Learningtomanagememoryinnon-Markovianenvironments,arXivpreprint
| arXiv:2512.19154v1, |     | 2025. |     |     |
| ------------------- | --- | ----- | --- | --- |
[3] V. Mnih, K. Kavukcuoglu, D. Silver, A. A. Rusu, J. Veness, M. G. Bellemare, A.
Graves, M. Riedmiller, A. K. Fidjeland, G. Ostrovski, et al., “Human-level control
through deep reinforcement learning,” Nature, vol. 518, no. 7540, 2015.
[4] J.MoodyandM.Saffell,“Learningtotradeviadirectreinforcement,”IEEETrans-
| actions | on Neural Networks, | vol. | 12, no. 4, 2001. |     |
| ------- | ------------------- | ---- | ---------------- | --- |
[5] S. Hochreiter and J. Schmidhuber, “Long short-term memory,” Neural Computa-
| tion, vol. | 9, no. 8, | pp. 1735–1780, | 1997. |     |
| ---------- | --------- | -------------- | ----- | --- |
[6] M. Hausknecht and P. Stone, “Deep recurrent Q-learning for partially observable
| MDPs,” | 2015. |     |     |     |
| ------ | ----- | --- | --- | --- |
[7] A. Demir, Learning what to memorize: Using intrinsic motivation to form useful
memoryinpartiallyobservablereinforcementlearning,arXivpreprintarXiv:2110.12810v2,
2022.
[8] B. Hambly, R. Xu, and H. Yang, “Recent advances in reinforcement learning in
| finance,” | Mathematical | Finance, | vol. 33, no. | 3, 2023. |
| --------- | ------------ | -------- | ------------ | -------- |
[9] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, “Proximal policy
optimization algorithms,” arXiv preprint arXiv:1707.06347, 2017.
[10] M.GhasemiandD.Ebrahimi,Introductiontoreinforcementlearning,arXivpreprint
| arXiv:2408.07712v3, |     | 2024. |     |     |
| ------------------- | --- | ----- | --- | --- |
[11] B.Bakker,“Reinforcementlearningwithlongshort-termmemory,”vol.14,2002.
20