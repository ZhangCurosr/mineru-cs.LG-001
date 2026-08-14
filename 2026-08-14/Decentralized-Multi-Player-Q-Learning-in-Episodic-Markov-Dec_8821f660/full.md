# Decentralized Multi-Player Q-Learning in Episodic Markov Decision Processes with Information Asymmetry

1<sup>st</sup> Larissa Xu

2<sup>nd</sup> King Bi

Department of Mathematics

Department of Computer Science

University of California, Los Angeles

3<sup>rd</sup> William Chang

Los Angeles, USA

University of California, Los Angeles

Los Angeles, USA

Department of Mathematics

xuzhiyun004119@g.ucla.edu

king0508@g.ucla.edu

University of California, Los Angeles

Los Angeles, USA

chang314@g.ucla.edu

Abstract—We study decentralized multi-player reinforcement learning in episodic tabular Markov decision processes (MDPs) under three forms of information asymmetry: (A) unobserved actions with common rewards, (B) observed actions with independent rewards, and (C) unobserved actions with independent rewards. Players cannot communicate during learning but may agree on a protocol a priori. For Problems A and B we propose mQ-learning and mQ-learning-intervals, achieving $\mathbf { \tilde { O } } ( \sqrt { H ^ { 4 } S A _ { \mathrm { j o i n t } } T } )$ regret, where H is the horizon, S the state count, T = KH the total steps, and $\begin{array} { r } { A _ { \bf j o i n t } = \prod _ { i = 1 } ^ { M } | \mathcal { A } _ { i } | } \end{array}$ the joint action space across M players. For Problem C we give mEXC and mEXC-Bellman, two-phase explore-then-commit algorithms with regret $\tilde { O } ( H ( S A _ { \mathbf { j o i n t } } ) ^ { 1 \tilde { / } 3 } T ^ { 2 / 3 } )$ . Against the centralized jointaction benchmark, decentralized learning under information asymmetry matches the single-agent Q-learning rate of [1] up to logarithmic factors. Because A<sub>joint</sub> grows exponentially in M, the bounds are most meaningful for small M or small per-player action sets.

Index Terms—multi-player reinforcement learning, Markov decision processes, Q-learning, information asymmetry, regret bounds, decentralized learning

## I. INTRODUCTION

Multi-agent reinforcement learning (MARL) arises in cooperative systems such as communication networks [2], multirobot coordination, and distributed resource allocation [3], where agents must coordinate without centralized control or explicit communication during learning.

The multi-armed bandit (MAB) framework is well understood in the single-player setting [4], [5]. Extensions to the multi-player setting [6] reveal that information asymmetry between players introduces substantial coordination challenges. Prior work on cooperative multi-player bandits has studied several asymmetry models [7], [8], but these results are limited to the stateless bandit setting, whereas many applications have state structure that agents must learn to navigate.

In the single-agent episodic MDP setting, Jin et al. [1] showed that a model-free Q-learning algorithm with upper confidence bounds achieves $\tilde { O } ( \sqrt { H ^ { 4 } S A T } )$ regret, nearoptimal up to polynomial factors in H. A natural question is whether similar guarantees hold for the multi-player setting with information asymmetry, or whether asymmetry imposes an additional cost beyond the centralized joint-action rate.

The key insight, first observed in the bandit setting [7], is that players can use a pre-agreed deterministic protocol to implicitly coordinate without communication: if all players follow the same deterministic algorithm with access to the same information, they independently reach the same decisions. We extend this to episodic MDPs, where coordination must also account for state transitions and multi-step planning.

a) Our Contributions:

• For Problem A (unobserved actions, common rewards), mQ-learning uses a lexicographic ordering of joint actions to coordinate implicitly and achieves $\tilde { O } ( \sqrt { H ^ { 4 } S A _ { \mathrm { j o i n t } } T } )$ regret (Theorem 3), where $A _ { \mathrm { j o i n t } } ~ =$ $\begin{array} { r } { \prod _ { i = 1 } ^ { M } | \mathcal { A } _ { i } | . } \end{array}$

• For Problem B (observed actions, independent rewards), mQ-learning-intervals maintains upper and lower confidence bounds at each state-action pair to coordinate action elimination across players, with the same $\tilde { O } ( \sqrt { H ^ { 4 } S A _ { \mathrm { j o i n t } } T } )$ regret (Theorem 5).

• For Problem C (unobserved actions, independent rewards), mEXC and mEXC-Bellman are explore-thencommit algorithms. With $K ^ { \prime } ~ \asymp ~ ( S A _ { \mathrm { i o i n t } } ) ^ { 1 / 3 } K ^ { 2 / 3 }$ , we obtain $\tilde { O } ( H ( S A _ { \mathrm { j o i n t } } ) ^ { 1 / 3 } T ^ { 2 / 3 } )$ regret (Theorem 7).

• We prove auxiliary lemmas on weighted learning rates and interval widths that may be of independent interest, and show that Algorithm 1 is operationally equivalent to centralized joint-action Q-learning—the agreement is achieved implicitly through deterministic tie-breaking rather than through observation.

b) Role of M.: Our bounds match the single-agent rate relative to the joint-action benchmark, but grow exponentially in M through $A _ { \mathrm { j o i n t } }$ . This is unavoidable in tabular MARL: even a centralized learner has $\Omega ( \sqrt { S A _ { \mathrm { j o i n t } } T } )$ regret [1]. The contribution is therefore that asymmetry imposes no additional factor beyond the centralized rate, not that asymmetry is free in absolute terms.

c) Organization.: Section II sets up the model. Section III treats Problems A and B; Section IV treats Problem C. Section VIII discusses related work; proofs are in the Appendix.

## II. PRELIMINARIES

## A. Multi-Player Episodic MDP Model

We consider a multiplayer episodic tabular MDP $( \mathcal { P } , \mathcal { S } , \mathcal { A } , H , \mathbb { P } , r )$ . Here $\begin{array} { c c l } { \mathcal { P } } & { = } & { \{ P _ { 1 } , \ldots , P _ { M } \} } \end{array}$ is the set of M players and S is the state space with $| S | = S$ . Each player $P _ { i }$ has actions $\mathcal { A } _ { i }$ , and the joint action space is $\mathcal { A } = \mathcal { A } _ { 1 } \times \cdot \cdot \cdot \times \mathcal { A } _ { M }$ with $\begin{array} { r } { | \mathcal { A } | = A _ { \mathrm { j o i n t } } = \prod _ { i = 1 } ^ { M } | \mathcal { A } _ { i } | } \end{array}$ , which scales as $A _ { \mathrm { m a x } } ^ { M }$ in the worst case—exponential in M. We use $A$ and $A _ { \mathrm { j o i n t } }$ interchangeably in regret bounds. $\mathbb { P } _ { h }$ is the set of unknown transition kernels at step h, one for each joint action $a \in { \mathcal { A } } ;$ entry $( \mathbb { P } _ { h } ) _ { i j } ( { \pmb a } )$ is the probability of moving from $x _ { i }$ to $x _ { j }$ under a. The horizon is $H ,$ and the reward r depends on the joint action. Without loss of generality, states are layered: the initial state is $s _ { 0 }$ and the final state is $s _ { H }$

At each time step, each player observes the current state and picks an arm from their set. The M-tuple of arms is denoted $\pmb { a } = ( a _ { 1 } , \dots , a _ { M } )$ . Throughout this paper, vectors and tuples are denoted in boldface. This generates a random reward $X _ { a } \in [ 0 , 1 ]$ from a 1-subgaussian distribution $F _ { a }$ with mean $\mu _ { a }$ . After performing the joint action, the players transition to a new state according to the unknown transition kernel $\mathbb { P } _ { h }$

## B. Value Functions and Bellman Equation

A policy π specifies, for each step h and state x, a joint action a. The value function of a policy π at step h and state x is defined as

$$
V _ { h } ^ { \pi } ( x ) = \mathbb { E } _ { \pi } \left[ \sum _ { h ^ { \prime } = h } ^ { H } r _ { h ^ { \prime } } ( x _ { h ^ { \prime } } , \pmb { a } _ { h ^ { \prime } } ) \ \middle | \ x _ { h } = x \right]\tag{1}
$$

and the corresponding action-value function (Q-function) is

$$
Q _ { h } ^ { \pi } ( x , \pmb { a } ) = r _ { h } ( x , \pmb { a } ) + \mathbb { E } _ { { \boldsymbol { x } } ^ { \prime } \sim \mathbb { P } _ { h } ( \cdot | { \boldsymbol { x } } , \pmb { a } ) } [ V _ { h + 1 } ^ { \pi } ( x ^ { \prime } ) ] .\tag{2}
$$

The optimal value function satisfies the Bellman optimality equation:

$$
Q _ { h } ^ { \ast } ( x , \pmb { a } ) = r _ { h } ( x , \pmb { a } ) + [ \mathbb { P } _ { h } V _ { h + 1 } ^ { \ast } ] ( x , \pmb { a } ) ,\tag{3}
$$

where $\begin{array} { r l r } { V _ { h } ^ { * } ( x ) } & { { } = } & { \operatorname* { m a x } _ { a } Q _ { h } ^ { * } ( x , a ) } \end{array}$ and $\begin{array} { r l } { [ \mathbb { P } _ { h } f ] ( x , a ) } & { { } = } \end{array}$ $\mathbb { E } _ { { x ^ { \prime } } \sim \mathbb { P } _ { h } \left( \cdot | { x } , { \pmb a } \right) } [ f ( { x ^ { \prime } } ) ]$

## C. Regret and Learning Objective

The players want to collectively identify the best policy, i.e., one that returns the highest rewards in expectation. This takes into account both the true reward means and the probability transition matrices. However, the players know neither the means $\mu _ { a }$ nor the transition functions $\mathbb { P } _ { h }$ . They must learn by playing and exploring. We capture learning efficiency via the per-player expected regret:

$$
R _ { T } = \sum _ { k = 1 } ^ { K } \left[ V _ { 1 } ^ { \ast } ( x _ { 1 } ^ { k } ) - V _ { 1 } ^ { \pi _ { k } } ( x _ { 1 } ^ { k } ) \right]\tag{4}
$$

where K is the number of episodes, $T = K H$ is the total number of steps, and $\pi _ { k }$ is the policy used in episode k. Our goal is to design decentralized algorithms with sublinear regret. Fundamental results for single-player MAB problems [4] suggest an Ω(log T) lower bound for the multi-player problem as well. In the episodic MDP setting, $\tilde { \Omega } ( \sqrt { H ^ { 2 } S A T } )$ is a known lower bound [1]. We specifically exclude any explicit communication between players during learning, though they may coordinate a priori.

## D. Information Asymmetry Models

We consider three types of information asymmetry, following [7].

a) Problem A: Action Asymmetry with Common $R e \mathrm { - }$ wards: Consider a set of M players $P _ { 1 } , \ldots , P _ { M }$ , where player $P _ { i }$ has a set $\kappa _ { i }$ of $K _ { i }$ arms. At each time instant, each player picks an arm from their set with the M-tuple denoted by ${ \textbf { } } ^ { a } =$ $( a _ { 1 } , \dotsc , a _ { M } )$ . The reward to all players equals $X _ { a ( t ) } ( t )$ , i.e., the reward is common, depends on the joint action $\mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf \Psi \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf \Psi \mathbf { } \mathbf { } \mathbf { } \mathbf \Psi \mathbf { } \mathbf { } \mathbf \Psi \mathbf { } \mathbf { } \mathbf \Psi \mathbf { } \mathbf \Psi \Psi \mathbf { } \mathbf \Psi \mathbf { } \mathbf \Psi \mathbf { } \mathbf \Psi \mathbf { } \mathbf \mathbf { } \mathbf \Psi \mathbf \Psi \Psi \Psi \mathbf { } \mathbf \mathbf \Psi \Psi \mathbf \Psi \Psi \mathbf \Psi \Psi \mathbf \Psi \Psi \mathbf \Psi \mathbf \Psi \Psi \mathbf \Psi \Psi \mathbf \Psi \mathbf \Psi \mathbf $ , and is independent across time. Crucially, no player can observe the actions of the other players. Formally, the information available to player i at time t is: $\{ x _ { h } ^ { k } , r _ { h } ^ { k } : k \le t , h \in [ H ] \}$ together with their own action history $\{ a _ { i } ^ { h , k } : k \le t , h \in [ H ] \}$

b) Problem B: Reward Asymmetry with Observed Actions: Each agent can observe the actions of all other agents, but the rewards are independent. If the arms pulled at time t are $\mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf \Psi \mathbf { } \Psi \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf \Psi \Psi \mathbf { } \mathbf { } \mathbf \Psi \Psi \mathbf { } \mathbf { } \mathbf \Psi \Psi \mathbf { } \mathbf { } \mathbf \Psi \mathbf { } \mathbf \Psi \Psi \mathbf { } \mathbf \Psi \Psi \mathbf { } \mathbf \Psi \mathbf { } \mathbf \Psi \Psi \mathbf \Psi \Psi \Psi \mathbf \Psi \Psi \Psi \mathbf \Psi \Psi \mathbf \Psi \Psi \mathbf \Psi \Psi \mathbf \Psi \Psi \mathbf \Psi \Psi \mathbf \Psi \mathbf \Psi \Psi \mathbf \Psi \mathbf \Psi \mathbf \Psi \Psi \mathbf \Psi$ , each player gets an i.i.d. copy of the reward: player i receives $X _ { \pmb { a } ( t ) } ^ { i } ( t )$ drawn independently from $F _ { \pmb { a } ( t ) }$ Since rewards are i.i.d., the expected regret is the same for all players. The information available to player i is: $\{ x _ { h } ^ { k } , \mathbf { \pmb { a } } _ { h } ^ { k } , r _ { h } ^ { i , k } : k \le t , h \in [ H ] \}$ , i.e., full action information but only their own rewards.

c) Problem C: Full Asymmetry: No player can observe the actions of the others, and rewards are independent from the same distribution. This combines the challenges of both Problems A and B. The information available to player i is only: $\{ x _ { h } ^ { k } , a _ { i } ^ { h , k } , r _ { h } ^ { i , k } : k \leq t , h \in [ H ] \}$

Remark 1: Problem C is the most general of the three, and Problem A is not a special case of Problem B (or vice versa), since they involve different types of asymmetry. The hierarchy is: Problem C generalizes both A and B.

## III. MAIN RESULTS

## A. Problem A: Asymmetry in Actions

The key difficulty in Problem A is that players cannot observe each other’s actions and must therefore coordinate implicitly. We use a lexicographic ordering from [7] to break ties deterministically.

Definition 2 (Lexicographic Order): For two M-tuples ${ \bf { \delta } } _ { \bf { { \delta } } } _ { \bf { { \delta } } } =$ $( x _ { 1 } , \ldots , x _ { M } )$ and $\pmb { y } = ( y _ { 1 } , \dots , y _ { M } )$ , we say $\textbf { \em x } < \textbf { \em y }$ if and only if there exists an n such that $\forall i < n , x _ { i } = y _ { i }$ , and $x _ { n } < y _ { n }$

Intuitively, $\mathbf x < \mathbf y$ if x is smaller than y when viewed as an M-digit number. When each player has the same number of actions A, we can view $\mathbf { \pmb { a } } \in \mathcal { A }$ as a number in base A. This definition remains applicable even when players have different numbers of actions.

Algorithm 1 mQ-learning (for each player P<sub>i</sub>)   
1: Initialize $Q _ { h } ^ { k } ( i , x , \pmb { a } )  H , \ N _ { h } ( i , x , \pmb { a } )  0$ for all   
$( x , \mathbf { a } , h ) \in \ddot { S } \times \mathcal { A } \times [ H ] .$   
2: for episode $k = 1 , \ldots , K$ do   
3: Receive initial state $x _ { 1 }$   
4: for step $h = 1 , \ldots , H$ do   
5: Pick $\mathbf { \delta } _  \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathrm { \~ \textit ~ { ~ a ~ } ~ } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \beta } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathbf { ~ } { ~ } \mathbf { \alpha } \mathbf { \beta } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm \mathrm { ~ } \mathrm { ~ } \mathrm \mathrm { ~ } \mathrm { ~ } \mathrm \mathrm { ~ ~ } \mathrm \mathrm { ~ } \mathrm \mathrm { ~ } \mathrm \mathrm { ~ ~ } \mathrm \mathrm { ~ } \mathrm \mathrm \mathrm { ~ } \mathrm \mathrm { ~ ~ } \mathrm \mathrm \mathrm { ~ ~ } \mathrm \mathrm \mathrm \mathrm { ~ ~ } \mathrm \mathrm \mathrm \mathrm { ~ ~ } \mathrm \mathrm \mathrm \mathrm \mathrm  ~ ~ \mathrm ~ \mathrm \mathrm ~ \mathrm \mathrm ~ \mathrm \mathrm \mathrm ~ \mathrm ~ \mathrm \mathrm \mathrm ~ \mathrm \mathrm \mathrm ~ \mathrm \mathrm \mathrm ~ \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm $ as the smallest action (via Definition 2) in   
arg $\operatorname* { m a x } _ { \pmb { a } ^ { \prime } } Q _ { h } ^ { k } ( i , x _ { h } , \pmb { a } ^ { \prime } ) .$   
6: Observe common reward $r _ { h }$ and next state $x _ { h + 1 } .$   
7: $t = N _ { h } ( i , x _ { h } , { \bf { a } } _ { h } )  N _ { h } ( i , x _ { h } , { \bf { a } } _ { h } ) + 1 .$   
8: $b _ { t }  c \sqrt { H ^ { 3 } \iota / t } .$   
9: $Q _ { h } ^ { k } ( i , x _ { h } ^ { \cdot } , \pmb { a } _ { h } )  ( 1 - \alpha _ { t } ) Q _ { h } ^ { k - 1 } ( i , x _ { h } , \pmb { a } _ { h } )$   
$+ \alpha _ { t } [ r _ { h } + V _ { h + 1 } ^ { k - 1 } ( x _ { h + 1 } ) + b _ { t } ] .$   
10: $V _ { h } ^ { k } ( x _ { h } ) \gets \operatorname* { m i n } \{ H , \operatorname* { m a x } _ { \pmb { a } ^ { \prime } } Q _ { h } ^ { k } ( i , x _ { h } , \pmb { a } ^ { \prime } ) \}$   
11: end for   
12: end for

The key observation is that in Problem A, all players receive the same reward and observe the same state transitions. Therefore, if all players run the same deterministic algorithm with the same tie-breaking rule, they will independently maintain identical Q-value estimates and select the same joint action at every step—without needing to observe each other’s actions.

The learning rate is $\alpha _ { t } ~ = ~ ( H + 1 ) / ( H + t )$ and $\iota =$ log $( S A _ { \mathrm { j o i n t } } T / p )$ . The algorithm is essentially the UCB-based Q-learning of [1] applied to the joint action space, with the deterministic tie-breaking of Definition 2 ensuring implicit coordination. Operationally, Algorithm 1 matches centralized joint-action Q-learning: no player observes the others’ actions, but because rewards and transitions are common and tie-breaking is deterministic, every player’s Q-table evolves identically and they select the same joint action at every step.

a) Synchronization under repeated tie-breaking.: Stability of implicit coordination follows from a one-line invariant. All players initialize $Q _ { h } ^ { 1 } \equiv H ,$ , so the arg max set is identical and the lexicographic-smallest element is uniquely determined. $\operatorname { I f } Q _ { h } ^ { k }$ is identical across players at the start of episode k, then in Problem A the update on line 8 depends only on quantities common to all players (same reward $r _ { h } .$ , same next state $x _ { h + 1 }$ same visit count t), so $Q _ { h } ^ { k + 1 }$ remains identical. The invariant is preserved across all KH steps; no probabilistic argument is required.

Theorem 3: Consider Problem A with M players and joint action space size $\begin{array} { r } { A _ { \mathrm { j o i n t } } = \prod _ { i = 1 } ^ { M } | \mathcal { A } _ { i } | } \end{array}$ . There exists $c > 0$ such that for any $p \in ( 0 , 1 )$ , with $\bar { b _ { t } } = c \sqrt { H ^ { 3 } \iota / t }$ and $\alpha _ { t } = ( H +$ $1 ) / ( H + t )$ , with probability at least $1 - p$ the total regret of mQ-learning for any player is $O ( \sqrt { H ^ { 4 } S A _ { \mathrm { j o i n t } } T \iota } )$ , where $\iota = \log ( S A _ { \mathrm { j o i n t } } T / p )$

Remark 4: The bound matches the single-agent rate of [1] against the joint-action benchmark: common rewards plus deterministic tie-breaking are enough to recover the centralized rate without communication. We do not claim asymmetry is free in an absolute sense, since $A _ { \mathrm { j o i n t } }$ itself grows exponentially in M. In a distributed system (e.g. M transmitters selecting channels, or M robots picking sub-tasks), the per-episode suboptimality decays as $\tilde { O } ( H ^ { 2 } \sqrt { S A _ { \mathrm { j o i n t } } / K } )$ , the same scaling a single centralized controller would achieve.

Algorithm 2 mQ-learning-intervals (for each player   
P<sub>i</sub>)   
1: Initialize $Q _ { h } ^ { k , \mathrm { u p } } ( i , x , \pmb { a } )  H , Q _ { h } ^ { k , \mathrm { l o w } } ( i , x , \pmb { a } )  0 ,$   
$N _ { h } ( i , x , \pmb { a } ) \gets 0$ for all $( x , \pmb { a } , h ) \in \mathcal { S } \times \mathcal { A } \times [ H ]$   
2: for episode $k = 1 , \ldots , K$ do   
3: Receive initial state $x _ { 1 } .$   
4: for step $h = 1 , \ldots , H$ do   
5: In state $x _ { h } ,$ consider the action in the desired set   
pulled the fewest times. Break ties via Definition 2.   
Call this a.   
6: I $\exists { \mathbf { } } a ^ { \prime }$ such that for some player i: $Q _ { h } ^ { k , \operatorname* { u p } } ( i , \pmb { a } , x _ { h } ^ { k } ) <$   
$Q _ { h } ^ { k , \mathrm { l o w } } ( i , { \pmb a } ^ { \prime } , x _ { h } ^ { k } )$ , then player i intentionally deviates   
from a.   
7: Call the realized joint action $\mathbf { \delta } _  \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathrm { \~ \textit ~ { ~ a ~ } ~ } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \beta } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathbf { \alpha } \mathbf { \beta } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm \mathrm { ~ } \mathrm { ~ } \mathrm \mathrm { ~ } \mathrm \mathrm { ~ } \mathrm \mathrm { ~ ~ } \mathrm \mathrm { ~ } \mathrm \mathrm { ~ } \mathrm \mathrm { ~ ~ } \mathrm \mathrm \mathrm { ~ } \mathrm \mathrm \mathrm { ~ ~ } \mathrm \mathrm \mathrm { ~ ~ } \mathrm \mathrm \mathrm \mathrm { ~ ~ } \mathrm \mathrm \mathrm \mathrm  ~ ~ \mathrm \mathrm \mathrm ~ \mathrm ~ \mathrm \mathrm \mathrm { ~ ~ } \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm { ~ ~ ~ ~ } \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm  ~ ~ ~ ~ \mathrm ~ \mathrm \mathrm \mathrm ~ \mathrm ~ \mathrm \mathrm ~ \mathrm \mathrm \mathrm \mathrm ~ \mathrm \mathrm ~ \mathrm \mathrm \mathrm \mathrm \mathrm ~ \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm $ . All players observe   
${ \pmb { a } } _ { h } .$   
8: If ${ \pmb a } _ { h } \neq { \pmb a } ,$ all players eliminate a from the desired   
set for state $x _ { h }$   
9: Each player observes own reward $r _ { h } ^ { i }$ and state $x _ { h + 1 } .$   
10: $t ~ = ~ N _ { h } ( i , x _ { h } , { \bf a } _ { h } ) ~  ~ N _ { h } ( i , x _ { h } , { \bf a } _ { h } ) + 1 ; ~ b _ { t } ~  ~$   
$c \sqrt { H ^ { 3 } \iota / t } .$   
11: $Q _ { h } ^ { \dot { k } , \mathfrak { u p } } ( i , x _ { h } , \pmb { a } _ { h } ) \gets ( 1 - \alpha _ { t } ) Q _ { h } ^ { k , \mathfrak { u p } } ( i , x _ { h } , \pmb { a } _ { h } )$   
$+ \alpha _ { t } [ r _ { h } ^ { i } + V _ { h + 1 } ( x _ { h + 1 } ) + b _ { t } ] .$   
12: $Q _ { h } ^ { k , \mathrm { l o w } } ( i , x _ { h } , \pmb { a } _ { h } ) \gets ( 1 - \alpha _ { t } ) \dot { Q } _ { h } ^ { k , \mathrm { l o w } } ( i , x _ { h } , \pmb { a } _ { h } )$   
$+ \alpha _ { t } [ r _ { h } ^ { i } + V _ { h + 1 } ( x _ { h + 1 } ) - b _ { t } ] .$   
13: $V _ { h } ( x _ { h } ) \stackrel { \ldots } {  } \operatorname* { m i n } \{ H , \operatorname* { m a x } _ { \pmb { a } ^ { \prime } } Q _ { h } ^ { k } ( i , x _ { h } , \pmb { a } ^ { \prime } ) \} .$   
14: end for   
15: end for

## B. Problem B: Asymmetry in Rewards

When players receive independent rewards, they cannot maintain identical Q-value estimates. Even though they observe the same actions, their different reward realizations lead to different confidence intervals for each state-action pair. We address this by maintaining both upper and lower confidence bounds, inspired by the approach of [8] in the bandit setting.

The key idea is that players maintain a “desired set” of plausible optimal joint actions at each state. An action is eliminated from the desired set when some player’s confidence interval shows it is definitively suboptimal. Since players can observe each other’s actions, a unilateral deviation from the agreedupon action serves as an implicit communication signal: it tells all players that the deviating player has determined the proposed action is suboptimal.

The learning rate is again $\alpha _ { t } = ( H + 1 ) / ( H + t )$ . The width of the confidence interval $Q ^ { \mathrm { u p } } - Q ^ { \mathrm { l o w } }$ controls the regret incurred when a suboptimal action from the desired set is played; Lemma 13 characterizes how it shrinks with visits.

a) Confidence stability across players.: Although each player’s reward stream is independent, all players use the same bonus schedule $b _ { t }$ and the same visit counts $N _ { h } ( \cdot , \cdot )$ (since actions are observable). By Lemma 13, the interval width at $( x , a )$ after t visits is exactly $2 \sum _ { i } \alpha _ { t } ^ { i } b _ { i }$ , a deterministic function of t identical across players. The intervals are merely shifted between players by independent reward noise. Optimism (Lemma 12, union-bounded over M players, contributing the $( 1 - p ) ^ { M }$ factor) guarantees each interval covers $Q _ { h } ^ { * } ( x , \pmb { a } )$ w.h.p., so player i’s deviation truthfully signals dominance: the desired set shrinks monotonically and never excludes the optimal action.

Theorem 5: Consider Problem B (asymmetry in rewards) with M players and joint action space size $\begin{array} { r } { A _ { \mathrm { j o i n t } } = \prod _ { i = 1 } ^ { M } | A _ { i } | . } \end{array}$ There exists a constant $c > 0$ such that for any $p \in ( 0 , 1 )$ , with $b _ { t } = c \sqrt { H ^ { 3 } \iota / t }$ , with probability at least $1 - p$ the total regret of mQ-learning-intervals for any player is at most $O ( \sqrt { H ^ { 4 } S A _ { \mathrm { j o i n t } } T \iota } )$ , where $\iota = \log ( M S A _ { \mathrm { j o i n t } } T / p )$

Remark 6: The regret analysis differs from Problem A because there is an additional “slack” term arising from the fact that the action played may not be the greedy action according to the player’s own Q-values. This term is bounded using the interval width from Lemma 13, which decays as $O ( \sqrt { H ^ { 3 } \iota / t } )$ with the number of visits. The same caveat applies as for Problem A: $\begin{array} { r } { A _ { \mathrm { j o i n t } } = \prod _ { i = 1 } ^ { M } | \mathcal { A } _ { i } | } \end{array}$ grows exponentially in M, so the bound is most informative for small numbers of players.

## IV. PROBLEM C: FULL INFORMATION ASYMMETRY

When both action and reward information are asymmetric, the challenges of Problems A and B combine. Players cannot observe each other’s actions (so they cannot use deviations as signals) and receive different rewards (so they cannot maintain identical estimates). We adopt a two-phase explorethen-commit strategy.

## A. Algorithm: mEXC

During the exploration phase (episodes $1 , \ldots , K ^ { \prime } )$ , players follow a deterministic exploration protocol: at each state, they play the least-visited joint action (with lexicographic tiebreaking). Since this rule depends only on the visit counts— which are identical across players because they visit the same state-action pairs—all players agree on the exploration action without communication.

During the commit phase (episodes $K ^ { \prime } + 1 , \ldots , K )$ , each player acts greedily with respect to their learned Q-values, using lexicographic tie-breaking to ensure agreement.

The exploration phase incurs regret at most $R _ { \mathrm { e x p l o r e } } \leq K ^ { \prime } H$ since rewards are bounded in [0, 1]. The commit phase regret depends on the quality of the Q-value estimates after exploration.

## B. Algorithm: mEXC-Bellman

An alternative approach uses the empirical Bellman equation rather than the incremental Q-learning update during the exploration phase.

The mEXC-Bellman variant uses the plug-in empirical Bellman equation with $\hat { \mathbb { P } } _ { h } ( x ^ { \prime } \mid x , \pmb { a } ) = N _ { h } ( x , \pmb { a } , x ^ { \prime } ) / N _ { h } ( x , \pmb { a } )$ (the standard model-based update; cf. [1]). It can yield tighter estimates when exploration is long, at the cost of storing transition counts.

Algorithm 3 mEXC (for each player $P _ { i } )$   
1: Initialize $Q _ { h } ^ { k } ( i , x , \pmb { a } )  H , \ N _ { h } ( i , x , \pmb { a } )  0$ for all   
$( x , \mathbf { a } , h ) \in \ddot { S } \times \mathcal { A } \times [ H ] .$   
2: for episode $k = 1 , \ldots , K ^ { \prime }$ (Exploration Phase) do   
3: Receive $x _ { 1 } .$   
4: for step $h = 1 , \ldots , H$ do   
5: Pick the least-visited action in the desired set; break   
ties via Definition 2. Call this $\mathbf { \delta } _  \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathrm { \~ \textit ~ { ~ a ~ } ~ } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \beta } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathbf { ~ } { ~ } \mathbf { \alpha } \mathbf { \beta } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm \mathrm { ~ } \mathrm { ~ } \mathrm \mathrm { ~ } \mathrm { ~ } \mathrm \mathrm { ~ ~ } \mathrm \mathrm { ~ } \mathrm \mathrm { ~ } \mathrm \mathrm { ~ ~ } \mathrm \mathrm { ~ } \mathrm \mathrm \mathrm { ~ } \mathrm \mathrm { ~ ~ } \mathrm \mathrm \mathrm { ~ ~ } \mathrm \mathrm \mathrm \mathrm { ~ ~ } \mathrm \mathrm \mathrm \mathrm { ~ ~ } \mathrm \mathrm \mathrm \mathrm \mathrm  ~ ~ \mathrm ~ \mathrm \mathrm ~ \mathrm \mathrm ~ \mathrm \mathrm \mathrm ~ \mathrm ~ \mathrm \mathrm \mathrm ~ \mathrm \mathrm \mathrm ~ \mathrm \mathrm \mathrm ~ \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm $   
6: All players take $\mathbf { \delta } _  \mathbf { \alpha } \mathbf { \delta } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \delta } \mathbf { \alpha } \mathbf { \delta } \mathbf { \alpha } \mathbf { \delta } \mathbf { \alpha } \mathbf { \delta } \mathbf { \alpha } \mathbf { \delta } \mathbf { \alpha } \mathbf { \delta } \mathbf { \alpha } \mathbf { \delta } \mathbf { \alpha } \mathbf { \delta } \mathbf { \alpha } \mathbf { \delta } \mathbf { \alpha } \mathrm { \langle } \mathbf { \delta } \mathbf { \delta } \mathbf { \alpha } \mathbf { \delta } \mathbf { \alpha } \mathbf { \delta } \mathbf { \alpha } \mathrm { \delta } \mathbf { \alpha } \mathbf { \delta } \mathbf { \alpha } \mathrm { \delta } \mathbf { \alpha } \mathrm { \delta } \mathbf { \alpha } \mathrm { \delta } \mathbf { \alpha } \mathrm { \delta } \mathbf { \alpha } \mathrm { \delta } \mathbf { \alpha } \mathrm { \delta } \mathbf { \alpha } \mathrm { \delta } \mathbf { \alpha \delta } \mathrm { \delta \alpha } \mathrm { \delta \delta } \mathbf { \alpha } \mathrm { \delta \delta } \mathrm \mathbf { \alpha } \mathrm { \delta \delta } \mathrm \mathbf { \alpha } \mathrm { \delta \delta } \mathrm \mathrm { \delta \delta } \mathrm \mathbf { \alpha } \delta \delta \delta \mathbf { \delta } \delta \delta \mathbf \delta \mathbf { \alpha } \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta$ , observe reward $r _ { h }$ and state   
$x _ { h + 1 } .$   
7: $t ~ = ~ N _ { h } ( i , x _ { h } , { \bf a } _ { h } ) ~  ~ N _ { h } ( i , x _ { h } , { \bf a } _ { h } ) ~ + ~ 1 ; ~ b _ { t } ~  ~$   
$c \sqrt { H ^ { 3 } \iota / t } .$   
8: $Q _ { h } ^ { \dot { k } } ( i , x _ { h } , \mathbf { 0 } _ { h } ) \gets ( 1 - \alpha _ { t } ) Q _ { h } ^ { k } ( i , x _ { h } , \mathbf { 0 } _ { h } )$   
$+ \alpha _ { t } [ r _ { h } + V _ { h + 1 } ( x _ { h + 1 } ) + b _ { t } ] .$   
9: $V _ { h } ( x _ { h } ) \gets \operatorname* { m i n } \{ H , \operatorname* { m a x } _ { a ^ { \prime } } Q _ { h } ^ { k } ( i , x _ { h } , \pmb { a } ^ { \prime } ) \}$   
10: end for   
11: end for   
12: for episode $k = K ^ { \prime } + 1 , \ldots , K$ (Commit Phase) do   
13: Receive $x _ { 1 } .$   
14: for step $h = 1 , \ldots , H$ do   
15: Pick $\mathbf { \delta } _  \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathrm { \bf ~ \textit ~ { ~ a ~ } } _ { h }$ as the smallest action in   
arg max<sub>a</sub>′ $Q _ { h } ^ { K ^ { \prime } } ( i , x _ { h } , { \bf { a } } ^ { \prime } )$   
16: end for   
17: end for

a) Exploration–commit transition.: Both algorithms use a hard switch at the predetermined episode $K ^ { \prime } .$ , which depends only on K, $S , \ A _ { \mathrm { j o i n t } }$ , H (all common knowledge), so the switch is communication-free. Visit counts are common to all players (since exploration is joint), so all players commit simultaneously. A hard switch is used rather than ϵ-greedy because the latter would require shared randomness when actions are unobserved.

Theorem 7: Consider Problem C (full information asymmetry) with M players and joint action space size $\begin{array} { r } { A _ { \mathrm { j o i n t } } = \prod _ { i = 1 } ^ { M } | A _ { i } | } \end{array}$ . Choose the exploration horizon $K ^ { \prime } =$ $\lceil \bar { ( } S A _ { \mathrm { j o i n t } } ) ^ { \bar { 1 } / \bar { 3 } } \bar { K ^ { 2 } } ^ { / 3 } \rceil$ . There exists a constant $c \ > \ 0$ such that for any $p \in \mathsf { \Gamma } ( 0 , 1 )$ , with $b _ { t } ~ = ~ c \sqrt { H ^ { 3 } \iota / t }$ and $\iota =$ $\log ( M S A _ { \mathrm { j o i n t } } T / p )$ , with probability at least $1 - p$ the total regret of mEXC (and of mEXC-Bellman) for any player is at most

$$
O \Big ( H ( S A _ { \mathrm { j o i n t } } ) ^ { 1 / 3 } T ^ { 2 / 3 } \iota ^ { 1 / 3 } \Big ) .
$$

Proof: (Sketch.) Total regret decomposes as $R _ { T } = R _ { \mathrm { e x p l o r e } } +$ $R _ { \mathrm { c o m m i t } } .$ . The exploration phase plays an arbitrary policy for $K ^ { \prime }$ episodes, so $R _ { \mathrm { e x p l o r e } } ~ \leq ~ K ^ { \prime } H$ . After $K ^ { \prime }$ episodes of round-robin exploration, every $( x , a )$ has been visited at least $\lfloor K ^ { \prime } / ( S A _ { \mathrm { j o i n t } } ) \rfloor$ times in expectation, so by Lemma 12 applied to each player’s Q-table together with a union bound over the M players, with probability $\geq 1 - p$ the post-exploration estimates satisfy

$$
| V _ { 1 } ^ { K ^ { \prime } } ( x _ { 1 } ) - V _ { 1 } ^ { * } ( x _ { 1 } ) | \le O \Bigl ( \sqrt { H ^ { 4 } S A _ { \mathrm { j o i n t } } \iota / K ^ { \prime } } \Bigr )
$$

Algorithm 4 mEXC-Bellman (for each player $P _ { i } )$   
1: Initialize $Q _ { h } ^ { k } ( i , x , \pmb { a } )  H , \ N _ { h } ( i , x , \pmb { a } )  0$ for all   
$( x , \mathbf { a } , h ) \in \ddot { S } \times \mathcal { A } \times [ H ] .$   
2: for episode $k = 1 , \ldots , K ^ { \prime }$ (Exploration Phase) do   
3: Receive $x _ { 1 } .$   
4: for step $h = 1 , \ldots , H$ do   
5: Pick the least-visited action; break ties via Defini  
tion 2. Call this $\mathbf { \delta } _  \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathrm { \~ \textit ~ { ~ a ~ h ~ } ~ }$   
6: All players take $\mathbf { \delta } _  \mathbf { \alpha } \mathbf { \delta } \mathbf { \alpha } \mathbf { \delta } \mathbf { \alpha } \mathbf { \delta } \mathbf { \alpha } \mathbf { \delta } \mathbf { \alpha } \mathbf { \delta } \mathbf { \alpha } \mathbf { \delta } \mathbf { \alpha } \mathbf { \delta } \mathbf { \alpha } \mathbf { \delta } \mathbf { \alpha } \mathbf { \delta } \mathbf { \alpha } \mathbf { \delta } \mathbf { \alpha } \mathrm { \langle } \mathbf { \delta } \mathbf { \delta } \mathbf { \alpha } \mathrm { \delta } \mathbf { \alpha } \mathbf { \delta } \mathbf { \alpha } \mathrm { \delta } \mathbf { \alpha } \mathrm { \langle } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta \alpha } \mathrm { \delta } \mathbf { \delta \alpha } \mathrm { \delta } \mathbf { \delta \alpha } \mathrm { \delta } \mathbf { \delta \alpha } \mathrm { \delta \delta } \mathbf { \alpha \delta } \mathrm { \delta \delta } \mathbf { \alpha \delta }$ , observe reward $r _ { h }$ and state   
$x _ { h + 1 } .$   
7: $t = N _ { h } ( i , x _ { h } , { \bf { a } } _ { h } )  N _ { h } ( i , x _ { h } , { \bf { a } } _ { h } ) + 1 .$   
8: Update empirical reward $\hat { r } ( x _ { h } , { \pmb a } _ { h } )$ and empirical   
transition $\hat { \mathbb { P } } _ { h } ( \cdot \mid x _ { h } , { \pmb a } _ { h } )$ using the latest sample.   
9: $\begin{array} { r l r } { Q _ { h } ^ { k } ( i , x _ { h } , { \bf a } _ { h } ) } & { { }  } & { \hat { r } ( x _ { h } , { \bf a } _ { h } ) + \sum _ { x ^ { \prime } \in S } \hat { \mathbb { P } } _ { h } ( x ^ { \prime } } \end{array}$   
$x _ { h } , { \pmb a } _ { h } ) V _ { h + 1 } ( { \pmb x } ^ { \prime } ) .$   
10: $V _ { h } ( x _ { h } ) \gets \operatorname* { m i n } \{ H , \operatorname* { m a x } _ { \pmb { a } ^ { \prime } } Q _ { h } ^ { k } ( i , x _ { h } , \pmb { a } ^ { \prime } ) \}$   
11: end for   
12: end for   
13: for episode $k = K ^ { \prime } + 1 , \ldots , K$ (Commit Phase) do   
14: Receive $x _ { 1 } .$   
15: for step $h = 1 , \ldots , H$ do   
16: Pick a<sub>h</sub> as the smallest action in   
arg max<sub>a</sub> $Q _ { h } ^ { K ^ { \prime } } ( i , x _ { h } , { \bf { a } } ^ { \prime } )$   
17: end for   
18: end for

for every player. Since each player commits to the (lexicographic-smallest) greedy policy under their own table, and on the high-probability event all M tables agree on the same greedy joint action (the optimism plus deterministic tiebreaking argument from Theorem 3 extends to Q-tables that are uniformly close to $Q ^ { * } )$ , the commit phase incurs perepisode suboptimality bounded by the above value-function error. Summing,

$$
R _ { \mathrm { c o m m i t } } \leq ( K - K ^ { \prime } ) \cdot O \Big ( \sqrt { H ^ { 4 } S A _ { \mathrm { j o i n t } } \iota / K ^ { \prime } } \Big ) .
$$

Balancing $\begin{array} { r l r } { R _ { \mathrm { e x p l o r e } } } & { { } \ = \ } & { K ^ { \prime } H } \end{array}$ and $\begin{array} { r l } { R _ { \mathrm { c o m m i t } } \quad } & { { } = } \end{array}$ $O ( K \sqrt { H ^ { 4 } S A _ { \mathrm { j o i n t } } \iota / K ^ { \prime } } )$ by setting $K ^ { \prime } ~ \asymp ~ ( S A _ { \mathrm { i o i n t } } ) ^ { 1 / 3 } K ^ { 2 / 3 }$ yields the claimed $\tilde { O } ( H ( S A _ { \mathrm { j o i n t } } ) ^ { 1 / 3 } T ^ { 2 / 3 } )$ regret. The argument for mEXC-Bellman is identical, using the standard model-based plug-in error bound $\| \hat { \mathbb { P } } - \mathbb { P } \| _ { 1 } \le O ( \sqrt { S \iota / N } )$ in place of the model-free concentration step. □

Remark 8: The $T ^ { 2 / 3 }$ rate is worse than the $\sqrt { T }$ rate achieved for Problems A and B, which is the standard penalty paid by explore-then-commit when the suboptimality gap is unknown. Whether the optimal rate for Problem C is $\dot { \sqrt { T } }$ or $T ^ { 2 / 3 }$ is left open.

Remark 9: In both mEXC variants, implicit coordination during the exploration phase is ensured because the leastvisited-action rule depends only on visit counts, which are common knowledge since all players visit the same stateaction pairs. During the commit phase, coordination relies on lexicographic tie-breaking applied to Q-values that, while not identical across players, are sufficiently concentrated around the true values after enough exploration.

## V. AUXILIARY RESULTS AND PROOF SKETCHES

We present the key technical lemmas that underlie our main results. Full proofs are deferred to the Appendix.

## A. Learning Rate Properties

For the learning rate $\alpha _ { t } = ( H + 1 ) / ( H + t )$ , we define the weights:

$$
\alpha _ { t } ^ { 0 } = \prod _ { j = 1 } ^ { t } ( 1 - \alpha _ { j } ) , \quad \alpha _ { t } ^ { i } = \alpha _ { i } \prod _ { j = i + 1 } ^ { t } ( 1 - \alpha _ { j } ) .\tag{5}
$$

These weights arise naturally when unrolling the recursive Q-learning update. The quantity $\alpha _ { t } ^ { i }$ represents the effective weight of the i-th observation after t total observations.

Lemma 10 (Weight Properties): The following properties hold for the weights defined in (5):

(a) $\begin{array} { r } { \frac { 1 } { \sqrt { t } } \leq \sum _ { i = 1 } ^ { t } \frac { \alpha _ { t } ^ { i } } { \sqrt { i } } \leq \frac { 2 } { \sqrt { t } } . } \end{array}$   
(b) $\begin{array} { r } { \operatorname* { m a x } _ { i \in [ t ] } \alpha _ { t } ^ { i } \leq \frac { 2 H } { t } . } \end{array}$ and $\begin{array} { r } { \sum _ { i = 1 } ^ { t } ( \alpha _ { t } ^ { i } ) ^ { 2 } \le \frac { 2 H } { t } . } \end{array}$   
(c) $\textstyle \sum _ { t = i } ^ { \infty } \alpha _ { t } ^ { i } = 1 + { \frac { 1 } { H } }$

Property (c) is crucial for the regret analysis: it controls how much a single observation’s “influence” accumulates across all future episodes where the same state-action pair is visited.

## B. Recursion and Optimism

Lemma 11 (Recursion): For any $( x , \pmb { a } , h ) \in \mathcal { S } \times \mathcal { A } \times [ H ]$ and episode $k \in [ K ]$ , let $t = N _ { h } ^ { k } ( x , \pmb { a } )$ and suppose $( x , a )$ was taken at step h of episodes $k _ { 1 } , \ldots , k _ { t } < k$ . Then for any player m:

$$
\begin{array} { l } { { ( Q _ { h } ^ { k } - Q _ { h } ^ { * } ) ( x , { \pmb a } , m ) = \alpha _ { t } ^ { 0 } ( H - Q _ { h } ^ { * } ( x , { \pmb a } , m ) ) } } \\ { { \displaystyle ~ + \sum _ { i = 1 } ^ { t } \alpha _ { t } ^ { i } \Big [ ( V _ { h + 1 } ^ { k _ { i } } - V _ { h + 1 } ^ { * } ) ( x _ { h + 1 } ^ { k _ { i } } , m ) } } \\ { { \displaystyle ~ + [ ( \hat { P } _ { h } ^ { k _ { i } } - P _ { h } ) V _ { h + 1 } ^ { * } ] ( x , { \pmb a } , m ) + b _ { i } \Big ] . } } \end{array}\tag{6}
$$

This recursion decomposes the estimation error $Q _ { h } ^ { k } - Q _ { h } ^ { * }$ into three components: (i) the initialization bias (decaying via $\alpha _ { t } ^ { 0 } )$ , (ii) the propagated estimation error from future steps, and (iii) the transition estimation error plus the exploration bonus. Lemma 12 (Optimism): There exists $c > 0$ such that for any $p \in ( 0 , 1 )$ , letting $b _ { t } = c \sqrt { H ^ { 3 } \iota / t }$ with $\iota = \log ( S A T / p )$ , with probability $\geq 1 - p .$ , for all $( x , a , h , k )$

$$
0 \leq ( Q _ { h } ^ { k } - Q _ { h } ^ { * } ) ( x , \pmb { a } , m ) \leq \alpha _ { t } ^ { 0 } H + \sum _ { i = 1 } ^ { t } \alpha _ { t } ^ { i } \phi _ { h + 1 } ^ { k _ { i } } + \beta _ { t }\tag{7}
$$

where $\underline { { \phi _ { h } ^ { k } } } \ : = \ : ( V _ { h } ^ { k } \ : - \ : V _ { h } ^ { * } ) ( x _ { h } ^ { k } )$ and $\begin{array} { r } { \beta _ { t } ~ = ~ 2 \sum _ { i = 1 } ^ { t } \alpha _ { t } ^ { i } b _ { i } ~ \le ~ } \end{array}$ $4 c \sqrt { H ^ { 3 } \iota / t } .$

The lower bound $Q _ { h } ^ { k } \ge Q _ { h } ^ { * }$ (optimism) is essential: it ensures that the algorithm does not prematurely discard optimal actions. The upper bound controls the overestimation error that drives the regret.

## C. Interval Width for Problem B

Lemma 13 (Interval Width): After state-action pair $( x , a )$ has been visited t times:

$$
Q ^ { \mathsf { u p } } ( x , \pmb { a } ; t ) - Q ^ { \mathrm { l o w } } ( x , \pmb { a } ; t ) = 2 \sum _ { i = 1 } ^ { t } \alpha _ { t } ^ { i } b _ { i } .\tag{8}
$$

Proof: For the base case, $Q ^ { \mathsf { u p } } ( x , \pmb { a } ; 1 ) - Q ^ { \mathrm { l o w } } ( x , \pmb { a } ; 1 ) = 2 \alpha _ { 1 } b _ { 1 }$ which satisfies the lemma. For the inductive step, let $n =$ $n ( x , \pmb { a } )$ :

$$
\begin{array} { l l } { { \displaystyle Q ^ { \mathsf { u p } } ( x , a ; n + 1 ) - Q ^ { \mathsf { l o w } } ( x , a ; n + 1 ) } } \\ { { \displaystyle ~ = ( 1 - \alpha _ { n + 1 } ) \big ( Q ^ { \mathsf { u p } } ( x , a ; n ) - Q ^ { \mathsf { l o w } } ( x , a ; n ) \big ) + 2 \alpha _ { n + 1 } b _ { n + 1 } } } \\ { { \displaystyle ~ = ( 1 - \alpha _ { n + 1 } ) \cdot 2 \sum _ { i = 1 } ^ { n } \alpha _ { n } ^ { i } b _ { i } + 2 \alpha _ { n + 1 } b _ { n + 1 } } } \\ { { \displaystyle ~ = 2 \sum _ { i = 1 } ^ { n + 1 } \alpha _ { n + 1 } ^ { i } b _ { i } , } } & { { ~ \odot } } \end{array}
$$

where the last step uses $\begin{array} { r } { \alpha _ { n + 1 } ^ { i } = \alpha _ { i } \prod _ { j = i + 1 } ^ { n + 1 } ( 1 - \alpha _ { j } ) } \end{array}$ , completing the induction. □

Using Lemma 10(a), the interval width satisfies $Q ^ { \mathrm { u p } } \mathrm { ~ - ~ }$ $Q ^ { \mathrm { l o w } } \leq O ( \sqrt { H ^ { 3 } \iota / t } )$ , which decays at the same rate as the bonus $b _ { t }$

## D. Concentration via Azuma-Hoeffding

The proofs also rely on the Azuma-Hoeffding inequality to control the martingale difference terms that arise from transition estimation errors. Define the martingale difference $\xi _ { h + 1 } ^ { k } : = [ ( \mathbb { P } _ { h } - \hat { \mathbb { P } } _ { h } ^ { k } ) ( V _ { h + 1 } ^ { * } - V _ { h + 1 } ^ { k } ) ] ( x _ { h } ^ { k } , a _ { h } ^ { k } )$ ). Since $| [ \hat { \mathbb { P } } _ { h } ^ { k _ { i } } -$ $\mathbb { P } _ { h } \big ) V _ { h + 1 } ^ { * } \big ] ( x , a ) \vert \le H$ , Azuma-Hoeffding gives:

$$
\operatorname* { P r } \left( \left| \sum _ { h = 1 } ^ { H } \sum _ { k = 1 } ^ { K } \xi _ { h + 1 } ^ { k } \right| \geq \epsilon \right) \leq 2 \exp \left( \frac { - \epsilon ^ { 2 } } { 2 K H ^ { 3 } } \right) .\tag{10}
$$

Setting the right-hand side equal to $p$ and solving for ϵ yields the claimed concentration bound.

## E. Proof Sketch for Theorem 3

Define $\delta _ { h } ^ { k } : = ( V _ { h } ^ { k } - V _ { h } ^ { \pi _ { k } } ) ( x _ { h } ^ { k } )$ and $\phi _ { h } ^ { k } : = ( V _ { h } ^ { k } - V _ { h } ^ { * } ) ( x _ { h } ^ { k } )$ By optimism (Lemma 12), $\begin{array} { r } { R _ { T } \leq \sum _ { k = 1 } ^ { K } \delta _ { 1 } ^ { k } } \end{array}$ . Decomposing $\delta _ { h } ^ { k }$ via the Bellman equation and Lemma 12 yields a recursive relation:

$$
\sum _ { k = 1 } ^ { K } \delta _ { h } ^ { k } \leq S A H + \bigg ( 1 + \frac { 1 } { H } \bigg ) \sum _ { k = 1 } ^ { K } \delta _ { h + 1 } ^ { k } + \sum _ { k = 1 } ^ { K } ( \beta _ { n _ { h } ^ { k } } + \xi _ { h + 1 } ^ { k } ) .\tag{11}
$$

Recursing over h and bounding the bonus terms via pigeonhole and Azuma-Hoeffding gives the result. See Appendix A for the complete proof.

## VI. COMPARISON OF APPROACHES

The three problem settings share a common skeleton— joint-action Q-learning with deterministic tie-breaking—but differ in how players reach agreement. In Problem A, common rewards and common transitions make all players’ Qtables identical at every step, so lexicographic tie-breaking on the arg max set picks the same joint action without any communication. In Problem B, independent rewards prevent identical Q-tables, but observable actions let each player use a unilateral deviation as an implicit dominance signal: when one player deviates from the candidate action, the others infer that player has found a strictly better alternative and eliminate the candidate from the desired set. In Problem C, neither mechanism is available, so we fall back to explorethen-commit with a deterministic schedule shared a priori.

The regret rates reflect this hierarchy. Problems A and B both achieve $\tilde { O } ( \sqrt { H ^ { 4 } S A _ { \mathrm { j o i n t } } T } )$ (Theorems 3 and 5), matching centralized single-agent learning [1] against the joint-action benchmark. Problem C achieves ${ \tilde { O } } ( H ( { \bar { S } } A _ { \mathrm { j o i n t } } ) ^ { 1 / 3 } { \bar { T } } ^ { 2 / 3 } )$ (Theorem 7), the standard penalty of explore-then-commit when the suboptimality gap is unknown. The $T ^ { 2 / 3 }$ rate is worse than $\sqrt { T }$ but still sublinear; whether it can be improved to $\sqrt { T }$ for Problem C remains open.

A useful way to think about the gap from Problems A/B to Problem C is that observability of either actions or rewards is enough to encode a low-bandwidth signal between players, and that signal suffices to recover the $\checkmark$ rate. Problem C removes both channels, and the only remaining shared information— visit counts of jointly-played actions—is too coarse to drive UCB-style adaptive exploration; hence the retreat to a nonadaptive schedule.

## VII. DISCUSSION

## A. Reduction to Centralized Joint-Action Q-Learning

The relationship between Algorithm 1 and a centralized learner is direct: a single agent running UCB-Q-learning [1] on the joint MDP $( \mathcal { S } , \mathcal { A } _ { 1 } \times \cdot \cdot \cdot \times \mathcal { A } _ { M } , H , \mathbb { P } , r )$ produces a Qtable identical to that of any one player in Algorithm 1, since Problem A players see the same reward, same next state, and same update with the same tie-breaking. The decentralized algorithm inherits the centralized regret bound. Problem B adds a one-bit-per-step deviation signal that carries exactly the information the centralized learner uses internally to eliminate dominated actions.

## B. Dependence on the Number of Players

The bounds depend on M only through $A _ { \mathrm { j o i n t } }$ . There is no separate polynomial factor in M: doubling M while keeping $A _ { \mathrm { j o i n t } }$ fixed—e.g. splitting one player with $| { \mathcal { A } } | \ = \ 4$ into two players with $| { \mathcal { A } } _ { i } | ~ = ~ 2 $ —does not change the regret. From the joint-action viewpoint, the partition into per-player components is irrelevant. The exponential dependence on M through $A _ { \mathrm { j o i n t } }$ is therefore a property of the tabular setting, not of the asymmetry: any worst-case bound against arbitrary joint policies must pay at least $\sqrt { A _ { \mathrm { j o i n t } } }$ <sub>t</sub>. Removing this requires structural assumptions on the joint Q-function (factored MDPs, mean-field couplings, linear function approximation), each a natural next direction.

## C. When the Bounds Are Practical

With $M = 2 , | \mathcal { A } _ { i } | = 3 , S = 1 0 , H = 2 0 .$ , our Problem A bound puts the per-episode suboptimality below 0.1 once $K \gtrsim$ $1 0 ^ { 5 } .$ , comparable to centralized Q-learning on the same joint MDP. With $M = 4$ and $| \mathcal { A } _ { i } | = 3 ( A _ { \mathrm { j o i n t } } = 8 1 )$ , the same threshold needs $K \gtrsim 1 0 ^ { 6 }$ . Beyond that, structural assumptions are needed.

## VIII. RELATED WORK

a) Multi-Player Bandits.: The multi-player MAB problem has been studied under various information structures. The collision model [6] forbids two players from choosing the same arm. In the cooperative setting without collisions, Chang et al. [7] introduced the information-asymmetry framework that we extend, and [8] achieved optimal regret without communication in the bandit case.

b) Episodic MDPs.: For single-agent episodic MDPs, Jin et al. [1] proved that optimistic Q-learning achieves $\tilde { O } ( \sqrt { H ^ { 4 } S A T } )$ regret model-free. Our work extends this to the multi-player setting under information asymmetry.

c) Game-Theoretic Learning.: The literature on uncoupled dynamics [9] studies whether players reach equilibria without knowing each other’s payoffs; [9] shows uncoupled dynamics cannot generally reach Nash equilibrium, suggesting fundamental limits to communication-free learning. Our cooperative setting is distinct from the competitive game-theoretic framework but shares the no-communication constraint.

## IX. CONCLUSION

We studied multi-player decentralized reinforcement learning in episodic MDPs under three information-asymmetry models. For Problems A and B we obtain $\tilde { O } ( \sqrt { H ^ { 4 } S A _ { \mathrm { j o i n t } } T } )$ regret, matching the single-agent rate of [1] against the jointaction benchmark up to logarithmic factors. For Problem C we obtain ${ \tilde { O } } ( H ( S { \dot { A } } _ { \mathrm { j o i n t } } ) ^ { 1 / 3 } T ^ { 2 / 3 } )$ via explore-then-commit. All bounds depend on $\begin{array} { r } { A _ { \mathrm { j o i n t } } ~ = ~ \prod _ { i = 1 } ^ { M } | A _ { i } | } \end{array}$ , which grows exponentially in M: asymmetry imposes no multiplicative penalty relative to a centralized learner over the same joint action space, but the tabular curse of dimensionality remains. Open directions include sharper bounds for Problem C, extensions to function approximation (linear MDPs), matching lower bounds under each asymmetry model, and regret– communication tradeoffs.

## ACKNOWLEDGMENT

The authors thank colleagues at UCLA for helpful discussions.

## APPENDIX

Proof: A direct computation gives $\begin{array} { r } { \alpha _ { t } ^ { i } = \frac { H + 1 } { H + i } \prod _ { j = i + 1 } ^ { t } \frac { j - 1 } { H + j } } \end{array}$ f $i \geq 1$

(a) Standard Beta-function identities (cf. [1, Lemma 4.1]) yield $\textstyle \sum _ { i } \alpha _ { t } ^ { i } = 1$ for $t \geq 1$ , and $\textstyle \sum _ { i } \alpha _ { t } ^ { i } / { \sqrt { i } }$ lies between $1 / \sqrt { t }$ and $2 / { \sqrt { t } } .$

(b) The ratio $\alpha _ { t } ^ { i } / \alpha _ { t } ^ { i + 1 }$ shows the sequence is maximized at $i = t ,$ , giving max<sub>i</sub> $\alpha _ { t } ^ { i } = \alpha _ { t } = ( H + 1 ) / ( H + t ) \le 2 H / t$ Hence $\bar { \sum _ { i } } ( \bar { \alpha _ { t } ^ { i } } ) ^ { 2 } \leq$ max<sub>i</sub> $\begin{array} { r } { \alpha _ { t } ^ { i } \cdot \sum _ { i } \alpha _ { t } ^ { i } \le 2 H / t . } \end{array}$

(c) Telescoping the product representation gives $\begin{array} { r } { \sum _ { t \ge i } \alpha _ { t } ^ { i } = } \\ { \qquad \bigstar \bigstar \bigstar \bigstar \bigstar \bigstar \bigstar \bigstar \bigstar \bigstar } \end{array}$ $( H + 1 ) / H = 1 + 1 / H$

Proof: By induction on $t = N _ { h } ^ { k } ( x , { \pmb a } ) . \mathrm { \ A t \ } t = 0 , Q _ { h } ^ { k } ( x , { \pmb a } ) =$ $H .$ , matching $\alpha _ { 0 } ^ { 0 } ( H - Q _ { h } ^ { * } )$ . For the inductive step, let $k _ { t }$ be the episode of the t-th visit. The update on line 8 of Algorithm 1, minus $Q _ { h } ^ { * } = r _ { h } + \mathbb { P } _ { h } V _ { h + 1 } ^ { * }$ , gives

$$
\begin{array} { r l } & { Q _ { h } ^ { k _ { t } } - Q _ { h } ^ { * } = ( 1 - \alpha _ { t } ) ( Q _ { h } ^ { k _ { t } , \mathrm { p r e } } - Q _ { h } ^ { * } ) } \\ & { \phantom { = } + \alpha _ { t } \bigl [ ( V _ { h + 1 } ^ { k _ { t } } - V _ { h + 1 } ^ { * } ) ( x _ { h + 1 } ^ { k _ { t } } ) } \\ & { \phantom { = } + [ ( \mathbb { P } _ { h } ^ { k _ { t } } - \mathbb { P } _ { h } ) V _ { h + 1 } ^ { * } ] ( x , \pmb { a } ) + b _ { t } \bigr ] , } \end{array}
$$

where $\hat { \mathbb { P } } _ { h } ^ { k _ { t } }$ is the one-sample empirical kernel. Using the identity $( 1 - \alpha _ { t } ) \alpha _ { t - 1 } ^ { i } = \alpha _ { t } ^ { i }$ for $\textit { i } < \textit { t }$ and $\alpha _ { t } ^ { t } \ = \ \alpha _ { t }$ , the inductive hypothesis unrolls to (6). □

Proof: We prove $Q _ { h } ^ { k } \ge Q _ { h } ^ { * }$ by induction backward on h (the case $h = H + 1$ is trivial). By Lemma 11,

$$
Q _ { h } ^ { k } - Q _ { h } ^ { * } = \alpha _ { t } ^ { 0 } ( H - Q _ { h } ^ { * } ) + \sum _ { i = 1 } ^ { t } \alpha _ { t } ^ { i } \big [ ( V _ { h + 1 } ^ { k _ { i } } - V _ { h + 1 } ^ { * } ) + \zeta _ { i } + b _ { i } \big ] ,
$$

where $\zeta _ { i } : = [ ( \hat { \mathbb { P } } _ { h } ^ { k _ { i } } - \mathbb { P } _ { h } ) V _ { h + 1 } ^ { * } ] ( x , \pmb { a } )$ is a bounded martingale difference $( | \zeta _ { i } | \le H )$ . The first two terms are non-negative by induction. By Azuma–Hoeffding and Lemma 10(b), with probability $\ge 1 - p / ( S A H K )$ ,

$$
\Big | \sum _ { i } \alpha _ { t } ^ { i } \zeta _ { i } \Big | \leq H \sqrt { 2 \iota \sum _ { i } ( \alpha _ { t } ^ { i } ) ^ { 2 } } \leq c \sqrt { H ^ { 3 } \iota / t } ,
$$

which is dominated by $\textstyle \sum _ { i } \alpha _ { t } ^ { i } b _ { i }$ for our choice $b _ { i } = c \sqrt { H ^ { 3 } \iota / i }$ Hence $Q _ { h } ^ { k } \ge Q _ { h } ^ { * }$ . Union-bounding over $( x , a , h , k )$ absorbs an $S A H K \ \leq \ S A T$ factor into ι. The upper bound (7) follows from the same decomposition, with the factor 2 in $\begin{array} { r } { \beta _ { t } = 2 \sum _ { i } \alpha _ { t } ^ { i } b _ { i } } \end{array}$ absorbing the (positive) contribution of $\zeta _ { i } ;$ Lemma 10(a) gives $\beta _ { t } \leq 4 c \sqrt { H ^ { 3 } \iota / t }$ □

Proof: Denote $\delta _ { h } ^ { k } : = ( V _ { h } ^ { k } - V _ { h } ^ { \pi _ { k } } ) ( x _ { h } ^ { k } )$ and $\phi _ { h } ^ { k } : = ( V _ { h } ^ { k } -$ $V _ { h } ^ { * } ) ( x _ { h } ^ { k } )$

By Lemma 12, with probability $1 - p , Q _ { h } ^ { k } \ge Q _ { h } ^ { * }$ and thus $V _ { h } ^ { k } \geq V _ { h } ^ { * }$ . The total regret is bounded:

$$
R _ { T } = \sum _ { k = 1 } ^ { K } ( V _ { 1 } ^ { * } - V _ { 1 } ^ { \pi _ { k } } ) ( x _ { 1 } ^ { k } ) \leq \sum _ { k = 1 } ^ { K } \delta _ { 1 } ^ { k } .\tag{12}
$$

For any fixed $( k , h )$ , let $t ~ = ~ N _ { h } ^ { k } ( x _ { h } ^ { k } , a _ { h } ^ { k } )$ , and suppose $( x _ { h } ^ { k } , a _ { h } ^ { k } )$ was taken at step h of episodes $k _ { 1 } , \ldots , k _ { t } < k$ . Then:

(13)

$$
\begin{array} { r l } {  { \delta _ { h } ^ { k } = ( V _ { h } ^ { k } - V _ { h } ^ { \pi _ { k } } ) ( x _ { h } ^ { k } ) \le ( Q _ { h } ^ { k } - Q _ { h } ^ { \pi _ { k } } ) ( x _ { h } ^ { k } , a _ { h } ^ { k } ) } } \\ & { \ = ( Q _ { h } ^ { k } - Q _ { h } ^ { * } ) ( x _ { h } ^ { k } , a _ { h } ^ { k } ) + ( Q _ { h } ^ { * } - Q _ { h } ^ { \pi _ { k } } ) ( x _ { h } ^ { k } , a _ { h } ^ { k } ) } \\ & { \le \alpha _ { t } ^ { 0 } H + \displaystyle \sum _ { i = 1 } ^ { t } \alpha _ { t } ^ { i } \phi _ { h + 1 } ^ { k _ { i } } + \beta _ { t } } \\ & { \quad + [ \mathbb { P } _ { h } ( V _ { h + 1 } ^ { * } - V _ { h + 1 } ^ { \pi _ { k } } ) ] ( x _ { h } ^ { k } , a _ { h } ^ { k } ) } \end{array}\tag{14}
$$

$$
= \alpha _ { t } ^ { 0 } H + \sum _ { i = 1 } ^ { t } \alpha _ { t } ^ { i } \phi _ { h + 1 } ^ { k _ { i } } + \beta _ { t } - \phi _ { h + 1 } ^ { k } + \delta _ { h + 1 } ^ { k } + \xi _ { h + 1 } ^ { k }\tag{15}
$$

where $\textstyle \beta _ { t } = 2 \sum \alpha _ { t } ^ { i } b _ { i } \leq O ( 1 ) \sqrt { H ^ { 3 } \iota / t }$ and $\xi _ { h + 1 } ^ { k } : = [ ( \mathbb { P } _ { h } -$ $\hat { \mathbb { P } } _ { h } ^ { k } \big ) ( V _ { h + 1 } ^ { * } - V _ { h + 1 } ^ { k } ) \big ] ( x _ { h } ^ { k } , a _ { h } ^ { k } )$ is a martingale difference sequence. Inequality (13) holds because $V _ { h } ^ { k } ( x _ { h } ^ { k } ) \le Q _ { h } ^ { k } ( x _ { h } ^ { k } , a _ { h } ^ { k } )$ and inequality (14) holds by Lemma 12 and the Bellman equation (3). Equality (15) follows from $\delta _ { h + 1 } ^ { k } \ : - \ : \phi _ { h + 1 } ^ { k } = $ $( V _ { h + 1 } ^ { * } - V _ { h + 1 } ^ { \pi _ { k } } ) ( x _ { h + 1 } ^ { k } )$

Computing $\textstyle \sum _ { k = 1 } ^ { K } \delta _ { h } ^ { k }$ , the initialization term gives:

$$
\sum _ { k = 1 } ^ { K } \alpha _ { n _ { h } ^ { k } } ^ { 0 } H = \sum _ { k = 1 } ^ { K } H \cdot \mathbb { I } [ n _ { h } ^ { k } = 0 ] \leq S A H .\tag{16}
$$

For the weighted sum term, we regroup: for every $k ^ { \prime } \in [ K ]$ the term $\phi _ { h + 1 } ^ { k ^ { \prime } }$ appears in the summand with $k > k ^ { \prime }$ whenever $( x _ { h } ^ { k } , a _ { h } ^ { k } ) = \bar { ( x _ { h } ^ { k ^ { \prime } } , a _ { h } ^ { k ^ { \prime } } ) }$ . Therefore:

$$
\sum _ { k = 1 } ^ { K } \sum _ { i = 1 } ^ { n _ { h } ^ { k } } \alpha _ { n _ { h } ^ { k } } ^ { i } \phi _ { h + 1 } ^ { k _ { i } } \leq \left( 1 + \frac { 1 } { H } \right) \sum _ { k = 1 } ^ { K } \phi _ { h + 1 } ^ { k }\tag{17}
$$

using $\textstyle \sum _ { t = i } ^ { \infty } \alpha _ { t } ^ { i } = 1 + 1 / H$ from Lemma 10(c). Since $\phi _ { h + 1 } ^ { k } \leq$ $\delta _ { h + 1 } ^ { k }$ (because $V ^ { * } \geq V ^ { \pi _ { k } } )$ :

$$
\begin{array} { r l r } {  { \sum _ { k = 1 } ^ { K } \delta _ { h } ^ { k } \le S A H + ( 1 + \frac { 1 } { H } ) \sum _ { k = 1 } ^ { K } \delta _ { h + 1 } ^ { k } } } \\ & { } & { + \sum _ { k = 1 } ^ { K } ( \beta _ { n _ { h } ^ { k } } + \xi _ { h + 1 } ^ { k } ) . ~ } \end{array}\tag{18}
$$

Recursing for $h = 1 , \ldots , H$ and using $\delta _ { H + 1 } ^ { k } \equiv 0 :$

$$
\sum _ { k = 1 } ^ { K } \delta _ { 1 } ^ { k } \leq O \left( H ^ { 2 } S A + \sum _ { h = 1 } ^ { H } \sum _ { k = 1 } ^ { K } ( \beta _ { n _ { h } ^ { k } } + \xi _ { h + 1 } ^ { k } ) \right) .\tag{19}
$$

By the pigeonhole principle, for any $h \in [ H ]$

$$
\sum _ { k = 1 } ^ { K } \beta _ { n _ { h } ^ { k } } \leq O ( 1 ) \sum _ { x , a } \sum _ { n = 1 } ^ { N _ { h } ^ { K } ( x , a ) } \sqrt { \frac { H ^ { 3 } \iota } { n } } \leq O ( \sqrt { H ^ { 2 } S A T \iota } )\tag{20}
$$

where the last inequality uses $\begin{array} { r } { \sum _ { x , a } N _ { h } ^ { K } ( x , a ) \ = \ K } \end{array}$ and is maximized when $N _ { h } ^ { K } ( x , a ) \ : = \ : \bar { K } / ( S A )$ for all $x , a$ . By Azuma-Hoeffding, with probability $1 - p \colon$

$$
\left| \sum _ { h = 1 } ^ { H } \sum _ { k = 1 } ^ { K } \xi _ { h + 1 } ^ { k } \right| \leq c H \sqrt { T \iota } .\tag{21}
$$

Combining, $\begin{array} { r } { \sum _ { k = 1 } ^ { K } \delta _ { 1 } ^ { k } \le O ( H ^ { 2 } S A + \sqrt { H ^ { 4 } S A T \iota } ) } \end{array}$ . When $T \geq H ^ { 2 } S A$ , the second term dominates. When $T < H ^ { 2 } S A$ the trivial bound $\begin{array} { r } { \sum _ { k = 1 } ^ { K } \delta _ { 1 } ^ { k } \le H K = T \le \sqrt { H ^ { 4 } S A T \iota } } \end{array}$ applies. Therefore $R _ { T } \leq O ( { \sqrt { H ^ { 4 } S A T \iota } } )$ with probability $\geq 1 - 2 p .$ Rescaling p to $p / 2$ completes the proof. □

Proof: By Lemma 12 applied to each player, with probability $( 1 - \dot { p } ) ^ { \dot { M } } , Q _ { h } ^ { k } \geq Q _ { h } ^ { * }$ for all players. Since regret is evaluated with respect to expectations and all players receive i.i.d. rewards, we omit the player index for the remainder.

Define $\delta _ { h } ^ { k }$ and $\phi _ { h } ^ { k }$ as in Appendix A. The decomposition for $\delta _ { h } ^ { k }$ is:

(22)

$$
\leq \operatorname* { m a x } _ { \pmb { a } } Q _ { h } ^ { k } ( x _ { h } ^ { k } , \pmb { a } ) - Q _ { h } ^ { \pi _ { k } } ( x _ { h } ^ { k } , \pmb { a } _ { h } ^ { k } )\tag{23}
$$

$$
\begin{array} { l } { { = \underbrace { \operatorname* { m a x } Q _ { h } ^ { k } ( x _ { h } ^ { k } , \pmb { a } ) - Q _ { h } ^ { k } ( x _ { h } ^ { k } , \pmb { a } _ { h } ^ { k } ) } _ { A } } } \\ { { + \ ( Q _ { h } ^ { k } - Q _ { h } ^ { * } ) ( x _ { h } ^ { k } , \pmb { a } _ { h } ^ { k } ) + ( Q _ { h } ^ { * } - Q _ { h } ^ { \pi _ { k } } ) ( x _ { h } ^ { k } , \pmb { a } _ { h } ^ { k } ) . } } \end{array}\tag{24}
$$

Term A accounts for the fact that the action $\mathbf { \Delta } \mathbf { a } _ { h } ^ { k }$ played (from the desired set) may not be the greedy action. Since actions in the desired set have upper Q-values within the interval width of the best action, by Lemma 13:

$$
A \leq Q ^ { \mathsf { u p } } ( x _ { h } ^ { k } , \pmb { a } _ { h } ^ { k } ; t ) - Q ^ { \mathrm { l o w } } ( x _ { h } ^ { k } , \pmb { a } _ { h } ^ { k } ; t ) = 2 \sum _ { i = 1 } ^ { t } \alpha _ { t } ^ { i } b _ { i } .\tag{25}
$$

Applying Lemma 12 and the Bellman equation for the remaining terms:

$$
\begin{array} { c } { { \displaystyle { \delta _ { h } ^ { k } \leq 2 \sum _ { i = 1 } ^ { t } \alpha _ { t } ^ { i } b _ { i } + \alpha _ { t } ^ { 0 } H + \sum _ { i = 1 } ^ { t } \alpha _ { t } ^ { i } \phi _ { h + 1 } ^ { k _ { i } } + \beta _ { t } } } } \\ { { - \phi _ { h + 1 } ^ { k } + \delta _ { h + 1 } ^ { k } + \xi _ { h + 1 } ^ { k } . } } \end{array}\tag{26}
$$

Therefore, the recursive bound becomes:

$$
\begin{array} { r l r } {  { \sum _ { k = 1 } ^ { K } \delta _ { h } ^ { k } \le \sum _ { k = 1 } ^ { K } 2 \sum _ { i = 1 } ^ { t } \alpha _ { t } ^ { i } b _ { i } + S A H } } \\ & { } & { \quad + ( 1 + \frac { 1 } { H } ) \sum _ { k = 1 } ^ { K } \delta _ { h + 1 } ^ { k } + \sum _ { k = 1 } ^ { K } ( \beta _ { n _ { h } ^ { k } } + \xi _ { h + 1 } ^ { k } ) . } \end{array}\tag{27}
$$

This is identical to the bound in Theorem 3 with an additional $\textstyle \sum _ { k = 1 } ^ { K } 2 \sum _ { i = 1 } ^ { t } \alpha _ { t } ^ { i } b _ { i }$ term. By Lemma 10(a), $\begin{array} { r } { 2 \sum _ { i = 1 } ^ { t } \alpha _ { t } ^ { i } b _ { i } \le \overline { { 2 c } } \sqrt { \dot { H } ^ { 3 } \iota } \cdot \frac { 2 } { \sqrt { \iota } } = O ( \sqrt { H ^ { 3 } \iota / t } ) } \end{array}$ , which is of the same order as $\beta _ { t }$ . Following the same recursion and pigeonhole arguments as in Appendix A, we obtain the stated regret bound of $O ( \sqrt { H ^ { 4 } S A T \iota } )$ □

## REFERENCES

[1] C. Jin, Z. Allen-Zhu, S. Bubeck, and M. I. Jordan, “Is q-learning provably efficient?,” Advances in neural information processing systems, vol. 31, 2018.

[2] A. Anandkumar, N. Michael, A. K. Tang, and A. Swami, “Distributed algorithms for learning and cognitive medium access with logarithmic regret,” IEEE Journal on Selected Areas in Communications, vol. 29, no. 4, pp. 731–745, 2011.

[3] R. S. Sutton and A. G. Barto, Reinforcement learning: An introduction. MIT press, 2018.

[4] T. L. Lai and H. Robbins, “Asymptotically efficient adaptive allocation rules,” Advances in applied mathematics, vol. 6, no. 1, pp. 4–22, 1985.

[5] P. Auer, N. Cesa-Bianchi, and P. Fischer, “Finite-time analysis of the multiarmed bandit problem,” Machine learning, vol. 47, no. 2, pp. 235– 256, 2002.

[6] I. Bistritz and A. Leshem, “Distributed multi-player bandits-a game of thrones approach,” Advances in Neural Information Processing Systems, vol. 31, 2018.

[7] W. Chang, M. Jafarnia-Jahromi, and R. Jain, “Online learning for cooperative multi-player multi-armed bandits,” arXiv preprint arXiv:2109.03818, 2021.

[8] W. Chang and Y. Lu, “Optimal cooperative multiplayer learning bandits with noisy rewards and no communication,” arXiv preprint arXiv:2311.06210, 2023.

[9] S. Hart and A. Mas-Colell, “Uncoupled dynamics do not lead to nash equilibrium,” American Economic Review, vol. 93, no. 5, pp. 1830–1836, 2003.