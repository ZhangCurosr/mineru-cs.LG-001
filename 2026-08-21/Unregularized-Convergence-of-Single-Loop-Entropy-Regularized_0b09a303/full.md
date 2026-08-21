# Unregularized Convergence of Single-Loop, Entropy-Regularized Natural Actor-Critic

Zhiqiang Tan<sup>1</sup>

August 21, 2026

Abstract. While entropy regularization is widely used to stabilize and accelerate Natural Policy Gradient methods, its ability to yield faster convergence rates for the unregularized objective remains underexplored. Existing analyses often rely on double-loop architectures and invoke a linear entropy penalty. To bridge the gap between theory and practice, we analyze a single-loop, entropy-regularized Natural Actor-Critic algorithm under compatible linear function approximation. By training an uncentered critic, our critic tracking can remain stable even as the training policy approaches determinism and the Fisher information matrix degenerates. We focus on two primary regimes for the optimization landscape: a Stochastic Regime, where we fuse coupled actorcritic updates into a joint Lyapunov recurrence, and a Deterministic Regime, where we pivot to a Policy Mirror Descent framework to circumvent the collapse of Euclidean geometry. By exploiting a positive Minimal Action Gap in the unregularized Markov decision process, we introduce an Exponential Translation mechanism that maps the regularized gap to the unregularized one up to an exponentially decaying tail. By tuning the fixed temperature, our algorithm achieves accelerated unregularized convergence rates, up to approximation-error terms: $\tilde { \mathcal { O } } ( T _ { t o t a l } ^ { - 1 } )$ in the Stochastic Regime, and $\tilde { \mathcal { O } } ( T _ { t o t a l } ^ { - 2 / 3 } )$ for the average iterate alongside $\tilde { \mathcal { O } } ( T _ { t o t a l } ^ { - 1 / 3 } )$ for the last iterate in the Deterministic Regime. Here, $T _ { t o t a l }$ denotes the total number of stochastic critic updates (or Monte Carlo rollouts). Furthermore, in the tabular setting, our positive-action-gap analysis yields a $\tilde { \mathcal { O } } ( T _ { t o t a l } ^ { - 2 / 3 } )$ average-iterate rate, surpassing the $\mathcal { O } ( T _ { t o t a l } ^ { - 1 / 2 } )$ worst-case statistical barrier that applies without a positive action margin.

Key words and phrases. Entropy Regularization, Finite-Sample Analysis, Linear Function Approximation, Natural Actor-Critic, Natural Policy Gradient, Policy Mirror Descent, Reinforcement Learning, Single-Loop Algorithms.

## Contents

1 Introduction 3   
2 Problem Formulation 8   
2.1 Setup and Optimization Regimes 8   
2.2 Uncentered Natural Actor-Critic Algorithm 11   
2.3 Structural Assumptions 14   
3 Suboptimality Identities and Translation Bounds 17   
3.1 Suboptimality Identities and Decompositions 17   
3.2 Exponential Translation Bounds 19   
I Stochastic Regime 20   
4 Parameter-Space Polyak- Lojasiewicz Condition 24   
5 Actor Progress and Critic Tracking 25   
5.1 Uncentered Gradients and Objective Smoothness 25   
5.2 Actor Progress and Algorithmic Coupling 26   
5.3 Coupled Critic Tracking . 27   
6 Fixed-Temperature Convergence (Stochastic Regime) 28   
6.1 Joint Actor-Critic System 28   
6.2 Average-Iterate Regularized Convergence 30   
6.3 Last-Iterate Regularized Convergence 31   
7 Unregularized Global Convergence (Stochastic Regime) 32   
7.1 Global Convergence Bounds . . . 32   
7.2 Average-Iterate Unregularized Convergence 34   
7.3 Last-Iterate Unregularized Convergence 35   
II Deterministic Regime 36   
8 Policy Mirror Descent 39   
9 Fixed-Temperature Convergence (Deterministic Regime) 40   
9.1 Actor Progress and Value Telescoping 41   
9.2 Uncoupled Critic Tracking . 42   
9.3 Average-Iterate Regularized Convergence 43   
9.4 Last-Iterate Regularized Convergence 44   
10 Unregularized Global Convergence (Deterministic Regime) 46   
10.1 Average-Iterate Unregularized Convergence 46   
10.2 Last-Iterate Unregularized Convergence 48   
10.3 Application to Tabular Setting 48   
References 49   
A Algorithm Comparison 53   
A.1 Standard NPG with Advantage-Driven, Centered Critic 53   
A.2 NPG with Q-Value-Driven, Centered Critic 54   
A.3 Equivalence under Exact Parameterization 54   
A.4 Algorithmic Design and Comparison 55   
B Generalization to Exploratory ν-Sampling 56   
B.1 Generalized Sampling Measure and Critic Assumptions 56   
B.2 Generalized Of-Policy Mismatches . 58   
B.3 Impact on Part I (Stochastic Regime) 60   
B.4 Impact on Part II (Deterministic Regime) 63   
C Concentrability-Free Analysis via L<sub>∞</sub> Approximation 64   
C.1 Complete Concentrability Elimination in Part II 64   
C.2 Generalization to Exploratory ν-Sampling (Deterministic Regime) 68   
C.3 Impact on Part I and Geometric Bottleneck 69   
D Related Works 70   
D.1 Agarwal et al. (2021) 70   
D.2 Yuan et al. (2023) 75   
D.3 Liu et al. (2020) 77   
D.4 Cayci et al. (2024) 81   
E Technical Details 85   
E.1 Proofs for Section 2 85   
E.2 Proofs for Section 3 87   
E.3 Proofs for Section 4 94   
E.4 Proofs for Section 5 96   
E.5 Proofs for Section 6 102   
E.6 Proofs for Section 7 106   
E.7 Proofs for Section 8 114   
E.8 Proofs for Section 9 115   
E.9 Proofs for Section 10 124   
F Unregularized Convergence via Linear Entropy Penalty (Stochastic Regime) 129   
F.1 Average-Iterate Unregularized Convergence 129   
F.2 Last-Iterate Unregularized Convergence 131   
F.3 Global Unregularized Convergence 132   
G Unifying Single-Loop and Double-Loop via Warm-Start Critic 133   
G.1 Warm-Start Critic Tracking Error 134   
G.2 Warm-Start Average-Iterate Regularized Convergence 138   
G.3 Warm-Start Unregularized Convergence (Minimal Action Gap) . 141   
G.4 Warm-Start Unregularized Convergence (Linear Entropy) 144

## 1 Introduction

Motivation. Policy gradient methods, particularly Natural Policy Gradient (NPG) (Kakade, 2001; Peters and Schaal, 2008), form the backbone of modern reinforcement learning. To encourage exploration and improve convergence, these algorithms are frequently deployed in tandem with entropy regularization (Williams and Peng, 1991; Peters et al., 2010; Mnih et al., 2016; Haarnoja et al., 2018). While the empirical benefits of entropy regularization—such as smoother optimization landscapes and faster convergence—are widely recognized, there remain substantive gaps in existing theoretical analyses to address the following question:

Compared with unregularized NPG algorithms, what convergence rates (if any faster) can entropy-regularized NPG algorithms be shown to achieve in terms of unregularized performance gaps by appropriate temperature tuning alongside step-size tuning?

Historically, much of the early convergence literature focused on the tabular Markov Decision Process (MDP). While early works established global convergence for both unregularized and regularized NPG using exact gradients (Mei et al., 2020; Bhandari and Russo, 2021; Cen et al., 2022), subsequent theoretical developments exploit the Policy Mirror Descent (PMD) framework, which is motivated by Trust Region Policy Optimization (TRPO) (Schulman et al., 2015). Under loglinear policy parameterizations, entropy-regularized NPG is equivalent to a PMD update with the Kullback-Leibler (KL) divergence as the proximity term. However, these PMD analyses typically study the unregularized and regularized settings in isolation (Shani et al., 2020; Lan, 2023), leaving the unregularized performance of regularized algorithms unevaluated. Even when regularized tabular algorithms are explicitly mapped to the unregularized objective, the translation relies on a standard linear O(τ ) entropy penalty. For Soft Policy Iteration (SPI) analyzed in Cen et al. (2022), the unregularized convergence rate is shown to be $\tilde { \mathcal { O } } ( T _ { t o t a l } ^ { - 1 / 2 } )$ (see Table 2).

Beyond the tabular setting, guarantees for entropy regularization with function approximation remain limited. For linear function approximation, Cayci et al. (2024) analyzed entropy-regularized NPG algorithms, obtaining an $\tilde { \mathcal { O } } ( T _ { t o t a l } ^ { - 1 / 3 } )$ convergence rate (up to the approximation error term) for the regularized objective, but without explicitly discussing the unregularized performance gap. When this analysis is mapped to the unregularized objective via the linear entropy penalty as shown in Appendix D.4, the resulting convergence rate would be $\tilde { \mathcal { O } } ( T _ { t o t a l } ^ { - 1 / 6 } )$ . This is significantly slower than the unregularized rates, $\tilde { \mathcal { O } } ( T _ { t o t a l } ^ { - 1 / 4 } )$ to $\tilde { \mathcal { O } } ( T _ { t o t a l } ^ { - 1 / 2 } )$ , established directly for unregularized

NPG algorithms (Agarwal et al., 2021; Liu et al., 2020; Yuan et al., 2023) (see Table 1). Identifying whether entropy-regularized algorithms can match or even break the existing unregularized rates via appropriate tuning remains an open theoretical challenge.

Compounding this challenge is that the existing analyses of NPG algorithms are heavily dominated by double-loop architectures. Within an outer iteration, these algorithms freeze the actor parameter and run an inner loop of stochastic gradient descent (SGD) to solve a least-squares projection for updating the critic parameter. While this double-loop approach simplifies the theoretical analysis, it can artificially inflate overall sample complexities or lead to highly skewed hyperparameter configurations to suppress statistical noise. Interestingly, the unbalanced configurations are required by the strongest existing theoretical rates: to achieve their $\tilde { \mathcal { O } } ( T _ { t o t a l } ^ { - 1 / 2 } )$ unregularized convergence limits, both the double-loop Q-NPG under function approximation (Yuan et al., 2023) (see Appendix D.2) and the tabular SPI (Cen et al., 2022) dictate large inner-loop iterations or batch evaluations alongside a very small number of outer policy updates. Such choices misalign with the single-loop implementations commonly used in practice.

Algorithm: Single-Loop, Entropy-Regularized Uncentered NAC. To bridge the gap between theory and practice, we study an entropy-regularized NPG algorithm in an actor-critic framework. Specifically, we analyze a single-loop, entropy-regularized Natural Actor-Critic (NAC) algorithm under a log-linear policy parameterization and a linear compatible critic. In contrast to double-loop methods, our algorithm updates the actor and the critic continuously at every iteration, executing exactly one critic SGD step per actor update (N = 1).

Notably, our algorithm evaluates the critic using uncentered features rather than action-centered features (the score function). Standard NPG algorithms typically project the advantage or Q-values onto centered features. However, such projections rely on the positive-definiteness of the centered feature covariance matrix (the Fisher information matrix). This reliance becomes problematic in the deterministic regime: as the temperature drops to zero to recover the unregularized MDP, the optimal policy collapses into a deterministic distribution. Consequently, as the training policy tracks this optimum and approaches determinism, its action randomness vanishes and its Fisher information matrix degenerates. By instead projecting the regularized Q-values onto uncentered features, our critic tracking can remain stable even when the Fisher information matrix degenerates.

Analysis and Results. To analyze our entropy-regularized algorithm as the temperature drops (τ → 0), we distinguish two optimization regimes: Stochastic Regime and Deterministic Regime,

Table 1: Comparison of sample-based NPG and NAC algorithms using linear function approximation (log-linear policies and linear compatible critics). $T _ { t o t a l } = N { \times } T$ denotes the total number of stochastic updates (or Monte Carlo rollouts) across T outer-loop iterations and N inner-loop steps, translating to the total number of environment interactions up to an additional $\mathcal { O } ( ( 1 - \gamma ) ^ { - 1 } )$ expected-horizon factor.
<table><tr><td>Reference</td><td>Algorithm</td><td>Setting</td><td>Obj</td><td>Rate</td></tr><tr><td>Agarwal et al. (2021) (Corollary 26)</td><td>Double-loop Q-NPG</td><td>Discounted, indep. sampling</td><td>Unreg</td><td> $\mathrm { A v g } \colon \mathcal { O } ( T _ { t o t a l } ^ { - 1 / 4 } ) + \mathcal { O } ( \sqrt { \epsilon _ { a p p } } )$ </td></tr><tr><td>Liu et al. (2020) (Theorem 4.9)</td><td>Double-loop NPG</td><td>Discounted, indep. sampling</td><td>Unreg</td><td>Avg:  $\mathcal { O } ( T _ { t o t a l } ^ { - 1 / 3 } ) + \mathcal { O } ( \sqrt { \epsilon _ { a p p } } )$ </td></tr><tr><td>Yuan et al. (2023) (Corollary 1)</td><td>Double-loop Q-NPG</td><td>Discounted, indep. sampling</td><td>Unreg</td><td>Last:  $\tilde { \mathcal { O } } ( T _ { t o t a l } ^ { - 1 / 2 } ) + \mathcal { O } ( \sqrt { \epsilon _ { a p p } } )$ </td></tr><tr><td>Cayci et al. (2024) (Prop. 4.1, Rem. 4.6)</td><td>Double-loop ent-reg NPG</td><td>Discounted, indep. sampling</td><td>Reg (τ &gt; 0)</td><td>Avg:  $\tilde { \mathcal { O } } ( T _ { t o t a l } ^ { - 1 / 3 } ) + \mathcal { O } ( \sqrt { \epsilon _ { a p p } } )$ </td></tr><tr><td>Wang et al. (2024) (Theorem 2)</td><td>Single-loop NAC</td><td>Average-reward, Markovian</td><td>Unreg</td><td>Avg:  $\tilde { \mathcal { O } } ( T _ { t o t a l } ^ { - 1 / 3 } ) + \mathcal { O } ( \sqrt { \epsilon _ { a p p } } )$ </td></tr><tr><td>Ours (Part II) (Theorems 10.1, 10.2)</td><td>Single-loop ent-reg, unc NAC</td><td>Discounted, indep. sampling</td><td>Unreg</td><td>Avg:  $\tilde { \mathcal { O } } ( T _ { t o t a l } ^ { - 2 / 3 } ) + \tilde { \mathcal { O } } ( \epsilon _ { a p p } )$  Last:  $\tilde { \mathcal { O } } ( T _ { t o t a l } ^ { - 1 / 3 } ) + \tilde { \mathcal { O } } ( \sqrt { \epsilon _ { a p p } } )$ </td></tr><tr><td>Ours (Part I) (Theorems 7.1, 7.2)</td><td>Single-loop ent-reg, unc NAC</td><td>Discounted, indep. sampling</td><td>Unreg</td><td>Avg:  $\tilde { \mathcal { O } } ( T _ { t o t a l } ^ { - 1 } ) + \tilde { \mathcal { O } } ( \epsilon _ { a p p } )$  Last:  $\tilde { \mathcal { O } } ( T _ { t o t a l } ^ { - 1 } ) + \tilde { \mathcal { O } } ( \epsilon _ { a p p } )$ </td></tr></table>

Obj: objective; Unreg: unregularized; Reg: regularized; ent-reg: entropy-regularized; unc: uncentered; indep. sampling: conditionally independent sampling from the discounted visitation measure of the current policy; Avg: average-iterate bound; Last: last-iterate bound. The stated Part-I bounds are understood over the admissible range for $\epsilon _ { a p p }$ , excluding $\epsilon _ { a p p } = 0 $ , as described in Remark 7.1.

Note: ν-sampling initializes trajectories from an exploratory restart distribution to ensure state-action coverage, whereas a Generative Model assumes access to a simulator capable of drawing next-state transitions from any arbitrary state-action query.

Table 2: Comparison of sample-based NPG and PMD algorithms in the tabular unregularized setting, free of concentrability conditions. $T _ { t o t a l }$ denotes the total number of stochastic updates, translating to environment interactions up to an additional $\mathcal { O } ( ( 1 - \gamma ) ^ { - 1 } )$ expected-horizon factor.
<table><tr><td>Reference</td><td>Algorithm</td><td>Unregularized Rate</td></tr><tr><td>Shani et al. (2020) (Thm. 5) TRPO (with ν-sampling)</td><td></td><td>Avg:  $\tilde { \mathcal { O } } ( T _ { t o t a l } ^ { - 1 / 4 } )$ </td></tr><tr><td>Cen et al. (2022) (Rem. 1)</td><td>SPI (Generative Model)</td><td>Last:  $\tilde { \mathcal { O } } ( T _ { t o t a l } ^ { - 1 / 2 } )$ </td></tr><tr><td>Lan (2023) (Prop. 2)</td><td>SPMD (Generative Model)</td><td> $\operatorname { A v g } \colon \tilde { \mathcal { O } } ( T _ { t o t a l } ^ { - 1 / 2 } )$ </td></tr><tr><td rowspan="2">Ours (Theorem 10.3)</td><td>ent-reg, unc NAC</td><td> $\operatorname { A v g } \colon \tilde { \mathcal { O } } ( T _ { t o t a l } ^ { - 2 / 3 } )$ </td></tr><tr><td>(with ν-sampling)</td><td>Last:  $\tilde { \mathcal { O } } ( T _ { t o t a l } ^ { - 1 / 3 } )$ </td></tr></table>

which have not been explicitly separated in prior literature. We develop tailored analytical tools for each regime and establish new sample complexities for the unregularized objective, summarized in Tables 1 and 2. For simplicity, we discuss convergence rates up to approximation error terms.

• Coupled Lyapunov Analysis (Stochastic Regime): When the regularized parametric optimum retains strict stochasticity as $\tau  0$ , the Fisher information matrix remains uniformly positivedefinite. By establishing a Parameter-Space Polyak- Lojasiewicz (PL) condition and an Actor Progress Bound, we obtain both Coupled Actor Recurrence and Coupled Critic Tracking. We then fuse these two coupled bounds into a joint Lyapunov function, proving an optimal $\tilde { \mathcal { O } } ( T _ { t o t a l } ^ { - 1 } )$ average-iterate and last-iterate regularized convergence rate.

• Policy Mirror Descent (Deterministic Regime): When the regularized parametric optimum approaches a deterministic policy as $\tau  0$ , the action randomness vanishes and the Fisher information degenerates, breaking the Euclidean parameter-space geometry. To survive this limit, we pivot to a Policy Mirror Descent (PMD) geometry. By evaluating algorithmic progress directly on the probability simplex via KL divergence, we decouple the actor’s progress and the uncentered critic tracking, establishing an accelerated $\tilde { \mathcal { O } } ( T _ { t o t a l } ^ { - 2 / 3 } )$ ) averageiterate and $\tilde { \mathcal { O } } ( T _ { t o t a l } ^ { - 1 / 3 } )$ last-iterate regularized convergence rate.

• Exponential Translation Mechanism: To map these regularized rates back to the unregularized MDP, we exploit the Minimal Action Gap, a structural condition dictating a positive margin between the optimal and suboptimal unregularized action values. Under this condition, we prove the unregularized suboptimality is bounded by the regularized gap plus an exponentially small translation tail. By tuning the fixed temperature to balance this exponential tail against the regularized optimization error, our algorithm achieves unregularized rates that match the regularized rates up to logarithmic factors.

• Warm-Start Unification of Single- and Double-Loop Architectures: To connect our single-loop algorithm with conventional double-loop implementations, we extend the PMD analysis to a warm-start architecture that performs N critic SGD steps per actor update while initializing each inner loop from the previous critic output. For average-iterate convergence, the accelerated $\tilde { \mathcal { O } } ( T _ { t o t a l } ^ { - 2 / 3 } )$ unregularized rate remains invariant across $N \in [ 1 , T _ { t o t a l } ^ { 2 / 9 } ]$ . Under the standard linear entropy penalty, the baseline $\tilde { \mathcal { O } } ( T _ { t o t a l } ^ { - 1 / 4 } )$ rate remains invariant across the larger window $N \in [ 1 , \sqrt { T _ { t o t a l } } ]$ , directly connecting the single-loop and balanced double-loop endpoints.

• Application to Tabular Setting: We apply our function approximation theory to the tabular setting equipped with a one-hot feature encoding. By employing an exploratory restart distribution (ν-sampling) to guarantee suficient state-action coverage during critic training, our single-loop, entropy-regularized algorithm achieves a $\tilde { \mathcal { O } } ( T _ { t o t a l } ^ { - 2 / 3 } )$ average-iterate unregularized rate under the Minimal Action Gap. This instance-dependent rate surpasses the $\mathcal { O } ( T _ { t o t a l } ^ { - 1 / 2 } )$ worst-case statistical barrier governing tabular MDPs without a positive action margin.

Related Works. The literature on policy gradient methods is extensive. We refer to $\mathrm { A p \mathrm { - } }$ pendix D for extended discussions of the four analyses of sample-based algorithms stated in Table 1 (Agarwal et al., 2021; Liu et al., 2020; Yuan et al., 2023; Cayci et al., 2024). Here, we focus on the non-asymptotic literature concerning sample-based Natural Actor-Critic (NAC) algorithms. A significant portion of these works explicitly address Markovian sampling. While our current analysis utilizes conditionally independent sampling (with an exploratory ν-sampling extension), we leave the extension of our theory to Markovian trajectories for future work.

In the tabular setting, Khodadadian et al. (2021) and Khodadadian et al. (2023) established finite-sample guarantees for double-loop and single-loop (two-timescale) NAC algorithms, respectively. Extending to linear function approximation, Xu et al. (2020) and Chen et al. (2022) analyzed double-loop NAC algorithms under Markovian sampling. Recently, Wang et al. (2024) analyzed a single-loop NAC algorithm for average-reward MDPs, achieving an $\tilde { \mathcal { O } } ( T _ { t o t a l } ^ { - 1 / 3 } )$ unregularized rate under Markovian sampling, similar to that of Liu et al. (2020) under conditionally independent sampling. Like Liu et al. (2020), this analysis operates within Euclidean parameter spaces, relying on the assumption that the Fisher information matrix remains uniformly positive-definite. While appropriate within our Stochastic Regime, this approach would break down in the Deterministic Regime, where the Fisher information degenerates as the training policy approaches determinism.

Notation. For a vector $x ,$ we use $\| x \| _ { 1 } , \| x \| _ { 2 }$ , and $\| x \| _ { \infty }$ to denote the standard $L _ { 1 }$ , Euclidean, and $L _ { \infty }$ norms, respectively. For a matrix A, $\| A \| _ { 2 }$ denotes its spectral norm. For a measurable function $f , \| f \| _ { \infty }$ denotes its essential supremum (or maximum when the underlying space is finite). We use $\mathcal { O } ( \cdot )$ and $\Theta ( \cdot )$ for standard asymptotic upper and matching-order bounds, respectively, and use $\tilde { \mathcal { O } } ( \cdot )$ and $\tilde { \Theta } ( \cdot )$ to additionally suppress logarithmic factors in the relevant asymptotic variables, such as $T , \ T _ { t o t a l }$ , and, when applicable, $\epsilon _ { a p p } ^ { - 1 }$ . Asymptotic notation may suppress fixed problemdependent constants, while dependence displayed explicitly, such as those on $( 1 - \gamma ) ^ { - 1 } , \kappa , \lambda$ , and $\Delta .$ , is retained. All logarithms are natural logarithms.

## 2 Problem Formulation

We consider an infinite-horizon discounted Markov Decision Process (MDP) defined by the tuple $( \mathcal { S } , \mathcal { A } , \mathcal { P } , r , \gamma , \mu )$ , where S is the state space (allowed to be infinite), A is the action space (assumed to be finite of size $| { \mathcal { A } } | ) , { \mathcal { P } } ( \cdot | s , a )$ is the transition probability measure over the next state given current $( s , a ) , r ( s , a ) \in [ 0 , 1 ]$ is the reward function, $\gamma \in ( 0 , 1 )$ is the discount factor, and $\mu$ is the initial state distribution. A policy $\pi ( a | s )$ defines the probability of selecting action a in state s.

Let $( s _ { t } , a _ { t } )$ denote the state-action pair at time step $t \geq 0$ . Throughout the paper, we use P and E to denote probability and expectation. To avoid ambiguity, we subscript the expectation operator to indicate the random variables and their associated distributions $( \mathrm { e . g . }$ , expectation over a policy trajectory $\mathbb { E } _ { \pi }$ , or over specific state-action marginals $\mathbb { E } _ { s \sim d ^ { \pi } }$ or $\mathbb { E } _ { a \sim \pi } )$ . The discounted state visitation measure for a policy $\pi$ is denoted by $d ^ { \pi }$ , defined such that for any measurable subset $\begin{array} { r } { \tilde { \mathcal { S } } \subset \mathcal { S } , d ^ { \pi } ( \tilde { \mathcal { S } } ) = ( 1 - \gamma ) \sum _ { t = 0 } ^ { \infty } \gamma ^ { t } \mathbb { P } ( s _ { t } \in \tilde { \mathcal { S } } \mid s _ { 0 } \sim \mu , \pi ) } \end{array}$ . For convenience, we use $d ^ { \pi } ( s )$ to denote its density (or mass in the discrete case). The dependence on $\mu$ is suppressed in the notation.

## 2.1 Setup and Optimization Regimes

The entropy-regularized objective from the initial state distribution $\mu$ is (Neu et al., 2017; Geist et al., 2019)

$$
\begin{array} { r l r } & { } & { J _ { \tau } ( \pi ) = \mathbb { E } _ { \pi , s _ { 0 } \sim \mu } \left[ \displaystyle \sum _ { t = 0 } ^ { \infty } \gamma ^ { t } \big ( r ( s _ { t } , a _ { t } ) - \tau \log \pi ( a _ { t } | s _ { t } ) \big ) \right] } \\ & { } & { ~ = \displaystyle \frac { 1 } { 1 - \gamma } \mathbb { E } _ { s \sim d ^ { \pi } } \Big [ \mathbb { E } _ { a \sim \pi } [ r ( s , a ) ] + \tau H ( \pi ( \cdot | s ) ) \Big ] , } \end{array}
$$

where $\tau > 0$ is the entropy temperature, and $\begin{array} { r } { H ( \pi ( \cdot | s ) ) \triangleq - \sum _ { a } \pi ( a | s ) \log \pi ( a | s ) } \end{array}$ is the Shannon entropy. Since we are primarily interested in the limits of entropy-regularized algorithms as the temperature decreases $( \tau  0 )$ , we consider the temperature to be bounded by a base constant throughout the paper, i.e., $\tau \leq \tau _ { m a x }$ for some $\tau _ { m a x } > 0$

The regularized state-value function includes the instantaneous entropy penalty at all steps:

$$
V _ { \tau } ^ { \pi } ( s ) = \mathbb { E } _ { \pi } \left[ \sum _ { t = 0 } ^ { \infty } \gamma ^ { t } \big ( r ( s _ { t } , a _ { t } ) - \tau \log \pi ( a _ { t } | s _ { t } ) \big ) \ \Bigg | \ s _ { 0 } = s \right] .
$$

The action-value function evaluates the expected return after taking a specific initial action a. Therefore, it excludes the entropy penalty for that first fixed action:

$$
Q _ { \tau } ^ { \pi } ( s , a ) = r ( s , a ) + \mathbb { E } _ { \pi } \left[ \sum _ { t = 1 } ^ { \infty } \gamma ^ { t } \big ( r ( s _ { t } , a _ { t } ) - \tau \log \pi ( a _ { t } | s _ { t } ) \big ) ~ \Bigg | ~ s _ { 0 } = s , a _ { 0 } = a \right] .
$$

These definitions satisfy the recursive soft Bellman relations:

$$
\begin{array} { r l } & { V _ { \tau } ^ { \pi } ( s ) = \displaystyle \sum _ { a } \pi ( a \vert s ) \big ( Q _ { \tau } ^ { \pi } ( s , a ) - \tau \log \pi ( a \vert s ) \big ) , } \\ & { } \\ & { Q _ { \tau } ^ { \pi } ( s , a ) = r ( s , a ) + \gamma \mathbb { E } _ { s ^ { \prime } \sim \mathcal { P } ( \cdot \vert s , a ) } [ V _ { \tau } ^ { \pi } ( s ^ { \prime } ) ] . } \end{array}
$$

The regularized advantage function is defined as $A _ { \tau } ^ { \pi } ( s , a ) = Q _ { \tau } ^ { \pi } ( s , a ) - \tau \log \pi ( a | s ) - V _ { \tau } ^ { \pi } ( s )$

We employ a log-linear softmax policy parameterized by $\omega \in \mathbb { R } ^ { d }$ , operating over a given feature map $\phi ( s , a ) \in \mathbb R ^ { d }$

$$
\pi _ { \omega } ( a | s ) = \frac { \exp ( \omega ^ { \top } \phi ( s , a ) ) } { \sum _ { a ^ { \prime } \in \mathcal { A } } \exp ( \omega ^ { \top } \phi ( s , a ^ { \prime } ) ) } .
$$

This can be written as $\pi _ { \omega } ( a | s ) = \exp ( \omega ^ { \top } \phi ( s , a ) ) / Z _ { \omega } ( s )$ using the normalizing constant $Z _ { \omega } ( s ) \ { \triangleq }$ $\begin{array} { r } { \sum _ { a ^ { \prime } } \exp ( \omega ^ { \top } \phi ( s , a ^ { \prime } ) ) } \end{array}$ . We assume that, for each $\tau \in ( 0 , \tau _ { m a x } ]$ , the parametric regularized objective admits a finite maximizer $\omega _ { \tau } ^ { * } \in \mathbb { R } ^ { d }$ . Let $\omega _ { \tau } ^ { * } \in \mathrm { a r g m a x } _ { \omega \in \mathbb { R } ^ { d } } J _ { \tau } ( \pi _ { \omega } )$ denote such an optimal parameter vector, and let $\pi _ { \omega _ { \tau } ^ { * } }$ be the corresponding regularized optimal policy within the parametric family. By abuse of notation, we also use $J _ { \tau } ( \omega ) \triangleq J _ { \tau } ( \pi _ { \omega } )$ . For a training policy $\pi _ { t } .$ , we define the parametric regularized performance gap as $\mathrm { G a p } _ { t } \triangleq J _ { \tau } ( \pi _ { \omega _ { \tau } ^ { * } } ) - J _ { \tau } ( \pi _ { t } )$ . We denote the discounted state visitation distribution of the regularized parametric optimum $\pi _ { \omega _ { \tau } ^ { * } }$ as $d ^ { \omega _ { \tau } ^ { * } } \triangleq d ^ { \pi _ { \omega _ { \tau } ^ { * } } }$

Furthermore, for any $\tau > 0$ , let $\pi _ { \tau } ^ { * }$ denote the unique globally optimal policy for the regularized $\mathrm { M D P }$ over the entire probability simplex, defined as the soft Bellman-optimal policy that maximizes the regularized state-value $V _ { \tau } ^ { \pi } ( s )$ for every state s. Thus $\pi _ { \tau } ^ { * }$ maximizes the scalar objective $J _ { \tau } ( \pi )$ for any initial distribution, including $\mu .$ By definition, the performance of the parametric optimum is bounded by the global optimum, such that $J _ { \tau } ( \pi _ { \omega _ { \tau } ^ { * } } ) \leq J _ { \tau } ( \pi _ { \tau } ^ { * } )$ . We define the global regularized performance gap as ${ \mathrm { G a p } } _ { t } ^ { \dagger } \triangleq J _ { \tau } ( \pi _ { \tau } ^ { * } ) - J _ { \tau } ( \pi _ { t } )$ , distinct from ${ \mathrm { G a p } } _ { t }$ . We denote the discounted state visitation distribution of the regularized globally optimal policy $\pi _ { \tau } ^ { * }$ as $d _ { \tau } ^ { \ast } \triangleq d ^ { \pi _ { \tau } ^ { \ast } }$ , and denote the state-value and action-value functions of $\pi _ { \tau } ^ { * }$ as $V _ { \tau } ^ { * } \triangleq V _ { \tau } ^ { \pi _ { \tau } ^ { * } }$ and $Q _ { \tau } ^ { * } \triangleq Q _ { \tau } ^ { \pi _ { \tau } ^ { * } }$

We aim to analyze the performance of entropy-regularized policy optimization in maximizing the unregularized objective while tuning the temperature. To formalize the limit as the temperature drops $( \tau  0 )$ , let $\pi _ { 0 } ^ { * }$ denote a globally optimal policy for the unregularized MDP, defined as maximizing the unregularized state-value $V _ { 0 } ^ { \pi } ( s )$ for every state s. By standard MDP theory, such a statewise global optimum always exists and can be chosen to be deterministic. We denote the discounted state visitation distribution of $\pi _ { 0 } ^ { * }$ as $d _ { 0 } ^ { * } \triangleq d ^ { \pi _ { 0 } ^ { * } }$ , and denote the state-value and actionvalue functions of $\pi _ { 0 } ^ { * }$ as $V _ { 0 } ^ { * } \triangleq V _ { 0 } ^ { \pi _ { 0 } ^ { * } }$ and $Q _ { 0 } ^ { * } \triangleq Q _ { 0 } ^ { \pi _ { 0 } ^ { * } }$

We primarily evaluate algorithmic performance against the global optimum, defining the global unregularized performance gap as $J _ { 0 } ( \pi _ { 0 } ^ { * } ) - J _ { 0 } ( \pi _ { t } )$ . To facilitate theoretical analysis, we assume a Minimal Action Gap on the unregularized advantage function for $\pi _ { 0 } ^ { * }$ (Assumption 5). From this structural property, we derive an Exponential Translation Bound (Theorem 3.1) to control the unregularized metric $J _ { 0 } ( \pi _ { 0 } ^ { * } ) - J _ { 0 } ( \pi _ { t } )$ via the regularized metric $\mathrm { G a p } _ { t } ^ { \dagger }$

Depending on the asymptotic behavior of the regularized optimal policies as $\tau  0$ , we focus on two primary regimes for the optimization landscape:

• Stochastic Regime: $\mathrm { A s } \tau  0$ , the sequence of regularized optimal policies $\pi _ { \omega _ { \tau } ^ { * } }$ converges to an unregularized parametric optimum $\pi _ { \omega _ { 0 } ^ { * } }$ with $\omega _ { \tau } ^ { * }  \omega _ { 0 } ^ { * } \in \mathbb { R } ^ { d }$ . Consequently, under Bounded Features (Assumption 1), for suficiently small $\tau _ { : }$ , the distributions $\pi _ { \omega _ { \tau } ^ { * } } ( \cdot | s )$ remain uniformly interior to the probability simplex for all states $s ,$ so that the action randomness does not vanish. To capture this persistent stochasticity and rule out redundant or action-independent feature directions, we assume that the centered feature covariance matrix (i.e., Fisher information matrix) at the parametric optimum $\pi _ { \omega _ { \tau } ^ { * } }$ remains uniformly positive-definite, independent of temperature $\tau$ (Assumption 6). In this regime, we also define the parametric unregularized performance gap as $J _ { 0 } ( \pi _ { \omega _ { 0 } ^ { * } } ) - J _ { 0 } ( \pi _ { t } )$ , distinct from $J _ { 0 } ( \pi _ { 0 } ^ { * } ) - J _ { 0 } ( \pi _ { t } )$

• Deterministic Regime: $\mathrm { A s } \tau  0$ , the sequence of regularized optimal policies $\pi _ { \omega _ { \tau } ^ { * } }$ approaches an unregularized parametric optimum that is a deterministic policy in the closure of the loglinear family. This deterministic collapse causes the action randomness to vanish and the Fisher information matrix to degenerate. A highly expressive log-linear family that perfectly captures the deterministic global optimum $\pi _ { 0 } ^ { * }$ naturally falls into this regime. However, deterministic collapse can also occur in a restricted family if its best unregularized policy in the closure is a suboptimal deterministic policy.

We analyze the two regimes sequentially in Parts I and II, both evaluating convergence against the global optimum $\pi _ { 0 } ^ { * }$ , based on the Minimal Action Gap assumption. In Part I, we tackle the Stochastic Regime by establishing a Parameter-Space Polyak- Lojasiewicz (PL) Condition (Lemma 4.2) and deriving an Actor Progress Bound (Lemma 5.3) through the smoothness of the regularized objective and a Positive-Definite Fisher Information assumption. In Part II, we address the Deterministic Regime, where the degeneration of the Fisher information matrix breaks the optimization geometry required for the Actor Progress Bound. To circumvent this, we pivot to a Policy Mirror Descent (PMD) framework, measuring optimization progress via Kullback-Leibler (KL) divergence (Lemma 8.1) and establishing a PMD Actor Progress Bound (Lemma 9.1). Alongside these diferences in measuring actor progress, the critic tracking is also analyzed in diferent manners: Part I couples the critic’s target drift to the objective descent via the Euclidean parameter geometry, whereas Part II bounds the target drift using a decoupled, worst-case parameter step. Finally, complementing our primary analysis, Appendices F and G.4 evaluate the unregularized performance gap in, respectively, the Stochastic and Deterministic Regimes using a standard linear entropy penalty. This alternative approach completely bypasses the Minimal Action Gap assumption, providing robust baseline convergence rates.

By the boundedness of the Shannon entropy $H ( \pi ( \cdot | s ) )$ , we state as a lemma the fact that the regularized value functions are unconditionally bounded for any policy.

Lemma 2.1 (Bounded Regularized Values). For any policy π, any state s, and any action a, the true regularized state-value and action-value are bounded:

$$
0 \leq V _ { \tau } ^ { \pi } ( s ) \leq \frac { 1 + \tau \log | A | } { 1 - \gamma } , \quad 0 \leq Q _ { \tau } ^ { \pi } ( s , a ) \leq \frac { 1 + \tau \log | A | } { 1 - \gamma } .
$$

Consequently for any temperature $\tau \in \mathsf { \Gamma } ( 0 , \tau _ { m a x } ]$ , we have $V _ { \tau } ^ { \pi } ( s ) \ \leq \ V _ { m a x }$ and $Q _ { \tau } ^ { \pi } ( s , a ) \ \leq \ V _ { m a x }$ globally, where $\begin{array} { r } { V _ { m a x } \triangleq \frac { 1 + \tau _ { m a x } \log | \mathcal { A } | } { 1 - \gamma } } \end{array}$

## 2.2 Uncentered Natural Actor-Critic Algorithm

We consider an entropy-regularized Natural Policy Gradient (NPG) algorithm with compatible function approximation in an actor-critic framework. An explicit policy (the actor ) is maintained to select actions, while a value function estimator (the critic) is learned to evaluate the natural policy gradient for guiding the policy update. For a training policy $\pi _ { t } \triangleq \pi _ { \omega _ { t } }$ parameterized by the actor parameter $\omega _ { t }$ , the ideal critic parameter $\theta _ { \tau } ^ { * } ( \omega _ { t } ) \in \mathbb { R } ^ { d }$ is defined as the minimizer of the Mean Squared Error (MSE) objective in the regularized Q-value projection:

$$
\theta _ { \tau } ^ { * } ( \omega _ { t } ) \triangleq \arg \operatorname* { m i n } _ { \theta \in \mathbb { R } ^ { d } } \mathbb { E } _ { s \sim d ^ { \pi _ { t } } , a \sim \pi _ { t } } \left[ \left( Q _ { \tau } ^ { \pi _ { t } } ( s , a ) - \theta ^ { \top } \phi ( s , a ) \right) ^ { 2 } \right] .
$$

The inherent approximation error is $\epsilon _ { t } ( s , a ) \ \triangleq \ Q _ { \tau } ^ { \pi _ { t } } ( s , a ) - \theta _ { \tau } ^ { * } ( \omega _ { t } ) ^ { \top } \phi ( s , a )$ . For the entropyregularized objective, the ideal uncentered NPG update incorporates the entropy penalty directly into the policy parameter space:

$$
\omega _ { t + 1 } = \omega _ { t } + \eta _ { t } \big ( \theta _ { \tau } ^ { * } ( \omega _ { t } ) - \tau \omega _ { t } \big ) ,
$$

where $\eta _ { t } > 0$ is the actor step size. This formulation adapts standard compatible NPG (Kakade, 2001; Peters and Schaal, 2008) to use uncentered Q-values and features. Under exact Q-function realizability, this coincides with the standard entropy-regularized NPG direction. Otherwise, it incurs an approximation bias quantified later (Lemma 5.1). See Appendix A for a detailed derivation and comparison. The critic parameter $\theta _ { \tau } ^ { * } ( \omega _ { t } )$ can be viewed to both parameterize $Q _ { \tau } ^ { \pi _ { t } } ( \cdot )$ and define the ideal actor update.

In practice, we concurrently train an actual critic parameter to track the ideal target $\theta _ { \tau } ^ { * } ( \omega _ { t } )$ using stochastic samples. Specifically, we define the filtration $\mathcal { F } _ { t }$ to include the algorithmic trajectory up to the parameters $( \theta _ { t } , \omega _ { t } )$ . At each iteration t, conditionally on $\mathcal { F } _ { t }$ , we draw a sample $( s _ { t } , a _ { t } ) \sim$ $d ^ { \pi _ { t } } \times \pi _ { t }$ (referred to as conditionally independent sampling) and obtain an unbiased estimate $\hat { Q } _ { t }$ of the regularized Q-value such that $\mathbb { E } [ \hat { Q } _ { t } \mid \mathcal { F } _ { t } , s _ { t } , a _ { t } ] = Q _ { \tau } ^ { \pi _ { t } } ( s _ { t } , a _ { t } )$ . We define the stochastic gradient of the critic update (corresponding to the gradient of half the MSE objective) as:

$$
\begin{array} { r } { g _ { t } ^ { c r } ( s _ { t } , a _ { t } ) = \big ( \theta _ { t } ^ { \top } \phi ( s _ { t } , a _ { t } ) - \hat { Q } _ { t } \big ) \phi ( s _ { t } , a _ { t } ) . } \end{array}\tag{1}
$$

Using this sample gradient, the critic parameter is updated to $\theta _ { t + 1 }$ via stochastic gradient descent (SGD), and the resulting critic tracking error is evaluated as $e _ { t } \triangleq \theta _ { t + 1 } - \theta _ { \tau } ^ { * } ( \omega _ { t } )$ . To ensure the unbiasedness of $\hat { Q } _ { t }$ , geometrically stopped Monte Carlo rollouts can be used as described in Yuan et al. (2023, Algorithm 3). Because such a rollout simulates a trajectory with multiple raw transitions until termination, translating the number of SGD updates into raw environment interactions incurs an additional $\mathcal { O } ( ( 1 - \gamma ) ^ { - 1 } )$ expected-horizon factor, which, for simplicity, is suppressed in our sample complexity discussion.

Alternatively, as in Temporal Diference (TD) learning (Sutton and Barto, 2018; Szepesv´ari, 2010), $\hat { Q } _ { t }$ can be constructed using bootstrapping, which incurs an approximation bias. The update of $\theta _ { t + 1 }$ using these two choices for $\hat { Q } _ { t }$ corresponds directly to the standard Monte Carlo and TD methods for action-value evaluation. While our theoretical analysis assumes unbiased Monte Carlo estimates to isolate the core dynamics, we leave the study of bootstrapping estimates to future work.

We employ an alternating update scheme: the actor parameter is updated to $\omega _ { t + 1 }$ using the freshly updated critic $\theta _ { t + 1 }$ . Accordingly, we define the efective gradient for the actor update as:

$$
g _ { t } ^ { a c } \triangleq \theta _ { t + 1 } - \tau \omega _ { t } .\tag{2}
$$

While early actor-critic theory often employs a simultaneous update scheme—where the actor evaluates its update using the concurrent critic $\theta _ { t }$ (e.g., Konda and Tsitsiklis, 2000; Bhatnagar et al., 2009; Wu et al., 2020)—our alternating update reflects the sequential execution flow of modern practical implementations. Nevertheless, we note that our theoretical analysis and convergence guarantees can be readily applied to the simultaneous update algorithm.

Algorithm 1 Single-Loop, Entropy-Regularized, Uncentered Natural Actor-Critic   
Input: Initial actor parameter ω<sub>0</sub> satisfying $\begin{array} { r } { \| \omega _ { 0 } \| _ { 2 } \le \frac { B _ { \theta } } { \tau _ { m a x } } } \end{array}$ , initial critic parameter $\theta _ { 0 }$ satisfying   
$\| \theta _ { 0 } \| _ { 2 } \le B _ { \theta }$ , temperature $\tau \in ( 0 , \tau _ { m a x } ]$ , critic projection radius $R _ { \theta } \left( = B _ { \theta } \right)$ , step-size schedules   
$\{ \eta _ { t } \} _ { t \ge 0 }$ and $\{ \alpha _ { t } \} _ { t \ge 0 } .$   
1: for $t = 0 , 1 , \ldots , T - 1$ do   
2: Sampling: Draw a state-action pair $( s _ { t } , a _ { t } ) \sim d ^ { \pi _ { t } } \times \pi _ { t }$ under the current policy $\pi _ { t } \triangleq \pi _ { \omega _ { t } }$   
3: Evaluation: Obtain an unbiased estimate $\hat { Q } _ { t }$ of the regularized action-value $Q _ { \tau } ^ { \pi _ { t } } ( s _ { t } , a _ { t } )$   
4: Critic Update: Compute the stochastic critic gradient (1) and update via constrained SGD:   
$\theta _ { t + 1 } = \Pi _ { R _ { \theta } } \Big [ \theta _ { t } - \alpha _ { t } g _ { t } ^ { c r } \big ( s _ { t } , a _ { t } \big ) \Big ] .$   
5: Actor Update: Update the actor parameter using the efective actor gradient (2):   
$\omega _ { t + 1 } = \omega _ { t } + \eta _ { t } g _ { t } ^ { a c } = \omega _ { t } + \eta _ { t } \big ( \theta _ { t + 1 } - \tau \omega _ { t } \big )$   
6: end for   
Output: The output policy sequence $\{ \pi _ { t } \} _ { t = 0 } ^ { T } .$

In summary, the entropy-regularized, uncentered Natural Actor-Critic (NAC) algorithm executes these coupled updates iteratively as a single-loop method (Algorithm 1). Instead of employing a double-loop architecture that freezes the actor to perform multiple critic updates, we continuously interleave a single critic update with a single actor update at every iteration. To guarantee algorithmic stability, the critic is constrained to a ball of radius $R _ { \theta }$ (taking $R _ { \theta } = B _ { \theta }$ for the structural constant $B _ { \theta }$ in Lemma 2.2 later). We reserve the unsubscripted expectation E[·] to denote the expectation over the randomness of the training algorithm (i.e., the dynamic sampling process).

We emphasize two core aspects of our algorithmic design: (i) using uncentered Q-values and uncentered features, and (ii) realizing the NPG update directly via the action-value parameter iterate $\theta _ { t + 1 }$ , instead of solving the least-squares projection in a double-loop architecture. While the use of Q-values in NPG (often referred to as Q-NPG) is considered in prior literature (Agarwal et al., 2021; Yuan et al., 2023), such theoretical analyses generally do not study estimating the NPG via the action-value parameter iterates in a single-loop scheme. We refer readers to Appendix A for a further comparison between this uncentered algorithmic choice and standard centered NPG algorithms with compatible function approximation.

In Appendix B, we extend our theoretical analysis to accommodate exploratory ν-sampling by introducing generalized of-policy mismatch bounds. While the main text focuses on on-policy sampling from the objective’s initial distribution $\mu ,$ employing a restart distribution ν is a standard mechanism to improve the state-action coverage for training the critic (Agarwal et al., 2021; Yuan et al., 2023). Subtle diferences emerge across our Part I and Part II analyses concerning how the concentrability and mismatch bounds are invoked. Nevertheless, subject to these generalized assumptions, the asymptotic convergence rates remain identical to those in the main text.

## 2.3 Structural Assumptions

Before detailing the specific conditions, we make a general note on the structural assumptions used throughout this paper. To accommodate our goal of establishing unregularized convergence as the temperature drops $( \tau  0 )$ , we formulate several assumptions (Assumptions 2–4 in this section, and Assumptions 6–10 later) requiring their respective structural constants to be independent of $\tau \in \mathsf { \Gamma } ( 0 , \tau _ { m a x } ]$ . However, for the regularized convergence with a fixed temperature $\tau > 0$ , our theoretical results remain valid even if these involved constants exhibit a τ -dependence.

For analytical simplicity, to control the magnitudes of the actor and critic updates, we first assume the feature representations of the state-action pairs are bounded.

Assumption 1 (Bounded Features). $\begin{array} { r } { \operatorname* { s u p } _ { s , a } \| \phi ( s , a ) \| _ { 2 } \leq B _ { \phi } } \end{array}$

Algorithm 1 relies on stochastic estimates of the regularized action-values. As shown in Lemma 2.1, the true regularized action-values $Q _ { \tau } ^ { \pi _ { t } } ( s , a )$ are bounded by the constant $V _ { m a x }$ which is independent of $\tau \in ( 0 , \tau _ { m a x } ]$ . For analytical simplicity, we assume a similar uniform, τ -independent bound on the second moment of the unbiased estimate $\hat { Q } _ { t }$

Assumption 2 (Bounded Q-value Second Moment). There exists a constant $Q _ { m a x } > 0$ , independent of the temperature $\tau \in ( 0 , \tau _ { m a x } ]$ , such that $\mathbb { E } [ \hat { Q } _ { t } ^ { 2 } \mid \mathcal { F } _ { t } , s _ { t } , a _ { t } ] \leq Q _ { m a x } ^ { 2 }$ almost surely for all iterations $t \geq 0$ under Algorithm 1. By the Tower Property, this directly implies $\mathbb { E } [ \hat { Q } _ { t } ^ { 2 } \mid \mathcal { F } _ { t } ] \leq Q _ { m a x } ^ { 2 }$ Consistent with the true regularized Q-values (Lemma 2.1), we assume $Q _ { m a x } = \mathcal { O } ( ( 1 - \gamma ) ^ { - 1 } )$ .

To ensure the critic’s tracking error contracts stably across both the stochastic and deterministic regimes, we require the uncentered feature moment matrix to remain uniformly positive-definite for all training policies. For any policy π, let $\bar { \Sigma } _ { u n c } ( \pi ) \triangleq \mathbb { E } _ { s \sim d ^ { \pi } , a \sim \pi } [ \phi ( s , a ) \phi ( s , a ) ^ { \top } ]$

Assumption 3 (Positive-Definite Uncentered Feature Moment). There exists a constant $\kappa > 0$ independent of the temperature $\tau \in \mathsf { \Gamma } ( 0 , \tau _ { m a x } ]$ , such that for all training policies $\{ \pi _ { t } \} _ { t \ge 0 }$ under Algorithm 1:

$$
\bar { \Sigma } _ { u n c } ( \pi _ { t } ) \succeq \kappa I ,
$$

where I is the identity matrix, and $A \succeq B$ denotes that $A - B$ is a positive semi-definite matrix.

By combining Assumptions 1, 2, 3, and the bounded regularized values (Lemma 2.1), we establish temperature-independent bounds on the ideal uncentered critic, the actor parameter iterates, and the actor and critic update directions.

Lemma 2.2 (Structural Bounds of the Algorithm). (i) Under Assumptions 1 and ${ \mathcal { B } } ,$ the ideal uncentered critic is bounded for all training policies by the τ -independent constant $B _ { \theta }$ :

$$
\| \theta _ { \tau } ^ { * } ( \omega _ { t } ) \| _ { 2 } \leq \frac { B _ { \phi } V _ { m a x } } { \kappa } \triangleq B _ { \theta } .
$$

Additionally under Assumption ${ \mathit { 2 } } ,$ the critic update direction is bounded by a temperatureindependent constant $G _ { c r } > 0$

$$
\mathbb { E } \big [ \| g _ { t } ^ { c r } ( s _ { t } , a _ { t } ) \| _ { 2 } ^ { 2 } \mid \mathcal { F } _ { t } \big ] \leq 2 B _ { \phi } ^ { 2 } \big ( B _ { \theta } ^ { 2 } B _ { \phi } ^ { 2 } + Q _ { m a x } ^ { 2 } \big ) \triangleq G _ { c r } ^ { 2 } .
$$

(ii) Under Assumptions 1 and 3, by setting the critic projection radius to $R _ { \theta } = B _ { \theta }$ , if the actor step size satisfies $\eta _ { t } \leq 1 / \tau$ and the actor is initialized such that $\begin{array} { r } { \| \omega _ { 0 } \| _ { 2 } \le \frac { B _ { \theta } } { \tau _ { m a x } } } \end{array}$ (for example, $\omega _ { 0 } = 0 )$ then the actor iterates $\omega _ { t }$ are bounded for all $t \geq 0$ by:

$$
\| \omega _ { t } \| _ { 2 } \leq \frac { B _ { \theta } } { \tau } .
$$

Moreover, the actor update direction is bounded by a temperature-independent constant $G _ { a c } > 0 .$

$$
\| g _ { t } ^ { a c } \| _ { 2 } \leq 2 B _ { \theta } \triangleq G _ { a c } .
$$

The following lemma establishes the Lipschitz continuity of the ideal uncentered critic mapping $\theta _ { \tau } ^ { * } ( \omega )$ , subject to the positive-definiteness of the uncentered feature moment matrix at the associated parameter $\omega .$ Consequently, by Assumption 3, this lemma can be applied to the parameters of any two training policies generated by Algorithm 1. We explicitly define a Lipschitz constant through the maximum temperature $\tau _ { m a x }$ to ensure asymptotic stability as $\tau  0$

Lemma 2.3 (Lipschitz Ideal Uncentered Critic). For any $\kappa > 0$ , define the parameter set $\Omega _ { \kappa } \triangleq$ $\{ \omega \in \mathbb { R } ^ { d } : \bar { \Sigma } _ { u n c } ( \pi _ { \omega } ) \succeq \kappa I \}$ . Under Assumption 1, the ideal uncentered critic mapping $\theta _ { \tau } ^ { * } ( \omega )$ is

$L _ { \theta , \tau ^ { - } } L i$ pschitz continuous over $\Omega _ { \kappa }$ . Specifically, the Lipschitz constant is bounded by:

$$
L _ { \theta , \tau } \triangleq \frac { 2 B _ { \phi } ^ { 2 } ( 1 + \tau \log | \cal { A } | ) } { \kappa ( 1 - \gamma ) ^ { 2 } } \left( 3 + \frac { B _ { \phi } ^ { 2 } } { \kappa } \right) ,
$$

such that for any ω $, \omega ^ { \prime } \in \Omega _ { \kappa } , \| \theta _ { \tau } ^ { * } ( \omega ) - \theta _ { \tau } ^ { * } ( \omega ^ { \prime } ) \| _ { 2 } \leq L _ { \theta , \tau } \| \omega - \omega ^ { \prime } \| _ { 2 }$ . Consequently, for any temperature $\tau \in ( 0 , \tau _ { m a x } ]$ , the mapping $\theta _ { \tau } ^ { * } ( \omega )$ is L<sub>θ</sub>-Lipschitz continuous over $\Omega _ { \kappa }$ , where $L _ { \theta } \triangleq L _ { \theta , \tau _ { m a x } }$

Under compatible function approximation, the expressivity of the ideal uncentered critic is tied to the feature representation of the log-linear policy class. We assume this shared architecture is suficiently expressive such that the critic $L _ { 2 }$ approximation error remains uniformly bounded across the sequence of training policies $\{ \pi _ { t } \} _ { t \ge 0 }$ . For the global convergence in Part I (Section 7), we additionally require this bound to hold at the regularized parametric optimum $\pi _ { \omega _ { \tau } ^ { * } }$

Assumption 4 (Critic $L _ { 2 }$ Approximation Error). The MSE ofthe ideal uncentered critic is bounded by $\epsilon _ { a p p }$ for all training policies $\{ \pi _ { t } \} _ { t \ge 0 }$ under Algorithm 1:

$$
\begin{array} { r } { \mathbb { E } _ { d ^ { \pi _ { t } } , \pi _ { t } } [ \epsilon _ { t } ( s , a ) ^ { 2 } ] \le \epsilon _ { a p p } . } \end{array}
$$

Additionally, for the global analysis in Part I (specifically, Lemma $\ 7 . 1 )$ , we assume this bound holds at the regularized parametric optimum:

$$
\begin{array} { r } { \mathbb { E } _ { d ^ { \omega _ { \tau } ^ { * } } , \pi _ { \omega _ { \tau } ^ { * } } } [ \epsilon _ { * } ( s , a ) ^ { 2 } ] \le \epsilon _ { a p p } , } \end{array}
$$

where $\epsilon _ { * } ( s , a ) \triangleq Q _ { \tau } ^ { \pi _ { \omega _ { \tau } ^ { * } } } ( s , a ) - \theta _ { \tau } ^ { * } ( \omega _ { \tau } ^ { * } ) ^ { \top } \phi ( s , a )$ . For analytical simplicity, we assume the bound $\epsilon _ { a p p }$ is independent of the temperature $\tau \in ( 0 , \tau _ { m a x } ]$

Remark 2.1 $( L _ { \infty }$ Approximation and Concentrability). Because the sample-based critic in Algorithm 1 is trained via stochastic gradient descent, the $L _ { 2 }$ error $\left( \epsilon _ { a p p } \right)$ is the native and practically aligned metric for our analysis. Our theoretical guarantees rely on density ratio bounds (Assumptions 7 and 9) to manage various distribution shifts, including shifting this $L _ { 2 }$ error across state measures, and to facilitate associated $\chi ^ { 2 } \mathrm { - t o - K L }$ divergence translations. In Appendix C, we demonstrate that if the function approximation error is instead uniformly bounded in the $L _ { \infty }$ norm, then the global joint concentrability constant $( C _ { j o i n t } ^ { \dagger } )$ can be eliminated in the Deterministic Regime (Part II). We also isolate a fundamental geometric bottleneck that prevents the full elimination of the parametric joint concentrability constant $\left( { { C } _ { j o i n t } } \right)$ in the Stochastic Regime (Part I).

To facilitate our study of unregularized convergence, we introduce a structural assumption regarding the advantage function of the unregularized MDP. By basic MDP theory, the unregularized optimal value functions $V _ { 0 } ^ { * }$ and $Q _ { 0 } ^ { * }$ are unique. We assume there exists a positive minimal action gap separating the best action from all suboptimal actions at every state. This is a standard regularity condition in tabular MDPs but may be violated in non-tabular MDPs. Due to this positive margin, there is a unique optimal action at each state. This uniqueness facilitates the derivation of the simple policy entropy bound (Lemma 3.4) used in the proof of Theorem 3.1.

Assumption 5 (Minimal Action Gap). There exists a positive minimal action gap $\Delta > 0$ such that for all states $s \in { \mathcal { S } }$ , there is a unique optimal action $a ^ { * } ( s ) = \arg \operatorname* { m a x } _ { a \in \mathcal { A } } Q _ { 0 } ^ { * } ( s , a )$ , and for all suboptimal actions $a \neq a ^ { * } ( s )$ , the unregularized advantage is bounded from below by $\Delta$

$$
V _ { 0 } ^ { * } ( s ) - Q _ { 0 } ^ { * } ( s , a ) \geq \Delta .
$$

Consequently, there exists a unique unregularized optimal policy $\pi _ { 0 } ^ { * }$ that maximizes the value function at every state. This policy is deterministic and satisfies $\pi _ { 0 } ^ { * } ( a ^ { * } ( s ) | s ) = 1$ for all $s \in { \mathcal { S } }$

## 3 Suboptimality Identities and Translation Bounds

## 3.1 Suboptimality Identities and Decompositions

In this section, we derive the suboptimality identities and decompositions that govern algorithmic progress. First, we present the standard performance diference lemma, adapted for the entropyregularized objective (Kakade and Langford, 2002; Mei et al., 2020; Cayci et al., 2024).

Lemma 3.1 (Regularized Performance Diference). For any two policies π and $\tilde { \pi } _ { }$ , the diference in their regularized objectives is given by:

$$
J _ { \tau } ( \tilde { \pi } ) - J _ { \tau } ( \pi ) = \frac { 1 } { 1 - \gamma } \mathbb { E } _ { s \sim d ^ { \tilde { \pi } } } \left[ \sum _ { a \in A } \tilde { \pi } ( a | s ) \big ( Q _ { \tau } ^ { \pi } ( s , a ) - \tau \log { \tilde { \pi } ( a | s ) } \big ) - V _ { \tau } ^ { \pi } ( s ) \right] .
$$

Next, we present a geometric identity relating the regularized performance gap of π against the regularized global optimum $\pi _ { \tau } ^ { * }$ to the state-averaged reverse KL divergence $\mathbb { E } _ { s \sim d ^ { \pi } } [ D _ { K L } ( \pi \| \pi _ { \tau } ^ { * } ) ]$ ], as in Mei et al. (2020, Lemma 26). For any two policies π and ${ \tilde { \pi } } ,$ the KL divergence at state s is defined as $\begin{array} { r } { D _ { K L } ( \tilde { \pi } ( \cdot | s ) \| \pi ( \cdot | s ) ) \triangleq \sum _ { a \in \mathcal { A } } \tilde { \pi } ( a | s ) \log \frac { \tilde { \pi } ( a | s ) } { \pi ( a | s ) } } \end{array}$ . Recall that $\pi _ { \tau } ^ { * }$ is the unique soft Bellman-optimal policy for the regularized MDP. Because this identity relies on the statewise optimality of $\pi _ { \tau } ^ { * }$ , it would not in general hold if $\pi _ { \tau } ^ { * }$ is replaced by the parametric optimum $\pi _ { \omega _ { \tau } ^ { * } }$

Lemma 3.2 (Global Regularized Suboptimality Identity). For any policy π, the performance gap between $\pi$ and the regularized global optimum $\pi _ { \tau } ^ { * }$ is proportional to the state-averaged reverse KL divergence under the state visitation distribution $d ^ { \pi }$ of policy π:

$$
J _ { \tau } ( \pi _ { \tau } ^ { * } ) - J _ { \tau } ( \pi ) = \frac { \tau } { 1 - \gamma } \mathbb { E } _ { s \sim d ^ { \pi } } \big [ D _ { K L } ( \pi ( \cdot | s ) \| \pi _ { \tau } ^ { * } ( \cdot | s ) ) \big ] .
$$

Finally, we prove useful decompositions of the regularized performance diference for a training policy $\pi _ { t }$ against any comparator policy π, such as $\pi _ { \omega _ { \tau } ^ { * } }$ or $\pi _ { \tau } ^ { * }$ . This lemma provides two distinct algebraic separations tailored to our two optimization regimes. The Ideal Critic Decomposition isolates the ideal algorithmic progress, which we bound in Part I to establish the Parameter-Space PL Condition (Lemma 4.2). Conversely, the Actual Critic Decomposition isolates the actor update direction alongside the critic tracking error. In Part II, we bound this algorithmic progress via the Bregman Three-Point identity (Lemma 8.1) and substitute it back into the Actual Critic Decomposition to establish the PMD Actor Progress Bound (Lemma 9.1). For notational clarity, for any state s, we define the inner product over the discrete action space between any two functions f and g as $\begin{array} { r } { \langle f , g \rangle _ { \mathcal { A } } \overset { \Delta } { = } \sum _ { a \in \mathcal { A } } f ( s , a ) g ( s , a ) } \end{array}$

Lemma 3.3 (Regularized Suboptimality Decompositions). For the log-linear policy parameterization, the regularized performance diference between a training policy $\pi _ { t }$ and a comparator policy π admits two exact decompositions, depending on whether it is evaluated using the ideal critic $\theta _ { \tau } ^ { * } ( \omega _ { t } )$ or the actual critic $\theta _ { t + 1 }$ :

(i) Ideal Critic Decomposition (Part I):

$$
\begin{array} { r l } & { ( 1 - \gamma ) \big ( J _ { \tau } ( \pi ) - J _ { \tau } ( \pi _ { t } ) \big ) = \underbrace { - \tau \mathbb { E } _ { d ^ { \pi } } \big [ D _ { K L } ( \pi \| \pi _ { t } ) \big ] } _ { R e s t o r a t i o n e ~ E n t r o p y ~ F o r c e } + \underbrace { \mathbb { E } _ { d ^ { \pi } } \Big [ \langle ( \theta _ { \tau } ^ { * } ( \omega _ { t } ) - \tau \omega _ { t } ) ^ { \top } \phi , \pi - \pi _ { t } \rangle _ { \mathcal { A } } \Big ] } _ { I d e a l ~ A l g o r i t h m i c ~ P r o g r e s s } } \\ & { \quad + \underbrace { \mathbb { E } _ { d ^ { \pi } } \big [ \langle \epsilon _ { t } , \pi - \pi _ { t } \rangle _ { \mathcal { A } } \big ] } _ { A p p r o x i m a t i o n ~ B i a s } . } \end{array}
$$

(ii) Actual Critic Decomposition (Part II):

$$
\begin{array} { r l } & { ( 1 - \gamma ) \big ( J _ { \tau } ( \pi ) - J _ { \tau } ( \pi _ { t } ) \big ) = - \tau \mathbb { E } _ { d ^ { \pi } } [ D _ { K L } ( \pi \| \pi _ { t } ) ] + \underbrace { { \mathbb { E } } _ { d ^ { \pi } } \Big [ \langle ( \theta _ { t + 1 } - \tau \omega _ { t } ) ^ { \top } \phi , \pi - \pi _ { t } \rangle _ { \mathcal { A } } \Big ] } _ { A l g o r i t h m i c ~ P r o g r e s s } } \\ & { ~ + \underbrace { { \mathbb { E } } _ { d ^ { \pi } } \Big [ \langle - e _ { t } ^ { \top } \phi , \pi - \pi _ { t } \rangle _ { \mathcal { A } } \Big ] } _ { C r i t i c ~ T r a c k i n g ~ E r r o r } + { \mathbb { E } } _ { d ^ { \pi } } \big [ \langle \epsilon _ { t } , \pi - \pi _ { t } \rangle _ { \mathcal { A } } \big ] . } \end{array}
$$

Remark 3.1 (Dual Perspectives of the Performance Diference Lemma). The proofs of Lemma 3.2 and Lemma 3.3 highlight an interesting diference in how the Regularized Performance Diference Lemma (Lemma 3.1) is applied. In Lemma 3.2, we evaluate the performance gap using the advantage function of the global optimum $\pi _ { \tau } ^ { * }$ , with expectations taken over the current policy’s state visitation distribution $( d ^ { \pi } )$ . Because the global optimum satisfies the soft Bellman equations, this “backward” application collapses the advantage terms, isolating the exact reverse KL divergence $D _ { K L } ( \pi \| \pi _ { \tau } ^ { * } )$ . In contrast, Lemma 3.3 applies the lemma in the standard “forward” direction. It evaluates the gap using the advantage function of the current policy $\pi ,$ taking expectations over the comparator policy’s state visitation distribution $( d ^ { \tilde { \pi } } )$ . This forward application exposes the inner product between the comparator’s action distribution and the current policy’s advantages, which naturally decomposes into the forward KL divergence and the algorithmic progress terms that drive both the Part I and Part II analyses.

## 3.2 Exponential Translation Bounds

In this section, we establish a bridge connecting the global regularized progress of a training policy to its ultimate unregularized performance under the Minimal Action Gap (Assumption 5). By evaluating the performance via the reverse KL divergence (as established in Lemma 3.2), we bound the global unregularized performance gap using the global regularized performance gap. Crucially, this yields an exponential translation bound that bypasses the linear $\mathcal { O } ( \tau )$ entropy penalty that traditionally bottlenecks unregularized bounds.

To execute this translation, we first present a lemma which bounds the Shannon entropy of any policy as a function of the probability mass it places on the suboptimal actions.

Lemma 3.4 (Policy Entropy Decomposition Bound). For any state s and policy $\pi ( \cdot | s )$ , let $a ^ { * } ( s )$ be an arbitrary designated action, and let $\begin{array} { r } { q ( s ) \triangleq \sum _ { a \neq a ^ { * } ( s ) } \pi ( a | s ) } \end{array}$ be the probability mass on all other actions. The pointwise Shannon entropy of the policy is bounded by:

$$
H ( \pi ( \cdot | s ) ) \leq - q ( s ) \log q ( s ) - ( 1 - q ( s ) ) \log ( 1 - q ( s ) ) + q ( s ) \log | { \cal A } | .
$$

Next, to facilitate application of Assumption 5, we provide a lemma which establishes a uniform bound on how much the regularized and unregularized optimal value functions deviate from one another, driven purely by the maximum possible entropy over the action space. Recall that $V _ { 0 } ^ { * }$ and $Q _ { 0 } ^ { \ast }$ denote the optimal state-value and action-value functions for the unregularized MDP, and $V _ { \tau } ^ { * }$ and $Q _ { \tau } ^ { * }$ denote the optimal value functions for the regularized MDP.

Lemma 3.5 (Uniform Regularized Value Bound). The regularized optimal values bound from above the unregularized optimal values, and the uniform approximation error is bounded by the maximum discounted entropy:

$$
0 \leq V _ { \tau } ^ { * } ( s ) - V _ { 0 } ^ { * } ( s ) \leq \frac { \tau \log | \mathcal { A } | } { 1 - \gamma } ,
$$

$$
0 \leq Q _ { \tau } ^ { * } ( s , a ) - Q _ { 0 } ^ { * } ( s , a ) \leq \frac { \tau \log | \mathcal { A } | } { 1 - \gamma } ,
$$

for all states $s \in { \mathcal { S } }$ and actions $a \in { \mathcal { A } }$

With these tools in place, we now establish the core translation mechanism. By exploiting the Minimal Action Gap for the unregularized MDP, we prove that for suficiently small temperatures, the global unregularized performance gap is bounded by the global regularized performance gap plus an exponentially small translation tail.

Theorem 3.1 (Exponential Translation of Regularized Suboptimality). Under Assumption 5, if the temperature is suficiently small such that $\begin{array} { r } { \tau \leq \frac { \Delta } { 2 \log C _ { \gamma } } } \end{array}$ , then the unregularized performance gap is bounded in terms of the state-averaged reverse KL divergence:

$$
J _ { 0 } ( \pi _ { 0 } ^ { * } ) - J _ { 0 } ( \pi _ { t } ) \leq \frac { A _ { m a x } } { 1 - \gamma } \left( \frac { 4 \tau } { \Delta } \mathbb { E } _ { d ^ { \pi _ { t } } } \big [ D _ { K L } ( \pi _ { t } ( \cdot | s ) | | \pi _ { \tau } ^ { * } ( \cdot | s ) ) \big ] + \sqrt { C _ { \gamma } } \exp \left( - \frac { \Delta } { 2 \tau } \right) \right) ,
$$

where $\begin{array} { r } { A _ { m a x } \triangleq \operatorname* { s u p } _ { s , a } ( V _ { 0 } ^ { * } ( s ) - Q _ { 0 } ^ { * } ( s , a ) ) } \end{array}$ is the maximum optimality gap, satisfying $\begin{array} { r } { A _ { m a x } \leq \frac { 1 } { 1 - \gamma } } \end{array}$ , and $C _ { \gamma } \triangleq e | \mathcal { A } | ^ { \frac { 3 - \gamma } { 1 - \gamma } }$ such that log $\begin{array} { r } { C _ { \gamma } = \frac { 2 \log | \mathcal { A } | } { 1 - \gamma } + \log | \mathcal { A } | + 1 } \end{array}$ . Equivalently, the unregularized performance gap is bounded in terms of the regularized performance gap $G a p _ { t } ^ { \dagger } \triangleq J _ { \tau } ( \pi _ { \tau } ^ { * } ) - J _ { \tau } ( \pi _ { t } )$ :

$$
J _ { 0 } ( \pi _ { 0 } ^ { * } ) - J _ { 0 } ( \pi _ { t } ) \leq \frac { 4 A _ { m a x } } { \Delta } G a p _ { t } ^ { \dagger } + \frac { A _ { m a x } \sqrt { C _ { \gamma } } } { 1 - \gamma } \exp \left( - \frac { \Delta } { 2 \tau } \right) .
$$

Theorem 3.1 requires the temperature to fall below a specific structural threshold. To ensure this translation mechanism can be safely applied even when the temperature might be high, we extend the exponential bound to hold for all $\tau > 0$

Corollary 3.1 (Universal Unregularized Suboptimality Bound). Under Assumption 5, the exponential translation bounds established in Theorem 3.1 can be extended to hold for all $\tau > 0$ Specifically, with the universal tail coeficient $\begin{array} { r } { C _ { t a i l } \triangleq \frac { A _ { m a x } C _ { \gamma } } { 1 - \gamma } } \end{array}$ , the unregularized performance gap is bounded in terms of the state-averaged reverse KL divergence for all $\tau > 0$

$$
J _ { 0 } ( \pi _ { 0 } ^ { * } ) - J _ { 0 } ( \pi _ { t } ) \leq \frac { 4 A _ { m a x } \tau } { \Delta ( 1 - \gamma ) } \mathbb { E } _ { d ^ { \pi _ { t } } } \big [ D _ { K L } ( \pi _ { t } ( \cdot | s ) | | \pi _ { \tau } ^ { * } ( \cdot | s ) ) \big ] + C _ { t a i l } \exp \left( - \frac { \Delta } { 2 \tau } \right) .
$$

Equivalently, it is bounded in terms of the regularized performance gap $G a p _ { t } ^ { \dagger }$ for all $\tau > 0$

$$
J _ { 0 } ( \pi _ { 0 } ^ { * } ) - J _ { 0 } ( \pi _ { t } ) \leq \frac { 4 A _ { m a x } } { \Delta } G a p _ { t } ^ { \dagger } + C _ { t a i l } \exp \left( - \frac { \Delta } { 2 \tau } \right) .
$$

## Part I Stochastic Regime

In the first part of our analysis, we investigate the convergence of Algorithm 1 in the Stochastic Regime. As described in Section 2, this regime is characterized by the sequence of regularized optimal policies $\pi _ { \omega _ { \tau } ^ { * } }$ converging to an unregularized parametric optimum $\pi _ { \omega _ { 0 } ^ { * } }$ with $\omega _ { \tau } ^ { * }  \omega _ { 0 } ^ { * } \in \mathbb { R } ^ { d }$ as the temperature drops $( \tau  0 )$ . This behavior can naturally arise, for example, when the restricted log-linear family lacks the expressivity to represent the deterministic global optimum $\pi _ { 0 } ^ { * } .$ , and the best available parametric approximation leads to stochastic action selection. Under Bounded Features (Assumption 1), for suficiently small $\tau _ { : }$ , the distributions $\pi _ { \omega _ { \tau } ^ { * } }$ remain uniformly interior to the probability simplex, so that the action randomness does not vanish.

We formally capture this persistent stochasticity and rule out redundant or action-independent feature directions through the expected centered feature covariance matrix, defined for any policy π as $\bar { \Sigma } _ { c e n } ( \pi ) \triangleq \mathbb { E } _ { s \sim d ^ { \pi } } \left[ \operatorname { C o v } _ { a \sim \pi ( \cdot \vert s ) } ( \phi ( s , a ) ) \right]$ . This matrix corresponds to the Fisher information matrix evaluated over $d ^ { \pi }$ . The positive-definiteness of this matrix at the parametric optimum $\pi _ { \omega _ { \tau } ^ { * } }$ provides the essential curvature which, when combined with the smoothness of the regularized objective, allows us to establish the Actor Progress Bound (Lemma 5.3). This progress bound serves a dual purpose in our joint convergence analysis: it is used directly to control the target drift in the Coupled Critic Tracking (Lemma 5.5), and, when combined with a Parameter-Space PL Condition (Lemma 4.2), it establishes a Coupled Actor Recurrence (Lemma 5.4) over the performance gap.

Assumption 6 (Positive-Definite Fisher Information). Define the temperature-dependent minimum Fisher eigenvalue at the parametric optimum as $\lambda _ { \tau } \triangleq \lambda _ { \operatorname* { m i n } } \left( \bar { \Sigma } _ { c e n } ( \pi _ { \omega _ { \tau } ^ { * } } ) \right)$ , where $\bar { \Sigma } _ { c e n } ( \pi _ { \omega _ { \tau } ^ { * } } ) =$ $\mathbb { E } _ { s \sim d ^ { \omega _ { \tau } ^ { * } } } \left[ \operatorname { C o v } _ { a \sim \pi _ { \omega _ { \tau } ^ { * } } ( \cdot | s ) } ( \phi ( s , a ) ) \right]$ . There exists a constant $\lambda > 0$ such that $\lambda _ { \tau } \geq \lambda$ for any $\tau \in ( 0 , \tau _ { m a x } ]$ Equivalently, for any $\tau \in ( 0 , \tau _ { m a x } ]$ 2

$$
\bar { \Sigma } _ { c e n } ( \pi _ { \omega _ { \tau } ^ { * } } ) = \mathbb { E } _ { s \sim d ^ { \omega _ { \tau } ^ { * } } } \left[ \operatorname { C o v } _ { a \sim \pi _ { \omega _ { \tau } ^ { * } } } ( \phi ( s , a ) ) \right] \succeq \lambda I .
$$

For completeness, we note that the Fisher information $\bar { \Sigma } _ { c e n } ( \pi _ { \tau } ^ { * } )$ at the global optimum $\pi _ { \tau } ^ { * }$ cannot remain uniformly positive-definite as $\tau  0$ under the Minimal Action Gap assumption. As $\tau \to 0 , \pi _ { \tau } ^ { * }$ collapses to the deterministic unregularized optimum $\pi _ { 0 } ^ { * }$ , causing the action randomness to vanish and the Fisher information matrix to degenerate. Thus, uniform positive-definiteness (Assumption 6) is only viable for the parametric optimum $\pi _ { \omega _ { \tau } ^ { * } }$ even in the Stochastic Regime.

To connect distributional properties of the parametric optimum $\pi _ { \omega _ { \tau } ^ { * } }$ to the trajectory of policies $\{ \pi _ { t } \} _ { t \ge 0 }$ generated during training, we require that, under Algorithm 1, suficient coverage is maintained over the target state-action visitation measure.

Assumption 7 (Parametric Joint Concentrability). Under Algorithm 1, the joint state-action density ratio of the regularized parametric optimum $\pi _ { \omega _ { \tau } ^ { * } }$ with respect to the training policy $\pi _ { t }$ is

bounded for all $t \geq 0$

$$
\operatorname* { s u p } _ { s , a } \frac { d ^ { \omega _ { \tau } ^ { * } } ( s ) \pi _ { \omega _ { \tau } ^ { * } } ( a | s ) } { d ^ { \pi _ { t } } ( s ) \pi _ { t } ( a | s ) } \leq C _ { j o i n t } ,
$$

where $C _ { j o i n t } > 0$ is a constant independent of the temperature $\tau \in ( 0 , \tau _ { m a x } ]$

We make several comments on the two preceding assumptions. First, the joint concentrability bound is naturally implied by the conjunction of separate state and policy concentrability bounds. Specifically, if the following holds for some constants (C, W) independent of $\tau \in ( 0 , \tau _ { m a x } ]$ :

• Parametric State Concentrability: $\| d ^ { \omega _ { \tau } ^ { * } } / d ^ { \pi _ { t } } \| _ { \infty } \leq C .$

• Parametric Policy Concentrability: $\begin{array} { r } { \operatorname* { s u p } _ { s , a } \frac { \pi _ { \omega _ { \tau } ^ { * } } ( a | s ) } { \pi _ { t } ( a | s ) } \leq W . } \end{array}$

then by the multiplicative rule of density ratios, the joint concentrability bound is satisfied with $C _ { j o i n t } = C W$ . Conversely, because $\begin{array} { r } { \mathbb { E } _ { a \sim \pi _ { t } } \Big [ \frac { \pi _ { \omega _ { \tau } ^ { * } } ( a | s ) } { \pi _ { t } ( a | s ) } \Big ] = 1 } \end{array}$ at each state s, the joint concentrability directly implies the state concentrability bound with $C = C _ { j o i n t } \colon \| d ^ { \omega _ { \tau } ^ { * } } / d ^ { \pi _ { t } } \| _ { \infty } \leq C _ { j o i n t }$ . However, the joint formulation allows the local policy ratio $\frac { \pi _ { \omega _ { \tau } ^ { * } } ( a | s ) } { \pi _ { t } ( a | s ) }$ to be unbounded so long as the associated state density ratio $\frac { d ^ { \omega _ { \tau } ^ { * } } ( s ) } { d ^ { \pi _ { t } } ( s ) }$ is suficiently small.

Second, for analysis of entropy-regularized MDPs, some prior works (Mei et al., 2020; Cayci et al., 2024) combine the state concentrability C with a minimum action probability condition: $\begin{array} { r } { \operatorname* { i n f } _ { t , s , a } \pi _ { t } ( a | s ) \geq c _ { \tau } } \end{array}$ for a constant $c _ { \tau } > 0$ depending on temperature τ. This condition directly implies a policy ratio bound of $\frac { \pi _ { \omega _ { \tau } ^ { * } } ( a | s ) } { \pi _ { t } ( a | s ) } \leq \frac { 1 } { c _ { \tau } }$ . Similar to Cayci et al. (2024, Lemma 3.2), the actor iterate bound (Lemma 2.2) leads to a τ -dependent bound: $\begin{array} { r } { c _ { \tau } = \frac { 1 } { | \mathcal { A } | } \exp \left( - \frac { 2 B _ { \theta } B _ { \phi } } { \tau } \right) } \end{array}$ . However, combining C with $\begin{array} { r } { W = \frac { 1 } { c _ { \tau } } } \end{array}$ and relying on $C _ { j o i n t } = C W$ would introduce an exponential τ -dependence into our structural constants as $\tau  0$ . Alternatively, a policy concentrability bound W independent of τ could be satisfied by requiring $\begin{array} { r } { \operatorname* { i n f } _ { t , s , a } \pi _ { t } ( a | s ) \geq c } \end{array}$ for a constant $c > 0$ independent of τ. While such a uniform lower bound on $\pi _ { t } ( a | s )$ is conceptually consistent with the persistent stochasticity that characterizes the Stochastic Regime, our joint concentrability formulation acts as a crucial relaxation, allowing even unbounded local policy ratios as $\tau  0$

Third, Assumption 7 is utilized in exactly two places in our Part I analysis. It is first used in the proof of the Parameter-Space PL Condition (Lemma 4.2) to facilitate the action-space $\chi ^ { 2 } \mathrm { - t o - K L }$ divergence translation while simultaneously shifting the state measure. It is then used in the proof of the Actor Progress Bound (Lemma 5.3) in conjunction with the Positive-Definite Fisher Information (Assumption 6). Notably, this is the only place where Assumption 6 is invoked. While Assumption 6 ensures the uniform positive-definiteness of the Fisher information matrix under the parametric optimum $\pi _ { \omega _ { \tau } ^ { * } }$ , the joint concentrability bound dynamically extends this geometric stability to all generated policies. As shown in Step (iii) of the proof for Lemma 5.3, the centered covariance for any training policy $\pi _ { t }$ satisfies $\begin{array} { r } { \bar { \Sigma } _ { c e n } ( \pi _ { t } ) \succeq \frac { 1 } { C _ { j o i n t } } \bar { \Sigma } _ { c e n } ( \pi _ { \omega _ { \tau } ^ { * } } ) } \end{array}$ , which implies $\begin{array} { r } { \bar { \Sigma } _ { c e n } ( { \pi } _ { t } ) \succeq \frac { { \lambda } } { { { C } _ { j o i n t } } } I . } \end{array}$ . Note that because $\bar { \Sigma } _ { u n c } ( \pi _ { t } ) \succeq \bar { \Sigma } _ { c e n } ( \pi _ { t } )$ , the Positive-Definite Uncentered Feature Moment (Assumption 3) is then satisfied by taking $\begin{array} { r } { \kappa = \frac { \lambda } { C _ { j o i n t } } } \end{array}$ in the Stochastic Regime. Nevertheless, for clarity, we keep track of the dependence on κ and λ separately in Part I.

Connections to Related Works. Assumptions 6 and 7 adapt standard regularity and concentrability requirements to the parametric setting. First, a positive-definite Fisher information matrix is required in several analyses (Liu et al., 2020; Wang et al., 2024). As discussed in $\mathrm { A p \mathrm { - } }$ pendix D.3, Liu et al. (2020) require this matrix to be uniformly positive-definite for all training policies, whereas Assumption 6 localizes this requirement to the parametric optimum $\pi _ { \omega _ { \tau } ^ { * } }$ and then dynamically extends the property to all training policies via Assumption 7. Second, joint concentrability bounds are implicitly or explicitly used in several analyses (Liu et al., 2020; Wang et al., 2024). For example, Liu et al. (2020) implicitly bundle a global counterpart of this requirement into a transferred approximation error bound evaluated at the unregularized global optimum. In contrast, our analysis utilizes the parametric joint concentrability to transfer the on-policy approximation error to that under the parametric optimum. Finally, in the on-policy case $( \nu = \mu \times \pi _ { t } )$ , the concentrability assumption in Agarwal et al. (2021) corresponds to the unregularized counterpart of our parametric joint concentrability, when the comparator policy is set to the parametric optimum. Note that in Appendix D.1, this is discussed when the comparator policy is set to the global optimum, to relate their assumption to our global joint concentrability (Assumption 9).

Finally, we point out an intrinsic constraint in the Stochastic Regime under the stated assumptions, particularly the conjunction of the Positive-Definite Fisher Information and the Minimal Action Gap. As $\tau  0 .$ , the Minimal Action Gap drives the regularized global optimum $\pi _ { \tau } ^ { * }$ toward the unregularized deterministic optimum $\pi _ { 0 } ^ { * }$ , whereas the Fisher lower bound requires the regularized parametric optimum $\pi _ { \omega _ { \tau } ^ { * } }$ to retain nonvanishing action randomness. At the same time, the approximation-error assumption, together with the Fisher lower bound and the global-to-parametric joint concentrability, controls the regularized gap between $\pi _ { \tau } ^ { * }$ and $\pi _ { \omega _ { \tau } ^ { * } }$ by a term proportional to $\epsilon _ { a p p } / \tau$ (Lemma 7.1). These properties jointly imply that $\epsilon _ { a p p }$ cannot be arbitrarily small independently of $\tau$ under the Part-I assumptions. This is formalized in Lemma 7.2,

## 4 Parameter-Space Polyak- Lojasiewicz Condition

In this section, we establish a fundamental geometric property that contributes to algorithmic convergence in the Stochastic Regime: the parameter-space Polyak- Lojasiewicz (PL) condition. Unlike standard PL conditions that bound suboptimality via objective gradients, our formulation operates within the parameter space. It bounds the parametric regularized performance $\mathrm { g a p } \left( \mathrm { G a p } _ { t } \right)$ via the Euclidean distance between the temperature-scaled actor parameter $( \tau \omega _ { t } )$ and the ideal uncentered critic target $\left( \theta _ { \tau } ^ { * } ( \omega _ { t } ) \right)$ ).

We first introduce a general statistical inequality. Under bounded density ratios as in Assumption 7, the following lemma bounds the $\chi ^ { 2 }$ divergence by the KL divergence, providing the central mechanism to control action-space distribution shifts.

Lemma 4.1 $( \chi ^ { 2 }$ to KL Divergence Bound). For any two probability measures P and $Q$ over a probability space X , if the density ratio is bounded such that $\begin{array} { r } { \operatorname* { s u p } _ { x \in \mathcal { X } } \frac { \mathrm { d } P } { \mathrm { d } Q } ( x ) \leq M } \end{array}$ , then:

$$
\chi ^ { 2 } ( P \| Q ) \leq 2 M D _ { K L } ( P \| Q ) ,
$$

where $\begin{array} { r } { \chi ^ { 2 } ( P \| Q ) \triangleq \int _ { \mathcal { X } } \left( \frac { \mathrm { d } P } { \mathrm { d } Q } - 1 \right) ^ { 2 } \mathrm { d } Q ~ a n d ~ D _ { K L } ( P \| Q ) \triangleq \int _ { \mathcal { X } } \log \left( \frac { \mathrm { d } P } { \mathrm { d } Q } \right) \mathrm { d } P . } \end{array}$

With this tool in place, we now establish the PL condition. We apply the Ideal Critic Decomposition from Lemma 3.3(i) to the parametric optimum $\pi = \pi _ { \omega _ { \tau } ^ { * } }$ . After bounding the Ideal Algorithmic Progress using H¨older’s and Pinsker’s inequalities, and the Approximation Bias via the Cauchy-Schwarz inequality alongside the $\chi ^ { 2 } \mathrm { - t o - K L }$ bound, we apply Young’s inequality to decouple the terms. This yields positive penalty terms that ofset the Restorative Entropy Force, leading to the desired parameter-space bound. Throughout, we denote the state-averaged forward KL divergence from this parametric optimum $\pi _ { \omega _ { \tau } ^ { * } }$ as $D _ { t } \triangleq \mathbb { E } _ { d ^ { \omega _ { \tau } ^ { * } } } [ D _ { K L } ( \pi _ { \omega _ { \tau } ^ { * } } \| \pi _ { t } ) ]$

Lemma 4.2 (Parameter-Space PL Condition). Under Assumptions 1, 4, and 7, the parametric regularized performance gap satisfies the PL condition, penalized by the MSE of the approximation error:

$$
G a p _ { t } \leq \frac { C _ { P L } } { \tau } \| \theta _ { \tau } ^ { * } ( \omega _ { t } ) - \tau \omega _ { t } \| _ { 2 } ^ { 2 } + \frac { C _ { e r r } } { \tau } \epsilon _ { a p p } ,
$$

where $\begin{array} { r } { C _ { P L } \triangleq \frac { B _ { \phi } ^ { 2 } } { 1 - \gamma } } \end{array}$ and $\begin{array} { r } { C _ { e r r } \triangleq \frac { C _ { j o i n t } } { 1 - \gamma } } \end{array}$

Remark 4.1 (Comparison with Standard PL Conditions). In classical optimization, the standard PL condition bounds the suboptimality via the Euclidean gradient norm. When combined with objective smoothness, the PL condition guarantees linear convergence. In the context of tabular

MDPs with a softmax parameterization, Mei et al. (2020) established two distinct Euclidean PL conditions: first for unregularized MDPs (which requires a suficient exploration assumption) and second for entropy-regularized MDPs. Specifically, for the regularized MDP evaluated with identical initial distributions for the objective and training $( \mu = \rho )$ , their PL condition (Eq. 27) translated to our notation is:

$$
J _ { \tau } ( \pi _ { \tau } ^ { * } ) - J _ { \tau } ( \pi _ { \omega } ) \leq \frac { | \mathcal { S } | \cdot \| d ^ { \pi _ { \tau } ^ { * } } / d ^ { \pi _ { \omega } } \| _ { \infty } } { 2 \tau \operatorname* { m i n } _ { s } \mu ( s ) \operatorname* { m i n } _ { s , a } \pi _ { \omega } ( a | s ) ^ { 2 } } \| \nabla _ { \omega } J _ { \tau } ( \omega ) \| _ { 2 } ^ { 2 } .
$$

This coeficient scales with the state-space size |S| and the state distribution mismatch, and requires positive initial state coverage $( \operatorname* { m i n } _ { s } \mu ( s ) > 0 )$

In contrast, Lemma 4.2 establishes a parameter-space PL condition tailored for compatible function approximation. Instead of the objective gradient, we bound the suboptimality directly via the Euclidean distance between the τ -scaled actor iterate and the ideal critic target: $\lVert \theta _ { \tau } ^ { * } ( \omega _ { t } ) - \tau \omega _ { t } \rVert _ { 2 } ^ { 2 }$ Under exact parameterization $( \epsilon _ { a p p } = 0 )$ , our PL condition reduces to $\begin{array} { r } { \mathrm { G a p } _ { t } \leq \frac { B _ { \phi } ^ { 2 } } { ( 1 - \gamma ) \tau } \Vert \theta _ { \tau } ^ { * } ( \omega _ { t } ) - \tau \omega _ { t } \Vert _ { 2 } ^ { 2 } } \end{array}$ eliminating the dependencies on $\begin{array} { r } { | \boldsymbol { S } | , \operatorname* { m i n } _ { s } \mu ( s ) , \operatorname* { m i n } _ { s , a } \pi _ { \omega } ( a | s ) } \end{array}$ , and $\| d ^ { \pi _ { \tau } ^ { * } } / d ^ { \pi _ { \omega } } \| _ { \infty }$

## 5 Actor Progress and Critic Tracking

In this section, we analyze the actor and critic updates of Algorithm 1. First, by combining the global smoothness of the regularized objective with the Positive-Definite Fisher Information assumption, we establish the Actor Progress Bound (Lemma 5.3), bounding the step-by-step reduction in the parametric performance gap up to the uncentered bias and critic tracking penalties. Next, we demonstrate how this progress bound and the parameter-space PL condition establish a fundamental two-way coupling: the critic’s tracking error penalizes the Coupled Actor Recurrence (Lemma 5.4), while the actor’s reduction in the performance gap dynamically bounds the target drift in the Coupled Critic Tracking (Lemma 5.5). This bidirectional dynamic lays the groundwork for the joint convergence analyzed in Section 6.

## 5.1 Uncentered Gradients and Objective Smoothness

A notable characteristic of our algorithm is that the ideal critic $\theta _ { \tau } ^ { * } ( \omega _ { t } )$ is trained on uncentered features $\phi ( s , a )$ . Because of this uncentered representation, mapping the resulting parameter update back to the exact policy gradient introduces a structural bias. We first quantify this bias. By the first-order optimality condition for the ideal critic $\theta _ { \tau } ^ { * } ( \omega _ { t } )$ , the approximation error $\boldsymbol { \epsilon } _ { t } ( s , a )$ is orthogonal to the raw features under the training distribution:

$$
\mathbb { E } _ { s \sim d ^ { \pi _ { t } } , a \sim \pi _ { t } } \big [ \phi ( s , a ) \epsilon _ { t } ( s , a ) \big ] = 0 .\tag{3}
$$

Let $\tilde { \phi } _ { t } ( s , a ) \triangleq \phi ( s , a ) - \mathbb { E } _ { a ^ { \prime } \sim \pi _ { t } } [ \phi ( s , a ^ { \prime } ) ]$ denote the action-centered feature. Then the expected centered feature covariance under the training distribution can be expressed as $\bar { \Sigma } _ { c e n } ( \pi _ { t } ) ~ =$ $\mathbb { E } _ { d ^ { \pi _ { t } } , \pi _ { t } } \big [ \tilde { \phi } _ { t } ( s , a ) \tilde { \phi } _ { t } ( s , a ) ^ { \top } \big ]$

Lemma 5.1 (Uncentered Gradient Identity). The exact policy gradient evaluated using the ideal uncentered critic decomposes into a centered gradient vector and a structural bias vector:

$$
\nabla J _ { \tau } ( \omega _ { t } ) = \frac { 1 } { 1 - \gamma } \bar { \Sigma } _ { c e n } ( \pi _ { t } ) \big ( \theta _ { \tau } ^ { * } ( \omega _ { t } ) - \tau \omega _ { t } \big ) + \frac { 1 } { 1 - \gamma } E _ { b i a s } ,
$$

where the bias vector is defined by the feature mean $\bar { \phi } _ { t } ( s ) \triangleq \mathbb { E } _ { \pi _ { t } } [ \phi ( s , a ^ { \prime } ) ]$ and the residual error:

$$
{ \mathit { E } } _ { b i a s } \triangleq - \mathbb { E } _ { s \sim d ^ { \pi _ { t } } } \left[ { \bar { \phi } } _ { t } ( s ) \mathbb { E } _ { a \sim \pi _ { t } } [ \epsilon _ { t } ( s , a ) \mid s ] \right] .
$$

Under Assumption $^ { 4 , }$ its norm is bounded by the critic approximation error: $\lVert E _ { b i a s } \rVert _ { 2 } \leq B _ { \phi } \sqrt { \epsilon _ { a p p } } .$

Next, we establish that the entropy-regularized objective $J _ { \tau } ( \omega )$ is globally smooth with respect to the log-linear policy parameters. This smoothness property, combined with the uncentered gradient identity and the positive-definite Fisher information, is instrumental in obtaining the Actor Progress Bound for analyzing algorithmic convergence. We explicitly define a smoothness constant through the maximum temperature $\tau _ { m a x }$ to ensure asymptotic stability as $\tau  0$

Lemma 5.2 (Smoothness of the Regularized Objective). Under Assumption 1, the entropyregularized objective $J _ { \tau } ( \omega )$ is $\boldsymbol { L } _ { J , \tau }$ -smooth with respect to the parameters ω. Specifically, the smoothness constant is bounded by:

$$
L _ { J , \tau } \triangleq \frac { 1 6 B _ { \phi } ^ { 2 } ( 1 + \tau \log | \cal { A } | ) + 4 \tau B _ { \phi } ^ { 2 } } { ( 1 - \gamma ) ^ { 3 } } ,
$$

such that for any $\omega , \omega ^ { \prime } \in \mathbb { R } ^ { d }$ , we have $\| \nabla J _ { \tau } ( \omega ) - \nabla J _ { \tau } ( \omega ^ { \prime } ) \| _ { 2 } \leq L J _ { \tau } \| \omega - \omega ^ { \prime } \| _ { 2 }$ . Consequently, for any temperature $\tau \in ( 0 , \tau _ { m a x } ]$ , the regularized objective $J _ { \tau } ( \omega )$ is $L _ { J }$ -smooth, where $L _ { J } \triangleq L _ { J , \tau _ { m a x } }$

## 5.2 Actor Progress and Algorithmic Coupling

We now analyze the step-by-step progress of the actor parameter update. By linking the actor update direction to the policy gradient (Lemma 5.1), we substitute this relation into the quadratic smoothness bound (Lemma 5.2) and leverage the positive-definite Fisher information to establish a progress inequality: the actor parameter update decreases the parametric performance gap, ofset only by the critic tracking error and the uncentered approximation bias.

Lemma 5.3 (Actor Progress Bound). Let $Z _ { t } \triangleq \mathbb { E } [ \Vert e _ { t } \Vert _ { 2 } ^ { 2 } ]$ be the expected critic tracking error. Under Assumptions $1 , \ 4 , \ 6 ,$ and 7, if the step size satisfies $\begin{array} { r } { \eta _ { t } \leq \frac { \lambda } { 2 L _ { J } C _ { j o i n t } ( 1 - \gamma ) } } \end{array}$ , the actor’s update guarantees an expected progress inequality for the parametric gap:

$$
\begin{array} { r } { \mathbb { E } [ G a p _ { t + 1 } ] \leq \mathbb { E } [ G a p _ { t } ] - \eta _ { t } \rho _ { a c } \mathbb { E } [ \| \theta _ { t + 1 } - \tau \omega _ { t } \| _ { 2 } ^ { 2 } ] + \eta _ { t } C _ { Z } Z _ { t } + \eta _ { t } C _ { b i a s } \epsilon _ { a p p } , } \end{array}
$$

where the descent coeficient is $\begin{array} { r } { \rho _ { a c } = \frac { \lambda } { 8 C _ { j o i n t } ( 1 - \gamma ) } } \end{array}$ , the critic tracking penalty is $\begin{array} { r } { C _ { Z } = \frac { 2 B _ { \phi } ^ { 2 } } { 1 - \gamma } } \end{array}$ , and the uncentered bias penalty is $\begin{array} { r } { C _ { b i a s } = \frac { 2 C _ { j o i n t } B _ { \phi } ^ { 2 } } { \lambda ( 1 - \gamma ) } } \end{array}$

The progress bound in Lemma 5.3 relies on the expected parameter distance $\mathbb { E } [ \| \theta _ { t + 1 } - \tau \omega _ { t } \| _ { 2 } ^ { 2 } ]$ which is dificult to control directly over the entire trajectory. To resolve this, we invoke the Parameter-Space PL Condition (Lemma 4.2) to bridge the physical parameter update back to the performance gap. This yields a coupled linear recurrence where optimization progress is intrinsically tied to the critic’s tracking accuracy.

Lemma 5.4 (Coupled Actor Recurrence). Under Assumptions 1, 4, 6, and 7, if the step size satisfies $\begin{array} { r } { \eta _ { t } \leq \frac { \lambda } { 2 L _ { J } C _ { j o i n t } ( 1 - \gamma ) } } \end{array}$ , the parametric regularized performance gap satisfies an expected linear recurrence:

$$
\begin{array} { r } { \mathbb { E } [ G a p _ { t + 1 } ] \leq ( 1 - \rho _ { g a p } \eta _ { t } ) \mathbb { E } [ G a p _ { t } ] + \eta _ { t } \tilde { C } _ { Z } Z _ { t } + \eta _ { t } C _ { \epsilon , g a p } \epsilon _ { a p p } , } \end{array}
$$

where the efective recurrence parameter is $\begin{array} { r } { \rho _ { g a p } \ \triangleq \ \frac { \rho _ { a c } \tau } { 2 C _ { P L } } } \end{array}$ , the critic-tracking coupling is ${ \tilde { C } } _ { Z } \triangleq$ $\rho _ { a c } + C _ { Z }$ , and the total approximation penalty is $\begin{array} { r } { C _ { \epsilon , g a p } \triangleq \frac { \rho _ { a c } C _ { e r r } } { 2 C _ { P L } } + C _ { b i a s } } \end{array}$

## 5.3 Coupled Critic Tracking

We turn to analyzing the critic tracking error, $e _ { t } \triangleq \theta _ { t + 1 } - \theta _ { \tau } ^ { * } ( \omega _ { t } )$ . Because the actor is updated concurrently with the critic, the ideal target $\theta _ { \tau } ^ { * } ( \omega _ { t } )$ continuously shifts as the policy evolves. To bound this target drift, we invoke the Lipschitz continuity of the ideal critic (Lemma 2.3). Furthermore, we use the Actor Progress Bound (Lemma 5.3) to connect this target drift back to the objective performance gap. By enforcing a two-timescale separation $\left( \alpha _ { t + 1 } = \xi \eta _ { t } \right)$ , we combine the geometric contraction of the strongly convex critic update with this objective-coupled drift penalty to form a recursive bound on the critic tracking error.

Lemma 5.5 (Coupled Critic Tracking). Let the critic step size be $\alpha _ { t + 1 } = \xi \eta _ { t }$ for all $t \geq 0$ , with an arbitrary initialization α<sub>0</sub>. Under Assumptions $1 \mathrm { - } \mathit { 4 }$ and 6–7, if the actor step size satisfies $\begin{array} { r } { \eta _ { t } \leq \operatorname* { m i n } \left( \frac { \lambda } { 2 L _ { J } C _ { j o i n t } ( 1 - \gamma ) } , \frac { 1 } { 2 \xi \kappa } \right) } \end{array}$ , then the expected squared tracking error $Z _ { t } \triangleq \mathbb { E } [ \Vert e _ { t } \Vert _ { 2 } ^ { 2 } ] = \mathbb { E } [ \Vert \theta _ { t + 1 } -$

$\theta _ { \tau } ^ { * } ( \omega _ { t } ) \lVert _ { 2 } ^ { 2 } \rVert$ satisfies the coupled recurrence:

$$
Z _ { t + 1 } \leq ( 1 - \rho _ { c r } \eta _ { t } ) Z _ { t } + C _ { g a p } \big ( \mathbb { E } [ G a p _ { t } ] - \mathbb { E } [ G a p _ { t + 1 } ] \big ) + \xi ^ { 2 } \eta _ { t } ^ { 2 } G _ { c r } ^ { 2 } + \eta _ { t } C _ { \epsilon , c r } \epsilon _ { a p p } ,
$$

where the native recurrence rate is $\begin{array} { r } { \rho _ { c r } \triangleq \xi \kappa - \frac { L _ { \theta } ^ { 2 } C _ { Z } } { \xi \kappa \rho _ { a c } } } \end{array}$ , the objective coupling constant is $\begin{array} { r } { C _ { g a p } \triangleq \frac { L _ { \theta } ^ { 2 } } { \xi \kappa \rho _ { a c } } } \end{array}$ and the approximation penalty is $C _ { \epsilon , c r } \triangleq C _ { g a p } C _ { b i a s }$

## 6 Fixed-Temperature Convergence (Stochastic Regime)

We show that Algorithm 1 achieves an $\tilde { \mathcal { O } } ( T ^ { - 1 } )$ average-iterate and an $\mathcal { O } ( T ^ { - 1 } )$ last-iterate convergence rate for the regularized performance gap. This is accomplished by fusing the Coupled Actor Recurrence and Coupled Critic Tracking established in Section 5 into a joint Lyapunov function that drives a linear recurrence for the overall algorithm.

## 6.1 Joint Actor-Critic System

We review the main results from Section 5.2 and 5.3 to establish the joint system. We implement a two-timescale step size scheme: $\alpha _ { t + 1 } = \xi \eta _ { t }$ for a timescale separation constant $\xi \ge 1$ . By the Coupled Actor Recurrence (Lemma 5.4), the expected performance gap evolves according to:

$$
\begin{array} { r } { \mathbb { E } [ \mathrm { G a p } _ { t + 1 } ] \leq ( 1 - \rho _ { g a p } \eta _ { t } ) \mathbb { E } [ \mathrm { G a p } _ { t } ] + \eta _ { t } \tilde { C } _ { Z } Z _ { t } + \eta _ { t } C _ { \epsilon , g a p } \epsilon _ { a p p } , } \end{array}\tag{4}
$$

where $\begin{array} { r } { \rho _ { g a p } \triangleq \frac { \rho _ { a c } \tau } { 2 C _ { P L } } = \frac { \lambda \tau } { 1 6 C _ { j o i n t } B _ { \phi } ^ { 2 } } , \tilde { C } _ { Z } \triangleq \rho _ { a c } + C _ { Z } = \frac { \lambda } { 8 C _ { j o i n t } \left( 1 - \gamma \right) } + \frac { 2 B _ { \phi } ^ { 2 } } { 1 - \gamma } } \end{array}$ , and $\begin{array} { r } { C _ { \epsilon , g a p } \triangleq \frac { \rho _ { a c } C _ { e r r } } { 2 C _ { P L } } + C _ { b i a s } = } \end{array}$ $\begin{array} { r } { \frac { \lambda } { 1 6 B _ { \phi } ^ { 2 } ( 1 - \gamma ) } + \frac { 2 C _ { j o i n t } B _ { \phi } ^ { 2 } } { \lambda ( 1 - \gamma ) } } \end{array}$ . Simultaneously, by the Coupled Critic Tracking (Lemma 5.5), the critic’s expected tracking error $Z _ { t }$ evolves according to:

$$
\begin{array} { r } { Z _ { t + 1 } \leq ( 1 - \rho _ { c r } \eta _ { t } ) Z _ { t } + C _ { g a p } \big ( \mathbb { E } [ \mathrm { G a p } _ { t } ] - \mathbb { E } [ \mathrm { G a p } _ { t + 1 } ] \big ) + \xi ^ { 2 } \eta _ { t } ^ { 2 } G _ { c r } ^ { 2 } + \eta _ { t } C _ { \epsilon , c r } \epsilon _ { a p p } , } \end{array}\tag{5}
$$

where $\begin{array} { r } { \rho _ { c r } \triangleq \xi \kappa - \frac { L _ { \theta } ^ { 2 } C _ { Z } } { \xi \kappa \rho _ { a c } } = \xi \kappa - \frac { 1 6 C _ { j \alpha i n t } B _ { \theta } ^ { 2 } L _ { \theta } ^ { 2 } } { \xi \kappa \lambda } , C _ { g a p } \triangleq \frac { L _ { \theta } ^ { 2 } } { \xi \kappa \rho _ { a c } } = \frac { 8 C _ { j \alpha i n t } ( 1 - \gamma ) L _ { \theta } ^ { 2 } } { \xi \kappa \lambda } , C _ { \epsilon , c r } \triangleq C _ { g a p } C _ { b i a s } = } \end{array}$ $\frac { 1 6 C _ { j o i n t } ^ { 2 } B _ { \phi } ^ { 2 } L _ { \theta } ^ { 2 } } { \xi \kappa \lambda ^ { 2 } }$ , and $G _ { c r } ^ { 2 }$ is a constant bound on $\mathbb { E } [ \left| \left| \boldsymbol { g } _ { t } ^ { c r } \right| \right| _ { 2 } ^ { 2 } | \mathcal { F } _ { t } ]$ (Lemma 2.2).

We construct a joint Lyapunov function that linearly fuses the expected performance gap with the expected critic tracking error:

$$
\begin{array} { r } { \mathcal { L } _ { t } \triangleq \mathbb { E } [ \mathrm { G a p } _ { t } ] + \beta \big ( Z _ { t } + C _ { g a p } \mathbb { E } [ \mathrm { G a p } _ { t } ] \big ) , } \end{array}
$$

where $\beta > 0$ is a coupling constant.

Theorem 6.1 (Coupled Lyapunov Recurrence). Suppose that Assumptions $1 \mathrm { - } \mathit { 4 }$ and $ 6 { - } 7$ hold, and the step sizes satisfy the conditions established for the actor and critic updates (Lemmas $5 . 4$ and 5.5): $\eta _ { t } ~ \leq$ min $\begin{array} { r } { \left( \frac { \lambda } { 2 L _ { J } C _ { j o i n t } \left( 1 - \gamma \right) } , \frac { 1 } { 2 \xi \kappa } \right) } \end{array}$ and $\alpha _ { t + 1 } = \xi \eta _ { t }$ . By choosing a timescale ratio $\xi \ \ge$ $\begin{array} { r } { \frac { 1 } { \kappa } \left( 2 \rho _ { g a p } + 2 L _ { \theta } \sqrt { \frac { C _ { Z } } { \rho _ { a c } } } \right) } \end{array}$ and setting the Lyapunov coupling constant to $\begin{array} { r } { \beta \triangleq \frac { 2 { \tilde { C } _ { Z } } } { \rho _ { c r } } } \end{array}$ , the joint Lyapunov

function satisfies an expected linear recurrence:

$$
\begin{array} { r } { \mathcal { L } _ { t + 1 } \leq ( 1 - \rho _ { \mathcal { L } } \eta _ { t } ) \mathcal { L } _ { t } + \eta _ { t } ^ { 2 } \Sigma _ { \mathcal { L } } + \eta _ { t } C _ { \epsilon , \mathcal { L } } \epsilon _ { a p p } , } \end{array}
$$

where the joint rate is $\begin{array} { r } { \rho _ { \mathcal { L } } \triangleq \frac { \rho _ { g a p } } { 1 + \beta C _ { g a p } } } \end{array}$ , the joint variance is $\Sigma _ { \mathcal { L } } \ \triangleq \ \beta \xi ^ { 2 } G _ { c r } ^ { 2 }$ , and the joint bias is $C _ { \epsilon , \mathcal { L } } \triangleq C _ { \epsilon , g a p } + \beta C _ { \epsilon , c r }$

We point out that the algebraic recurrence in Theorem 6.1 is valid regardless of the sign of the multiplier $\left( 1 - \rho \_ { \mathit { l } } \eta _ { t } \right)$ . In the convergence guarantees later, the average-iterate analysis (Theorem 6.2) accommodates this recurrence via a telescoping sequence. In contrast, the last-iterate analysis (Theorem 6.3) requires the multiplier to be non-negative to ensure pointwise geometric contraction via Chung’s Lemma. This condition is explicitly satisfied by enforcing a suficiently large initial step ofset $t _ { 0 } \geq \rho _ { \mathcal { L } } c _ { \eta }$ in the step-size schedule.

Remark 6.1 (Structural Order Evaluation). To facilitate the convergence analysis, we explicitly trace the structural orders of the system’s composite constants with respect to the temperature $\tau ,$ the uncentered feature moment minimum eigenvalue κ, the Fisher information minimum eigenvalue $\lambda ,$ and the efective horizon $( 1 - \gamma ) ^ { - 1 }$ . Throughout this discussion, the stated orders refer to the structural constants and upper-envelope constants defined in the preceding analysis.

• Efective Gap Recurrence: The efective recurrence parameter scales linearly the temperature and this minimum eigenvalue: $\begin{array} { r } { \rho _ { g a p } = \frac { \lambda \tau } { 1 6 C _ { j o i n t } B _ { \phi } ^ { 2 } } = \Theta ( \lambda \tau ) } \end{array}$ . Because the horizon dependencies of $\rho _ { a c }$ and $C _ { P L }$ cancel, $\rho _ { g a p }$ is independent of $( 1 - \gamma )$

• Timescale Separation: The joint recurrence requires $\begin{array} { r } { \xi \ge \frac { 1 } { \kappa } \big ( 2 \rho _ { g a p } + 2 L _ { \theta } \sqrt { \frac { C _ { Z } } { \rho _ { a c } } } \big ) } \end{array}$ . From Lemma 2.3, the ideal critic’s Lipschitz constant evaluates to $L _ { \theta } = \Theta ( ( 1 - \gamma ) ^ { - 2 } \kappa ^ { - 2 } )$ . Because $C _ { Z } = \Theta ( ( 1 - \gamma ) ^ { - 1 } )$ and $\rho _ { a c } = \Theta ( ( 1 - \gamma ) ^ { - 1 } \lambda )$ , the target drift term $S \triangleq L _ { \theta } \sqrt { \frac { C _ { Z } } { \rho _ { a c } } }$ scales as $\Theta ( ( 1 - \gamma ) ^ { - 2 } \kappa ^ { - 2 } \lambda ^ { - 1 / 2 } )$ . Because $\rho _ { g a p } = \Theta ( \tau )$ , the $2 \rho _ { g a p }$ term is dominated by S. For concreteness, we set $\begin{array} { r } { \xi = \frac { c } { \kappa } ( 2 \rho _ { g a p } + 2 S ) } \end{array}$ for a constant $c \geq 1$ , yielding the temperature-independent order $\xi = \Theta ( ( 1 - \gamma ) ^ { - 2 } \kappa ^ { - 3 } \lambda ^ { - 1 / 2 } )$

• Critic Recurrence: $ { \mathrm { A s } } \ \tau \to 0$ , the gap recurrence term vanishes $( \rho _ { g a p }  0 )$ . By setting our timescale ratio to $\begin{array} { r } { \xi = \frac { c } { \kappa } ( 2 \rho _ { g a p } + 2 S ) } \end{array}$ , we obtain $\kappa \xi  2 c S$ . Substituting this limit into the native critic recurrence yields $\rho _ { c r }  2 c S - { \textstyle \frac { S ^ { 2 } } { 2 c S } } = ( 2 c - { \textstyle \frac { 1 } { 2 c } } ) S$ . Since $c \geq 1$ , this guarantees $\rho _ { c r } = \Theta ( ( 1 - \gamma ) ^ { - 2 } \kappa ^ { - 2 } \lambda ^ { - 1 / 2 } )$ , safely bounded away from zero.

• Coupling Constants: Because $\rho _ { c r } = \Theta ( ( 1 - \gamma ) ^ { - 2 } \kappa ^ { - 2 } \lambda ^ { - 1 / 2 } )$ and $\tilde { C } _ { Z } = \Theta ( ( 1 - \gamma ) ^ { - 1 } )$ , the required Lyapunov weight is $\begin{array} { r } { \beta = \frac { 2 \tilde { C } _ { Z } } { \rho _ { c r } } = \Theta ( ( 1 - \gamma ) \kappa ^ { 2 } \lambda ^ { 1 / 2 } ) } \end{array}$ . Substituting $L _ { \theta }$ and $\xi ,$ the objective coupling evaluates to $\begin{array} { r } { C _ { g a p } = \frac { 8 C _ { j o i n t } ( 1 - \gamma ) L _ { \theta } ^ { 2 } } { \xi \kappa \lambda } = \Theta ( ( 1 - \gamma ) ^ { - 1 } \kappa ^ { - 2 } \lambda ^ { - 1 / 2 } ) } \end{array}$ . The horizon dependencies cancel, yielding a structural invariant: $\beta C _ { g a p } = \Theta ( 1 )$

• Variance and Bias: The critic-gradient second moment is upper-bounded by the envelope $G _ { c r } ^ { 2 } \ : = \ : 2 B _ { \phi } ^ { 2 } ( B _ { \theta } ^ { 2 } B _ { \phi } ^ { 2 } \ : + \ : Q _ { m a x } ^ { 2 } )$ . Because $B _ { \theta } = \Theta ( ( 1 - \gamma ) ^ { - 1 } \kappa ^ { - 1 } )$ and $Q _ { m a x } = \mathcal { O } ( ( 1 - \gamma ) ^ { - 1 } )$ , this envelope satisfies $G _ { c r } ^ { 2 } = \Theta ( ( 1 - \gamma ) ^ { - 2 } \kappa ^ { - 2 } )$ . Thus, the joint tracking variance evaluates to $\Sigma _ { \mathcal { L } } = \beta \xi ^ { 2 } G _ { c r } ^ { 2 } = \Theta ( ( 1 - \gamma ) ^ { - 5 } \kappa ^ { - 6 } \lambda ^ { - 1 / 2 } )$ . The total approximation penalty evaluates to $\begin{array} { r } { C _ { \epsilon , g a p } = \frac { \lambda } { 1 6 B _ { \phi } ^ { 2 } ( 1 - \gamma ) } + C _ { b i a s } } \end{array}$ . The first term scaling as $\Theta ( ( 1 - \gamma ) ^ { - 1 } \lambda )$ is dominated by the second term $\begin{array} { r } { C _ { b i a s } = \frac { 2 C _ { j o i n t } B _ { \phi } ^ { 2 } } { \lambda ( 1 - \gamma ) } } \end{array}$ , scaling as $\Theta ( ( 1 - \gamma ) ^ { - 1 } \lambda ^ { - 1 } )$ . Thus, $C _ { \epsilon , g a p } = \Theta ( ( 1 - \gamma ) ^ { - 1 } \lambda ^ { - 1 } )$ . Because $\beta C _ { g a p } = \Theta ( 1 )$ , the joint bias evaluates to $C _ { \epsilon , \mathcal { L } } = C _ { \epsilon , g a p } + \beta C _ { g a p } C _ { b i a s } = \Theta ( ( 1 - \gamma ) ^ { - 1 } \lambda ^ { - 1 } )$

• Joint Recurrence Rate: Because $\beta C _ { g a p } = \Theta ( 1 )$ , the denominator $( 1 + \beta C _ { g a p } )$ reduces to a constant multiplier. The joint Lyapunov recurrence rate preserves the native actor’s efective recurrence rate: $\begin{array} { r } { \rho _ { \mathcal { L } } = \frac { \rho _ { g a p } } { 1 + \beta C _ { g a p } } = \Theta ( \lambda \tau ) } \end{array}$

## 6.2 Average-Iterate Regularized Convergence

Based on the joint Lyapunov recurrence established in Section 6.1, we show that Algorithm 1 achieves an $\tilde { \mathcal { O } } ( T ^ { - 1 } )$ average-iterate convergence to the regularized parametric optimum (up to the approximation bias) by deploying a diminishing Robbins-Monro step-size schedule. We first establish a general telescoping bound for the resulting average sequence.

Lemma 6.1 (Telescoping Robbins-Monro Sequence). Let $\{ x _ { t } \} _ { t = 0 } ^ { \infty }$ be a non-negative sequence satisfying the recursive bound:

$$
x _ { t + 1 } \leq ( 1 - r \eta _ { t } ) x _ { t } + a \eta _ { t } ^ { 2 } + b \eta _ { t } , \quad t \geq 0 ,
$$

where $r > 0 , a > 0$ , and $b \geq 0$ . Assume the diminishing step size is $\begin{array} { r } { \eta _ { t } = \frac { c _ { \eta } } { t + t _ { 0 } } } \end{array}$ , with an initial ofset $t _ { 0 } \geq 1$ and a numerator chosen such that $c \triangleq r c _ { \eta } > 1$ . Then, the average sequence is bounded by:

$$
\frac { 1 } { T } \sum _ { t = 0 } ^ { T - 1 } x _ { t } \le \frac { t _ { 0 } - 1 } { T ( c - 1 ) } x _ { 0 } + \frac { a c _ { \eta } ^ { 2 } } { c - 1 } \left( \frac { 1 + \log T } { T } \right) + \frac { b c _ { \eta } } { c - 1 } .
$$

Theorem 6.2 (Average-Iterate Regularized Convergence). For Algorithm 1, suppose that Assumptions 1–4 and 6–7 hold. Let the actor step size be $\begin{array} { r } { \eta _ { t } = \frac { c _ { \eta } } { t + t _ { 0 } } } \end{array}$ , with a schedule numerator $\begin{array} { r } { c _ { \eta } > \frac { 1 } { \rho _ { \mathcal { L } } } } \end{array}$ and an initial ofset $\begin{array} { r } { t _ { 0 } \geq \operatorname* { m a x } \left( 1 , \frac { c _ { \eta } } { \eta _ { m a x } } \right) } \end{array}$ , and let the critic step size be $\alpha _ { t + 1 } = \xi \eta _ { t }$ , with a timescale ratio $\xi$ chosen as in Remark 6.1, where $\begin{array} { r } { \eta _ { m a x } \triangleq \operatorname* { m i n } \left( \frac { \lambda } { 2 L _ { J } C _ { j o i n t } \left( 1 - \gamma \right) } , \frac { 1 } { 2 \xi \kappa } \right) } \end{array}$ is the maximum allowable actor step size. Then, the average Lyapunov sequence is bounded by:

$$
\frac { 1 } { T } \sum _ { t = 0 } ^ { T - 1 } \mathcal { L } _ { t } \leq \frac { t _ { 0 } - 1 } { T ( \rho _ { \mathcal { L } } c _ { \eta } - 1 ) } \mathcal { L } _ { 0 } + \frac { c _ { \eta } ^ { 2 } \Sigma _ { \mathcal { L } } } { \rho _ { \mathcal { L } } c _ { \eta } - 1 } \left( \frac { 1 + \log T } { T } \right) + \frac { c _ { \eta } C _ { \epsilon , \mathcal { L } } } { \rho _ { \mathcal { L } } c _ { \eta } - 1 } \epsilon _ { a p p } .
$$

By choosing the step-size parameters $\begin{array} { r } { c _ { \eta } = \Theta \bigl ( \frac { 1 } { \lambda \tau } \bigr ) } \end{array}$ and $\begin{array} { r } { t _ { 0 } = \Theta \big ( \frac { 1 } { ( 1 - \gamma ) ^ { 2 } \lambda ^ { 2 } \tau } + \frac { 1 } { ( 1 - \gamma ) ^ { 2 } \kappa ^ { 2 } \lambda ^ { 3 / 2 } \tau } \big ) } \end{array}$ subject to the stated conditions, the expected average regularized performance gap satisfies the convergence rate:

$$
\frac { 1 } { T } \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } [ G a p _ { t } ] \leq \mathcal { O } \left( \frac { 1 } { ( 1 - \gamma ) ^ { 5 } \kappa ^ { 6 } \lambda ^ { 5 / 2 } \tau ^ { 2 } } \frac { \log T } { T } \right) + \mathcal { O } \left( \frac { \epsilon _ { a p p } } { ( 1 - \gamma ) \lambda ^ { 2 } \tau } \right) .
$$

## 6.3 Last-Iterate Regularized Convergence

While the average-iterate sequence provides a robust empirical estimator, we further prove that the terminal policy converges to the regularized parametric optimum at an $\mathcal { O } ( T ^ { - 1 } )$ rate (up to the approximation bias). Because our step-size schedule explicitly enforces a non-negative multiplier, the joint Lyapunov recurrence yields a formal linear contraction. We first present a standard variant of Chung’s Lemma (Chung, 1954) to bound the resulting recursive sequence.

Lemma 6.2 (Chung’s Lemma for Robbins-Monro Sequences). Let $\{ x _ { t } \} _ { t = 0 } ^ { \infty }$ be a non-negative sequence satisfying the recursive bound:

$$
x _ { t + 1 } \leq \left( 1 - { \frac { c } { t + t _ { 0 } } } \right) x _ { t } + { \frac { a } { ( t + t _ { 0 } ) ^ { 2 } } } + { \frac { b } { t + t _ { 0 } } } , \quad t \geq 0 ,
$$

where the constants satisfy $c > 1 , a > 0 , b \geq 0$ , and $t _ { 0 } \geq c$ . Then, the sequence is bounded by:

$$
x _ { t } \leq \frac { v } { t + t _ { 0 } } + \frac { b } { c - 1 } , \quad t \geq 0 ,
$$

where $v \triangleq$ max $\textstyle \left( t _ { 0 } x _ { 0 } , { \frac { a } { c - 1 } } \right)$

Theorem 6.3 (Last-Iterate Regularized Convergence). For Algorithm 1 with T replaced by $T + 1$ suppose that Assumptions 1–4 and 6–7 hold. Consider the actor and critic step-size schedules defined in Theorem 6.2, with the additional requirement that the initial ofset satisfies $t _ { 0 } \geq \rho _ { \mathcal { L } } c _ { \eta }$ The Lyapunov sequence of the last iterate is bounded by:

$$
\mathcal { L } _ { T } \leq \frac { v } { T + t _ { 0 } } + \frac { c _ { \eta } C _ { \epsilon , \mathcal { L } } } { \rho _ { \mathcal { L } } c _ { \eta } - 1 } \epsilon _ { a p p } ,
$$

where the terminal constant is defined as $v \triangleq$ max $\begin{array} { r } { \left( t _ { 0 } \mathcal { L } _ { 0 } , \frac { c _ { \eta } ^ { 2 } \Sigma _ { \mathcal { L } } } { \rho _ { \mathcal { L } } c _ { \eta } - 1 } \right) } \end{array}$ . By choosing the step-size $p a \textmd { - }$ rameters $\begin{array} { r } { c _ { \eta } = \Theta \bigl ( \frac { 1 } { \lambda \tau } \bigr ) } \end{array}$ and $\begin{array} { r } { t _ { 0 } = \Theta \big ( \frac { 1 } { ( 1 - \gamma ) ^ { 2 } \lambda ^ { 2 } \tau } + \frac { 1 } { ( 1 - \gamma ) ^ { 2 } \kappa ^ { 2 } \lambda ^ { 3 / 2 } \tau } \big ) } \end{array}$ subject to the stated conditions, the expected last-iterate regularized performance gap satisfies the convergence rate:

$$
\mathbb { E } [ G a p _ { T } ] \le \mathcal { O } \left( \frac { 1 } { ( 1 - \gamma ) ^ { 5 } \kappa ^ { 6 } \lambda ^ { 5 / 2 } \tau ^ { 2 } T } \right) + \mathcal { O } \left( \frac { \epsilon _ { a p p } } { ( 1 - \gamma ) \lambda ^ { 2 } \tau } \right) .
$$

The additional iteration in Theorem 6.3 is required only by the indexing of the critic tracking error. Specifically, $\mathcal { L } _ { T }$ contains $Z _ { T } = \mathbb { E } [ \| \theta _ { T + 1 } - \theta _ { \tau } ^ { * } ( \omega _ { T } ) \| _ { 2 } ^ { 2 } ]$ , so that one additional critic update at $t = T$ is needed to establish the bound for the policy $\pi _ { T }$ . This extra update changes the iteration count from $T$ to $T + 1$ and has no efect on the stated asymptotic convergence rate.

Remark 6.2 (Equivalence of Average and Last-Iterate Rates). It is noteworthy that the last-iterate convergence rate (Theorem 6.3) asymptotically matches the average-iterate rate (Theorem 6.2) up to a logarithmic factor, both achieving an optimal $\tilde { \mathcal { O } } ( T ^ { - 1 } )$ dependence. Under a two-timescale step-size scheme, the actor’s parameter-space PL condition and the objective smoothness combine to facilitate a joint Lyapunov recurrence for the actor-critic system. In classical unconstrained optimization, the standard PL condition and objective smoothness are known to enable optimal $\tilde { \mathcal { O } } ( T ^ { - 1 } )$ or $\mathcal { O } ( T ^ { - 1 } )$ convergence rates for both the average and last iterates of stochastic gradient descent (Karimi et al., 2016; Khaled and Richt´arik, 2023). For our algorithm under the $\mathcal { O } ( 1 / t )$ diminishing step-size schedule, this joint recurrence similarly leads to optimal rates for both the last-iterate and average-iterate convergence.

## 7 Unregularized Global Convergence (Stochastic Regime)

While Section 6 establishes the convergence of Algorithm 1 under a fixed regularization temperature $\tau ,$ our ultimate goal is to solve the unregularized MDP. Recall from Section 2.1 that $\pi _ { \omega _ { 0 } ^ { * } }$ denotes an unregularized parametrically optimal policy, and $\pi _ { 0 } ^ { * }$ denotes the unregularized globally optimal policy. To establish convergence against this global optimum, we first transition our convergence guarantees from being against $\pi _ { \omega _ { \tau } ^ { * } }$ to against $\pi _ { \tau } ^ { * }$ . We then feed this extended global regularized bound directly into the Exponential Translation Bounds (Section 3.2) under Assumption 5. By tuning the temperature $\tau$ to balance the regularized optimization error against the exponentially small translation tail, this bypasses the standard linear entropy penalty, extracting the optimal $\tilde { \mathcal { O } } ( T ^ { - 1 } )$ global convergence rates for the unregularized problem.

For environments where the Minimal Action Gap condition (Assumption 5) is violated, we provide a robust fallback analysis in Appendix F that utilizes the standard linear entropy penalty to guarantee $\tilde { \mathcal { O } } ( T ^ { - 1 / 3 } )$ unregularized convergence rates.

## 7.1 Global Convergence Bounds

We have so far bounded the convergence of Algorithm 1 relative to the regularized parametrically optimal policy $\pi _ { \omega _ { \tau } ^ { * } }$ . We now extend these bounds to the regularized globally optimal policy $\pi _ { \tau } ^ { * }$ over

the entire probability simplex. We begin by formalizing the density ratio bounds of the regularized global optimum relative to the parametric optimum.

Assumption 8 (Global-to-Parametric Joint Concentrability). The joint state-action density ratio of the regularized global optimum $\pi _ { \tau } ^ { * }$ with respect to the parametric optimum $\pi _ { \omega _ { \tau } ^ { * } }$ is bounded:

$$
\operatorname* { s u p } _ { s , a } \frac { d _ { \tau } ^ { \ast } ( s ) \pi _ { \tau } ^ { \ast } ( a | s ) } { d ^ { \omega _ { \tau } ^ { \ast } } ( s ) \pi _ { \omega _ { \tau } ^ { \ast } } ( a | s ) } \leq C _ { j o i n t } ^ { \ast } ,
$$

where $C _ { j o i n t } ^ { * } > 0$ is a constant independent of the temperature $\tau \in ( 0 , \tau _ { m a x } ]$

By leveraging the suboptimality decomposition at the parametric optimum, we prove that the regularized performance diference between the regularized global optimum and its parametric counterpart is tightly controlled by the uncentered critic’s approximation error.

Lemma 7.1 (Global Class Approximation Error). Under Assumptions 1, 4, 6, and 8, the regularized performance gap between the regularized globally optimal policy $\pi _ { \tau } ^ { * }$ and the regularized parametrically optimal policy $\pi _ { \omega _ { \tau } ^ { * } }$ is bounded by:

$$
J _ { \tau } ( \pi _ { \tau } ^ { * } ) - J _ { \tau } ( \pi _ { \omega _ { \tau } ^ { * } } ) \leq \frac { C _ { c l a s s } } { \tau } \epsilon _ { a p p } ,
$$

where $\begin{array} { r } { C _ { c l a s s } \triangleq \frac { B _ { \phi } ^ { 4 } } { ( 1 - \gamma ) \lambda ^ { 2 } } + \frac { C _ { j o i n t } ^ { * } } { 1 - \gamma } } \end{array}$

Because the global regularized performance gap decomposes as $\mathrm { G a p } _ { t } ^ { \dagger } = \mathrm { G a p } _ { t } + \left( J _ { \tau } ( \pi _ { \tau } ^ { * } ) - \right.$ $J _ { \tau } ( \pi _ { \omega _ { \tau } ^ { * } } ) )$ , Lemma 7.1 guarantees that the global performance gap inherits the same convergence rates established for the parametric performance gap in Theorems 6.2 and 6.3, subject only to an additional representation penalty.

Corollary 7.1 (Global Regularized Performance Bound). Under Assumptions $1 , \ 4 , \ 6 ,$ and $\delta ,$ for any training policy $\pi _ { t } { } _ { ; }$ , the global regularized performance gap is bounded by the parametric regularized performance gap plus the class approximation penalty:

$$
G a p _ { t } ^ { \dagger } \le G a p _ { t } + \frac { C _ { c l a s s } } { \tau } \epsilon _ { a p p } .
$$

The class approximation bound (Lemma 7.1), together with the Positive-Definite Fisher Information in the Stochastic Regime and the Minimal Action Gap, reveals an intrinsic compatibility between the approximation error $\epsilon _ { a p p }$ and the temperature τ , as formalized by the following lemma.

Lemma 7.2 (Part-I Approximation-Temperature Compatibility). Under Assumptions 1, $ 4 - 6 ,$ and 8, for any temperature $\tau \in ( 0 , \tau _ { m a x } ]$ , we have

$$
\epsilon _ { a p p } \geq C _ { c o m p } \tau ,
$$

where

$$
C _ { c o m p } \triangleq \frac { q _ { 0 } \Delta } { 2 ( 1 - \gamma ) C _ { c l a s s } } \operatorname* { m i n } \left\{ 1 , \frac { q _ { 0 } \Delta } { 2 \tau _ { m a x } \log | \mathscr { A } | } \right\} , \quad q _ { 0 } \triangleq \frac { d \lambda } { 4 B _ { \phi } ^ { 2 } } ,
$$

and $C _ { c l a s s }$ is defined in Lemma 7.1. Consequently, arbitrarily small $\epsilon _ { a p p }$ , independent of temperature $\tau \in ( 0 , \tau _ { m a x } ]$ , are incompatible with the stated Part-I assumptions, and the exact-realizability case $\epsilon _ { a p p } = 0$ is excluded in the global analysis under the Stochastic Regime.

## 7.2 Average-Iterate Unregularized Convergence

We now map the global regularized bounds to the unregularized objective. Unlike the parametric bounds analyzed in Appendix F—which are bottlenecked by the linear $\mathcal { O } ( \tau )$ entropy penalty of the training policy—we utilize the Universal Exponential Translation Bound (Corollary 3.1). Because this translation absorbs the policy entropy, we tune the fixed temperature $\tau$ to decrease much more slowly, optimally balancing the regularized optimization error against the exponentially small translation tail to extract an $\tilde { \mathcal { O } } ( T ^ { - 1 } )$ convergence rate.

Theorem 7.1 (Unregularized Average-Iterate Convergence). For Algorithm 1, suppose that Assumptions 1–8 hold. Define the two-stage temperature:

$$
\tau _ { T } \triangleq \operatorname* { m a x } \left( \frac { \Delta } { 2 \log ( C _ { \gamma } T ) } , \frac { \Delta } { 2 \log ( C _ { \gamma } ( 1 + \epsilon _ { a p p } ^ { - 1 } ) ) } \right) .
$$

$I f T$ is suficiently large and $\epsilon _ { a p p }$ is suficiently small such that $\tau _ { T } \le \tau _ { m a x }$ , then by choosing the step-size parameters as in Theorem 6.2 evaluated at $\tau = \tau _ { T }$ subject to the stated conditions, the expected average global unregularized performance gap satisfies the convergence rate:

$$
\begin{array} { r l r } {  { \frac { 1 } { T } \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } \big [ J _ { 0 } ( \pi _ { 0 } ^ { * } ) - J _ { 0 } ( \pi _ { t } ) \big ] \leq \mathcal { O } ( \frac { ( \log T ) / ( 1 - \gamma ) ^ { 2 } + ( \log T ) ^ { 3 } } { ( 1 - \gamma ) ^ { 6 } \kappa ^ { 6 } \lambda ^ { 5 / 2 } \Delta ^ { 3 } T } ) } } \\ & { } & { \quad + \mathcal { O } ( \frac { ( 1 - \gamma ) ^ { - 1 } + \log ( 1 + \epsilon _ { a p p } ^ { - 1 } ) } { ( 1 - \gamma ) ^ { 2 } \lambda ^ { 2 } \Delta ^ { 2 } } \epsilon _ { a p p } ) } \\ & { } & { \quad = \tilde { \mathcal { O } } ( \frac { 1 } { T } ) + \tilde { \mathcal { O } } ( \epsilon _ { a p p } ) . } \end{array}
$$

Remark 7.1 (Admissible Approximation-Error Range in Part I). The phrase $^ { 6 } \epsilon _ { a p p }$ suficiently small” in Theorem 7.1 refers to the requirement that the approximation-controlled branch of the two-stage temperature remain below $\tau _ { m a x }$ . This restriction needs to be considered together with the Part-I compatibility condition in Lemma 7.2. Indeed, by the definition of $\tau _ { T }$ , we have $\tau _ { T } \geq$ $\frac { \Delta } { 2 \log ( C _ { \gamma } ( 1 + \epsilon _ { a p p } ^ { - 1 } ) ) }$ . Then the compatibility condition $\epsilon _ { a p p } \geq C _ { c o m p } \tau _ { T }$ implies

$$
\epsilon _ { a p p } \log \bigl ( C _ { \gamma } ( 1 + \epsilon _ { a p p } ^ { - 1 } ) \bigr ) \geq \frac { C _ { c o m p } \Delta } { 2 } ,
$$

which excludes arbitrarily small $\epsilon _ { a p p }$ because $\epsilon _ { a p p } \log ( 1 + \epsilon _ { a p p } ^ { - 1 } ) \to 0$ as $\epsilon _ { a p p }  0$ . Meanwhile, the requirement $\tau _ { T } \leq \tau _ { m a x }$ imposes an upper-side restriction on $\epsilon _ { a p p }$

$$
\log ( C _ { \gamma } ( 1 + \epsilon _ { a p p } ^ { - 1 } ) ) \geq \frac { \Delta } { 2 \tau _ { m a x } } .
$$

Thus, the Part-I assumptions and the selected temperature jointly constrain the admissible range of $\epsilon _ { a p p } .$ The statement of Theorem 7.1 is understood over this range.

## 7.3 Last-Iterate Unregularized Convergence

We extend our optimal unregularized convergence bounds to the final output policy of Algorithm 1. Because the last-iterate regularized bound (Theorem 6.3) avoids the log T penalty present in the average-iterate bound, applying the Exponential Translation mechanism leads to a sharper unregularized convergence rate (by a factor of log T) for the final iteration.

Theorem 7.2 (Unregularized Last-Iterate Convergence). For Algorithm 1 with T replaced by $T + 1$ suppose that Assumptions 1–8 hold. Consider the same two-stage temperature as in Theorem 7.1:

$$
\tau _ { T } \triangleq \operatorname* { m a x } \left( \frac { \Delta } { 2 \log ( C _ { \gamma } T ) } , \frac { \Delta } { 2 \log ( C _ { \gamma } ( 1 + \epsilon _ { a p p } ^ { - 1 } ) ) } \right) .
$$

$I f T$ is suficiently large and $\epsilon _ { a p p }$ is suficiently small such that $\tau _ { T } \le \tau _ { m a x }$ , then by choosing the step-size parameters as in Theorem 6.3 evaluated at $\tau = \tau _ { T }$ subject to the stated conditions, the expected last-iterate global unregularized performance gap satisfies the convergence rate:

$$
\begin{array} { r l } & { \mathbb { E } \big [ J _ { 0 } ( \pi _ { 0 } ^ { * } ) - J _ { 0 } ( \pi _ { T } ) \big ] \leq \mathcal { O } \left( \frac { ( 1 - \gamma ) ^ { - 2 } + ( \log T ) ^ { 2 } } { ( 1 - \gamma ) ^ { 6 } \kappa ^ { 6 } \lambda ^ { 5 / 2 } \Delta ^ { 3 } T } \right) + \mathcal { O } \left( \frac { ( 1 - \gamma ) ^ { - 1 } + \log ( 1 + \epsilon _ { a p p } ^ { - 1 } ) } { ( 1 - \gamma ) ^ { 2 } \lambda ^ { 2 } \Delta ^ { 2 } } \epsilon _ { a p p } \right) } \\ & { \quad \quad \quad \quad = \tilde { \mathcal { O } } \left( \frac { 1 } { T } \right) + \tilde { \mathcal { O } } ( \epsilon _ { a p p } ) . } \end{array}
$$

Remark 7.2 (Power of Minimal Action Gap). Comparing Theorems 7.1 and 7.2 against $\mathrm { A p \mathrm { - } }$ pendix F signifies the theoretical power of the Minimal Action Gap (Assumption 5). By leveraging the deterministic nature of the unregularized MDP to unlock the Exponential Translation Bounds (Section 3.2), this mechanism enables the slower $\tau \propto ( \log T ) ^ { - 1 }$ temperature tuning and accelerates the unregularized convergence rate to $\tilde { \mathcal { O } } ( T ^ { - 1 } )$ . Consequently, Algorithm 1 escapes the $\mathcal { O } ( \tau )$ linear entropy penalty that would otherwise restrict the unregularized convergence to $\tilde { \mathcal { O } } ( T ^ { - 1 / 3 } )$

## Part II Deterministic Regime

In the second part, we investigate the convergence of Algorithm 1 in the Deterministic Regime. As established in Section 2, this regime is characterized by the sequence of regularized optimal policies $\pi _ { \omega _ { \tau } ^ { * } }$ approaching an unregularized parametric optimum that is a deterministic policy in the closure of the log-linear family as the temperature drops $( \tau  0 )$ . This behavior naturally arises when a highly expressive log-linear family perfectly captures the deterministic unregularized global optimum $\pi _ { 0 } ^ { * } .$ , though it can also occur in a restricted family if the best policy in its closure is a suboptimal deterministic policy.

As this deterministic collapse occurs, the target probabilities for suboptimal actions decay to zero. Therefore, the action randomness vanishes and the minimum eigenvalue of the centered feature covariance collapses $( \lambda _ { \tau }  0 )$ , resulting in a degenerate Euclidean geometry. This degeneration violates the positive-definite Fisher Information assumption used to establish the Actor Progress Bound (Lemma 5.3), which alongside the Parameter-Space PL Condition (Lemma 4.2) is essential for establishing the two-way coupling of the actor-critic system (Lemmas 5.4 and 5.5). Consequently, the convergence analysis in Part I breaks down in the Deterministic Regime.

To survive the deterministic limit, our analysis abandons parameter-distance metrics, such as $\lVert \theta _ { \tau } ^ { * } ( \omega _ { t } ) - \tau \omega _ { t } \rVert _ { 2 }$ in Lemmas 4.2 and 5.3. Instead, we leverage the Policy Mirror Descent (PMD) framework for the actor parameter update to evaluate algorithmic progress against the regularized globally optimal policy $\pi _ { \tau } ^ { * }$ via the Kullback-Leibler (KL) divergence, bypassing the need for a positive-definite Fisher information matrix at the parametric optimum $\pi _ { \omega _ { \tau } ^ { * } }$ . Furthermore, because the Actor Progress Bound is no longer available to couple the critic’s target drift to the actor’s performance gap, we decouple the critic tracking from the actor progress and bound the target drift using a worst-case parameter step. By carefully balancing the decoupled step sizes, we control the competing algorithmic components to guarantee convergence.

To manage distribution shifts relative to the regularized global optimum $\pi _ { \tau } ^ { * }$ , we require that, under Algorithm 1, suficient coverage is maintained over the target state-action visitation measure. Note that Assumption 9 is utilized in exactly one place in our Part II analysis: in Step (ii) of the proof of the PMD Actor Progress Bound (Lemma 9.1), where it facilitates applying the action-space $\chi ^ { 2 } \mathrm { - t o - K L }$ divergence bound and shifting the state measure to control the approximation bias.

Assumption 9 (Global Joint Concentrability). Under Algorithm 1, the joint state-action density ratio of the regularized global optimum $\pi _ { \tau } ^ { * }$ with respect to the training policy $\pi _ { t }$ is bounded for all $t \geq 0$ :

$$
\operatorname* { s u p } _ { s , a } \frac { d _ { \tau } ^ { \ast } ( s ) \pi _ { \tau } ^ { \ast } ( a | s ) } { d ^ { \pi _ { t } } ( s ) \pi _ { t } ( a | s ) } \leq C _ { j o i n t } ^ { \dagger } ,
$$

where $C _ { j o i n t } ^ { \dagger } > 0$ is a constant independent of the temperature $\tau \in ( 0 , \tau _ { m a x } ]$

We make several comments related to Assumption 9. First, similarly to Assumption $^ { 7 , }$ the global joint concentrability bound is naturally implied by the conjunction of separate global state and policy concentrability bounds. Specifically, if the following density ratio bounds hold for some constants $( C ^ { \dagger } , W ^ { \dagger } )$ independent of $\tau \in ( 0 , \tau _ { m a x } ]$ :

• Global State Concentrability: $\| d _ { \tau } ^ { * } / d ^ { \pi _ { t } } \| _ { \infty } \leq C ^ { \dagger }$

• Global Policy Concentrability: $\begin{array} { r } { \operatorname* { s u p } _ { s , a } \frac { \pi _ { \tau } ^ { * } ( a | s ) } { \pi _ { t } ( a | s ) } \le W ^ { \dagger } } \end{array}$

then by the multiplicative rule of density ratios, the joint concentrability bound is satisfied with $C _ { j o i n t } ^ { \dagger } = C ^ { \dagger } W ^ { \dagger }$ . Conversely, because $\begin{array} { r } { \mathbb { E } _ { a \sim \pi _ { t } } \left[ \frac { \pi _ { \tau } ^ { * } ( a | s ) } { \pi _ { t } ( a | s ) } \right] = 1 } \end{array}$ at each state $s ,$ the joint concentrability directly implies the global state concentrability bound with $C ^ { \dagger } = C _ { j o i n t } ^ { \dagger } \mathrm { : } \quad \lVert d _ { \tau } ^ { \ast } / d ^ { \pi _ { t } } \rVert _ { \infty } \leq C _ { j o i n t } ^ { \dagger } \mathrm { . }$ However, the joint formulation allows the local policy ratio $\frac { \pi _ { \tau } ^ { * } ( a | s ) } { \pi _ { t } ( a | s ) }$ to be unbounded so long as the associated state density ratio $\frac { d _ { \tau } ^ { \ast } ( s ) } { d ^ { \pi } t \left( s \right) }$ is suficiently small.

Second, a simple suficient condition for the global policy concentrability bound $W ^ { \dagger }$ is a minimum action probability along the algorithmic trajectory. Specifically, if in $\dot { \mathbf { \Phi } _ { t , s , a } } \pi _ { t } ( a | s ) \mathbf { \Phi } \geq c$ for some constant $c > 0$ independent of $\tau .$ , then the policy concentrability condition holds with $W ^ { \dagger } = 1 / c$ However, such a uniform lower bound on $\pi _ { t } ( a | s )$ can be unrealistic in the Deterministic Regime, where probabilities of suboptimal actions may naturally vanish as $\tau  0$ . Our joint concentrability formulation is more flexible: it allows the local policy ratio $\frac { \pi _ { \tau } ^ { * } ( a | s ) } { \pi _ { t } ( a | s ) }$ to become large provided that it is compensated by a suficiently small state density ratio $\frac { d _ { \tau } ^ { \ast } ( s ) } { d ^ { \pi } t \left( s \right) }$

Third, while Assumption 7 in Part I bounds the algorithmic trajectory relative to the parametric optimum, Assumption 9 bounds it directly relative to the global optimum. If $\pi _ { \tau } ^ { * }$ were replaced by $\pi _ { \omega _ { \tau } ^ { * } }$ in Assumption 9, the entire analysis of regularized convergence in Section 9 would remain valid, yielding convergence bounds for the parametric gap ${ \mathrm { G a p } } _ { t }$ . However, the bound on the regularized performance diference between $\pi _ { \tau } ^ { * }$ and $\pi _ { \omega _ { \tau } ^ { * } }$ (Lemma 7.1) breaks down because $\lambda _ { \tau } \to 0 \mathrm { ~ a s ~ } \tau \to 0$ in the Deterministic Regime. Without this structural bridge, the Exponential Translation (Theorem 3.1) could not be applied to derive unregularized convergence bounds. Evaluating the trajectory directly against the global optimum bypasses this broken bridge. Furthermore, by the multiplicative rule of density ratios, this global assumption is naturally satisfied with $C _ { j o i n t } ^ { \dagger } = C _ { j o i n t } C _ { j o i n t } ^ { * }$ if both the Parametric Joint Concentrability (Assumption 7) and the Global-to-Parametric Joint Concentrability (Assumption 8) hold simultaneously.

Connections to Related Works. Assumption 9 serves as the regularized generalization of the standard concentrability and mismatch bounds widely utilized in the unregularized MDP literature (Agarwal et al., 2021; Yuan et al., 2023). While Agarwal et al. (2021) rely on $L _ { \infty }$ bounds that directly represent the unregularized counterpart of our assumption, Yuan et al. (2023) employ $L _ { 2 ^ { - } }$ based relaxations of these density ratios. Additionally, as noted in Part I, some analyses implicitly bundle this global distribution shift into a transferred approximation error bound evaluated under the global optimum (Liu et al., 2020). In Appendix D, we provide an extended discussion of these existing concentrability conditions, comparing them against our generalized mismatch bounds while accommodating exploratory ν-sampling.

To analyze critic tracking, we still require the Positive-Definite Uncentered Feature Moment (Assumption 3). For pedagogical purposes, to contrast with the Positive-Definite Fisher Information (Assumption 6) in Part I, we introduce a static assumption on the uncentered feature moment matrix evaluated at the regularized global optimum.

Assumption 10 (Positive-Definite Uncentered Feature Moment at Optimum). There exists a constant $\kappa ^ { * } > 0$ , independent of the temperature $\tau \in ( 0 , \tau _ { m a x } ]$ , such that

$$
\bar { \Sigma } _ { u n c } ( \pi _ { \tau } ^ { * } ) = \mathbb { E } _ { d _ { \tau } ^ { * } , \pi _ { \tau } ^ { * } } [ \phi ( s , a ) \phi ( s , a ) ^ { \top } ] \succeq \kappa ^ { * } I .
$$

When combined with the Global Density Ratio Bounds (Assumption 9), this static property dynamically guarantees the required positive-definiteness along the entire algorithmic trajectory. Specifically, for any training policy $\pi _ { t }$ and any vector $v ,$ Assumptions 9 and 10 together guarantee:

$$
v ^ { \top } \bar { \Sigma } _ { u n c } ( \pi _ { t } ) v \geq \frac { 1 } { C _ { j o i n t } ^ { \dag } } v ^ { \top } \bar { \Sigma } _ { u n c } ( \pi _ { \tau } ^ { * } ) v \geq \frac { \kappa ^ { * } } { C _ { j o i n t } ^ { \dag } } \| v \| _ { 2 } ^ { 2 } .
$$

Then the Positive-Definite Uncentered Feature Moment (Assumption 3) introduced in Section 2.3 is satisfied by taking $\begin{array} { r } { \kappa = \frac { \kappa ^ { * } } { C _ { j o i n t } ^ { \dagger } } } \end{array}$ in the Deterministic Regime.

This Positive-Definite Uncentered Feature Moment at Optimum difers sharply from the Positive-Definite Fisher Information (Assumption 6) used in Part I. While Assumption 10 evaluates the moment at the global optimum $\pi _ { \tau } ^ { * }$ rather than the parametric optimum $\pi _ { \omega _ { \tau } ^ { * } }$ , the fundamental distinction lies in the matrix structure. In Part I, positive-definiteness is assumed on the centered feature covariance matrix, which degenerates $( \lambda _ { \tau } \to 0 )$ as the temperature $\tau  0$ in the Deterministic Regime. However, the uncentered feature moment matrix can remain positive-definite even for deterministic policies, safely anchoring the critic updates in this regime.

Finally, it is worth noting that the PMD analysis developed in this part is agnostic to the value of $\lambda _ { \tau }$ and is therefore applicable to the Stochastic Regime as well. However, when $\lambda _ { \tau }$ is bounded away from 0 as $\tau  0$ , the analysis in Part I leverages the positive-definite Fisher Information to establish the Actor Progress Bound, which combines with the PL condition to extract significantly faster unregularized convergence rates. Because the PMD framework does not exploit this strict Euclidean curvature, it yields comparatively slower unregularized rates. Therefore, this PMD analysis is exclusively required for—and specifically tailored to—the Deterministic Regime.

## 8 Policy Mirror Descent

To bypass the degenerate Euclidean geometry in the Deterministic Regime, we analyze Algorithm 1 through the lens of Policy Mirror Descent (PMD) (Shani et al., 2020; Lan, 2023; Yuan et al., 2023), evaluating algorithmic progress directly on the probability simplex. Standard tabular PMD executes independent, state-by-state updates on the probability simplex. While applying this to function approximation typically incurs projection errors, the log-linear policy parameterization circumvents this issue (Yuan et al., 2023), establishing that NPG methods correspond exactly to unconstrained PMD with an appropriate functional gradient. We demonstrate how our specific entropy-regularized, uncentered NPG update also exhibits this structural equivalence.

In the exact PMD framework, a policy is updated by moving along a functional gradient $g ( s , a )$ while penalizing the KL divergence from the current policy. For a step size $\eta _ { t }$ , this executes the following functional optimization over the probability simplex $\Delta ( \mathcal { A } )$

$$
\pi _ { t + 1 } ( \cdot | s ) = \underset { p \in \Delta ( A ) } { \arg \operatorname* { m a x } } \left\{ \langle g ( s , \cdot ) , p \rangle _ { A } - \frac { 1 } { \eta _ { t } } D _ { K L } ( p \| \pi _ { t } ( \cdot | s ) ) \right\} .
$$

This yields the closed-form multiplicative update: $\pi _ { t + 1 } ( a | s ) \propto \pi _ { t } ( a | s ) \exp ( \eta _ { t } g ( s , a ) )$ . As shown in Appendix A, our actor parameter update $\omega _ { t + 1 } = \omega _ { t } + \eta _ { t } ( \theta _ { t + 1 } - \tau \omega _ { t } )$ is derived from the NPG direction. Under the log-linear structure $\pi _ { t } ( a | s ) \propto \exp ( \omega _ { t } ^ { \top } \phi ( s , a ) )$ , this translates directly to the probability simplex as:

$$
\pi _ { t + 1 } ( a | s ) \propto \exp \big ( ( \omega _ { t } + \eta _ { t } ( \theta _ { t + 1 } - \tau \omega _ { t } ) ) ^ { \top } \phi ( s , a ) \big ) = \pi _ { t } ( a | s ) \exp \big ( \eta _ { t } ( \theta _ { t + 1 } - \tau \omega _ { t } ) ^ { \top } \phi ( s , a ) \big ) .
$$

Therefore, our parameter update intrinsically executes an exact PMD step. The efective functional gradient evaluates exactly to $( g _ { t } ^ { a c } ) ^ { \top } \phi ( s , a ) = ( \theta _ { t + 1 } - \tau \omega _ { t } ) ^ { \top } \phi ( s , a )$

Consequently, the algorithmic progress term $\langle ( \theta _ { t + 1 } - \tau \omega _ { t } ) ^ { \top } \phi , \pi _ { \tau } ^ { * } - \pi _ { t } \rangle _ { A }$ from the Actual Critic Decomposition (Lemma 3.3(ii)) aligns with the PMD functional gradient. Because our update embodies an exact PMD step, this progress term can be bounded using the Bregman Three-Point identity, as derived in the following lemma.

Lemma 8.1 (PMD Algorithmic Progress). Under Assumptions 1 and 3, if the actor step size satisfies $\eta _ { t } \leq 1 / \tau$ , the algorithmic progress inner product satisfies the Bregman Three-Point inequality for any state s:

$$
\langle \left( \theta _ { t + 1 } - \tau \omega _ { t } \right) ^ { \top } \phi , \pi _ { \tau } ^ { * } - \pi _ { t } \rangle _ { A } \leq \frac { D _ { K L } ( \pi _ { \tau } ^ { * } ( \cdot | s ) \| \pi _ { t } ( \cdot | s ) ) - D _ { K L } ( \pi _ { \tau } ^ { * } ( \cdot | s ) \| \pi _ { t + 1 } ( \cdot | s ) ) } { \eta _ { t } } + \underbrace { \frac { \eta _ { t } } { 2 } G _ { a c } ^ { 2 } B _ { \phi } ^ { 2 } } _ { P M D \ L o c a l \ E r r o r } .
$$

Remark 8.1 (Conceptual PMD vs Algebraic Proofs). While we frame our actor update as conceptually executing a PMD step to motivate the shift to KL geometry in the Deterministic Regime, our mathematical proofs do not invoke the PMD representation (i.e., the arg max projection over the probability simplex). Standard PMD analyses (Shani et al., 2020; Lan, 2023; Yuan et al., 2023) rely on the variational inequalities (first-order optimality conditions) of this projection, which bound algorithmic progress by evaluating the functional gradient against the updated policy $\pi _ { t + 1 }$ $\mathrm { ( i . e . , ~ } \langle g _ { t } , \pi _ { \tau } ^ { * } - \pi _ { t + 1 } \rangle _ { A } \mathrm { ) }$ . In contrast, similar to Agarwal et al. (2021), our analysis in Lemma 8.1 substitutes the exact log-linear parameter update directly into the Bregman Three-Point identity, evaluating the gradient against the current policy $\pi _ { t }$ . This approach isolates the exact local step error $D _ { K L } ( \pi _ { t } \Vert \pi _ { t + 1 } )$ and bounds it via Hoefding’s Lemma, allowing us to exploit the PMD geometry without relying on the variational inequalities.

## 9 Fixed-Temperature Convergence (Deterministic Regime)

We show that Algorithm 1 achieves an $\mathcal { O } ( T ^ { - 2 / 3 } )$ average-iterate and an $\tilde { \mathcal { O } } ( T ^ { - 1 / 3 } )$ last-iterate convergence rate for the global regularized performance gap. This is accomplished by synthesizing the actor’s PMD geometry, established in Section 8, with the uncoupled critic tracking dynamics developed in this section, dynamically balancing the decoupled diminishing step sizes to control the competing forces from the actor and critic updates.

## 9.1 Actor Progress and Value Telescoping

To track global progress, we recall the global regularized performance gap, ${ \mathrm { G a p } } _ { t } ^ { \dagger } \triangleq J _ { \tau } ( \pi _ { \tau } ^ { * } ) - J _ { \tau } ( \pi _ { t } )$ introduced in Section 2.1, and we denote the state-averaged forward KL divergence from the global optimum $\pi _ { \tau } ^ { * }$ as $D _ { t } ^ { \dagger } \triangleq { \mathbb E } _ { d _ { \tau } ^ { * } } [ D _ { K L } ( \pi _ { \tau } ^ { * } \| \pi _ { t } ) ]$ , distinct from $D _ { t }$ used in Part I.

A notable diference distinguishes this analysis from that in Part I. In Part I, we rely on the Positive-Definite Fisher Information at parametric optimum $\pi _ { \omega _ { \tau } ^ { * } }$ (Assumption 6) to establish the Actor Progress Bound, ensuring algorithmic progress along the Euclidean parameter update. This Euclidean progress is then combined with the parameter-space PL condition to obtain the Coupled Actor Recurrence. In Part II, as $\tau  0$ , the deterministic collapse forces the minimum eigenvalue of the Fisher Information to vanish $( \lambda _ { \tau }  0 )$ , causing this Euclidean geometry to fail.

In contrast, the PMD geometry directly tracks the KL divergence. This allows us to directly bound the step-by-step algorithmic progress in the Actual Critic Decomposition from Lemma 3.3(ii), thereby bypassing the need for a non-degenerate Fisher information matrix. We first derive the fundamental single-step inequality, and subsequently apply a value telescoping argument to extract the average-iterate performance gap.

Lemma 9.1 (PMD Actor Progress Bound). Let $Z _ { t } \triangleq \mathbb { E } [ \Vert e _ { t } \Vert _ { 2 } ^ { 2 } ]$ be the expected critic tracking error. Under Assumptions 1, 3, 4, and $^ { g } ,$ if the actor step size satisfies $\eta _ { t } \leq 1 / \tau$ , the expected global performance gap satisfies the single-step inequality:

$$
( 1 - \gamma ) \mathbb { E } [ G a p _ { t } ^ { \dagger } ] \le - \frac { 3 \tau } { 4 } \mathbb { E } [ D _ { t } ^ { \dagger } ] + \frac { \mathbb { E } [ D _ { t } ^ { \dagger } ] - \mathbb { E } [ D _ { t + 1 } ^ { \dagger } ] } { \eta _ { t } } + \frac { \eta _ { t } } { 2 } G _ { a c } ^ { 2 } B _ { \phi } ^ { 2 } + \frac { 4 C _ { j o i n t } ^ { \dagger } } { \tau } \epsilon _ { a p p } + \frac { 4 B _ { \phi } ^ { 2 } } { \tau } Z _ { t } .
$$

Notably, the joint concentrability constant $C _ { j o i n t } ^ { \dagger }$ only penalizes the approximation bias, keeping the tracking error penalty completely free of concentrability assumptions.

To extract the average-iterate performance gap from the single-step PMD bound, we need to manage the accumulated tracking and approximation errors. To prevent these errors from being amplified, we choose a diminishing step-size schedule $\begin{array} { r } { \begin{array} { r } { \eta _ { t } = \frac { c _ { \eta } } { t + t _ { 0 } } } \end{array} } \end{array}$ to cancel the restorative KL divergence drift, creating a clean telescoping sequence across the training horizon.

Lemma 9.2 (PMD Value-Telescoping Bound). Let the actor step size be $\begin{array} { r } { \begin{array} { r } { \eta _ { t } = \frac { c _ { \eta } } { t + t _ { 0 } } } \end{array} } \end{array}$ , with a numerator constant chosen such that $c _ { \eta } ~ \geq ~ \frac { 4 } { 3 \tau }$ , and an initial ofset satisfying $t _ { 0 } ~ \ge ~ c _ { \eta } \tau$ . Under Assumptions 1, 3, 4, and 9, the expected average regularized performance gap is bounded by:

$$
( 1 - \gamma ) \frac { 1 } { T } \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } [ G a p _ { t } ^ { \dag } ] \leq \frac { t _ { 0 } - 1 } { c _ { \eta } T } D _ { 0 } ^ { \dag } + \frac { 1 } { T } \sum _ { t = 0 } ^ { T - 1 } \mathcal { E } _ { n o i s e } ( t ) ,
$$

where the noise envelope is defined as $\begin{array} { r } { \mathcal { E } _ { n o i s e } ( t ) \triangleq \frac { \eta _ { t } } { 2 } G _ { a c } ^ { 2 } B _ { \phi } ^ { 2 } + \frac { 4 C _ { j o i n t } ^ { \dagger } } { \tau } \epsilon _ { a p p } + \frac { 4 B _ { \phi } ^ { 2 } } { \tau } Z _ { t } } \end{array}$

## 9.2 Uncoupled Critic Tracking

With the actor’s PMD progress established, we bound the remaining critic tracking error. Here, another important diference from Part I arises in controlling the critic target drift. In Part I, the $\mathrm { A c } -$ tor Progress Bound—enabled by the Positive-Definite Fisher Information (Assumption 6)—bounds the Euclidean parameter step by the expected reduction in the performance gap, establishing the Coupled Critic Tracking. In Part II, the deterministic collapse $( \lambda _ { \tau } ~  ~ 0 )$ severs this objective coupling. Consequently, we bound the ideal critic target drift using the worst-case, uncoupled parameter step. From Lemma 2.2(ii), the absolute change in the actor parameter is bounded by the temperature-independent constant $G _ { a c } \colon \| \omega _ { t + 1 } - \omega _ { t } \| _ { 2 } = \eta _ { t } \| \theta _ { t + 1 } - \tau \omega _ { t } \| _ { 2 } \leq \eta _ { t } G _ { a c } .$

The telescoping bound from Lemma 9.2 dictates that the actor step size decays as $\eta _ { t } = \mathcal { O } ( t ^ { - 1 } )$ To facilitate fast convergence, the critic step size $\alpha _ { t + 1 }$ needs to be carefully tuned. As shown in the proof of Lemma 9.4, the expected tracking error $Z _ { t } \triangleq \mathbb { E } [ \Vert e _ { t } \Vert _ { 2 } ^ { 2 } ]$ satisfies the recurrence

$$
Z _ { t + 1 } \leq ( 1 - \alpha _ { t + 1 } \kappa ) Z _ { t } + \underbrace { \alpha _ { t + 1 } ^ { 2 } G _ { c r } ^ { 2 } } _ { \mathrm { S G D ~ E r r o r } } + \underbrace { \frac { 1 } { \alpha _ { t + 1 } \kappa } L _ { \theta } ^ { 2 } \eta _ { t } ^ { 2 } G _ { a c } ^ { 2 } } _ { \mathrm { T a r g e t ~ D r i f t } } .
$$

This recurrence reveals two competing, uncoupled noise forces: the SGD error scaling as $\mathcal { O } ( \alpha _ { t + 1 } ^ { 2 } )$ and the target drift scaling as $\mathcal { O } ( \eta _ { t } ^ { 2 } / \alpha _ { t + 1 } )$ . To balance these two forces $( \alpha _ { t + 1 } ^ { 2 } \ \mathrm { v s } \ \eta _ { t } ^ { 2 } / \alpha _ { t + 1 } )$ , we set $\alpha _ { t + 1 } \propto \eta _ { t } ^ { 2 / 3 }$ . Given the actor decay $\eta _ { t } = \mathcal { O } ( t ^ { - 1 } )$ , this yields the critic schedule $\alpha _ { t + 1 } = \mathcal { O } ( t ^ { - 2 / 3 } )$ The choice $\alpha _ { t + 1 } \propto \eta _ { t } ^ { 2 / 3 }$ not only ensures that both noise components decay as $\mathcal { O } ( t ^ { - 4 / 3 } )$ , but also balances their dependence on the temperature τ : this fractional coupling guarantees that both the SGD error and the target drift share an identical $\Theta ( \tau ^ { - 4 / 3 } )$ temperature dependence. See the order evaluation of the noise envelope K in the proof of Theorem 9.1.

To formally prove that the critic’s tracking error achieves this balanced $\mathcal { O } ( t ^ { - 2 / 3 } )$ rate, we require a specialized stochastic approximation tool. Unlike standard Robbins-Monro sequences that decay as $\mathcal { O } ( 1 / t )$ , our critic operates on a slower fractional timescale. The following lemma provides the pointwise convergence guarantee for this fractional geometry.

Lemma 9.3 (Fractional Chung’s Lemma). Let $\{ x _ { t } \} _ { t = 0 } ^ { \infty }$ be a non-negative sequence satisfying the recursive bound:

$$
x _ { t + 1 } \leq \left( 1 - { \frac { c } { ( t + t _ { 0 } ) ^ { 2 / 3 } } } \right) x _ { t } + { \frac { K } { ( t + t _ { 0 } ) ^ { 4 / 3 } } } , \quad t \geq 0 ,
$$

where $c > 0 , K \geq 0$ , and the initial ofset satisfies $t _ { 0 } \geq$ max $\left( c ^ { 3 / 2 } , \left( \frac { 4 } { 3 c } \right) ^ { 3 } \right)$ . Then, the sequence is

bounded pointwise by:

$$
x _ { t } \leq \frac { v } { ( t + t _ { 0 } ) ^ { 2 / 3 } } , \quad t \geq 0 ,
$$

where the bounding constant is defined as $v \triangleq \operatorname* { m a x } \left( t _ { 0 } ^ { 2 / 3 } x _ { 0 } , \frac { 2 K } { c } \right)$

With the fractional convergence tool established, we now bound the uncoupled critic tracking error. By evaluating the constrained SGD update alongside the worst-case target drift, we construct the recursive sequence required to apply the fractional tracking limit.

Lemma 9.4 (Pointwise Critic Tracking Error). Suppose that Assumptions 1–3 hold. For any independent constants $c _ { \eta } , c _ { \alpha } > 0$ , let the actor and critic step sizes be $\begin{array} { r } { \eta _ { t } = \frac { c _ { \eta } } { t + t _ { 0 } } } \end{array}$ and $\begin{array} { r } { \alpha _ { t + 1 } = \frac { c _ { \alpha } } { ( t + t _ { 0 } ) ^ { 2 / 3 } } } \end{array}$ for all $t \geq 0$ , with an arbitrary initialization $\alpha _ { 0 }$ and an initial ofset $\begin{array} { r } { t _ { 0 } \geq \operatorname* { m a x } \left( c _ { \eta } \tau , ( 2 c _ { \alpha } \kappa ) ^ { 3 / 2 } , ( \frac { 4 } { 3 c _ { \alpha } \kappa } ) ^ { 3 } \right) } \end{array}$ Then the expected tracking error $Z _ { t } \triangleq \mathbb { E } [ \Vert e _ { t } \Vert _ { 2 } ^ { 2 } ]$ is bounded pointwise for all $t \geq 0$ by:

$$
Z _ { t } \le \frac { v } { ( t + t _ { 0 } ) ^ { 2 / 3 } } ,
$$

where the bounding coeficient is $v \triangleq$ max $\begin{array} { r } { \left( t _ { 0 } ^ { 2 / 3 } Z _ { 0 } , \frac { 2 K } { c _ { \alpha } \kappa } \right) } \end{array}$ , and the noise envelope constant is $K \triangleq$ $\begin{array} { r } { c _ { \alpha } ^ { 2 } G _ { c r } ^ { 2 } + \frac { L _ { \theta } ^ { 2 } c _ { \eta } ^ { 2 } G _ { a c } ^ { 2 } } { c _ { \alpha } \kappa } } \end{array}$

## 9.3 Average-Iterate Regularized Convergence

We combine the PMD Value-Telescoping Bound (Lemma 9.2) and the Pointwise Critic Tracking Error (Lemma 9.4) to bound the total algorithmic error. The expected performance gap decomposes into an initialization error, a PMD local error, the critic tracking error, and the approximation bias. By evaluating these terms under the selected step-size schedule, the following theorem establishes a $\mathcal { O } ( T ^ { - 2 / 3 } )$ average-iterate convergence rate for the regularized global objective.

Theorem 9.1 (Average-Iterate Regularized Convergence). For Algorithm 1, suppose that Assumptions $1 \mathrm { - } \mathit { 4 }$ and 9 hold. For independent constants $c _ { \eta } \geq \frac { 4 } { 3 \tau }$ and $\zeta > 0 ,$ , let the actor and critic step sizes be $\begin{array} { r } { \eta _ { t } ~ = ~ \frac { c _ { \eta } } { t + t _ { 0 } } } \end{array}$ and $\begin{array} { r } { \alpha _ { t + 1 } ~ = ~ \frac { c _ { \alpha } } { ( t + t _ { 0 } ) ^ { 2 / 3 } } } \end{array}$ for all $t \geq 0$ , with $c _ { \alpha } \triangleq \zeta c _ { \eta } ^ { 2 / 3 }$ and an initial $o f f -$ set $t _ { 0 } \geq$ max $\left( c _ { \eta } \tau , ( 2 c _ { \alpha } \kappa ) ^ { 3 / 2 } , ( \frac { 4 } { 3 c _ { \alpha } \kappa } ) ^ { 3 } \right)$ . Then the expected average regularized performance gap is bounded by:

$$
\begin{array} { r l } & { \frac { 1 } { T } \displaystyle \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } [ G a p _ { t } ^ { \dag } ] \leq \frac { 1 } { 1 - \gamma } \Bigg [ \underbrace { \frac { t _ { 0 } - 1 } { c _ { \eta } T } D _ { 0 } ^ { \dag } } _ { P M D ~ I n i t a l i z a t i o n ~ E r r o r } + \underbrace { \frac { c _ { \eta } G _ { a c } ^ { 2 } B _ { \phi } ^ { 2 } } { 2 T } ( 1 + \log T ) } _ { P M D ~ L o c a l ~ E r r o r } } \\ & { \quad \quad \quad \quad \quad + \underbrace { \frac { 1 2 B _ { \phi } ^ { 2 } v } { \tau } T ^ { - 2 / 3 } \left( 1 + \frac { t _ { 0 } } { T } \right) ^ { 1 / 3 } } _ { C r i t i c ~ T r a c k i n g ~ E r r o r } + \underbrace { \frac { 4 C _ { j o i n t } ^ { \dag } } { \tau } \epsilon _ { a p p } } _ { A p p r o x i m a t i o n ~ B i a s } \Bigg ] . } \end{array}
$$

By choosing the step-size constants $c _ { \eta } = \Theta ( \tau ^ { - 1 } )$ and $\zeta = \Theta ( \kappa ^ { - 1 } )$ and the initial ofset $t _ { 0 } = \Theta ( \tau ^ { - 1 } )$ subject to the stated conditions, the asymptotic convergence rate is:

$$
\frac { 1 } { T } \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } [ G a p _ { t } ^ { \dag } ] \leq \underbrace { \mathcal { O } \left( \frac { 1 } { ( 1 - \gamma ) ^ { 7 } \kappa ^ { 6 } \tau ^ { 5 / 3 } } T ^ { - 2 / 3 } \right) } _ { P r i m a r y \ T h o c k i n g \ E r r o r } + \underbrace { \tilde { \mathcal { O } } \left( \frac { 1 } { ( 1 - \gamma ) ^ { 7 } \kappa ^ { 6 } \tau ^ { 2 } } T ^ { - 1 } \right) } _ { T r a n s i e n t \ C o n v e r g e n c e \ E r r o r } + \underbrace { \mathcal { O } \left( \frac { \epsilon _ { a p p } } { ( 1 - \gamma ) \tau } \right) } _ { A p p r o x i m a t i o n \ B i a s } .
$$

## 9.4 Last-Iterate Regularized Convergence

While the telescoping sequence extracts an accelerated $\mathcal { O } ( T ^ { - 2 / 3 } )$ rate for the average policy, we now establish the convergence of the final output policy. Because the PMD framework bounds algorithmic progress via the forward KL divergence, we evaluate the last-iterate performance gap by passing the terminal divergence through Pinsker’s inequality. To accomplish this, we establish the pointwise geometric contraction of the forward KL divergence. We first introduce a specialized variant of Chung’s Lemma that handles fractional noise and an additive bias.

Lemma 9.5 (Chung’s Lemma with Fractional Noise). Let $\{ x _ { t } \} _ { t = 0 } ^ { \infty }$ be a non-negative sequence satisfying the recursive bound:

$$
x _ { t + 1 } \leq \left( 1 - { \frac { c } { t + t _ { 0 } } } \right) x _ { t } + { \frac { A } { ( t + t _ { 0 } ) ^ { 5 / 3 } } } + { \frac { B } { t + t _ { 0 } } } , \quad t \geq 0 ,
$$

where $c \geq 1 , A \geq 0 , B \geq 0$ , and the initial ofset satisfies $t _ { 0 } \geq c ,$ Then, the sequence is bounded pointwise by:

$$
x _ { t } \leq \frac { u } { ( t + t _ { 0 } ) ^ { 2 / 3 } } + \frac { B } { c } , \quad t \geq 0 ,
$$

where the bounding coeficient is $u \triangleq \operatorname* { m a x } \left( t _ { 0 } ^ { 2 / 3 } x _ { 0 } , \frac { 3 A } { 3 c - 2 } \right)$

Next, we derive the pointwise contraction of the forward KL divergence. By isolating the divergence within the single-step PMD Actor Progress Bound and substituting the Pointwise Critic Tracking Error, we construct a recurrence relation that aligns with our specialized Chung’s Lemma to establish the convergence of the forward KL divergence.

Lemma 9.6 (Last-Iterate Forward KL Convergence). Under the identical assumptions and stepsize conditions of Theorem 9.1, the expected state-averaged forward KL divergence of the final iterate is bounded by:

$$
\mathbb { E } [ D _ { T } ^ { \dagger } ] \le \frac { u } { ( T + t _ { 0 } ) ^ { 2 / 3 } } + \frac { 1 6 C _ { j o i n t } ^ { \dagger } } { 3 \tau ^ { 2 } } \epsilon _ { a p p } ,
$$

where the bounding coeficient is $u \overset { \triangle } { = }$ max $\left( t _ { 0 } ^ { 2 / 3 } D _ { 0 } ^ { \dagger } , \frac { 3 A } { 3 c - 2 } \right)$ , with $c \triangleq \frac { 3 \tau } { 4 } c _ { \eta }$ and $\begin{array} { r } { A \triangleq \frac { c _ { \eta } ^ { 2 } G _ { a c } ^ { 2 } B _ { \phi } ^ { 2 } } { 2 } + \frac { 4 c _ { \eta } B _ { \phi } ^ { 2 } v } { \tau } } \end{array}$ By choosing the step-size constants $c _ { \eta } = \Theta ( \tau ^ { - 1 } )$ and $\zeta = \Theta ( \kappa ^ { - 1 } )$ and the initial ofset $t _ { 0 } = \Theta ( \tau ^ { - 1 } )$

subject to the stated conditions, the asymptotic convergence rate is:

$$
\mathbb { E } [ D _ { T } ^ { \dagger } ] \le \mathcal { O } \left( \frac { 1 } { ( 1 - \gamma ) ^ { 6 } \kappa ^ { 6 } \tau ^ { 8 / 3 } } T ^ { - 2 / 3 } \right) + \mathcal { O } \left( \frac { \epsilon _ { a p p } } { \tau ^ { 2 } } \right) .
$$

With the geometric contraction of the forward KL divergence established, we translate this divergence back into the global performance gap. While the Exact Suboptimality Identity (Lemma 3.2) captures the suboptimality via the reverse KL divergence under the training state distribution $d ^ { \pi _ { t } }$ our global convergence analysis tracks the forward KL divergence under the optimal state distribution $d _ { \tau } ^ { * }$ to manage distribution shift. To bridge this analytical divide, we build on Lemma 3.3 and establish an upper bound on the performance gap through the forward KL divergence.

Lemma 9.7 (Forward KL Bound on Regularized Suboptimality). Under Assumptions 1 and 3, if the actor step size satisfies $\eta _ { t } \leq 1 / \tau$ , then the global regularized performance gap is bounded by the state-averaged forward KL divergence under $d _ { \tau } ^ { * }$ :

$$
J _ { \tau } ( \pi _ { \tau } ^ { * } ) - J _ { \tau } ( \pi _ { t } ) \leq \frac { \sqrt { 2 } M _ { \tau } } { 1 - \gamma } \sqrt { \mathbb { E } _ { s \sim d _ { \tau } ^ { * } } [ D _ { K L } ( \pi _ { \tau } ^ { * } ( \cdot | s ) \| \pi _ { t } ( \cdot | s ) ) ] } ,
$$

where the capacity bound is $M _ { \tau } \triangleq V _ { m a x } + 2 B _ { \theta } B _ { \phi } + \tau \log | \boldsymbol { \mathcal { A } } |$ . Consequently, for any temperature $\tau \in ( 0 , \tau _ { m a x } ]$ 2

$$
J _ { \tau } ( \pi _ { \tau } ^ { * } ) - J _ { \tau } ( \pi _ { t } ) \leq \frac { \sqrt { 2 } M _ { m a x } } { 1 - \gamma } \sqrt { \mathbb { E } _ { s \sim d _ { \tau } ^ { * } } [ D _ { K L } ( \pi _ { \tau } ^ { * } ( \cdot | s ) \| \pi _ { t } ( \cdot | s ) ) ] } ,
$$

where $M _ { m a x } \triangleq V _ { m a x } + 2 B _ { \theta } B _ { \phi } + \tau _ { m a x } \log \left| A \right|$

By applying this Forward KL Bound to the terminal divergence in Lemma 9.6, we extract a $\mathcal { O } ( T ^ { - 1 / 3 } )$ last-iterate convergence rate for the global regularized performance gap.

Theorem 9.2 (Last-Iterate Regularized Convergence). Under the identical assumptions and stepsize conditions of Theorem 9.1, the expected last-iterate regularized performance gap is bounded by:

$$
\mathbb { E } [ G a p _ { T } ^ { \dagger } ] \le \frac { \sqrt { 2 u } M _ { m a x } } { 1 - \gamma } \frac { 1 } { ( T + t _ { 0 } ) ^ { 1 / 3 } } + \frac { 4 M _ { m a x } \sqrt { 2 C _ { j o i n t } ^ { \dagger } } } { \sqrt { 3 } \tau ( 1 - \gamma ) } \sqrt { \epsilon _ { a p p } } ,
$$

where $M _ { m a x }$ is defined in Lemma 9.7, and u is the structural constant defined in Lemma 9.6. By choosing the step-size parameters $c _ { \eta } = \Theta ( \tau ^ { - 1 } )$ and $\zeta = \Theta ( \kappa ^ { - 1 } )$ and the initial ofset $t _ { 0 } = \Theta ( \tau ^ { - 1 } )$ subject to the stated conditions, the asymptotic convergence rate is:

$$
\mathbb { E } [ G a p _ { T } ^ { \dagger } ] \le \mathcal { O } \left( \frac { 1 } { ( 1 - \gamma ) ^ { 5 } \kappa ^ { 4 } \tau ^ { 4 / 3 } } T ^ { - 1 / 3 } \right) + \mathcal { O } \left( \frac { 1 } { ( 1 - \gamma ) ^ { 2 } \kappa T } \sqrt { \epsilon _ { a p p } } \right) .
$$

Remark 9.1 (Methodological Divergence of Average and Last-Iterate Bounds). The single PMD inequality derived in Lemma 9.1 serves as the common foundation for both our average-iterate and

last-iterate convergence guarantees, but algebraic constraints force the two metrics to take diferent paths. The core expected inequality is structured as:

$$
( 1 - \gamma ) \mathbb { E } [ \mathrm { G a p } _ { t } ^ { \dagger } ] + \frac { 3 \tau } { 4 } \mathbb { E } [ D _ { t } ^ { \dagger } ] \leq \frac { \mathbb { E } [ D _ { t } ^ { \dagger } ] - \mathbb { E } [ D _ { t + 1 } ^ { \dagger } ] } { \eta _ { t } } + \mathcal { E } _ { n o i s e } ( t ) ,
$$

where the noise envelope $\mathcal { E } _ { n o i s e } ( t ) = \mathcal { O } ( \eta _ { t } ) + \mathcal { O } ( Z _ { t } / \tau ) + \mathcal { O } ( \epsilon _ { a p p } / \tau )$ encapsulates the PMD local error, the critic tracking error, and the approximation bias.

For the average-iterate bound, telescoping this inequality over T steps utilizing the diminishing schedule $\begin{array} { r } { \eta _ { t } = \frac { c _ { \eta } } { t + t _ { 0 } } } \end{array}$ neutralizes the step-size division by multiplying through by $( t + t _ { 0 } )$ as shown in the proof. Because the uncancelled boundary term from the telescoping resolves to the initial divergence scaled by the iteration budget $\begin{array} { r } { \big ( \frac { t _ { 0 } - 1 } { c _ { \eta } T } D _ { 0 } ^ { \dagger } \big ) } \end{array}$ , we balance this transient divergence penalty against the noise, unlocking the accelerated $\mathcal { O } ( T ^ { - 2 / 3 } )$ regularized rate.

For the last-iterate bound, however, direct extraction is impossible. Without telescoping, the terminal performance gap $\mathbb { E } [ \mathrm { G a p } _ { T } ^ { \dagger } ]$ would be bounded by $\frac { \mathbb { E } [ D _ { T } ^ { \dagger } ] } { \eta _ { T } }$ . As the policy sequence converges, the divergence $\mathbb { E } [ D _ { T } ^ { \dagger } ]$ only decays to a noise floor of $\mathcal { O } ( T ^ { - 2 / 3 } ) + \mathcal { O } ( \epsilon _ { a p p } )$ as shown in Lemma 9.6. Dividing this floor by the decaying step size $\eta _ { T } = \mathcal { O } ( T ^ { - 1 } )$ is catastrophic: the amplified noise contains components scaling as $T ^ { 1 / 3 }$ and $T \epsilon _ { a p p }$ , blowing up to infinity.

To achieve pointwise convergence, we abandon the $\frac { 1 } { \eta _ { t } }$ translation. Instead, we first prove that the divergence $\mathbb { E } [ D _ { T } ^ { \dagger } ]$ undergoes a geometric contraction to the noise floor via Chung’s Lemma $( \mathbb { E } [ D _ { T } ^ { \dagger } ] \leq \mathcal { O } ( T ^ { - 2 / 3 } ) + \mathcal { O } ( \epsilon _ { a p p } ) )$ , and subsequently translate this divergence back to the value gap via Pinsker’s inequality $( \mathbb { E } [ \mathrm { G a p } _ { T } ^ { \dag } ] \leq \mathcal { O } ( 1 ) \sqrt { \mathbb { E } [ D _ { T } ^ { \dag } ] } )$ . This sidesteps the $\eta _ { T } ^ { - 1 }$ amplification, but taking the square root slows the last-iterate regularized rate to $\mathcal { O } ( T ^ { - 1 / 3 } )$ .

## 10 Unregularized Global Convergence (Deterministic Regime)

We show that Algorithm 1 achieves an $\tilde { \mathcal { O } } ( T ^ { - 2 / 3 } )$ average-iterate and an $\tilde { \mathcal { O } } ( T ^ { - 1 / 3 } )$ last-iterate convergence rate for the unregularized performance gap. This is accomplished by applying the exponential translation mechanism alongside a two-stage temperature schedule, which balances the regularized optimization error against an exponentially decaying target tail to successfully bypass the standard linear entropy bottleneck.

## 10.1 Average-Iterate Unregularized Convergence

We now map the global regularized bounds to the unregularized objective. By utilizing the Universal Exponential Translation Bound (Corollary 3.1), we absorb the policy entropy into an exponentially small translation tail, allowing us to tune the fixed temperature τ to decrease much more slowly. By optimally balancing the regularized optimization error against this exponential term, we bypass the standard linear entropy bottleneck to extract an accelerated $\tilde { \mathcal { O } } ( T ^ { - 2 / 3 } )$ average-iterate convergence rate for the unregularized problem.

Theorem 10.1 (Unregularized Average-Iterate Convergence). For Algorithm 1, suppose that Assumptions 1–5 and 9 hold. Define the two-stage temperature:

$$
\tau _ { T } \triangleq \operatorname* { m a x } \left( \frac { \Delta } { 2 \log ( C _ { \gamma } T ^ { 2 / 3 } ) } , \frac { \Delta } { 2 \log ( C _ { \gamma } ( 1 + \epsilon _ { a p p } ^ { - 1 } ) ) } \right) .
$$

If $T$ is suficiently large and $\epsilon _ { a p p }$ is suficiently small such that $\tau _ { T } \le \tau _ { m a x }$ , then by choosing the step-size parameters as in Theorem 9.1 evaluated at $\tau = \tau _ { T }$ subject to the stated conditions, the expected average global unregularized performance gap satisfies the convergence rate:

$$
\begin{array} { r l } & { \displaystyle \frac { 1 } { T } \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } \big [ J _ { 0 } ( \pi _ { 0 } ^ { * } ) - J _ { 0 } ( \pi _ { t } ) \big ] \leq \mathcal { O } \left( \frac { ( 1 - \gamma ) ^ { - 5 / 3 } + ( \log T ) ^ { 5 / 3 } } { ( 1 - \gamma ) ^ { 8 } \kappa ^ { 6 } \Delta ^ { 8 / 3 } T ^ { 2 / 3 } } \right) } \\ & { \quad \quad \quad \quad \quad \quad + \mathcal { O } \left( \frac { ( 1 - \gamma ) ^ { - 1 } + \log ( 1 + \epsilon _ { a p p } ^ { - 1 } ) } { ( 1 - \gamma ) ^ { 2 } \Delta ^ { 2 } } \epsilon _ { a p p } \right) } \\ & { \quad \quad \quad \quad = \tilde { \mathcal { O } } \left( T ^ { - 2 / 3 } \right) + \tilde { \mathcal { O } } ( \epsilon _ { a p p } ) . } \end{array}
$$

In the case of $\epsilon _ { a p p } = 0$ , we interpret $\begin{array} { r } { \tau _ { T } = \frac { \Delta } { 2 \log ( C _ { \gamma } T ^ { 2 / 3 } ) } } \end{array}$ and $\epsilon _ { a p p } \log ( 1 + \epsilon _ { a p p } ^ { - 1 } ) = 0$

Remark 10.1 (Unification with Double-Loop Architectures). Balancing the average-iterate regularized convergence rate established in Theorem 9.1 with the standard linear entropy penalty leads to an $\mathcal { O } ( T ^ { - 1 / 4 } )$ average-iterate unregularized convergence rate for our single-loop algorithm. Interestingly, the same $\mathcal { O } ( T _ { t o t a l } ^ { - 1 / 4 } )$ rate was established by Agarwal et al. (2021) for a double-loop Q-NPG algorithm (see the extended discussion in Appendix D.1). However, their algorithm utilizes a coldstart inner critic, which requires a specific balancing of the outer-loop iterations T and inner-loop iterations N such that $N = T = T _ { t o t a l } ^ { 1 / 2 }$ to achieve this convergence rate.

To reconcile our single-loop algorithm $( N = 1 )$ with their double-loop configuration $( N \ =$ $T _ { t o t a l } ^ { 1 / 2 } )$ , we provide a unified analysis of a warm-start double-loop architecture in Appendix G. We demonstrate that by initializing the inner critic with the parameter from the previous outer loop, the convergence rate is no longer restricted to a single loop split. Under the standard linear entropy penalty, the warm-start analysis yields a uniform $\tilde { \mathcal { O } } ( T _ { t o t a l } ^ { - 1 / 4 } )$ convergence envelope across $N \in [ 1 , \sqrt { T _ { t o t a l } } ]$ , thereby connecting the single-loop and balanced double-loop endpoints. Under our Exponential Translation mechanism, the accelerated $\tilde { \mathcal { O } } ( T _ { t o t a l } ^ { - 2 / 3 } )$ convergence rate remains invariant across the window $N \in [ 1 , T _ { t o t a l } ^ { 2 / 9 } ]$

## 10.2 Last-Iterate Unregularized Convergence

We now extend the unregularized convergence bounds to the final output policy of Algorithm 1. By applying the Exponential Translation mechanism to the terminal policy using the two-stage temperature schedule as in Theorem 10.1, we demonstrate an $\tilde { \mathcal { O } } ( T ^ { - 1 / 3 } )$ last-iterate convergence rate for the unregularized problem, bypassing the standard linear entropy penalty.

Theorem 10.2 (Unregularized Last-Iterate Convergence). For Algorithm 1, suppose that Assumptions 1–5 and 9 hold. Consider the same two-stage temperature as in Theorem $1 0 . 1 \colon$

$$
\tau _ { T } \triangleq \operatorname* { m a x } \left( \frac { \Delta } { 2 \log ( C _ { \gamma } T ^ { 2 / 3 } ) } , \frac { \Delta } { 2 \log ( C _ { \gamma } ( 1 + \epsilon _ { a p p } ^ { - 1 } ) ) } \right) .
$$

$I f T$ is suficiently large and $\epsilon _ { a p p }$ is suficiently small such that $\tau _ { T } \leq \tau _ { m a x } ,$ then by choosing the step-size parameters as in Theorem 9.2 evaluated at $\tau = \tau _ { T }$ subject to the stated conditions, the expected last-iterate unregularized performance gap satisfies the convergence rate:

$$
\begin{array} { r l } & { \mathbb { E } \big [ J _ { 0 } ( \pi _ { 0 } ^ { * } ) - J _ { 0 } ( \pi _ { T } ) \big ] \leq \mathcal { O } \left( \frac { ( 1 - \gamma ) ^ { - 4 / 3 } + ( \log T ) ^ { 4 / 3 } } { ( 1 - \gamma ) ^ { 6 } \kappa ^ { 4 } \Delta ^ { 7 / 3 } T ^ { 1 / 3 } } \right) + \mathcal { O } \left( \frac { ( 1 - \gamma ) ^ { - 1 } + \log ( 1 + \epsilon _ { a p p } ^ { - 1 } ) } { ( 1 - \gamma ) ^ { 3 } \kappa \Delta ^ { 2 } } \sqrt { \epsilon _ { a p p } } \right) } \\ & { \qquad = \tilde { \mathcal { O } } \left( T ^ { - 1 / 3 } \right) + \tilde { \mathcal { O } } ( \sqrt { \epsilon _ { a p p } } ) . } \end{array}
$$

In the case $o f \epsilon _ { a p p } = 0$ , we interpret $\begin{array} { r } { \tau _ { T } = \frac { \Delta } { 2 \log ( C _ { \gamma } T ^ { 2 / 3 } ) } } \end{array}$ and $\sqrt { \epsilon _ { a p p } } \log ( 1 + \epsilon _ { a p p } ^ { - 1 } ) = 0$

## 10.3 Application to Tabular Setting

To illustrate the strength of our theory, we establish new unregularized convergence guarantees for the tabular setting. In this setting, both the state space S and the action space A are finite, and the log-linear policy uses the one-hot feature encoding $\phi ( s , a ) \in \mathbb { R } ^ { | S | | \mathcal { A } | }$ , i.e., the basis vector with 1 at the index corresponding to $( s , a )$ and 0 elsewhere. We synthesize the unregularized convergence bounds (Theorems 10.1 and 10.2) with the exploratory ν-sampling extension (Appendix B) and the concentrability-free $L _ { \infty }$ approximation analysis (Appendix C). We demonstrate that Algorithm 1 with a uniformly positive restart distribution ν achieves $\tilde { \mathcal { O } } ( T ^ { - 2 / 3 } )$ average-iterate and $\tilde { \mathcal { O } } ( T ^ { - 1 / 3 } )$ last-iterate convergence rates, under the Minimal Action Gap (Assumption 5) as the only primitive condition, but without relying on any concentrability assumptions.

Theorem 10.3 (Unregularized Convergence in the Tabular Setting). Consider a tabular MDP equipped with the one-hot feature representation. For Algorithm 1 with an exploratory restart distribution ν satisfying min $_ { s , a } \nu ( s , a ) \ > \ 0$ and the unbiased Q-value estimator $\hat { Q } _ { t }$ constructed via geometrically stopped Monte Carlo rollouts, suppose that the Minimal Action Gap (Assumption 5)

holds. Define $\kappa _ { e x p } \triangleq ( 1 - \gamma ) \operatorname* { m i n } _ { s , a } \nu ( s , a )$ and the T-dependent temperature

$$
\tau _ { T } \triangleq \frac { \Delta } { 2 \log ( C _ { \gamma } T ^ { 2 / 3 } ) } .
$$

If T is suficiently large such that $\tau _ { T } \ \leq \ \tau _ { m a x }$ , then by choosing the step-size parameters as in Theorem C.1, with κ replaced by $\kappa _ { e x p }$ and evaluated at $\tau = \tau _ { T }$ , the expected global unregularized performance gap satisfies:

$$
\frac { 1 } { T } \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } \big [ J _ { 0 } ( \pi _ { 0 } ^ { * } ) - J _ { 0 } ( \pi _ { t } ) \big ] \leq \mathcal { O } \left( \frac { ( 1 - \gamma ) ^ { - 5 / 3 } + ( \log T ) ^ { 5 / 3 } } { ( 1 - \gamma ) ^ { 8 } \kappa _ { e x p } ^ { 6 } \Delta ^ { 8 / 3 } } T ^ { - 2 / 3 } \right) ,
$$

$$
\mathbb { E } \left[ J _ { 0 } ( \pi _ { 0 } ^ { * } ) - J _ { 0 } ( \pi _ { T } ) \right] \leq { \cal O } \left( \frac { ( 1 - \gamma ) ^ { - 4 / 3 } + ( \log T ) ^ { 4 / 3 } } { ( 1 - \gamma ) ^ { 6 } \kappa _ { e x p } ^ { 4 } \Delta ^ { 7 / 3 } } T ^ { - 1 / 3 } \right) .
$$

Remark 10.2 (Breaking Tabular Stochastic Barrier). For unregularized tabular MDPs, standard sample-based NPG and PMD algorithms—such as those analyzed by Shani et al. (2020), Agarwal et al. (2021), and Lan (2023)—yield an $\mathcal { O } ( T _ { t o t a l } ^ { - 1 / 2 } )$ convergence rate (or $\mathcal { O } ( 1 / \epsilon ^ { 2 } )$ sample complex-$\mathrm { i t y ) }$ . While the minimax lower bound for sample-based tabular MDPs is known to be $\mathcal { O } ( T _ { t o t a l } ^ { - 1 / 2 } )$ by worst-case environments with vanishing margins (Azar et al., 2013), our analysis operates in the instance-dependent regime. By running the stochastic updates in the entropy-regularized space (extracting the restorative $- \tau D _ { t } ^ { \dagger }$ contraction) and exploiting a positive Minimal Action Gap $( \Delta > 0 )$ 2 our Exponential Translation mechanism bypasses this worst-case statistical barrier, mapping the regularized progress back to the unregularized objective to accelerate the average-iterate convergence rate to $\tilde { \mathcal { O } } ( T _ { t o t a l } ^ { - 2 / 3 } )$ . It remains an interesting open question whether the last-iterate rate can also be improved to $\tilde { \mathcal { O } } ( T _ { t o t a l } ^ { - 2 / 3 } )$ via alternative analytical techniques or algorithmic designs.

## References

Agarwal, A., Kakade, S. M., Lee, J. D., and Mahajan, G. (2021). On the theory of policy gradient methods: Optimality, approximation, and distribution shift. Journal of Machine Learning Research, 22:1–76.

Azar, M. G., Munos, R., and Kappen, H. J. (2013). Minimax pac bounds on the sample complexity of reinforcement learning with a generative model. Machine learning, 91:325–349.

Bhandari, J. and Russo, D. (2021). On the linear convergence of policy gradient methods for finite mdps. In International Conference on Artificial Intelligence and Statistics, pages 2386–2394.

Bhatnagar, S., Sutton, R. S., Ghavamzadeh, M., and Lee, M. (2009). Natural actor–critic algorithms. Automatica, 45:2471–2482.

Cayci, S., He, N., and Srikant, R. (2024). Convergence of entropy-regularized natural policy gradient with linear function approximation. SIAM Journal on Optimization, 34:2729–2755.

Cen, S., Cheng, C., Chen, Y., Wei, Y., and Chi, Y. (2022). Fast global convergence of natural policy gradient methods with entropy regularization. Operations Research, 70:2563–2578.

Chen, Z., Khodadadian, S., and Maguluri, S. T. (2022). Finite-sample analysis of of-policy natural actor–critic with linear function approximation. IEEE Control Systems Letters, 6:2611–2616.

Chung, K. L. (1954). On a stochastic approximation method. The Annals of Mathematical Statistics, 25:463–483.

Geist, M., Scherrer, B., and Pietquin, O. (2019). A theory of regularized markov decision processes. In International Conference on Machine Learning, pages 2160–2169.

Haarnoja, T., Zhou, A., Abbeel, P., and Levine, S. (2018). Soft actor-critic: Of-policy maximum entropy deep reinforcement learning with a stochastic actor. In International Conference on Machine Learning, pages 1861–1870.

Kakade, S. and Langford, J. (2002). Approximately optimal approximate reinforcement learning. In International Conference on Machine Learning, pages 267–274.

Kakade, S. M. (2001). A natural policy gradient. In Advances in Neural Information Processing Systems, pages 1531–1538.

Karimi, H., Nutini, J., and Schmidt, M. (2016). Linear convergence of gradient and proximalgradient methods under the polyak- lojasiewicz condition. In European Conference on Machine Learning and Principles and Practice of Knowledge Discovery in Databases, pages 795–811.

Khaled, A. and Richt´arik, P. (2023). Better theory for SGD in the nonconvex world. Transactions on Machine Learning Research.

Khodadadian, S., Chen, Z., and Maguluri, S. T. (2021). Finite-sample analysis of of-policy natural actor-critic algorithm. In International Conference on Machine Learning, pages 5420–5431.

Khodadadian, S., Doan, T. T., Romberg, J., and Maguluri, S. T. (2023). Finite-sample analysis of two-time-scale natural actor-critic algorithm. IEEE Transactions on Automatic Control, 68:3273– 3284.

Konda, V. R. and Tsitsiklis, J. N. (2000). Actor-critic algorithms. In Advances in Neural Information Processing Systems, pages 1008–1014.

Lan, G. (2023). Policy mirror descent for reinforcement learning: Linear convergence, new sampling complexity, and generalized problem classes. Mathematical programming, 198:1059–1106.

Liu, Y., Zhang, K., Ba¸sar, T., and Yin, W. (2020). An improved analysis of (variance-reduced) policy gradient and natural policy gradient methods. In Advances in Neural Information Processing Systems, pages 7624–7636.

Mei, J., Xiao, C., Szepesvari, C., and Schuurmans, D. (2020). Global convergence of policy gradient methods for the tabular markov decision process. In International Conference on Machine Learning, pages 6875–6887.

Mnih, V., Badia, A. P., Mirza, M., Graves, A., Lillicrap, T., Harley, T., Silver, D., and Kavukcuoglu, K. (2016). Asynchronous methods for deep reinforcement learning. In International Conference on Machine Learning, pages 1928–1937.

Neu, G., Jonsson, A., and G´omez, V. (2017). A unified view of entropy-regularized markov decision processes. arXiv preprint arXiv:1705.07798.

Peters, J., Mulling, K., and Altun, Y. (2010). Relative entropy policy search. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 24, pages 1607–1612.

Peters, J. and Schaal, S. (2008). Natural actor-critic. Neurocomputing, 71:1180–1190.

Schulman, J., Levine, S., Abbeel, P., Jordan, M., and Moritz, P. (2015). Trust region policy optimization. In International Conference on Machine Learning, pages 1889–1897.

Shani, L., Efroni, Y., and Mannor, S. (2020). Adaptive trust region policy optimization: Global convergence and faster rates for regularized mdps. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 34, pages 5668–5675.

Sutton, R. S. and Barto, A. G. (2018). Reinforcement Learning: An Introduction. MIT Press.

Sutton, R. S., McAllester, D., Singh, S., and Mansour, Y. (1999). Policy gradient methods for reinforcement learning with function approximation. In Advances in Neural Information Processing Systems, pages 1057–1063.

Szepesv´ari, C. (2010). Algorithms for Reinforcement Learning. Springer.

Wang, Y., Wang, Y., Zhou, Y., and Zou, S. (2024). Non-asymptotic analysis for single-loop (natural) actor-critic with compatible function approximation. In International Conference on Machine Learning, volume 235, pages 51771–51824.

Williams, R. J. and Peng, J. (1991). Function optimization using connectionist reinforcement learning algorithms. Connection Science, 3:241–268.

Wu, Y., Zhang, W., Xu, P., and Gu, Q. (2020). A finite-time analysis of two time-scale actor-critic methods. In Advances in Neural Information Processing Systems, pages 17617–17628.

Xu, T., Wang, Z., and Liang, Y. (2020). Improving sample complexity bounds for (natural) actorcritic algorithms. In Advances in Neural Information Processing Systems, pages 4358–4369.

Yuan, R., Du, S. S., Gower, R. M., Lazaric, A., and Xiao, L. (2023). Linear convergence of natural policy gradient methods with log-linear policies. In International Conference on Learning Representations.

## A Algorithm Comparison

In this section, we provide a detailed mathematical comparison between our uncentered algorithm and standard entropy-regularized Natural Policy Gradient (NPG) algorithms operating with compatible function approximation. Specifically, we contrast our approach with the standard advantage projection and the $Q \mathrm { - }$ value projection onto centered features in the entropy-regularized NPG with averaging studied by Cayci et al. (2024).

## A.1 Standard NPG with Advantage-Driven, Centered Critic

For a log-linear policy $\pi _ { \omega _ { t } }$ ∝ $\exp ( \omega _ { t } ^ { \top } \phi )$ , the score function corresponds exactly to the centered feature vector $\tilde { \phi } _ { t } ( s , a ) \triangleq \phi ( s , a ) - \mathbb { E } _ { a ^ { \prime } \sim \pi _ { t } ( \cdot | s ) } [ \phi ( s , a ^ { \prime } ) ]$ . For compatible function approximation, the standard approach evaluates the update direction by finding the optimal parameter $\theta _ { \tau } ^ { * A } ( \omega _ { t } )$ that minimizes the mean squared error between the true regularized advantage and the centered feature approximation:

$$
\theta _ { \tau } ^ { * A } ( \omega _ { t } ) \triangleq \arg \operatorname* { m i n } _ { \theta \in \mathbb { R } ^ { d } } \mathbb { E } _ { s \sim d ^ { \pi _ { t } } , a \sim \pi _ { t } } \left[ \left( A _ { \tau } ^ { \pi _ { t } } ( s , a ) - \theta ^ { \top } \tilde { \phi } _ { t } ( s , a ) \right) ^ { 2 } \right] .
$$

Because the state-dependent baseline $V _ { \tau } ^ { \pi _ { t } } ( s )$ is orthogonal to the centered features $\tilde { \phi } _ { t } ( s , a )$ under the action expectation, this projection can be equivalently defined directly without the state-value function:

$$
\theta _ { \tau } ^ { * A } ( \omega _ { t } ) = \arg \operatorname* { m i n } _ { \theta \in \mathbb { R } ^ { d } } \mathbb { E } _ { s \sim d ^ { \pi _ { t } } , a \sim \pi _ { t } } \left[ \left( Q _ { \tau } ^ { \pi _ { t } } ( s , a ) - \tau \log \pi _ { t } ( a | s ) - \theta ^ { \top } \tilde { \phi } _ { t } ( s , a ) \right) ^ { 2 } \right] .\tag{6}
$$

Throughout our discussion, when the relevant least-squares minimizers are not unique, the stated parameter equalities should be interpreted as corresponding relations between the minimizer sets (or the induced fitted functions).

By the Policy Gradient Theorem, the exact Euclidean gradient of the entropy-regularized objective evaluates to (Sutton et al., 1999):

$$
\nabla J _ { \tau } ( \omega _ { t } ) = \frac { 1 } { 1 - \gamma } \bar { \Sigma } _ { c e n } ( \pi _ { t } ) \theta _ { \tau } ^ { * A } ( \omega _ { t } ) ,\tag{7}
$$

where $\bar { \Sigma } _ { c e n } ( \pi _ { t } ) = \mathbb { E } _ { s \sim d ^ { \pi _ { t } } , a \sim \pi _ { t } } \big [ \tilde { \phi } _ { t } ( s , a ) \tilde { \phi } _ { t } ( s , a ) ^ { \top } \big ]$ is the Fisher information matrix (the centered feature covariance). By preconditioning, the natural policy gradient is $\bar { \Sigma } _ { c e n } ( \pi _ { t } ) ^ { - 1 } \nabla J _ { \tau } ( \omega _ { t } ) \propto \theta _ { \tau } ^ { * A } ( \omega _ { t } )$ if the Fisher information matrix is invertible (or a generalized inverse is used if not invertible). Thus, the standard NPG parameter update is defined directly by this advantage-driven, centered critic (Kakade, 2001; Peters and Schaal, 2008):

$$
\omega _ { t + 1 } = \omega _ { t } + \eta _ { t } \theta _ { \tau } ^ { * A } ( \omega _ { t } ) .
$$

## A.2 NPG with Q-Value-Driven, Centered Critic

A closely related variant projects the regularized action-values directly onto the centered features. We define the Q-value-driven, centered critic as:

$$
\theta _ { \tau } ^ { * Q } ( \omega _ { t } ) \triangleq \arg \operatorname* { m i n } _ { \theta \in \mathbb { R } ^ { d } } \mathbb { E } _ { s \sim d ^ { \pi _ { t } } , a \sim \pi _ { t } } \left[ \left( Q _ { \tau } ^ { \pi _ { t } } ( s , a ) - \theta ^ { \top } \tilde { \phi } _ { t } ( s , a ) \right) ^ { 2 } \right] .
$$

Because any state-dependent mean $\mathbb { E } _ { a ^ { \prime } } [ Q _ { \tau } ^ { \pi _ { t } } ( s , a ^ { \prime } ) ]$ is orthogonal to $\tilde { \phi } _ { t } ( s , a )$ , projecting the uncentered Q-values is identical to projecting the centered Q-values $( \tilde { Q } _ { \tau } ^ { \pi _ { t } } ( s , a ) \triangleq Q _ { \tau } ^ { \pi _ { t } } ( s , a ) \mathrm { ~ - ~ }$ $\mathbb { E } _ { a ^ { \prime } } [ Q _ { \tau } ^ { \pi _ { t } } ( s , a ^ { \prime } ) ] )$ in the entropy-regularized NPG with averaging analyzed by Cayci et al. (2024):

$$
\boldsymbol { \theta } _ { \tau } ^ { \ast C H S } ( \omega _ { t } ) \triangleq \arg \operatorname* { m i n } _ { \boldsymbol { \theta } \in \mathbb { R } ^ { d } } \mathbb { E } _ { \boldsymbol { s } \sim \boldsymbol { d } ^ { \pi _ { t } } , \boldsymbol { a } \sim \pi _ { t } } \left[ \left( \tilde { Q } _ { \tau } ^ { \pi _ { t } } ( \boldsymbol { s } , \boldsymbol { a } ) - \boldsymbol { \theta } ^ { \top } \tilde { \phi } _ { t } ( \boldsymbol { s } , \boldsymbol { a } ) \right) ^ { 2 } \right] .
$$

Therefore, $\theta _ { \tau } ^ { * Q } ( \omega _ { t } ) = \theta _ { \tau } ^ { * C H S } ( \omega _ { t } )$ , even in the presence of approximation error.

Next, we substitute the expression τ log $\pi _ { t } ( a | s ) = \tau \omega _ { t } ^ { \top } \phi ( s , a ) - \tau \log Z _ { t } ( s )$ into the equivalent advantage projection in (6). The orthogonality of log $Z _ { t } ( s )$ with the centered features $\tilde { \phi } _ { t } ( s , a )$ reveals that projecting the entropy penalty onto $\tilde { \phi } _ { t } ( s , a )$ yields exactly $- \tau \omega _ { t }$ . By the linearity of the least-squares projection, this establishes a geometric relationship between the advantage and Q-value projections (onto centered features):

$$
\theta _ { \tau } ^ { * A } ( \omega _ { t } ) = \theta _ { \tau } ^ { * Q } ( \omega _ { t } ) - \tau \omega _ { t } .\tag{8}
$$

Consequently, the standard NPG update can be equivalently written in terms of the centered Q-value projection as:

$$
\omega _ { t + 1 } = \omega _ { t } + \eta _ { t } \big ( \theta _ { \tau } ^ { * Q } ( \omega _ { t } ) - \tau \omega _ { t } \big ) .
$$

## A.3 Equivalence under Exact Parameterization

While the centered critic parameters maintain algebraic relationships with each other unconditionally, they can be related to our uncentered critic $\theta _ { \tau } ^ { * } ( \omega _ { t } )$ under the idealized assumption of zero approximation error.

Lemma A.1 (Critic Equivalence under Exact Parameterization). Assume there is no approximation error in the uncentered critic, such that $Q _ { \tau } ^ { \pi _ { t } } ( s , a ) = \theta _ { \tau } ^ { * } ( \omega _ { t } ) ^ { \top } \phi ( s , a )$ holds exactly for all states and actions. Then the centered projections satisfy:

$$
\theta _ { \tau } ^ { * Q } ( \omega _ { t } ) = \theta _ { \tau } ^ { * C H S } ( \omega _ { t } ) = \theta _ { \tau } ^ { * } ( \omega _ { t } ) \quad a n d \quad \theta _ { \tau } ^ { * A } ( \omega _ { t } ) = \theta _ { \tau } ^ { * } ( \omega _ { t } ) - \tau \omega _ { t } .
$$

Proof. Substituting the exact parameterization into the centered Q-value yields:

$$
\tilde { Q } _ { \tau } ^ { \pi _ { t } } ( s , a ) = \theta _ { \tau } ^ { * } ( \omega _ { t } ) ^ { \top } \phi ( s , a ) - \mathbb { E } _ { a ^ { \prime } \sim \pi _ { t } } \big [ \theta _ { \tau } ^ { * } ( \omega _ { t } ) ^ { \top } \phi ( s , a ^ { \prime } ) \big ]
$$

$$
\begin{array} { r l } & { = \theta _ { \tau } ^ { * } ( \omega _ { t } ) ^ { \top } \big ( \phi ( s , a ) - \mathbb { E } _ { a ^ { \prime } \sim \pi _ { t } } [ \phi ( s , a ^ { \prime } ) ] \big ) } \\ & { } \\ & { = \theta _ { \tau } ^ { * } ( \omega _ { t } ) ^ { \top } \tilde { \phi } _ { t } ( s , a ) . } \end{array}
$$

Because $\tilde { Q } _ { \tau } ^ { \pi _ { t } } ( s , a )$ is perfectly captured by the centered features with parameter $\theta _ { \tau } ^ { * } ( \omega _ { t } )$ , the projection trivially yields $\theta _ { \tau } ^ { * C H S } ( \omega _ { t } ) = \theta _ { \tau } ^ { * } ( \omega _ { t } )$ . As established in the text, $\theta _ { \tau } ^ { * Q } ( \omega _ { t } ) = \theta _ { \tau } ^ { * C H S } ( \omega _ { t } )$ and $\theta _ { \tau } ^ { * A } ( \omega _ { t } ) = \theta _ { \tau } ^ { * Q } ( \omega _ { t } ) - \tau \omega _ { t }$ . Substituting these identities yields $\theta _ { \tau } ^ { * Q } ( \omega _ { t } ) = \theta _ { \tau } ^ { * } ( \omega _ { t } )$ and $\theta _ { \tau } ^ { * A } ( \omega _ { t } ) =$ $\theta _ { \tau } ^ { * } ( \omega _ { t } ) - \tau \omega _ { t }$ , completing the proof. □

It is important to note that this exact parameter-space equivalence between the uncentered and centered critics relies on the absence of approximation error. In general, under compatible function approximation with non-zero approximation error, this equivalence does not hold, and the respective critic parameters may difer because they solve diferent projection problems.

## A.4 Algorithmic Design and Comparison

The equivalence relation $\theta _ { \tau } ^ { * A } ( \omega _ { t } ) = \theta _ { \tau } ^ { * } ( \omega _ { t } ) - \tau \omega _ { t }$ holds exactly under exact parameterization. Motivated by this idealized equivalence, our algorithm explicitly chooses to define the parameter update using the uncentered critic parameter directly:

$$
\omega _ { t + 1 } = \omega _ { t } + \eta _ { t } \big ( \theta _ { \tau } ^ { * } ( \omega _ { t } ) - \tau \omega _ { t } \big ) .
$$

In practice, operating with the uncentered critic parameter $\theta _ { \tau } ^ { * } ( \omega _ { t } )$ incurs a structural approx imation bias relative to the exact NPG direction. However, as we established in the Uncentered Gradient Identity (Lemma 5.1), the true policy gradient decomposes as:

$$
\nabla J _ { \tau } ( \omega _ { t } ) = \frac { 1 } { 1 - \gamma } \bar { \Sigma } _ { c e n } ( \pi _ { t } ) \big ( \theta _ { \tau } ^ { * } ( \omega _ { t } ) - \tau \omega _ { t } \big ) + \frac { 1 } { 1 - \gamma } E _ { b i a s } .
$$

Multiplying by the inverse Fisher information matrix $\bar { \Sigma } _ { c e n } ( \pi _ { t } ) ^ { - 1 }$ reveals that our chosen update direction, $\theta _ { \tau } ^ { * } ( \omega _ { t } ) - \tau \omega _ { t }$ , difers from the scaled exact natural gradient by the residual term $\bar { \Sigma } _ { c e n } ( \pi _ { t } ) ^ { - 1 } E _ { b i a s }$ , which is bounded by the approximation error of the uncentered representation $\theta _ { \tau } ^ { * } ( \omega _ { t } ) ^ { \top } \phi ( s , a )$ in the Actor Progress Bound (Lemma 5.3).

Remark A.1 (Centered vs. Uncentered Critic). Our analysis in Part I can be readily extended to a centered NAC algorithm (e.g., utilizing the centered Q-value critic $\theta _ { \tau } ^ { * Q } )$ . By using the centered critic, the policy gradient evaluates to the exact natural policy gradient update direction without structural bias (from (7) and (8)):

$$
\nabla J _ { \tau } ( \omega _ { t } ) = \frac { 1 } { 1 - \gamma } \bar { \Sigma } _ { c e n } ( \pi _ { t } ) \big ( \theta _ { \tau } ^ { * Q } ( \omega _ { t } ) - \tau \omega _ { t } \big ) .
$$

Consequently, the Actor Progress Bound can be established free of the $\epsilon _ { a p p }$ bias term. However, the corresponding PL condition still involves an $\epsilon _ { a p p }$ term. Therefore, the resulting Coupled Actor Recurrence inequality retains a similar $\epsilon _ { a p p }$ dependence, yielding similar convergence rates to those of the uncentered NAC algorithm in the Stochastic Regime.

In contrast, the centered and uncentered formulations behave diferently in the Deterministic Regime considered in Part II. As the temperature drops $( \tau  0 )$ , the deterministic collapse forces the minimum eigenvalue of the centered feature covariance to vanish $( \lambda _ { \tau }  0 )$ , eliminating the uniform positive-definiteness needed to control the centered least-squares projection. Consequently, the centered NAC formulation would not be covered by our Part II analysis under the current assumptions. Our use of the uncentered critic circumvents this degenerating geometry by relying instead on the Positive-Definite Uncentered Feature Moment (Assumption 3), while introducing a residual term that can be controlled by the critic approximation error.

## B Generalization to Exploratory ν-Sampling

In the main text, Algorithm 1 generates trajectories starting from the objective’s initial state distribution $\mu ,$ , efectively sampling from the on-policy visitation measure. In this section, we generalize our theoretical framework to accommodate algorithms that utilize an exploratory initial state-action restart distribution, $\nu ( s , a )$ , a standard mechanism of using of-policy data to improve action-space exploration for the critic (Agarwal et al., 2021; Yuan et al., 2023).

## B.1 Generalized Sampling Measure and Critic Assumptions

Under generalized ν-sampling, at each iteration $t ,$ Algorithm 1 draws an initial state-action pair $( s _ { 0 } , a _ { 0 } ) \sim \nu ,$ executes $a _ { 0 }$ , and thereafter follows the current policy $\pi _ { t }$ . This induces the generalized discounted training measure:

$$
d _ { \nu } ^ { \pi _ { t } } \big ( s , a \big ) \triangleq \big ( 1 - \gamma \big ) \sum _ { k = 0 } ^ { \infty } \gamma ^ { k } \mathbb { P } \big ( s _ { k } = s , a _ { k } = a \mid \big ( s _ { 0 } , a _ { 0 } \big ) \sim \nu , a _ { 1 : k } \sim \pi _ { t } \big ) .
$$

Because all subsequent probabilities in the geometric sum are non-negative, this definition provides a simple, policy-independent lower bound on state-action coverage by the first step: $d _ { \nu } ^ { \pi _ { t } } ( s , a ) \geq ( 1 - \gamma ) \nu ( s , a )$ . In the on-policy specialization where the restart distribution at iteration t is $\nu _ { t } ( s , a ) = \mu ( s ) \pi _ { t } ( a | s )$ , the training measure reduces to the standard on-policy distribution: $d _ { \nu _ { t } } ^ { \pi _ { t } } ( s , a ) = d ^ { \pi _ { t } } ( s ) \pi _ { t } ( a | s )$

Because the sample-based critic is trained via SGD over trajectories drawn from $d _ { \nu } ^ { \pi _ { t } }$ , it natively minimizes the least-squares error weighted by this specific distribution. Consequently, the ideal tracking target is redefined to align with the training measure:

$$
\theta _ { \tau } ^ { * } ( \omega _ { t } ) \triangleq \arg \operatorname* { m i n } _ { \theta \in \mathbb { R } ^ { d } } \mathbb { E } _ { ( s , a ) \sim d _ { \nu } ^ { \pi _ { t } } } \left[ \left( Q _ { \tau } ^ { \pi _ { t } } ( s , a ) - \theta ^ { \top } \phi ( s , a ) \right) ^ { 2 } \right] .
$$

By the first-order optimality condition, the approximation error $\epsilon _ { t } ( s , a ) \triangleq Q _ { \tau } ^ { \pi _ { t } } ( s , a ) - \theta _ { \tau } ^ { * } ( \omega _ { t } ) ^ { \top } \phi ( s , a )$ is orthogonal to the features under the generalized training measure: $\mathbb { E } _ { d _ { \nu } ^ { \pi _ { t } } } [ \phi \epsilon _ { t } ] = 0$ . The critic tracking error is defined as before, $e _ { t } \triangleq \theta _ { t + 1 } - \theta _ { \tau } ^ { * } ( \omega _ { t } )$ , but now relative to the redefined ideal critic.

By the critic’s new training measure, the following assumption replaces the Critic $L _ { 2 }$ Approximation Error (Assumption 4) from the main text.

Assumption 11 (Critic $L _ { 2 }$ Approximation Error under Exploratory Training Measure). The MSE of the ideal uncentered critic under the exploratory training measure is bounded by $\epsilon _ { a p p }$ for all training policies $\{ \pi _ { t } \} _ { t \ge 0 }$ under Algorithm 1:

$$
\begin{array} { r } { \mathbb { E } _ { d _ { \nu } ^ { \pi _ { t } } } [ \epsilon _ { t } ( s , a ) ^ { 2 } ] \le \epsilon _ { a p p } . } \end{array}
$$

Additionally, for the global analysis in Part I, we assume the same bound holds at the regularized parametric optimum:

$$
\mathbb { E } _ { d _ { \nu } ^ { \omega _ { \tau } ^ { * } } } [ \epsilon _ { * } ( s , a ) ^ { 2 } ] \le \epsilon _ { a p p } ,
$$

where $d _ { \nu } ^ { \omega _ { \tau } ^ { * } } \triangleq d _ { \nu } ^ { \pi _ { \omega _ { \tau } ^ { * } } }$ and $\epsilon _ { * } ( s , a ) \triangleq Q _ { \tau } ^ { \pi _ { \omega _ { \tau } ^ { * } } } ( s , a ) - \theta _ { \tau } ^ { * } ( \omega _ { \tau } ^ { * } ) ^ { \top } \phi ( s , a )$ . As in Assumption $^ { 4 , }$ we assume $\epsilon _ { a p p }$ is independent of $\tau \in ( 0 , \tau _ { m a x } ]$

For any state-action distribution $d ,$ such as ν and $d _ { \nu } ^ { \pi _ { t } }$ , we define the uncentered feature moment matrices evaluated under d as:

$$
\bar { \Sigma } _ { u n c } ( d ) \triangleq \mathbb { E } _ { ( s , a ) \sim d } \big [ \phi ( s , a ) \phi ( s , a ) ^ { \top } \big ] .
$$

The following assumption replaces the Positive-Definite Uncentered Feature Moment (Assumption 3) from the main text. Note that the Positive-Definite Fisher Information (Assumption 6) remains necessary in Part I, as the actor’s update is governed by the objective’s on-policy distribution and unafected by the critic’s exploratory training measure.

Assumption 12 (Positive-Definite Moment under Exploratory Training Measure). We assume that the uncentered feature moment matrix evaluated under the training measure is uniformly positive-definite. Specifically, there exists a constant $\kappa _ { e x p } > 0 ,$ , independent of the temperature $\tau \in ( 0 , \tau _ { m a x } ]$ , such that for all $t \geq 0$ under Algorithm 1:

$$
\begin{array} { r } { \bar { \Sigma } _ { u n c } ( d _ { \nu } ^ { \pi _ { t } } ) \succeq \kappa _ { e x p } I . } \end{array}
$$

Because $d _ { \nu } ^ { \pi _ { t } } \geq ( 1 - \gamma ) \nu$ , this assumption holds natively if the restart distribution itself supports the feature space, i.e., $\bar { \Sigma } _ { u n c } ( \nu ) \succeq \kappa _ { 0 } I .$ , yielding $\kappa _ { e x p } = ( 1 - \gamma ) \kappa _ { 0 }$

Impact of $\kappa _ { e x p }$ on Structural Bounds: Because the transition to ν-sampling modifies the underlying training measure, we briefly note its impact on the structural bounds:

(i) Critic and Actor Bounds: In the main text, the bound on the ideal critic is $\begin{array} { r } { B _ { \theta } = \frac { B _ { \phi } V _ { m a x } } { \kappa } } \end{array}$ Under ν-sampling, this definition updates to $\frac { B _ { \phi } V _ { m a x } } { \kappa _ { e x p } }$ . The actor gradient bound, $G _ { a c } = 2 B _ { \theta }$ changes accordingly. Both bounds remain finite, temperature-independent constants.

(ii) Lipschitz Critic $\left( L _ { \theta } \right) :$ The general derivative bound (20) used in the proof of Lemma 2.3 continues to hold when the on-policy measure is replaced by $d _ { \nu } ^ { \pi _ { \omega } }$ . Indeed, the restart distribution ν is independent of $\omega ,$ and hence only the subsequent policy-dependent actions contribute score-function terms, whose discounted sum is bounded by the same $\frac { 2 B _ { \phi } } { 1 - \gamma }$ factor. Therefore, the ideal critic $\theta _ { \tau } ^ { * } ( \omega _ { t } )$ remains Lipschitz continuous as in Lemma 2.3, with κ replaced by $\kappa _ { e x p }$ in the Lipschitz constant $L _ { \theta , }$ <sub>τ</sub> and $L _ { \theta } \triangleq L _ { \theta , \tau _ { m a x } }$

## B.2 Generalized Of-Policy Mismatches

Under exploratory ν-sampling, the critic approximation error is controlled under the training measure $d _ { \nu } ^ { \pi _ { t } }$ , whereas the actor progress and comparator-based performance bounds are evaluated under diferent state-action measures. We therefore introduce generalized of-policy mismatch conditions to control the resulting changes of measure.

Assumption 13 (Generalized Of-Policy Mismatches). We assume the following density ratios are bounded by finite constants independent of the temperature $\tau \in ( 0 , \tau _ { m a x } ]$ :

(i) Training and Optimum Of-Policy Mismatch (For Part I):

$$
\operatorname* { s u p } _ { t \geq 0 } \operatorname* { s u p } _ { s , a } \frac { d ^ { \pi _ { t } } ( s ) \pi _ { t } ( a \vert s ) } { d _ { \nu } ^ { \pi _ { t } } ( s , a ) } \leq C _ { \mu , \nu } , \qquad \operatorname* { s u p } _ { s , a } \frac { d ^ { \omega _ { \tau } ^ { * } } ( s ) \pi _ { \omega _ { \tau } ^ { * } } ( a \vert s ) } { d _ { \nu } ^ { \omega _ { \tau } ^ { * } } ( s , a ) } \leq C _ { \mu , \nu } .
$$

(ii) Parametric Of-Policy Joint Mismatch (For Part I):

$$
\operatorname* { s u p } _ { t \geq 0 } \operatorname* { s u p } _ { s , a } \left[ \frac { d ^ { \omega _ { \tau } ^ { * } } ( s ) \pi _ { t } ( a | s ) } { d _ { \nu } ^ { \pi _ { t } } ( s , a ) } \operatorname* { m a x } _ { a ^ { \prime } } \frac { \pi _ { \omega _ { \tau } ^ { * } } ( a ^ { \prime } | s ) } { \pi _ { t } ( a ^ { \prime } | s ) } \right] \leq C _ { j o i n t , \nu } .
$$

(iii) Global-to-Parametric Of-Policy Joint Mismatch (For Part I):

$$
\operatorname* { s u p } _ { s , a } \left[ \frac { d _ { \tau } ^ { \ast } ( s ) \pi _ { \omega _ { \tau } ^ { \ast } } ( a | s ) } { d _ { \nu } ^ { \omega _ { \tau } ^ { \ast } } ( s , a ) } \operatorname* { m a x } _ { a ^ { \prime } } \frac { \pi _ { \tau } ^ { \ast } ( a ^ { \prime } | s ) } { \pi _ { \omega _ { \tau } ^ { \ast } } ( a ^ { \prime } | s ) } \right] \leq C _ { j o i n t , \nu } ^ { \ast } .
$$

(iv) Global Of-Policy Joint Mismatch (For Part II):

$$
\operatorname* { s u p } _ { t \geq 0 } \operatorname* { s u p } _ { s , a } \left[ \frac { d _ { \tau } ^ { \ast } ( s ) \pi _ { t } ( a \vert s ) } { d _ { \nu } ^ { \pi _ { t } } ( s , a ) } \operatorname* { m a x } _ { a ^ { \prime } } \frac { \pi _ { \tau } ^ { \ast } ( a ^ { \prime } \vert s ) } { \pi _ { t } ( a ^ { \prime } \vert s ) } \right] \leq C _ { j o i n t , \nu } ^ { \dagger } .
$$

In Part I, the of-policy mismatch bounds $C _ { \mu , \nu } , \ C _ { j o i n t , \nu } .$ , and $C _ { j o i n t , \nu } ^ { * }$ operate alongside the parametric joint concentrability bound $C _ { j o i n t }$ (Assumption 7). In Part II, the of-policy mismatch bound $C _ { j o i n t , \nu } ^ { \dagger }$ generalizes the global joint concentrability bound $C _ { j o i n t } ^ { \dagger }$ (Assumption 9). We make three comments on the generalized of-policy mismatches.

Reduction to the Main Text. Our main-text setting is recovered by choosing the on-policy restart distribution $\nu _ { t } ( s , a ) = \mu ( s ) \pi _ { t } ( a | s )$ for each training policy and, at the parametric optimum, $\nu _ { \tau } ^ { \ast } ( s , a ) = \mu ( s ) \pi _ { \omega _ { \tau } ^ { \ast } } ( a | s )$ . Then $d _ { \nu _ { t } } ^ { \pi _ { t } } ( s , a ) = d ^ { \pi _ { t } } ( s ) \pi _ { t } ( a | s )$ and $d _ { \nu _ { \tau } ^ { * } } ^ { \pi _ { \omega _ { \tau } ^ { * } } } ( s , a ) = d ^ { \omega _ { \tau } ^ { * } } ( s ) \pi _ { \omega _ { \tau } ^ { * } } ( a | s )$ Consequently, the Training and Optimum Of-Policy Mismatch holds with $C _ { \mu , \nu } = 1$ , while the Parametric, Global-to-Parametric, and Global Of-Policy Joint Mismatches reduce to our original joint concentrability bounds $C _ { j o i n t } , C _ { j o i n t } ^ { * }$ , and $C _ { j o i n t } ^ { \dagger }$ in Assumptions 7, 8, and 9 respectively.

Relation to Separate Mismatch Bounds: Similarly to our discussions of Assumption 7 and 9 in the main text, the parametric of-policy joint mismatch (Assumption $1 3 ( \mathrm { i i } ) )$ is naturally implied by the conjunction of separate of-policy mismatch and policy concentrability bounds. Specifically, if the following holds for some constants $( C _ { \nu } , W )$ independent of $\tau \in ( 0 , \tau _ { m a x } ]$

• Parametric Of-Policy Mismatch: $\begin{array} { r } { \operatorname* { s u p } _ { s , a } \frac { d ^ { \omega _ { \tau } ^ { * } } ( s ) \pi _ { t } ( a | s ) } { d _ { \nu } ^ { \pi _ { t } } ( s , a ) } \leq C _ { \nu } . } \end{array}$

• Parametric Policy Concentrability: $\begin{array} { r } { \operatorname* { s u p } _ { s , a } \frac { \pi _ { \omega _ { \tau } ^ { * } } ( a | s ) } { \pi _ { t } ( a | s ) } \leq W , } \end{array}$

then by the multiplicative rule, the joint bound is satisfied with $C _ { j o i n t , \nu } = C _ { \nu } W$ . Conversely, because $\begin{array} { r } { \mathbb { E } _ { a ^ { \prime } \sim \pi _ { t } } \Big [ \frac { \pi _ { \omega _ { \tau } ^ { * } } ( a ^ { \prime } | s ) } { \pi _ { t } ( a ^ { \prime } | s ) } \Big ] = 1 } \end{array}$ , Assumption 13(ii) implies the parametric of-policy mismatch bound with $C _ { \nu } = C _ { j o i n t , \nu } \mathrm { : }$

$$
\operatorname* { s u p } _ { s , a } \frac { d ^ { \omega _ { \tau } ^ { * } } ( s ) \pi _ { t } ( a | s ) } { d _ { \nu } ^ { \pi _ { t } } ( s , a ) } \leq C _ { j o i n t , \nu } .
$$

Furthermore, by taking $a ^ { \prime } = a$ inside the supremum, Assumption 13(ii) also implies:

$$
\operatorname* { s u p } _ { s , a } \frac { d ^ { \omega _ { \tau } ^ { * } } ( s ) \pi _ { \omega _ { \tau } ^ { * } } ( a \vert s ) } { d _ { \nu } ^ { \pi _ { t } } ( s , a ) } \leq C _ { j o i n t , \nu } .
$$

Identical relations hold for the global-to-parametric of-policy joint mismatch (Assumption 13(iii)) and for the global of-policy joint mismatch (Assumption 13(iv)). In particular, the latter joint bound is satisfied with $C _ { j o i n t , \nu } ^ { \dagger } = C _ { \nu } ^ { \dagger } W ^ { \dagger }$ under the conjunction of the following density ratio bounds for some constants $( C _ { \nu } ^ { \dagger } , W ^ { \dagger } )$ independent of $\tau \in ( 0 , \tau _ { m a x } ]$

• Global Of-Policy Mismatch: $\begin{array} { r } { \operatorname* { s u p } _ { s , a } \frac { d _ { \tau } ^ { * } ( s ) \pi _ { t } ( a \vert s ) } { d _ { \nu } ^ { \pi _ { t } } ( s , a ) } \le C _ { \nu } ^ { \dagger } , } \end{array}$

• Global Policy Concentrability: $\begin{array} { r } { \operatorname* { s u p } _ { s , a } \frac { \pi _ { \tau } ^ { * } ( a | s ) } { \pi _ { t } ( a | s ) } \le W ^ { \dagger } } \end{array}$

Conversely, similar to the parametric case, Assumption 13(iv) implies the global of-policy mismatch bound with $\begin{array} { r } { C _ { \nu } ^ { \dagger } = C _ { j o i n t , \nu } ^ { \dagger } \mathrm { ; } } \end{array}$

$$
\operatorname* { s u p } _ { s , a } \frac { d _ { \tau } ^ { \ast } ( s ) \pi _ { t } ( a | s ) } { d _ { \nu } ^ { \pi _ { t } } ( s , a ) } \leq C _ { j o i n t , \nu } ^ { \dagger } .\tag{9}
$$

By taking $a ^ { \prime } = a .$ , Assumption 13(iv) also implies:

$$
\operatorname* { s u p } _ { s , a } \frac { d _ { \tau } ^ { \ast } ( s ) \pi _ { \tau } ^ { \ast } ( a | s ) } { d _ { \nu } ^ { \pi _ { t } } ( s , a ) } \leq C _ { j o i n t , \nu } ^ { \dagger } .\tag{10}
$$

Elimination of Independent Policy Concentrability: By embedding the local policy mismatches

$$
W ( s ) \triangleq \operatorname* { m a x } _ { a ^ { \prime } } \frac { \pi _ { \omega _ { \tau } ^ { * } } ( a ^ { \prime } | s ) } { \pi _ { t } ( a ^ { \prime } | s ) } , \qquad W ^ { * } ( s ) \triangleq \operatorname* { m a x } _ { a ^ { \prime } } \frac { \pi _ { \tau } ^ { * } ( a ^ { \prime } | s ) } { \pi _ { \omega _ { \tau } ^ { * } } ( a ^ { \prime } | s ) } , \qquad W ^ { \dagger } ( s ) \triangleq \operatorname* { m a x } _ { a ^ { \prime } } \frac { \pi _ { \tau } ^ { * } ( a ^ { \prime } | s ) } { \pi _ { t } ( a ^ { \prime } | s ) } .
$$

into the generalized joint mismatch assumptions $C _ { j o i n t , \nu } , \ C _ { j o i n t , \nu } ^ { * } ,$ and $C _ { j o i n t , \nu } ^ { \dagger }$ respectively, we avoid requiring separate policy concentrability bounds, for example W and $W ^ { \dagger }$ , under exploratory ν-sampling. This mirrors the consolidation in the main text: the local policy ratios themselves need not be uniformly bounded, provided the evaluation measures, such as $d ^ { \omega _ { \tau } ^ { * } } ( s ) \pi _ { t } ( a | s )$ and $d _ { \tau } ^ { * } ( s ) \pi _ { t } ( a | s )$ are suficiently small relative to the exploratory training measure $d _ { \nu } ^ { \pi _ { t } } ( s , a )$

## B.3 Impact on Part I (Stochastic Regime)

In Part I, algorithmic progress is evaluated under the on-policy discounted state-action measure $d ^ { \pi _ { t } } \times \pi _ { t } .$ , induced by the objective’s initial state distribution $\mu .$ If the critic is instead trained with an exploratory restart distribution ν, a distribution mismatch occurs between the actor’s on-policy evaluation metrics and the critic’s training measure.

Extension of Uncentered Gradient Identity (Lemma 5.1): Under exploratory $\nu -$ sampling, the critic orthogonality condition $\mathbb { E } _ { d _ { \nu } ^ { \pi _ { t } } } [ \phi \epsilon _ { t } ] = 0$ holds under the training measure rather than the objective’s on-policy measure $d ^ { \pi _ { t } } \times \pi _ { t }$ . Consequently, the term $\mathbb { E } _ { d ^ { \pi _ { t } } , \pi _ { t } } [ \phi \epsilon _ { t } ]$ no longer vanishes in the policy gradient decomposition, yielding

$$
\nabla J _ { \tau } ( \omega _ { t } ) = \frac { 1 } { 1 - \gamma } \bar { \Sigma } _ { c e n } ( \pi _ { t } ) \bigl ( \theta _ { \tau } ^ { * } ( \omega _ { t } ) - \tau \omega _ { t } \bigr ) + \frac { 1 } { 1 - \gamma } E _ { b i a s } + \frac { 1 } { 1 - \gamma } E _ { m i s m a t c h } ,
$$

where the mismatch term is defined as $E _ { m i s m a t c h } \triangleq \mathbb { E } _ { d ^ { \pi _ { t } } , \pi _ { t } } \bigl [ \phi ( s , a ) \epsilon _ { t } ( s , a ) \bigr ]$

Propagation of $C _ { \mu , \nu }$ into Actor Progress Bound (Lemma 5.3): Because both the new mismatch term $E _ { m i s m a t c h }$ and the original approximation bias $E _ { b i a s }$ are defined as expectations under the objective’s on-policy measure $d ^ { \pi _ { t } } \times \pi _ { t } ,$ we shift the measure to $d _ { \nu } ^ { \pi _ { t } }$ using the Training

Of-Policy Mismatch $C _ { \mu , \nu }$ (Assumption 13(i)). The mismatch term is bounded as:

$$
\| E _ { m i s m a t c h } \| _ { 2 } \leq B _ { \phi } \sqrt { C _ { \mu , \nu } \mathbb { E } _ { d _ { \nu } ^ { \pi _ { t } } } [ \epsilon _ { t } ^ { 2 } ] } \leq B _ { \phi } \sqrt { C _ { \mu , \nu } \epsilon _ { a p p } } .
$$

Similarly, the original bias term is bounded by $\lVert E _ { b i a s } \rVert _ { 2 } \leq B _ { \phi } \sqrt { C _ { \mu , \nu } \epsilon _ { a p p } } .$ . Bounding the inner products $\langle E _ { m i s m a t c h } , \Delta \omega _ { t } \rangle$ and $\langle E _ { b i a s } , \Delta \omega _ { t } \rangle$ using Young’s inequality leads to a progress bound where the overall bias constant $C _ { b i a s }$ is inflated by a factor proportional to $C _ { \mu , \nu }$ . In contrast, the joint concentrability bound $C _ { j o i n t }$ is used to lower-bound the on-policy Fisher information matrix $\begin{array} { r } { \big ( \bar { \Sigma } _ { c e n } ( \pi _ { t } ) \succeq \frac { \lambda } { C _ { j o i n t } } I \big ) } \end{array}$ , which involves evaluating the current policy directly against the parametric optimum and is independent of the critic’s exploratory measure ν. Thus the original $C _ { j o i n t }$ bound (Assumption 7) applies without modification.

Propagation into PL Condition (Lemma 4.2): In Step (ii) of the PL Condition proof, when bounding the approximation bias term, we first define the local policy mismatch $W ( s ) \triangleq$ ma $\tau _ { a ^ { \prime } } \frac { \pi _ { \omega _ { \tau } ^ { * } } ( a ^ { \prime } | s ) } { \pi _ { t } ( a ^ { \prime } | s ) }$ . Applying the Cauchy-Schwarz inequality over the action and state spaces, alongside the $\chi ^ { 2 }$ bound, isolates the error term $\mathbb { E } _ { d ^ { \omega _ { \tau } ^ { * } } } [ W ( s ) \mathbb { E } _ { \pi _ { t } } [ \epsilon _ { t } ^ { 2 } ] ]$ . To control this term, we shift the expectation from the measure $d ^ { \omega _ { \tau } ^ { * } } \times \pi _ { t }$ to the training measure $d _ { \nu } ^ { \pi _ { t } }$ via the Parametric Of-Policy Joint Mismatch $C _ { j o i n t , \nu }$ (Assumption 13(ii)):

$$
\begin{array} { r l } & { \mathbb { E } _ { d ^ { \omega _ { \tau } ^ { * } } } \bigl [ W ( s ) \mathbb { E } _ { \pi _ { t } } [ \epsilon _ { t } ^ { 2 } ] \bigr ] = \mathbb { E } _ { d _ { \nu } ^ { \pi _ { t } } } \left[ \left( \frac { d ^ { \omega _ { \tau } ^ { * } } ( s ) \pi _ { t } ( a | s ) } { d _ { \nu } ^ { \pi _ { t } } ( s , a ) } W ( s ) \right) \epsilon _ { t } ( s , a ) ^ { 2 } \right] } \\ & { \qquad \leq C _ { j o i n t , \nu } \mathbb { E } _ { d _ { \nu } ^ { \pi _ { t } } } [ \epsilon _ { t } ( s , a ) ^ { 2 } ] \leq C _ { j o i n t , \nu } \epsilon _ { a p p } . } \end{array}
$$

For the ideal algorithmic progress term, the bound in Step (i) relies on H¨older’s and Pinsker’s inequalities and remains unafected. Repeating the remaining steps in the proof of Lemma 4.2 therefore yields the same PL condition with $C _ { e r r }$ replaced by $\begin{array} { r } { C _ { e r r , \nu } \triangleq \frac { C _ { j o i n t , \nu } } { 1 - \gamma } } \end{array}$

Propagation into Global Class Approximation Error (Lemma 7.1): At the parametric optimum $\pi _ { \omega _ { \tau } ^ { * } }$ , the extended Uncentered Gradient Identity gives

$$
0 = \bar { \Sigma } _ { c e n } ( \pi _ { \omega _ { \tau } ^ { * } } ) \big ( \theta _ { \tau } ^ { * } ( \omega _ { \tau } ^ { * } ) - \tau \omega _ { \tau } ^ { * } \big ) + E _ { b i a s } ^ { * } + E _ { m i s m a t c h } ^ { * } ,
$$

where $E _ { b i a s } ^ { * }$ and $E _ { m i s m a t c h } ^ { * }$ are defined at $\pi _ { \omega _ { \tau } ^ { * } }$ , analogously to $E _ { b i a s }$ and $E _ { m i s m a t c h }$ . By the Optimum Of-Policy Mismatch $C _ { \mu , \nu }$ and Assumption 11, $\| E _ { b i a s } ^ { * } \| _ { 2 } \le B _ { \phi } \sqrt { C _ { \mu , \nu } \epsilon _ { a p p } }$ and $\| E _ { m i s m a t c h } ^ { * } \| _ { 2 } \leq$ $B _ { \phi } \sqrt { C _ { \mu , \nu } \epsilon _ { a p p } } ,$ and hence the Positive-Definite Fisher Information assumption gives

$$
\| \theta _ { \tau } ^ { * } ( \omega _ { \tau } ^ { * } ) - \tau \omega _ { \tau } ^ { * } \| _ { 2 } \leq \frac { 2 B _ { \phi } } { \lambda } \sqrt { C _ { \mu , \nu } \epsilon _ { a p p } } .
$$

For the approximation bias term in Step (iii) of Lemma 7.1, defining $\begin{array} { r } { W ^ { * } ( s ) \triangleq \operatorname* { m a x } _ { a ^ { \prime } } \frac { \pi _ { \tau } ^ { * } ( a ^ { \prime } | s ) } { \pi _ { \omega _ { \tau } ^ { * } } ( a ^ { \prime } | s ) } } \end{array}$ , the

Global-to-Parametric Of-Policy Joint Mismatch $C _ { j o i n t , \nu } ^ { * }$ yields

$$
\begin{array} { r l } & { \mathbb { E } _ { d _ { \tau } ^ { * } } \big [ W ^ { * } ( s ) \mathbb { E } _ { \pi _ { \omega _ { \tau } ^ { * } } } [ \epsilon _ { * } ^ { 2 } ] \big ] = \mathbb { E } _ { d _ { \nu } ^ { \omega _ { \tau } ^ { * } } } \left[ \left( \frac { d _ { \tau } ^ { * } ( s ) \pi _ { \omega _ { \tau } ^ { * } } ( a | s ) } { d _ { \nu } ^ { \omega _ { \tau } ^ { * } } ( s , a ) } W ^ { * } ( s ) \right) \epsilon _ { * } ( s , a ) ^ { 2 } \right] } \\ & { \qquad \leq C _ { j o i n t , \nu } ^ { * } \mathbb { E } _ { d _ { \nu } ^ { \omega _ { \tau } ^ { * } } } [ \epsilon _ { * } ^ { 2 } ] \leq C _ { j o i n t , \nu } ^ { * } \epsilon _ { a p p } . } \end{array}
$$

Repeating the remaining steps in the proof of Lemma 7.1 therefore yields

$$
J _ { \tau } ( \pi _ { \tau } ^ { * } ) - J _ { \tau } ( \pi _ { \omega _ { \tau } ^ { * } } ) \leq \frac { C _ { c l a s s , \nu } } { \tau } \epsilon _ { a p p } ,
$$

where

$$
C _ { c l a s s , \nu } \triangleq \frac { 4 B _ { \phi } ^ { 4 } C _ { \mu , \nu } } { ( 1 - \gamma ) \lambda ^ { 2 } } + \frac { C _ { j o i n t , \nu } ^ { * } } { 1 - \gamma } .
$$

The lower-bound argument in Lemma 7.2 is unafected by the exploratory critic training measure, yielding

$$
\epsilon _ { a p p } \geq C _ { c o m p , \nu } \tau , \quad \tau \in ( 0 , \tau _ { m a x } ] ,
$$

where

$$
C _ { c o m p , \nu } \triangleq \frac { q _ { 0 } \Delta } { 2 ( 1 - \gamma ) C _ { c l a s s , \nu } } \operatorname* { m i n } \left\{ 1 , \frac { q _ { 0 } \Delta } { 2 \tau _ { m a x } \log | \mathcal { A } | } \right\} , \quad q _ { 0 } \triangleq \frac { d \lambda } { 4 B _ { \phi } ^ { 2 } } .
$$

Thus, the Part-I approximation-temperature compatibility persists under exploratory ν-sampling, with the compatibility constant modified through $C _ { c l a s s , \nu }$

Impact on Critic Tracking: The critic tracking analysis relies on the orthogonality based on the ideal critic. Taking the conditional expectation of the critic SGD step yields:

$$
\mathbb { E } \big [ g _ { t + 1 } ^ { c r } \mid \mathcal { F } _ { t + 1 } \big ] = \bar { \Sigma } _ { u n c } \big ( d _ { \nu } ^ { \pi _ { t + 1 } } \big ) \tilde { e } _ { t } - \mathbb { E } _ { d _ { \nu } ^ { \pi _ { t + 1 } } } \big [ \phi ( s , a ) \epsilon _ { t + 1 } \big ( s , a \big ) \big ] ,
$$

where $\tilde { e } _ { t } \triangleq \theta _ { t + 1 } - \theta _ { \tau } ^ { \ast } ( \omega _ { t + 1 } )$ . Redefining the ideal critic and the associated errors to match the $\nu -$ measure (Appendix B.1) zeros out the cross-term, ensuring the expected update remains unbiased. By Assumption 12, the tracking argument in the main text applies with κ replaced by $\kappa _ { e x p } ,$ , yielding the corresponding contraction factor $1 - \alpha _ { t + 1 } \kappa _ { e x p }$ in the final tracking recurrence.

Conclusion for Part I: Under exploratory ν-sampling, the uncentered feature moment lower bound κ is replaced by $\kappa _ { e x p }$ . The of-policy shifts associated with critic approximation along the algorithmic trajectory are controlled by $C _ { \mu , \nu }$ and $C _ { j o i n t , \nu }$ , while the global class approximation error is controlled by $C _ { \mu , \nu }$ and $C _ { j o i n t , \nu } ^ { * }$ . The original joint concentrability $C _ { j o i n t }$ remains responsible for lower-bounding the on-policy Fisher information matrix. The Part-I approximation-temperature compatibility persists, with $C _ { c l a s s }$ and $C _ { c o m p }$ replaced by $C _ { c l a s s , \nu }$ and $C _ { c o m p , \nu } ,$ respectively. Subject to these generalized assumptions, the corresponding replacement of the critic-side structural constants, and the resulting admissible range for $\epsilon _ { a p p }$ , the Part I analysis carries over. With respect to the iteration budget T, the final average-iterate and last-iterate unregularized convergence rates remain $\tilde { \mathcal { O } } ( T ^ { - 1 } )$ as in the main text.

## B.4 Impact on Part II (Deterministic Regime)

In Part II, algorithmic progress is evaluated against the globally optimal policy $\pi _ { \tau } ^ { * }$ , rather than the parametrically optimal policy $\pi _ { \omega _ { \tau } ^ { * } }$ analyzed in Part I. Consequently, when shifting the expectation of the approximation error from the target evaluation measure to the training measure, the analysis relies on the Global Of-Policy Joint Mismatch evaluated under $d _ { \tau } ^ { * }$ (Assumption 13(iv)).

Extension of PMD Actor Progress Bound (Lemma 9.1): In Step (ii) of the PMD Actor Progress Bound proof, when bounding the approximation bias term, we first define the local policy mismatch $\begin{array} { r } { W ^ { \dagger } ( s ) \triangleq \operatorname* { m a x } _ { a ^ { \prime } } \frac { \pi _ { \tau } ^ { * } ( a ^ { \prime } | s ) } { \pi _ { t } ( a ^ { \prime } | s ) } } \end{array}$ . Applying the Cauchy-Schwarz inequality over the action and state spaces, alongside the $\chi ^ { 2 }$ bound, isolates the error term $\mathbb { E } _ { d _ { \tau } ^ { * } } \big [ W ^ { \dagger } ( s ) \mathbb { E } _ { \pi _ { t } } [ \epsilon _ { t } ^ { 2 } ] \big ]$ . To control this term, we shift the expectation from the measure $d _ { \tau } ^ { * } \times \pi _ { t }$ to the training measure $d _ { \nu } ^ { \pi _ { t } }$ via the Global Of-Policy Joint Mismatch $C _ { j o i n t , \nu } ^ { \dagger }$ (Assumption $1 3 ( \mathrm { i v } ) )$ :

$$
\begin{array} { r l } & { \mathbb { E } _ { d _ { \tau } ^ { * } } \left[ W ^ { \dagger } ( s ) \mathbb { E } _ { \pi _ { t } } [ \epsilon _ { t } ^ { 2 } ] \right] = \mathbb { E } _ { d _ { \nu } ^ { \pi _ { t } } } \left[ \left( \frac { d _ { \tau } ^ { * } ( s ) \pi _ { t } ( a | s ) } { d _ { \nu } ^ { \pi _ { t } } ( s , a ) } W ^ { \dagger } ( s ) \right) \epsilon _ { t } ( s , a ) ^ { 2 } \right] } \\ & { \qquad \leq C _ { j o i n t , \nu } ^ { \dagger } \mathbb { E } _ { d _ { \nu } ^ { \pi _ { t } } } [ \epsilon _ { t } ( s , a ) ^ { 2 } ] \leq C _ { j o i n t , \nu } ^ { \dagger } \epsilon _ { a p p } . } \end{array}
$$

For the tracking error penalty, the bound in Step (iii) remains unafected without requiring any state or policy mismatch bounds.

Extension of Critic Tracking Bound (Lemma 9.4): The analysis of critic tracking relies on the same orthogonality mechanism established in Part I. As shown above, redefining the ideal critic zeroes out the cross-term. The proof of Lemma 9.4 therefore applies with the original feature moment lower bound κ replaced by the exploratory restart parameter $\kappa _ { e x p } .$ yielding the corresponding contraction factor $1 - \alpha _ { t + 1 } \kappa _ { e x p }$ in the final tracking recurrence.

Conclusion for Part II: Under exploratory ν-sampling, the critic’s training geometry is governed by the exploratory feature moment lower bound $\kappa _ { e x p }$ in place of $\kappa ,$ while the of-policy shift of the critic approximation error is controlled by the Global Of-Policy Joint Mismatch $C _ { j o i n t , \nu } ^ { \dagger }$ in place of the joint concentrability bound $C _ { j o i n t } ^ { \dagger }$ . By these replacements, the PMD Actor Progress and critic tracking arguments hold as in the main text, with only the corresponding structural constants modified. Consequently, the subsequent Part II analysis applies directly. With respect to the iteration budget T, the final average-iterate and last-iterate unregularized convergence rates remain $\tilde { \mathcal { O } } ( T ^ { - 2 / 3 } )$ and $\tilde { \mathcal { O } } ( T ^ { - 1 / 3 } )$ , respectively as in the main text.

## C Concentrability-Free Analysis via $L _ { \infty }$ Approximation

In the main text, our analysis relies on bounding the critic’s approximation error via the $L _ { 2 }$ mean squared error $( \epsilon _ { a p p }$ in Assumption 4), which aligns natively with the stochastic gradient descent updates of the critic. However, evaluating this $L _ { 2 }$ approximation bias incurs penalties from the joint concentrability bounds $( C _ { j o i n t }$ and $C _ { j o i n t } ^ { \dagger } )$ to handle the state measure shifts and the action-space $\chi ^ { 2 } \mathrm { - t o - K L }$ divergence translations. Note that these joint concentrability constants, $C _ { j o i n t }$ and $C _ { j o i n t } ^ { \dagger } ,$ are absent from, respectively, the algorithmic progress term $( \lVert \theta _ { \tau } ^ { * } ( \omega _ { t } ) - \tau \omega _ { t } \rVert _ { 2 } ^ { 2 } )$ and the tracking error term $\left( Z _ { t } \right)$ in Lemmas 4.2 and 9.1.

In this section, we demonstrate that by substituting the $L _ { 2 }$ error assumption with an $L _ { \infty }$ error bound, we completely eliminate the remaining joint concentrability constants attached to the approximation bias, freeing the Deterministic Regime (Part II) from all concentrability dependencies. We also isolate the fundamental bottlenecks that prevent full elimination in our current analysis of the Stochastic Regime (Part I).

Assumption 14 $( L _ { \infty }$ Critic Approximation Error). The absolute approximation error of the ideal uncentered critic is uniformly bounded by $\epsilon _ { \infty }$ across the state-action space for all training policies $\{ \pi _ { t } \} _ { t \ge 0 }$ under Algorithm 1:

$$
\operatorname* { s u p } _ { s , a } | \epsilon _ { t } ( s , a ) | \leq \epsilon _ { \infty } .
$$

## C.1 Complete Concentrability Elimination in Part II

In Part II, algorithmic progress is driven by the PMD geometry over the probability simplex. We show that under the $L _ { \infty }$ approximation error assumption, the global joint concentrability bound $C _ { j o i n t } ^ { \dagger }$ is removed from the PMD Actor Progress Bound (Lemma 9.1).

Lemma C.1 (Concentrability-Free PMD Actor Progress). Under Assumptions 1, 3, and $1 \not \downarrow ,$ , if the actor step size satisfies $\eta _ { t } \leq 1 / \tau$ , the expected global performance gap satisfies the single-step inequality without any dependence on $C _ { j o i n t } ^ { \dagger }$ :

$$
( 1 - \gamma ) \mathbb { E } [ G a p _ { t } ^ { \dag } ] \le - \frac { 7 \tau } { 8 } \mathbb { E } [ D _ { t } ^ { \dag } ] + \frac { \mathbb { E } [ D _ { t } ^ { \dag } ] - \mathbb { E } [ D _ { t + 1 } ^ { \dag } ] } { \eta _ { t } } + \frac { \eta _ { t } } { 2 } G _ { a c } ^ { 2 } B _ { \phi } ^ { 2 } + 2 \epsilon _ { \infty } + \frac { 4 B _ { \phi } ^ { 2 } } { \tau } Z _ { t } ,
$$

where $Z _ { t } \triangleq \mathbb { E } [ \Vert e _ { t } \Vert _ { 2 } ^ { 2 } ]$ is the expected critic tracking error. Notably, this completely avoids $t h e \tau ^ { - 1 }$ penalty on the bias seen in the $L _ { 2 }$ analysis.

Proof. We revisit inequality (23) from the proof of Lemma 9.1. As shown in Step (iii) of that proof,

the tracking error penalty is bounded by $\textstyle \frac { \tau } { 8 } D _ { t } ^ { \dagger } + \frac { 4 B _ { \phi } ^ { 2 } } { \tau } \| e _ { t } \| _ { 2 } ^ { 2 }$ , without any concentrability constants.   
Therefore, we only need to re-evaluate the approximation bias term.

By H¨older’s inequality over the action space and Assumption 14, the approximation bias term is bounded:

$$
\mathbb { E } _ { d _ { \tau } ^ { * } } \Big [ \langle \epsilon _ { t } , \pi _ { \tau } ^ { * } - \pi _ { t } \rangle _ { \boldsymbol { \mathcal { A } } } \Big ] \leq \mathbb { E } _ { d _ { \tau } ^ { * } } \Big [ \operatorname* { m a x } _ { a } | \epsilon _ { t } ( s , a ) | \| \pi _ { \tau } ^ { * } - \pi _ { t } \| _ { 1 } \Big ] \leq 2 \epsilon _ { \infty } ,
$$

where we use the fact that the $L _ { 1 }$ distance is bounded by $\| \pi _ { \tau } ^ { * } - \pi _ { t } \| _ { 1 } \leq 2$ . This bound requires no concentrability bounds (i.e., it is completely free of $C _ { j o i n t } ^ { \dagger } )$

Substituting this approximation bias bound and the established tracking error bound back into inequality (23) (which provides $- \tau D _ { t } ^ { \dagger }$ and the algorithmic progress) yields:

$$
( 1 - \gamma ) \mathrm { G a p } _ { t } ^ { \dag } \leq - \tau D _ { t } ^ { \dag } + \frac { D _ { t } ^ { \dag } - D _ { t + 1 } ^ { \dag } } { \eta _ { t } } + \frac { \eta _ { t } } { 2 } G _ { a c } ^ { 2 } B _ { \phi } ^ { 2 } + 2 \epsilon _ { \infty } + \left( \frac { \tau } { 8 } D _ { t } ^ { \dag } + \frac { 4 B _ { \phi } ^ { 2 } } { \tau } \| e _ { t } \| _ { 2 } ^ { 2 } \right) .
$$

Combining the $D _ { t } ^ { \dagger }$ terms, $\begin{array} { r } { - \tau D _ { t } ^ { \dagger } + \frac { \tau } { 8 } D _ { t } ^ { \dagger } = - \frac { 7 \tau } { 8 } D _ { t } ^ { \dagger } } \end{array}$ , and taking the total expectation $\mathbb { E } [ \cdot ]$ over the algorithm trajectory completes the proof. □

By replacing Lemma 9.1 with Lemma C.1 in the Part II analysis, the subsequent convergence bounds hold without requiring any concentrability assumptions. We formalize these regularized convergence bounds below.

Theorem C.1 (Concentrability-Free Regularized Convergence). For Algorithm $^ { 1 , }$ suppose that Assumptions 1–3 and $1 \mathit { 4 }$ hold. For independent constants $\begin{array} { r } { c _ { \eta } \ge \frac { 4 } { 3 \tau } } \end{array}$ and $\zeta > 0$ , let the actor and critic step sizes be $\begin{array} { r } { \begin{array} { r } { \eta _ { t } = \frac { c _ { \eta } } { t + t _ { 0 } } } \end{array} } \end{array}$ and $\begin{array} { r } { \alpha _ { t + 1 } ~ = ~ \frac { c _ { \alpha } } { ( t + t _ { 0 } ) ^ { 2 / 3 } } } \end{array}$ , with $c _ { \alpha } \ { \triangleq } \ \zeta c _ { \eta } ^ { 2 / 3 }$ and an initial ofset $t _ { 0 } \geq$ max $\left( c _ { \eta } \tau , ( 2 c _ { \alpha } \kappa ) ^ { 3 / 2 } , ( \frac { 4 } { 3 c _ { \alpha } \kappa } ) ^ { 3 } \right)$ . By choosing the step-size constants $c _ { \eta } = \Theta ( \tau ^ { - 1 } )$ and $\zeta = \Theta ( \kappa ^ { - 1 } )$ and the initial ofset $t _ { 0 } = \Theta ( \tau ^ { - 1 } )$ subject to the stated conditions, the expected global regularized performance gap satisfies the following asymptotic rates for the average iterate and the last iterate, respectively:

$$
\frac { 1 } { T } \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } [ G a p _ { t } ^ { \dag } ] \leq \mathcal { O } \left( \frac { 1 } { ( 1 - \gamma ) ^ { 7 } \kappa ^ { 6 } \tau ^ { 5 / 3 } } T ^ { - 2 / 3 } \right) + \tilde { \mathcal { O } } \left( \frac { 1 } { ( 1 - \gamma ) ^ { 7 } \kappa ^ { 6 } \tau ^ { 2 } } T ^ { - 1 } \right) + \mathcal { O } \left( \frac { \epsilon _ { \infty } } { 1 - \gamma } \right) ,
$$

$$
\mathbb { E } [ G a p _ { T } ^ { \dagger } ] \le \mathcal { O } \left( \frac { 1 } { ( 1 - \gamma ) ^ { 5 } \kappa ^ { 4 } \tau ^ { 4 / 3 } } T ^ { - 1 / 3 } \right) + \mathcal { O } \left( \frac { \sqrt { \epsilon _ { \infty } } } { ( 1 - \gamma ) ^ { 2 } \kappa \tau ^ { 1 / 2 } } \right) .
$$

Proof. (i) Average-Iterate Bound. We repeat the PMD value-telescoping argument used in Theorem 9.1, replacing the PMD Actor Progress Bound (Lemma 9.1) by its concentrability-free counterpart, Lemma C.1. The only substantive diference is that the $L _ { 2 }$ approximation term $\mathcal { O } ( \epsilon _ { a p p } / \tau )$ in the single-step PMD inequality is replaced by the direct $L _ { \infty }$ penalty $2 \epsilon _ { \infty }$ . Consequently, the average-iterate convergence bound remains the same as in Theorem 9.1, except with $\begin{array} { r } { \mathcal { O } \big ( \frac { \epsilon _ { a p p } } { ( 1 - \gamma ) \tau } \big ) } \end{array}$ replaced by $\begin{array} { r } { \mathcal { O } ( \frac { \epsilon _ { \infty } } { 1 - \gamma } ) } \end{array}$ .

(ii) Last-Iterate Bound. For the last iterate, the $L _ { \infty }$ approximation error enters diferently because the performance gap is recovered through the terminal forward KL divergence. Dropping the non-negative performance gap in Lemma C.1 and multiplying by η<sub>t</sub> gives

$$
\mathbb { E } [ D _ { t + 1 } ^ { \dagger } ] \leq \left( 1 - \frac { 7 \tau } { 8 } \eta _ { t } \right) \mathbb { E } [ D _ { t } ^ { \dagger } ] + \frac { \eta _ { t } ^ { 2 } } { 2 } G _ { a c } ^ { 2 } B _ { \phi } ^ { 2 } + \frac { 4 B _ { \phi } ^ { 2 } \eta _ { t } } { \tau } Z _ { t } + 2 \eta _ { t } \epsilon _ { \infty } .
$$

Using $\begin{array} { r } { \eta _ { t } = \frac { c _ { \eta } } { t + t _ { 0 } } } \end{array}$ and $\begin{array} { r } { Z _ { t } \le \frac { v } { ( t + t _ { 0 } ) ^ { 2 / 3 } } } \end{array}$ , this recurrence takes the form of Lemma 9.5 with parameters $c = c _ { \infty } , A = A _ { \infty }$ , and $B = B _ { \infty }$

$$
c _ { \infty } \triangleq \frac { 7 \tau c _ { \eta } } { 8 } , \qquad A _ { \infty } \triangleq \frac { c _ { \eta } ^ { 2 } G _ { a c } ^ { 2 } B _ { \phi } ^ { 2 } } { 2 } + \frac { 4 c _ { \eta } B _ { \phi } ^ { 2 } v } { \tau } , \qquad B _ { \infty } \triangleq 2 c _ { \eta } \epsilon _ { \infty } .
$$

Since $c _ { \infty } > 1$ (implied by $\begin{array} { r } { c _ { \eta } \geq \frac { 4 } { 3 \tau } ) } \end{array}$ and $t _ { 0 } \geq c _ { \infty }$ (implied by $t _ { 0 } \geq c _ { \eta } \tau )$ , Chung’s Lemma yields

$$
\mathbb { E } [ D _ { T } ^ { \dagger } ] \le \frac { u _ { \infty } } { ( T + t _ { 0 } ) ^ { 2 / 3 } } + \frac { B _ { \infty } } { c _ { \infty } } = \frac { u _ { \infty } } { ( T + t _ { 0 } ) ^ { 2 / 3 } } + \frac { 1 6 } { 7 \tau } \epsilon _ { \infty } ,
$$

where $u _ { \infty } \stackrel { \Delta } { = }$ max $\begin{array} { r } { \left( t _ { 0 } ^ { 2 / 3 } D _ { 0 } ^ { \dagger } , \frac { 3 A _ { \infty } } { 3 c _ { \infty } - 2 } \right) } \end{array}$ . Under the stated step-size choices, the same structural calculation as in Lemma 9.6 gives

$$
u _ { \infty } = \mathcal { O } \left( \frac { 1 } { ( 1 - \gamma ) ^ { 6 } \kappa ^ { 6 } \tau ^ { 8 / 3 } } \right) .
$$

Finally, applying the Forward KL Bound (Lemma 9.7), Jensen’s inequality, and $\sqrt { x + y } \leq$ ${ \sqrt { x } } + { \sqrt { y } }$ gives

$$
\mathbb { E } [ \mathrm { G a p } _ { T } ^ { \dagger } ] \le \frac { \sqrt { 2 u _ { \infty } } M _ { m a x } } { 1 - \gamma } \frac { 1 } { ( T + t _ { 0 } ) ^ { 1 / 3 } } + \frac { 4 \sqrt { 2 } M _ { m a x } } { \sqrt { 7 } ( 1 - \gamma ) \tau ^ { 1 / 2 } } \sqrt { \epsilon _ { \infty } } .
$$

Using ${ \cal M } _ { m a x } = \mathcal { O } ( ( 1 - \gamma ) ^ { - 1 } \kappa ^ { - 1 } )$ from the proof of Theorem 9.2 yields the stated last-iterate convergence bound, completing the proof. □

For the average iterate, the $L _ { \infty }$ approximation bias in Theorem C.1 is independent of the temperature τ . Therefore, unlike the $L _ { 2 }$ analysis in the main text, a one-stage temperature can be chosen solely to balance the regularized convergence error against the exponential translation tail. For the last iterate, however, the forward-KL analysis introduces a residual $\tau ^ { - 1 / 2 } \sqrt { \epsilon _ { \infty } }$ approximation term. To prevent this factor from growing as the temperature decreases with $T ,$ , we employ a two-stage temperature for the last-iterate guarantee.

Theorem C.2 (Concentrability-Free Unregularized Convergence). For Algorithm 1, suppose that Assumptions 1–3, 5, and 14 hold.

For the average iterate, define the one-stage temperature

$$
\tau _ { T } ^ { a v g } \triangleq \frac { \Delta } { 2 \log ( C _ { \gamma } T ^ { 2 / 3 } ) } .
$$

If T is suficiently large such that $\tau _ { T } ^ { a v g } \leq \tau _ { m a x }$ , then by choosing the step-size parameters as in Theorem C.1 evaluated at $\tau = \tau _ { T } ^ { a v g }$ subject to the stated conditions, the expected average global unregularized performance gap satisfies

$$
\begin{array} { r l } & { \displaystyle \frac { 1 } { T } \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } \big [ J _ { 0 } ( \pi _ { 0 } ^ { * } ) - J _ { 0 } ( \pi _ { t } ) \big ] \leq \mathcal { O } \left( \frac { ( 1 - \gamma ) ^ { - 5 / 3 } + ( \log T ) ^ { 5 / 3 } } { ( 1 - \gamma ) ^ { 8 } \kappa ^ { 6 } \Delta ^ { 8 / 3 } T ^ { 2 / 3 } } \right) + \mathcal { O } \left( \frac { \epsilon _ { \infty } } { ( 1 - \gamma ) ^ { 2 } \Delta } \right) } \\ & { \quad \quad \quad \quad = \tilde { \mathcal { O } } ( T ^ { - 2 / 3 } ) + \mathcal { O } ( \epsilon _ { \infty } ) . } \end{array}
$$

For the last iterate, define the two-stage temperature

$$
\tau _ { T } ^ { l a s t } \triangleq \operatorname* { m a x } \left( \frac { \Delta } { 2 \log ( C _ { \gamma } T ^ { 2 / 3 } ) } , \frac { \Delta } { 2 \log ( C _ { \gamma } ( 1 + \epsilon _ { \infty } ^ { - 1 } ) ) } \right) .
$$

$I f T$ is suficiently large and $\epsilon _ { \infty }$ is suficiently small such that $\tau _ { T } ^ { l a s t } \le \tau _ { m a x }$ , then by choosing the step-size parameters as in Theorem C.1 evaluated at $\tau = \tau _ { T } ^ { l a s t }$ subject to the stated conditions, the expected last-iterate global unregularized performance gap satisfies

$$
\begin{array} { r l } & { \mathbb { E } \left[ J _ { 0 } ( \pi _ { 0 } ^ { * } ) - J _ { 0 } ( \pi _ { T } ) \right] \leq \mathcal { O } \left( \frac { ( 1 - \gamma ) ^ { - 4 / 3 } + ( \log T ) ^ { 4 / 3 } } { ( 1 - \gamma ) ^ { 6 } \kappa ^ { 4 } \Delta ^ { 7 / 3 } T ^ { 1 / 3 } } \right) } \\ & { \quad \quad \quad \quad + \mathcal { O } \left( \frac { ( 1 - \gamma ) ^ { - 1 / 2 } + \sqrt { \log ( 1 + \epsilon _ { \infty } ^ { - 1 } ) } } { ( 1 - \gamma ) ^ { 3 } \kappa \Delta ^ { 3 / 2 } } \sqrt { \epsilon _ { \infty } } \right) } \\ & { \quad \quad \quad = \tilde { \mathcal { O } } ( T ^ { - 1 / 3 } ) + \tilde { \mathcal { O } } ( \sqrt { \epsilon _ { \infty } } ) . } \end{array}
$$

In the case of $\epsilon _ { \infty } = 0$ , we interpret $\begin{array} { r } { \tau _ { T } ^ { l a s t } = \frac { \Delta } { 2 \log ( C _ { \gamma } T ^ { 2 / 3 } ) } } \end{array}$ and $\epsilon _ { \infty } \log ( 1 + \epsilon _ { \infty } ^ { - 1 } ) = 0$

Proof. (i) Average-Iterate Bound. By the Universal Unregularized Suboptimality Bound (Corollary 3.1) and the average-iterate bound in Theorem C.1,

$$
\begin{array} { r l } & { \displaystyle \frac { 1 } { T } \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } \big [ J _ { 0 } ( \pi _ { 0 } ^ { * } ) - J _ { 0 } ( \pi _ { t } ) \big ] \leq \mathcal { O } \left( \frac { T ^ { - 2 / 3 } } { ( 1 - \gamma ) ^ { 8 } \kappa ^ { 6 } \Delta ( \tau _ { T } ^ { a v g } ) ^ { 5 / 3 } } \right) + \tilde { \mathcal { O } } \left( \frac { T ^ { - 1 } } { ( 1 - \gamma ) ^ { 8 } \kappa ^ { 6 } \Delta ( \tau _ { T } ^ { a v g } ) ^ { 2 } } \right) } \\ & { \quad \quad \quad \quad \quad \quad + \mathcal { O } \left( \frac { \epsilon _ { \infty } } { ( 1 - \gamma ) ^ { 2 } \Delta } \right) + C _ { t a i l } \exp \left( - \frac { \Delta } { 2 \tau _ { T } ^ { a v g } } \right) . } \end{array}
$$

Substituting $\begin{array} { r } { \tau _ { T } ^ { a v g } = \frac { \Delta } { 2 \log ( C _ { \gamma } T ^ { 2 / 3 } ) } } \end{array}$ and following similar calculation as in the proof of Theorem 10.1 leads to the stated average-iterate bound.

(ii) Last-Iterate Bound. We first consider $\epsilon _ { \infty } > 0$ . When $\epsilon _ { \infty } = 0$ , the approximation term vanishes and the result follows from the sample-limited temperature $\begin{array} { r } { \tau _ { T } ^ { l a s t } = \frac { \Delta } { 2 \log ( C _ { \gamma } T ^ { 2 / 3 } ) } } \end{array}$

By Corollary 3.1 and the last-iterate bound in Theorem C.1,

$$
\begin{array} { r l } & { \mathbb { E } \left[ J _ { 0 } ( \pi _ { 0 } ^ { * } ) - J _ { 0 } ( \pi _ { T } ) \right] \leq { \cal O } \left( \frac { T ^ { - 1 / 3 } } { ( 1 - \gamma ) ^ { 6 } \kappa ^ { 4 } \Delta ( \tau _ { T } ^ { l a s t } ) ^ { 4 / 3 } } \right) + { \cal O } \left( \frac { \sqrt { \epsilon _ { \infty } } } { ( 1 - \gamma ) ^ { 3 } \kappa \Delta ( \tau _ { T } ^ { l a s t } ) ^ { 1 / 2 } } \right) } \\ & { \quad \quad \quad \quad + C _ { t a i l } \exp \left( - \frac { \Delta } { 2 \tau _ { T } ^ { l a s t } } \right) . } \end{array}
$$

Substituting the two-stage temperature $\tau _ { T } ^ { l a s t }$ and following similar calculation as in the proof of Theorem 10.2 leads to the stated last-iterate bound. □

## C.2 Generalization to Exploratory ν-Sampling (Deterministic Regime)

We extend the concentrability-free unregularized convergence bounds in the Deterministic Regime (Part II) to accommodate the exploratory ν-sampling introduced in Appendix B. Because the $L _ { \infty }$ approximation bound bypasses the need for state measure shifts, this extension does not require the Global Of-Policy Joint Mismatch (Assumption 13(iv)), while it relies on the positive-definite moment condition in Assumption 12, replacing the main-text lower bound κ by $\kappa _ { e x p }$ . Therefore, Theorem C.2 applies directly with this replacement, retaining the one-stage temperature for the average iterate and the two-stage temperature for the last iterate.

Corollary C.1 (Concentrability-Free Unregularized Convergence under ν-Sampling). For Algorithm 1 with exploratory ν-sampling, suppose that Assumptions 1–2, 5, 12, and $1 \mathit { 4 }$ hold.

For the average iterate, define the one-stage temperature

$$
\tau _ { T } ^ { a v g } \triangleq \frac { \Delta } { 2 \log ( C _ { \gamma } T ^ { 2 / 3 } ) } .
$$

$I f T$ is suficiently large such that $\tau _ { T } ^ { a v g } \leq \tau _ { m a x }$ , then by choosing the step-size parameters as in Theorem C.1 with κ replaced by $\kappa _ { e x p }$ and evaluated at $\tau = \tau _ { T } ^ { a v g }$ , the expected average global unregularized performance gap satisfies

$$
\begin{array} { r l } & { \displaystyle \frac { 1 } { T } \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } \big [ J _ { 0 } ( \pi _ { 0 } ^ { * } ) - J _ { 0 } ( \pi _ { t } ) \big ] \leq \mathcal { O } \left( \frac { ( 1 - \gamma ) ^ { - 5 / 3 } + ( \log T ) ^ { 5 / 3 } } { ( 1 - \gamma ) ^ { 8 } \kappa _ { e x p } ^ { 6 } \Delta ^ { 8 / 3 } T ^ { 2 / 3 } } \right) + \mathcal { O } \left( \frac { \epsilon _ { \infty } } { ( 1 - \gamma ) ^ { 2 } \Delta } \right) } \\ & { \quad \quad \quad \quad = \tilde { \mathcal { O } } ( T ^ { - 2 / 3 } ) + \mathcal { O } ( \epsilon _ { \infty } ) . } \end{array}
$$

For the last iterate, define the two-stage temperature

$$
\tau _ { T } ^ { l a s t } \triangleq \operatorname* { m a x } \left( \frac { \Delta } { 2 \log ( C _ { \gamma } T ^ { 2 / 3 } ) } , \frac { \Delta } { 2 \log ( C _ { \gamma } ( 1 + \epsilon _ { \infty } ^ { - 1 } ) ) } \right) .
$$

If T is suficiently large and $\epsilon _ { \infty }$ is suficiently small such that $\tau _ { T } ^ { l a s t } \le \tau _ { m a x } ,$ then by choosing the step-size parameters as in Theorem C.1 with κ replaced by $\kappa _ { e x p }$ and evaluated at $\tau = \tau _ { T } ^ { l a s t }$ , the

expected last-iterate global unregularized performance gap satisfies

$$
\begin{array} { r l } & { \mathbb { E } \left[ J _ { 0 } ( \pi _ { 0 } ^ { * } ) - J _ { 0 } ( \pi _ { T } ) \right] \leq \mathcal { O } \left( \frac { ( 1 - \gamma ) ^ { - 4 / 3 } + ( \log T ) ^ { 4 / 3 } } { ( 1 - \gamma ) ^ { 6 } \kappa _ { e x p } ^ { 4 } \Delta ^ { 7 / 3 } T ^ { 1 / 3 } } \right) } \\ & { \quad \quad \quad \quad + \mathcal { O } \left( \frac { ( 1 - \gamma ) ^ { - 1 / 2 } + \sqrt { \log ( 1 + \epsilon _ { \infty } ^ { - 1 } ) } } { ( 1 - \gamma ) ^ { 3 } \kappa _ { e x p } \Delta ^ { 3 / 2 } } \sqrt { \epsilon _ { \infty } } \right) } \\ & { \quad \quad \quad = \tilde { \mathcal { O } } ( T ^ { - 1 / 3 } ) + \tilde { \mathcal { O } } ( \sqrt { \epsilon _ { \infty } } ) . } \end{array}
$$

In the case $o f \epsilon _ { \infty } = 0$ , we interpret $\begin{array} { r } { \tau _ { T } ^ { l a s t } = \frac { \Delta } { 2 \log ( C _ { \gamma } T ^ { 2 / 3 } ) } } \end{array}$ and $\epsilon _ { \infty } \log ( 1 + \epsilon _ { \infty } ^ { - 1 } ) = 0$

## C.3 Impact on Part I and Geometric Bottleneck

Applying the same $L _ { \infty }$ mechanism to the Stochastic Regime removes the joint concentrability assumption from the Parameter-Space PL Condition (Lemma 4.2) and improves the bias scaling, bypassing the $\tau ^ { - 1 }$ amplification associated with $\epsilon _ { a p p }$

Lemma C.2 (Concentrability-Free PL Condition). Under Assumptions 1 and $^ { 1 \downarrow , }$ the parametric regularized performance gap satisfies the PL condition without any dependence on $C _ { j o i n t }$

$$
G a p _ { t } \leq \frac { B _ { \phi } ^ { 2 } } { 2 ( 1 - \gamma ) \tau } \| \theta _ { \tau } ^ { * } ( \omega _ { t } ) - \tau \omega _ { t } \| _ { 2 } ^ { 2 } + \frac { 2 } { 1 - \gamma } \epsilon _ { \infty } .
$$

Proof. By the Ideal Critic Decomposition (Lemma 3.3(i)) evaluated at $\pi = \pi _ { \omega _ { \tau } ^ { * } }$ , the gap is:

$$
\begin{array} { r } { ( 1 - \gamma ) \mathrm { G a p } _ { t } = - \tau D _ { t } + \mathbb { E } _ { d ^ { \omega _ { \tau } ^ { * } } } \Big [ \langle ( \theta _ { \tau } ^ { * } ( \omega _ { t } ) - \tau \omega _ { t } ) ^ { \top } \phi , \pi _ { \omega _ { \tau } ^ { * } } - \pi _ { t } \rangle _ { A } \Big ] + \mathbb { E } _ { d ^ { \omega _ { \tau } ^ { * } } } \big [ \langle \epsilon _ { t } , \pi _ { \omega _ { \tau } ^ { * } } - \pi _ { t } \rangle _ { A } \big ] . } \end{array}
$$

By H¨older’s inequality and Assumption 14, the approximation bias is bounded:

$$
\mathbb { E } _ { d ^ { \omega _ { \tau } ^ { * } } } \left[ \langle \epsilon _ { t } , \pi _ { \omega _ { \tau } ^ { * } } - \pi _ { t } \rangle _ { \mathfrak { A } } \right] \leq \mathbb { E } _ { d ^ { \omega _ { \tau } ^ { * } } } \left[ \operatorname* { m a x } _ { a } | \epsilon _ { t } ( s , a ) | \| \pi _ { \omega _ { \tau } ^ { * } } - \pi _ { t } \| _ { 1 } \right] \leq 2 \epsilon _ { \infty } .
$$

For the ideal algorithmic progress, we revisit the bound established in Step (i) of the proof of Lemma 4.2:

$$
\begin{array} { r } { \mathbb { E } _ { d ^ { \omega _ { \tau } ^ { * } } } \Big [ \langle ( \theta _ { \tau } ^ { * } ( \omega _ { t } ) - \tau \omega _ { t } ) ^ { \top } \phi , \pi _ { \omega _ { \tau } ^ { * } } - \pi _ { t } \rangle _ { A } \Big ] \leq B _ { \phi } \| \theta _ { \tau } ^ { * } ( \omega _ { t } ) - \tau \omega _ { t } \| _ { 2 } \sqrt { 2 D _ { t } } . } \end{array}
$$

We decouple this product using Young’s inequality $\begin{array} { r } { ( x y \le \frac { \tau } { 2 } x ^ { 2 } + \frac { 1 } { 2 \tau } y ^ { 2 } ) } \end{array}$ with $x = \sqrt { 2 D _ { t } }$ and $y =$ $B _ { \phi } \| \theta _ { \tau } ^ { * } ( \omega _ { t } ) - \tau \omega _ { t } \| _ { 2 } \colon$

$$
B _ { \phi } \| \theta _ { \tau } ^ { * } ( \omega _ { t } ) - \tau \omega _ { t } \| _ { 2 } \sqrt { 2 D _ { t } } \leq \tau D _ { t } + \frac { B _ { \phi } ^ { 2 } } { 2 \tau } \| \theta _ { \tau } ^ { * } ( \omega _ { t } ) - \tau \omega _ { t } \| _ { 2 } ^ { 2 } .
$$

Substituting these bounds back into the decomposition, the $\tau D _ { t }$ term cancels the restorative entropy force $- \tau D _ { t }$ . Dividing the entire inequality by $1 - \gamma _ { \mathrm { ~ J ~ } }$ ields the stated result. □

The $C _ { j o i n t }$ Geometric Bottleneck: While Lemma C.2 eliminates $C _ { j o i n t }$ from the PL condition, the Stochastic Regime cannot be completely freed from this concentrability bound under our current assumptions. As derived in the Actor Progress Bound (Lemma 5.3), establishing Euclidean parameter progress requires lower-bounding the eigenvalues of the on-policy Fisher information matrix: $\bar { \Sigma } _ { c e n } ( \pi _ { t } )$ . Because the curvature guarantee λ (Assumption 6) is anchored at the parametric optimum $\pi _ { \omega _ { \tau } ^ { * } }$ , we are forced to shift the measure: $\begin{array} { r } { \bar { \Sigma } _ { c e n } ( \pi _ { t } ) \succeq \frac { 1 } { C _ { j o i n t } } \bar { \Sigma } _ { c e n } ( \pi _ { \omega _ { \tau } ^ { * } } ) \succeq \frac { \lambda } { C _ { j o i n t } } I } \end{array}$ . This mapping of the current policy’s state-action measure to the optimal policy’s measure introduces the joint concentrability constant $C _ { j o i n t }$ as a geometric bottleneck. However, if it were assumed that $\textstyle { \bar { \Sigma } } _ { c e n } ( \pi _ { t } )$ is uniformly positive-definite across the training trajectory (analogous to the uncentered feature moment in Assumption 3), this reliance on $C _ { j o i n t }$ would be bypassed.

## D Related Works

## D.1 Agarwal et al. (2021)

Agarwal et al. (2021) studied an unregularized Q-NPG algorithm with a double-loop architecture and an exploratory ν-sampling measure. We mainly discuss their Corollary 21 and compare it with our Part II results under generalized ν-sampling (Appendix B).

Unregularized Q-NPG Algorithm. Translated to our notation, the unregularized Q-NPG algorithm updates the actor parameter $\omega _ { t }$ via:

$$
\omega _ { t + 1 } = \omega _ { t } + \eta _ { t } \theta _ { t + 1 } ,
$$

where the critic parameter $\theta _ { t + 1 }$ approximates the ideal minimizer $\theta _ { 0 } ^ { * } ( \omega _ { t } ) \triangleq$ arg $\mathrm { m i n } _ { \theta \in \mathbb { R } ^ { d } } L ( \theta ; \omega _ { t } , d _ { \nu } ^ { \pi _ { t } } )$ and the MSE objective in the unregularized Q-value projection is

$$
L ( \theta ; \omega _ { t } , d _ { \nu } ^ { \pi _ { t } } ) \triangleq \mathbb { E } _ { ( s , a ) \sim d _ { \nu } ^ { \pi _ { t } } } \Big [ \big ( Q _ { 0 } ^ { \pi _ { t } } ( s , a ) - \theta ^ { \top } \phi ( s , a ) \big ) ^ { 2 } \Big ]
$$

under the exploratory ν-sampling measure as described in Appendix B.

To estimate this $Q \cdot$ -value projection from samples, Agarwal et al. (2021) employ a double-loop architecture (Algorithm 2). The algorithm performs T outer-loop actor updates. Within each outer-loop iteration t, the actor parameter $\omega _ { t }$ is frozen, and an inner loop executes N stochastic gradient descent (SGD) steps to train the critic. The total number of SGD updates is therefore $T _ { t o t a l } = T \times N$ . As with our Algorithm 1, translating this into raw environment interactions incurs an additional $\mathcal { O } ( ( 1 - \gamma ) ^ { - 1 } )$ expected-horizon factor due to the Monte Carlo rollouts.

Notation Mapping. To bridge Agarwal et al. (2021) to our generalized ν-sampling analysis (Appendix B), we map the error bounds used in Agarwal et al. (2021) to our notation.

• Statistical Estimation Error $( \epsilon _ { s t a t } )$ : Agarwal et al. (2021) define $\epsilon _ { s t a t }$ to bound the excess risk

```latex
Algorithm 2 Double-Loop Unregularized Q-NPG (Agarwal et al., 2021; Yuan et al., 2023)
Input: Initial actor parameter ω , exploratory distribution $\nu ,$ inner-loop iterations N, outer-loop
iterations $T ,$ step sizes $\{ \eta _ { t } \} _ { t \ge 0 }$ and $\{ \alpha _ { t } \} _ { t \ge 0 }$
1: for $t = 0 , 1 , \ldots , T - 1$ do
2: Actor Freeze: Fix the current policy $\pi _ { t } \triangleq \pi _ { \omega _ { t } }$
3: Initialize inner critic parameter $\theta _ { t } ^ { ( 0 ) }$
4: for $k = 0 , 1 , \ldots , N - 1$ do
5: Sampling: Draw a state-action pair $( s _ { k } , a _ { k } ) \sim d _ { \nu } ^ { \pi _ { t } }$
6: Evaluation: Obtain an unbiased estimate $\hat { Q } _ { k }$ for the unregularized Q-value $Q _ { 0 } ^ { \pi _ { t } } ( s _ { k } , a _ { k } )$
7: Critic SGD: Update the inner critic parameter (possibly with additional projection):
$\theta _ { t } ^ { ( k + 1 ) } = \theta _ { t } ^ { ( k ) } + \alpha _ { t } \big ( \hat { Q } _ { k } - \theta _ { t } ^ { ( k ) \top } \phi ( s _ { k } , a _ { k } ) \big ) \phi ( s _ { k } , a _ { k } )$
8: end for
9: Critic Assignment: Set the outer-loop critic parameter $\theta _ { t + 1 } = \theta _ { t } ^ { ( N ) }$ (or an average of the
inner iterates).
10: Actor Update: Update the actor parameter:
$\omega _ { t + 1 } = \omega _ { t } + \eta _ { t } \theta _ { t + 1 }$
11: end for
Output: The output policy sequence $\{ \pi _ { t } \} _ { t = 0 } ^ { T } .$
of the sample-based critic parameter $\theta _ { t + 1 }$ (denoted $w ^ { ( t ) }$ in their work) compared to the ideal
least-squares minimizer $\theta _ { 0 } ^ { * } ( \omega _ { t } )$ , evaluated under the ν-induced training distribution:
$\begin{array} { r } { \mathcal { E } _ { s t a t } ( t ) \triangleq \mathbb { E } \left[ L ( \theta _ { t + 1 } ; \omega _ { t } , d _ { \nu } ^ { \pi _ { t } } ) - L ( \theta _ { 0 } ^ { * } ( \omega _ { t } ) ; \omega _ { t } , d _ { \nu } ^ { \pi _ { t } } ) \right] \leq \epsilon _ { s t a t } . } \end{array}$
Because $\theta _ { 0 } ^ { * } ( \omega _ { t } )$ is a minimum of this quadratic objective, the excess risk evaluates exactly
to the expected squared tracking error weighted by the uncentered feature moment matrix
under the ν-measure:
$\begin{array} { r } { \mathcal { E } _ { s t a t } ( t ) = \mathbb { E } \Big [ \big ( \theta _ { t + 1 } - \theta _ { 0 } ^ { * } ( \omega _ { t } ) \big ) ^ { \top } \bar { \Sigma } _ { u n c } ( d _ { \nu } ^ { \pi _ { t } } ) \big ( \theta _ { t + 1 } - \theta _ { 0 } ^ { * } ( \omega _ { t } ) \big ) \Big ] } \end{array}$
Under the feature bound $\bar { \Sigma } _ { u n c } ( d _ { \nu } ^ { \pi _ { t } } ) \preceq B _ { \phi } ^ { 2 } I$ and the exploratory positive-definiteness condition
```

$\bar { \Sigma } _ { u n c } ( d _ { \nu } ^ { \pi _ { t } } ) \succeq \kappa _ { e x p } I$ , we have

$$
\frac { \mathcal { E } _ { s t a t } ( t ) } { B _ { \phi } ^ { 2 } } \leq \mathbb { E } \bigl [ \| \theta _ { t + 1 } - \theta _ { 0 } ^ { * } ( \omega _ { t } ) \| _ { 2 } ^ { 2 } \bigr ] \leq \frac { \mathcal { E } _ { s t a t } ( t ) } { \kappa _ { e x p } } \leq \frac { \epsilon _ { s t a t } } { \kappa _ { e x p } } .
$$

Thus, their $\epsilon _ { s t a t }$ directly controls our expected critic tracking error.

For our comparison, we focus on the strengthened statistical setting in Remark 27 of Agarwal et al. (2021), where a positive lower bound on the minimum eigenvalue of the training feature moment matrix yields $\epsilon _ { s t a t } = \mathcal { O } ( d / N )$ . This is consistent with our exploratory positivedefiniteness assumption $\bar { \Sigma } _ { u n c } ( d _ { \nu } ^ { \pi _ { t } } ) \succeq \kappa _ { e x p } I .$ , and we absorb the feature dimension d into the structural constants to write $\epsilon _ { s t a t } = \mathcal { O } ( 1 / N )$

• Approximation Error $( \epsilon _ { a p p r o x } ) .$ Agarwal et al. (2021) also define $\epsilon _ { a p p r o x }$ as the inherent mean squared error of the ideal linear fit under the ν-induced training distribution:

$$
\begin{array} { r } { \mathbb { E } _ { ( s , a ) \sim d _ { \nu } ^ { \pi _ { t } } } \left[ \left( Q _ { 0 } ^ { \pi _ { t } } ( s , a ) - \theta _ { 0 } ^ { * } ( \omega _ { t } ) ^ { \top } \phi ( s , a ) \right) ^ { 2 } \right] = \mathbb { E } _ { d _ { \nu } ^ { \pi _ { t } } } \left[ \epsilon _ { t } ( s , a ) ^ { 2 } \right] \le \epsilon _ { a p p r o x } . } \end{array}
$$

Hence, their $\epsilon _ { a p p r o x }$ bound corresponds directly to our $\epsilon _ { a p p }$ bound.

Structural Assumptions. To control the distribution mismatch, Corollary 21 of Agarwal et al. (2021) requires a single uniform bound:

$$
\operatorname* { s u p } _ { s , a } \frac { d _ { 0 } ^ { \ast } ( s ) \frac { 1 } { | A | } } { \nu ( s , a ) } \leq C _ { A K L M } .
$$

While Agarwal et al. (2021) allow $d _ { 0 } ^ { \ast }$ to be an arbitrary comparator state visitation distribution, for concreteness, we take $d _ { 0 } ^ { \ast }$ to be that associated with the unregularized global optimum $\pi _ { 0 } ^ { * }$ Note that because bounding the density ratio inherently bounds the feature moment matrix, their $C _ { A K L M }$ assumption implies their Assumption 6.2, which requires a finite relative condition number sup<sub>θ</sub> $\frac { \theta ^ { \top } \bar { \Sigma } _ { u n c } ( d _ { 0 } ^ { * } \times \frac { 1 } { | \mathcal { A } | } ) \theta } { \theta ^ { \top } \bar { \Sigma } _ { u n c } ( \nu ) \theta } < \infty$

As shown in their analysis and replicated in Eq. (13), bounding the error term $\mathbb { E } _ { d _ { 0 } ^ { * } } \big [ \langle e r r _ { t } , \pi _ { 0 } ^ { * } -$ $\pi _ { t } \rangle _ { \ r { A } } ]$ only requires two specific mismatch bounds:

$$
\operatorname* { s u p } _ { s , a } \frac { d _ { 0 } ^ { \ast } ( s ) \pi _ { t } ( a | s ) } { \nu ( s , a ) } \leq \tilde { C } _ { A K L M } \quad \mathrm { a n d } \quad \operatorname* { s u p } _ { s , a } \frac { d _ { 0 } ^ { \ast } ( s ) \pi _ { 0 } ^ { \ast } ( a | s ) } { \nu ( s , a ) } \leq \tilde { C } _ { A K L M } .
$$

Because $d _ { \nu } ^ { \pi _ { t } } ( s , a ) \geq ( 1 - \gamma ) \nu ( s , a )$ , these mismatch bounds yield

$$
\operatorname* { s u p } _ { s , a } \frac { d _ { 0 } ^ { * } ( s ) \pi _ { t } ( a | s ) } { d _ { \nu } ^ { \pi _ { t } } ( s , a ) } \leq \frac { \tilde { C } _ { A K L M } } { 1 - \gamma } \quad \mathrm { a n d } \quad \operatorname* { s u p } _ { s , a } \frac { d _ { 0 } ^ { * } ( s ) \pi _ { 0 } ^ { * } ( a | s ) } { d _ { \nu } ^ { \pi _ { t } } ( s , a ) } \leq \frac { \tilde { C } _ { A K L M } } { 1 - \gamma } .\tag{11}
$$

Their original $1 / | { \mathcal { A } } |$ formulation acts as a conservative, policy-independent suficient condition for these bounds. The two mismatch bounds in (11) correspond to the unregularized limits of those in (9) and (10) respectively, deduced from our Assumption 13(iv). Due to the presence of the

additional local policy mismatch term, the unregularized limit of our Assumption $1 3 ( \mathrm { i v } )$ is, in general, more restrictive than the two bounds in (11).

In the on-policy specialization where, at each iteration t, the restart distribution is chosen as $\nu _ { t } ( s , a ) = \mu ( s ) \pi _ { t } ( a | s )$ , the induced training measure reduces t $\begin{array} { r } { d _ { \nu _ { t } } ^ { \pi _ { t } } \big ( s , a \big ) = d ^ { \pi _ { t } } \big ( s \big ) \pi _ { t } \big ( a | s \big ) } \end{array}$ . Under this specialization, the two bounds reduce to:

$$
\operatorname* { s u p } _ { s } \frac { d _ { 0 } ^ { * } ( s ) } { d ^ { \pi _ { t } } ( s ) } \leq \frac { \tilde { C } _ { A K L M } } { 1 - \gamma } \quad \mathrm { a n d } \quad \operatorname* { s u p } _ { s , a } \frac { d _ { 0 } ^ { * } ( s ) \pi _ { 0 } ^ { * } ( a | s ) } { d ^ { \pi _ { t } } ( s ) \pi _ { t } ( a | s ) } \leq \frac { \tilde { C } _ { A K L M } } { 1 - \gamma } ,
$$

respectively. The second bound, which implies the first, represents the exact unregularized counterpart to our global joint concentrability assumption (Assumption 9).

Convergence Bounds. Agarwal et al. (2021), Corollary 21, establishes the following bound for the best-iterate performance gap by employing a constant actor step size $\eta _ { t } = \eta \propto 1 / \sqrt { T }$

$$
\mathbb { E } \left[ \operatorname* { m i n } _ { t \in T } \left\{ J _ { 0 } ( \pi _ { 0 } ^ { * } ) - J _ { 0 } ( \pi _ { t } ) \right\} \right] \leq \mathcal { O } \left( \frac { 1 } { \sqrt { T } } \right) + \mathcal { O } \left( \sqrt { \epsilon _ { s t a t } } \right) + \mathcal { O } \left( \sqrt { \epsilon _ { a p p } } \right) ,\tag{12}
$$

where the proof also yields the corresponding average-iterate bound. Substituting the stronglyconvex inner-loop decay $\epsilon _ { s t a t } = \mathcal { O } ( 1 / N )$ , the bound evaluates to:

$$
\mathbb { E } \left[ \operatorname* { m i n } _ { t \in \mathcal { T } } \left\{ J _ { 0 } ( \pi _ { 0 } ^ { * } ) - J _ { 0 } ( \pi _ { t } ) \right\} \right] \leq \mathcal { O } \left( \frac { 1 } { \sqrt { T } } \right) + \mathcal { O } \left( \frac { 1 } { \sqrt { N } } \right) + \mathcal { O } \left( \sqrt { \epsilon _ { a p p } } \right) ,
$$

as mentioned in Agarwal et al. (2021), Remark 27.

To minimize the total algorithmic error subject to a fixed budget of SGD updates $T _ { t o t a l } = T \times N$ the outer-loop decay is balanced against the inner-loop statistical error by setting $1 / \sqrt { T } \approx 1 / \sqrt { N }$ yielding $T = N = \sqrt { T _ { t o t a l } }$ . Substituting these balanced sizes back shows that the unregularized performance gap of the double-loop Q-NPG algorithm is bounded by:

$$
\mathbb { E } \left[ \operatorname* { m i n } _ { t < T } \left\{ J _ { 0 } ( \pi _ { 0 } ^ { * } ) - J _ { 0 } ( \pi _ { t } ) \right\} \right] \leq \mathcal { O } \left( \frac { 1 } { T _ { t o t a l } ^ { 1 / 4 } } \right) + \mathcal { O } \left( \sqrt { \epsilon _ { a p p } } \right) .
$$

Main Steps of Analysis. The analysis in Agarwal et al. (2021) relies on the Performance Diference Lemma (PDL) and the Bregman Three-Point Identity. By PDL, the unregularized performance gap decomposes into the algorithmic update direction and the error $e r r _ { t } ( s , a ) \triangleq Q _ { 0 } ^ { \pi _ { t } } ( s , a ) - \theta _ { t + 1 } ^ { \top } \phi ( s , a )$ , notably lacking a negative KL divergence term:

$$
\begin{array} { r } { ( 1 - \gamma ) \big ( J _ { 0 } ( \pi _ { 0 } ^ { * } ) - J _ { 0 } ( \pi _ { t } ) \big ) = \mathbb { E } _ { d _ { 0 } ^ { * } } \Big [ \langle \theta _ { t + 1 } ^ { \top } \phi , \pi _ { 0 } ^ { * } - \pi _ { t } \rangle _ { \cal A } \Big ] + \mathbb { E } _ { d _ { 0 } ^ { * } } \Big [ \langle e r r _ { t } , \pi _ { 0 } ^ { * } - \pi _ { t } \rangle _ { \cal A } \Big ] . } \end{array}\tag{13}
$$

By substituting the log-linear parameterization into the Bregman Three-Point Identity evaluated against the current policy $\pi _ { t } ( \mathrm { i . e . , } \ \langle \log \pi _ { t + 1 } - \log \pi _ { t } , \pi _ { 0 } ^ { * } - \pi _ { t } \rangle _ { A } = D _ { K L } ( \pi _ { 0 } ^ { * } \| \pi _ { t } ) - D _ { K L } ( \pi _ { 0 } ^ { * } \| \pi _ { t + 1 } ) +$ $D _ { K L } ( \pi _ { t } \| \pi _ { t + 1 } ) )$ , the first term is directly related to the reduction in KL divergences:

$$
\eta _ { t } \langle \theta _ { t + 1 } ^ { \top } \phi , \pi _ { 0 } ^ { * } - \pi _ { t } \rangle _ { A } = D _ { K L } \big ( \pi _ { 0 } ^ { * } \| \pi _ { t } \big ) - D _ { K L } \big ( \pi _ { 0 } ^ { * } \| \pi _ { t + 1 } \big ) + D _ { K L } \big ( \pi _ { t } \| \pi _ { t + 1 } \big ) .
$$

Bounding the local distance $D _ { K L } ( \pi _ { t } \Vert \pi _ { t + 1 } )$ by $\mathcal { O } ( \eta _ { t } ^ { 2 } )$ leads to the inequality (which acts similarly to our PMD Algorithmic Progress bound):

$$
\langle \theta _ { t + 1 } ^ { \top } \phi , \pi _ { 0 } ^ { * } - \pi _ { t } \rangle _ { \mathcal A } \leq \frac { 1 } { \eta _ { t } } \Big ( D _ { K L } \big ( \pi _ { 0 } ^ { * } \| \pi _ { t } \big ) - D _ { K L } \big ( \pi _ { 0 } ^ { * } \| \pi _ { t + 1 } \big ) \Big ) + \mathcal O ( \eta _ { t } ) .
$$

Taking the expectation over $d _ { 0 } ^ { \ast }$ and substituting this into the PDL equation (13) yields a progress bound (mirroring inequality (23) in our proof of PMD Actor Progress Bound):

$$
( 1 - \gamma ) \big ( J _ { 0 } ( \pi _ { 0 } ^ { * } ) - J _ { 0 } ( \pi _ { t } ) \big ) \leq \frac { D _ { t } ^ { \dagger } - D _ { t + 1 } ^ { \dagger } } { \eta _ { t } } + \mathbb { E } _ { d _ { 0 } ^ { * } } \Big [ \langle e r r _ { t } , \pi _ { 0 } ^ { * } - \pi _ { t } \rangle _ { \cal A } \Big ] + { \cal O } ( \eta _ { t } ) ,\tag{14}
$$

where $D _ { t } ^ { \dagger } \triangleq \mathbb { E } _ { d _ { 0 } ^ { * } } [ D _ { K L } ( \pi _ { 0 } ^ { * } \Vert \pi _ { t } ) ]$ , serving as the unregularized counterpart to $D _ { t } ^ { \dagger }$ in our analysis. Averaging this inequality over T iterations with a constant step size $\eta \ : \mathrm { y }$ ields a bound of $\begin{array} { r l } {  { \frac { D _ { 0 } ^ { \dagger } } { \eta T } + \mathcal { O } ( \eta ) } } \end{array}$ up to the remaining error term. For this error term, Agarwal et al. (2021) apply Cauchy-Schwarz over the state-action space $( \mathrm { e . g . , } \mathbb { E } _ { d _ { 0 } ^ { \ast } , \pi _ { 0 } ^ { \ast } } [ e r r _ { t } ] \leq \sqrt { \mathbb { E } _ { d _ { 0 } ^ { \ast } , \pi _ { 0 } ^ { \ast } } [ e r r _ { t } ^ { 2 } ] } )$ , trapping both the statistical estimation and approximation errors under a square root $( \mathcal { O } ( \sqrt { \epsilon _ { s t a t } } ) + \mathcal { O } ( \sqrt { \epsilon _ { a p p } } ) )$ . Combining these components yields exactly the convergence bound stated in (12).

Comparison. Because both the analysis in Agarwal et al. (2021) and our Part II analysis operate within the PMD geometry, we compare the resulting average-iterate bounds. In contrast to their $\mathcal { O } ( T _ { t o t a l } ^ { - 1 / 4 } ) + \mathcal { O } ( \sqrt { \epsilon _ { a p p } } )$ bound, our single-loop average-iterate bound (Theorem 10.1) achieves an accelerated $\tilde { \mathcal { O } } ( T _ { t o t a l } ^ { - 2 / 3 } )$ rate and a linear $\tilde { \mathcal { O } } ( \epsilon _ { a p p } )$ bias dependence.

This improvement is driven by three core analytical choices in Part II:

• relying on entropy regularization to extract a Restorative Entropy Force term, which introduces a geometric contraction that enables PMD Value-Telescoping with diminishing step sizes $( \eta _ { t } = \mathcal { O } ( t ^ { - 1 } ) )$ to bypass the $1 / \sqrt { T }$ barrier;

• exploiting the Global Joint Concentrability assumption or, with ν-sampling, the Generalized Of-Policy Mismatch assumption, to simultaneously manage the state measure shift and facilitate the action-space $\chi ^ { 2 } \mathrm { - t o - K L }$ divergence translation, decoupling the error inner product via Young’s inequality after Cauchy-Schwarz and absorbing the resulting divergence penalty back into the Restorative Entropy Force; and

• deriving the exponential translation mechanism under the Minimal Action Gap assumption to map regularized progress back to the unregularized objective, bypassing the standard linear entropy penalty.

## D.2 Yuan et al. (2023)

Yuan et al. (2023) analyzed the unregularized Q-NPG algorithm similar to that in Agarwal et al. (2021). The algorithm utilizes a double-loop architecture and trains the critic using an exploratory ν-sampling measure as shown in Algorithm 2.

Notation and Assumptions. To bridge Yuan et al. (2023) to our framework, we align their structural assumptions and error definitions with our notation:

• Distribution Mismatch $( \vartheta _ { \mu } ) .$ : They define a state distribution mismatch coeficient $\vartheta _ { \mu } \triangleq { \underline { { \underline { { \Delta } } } } }$ $\frac { 1 } { 1 - \gamma } \| d _ { 0 } ^ { * } / \mu \| _ { \infty }$ , which satisfies $\vartheta _ { \mu } \geq 1 / ( 1 - \gamma )$ . While Yuan et al. (2023) allow $d _ { 0 } ^ { \ast }$ to be an arbitrary comparator state visitation distribution, for concreteness, we take $d _ { 0 } ^ { \ast }$ to be that associated with the unregularized global optimum $\pi _ { 0 } ^ { * }$ . Because $d ^ { \pi _ { t } } ( s ) \geq ( 1 - \gamma ) \mu ( s )$ , this coeficient also satisfies $\vartheta _ { \mu } \geq \| d _ { 0 } ^ { \ast } / d ^ { \pi _ { t } } \| _ { \infty }$ . This serves as the unregularized counterpart to the implied state concentrability bound $\| d _ { \tau } ^ { * } / d ^ { \pi _ { t } } \| _ { \infty } \le C _ { j o i n t } ^ { \dagger }$ from our Assumption 9.

$L _ { 2 } – C o n c e n t r a b i l i t y .$ Their concentrability assumption (Assumption 6 in their paper) is postulated for several state-action distributions in terms of the $L _ { 2 }$ norm of the density ratio over $d _ { \nu } ^ { \pi _ { t } }$ instead of the $L _ { \infty }$ norm, including among others

$$
\mathbb { E } _ { ( s , a ) \sim d _ { \nu } ^ { \pi _ { t } } } \left[ \left( \frac { d _ { 0 } ^ { \ast } ( s ) \pi _ { t } ( a \vert s ) } { d _ { \nu } ^ { \pi _ { t } } ( s , a ) } \right) ^ { 2 } \right] \leq C _ { Y D G L X } , \quad \mathbb { E } _ { ( s , a ) \sim d _ { \nu } ^ { \pi _ { t } } } \left[ \left( \frac { d _ { 0 } ^ { \ast } ( s ) \pi _ { 0 } ^ { \ast } ( a \vert s ) } { d _ { \nu } ^ { \pi _ { t } } ( s , a ) } \right) ^ { 2 } \right] \leq C _ { Y D G L X } .
$$

These bounds correspond to the unregularized $L _ { 2 }$ limits of the $L _ { \infty }$ mismatch bounds in (9) and (10) respectively, deduced from our Assumption 13(iv).

• Error Terms $( \epsilon _ { s t a t } ~ a n d \epsilon _ { a p p } ) _ { }$ : Their statistical estimation error $\epsilon _ { s t a t }$ and linear approximation error ϵ<sub>approx</sub> map identically to those defined in the Agarwal et al. (2021) above.

Convergence Bounds. Under these structural assumptions, their Theorem 3 establishes that for a geometrically increasing step size satisfying $\eta _ { t + 1 } \geq { \frac { 1 } { \gamma } } \eta _ { t }$ (with an initialization $\eta _ { 0 } ~ \geq$ $\begin{array} { r } { \frac { 1 - \gamma } { \gamma } D _ { K L } ( \pi _ { 0 } ^ { * } \lVert \boldsymbol { \pi } _ { 0 } ) ) } \end{array}$ , the expected unregularized performance gap at iteration T is bounded by:

$$
\mathbb { E } \left[ J _ { 0 } ( \pi _ { 0 } ^ { * } ) - J _ { 0 } ( \pi _ { T } ) \right] \leq \frac { 2 } { 1 - \gamma } \left( 1 - \frac { 1 } { \vartheta _ { \mu } } \right) ^ { T } + \mathcal { O } \left( \sqrt { \epsilon _ { s t a t } } + \sqrt { \epsilon _ { a p p } } \right) .\tag{15}
$$

Substituting the standard inner-loop critic decay $\epsilon _ { s t a t } = \mathcal { O } ( 1 / N )$ into Theorem 3 yields the performance gap in terms of the algorithmic loops:

$$
\mathbb { E } \left[ J _ { 0 } ( \pi _ { 0 } ^ { * } ) - J _ { 0 } ( \pi _ { T } ) \right] \leq \mathcal { O } \left( \left( 1 - \frac { 1 } { \vartheta _ { \mu } } \right) ^ { T } \right) + \mathcal { O } \left( \frac { 1 } { \sqrt { N } } \right) + \mathcal { O } \left( \sqrt { \epsilon _ { a p p } } \right) .
$$

To extract the final sample complexity (Corollary 1), Yuan et al. tune the outer-loop size $T$ and inner-loop size N to achieve a target convergence error ϵ:

• Outer Loop (T): To ensure the exponentially decaying convergence error falls below ϵ, the number of actor steps scales as: $T = \mathcal { O } \left( \log ( 1 / \epsilon ) \right)$

• Inner Loop (N): To ensure the square-rooted statistical error $\mathcal { O } ( 1 / \sqrt { N } )$ falls below ϵ, the number of critic SGD steps per actor update grows as: $N = \mathcal { O } ( 1 / \epsilon ^ { 2 } )$

From these choices, the total budget of SGD updates required is $\begin{array} { r } { T _ { t o t a l } = T \times N = \mathcal { O } \left( \frac { 1 } { \epsilon ^ { 2 } } \log \frac { 1 } { \epsilon } \right) } \end{array}$ Inverting this relationship with respect to $T _ { t o t a l }$ , their Corollary 1 implies that the unregularized performance gap of the double-loop Q-NPG algorithm is bounded by:

$$
\mathbb { E } \left[ J _ { 0 } ( \pi _ { 0 } ^ { * } ) - J _ { 0 } ( \pi _ { T } ) \right] \leq \tilde { \mathcal { O } } \left( \frac { 1 } { \sqrt { T _ { t o t a l } } } \right) + { \mathcal { O } } \left( \sqrt { \epsilon _ { a p p } } \right) .
$$

Main Steps of Analysis. To bypass the $1 / \sqrt { T }$ bottleneck, Yuan et al. (2023) analyze the unregularized PMD update via its first-order optimality condition (i.e., $\eta _ { t } \langle \theta _ { t + 1 } ^ { \top } \phi , \pi _ { 0 } ^ { * } - \pi _ { t + 1 } \rangle _ { A } \leq$ $\langle \log \pi _ { t + 1 } - \log \pi _ { t } , \pi _ { 0 } ^ { * } - \pi _ { t + 1 } \rangle _ { \cal A } \rangle$ . Applying the Bregman Three-Point Identity evaluated against the next policy $\pi _ { t + 1 } \left( \mathrm { i . e . , \ } \langle \log \pi _ { t + 1 } - \log \pi _ { t } , \pi _ { 0 } ^ { \ast } - \pi _ { t + 1 } \rangle _ { \cal A } = D _ { K L } ( \pi _ { 0 } ^ { \ast } \| \pi _ { t } ) - D _ { K L } ( \pi _ { 0 } ^ { \ast } \| \pi _ { t + 1 } ) - D _ { K L } ( \pi _ { t + 1 } \| \pi _ { t } ) \right)$ and dropping the non-positive local distance, $- D _ { K L } ( \pi _ { t + 1 } \| \pi _ { t } )$ , yields a one-step bound:

$$
\begin{array} { r l } & { \eta _ { t } \langle \theta _ { t + 1 } ^ { \top } \phi , \pi _ { 0 } ^ { * } - \pi _ { t + 1 } \rangle _ { A } \leq D _ { K L } ( \pi _ { 0 } ^ { * } \| \pi _ { t } ) - D _ { K L } ( \pi _ { 0 } ^ { * } \| \pi _ { t + 1 } ) - D _ { K L } ( \pi _ { t + 1 } \| \pi _ { t } ) } \\ & { \qquad \leq D _ { K L } ( \pi _ { 0 } ^ { * } \| \pi _ { t } ) - D _ { K L } ( \pi _ { 0 } ^ { * } \| \pi _ { t + 1 } ) . } \end{array}
$$

Then they split the inner product: $\begin{array} { r } { \langle \theta _ { t + 1 } ^ { \top } \phi , \pi _ { 0 } ^ { * } - \pi _ { t + 1 } \rangle _ { A } = \langle \theta _ { t + 1 } ^ { \top } \phi , \pi _ { 0 } ^ { * } - \pi _ { t } \rangle _ { A } + \langle \theta _ { t + 1 } ^ { \top } \phi , \pi _ { t } - \pi _ { t + 1 } \rangle _ { A } } \end{array}$ . Taking the expectation of the first term over $d _ { 0 } ^ { \ast }$ and applying PDL yields $( 1 - \gamma ) \mathrm { G a p } _ { t } ^ { \dag } - \mathbb { E } _ { d _ { 0 } ^ { * } } \Big [ \langle e r r _ { t } , \pi _ { 0 } ^ { * } -$ $\pi _ { t } \rangle _ { \ r { A } } ]$ as in (13), where ${ \mathrm { G a p } } _ { t } ^ { \dagger } = J _ { 0 } ( \pi _ { 0 } ^ { * } ) - J _ { 0 } ( \pi _ { t } )$ . For the second term $\langle \theta _ { t + 1 } ^ { \top } \phi , \pi _ { t } - \pi _ { t + 1 } \rangle _ { \mathcal { A } }$ , which is non-positive, they shift the state measure to $d ^ { \pi _ { t + 1 } }$ via the mismatch coeficient $\vartheta _ { \mu }$ . Applying PDL again to this shifted term yields the bound:

$$
\begin{array} { r } { \mathbb { E } _ { d _ { 0 } ^ { s } } \left[ \langle \theta _ { t + 1 } ^ { \top } \phi , \pi _ { t } - \pi _ { t + 1 } \rangle _ { \mathcal { A } } \right] \geq \vartheta _ { \mu } \left( ( 1 - \gamma ) ( \mathrm { G a p } _ { t + 1 } ^ { \dagger } - \mathrm { G a p } _ { t } ^ { \dagger } ) - \mathbb { E } _ { d ^ { \pi _ { t + 1 } } } \left[ \langle e r r _ { t } , \pi _ { t } - \pi _ { t + 1 } \rangle _ { \mathcal { A } } \right] \right) . } \end{array}
$$

Substituting these back and rearranging yields a contractive recurrence for the next gap:

$$
\begin{array} { r l } & { ( 1 - \gamma ) \mathrm { G a p } _ { t + 1 } ^ { \dag } \le \left( 1 - \frac { 1 } { \vartheta _ { \mu } } \right) ( 1 - \gamma ) \mathrm { G a p } _ { t } ^ { \dag } + \frac { 1 } { \eta _ { t } \vartheta _ { \mu } } \Big ( D _ { K L } ( \pi _ { 0 } ^ { * } \| \pi _ { t } ) - D _ { K L } \big ( \pi _ { 0 } ^ { * } \| \pi _ { t + 1 } ) \Big ) } \\ & { \qquad + \frac { 1 } { \vartheta _ { \mu } } \mathbb { E } _ { d _ { 0 } ^ { * } } \Big [ \langle e r r _ { t } , \pi _ { 0 } ^ { * } - \pi _ { t } \rangle _ { \mathcal { A } } \Big ] + \mathbb { E } _ { d } \pi _ { t + 1 } \Big [ \langle e r r _ { t } , \pi _ { t } - \pi _ { t + 1 } \rangle _ { \mathcal { A } } \Big ] . } \end{array}
$$

By deploying geometrically increasing step sizes $( \eta _ { t } \propto \gamma ^ { - t } )$ , Yuan et al. (2023) telescope this recursion to derive a linear $\mathcal { O } ( ( 1 - 1 / \vartheta _ { \mu } ) ^ { T } )$ outer-loop convergence rate as stated in (15). By choosing a highly unbalanced N and $T \ ( N \gg T )$ , they achieve an overall last-iterate convergence rate of $\mathcal { O } ( 1 / \sqrt { T _ { t o t a l } } )$ in terms of the total number of SGD updates $T _ { t o t a l } = N \times T$

Comparison. We compare the analysis in Yuan et al. (2023) with our Part II analysis. In contrast to their unbalanced double-loop structure and exponentially growing step-size schedule $( \eta _ { t } \propto \gamma ^ { - t } )$ , our single-loop algorithm $( N = 1 , T = T _ { t o t a l } )$ uses a stable, diminishing $\mathcal { O } ( t ^ { - 1 } )$ stepsize schedule. Furthermore, compared to their $\tilde { \mathcal { O } } ( T _ { t o t a l } ^ { - 1 / 2 } ) + \mathcal { O } ( \sqrt { \epsilon _ { a p p } } )$ bound, our theory leads to an $\tilde { \mathcal { O } } ( T _ { t o t a l } ^ { - 2 / 3 } ) + \tilde { \mathcal { O } } ( \epsilon _ { a p p } )$ average-iterate bound, and an $\tilde { \mathcal { O } } ( T _ { t o t a l } ^ { - 1 / 3 } ) + \tilde { \mathcal { O } } ( \sqrt { \epsilon _ { a p p } } )$ last-iterate bound for the unregularized MDP.

As discussed in the comparison with Agarwal et al. (2021), our theoretical analysis in Part II achieves this by (1) exploiting entropy regularization to extract a Restorative Entropy Force, (2) utilizing the Global Joint Concentrability assumption or, with ν-sampling, the Generalized Of-Policy Mismatch assumption, to simultaneously manage the state measure shift and facilitate the action-space $\chi ^ { 2 } \mathrm { - t o - K I }$ divergence translation, and (3) applying the exponential translation mechanism to bypass the standard linear entropy penalty.

## D.3 Liu et al. (2020)

Liu et al. (2020) analyzed an unregularized NPG algorithm. Similar to the NPG algorithm in Agarwal et al. (2021), their algorithm utilizes a double-loop architecture but operates with a horizon-independent, constant outer-loop step size by exploiting a positive-definite Fisher information matrix assumption to achieve faster convergence.

Unregularized NPG Algorithm. Translated to our notation, the double-loop NPG algorithm studied by Liu et al. (2020) updates the actor parameter $\omega _ { t }$ via:

$$
\omega _ { t + 1 } = \omega _ { t } + \eta \theta _ { t + 1 } ,
$$

where the critic parameter $\theta _ { t + 1 }$ approximates the ideal minimizer in the unregularized advantage projection (similar to $\theta _ { \tau } ^ { * A } ( \omega _ { t } )$ in Appendix A):

$$
\boldsymbol { \theta } _ { 0 } ^ { \ast A } ( \omega _ { t } ) \triangleq \arg \operatorname* { m i n } _ { \boldsymbol { \theta } \in \mathbb { R } ^ { d } } \mathbb { E } _ { d ^ { \pi _ { t } } , \pi _ { t } } \left[ \left( A _ { 0 } ^ { \pi _ { t } } ( s , a ) - \boldsymbol { \theta } ^ { \top } \tilde { \boldsymbol { \phi } } _ { t } ( s , a ) \right) ^ { 2 } \right] .
$$

By the Policy Gradient Theorem, this projection is proportional to the natural policy gradient direction: $\theta _ { 0 } ^ { * A } ( \omega _ { t } ) = ( 1 - \gamma ) \bar { \Sigma } _ { c e n } ( \pi _ { t } ) ^ { - 1 } \nabla J _ { 0 } ( \omega _ { t } )$ . The algorithm estimates this direction by solving the regression problem over N inner-loop SGD steps via conditionally independent sampling with fixed policy $\pi _ { t }$ (Algorithm 3). The total number of SGD updates is $T _ { t o t a l } = T \times N$

Notation and Assumptions. To bridge Liu et al. (2020) to our framework, we align their

Algorithm 3 Double-Loop Unregularized NPG (Liu et al., 2020)   
Input: Initial actor parameter ω , inner-loop iterations $N .$ , outer-loop iterations T, constant actor   
step size $\eta ,$ inner critic step size α.   
1: for $t = 0 , 1 , \ldots , T - 1$ do   
2: Actor Freeze: Fix the current policy $\pi _ { t } \triangleq \pi _ { \omega _ { t } }$   
3: Initialize inner critic parameter $\theta _ { t } ^ { ( 0 ) } = 0 .$   
4: for $k = 0 , 1 , \ldots , N - 1$ do   
5: Sampling: Draw state-action pairs $( s _ { k } , a _ { k } ) \sim d ^ { \pi _ { t } } \times \pi _ { t } .$   
6: Evaluation: Obtain an unbiased estimate $\hat { A } _ { k }$ for the advantage $A _ { 0 } ^ { \pi _ { t } } ( s _ { k } , a _ { k } )$   
7: Critic SGD: Update the critic parameter:   
$\theta _ { t } ^ { ( k + 1 ) } = \theta _ { t } ^ { ( k ) } + \alpha \Big ( \hat { A } _ { k } - \theta _ { t } ^ { ( k ) \top } \tilde { \phi } _ { t } ( s _ { k } , a _ { k } ) \Big ) \tilde { \phi } _ { t } ( s _ { k } , a _ { k } )$   
8: end for   
9: Critic Assignment: Set the outer-loop critic parameter as the average of the inner iterates:   
$\begin{array} { r } { \theta _ { t + 1 } = \frac { 1 } { N } \sum _ { k = 1 } ^ { N } \theta _ { t } ^ { ( k ) } } \end{array}$   
10: Actor Update: Update the actor parameter:   
$\omega _ { t + 1 } = \omega _ { t } + \eta \theta _ { t + 1 }$   
11: end for   
Output: The output policy sequence $\{ \pi _ { t } \} _ { t = 0 } ^ { T } .$

structural assumptions with our notation:

• Positive-Definite Fisher Information: Their Assumption 2.1 requires the Fisher information matrix $\bar { \Sigma } _ { c e n } ( \pi _ { \omega } )$ to be uniformly positive-definite for all $\omega \in \mathbb { R } ^ { d }$ . This serves a similar purpose to Assumption 6 in Part I. In fact, their proof also holds provided $\bar { \Sigma } _ { c e n } ( { \pi } _ { t } ) \succeq { \mu _ { F } } { I }$ for all training policies $\{ \pi _ { t } \} _ { t \ge 0 }$ . This condition is satisfied if $\bar { \Sigma } _ { c e n } ( \pi _ { \omega _ { 0 } ^ { * } } )$ is positive-definite at the parametric optimum $\pi _ { \omega _ { 0 } ^ { * } }$ and the joint concentrability bound holds: $\begin{array} { r } { \operatorname* { s u p } _ { s , a } \frac { d ^ { \omega _ { 0 } ^ { * } } ( s ) \pi _ { \omega _ { 0 } ^ { * } } ( a | s ) } { d ^ { \pi _ { t } } ( s ) \pi _ { t } ( a | s ) } \leq } \end{array}$ $C _ { j o i n t , 0 }$ for a constant $C _ { j o i n t , 0 } > 0$ , where $d ^ { \omega _ { 0 } ^ { * } } \triangleq d ^ { \pi _ { \omega _ { 0 } ^ { * } } }$ . As noted below Assumption 6, the Fisher information $\bar { \Sigma } _ { c e n } ( \pi _ { 0 } ^ { * } )$ is usually degenerate even in the Stochastic Regime because the global optimum $\pi _ { 0 } ^ { * }$ is deterministic. • Transferred Approximation Error $( \epsilon _ { b i a s } )$ : Their Assumption 4.4 defines the approximation error directly under the state-action distribution of the global optimum $\pi _ { 0 } ^ { * }$ :

$$
\begin{array} { r } { \mathbb { E } _ { d _ { 0 } ^ { * } , \pi _ { 0 } ^ { * } } \Big [ \big ( A _ { 0 } ^ { \pi t } ( s , a ) - \theta _ { 0 } ^ { * A } ( \omega _ { t } ) ^ { \top } \tilde { \phi } _ { t } ( s , a ) \big ) ^ { 2 } \Big ] \leq \epsilon _ { b i a s } . } \end{array}
$$

This implicitly bundles the $L _ { 2 }$ approximation error $\left( \epsilon _ { a p p } \right)$ and the global joint concentrability bound $\begin{array} { r } { ( \operatorname* { s u p } _ { s , a } \frac { d _ { 0 } ^ { * } ( s ) \pi _ { 0 } ^ { * } ( a | s ) } { d ^ { \pi _ { t } } ( s ) \pi _ { t } ( a | s ) } \le C _ { j o i n t , 0 } ^ { \dag } } \end{array}$ for a constant $C _ { j o i n t , 0 } ^ { \dagger } > 0 )$ into a single “transferred” constant, explaining the absence of explicit distribution mismatch coeficients in their convergence bounds. Here, $C _ { j o i n t , 0 }$ parallels $C _ { j o i n t }$ in our Part I analysis, while the direct global-to-current mismatch $C _ { j o i n t , 0 } ^ { \dagger }$ corresponds to the combined shift controlled by $C _ { j o i n t } C _ { j o i n t } ^ { * } .$

• Statistical Estimation Error: The convergence error of their inner-loop SGD is bounded by $\mathbb { E } [ \| \theta _ { t + 1 } - \theta _ { 0 } ^ { * A } ( \omega _ { t } ) \| _ { 2 } ^ { 2 } ] = \mathcal { O } ( 1 / N )$ , relying on the strong convexity provided by the positivedefinite Fisher information in the advantage regression problem.

Convergence Bounds. Under these assumptions, their Theorem 4.9 establishes the following bound for the average unregularized performance gap with a constant step size $\eta = \Theta ( 1 )$ :

$$
\frac { 1 } { T } \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } \big [ J _ { 0 } ( \pi _ { 0 } ^ { * } ) - J _ { 0 } ( \pi _ { t } ) \big ] \leq \mathcal { O } \left( \frac { 1 } { \eta T } \right) + \mathcal { O } \left( \frac { 1 } { \sqrt { N } } \right) + \mathcal { O } \left( \sqrt { \epsilon _ { b i a s } } \right) .\tag{16}
$$

To balance the optimization error against the inner-loop statistical error for a fixed budget of SGD updates $T _ { t o t a l } = T \times N$ , they tune the loop sizes such that $T = \mathcal { O } ( T _ { t o t a l } ^ { 1 / 3 } )$ and $N = \mathcal { O } ( T _ { t o t a l } ^ { 2 / 3 } )$ . This yields the overall convergence bound:

$$
\frac { 1 } { T } \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } \big [ J _ { 0 } ( \pi _ { 0 } ^ { * } ) - J _ { 0 } ( \pi _ { t } ) \big ] \leq \mathcal { O } \left( T _ { t o t a l } ^ { - 1 / 3 } \right) + \mathcal { O } \left( \sqrt { \epsilon _ { b i a s } } \right) .
$$

Main Steps of Analysis. The theoretical analysis in Liu et al. (2020) employs an interesting set of techniques, mixing Policy Mirror Descent (PMD) bounding (akin to our Part II) with objective smoothness and the positive-definite Fisher information (akin to our Part I).

First, they derive a bound on the performance gap (Proposition 4.5) that is structurally similar to averaging the inequality (14) in Agarwal et al. (2021) and (23) in our analysis:

$$
\frac { 1 } { T } \sum _ { t = 0 } ^ { T - 1 } \left( J _ { 0 } ( \pi _ { 0 } ^ { * } ) - J _ { 0 } ( \pi _ { t } ) \right) \leq \frac { \sqrt { \epsilon _ { b i a s } } } { 1 - \gamma } + \frac { D _ { 0 } ^ { \dagger } } { \eta T } + \frac { G } { T } \sum _ { t = 0 } ^ { T - 1 } \| \theta _ { t + 1 } - \theta _ { 0 } ^ { * A } ( \omega _ { t } ) \| _ { 2 } + \frac { M \eta } { 2 T } \sum _ { t = 0 } ^ { T - 1 } \| \theta _ { t + 1 } \| _ { 2 } ^ { 2 } ,\tag{17}
$$

where $G$ is an upper bound of $\| \nabla \log \pi _ { \omega } \|$ and M is a Lipschitz constant of ∇ log $\pi _ { \omega }$ in $\omega .$

Next, to control the accumulated step penalty $\begin{array} { r l } { \sum _ { t = 0 } ^ { T - 1 } \| \theta _ { t + 1 } \| _ { 2 } ^ { 2 } } & { { } } \end{array}$ , they employ a smoothness expansion over the objective $J _ { 0 }$ . After substituting the actor update $\omega _ { t + 1 } = \omega _ { t } + \eta \theta _ { t + 1 }$ , the step-wise progress is bounded by:

$$
J _ { 0 } ( \omega _ { t + 1 } ) - J _ { 0 } ( \omega _ { t } ) \geq \eta \langle \nabla J _ { 0 } ( \omega _ { t } ) , \theta _ { t + 1 } \rangle - \frac { L _ { J } \eta ^ { 2 } } { 2 } \| \theta _ { t + 1 } \| _ { 2 } ^ { 2 } ,
$$

where $L _ { J }$ is a smoothness constant of the objective $J _ { 0 } ( \omega )$ (i.e., a Lipschitz constant of its gradient). Then they decompose the actual update $\theta _ { t + 1 }$ into the ideal target $\theta _ { 0 } ^ { * A } ( \omega _ { t } ) ~ = ~ ( 1 ~ -$ $\gamma ) \bar { \Sigma } _ { c e n } ( \pi _ { t } ) ^ { - 1 } \nabla J _ { 0 } ( \omega _ { t } )$ and the tracking error $\theta _ { t + 1 } - \theta _ { 0 } ^ { * A } ( \omega _ { t } )$ . By invoking the positive-definite Fisher information $( \bar { \Sigma } _ { c e n } ( { \pi } _ { t } ) \succeq { \mu _ { F } } I )$ and bounded centered features $( \bar { \Sigma } _ { c e n } ( \pi _ { t } ) \preceq G ^ { 2 } I )$ , the ideal progress is lower-bounded $\begin{array} { r } { \big ( \langle \nabla J _ { 0 } ( \omega _ { t } ) , \theta _ { 0 } ^ { * A } ( \omega _ { t } ) \rangle \geq \frac { 1 - \gamma } { G ^ { 2 } } \| \nabla J _ { 0 } ( \omega _ { t } ) \| _ { 2 } ^ { 2 } \big ) } \end{array}$ , and the ideal step penalty is upper-bounded $\begin{array} { r } { ( \| \theta _ { 0 } ^ { * A } ( \omega _ { t } ) \| _ { 2 } ^ { 2 } \leq \frac { ( 1 - \gamma ) ^ { 2 } } { \mu _ { F } ^ { 2 } } \| \nabla J _ { 0 } ( \omega _ { t } ) \| _ { 2 } ^ { 2 } ) } \end{array}$

Substituting these bounds and telescoping the step-wise progress bound controls the average gradient norm $\begin{array} { r } { \frac { 1 } { T } \sum _ { t = 0 } ^ { T - 1 } \| \nabla J _ { 0 } ( \omega _ { t } ) \| _ { 2 } ^ { 2 } } \end{array}$ by the total objective progress $\begin{array} { r } { \mathcal { O } \big ( \frac { J _ { 0 } ( \omega _ { T } ) - J _ { 0 } ( \omega _ { 0 } ) } { T } \big ) } \end{array}$ up to the average expected tracking error $\begin{array} { r } { \frac { 1 } { T } \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } [ \| \theta _ { t + 1 } - \theta _ { 0 } ^ { * A } ( \omega _ { t } ) \| _ { 2 } ^ { 2 } ] } \end{array}$ . Because the inner-loop SGD drives this tracking error down, this bound on the average gradient norm subsequently shows that the accumulated step penalty in the PMD bound (17), $\begin{array} { r l } { { \frac { \eta } { T } } \sum _ { t = 0 } ^ { T - 1 } { \| \theta _ { t + 1 } \| _ { 2 } ^ { 2 } } } & { { } } \end{array}$ $\begin{array} { r } { \mathcal { O } \big ( \frac { 1 } { \eta T } \big ) } \end{array}$ (up to the tracking error). With a constant step size $\eta = \Theta ( 1 )$ , this leads to the stated convergence bound (16), thereby accelerating the outer-loop convergence to $\mathcal { O } ( 1 / T )$ and the overall rate to $\mathcal { O } ( T _ { t o t a l } ^ { - 1 / 3 } )$ .

Comparison. While Liu et al. (2020) improves upon the $\mathcal { O } ( T _ { t o t a l } ^ { - 1 / 4 } )$ rate of Agarwal et al. (2021), their analysis is appropriate only within the Stochastic Regime due to the reliance on a uniformly positive-definite Fisher information matrix. Hence, it is meaningful to compare their theoretical framework with our Part I analysis.

• Roles of the Fisher Information. As NPG is used, the positive-definite Fisher information serves two distinct purposes in their framework. First, it provides the strong convexity when solving the advantage value projection. Second, it provides a mechanism to control the step penalty from the smoothness expansion. In contrast, for our uncentered algorithm (Q-NPG), we rely on the positive-definite uncentered feature moment $( \kappa > 0 )$ for the inner-loop critic convergence, while employing the Fisher information $( \lambda > 0 )$ for one purpose: to lower-bound the algorithmic progress in the smoothness expansion for the outer loop.

• Gradient Space vs Parameter Space. Liu et al. (2020) choose to maintain and bound the average gradient norm $\lVert \nabla J _ { 0 } ( \omega _ { t } ) \rVert _ { 2 } ^ { 2 }$ in the gradient space. In contrast, our proof relates this objective progress to the parameter distance $\lVert \theta _ { t + 1 } - \tau \omega _ { t } \rVert _ { 2 } ^ { 2 }$ as in the Actor Progress Bound (Lemma 5.3). In addition, we establish a Parameter-Space PL Condition, relating the performance gap directly to the parameter distance $\lVert \theta _ { \tau } ^ { * } ( \omega _ { t } ) - \tau \omega _ { t } \rVert _ { 2 } ^ { 2 }$

• Telescoping vs Joint Lyapunov. To finalize their convergence guarantees, Liu et al. (2020)

still use telescoping to bound the average gap. Our approach takes a fundamentally diferent path: we use the Actor Progress Bound to derive a Coupled Actor Recurrence and a Coupled Critic Tracking, leading to a joint Lyapunov analysis.

Ultimately, via entropy regularization and the Exponential Translation mechanism, our Part I analysis achieves an optimal $\tilde { \mathcal { O } } ( T _ { t o t a l } ^ { - 1 } )$ unregularized rate for both average-iterate and last-iterate convergence (up to the approximation error) in the Stochastic Regime, accelerating beyond the $\mathcal { O } ( T _ { t o t a l } ^ { - 1 / 3 } )$ rate established in Liu et al. (2020).

## D.4 Cayci et al. (2024)

Cayci et al. (2024) studied an entropy-regularized NPG algorithm with averaging. Translated to our notation, their analysis operates under a fixed regularization temperature $\tau > 0$ and projects the regularized action-values onto centered features. See Appendix A for a discussion of Q-value projections in entropy-regularized NPG algorithms.

Entropy-Regularized NPG Algorithm. Cayci et al. (2024) employ a double-loop architecture as shown in Algorithm 4: an inner Temporal Diference (TD) learning subroutine to estimate the centered Q-values $\left( \tilde { Q } _ { \tau } ^ { \pi _ { t } } \right)$ , followed by an inner, constrained SGD subroutine to project these centered Q-values onto the centered features $( \tilde { \phi } _ { t } )$

The exact minimizer in this centered Q-value projection is defined as

$$
\theta _ { \tau } ^ { * R } ( \omega _ { t } ) = \arg \operatorname* { m i n } _ { \| \theta \| _ { 2 } \leq R } \overline { { L } } ( \theta , \omega _ { t } ) \triangleq \mathbb { E } _ { d ^ { \pi _ { t } } \times \pi _ { t } } \big [ \big ( \tilde { Q } _ { \tau } ^ { \pi _ { t } } ( s , a ) - \theta ^ { \top } \tilde { \phi } _ { t } ( s , a ) \big ) ^ { 2 } \big ] ,
$$

where $R > 0$ is a projection radius, $\tilde { \phi } _ { t } ( s , a ) \triangleq \nabla \log \pi _ { t } ( a | s ) = \phi ( s , a ) - \mathbb { E } _ { a ^ { \prime } \sim \pi _ { t } } [ \phi ( s , a ^ { \prime } ) ]$ denotes the centered features and $\tilde { Q } _ { \tau } ^ { \pi _ { t } } \big ( s , a \big ) \triangleq Q _ { \tau } ^ { \pi _ { t } } \big ( s , a \big ) - \mathbb { E } _ { a ^ { \prime } \sim \pi _ { t } } \big [ Q _ { \tau } ^ { \pi _ { t } } \big ( s , a ^ { \prime } \big ) \big ]$ denotes the centered Q-value. See Appendix A for a discussion on the relationship between our ideal critic $\theta _ { \tau } ^ { * } ( \omega _ { t } )$ and the unconstrained version of $\theta _ { \tau } ^ { * R } ( \omega _ { t } )$ , denoted as $\theta _ { \tau } ^ { \ast C H S } ( \omega _ { t } )$ .

Notation and Assumptions. To bridge Cayci et al. (2024) to our framework, we align their structural assumptions and error definitions with our notation:

• Projection Radius $( R )$ : In their analysis of constrained SGD (Proposition 4.5), the projection radius R is assumed to satisfy a structural lower bound $( R > \bar { R } )$ . This is similar to our constrained SGD, where the projection radius $R _ { \theta }$ is tied to the structural parameter $B _ { \theta }$

• Minimum Action Probability $( p _ { m i n } )$ : The global minimum action probability over the trajectory is defined as $p _ { m i n } \triangleq \operatorname* { i n f } _ { t , s , a } \pi _ { t } ( a | s )$ . For a fixed τ , because the critic is constrained in bound for the best-iterate, parametric regularized performance gap (which also applies to the average-iterate gap) by employing the step-size schedule $\begin{array} { r } { \eta _ { t } = \frac { 1 } { \tau ( t + 1 ) } } \end{array}$ :

Algorithm 4 Sample-Based Entropy-Regularized NPG with Averaging (Cayci et al., 2024)   
Input: Projection radius $R > 0$ , regularization $\tau > 0$ , TD iterations $K$ , projection iterations $N$   
step-sizes $\begin{array} { r } { \eta _ { t } = \frac { 1 } { \tau ( t + 1 ) } } \end{array}$   
1: Initialization: $\omega _ { 0 } = 0 .$   
2: for $t = 0 , 1 , \ldots , T - 1$ do   
3: Actor Freeze: Fix the current policy $\pi _ { t } \triangleq \pi _ { \omega _ { t } }$   
4: TD Learning (Critic): Run K steps of sample-based TD learning to obtain an estimator   
$\hat { Q } _ { \tau } ^ { \pi _ { t } }$ for the regularized Q-value $Q _ { \tau } ^ { \pi _ { t } }$ . Compute the centered estimator $\hat { \tilde { Q } } _ { \tau } ^ { \pi _ { t } } ( s , a ) = \hat { Q } _ { \tau } ^ { \pi _ { t } } ( s , a ) -$   
$\mathbb { E } _ { a ^ { \prime } \sim \pi _ { t } } \hat { Q } _ { \tau } ^ { \pi _ { t } } ( s , a ^ { \prime } )$ for the centered Q-value $\tilde { Q } _ { \tau } ^ { \pi _ { t } } ( s , a )$   
5: SGD Projection (Critic): Run N steps of constrained SGD using $\hat { \tilde { Q } } _ { \tau } ^ { \pi _ { t } }$ to obtain an esti  
mate $\theta _ { t + 1 }$ for the exact minimizer $\theta _ { \tau } ^ { * R } ( \omega _ { t } )$ , constrained to the ball $\lVert \theta _ { t + 1 } \rVert _ { 2 } \leq R .$   
6: Actor Update: Update the actor parameter:   
$\omega _ { t + 1 } = \omega _ { t } + \eta _ { t } ( \theta _ { t + 1 } - \tau \omega _ { t } ) .$   
7: end for   
Output: The output policy sequence $\{ \pi _ { t } \} _ { t = 0 } ^ { T } .$   
the inner SGD subroutine $( \| \theta _ { t + 1 } \| _ { 2 } \leq R )$ , the actor’s update rule structurally guarantees that   
the actor parameter iterate remains bounded: $\| \omega _ { t } \| _ { 2 } \leq R / \tau$ for all $t \geq 0$ . As a result, $p _ { m i n }$ is   
bounded by $\begin{array} { r } { p _ { m i n } \ge \frac { 1 } { | \mathcal { A } | } \exp ( - 2 R / \tau ) > 0 } \end{array}$ . See a related discussion after Assumption 7.   
• Distribution Mismatch $\langle \| d ^ { \omega _ { \tau } ^ { * } } / \mu \| _ { \infty } \rangle$ : Cayci et al. (2024) use a state distribution mismatch   
coeficient $\| d ^ { \omega _ { \tau } ^ { * } } / \mu \| _ { \infty }$ , relative to the parametric optimum $\pi _ { \omega _ { \tau } ^ { * } }$ . Because $d ^ { \pi _ { t } } ( s ) \geq ( 1 - \gamma ) \mu ( s )$   
this coeficient upper bounds (up to a scaling factor) the implied state concentrability bound   
$\| d ^ { \omega _ { \tau } ^ { * } } / d ^ { \pi _ { t } } \| _ { \infty } \leq C _ { j o i n t }$ from our Assumption 7.   
• Constrained Approximation Error $( \epsilon ( R ) )$ : Their approximation bias is defined as $\epsilon ( R ) \ { \stackrel { \triangle } { = } }$   
$\mathrm { s u p } _ { t \geq 0 } \overline { { L } } ( \theta _ { t } ^ { * R } , \omega _ { t } )$ . This is comparable to our approximation error $\epsilon _ { a p p }$   
• Statistical Error $( \epsilon _ { s t a t } ) .$ : Their $\epsilon _ { c r i t i c }$ bounds the mean squared error of the inner TD evalu  
ation subroutine, and $\epsilon _ { a c t o r }$ bounds the convergence error of the inner SGD projection sub  
routine. The sum of these two variances $( \epsilon _ { c r i t i c } + \epsilon _ { a c t o r } )$ maps directly to our total expected   
statistical tracking error, $\epsilon _ { s t a t }$   
Convergence Bounds. Under these definitions, their Proposition 4.1 establishes the following

$$
\mathbb { E } \left[ \operatorname* { m i n } _ { t < T } \left\{ J _ { \tau } ( \pi _ { \omega _ { \tau } ^ { * } } ) - J _ { \tau } ( \pi _ { t } ) \right\} \right] \leq O \left( \frac { \log T } { \tau T } \right) + O \left( \sqrt { \frac { \epsilon ( R ) + \epsilon _ { s t a t } } { p _ { m i n } } } \right) .\tag{18}
$$

To optimize the total error subject to a fixed budget of stochastic updates, let K and N denote the numbers of inner TD and SGD steps, respectively. For simplicity, we take $K = \Theta ( N )$ , so that the total stochastic-update budget satisfies $T _ { t o t a l } = T ( K + N ) = \Theta ( T N )$ . Under the full-rank assumptions described in their Remark 4.6, taking $K = \Theta ( N )$ gives $\epsilon _ { s t a t } = \tilde { \mathcal { O } } ( 1 / N )$ . Balancing the outer-loop decay $\mathcal { O } ( T ^ { - 1 } )$ against the inner-loop statistical error $\mathcal { O } ( N ^ { - 1 / 2 } )$ requires $T \approx \sqrt { N }$ . This yields $N \approx T _ { t o t a l } ^ { 2 / 3 }$ and $T \approx T _ { t o t a l } ^ { 1 / 3 }$ . Substituting these balanced loop sizes back into the gap bound establishes an overall convergence rate for the regularized objective:

$$
\mathbb { E } \Big [ \operatorname* { m i n } _ { t < T } \left\{ J _ { \tau } ( \pi _ { \omega _ { \tau } ^ { * } } ) - J _ { \tau } ( \pi _ { t } ) \right\} \Big ] \leq \mathcal { O } \left( \frac { \log T } { \tau T _ { t o t a l } ^ { 1 / 3 } } \right) + \mathcal { O } \left( \sqrt { \frac { \epsilon ( R ) } { p _ { m i n } } } \right) ,
$$

as mentioned in Cayci et al. (2024), Remark 4.6.

Main Steps of Analysis. The core analysis in Cayci et al. (2024) (Theorem 3.6) relies on a Lyapunov drift inequality (their Lemma 3.5):

$$
D _ { t + 1 } - D _ { t } \leq - \eta _ { t } \tau D _ { t } - \eta _ { t } ( 1 - \gamma ) \big ( J _ { \tau } ( \pi _ { \omega _ { \tau } ^ { * } } ) - J _ { \tau } ( \pi _ { t } ) \big ) + \eta _ { t } \mathbb { E } _ { d ^ { \omega _ { \tau } ^ { * } } } \Big [ \langle e r r _ { t } , \pi _ { \omega _ { \tau } ^ { * } } - \pi _ { t } \rangle _ { A } \Big ] + \mathcal { O } ( \eta _ { t } ^ { 2 } ) ,\tag{19}
$$

where $D _ { t } \ \triangleq \ \mathbb { E } _ { d ^ { \omega _ { \tau } ^ { * } } } [ D _ { K L } ( \pi _ { \omega _ { \tau } ^ { * } } \| \pi _ { t } ) ]$ serves as the potential function, and $e r r _ { t } ( s , a ) \ \triangleq \ \tilde { Q } _ { \tau } ^ { \pi _ { t } } ( s , a ) \ -$ $\theta _ { t + 1 } ^ { \top } \tilde { \phi } _ { t } ( s , a )$ encompasses both the approximation bias and the statistical error.

To bound the error inner product, Cayci et al. (2024) apply the Cauchy-Schwarz inequality over the state-action space and, in the first term, shift the state measure to the training distribution via the mismatch coeficient $\| d ^ { \omega _ { \tau } ^ { * } } / \mu \| _ { \infty }$ :

$$
\mathbb { E } _ { d ^ { \omega _ { \tau } ^ { * } } } \left[ \langle e r r _ { t } , \pi _ { \omega _ { \tau } ^ { * } } - \pi _ { t } \rangle _ { \mathsf { A } } \right] \leq \sqrt { \frac { \| d ^ { \omega _ { \tau } ^ { * } } / \mu \| _ { \infty } } { 1 - \gamma } \cdot \left( \epsilon ( R ) + \epsilon _ { s t a t } \right) } \cdot \sqrt { \mathbb { E } _ { d ^ { \omega _ { \tau } ^ { * } } } \left[ \chi ^ { 2 } ( \pi _ { \omega _ { \tau } ^ { * } } ( \cdot | s ) \| \pi _ { t } ( \cdot | s ) ) \right] } .
$$

Then, they control the $\chi ^ { 2 } \mathrm { - d i v e r g e n c e }$ term via the inequality $\chi ^ { 2 } ( \pi _ { \omega _ { \tau } ^ { * } } ( \cdot | s ) \| \pi _ { t } ( \cdot | s ) ) \ \leq \ 2 / p _ { m i n }$ for any s, depending on the minimum action probability $p _ { m i n }$ . Substituting this bound into the drift inequality (19) and telescoping over T iterations with the diminishing step-size $\eta _ { t } = \mathcal { O } ( ( \tau t ) ^ { - 1 } )$ leads to their final bound described in (18).

Comparison for Regularized MDP. Up to the distinction between $D _ { t }$ and $D _ { t } ^ { \dagger }$ and the precise definitions of $e r r _ { t } ( s , a )$ , the drift inequality (19) is mathematically equivalent to inequality (23) in our proof of the PMD Actor Progress Bound, both of which mirror the Bregman progress bound (14) in Agarwal et al. (2021). A crucial diference between Cayci et al. (2024) and our analysis for the regularized MDP lies in how the error inner product is decoupled and bounded. Their Cauchy-Schwarz argument traps both the approximation and statistical errors under a square root, which is then amplified by the $1 / \sqrt { p _ { m i n } }$ factor.

In contrast, our Part II analysis avoids both the square-root barrier and $1 / \sqrt { p _ { m i n } }$ amplification in deriving the PMD Actor Progress Bound. By exploiting the global joint concentrability assumption (Assumption 9), we simultaneously manage the state measure shift and facilitate the action-space $\chi ^ { 2 } { \mathrm { - t o - K L } }$ divergence translation. This translation allows us to fully decouple the error inner product via Young’s inequality after Cauchy-Schwarz, absorbing the resulting divergence penalty back into the restorative entropy force $- \eta _ { t } \tau D _ { t } ^ { \dagger }$ . This approach frees the approximation and statistical errors from the square-root trap and removes the $1 / \sqrt { p _ { m i n } }$ amplification.

Furthermore, to circumvent the double-loop architecture, our analysis employs the fractional Chung’s lemma to balance the actor and critic step sizes and derive an uncoupled Critic Tracking Error bound. For our single-loop algorithm $( N = 1 , T = T _ { t o t a l } )$ , our theory in Part II establishes an accelerated average-iterate $\mathcal { O } ( T _ { t o t a l } ^ { - 2 / 3 } ) + \mathcal { O } ( \epsilon _ { a p p } )$ bound on the regularized performance gap. This improves upon their double-loop $\tilde { \mathcal { O } } ( T _ { t o t a l } ^ { - 1 / 3 } ) + \mathcal { O } ( \sqrt { \epsilon _ { a p p } } )$ bound.

Implications for Unregularized MDP. Cayci et al. (2024) restrict their analysis entirely to the fixed-τ regularized MDP, and do not provide explicit bounds on the unregularized performance gap. Nevertheless, we discuss implications of their analysis for solving the unregularized MDP.

Extending their guarantees to the unregularized objective in our Deterministic Regime encounters two degeneracies as $\tau  0$ . First, the optimal policy $\pi _ { \omega _ { \tau } ^ { * } }$ approaches a deterministic distribution, so that $p _ { m i n }  0$ and their $1 / \sqrt { p _ { m i n } }$ error factor deteriorates. Second, the minimum eigenvalue of the centered feature covariance satisfies $\lambda _ { \tau }  0$ , eliminating a temperature-uniform strong-convexity constant for the centered least-squares Q-value projection and causing the constants underlying the inner-loop rate $\epsilon _ { s t a t } = \mathcal { O } ( 1 / N )$ to deteriorate.

In the Stochastic Regime, provided $p _ { m i n }$ and $\lambda _ { \tau }$ remain bounded away from zero, their fixedtemperature regularized bound could be combined with a temperature-tuning argument. The resulting convergence rate, based on their $\tilde { \mathcal { O } } ( T _ { t o t a l } ^ { - 1 / 3 } )$ regularized rate up to the approximation error term, would still be slower than ours in Part I: either $\tilde { \mathcal { O } } ( T _ { t o t a l } ^ { - 1 / 6 } )$ under the standard linear entropy penalty, or $\tilde { \mathcal { O } } ( T _ { t o t a l } ^ { - 1 / 3 } )$ if combined with our Exponential Translation Bounds.

## E Technical Details

## E.1 Proofs for Section 2

Proof of Lemma 2.1 (Bounded Regularized Values). By definition, the regularized statevalue is:

$$
\begin{array} { l } { { \displaystyle V _ { \tau } ^ { \pi } ( s ) = \mathbb { E } _ { \pi } \left[ \sum _ { t = 0 } ^ { \infty } \gamma ^ { t } \big ( r ( s _ { t } , a _ { t } ) - \tau \log \pi ( a _ { t } | s _ { t } ) \big ) \ \Bigg | \ s _ { 0 } = s \right] } } \\ { { \displaystyle \quad = \mathbb { E } _ { \pi } \left[ \sum _ { t = 0 } ^ { \infty } \gamma ^ { t } \Big ( \mathbb { E } _ { a _ { t } \sim \pi } [ r ( s _ { t } , a _ { t } ) | s _ { t } ] + \tau H ( \pi ( \cdot | s _ { t } ) ) \Big ) \ \Bigg | \ s _ { 0 } = s \right] . } } \end{array}
$$

Since the reward is bounded in $[ 0 , 1 ]$ and the entropy of a discrete distribution over A is bounded by $0 \leq H \leq \log | \mathcal { A } |$ , the inner term for each step is bounded between 0 and $1 + \tau \log \left| \mathcal { A } \right|$ . Evaluating the geometric series yields $\begin{array} { r } { 0 \leq V _ { \tau } ^ { \pi } ( s ) \leq \frac { 1 + \tau \log | \mathcal { A } | } { 1 - \gamma } } \end{array}$

For the action-value $Q _ { \tau } ^ { \pi } ( s , a )$ , the initial step $t = 0$ uses a fixed action a without an instantaneous entropy penalty, meaning the $t = 0$ term is simply $0 \leq r ( s , a ) \leq 1 ( \leq 1 + \tau \log | \mathcal { A } | )$ ). Summing the remaining discounted terms yields $\begin{array} { r } { 0 \leq Q _ { \tau } ^ { \pi } ( s , a ) \leq \frac { 1 + \tau \log | \mathcal { A } | } { 1 - \gamma } } \end{array}$ , the same as for $V _ { \tau } ^ { \pi } ( s )$ □

Proof of Lemma 2.2 (Structural Bounds of the Algorithm). (i) Ideal Critic Bound. By definition, the ideal uncentered critic is a solution to the uncentered least-squares objective: $\theta _ { \tau } ^ { * } ( \omega _ { t } ) \ : = \ : \bar { \Sigma } _ { u n c } ( \pi _ { t } ) ^ { - 1 } \mathbb { E } _ { d ^ { \pi _ { t } } , \pi _ { t } } [ \phi ( s , a ) Q _ { \tau } ^ { \pi _ { t } } ( s , a ) ]$ . By Assumption 3, the spectral norm of the inverse feature moment matrix is bounded by $\begin{array} { r } { \| \bar { \Sigma } _ { u n c } ( \pi _ { t } ) ^ { - 1 } \| _ { 2 } \le \frac { 1 } { \kappa } } \end{array}$ . For the target vector, applying Jensen’s inequality alongside the Bounded Features (Assumption 1) and Bounded Regularized Values (Lemma 2.1) yields:

$$
\begin{array} { r } { \left\| \mathbb { E } _ { d ^ { \pi _ { t } } , \pi _ { t } } [ \phi ( s , a ) Q _ { \tau } ^ { \pi _ { t } } ( s , a ) ] \right\| _ { 2 } \leq \mathbb { E } _ { d ^ { \pi _ { t } } , \pi _ { t } } \Big [ \| \phi ( s , a ) \| _ { 2 } \big | Q _ { \tau } ^ { \pi _ { t } } ( s , a ) \big | \Big ] \leq B _ { \phi } V _ { m a x } . } \end{array}
$$

Taking the norm of the ideal critic identity and applying the sub-multiplicative property establishes the bound:

$$
\| \theta _ { \tau } ^ { * } ( \omega _ { t } ) \| _ { 2 } \leq \| \bar { \Sigma } _ { u n c } ( \pi _ { t } ) ^ { - 1 } \| _ { 2 } \big \| \mathbb { E } _ { d ^ { \pi _ { t } } , \pi _ { t } } [ \phi ( s , a ) Q _ { \tau } ^ { \pi _ { t } } ( s , a ) ] \big \| _ { 2 } \leq \frac { B _ { \phi } V _ { m a x } } { \kappa } = B _ { \theta } .
$$

(ii) Actor Iterate Bound. Because the actual critic is projected onto $R _ { \theta } = B _ { \theta }$ , we have $\| \theta _ { t + 1 } \| _ { 2 } \leq$ $B _ { \theta }$ . The actor update rule is $\omega _ { t + 1 } = \omega _ { t } + \eta _ { t } ( \theta _ { t + 1 } - \tau \omega _ { t } ) = ( 1 - \tau \eta _ { t } ) \omega _ { t } + \eta _ { t } \theta _ { t + 1 }$ . Assuming inductively that $\begin{array} { r } { \| \omega _ { t } \| _ { 2 } \leq \frac { B _ { \theta } } { \tau } } \end{array}$ , and noting $( 1 - \tau \eta _ { t } ) \ge 0$ since $\eta _ { t } \leq 1 / \tau$ , the triangle inequality yields:

$$
\| \omega _ { t + 1 } \| _ { 2 } \leq ( 1 - \tau \eta _ { t } ) \| \omega _ { t } \| _ { 2 } + \eta _ { t } \| \theta _ { t + 1 } \| _ { 2 } \leq ( 1 - \tau \eta _ { t } ) \frac { B _ { \theta } } { \tau } + \eta _ { t } B _ { \theta } = \frac { B _ { \theta } } { \tau } .
$$

Since $\tau \leq \tau _ { m a x }$ , the initialization satisfies $\begin{array} { r } { \| \omega _ { 0 } \| _ { 2 } \le \frac { B _ { \theta } } { \tau _ { m a x } } \le \frac { B _ { \theta } } { \tau } } \end{array}$ . Therefore, by induction, the bound

holds for all $t \geq 0$

(iii) Update Direction Bounds. For the critic gradient, by the bounded feature assumption $( \| \phi ( s , a ) \| _ { 2 } \le B _ { \phi } )$ and the projection step $( \lVert \theta _ { t } \rVert _ { 2 } \leq B _ { \theta } )$ , we bound the squared norm using the inequality $( x + y ) ^ { 2 } \leq 2 x ^ { 2 } + 2 y ^ { 2 }$

$$
\begin{array} { r } { \Vert g _ { t } ^ { c r } ( s _ { t } , a _ { t } ) \Vert _ { 2 } ^ { 2 } = \vert \theta _ { t } ^ { \top } \phi _ { t } - \hat { Q } _ { t } \vert ^ { 2 } \Vert \phi _ { t } \Vert _ { 2 } ^ { 2 } \le 2 B _ { \phi } ^ { 2 } \left( ( \theta _ { t } ^ { \top } \phi _ { t } ) ^ { 2 } + \hat { Q } _ { t } ^ { 2 } \right) \le 2 B _ { \phi } ^ { 2 } \left( B _ { \theta } ^ { 2 } B _ { \phi } ^ { 2 } + \hat { Q } _ { t } ^ { 2 } \right) . } \end{array}
$$

Taking the conditional expectation and substituting the Q-value second moment bound $\mathbb { E } [ \hat { Q } _ { t } ^ { 2 } \ |$ $\mathcal { F } _ { t } ] \leq Q _ { m a x } ^ { 2 }$ yields the bound $G _ { c r } ^ { 2 }$ :

$$
\mathbb { E } \big [ \| g _ { t } ^ { c r } ( s _ { t } , a _ { t } ) \| _ { 2 } ^ { 2 } \mid \mathcal { F } _ { t } \big ] \leq 2 B _ { \phi } ^ { 2 } \left( B _ { \theta } ^ { 2 } B _ { \phi } ^ { 2 } + Q _ { m a x } ^ { 2 } \right) = G _ { c r } ^ { 2 } .
$$

For the actor’s update direction, we apply the triangle inequality and leverage the actor iterate bound $\begin{array} { r } { \| \omega _ { t } \| _ { 2 } \leq \frac { B _ { \theta } } { \tau } } \end{array}$ from the previous step:

$$
\begin{array} { r l } & { \| g _ { t } ^ { a c } \| _ { 2 } = \| \theta _ { t + 1 } - \tau \omega _ { t } \| _ { 2 } \leq \| \theta _ { t + 1 } \| _ { 2 } + \tau \| \omega _ { t } \| _ { 2 } } \\ & { \qquad \leq B _ { \theta } + \tau \frac { B _ { \theta } } { \tau } = 2 B _ { \theta } = G _ { a c } . } \end{array}
$$

Crucially, the temperature τ cancels out the $1 / \tau$ dependence in the actor iterate bound. □

Proof of Lemma 2.3 (Lipschitz Ideal Uncentered Critic). By definition, the ideal uncentered critic is $\theta _ { \tau } ^ { * } ( \omega ) = \bar { \Sigma } ( \omega ) ^ { - 1 } b ( \omega )$ , where $\bar { \Sigma } ( \omega ) \triangleq \bar { \Sigma } _ { u n c } ( \pi _ { \omega } )$ and $b ( \omega ) \triangleq \mathbb { E } _ { d ^ { \pi _ { \omega } } , \pi _ { \omega } } [ \phi ( s , a ) Q _ { \tau } ^ { \pi _ { \omega } } ( s , a ) ]$ By the Policy Gradient Theorem, the derivative of any expectation over the discounted state-action visitation measure evaluates as:

$$
\nabla _ { \omega } \mathbb { E } _ { d ^ { \pi _ { \omega } } , \pi _ { \omega } } [ f ( s , a ) ] = \mathbb { E } _ { d ^ { \pi _ { \omega } } , \pi _ { \omega } } [ \nabla _ { \omega } f ( s , a ) ] + \mathbb { E } _ { d ^ { \pi _ { \omega } } , \pi _ { \omega } } \Big [ \nabla _ { \omega } \log \pi _ { \omega } ( a | s ) Q _ { f } ^ { \pi _ { \omega } } ( s , a ) \Big ]
$$

where $\begin{array} { r } { Q _ { f } ^ { \pi _ { \omega } } ( s , a ) \ \triangleq \ \mathbb { E } _ { \pi _ { \omega } } \big [ \sum _ { t = 0 } ^ { \infty } \gamma ^ { t } f ( s _ { t } , a _ { t } ) \ \mid \ s _ { 0 } \ = \ s , a _ { 0 } \ = \ a \big ] } \end{array}$ . For a log-linear policy, the score function is bounded by $\| \nabla _ { \omega } \log \pi _ { \omega } \| _ { 2 } ~ = ~ \| \phi ( s , a ) - \mathbb { E } _ { \pi _ { \omega } } [ \phi ] \| _ { 2 } ~ \le ~ 2 B _ { \phi }$ . Furthermore, the accumulated sum is bounded by the supremum over the entire state-action space: $\Vert Q _ { f } ^ { \pi _ { \omega } } ( s , a ) \Vert _ { 2 } \ \leq$ $\begin{array} { r l } { } & { { \frac { 1 } { 1 - \gamma } } \operatorname* { s u p } _ { s ^ { \prime } , a ^ { \prime } } \| f ( s ^ { \prime } , a ^ { \prime } ) \| _ { 2 } } \end{array}$ . Taking the norm and applying the triangle inequality yields the general derivative bound:

$$
\| \nabla _ { \omega } \mathbb { E } _ { d ^ { \pi _ { \omega } } , \pi _ { \omega } } [ f ] \| _ { 2 } \le \mathbb { E } _ { d ^ { \pi _ { \omega } } , \pi _ { \omega } } [ \| \nabla _ { \omega } f \| _ { 2 } ] + \frac { 2 B _ { \phi } } { 1 - \gamma } \operatorname* { s u p } _ { s , a } \| f ( s , a ) \| _ { 2 } .\tag{20}
$$

We bound the global Lipschitz continuity of the individual components $\bar { \Sigma } ( \omega )$ and $b ( \omega )$

• Moment Matrix: Because the raw feature $\phi ( s , a )$ is independent of ω, we have $\nabla _ { \omega } ( \phi \phi ^ { \top } ) = 0$ Applying (20) yields a global bound on the gradient: $\begin{array} { r } { \| \nabla _ { \omega } \bar { \Sigma } ( \omega ) \| _ { 2 } \leq 0 + \frac { 2 B _ { \phi } } { 1 - \gamma } \operatorname* { s u p } _ { s , a } \| \phi \phi ^ { \top } \| _ { 2 } \leq } \end{array}$ $\frac { 2 B _ { \phi } ^ { 3 } } { 1 - \gamma } \triangleq L _ { \Sigma }$ . By the Mean Value Theorem, for any $\omega , \omega ^ { \prime } \in \mathbb { R } ^ { d } , \| \bar { \Sigma } ( \omega ) - \bar { \Sigma } ( \omega ^ { \prime } ) \| _ { 2 } \le L _ { \Sigma } \| \omega - \omega ^ { \prime } \| _ { 2 }$ • Target Vector: For $f ( s , a ) = \phi ( s , a ) Q _ { \tau } ^ { \pi _ { \omega } } ( s , a )$ , we have $\nabla _ { \omega } f = \phi ( \nabla _ { \omega } Q _ { \tau } ^ { \pi _ { \omega } } ) ^ { \top }$ . Let $r _ { m a x } \triangleq$ $1 + \tau \log | \mathcal { A } |$ . From Lemma 2.1, $\begin{array} { r } { | Q _ { \tau } ^ { \pi _ { \omega } } | \le \frac { r _ { m a x } } { 1 - \gamma } } \end{array}$ . From Step (ii) in the proof of Lemma $5 . 2 .$ $\begin{array} { r } { \| \nabla _ { \omega } Q _ { \tau } ^ { \pi _ { \omega } } \| _ { 2 } \leq \frac { 4 \gamma B _ { \phi } r _ { m a x } } { ( 1 - \gamma ) ^ { 2 } } } \end{array}$ . Applying (20) gives a global bound: $\begin{array} { r } { \| \nabla _ { \omega } b ( \omega ) \| _ { 2 } \ \leq \ \frac { 4 \gamma B _ { \phi } ^ { 2 } r _ { m a x } } { ( 1 - \gamma ) ^ { 2 } } \ + } \end{array}$ $\begin{array} { r } { \frac { 2 B _ { \phi } ^ { 2 } r _ { m a x } } { ( 1 - \gamma ) ^ { 2 } } \leq \frac { 6 B _ { \phi } ^ { 2 } r _ { m a x } } { ( 1 - \gamma ) ^ { 2 } } \triangleq L _ { b } } \end{array}$ . Thus, for any $\omega , \omega ^ { \prime } \in \mathbb { R } ^ { d } , \| b ( \omega ) - b ( \omega ^ { \prime } ) \| _ { 2 } \le L _ { b } \| \omega - \omega ^ { \prime } \| _ { 2 }$

To compute the diference in the ideal critic, we use the algebraic resolvent identity $A ^ { - 1 } - B ^ { - 1 } =$ $A ^ { - 1 } ( B - A ) B ^ { - 1 }$ to expand $\theta _ { \tau } ^ { * } ( \omega ) - \theta _ { \tau } ^ { * } ( \omega ^ { \prime } )$ :

$$
\begin{array} { r l } & { \theta _ { \tau } ^ { * } ( \omega ) - \theta _ { \tau } ^ { * } ( \omega ^ { \prime } ) = \bar { \Sigma } ( \omega ) ^ { - 1 } b ( \omega ) - \bar { \Sigma } ( \omega ^ { \prime } ) ^ { - 1 } b ( \omega ^ { \prime } ) } \\ & { \qquad = \bar { \Sigma } ( \omega ) ^ { - 1 } \big [ b ( \omega ) - b ( \omega ^ { \prime } ) \big ] + \big [ \bar { \Sigma } ( \omega ) ^ { - 1 } - \bar { \Sigma } ( \omega ^ { \prime } ) ^ { - 1 } \big ] b ( \omega ^ { \prime } ) } \\ & { \qquad = \bar { \Sigma } ( \omega ) ^ { - 1 } \big [ b ( \omega ) - b ( \omega ^ { \prime } ) \big ] + \bar { \Sigma } ( \omega ) ^ { - 1 } \big [ \bar { \Sigma } ( \omega ^ { \prime } ) - \bar { \Sigma } ( \omega ) \big ] \bar { \Sigma } ( \omega ^ { \prime } ) ^ { - 1 } b ( \omega ^ { \prime } ) . } \end{array}
$$

Taking the norm and applying the triangle and sub-multiplicative properties:

$$
\begin{array} { r } { \| \theta _ { \tau } ^ { * } ( \omega ) - \theta _ { \tau } ^ { * } ( \omega ^ { \prime } ) \| _ { 2 } \leq \| \bar { \Sigma } ( \omega ) ^ { - 1 } \| _ { 2 } \| b ( \omega ) - b ( \omega ^ { \prime } ) \| _ { 2 } + \| \bar { \Sigma } ( \omega ) ^ { - 1 } \| _ { 2 } \| \bar { \Sigma } ( \omega ^ { \prime } ) - \bar { \Sigma } ( \omega ) \| _ { 2 } \| \bar { \Sigma } ( \omega ^ { \prime } ) ^ { - 1 } \| _ { 2 } \| b ( \omega ^ { \prime } ) \| _ { 2 } . } \end{array}
$$

For two parameters $\omega$ and $\omega ^ { \prime }$ satisfying $\bar { \Sigma } ( \omega ) \succeq \kappa I$ and $\bar { \Sigma } ( \omega ^ { \prime } ) \succeq \kappa I$ , we have the spectral limits $\begin{array} { r } { \| \bar { \Sigma } ( \omega ) ^ { - 1 } \| _ { 2 } \leq \frac { 1 } { \kappa } } \end{array}$ and $\begin{array} { r } { \| \bar { \Sigma } ( \omega ^ { \prime } ) ^ { - 1 } \| _ { 2 } \leq \frac { 1 } { \kappa } } \end{array}$ . The target vector is bounded by $\begin{array} { r } { \| b ( \omega ^ { \prime } ) \| _ { 2 } \le \frac { B _ { \phi } r _ { m a x } } { 1 - \gamma } } \end{array}$ Substituting our Lipschitz components yields:

$$
\begin{array} { r l r } {  { \| \theta _ { \tau } ^ { * } ( \omega ) - \theta _ { \tau } ^ { * } ( \omega ^ { \prime } ) \| _ { 2 } \le ( \frac { 1 } { \kappa } ) L _ { b } \| \omega - \omega ^ { \prime } \| _ { 2 } + ( \frac { 1 } { \kappa } ) ^ { 2 } L _ { \Sigma } \| \omega - \omega ^ { \prime } \| _ { 2 } ( \frac { B _ { \phi } r _ { m a x } } { 1 - \gamma } ) } } \\ & { } & { = \lceil \frac { 1 } { \kappa } \frac { 6 B _ { \phi } ^ { 2 } r _ { m a x } } { ( 1 - \gamma ) ^ { 2 } } + \frac { 1 } { \kappa ^ { 2 } } \frac { 2 B _ { \phi } ^ { 3 } } { 1 - \gamma } \frac { B _ { \phi } r _ { m a x } } { 1 - \gamma } \rceil \| \omega - \omega ^ { \prime } \| _ { 2 } . ~ } \end{array}
$$

Factoring out the shared constant $\frac { 2 B _ { \phi } ^ { 2 } r _ { m a x } } { \kappa ( 1 - \gamma ) ^ { 2 } }$ yields exactly $\boldsymbol { L } _ { \boldsymbol { \theta } , \tau }$ , completing the proof. □

## E.2 Proofs for Section 3

Proof of Lemma 3.1 (Regularized Performance Diference). For any policy ˜π, the soft state-value function satisfies the regularized Bellman equation:

$$
V _ { \tau } ^ { \tilde { \pi } } ( s ) = \mathbb { E } _ { a \sim \tilde { \pi } ( \cdot | s ) } \left[ r ( s , a ) - \tau \log \tilde { \pi } ( a | s ) + \gamma \mathbb { E } _ { s ^ { \prime } \sim \mathcal { P } ( \cdot | s , a ) } [ V _ { \tau } ^ { \tilde { \pi } } ( s ^ { \prime } ) ] \right] .
$$

We subtract the soft value function of a reference policy $V _ { \tau } ^ { \pi } ( s )$ from both sides. To create a telescoping sum, we add and subtract $\gamma \mathbb { E } _ { s ^ { \prime } } [ V _ { \tau } ^ { \pi } ( s ^ { \prime } ) ]$ inside the expectation:

$$
V _ { \tau } ^ { \widehat \pi } ( s ) - V _ { \tau } ^ { \pi } ( s ) = \mathbb { E } _ { a \sim \widehat { \pi } } \left[ r ( s , a ) - \tau \log \widehat { \pi } ( a | s ) + \gamma \mathbb { E } _ { s ^ { \prime } } [ V _ { \tau } ^ { \widehat \pi } ( s ^ { \prime } ) - V _ { \tau } ^ { \pi } ( s ^ { \prime } ) ] + \gamma \mathbb { E } _ { s ^ { \prime } } [ V _ { \tau } ^ { \pi } ( s ^ { \prime } ) ] \right] - V _ { \tau } ^ { \pi } ( s ) .
$$

By the definition of the soft action-value function, $Q _ { \tau } ^ { \pi } ( s , a ) = r ( s , a ) + \gamma \mathbb { E } _ { s ^ { \prime } } [ V _ { \tau } ^ { \pi } ( s ^ { \prime } ) ]$ . Substituting this simplifies the expression to:

$$
V _ { \tau } ^ { \bar { \pi } } ( s ) - V _ { \tau } ^ { \pi } ( s ) = \mathbb { E } _ { a \sim \bar { \pi } } \big [ Q _ { \tau } ^ { \pi } ( s , a ) - \tau \log \bar { \pi } ( a | s ) - V _ { \tau } ^ { \pi } ( s ) \big ] + \gamma \mathbb { E } _ { a \sim \bar { \pi } , s ^ { \prime } \sim \mathcal { P } } \big [ V _ { \tau } ^ { \bar { \pi } } ( s ^ { \prime } ) - V _ { \tau } ^ { \pi } ( s ^ { \prime } ) \big ] .
$$

By unrolling this recursive relationship over the infinite horizon trajectory generated by following $\tilde { \pi }$ from a fixed initial state $s _ { 0 }$ , the discounted future diferences evaluate to the expectation over $d _ { s _ { 0 } } ^ { \tilde { \pi } }$ , the discounted state visitation distribution conditioned on s<sub>0</sub>:

$$
V _ { \tau } ^ { \tilde { \pi } } ( s _ { 0 } ) - V _ { \tau } ^ { \pi } ( s _ { 0 } ) = \frac { 1 } { 1 - \gamma } \mathbb { E } _ { s \sim d _ { s _ { 0 } } ^ { \tilde { \pi } } } \left[ \sum _ { a \in A } \tilde { \pi } ( a | s ) \big ( Q _ { \tau } ^ { \pi } ( s , a ) - \tau \log \tilde { \pi } ( a | s ) - V _ { \tau } ^ { \pi } ( s ) \big ) \right] .
$$

Taking the expectation over the initial state distribution $s _ { 0 } ~ \sim ~ \mu$ yields the global regularized objective diference $J _ { \tau } ( \tilde { \pi } ) - J _ { \tau } ( \pi )$ on the left-hand side. Because $d ^ { \tilde { \pi } } ( s ) = \mathbb { E } _ { s _ { 0 } \sim \mu } [ d _ { s _ { 0 } } ^ { \tilde { \pi } } ( s ) ]$ , we obtain the final identity:

$$
J _ { \tau } ( \tilde { \pi } ) - J _ { \tau } ( \pi ) = \frac { 1 } { 1 - \gamma } \mathbb { E } _ { s \sim d ^ { \tilde { \pi } } } \left[ \sum _ { a \in A } \tilde { \pi } ( a | s ) \big ( Q _ { \tau } ^ { \pi } ( s , a ) - \tau \log { \tilde { \pi } ( a | s ) } - V _ { \tau } ^ { \pi } ( s ) \big ) \right] ,
$$

completing the proof.

Proof of Lemma 3.2 (Global Regularized Suboptimality Identity). (i) Optimal Soft Advantage and Boltzmann Target. By definition, the regularized optimal value function maximizes the soft Bellman equation over all possible probability distributions $\tilde { \pi } ( \cdot | s )$

$$
V _ { \tau } ^ { * } ( s ) = \operatorname* { m a x } _ { \tilde { \pi } } \sum _ { a } \tilde { \pi } ( a | s ) \big [ Q _ { \tau } ^ { * } ( s , a ) - \tau \log \tilde { \pi } ( a | s ) \big ] .
$$

By factoring out τ and introducing the normalizing partition function $\begin{array} { r } { Z ( s ) \triangleq \sum _ { a ^ { \prime } } \exp ( Q _ { \tau } ^ { \ast } ( s , a ^ { \prime } ) / \tau ) } \end{array}$ ， we rewrite the term inside the brackets:

$$
\tau \sum _ { a } \tilde { \pi } ( a | s ) \left[ \log \left( \frac { \exp ( Q _ { \tau } ^ { * } ( s , a ) / \tau ) / Z ( s ) } { \tilde { \pi } ( a | s ) } \right) + \log Z ( s ) \right] .
$$

Recognizing that $\frac { \exp ( Q _ { \tau } ^ { * } ( s , a ) / \tau ) } { Z ( s ) }$ is exactly the Boltzmann distribution, and noting that $\textstyle \sum _ { a } { \tilde { \pi } } ( a | s )$ log Z(s) = log $Z ( s )$ because probabilities sum to 1, the maximum reduces to:

$$
V _ { \tau } ^ { * } ( s ) = \operatorname* { m a x } _ { \tilde { \pi } } [ \tau \log Z ( s ) - \tau D _ { K L } ( \tilde { \pi } ( \cdot | s ) | | \frac { \exp ( Q _ { \tau } ^ { * } ( s , \cdot ) / \tau ) } { Z ( s ) }  ) ] .
$$

Because the Kullback-Leibler divergence is non-negative and is zero if and only if the distributions match, this expression is uniquely maximized when ˜π is exactly the Boltzmann policy $\pi _ { \tau } ^ { * } ( a | s ) =$ $\frac { \exp ( Q _ { \tau } ^ { * } ( s , a ) / \tau ) } { Z ( s ) }$ . Substituting this optimal policy back zeroes out the KL divergence, yielding the identity for the regularized optimal state-value and action-value:

$$
V _ { \tau } ^ { * } ( s ) = \tau \log Z ( s ) = \tau \log \sum _ { a ^ { \prime } } \exp \left( \frac { Q _ { \tau } ^ { * } ( s , a ^ { \prime } ) } { \tau } \right) .
$$

Consequently, the optimal Boltzmann policy can be equivalently written as $\begin{array} { r } { \pi _ { \tau } ^ { * } ( a | s ) = \frac { \exp \left( Q _ { \tau } ^ { * } ( s , a ) / \tau \right) } { \exp \left( V _ { \tau } ^ { * } ( s ) / \tau \right) } } \end{array}$ Taking the logarithm and multiplying by τ yields an exact relation for the optimal log-probability:

$$
\tau \log \pi _ { \tau } ^ { * } ( a | s ) = Q _ { \tau } ^ { * } ( s , a ) - V _ { \tau } ^ { * } ( s ) .
$$

That is, the regularized advantage is zero: $A _ { \tau } ^ { \pi _ { \tau } ^ { * } } ( s , a ) = Q _ { \tau } ^ { * } ( s , a ) - \tau \log \pi _ { \tau } ^ { * } ( a | s ) - V _ { \tau } ^ { * } ( s ) = 0 .$

(ii) Synthesis via KL Divergence. By the Regularized Performance Diference (Lemma 3.1) evaluated between $\pi$ and $\pi _ { \tau } ^ { * }$ , the objective diference is:

$$
J _ { \tau } ( \pi ) - J _ { \tau } ( \pi _ { \tau } ^ { * } ) = \frac { 1 } { 1 - \gamma } \mathbb { E } _ { s \sim d ^ { \pi } } \left[ \sum _ { a } \pi ( a | s ) \big ( Q _ { \tau } ^ { * } ( s , a ) - \tau \log \pi ( a | s ) - V _ { \tau } ^ { * } ( s ) \big ) \right] .
$$

We substitute the preceding identity for $Q _ { \tau } ^ { \ast } ( s , a ) - V _ { \tau } ^ { \ast } ( s )$ into this objective diference:

$$
J _ { \tau } ( \pi ) - J _ { \tau } ( \pi _ { \tau } ^ { * } ) = \frac { 1 } { 1 - \gamma } \mathbb { E } _ { s \sim d ^ { \pi } } \left[ \sum _ { a } \pi ( a | s ) \big ( \tau \log \pi _ { \tau } ^ { * } ( a | s ) - \tau \log \pi ( a | s ) \big ) \right] .
$$

Factoring out −τ isolates the definition of the Kullback-Leibler divergence:

$$
J _ { \tau } ( \pi ) - J _ { \tau } ( \pi _ { \tau } ^ { * } ) = - \frac { \tau } { 1 - \gamma } \mathbb { E } _ { s \sim d ^ { \pi } } \left[ \sum _ { a } \pi ( a | s ) \log \frac { \pi ( a | s ) } { \pi _ { \tau } ^ { * } ( a | s ) } \right] = - \frac { \tau } { 1 - \gamma } \mathbb { E } _ { s \sim d ^ { \pi } } \left[ D _ { K L } ( \pi ( \cdot | s ) \| \pi _ { \tau } ^ { * } ( \cdot | s ) ) \right] .
$$

This matches the stated identity, completing the proof.

Proof of Lemma 3.3 (Regularized Suboptimality Decompositions). (i) Extracting KL Divergence. Starting from the Regularized Performance Diference (Lemma 3.1) evaluated between the comparator π and the training policy $\pi _ { t } .$ , we add and subtract τ log $\pi _ { t }$ to extract the KL divergence:

$$
\begin{array} { r l } {  { ( 1 - \gamma ) \big ( J _ { \tau } ( \boldsymbol { \pi } ) - J _ { \tau } ( \boldsymbol { \pi } _ { t } ) \big ) = \mathbb { E } _ { d ^ { \tau } } [ \sum _ { a } \pi \big ( Q _ { \tau } ^ { \pi _ { t } } - \tau \log \pi \big ) - V _ { \tau } ^ { \pi _ { t } } ] } } \\ & { = \mathbb { E } _ { d ^ { \tau } } [ \sum _ { a } \pi \big ( Q _ { \tau } ^ { \pi _ { t } } - \tau \log \pi _ { t } \big ) - \tau D _ { K L } ( \pi \| \pi _ { t } ) - V _ { \tau } ^ { \pi _ { t } } ] . } \end{array}
$$

Substituting $\begin{array} { r } { V _ { \tau } ^ { \pi _ { t } } ( s ) = \sum _ { a } \pi _ { t } ( a | s ) \big ( Q _ { \tau } ^ { \pi _ { t } } ( s , a ) - \tau \log \pi _ { t } ( a | s ) \big ) } \end{array}$ combines the action summations into a single inner product:

$$
( 1 - \gamma ) \big ( J _ { \tau } ( \pi ) - J _ { \tau } ( \pi _ { t } ) \big ) = - \tau \mathbb { E } _ { d ^ { \pi } } [ D _ { K L } ( \pi \| \pi _ { t } ) ] + \mathbb { E } _ { d ^ { \pi } } \Big [ \langle Q _ { \tau } ^ { \pi _ { t } } - \tau \log \pi _ { t } , \pi - \pi _ { t } \rangle _ { \cal A } \Big ] .
$$

(ii) Injecting the Policy Parameterization. We substitute the log-linear identity of the actor policy, τ log $\pi _ { t } ( a | s ) = \tau \omega _ { t } ^ { \top } \phi ( s , a ) - \tau \log Z _ { t } ( s )$ . Because τ log $Z _ { t } ( s )$ is a state-dependent baseline, it acts as a constant in the action inner product and cancels against $\begin{array} { r } { \sum _ { a } ( \pi - \pi _ { t } ) = 0 } \end{array}$ :

$$
( 1 - \gamma ) \big ( J _ { \tau } ( \pi ) - J _ { \tau } ( \pi _ { t } ) \big ) = - \tau \mathbb { E } _ { d ^ { \tau } } [ D _ { K L } ( \pi \| \pi _ { t } ) ] + \mathbb { E } _ { d ^ { \pi } } \Big [ \langle Q _ { \tau } ^ { \pi _ { t } } - \tau \omega _ { t } ^ { \top } \phi , \pi - \pi _ { t } \rangle _ { A } \Big ] .
$$

(iii) Ideal Critic Substitution. We substitute the ideal uncentered critic $\begin{array} { r l } { Q _ { \tau } ^ { \pi _ { t } } ( s , a ) } & { { } = } \end{array}$

$\theta _ { \tau } ^ { * } ( \omega _ { t } ) ^ { \top } \phi ( s , a ) + \epsilon _ { t } ( s , a )$ into the inner product, which yields the first stated decomposition:

$$
\begin{array} { r } { ( 1 - \gamma ) \big ( J _ { \tau } ( \pi ) - J _ { \tau } ( \pi _ { t } ) \big ) = - \tau \mathbb { E } _ { d ^ { \tau } } [ D _ { K L } ( \pi \| \pi _ { t } ) ] + \mathbb { E } _ { d ^ { \tau } } \Big [ \langle ( \theta _ { \tau } ^ { * } ( \omega _ { t } ) - \tau \omega _ { t } ) ^ { \top } \phi , \pi - \pi _ { t } \rangle _ { \mathcal A } \Big ] + \mathbb { E } _ { d ^ { \tau } } \big [ \langle \epsilon _ { t } , \pi - \pi _ { t } \rangle _ { \mathcal A } \big ] . } \end{array}
$$

(iv) Actual Critic Separation. By definition, the critic tracking error is $e _ { t } = \theta _ { t + 1 } - \theta _ { \tau } ^ { * } ( \omega _ { t } )$ . We substitute $\theta _ { \tau } ^ { * } ( \omega _ { t } ) = \theta _ { t + 1 } - e _ { t }$ directly into the ideal algorithmic progress:

$$
\begin{array} { r } { ( \theta _ { \tau } ^ { * } ( \omega _ { t } ) - \tau \omega _ { t } ) ^ { \top } \phi = ( \theta _ { t + 1 } - \tau \omega _ { t } ) ^ { \top } \phi - e _ { t } ^ { \top } \phi . } \end{array}
$$

Substituting this identity into the ideal critic decomposition separates the algorithmic progress driven by the physical update $\theta _ { t + 1 }$ from the statistical tracking deviation $e _ { t } .$ , yielding the second stated decomposition and completing the proof. □

Proof of Lemma 3.4 (Policy Entropy Decomposition Bound). We evaluate the entropy of the action distribution by considering a two-step hierarchical decision process for a given state s. We define a binary indicator variable $I \in \{ 0 , 1 \}$ such that $I = 0$ if the selected action is $a ^ { * } ( s )$ , and $I = 1$ if the selected action is any $a \neq a ^ { * } ( s )$ . The probabilities given state s are $P ( I = 0 \mid s ) = 1 - q ( s )$ and $P ( I = 1 \mid s ) = q ( s )$ . Because I is a deterministic function of the chosen action, the joint entropy equals the entropy of the policy: $H ( \pi ( \cdot | s ) , I ) = H ( \pi ( \cdot | s ) )$ . By the Chain Rule of Entropy, the joint entropy can be decomposed into the marginal entropy of the indicator and the conditional entropy of the action:

$$
H ( \pi ( \cdot | s ) ) = H ( I ) + H ( \pi ( \cdot | s ) \mid I ) .
$$

We bound each term independently:

• Binary Entropy $( H ( I ) )$ : The indicator is a Bernoulli random variable with parameter $q ( s )$ Its entropy is exactly the binary entropy function:

$$
H ( I ) = - q ( s ) \log q ( s ) - ( 1 - q ( s ) ) \log ( 1 - q ( s ) ) .
$$

• Conditional Entropy $( H ( \pi ( \cdot | s ) \mid I ) )$ : By definition, $H ( \pi ( \cdot | s ) \mid I ) = P ( I = 0 \mid s ) H ( \pi ( \cdot | s ) \mid$ $I = 0 ) + P ( I = 1 \mid s ) H ( \pi ( \cdot | s ) \mid I = 1 )$ . If $I = 0$ , the action is known to be $a ^ { * } ( s )$ , meaning the uncertainty $H ( \pi ( \cdot | s ) \mid I = 0 ) = 0$ . If $I = 1$ , the action is distributed among the $| { \mathcal { A } } | - 1$ remaining actions. The maximum possible entropy for a distribution over $| { \mathcal { A } } | - 1$ elements is $\log ( | \mathcal { A } | - 1 )$ . Therefore, the conditional entropy is bounded by:

$$
H ( \pi ( \cdot | s ) \mid I ) \leq ( 1 - q ( s ) ) \cdot 0 + q ( s ) \log ( | A | - 1 ) \leq q ( s ) \log | A | .
$$

Substituting these two bounds back into the Chain Rule decomposition yields the stated result. □

Proof of Lemma 3.5 (Uniform Regularized Value Bound). For any arbitrary policy $\pi ,$ let $V _ { 0 } ^ { \pi } ( s )$ be its standard unregularized expected return, defined as:

$$
V _ { 0 } ^ { \pi } ( s ) = \mathbb { E } _ { \pi } \left[ \sum _ { t = 0 } ^ { \infty } \gamma ^ { t } r ( s _ { t } , a _ { t } ) \Bigg | s _ { 0 } = s \right] .
$$

Let $V _ { \tau } ^ { \pi } ( s )$ be its entropy-regularized expected return, defined as:

$$
V _ { \tau } ^ { \pi } ( s ) = V _ { 0 } ^ { \pi } ( s ) + \tau \mathbb { E } _ { \pi } \left[ \sum _ { t = 0 } ^ { \infty } \gamma ^ { t } H ( \pi ( \cdot | s _ { t } ) ) \biggm | s _ { 0 } = s \right] .
$$

Because the Shannon entropy is bounded by $0 \leq H ( \pi ( \cdot | s ) ) \leq \log | { \mathcal { A } } |$ , the discounted sum of entropy is bounded between 0 and $\frac { \log | \mathcal { A } | } { 1 - \gamma }$ . Thus, for any policy $\pi ,$ the regularized and unregularized values satisfy:

$$
V _ { 0 } ^ { \pi } ( s ) \leq V _ { \tau } ^ { \pi } ( s ) \leq V _ { 0 } ^ { \pi } ( s ) + \frac { \tau \log | \mathcal { A } | } { 1 - \gamma } .\tag{21}
$$

We evaluate the optimal policies against each other to establish tight upper and lower bounds on $V _ { \tau } ^ { * } ( s )$

• Lower Bound: Because $\pi _ { \tau } ^ { * }$ maximizes the regularized objective, its regularized value is at least as high as the regularized value of the unregularized globally optimal policy $\pi _ { 0 } ^ { * }$ . Thus, $V _ { \tau } ^ { * } ( s ) = V _ { \tau } ^ { \pi _ { \tau } ^ { * } } ( s ) \geq V _ { \tau } ^ { \pi _ { 0 } ^ { * } } ( s )$ . Applying (21) to $\pi _ { 0 } ^ { * }$ yields $V _ { \tau } ^ { \pi _ { 0 } ^ { * } } ( s ) \geq V _ { 0 } ^ { \pi _ { 0 } ^ { * } } ( s ) = V _ { 0 } ^ { * } ( s )$ . Therefore, $V _ { \tau } ^ { * } ( s ) \geq V _ { 0 } ^ { * } ( s )$

• Upper Bound: Because $\pi _ { 0 } ^ { * }$ maximizes the unregularized objective, its unregularized value is at least as high as the unregularized value of the regularized globally optimal policy $\pi _ { \tau } ^ { * }$ . Thus, $V _ { 0 } ^ { * } ( s ) = V _ { 0 } ^ { \pi _ { 0 } ^ { * } } ( s ) \geq V _ { 0 } ^ { \pi _ { \tau } ^ { * } } ( s )$ . Applying (21) to the soft policy $\pi _ { \tau } ^ { * }$ gives $\begin{array} { r } { V _ { \tau } ^ { * } ( s ) \leq V _ { 0 } ^ { \pi _ { \tau } ^ { * } } ( s ) + \frac { \tau \log | \mathcal { A } | } { 1 - \gamma } } \end{array}$ Substituting the first inequality yields $\begin{array} { r } { V _ { \tau } ^ { * } ( s ) \le V _ { 0 } ^ { * } ( s ) + \frac { \tau \log | \mathcal { A } | } { 1 - \gamma } } \end{array}$

Combining the bounds proves the result for the state-value functions. For the action-value functions, we invoke the standard Bellman equations: $Q _ { 0 } ^ { \pi } ( s , a ) = r ( s , a ) + \gamma \mathbb { E } _ { s ^ { \prime } } [ V _ { 0 } ^ { \pi } ( s ^ { \prime } ) ]$ . Because the reward and transition dynamics are identical, subtracting the two equations yields:

$$
\begin{array} { r } { Q _ { \tau } ^ { * } ( s , a ) - Q _ { 0 } ^ { * } ( s , a ) = \gamma \mathbb { E } _ { s ^ { \prime } \sim \mathcal { P } ( \cdot \vert s , a ) } \big [ V _ { \tau } ^ { * } ( s ^ { \prime } ) - V _ { 0 } ^ { * } ( s ^ { \prime } ) \big ] . } \end{array}
$$

Since $\begin{array} { r } { 0 \le V _ { \tau } ^ { \ast } ( s ^ { \prime } ) - V _ { 0 } ^ { \ast } ( s ^ { \prime } ) \le \frac { \tau \log | \mathcal { A } | } { 1 - \gamma } } \end{array}$ for all states, the expectation preserves the bound. Thus $\begin{array} { r } { 0 \le Q _ { \tau } ^ { * } ( s , a ) - Q _ { 0 } ^ { * } ( s , a ) \le \frac { \gamma \tau \log | { \cal A } | } { 1 - \gamma } \le \frac { \tau \log | { \cal A } | } { 1 - \gamma } } \end{array}$ , completing the proof. □

Proof of Theorem 3.1 (Exponential Translation of Regularized Suboptimality). The proof proceeds in four steps. First, we upper-bound the unregularized performance gap by the expected suboptimal probability mass. Second, we express the regularized performance gap as the reverse KL divergence and lower-bound its cross-entropy and policy entropy components using the suboptimal mass. Third, we combine these lower bounds and invert the resulting non-linear inequality to isolate the suboptimal mass. Finally, we synthesize these results to translate regularized algorithmic progress into bounds on the unregularized performance gap.

(i) Unregularized Performance Diference. By the standard Performance Diference Lemma applied to the unregularized objective $J _ { 0 }$

$$
J _ { 0 } ( \pi _ { 0 } ^ { * } ) - J _ { 0 } ( \pi _ { t } ) = \frac { 1 } { 1 - \gamma } \mathbb { E } _ { s \sim d ^ { \pi _ { t } } } \left[ \sum _ { a } \pi _ { t } ( a | s ) \big ( V _ { 0 } ^ { * } ( s ) - Q _ { 0 } ^ { * } ( s , a ) \big ) \right] .
$$

Because the rewards are bounded in $[ 0 , 1 ]$ , we have $\begin{array} { r } { V _ { 0 } ^ { * } ( s ) \leq \sum _ { t = 0 } ^ { \infty } \gamma ^ { t } \cdot 1 = \frac { 1 } { 1 - \gamma } } \end{array}$ and $Q _ { 0 } ^ { * } ( s , a ) \geq 0$ The maximum optimality gap $A _ { m a x } \triangleq \operatorname* { s u p } _ { s , a } ( V _ { 0 } ^ { * } ( s ) - Q _ { 0 } ^ { * } ( s , a ) )$ is bounded by $\frac { 1 } { 1 - \gamma }$ . Under the Minimal Action Gap (Assumption 5), the unregularized globally optimal policy $\pi _ { 0 } ^ { * }$ concentrates all probability mass on the unique optimal action $a ^ { * } ( s )$ , yielding $V _ { 0 } ^ { * } ( s ) - Q _ { 0 } ^ { * } ( s , a ^ { * } ( s ) ) = 0$ . For all suboptimal actions $a \neq a ^ { * } ( s )$ , we substitute the maximum gap $A _ { m a x }$ . Factoring this out bounds the unregularized performance gap by:

$$
J _ { 0 } ( \pi _ { 0 } ^ { * } ) - J _ { 0 } ( \pi _ { t } ) \leq \frac { A _ { m a x } } { 1 - \gamma } \mathbb { E } _ { d ^ { \pi _ { t } } } \left[ \sum _ { a \neq a ^ { * } ( s ) } \pi _ { t } ( a | s ) \right] = \frac { A _ { m a x } } { 1 - \gamma } \mathbb { E } _ { d ^ { \pi _ { t } } } [ q _ { t } ( s ) ] ,
$$

where $\begin{array} { r } { q _ { t } ( s ) \triangleq \sum _ { a \neq a ^ { * } ( s ) } \pi _ { t } ( a | s ) } \end{array}$ is the probability mass on suboptimal actions.

(ii) Reverse KL Identity and Bounds. By the Global Regularized Suboptimality Identity (Lemma 3.2), the global regularized performance gap is proportional to the state-averaged reverse KL divergence:

$$
\mathrm { G a p } _ { t } ^ { \dagger } = \frac { \tau } { 1 - \gamma } \mathbb { E } _ { d ^ { \pi _ { t } } } \bigl [ D _ { K L } ( \pi _ { t } ( \cdot | s ) | | \pi _ { \tau } ^ { * } ( \cdot | s ) ) \bigr ] .
$$

To link this divergence to the suboptimal mass $q _ { t } ( s )$ , we expand the pointwise reverse KL divergence into the cross-entropy and the negative policy entropy:

$$
D _ { K L } ( \pi _ { t } \| \pi _ { \tau } ^ { * } ) = \sum _ { a } \pi _ { t } ( a | s ) \log \frac { \pi _ { t } ( a | s ) } { \pi _ { \tau } ^ { * } ( a | s ) } = - \sum _ { a } \pi _ { t } ( a | s ) \log \pi _ { \tau } ^ { * } ( a | s ) - H ( \pi _ { t } ( \cdot | s ) ) .
$$

We independently lower-bound the cross-entropy and negative entropy terms using the suboptimal probability mass.

• Cross-Entropy: The regularized globally optimal policy $\pi _ { \tau } ^ { * } ( a | s ) \propto \exp ( Q _ { \tau } ^ { * } ( s , a ) / \tau )$ naturally assigns exponentially small probability to suboptimal actions. Under Assumption 5, the action gap satisfies $Q _ { 0 } ^ { * } ( s , a ^ { * } ( s ) ) - Q _ { 0 } ^ { * } ( s , a ) \geq \Delta$ for $a \neq a ^ { * } ( s )$ . By the Uniform Regularized Value Bound (Lemma 3.5), the regularized and unregularized Q-values are bounded by $\parallel Q _ { \tau } ^ { * } -$ $\begin{array} { r } { Q _ { 0 } ^ { * } \| _ { \infty } \le \frac { \tau \log | \mathcal { A } | } { 1 - \gamma } } \end{array}$ . Then the regularized globally optimal policy’s probability of a suboptimal

action a is bounded by:

$$
\pi _ { \tau } ^ { * } ( a | s ) \leq \frac { \pi _ { \tau } ^ { * } ( a | s ) } { \pi _ { \tau } ^ { * } ( a ^ { * } ( s ) | s ) } = \frac { \exp ( Q _ { \tau } ^ { * } ( s , a ) / \tau ) } { \exp ( Q _ { \tau } ^ { * } ( s , a ^ { * } ( s ) ) / \tau ) } \leq \exp \left( - \frac { \Delta } { \tau } + \frac { 2 \log | \cal A | } { 1 - \gamma } \right) .\tag{22}
$$

Taking the negative logarithm yields − log $\begin{array} { r } { \pi _ { \tau } ^ { * } ( a | s ) \geq \frac { \Delta } { \tau } - \frac { 2 \log | \mathcal { A } | } { 1 - \gamma } } \end{array}$ . The unregularized globally optimal action term is − log $\pi _ { \tau } ^ { * } ( a ^ { * } ( s ) | s ) \geq 0$ . Dropping it lower-bounds the cross-entropy:

$$
- \sum _ { a } \pi _ { t } ( a | s ) \log \pi _ { \tau } ^ { * } ( a | s ) \geq - \sum _ { a \neq a ^ { * } ( s ) } \pi _ { t } ( a | s ) \log \pi _ { \tau } ^ { * } ( a | s ) \geq q _ { t } ( s ) \left( { \frac { \Delta } { \tau } } - { \frac { 2 \log | A | } { 1 - \gamma } } \right) .
$$

• Negative Entropy: We invoke the Policy Entropy Decomposition Bound (Lemma 3.4), designating the unique optimal action $a ^ { * } ( s )$ under Assumption 5 as the single anchor:

$$
- H ( \pi _ { t } ( \cdot | s ) ) \geq q _ { t } ( s ) \log q _ { t } ( s ) + ( 1 - q _ { t } ( s ) ) \log ( 1 - q _ { t } ( s ) ) - q _ { t } ( s ) \log | \cal A | .
$$

Using the inequality $( 1 - x ) \log ( 1 - x ) \geq - x$ for $x \in [ 0 , 1 ]$ , we bound the binary entropy term:

$$
\begin{array} { r } { - H ( \pi _ { t } ( \cdot | s ) ) \geq q _ { t } ( s ) \log q _ { t } ( s ) - q _ { t } ( s ) - q _ { t } ( s ) \log | \cal A | . } \end{array}
$$

(iii) Inverting KL Bound for Suboptimal Mass. Summing the cross-entropy and negative entropy lower bounds yields the combined inequality:

$$
{ \cal D } _ { K L } ( \pi _ { t } | | \pi _ { \tau } ^ { * } ) \geq q _ { t } ( s ) \log q _ { t } ( s ) + q _ { t } ( s ) \left( \frac { \Delta } { \tau } - \left[ \frac { 2 \log | { \cal A } | } { 1 - \gamma } + \log | { \cal A } | + 1 \right] \right) .
$$

Substituting our definition log $\begin{array} { r } { C _ { \gamma } \triangleq \frac { 2 \log | \mathcal { A } | } { 1 - \gamma } + \log | \mathcal { A } | + 1 } \end{array}$ and letting $B \triangleq { \frac { \Delta } { \tau } } - \log C _ { \gamma } .$ , we obtain $D _ { K L } ( \pi _ { t } \| \pi _ { \tau } ^ { * } ) \geq q _ { t } ( s ) B + q _ { t } ( s ) \log q _ { t } ( s )$ . Then we bound $q _ { t } \triangleq q _ { t } ( s )$ by partitioning its domain [0, 1]:

• Case 1: $q _ { t } \ge e ^ { - B / 2 }$ . Then log $q _ { t } \ge - B / 2$ . Substituting this yields $D _ { K L } \geq q _ { t } B - q _ { t } ( B / 2 ) =$ $q _ { t } ( B / 2 )$ , which implies $\begin{array} { r } { q _ { t } \leq \frac { 2 } { B } D _ { K L } ( \pi _ { t } \| \pi _ { \tau } ^ { * } ) } \end{array}$

• Case 2: $q _ { t } < e ^ { - B / 2 }$ . The mass is trivially bounded by the exponential threshold $q _ { t } < e ^ { - B / 2 } =$ $\begin{array} { r } { e ^ { \frac { 1 } { 2 } \log C _ { \gamma } } \exp ( - \frac { \Delta } { 2 \tau } ) = \sqrt { C _ { \gamma } } \exp ( - \frac { \Delta } { 2 \tau } ) . } \end{array}$

Combining these cases yields $\begin{array} { r } { q _ { t } ( s ) \leq \frac { 2 } { \frac { \Delta } { \tau } - \log C _ { \gamma } } D _ { K L } ( \pi _ { t } \| \pi _ { \tau } ^ { * } ) + \sqrt { C _ { \gamma } } \exp \left( - \frac { \Delta } { 2 \tau } \right) } \end{array}$

(iv) Synthesis. From the threshold assumption $\begin{array} { r } { \tau \leq \frac { \Delta } { 2 \log C _ { \gamma } } } \end{array}$ , the denominator satisfies $\frac { \Delta } { \tau } ~ -$ log $\begin{array} { r } { C _ { \gamma } \geq \frac { \Delta } { 2 \tau } } \end{array}$ . Substituting this and taking the expectation over $d ^ { \pi _ { t } }$ yields:

$$
\mathbb { E } _ { d ^ { \pi _ { t } } } [ q _ { t } ( s ) ] \leq \frac { 4 \tau } { \Delta } \mathbb { E } _ { d ^ { \pi _ { t } } } \big [ D _ { K L } ( \pi _ { t } \| \pi _ { \tau } ^ { * } ) \big ] + \sqrt { C _ { \gamma } } \exp \left( - \frac { \Delta } { 2 \tau } \right) .
$$

Substituting this mass bound into the unregularized performance diference from Step (i) yields the first stated result. Finally, substituting the Global Regularized Suboptimality Identity $\begin{array} { r } { \big ( \frac { \tau } { 1 - \gamma } \mathbb { E } [ D _ { K L } ] = \mathrm { G a p } _ { t } ^ { \dagger } \big ) } \end{array}$ into this $J _ { 0 }$ bound completes the proof. □

Proof of Corollary 3.1 (Universal Unregularized Suboptimality Bound). We partition

the global temperature space into two regimes.

• Case 1 (Cold Regime): $\begin{array} { r } { \tau \leq \frac { \Delta } { 2 \log C _ { \gamma } } } \end{array}$ . The bounds hold by Theorem 3.1. Because $C _ { \gamma } \geq 1$ replacing the $\frac { A _ { m a x } \sqrt { C _ { \gamma } } } { 1 - \gamma }$ tail coeficient with the larger coeficient $C _ { t a i l } \triangleq { \frac { A _ { m a x } C _ { \gamma } } { 1 - \gamma } }$ preserves both inequalities.

• Case 2 (Hot Regime): $\begin{array} { r } { \tau > \frac { \Delta } { 2 \log C _ { \gamma } } } \end{array}$ . In this regime, the temperature is too high to guarantee the positive linear relationship when inverting the KL bound. However, as established in Step (i) of Theorem 3.1, $\begin{array} { r } { J _ { 0 } ( \pi _ { 0 } ^ { * } ) - J _ { 0 } ( \pi _ { t } ) \le \frac { A _ { m a x } } { 1 - \gamma } \mathbb { E } _ { d ^ { \pi _ { t } } } [ q _ { t } ( s ) ] } \end{array}$ . Since probability mass evaluates to at most 1, $\begin{array} { r } { J _ { 0 } ( \pi _ { 0 } ^ { * } ) - J _ { 0 } ( \pi _ { t } ) \le \frac { A _ { m a x } } { 1 - \gamma } } \end{array}$ trivially holds. We rearrange the regime condition $\begin{array} { r } { \tau > \frac { \Delta } { 2 \log C _ { \gamma } } } \end{array}$ to $\begin{array} { r } { 1 < C _ { \gamma } \exp \left( - \frac { \Delta } { 2 \tau } \right) } \end{array}$ , and substitute this into our trivial global bound:

$$
J _ { 0 } ( \pi _ { 0 } ^ { * } ) - J _ { 0 } ( \pi _ { t } ) \leq \frac { A _ { m a x } } { 1 - \gamma } \leq \frac { A _ { m a x } C _ { \gamma } } { 1 - \gamma } \exp \left( - \frac { \Delta } { 2 \tau } \right) = C _ { t a i l } \exp \left( - \frac { \Delta } { 2 \tau } \right) .
$$

Because the state-averaged KL divergence and the regularized performance gap are both non-negative, adding either the $\frac { 4 A _ { m a x } \tau } { \Delta ( 1 - \gamma ) } \mathbb { E } [ D _ { K L } ]$ term or the $\frac { 4 A _ { m a x } } { \Delta } \mathrm { G a p } _ { t } ^ { \dagger }$ term to the right side naturally maintains the respective inequality.

Combining the two cases completes the proof.

## E.3 Proofs for Section 4

Proof of Lemma 4.1 $( \chi ^ { 2 }$ to KL Divergence Bound). We use the analytic inequality that for any $x \in [ 0 , M ]$

$$
( x - 1 ) ^ { 2 } \leq 2 M ( x \log x - x + 1 ) .
$$

We substitute $\begin{array} { r } { x = { \frac { \mathrm { d } P } { \mathrm { d } Q } } } \end{array}$ and integrate with respect to $Q$ over the space $\mathcal { X } \mathrm { : }$

$$
\int _ { \mathcal { X } } \left( \frac { \mathrm { d } P } { \mathrm { d } Q } - 1 \right) ^ { 2 } \mathrm { d } Q \le 2 M \int _ { \mathcal { X } } \left( \frac { \mathrm { d } P } { \mathrm { d } Q } \log \frac { \mathrm { d } P } { \mathrm { d } Q } - \frac { \mathrm { d } P } { \mathrm { d } Q } + 1 \right) \mathrm { d } Q .
$$

By definition, the left-hand side is the $\chi ^ { 2 } ( P \| Q )$ divergence, whereas the right-hand side evaluates to $2 M D _ { K L } ( P \Vert Q )$ , completing the proof. □

Proof of Lemma 4.2 (Parameter-Space PL Condition). We separately bound the Ideal Algorithmic Progress and the Approximation Bias from the Ideal Critic Decomposition evaluated at $\pi = \pi _ { \omega _ { \tau } ^ { * } }$ in Lemma 3.3(i):

$$
\begin{array} { r } { ( 1 - \gamma ) \mathrm { G a p } _ { t } = - \tau D _ { t } + \underbrace { \mathbb { E } _ { d ^ { \omega _ { \tau } ^ { * } } } \left[ \langle ( \theta _ { \tau } ^ { * } ( \omega _ { t } ) - \tau \omega _ { t } ) ^ { \top } \phi , \pi _ { \omega _ { \tau } ^ { * } } - \pi _ { t } \rangle _ { \mathcal { A } } \right] } _ { \mathrm { I d e a l ~ A l g o r i t h m i c ~ P r o g r e s s } } + \underbrace { \mathbb { E } _ { d ^ { \omega _ { \tau } ^ { * } } } \left[ \langle \epsilon _ { t } , \pi _ { \omega _ { \tau } ^ { * } } - \pi _ { t } \rangle _ { \mathcal { A } } \right] } _ { \mathrm { A p p r o x i m a t i o n ~ B i a s } } . } \end{array}
$$

(i) Bounding Ideal Algorithmic Progress. For any state s, we apply H¨older’s inequality over the action space and Pinsker’s inequality $( \| \pi _ { \omega _ { \tau } ^ { * } } - \pi _ { t } \| _ { 1 } \leq \sqrt { 2 D _ { K L } ( \pi _ { \omega _ { \tau } ^ { * } } \| \pi _ { t } ) } )$ to the inner product:

$$
\begin{array} { r l } & { \langle ( \theta _ { \tau } ^ { * } ( \omega _ { t } ) - \tau \omega _ { t } ) ^ { \top } \phi , \pi _ { \omega _ { \tau } ^ { * } } - \pi _ { t } \rangle _ { A } \leq \operatorname* { m a x } _ { a } \left| ( \theta _ { \tau } ^ { * } ( \omega _ { t } ) - \tau \omega _ { t } ) ^ { \top } \phi ( s , a ) \right| \| \pi _ { \omega _ { \tau } ^ { * } } - \pi _ { t } \| _ { 1 } } \\ & { \qquad \leq B _ { \phi } \| \theta _ { \tau } ^ { * } ( \omega _ { t } ) - \tau \omega _ { t } \| _ { 2 } \sqrt { 2 D _ { K L } ( \pi _ { \omega _ { \tau } ^ { * } } \| \pi _ { t } ) } . } \end{array}
$$

Taking the expectation over $d ^ { \omega _ { \tau } ^ { * } } ( s )$ and applying Jensen’s Inequality, $\begin{array} { r l } { \mathbb { E } _ { d ^ { \omega _ { \tau } ^ { * } } } [ \sqrt { D _ { K L } } ] } & { { } \leq } \end{array}$ $\sqrt { \mathbb { E } _ { d ^ { \omega _ { \tau } ^ { * } } } [ D _ { K L } ] } = \sqrt { D _ { t } } .$ , yields:

$$
\begin{array} { r } { \mathbb { E } _ { d ^ { \omega _ { \tau } ^ { * } } } \Big [ \langle ( \theta _ { \tau } ^ { * } ( \omega _ { t } ) - \tau \omega _ { t } ) ^ { \top } \phi , \pi _ { \omega _ { \tau } ^ { * } } - \pi _ { t } \rangle _ { A } \Big ] \leq B _ { \phi } \| \theta _ { \tau } ^ { * } ( \omega _ { t } ) - \tau \omega _ { t } \| _ { 2 } \sqrt { 2 D _ { t } } . } \end{array}
$$

We decouple the resulting product using Young’s Inequality $\begin{array} { r } { ( x y \le \frac { \tau } { 2 } x ^ { 2 } + \frac { 1 } { 2 \tau } y ^ { 2 } ) } \end{array}$ with $x = \sqrt { D _ { t } }$ and $y = \sqrt { 2 } B _ { \phi } \Vert \theta _ { \tau } ^ { * } ( \omega _ { t } ) - \tau \omega _ { t } \Vert _ { 2 } \mathrm { { : } }$

$$
\mathbb { E } _ { d ^ { \omega _ { \tau } ^ { * } } } \Big [ \langle ( \theta _ { \tau } ^ { * } ( \omega _ { t } ) - \tau \omega _ { t } ) ^ { \top } \phi , \pi _ { \omega _ { \tau } ^ { * } } - \pi _ { t } \rangle _ { A } \Big ] \leq \frac { \tau } { 2 } D _ { t } + \frac { B _ { \phi } ^ { 2 } } { \tau } \| \theta _ { \tau } ^ { * } ( \omega _ { t } ) - \tau \omega _ { t } \| _ { 2 } ^ { 2 } .
$$

(ii) Bounding Approximation Bias. For any state s, we define the local policy mismatch $W ( s ) \triangleq$ $\begin{array} { r } { \operatorname* { m a x } _ { a } \frac { \pi _ { \omega _ { \tau } ^ { * } } ( a | s ) } { \pi _ { t } ( a | s ) } } \end{array}$ . We apply Action-Space Cauchy-Schwarz inequality:

$$
\begin{array} { r l r } {  { \langle \epsilon _ { t } , \pi _ { \omega _ { \tau } ^ { * } } - \pi _ { t } \rangle _ { \ r { \mathcal A } } = \sum _ { a } \pi _ { t } ( a | s ) \epsilon _ { t } ( s , a ) ( \frac { \pi _ { \omega _ { \tau } ^ { * } } ( a | s ) } { \pi _ { t } ( a | s ) } - 1 ) } } \\ & { } & { \leq \sqrt { \mathbb { E } _ { a \sim \pi _ { t } } [ \epsilon _ { t } ( s , a ) ^ { 2 } ] } \sqrt { \sum _ { a } \pi _ { t } ( a | s ) ( \frac { \pi _ { \omega _ { \tau } ^ { * } } ( a | s ) } { \pi _ { t } ( a | s ) } - 1 ) ^ { 2 } } . } \end{array}
$$

The second square root yields the $\chi ^ { 2 } \cdot$ -divergence. By applying the $\chi ^ { 2 }$ bound (Lemma 4.1) $\chi ^ { 2 } ( \pi _ { \omega _ { \tau } ^ { * } } ( \cdot | s ) | | \pi _ { t } ( \cdot | s ) ) \leq 2 W ( s ) D _ { K L } ( \pi _ { \omega _ { \tau } ^ { * } } ( \cdot | s ) | | \pi _ { t } ( \cdot | s ) )$ pointwise at s, we obtain:

$$
\begin{array} { r } { \langle \epsilon _ { t } , \pi _ { \omega _ { \tau } ^ { * } } - \pi _ { t } \rangle _ { \mathcal { A } } \leq \sqrt { \mathbb { E } _ { a \sim \pi _ { t } } [ \epsilon _ { t } ( s , a ) ^ { 2 } ] } \sqrt { 2 W ( s ) D _ { K L } ( \pi _ { \omega _ { \tau } ^ { * } } ( \cdot | s ) \| \pi _ { t } ( \cdot | s ) ) } . } \end{array}
$$

Taking the expectation over $d ^ { \omega _ { \tau } ^ { * } } ( s )$ and applying State-Space Cauchy-Schwarz:

$$
\begin{array} { r l } & { \mathbb { E } _ { d ^ { \omega _ { \tau } ^ { * } } } \left[ \left. \epsilon _ { t } , \pi _ { \omega _ { \tau } ^ { * } } - \pi _ { t } \right. _ { A } \right] \leq \mathbb { E } _ { d ^ { \omega _ { \tau } ^ { * } } } \left[ \sqrt { 2 D _ { K L } ( \pi _ { \omega _ { \tau } ^ { * } } ( \cdot | s ) \| \pi _ { t } ( \cdot | s ) ) } \sqrt { W ( s ) \mathbb { E } _ { a \sim \pi _ { t } } \left[ \epsilon _ { t } ( s , a ) ^ { 2 } \right] } \right] } \\ & { \qquad \leq \sqrt { 2 D _ { t } } \sqrt { \mathbb { E } _ { d ^ { \omega _ { \tau } ^ { * } } } \left[ W ( s ) \mathbb { E } _ { \pi _ { t } } \left[ \epsilon _ { t } ^ { 2 } \right] \right] } . } \end{array}
$$

To control the second root, we expand the expectation over the state space and apply the joint concentrability bound $\begin{array} { r } { \frac { d ^ { \omega _ { \tau } ^ { * } } ( s ) } { d ^ { \pi _ { t } } ( s ) } W ( s ) = \operatorname* { m a x } _ { a } \frac { d ^ { \omega _ { \tau } ^ { * } } ( s ) \pi _ { \omega _ { \tau } ^ { * } } ( a \mid s ) } { d ^ { \pi _ { t } } ( s ) \pi _ { t } ( a \mid s ) } \leq C _ { j o i n t } } \end{array}$ (Assumption 7):

$$
\begin{array} { r l } & { \mathbb { E } _ { d ^ { \omega _ { \tau } ^ { * } } } \left[ W ( s ) \mathbb { E } _ { \pi _ { t } } [ \epsilon _ { t } ^ { 2 } ] \right] = \mathbb { E } _ { d ^ { \pi _ { t } } } \left[ \left( \frac { d ^ { \omega _ { \tau } ^ { * } } ( s ) } { d ^ { \pi _ { t } } ( s ) } W ( s ) \right) \mathbb { E } _ { a \sim \pi _ { t } } [ \epsilon _ { t } ( s , a ) ^ { 2 } ] \right] } \\ & { \qquad \leq C _ { j o i n t } \mathbb { E } _ { d ^ { \pi _ { t } } } \left[ \mathbb { E } _ { a \sim \pi _ { t } } [ \epsilon _ { t } ( s , a ) ^ { 2 } ] \right] } \\ & { \qquad = C _ { j o i n t } \mathbb { E } _ { d ^ { \pi _ { t } } , \pi _ { t } } [ \epsilon _ { t } ^ { 2 } ] \leq C _ { j o i n t } \epsilon _ { a p p } , } \end{array}
$$

where the last step uses Assumption 4. Applying Young’s Inequality $\begin{array} { r } { ( x y \le \frac { \tau } { 2 } x ^ { 2 } + \frac { 1 } { 2 \tau } y ^ { 2 } ) } \end{array}$ with

$x = \sqrt { D _ { t } }$ and $y = \sqrt { 2 C _ { j o i n t } \epsilon _ { a p p } }$ yields:

$$
\mathbb { E } _ { d ^ { \omega _ { \tau } ^ { * } } } \big [ \langle \epsilon _ { t } , \pi _ { \omega _ { \tau } ^ { * } } - \pi _ { t } \rangle _ { A } \big ] \leq \frac { \tau } { 2 } D _ { t } + \frac { C _ { j o i n t } } { \tau } \epsilon _ { a p p } .
$$

(iii) Synthesis. We substitute the decoupled ideal algorithmic progress and approximation bias bounds back into the Ideal Critic Decomposition (Lemma 3.3) for $\pi = \pi _ { \omega _ { \tau } ^ { * } }$

$$
\begin{array} { r l } & { ( 1 - \gamma ) { \mathrm { G a p } } _ { t } \leq - \tau D _ { t } + \left( \frac { \tau } { 2 } D _ { t } + \frac { B _ { \phi } ^ { 2 } } { \tau } \| \theta _ { \tau } ^ { * } ( \omega _ { t } ) - \tau \omega _ { t } \| _ { 2 } ^ { 2 } \right) + \left( \frac { \tau } { 2 } D _ { t } + \frac { C _ { j o i n t } } { \tau } \epsilon _ { a p p } \right) } \\ & { \qquad = \frac { B _ { \phi } ^ { 2 } } { \tau } \| \theta _ { \tau } ^ { * } ( \omega _ { t } ) - \tau \omega _ { t } \| _ { 2 } ^ { 2 } + \frac { C _ { j o i n t } } { \tau } \epsilon _ { a p p } . } \end{array}
$$

Remarkably, the positive $\tau D _ { t }$ penalties from Young’s inequality perfectly cancel the $- \tau D _ { t }$ restorative entropy force. Dividing both sides by $1 - \gamma$ yields the stated PL condition. □

## E.4 Proofs for Section 5

Proof of Lemma 5.1 (Uncentered Gradient Identity). (i) Gradient Expansion. By the Policy Gradient Theorem for entropy-regularized MDPs, the exact gradient is:

$$
\nabla J _ { \tau } ( \omega _ { t } ) = \frac { 1 } { 1 - \gamma } \mathbb { E } _ { d ^ { \pi _ { t } } , \pi _ { t } } \Big [ \tilde { \phi } _ { t } ( s , a ) \big ( Q _ { \tau } ^ { \pi _ { t } } ( s , a ) - \tau \log \pi _ { t } ( a | s ) \big ) \Big ] .
$$

We substitute the ideal uncentered critic $Q _ { \tau } ^ { \pi _ { t } } = \theta _ { \tau } ^ { * } ( \omega _ { t } ) ^ { \top } \phi + \epsilon _ { t }$ and the log-linear policy identity τ log $\pi _ { t } = \tau \omega _ { t } ^ { \top } \phi - \tau$ log $Z _ { t }$ . Because τ log $Z _ { t } ( s )$ depends only on the state, it vanishes when multiplied by the action-centered features $\tilde { \phi } _ { t } ( s , a )$ and taking the action expectation:

$$
\nabla J _ { \tau } ( \omega _ { t } ) = \frac { 1 } { 1 - \gamma } \mathbb { E } _ { d ^ { \pi _ { t } } , \pi _ { t } } \Big [ \tilde { \phi } _ { t } ( s , a ) \big ( ( \theta _ { \tau } ^ { * } ( \omega _ { t } ) - \tau \omega _ { t } ) ^ { \top } \phi ( s , a ) + \epsilon _ { t } ( s , a ) \big ) \Big ] .
$$

(ii) Feature Decomposition. We separate the uncentered features via $\phi ( s , a ) = \tilde { \phi } _ { t } ( s , a ) + \bar { \phi } _ { t } ( s )$ For the first term in the expectation:

$$
\begin{array} { r } { \mathbb { E } _ { d ^ { n t } , \pi _ { t } } \Big [ \tilde { \phi } _ { t } \phi ^ { \top } \Big ] \big ( \theta _ { \tau } ^ { * } ( \omega _ { t } ) - \tau \omega _ { t } \big ) = \mathbb { E } _ { d ^ { n t } , \pi _ { t } } \Big [ \tilde { \phi } _ { t } \big ( \tilde { \phi } _ { t } + \tilde { \phi } _ { t } \big ) ^ { \top } \Big ] \big ( \theta _ { \tau } ^ { * } ( \omega _ { t } ) - \tau \omega _ { t } \big ) = \bar { \Sigma } _ { c e n } ( \pi _ { t } ) \big ( \theta _ { \tau } ^ { * } ( \omega _ { t } ) - \tau \omega _ { t } \big ) , } \end{array}
$$

where the cross-term $\mathbb { E } _ { d ^ { \pi _ { t } } , \pi _ { t } } [ \tilde { \phi } _ { t } \bar { \phi } _ { t } ^ { \top } ]$ is zero because $\mathbb { E } _ { a \sim \pi _ { t } } [ \tilde { \phi } _ { t } ] = 0$

(iii) Bias Extraction. For the second term, we again substitute $\tilde { \phi } _ { t } = \phi - \bar { \phi } _ { t }$

$$
\mathbb { E } _ { d ^ { \pi _ { t } } , \pi _ { t } } \big [ \tilde { \phi } _ { t } \epsilon _ { t } \big ] = \mathbb { E } _ { d ^ { \pi _ { t } } , \pi _ { t } } \big [ \phi \epsilon _ { t } \big ] - \mathbb { E } _ { d ^ { \pi _ { t } } , \pi _ { t } } \big [ \bar { \phi } _ { t } \epsilon _ { t } \big ] .
$$

By the uncentered orthogonality condition (3), the first term is exactly zero. The remaining term evaluates to $E _ { b i a s }$ . Applying Cauchy-Schwarz and Jensen’s inequality to the bias definition yields $\| E _ { b i a s } \| _ { 2 } \le \mathbb { E } [ \| \bar { \phi } _ { t } \| _ { 2 } | \epsilon _ { t } | ] \le B _ { \phi } \sqrt { \mathbb { E } [ \epsilon _ { t } ^ { 2 } ] } \le B _ { \phi } \sqrt { \epsilon _ { a p p } } ,$ , completing the proof. □

Proof of Lemma 5.2 (Smoothness of the Regularized Objective). We establish the smoothness bound via recursive diferentiation of the value function.

(i) Local Policy Derivatives. For the log-linear softmax policy, the gradient of the log-probability is the centered feature: ∇ log $\pi _ { \omega } ( a | s ) = \phi ( s , a ) - \mathbb { E } _ { \pi _ { \omega } } [ \phi ]$ . Because $\| \phi \| _ { 2 } \le B _ { \phi }$ , we have $\| \nabla \log \pi _ { \omega } \| _ { 2 } \leq$ $2 B _ { \phi }$ . Consequently, the gradient of the policy itself is bounded by:

$$
\sum _ { a } \| \nabla \pi _ { \omega } ( a | s ) \| _ { 2 } = \sum _ { a } \pi _ { \omega } ( a | s ) \| \nabla \log \pi _ { \omega } ( a | s ) \| _ { 2 } \leq 2 B _ { \phi } .
$$

For the Hessian, applying the product rule to $\nabla \pi _ { \omega } = \pi _ { \omega } \nabla$ log $\pi _ { \omega }$ yields $\nabla ^ { 2 } \pi _ { \omega } = \pi _ { \omega } \nabla ^ { 2 } \log \pi _ { \omega } +$ $\begin{array} { r } { \frac { 1 } { \pi _ { \omega } } \nabla \pi _ { \omega } \nabla \pi _ { \omega } ^ { \top } } \end{array}$ . The Hessian of the log-policy is the negative feature covariance: $\nabla ^ { 2 } \log \pi _ { \omega } ( a | s ) =$ $- \mathrm { C o v } _ { a ^ { \prime } \sim \pi _ { \omega } } ( \phi ( s , a ^ { \prime } ) )$ . Since $\| \phi \| _ { 2 } \le B _ { \phi }$ , the spectral norm of this covariance matrix is bounded by $B _ { \phi } ^ { 2 }$ . Thus, the local policy Hessian is bounded by:

$$
\sum _ { a } \| \nabla ^ { 2 } \pi _ { \omega } ( a | s ) \| _ { 2 } \leq \sum _ { a } \pi _ { \omega } ( a | s ) \big ( B _ { \phi } ^ { 2 } + ( 2 B _ { \phi } ) ^ { 2 } \big ) = 5 B _ { \phi } ^ { 2 } .
$$

Finally, by Lemma 2.1, the regularized action-value is globally bounded in magnitude by $\begin{array} { r } { | Q _ { \tau } ^ { \pi _ { \omega } } ( s , a ) | \leq \frac { 1 + \tau \log | { \cal A } | } { 1 - \gamma } \triangleq \frac { r _ { m a x } } { 1 - \gamma } } \end{array}$

(ii) Recursive Value Gradient. By definition, $\begin{array} { r } { V _ { \tau } ^ { \pi _ { \omega } } ( s ) = \sum _ { a } \pi _ { \omega } ( a \vert s ) \big ( Q _ { \tau } ^ { \pi _ { \omega } } ( s , a ) - \tau \log \pi _ { \omega } ( a \vert s ) \big ) } \end{array}$ By the product rule, its gradient satisfies the recursive identity:

$$
\begin{array} { l } { { \nabla V _ { \tau } ^ { \pi _ { \omega } } ( s ) = \displaystyle \sum _ { a } \nabla \pi _ { \omega } ( a | s ) \big ( Q _ { \tau } ^ { \pi _ { \omega } } ( s , a ) - \tau \log \pi _ { \omega } ( a | s ) \big ) + \displaystyle \sum _ { a } \pi _ { \omega } ( a | s ) \big ( \nabla Q _ { \tau } ^ { \pi _ { \omega } } ( s , a ) - \tau \nabla \log \pi _ { \omega } ( a | s ) \big ) } } \\ { { \qquad = \displaystyle \sum _ { a } \nabla \pi _ { \omega } ( a | s ) \big ( Q _ { \tau } ^ { \pi _ { \omega } } ( s , a ) - \tau \log \pi _ { \omega } ( a | s ) \big ) + \gamma \displaystyle \sum _ { a } \pi _ { \omega } ( a | s ) \mathbb { E } _ { s ^ { \prime } \sim \mathcal { P } ( \cdot | s , a ) } [ \nabla V _ { \tau } ^ { \pi _ { \omega } } ( s ^ { \prime } ) ] , } } \end{array}
$$

where $\begin{array} { r } { \sum _ { a } \pi _ { \omega } ( a | s ) \nabla \log \pi _ { \omega } ( a | s ) = \sum _ { a } \nabla \pi _ { \omega } ( a | s ) = 0 } \end{array}$ . Taking the vector norm and substituting the identity $\nabla \pi _ { \omega } = \pi _ { \omega } \nabla$ log $\pi _ { \omega }$ :

$$
\| \nabla V _ { \tau } ^ { \pi _ { \omega } } ( s ) \| _ { 2 } \leq \sum _ { a } \pi _ { \omega } ( a | s ) \| \nabla \log \pi _ { \omega } ( a | s ) \| _ { 2 } \big | Q _ { \tau } ^ { \pi _ { \omega } } ( s , a ) - \tau \log \pi _ { \omega } ( a | s ) \big | + \gamma \operatorname* { s u p } _ { s ^ { \prime } } \| \nabla V _ { \tau } ^ { \pi _ { \omega } } ( s ^ { \prime } ) \| _ { 2 } .
$$

From the score bound, we have $\| \nabla \log \pi _ { \omega } ( a | s ) \| _ { 2 } \leq 2 B _ { \phi }$ . To bound the absolute value term, we apply the triangle inequality. First, $\begin{array} { r } { | Q _ { \tau } ^ { \pi _ { \omega } } ( s , a ) | \le \frac { r _ { m a x } } { 1 - \gamma } } \end{array}$ . Second, the expected absolute entropy evaluates to the Shannon entropy: $\begin{array} { r } { \sum _ { a } \pi _ { \omega } ( a | s ) | \tau \log \pi _ { \omega } ( a | s ) | = \tau H ( \pi _ { \omega } ( \cdot | s ) ) \leq \tau \log | A | \leq \frac { r _ { m a x } } { 1 - \gamma } } \end{array}$ Thus, the entire first term is bounded by $2 B _ { \phi } \big ( \frac { 2 r _ { m a x } } { 1 - \gamma } \big )$ , yielding:

$$
\| \nabla V _ { \tau } ^ { \pi _ { \omega } } ( s ) \| _ { 2 } \leq \frac { 4 B _ { \phi } r _ { m a x } } { 1 - \gamma } + \gamma \operatorname* { s u p } _ { s ^ { \prime } } \| \nabla V _ { \tau } ^ { \pi _ { \omega } } ( s ^ { \prime } ) \| _ { 2 } .
$$

Solving this recurrence relation across the state space geometrically bounds the state-value gradient by $\begin{array} { r } { \| \nabla V _ { \tau } ^ { \pi _ { \omega } } \| _ { 2 } \leq \frac { 4 B _ { \phi } r _ { m a x } } { ( 1 - \gamma ) ^ { 2 } } } \end{array}$ . Because the transition dynamics are independent of $\omega ,$ the Q-function gradient is $\nabla Q _ { \tau } ^ { \pi _ { \omega } } ( s , a ) = \gamma \mathbb { E } _ { s ^ { \prime } \sim \mathcal { P } ( \cdot \vert s , a ) } [ \nabla V _ { \tau } ^ { \pi _ { \omega } } ( s ^ { \prime } ) ]$ . Taking the norm yields $\begin{array} { r } { \| \nabla Q _ { \tau } ^ { \pi _ { \omega } } \| _ { 2 } \leq \frac { 4 \gamma B _ { \phi } r _ { m a x } } { ( 1 - \gamma ) ^ { 2 } } } \end{array}$

(iii) Recursive Value Hessian. By the multivariate product rule, the Hessian gains one additional term from the derivative of $\nabla \pi _ { \omega }$

$$
\begin{array} { r l } & { \nabla ^ { 2 } V _ { \tau } ^ { \pi _ { \omega } } ( s ) = \displaystyle \sum _ { a } \nabla ^ { 2 } \pi _ { \omega } ( a | s ) \left( Q _ { \tau } ^ { \pi _ { \omega } } ( s , a ) - \tau \log \pi _ { \omega } ( a | s ) \right) } \\ & { \quad \quad \quad + \displaystyle \sum _ { a } \nabla \pi _ { \omega } ( a | s ) \nabla Q _ { \tau } ^ { \pi _ { \omega } } ( s , a ) ^ { \top } + \displaystyle \sum _ { a } \nabla Q _ { \tau } ^ { \pi _ { \omega } } ( s , a ) \nabla \pi _ { \omega } ( a | s ) ^ { \top } } \\ & { \quad \quad \quad - \tau \displaystyle \sum _ { a } \pi _ { \omega } ( a | s ) \nabla \log \pi _ { \omega } ( a | s ) \nabla \log \pi _ { \omega } ( a | s ) ^ { \top } + \gamma \displaystyle \sum _ { a } \pi _ { \omega } ( a | s ) \mathbb { E } _ { s ^ { \prime } \sim \mathcal { P } ( \cdot | s , a ) } [ \nabla ^ { 2 } V _ { \tau } ^ { \pi _ { \omega } } ( s ^ { \prime } ) ] . } \end{array}
$$

We take the spectral norm of both sides via the triangle inequality. The second and third are transposes of each other, meaning their spectral norms are identical and contribute equally. The fourth term is the negative feature covariance scaled by $\tau ,$ bounded by $\tau \| \mathrm { C o v } _ { \pi _ { \omega } } ( \phi ) \| _ { 2 } \leq 4 \tau B _ { \phi } ^ { 2 }$ Substituting the uniform bounds from Steps (i) and (ii) into this recurrence yields:

$$
\| \nabla ^ { 2 } V _ { \tau } ^ { \pi _ { \omega } } ( s ) \| _ { 2 } \leq 5 B _ { \phi } ^ { 2 } \left( \frac { 2 r _ { m a x } } { 1 - \gamma } \right) + 2 ( 2 B _ { \phi } ) \frac { 4 \gamma B _ { \phi } r _ { m a x } } { ( 1 - \gamma ) ^ { 2 } } + 4 \tau B _ { \phi } ^ { 2 } + \gamma \operatorname* { s u p } _ { s ^ { \prime } } \| \nabla ^ { 2 } V _ { \tau } ^ { \pi _ { \omega } } ( s ^ { \prime } ) \| _ { 2 } .
$$

Solving this recurrence relation gives:

$$
\| \nabla ^ { 2 } V _ { \tau } ^ { \pi _ { \omega } } \| _ { 2 } \leq \frac { 1 0 B _ { \phi } ^ { 2 } r _ { m a x } } { ( 1 - \gamma ) ^ { 2 } } + \frac { 1 6 \gamma B _ { \phi } ^ { 2 } r _ { m a x } } { ( 1 - \gamma ) ^ { 3 } } + \frac { 4 \tau B _ { \phi } ^ { 2 } } { 1 - \gamma } \leq \frac { 1 0 + 6 \gamma } { ( 1 - \gamma ) ^ { 3 } } B _ { \phi } ^ { 2 } r _ { m a x } + \frac { 4 \tau B _ { \phi } ^ { 2 } } { ( 1 - \gamma ) ^ { 3 } } .
$$

Since $\gamma < 1$ , we have $1 0 + 6 \gamma \leq 1 6$ , yielding the final uniform Hessian bound:

$$
\| \nabla ^ { 2 } V _ { \tau } ^ { \pi _ { \omega } } \| _ { 2 } \leq \frac { 1 6 B _ { \phi } ^ { 2 } r _ { m a x } + 4 \tau B _ { \phi } ^ { 2 } } { ( 1 - \gamma ) ^ { 3 } } .
$$

(iv) Objective Smoothness. The algorithm objective is the expected value under the initial state distribution $\mu ,$ so $J _ { \tau } ( \omega ) = \mathbb { E } _ { s _ { 0 } \sim \mu } [ V _ { \tau } ^ { \pi _ { \omega } } ( s _ { 0 } ) ]$ . By Jensen’s inequality, the spectral norm of the objective’s Hessian is bounded by the maximum spectral norm of the value Hessian over the state space. Therefore, $\Vert \nabla ^ { 2 } J _ { \tau } ( \omega ) \Vert _ { 2 } \leq L _ { J }$ , guaranteeing the objective is globally $L _ { J } { \mathrm { - s m o o t h } }$ □

Proof of Lemma 5.3 (Actor Progress Bound). (i) Smoothness Expansion. By the global smoothness of the regularized objective (Lemma 5.2), the step-wise progress is governed by:

$$
J _ { \tau } ( \omega _ { t + 1 } ) - J _ { \tau } ( \omega _ { t } ) \geq \eta _ { t } \langle \nabla J _ { \tau } ( \omega _ { t } ) , \Delta \omega _ { t } \rangle - \frac { L _ { J } \eta _ { t } ^ { 2 } } { 2 } \| \Delta \omega _ { t } \| _ { 2 } ^ { 2 } ,
$$

where $\Delta \omega _ { t } = \theta _ { t + 1 } - \tau \omega _ { t }$ such that $\omega _ { t + 1 } - \omega _ { t } = \eta _ { t } \Delta \omega _ { t }$ . We substitute the Uncentered Gradient Identity (Lemma 5.1) into the inner product:

$$
\langle \nabla J _ { \tau } , \Delta \omega _ { t } \rangle = \frac { 1 } { 1 - \gamma } \langle \bar { \Sigma } _ { c e n } ( \pi _ { t } ) ( \theta _ { \tau } ^ { * } ( \omega _ { t } ) - \tau \omega _ { t } ) , \Delta \omega _ { t } \rangle + \frac { 1 } { 1 - \gamma } \langle E _ { b i a s } , \Delta \omega _ { t } \rangle .
$$

(ii) Decoupling Centered Matrix Term. For any positive semi-definite symmetric matrix $\Sigma .$ $\begin{array} { r } { a ^ { \top } \Sigma b = \frac 1 2 \| b \| _ { \Sigma } ^ { 2 } + \frac 1 2 \| a \| _ { \Sigma } ^ { 2 } - \frac 1 2 \| b - a \| _ { \Sigma } ^ { 2 } \geq \frac 1 2 \| b \| _ { \Sigma } ^ { 2 } - \frac 1 2 \| b - a \| _ { \Sigma } ^ { 2 } } \end{array}$ . Setting $a = \theta _ { \tau } ^ { * } ( \omega _ { t } ) - \tau \omega _ { t }$ and $b = \Delta \omega _ { t }$

we have $b - a = \theta _ { t + 1 } - \theta _ { \tau } ^ { * } ( \omega _ { t } ) = e _ { t }$ . Applying this yields:

$$
\frac { 1 } { 1 - \gamma } \langle \bar { \Sigma } _ { c e n } ( \pi _ { t } ) ( \theta _ { \tau } ^ { * } ( \omega _ { t } ) - \tau \omega _ { t } ) , \Delta \omega _ { t } \rangle \geq \underbrace { \frac { 1 } { 2 ( 1 - \gamma ) } \| \Delta \omega _ { t } \| _ { \bar { \Sigma } _ { c e n } ( \pi _ { t } ) } ^ { 2 } } _ { \mathrm { A l g o r i t h m i c ~ P r o g r e s s } } - \underbrace { \frac { 1 } { 2 ( 1 - \gamma ) } \| e _ { t } \| _ { \bar { \Sigma } _ { c e n } ( \pi _ { t } ) } ^ { 2 } } _ { \mathrm { T r a c k i n g ~ P e n a l t y } } .
$$

(iii) Lower-Bounding Algorithmic Progress and Upper-Bounding Tracking Penalty. To bound the eigenvalues of ${ \bar { \Sigma } } _ { c e n } ( \pi _ { t } )$ for the algorithmic progress, we shift the measure from $d ^ { \pi _ { t } } \times \pi _ { t }$ to $d ^ { \omega _ { \tau } ^ { * } } \times \pi _ { \omega _ { \tau } ^ { * } }$ via the joint concentrability bound $C _ { j o i n t }$ (Assumption 7). For any vector v, we have:

$$
v ^ { \top } \bar { \Sigma } _ { c e n } ( \pi _ { t } ) v = \mathbb { E } _ { d ^ { \pi _ { t } } , \pi _ { t } } \big [ ( v ^ { \top } \tilde { \phi } _ { t } ) ^ { 2 } \big ] \geq \frac { 1 } { C _ { j o i n t } } \mathbb { E } _ { d ^ { \omega _ { \tau } ^ { * } } , \pi _ { \omega _ { \tau } ^ { * } } } \big [ ( v ^ { \top } \tilde { \phi } _ { t } ) ^ { 2 } \big ] .
$$

Note that $\tilde { \phi } _ { t } ( s , a ) = \phi ( s , a ) - \bar { \phi } _ { t } ( s )$ is centered around the training mean $\bar { \phi } _ { t } ( s )$ , distinct from the feature mean under the parametric optimum $\bar { \phi } ^ { \omega _ { \tau } ^ { \ast } } ( s ) \triangleq \mathbb { E } _ { a \sim \pi _ { \omega _ { \tau } ^ { \ast } } ( \cdot | s ) } [ \phi ( s , a ) ]$ . For any state s, because the expected squared distance of a random variable is minimized when centered around its true mean, this mismatched centering yields a lower bound over the action expectation:

$$
\begin{array} { r l } & { \mathbb { E } _ { a \sim \pi _ { \omega _ { \tau } ^ { * } } } \Big [ \big ( v ^ { \top } ( \phi ( s , a ) - \bar { \phi } _ { t } ( s ) ) \big ) ^ { 2 } \Big | s \Big ] \geq \mathbb { E } _ { a \sim \pi _ { \omega _ { \tau } ^ { * } } } \Big [ \big ( v ^ { \top } ( \phi ( s , a ) - \bar { \phi } ^ { \omega _ { \tau } ^ { * } } ( s ) ) \big ) ^ { 2 } \Big | s \Big ] . } \end{array}
$$

Taking the outer expectation over the state distribution $s \sim d ^ { \omega _ { \tau } ^ { * } }$ preserves this pointwise inequal ity. Therefore, $\begin{array} { r } { \bar { \Sigma } _ { c e n } ( \pi _ { t } ) ~ \succeq ~ \frac { 1 } { C _ { j o i n t } } \bar { \Sigma } _ { c e n } ( \pi _ { \omega _ { \tau } ^ { * } } ) } \end{array}$ . Applying the Positive-Definite Fisher Information (Assumption 6), $\begin{array} { r } { \bar { \Sigma } _ { c e n } ( \pi _ { \omega _ { \tau } ^ { * } } ) \succeq \lambda I . } \end{array}$ , yields the lower bound $\begin{array} { r } { \bar { \Sigma } _ { c e n } ( \pi _ { t } ) \succeq \frac { \lambda } { C _ { j o i n t } } I . } \end{array}$

For the tracking penalty term, by the bounded feature assumption (Assumption 1), the centered covariance is directly upper-bounded: $\bar { \Sigma } _ { c e n } ( \pi _ { t } ) \preceq 4 B _ { \phi } ^ { 2 } I .$ . Substituting these respective spectrum limits into the decoupling inequality yields:

$$
\frac { 1 } { 1 - \gamma } \langle \bar { \Sigma } _ { c e n } ( \pi _ { t } ) ( \theta _ { \tau } ^ { * } ( \omega _ { t } ) - \tau \omega _ { t } ) , \Delta \omega _ { t } \rangle \geq \frac { \lambda } { 2 C _ { j o i n t } ( 1 - \gamma ) } \| \Delta \omega _ { t } \| _ { 2 } ^ { 2 } - \frac { 2 B _ { \phi } ^ { 2 } } { 1 - \gamma } \| e _ { t } \| _ { 2 } ^ { 2 } .
$$

(iv) Decoupling Uncentered Bias Term. We decouple the uncentered bias inner product using Young’s inequality $\begin{array} { r } { ( x y \le \frac { c } { 2 } x ^ { 2 } + \frac { 1 } { 2 c } y ^ { 2 } ) } \end{array}$ with carefully chosen $\begin{array} { r } { c = \frac { \lambda } { 4 C _ { j o i n t } } } \end{array}$ :

$$
\left| \frac { 1 } { 1 - \gamma } \langle E _ { b i a s } , \Delta \omega _ { t } \rangle \right| \leq \frac { 1 } { 1 - \gamma } \| E _ { b i a s } \| _ { 2 } \| \Delta \omega _ { t } \| _ { 2 } \leq \frac { \lambda } { 8 C _ { j o i n t } ( 1 - \gamma ) } \| \Delta \omega _ { t } \| _ { 2 } ^ { 2 } + \frac { 2 C _ { j o i n t } } { \lambda ( 1 - \gamma ) } \| E _ { b i a s } \| _ { 2 } ^ { 2 } .
$$

Using the bound $\| E _ { b i a s } \| _ { 2 } ^ { 2 } \le B _ { \phi } ^ { 2 } \epsilon _ { a p p }$ from Lemma 5.1 gives:

$$
\bigg | \frac { 1 } { 1 - \gamma } \big \langle E _ { b i a s } , \Delta \omega _ { t } \big \rangle \bigg | \leq \frac { \lambda } { 8 C _ { j o i n t } ( 1 - \gamma ) } \| \Delta \omega _ { t } \| _ { 2 } ^ { 2 } + \frac { 2 C _ { j o i n t } B _ { \phi } ^ { 2 } } { \lambda ( 1 - \gamma ) } \epsilon _ { a p p } .
$$

(v) Synthesis. We substitute the decoupled components back into the smoothness bound:

$$
\begin{array} { r l } & { \quad J _ { \tau } ( \omega _ { t + 1 } ) - J _ { \tau } ( \omega _ { t } ) } \\ & { \geq \eta _ { t } \left( \frac { \lambda } { 2 C _ { j o i n t } ( 1 - \gamma ) } - \frac { \lambda } { 8 C _ { j o i n t } ( 1 - \gamma ) } - \frac { L _ { J } \eta _ { t } } { 2 } \right) \| \Delta \omega _ { t } \| _ { 2 } ^ { 2 } - \eta _ { t } C _ { Z } \| e _ { t } \| _ { 2 } ^ { 2 } - \eta _ { t } C _ { b i a s } \epsilon _ { a p p } . } \end{array}
$$

By setting the step size $\begin{array} { r } { \eta _ { t } \ \leq \ \frac { \lambda } { 2 L _ { J } C _ { j o i n t } ( 1 - \gamma ) } } \end{array}$ , we guarantee $\begin{array} { r } { \frac { L _ { J } \eta _ { t } } { 2 } \le \frac { \lambda } { 4 C _ { j o i n t } ( 1 - \gamma ) } } \end{array}$ . The coeficient

associated with $\eta _ { t } \| \Delta \omega _ { t } \| _ { 2 } ^ { 2 }$ is lower bounded by:

$$
\frac { \lambda } { 2 C _ { j o i n t } ( 1 - \gamma ) } - \frac { \lambda } { 8 C _ { j o i n t } ( 1 - \gamma ) } - \frac { \lambda } { 4 C _ { j o i n t } ( 1 - \gamma ) } = \frac { \lambda } { 8 C _ { j o i n t } ( 1 - \gamma ) } \triangleq \rho _ { a c } .
$$

Substituting $J _ { \tau } ( \omega _ { t + 1 } ) - J _ { \tau } ( \omega _ { t } ) = \mathrm { G a p } _ { t } - \mathrm { G a p } _ { t + 1 }$ and taking the expectation completes the proof. □

Proof of Lemma 5.4 (Coupled Actor Recurrence). We establish the expected linear recurrence by coupling the Actor Progress Bound with the objective’s PL geometry and the actor’s parameter update.

(i) Inverting PL Geometry. The term $\lVert \theta _ { \tau } ^ { * } ( \omega _ { t } ) - \tau \omega _ { t } \rVert _ { 2 } ^ { 2 }$ measures the distance of the actor’s scaled parameter to the ideal uncentered critic parameter. We invert the Parameter-Space PL Condition (Lemma 4.2) to lower-bound this ideal target distance:

$$
\| \theta _ { \tau } ^ { * } ( \omega _ { t } ) - \tau \omega _ { t } \| _ { 2 } ^ { 2 } \geq \frac { \tau } { C _ { P L } } \mathrm { G a p } _ { t } - \frac { C _ { e r r } } { C _ { P L } } \epsilon _ { a p p } .
$$

(ii) Bridging to Actor’s Parameter Update. We relate the ideal target distance to the actor’s update direction $\Delta \omega _ { t } = \theta _ { t + 1 } - \tau \omega _ { t }$ via the triangle inequality: $\lVert \theta _ { \tau } ^ { * } ( \omega _ { t } ) - \tau \omega _ { t } \rVert _ { 2 } ^ { 2 } \leq 2 \lVert \theta _ { t + 1 } - \tau \omega _ { t } \rVert _ { 2 } ^ { 2 } +$ $2 \| e _ { t } \| _ { 2 } ^ { 2 }$ , where the tracking residual is $e _ { t } = \theta _ { t + 1 } - \theta _ { \tau } ^ { * } ( \omega _ { t } )$ . Substituting this into our inverted PL bound and rearranging provides a lower bound on the actor’s actual update norm:

$$
\| \theta _ { t + 1 } - \tau \omega _ { t } \| _ { 2 } ^ { 2 } \geq \frac { \tau } { 2 C _ { P L } } \mathrm { G a p } _ { t } - \frac { C _ { e r r } } { 2 C _ { P L } } \epsilon _ { a p p } - \| e _ { t } \| _ { 2 } ^ { 2 } .
$$

(iii) Synthesis. We take the expectation of this bound and substitute it directly into the Actor Progress Bound (Lemma 5.3):

$$
\mathbb { E } [ \mathrm { G a p } _ { t + 1 } ] \leq \mathbb { E } [ \mathrm { G a p } _ { t } ] - \eta _ { t } \rho _ { a c } \left( \frac { \tau } { 2 C _ { P L } } \mathbb { E } [ \mathrm { G a p } _ { t } ] - \frac { C _ { e r r } } { 2 C _ { P L } } \epsilon _ { a p p } - Z _ { t } \right) + \eta _ { t } C _ { Z } Z _ { t } + \eta _ { t } C _ { b i a s } \epsilon _ { a p p } .
$$

Grouping the coeficients for $\mathbb { E } [ \mathrm { G a p } _ { t } ] , Z _ { t } .$ , and $\epsilon _ { a p p }$ exactly yields the composite constants $\rho _ { g a p } , \tilde { C } _ { Z }$ and $C _ { \epsilon , g a p } .$ establishing the stated linear descent. □

Proof of Lemma 5.5 (Coupled Critic Tracking). (i) Unprojected Tracking Error. We bound the expected tracking error at the next step, $Z _ { t + 1 } = \mathbb { E } [ \Vert e _ { t + 1 } \Vert _ { 2 } ^ { 2 } ]$ , where $e _ { t + 1 } = \theta _ { t + 2 } - \theta _ { \tau } ^ { * } ( \omega _ { t + 1 } )$ By definition, $\theta _ { t + 2 } = \Pi _ { R _ { \theta } } \big ( \theta _ { t + 1 } - \alpha _ { t + 1 } g _ { t + 1 } ^ { c r } ( s _ { t + 1 } , a _ { t + 1 } ) \big )$ . By Lemma 2.2 and the choice $R _ { \theta } = B _ { \theta }$ the ideal critic resides within the capacity ball, $\theta _ { \tau } ^ { * } ( \omega _ { t + 1 } ) = \Pi _ { R _ { \theta } } ( \theta _ { \tau } ^ { * } ( \omega _ { t + 1 } ) )$ . Because Euclidean projection onto a convex set is non-expansive, the error is bounded by the unprojected distance:

$$
\begin{array} { r } { \| e _ { t + 1 } \| _ { 2 } ^ { 2 } \leq \big \| \theta _ { t + 1 } - \alpha _ { t + 1 } g _ { t + 1 } ^ { c r } ( s _ { t + 1 } , a _ { t + 1 } ) - \theta _ { \tau } ^ { * } ( \omega _ { t + 1 } ) \big \| _ { 2 } ^ { 2 } = \big \| \tilde { e } _ { t } - \alpha _ { t + 1 } g _ { t + 1 } ^ { c r } ( s _ { t + 1 } , a _ { t + 1 } ) \big \| _ { 2 } ^ { 2 } . } \end{array}
$$

where $\tilde { e } _ { t } \triangleq \theta _ { t + 1 } - \theta _ { \tau } ^ { \ast } ( \omega _ { t + 1 } )$ denotes the intermediate tracking error prior to the next critic update.

(ii) SGD Contraction. We take the conditional expectation with respect to the filtration $\mathcal { F } _ { t + 1 }$ Because the parameters $\theta _ { t + 1 }$ and $\omega _ { t + 1 }$ are determined by $\mathcal { F } _ { t + 1 }$ , the intermediate distance prior to the SGD update, $\tilde { e } _ { t } = \theta _ { t + 1 } - \theta _ { \tau } ^ { * } ( \omega _ { t + 1 } )$ , is deterministic conditioned on $\mathcal { F } _ { t + 1 }$ . Expanding the squared norm yields:

$$
\left. | \tilde { e } _ { t } - \alpha _ { t + 1 } g _ { t + 1 } ^ { c r } \right. | _ { 2 } ^ { 2 } = \lVert \tilde { e } _ { t } \rVert _ { 2 } ^ { 2 } - 2 \alpha _ { t + 1 } \langle \tilde { e } _ { t } , g _ { t + 1 } ^ { c r } \rangle + \alpha _ { t + 1 } ^ { 2 } \lVert g _ { t + 1 } ^ { c r } \rVert _ { 2 } ^ { 2 } .
$$

Because the state-action pair is sampled from the current policy’s visitation measure $( s _ { t + 1 } , a _ { t + 1 } ) \sim$ $d ^ { \pi _ { t + 1 } } \times \pi _ { t + 1 }$ and $\hat { Q } _ { t + 1 }$ is a conditionally unbiased estimator such that $\mathbb E [ \hat { Q } _ { t + 1 } \mid \mathcal F _ { t + 1 } , s _ { t + 1 } , a _ { t + 1 } ] =$ $Q _ { \tau } ^ { \pi _ { t + 1 } } ( s _ { t + 1 } , a _ { t + 1 } )$ , we apply the Tower Property to evaluate the expected stochastic gradient:

$$
\begin{array} { r l } & { \mathbb { E } \big [ g _ { t + 1 } ^ { c r } ( s _ { t + 1 } , a _ { t + 1 } ) \mid \mathcal { F } _ { t + 1 } \big ] } \\ & { = \mathbb { E } \Big [ \mathbb { E } \big [ \big ( \theta _ { t + 1 } ^ { \top } \phi ( s _ { t + 1 } , a _ { t + 1 } ) - \hat { Q } _ { t + 1 } \big ) \phi ( s _ { t + 1 } , a _ { t + 1 } ) \mid \mathcal { F } _ { t + 1 } , s _ { t + 1 } , a _ { t + 1 } \big ] \Big | \mathcal { F } _ { t + 1 } \Big ] } \\ & { = \mathbb { E } \Big [ \phi ( s _ { t + 1 } , a _ { t + 1 } ) \phi ( s _ { t + 1 } , a _ { t + 1 } ) ^ { \top } \theta _ { t + 1 } - \phi ( s _ { t + 1 } , a _ { t + 1 } ) Q _ { \tau } ^ { \pi _ { t + 1 } } ( s _ { t + 1 } , a _ { t + 1 } ) \Big | \mathcal { F } _ { t + 1 } \Big ] . } \end{array}
$$

Because given $\mathcal { F } _ { t + 1 } , ( s _ { t + 1 } , a _ { t + 1 } )$ is sampled from $d ^ { \pi _ { t + 1 } } \times \pi _ { t + 1 }$ , this expected value evaluates to the gradient of the expected least-squares objective:

$$
\begin{array} { r l } & { \mathbb { E } [ g _ { t + 1 } ^ { c r } ( s _ { t + 1 } , a _ { t + 1 } ) \mid \mathcal { F } _ { t + 1 } ] = \mathbb { E } _ { ( s , a ) \sim d ^ { \pi _ { t + 1 } } \times \pi _ { t + 1 } } \left[ \phi ( s , a ) \phi ( s , a ) ^ { \top } \theta _ { t + 1 } - \phi ( s , a ) Q _ { \tau } ^ { \pi _ { t + 1 } } ( s , a ) \right] } \\ & { \qquad = \bar { \Sigma } _ { u n c } ( \pi _ { t + 1 } ) \theta _ { t + 1 } - b ( \pi _ { t + 1 } ) , } \end{array}
$$

where $\bar { \Sigma } _ { u n c } ( \pi _ { t + 1 } ) \triangleq \mathbb { E } _ { d ^ { \pi _ { t + 1 } } , \pi _ { t + 1 } } [ \phi ( s , a ) \phi ( s , a ) ^ { \top } ]$ and $b ( \pi _ { t + 1 } ) \triangleq \mathbb { E } _ { d ^ { \pi _ { t + 1 } } , \pi _ { t + 1 } } [ \phi ( s , a ) Q _ { \tau } ^ { \pi _ { t + 1 } } ( s , a ) ]$ . By the first-order condition for the ideal critic, $\bar { \Sigma } _ { u n c } ( \pi _ { t + 1 } ) \theta _ { \tau } ^ { * } ( \omega _ { t + 1 } ) = b ( \pi _ { t + 1 } )$ . Substituting this identity yields $\mathbb { E } [ g _ { t + 1 } ^ { c r } ( s _ { t + 1 } , a _ { t + 1 } ) \mid \mathcal { F } _ { t + 1 } ] = \bar { \Sigma } _ { u n c } ( \pi _ { t + 1 } ) ( \theta _ { t + 1 } - \theta _ { \tau } ^ { * } ( \omega _ { t + 1 } ) ) = \bar { \Sigma } _ { u n c } ( \pi _ { t + 1 } ) \tilde { e } _ { t } .$

Substituting this expected gradient back into the inner product yields the quadratic form $\langle \tilde { e } _ { t } , \bar { \Sigma } _ { u n c } ( \pi _ { t + 1 } ) \tilde { e } _ { t } \rangle$ . By the Uncentered Feature Positive-Definiteness (Assumption 3), this is bounded below by $\kappa \| \tilde { e } _ { t } \| _ { 2 } ^ { 2 }$ . Applying the gradient second moment bound $\mathbb { E } [ \| g _ { t + 1 } ^ { c r } \| _ { 2 } ^ { 2 } \ | \ \mathcal { F } _ { t + 1 } ] \le G _ { c r } ^ { 2 }$ (Lemma 2.2(i)), the expected SGD step contracts as:

$$
\begin{array} { r } { \mathbb { E } [ \| e _ { t + 1 } \| _ { 2 } ^ { 2 } | \mathcal { F } _ { t + 1 } ] \le ( 1 - 2 \alpha _ { t + 1 } \kappa ) \| \tilde { e } _ { t } \| _ { 2 } ^ { 2 } + \alpha _ { t + 1 } ^ { 2 } G _ { c r } ^ { 2 } . } \end{array}
$$

(iii) Young’s Inequality. We decompose the intermediate error $\tilde { e } _ { t }$ by separating $e _ { t } = \theta _ { t + 1 } - \theta _ { \tau } ^ { * } ( \omega _ { t } )$ from the target drift:

$$
\tilde { e } _ { t } = e _ { t } + \left( \theta _ { \tau } ^ { * } ( \omega _ { t } ) - \theta _ { \tau } ^ { * } ( \omega _ { t + 1 } ) \right) .
$$

Applying Young’s Inequality $( \| x + y \| _ { 2 } ^ { 2 } \leq ( 1 + p ) \| x \| _ { 2 } ^ { 2 } + ( 1 + 1 / p ) \| y \| _ { 2 } ^ { 2 } )$ with parameter $p = \alpha _ { t + 1 } \kappa$ yields:

$$
\| \tilde { e } _ { t } \| _ { 2 } ^ { 2 } \leq ( 1 + \alpha _ { t + 1 } \kappa ) \| e _ { t } \| _ { 2 } ^ { 2 } + \left( 1 + \frac { 1 } { \alpha _ { t + 1 } \kappa } \right) \| \theta _ { \tau } ^ { * } ( \omega _ { t } ) - \theta _ { \tau } ^ { * } ( \omega _ { t + 1 } ) \| _ { 2 } ^ { 2 } .
$$

From the step-size condition $\textstyle \alpha _ { t + 1 } = \xi \eta _ { t } \leq { \frac { 1 } { 2 \kappa } }$ , the SGD contraction multiplier is non-negative $( 1 - 2 \alpha _ { t + 1 } \kappa \geq 0 )$ . This allows us to substitute the upper bound for $\| \tilde { e } _ { t } \| _ { 2 } ^ { 2 }$ into the SGD contraction from Step (ii). Applying the relaxations $( 1 - 2 \alpha _ { t + 1 } \kappa ) ( 1 + \alpha _ { t + 1 } \kappa ) \leq 1 - \alpha _ { t + 1 } \kappa$ and $( 1 - 2 \alpha _ { t + 1 } \kappa ) ( 1 +$ $\begin{array} { r } { \frac { 1 } { \alpha _ { t + 1 } \kappa } \big ) \leq \frac { 1 - \alpha _ { t + 1 } \kappa } { \alpha _ { t + 1 } \kappa } \leq \frac { 1 } { \alpha _ { t + 1 } \kappa } } \end{array}$ gives:

$$
{  { \mathbb E } } [ \| e _ { t + 1 } \| _ { 2 } ^ { 2 } \ | \ {  { \mathcal F } } _ { t + 1 } ] \le ( 1 - \alpha _ { t + 1 } \kappa ) \| e _ { t } \| _ { 2 } ^ { 2 } + \frac { 1 } { \alpha _ { t + 1 } \kappa } \| \theta _ { \tau } ^ { * } ( \omega _ { t } ) - \theta _ { \tau } ^ { * } ( \omega _ { t + 1 } ) \| _ { 2 } ^ { 2 } + \alpha _ { t + 1 } ^ { 2 } G _ { c r } ^ { 2 } .
$$

Taking the total expectation $\mathbb { E } [ \cdot ]$ over the algorithmic trajectory yields the single-step recurrence:

$$
Z _ { t + 1 } \le \underbrace { ( 1 - \alpha _ { t + 1 } \kappa ) Z _ { t } + \alpha _ { t + 1 } ^ { 2 } G _ { c r } ^ { 2 } } _ { \mathrm { S G D ~ E r r o r } } + \underbrace { \frac { 1 } { \alpha _ { t + 1 } \kappa } \mathbb { E } [ \| \theta _ { \tau } ^ { * } ( \omega _ { t } ) - \theta _ { \tau } ^ { * } ( \omega _ { t + 1 } ) \| _ { 2 } ^ { 2 } ] } _ { \mathrm { T a r g e t ~ D r i f t } } .
$$

(iv) Controlling Target Drift. Because the ideal critic is $L _ { \theta ^ { - } } \mathrm { L i p s c h i t z }$ continuous (Lemma 2.3), the target drift is bounded by the actor’s update:

$$
\begin{array} { r } { \| \theta _ { \tau } ^ { * } ( \omega _ { t } ) - \theta _ { \tau } ^ { * } ( \omega _ { t + 1 } ) \| _ { 2 } ^ { 2 } \leq L _ { \theta } ^ { 2 } \| \omega _ { t + 1 } - \omega _ { t } \| _ { 2 } ^ { 2 } = L _ { \theta } ^ { 2 } \eta _ { t } ^ { 2 } \| \theta _ { t + 1 } - \tau \omega _ { t } \| _ { 2 } ^ { 2 } . } \end{array}
$$

We rearrange the Actor Progress Bound (Lemma 5.3) to bound the expected magnitude of the actor update by the objective progress:

$$
\eta _ { t } \mathbb { E } [ \| \theta _ { t + 1 } - \tau \omega _ { t } \| _ { 2 } ^ { 2 } ] \le \frac { 1 } { \rho _ { a c } } \big ( \mathbb { E } [ \mathrm { G a p } _ { t } ] - \mathbb { E } [ \mathrm { G a p } _ { t + 1 } ] \big ) + \frac { C _ { Z } \eta _ { t } } { \rho _ { a c } } Z _ { t } + \frac { C _ { b i a s } \eta _ { t } } { \rho _ { a c } } \epsilon _ { a p p } .
$$

Combining these two bounds leads to:

$$
\mathbb { E } [ \| \theta _ { \tau } ^ { * } ( \omega _ { t } ) - \theta _ { \tau } ^ { * } ( \omega _ { t + 1 } ) \| _ { 2 } ^ { 2 } ] \leq L _ { \theta } ^ { 2 } \eta _ { t } \left( \frac { 1 } { \rho _ { a c } } ( \mathbb { E } [ \mathrm { G a p } _ { t } ] - \mathbb { E } [ \mathrm { G a p } _ { t + 1 } ] ) + \frac { C _ { Z } \eta _ { t } } { \rho _ { a c } } Z _ { t } + \frac { C _ { b i a s } \eta _ { t } } { \rho _ { a c } } \epsilon _ { a p p } \right) .
$$

(v) Synthesis. Substituting the target drift bound into the tracking recurrence yields:

$$
Z _ { t + 1 } \leq ( 1 - \alpha _ { t + 1 } \kappa ) Z _ { t } + \frac { L _ { \theta } ^ { 2 } \eta _ { t } } { \alpha _ { t + 1 } \kappa } \left( \frac { 1 } { \rho _ { a c } } \big ( \mathbb { E } [ \mathrm { G a p } _ { t } ] - \mathbb { E } [ \mathrm { G a p } _ { t + 1 } ] \big ) + \frac { C _ { Z } \eta _ { t } } { \rho _ { a c } } Z _ { t } + \frac { C _ { b i a s } \eta _ { t } } { \rho _ { a c } } \epsilon _ { a p p } \right) + \alpha _ { t + 1 } ^ { 2 } G _ { c r } ^ { 2 } .
$$

By setting $\alpha _ { t + 1 } = \xi \eta _ { t }$ , the ratio evaluates to $\begin{array} { r } { \frac { \eta _ { t } } { \alpha _ { t + 1 } } = \frac { 1 } { \xi } } \end{array}$ , and the critic decay evaluates to $\alpha _ { t + 1 } \kappa =$ $\xi \kappa \eta _ { t }$ . Substituting this simplifies the inequality to:

$$
\begin{array} { r l } & { Z _ { t + 1 } \leq \left( 1 - \xi \kappa \eta _ { t } \right) Z _ { t } + \frac { L _ { \theta } ^ { 2 } } { \xi \kappa \rho _ { a c } } \bigl ( \mathbb { E } [ \mathrm { G a p } _ { t } ] - \mathbb { E } [ \mathrm { G a p } _ { t + 1 } ] \bigr ) } \\ & { \quad \quad + \eta _ { t } \frac { L _ { \theta } ^ { 2 } C _ { Z } } { \xi \kappa \rho _ { a c } } Z _ { t } + \eta _ { t } \frac { L _ { \theta } ^ { 2 } C _ { b i a s } } { \xi \kappa \rho _ { a c } } \epsilon _ { a p p } + \xi ^ { 2 } \eta _ { t } ^ { 2 } G _ { c r } ^ { 2 } . } \end{array}
$$

Grouping the coeficients for $Z _ { t }$ , the objective diference, and ϵ exactly forms the composite $\epsilon _ { a p p }$ constants $\rho _ { c r } , C _ { g a p } ,$ and $C _ { \epsilon , c r } ,$ completing the proof. □

## E.5 Proofs for Section 6

Proof of Theorem 6.1 (Coupled Lyapunov Recurrence). (i) Algebraic System Fusion. We rearrange the Coupled Critic Tracking sequence (Lemma 5.5) to align the expected performance

gap terms:

$$
Z _ { t + 1 } + C _ { g a p } \mathbb { E } [ \mathrm { G a p } _ { t + 1 } ] \leq ( 1 - \rho _ { c r } \eta _ { t } ) Z _ { t } + C _ { g a p } \mathbb { E } [ \mathrm { G a p } _ { t } ] + \xi ^ { 2 } \eta _ { t } ^ { 2 } G _ { c r } ^ { 2 } + \eta _ { t } C _ { \epsilon , c r } \epsilon _ { a p p } .
$$

We multiply this by the coupling constant $\beta > 0$ and add it to the Coupled Actor Recurrence inequality (Lemma 5.4):

$$
\begin{array} { r l } & { \mathcal { L } _ { t + 1 } \leq \mathbb { E } [ \mathrm { G a p } _ { t } ] - \rho _ { g a p } \eta _ { t } \mathbb { E } [ \mathrm { G a p } _ { t } ] + \eta _ { t } \tilde { C } _ { Z } Z _ { t } + \beta ( 1 - \rho _ { c r } \eta _ { t } ) Z _ { t } + \beta C _ { g a p } \mathbb { E } [ \mathrm { G a p } _ { t } ] } \\ & { \qquad + \beta \xi ^ { 2 } \eta _ { t } ^ { 2 } G _ { c r } ^ { 2 } + \eta _ { t } ( C _ { \epsilon , g a p } + \beta C _ { \epsilon , c r } ) \epsilon _ { a p p } . } \end{array}
$$

(ii) Constraints for Joint Recurrence. To prove the joint recurrence $\mathcal { L } _ { t + 1 } \leq ( 1 - \rho \varsigma \eta _ { t } ) \mathcal { L } _ { t } + . . .$ we group the coeficients for $\mathbb { E } [ \mathrm { G a p } _ { t } ]$ and $Z _ { t }$ and demand that they are bounded by $( 1 - \rho _ { \mathcal { L } } \eta _ { t } ) \mathcal { L } _ { t } \mathrm { : }$

$$
( 1 - \rho \mathcal { L } \eta _ { t } ) \mathcal { L } _ { t } = ( 1 - \rho \mathcal { \ L } \eta _ { t } ) ( 1 + \beta C _ { g a p } ) \mathbb { E } [ \mathrm { G a p } _ { t } ] + ( 1 - \rho \mathcal { \ L } \eta _ { t } ) \beta Z _ { t } .
$$

For this to hold, the efective decay rates for both components must match or exceed the target rate $\rho _ { \mathcal { L } }$ , yielding two independent constraints:

• Actor Constraint: $\begin{array} { r } { 1 + \beta C _ { g a p } - \rho _ { g a p } \eta _ { t } \le ( 1 - \rho _ { \mathcal { L } } \eta _ { t } ) ( 1 + \beta C _ { g a p } ) \implies \rho _ { \mathcal { L } } \le \frac { \rho _ { g a p } } { 1 + \beta C _ { g a p } } . } \end{array}$

• Critic Constraint: $\begin{array} { r } { \beta - \beta \rho _ { c r } \eta _ { t } + \eta _ { t } \tilde { C } _ { Z } \leq \beta - \beta \rho _ { \mathcal { L } } \eta _ { t } \implies \rho _ { \mathcal { L } } \leq \rho _ { c r } - \frac { \tilde { C } _ { Z } } { \beta } . } \end{array}$

(iii) Derivation of Joint Rate and Coupling Weight. To maximize progress, we set the joint rate exactly to the actor’s bottleneck: $\begin{array} { r } { \rho _ { \mathcal { L } } \triangleq \frac { \rho _ { g a p } } { 1 + \beta C _ { g a p } } } \end{array}$ . To satisfy the critic constraint, we ensure $\begin{array} { r } { \rho _ { \mathcal { L } } \leq \rho _ { c r } - \frac { \tilde { C } _ { Z } } { \beta } } \end{array}$ . For the algorithm to converge $( \rho _ { \mathscr { L } } > 0 )$ , we require $\begin{array} { r } { \beta > \frac { { { \tilde { C } } _ { Z } } } { \rho _ { c r } } } \end{array}$ . To guarantee stability with a safe margin, we double this minimum requirement by setting $\begin{array} { r } { \beta \triangleq \frac { 2 { { { \tilde { C } } } _ { Z } } } { \rho _ { c r } } } \end{array}$ . Substituting this choice simplifies the critic constraint to $\rho { c } \leq \frac { \rho _ { c r } } { 2 }$ . Because $\rho _ { \mathcal { L } } \leq \rho _ { g a p } ,$ a suficient condition for joint recurrence is simply $\rho _ { c r } \ge 2 \rho _ { g a p }$

(iv) Timescale Separation Limit. We substitute the native critic rate $\begin{array} { r } { \rho _ { c r } \triangleq \xi \kappa - \frac { S ^ { 2 } } { \xi \kappa } } \end{array}$ , where $S \triangleq L _ { \theta } \sqrt { \frac { C _ { Z } } { \rho _ { a c } } }$ , into the stability requirement:

$$
\xi \kappa - \frac { S ^ { 2 } } { \xi \kappa } \geq 2 \rho _ { g a p } \implies ( \kappa \xi ) ^ { 2 } - 2 \rho _ { g a p } ( \kappa \xi ) - S ^ { 2 } \geq 0 .
$$

By the quadratic formula and ${ \sqrt { x + y } } \leq { \sqrt { x } } + { \sqrt { y } }$ , this is satisfied for any $\kappa \xi \geq 2 \rho _ { g a p } + S$ . However, the choice of $\xi$ dictates the joint tracking variance $\Sigma _ { \mathcal { L } } = \beta \xi ^ { 2 } G _ { c r } ^ { 2 }$ . Substituting $\beta ,$ , the variance physically scales as $\begin{array} { r } { \Sigma _ { \mathcal { L } } \propto \frac { \xi ^ { 2 } } { \rho _ { c r } } = \frac { \xi ^ { 3 } \kappa } { ( \kappa \xi ) ^ { 2 } - S ^ { 2 } } } \end{array}$ . To minimize this variance, we diferentiate with respect to $\xi$ and set the numerator to zero:

$$
3 \xi ^ { 2 } \big ( ( \kappa \xi ) ^ { 2 } - S ^ { 2 } \big ) - \xi ^ { 3 } ( 2 \kappa ^ { 2 } \xi ) = 0 \Longleftrightarrow ( \kappa \xi ) ^ { 2 } = 3 S ^ { 2 } .
$$

This reveals that the theoretical minimum occurs at $\kappa \xi = \sqrt { 3 } S$ . For a simple and conservative

structural bound, we explicitly define the threshold as $\kappa \xi \geq 2 \rho _ { g a p } + 2 S$ . Substituting the definition of S yields the required structural lower bound on the timescale ratio $\xi .$

(v) Synthesis. With $\beta$ and $\xi$ satisfying the joint recurrence constraints, we consolidate the remaining additive terms. The joint variance evaluates to $\Sigma _ { \mathcal { L } } = \beta \xi ^ { 2 } G _ { c r } ^ { 2 }$ , and the joint bias groups to $C _ { \epsilon , \mathcal { L } } \epsilon _ { a p p } = ( C _ { \epsilon , g a p } + \beta C _ { \epsilon , c r } ) \epsilon _ { a p p }$ , completing the proof. □

Proof of Lemma 6.1 (Telescoping Robbins-Monro Sequence). We rearrange the recursive bound to isolate the linear decay:

$$
r \eta _ { t } x _ { t } \leq x _ { t } - x _ { t + 1 } + a \eta _ { t } ^ { 2 } + b \eta _ { t } .
$$

Dividing both sides by $r \eta _ { t }$ yields $\begin{array} { r } { x _ { t } \le \frac { 1 } { r \eta _ { t } } x _ { t } - \frac { 1 } { r \eta _ { t } } x _ { t + 1 } + \frac { a } { r } \eta _ { t } + \frac { b } { r } } \end{array}$ . We sum this inequality from $t = 0$ to $T - 1$ . Substituting $\begin{array} { r } { \frac { 1 } { r \eta _ { t } } = \frac { t + t _ { 0 } } { c } } \end{array}$ and expanding the sum:

$$
\sum _ { t = 0 } ^ { T - 1 } x _ { t } \leq { \frac { t _ { 0 } } { c } } x _ { 0 } - { \frac { T - 1 + t _ { 0 } } { c } } x _ { T } + \sum _ { t = 1 } ^ { T - 1 } x _ { t } \left( { \frac { t + t _ { 0 } } { c } } - { \frac { t - 1 + t _ { 0 } } { c } } \right) + { \frac { a } { r } } \sum _ { t = 0 } ^ { T - 1 } \eta _ { t } + { \frac { b T } { r } } .
$$

Because $x _ { T } \geq 0$ , we drop the terminal term. The diference inside the summation simplifies to ${ \frac { 1 } { c } } .$ To consolidate the $x _ { t }$ sequences, we add and subtract ${ \scriptstyle { \frac { 1 } { c } } } x _ { 0 } \colon$

$$
\sum _ { t = 0 } ^ { T - 1 } x _ { t } \leq \frac { t _ { 0 } - 1 } { c } x _ { 0 } + \frac { 1 } { c } \sum _ { t = 0 } ^ { T - 1 } x _ { t } + \frac { a c _ { \eta } } { r } \sum _ { t = 0 } ^ { T - 1 } \frac { 1 } { t + t _ { 0 } } + \frac { b T } { r } .
$$

We subtract the inner sum from both sides to get $( 1 - \frac { 1 } { c } ) \sum x _ { t }$ , and multiply the entire inequality by $\frac { c } { c - 1 }$

$$
\sum _ { t = 0 } ^ { T - 1 } x _ { t } \leq { \frac { t _ { 0 } - 1 } { c - 1 } } x _ { 0 } + { \frac { a c c _ { \eta } } { r ( c - 1 ) } } \sum _ { t = 0 } ^ { T - 1 } { \frac { 1 } { t + t _ { 0 } } } + { \frac { b c T } { r ( c - 1 ) } } .
$$

Substituting $c = r c _ { \eta } .$ the coeficients simplify to $\frac { a c _ { \eta } ^ { 2 } } { c - 1 }$ and $\frac { b c _ { \eta } T } { c - 1 }$ . Because $t _ { 0 } \geq 1$ , the harmonic sum is bounded by $\begin{array} { r } { \sum _ { t = 0 } ^ { T - 1 } \frac { 1 } { t + t _ { 0 } } \leq 1 + \log T } \end{array}$ . Dividing the entire expression by $T$ completes the proof. □

Proof of Theorem 6.2 (Average-Iterate Regularized Convergence). Because the initial ofset $\begin{array} { r } { t _ { 0 } \geq \frac { c _ { \eta } } { \eta _ { m a x } } } \end{array}$ and the step size schedule is decreasing, the maximum step size occurs at $t =$ $0$ and satisfies $\begin{array} { r } { \eta _ { t } \ \leq \ \eta _ { 0 } \ = \ \frac { c _ { \eta } } { t _ { 0 } } \ \leq \ \eta _ { m a x } } \end{array}$ for all $t ~ \geq ~ 0$ . By the definition of $\eta _ { m a x }$ , this ensures the required step-size condition $\begin{array} { r } { \eta _ { t } \ \leq \ \operatorname* { m i n } \left( \frac { \lambda } { 2 L _ { J } C _ { j o i n t } ( 1 - \gamma ) } , \frac { 1 } { 2 \xi \kappa } \right) } \end{array}$ . Therefore, the joint Lyapunov recurrence (Theorem 6.1) is valid, meaning the Lyapunov sequence satisfies $\mathscr { L } _ { t + 1 } \leq ( 1 - \rho _ { \mathscr { L } } \eta _ { t } ) \mathscr { L } _ { t } +$ $a \eta _ { t } ^ { 2 } + b \eta _ { t }$ , where $a = \Sigma c$ and $b = C _ { \epsilon , \mathcal { L } } \epsilon _ { a p p }$

The remaining conditions on $t _ { 0 }$ and $c _ { \eta }$ guarantee the structural prerequisites for the Telescoping Robbins-Monro Sequence bound (Lemma 6.1). The choice $\begin{array} { r } { c _ { \eta } > \frac { 1 } { \rho _ { \mathcal { L } } } } \end{array}$ satisfies the numerator requirement $c \triangleq \rho _ { \mathcal { L } } c _ { \eta } > 1$ , and $t _ { 0 } \geq 1$ provides the required initial ofset. Applying Lemma 6.1 with $x _ { t } = \mathcal L _ { t }$ and $r = \rho _ { \mathcal { L } }$ establishes the bound on the average Lyapunov sequence:

$$
\frac { 1 } { T } \sum _ { t = 0 } ^ { T - 1 } \mathcal { L } _ { t } \leq \frac { t _ { 0 } - 1 } { T ( \rho _ { \mathcal { L } } c _ { \eta } - 1 ) } \mathcal { L } _ { 0 } + \frac { c _ { \eta } ^ { 2 } \Sigma _ { \mathcal { L } } } { \rho _ { \mathcal { L } } c _ { \eta } - 1 } \left( \frac { 1 + \log T } { T } \right) + \frac { c _ { \eta } C _ { \epsilon , \mathcal { L } } } { \rho _ { \mathcal { L } } c _ { \eta } - 1 } \epsilon _ { a p p } .
$$

To evaluate the structural orders, we recall from Remark 6.1 that $\begin{array} { r } { \rho _ { \mathscr { L } } = \Theta ( \lambda \tau ) , \Sigma _ { \mathscr { L } } = \Theta ( ( 1 - } \end{array}$ $\gamma ) ^ { - 5 } \kappa ^ { - 6 } \lambda ^ { - 1 / 2 } )$ , and $C _ { \epsilon , \mathcal { L } } = \Theta ( ( 1 - \gamma ) ^ { - 1 } \lambda ^ { - 1 } )$ . We choose the schedule numerator $\begin{array} { r } { c _ { \eta } = \frac { c } { \rho _ { \mathcal { L } } } } \end{array}$ for a constant $c > 1$ , meaning $c _ { \eta } = \Theta ( \frac { 1 } { \lambda \tau } )$ . Then the bounding factor $\rho _ { \mathcal { L } } c _ { \eta } - 1$ evaluates to $c - 1 = \Theta ( 1 )$ From Lemma 5.2, the objective smoothness constant evaluates to $L _ { J } = \Theta ( ( 1 - \gamma ) ^ { - 3 } )$ , and the actor progress threshold evaluates to $\begin{array} { r } { \frac { \lambda } { 2 L _ { J } C _ { j o i n t } ( 1 - \gamma ) } = \Theta ( ( 1 - \gamma ) ^ { 2 } \lambda ) } \end{array}$ . Because the timescale ratio satisfies $\xi = \Theta ( ( 1 - \gamma ) ^ { - 2 } \kappa ^ { - 3 } \lambda ^ { - 1 / 2 } )$ , we have $\begin{array} { r } { \frac { 1 } { 2 \xi \kappa } = \Theta ( ( 1 - \gamma ) ^ { 2 } \kappa ^ { 2 } \lambda ^ { 1 / 2 } ) } \end{array}$ . Thus, the maximum allowable step size satisfies $\begin{array} { r } { \frac { 1 } { \eta _ { m a x } } = \Theta \bigl ( \frac { 1 } { ( 1 - \gamma ) ^ { 2 } \lambda } + \frac { 1 } { ( 1 - \gamma ) ^ { 2 } \kappa ^ { 2 } \lambda ^ { 1 / 2 } } \bigr ) } \end{array}$ . This allows us to satisfy the initial ofset condition by choosing $\begin{array} { r } { t _ { 0 } = \Theta ( \frac { c _ { \eta } } { \eta _ { m a x } } ) = \Theta \big ( \frac { 1 } { ( 1 - \gamma ) ^ { 2 } \lambda ^ { 2 } \tau } + \frac { 1 } { ( 1 - \gamma ) ^ { 2 } \kappa ^ { 2 } \lambda ^ { 3 / 2 } \tau } \big ) } \end{array}$

• Initialization Error: The initial Lyapunov function is $\mathcal { L } _ { 0 } = \mathbb { E } [ \mathrm { G a p } _ { 0 } ] + \beta ( Z _ { 0 } + C _ { g a p } \mathbb { E } [ \mathrm { G a p } _ { 0 } ] ) =$ $( 1 + \beta C _ { g a p } ) \mathbb { E } [ \mathrm { G a p } _ { 0 } ] + \beta Z _ { 0 }$ . By definition, $Z _ { 0 } = \mathbb { E } [ \| \theta _ { 1 } - \theta _ { \tau } ^ { * } ( \omega _ { 0 } ) \| _ { 2 } ^ { 2 } ]$ , which satisfies $Z _ { 0 } \le ( 2 B _ { \theta } ) ^ { 2 } =$ $G _ { a c } ^ { 2 }$ because $\theta _ { 1 }$ (by constrained SGD) and $\theta _ { \tau } ^ { * } ( \omega _ { 0 } )$ (by Lemma 2.2(i)) are in the ball of radius $B _ { \theta } .$ . Since $G _ { a c } ^ { 2 } = \Theta ( \kappa ^ { - 2 } ( 1 - \gamma ) ^ { - 2 } )$ , we have $Z _ { 0 } = \mathcal { O } ( \kappa ^ { - 2 } ( 1 - \gamma ) ^ { - 2 } )$ . Substituting $\beta = \Theta ( ( 1 -$ $\gamma ) \kappa ^ { 2 } \lambda ^ { 1 / 2 } )$ and $\beta C _ { g a p } = \Theta ( 1 )$ , we obtain $\mathcal { L } _ { 0 } = \mathcal { O } ( ( 1 - \gamma ) ^ { - 1 } ) + \mathcal { O } ( ( 1 - \gamma ) ^ { - 1 } \lambda ^ { 1 / 2 } ) = \mathcal { O } ( ( 1 - \gamma ) ^ { - 1 } )$ The initialization multiplier evaluates to $\begin{array} { r } { \frac { t _ { 0 } - 1 } { \rho _ { \mathscr { L } } c _ { \eta } - 1 } = \Theta ( t _ { 0 } ) = \Theta \bigl ( \frac { 1 } { ( 1 - \gamma ) ^ { 2 } \lambda ^ { 2 } \tau } + \frac { 1 } { ( 1 - \gamma ) ^ { 2 } \kappa ^ { 2 } \lambda ^ { 3 / 2 } \tau } \bigr ) } \end{array}$ . Thus, the expected initialization term scales as $\begin{array} { r } { \mathcal { O } \big ( \frac { 1 } { ( 1 - \gamma ) ^ { 3 } \lambda ^ { 2 } \tau T } + \frac { 1 } { ( 1 - \gamma ) ^ { 3 } \kappa ^ { 2 } \lambda ^ { 3 / 2 } \tau T } \big ) } \end{array}$

• Optimization Variance: The variance multiplier evaluates to:

$$
\frac { c _ { \eta } ^ { 2 } \Sigma _ { \mathcal { L } } } { \rho \mathscr { L } c _ { \eta } - 1 } = \Theta \left( \frac { 1 } { \lambda ^ { 2 } \tau ^ { 2 } } \frac { 1 } { ( 1 - \gamma ) ^ { 5 } \kappa ^ { 6 } \lambda ^ { 1 / 2 } } \right) = \Theta \left( \frac { 1 } { ( 1 - \gamma ) ^ { 5 } \kappa ^ { 6 } \lambda ^ { 5 / 2 } \tau ^ { 2 } } \right) .
$$

Multiplying by $\frac { 1 + \log T } { T }$ yields an $\mathcal { O } ( T ^ { - 1 } \log T )$ variance term, which dominates the $\mathcal { O } ( T ^ { - 1 } )$ initialization error above, taking account of the dependence on $\tau , \kappa , \lambda ,$ , and $( 1 - \gamma ) ^ { - 1 }$

• Approximation Bias: The bias multiplier evaluates to:

$$
\frac { c _ { \eta } C _ { \epsilon , \mathcal { L } } } { \rho _ { \mathcal { L } } c _ { \eta } - 1 } = \Theta \left( \frac { 1 } { \lambda \tau } \frac { 1 } { ( 1 - \gamma ) \lambda } \right) = \Theta \left( \frac { 1 } { ( 1 - \gamma ) \lambda ^ { 2 } \tau } \right) .
$$

Because the expected regularized performance gap is upper-bounded by the joint Lyapunov function $( \mathbb { E } [ \mathrm { G a p } _ { t } ] \leq \mathcal { L } _ { t } )$ , summing these evaluated components and dropping the dominated initialization term yields the stated asymptotic convergence rate. □

Proof of Theorem 6.3 (Last-Iterate Regularized Convergence). As established in the

proof of Theorem 6.2, the step-size schedule guarantees the required structural condition for the actor update for all $t \geq 0$ . Substituting $\begin{array} { r } { \begin{array} { r } { \eta _ { t } = \frac { c _ { \eta } } { t + t _ { 0 } } } \end{array} } \end{array}$ directly into the joint Lyapunov recurrence (Theorem 6.1) yields:

$$
\mathcal { L } _ { t + 1 } \leq \left( 1 - \frac { \rho \mathcal { L } c _ { \eta } } { t + t _ { 0 } } \right) \mathcal { L } _ { t } + \frac { c _ { \eta } ^ { 2 } \Sigma _ { \mathcal { L } } } { ( t + t _ { 0 } ) ^ { 2 } } + \frac { c _ { \eta } C _ { \epsilon , \mathcal { L } } \epsilon _ { a p p } } { t + t _ { 0 } } .
$$

We apply Chung’s Lemma (Lemma 6.2) by identifying the sequence $x _ { t } ~ = ~ \mathcal { L } _ { t }$ , the contraction constant $c = \rho _ { \mathscr { L } } c _ { \eta } .$ , the variance constant $a = c _ { \eta } ^ { 2 } \Sigma _ { \mathcal { L } }$ , and the bias constant $b = c _ { \eta } C _ { \epsilon , \mathcal { L } } \epsilon _ { a p p }$ . The choice $\begin{array} { r } { c _ { \eta } > \frac { 1 } { \rho _ { \mathcal { L } } } } \end{array}$ guarantees $c \triangleq \rho _ { \mathcal { L } } c _ { \eta } > 1$ , and the additional schedule condition $t _ { 0 } \geq \rho _ { \mathcal { L } } c _ { \eta }$ satisfies the initial ofset requirement $t _ { 0 } \geq c ,$ establishing the bound $\begin{array} { r } { \mathscr { L } _ { T } \leq \frac { v } { T + t _ { 0 } } + \frac { b } { c - 1 } } \end{array}$

To evaluate the structural orders, we rely on the identical choices of step-size parameters from Theorem 6.2: we choose $\begin{array} { r } { c _ { \eta } = \frac { c } { \rho _ { \mathcal { L } } } } \end{array}$ for a constant $c > 1$ , meaning $\begin{array} { r } { c _ { \eta } = \Theta ( \frac { 1 } { \lambda \tau } ) } \end{array}$ , and choose the initial ofset $\begin{array} { r } { t _ { 0 } = \Theta \big ( \frac { 1 } { ( 1 - \gamma ) ^ { 2 } \lambda ^ { 2 } \tau } + \frac { 1 } { ( 1 - \gamma ) ^ { 2 } \kappa ^ { 2 } \lambda ^ { 3 / 2 } \tau } \big ) } \end{array}$ , with the scaling constant for $t _ { 0 }$ suficiently large to guarantee the additional requirement $t _ { 0 } \geq \rho _ { \mathcal { L } } c _ { \eta } \left( = c \right)$

• Bounding Constant (v): The terminal constant is $\begin{array} { r } { v = \operatorname* { m a x } \left( t _ { 0 } \mathcal { L } _ { 0 } , \frac { c _ { \eta } ^ { 2 } \Sigma _ { \mathcal { L } } } { \rho _ { \mathcal { L } } c _ { \eta } - 1 } \right) } \end{array}$ . Substituting the bound $\mathcal { L } _ { 0 } = \mathcal { O } ( ( 1 - \gamma ) ^ { - 1 } )$ established in the proof of Theorem 6.2, the initialization argument evaluates to $\begin{array} { r } { t _ { 0 } \mathcal { L } _ { 0 } = \mathcal { O } \Big ( \frac { 1 } { ( 1 - \gamma ) ^ { 3 } \lambda ^ { 2 } \tau } + \frac { 1 } { ( 1 - \gamma ) ^ { 3 } \kappa ^ { 2 } \lambda ^ { 3 / 2 } \tau } \Big ) } \end{array}$ . As shown in the proof of Theorem 6.2, the variance argument evaluates to:

$$
\frac { c _ { \eta } ^ { 2 } \Sigma _ { \mathcal { L } } } { \rho _ { \mathcal { L } } c _ { \eta } - 1 } = \Theta \left( \frac { 1 } { \lambda ^ { 2 } \tau ^ { 2 } } \frac { 1 } { ( 1 - \gamma ) ^ { 5 } \kappa ^ { 6 } \lambda ^ { 1 / 2 } } \right) = \Theta \left( \frac { 1 } { ( 1 - \gamma ) ^ { 5 } \kappa ^ { 6 } \lambda ^ { 5 / 2 } \tau ^ { 2 } } \right)
$$

Taking account of the dependence on $\tau , \kappa , \lambda .$ , and $( 1 - \gamma ) ^ { - 1 }$ , the variance argument dominates the initialization argument. Thus, the bounding constant evaluates to $\begin{array} { r } { v = \Theta \left( \frac { 1 } { ( 1 - \gamma ) ^ { 5 } \kappa ^ { 6 } \lambda ^ { 5 / 2 } \tau ^ { 2 } } \right) } \end{array}$

• Optimization Variance: Dividing the bounding constant by $T + t _ { 0 }$ yields the variance term, which is upper-bounded by $\begin{array} { r } { \frac { v } { T } = \mathcal { O } \left( \frac { 1 } { ( 1 - \gamma ) ^ { 5 } \kappa ^ { 6 } \lambda ^ { 5 / 2 } \tau ^ { 2 } T } \right) } \end{array}$

• Approximation Bias: The bias term matches the average-iterate bound:

$$
\frac { c _ { \eta } C _ { \epsilon , \mathcal { L } } } { \rho _ { \mathcal { L } } c _ { \eta } - 1 } \epsilon _ { a p p } = \Theta \left( \frac { \epsilon _ { a p p } } { ( 1 - \gamma ) \lambda ^ { 2 } \tau } \right) .
$$

Because $\mathbb { E } [ \mathrm { G a p } _ { T } ] \leq \mathcal { L } _ { T }$ , summing the optimization variance and approximation bias yields the stated asymptotic last-iterate convergence rate. □

## E.6 Proofs for Section 7

Proof of Lemma 7.1 (Global Class Approximation Error). We mirror the structure of the PL Condition (Lemma 4.2) to bound the representation gap. Let $D _ { * } \triangleq \mathbb { E } _ { d _ { \tau } ^ { * } } [ D _ { K L } ( \pi _ { \tau } ^ { * } \| \pi _ { \omega _ { \tau } ^ { * } } ) ]$ denote the state-averaged forward KL divergence from the global optimum to the parametric optimum.

(i) Ideal Critic Decomposition. We apply the Regularized Suboptimality Decompositions (Lemma 3.3) evaluated at the training policy $\pi _ { t } ~ = ~ \pi _ { \omega _ { \tau } ^ { * } }$ against the comparator $\pi ~ = ~ \pi _ { \tau } ^ { * }$ . Let $\epsilon _ { * } ( s , a )$ denote the approximation error of the ideal critic $\theta _ { \tau } ^ { * } ( \omega _ { \tau } ^ { * } )$ trained under the distribution $d ^ { \omega _ { \tau } ^ { * } } \times \pi _ { \omega _ { \tau } ^ { * } }$ . The exact decomposition is:

$$
\begin{array} { r l } & { ( 1 - \gamma ) \big ( J _ { \tau } ( \pi _ { \tau } ^ { * } ) - J _ { \tau } ( \pi _ { \omega _ { \tau } ^ { * } } ) \big ) = - \tau D _ { * } + \mathbb { E } _ { d _ { \tau } ^ { * } } \Big [ \big \langle \big ( \theta _ { \tau } ^ { * } ( \omega _ { \tau } ^ { * } ) - \tau \omega _ { \tau } ^ { * } \big ) ^ { \top } \phi , \pi _ { \tau } ^ { * } - \pi _ { \omega _ { \tau } ^ { * } } \big \rangle A \Big ] } \\ & { \qquad + \mathbb { E } _ { d _ { \tau } ^ { * } } \big [ \langle \epsilon _ { * } , \pi _ { \tau } ^ { * } - \pi _ { \omega _ { \tau } ^ { * } } \rangle A \big ] . } \end{array}
$$

(ii) Bounding the Ideal Target Distance at Optimum. For any fixed temperature $\tau > 0$ , the regularized optimal parameter $\omega _ { \tau } ^ { * } \in \mathbb { R } ^ { d }$ is finite. As a finite, unconstrained maximizer of the parametric objective $J _ { \tau } ( \omega )$ , its policy gradient is zero: $\nabla J _ { \tau } ( \omega _ { \tau } ^ { * } ) = 0$ . Substituting this into the Uncentered Gradient Identity (Lemma 5.1), we have:

$$
0 = \bar { \Sigma } _ { c e n } ( \pi _ { \omega _ { \tau } ^ { * } } ) \big ( \theta _ { \tau } ^ { * } ( \omega _ { \tau } ^ { * } ) - \tau \omega _ { \tau } ^ { * } \big ) + E _ { b i a s } ^ { * } ,
$$

where $E _ { b i a s } ^ { * } \ \triangleq - \mathbb { E } _ { s \sim d ^ { \omega _ { \tau } ^ { * } } } \left[ \bar { \phi } _ { * } ( s ) \mathbb { E } _ { a \sim \pi _ { \omega _ { \tau } ^ { * } } } [ \epsilon _ { * } ( s , a ) \ | \ s ] \right]$ , with $\bar { \phi } _ { * } ( s ) \triangleq \mathbb { E } _ { a \sim \pi _ { \omega _ { \tau } ^ { * } } } [ \phi ( s , a ) ]$ . Because this covariance is evaluated at the parametric optimum, the Positive-Definite Fisher Information $( \mathrm { A s } -$ sumption 6) applies directly without any density shift: $\bar { \Sigma } _ { c e n } ( \pi _ { \omega _ { \tau } ^ { * } } ) \succeq \lambda I$ . Rearranging and bounding the target distance yields:

$$
\| { \theta } _ { \tau } ^ { * } ( \omega _ { \tau } ^ { * } ) - \tau \omega _ { \tau } ^ { * } \| _ { 2 } \leq \| \bar { \Sigma } _ { c e n } ( \pi _ { \omega _ { \tau } ^ { * } } ) ^ { - 1 } \| _ { 2 } \| E _ { b i a s } ^ { * } \| _ { 2 } \leq \frac { B _ { \phi } } { \lambda } \sqrt { \epsilon _ { a p p } } ,
$$

where the uniform bias bound is applied, $\lVert E _ { b i a s } ^ { * } \rVert _ { 2 } \leq B _ { \phi } \sqrt { \epsilon _ { a p p } } .$

(iii) Decoupling Inner Products. We apply H¨older’s inequality over the action space and Pinsker’s inequality $( \| \pi _ { \tau } ^ { * } - \pi _ { \omega _ { \tau } ^ { * } } \| _ { 1 } \leq \sqrt { 2 D _ { K L } ( \pi _ { \tau } ^ { * } \| \pi _ { \omega _ { \tau } ^ { * } } ) } )$ to the ideal algorithmic progress term:

$$
\begin{array} { r l } & { \langle ( \theta _ { \tau } ^ { * } ( \omega _ { \tau } ^ { * } ) - \tau \omega _ { \tau } ^ { * } ) ^ { \top } \phi , \pi _ { \tau } ^ { * } - \pi _ { \omega _ { \tau } ^ { * } } \rangle _ { A } \leq \operatorname* { m a x } _ { a } \big | ( \theta _ { \tau } ^ { * } ( \omega _ { \tau } ^ { * } ) - \tau \omega _ { \tau } ^ { * } ) ^ { \top } \phi ( s , a ) \big | \| \pi _ { \tau } ^ { * } - \pi _ { \omega _ { \tau } ^ { * } } \| _ { 1 } } \\ & { \qquad \leq B _ { \phi } \| \theta _ { \tau } ^ { * } ( \omega _ { \tau } ^ { * } ) - \tau \omega _ { \tau } ^ { * } \| _ { 2 } \sqrt { 2 D _ { K L } ( \pi _ { \tau } ^ { * } \| \pi _ { \omega _ { \tau } ^ { * } } ) } . } \end{array}
$$

Taking the expectation over $d _ { \tau } ^ { \ast } ( s )$ and applying Jensen’s Inequality $( \mathbb { E } _ { d _ { \tau } ^ { * } } [ \sqrt { D _ { K L } } ] \leq \sqrt { D _ { * } } )$ :

$$
\begin{array} { r } { \mathbb { E } _ { d _ { \tau } ^ { * } } \Big [ \langle ( \theta _ { \tau } ^ { * } ( \omega _ { \tau } ^ { * } ) - \tau \omega _ { \tau } ^ { * } ) ^ { \top } \phi , \pi _ { \tau } ^ { * } - \pi _ { \omega _ { \tau } ^ { * } } \rangle _ { \cal A } \Big ] \leq B _ { \phi } \| \theta _ { \tau } ^ { * } ( \omega _ { \tau } ^ { * } ) - \tau \omega _ { \tau } ^ { * } \| _ { 2 } \sqrt { 2 D _ { * } } . } \end{array}
$$

Applying Young’s Inequality $\begin{array} { r } { ( x y \le \frac { \tau } { 2 } x ^ { 2 } + \frac { 1 } { 2 \tau } y ^ { 2 } ) } \end{array}$ with $x = \sqrt { D _ { * } }$ yields:

$$
\mathbb { E } _ { d _ { \tau } ^ { * } } \Big [ \langle ( \theta _ { \tau } ^ { * } ( \omega _ { \tau } ^ { * } ) - \tau \omega _ { \tau } ^ { * } ) ^ { \top } \phi , \pi _ { \tau } ^ { * } - \pi _ { \omega _ { \tau } ^ { * } } \rangle _ { \cal A } \Big ] \leq \frac { \tau } { 2 } D _ { * } + \frac { B _ { \phi } ^ { 2 } } { \tau } \| \theta _ { \tau } ^ { * } ( \omega _ { \tau } ^ { * } ) - \tau \omega _ { \tau } ^ { * } \| _ { 2 } ^ { 2 } .
$$

For the approximation bias term, for any state $s ,$ define the local policy mismatch $W ^ { \ast } ( s ) \triangleq$

$\begin{array} { r } { \operatorname* { m a x } _ { a } \frac { \pi _ { \tau } ^ { * } ( a | s ) } { \pi _ { \omega _ { \tau } ^ { * } } ( a | s ) } } \end{array}$ . We apply Action-Space Cauchy-Schwarz and the $\chi ^ { 2 }$ bound (Lemma 4.1):

$$
\begin{array} { r } { \langle \epsilon _ { * } , \pi _ { \tau } ^ { * } - \pi _ { \omega _ { \tau } ^ { * } } \rangle _ { A } \leq \sqrt { 2 W ^ { * } ( s ) D _ { K L } ( \pi _ { \tau } ^ { * } ( \cdot | s ) \| \pi _ { \omega _ { \tau } ^ { * } } ( \cdot | s ) ) } \sqrt { \mathbb { E } _ { a \sim \pi _ { \omega _ { \tau } ^ { * } } } [ \epsilon _ { * } ( s , a ) ^ { 2 } ] } . } \end{array}
$$

Taking the expectation over $d _ { \tau } ^ { \ast } ( s )$ and applying State-Space Cauchy-Schwarz:

$$
\begin{array} { r l } & { \mathbb { E } _ { d _ { \tau } ^ { * } } \big [ \langle \epsilon _ { * } , \pi _ { \tau } ^ { * } - \pi _ { \omega _ { \tau } ^ { * } } \rangle _ { A } \big ] \leq \mathbb { E } _ { d _ { \tau } ^ { * } } \Big [ \sqrt { 2 D _ { K L } ( \pi _ { \tau } ^ { * } ( \cdot | s ) \| \pi _ { \omega _ { \tau } ^ { * } } ( \cdot | s ) ) } \sqrt { W ^ { * } ( s ) \mathbb { E } _ { a \sim \pi _ { \omega _ { \tau } ^ { * } } } [ \epsilon _ { * } ( s , a ) ^ { 2 } ] } \Big ] } \\ & { \quad \quad \quad \quad \leq \sqrt { 2 D _ { * } } \sqrt { \mathbb { E } _ { d _ { \tau } ^ { * } } \big [ W ^ { * } ( s ) \mathbb { E } _ { a \sim \pi _ { \omega _ { \tau } ^ { * } } } [ \epsilon _ { * } ^ { 2 } ] \big ] } . } \end{array}
$$

To control the second root, we expand the expectation over the state space and apply the joint concentrability bound $\begin{array} { r } { \frac { d _ { \tau } ^ { * } ( s ) } { d ^ { \omega _ { \tau } ^ { * } } ( s ) } W ^ { * } ( s ) = \operatorname* { m a x } _ { a } \frac { d _ { \tau } ^ { * } ( s ) \pi _ { \tau } ^ { * } ( a | s ) } { d ^ { \omega _ { \tau } ^ { * } } ( s ) \pi _ { \omega _ { \tau } ^ { * } } ( a | s ) } \le C _ { j o i n t } ^ { * } } \end{array}$ (Assumption 8):

$$
\begin{array} { r l } & { \mathbb { E } _ { d _ { \tau } ^ { * } } \left[ W ^ { * } ( s ) \mathbb { E } _ { a \sim \pi _ { \omega _ { \tau } ^ { * } } } [ \epsilon _ { * } ^ { 2 } ] \right] = \mathbb { E } _ { d ^ { \omega _ { \tau } ^ { * } } } \left[ \left( \frac { d _ { \tau } ^ { * } ( s ) } { d ^ { \omega _ { \tau } ^ { * } } ( s ) } W ^ { * } ( s ) \right) \mathbb { E } _ { a \sim \pi _ { \omega _ { \tau } ^ { * } } } [ \epsilon _ { * } ( s , a ) ^ { 2 } ] \right] } \\ & { \quad \quad \quad \quad \quad \leq C _ { j o i n t } ^ { * } \mathbb { E } _ { d ^ { \omega _ { \tau } ^ { * } } } \left[ \mathbb { E } _ { a \sim \pi _ { \omega _ { \tau } ^ { * } } } [ \epsilon _ { * } ( s , a ) ^ { 2 } ] \right] } \\ & { \quad \quad \quad \quad = C _ { j o i n t } ^ { * } \mathbb { E } _ { d ^ { \omega _ { \tau } ^ { * } } , \pi _ { \omega _ { \tau } ^ { * } } } [ \epsilon _ { * } ^ { 2 } ] \leq C _ { j o i n t } ^ { * } \epsilon _ { a p p } , } \end{array}
$$

where the last step uses Assumption 4. Applying Young’s Inequality $\begin{array} { r } { ( x y \le \frac { \tau } { 2 } x ^ { 2 } + \frac { 1 } { 2 \tau } y ^ { 2 } ) } \end{array}$ with $x = \sqrt { D _ { * } }$ and $y = \sqrt { 2 C _ { j o i n t } ^ { * } \epsilon _ { a p p } }$ yields:

$$
\mathbb { E } _ { d _ { \tau } ^ { * } } \big [ \langle \epsilon _ { * } , \pi _ { \tau } ^ { * } - \pi _ { \omega _ { \tau } ^ { * } } \rangle _ { \cal A } \big ] \leq \frac { \tau } { 2 } D _ { * } + \frac { C _ { j o i n t } ^ { * } } { \tau } \epsilon _ { a p p } .
$$

(iv) Synthesis. We substitute the decoupled algorithmic progress and approximation bias back into the suboptimality decomposition. The two positive ${ \frac { \tau } { 2 } } D _ { * }$ <sub>∗</sub> penalties cancel the restorative entropy force $- \tau D _ { * }$

$$
( 1 - \gamma ) \big ( J _ { \tau } ( \pi _ { \tau } ^ { * } ) - J _ { \tau } ( \pi _ { \omega _ { \tau } ^ { * } } ) \big ) \leq \frac { B _ { \phi } ^ { 2 } } { \tau } \| \theta _ { \tau } ^ { * } ( \omega _ { \tau } ^ { * } ) - \tau \omega _ { \tau } ^ { * } \| _ { 2 } ^ { 2 } + \frac { C _ { j o i n t } ^ { * } } { \tau } \epsilon _ { a p p } .
$$

Substituting the bound $\lVert \theta _ { \tau } ^ { * } ( \omega _ { \tau } ^ { * } ) - \tau \omega _ { \tau } ^ { * } \rVert _ { 2 } ^ { 2 } \leq \frac { B _ { \phi } ^ { 2 } } { \lambda ^ { 2 } } \epsilon _ { a p p }$ into the first term and dividing the entire inequality by $1 - \gamma$ yields the stated constant $C _ { c l a s s }$ , completing the proof. □

Proof of Corollary 7.1 (Global Regularized Performance Bound). This follows directly from the algebraic decomposition $J _ { \tau } ( \pi _ { \tau } ^ { * } ) - J _ { \tau } ( \pi _ { t } ) = \left( J _ { \tau } ( \pi _ { \tau } ^ { * } ) - J _ { \tau } ( \pi _ { \omega _ { \tau } ^ { * } } ) \right) + \left( J _ { \tau } ( \pi _ { \omega _ { \tau } ^ { * } } ) - J _ { \tau } ( \pi _ { t } ) \right)$ Substituting the bound from Lemma 7.1 for the first term and the definition of ${ \mathrm { G a p } } _ { t }$ for the second completes the proof. □

Proof of Lemma 7.2 (Part-I Approximation-Temperature Compatibility). (i) Lower-Bounding Suboptimal Mass. Define the suboptimal-action mass at state s by $q _ { \tau } ( s ) \triangleq 1 -$ $\pi _ { \omega _ { \tau } ^ { * } } ( a ^ { * } ( s ) | s )$ . We first show that the uniform Fisher lower bound forces $\pi _ { \omega _ { \tau } ^ { * } }$ to retain a nonvanishing suboptimal-action probability. By Assumption 6, $\bar { \Sigma } _ { c e n } ( \pi _ { \omega _ { \tau } ^ { * } } ) \succeq \lambda I$ , taking traces yields

$$
d \lambda \leq \mathbb { E } _ { s \sim d ^ { \omega _ { \tau } ^ { * } } } \left[ \operatorname { t r } { \mathrm { C o v } _ { a \sim \pi _ { \omega _ { \tau } ^ { * } } ( \cdot | s ) } ( \phi ( s , a ) ) } \right] .
$$

For any fixed state $s ,$ the expected squared distance from the mean is no larger than that from any fixed vector. Taking the feature vector of the optimal action $\phi ( s , a ^ { * } ( s ) )$ as this reference and using Assumption 1 gives

$$
\begin{array} { r l } & { \mathrm { t r } { \mathrm { C o v } _ { a \sim \pi _ { \omega _ { \tau } ^ { * } } ( \cdot | s ) } } ( \phi ( s , a ) ) = \mathbb { E } _ { a \sim \pi _ { \omega _ { \tau } ^ { * } } ( \cdot | s ) } \left[ \| \phi ( s , a ) - \mathbb { E } _ { \pi _ { \omega _ { \tau } ^ { * } } } [ \phi ( s , \cdot ) ] \| _ { 2 } ^ { 2 } \right] } \\ & { \quad \quad \quad \quad \quad \leq \mathbb { E } _ { a \sim \pi _ { \omega _ { \tau } ^ { * } } ( \cdot | s ) } \left[ \| \phi ( s , a ) - \phi ( s , a ^ { * } ( s ) ) \| _ { 2 } ^ { 2 } \right] } \\ & { \quad \quad \quad \quad \leq 4 B _ { \phi } ^ { 2 } q _ { \tau } ( s ) . } \end{array}
$$

Therefore,

$$
\mathbb { E } _ { s \sim d ^ { \omega _ { \tau } ^ { * } } } [ q _ { \tau } ( s ) ] \geq \frac { d \lambda } { 4 B _ { \phi } ^ { 2 } } \triangleq q _ { 0 } .
$$

(ii) Lower-Bounding Global-to-Parametric Gap. Next, by the Performance Diference Lemma for the unregularized global optimum and the Minimal Action Gap (Assumption 5),

$$
\begin{array} { l } { { \displaystyle { J _ { 0 } ( \pi _ { 0 } ^ { * } ) - J _ { 0 } ( \pi _ { \omega _ { \tau } ^ { * } } ) = \frac { 1 } { 1 - \gamma } \mathbb { E } _ { d ^ { \omega _ { \tau } ^ { * } } , \pi _ { \omega _ { \tau } ^ { * } } } \left[ V _ { 0 } ^ { * } ( s ) - Q _ { 0 } ^ { * } ( s , a ) \right] } } } \\ { { \displaystyle { \quad \geq \frac { \Delta } { 1 - \gamma } \mathbb { E } _ { d ^ { \omega _ { \tau } ^ { * } } } [ q _ { \tau } ( s ) ] \geq \frac { q _ { 0 } \Delta } { 1 - \gamma } } . } } \end{array}
$$

Because $\pi _ { \tau } ^ { * }$ is globally optimal for the regularized objective and $\pi _ { 0 } ^ { * }$ is deterministic,

$$
\begin{array} { r l } & { J _ { \tau } ( \pi _ { \tau } ^ { * } ) - J _ { \tau } ( \pi _ { \omega _ { \tau } ^ { * } } ) \geq J _ { \tau } ( \pi _ { 0 } ^ { * } ) - J _ { \tau } ( \pi _ { \omega _ { \tau } ^ { * } } ) } \\ & { \qquad = J _ { 0 } ( \pi _ { 0 } ^ { * } ) - J _ { 0 } ( \pi _ { \omega _ { \tau } ^ { * } } ) - \displaystyle \frac { \tau } { 1 - \gamma } \mathbb { E } _ { s \sim d \omega _ { \tau } ^ { * } } [ H ( \pi _ { \omega _ { \tau } ^ { * } } ( \cdot | s ) ) ] } \\ & { \qquad \geq \displaystyle \frac { q _ { 0 } \Delta - \tau \log | A | } { 1 - \gamma } . } \end{array}
$$

Hence, whenever τ log $| \mathcal { A } | \le q _ { 0 } \Delta / 2$

$$
J _ { \tau } ( \pi _ { \tau } ^ { * } ) - J _ { \tau } ( \pi _ { \omega _ { \tau } ^ { * } } ) \geq \frac { q _ { 0 } \Delta } { 2 ( 1 - \gamma ) } .
$$

(iii) Uniform Compatibility Bound. Define the reference temperature

$$
\bar { \tau } \triangleq \operatorname* { m i n } \left\{ \tau _ { m a x } , \frac { q _ { 0 } \Delta } { 2 \log | \mathcal { A } | } \right\} .
$$

By construction, $\tau \log | \mathcal { A } | \leq q _ { 0 } \Delta / 2$ . Applying the preceding lower bound at $\tau = \bar { \tau }$ gives

$$
J _ { \bar { \tau } } ( \pi _ { \bar { \tau } } ^ { * } ) - J _ { \bar { \tau } } ( \pi _ { \omega _ { \bar { \tau } } ^ { * } } ) \geq \frac { q _ { 0 } \Delta } { 2 ( 1 - \gamma ) } .
$$

On the other hand, Lemma 7.1 gives

$$
J _ { \bar { \tau } } ( \pi _ { \bar { \tau } } ^ { * } ) - J _ { \bar { \tau } } ( \pi _ { \omega _ { \bar { \tau } } ^ { * } } ) \leq \frac { C _ { c l a s s } } { \bar { \tau } } \epsilon _ { a p p } .
$$

Combining these inequalities yields

$$
\epsilon _ { a p p } \geq \frac { q _ { 0 } \Delta } { 2 ( 1 - \gamma ) C _ { c l a s s } } \bar { \tau } .
$$

Because $\epsilon _ { a p p }$ and the structural constants are uniform over $\tau \in ( 0 , \tau _ { m a x } ]$ , this bound obtained at the reference temperature $\bar { \tau }$ constrains the same quantities throughout this temperature range. For any $\tau \in ( 0 , \tau _ { m a x } ]$ , we have $\bar { \tau } \geq ( \bar { \tau } / \tau _ { m a x } ) \tau$ . Therefore,

$$
\epsilon _ { a p p } \geq \frac { q _ { 0 } \Delta } { 2 ( 1 - \gamma ) C _ { c l a s s } } \frac { \bar { \tau } } { \tau _ { m a x } } \tau = C _ { c o m p } \tau ,
$$

which completes the proof.

Proof of Theorem 7.1 (Unregularized Average-Iterate Convergence). We consider $\epsilon _ { a p p } > 0$ throughout, as required by the Part-I compatibility condition in Lemma 7.2.

(i) Global Translation Bound. Because the temperature $\tau _ { T }$ may initially exceed the structural threshold required in Theorem 3.1, we apply the Universal Unregularized Suboptimality Bound (Corollary 3.1). This guarantees the following inequality holds for any $\tau _ { T } > 0 $

$$
J _ { 0 } ( \pi _ { 0 } ^ { * } ) - J _ { 0 } ( \pi _ { t } ) \leq \frac { 4 A _ { m a x } } { \Delta } \mathrm { G a p } _ { t } ^ { \dag } ( \tau _ { T } ) + C _ { t a i l } \exp \left( - \frac { \Delta } { 2 \tau _ { T } } \right) .
$$

Taking the expectation and averaging this bound over the trajectory from $t = 0$ to $T - 1$ yields:

$$
\frac { 1 } { T } \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } \big [ J _ { 0 } ( \pi _ { 0 } ^ { * } ) - J _ { 0 } ( \pi _ { t } ) \big ] \leq \frac { 4 A _ { m a x } } { \Delta } \left( \frac { 1 } { T } \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } [ \mathrm { G a p } _ { t } ^ { \dag } ( \tau _ { T } ) ] \right) + C _ { t a i l } \exp \left( - \frac { \Delta } { 2 \tau _ { T } } \right) .
$$

(ii) Regularized Order Substitution. By the Global Regularized Performance Bound (Corollary 7.1), the expected global regularized gap is bounded by the parametric gap plus the representation penalty: $\begin{array} { r } { \mathbb { E } [ \mathrm { G a p } _ { t } ^ { \dagger } ] \leq \mathbb { E } [ \mathrm { G a p } _ { t } ] + \frac { C _ { c l a s s } } { \tau _ { T } } \epsilon _ { a p p } } \end{array}$ . We substitute the average parametric regularized performance gap from Theorem 6.2:

$$
\frac { 1 } { T } \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } [ \mathrm { G a p } _ { t } ^ { \dag } ] \leq \underbrace { \mathcal { O } \left( \frac { 1 } { { ( 1 - \gamma ) } ^ { 5 } \kappa ^ { 6 } \lambda ^ { 5 / 2 } \tau _ { T } ^ { 2 } } \frac { \log T } { T } \right) } _ { \mathrm { F i n i t e - T i m e ~ O p t i m i z a t i o n ~ E r r o r } } + \underbrace { \mathcal { O } \left( \frac { \epsilon _ { a p p } } { ( 1 - \gamma ) \lambda ^ { 2 } \tau _ { T } } \right) + \frac { C _ { c l a s s } } { \tau _ { T } } \epsilon _ { a p p } } _ { \mathrm { T o t a l ~ A p p r o x i m a t i o n ~ B i a s } } .
$$

Because $\begin{array} { r } { C _ { c l a s s } \triangleq \frac { B _ { \phi } ^ { 4 } } { ( 1 - \gamma ) \lambda ^ { 2 } } + \frac { C _ { j o i n t } ^ { * } } { 1 - \gamma } = \mathcal { O } \big ( \frac { 1 } { ( 1 - \gamma ) \lambda ^ { 2 } } \big ) } \end{array}$ , the representation penalty is absorbed into the approximation bias order. Feeding this into the translation bound isolates the three competing components:

$$
\mathrm { T o t a l ~ E r r o r } \le \underbrace { \mathcal { O } \left( \frac { A _ { m a x } } { \Delta ( 1 - \gamma ) ^ { 5 } \kappa ^ { 6 } \lambda ^ { 5 / 2 } \tau _ { T } ^ { 2 } } \frac { \log { T } } { T } \right) } _ { \mathrm { F i n i t e - T i m e ~ O p t i m i z a t i o n ~ E r r o r } } + \underbrace { \mathcal { O } \left( \frac { A _ { m a x } \epsilon _ { a p p } } { \Delta ( 1 - \gamma ) \lambda ^ { 2 } \tau _ { T } } \right) } _ { \mathrm { T o t a l ~ A p p r o x i m a t i o n ~ B i a s } } + \underbrace { C _ { t a i l } \exp { \left( - \frac { \Delta } { 2 \tau _ { T } } \right) } } _ { \mathrm { E n t r o p y ~ T a i l } } .
$$

(iii) Regime 1: Sample-Limited Phase. When log $T \leq \log ( 1 + \epsilon _ { a p p } ^ { - 1 } )$ , the temperature is tuned to $\begin{array} { r } { \tau _ { T } = \frac { \Delta } { 2 \log ( C _ { \gamma } T ) } } \end{array}$ . Substituting this alongside $\begin{array} { r } { A _ { m a x } \leq \frac { 1 } { 1 - \gamma } , C _ { t a i l } \triangleq \frac { A _ { m a x } C _ { \gamma } } { 1 - \gamma } \leq \frac { C _ { \gamma } } { ( 1 - \gamma ) ^ { 2 } } } \end{array}$ , and log $C _ { \gamma } =$

$\begin{array} { r } { \mathcal { O } ( \frac { 1 } { 1 - \gamma } ) } \end{array}$ into the bounded components yields:

$$
\begin{array} { r l } & { \mathrm { F i n i t e - T i m e ~ E r r o r } = \displaystyle \mathcal { O } \Big ( \frac { 1 } { ( 1 - \gamma ) ^ { 6 } \kappa ^ { 6 } \lambda ^ { 5 / 2 } \Delta ^ { 3 } } \big ( \log ( C _ { \gamma } T ) \big ) ^ { 2 } ( \log T ) T ^ { - 1 } \Big ) } \\ & { \quad \quad \quad \quad = \mathcal { O } \Big ( \frac { 1 } { ( 1 - \gamma ) ^ { 6 } \kappa ^ { 6 } \lambda ^ { 5 / 2 } \Delta ^ { 3 } } \Big ( \frac { \log T } { ( 1 - \gamma ) ^ { 2 } } + ( \log T ) ^ { 3 } \Big ) T ^ { - 1 } \Big ) , } \end{array}
$$

$$
\begin{array} { r l } & { \mathrm { A p p r o x i m a t i o n ~ B i a s } = \displaystyle { \mathcal O } \Big ( \frac { 1 } { ( 1 - \gamma ) ^ { 2 } \lambda ^ { 2 } \Delta ^ { 2 } } \log ( C _ { \gamma } T ) \epsilon _ { a p p } \Big ) } \\ & { \qquad \leq \displaystyle { \mathcal O } \Big ( \frac { 1 } { ( 1 - \gamma ) ^ { 2 } \lambda ^ { 2 } \Delta ^ { 2 } } \Big ( \frac { 1 } { 1 - \gamma } + \log ( 1 + \epsilon _ { a p p } ^ { - 1 } ) \Big ) \epsilon _ { a p p } \Big ) , } \\ & { \mathrm { E n t r o p y ~ T a i l } \leq \displaystyle \frac { C _ { \gamma } } { ( 1 - \gamma ) ^ { 2 } } \exp \big ( - \log ( C _ { \gamma } T ) \big ) = \frac { 1 } { ( 1 - \gamma ) ^ { 2 } T } . } \end{array}
$$

The regime condition log $T \leq \log ( 1 + \epsilon _ { a p p } ^ { - 1 } )$ guarantees

$$
\log ( C _ { \gamma } T ) = \log C _ { \gamma } + \log T \le \log C _ { \gamma } + \log ( 1 + \epsilon _ { a p p } ^ { - 1 } ) ,
$$

which gives the stated approximation-bias envelope. The $C _ { \gamma }$ prefactor cancels in the exponential tail, yielding a pure $\mathcal { O } ( T ^ { - 1 } )$ tail that is absorbed by the dominant $\tilde { \mathcal { O } } ( T ^ { - 1 } )$ optimization error.

(iv) Regime 2: Approximation-Limited Phase. When log $T > \log ( 1 + \epsilon _ { a p p } ^ { - 1 } )$ , the temperature locks to the approximation-dependent floor: $\begin{array} { r } { \tau _ { T } = \frac { \Delta } { 2 \log ( C _ { \gamma } ( 1 + \epsilon _ { a p p } ^ { - 1 } ) ) } } \end{array}$ . Substituting this fixed temperature into the bounded components yields:

$$
\begin{array} { r l } & { \mathrm { F i n i t e - T i m e ~ E r r o r } = \mathcal { O } \Big ( \frac { 1 } { ( 1 - \gamma ) ^ { 6 } \kappa ^ { 6 } \lambda ^ { 5 / 2 } \Delta ^ { 3 } } \big ( \log ( C _ { \gamma } ( 1 + \epsilon _ { a p p } ^ { - 1 } ) ) \big ) ^ { 2 } ( \log T ) T ^ { - 1 } \Big ) } \\ & { \qquad \leq \mathcal { O } \Big ( \frac { 1 } { ( 1 - \gamma ) ^ { 6 } \kappa ^ { 6 } \lambda ^ { 5 / 2 } \Delta ^ { 3 } } \big ( \frac { \log T } { ( 1 - \gamma ) ^ { 2 } } + ( \log T ) ^ { 3 } \big ) T ^ { - 1 } \Big ) , } \\ & { \mathrm { A p p r o x i m a t i o n ~ B i a s } = \mathcal { O } \Big ( \frac { 1 } { ( 1 - \gamma ) ^ { 2 } \lambda ^ { 2 } \Delta ^ { 2 } } \log ( C _ { \gamma } ( 1 + \epsilon _ { a p p } ^ { - 1 } ) ) \epsilon _ { a p p } \Big ) } \\ & { \qquad = \mathcal { O } \Big ( \frac { 1 } { ( 1 - \gamma ) ^ { 2 } \lambda ^ { 2 } \Delta ^ { 2 } } \Big ( \frac { 1 } { 1 - \gamma } + \log ( 1 + \epsilon _ { a p p } ^ { - 1 } ) \Big ) \epsilon _ { a p p } \Big ) , } \\ & { \mathrm { E n t r o p y ~ T a i l } \leq \frac { C _ { \gamma } } { ( 1 - \gamma ) ^ { 2 } } \exp \big ( - \log ( C _ { \gamma } ( 1 + \epsilon _ { a p p } ^ { - 1 } ) ) - \frac { \epsilon _ { a p p } / ( 1 + \epsilon _ { a p p } ) } { ( 1 - \gamma ) ^ { 2 } } \leq \frac { \epsilon _ { a p p } } { ( 1 - \gamma ) ^ { 2 } } . } \end{array}
$$

For the finite-time error, the regime condition $\log ( 1 + \epsilon _ { a p p } ^ { - 1 } ) < \log T$ implies

$$
\log ( C _ { \gamma } ( 1 + \epsilon _ { a p p } ^ { - 1 } ) ) \le \log C _ { \gamma } + \log T ,
$$

which yields the stated finite-time envelope. The fixed temperature floor prevents the $\tau _ { T } ^ { - 1 }$ approximation bias from growing with $T ,$ , leading to the $\tilde { \mathcal { O } } ( \epsilon _ { a p p } )$ rate. Simultaneously, the exponential entropy tail is bounded by the linear scale of the approximation error, yielding the $\mathcal { O } ( \epsilon _ { a p p } )$ rate.

(v) Synthesis. Summing these evaluated components across the two regimes bounds the overall expected global unregularized performance gap by their respective envelopes, yielding:

$$
\frac { 1 } { T } \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } \big [ J _ { 0 } ( \pi _ { 0 } ^ { * } ) - J _ { 0 } ( \pi _ { t } ) \big ] \leq \mathcal { O } \left( \frac { ( \log T ) / ( 1 - \gamma ) ^ { 2 } + ( \log T ) ^ { 3 } } { ( 1 - \gamma ) ^ { 6 } \kappa ^ { 6 } \lambda ^ { 5 / 2 } \Delta ^ { 3 } T } \right)
$$

$$
+ O \left( \frac { 1 / ( 1 - \gamma ) + \log ( 1 + \epsilon _ { a p p } ^ { - 1 } ) } { ( 1 - \gamma ) ^ { 2 } \lambda ^ { 2 } \Delta ^ { 2 } } \epsilon _ { a p p } \right) .
$$

The approximation term is proportional to $\epsilon _ { a p p }$ up to the logarithmic factor $\log ( 1 + \epsilon _ { a p p } ^ { - 1 } )$ . Thus, subject to the admissible range of $\epsilon _ { a p p }$ in Remark 7.1, the overall convergence rate can be summarized as $\tilde { \mathcal { O } } ( T ^ { - 1 } ) + \tilde { \mathcal { O } } ( \epsilon _ { a p p } )$ , completing the proof. □

Proof of Theorem 7.2 (Unregularized Last-Iterate Convergence). We consider $\epsilon _ { a p p } > 0$ throughout, as required by the Part-I compatibility condition in Lemma 7.2.

(i) Global Translation Bound. We apply the Universal Unregularized Suboptimality Bound (Corollary 3.1) directly to the terminal policy $\pi _ { T }$ . This guarantees the following inequality holds for any $\tau _ { T } > 0$

$$
J _ { 0 } ( \pi _ { 0 } ^ { * } ) - J _ { 0 } ( \pi _ { T } ) \leq \frac { 4 A _ { m a x } } { \Delta } \mathrm { G a p } _ { T } ^ { \dag } ( \tau _ { T } ) + C _ { t a i l } \exp \left( - \frac { \Delta } { 2 \tau _ { T } } \right) .
$$

Taking the expectation bounds the last-iterate global unregularized performance gap:

$$
\mathbb { E } \big [ J _ { 0 } ( \pi _ { 0 } ^ { * } ) - J _ { 0 } ( \pi _ { T } ) \big ] \leq \frac { 4 A _ { m a x } } { \Delta } \mathbb { E } [ \mathrm { G a p } _ { T } ^ { \dag } ( \tau _ { T } ) ] + C _ { t a i l } \exp \left( - \frac { \Delta } { 2 \tau _ { T } } \right) .
$$

(ii) Regularized Order Substitution. By the Global Regularized Performance Bound (Corollary 7.1), the expected global regularized gap is bounded by the parametric gap plus the representation penalty: $\begin{array} { r } { \mathbb { E } [ \mathrm { G a p } _ { T } ^ { \dagger } ] \leq \mathbb { E } [ \mathrm { G a p } _ { T } ] + \frac { C _ { c l a s s } } { \tau _ { T } } \epsilon _ { a p p } } \end{array}$ . We substitute the last-iterate parametric regularized performance gap from Theorem 6.3:

$$
\mathbb { E } [ \mathrm { G a p } _ { T } ^ { \dagger } ] \leq \underbrace { \mathcal { O } \left( \frac { 1 } { ( 1 - \gamma ) ^ { 5 } \kappa ^ { 6 } \lambda ^ { 5 / 2 } \tau _ { T } ^ { 2 } T } \right) } _ { \mathrm { F i n i t e - T i m e ~ O p t i m i z a t i o n ~ E r r o r ~ } } + \underbrace { \mathcal { O } \left( \frac { \epsilon _ { a p p } } { ( 1 - \gamma ) \lambda ^ { 2 } \tau _ { T } } \right) + \frac { C _ { c l a s s } } { \tau _ { T } } \epsilon _ { a p p } } _ { \mathrm { T o t a l ~ A p p r o x i m a t i o n ~ B i a s } } .
$$

Because $\begin{array} { r } { C _ { c l a s s } = \mathcal { O } \big ( \frac { 1 } { ( 1 - \gamma ) \lambda ^ { 2 } } \big ) } \end{array}$ , the representation penalty is absorbed into the approximation bias order. Feeding this into the translation bound isolates the three competing components:

$$
\mathrm { T o t a l ~ E r r o r } \leq \underbrace { \mathcal { O } \left( \frac { A _ { m a x } } { \Delta ( 1 - \gamma ) ^ { 5 } \kappa ^ { 6 } \lambda ^ { 5 / 2 } \tau _ { T } ^ { 2 } T } \right) } _ { \mathrm { F i n i t e - T i m e ~ O p t i m i z a t i o n ~ E r r o r } } + \underbrace { \mathcal { O } \left( \frac { A _ { m a x } \epsilon _ { a p p } } { \Delta ( 1 - \gamma ) \lambda ^ { 2 } \tau _ { T } } \right) } _ { \mathrm { T o t a l ~ A p p r o x i m a t i o n ~ B i a s } } + \underbrace { C _ { t a i l } \exp \left( - \frac { \Delta } { 2 \tau _ { T } } \right) } _ { \mathrm { E n t r o p y ~ T a i l } } .
$$

(iii) Regime 1: Sample-Limited Phase. When log $T \leq \log ( 1 + \epsilon _ { a p p } ^ { - 1 } )$ , the temperature is tuned to $\begin{array} { r } { \tau _ { T } = \frac { \Delta } { 2 \log ( C _ { \gamma } T ) } } \end{array}$ . Substituting this alongside $\begin{array} { r } { A _ { m a x } \leq \frac { 1 } { 1 - \gamma } , C _ { t a i l } \triangleq \frac { A _ { m a x } C _ { \gamma } } { 1 - \gamma } \leq \frac { C _ { \gamma } } { ( 1 - \gamma ) ^ { 2 } } } \end{array}$ , and log $C _ { \gamma } =$ $\begin{array} { r } { \mathcal { O } ( \frac { 1 } { 1 - \gamma } ) } \end{array}$ into the bounded components yields:

$$
\begin{array} { r l } & { \mathrm { F i n i t e - T i m e ~ E r r o r } = \displaystyle \mathcal { O } \Big ( \frac { 1 } { ( 1 - \gamma ) ^ { 6 } \kappa ^ { 6 } \lambda ^ { 5 / 2 } \Delta ^ { 3 } } \big ( \log ( C _ { \gamma } T ) \big ) ^ { 2 } T ^ { - 1 } \Big ) } \\ & { \quad \quad \quad \quad = \mathcal { O } \Big ( \frac { 1 } { ( 1 - \gamma ) ^ { 6 } \kappa ^ { 6 } \lambda ^ { 5 / 2 } \Delta ^ { 3 } } \Big ( \frac { 1 } { ( 1 - \gamma ) ^ { 2 } } + ( \log T ) ^ { 2 } \Big ) T ^ { - 1 } \Big ) , } \end{array}
$$

$$
\begin{array} { r l } & { \mathrm { A p p r o x i m a t i o n ~ B i a s } = \displaystyle { \mathcal O } \Big ( \frac { 1 } { ( 1 - \gamma ) ^ { 2 } \lambda ^ { 2 } \Delta ^ { 2 } } \log ( C _ { \gamma } T ) \epsilon _ { a p p } \Big ) } \\ & { \qquad \leq \displaystyle { \mathcal O } \Big ( \frac { 1 } { ( 1 - \gamma ) ^ { 2 } \lambda ^ { 2 } \Delta ^ { 2 } } \Big ( \frac { 1 } { 1 - \gamma } + \log ( 1 + \epsilon _ { a p p } ^ { - 1 } ) \Big ) \epsilon _ { a p p } \Big ) , } \end{array}
$$

$$
{ \mathrm { E n t r o p y ~ T a i l } } \leq { \frac { C _ { \gamma } } { ( 1 - \gamma ) ^ { 2 } } } \exp { \big ( } - \log ( C _ { \gamma } T ) { \big ) } = { \frac { 1 } { ( 1 - \gamma ) ^ { 2 } T } } .
$$

The regime condition log $T \leq \log ( 1 + \epsilon _ { a p p } ^ { - 1 } )$ guarantees

$$
\log ( C _ { \gamma } T ) = \log C _ { \gamma } + \log T \le \log C _ { \gamma } + \log ( 1 + \epsilon _ { a p p } ^ { - 1 } ) ,
$$

which gives the stated approximation-bias envelope. The $C _ { \gamma }$ prefactor cancels in the exponential tail, which scales as $\mathcal { O } ( T ^ { - 1 } )$ and is absorbed by the dominant $\tilde { \mathcal { O } } ( T ^ { - 1 } )$ optimization error. Notably, the final optimization error is sharper than the average-iterate bound by a factor of log T.

(iv) Regime 2: Approximation-Limited Phase. When log $T > \log ( 1 + \epsilon _ { a p p } ^ { - 1 } )$ , the temperature locks to the approximation-dependent floor: $\begin{array} { r } { \tau _ { T } = \frac { \Delta } { 2 \log ( C _ { \gamma } ( 1 + \epsilon _ { a p p } ^ { - 1 } ) ) } } \end{array}$ . Substituting this fixed temperature into the bounded components yields:

$$
\begin{array} { r l } & { \mathrm { F i n i t e - T i m e ~ E r r o r } = \displaystyle \mathcal { O } \Big ( \frac { 1 } { ( 1 - \gamma ) ^ { 6 } \kappa ^ { 6 } \lambda ^ { 5 / 2 } \Delta ^ { 3 } } \big ( \log ( C _ { \gamma } ( 1 + \epsilon _ { a p p } ^ { - 1 } ) ) \big ) ^ { 2 } T ^ { - 1 } \Big ) } \\ & { \quad \quad \quad \quad \leq \mathcal { O } \Big ( \frac { 1 } { ( 1 - \gamma ) ^ { 6 } \kappa ^ { 6 } \lambda ^ { 5 / 2 } \Delta ^ { 3 } } \Big ( \frac { 1 } { ( 1 - \gamma ) ^ { 2 } } + ( \log T ) ^ { 2 } \Big ) T ^ { - 1 } \Big ) , } \end{array}
$$

$$
\begin{array} { r l } & { \mathrm { A p p r o x i m a t i o n ~ B i a s } = \mathcal { O } \Big ( \frac { 1 } { ( 1 - \gamma ) ^ { 2 } \lambda ^ { 2 } \Delta ^ { 2 } } \log ( C _ { \gamma } ( 1 + \epsilon _ { a p p } ^ { - 1 } ) ) \epsilon _ { a p p } \Big ) } \\ & { \qquad = \mathcal { O } \Big ( \frac { 1 } { ( 1 - \gamma ) ^ { 2 } \lambda ^ { 2 } \Delta ^ { 2 } } \Big ( \frac { 1 } { 1 - \gamma } + \log ( 1 + \epsilon _ { a p p } ^ { - 1 } ) \Big ) \epsilon _ { a p p } \Big ) , } \\ & { \mathrm { E n t r o p y ~ T a i l } \le \frac { C _ { \gamma } } { ( 1 - \gamma ) ^ { 2 } } \exp \big ( - \log ( C _ { \gamma } ( 1 + \epsilon _ { a p p } ^ { - 1 } ) ) \big ) = \frac { \epsilon _ { a p p } / ( 1 + \epsilon _ { a p p } ) } { ( 1 - \gamma ) ^ { 2 } } \le \frac { \epsilon _ { a p p } } { ( 1 - \gamma ) ^ { 2 } } . } \end{array}
$$

For the finite-time error, the regime condition $\log ( 1 + \epsilon _ { a p p } ^ { - 1 } ) < \log T$ implies

$$
\log ( C _ { \gamma } ( 1 + \epsilon _ { a p p } ^ { - 1 } ) ) \le \log C _ { \gamma } + \log T ,
$$

which yields the stated finite-time envelope. The fixed temperature floor prevents the $\tau _ { T } ^ { - 1 }$ approximation bias from growing with $T ,$ , preserving the $\tilde { \mathcal { O } } ( \epsilon _ { a p p } )$ rate. Simultaneously, the exponential entropy tail is bounded by the linear scale of the approximation error, yielding the $\mathcal { O } ( \epsilon _ { a p p } )$ rate.

(v) Synthesis. Summing these evaluated components across the two regimes bounds the expected last-iterate global unregularized performance gap by their respective envelopes, yielding:

$$
\begin{array} { r l } & { \mathbb { E } \left[ J _ { 0 } ( \pi _ { 0 } ^ { * } ) - J _ { 0 } ( \pi _ { T } ) \right] \leq { \mathcal O } \left( \displaystyle \frac { ( 1 - \gamma ) ^ { - 2 } + ( \log T ) ^ { 2 } } { ( 1 - \gamma ) ^ { 6 } \kappa ^ { 6 } \lambda ^ { 5 / 2 } \Delta ^ { 3 } T } \right) } \\ & { \quad \quad \quad \quad \quad \quad + { \mathcal O } \left( \displaystyle \frac { ( 1 - \gamma ) ^ { - 1 } + \log ( 1 + \epsilon _ { a p p } ^ { - 1 } ) } { ( 1 - \gamma ) ^ { 2 } \lambda ^ { 2 } \Delta ^ { 2 } } \epsilon _ { a p p } \right) . } \end{array}
$$

The approximation term is proportional to $\epsilon _ { a p p } ~ \mathrm { u p }$ to the logarithmic factor $\log ( 1 + \epsilon _ { a p p } ^ { - 1 } )$ . Thus,

subject to the admissible range of $\epsilon _ { a p p }$ in Remark 7.1, the overall convergence rate can be summarized as $\tilde { \mathcal { O } } ( T ^ { - 1 } ) + \tilde { \mathcal { O } } ( \epsilon _ { a p p } )$ , completing the proof. □

## E.7 Proofs for Section 8

Proof of Lemma 8.1 (PMD Algorithmic Progress). (i) Bregman Identity. The standard Bregman Three-Point identity over the probability simplex dictates that for any state s:

$$
D _ { K L } ( \pi _ { \tau } ^ { * } | | \pi _ { t + 1 } ) = D _ { K L } ( \pi _ { \tau } ^ { * } | | \pi _ { t } ) + D _ { K L } ( \pi _ { t } | | \pi _ { t + 1 } ) - \langle \log \pi _ { t + 1 } - \log \pi _ { t } , \pi _ { \tau } ^ { * } - \pi _ { t } \rangle _ { \cal A } .
$$

By substituting the log-linear policy parameterization, the log-probability update is log $\pi _ { t + 1 } ( a | s )$ log $\begin{array} { r } { \pi _ { t } ( { a } | s ) = \eta _ { t } ( g _ { t } ^ { a c } ) ^ { \top } \phi ( s , a ) - \log \frac { Z _ { t + 1 } ( s ) } { Z _ { t } ( s ) } } \end{array}$ . Substituting this into the inner product eliminates the state-dependent normalizer, as both policies sum to 1 over the action space:

$$
\begin{array} { r l } & { \langle \log \pi _ { t + 1 } - \log \pi _ { t } , \pi _ { \tau } ^ { * } - \pi _ { t } \rangle _ { A } = \displaystyle \sum _ { a } \left( \pi _ { \tau } ^ { * } ( a | s ) - \pi _ { t } ( a | s ) \right) \left( \eta _ { t } ( g _ { t } ^ { a c } ) ^ { \top } \phi ( s , a ) - \log \frac { Z _ { t + 1 } ( s ) } { Z _ { t } ( s ) } \right) } \\ & { \qquad = \eta _ { t } \langle ( g _ { t } ^ { a c } ) ^ { \top } \phi , \pi _ { \tau } ^ { * } - \pi _ { t } \rangle _ { A } . } \end{array}
$$

Rearranging the Bregman identity isolates the scaled algorithmic progress:

$$
\begin{array} { r } { \eta _ { t } \langle ( g _ { t } ^ { a c } ) ^ { \top } \phi , \pi _ { \tau } ^ { * } - \pi _ { t } \rangle _ { A } = D _ { K L } ( \pi _ { \tau } ^ { * } \| \pi _ { t } ) - D _ { K L } ( \pi _ { \tau } ^ { * } \| \pi _ { t + 1 } ) + D _ { K L } ( \pi _ { t } \| \pi _ { t + 1 } ) . } \end{array}
$$

(ii) Local Step Distance. To establish the lemma, it sufices to upper-bound the local step distance $D _ { K L } ( \pi _ { t } \Vert \pi _ { t + 1 } )$ . By definition of the forward KL divergence between successive policies:

$$
D _ { K L } ( \pi _ { t } \| \pi _ { t + 1 } ) = \sum _ { a } \pi _ { t } ( a | s ) \big ( \log \pi _ { t } ( a | s ) - \log \pi _ { t + 1 } ( a | s ) \big ) .
$$

Substituting the exact log-probability update yields:

$$
D _ { K L } ( \pi _ { t } \| \pi _ { t + 1 } ) = - \eta _ { t } \langle ( g _ { t } ^ { a c } ) ^ { \top } \phi , \pi _ { t } \rangle _ { \cal A } + \log \frac { Z _ { t + 1 } ( s ) } { Z _ { t } ( s ) } .
$$

(iii) Hoefding’s Bound. We express the partition function ratio as an expectation over the current policy π<sub>t</sub>:

$$
\frac { Z _ { t + 1 } ( s ) } { Z _ { t } ( s ) } = \sum _ { a } \pi _ { t } ( a | s ) \exp \left( \eta _ { t } ( g _ { t } ^ { a c } ) ^ { \top } \phi ( s , a ) \right) = \mathbb { E } _ { a \sim \pi _ { t } } \big [ \exp \left( \eta _ { t } ( g _ { t } ^ { a c } ) ^ { \top } \phi ( s , a ) \right) \big ] .
$$

Because the step size satisfies $\eta _ { t } \leq 1 / \tau$ , Lemma 2.2(ii) guarantees that the actor update direction is bounded under Assumptions 1 and 3. Then the functional update direction is bounded by a temperature-independent constant: $| ( g _ { t } ^ { a c } ) ^ { \top } \phi ( s , a ) | \le \| g _ { t } ^ { a c } \| _ { 2 } \| \phi ( s , a ) \| _ { 2 } \le G _ { a c } B _ { \phi }$ . Therefore, we apply Hoefding’s Lemma to bound the log-moment generating function:

$$
\log \frac { Z _ { t + 1 } ( s ) } { Z _ { t } ( s ) } \leq \eta _ { t } \mathbb { E } _ { a \sim \pi _ { t } } \big [ ( g _ { t } ^ { a c } ) ^ { \top } \phi ( s , a ) \big ] + \frac { \eta _ { t } ^ { 2 } ( 2 G _ { a c } B _ { \phi } ) ^ { 2 } } { 8 } = \eta _ { t } \langle ( g _ { t } ^ { a c } ) ^ { \top } \phi , \pi _ { t } \rangle _ { A } + \frac { \eta _ { t } ^ { 2 } } { 2 } G _ { a c } ^ { 2 } B _ { \phi } ^ { 2 } .
$$

(iv) Synthesis. Substituting this log-partition bound back into Step (ii) cancels the inner product, bounding the local step distance purely by the worst-case update magnitude (the PMD Local Error):

$$
D _ { K L } ( \pi _ { t } \| \pi _ { t + 1 } ) \leq - \eta _ { t } \langle ( g _ { t } ^ { a c } ) ^ { \top } \phi , \pi _ { t } \rangle _ { \cal A } + \left( \eta _ { t } \langle ( g _ { t } ^ { a c } ) ^ { \top } \phi , \pi _ { t } \rangle _ { \cal A } + \frac { \eta _ { t } ^ { 2 } } { 2 } G _ { a c } ^ { 2 } B _ { \phi } ^ { 2 } \right) = \frac { \eta _ { t } ^ { 2 } } { 2 } G _ { a c } ^ { 2 } B _ { \phi } ^ { 2 } .
$$

Substituting this upper bound into the Bregman identity from Step (i) and dividing by $\eta _ { t }$ completes the proof. □

## E.8 Proofs for Section 9

Proof of Lemma 9.1 (PMD Actor Progress Bound). By the Actual Critic Decomposition with $\pi = \pi _ { \tau } ^ { * }$ in Lemma 3.3(ii), the gap evaluates via the PMD geometry to:

$$
\begin{array} { r l } & { ( 1 - \gamma ) \mathrm { G a p } _ { t } ^ { \dagger } = - \tau D _ { t } ^ { \dagger } + \underbrace { { \mathbb { E } } _ { d ^ { \pi _ { \tau } ^ { * } } } \left[ \langle ( \theta _ { t + 1 } - \tau \omega _ { t } ) ^ { \top } \phi , \pi _ { \tau } ^ { * } - \pi _ { t } \rangle _ { \cal A } \right] } _ { \mathrm { A l g o r i t h m i c ~ P r o g r e s s } } } \\ & { ~ + \underbrace { { \mathbb { E } } _ { d ^ { \pi _ { \tau } ^ { * } } } \left[ \langle - e _ { t } ^ { \top } \phi , \pi _ { \tau } ^ { * } - \pi _ { t } \rangle _ { \cal A } \right] } _ { \mathrm { C r i t i c ~ T r a c k i n g ~ E r r o r } } + \underbrace { { \mathbb { E } } _ { d ^ { \pi _ { \tau } ^ { * } } } \left[ \langle \epsilon _ { t } , \pi _ { \tau } ^ { * } - \pi _ { t } \rangle _ { \cal A } \right] } _ { \mathrm { A p p r o x i m a t i o n ~ B i a s } } . } \end{array}
$$

We analyze the three components on the right-hand side independently.

(i) PMD Algorithmic Progress. Because $\eta _ { t } \leq 1 / \tau$ , Lemma 8.1 applies to bound the progress inner product driven by the log-linear update, yielding:

$$
\mathbb { E } _ { d _ { \tau } ^ { * } } \Big [ \langle ( \theta _ { t + 1 } - \tau \omega _ { t } ) ^ { \top } \phi , \pi _ { \tau } ^ { * } - \pi _ { t } \rangle _ { \cal A } \Big ] \leq \frac { D _ { t } ^ { \dagger } - D _ { t + 1 } ^ { \dagger } } { \eta _ { t } } + \frac { \eta _ { t } } { 2 } G _ { a c } ^ { 2 } B _ { \phi } ^ { 2 } .
$$

Substituting this into the gap decomposition gives:

$$
( 1 - \gamma ) { \mathrm { G a p } } _ { t } ^ { \dagger } \leq - \tau D _ { t } ^ { \dagger } + \frac { D _ { t } ^ { \dagger } - D _ { t + 1 } ^ { \dagger } } { \eta _ { t } } + \frac { \eta _ { t } } { 2 } G _ { a c } ^ { 2 } B _ { \phi } ^ { 2 } + \mathbb { E } _ { d _ { \tau } ^ { * } } \Big [ \langle - e _ { t } ^ { \top } \phi + \epsilon _ { t } , \pi _ { \tau } ^ { * } - \pi _ { t } \rangle _ { A } \Big ] .\tag{23}
$$

(ii) Approximation Bias Transfer. For any state $s ,$ define the local policy mismatch $W ^ { \dagger } ( s ) \triangleq$ max<sub>a</sub> $\frac { \pi _ { \tau } ^ { * } ( a | s ) } { \pi _ { t } ( a | s ) }$ . By Action-Space Cauchy-Schwarz and the $\chi ^ { 2 }$ bound (Lemma 4.1), the approximation bias satisfies:

$$
\begin{array} { r } { \langle \epsilon _ { t } , \pi _ { \tau } ^ { * } - \pi _ { t } \rangle _ { A } \leq \sqrt { 2 W ^ { \dagger } ( s ) D _ { K L } \bigl ( \pi _ { \tau } ^ { * } ( \cdot | s ) \| \pi _ { t } ( \cdot | s ) \bigr ) } \sqrt { \mathbb { E } _ { a \sim \pi _ { t } } \bigl [ \epsilon _ { t } ( s , a ) ^ { 2 } \bigr ] } . } \end{array}
$$

Taking the expectation over $d _ { \tau } ^ { \ast } ( s )$ and applying State-Space Cauchy-Schwarz gives:

$$
\begin{array} { r l } & { \mathbb { E } _ { d _ { \tau } ^ { * } } \big [ \langle \epsilon _ { t } , \pi _ { \tau } ^ { * } - \pi _ { t } \rangle _ { \ r { A } } \big ] \leq \mathbb { E } _ { d _ { \tau } ^ { * } } \Big [ \sqrt { 2 D _ { K L } ( \pi _ { \tau } ^ { * } ( \cdot | s ) | | \pi _ { t } ( \cdot | s ) ) } \sqrt { W ^ { \dagger } ( s ) \mathbb { E } _ { a \sim \pi _ { t } } [ \epsilon _ { t } ( s , a ) ^ { 2 } ] } \Big ] } \\ & { \quad \quad \quad \leq \sqrt { 2 D _ { t } ^ { \dagger } } \sqrt { \mathbb { E } _ { d _ { \tau } ^ { * } } \big [ W ^ { \dagger } ( s ) \mathbb { E } _ { a \sim \pi _ { t } } [ \epsilon _ { t } ( s , a ) ^ { 2 } ] \big ] } . } \end{array}
$$

To control the second root, we change the state measure to $d ^ { \pi _ { t } }$ and apply the joint concentrability

bound $\begin{array} { r } { \frac { d _ { \tau } ^ { * } ( s ) } { d ^ { \pi _ { t } } ( s ) } W ^ { \dagger } ( s ) = \operatorname* { m a x } _ { a } \frac { d _ { \tau } ^ { * } ( s ) \pi _ { \tau } ^ { * } ( a | s ) } { d ^ { \pi _ { t } } ( s ) \pi _ { t } ( a | s ) } \le C _ { j o i n t } ^ { \dagger } } \end{array}$ (Assumption 9):

$$
\begin{array} { r l } { \mathbb { E } _ { d _ { \tau } ^ { * } } \left[ W ^ { \dagger } ( s ) \mathbb { E } _ { a \sim \pi _ { t } } [ \epsilon _ { t } ( s , a ) ^ { 2 } ] \right] = \mathbb { E } _ { d ^ { \pi _ { t } } } \left[ \left( \frac { d _ { \tau } ^ { * } ( s ) } { d ^ { \pi _ { t } } ( s ) } W ^ { \dagger } ( s ) \right) \mathbb { E } _ { a \sim \pi _ { t } } [ \epsilon _ { t } ( s , a ) ^ { 2 } ] \right] } & { } \\ & { \leq C _ { j o i n t } ^ { \dagger } \mathbb { E } _ { d ^ { \pi _ { t } } } \left[ \mathbb { E } _ { a \sim \pi _ { t } } [ \epsilon _ { t } ( s , a ) ^ { 2 } ] \right] } \\ & { = C _ { j o i n t } ^ { \dagger } \mathbb { E } _ { d ^ { \pi _ { t } } , \pi _ { t } } [ \epsilon _ { t } ^ { 2 } ] \leq C _ { j o i n t } ^ { \dagger } \epsilon _ { a p p } , } \end{array}
$$

where the last step uses Assumption 4. Applying Young’s Inequality $\begin{array} { r } { ( x y \le \frac { \tau } { 8 } x ^ { 2 } + \frac { 2 } { \tau } y ^ { 2 } ) } \end{array}$ with $x = \sqrt { D _ { t } ^ { \dagger } }$ and $y = \sqrt { 2 C _ { j o i n t } ^ { \dagger } \epsilon _ { a p p } } ,$ this term is bounded by $\begin{array} { r } { \frac { \tau } { 8 } D _ { t } ^ { \dagger } + \frac { 4 C _ { j o i n t } ^ { \dagger } } { \tau } \epsilon _ { a p p } . } \end{array}$

(iii) Tracking Error Bound. For the tracking error, the bounded features assumption guarantees $| e _ { t } ^ { \top } \phi ( s , a ) | \leq \| e _ { t } \| _ { 2 } \| \phi ( s , a ) \| _ { 2 } \leq B _ { \phi } \| e _ { t } \| _ { 2 }$ . Applying H¨older’s inequality over the action space and Pinsker’s inequality $( \| \pi _ { \tau } ^ { * } - \pi _ { t } \| _ { 1 } \leq \sqrt { 2 D _ { K L } ( \pi _ { \tau } ^ { * } \| \pi _ { t } ) } )$ yields:

$$
\begin{array} { r } { \mathbb { E } _ { d _ { \tau } ^ { * } } \Big [ \langle - e _ { t } ^ { \top } \phi , \pi _ { \tau } ^ { * } - \pi _ { t } \rangle _ { \mathcal { A } } \Big ] \leq \mathbb { E } _ { d _ { \tau } ^ { * } } \Big [ \operatorname* { m a x } _ { a } | e _ { t } ^ { \top } \phi ( s , a ) | \| \pi _ { \tau } ^ { * } - \pi _ { t } \| _ { 1 } \Big ] \leq \mathbb { E } _ { d _ { \tau } ^ { * } } \Big [ B _ { \phi } \| e _ { t } \| _ { 2 } \sqrt { 2 D _ { K L } ( \pi _ { \tau } ^ { * } | | \pi _ { t } ) } \Big ] . } \end{array}
$$

Applying Jensen’s inequality over the state distribution yields $B _ { \phi } \| e _ { t } \| _ { 2 } \sqrt { 2 D _ { t } ^ { \dagger } }$ . Crucially, because $\| e _ { t } \| _ { 2 }$ is a global scalar and we use Pinsker’s inequality, this term is free of any joint concentrability. Applying Young’s Inequality $\begin{array} { r } { ( x y \le \frac { \tau } { 8 } x ^ { 2 } + \frac { 2 } { \tau } y ^ { 2 } ) } \end{array}$ with $x = \sqrt { D _ { t } ^ { \dagger } }$ and $y = \sqrt { 2 } B _ { \phi } \| e _ { t } \| _ { 2 }$ bounds the tracking penalty:

$$
B _ { \phi } \| e _ { t } \| _ { 2 } \sqrt { 2 D _ { t } ^ { \dagger } } \le \frac { \tau } { 8 } D _ { t } ^ { \dagger } + \frac { 2 } { \tau } \big ( 2 B _ { \phi } ^ { 2 } \| e _ { t } \| _ { 2 } ^ { 2 } \big ) = \frac { \tau } { 8 } D _ { t } ^ { \dagger } + \frac { 4 B _ { \phi } ^ { 2 } } { \tau } \| e _ { t } \| _ { 2 } ^ { 2 } .
$$

(iv) Synthesis. Summing the bounds from (ii) and (iii) yields $\begin{array} { r } { \frac { \tau } { 4 } D _ { t } ^ { \dagger } + \frac { 4 C _ { j o i n t } ^ { \dagger } } { \tau } \epsilon _ { a p p } + \frac { 4 B _ { \phi } ^ { 2 } } { \tau } \| e _ { t } \| _ { 2 } ^ { 2 } . } \end{array}$ Substituting this into inequality (23), and combining the $D _ { t } ^ { \dagger }$ terms, $\begin{array} { r } { - \tau D _ { t } ^ { \dagger } + \frac { \tau } { 4 } D _ { t } ^ { \dagger } = - \frac { 3 \tau } { 4 } D _ { t } ^ { \dagger } } \end{array}$ , directly yields the single-step inequality:

$$
( 1 - \gamma ) { \mathrm { G a p } } _ { t } ^ { \dagger } \leq - \frac { 3 \tau } { 4 } D _ { t } ^ { \dagger } + \frac { D _ { t } ^ { \dagger } - D _ { t + 1 } ^ { \dagger } } { \eta _ { t } } + \frac { \eta _ { t } } { 2 } G _ { a c } ^ { 2 } B _ { \phi } ^ { 2 } + \frac { 4 C _ { j o i n t } ^ { \dagger } } { \tau } \epsilon _ { a p p } + \frac { 4 B _ { \phi } ^ { 2 } } { \tau } \| e _ { t } \| _ { 2 } ^ { 2 } .
$$

Taking the total expectation $\mathbb { E } [ \cdot ]$ over the algorithm trajectory and substituting $Z _ { t } \ \triangleq \ \mathbb { E } [ \| e _ { t }$ ∥<sup>2</sup><sub>2</sub>] completes the proof. □

Proof of Lemma 9.2 (PMD Value-Telescoping Bound). Because the initial ofset satisfies $t _ { 0 } \geq c _ { \eta } \tau$ and the step-size schedule is decreasing, we have $\begin{array} { r } { \eta _ { t } \le \eta _ { 0 } = \frac { c _ { \eta } } { t _ { 0 } } \le \frac { 1 } { \tau } } \end{array}$ for all $t \geq 0$ . This satisfies the step-size condition required to apply the PMD Actor Progress Bound (Lemma 9.1). We rearrange this single-step recursive bound to isolate the next-step divergence:

$$
\mathbb { E } [ D _ { t + 1 } ^ { \dagger } ] \leq \left( 1 - \frac { 3 \tau } { 4 } \eta _ { t } \right) \mathbb { E } [ D _ { t } ^ { \dagger } ] - \eta _ { t } ( 1 - \gamma ) \mathbb { E } [ \mathrm { G a p } _ { t } ^ { \dagger } ] + \eta _ { t } \mathcal { E } _ { n o i s e } ( t ) .
$$

Substituting the diminishing step size $\begin{array} { r } { \eta _ { t } = \frac { c _ { \eta } } { t + t _ { 0 } } } \end{array}$ , the contraction coeficient evaluates to $1 - \frac { 3 \tau c _ { \eta } / 4 } { t + t _ { 0 } }$

Because the step-size constant satisfies $\begin{array} { r } { \frac { 3 \tau } { 4 } c _ { \eta } \geq 1 } \end{array}$ , we upper-bound this coeficient:

$$
1 - \frac { 3 \tau c _ { \eta } / 4 } { t + t _ { 0 } } \leq 1 - \frac { 1 } { t + t _ { 0 } } = \frac { t + t _ { 0 } - 1 } { t + t _ { 0 } } .
$$

Substituting this upper bound into the recurrence gives:

$$
\mathbb { E } [ D _ { t + 1 } ^ { \dagger } ] \leq \frac { t + t _ { 0 } - 1 } { t + t _ { 0 } } \mathbb { E } [ D _ { t } ^ { \dagger } ] - \frac { c _ { \eta } } { t + t _ { 0 } } ( 1 - \gamma ) \mathbb { E } [ \operatorname { G a p } _ { t } ^ { \dagger } ] + \frac { c _ { \eta } } { t + t _ { 0 } } \mathcal { E } _ { n o i s e } ( t ) .
$$

We multiply the entire inequality by $( t + t _ { 0 } )$ and rearrange the terms to isolate the gap on the left side:

$$
c _ { \eta } ( 1 - \gamma ) \mathbb { E } [ \mathrm { G a p } _ { t } ^ { \dagger } ] \leq ( t + t _ { 0 } - 1 ) \mathbb { E } [ D _ { t } ^ { \dagger } ] - ( t + t _ { 0 } ) \mathbb { E } [ D _ { t + 1 } ^ { \dagger } ] + c _ { \eta } \mathcal { E } _ { n o i s e } ( t ) .
$$

Summing this inequality from $t = 0$ to $T - 1$ , the divergence terms telescope:

$$
\begin{array} { l } { \displaystyle c _ { \eta } ( 1 - \gamma ) \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } [ \mathbb { G } \mathbf { a } \mathbf { p } _ { t } ^ { \dagger } ] \leq \sum _ { t = 0 } ^ { T - 1 } \Big ( ( t + t _ { 0 } - 1 ) \mathbb { E } [ D _ { t } ^ { \dagger } ] - ( t + t _ { 0 } ) \mathbb { E } [ D _ { t + 1 } ^ { \dagger } ] \Big ) + c _ { \eta } \sum _ { t = 0 } ^ { T - 1 } \mathcal { E } _ { n o i s e } ( t ) } \\ { \displaystyle \qquad = ( t _ { 0 } - 1 ) D _ { 0 } ^ { \dagger } - ( T + t _ { 0 } - 1 ) \mathbb { E } [ D _ { T } ^ { \dagger } ] + c _ { \eta } \sum _ { t = 0 } ^ { T - 1 } \mathcal { E } _ { n o i s e } ( t ) , } \end{array}
$$

where we used $\mathbb { E } [ D _ { 0 } ^ { \dagger } ] = D _ { 0 } ^ { \dagger }$ because the initial actor parameter $\omega _ { 0 }$ is fixed. Dropping the nonpositive terminal term $- ( T + t _ { 0 } - 1 ) \mathbb { E } [ D _ { T } ^ { \dagger } ]$ and dividing the entire inequality by $c _ { \eta } T$ isolates the average sequence and completes the proof. □

Proof of Lemma 9.3 (Fractional Chung’s Lemma). Let $s = t + t _ { 0 }$ . Because $t _ { 0 } \geq c ^ { 3 / 2 }$ , we have $s \geq c ^ { 3 / 2 }$ , which guarantees the contraction multiplier $\textstyle \left( 1 - { \frac { c } { s ^ { 2 / 3 } } } \right)$ is non-negative. We prove $\begin{array} { r } { x _ { t } \leq \frac { v } { s ^ { 2 / 3 } } } \end{array}$ by induction.

The base case at $t = 0$ holds trivially by the definition of v: $\begin{array} { r } { x _ { 0 } \le \frac { v } { t _ { 0 } ^ { 2 / 3 } } } \end{array}$ . Assume the bound holds at step t. Substituting the inductive hypothesis into the recurrence yields:

$$
x _ { t + 1 } \leq \left( 1 - { \frac { c } { s ^ { 2 / 3 } } } \right) { \frac { v } { s ^ { 2 / 3 } } } + { \frac { K } { s ^ { 4 / 3 } } } = { \frac { v } { s ^ { 2 / 3 } } } - { \frac { c v - K } { s ^ { 4 / 3 } } } .
$$

By the convexity of $f ( s ) = s ^ { - 2 / 3 }$ , the function is lower-bounded by its first-order Taylor expansion $f ( s + 1 ) \geq f ( s ) + f ^ { \prime } ( s ) ( s + 1 - s )$ :

$$
{ \frac { 1 } { ( s + 1 ) ^ { 2 / 3 } } } \geq { \frac { 1 } { s ^ { 2 / 3 } } } - { \frac { 2 } { 3 s ^ { 5 / 3 } } } .
$$

Therefore, to guarantee $\begin{array} { r } { x _ { t + 1 } \leq \frac { v } { ( s + 1 ) ^ { 2 / 3 } } } \end{array}$ , it is suficient to show:

$$
\frac { c v - K } { s ^ { 4 / 3 } } \geq \frac { 2 v } { 3 s ^ { 5 / 3 } } \Longleftrightarrow c v - K \geq \frac { 2 v } { 3 s ^ { 1 / 3 } } .
$$

Because the initial ofset ensures $s \geq t _ { 0 } \geq ( \frac { 4 } { 3 c } ) ^ { 3 }$ , the right-hand side is bounded by: $\frac { 2 v } { 3 s ^ { 1 / 3 } } ~ \leq$ $\begin{array} { r } { \frac { 2 v } { 3 } \left( \frac { 3 c } { 4 } \right) = \frac { c } { 2 } v } \end{array}$ . A suficient condition is then $c v - K \geq \frac { c } { 2 } v$ or equivalently ${ \frac { c } { 2 } } v \geq K$ . This matches our definition $v \geq { \frac { 2 K } { c } }$ , completing the induction. □

Proof of Lemma 9.4 (Pointwise Critic Tracking Error). (i) Step-Size Validity. We verify that the initial ofset $t _ { 0 }$ guarantees the required structural conditions. First, $t _ { 0 } \geq c _ { \eta } \tau$ guarantees that $\begin{array} { r } { \eta _ { t } \leq \eta _ { 0 } = \frac { c _ { \eta } } { t _ { 0 } } \leq 1 / \tau } \end{array}$ for all $t \geq 0$ , which ensures the actor update direction is bounded by $G _ { a c }$ via Lemma 2.2(ii). Second, to ensure the SGD contraction multiplier is non-negative $( 1 - 2 \alpha _ { t + 1 } \kappa \ge 0 )$ for the Young’s inequality substitution, we require $\begin{array} { r } { \alpha _ { 1 } \le \frac { 1 } { 2 \kappa } } \end{array}$ , which is guaranteed by $t _ { 0 } \geq ( 2 c _ { \alpha } \kappa ) ^ { 3 / 2 }$ Third, to apply the Fractional Chung’s Lemma (Lemma 9.3) with contraction parameter $c = c _ { \alpha } \kappa .$ we require $t _ { 0 } \geq$ max $\left( c ^ { 3 / 2 } , ( \frac { 4 } { 3 c } ) ^ { 3 } \right)$ . The first term evaluates to $c ^ { 3 / 2 } = ( c _ { \alpha } \kappa ) ^ { 3 / 2 }$ , and the second term evaluates to $\begin{array} { r } { ( \frac { 4 } { 3 c } ) ^ { 3 } = ( \frac { 4 } { 3 c _ { \alpha } \kappa } ) ^ { 3 } } \end{array}$ . Both terms are enveloped by the assumed lower bound on $t _ { 0 }$

(ii) SGD Contraction and Young’s Inequality. Following the identical SGD contraction and Young’s inequality as derived in Steps (i)–(iii) of the Coupled Critic Tracking proof (Lemma 5.5), the tracking error conditioned on $\mathcal { F } _ { t + 1 }$ satisfies:

$$
\mathbb { E } [ \| e _ { t + 1 } \| _ { 2 } ^ { 2 } \mid \mathcal { F } _ { t + 1 } ] \le ( 1 - \alpha _ { t + 1 } \kappa ) \| e _ { t } \| _ { 2 } ^ { 2 } + \alpha _ { t + 1 } ^ { 2 } G _ { c r } ^ { 2 } + \frac { 1 } { \alpha _ { t + 1 } \kappa } \| \theta _ { \tau } ^ { * } ( \omega _ { t } ) - \theta _ { \tau } ^ { * } ( \omega _ { t + 1 } ) \| _ { 2 } ^ { 2 } .
$$

Taking the total expectation $\mathbb { E } [ \cdot ]$ over the algorithmic trajectory yields the single-step recurrence:

$$
Z _ { t + 1 } \le \underbrace { ( 1 - \alpha _ { t + 1 } \kappa ) Z _ { t } + \alpha _ { t + 1 } ^ { 2 } G _ { c r } ^ { 2 } } _ { \mathrm { S G D ~ E r r o r } } + \underbrace { \frac { 1 } { \alpha _ { t + 1 } \kappa } \mathbb { E } \big [ \| \theta _ { \tau } ^ { * } ( \omega _ { t } ) - \theta _ { \tau } ^ { * } ( \omega _ { t + 1 } ) \| _ { 2 } ^ { 2 } \big ] } _ { \mathrm { T a r g e t ~ D r i f t } } .
$$

(iii) Controlling Target Drift. In Part I, we dynamically coupled this drift to the objective performance gap. Here, because $\lambda _ { \tau }  0$ prevents that coupling, we evaluate it as an uncoupled worst-case bound. We utilize the Lipschitz continuity of the ideal critic mapping (Lemma 2.3) and the uncoupled parameter step bound $\| \omega _ { t + 1 } - \omega _ { t } \| _ { 2 } = \eta _ { t } \| g _ { t } ^ { a c } \| _ { 2 } \leq \eta _ { t } G _ { a c }$ (Lemma 2.2(ii)):

$$
\begin{array} { r } { \mathbb { E } \big [ \| \theta _ { \tau } ^ { * } ( \omega _ { t } ) - \theta _ { \tau } ^ { * } ( \omega _ { t + 1 } ) \| _ { 2 } ^ { 2 } \big ] \leq L _ { \theta } ^ { 2 } \mathbb { E } \big [ \| \omega _ { t + 1 } - \omega _ { t } \| _ { 2 } ^ { 2 } \big ] \leq L _ { \theta } ^ { 2 } \eta _ { t } ^ { 2 } G _ { a c } ^ { 2 } . } \end{array}
$$

(iv) Pointwise Recurrence. Substituting the target drift bound into the tracking recurrence yields:

$$
Z _ { t + 1 } \leq \left( 1 - \alpha _ { t + 1 } \kappa \right) Z _ { t } + \alpha _ { t + 1 } ^ { 2 } G _ { c r } ^ { 2 } + \frac { 1 } { \alpha _ { t + 1 } \kappa } L _ { \theta } ^ { 2 } \eta _ { t } ^ { 2 } G _ { a c } ^ { 2 } .
$$

Substituting the step sizes $\begin{array} { r } { \alpha _ { t + 1 } = \frac { c _ { \alpha } } { ( t + t _ { 0 } ) ^ { 2 / 3 } } } \end{array}$ and $\begin{array} { r } { \begin{array} { r } { \eta _ { t } = \frac { c _ { \eta } } { t + t _ { 0 } } } \end{array} } \end{array}$ into the additive noise terms exactly extracts our defined constant $K$

$$
\frac { c _ { \alpha } ^ { 2 } G _ { c r } ^ { 2 } } { ( t + t _ { 0 } ) ^ { 4 / 3 } } + \frac { L _ { \theta } ^ { 2 } c _ { \eta } ^ { 2 } G _ { a c } ^ { 2 } } { \kappa c _ { \alpha } ( t + t _ { 0 } ) ^ { 4 / 3 } } = \frac { K } { ( t + t _ { 0 } ) ^ { 4 / 3 } } .
$$

The sequence satisfies the recurrence $\begin{array} { r } { Z _ { t + 1 } \le \left( 1 - \frac { c _ { \alpha } \kappa } { ( t + t _ { 0 } ) ^ { 2 / 3 } } \right) Z _ { t } + \frac { K } { ( t + t _ { 0 } ) ^ { 4 / 3 } } } \end{array}$ . Applying the Fractional Chung’s Lemma (Lemma 9.3) with contraction coeficient $c = c _ { \alpha } \kappa _ { \mathrm { ~ J ~ } }$ yields the stated pointwise bound and completes the proof. □

Proof of Theorem 9.1 (Average-Iterate Regularized Convergence). (i) Telescoping Expansion. By the PMD Value-Telescoping Bound (Lemma 9.2), the average gap is bounded by:

$$
( 1 - \gamma ) \frac { 1 } { T } \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } [ \mathrm { G a p } _ { t } ^ { \dag } ] \leq \frac { t _ { 0 } - 1 } { c _ { \eta } T } D _ { 0 } ^ { \dag } + \frac { 1 } { T } \sum _ { t = 0 } ^ { T - 1 } \mathcal { E } _ { n o i s e } ( t ) ,
$$

where the noise envelope is $\begin{array} { r } { \mathcal { E } _ { n o i s e } ( t ) = \frac { \eta _ { t } } { 2 } G _ { a c } ^ { 2 } B _ { \phi } ^ { 2 } + \frac { 4 C _ { j o i n t } ^ { \dagger } } { \tau } \epsilon _ { a p p } + \frac { 4 B _ { \phi } ^ { 2 } } { \tau } Z _ { t } . } \end{array}$

(ii) Evaluating Sums. By the Pointwise Critic Tracking Error (Lemma 9.4), the expected tracking error is bounded by $Z _ { t } \le v ( t + t _ { 0 } ) ^ { - 2 / 3 }$ . Substituting $\eta _ { t }$ and $Z _ { t }$ into the noise sum yields:

$$
\frac { 1 } { T } \sum _ { t = 0 } ^ { T - 1 } \mathcal { E } _ { n o i s e } ( t ) \leq \frac { c _ { \eta } G _ { a c } ^ { 2 } B _ { \phi } ^ { 2 } } { 2 T } \sum _ { t = 0 } ^ { T - 1 } \frac { 1 } { t + t _ { 0 } } + \frac { 4 C _ { j o i n t } ^ { \dagger } } { \tau } \epsilon _ { a p p } + \frac { 4 B _ { \phi } ^ { 2 } v } { \tau T } \sum _ { t = 0 } ^ { T - 1 } \frac { 1 } { ( t + t _ { 0 } ) ^ { 2 / 3 } } .
$$

Because $\begin{array} { r } { t _ { 0 } \geq c _ { \eta } \tau \geq \frac { 4 } { 3 } > 1 } \end{array}$ , we bound the harmonic sum by $\begin{array} { r } { \sum _ { t = 0 } ^ { T - 1 } \frac { 1 } { t + t _ { 0 } } \leq 1 + \log T } \end{array}$ . For the fractional tracking sum, we bound it via the integral $\begin{array} { r } { \int _ { 0 } ^ { T } ( x + t _ { 0 } - 1 ) ^ { - 2 / 3 } d x = 3 ( T + t _ { 0 } - 1 ) ^ { 1 / 3 } - 3 ( t _ { 0 } - 1 ) ^ { 1 / 3 } \leq } \end{array}$ $3 ( T + t _ { 0 } ) ^ { 1 / 3 }$ . Dividing this tracking sum by $T$ evaluates to $\begin{array} { r } { 3 T ^ { - 2 / 3 } \left( 1 + \frac { t _ { 0 } } { T } \right) ^ { 1 / 3 } } \end{array}$ . Therefore, the tracking penalty is bounded by $\frac { 1 2 B _ { \phi } ^ { 2 } v } { \tau } T ^ { - 2 / 3 } \left( 1 + \frac { t _ { 0 } } { T } \right) ^ { 1 / 3 }$

(iii) Synthesis and Asymptotic Order. Substituting these bounded sums back into the PMD expansion yields the bound:

$$
\begin{array} { r } { ( 1 - \gamma ) \displaystyle \frac { 1 } { T } \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } [ \mathrm { G a p } _ { t } ^ { \dag } ] \leq \underbrace { \frac { t _ { 0 } - 1 } { c _ { \eta } T } D _ { 0 } ^ { \dag } } _ { \mathrm { P M D ~ I m i t i a l i z a t i o n ~ E r r o r } } + \underbrace { \frac { c _ { \eta } G _ { a c } ^ { 2 } B _ { \phi } ^ { 2 } } { 2 T } ( 1 + \log T ) } _ { \mathrm { P M D ~ L o c a l ~ E r r o r } } } \\ { + \underbrace { \frac { 1 2 B _ { \phi } ^ { 2 } v } { \tau } T ^ { - 2 / 3 } \left( 1 + \frac { t _ { 0 } } { T } \right) ^ { 1 / 3 } } _ { \mathrm { C r i t i c ~ T r a c k i n g ~ E r r o r } } + \underbrace { \frac { 4 C _ { j o i n t } ^ { \dag } } { \tau } \epsilon _ { a p p } } _ { \mathrm { A p p r o x i m a t i o n ~ B i a s } } . } \end{array}
$$

To evaluate the structural orders of the bound, we explicitly track the dependencies on the temperature $\tau _ { : }$ , the feature moment minimum eigenvalue $\kappa ,$ and the horizon $( 1 - \gamma ) ^ { - 1 }$ . We choose $c _ { \eta } = \Theta ( \tau ^ { - 1 } )$ subject to $\begin{array} { r } { c _ { \eta } \ge \frac { 4 } { 3 \tau } } \end{array}$ . By choosing the critic step-size constant $\textstyle \zeta = \Theta ( \frac { 1 } { \kappa } )$ , the product $\zeta \kappa = \Theta ( 1 )$ . Consequently, the lower bound on $t _ { 0 }$ is, up to constant factors, max $\begin{array} { r } { \left( c _ { \eta } \tau , c _ { \eta } , \frac { 6 4 } { 2 7 c _ { \eta } ^ { 2 } } \right) } \end{array}$ eliminating the conflicting κ dependencies. Because $c _ { \eta } = \Theta ( \tau ^ { - 1 } )$ , the dominant term in the lower bound for $t _ { 0 }$ is $c _ { \eta }$ . This allows us to satisfy the initial ofset condition by choosing $t _ { 0 } = \Theta ( \tau ^ { - 1 } )$

• Noise Envelope $( K ) .$ : We evaluate $\begin{array} { r } { K = c _ { \alpha } ^ { 2 } G _ { c r } ^ { 2 } + \frac { L _ { \theta } ^ { 2 } c _ { \eta } ^ { 2 } G _ { a c } ^ { 2 } } { \kappa c _ { \alpha } } = ( \zeta c _ { \eta } ^ { 2 / 3 } ) ^ { 2 } G _ { c r } ^ { 2 } + \frac { L _ { \theta } ^ { 2 } c _ { \eta } ^ { 4 / 3 } G _ { a c } ^ { 2 } } { \kappa \zeta } } \end{array}$ . By Lemmas 2.2 and 2.3, the structural bounds scale as $G _ { c r } ^ { 2 } = \Theta ( \kappa ^ { - 2 } ( 1 - \gamma ) ^ { - 2 } ) , G _ { a c } = \Theta ( \kappa ^ { - 1 } ( 1 -$ $\gamma ) ^ { - 1 } )$ , and $L _ { \theta } = \Theta ( \kappa ^ { - 2 } ( 1 - \gamma ) ^ { - 2 } )$ . Substituting these alongside $c _ { \eta } ^ { 4 / 3 } = \Theta ( \tau ^ { - 4 / 3 } )$ and $\zeta \kappa =$ $\Theta ( 1 )$ , the first term (SGD error) evaluates to $\Theta ( \tau ^ { - 4 / 3 } \kappa ^ { - 4 } ( 1 - \gamma ) ^ { - 2 } )$ , while the second term (target drift) evaluates to $\Theta ( \tau ^ { - 4 / 3 } \kappa ^ { - 6 } ( 1 - \gamma ) ^ { - 6 } )$ . Because both terms share an identical $\Theta ( \tau ^ { - 4 / 3 } )$ temperature dependence, the second term dominates in its dependence on κ and

$( 1 - \gamma ) ^ { - 1 }$ , yielding $K = \Theta ( \tau ^ { - 4 / 3 } \kappa ^ { - 6 } ( 1 - \gamma ) ^ { - 6 } )$

• Tracking Bounding Coeficient (v): The bounding coeficient is $\begin{array} { r } { v = \operatorname* { m a x } \left( t _ { 0 } ^ { 2 / 3 } Z _ { 0 } , \frac { 2 K } { \zeta c _ { \eta } ^ { 2 / 3 } \kappa } \right) } \end{array}$ Since $\zeta \kappa ~ = ~ \Theta ( 1 )$ and $c _ { \eta } ^ { 2 / 3 } ~ = ~ \Theta ( \tau ^ { - 2 / 3 } )$ , the second argument scales as $\Theta ( K \tau ^ { 2 / 3 } ) =$ $\Theta ( \tau ^ { - 2 / 3 } \kappa ^ { - 6 } ( 1 - \gamma ) ^ { - 6 } )$ . As shown in the proof of Theorem 6.2, $Z _ { 0 } ~ \le ~ ( 2 B _ { \theta } ) ^ { 2 } ~ = ~ G _ { a c } ^ { 2 } ~ =$ $\Theta ( \kappa ^ { - 2 } ( 1 - \gamma ) ^ { - 2 } )$ . Thus, the initial error term evaluates to $t _ { 0 } ^ { 2 / 3 } Z _ { 0 } = \mathcal { O } ( \tau ^ { - 2 / 3 } \kappa ^ { - 2 } ( 1 - \gamma ) ^ { - 2 } )$ This is dominated by the second argument, yielding $v = \Theta ( \tau ^ { - 2 / 3 } \kappa ^ { - 6 } ( 1 - \gamma ) ^ { - 6 } )$ . Multiplying by $\scriptstyle { \frac { 1 } { \tau } }$ gives the tracking multiplier $\begin{array} { r } { \frac { v } { \tau } = \Theta ( \tau ^ { - 5 / 3 } \kappa ^ { - 6 } ( 1 - \gamma ) ^ { - 6 } ) } \end{array}$

• Critic Tracking Error: Because T appears inside the cube root, we apply the subadditivity inequality $( 1 + t _ { 0 } / T ) ^ { 1 / 3 } \leq 1 + ( t _ { 0 } / T ) ^ { 1 / 3 }$ . Since $t _ { 0 } = \Theta ( \tau ^ { - 1 } )$ , the root evaluates to $( t _ { 0 } / T ) ^ { 1 / 3 } =$ $\Theta ( \tau ^ { - 1 / 3 } T ^ { - 1 / 3 } )$ . Multiplying this by the leading tracking multiplier decomposes the tracking error into two asymptotic upper bounds:

$$
\begin{array} { r l } & { \quad \mathcal { O } ( \tau ^ { - 5 / 3 } \kappa ^ { - 6 } ( 1 - \gamma ) ^ { - 6 } ) T ^ { - 2 / 3 } \left( 1 + \Theta ( \tau ^ { - 1 / 3 } T ^ { - 1 / 3 } ) \right) } \\ & { } \\ & { = \cal { O } ( \tau ^ { - 5 / 3 } \kappa ^ { - 6 } ( 1 - \gamma ) ^ { - 6 } T ^ { - 2 / 3 } ) + \mathcal { O } ( \tau ^ { - 2 } \kappa ^ { - 6 } ( 1 - \gamma ) ^ { - 6 } T ^ { - 1 } ) . } \end{array}
$$

• PMD Initialization and Local Errors: By assumption $\begin{array} { r } { \| \omega _ { 0 } \| _ { 2 } \le \frac { B _ { \theta } } { \tau _ { m a x } } } \end{array}$ , the initial policy logprobability is bounded by $\begin{array} { r } { - \log \pi _ { 0 } ( a | s ) = - \omega _ { 0 } ^ { \top } \phi ( s , a ) + \log \sum _ { a ^ { \prime } } \exp ( \omega _ { 0 } ^ { \top } \phi ( s , a ^ { \prime } ) ) \leq 2 \| \omega _ { 0 } \| _ { 2 } B _ { \phi } + } \end{array}$ log $\begin{array} { r } { | \mathcal { A } | \le \frac { 2 B _ { \theta } B _ { \phi } } { \tau _ { m a x } } + \log | \mathcal { A } | } \end{array}$ . Thus, the initial KL divergence is bounded by $\begin{array} { r } { D _ { 0 } ^ { \dagger } \le \frac { 2 B _ { \theta } B _ { \phi } } { \tau _ { m a x } } + } \end{array}$ $\log | \mathcal { A } | = \mathcal { O } ( \kappa ^ { - 1 } ( 1 { - } \gamma ) ^ { - 1 } )$ . Because $\frac { t _ { 0 } - 1 } { c _ { \eta } } = \Theta ( 1 )$ , the initialization error evaluates to $\begin{array} { r } { \frac { t _ { 0 } - 1 } { c _ { \eta } T } D _ { 0 } ^ { \dagger } = } \end{array}$ $\mathcal { O } ( \kappa ^ { - 1 } ( 1 - \gamma ) ^ { - 1 } T ^ { - 1 } )$ . The PMD local error evaluates to $\begin{array} { r } { \frac { c _ { \eta } G _ { a c } ^ { 2 } B _ { \phi } ^ { 2 } } { 2 T } \log T = \mathcal { O } ( \tau ^ { - 1 } \kappa ^ { - 2 } ( 1 - } \end{array}$ $\gamma ) ^ { - 2 } T ^ { - 1 } \log T )$ . Both terms are subsumed by the $\mathcal { O } ( \tau ^ { - 2 } \kappa ^ { - 6 } ( 1 - \gamma ) ^ { - 6 } T ^ { - 1 } )$ transient critic tracking error (second term above), up to a log T factor.

Dividing the entire expression by $( 1 - \gamma )$ distributes the $( 1 - \gamma ) ^ { - 1 }$ factor across all asymptotic terms. This elevates the $( 1 - \gamma ) ^ { - 6 }$ dependencies to $( 1 - \gamma ) ^ { - 7 }$ , yielding the final stated expected gap bound and completing the proof. □

Proof of Lemma 9.5 (Chung’s Lemma with Fractional Noise). Let $s = t + t _ { 0 }$ . Since $c \geq 1$ and $t _ { 0 } \geq c ,$ , we have $s \geq c \geq 1$ , guaranteeing the contraction multiplier $( 1 - c / s )$ is non-negative. We proceed by induction to prove $\begin{array} { r } { x _ { t } \leq \frac { u } { s ^ { 2 / 3 } } + \frac { B } { c } } \end{array}$

The base case at $t = 0$ holds trivially by the definition of u: $\begin{array} { r } { x _ { 0 } \le \frac { u } { t _ { 0 } ^ { 2 / 3 } } \le \frac { u } { t _ { 0 } ^ { 2 / 3 } } + \frac { B } { c } } \end{array}$ . Assume the bound holds at step t. Substituting the inductive hypothesis into the recurrence yields:

$$
x _ { t + 1 } \leq \left( 1 - { \frac { c } { s } } \right) \left( { \frac { u } { s ^ { 2 / 3 } } } + { \frac { B } { c } } \right) + { \frac { A } { s ^ { 5 / 3 } } } + { \frac { B } { s } }
$$

$$
{ \begin{array} { r l } & { = { \cfrac { u } { s ^ { 2 / 3 } } } - { \cfrac { c u - A } { s ^ { 5 / 3 } } } + { \cfrac { B } { c } } - { \cfrac { B } { s } } + { \cfrac { B } { s } } } \\ & { = { \cfrac { u } { s ^ { 2 / 3 } } } - { \cfrac { c u - A } { s ^ { 5 / 3 } } } + { \cfrac { B } { c } } . } \end{array} }
$$

By the convexity of $f ( s ) = s ^ { - 2 / 3 }$ , the function is lower-bounded by its first-order Taylor expansion: $\begin{array} { r } { \frac { 1 } { ( s + 1 ) ^ { 2 / 3 } } \geq \frac { 1 } { s ^ { 2 / 3 } } - \frac { 2 } { 3 s ^ { 5 / 3 } } } \end{array}$ . Therefore, to guarantee $\begin{array} { r } { x _ { t + 1 } \leq \frac { u } { ( s + 1 ) ^ { 2 / 3 } } + \frac { B } { c } } \end{array}$ , it is suficient to show:

$$
\frac { c u - A } { s ^ { 5 / 3 } } \geq \frac { 2 u } { 3 s ^ { 5 / 3 } } \Longleftrightarrow c u - A \geq \frac { 2 } { 3 } u \Longleftrightarrow \left( c - \frac { 2 } { 3 } \right) u \geq A .
$$

Because $c \geq 1$ , we are guaranteed $c - 2 / 3 \geq 1 / 3 > 0$ . This suficient condition matches our definition $\begin{array} { r } { u \geq \frac { A } { c - 2 / 3 } = \frac { 3 A } { 3 c - 2 } } \end{array}$ , completing the induction. □

Proof of Lemma 9.6 (Last-Iterate Forward KL Convergence). (i) Pointwise Recurrence. We isolate the forward KL divergence from the single-step PMD Actor Progress Bound (Lemma 9.1) by dropping the non-negative $( 1 - \gamma ) \mathbb { E } [ \mathrm { G a p } _ { t } ^ { \dagger } ]$ term:

$$
\mathbb { E } [ D _ { t + 1 } ^ { \dagger } ] \leq \left( 1 - \frac { 3 \tau } { 4 } \eta _ { t } \right) \mathbb { E } [ D _ { t } ^ { \dagger } ] + \eta _ { t } \mathcal { E } _ { n o i s e } ( t ) .
$$

Let $s = t + t _ { 0 }$ . Substituting $\eta _ { t } = c _ { \eta } s ^ { - 1 }$ and the pointwise tracking bound $Z _ { t } ~ \le ~ v s ^ { - 2 / 3 }$ from Lemma 9.4, the additive noise evaluates to:

$$
\eta _ { t } \mathcal { E } _ { n o i s e } ( t ) = \frac { c _ { \eta } ^ { 2 } G _ { a c } ^ { 2 } B _ { \phi } ^ { 2 } } { 2 s ^ { 2 } } + \frac { 4 c _ { \eta } B _ { \phi } ^ { 2 } v } { \tau s ^ { 5 / 3 } } + \frac { 4 c _ { \eta } C _ { j o i n t } ^ { \dag } } { \tau s } \epsilon _ { a p p } .
$$

(ii) Applying Chung’s Lemma with Fractional Noise. For $s \geq 1$ , we upper-bound the $s ^ { - 2 }$ PMD local error term by s $- 5 / 3$ . Substituting the definitions $\begin{array} { r } { A \triangleq \frac { c _ { \eta } ^ { 2 } G _ { a c } ^ { 2 } B _ { \phi } ^ { 2 } } { 2 } + \frac { 4 c _ { \eta } B _ { \phi } ^ { 2 } v } { \tau } } \end{array}$ and $B \triangleq \frac { 4 c _ { \eta } C _ { j o i n t } ^ { \dagger } } { \tau } \epsilon _ { a p p } .$ the recurrence simplifies to:

$$
\mathbb { E } [ D _ { t + 1 } ^ { \dagger } ] \leq \left( 1 - \frac { c } { s } \right) \mathbb { E } [ D _ { t } ^ { \dagger } ] + \frac { A } { s ^ { 5 / 3 } } + \frac { B } { s } ,
$$

where $c \triangleq { \frac { 3 \tau } { 4 } } c _ { \eta }$ . The step-size condition $\begin{array} { r } { c _ { \eta } \geq \frac { 4 } { 3 \tau } } \end{array}$ guarantees $c \geq 1$ . Additionally, the initial ofset condition (Theorem 9.1) requires $t _ { 0 } \geq c _ { \eta } \tau$ , which ensures $\begin{array} { r } { t _ { 0 } \geq \frac { 4 } { 3 } c > c } \end{array}$ for Lemma 9.5. We apply Chung’s Lemma with Fractional Noise (Lemma 9.5) using the sequence $x _ { t } = \mathbb { E } [ D _ { t } ^ { \dagger } ]$ . The bias fraction cancels the actor step-size constant: $\begin{array} { r } { \frac { B } { c } = \frac { 4 c _ { \eta } C _ { j o i n t } ^ { \dagger } \epsilon _ { a p p } } { \tau } \frac { 4 } { 3 \tau c _ { \eta } } = \frac { 1 6 C _ { j o i n t } ^ { \dagger } } { 3 \tau ^ { 2 } } \epsilon _ { a p p } } \end{array}$ . This establishes the pointwise bound

$$
\mathbb { E } [ D _ { T } ^ { \dagger } ] \le \frac { u } { ( T + t _ { 0 } ) ^ { 2 / 3 } } + \frac { 1 6 C _ { j o i n t } ^ { \dagger } } { 3 \tau ^ { 2 } } \epsilon _ { a p p } ,
$$

where $u \overset { \triangle } { = }$ max $\left( t _ { 0 } ^ { 2 / 3 } D _ { 0 } ^ { \dagger } , \frac { 3 A } { 3 c - 2 } \right)$

(iii) Asymptotic Order. To evaluate the structural orders of the bounding constant $u ,$ we explicitly track the dependencies on the temperature τ , the feature moment minimum eigenvalue $\kappa ,$ and the horizon $( 1 - \gamma ) ^ { - 1 }$ . As established in the proof of Theorem 9.1, we satisfy the required conditions on the step sizes by choosing $c _ { \eta } = \Theta ( \tau ^ { - 1 } ) , \zeta = \Theta ( \kappa ^ { - 1 } )$ , and $t _ { 0 } = \Theta ( \tau ^ { - 1 } )$ . Under these choices, the tracking coeficient scales as $v = \Theta ( \tau ^ { - 2 / 3 } \kappa ^ { - 6 } ( 1 - \gamma ) ^ { - 6 } )$ .

• Initialization Term: By the initial KL divergence bound $D _ { 0 } ^ { \dagger } = \mathcal { O } ( \kappa ^ { - 1 } ( 1 - \gamma ) ^ { - 1 } )$ established in the proof of Theorem 9.1 and $t _ { 0 } = \Theta ( \tau ^ { - 1 } )$ , the initialization term evaluates to $t _ { 0 } ^ { 2 / 3 } D _ { 0 } ^ { \dagger } =$ $\mathcal { O } ( \tau ^ { - 2 / 3 } \kappa ^ { - 1 } ( 1 - \gamma ) ^ { - 1 } )$

• Fractional Noise Coeficient (A): For the numerator $\begin{array} { r } { A \ = \ \frac { c _ { \eta } ^ { 2 } G _ { a c } ^ { 2 } B _ { \phi } ^ { 2 } } { 2 } + \frac { 4 c _ { \eta } B _ { \phi } ^ { 2 } v } { \tau } } \end{array}$ , we evaluate the two components. By Lemma 2.2, the actor update bound scales as $G _ { a c } ^ { 2 } \ =$ $\Theta ( \kappa ^ { - 2 } ( 1 - \gamma ) ^ { - 2 } )$ . Thus, the PMD local error evaluates to $c _ { \eta } ^ { 2 } G _ { a c } ^ { 2 } = \Theta ( \tau ^ { - 2 } ) \Theta ( \kappa ^ { - 2 } ( 1 - \gamma ) ^ { - 2 } ) =$ $\Theta ( \tau ^ { - 2 } \kappa ^ { - 2 } ( 1 - \gamma ) ^ { - 2 } )$ . The critic tracking penalty evaluates to $\begin{array} { r } { \frac { c _ { \eta } v } { \tau } = \frac { \Theta ( \tau ^ { - 1 } ) \Theta ( \tau ^ { - 2 / 3 } \kappa ^ { - 6 } ( 1 - \gamma ) ^ { - 6 } ) } { \tau } = } \end{array}$ $\Theta ( \tau ^ { - 8 / 3 } \kappa ^ { - 6 } ( 1 - \gamma ) ^ { - 6 } )$ . Because $\tau ^ { - 8 / 3 }$ dominates $\tau ^ { - 2 }$ as $\tau  0$ , the critic tracking term dominates, yielding $A = \Theta ( \tau ^ { - 8 / 3 } \kappa ^ { - 6 } ( 1 - \gamma ) ^ { - 6 } )$

• Bounding Constant (u): The choice $c _ { \eta } = \Theta ( \tau ^ { - 1 } )$ guarantees $\begin{array} { r } { c \triangleq \frac { 3 \tau } { 4 } c _ { \eta } = \Theta ( 1 ) } \end{array}$ , meaning the denominator $3 c - 2 = \Theta ( 1 )$ . Therefore, $\textstyle { \frac { 3 A } { 3 c - 2 } } = \Theta ( \tau ^ { - 8 / 3 } \kappa ^ { - 6 } ( 1 - \gamma ) ^ { - 6 } )$ . Taking the maximum with the initialization term $t _ { 0 } ^ { 2 / 3 } D _ { 0 } ^ { \dagger } = \mathcal { O } ( \tau ^ { - 2 / 3 } \kappa ^ { - 1 } ( 1 - \gamma ) ^ { - 1 } )$ yields $u = \Theta ( \tau ^ { - 8 / 3 } \kappa ^ { - 6 } ( 1 - \gamma ) ^ { - 6 } )$

Substituting u into the pointwise bound yields the stated asymptotic convergence rate and completes the proof. □

Proof of Lemma 9.7 (Forward KL Bound on Regularized Suboptimality). (i) Extracting the KL Divergence. By the first step of the Regularized Suboptimality Decompositions (Lemma 3.3) evaluated for the training policy $\pi _ { t }$ against the optimal comparator $\pi _ { \tau } ^ { * }$ , the exact performance diference is:

$$
\begin{array} { r } { ( 1 - \gamma ) \big ( J _ { \tau } ( \pi _ { \tau } ^ { * } ) - J _ { \tau } ( \pi _ { t } ) \big ) = - \tau \mathbb { E } _ { d _ { \tau } ^ { * } } [ D _ { K L } ( \pi _ { \tau } ^ { * } \| \pi _ { t } ) ] + \mathbb { E } _ { d _ { \tau } ^ { * } } \Big [ \langle Q _ { \tau } ^ { \pi _ { t } } - \tau \log \pi _ { t } , \pi _ { \tau } ^ { * } - \pi _ { t } \rangle _ { \mathcal A } \Big ] . } \end{array}
$$

(ii) H¨older’s and Pinsker’s Bounds. Because the KL divergence is non-negative, dropping $- \tau D _ { K L } ( \pi _ { \tau } ^ { * } \| \pi _ { t } )$ establishes an upper bound:

$$
J _ { \tau } ( \pi _ { \tau } ^ { * } ) - J _ { \tau } ( \pi _ { t } ) \leq \frac { 1 } { 1 - \gamma } \mathbb { E } _ { d _ { \tau } ^ { * } } \Big [ \langle Q _ { \tau } ^ { \pi _ { t } } - \tau \log \pi _ { t } , \pi _ { \tau } ^ { * } - \pi _ { t } \rangle _ { \cal A } \Big ] .
$$

Applying H¨older’s inequality $\left( L _ { 1 } \times L _ { \infty } \right)$ to the summation and then applying Pinsker’s inequality $( \| \pi _ { \tau } ^ { * } - \pi _ { t } \| _ { 1 } \leq \sqrt { 2 D _ { K L } ( \pi _ { \tau } ^ { * } \| \pi _ { t } ) } )$ yields:

$$
J _ { \tau } ( \pi _ { \tau } ^ { * } ) - J _ { \tau } ( \pi _ { t } ) \leq \frac { 1 } { 1 - \gamma } \mathbb { E } _ { d _ { \tau } ^ { * } } \left[ \| \pi _ { \tau } ^ { * } ( a | s ) - \pi _ { t } ( a | s ) \| _ { 1 } \big | Q _ { \tau } ^ { \pi _ { t } } ( s , a ) - \tau \log \pi _ { t } ( a | s ) \big | \right]
$$

$$
\leq \frac { \sqrt { 2 } } { 1 - \gamma } \mathbb { E } _ { d _ { \tau } ^ { * } } \Big [ \operatorname* { m a x } _ { a } \big | Q _ { \tau } ^ { \pi _ { t } } ( s , a ) - \tau \log \pi _ { t } ( a | s ) \big | \sqrt { D _ { K L } ( \pi _ { \tau } ^ { * } ( \cdot | s ) \| \pi _ { t } ( \cdot | s ) ) } \Big ] .
$$

(iii) Bounding Q-value and Log-Policy via Parameter Capacity. We bound the magnitude of the Q-function and log-policy term for all actions. By Lemma 2.1, the regularized Q-value is bounded by $0 \leq Q _ { \tau } ^ { \pi _ { t } } ( s , a ) \leq V _ { m a x }$ . To bound the log-policy magnitude $\left| \tau \log \pi _ { t } ( a | s ) \right|$ , we substitute the log-linear identity τ log $\pi _ { t } ( a | s ) = \tau \omega _ { t } ^ { \top } \phi ( s , a ) - \tau \log Z _ { t } ( s )$ . By Lemma 2.2(ii), the actor iterate is bounded by $\| \omega _ { t } \| _ { 2 } \leq B _ { \theta } / \tau$ . Thus, the feature inner product is bounded by:

$$
| \tau \omega _ { t } ^ { \top } \phi ( s , a ) | \leq \tau \| \omega _ { t } \| _ { 2 } \| \phi ( s , a ) \| _ { 2 } \leq \tau \left( \frac { B _ { \theta } } { \tau } \right) B _ { \phi } = B _ { \theta } B _ { \phi } .
$$

The log-partition function is then bounded by:

$$
\tau \log Z _ { t } ( s ) = \tau \log \sum _ { a ^ { \prime } } \exp ( \omega _ { t } ^ { \top } \phi ( s , a ^ { \prime } ) ) \leq \tau \log \left( | A | \exp \left( \frac { B _ { \theta } B _ { \phi } } { \tau } \right) \right) = B _ { \theta } B _ { \phi } + \tau \log | A | .
$$

Similarly, τ log $\begin{array} { r } { Z _ { t } ( s ) \geq \tau \log \left( \exp \left( - \frac { B _ { \theta } B _ { \phi } } { \tau } \right) \right) = - B _ { \theta } B _ { \phi } } \end{array}$ . Therefore, |τ log $Z _ { t } ( s ) | \leq B _ { \theta } B _ { \phi } + \tau \log | A |$ By the triangle inequality, the log-policy magnitude is bounded by |τ log $\pi _ { t } ( a | s ) | \ \leq \ 2 B _ { \theta } B _ { \phi } \ +$ τ log |A|. Consequently, the magnitude of the Q-function and log-policy term is bounded by:

$$
\big | Q _ { \tau } ^ { \pi _ { t } } ( s , a ) - \tau \log \pi _ { t } ( a | s ) \big | \le V _ { m a x } + 2 B _ { \theta } B _ { \phi } + \tau \log | A | \triangleq M _ { \tau } .
$$

(iv) Jensen’s Inequality and $M _ { m a x }$ Relaxation. Applying Jensen’s inequality $\begin{array} { r l r } { \mathbb { E } _ { d _ { \tau } ^ { * } } [ \sqrt { D _ { K L } ( \pi _ { \tau } ^ { * } ( \cdot | s ) \| \pi _ { t } ( \cdot | s ) ) } ] } & { \leq } & { \sqrt { \mathbb { E } _ { d _ { \tau } ^ { * } } [ D _ { K L } ( \pi _ { \tau } ^ { * } ( \cdot | s ) \| \pi _ { t } ( \cdot | s ) ) ] } } \end{array}$ establishes the τ-dependent bound. Relaxing $M _ { \tau } \leq M _ { m a x }$ via $\tau \leq \tau _ { m a x }$ yields the τ -independent bound and completes the proof. □

Proof of Theorem 9.2 (Last-Iterate Regularized Convergence). (i) Forward KL Bound. By the Forward KL Bound on Regularized Suboptimality (Lemma 9.7), for each realized policy π<sub>T</sub> we have

$$
\mathrm { G a p } _ { T } ^ { \dag } \leq \frac { \sqrt { 2 } M _ { m a x } } { 1 - \gamma } \sqrt { D _ { T } ^ { \dag } } .
$$

Taking the expectation over the algorithmic trajectory and applying Jensen’s inequality yields:

$$
\mathbb { E } [ \mathrm { G a p } _ { T } ^ { \dag } ] \leq \frac { \sqrt { 2 } M _ { m a x } } { 1 - \gamma } \mathbb { E } \big [ \sqrt { D _ { T } ^ { \dag } } \big ] \leq \frac { \sqrt { 2 } M _ { m a x } } { 1 - \gamma } \sqrt { \mathbb { E } [ D _ { T } ^ { \dag } ] } .
$$

Substituting the explicit bound for the terminal divergence from Lemma 9.6 yields:

$$
\mathbb { E } [ \mathrm { G a p } _ { T } ^ { \dag } ] \leq \frac { \sqrt { 2 } M _ { m a x } } { 1 - \gamma } \sqrt { \frac { u } { ( T + t _ { 0 } ) ^ { 2 / 3 } } + \frac { 1 6 C _ { j o i n t } ^ { \dag } } { 3 \tau ^ { 2 } } \epsilon _ { a p p } } .
$$

Applying square-root subadditivity $( { \sqrt { x + y } } \leq { \sqrt { x } } + { \sqrt { y } } )$ separates the bound into two terms:

$$
\mathbb { E } [ \mathrm { G a p } _ { T } ^ { \dagger } ] \leq \frac { \sqrt { 2 u } M _ { m a x } } { 1 - \gamma } \frac { 1 } { ( T + t _ { 0 } ) ^ { 1 / 3 } } + \frac { 4 M _ { m a x } \sqrt { 2 C _ { j o i n t } ^ { \dagger } } } { \sqrt { 3 } \tau ( 1 - \gamma ) } \sqrt { \epsilon _ { a p p } } .
$$

(ii) Asymptotic Order. As established in Lemma 9.6, the fractional noise bounding constant scales as $u = \Theta ( \tau ^ { - 8 / 3 } \kappa ^ { - 6 } ( 1 - \gamma ) ^ { - 6 } )$ . Consequently, its square root guarantees $\sqrt { u } = \Theta ( \tau ^ { - 4 / 3 } \kappa ^ { - 3 } ( 1 -$ $\gamma ) ^ { - 3 } )$ . Recall from Lemma 9.7 that $M _ { m a x } = V _ { m a x } + 2 B _ { \theta } B _ { \phi } + \tau _ { m a x } \log | \cal { A } |$ . By Lemma 2.2, $B _ { \theta } =$ $\frac { B _ { \phi } V _ { m a x } } { \kappa }$ , while $V _ { m a x } = \Theta ( ( 1 - \gamma ) ^ { - 1 } )$ . Therefore, ${ \cal M } _ { m a x } = \Theta ( ( 1 - \gamma ) ^ { - 1 } \kappa ^ { - 1 } )$ . Hence, the leading coeficients for the finite-time term and approximation bias evaluate to:

$$
\frac { \sqrt { u } M _ { m a x } } { 1 - \gamma } = \Theta \left( \frac { 1 } { ( 1 - \gamma ) ^ { 5 } \kappa ^ { 4 } \tau ^ { 4 / 3 } } \right) , \quad \mathrm { a n d } \quad \frac { M _ { m a x } } { \tau ( 1 - \gamma ) } = \Theta \left( \frac { 1 } { ( 1 - \gamma ) ^ { 2 } \kappa \tau } \right) .
$$

Because $t _ { 0 } \geq 1$ , we bound $( T + t _ { 0 } ) ^ { - 1 / 3 } \leq T ^ { - 1 / 3 }$ . Substituting these structural orders yields the stated asymptotic bounds, completing the proof. □

## E.9 Proofs for Section 10

Proof of Theorem 10.1 (Unregularized Average-Iterate Convergence). We first consider $\epsilon _ { a p p } > 0$ . When $\epsilon _ { a p p } = 0$ , the approximation bias term vanishes and the result follows from the sample-limited calculation with $\begin{array} { r } { \tau _ { T } = \frac { \Delta } { 2 \log ( C _ { \gamma } T ^ { 2 / 3 } ) } } \end{array}$

(i) Global Unregularized Bound. Because the finite-time schedule $\tau _ { T }$ may initially exceed the temperature threshold required in Theorem 3.1, we apply the Universal Unregularized Suboptimality Bound (Corollary 3.1). This guarantees the following global inequality holds for any $\tau _ { T } > 0$

$$
\frac { 1 } { T } \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } \big [ J _ { 0 } ( \pi _ { 0 } ^ { * } ) - J _ { 0 } ( \pi _ { t } ) \big ] \leq \frac { 4 A _ { m a x } } { \Delta } \left( \frac { 1 } { T } \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } [ \mathrm { G a p } _ { t } ^ { \dag } ( \tau _ { T } ) ] \right) + C _ { t a i l } \exp \left( - \frac { \Delta } { 2 \tau _ { T } } \right)
$$

Substituting the explicit average regularized gap bound from Theorem 9.1 isolates the competing algorithmic components:

$$
\begin{array} { r } { \mathrm { T o t a l ~ E r r o r } \leq \underbrace { \mathcal { O } \left( \frac { A _ { m a x } } { \Delta ( 1 - \gamma ) ^ { 7 } \kappa ^ { 6 } \tau _ { T } ^ { 5 / 3 } } T ^ { - 2 / 3 } \right) } _ { \mathrm { P r i m a r y ~ T r a c k i n g ~ E r r o r } } + \underbrace { \mathcal { O } \left( \frac { A _ { m a x } } { \Delta ( 1 - \gamma ) \tau _ { T } } \epsilon _ { a p p } \right) } _ { \mathrm { A p p r o x i m a t i o n ~ B i a s } } } \\ { + \underbrace { C _ { t a i l } \exp \left( - \frac { \Delta } { 2 \tau _ { T } } \right) } _ { \mathrm { E n t r o p y ~ T a i l } } + \underbrace { \tilde { \mathcal { O } } \left( \frac { A _ { m a x } } { \Delta ( 1 - \gamma ) ^ { 7 } \kappa ^ { 6 } \tau _ { T } ^ { 2 } } T ^ { - 1 } \right) } _ { \mathrm { T r a n s i e n t ~ C o n v e r g e n c e ~ E r r o r } } . } \end{array}
$$

(ii) Regime 1: Sample-Limited Phase. When log $( T ^ { 2 / 3 } ) \le \log ( 1 + \epsilon _ { a p p } ^ { - 1 } )$ , the temperature is tuned to $\begin{array} { r } { \tau _ { T } = \frac { \Delta } { 2 \log ( C _ { \gamma } T ^ { 2 / 3 } ) } } \end{array}$ . Substituting this alongside $\begin{array} { r } { A _ { m a x } \leq \frac { 1 } { 1 - \gamma } , C _ { t a i l } \triangleq \frac { A _ { m a x } C _ { \gamma } } { 1 - \gamma } \leq \frac { C _ { \gamma } } { ( 1 - \gamma ) ^ { 2 } } } \end{array}$ , and log $\begin{array} { r } { C _ { \gamma } = \mathcal { O } ( \frac { 1 } { 1 - \gamma } ) } \end{array}$ into the global inequality yields:

$$
\begin{array} { r l } & { \mathrm { P r i m a r y ~ T r a c k i n g ~ E r r o r } = \displaystyle \mathcal { O } \Big ( \frac { 1 } { ( 1 - \gamma ) ^ { 8 } \kappa ^ { 6 } \Delta ^ { 8 / 3 } } ( \log ( C _ { \gamma } T ^ { 2 / 3 } ) ) ^ { 5 / 3 } T ^ { - 2 / 3 } \Big ) } \\ & { \quad \quad \quad \quad = \displaystyle \mathcal { O } \Big ( \frac { 1 } { ( 1 - \gamma ) ^ { 8 } \kappa ^ { 6 } \Delta ^ { 8 / 3 } } \Big ( \frac { 1 } { ( 1 - \gamma ) ^ { 5 / 3 } } + ( \log T ) ^ { 5 / 3 } \Big ) T ^ { - 2 / 3 } \Big ) , } \end{array}
$$

$$
\mathrm { A p p r o x i m a t i o n \ B i a s } = \mathcal { O } \Big ( \frac { 1 } { ( 1 - \gamma ) ^ { 2 } \Delta ^ { 2 } } \log ( C _ { \gamma } T ^ { 2 / 3 } ) \epsilon _ { a p p } \Big )
$$

$$
\leq \mathcal { O } \Big ( \frac { 1 } { ( 1 - \gamma ) ^ { 2 } \Delta ^ { 2 } } \Big ( \frac { 1 } { 1 - \gamma } + \log ( 1 + \epsilon _ { a p p } ^ { - 1 } ) \Big ) \epsilon _ { a p p } \Big ) ,
$$

$$
\mathrm { E n t r o p y ~ T a i l } \le \frac { C _ { \gamma } } { ( 1 - \gamma ) ^ { 2 } } \exp \big ( - \log ( C _ { \gamma } T ^ { 2 / 3 } ) \big ) = \frac { 1 } { ( 1 - \gamma ) ^ { 2 } T ^ { 2 / 3 } } ,
$$

$$
\mathrm { T r a n s i e n t \ E r r o r } = \tilde { \mathcal { O } } \Big ( \frac { 1 } { ( 1 - \gamma ) ^ { 8 } \kappa ^ { 6 } \Delta ^ { 3 } } ( \log ( C _ { \gamma } T ^ { 2 / 3 } ) ) ^ { 2 } T ^ { - 1 } \Big ) .
$$

The regime condition $\log ( T ^ { 2 / 3 } ) \leq \log ( 1 + \epsilon _ { a p p } ^ { - 1 } )$ guarantees

$$
\log ( C _ { \gamma } T ^ { 2 / 3 } ) \le \log C _ { \gamma } + \log ( 1 + \epsilon _ { a p p } ^ { - 1 } ) ,
$$

which gives the stated approximation-bias envelope. The $C _ { \gamma }$ prefactor cancels in the exponential tail, yielding a pure $\mathcal { O } ( T ^ { - 2 / 3 } )$ tail that is absorbed by the primary tracking error. The transient convergence error (which encapsulates the transient tracking error and PMD initialization and local errors) decays as $\tilde { \mathcal { O } } ( T ^ { - 1 } )$ , and is asymptotically dominated by the $\tilde { \mathcal { O } } ( T ^ { - 2 / 3 } )$ primary tracking error.

(iii) Regime 2: Approximation-Limited Phase. When log $( T ^ { 2 / 3 } ) > \log ( 1 + \epsilon _ { a p p } ^ { - 1 } )$ , the temperature locks to the approximation-dependent floor:

$$
\tau _ { T } = \frac { \Delta } { 2 \log ( C _ { \gamma } ( 1 + \epsilon _ { a p p } ^ { - 1 } ) ) } .
$$

Substituting this fixed temperature into the global inequality yields:

Primary Tracking Error

$$
= \mathcal { O } \Big ( \frac { 1 } { ( 1 - \gamma ) ^ { 8 } \kappa ^ { 6 } \Delta ^ { 8 / 3 } } \big ( \log ( C _ { \gamma } ( 1 + \epsilon _ { a p p } ^ { - 1 } ) ) \big ) ^ { 5 / 3 } T ^ { - 2 / 3 } \Big )
$$

$$
\leq \mathcal { O } \Big ( \frac { 1 } { ( 1 - \gamma ) ^ { 8 } \kappa ^ { 6 } \Delta ^ { 8 / 3 } } \Big ( \frac { 1 } { ( 1 - \gamma ) ^ { 5 / 3 } } + ( \log T ) ^ { 5 / 3 } \Big ) T ^ { - 2 / 3 } \Big ) ,
$$

Approximation

$$
\begin{array} { r l } & { \mathrm { B i a s } = \mathcal { O } \Big ( \frac { 1 } { ( 1 - \gamma ) ^ { 2 } \Delta ^ { 2 } } \log ( C _ { \gamma } ( 1 + \epsilon _ { a p p } ^ { - 1 } ) ) \epsilon _ { a p p } \Big ) } \\ & { \quad \quad = \mathcal { O } \Big ( \frac { 1 } { ( 1 - \gamma ) ^ { 2 } \Delta ^ { 2 } } \Big ( \frac { 1 } { 1 - \gamma } + \log ( 1 + \epsilon _ { a p p } ^ { - 1 } ) \Big ) \epsilon _ { a p p } \Big ) , } \end{array}
$$

$$
\mathrm { E n t r o p y ~ T a i l } \leq \frac { C _ { \gamma } } { ( 1 - \gamma ) ^ { 2 } } \exp \big ( - \log ( C _ { \gamma } ( 1 + \epsilon _ { a p p } ^ { - 1 } ) ) \big ) = \frac { \epsilon _ { a p p } / ( 1 + \epsilon _ { a p p } ) } { ( 1 - \gamma ) ^ { 2 } } \leq \frac { \epsilon _ { a p p } } { ( 1 - \gamma ) ^ { 2 } } ,
$$

$$
\begin{array} { r l r } {  { \mathrm { T r a n s i e n t ~ E r r o r } = \tilde { \mathcal { O } } \Big ( \frac { 1 } { ( 1 - \gamma ) ^ { 8 } \kappa ^ { 6 } \Delta ^ { 3 } } \big ( \log ( C _ { \gamma } ( 1 + \epsilon _ { a p p } ^ { - 1 } ) ) \big ) ^ { 2 } T ^ { - 1 } \Big ) } } \\ & { } & { \leq \tilde { \mathcal { O } } \Big ( \frac { 1 } { ( 1 - \gamma ) ^ { 8 } \kappa ^ { 6 } \Delta ^ { 3 } } ( \log ( C _ { \gamma } T ^ { 2 / 3 } ) ) ^ { 2 } T ^ { - 1 } \Big ) . ~ } \end{array}
$$

For the finite-time terms, the regime condition $\log ( 1 + \epsilon _ { a p p } ^ { - 1 } ) < \log ( T ^ { 2 / 3 } )$ implies

$$
\log ( C _ { \gamma } ( 1 + \epsilon _ { a p p } ^ { - 1 } ) ) < \log ( C _ { \gamma } ) + \log ( T ^ { 2 / 3 } ) ,
$$

which yields the stated finite-time envelopes. The fixed temperature floor prevents the $\tau _ { T } ^ { - 1 }$ approximation bias from growing with $T ,$ preserving the $\tilde { \mathcal { O } } ( \epsilon _ { a p p } )$ rate. Simultaneously, the exponential entropy tail is bounded by the linear scale of the approximation error, yielding $\mathcal { O } ( \epsilon _ { a p p } )$ . The transient convergence error remains asymptotically dominated by the primary tracking error.

(iv) Synthesis. Summing the evaluated components across the two regimes bounds the expected global unregularized performance gap by their respective envelopes, yielding:

$$
\begin{array} { r l } & { \displaystyle \frac { 1 } { T } \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } \big [ J _ { 0 } ( \pi _ { 0 } ^ { * } ) - J _ { 0 } ( \pi _ { t } ) \big ] \leq \mathcal { O } \left( \frac { ( 1 - \gamma ) ^ { - 5 / 3 } + ( \log T ) ^ { 5 / 3 } } { ( 1 - \gamma ) ^ { 8 } \kappa ^ { 6 } \Delta ^ { 8 / 3 } T ^ { 2 / 3 } } \right) } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad + \mathcal { O } \left( \frac { ( 1 - \gamma ) ^ { - 1 } + \log ( 1 + \epsilon _ { a p p } ^ { - 1 } ) } { ( 1 - \gamma ) ^ { 2 } \Delta ^ { 2 } } \epsilon _ { a p p } \right) . } \end{array}
$$

Finally, as $\epsilon _ { a p p }  0 , \log ( 1 + \epsilon _ { a p p } ^ { - 1 } ) = \Theta ( \log ( 1 / \epsilon _ { a p p } ) )$ and $\epsilon _ { a p p } \log ( 1 + \epsilon _ { a p p } ^ { - 1 } ) = \tilde { \mathcal { O } } ( \epsilon _ { a p p } )$ . Therefore, the overall convergence rate can be summarized as $\tilde { \mathcal { O } } ( T ^ { - 2 / 3 } ) + \tilde { \mathcal { O } } ( \epsilon _ { a p p } )$ , completing the proof. □

Proof of Theorem 10.2 (Unregularized Last-Iterate Convergence). We first consider $\epsilon _ { a p p } > 0$ . When $\epsilon _ { a p p } = 0$ , the approximation bias term vanishes and the result follows from the sample-limited calculation with $\begin{array} { r } { \tau _ { T } = \frac { \Delta } { 2 \log ( C _ { \gamma } T ^ { 2 / 3 } ) } } \end{array}$

(i) Global Unregularized Bound. Because the finite-time schedule $\tau _ { T }$ may initially exceed the temperature threshold required in Theorem 3.1, we apply the Universal Unregularized Suboptimality Bound (Corollary 3.1) directly to the terminal policy $\pi _ { T }$ . This guarantees the following global inequality holds for any $\tau _ { T } > 0$

$$
\mathbb { E } \left[ J _ { 0 } ( \pi _ { 0 } ^ { * } ) - J _ { 0 } ( \pi _ { T } ) \right] \leq \frac { 4 A _ { m a x } } { \Delta } \mathbb { E } [ \mathrm { G a p } _ { T } ^ { \dag } ( \tau _ { T } ) ] + C _ { t a i l } \exp \left( - \frac { \Delta } { 2 \tau _ { T } } \right) .
$$

Substituting the explicit regularized last-iterate bound from Theorem 9.2 isolates the three competing algorithmic components:

$$
\begin{array} { r l } & { \mathrm { T o t a l ~ E r r o r } \leq \underbrace { \mathcal { O } \left( \frac { A _ { m a x } } { \Delta ( 1 - \gamma ) ^ { 5 } \kappa ^ { 4 } \tau _ { T } ^ { 4 / 3 } } T ^ { - 1 / 3 } \right) } _ { \mathrm { P r i m a r y ~ O p t i m i z a t i o n ~ E r r o r } } + \underbrace { \mathcal { O } \left( \frac { A _ { m a x } } { \Delta ( 1 - \gamma ) ^ { 2 } \kappa \tau _ { T } } \sqrt { \epsilon _ { a p p } } \right) } _ { \mathrm { A p p r o x i m a t i o n ~ B i a s } } } \\ & { \quad \quad \quad + \underbrace { C _ { t a i l } \exp \left( - \frac { \Delta } { 2 \tau _ { T } } \right) } _ { \mathrm { E n t r o p y ~ T a i l } } . } \end{array}
$$

(ii) Regime 1: Sample-Limited Phase. When log $\cdot ( T ^ { 2 / 3 } ) \le \log ( 1 + \epsilon _ { a p p } ^ { - 1 } )$ , the temperature is tuned to $\begin{array} { r } { \tau _ { T } = \frac { \Delta } { 2 \log ( C _ { \gamma } T ^ { 2 / 3 } ) } } \end{array}$ . Substituting this alongside $\begin{array} { r } { A _ { m a x } \leq \frac { 1 } { 1 - \gamma } , C _ { t a i l } \triangleq \frac { A _ { m a x } C _ { \gamma } } { 1 - \gamma } \leq \frac { C _ { \gamma } } { ( 1 - \gamma ) ^ { 2 } } } \end{array}$ , and log $\begin{array} { r } { C _ { \gamma } = \mathcal { O } ( \frac { 1 } { 1 - \gamma } ) } \end{array}$ into the global inequality yields:

Primary Optimization Erro $\begin{array} { l } { \displaystyle \mathrm { r } = \mathcal { O } \Big ( \frac { 1 } { ( 1 - \gamma ) ^ { 6 } \kappa ^ { 4 } \Delta ^ { 7 / 3 } } ( \log ( C _ { \gamma } T ^ { 2 / 3 } ) ) ^ { 4 / 3 } T ^ { - 1 / 3 } \Big ) } \\ { \displaystyle = \mathcal { O } \Big ( \frac { 1 } { ( 1 - \gamma ) ^ { 6 } \kappa ^ { 4 } \Delta ^ { 7 / 3 } } \Big ( \frac { 1 } { ( 1 - \gamma ) ^ { 4 / 3 } } + ( \log T ) ^ { 4 / 3 } \Big ) T ^ { - 1 / 3 } \Big ) , } \end{array}$

$$
\begin{array} { r l } & { \mathrm { A p p r o x i m a t i o n ~ B i a s } = \displaystyle { \mathcal O } \Big ( \frac { 1 } { ( 1 - \gamma ) ^ { 3 } \kappa \Delta ^ { 2 } } \log ( C _ { \gamma } T ^ { 2 / 3 } ) \sqrt { \epsilon _ { a p p } } \Big ) } \\ & { \qquad \leq \displaystyle { \mathcal O } \Big ( \frac { 1 } { ( 1 - \gamma ) ^ { 3 } \kappa \Delta ^ { 2 } } \Big ( \frac { 1 } { 1 - \gamma } + \log ( 1 + \epsilon _ { a p p } ^ { - 1 } ) \Big ) \sqrt { \epsilon _ { a p p } } \Big ) , } \end{array}
$$

$$
\mathrm { E n t r o p y ~ T a i l } \le \frac { C _ { \gamma } } { ( 1 - \gamma ) ^ { 2 } } \exp \big ( - \log ( C _ { \gamma } T ^ { 2 / 3 } ) \big ) = \frac { 1 } { ( 1 - \gamma ) ^ { 2 } T ^ { 2 / 3 } } .
$$

The regime condition log $( T ^ { 2 / 3 } ) \le \log ( 1 + \epsilon _ { a p p } ^ { - 1 } )$ guarantees

$$
\log ( C _ { \gamma } T ^ { 2 / 3 } ) \le \log C _ { \gamma } + \log ( 1 + \epsilon _ { a p p } ^ { - 1 } ) ,
$$

which gives the stated approximation-bias envelope. The $C _ { \gamma }$ prefactor cancels in the exponential tail, which scales as $\mathcal { O } ( T ^ { - 2 / 3 } )$ and is asymptotically dominated by the $\tilde { \mathcal { O } } ( T ^ { - 1 / 3 } )$ primary optimization error.

(iii) Regime 2: Approximation-Limited Phase. When log $( T ^ { 2 / 3 } ) > \log ( 1 + \epsilon _ { a p p } ^ { - 1 } )$ , the temperature locks to the approximation-dependent floor:

$$
\tau _ { T } = \frac { \Delta } { 2 \log ( C _ { \gamma } ( 1 + \epsilon _ { a p p } ^ { - 1 } ) ) } .
$$

Substituting this fixed temperature into the global inequality yields:

$$
\begin{array} { l } { \displaystyle \mathrm { r } = { \mathcal O } \Big ( \frac 1 { ( 1 - \gamma ) ^ { 6 } \kappa ^ { 4 } \Delta ^ { 7 / 3 } } \big ( \log ( C _ { \gamma } ( 1 + \epsilon _ { a p p } ^ { - 1 } ) ) \big ) ^ { 4 / 3 } T ^ { - 1 / 3 } \Big ) } \\ { \displaystyle \le { \mathcal O } \Big ( \frac 1 { ( 1 - \gamma ) ^ { 6 } \kappa ^ { 4 } \Delta ^ { 7 / 3 } } \Big ( \frac 1 { ( 1 - \gamma ) ^ { 4 / 3 } } + ( \log T ) ^ { 4 / 3 } \Big ) T ^ { - 1 / 3 } \Big ) , } \end{array}
$$

$$
\begin{array} { r l } & { \mathrm { A p p r o x i m a t i o n ~ B i a s } = \displaystyle { \mathcal O } \Big ( \frac { 1 } { ( 1 - \gamma ) ^ { 3 } \kappa \Delta ^ { 2 } } \log ( C _ { \gamma } ( 1 + \epsilon _ { a p p } ^ { - 1 } ) ) \sqrt { \epsilon _ { a p p } } \Big ) } \\ & { \quad \quad \quad = \displaystyle { \mathcal O } \Big ( \frac { 1 } { ( 1 - \gamma ) ^ { 3 } \kappa \Delta ^ { 2 } } \Big ( \frac { 1 } { 1 - \gamma } + \log ( 1 + \epsilon _ { a p p } ^ { - 1 } ) \Big ) \sqrt { \epsilon _ { a p p } } \Big ) , } \end{array}
$$

$$
\mathrm { T a i l } \le \frac { C _ { \gamma } } { ( 1 - \gamma ) ^ { 2 } } \exp \big ( - \log ( C _ { \gamma } ( 1 + \epsilon _ { a p p } ^ { - 1 } ) ) \big ) = \frac { \epsilon _ { a p p } / ( 1 + \epsilon _ { a p p } ) } { ( 1 - \gamma ) ^ { 2 } } \le \frac { \epsilon _ { a p p } } { ( 1 - \gamma ) ^ { 2 } } .
$$

For the finite-time error, the regime condition log $\begin{array} { r } { \mathrm { ; } ( 1 + \epsilon _ { a p p } ^ { - 1 } ) < \log ( T ^ { 2 / 3 } ) } \end{array}$ implies

$$
\log ( C _ { \gamma } ( 1 + \epsilon _ { a p p } ^ { - 1 } ) ) < \log ( C _ { \gamma } ) + \log ( T ^ { 2 / 3 } ) ,
$$

which yields the stated finite-time envelope. The fixed temperature floor prevents the $\tau _ { T } ^ { - 1 }$ factor from growing with $T _ { i }$ , preserving the $\tilde { \mathcal { O } } ( \sqrt { \epsilon _ { a p p } } )$ approximation-bias rate. Simultaneously, the exponential entropy tail is bounded by $\mathcal { O } ( \epsilon _ { a p p } )$ and is therefore asymptotically subsumed by the slower $\tilde { \mathcal { O } } ( \sqrt { \epsilon _ { a p p } } )$ approximation bias.

(iv) Synthesis. Summing these evaluated components across the two regimes bounds the expected last-iterate global unregularized performance gap by their respective envelopes, yielding:

$$
\begin{array} { r l } & { \mathbb { E } \left[ J _ { 0 } ( \pi _ { 0 } ^ { * } ) - J _ { 0 } ( \pi _ { T } ) \right] \leq { \mathcal O } \left( \displaystyle \frac { ( 1 - \gamma ) ^ { - 4 / 3 } + ( \log T ) ^ { 4 / 3 } } { ( 1 - \gamma ) ^ { 6 } \kappa ^ { 4 } \Delta ^ { 7 / 3 } T ^ { 1 / 3 } } \right) } \\ & { \quad \quad \quad \quad \quad \quad + { \mathcal O } \left( \displaystyle \frac { ( 1 - \gamma ) ^ { - 1 } + \log ( 1 + \epsilon _ { a p p } ^ { - 1 } ) } { ( 1 - \gamma ) ^ { 3 } \kappa \Delta ^ { 2 } } \sqrt { \epsilon _ { a p p } } \right) . } \end{array}
$$

Finally, as $\epsilon _ { a p p }  0$ , lo $\displaystyle \mathop { ? } \bigl ( 1 + \epsilon _ { a p p } ^ { - 1 } \bigr ) = \Theta \bigl ( \log ( 1 / \epsilon _ { a p p } ) \bigr )$ and $\sqrt { \epsilon _ { a p p } } \log ( 1 + \epsilon _ { a p p } ^ { - 1 } ) = \tilde { \mathcal { O } } ( \sqrt { \epsilon _ { a p p } } )$ . Therefore,

the overall convergence rate can be summarized as $\tilde { \mathcal { O } } ( T ^ { - 1 / 3 } ) + \tilde { \mathcal { O } } ( \sqrt { \epsilon _ { a p p } } )$ , completing the proof. □

Proof of Theorem 10.3 (Unregularized Convergence in Tabular Setting). To apply Corollary C.1, we systematically verify the required structural and trajectory-dependent assumptions for the tabular setting with one-hot features.

(i) Verifying Assumption 1 (Bounded Features): Because $\phi ( s , a )$ is the one-hot basis vector, with Euclidean norm 1 for all $( s , a )$ , Assumption 1 holds trivially with $B _ { \phi } = 1$

(ii) Verifying Assumption 2 (Bounded Q-value Second Moment): We construct the unbiased estimator $\hat { Q } _ { t }$ using a geometrically stopped Monte Carlo rollout and evaluate the entropy contribution in its statewise form $H ( \pi _ { t } ( \cdot | s ) )$ . Let H denote the random rollout horizon, with $\mathbb { P } ( H \geq k ) = \gamma ^ { k }$ Since the original reward is bounded in [0, 1] and $H ( \pi _ { t } ( \cdot | s ) ) \leq \log | \mathcal { A } |$ , every statewise regularized reward contribution is bounded by $1 + \tau _ { m a x } \log | \mathcal { A } |$ . Hence,

$$
| \hat { Q } _ { t } | \leq \big ( 1 + \tau _ { m a x } \log | \mathcal { A } | \big ) ( H + 1 ) .
$$

For the geometric stopping rule, $\begin{array} { r } { \mathbb { E } [ ( H + 1 ) ^ { 2 } ] = \frac { 1 + \gamma } { ( 1 - \gamma ) ^ { 2 } } \le \frac { 2 } { ( 1 - \gamma ) ^ { 2 } } } \end{array}$ . Therefore,

$$
\mathbb { E } [ \hat { Q } _ { t } ^ { 2 } \mid \mathcal { F } _ { t } , s _ { t } , a _ { t } ] \leq 2 \left( \frac { 1 + \tau _ { m a x } \log | \cal { A } | } { 1 - \gamma } \right) ^ { 2 } = 2 V _ { m a x } ^ { 2 } .
$$

Thus, Assumption 2 holds with $Q _ { m a x } = \sqrt { 2 } V _ { m a x } = \mathcal { O } ( ( 1 - \gamma ) ^ { - 1 } )$

(iii) Verifying Assumption 12 (Positive-Definite Moment under Exploratory Training Measure): Under the one-hot representation,

$$
\bar { \Sigma } _ { u n c } ( d _ { \nu } ^ { \pi _ { t } } ) = \mathbb { E } _ { d _ { \nu } ^ { \pi _ { t } } } \left[ \phi ( s , a ) \phi ( s , a ) ^ { \top } \right] = \mathrm { d i a g } \left( d _ { \nu } ^ { \pi _ { t } } ( s , a ) \right) .
$$

Because $d _ { \nu } ^ { \pi _ { t } } ( s , a ) \geq ( 1 - \gamma ) \nu ( s , a )$ , we have

$$
\bar { \Sigma } _ { u n c } ( d _ { \nu } ^ { \pi _ { t } } ) \succeq ( 1 - \gamma ) \operatorname* { m i n } _ { s , a } \nu ( s , a ) I = \kappa _ { e x p } I .
$$

Thus, Assumption 12 holds with $\begin{array} { r } { \kappa _ { e x p } = ( 1 - \gamma ) \operatorname* { m i n } _ { s , a } \nu ( s , a ) > 0 } \end{array}$

(iv) Verifying Assumption $1 4 \ ( L _ { \infty }$ Critic Approximation Error): Because the one-hot features span the entire space of state-action functions, the regularized Q-function is represented exactly. For every training policy $\pi _ { t }$ , we have for all $( s , a )$

$$
\theta _ { \tau } ^ { * } ( \omega _ { t } ) _ { ( s , a ) } = Q _ { \tau } ^ { \pi _ { t } } ( s , a ) ,
$$

and hence $\epsilon _ { t } ( s , a ) = Q _ { \tau } ^ { \pi _ { t } } ( s , a ) - \theta _ { \tau } ^ { * } ( \omega _ { t } ) ^ { \top } \phi ( s , a ) = 0$ . Thus, Assumption 14 holds with $\epsilon _ { \infty } = 0$

(v) Synthesis: The preceding arguments verify Assumptions 1, 2, 12, and 14, while Assumption 5 holds by hypothesis. Therefore, Corollary C.1 applies. Since $\epsilon _ { \infty } = 0$ , its two-stage last-iterate

temperature reduces by convention to

$$
\tau _ { T } ^ { l a s t } = \frac { \Delta } { 2 \log ( C _ { \gamma } T ^ { 2 / 3 } ) } = \tau _ { T } ^ { a v g } .
$$

Thus, the common temperature $\tau _ { T }$ can be used as stated. Substituting $\epsilon _ { \infty } = 0$ and $\kappa = \kappa _ { e x p }$ into Corollary C.1 yields the stated convergence bounds, completing the proof. □

## F Unregularized Convergence via Linear Entropy Penalty (Stochastic Regime)

In this appendix, we provide the unregularized convergence analysis relying on the standard linear entropy penalty for Algorithm 1 in the Stochastic Regime. While these convergence rates $( \tilde { \mathcal { O } } ( T ^ { - 1 / 3 } ) )$ are slower than the optimal $\tilde { \mathcal { O } } ( T ^ { - 1 } )$ bounds established in the main text via the exponential translation bounds, this analysis remains theoretically valuable. Crucially, these bounds do not require the Minimal Action Gap condition (Assumption 5). Therefore, in environments lacking a uniformly positive action margin—where our exponential translation bounds are unavailable—this linear entropy penalty framework provides a robust theoretical fallback to guarantee convergence for the parametric unregularized performance gap.

## F.1 Average-Iterate Unregularized Convergence

We map the average-iterate regularized performance gap bound established in Section 6.2 to the unregularized objective. We establish the unregularized convergence rate by tuning the fixed temperature τ to optimally balance the optimization error against the induced entropy penalty.

Theorem F.1 (Average-Iterate Unregularized Convergence). For Algorithm 1, suppose that Assumptions $1 \mathrm { - } \mathit { 4 }$ and $6 \mathrm { - } 7$ hold. Define the two-stage temperature:

$$
\tau _ { T } \triangleq \operatorname* { m a x } \left( \left( \frac { \log T } { ( 1 - \gamma ) ^ { 4 } \kappa ^ { 6 } \lambda ^ { 5 / 2 } T } \right) ^ { 1 / 3 } , \frac { \sqrt { \epsilon _ { a p p } } } { \lambda } \right) .
$$

If T is suficiently large and $\epsilon _ { a p p }$ is suficiently small such that $\tau _ { T } \le \tau _ { m a x }$ , then by choosing the step-size parameters as in Theorem 6.2 evaluated at $\tau = \tau _ { T }$ subject to the stated conditions, the expected average parametric unregularized performance gap satisfies the convergence rate:

$$
\begin{array} { r l } & { \frac { 1 } { T } \displaystyle \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } \big [ J _ { 0 } ( \pi _ { \omega _ { 0 } ^ { * } } ) - J _ { 0 } ( \pi _ { t } ) \big ] \leq \mathcal { O } \left( \frac { ( \log T ) ^ { 1 / 3 } } { ( 1 - \gamma ) ^ { 7 / 3 } \kappa ^ { 2 } \lambda ^ { 5 / 6 } T ^ { 1 / 3 } } \right) + \mathcal { O } \left( \frac { \sqrt { \epsilon _ { a p p } } } { ( 1 - \gamma ) \lambda } \right) } \\ & { \quad \quad \quad \quad = \tilde { \mathcal { O } } \left( T ^ { - 1 / 3 } \right) + \mathcal { O } ( \sqrt { \epsilon _ { a p p } } ) . } \end{array}
$$

Proof. (i) Structural Error Bound. Let $\pi _ { \omega _ { 0 } ^ { * } }$ be the unregularized parametrically optimal policy.

Because the entropy is non-negative, evaluating this optimal policy under the regularized objective yields $J _ { 0 } ( \pi _ { \omega _ { 0 } ^ { * } } ) ~ \le ~ J _ { \tau } ( \pi _ { \omega _ { 0 } ^ { * } } )$ . By the definition of the regularized parametric optimum, we have $J _ { \tau } ( \pi _ { \omega _ { 0 } ^ { * } } ) \leq J _ { \tau } ( \pi _ { \omega _ { \tau } ^ { * } } )$ . Thus, for any policy $\pi _ { t } ,$ the parametric unregularized performance gap is bounded by the sum of the parametric regularized performance gap and the entropy penalty:

$$
\begin{array} { r l } & { J _ { 0 } ( \pi _ { \omega _ { 0 } ^ { * } } ) - J _ { 0 } ( \pi _ { t } ) \leq J _ { \tau } ( \pi _ { \omega _ { \tau } ^ { * } } ) - J _ { 0 } ( \pi _ { t } ) } \\ & { \qquad = \Big ( J _ { \tau } ( \pi _ { \omega _ { \tau } ^ { * } } ) - J _ { \tau } ( \pi _ { t } ) \Big ) + \Big ( J _ { \tau } ( \pi _ { t } ) - J _ { 0 } ( \pi _ { t } ) \Big ) } \\ & { \qquad = \mathrm { G a p } _ { t } + \frac { \tau } { 1 - \gamma } \mathbb { E } _ { s \sim d ^ { \pi _ { t } } } [ H ( \pi _ { t } ( \cdot | s ) ) ] . } \end{array}
$$

Taking the expectation and averaging this bound over the trajectory from $t = 0$ to $T - 1$ yields:

$$
\frac { 1 } { T } \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } \left[ J _ { 0 } ( \pi _ { \omega _ { 0 } ^ { * } } ) - J _ { 0 } ( \pi _ { t } ) \right] \leq \frac { 1 } { T } \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } [ \mathrm { G a p } _ { t } ] + \frac { \tau } { 1 - \gamma } \left( \frac { 1 } { T } \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } \left[ \mathbb { E } _ { s \sim d ^ { \pi _ { t } } } [ H ( \pi _ { t } ( \cdot | s ) ) ] \right] \right)
$$

The policy entropy is universally bounded by $H ( \pi _ { t } ) \leq \log | \mathcal { A } | = \mathcal { O } ( 1 )$

(ii) Regularized Order Substitution. Substituting the average regularized performance gap bound established in Theorem 6.2, the average parametric unregularized performance gap is bounded by:

$$
\frac { 1 } { T } \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } \big [ J _ { 0 } ( \pi _ { \omega _ { 0 } ^ { * } } ) - J _ { 0 } ( \pi _ { t } ) \big ] \leq \underbrace { \mathcal { O } \left( \frac { 1 } { ( 1 - \gamma ) ^ { 5 } \kappa ^ { 6 } \lambda ^ { 5 / 2 } \tau ^ { 2 } } \frac { \log T } { T } \right) } _ { \mathrm { F i n i t e - T i m e ~ O p t i m i z a t i o n ~ E r r o r } } + \underbrace { \mathcal { O } \left( \frac { \epsilon _ { a p p } } { ( 1 - \gamma ) \lambda ^ { 2 } \tau } \right) } _ { \mathrm { A p p r o x i m a t i o n ~ B i a s } } + \underbrace { \mathcal { O } \left( \frac { \tau } { 1 - \gamma } \right) } _ { \mathrm { E n t r o p y ~ P e n a l t y } }
$$

(iii) Two-Stage Hyperparameter Balancing. To identify the appropriate temperature scale, we balance the dominant regularized optimization error against the Entropy Penalty.

• Sample-Limited Regime: When the average optimization error dominates the approximation bias, balancing the former against the entropy penalty, $\begin{array} { r } { \frac { 1 } { ( 1 - \gamma ) ^ { 5 } \kappa ^ { 6 } \lambda ^ { 5 / 2 } \tau ^ { 2 } } \frac { \log T } { T } = \frac { \tau } { 1 - \gamma } , } \end{array}$ yields the tuned temperature $\begin{array} { r } { \tau = \big ( \frac { \log T } { ( 1 - \gamma ) ^ { 4 } \kappa ^ { 6 } \lambda ^ { 5 / 2 } T } \big ) ^ { 1 / 3 } } \end{array}$ . Substituting this choice gives an unregularized performance gap bound of $\begin{array} { r } { \mathcal { O } \big ( \frac { ( \log T ) ^ { 1 / 3 } } { ( 1 - \gamma ) ^ { 7 / 3 } \kappa ^ { 2 } \lambda ^ { 5 / 6 } T ^ { 1 / 3 } } \big ) } \end{array}$

• Approximation-Limited Regime: When the approximation bias dominates the average optimization error, balancing $\begin{array} { r } { \frac { \epsilon _ { a p p } } { ( 1 - \gamma ) \lambda ^ { 2 } \tau } = \frac { \tau } { 1 - \gamma } } \end{array}$ yields the tuned temperature $\begin{array} { r } { \tau = \frac { \sqrt { \epsilon _ { a p p } } } { \lambda } } \end{array}$ . Substituting this choice gives an unregularized performance gap bound of $\mathcal { O } \left( \frac { \sqrt { \epsilon _ { a p p } } } { ( 1 - \gamma ) \lambda } \right)$

(iv) Synthesis. By setting the tuned temperature to the maximum of these two candidate temperatures, the expected average parametric unregularized performance gap is bounded by the sum of these respective envelopes, completing the proof. □

## F.2 Last-Iterate Unregularized Convergence

We map the last-iterate regularized performance gap bound in Section 6.3 to the unregularized objective. Because this bound avoids the log T penalty, we establish a sharper unregularized convergence rate (by a factor of $( \log T ) ^ { 1 / 3 } )$ when tuning the fixed temperature τ to balance this terminal optimization error against the induced entropy penalty.

Theorem F.2 (Last-Iterate Unregularized Convergence). For Algorithm 1 with T replaced by $T { + 1 }$ suppose that Assumptions $1 \mathrm { - } \mathit { 4 }$ and 6–7 hold. Consider the two-stage temperature:

$$
\tau _ { T } \triangleq \operatorname* { m a x } \left( \left( \frac { 1 } { ( 1 - \gamma ) ^ { 4 } \kappa ^ { 6 } \lambda ^ { 5 / 2 } T } \right) ^ { 1 / 3 } , \frac { \sqrt { \epsilon _ { a p p } } } { \lambda } \right) .
$$

$I f T$ is suficiently large and $\epsilon _ { a p p }$ is suficiently small such that $\tau _ { T } \le \tau _ { m a x }$ , then by choosing the step-size parameters as in Theorem 6.3 evaluated at $\tau = \tau _ { T }$ subject to the stated conditions, the expected last-iterate parametric unregularized performance gap satisfies the convergence rate:

$$
\begin{array} { r l } & { \mathbb { E } \left[ J _ { 0 } ( \pi _ { \omega _ { 0 } ^ { * } } ) - J _ { 0 } ( \pi _ { T } ) \right] \leq \mathcal { O } \left( \frac { 1 } { ( 1 - \gamma ) ^ { 7 / 3 } \kappa ^ { 2 } \lambda ^ { 5 / 6 } T ^ { 1 / 3 } } \right) + \mathcal { O } \left( \frac { \sqrt { \epsilon _ { a p p } } } { ( 1 - \gamma ) \lambda } \right) } \\ & { \quad \quad \quad \quad = \mathcal { O } \left( T ^ { - 1 / 3 } \right) + \mathcal { O } ( \sqrt { \epsilon _ { a p p } } ) . } \end{array}
$$

Proof. (i) Structural Error Bound. We apply the unregularized bound in Step (i) of the proof of Theorem F.1 to the terminal policy $\pi _ { T }$ . Taking the expectation, the parametric unregularized performance gap is bounded by the sum of the regularized performance gap and the entropy penalty:

$$
\mathbb { E } \big [ J _ { 0 } ( \pi _ { \omega _ { 0 } ^ { * } } ) - J _ { 0 } ( \pi _ { T } ) \big ] \leq \mathbb { E } [ \mathrm { G a p } _ { T } ] + \frac { \tau } { 1 - \gamma } \mathbb { E } \Big [ \mathbb { E } _ { s \sim d ^ { \pi _ { T } } } [ H ( \pi _ { T } ( \cdot | s ) ) ] \Big ] .
$$

(ii) Regularized Order Substitution. We substitute the last-iterate convergence bound for $\mathbb { E } [ \mathrm { G a p } _ { T } ]$ established in Theorem 6.3. Recalling that the policy entropy is bounded by $H ( \pi _ { T } ) \leq$ $\log | \mathcal { A } | = \mathcal { O } ( 1 )$ , the final parametric unregularized performance gap is bounded by:

$$
\mathbb { E } \left[ J _ { 0 } ( \pi _ { \omega _ { 0 } ^ { * } } ) - J _ { 0 } ( \pi _ { T } ) \right] \leq \underbrace { \mathcal { O } \left( \frac { 1 } { ( 1 - \gamma ) ^ { 5 } \kappa ^ { 6 } \lambda ^ { 5 / 2 } \tau ^ { 2 } T } \right) } _ { \mathrm { F i n i t e - T i m e ~ O p t i m i z a t i o n ~ E r r o r } } + \underbrace { \mathcal { O } \left( \frac { \epsilon _ { a p p } } { ( 1 - \gamma ) \lambda ^ { 2 } \tau } \right) } _ { \mathrm { A p p r o x i m a t i o n ~ B i a s } } + \underbrace { \mathcal { O } \left( \frac { \tau } { 1 - \gamma } \right) } _ { \mathrm { E n t r o p y ~ P e n a l t y } } .
$$

(iii) Two-Stage Hyperparameter Balancing. We identify the appropriate temperature scale analogously to the average-iterate case:

• Sample-Limited Regime: When the last-iterate optimization error dominates the approximation bias, balancing at $\tau = \left( ( 1 - \gamma ) ^ { 4 } \kappa ^ { 6 } \lambda ^ { 5 / 2 } T \right) ^ { - 1 / 3 }$ yields an unregularized performance gap bound of $\mathcal { O } \left( \frac { 1 } { ( 1 - \gamma ) ^ { 7 / 3 } \kappa ^ { 2 } \lambda ^ { 5 / 6 } T ^ { 1 / 3 } } \right)$

• Approximation-Limited Regime: When the approximation bias dominates the last-iterate optimization error, balancing at $\begin{array} { r } { \tau = \frac { \sqrt { \epsilon _ { a p p } } } { \lambda } } \end{array}$ yields an unregularized performance gap bound of $\mathcal { O } \left( \frac { \sqrt { \epsilon _ { a p p } } } { ( 1 - \gamma ) \lambda } \right)$

(iv) Synthesis. By setting the tuned temperature to the maximum of these two candidate temperatures, the final expected parametric unregularized performance gap is bounded by the sum of these respective envelopes, completing the proof. □

Remark F.1 (Equivalence of Unregularized Average and Last-Iterate Rates). Echoing the equivalence for regularized bounds in Remark 6.2, the unregularized last-iterate convergence rate (Theorem F.2) asymptotically matches the average-iterate rate (Theorem F.1) up to a logarithmic factor, both achieving a $\tilde { \mathcal { O } } ( T ^ { - 1 / 3 } )$ dependence.

## F.3 Global Unregularized Convergence

By invoking the Global-to-Parametric Joint Concentrability (Assumption 8), we extend the parametric unregularized convergence bounds to the true global optimum. Because the representation penalty from the Global Class Approximation Error (Lemma 7.1) shares the identical $\mathcal { O } ( 1 / \tau )$ dependence as the standard approximation bias, incorporating this representation penalty leads to identical unregularized convergence rates as in Theorems F.1 and F.2.

Corollary F.1 (Global Unregularized Convergence). Suppose that Assumptions 1–4 and 6–8 hold. By tuning the temperature as established in Theorems F.1 and F.2, the following unregularized convergence rates hold with respect to the global optimum. For the last-iterate convergence:

$$
\mathbb { E } \left[ J _ { 0 } ( \pi _ { 0 } ^ { * } ) - J _ { 0 } ( \pi _ { T } ) \right] \leq \mathcal { O } \left( \frac { 1 } { ( 1 - \gamma ) ^ { 7 / 3 } \kappa ^ { 2 } \lambda ^ { 5 / 6 } T ^ { 1 / 3 } } \right) + \mathcal { O } \left( \frac { \sqrt { \epsilon _ { a p p } } } { ( 1 - \gamma ) \lambda } \right) .
$$

For the average-iterate convergence:

$$
\frac { 1 } { T } \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } \big [ J _ { 0 } ( \pi _ { 0 } ^ { * } ) - J _ { 0 } ( \pi _ { t } ) \big ] \leq \mathcal { O } \left( \frac { ( \log T ) ^ { 1 / 3 } } { ( 1 - \gamma ) ^ { 7 / 3 } \kappa ^ { 2 } \lambda ^ { 5 / 6 } T ^ { 1 / 3 } } \right) + \mathcal { O } \left( \frac { \sqrt { \epsilon _ { a p p } } } { ( 1 - \gamma ) \lambda } \right) .
$$

Proof. Because the policy entropy is non-negative, $J _ { 0 } ( \pi _ { 0 } ^ { * } ) \le J _ { \tau } ( \pi _ { 0 } ^ { * } )$ . Since $\pi _ { \tau } ^ { * }$ maximizes the regularized objective globally, $J _ { \tau } ( \pi _ { 0 } ^ { * } ) \le J _ { \tau } ( \pi _ { \tau } ^ { * } )$ . Taking the expectation over the algorithmic trajectory, the global unregularized performance gap decomposes into the representation penalty, the regularized parametric gap, and the entropy penalty:

$$
\begin{array} { r l } & { \mathbb { E } \big [ J _ { 0 } ( \pi _ { 0 } ^ { * } ) - J _ { 0 } ( \pi _ { T } ) \big ] \leq \mathbb { E } \big [ J _ { \tau } ( \pi _ { \tau } ^ { * } ) - J _ { 0 } ( \pi _ { T } ) \big ] } \\ & { \qquad = \Big ( J _ { \tau } ( \pi _ { \tau } ^ { * } ) - J _ { \tau } ( \pi _ { \omega _ { \tau } ^ { * } } ) \Big ) + \mathbb { E } [ \mathrm { G a p } _ { T } ] + \frac { \tau } { 1 - \gamma } \mathbb { E } \Big [ \mathbb { E } _ { s \sim d ^ { \pi _ { T } } } [ H ( \pi _ { T } ( \cdot | s ) ) ] \Big ] . } \end{array}
$$

By Lemma 7.1, the representation penalty is bounded by $\frac { C _ { c l a s s } } { \tau } \epsilon _ { a p p }$ . Thus:

$$
\mathbb { E } \big [ J _ { 0 } ( \pi _ { 0 } ^ { * } ) - J _ { 0 } ( \pi _ { T } ) \big ] \leq \frac { C _ { c l a s s } } { \tau } \epsilon _ { a p p } + \mathbb { E } [ \mathrm { G a p } _ { T } ] + \frac { \tau } { 1 - \gamma } \mathbb { E } \Big [ \mathbb { E } _ { s \sim d ^ { \pi _ { T } } } [ H ( \pi _ { T } ( \cdot | s ) ) ] \Big ] .
$$

Because $\begin{array} { r } { C _ { c l a s s } = \mathcal { O } ( \frac { 1 } { ( 1 - \gamma ) \lambda ^ { 2 } } ) } \end{array}$ , the representation penalty $\frac { C _ { c l a s s } } { \tau } \epsilon _ { a p p }$ shares the same dependence on τ as well as on $( 1 - \gamma ) ^ { - 1 }$ and λ as the approximation bias within $\mathbb { E } [ \mathrm { G a p } _ { T } ]$ . The same twostage temperature applies to balance the terms as in Theorem F.2, preserving the last-iterate rate. The same argument applies to the averaged sequence, preserving the average-iterate rate from Theorem F.1, completing the proof. □

## G Unifying Single-Loop and Double-Loop via Warm-Start Critic

In this section, we generalize our analysis to a warm-start double-loop architecture (Algorithm 5). In this setting, the algorithm performs T outer-loop actor updates, and for each fixed actor parameter $\omega _ { t } .$ , runs N inner-loop critic SGD steps. Instead of independently re-initializing the critic at each outer loop (as in standard, cold-start double-loop algorithms like Algorithm 2), we warm-start the inner critic using the final parameter of the previous loop $( \theta _ { t } ^ { ( 0 ) } = \theta _ { t } )$ . The total number of SGD updates is defined as $T _ { t o t a l } = T \times N$

This architecture mathematically unifies the single-loop $( N = 1 , T = T _ { t o t a l } )$ and double-loop $( T = \sqrt { T _ { t o t a l } } , N = \sqrt { T _ { t o t a l } } )$ algorithms. By analyzing the geometric contraction over the N inner steps, we prove a structural invariance across inner-loop sizes $N .$ , based on our PMD analysis in Part II (which applies to the Deterministic Regime as well as the Stochastic Regime). First, under our primary Exponential Translation mechanism, we demonstrate that any intermediate innerloop choice $N \in [ 1 , T _ { t o t a l } ^ { 2 / 9 } ]$ yields the same $\tilde { \mathcal { O } } ( T _ { t o t a l } ^ { - 2 / 3 } )$ average-iterate unregularized convergence rate. Subsequently, to reveal the underlying barrier that the exponential translation bypasses, we analyze the algorithm under the standard linear entropy penalty. We establish that without the exponential mechanism, the PMD geometry leads to a $\tilde { \mathcal { O } } ( T _ { t o t a l } ^ { - 1 / 4 } )$ convergence rate, remaining invariant across the shifted window $N \in [ 1 , \sqrt { T _ { t o t a l } } ]$

Throughout this appendix, when invoking trajectory-dependent assumptions from the main text, we assume their corresponding bounds hold along the training policies and inner-loop samples generated by Algorithm 5, with the same structural constants.

Algorithm 5 Warm-Start Double-Loop, Entropy-Regularized, Uncentered Natural Actor-Critic   
Input: Initial actor parameter ω<sub>0</sub> satisfying $\begin{array} { r } { \| \omega _ { 0 } \| _ { 2 } \le \frac { B _ { \theta } } { \tau _ { m a x } } } \end{array}$ , initial critic parameter $\theta _ { 0 }$ , temperature   
$\tau \in ( 0 , \tau _ { m a x } ]$ , critic projection radius $R _ { \theta } \left( = B _ { \theta } \right)$ , inner-loop iterations $N ,$ outer-loop iterations   
T, step-size schedules $\{ \eta _ { t } \} _ { t \ge 0 }$ and $\{ \alpha _ { t } \} _ { t \ge 0 }$   
1: for $t = 0 , 1 , \ldots , T - 1$ do   
2: Actor Freeze: Fix the current policy $\pi _ { t } \triangleq \pi _ { \omega _ { t } }$   
3: Critic Warm-Start: Initialize the inner critic parameter with the previous outer-loop out  
put: $\theta _ { t } ^ { ( 0 ) } = \theta _ { t }$   
4: for $k = 0 , 1 , \ldots , N - 1$ do   
5: Sampling: Draw a state-action pair $( s _ { k } , a _ { k } ) \sim d ^ { \pi _ { t } } \times \pi _ { t }$   
6: Evaluation: Obtain an unbiased estimate $\hat { Q } _ { k }$ of the regularized action-value $Q _ { \tau } ^ { \pi _ { t } } ( s _ { k } , a _ { k } )$   
7: Critic SGD: Compute the stochastic critic gradient $g _ { t } ^ { c r } ( s _ { k } , a _ { k } ) ~ = ~ ( \theta _ { t } ^ { ( k ) \top } \phi ( s _ { k } , a _ { k } ) ~ -$   
${ \hat { Q } } _ { k } \mathbf { ) } \phi ( s _ { k } , a _ { k } )$ and update via constrained SGD:   
$\boldsymbol { \theta } _ { t } ^ { ( k + 1 ) } = \Pi _ { R _ { \boldsymbol { \theta } } } \Big [ \boldsymbol { \theta } _ { t } ^ { ( k ) } - \alpha _ { t } g _ { t } ^ { c r } ( \boldsymbol { s } _ { k } , \boldsymbol { a } _ { k } ) \Big ] .$   
8: end for   
9: Critic Assignment: Set the outer-loop critic parameter to the final inner-loop iterate:   
$\theta _ { t + 1 } = \theta _ { t } ^ { ( N ) }$   
10: Actor Update: Update the actor parameter:   
$\omega _ { t + 1 } = \omega _ { t } + \eta _ { t } \big ( \theta _ { t + 1 } - \tau \omega _ { t } \big )$   
11: end for   
Output: The output policy sequence $\{ \pi _ { t } \} _ { t = 0 } ^ { T } .$

## G.1 Warm-Start Critic Tracking Error

To analyze the warm-start double-loop algorithm, standard fractional variants of Chung’s lemma (Lemma 9.3) provide tight pointwise bounds but sufer from artificial horizon inflation when integrated directly over an outer-loop trajectory. The following lemma introduces a fractional summation bound to safely isolate the initialization footprint, a mechanism that is critical for establishing loop invariance across intermediate inner-loop sizes N.

Lemma G.1 (Fractional Chung’s Summation Bound). Let $\{ x _ { t } \} _ { t = 0 } ^ { \infty }$ be a non-negative sequence satisfying the recursive bound:

$$
x _ { t + 1 } \leq \left( 1 - \frac { c } { ( t + t _ { 0 } ) ^ { 2 / 3 } } \right) x _ { t } + \frac { K } { ( t + t _ { 0 } ) ^ { 4 / 3 } } , \quad t \geq 0 ,
$$

where $c > 0 , K \geq 0$ , and the initial ofset satisfies $t _ { 0 } \geq \operatorname* { m a x } \left( 1 , ( \frac { 4 } { 3 c } ) ^ { 3 } \right)$ . The average of the sequence is bounded $b y \colon$

$$
\frac { 1 } { T } \sum _ { t = 0 } ^ { T - 1 } x _ { t } \le \frac { 2 t _ { 0 } ^ { 2 / 3 } } { c T } x _ { 0 } + \frac { 6 K } { c T } ( T + t _ { 0 } ) ^ { 1 / 3 } .
$$

Proof. We rearrange the recurrence to isolate $x _ { t }$ alongside the fractional step size:

$$
x _ { t } \leq \frac { ( t + t _ { 0 } ) ^ { 2 / 3 } } { c } ( x _ { t } - x _ { t + 1 } ) + \frac { K } { c ( t + t _ { 0 } ) ^ { 2 / 3 } } .
$$

Summing this inequality from $t = 0$ to $T - 1$ , we apply summation by parts to the first term:

$$
\sum _ { t = 0 } ^ { T - 1 } x _ { t } \leq \frac { t _ { 0 } ^ { 2 / 3 } } { c } x _ { 0 } - \frac { ( T - 1 + t _ { 0 } ) ^ { 2 / 3 } } { c } x _ { T } + \sum _ { t = 1 } ^ { T - 1 } x _ { t } \frac { ( t + t _ { 0 } ) ^ { 2 / 3 } - ( t - 1 + t _ { 0 } ) ^ { 2 / 3 } } { c } + \frac { K } { c } \sum _ { t = 0 } ^ { T - 1 } \frac { 1 } { ( t + t _ { 0 } ) ^ { 2 / 3 } } .
$$

We drop the non-positive $- \boldsymbol { x } _ { T }$ term. By the concavity of the fractional power, the diference is bounded by its derivative: $( t + t _ { 0 } ) ^ { 2 / 3 } - ( t - 1 + t _ { 0 } ) ^ { 2 / 3 } \leq \frac { 2 } { 3 } ( t - 1 + t _ { 0 } ) ^ { - 1 / 3 }$ . Because $t \geq 1$ , this is bounded by $\frac { 2 } { 3 } t _ { 0 } ^ { - 1 / 3 }$ . Furthermore, with $t _ { 0 } \geq 1$ , the fractional sum is bounded by: $\begin{array} { r } { \sum _ { t = 0 } ^ { T - 1 } ( t + t _ { 0 } ) ^ { - 2 / 3 } \leq } \end{array}$ $\begin{array} { r } { \int _ { 0 } ^ { T } ( x + t _ { 0 } - 1 ) ^ { - 2 / 3 } d x \leq 3 ( T + t _ { 0 } ) ^ { 1 / 3 } } \end{array}$ . Substituting these bounds gives:

$$
\sum _ { t = 0 } ^ { T - 1 } x _ { t } \leq \frac { t _ { 0 } ^ { 2 / 3 } } { c } x _ { 0 } + \frac { 2 } { 3 c t _ { 0 } ^ { 1 / 3 } } \sum _ { t = 1 } ^ { T - 1 } x _ { t } + \frac { 3 K } { c } ( T + t _ { 0 } ) ^ { 1 / 3 } .
$$

By the initial ofset condition $\begin{array} { r } { t _ { 0 } \geq ( \frac { 4 } { 3 c } ) ^ { 3 } } \end{array}$ , the multiplier on the right-hand sum satisfies $\frac { 2 } { 3 c t _ { 0 } ^ { 1 / 3 } } \leq \frac { 1 } { 2 }$ Subtracting $\textstyle { \frac { 1 } { 2 } } \sum _ { t = 0 } ^ { T - 1 } x _ { t }$ from both sides and multiplying the entire inequality by $\textstyle { \frac { 2 } { T } }$ establishes the stated average bound. □

Remark G.1 (Comparison with Pointwise Summation). To understand the benefit of Lemma G.1, consider the alternative of directly summing the pointwise bound $x _ { t } \leq v ( t + t _ { 0 } ) ^ { - 2 / 3 }$ provided by the Fractional Chung’s Lemma (Lemma 9.3), where $v = \mathrm { m a x } ( t _ { 0 } ^ { 2 / 3 } x _ { 0 } , 2 K / c )$ . Summing this polynomial envelope via an integral bound $\begin{array} { r } { ( \sum _ { t = 0 } ^ { T - 1 } ( t + t _ { 0 } ) ^ { - 2 / 3 } \le \int _ { 0 } ^ { T } ( x + t _ { 0 } - 1 ) ^ { - 2 / 3 } d x \le 3 ( T + t _ { 0 } ) ^ { 1 / 3 } ) } \end{array}$ yields $\textstyle \sum _ { t = 0 } ^ { T - 1 } x _ { t } \leq 3 v ( T + t _ { 0 } ) ^ { 1 / 3 }$ . When the initial error term $t _ { 0 } ^ { 2 / 3 } x _ { 0 }$ dominates v, this standard approach bounds the initialization sum by $\mathcal { O } ( x _ { 0 } t _ { 0 } ^ { 2 / 3 } T ^ { 1 / 3 } )$ . In contrast, the summation by parts in Lemma G.1 captures the exponential decay of the initial error, decoupling it from the harmonic integral and bounding it by $\mathcal { O } \big ( \frac { x _ { 0 } t _ { 0 } ^ { 2 / 3 } } { c } \big )$ . Therefore, directly integrating the pointwise envelope artificially inflates the initialization footprint by a factor of $c T ^ { 1 / 3 }$

With this fractional summation tool established, we bound the critic’s tracking error across the warm-start architecture. By unrolling the geometric contraction of the N inner-loop SGD steps starting from the warm-started parameter $\theta _ { t }$ , we compress the inner-loop dynamics into a single efective outer-loop recurrence, allowing us to apply the fractional summation bound.

Lemma G.2 (Warm-Start Critic Tracking Error). Suppose that Assumptions 1–3 hold. For independent constants $c _ { \eta } , c _ { \beta } > 0$ , let the actor step size be $\begin{array} { r } { \eta _ { t } = \frac { c _ { \eta } } { t + t _ { 0 } } } \end{array}$ , and let the inner critic step size be $\alpha _ { t + 1 } = N ^ { - 1 } \beta _ { t + 1 }$ and the efective outer schedule be $\begin{array} { r } { \beta _ { t + 1 } = \frac { 2 c _ { \beta } } { ( t + t _ { 0 } ) ^ { 2 / 3 } } } \end{array}$ for all $t \geq 0$ , with an arbitrary initialization $\alpha _ { 0 }$ and an initial ofset $t _ { 0 } \geq$ max $\left( 1 , c _ { \eta } \tau , ( 4 c _ { \beta } \kappa ) ^ { 3 / 2 } , ( \frac { 4 } { 3 c _ { \beta } \kappa } ) ^ { 3 } \right)$ . Then the expected tracking error $Z _ { t } \triangleq \mathbb { E } [ \Vert e _ { t } \Vert _ { 2 } ^ { 2 } ] = \mathbb { E } [ \Vert \theta _ { t + 1 } - \theta _ { \tau } ^ { * } ( \omega _ { t } ) \Vert _ { 2 } ^ { 2 } ]$ is bounded pointwise for all $t \geq 0$ by:

$$
Z _ { t } \le \frac { v } { ( t + t _ { 0 } ) ^ { 2 / 3 } } ,
$$

where the bounding coeficient is $v \triangleq$ max $\begin{array} { r } { \left( t _ { 0 } ^ { 2 / 3 } Z _ { 0 } , \frac { 2 K _ { N } } { c _ { \beta } \kappa } \right) } \end{array}$ , and the noise envelope constant is $K _ { N } \triangleq$ $\begin{array} { r } { \frac { 4 c _ { \beta } ^ { 2 } G _ { c r } ^ { 2 } } { N } + \frac { L _ { \theta } ^ { 2 } c _ { \eta } ^ { 2 } G _ { a c } ^ { 2 } } { c _ { \beta } \kappa } } \end{array}$ . Furthermore, the average tracking error is bounded by:

$$
\frac { 1 } { T } \sum _ { t = 0 } ^ { T - 1 } Z _ { t } \le \frac { 2 t _ { 0 } ^ { 2 / 3 } } { c _ { \beta } \kappa T } Z _ { 0 } + \frac { 6 K _ { N } } { c _ { \beta } \kappa T } ( T + t _ { 0 } ) ^ { 1 / 3 } .
$$

Proof. (i) Step-Size Validity. We first verify that the initial ofset $t _ { 0 }$ guarantees the condition $\begin{array} { r } { \alpha _ { t + 1 } \kappa \leq \frac { 1 } { 2 N } } \end{array}$ for all $t \geq 0$ , which is equivalent to $\beta _ { t + 1 } \kappa \leq \frac { 1 } { 2 }$ . This condition holds because $\beta _ { t + 1 } \kappa \leq$ $\begin{array} { r } { \beta _ { 1 } \kappa \le \frac { 2 c _ { \beta } \kappa } { t _ { 0 } ^ { 2 / 3 } } \le \frac { 1 } { 2 } } \end{array}$ , due to the requirement $t _ { 0 } \ge ( 4 c _ { \beta } \kappa ) ^ { 3 / 2 }$ . Additionally, the condition $t _ { 0 } \geq c _ { \eta } \tau$ guarantees that the actor step size satisfies $\begin{array} { r } { \eta _ { t } \leq \frac { c _ { \eta } } { t _ { 0 } } \leq 1 / \tau } \end{array}$ for all $t \geq 0$ , which ensures the actor update direction is bounded by $G _ { a c }$ via Lemma 2.2(ii).

(ii) Exact Inner-Loop Contraction. Consider outer loop $t + 1$ , which uses the critic step size $\alpha _ { t + 1 }$ and fixes the target $\theta _ { \tau } ^ { * } ( \omega _ { t + 1 } )$ . Let $e _ { t + 1 } ^ { ( j ) } \triangleq \theta _ { t + 1 } ^ { ( j ) } - \theta _ { \tau } ^ { * } ( \omega _ { t + 1 } )$ denote the tracking error at the j-th inner-loop SGD step, and let $\mathcal { F } _ { t + 1 } ^ { ( j ) }$ denote the filtration including the history up to this step. As derived in Lemma 5.5, a single inner SGD step satisfies the exact contraction:

$$
\mathbb { E } [ \| e _ { t + 1 } ^ { ( j + 1 ) } \| _ { 2 } ^ { 2 } \mid \mathcal { F } _ { t + 1 } ^ { ( j ) } ] \leq ( 1 - 2 \alpha _ { t + 1 } \kappa ) \| e _ { t + 1 } ^ { ( j ) } \| _ { 2 } ^ { 2 } + \alpha _ { t + 1 } ^ { 2 } G _ { c r } ^ { 2 } .
$$

We take the total expectation conditioned on the outer-loop filtration $\mathcal { F } _ { t + 1 }$ (which fixes the actor parameter $\omega _ { t + 1 }$ and the warm-start initialization $\theta _ { t + 1 } ^ { ( 0 ) } = \theta _ { t + 1 } )$ . Unrolling this exact SGD contraction over N steps, the terminal expected error satisfies:

$$
\mathbb { E } [ \| e _ { t + 1 } ^ { ( N ) } \| _ { 2 } ^ { 2 } \mid \mathcal { F } _ { t + 1 } ] \le ( 1 - 2 \alpha _ { t + 1 } \kappa ) ^ { N } \| \tilde { e } _ { t } \| _ { 2 } ^ { 2 } + \alpha _ { t + 1 } ^ { 2 } G _ { c r } ^ { 2 } \sum _ { j = 0 } ^ { N - 1 } ( 1 - 2 \alpha _ { t + 1 } \kappa ) ^ { j } ,
$$

where $\tilde { e } _ { t } \triangleq \theta _ { t + 1 } - \theta _ { \tau } ^ { \ast } ( \omega _ { t + 1 } )$ . Note that $e _ { t + 1 } ^ { ( 0 ) } = \theta _ { t + 1 } ^ { ( 0 ) } - \theta _ { \tau } ^ { \ast } ( \omega _ { t + 1 } ) = \theta _ { t + 1 } - \theta _ { \tau } ^ { \ast } ( \omega _ { t + 1 } ) = \tilde { e } _ { t }$ , and

$e _ { t + 1 } ^ { ( N ) } = \theta _ { t + 1 } ^ { ( N ) } - \theta _ { \tau } ^ { * } ( \omega _ { t + 1 } ) = \theta _ { t + 2 } - \theta _ { \tau } ^ { * } ( \omega _ { t + 1 } ) = e _ { t + 1 }$ . Defining $\begin{array} { r } { \bar { \alpha } _ { t + 1 } \triangleq \frac { 1 - ( 1 - 2 \alpha _ { t + 1 } \kappa ) ^ { N } } { \kappa } } \end{array}$ , we have <sup>N</sup>X<sup>−1</sup>(1 − 2α<sub>t+1</sub>κ)<sup>j</sup> = 1 − (1 − 2α<sub>t+1</sub>κ)<sup>N</sup> α¯<sub>t+1</sub> j=0 2α<sub>t+1</sub>κ 2α<sub>t+1</sub>

so that the preceding inequality reduces to:

$$
\mathbb { E } [ \| e _ { t + 1 } \| _ { 2 } ^ { 2 } \mid \mathcal { F } _ { t + 1 } ] \leq ( 1 - \bar { \alpha } _ { t + 1 } \kappa ) \| \tilde { e } _ { t } \| _ { 2 } ^ { 2 } + \frac { \alpha _ { t + 1 } \bar { \alpha } _ { t + 1 } } { 2 } G _ { c r } ^ { 2 } .
$$

(iii) Young’s Inequality and Controlling Target Drift. We separate the outer-loop tracking error $e _ { t } = \theta _ { t + 1 } - \theta _ { \tau } ^ { * } ( \omega _ { t } )$ from the target drift:

$$
\tilde { e } _ { t } = e _ { t } + \left( \theta _ { \tau } ^ { * } ( \omega _ { t } ) - \theta _ { \tau } ^ { * } ( \omega _ { t + 1 } ) \right) .
$$

Applying Young’s inequality $\| x + y \| _ { 2 } ^ { 2 } \leq ( 1 + p ) \| x \| _ { 2 } ^ { 2 } + ( 1 + 1 / p ) \| y \| _ { 2 } ^ { 2 }$ with parameter $p = \bar { \alpha } _ { t + 1 } \kappa / 2$ yields:

$$
\| \tilde { e } _ { t } \| _ { 2 } ^ { 2 } \le \left( 1 + \frac { \bar { \alpha } _ { t + 1 } \kappa } { 2 } \right) \| e _ { t } \| _ { 2 } ^ { 2 } + \left( 1 + \frac { 2 } { \bar { \alpha } _ { t + 1 } \kappa } \right) \| \theta _ { \tau } ^ { * } ( \omega _ { t } ) - \theta _ { \tau } ^ { * } ( \omega _ { t + 1 } ) \| _ { 2 } ^ { 2 } .
$$

The multiplier $1 - \bar { \alpha } _ { t + 1 } \kappa = ( 1 - 2 \alpha _ { t + 1 } \kappa ) ^ { N }$ is non-negative, because $2 \alpha _ { t + 1 } \kappa \leq \frac { 1 } { N } \leq 1$ by Step (i). This allows us to substitute the Young bound into the contraction from Step (ii). Using $\begin{array} { r } { ( 1 - \bar { \alpha } _ { t + 1 } \kappa ) ( 1 + \frac { \bar { \alpha } _ { t + 1 } \kappa } { 2 } ) \leq 1 - \frac { \bar { \alpha } _ { t + 1 } \kappa } { 2 } } \end{array}$ and $\begin{array} { r } { ( 1 - \bar { \alpha } _ { t + 1 } \kappa ) ( 1 + \frac { 2 } { \bar { \alpha } _ { t + 1 } \kappa } ) \leq \frac { 2 } { \bar { \alpha } _ { t + 1 } \kappa } } \end{array}$ together with the Lipschitz target-drift bound

$$
\| \theta _ { \tau } ^ { * } ( \omega _ { t } ) - \theta _ { \tau } ^ { * } ( \omega _ { t + 1 } ) \| _ { 2 } ^ { 2 } \leq L _ { \theta } ^ { 2 } \| \omega _ { t + 1 } - \omega _ { t } \| _ { 2 } ^ { 2 } \leq L _ { \theta } ^ { 2 } \eta _ { t } ^ { 2 } G _ { a c } ^ { 2 } ,
$$

and taking the total expectation E[·] yields the single-step outer-loop recurrence:

$$
Z _ { t + 1 } \leq \left( 1 - \frac { \bar { \alpha } _ { t + 1 } \kappa } { 2 } \right) Z _ { t } + \frac { \alpha _ { t + 1 } \bar { \alpha } _ { t + 1 } } { 2 } G _ { c r } ^ { 2 } + \frac { 2 } { \bar { \alpha } _ { t + 1 } \kappa } L _ { \theta } ^ { 2 } \eta _ { t } ^ { 2 } G _ { a c } ^ { 2 } .
$$

(iv) Bounding $\bar { \alpha } _ { t + 1 }$ and Extracting $K _ { N }$ . We bound $\bar { \alpha } _ { t + 1 }$ against the efective outer schedule $\beta _ { t + 1 } = N \alpha _ { t + 1 }$ . Let $x = 2 \alpha _ { t + 1 } \kappa$ , which, by Step (i), satisfies $N x = 2 \beta _ { t + 1 } \kappa \leq 1$ . Because $x \le 1$ Bernoulli’s inequality yields $( 1 - x ) ^ { N } \geq 1 - N x .$ , ensuring

$$
\bar { \alpha } _ { t + 1 } \kappa = 1 - ( 1 - x ) ^ { N } \leq N x = 2 \beta _ { t + 1 } \kappa .
$$

Hence $\bar { \alpha } _ { t + 1 } \leq 2 \beta _ { t + 1 }$ . Conversely, we bound the Binomial expansion $\begin{array} { r } { ( 1 - x ) ^ { N } = \sum _ { k = 0 } ^ { N } { \binom { N } { k } } ( - x ) ^ { k } } \end{array}$ Letting $a _ { k } = { \binom { N } { k } } x ^ { k }$ , the ratio of adjacent magnitudes is $\begin{array} { r } { \frac { a _ { k + 1 } } { a _ { k } } = \frac { N - k } { k + 1 } x \leq \frac { N x } { k + 1 } } \end{array}$ . Because $N x \leq 1$ this ratio satisfies $\begin{array} { r } { \frac { a _ { k + 1 } } { a _ { k } } \leq \frac { 1 } { k + 1 } < 1 } \end{array}$ for all $k \geq 1$ . Since the terms decrease in magnitude, grouping the alternating remainder terms starting from $k = 3$ yields a non-positive sum: $( - a _ { 3 } + a _ { 4 } ) +$ $( - a _ { 5 } + a _ { 6 } ) + \cdot \cdot \cdot \leq 0$ . Thus, $\begin{array} { r } { ( 1 - x ) ^ { N } \leq 1 - N x + \frac { N ( N - 1 ) } { 2 } x ^ { 2 } } \end{array}$ . Bounding $N ( N - 1 ) \leq N ^ { 2 }$ gives $\begin{array} { r } { ( 1 - x ) ^ { N } \leq 1 - N x + \frac { N ^ { 2 } x ^ { 2 } } { 2 } = 1 - N x ( 1 - \frac { N x } { 2 } ) } \end{array}$ . Because $N x \leq 1$ , we have $\begin{array} { r } { 1 - \frac { N x } { 2 } \ge \frac { 1 } { 2 } } \end{array}$ , yielding

$$
\bar { \alpha } _ { t + 1 } \kappa = 1 - ( 1 - x ) ^ { N } \geq \frac 1 2 N x = \beta _ { t + 1 } \kappa .
$$

Thus, $\beta _ { t + 1 } \leq \bar { \alpha } _ { t + 1 } \leq 2 \beta _ { t + 1 }$

Substituting these bounds into the recurrence yields:

$$
Z _ { t + 1 } \leq \left( 1 - \frac { 1 } { 2 } \beta _ { t + 1 } \kappa \right) Z _ { t } + \frac { \beta _ { t + 1 } ^ { 2 } } { N } G _ { c r } ^ { 2 } + \frac { 2 } { \beta _ { t + 1 } \kappa } L _ { \theta } ^ { 2 } \eta _ { t } ^ { 2 } G _ { a c } ^ { 2 } .
$$

Substituting the step-size schedules $\begin{array} { r } { \beta _ { t + 1 } ~ = ~ \frac { 2 c _ { \beta } } { ( t + t _ { 0 } ) ^ { 2 / 3 } } } \end{array}$ and $\begin{array} { r } { \eta _ { t } ~ = ~ \frac { c _ { \eta } } { t + t _ { 0 } } } \end{array}$ , the contraction multiplier evaluates to $1 - \frac { c _ { \beta } \kappa } { ( t + t _ { 0 } ) ^ { 2 / 3 } }$ . The additive noise terms evaluate to:

$$
\frac { 1 } { N } \frac { 4 c _ { \beta } ^ { 2 } } { ( t + t _ { 0 } ) ^ { 4 / 3 } } G _ { c r } ^ { 2 } + \frac { 2 ( t + t _ { 0 } ) ^ { 2 / 3 } } { 2 c _ { \beta } \kappa } L _ { \theta } ^ { 2 } \frac { c _ { \eta } ^ { 2 } } { ( t + t _ { 0 } ) ^ { 2 } } G _ { a c } ^ { 2 } = \frac { \frac { 4 c _ { \beta } ^ { 2 } G _ { c r } ^ { 2 } } { N } + \frac { L _ { \theta } ^ { 2 } c _ { \eta } ^ { 2 } G _ { a c } ^ { 2 } } { c _ { \beta } \kappa } } { ( t + t _ { 0 } ) ^ { 4 / 3 } } = \frac { K _ { N } } { ( t + t _ { 0 } ) ^ { 4 / 3 } } ,
$$

explicitly extracting our defined constant $K _ { N }$

(v) Pointwise and Average Bounds. The sequence satisfies the single-step recurrence:

$$
Z _ { t + 1 } \leq \left( 1 - { \frac { c _ { \beta } \kappa } { ( t + t _ { 0 } ) ^ { 2 / 3 } } } \right) Z _ { t } + { \frac { K _ { N } } { ( t + t _ { 0 } ) ^ { 4 / 3 } } } .
$$

To invoke Fractional Chung’s Lemma (Lemma 9.3) with the contraction coeficient $c = c _ { \beta } \kappa$ , we verify that the initial ofset satisfies $\begin{array} { r } { t _ { 0 } \geq \operatorname* { m a x } \left( ( c _ { \beta } \kappa ) ^ { 3 / 2 } , ( \frac { 4 } { 3 c _ { \beta } \kappa } ) ^ { 3 } \right) } \end{array}$ . The term $( c _ { \beta } \kappa ) ^ { 3 / 2 }$ is subsumed by the stated $( 4 c _ { \beta } \kappa ) ^ { 3 / 2 }$ requirement, and $\big ( \frac { 4 } { 3 c _ { \beta } \kappa } \big ) ^ { 3 }$ is directly enveloped by the assumed lower bound on $t _ { 0 }$ . Thus, Lemma 9.3 yields the pointwise bound $\begin{array} { r } { Z _ { t } \le \frac { v } { ( t + t _ { 0 } ) ^ { 2 / 3 } } } \end{array}$ with $\begin{array} { r } { v = \operatorname* { m a x } \left( t _ { 0 } ^ { 2 / 3 } Z _ { 0 } , \frac { 2 K _ { N } } { c _ { \beta } \kappa } \right) } \end{array}$ Furthermore, by the initial ofset condition $\begin{array} { r } { t _ { 0 } \geq \operatorname* { m a x } \left( 1 , ( \frac { 4 } { 3 c _ { \beta } \kappa } ) ^ { 3 } \right) } \end{array}$ , applying the Fractional Chung’s Summation Bound (Lemma G.1) to the identical sequence directly yields the average bound:

$$
\frac { 1 } { T } \sum _ { t = 0 } ^ { T - 1 } Z _ { t } \le \frac { 2 t _ { 0 } ^ { 2 / 3 } } { c _ { \beta } \kappa T } Z _ { 0 } + \frac { 6 K _ { N } } { c _ { \beta } \kappa T } ( T + t _ { 0 } ) ^ { 1 / 3 } ,
$$

completing the proof.

## G.2 Warm-Start Average-Iterate Regularized Convergence

We now integrate the warm-start tracking bound into our PMD analysis. By substituting the efective fractional noise envelope into the PMD Value-Telescoping Bound (Lemma 9.2), we establish the average-iterate convergence rate for the regularized objective. This demonstrates how the warm-start architecture separates the total algorithmic error into an invariant primary tracking component and rapidly decaying transient optimization terms.

Theorem G.1 (Warm-Start Average-Iterate Regularized Convergence). For Algorithm $^ { 5 , }$ suppose that Assumptions $1 \mathrm { - } \mathit { 4 }$ and 9 hold. For independent constants $\begin{array} { r } { c _ { \eta } \ge \frac { 4 } { 3 \tau } } \end{array}$ and $\zeta > 0$ , define the base schedule constant $c _ { \beta } \triangleq \frac { 1 } { 2 } \zeta c _ { \eta } ^ { 2 / 3 } N ^ { 1 / 3 }$ . Let the actor step size be $\begin{array} { r } { \eta _ { t } = \frac { c _ { \eta } } { t + t _ { 0 } } } \end{array}$ . Let $\alpha _ { 0 }$ be arbitrary, and for all $t \geq 0$ , let the subsequent inner critic step sizes satisfy $\alpha _ { t + 1 } = N ^ { - 1 } \beta _ { t + 1 }$ with the efective outer schedule $\begin{array} { r } { \beta _ { t + 1 } = \frac { 2 c _ { \beta } } { ( t + t _ { 0 } ) ^ { 2 / 3 } } } \end{array}$ , and let the initial ofset satisfy $\begin{array} { r } { t _ { 0 } \geq \operatorname* { m a x } \left( c _ { \eta } \tau , ( 4 c _ { \beta } \kappa ) ^ { 3 / 2 } , ( \frac { 4 } { 3 c _ { \beta } \kappa } ) ^ { 3 } \right) } \end{array}$

By tuning the constants to the asymptotic orders $c _ { \eta } = \Theta ( \tau ^ { - 1 } ) , \zeta = \Theta ( \kappa ^ { - 1 } )$ , and $t _ { 0 } = \Theta ( N ^ { 1 / 2 } \tau ^ { - 1 } )$ subject to the stated conditions, the expected average regularized performance gap for the warm-start double-loop algorithm satisfies:

$$
\begin{array} { r l } { \displaystyle \frac { 1 } { T } \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } [ G a p _ { t } ^ { \dag } ] \leq \underbrace { \mathcal { O } \left( \frac { 1 } { ( 1 - \gamma ) ^ { 7 } \kappa ^ { 6 } \tau ^ { 5 / 3 } T _ { t o t a l } ^ { 2 / 3 } } \right) } _ { I n v a r i a n t \ T r a c k i n g \ E r r o r } + \underbrace { \mathcal { O } \left( \frac { \epsilon _ { a p p } } { ( 1 - \gamma ) \tau } \right) } _ { A p p r o x i m a t i o n \ B i a s } } & { } \\ { \displaystyle } & { + \underbrace { \tilde { \mathcal { O } } \left( \frac { N ^ { 1 / 2 } } { ( 1 - \gamma ) ^ { 2 } \kappa T } + \frac { N ^ { - 1 / 2 } } { ( 1 - \gamma ) ^ { 7 } \kappa ^ { 6 } \tau ^ { 2 } T } + \frac { 1 } { ( 1 - \gamma ) ^ { 3 } \kappa ^ { 2 } \tau T } \right) } _ { T r a n s i e n t \ C o n v e r o e n c e \ E r r o r } . } \end{array}
$$

Proof. (i) Telescoping Expansion. By substituting the average tracking bound from Lemma G.2 into the PMD Value-Telescoping Bound (Lemma 9.2) with $\begin{array} { r } { c _ { \eta } \geq \frac { 4 } { 3 \tau } } \end{array}$ , the expected average regularized performance gap expands as:

$$
\begin{array} { r l } { ( 1 - \gamma ) \displaystyle \frac { 1 } { T } \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } [ \mathbb { G } \mathrm { a p } _ { t } ^ { \dag } ] \leq } & { \underbrace { \frac { t _ { 0 } - 1 } { c _ { 0 } T } D _ { 0 } ^ { \dag } } _ { \mathrm { P M D ~ I m i t i a l i z a t i o n ~ E r r o r } } + \underbrace { \frac { c _ { \eta } G _ { a c } ^ { 2 } B _ { \phi } ^ { 2 } } { 2 T } ( 1 + \log T ) } _ { \mathrm { P M D ~ L o c a l ~ E r r o r } } } \\ { + } & { \underbrace { \frac { 4 C _ { j \ o i n t } ^ { 2 } } { T } \epsilon _ { a p p } } _ { \mathrm { A p p r o x i m a t i o n ~ B i a s } } + \underbrace { \frac { 8 B _ { \phi } ^ { 2 } t _ { 0 } ^ { 2 / 3 } } { T c _ { \beta } \kappa T } Z _ { 0 } } _ { \mathrm { T i n i t a l i z a t i o n ~ T r a c k i n g ~ P e n a l t y } } + \underbrace { \frac { 2 4 B _ { \phi } ^ { 2 } K _ { N } } { \gamma c _ { \beta } \kappa T } ( T + t _ { 0 } ) ^ { 1 / 3 } } _ { \mathrm { F r a c t i o n a l ~ T r a c k i n g ~ P e n a l t y } } . } \end{array}
$$

Since $( T + t _ { 0 } ) ^ { 1 / 3 } = \Theta ( T ^ { 1 / 3 } + t _ { 0 } ^ { 1 / 3 } )$ , the fractional tracking penalty decomposes into a $\Theta \big ( \frac { K _ { N } } { \tau c _ { \beta } \kappa T ^ { 2 / 3 } } \big )$ primary term and $\mathrm { ~ a ~ } \Theta \bigl ( \frac { K _ { N } t _ { 0 } ^ { 1 / 3 } } { \tau c _ { \beta } \kappa T } \bigr )$ secondary transient term.

(ii) Invoking Average Tracking Bound. By choosing $\zeta = \Theta ( \kappa ^ { - 1 } )$ and setting $c _ { \eta } = \Theta ( \tau ^ { - 1 } )$ , the efective contraction coeficient scales as $c _ { \beta } \kappa = \Theta ( N ^ { 1 / 3 } \tau ^ { - 2 / 3 } )$ . Note that $t _ { 0 } \geq 1$ is guaranteed by $t _ { 0 } \geq c _ { \eta } \tau$ and $c _ { \eta } \geq \frac { 4 } { 3 \tau }$ . To satisfy the initial ofset condition $t _ { 0 } \geq$ max $\left( c _ { \eta } \tau , ( 4 c _ { \beta } \kappa ) ^ { 3 / 2 } , ( \textstyle \frac { 4 } { 3 c _ { \beta } \kappa } ) ^ { 3 } \right)$ we verify the dominant term. For $\tau \ \leq \ \tau _ { m a x }$ , the term $( 4 c _ { \beta } \kappa ) ^ { 3 / 2 } \ = \ \Theta ( N ^ { 1 / 2 } \tau ^ { - 1 } )$ dominates $\begin{array} { r } { \big ( \frac { 4 } { 3 c _ { \beta } \kappa } \big ) ^ { 3 } = \Theta ( N ^ { - 1 } \tau ^ { 2 } ) } \end{array}$ as well as $c _ { \eta } \tau = \Theta ( 1 )$ . This allows us to fulfill the requirement by setting $t _ { 0 } = \Theta ( N ^ { 1 / 2 } \tau ^ { - 1 } )$

(iii) Structural Order of Noise Envelope. We evaluate the noise envelope $K _ { N }$ . By Lemmas 2.2 and 2.3, the structural bounds scale as $G _ { c r } ^ { 2 } = \Theta ( \kappa ^ { - 2 } ( 1 - \gamma ) ^ { - 2 } ) , G _ { a c } = \Theta ( \kappa ^ { - 1 } ( 1 - \gamma ) ^ { - 1 } )$ , and $L _ { \theta } ^ { 2 } =$ $\Theta ( \kappa ^ { - 4 } ( 1 - \gamma ) ^ { - 4 } )$ . Substituting these alongside our choices $c _ { \eta } = \Theta ( \tau ^ { - 1 } )$ and $c _ { \beta } = \Theta ( \kappa ^ { - 1 } \tau ^ { - 2 / 3 } N ^ { 1 / 3 } )$ we evaluate the two additive components of $K _ { N }$ separately. The inner-loop SGD noise component evaluates to:

$$
\frac { 4 c _ { \beta } ^ { 2 } G _ { c r } ^ { 2 } } { N } = \Theta \left( N ^ { - 1 / 3 } \tau ^ { - 4 / 3 } \kappa ^ { - 4 } ( 1 - \gamma ) ^ { - 2 } \right) .
$$

The actor target drift component evaluates to:

$$
\frac { L _ { \theta } ^ { 2 } c _ { \eta } ^ { 2 } G _ { a c } ^ { 2 } } { c _ { \beta } \kappa } = \Theta \left( N ^ { - 1 / 3 } \tau ^ { - 4 / 3 } \kappa ^ { - 6 } ( 1 - \gamma ) ^ { - 6 } \right) .
$$

While both terms share an identical $\Theta ( N ^ { - 1 / 3 } \tau ^ { - 4 / 3 } )$ dependence, the second term dominates in its dependence on κ and $( 1 - \gamma ) ^ { - 1 }$ . Combining them yields the final noise envelope:

$$
K _ { N } = \Theta \left( N ^ { - 1 / 3 } \tau ^ { - 4 / 3 } \kappa ^ { - 6 } ( 1 - \gamma ) ^ { - 6 } \right) .
$$

(iv) Synthesis. We substitute the evaluated orders back into the tracking and PMD penalties:

• Initialization Tracking Penalty: The geometric coeficient evaluates to $\begin{array} { r } { \frac { t _ { 0 } ^ { 2 / 3 } } { c _ { \beta } \kappa } = \frac { \Theta ( N ^ { 1 / 3 } \tau ^ { - 2 / 3 } ) } { \Theta ( N ^ { 1 / 3 } \tau ^ { - 2 / 3 } ) } = } \end{array}$ $\Theta ( 1 )$ . Thus, the initialization tracking penalty evaluates to $\Theta \left( { \frac { Z _ { 0 } } { \tau T } } \right)$ . As shown in the proof of Theorem 6.2, the initial tracking error $Z _ { 0 } = \mathbb { E } [ \| \theta _ { 1 } - \theta _ { \tau } ^ { * } ( \omega _ { 0 } ) \| _ { 2 } ^ { 2 } ]$ is bounded by $Z _ { 0 } \le ( 2 B _ { \theta } ) ^ { 2 } =$ $G _ { a c } ^ { 2 } = \Theta ( \kappa ^ { - 2 } ( 1 - \gamma ) ^ { - 2 } )$ . Substituting this bound and dividing by $1 - \gamma$ , the penalty yields $\begin{array} { r } { \mathcal O \left( \frac { 1 } { ( 1 - \gamma ) ^ { 3 } \kappa ^ { 2 } \tau T } \right) } \end{array}$ , which is subsumed by the PMD local error $\begin{array} { r } { \mathcal { O } \left( \frac { \log T } { ( 1 - \gamma ) ^ { 3 } \kappa ^ { 2 } \tau T } \right) } \end{array}$

• Invariant Tracking Error: The primary fractional tracking penalty evaluates to $\begin{array} { r } { \Theta \left( \frac { K _ { N } } { \tau c _ { \beta } \kappa T ^ { 2 / 3 } } \right) = \Theta \left( \frac { N ^ { - 2 / 3 } } { \tau ^ { 5 / 3 } \kappa ^ { 6 } ( 1 - \gamma ) ^ { 6 } T ^ { 2 / 3 } } \right) } \end{array}$ . Because $N ^ { 2 / 3 } T ^ { 2 / 3 } = ( N \times T ) ^ { 2 / 3 } = T _ { t o t a l } ^ { 2 / 3 }$ , the individual loop sizes combine into the total size $T _ { t o t a l }$ . After dividing the entire inequality by $1 - \gamma$ , this yields the invariant tracking error envelope $\Theta \left( \frac { 1 } { ( 1 - \gamma ) ^ { 7 } \kappa ^ { 6 } \tau ^ { 5 / 3 } T _ { t o t a l } ^ { 2 / 3 } } \right)$

• Transient Convergence Error: The secondary fractional tracking penalty divided by $1 - \gamma$ evaluates to $\begin{array} { r } { \Theta \left( \frac { K _ { N } t _ { 0 } ^ { 1 / 3 } } { ( 1 - \gamma ) \tau c _ { \beta } \kappa T } \right) = \Theta \left( \frac { N ^ { - 1 / 2 } } { ( 1 - \gamma ) ^ { 7 } \kappa ^ { 6 } \tau ^ { 2 } T } \right) } \end{array}$ . As shown in the proof of Theorem 9.1, the initial KL divergence is bounded by $\begin{array} { r } { D _ { 0 } ^ { \dagger } \le \frac { 2 B _ { \theta } B _ { \phi } } { \tau _ { m a x } } + \log | \boldsymbol { \mathcal { A } } | = \mathcal { O } ( \kappa ^ { - 1 } ( 1 - \gamma ) ^ { - 1 } ) } \end{array}$ . Since $t _ { 0 } / c _ { \eta } = \Theta ( N ^ { 1 / 2 } )$ , the PMD initialization error divided by $1 - \gamma$ evaluates to $\begin{array} { r } { \mathcal { O } \left( \frac { t _ { 0 } D _ { 0 } ^ { \dagger } } { ( 1 - \gamma ) c _ { \eta } T } \right) = } \end{array}$ $\begin{array} { r } { \mathcal { O } \left( \frac { N ^ { 1 / 2 } } { ( 1 - \gamma ) ^ { 2 } \kappa T } \right) } \end{array}$ . Finally, dividing the PMD local error by $1 - \gamma$ yields Θ $\left( \frac { \log T } { ( 1 - \gamma ) ^ { 3 } \kappa ^ { 2 } \tau T } \right)$   
Combining the invariant tracking error envelope with these transient optimization terms and the   
$\Theta ( \tau ^ { - 1 } \epsilon _ { a p p } )$ approximation bias completes the proof. □

Remark G.2 (Advantage Over Pointwise Summation and Last-Iterate Collapse). The proof of Theorem G.1 fundamentally relies on the average tracking error derived from the Fractional Chung’s Summation Bound (Lemma G.1). Alternatively, if we were to sum the pointwise tracking error $Z _ { t } \le v ( t + t _ { 0 } ) ^ { - 2 / 3 }$ derived from the standard Fractional Chung’s Lemma, the resulting bound would include a heavily inflated penalty.

Specifically, the pointwise bounding coeficient is $\begin{array} { r } { v \triangleq \operatorname* { m a x } \left( t _ { 0 } ^ { 2 / 3 } Z _ { 0 } , \frac { 2 K _ { N } } { c _ { \beta } \kappa } \right) } \end{array}$ , with an initialization contribution $t _ { 0 } ^ { 2 / 3 } Z _ { 0 }$ . Summing the pointwise bound via an integral $( \sum _ { t = 0 } ^ { T - 1 } ( t + t _ { 0 } ) ^ { - 2 / 3 } \leq 3 ( T + t _ { 0 } ) ^ { 1 / 3 } )$

yields $\begin{array} { r } { \frac { 1 } { T } \sum _ { t = 0 } ^ { T - 1 } Z _ { t } = \mathcal { O } \big ( \frac { v ( T + t _ { 0 } ) ^ { 1 / 3 } } { T } \big ) } \end{array}$ . Using $( T + t _ { 0 } ) ^ { 1 / 3 } \leq T ^ { 1 / 3 } + t _ { 0 } ^ { 1 / 3 }$ , the component proportional to $T ^ { 1 / 3 }$ gives an average tracking penalty of $\mathcal { O } \left( \frac { v } { \tau T ^ { 2 / 3 } } \right)$ , with an initialization contribution $\mathcal { O } \big ( \frac { t _ { 0 } ^ { 2 / 3 } Z _ { 0 } } { \tau T ^ { 2 / 3 } } \big )$ in the performance gap bound. As evaluated in Step (ii) of the proof of Theorem G.1, the ofset condition $t _ { 0 } \ge ( 4 c _ { \beta } \kappa ) ^ { 3 / 2 }$ (ensuring a stable inner-loop contraction) requires an initial ofset $t _ { 0 } = \Theta ( N ^ { 1 / 2 } \tau ^ { - 1 } )$ . By substituting this and the outer-loop size $T = T _ { t o t a l } / N$ , the initialization contribution to the average tracking penalty becomes

$$
\mathcal { O } \left( \frac { N ^ { 1 / 3 } Z _ { 0 } } { \tau ^ { 5 / 3 } T ^ { 2 / 3 } } \right) = \mathcal { O } \left( \frac { N Z _ { 0 } } { \tau ^ { 5 / 3 } T _ { t o t a l } ^ { 2 / 3 } } \right) .
$$

In contrast, by utilizing summation by parts, Lemma G.1 captures the exponential decay of the initial error, decoupling it from the harmonic integral and bounding its contribution to the average penalty by $\begin{array} { r } { \mathcal { O } \big ( \frac { t _ { 0 } ^ { 2 / 3 } Z _ { 0 } } { c _ { \beta } \kappa \tau T } \big ) } \end{array}$ . Because $t _ { 0 } ^ { 2 / 3 }$ and the efective contraction rate $c _ { \beta } \kappa$ both scale as $\Theta ( N ^ { 1 / 3 } \tau ^ { - 2 / 3 } )$ , the average penalty reduces to $\begin{array} { r } { \mathcal { O } \bigl ( \frac { Z _ { 0 } } { \tau T } \bigr ) = \mathcal { O } \bigl ( \frac { N Z _ { 0 } } { \tau T _ { t o t a l } } \bigr ) } \end{array}$ . The pointwise approach inflates this penalty by a factor of $c _ { \beta } \kappa T ^ { 1 / 3 } = \Theta ( T _ { t o t a l } ^ { 1 / 3 } \tau ^ { - 2 / 3 } )$ , consistent with Remark G.1.

Under the Minimal Action Gap, in the sample-limited phase, the temperature is tuned to $\tau = \Theta ( 1 / \log T _ { t o t a l } )$ to achieve the $\tilde { \mathcal { O } } ( T _ { t o t a l } ^ { - 2 / 3 } )$ average-iterate unregularized rate. The initialization component of the pointwise bound, $\mathcal { O } ( N Z _ { 0 } \tau ^ { - 5 / 3 } T _ { t o t a l } ^ { - 2 / 3 } )$ , becomes $\tilde { \mathcal { O } } ( N Z _ { 0 } T _ { t o t a l } ^ { - 2 / 3 } )$ . For a critic initialization with $Z _ { 0 }$ bounded away from zero, keeping this term within the $\tilde { \mathcal { O } } ( T _ { t o t a l } ^ { - 2 / 3 } )$ target rate requires $N = { \tilde { \Theta } } ( 1 )$ , causing the invariance window to collapse. In contrast, Lemma G.1 safely bounds the initialization penalty by $\tilde { \mathcal { O } } ( N Z _ { 0 } T _ { t o t a l } ^ { - 1 } )$ , allowing $N$ to grow unbounded without breaking the target rate envelope (Section G.3).

Likewise, relying on the pointwise tracking error to establish the last-iterate regularized convergence—which involves evaluating the terminal error $Z _ { T } \le v T ^ { - 2 / 3 }$ directly to establish the convergence of the forward KL divergence—faces the same structural bottleneck, leading to a similar collapse of the last-iterate invariance window to $N = { \tilde { \Theta } } ( 1 )$ .

## G.3 Warm-Start Unregularized Convergence (Minimal Action Gap)

Building upon the regularized guarantees, we map the warm-start performance gap to the unregularized objective for Algorithm 5. By leveraging the Universal Exponential Translation Bound and deploying a two-stage temperature schedule, we balance the invariant tracking error against an exponentially small target tail. This mechanism bypasses the standard linear entropy bottleneck, extracting an accelerated $\tilde { \mathcal { O } } ( T _ { t o t a l } ^ { - 2 / 3 } )$ average-iterate unregularized convergence rate that remains invariant for any intermediate inner-loop choice up to $N \leq T _ { t o t a l } ^ { 2 / 9 }$

Theorem G.2 (Warm-Start Exponential Translation Convergence). For Algorithm 5, suppose that Assumptions 1–5 and 9 hold. Define the two-stage temperature:

$$
\tau _ { T _ { t o t a l } } \triangleq \operatorname* { m a x } \left( \frac { \Delta } { 2 \log ( C _ { \gamma } T _ { t o t a l } ^ { 2 / 3 } ) } , \frac { \Delta } { 2 \log ( C _ { \gamma } ( 1 + \epsilon _ { a p p } ^ { - 1 } ) ) } \right) .
$$

For any intermediate inner-loop choice $N \in [ 1 , T _ { t o t a l } ^ { 2 / 9 } ]$ and $T = T _ { t o t a l } / N$ , if $. T _ { t o t a l }$ is suficiently large and $\epsilon _ { a p p }$ is suficiently small such that $\tau _ { T _ { t o t a l } } \leq \tau _ { m a x }$ , then by choosing the step-size parameters as in Theorem G.1 evaluated at $\tau = \tau _ { T _ { t o t a l } }$ , the expected average unregularized performance gap satisfies the convergence rate:

$$
\begin{array} { r l r } {  { \frac { 1 } { T } \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } \big [ J _ { 0 } ( \pi _ { 0 } ^ { * } ) - J _ { 0 } ( \pi _ { t } ) \big ] \leq \mathcal { O } ( \frac { ( 1 - \gamma ) ^ { - 5 / 3 } + ( \log T _ { t o t a l } ) ^ { 5 / 3 } } { ( 1 - \gamma ) ^ { 8 } \kappa ^ { 6 } \Delta ^ { 8 / 3 } T _ { t o t a l } ^ { 2 / 3 } } ) } } \\ & { } & { + \mathcal { O } ( \frac { ( 1 - \gamma ) ^ { - 1 } + \log ( 1 + \epsilon _ { a p p } ^ { - 1 } ) } { ( 1 - \gamma ) ^ { 2 } \Delta ^ { 2 } } \epsilon _ { a p p } ) } \\ & { } & { = \tilde { \mathcal { O } } ( T _ { t o t a l } ^ { - 2 / 3 } ) + \tilde { \mathcal { O } } ( \epsilon _ { a p p } ) . } \end{array}
$$

In the case $o f \epsilon _ { a p p } = 0$ , we interpret $\begin{array} { r } { \tau _ { T _ { t o t a l } } = \frac { \Delta } { 2 \log ( C _ { \gamma } T _ { t o t a l } ^ { 2 / 3 } ) } } \end{array}$ and $\epsilon _ { a p p } \log ( 1 + \epsilon _ { a p p } ^ { - 1 } ) = 0$

Proof. We first consider $\epsilon _ { a p p } > 0$ . When $\epsilon _ { a p p } = 0$ , the approximation bias term vanishes and the result follows from the sample-limited calculation with $\begin{array} { r } { \tau _ { T _ { t o t a l } } = \frac { \Delta } { 2 \log ( C _ { \gamma } T _ { t o t a l } ^ { 2 / 3 } ) } . } \end{array}$

(i) Global Unregularized Bound. By applying the Universal Unregularized Suboptimality Bound (Corollary 3.1) to the warm-start average-iterate regularized gap from Theorem G.1, the global unregularized performance gap is bounded for any $\tau > 0$ by:

$$
\begin{array} { r l } & { \quad \frac { 1 } { T } \displaystyle \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } \big [ J _ { 0 } ( \pi _ { 0 } ^ { * } ) - J _ { 0 } ( \pi _ { t } ) \big ] } \\ & { \leq \underbrace { \mathcal { O } \left( \frac { 1 } { \Delta ( 1 - \gamma ) ^ { 8 } \kappa ^ { 6 } \tau ^ { 5 / 3 } T _ { t o t a l } ^ { 2 / 3 } } \right) } _ { \mathrm { I m w a r i a n t ~ T r a c k i n g ~ E r o r m } } + \underbrace { \mathcal { O } \left( \frac { \epsilon _ { a p p } } { \Delta ( 1 - \gamma ) ^ { 2 } \tau } \right) } _ { \mathrm { A p p r o x i m u t i o n ~ B i a s } } + \underbrace { C _ { t a i l } \exp \left( - \frac { \Delta } { 2 \tau } \right) } _ { \mathrm { E n t r o p ~ T a i l } } } \\ & { \quad + \underbrace { \tilde { \mathcal { O } } \left( \frac { N ^ { 1 / 2 } } { \Delta ( 1 - \gamma ) ^ { 3 } \kappa T } + \frac { N ^ { - 1 / 2 } } { \Delta ( 1 - \gamma ) ^ { 8 } \kappa ^ { 6 } \tau ^ { 2 } T } + \frac { 1 } { \Delta ( 1 - \gamma ) ^ { 4 } \kappa ^ { 2 } \tau T } \right) } _ { \mathrm { T r a n s i o n t ~ C o u r ~ C o v e r o r c } } . } \end{array}
$$

(ii) Regime 1: Sample-Limited Phase. When $\log ( T _ { t o t a l } ^ { 2 / 3 } ) \leq \log ( 1 + \epsilon _ { a p p } ^ { - 1 } )$ , the temperature is tuned to $\begin{array} { r } { \tau _ { T _ { t o t a l } } = \frac { \Delta } { 2 \log ( C _ { \gamma } T _ { t o t a l } ^ { 2 / 3 } ) } } \end{array}$ . Substituting this alongside $\begin{array} { r } { A _ { m a x } \leq \frac { 1 } { 1 - \gamma } , C _ { t a i l } \triangleq \frac { A _ { m a x } C _ { \gamma } } { 1 - \gamma } \leq \frac { C _ { \gamma } } { ( 1 - \gamma ) ^ { 2 } } } \end{array}$ , and log $C _ { \gamma } = \mathcal { O } ( ( 1 - \gamma ) ^ { - 1 } )$ into the leading components yields:

Invariant Tracking Error $= \mathcal { O } \left( \frac { ( \log ( C _ { \gamma } T _ { t o t a l } ^ { 2 / 3 } ) ) ^ { 5 / 3 } } { ( 1 - \gamma ) ^ { 8 } \kappa ^ { 6 } \Delta ^ { 8 / 3 } T _ { t o t a l } ^ { 2 / 3 } } \right)$

$$
\begin{array} { r l } & { \qquad = \mathcal { O } \left( \frac { ( 1 - \gamma ) ^ { - 5 / 3 } + ( \log T _ { \mathit { t e a t a l } } ) ^ { 5 / 3 } } { ( 1 - \gamma ) ^ { 8 } \kappa ^ { 6 } \Delta ^ { 8 / 3 } T _ { \mathit { t e a t a l } } ^ { 2 / 3 } } \right) , } \\ & { \qquad \mathrm { A p p r o x i m a t i o n ~ B i a s } = \mathcal { O } \left( \frac { \log ( C _ { \gamma } T _ { \mathit { t e a d } } ^ { 2 / 3 } ) } { ( 1 - \gamma ) ^ { 2 } \Delta ^ { 2 } } \epsilon _ { a p p } \right) } \\ & { \qquad \leq \mathcal { O } \left( \frac { ( 1 - \gamma ) ^ { - 1 } + \log ( 1 + \epsilon _ { a p p } ^ { - 1 } ) } { ( 1 - \gamma ) ^ { 2 } \Delta ^ { 2 } } \epsilon _ { a p p } \right) , } \\ & { \qquad \mathrm { E n t r o p y ~ T a i l } \leq \frac { C _ { \gamma } } { ( 1 - \gamma ) ^ { 2 } } \exp \left( - \log ( C _ { \gamma } T _ { \mathit { t e a d } } ^ { 2 / 3 } ) \right) = \frac { 1 } { ( 1 - \gamma ) ^ { 2 } T _ { \mathit { t e a t a l } } ^ { 2 / 3 } } . } \end{array}
$$

The regime condition log $\cdot ( T _ { t o t a l } ^ { 2 / 3 } ) \le \log ( 1 + \epsilon _ { a p p } ^ { - 1 } )$ guarantees

$$
\log ( C _ { \gamma } T _ { t o t a l } ^ { 2 / 3 } ) \le \log C _ { \gamma } + \log ( 1 + \epsilon _ { a p p } ^ { - 1 } ) ,
$$

which gives the stated approximation-bias envelope. The $C _ { \gamma }$ prefactor cancels in the exponential tail, yielding an $\mathcal { O } ( T _ { t o t a l } ^ { - 2 / 3 } )$ tail that is absorbed by the invariant tracking error.

We next evaluate the transient errors to verify loop invariance:

• PMD Initialization Error: $\begin{array} { r } { \tilde { \mathcal { O } } \big ( \frac { N ^ { 1 / 2 } } { \Delta ( 1 - \gamma ) ^ { 3 } \kappa T } \big ) = \tilde { \mathcal { O } } \big ( \frac { N ^ { 3 / 2 } T _ { t o t a l } ^ { - 1 } } { \Delta ( 1 - \gamma ) ^ { 3 } \kappa } \big ) } \end{array}$ . To ensure that this error remains within the $\tilde { \mathcal { O } } ( T _ { t o t a l } ^ { - 2 / 3 } )$ target envelope, it sufices to require $N ^ { 3 / 2 } T _ { t o t a l } ^ { - 1 } \le T _ { t o t a l } ^ { - 2 / 3 }$ , establishing the invariance boundary $N \leq T _ { t o t a l } ^ { 2 / 9 }$

• Secondary Fractional Tracking Penalty: Under the tuned temperature,

$$
\tilde { \mathcal { O } } \left( \frac { N ^ { - 1 / 2 } } { \Delta ( 1 - \gamma ) ^ { 8 } \kappa ^ { 6 } \tau ^ { 2 } T } \right) = \tilde { \mathcal { O } } \left( \frac { N ^ { 1 / 2 } T _ { t o t a l } ^ { - 1 } ( \log ( C _ { \gamma } T _ { t o t a l } ^ { 2 / 3 } ) ) ^ { 2 } } { \Delta ^ { 3 } ( 1 - \gamma ) ^ { 8 } \kappa ^ { 6 } } \right) .
$$

For $N \leq T _ { t o t a l } ^ { 2 / 9 }$ , this term decays as $\tilde { \mathcal { O } } ( T _ { t o t a l } ^ { - 8 / 9 } )$ with respect to $T _ { t o t a l }$ and is therefore dominated by the $\tilde { \mathcal { O } } ( T _ { t o t a l } ^ { - 2 / 3 } )$ target envelope.

• PMD Local Error: Under the tuned temperature,

$$
\tilde { \mathcal { O } } \left( \frac { 1 } { \Delta ( 1 - \gamma ) ^ { 4 } \kappa ^ { 2 } \tau T } \right) = \tilde { \mathcal { O } } \left( \frac { N T _ { t o t a l } ^ { - 1 } \log ( C _ { \gamma } T _ { t o t a l } ^ { 2 / 3 } ) } { \Delta ^ { 2 } ( 1 - \gamma ) ^ { 4 } \kappa ^ { 2 } } \right) .
$$

For $N \leq T _ { t o t a l } ^ { 2 / 9 }$ , this term decays as $\tilde { \mathcal { O } } ( T _ { t o t a l } ^ { - 7 / 9 } )$ with respect to $T _ { t o t a l }$ and is also dominated by the $\tilde { \mathcal { O } } ( T _ { t o t a l } ^ { - 2 / 3 } )$ target envelope.

Thus, for any inner-loop choice $N \in [ 1 , T _ { t o t a l } ^ { 2 / 9 } ]$ , all transient errors remain within the target envelope, establishing the stated loop-invariance window.

(iii) Regime 2: Approximation-Limited Phase. When $\log ( T _ { t o t a l } ^ { 2 / 3 } ) > \log ( 1 + \epsilon _ { a p p } ^ { - 1 } )$ , the temperature locks to the approximation-dependent floor: $\begin{array} { r } { \tau _ { T _ { t o t a l } } = \frac { \Delta } { 2 \log ( C _ { \gamma } ( 1 + \epsilon _ { a p p } ^ { - 1 } ) ) } } \end{array}$ . Substituting this fixed

temperature into the leading components yields:

$$
\begin{array} { r l r } & { } & { \mathrm { I n v a r i a n t \ T r a c k i n g \ E r r o r } = \displaystyle { \mathcal O } \left( \frac { ( \log ( C _ { \gamma } ( 1 + \epsilon _ { a p p } ^ { - 1 } ) ) ) ^ { 5 / 3 } } { ( 1 - \gamma ) ^ { 8 } \kappa ^ { 6 } \Delta ^ { 8 / 3 } T _ { t o t a l } ^ { 2 / 3 } } \right) } \\ & { } & { \quad \leq \mathcal O \left( \frac { ( 1 - \gamma ) ^ { - 5 / 3 } + ( \log T _ { t o t a l } ) ^ { 5 / 3 } } { ( 1 - \gamma ) ^ { 8 } \kappa ^ { 6 } \Delta ^ { 8 / 3 } T _ { t o t a l } ^ { 2 / 3 } } \right) , } \end{array}
$$

$$
\begin{array} { l } { \displaystyle \mathrm { A p p r o x i m a t i o n ~ B i a s } = { \mathcal O } \left( \frac { \log ( C _ { \gamma } ( 1 + \epsilon _ { a p p } ^ { - 1 } ) ) } { ( 1 - \gamma ) ^ { 2 } \Delta ^ { 2 } } \epsilon _ { a p p } \right) } \\ { = { \mathcal O } \left( \frac { ( 1 - \gamma ) ^ { - 1 } + \log ( 1 + \epsilon _ { a p p } ^ { - 1 } ) } { ( 1 - \gamma ) ^ { 2 } \Delta ^ { 2 } } \epsilon _ { a p p } \right) , } \end{array}
$$

$$
\begin{array} { r l } & { \mathrm { E n t r o p y ~ T a i l } \le \displaystyle \frac { C _ { \gamma } } { ( 1 - \gamma ) ^ { 2 } } \exp \left( - \log ( C _ { \gamma } ( 1 + \epsilon _ { a p p } ^ { - 1 } ) ) \right) } \\ & { \quad \quad \quad = \displaystyle \frac { \epsilon _ { a p p } / ( 1 + \epsilon _ { a p p } ) } { ( 1 - \gamma ) ^ { 2 } } \le \frac { \epsilon _ { a p p } } { ( 1 - \gamma ) ^ { 2 } } . } \end{array}
$$

Because log $( 1 + \epsilon _ { a p p } ^ { - 1 } ) < \log ( T _ { t o t a l } ^ { 2 / 3 } )$ , we have

$$
\log ( C _ { \gamma } ( 1 + \epsilon _ { a p p } ^ { - 1 } ) ) < \log ( C _ { \gamma } ) + \log ( T _ { t o t a l } ^ { 2 / 3 } ) .
$$

Therefore, the secondary fractional tracking penalty and PMD local error are bounded by their corresponding Regime 1 envelopes, while the PMD initialization error is independent of $\tau .$ . Hence, for any inner-loop choice $N \in [ 1 , T _ { t o t a l } ^ { 2 / 9 } ]$ , all transient errors remain safely within the $\tilde { \mathcal { O } } ( T _ { t o t a l } ^ { - 2 / 3 } )$ target envelope.

(iv) Synthesis. Summing the evaluated components across the two regimes yields:

$$
\begin{array} { r l } & { \displaystyle \frac { 1 } { T } \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } \big [ J _ { 0 } ( \pi _ { 0 } ^ { * } ) - J _ { 0 } ( \pi _ { t } ) \big ] \leq \mathcal { O } \left( \frac { ( 1 - \gamma ) ^ { - 5 / 3 } + ( \log T _ { t o t a l } ) ^ { 5 / 3 } } { ( 1 - \gamma ) ^ { 8 } \kappa ^ { 6 } \Delta ^ { 8 / 3 } T _ { t o t a l } ^ { 2 / 3 } } \right) } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad + \mathcal { O } \left( \frac { ( 1 - \gamma ) ^ { - 1 } + \log ( 1 + \epsilon _ { a p p } ^ { - 1 } ) } { ( 1 - \gamma ) ^ { 2 } \Delta ^ { 2 } } \epsilon _ { a p p } \right) . } \end{array}
$$

Finally, as $\epsilon _ { a p p }  0 , \log ( 1 + \epsilon _ { a p p } ^ { - 1 } ) = \Theta ( \log ( 1 / \epsilon _ { a p p } ) )$ and $\epsilon _ { a p p } \log ( 1 + \epsilon _ { a p p } ^ { - 1 } ) = \tilde { \mathcal { O } } ( \epsilon _ { a p p } )$ . Therefore, the overall convergence rate is $\tilde { \mathcal { O } } ( T _ { t o t a l } ^ { - 2 / 3 } ) + \tilde { \mathcal { O } } ( \epsilon _ { a p p } )$ , completing the proof. □

## G.4 Warm-Start Unregularized Convergence (Linear Entropy)

To expose the underlying barrier bypassed by the exponential translation mechanism, we provide a comparative unregularized convergence analysis relying on the standard linear entropy penalty for Algorithm 5. Crucially, this result does not require the Minimal Action Gap assumption, providing a robust theoretical fallback for environments lacking a uniformly positive action margin. By balancing the invariant tracking error against the $\mathcal { O } ( \tau )$ entropy penalty, we establish a $\tilde { \mathcal { O } } ( T _ { t o t a l } ^ { - 1 / 4 } )$ average-iterate unregularized convergence rate, which maintains its loop invariance across

the shifted inner-loop window $N \in [ 1 , \sqrt { T _ { t o t a l } } ]$

Theorem G.3 (Warm-Start Linear Entropy Convergence). For Algorithm $^ { 5 , }$ suppose that $A s _ { - }$ sumptions $1 \mathrm { - } \mathit { 4 }$ and 9 hold. Define the two-stage temperature:

$$
\tau _ { T _ { t o t a l } } \triangleq \operatorname* { m a x } \left( \frac { 1 } { ( 1 - \gamma ) ^ { 9 / 4 } \kappa ^ { 9 / 4 } T _ { t o t a l } ^ { 1 / 4 } } , \sqrt { \epsilon _ { a p p } } \right) .
$$

For any intermediate inner-loop choice $N \in [ 1 , \sqrt { T _ { t o t a l } } ]$ and $T = T _ { t o t a l } / N$ , if $T _ { t o t a l }$ is suficiently large and $\epsilon _ { a p p }$ is suficiently small such that $\tau _ { T _ { t o t a l } } \leq \tau _ { m a x } .$ , then by choosing the step-size parameters as in Theorem G.1 evaluated at $\tau = \tau _ { T _ { t o t a l } }$ , the expected average unregularized performance gap satisfies the convergence rate:

$$
\begin{array} { r l } & { \frac { 1 } { T } \displaystyle \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } \big [ J _ { 0 } ( \pi _ { 0 } ^ { * } ) - J _ { 0 } ( \pi _ { t } ) \big ] \leq \tilde { \mathcal { O } } \left( \frac { 1 } { ( 1 - \gamma ) ^ { 1 3 / 4 } \kappa ^ { 9 / 4 } T _ { t o t a l } ^ { 1 / 4 } } \right) + \mathcal { O } \left( \frac { \sqrt { \epsilon _ { a p p } } } { 1 - \gamma } \right) } \\ & { \quad \quad \quad \quad \quad = \tilde { \mathcal { O } } \left( T _ { t o t a l } ^ { - 1 / 4 } \right) + \mathcal { O } ( \sqrt { \epsilon _ { a p p } } ) . } \end{array}
$$

Proof. (i) Global Unregularized Bound. Because the policy entropy is non-negative, evaluating the optimal policy under the regularized objective yields $J _ { 0 } ( \pi _ { 0 } ^ { * } ) \le J _ { \tau } ( \pi _ { 0 } ^ { * } ) \le J _ { \tau } ( \pi _ { \tau } ^ { * } )$ . For any training policy $\pi _ { t } ,$ the global unregularized performance gap is bounded by the global regularized performance gap plus the linear entropy penalty:

$$
\begin{array} { l } { { J _ { 0 } ( \pi _ { 0 } ^ { * } ) - J _ { 0 } ( \pi _ { t } ) \le J _ { \tau } ( \pi _ { \tau } ^ { * } ) - J _ { 0 } ( \pi _ { t } ) } } \\ { \displaystyle \qquad = \Big ( J _ { \tau } ( \pi _ { \tau } ^ { * } ) - J _ { \tau } ( \pi _ { t } ) \Big ) + \Big ( J _ { \tau } ( \pi _ { t } ) - J _ { 0 } ( \pi _ { t } ) \Big ) } \\ { \displaystyle \qquad \le \mathrm { G a p } _ { t } ^ { \dag } + \frac { \tau \log | \cal A | } { 1 - \gamma } . } \end{array}
$$

Taking the expectation and substituting the average warm-start regularized gap from Theorem G.1 explicitly yields the total unregularized error bound:

$$
\begin{array} { r l } & { \underbrace { \frac { 1 } { T } \displaystyle \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } \big [ J _ { 0 } ( { \boldsymbol { \pi } } _ { 0 } ^ { * } ) - J _ { 0 } ( { \boldsymbol { \pi } } _ { t } ) \big ] } _ { \mathrm { \normalfont { I v e r }  { \left( { \frac { 1 } { \delta } } \right) } } \chi ^ { 6 } \tau ^ { 5 / 3 } T _ { t o t a l } ^ { 2 / 3 } } } \\ & { \leq \underbrace { \mathcal { O } \left( \frac { 1 } { ( 1 - \gamma ) ^ { 7 } \kappa ^ { 6 } \tau ^ { 5 / 3 } T _ { t o t a l } ^ { 2 / 3 } } \right) } _ { \mathrm { \normalfont { I n v a r i a n t ~ T r a c k i n g ~ E r r r o r r } } } \cdot \underbrace { \mathcal { O } \left( \frac { \epsilon _ { a p p } } { ( 1 - \gamma ) \tau } \right) } _ { \mathrm { \normalfont { A p p r o s i m a t i o n ~ B i a s } } } + \underbrace { \mathcal { O } \left( \frac { \tau } { 1 - \gamma } \right) } _ { \mathrm { \normalfont { F n u t r o p y ~ P e n a l t y } } } } \\ & { \quad + \underbrace { \tilde { \mathcal { O } } \left( \frac { N ^ { 1 / 2 } } { ( 1 - \gamma ) ^ { 2 } \kappa T } + \frac { N ^ { - 1 / 2 } } { ( 1 - \gamma ) ^ { 7 } \kappa ^ { 6 } \tau ^ { 2 } T } + \frac { 1 } { ( 1 - \gamma ) ^ { 3 } \kappa ^ { 2 } \tau T } \right) } _ { \mathrm { \normalfont { I n - \gamma }  { \left( { \frac { 1 } { \delta } } \right) } } \tau ^ { 2 } T } . } \end{array}
$$

(ii) Regime 1: Sample-Limited Phase. When $\frac { 1 } { ( 1 - \gamma ) ^ { 9 / 4 } \kappa ^ { 9 / 4 } T _ { t o t a l } ^ { 1 / 4 } } \geq \sqrt { \epsilon _ { a p p } }$ , we balance the Invariant Tracking Error against the Entropy Penalty. Setting $\begin{array} { r } { \frac { 1 } { ( 1 - \gamma ) ^ { 7 } \kappa ^ { 6 } \tau ^ { 5 / 3 } T _ { t o t a l } ^ { 2 / 3 } } = \frac { \tau } { 1 - \gamma } } \end{array}$ yields the tuned temperature $\begin{array} { r } { \tau _ { T _ { t o t a l } } = \frac { 1 } { ( 1 - \gamma ) ^ { 9 / 4 } \kappa ^ { 9 / 4 } T _ { t o t a l } ^ { 1 / 4 } } } \end{array}$ . Under this tuning, both the Invariant Tracking Error and the Entropy Penalty evaluate to $\mathcal { O } \big ( \frac { 1 } { ( 1 - \gamma ) ^ { 1 3 / 4 } \kappa ^ { 9 / 4 } T _ { t o t a l } ^ { 1 / 4 } } \big )$ , and the Approximation Bias is dominated by this leading order. We evaluate the transient errors under this tuning to verify loop invariance:

• PMD Initialization Error: $\begin{array} { r } { \tilde { \mathcal { O } } \big ( \frac { N ^ { 1 / 2 } } { ( 1 - \gamma ) ^ { 2 } \kappa T } \big ) = \tilde { \mathcal { O } } \big ( \frac { N ^ { 3 / 2 } T _ { t o t a l } ^ { - 1 } } { ( 1 - \gamma ) ^ { 2 } \kappa } \big ) } \end{array}$ . To ensure that this error remains within the $\tilde { \mathcal { O } } ( T _ { t o t a l } ^ { - 1 / 4 } )$ target envelope, it sufices to require $N ^ { 3 / 2 } T _ { t o t a l } ^ { - 1 } \le T _ { t o t a l } ^ { - 1 / 4 }$ , establishing the invariance boundary $N \leq \sqrt { T _ { t o t a l } }$

• Secondary Fractional Tracking Penalty: $\begin{array} { r } { \tilde { \mathcal { O } } \big ( \frac { N ^ { - 1 / 2 } } { ( 1 - \gamma ) ^ { 7 } \kappa ^ { 6 } \tau ^ { 2 } T } \big ) ~ = ~ \tilde { \mathcal { O } } \big ( \frac { N ^ { 1 / 2 } T _ { t o t a l } ^ { - 1 } } { ( 1 - \gamma ) ^ { 7 } \kappa ^ { 6 } \tau ^ { 2 } } \big ) } \end{array}$ . Because $\tau =$ $\Theta ( T _ { t o t a l } ^ { - 1 / 4 } )$ , this term evaluates to $\tilde { \mathcal { O } } ( N ^ { 1 / 2 } T _ { t o t a l } ^ { - 1 / 2 } )$ . For $N \leq \sqrt { T _ { t o t a l } } .$ , it is bounded by $\tilde { \mathcal { O } } ( T _ { t o t a l } ^ { - 1 / 4 } )$ and therefore remains within the target envelope.

• PMD Local Error: $\begin{array} { r } { \tilde { \mathcal { O } } \big ( \frac { 1 } { ( 1 - \gamma ) ^ { 3 } \kappa ^ { 2 } \tau T } \big ) = \tilde { \mathcal { O } } \big ( \frac { N T _ { t o t a l } ^ { - 1 } } { ( 1 - \gamma ) ^ { 3 } \kappa ^ { 2 } \tau } \big ) } \end{array}$ . Because $\tau = \Theta ( T _ { t o t a l } ^ { - 1 / 4 } )$ , this term evaluates to $\tilde { \mathcal { O } } ( N T _ { t o t a l } ^ { - 3 / 4 } )$ . For $N \leq \sqrt { T _ { t o t a l } }$ , it is bounded by $\tilde { \mathcal { O } } ( T _ { t o t a l } ^ { - 1 / 4 } )$ and therefore also remains within the target envelope.

Thus, for any inner-loop choice $N \in [ 1 , \sqrt { T _ { t o t a l } } ]$ , all transient errors remain within the $\tilde { \mathcal { O } } ( T _ { t o t a l } ^ { - 1 / 4 } )$ target envelope, establishing the stated loop-invariance window.

(iii) Regime 2: Approximation-Limited Phase. When $T _ { t o t a l }$ grows such that $\frac { 1 } { ( 1 - \gamma ) ^ { 9 / 4 } \kappa ^ { 9 / 4 } T _ { t o t a l } ^ { 1 / 4 } } <$ $\sqrt { \epsilon _ { a p p } }$ , the temperature locks to the constant floor $\tau _ { T _ { t o t a l } } = \sqrt { \epsilon _ { a p p } }$ . The Approximation Bias and the Entropy Penalty evaluate to $\mathcal { O } \big ( \textstyle \frac { \sqrt { \epsilon _ { a p p } } } { 1 - \gamma } \big )$ , and the Invariant Tracking Error is dominated by this leading order. Because $\frac { 1 } { ( 1 - \gamma ) ^ { 9 / 4 } \kappa ^ { 9 / 4 } T _ { t o t a l } ^ { 1 / 4 } } < \sqrt { \epsilon _ { a p p } } ,$ the secondary fractional tracking penalty and PMD local error are bounded by their corresponding Regime 1 envelopes, while the PMD initialization error is independent of τ. Thus, for any inner-loop choice $N \in [ 1 , \sqrt { T _ { t o t a l } } ]$ , all transient errors remain safely within the $\tilde { \mathcal { O } } ( T _ { t o t a l } ^ { - 1 / 4 } )$ target envelope.

(iv) Synthesis. By setting the tuned temperature $\tau _ { T _ { t o t a l } }$ to the maximum of these two balanced bounds, the expected average unregularized performance gap is bounded by the sum of their respective envelopes, completing the proof. □