# Direct Acceleration of Stochastic Root-Finding Without Variance Reduction and Regularization

TaeHo Yoon<sup>\*</sup>

Nicolas Loizou<sup>\*</sup>

## Abstract

Acceleration for deterministic root-finding problems has been extensively studied in recent years; specifically, the anchor-based, or Halpern-type methods achieve optimal convergence rates with respect to the operator norm. However, acceleration via these methods does not directly carry over to stochastic setting due to accumulation of errors, unless one enforces diminishing variance via increasing batch sizes or variance reduction techniques. In this work, we show that another class of acceleration, namely the dual-anchor mechanism, extends to the stochastic setting without such error accumulation, in contrast to anchor-based algorithms. Consequently, we cleanly achieve $\mathcal { O } ( \epsilon ^ { - 3 } )$ complexity with iteration-independent batch size, without any variance reduction or double-loop recursive regularization, for stochastic root-finding (resp. fixed-point) problems with cocoercivity (resp. square-nonexpansivity) in expectation. For strongly monotone operators, the same algorithm attains a sharper $\widetilde { \mathcal { O } } ( \epsilon ^ { - 2 } )$ complexity, nearly matching the lower bound in terms of ϵ-dependence.

## 1 Introduction

Root-finding problems for an operator $\mathbb { F } \colon  { \mathbb { R } ^ { d } } \to  { \mathbb { R } ^ { d } }$ , stated as

$$
{ \underset { x \in \mathbb { R } ^ { d } } { \mathrm { f i n d } } } \quad \mathbb { F } ( x ) = 0 ,\tag{1}
$$

generalize minimization problems and have been studied under distinct formulations including monotone inclusion [36, 42, 3, 43], minimax optimization [38, 19, 33, 13, 35, 31], equilibrium search [47, 17, 46], or fixed-point problems [24, 34, 21, 51, 45]. In the past few years, there have been significant developments in accelerated algorithms for reducing the squared operator/residual norm with the optimal $\| \mathbb { F } ( x _ { k } ) \| ^ { 2 } =$ $\mathcal { O } ( 1 / k ^ { 2 } )$ rate [29, 16, 22, 40, 53, 27]. A recurring message from the optimization literature, however, is that deterministic acceleration is often brittle under oracle errors; in the stochastic setting where $\mathbb { F } ( x ) =$ $\mathbb { E } \left\lceil \widehat { \mathbb { F } } ( x ; \xi ) \right\rceil$ and the algorithm uses $\widehat { \mathbb { F } } ( x ; \xi )$ , error may accumulate quickly with these accelerated methods and break their fast convergence from the noiseless setting. Consequently, for stochastic optimization problems, taking advantage of the study of acceleration was not straightforward—it required decaying oracle inexactness, variance reduction, restart mechanisms, or stronger assumptions on the problem [15, 18, 28, 9]. Specifically for stochastic monotone inclusion problems, stochastic Halpern iteration and accelerated (anchored) extragradient type analyses sufered precisely from these issues [27, 9, 10].

We take a diferent viewpoint toward handling stochastic instability. Recent work [55] showed that the acceleration mechanism in fixed-point and minimax problems is not unique: besides anchor-based algorithms [29, 22, 53, 27, 50], there exist their H-dual algorithms, which display materially diferent update rules but share exactly the same deterministic last-iterate rate with the optimal anchor-based acceleration. Hence, the deterministic worst-case complexity alone does not identify a single preferred accelerated method. In this paper, we demonstrate that the stochastic extension can break this tie: the Stochastic Dual Optimal Halpern Method (S-Dual-OHM) attains $\mathbb { E } \left[ \lVert \mathbb { F } ( x _ { N - 1 } ) \rVert \right] \leq \epsilon$ ϵ with $\mathcal { O } ( \epsilon ^ { - 3 } )$ oracle complexity, with iteration complexity $N = \mathcal { O } ( \epsilon ^ { - 1 } )$ and with constant mini-batch size $\mathcal { O } ( \epsilon ^ { - 2 } )$ . This is done cleanly and more directly without assuming diminishing variance or using variance reduction techniques, which is not what stochastic anchoring methods were capable of [27, 9, 10].

Table 1: Comparison of stochastic algorithms for finding an ϵ-approximate solution with $\mathbb { E } [ \| \mathbb { F } ( x ) \| ] \le \epsilon$ for 1/L-cocoercive operator �.
<table><tr><td>Algorithms</td><td>Complexity</td><td>Last- iterate</td><td>Variance- reduction free</td><td>Single- loop</td><td>Algorithm parameters</td><td>Theory requirements</td></tr><tr><td>SGDA/SEG [20]</td><td> $\mathcal { O } ( \epsilon ^ { - 4 } ) ^ { \dagger }$ </td><td>x</td><td>√</td><td>√</td><td> $L , B$ </td><td> $D , \sigma$ </td></tr><tr><td>S-HALPERN-PAGE [9]</td><td> $\mathcal { O } ( \epsilon ^ { - 3 } )$ </td><td>√</td><td>x</td><td>√</td><td> $L , p _ { k } , S _ { 1 } ^ { ( k ) } , S _ { 2 } ^ { ( k ) }$ </td><td> $D , \sigma$ </td></tr><tr><td>S-HALPERN-VR-FINITE [10]</td><td> ${ \widetilde { \mathcal { O } } } ( n + { \sqrt { n } } \epsilon ^ { - 1 } ) ^ { \sharp }$ </td><td>√</td><td>x</td><td>√</td><td> $L , B , \lambda _ { k } ; p _ { k } \ \mathrm { o r } \ M _ { k }$ </td><td> $D , n$ </td></tr><tr><td>RAIN [12]</td><td> $\widetilde { \mathcal { O } } ( \epsilon ^ { - 2 } )$ </td><td>√</td><td>√</td><td>x*</td><td> $L , \lambda , \gamma , N _ { s } , K _ { s }$ </td><td> $D , \sigma$ </td></tr><tr><td>S-DUAL-OHM (This work)</td><td> $\mathcal { O } ( \epsilon ^ { - 3 } )$ </td><td>√</td><td>√</td><td></td><td> $L , B , N$ </td><td> $D , \sigma$ </td></tr></table>

SGDA denotes Stochastic Gradient Descent-Ascent $x _ { k + 1 } = x _ { k } - \alpha \mathbb { F } ( x _ { k } ; \xi _ { k } )$ and SEG denotes stochastic version of Extragradient [23]. “Last-iterate” indicates whether the stated guarantee is for the final iterate and not the averaged, best, or uniform random iterate. “Variance-reduction free” indicates whether the algorithm avoids using variance reduction techniques. “Single-loop” indicates whether the algorithm avoids using outer–inner loop structure (e.g., regularization techniques). “Algorithm parameters” lists the hyperparemeters whose values should be specified for the run, where the cocoercivity/Lipschitzness parameter L is used for determining the step-sizes of the algorithm. “Theory requirements” lists the information of problem-related parameters that are needed for selecting the hyperparameters that theoretically guarantee $\mathbb { E } [ \| \mathbb { F } ( x ) \| ] \le \epsilon .$ Here N is the total number of algorithm iterations, $D = \| \bar { \boldsymbol { x } } _ { 0 } - \boldsymbol { x } _ { \star } \| ,$ σ is a bound on the operator variance, and n is the number of component operators in the case where the problem has a finite-sum structure. The $\widetilde { \mathcal { O } } ( \cdot )$ notation hides logarithmic factors.  
<sup>†</sup>[20, Corollary E.4] shows this for uniform random iterate of SEG, and a similar proof for SGDA can be derived.  
<sup>‡</sup>Works only for finite-sum problems where $\begin{array} { r } { \mathbb { F } ( \cdot ) = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \mathbb { F } ( \cdot ; \xi _ { i } ) . } \end{array}$  
<sup>\*</sup>[12] presents a single-loop version of RAIN for numerical experiments (which we use in our experiments for comparison), but their theoretical convergence analysis still requires a nontrivial double-loop structure.

In our view, this result has a conceptual implication beyond simply providing another eficient stochastic algorithm. It challenges the folklore view that acceleration, as studied in deterministic optimization, is brittle and incompatible with stochastic noise. Equivalently, optimal deterministic methods can behave very diferently under stochastic perturbations, and selecting the right representation of acceleration may enable the design of eficient stochastic algorithms.

Contributions. We summarize our main contributions as follows.

⋄ We show that under cocoercivity of � in expectation, our proposed S-Dual-OHM with mini-batching over $\widehat { \mathbb { F } } ( \cdot ; \xi )$ with constant step-size $\textstyle \alpha \in \left( 0 , { \frac { 2 } { L } } \right]$ and constant batch size $B = \mathcal { O } ( \epsilon ^ { - 2 } )$ achieves the bound $\mathbb { E } \left[ \lVert \mathbb { F } ( x _ { N - 1 } ) \rVert \right] \leq \epsilon$ within $N = \mathcal { O } ( \epsilon ^ { - 1 } )$ iterations. The resulting total oracle complexity $\mathcal { O } ( \epsilon ^ { - 3 } )$ , to the best of our knowledge, has not been achieved in prior work without variance reduction or doubleloop regularization techniques (Table 1).

⋄ We show that when � is additionally strongly monotone, the same S-Dual-OHM algorithm can be early-stopped in $k = \widetilde { \mathcal { O } } ( L / \mu \log { \epsilon ^ { - 1 } } )$ iterations to attain $\mathbb { E } \left[ \lVert \mathbb { F } ( x _ { k } ) \rVert \right] \leq \epsilon .$ . This yields the oracle complexity $\widetilde { \mathcal { O } } ( \epsilon ^ { - 2 } )$ , which has near-optimal dependence on ϵ.

⋄ We provide numerical experiments demonstrating that Dual-OHM is indeed more robust to stochastic noise than Halpern iteration, and that S-Dual-OHM can efectively solve stochastic root-finding problems.

## 2 Related Work

Fixed-point problems, Halpern iteration and acceleration. For fixed-point problems

$$
\operatorname { f i n d } _ { x \in \mathbb { R } ^ { d } } \ x = \mathbb { T } ( x ) ,\tag{2}
$$

with a nonexpansive operator $\mathbb { T } \colon  { \mathbb { R } ^ { d } } \to  { \mathbb { R } ^ { d } }$ , the Halpern iteration $x _ { k + 1 } = \beta _ { k } x _ { 0 } + ( 1 - \beta _ { k } ) \mathbb { T } x _ { k }$ has been studied over a long history in the literature [21, 51], and has been found to provide accelerated worst-case convergence rate $\| x _ { k } - \mathbb { T } ( x _ { k } ) \| = \mathcal { O } ( 1 / \bar { k } )$ with $\beta _ { k } = \mathcal { O } ( 1 / k )$ [45, 16, 29, 22, 25]. The best choice $\begin{array} { r } { \beta _ { k } = \frac { 1 } { k + 2 } } \end{array}$ , due to [29, 22], was later shown to yield an optimal rate $\begin{array} { r } { { \left\| { \boldsymbol x } _ { k } - { \mathbb T } ( { \boldsymbol x } _ { k } ) \right\| } ^ { 2 } \leq \frac { 4 \| { \boldsymbol x } _ { 0 } - { \boldsymbol x } _ { \star } \| ^ { 2 } } { ( k + 1 ) ^ { 2 } } } \end{array}$ exactly matching the lower bound [40]. More recently, it was discovered that the same exact optimal rate for a fixed finite horizon N can be achieved by distinct types of acceleration forming an infinite family of algorithms [56] including Dual-OHM [55], the basis of this work.

Acceleration of minimax optimization and monotone inclusion. Minimax optimization with convex-concave objective can be recast into monotone inclusion, i.e., root-finding problem (1) with saddle gradient operator �, which is a monotone operator [42, 17]. Some early works including [44, 16] first connected Halpern acceleration to these problems, and [16] achieved near-optimal complexity for Lipschitz continuous monotone inclusions. The remaining logarithmic gap was removed by the Extra Anchored Gradient algorithm [53], and the so-called anchor acceleration for minimax optimization and monotone inclusion has been studied extensively since then [27, 50, 54, 48, 11, 5, 1, 26]. Several works proposed its extension via distinct interpretation in connection with Nesterov momentum [6, 49]. On the other hand, the dual-anchor acceleration analogous to Dual-OHM from fixed-point problems is achievable in this setting as well [55].

Residual-reducing algorithms for stochastic fixed-point and monotone inclusion problems. The Fast Extragradient algorithm [27], an anchor-based accelerated minimax algorithm, was proposed with its stochastic extension under the assumption that the operator’s variance at k-th iteration diminishes at the order of $\mathcal { O } ( 1 / k )$ . Then [9] showed that stochastic Halpern iteration and its minimax variants, combined with PAGE variance reduction [28] can reduce the complexity of achieving $\mathbb { E } \left[ \lVert \mathbb { F } ( x _ { k } ) \rVert \right] \leq \epsilon$ to $\mathcal { O } ( \epsilon ^ { - 3 } )$ , which could be further improved to $\widetilde { \mathcal { O } } ( n + \sqrt { n } \epsilon ^ { - 1 } )$ [10] in the finite-sum setting where � is the average of $n = o ( \epsilon ^ { - 3 } )$ sample operators. Near-optimal complexity of $\widetilde { \mathcal { O } } ( \epsilon ^ { - 2 } )$ for stochastic monotone inclusion has been achieved in [12] with the matching $\Omega ( \epsilon ^ { - 2 } )$ lower bound via more sophisticated recursive anchoring (regularization) algorithm. On another line of work, $[ 7 , 4 1 , 8 ]$ studied stochastic Krasnosel’ski˘ı–Mann or Halpern iterations in general normed spaces, where the lower bound of $\Omega ( \epsilon ^ { - 3 } )$ was established [8].

Broader view of extending acceleration to stochastic settings. In smooth convex minimization, it is now standard that accelerated methods are fragile to persistent oracle error: preserving acceleration under stochasticity typically requires either variance reduction or catalyst-type outer–inner regularization [15, 30, 2]. A similar viewpoint has been adopted for monotone root-finding problems in prior works [9, 10, 12]. Our approach departs from this line of work. Rather than stabilizing Halpern-type acceleration by additional variance-reduction or regularization, we leverage the fact that deterministic acceleration is not unique, and there exist distinct optimal accelerated algorithms with the same worst-case deterministic rate. We deliver the new observation that among them, Dual-OHM is intrinsi cally less sensitive to stochastic perturbations than Halpern-type algorithms, enabling direct stochastic extension.

## 3 Preliminaries and Assumptions

In this section, we establish some necessary preliminary concepts for subsequent analysis.

## 3.1 Monotonicity, cocoercivity, and nonexpansiveness of operators

We say that an operator $\mathbb { F } \colon  { \mathbb { R } ^ { d } } \to  { \mathbb { R } ^ { d } }$ is

\- Monotone if $\langle \mathbb { F } ( x ) - \mathbb { F } ( y ) , x - y \rangle \geq 0$ for all $x , y \in \mathbb { R } ^ { d }$

\- µ-strongly monotone if $\langle \mathbb { F } ( x ) - \mathbb { F } ( y ) , x - y \rangle \geq \mu \left. x - y \right. ^ { 2 }$ for all $x , y \in \mathbb { R } ^ { d }$

\- M-Lipschitz if $\| \mathbb { F } ( x ) - \mathbb { F } ( y ) \| \le M \| x - y \|$ for all $x , y \in \mathbb { R } ^ { d }$

\- 1/L-Cocoercive if $\begin{array} { r } { \langle \mathbb { F } ( x ) - \mathbb { F } ( y ) , x - y \rangle \geq \frac { 1 } { L } \left. \mathbb { F } ( x ) - \mathbb { F } ( y ) \right. ^ { 2 } } \end{array}$ for all $x , y \in \mathbb { R } ^ { d }$

where $\mu , M , L > 0$ . In particular, 1/L-cocoercivity implies monotonicity and L-Lipschitzness, but not vice versa. If � is M-Lipschitz and µ-strongly monotone, then it is 1/L-cocoercive with $L = M ^ { 2 } / \mu$ [17]. When � is 1/L-cocoercive, the problem (1) can be recast into a fixed-point problem (2) where $\mathbf { \bar { T } } = \mathbf { I } - \alpha \mathbf { F } \colon \mathbb { R } ^ { d } \to \mathbb { R } ^ { d }$ is nonexpansive, i.e., 1-Lipschitz for $\begin{array} { r } { 0 < \alpha \le \frac { 2 } { L } \ [ 3 } \end{array}$ , Proposition 4.39], where $\mathbb { I } \colon  { \mathbb { R } ^ { d } } \to  { \mathbb { R } ^ { d } }$ is the identity operator. This is because

$$
\mathrm { Z e r } \left( \mathbb { F } \right) : = \left\{ x \in \mathbb { R } ^ { d } \vert \mathbb { F } ( x ) = 0 \right\} = \left\{ x \in \mathbb { R } ^ { d } \vert \mathbb { T } ( x ) = x \right\} : = \mathrm { F i x } \left( \mathbb { T } \right) .
$$

Given an unconstrained minimax problem

$$
\operatorname* { m i n i m i z e } _ { \boldsymbol { u } \in \mathbb { R } ^ { m } } \operatorname* { m a x i m i z e } _ { \boldsymbol { v } \in \mathbb { R } ^ { n } } \boldsymbol { \Phi } ( \boldsymbol { u } , \boldsymbol { v } )\tag{3}
$$

with a convex-concave (resp. µ-strongly-convex-strongly-concave) and M-smooth Φ: $\mathbb { R } ^ { m } \times \mathbb { R } ^ { n } \to \mathbb { R }$ , its saddle operator

$$
\mathbb { F } ( u , v ) = ( \nabla _ { u } \Phi ( u , v ) , - \nabla _ { v } \Phi ( u , v ) )
$$

is monotone (resp. µ-strongly monotone) and M-Lipschitz $[ 1 7 ]$

## 3.2 Stochastic oracle models

We consider an unbiased stochastic operator oracle with bounded variance.

Assumption 3.1 (Stochastic operator oracle). Given $x \in \mathbb { R } ^ { d }$ , one can query $\widehat { \mathbb { F } } ( x ; \xi )$ such that

$$
\begin{array} { r } { \mathbb { E } \left[ \widehat { \mathbb { F } } ( x ; \xi ) \right] = \mathbb { F } ( x ) , \qquad \mathbb { E } \left[ \left\| \widehat { \mathbb { F } } ( x ; \xi ) - \mathbb { F } ( x ) \right\| ^ { 2 } \right] \leq \sigma ^ { 2 } . } \end{array}
$$

For our main result (Theorem 4.1), we additionally assume the following condition.

Assumption 3.2 (Cocoercivity in expectation). For any $x , y \in \mathbb { R } ^ { d }$ , we have

$$
\langle \mathbb { F } ( x ) - \mathbb { F } ( y ) , x - y \rangle \geq \mathbb { E } \left[ \frac { 1 } { L } \left\| \widehat { \mathbb { F } } ( x ; \xi ) - \widehat { \mathbb { F } } ( y ; \xi ) \right\| ^ { 2 } \right] .
$$

This condition holds for some natural cases, including: (i) stochastic oracle with additive noise $\widehat { \mathbb { F } } ( x ; \xi ) = \mathbb { F } ( x ) + \zeta ( \xi )$ , where the noise $\zeta ( \xi )$ is independent of $x ,$ (ii) finite sum or expectation models where each sample operator is 1/L-cocoercive, and as its special case, (iii) sample operators that are each µ-strongly monotone $( \mu > 0 )$ and M-Lipschitz, which implies sample-wise cocoercivity [17]. Note that Assumption 3.2 does not capture general monotone and Lipschitz sample operators or the cases where the noise $\zeta ( \xi )$ depends on $x .$ . Similar assumptions or even stronger samplewise cocoercivity have been commonly used in stochastic optimization [39, 2, 32, 37, 4, 58, 57, 14, 10].

Given batch size $B \geq 1$ , we denote $\begin{array} { r } { \mathbb { F } _ { \boldsymbol { B } } ( \boldsymbol { x } ) = \frac { 1 } { B } \sum _ { b = 1 } ^ { B } \widehat { \mathbb { F } } ( \boldsymbol { x } ; \boldsymbol { \xi } _ { b } ) } \end{array}$ , where $\xi _ { b }$ are conditionally independent samples given x and $\boldsymbol { \mathcal { B } } = \{ \xi _ { 1 } , . . . , \xi _ { B } \}$ . Then $\mathbb { F } _ { B } ( \boldsymbol { x } )$ is an unbiased estimator of $\mathbb { F } ( x )$ with $\begin{array} { r } { \mathbb { E } \left\lceil \left\| \mathbb { F } _ { B } ( x ) - \mathbb { F } ( x ) \right\| ^ { 2 } \right\rceil \leq \frac { \sigma ^ { 2 } } { B } } \end{array}$ and $\begin{array} { r } { \langle \mathbb { F } ( x ) - \mathbb { F } ( y ) , x - y \rangle \geq \mathbb { E } \left[ \frac { 1 } { L } \left. \mathbb { F } _ { B } ( x ) - \mathbb { F } _ { B } ( y ) \right. ^ { 2 } \right] } \end{array}$ by Jensen’s inequality. For $\begin{array} { r } { 0 < \alpha \le \frac { 2 } { L } . } \end{array}$ , if we let ${ \widehat { \mathbb { T } } } ( \cdot ; \xi ) = \mathbb { I } - \alpha { \widehat { \mathbb { F } } } ( \cdot ; \xi )$ and $\begin{array} { r } { \mathbb { T } _ { \mathcal { B } } ( \cdot ) = \frac { 1 } { B } \sum _ { b = 1 } ^ { B } \widehat { \mathbb { T } } ( \cdot ; \xi _ { b } ) } \end{array}$ , then these are unbiased estimators of the nonexpansive $\mathbb { T } = \mathbb { I } - \alpha \mathbb { F }$ with variance bounds $\alpha ^ { 2 } \sigma ^ { 2 }$ and $\frac { \alpha ^ { 2 } \sigma ^ { 2 } } { B }$ , respectively. Furthermore, we have: $\mathbb { E } \left[ \left. \mathbb { T } _ { B } ( x ) - \mathbb { T } _ { B } ( y ) \right. ^ { 2 } \right] \leq \left. x - y \right. ^ { 2 }$ (square-nonexpansivity in expectation).

## 3.3 Known acceleration in fixed-point problems and its extension

Our general strategy is to reformulate (1) with cocoercive � into (2) with $\mathbb { T } = \mathbb { I } - \alpha \mathbb { F }$ and build upon deterministic acceleration results. Consider the following algorithms: given $y _ { 0 } \in \mathbb { R } ^ { d }$

$$
y _ { k + 1 } = { \frac { 1 } { k + 2 } } y _ { 0 } + { \frac { k + 1 } { k + 2 } } \mathbb { T } y _ { k }\tag{OHM}
$$

and, with the convention $\mathbb { T } y _ { - 1 } = y _ { 0 }$

$$
y _ { k + 1 } = y _ { k } + { \frac { N - k - 1 } { N - k } } \left( \mathbb { T } y _ { k } - \mathbb { T } y _ { k - 1 } \right)\tag{Dual-OHM}
$$

Here, OHM is an anytime algorithm whose update is defined for all $k = 0 , 1 , \ldots ,$ while for Dual-OHM, the terminal iteration number N is fixed, and the update is defined only for $k = 0 , \ldots , N - 2$ . This is a substantive distinction that makes the two algorithms behave diferently under stochastic extension (see the discussion in Section 4.1.1 for further details).

Proposition 3.3 (Deterministic acceleration [29, 22, 55]). For a nonexpansive operator �: $\mathbb { R } ^ { d } \to \mathbb { R } ^ { d }$ with a fixed point $y _ { \star } \in \operatorname { F i x } \mathbb { T }$ and $N \geq 2$ , both OHM and Dual-OHM exhibit the rate

$$
\left\| y _ { N - 1 } - \mathbb { T } ( y _ { N - 1 } ) \right\| ^ { 2 } \leq \frac { 4 \left\| y _ { 0 } - y _ { \star } \right\| ^ { 2 } } { N ^ { 2 } } .
$$

With $\mathbb { T } = \mathbb { I } - \alpha \mathbb { F }$ and $\begin{array} { r } { \alpha = \frac { 1 } { L } } \end{array}$ , Proposition 3.3 shows that $\begin{array} { r } { \| \mathbb { F } ( y _ { N - 1 } ) \| = \frac { \| y _ { N - 1 } - \mathbb { T } ( y _ { N - 1 } ) \| } { \alpha } \le \epsilon } \end{array}$ is achieved in $\mathcal { O } \left( \begin{array} { c c } { \underline { { L D } } } \\ { \epsilon } \end{array} \right)$ iterations, where $D = \| y _ { 0 } - y _ { \star } \|$

Proposition 3.4 (Cai et al. [9], Informal). Stochastic OHM (S-OHM), which has the update rule $\begin{array} { r } { x _ { k + 1 } = \frac { 1 } { k + 2 } x _ { 0 } + \frac { k + 1 } { k + 2 } \mathbb { T } _ { \mathcal { B } _ { k } } ( x _ { k } ) } \end{array}$ , maintains the deterministic iteration bound $N = \mathcal { O } ( L \| x _ { 0 } - x _ { \star } \| / \epsilon )$ $\begin{array} { r } { \mathrm { i f ~ } \mathbb { E } \left\| \mathbb { F } _ { \mathcal { B } _ { k } } ( x _ { k } ) - \mathbb { F } ( x _ { k } ) \right\| ^ { 2 } \lesssim \frac { \epsilon ^ { 2 } } { k } . } \end{array}$ Simple mini-batching with $| B _ { k } | = \Theta ( k \epsilon ^ { - 2 } )$ yields $\mathcal { O } ( \epsilon ^ { - 4 } )$ stochastic oracle complexity, and using PAGE variance reduction, this can be reduced to $\mathcal { O } ( \epsilon ^ { - 3 } )$

## 4 Stochastic Dual-Anchor Acceleration

In this section, we analyze Stochastic Dual-OHM (S-Dual-OHM): with $\mathbb { T } _ { B _ { - 1 } } ( x _ { - 1 } ) = x _ { 0 }$

$$
x _ { k + 1 } = x _ { k } + \frac { N - k - 1 } { N - k } \left( \mathbb { T } _ { \mathcal { B } _ { k } } ( x _ { k } ) - \mathbb { T } _ { \mathcal { B } _ { k - 1 } } ( x _ { k - 1 } ) \right)\tag{S-Dual-OHM}
$$

for $k = 0 , \ldots , N - 2 .$ and present the following main result of this paper.

Theorem 4.1. Consider the root-finding problem (1) with a solution $x _ { \star } .$ satisfying Assumptions 3.1 and 3.2. Then for $\begin{array} { r } { 0 < \alpha \le \frac { 2 } { L } } \end{array}$ , S-Dual-OHM with $\mathbb { T } _ { \mathcal { B } _ { k } } = \mathbb { I } - \alpha \mathbb { F } _ { \mathcal { B } _ { k } }$ with $| B _ { k } | \equiv B$ satisfies

$$
\mathbb { E } \left[ \left. \mathbb { F } ( x _ { N - 1 } ) \right. \right] ^ { 2 } \leq \mathbb { E } \left[ \left. \mathbb { F } ( x _ { N - 1 } ) \right. ^ { 2 } \right] \leq \frac { 4 \left. x _ { 0 } - x _ { \star } \right. ^ { 2 } } { \alpha ^ { 2 } N ^ { 2 } } + \frac { 6 \sigma ^ { 2 } } { B } .
$$

This shows that we will have $\mathbb { E } \left[ \lVert \mathbb { F } ( x _ { N - 1 } ) \rVert \right] \leq \epsilon$ with $N = \Theta ( 1 / \epsilon )$ and $B = \Theta ( 1 / \epsilon ^ { 2 } )$ , so in total, $\mathcal { O } ( 1 / \epsilon ^ { 3 } )$ oracle complexity. This also proves an equivalent bound on the expected fixed-point residual $\begin{array} { r } { \mathbb { E } \left[ \left\| x _ { N - 1 } - \mathbb { T } ( x _ { N - 1 } ) \right\| \right] ^ { 2 } \leq \frac { 4 \left\| x _ { 0 } - x _ { \star } \right\| ^ { 2 } } { N ^ { 2 } } + \frac { 6 \alpha ^ { 2 } \sigma ^ { 2 } } { B } } \end{array}$ . While we defer technical details to Appendix $\mathrm { A } ,$ we provide the outline of key arguments below. Then, we show that with the additional assumption of strong monotonicity of $\mathbb { F } ,$ the total complexity can be reduced to $\widetilde { \mathcal { O } } ( 1 / \epsilon ^ { 2 } )$ with early stopping.

## 4.1 Proof outline for Theorem 4.1

We start with showing the identity characterizing the S-Dual-OHM algorithm.

Lemma 4.2. For S-Dual-OHM with $\mathbb { T } _ { B _ { k } } = \mathbb { I } - \alpha \mathbb { F } _ { B _ { k } }$ , the following identity holds:

$$
0 = \frac { ( N - 1 ) \alpha ^ { 2 } } { 4 } \left\| \mathbb { F } ( x _ { N - 1 } ) \right\| ^ { 2 } + \frac { \alpha } { 2 } \left. \mathbb { F } ( x _ { N - 1 } ) , x _ { N - 1 } - x _ { 0 } \right. + \sum _ { j = 1 } ^ { N - 1 } \lambda _ { N , j } \mathcal { Q } _ { N , j }
$$

where for $\begin{array} { r } { j = 1 , \dots , N - 1 , \lambda _ { N , j } = \frac { N \alpha } { 2 ( N - j ) ( N - j + 1 ) } } \end{array}$ and

$$
\mathcal { Q } _ { N , j } = \left. x _ { j - 1 } - x _ { N - 1 } , \mathbb { F } _ { B _ { j - 1 } } ( x _ { j - 1 } ) - \mathbb { F } ( x _ { N - 1 } ) \right. - \frac { \alpha } { 2 } \left. \mathbb { F } _ { B _ { j - 1 } } ( x _ { j - 1 } ) - \mathbb { F } ( x _ { N - 1 } ) \right. ^ { 2 } .
$$

Here, when the problem is deterministic so that $\mathbb { F } _ { B _ { k } } = \mathbb { F }$ for all $k = 0 , \ldots , N - 1$ and $\begin{array} { r } { 0 < \alpha \le \frac { 2 } { L } } \end{array}$ then by cocoercivity $\mathcal { Q } _ { N , j } \geq 0$ for all $j = 1 , \ldots , N - 1$ , so $\begin{array} { r } { \sum _ { j = 1 } ^ { N - 1 } \lambda _ { N , j } \mathcal { Q } _ { N , j } \ge 0 } \end{array}$ . The resulting inequality,

together with Young’s inequality and one more application of cocoercivity: $\langle \mathbb { F } ( x _ { N - 1 } ) , x _ { N - 1 } - x _ { \star } \rangle \geq$ $\begin{array} { r } { \frac { 1 } { L } \left\| \mathbb { F } ( x _ { N - 1 } ) \right\| ^ { 2 } \geq \frac { \alpha } { 2 } \left\| \mathbb { F } ( x _ { N - 1 } ) \right\| ^ { 2 } } \end{array}$ , yields

$$
0 \geq \frac { ( N - 1 ) \alpha ^ { 2 } } { 4 } \left. \mathbb { F } ( x _ { N - 1 } ) \right. ^ { 2 } + \frac { \alpha } { 2 } \left. \mathbb { F } ( x _ { N - 1 } ) , x _ { N - 1 } - x _ { 0 } \right.\tag{4}
$$

$$
\geq { \frac { N \alpha ^ { 2 } } { 4 } } \left. \mathbb { F } ( x _ { N - 1 } ) \right. ^ { 2 } - { \frac { N \alpha ^ { 2 } } { 8 } } \left. \mathbb { F } ( x _ { N - 1 } ) \right. ^ { 2 } - { \frac { 1 } { 2 N } } \left. x _ { \star } - x _ { 0 } \right. ^ { 2 }\tag{5}
$$

which implies $\begin{array} { r } { \left\| \mathbb { F } ( x _ { N - 1 } ) \right\| ^ { 2 } \leq \frac { 4 \left\| x _ { 0 } - x _ { \star } \right\| ^ { 2 } } { \alpha ^ { 2 } N ^ { 2 } } } \end{array}$

On the other hand, in the stochastic setting, we do not have $\mathcal { Q } _ { N , j } ~ \ge ~ 0$ due to noise, and these quantities are not local as in typical stochastic analyses, making the convergence proof seemingly much harder. However, we can isolate the dependence on stochastic errors at each iteration and control them through the following two lemmas. These key results allow us to follow the reasoning similar to the deterministic case above.

Lemma 4.3. Under the setting of Theorem 4.1, let $e _ { k } = \mathbb { F } _ { B _ { k } } ( x _ { k } ) - \mathbb { F } ( x _ { k } )$ for $k = 0 , \ldots , N - 2$ . Then

$$
0 \geq \mathbb { E } \left[ \frac { ( N - 1 ) \alpha ^ { 2 } } { 4 } \left\| \mathbf { F } ( x _ { N - 1 } ) \right\| ^ { 2 } + \frac { \alpha } { 2 } \left. \mathbf { F } ( x _ { N - 1 } ) , x _ { N - 1 } - x _ { 0 } \right. \right] - \sum _ { i = 1 } ^ { N - 1 } \lambda _ { N , j } \left( \frac { \alpha \sigma ^ { 2 } } { 2 B } + \mathbb { E } \left[ \langle e _ { j - 1 } , \mathbb { T } x _ { N - 1 } \rangle \right] \right) .\tag{6}
$$

## Lemma 4.4. Under the setting of Theorem $\begin{array} { r } { 4 . 1 , \mathbb { E } \left[ \left. e _ { j - 1 } , \mathbb { T } x _ { N - 1 } \right. \right] \leq \frac { \alpha \sigma ^ { 2 } } { B } \mathrm { ~ f o r ~ } j = 1 , \dots , N - 1 . } \end{array}$

Lemma 4.4 states that the amount by which the error $e _ { j - 1 }$ propagates to the last iterate and afects the analysis is uniformly bounded by a constant independent of N. Combining Lemmas 4.3 and 4.4, and using $\begin{array} { r } { \sum _ { j = 1 } ^ { N - 1 } \lambda _ { N , j } = \frac { \alpha ( N - 1 ) } { 2 } } \end{array}$ , we can proceed as in (4) to obtain

$$
\begin{array} { l } { 0 \geq \mathbb { E } \left[ \displaystyle \frac { N \alpha ^ { 2 } } { 4 } \left\| \mathbb { F } ( x _ { N - 1 } ) \right\| ^ { 2 } + \displaystyle \frac { \alpha } { 2 } \left. \mathbb { F } ( x _ { N - 1 } ) , x _ { \star } - x _ { 0 } \right. \right] - \displaystyle \frac { 3 ( N - 1 ) \alpha ^ { 2 } \sigma ^ { 2 } } { 4 B } } \\ { \geq \mathbb { E } \left[ \displaystyle \frac { N \alpha ^ { 2 } } { 4 } \left\| \mathbb { F } ( x _ { N - 1 } ) \right\| ^ { 2 } - \displaystyle \frac { N \alpha ^ { 2 } } { 8 } \left\| \mathbb { F } ( x _ { N - 1 } ) \right\| ^ { 2 } - \displaystyle \frac { 1 } { 2 N } \left\| x _ { 0 } - x _ { \star } \right\| ^ { 2 } \right] - \displaystyle \frac { 3 ( N - 1 ) \alpha ^ { 2 } \sigma ^ { 2 } } { 4 B } . } \end{array}
$$

Rearranging, we have the desired bound of Theorem 4.1.

## 4.1.1 Intuition for why S-Dual-OHM is stable under noise

The previous analysis shows that the net accumulation of error in S-Dual-OHM is proportional to $\begin{array} { r } { \sum _ { j = 1 } ^ { N - 1 } \lambda _ { N , j } = \frac { \alpha ( N - 1 ) } { 2 } = \mathcal { O } ( N ) } \end{array}$ . On the other hand, stochastic OHM satisfies the similar identity

$$
0 = \frac { ( N - 1 ) \alpha ^ { 2 } } { 4 } \left\| \mathbf { F } ( x _ { N - 1 } ) \right\| ^ { 2 } + \frac { \alpha } { 2 } \left. \mathbf { F } ( x _ { N - 1 } ) , x _ { N - 1 } - x _ { 0 } \right. + \sum _ { j = 1 } ^ { N - 1 } \nu _ { j + 1 , j } \mathcal { Q } _ { j + 1 , j }
$$

where $\begin{array} { r } { \mathcal { Q } _ { j + 1 , j } = \left. x _ { j - 1 } - x _ { j } , \mathbb { F } _ { \mathcal { B } _ { j - 1 } } ( x _ { j - 1 } ) - \mathbb { F } _ { \mathcal { B } _ { j } } ( x _ { j } ) \right. - \frac { \alpha } { 2 } \left\| \mathbb { F } _ { \mathcal { B } _ { j - 1 } } ( x _ { j - 1 } ) - \mathbb { F } _ { \mathcal { B } _ { j } } ( x _ { j } ) \right\| ^ { 2 } } \end{array}$ but with $\nu _ { j + 1 , j } =$ $\frac { \alpha j ( j + 1 ) } { 2 N }$ . Here each $\mathcal { Q } _ { j + 1 , j }$ is not nonnegative, but contributes a negative lower bound proportional to $\sigma ^ { 2 }$ in expectation, which then gets multiplied with $\begin{array} { r } { \sum _ { j = 1 } ^ { N - 1 } \nu _ { j + 1 , j } = \Theta ( N ^ { 2 } ) } \end{array}$ and the resulting error term is no longer controllable as in Theorem 4.1. This is a central diference that makes the dual-anchor mechanism more amenable to stochastic extension than anchoring. A concurrent work [52] has also provided a related theory that the sum of weights associated with inequalities used in the convergence proof measures a fixed-point algorithm’s robustness to oracle noise, albeit in deterministic setting.

A complementary interpretation is that OHM is an anytime optimal algorithm, achieving the rate $\begin{array} { r } { \left\| \mathbb { F } ( x _ { k - 1 } ) \right\| ^ { 2 } \leq \frac { 4 \| x _ { 0 } - x _ { \star } \| ^ { 2 } } { \alpha ^ { 2 } k ^ { 2 } } } \end{array}$ for all $k = 1 , 2 , \ldots$ . This means that OHM must maintain the optimal guarantee at every iteration, forcing it to exploit the geometric property of $\mathbb { F } ,$ i.e. cocoercivity, throughout the trajectory. Its analysis is therefore more tightly tuned to the exact problem class and is more easily disrupted by oracle noise. Dual-OHM, by contrast, is optimized only for the prescribed last iterate and does not sufer from the same accumulation of errors.

## 4.2 Faster convergence under strong monotonicity

In this section, we show that if � is additionally strongly monotone, then S-Dual-OHM can achieve the desired accuracy $\mathbb { E } \left[ \lVert \mathbb { F } ( x _ { k } ) \rVert \right] \leq \epsilon$ much earlier than the prescribed horizon N. Suppose that � is $\mu { \cdot }$ -strongly monotone and $1 / L \cdot$ -cocoercive. Then, with $0 < \bar { \alpha } < \frac { 2 } { L } , \mathbb { T } = \mathbb { I } - \alpha \mathbb { F }$ is $\gamma \mathrm { - }$ -contractive $( { \mathrm { i . e . } }$ γ-Lipschitz) with $\gamma = \sqrt { 1 - \alpha \mu ( 2 - \alpha L ) }$ because

$$
\begin{array} { r l } & { \left\| \mathbb { T } ( x ) - \mathbb { T } ( y ) \right\| ^ { 2 } = \left\| x - y \right\| ^ { 2 } - 2 \alpha \left. x - y , \mathbb { F } ( x ) - \mathbb { F } ( y ) \right. + \alpha ^ { 2 } \left\| \mathbb { F } ( x ) - \mathbb { F } ( y ) \right\| ^ { 2 } } \\ & { \qquad \leq \left\| x - y \right\| ^ { 2 } - \alpha ( 2 - \alpha L ) \left. x - y , \mathbb { F } ( x ) - \mathbb { F } ( y ) \right. \leq \left( 1 - \alpha \mu ( 2 - \alpha L ) \right) \left\| x - y \right\| ^ { 2 } . } \end{array}
$$

Theorem 4.5. Consider the root-finding problem (1) with $^ { 1 / \underline { { L } } . }$ -cocoercive and µ-strongly monotone � and a solution $x _ { \star }$ , satisfying Assumption 3.1. Let $\textstyle 0 < \alpha < { \frac { 2 } { L } }$ and $\gamma = \sqrt { 1 - \alpha \mu ( 2 - \alpha L ) } \in [ 0 , 1 )$ Let $y _ { k }$ be the iterates from deterministic Dual-OHM using true evaluations of �, with $x _ { 0 } = y _ { 0 }$ . Then, for $k = 0 , \ldots , N - 1$ , S-Dual-OHM with $\mathbb { T } _ { B _ { k } } = \mathbb { I } - \alpha \mathbb { F } _ { B _ { k } }$ satisfies

$$
\mathbb { E } \left[ \left. x _ { k } - y _ { N - 1 } \right. ^ { 2 } \right] \leq 2 \left( \frac { 1 + \gamma } { 1 - \gamma } \right) ^ { 2 } \gamma ^ { 2 k } \left. x _ { 0 } - x _ { \star } \right. ^ { 2 } + \frac { 2 \alpha ^ { 2 } \sigma ^ { 2 } } { B ( 1 - \gamma ^ { 2 } ) } .\tag{7}
$$

In particular, when $\begin{array} { r } { \alpha = \frac { 1 } { L } } \end{array}$ , we have

$$
\mathbb { E } \left[ \left. \mathbb { F } ( x _ { k } ) \right. ^ { 2 } \right] \leq \frac { 8 L ^ { 2 } \left. x _ { 0 } - x _ { \star } \right. ^ { 2 } } { N ^ { 2 } } + \frac { 6 4 L ^ { 4 } } { \mu ^ { 2 } } e ^ { - k \frac { \mu } { L } } \left. x _ { 0 } - x _ { \star } \right. ^ { 2 } + \frac { L } { \mu } \frac { 4 \sigma ^ { 2 } } { B } .\tag{8}
$$

Notably, Theorem 4.5 uses only Assumption 3.1 and cocoercivity of $\mathbb { F } ,$ but not Assumption 3.2.

The reason why this result holds is two-fold. First, when � is contractive, S-Dual-OHM does not deviate far from deterministic Dual-OHM (Lemma 4.6). Second, deterministic Dual-OHM achieves the designated terminal accuracy quickly as it linearly converges to its final iterate $y _ { N - 1 }$ (Lemma 4.7). Because Dual-OHM has a small residual norm guarantee at $y _ { N - 1 }$ , this implies that $x _ { k }$ also attains the comparably small residual quickly.

Lemma 4.6. Under the conditions of Theorem 4.5, E $\begin{array} { r } { \left[ \left\| x _ { k } - y _ { k } \right\| ^ { 2 } \right] \leq \frac { \alpha ^ { 2 } \sigma ^ { 2 } } { B ( 1 - \gamma ^ { 2 } ) } } \end{array}$ for $k = 0 , \ldots , N - 1$

Lemma 4.7. Under the conditions of Lemma 4.6, we have $\begin{array} { r } { \| y _ { k } - y _ { N - 1 } \| \le \left( \frac { 1 + \gamma } { 1 - \gamma } \right) \gamma ^ { k } \left\| x _ { 0 } - x _ { \star } \right\| } \end{array}$ for $k = 0 , \ldots , N - 1$ and $x _ { \star } \in \operatorname { F i x } \mathbb { T }$

Proof of Theorem 4.5. Note that $\mathbb { E } \left[ \left. x _ { k } - y _ { N - 1 } \right. ^ { 2 } \right] \leq 2 \mathbb { E } \left[ \left. x _ { k } - y _ { k } \right. ^ { 2 } \right] + 2 \mathbb { E } \left[ \left. y _ { k } - y _ { N - 1 } \right. ^ { 2 } \right]$ . Then, by Lemmas 4.6 and $\begin{array} { r } { 4 . 7 , \mathbb { E } \left[ \left. x _ { k } - y _ { N - 1 } \right. ^ { 2 } \right] \leq \frac { 2 \alpha ^ { 2 } \sigma ^ { 2 } } { B ( 1 - \gamma ^ { 2 } ) } + 2 \left( \frac { 1 + \gamma } { 1 - \gamma } \right) ^ { 2 } \gamma ^ { 2 k } \left. x _ { 0 } - x _ { \star } \right. ^ { 2 } } \end{array}$ , which proves (7). With $\begin{array} { r } { \alpha = \frac { 1 } { L } } \end{array}$ , we have $\textstyle 1 - \gamma ^ { 2 } { \overset { \cdot } { = } } { \frac { \mu } { L } }$ , so combining (7) with Proposition 3.3 we obtain

$$
\begin{array} { r l r } {  { \mathbb { E } [ \| \mathbb { F } ( x _ { k } ) \| ^ { 2 } ] \leq 2 \mathbb { E } [ \| \mathbb { F } ( y _ { N - 1 } ) \| ^ { 2 } ] + 2 \mathbb { E } [ \| \mathbb { F } ( x _ { k } ) - \mathbb { F } ( y _ { N - 1 } ) \| ^ { 2 } ] } } \\ & { } & { \leq 2 \mathbb { E } [ \| \mathbb { F } ( y _ { N - 1 } ) \| ^ { 2 } ] + 2 L ^ { 2 } \mathbb { E } [ \| x _ { k } - y _ { N - 1 } \| ^ { 2 } ] } \\ & { } & { \leq \frac { 8 L ^ { 2 } \| x _ { 0 } - x _ { \star } \| ^ { 2 } } { N ^ { 2 } } + 2 L ^ { 2 } [ \frac { 2 \sigma ^ { 2 } } { \mu L B } + 2 ( \frac { 1 + \gamma } { 1 - \gamma } ) ^ { 2 } \gamma ^ { 2 k } \| x _ { 0 } - x _ { \star } \| ^ { 2 } ] . } \end{array}
$$

where the second inequality uses L-Lipschitzness of �. Rearranging and using $\gamma < 1 , ( 1 - \gamma ) ^ { 2 } \geq$ $\begin{array} { r } { \frac { ( 1 - \gamma ) ^ { 2 } ( 1 + \gamma ) ^ { 2 } } { 4 } = \frac { \mu ^ { 2 } } { 4 L ^ { 2 } } } \end{array}$ and $\begin{array} { r } { \gamma ^ { 2 k } = \left( 1 - \frac { \mu } { L } \right) ^ { k } \leq e ^ { - k \frac { \mu } { L } } } \end{array}$ , we obtain (8). □

Let $\| x _ { 0 } - x _ { \star } \| = D$ . The bound (8) shows that if one chooses, e.g., $\begin{array} { r } { N \ge \frac { 4 L D } { \epsilon } , k \ge \frac { L } { \mu } \log \frac { 2 5 6 L ^ { 4 } D ^ { 2 } } { \mu ^ { 2 } \epsilon ^ { 2 } } } \end{array}$ and $\begin{array} { r } { B \ge \frac { 1 6 L \sigma ^ { 2 } } { \mu \epsilon ^ { 2 } } } \end{array}$ , then $\mathbb { E } \left[ \| \mathbb { F } ( x _ { k } ) \| \right] ^ { 2 } \le \mathbb { E } \left[ \| \mathbb { F } ( x _ { k } ) \| ^ { 2 } \right] \le \epsilon ^ { 2 } ,$ . If we run S-Dual-OHM with these choices of N and $B ,$ we will achieve the accuracy ϵ in $\begin{array} { r } { \mathcal { O } \left( \frac { L } { \mu } \log \epsilon ^ { - 1 } \right) } \end{array}$ iterations without the need to run all N iterations. With this early stopping, S-Dual-OHM has improved $\mathcal { O } \left( \textstyle \frac { L ^ { 2 } \sigma ^ { 2 } } { \mu ^ { 2 } \epsilon ^ { 2 } } \log \frac { 1 } { \epsilon } \right)$ oracle complexity for strongly monotone $\mathbb { F } ,$ which has near-optimal dependence on ϵ [12]. The dependence on condition number, however, is still suboptimal, and improving it based on the Dual-OHM mechanism remains an open question.

## 5 Numerical Experiments

We provide numerical evaluations to demonstrate that the dual-anchor mechanism is indeed preferable to anchoring (Halpern algorithms) under stochasticity, and compare the empirical performance of S-Dual-OHM with prior algorithms for stochastic monotone inclusions/fixed-point problems. We consider three distinct problems: the worst-case nonexpansive fixed-point operator designed by [40], a finite-sum cocoercive operator using uniform random vectors from the unit sphere, and a strongly-convex–stronglyconcave minimax problem with Huber-type regularizers following [53, 12]. All experiments were run on a MacBook Air with Apple M3 Chip and 24GB Memory.

Baseline algorithms. We consider the simple algorithm $x _ { k + 1 } = x _ { k } - \alpha \mathbb { F } _ { B _ { k } } ( x _ { k } )$ , which we call Stochastic Gradient Descent-Ascent (SGDA) following the nomenclature of the minimax optimization literature. We consider the Stochastic Extragradient (SEG) with the update rule

$$
x _ { k + 1 / 2 } = x _ { k } - \alpha \mathbb { F } _ { \mathcal { B } _ { k } } ( x _ { k } ) , \quad x _ { k + 1 } = x _ { k } - \alpha \mathbb { F } _ { \mathcal { B } _ { k + 1 / 2 } } ( x _ { k + 1 / 2 } ) .
$$

Next, we consider the naive stochastic extension of OHM: $\begin{array} { r } { x _ { k + 1 } = \frac { 1 } { k + 2 } x _ { 0 } + \frac { k + 1 } { k + 2 } \left( x _ { k } - \alpha \mathbb { F } _ { \mathcal { B } _ { k } } ( x _ { k } ) \right) } \end{array}$ , using the same constant mini-batch size as S-Dual-OHM. We refer to this algorithm as S-OHM (with constant B). This is included to isolate the efect of using the dual-anchor update rather than the usual anchor acceleration and directly contrast the two mechanisms.

We also include the algorithms carefully designed for reducing the expected last-iterate residual. First, the Halpern-PAGE algorithm from [9] uses the S-OHM update but replaces $\mathbb { F } _ { B _ { k } } ( x _ { k } )$ by a PAGE-type estimator which uses a larger batch size $s _ { 1 }$ with probability $\begin{array} { r } { p _ { k } = \operatorname* { m i n } \left\{ 1 , \frac { 2 } { k + 2 } \right\} } \end{array}$ and otherwise uses a sample operator diference with batch size $s _ { 2 }$ . Second, we use the single-loop simplification of RAIN algorithm from [12] used in their experiments, which can be written as

$$
z _ { k + 1 / 2 } = z _ { k } - \eta \bigl ( \widehat { \mathbf { F } } ( z _ { k } ; \xi _ { k } ) + r _ { k } ( z _ { k } ) \bigr ) , \qquad z _ { k + 1 } = z _ { k } - \eta \bigl ( \widehat { \mathbf { F } } ( z _ { k + 1 / 2 } ; \xi _ { k + 1 / 2 } ) + r _ { k } ( z _ { k + 1 / 2 } ) \bigr )
$$

where $\begin{array} { r } { r _ { k } ( x ) = \lambda \gamma ( S _ { k } x - m _ { k } ) , S _ { k } = \sum _ { t < k } ( 1 + \gamma ) ^ { t } , m _ { k } = \sum _ { t < k } ( 1 + \gamma ) ^ { t } z _ { t } } \end{array}$ , and $\eta , \lambda , \gamma > 0$ are hyperparameters. Finally, Halpern-VR-Finite from [10] is similar to Halpern-PAGE but applicable only to finite sum problems of the form $\begin{array} { r } { \mathbb { F } = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \overline { { \mathbb { F } \big ( \cdot ; \xi _ { i } \big ) } } } \end{array}$ and uses either full-batch (true) operator � or batches of size $\lceil { \sqrt { n } } \rceil$

We tune all baseline algorithms and S-Dual-OHM under the same total sample budget $Q = N B$ via grid search, where computing $\mathbb { F } _ { B _ { k } }$ with $| B _ { k } | = B$ counts as B operator evaluations. The only hyperparameter B for S-Dual-OHM is tuned over $B \in \{ 1 , 1 0 , 2 0 , 5 0 , 1 0 0 \}$ , and S-OHM (Constant $B )$ is then run with the same selected $B . ^ { 1 }$ The Halpern-PAGE parameters are tuned over $s _ { 1 } \in \{ 1 , 1 0 , 2 0 , 5 0 , 1 0 0 \}$ and $s _ { 2 } ~ \in ~ \{ 1 , 5 , 1 0 , 2 0 \}$ , and RAIN hyperparameters are tuned over $\eta ~ \in ~ \{ 0 . 0 0 5 , 0 . 0 1 , 0 . 0 5 , 0 . 1 , 1 \}$ $\lambda \in \{ 0 . 0 0 1 , 0 . 0 1 , 0 . 1 , 1 \}$ , and $\gamma \in \lbrace 0 . 0 0 1 , 0 . 0 1 , 0 . 1 , 1 \rbrace$ . SGDA and SEG step-sizes were tuned over $\alpha \in \{ 0 . 0 0 5 , 0 . 0 1 , 0 . 0 5 , 0 . 1 , 1 \}$ , the same step-size grid as RAIN. We run each experiment with 10 independent random seeds and report the average residual norm $\| \mathbb { F } ( x _ { k } ) \|$ together with the shaded region indicating the empirical 5th–95th percentile band.

![](images/f184b315664ec59aba06fe0c1171e59623f6fc5288150fd8b24ff55e6300ab08.jpg)

![](images/c32fc8002d48cec1ed43f2a09876d6c3541fa15eafc0053cef60fb363bf9fc64.jpg)  
Figure 1: Plot of residual norm versus stochastic samples. (Left) Worst-case nonexpansive operator construction (Experiment 1). (Right) Finite-sum random cocoercive operator construction (Experiment 2). Solid curves indicate means over 10 independent runs, and shaded bands denote empirical 5th–95th percentiles.

Experiment 1: Worst-case nonexpansive operator. We use the afine lower-bound construction of [40]: define ℍ : $\begin{array} { r } { \mathbb { R } ^ { d } \to \mathbb { R } ^ { d } \mathrm { ~ b y ~ } \mathbb { H } ( x _ { 1 } , \dots , x _ { d } ) = \left( x _ { d } - \frac { 2 } { \sqrt { d } } , - x _ { 1 } , - x _ { 2 } , \dots , - x _ { d - 1 } \right) } \end{array}$ , and after choosing a shift vector $s \in \mathbb { R } ^ { d }$ , let $\mathbb { F } ( x ) = \mathbb { H } ( x - s ) + ( x - s )$ and $\mathbb { T } = \mathbb { I } - \mathbb { F }$ . Then � is $\frac { 1 } { 2 }$ -cocoercive and � is nonexpansive. We use Gaussian noise oracle ${ \widehat { \mathbb { F } } } ( x ; \xi ) = \mathbb { F } ( x ) + \xi$ with $\textstyle \xi \sim { \mathcal { N } } \left( 0 , { \frac { \sigma ^ { 2 } } { d } } I _ { d } \right)$ , and use a budget of 2000 samples. We set $d = 2 0 0 1 , x _ { 0 } = 0 , \alpha = 1 , \sigma = 0 . 1$ and $s \sim \mathcal { N } ( 0 , I _ { d } )$

The left panel of Figure 1 illustrates that S-Dual-OHM is not an anytime algorithm, so its intermediate residuals can be worse than those of the other algorithms; however, in the end, it achieves the smallest last-iterate residual among all algorithms compared. This is in sharp contrast with $\mathrm { S } -$ OHM with constant B, which fails to converge and exhibits diverging residual. This demonstrates that Halpern-type algorithms can indeed be prone to accumulation of stochastic errors. However, with variance reduction (Halpern-PAGE) or recursively adjusted anchor (RAIN), the residual convergence gets stabilized. Finally, SEG and SGDA converge steadily although they were not specifically designed for residual minimization.

Experiment 2: Finite-sum cocoercive operator. We consider a finite-sum problem

$$
\mathbb { F } ( x ) = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \mathbb { F } _ { i } ( x ) , \qquad \mathbb { F } _ { i } ( x ) = \binom { q _ { i } ( q _ { i } ^ { \top } x _ { 1 : r } ) } { 0 } \in \mathbb { R } ^ { d } ,
$$

where each $q _ { i } \in \mathbb { R } ^ { r }$ is sampled uniformly from the unit sphere. Here each $\mathbb { F } _ { i }$ is 1-cocoercive and linear, and therefore � is cocoercive with $L \leq 1$ but not strongly monotone when $r < d$ since the common null space of $\mathbb { F } _ { i }$ is nontrivial. A stochastic oracle call samples component indices and returns the average of the queried $\mathbb { F } _ { i } { \mathrm { : s } } _ { }$ . The operator variance is bounded over the region that trajectories of all algorithms stay within during all experiment runs. We use a budget of 2000 stochastic samples, and set $n = 2 0 0$ $d = 2 0 0 , r = 1 9 9$ and $\mathbb { T } = \mathbb { I } - \alpha \mathbb { F }$ with $\alpha = 1$ We take $x _ { 0 } \sim 1 0 \cdot \mathcal { N } ( 0 , I _ { d } )$ This is a generic, non-worst-case problem that basic methods like SGDA can solve efectively. Still, S-Dual-OHM is also fairly competent, and importantly, outperforms its primal counterpart S-OHM. Notably, Halpern-VR-Finite makes very small progress because the total sample budget 2000 is restrictive with our choice $n = 2 0 0$ , even though it has the best asymptotic dependence on ϵ in theory.

Experiment 3: SCSC minimax problem with Huber-type regularizers. We consider

$$
\Phi ( x , y ) = ( 1 - \delta ) \sum _ { i } g _ { \nu } ( x _ { i } ) + \delta x ^ { \top } y - ( 1 - \delta ) \sum _ { i } g _ { \nu } ( y _ { i } ) + \frac { \mu } { 2 } \left. x \right. ^ { 2 } - \frac { \mu } { 2 } \left. y \right. ^ { 2 } ,
$$

![](images/dac222cfe24ca5de2ddff30a48d37e53fa22cc218019f79f9c15ccb1327aa8e4.jpg)

![](images/ba77b3d70c16ba8ab0e588524cd44fe2ba92aa579adb05242b2981fd44c9a0e8.jpg)  
Figure 2: Plot of residual norm versus stochastic samples for SCSC Huber-type minimax problem (Experiment 3). (Left) Low-variance regime $\sigma = 0 . 0 5$ . (Right) High-variance regime $\sigma = 1 . 5$ . Solid curves are means over 10 independent runs, and shaded bands indicate empirical 5th–95th percentiles.

for $z ~ = ~ ( x , y ) ~ \in ~ \mathbb { R } ^ { d } \times \mathbb { R } ^ { d }$ , where $\begin{array} { r } { g _ { \nu } ( u ) ~ = ~ \left\{ \begin{array} { l l } { \frac 1 2 u ^ { 2 } , } & { | u | \leq \nu } \\ { \nu | u | - \frac 1 2 \nu ^ { 2 } , } & { | u | > \nu } \end{array} \right. . } \end{array}$ The saddle operator $\mathbb { F } ( z ) ~ =$ $( \nabla _ { x } \Phi ( x , y ) , - \nabla _ { y } \Phi ( x , y ) )$ is then given as $\mathbb { F } ( z ) ~ = ~ \left( { 1 - \delta } \right) \mathrm { c l i p } ( x , - \nu , \nu ) + \delta y + \mu x \mathrm { \bf ~ ) } + { \delta } z$ , which is $\mu -$ strongly monotone and M-Lipschitz with $M = 1 + \mu$ . We use the Gaussian noise oracle ${ \widehat { \mathbb { F } } } ( z ; \xi ) = \mathbb { F } ( z ) + \xi$ with $\begin{array} { r } { \xi \sim \mathcal { N } \left( 0 , \frac { \sigma ^ { 2 } } { 2 d } I _ { 2 d } \right) } \end{array}$ , and use a budget of 10000 stochastic samples. We set $d = 5 0 , \delta = 1 0 ^ { - 2 }$ $\textstyle \nu = 5 \times 1 0 ^ { - 5 } , \mu = 0 . 1 , L = { \frac { M ^ { 2 } } { \mu } }$ and $z _ { 0 } = ( x _ { 0 } , y _ { 0 } )$ is a uniform random unit vector. We use $\begin{array} { r } { \alpha = \frac { 1 } { L } } \end{array}$ for S-Dual-OHM, S-OHM and Halpern-PAGE.

We simulate both low noise $( \sigma = 0 . 0 5 )$ and high noise $( \sigma = 1 . 5 )$ regimes. When $\sigma = 0 . 0 5 ,$ , we observe rapid progression in earlier iterations of S-Dual-OHM, but then it stagnates (Figure 2). This is because the noise term is small, so the best batch size choice is $B ^ { * } = 1$ , i.e., mini-batching does not benefit the final convergence. In this regime, RAIN, SGDA, SEG converges more steadily than S-Dual-OHM, although we still observe that S-OHM is inferior to S-Dual-OHM. With larger noise with $\sigma = 1 . 5$ , we observe that mini-batching becomes efective and $B ^ { * } = 1 0$ is selected; in this case, dual-anchor with mini-batching exhibits comparable performance to classical methods and deliberately designed near-optimal algorithm RAIN.

## 6 Conclusion and Open Questions

We demonstrate that dual-anchor acceleration from deterministic root-finding or fixed-point problems admits a direct stochastic extension, unlike anchor-type accelerated algorithms. This is based on the observation that distinct deterministic accelerations are intrinsically diferent in their behavior under stochasticity. We believe that our work enables interesting future work on acceleration and optimal complexities for stochastic optimization.

Two immediate limitations of our work, which thus remain as open questions, are: (1) whether the dual-anchor mechanism extends to root-finding problems with monotone and Lipschitz operators � and (2) whether we can further improve the complexity to the optimal level [12] $\mathcal { O } ( \epsilon ^ { - 2 } \log \epsilon ^ { - 1 } )$ via dual-anchor acceleration. As the dual-anchor acceleration also exists in the monotone Lipschitz inclusion setting [55], we believe that the extension would be possible with novel insights not relying on nonexpansivity of $\mathbb { I } - \alpha \mathbb { F } .$ . We also expect our result to serve as a baseline that can be combined with broader techniques in stochastic optimization for potentially achieving the optimal complexity.

## Acknowledgments

TaeHo Yoon’s contribution to this work was supported by NSF CCF 2504626. Nicolas Loizou’s contribution to this work was supported by NSF CCF 2504626 and NSF CAREER 2542902.

## References

[1] James K Alcala, Yat Tin Chow, and Mahesh Sunkula. Moving anchor extragradient methods for smooth structured minimax problems. arXiv:2308.12359, 2023.

[2] Zeyuan Allen-Zhu. Katyusha: The first direct acceleration of stochastic gradient methods. Journal of Machine Learning Research, 18(221):1–51, 2018.

[3] Heinz H. Bauschke and Patrick L. Combettes. Convex Analysis and Monotone Operator Theory in Hilbert Spaces. Springer International Publishing, 2nd edition, 2017.

[4] Aleksandr Beznosikov, Eduard Gorbunov, Hugo Berard, and Nicolas Loizou. Stochastic gradient descent-ascent: Unified theory and new eficient methods. International Conference on Artificial Intelligence and Statistics, 2023.

[5] Radu I. Bot¸ and Enis Chenchene. Extragradient method with flexible anchoring: Strong convergence and fast residual decay. SIAM Journal on Optimization, 36(3):1420–1445, 2026.

[6] Radu Ioan Bot¸, Ern¨o Robert Csetnek, and Dang-Khoa Nguyen. Fast Optimistic Gradient Descent Ascent (OGDA) method in continuous and discrete time. Foundations of Computational Mathematics, 2023.

[7] Mario Bravo and Roberto Cominetti. Stochastic fixed-point iterations for nonexpansive maps: Convergence and error bounds. SIAM Journal on Control and Optimization, 62(1):191–219, 2024.

[8] Mario Bravo and Juan Pablo Contreras. Stochastic Halpern iteration in normed spaces and applications to reinforcement learning. Mathematical Programming, 2026.

[9] Xufeng Cai, Chaobing Song, Crist´obal A Guzm´an, and Jelena Diakonikolas. Stochastic halpern iteration with variance reduction for stochastic monotone inclusions. Neural Information Processing Systems, 2022.

[10] Xufeng Cai, Ahmet Alacaoglu, and Jelena Diakonikolas. Variance reduced halpern iteration for finite-sum monotone inclusions. International Conference on Learning Representations, 2024.

[11] Yang Cai and Weiqiang Zheng. Accelerated single-call methods for constrained min-max optimization. International Conference on Learning Representations, 2023.

[12] Lesi Chen and Luo Luo. Near-optimal algorithms for making the gradient small in stochastic minimax optimization. Journal of Machine Learning Research, 25(387):1–44, 2024.

[13] Constantinos Daskalakis, Andrew Ilyas, Vasilis Syrgkanis, and Haoyang Zeng. Training GANs with optimism. International Conference on Learning Representations, 2018.

[14] Damek Davis. Variance reduction for root-finding problems. Mathematical Programming, 197(1): 375–410, 2023.

[15] Olivier Devolder, Fran¸cois Glineur, and Yurii Nesterov. First-order methods of smooth convex optimization with inexact oracle. Mathematical Programming, 146(1):37–75, 2014.

[16] Jelena Diakonikolas. Halpern iteration for near-optimal and parameter-free monotone inclusion and strong solutions to variational inequalities. Conference on Learning Theory, 2020.

[17] Francisco Facchinei and Jong-Shi Pang. Finite-Dimensional Variational Inequalities and Complementarity Problems. Springer-Verlag, 2003.

[18] Saeed Ghadimi and Guanghui Lan. Accelerated gradient methods for nonconvex nonlinear and stochastic programming. Mathematical Programming, 156(1):59–99, 2016.

[19] Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. Neural Information Processing Systems, 2014.

[20] Eduard Gorbunov, Hugo Berard, Gauthier Gidel, and Nicolas Loizou. Stochastic extragradient: General analysis and improved rates. International Conference on Artificial Intelligence and Statistics, 2022.

[21] Benjamin Halpern. Fixed points of nonexpanding maps. Bulletin of the American Mathematical Society, 73(6):957–961, 1967.

[22] Donghwan Kim. Accelerated proximal point method for maximally monotone operators. Mathematical Programming, 190(1–2):57–87, 2021.

[23] G. M. Korpelevich. The extragradient method for finding saddle points and other problems. Ekonomika i Matematicheskie Metody, 12(4):747–756, 1976.

[24] M. A. Krasnosel’skii. Two remarks on the method of successive approximations. Uspekhi Matematicheskikh Nauk, 10(1):123–127, 1955.

[25] Jongmin Lee and Ernest K. Ryu. Accelerating value iteration with anchoring. Neural Information Processing Systems, 2023.

[26] Jongmin Lee, Mario Bravo, and Roberto Cominetti. Near-optimal sample complexity for MDPs via anchoring. Proceedings of the 42nd international conference on machine learning, 2025.

[27] Sucheol Lee and Donghwan Kim. Fast extra gradient methods for smooth structured nonconvex– nonconcave minimax problems. Neural Information Processing Systems, 2021.

[28] Zhize Li, Hongyan Bao, Xiangliang Zhang, and Peter Richtarik. PAGE: A simple and optimal probabilistic gradient estimator for nonconvex optimization. International Conference on Machine Learning, 2021.

[29] Felix Lieder. On the convergence rate of the Halpern-iteration. Optimization Letters, 15(2):405– 418, 2021.

[30] Hongzhou Lin, Julien Mairal, and Zaid Harchaoui. A universal catalyst for first-order optimization. Neural Information Processing Systems, 2015.

[31] Tianyi Lin, Chi Jin, and Michael I. Jordan. Near-optimal algorithms for minimax optimization. Conference on Learning Theory, 125, 2020.

[32] Nicolas Loizou, Hugo Berard, Gauthier Gidel, Ioannis Mitliagkas, and Simon Lacoste-Julien. Stochastic gradient descent-ascent and consensus optimization for smooth games: Convergence analysis under expected co-coercivity. Neural Information Processing Systems, 2021.

[33] Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt, Dimitris Tsipras, and Adrian Vladu. Towards deep learning models resistant to adversarial attacks. International Conference on Learning Representations, 2018.

[34] William Robert Mann. Mean value methods in iteration. Proceedings of the American Mathematical Society, 4(3):506–510, 1953.

[35] Panayotis Mertikopoulos, Bruno Lecouat, Houssam Zenati, C.-S. Foo, Vijay Chandrasekhar, and Georgios Piliouras. Optimistic mirror descent in saddle-point problems: Going the extra (gradient) mile. International Conference on Learning Representations, 2019.

[36] George J. Minty. Monotone (nonlinear) operators in Hilbert space. Duke Mathematical Journal, 29(3):341–346, 1962.

[37] Martin Morin and Pontus Giselsson. Cocoercivity, smoothness and bias in variance-reduced stochastic gradient methods. Numerical Algorithms, 91(2):749–772, 2022.

[38] Arkadi Nemirovski. Prox-method with rate of convergence $O ( 1 / t )$ for variational inequalities with Lipschitz continuous monotone operators and smooth convex-concave saddle point problems. SIAM Journal on Optimization, 15(1):229–251, 2004.

[39] Lam M. Nguyen, Jie Liu, Katya Scheinberg, and Martin Tak´aˇc. SARAH: A novel method for machine learning problems using stochastic recursive gradient. International Conference on Machine Learning, 2017.

[40] Jisun Park and Ernest K Ryu. Exact optimal accelerated complexity for fixed-point iterations. International Conference on Machine Learning, 2022.

[41] Nicholas Pischke and Thomas Powell. Asymptotic Regularity of a Generalised Stochastic Halpern Scheme. Journal of Optimization Theory and Applications, 210(1):3, 2026.

[42] R. Tyrrell Rockafellar. Monotone operators and the proximal point algorithm. SIAM Journal on Control and Optimization, 14(5):877–898, 1976.

[43] Ernest K Ryu and Wotao Yin. Large-Scale Convex Optimization via Monotone Operators. Cambridge University Press, 2022.

[44] Ernest K. Ryu, Kun Yuan, and Wotao Yin. ODE analysis of stochastic gradient methods with optimism and anchoring for minimax problems and GANs. arXiv:1905.10899, 2019.

[45] Shoham Sabach and Shimrit Shtern. A first order method for solving convex bilevel optimization problems. SIAM Journal on Optimization, 27(2):640–660, 2017.

[46] Gesualdo Scutari, Francisco Facchinei, Jong-Shi Pang, and Daniel P. Palomar. Real and complex monotone communication games. IEEE Transactions on Information Theory, 60(7):4197–4231, 2014.

[47] Guido Stampacchia. Formes bilineaires coercitives sur les ensembles convexes. Comptes Rendus Hebdomadaires Des Seances De L Academie Des Sciences, 258(18):4413, 1964.

[48] Jaewook J. Suh, Jisun Park, and Ernest K. Ryu. Continuous-time analysis of anchor acceleration. Neural Information Processing Systems, 2023.

[49] Quoc Tran-Dinh. From Halpern’s fixed-point iterations to Nesterov’s accelerated interpretations for root-finding problems. Computational Optimization and Applications, 87(1):181–218, 2024.

[50] Quoc Tran-Dinh and Yang Luo. Halpern-type accelerated and splitting algorithms for monotone inclusions. arXiv:2110.08150, 2021.

[51] Rainer Wittmann. Approximation of fixed points of nonexpansive mappings. Archiv der Mathematik, 58(5):486–491, 1992.

[52] TaeHo Yoon and Benjamin Grimmer. A Theory of Composition and Duality of Extremal Optimal Fixed-Point Algorithms. arXiv:2605.02231, 2026.

[53] TaeHo Yoon and Ernest K. Ryu. Accelerated algorithms for smooth convex-concave minimax problems with $\mathcal { O } ( 1 / k ^ { 2 } )$ rate on squared gradient norm. International Conference on Machine Learning, 2021.

[54] TaeHo Yoon and Ernest K. Ryu. Accelerated minimax algorithms flock together. SIAM Journal on Optimization, 35(1):180–209, 2025.

[55] TaeHo Yoon, Jaeyeon Kim, Jaewook J. Suh, and Ernest K. Ryu. Optimal acceleration for minimax and fixed-point problems is not unique. International Conference on Machine Learning, 2024.

[56] TaeHo Yoon, Ernest K Ryu, and Benjamin Grimmer. H-invariance theory: A complete characterization of minimax optimal fixed-point algorithms. Accepted for publication in Mathematical Programming, 2025.

[57] TaeHo Yoon, Sayantan Choudhury, and Nicolas Loizou. Multiplayer federated learning: Reaching equilibrium with less communication. Advances in Neural Information Processing Systems, 2026.

[58] Siqi Zhang, Sayantan Choudhury, Sebastian Stich, and Nicolas Loizou. Communication-eficient gradient descent-accent methods for distributed variational inequalities: Unified analysis and local updates. International Conference on Learning Representations, 2024.

## Appendix

## Contents

1 Introduction 1   
2 Related Work 2   
3 Preliminaries and Assumptions 3   
3.1 Monotonicity, cocoercivity, and nonexpansiveness of operators . 3   
3.2 Stochastic oracle models . . . 4   
3.3 Known acceleration in fixed-point problems and its extension 4   
4 Stochastic Dual-Anchor Acceleration 5   
4.1 Proof outline for Theorem 4.1 . . . 5   
4.1.1 Intuition for why S-Dual-OHM is stable under noise 6   
4.2 Faster convergence under strong monotonicity . 7   
5 Numerical Experiments 8   
6 Conclusion and Open Questions 10   
A Missing Proofs 16   
A.1 Proof of Lemma 4.2 16   
A.2 Proof of Lemma 4.3 17   
A.3 Proof of Lemma 4.4 18   
A.4 Proof of Lemma 4.6 20   
A.5 Proof of Lemma 4.7 20   
B Additional Experiments 22

## A Missing Proofs

We denote by ${ \mathcal { F } } _ { 1 } \subseteq { \mathcal { F } } _ { 2 } \subseteq . .$ . the natural filtration generated by each iteration of the algorithm (selection of batches $B _ { i } ) , \mathrm { i . e . , } \mathcal { F } _ { k } = \sigma ( B _ { 0 } , B _ { 1 } , \dots , B _ { k - 1 } )$

## A.1 Proof of Lemma 4.2

Proof. The proof is purely algebraic, and does not use any property of � as an operator or stochastic property of the mini-batches $B _ { j }$ . The case $N = 1$ is immediate (we have the vacuous identity $0 = 0 )$ so assume $N \geq 2$

For $j = 1 , \ldots , N$ , define

$$
G _ { j } = \mathbb { F } _ { \mathcal { B } _ { j - 1 } } ( x _ { j - 1 } ) , \qquad g _ { j } = \frac { \alpha } 2 G _ { j } , \qquad u _ { j } = x _ { j - 1 } - g _ { j } .
$$

Here, $\boldsymbol { B } _ { N - 1 }$ is a ghost (auxiliary) batch introduced only for the analysis, and is not used by the algorithm update; in fact, one may ignore this ghost batch $( \mathrm { i . e . }$ , safely assume that $\boldsymbol { B } _ { N - 1 }$ is the full batch) and simply let $G _ { N } = \mathbb { F } ( x _ { N - 1 } )$ . With this notation, we have

$$
\begin{array} { r } { \mathbb { T } _ { \mathcal { B } _ { j - 1 } } ( x _ { j - 1 } ) = x _ { j - 1 } - \alpha G _ { j } = x _ { j - 1 } - 2 g _ { j } . } \end{array}
$$

Because $\mathbb { T } _ { B _ { - 1 } } ( x _ { - 1 } ) = x _ { 0 }$ , the $k = 0$ update of S-Dual-OHM is

$$
x _ { 1 } = x _ { 0 } + \frac { N - 1 } { N } \big ( \mathbb { T } _ { \mathcal { B } _ { 0 } } ( x _ { 0 } ) - x _ { 0 } \big ) .
$$

Let

$$
a _ { k } = \frac { N - k - 1 } { N - k } , \qquad z _ { 0 } = 0 , \qquad z _ { k + 1 } = \mathbb { T } _ { \mathcal { B } _ { k } } ( x _ { k } ) - x _ { k + 1 }
$$

for $k = 0 , \ldots , N - 2$ . Then, from the update rule,

$$
\begin{array} { l } { z _ { k + 1 } = \mathbb { T } _ { \mathcal { B } _ { k } } ( x _ { k } ) - x _ { k } - a _ { k } \bigl ( \mathbb { T } _ { \mathcal { B } _ { k } } ( x _ { k } ) - \mathbb { T } _ { \mathcal { B } _ { k - 1 } } ( x _ { k - 1 } ) \bigr ) } \\ { \quad \quad = ( 1 - a _ { k } ) \bigl ( x _ { k } - 2 g _ { k + 1 } \bigr ) - x _ { k } + a _ { k } \mathbb { T } _ { \mathcal { B } _ { k - 1 } } ( x _ { k - 1 } ) } \\ { \quad \quad = a _ { k } \left( \mathbb { T } _ { \mathcal { B } _ { k - 1 } } ( x _ { k - 1 } ) - x _ { k } \right) - 2 ( 1 - a _ { k } ) g _ { k + 1 } } \\ { \quad \quad = \displaystyle \frac { N - k - 1 } { N - k } z _ { k } - \frac { 2 } { N - k } g _ { k + 1 } . } \end{array}\tag{9}
$$

Then, by definition of $z _ { k + 1 }$ 2

$$
x _ { k + 1 } = \mathbb { T } _ { \mathcal { B } _ { k } } ( x _ { k } ) - z _ { k + 1 } = x _ { k } - 2 g _ { k + 1 } - z _ { k + 1 } .\tag{10}
$$

Now define, for $k = 0 , \ldots , N - 1$

$$
V _ { k } = - \frac { N - k - 1 } { N - k } \left\| z _ { k } + 2 g _ { N } \right\| ^ { 2 } + \frac { 2 } { N - k } \left. z _ { k } + 2 g _ { N } , x _ { k } - x _ { N - 1 } \right. .
$$

Note that $V _ { N - 1 } = 0$ . Next, we analyze $V _ { k } \mathrm { ~ - ~ } V _ { k + 1 }$ . To simplify the notations, fix $k \in \{ 0 , \ldots , N - 2 \}$ , and write

$$
\begin{array} { r l r l r l r l r } { m = N - k , } & { { } } & { w _ { k } = z _ { k } + 2 g _ { N } , } & { { } } & { d _ { k } = g _ { k + 1 } - g _ { N } , } & { { } } & { p _ { k } = x _ { k } - x _ { N - 1 } . } \end{array}
$$

From (9), we have

$$
w _ { k + 1 } - 2 g _ { N } = \frac { N - k - 1 } { N - k } ( w _ { k } - 2 g _ { N } ) - \frac { 2 } { N - k } g _ { k + 1 } \implies w _ { k + 1 } = \frac { m - 1 } { m } w _ { k } - \frac { 2 } { m } d _ { k }
$$

and (10) yields

$$
p _ { k + 1 } = x _ { k + 1 } - x _ { N - 1 } = ( x _ { k } - x _ { N - 1 } ) - ( z _ { k + 1 } + 2 g _ { N } ) - 2 ( g _ { k + 1 } - g _ { N } ) = p _ { k } - w _ { k + 1 } - 2 d _ { k } .
$$

Plugging the last identity into the expansion of $V _ { k } - V _ { k + 1 }$ below, we proceed as

$$
\begin{array} { r l } & { { \displaystyle { V _ { k } - V _ { k + 1 } = - \frac { m - 1 } { m } \left\| w _ { k } \right\| ^ { 2 } + \frac { 2 } { m } \left. w _ { k } , p _ { k } \right. + \frac { m - 2 } { m - 1 } \left\| w _ { k + 1 } \right\| ^ { 2 } - \frac { 2 } { m - 1 } \left. w _ { k + 1 } , p _ { k + 1 } \right. } } } \\ & { { \displaystyle \quad \quad = - \frac { m - 1 } { m } \left\| w _ { k } \right\| ^ { 2 } + \frac { m } { m - 1 } \left\| w _ { k + 1 } \right\| ^ { 2 } + \left. \frac { 2 } { m } w _ { k } - \frac { 2 } { m - 1 } w _ { k + 1 } , p _ { k } \right. + \frac { 4 } { m - 1 } \left. w _ { k + 1 } , d _ { k } \right. } } \\ & { { \displaystyle \quad \quad = \frac { 4 } { m ( m - 1 ) } \left. p _ { k } , d _ { k } \right. - \frac { 4 } { m ( m - 1 ) } \left\| d _ { k } \right\| ^ { 2 } } } \\ & { { \displaystyle \quad \quad = \frac { 4 } { ( N - k ) ( N - k - 1 ) } \left. u _ { k + 1 } - u _ { N } , g _ { k + 1 } - g _ { N } \right. } , } \end{array}
$$

where in the third equality we substituted $\begin{array} { r } { w _ { k + 1 } = \frac { m - 1 } { m } w _ { k } - \frac { 2 } { m } d _ { k } } \end{array}$ and canceled out the terms, and in the last equality we used

$$
p _ { k } - d _ { k } = x _ { k } - x _ { N - 1 } - \left( g _ { k + 1 } - g _ { N } \right) = u _ { k + 1 } - u _ { N } .
$$

Summing over $k = 0 , \ldots , N - 2$ and using $V _ { N - 1 } = 0$ yields

$$
V _ { 0 } = \sum _ { j = 1 } ^ { N - 1 } \frac { 4 } { ( N - j + 1 ) ( N - j ) } \left. u _ { j } - u _ { N } , g _ { j } - g _ { N } \right. .
$$

On the other hand, since $z _ { 0 } = 0$

$$
V _ { 0 } = - \frac { N - 1 } { N } \left\| 2 g _ { N } \right\| ^ { 2 } + \frac { 2 } { N } \left. 2 g _ { N } , x _ { 0 } - x _ { N - 1 } \right. = - \frac { 4 } { N } \left( \left( N - 1 \right) \left\| g _ { N } \right\| ^ { 2 } + \left. g _ { N } , x _ { N - 1 } - x _ { 0 } \right. \right) .
$$

Therefore, equating the two expressions for $V _ { 0 }$ and multiplying $\textstyle { \frac { N } { 4 } }$ throughout, we obtain

$$
0 = ( N - 1 ) \left\| g _ { N } \right\| ^ { 2 } + \left. g _ { N } , x _ { N - 1 } - x _ { 0 } \right. + \sum _ { j = 1 } ^ { N - 1 } \frac { N } { ( N - j ) ( N - j + 1 ) } \left. u _ { j } - u _ { N } , g _ { j } - g _ { N } \right. .\tag{11}
$$

Finally, substituting $u _ { j } = x _ { j - 1 } - g _ { j }$ and $\begin{array} { r } { g _ { j } = \frac { \alpha } { 2 } G _ { j } } \end{array}$ , we have

$$
\begin{array} { l } { \displaystyle \left. u _ { j } - u _ { N } , g _ { j } - g _ { N } \right. = \left. x _ { j - 1 } - x _ { N - 1 } , g _ { j } - g _ { N } \right. - \left\| g _ { j } - g _ { N } \right\| ^ { 2 } } \\ { \displaystyle \qquad = \frac { \alpha } { 2 } \left( \left. x _ { j - 1 } - x _ { N - 1 } , G _ { j } - G _ { N } \right. - \frac { \alpha } { 2 } \left\| G _ { j } - G _ { N } \right\| ^ { 2 } \right) , } \end{array}
$$

and

$$
\left( N - 1 \right) \left. g _ { N } \right. ^ { 2 } = \frac { ( N - 1 ) \alpha ^ { 2 } } { 4 } \left. G _ { N } \right. ^ { 2 } , \qquad \left. g _ { N } , x _ { N - 1 } - x _ { 0 } \right. = \frac { \alpha } { 2 } \left. G _ { N } , x _ { N - 1 } - x _ { 0 } \right. .
$$

Hence, recalling $G _ { j } = \mathbb { F } _ { B _ { j - 1 } } ( x _ { j - 1 } )$ and $\begin{array} { r } { \lambda _ { N , j } = \frac { N \alpha } { 2 ( N - j ) ( N - j + 1 ) } } \end{array}$ , (11) is exactly the claimed identity.

## A.2 Proof of Lemma 4.3

We can rewrite Lemma 4.2 as

$$
\begin{array} { l } { \displaystyle 0 \geq \mathbb { E } \left[ \frac { ( N - 1 ) \alpha ^ { 2 } } { 4 } \left\| \mathbb { F } ( x _ { N - 1 } ) \right\| ^ { 2 } + \frac { \alpha } { 2 } \left. \mathbb { F } ( x _ { N - 1 } ) , x _ { N - 1 } - x _ { 0 } \right. \right] } \\ { \displaystyle \quad + \sum _ { i = 1 } ^ { N - 1 } \lambda _ { N , j } \mathbb { E } \left[ \left. x _ { j - 1 } - x _ { N - 1 } , \mathbb { F } _ { B _ { j - 1 } } ( x _ { j - 1 } ) - \mathbb { F } ( x _ { N - 1 } ) \right. - \frac { \alpha } { 2 } \left\| \mathbb { F } _ { B _ { j - 1 } } ( x _ { j - 1 } ) - \mathbb { F } ( x _ { N - 1 } ) \right\| ^ { 2 } \right] . } \end{array}\tag{12}
$$

Now let $e _ { j - 1 } = \mathbb { F } _ { B _ { j - 1 } } ( x _ { j - 1 } ) - \mathbb { F } ( x _ { j - 1 } )$ be the error term at iteration $j - 1$ . Then we can write

$$
\begin{array} { l } { \displaystyle \mathbb { E } \left[ \left. x _ { j - 1 } - x _ { N - 1 } , \mathbb { F } _ { \mathcal { B } _ { j - 1 } } ( x _ { j - 1 } ) - \mathbb { F } ( x _ { N - 1 } ) \right. - \frac { \alpha } { 2 } \left\| \mathbb { F } _ { \mathcal { B } _ { j - 1 } } ( x _ { j - 1 } ) - \mathbb { F } ( x _ { N - 1 } ) \right\| ^ { 2 } \right] } \\ { \displaystyle = \mathbb { E } \left[ \left. x _ { j - 1 } - x _ { N - 1 } , \mathbb { F } ( x _ { j - 1 } ) - \mathbb { F } ( x _ { N - 1 } ) \right. - \frac { \alpha } { 2 } \left\| \mathbb { F } ( x _ { j - 1 } ) - \mathbb { F } ( x _ { N - 1 } ) \right\| ^ { 2 } \right] } \end{array}
$$

$$
\begin{array} { r l } & { \quad + \mathbb E \left[ \langle x _ { j - 1 } - x _ { N - 1 } , e _ { j - 1 } \rangle - \alpha \langle \mathbf F ( x _ { j - 1 } ) - \mathbf F ( x _ { N - 1 } ) , e _ { j - 1 } \rangle - \frac { \alpha } { 2 } \| e _ { j - 1 } \| ^ { 2 } \right] } \\ & { \quad + \mathbb E \left[ \langle - x _ { N - 1 } + \alpha \mathbf F ( x _ { N - 1 } ) , e _ { j - 1 } \rangle \right] - \frac { \alpha \sigma ^ { 2 } } { 2 B } } \end{array}
$$

where the last inequality uses cocoercivity of $\mathbb { F } , \mathbb { E } \left[ e _ { j - 1 } | \mathcal { F } _ { j - 1 } \right] = 0$ and $\boldsymbol { x } _ { j - 1 } - \alpha \mathbb { F } ( \boldsymbol { x } _ { j - 1 } )$ is deterministic on $\mathcal { F } _ { j - 1 }$ , and $\begin{array} { r } { \mathbb { E } \left[ \left. e _ { j - 1 } \right. ^ { 2 } \bigg \rvert \mathcal { F } _ { j - 1 } \right] \leq \frac { \sigma ^ { 2 } } { B } } \end{array}$ . Substituting the above into (12) and using $\mathbb { T } = \mathbb { I } - \alpha \mathbb { F } .$ we obtain (6).

## A.3 Proof of Lemma 4.4

Lemma A.1. For $k = 1 , \ldots , N - 1$ , the S-Dual-OHM iterates can be expressed as

$$
x _ { k } = \frac { 1 } { N } x _ { 0 } + \sum _ { t = 0 } ^ { k - 2 } \frac { 1 } { ( N - t - 1 ) ( N - t ) } \mathbb { T } _ { B _ { t } } ( x _ { t } ) + \frac { N - k } { N - k + 1 } \mathbb { T } _ { B _ { k - 1 } } ( x _ { k - 1 } ) .\tag{13}
$$

Proof. We use induction on $k .$ For k = 1, $\begin{array} { r } { x _ { 1 } = x _ { 0 } + \frac { N - 1 } { N } \big ( \mathbb { T } _ { \mathcal { B } _ { 0 } } ( x _ { 0 } ) - x _ { 0 } \big ) = \frac { 1 } { N } x _ { 0 } + \frac { N - 1 } { N } \mathbb { T } _ { \mathcal { B } _ { 0 } } ( x _ { 0 } ) } \end{array}$ , which agrees with (13). Now assume (13) holds for some $k \in \mathbf { \bar { \{ 1 , . . . , N - 2 \} } }$ . Then, by induction hypothesis,

$$
\begin{array} { l } { { \displaystyle x _ { k + 1 } = x _ { k } + \frac { N - k - 1 } { N - k } \Big ( \mathbb { T } _ { \mathcal { B } _ { k } } ( x _ { k } ) - \mathbb { T } _ { \mathcal { B } _ { k - 1 } } ( x _ { k - 1 } ) \Big ) } } \\ { { \displaystyle \quad = \frac { 1 } { N } x _ { 0 } + \sum _ { t = 0 } ^ { k - 2 } \frac { 1 } { ( N - t - 1 ) ( N - t ) } \mathbb { T } _ { \mathcal { B } _ { t } } ( x _ { t } ) + \left( \frac { N - k } { N - k + 1 } - \frac { N - k - 1 } { N - k } \right) \mathbb { T } _ { \mathcal { B } _ { k - 1 } } ( x _ { k - 1 } ) } } \\ { { \displaystyle \quad \quad + \frac { N - k - 1 } { N - k } \mathbb { T } _ { \mathcal { B } _ { k } } ( x _ { k } ) } . } \end{array}
$$

The induction (hence the proof) is then complete, as $\begin{array} { r } { \frac { N - k } { N - k + 1 } - \frac { N - k - 1 } { N - k } = \frac { 1 } { ( N - k ) ( N - k + 1 ) } . } \end{array}$

Lemma A.2 (Leave-one-out stability). For $s \in \{ 0 , \ldots , N - 2 \}$ , let $\left\{ x _ { k } ^ { \left( s \right) } \right\} _ { k = 0 } ^ { N - 1 }$ be the trajectory obtained from S-Dual-OHM by replacing only the minibatch $B _ { s }$ by an independent copy $B _ { s } ^ { \prime }$ , while keeping all other minibatches the same in the two runs. Then, for every $k = s + 1 , \ldots , N - 1$

$$
\mathbb { E } \left[ \left. x _ { k } - x _ { k } ^ { ( s ) } \right. ^ { 2 } \right] \leq { \frac { \alpha ^ { 2 } \sigma ^ { 2 } } { B } } \left( 1 + { \frac { ( N - k ) ( N - k - 1 ) } { ( N - s ) ( N - s - 1 ) } } \right) .\tag{14}
$$

In particular,

$$
\mathbb { E } \left[ \left. x _ { N - 1 } - x _ { N - 1 } ^ { ( s ) } \right. ^ { 2 } \right] \leq \frac { \alpha ^ { 2 } \sigma ^ { 2 } } { B } .\tag{15}
$$

Proof. Define $w _ { k , t } = \left\{ \begin{array} { l l } { \displaystyle \frac { 1 } { ( N - t - 1 ) ( N - t ) } , } & { 0 \leq t \leq k - 2 , } \\ { \displaystyle \frac { N - k } { N - k + 1 } , } & { t = k - 1 } \end{array} \right.$ so that by Lemma A.1,

$$
x _ { k } = \frac { 1 } { N } x _ { 0 } + \sum _ { t = 0 } ^ { k - 1 } w _ { k , t } \mathbb { T } _ { \mathcal { B } _ { t } } ( x _ { t } ) , \qquad x _ { k } ^ { ( s ) } = \frac { 1 } { N } x _ { 0 } + \sum _ { t = 0 } ^ { k - 1 } w _ { k , t } \mathbb { T } _ { \mathcal { B } _ { t } ^ { ( s ) } } ( x _ { t } ^ { ( s ) } ) ,
$$

where $B _ { t } ^ { ( s ) } = B _ { t }$ for $t \neq s$ , and $B _ { s } ^ { ( s ) } = B _ { s } ^ { \prime }$ . Since $\begin{array} { r } { \sum _ { t = 0 } ^ { k - 1 } w _ { k , t } = 1 - \frac { 1 } { N } } \end{array}$ , by Jensen’s inequality,

$$
\begin{array} { r l } & { d _ { k } ^ { ( s ) } : = \mathbb { E } \left[ \left. x _ { k } - x _ { k } ^ { ( s ) } \right. ^ { 2 } \right] \leq \mathbb { E } \left[ \displaystyle \sum _ { t = 0 } ^ { k - 1 } w _ { k , t } \left. \mathbb { T } _ { \mathcal { B } _ { t } } ( x _ { t } ) - \mathbb { T } _ { \mathcal { B } _ { t } ^ { ( s ) } } \big ( x _ { t } ^ { ( s ) } \big ) \right. ^ { 2 } \right] } \\ & { \quad = \displaystyle \sum _ { t = s } ^ { k - 1 } w _ { k , t } \mathbb { E } \left[ \left. \mathbb { T } _ { \mathcal { B } _ { t } } ( x _ { t } ) - \mathbb { T } _ { \mathcal { B } _ { t } ^ { ( s ) } } \big ( x _ { t } ^ { ( s ) } \big ) \right. ^ { 2 } \right] . } \end{array}\tag{16}
$$

Let $\textstyle \nu ^ { 2 } = { \frac { \alpha ^ { 2 } \sigma ^ { 2 } } { B } }$ for simplicity. We will show by induction on k that

$$
d _ { k } ^ { ( s ) } \leq \nu ^ { 2 } \left( 1 + \frac { ( N - k ) ( N - k - 1 ) } { ( N - s ) ( N - s - 1 ) } \right) .\tag{17}
$$

For $k = s + 1$ , from (16) we have

$$
d _ { s + 1 } ^ { ( s ) } \leq w _ { s + 1 , s } \mathbb { E } \left[ \left\| \mathbb { T } _ { \mathcal { B } _ { s } } ( x _ { s } ) - \mathbb { T } _ { \mathcal { B } _ { s } ^ { \prime } } ( x _ { s } ) \right\| ^ { 2 } \right] \leq \frac { N - s - 1 } { N - s } \cdot 2 \nu ^ { 2 }
$$

which agrees with (17). Here we used the fact that $\mathbb { T } _ { B _ { s } } ( x _ { s } )$ and $\mathbb { T } _ { B _ { \mathrm { s } } ^ { \prime } } ( x _ { s } )$ are estimators of $\mathbb { T } ( x _ { s } )$ of variance $\begin{array} { r } { \le \frac { \alpha ^ { 2 } \sigma ^ { 2 } } { B } = \nu ^ { 2 } } \end{array}$ , and they are conditionally independent on $\mathcal { F } _ { s }$ . Next, let $s + 2 \leq \ell \leq N - 1$ , and suppose (17) holds for all $t = s + 1 , \dotsc , \ell - 1$ . Note that in (16), we have $B _ { t } ^ { ( s ) } = B _ { t }$ for $t > s ,$ so by Assumption 3.2,

$$
\begin{array} { r l } & { \mathbb { E } \left[ \left. \mathbb { T } _ { \mathcal { B } _ { t } } ( x _ { t } ) - \mathbb { T } _ { \mathcal { B } _ { t } ^ { ( s ) } } \big ( x _ { t } ^ { ( s ) } \big ) \right. ^ { 2 } \right] = \mathbb { E } \left[ \mathbb { E } \left[ \left. \mathbb { T } _ { \mathcal { B } _ { t } } ( x _ { t } ) - \mathbb { T } _ { \mathcal { B } _ { t } } \big ( x _ { t } ^ { ( s ) } \big ) \right. ^ { 2 } \bigg | \mathcal { G } _ { t } ^ { ( s ) } \right] \right] } \\ & { \qquad \leq \mathbb { E } \left[ \mathbb { E } \left[ \left. x _ { t } - x _ { t } ^ { ( s ) } \right. ^ { 2 } \bigg | \mathcal { G } _ { t } ^ { ( s ) } \right] \right] = \mathbb { E } \left[ \left. x _ { t } - x _ { t } ^ { ( s ) } \right. ^ { 2 } \right] = d _ { t } ^ { ( s ) } } \end{array}
$$

where $\mathcal { G } _ { t } ^ { ( s ) } = \sigma ( \mathcal { B } _ { 0 } , \ldots , \mathcal { B } _ { t - 1 } , \mathcal { B } _ { s } ^ { \prime } )$ is the σ-algebra enlarged from $\mathcal { F } _ { t }$ to make $x _ { t } ^ { ( s ) }$ measurable. Applying this to (16) with the induction hypothesis, we obtain

$$
\begin{array} { r l r } {  { d _ { \ell } ^ { ( s ) } \leq 2 w _ { \ell , s } \nu ^ { 2 } + \sum _ { t = s + 1 } ^ { \ell - 1 } w _ { \ell , t } d _ { t } ^ { ( s ) } } } \\ & { } & { \leq 2 w _ { \ell , s } \nu ^ { 2 } + \sum _ { t = s + 1 } ^ { \ell - 1 } w _ { \ell , t } \nu ^ { 2 } ( 1 + \frac { ( N - t ) ( N - t - 1 ) } { ( N - s ) ( N - s - 1 ) } ) } \\ & { } & { = \nu ^ { 2 } [ 2 w _ { \ell , s } + \sum _ { t = s + 1 } ^ { \ell - 1 } w _ { \ell , t } ] + \frac { \nu ^ { 2 } } { ( N - s ) ( N - s - 1 ) } \sum _ { t = s + 1 } ^ { \ell - 1 } w _ { \ell , t } ( N - t ) ( N - t - 1 ) . } \end{array}\tag{18}
$$

Now using the explicit formula for $w _ { \ell , t } .$ , we can directly verify

$$
2 w _ { \ell , s } + \sum _ { t = s + 1 } ^ { \ell - 1 } w _ { \ell , t } = 1 - \frac { 1 } { N - s } + \frac { 1 } { ( N - s ) ( N - s - 1 ) } ,
$$

and

$$
\begin{array} { r l } { \ } & { \underset { t = s + 1 } { \overset { \ell - 1 } { \sum } } w _ { \ell , t } ( N - t ) ( N - t - 1 ) = \underset { t = s + 1 } { \overset { \ell - 2 } { \sum } } 1 + \underset { N - \ell + 1 } { \overset { N - \ell } { \sum } } ( N - \ell + 1 ) ( N - \ell ) } \\ & { = \ell - s - 2 + ( N - \ell ) ^ { 2 } = N - s - 2 + ( N - \ell ) ( N - \ell - 1 ) . } \end{array}
$$

Hence

$$
\begin{array} { c l c r } { { d _ { \ell } ^ { ( s ) } \leq \nu ^ { 2 } \left( 1 - \displaystyle \frac { 1 } { N - s } \right) + \displaystyle \frac { \nu ^ { 2 } } { ( N - s ) ( N - s - 1 ) } \Big ( N - s - 1 + ( N - \ell ) ( N - \ell - 1 ) \Big ) } } \\ { { \mathrm { } } } & { { = \nu ^ { 2 } \left( 1 + \displaystyle \frac { ( N - \ell ) ( N - \ell - 1 ) } { ( N - s ) ( N - s - 1 ) } \right) , } } \end{array}
$$

which completes the induction, proving (14). Then (15) follows immediately by setting $k = N - 1 . \quad \bigsqcup$ 1

Now we are ready to prove Lemma 4.4. Fix $j \in \{ 1 , \dots , N - 1 \}$ , and let $x _ { N - 1 } ^ { ( j - 1 ) }$ denote the leave-oneout terminal iterate from Lemma $\mathrm { A . 2 , }$ obtained by replacing only the minibatch $B _ { j - 1 }$ by an independent

copy. By the fact that $x _ { N - 1 } ^ { ( j - 1 ) }$ is conditionally independent with $e _ { j - 1 }$ (which depends only on $B _ { j - 1 } )$ on $\mathcal { F } _ { j - 1 }$ and the tower property, we have E $\left[ \Big \langle \mathrm { T } \big ( x _ { N - 1 } ^ { ( j - 1 ) } \big ) , e _ { j - 1 } \Big \rangle \right] = 0$ . Hence

$$
\begin{array} { r l } & { \mathbb { E } \left[ \langle e _ { j - 1 } , \mathbb { T } x _ { N - 1 } \rangle \right] = \mathbb { E } \left[ \left. e _ { j - 1 } , \mathbb { T } ( x _ { N - 1 } ) - \mathbb { T } \big ( x _ { N - 1 } ^ { ( j - 1 ) } \big ) \right. \right] } \\ & { \qquad \leq \sqrt { \mathbb { E } \left[ \left. e _ { j - 1 } \right. ^ { 2 } \right] \mathbb { E } \left[ \left. \mathbb { T } ( x _ { N - 1 } ) - \mathbb { T } \big ( x _ { N - 1 } ^ { ( j - 1 ) } \big ) \right. ^ { 2 } \right] } } \\ & { \qquad \leq \sqrt { \frac { \sigma ^ { 2 } } { B } \mathbb { E } \left[ \left. x _ { N - 1 } - x _ { N - 1 } ^ { ( j - 1 ) } \right. ^ { 2 } \right] } } \\ & { \qquad \leq \sqrt { \frac { \sigma ^ { 2 } } { B } \cdot \frac { \alpha ^ { 2 } \sigma ^ { 2 } } { B } } = \frac { \alpha \sigma ^ { 2 } } { B } . } \end{array}
$$

## A.4 Proof of Lemma 4.6

Let

$$
s _ { k } = \mathbb { E } \left\| x _ { k } - y _ { k } \right\| ^ { 2 } , \qquad M _ { k } = \operatorname* { m a x } _ { 0 \leq i \leq k } s _ { i } .
$$

It sufices to show that $\begin{array} { r } { M _ { k } \le \frac { \alpha ^ { 2 } \sigma ^ { 2 } } { B ( 1 - \gamma ^ { 2 } ) } } \end{array}$ . For $k = 0$ , we have $x _ { 0 } = y _ { 0 }$ , so we have $s _ { 0 } = M _ { 0 } = 0$ and the result is trivial. Let $k \geq 1 . \mathrm { \ A p p l y i n g { \Sigma } }$ Lemma A.1 for both S-Dual-OHM and its full-batch version (Dual-OHM), we obtain

$$
x _ { k } - y _ { k } = \sum _ { t = 0 } ^ { k - 1 } w _ { k , t } \left[ \mathbb { T } ( x _ { t } ) - \mathbb { T } ( y _ { t } ) + e _ { t } \right]
$$

where $e _ { t } = \mathbb { T } _ { B _ { t } } ( x _ { t } ) \ – \mathbb { T } ( x _ { t } )$ and $w _ { k , t } = \left\{ \begin{array} { l l } { \displaystyle { \frac { 1 } { ( N - t - 1 ) ( N - t ) } } , } & { 0 \leq t \leq k - 2 } \\ { \displaystyle { \frac { N - k } { N - k + 1 } } , } & { t = k - 1 } \end{array} \right.$ . Note that here $\mathbb { E } \left[ e _ { t } \vert \mathcal { F } _ { t } \right] =$

0 and $\begin{array} { r } { \mathbb { E } \left[ \left. e _ { t } \right. ^ { 2 } \Big | \mathcal { F } _ { t } \right] \leq \nu ^ { 2 } : = \frac { \alpha ^ { 2 } \sigma ^ { 2 } } { B } } \end{array}$ . Using $\begin{array} { r } { \sum _ { t = 0 } ^ { k - 1 } w _ { k , t } = 1 - \frac { 1 } { N } < 1 } \end{array}$ and Jensen’s inequality gives

$$
\begin{array} { r l } & { s _ { k } = \mathbb { E } \left[ \left. x _ { k } - y _ { k } \right. ^ { 2 } \right] \leq \displaystyle \sum _ { t = 0 } ^ { k - 1 } w _ { k , t } \mathbb { E } \left[ \left. \mathbb { T } ( x _ { t } ) - \mathbb { T } ( y _ { t } ) + e _ { t } \right. ^ { 2 } \right] } \\ & { \leq \displaystyle \sum _ { t = 0 } ^ { k - 1 } w _ { k , t } \left( \mathbb { E } \left[ \left. \mathbb { T } ( x _ { t } ) - \mathbb { T } ( y _ { t } ) \right. ^ { 2 } + \left. e _ { t } \right. ^ { 2 } \right] \right) \leq \displaystyle \sum _ { t = 0 } ^ { k - 1 } w _ { k , t } \left( \gamma ^ { 2 } \mathbb { E } \left[ \left. x _ { t } - y _ { t } \right. ^ { 2 } \right] + \nu ^ { 2 } \right) \leq \gamma ^ { 2 } M _ { k - 1 } + \nu ^ { 2 } } \end{array}
$$

where for the second inequality, we used conditional unbiasedness together with the tower property:

$$
\mathbb { E } \left[ \left. e _ { t } , \mathbb { T } ( x _ { t } ) - \mathbb { T } ( y _ { t } ) \right. \right] = \mathbb { E } \left[ \mathbb { E } \left[ \left. e _ { t } , \mathbb { T } ( x _ { t } ) - \mathbb { T } ( y _ { t } ) \right. | \mathcal { F } _ { t } \right] \right] = 0
$$

which holds because $\mathbb { T } ( x _ { t } ) - \mathbb { T } ( y _ { t } )$ is deterministic conditioned on $\mathcal { F } _ { t }$ and $\mathbb { E } \left[ e _ { t } \vert \mathcal { F } _ { t } \right] = 0$ . Now starting with $M _ { 0 } = 0$ and using induction on $k ,$ the above shows that

$$
M _ { k } \leq \operatorname* { m a x } \left\{ M _ { k - 1 } , \gamma ^ { 2 } M _ { k - 1 } + \nu ^ { 2 } \right\} \leq \operatorname* { m a x } \left\{ M _ { k - 1 } , \gamma ^ { 2 } { \frac { \nu ^ { 2 } } { 1 - \gamma ^ { 2 } } } + \nu ^ { 2 } \right\} = { \frac { \nu ^ { 2 } } { 1 - \gamma ^ { 2 } } } .
$$

## A.5 Proof of Lemma 4.7

Observe that for $k = 1 , 2 , \ldots$

$$
\begin{array} { r l } & { \displaystyle \| y _ { k + 1 } - y _ { k } \| = \frac { N - k - 1 } { N - k } \| \mathbb { T } ( y _ { k } ) - \mathbb { T } ( y _ { k - 1 } ) \| \le \gamma \| y _ { k } - y _ { k - 1 } \| } \\ & { \displaystyle \implies \| y _ { k + 1 } - y _ { k } \| \le \gamma ^ { k } \| y _ { 1 } - y _ { 0 } \| = \gamma ^ { k } \left\| \frac { N - 1 } { N } ( \mathbb { T } ( y _ { 0 } ) - y _ { 0 } ) \right\| \le \gamma ^ { k } ( 1 + \gamma ) \| y _ { 0 } - x _ { * } \| } \end{array}
$$

where the last inequality uses $\begin{array} { r } { \| \mathbb { T } ( y _ { 0 } ) - y _ { 0 } \| \leq \| \mathbb { T } ( y _ { 0 } ) - x _ { \star } \| + \| x _ { \star } - y _ { 0 } \| \leq ( 1 + \gamma ) \| y _ { 0 } - x _ { \star } } \end{array}$ ∥ (recall that we assume $x _ { 0 } = y _ { 0 } )$ . Thus

$$
\left\| y _ { N - 1 } - y _ { k } \right\| \leq \sum _ { j = k } ^ { N - 2 } \left\| y _ { j + 1 } - y _ { j } \right\| \leq \left( 1 + \gamma \right) \left\| y _ { 0 } - x _ { \star } \right\| \sum _ { j = k } ^ { N - 2 } \gamma ^ { j } \leq \frac { 1 + \gamma } { 1 - \gamma } \gamma ^ { k } \left\| x _ { 0 } - x _ { \star } \right\| .
$$

## B Additional Experiments

In the experiments of Section 5, we used for S-OHM the same batch size B selected for S-Dual-OHM. This was designed to isolate the efect of replacing the anchor update by the dual-anchor update while keeping all the other configurations identical. However, to verify that the observed empirical advantage of S-Dual-OHM is not an artifact of this experiment design, we repeat all four experiments while separately tuning the batch size of S-OHM over the same grid $B \in \{ 1 , 1 0 , 2 0 , 5 0 , 1 0 0 \}$ , under the same total sample budget $Q = N B$ . All the other experimental settings are unchanged. For simplicity, we only compare the results from S-Dual-OHM with tuned B and runs of S-OHM using the shared and tuned batch sizes.

![](images/96ed2e1d0c15cbb42d9b6efe4b52b0e4582469c743d1bdf145722a6a4bbc1138.jpg)

![](images/b10d58264e88a386347f89cfff7f01310a5cc47b57bdc29a612d6a66b70a1257.jpg)

![](images/b30600da6696c3a11e90c18d4629965eeef8dbea95df9c19828dceef82a4740f.jpg)

![](images/bea3529b9f1cd52875b17fc275694c2853a74ca4e10d1d2bbe56e5164fb841f9.jpg)  
Figure 3: Comparison after independently tuning the constant batch size B of S-OHM. (Top left) Worst-case nonexpansive operator (Experiment 1). (Top right) Finite-sum cocoercive operator (Experiment 2). (Bottom left) SCSC Huber-type minimax problem (Experiment 3), in the lowvariance regime with $\sigma = 0 . 0 5$ . (Bottom right) The same minimax problem from Experiment 3 in the high-variance regime with $\sigma \ : = \ : 1 . 5$ . The plots labeled as S-OHM (Tuned B) use the best $B \in \{ 1 , 1 0 , 2 0 , 5 0 , 1 0 0 \}$ selected under the given total sample budget. Solid curves are means over 10 independent runs, and shaded bands denote empirical 5th–95th percentiles.

As shown in Figure 3, independent tuning selects the same batch size for S-OHM and S-Dual-OHM in Experiment 2 and the low-variance regime in Experiment 3. In Experiment 1 and the high variance regime in Experiment 3, the batch sizes independently chosen for S-OHM difer. Nevertheless, even with its own optimized batch size, the final residual attained by S-OHM remains strictly larger than that of S-Dual-OHM. These results from controlled experiments further supports the empirical benefit/robustness of the dual-anchor mechanism over anchoring in stochastic settings.