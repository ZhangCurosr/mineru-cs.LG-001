# On the Slow Convergence to Trivial Solutions of Algorithms for Hard Optimization Problems

Ali Hussaini Umar<sup>1∗</sup> Jean Barbier<sup>2</sup> Matthieu Jonckheere<sup>3</sup> Manuel Sáenz<sup>4</sup>

<sup>1</sup>SISSA, Trieste, Italy. <sup>2</sup>International Center for Theoretical Physics, Trieste, Italy. <sup>3</sup>LAAS–CNRS, Toulouse, France. <sup>4</sup>Universidad de San Andrés, Argentina.

August 20, 2026

## Abstract

Hard combinatorial optimization problems, many of which are NP-hard, present fundamental algorithmic challenges. Average-case analysis on random instances has emerged as a powerful framework for understanding typical algorithmic performance beyond worst-case guarantees. A substantial body of work has established negative results: for suficiently hard instances (often controlled by the underlying graph connectivity/constraints density), no known polynomialtime algorithm can significantly outperform naive heuristics in the double asymptotic limit where both problem size and constraints density tend to infinity. We revisit this picture by studying the finite-size behavior of some optimization algorithms across easy, intermediate, and hard regimes. Through rigorous analysis of large-graph asymptotics combined with numerical experiments on canonical problems (maximum independent set and maximum K-SAT), we demonstrate that while algorithms do eventually converge to theoretically predicted bounds, this convergence can be remarkably slow. In the intermediate regime where instances are already highly constrained, local algorithms achieve solutions substantially better than their predicted performance in the highconstraint-density limit. This gap between finite-regime and asymptotic behavior has important practical implications: sophisticated algorithmic design remains crucial even when asymptotic theory predicts inevitable failure.

## 1 Introduction

Hard (combinatorial) optimization problems [22] ask for the best value of an objective over a discrete, typically exponentially large set of feasible configurations. Satisfiability, routing and scheduling, graph partitioning, and covering/packing are canonical examples of this type of problems. They serve both as modeling primitives in operations research (e.g., logistics via routing-type formulations [26]) and as algorithmic backbones in automated reasoning and formal verification (e.g., SAT-based verification [4]). Many such problems are NP-hard or NP-complete [21, 10], and a large body of foundational work documents this hardness across optimization tasks.

A paradigmatic representative is the maximum independent set (MIS) problem [10]. Given an undirected graph $G = ( V , E )$ , an independent set is a subset $I \subseteq V$ of pairwise non-adjacent nodes. An independent set is maximal if no node can be added to it, and maximum if its cardinality is the largest among all independent sets of $G ;$ this cardinality is the independence number $\sigma ( G )$ . Computing $\sigma ( G )$ is NP-hard [15], yet the problem arises naturally in scheduling [1], communication systems [23], interference-free resource allocation in wireless networks [24], or deposition dynamics in statistical physics [14]. Given an algorithm A that outputs an independent set of size $\sigma _ { \mathcal { A } } ( G )$ , we quantify its quality via the performance ratio

$$
{ \mathcal { P } } _ { A } ( G ) = { \frac { \sigma _ { A } ( G ) } { \sigma ( G ) } } .\tag{1}
$$

A widely used approach to understanding the typical behavior of algorithms on such problems is average-case analysis on random instances drawn from well-defined graph ensembles [7]. For MIS one typically considers Erdős-Rényi, random regular, or configuration-model graphs [5] and studies the asymptotic independence ratio achieved by constructive procedures [11]. On sparse Erdős-Rényi graphs $G ( n , \lambda / n )$ of mean degree $\lambda > 0 .$ , greedy heuristics that select nodes based on degree are provably near-optimal when $\lambda < e [ 1 4 , 3 0 ]$ , where e denotes Euler’s number. Analogous phase transitions have been established for configuration-model graphs with prescribed degree distributions [14]. When $\lambda > e ,$ , degree-greedy optimality remains unproven, though it is widely conjectured, and supported by extensive numerical evidence, that greedy strategies become strictly sub-optimal.

Algorithms for MIS span a broad spectrum. At one end are sequential greedy heuristics [8, 11, 13], which build an independent set by irrevocably adding nodes whose neighbors have not been selected; these admit eficient parallel implementations [18]. At the other end are message-passing methods such as belief propagation (BP) and its zero-temperature version, i.e. the max-product (MP) algorithm [32], a class of probabilistic algorithms rooted in statistical physics [3, 12, 19, 29] which iteratively exchange local messages to approximate maximum a posteriori (MAP) assignments. Applied to weighted independent set formulations, MP can recover near-optimal configurations and, under certain conditions, solves an LP relaxation of the original problem, although convergence guarantees on loopy graphs remain limited [28, 24]. We will see in this paper that, when properly formulated, specific algorithms in these two classes can yield similar performance.

An influential perspective on algorithmic limits was articulated by Coja-Oghlan and Efthymiou [6], who conjectured inherent barriers to surpassing simple heuristics on sparse random graphs based on intrinsic geometric properties of the space of solutions. For $G ( n , \lambda / n )$ with average degree $\lambda \gg 1$ a trivial (degree-blind) greedy procedure finds an independent set of size $( 1 + o ( 1 ) ) n \ln ( \lambda ) / \lambda$ with high probability [8, 11], which is asymptotically half the independence number $( { \mathrm { i . e . } } \ \sigma ( G ) / 2 )$ . No polynomial-time algorithm is known to exceed this threshold by a factor $( 1 + \epsilon )$ for any fixed $\epsilon > 0 .$ The combinatorial geometry of the solution space in this regime exhibits features, notably the overlap gap property [9], that appear to trap local and stable algorithms, leading to conjectures that no eficient algorithm can outperform the greedy baseline by a non-vanishing margin on such instances [6]. These conjectures parallel broader algorithmic-barrier hypotheses in random constraint satisfaction problems [16, 20].

Despite these conjectured barriers, recent work [31] has reported evidence that the greedy benchmark can be exceeded on finite-size instances by simple polynomial-time procedures. In this paper, we investigate this phenomenon systematically for the MIS problem across several families of algorithms (sequential greedy heuristics and message-passing methods) and show that, while the asymptotic convergence to the half-optimality threshold does appear to hold, the approach to this limit is remarkably slow as a function of the mean degree λ. As a result, practical algorithmic performance can deviate significantly from asymptotic predictions even in regimes traditionally considered hard. To demonstrate that this slow-convergence phenomenon is not specific to graph problems, we complement our analysis with an analogous study of algorithms for the Max K-SAT problem [20], highlighting the broader relevance of these conclusions across random combinatorial optimization. This conclusion could be particularly relevant due to the growing use of neural network-based solvers for combinatorial optimization, as in [25, 2, 31], whose empirical evaluations necessarily take place in the finite-size regime where classical baselines still perform well above their asymptotic limits.

The remainder ofthe paper is organized as follows. Section 2 introduces the algorithmic approaches studied: the max-product belief propagation algorithm for MIS, including its density-evolution description via population dynamics, and the sequential greedy algorithms (static degree greedy (SDG) and dynamic degree greedy (DG)). Section 3 presents our main results for MIS: the asymptotic characterization of SDG performance (Theorem 1), finite-size simulations for DG, SDG, and MP on Erdős-Rényi graphs, and a quantitative analysis of the rate at which the performance ratio converges to the halfoptimality bound. Section 4 extends the analysis beyond graph problems to the Max K-SAT setting, introducing the Random Clause (RC) algorithm, establishing its asymptotic performance (Theorem 2),

and presenting analogous finite-size results.

The code for all simulations can be found in the repository: https://github.com/Alingoge/ MIS-KSAT-code.

## 2 Algorithms for MIS and their asymptotic analysis

## 2.1 Message passing: The max-product algorithm

We employ max-product (MP), a message-passing algorithm for MAP estimation (see [19] for details), to find an MIS in the intermediate-hard regime.

Following [24], the (weighted) MIS problem can be cast as MAP inference in a binary pairwise Markov random field. Given a graph $G = ( V , E )$ with node weights $w _ { i } > 0$ , associate to each node a binary variable $x _ { i } \in \{ 0 , 1 \}$ , where $x _ { i } = 1$ indicates inclusion in the independent set and $x _ { i } = 0$ otherwise. The joint distribution of a node assignment takes the form

$$
P ( \mathbf { x } ) = \frac { 1 } { Z } \prod _ { i \in V } \exp ( w _ { i } x _ { i } ) \prod _ { ( i , j ) \in E } \mathbf { 1 } _ { \{ x _ { i } x _ { j } = 0 \} } .\tag{2}
$$

By construction, $P ( \mathbf { x } )$ assigns positive probability only to feasible independent set configurations, and its MAP maximizer therefore corresponds to a maximum-weight independent set.

Max-product (MP) algorithm. At every iteration $t ,$ every node $i \in V$ sends a pair of messages $\{ m _ { i  j } ^ { t } ( 0 ) , m _ { i  j } ^ { t } ( 1 ) \}$ to each of its neighbors $j \in \partial i$ , where ∂i represents the set of neighbors of i. Moreover, each node also maintains a set of beliefs $\{ b _ { i } ^ { t } ( 0 ) , b _ { i } ^ { t } ( 1 ) \}$ , which serves as the core mechanism to determine whether or not a node i should be included in the MIS according to its present state. To make implementation and analysis more straightforward, we perform the following transformation

$$
\gamma _ { i  j } ^ { t } = \ln \bigg ( \frac { m _ { i  j } ^ { t } ( 0 ) } { m _ { i  j } ^ { t } ( 1 ) } \bigg ) .\tag{3}
$$

This reparameterization allows each node to receive only one message per neighbor and maintain a single belief. This modification of the max-product is often called the “min-sum” algorithm [19], and is just a convenient reparameterization. In the rest of the paper, we refer to this as simply the max-product algorithm. The algorithm proceeds as follows:

1. Initialization. Without loss of generality, we set $m _ { i \to j } ^ { 0 } ( 0 ) = m _ { i \to j } ^ { 0 } ( 1 ) = 1$ at initialization. Therefore $\gamma _ { i  j } ^ { 0 } = 0$ for all $( i , j ) \in E$

2. Message updates. For each directed edge $( i , j )$ , update the message using the max-product recursion:

$$
\gamma _ { i \to j } ^ { t + 1 } = \left[ w _ { i } - \sum _ { u \in \partial i \backslash \{ j \} } \gamma _ { u \to i } ^ { t } \right] _ { + } \mathrm { w h e r e } [ z ] _ { + } : = \operatorname* { m a x } \{ z , 0 \} .\tag{4}
$$

3. Log-belief ratio computation. For each node $i ,$ compute the log-belief ratio:

$$
\sigma _ { i } ^ { t } = w _ { i } - \sum _ { u \in \partial i } \gamma _ { u  i } ^ { t } .\tag{5}
$$

4. State assignment. Based on the log-belief ratios, determine the new node states:

$$
x _ { i } ^ { t } = \left\{ \begin{array} { l l } { 1 } & { \mathrm { i f } \sigma _ { i } ^ { t } > 0 , } \\ { 0 } & { \mathrm { i f } \sigma _ { i } ^ { t } < 0 , } \\ { \frac 1 2 } & { \mathrm { i f } \sigma _ { i } ^ { t } = 0 . } \end{array} \right.\tag{6}
$$

The value $x _ { i } ^ { t } = 1 / 2$ indicates an undecided state.

5. Convergence check. Increment t and repeat from step 2 until the configuration vector $\mathbf { x } ^ { t }$ $( x _ { i } ^ { t } ) _ { i \in V }$ stabilizes.

When the algorithm is applied to a general loopy graph, it can lead to three possibilities: (i) the algorithm converges to the MAP estimate; (ii) the algorithm converges to a fixed point that does not correspond to the MAP assignment; or, (iii) the algorithm does not converge, which may occur in certain cases with too many cycles or other challenging structures in the graph. In practice, the algorithm often performs well and provides accurate approximate solutions for various graphical models and inference tasks.

Asymptotic analysis of max-product by density evolution and population dynamics. On sparse Erdős-Rényi graphs, neighborhoods are locally tree-like, enabling a density evolution description of max-product in the large graph, $n  \infty ,$ limit [19]. Let $\Gamma ^ { t }$ denote a random message on a typical (uniformly picked) directed edge at iteration t. For mean degree $\lambda ,$ the update induced by (4) takes the distributional form

$$
\Gamma ^ { t + 1 } \ { \overset { \underset { \mathrm { \tiny ~ l a w } } { } } { = } } \ \left[ W - \sum _ { r = 1 } ^ { D } \Gamma _ { r } ^ { t } \right] _ { + } ,\tag{7}
$$

where $D \sim \mathrm { P o i s s o n } ( \lambda ) , ( \Gamma _ { r } ^ { t } ) _ { r }$ are independent and identically distributed (i.i.d.) copies of $\Gamma ^ { t } ,$ and W is a random node weight drawn from Law $( w _ { i } )$ . Closed forms for (7) are rarely available, so we approximate the evolving law of $\Gamma ^ { t }$ via population dynamics (PD) [19]. The idea is to approximate the distribution of Γ through a sample of n i.i.d. copies of Γ. As $n \gg 1$ , the empirical distribution of such a sample should converge to the actual distribution of Γ. We present the PD implementation in the following algorithm 1.

Algorithm 1: Population Dynamics for Density Evolution   
Input : λ: average degree, n: population size, T: maximum iterations, $f _ { w } ( x ) = e ^ { - x } \colon$   
Output: $\Gamma ^ { T }$ : final population of messages, $\sigma ^ { T }$ : final population of beliefs.   
Initialize $\Gamma _ { i } ^ { 0 } = 0$ and $\sigma _ { i } ^ { 0 } = 0$ for all $i \in [ n ]$   
for t = 1 to T do   
foreach $i \in [ 1 , n ]$ do   
k ← Sample from Poisson $( \lambda ) + 1$   
$w _ { i } \gets f _ { w } ( k )$   
Sample k − 1 indices $\{ i ( 1 ) , i ( 2 ) , \ldots , i ( k - 1 ) \}$ independently and identically from [n]   
$\begin{array} { r } { \Gamma _ { i } ^ { t } \dot {  } [ w _ { i } - \sum _ { u = 1 } ^ { k - 1 } \Gamma _ { i ( u ) } ^ { t - 1 } ] _ { + } } \end{array}$   
foreach $i \in [ 1 , n ]$ do   
d ← Sample from Poisson(λ)   
$w _ { i } \gets f _ { w } ( d )$   
Sample d indices $\{ i ( 1 ) , i ( 2 ) , \ldots , i ( d ) \}$ independently and identically from $[ n ]$   
$\begin{array} { r } { \sigma _ { i } ^ { t } \gets w _ { i } - \sum _ { u = 1 } ^ { d } \Gamma _ { i ( u ) } ^ { t } } \end{array}$   
return $\Gamma ^ { T } , \sigma ^ { T }$

## 2.2 Sequential greedy algorithms

We consider sequential algorithms that construct an independent set by iteratively exploring nodes and making irrevocable decisions. At each step $t \geq 0$ , the node set is partitioned into three disjoint subsets

$$
V = \mathcal { U } _ { t } \cup \mathcal { A } _ { t } \cup \mathcal { B } _ { t } ,
$$

where $\mathcal { U } _ { t }$ are unexplored nodes, $\boldsymbol { A } _ { t }$ are active nodes (selected into the independent $\mathbf { \Pi } _ { s \in \mathbb { T } }$ , and $B _ { t }$ are blocked nodes (excluded because they are in the neighborhood of active nodes). The initialization is

$$
\mathcal { U } _ { 0 } = V , \qquad \mathcal { A } _ { 0 } = \emptyset , \quad \mathrm { a n d } \quad \mathcal { B } _ { 0 } = \emptyset .
$$

Given a selection rule for choosing a node $i _ { t } \in \mathcal { U } _ { t - 1 }$ , the update is

$$
\mathcal { N } ( i _ { t } ) = \{ j \in \mathcal { U } _ { t - 1 } : ( i _ { t } , j ) \in E \} ,
$$

$$
\begin{array} { r } { \mathcal { A } _ { t } = \mathcal { A } _ { t - 1 } \cup \{ i _ { t } \} , \qquad \mathcal { B } _ { t } = \mathcal { B } _ { t - 1 } \cup \mathcal { N } ( i _ { t } ) , \quad \mathrm { a n d } \quad \mathcal { U } _ { t } = \mathcal { U } _ { t - 1 } \setminus \big ( \{ i _ { t } \} \cup \mathcal { N } ( i _ { t } ) \big ) . } \end{array}
$$

The procedure stops at the first time $t ^ { \star }$ such that $\mathcal { U } _ { t ^ { \star } } = \emptyset$ . By construction, $\boldsymbol { \mathcal { A } } _ { t ^ { \star } }$ is a maximal independent set.

Static degree greedy (SDG) algorithm. Static degree greedy is a degree-based sequential algorithm. Let $d _ { i }$ denote the degree of node i in the original graph G (degrees are not updated during the run). At each iteration, SDG selects, uniformly at random, an unexplored node of minimum original degree. That is,

$$
i _ { t } \in \arg \operatorname* { m i n } _ { i \in \mathcal { U } _ { t - 1 } } d _ { i } .
$$

The sets $( \mathcal { U } _ { t } , \mathcal { A } _ { t } , \mathcal { B } _ { t } )$ are then updated exactly as above using $\mathcal { N } ( i _ { t } )$ . The algorithm terminates when ${ { \mathcal { U } } _ { t } } = \emptyset$ , returning $\boldsymbol { A } _ { t }$ as the output maximal independent set.

Dynamic degree greedy (DG) algorithm. Dynamic degree greedy is also a degree-based sequential algorithm but, in contrast to SDG, it selects nodes using the current degree of nodes within the unexplored subgraph. Let $G [ \mathcal { U } _ { t - 1 } ]$ denote the subgraph induced by the unexplored set at step $t - 1$ and let

$$
d _ { i } ^ { ( t - 1 ) } : = \deg _ { G [ \mathcal { U } _ { t - 1 } ] } ( i ) , \qquad i \in \mathcal { U } _ { t - 1 } ,
$$

be the degree of i restricted to unexplored nodes. At iteration t, DG selects, uniformly at random, a node of $\mathcal { U } _ { t - 1 }$ with minimum degree on $G [ \mathcal { U } _ { t - 1 } ]$ . That is,

$$
i _ { t } \in \arg \operatorname* { m i n } _ { i \in \mathcal { U } _ { t - 1 } } d _ { i } ^ { ( t - 1 ) } .
$$

It then updates $( \mathcal { U } _ { t } , \mathcal { A } _ { t } , \mathcal { B } _ { t } )$ by activating $i _ { t }$ and blocking its unexplored neighbors $\mathcal { N } ( i _ { t } )$ as before. DG terminates when ${ { \mathcal U } _ { t } } \ = \ \varnothing$ , returning a maximal independent set $\boldsymbol { A } _ { t }$ . Compared to SDG, DG is more adaptive because node priorities evolve with the remaining graph structure.

SDG difers from the DG rule only in the selection criterion: SDG uses fixed degrees from the original graph, whereas DG recomputes degrees on the induced subgraph of unexplored nodes. As every sequential algorithm, both run in at most $| V |$ iterations.

Asymptotic performance of SDG. Let $G \sim \mathrm { C M } _ { n } ( d _ { n } )$ be a random graph drawn from the configuration model of size $n \geq 1$ and degree sequence $d _ { n } ,$ with the empirical distribution of $d _ { n }$ converging weakly to some asymptotic degree distribution $( p _ { i } ) _ { i \geq 0 }$ . Here we will state our main theoretical result, which concerns the SDG algorithm. It allows us to track its performance in the large graph limit, which mirrors the density evolution analysis for max-product. For this, we will first define recursively an array of values that will appear in the theorem. For each $i , j \geq 1$ , define $m _ { i } ( j )$ according to

$$
\left\{ { \begin{array} { l l } { m _ { i } ( j ) = 0 } & { { \mathrm { f o r ~ } } j \leq i } \\ { m _ { i } ( j ) = { \frac { m _ { i - 1 } ( j ) } { \left( 1 + { \frac { m _ { i - 1 } ( i ) } { u _ { i - 1 } } } i ( i - 2 ) \right) ^ { \frac { 1 } { i - 2 } } } } } & { { \mathrm { f o r ~ } } j > i } \end{array} } \right. ;\tag{8}
$$

while, for phase-2, we get

$$
\left\{ { \begin{array} { l l } { m _ { 2 } ( j ) = 0 } & { { \mathrm { f o r ~ } } j \leq 2 } \\ { m _ { 2 } ( j ) = m _ { i - 1 } ( j ) e ^ { - 2 j { \frac { m _ { 1 } ( 2 ) } { u _ { 1 } } } } } & { { \mathrm { f o r ~ } } j > 2 } \end{array} } . \right.\tag{9}
$$

Define a second auxiliary sequence according to

$$
u _ { i } = \left\{ \begin{array} { l l } { \frac { u _ { i - 1 } } { \left( 1 + \frac { m _ { i - 1 } ( i ) } { u _ { i - 1 } } i ( i - 2 ) \right) ^ { \frac { 2 } { i - 2 } } } } & { \mathrm { f o r } i \neq 2 } \\ { u _ { i - 1 } e ^ { - 2 \frac { m _ { 1 } ( 2 ) } { u _ { 1 } } } } & { \mathrm { f o r } i = 2 } \end{array} . \right.
$$

The initial values of the sequences are fixed according to

$$
m _ { i } ( 0 ) = p _ { i } \quad ( \mathrm { f o r ~ a l l } i = 0 , 1 , . . . )
$$

and

$$
u _ { 0 } = \sum _ { i = 1 } ^ { \infty } i p _ { i } .
$$

In terms of these sequences, we define, for all $i \geq 1$

$$
I _ { i } = \left\{ \begin{array} { l l } { p _ { 0 } } & { \mathrm { f o r } i = 0 } \\ { \frac { u _ { i - 1 } } { 2 i } \left( 1 - \left( 1 + \frac { m _ { i - 1 } ( i ) } { u _ { i - 1 } } i ( i - 2 ) \right) ^ { - \frac { 2 } { i - 2 } } \right) } & { \mathrm { f o r } i \neq 0 , 2 } \\ { \frac { u _ { 1 } } { 4 } \left( 1 - e ^ { - 2 \frac { m _ { 1 } ( 2 ) } { u _ { 1 } } } \right) } & { \mathrm { f o r } i = 2 } \end{array} \right. .\tag{10}
$$

We then have the following characterization of the asymptotic size of the independent sets found by the SDG algorithm.

Theorem 1. Let $G \sim C M _ { n } ( d _ { n } )$ be a random graph drawn from the configuration model of size $n \geq 1$ and degree sequence $d _ { n } ,$ with the empirical distribution $o f d _ { n }$ converging weakly to some asymptotic degree distribution $( p _ { i } ) _ { i \geq 0 }$ . Also assume that the second moment of the empirical distribution of $\cdot \ d _ { n }$ converges to $\textstyle \sum _ { i \geq 1 } i ^ { 2 } p _ { i }$ . Ifwe denote the proportion ofnodes contained in the independent setfound by the static degree greedy algorithm run on $G b y \rho _ { S D G } ( G ) ^ { 1 }$ , we then have that, as n goes to infinity,

$$
\rho _ { S D G } ( G ) \stackrel { \mathbb { P } } {  } \sum _ { i = 0 } ^ { \infty } I _ { i } .
$$

By means of this theorem, we can then compute the asymptotic fraction of nodes contained in independent sets constructed by SDG. We provide its proof in Appendix A and use the result in the following section to study the asymptotic performance of SDG across diferent regimes of the mean degree

## 3 Discussion of the results for MIS

We now present the results for the algorithms we introduced in Section 2 on a sparse Erdős-Rényi graph $G ( n , \lambda / n )$ . Our primary objective is to compute the performance ratio of those algorithms. Interestingly, [6] shows that for $G ( n , \lambda / n )$ with a mean degree of $\lambda \gg 1$ , the independence number is approximately equal to $\sigma ( G ) \approx 2 n \ln ( \lambda ) / \lambda$ . Therefore, for sparse Erdős-Rényi graphs of finite-butlarge degree, the performance ratio $\mathcal { P } _ { A }$ will be approximated for our tested algorithms by replacing the denominator in (1) by this approximation of $\sigma ( G )$

![](images/93d3bfce8f2b46ad9e0348c075595b4d98675cefe23761fc5676e465a3216a70.jpg)  
Figure 1: Average performance ratio of the degree greedy (DG), static degree greedy (SDG), and max-product (MP) algorithms as a function of mean degree λ. The solid blue line represents the theoretical asymptotic performance o $: \mathrm { { S D G } } ,$ while the blue dots with error bars represent its finite-size simulation results. The red dashed line represents the DG simulation results. The orange dots with error bars represent the MP numerical results, and the orange diamonds represent the population dynamics (PD) predictions. For all three algorithms, the finite-size performance remains above the 1/2 asymptotic threshold. This provides comprehensive (both theoretical and numerical) evidence that local algorithms can produce independent sets of size exceeding the theoretical threshold in sparse Erdős-Rényi graphs even for practically large sizes and degree means. For DG and SDG, the results are averaged across 10 realizations of Erdős-Rényi graphs $G ( n , \lambda / n )$ with $n = 1 0 ^ { 5 }$ , while the MP results are averaged over 25 independent realizations of the same graph ensemble.

## 3.1 Results for sequential greedy algorithms

We start with sequential algorithms run on instances of $G ( n , \lambda / n )$ . We consider values of λ ranging from 10 to $1 0 ^ { 3 }$ and fix $n = 1 0 ^ { 5 }$ . On each graph instance, we apply the DG and SDG algorithms and record their performance ratios. For the SDG, we study both finite-size simulations and the asymptotic predictions provided by Theorem 1, while for DG, we only study its finite-size simulations. Unlike SDG, which selects according to the original fixed degree sequence, DG selects by degree in the evolving subgraph $G [ \mathcal { U } _ { t - 1 } ]$ . Thus, analyzing DG asymptotically requires tracking the full empirical degree distribution of $G [ \mathcal { U } _ { t } ]$ , rather than a finite phase recursion, which we leave for future work. Figure 1 shows the average performance ratio of these algorithms over 10 i.i.d. realizations of $G ( n , \lambda / n )$

We first discuss the DG results (red dots with error bars). The numerical simulations indicate that the DG algorithm produces independent sets whose sizes well exceed the theoretical threshold on average for sparse Erdős-Rényi graphs. Recall that, according to existing theoretical results, no classical algorithm, either supported by rigorous analysis or numerical evidence, has been shown to find, with high probability in polynomial time, an independent set of size $( 1 + \epsilon ) n \ln ( \lambda ) / \lambda$ for any fixed $\epsilon > 0$ Similarly, no classical algorithm is known to achieve a performance ratio that exceeds $1 / 2 ,$ , represented by the gray horizontal dashed line in Figure 1. Nevertheless, the results clearly show that incorporating degree information into the exploration process yields performance ratios above $1 / 2 .$ . For all cases tested with $\lambda \gg 1$ and $\lambda / n \ll 1$ , the performance ratio approaches an asymptotic value of approximately 0.64.

We next discuss the SDG results. First, the asymptotic predictions are in good agreement with the finite-size simulations, as shown by the close match between the blue solid line and the finite-size simulation points. Second, as expected, SDG performs worse than DG. Unlike DG, SDG is non-adaptive: it does not recompute node degrees on the induced subgraph of unexplored nodes. However, SDG also achieves performance ratios above the theoretical threshold $1 / 2 .$ Importantly, this improvement remains significant even for practically large values of λ.

![](images/a65334e96f0e63c4c326cd0d37d1a13d3baa65b6ff75f81720d15b597e50ece9.jpg)  
Figure 2: Convergence of the performance ratio of static degree greedy (SDG) towards the half-bound threshold. The plot displays $\mathcal { R } _ { \mathrm { S D G } } ( \lambda )$ given by (11), interpreted as a natural performance lower bound for a large class of non-trivial algorithms for MIS, on Erdős-Rényi graphs $G ( n , \lambda / n )$ as a function of the degree λ in the limit of large n. The blue solid line represents the convergence of the theoretical results of SDG, while the red dashed line represents a power law fit for large values of λ.

SDG being arguably the simplest non-trivial algorithm for MIS, its asymptotic theoretical prediction provides a natural lower bound/baseline for more sophisticated algorithms, such as DG. This motivates quantifying the rate at which the asymptotic performance ratio of SDG converges to the theoretical threshold as a function of the mean degree λ. In Figure 2, we plot, on a log–linear scale, the quantity

$$
\mathcal { R } _ { \mathrm { S D G } } ( \lambda ) = \frac { \sigma _ { \mathrm { S D G } } ( \lambda ) } { n \ln ( \lambda ) / \lambda } - 1 ,\tag{11}
$$

that is, the relative performance gain above the half-threshold bound, as a function of λ. To quantify the decay of this gain, we fit the data at large values of $\lambda ^ { 2 }$ with a power-law function of the form $y =$ $\lambda ^ { - a }$ . The fit yields a decay exponent $a \approx 0 . 4 2 3 1$ , indicating that the convergence of the performance gain toward zero is remarkably slow. Notably, there exists a broad range of large mean degrees over which the performance gain remains non-negligible. This is even the case at $\lambda = 1 0 ^ { 4 }$ . This behavior is consistent with the intuition that, for a very large mean degree, the graph becomes increasingly homogeneous, with most nodes having approximately the same degree. In this regime, SDG should behave similarly to simple greedy dynamics which are degree-blind and thus expected to approach the best algorithmic bound rather slowly.

The performance gain observed for both DG and SDG is not restricted to the asymptotic regime $n  \infty$ . It remains nearly unchanged even for sparse finite-size graphs. Figure 3 illustrates this behavior through a heatmap of the performance ratio of the sequential algorithms for various values of n, ranging from $1 0 ^ { 3 }$ to $1 0 ^ { 5 }$ . A similar performance gain is also observed for DG and other sophisticated classical and neural network-based algorithms in [31].

## 3.2 Results for max-product

Before implementing MP, we introduced a modification that improves both its convergence and performance. A priori, MP is expected to belong to the class of algorithms that cannot generate, with high probability and in polynomial time, an independent set of size $( 1 + \epsilon ) n \ln ( \lambda ) / \lambda$ in the limit of large n. Surprisingly, we find that parameterizing the local evidence with a positive weight that depends on the node’s degree significantly improves ${ \mathrm { M P } } ^ { * } s$ performance compared to the implementation of [24]. The weight is chosen as a monotonically decreasing function of a node’s degree, so that nodes with fewer neighbors receive larger weights. We bias the algorithm toward selecting low-degree nodes by assigning to each node $i \in V$ a positive weight

![](images/9a726176186ded68fcb15e879ffd7bde9506e11f7aa986ce68011ef7b5af6e18.jpg)

![](images/932e461f547c1f67dc0c1d7b620e5a31499b7fc3665a8109447aa32767294a89.jpg)  
Figure 3: Heatmap ofthe performance ratios ofthe degree greedy (DG) and static degree greedy (SDG) algorithms on Erdős-Rényi graphs $G ( n , \lambda / n )$ . The left panel presents the results for DG, while the right panel shows the results for SDG. Both greedy algorithms achieve high ratios (above $1 / 2 )$ for sparser graphs, but the ratios fall below $1 / 2$ for denser graphs.

$$
w _ { i } = \exp ( - | \partial i | ) .\tag{12}
$$

This modification encourages MP to construct independent sets consisting of weakly connected nodes, rather than treating all nodes as equally likely candidates. Such a prior also improves the stability of the message-passing iterations. However, this improvement does not hold for arbitrary monotonically decreasing functions of node degree. Empirically, we find that it is restricted to functions exhibiting at least exponential decay. In contrast, functions with polynomial decay do not induce a suficiently strong bias and were observed to hinder the convergence of MP.

In Figure 1, we report the average performance ratio of MP over 25 independent Erdős-Rényi graphs, for λ ranging from 10 to 500 and $n = 1 0 ^ { 5 }$ . The results are shown as orange points with error bars. We observe that the algorithm not only exceeds the theoretical threshold of $1 / 2 ,$ but also closely matches the asymptotic behavior ofSDG. To confirm the asymptotic behavior ofMP, we employ a PD algorithm to approximate the message distribution in the large graph limit. The approximated message distribution is then used to estimate the density of nodes that satisfy the inclusion condition for the independent set. The orange diamonds represent the estimated performance ratio of the MP algorithm obtained from PD, using population sizes of $\mathrm { 5 \times 1 0 ^ { 6 } }$ for all points except $\lambda = 5 \times 1 0 ^ { 2 }$ , where we used $1 0 ^ { 7 }$ to obtain a better estimate.

Coming back to the close alignment between the asymptotic MP and SDG average performance on Erdős-Rényi graphs, we find this result quite remarkable and interesting. Indeed, the two algorithms belong to fundamentally diferent families, with distinct implementation procedures and decision-making mechanisms. However, although MP and SDG both favor maximal independent sets containing lowdegree nodes, the correspondence between the two is not exact. In SDG, this preference is explicit: at each step it selects uniformly among unexplored nodes of minimum original degree, so that the resulting maximal independent set can be seen as a random outcome of an exploration process that repeatedly exhausts the current minimum-degree class before moving to higher degrees. In contrast, MP does not impose an explicit exploration order, but the choice of weights introduces a strong energetic preference for including low-degree nodes. In fact, MP approximately searches for configurations $\mathbf { x } \in \{ 0 , 1 \} ^ { V }$ maximizing $\sum _ { i } w _ { i } x _ { i }$ under the hard constraints $\textstyle \prod _ { i , j } \mathbf { 1 } _ { \{ x _ { i } x _ { j } = 0 \} }$ , and because of the choice of $w _ { i }$ in (12), the objective becomes monotonic in the degree classes. More precisely, if one groups nodes by their original degree and writes $\textstyle V = \bigcup _ { d \geq 0 } \mathcal { V } _ { d }$ , then the weighted objective can be decomposed as $\begin{array} { r } { \sum _ { d > 0 } e ^ { - d } | \{ i \in \mathcal { V } _ { d } : x _ { i } = 1 \} | } \end{array}$ . In the limit of infinitely fast decay (i.e., replacing $e ^ { - d } \mathfrak { b } \mathfrak { y } \varepsilon ^ { d }$ and letting $\varepsilon \downarrow 0$ or some other such limit), maximizing this quantity is equivalent to first maximizing the number of selected nodes in $\nu _ { d _ { \mathrm { m i n } } } .$ , then, among those maximizers, maximizing the number selected in $\nu _ { d _ { \mathrm { m i n } } + 1 }$ , and so on, which amounts to a hierarchical optimization over conditional measures restricted to progressively higher degree classes. This is closely aligned with the SDG philosophy of saturating the minimum-degree class before proceeding to the next one. While MP operates through soft local messages rather than an explicit greedy schedule, the exponential weighting makes its efective target measure concentrate on independent sets that are “as low-degree as possible”. This gives an intuition for why the two methods exhibit nearly identical performance in our experiments.

## 4 Slow convergence beyond the MIS: the case of Max K-SAT

The K-satisfiability (K-SAT) problem consists of finding an assignment of n Boolean variables $V _ { n } =$ $\{ x _ { 1 } , x _ { 2 } , \ldots , x _ { n } \}$ that satisfies a Boolean formula $\Phi _ { n }$ in conjunctive normal form (CNF) comprising m clauses. Each clause is a logical disjunction (OR) of exactly K literals, where a literal is either a variable $x _ { i }$ or its negation $\neg x _ { i }$ . In the random $K { \cdot } S A \mathrm { T }$ problem, the K literals in each clause are chosen uniformly at random, and the typical hardness of the problem is controlled primarily by the clause-tovariable ratio $\alpha = m / n$ , also called the clause density.

Standard SAT solvers ofer limited information when a formula is unsatisfiable: they simply report that no solution exists, even though assignments that violate only a small number of clauses may be perfectly acceptable in practice. To address this limitation, one introduces a cost function that measures the fraction of satisfied clauses, and the goal becomes finding the assignment that maximizes it. This optimization variant, which remains well-defined even when no fully satisfying assignment exists, is known as Max K-SAT [17].

## 4.1 Sequential algorithms for Max K-SAT

The NP-hardness of Max K-SAT naturally leads to the use of sequential algorithms, which employ greedy or randomized strategies to satisfy as large a fraction of clauses as possible. The simplest such algorithm is the random assignment, in which each Boolean variable is independently set to true with probability $1 / 2 [ 2 7 ]$ ; its performance serves as a baseline for any non-trivial algorithm.

Random clause (RC) algorithm. Here, we analyze the behavior of a similar sequential algorithm which we call the random clause (RC) algorithm, and motivated by the fact that it follows a strategy analogous to that of SDG. It operates as follows. Consider $K \ge 2$ and $\alpha > 0$ . The random K-CNF formula $\Phi _ { n } = \Phi _ { n } ( K , \alpha )$ is a conjunction of $m = \lfloor \alpha n \rfloor$ clauses defined over the variable set $V _ { n } = \{ x _ { 1 } , \ldots , x _ { n } \}$ . Each clause is formed by selecting K variables independently and uniformly at random from $V _ { n } ,$ , negating each independently with probability $1 / 2 .$ Starting from the empty assignment, the RC algorithm constructs a solution greedily as follows. At each step, it selects uniformly at random one undecided clause $( \mathrm { i . e . , }$ a clause that is neither already satisfied nor falsified by the current partial assignment) then picks uniformly at random one of its currently unassigned literals and sets the corresponding variable so that the chosen literal evaluates to true. By construction, the selected clause is immediately satisfied; however, other undecided clauses that share the newly assigned variable may lose a free literal or become satisfied as a side efect. Note that the algorithm treats all undecided clauses uniformly, without any preference for clauses with fewer remaining free literals. After exactly n steps, every variable has been assigned.

Asymptotic performance of RC. Here we state the main theoretical result regarding the RC algorithm for random Max $K { \cdot } S A \mathrm { T }$ In analogy with the SDG case, we first introduce the deterministic quantities that will appear in the theorem and that describe the limiting evolution of the algorithm.

Fix $K \geq 2$ and $\alpha > 0$ . Let $\Phi _ { n }$ be a random K-CNF formula with $m = \lfloor \alpha n \rfloor$ clauses on n variables, constructed with i.i.d. uniform variables and i.i.d. unbiased literal signs.

In order to describe the limiting performance of the algorithm, we introduce a family of deterministic functions $\big ( u _ { j } ( t ) \big ) _ { j = 1 , \dots , K } , s ( t )$ , and $f ( t )$ where $t \in [ 0 , 1 ]$ . Also define

$$
U ( t ) = \sum _ { j = 1 } ^ { K } u _ { j } ( t ) .
$$

Define $x ( t ) = 1 - t$ to be the density of unassigned variables. These functions are defined as the unique solution, on the maximal interval where $x ( t ) > 0$ and $U ( t ) > 0$ , of the following system of diferential equations, with $j = 1 , \ldots , K$

$$
\left\{ \begin{array} { l l } { \displaystyle \frac { d u _ { j } } { d t } = - \frac { j u _ { j } } { x } - \frac { u _ { j } } { U } + \frac { ( j + 1 ) u _ { j + 1 } } { 2 x } } \\ { \displaystyle \frac { d s } { d t } = \sum _ { j = 1 } ^ { K } \left( \frac { j u _ { j } } { 2 x } + \frac { u _ { j } } { U } \right) } \\ { \displaystyle \frac { d f } { d t } = \frac { u _ { 1 } } { 2 x } } \end{array} \right.\tag{13}
$$

The initial conditions are determined by the structure of the random K-CNF at time $t = 0 .$ . Since initially all clauses have length K and none are satisfied or falsified, we impose $u _ { K } ( 0 ) = \alpha$ , (for $j \le K - 1 ) u _ { j } ( 0 ) = 0 , s ( 0 ) = 0$ , and $f ( 0 ) = 0$ . In terms of this deterministic evolution, we may now characterize the asymptotic performance of the RC algorithm.

Theorem 2. Let $\Phi _ { n }$ be a random K-CNFformula with $m = \lfloor \alpha n \rfloor$ clauses and let $\rho _ { \mathrm { R C } } ( \Phi _ { n } )$ denote the proportion ofclauses satisfied by the random clause algorithm applied to $\Phi _ { n } . { \cal I } f t _ { * } = \operatorname* { s u p } \{ t \geq 0 : { \cal U } ( t ) >$ $0 \}$ , then, as $n  \infty ,$ , we have

$$
\rho _ { \mathrm { R C } } ( \Phi _ { n } ) \stackrel { \mathbb { P } } { \to } s ( t _ { * } ) ,
$$

with $U ( t )$ and $s ( t )$ as above.

Thus, the asymptotic satisfied-clause proportion achieved by the RC algorithm is completely characterized by the solution of the deterministic system (13), in the same spirit as the SDG limit is characterized by the recursively defined quantities $( m _ { i } ( j ) , u _ { i } , I _ { i } )$ . In Appendix B, we present the proof of this theorem along with the dynamical interpretations of the equations in (13).

## 4.2 Discussion of the results for RC

We now present the results for the RC algorithm for random Max K-SAT. We consider clause densities $\alpha = m / n$ ranging over a logarithmically spaced grid, and we study both finite-size simulations and the asymptotic predictions provided by Theorem 2. All finite-size simulations correspond to instances with $n = 1 0 ^ { 3 }$ variables, and each data point is averaged over 10 independent realizations.

Before discussing the simulations, we will argue that the performance of the random assignment of truth values is $1 - 2 ^ { - K }$ . If each variable is assigned independently and uniformly at random, a fixed $K \cdot$ clause is satisfied unless all its K literals are false. Since the literals have independent unbiased signs and the assignment is uniform, the probability that a given literal is false is $1 / 2 ,$ , and therefore the probability that all K literals are simultaneously false equals $2 ^ { - K }$ . Consequently, the expected fraction of satisfied clauses under random assignment is $1 - 2 ^ { - K }$ . By standard concentration arguments, in random $K { \cdot } S A \mathrm { T }$ this fraction concentrates around $1 - 2 ^ { - K }$ with high probability as $n \to \infty$ . Hence $1 - 2 ^ { - K }$ provides a natural baseline against which the performance of any algorithm should be compared. Similarly to the case of MIS, as the density of clauses $\alpha = m / n$ grows, we will see that the performance of RC slowly converges to that of a trivial solution: that is, to a random assignment.

![](images/4c83e8e37a4c84ff66334805d879b4266e4f7693c480272563d929a415b4f65d.jpg)

![](images/bb879ce4c9ebf816bc9ff6350e58d2af02e02d58f7d1c129cc9eade8d4b75434.jpg)  
Figure 4: Asymptotic performance of RC for Max K-SAT. On the left, we compare the ratio of satisfied clauses as a function of the clause density $( \alpha = m / n )$ of RC on simulations $( n = 1 0 0 0$ , ten simulations for each point) with the predictions from the limit equations for $K = 4$ . Here we can appreciate that this ratio converges asymptotically, as argued, to the random assignment performance. On the right, asymptotic ratio of satisfied clauses as a function of clause density $( \alpha = m / n )$ for diferent values of $K$ . These simulations suggest that the convergence to the limit follows a similar rate of decay for all K considered. In both graphs, the asymptotic performance of random assignments are plotted as dashed horizontal blue lines.

When the clause density $\alpha$ is large, each variable appears in a large number of clauses, and, by symmetry of the random model, approximately half of these appearances are as a positive literal and half as a negative literal. More precisely, for a fixed variable $x _ { i } .$ , the numbers of clauses in which it appears positively and negatively are both of order $\alpha ,$ and their diference is negligible compared to their magnitude as $\alpha \to \infty$ . As a consequence, assigning $x _ { i }$ the value 1 or 0 produces, in expectation, nearly the same net efect on the total number of satisfied clauses. Since this balance becomes increasingly sharp at high density, any local advantage in choosing one value over the other vanishes asymptotically. The assignment decisions therefore become efectively neutral with respect to the global clause count. In this regime, each step of the RC dynamics behaves, from the perspective of the overall number of satisfied clauses, almost as if a value were chosen uniformly at random. This explains why, as $\alpha  \infty ,$ the satisfied-clause proportion converges to the random assignment benchmark $1 - 2 ^ { - K } ;$ in the highdensity limit, random assignment is asymptotically optimal among strategies based only on such local information.

We now turn to the numerics. All of them can be found in Figure 4. In the first experiment, we fix $K = 4$ and compare finite-size simulations with $n = 1 0 ^ { 3 }$ against the theoretical prediction obtained by solving the limiting ODE system of Theorem 2. For each value of $\alpha ,$ we compute the empirical mean and standard deviation of the satisfied-clause proportion achieved by RC and compare these with the asymptotic prediction. The results show very good agreement between theory and simulation, even at this moderate system size. The finite-size error bars remain small across the explored density range, and the theoretical curve closely tracks the empirical averages.

As α increases, both the simulations and the theoretical curve exhibit a gradual convergence toward the horizontal baseline $1 - 2 ^ { - 4 } = 1 5 / 1 6$ . This confirms the qualitative prediction discussed above: in the high-density regime, the RC algorithm becomes indistinguishable from random assignment in terms of its achieved satisfied-clause proportion.

In a second set of experiments, we compute the theoretical predictions of Theorem 2 for several values of $K$ , keeping α on a logarithmic scale. The resulting curves display the same qualitative behaviour for all $K$ . For small and moderate clause densities, the RC algorithm significantly outperforms the random assignment baseline. However, as α grows, each curve slowly approaches its corresponding limit $1 - 2 ^ { - K }$ . The convergence toward this baseline is monotone and consistent across values of K, but it is slow on the logarithmic scale.

Even though the asymptotic theory predicts collapse to random assignment performance at high density, the approach to the limit occurs only for large values of α. Over a wide intermediate range of clause densities, the RC algorithm maintains a non-trivial advantage over random assignment. In this sense, the situation parallels the MIS setting discussed previously: although a simple local algorithm eventually matches the trivial baseline in a dense regime, it retains a meaningful performance gap for practically relevant densities.

## 5 Conclusion

In this work, we have analyzed sequential greedy heuristics and message-passing algorithms for finding maximum independent sets in sparse Erdős-Rényi graphs. Through rigorous analysis of large-graph asymptotics, coupled with extensive numerical experiments, we have shown that for large graphs of large-but-finite degree, those algorithms typically find an independent set of size well above the theoretically predicted bound for local algorithms valid for diverging degree; this phenomenology was also discovered numerically by [31]. The theoretical threshold is ultimately attained, but the approach to this limit is remarkably slow as a function of the mean degree. As a result, for practical large mean degree $( \mathrm { e . g . ~ } \lambda = 1 0 ^ { 4 } )$ , algorithmic performance can deviate significantly from asymptotic predictions even in regimes traditionally considered hard. We confirmed these conclusions for the Max K-SAT problem [20], highlighting the broader relevance of these conclusions across random combinatorial optimization. This gap between finite-regime and asymptotic behavior has important practical implications: sophisticated algorithmic design remains crucial, and evaluation practices need to be revisited accordingly.

Our conclusions are relevant within machine learning given the growing use of neural networkbased solvers for combinatorial optimization [25, 2, 31], whose empirical evaluations necessarily take place in the finite-but-large-size regime we study here. This ofers a theoretical account of an alreadyobserved empirical pattern: [2] found that graph neural networks trained for maximum independent set are outperformed by simple degree-based greedy, and [31] found that leading AI-inspired methods are beaten not just by state-of-the-art classical solvers but sometimes by degree-greedy itself, with many non-backtracking neural methods implicitly reasoning like degree-greedy without matching its performance. Our results suggest an underlying reason: at finite size, degree-biased exploration exploits slow corrections to the asymptotic limit that a solver optimized only toward the asymptotic optimum will miss. Consequently, a neural network solver’s distance from the asymptotic bound is a poor proxy for its practical quality. The relevant benchmark for a neural network solver is therefore not the asymptotic bound, but the finite-size performance already reachable by simple local algorithms — a standard that benchmarking eforts such as [25] are well positioned to adopt.

## References

[1] Christoph Ambühl and Monaldo Mastrolilli. Single machine precedence constrained scheduling is a vertex cover problem. Algorithmica, 53(4):488–503, 2009.

[2] Maria Chiara Angelini and Federico Ricci-Tersenghi. Modern graph neural networks do worse than classical greedy algorithms in solving combinatorial optimization problems like maximum independent set. Nature Machine Intelligence, 5(1):29–31, 2023.

[3] Jean Barbier, Florent Krzakala, Lenka Zdeborová, and Pan Zhang. The hard-core model on random graphs revisited. In Journal of Physics: Conference Series, volume 473, page 012021, 2013.

[4] Armin Biere, Hans van Maaren, and Toby Walsh. Handbook of satisfiability. SAGE Publications Limited, 2009.

[5] Béla Bollobás. Random graphs. In Modern graph theory, pages 215–252. Springer, 2011.

[6] Amin Coja-Oghlan and Charilaos Efthymiou. On independent sets in random graphs. Random Structures & Algorithms, 47(3):436–486, 2015.

[7] Alan Frieze and Colin McDiarmid. Algorithmic theory of random graphs. Random Structures & Algorithms, 10(1-2):5–42, 1997.

[8] Alan M Frieze. On the independence number of random graphs. Discrete Mathematics, 81(2):171– 175, 1990.

[9] David Gamarnik. The overlap gap property: A topological barrier to optimizing over random structures. Proceedings ofthe National Academy ofSciences, 118(41):e2108492118, 2021.

[10] Michael R Garey and David S Johnson. Computers and intractability, volume 29. wh freeman New York, 2002.

[11] Geofrey R Grimmett and Colin JH McDiarmid. On colouring random graphs. In Mathematical Proceedings of the Cambridge Philosophical Society, volume 77, pages 313–324. Cambridge University Press, 1975.

[12] Alexander K Hartmann and Martin Weigt. Phase transitions in combinatorial optimization problems: basics, algorithms and statistical mechanics. John Wiley & Sons, 2006.

[13] David S Johnson. Approximation algorithms for combinatorial problems. In Proceedings of the fifth annual ACM symposium on Theory of computing, pages 38–49, 1973.

[14] Matthieu Jonckheere and Manuel Sáenz. Asymptotic optimality of degree-greedy discovering of independent sets in configuration model graphs. Stochastic Processes and their Applications, 131:122–150, 2021.

[15] Richard M Karp. Reducibility among combinatorial problems. In 50 Years of Integer Programming 1958-2008: from the Early Years to the State-of-the-Art, pages 219–241. Springer, 2009.

[16] Florent Krzakała, Andrea Montanari, Federico Ricci-Tersenghi, Guilhem Semerjian, and Lenka Zdeborová. Gibbs states and the set of solutions of random constraint satisfaction problems. Proceedings of the National Academy of Sciences, 104(25):10318–10323, 2007.

[17] Chu Min Li and Felip Manya. Maxsat, hard and soft constraints. In Handbook of satisfiability, pages 613–631. IOS Press, 2009.

[18] Michael Luby. A simple parallel algorithm for the maximal independent set problem. In Proceedings of the seventeenth annual ACM symposium on Theory of computing, pages 1–10, 1985.

[19] Marc Mezard and Andrea Montanari. Information, physics, and computation. Oxford University Press, 2009.

[20] Marc Mézard and Riccardo Zecchina. Random k-satisfiability problem: From an analytic solution to an eficient algorithm. Physical Review E, 66(5):056126, 2002.

[21] Cristopher Moore and Stephan Mertens. The nature of computation. Oxford University Press, 2011.

[22] Christos H Papadimitriou and Kenneth Steiglitz. Combinatorial optimization: algorithms and complexity. Courier Corporation, 1998.

[23] Maytham Safar and Sami Habib. Hard constrained vertex-cover communication algorithm for wsn. In International Conference on embedded and ubiquitous computing, pages 635–649. Springer, 2007.

[24] Sujay Sanghavi, Devavrat Shah, and Alan S Willsky. Message passing for maximum weight independent set. IEEE Transactions on Information Theory, 55(11):4822–4834, 2009.

[25] Geri Skenderi, Lorenzo Bufoni, Francesco D’Amico, David Machado, Rafaele Marino, Matteo Negri, Federico Ricci-Tersenghi, Carlo Lucibello, and Maria Chiara Angelini. Benchmarking graph neural networks in solving hard constraint satisfaction problems. arXiv preprint arXiv:2602.18419, 2026.

[26] Paolo Toth and Daniele Vigo. Vehicle routing: problems, methods, and applications. SIAM, 2014.

[27] Vijay V Vazirani. Approximation algorithms, volume 1. Springer, 2001.

[28] Martin Wainwright, Tommi Jaakkola, and Alan Willsky. Tree consistency and bounds on the performance of the max-product algorithm and its generalizations. Statistics and computing, 14:143– 166, 2004.

[29] Martin Weigt and Haijun Zhou. Message passing for vertex covers. Physical Review E—Statistical, Nonlinear, and Soft Matter Physics, 74(4):046110, 2006.

[30] Nicholas C Wormald. Diferential equations for random processes and random graphs. The annals of applied probability, pages 1217–1235, 1995.

[31] Yikai Wu, Haoyu Zhao, and Sanjeev Arora. Unrealized expectations: Comparing AI methods vs classical algorithms for maximum independent set. Transactions on Machine Learning Research, 2026.

[32] Jonathan S Yedidia, William T Freeman, Yair Weiss, et al. Understanding belief propagation and its generalizations. Exploring artificial intelligence in the new millennium, 8(236-239):0018–9448, 2003.

## A Limit equations for the static degree greedy algorithm

Let $G \sim { \cal C } { \cal M } _ { n } ( \bar { d } )$ be a random graph drawn from the configuration model of size $n \geq 1$ and degree sequence <sup>¯</sup>d. Assume that $\textstyle \mu ( \cdot ) : = \sum _ { i < n } \delta _ { d _ { i } } ( \cdot )$ divided by n converges weakly towards some asymptotic degree distribution $( p _ { k } ) _ { k \geq 1 }$ . For every $i \geq 1$ we will denote by $\nu _ { i }$ the set of nodes of degree i in the graph G.

The SDG dynamics is divided into diferent stages that we will call phase-1, phase-2, etc. At first the dynamics is in phase-1 and only the unexplored nodes with degree 1 in G have an exponential clock. Thus, during this phase, only nodes of degree 1 are added to the activated set. This phase goes on until the time $\tau _ { 1 } : = \operatorname* { i n f } \{ t \geq 0 : \mathcal { V } _ { 1 } \cap \mathcal { U } _ { t } = \varnothing \}$ in which there are no more degree 1 nodes to explore. After this time, phase-2 begins. Analogously, during phase-2, each unexplored degree-2 node has an exponential clock. When some of these clocks rings, the corresponding node and its neighborhood are explored in the way described above. This phase goes on until stopping time $\tau _ { 2 } : = \operatorname* { i n f } \{ t \geq \tau _ { 1 } : \mathcal { V } _ { 2 } \cap \mathcal { U } _ { t } = \emptyset \}$ . In general, i $\mathrm { ~ f ~ } i \geq 1$ and we define inductively $\tau _ { i } : = \operatorname* { i n f } \{ t \geq \tau _ { i - 1 } : \mathcal { V } _ { i } \cap \mathcal { U } _ { t } = \emptyset \}$ , phase-i starts after $\tau _ { i - 1 }$ and finishes at time $\tau _ { i }$ and during this time all the remaining unexplored nodes of degree i are either activated or blocked.

For every $t \geq 0 _ { : }$ , we will let $\textstyle \mu _ { t } ( \cdot ) : = \sum _ { i \in \mathcal { U } _ { t } } \delta _ { d _ { i } } ( \cdot )$ be the degree counting measure of the unexplored nodes at time t and $U _ { t } : = \mu _ { t } ( \mathbb { N } ) + B _ { t }$ be the total number of unpaired half-edges at time t in $G ,$ where $\begin{array} { r } { \mu _ { t } ( \mathbb { N } ) = \sum _ { i > 1 } j \mu _ { t } ( j ) } \end{array}$ is the number of unpaired half-edges of unexplored nodes at time t and $B _ { t }$ be the number of half-edges of blocked nodes that have not been paired at time t.

To describe this dynamics as a Markov jump process in continuous time, we will use the principle ofdeferred decision which implies that, no matter the order in which the half-edges that compose the graph are matched, if the matching is at each time done uniformly among the unmatched half-edges, the resulting graph has the same distribution. We will thus only match half-edges when required by the dynamics. Therefore, we will only reveal the graph when we need to do so. Using this, we will now show that the dynamics can be tracked according to a Markov jump process with respect to the quantities $( U _ { t } , \mu _ { t } )$ . Suppose that, at a certain time $t \geq 0 _ { : }$ , the smallest degree of positive mass with respect to $\mu _ { t } ( \cdot )$ is $i \geq 1$ . We are then in the phase-i of the exploration process. For simplifying the limiting equations, the rate of activation of each of the degree-i nodes during phase-i will be fixed (for every $t < \tau _ { i } )$ to be equal to $U _ { t } / \mu _ { t } ( i )$

Notice that when a clock rings we should match all of its half-edges and update the state of its unexplored neighbours. For each one of these matchings and $j \geq 1$ , there is a probability $j \mu _ { t } ( j ) / U _ { t }$ that the corresponding half-edge is matched to an unexplored node of degree $j .$ Also, there is a probability $\begin{array} { r } { B _ { t } / U _ { t } = 1 - U _ { t } ^ { - 1 } \sum _ { i > 1 } i \mu _ { t } ( i ) } \end{array}$ that it is matched to a half-edge of a blocked node. Moreover, if this half-edge is matched to a half-edge of a degree-j node, the quantities of the process are modified in the following way

$$
\begin{array} { r } { \left\{ \begin{array} { l l } { U _ { t } \longrightarrow U _ { t } - 2 } \\ { \mu _ { t } ( i ) \longrightarrow \mu _ { t } ( i ) - 1 } \\ { \mu _ { t } ( j ) \longrightarrow \mu _ { t } ( j ) - 1 ; } \end{array} \right. } \end{array}
$$

while, if the half-edge of the activated node is matched to a half-edge of a blocked node, they are changed according to

$$
\begin{array} { r } { \left\{ \begin{array} { l l } { U _ { t } \longrightarrow U _ { t } - 2 } \\ { \mu _ { t } ( i ) \longrightarrow \mu _ { t } ( i ) - 1 . } \end{array} \right. } \end{array}
$$

By proofs similar to those in [14], during the phase-i the quantities $U _ { t - \tau _ { i - 1 } } / n$ and $\mu _ { t - \tau _ { i - 1 } } / n$ will converge in probability towards $u _ { t }$ and $\bar { \mu } _ { t }$ given by the solutions of

$$
\left\{ \begin{array} { l l } { \frac { d { \bar { u } } _ { t } } { d t } = - 2 i { \bar { u } } _ { t } } & { } \\ { \frac { d { \bar { \mu } } _ { t } ( i ) } { d t } = - \left( { \bar { u } } _ { t } + i ^ { 2 } { \bar { \mu } } _ { t } ( i ) \right) } & { } \\ { \frac { d { \bar { \mu } } _ { t } ( j ) } { d t } = - i j { \bar { \mu } } _ { t } ( j ) } & { \mathrm { f o r ~ } j \neq i . } \end{array} \right.
$$

Let $\bar { t } _ { i } : = \operatorname* { i n f } \{ t \geq 0 : \bar { \mu } _ { t } ( i ) = 0 \}$ . For each $i , j \geq 1$ , let $m _ { i } ( j )$ and $u _ { i }$ be the number of nodes of initial degree $j$ which remain unexplored and of unexplored half-edges, respectively, at the end of phase $i .$ These equations can then be easily integrated to obtain that, for $0 < t \leq \bar { t } _ { i }$

$$
\begin{array} { r } { \left\{ \begin{array} { l l } { \bar { u } _ { t } = u _ { i - 1 } e ^ { - 2 i t } } \\ { \bar { \mu } _ { t } ( j ) = 0 } & { \mathrm { f o r ~ } j < i } \\ { \bar { \mu } _ { t } ( i ) = - \frac { u _ { i - 1 } } { i ( i - 2 ) } e ^ { - 2 i t } + \left( m _ { i - 1 } ( i ) + \frac { u _ { i - 1 } } { i ( i - 2 ) } \right) e ^ { - i ^ { 2 } t } } \\ { \bar { \mu } _ { t } ( j ) = m _ { i - 1 } ( j ) e ^ { - i j t } } & { \mathrm { f o r ~ } j > i } \end{array} \right. } \end{array}
$$

for phase-i (with $i \neq 2 )$ ; and

$$
\left\{ \begin{array} { l l } { { \bar { u } _ { t } = u _ { 1 } e ^ { - 4 t } } } & { { } } \\ { { \bar { \mu } _ { t } ( j ) = 0 } } & { { \mathrm { f o r ~ } j < 2 } } \\ { { \bar { \mu } _ { t } ( 2 ) = - u _ { 1 } t e ^ { - 4 t } + m _ { 1 } ( 2 ) e ^ { - 4 t } } } & { { } } \\ { { \bar { \mu } _ { t } ( j ) = m _ { 1 } ( j ) e ^ { - 2 j t } } } & { { \mathrm { f o r ~ } j > 2 } } \end{array} \right.
$$

for phase-2. From this we get that, for $i \neq 2$

$$
\bar { t } _ { i } = \frac { 1 } { i ( i - 2 ) } \ln \left( 1 + \frac { m _ { i - 1 } ( i ) } { u _ { i - 1 } } i ( i - 2 ) \right) ;
$$

and

$$
\bar { t } _ { 2 } = \frac { m _ { 1 } ( 2 ) } { u _ { 1 } } .
$$

From the above explicit solutions, we get that, for $i \neq 2$

$$
\left\{ { \begin{array} { l l } { m _ { i } ( j ) = 0 } & { { \mathrm { f o r ~ } } j \leq i } \\ { m _ { i } ( j ) = { \frac { m _ { i - 1 } ( j ) } { \left( 1 + { \frac { m _ { i - 1 } ( i ) } { u _ { i - 1 } } } i ( i - 2 ) \right) ^ { \frac { 1 } { i - 2 } } } } } & { { \mathrm { f o r ~ } } j > i } \end{array} } \right. ;
$$

while, for phase-2, we get

$$
\begin{array} { r } { \left\{ \begin{array} { l l } { m _ { i } ( j ) = 0 } & { \mathrm { f o r ~ } j \leq 2 } \\ { m _ { 2 } ( j ) = m _ { i - 1 } ( j ) e ^ { - 2 j \frac { m _ { 1 } ( 2 ) } { u _ { 1 } } } } & { \mathrm { f o r ~ } j > 2 } \end{array} \right. . } \end{array}
$$

In a similar way, we get a second recursion for the number of unexplored nodes by the end of phase-i given by

$$
u _ { i } = \left\{ \begin{array} { l l } { \frac { u _ { i - 1 } } { \left( 1 + \frac { m _ { i - 1 } ( i ) } { u _ { i - 1 } } i ( i - 2 ) \right) ^ { \frac { 2 } { i - 2 } } } } & { \mathrm { f o r } i \neq 2 } \\ { u _ { i - 1 } e ^ { - 2 \frac { m _ { 1 } ( 2 ) } { u _ { 1 } } } } & { \mathrm { f o r } i = 2 } \end{array} . \right.
$$

Through this recursion, the values of all the $m _ { i } ( j )$ can be inductively obtained.

This allows us to track the values of the asymptotic number of nodes of each degree through the evolution of the dynamics. But we are ultimately not interested in this but rather the size of the independent set obtained by the end of the exploration process. However, we are already in a position to derive an expression for this. Notice that during each phase, the number of nodes added to the independent set is equal to the number of half-edges matched divided by 2i. This is true because during each exploration of a node, exactly 2i half-edges are matched. Thus, the limit of the number of nodes added to the independent set during phase-i is asymptotically given by the diference in number of unmatched half-edges at time $\tau _ { i - 1 }$ and $\tau _ { i }$ divided by 2i. Let $\mathcal { T } _ { i }$ be the number of nodes added to the independent set during phase-i divided by n. We then have that

$$
\mathcal { T } _ { i } = \left\{ \begin{array} { l l } { p _ { 0 } } & { \mathrm { f o r } i = 0 } \\ { \frac { 1 } { 2 i n } \left( U _ { \tau _ { i - 1 } } - U _ { \tau _ { i } } \right) } & { \mathrm { f o r } i \geq 1 } \end{array} \right. .
$$

By using that $\begin{array} { r } { \tau _ { i } \stackrel { \mathbb { P } } {  } \sum _ { j \leq i } \bar { t } _ { j } } \end{array}$ and the limiting evolution of $U _ { t } / n$ we then have that

$$
\begin{array} { r } { \mathcal { Z } _ { i } \stackrel { \mathbb { P } } { \to } \left\{ \begin{array} { l l } { p _ { 0 } } & { \mathrm { f o r } i = 0 } \\ { \frac { u _ { i - 1 } } { 2 i } \left( 1 - \left( 1 + \frac { m _ { i - 1 } ( i ) } { u _ { i - 1 } } i ( i - 2 ) \right) ^ { - \frac { 2 } { i - 2 } } \right) } & { \mathrm { f o r } i \not = 0 , 2 } \\ { \frac { u _ { 1 } } { 4 } \left( 1 - e ^ { - 2 \frac { m _ { 1 } ( 2 ) } { u _ { 1 } } } \right) } & { \mathrm { f o r } i = 2 } \end{array} \right. . } \end{array}
$$

This, then, proves the theorem.

## B Limit equations for the random clause algorithm

The proof for the limit equations for Random Clause follows a similar strategy to that of SDG in the previous section but has certain characteristics that render it simpler. In particular, there is no need to embed the process in continuous time and choose an appropriate activation rate for the continuous time process. Furthermore, there are no phases in its dynamics meaning that the drift of the stochastic process remains the same throughout the exploration process. This, in particular, implies that there is no need for an explicit integration of the equations that would be later used to find limiting durations of each phase, as was the case with SDG.

Fix an integer $K \geq 2$ and a density parameter $\alpha > 0$ . For each $n \geq 1$ , let $\Phi _ { n }$ be a random $K \cdot$ CNF formula on the variable set $V _ { n } = \{ x _ { 1 } , \ldots , x _ { n } \}$ with $m = \lfloor \alpha n \rfloor$ clauses. Each clause is generated independently by selecting $K$ variables uniformly from $V _ { n }$ (with or without replacement, which does not afect the scaling limit) and assigning independent random signs to the corresponding literals, each sign being positive or negative with probability $1 / 2$

We consider the following sequential assignment dynamics, which we call the Random Clause (RC) algorithm. Starting from the empty assignment at step $\ell = 0 _ { : }$ , at each step $\ell = 0 , 1 , \dots , n - 1$ we choose uniformly at random one clause among all clauses that are currently undecided, namely those clauses that are neither satisfied nor falsified by the current partial assignment. Inside the chosen clause we select uniformly one of its currently unassigned literals and assign its variable so that this literal becomes true. This rule is applied regardless of whether unit clauses are present: at every step the choice is uniform over all undecided clauses.

Let ℓ denote the discrete time index and set $t = \ell / n \in [ 0 , 1 ]$ for the associated rescaled time. A clause is said to have current length $j \in \{ 1 , \ldots , K \}$ at step ℓ if it is undecided and has exactly $j$ unassigned literals under the current partial assignment. We introduce the random variables $U _ { j } ^ { ( n ) } ( \ell )$ for the number of undecided clauses of current length j, $S ^ { ( n ) } ( \ell )$ for the number of satisfied clauses, and $F ^ { ( n ) } ( \ell )$ for the number of falsified clauses. We define the corresponding densities

$$
\begin{array} { l } { { u _ { j } ^ { ( n ) } ( t ) = \displaystyle \frac { 1 } { n } U _ { j } ^ { ( n ) } ( \lfloor n t \rfloor ) , } } \\ { { \displaystyle s ^ { ( n ) } ( t ) = \displaystyle \frac { 1 } { n } S ^ { ( n ) } ( \lfloor n t \rfloor ) , } } \\ { { \displaystyle f ^ { ( n ) } ( t ) = \displaystyle \frac { 1 } { n } F ^ { ( n ) } ( \lfloor n t \rfloor ) , } } \end{array}
$$

and the total density of undecided clauses

$$
U ^ { ( n ) } ( t ) = \sum _ { j = 1 } ^ { K } u _ { j } ^ { ( n ) } ( t ) .
$$

At time $t = 0$ we have $s ^ { ( n ) } ( 0 ) = f ^ { ( n ) } ( 0 ) = 0$ and $u _ { K } ^ { ( n ) } ( 0 ) \to \alpha$ in probability while $u _ { j } ^ { ( n ) } ( 0 ) \to 0$ for $j \le K - 1$

The process

$$
\left( U _ { 1 } ^ { ( n ) } ( \ell ) , \ldots , U _ { K } ^ { ( n ) } ( \ell ) , S ^ { ( n ) } ( \ell ) , F ^ { ( n ) } ( \ell ) \right)
$$

is Markovian. Indeed, once the current partial assignment and the current status of each clause are known, the distribution of the next step depends only on this information. As in the proof of the limit equations for SDG in the previous section, in order to compute the conditional expected increments of these variables, we invoke the principle of deferred decision. Conditional on the current filtration $\mathcal { F } _ { \ell }$ the identities of the not-yet-exposed literals inside undecided clauses may be regarded as uniformly random among the $x ( t ) n$ unassigned variables, and the signs of those literals remain independent and unbiased. In particular, for a typical undecided clause of current length $j ,$ , its $j$ unassigned variables behave, to leading order as $n  \infty ,$ like $j$ independent uniform samples from the $x ( t ) n$ unassigned variables.

Fix ℓ and write $t = \ell / n$ . We compute the conditional drift of $U _ { j } ^ { ( n ) } ( \ell )$ . There are two distinct mechanisms that modify the number of undecided clauses of current length $j$ when the $( \ell + 1 ) { \ - } \mathtt { s t }$ assignment is performed.

First, the clause chosen by the RC rule is removed from the undecided pool and becomes satisfied. Since the choice is uniform among undecided clauses, the conditional probability that the chosen clause

has current length j equals

$$
\frac { U _ { j } ^ { ( n ) } ( \ell ) } { \sum _ { r = 1 } ^ { K } U _ { r } ^ { ( n ) } ( \ell ) } .
$$

Whenever this occurs, $U _ { j } ^ { ( n ) }$ decreases by one. This produces a contribution

$$
- \frac { U _ { j } ^ { ( n ) } ( \ell ) } { \sum _ { r = 1 } ^ { K } U _ { r } ^ { ( n ) } ( \ell ) }
$$

to the conditional expectation of $\Delta U _ { i } ^ { ( n ) } ( \ell ) = U _ { i } ^ { ( n ) } ( \ell + 1 ) - U _ { i } ^ { ( n ) } ( \ell )$

Second, we must account for all other clauses that contain the variable v assigned at step $\ell + 1$ Conditional on $\mathcal { F } _ { \ell } .$ , let v denote the variable set at this step. Consider a fixed undecided clause C of current length j. Under deferred decision, the probability that v appears among the j unassigned literals of C equals $j / ( x n ) + o ( 1 / n )$ . Summing over the $U _ { j } ^ { ( n ) } ( \ell )$ such clauses, the expected number of length-j undecided clauses that contain v is

$$
\frac { j U _ { j } ^ { ( n ) } ( \ell ) } { x n } + o ( 1 ) .
$$

Whenever v appears in such a clause, that clause ceases to be length j after the assignment. Indeed, since literal signs are independent and unbiased, the literal involving v is satisfied with probability $1 / 2$ and falsified with probability $1 / 2$ . In the first case the clause becomes satisfied and leaves the undecided pool; in the second case that literal is removed and the clause remains undecided but with current length $j - 1$ . In both cases the clause is no longer counted in $U _ { j } ^ { ( n ) }$ . Therefore this mechanism contributes an expected decrease

$$
- \frac { j U _ { j } ^ { ( n ) } ( \ell ) } { x n } + o ( 1 )
$$

to E $[ \Delta U _ { i } ^ { ( n ) } ( \ell ) \mid \mathcal { F } _ { \ell } ] .$

Finally, clauses of current length $j + 1$ may become length $j$ when they contain v and the corresponding literal is falsified. The expected number of length- $( j + 1 )$ clauses containing v is

$$
\frac { \left( j + 1 \right) U _ { j + 1 } ^ { \left( n \right) } ( \ell ) } { x n } + o ( 1 ) .
$$

Among these, half in expectation have the literal falsified, hence remain undecided and decrease their length by one. This yields an expected increase

$$
\frac { ( j + 1 ) U _ { j + 1 } ^ { ( n ) } ( \ell ) } { 2 x n } + o ( 1 )
$$

in $U _ { j } ^ { ( n ) }$

Combining the three contributions we obtain

$$
\mathbb { E } \left[ \Delta U _ { j } ^ { ( n ) } ( \ell ) \mid \mathcal { F } _ { \ell } \right] = - \frac { j U _ { j } ^ { ( n ) } ( \ell ) } { x n } - \frac { U _ { j } ^ { ( n ) } ( \ell ) } { \sum _ { r = 1 } ^ { K } U _ { r } ^ { ( n ) } ( \ell ) } + \frac { ( j + 1 ) U _ { j + 1 } ^ { ( n ) } ( \ell ) } { 2 x n } + o ( 1 ) ,
$$

with the convention $U _ { K + 1 } ^ { ( n ) } \equiv 0 .$

A similar reasoning gives the drift of $S ^ { ( n ) }$ and $F ^ { ( n ) }$ . The satisfied count increases by one whenever the chosen clause is removed, and additionally by one for each clause containing v whose literal is satisfied. The expected number of such additional satisfactions equals

$$
\sum _ { r = 1 } ^ { K } \frac { r U _ { r } ^ { ( n ) } ( \ell ) } { 2 x n } + o ( 1 ) .
$$

Hence

$$
\mathbb { E } \left[ \Delta S ^ { ( n ) } ( \ell ) \mid \mathcal { F } _ { \ell } \right] = 1 + \sum _ { r = 1 } ^ { K } \frac { r U _ { r } ^ { ( n ) } ( \ell ) } { 2 x n } + o ( 1 ) .
$$

A falsified clause can be created only when a unit clause contains v and its unique literal is falsified. The expected number of unit clauses touched is $U _ { 1 } ^ { ( n ) } ( \ell ) / ( x n ) + o ( 1 )$ , and half of them are falsified in expectation. Therefore

$$
\mathbb { E } \left[ \Delta F ^ { ( n ) } ( \ell ) \mid \mathcal { F } _ { \ell } \right] = \frac { U _ { 1 } ^ { ( n ) } ( \ell ) } { 2 x n } + o ( 1 ) .
$$

Dividing the drift equations by n and passing formally to the limit suggests that the limiting deterministic functions $u _ { j } ( t ) , s ( t )$ and $f ( t )$ satisfy, for $x ( t ) = 1 - t$ and $\begin{array} { r } { U ( t ) = \sum _ { j = 1 } ^ { K } u _ { j } ( t ) } \end{array}$

$$
\frac { d u _ { j } } { d t } = - \frac { j u _ { j } } { x } - \frac { u _ { j } } { U } + \frac { ( j + 1 ) u _ { j + 1 } } { 2 x } ,
$$

with $j = 1 , \ldots , K$ and $u _ { K + 1 } \equiv 0$ , together with

$$
\frac { d s } { d t } = \sum _ { j = 1 } ^ { K } \left( \frac { j u _ { j } } { 2 x } + \frac { u _ { j } } { U } \right) ,
$$

and

$$
{ \frac { d f } { d t } } = { \frac { u _ { 1 } } { 2 x } } .
$$

To justify this convergence rigorously, observe that each assignment step changes the state by at most $O ( 1 )$ clauses. Indeed, the number of clauses containing a fixed variable in a random $K { \mathrm { - C N F } }$ with m = αn clauses is asymptotically Poisson with mean αK, hence is tight uniformly in n. Therefore the increments of each density coordinate are bounded by $O ( 1 / n )$ with high probability. Furthermore, on any domain where $x ( t )$ and $U ( t )$ are bounded away from zero, the right-hand sides of the above ODE system are locally Lipschitz functions of $( u _ { 1 } , \dots , u _ { K } , s , f , t )$

By the standard diferential equation method [30], it follows that for every $\varepsilon > 0$ and for $t \in$ $[ \delta , 1 ]$ on which $x ( t ) \geq \varepsilon$ and $U ( t ) \geq \varepsilon$ , the process $( u _ { 1 } ^ { ( n ) } ( t ) , \ldots , u _ { K } ^ { ( n ) } ( t ) , s ^ { ( n ) } ( t ) , f ^ { ( n ) } ( t ) )$ converges in probability, uniformly on $[ 0 , T ]$ , to the unique solution of the above ODE system with initial conditions $u _ { K } ( 0 ) = \alpha , ( \mathrm { f o r } J \le K - 1 ) u _ { j } ( 0 ) = 0 , s ( 0 ) = 0 ,$ and $f ( 0 ) = 0$ . This finishes the proof of the theorem.