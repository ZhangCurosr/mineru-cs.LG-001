# Toward the Optimal Regret–Instability Trade-of in Multi-Armed Bandits

Kaifei Wang<sup>∗</sup> Yinyu Ye<sup>†‡</sup> Han Zhong<sup>‡</sup>

## Abstract

Multi-armed bandit algorithms are evaluated by regret, yet comparable regret can coexist with diferent allocations across independent runs. We study the trade-of between worst-case regret $\mathcal { R } _ { K , T }$ and instability $\scriptstyle { \mathcal { S } } _ { K , T }$ , defined as the largest standard deviation of a terminal pull count, for K arms and T rounds. We prove the finite-time lower bound $\mathcal { R } _ { K , T } S _ { K , T } \geq C T ^ { 3 / 2 }$ , where C is independent of K and T, under a finite-time regret condition and without the regularity assumptions imposed in the prior asymptotic analysis. We also introduce Stabilized Lower-Envelope UCB (SLE-UCB), a new tunable algorithm combining a running lower-envelope index with a decreasing pull-count stabilizer. SLE-UCB satisfies $\mathcal { R } _ { K , T } \mathcal { S } _ { K , T } = O ( T ^ { 3 / 2 }$ log K), with an implicit constant independent of K and T, matching the lower bound exactly in T and within a logarithmic factor in K. To prove the instability bound, we develop a new ofline top-prefix representation that removes path dependence from online decisions. Together with single-reward perturbations and the Efron–Stein inequality, this representation controls pull-count variance. Thus, regret and instability depend reciprocally on K, while their product has no polynomial dependence on K. These results resolve the open question raised in the literature concerning the sharp arm-dependent regret–instability frontier.

## 1 Introduction

Learning and decision-making often occur simultaneously. In clinical trials, online recommendation, and sequential resource allocation, each opportunity must be assigned while the rewards of the available alternatives remain unknown. The multi-armed bandit model captures these problems in its simplest form: every action produces an immediate reward while also generating information for future decisions (Li et al., 2010; Gittins et al., 2011; Villar et al., 2015). The central challenge is to balance information acquisition with immediate reward. Regret has become the standard measure of this balance, comparing the reward collected by an algorithm with that attainable if the best arm were known in advance (Lattimore and Szepesv´ari, 2020).

Regret, however, is an average criterion. It reflects pull counts through their expectations but does not control how realized rewards and pull counts vary from one run to another. Two algorithms can therefore have similar regret while producing very diferent allocations across independent runs. This distinction matters because pull counts determine how many patients receive each treatment, how much trafic is assigned to each option, and how much data are available for subsequent inference.

A related concern arises at the level of realized performance. For example, in discussing MOSS (Audibert and Bubeck, 2009), Lattimore and Szepesv´ari (2020, Section 9.2) use a carefully tuned two-arm policy to illustrate how favorable average performance can coexist with markedly diferent outcomes across independent runs. Subsequent work formalizes this issue through regret tails and risk-sensitive policy design (Simchi-Levi et al., 2022, 2023; Fan and Glynn, 2025; Zhu and Simchi-Levi, 2025). These studies focus on reward outcomes rather than arm allocations, but they reinforce the same message: regret alone does not control an algorithm’s behavior across independent runs.

The allocation side of this issue is studied through sampling stability, rooted in Lai and Wei (1982). Broadly, stability requires the number of pulls of each arm to concentrate around a deterministic scale, thereby supporting conventional statistical inference after adaptive data collection. This property has been examined for UCB (Kalvit and Zeevi, 2021; Khamaru and Zhang, 2024; Han et al., 2024; Chen and Lu, 2025; Fan et al., 2025) and Thompson sampling and its variants (Kalvit and Zeevi, 2021; Fan and Glynn, 2022; Halder et al., 2025; Han, 2026; Yan and Zhong, 2026). Related experimental-design work studies the trade-of between online performance and statistical power (Simchi-Levi and Wang, 2023). These studies make clear that predictable allocation is not an automatic consequence of low regret; it is a separate property of the algorithm.

This longstanding literature raises a more structural question: can an algorithm be both regret-optimal and stable? Praharaj and Khamaru (2025) show that a broad class of minimax-optimal optimism-based algorithms fails to satisfy sampling stability. Subsequent work studies this question under complementary notions of stability. Halder et al. (2026) study stabilization through regularization under a related stability notion. Taken together, these studies reveal a general tension between regret and allocation stability. Chen and Lu (2026) provide a quantitative formulation of this question by studying the Pareto trade-of between worst-case regret and pull-count instability. Specifically, for a K-armed bandit run for T rounds, let $\mathcal { R } _ { K , T }$ denote its worst-case regret and let $\mathcal { S } _ { K , T }$ denote its worst-case instability, measured by the largest standard deviation of an arm’s terminal pull count. Formal definitions are provided in Section 2. Their lower bound applies to algorithms that treat identical arms symmetrically, pull every arm at least once in expectation, and achieve asymptotically sublinear worst-case regret. For every fixed K, their result implies that there exists a constant $C _ { 0 } > 0$ , independent of T, such that<sup>1</sup>

$$
\begin{array}{c} \operatorname* { l i m } _ { T  \infty } \operatorname* { i n f } _ { T ^ { 3 / 2 } } \mathcal { S } _ { K , T }  \\ { T  \infty } \end{array} > C _ { 0 } .\tag{1.1}
$$

On the achievability side, they introduce UCB-f, a tunable generalization of UCB1 (Auer et al., 2002) controlled by an exploration function $f ( \cdot )$ , and establish

$$
\mathcal { R } _ { K , T } = O _ { K } \left( \sqrt { T } f ( T ) \right) , \qquad S _ { K , T } = O _ { K } \left( \frac { T \log T } { f ( T ) } \right) .
$$

Here $O _ { K } ( \cdot )$ hides the dependence on K. Multiplying these bounds gives a regret–instability product of $O _ { K } ( T ^ { 3 / 2 } \log T )$ . With the dependence on K left implicit, the upper bound matches the $T ^ { 3 / 2 }$ polynomial dependence of the lower bound above but leaves a factor log T gap. To further understand the optimal regret–instability trade-of in K-armed bandits, three questions remain open: (i) whether the logarithmic gap in T can be closed, (ii) what the explicit and sharp dependence on K is, and (iii) whether a tunable algorithm can achieve this trade-of.

Contributions. Our contributions are fourfold.

• A finite-time lower bound. Building on the asymptotic lower bound of Chen and Lu (2026), we establish a finite-time counterpart under a mild condition that permits linear-in-T regret. Specifically, we prove

$$
\mathcal { R } _ { K , T } S _ { K , T } \geq C T ^ { 3 / 2 } ,
$$

where $C > 0$ is independent of T and K. Our proof adapts their argument without separately assuming symmetry, nondegenerate exploration, or asymptotic active learning. See Remark 7.3 for details.

• A new stabilized UCB design. We introduce Stabilized Lower-Envelope UCB (SLE-UCB), which combines two distinctive design ingredients: a running lower-envelope UCB index and a decreasing pullcount stabilizer. The lower envelope makes the reward-dependent component of the index monotone, while the stabilizer creates a deterministic gap between diferent pull counts. The stabilizer strength is tunable, allowing the algorithm to trade regret for lower allocation variability.

• A sharp K-dependent frontier. For SLE-UCB, we establish regret and instability bounds with $\sqrt { K }$ and $1 / \sqrt { K }$ dependence, respectively, up to logarithmic factors. Multiplying these bounds gives

$$
\mathcal { R } _ { K , T } ( \alpha ) \mathcal { S } _ { K , T } ( \alpha ) = O \left( T ^ { 3 / 2 } \log K \right) .
$$

![](images/fd35073b3a2e63e5eb4979ff01d8e5fad2ada48f438703b790d2036d8556fa4d.jpg)  
Figure 1: Schematic K-dependent regret–instability product comparison, with logarithmic factors and constants suppressed in the axis annotations. The solid blue and dashed orange curves denote the lower bound and the SLE-UCB upper bound, respectively. In the regret regime covered by Theorem 3.1, the gray region is excluded and the product bounds difer by at most a factor log K.

Together with the lower bound, this provides a near-optimal characterization of the regret–instability Pareto frontier for K-armed bandits, as illustrated in Figure 1. By varying $\alpha ,$ SLE-UCB traces this frontier, matching the lower bound exactly in T and up to a factor log K in $K .$ . This broader characterization both removes the log T gap left open by Chen and Lu (2026) and settles their open question about the K-dependence to logarithmic accuracy.

• A new instability analysis. We develop an ofline top-prefix representation of the online decisions made by SLE-UCB. This representation removes the path dependence created by sequential arm selection and converts a single-reward perturbation into a controlled change in terminal pull counts. Combining this perturbation argument with the Efron–Stein inequality (Efron and Stein, 1981; Boucheron et al., 2013) yields the finite-time instability bound. To the best of our knowledge, this combination has not previously been used to control terminal pull-count variance in bandit analysis. The approach may also be of independent interest for studying random sample sizes generated by other adaptive allocation rules.

Organization. Section 2 introduces the bandit model and performance criteria. Section 3 presents the lower bound, the SLE-UCB algorithm, and the resulting frontier comparison. Section 4 establishes the clippingbias and concentration results. Sections 5 and 6 provide the regret and instability analyses, respectively. Section 7 proves the K-armed lower bound, and Section 8 concludes.

## 2 Model and performance metrics

Multi-armed bandits. Consider a stochastic multi-armed bandit with $K \geq 2$ arms. For each arm $i \in [ K ]$ let $\{ X _ { i , n } \} _ { n \geq 1 }$ be an i.i.d. reward sequence with mean $\mu _ { i }$ , where $X _ { i , n }$ is the reward revealed on the n-th pull of arm i. These reward sequences are mutually independent across arms. Let $\mu ^ { \star } : = \operatorname* { m a x } _ { i \in [ K ] } \mu _ { i }$ denote the optimal mean reward, and let $\nu = \left( \nu _ { 1 } , \dots , \nu _ { K } \right)$ denote the bandit instance, where $\nu _ { i }$ is the distribution of $X _ { i , 1 }$ . We make the following assumption on the reward distributions.

Assumption 2.1 (Uniformly bounded means and sub-Gaussian rewards). There is a constant $M > 0$ such that $\mu _ { i } \in [ - M , M ]$ for every arm $i \in [ K ]$ and there is a constant $\sigma > 0$ such that the reward distribution of each arm is σ-sub-Gaussian, in the sense that for all $u \in \mathbb { R }$ and $i \in [ K ]$

$$
\mathbb { E } [ \exp \left( u ( X _ { i , 1 } - \mu _ { i } ) \right) ] \leq \exp \left( \frac { \sigma ^ { 2 } u ^ { 2 } } { 2 } \right) .
$$

Online learning and objective. The reward distributions are unknown, and the algorithm learns online from bandit feedback. Fix a number of rounds $T > K$ . In each round $t \in [ T ]$ , the algorithm selects an arm

$A _ { t } \in [ K ]$ based on the past and its internal randomization, and observes only the selected arm’s reward. For every $i \in [ K ]$ and $t \in [ T ]$ , define

$$
N _ { i } ( t ) : = \sum _ { s = 1 } ^ { t } { \bf 1 } \{ A _ { s } = i \} , \qquad N _ { i } ( 0 ) : = 0 .
$$

Thus, the reward observed at time t is $X _ { A _ { t } , N _ { A _ { t } } ( t ) }$ . We evaluate an algorithm along two dimensions: regret and instability. Regret captures learning performance, whereas instability, following Chen and Lu (2026), captures the variability of pull counts.

Definition 2.2 (Regret and instability). For a given instance ν and a given algorithm, regret and instability with K arms over T rounds are defined as

$$
\mathcal { R } _ { K , T } ( \nu ) = \sum _ { t = 1 } ^ { T } \mathbb { E } \left[ \mu ^ { \star } - \mu _ { A _ { t } } \right] , \quad \mathcal { S } _ { K , T } ( \nu ) = \operatorname* { m a x } _ { i \in [ K ] } \sqrt { \mathrm { V a r } [ N _ { i } ( T ) ] } .
$$

The regret $\mathcal { R } _ { K , T } ( \nu )$ measures the expected cumulative reward loss compared to the optimal arm. The instability $\mathcal { S } _ { K , T } ( \nu )$ is the largest standard deviation of an arm’s pull count across independent runs. Smaller instability therefore means that the algorithm produces more reproducible allocations across independent runs. We evaluate a bandit algorithm by its worst-case performance over the full class E of K-armed instances satisfying Assumption 2.1.

Definition 2.3 (Worst-case regret and instability). For a given algorithm, the worst-case regret and instability with K arms over T rounds are defined as

$$
\mathcal { R } _ { K , T } ( \mathcal { E } ) = \operatorname* { s u p } _ { \nu \in \mathcal { E } } \mathcal { R } _ { K , T } ( \nu ) , \qquad \mathcal { S } _ { K , T } ( \mathcal { E } ) = \operatorname* { s u p } _ { \nu \in \mathcal { E } } \mathcal { S } _ { K , T } ( \nu ) .
$$

We henceforth omit the instance-class E and simply write $\mathcal { R } _ { K , T }$ and $\mathcal { S } _ { K , T }$ for these worst-case quantities.

## 3 Fundamental limit and Stabilized Lower-Envelope UCB

In this section, we characterize the regret–instability trade-of for K-armed bandits, including its dependence on both the number of rounds T and the number of arms K. We first establish a lower bound and then present an achievability result that matches it exactly in T and up to a logarithmic factor in K.

## 3.1 K-armed lower bound

While Chen and Lu (2026) establish the $T ^ { 3 / 2 }$ regret–instability lower bound only asymptotically, under symmetry, nondegenerate exploration, and an asymptotic active-learning condition, the following theorem holds for every $K \geq 2$ and $T > K$ . It requires the regret condition $\mathcal { R } _ { K , T } \ \leq \ M T / 8$ , without separately imposing any of those three conditions.

Theorem 3.1 (K-armed regret–instability lower bound). Fix $K \ge 2$ and $T > K$ . Let $\mathcal { R } _ { K , T }$ and $\mathcal { S } _ { K , T }$ denote the worst-case regret and instability as in Definition 2.3. For the class of K-armed instances satisfying Assumption ${ \it 2 . 1 , }$ every algorithm whose worst-case regret satisfies $\mathcal { R } _ { K , T } \leq M T / 8$ obeys

$$
\mathcal { R } _ { K , T } \mathcal { S } _ { K , T } \geq C T ^ { 3 / 2 } .
$$

Here $C = C ( M , \sigma ) > 0$ depends only on the instance-class parameters $( M , \sigma )$ and is independent of $( T , K )$ Proof. See Section 7 for a detailed proof. □

Our proof largely follows the two-armed proof of Chen and Lu (2026) but uses a refined analysis to establish the result under weaker conditions; see Section 7 for details. Theorem 3.1 makes the trade-of explicit. At regret scale $O ( \alpha \sqrt { K T } )$ , instability must be $\Omega ( T / ( \alpha \sqrt { K } ) )$ . In particular, at the minimax regret rate $O ( \sqrt { K T } )$ , attained for example by MOSS (Audibert and Bubeck, 2009), instability must be $\Omega ( T / { \sqrt { K } } )$ Thus, a minimax-regret policy cannot keep terminal pull counts tightly concentrated across all instances, consistent with the instability established for a broad class of minimax-optimal optimism-based algorithms by Praharaj and Khamaru (2025). We next turn to the achievability side and propose an algorithm matching this scaling up to logarithmic factors.

Algorithm 1 Stabilized Lower-Envelope UCB (SLE-UCB)   
1: Input: number of rounds T, number of arms K, positive constants M, σ, and tuning parameter $\alpha \geq 1 .$   
2: Compute $\iota = \sqrt { \log ( K T ) } , B = M + 4 \sigma \iota , \beta = 8 B \iota , \mathrm { a n d } \lambda = \alpha \beta \sqrt { K / T } .$   
3: Sample a uniform random priority permutation $\omega \in { \mathfrak { S } } _ { K }$   
4: Initialization: Pull each arm once, observe the rewards and clip them.   
After initialization, each arm has one pull, so $N _ { i } ( K ) = 1$ for all i.   
5: for $t = K + 1 , \ldots , T$ do   
6: Compute $U _ { i } ( N _ { i } ( t - 1 ) )$ from (3.2) and $I _ { i } ( N _ { i } ( t - 1 ) )$ from (3.3) for each arm i.   
7: Choose $A _ { t } \in \arg \operatorname* { m a x } _ { i \in [ K ] } I _ { i } ( N _ { i } ( t - 1 ) )$ , breaking ties by the smallest priority rank $\omega ( i )$   
8: Pull $A _ { t } .$ , observe the raw reward from that arm, clip it, and update the corresponding count.   
9: end for

## 3.2 Stabilized Lower-Envelope UCB

We propose Stabilized Lower-Envelope UCB (SLE-UCB), a tunable algorithm for balancing regret and instability. It is a UCB-type algorithm with two design components: (i) replacing the standard empirical-meanplus-bonus statistic with its running lower envelope; and (ii) adding a logarithmic stabilizer that decreases with the pull count. These two components together create a stabilized index for each arm. At each time, the algorithm selects the arm with the largest stabilized index.

Raw rewards are clipped before computing the index. For each arm $i \in [ K ]$ and $s \in [ T ]$ , define

$$
Y _ { i , s } : = \operatorname* { m i n } \{ B , \operatorname* { m a x } \{ - B , X _ { i , s } \} \} , \quad B = M + 4 \sigma \iota , \quad \iota = \sqrt { \log ( K T ) } .\tag{3.1}
$$

When arm i is pulled for the s-th time, the algorithm uses $Y _ { i , s }$ in place of the raw reward $X _ { i , s }$ . The threshold B depends only on $M , \sigma , K$ , and $T ,$ and Lemma 4.1 shows that the resulting bias is negligible. Let $Y = ( Y _ { i , s } ) _ { i \in [ K ] , s \in [ T ] }$ denote the full array of independent clipped rewards, revealed sequentially as the arms are pulled. For $n \geq 1$ , define the average clipped reward by $\textstyle { \overline { { Y } } } _ { i } ( n ) = ( \sum _ { s = 1 } ^ { n } Y _ { i , s } ) / n$ . The lower-envelope UCB statistic is defined as

$$
U _ { i } ( n ) = \operatorname* { m i n } _ { 1 \leq s \leq n } \left\{ { \overline { { Y } } } _ { i } ( s ) + { \frac { \beta } { \sqrt { s } } } \right\} , \quad { \mathrm { w h e r e ~ } } \beta = 8 B \iota .\tag{3.2}
$$

The lower envelope is the minimum of the standard UCB statistic over all previous prefixes. Thus $U _ { i } ( n )$ is nonincreasing in n. For a tuning parameter $\alpha \geq 1$ , define the stabilized index by

$$
I _ { i } ( n ) = U _ { i } ( n ) + \lambda \log { \left( \frac { T } { n } \right) } , \quad \mathrm { w h e r e } ~ \lambda = \alpha \beta \sqrt { \frac { K } { T } } .\tag{3.3}
$$

Since $U _ { i } ( n )$ is nonincreasing in n and λ log $( T / n )$ is strictly decreasing in n, the stabilized index $I _ { i } ( n )$ is a strictly decreasing sequence in n for each arm i. During the initial period $t \leq K$ , SLE-UCB pulls each arm once. For $t > K$ , it chooses an arm with the largest stabilized index and resolves ties using the sampled priority order:

$$
A _ { t } \in \operatorname * { a r g m a x } _ { i \in [ K ] } I _ { i } ( N _ { i } ( t - 1 ) ) .\tag{3.4}
$$

We impose a priority order on the arms to handle numerical ties in the index values. To avoid arm-label bias, this order is chosen at random: before starting, SLE-UCB samples an independent uniform permutation $\omega \in { \mathfrak { S } } _ { K }$ and breaks ties in favor of the arm with the smallest rank $\omega ( i )$

In Section 6.1, we show that the monotonicity of $I _ { i } ( n )$ permits an ofline view of the online selection rule, which is crucial for the analysis of instability. This equivalence is formalized in Lemma 6.1. Moreover, the logarithmic stabilizer λ log $( T / n )$ gives a deterministic lower bound on the index gap between two pull counts of the same arm. Specifically, for all $1 \leq n _ { 1 } \leq n _ { 2 } \leq T$

$$
I _ { i } ( n _ { 1 } ) - I _ { i } ( n _ { 2 } ) = U _ { i } ( n _ { 1 } ) - U _ { i } ( n _ { 2 } ) + \lambda \log \frac { n _ { 2 } } { n _ { 1 } } \geq \lambda \log \frac { n _ { 2 } } { n _ { 1 } } \geq 0 .\tag{3.5}
$$

In the instability proof, this gap converts a bound on a single-reward index perturbation into a bound on the resulting change in the terminal pull count; see Section 6.3. Although the logarithmic stabilizer may influence arm selection, Section 5 shows that it only contributes a controlled additional term to regret.

## 3.3 Achievability and the Pareto frontier

The next theorem quantifies this trade-of through finite-time regret and instability bounds.

Theorem 3.2 (Regret–instability guarantees for SLE-UCB). Fix $2 \leq K < T$ and $\alpha \geq 1$ . Recall from (3.1) and (3.2) that $\iota = \sqrt { \log ( K T ) }$ , $B = M + 4 \sigma \iota$ , and $\beta = 8 B \iota$ . For SLE-UCB tuned with parameter α, let $\mathcal { R } _ { K , T } ( \alpha )$ and ${ \mathcal { S } } _ { K , T } ( \alpha )$ denote its worst-case regret and instability as in Definition 2.3. These quantities satisfy

$$
\mathcal { R } _ { K , T } ( \alpha ) \leq \operatorname* { m i n } \left\{ 2 M T , \big ( 5 + \alpha \log ( 2 K ) \big ) \beta \sqrt { K T } \right\} , \qquad \mathcal { S } _ { K , T } ( \alpha ) \leq 2 \sqrt { T } + \frac { 1 8 \sqrt { 2 } \sigma T } { \alpha \beta \sqrt { K } } .\tag{3.6}
$$

Consequently, for fixed M and σ, uniformly over $\alpha \geq 1$

$$
\mathcal { R } _ { K , T } ( \alpha ) \mathcal { S } _ { K , T } ( \alpha ) = O \left( T ^ { 3 / 2 } \log K \right) ,
$$

where the implicit constant is independent of K, T, and $\alpha .$

Proof.

See Sections 5 and 6 for detailed proofs of the regret and instability bounds, respectively. For the product bound, applying the two branches of the regret estimate in (3.6) to the corresponding terms of the instability estimate gives

$$
\begin{array} { r l } { \mathcal { R } _ { K , T } ( \alpha ) S _ { K , T } ( \alpha ) \le 2 \sqrt { T } \mathcal { R } _ { K , T } ( \alpha ) + \displaystyle \frac { 1 8 \sqrt { 2 } \sigma T } { \alpha \beta \sqrt { K } } \mathcal { R } _ { K , T } ( \alpha ) } & { } \\ { \le 4 M T ^ { 3 / 2 } + 1 8 \sqrt { 2 } \sigma \left( \displaystyle \frac { 5 } { \alpha } + \log ( 2 K ) \right) T ^ { 3 / 2 } } & { } \\ { = O \left( T ^ { 3 / 2 } \log K \right) , } \end{array}
$$

where the last equality uses $\alpha \geq 1$ and $K \geq 2$

Theorem 3.2 provides finite-time guarantees for the tunable SLE-UCB algorithm over the full range $\alpha \geq 1$ , and its product bound is uniform in α. Whenever $\mathcal { R } _ { K , T } ( \alpha ) \leq M T / 8$ , this product matches the lower bound exactly in its $T ^ { 3 / 2 }$ dependence and within a factor log K in its dependence on the number of arms. Since the regret admits the universal bound $\mathcal { R } _ { K , T } ( \alpha ) \leq 2 M T$ , we focus on the tuning range

$$
1 \leq \alpha = O \left( { \frac { \sqrt { T / K } } { \log K \log ( K T ) } } \right) ,
$$

over which the tunable regret bound is at most $O ( T )$ . Because $\beta = \Theta ( \log ( K T ) )$ for fixed M and $\sigma ,$ the two guarantees take the simpler form

$$
\mathcal { R } _ { K , T } ( \alpha ) = O \left( \alpha \log K \log ( K T ) \sqrt { K T } \right) , \qquad \mathcal { S } _ { K , T } ( \alpha ) = O \left( \frac { T } { \alpha \sqrt { K } \log ( K T ) } \right) .
$$

As α increases within this range, the regret bound rises while the instability bound falls, tracing the regret– instability frontier. Compared with the $O _ { K } ( T ^ { 3 / 2 } \log T )$ product obtained from the reported UCB-f bounds of Chen and Lu (2026), our ${ \cal O } ( T ^ { 3 / 2 }$ log K) guarantee replaces the explicit log T factor with a log K factor, thereby answering their open K-dependence question up to a logarithmic factor while retaining the optimal dependence on T.

For comparison, the UCB-f analysis (Chen and Lu, 2026) controls random pull counts around a deterministic fluid approximation through armwise UCB-index comparisons and concentration. The sharper K-dependence in Theorem 3.2 instead relies on the monotone lower-envelope indices and deterministic stabilizer gap of SLE-UCB, together with the ofline top-prefix representation and single-reward perturbation analysis developed in Sections 6.1 and 6.3. The Efron–Stein inequality (see Efron and Stein (1981); Boucheron et al. (2013) or Lemma 6.2) then converts these perturbation bounds into variance control. See Sections 5 and 6 for details.

## 4 Clipping bias and concentration

In this section, we characterize the properties of the clipped reward.

Lemma 4.1 (Clipping bias). Let $\theta _ { i } = \mathbb { E } [ Y _ { i , 1 } ]$ be the expectation of the clipped reward for arm i. For every arm i, the clipping bias of mean estimation is upper bounded by

$$
| \theta _ { i } - \mu _ { i } | \leq \sigma ( K T ) ^ { - 6 } .
$$

Proof. Let $X _ { i , 1 }$ be a reward from arm i and $Y _ { i , 1 }$ be the corresponding clipped reward. By Jensen’s inequality, we can bound the clipping bias as follows:

$$
\begin{array} { r } { | \theta _ { i } - \mu _ { i } | = | \mathbb { E } [ Y _ { i , 1 } ] - \mathbb { E } [ X _ { i , 1 } ] | \leq \mathbb { E } [ | Y _ { i , 1 } - X _ { i , 1 } | ] . } \end{array}
$$

As $Y _ { i , 1 }$ and $X _ { i , 1 }$ can difer only when $| X _ { i , 1 } | > B _  \mathrm  ~ $ , we have $| Y _ { i , 1 } - X _ { i , 1 } | = | Y _ { i , 1 } - X _ { i , 1 } | \cdot \mathbf { 1 } \{ | X _ { i , 1 } | > B \}$ . By Assumption 2.1, $| \mu _ { i } | \leq M < B$ . Thus, when the reward exceeds B in absolute value, the diference between the clipped reward and the raw reward can be bounded by

$$
\begin{array} { r l } & { | Y _ { i , 1 } - X _ { i , 1 } | \cdot \mathbf { 1 } \{ | X _ { i , 1 } | > B \} \le | X _ { i , 1 } - \mu _ { i } | \cdot \mathbf { 1 } \{ | X _ { i , 1 } | > B \} } \\ & { \qquad \le | X _ { i , 1 } - \mu _ { i } | \cdot \mathbf { 1 } \{ | X _ { i , 1 } - \mu _ { i } | > 4 \sigma \iota \} . } \end{array}
$$

The second inequality holds because $\lvert X _ { i , 1 } \rvert > B = M + 4 \sigma \iota$ and $| \mu _ { i } | \leq M$ imply $\left| X _ { i , 1 } - \mu _ { i } \right| > 4 \sigma \iota$ . We rewrite the truncated tail expectation using the tail-integral representation:

$$
\begin{array} { r l } & { \left| \theta _ { i } - \mu _ { i } \right| \le { \mathbb E } [ | X _ { i , 1 } - \mu _ { i } | \mathbf { 1 } \{ | X _ { i , 1 } - \mu _ { i } | > 4 \sigma \iota \} ] } \\ & { \qquad = 4 \sigma \iota { \mathbb P } ( | X _ { i , 1 } - \mu _ { i } | > 4 \sigma \iota ) + \displaystyle \int _ { 4 \sigma \iota } ^ { \infty } { \mathbb P } ( | X _ { i , 1 } - \mu _ { i } | > t ) \mathrm { ~ d } t . } \end{array}\tag{4.1}
$$

Substituting the sub-Gaussian tail bound into (4.1) and evaluating the Gaussian tail integral, we have

$$
\begin{array} { l } { \displaystyle | \theta _ { i } - \mu _ { i } | \le 8 \sigma \iota e ^ { - 8 \iota ^ { 2 } } + 2 \int _ { 4 \sigma \iota } ^ { \infty } \exp \left( - \frac { t ^ { 2 } } { 2 \sigma ^ { 2 } } \right) \mathrm { d } t } \\ { \displaystyle \qquad \le 8 \sigma \iota e ^ { - 8 \iota ^ { 2 } } + \frac { 1 } { 2 \sigma \iota } \int _ { 4 \sigma \iota } ^ { \infty } t \exp \left( - \frac { t ^ { 2 } } { 2 \sigma ^ { 2 } } \right) \mathrm { d } t } \\ { \displaystyle \qquad = \left( 8 \iota + \frac 1 { 2 \iota } \right) \sigma e ^ { - 8 \iota ^ { 2 } } \le \sigma e ^ { - 6 \iota ^ { 2 } } = \sigma ( K T ) ^ { - 6 } . } \end{array}
$$

The last inequality holds since $8 \iota + 1 / ( 2 \iota ) \leq e ^ { 2 \iota ^ { 2 } }$ when $\iota = \sqrt { \log ( K T ) } \geq \sqrt { \log 4 } .$

According to Lemma 4.1, the gap between the expectation of the clipped reward and the true mean is negligible, so that the UCB statistic computed from the clipped rewards still reflects the performance of the arms. In the following lemma, we show the concentration of the clipped reward.

Lemma 4.2 (Clipped reward concentration). Define the concentration event G by

$$
\mathcal { G } = \left\{ \left| \overline { { Y } } _ { i } ( n ) - \theta _ { i } \right| \leq \frac { \beta } { 2 \sqrt { n } } \ f o r \ a l l \ i \in [ K ] , \ 1 \leq n \leq T \right\} .\tag{4.2}
$$

The event G satisfies $\mathbb { P } ( \mathcal { G } ^ { c } ) \le 2 ( K T ) ^ { - 7 }$

Proof. Since the clipped rewards lie in $[ - B , B ]$ , Hoefding’s inequality gives, for each fixed i and $n ,$

$$
\mathbb { P } \left( \left| \overline { { Y } } _ { i } ( n ) - \theta _ { i } \right| > \frac { \beta } { 2 \sqrt { n } } \right) \le 2 \exp \left( - \frac { \beta ^ { 2 } } { 8 B ^ { 2 } } \right) .
$$

Applying a union bound over at most KT items (i, n) and using $\beta = 8 B \iota = 8 B \sqrt { \log ( K T ) }$ gives

$$
\mathbb { P } ( \mathcal { G } ^ { c } ) \leq K T \cdot \mathbb { P } \left( \left| \overline { { Y } } _ { i } ( n ) - \theta _ { i } \right| > \frac { \beta } { 2 \sqrt { n } } \right) \leq 2 K T \cdot ( K T ) ^ { - 8 } = 2 ( K T ) ^ { - 7 } .
$$

This finishes the proof of Lemma 4.2.

## 5 Regret Analysis

Proof. Fix an admissible instance ν. We prove the regret bound in (3.6) and the worst-case statement then follows by taking the supremum over ν. Decompose the regret according to the concentration event $\mathcal { G }$ from Lemma 4.2:

$$
\mathcal { R } _ { K , T } ( \nu ) = \underbrace { \mathbb { E } \bigg [ \mathbf { 1 } _ { \mathcal { G } } \sum _ { i = 1 } ^ { K } ( \mu ^ { \star } - \mu _ { i } ) N _ { i } ( T ) \bigg ] } _ { \mathrm { T e r m ~ ( I ) } } + \underbrace { \mathbb { E } \bigg [ \mathbf { 1 } _ { \mathcal { G } ^ { c } } \sum _ { i = 1 } ^ { K } ( \mu ^ { \star } - \mu _ { i } ) N _ { i } ( T ) \bigg ] } _ { \mathrm { T e r m ~ ( I I ) } } .\tag{5.1}
$$

Term (I) in (5.1). Let $\theta ^ { \star } = \operatorname* { m a x } _ { i \in [ K ] } \theta _ { i }$ be the largest clipped mean. By Lemma 4.1, each true mean gap is bounded above by the corresponding clipped mean gap plus twice the clipping-bias bound. Since $\begin{array} { r } { \sum _ { i = 1 } ^ { K } N _ { i } ( T ) = T } \end{array}$ , we have

$$
\mathbb { E } \bigg [ \mathbf { 1 } _ { \mathcal { G } } \sum _ { i = 1 } ^ { K } ( \mu ^ { \star } - \mu _ { i } ) N _ { i } ( T ) \bigg ] \leq \mathbb { E } \bigg [ \mathbf { 1 } _ { \mathcal { G } } \sum _ { i = 1 } ^ { K } ( \theta ^ { \star } - \theta _ { i } ) N _ { i } ( T ) \bigg ] + 2 \sigma T ( K T ) ^ { - 6 } .\tag{5.2}
$$

It remains to bound the first term in (5.2). We divide it into the contribution from the initial pulls and the contribution from the noninitial pulls. For the initialization phase $( t \le K ) , \theta _ { i } \in [ - B , B ]$ for every arm i since the rewards are clipped, so the total is at most 2BK. Since $K \leq T$ and $\beta = 8 B \iota > 2 B$ , we can bound the contribution of the initialization phase as follows:

$$
\sum _ { i = 1 } ^ { K } ( { \theta ^ { \star } } - \theta _ { i } ) \leq 2 B K \leq \beta \sqrt { K T } .\tag{5.3}
$$

Now consider a noninitial time $t ,$ at which SLE-UCB pulls arm $i .$ Let $j$ be the arm satisfying $\theta _ { j } = \theta ^ { \star }$ According to the selection rule of SLE-UCB (3.4), we have

$$
I _ { i } ( N _ { i } ( t - 1 ) ) \geq I _ { j } ( N _ { j } ( t - 1 ) ) .\tag{5.4}
$$

On ${ \mathcal { G } } ,$ we first upper bound the index of the selected arm i. Since $U _ { i } ( n )$ is the minimum over all prefixes up to n, evaluating it at the full prefix gives

$$
\begin{array} { r l r } {  { I _ { i } ( N _ { i } ( t - 1 ) ) = U _ { i } ( N _ { i } ( t - 1 ) ) + \lambda \log \bigg ( \frac { T } { N _ { i } ( t - 1 ) } \bigg ) } } \\ & { } & { \leq \overline { { Y } } _ { i } ( N _ { i } ( t - 1 ) ) + \frac { \beta } { \sqrt { N _ { i } ( t - 1 ) } } + \lambda \log \bigg ( \frac { T } { N _ { i } ( t - 1 ) } \bigg ) } \\ & { } & { \leq \theta _ { i } + \frac { 3 \beta } { 2 \sqrt { N _ { i } ( t - 1 ) } } + \lambda \log \bigg ( \frac { T } { N _ { i } ( t - 1 ) } \bigg ) . } \end{array}\tag{5.5}
$$

The first inequality follows from the definition of the lower-envelope UCB statistic (3.2), and the second inequality follows from the definition of event G. Next, for the arm $j$ satisfying $\theta _ { j } = \theta ^ { \star }$ , let

$$
s _ { j } \in \mathop { \arg \operatorname* { m i n } } _ { 1 \leq s \leq N _ { j } ( t - 1 ) } \left\{ { \overline { { Y } } } _ { j } ( s ) + { \frac { \beta } { \sqrt { s } } } \right\} .
$$

Then the index of arm $j$ satisfies

$$
\begin{array} { r l } & { I _ { j } ( N _ { j } ( t - 1 ) ) = \overline { { Y } } _ { j } ( s _ { j } ) + \frac { \beta } { \sqrt { s _ { j } } } + \lambda \log \bigg ( \frac { T } { N _ { j } ( t - 1 ) } \bigg ) } \\ & { \qquad \geq \theta _ { j } + \frac { \beta } { 2 \sqrt { s _ { j } } } + \lambda \log \bigg ( \frac { T } { N _ { j } ( t - 1 ) } \bigg ) \geq \theta ^ { \star } . } \end{array}\tag{5.6}
$$

The first inequality follows from the event ${ \mathcal { G } } ,$ and the last inequality uses $\theta _ { j } = \theta ^ { \star }$ and $N _ { j } ( t - 1 ) \leq T$

Combining (5.4), (5.5) and (5.6), for every noninitial round $t > K$ with $A _ { t } = i .$

$$
\theta ^ { \star } - \theta _ { i } \leq \frac { 3 \beta } { 2 \sqrt { N _ { i } ( t - 1 ) } } + \lambda \log \bigg ( \frac { T } { N _ { i } ( t - 1 ) } \bigg ) .\tag{5.7}
$$

Summing (5.7) over all noninitial pulls, we upper bound the noninitial contribution to Term (I) by

$$
\sum _ { i = 1 } ^ { K } ( \theta ^ { \star } - \theta _ { i } ) ( N _ { i } ( T ) - 1 ) \leq \frac { 3 \beta } { 2 } \sum _ { i = 1 } ^ { K } \sqrt { N _ { i } ( T ) - 1 } + \lambda \sum _ { i : N _ { i } ( T ) > 1 } \left( N _ { i } ( T ) - 1 \right) \log \left( \frac { T } { N _ { i } ( T ) - 1 } \right)\tag{5.8}
$$

$$
\leq \frac { 3 \beta } { 2 } \sqrt { K T } + \lambda ( T - K ) \log \bigg ( \frac { K T } { T - K } \bigg ) .\tag{5.9}
$$

In (5.8), we take t in (5.7) to be the last time arm i is pulled, so that the previous pull count equals $N _ { i } ( T ) - 1$ The second inequality (5.9) uses $\begin{array} { r } { \sum _ { i = 1 } ^ { K } N _ { i } ( T ) = T } \end{array}$ . The first term follows from the Cauchy–Schwarz inequality and the second term follows from the concavity of $x \mapsto x \log ( T / x )$

Combining the initial phase contribution (5.3) and the noninitial phase contribution (5.9), substituting $\lambda = \alpha \beta \sqrt { K / T }$ , we have

$$
\begin{array} { l } { \displaystyle \sum _ { t = 1 } ^ { T } ( \theta ^ { \star } - \theta _ { A _ { t } } ) \leq \beta \sqrt { K T } + \frac { 3 \beta } { 2 } \sqrt { K T } + \alpha \beta \sqrt { K T } \cdot \frac { T - K } { T } \log \left( \frac { K T } { T - K } \right) } \\ { \leq \bigg ( 3 + \alpha \log K + \frac { \alpha ( T - K ) } { T } \log \left( \frac { T } { T - K } \right) \bigg ) \beta \sqrt { K T } } \\ { \leq \big ( 3 + \alpha \log ( 2 K ) \big ) \beta \sqrt { K T } . } \end{array}\tag{5.10}
$$

The last line holds because $( T - K ) / T < 1$ and $x \log ( 1 / x ) < 1 / e < \log 2$ when $x < 1$ . Substituting (5.10) into (5.2), Term (I) is bounded by

$$
\mathbb { E } \bigg [ \mathbf { 1 } _ { \mathcal { G } } \sum _ { i = 1 } ^ { K } ( \mu ^ { \star } - \mu _ { i } ) N _ { i } ( T ) \bigg ] \leq \big ( 3 + \alpha \log ( 2 K ) \big ) \beta \sqrt { K T } + 2 \sigma T ( K T ) ^ { - 6 } .\tag{5.11}
$$

Term (II) in (5.1). On $\mathcal G ^ { c }$ , each mean gap is at most 2M, since the true means are bounded in $[ - M , M ]$ Thus, the regret is at most 2MT. According to Lemma 4.2, the probability of $\mathcal G ^ { c }$ is at most $2 ( K T ) ^ { - 7 } ,$ so Term (II) can be bounded by

$$
\mathbb { E } \bigg [ \mathbf { 1 } _ { \mathcal { G } ^ { c } } \sum _ { i = 1 } ^ { K } ( \mu ^ { \star } - \mu _ { i } ) N _ { i } ( T ) \bigg ] \leq 2 M T \cdot 2 ( K T ) ^ { - 7 } = 4 M T ( K T ) ^ { - 7 } .\tag{5.12}
$$

Combining the bounds. Combining (5.11) and (5.12), we obtain

$$
\begin{array} { r l } & { \mathcal { R } _ { K , T } ( \nu ) \leq \big ( 3 + \alpha \log ( 2 K ) \big ) \beta \sqrt { K T } + 2 \sigma T ( K T ) ^ { - 6 } + 4 M T ( K T ) ^ { - 7 } } \\ & { \qquad \leq \big ( 5 + \alpha \log ( 2 K ) \big ) \beta \sqrt { K T } . } \end{array}
$$

The last inequality uses basic algebraic manipulations. By the definitions $\beta \ : = \ : 8 B \iota$ and $B = M + 4 \sigma \iota$ ， together with $\iota \geq 1$ , we have $\sigma \le \beta$ and $M \leq \beta$ . Thus, for $K , T \geq 2$ , we have $2 \sigma T ( K T ) ^ { - 6 } \leq \beta \sqrt       { K T }$ and $4 \bar { M } T ( K T ) ^ { - 7 } \leq \beta \sqrt { K T }$ . Taking the supremum over all admissible instances proves (3.6). □

## 6 Instability Analysis

This section proves the instability bound in (3.6). Fix an admissible instance ν and an arm $i \in [ K ]$ . Recall from Section 3.2 that $Y = ( Y _ { a , m } ) _ { a \in [ K ] , m \in [ T ] }$ denotes the full array of independent clipped potential rewards. We use $\mathcal G ( Y )$ to denote the event that Y satisfies the concentration condition in (4.2). Since $N _ { i } ( T )$ is

![](images/2afdbbc10d959509d3a8eccc2d2e917e0d68bfc98c477480e5b85a5aa0118516.jpg)

![](images/672177ed3492f6fbaf0ecc5048c8663e728cfb6be72f231970f2642c95174dd8.jpg)  
Figure 2: Ofline and online selection $( K = 3 , T = 8 )$ . An item $( i , n )$ represents the n-th pull of arm i, has value $V _ { i , n } = I _ { i } ( n - 1 )$ , and is post-initial when $n \geq 2$ . Consider a permutation $( \omega ( 1 ) , \omega ( 2 ) , \omega ( 3 ) ) = ( 2 , 3 , 1 )$ Assume $I _ { 2 } ( 1 ) > I _ { 1 } ( 1 ) > I _ { 3 } ( 1 ) = I _ { 1 } ( 2 ) > I _ { 2 } ( 2 )$ and all other indices are smaller. For ofline selection, after sorting $V _ { i , n } = I _ { i } ( n - 1 )$ by decreasing value and resolving the tie using $\omega ,$ the first five items in the ofline postinitial order are $( 2 , 2 ) , ( 1 , 2 ) , ( 3 , 2 ) , ( 1 , 3 ) , ( 2 , 3 )$ . For online selection, the first five choices after initialization are $2  1  3  1  2$ , which corresponds to the same items $( 2 , 2 ) , ( 1 , 2 ) , ( 3 , 2 ) , ( 1 , 3 ) , ( 2 , 3 )$ as in the ofline selection.

determined by the clipped reward array Y and the random tie-breaking permutation ω, we write the final count as $N _ { i } ( Y , \omega , T )$ . Applying the law of total variance gives

$$
\begin{array} { r } { \mathrm { V a r } ( N _ { i } ( Y , \omega , T ) ) = \underbrace { \mathrm { V a r } _ { \omega } \big ( \mathbb { E } _ { Y } [ N _ { i } ( Y , \omega , T ) \mid \omega ] \big ) } _ { \mathrm { T e r m ~ } ( \star ) } + \underbrace { \mathbb { E } _ { \omega } \big [ \mathrm { V a r } _ { Y } ( N _ { i } ( Y , \omega , T ) \mid \omega ) \big ] } _ { \mathrm { T e r m ~ } ( \star \star ) } . } \end{array}\tag{6.1}
$$

Here and throughout this section, all randomness is induced only by the clipped reward array Y and the random tie-breaking permutation ω. The subscripts on expectations and variances specify which of these two sources of randomness is being averaged over: $\mathbb { E } _ { \omega }$ and $\mathrm { V a r } _ { \omega }$ are taken only with respect to $\omega ,$ while $\mathbb { E } _ { Y }$ and Var<sub>Y</sub> are taken only with respect to $Y ;$ the other source of randomness is held fixed or conditioned on as displayed. We will bound Term (⋆) in Section 6.2 and Term (⋆⋆) in Section 6.3, respectively.

The key device is an ofline top-prefix representation of the SLE-UCB selection rule. Conditional on the permutation ω, the final count $N _ { i } ( Y , \omega , T )$ is a deterministic function of the reward array $Y$ , so the Efron–Stein inequality reduces its conditional variance to single-coordinate perturbations of Y. A direct online analysis is dificult because a single perturbation may alter all subsequent decisions. The ofline representation removes this path dependence, thereby enabling the required perturbation analysis. We begin by establishing the online–ofline equivalence.

## 6.1 Online–ofline equivalence

Conditional on a reward array Y and a permutation ω, all stabilized indices are deterministic. We introduce the item set $\mathcal { T } _ { T } = \{ ( i , n ) : i \in [ K ] , 1 \leq n \leq T \}$ and define two selection rules on it, where $( i , n )$ represents the n-th pull of arm i.

For each item $( i , n )$ , define its value as $V _ { i , n } ~ = ~ I _ { i } ( n - 1 )$ , where $I _ { i } ( 0 ) = \infty$ . Sort all items in $\mathcal { I } _ { T }$ by decreasing value $V _ { i , n }$ , resolving ties by the smaller priority rank $\omega ( i )$

For a fixed number of rounds $T \geq K$ , compare the following two selection rules:

• Ofline top-prefix rule. Select top T items in the sorted list. These consist of the K initial items $\{ ( i , 1 ) \} _ { i \in [ K ] }$ and the top $T - K$ post-initial items.

• Online greedy rule (SLE-UCB selection rule). Select $\{ ( i , 1 ) \} _ { i \in [ K ] }$ during the first K rounds. At each later round $t > K$ , choose an arm with the largest current value $I _ { i } ( \dot { N _ { i } } ( \dot { t } - 1 ) )$ and select the corresponding next item.

Figure 2 illustrates the selection processes for both rules. With the following lemma, we show that the two rules select the same items.

Lemma 6.1 (Online–ofline equivalence). For every reward array Y and permutation $\omega ,$ the item set selected by the online greedy rule up to time $T \geq K$ is identical to the item set selected by the ofline top-prefix rule.

Proof. Both rules select $\{ ( i , 1 ) \} _ { i \in [ K ] }$ , so it remains to compare the remaining $T - K$ post-initial selections. Let $S _ { r } ^ { \mathrm { { o f f } } }$ be the first r post-initial items in the ofline sorted list, and let $S _ { r } ^ { \mathrm { o n } }$ be the first r post-initial items selected by the online greedy rule. We prove by induction that

$$
S _ { r } ^ { \mathrm { o n } } = S _ { r } ^ { \mathrm { o f f } } , \qquad 0 \le r \le T - K .
$$

The claim is immediate for $r = 0$ . Suppose it holds for some $r < T - K$ , and let $( a , n )$ be the next item in the ofline sorted list. By the induction hypothesis, the first r post-initial items selected by the two rules form the same set. Hence, for each arm i, the ofline rule has selected all items $( i , j )$ with $j \le N _ { i } ( K + r )$ and its highest-ranked unselected item from that arm is $( i , N _ { i } ( K + r ) + 1 )$ because $V _ { i , j } = I _ { i } ( j - 1 )$ is strictly decreasing in $j .$ . Consequently, $( a , n )$ is the item with the largest value among the items $( i , N _ { i } ( K + r ) + 1 )$ $i \in [ K ]$ , with ties resolved by ω. At round $K + r + 1$ , the online greedy rule compares exactly these same items and uses the same tie-breaking rule, so it also selects $( a , n )$ . This proves the induction step. □

Directly under the online rule, changing the tie-breaking permutation or replacing a single reward may alter the next selected arm and thereby propagate through the entire subsequent decision trajectory, making the sensitivity of the final pull counts dificult to track. With the online–ofline equivalence established, we use the ofline top-prefix representation in both parts of the variance analysis. In Section 6.2, it isolates the efect of random tie-breaking and shows that changing the priority permutation can alter the final count of any arm by at most one. In Section 6.3, it converts a single-coordinate reward perturbation into a movement of the ofline selection boundary, thereby avoiding a recursive analysis of subsequent online decisions.

## 6.2 Bound of Term (⋆) in (6.1)

Proof. Fix a clipped reward array Y, the target arm $i ,$ and two permutations ω and $\omega ^ { \prime }$ . By Lemma 6.1, under either permutation the algorithm selects the initial items $\{ ( j , 1 ) \} _ { j \in [ K ] }$ and then selects the top $T - K$ post-initial items according to the sorted item values, with ties resolved by the permutation. Let c be the value of the last selected post-initial item.

Items with value strictly larger than c are selected under both permutations, and items with value strictly smaller than c are selected under neither permutation. Thus, the permutation can afect only items whose value is exactly $c .$ For the target arm $i ,$ according to (3.5), the sequence $\{ I _ { i } ( n ) \} _ { n = 1 } ^ { T }$ is strictly decreasing. Hence arm i has at most one post-initial item with value c, and changing the permutation can change the final count of arm i by at most one: $| N _ { i } ( Y , \omega , T ) - N _ { i } ( Y , \omega ^ { \prime } , T ) | \leq 1$ . With ω and $\omega ^ { \prime }$ fixed, taking expectations over $Y$ gives

$$
\begin{array} { r } { | \mathbb { E } _ { Y } [ N _ { i } ( Y , \omega , T ) \mid \omega ] - \mathbb { E } _ { Y } [ N _ { i } ( Y , \omega ^ { \prime } , T ) \mid \omega ^ { \prime } ] | \leq \mathbb { E } _ { Y } [ | N _ { i } ( Y , \omega , T ) - N _ { i } ( Y , \omega ^ { \prime } , T ) | ] \leq 1 . } \end{array}
$$

Therefore the conditional expectations lie in an interval of length at most one as ω varies. Since the target arm i was arbitrary, for every $i \in [ K ]$ ，

$$
\operatorname { V a r } _ { \omega } ( \mathbb { E } _ { Y } [ N _ { i } ( Y , \omega , T ) \mid \omega ] ) \leq 1 .\tag{6.2}
$$

This provides an upper bound for Term (⋆).

## 6.3 Bound of Term (⋆⋆) in (6.1)

We fix the permutation $\omega$ and control the sensitivity of the final count of the target arm i to a singlecoordinate perturbation of the reward array. The proof has three steps: first, we bound the change in the stabilized indices caused by replacing one clipped reward; second, we translate this index perturbation into a pull-count perturbation; third, we sum the squared influences through the Efron–Stein inequality.

Proof. Consider clipped reward arrays $Y , Y ^ { \prime } \in \mathbb { R } ^ { K \times T }$ difering only in the m-th clipped reward of arm $a ,$ where $Y _ { a , m } ^ { \prime }$ is an independent copy of $Y _ { a , m } .$ . Let $\overline { { Y } } _ { i } ^ { \prime } ( n ) , U _ { i } ^ { \prime } ( n )$ , and $I _ { i } ^ { \prime } ( n )$ denote the corresponding averages, lower-envelope UCB statistics, and stabilized indices computed under ${ \dot { Y } } ^ { \prime }$ . For simplicity, since ω and $T$ are fixed in this subsection, we write $N _ { j } ( Y ) = N _ { j } ( Y , \omega , T )$ for every arm $j \in [ K ]$

Stabilized-index perturbation. We first bound how a single-coordinate replacement changes the stabi lized indices. For every arm $j \neq a ,$ , the reward array is unchanged, so $I _ { j } ^ { \prime } ( n ) = I _ { j } ( n )$ for all $n \leq T$ . For arm $^ { a , }$ the index is unchanged for $n < m$ , because the replaced reward has not entered the prefix average. For $n \geq m$ , we define the index attaining the minimum in $U _ { a } ( n )$ under $Y$ as

$$
s ( n ) \in \mathop { \arg \operatorname* { m i n } } _ { 1 \leq q \leq n } \left\{ \overline { { Y } } _ { a } ( q ) + \frac { \beta } { \sqrt { q } } \right\} .
$$

Using the definition of the lower-envelope UCB statistic, we have

$$
U _ { a } ^ { \prime } ( n ) \leq \overline { { Y } } _ { a } ^ { \prime } ( s ( n ) ) + \frac { \beta } { \sqrt { s ( n ) } } \leq \overline { { Y } } _ { a } ( s ( n ) ) + \frac { | Y _ { a , m } ^ { \prime } - Y _ { a , m } | } { s ( n ) } + \frac { \beta } { \sqrt { s ( n ) } } = U _ { a } ( n ) + \frac { | Y _ { a , m } ^ { \prime } - Y _ { a , m } | } { s ( n ) } .
$$

We next show that the minimizer $s ( n )$ cannot be much smaller than n on the concentration event $\mathcal G ( Y )$ Since $s ( n )$ is a minimizer and $\mathcal G ( Y )$ holds, we have

$$
\theta _ { a } + \frac { \beta } { 2 \sqrt { s ( n ) } } \leq \overline { { Y } } _ { a } ( s ( n ) ) + \frac { \beta } { \sqrt { s ( n ) } } \leq \overline { { Y } } _ { a } ( n ) + \frac { \beta } { \sqrt { n } } \leq \theta _ { a } + \frac { 3 \beta } { 2 \sqrt { n } } .
$$

The first and last inequalities use the event $\mathcal G ( Y )$ , and the chain implies $s ( n ) \geq n / 9$

Pull-count perturbation bound on $\mathcal { G } ( Y ) \cap \mathcal { G } ( Y ^ { \prime } )$ . Assume that $\mathcal G ( Y )$ and $\mathcal { G } ( Y ^ { \prime } )$ both hold. Applying $s ( n ) \geq n / 9$ to $Y$ , and then interchanging the roles of the two arrays, gives

$$
I _ { a } ^ { \prime } ( n ) = I _ { a } ( n ) \quad { \mathrm { i f ~ } } n < m , \qquad | I _ { a } ^ { \prime } ( n ) - I _ { a } ( n ) | \leq { \frac { 9 | Y _ { a , m } ^ { \prime } - Y _ { a , m } | } { n } } \quad { \mathrm { i f ~ } } n \geq m .\tag{6.3}
$$

We now convert the index perturbation in (6.3) into a perturbation bound for the final pull counts. If $m > N _ { a } ( Y )$ , then $I _ { i } ^ { \prime } ( n ) = I _ { j } ( n )$ for all $j \in [ K ]$ and $n \leq N _ { j } ( Y )$ , hence the selected item set is unchanged. Thus, for every $j \in [ K ]$ 2

$$
N _ { j } ( Y ) = N _ { j } ( Y ^ { \prime } ) .\tag{6.4}
$$

Similarly, if $m > N _ { a } ( Y ^ { \prime } )$ , then $I _ { j } ( n ) = I _ { j } ^ { \prime } ( n )$ for all $j \in [ K ]$ and $n \leq N _ { j } ( Y ^ { \prime } )$ , which implies $N _ { j } ( Y ) = N _ { j } ( Y ^ { \prime } )$ for every $j \in [ K ]$

As for the case $m \leq \operatorname* { m i n } \{ N _ { a } ( Y ) , N _ { a } ( Y ^ { \prime } ) \}$ , without loss of generality, we assume that $N _ { a } ( Y ) < N _ { a } ( Y ^ { \prime } )$ . By Lemma 6.1, both runs are equivalent to top-prefix selections under the same permutation. Thus, $( a , N _ { a } ( Y ) + 1 )$ is not selected under Y, while $\left( a , N _ { a } ( Y ^ { \prime } ) \right)$ is selected under $Y ^ { \prime }$ . Since the total number of selected items is fixed, there exists an unchanged arm j such that the item $( j , \ell )$ is selected under Y and not selected under $Y ^ { \prime }$ . Hence

$$
I _ { a } ( N _ { a } ( Y ) ) \leq I _ { j } ( \ell - 1 ) = I _ { j } ^ { \prime } ( \ell - 1 ) \leq I _ { a } ^ { \prime } ( N _ { a } ( Y ^ { \prime } ) - 1 ) .\tag{6.5}
$$

Subtracting $I _ { a } ( N _ { a } ( Y ^ { \prime } ) - 1 )$ from both sides of (6.5), applying (6.3), and using the logarithmic gap of stabilized indices (3.5), we obtain

$$
\begin{array} { r l } & { \lambda \log \left( \frac { N _ { a } ( Y ^ { \prime } ) - 1 } { N _ { a } ( Y ) } \right) \leq I _ { a } ( N _ { a } ( Y ) ) - I _ { a } ( N _ { a } ( Y ^ { \prime } ) - 1 ) } \\ & { \qquad \leq I _ { a } ^ { \prime } ( N _ { a } ( Y ^ { \prime } ) - 1 ) - I _ { a } ( N _ { a } ( Y ^ { \prime } ) - 1 ) } \\ & { \qquad \leq \frac { 9 | Y _ { a , m } ^ { \prime } - Y _ { a , m } | } { N _ { a } ( Y ^ { \prime } ) - 1 } . } \end{array}
$$

Using log $( 1 + x ) \geq x / ( 1 + x )$ for $x \geq 0$ , we have

$$
\lambda \cdot \frac { N _ { a } ( Y ^ { \prime } ) - 1 - N _ { a } ( Y ) } { N _ { a } ( Y ^ { \prime } ) - 1 } \leq \frac { 9 | Y _ { a , m } ^ { \prime } - Y _ { a , m } | } { N _ { a } ( Y ^ { \prime } ) - 1 } .
$$

Exchanging the roles of Y and $Y ^ { \prime }$ gives the same bound when $N _ { a } ( Y ) > N _ { a } ( Y ^ { \prime } )$ . Therefore, when $\mathcal G ( Y )$ and $\mathcal { G } ( Y ^ { \prime } )$ both hold, we have $| N _ { a } ( Y ) - N _ { a } ( Y ^ { \prime } ) | \leq ( 1 + 9 | Y _ { a , m } ^ { \prime } - Y _ { a , m } | / \lambda ) \mathbf { 1 } _ { \{ m \leq \operatorname* { m i n } \{ N _ { a } ( Y ) , N _ { a } ( Y ^ { \prime } ) \} \} }$

For an unchanged arm $j \neq a ,$ remove arm a and sort all unchanged-arm items in their common order. Conditional on the number of selected items from arm $^ { a , }$ the selected items from the unchanged arms form a prefix of this remaining list. Hence the count change of every unchanged arm is bounded by the count change of arm a. Consequently, when $\mathcal G ( Y )$ and $\mathcal { G } ( Y ^ { \prime } )$ both hold, for every $j \in [ K ]$ ],

$$
| N _ { j } ( Y ) - N _ { j } ( Y ^ { \prime } ) | \leq \left( 1 + \frac { 9 | Y _ { a , m } ^ { \prime } - Y _ { a , m } | } { \lambda } \right) \mathbf { 1 } _ { \{ m \leq \operatorname* { m i n } \{ N _ { a } ( Y ) , N _ { a } ( Y ^ { \prime } ) \} \} } .\tag{6.6}
$$

Expectation of the pull-count perturbation. Using (6.4), (6.6), and the trivial bound $| N _ { i } ( Y ) -$ $N _ { i } ( Y ^ { \prime } ) | \leq T$ when either $\mathcal G ( Y )$ or $\mathcal { G } ( Y ^ { \prime } )$ fails, we obtain

$$
\begin{array} { r l } & { \mathbb { E } _ { Y , Y ^ { \prime } } \left[ ( N _ { i } ( Y ) - N _ { i } ( Y ^ { \prime } ) ) ^ { 2 } \right] } \\ & { \qquad \leq \mathbb { E } _ { Y , Y ^ { \prime } } \left[ ( N _ { i } ( Y ) - N _ { i } ( Y ^ { \prime } ) ) ^ { 2 } { \mathbf 1 } _ { \mathcal { G } ( Y ) \cap \mathcal { G } ( Y ^ { \prime } ) } \right] + \mathbb { E } _ { Y , Y ^ { \prime } } \left[ ( N _ { i } ( Y ) - N _ { i } ( Y ^ { \prime } ) ) ^ { 2 } { \mathbf 1 } _ { \mathcal { G } ( Y ) ^ { c } \cup \mathcal { G } ( Y ^ { \prime } ) } \right] } \end{array}
$$

$$
\leq \mathbb { E } _ { Y , Y ^ { \prime } } \Bigg [ \Bigg ( 1 + \frac { 9 | Y _ { a , m } ^ { \prime } - Y _ { a , m } | } { \lambda } \Bigg ) ^ { 2 } \Bigg ] \mathbb { P } ( m \leq N _ { a } ( Y ) ) + 4 T ^ { 2 } ( K T ) ^ { - 7 }\tag{6.7}
$$

$$
\leq \bigg ( 1 + \frac { 9 \sqrt { 2 } \sigma } { \lambda } \bigg ) ^ { 2 } \mathbb { P } ( m \leq N _ { a } ( Y ) ) + 4 T ^ { 2 } ( K T ) ^ { - 7 } .\tag{6.8}
$$

The first term in (6.7) uses the independence of ${ \bf 1 } _ { \{ m \leq \operatorname* { m i n } \{ N _ { a } ( Y ) , N _ { a } ( Y ^ { \prime } ) \} \} }$ and $( Y _ { a , m } , Y _ { a , m } ^ { \prime } )$ : under $Y _ { ; }$ , whether the m-th pull of arm a is selected is determined before $\dot { Y } _ { a , m }$ is observed, and similarly for $Y ^ { \prime }$ . The second term uses the union bound and Lemma 4.2 to bound the probability of $\mathcal G ( Y ) ^ { c } \cup \mathcal G ( Y ^ { \prime } ) ^ { c }$ . The last inequality uses Minkowski’s inequality and bounds $\mathbb { E } _ { Y , Y ^ { \prime } } [ ( Y _ { a , m } - Y _ { a , m } ^ { \prime } ) ^ { 2 } ]$ by twice the variance of raw rewards.

Conditional variance bound. We now apply the Efron–Stein inequality in Efron and Stein (1981), in the replacement form stated in Boucheron et al. (2013, Theorem 3.1) to the reward array and bound the conditional variance using the preceding pull-count estimate.

Lemma 6.2 (Efron–Stein inequality). Let $X _ { 1 } , \ldots , X _ { n }$ be independent random variables, and let $X _ { \ell } ^ { \prime }$ be an independent copy of $X _ { \ell } ~ f o r$ all $\ell \in \ \lceil n \rceil$ . Let $F = f ( X _ { 1 } , \ldots , X _ { n } ) $ be square-integrable. For $\ell \in [ n ]$ , define $F ^ { ( \ell ) } = f ( X _ { 1 } , \ldots , X _ { \ell - 1 } , X _ { \ell } ^ { \prime } , X _ { \ell + 1 } , \ldots , { \bar { X } } _ { n } )$ . Then we have

$$
\operatorname { V a r } ( F ) \leq { \frac { 1 } { 2 } } \sum _ { \ell = 1 } ^ { n } \mathbb { E } \left[ ( F - F ^ { ( \ell ) } ) ^ { 2 } \right] .
$$

For a coordinate $( a , m )$ satisfying $a \in [ K ]$ and $m \in [ T ]$ , let $Y ^ { ( a , m ) }$ be the array obtained by replacing $Y _ { a , m }$ by an independent copy $Y _ { a , m } ^ { \prime }$ and leaving every other coordinate unchanged. Using Lemma 6.2 and (6.8), with $\omega$ fixed, the conditional variance of the fixed target count can be bounded by

(6.9)

$$
\begin{array} { l } { \displaystyle \mathrm { V a r } _ { Y } ( N _ { i } ( Y , \omega , T ) \mid \omega ) \le \frac { 1 } { 2 } \displaystyle \sum _ { a = 1 } ^ { K } \sum _ { m = 1 } ^ { T } \mathbb { E } _ { Y , Y ^ { ( a , m ) } } \left[ \left( N _ { i } ( Y , \omega , T ) - N _ { i } ( Y ^ { ( a , m ) } , \omega , T ) \right) ^ { 2 } \right] } \\ { \displaystyle \le \frac { ( 1 + 9 \sqrt { 2 } \sigma / \lambda ) ^ { 2 } } { 2 } \sum _ { a = 1 } ^ { K } \sum _ { m = 1 } ^ { T } \mathbb { P } _ { Y } ( m \le N _ { a } ( Y , \omega , T ) ) + 2 T ^ { 2 } ( K T ) ^ { - 6 } } \\ { \displaystyle = \frac { ( 1 + 9 \sqrt { 2 } \sigma / \lambda ) ^ { 2 } } { 2 } \sum _ { a = 1 } ^ { K } \mathbb { E } _ { Y } [ N _ { a } ( Y , \omega , T ) ] + 2 T ^ { 2 } ( K T ) ^ { - 6 } } \\ { \displaystyle = \frac { ( 1 + 9 \sqrt { 2 } \sigma / \lambda ) ^ { 2 } T } { 2 } + 2 T ^ { 2 } ( K T ) ^ { - 6 } \le ( 1 + 9 \sqrt { 2 } \sigma / \lambda ) ^ { 2 } T } \end{array}\tag{6.10}
$$

(6.11)

The first inequality (6.9) applies the Efron–Stein inequality over the replacement coordinates $( a , m )$ . The second inequality (6.10) uses the single-coordinate perturbation bound in (6.8). The last line (6.11) uses $\begin{array} { r } { \sum _ { a = 1 } ^ { K } \mathbb { E } _ { Y } [ N _ { a } ( Y , \omega , T ) ] = T } \end{array}$ . Since $K , T \geq 2$ , we have $2 T ^ { 2 } ( K T ) ^ { - 6 } \leq T / 2 \leq ( 1 + 9 { \sqrt { 2 } } \sigma / \lambda ) ^ { 2 } T / 2$ , so the second term in (6.10) is absorbed into the leading term.

Taking expectation over ω, we can bound Term (⋆⋆) by

$$
\mathbb { E } _ { \omega } [ \mathrm { V a r } _ { Y } ( N _ { i } ( Y , \omega , T ) \mid \omega ) ] \leq ( 1 + 9 \sqrt { 2 } \sigma / \lambda ) ^ { 2 } T .\tag{6.12}
$$

This proves the desired bound for Term (⋆⋆).

## 6.4 Combining the bounds

Combining the variance decomposition (6.1), the bound for Term (⋆) in (6.2), and the bound for Term (⋆⋆) in (6.12), we obtain that for every arm $i \in [ K ]$

$$
\operatorname { V a r } ( N _ { i } ( Y , \omega , T ) ) \leq 1 + ( 1 + 9 \sqrt { 2 } \sigma / \lambda ) ^ { 2 } T \leq 4 ( 1 + 9 \sqrt { 2 } \sigma / \lambda ) ^ { 2 } T .
$$

The last relaxation uses $T \geq 1$ . Substituting $\lambda = \alpha \beta \sqrt { K / T }$ , we have

$$
\begin{array} { l } { \displaystyle { \mathcal { S } _ { K , T } ( \alpha ) \leq 2 \sqrt { T } + \frac { 1 8 \sqrt { 2 } \sigma T } { \alpha \beta \sqrt { K } } . } } \end{array}
$$

Because this bound is uniform over admissible instances ν and arms i, taking the maximum over arms and then the supremum over instances proves the instability bound in (3.6).

## 7 Proof of the K-armed lower bound

Remark 7.1 (Fixed-K reduction). We establish the fixed-K extension in (1.1) through the following padding reduction. Pad the instance with $K - 2$ independent $\mathcal { N } ( - M , \sigma ^ { 2 } )$ dummy arms. A two-armed algorithm B simulates a K-armed algorithm A exactly on the first two arms. On dummy-arm requests, it alternates between the real arms from a uniformly random starting arm, discards the reward, and supplies an independent dummy reward. This gives $\mathcal { R } _ { 2 , T } ^ { B } \leq \dot { \mathcal { R } } _ { K , T } ^ { A }$ and $S _ { 2 , T } ^ { B } \stackrel {  } { \leq } S _ { K , T } ^ { A } + 1 / 2$ while preserving the required conditions. Since $\mathcal { R } _ { 2 , T } ^ { B } \leq \mathcal { R } _ { K , T } ^ { A } = o ( T )$ , their theorem gives lim in $\mathrm { f } _ { T  \infty } \mathcal { R } _ { 2 , T } ^ { B } S _ { 2 , T } ^ { B } / T ^ { 3 / 2 } > C _ { 0 }$ for some constant $C _ { 0 } > 0$ Together with the two inequalities above, this yields the fixed-K conclusion.

Proof of Theorem 3.1. Our proof largely follows the approach of Chen and Lu (2026). Fix $K \ge 2 , T > K$ Let A be an algorithm whose worst-case regret satisfies $\mathcal { R } _ { K , T } \ \leq \ M T / 8$ . Consider the all-zero Gaussian instance $\nu ( 0 ) = ( \mathcal { N } ( 0 , \sigma ^ { 2 } ) ) _ { i = 1 } ^ { K } ,$ , where the reward distributions of all arms are ${ \mathcal { N } } ( 0 , \sigma ^ { 2 } )$ . Choose an arm attaining the smallest expected pull count under this instance and denote it by arm $j .$ . We consider the following family of instances, in which arm $j$ is the unique optimal arm whenever $\Delta > 0 \colon$

$$
\begin{array} { r } { ( \nu ( \Delta ) ) _ { i } = \left\{ \begin{array} { l l } { N ( 0 , \sigma ^ { 2 } ) , } & { i = j , } \\ { N ( - \Delta , \sigma ^ { 2 } ) , } & { i \ne j . } \end{array} \right. } \end{array}
$$

We aggregate all suboptimal pulls. Suppressing the dependence on A, let $\mathbb { P } _ { \Delta } , \ \mathbb { E } _ { \Delta }$ , and $\mathrm { V a r } _ { \Delta }$ denote probability, expectation, and variance under the interaction with $\nu ( \Delta )$ , and define

$$
N _ { - } ( T ) : = \sum _ { i \neq j } N _ { i } ( T ) = T - N _ { j } ( T ) , \qquad G ( \Delta ) : = \mathbb { E } _ { \Delta } [ N _ { - } ( T ) ] .\tag{7.1}
$$

The equal-gap construction gives

$$
\mathcal { R } _ { K , T } ( \nu ( \Delta ) ) = \Delta \cdot G ( \Delta ) , \qquad G ( 0 ) = T - \mathbb { E } _ { 0 } [ N _ { j } ( T ) ] \geq \frac { K - 1 } { K } T ,\tag{7.2}
$$

The inequality for $G ( 0 )$ follows because arm $j$ has the smallest expected pull count at $\Delta = 0$ . Define a candidate gap by

$$
\Delta _ { \star } : = \frac { 8 \mathcal { R } _ { K , T } } { T } .
$$

The regret requirement $\mathcal { R } _ { K , T } \ \leq \ M T / 8$ gives $\Delta _ { \star } \leq M$ , and hence $\nu ( \Delta _ { \star } )$ is admissible. We connect the all-zero instance to $\nu ( \Delta _ { \star } )$ through a sequence of nearby instances. The increments are chosen so that the KL

divergence between the interaction histories at consecutive gaps is at most $1 / 2$ . Set $\Delta _ { 0 } = 0$ and, recursively, define

$$
\Delta _ { \ell + 1 } : = \operatorname* { m i n } \left\{ \Delta _ { \star } , \Delta _ { \ell } + \frac { \sigma } { \sqrt { G ( \Delta _ { \ell } ) } } \right\} .\tag{7.3}
$$

At the zero instance, $G ( 0 ) > 0 ;$ , and hence $\mathbb { P } _ { 0 } ( N _ { - } ( T ) > 0 ) > 0$ . Only the rewards from arms $i \neq j$ difer between $\nu ( 0 )$ and $\nu ( \Delta )$ . For every finite $\Delta .$ , the chain rule for KL divergence and the Gaussian KL formula therefore give $D _ { \mathrm { K L } } ( \mathbb { P } _ { 0 } \| \mathbb { P } _ { \Delta } ) = G ( 0 ) \Delta ^ { 2 } / ( 2 \sigma ^ { 2 } ) < \infty$ , so $\mathbb { P } _ { 0 } \ll \mathbb { P } _ { \Delta }$ . Consequently, $\mathbb { P } _ { \Delta } ( N _ { - } ( T ) > 0 ) > 0$ , and thus $G ( \Delta ) > 0$ . Therefore, the sequence $\{ \Delta _ { \ell } \} _ { \ell \ge 0 }$ is well defined.

Since $G ( \Delta ) \leq T$ , every untruncated increment in (7.3) is at least $\sigma / { \sqrt { T } }$ . Thus the sequence reaches $\Delta .$ after finitely many steps. Let L be the first index at which the sequence reaches $\Delta _ { \star }$ , namely, $L : = \operatorname* { m i n } \{ \ell \geq$ $1 : \Delta _ { \ell } = \Delta _ { \star } \}$ . Since only the final increment can be truncated,

$$
L \leq 1 + \frac { \Delta _ { \star } \sqrt { T } } { \sigma } = 1 + \frac { 8 \mathcal { R } _ { K , T } } { \sigma \sqrt { T } } .\tag{7.4}
$$

All points in the sequence belong to $[ 0 , \Delta _ { \star } ]$ , so all corresponding instances are admissible. We next establish an order-T gap between the endpoint values of G along this sequence. By (7.2), $\Delta _ { \star } \cdot G ( \Delta _ { \star } ) = { \mathcal { R } } _ { K , T } ( \nu ( \Delta _ { \star } ) ) \leq$ $\mathcal { R } _ { K , T }$ . The definition of $\Delta ,$ therefore gives $G ( \Delta _ { \star } ) \leq T / 8$

Along this sequence, the triangle inequality implies

$$
\sum _ { \ell = 0 } ^ { L - 1 } | G ( \Delta _ { \ell + 1 } ) - G ( \Delta _ { \ell } ) | \geq | G ( \Delta _ { L } ) - G ( \Delta _ { 0 } ) | = | G ( \Delta _ { \star } ) - G ( \Delta _ { 0 } ) | \geq \frac { T } { 2 } - \frac { T } { 8 } = \frac { 3 T } { 8 } .\tag{7.5}
$$

Hence, for some $\ell \in \{ 0 , \ldots , L - 1 \} , | G ( \Delta _ { \ell + 1 } ) - G ( \Delta _ { \ell } ) | \geq 3 T / ( 8 L )$

The following lemma is a variance-rescaled variant of Lemma 6.1 in Chen and Lu (2026), applied to the aggregate count $N _ { - } ( T )$ . It compares two nearby Gaussian instances and converts the preceding gap into an instability lower bound.

Lemma 7.2. Let $0 \leq \Delta < \Delta ^ { \prime } \leq M$ and $\Delta ^ { \prime } \leq \Delta + \sigma / \sqrt { G ( \Delta ) }$ , where $G ( \Delta )$ is defined in (7.1). Under any algorithm,

$$
\operatorname* { m a x } \left\{ \begin{array} { l l } { \displaystyle S _ { K , T } ( \nu ( \Delta ) ) , S _ { K , T } ( \nu ( \Delta ^ { \prime } ) ) \} \geq \frac { e ^ { - 1 / 4 } } { 4 \sqrt { 2 } } \left| G ( \Delta ^ { \prime } ) - G ( \Delta ) \right| . } \end{array} \right.
$$

See Section 7.1 for a detailed proof. Applying Lemma 7.2 to the preceding gap and using (7.4), we obtain

$$
\begin{array} { r l } & { S _ { K , T } \geq \operatorname* { m a x } \{ S _ { K , T } ( \nu ( \Delta _ { \ell } ) ) , S _ { K , T } ( \nu ( \Delta _ { \ell + 1 } ) ) \} } \\ & { \qquad \geq \frac { 3 e ^ { - 1 / 4 } T } { 3 2 \sqrt { 2 } L } \geq \frac { 3 e ^ { - 1 / 4 } } { 2 5 6 \sqrt { 2 } } \cdot \frac { T } { 1 + \mathcal { R } _ { K , T } / ( \sigma \sqrt { T } ) } . } \end{array}\tag{7.6}
$$

The last inequality relaxes the bound on L in (7.4) to $L \leq 8 ( 1 + \mathcal { R } _ { K , T } / ( \sigma \sqrt { T } ) )$ . Multiplying (7.6) by $\mathcal { R } _ { K , T }$ gives

$$
\mathcal { R } _ { K , T } S _ { K , T } \geq \frac { 3 e ^ { - 1 / 4 } \sigma } { 2 5 6 \sqrt { 2 } } \cdot \frac { \mathcal { R } _ { K , T } } { \sigma \sqrt { T } + \mathcal { R } _ { K , T } } \cdot T ^ { 3 / 2 } .\tag{7.7}
$$

The standard Gaussian minimax construction in Exercise 15.2 of Lattimore and Szepesv´ari (2020), after scaling to variance $\sigma ^ { 2 }$ and restricting the means to $[ - M , M ]$ , gives

$$
\mathcal { R } _ { K , T } \geq \operatorname* { m i n } \left\{ \frac { \sigma } { 8 } , \frac { M } { 4 } \right\} \sqrt { K T } = \sigma c _ { M , \sigma } \sqrt { K T } , \qquad c _ { M , \sigma } : = \operatorname* { m i n } \left\{ \frac { 1 } { 8 } , \frac { M } { 4 \sigma } \right\} .\tag{7.8}
$$

See Section 7.2 for a detailed proof. Since $K \geq 2$ and $x \mapsto x / ( 1 + x )$ is increasing on $[ 0 , \infty )$ , it follows that $\mathcal { R } _ { K , T } / ( \sigma \sqrt { T } + \mathcal { R } _ { K , T } ) \geq c _ { M , \sigma } \sqrt { K } / ( 1 + c _ { M , \sigma } \sqrt { K } ) \geq \sqrt { 2 } c _ { M , \sigma } \big / ( 1 + \sqrt { 2 } c _ { M , \sigma } )$ . Substituting it into (7.7) yields the K-independent bound

$$
\mathcal { R } _ { K , T } S _ { K , T } \geq \frac { 3 e ^ { - 1 / 4 } \sigma c _ { M , \sigma } } { 2 5 6 \left( 1 + \sqrt { 2 } c _ { M , \sigma } \right) } T ^ { 3 / 2 } .
$$

Remark 7.3 (Comparison of lower-bound conditions). Choosing an arm with the smallest expected pull count at $\Delta = 0$ as the optimal arm in the instance construction yields the lower bound on $G ( 0 )$ needed for (7.5), which Chen and Lu (2026) obtain from symmetry. Absolute continuity of the Gaussian interaction laws implies $G ( \Delta ) > 0$ for every finite $\Delta ,$ ensuring that the recursion in (7.3) is well defined without nondegenerate exploration. Finally, under $\mathcal { R } _ { K , T } \leq M T / 8 , \Delta _ { \star } \leq M$ , while the regret identity gives $G ( \Delta _ { \star } ) \leq T / 8$ . This admissible endpoint for fixed $T _ { i }$ , together with the lower bound on $G ( 0 )$ , yields the order-T decrease in G required by (7.5), replacing the asymptotic active-learning argument of Chen and Lu (2026).

## 7.1 Proof of Lemma 7.2

Proof of Lemma 7.2. The claim is immediate if $G ( \Delta ) = G ( \Delta ^ { \prime } )$ . Otherwise, define an event

$$
\begin{array} { r } { A : = \left\{ \begin{array} { l l } { N _ { - } ( T ) < \displaystyle \frac { G ( \Delta ) + G ( \Delta ^ { \prime } ) } { 2 } , } & { G ( \Delta ) > G ( \Delta ^ { \prime } ) , } \\ { N _ { - } ( T ) \geq \displaystyle \frac { G ( \Delta ) + G ( \Delta ^ { \prime } ) } { 2 } , } & { G ( \Delta ) < G ( \Delta ^ { \prime } ) . } \end{array} \right. } \end{array}
$$

$\mathrm { O n } \ A , \ N _ { - } ( T )$ is at least $| G ( \Delta ^ { \prime } ) - G ( \Delta ) | / 2$ away from $G ( \Delta )$ , while on $\mathcal { A } ^ { c }$ it is at least the same distance from $G ( \Delta ^ { \prime } )$ . Since $N _ { - } ( T ) { } = { } T - N _ { i } ( T )$ , it follows that

$$
( S _ { K , T } ( \nu ( \Delta ) ) ) ^ { 2 } \geq \mathrm { V a r } _ { \Delta } ( N _ { j } ( T ) ) = \mathrm { V a r } _ { \Delta } ( N _ { - } ( T ) ) \geq \mathbb { P } _ { \Delta } ( \mathcal { A } ) \frac { | G ( \Delta ^ { \prime } ) - G ( \Delta ) | ^ { 2 } } { 4 } .
$$

Similarly, we have $( S _ { K , T } ( \nu ( \Delta ^ { \prime } ) ) ) ^ { 2 } \geq \mathbb { P } _ { \Delta ^ { \prime } } ( \mathcal { A } ^ { c } ) \cdot | G ( \Delta ^ { \prime } ) - G ( \Delta ) | ^ { 2 } / 4$ . Only the reward distributions of arms $i \neq j$ difer between the two instances. The chain rule for KL divergence and the Gaussian KL formula therefore give

$$
D _ { \mathrm { K L } } ( \mathbb { P } _ { \Delta } | | \mathbb { P } _ { \Delta ^ { \prime } } ) = \sum _ { i \neq j } \mathbb { E } _ { \Delta } [ N _ { i } ( T ) ] D _ { \mathrm { K L } } \big ( \mathcal { N } ( - \Delta , \sigma ^ { 2 } ) | | \mathcal { N } ( - \Delta ^ { \prime } , \sigma ^ { 2 } ) \big ) = G ( \Delta ) \frac { ( \Delta ^ { \prime } - \Delta ) ^ { 2 } } { 2 \sigma ^ { 2 } } \leq \frac { 1 } { 2 } .
$$

The Bretagnolle–Huber inequality therefore implies $\begin{array} { r } { \mathbb { P } _ { \Delta } ( \mathcal { A } ) + \mathbb { P } _ { \Delta ^ { \prime } } ( \mathcal { A } ^ { c } ) \geq \exp \big ( - D _ { \mathrm { K L } } ( \mathbb { P } _ { \Delta } | | \mathbb { P } _ { \Delta ^ { \prime } } ) \big ) / 2 \geq e ^ { - 1 / 2 } / 2 . } \end{array}$ Consequently,

$$
\begin{array} { r l } & { \operatorname* { m a x } \left\{ S _ { K , T } ( \nu ( \Delta ) ) , S _ { K , T } ( \nu ( \Delta ^ { \prime } ) ) \right\} \geq \frac { | G ( \Delta ^ { \prime } ) - G ( \Delta ) | } { 4 } \left( \sqrt { \mathbb { P } _ { \Delta } ( A ) } + \sqrt { \mathbb { P } _ { \Delta ^ { \prime } } ( A ^ { c } ) } \right) } \\ & { \qquad \geq \frac { | G ( \Delta ^ { \prime } ) - G ( \Delta ) | } { 4 } \sqrt { \mathbb { P } _ { \Delta } ( A ) + \mathbb { P } _ { \Delta ^ { \prime } } ( \mathcal { A } ^ { c } ) } \geq \frac { e ^ { - 1 / 4 } } { 4 \sqrt { 2 } } | G ( \Delta ^ { \prime } ) - G ( \Delta ) | . } \end{array}
$$

This concludes the proof of Lemma 7.2.

## 7.2 Proof of (7.8)

Proof of the Gaussian minimax bound (7.8).

The proof applies the Gaussian minimax construction in Exercise 15.2 of Lattimore and Szepesv´ari (2020), with the variance scaled to $\sigma ^ { 2 }$ and the means restricted to $[ - M , M ]$ . Set

$$
c _ { M , \sigma } : = \operatorname* { m i n } \left\{ \frac { 1 } { 8 } , \frac { M } { 4 \sigma } \right\} , \qquad \delta : = 4 c _ { M , \sigma } \sigma \sqrt { \frac { K } { T } } .
$$

For each $i \in [ K ]$ , consider the Gaussian instance $\nu ^ { ( i ) }$ in which arm i has mean $\delta$ and all other arms have mean zero, with variance $\sigma ^ { 2 }$ throughout; let $\nu ^ { ( 0 ) }$ be the all-zero instance. These instances are admissible because $\delta \leq M$ . Writing $\mathbb { P } _ { \nu ^ { ( i ) } } , \mathbb { E } _ { \nu ^ { ( i ) } } , R _ { \nu ^ { ( i ) } }$ for the corresponding probability, expectation and regret under the interaction with instance $\nu ^ { ( i ) }$ , Pinsker’s inequality and the divergence decomposition give

$$
\begin{array} { r l } & { \mathbb { E } _ { \nu ^ { ( \star ) } } [ N _ { i } ( T ) ] \leq \mathbb { E } _ { \nu ^ { ( 0 ) } } [ N _ { i } ( T ) ] + T \cdot \mathrm { T V } ( \mathbb { P } _ { \nu ^ { ( i ) } } , \mathbb { P } _ { \nu ^ { ( 0 ) } } ) \leq \mathbb { E } _ { \nu ^ { ( 0 ) } } [ N _ { i } ( T ) ] + T \sqrt { \frac { 1 } { 2 } } D _ { \mathrm { K L } } ( \mathbb { P } _ { \nu ^ { ( 0 ) } } \| \mathbb { P } _ { \nu ^ { ( i ) } } ) } \\ & { \qquad = \mathbb { E } _ { \nu ^ { ( 0 ) } } [ N _ { i } ( T ) ] + 2 c _ { M , \sigma } \sqrt { K T } \mathbb { E } _ { \nu ^ { ( 0 ) } } [ N _ { i } ( T ) ] , } \end{array}
$$

where $\mathrm { T V } ( \mathbb P _ { \nu ^ { ( i ) } } , \mathbb P _ { \nu ^ { ( 0 ) } } ) = \operatorname* { s u p } _ { A } | \mathbb P _ { \nu ^ { ( i ) } } ( A ) - \mathbb P _ { \nu ^ { ( 0 ) } } ( A ) |$ . Summing over i and applying the Cauchy–Schwarz inequality yields

$$
\sum _ { i = 1 } ^ { K } \mathbb { E } _ { \nu ^ { ( i ) } } [ N _ { i } ( T ) ] \leq T + 2 c _ { M , \sigma } K T .
$$

Hence, with $R _ { \nu ^ { ( i ) } } = \delta ( T - \mathbb { E } _ { \nu ^ { ( i ) } } [ N _ { i } ( T ) ] )$

$$
\mathcal { R } _ { K , T } \geq \frac { 1 } { K } \sum _ { i = 1 } ^ { K } R _ { \nu ^ { ( i ) } } \geq 4 c _ { M , \sigma } \sigma \left( 1 - \frac { 1 } { K } - 2 c _ { M , \sigma } \right) \sqrt { K T } \geq c _ { M , \sigma } \sigma \sqrt { K T } = \operatorname* { m i n } \left\{ \frac { \sigma } { 8 } , \frac { M } { 4 } \right\} \sqrt { K T } ,
$$

where the last inequality uses $K \geq 2$ and $c _ { M , \sigma } \leq 1 / 8$

## 8 Conclusion

We characterize the finite-time regret–instability trade-of for K-armed bandits. Our lower bound has a constant independent of K, while the tunable SLE-UCB algorithm achieves a regret–instability product of $O ( T ^ { 3 / 2 } \log K )$ . These bounds match in their $T ^ { 3 / 2 }$ dependence and up to a logarithmic factor in K, thereby characterizing the frontier to logarithmic accuracy. In adaptive experimentation and dynamic resource allo cation, the tuning parameter lets a decision maker choose how much regret to accept for more reproducible allocations, while the lower bound shows that this trade-of is unavoidable within the model. It would be interesting to investigate whether these ideas extend to contextual bandits and reinforcement learning.

## References

Audibert, J.-Y. and Bubeck, S. (2009). Minimax policies for adversarial and stochastic bandits. In Colt.

Auer, P., Cesa-Bianchi, N. and Fischer, P. (2002). Finite-time analysis of the multiarmed bandit problem. Machine learning, 47 235–256.

Boucheron, S., Lugosi, G. and Massart, P. (2013). Bounding the variance. In Concentration Inequalities: A Nonasymptotic Theory of Independence. Oxford University Press, 52–82.

Chen, Y. and Lu, J. (2025). A characterization of sample adaptivity in ucb data. arXiv preprint arXiv:2503.04855.

Chen, Y. and Lu, J. (2026). Bandit allocational instability. arXiv preprint arXiv:2602.07472.

Efron, B. and Stein, C. (1981). The Jackknife Estimate of Variance. The Annals of Statistics, 9 586 – 596.

Fan, L. and Glynn, P. W. (2022). The typical behavior of bandit algorithms. arXiv preprint arXiv:2210.05660.

Fan, L. and Glynn, P. W. (2025). The fragility of optimized bandit algorithms. Operations Research, 73 3173–3198.

Fan, Y., Han, Y., Lv, J., Xu, X. and Zhou, Z. (2025). Precise asymptotics and refined regret of variance-aware ucb. Advances in Neural Information Processing Systems, 38 160795–160837.

Gittins, J., Glazebrook, K. and Weber, R. (2011). Multi-armed bandit allocation indices. John Wiley & Sons.

Halder, B., Pan, S. and Khamaru, K. (2025). Stable thompson sampling: Valid inference via variance inflation. arXiv preprint arXiv:2505.23260.

Halder, B., Sengupta, I., Chowdhury, K., Praharaj, S. and Khamaru, K. (2026). Stabilizing bandits using regularization: Precise regret and a quantitative central limit theorem.

Han, Q. (2026). Thompson sampling: Precise arm-pull dynamics and adaptive inference. arXiv preprint arXiv:2601.21131.

Han, Q., Khamaru, K. and Zhang, C.-H. (2024). Ucb algorithms for multi-armed bandits: Precise regret and adaptive inference. arXiv preprint arXiv:2412.06126.

Kalvit, A. and Zeevi, A. (2021). A closer look at the worst-case behavior of multi-armed bandit algorithms. Advances in Neural Information Processing Systems, 34 8807–8819.

Khamaru, K. and Zhang, C.-H. (2024). Inference with the upper confidence bound algorithm. arXiv preprint arXiv:2408.04595.

Lai, T. L. and Wei, C. Z. (1982). Least squares estimates in stochastic regression models with applications to identification and control of dynamic systems. The Annals of Statistics 154–166.

Lattimore, T. and Szepesv´ari, C. (2020). Bandit algorithms. Cambridge University Press.

Li, L., Chu, W., Langford, J. and Schapire, R. E. (2010). A contextual-bandit approach to personalized news article recommendation. In Proceedings of the 19th international conference on World wide web.

Praharaj, S. and Khamaru, K. (2025). On instability of minimax optimal optimism-based bandit algorithms. arXiv preprint arXiv:2511.18750.

Simchi-Levi, D. and Wang, C. (2023). Multi-armed bandit experimental design: Online decision-making and adaptive inference. In International Conference on Artificial Intelligence and Statistics. PMLR.

Simchi-Levi, D., Zheng, Z. and Zhu, F. (2022). A simple and optimal policy design for online learning with safety against heavy-tailed risk. Advances in Neural Information Processing Systems, 35 33795–33805.

Simchi-Levi, D., Zheng, Z. and Zhu, F. (2023). Stochastic multi-armed bandits: Optimal trade-of among optimality, consistency, and tail risk. Advances in Neural Information Processing Systems, 36 35619–35630.

Villar, S. S., Bowden, J. and Wason, J. (2015). Multi-armed bandit models for the optimal design of clinical trials: benefits and challenges. Statistical science: a review journal of the Institute of Mathematical Statistics, 30 199.

Yan, S. and Zhong, H. (2026). Optimism stabilizes thompson sampling for adaptive inference. arXiv preprint arXiv:2602.06014.

Zhu, F. and Simchi-Levi, D. (2025). Adaptive variance inflation in thompson sampling: Eficiency, safety, robustness, and beyond. Advances in Neural Information Processing Systems, 38 50466–50484.