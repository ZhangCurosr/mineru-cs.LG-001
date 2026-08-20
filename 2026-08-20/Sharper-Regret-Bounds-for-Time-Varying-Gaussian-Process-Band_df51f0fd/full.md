# Sharper Regret Bounds for Time-Varying Gaussian Process Bandits with Constant Exploration

Matthias Mandl Delft Institute of Applied Mathematics Delft University of Technology Mekelweg 4, 2628 CD, Delft M.J.Mandl-2@tudelft.nl

Hanne Kekkonen Delft Institute of Applied Mathematics Delft University of Technology Mekelweg 4, 2628 CD, Delft H.N.Kekkonen@tudelft.nl

## Abstract

We study Bayesian optimization in a time-varying environment where the unknown reward function evolves according to a Gaussian process drift model. Existing GP-UCB analyses in this setting typically require the exploration parameter to grow with the horizon to maintain uniform confidence bounds. Using per-round local confidence events, we show that GP-UCB can instead be run with a constant exploration parameter and obtain an expected-regret bound whose coefficient depends on the drift rate. We also derive a sharper time-varying maximum-informationgain bound. For the squared exponential kernel, it yields $\tilde { \gamma } _ { T } / T = \widetilde { \mathcal { O } } ( \epsilon ^ { 1 / 2 } )$ and expected average regret $\widetilde { \mathcal { O } } ( \epsilon ^ { 1 / 4 } )$ in the persistent-drift regime. The same constantexploration analysis also yields realized-regret guarantees. Simulations support the predicted logarithmic dependence of the bound-suggested exploration parameter on $1 / \epsilon .$

## 1 Introduction

Many sequential decision problems require optimizing an unknown objective through a limited number of costly evaluations. Examples include tuning the hyperparameters of machine learning models [Snoek et al., 2012], adapting control policies in robotics [Calandra et al., 2016], and selecting treatment strategies in clinical trials [Villar et al., 2015]. In these settings, each evaluation reveals only imperfect information about the objective, forcing a fundamental trade-off between exploration and exploitation.

Bayesian optimization (BO) provides a framework for sequentially optimizing an unknown objective by combining a probabilistic surrogate model with an acquisition rule. In the standard GP bandit formulation, the objective is a fixed function $f : S  \mathbb { R }$ . At each round t, the learner selects $x _ { t } \in S$ and observes $y _ { t } = f ( x _ { t } ) + z _ { t }$ , where $z _ { t }$ is observation noise. Gaussian processes (GPs) are commonly used as surrogate models because they provide posterior mean and uncertainty estimates in closed form.

When decisions affect performance throughout the learning process, performance is typically measured by cumulative regret, $\begin{array} { r } { R _ { T } = \sum _ { t = 1 } ^ { T } ( \operatorname* { m a x } _ { x \in \mathcal { S } } f ( x ) - f ( x _ { t } ) ) } \end{array}$ . This criterion originates in the multi-armed bandit literature [Robbins, 1952, Lai and Robbins, 1985, Auer et al., 2002]. Algorithms such as Gaussian Process Upper Confidence Bound (GP-UCB) [Srinivas et al., 2012] and GP-Thompson sampling [Russo and Van Roy, 2014, Chowdhury and Gopalan, 2017] extend bandit methods to continuous domains and achieve sublinear regret under suitable kernel regularity assumptions, with guarantees typically expressed through the maximum information gain [Srinivas et al., 2012, Vakili et al., 2021, Scarlett et al., 2017].

However, many applications are inherently dynamic. System performance may drift due to environmental changes or evolving user behavior, necessitating a time-dependent reward model $f _ { t } : S  \mathbb { R }$ When the objective changes over time, sublinear cumulative regret is generally unachievable. The relevant theoretical goal shifts to bounding the average regret, $\bar { R r / T }$ , as a function of the environmental drift rate.

A principled probabilistic model for this setting was introduced by Bogunovic et al. [2016]. In their time-varying Gaussian process (TV-GP) model, the latent function evolves according to a Markov recursion driven by $\mathrm { G P }$ innovations, which induces a separable spatio-temporal kernel. Under this model, the learner updates a temporally discounted $\mathrm { G P }$ posterior and selects actions using a modified GP-UCB rule. Their analysis shows that near-optimal regret rates are achievable up to logarithmic factors. Subsequent work has studied related models of nonstationarity, including change-point structure [Brunzema et al., 2022], irregular evaluation times [Imamura et al., 2020], and bounded variation in RKHS norms [Deng et al., 2022]. A separate frequentist line studies nonstationary kernelized bandits under sup-norm variation, including algorithms [Zhou and Shroff, 2021, Hong et al., 2023, Iwazaki and Takeno, 2025] and lower bounds [Cai and Scarlett, 2025]. Other extensions include kernel-agnostic adaptation [Ziomek et al., 2024] and stale-data removal in dynamic Bayesian optimization [Bardou et al., 2024, Bardou and Thiran, 2025].

A recurring theoretical and practical limitation in these analyses is the exploration schedule. Existing regret guarantees typically require the GP-UCB exploration parameter, $\beta _ { t }$ , to grow with time t. This leads to deteriorating performance on long time horizons and introduces schedule tuning parameters. In practice, such schedules are difficult to calibrate and induce unnecessary over-exploration.

This paper revisits the TV-GP setting of Bogunovic et al. [2016], where the temporal evolution is governed by a drift parameter $\epsilon > 0 ,$ and shows that GP-UCB can be run with a constant exploration parameter while retaining rigorous performance guarantees. Our analysis uses local confidence events at each round rather than a single event that holds uniformly over the horizon. For any fixed drift level, Theorem 1 gives an expected regret bound with linear dependence on $T$ and a coefficient controlled by $\tilde { \gamma } _ { T } / \bar { T }$ . In the persistent-drift regime, this yields a coefficient that decreases as the drift rate decreases. This differs from standard deterministic uniform-confidence GP-UCB analyses in static environments, where the confidence parameter generally depends on the horizon or time.

The analysis also gives a sharper dependence on the drift rate through the time-varying maximum information gain. If the static information gain satisfies $\gamma _ { N } = \widetilde { \mathcal { O } } ( N ^ { q } )$ ), then in the persistent-drift regime

$$
\frac { \tilde { \gamma } _ { T } } { T } = \widetilde { \mathcal { O } } \Bigl ( \epsilon ^ { \frac { 2 ( 1 - q ) } { 4 - q } } \Bigr ) , \qquad \frac { \mathbb { E } [ R _ { T } ] } { T } = \widetilde { \mathcal { O } } \Bigl ( \epsilon ^ { \frac { 1 - q } { 4 - q } } \Bigr ) .
$$

For the squared exponential kernel, the latter becomes $\widetilde { \mathcal { O } } ( \epsilon ^ { 1 / 4 } )$ . Our analysis also isolates how the bound-suggested constant exploration parameter depends on the drift rate. In the regret bounds of Bogunovic et al. [2016], the prescribed exploration schedule is independent of $\epsilon .$ . In contrast, balancing the upper-bound scaling suggests that the bound-suggested constant $\beta$ increases as ϵ decreases. This does not by itself imply uniformly more exploration, since the UCB rule depends on the product of $\sqrt { \beta }$ and the posterior standard deviation. Rather, when drift is slower, past observations remain informative for longer and the posterior variance is typically smaller. A larger $\beta$ compensates for this reduced variance by assigning more value to reducing the remaining uncertainty, which is also more consequential because information acquired now remains useful over a longer effective horizon.

The dependence of the bound-suggested exploration parameter on the temporal dynamics also gives a simple rule for changes in sampling frequency. If observations are collected every $\Delta$ time units, larger $\Delta$ corresponds to greater effective drift between successive evaluations. Temporal correlations then decay faster, past observations become obsolete more quickly, and the analysis predicts a smaller bound-suggested exploration parameter.

The remainder of the paper is organized as follows. Section 2 introduces the time-varying GP model, the domain regularity assumptions, and the temporally discounted GP-UCB policy. Section 3 presents the expected-regret bound and the post-hoc and prescribed-confidence guarantees for realized regret. Section 4 derives the drift-dependent information-gain and regret rates and analyzes the bound-suggested exploration parameter. Section 5 provides a supporting empirical simulation study. Section 6 provides the conclusion. Additional proofs and technical results are deferred to the appendices.

## 2 Problem setting

We start by introducing the dynamic regret setting. Let $S \subseteq \mathbb { R } ^ { d }$ be compact. At each round $t = 1 , 2 , \ldots$ , the algorithm selects a query point $x _ { t } \in S$ and observes

$$
y _ { t } = f _ { t } ( x _ { t } ) + z _ { t } , \qquad z _ { t } \sim \mathcal { N } ( 0 , \sigma ^ { 2 } ) \ \mathrm { i . i . d . } ,\tag{1}
$$

where $f _ { t } : S $ R is an unknown reward function that changes with time. Let

$$
x _ { t } ^ { \star } \in \arg \operatorname* { m a x } _ { x \in S } f _ { t } ( x )
$$

denote a maximizer at round t. The instantaneous regret is

$$
r _ { t } : = f _ { t } ( x _ { t } ^ { \star } ) - f _ { t } ( x _ { t } ) \geq 0 ,\tag{2}
$$

and the cumulative regret over $T$ rounds is

$$
R _ { T } : = \sum _ { t = 1 } ^ { T } r _ { t } .\tag{3}
$$

In the non-stationary setting, sublinear cumulative regret is not expected in general, so the main objective is to control the linear growth rate and its dependence on the temporal properties of $f _ { t }$ . We model the function sequence $f _ { t }$ using the time-varying Gaussian process construction of Bogunovic et al. [2016]. Let $( g _ { t } ) _ { t \geq 1 }$ be i.i.d. draws from $\mathcal { G P } ( 0 , k _ { S } )$ , where k is a spatial kernel on S. The latent process is defined by

$$
f _ { 1 } ( x ) = g _ { 1 } ( x ) , \qquad f _ { t + 1 } ( x ) = { \sqrt { 1 - \epsilon } } f _ { t } ( x ) + { \sqrt { \epsilon } } g _ { t + 1 } ( x ) , \quad t \geq 1 ,\tag{4}
$$

with drift parameter $\epsilon \in ( 0 , 1 ]$ . For each fixed $t ,$ the marginal law remains $\mathcal { G P } ( 0 , k _ { S } )$ , while the temporal dependence decays geometrically. Equivalently, $\{ f _ { t } ( x ) \} _ { ( x , t ) }$ is a Gaussian process on $\mathcal { S } \times \mathbb { N }$ with separable kernel

$$
k \big ( ( x , t ) , ( x ^ { \prime } , t ^ { \prime } ) \big ) = k _ { S } ( x , x ^ { \prime } ) k _ { T } ( t , t ^ { \prime } ) , \qquad k _ { T } ( t , t ^ { \prime } ) = ( \sqrt { 1 - \epsilon } ) ^ { | t - t ^ { \prime } | } .\tag{5}
$$

We refer to (5) as the time-decayed kernel, since its temporal factor discounts correlations geometrically with temporal separation. We assume throughout that the prior variance is uniformly bounded,

$$
k _ { S } ( x , x ) \leq 1 \qquad { \mathrm { f o r ~ a l l ~ } } x \in S .
$$

Given past data

$$
\mathcal { H } _ { t - 1 } = \{ ( x _ { i } , y _ { i } ) \} _ { i = 1 } ^ { t - 1 } , \qquad y _ { 1 : t - 1 } = ( y _ { 1 } , \tiny { \cdot \cdot \cdot } , y _ { t - 1 } ) ^ { \top } ,
$$

Gaussian process regression under the kernel (5) yields a Gaussian posterior for $f _ { t } ( x )$ with mean and variance

$$
\mu _ { t - 1 } ( x ) = \tilde { k } _ { t - 1 } ( x ) ^ { \top } \big ( \tilde { K } _ { t - 1 } + \sigma ^ { 2 } I \big ) ^ { - 1 } y _ { 1 : t - 1 } ,\tag{6}
$$

$$
\sigma _ { t - 1 } ^ { 2 } ( x ) = k _ { S } ( x , x ) - \tilde { k } _ { t - 1 } ( x ) ^ { \top } \big ( \tilde { K } _ { t - 1 } + \sigma ^ { 2 } I \big ) ^ { - 1 } \tilde { k } _ { t - 1 } ( x ) ,\tag{7}
$$

where

$$
\tilde { k } _ { t - 1 } ( x ) : = \big [ k _ { S } ( x _ { i } , x ) ( \sqrt { 1 - \epsilon } ) ^ { | t - i | } \big ] _ { i = 1 } ^ { t - 1 } , \qquad \tilde { K } _ { t - 1 } : = \big [ k _ { S } ( x _ { i } , x _ { j } ) ( \sqrt { 1 - \epsilon } ) ^ { | i - j | } \big ] _ { i , j = 1 } ^ { t - 1 } .\tag{8}
$$

These expressions are obtained by evaluating the spatio-temporal kernel (5) at the observed pairs $( x _ { i } , i )$ and the prediction point $( x , t )$ . The factors $( { \sqrt { 1 - \epsilon } } ) ^ { \left| t - i \right| }$ account for the fact that older observations are less informative about the current function $f _ { t } .$

As in standard GP bandit analysis, we measure cumulative uncertainty by the time-varying maximum information gain [Bogunovic et al., 2016], which controls sums of posterior variances in regret bounds [Srinivas et al., 2012, Lemma 5.4]. For a design set

$$
A = \{ ( x _ { 1 } , 1 ) , \ldots , ( x _ { T } , T ) \} ,
$$

let $f _ { A } = ( f _ { 1 } ( x _ { 1 } ) , \ldots , f _ { T } ( x _ { T } ) ) ^ { \top }$ and $y _ { A } = ( y _ { 1 } , \ldots , y _ { T } ) ^ { \top }$ . Since $y _ { A } = f _ { A } + z _ { A }$ with independent Gaussian noise,

$$
I ( y _ { A } ; f _ { A } ) = \frac 1 2 \log \operatorname * { d e t } { \big ( } I + \sigma ^ { - 2 } \tilde { K } _ { A } { \big ) } ,\tag{9}
$$

where $\tilde { K } _ { A }$ is the covariance matrix of $f _ { A }$ induced by (5). We define the time-varying maximum information gain as

$$
\tilde { \gamma } _ { T } : = \operatorname* { m a x } _ { x _ { 1 : T } \in S ^ { T } } I ( y _ { A } ; f _ { A } ) ,\tag{10}
$$

To control the discretization error incurred on a continuous domain, we impose the following standard GP-UCB regularity condition [Srinivas et al., 2012, Bogunovic et al., 2016].

Assumption 1 (Derivative tail bound). There exist constants $a \geq 1$ and $b > 0$ such that, for each $t \geq 1$ , each spatial dimension $j \in \{ 1 , \ldots , d \}$ , and all $L > 0$

$$
\mathbb { P } \Big \{ \operatorname* { s u p } _ { x \in \cal S } \Big | \frac { \partial f _ { t } ( x ) } { \partial x ^ { ( j ) } } \Big | > L \Big \} \Big \le a \exp \Big ( - \big ( \frac { L } { b } \big ) ^ { 2 } \Big ) .\tag{11}
$$

This condition holds for many commonly used kernels, including the squared exponential kernel and Matérn kernels with smoothness parameter $\nu > 2 .$ , under appropriate differentiability conditions on $k _ { S } ;$ see Ghosal and Roy [2006, Theorem 5]. The TV-GP-UCB selection rule is given in Algorithm 1. At round t, the algorithm selects

$$
x _ { t } \in \arg \operatorname* { m a x } _ { x \in S } \Big ( \mu _ { t - 1 } ( x ) + \sqrt { \beta _ { t } } \sigma _ { t - 1 } ( x ) \Big ) .\tag{12}
$$

The main result of this paper shows that under the time-varying model (4), one can take $\beta _ { t } \equiv \beta$ constant and obtain a regret guarantee with improved long-horizon scaling. This contrasts with earlier analyses that rely on time-increasing exploration schedules.

Algorithm 1 Constant-β Time-Varying GP-UCB (TV-GP-UCB)   
Require: Domain S, GP prior $( 0 , k _ { S } , \sigma ^ { 2 } )$ , drift parameter $\epsilon \in ( 0 , 1 ]$ , constant exploration parameter   
$\beta > 0$   
1: for $t = 1 , 2 , \dots .$ do   
2: Choose x<sub>t</sub> ∈ arg max<sub>x∈S</sub> $\left( \mu _ { t - 1 } ( x ) + \sqrt { \beta } \sigma _ { t - 1 } ( x ) \right)$   
3: Observe $y _ { t } = f _ { t } ( x _ { t } ) + z _ { t }$   
4: Update posterior mean $\mu _ { t }$ and variance $\sigma _ { t } ^ { 2 }$ via (6) and (7)   
5: end for

## 3 Improved Regret Bound

Static GP-UCB analyses usually prove a single confidence event that holds simultaneously over all rounds and all points in a discretization. To keep this event at fixed probability over a horizon $T$ , the per-round failure probabilities must be summable, which forces the exploration parameter $\beta _ { t }$ to grow with t. Existing analyses of the time-varying GP model (4), in particular Bogunovic et al. [2016], use the same time-increasing exploration schedule. This introduces horizon-dependent tuning, leads to average-regret bounds that grow with T, and obscures the dependence of exploration on the drift rate ϵ.

We avoid this global-in-time confidence event. When $\epsilon > 0 .$ , older observations are progressively discounted, so posterior uncertainty need not decrease monotonically. This built-in forgetting prevents the algorithm from becoming permanently overconfident at a fixed point and removes the need for a time-increasing exploration schedule in the analysis. We prove that Algorithm 1 with a constant exploration parameter attains an $\mathcal { O } ( T )$ expected cumulative regret bound for every fixed $\epsilon > 0$ . The proof combines a per-round UCB regret bound on a local confidence event with an algorithm-agnostic bound on the rare complement event.

## 3.1 Expected regret with a constant exploration parameter

Theorem 1 (Expected regret for constant- $\cdot \beta$ TV-GP-UCB). Let $S \subseteq [ 0 , r ] ^ { d }$ be compact and convex. Assume the time-varying GP model (4) with drift $\epsilon \in ( 0 , 1 ]$ , spatial kernel $k _ { S }$ satisfying $k _ { S } ( x , x ) \leq 1$ for all $x \in S$ , and the tail condition (11). Assume observations are generated by

$$
y _ { t } = f _ { t } ( x _ { t } ) + z _ { t } , \qquad z _ { t } \sim \mathcal { N } ( 0 , \sigma ^ { 2 } )
$$

with independent noise. For $t \geq 1$ , define the spatial span

$$
W _ { t } : = \operatorname* { s u p } _ { x \in S } f _ { t } ( x ) - \operatorname* { i n f } _ { x \in S } f _ { t } ( x ) ,
$$

let $\mu _ { w } = \mathbb { E } [ W _ { t } ] _ { : }$ , and let $\sigma _ { w } ^ { 2 }$ be anyfinite sub-Gaussian variance proxyfor $W _ { t } ,$ so that

$$
\mathbb { P } \{ W _ { t } - \mu _ { w } > u \} \le \exp \left( - \frac { u ^ { 2 } } { 2 \sigma _ { w } ^ { 2 } } \right) \qquad f o r { a l l } u > 0 .
$$

Fix $\delta \in ( 0 , 1 )$ and $\tau \in \mathbb { N } ^ { + }$ . Ifthe evaluation points $\{ x _ { t } \} _ { t = 1 } ^ { T }$ are selected according to Algorithm 1 with any constant exploration parameter satisfying

$$
\begin{array} { r } { \beta \geq 2 \Big ( \log \big ( \frac 6 \delta \big ) + d \log ( \tau ) \Big ) . } \end{array}\tag{13}
$$

then the expected cumulative regret satisfies

$$
\mathbb { E } [ R _ { T } ] \le \sqrt { C _ { 1 } T \beta \tilde { \gamma } _ { T } } + T C _ { 2 } + T \delta C _ { 3 } ,\tag{14}
$$

where $\tilde { \gamma } _ { T } = \tilde { \gamma } _ { T } ( S , k _ { S } , \epsilon , \sigma )$ is the size- ${ \mathbf { \nabla } } \cdot { T }$ maximum information gain (10) under the time-decayed kernel, and

$$
C _ { 1 } = \frac { 8 } { \log ( 1 + \sigma ^ { - 2 } ) } , \qquad C _ { 2 } = \frac { r d b } { \tau } \sqrt { \log \left( \frac { 3 a d } { \delta } \right) } , \qquad C _ { 3 } = \mu _ { w } + 2 \sigma _ { w } \sqrt { 2 \log ( e / \delta ) } .
$$

For comparison, since $r _ { t } \le W _ { t }$ for every policy, the policy-independent span bound is

$$
R _ { T } \leq \sum _ { t = 1 } ^ { T } W _ { t } , \qquad { \frac { \mathbb { E } [ R _ { T } ] } { T } } \leq \mu _ { w } .
$$

Because each $f _ { t }$ has marginal law $\mathcal { G P } ( 0 , k _ { S } )$ , the quantity $\mu _ { w }$ is independent of $\epsilon .$ Thus Theorem 1 improves this policy-independent baseline whenever

$$
\sqrt { C _ { 1 } \beta \frac { \tilde { \gamma } _ { T } } { T } } + C _ { 2 } + \delta C _ { 3 } < \mu _ { w } .
$$

Under the drift-dependent optimization in Section 4.1, the algorithm-specific coefficient decreases as ϵ decreases in the persistent-drift regime, whereas $\mu _ { w }$ does not.

The span constants $\mu _ { w }$ and $\sigma _ { w }$ are finite by Borell–TIS, since $W _ { t }$ is the supremum of the centered Gaussian difference process $f _ { t } ( x ) - f _ { t } ( y )$ over $s \times s$ and the sample paths are almost surely bounded on the compact domain.

The proof, given in Appendix $\mathbf { A } ,$ constructs a local confidence event $\mathcal { E } _ { t }$ at each round rather than a single event over the full horizon. The event combines posterior concentration at the selected point, posterior concentration over a spatial grid of size at most $\tau ^ { d }$ , and the derivative tail bound (11). On $\mathcal { E } _ { t } ,$ , the instantaneous regret satisfies

$$
r _ { t } \le 2 \sqrt { \beta } \sigma _ { t - 1 } ( x _ { t } ) + C _ { 2 } ,
$$

where $C _ { 2 }$ is the discretization error. On the complement $\mathcal { E } _ { t } ^ { c }$ , we use only the deterministic inequality $r _ { t } \le W _ { t }$ . The sub-Gaussian tail of $W _ { t }$ gives

$$
\mathbb { E } \big [ W _ { t } \mathbb { 1 } _ { \mathcal { E } _ { t } ^ { c } } \big ] \leq \delta C _ { 3 } .
$$

Therefore,

$$
\begin{array} { r } { \mathbb { E } [ r _ { t } ] \leq 2 \sqrt { \beta } \mathbb { E } [ \sigma _ { t - 1 } ( x _ { t } ) ] + C _ { 2 } + \delta C _ { 3 } . } \end{array}
$$

Summing over t and applying the standard information-gain bound on posterior variances yields (14).

We use the notation $\delta$ and $\tau$ from Bogunovic et al. [2016]: $\delta$ controls the confidence level and $\tau ^ { d }$ is the spatial grid size. The distinction is that in their analysis δ is tied to a single confidence event over the full horizon, and therefore determines the time-varying exploration schedule $\beta _ { t }$ . Here $\delta$ only controls the local event $\mathcal { E } _ { t }$ used in the expected-regret decomposition. Thus $\delta$ and τ are analysis parameters: each choice gives a corresponding constant $\beta$ and expected-regret bound.

Since $\tilde { \gamma } _ { T }$ depends on the drift rate $\epsilon ,$ the bound-balancing choice of $( \delta , \tau )$ generally depends on ϵ as well. This is how the bound-suggested constant exploration parameter inherits a dependence on the temporal dynamics. Section 4.2 makes this explicit: under the scaling $\tilde { \gamma } _ { T } = \widetilde { \mathcal { O } } ( T \epsilon ^ { \alpha } )$ , the resulting bound-suggested choice satisfies $\beta _ { \mathrm { o p t } } = \mathcal { O } ( \bar { \log ( 1 / \epsilon ) } )$ ).

## 3.2 Realized cumulative regret and confidence guarantees

Theorem 1 bounds the expected cumulative regret. For realized regret, we distinguish two uses of a constant exploration parameter. In the post-hoc setting, $\beta$ is fixed without reference to the reporting confidence level ω, whereas in the prescribed-confidence setting, ω is specified before the run and is used to select a constant $\beta .$ In both cases, $\beta$ is independent of t and $T .$

Standard analyses of time-varying GP bandits construct probabilistic guarantees using a timeincreasing exploration schedule, typically $\beta _ { t } = \mathcal { O } ( \log ( t / \omega ) )$ . This enforces a global union bound over the horizon but introduces an $\mathcal { O } ( \sqrt { \log T } )$ factor in the leading information-gain term. We instead retain a constant exploration parameter and control the accumulated contribution of the local confidence failures.

Post-hoc confidence. Let

$$
U _ { T } : = \sqrt { C _ { 1 } T \beta \tilde { \gamma } _ { T } } + T C _ { 2 }
$$

denote the deterministic contribution from the UCB confidence widths and the discretization error. The proof bounds the realized pathwise regret by combining this term with the accumulated span over failed confidence events and applying Markov’s inequality to the latter.

Proposition 2 (Realized cumulative regret). Assume the setting and parameter choices of Theorem 1. For any targetfailure probability $\omega \in ( 0 , 1 )$ , the realized cumulative regret $R _ { T }$ satisfies the following bound with probability at least $1 - \omega { : }$

$$
R _ { T } \ \leq \ U _ { T } \ + \ { \frac { T \delta C _ { 3 } } { \omega } } .\tag{15}
$$

The proof is deferred to Appendix C.3. Here δ is the per-round analysis parameter that determines the constant $\beta ,$ whereas $\omega$ enters only in the reported bound. Thus the algorithm can be run with $\beta$ fixed independently of $\omega ,$ and the bound can subsequently be evaluated at any fixed reporting level $\omega$ chosen independently of the realized data. The cost of this post-hoc guarantee is the polynomial $1 / \omega$ dependence.

Prescribed confidence. If the target failure probability is specified before the run, the same result gives logarithmic dependence on $1 \bar { / } \omega$ while retaining a constant exploration parameter.

Corollary 3 (Prescribed-confidence realized regret). Assume the setting ofTheorem 1 and suppose that, in the regime under consideration,

$$
\frac { \tilde { \gamma } _ { T } } { T } = \widetilde { \mathcal { O } } ( \epsilon ^ { \alpha } ) , \qquad \alpha > 0 .
$$

Fix $\omega \in ( 0 , 1 )$ and choose

$$
\delta = \omega \epsilon ^ { \alpha / 2 } , \qquad \tau = \left\lceil \epsilon ^ { - \alpha / 2 } \right\rceil ,
$$

together with the constant exploration parameter

$$
\beta = 2 \left( \log \frac { 6 } { \omega \epsilon ^ { \alpha / 2 } } + d \log \left[ \epsilon ^ { - \alpha / 2 } \right] \right) = \mathcal { O } ( \log ( 1 / \omega ) + \log ( 1 / \epsilon ) ) .
$$

Then, with probability at least $1 - \omega _ { ; }$

$$
\frac { R _ { T } } { T } = \widetilde { \mathcal { O } } \left( \epsilon ^ { \alpha / 2 } \sqrt { 1 + \log ( 1 / \omega ) + \log ( 1 / \epsilon ) } \right) .\tag{16}
$$

In particular, $\beta$ remains independent of t and T.

Proposition 2 gives

$$
\frac { R _ { T } } { T } \leq \sqrt { C _ { 1 } \beta \frac { \tilde { \gamma } _ { T } } { T } } + C _ { 2 } + \frac { \delta C _ { 3 } } { \omega } .
$$

Under the choices above, the three terms are respectively

$$
\widetilde { \cal O } \left( \epsilon ^ { \alpha / 2 } \sqrt { \log ( 1 / \omega ) + \log ( 1 / \epsilon ) } \right) , \qquad { \cal O } \left( \epsilon ^ { \alpha / 2 } \sqrt { 1 + \log ( 1 / \omega ) + \log ( 1 / \epsilon ) } \right) ,
$$

and

$$
\mathcal { O } \left( \epsilon ^ { \alpha / 2 } \sqrt { 1 + \log ( 1 / \omega ) + \log ( 1 / \epsilon ) } \right) ,
$$

which yields (16).

Policy-independent fallback. Independently of the sampling policy,

$$
R _ { T } \leq \sum _ { t = 1 } ^ { T } W _ { t } .
$$

As established in Appendix C.2, with $\rho = \sqrt { 1 - \epsilon } .$ , this gives with probability at least $1 - \omega$

$$
R _ { T } \leq \sum _ { t = 1 } ^ { T } W _ { t } \leq T \mu _ { w } + \sqrt { 8 T \frac { 1 + \rho } { 1 - \rho } \log ( 1 / \omega ) } .\tag{17}
$$

This bound does not use the sampling decisions and is therefore not a replacement for the algorithmspecific guarantees above.

## 4 Drift-dependent regret rates and exploration

Theorem 1 gives a constant-exploration regret bound whose leading term is controlled by $\tilde { \gamma } _ { T } / T$ We first combine it with the time-varying information-gain bound in Appendix B to derive explicit drift-dependent rates, and then examine the corresponding bound-suggested choice of $\beta .$

## 4.1 Information-gain and regret rates

The time-varying information-gain bound in Appendix B gives an explicit dependence on the drift rate. Suppose the static maximum information gain satisfies

$$
\gamma _ { \cal N } = \widetilde { \mathcal { O } } ( N ^ { q } ) , \qquad q \in [ 0 , 1 ) .
$$

At the polynomial scale, the corresponding block length is

$$
N _ { \epsilon } = \epsilon ^ { - \frac { 2 } { 4 - q } } .
$$

In the persistent-drift regime where T exceeds this effective block length, Corollary B.4 gives

$$
\frac { \tilde { \gamma } _ { T } } { T } = \widetilde { \mathcal { O } } \left( \epsilon ^ { \frac { 2 ( 1 - q ) } { 4 - q } } \right) .\tag{18}
$$

Balancing the remaining terms in Theorem 1 as in Section 4.2 then yields

$$
\frac { \mathbb { E } [ R _ { T } ] } { T } = \widetilde { \mathcal { O } } \left( \epsilon ^ { \frac { 1 - q } { 4 - q } } \right) .\tag{19}
$$

For the squared exponential kernel, $\gamma _ { N }$ is polylogarithmic in N [Srinivas et al., 2012, Vakili et al., 2021, Iwazaki, 2025], so $q = 0$ up to logarithmic factors and

$$
\frac { \tilde { \gamma } _ { T } } { T } = \widetilde { \mathcal { O } } ( \epsilon ^ { 1 / 2 } ) , \qquad \frac { \mathbb { E } [ R _ { T } ] } { T } = \widetilde { \mathcal { O } } ( \epsilon ^ { 1 / 4 } ) .
$$

For a Matérn kernel with smoothness $\nu > 2$ , the static rate $q = d / ( 2 \nu + d )$ [Vakili et al., 2021, Iwazaki, 2025] gives

$$
\frac { \widetilde { \gamma } _ { T } } { T } = \widetilde { \mathcal { O } } \Bigl ( \epsilon ^ { \frac { 4 \nu } { 8 \nu + 3 d } } \Bigr ) , \qquad \frac { \mathbb { E } [ R _ { T } ] } { T } = \widetilde { \mathcal { O } } \Bigl ( \epsilon ^ { \frac { 2 \nu } { 8 \nu + 3 d } } \Bigr ) .
$$

These rates are not fixed- $T , \epsilon \to 0$ statements. If $N _ { \epsilon } > T$ , taking $\tilde { N } = T$ in Theorem B.3 gives

$$
\tilde { \gamma } _ { T } \leq \gamma _ { T } + \frac { \sigma ^ { - 2 } \epsilon T \sqrt { T ^ { 2 } - 1 } } { 2 \sqrt { 3 } } \sqrt { \gamma _ { T } } ,
$$

and hence $\tilde { \gamma } _ { T }  \gamma _ { T }$ for fixed T as $\epsilon \to 0$ . The constant-exploration result is therefore a positive-drift result and is not claimed to recover the usual static sublinear-regret guarantee at $\epsilon = 0 ;$ the static case requires a separate analysis [Takeno et al., 2025].

Comparison with previous rates. The information-gain improvement is separate from the confidence-schedule improvement. If $\gamma _ { N } = \widetilde { \mathcal { O } } ( N ^ { q } )$ , the original block perturbation in Bogunovic et al. [2016] gives the first rate below. The $\mathcal { O } ( \epsilon N ^ { 5 / 2 } )$ block order from the unit-time specialization of Imamura et al. [2020] gives the second, and Theorem B.3 gives the third:

$$
\frac { \widetilde { \gamma } _ { T } } { T } = \widetilde { \mathcal { O } } \left( \epsilon ^ { \frac { 1 - q } { 3 - q } } \right) , \qquad \widetilde { \mathcal { O } } \left( \epsilon ^ { \frac { 2 \left( 1 - q \right) } { 5 - 2 q } } \right) , \qquad \widetilde { \mathcal { O } } \left( \epsilon ^ { \frac { 2 \left( 1 - q \right) } { 4 - q } } \right) .
$$

The time-growing confidence schedule contributes a factor of order $\sqrt { \log ( T / \omega ) }$ to the leading regret term, whereas the prescribed-confidence constant- $- \beta$ choice uses

$$
\beta = \mathcal { O } ( \log ( 1 / \omega ) + \log ( 1 / \epsilon ) ) ,
$$

so the corresponding confidence factor is independent of $T$

## 4.2 Bound-suggested $\beta$ dependence on ϵ

Here $\beta$ denotes any constant exploration parameter satisfying (13), while $\beta _ { \mathrm { o p t } }$ denotes the boundsuggested value obtained from the scaling choice of the analytical parameters $( \delta , \tau )$ , rather than a policy-level optimal parameter. The drift parameter ϵ is already required to construct the $\mathrm { T V } { \bf - G P }$ posterior, so using it to guide $\beta$ introduces no additional model parameter. The scaling $\beta _ { \mathrm { o p t } } =$ ${ \bar { \mathcal { O } } } ( \log ( 1 / \epsilon ) )$ below is a bound-balancing recommendation rather than a validity condition; Theorem 1 holds for any constant $\beta$ satisfying (13). The analysis of Bogunovic et al. [2016] specifies the time-growing schedule

$$
\beta _ { t } = 2 \log \frac { \pi ^ { 2 } t ^ { 2 } } { 2 \delta } + 2 d \log \left( r d b t ^ { 2 } \sqrt { \log \frac { d a \pi ^ { 2 } t ^ { 2 } } { 2 \delta } } \right) ,\tag{20}
$$

where $( a , b , r , d )$ are spatial domain and smoothness constants, and δ is a failure probability. This schedule is determined by the spatial discretization argument and is not optimized as a function of the drift parameter $\epsilon .$

Our theorem does not prescribe $\beta$ as an explicit function of ϵ directly. Instead, ϵ enters through the time-decayed maximum information gain $\tilde { \gamma } _ { T }$ . Balancing the upper-bound terms through the analytical parameters $( \delta , \tau )$ therefore induces an ϵ-dependent bound-suggested constant $\beta _ { \mathrm { o p t } } ( \epsilon )$ . To make this dependence explicit, suppose that, for the chosen spatial kernel and in the long-horizon regime where $T$ exceeds the effective block length, the time-decayed information gain satisfies

$$
\tilde { \gamma } _ { T } = \widetilde { \mathcal { O } } ( T \epsilon ^ { \alpha } ) , \qquad \alpha > 0 .\tag{21}
$$

For fixed $T ,$ this scaling should not be interpreted as an $\epsilon  0$ limit, since $\tilde { \gamma } _ { T }$ then approaches the static information gain. Appendix B verifies such a scaling for common kernels through the standard growth rate of the time-invariant information gain. Taking equality in (13), $\beta = 2 ( \log ( \bar { 6 } / \delta ) + d \log \tau )$ while $C _ { 2 } = ( r d b / \tau ) \sqrt { \log ( 3 a d / \delta ) }$ . Thus the three per-round contributions in the expected regret bound are controlled by $\sqrt { \beta \widetilde { \gamma } _ { T } / T } , C _ { 2 }$ , and $\delta C _ { 3 }$

Under (21), the information-gain contribution per round scales as $\sqrt { \beta } \epsilon ^ { \alpha / 2 }$ up to logarithmic factors. For $\epsilon \in ( 0 , 1 )$ , the choice below balances the polynomial dependence on ϵ across the information-gain, discretization, and failure-event terms and is sufficient to identify the logarithmic dependence of $\beta _ { \mathrm { o p t } }$ on $1 / \epsilon \colon$

$$
\delta = \epsilon ^ { \alpha / 2 } , \qquad \tau = \Bigl \lceil \epsilon ^ { - \alpha / 2 } \Bigr \rceil .\tag{22}
$$

With this choice, the discretization and failure-event contributions match the same polynomial order, up to logarithmic factors:

$$
C _ { 2 } = \mathcal { O } \Big ( \epsilon ^ { \alpha / 2 } \sqrt { \log ( \epsilon ^ { - 1 } ) } \Big ) ,\tag{23}
$$

$$
\delta C _ { 3 } = { \mathcal O } \Bigl ( \epsilon ^ { \alpha / 2 } \sqrt { \log ( \epsilon ^ { - 1 } ) } \Bigr ) .\tag{24}
$$

Substituting (22) into (13) at equality gives

$$
\beta _ { \mathrm { o p t } } ( \epsilon ) \approx 2 \log ( 6 ) + \alpha ( 1 + d ) \log \Big ( \frac { 1 } { \epsilon } \Big ) .\tag{25}
$$

Across environments with the same spatial kernel but different drift rates, the bound suggests that $\beta _ { \mathrm { o p t } }$ grows logarithmically with $1 / \epsilon$ . This scaling should be interpreted together with the posterior variance: smaller ϵ also makes past data more informative and typically reduces $\sigma _ { t - 1 } ( x )$ .

## 4.3 Sampling frequency and effective drift

The same calculation gives a simple rule for changing the sampling frequency. Suppose the underlying process evolves at drift rate ϵ, but observations are collected every $\bar { \Delta } \in \mathbb { N } ^ { + }$ time units. Then the correlation between successive observations is $( \sqrt { 1 - \epsilon } ) ^ { \Delta }$ , which corresponds to the effective discretetime drift

$$
\epsilon _ { \Delta } = 1 - ( 1 - \epsilon ) ^ { \Delta } .\tag{26}
$$

Larger $\Delta$ increases $\epsilon _ { \Delta }$ , so past observations become less relevant. Substituting this effective drift into (25) yields the heuristic

$$
\beta _ { \mathrm { o p t } } ( \epsilon _ { \Delta } ) \propto \log \Big ( \frac { 1 } { \epsilon _ { \Delta } } \Big ) = \log \Big ( \frac { 1 } { 1 - ( 1 - \epsilon ) ^ { \Delta } } \Big ) .\tag{27}
$$

This adjustment reflects the fact that less frequent sampling increases the effective drift between observations and reduces the useful lifetime of past data.

## 4.4 Average-reward perspective

The dynamic setting also changes the interpretation of the benchmark. Let $M _ { t } : = \operatorname* { s u p } _ { x \in S } f _ { t } ( x )$ denote the round-t optimal reward. In the static case, M is the same random variable for all t, so the benchmark remains tied to a single realization of the objective. Regret bounds must therefore account for the joint distribution of the realized function and the algorithm’s decisions.

When $\epsilon > 0$ , this coupling is weaker in the long run. The sequence $( M _ { t } ) _ { t \geq 1 }$ is stationary and has exponentially decaying temporal dependence. As shown in Appendix $\mathrm { { C . l } }$

$$
\frac { 1 } { T } \sum _ { t = 1 } ^ { T } M _ { t } \xrightarrow [ T  \infty ] { \mathrm { P r } } \mathbb { E } [ M _ { 1 } ] .
$$

Thus the oracle part of the regret concentrates around a deterministic quantity. Consequently, asymptotic average regret can be viewed as the gap between the deterministic benchmark $\mathbb { E } [ M _ { 1 } ]$ and the algorithm’s realized average reward. Cumulative regret remains the relevant finite-horizon criterion, but this perspective separates the intrinsic variability of the moving optimum from the performance loss due to the algorithm’s decisions.

## 5 Simulation study

We empirically evaluate constant-β GP-UCB under the TV-GP model on the spatial domain ${ \boldsymbol { S } } =$ $[ 0 , 1 ] ^ { 2 }$ The domain is discretized using a $2 0 \times 2 0$ grid, giving a state dimension of $N = 4 0 0$ The latent function uses a radial basis function (RBF, squared exponential) kernel with lengthscale $l = 0 . 1$ and prior variance 1.0, while the observation noise variance is fixed at $\sigma ^ { 2 } = 0 . { \overset { \smile } { 0 } } 1$ . We simulate $T \doteq 1 0 , 0 0 0$ rounds for each parameter pair $( \epsilon , \beta )$ and record the time-averaged regret $R _ { T } / T$ over 100 independent trials. This horizon is used as a long-run stress test rather than as an application-specific benchmark; it is made feasible by the Kalman implementation described below. The simulation setup and compute resources are reported in Appendix E, and the accompanying code archive contains the scripts used to reproduce Figure 1.

Exact inference via Kalman filtering. A direct GP implementation would require repeatedly conditioning on the full observation history, making long-horizon simulations computationally expensive. To avoid this bottleneck, we exploit the autoregressive structure of the TV-GP prior. On the finite grid, the latent function can be represented as a state vector $f _ { t } \in \mathbb { R } ^ { 4 0 0 }$ , and the TV-GP model becomes a linear Gaussian state-space model. We therefore compute the GP posterior exactly using a Kalman filter. This reduces the per-round posterior update from the $\mathcal { O } ( t ^ { 3 } )$ history-dependent cost of naive GP regression to a fixed-grid $\mathcal { O } ( N ^ { 2 } )$ recursion. The full recursion is given in Appendix D.

Results. The left panel of Figure 1 displays the average regret $R _ { T } / T$ as a function of the constant exploration parameter $\beta$ for three drift rates, $\epsilon \in \{ 1 0 ^ { - 1 } , 1 0 ^ { - 2 } , 1 0 ^ { - 3 } \}$ . For each drift rate, let $\hat { \beta } _ { \mathrm { o p t } }$ denote the value of $\beta$ on the tested grid that minimizes the empirical average regret. The main empirical test is whether $\hat { \beta } _ { \mathrm { o p t } }$ increases with $- \log ( \epsilon )$ as predicted by Section 4.2. The empirical minimizer shifts toward larger values as ϵ decreases: $\hat { \beta } _ { \mathrm { o p t } } \approx 2 . 1 2$ for $\epsilon = 0 . 1 , \hat { \beta } _ { \mathrm { o p t } }$ ≈ 4.34 for $\epsilon = 0 . 0 1$ , and $\hat { \beta } _ { \mathrm { o p t } } \approx 6 . 3 6$ for $\epsilon = 0 . 0 0 1$

![](images/93d81a1b822fd3c8a77e298870dba5844a43e26e31c4b6fce62b24caf41cf59f.jpg)

![](images/1a9d334cd2dbcd87bd0170380df7926b6d13e42842e15155962b204c776d37bd.jpg)  
Figure 1: Simulation study for constant-β GP-UCB under the time-varying GP model. Left: timeaveraged regret $R _ { T } / T$ over $T = 1 0 . 0 0 0$ rounds as a function of the constant exploration parameter $\beta$ for three drift rates $\stackrel { \cdot } { \epsilon } \in \{ 1 0 ^ { - 1 } , 1 0 ^ { - 2 } , 1 0 ^ { - 3 } \}$ ; shaded bands denote 95% confidence intervals over 100 independent trials, and dashed horizontal lines show the logarithmic time-varying- $\cdot \beta _ { t }$ heuristic. Right: bootstrapped estimates of the empirical regret-minimizing constant $\hat { \beta } _ { \mathrm { o p t } }$ versus $- \log ( \epsilon )$ across the full ϵ grid. Points denote mean bootstrap estimates, vertical bars denote 95% percentile intervals, and the dashed line shows a least-squares linear fit used to visualize the log-linear trend.

Because the exact theoretical schedule in Equation (20) is overly conservative in practice [Srinivas et al., 2012, Bogunovic et al., 2016], empirical evaluations often use tuned logarithmic schedules. Following Bogunovic et al. [2016], we compare against $\beta _ { t } = c _ { 1 } \log ( c _ { 2 } t )$ with $c _ { 1 } = 0 . 8$ and $c _ { 2 } = 4$ Across all three drift rates, a broad interval of constant $\beta$ values achieves lower average regret than this time-growing heuristic over the 10,000-step horizon.

The right panel of Figure 1 examines this dependence across the full grid of drift rates. The relationship between $\hat { \beta } _ { \mathrm { o p t } }$ and − log(ϵ) is approximately linear, matching the qualitative prediction $\beta _ { \mathrm { o p t } } ( \epsilon ) \propto \bar { \log } ( 1 / \epsilon )$ from Equation (25). This agreement is nontrivial: the theory balances terms in an upper bound, and such bounds need not match the exploration scale that performs best in finite simulations.

We do not interpret the fitted slope as an estimate of the information-gain exponent α. Such an interpretation would require the analytical discretization choice $\tau = \tau ( \epsilon )$ and the global spatial union-bound factor in Equation (25) to be tight. Neither condition holds here: the simulation uses a fixed $2 0 \times 2 0$ action grid, and the adaptive policy concentrates its evaluations near high-reward regions rather than exploring the full domain uniformly. Thus the slope is best viewed as empirical support for the logarithmic dependence of the regret-minimizing constant on the drift rate, not as a quantitative estimate of the information-gain scaling.

## 6 Conclusion

Existing theoretical guarantees for Gaussian process bandits in time-varying environments typically require exploration schedules that grow with time. We showed that, under the TV-GP drift model, a constant exploration parameter can instead be analyzed using local confidence events at each round rather than a single horizon-wide confidence event. The resulting expected-regret bound has a drift-dependent leading term controlled by the time-varying information gain, and the same local-confidence analysis yields realized-regret guarantees with $\beta$ constant in t and T.

We also derived a sharper bound on the time-varying maximum information gain. $\mathrm { I f } \gamma _ { N } = \widetilde { \mathcal { O } } ( N ^ { q } )$ the resulting persistent-drift average-regret rate is $\widetilde { \mathcal { O } } ( \epsilon ^ { ( 1 - q ) / ( 4 - q ) } )$ ); for the squared exponential kernel this gives $\widetilde { \mathcal { O } } ( \epsilon ^ { 1 / 4 } )$ . These rates apply when the optimized block length fits within the horizon and are not fixed-T, ϵ → 0 statements. Separately, balancing the remaining terms in the regret bound suggests a bound-suggested constant exploration parameter with $\beta _ { \mathrm { o p t } } \bar { ( } \epsilon \bar { ) } \propto \log ( 1 / \epsilon )$ . This provides a simple tuning principle when changing the sampling frequency changes the effective drift between observations. The same Markov structure also implies exponential covariance decay for the oracle-value process, so long-run average performance can be compared against a deterministic benchmark.

Several extensions remain open. Most importantly, the present analysis relies on the geometric temporal kernel induced by the Markov drift model. Extending constant-exploration guarantees to more general temporal kernels would clarify which forms of nonstationarity are sufficient to prevent posterior overconfidence. Another important direction is to remove the assumption that the drift parameter is known and fixed, allowing $\beta$ to be adapted online when the rate of temporal variation is unknown or changes over time. Finally, sharper concentration tools for the realized regret of TV-GP-UCB could improve the current high-probability bounds and better reflect the empirical stability observed in the simulations.

## Acknowledgments and Disclosure of Funding

The first author gratefully acknowledges support from the NWO Spinoza Prize awarded to Aad van der Vaart.

## References

Robert J Adler and Jonathan E Taylor. Randomfields and geometry. Springer Science & Business Media, 2009.

Peter Auer, Nicolo Cesa-Bianchi, and Paul Fischer. Finite-time analysis of the multiarmed bandit problem. Machine learning, 47:235–256, 2002.

Anthony Bardou and Patrick Thiran. Time-varying Bayesian optimization without a metronome. arXiv preprint arXiv:2501.18963, 2025.

Anthony Bardou, Patrick Thiran, and Giovanni Ranieri. This too shall pass: Removing stale observations in dynamic Bayesian optimization. Advances in Neural Information Processing Systems, 37:42696–42737, 2024.

Ilija Bogunovic, Jonathan Scarlett, and Volkan Cevher. Time-varying Gaussian process bandit optimization. In Artificial Intelligence and Statistics, pages 314–323. PMLR, 2016.

Paul Brunzema, Alexander von Rohr, Friedrich Solowjow, and Sebastian Trimpe. Event-triggered time-varying Bayesian optimization. arXiv preprint arXiv:2208.10790, 2022.

Xu Cai and Jonathan Scarlett. Lower bounds for time-varying kernelized bandits. In Proceedings of The 28th International Conference on Artificial Intelligence and Statistics, volume 258 of Proceedings ofMachine Learning Research, pages 73–81. PMLR, 2025.

Roberto Calandra, André Seyfarth, Jan Peters, and Marc Peter Deisenroth. Bayesian optimization for learning gaits under uncertainty: An experimental comparison on a dynamic bipedal walker. Annals of Mathematics and Artificial Intelligence, 76(1):5–23, 2016.

Sayak Ray Chowdhury and Aditya Gopalan. On kernelized multi-armed bandits. In International Conference on Machine Learning, pages 844–853. PMLR, 2017.

Yuntian Deng, Xingyu Zhou, Baekjin Kim, Ambuj Tewari, Abhishek Gupta, and Ness Shroff. Weighted Gaussian process bandits for non-stationary environments. In International Conference on Artificial Intelligence and Statistics, pages 6909–6932. PMLR, 2022.

Subhashis Ghosal and Anindya Roy. Posterior consistency of Gaussian process prior for nonparametric binary regression. The Annals ofStatistics, 34(5):2413–2429, 2006.

Kihyuk Hong, Yuhang Li, and Ambuj Tewari. An optimization-based algorithm for non-stationary kernel bandits without prior knowledge. In International Conference on Artificial Intelligence and Statistics, pages 3048–3085. PMLR, 2023.

Hideaki Imamura, Nontawat Charoenphakdee, Futoshi Futami, Issei Sato, Junya Honda, and Masashi Sugiyama. Time-varying Gaussian process bandit optimization with non-constant evaluation time. arXiv preprint arXiv:2003.04691, 2020.

Shogo Iwazaki. Improved regret bounds for Gaussian process upper confidence bound in Bayesian optimization. Advances in Neural Information Processing Systems, 38:96922–96964, 2025.

Shogo Iwazaki and Shion Takeno. Near-optimal algorithm for non-stationary kernelized bandits. In Proceedings ofThe 28th International Conference on Artificial Intelligence and Statistics, volume 258 of Proceedings of Machine Learning Research, pages 406–414. PMLR, 2025.

Tze Leung Lai and Herbert Robbins. Asymptotically efficient adaptive allocation rules. Advances in applied mathematics, 6(1):4–22, 1985.

Herbert Robbins. Some aspects of the sequential design of experiments. Bulletin ofthe American Mathematical Society, 58(5):527–535, 1952.

Daniel Russo and Benjamin Van Roy. Learning to optimize via posterior sampling. Mathematics of Operations Research, 39(4):1221–1243, 2014.

Jonathan Scarlett, Ilija Bogunovic, and Volkan Cevher. Lower bounds on regret for noisy Gaussian process bandit optimization. In Conference on Learning Theory, pages 1723–1742. PMLR, 2017.

Jasper Snoek, Hugo Larochelle, and Ryan P Adams. Practical Bayesian optimization of machine learning algorithms. Advances in neural information processing systems, 25, 2012.

Niranjan Srinivas, Andreas Krause, Sham M Kakade, and Matthias W Seeger. Information-theoretic regret bounds for Gaussian process optimization in the bandit setting. IEEE transactions on information theory, 58(5):3250–3265, 2012.

Shion Takeno, Yu Inatsu, and Masayuki Karasuyama. Regret analysis for randomized Gaussian process upper confidence bound. Journal of Artificial Intelligence Research, 84(18), 2025. doi: 10.1613/jair.1.19393.

Sattar Vakili, Kia Khezeli, and Victor Picheny. On information gain and regret bounds in Gaussian process bandits. In International Conference on Artificial Intelligence and Statistics, pages 82–90. PMLR, 2021.

Sofía S Villar, Jack Bowden, and James Wason. Multi-armed bandit models for the optimal design of clinical trials: benefits and challenges. Statistical science: a review journal of the Institute of Mathematical Statistics, 30(2):199, 2015.

Xingyu Zhou and Ness Shroff. No-regret algorithms for time-varying Bayesian optimization. In 2021 55th Annual Conference on Information Sciences and Systems (CISS), pages 1–6. IEEE, 2021.

Juliusz Ziomek, Masaki Adachi, and Michael A Osborne. Time-varying Gaussian process bandits with unknown prior. arXiv preprint arXiv:2402.01632, 2024.

## A Improved Expected Regret Bound

This appendix proves Theorem 1. The argument proceeds in three steps: (i) bounding the instantaneous regret $r _ { t }$ for Algorithm 1 conditionally on a high-probability event; (ii) bounding the expected regret on the complement event via concentration inequalities for the extrema of Gaussian processes; and (iii) summing the expected regret over t and bounding the sum of predictive variances by the maximum information gain.

## A.1 Algorithm-specific bound

Lemma A.1 (Instantaneous regret under constant $\beta )$ . Consider the setting of Theorem 1. Fix $\delta \in ( 0 , 1 )$ and $\tau \in \mathbb { N } ^ { + }$ . Let $S ^ { * } \subset S$ be a discretization with $| S ^ { * } | \leq \tau ^ { d }$ satisfying

$$
\| x - [ x ] \| _ { 1 } \leq { \frac { r d } { \tau } } , \qquad f o r a l l \ x \in \mathcal { S } ,\tag{28}
$$

where $[ x ] \in S ^ { * }$ is an $\ell ^ { 1 }$ -closest grid point. Let $\beta$ and $C _ { 2 }$ be defined as in Theorem 1. ${ \cal I } f x _ { t }$ is selected according to Algorithm 1, thenfor eachfixed t, the instantaneous regret satisfies

$$
r _ { t } = f _ { t } ( x _ { t } ^ { * } ) - f _ { t } ( x _ { t } ) \ \leq \ 2 \sqrt { \beta } \sigma _ { t - 1 } ( x _ { t } ) + C _ { 2 }\tag{29}
$$

with probability at least $1 - \delta .$

Proof. We bound the instantaneous regret by conditioning on the intersection of three high-probability events at a fixed time t. Let $\mathcal { E } _ { t } = E _ { 1 , t } \bar { \cap } E _ { 2 , t } \cap E _ { 3 , t }$ , defined as follows:

(i) Pointwise concentration: Given the history $\mathcal { H } _ { t - 1 }$ , the query point $x _ { t }$ is deterministic. Because $f _ { t } ( x _ { t } ) \sim \mathcal { N } ( \mu _ { t - 1 } ( x _ { t } ) , \sigma _ { t - 1 } ^ { 2 } ( x _ { t } ) )$ , standard Gaussian tail bounds ensure the event

$$
| f _ { t } ( x _ { t } ) - \mu _ { t - 1 } ( x _ { t } ) | \leq \sqrt { \beta } \sigma _ { t - 1 } ( x _ { t } )\tag{30}
$$

holds with probability at least $1 - \delta / 3$ under the specified constant $\beta .$

(ii) Uniform grid concentration: Applying a union bound over the $| S ^ { * } | \le \tau ^ { d }$ elements of the discretization, the event

$$
| f _ { t } ( x ) - \mu _ { t - 1 } ( x ) | \leq { \sqrt { \beta } } \sigma _ { t - 1 } ( x ) , \quad { \mathrm { f o r ~ a l l ~ } } x \in { \mathcal { S } } ^ { * }\tag{31}
$$

holds with probability at least $1 - \delta / 3$

(iii) Lipschitz continuity: By the derivative tail assumption (11) and a union bound over the d spatial dimensions, setting $L = b \sqrt { \log ( 3 a d / \delta ) }$ ensures the event

$$
| f _ { t } ( x ) - f _ { t } ( x ^ { \prime } ) | \leq L \| x - x ^ { \prime } \| _ { 1 } , \quad { \mathrm { f o r ~ a l l ~ } } x , x ^ { \prime } \in S\tag{32}
$$

holds with an unconditional probability of at least $1 - \delta / 3$ under the prior measure.

Because events (i) and (ii) hold conditionally with probability $1 - \delta / 3$ for any history $\mathcal { H } _ { t - 1 }$ , the law of total expectation guarantees they also hold with an unconditional probability of at least $1 - \delta / 3 .$ . Applying a union bound over the unconditional probabilities of these three events yields $\mathbb { P } ( \mathcal { E } _ { t } ^ { c } ) \leq \delta ,$ guaranteeing the joint event holds unconditionally with $\mathbb { P } ( \mathcal { E } _ { t } ) \geq 1 - \delta$ . Conditioned on the realization of $\mathcal { E } _ { t }$ , the continuity event (iii) and the spatial discretization bound (28) guarantee that the approximation error for any point in the domain is bounded by $L ( r d / \tau ) = C _ { 2 }$

Evaluating this spatial bound at the instantaneous optimum $\boldsymbol { x } _ { t } ^ { * }$ yields $f _ { t } ( x _ { t } ^ { * } ) \leq f _ { t } ( [ x _ { t } ^ { * } ] ) + C _ { 2 }$ Because the projection $[ x _ { t } ^ { * } ] \in S ^ { * }$ , we apply the uniform concentration event (ii) to bound $f _ { t } ( [ x _ { t } ^ { * } ] )$ followed by the selection rule of Algorithm 1, which maximizes the upper confidence bound at $x _ { t } \colon$

$$
\begin{array} { r l } & { f _ { t } ( x _ { t } ^ { * } ) \leq \mu _ { t - 1 } ( [ x _ { t } ^ { * } ] ) + \sqrt { \beta } \sigma _ { t - 1 } ( [ x _ { t } ^ { * } ] ) + C _ { 2 } } \\ & { \qquad \leq \mu _ { t - 1 } ( x _ { t } ) + \sqrt { \beta } \sigma _ { t - 1 } ( x _ { t } ) + C _ { 2 } . } \end{array}\tag{33}
$$

Finally, applying the pointwise concentration event (i) at $x _ { t }$ yields

$$
\mu _ { t - 1 } ( x _ { t } ) \leq f _ { t } ( x _ { t } ) + \sqrt { \beta } \sigma _ { t - 1 } ( x _ { t } ) .
$$

Substituting this inequality into (33) bounds the instantaneous regret:

$$
r _ { t } = f _ { t } ( x _ { t } ^ { * } ) - f _ { t } ( x _ { t } ) \leq 2 \sqrt { \beta } \sigma _ { t - 1 } ( x _ { t } ) + C _ { 2 } ,
$$

which concludes the proof.

(34)

## A.2 Algorithm-agnostic tail control

The span $W _ { t }$ can be written as

$$
W _ { t } = \operatorname* { s u p } _ { x , y \in S } \big ( f _ { t } ( x ) - f _ { t } ( y ) \big ) .
$$

Thus $W _ { t }$ is the supremum of a centered Gaussian process indexed by $\mathcal { S } \times \mathcal { S } ,$ . Since the sample paths are almost surely bounded on the compact domain and $k _ { S } ( x , x ) \dot { \leq } 1$ , Borell–TIS implies that $\bar { W } _ { t } - \mathbb { E } [ W _ { t } ]$ has a sub-Gaussian upper tail with a finite variance proxy. We denote this proxy by $\sigma _ { w } ^ { 2 } .$

Lemma A.2 (Tail expectation bound). Assume $W _ { t } - \mu _ { w }$ has the sub-Gaussian upper tail

$$
\mathbb { P } \{ W _ { t } - \mu _ { w } > u \} \le \exp \left( - \frac { u ^ { 2 } } { 2 \sigma _ { w } ^ { 2 } } \right) \qquad f o r { a l l } u > 0 .
$$

Then, $f o r$ every event A with $\mathbb { P } ( A ) \leq \delta ,$

$$
\mathbb { E } [ W _ { t } \mathbf { 1 } _ { A } ] \le \delta \left( \mu _ { w } + 2 \sigma _ { w } \sqrt { 2 \log ( e / \delta ) } \right) .
$$

Proof. Write $X = ( W _ { t } - \mu _ { w } ) _ { + }$ . Then

$$
\mathbb { E } [ W _ { t } \mathbf { 1 } _ { A } ] \leq \mu _ { w } \mathbb { P } ( A ) + \mathbb { E } [ X \mathbf { 1 } _ { A } ] .
$$

Let $q = \sigma _ { w } \sqrt { 2 \log ( e / \delta ) }$ . We split

$$
{ \mathbb E } [ X \mathbf { 1 } _ { A } ] \leq q { \mathbb P } ( A ) + { \mathbb E } [ X \mathbf { 1 } _ { \{ X > q \} } ] .
$$

Using the sub-Gaussian tail and integrating,

$$
\mathbb { E } [ X \mathbf { 1 } _ { \{ X > q \} } ] = q \mathbb { P } ( X > q ) + \int _ { q } ^ { \infty } \mathbb { P } ( X > u ) d u .
$$

Since $q = \sigma _ { w } \sqrt { 2 \log ( e / \delta ) }$ , we have

$$
\mathbb { P } ( X > q ) \leq \exp \left( - \frac { q ^ { 2 } } { 2 \sigma _ { w } ^ { 2 } } \right) = \frac { \delta } { e } .
$$

Moreover, the standard Gaussian tail integral bound gives

$$
\int _ { q } ^ { \infty } \exp { \left( - \frac { u ^ { 2 } } { 2 \sigma _ { w } ^ { 2 } } \right) } \ d u \leq \frac { \sigma _ { w } ^ { 2 } } { q } \exp { \left( - \frac { q ^ { 2 } } { 2 \sigma _ { w } ^ { 2 } } \right) } = \frac { \sigma _ { w } ^ { 2 } } { q } \frac { \delta } { e } \leq q \frac { \delta } { e } ,
$$

where the last inequality uses $q \geq \sigma _ { w }$ . Therefore,

$$
\mathbb { E } [ X \mathbf { 1 } _ { \{ X > q \} } ] \leq 2 q \frac \delta e \leq q \delta .
$$

Combining this with $q \mathbb { P } ( A ) \leq q \delta$ yields

$$
\mathbb { E } [ X \mathbf { 1 } _ { A } ] \leq 2 q \delta .
$$

## A.3 Expected cumulative regret

ProofofTheorem 1. Fix $\delta \in \mathsf { \Gamma } ( 0 , 1 )$ and $\tau \in \mathbb { N } ^ { + }$ , and define $\beta$ and $C _ { 2 }$ as in Theorem 1. By Lemma A.1, for each time step t there exists a confidence event $\mathcal { E } _ { t }$ with $\mathbb { P } ( \mathcal { E } _ { t } ) \geq 1 - \delta$ such that, conditioned on $\mathcal { E } _ { t } ,$ the instantaneous regret satisfies $r _ { t } \le 2 \sqrt { \beta } \sigma _ { t - 1 } ( x _ { t } ) + C _ { 2 }$

Because instantaneous regret is always non-negative, we can bound it pathwise over the entire probability space by introducing the spatial span $\begin{array} { r } { \mathbf { \bar { \phi } } W _ { t } = \operatorname* { m a x } _ { x } f _ { t } ( x ) - \operatorname* { m i n } _ { x } f _ { t } ( x ) } \end{array}$ as the worst-case penalty when the confidence event fails:

$$
r _ { t } \ \leq \ 2 \sqrt { \beta } \sigma _ { t - 1 } ( x _ { t } ) + C _ { 2 } + W _ { t } \mathbb { 1 } _ { \mathcal { E } _ { t } ^ { c } } ,\tag{35}
$$

where $\mathbb { 1 } _ { \mathcal { E } _ { t } ^ { c } }$ is the indicator for the complement event. Taking the unconditional expectation of both sides, and applying Lemma A.2 to bound the expected tail contribution $\mathbb { E } [ W _ { t } \mathbb { 1 } _ { \mathcal { E } _ { t } ^ { c } } ] \le \delta C _ { 3 }$ , yields the valid deterministic bound:

$$
\begin{array} { r } { \mathbb { E } [ r _ { t } ] \ \leq \ 2 \sqrt { \beta } \mathbb { E } [ \sigma _ { t - 1 } ( x _ { t } ) ] + C _ { 2 } + \delta C _ { 3 } . } \end{array}\tag{36}
$$

Summing (36) over $t = 1 , \dots , T$ , we obtain:

$$
\mathbb { E } [ R _ { T } ] \le 2 \sqrt { \beta } \mathbb { E } \Bigg [ \sum _ { t = 1 } ^ { T } \sigma _ { t - 1 } ( x _ { t } ) \Bigg ] + T C _ { 2 } + T \delta C _ { 3 } .\tag{37}
$$

To bound the sum of the posterior standard deviations, we apply the Cauchy-Schwarz inequality strictly inside the expectation:

$$
\mathbb { E } \left[ \sum _ { t = 1 } ^ { T } \sigma _ { t - 1 } ( x _ { t } ) \right] \ \leq \ \mathbb { E } \left[ \sqrt { T \sum _ { t = 1 } ^ { T } \sigma _ { t - 1 } ^ { 2 } ( x _ { t } ) } \right] .\tag{38}
$$

We then bound the sum of predictive variances by the maximum information gain. For every realized query sequence $x _ { 1 : T }$ , the Schur-complement determinant identity gives

$$
\frac { 1 } { 2 } \sum _ { t = 1 } ^ { T } \log \left( 1 + \sigma ^ { - 2 } \sigma _ { t - 1 } ^ { 2 } ( x _ { t } ) \right) = \frac { 1 } { 2 } \log \operatorname* { d e t } \left( I _ { T } + \sigma ^ { - 2 } \tilde { K } _ { T } ( x _ { 1 : T } ) \right) \leq \tilde { \gamma } _ { T } .\tag{39}
$$

Combining this pathwise identity with the variance-to-logarithm inequality from Srinivas et al. [2012, Lemma 5.4] under the assumption that $k _ { S } ( x , x ) \leq 1$ yields almost surely

$$
\sum _ { t = 1 } ^ { T } \sigma _ { t - 1 } ^ { 2 } ( x _ { t } ) \leq \frac { 2 } { \log ( 1 + \sigma ^ { - 2 } ) } \tilde { \gamma } _ { T } .\tag{40}
$$

Defining $C _ { 1 } = 8 / \log ( 1 + \sigma ^ { - 2 } )$ , we have $\begin{array} { r } { \sum _ { t = 1 } ^ { T } 4 \beta \sigma _ { t - 1 } ^ { 2 } ( x _ { t } ) \le C _ { 1 } \beta \tilde { \gamma } _ { T } } \end{array}$ . Because $\tilde { \gamma } _ { T }$ is defined as the maximum information gain over all possible sequences, it is a deterministic upper bound. The expectation of a deterministic bound is simply the bound itself. Substituting this into the cumulative sum yields the final result:

$$
\mathbb { E } [ R _ { T } ] \leq \sqrt { C _ { 1 } T \beta \tilde { \gamma } _ { T } } + T C _ { 2 } + T \delta C _ { 3 } .\tag{41}
$$

The explicit dependence of $\tilde { \gamma } _ { T }$ on $T$ and ϵ for the time-decayed kernel is detailed in Appendix B.

## B Maximum Information Gain

Theorem B.3 (Time-varying maximum information gain). Let $\tilde { \gamma } _ { T } : = \operatorname* { s u p } _ { x _ { 1 : T } \in S ^ { T } } \frac { 1 } { 2 }$ log det $( I _ { T } +$ $\sigma ^ { - 2 } \tilde { K } _ { T } )$ denote the maximum information gain under the separable spatio-temporal kernel $\tilde { k } ( ( x , t ) , ( x ^ { \prime } , t ^ { \prime } ) ) = k _ { S } ( x , x ^ { \prime } ) \rho ^ { | t - t ^ { \prime } | }$ with $\rho = \sqrt { 1 - \epsilon } \in [ 0 , 1 )$ . Assume $| k _ { S } ( x , x ^ { \prime } ) | \ \leq \ 1$ for all $x , x ^ { \prime } \in S .$ . Thenfor any block length $\tilde { N } \in \mathbb { N } ^ { + }$

$$
\begin{array} { r l r } { \tilde { \gamma } _ { T } \le \left( \left[ \frac { T } { \tilde { N } } \right] \right) \left( \gamma _ { \tilde { N } } + \frac { \sigma ^ { - 2 } \epsilon \tilde { N } \sqrt { \tilde { N } ^ { 2 } - 1 } } { 2 \sqrt { 3 } } \sqrt { \gamma _ { \tilde { N } } } \right) \le } & { \left( \frac { T } { \tilde { N } } + 1 \right) \left( \gamma _ { \tilde { N } } + \frac { \sigma ^ { - 2 } \epsilon \tilde { N } \sqrt { \tilde { N } ^ { 2 } - 1 } } { 2 \sqrt { 3 } } \sqrt { \gamma _ { \tilde { N } } } \right) , } \end{array}\tag{42}
$$

where $\gamma _ { \tilde { N } } : = \operatorname* { s u p } _ { x _ { 1 } . . . \tilde { N } } \in S ^ { \tilde { N } } ^ { \mathrm { ~ \frac { 1 } { 2 } ~ } }$ log det $( I _ { \tilde { N } } + \sigma ^ { - 2 } K _ { \tilde { N } } )$ is the standard time-invariant maximum information gain for $f \sim \mathcal { G P } ( 0 , k _ { S } )$

Proof. Step 1 (block decomposition). Partition the sequence $\{ 1 , \ldots , T \}$ into $m = \lceil T / \tilde { N } \rceil$ consecutive blocks, each of length at most $\tilde { N }$ . This partition is used only in the analysis; the GP posterior is

neither reset nor truncated at block boundaries. Let $( \mathbf { f } ^ { ( i ) } , \mathbf { y } ^ { ( i ) } )$ denote the latent function values and noisy observations within block i. By the chain rule for mutual information,

$$
I ( \mathbf { f } _ { T } ; \mathbf { y } _ { T } ) = \sum _ { i = 1 } ^ { m } I ( \mathbf { f } _ { T } ; \mathbf { y } ^ { ( i ) } \mid \mathbf { y } ^ { ( < i ) } ) .
$$

Because $\mathbf { y } ^ { ( i ) } = \mathbf { f } ^ { ( i ) } + \mathbf { z } ^ { ( i ) }$ with $\mathbf { z } ^ { ( i ) }$ drawn independently of $\left( \mathbf { f } _ { T } , \mathbf { y } ^ { ( < i ) } \right)$ , the variables satisfy the Markov property $\mathbf { y } ^ { ( i ) }$ ⊥⊥ $\left( \mathbf { f } _ { T } \setminus \mathbf { f } ^ { ( i ) } \right) \mid \mathbf { f } ^ { ( i ) }$ . Consequently, $\bar { I } ( \mathbf { f } _ { T } ; \mathbf { y } ^ { ( i ) } \mid \mathbf { y } ^ { ( < i ) } ) = I ( \mathbf { f } ^ { ( i ) } ; \mathbf { y } ^ { ( i ) } \mid \mathbf { y } ^ { ( < i ) } )$ .

Furthermore, because the additive Gaussian noise $\mathbf { z } ^ { ( i ) }$ is independent of the sequence history, the conditional entropy simplifies to $H ( \mathbf { y } ^ { ( i ) } \mid \mathbf { f } ^ { ( i ) } , \mathbf { y } ^ { ( < i ) } ) = \bar { H ( \mathbf { z } ^ { ( i ) } ) }$ . While conditioning reduces entropy $( H ( \mathbf { y } ^ { ( i ) } \mid \bar { \mathbf { y } } ^ { ( < i ) } ) \overset { \cdot } { \leq } H ( \mathbf { y } ^ { ( i ) } ) )$ , it does not universally reduce mutual information. However, due to this independent noise structure, the explicit algebraic cancellation holds:

$$
I ( \mathbf { f } ^ { ( i ) } ; \mathbf { y } ^ { ( i ) } \mid \mathbf { y } ^ { ( < i ) } ) = H ( \mathbf { y } ^ { ( i ) } \mid \mathbf { y } ^ { ( < i ) } ) - H ( \mathbf { z } ^ { ( i ) } ) \leq H ( \mathbf { y } ^ { ( i ) } ) - H ( \mathbf { z } ^ { ( i ) } ) = I ( \mathbf { f } ^ { ( i ) } ; \mathbf { y } ^ { ( i ) } ) .\tag{43}
$$

It follows that

$$
I ( \mathbf { f } _ { T } ; \mathbf { y } _ { T } ) \leq \sum _ { i = 1 } ^ { m } I ( \mathbf { f } ^ { ( i ) } ; \mathbf { y } ^ { ( i ) } ) .
$$

Taking the supremum over the choice of locations $\boldsymbol { x } _ { 1 : T } \in \mathcal { S } ^ { T }$ yields $\begin{array} { r } { \tilde { \gamma } _ { T } \leq \sum _ { i = 1 } ^ { m } \tilde { \gamma } _ { n _ { i } } \leq m \tilde { \gamma } _ { \tilde { N } } } \end{array}$ where $n _ { i } \leq \tilde { N }$ is the size of block i and the sequence $\tilde { \gamma } _ { n }$ is monotonically non-decreasing.

Step 2 (one-block bound). Fix a block of length $\tilde { N }$ . Define its spatial kernel matrix as $K : = K _ { \tilde { N } }$ and its temporal correlation matrix as $D : = D _ { \tilde { N } }$ , where $D _ { i j } = \rho ^ { | i - j | }$ . The spatio-temporal covariance over this block is the Hadamard product ${ \tilde { K } } : = K \circ D$ . By the Schur product theorem, $\tilde { K }$ is positive semi-definite because both $K \succeq 0$ and $D \succeq 0$ . Let 1 denote the $\tilde { N } \times \tilde { N }$ all-ones matrix and define the difference matrix $A : = \tilde { K } - K = K \circ ( D - { \bf 1 } )$ . Let

$$
M : = I _ { \tilde { N } } + \sigma ^ { - 2 } K \succ 0 , \qquad E : = \sigma ^ { - 2 } A ,
$$

so that $I _ { \tilde { N } } + \sigma ^ { - 2 } \tilde { K } = M + E \succ 0$ . Applying the congruence factorization $M + E = M ^ { 1 / 2 } \big ( I _ { \tilde { N } } +$ $B ) M ^ { 1 / 2 }$ with $B : = M ^ { - 1 / 2 } E M ^ { - 1 / 2 }$ yields

$$
{ \textstyle \frac { 1 } { 2 } } \log \operatorname* { d e t } ( M + E ) = { \textstyle \frac { 1 } { 2 } } \log \operatorname* { d e t } ( M ) + { \textstyle \frac { 1 } { 2 } } \log \operatorname* { d e t } ( I _ { \tilde { N } } + B ) .
$$

The matrix $B$ is symmetric and satisfies $I _ { \tilde { N } } + B \succ 0$ . Therefore, every eigenvalue of $B$ is strictly greater than $- 1$ . Using the scalar inequality $\log ( 1 + u ) \ \leq \ u$ for all u $> - 1$ , we bound the log-determinant term by the trace:

$$
\log \operatorname* { d e t } ( I _ { \tilde { N } } + B ) = \sum _ { j = 1 } ^ { \tilde { N } } \log \big ( 1 + \lambda _ { j } ( B ) \big ) \leq \sum _ { j = 1 } ^ { \tilde { N } } \lambda _ { j } ( B ) = \mathrm { t r } ( B ) .
$$

Substituting this trace inequality provides the upper bound

$$
\textstyle { \frac { 1 } { 2 } } \log \operatorname* { d e t } ( M + E ) \leq { \frac { 1 } { 2 } } \log \operatorname* { d e t } ( M ) + { \frac { 1 } { 2 } } \operatorname { t r } ( B ) .
$$

By the cyclicity of the trace operator, t $\operatorname { r } ( B ) = \operatorname { t r } ( M ^ { - 1 } E )$ . Since $D _ { i i } = 1$ , the diagonal entries of $\dot { A = K \circ ( D - { \bf 1 } ) }$ vanish, and hence $\mathrm { t r } ( E ) { } = 0$ . Therefore,

$$
\operatorname { t r } ( B ) = \operatorname { t r } \big ( ( M ^ { - 1 } - I _ { \tilde { N } } ) E \big ) .
$$

Using the Frobenius inner product,

$$
\mathrm { t r } ( B ) \leq \left| \mathrm { t r } \big ( ( M ^ { - 1 } - I _ { \tilde { N } } ) E \big ) \right| \leq \| M ^ { - 1 } - I _ { \tilde { N } } \| _ { F } \| E \| _ { F } .
$$

Write

$$
\Gamma ( K ) : = \frac 1 2 \log \operatorname * { d e t } ( I _ { \tilde { N } } + \sigma ^ { - 2 } K ) .
$$

Let $\lambda _ { 1 } , \ldots , \lambda _ { \tilde { N } }$ be the eigenvalues of $K$ and set $u _ { j } = \sigma ^ { - 2 } \lambda _ { j } \geq 0$ . Then

$$
\| M ^ { - 1 } - I _ { \tilde { N } } \| _ { F } ^ { 2 } = \sum _ { j = 1 } ^ { \tilde { N } } \left( \frac { u _ { j } } { 1 + u _ { j } } \right) ^ { 2 } \le \sum _ { j = 1 } ^ { \tilde { N } } \log ( 1 + u _ { j } ) = 2 \Gamma ( K ) .
$$

Here the inequality follows from

$$
\left( { \frac { u } { 1 + u } } \right) ^ { 2 } \leq \log ( 1 + u ) , \qquad u \geq 0 ,
$$

since the difference has value zero at $u = 0$ and derivative $( 1 + u ^ { 2 } ) / ( 1 + u ) ^ { 3 } > 0$ . Hence

$$
\| M ^ { - 1 } - I _ { \tilde { N } } \| _ { F } \leq \sqrt { 2 \Gamma ( K ) } .
$$

It remains to bound $\| E \| _ { F }$ . As above,

$$
| D _ { i j } - 1 | = 1 - \rho ^ { | i - j | } \leq \epsilon | i - j | .
$$

Since $| K _ { i j } | \le 1$

$$
\| A \| _ { F } ^ { 2 } \le \epsilon ^ { 2 } \sum _ { i , j = 1 } ^ { \tilde { N } } ( i - j ) ^ { 2 } = \frac { \epsilon ^ { 2 } \tilde { N } ^ { 2 } ( \tilde { N } ^ { 2 } - 1 ) } { 6 } ,
$$

and therefore

$$
\| E \| _ { F } = \sigma ^ { - 2 } \| A \| _ { F } \leq \frac { \sigma ^ { - 2 } \epsilon \tilde { N } \sqrt { \tilde { N } ^ { 2 } - 1 } } { \sqrt { 6 } } .
$$

Combining the two norm bounds gives

$$
\frac { 1 } { 2 } \log \operatorname* { d e t } ( I _ { \tilde { N } } + \sigma ^ { - 2 } \tilde { K } ) \le \Gamma ( K ) + \frac { \sigma ^ { - 2 } \epsilon \tilde { N } \sqrt { \tilde { N } ^ { 2 } - 1 } } { 2 \sqrt { 3 } } \sqrt { \Gamma ( K ) } .
$$

Since $s \mapsto s + c \sqrt { s }$ is increasing for $s \geq 0$ and $\Gamma ( K ) \le \gamma _ { \tilde { N } }$

$$
\tilde { \gamma } _ { \tilde { N } } \leq \gamma _ { \tilde { N } } + \frac { \sigma ^ { - 2 } \epsilon \tilde { N } \sqrt { \tilde { N } ^ { 2 } - 1 } } { 2 \sqrt { 3 } } \sqrt { \gamma _ { \tilde { N } } } .
$$

Summing this bound over the m blocks in Step 1 completes the proof.

Remark. The block decomposition follows the approach of Bogunovic et al. [2016]. For unitspaced observations, an $\mathcal { O } ( \epsilon \tilde { N } ^ { 5 / 2 } )$ block perturbation also follows from the more general analysis of Imamura et al. [2020]. Without the zero-trace step above, the congruence argument recovers this order directly by combining $\lVert M ^ { - 1 } \rVert _ { F } \leq \sqrt { \tilde { N } }$ with $\lVert A \rVert _ { F } \le \epsilon \tilde { N } ^ { 2 }$ , without requiring sign restrictions on the perturbation. The additional identity $\operatorname { t r } ( E ) { \overset { } { = } } 0$ allows $\dot { M } ^ { - 1 }$ to be replaced by $M ^ { - 1 } - I _ { \tilde { N } }$ yielding the sharper bound in Theorem B.3.

## B.1 Asymptotic Scaling with Drift

Theorem B.3 gives a two-regime dependence on the drift rate once the block length is chosen according to the growth of the time-invariant maximum information gain.

Corollary B.4 (Information gain scaling). Suppose the spatial kernel admits a time-invariant maximum information gain satisfying

$$
\gamma _ { \tilde { N } } = \widetilde { \mathcal { O } } ( \tilde { N } ^ { q } )
$$

for some capacity exponent $q \in [ 0 , 1 )$ . Define the polynomial block scale

$$
N _ { \epsilon } : = \epsilon ^ { - \frac { 2 } { 4 - q } } ,
$$

and choose

$$
\tilde { N } = \operatorname* { m i n } \left\{ T , \left\lceil N _ { \epsilon } \right\rceil \right\} .
$$

Then

$$
\tilde { \gamma } _ { T } = \widetilde { \mathcal { O } } \left( T ^ { q } + T \epsilon ^ { \frac { 2 ( 1 - q ) } { 4 - q } } \right) .\tag{44}
$$

In particular, in the persistent-drift regime $T \gtrsim N _ { \epsilon }$

$$
\tilde { \gamma } _ { T } = \widetilde { \mathcal { O } } \left( T \epsilon ^ { \frac { 2 ( 1 - q ) } { 4 - q } } \right) ,\tag{45}
$$

whereas for $T \lesssim N _ { \epsilon }$ one has

$$
\tilde { \gamma } _ { T } = \widetilde { \mathcal { O } } ( T ^ { q } ) .
$$

For fixed $T ,$ choosing $\tilde { N } = T$ in Theorem B.3 gives

$$
\tilde { \gamma } _ { T } \leq \gamma _ { T } + \frac { \sigma ^ { - 2 } \epsilon T \sqrt { T ^ { 2 } - 1 } } { 2 \sqrt { 3 } } \sqrt { \gamma _ { T } } ,\tag{46}
$$

so the perturbation vanishes as $\epsilon  0$ and the static information gain is recovered.

Proof. For $\tilde { N } \leq T$ , Theorem B.3 and $\sqrt { \tilde { N } ^ { 2 } - 1 } \leq \tilde { N }$ give

$$
\tilde { \gamma } _ { T } = \widetilde { \mathcal { O } } \left( \frac { T } { \tilde { N } } \left( \tilde { N } ^ { q } + \epsilon \tilde { N } ^ { 2 + q / 2 } \right) \right) = \widetilde { \mathcal { O } } \left( T \tilde { N } ^ { q - 1 } + T \epsilon \tilde { N } ^ { 1 + q / 2 } \right) .
$$

Balancing the polynomial orders gives

$$
\tilde { N } ^ { q - 1 } \asymp \epsilon \tilde { N } ^ { 1 + q / 2 } , \qquad \mathrm { o r ~ e q u i v a l e n t l y } \qquad \tilde { N } \asymp \epsilon ^ { - \frac { 2 } { 4 - q } } = N _ { \epsilon } .
$$

Thus, when $T \gtrsim N _ { \epsilon }$ , choosing $\tilde { N }$ at this scale gives

$$
\tilde { \gamma } _ { T } = \widetilde { \mathcal { O } } \left( T \epsilon ^ { \frac { 2 ( 1 - q ) } { 4 - q } } \right) .
$$

When $T \lesssim N _ { \epsilon }$ , choose $\tilde { N } = T$ . Equation (46) and $\gamma _ { T } = { \widetilde O } ( T ^ { q } )$ give

$$
\tilde { \gamma } _ { T } = \widetilde { \mathcal { O } } \left( T ^ { q } + \epsilon T ^ { 2 + q / 2 } \right) .
$$

The condition $T \lesssim N _ { \epsilon }$ implies $\epsilon T ^ { 2 - q / 2 } \lesssim 1$ , so the perturbation term is at most $\widetilde { \mathcal { O } } ( T ^ { q } )$ . Combining the two regimes yields (44). 口

For fixed T, the temporal matrix satisfies $D _ { i j } \to 1$ as $\epsilon \to 0$ . For the continuous kernels considered here on the compact domain $s ,$ continuity of the log determinant under maximization therefore gives $\tilde { \gamma } _ { T }  \gamma _ { T }$

## C Concentration of $R _ { T }$

This appendix analyzes the concentration properties of the time-varying setting. Section C.1 establishes the convergence of the oracle-value process via exponential covariance decay, which justifies the asymptotic average reward perspective discussed in Section 4.4. Section C.2 proves the spanonly concentration fallback, and Section C.3 proves the algorithm-specific realized-regret bound in Proposition 2.

## C.1 Concentration of the oracle-value process via covariance decay

Define $\begin{array} { r } { M _ { t } : = \operatorname* { s u p } _ { x \in \mathcal { S } } f _ { t } ( x ) } \end{array}$ for $t \geq 1$ . Under the TV-GP dynamics, the sequence $\{ f _ { t } \}$ is strictly stationary with each $\mathsf { \bar { f } } _ { t } \sim \mathcal G \mathcal P ( 0 , k _ { S } )$ . Consequently, $\{ M _ { t } \}$ is stationary with a constant expectation $\mathbb { E } [ M _ { t } ]$ . Assume the functions $\dot { f } _ { t }$ have almost surely bounded sample paths on $s$ and are separable, implying $M _ { t } < \infty$ almost surely. By the Borell–TIS inequality [Adler and Taylor, 2009, Theorem 2.1.1], $M _ { t }$ is sub-Gaussian around its mean:

$$
\operatorname* { P r } \bigl ( | M _ { t } - \mathbb { E } [ M _ { t } ] | > u \bigr ) \leq 2 \exp \Bigl ( - \frac { u ^ { 2 } } { 2 \sigma _ { M } ^ { 2 } } \Bigr ) , \qquad \sigma _ { M } ^ { 2 } : = \operatorname* { s u p } _ { x \in \mathcal { S } } \mathrm { V a r } \bigl ( f _ { t } ( x ) \bigr ) \leq 1 .
$$

This concentration ensures $\mathbb { E } [ M _ { t } ^ { 2 } ] < \infty$ . The following generic lemma controls the variance of a sum of random variables that exhibit exponentially decaying covariance.

Lemma C.5 (Averaging under exponential covariance decay). Let $\{ X _ { t } \} _ { t \ge 1 }$ be random variables with $\mathbb { E } [ X _ { t } ^ { 2 } ] < \infty$ and suppose there exist $m > 0$ and $c \in [ 0 , 1 )$ ) such that $\bar { \mathrm { C o v } } ( X _ { i } , X _ { j } ) \le m c ^ { | i - j | }$ for all i, j. Let $\begin{array} { r } { S _ { T } : = \sum _ { t = 1 } ^ { T } X _ { t } } \end{array}$ . Then:

(i) $\begin{array} { r } { \mathrm { V a r } ( S _ { T } ) \leq m \Big ( T + 2 \sum _ { h = 1 } ^ { T - 1 } ( T - h ) c ^ { h } \Big ) \leq m T \frac { 1 + c } { 1 - c } } \end{array}$ <sup>c</sup> .

(ii) For any $\eta > 0 ,$

$$
\operatorname* { P r } \left( \left| \frac { S _ { T } } { T } - \frac { \mathbb { E } [ S _ { T } ] } { T } \right| > \eta \right) \le \frac { m } { T \eta ^ { 2 } } \cdot \frac { 1 + c } { 1 - c } \xrightarrow [ T \to \infty ] { } 0 ,
$$

hence $\begin{array} { r } { \frac { S _ { T } } { T } - \frac { \mathbb { E } [ S _ { T } ] } { T }  0 } \end{array}$ in probability.

(iii) For any $\omega \in ( 0 , 1 )$

$$
S _ { T } \le \mathbb { E } [ S _ { T } ] + \sqrt { \mathrm { V a r } ( S _ { T } ) / \omega } \le \mathbb { E } [ S _ { T } ] + \sqrt { \frac { m T } { \omega } } \cdot \frac { 1 + c } { 1 - c } w i t h p r o b a b i l i t y a t l e a s t 1 - \omega .
$$

Proof. By bilinearity of covariance, grouping terms by lag $h = | i - j |$ yields

$$
\operatorname { V a r } ( S _ { T } ) = \sum _ { i = 1 } ^ { T } \sum _ { j = 1 } ^ { T } \operatorname { C o v } ( X _ { i } , X _ { j } ) \leq m \sum _ { i = 1 } ^ { T } \sum _ { j = 1 } ^ { T } c ^ { | i - j | } = m { \Big ( } T + 2 \sum _ { h = 1 } ^ { T - 1 } ( T - h ) c ^ { h } { \Big ) } .
$$

Since $( T - h ) \leq T$ and $\textstyle \sum _ { h = 1 } ^ { \infty } c ^ { h } = c / ( 1 - c )$ , we obtain

$$
T + 2 \sum _ { h = 1 } ^ { T - 1 } ( T - h ) c ^ { h } \leq T + 2 T \sum _ { h = 1 } ^ { \infty } c ^ { h } = T { \frac { 1 + c } { 1 - c } } ,
$$

proving (i). Items (ii) and (iii) follow by Chebyshev’s inequality applied to $S _ { T } / T$ and $S _ { T }$ , respectively. □

The next lemma bounds the covariance of suprema under the canonical Gaussian autoregressive coupling that underlies the TV-GP recursion.

Lemma C.6 (Covariance contraction for Gaussian suprema). Let f and g be centered, separable Gaussian processes indexed by a set $T ,$ with $g ( t ) = \rho f ( t ) + \sqrt { 1 - \rho ^ { 2 } } h ( t ) f o r \rho \in [ 0 , 1 ]$ , where h is an independent copy of f. Assume su $\ u _ { t \in T } \mathrm { V a r } ( f ( t ) ) \leq \sigma _ { T } ^ { 2 } <$ ∞ and $\textstyle \operatorname* { s u p } _ { t \in T } | f ( t ) | <$ ∞ almost surely. Then Cov $\begin{array} { r } { \big ( \operatorname* { s u p } _ { t \in T } f ( t ) , \operatorname* { s u p } _ { t \in T } g ( t ) \big ) \leq \rho \sigma _ { T } ^ { 2 } } \end{array}$

Proof. We first prove the claim for a finite set $T = \{ 1 , \ldots , k \}$ and then pass to a general separable T. Let $A = ( { \dot { f } } ( 1 ) , \dots , f ( k ) )$ and $C = ( g ( 1 ) , \ldots , \bar { g } ( k ) )$ . By construction, the joint distribution of $( A , C )$ is a zero-mean Gaussian with cross-covariance Cov $( A _ { i } , C _ { j } ) = \rho \Sigma _ { i j } ^ { - }$ , where $\Sigma _ { i j } =$ $\operatorname { C o v } ( A _ { i } , A _ { j } )$

Let $\begin{array} { r } { \phi _ { n } ( u ) : = \frac { 1 } { n } \log \big ( \sum _ { i = 1 } ^ { k } e ^ { n u _ { i } } \big ) } \end{array}$ be the smooth approximation of the maximum function. The gradient $\nabla \phi _ { n } ( u )$ is the softmax function, ensuring that for any u, $\nabla \phi _ { n } ( u )$ is a probability vector $( \mathbf { \bar { \| } } \nabla \phi _ { n } ( u ) \mathbf { \| _ { 1 } } = \dot { 1 }$ with strictly non-negative components).

To compute the covariance $\operatorname { C o v } ( \phi _ { n } ( A ) , \phi _ { n } ( C ) )$ , we employ the Gaussian interpolation technique standard in the proofs of Gaussian comparison inequalities [see, e.g., Adler and Taylor, 2009, Section 2.2]. Consider the interpolation path parameterized by $t \in [ 0 , 1 ]$ with the block covariance matrix

$$
\Gamma ( t ) = \left( { { \sum _ { { t \rho \Sigma } } } ^ { { t \rho \Sigma } } { \bf \Sigma } } \right) .
$$

Let $( A ( t ) , C ( t ) ) \ \sim \ { \mathcal { N } } ( 0 , \Gamma ( t ) )$ For $t ~ = ~ 0 , ~ A ( 0 )$ and $C ( 0 )$ are independent, implying $\mathrm { C o v } ( \dot { \phi } _ { n } ( A ( 0 ) ) , \dot { \phi } _ { n } ( C ( 0 ) ) ) = \ddot { 0 . }$ For $t = 1$ , we recover the joint distribution of $( A , C )$ . Differentiating the expectation of $\phi _ { n } ( A ( t ) ) \phi _ { n } ( C ( t ) )$ with respect to t and applying multivariate Gaussian integration by parts [Adler and Taylor, 2009, Lemma 2.1.4] yields:

$$
\frac { d } { d t } \mathbb { E } [ \phi _ { n } ( A ( t ) ) \phi _ { n } ( C ( t ) ) ] = \sum _ { i = 1 } ^ { k } \sum _ { j = 1 } ^ { k } \frac { d \Gamma _ { A C , i j } ( t ) } { d t } \mathbb { E } \bigg [ \frac { \partial \phi _ { n } ( A ( t ) ) } { \partial A _ { i } } \frac { \partial \phi _ { n } ( C ( t ) ) } { \partial C _ { j } } \bigg ] .
$$

Since $\begin{array} { r } { \frac { d \Gamma _ { A C , i j } ( t ) } { d t } = \rho \Sigma _ { i j } } \end{array}$ , integrating from $t = 0 \mathrm { t o } t = 1$ establishes the exact identity:

$$
\operatorname { C o v } ( \phi _ { n } ( A ) , \phi _ { n } ( C ) ) = \rho \int _ { 0 } ^ { 1 } \mathbb { E } [ \langle \nabla \phi _ { n } ( A ( t ) ) , \Sigma \nabla \phi _ { n } ( C ( t ) ) \rangle ] d t .
$$

We bound the bilinear form inside the expectation via $\ell _ { 1 } / \ell _ { \infty }$ duality. Let $p = \nabla \phi _ { n } ( A ( t ) )$ and $q = \nabla \phi _ { n } ( C ( t ) ,$ ). Because $p$ and q are probability vectors,

$$
\sum _ { i , j } p _ { i } \Sigma _ { i j } q _ { j } \leq \left( \operatorname* { m a x } _ { i , j } \left| \Sigma _ { i j } \right| \right) \sum _ { i , j } p _ { i } q _ { j } = \operatorname* { m a x } _ { i , j } \left| \Sigma _ { i j } \right| .
$$

Because Σ is positive semi-definite, its maximum absolute entry lies on the diagonal: ma $\tau _ { i , j } \left| \Sigma _ { i j } \right| \leq$ m $\mathrm { a x } _ { i } \Sigma _ { i i }$ . By assumption, ma $\mathrm { x } _ { i } \ : \Sigma _ { i i } \leq \sigma _ { T } ^ { 2 }$ . Therefore, the bilinear form is bounded almost surely by $\sigma _ { T } ^ { 2 }$ , independent of the dimension k.

Substituting this bound into the integral yields:

$$
\mathrm { C o v } ( \phi _ { n } ( A ) , \phi _ { n } ( C ) ) \leq \rho \int _ { 0 } ^ { 1 } \mathbb { E } [ \sigma _ { T } ^ { 2 } ] d t = \rho \sigma _ { T } ^ { 2 } .
$$

The approximations $\phi _ { n } ( A )$ and $\phi _ { n } ( C )$ converge almost surely to $\phi ( A )$ and $\phi ( C )$ . The Borell–TIS inequality ensures that $\{ \dot { \phi } _ { n } ( A ) \}$ and $\{ \phi _ { n } ( C ) \}$ are uniformly square-integrable. Vitali’s convergence theorem permits passing the limit through the covariance, which yields Cov(max<sub>i</sub> $A _ { i } , \operatorname* { m a x } _ { i } { \bar { C } } _ { i } ) \leq$ $\rho \sigma _ { T } ^ { 2 }$

To extend this to a general separable set $T ,$ , let $\{ T _ { m } \} _ { m \ge 1 }$ be an increasing sequence of finite subsets whose union is dense in a separability set for $f$ and $g .$ . This guarantees that $\operatorname { s u p } _ { t \in T _ { m } } f ( t )$ and $\textstyle \operatorname* { s u p } _ { t \in T _ { m } } g ( t )$ converge almost surely to their respective suprema over $T . { \mathrm { ~ \ A p p l y i n g ~ } }$ the finite-set result yields $\begin{array} { r } { \mathrm { C o v } ( \operatorname* { s u p } _ { t \in T _ { m } } f ( t ) , \operatorname* { s u p } _ { t \in T _ { m } } g ( t ) ) \leq \rho \sigma _ { T } ^ { 2 } } \end{array}$ for all $m$ . The Borell–TIS inequality provides s $\begin{array} { r } { \operatorname * { u p } _ { m } \mathbb { E } [ ( \operatorname* { s u p } _ { t \in T _ { m } } f ( t ) ) ^ { 2 } ] < \infty } \end{array}$ . Uniform integrability allows taking the limit as $m  \infty ,$ which establishes the bound for the general separable set $T$ □

In the TV-GP model, the transition rule is $f _ { t + h } ( \cdot ) = \rho ^ { h } f _ { t } ( \cdot ) + \sqrt { 1 - \rho ^ { 2 h } } \tilde { g } _ { t , h } ( \cdot )$ , where $\rho : = \sqrt { 1 - \epsilon }$ and $\tilde { g } _ { t , h }$ is a centered Gaussian process with the same spatial kernel as $f _ { t }$ . The process $\tilde { g } _ { t , h }$ is independent of $f _ { t }$ because it is a linear combination of the independent innovations $\{ g _ { t + 1 } , \ldots , g _ { t + h } \}$ Setting $T = s$ in Lemma C.6 gives $\sigma _ { T } ^ { 2 } \le 1$ and $\mathrm { C o v } ( M _ { t } , M _ { t + h } ) \le \rho ^ { h }$ for $h \geq 0$ . We apply Lemma C.5 by substituting $X _ { t } : = M _ { t } , \bar { m } = 1$ , and $c = \rho$ . For any $\eta > 0$

$$
\operatorname* { P r } \left( \left| \frac { 1 } { T } \sum _ { t = 1 } ^ { T } M _ { t } - \mathbb { E } [ M _ { 1 } ] \right| > \eta \right) \leq \operatorname* { P r } \left( \left| \frac { 1 } { T } \sum _ { t = 1 } ^ { T } M _ { t } - \frac { \mathbb { E } [ \sum _ { t = 1 } ^ { T } M _ { t } ] } { T } \right| > \eta \right) \to 0 ,
$$

since $\mathbb { E } [ \sum _ { t = 1 } ^ { T } M _ { t } ] / T = \mathbb { E } [ M _ { 1 } ]$ by stationarity. Equivalently, the time average $\begin{array} { r } { \frac { 1 } { T } \sum _ { t = 1 } ^ { T } f _ { t } \big ( x _ { t } ^ { \star } \big ) } \end{array}$ converges in probability to $\dot { \mathbb E } [ \operatorname { s u p } _ { x \in { \cal S } } f _ { 1 } ( x ) ]$ . The deviation bound in Lemma C.5(iii) provides an explicit $\mathcal { O } ( \sqrt { T } )$ concentration inequality, with constants depending on ϵ entirely through the factor $( 1 + \rho ) / ( 1 - \rho )$

## C.2 Span-only concentration fallback

This subsection justifies the span-only fallback bound used in Section 3.2. Recall that

$$
W _ { t } = \operatorname* { s u p } _ { x , y \in { \mathcal { S } } } \big ( f _ { t } ( x ) - f _ { t } ( y ) \big ) .
$$

Since $r _ { t } \le W _ { t }$ for every policy and every round,

$$
R _ { T } \leq \sum _ { t = 1 } ^ { T } W _ { t } .
$$

Define

$$
S _ { T } ^ { W } : = \sum _ { t = 1 } ^ { T } W _ { t } .
$$

Equivalently,

$$
S _ { T } ^ { W } = \operatorname* { s u p } _ { ( x _ { 1 } , y _ { 1 } ) , \ldots , ( x _ { T } , y _ { T } ) \in { \mathcal { S } } \times { \mathcal { S } } } \sum _ { t = 1 } ^ { T } { \big ( } f _ { t } ( x _ { t } ) - f _ { t } ( y _ { t } ) { \big ) } .
$$

Thus $S _ { T } ^ { W }$ is the supremum of a centered Gaussian process indexed by $( S \times S ) ^ { T }$

Let $\rho = { \sqrt { 1 - \epsilon } } .$ . For any fixed index sequence $( x _ { 1 } , y _ { 1 } ) , \dotsc , ( x _ { T } , y _ { T } )$ , the variance of the corresponding Gaussian variable satisfies

$$
\begin{array} { l } { \displaystyle \mathrm { V a r } \Bigg ( \sum _ { t = 1 } ^ { T } \big ( f _ { t } ( x _ { t } ) - f _ { t } ( y _ { t } ) \big ) \Bigg ) = \sum _ { t , s = 1 } ^ { T } \rho ^ { | t - s | } \Big [ k _ { S } ( x _ { t } , x _ { s } ) - k _ { S } ( x _ { t } , y _ { s } ) - k _ { S } ( y _ { t } , x _ { s } ) + k _ { S } ( y _ { t } , y _ { s } ) \Big ] } \\ { \displaystyle \quad \leq 4 \sum _ { t , s = 1 } ^ { T } \rho ^ { | t - s | } } \\ { \displaystyle \qquad \leq 4 T \frac { 1 + \rho } { 1 - \rho } . } \end{array}
$$

Here we used $k _ { S } ( x , x ) \leq 1$ , which implies

$$
\operatorname { V a r } \left( f ( x ) - f ( y ) \right) \leq 4
$$

and bounds the absolute covariance between two such increments by 4.

By Borell–TIS, for all $u > 0 .$

$$
\mathbb { P } \big \{ S _ { T } ^ { W } - \mathbb { E } [ S _ { T } ^ { W } ] > u \big \} \le \exp \left( - \frac { u ^ { 2 } } { 8 T ( 1 + \rho ) / ( 1 - \rho ) } \right) .
$$

Stationarity gives $\mathbb { E } [ S _ { T } ^ { W } ] = T \mu _ { w }$ . Therefore, with probability at least $1 - \omega ,$

$$
S _ { T } ^ { W } \leq T \mu _ { w } + \sqrt { 8 T \frac { 1 + \rho } { 1 - \rho } \log ( 1 / \omega ) } .
$$

Since $R _ { T } \leq S _ { T } ^ { W }$ , we obtain

$$
R _ { T } \leq T \mu _ { w } + \sqrt { 8 T \frac { 1 + \rho } { 1 - \rho } \log ( 1 / \omega ) }
$$

with probability at least $1 - \omega$

## C.3 Concentration of realized regret (Proof of Proposition 2)

Lemma A.1 establishes that the combined confidence and regularity event $\mathcal { E } _ { t }$ holds with an unconditional probability of $\mathbb { P } ( \mathcal { E } _ { t } ) \geq 1 - \delta$ . Because all instantaneous regret terms are non-negative, $r _ { t }$ is bounded pathwise by the UCB width on the success event and the spatial span on the failure event:

$$
r _ { t } \ \leq \ 2 \sqrt { \beta } \sigma _ { t - 1 } ( x _ { t } ) + C _ { 2 } + W _ { t } \mathbb { I } _ { \mathcal { E } _ { t } ^ { c } } ,\tag{47}
$$

where $\begin{array} { r } { W _ { t } = \operatorname* { s u p } _ { x \in S } f _ { t } ( x ) - \operatorname* { i n f } _ { x \in S } f _ { t } ( x ) } \end{array}$ . Summing over $T$ and applying the Cauchy-Schwarz inequality to the standard deviations yields the realized cumulative regret:

$$
R _ { T } \ \leq \ \underbrace { \sqrt { C _ { 1 } T \beta \tilde { \gamma } _ { T } } \ + \ T C _ { 2 } } _ { : = U _ { T } } + \ \underbrace { \sum _ { t = 1 } ^ { T } W _ { t } \mathbb { I } _ { \ E _ { t } ^ { c } } } _ { : = F _ { T } } .\tag{48}
$$

Since $\tilde { \gamma } _ { T }$ deterministically bounds the information gain for any sequence of length $T , U _ { T }$ is a valid upper bound almost surely.

To bound the stochastic failure penalty $F _ { T } .$ , we rely on the unconditional properties of the spatial span. By Lemma $\mathbf { A . } 2 ,$ the expected product of the sub-Gaussian span and the failure indicator satisfies

$$
\mathbb { E } [ W _ { t } \mathbb { I } _ { \mathcal { E } _ { t } ^ { c } } ] \leq \delta C _ { 3 } .
$$

Because instantaneous regret is strictly non-negative, $F _ { T } \geq 0$ . Applying Markov’s inequality directly to the sum yields:

$$
\mathbb { P } \left( F _ { T } > \frac { T \delta C _ { 3 } } { \omega } \right) \le \frac { \mathbb { E } [ F _ { T } ] } { ( T \delta C _ { 3 } ) / \omega } = \omega .\tag{49}
$$

Substituting this limit into (48) yields the bound in (15).

## D Kalman filtering implementation for the simulation study

This appendix describes the exact posterior updates used in Section 5. On the finite simulation grid $\{ x ^ { ( 1 ) } , \ldots , x ^ { ( N ) } \}$ , define $f _ { t } = ( f _ { t } ( \overline { { x } } ^ { ( 1 ) } ) , \ldots , \overline { { f } } _ { t } ( x ^ { ( N ) } ) ) ^ { \top } \in \mathbb { R } ^ { N }$ , and let $K _ { S } \in \mathbb { R } ^ { N \times N }$ be the spatial covariance matrix with entries $( K _ { S } ) _ { i j } = k _ { S } ( x ^ { ( i ) } , x ^ { ( j ) } )$ . Restricting the TV-GP recursion (4) to this grid gives the linear Gaussian state-space model

$$
\begin{array} { r l r l } & { f _ { t } = \rho f _ { t - 1 } + \xi _ { t } , } & & { \xi _ { t } \sim \mathcal { N } ( 0 , \epsilon K _ { S } ) , } \\ & { y _ { t } = H _ { t } f _ { t } + v _ { t } , } & & { v _ { t } \sim \mathcal { N } ( 0 , \sigma ^ { 2 } ) , } \end{array}
$$

where $\rho = \sqrt { 1 - \epsilon }$ and $H _ { t } \in \mathbb { R } ^ { 1 \times N }$ selects the coordinate corresponding to the sampled grid point $x _ { t }$

Let $\mu _ { t \mid t }$ and $P _ { t \mid t }$ denote the posterior mean and covariance of $f _ { t }$ after observing $y _ { t } ,$ , and let $\mu _ { t \mid t - 1 }$ and $\dot { P _ { t \vert t - 1 } }$ denote the one-step predictive quantities. We initialize $\mu _ { 1 | 0 } = 0$ and $P _ { 1 | 0 } = K _ { S }$ . Exact inference is performed by the Kalman recursion

$$
\begin{array} { r l r l } & { \mu _ { t | t - 1 } = \rho \mu _ { t - 1 | t - 1 } , } & & { P _ { t | t - 1 } = \rho ^ { 2 } P _ { t - 1 | t - 1 } + \epsilon K _ { S } , } \\ & { } & & { S _ { t } = H _ { t } P _ { t | t - 1 } H _ { t } ^ { \top } + \sigma ^ { 2 } , } & & { L _ { t } = P _ { t | t - 1 } H _ { t } ^ { \top } S _ { t } ^ { - 1 } , } \\ & { } & & { \mu _ { t | t } = \mu _ { t | t - 1 } + L _ { t } ( y _ { t } - H _ { t } \mu _ { t | t - 1 } ) , } & & { P _ { t | t } = P _ { t | t - 1 } - L _ { t } H _ { t } P _ { t | t - 1 } . } \end{array}
$$

The acquisition rule is evaluated on the grid using the predictive mean and marginal variances,

$$
i _ { t } \in \arg \operatorname* { m a x } _ { i \in \{ 1 , \dots , N \} } \left\{ ( \mu _ { t | t - 1 } ) _ { i } + \sqrt { \beta } \sqrt { ( P _ { t | t - 1 } ) _ { i i } } \right\} .
$$

This recursion is equivalent to GP posterior inference under the spatio-temporal kernel (5) restricted to the grid, but avoids recomputing the posterior from the full observation history. Since N is fixed, each posterior update costs $\dot { \mathcal { O } } ( N ^ { 2 } )$ rather than growing with $t ,$ enabling exact simulations over long horizons.

## E Simulation reproducibility details

This appendix provides the reproducibility details for the simulation study in Section 5. The experiments use synthetic data generated directly from the TV-GP model in Equation (4). No external datasets, pretrained models, or human-subject data are used.

The action space is the fixed $2 0 \times 2 0$ grid on $\mathcal { S } = [ 0 , 1 ] ^ { 2 }$ , giving $N = 4 0 0$ candidate actions. The spatial kernel is the squared exponential kernel with lengthscale $l = 0 . 1$ and prior variance 1.0. The observation noise variance is fixed at $\sigma ^ { 2 } = 0 . 0 1$ . Each reported average regret value is computed over 100 independent TV-GP sample paths of length $T = 1 0 , 0 0 0$

For each drift rate ϵ and exploration parameter $\beta ,$ the latent process is simulated according to the recursion in Equation (4). Posterior inference is performed exactly on the finite grid using the Kalman filtering recursion in Appendix D. The constant-β TV-GP-UCB policy selects actions according to

$$
i _ { t } \in \arg \operatorname* { m a x } _ { i \in \{ 1 , \dots , N \} } \left\{ ( \mu _ { t | t - 1 } ) _ { i } + \sqrt { \beta } \sqrt { ( P _ { t | t - 1 } ) _ { i i } } \right\} .
$$

The logarithmic baseline uses the schedule

$$
\beta _ { t } = c _ { 1 } \log ( c _ { 2 } t ) , \qquad c _ { 1 } = 0 . 8 , \quad c _ { 2 } = 4 ,
$$

as described in Section 5.

The accompanying code archive contains the scripts used to generate the synthetic sample paths, run constant-β TV-GP-UCB, run the logarithmic $\mathrm { t i m e - v a r y i n g - } \beta _ { t }$ baseline, compute the average regret curves, bootstrap the empirical minimizers $\hat { \beta } _ { \mathrm { o p t } }$ , and regenerate Figure 1.

For the left panel of Figure 1, shaded bands denote approximate 95% confidence intervals for the mean average regret, computed as ±1.96 standard errors over the 100 independent trials. For the right panel, the uncertainty intervals for $\hat { \beta } _ { \mathrm { o p t } }$ are computed using bootstrap percentile intervals over the independent trials.

The computations for Figure 1 were run on a MacBook Pro with an Apple M1 Pro chip, 8 CPU cores, and 16 GB of unified memory. The simulations used CPU multiprocessing and no GPU acceleration. The complete run used to generate the final figure took approximately 48 hours of wall-clock time. Memory usage was modest because the Kalman filter operates on a fixed N = 400 state dimension, but the total runtime is dominated by the large number of independent trials and parameter settings.