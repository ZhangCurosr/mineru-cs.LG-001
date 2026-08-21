# Time-Uniform Self-Normalized Concentration for Discounted Least Squares: Limits and Corrections

Yi-Shan Wu<sup>∗</sup>

## Abstract

Self-normalized concentration inequalities are standard tools in bandit and reinforcement-learning analyses. A widely used weighted extension claims an analogous time-uniform guarantee for discounted least-squares estimators in non-stationary problems. A simple scalar Gaussian counterexample with a fixed parameter shows that the claimed bounded radius is crossed with probability one. For fixed discount and regularization parameters, we further show that, when $\delta \leq 1 / 2$ and T/δ is suficiently large, any deterministic anytime boundary valid uniformly over the stated conditionally sub-Gaussian model class must be at least of order $R \sqrt { \log ( T / \delta ) }$ at some time by horizon T; for nondecreasing boundaries, this order is required at time T. We identify the proof error: diferent terminal times use diferent Gaussian mixing distributions, so the fixed-time mixtures do not form one supermartingale, and the stopping-time argument does not repair this failure. Finally, we show that the weighted inequality remains valid at each fixed deterministic time, give valid finite- and infinite-horizon corrections, and discuss consequences for downstream analyses.

## 1 Introduction

Self-normalized concentration inequalities are a basic tool in sequential learning. In linear bandits and online regression, covariates may be chosen using past observations, so the accumulated noise does not have one fixed variance scale. Instead, its scale depends on the directions and amount of information collected so far. Self-normalized inequalities control this noise after normalizing it by the corresponding empirical covariance matrix. They are the usual route to confidence sets for an unknown fixed linear parameter θ<sup>∗</sup>; the classical linear-bandit confidence construction of Abbasi-Yadkori et al. [2011] is a standard example. A time-uniform guarantee is often preferred because subsequent decisions and stopping times may depend on the observed data; it provides one high-probability event that remains valid throughout the trajectory.

In a non-stationary problem, the parameter to be estimated may change over time. Then old observations can become less informative about the current parameter θ , and it is natural to give recent data more weigh than older data. Discounted or weighted least-squares estimators implement exactly this idea. They are used in non-stationary linear and generalized-linear bandits, where discounting provides a simple way to track a changing parameter; see, for example, Russac et al. [2019, 2021], Wang et al. [2023]. To analyze such methods, one needs a confidence bound for the corresponding weighted noise process that holds along the whole trajectory. Under terminal-time discounting, however, the weights assigned to past observations change as time moves forward. Thus, a bound valid at each fixed time does not automatically give one event that is valid simultaneously over all times.

This note examines a widely used claimed time-uniform weighted self-normalized inequality in discounted bandit and reinforcement-learning analyses. We show by a scalar Gaussian example that its bounded radius is false and derive a lower bound for every horizon T. For fixed discount and regularization parameters, this lower bound requires the running envelope of any valid anytime boundary to be at least of order $R \sqrt { \log ( T / \delta ) }$ whenever $\delta \leq 1 / 2$ and T/δ is suficiently large. In particular, for nondecreasing deterministic radius functions, the radius at time T itself must grow at this rate. We also identify the failure in the proposed stopping-time argument, clarify that the weighted inequality remains valid at each fixed deterministic time, and give valid finite- and infinite-horizon corrections through explicit confidence allocation.

Relation to prior work. Abbasi-Yadkori et al. [2011] established the standard anytime self-normalized inequality for the unweighted process with a fixed ridge regularizer, whereas Russac et al. [2019] proposed the widely used weighted extension whose discounted specialization motivates this note. Our contribution is a correction-and-consequences analysis: we distinguish fixed-time from time-uniform validity, give a counterexample and a matching-order lower bound for the discounted process, and trace how the claimed anytime form enters later analyses. Section 6 provides a more detailed discussion of the relevant literature, including direct downstream uses of the claimed inequality and later treatments based on fixed-time or finite-horizon control.

Organization. The note is organized as follows. Section 2 introduces the classical self-normalized inequality, the claimed discounted extension, and valid finite- and infinite-horizon corrections. Section 3 constructs a scalar Gaussian counterexample to the time-uniform inequality claimed by Russac et al. [2019] and gives lower bounds for valid anytime boundaries. Section 4 revisits the proof of the classical self-normalized inequality and presents a valid extension to one predictable sequence of weights. Section 5 identifies the invalid step in the proposed time-uniform argument and explains what remains valid at a fixed deterministic time. Finally, Section 6 discusses the scope of the issue and its consequences for downstream analyses in bandit and reinforcement-learning literature.

## 2 Background and the Claimed Inequality

Stationary ridge estimation. We first recall how self-normalized quantities arise in the usual stationary linear model when performing sequential decision-making. At each time t, the learner chooses a feature vector $X _ { t } \in \mathbb { R } ^ { d }$ and observes $Y _ { t } = X _ { t } ^ { \top } \theta ^ { * } + \eta _ { t }$ , where $\theta ^ { * } \in \mathbb { R } ^ { d }$ is an unknown but fixed parameter and $\eta _ { t }$ is noise. For $\lambda > 0$ , the ridge estimator is the solution of

$$
\widehat { \theta } _ { t } \triangleq \underset { \theta \in \mathbb { R } ^ { d } } { \arg \operatorname* { m i n } } \left\{ \sum _ { s = 1 } ^ { t } \big ( Y _ { s } - X _ { s } ^ { \top } \theta \big ) ^ { 2 } + \lambda \| \theta \| _ { 2 } ^ { 2 } \right\} .
$$

Its equivalent closed-form expression is

$$
\widehat { \theta } _ { t } \triangleq { \boldsymbol { V } } _ { t } ^ { - 1 } \sum _ { s = 1 } ^ { t } X _ { s } Y _ { s } , \quad \mathrm { w h e r e } \quad V _ { t } \triangleq \lambda I _ { d } + \sum _ { s = 1 } ^ { t } X _ { s } X _ { s } ^ { \top } .
$$

Substituting the model into the closed-form estimator yields the error decomposition

$$
\widehat { \theta } _ { t } - \theta ^ { * } = V _ { t } ^ { - 1 } \big ( S _ { t } - \lambda \theta ^ { * } \big ) , \quad \mathrm { w h e r e } \quad S _ { t } \triangleq \sum _ { s = 1 } ^ { t } \eta _ { s } X _ { s } ,
$$

Measuring error in the data-dependent norm $\| u \| _ { V _ { t } } \triangleq \sqrt { u ^ { \top } V _ { t } u }$ , which reflects the information collected in each direction, gives

$$
\begin{array} { r } { \| \widehat { \theta } _ { t } - \theta ^ { * } \| _ { V _ { t } } = \big \| \big ( S _ { t } - \lambda \theta ^ { * } \big ) \big \| _ { V _ { t } ^ { - 1 } } \leq \| S _ { t } \| _ { V _ { t } ^ { - 1 } } + \| \lambda \theta ^ { * } \| _ { V _ { t } ^ { - 1 } } \leq \| S _ { t } \| _ { V _ { t } ^ { - 1 } } + \sqrt { \lambda } \| \theta ^ { * } \| _ { 2 } . } \end{array}
$$

The second term is a constant, and the derivation is due to $V _ { t } \succeq \lambda I _ { d }$ , and $S _ { t }$ is the accumulated noise in the estimator, while $V _ { t }$ records the amount and directions of information collected so far. Hence, constructing a confidence set for $\theta ^ { * }$ reduces to controlling the self-normalized noise term $\lVert S _ { t } \rVert _ { V _ { t } ^ { - 1 } }$

To state the sequential assumptions precisely, let $( \mathcal { F } _ { t } ) _ { t \geq 0 }$ be a filtration. Assume that $X _ { t }$ is $\mathcal { F } _ { t - 1 } -$ measurable and satisfies $\| X _ { t } \| _ { 2 } \leq L$ , while $\eta _ { t }$ is $\mathcal { F } _ { t }$ -measurable and conditionally R-sub-Gaussian given $\mathcal { F } _ { t - 1 }$

Here $L , R > 0 .$ , and fix $\delta \in ( 0 , 1 )$ . Under these assumptions, the self-normalized inequality of Abbasi-Yadkori et al. [2011], with ridge regularization $\lambda I _ { d } .$ gives the time-uniform statement

$$
\mathbb { P } \left( \exists t \geq 0 : \| S _ { t } \| _ { V _ { t } ^ { - 1 } } > R \sqrt { 2 \log \frac { \operatorname* { d e t } ( V _ { t } ) ^ { 1 / 2 } } { \delta \lambda ^ { d / 2 } } } \right) \leq \delta .
$$

Moreover,

$$
\frac { \operatorname * { d e t } ( V _ { t } ) ^ { 1 / 2 } } { \lambda ^ { d / 2 } } \leq \left( 1 + \frac { t L ^ { 2 } } { \lambda d } \right) ^ { d / 2 } .
$$

Hence,

$$
\mathbb { P } \left( \exists t \geq 0 : \| S _ { t } \| _ { V _ { t } ^ { - 1 } } > R \sqrt { 2 \log \frac { 1 } { \delta } + d \log \left( 1 + \frac { t L ^ { 2 } } { \lambda d } \right) } \right) \leq \delta .\tag{1}
$$

The radius in (1) grows approximately as $R \surd$ d log t.

Discounted estimation. In a non-stationary problem, the parameter may instead change with time, so that $Y _ { s } = X _ { s } ^ { \top } \theta _ { s } + \eta _ { s }$ . Typically the parameter is assumed to change slowly, so that $\theta _ { t }$ is close to $\theta _ { t - 1 }$ . In this case, it is natural to give recent observations more weight than older ones. A simple way to implement this idea is through exponential discounting. For a fixed discount factor $0 < \gamma < 1$ , the corresponding weighted ridge estimator and the weighted design matrix are

$$
\widehat { \theta } _ { t } ^ { \gamma } \triangleq ( V _ { t } ^ { \gamma } ) ^ { - 1 } \sum _ { s = 1 } ^ { t } \gamma ^ { t - s } X _ { s } Y _ { s } , \qquad V _ { t } ^ { \gamma } \triangleq \lambda I _ { d } + \sum _ { s = 1 } ^ { t } \gamma ^ { t - s } X _ { s } X _ { s } ^ { \top } , \qquad 0 < \gamma < 1 .
$$

The random noise contribution to this estimator and its self-normalizer are

$$
S _ { t } ^ { \gamma } \triangleq \sum _ { s = 1 } ^ { t } \gamma ^ { t - s } \eta _ { s } X _ { s } , \qquad \widetilde { V } _ { t } ^ { \gamma } \triangleq \lambda I _ { d } + \sum _ { s = 1 } ^ { t } \gamma ^ { 2 ( t - s ) } X _ { s } X _ { s } ^ { \top } .
$$

The weights in $\widetilde { V } _ { t } ^ { \gamma }$ are squared because it normalizes the variance proxy of the weighted sum $S _ { t } ^ { \gamma }$ . Thus, it is distinct from the estimator’s discounted covariance matrix $V _ { t } ^ { \gamma }$ , which uses the unsquared weights $\gamma ^ { t - s }$ Corollary 3 of Russac et al. [2019], derived from their Proposition 1, claims the time-uniform bound

$$
\mathbb { P } \left( \exists t \geq 1 : \| S _ { t } ^ { \gamma } \| _ { ( \widetilde { V } _ { t } ^ { \gamma } ) ^ { - 1 } } > r _ { t } ^ { \gamma } ( \delta ) \right) \leq \delta ,\tag{2}
$$

where for any $\delta \in ( 0 , 1 )$ ,

$$
r _ { t } ^ { \gamma } ( \delta ) \triangleq R \sqrt { 2 \log \frac { 1 } { \delta } + d \log \left( 1 + \frac { L ^ { 2 } ( 1 - \gamma ^ { 2 t } ) } { \lambda d ( 1 - \gamma ^ { 2 } ) } \right) } .\tag{3}
$$

Note that when $t \to \infty , ( 1 - \gamma ^ { 2 t } ) \to 1$ , meaning that the radius is bounded.

Why the claimed bound cannot hold. Discounting gives the process a finite efective memory. The process does not become progressively more stable: old noise disappears, but new noise continues to arrive. Hence, a bounded threshold cannot control the discounted process forever. In fact, in Section 3, we show a counterexample to the Russac bound that supports this intuition.

Correction. First of all, radius (3) remains valid pointwise, i.e., for any $t \geq 1$

$$
\forall t \geq 1 , \qquad \mathbb { P } \left( \| S _ { t } ^ { \gamma } \| _ { ( \widetilde { V } _ { t } ^ { \gamma } ) ^ { - 1 } } > r _ { t } ^ { \gamma } ( \delta ) \right) \leq \delta .\tag{4}
$$

A correct time-uniform statement can be obtained by, $\mathrm { e . g . }$ , letting

$$
\pi _ { t } = \frac { 1 } { 2 t ( 1 + \log t ) ^ { 2 } } , \qquad \delta _ { t } = \delta \pi _ { t } .
$$

Indeed, $\pi _ { t }$ is decreasing on $[ 1 , \infty )$ , so

$$
\sum _ { t = 1 } ^ { \infty } \pi _ { t } \leq \pi _ { 1 } + \int _ { 1 } ^ { \infty } { \frac { \mathrm { d } x } { 2 x ( 1 + \log x ) ^ { 2 } } } .
$$

Here $\textstyle \pi _ { 1 } = { \frac { 1 } { 2 } }$ and the integral equals $1 / 2 { : }$ with the substitution $u = 1 +$ log $x ,$ it becomes

$$
\int _ { 1 } ^ { \infty } { \frac { \mathrm { d } x } { 2 x ( 1 + \log x ) ^ { 2 } } } = { \frac { 1 } { 2 } } \int _ { 1 } ^ { \infty } u ^ { - 2 } \mathrm { d } u = { \frac { 1 } { 2 } } .
$$

Therefore, $\begin{array} { r } { \sum _ { t = 1 } ^ { \infty } \pi _ { t } \leq \frac { 1 } { 2 } + \frac { 1 } { 2 } = 1 } \end{array}$ . Applying the pointwise result with confidence level $\delta _ { t }$ and taking a union bound gives

$$
\mathbb { P } \left( \exists t \geq 1 : \| S _ { t } ^ { \gamma } \| _ { ( \widetilde { V } _ { t } ^ { \gamma } ) ^ { - 1 } } > r _ { t } ^ { \gamma } ( \delta _ { t } ) \right) \leq \delta .\tag{5}
$$

For fixed $\gamma$ and fixed remaining parameters, the radius

$$
r _ { t } ^ { \gamma } ( \delta _ { t } ) = R \sqrt { 2 \log t + 4 \log ( 1 + \log t ) + O ( 1 ) } = R \sqrt { 2 \log t + O ( \log \log t ) } .
$$

Thus, it has order $R \sqrt { \log t }$ just as in (1).

## 3 A Counterexample to (2) and Lower Bounds for Anytime Boundaries

This section has two parts. We first give a scalar Gaussian example in which the proposed bounded radius in (2) is crossed with probability one. We then establish lower bounds for anytime boundaries. The main result lower-bounds the running envelope without any monotonicity assumption; for nondecreasing boundaries, it gives a direct lower bound on the boundary value at each time. This rules out any bounded anytime radius under the same assumptions.

## 3.1 A scalar Gaussian counterexample

We now construct a one-dimensional Gaussian example in which the proposed boundary is crossed with probability one. Take $d = 1 , L = 1 , \theta ^ { * } = 0 , X _ { t } = 1$ for all $t \geq 1$ , and $\lambda > 0$ . Let $( \eta _ { t } ) _ { t \geq 1 }$ be independent random variables satisfying $\eta _ { t } \sim \mathcal { N } ( 0 , R ^ { 2 } )$ . Then $Y _ { t } = \eta _ { t }$ , so this is a stationary linear model. The deterministic choice $X _ { t } = 1$ is valid because it is $\mathcal { F } _ { t - 1 }$ -measurable and satisfies $| X _ { t } | \le L = 1$ ; in the bandit scenario, it efectively takes the action set to be only {1}. With respect to the natural filtration, this example satisfies the assumptions of Section 2. Setting $S _ { 0 } ^ { \gamma } = \bar { 0 }$ , we have

$$
S _ { t } ^ { \gamma } = \gamma S _ { t - 1 } ^ { \gamma } + \eta _ { t } , \qquad \widetilde { V } _ { t } ^ { \gamma } = \lambda + \frac { 1 - \gamma ^ { 2 t } } { 1 - \gamma ^ { 2 } } .\tag{6}
$$

The first recursion in (6) is a stable first-order autoregression, usually called an $\operatorname { A R } ( 1 )$ process: at each time, the process retains a fraction γ of its previous value and receives a fresh independent Gaussian noise. Iterating the recursion gives $\begin{array} { r } { S _ { t } ^ { \gamma } = \sum _ { s = 1 } ^ { t } \gamma ^ { t - s } \eta _ { s } } \end{array}$ , and hence

$$
\mathrm { V a r } ( S _ { t } ^ { \gamma } ) = R ^ { 2 } \frac { 1 - \gamma ^ { 2 t } } { 1 - \gamma ^ { 2 } } = R ^ { 2 } \big ( \widetilde { V } _ { t } ^ { \gamma } - \lambda \big ) \longrightarrow \frac { R ^ { 2 } } { 1 - \gamma ^ { 2 } } .
$$

Thus, discounting keeps the variance uniformly bounded over time, but does not make the random fluctuations vanish. The same identity explains the term self-normalizer: apart from the factor $R ^ { 2 }$ and the regularization term $\lambda , \widetilde { V } _ { t } ^ { \gamma }$ is the variance scale of the random quantity $S _ { t } ^ { \gamma }$ that it normalizes. This calculation does not by itself prove an almost-sure boundary crossing, but it explains why a bounded time-uniform boundary appears suspicious. In fact, we will see in the next proposition that the process crosses the proposed bounded boundary with probability one. The proof is deferred to Section $\mathrm { A . 1 }$

Proposition 1. For every $\delta \in ( 0 , 1 )$ , the process in (6) satisfies

$$
\mathbb { P } \left( \exists t \geq 1 : \| S _ { t } ^ { \gamma } \| _ { ( \widetilde { V } _ { t } ^ { \gamma } ) ^ { - 1 } } > r _ { t } ^ { \gamma } ( \delta ) \right) = 1 ,
$$

where $r _ { t } ^ { \gamma } ( \delta )$ is the radius defined in (3).

## 3.2 Lower bounds for anytime boundaries

The previous proposition shows that a bounded time-uniform radius cannot be valid. We first prove a finite-horizon lower bound for one common level controlling the first T times. This formulation makes no monotonicity assumption and therefore lower-bounds the running envelope of every deterministic anytime boundary. For every deterministic nondecreasing anytime boundary, the running envelope equals $B _ { T } ( \delta )$ . In particular, for fixed $\gamma$ and $\lambda ,$ there are positive constants $c _ { \gamma , \lambda }$ and $x _ { 0 }$ such that, if $\delta \leq 1 / 2$ and $T / \delta \geq x _ { 0 }$ every such boundary must satisfy

$$
B _ { T } ( \delta ) \geq c _ { \gamma , \lambda } R \sqrt { \log ( T / \delta ) } .
$$

The corrected radius in (5) is one boundary in this class and achieves this order. We now explain the finite-horizon argument. Suppose that we have an anytime boundary $B _ { t } ( \delta )$ such that

$$
\mathbb { P } \left( \frac { | S _ { t } ^ { \gamma } | } { \sqrt { \widetilde V _ { t } ^ { \gamma } } } \le B _ { t } ( \delta ) \ \mathrm { f o r ~ e v e r y ~ } t \ge 1 \right) \ge 1 - \delta .
$$

Fix a horizon T. Whenever the anytime guarantee holds, it also holds at each of the first $T$ times. Let

$$
C _ { T } ( \delta ) \triangleq \operatorname* { m a x } _ { 1 \leq t \leq T } B _ { t } ( \delta ) .
$$

We make no monotonicity assumption on $t \mapsto B _ { t } ( \delta )$ . Thus, $C _ { T } ( \delta )$ is simply the largest boundary value among the first $T$ times, and it need not equal $B _ { T } ( \delta )$ . Since every $B _ { t } ( \delta )$ is at most $C _ { T } ( \delta )$ , the normalized process also stays below the single level $C _ { T } ( \delta )$ throughout the first T times. Therefore,

$$
\mathbb { P } \left( \operatorname* { m a x } _ { 1 \leq t \leq T } \frac { \lvert S _ { t } ^ { \gamma } \rvert } { \sqrt { \widetilde V _ { t } ^ { \gamma } } } \leq C _ { T } ( \delta ) \right) \geq 1 - \delta .
$$

This finite-horizon statement is weaker than the original anytime guarantee, because it replaces the separate thresholds $B _ { 1 } ( \delta ) , \dots , B _ { T } ( \delta )$ by their largest value. However, every valid anytime boundary gives such a common level $C _ { T } ( \delta )$ . The next proposition gives a lower bound on $C _ { T } ( \delta )$

Proposition 2 (Finite-horizon lower bound). Consider the same scalar Gaussian process as in $( 6 )$ . Fix $T \geq 1$ and $\delta \in ( 0 , 1 )$ . Let $C _ { T } ( \delta )$ be any deterministic boundary satisfying

$$
\mathbb { P } \left( \operatorname* { m a x } _ { 1 \leq t \leq T } \frac { \lvert S _ { t } ^ { \gamma } \rvert } { \sqrt { \widetilde V _ { t } ^ { \gamma } } } \leq C _ { T } ( \delta ) \right) \geq 1 - \delta .
$$

Then

$$
C _ { T } ( \delta ) \geq \frac { R } { ( 1 + \gamma ) \sqrt { \lambda + ( 1 - \gamma ^ { 2 } ) ^ { - 1 } } } \Phi ^ { - 1 } \left( \frac { 1 + ( 1 - \delta ) ^ { 1 / T } } { 2 } \right) ,
$$

where Φ is the standard normal distribution function. In particular, for fixed $\gamma$ and $\lambda$ , there are positive constants $c _ { \gamma , \lambda }$ and $x _ { 0 }$ such that, for $\delta \in ( 0 , 1 / 2 ]$ and $T / \delta \geq x _ { 0 }$

$$
C _ { T } ( \delta ) \geq c _ { \gamma , \lambda } R \sqrt { \log \frac { T } { \delta } } .
$$

Thus, for an arbitrary deterministic anytime boundary, the proposition lower-bounds its running envelope $C _ { T } ( \delta ) = \operatorname* { m a x } _ { 1 \leq t \leq T } B _ { t } ( \delta )$ , rather than any particular value $B _ { T } ( \delta )$

The proof is deferred to Section $\mathrm { A . 2 }$ . Its main idea is simple. Since $\widetilde { V } _ { t } ^ { \gamma } \leq V _ { \operatorname* { m a x } }$ , the normalized bound in the proposition implies that $| S _ { t } ^ { \gamma } |$ must remain below the common level $C _ { T } ( \delta ) \sqrt { V _ { \mathrm { m a x } } }$ for all $1 \leq t \leq T$ . We can therefore ask how large this deterministic interval must be in order to contain the process throughout the first T times with probability at least $1 - \delta .$ The recursion then implies that each of the T independent noises must lie in a corresponding interval. By independence, the probability that one Gaussian noise lies outside this interval must be at most of order $\delta / T$ . This is possible only when $C _ { T } ( \delta )$ has order at least $R \sqrt { \log ( T / \delta ) }$

The finite-horizon formulation is a more general result: it gives the running-envelope lower bound without imposing monotonicity. The following corollary gives the more familiar pointwise consequence for nondecreasing deterministic anytime boundaries. The corrected boundary $r _ { t } ^ { \gamma } ( \delta _ { t } )$ in (5) is nondecreasing: π<sub>t</sub> decreases with t, so $\log ( 1 / \delta _ { t } )$ increases, and $1 - \gamma ^ { 2 t }$ also increases. Hence, both terms inside the defining radius are nondecreasing, and the corollary applies to this correction.

Corollary 1 (Nondecreasing anytime boundaries). Under the same assumptions, suppose $B _ { t } ( \delta )$ is a nondecreasing deterministic anytime boundary satisfying

$$
\mathbb { P } \left( \frac { | S _ { t } ^ { \gamma } | } { \sqrt { \widetilde V _ { t } ^ { \gamma } } } \le B _ { t } ( \delta ) \ f o r \ e v e r y \ t \ge 1 \right) \ge 1 - \delta ,
$$

Then there are positive constants $c _ { \gamma , \lambda }$ and $x _ { 0 }$ such that, $i f \delta \in ( 0 , 1 / 2 ]$ and $t / \delta \geq x _ { 0 }$ , then

$$
B _ { t } ( \delta ) \geq c _ { \gamma , \lambda } R \sqrt { \log \frac { t } { \delta } } .
$$

Proof. Set $C _ { t } ( \delta ) = \mathrm { m a x } _ { 1 \leq s \leq t } B _ { s } ( \delta )$ . By nondecreasingness, $C _ { t } ( \delta ) = B _ { t } ( \delta )$ . The anytime guarantee implies the finite-horizon common-boundary condition for $C _ { t } ( \delta )$ , so Proposition 2 completes the proof.

Together, these two propositions rule out the proposed bounded radius. The first proposition shows that the original radius is eventually crossed with probability one. The second gives a lower bound for every horizon T and, when $\delta \leq 1 / 2$ and $T / \delta$ is suficiently large, shows that the running envelope of any valid anytime boundary must be at least of order $R \sqrt { \log ( T / \delta ) }$ . Thus, the union-bound correction (5) matches the lower-bound order in this scalar Gaussian example. Since this correction is nondecreasing, the corollary also shows that its $\sqrt { \log T }$ growth is unavoidable among nondecreasing deterministic anytime boundaries. In particular, this scalar Gaussian process already rules out the claimed bounded anytime radius under the assumptions of Section 2. Therefore, no correct stopping-time argument can establish that bounded radius for the full class of processes covered by those assumptions.

## 4 The Classical Self-Normalized Argument

In this section, we revisit the proof of the self-normalized concentration inequality of Abbasi-Yadkori et al. [2011] using Ville’s inequality. The proof consists of three steps: constructing an exponential supermartingale for each fixed direction, mixing these supermartingales over a fixed Gaussian distribution, and applying Ville’s inequality to the resulting mixture process.

## 4.1 Ville’s inequality

We first recall Ville’s inequality used in the proof.

Lemma 1 (Ville’s inequality). Let $( M _ { t } ) _ { t \geq 0 }$ be a nonnegative supermartingale with respect to $( \mathcal { F } _ { t } ) _ { t \geq 0 }$ . Then, for every $u > 0$ ，

$$
\mathbb { P } \left( \exists t \ge 0 : M _ { t } \ge u \right) \le \frac { \mathbb { E } \left[ M _ { 0 } \right] } { u } .
$$

For any fixed deterministic time $t ,$ Markov’s inequality gives

$$
\mathbb { P } \left( M _ { t } \geq u \right) \leq \frac { \mathbb { E } \left[ M _ { t } \right] } { u } .
$$

If $( M _ { t } ) _ { t \geq 0 }$ is a nonnegative supermartingale, Ville’s inequality controls the probability that the process ever crosses the threshold u. Its application requires one nonnegative supermartingale indexed by time, rather than a collection of random variables or mixture constructions that are valid only separately at each time.

## 4.2 Exponential supermartingales in a fixed direction

Recall that

$$
\boldsymbol { S _ { t } } = \sum _ { s = 1 } ^ { t } \eta _ { s } \boldsymbol { X _ { s } } , \qquad V _ { t } = \lambda I _ { d } + \sum _ { s = 1 } ^ { t } \boldsymbol { X _ { s } } \boldsymbol { X _ { s } } ^ { \top } ,
$$

where $\lambda > 0$ is fixed. For any fixed $x \in \mathbb { R } ^ { d }$ , define

$$
M _ { t } ( x ) = \exp \left( \frac { x ^ { \top } S _ { t } } { R } - \frac { 1 } { 2 } x ^ { \top } ( V _ { t } - \lambda I _ { d } ) x \right) .\tag{7}
$$

Since $X _ { t }$ is $\mathcal { F } _ { t - 1 }$ -measurable and $\eta _ { t }$ is conditionally R-sub-Gaussian,

$$
\mathbb { E } \left[ M _ { t } ( x ) \mid { \mathcal { F } } _ { t - 1 } \right] = M _ { t - 1 } ( x ) \exp \left( - { \frac { 1 } { 2 } } ( x ^ { \top } X _ { t } ) ^ { 2 } \right) \mathbb { E } \left[ \exp \left( { \frac { \eta _ { t } x ^ { \top } X _ { t } } { R } } \right) \Bigg | { \mathcal { F } } _ { t - 1 } \right] \leq M _ { t - 1 } ( x ) .\tag{8}
$$

Therefore, $( M _ { t } ( x ) ) _ { t \geq 0 }$ is a nonnegative supermartingale for every fixed direction x.

## 4.3 Gaussian mixture of supermartingales

To control all directions simultaneously, let h be the density of the fixed Gaussian distribution $\mathcal { N } ( 0 , \lambda ^ { - 1 } I _ { d } )$ and define

$$
M _ { t } = \int _ { \mathbb R ^ { d } } M _ { t } ( x ) h ( x ) d x .
$$

Because $M _ { t } ( x ) h ( x ) \geq 0$ , conditional Tonelli’s theorem allows us to exchange conditional expectation and integration. Hence, the mixture $( M _ { t } ) _ { t \geq 0 }$ is also a nonnegative supermartingale:

$$
\mathbb { E } [ M _ { t } \mid \mathcal { F } _ { t - 1 } ] = \int _ { \mathbb R ^ { d } } \mathbb { E } [ M _ { t } ( x ) \mid \mathcal { F } _ { t - 1 } ] h ( x ) d x \leq \int _ { \mathbb R ^ { d } } M _ { t - 1 } ( x ) h ( x ) d x = M _ { t - 1 } ,
$$

with $M _ { 0 } = 1$ . Substituting the Gaussian density and $M _ { t } ( x )$ in (7) gives

$$
M _ { t } = \frac { \lambda ^ { d / 2 } } { ( 2 \pi ) ^ { d / 2 } } \int _ { \mathbb R ^ { d } } \exp \left( \frac { x ^ { \top } S _ { t } } R - \frac 1 2 x ^ { \top } V _ { t } x \right) ~ d x = \left( \frac { \lambda ^ { d } } { \operatorname* { d e t } ( V _ { t } ) } \right) ^ { 1 / 2 } \exp \left( \frac { \| S _ { t } \| _ { V _ { t } ^ { - 1 } } ^ { 2 } } { 2 R ^ { 2 } } \right) .\tag{9}
$$

Therefore,

$$
\mathbb P \left\{ \exists t \geq 0 : \| S _ { t } \| _ { V _ { t } ^ { - 1 } } > R \sqrt { 2 \log \frac { \operatorname* { d e t } ( V _ { t } ) ^ { 1 / 2 } } { \delta \lambda ^ { d / 2 } } } \right\} = \mathbb P \left\{ \exists t \geq 0 : M _ { t } > \frac { 1 } { \delta } \right\} \leq \delta ,
$$

where the inequality is due to applying Ville’s inequality to the single mixture nonnegative supermartingale $( M _ { t } ) _ { t \geq 0 }$ with $M _ { 0 } = 1$

## 4.4 A Valid Extension for Fixed Predictable Weights

The preceding argument remains valid for one predictable sequence of weights and a fixed regularizer. This valid special case is also contained in the weighted setup considered by Russac et al. [2019]. In particular, let $( w _ { t } ) _ { t \geq 1 }$ be a real-valued predictable process, so that $w _ { t }$ is $\mathcal { F } _ { t - 1 }$ -measurable, and define

$$
S _ { t } ^ { w } = \sum _ { s = 1 } ^ { t } w _ { s } \eta _ { s } X _ { s } , \qquad V _ { t } ^ { w } = \lambda I _ { d } + \sum _ { s = 1 } ^ { t } w _ { s } ^ { 2 } X _ { s } X _ { s } ^ { \top } ,
$$

where $\lambda > 0$ is fixed. Since $w _ { t } X _ { t }$ remains $\mathcal { F } _ { t - 1 }$ -measurable, the previous result can be applied with $X _ { t }$ replaced by $w _ { t } X _ { t }$ . Therefore,

$$
\mathbb { P } \left( \exists t \ge 0 : \| S _ { t } ^ { w } \| _ { ( V _ { t } ^ { w } ) ^ { - 1 } } > R \sqrt { 2 \log \frac { \operatorname* { d e t } ( V _ { t } ^ { w } ) ^ { 1 / 2 } } { \delta \lambda ^ { d / 2 } } } \right) \le \delta .\tag{10}
$$

Thus, predictable weighting by itself does not destroy the time-uniform argument. In $( 1 0 ) , ( w _ { s } ) _ { s \geq 1 }$ is one sequence indexed by observation time: once observation s is assigned weight $w _ { s }$ , that weight does not change as the terminal time continues. Also, both the regularizer $\lambda I _ { d }$ and the Gaussian mixing distribution $\mathcal { N } ( 0 , \lambda ^ { - 1 } I _ { d } )$ remain fixed over time.

## 5 Where the Time-Uniform Argument Fails

Recall the discounted quantities

$$
S _ { t } ^ { \gamma } = \sum _ { s = 1 } ^ { t } \gamma ^ { t - s } \eta _ { s } X _ { s } , \qquad \widetilde V _ { t } ^ { \gamma } = \lambda I _ { d } + \sum _ { s = 1 } ^ { t } \gamma ^ { 2 ( t - s ) } X _ { s } X _ { s } ^ { \top } .
$$

The weights $\gamma ^ { t - s }$ depend on the terminal time $t ,$ so they do not form one predictable sequence to which (10) can be applied simultaneously over time. The argument of Russac et al. [2019] instead uses an equivalent rescaling that fixes the observation-time weights but makes the regularizer depend on $t .$ Define

$$
\overline { { S } } _ { t } ^ { \gamma } \triangleq \sum _ { s = 1 } ^ { t } \gamma ^ { - s } \eta _ { s } X _ { s } , \qquad \overline { { V } } _ { t } ^ { \gamma } \triangleq \gamma ^ { - 2 t } \lambda I _ { d } + \sum _ { s = 1 } ^ { t } \gamma ^ { - 2 s } X _ { s } X _ { s } ^ { \top } .
$$

Then $S _ { t } ^ { \gamma } = \gamma ^ { t } \overline { { S } } _ { t } ^ { \gamma }$ and $\widetilde { V } _ { t } ^ { \gamma } = \gamma ^ { 2 t } \overline { { V } } _ { t } ^ { \gamma }$ . The observation-time weights $\gamma ^ { - s }$ are now fixed, but the regularizer $\gamma ^ { - 2 t } \lambda I _ { d }$ varies with the terminal time.

For every fixed $x \in \mathbb { R } ^ { d }$ , define

$$
M _ { t } ( x ) = \exp \left( \frac { x ^ { \top } \overline { { S } } _ { t } ^ { \gamma } } { R } - \frac { 1 } { 2 } x ^ { \top } \left( \overline { { V } } _ { t } ^ { \gamma } - \gamma ^ { - 2 t } \lambda I _ { d } \right) x \right) .\tag{11}
$$

Since $\begin{array} { r } { \overline { { V } } _ { t } ^ { \gamma } - \gamma ^ { - 2 t } \lambda I _ { d } = \sum _ { s = 1 } ^ { t } \gamma ^ { - 2 s } X _ { s } X _ { s } ^ { \top } } \end{array}$ , this is exactly the fixed-weight construction in $( 7 )$ with $X _ { s }$ replaced by $\gamma ^ { - s } X _ { s }$ . Therefore, $( M _ { t } ( x ) ) _ { t \geq 0 }$ is a nonnegative supermartingale with $M _ { 0 } ( x ) = 1$ for every fixed direction x.

## 5.1 The mixtures do not form one single supermartingale

For each $t \geq 0$ , let $h _ { t }$ be the density of $\mathcal { N } ( 0 , \gamma ^ { 2 t } \lambda ^ { - 1 } I _ { d } )$ and define

$$
\overline { { M } } _ { t } \triangleq \int _ { \mathbb { R } ^ { d } } M _ { t } ( x ) h _ { t } ( x ) d x .
$$

Substituting the definitions of $M _ { t } ( x )$ and $h _ { t } ( x )$ , the integral

$$
\overline { { M } } _ { t } = \left( \frac { \gamma ^ { - 2 t d } \lambda ^ { d } } { \operatorname * { d e t } ( \overline { { V } } _ { t } ^ { \gamma } ) } \right) ^ { 1 / 2 } \exp \left( \frac { \Vert \overline { { S } } _ { t } ^ { \gamma } \Vert _ { ( \overline { { V } } _ { t } ^ { \gamma } ) ^ { - 1 } } ^ { 2 } } { 2 R ^ { 2 } } \right) = \left( \frac { \lambda ^ { d } } { \operatorname * { d e t } ( \widetilde { V } _ { t } ^ { \gamma } ) } \right) ^ { 1 / 2 } \exp \left( \frac { \Vert S _ { t } ^ { \gamma } \Vert _ { ( \widetilde { V } _ { t } ^ { \gamma } ) ^ { - 1 } } ^ { 2 } } { 2 R ^ { 2 } } \right) .\tag{12}
$$

If $( \overline { { M } } _ { t } ) _ { t \geq 0 }$ were one nonnegative supermartingale with $\overline { { M } } _ { 0 } = 1$ , Ville’s inequality would provide the desired time-uniform result.

The problem is that the Gaussian density used in the mixture changes with t. Conditional Tonelli’s theorem and the directional supermartingale property (8) only give

$$
\mathbb { E } [ \overline { { M } } _ { t } \mid \mathcal { F } _ { t - 1 } ] = \int _ { \mathbb R ^ { d } } \mathbb { E } [ M _ { t } ( x ) \mid \mathcal { F } _ { t - 1 } ] h _ { t } ( x ) d x \leq \int _ { \mathbb R ^ { d } } M _ { t - 1 } ( x ) h _ { t } ( x ) d x ,
$$

whereas

$$
\overline { { M } } _ { t - 1 } = \int _ { \mathbb { R } ^ { d } } M _ { t - 1 } ( x ) h _ { t - 1 } ( x ) d x .
$$

The two expressions integrate the same random function $M _ { t - 1 } ( x )$ against diferent densities. Since $h _ { t } \neq h _ { t - 1 } .$ the first integral is not necessarily bounded by $\overline { { M } } _ { t - 1 }$ . The directional supermartingale property compares $M _ { t } ( x )$ and $M _ { t - 1 } ( x )$ for the same fixed direction $x ;$ it does not compare mixtures formed using diferent densities. Therefore, the construction does not establish

$$
\mathbb { E } [ \overline { { M } } _ { t } \ | \mathcal F _ { t - 1 } ] \le \overline { { M } } _ { t - 1 } ,
$$

and the fixed-time mixtures $( \overline { { M } } _ { t } ) _ { t \geq 0 }$ do not, in general, form one nonnegative supermartingale to which Ville’s inequality can be applied. Of course, this does not say the inequality is false and maybe there is a way to bypass it. However, the counterexample in Section 3 shows that this is not just a missing proof: the conclusion indeed is false.

## 5.2 The stopping-time repair is invalid

Lemma 3 of Russac et al. [2019] attempts to bypass the missing supermartingale property using an auxiliary sequence $( Z _ { t } ) _ { t \geq 0 }$ , independent of the data-generating process, such that $Z _ { t }$ has density $h _ { t }$ . Let τ be a stopping time with respect to $( \mathcal { F } _ { t } ) _ { t \geq 0 } \mathrm { ; }$ for simplicity, one may first take $\tau$ to be bounded. Independence of the auxiliary Gaussian sequence gives

$$
\mathbb { E } [ \overline { { M } } _ { \tau } ] = \mathbb { E } [ M _ { \tau } ( Z _ { \tau } ) ] .
$$

For every fixed $x \in \mathbb { R } ^ { d }$ , optional stopping applied to the nonnegative supermartingale $( M _ { t } ( x ) ) _ { t \geq 0 }$ gives

$$
\mathbb { E } [ M _ { \tau } ( x ) ] \leq 1 .
$$

The proof then conditions on the entire Gaussian sequence $( Z _ { t } ) _ { t \geq 0 }$ and attempts to apply this fixed-direction inequality to $M _ { \tau } ( Z _ { \tau } )$ . The dificulty is that conditioning fixes the whole sequence of directions, but it does not produce one common direction. Indeed, conditionally on $( Z _ { t } ) _ { t \geq 0 } = ( z _ { t } ) _ { t \geq 0 }$ <sub>0</sub>,

$$
M _ { \tau } ( Z _ { \tau } ) = \sum _ { t \geq 0 } { \mathbf 1 } \{ \tau = t \} M _ { t } ( z _ { t } ) ,
$$

whereas, for one fixed $x ,$

$$
M _ { \tau } ( x ) = \sum _ { t \geq 0 } \mathbf { 1 } \{ \tau = t \} M _ { t } ( x ) .
$$

In the first expression, the direction is $z _ { t }$ on the branch $\{ \tau = t \}$ ; in the second, the same direction $x$ is used on every possible branch. Conditioning makes each $z _ { t }$ deterministic, but it does not make the sequence $( z _ { t } ) _ { t \geq 0 }$

constant. Equivalently, the time-indexed process $M _ { t } ( z _ { t } )$ is not covered by the fixed-direction supermartingale result. Hence,

$$
\forall x \in \mathbb { R } ^ { d } , \qquad \mathbb { E } [ M _ { \tau } ( x ) ] \leq 1
$$

does not imply

$$
\mathbb { E } [ M _ { \tau } ( Z _ { \tau } ) ] \leq 1 .
$$

The proof then chooses τ as the first time at which the proposed self-normalized boundary is crossed (over a finite horizon $T _ { : }$ , its bounded truncation $\tau \wedge T ,$ )and applies Markov’s inequality using the claimed bound $\mathbb { E } [ \overline { { M } } _ { \tau } ] \leq 1$ . Since the expectation bound has not been established, the stopping-time argument does not prove the claimed time-uniform inequality.

## 5.3 What remains valid at a fixed deterministic time

Although the mixtures do not form one supermartingale over time, each mixture remains valid at a fixed deterministic time. Fix $t \geq 1$ . Since $h _ { t }$ is then one fixed probability density, the same Tonelli argument as in Section 4.3, together with $\mathbb { E } [ M _ { t } ( x ) ] \le 1$ for every fixed x, gives

$$
\mathbb { E } [ \overline { { M } } _ { t } ] = \int _ { \mathbb { R } ^ { d } } \mathbb { E } [ M _ { t } ( x ) ] h _ { t } ( x ) d x \leq \int _ { \mathbb { R } ^ { d } } h _ { t } ( x ) d x = 1 .
$$

Combining this with (12) and applying Markov’s inequality gives, for every fixed deterministic $t ,$

$$
\mathbb { P } \left( \| S _ { t } ^ { \gamma } \| _ { ( \widetilde { V } _ { t } ^ { \gamma } ) ^ { - 1 } } > R \sqrt { 2 \log \frac { \operatorname* { d e t } ( \widetilde { V } _ { t } ^ { \gamma } ) ^ { 1 / 2 } } { \delta \lambda ^ { d / 2 } } } \right) \le \delta .
$$

The determinant bound in Section 2 yields

$$
\mathbb { P } \left( \| S _ { t } ^ { \gamma } \| _ { ( \widetilde { V } _ { t } ^ { \gamma } ) ^ { - 1 } } > r _ { t } ^ { \gamma } ( \delta ) \right) \le \delta .
$$

Thus, the valid conclusion is that the inequality holds separately for every fixed deterministic t.

## 6 Related Work and Downstream Consequences

The literature presents a mixed picture. Some analyses apply the claimed anytime inequality directly, whereas others retain a fixed-time statement or explicitly pay for simultaneous control over a finite horizon. The latter treatments suggest that the dificulty created by the time index was likely recognized in parts of the literature. However, in preparing this note, we did not find a prior written discussion that both isolates the invalid stopping-time step and establishes that the resulting anytime claim is false. We therefore record this distinction here, especially because the claim has continued to appear in later analyses; the examples below are not intended to be exhaustive.

## 6.1 Scope of the counterexample and available repairs

The counterexample in Section 3 disproves the time-uniform weighted self-normalized inequality in (2), whose bounded radius contains no cost for time-uniformity. Therefore, every downstream proof that uses (2) as one event holding simultaneously for all times has a gap at that step. This does not by itself imply that the underlying algorithm or the polynomial order of its final regret bound is incorrect.

The pointwise inequality in (4) remains valid. For a known finite horizon $T ,$ applying it at confidence level $\delta / T$ and taking a union bound gives

$$
\begin{array} { r } { \mathbb { P } \left( \exists 1 \leq t \leq T : \| S _ { t } ^ { \gamma } \| _ { ( \widetilde { V } _ { t } ^ { \gamma } ) ^ { - 1 } } > r _ { t } ^ { \gamma } ( \delta / T ) \right) \leq \delta . } \end{array}
$$

Relative to $r _ { t } ^ { \gamma } ( \delta )$ , this repair adds 2 log T inside the expression under the square root. For an infinite horizon, the allocation in (5) gives a radius of order log t. If such replacement of the confidence radius is the only required modification, the correction changes only logarithmic factors and hence does not change the polynomial order hidden by Oe-notation. Whether this is suficient should be checked in each application, since the confidence radius may also afect the algorithm or its tuning.

## 6.2 Direct downstream uses

For each direct use below, we state the problem studied and the step that uses the invalid simultaneous inequality.

Russac et al. [2020]. This paper studies non-stationary generalized linear bandits with piecewise changes and develops sliding-window and discounted UCB algorithms. Its Corollary 5 restates the discounted time-uniform inequality of Russac et al. [2019]. The proof of Proposition 2 applies Corollary 5 to control the weighted noise term simultaneously for all $t ;$ Proposition 2 then yields the confidence bound used in Corollary 3 and the regret analysis of the discounted algorithm. The paper explicitly explains that, unlike the sliding-window analysis, the discounted analysis contains log(1/δ) rather than $\log ( T / \delta )$ because Corollary 5 is an anytime deviation inequality. Thus, both the simultaneous noise-control event and the resulting removal of the time-union cost are not justified.

Kim and Tewari [2020]. This paper develops randomized-exploration algorithms for non-stationary stochastic linear bandits. Its Lemma 4 restates Proposition 3 of Russac et al. [2019] as a confidence event $E _ { \mathrm { w l s } }$ claimed to hold simultaneously over $t \in [ T ]$ ]. This event is then used in the regret analyses of D-RandLinUCB and D-LinTS.

Touati and Vincent [2020]. This paper studies episodic reinforcement learning in non-stationary linear MDPs, where both the rewards and transition kernels are linear in known features and may evolve over episodes. Its Lemma 10 states the weighted self-normalized inequality with a deterministic time-varying regularizer and a simultaneous-in-t conclusion. The proof of Lemma 13 applies Lemma 10 to obtain uniform concentration for the estimated transition model; the resulting confidence control is then used in the proof of Theorem 1 and its dynamic-regret consequence in Corollary 1.

Faury et al. [2021]. This paper studies non-stationary generalized linear bandits. The proof of Lemma 1 directly applies Proposition 1 of Russac et al. [2019] to obtain a discounted noise bound holding for all $t \geq 1$ Lemma 1 is then used in Lemma 5 to control the prediction error, and this control enters the regret analysis in Theorem 1.

Deng et al. [2022]. This paper studies non-stationary Gaussian-process bandits and uses weighted Gaussianprocess regression to discount old observations. Its Lemmas 10 and 11 claim a weighted self-normalized bound holding simultaneously for all $t ,$ obtained by adapting the standard fixed-regularizer argument to the time-dependent regularizer $\alpha _ { t } I$ and then expressing the result in terms of weighted information gain. Lemma 9 shows that this framework recovers the weighted linear bandit of Russac et al. [2019] as a linear-feature special case. The resulting concentration bound is used in Theorems 2 and 3; Theorem 3 then enters the dynamic-regret analysis of Theorem 4.

Wang et al. [2023]. This paper revisits weighted strategies for non-stationary linear, generalized-linear, and self-concordant bandits. Its Theorem 5 restates the claimed weighted self-normalized inequality for a positive weight sequence $( w _ { s } ) _ { s \geq 1 }$ and time-varying scalar regularizers $( \mu _ { t } ) _ { t \geq 1 }$ . In the proof of Lemma 5, the authors set $w _ { s } = \gamma ^ { t - s - 1 }$ and $\mu _ { t } = \lambda$ and use Theorem 5 to obtain a bound holding for all t. Since this choice of $w _ { s }$ depends on the terminal time, it does not correspond to one fixed weight sequence across time, so the simultaneous conclusion does not directly follow from Theorem 5. Lemma 5 is then used in Lemma 1 and Theorem 1 for linear bandits; the analogous generalized-linear-bandit chain is Lemma 7, Lemma 2, and Theorem 2. In the self-concordant-bandit analysis, Theorem 6 is stated for a fixed time, while Lemma 3 uses it to obtain a conclusion for all t, which then enters Theorem 3.

Wang et al. [2026]. This paper studies how limited computational resources should be allocated among multiple learning tasks whose losses must reach prescribed targets by their deadlines. Because each task’s loss curve changes as training progresses, the authors use discounted least squares to estimate its current parameters, giving less weight to older observations that are less representative of its present progress. Theorem 2 in the appendix restates Theorem 1 of Russac et al. [2019]. Its proof sets $w _ { i } = \gamma ^ { n - i }$ and $\mu _ { n } = \lambda$ and then asserts that the resulting confidence bound holds for all n. Since this choice of $w _ { i }$ depends on the terminal time n, the simultaneous conclusion does not directly follow from the stated result. The resulting radius $\beta _ { n }$ is then used to guide the resource-allocation decision.

## 6.3 Fixed-time and finite-horizon control

The following works use fixed-time concentration or explicit finite-horizon confidence allocation in their final analyses. They therefore avoid relying on the claimed anytime radius, but do not provide a new anytime inequality.

Russac et al. [2021]. This paper studies self-concordant generalized linear bandits, including logistic and Poisson bandits, with forgetting implemented through a sliding window or exponential weights. Remark 2 notes that time-dependent regularization destroys the conditional supermartingale relation available with a fixed regularizer. Remark 3 further explains that the terminal-time dependence prevents the standard stopping-time argument from being applied. The paper therefore states its deviation result only at a fixed deterministic time and requires a union bound to control the entire trajectory.

Werge et al. [2026]. This paper studies non-stationary linear contextual bandits through weighted sequential Bayesian inference, maintaining a posterior distribution over the time-varying reward parameter. Appendix B of an earlier preprint version (arXiv v3) contains the same stopping-time error. In the later UAI 2026 version, the proof of Lemma 5 instead applies a fixed-time inequality at confidence level $\delta / T$ over the finite horizon and takes a union bound. The latest version therefore obtains simultaneous control by explicitly paying for the finite horizon rather than treating the fixed-time radius as an anytime radius.

## Acknowledgments

This work was supported by the Academia Sinica Postdoctoral Scholar Program, Grant No. AS-PD-1151- M15-2. The author thanks Melih Kandemir, Pei-Yuan Wu, Po-An Wang, Julian Zimmert, Yanlin Chen, and Nicklas Werge for helpful feedback and discussions on earlier versions of this note. The author also thanks the Institute of Statistics and Data Science, National Tsing Hua University, for providing access to workspace and facilities.

## References

Y. Abbasi-Yadkori, D. Pál, and C. Szepesvári. Improved algorithms for linear stochastic bandits. In Advances in Neural Information Processing Systems (NeurIPS), 2011.

Y. Deng, X. Zhou, B. Kim, A. Tewari, A. Gupta, and N. Shrof. Weighted gaussian process bandits for non-stationary environments. In Proceedings on the International Conference on Artificial Intelligence and Statistics (AISTATS), 2022.

L. Faury, Y. Russac, M. Abeille, and C. Calauzènes. Regret bounds for generalized linear bandits under parameter drift. arXiv preprint arXiv:2103.05750, 2021.

B. Kim and A. Tewari. Randomized exploration for non-stationary stochastic linear bandits. In Proceedings of the Conference on Uncertainty in Artificial Intelligence (UAI), 2020.

Y. Russac, C. Vernade, and O. Cappé. Weighted linear bandits for non-stationary environments. Advances in Neural Information Processing Systems (NeurIPS), 2019.

Y. Russac, O. Cappé, and A. Garivier. Algorithms for non-stationary generalized linear bandits. arXiv preprint arXiv:2003.10113, 2020.

Y. Russac, L. Faury, O. Cappé, and A. Garivier. Self-concordant analysis of generalized linear bandits with forgetting. In Proceedings on the International Conference on Artificial Intelligence and Statistics (AISTATS), 2021.

A. Touati and P. Vincent. Eficient learning in non-stationary linear markov decision processes. arXiv preprint arXiv:2010.12870, 2020.

J. Wang, P. Zhao, and Z.-H. Zhou. Revisiting weighted strategy for non-stationary parametric bandits. In Proceedings on the International Conference on Artificial Intelligence and Statistics (AISTATS), 2023.

J. Wang, X.-T. Liu, and Z.-H. Zhou. Core-learning with look-ahead and immediate resource allocation. In Proceedings of the AAAI Conference on Artificial Intelligence (AAAI), 2026.

N. Werge, Y.-S. Wu, A. Akgül, and M. Kandemir. Weighted sequential bayesian inference for non-stationary linear contextual bandits. In Proceedings of the Conference on Uncertainty in Artificial Intelligence $( U A I ) ,$ , 2026.

## A Proofs for Section 3

## A.1 Proof of Proposition 1

Proof. In one dimension, $\| S _ { t } ^ { \gamma } \| _ { ( \widetilde { V } _ { t } ^ { \gamma } ) ^ { - 1 } } = | S _ { t } ^ { \gamma } | / \sqrt { \widetilde { V } _ { t } ^ { \gamma } }$ . Thus, the proposed boundary is crossed at time t precisely when

$$
\lvert S _ { t } ^ { \gamma } \rvert > r _ { t } ^ { \gamma } ( \delta ) \sqrt { \widetilde { V } _ { t } ^ { \gamma } } .
$$

We first bound these time-dependent boundaries $r _ { t } ^ { \gamma } ( \delta ) \sqrt { \widetilde { V } _ { t } ^ { \gamma } }$ by one finite constant $C ,$ and then show that the process $( S _ { t } ) _ { t \geq 1 }$ cannot remain within $[ - C , C ]$ forever.

Since $r _ { t } ^ { \gamma } ( \delta )$ is bounded as discussed in Section 2, and $\widetilde { V } _ { t } ^ { \gamma } \to \lambda + ( 1 - \gamma ^ { 2 } ) ^ { - 1 }$ , the constant

$$
C \triangleq \operatorname* { s u p } _ { t \geq 1 } r _ { t } ^ { \gamma } ( \delta ) \sqrt { \widetilde V _ { t } ^ { \gamma } }
$$

is finite. $\mathrm { B y }$ definition, $r _ { t } ^ { \gamma } ( \delta ) \sqrt { \widetilde { V } _ { t } ^ { \gamma } } \leq C$ for every $t \geq 1$ . Consequently,

$$
| S _ { t } ^ { \gamma } | > C \implies \lVert S _ { t } ^ { \gamma } \rVert _ { ( \widetilde { V } _ { t } ^ { \gamma } ) ^ { - 1 } } > r _ { t } ^ { \gamma } ( \delta ) .
$$

It therefore sufices to prove

$$
\mathbb { P } \left( \exists t \geq 1 : | S _ { t } ^ { \gamma } | > C \right) = 1 .
$$

Let $q _ { C } \triangleq \mathbb { P } \big ( | \eta _ { 1 } | > ( 1 + \gamma ) C \big ) > 0$ , where positivity is due to $C < \infty$ and a nondegenerate Gaussian random variable has unbounded support. For any $m \geq 1$ , suppose that max $\begin{array} { r } { \vert \le t \le m \vert S _ { t } ^ { \gamma } \vert \le C } \end{array}$ . Since $S _ { 0 } ^ { \gamma } = 0$ the recursion implies, for every $1 \leq t \leq m$ ,

$$
| \eta _ { t } | = | S _ { t } ^ { \gamma } - \gamma S _ { t - 1 } ^ { \gamma } | \leq | S _ { t } ^ { \gamma } | + \gamma | S _ { t - 1 } ^ { \gamma } | \leq ( 1 + \gamma ) C .
$$

Therefore,

$$
\left\{ \operatorname* { m a x } _ { 1 \leq t \leq m } | S _ { t } ^ { \gamma } | \leq C \right\} \subseteq \bigcap _ { t = 1 } ^ { m } \left\{ | \eta _ { t } | \leq ( 1 + \gamma ) C \right\} .
$$

Using independence of the noises,

$$
\mathbb { P } \left( \operatorname* { m a x } _ { 1 \leq t \leq m } | S _ { t } ^ { \gamma } | \leq C \right) \leq \mathbb { P } \left( \bigcap _ { t = 1 } ^ { m } \left\{ | \eta _ { t } | \leq ( 1 + \gamma ) C \right\} \right) = ( 1 - q _ { C } ) ^ { m } \longrightarrow 0 .
$$

The finite-horizon events on the left decrease as m increases, and their intersection is the event that the process remains in $[ - C , C ]$ forever. Hence,

$$
\mathbb { P } \left( | S _ { t } ^ { \gamma } | \leq C { \mathrm { ~ f o r ~ e v e r y ~ } } t \geq 1 \right) = \operatorname* { l i m } _ { m \to \infty } \mathbb { P } \left( \operatorname* { m a x } _ { 1 \leq t \leq m } | S _ { t } ^ { \gamma } | \leq C \right) = 0 .
$$

Taking complements gives $\mathbb { P } \left( \exists t \geq 1 : | S _ { t } ^ { \gamma } | > C \right) = 1$

## A.2 Proof of Proposition 2

Proof. Let $\begin{array} { r } { V _ { \mathrm { m a x } } \triangleq \lambda + \frac { 1 } { 1 - \gamma ^ { 2 } } } \end{array}$ . Whenever the maximum in the proposition’s assumption is at most $C _ { T } ( \delta )$ , we have, for every $1 \leq t \leq T .$

$$
\begin{array} { r } { | S _ { t } ^ { \gamma } | \leq C _ { T } ( \delta ) \sqrt { \widetilde { V } _ { t } ^ { \gamma } } \leq C _ { T } ( \delta ) \sqrt { V _ { \operatorname* { m a x } } } . } \end{array}
$$

The last inequality uses $\widetilde { V } _ { t } ^ { \gamma } \leq V _ { \operatorname* { m a x } }$ . Hence, the assumption of $C _ { T } ( \delta )$ in the proposition implies

$$
\mathbb { P } \left( \operatorname* { m a x } _ { 1 \leq t \leq T } | S _ { t } ^ { \gamma } | \leq C _ { T } ( \delta ) \sqrt { V _ { \operatorname* { m a x } } } \right) \geq 1 - \delta .
$$

The bound on the right-hand side is deterministic, while the recursion depends on $T$ independent Gaussian noises. As in the proof of Proposition 1,

$$
\operatorname* { m a x } _ { 1 \leq t \leq T } | S _ { t } ^ { \gamma } | \leq C _ { T } ( \delta ) \sqrt { V _ { \operatorname* { m a x } } } \quad \implies \quad | \eta _ { t } | \leq ( 1 + \gamma ) C _ { T } ( \delta ) \sqrt { V _ { \operatorname* { m a x } } } \quad \mathrm { f o r ~ e v e r y ~ } 1 \leq t \leq T .
$$

Thus, by independence of the noises,

$$
1 - \delta \leq \mathbb { P } \left( \operatorname* { m a x } _ { 1 \leq t \leq T } \lvert S _ { t } ^ { \gamma } \rvert \leq C _ { T } ( \delta ) \sqrt { V _ { \operatorname* { m a x } } } \right) \leq \left[ \mathbb { P } \left( \lvert \eta _ { 1 } \rvert \leq ( 1 + \gamma ) C _ { T } ( \delta ) \sqrt { V _ { \operatorname* { m a x } } } \right) \right] ^ { T } .
$$

Thus, to make this probability at least $1 - \delta .$ , the probability for each noise to remain in that interval must be at least $( 1 - \delta ) ^ { 1 / T }$ . This condition will force $C _ { T } ( \delta )$ to be large. Since $\eta _ { 1 } / R \sim \mathcal { N } ( 0 , 1 )$ , this implies

$$
2 \Phi \left( \frac { ( 1 + \gamma ) C _ { T } ( \delta ) \sqrt { V _ { \operatorname* { m a x } } } } { R } \right) - 1 \ge ( 1 - \delta ) ^ { 1 / T } ,
$$

where we recall that for a standard normal random variable $Z , \Phi ( z )$ is the cumulative distribution function, and $2 \Phi ( z ) - 1$ is the probability that $| Z | \le z$ . Solving for $C _ { T } ( \delta )$ gives the displayed lower bound.

To obtain the order statement, we first let $\begin{array} { r } { a _ { T } \triangleq \frac { ( 1 + \gamma ) C _ { T } ( \delta ) \sqrt { V _ { \operatorname* { m a x } } } } { R } } \end{array}$ . The previous formula can be rewritten as

$$
\Phi ( a _ { T } ) \geq \frac { 1 + ( 1 - \delta ) ^ { 1 / T } } { 2 } .
$$

Since $\mathbb { P } ( Z > a _ { T } ) = 1 - \Phi ( a _ { T } )$ , taking complements $\mathrm { g i }$ ves

$$
\mathbb { P } ( Z > a _ { T } ) \le \frac { 1 - ( 1 - \delta ) ^ { 1 / T } } { 2 } \le \frac { - \log ( 1 - \delta ) } { 2 T } \le \frac { \delta } { T } ,
$$

where we used $1 - e ^ { - x } \leq x$ and $- \log ( 1 - \delta ) \leq 2 \delta$ for $\delta \in ( 0 , 1 / 2 ]$ . Thus, the threshold $a _ { T }$ must make the one-sided Gaussian tail no larger than $\delta / T . \mathrm { ~ A ~ }$ standard Gaussian tail lower bound shows that there are universal constants $c > 0$ and $u _ { 0 } > 0$ such that, whenever $u \leq u _ { 0 }$

$$
\mathbb { P } ( Z > a ) \leq u \qquad \Longrightarrow \qquad a \geq c \sqrt { \log \frac { 1 } { u } } .
$$

When $T / \delta$ is suficiently large, we may apply this bound with $u = \delta / T$ . It gives

$$
a _ { T } \geq c \sqrt { \log \frac { T } { \delta } } .
$$

Recalling the definition of $a _ { T }$ proves the final claim.