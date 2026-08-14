# Active-Trace Complexity Bounds for Moreau–Yosida Unadjusted Langevin Sampling

Yuchen Xin<sup>\*</sup>, Zhihua Zhang<sup>†</sup>

## Abstract

We study the Moreau–Yosida unadjusted Langevin algorithm (MYULA) for the nonsmooth composite target

$$
\pi ( d x ) \propto \exp \{ - f ( x ) - g ( x ) \} d x , \qquad x \in \mathbb { R } ^ { d } ,
$$

where $f$ is m-strongly convex with $L _ { f }$ -Lipschitz gradient and $g$ is convex and G-Lipschitz. Let $g _ { \lambda }$ be the Moreau envelope of $g , \pi \lambda$ the corresponding smoothed target, and $a _ { \lambda } = \operatorname { t r } H _ { \lambda } .$ , where $H _ { \lambda }$ is the $\mathrm { a . e . } / $ weak Hessian of $g _ { \lambda }$ . We show that the leading MYULA discretization error is controlled by the reference active trace $B _ { \mathrm { r e f } }$ , the average of $a _ { \lambda }$ along the heat substep of one MYULA update started from $\pi _ { \lambda } ,$ rather than by the global curvature bound $d / \lambda$ . If $M _ { \lambda }$ is an a.e. upper bound for $a \lambda \mathrm { ; }$ , then, up to logarithmic factors,

$$
N \lesssim \frac { 1 } { m } \left[ L _ { f } + \frac { \tau _ { f } + G ^ { 2 } + B _ { \mathrm { r e f } } } { \varepsilon _ { \mathrm { a l g } } ^ { 2 } } + \frac { M _ { \lambda } } { \varepsilon _ { \mathrm { a l g } } } \right] , \qquad \tau _ { f } : = \operatorname* { s u p } _ { x } \mathrm { t r } \nabla ^ { 2 } f ( x ) ,
$$

iterations sufice to ensure $\sqrt { m } W _ { 2 } ( \mu _ { N } , \pi _ { \lambda } ) \leq \varepsilon _ { \mathrm { a l g } }$ , where $\mu _ { N }$ is the law of the N-th iterate and $W _ { 2 }$ is the quadratic Wasserstein distance. We also prove the Moreau-bias bound

$$
\sqrt { m } W _ { 2 } ( \pi _ { \lambda } , \pi ) \leq \frac { G ^ { 2 } \lambda } { 4 } .
$$

Thus, choosing $\lambda \times \varepsilon / G ^ { 2 }$ gives an end-to-end guarantee for π. The universal estimate $B _ { \mathrm { r e f } } \leq d / \lambda$ yields $\widetilde { \cal O } ( \varepsilon ^ { - 3 } )$ accuracy dependence. For the structured piecewise-linear, lasso-type, group, and total-variation penalties considered here, curvature–tube estimates make $B _ { \mathrm { r e f } }$ independent of $\lambda ,$ yielding $\widetilde { O } ( \varepsilon ^ { - 2 } )$ for the same classical MYULA kernel.

## Contents

1 Introduction 2   
2 Related Work 6   
2.1 Proximal MCMC and the original analysis of myula . 6   
2.2 Average smoothness and trace-sensitive Langevin bounds 9   
2.3 Proximal splitting methods for the original composite target . 10   
2.4 Active manifolds, proximal Jacobians, and degrees of freedom 10   
2.5 Alternative envelopes and smoothing-based Langevin methods . 11   
2.6 Direct nonsmooth sampling and stronger oracle models . 11   
3 Preliminaries 12   
3.1 Basic notation 12   
3.2 Nonsmooth calculus for Lipschitz and convex functions . 13   
3.3 Moreau–Yosida regularization and weak second-order structure 13   
3.4 The heat energy identity for $C ^ { 1 , 1 }$ test functions 14   
3.5 Entropy EVI for the heat flow . 15   
4 Problem Setup and Standing Assumptions 15   
5 Main Result: Reference Active Trace and Total Error 16   
5.1 Universal Active-Trace KL–EVI Recursion . 17   
5.2 Closing the Recursion by Active-Trace Transfer 19   
5.3 Moreau Approximation Bias . 21   
5.4 Proof of the main theorem 22   
6 Bounding the Reference Active Trace 22   
6.1 An abstract curvature-tube interface 23   
6.2 Slice density of $\pi _ { \lambda }$ 24   
6.3 Density propagation by $T _ { h }$ and by the heat step 25   
6.4 Slab, ball, and inverse-radius consequences 27   
Examples and Verification Modules 28   
7.1 One-dimensional finite-kink piecewise-linear penalties . 29   
7.2 Separable finite-kink PL penalties and weighted lasso 31   
7.3 Group lasso 32   
7.4 Generalized lasso and anisotropic total variation 34   
8 Conclusion 38   
A Real-Analysis Primer for A.e. and Weak Hessians 38   
A.1 Rademacher’s theorem . . 38   
A.2 A.e. Hessians, weak derivatives, and the $L ^ { \infty }$ weak Hessian 39   
A.3 Symmetry, convexity, and matrix bounds 39   
B Details for Moreau Weak Second-Order Regularity 40   
C Proof of the Heat Identity for $C ^ { 1 , 1 }$ Test Functions 40

## 1 Introduction

Sampling from a probability distribution with density

$$
\pi ( x ) \propto \exp \{ - f ( x ) - g ( x ) \} , \qquad x \in \mathbb { R } ^ { d } ,
$$

is a fundamental computational task in Bayesian inverse problems, high-dimensional statistics, and machine learning. The composite form is particularly common when the smooth term f represents a data-fidelity or negative log-likelihood and the convex, possibly nonsmooth term g encodes structural information such as sparsity, group sparsity, analysis sparsity, or total variation. Proximal optimization methods can exploit this structure directly, but conventional gradient-based

Langevin algorithms require a diferentiable potential with a suficiently regular gradient. This mismatch has motivated a broad class of proximal and smoothing-based Markov chain Monte Carlo methods; see, among others, [22, 6, 4].

A particularly influential method is the Moreau–Yosida unadjusted Langevin algorithm (myula) of [6]. It replaces $g$ by its Moreau envelope $g _ { \lambda }$ , targets the smoothed measure $\pi _ { \lambda }$ (see (4.2)), and applies the Euler–Maruyama scheme (see (4.8)). The resulting drift is explicit whenever one can evaluate $\nabla f$ and $\operatorname { p r o x } _ { \lambda g }$ , since

$$
\nabla g _ { \lambda } ( x ) = \lambda ^ { - 1 } \big ( x - \mathrm { p r o x } _ { \lambda g } ( x ) \big ) .
$$

This construction has proved especially useful in imaging and sparse Bayesian inference because it converts proximal primitives already available for optimization into a simple Langevin sampler.

The standard smoothness description of the Moreau envelope is nevertheless potentially pessimistic. Globally,

$$
\mathrm { L i p } ( \nabla g _ { \lambda } ) \leq \lambda ^ { - 1 } , \qquad 0 \preceq \nabla ^ { 2 } g _ { \lambda } ( x ) \preceq \lambda ^ { - 1 } I \quad \mathrm { f o r ~ a . e . ~ } x ,
$$

so a worst-case analysis treats the full trace of the Moreau curvature as being as large as $d / \lambda$ everywhere. At the same time, decreasing λ is necessary to reduce the discrepancy between $\pi _ { \lambda }$ and the original target $\pi$ . This creates the familiar tension between regularization bias and discretization stability: a smaller smoothing parameter improves the target approximation but apparently makes the Langevin dynamics uniformly stifer.

For many structured penalties, however, the curvature $\lambda ^ { - 1 }$ is not present throughout the state space. For the scalar absolute value, for example, $g _ { \lambda } ^ { \prime \prime } ( x ) = \lambda ^ { - 1 }$ only inside a threshold interval of width $O ( \lambda )$ and vanishes outside that interval. Coordinatewise piecewise-linear penalties generate thin slabs; group penalties generate a small central ball together with integrable tangential curvature; and polyhedral analysis penalties generate neighborhoods of active faces. Thus, the largest Moreau curvature and the probability of encountering it are coupled. A bound based only on su $\mathrm { p } _ { x } \| \nabla ^ { 2 } g _ { \lambda } ( x ) \|$ discards this coupling and charges every Langevin step for curvature that may be visited only with probability $O ( \lambda )$

This paper develops a distribution-dependent analysis that retains that coupling. Let $H _ { \lambda }$ be a measurable representative of the a.e./weak Hessian of the $C ^ { 1 , 1 }$ function $g _ { \lambda }$ . We introduce the Moreau active trace (see Definition 3.2)

$$
a _ { \lambda } ( x ) : = \operatorname { t r } H _ { \lambda } ( x ) = { \frac { 1 } { \lambda } } \operatorname { t r } \bigl ( I - D \operatorname { p r o x } _ { \lambda g } ( x ) \bigr ) \quad { \mathrm { f o r ~ a . e . ~ } } x .
$$

In addition to being the trace of the local Moreau curvature, the quantity $\lambda a _ { \lambda } ( x )$ is the total local shrinkage of the proximal map. Therefore, it describes how many directions are locally suppressed by the active structure of $g .$ Although the universal bound $a _ { \lambda } \leq d / \lambda$ always holds, the distributional average of $a _ { \lambda }$ can be much smaller.

The relevant average in our analysis is determined by one myula step. If $X _ { \mathrm { r e f } } \sim \pi _ { \lambda }$ , the deterministic Euler map is $T _ { h } ( x ) = x - h \nabla ( f + g _ { \lambda } ) ( x )$ , and $W _ { t }$ is a Brownian motion independent $X _ { \mathrm { r e f } }$ . Define

$$
B _ { \mathrm { r e f } } : = \frac { 1 } { h } \int _ { 0 } ^ { h } \mathbb { E } \Big [ a _ { \lambda } \Big ( T _ { h } ( X _ { \mathrm { r e f } } ) + \sqrt { 2 } W _ { t } \Big ) \Big ] \ \mathrm { d } t ,
$$

which we call the reference heat-path active trace (see Definition 5.1 ). The definition is algorithmaware: it averages the weak Moreau curvature over the stationary input, the Euler drift, and the Gaussian heat interpolation that together form a single discretization step. It is also the quantity

that appears naturally when the change of the potential along the heat step is written as an integral of its weak Laplacian.

Our main theorem shows that this pathwise average, rather than the global trace bound, controls the leading nonsmooth discretization term. In particular, under the standing assumptions of Section 4, Theorem 5.2 reads

$$
\Phi _ { N } \lesssim e ^ { - m h N / 2 } \Phi _ { 0 } + \frac { 1 } { m } \left\{ ( \tau _ { f } + G ^ { 2 } + B _ { \mathrm { r e f } } ) h + M _ { \lambda } ^ { 2 } h ^ { 2 } \right\} , \qquad \Phi _ { k } = W _ { 2 } ^ { 2 } ( \mu _ { k } , \pi _ { \lambda } ) + 2 h \mathrm { K L } ( \mu _ { k } \| \pi _ { \lambda } ) ,
$$

whenever $0 \le a _ { \lambda } \le M _ { \lambda }$ a.e. Equivalently, the fixed-λ iteration complexity (see (5.36)) is, up to logarithmic factors,

$$
N _ { \lambda } ( \varepsilon _ { \mathrm { a l g } } ) \lesssim \frac { 1 } { m } \left( L _ { f } + \frac { \tau _ { f } + G ^ { 2 } + B _ { \mathrm { r e f } } } { \varepsilon _ { \mathrm { a l g } } ^ { 2 } } + \frac { M _ { \lambda } } { \varepsilon _ { \mathrm { a l g } } } \right) .
$$

The leading $\varepsilon _ { \mathrm { a l g } } ^ { - 2 }$ term depends on the reference active trace; the worst-case curvature $M _ { \lambda }$ survives only in a lower order transfer term. Proposition 5.9 separately proves

$$
\sqrt { m } W _ { 2 } ( \pi _ { \lambda } , \pi ) \leq \frac { G ^ { 2 } \lambda } { 4 } ,
$$

so the fixed-λ estimate can be combined with a transparent regularization-bias choice.

This decomposition makes the structural gain explicit. With the symmetric error split and $\lambda \times \varepsilon / G ^ { 2 }$ , replacing $B _ { \mathrm { r e f } }$ by the universal bound $d / \lambda$ in our own theorem produces a conservative contribution of order $d G ^ { 2 } \varepsilon ^ { - 3 }$ . In contrast, if $B _ { \mathrm { r e f } }$ is bounded independently of $\lambda ,$ this cubic contribution disappears and the structured examples of Section 7 have $\tilde { O } ( \varepsilon ^ { - 2 } )$ dependence on the target accuracy, with the remaining dimension and geometry factors displayed explicitly in Propositions 7.2–7.9 and Corollaries 7.4 and 7.10.

Section 2.1 places this gain against a metric- and assumption-aligned global-smoothness benchmark. Applying the Wasserstein–EVI analysis of [4, Corollary 10] to the same smoothed potential $U _ { \lambda } .$ , followed by the end-to-end choice $\lambda \asymp \varepsilon / G ^ { 2 }$ , gives $\widetilde { O } ( \varepsilon ^ { - 3 } )$ dependence, matching the universal global-trace specialization of our theorem. For the structured penalties considered here, the active-trace bounds replace this global curvature charge by an occupation-weighted quantity and yield $\widetilde { O } ( \varepsilon ^ { - 2 } )$ without changing the classical myula transition. The same section explains why the $\varepsilon ^ { - 2 } .$ -type total-variation bounds of [6] are fixed-λ statements rather than λ-free end-to-end rates.

The mechanism for bounding $B _ { \mathrm { r e f } }$ is geometric. Proposition 6.3 formalizes a curvature–tube interface: if a component of $a _ { \lambda }$ has height $O ( \lambda ^ { - 1 } )$ inside an $O ( \lambda )$ neighborhood of an active stratum $\Sigma ,$ and the reference heat path assigns mass $O ( r ^ { q } )$ to an r-tube around $\Sigma .$ then this component contributes $O ( \lambda ^ { q - 1 } )$ to $B _ { \mathrm { r e f } }$ . Codimension-one layers therefore contribute $O ( 1 )$ , while higher-codimension central regions can contribute even less as $\lambda \downarrow 0$ . The slice-density and densitypropagation estimates in Lemmas 6.5 and 6.6 convert this interface into verifiable bounds without invoking the global Moreau smoothness scale.

The examples illustrate several forms of active geometry. For one-dimensional finite-kink and separable piecewise-linear penalties, the Moreau curvature is confined to intervals or coordinate slabs whose total width is $O ( \lambda )$ . For the group lasso, a small-ball estimate controls the central $q / \lambda$ curvature and an inverse-radius estimate controls the tangential part. For generalized lasso and anisotropic total variation, the proximal map is afine on polyhedral cells, with

$$
D \operatorname { p r o x } _ { \lambda g } ( x ) = P _ { \ker D _ { A } } , \qquad a _ { \lambda } ( x ) = \frac { \operatorname { r a n k } ( D _ { A } ) } { \lambda } ,
$$

where A is the active row set; the corresponding curvature is localized by thin row slabs. These calculations connect the sampling error to active-set rank, block geometry, and tube probabilities rather than to a uniform $d / \lambda$ penalty.

Conceptually, the closest smooth analogue is the recent average-smoothness analysis of Langevin Monte Carlo [3]. That work shows that the leading LMC discretization term can depend on an average of coordinatewise smoothness constants instead of the largest global smoothness constant. Both results reflect the fact that isotropic Gaussian noise naturally interacts with a trace-type measure of curvature. The distinction is that average smoothness averages over directions while retaining a supremum over spatial locations. The reference active trace additionally averages over the locations actually visited by the one-step heat path. This extra spatial averaging is essential for Moreau envelopes of lasso-type penalties: every active coordinate may have worst-case curvature $1 / \lambda$ , even though that curvature is confined to a slab with probability $O ( \lambda )$

Our objective is neither to propose a new sampler nor to give the first $\widetilde { O } ( \varepsilon ^ { - 2 } )$ result for nonsmooth composite sampling. Proximal splitting Langevin methods such as pgla and psgla update the nonsmooth component directly and admit strong guarantees for the original composite target [4, 26]. Other methods use alternative envelopes, subgradient dynamics, Metropolis corrections, or stronger restricted-Gaussian oracles. These methods change the transition kernel, the target treated by each step, or the oracle model. In contrast, the present work keeps the classical myula iteration and asks a narrower question: which part of the Moreau curvature is actually paid for by its discretization error?

The main contributions can be summarized as follows.

1. We derive a universal KL–EVI recursion for myula in which the deterministic gradient-step error depends on the Lipschitz magnitude $G ^ { 2 }$ of the nonsmooth penalty, rather than on a power of $\lambda ^ { - 1 }$ ; see Lemma 5.4 and Theorem 5.5.

2. We introduce the reference heat-path active trace and close the recursion through an activetrace transfer argument; see Proposition 5.6 and Theorem 5.7. Together with the Wasserstein Moreau-bias estimate in Proposition 5.9, this yields a complete guarantee for the original nonsmooth target.

3. We develop a curvature–tube principle and accompanying slice-density estimates that turn localization of weak Moreau curvature into quantitative bounds on $B _ { \mathrm { r e f } } ;$ see Section 6.

4. We verify the framework for finite-kink piecewise-linear penalties, weighted lasso, group lasso, generalized lasso, and anisotropic total variation. In these examples $B _ { \mathrm { r e f } }$ is bounded independently of λ, removing the cubic accuracy contribution generated by the universal global-trace substitution.

The remainder of the paper is organized as follows. Section 2 presents the related work. Section 3 fixes the nonsmooth second-order conventions and the heat-flow tools used throughout. Section 4 introduces the composite target, the myula update, and the standing assumptions. Section 5 proves the active-trace recursion, closes it through the reference-path transfer, and controls the Moreau approximation bias. Section 6 develops the curvature–tube and density-propagation machinery. Section 7 treats the structured examples, and the appendices provide the real-analysis details needed for a.e. and weak Hessians.

## 2 Related Work

## 2.1 Proximal MCMC and the original analysis of myula

Proximal MCMC and the original MYULA bounds. Proximal MCMC was introduced by [22] to make gradient-based sampling applicable to log-concave but nonsmooth models whose proximal mappings are computationally accessible. The Moreau–Yosida unadjusted Langevin algorithm (myula) of [6] applies ULA to

$$
U _ { \lambda } = f + g _ { \lambda } , \qquad \pi _ { \lambda } ( \mathrm { d } x ) \propto \exp \{ - U _ { \lambda } ( x ) \} \mathrm { d } x .
$$

Its classical analysis uses the global regularity estimate

$$
L _ { \lambda } : = \mathrm { L i p } ( \nabla U _ { \lambda } ) \leq L _ { f } + \lambda ^ { - 1 } , \qquad \gamma \leq L _ { \lambda } ^ { - 1 } = { \frac { \lambda } { 1 + \lambda L _ { f } } } ,\tag{2.1}
$$

and, when $g$ is G-Lipschitz, the regularization bound

$$
\| \pi _ { \lambda } - \pi \| _ { \mathrm { T V } } \leq \lambda G ^ { 2 } .\tag{2.2}
$$

Let $\mu _ { n }$ be the law of the MYULA iterate, and let $\varepsilon _ { \mathrm { a l g } }$ denote the prescribed error to the fixed smoothed target:

$$
\| \mu _ { n } - \pi _ { \lambda } \| _ { \mathrm { T V } } \leq \varepsilon _ { \mathrm { a l g } } .\tag{2.3}
$$

There is a minor notational carryover in the presentation of Theorems 2–3 of [6]. Their displayed total-variation conclusions use the symbol $\pi _ { \ i }$ , but the transition kernel $R _ { \gamma }$ is generated by $U _ { \lambda } = f + g _ { \lambda }$ and the proofs apply the generic smooth-ULA results of [5] with $U = U _ { \lambda }$ . The target after this specialization is therefore the corresponding Gibbs law $\pi _ { \lambda } .$ with the approximation error ${ \left| { \left| \pi _ { \lambda } - \pi \right| } \right| } _ { \mathrm { T V } }$ controlled separately in Proposition 1. Accordingly, we read the displayed conclusions as fixed-λ bounds to $\pi _ { \lambda } .$ , not as end-to-end bounds to $\pi .$

Under this proof-supported interpretation, and renaming the prescribed fixed-target tolerance as $\varepsilon _ { \mathrm { a l g } }$ , [6] reports the worst-case iteration dependences

$$
O \left( d ^ { 5 } \log ^ { 2 } ( \varepsilon _ { \mathrm { a l g } } ^ { - 1 } ) \varepsilon _ { \mathrm { a l g } } ^ { - 2 } \right)
$$

under its general coercive-tail condition H3, and

$$
O \Big ( d \log ( d ) \log ^ { 2 } ( \varepsilon _ { \mathrm { a l g } } ^ { - 1 } ) \varepsilon _ { \mathrm { a l g } } ^ { - 2 } \Big )
$$

under the stronger tail-convexity condition H4. These displayed orders describe the dependence on the fixed-λ algorithmic tolerance; the smoothness parameter $L _ { \lambda }$ , together with the initialization, Lyapunov, tail, and convexity constants in the detailed bounds, remains part of the problem dependence.

Fixed-λ versus end-to-end accuracy. A guarantee for the original nonsmooth target requires combining the fixed-λ algorithmic error with the Moreau regularization bias:

$$
\begin{array} { r l } & { \| \mu _ { n } - \pi \| _ { \mathrm { T V } } \leq \| \mu _ { n } - \pi _ { \lambda } \| _ { \mathrm { T V } } + \| \pi _ { \lambda } - \pi \| _ { \mathrm { T V } } } \\ & { \qquad \leq \varepsilon _ { \mathrm { a l g } } + \lambda G ^ { 2 } . } \end{array}\tag{2.4}
$$

For example, to obtain a total tolerance ε, a constant-fraction error allocation is

$$
\varepsilon _ { \mathrm { a l g } } = { \frac { \varepsilon } { 2 } } , \qquad \lambda \leq { \frac { \varepsilon } { 2 G ^ { 2 } } } , \qquad L _ { \lambda ( \varepsilon ) } = O \left( L _ { f } + { \frac { G ^ { 2 } } { \varepsilon } } \right) .\tag{2.5}
$$

Consequently, the published $\varepsilon _ { \mathrm { a l g } } ^ { - 2 }$ summaries in [6] should not be read as λ-free end-to-end $\widetilde { O } ( \varepsilon ^ { - 2 } )$ rates for the original target. For example, the explicit $L _ { \lambda ( \varepsilon ) } ^ { 2 } \varepsilon _ { \mathrm { a l g } } ^ { - 2 }$ factor in the underlying step-size bound formally becomes

$$
{ \cal O } \left[ \left( L _ { f } + \frac { G ^ { 2 } } { \varepsilon } \right) ^ { 2 } \varepsilon ^ { - 2 } \right] ,\tag{2.6}
$$

whose highest-order visible term is $O ( G ^ { 4 } \varepsilon ^ { - 4 } )$ . This is only a schematic substitution: the remaining constants in [6] were not optimized uniformly in $\lambda .$

For compactly supported log-concave targets, [2] performs the smoothing–discretization balance explicitly. There $g$ is an indicator function, the smoothing bias has a diferent form, and λ is chosen of order $\epsilon ^ { 2 }$ , up to dimension and geometric factors, leading to $\widetilde { O } ( \varepsilon ^ { - 6 } )$ complexity in both total variation and $W _ { 1 }$

A global-smoothness benchmark in the Wasserstein–EVI framework. A closely aligned benchmark follows directly from [4, Corollary 10]: after setting $U = U _ { \lambda }$ , it applies to the same ULA kernel and smoothed target in $W _ { 2 }$ under strong convexity, and its underlying proof uses the same gradient-step/heat-step decomposition. For an m-strongly convex potential with L-Lipschitz gradient, that result states that, for any squared-Wasserstein tolerance $\delta > 0$ , the constant-step choices

$$
h _ { \delta } \leq \operatorname* { m i n } \left. \frac { m \delta } { 4 L d } , \frac { 1 } { L } \right. , \qquad N _ { \delta } \geq \frac { 1 } { m h _ { \delta } } \log \left( \frac { 2 W _ { 2 } ^ { 2 } ( \mu _ { 0 } , \pi ) } { \delta } \right)\tag{2.7}
$$

ensure

$$
W _ { 2 } ^ { 2 } ( \mu _ { N _ { \delta } } , \pi ) \le \delta .
$$

Here $\delta$ is $\mathrm { a }$ tolerance for $W _ { 2 } ^ { 2 }$ , whereas our accuracy parameter controls $\sqrt { m } W _ { 2 }$

Applying (2.7) to $U _ { \lambda } = f + g _ { \lambda }$ , with

$$
L = L _ { \lambda } \leq L _ { f } + \lambda ^ { - 1 } ,
$$

and setting

$$
\delta = \frac { \varepsilon _ { \mathrm { a l g } } ^ { 2 } } { m }
$$

gives

$$
h \lesssim \operatorname* { m i n } \left\{ L _ { \lambda } ^ { - 1 } , \frac { \varepsilon _ { \mathrm { a l g } } ^ { 2 } } { d L _ { \lambda } } \right\}\tag{2.8}
$$

and, up to logarithmic factors,

$$
N _ { \lambda } ^ { \mathrm { g l o b } } ( \varepsilon _ { \mathrm { a l g } } ) = \widetilde { O } \left[ \frac { L _ { \lambda } } { m } \left( 1 + \frac { d } { \varepsilon _ { \mathrm { a l g } } ^ { 2 } } \right) \right] = \widetilde { O } \left[ \frac { L _ { f } + \lambda ^ { - 1 } } { m } \left( 1 + \frac { d } { \varepsilon _ { \mathrm { a l g } } ^ { 2 } } \right) \right] .\tag{2.9}
$$

To obtain an end-to-end guarantee for the original nonsmooth target, combine (2.9) with the Wasserstein Moreau-bias estimate in Theorem 5.2,

$$
\sqrt { m } W _ { 2 } ( \pi _ { \lambda } , \pi ) \leq \frac { G ^ { 2 } \lambda } { 4 } .\tag{2.10}
$$

With the symmetric allocation

$$
\varepsilon _ { \mathrm { a l g } } = \varepsilon _ { \mathrm { b i a s } } = \frac { \varepsilon } { 2 } , \qquad \lambda \asymp \frac { \varepsilon } { G ^ { 2 } } ,
$$

equation (2.9) yields

$$
N _ { \mathrm { g l o b } } ( \varepsilon ) = \widetilde O \left[ \frac { 1 } { m } \left( L _ { f } + \frac { G ^ { 2 } } { \varepsilon } \right) \left( 1 + \frac { d } { \varepsilon ^ { 2 } } \right) \right] = \widetilde O \left( \frac { d L _ { f } } { m \varepsilon ^ { 2 } } + \frac { d G ^ { 2 } } { m \varepsilon ^ { 3 } } \right) ,\tag{2.11}
$$

where lower-order terms are suppressed for $d \geq 1$ and $0 < \varepsilon \le 1$

Thus, within the Wasserstein–EVI framework of [4], applying the global smoothness $L _ { \lambda } \ \leq$ $L _ { f } + \lambda ^ { - 1 }$ to the Moreau-smoothed target yields the end-to-end benchmark $\widetilde { \cal O } ( \varepsilon ^ { - 3 } )$

Active-trace refinement within the same proof architecture. In [4], the global constant L enters both the deterministic-step stability estimate and the heat-step energy increment; for $U = U _ { \lambda }$ , both are therefore charged through $L _ { \lambda } \leq L _ { f } + \lambda ^ { - 1 }$ . Our analysis separates these two roles: the deterministic step uses $\| \nabla g _ { \lambda } \| \le G$ with the baseline restriction $h \lesssim L _ { f } ^ { - 1 }$ , whereas the heat increment is controlled by $\tau _ { f } + B _ { k }$ rather than $d L _ { \lambda }$ . Transferring $B _ { k }$ to the stationary reference path yields the fixed-λ complexity of Theorem 5.2,

$$
N _ { \lambda } ( \varepsilon _ { \mathrm { a l g } } ) \lesssim \frac { 1 } { m } \left[ L _ { f } + \frac { \tau _ { f } + G ^ { 2 } + B _ { \mathrm { r e f } } } { \varepsilon _ { \mathrm { a l g } } ^ { 2 } } + \frac { M _ { \lambda } } { \varepsilon _ { \mathrm { a l g } } } \right] ,\tag{2.12}
$$

up to logarithmic factors.

If no active-set information is used, the universal estimates

$$
B _ { \mathrm { r e f } } \leq M _ { \lambda } \leq \frac { d } { \lambda }
$$

and the choice $\lambda \asymp \varepsilon / G ^ { 2 }$ give

$$
\frac { B _ { \mathrm { r e f } } } { \varepsilon ^ { 2 } } \lesssim \frac { d G ^ { 2 } } { \varepsilon ^ { 3 } } .\tag{2.13}
$$

Thus the global-trace specialization of our theorem matches the $\varepsilon ^ { - 3 }$ dependence obtained by applying the Wasserstein–EVI bound to $U _ { \lambda }$ , while resolving the smooth contribution through $\tau _ { f }$ rather than $d L _ { f }$

For the structured penalties studied in Sections 6–7, the curvature–tube estimates instead give

$$
B _ { \mathrm { r e f } } \leq A _ { g } ,\tag{2.14}
$$

where $A _ { g }$ is independent of λ. If, in addition,

$$
M _ { \lambda } \leq \frac { r _ { g } } { \lambda } ,
$$

then

$$
\frac { M _ { \lambda } } { \varepsilon } \lesssim \frac { r _ { g } G ^ { 2 } } { \varepsilon ^ { 2 } } \qquad \mathrm { w h e n } ~ \lambda \asymp \frac { \varepsilon } { G ^ { 2 } } .\tag{2.15}
$$

Substitution into (2.12) therefore gives

$$
N _ { \mathrm { a c t i v e } } ( \varepsilon ) = \widetilde { \cal O } \left[ \frac { 1 } { m } \left\{ L _ { f } + \frac { \tau _ { f } + A _ { g } + ( 1 + r _ { g } ) G ^ { 2 } } { \varepsilon ^ { 2 } } \right\} \right] .\tag{2.16}
$$

Suppressing dimensions, problem constants, and logarithms, the resulting accuracy comparison is

$$
\mathrm { W a s s e r s t e i n - E V I ~ b o u n d ~ a p p l i e d ~ g l o b a l l y ~ t o ~ } U _ { \lambda } ~ : ~ \widetilde { O } ( \varepsilon ^ { - 3 } ) ,\tag{2.17}
$$

$$
\mathrm { s t r u c t u r e d ~ a c t i v e - t r a c e ~ a n a l y s i s ~ o f ~ t h e ~ s a m e ~ M Y U L A ~ k e r n e l ~ \chi _ { \chi } ~ } \widetilde { O } ( \varepsilon ^ { - 2 } ) .
$$

The gain comes from replacing the global $O ( \lambda ^ { - 1 } )$ curvature charge in the leading error term by the occupation-weighted $B _ { \mathrm { r e f } }$ , which is uniform in λ for the structured examples.

Related proximal-MCMC developments. Subsequent work has developed the proximal-MCMC viewpoint in several algorithmic directions. Moreau smoothing has been combined with stabilized integrators and relaxed proximal-point iterations [23, 13]; inexact-proximal analyses account for iterative or approximate evaluation of the proximal map [8]; and successive-Moreau schemes vary the smoothing scale during sampling rather than fixing a single λ [11]. These works modify the integrator, the accuracy of the proximal computation, or the smoothing schedule. The active-trace theory is complementary: it keeps the classical fixed-λ MYULA kernel and sharpens the geometric quantity that controls its discretization error.

## 2.2 Average smoothness and trace-sensitive Langevin bounds

The closest result in spirit is the average-smoothness theory of [3] for standard LMC on smooth strongly log-concave targets. Let $M _ { \infty }$ denote the global Lipschitz constant of the gradient, and let

$$
M _ { \mathrm { a v } } = { \frac { 1 } { d } } \sum _ { j = 1 } ^ { d } M _ { j }
$$

be the average of coordinatewise smoothness constants. When the potential is twice diferentiable, $M _ { \infty }$ uniformly controls the largest Hessian eigenvalue, whereas $M _ { \mathrm { a v } }$ controls a coordinatewise trace-type quantity. Their main constant-step estimate has the form

$$
W _ { 2 } ^ { 2 } ( \nu _ { k } , \pi ) \leq e ^ { - 2 m k h } W _ { 2 } ^ { 2 } ( \nu _ { 0 } , \pi ) + \frac { ( M _ { \mathrm { a v } } + m ) h d } { 2 m } , \qquad h \leq M _ { \infty } ^ { - 1 } .
$$

Consequently, the leading discretization term depends on an average condition number, although the maximal smoothness remains in the step-size restriction.

The active-trace result shares two features with this analysis. Both replace a maximal-curvature contribution in the leading error by a trace-like quantity, and both can be interpreted through the isotropic Gaussian noise in the Langevin update. However, the two averaging operations are diferent. The constants $M _ { j }$ in average smoothness still take a supremum over all spatial locations. In particular, for a coordinatewise soft-threshold Moreau envelope one still has $M _ { j } = 1 / \lambda$ for every penalized coordinate, regardless of how narrow the threshold region is. By contrast,

$$
B _ { \mathrm { r e f } } = \frac { 1 } { h } \int _ { 0 } ^ { h } \mathbb { E } \operatorname { t r } H _ { \lambda } ( Y _ { t } ^ { \mathrm { r e f } } ) \mathrm { d } t
$$

averages over both directions and the spatial occupation measure of the reference heat path. It can therefore remain $O ( 1 )$ even when sup<sub>x</sub> tr $H _ { \lambda } ( x ) = O ( d / \lambda )$

There is also a regularity distinction. Average-smoothness LMC is formulated for a smooth potential using directional first-order inequalities or classical Hessian quantities. The Moreau envelope in the present work is only $C ^ { 1 , 1 }$ in general. Our trace is consequently defined through the a.e./weak Hessian, and the heat energy identity is proved at this regularity. Thus, one concise positioning statement is that the present framework is a nonsmooth, distribution-weighted, active-set refinement of the average-curvature principle: average smoothness averages across directions but remains worst-case over space, whereas active trace also averages over where the discretized path goes.

Trace, Frobenius, and average-curvature quantities have appeared in other refinements of Langevin discretization theory, especially when stronger second- or third-order regularity is available. Those results support the broader message that operator-norm smoothness can be too coarse for sampling. The specific feature here is the cancellation between a $\lambda ^ { - 1 }$ curvature height and the $O ( \lambda )$ occupation probability of active tubes.

## 2.3 Proximal splitting methods for the original composite target

A separate line of work treats the nonsmooth component through a proximal splitting step rather than by first replacing the target with $\pi _ { \lambda }$ . The Wasserstein gradient-flow and convex-optimization viewpoint of [4] gives a systematic analysis of Langevin algorithms and includes nonsmooth proximal schemes. In the strongly convex composite setting, proximal stochastic gradient Langevin algorithms and their primal–dual interpretation yield nonasymptotic $W _ { 2 }$ guarantees, including $O ( \varepsilon ^ { - 2 } )$ iteration complexity under the assumptions of [25, 26]. These methods can also accommodate extended-valued regularizers, such as indicators of convex constraints, that fall outside the finite-valued Lipschitz assumption used here.

This literature is crucial for delimiting our claim. The present paper does not give the first $O ( \varepsilon ^ { - 2 } )$ guarantee for sampling a nonsmooth strongly log-concave distribution. The distinction is algorithmic and analytic. A typical proximal splitting update applies a proximal map to a noisy or forward-gradient proposal and is designed to approximate or preserve the original composite target. myula instead runs an explicit Euler step for the smooth surrogate $f + g _ { \lambda }$ and incurs a separate smoothing bias. Our question is therefore not whether a diferent proximal Langevin scheme can attain a better generic rate, but whether the widely used myula iteration must pay the global $1 / \lambda$ curvature at every step. The active-trace theorem shows that, for structured penalties, the leading answer is no.

A useful way to present the relationship is by four axes: the transition kernel, the reference target of one step, the oracle, and the error metric. myula and proximal splitting methods may both use one gradient and one proximal evaluation per iteration, but the position of the prox operation and the invariant or approximate target difer. Accordingly, complexity statements should not be ranked solely by their exponent in ε. They should also report whether the guarantee concerns π or $\pi _ { \lambda }$ , whether smoothing bias is present, whether g may be extended-valued, and whether the result is in $W _ { 2 }$ , KL, or total variation.

## 2.4 Active manifolds, proximal Jacobians, and degrees of freedom

The geometric interpretation of $a _ { \lambda }$ is related to the literature on partial smoothness, active manifolds, and identification in nonsmooth optimization. Partly smooth regularizers behave smoothly along an active manifold and sharply in normal directions; proximal and forward–backward algorithms can identify this manifold and subsequently exhibit a lower dimensional local dynamics [17, 18]. For polyhedral and sparsity-promoting penalties, the Jacobian of the proximal map is piecewise constant or admits an explicit tangent-space description.

Closely related formulas also arise in statistical degrees-of-freedom calculations. The divergence or trace of an estimator’s Jacobian measures the efective number of fitted degrees of freedom, and this program has been developed for lasso, generalized lasso, analysis sparsity, group structure, and more general partly smooth regularizers [27, 28]. The identity

$$
\lambda a _ { \lambda } ( x ) = d - \mathrm { t r } D \mathrm { p r o x } _ { \lambda g } ( x )
$$

therefore has a natural interpretation: it is the complement of the local proximal degrees of freedom, or the total number of directions locally contracted by the regularizer. In the generalized-lasso cells of Subsection 7.4, this becomes exactly $\lambda a _ { \lambda } ( x ) = \mathrm { r a n k } ( D _ { A } )$

The proximal-Jacobian and active-manifold identities themselves are not the novelty claimed here. The new step is to insert this local codimension into a Langevin discretization inequality and then average it over a stochastic heat path. The curvature–tube interface additionally quantifies how the normal geometry of an active stratum interacts with its occupation probability. Thus, the paper connects two previously separate uses of proximal geometry: local dimension and sensitivity in optimization/statistics, and nonasymptotic discretization error in sampling.

## 2.5 Alternative envelopes and smoothing-based Langevin methods

Moreau smoothing is only one way to regularize a composite potential. The forward–backward envelope has been used to construct Langevin schemes that retain additional optimization structure, including preservation of the MAP point under suitable conditions [7, 10]. Bregman–Moreau envelopes and Bregman proximal maps extend this idea to non-Euclidean geometries and relativesmoothness settings [15]. These methods alter the surrogate potential and therefore alter the curvature object that enters the discretization analysis. The reference active-trace idea suggests a possible extension: for another envelope, one may seek the trace of its generalized local curvature and average it over the associated one-step interpolation rather than bounding it uniformly.

Other work focuses on numerical implementation rather than the choice of envelope. Stabilized explicit schemes permit larger stability domains for stif smoothed drifts [23], relaxed proximal-point algorithms modify the implicitness or acceleration of the update [13], and inexact-proximal methods quantify the error caused by terminating an inner proximal solver [8]. These concerns are orthogonal to the present analysis, which assumes exact evaluations of $\operatorname { p r o x } _ { \lambda g }$ and asks how the exact Moreau curvature is weighted by the path distribution.

## 2.6 Direct nonsmooth sampling and stronger oracle models

Several methods avoid a fixed Moreau approximation altogether. Subgradient Langevin schemes replace the gradient of the nonsmooth component by a subgradient or a primal–dual construction and analyze the resulting nonsmooth dynamics directly [12]. Other algorithms use proximal proposals inside a Metropolis–Hastings correction, thereby targeting the original distribution exactly at stationarity [20]. These methods answer a diferent question from ours: whether one can sample the original nonsmooth target without accepting a fixed regularization bias.

At the other end of the oracle spectrum, restricted-Gaussian-oracle methods assume the ability to sample from densities proportional to

$$
\exp \Biggl \{ - g ( x ) - \frac { \| x - y \| ^ { 2 } } { 2 \eta } \Biggr \} ,
$$

which is a sampling analogue of a proximal evaluation. Recent proximal-gradient samplers achieve polylogarithmic dependence on the target precision in strongly log-concave composite problems under this stronger oracle [19]. Such results are important benchmarks for composite sampling but are not directly comparable to myula, whose basic iteration uses a deterministic proximal map and one Gaussian increment. Any empirical or theoretical comparison should make the oracle cost explicit.

In summary, the present paper occupies a specific position within this landscape. It does not replace the many algorithmic strategies for nonsmooth sampling. It refines the theory of one established strategy by showing that the leading discretization cost can be governed by an occupation-weighted, active-set measure of Moreau curvature. The closest analytic precedent is average-smoothness $\mathrm { L M C } ;$ the closest algorithmic alternatives are proximal splitting methods; and the closest geometric precedents are active-manifold and degrees-of-freedom analyses of proximal maps. The active-trace framework combines these three viewpoints in a form adapted to the stochastic heat step of myula.

## 3 Preliminaries

## 3.1 Basic notation

All probability measures are defined on $\mathbb { R } ^ { d }$ equipped with its Borel σ-algebra. The Euclidean norm and inner product are denoted by $\lVert \cdot \rVert$ and $\langle \cdot , \cdot \rangle$ . For $\mu , \nu \in \mathcal { P } _ { 2 } ( \mathbb { R } ^ { d } )$ , the quadratic Wasserstein distance is

$$
W _ { 2 } ^ { 2 } ( \mu , \nu ) : = \operatorname* { i n f } _ { ( X , Y ) : \mathcal { L } ( X ) = \mu , \mathcal { L } ( Y ) = \nu } \mathbb { E } \left\| X - Y \right\| ^ { 2 } .\tag{3.1}
$$

If $\mu \ll \nu ,$ the relative entropy is

$$
\mathrm { K L } ( \mu \| \nu ) : = \int \log \left( \frac { \mathrm { d } \mu } { \mathrm { d } \nu } \right) \mathrm { d } \mu ,\tag{3.2}
$$

and otherwise $\mathrm { K L } ( \mu \| \nu ) : = + \infty$ . If $\mu ( \mathrm { d } x ) = \rho ( x )$ dx, its Lebesgue entropy is

$$
\operatorname { E n t } ( \mu ) : = \int _ { \mathbb { R } ^ { d } } \rho ( x ) \log \rho ( x ) \mathrm { d } x ,\tag{3.3}
$$

with the convention Ent ${ \mathrm { \Omega } } ( \mu ) = + \infty$ if the expression is not well-defined.

The total variation distance is

$$
\| \mu - \nu \| _ { \mathrm { T V } } : = \operatorname* { s u p } _ { A } | \mu ( A ) - \nu ( A ) | ,\tag{3.4}
$$

where the supremum is over Borel sets. $\operatorname { I f } \mu$ and $\nu$ have densities $p$ and $q ,$ then

$$
\| \mu - \nu \| _ { \mathrm { T V } } = \frac { 1 } { 2 } \int _ { \mathbb { R } ^ { d } } \left| p ( x ) - q ( x ) \right| \mathrm { d } x .\tag{3.5}
$$

For a measurable map $T : \mathbb { R } ^ { d }  \mathbb { R } ^ { d }$ and a probability measure $\mu ,$ the pushforward $T _ { \# } \mu$ is the law of $T ( X )$ when $X \sim \mu$

Let $( P _ { t } ) _ { t \geq 0 }$ be the heat semigroup with generator $\Delta$

$$
P _ { t } \varphi ( x ) = \mathbb { E } \varphi ( x + { \sqrt { 2 t } } Z ) , \qquad Z \sim N ( 0 , I _ { d } ) .\tag{3.6}
$$

Thus, if $\nu _ { t } = \nu _ { 0 } P _ { t }$ , then $\nu _ { t }$ is the law of $Y _ { t } = Y _ { 0 } + \sqrt { 2 t } Z$ for $Y _ { 0 } \sim \nu _ { 0 }$ independent of $Z$

Throughout the paper, $C , c \in ( 0 , \infty )$ denote universal constants whose value may change from line to line. They never depend on $d , \lambda , h ,$ k or on the particular measures under consideration, unless explicitly stated otherwise.

For symmetric matrices $A , B \in \mathbb { R } ^ { d \times d } ,$ , we write $A \preceq B$ if $v ^ { \top } A v \leq v ^ { \top } B v$ for every $v \in \mathbb { R } ^ { d }$ . This is the Loewner order, and we write $\left\| A \right\| _ { \mathrm { o p } } : = \operatorname* { s u p } _ { \| v \| = 1 } \left\| A v \right\|$

## 3.2 Nonsmooth calculus for Lipschitz and convex functions

For ordinary notation, if $T : \mathbb { R } ^ { d _ { 1 } }  \mathbb { R } ^ { d _ { 2 } }$ is diferentiable at x, then $D T ( x )$ denotes its Jacobian matrix:

$$
D T ( x ) _ { i j } = \frac { \partial T _ { i } } { \partial x _ { j } } ( x ) .\tag{3.7}
$$

We use the global convention

$$
C ^ { 1 , 1 } ( \mathbb { R } ^ { d } ) : = \{ V \in C ^ { 1 } ( \mathbb { R } ^ { d } ) : \nabla V { \mathrm { ~ i s ~ g l o b a l l y ~ L i p s c h i t z } } \} .
$$

For $V \in C ^ { 1 , 1 } ( \mathbb { R } ^ { d } )$ , a standard theorem of Rademacher (reviewed in Section A.1) says that $\nabla V$ is diferentiable a.e.; at those points we write

$$
\nabla ^ { 2 } V ( x ) : = D ( \nabla V ) ( x ) .
$$

This is the $\mathrm { { a . e . } }$ Hessian. The same object can also be described as the $L ^ { \infty }$ weak Hessian. Here “weak” means defined through integration by parts rather than by pointwise second derivatives, and $L ^ { \infty }$ means bounded outside a null set. A formal explanation is given in Section A.2. In the main text, every Hessian of $g _ { \lambda }$ is understood in this a.e./weak sense, never as an everywhere classical $C ^ { 2 }$ Hessian.

If $g$ is convex, its subdiferential at x is

$$
\partial g ( x ) : = \{ s \in \mathbb { R } ^ { d } : \ g ( y ) \geq g ( x ) + \langle s , y - x \rangle \ { \mathrm { ~ f o r ~ a l l ~ } } y \in \mathbb { R } ^ { d } \} .\tag{3.8}
$$

The subdiferential is monotone: if $s \in \partial g ( x )$ and $t \in \partial g ( y )$ , then

$$
\langle s - t , x - y \rangle \geq 0 .\tag{3.9}
$$

## 3.3 Moreau–Yosida regularization and weak second-order structure

The Moreau envelope is the smoothing device used throughout the paper. Let $g : \mathbb { R } ^ { d }  ( - \infty , + \infty ]$ be proper, lower semicontinuous, and convex. For $\lambda > 0$ , define the Moreau envelope and proximal map by

$$
g _ { \lambda } ( x ) : = \operatorname* { i n f } _ { y \in \mathbb { R } ^ { d } } \left\{ g ( y ) + \frac { \| x - y \| ^ { 2 } } { 2 \lambda } \right\} , \qquad p _ { \lambda } ( x ) : = \operatorname { p r o x } _ { \lambda g } ( x ) : = \operatorname* { a r g m i n } _ { y \in \mathbb { R } ^ { d } } \left\{ g ( y ) + \frac { \| x - y \| ^ { 2 } } { 2 \lambda } \right\}\tag{3.10}
$$

The following facts are standard in convex analysis and monotone operator theory; see, for example, [1, Chs. 12 and 23] and [24, Ch. 1.G]. Detailed justification is given in Section B.

Lemma 3.1 (Moreau regularity and a.e. Hessian). Let $g : \mathbb { R } ^ { d }  ( - \infty , + \infty ]$ be proper, lower semicontinuous, and convex. For every $\lambda > 0$ , the proximal map $p _ { \lambda }$ is single-valued, and the Moreau envelope $g _ { \lambda }$ is convex and belongs to $C ^ { 1 , 1 } ( \mathbb { R } ^ { d } )$ . Moreover,

$$
\nabla g _ { \lambda } ( x ) = { \frac { x - p _ { \lambda } ( x ) } { \lambda } } , \qquad \mathrm { L i p } ( \nabla g _ { \lambda } ) \leq \lambda ^ { - 1 } .\tag{3.11}
$$

There exists a set $E _ { \lambda } \subset \mathbb { R } ^ { d }$ , whose complement is Lebesgue-null, such that the following statements hold for every $x \in E _ { \lambda }$ . The map $\nabla g _ { \lambda }$ is diferentiable at x; writing

$$
\nabla ^ { 2 } g _ { \lambda } ( x ) : = D ( \nabla g _ { \lambda } ) ( x ) ,
$$

the matrix $\nabla ^ { 2 } g _ { \lambda } ( x )$ is symmetric and satisfies

$$
0 \preceq \nabla ^ { 2 } g _ { \lambda } ( x ) \preceq \lambda ^ { - 1 } I _ { d } .
$$

Furthermore, $p _ { \lambda }$ is diferentiable at $x ,$ and

$$
{ \cal D } p _ { \lambda } ( x ) = I _ { d } - \lambda \nabla ^ { 2 } g _ { \lambda } ( x ) , \qquad \nabla ^ { 2 } g _ { \lambda } ( x ) = \lambda ^ { - 1 } \{ I _ { d } - { \cal D } p _ { \lambda } ( x ) \} .
$$

We next fix the curvature quantity that will enter all heat-flow estimates. By Lemma 3.1, we choose a measurable representative $H _ { \lambda }$ of the a.e./weak Hessian of $g _ { \lambda }$ such that $H _ { \lambda } ( x ) = \nabla ^ { 2 } g _ { \lambda } ( x )$ for a.e. x.

Definition 3.2 (Active trace). The Moreau active trace is

$$
a _ { \lambda } ( x ) : = \mathrm { t r } H _ { \lambda } ( x ) .
$$

Thus

$$
0 \leq a _ { \lambda } ( x ) \leq d / \lambda \quad d x \mathrm { - a . e . }
$$

By Lemma 3.1,

$$
a _ { \lambda } ( x ) = \lambda ^ { - 1 } \mathrm { t r } \{ I _ { d } - D p _ { \lambda } ( x ) \} \quad d x \mathrm { - a . e . }
$$

Hence $a _ { \lambda }$ is the trace of the local Moreau curvature. Equivalently, up to the factor $\lambda ^ { - 1 }$ , it measures the total local shrinkage of the proximal map.

Lemma 3.3 (Moreau envelope preserves Lipschitz constants). Assume that $g : \mathbb { R } ^ { d }  \mathbb { R }$ is convex and G-Lipschitz:

$$
| g ( x ) - g ( y ) | \leq G \left. x - y \right. , \qquad x , y \in \mathbb { R } ^ { d } .\tag{3.12}
$$

Then, for every $\lambda > 0$ and every $\boldsymbol { x } \in \mathbb { R } ^ { d }$

$$
\| \nabla g _ { \lambda } ( x ) \| \leq G .\tag{3.13}
$$

## 3.4 The heat energy identity for $C ^ { 1 , 1 }$ test functions

We record a basic identity for Gaussian smoothing. Let $Y _ { t } = Y _ { 0 } + { \sqrt { 2 } } W _ { t }$ . For a smooth test function V , Itˆo’s formula gives

$$
\mathbb { E } V ( Y _ { h } ) - \mathbb { E } V ( Y _ { 0 } ) = \int _ { 0 } ^ { h } \mathbb { E } \Delta V ( Y _ { t } ) d t .
$$

Thus the expected change of V along the heat flow is controlled by its Laplacian.

We will need this identity for functions that are only $C ^ { 1 , 1 }$ , since Moreau envelopes have Lipschitz gradients but need not be twice continuously diferentiable. In this case the Hessian is interpreted in the a.e./weak sense. The following lemma gives the precise statement.

Lemma 3.4 (Heat energy identity for $C ^ { 1 , 1 }$ test functions). Let $V \in C ^ { 1 , 1 } (  { \mathbb { R } } ^ { d } )$ . Choose any Lebesgue measurable representative $H _ { V }$ of its a.e./weak Hessian and put

$$
\Delta V : = \operatorname { t r } H _ { V } .
$$

Let $W _ { t }$ be a standard Brownian motion in $\mathbb { R } ^ { d }$ , independent of $Y _ { 0 } \in L ^ { 2 }$ , and set

$$
Y _ { t } = Y _ { 0 } + { \sqrt { 2 } } W _ { t } .
$$

Then, for every $h > 0$

$$
\mathbb { E } V ( Y _ { h } ) - \mathbb { E } V ( Y _ { 0 } ) = \int _ { 0 } ^ { h } \mathbb { E } \Delta V ( Y _ { t } ) d t .
$$

The proof is postponed to Section C. It is based on the Itˆo–Krylov formula for functions with generalized second derivatives, together with a localization argument.

## 3.5 Entropy EVI for the heat flow

The next lemma is a standard finite-time inequality for the heat semigroup. It will be used to control the entropy change during the Gaussian-noise part of one MYULA step.

Lemma 3.5 (Entropy EVI for the heat semigroup). Let $\nu _ { 0 } \in \mathcal { P } _ { 2 } ( \mathbb { R } ^ { d } )$ , and set

$$
\nu _ { t } = \nu _ { 0 } P _ { t } , \qquad t \geq 0 ,
$$

where $P _ { t }$ is the heat semigroup defined in (3.6). Then, for every $\sigma \in \mathcal { P } _ { 2 } ( \mathbb { R } ^ { d } )$ with $\operatorname { E n t } ( \sigma ) < \infty$ , and every $h > 0$ ，

$$
2 h \{ \mathrm { E n t } ( \nu _ { h } ) - \mathrm { E n t } ( \sigma ) \} \leq W _ { 2 } ^ { 2 } ( \nu _ { 0 } , \sigma ) - W _ { 2 } ^ { 2 } ( \nu _ { h } , \sigma ) .
$$

Reference. This is Lemma 5 of [4].

## 4 Problem Setup and Standing Assumptions

We consider the nonsmooth composite target

$$
\pi ( \mathrm { d } \boldsymbol { x } ) = Z ^ { - 1 } \exp \{ - f ( \boldsymbol { x } ) - g ( \boldsymbol { x } ) \} \mathrm { d } \boldsymbol { x } ,\tag{4.1}
$$

where $f$ is smooth and strongly convex and $g$ is convex but possibly nonsmooth. For a fixed Moreau parameter $\lambda > 0$ , define

$$
U _ { \lambda } ( x ) : = f ( x ) + g _ { \lambda } ( x ) , \qquad \pi _ { \lambda } ( \mathrm { d } x ) : = Z _ { \lambda } ^ { - 1 } \exp \{ - U _ { \lambda } ( x ) \} \mathrm { d } x .\tag{4.2}
$$

Assumption 4.1 (Smooth strongly convex part). The function $f \in C ^ { 2 } (  { \mathbb { R } } ^ { d } )$ satisfies, for some $0 < m \leq L _ { f } < \infty _ { ; }$

$$
m I _ { d } \preceq \nabla ^ { 2 } f ( x ) \preceq L _ { f } I _ { d } , \qquad x \in \mathbb { R } ^ { d } .\tag{4.3}
$$

We also set

$$
\tau _ { f } : = \operatorname* { s u p } _ { x \in \mathbb { R } ^ { d } } \operatorname { t r } \nabla ^ { 2 } f ( x ) \leq d L _ { f } .\tag{4.4}
$$

Assumption 4.2 (Convex Lipschitz nonsmooth part). The function $g : \mathbb { R } ^ { d }  \mathbb { R }$ is closed, convex, and G-Lipschitz:

$$
| g ( x ) - g ( y ) | \leq G \left. x - y \right. , \qquad x , y \in \mathbb { R } ^ { d } .\tag{4.5}
$$

Under Assumptions 4.1 and 4.2, the function f is $C ^ { 2 }$ with bounded Hessian, while $g _ { \lambda }$ is $C ^ { 1 , 1 }$ by Lemma 3.1. Hence

$$
U _ { \lambda } = f + g _ { \lambda }\tag{4.6}
$$

belongs to $C ^ { 1 , 1 }$ and has an a.e./weak Hessian in the sense of Section A.2:

$$
H _ { U _ { \lambda } } ( x ) = \nabla ^ { 2 } f ( x ) + H _ { \lambda } ( x )\tag{4.7}
$$

for a.e. $x .$ Since $f$ is m-strongly convex and $g _ { \lambda }$ is convex, $U _ { \lambda }$ is m-strongly convex. Therefore $\pi _ { \lambda }$ is well-defined and has finite second moment.

The MYULA transition with step size $h > 0$ is the explicit Euler transition for $\pi _ { \lambda }$

$$
X _ { k + 1 } = X _ { k } - h \nabla U _ { \lambda } ( X _ { k } ) + \sqrt { 2 h } \xi _ { k + 1 } , \xi _ { k + 1 } \sim N ( 0 , I _ { d } ) .\tag{4.8}
$$

We analyze the error to the fixed smoothed target $\pi _ { \lambda }$ first. The regularization bias $W _ { 2 } ( \pi _ { \lambda } , \pi )$ is separated from the fixed-λ discretization analysis.

Let $\mu _ { k } : = \mathcal { L } ( X _ { k } )$ and define

$$
D _ { k } : = W _ { 2 } ^ { 2 } ( \mu _ { k } , \pi _ { \lambda } ) , \qquad K _ { k } : = \mathrm { K L } ( \mu _ { k } \| \pi _ { \lambda } ) .\tag{4.9}
$$

Throughout the fixed-λ analysis we assume $\mu _ { 0 } \in \mathcal P _ { 2 } ( \mathbb { R } ^ { d } )$ and $K _ { 0 } < \infty$

It is useful to split one MYULA step into a deterministic gradient step followed by a heat step. Define

$$
T _ { h } ( x ) : = x - h \nabla U _ { \lambda } ( x ) , \qquad { \bar { \mu } } _ { k } : = ( T _ { h } ) _ { \# } \mu _ { k } .\tag{4.10}
$$

Then $\mu _ { k + 1 } = \bar { \mu } _ { k } P _ { h }$ . Let $( W _ { t } ^ { ( k ) } ) _ { 0 \leq t \leq h }$ be a standard Brownian motion in $\mathbb { R } ^ { d }$ , independent of $X _ { k }$ and define the heat interpolation

$$
Y _ { k , t } : = T _ { h } ( X _ { k } ) + \sqrt { 2 } W _ { t } ^ { ( k ) } , \qquad 0 \le t \le h .
$$

Definition 4.3 (Stepwise heat-path active trace). Let $a _ { \lambda } = \operatorname { t r } H _ { \lambda }$ be the active-trace representative fixed in Definition 3.2. For the heat interpolation $( Y _ { k , t } ) _ { 0 \leq t \leq h }$ , define

$$
B _ { k } : = \frac { 1 } { h } \int _ { 0 } ^ { h } \mathbb { E } a _ { \lambda } ( Y _ { k , t } ) \mathrm { d } t .\tag{4.11}
$$

The time integral may equivalently be read as an integral over $( 0 , h ]$ , since changing the integrand at the single time $t = 0$ has no efect. For every $t > 0$ , the law of $Y _ { k , t }$ has a density, so $B _ { k }$ is independent of the values assigned to $H _ { \lambda }$ on null sets. The trivial global bound is $B _ { k } \leq d / \lambda$

## 5 Main Result: Reference Active Trace and Total Error

To state the complete error guarantee, we first introduce a reference version of the heat-path active trace. Unlike $B _ { k }$ , which is computed from the current law $\mu _ { k }$ , this quantity is computed from the reference target law $\pi _ { \lambda }$

Definition 5.1 (Reference heat-path active trace). Let $X ^ { \mathrm { r e f } } \sim \pi _ { \lambda }$ , and let $( W _ { t } ^ { \mathrm { r e f } } ) _ { 0 \leq t \leq h }$ be a standard Brownian motion in $\mathbb { R } ^ { d }$ , independent of $X ^ { \mathrm { r e f } }$ . Define

$$
Y _ { t } ^ { \mathrm { r e f } } : = T _ { h } ( X ^ { \mathrm { r e f } } ) + \sqrt { 2 } W _ { t } ^ { \mathrm { r e f } } , \qquad 0 \le t \le h .
$$

The reference active trace is

$$
B _ { \mathrm { r e f } } : = \frac 1 h \int _ { 0 } ^ { h } \mathbb { E } a _ { \lambda } ( Y _ { t } ^ { \mathrm { r e f } } ) d t .
$$

Equivalently, $B _ { \mathrm { r e f } }$ is the same heat-path average as $B _ { k }$ , with the initialization $X _ { k } \sim \mu _ { k }$ replaced by $X ^ { \mathrm { r e f } } \sim \pi _ { \lambda }$ . This quantity depends on λ and h, but this dependence is suppressed in the notation. As with $B _ { k }$ , the integral may be read over $( 0 , h ]$ , so the value is independent of the chosen null-set version of $a _ { \lambda }$

The main theorem below expresses the fixed-λ discretization error in terms of $B _ { \mathrm { r e f } }$ , and then adds the Moreau approximation bias to control the error to the original target π.

Theorem 5.2 (Main active-trace guarantee). Assume Assumptions $\it 4 . 1$ and 4.2. Fix $\lambda > 0$ and $h > 0$ . Let $M _ { \lambda }$ be any finite constant such that $0 \le a _ { \lambda } \le M _ { \lambda }$ Lebesgue-a.e. The universal choice $M _ { \lambda } = d / \lambda$ is always admissible. There exist universal constants $c , C > 0$ such that, whenever $0 < h \leq c / L _ { f }$ , the MYULA iterates satisfy

$$
\Phi _ { N } \leq e ^ { - m h N / 2 } \Phi _ { 0 } + \frac { C } { m } \Big \{ ( \tau _ { f } + G ^ { 2 } + B _ { \mathrm { r e f } } ) h + M _ { \lambda } ^ { 2 } h ^ { 2 } \Big \} , \qquad \Phi _ { k } : = D _ { k } + 2 h K _ { k } .\tag{5.1}
$$

Consequently, for any algorithmic tolerance $\varepsilon _ { \mathrm { a l g } } \in ( 0 , 1 )$ , if

$$
h \leq c \operatorname * { m i n } \left\{ L _ { f } ^ { - 1 } , \frac { \varepsilon _ { \mathrm { a l g } } ^ { 2 } } { \tau _ { f } + G ^ { 2 } + B _ { \mathrm { r e f } } } , \frac { \varepsilon _ { \mathrm { a l g } } } { M _ { \lambda } } \right\}\tag{5.2}
$$

and

$$
N \geq \frac { C } { m h } \log \left( 1 + \frac { m \Phi _ { 0 } } { \varepsilon _ { \mathrm { a l g } } ^ { 2 } } \right) ,\tag{5.3}
$$

then

$$
\sqrt { m } W _ { 2 } ( \mu _ { N } , \pi _ { \lambda } ) \leq \varepsilon _ { \mathrm { a l g } } .\tag{5.4}
$$

Moreover, the original nonsmooth target π satisfies the Moreau bias bound

$$
\sqrt { m } W _ { 2 } ( \pi _ { \lambda } , \pi ) \leq \frac { G ^ { 2 } \lambda } { 4 } .\tag{5.5}
$$

Therefore, $i f \varepsilon _ { \mathrm { b i a s } } > 0$ and

$$
\lambda \leq \frac { 4 \varepsilon _ { \mathrm { b i a s } } } { G ^ { 2 } } ,\tag{5.6}
$$

then the total error obeys

$$
\sqrt { m } W _ { 2 } ( \mu _ { N } , \pi ) \leq \varepsilon _ { \mathrm { a l g } } + \varepsilon _ { \mathrm { b i a s } } .\tag{5.7}
$$

In the common symmetric choice $\varepsilon _ { \mathrm { a l g } } = \varepsilon _ { \mathrm { b i a s } } = \varepsilon / 2$ , it is enough to take $\lambda \le 2 \varepsilon / G ^ { 2 }$ , and the suficient iteration complexity is, up to logarithmic factors,

$$
N ( \varepsilon ) \lesssim \frac { 1 } { m } \left( L _ { f } + \frac { \tau _ { f } + G ^ { 2 } + B _ { \mathrm { r e f } } } { \varepsilon ^ { 2 } } + \frac { M _ { \lambda } } { \varepsilon } \right) .\tag{5.8}
$$

Remark 5.3. The term $B _ { \mathrm { r e f } }$ is the main active-geometry quantity. If only the universal bound $a _ { \lambda } \leq d / \lambda$ is available, then $B _ { \mathrm { r e f } } \leq d / \lambda$ and the theorem reduces to a conservative worst-case result. The purpose of Section 6 is to prove that $B _ { \mathrm { r e f } }$ is much smaller than $d / \lambda$ in structured examples.

The proof of Theorem 5.2 is given in Section 5.4 after all three ingredients have been established.

## 5.1 Universal Active-Trace KL–EVI Recursion

We first derive a one-step recursion for the fixed-λ target $\pi _ { \lambda }$ . The argument separates one MYULA update into a deterministic gradient step and a heat step. The gradient step is controlled by an approximate EVI estimate whose error depends on the Lipschitz size $G ^ { 2 }$ of $g ,$ , rather than on the Moreau smoothness scale $\lambda ^ { - 2 }$ . Combining this estimate with the entropy EVI for the heat step yields a recursion involving the stepwise active trace $B _ { k }$

Lemma 5.4 (Gradient-step approximate EVI without $1 / \lambda )$ . Assume Assumptions 4.1 and $4 . 2 .$ There exist universal constants $c _ { 0 } , C _ { 0 } > 0$ such that, if $0 < h \leq c _ { 0 } / L _ { f }$ , then for all $x , y \in \mathbb { R } ^ { d }$ , with $x ^ { + } = T _ { h } ( x )$

$$
2 h \{ U _ { \lambda } ( x ^ { + } ) - U _ { \lambda } ( y ) \} \le \| x - y \| ^ { 2 } - \left\| x ^ { + } - y \right\| ^ { 2 } - m h \left\| x - y \right\| ^ { 2 } + C _ { 0 } G ^ { 2 } h ^ { 2 } .\tag{5.9}
$$

Proof. Write

$$
\begin{array} { r } { s : = \nabla f ( x ) , \qquad r : = \nabla g _ { \lambda } ( x ) , \qquad v : = s + r = \nabla U _ { \lambda } ( x ) , \qquad a : = x - y . } \end{array}\tag{5.10}
$$

Then $x ^ { + } = x - h v$ . By the strong convexity and smoothness of $f$

$$
f ( x ^ { + } ) - f ( y ) = f ( x ) - f ( y ) + f ( x ^ { + } ) - f ( x ) \leq \langle s , a \rangle - \frac { m } { 2 } \left\| a \right\| ^ { 2 } - h \langle s , v \rangle + \frac { L _ { f } h ^ { 2 } } { 2 } \left\| v \right\| ^ { 2 } .\tag{5.11}
$$

By convexity of $g _ { \lambda }$ and by Lemma 3.3, $g _ { \lambda }$ is G-Lipschitz; hence

$$
g _ { \lambda } ( x ^ { + } ) - g _ { \lambda } ( y ) = g _ { \lambda } ( x ) - g _ { \lambda } ( y ) + g _ { \lambda } ( x ^ { + } ) - g _ { \lambda } ( x ) \leq \langle r , a \rangle + G h \left. v \right. .\tag{5.12}
$$

Adding (5.11) and (5.12), multiplying by $2 h ,$ and using

$$
2 h \left. v , a \right. = \left\| a \right\| ^ { 2 } - \left\| a - h v \right\| ^ { 2 } + h ^ { 2 } \left\| v \right\| ^ { 2 } ,\tag{5.13}
$$

we obtain

$$
\begin{array} { r l } & { 2 h \{ U _ { \lambda } ( x ^ { + } ) - U _ { \lambda } ( y ) \} \le \| a \| ^ { 2 } - \| a - h v \| ^ { 2 } - m h \left\| a \right\| ^ { 2 } } \\ & { \qquad + h ^ { 2 } \left\| v \right\| ^ { 2 } - 2 h ^ { 2 } \left. s , v \right. + L _ { f } h ^ { 3 } \left\| v \right\| ^ { 2 } + 2 G h ^ { 2 } \left\| v \right\| . } \end{array}\tag{5.14}
$$

The key cancellation is

$$
\left\| v \right\| ^ { 2 } - 2 \left. s , v \right. = \left\| s + r \right\| ^ { 2 } - 2 \left. s , s + r \right. = \left\| r \right\| ^ { 2 } - \left\| s \right\| ^ { 2 } \leq G ^ { 2 } - \left\| s \right\| ^ { 2 } .\tag{5.15}
$$

Moreover, $\| v \| \leq \| s \| + G$ . If $h \leq c _ { 0 } / L _ { f }$ with $c _ { 0 } > 0$ suficiently small, then

$$
L _ { f } h ^ { 3 } \left\| v \right\| ^ { 2 } + 2 G h ^ { 2 } \left\| v \right\| \leq { \frac { 1 } { 2 } } h ^ { 2 } \left\| s \right\| ^ { 2 } + C G ^ { 2 } h ^ { 2 } .\tag{5.16}
$$

Substituting this bound into (5.14) absorbs the negative $- h ^ { 2 } \left\| s \right\| ^ { 2 }$ term and gives (5.9).

Theorem 5.5 (Universal active-trace KL–EVI recursion). Assume Assumptions 4.1 and 4.2. There exist universal constants $c , C > 0$ such that, for every $\lambda > 0$ and every step size $0 < h \leq c / L _ { f }$ , the MYULA iterates satisfy

$$
D _ { k + 1 } + 2 h K _ { k + 1 } \leq ( 1 - m h ) D _ { k } + C \big ( \tau _ { f } + G ^ { 2 } + B _ { k } \big ) h ^ { 2 } .\tag{5.17}
$$

Proof. Let $( X , Y )$ be an optimal W<sub>2</sub>-coupling of $\mu _ { k }$ and $\pi _ { \lambda }$ . Apply Lemma 5.4 pointwise to $( X , Y )$ and integrate. Since $( T _ { h } ( X ) , Y )$ is a coupling of $\bar { \mu } _ { k }$ and $\pi _ { \lambda }$ , we have

$$
2 h \left\{ \int U _ { \lambda } \mathrm { d } \bar { \mu } _ { k } - \int U _ { \lambda } \mathrm { d } \pi _ { \lambda } \right\} \leq D _ { k } - W _ { 2 } ^ { 2 } ( \bar { \mu } _ { k } , \pi _ { \lambda } ) - m h D _ { k } + C G ^ { 2 } h ^ { 2 } .\tag{5.18}
$$

Next, apply Lemma 3.5 with $\nu _ { 0 } = \bar { \mu } _ { k } , \nu _ { h } = \mu _ { k + 1 }$ , and $\sigma = \pi _ { \lambda } \mathrm { : }$

$$
2 h \{ \mathrm { E n t } ( \mu _ { k + 1 } ) - \mathrm { E n t } ( \pi _ { \lambda } ) \} \leq W _ { 2 } ^ { 2 } ( { \bar { \mu } } _ { k } , \pi _ { \lambda } ) - D _ { k + 1 } .\tag{5.19}
$$

By Lemma 3.4 applied to the $C ^ { 1 , 1 }$ function $U _ { \lambda } = f + g _ { \lambda }$ , with weak/a.e. Hessian

$$
H _ { U _ { \lambda } } = \nabla ^ { 2 } f + H _ { \lambda } ,\tag{5.20}
$$

we have

$$
\int U _ { \lambda } \mathrm { d } \mu _ { k + 1 } - \int U _ { \lambda } \mathrm { d } \bar { \mu } _ { k } = \int _ { 0 } ^ { h } \mathbb { E } \Delta U _ { \lambda } ( Y _ { k , t } ) \mathrm { d } t .\tag{5.21}
$$

Here

$$
\Delta U _ { \lambda } = \mathrm { t r } \nabla ^ { 2 } f + \mathrm { t r } H _ { \lambda }\tag{5.22}
$$

in the wea $\tau / \alpha . \mathrm { e }$ . sense explained in Section A.2. Since tr $\nabla ^ { 2 } f \leq \tau _ { f }$ pointwise and $a _ { \lambda } = \operatorname { t r } H _ { \lambda }$ , the right-hand side of (5.21) is bounded by

$$
h \tau _ { f } + h B _ { k } .\tag{5.23}
$$

Because

$$
K _ { k + 1 } = \mathrm { E n t } ( \mu _ { k + 1 } ) - \mathrm { E n t } ( \pi _ { \lambda } ) + \int U _ { \lambda } \mathrm { d } \mu _ { k + 1 } - \int U _ { \lambda } \mathrm { d } \pi _ { \lambda } ,\tag{5.24}
$$

summing (5.18), (5.19), and 2h times (5.21) gives

$$
2 h K _ { k + 1 } \leq D _ { k } - D _ { k + 1 } - m h D _ { k } + C G ^ { 2 } h ^ { 2 } + 2 ( \tau _ { f } + B _ { k } ) h ^ { 2 } .\tag{5.25}
$$

Rearranging yields (5.17).

## 5.2 Closing the Recursion by Active-Trace Transfer

The previous theorem reduces the fixed-λ analysis to a bound on $B _ { k }$ . Now we bound $B _ { k }$ by comparing the current heat path to the reference heat path introduced in Section 5. For $0 \leq t \leq h$ let $Q _ { t }$ be the Markov kernel obtained by applying $T _ { h }$ and then running the heat semigroup for time t:

$$
Q _ { t } ( x , \cdot ) : = ( \delta _ { T _ { h } ( x ) } P _ { t } ) ( \cdot ) .
$$

Equivalently, $Q _ { t } ( x , \cdot )$ is the law of $T _ { h } ( x ) + \sqrt { 2 t } Z$ , where $Z \sim N ( 0 , I _ { d } )$ . Hence

$$
\begin{array} { r } { \mathcal { L } ( Y _ { k , t } ) = \mu _ { k } Q _ { t } , \qquad \mathcal { L } ( Y _ { t } ^ { \mathrm { r e f } } ) = \pi _ { \lambda } Q _ { t } . } \end{array}
$$

Therefore,

$$
B _ { k } = \frac { 1 } { h } \int _ { 0 } ^ { h } \int a _ { \lambda } \mathrm { d } ( \mu _ { k } Q _ { t } ) d t , \qquad B _ { \mathrm { r e f } } = \frac { 1 } { h } \int _ { 0 } ^ { h } \int a _ { \lambda } \mathrm { d } ( \pi _ { \lambda } Q _ { t } ) d t .
$$

Proposition 5.6 (Bounding $B _ { k }$ by the reference active trace). Assume that $a _ { \lambda } = \operatorname { t r } H _ { \lambda }$ satisfies $0 \le a _ { \lambda } \le M _ { \lambda }$ a.e. Then, for every $k _ { i }$

$$
B _ { k } \le B _ { \mathrm { r e f } } + M _ { \lambda } \sqrt { K _ { k } / 2 } .\tag{5.26}
$$

Proof. For every $t > 0$ , the measures $\mu _ { k } Q _ { t }$ and $\pi _ { \lambda } Q _ { t }$ are absolutely continuous with respect to Lebesgue measure, since $Q _ { t }$ adds a nondegenerate Gaussian noise. Hence the a.e. bound $0 \le a _ { \lambda } \le M _ { \lambda }$ may be used under both measures. Therefore,

$$
\int a _ { \lambda } \mathrm { d } ( \mu _ { k } Q _ { t } ) \leq \int a _ { \lambda } \mathrm { d } ( \pi _ { \lambda } Q _ { t } ) + M _ { \lambda } \Vert \mu _ { k } Q _ { t } - \pi _ { \lambda } Q _ { t } \Vert _ { \mathrm { T V } }\tag{5.27}
$$

$$
\leq \int a _ { \lambda } \mathrm { d } ( \pi _ { \lambda } Q _ { t } ) + M _ { \lambda } \| \mu _ { k } - \pi _ { \lambda } \| _ { \mathrm { T V } } .\tag{5.28}
$$

The second inequality is contraction of total variation under Markov kernels. Pinsker’s inequality gives

$$
\| \mu _ { k } - \pi _ { \lambda } \| _ { \mathrm { T V } } \leq \sqrt { \mathrm { K L } ( \mu _ { k } \| \pi _ { \lambda } ) / 2 } = \sqrt { K _ { k } / 2 } .\tag{5.29}
$$

Integrating the preceding display over $t \in ( 0 , h ]$ and dividing by h gives the same value as integrating over [0, h]. Hence (5.26) follows. □

Substituting this transfer estimate into the universal recursion gives a closed Lyapunov recursion in terms of $B _ { \mathrm { r e f } }$ only.

Theorem 5.7. Assume Assumption 4.1, Assumption 4.2 and the assumption of Proposition 5.6. Let

$$
\Phi _ { k } : = D _ { k } + 2 h K _ { k } .\tag{5.30}
$$

There exist universal constants $c , C > 0$ such that, whenever $0 < h \leq c / L _ { f }$

$$
\Phi _ { k + 1 } \leq \left( 1 - \frac { m h } { 2 } \right) \Phi _ { k } + C \big \{ ( \tau _ { f } + G ^ { 2 } + B _ { \mathrm { r e f } } ) h ^ { 2 } + M _ { \lambda } ^ { 2 } h ^ { 3 } \big \} .\tag{5.31}
$$

Consequently,

$$
\Phi _ { N } \leq e ^ { - m h N / 2 } \Phi _ { 0 } + \frac { C } { m } \big \{ ( \tau _ { f } + G ^ { 2 } + B _ { \mathrm { r e f } } ) h + M _ { \lambda } ^ { 2 } h ^ { 2 } \big \} .\tag{5.32}
$$

In particular, given $\varepsilon _ { \mathrm { a l g } } \in ( 0 , 1 )$ , if

$$
h \leq c \operatorname* { m i n } \left\{ L _ { f } ^ { - 1 } , \frac { \varepsilon _ { \mathrm { a l g } } ^ { 2 } } { \tau _ { f } + G ^ { 2 } + B _ { \mathrm { r e f } } } , \frac { \varepsilon _ { \mathrm { a l g } } } { M _ { \lambda } } \right\} ,\tag{5.33}
$$

and

$$
N \geq \frac { C } { m h } \log \left( 1 + \frac { m \Phi _ { 0 } } { \varepsilon _ { \mathrm { a l g } } ^ { 2 } } \right) ,\tag{5.34}
$$

then

$$
\sqrt { m } W _ { 2 } ( \mu _ { N } , \pi _ { \lambda } ) \leq \varepsilon _ { \mathrm { a l g } } .\tag{5.35}
$$

Equivalently, up to logarithmic factors, the fixed-λ iteration complexity is

$$
N _ { \lambda } ( \varepsilon _ { \mathrm { a l g } } ) \lesssim \frac { 1 } { m } \left( L _ { f } + \frac { \tau _ { f } + G ^ { 2 } + B _ { \mathrm { r e f } } } { \varepsilon _ { \mathrm { a l g } } ^ { 2 } } + \frac { M _ { \lambda } } { \varepsilon _ { \mathrm { a l g } } } \right) .\tag{5.36}
$$

Proof. By Theorem 5.5 and Proposition 5.6,

$$
\Phi _ { k + 1 } \leq ( 1 - m h ) D _ { k } + C ( \tau _ { f } + G ^ { 2 } + B _ { \mathrm { r e f } } ) h ^ { 2 } + C M _ { \lambda } h ^ { 2 } \sqrt { K _ { k } } .\tag{5.37}
$$

Young’s inequality gives

$$
C M _ { \lambda } h ^ { 2 } \sqrt { K _ { k } } \le h K _ { k } + C M _ { \lambda } ^ { 2 } h ^ { 3 } .\tag{5.38}
$$

If m $\iota h \leq 1$ , then

$$
( 1 - m h ) D _ { k } + h K _ { k } \le \left( 1 - \frac { m h } { 2 } \right) ( D _ { k } + 2 h K _ { k } ) = \left( 1 - \frac { m h } { 2 } \right) \Phi _ { k } .\tag{5.39}
$$

This proves (5.31). Iterating the afine recursion yields (5.32). The choices (5.33) and (5.34) ensure that mΦ $N \leq \varepsilon _ { \mathrm { a l g } } ^ { 2 } ,$ while $D _ { N } \leq \Phi _ { N }$ , so the Wasserstein conclusion follows. □

## 5.3 Moreau Approximation Bias

It remains to relate the smoothed target $\pi _ { \lambda }$ to the original nonsmooth target π. The next estimates give a universal bound on the bias introduced by replacing g with its Moreau envelope $g _ { \lambda }$

Let

$$
\delta _ { \lambda } ( x ) : = g ( x ) - g _ { \lambda } ( x ) .\tag{5.40}
$$

By choosing $y = x$ in the Moreau envelope, $\delta _ { \lambda } \geq 0$ . If g is G-Lipschitz, then

$$
0 \leq \delta _ { \lambda } ( x ) \leq { \frac { G ^ { 2 } \lambda } { 2 } } .\tag{5.41}
$$

Indeed, $g ( y ) \geq g ( x ) - G \| x - y \|$ , so

$$
g _ { \lambda } ( x ) \geq g ( x ) + \operatorname* { i n f } _ { r \geq 0 } \left\{ - G r + { \frac { r ^ { 2 } } { 2 \lambda } } \right\} = g ( x ) - { \frac { G ^ { 2 } \lambda } { 2 } } .\tag{5.42}
$$

Moreover,

$$
\frac { \mathrm { d } \pi _ { \lambda } } { \mathrm { d } \pi } ( x ) = \frac { \exp \{ \delta _ { \lambda } ( x ) \} } { \mathbb { E } _ { \pi } \exp \{ \delta _ { \lambda } ( X ) \} } .\tag{5.43}
$$

Lemma 5.8 (Bounded exponential tilt). Let $P$ be a probability measure and let $0 \leq \delta \leq a$ . Define $Q$ by

$$
\frac { \mathrm { d } Q } { \mathrm { d } P } = \frac { e ^ { \delta } } { \mathbb { E } _ { P } e ^ { \delta } } .\tag{5.44}
$$

Then

$$
\mathrm { K L } ( Q \| P ) \leq { \frac { a ^ { 2 } } { 8 } } .\tag{5.45}
$$

Proof. Let $K ( t ) : = \log \mathbb { E } _ { P } e ^ { t \delta }$ . Then $K ^ { \prime \prime } ( t ) = \mathrm { V a r } _ { P _ { t } } ( \delta )$ , where $\mathrm { d } P _ { t } = e ^ { t \delta - K ( t ) } \mathrm { d } P$ . Since $\delta \in [ 0 , a ]$ $\mathrm { V a r } _ { P _ { t } } ( \delta ) \leq a ^ { 2 } / 4$ . Therefore

$$
\mathrm { K L } ( Q \| P ) = K ^ { \prime } ( 1 ) - K ( 1 ) = \int _ { 0 } ^ { 1 } t K ^ { \prime \prime } ( t ) \mathrm { d } t \leq \frac { a ^ { 2 } } { 8 } .\tag{5.46}
$$

Proposition 5.9 (Universal Moreau bias). Under Assumptions $\it 4 . 1$ and $4 . 2 ,$

$$
\mathrm { K L } ( \pi _ { \lambda } \| \pi ) \leq \frac { G ^ { 4 } \lambda ^ { 2 } } { 3 2 } .\tag{5.47}
$$

Consequently,

$$
\sqrt { m } W _ { 2 } ( \pi _ { \lambda } , \pi ) \leq \frac { G ^ { 2 } \lambda } { 4 } .
$$

In particular, to make the Moreau bias at most $\varepsilon _ { \mathrm { b i a s } }$ in $\sqrt { m } W _ { 2 }$ , it is suficient to choose

$$
\lambda \leq \frac { 4 \varepsilon _ { \mathrm { b i a s } } } { G ^ { 2 } } .
$$

Proof. By the bound $0 \leq \delta _ { \lambda } \leq G ^ { 2 } \lambda / 2$ and the density-ratio identity

$$
\frac { \mathrm { d } \pi _ { \lambda } } { \mathrm { d } \pi } ( x ) = \frac { \exp \{ \delta _ { \lambda } ( x ) \} } { \mathbb { E } _ { \pi } \exp \{ \delta _ { \lambda } ( X ) \} } ,
$$

Lemma 5.8, applied with $a = G ^ { 2 } \lambda / 2$ , gives

$$
\mathrm { K L } ( \pi _ { \lambda } \| \pi ) \leq \frac { 1 } { 8 } \left( \frac { G ^ { 2 } \lambda } { 2 } \right) ^ { 2 } = \frac { G ^ { 4 } \lambda ^ { 2 } } { 3 2 } .
$$

Since $f + g$ is m-strongly convex, π is m-strongly log-concave. The Talagrand $T _ { 2 }$ inequality [21, 29] therefore yields

$$
W _ { 2 } ^ { 2 } ( \pi _ { \lambda } , \pi ) \leq \frac { 2 } { m } \mathrm { K L } ( \pi _ { \lambda } \| \pi ) \leq \frac { G ^ { 4 } \lambda ^ { 2 } } { 1 6 m } .
$$

Taking square roots gives the claimed Wasserstein bound.

## 5.4 Proof of the main theorem

Proof of Theorem 5.2. Theorem 5.7 gives

$$
\Phi _ { N } \leq e ^ { - m h N / 2 } \Phi _ { 0 } + \frac { C } { m } \Bigl \{ ( \tau _ { f } + G ^ { 2 } + B _ { \mathrm { r e f } } ) h + M _ { \lambda } ^ { 2 } h ^ { 2 } \Bigr \} .
$$

This is (5.1). The step-size and iteration choices in (5.2)–(5.3) make the right-hand side at most $\varepsilon _ { \mathrm { a l g } } ^ { 2 } / m ,$ , and therefore $\sqrt { m } W _ { 2 } ( \mu _ { N } , \pi _ { \lambda } ) \leq \varepsilon _ { \mathrm { a l g } }$

The bias estimate (5.5) is exactly Proposition 5.9. If $\lambda \leq 4 \varepsilon _ { \mathrm { b i a s } } / G ^ { 2 }$ <sub>,</sub> <sub>then</sub> √<sub>m</sub> $W _ { 2 } ( \pi _ { \lambda } , \pi ) \le \varepsilon _ { \mathrm { b i a s } }$ The triangle inequality for $W _ { 2 }$ gives

$$
\sqrt { m } W _ { 2 } ( \mu _ { N } , \pi ) \leq \sqrt { m } W _ { 2 } ( \mu _ { N } , \pi _ { \lambda } ) + \sqrt { m } W _ { 2 } ( \pi _ { \lambda } , \pi ) \leq \varepsilon _ { \mathrm { a l g } } + \varepsilon _ { \mathrm { b i a s } } .
$$

The displayed complexity follows by substituting $\varepsilon _ { \mathrm { a l g } } = \varepsilon _ { \mathrm { b i a s } } = \varepsilon / 2$ and suppressing universal constants and logarithmic factors. □

## 6 Bounding the Reference Active Trace

The main theorem reduces the fixed-λ analysis to the reference active trace

$$
B _ { \mathrm { r e f } } : = \frac { 1 } { h } \int _ { 0 } ^ { h } \mathbb { E } a _ { \lambda } ( Y _ { t } ^ { \mathrm { r e f } } ) \mathrm { d } t , \qquad Y _ { t } ^ { \mathrm { r e f } } : = T _ { h } ( X ^ { \mathrm { r e f } } ) + \sqrt { 2 } W _ { t } ^ { \mathrm { r e f } } , \qquad X ^ { \mathrm { r e f } } \sim \pi _ { \lambda } .
$$

Here $W ^ { \mathrm { r e f } }$ is a standard Brownian motion in $\mathbb { R } ^ { d }$ , independent of $X ^ { \mathrm { r e f } } , T _ { h } ( x ) = x - h \nabla U _ { \lambda } ( x )$ , and $a _ { \lambda } = \operatorname { t r } H _ { \lambda }$ . Equivalently, for each fixed t, $Y _ { t } ^ { \mathrm { r e f } }$ has the same law as $T _ { h } ( X ^ { \mathrm { r e f } } ) + \sqrt { 2 t } Z$ , where $Z \sim N ( 0 , I _ { d } )$

The purpose of this section is to give practical bounds on $B _ { \mathrm { r e f } }$ from the geometry of the nonsmooth penalty $g .$ The basic mechanism is that the Moreau curvature may be of order $1 / \lambda$ , but in many structured examples it is concentrated in an $O ( \lambda )$ -neighborhood of an active set. If the reference heat path assigns probability of order λ to such neighborhoods, then the factor $1 / \lambda$ is cancelled.

We first formulate this idea as an abstract curvature–tube interface. We then prove slice-density and propagation estimates for the reference heat path, which provide the tube bounds used below.

## 6.1 An abstract curvature-tube interface

The first result is only an interface: it says that a curvature decomposition plus a tube-mass bound implies a bound on $B _ { \mathrm { r e f } }$ . The following subsections provide general density estimates that make these tube bounds easy to verify.

Assumption 6.1 (Curvature decomposition). For the fixed $\lambda > 0 ,$ suppose that there are a nonnegative measurable function $b _ { \lambda }$ , closed sets $\Sigma _ { 1 } , \dots , \Sigma _ { J } \subset \mathbb { R } ^ { d }$ , and constants $a _ { j } , c _ { j } > 0$ such that, for Lebesgue-a.e. x,

$$
a _ { \lambda } ( x ) \leq b _ { \lambda } ( x ) + \sum _ { j = 1 } ^ { J } { \frac { c _ { j } } { \lambda } } \mathbf { 1 } \{ \mathrm { d i s t } ( x , \Sigma _ { j } ) \leq a _ { j } \lambda \} .\tag{6.1}
$$

Here $b _ { \lambda }$ represents the part of the curvature that is already integrable along the reference path. The sets $\Sigma _ { j }$ represent active or singular sets where the Moreau curvature may be large.

Assumption 6.2 (Reference tube mass). For the reference path $( Y _ { t } ^ { \mathrm { r e f } } ) _ { 0 \leq t \leq h }$ , suppose that there are constants $B _ { 0 } , S _ { j } , r _ { 0 } > 0$ such that

$$
\frac { 1 } { h } \int _ { 0 } ^ { h } \mathbb { E } b _ { \lambda } ( Y _ { t } ^ { \mathrm { r e f } } ) \mathrm { ~ d } t \leq B _ { 0 } ,\tag{6.2}
$$

and, for every $j \in \{ 1 , \ldots , J \}$ and every $0 < r \leq r _ { 0 }$ 2

$$
\frac { 1 } { h } \int _ { 0 } ^ { h } \mathbb { P } \{ \mathrm { d i s t } ( Y _ { t } ^ { \mathrm { r e f } } , \Sigma _ { j } ) \leq r \} ~ \mathrm { d } t \leq S _ { j } r ^ { q _ { j } } .\tag{6.3}
$$

The exponent $q _ { j }$ describes the tube-mass scaling near $\Sigma _ { j }$ . For instance, a codimension-one slab has exponent one, while a ball in a q-dimensional block has exponent $q .$

Proposition 6.3 (Curvature-tube control of $B _ { \mathrm { r e f } } )$ . Assume Assumptions 6.1 and 6.2. If $a _ { j } \lambda \le r _ { 0 }$ for every $j ,$ then

$$
B _ { \mathrm { r e f } } \leq B _ { 0 } + \sum _ { j = 1 } ^ { J } c _ { j } a _ { j } ^ { q _ { j } } S _ { j } \lambda ^ { q _ { j } - 1 } .\tag{6.4}
$$

In particular, every layer with $q _ { j } \geq 1$ produces no $1 / \lambda$ blow-up in the reference active trace.

Proof. For every $t > 0$ , the law of $Y _ { t } ^ { \mathrm { r e f } }$ has a density. Hence the a.e. curvature inequality (6.1) may be integrated along the path; the single endpoint $t = 0$ has no efect on the time average. Therefore

$$
\begin{array} { r l } & { B _ { \mathrm { r e f } } \leq \displaystyle \frac { 1 } { h } \int _ { 0 } ^ { h } \mathbb { E } b _ { \lambda } \big ( Y _ { t } ^ { \mathrm { r e f } } \big ) \mathrm { d } t + \sum _ { j = 1 } ^ { J } \frac { c _ { j } } { \lambda h } \int _ { 0 } ^ { h } \mathbb { P } \big \{ \mathrm { d i s t } ( Y _ { t } ^ { \mathrm { r e f } } , \Sigma _ { j } ) \leq a _ { j } \lambda \big \} \mathrm { d } t } \\ & { \qquad \leq { B _ { 0 } } + \displaystyle \sum _ { j = 1 } ^ { J } \frac { c _ { j } } { \lambda } S _ { j } ( a _ { j } \lambda ) ^ { q _ { j } } . } \end{array}\tag{6.5}
$$

Remark 6.4 (Role of the interface). Proposition 6.3 is a bookkeeping device rather than a final assumption. The rest of this section gives suficient conditions under which the tube-mass constants $S _ { j }$ and exponents $q _ { j }$ follow from slice-density bounds for $\pi _ { \lambda }$ and their propagation under $T _ { h }$ and the heat step.

## 6.2 Slice density of $\pi _ { \lambda }$

We now develop the density estimates used to control slab and ball probabilities for the reference heat path. The starting point is a slice-density bound for the stationary law $\pi _ { \lambda }$ . The key feature is that, along a chosen subspace $E ,$ , the bound is governed by directional quantities for $f$ and $^ { g , }$ rather than by the global Moreau smoothness scale $\lambda ^ { - 1 }$

Let $E \subset \mathbb { R } ^ { d }$ be a fixed linear subspace with dimension $q \geq 1$ . Let $P _ { E }$ denote the orthogonal projection onto $E$ , and let $E ^ { \perp }$ be the orthogonal complement. Every point $\boldsymbol { x } \in \mathbb { R } ^ { d }$ can be written uniquely as $x = z + u$ , where $z \in E ^ { \perp }$ and $u \in E$ . Lebesgue measure on $E$ is denoted by du.

Define the directional smoothness of f along E by

$$
L _ { E } : = \operatorname* { s u p } _ { x \in \mathbb { R } ^ { d } } \| P _ { E } \nabla ^ { 2 } f ( x ) P _ { E } \| _ { \mathrm { o p } } .\tag{6.6}
$$

By Assumption 4.1, $L _ { E } \leq L _ { f }$ . Define the directional Lipschitz size of $g$ along E by

$$
G _ { E } : = \operatorname* { s u p } \big \{ \| P _ { E } s \| : x \in \mathbb { R } ^ { d } , \ s \in \partial g ( x ) \big \} .\tag{6.7}
$$

Since $g$ is G-Lipschitz on $\mathbb { R } ^ { d } , G _ { E } \le G$ . Moreover, by the proximal optimality condition,

$$
\nabla g _ { \lambda } ( x ) \in \partial g ( p _ { \lambda } ( x ) ) , \qquad p _ { \lambda } ( x ) = \mathrm { p r o x } _ { \lambda g } ( x ) ,\tag{6.8}
$$

so $\| P _ { E } \nabla g _ { \lambda } ( x ) \| \le G _ { E }$ for every x.

Finally set

$$
B _ { E } : = \left[ \int _ { E } \exp \left\{ - \frac { L _ { E } } { 2 } \| u \| ^ { 2 } - 2 G _ { E } \| u \| \right\} ~ \mathrm { d } u \right] ^ { - 1 } .\tag{6.9}
$$

This constant is the scale of the slice-density bound below. It depends on the chosen subspace E only through $q , L _ { E } ,$ , and $G _ { E } ,$ and it does not involve the Moreau parameter λ.

When $E = \operatorname { s p a n } ( v )$ for a unit vector v, we write $L _ { v } , G _ { v }$ , and $B _ { v }$ in place of $L _ { E } , G _ { E }$ , and $B _ { E }$ In this case

$$
B _ { v } = \left[ \int _ { \mathbb { R } } \exp \left\{ - \frac { L _ { v } } { 2 } s ^ { 2 } - 2 G _ { v } | s | \right\} ~ \mathrm { d } s \right] ^ { - 1 } .\tag{6.10}
$$

Lemma 6.5 (Slice conditional density). Let $X ^ { \mathrm { r e f } } \sim \pi _ { \lambda }$ . For every $z \in E ^ { \bot }$ , the conditional density of $P _ { E } X ^ { \mathrm { r e f } }$ on the slice $z + E$ is bounded by $B _ { E }$ . Equivalently, the density proportional to

$$
u \mapsto \exp \{ - U _ { \lambda } ( z + u ) \} , \qquad u \in E ,\tag{6.11}
$$

has $L ^ { \infty } ( E )$ -norm at most $B _ { E }$

Proof. Fix $z \in E ^ { \perp }$ and write

$$
\phi _ { z } ( u ) : = U _ { \lambda } ( z + u ) , \qquad u \in E .\tag{6.12}
$$

The function $\phi _ { z }$ is convex and coercive on $E _ { \mathrm { { i } } }$ , because $f$ is strongly convex and $g _ { \lambda }$ is convex. Let $u _ { z }$ be a minimizer of $\phi _ { z }$ . The first-order condition on the slice gives

$$
P _ { E } \nabla f ( z + u _ { z } ) + P _ { E } \nabla g _ { \lambda } ( z + u _ { z } ) = 0 .\tag{6.13}
$$

By (6.8),

$$
\| P _ { E } \nabla f ( z + u _ { z } ) \| = \| P _ { E } \nabla g _ { \lambda } ( z + u _ { z } ) \| \le G _ { E } .\tag{6.14}
$$

For any $v \in E$ , the definition of $L _ { E }$ gives

$$
f ( z + u _ { z } + v ) \leq f ( z + u _ { z } ) + \langle P _ { E } \nabla f ( z + u _ { z } ) , v \rangle + \frac { L _ { E } } { 2 } \| v \| ^ { 2 } .\tag{6.15}
$$

Also, because $g _ { \lambda }$ is $G _ { E ^ { - } } \mathrm { L i }$ pschitz along $E _ { \mathrm { { i } } }$

$$
g _ { \lambda } ( z + u _ { z } + v ) \leq g _ { \lambda } ( z + u _ { z } ) + G _ { E } \| v \| .\tag{6.16}
$$

Combining the previous three displays,

$$
\phi _ { z } ( u _ { z } + v ) \leq \phi _ { z } ( u _ { z } ) + \frac { L _ { E } } { 2 } \| v \| ^ { 2 } + 2 G _ { E } \| v \| .\tag{6.17}
$$

The conditional normalizing constant on the slice is therefore at least

$$
e ^ { - \phi _ { z } ( u _ { z } ) } \int _ { E } \exp \left\{ - \frac { L _ { E } } { 2 } \| v \| ^ { 2 } - 2 G _ { E } \| v \| \right\} \ \mathrm { d } v .\tag{6.18}
$$

Since the conditional density is maximized at a minimizer of $\phi _ { z } ,$ , its maximum is bounded by $B _ { E }$ .

## 6.3 Density propagation by $T _ { h }$ and by the heat step

The previous lemma gives slice-density bounds for the stationary input $X ^ { \mathrm { r e f } } \sim \pi _ { \lambda }$ . The reference active trace, however, is evaluated along the one-step reference heat path

$$
Y _ { t } ^ { \mathrm { r e f } } = T _ { h } ( X ^ { \mathrm { r e f } } ) + \sqrt { 2 t } Z .
$$

We therefore need to propagate the slice-density bound through the deterministic Euler map $T _ { h }$ and then through the Gaussian heat step.

The deterministic part requires a small-step condition ensuring that $T _ { h }$ is injective on each E-slice. Once this is available, the heat step is harmless: convolution with a probability density cannot increase an $L ^ { \infty }$ density bound.

For the same subspace $E ,$ define

$$
\alpha _ { E } : = 1 - h ( L _ { E } + \lambda ^ { - 1 } ) .\tag{6.19}
$$

The useful case is $\alpha _ { E } > 0$

Lemma 6.6 (Slice density propagation). Let $X ^ { \mathrm { r e f } } \sim \pi _ { \lambda }$ and $Y _ { t } ^ { \mathrm { r e f } } = T _ { h } ( X ^ { \mathrm { r e f } } ) + \sqrt { 2 t } Z$ , where $Z \sim N ( 0 , I _ { d } )$ is independent of $X ^ { \mathrm { r e f } }$ $I f \alpha _ { E } > 0$ , then, for every Borel set $A \subset E$ and every $0 \leq t \leq h _ { \cdot }$

$$
\mathbb { P } \{ P _ { E } Y _ { t } ^ { \mathrm { r e f } } \in A \} \le \frac { B _ { E } } { \alpha _ { E } ^ { q } } \mathrm { \ v o l } _ { E } ( A ) ,\tag{6.20}
$$

where vol denotes Lebesgue measure on $E .$

Proof. For fixed $z \in E ^ { \bot }$ , consider the map from the slice $z + E$ to the E-coordinate after the deterministic step:

$$
S _ { z } ( u ) : = P _ { E } T _ { h } ( z + u ) , \qquad u \in E .\tag{6.21}
$$

We first study the deterministic map $S _ { z }$ on each slice. Fix $z \in E ^ { \perp }$ . For $u , w \in E$ , the fundamental theorem of calculus and the definition of $L _ { E }$ give

$$
\left\| P _ { E } \big \{ \nabla f ( z + u ) - \nabla f ( z + w ) \big \} \right\| = \left\| \int _ { 0 } ^ { 1 } P _ { E } \nabla ^ { 2 } f \big ( z + w + s ( u - w ) \big ) P _ { E } ( u - w ) d s \right\| \leq L _ { E } \| u - w \| .
$$

Here we used $P _ { E } ( u - w ) = u - w$ . Moreover, by Lemma 3.1, $\mathrm { L i p } ( \nabla g _ { \lambda } ) \leq \lambda ^ { - 1 }$ , and hence

$$
\begin{array} { r } { \left\| P _ { E } \big \{ \nabla g _ { \lambda } ( z + u ) - \nabla g _ { \lambda } ( z + w ) \big \} \right\| \leq \lambda ^ { - 1 } \| u - w \| . } \end{array}
$$

Since $U _ { \lambda } = f + g _ { \lambda }$ , it follows that

$$
\begin{array} { r } { \left\| P _ { E } \big \{ \nabla U _ { \lambda } ( z + u ) - \nabla U _ { \lambda } ( z + w ) \big \} \right\| \le ( L _ { E } + \lambda ^ { - 1 } ) \| u - w \| . } \end{array}\tag{6.22}
$$

Using the definition of $S _ { z }$ in (6.21), the reverse triangle inequality and (6.22) yield

$$
\begin{array} { r l } & { \| S _ { z } ( u ) - S _ { z } ( w ) \| = \left\| u - w - h P _ { E } \big \{ \nabla U _ { \lambda } ( z + u ) - \nabla U _ { \lambda } ( z + w ) \big \} \right\| } \\ & { \qquad \geq \| u - w \| - h \left\| P _ { E } \big \{ \nabla U _ { \lambda } ( z + u ) - \nabla U _ { \lambda } ( z + w ) \big \} \right\| } \\ & { \qquad \geq \left\{ 1 - h ( L _ { E } + \lambda ^ { - 1 } ) \right\} \| u - w \| } \\ & { \qquad = \alpha _ { E } \| u - w \| . } \end{array}\tag{6.23}
$$

In particular, $S _ { z }$ is injective.

We next prove that $S _ { z }$ is surjective. Fix $y \in E$ . Consider the self-map of E given by

$$
u \longmapsto y + h P _ { E } \nabla U _ { \lambda } ( z + u ) .
$$

By (6.22), this map is Lipschitz with constant

$$
h ( L _ { E } + \lambda ^ { - 1 } ) = 1 - \alpha _ { E } < 1 .
$$

Since E is a finite-dimensional Hilbert space and therefore complete, the Banach fixed-point theorem gives a unique $u \in E$ such that

$$
\boldsymbol { u } = \boldsymbol { y } + h P _ { E } \nabla U _ { \lambda } ( \boldsymbol { z } + \boldsymbol { u } ) .
$$

By (6.21), this identity is equivalent to $S _ { z } ( u ) = y$ . Since $y \in E$ was arbitrary, $S _ { z } : E \to E$ is surjective and hence bijective.

Furthermore, (6.23) implies that its inverse $S _ { z } ^ { - 1 } : E \to E$ satisfies

$$
\| S _ { z } ^ { - 1 } ( y ) - S _ { z } ^ { - 1 } ( y ^ { \prime } ) \| \leq \alpha _ { E } ^ { - 1 } \| y - y ^ { \prime } \| , \qquad y , y ^ { \prime } \in E .
$$

Thus $S _ { z } ^ { - 1 }$ is globally $\alpha _ { E } ^ { - 1 } { } ^ { - 1 }$ Lipschitz on all of E.

After choosing an orthonormal basis of E, we identify E isometrically with R<sup>q</sup>. The standard measure-distortion inequality for Lipschitz maps [9, Theorem 2.8] can therefore be applied directly to the globally defined map $S _ { z } ^ { - 1 } : E \to E$ . Consequently, for every Borel set $A \subset E$

$$
\operatorname { v o l } _ { E } \left( S _ { z } ^ { - 1 } ( A ) \right) \leq \operatorname { L i p } ( S _ { z } ^ { - 1 } ) ^ { q } \operatorname { v o l } _ { E } ( A ) \leq \alpha _ { E } ^ { - q } \operatorname { v o l } _ { E } ( A ) .\tag{6.24}
$$

Here

$$
S _ { z } ^ { - 1 } ( A ) = \{ u \in E : S _ { z } ( u ) \in A \} .
$$

This set is Borel because $S _ { z }$ is continuous.

We now propagate the conditional slice-density bound. For $z \in E ^ { \bot }$ , define

$$
\rho _ { \lambda , z } ( u ) : = \frac { \exp \{ - U _ { \lambda } ( z + u ) \} } { \displaystyle \int _ { E } \exp \{ - U _ { \lambda } ( z + v ) \} ~ \mathrm { d } v } , \qquad u \in E .
$$

The kernel $\rho _ { \lambda , z } ( u )$ du is a version of the conditional law of $P _ { E } X ^ { \mathrm { r e f } }$ given $P _ { E ^ { \bot } } X ^ { \mathrm { r e f } } = z$ . By Lemma 6.5,

$$
\| \rho _ { \lambda , z } \| _ { L ^ { \infty } ( E ) } \leq B _ { E } .
$$

Therefore, for every Borel set $A \subset E$

$$
\begin{array} { r l } & { \mathbb { P } \big \{ P _ { E } T _ { h } ( X ^ { \mathrm { r e f } } ) \in A \ \big \vert \ P _ { E ^ { \perp } } X ^ { \mathrm { r e f } } = z \big \} } \\ & { \quad = \displaystyle \int _ { S _ { z } ^ { - 1 } ( A ) } \rho _ { \lambda , z } ( u ) \mathrm { d } u } \\ & { \quad \le B _ { E } \mathrm { v o l } _ { E } \big ( S _ { z } ^ { - 1 } ( A ) \big ) } \\ & { \quad \le B _ { E } \alpha _ { E } ^ { - q } \mathrm { v o l } _ { E } ( A ) , } \end{array}
$$

where the last inequality follows from (6.24). The bound is uniform in $z \in E ^ { \bot }$ . Integrating with respect to the law of $P _ { E ^ { \bot } } X ^ { \mathrm { r e f } }$ gives

$$
\mathbb { P } \{ P _ { E } T _ { h } ( X ^ { \mathrm { r e f } } ) \in A \} \le B _ { E } \alpha _ { E } ^ { - q } \mathrm { v o l } _ { E } ( A ) .\tag{6.25}
$$

Finally, let $0 \leq t \leq h$ . Since $Z$ is independent of $X ^ { \mathrm { r e f } }$ , (6.25) applied conditionally on $Z$ gives

$$
\begin{array} { r l } & { \mathbb { P } \big \{ P _ { E } Y _ { t } ^ { \mathrm { r e f } } \in A \big \} = \mathbb { E } \left[ \mathbb { P } \left\{ P _ { E } T _ { h } ( X ^ { \mathrm { r e f } } ) \in A - \sqrt { 2 t } P _ { E } Z \Big | Z \right\} \right] } \\ & { \qquad \leq B _ { E } \alpha _ { E } ^ { - q } \mathbb { E } \left[ \mathrm { v o l } _ { E } \big ( A - \sqrt { 2 t } P _ { E } Z \big ) \right] } \\ & { \qquad = B _ { E } \alpha _ { E } ^ { - q } \mathrm { v o l } _ { E } ( A ) , } \end{array}
$$

where the last equality follows from translation invariance of Lebesgue measure on $E .$ This proves (6.20) simultaneously for all $0 \leq t \leq h$ □

## 6.4 Slab, ball, and inverse-radius consequences

Lemma 6.6 immediately gives the estimates used in the examples. We state them separately to make later verification modules short.

Corollary 6.7 (Slab probability). Let $v \in \mathbb { R } ^ { d }$ be a unit vector and let $\alpha _ { v } = 1 - h ( L _ { v } + \lambda ^ { - 1 } ) > 0$ Then, for every $a \in \mathbb { R } , r > 0$ , and $0 \leq t \leq h$

$$
\mathbb { P } \{ | \langle v , Y _ { t } ^ { \mathrm { r e f } } \rangle - a | \le r \} \le \frac { 2 r B _ { v } } { \alpha _ { v } } .\tag{6.26}
$$

Equivalently, if $\mathcal { \cdot } d _ { 0 } \in \mathbb { R } ^ { d } \backslash \{ 0 \}$ and $v = d _ { 0 } / \lVert d _ { 0 } \rVert$ , then

$$
\mathbb { P } \{ | d _ { 0 } ^ { \top } Y _ { t } ^ { \mathrm { r e f } } - a | \leq r \} \leq \frac { 2 r B _ { v } } { \| d _ { 0 } \| \alpha _ { v } } .\tag{6.27}
$$

Proof. Apply Lemma 6.6 with $E = \operatorname { s p a n } ( v )$ and with $A = \{ s v : s \in [ a - r , a + r ] \}$ . For (6.27), rewrite the event as

$$
\left| \left. v , Y _ { t } ^ { \mathrm { r e f } } \right. - \frac a { \| d _ { 0 } \| } \right| \leq \frac { r } { \| d _ { 0 } \| } .\tag{6.28}
$$

Corollary 6.8 (Ball probability). Let $E \subset \mathbb { R } ^ { d }$ have dimension q, and assume $\alpha _ { E } > 0$ . Let $v _ { q }$ be the volume of the unit ball in R<sup>q</sup>. Then, for every $y \in E , r > 0$ , and $0 \leq t \leq h$ 2

$$
\mathbb { P } \{ \| P _ { E } Y _ { t } ^ { \mathrm { r e f } } - y \| \le r \} \le \frac { B _ { E } } { \alpha _ { E } ^ { q } } v _ { q } r ^ { q } .\tag{6.29}
$$

Proof. Apply Lemma 6.6 to the ball $A = \left\{ u \in E : \| u - y \| \leq r \right\}$ , whose E-volume is $v _ { q } r ^ { q }$ □

Corollary 6.9 (Inverse-radius bound). Let $E \subset \mathbb { R } ^ { d }$ have dimension $q \geq 2$ , and assume $\alpha _ { E } > 0$ Then, for every $y \in E$ and $0 \leq t \leq h$

$$
\mathbb { E } \frac { 1 } { \| P _ { E } Y _ { t } ^ { \mathrm { r e f } } - y \| } \le C _ { q } \left( \frac { B _ { E } } { \alpha _ { E } ^ { q } } \right) ^ { 1 / q } , \qquad C _ { q } : = \frac { q } { q - 1 } { v } _ { q } ^ { 1 / q } .\tag{6.30}
$$

Proof. Let $W = P _ { E } Y _ { t } ^ { \mathrm { r e f } }$ . By Lemma 6.6, W has density at most

$$
\overline { { B } } _ { E } : = \frac { B _ { E } } { \alpha _ { E } ^ { q } } .\tag{6.31}
$$

Therefore, for every $s > 0$

$$
\begin{array} { r } { { \mathbb P } \{ \| W - y \| ^ { - 1 } > s \} = { \mathbb P } \{ \| W - y \| < s ^ { - 1 } \} \le \operatorname* { m i n } \{ 1 , \overline { B } _ { E } v _ { q } s ^ { - q } \} . } \end{array}\tag{6.32}
$$

Integrating the tail bound gives

$$
\begin{array} { l } { \displaystyle \mathbb { E } \frac { 1 } { \| W - y \| } = \int _ { 0 } ^ { \infty } \mathbb { P } \{ \| W - y \| ^ { - 1 } > s \} \ \mathrm { d } s } \\ { \displaystyle \qquad \leq \int _ { 0 } ^ { ( \overline { { B } } _ { E } v _ { q } ) ^ { 1 / q } } 1 \ \mathrm { d } s + \overline { { B } } _ { E } v _ { q } \int _ { ( \overline { { B } } _ { E } v _ { q } ) ^ { 1 / q } } ^ { \infty } s ^ { - q } \ \mathrm { d } s } \\ { \displaystyle \qquad = \frac { q } { q - 1 } ( \overline { { B } } _ { E } v _ { q } ) ^ { 1 / q } . } \end{array}\tag{6.33}
$$

## 7 Examples and Verification Modules

This section applies the reference active-trace framework to four classes of nonsmooth penalties. We retain the notation of the previous sections. Let $X ^ { \mathrm { r e f } } \sim \pi _ { \lambda }$ , let $( W _ { t } ^ { \mathrm { r e f } } ) _ { t \geq 0 }$ be a standard Brownian motion in $\mathbb { R } ^ { d }$ , independent of $X ^ { \mathrm { r e f } }$ , and define

$$
Y _ { t } ^ { \mathrm { r e f } } : = T _ { h } ( X ^ { \mathrm { r e f } } ) + \sqrt { 2 } W _ { t } ^ { \mathrm { r e f } } , \qquad 0 \le t \le h ,
$$

where $T _ { h } ( x ) = x - h \nabla U _ { \lambda } ( x ) , U _ { \lambda } = f + g _ { \lambda }$ . The corresponding reference active trace is

$$
B _ { \mathrm { r e f } } : = \frac { 1 } { h } \int _ { 0 } ^ { h } \mathbb { E } a _ { \lambda } ( Y _ { t } ^ { \mathrm { r e f } } ) \mathrm { d } t .
$$

Each example has the same structure: we first identify the region on which the Moreau curvature $a _ { \lambda }$ is active, then control the mass of that region under the reference heat path using the slice-density estimates of Section $6 ,$ and finally substitute the resulting bound on $B _ { \mathrm { r e f } }$ into Theorem 5.2.

We use the symmetric error split $\varepsilon _ { \mathrm { a l g } } = \varepsilon _ { \mathrm { b i a s } } = \varepsilon / 2$ and the universal Moreau-bias choice

$$
\lambda = \frac { 2 \varepsilon } { G ^ { 2 } } .\tag{7.1}
$$

Automatic compatibility with the density-propagation estimates. For a subspace $E _ { \mathrm { { i } } }$ Lemma 6.6 is applied with

$$
\alpha _ { E } : = 1 - h ( L _ { E } + \lambda ^ { - 1 } ) .
$$

In the estimates below, it is suficient to have $\alpha _ { E } \geq 1 / 2$ . This is not an additional assumption. It is automatic under the step-size requirement (5.2) in Theorem 5.2.

Indeed, the $L _ { f } ^ { - 1 }$ restriction gives

$$
h L _ { f } \leq c ,
$$

whereas the $\varepsilon _ { \mathrm { a l g } } ^ { 2 } / ( \tau _ { f } + G ^ { 2 } + B _ { \mathrm { r e f } } )$ restriction gives, after adjusting the universal constant $c ,$

$$
h \leq c \frac { \varepsilon ^ { 2 } } { G ^ { 2 } } .
$$

Since $L _ { E } \leq L _ { f } , \lambda = 2 \varepsilon / G ^ { 2 }$ , and $0 < \varepsilon \le 1$ , it follows that

$$
h ( L _ { E } + \lambda ^ { - 1 } ) \leq h L _ { f } + { \frac { h G ^ { 2 } } { 2 \varepsilon } } \leq c + { \frac { c \varepsilon } { 2 } } \leq { \frac { 3 c } { 2 } } .
$$

Taking the universal constant c suficiently small, every step-size choice below therefore satisfies, for each relevant subspace $E$

$$
h ( L _ { E } + \lambda ^ { - 1 } ) \leq { \frac { 1 } { 2 } } , \qquad \alpha _ { E } \geq { \frac { 1 } { 2 } } .\tag{7.2}
$$

## 7.1 One-dimensional finite-kink piecewise-linear penalties

Let $d = 1$ , and let $g : \mathbb { R } $ R be a convex piecewise-linear function with finitely many kink points $\theta _ { j }$ . Denote the left and right slopes at $\theta _ { j }$ by $s _ { j } ^ { - }$ and $s _ { j } ^ { + }$ , and put

$$
\Delta _ { j } = s _ { j } ^ { + } - s _ { j } ^ { - } > 0 , \qquad \Delta = \sum _ { j } \Delta _ { j } .\tag{7.3}
$$

Let $G = \mathrm { L i p } ( g ) = \operatorname* { s u p } | s |$ , where the supremum is over the slopes of $^ { g , }$ and define

$$
B = \left[ \int _ { \mathbb { R } } \exp \left\{ - \frac { L _ { f } } { 2 } u ^ { 2 } - 2 G | u | \right\} ~ \mathrm { d } u \right] ^ { - 1 } .\tag{7.4}
$$

Lemma 7.1 (Moreau curvature of a one-dimensional finite-kink PL function). There is a Borel set $A _ { \lambda } \subset \mathbb { R }$ such that

$$
g _ { \lambda } ^ { \prime \prime } ( x ) \leq { \frac { 1 } { \lambda } } { \mathbf { 1 } } _ { A _ { \lambda } } ( x ) \quad f o r { \ a . e . \ x } , \quad \quad | A _ { \lambda } | \leq \lambda \Delta .\tag{7.5}
$$

Indeed, we may take

$$
A _ { \lambda } = \bigcup _ { j } [ \theta _ { j } + \lambda s _ { j } ^ { - } , \theta _ { j } + \lambda s _ { j } ^ { + } ] .\tag{7.6}
$$

Proof. Let $p _ { \lambda } = \mathrm { p r o x } _ { \lambda g }$ . The optimality condition is

$$
{ \frac { x - p _ { \lambda } ( x ) } { \lambda } } \in \partial g ( p _ { \lambda } ( x ) ) .\tag{7.7}
$$

Consider an open afine piece I of $g ,$ and let its slope be $s _ { I }$ . If $p _ { \lambda } ( x _ { 0 } ) \in I _ { \lambda }$ , then $x _ { 0 } = p _ { \lambda } ( x _ { 0 } ) + \lambda s _ { I }$ . For all x close enough to x<sub>0</sub>, the point $x - \lambda s _ { I }$ still lies in I. It satisfies $( x - ( x - \lambda s _ { I } ) ) / \lambda = s _ { I } \in \partial g ( x - \lambda s _ { I } )$ so by uniqueness of the proximal point,

$$
p _ { \lambda } ( x ) = x - \lambda s _ { I }
$$

for all such $x .$ Hence $D p _ { \lambda } ( x ) = 1$ locally on this set. By the Moreau identity $g _ { \lambda } ^ { \prime \prime } = \lambda ^ { - 1 } ( 1 - D p _ { \lambda } )$ at a.e. diferentiability points of $p _ { \lambda }$ , the Moreau curvature is zero whenever the proximal point belongs to the interior of an afine piece.

Therefore nonzero curvature can occur only at points x for which $p _ { \lambda } ( x )$ is a kink. If $p _ { \lambda } ( x ) = \theta _ { j }$ then (7.7) gives

$$
x = \theta _ { j } + \lambda s , \qquad s \in \partial g ( \theta _ { j } ) = [ s _ { j } ^ { - } , s _ { j } ^ { + } ] .
$$

Thus such $x _ { \mathrm { ~ S ~ } } ^ { \prime }$ lie in the interval $[ \theta _ { j } + \lambda s _ { i } ^ { - } , \theta _ { j } + \lambda s _ { i } ^ { + } ]$ , whose length is $\lambda \Delta _ { j }$ . Taking the union over all kinks gives (7.6) and $\begin{array} { r } { | A _ { \lambda } | \le \sum _ { j } \lambda \Delta _ { j } = \overset { \cdot } { \lambda } \Delta } \end{array}$ . The pointwise bound $0 \leq g _ { \lambda } ^ { \prime \prime } \leq \lambda ^ { - 1 }$ gives (7.5).

Proposition 7.2 (One-dimensional finite-kink PL reference trace). Let

$$
A _ { \mathrm { 1 d } } : = 2 B \Delta .\tag{7.8}
$$

There exist universal constants $c , C > 0$ such that the following holds. For $0 < \varepsilon \le 1$ , assume $G > 0$ set

$$
\lambda = \frac { 2 \varepsilon } { G ^ { 2 } } ,\tag{7.9}
$$

and choose the step size

$$
h = c \operatorname* { m i n } \left\{ L _ { f } ^ { - 1 } , \frac { \varepsilon ^ { 2 } } { \tau _ { f } + G ^ { 2 } + A _ { \mathrm { 1 d } } } , \varepsilon \lambda \right\} .\tag{7.10}
$$

Then

$$
B _ { \mathrm { r e f } } \leq A _ { 1 \mathrm { d } } , \qquad M _ { \lambda } = \frac { 1 } { \lambda } .\tag{7.11}
$$

If

$$
N \geq \frac { C } { m h } \log \left( 1 + \frac { m \Phi _ { 0 } ( \lambda , h ) } { \varepsilon ^ { 2 } } \right) ,\tag{7.12}
$$

then $\sqrt { m } W _ { 2 } ( \mu _ { N } , \pi ) \le \varepsilon$ . Equivalently, with the step size (7.10), it is suficient to take

$$
N _ { \mathrm { 1 d } } ( \varepsilon ) \leq \frac { C } { m } \left[ L _ { f } + \frac { \tau _ { f } + G ^ { 2 } + A _ { \mathrm { 1 d } } } { \varepsilon ^ { 2 } } \right] \log \left( 1 + \frac { m \Phi _ { 0 } ( \lambda , h ) } { \varepsilon ^ { 2 } } \right) .\tag{7.13}
$$

Proof. By Lemma 7.1,

$$
B _ { \mathrm { r e f } } \leq \frac { 1 } { \lambda h } \int _ { 0 } ^ { h } \mathbb { P } \{ Y _ { t } ^ { \mathrm { r e f } } \in A _ { \lambda } \} \ \mathrm { d } t .
$$

The step size (7.10) implies (7.2), as explained at the beginning of the section. By Lemma $6 . 6 ,$

$$
\mathbb { P } \{ Y _ { t } ^ { \mathrm { r e f } } \in A _ { \lambda } \} \le 2 B | A _ { \lambda } | \le 2 B \lambda \Delta .
$$

This proves $B _ { \mathrm { r e f } } \leq A _ { \mathrm { 1 d } }$ . The value of $M _ { \lambda }$ is the one-dimensional global trace bound. The choices (7.9) and (7.10) are exactly Theorem 5.2, with the symmetric error split and with $B _ { \mathrm { r e f } } \leq A _ { \mathrm { 1 d } }$ , up to universal constants. This gives $\sqrt { m } W _ { 2 } ( \mu _ { N } , \pi ) \le \varepsilon$ under (7.12). Taking the reciprocal of (7.10) and using $1 / ( \varepsilon \lambda ) = G ^ { 2 } / ( 2 \varepsilon ^ { 2 } )$ gives (7.13). □

## 7.2 Separable finite-kink PL penalties and weighted lasso

Now let $d \geq 1$ and consider

$$
g ( x ) = \sum _ { i = 1 } ^ { d } { g _ { i } ( x _ { i } ) } ,\tag{7.14}
$$

where each $g _ { i } : \mathbb { R }  \mathbb { R }$ is convex, Lipschitz, and piecewise-linear with finitely many kinks. Let $G _ { i } = \mathrm { L i p } ( g _ { i } )$ , and let $\Delta _ { i }$ be the total slope jump of $g _ { i }$ , as in (7.3). A global Lipschitz constant of $g$ is

$$
G = \left( \sum _ { i = 1 } ^ { d } G _ { i } ^ { 2 } \right) ^ { 1 / 2 } .\tag{7.15}
$$

For each coordinate set

$$
L _ { i } = \underset { x \in \mathbb { R } ^ { d } } { \operatorname* { s u p } } \partial _ { i i } ^ { 2 } f ( x ) , \qquad B _ { i } = \left[ \int _ { \mathbb { R } } \exp \left\{ - \frac { L _ { i } } { 2 } u ^ { 2 } - 2 G _ { i } | u | \right\} \ \mathrm { d } u \right] ^ { - 1 } .\tag{7.16}
$$

The smooth part f is not assumed to be separable. The constants $L _ { i }$ and $B _ { i }$ only control onedimensional conditional slices in coordinate directions.

Proposition 7.3 (Separable finite-kink PL reference trace). Let

$$
A _ { \mathrm { s e p } } : = 2 \sum _ { i = 1 } ^ { d } B _ { i } \Delta _ { i } .\tag{7.17}
$$

There exist universal constants $c , C > 0$ such that the following holds. For $0 < \varepsilon \le 1$ , assume $G > 0$ set

$$
\lambda = \frac { 2 \varepsilon } { G ^ { 2 } } ,\tag{7.18}
$$

and choose

$$
h = c \operatorname* { m i n } \left\{ L _ { f } ^ { - 1 } , \frac { \varepsilon ^ { 2 } } { \tau _ { f } + G ^ { 2 } + A _ { \mathrm { s e p } } } , \frac { \varepsilon \lambda } { d } \right\} .\tag{7.19}
$$

Then

$$
B _ { \mathrm { r e f } } \leq A _ { \mathrm { s e p } } , \qquad M _ { \lambda } = \frac { d } { \lambda } .\tag{7.20}
$$

If

$$
N \geq \frac { C } { m h } \log \left( 1 + \frac { m \Phi _ { 0 } ( \lambda , h ) } { \varepsilon ^ { 2 } } \right) ,\tag{7.21}
$$

then $\sqrt { m } W _ { 2 } ( \mu _ { N } , \pi ) \le \varepsilon$ . Equivalently, with the step size (7.19), it is suficient to take

$$
N _ { \mathrm { s e p } } ( \varepsilon ) \leq \frac { C } { m } \left[ L _ { f } + \frac { \tau _ { f } + A _ { \mathrm { s e p } } + d G ^ { 2 } } { \varepsilon ^ { 2 } } \right] \log \left( 1 + \frac { m \Phi _ { 0 } ( \lambda , h ) } { \varepsilon ^ { 2 } } \right) .\tag{7.22}
$$

Proof. The Moreau envelope of a sum of coordinate functions is the sum of their one-dimensional Moreau envelopes. Lemma 7.1 gives sets $A _ { i , \lambda } \subset \mathbb { R }$ with $| A _ { i , \lambda } | \le \lambda \Delta _ { i }$ such that

$$
a _ { \lambda } ( x ) \leq { \frac { 1 } { \lambda } } \sum _ { i = 1 } ^ { d } \mathbf { 1 } _ { \left\{ x _ { i } \in A _ { i , \lambda } \right\} } \quad { \mathrm { f o r ~ a . e . ~ } } x .\tag{7.23}
$$

The step size (7.19) implies (7.2). By Lemma 6.6 in the i-th coordinate direction,

$$
\mathbb { P } \{ Y _ { t , i } ^ { \mathrm { r e f } } \in A _ { i , \lambda } \} \le 2 B _ { i } | A _ { i , \lambda } | \le 2 B _ { i } \lambda \Delta _ { i } .
$$

Substitution in (7.23) and time averaging prove $B _ { \mathrm { r e f } } \leq A _ { \mathrm { s e p } }$ . The trace bound $M _ { \lambda } = d / \lambda$ follows from $0 \preceq H _ { \lambda } \preceq \lambda ^ { - 1 } I$ . The choices (7.18) and (7.19) are Theorem 5.2 with $B _ { \mathrm { r e f } } \leq A _ { \mathrm { s e p } }$ and $M _ { \lambda } = d / \lambda$ 1, up to universal constants. Since $d / ( \varepsilon \lambda ) = d G ^ { 2 } / ( 2 \varepsilon ^ { 2 } )$ , the displayed complexity follows. □

Corollary 7.4 (Weighted lasso). Let

$$
g ( x ) = \sum _ { i = 1 } ^ { d } \gamma _ { i } | x _ { i } | , \qquad \gamma _ { i } \ge 0 .\tag{7.24}
$$

Then $\begin{array} { r } { G ^ { 2 } = \sum _ { i } \gamma _ { i } ^ { 2 } , \Delta _ { i } = 2 \gamma _ { i } } \end{array}$ , and

$$
A _ { \mathrm { w l } } : = 4 \sum _ { i = 1 } ^ { d } \gamma _ { i } B _ { i } .\tag{7.25}
$$

There exist universal constants $c , C > 0$ such that, for $0 < \varepsilon \le 1$ and $G > 0$ , set

$$
\lambda = \frac { 2 \varepsilon } { G ^ { 2 } } ,\tag{7.26}
$$

and choose

$$
h = c \operatorname* { m i n } \left\{ L _ { f } ^ { - 1 } , \frac { \varepsilon ^ { 2 } } { \tau _ { f } + G ^ { 2 } + A _ { \mathrm { w l } } } , \frac { \varepsilon \lambda } { d } \right\} .\tag{7.27}
$$

Then

$$
B _ { \mathrm { r e f } } \leq A _ { \mathrm { w l } } , \qquad M _ { \lambda } = \frac { d } { \lambda } .\tag{7.28}
$$

If

$$
N \geq \frac { C } { m h } \log \left( 1 + \frac { m \Phi _ { 0 } ( \lambda , h ) } { \varepsilon ^ { 2 } } \right) ,\tag{7.29}
$$

then $\sqrt { m } W _ { 2 } ( \mu _ { N } , \pi ) \le \varepsilon$ . In particular,

$$
N _ { \mathrm { w l } } ( \varepsilon ) \leq \frac { C } { m } \left[ L _ { f } + \frac { \tau _ { f } + A _ { \mathrm { w l } } + d G ^ { 2 } } { \varepsilon ^ { 2 } } \right] \log \left( 1 + \frac { m \Phi _ { 0 } ( \lambda , h ) } { \varepsilon ^ { 2 } } \right) .\tag{7.30}
$$

## 7.3 Group lasso

Let $\{ 1 , \ldots , d \}$ be partitioned into disjoint blocks b. The block variable is denoted by $x _ { b } \in \mathbb { R } ^ { q _ { b } }$ , where $q _ { b }$ is the block size and $\textstyle \sum _ { b } q _ { b } = d .$ . Consider

$$
g ( x ) = \sum _ { b } \gamma _ { b } \| x _ { b } \| , \gamma _ { b } > 0 .\tag{7.31}
$$

A global Lipschitz constant is

$$
G = \left( \sum _ { b } \gamma _ { b } ^ { 2 } \right) ^ { 1 / 2 } .\tag{7.32}
$$

Let $P _ { b }$ be the orthogonal projection onto block b. Define

$$
L _ { b } = \operatorname* { s u p } _ { x \in \mathbb { R } ^ { d } } \| P _ { b } \nabla ^ { 2 } f ( x ) P _ { b } \| _ { \mathrm { o p } } , \qquad B _ { b } = \left[ \int _ { \mathbb { R } ^ { q _ { b } } } \exp \left\{ - \frac { L _ { b } } { 2 } \| u \| ^ { 2 } - 2 \gamma _ { b } \| u \| \right\} ~ \mathrm { d } u \right] ^ { - 1 } .\tag{7.33}
$$

Let $v _ { q }$ be the volume of the unit ball in $\mathbb { R } ^ { q }$ . For $q \geq 2$ , set

$$
C _ { q } = \frac { q } { q - 1 } v _ { q } ^ { 1 / q } .\tag{7.34}
$$

Lemma 7.5 (Block Moreau trace for $\gamma \Vert z \Vert )$ . Let $q \ge 1 , \gamma > 0$ , and $\varphi ( z ) = \gamma \| z \|$ on $\mathbb { R } ^ { q }$ . For $r = \| z \|$ ，

$$
\mathrm { t r } \nabla ^ { 2 } \varphi _ { \lambda } ( z ) = \frac { q } { \lambda } { \bf 1 } _ { \{ r \leq \lambda \gamma \} } + \frac { \gamma ( q - 1 ) } { r } { \bf 1 } _ { \{ r > \lambda \gamma \} } \quad f o r \ a . e . \ z .\tag{7.35}
$$

Proof. The proximal map of $\lambda \gamma \| \cdot \|$ is block soft-thresholding:

$$
\operatorname { p r o x } _ { \lambda \varphi } ( z ) = \left( 1 - { \frac { \lambda \gamma } { \| z \| } } \right) _ { + } z .
$$

If $r < \lambda \gamma$ , the proximal map is locally constant, so $\nabla ^ { 2 } \varphi _ { \lambda } = \lambda ^ { - 1 } I _ { q }$ and the trace is $q / \lambda$ . If $r > \lambda \gamma$ then $\varphi _ { \lambda } ( z ) = \gamma r - \lambda \gamma ^ { 2 } / 2$ . The Hessian of r has eigenvalue 0 in the radial direction and eigenvalue $1 / r$ in the $q - 1$ tangential directions. The sphere $r = \lambda \gamma$ is null and its value is irrelevant. □

Proposition 7.6 (Group lasso reference trace). Let

$$
A _ { \mathrm { g r p } } : = \sum _ { b } q _ { b } v _ { q _ { b } } 2 ^ { q _ { b } } B _ { b } \gamma _ { b } ^ { q _ { b } } + \sum _ { b : q _ { b } \geq 2 } 2 \gamma _ { b } ( q _ { b } - 1 ) C _ { q _ { b } } B _ { b } ^ { 1 / q _ { b } } .\tag{7.36}
$$

There exist universal constants $c , C > 0$ such that the following holds. For $0 < \varepsilon \leq \operatorname* { m i n } \{ 1 , G ^ { 2 } / 2 \}$ ， assume $G > 0$ , set

$$
\lambda = \frac { 2 \varepsilon } { G ^ { 2 } } ,\tag{7.37}
$$

and choose

$$
h = c \operatorname* { m i n } \left\{ L _ { f } ^ { - 1 } , \frac { \varepsilon ^ { 2 } } { \tau _ { f } + G ^ { 2 } + A _ { \mathrm { g r p } } } , \frac { \varepsilon \lambda } { d } \right\} .\tag{7.38}
$$

Then $\lambda \leq 1$ and

$$
B _ { \mathrm { r e f } } \leq A _ { \mathrm { g r p } } , \qquad M _ { \lambda } = \frac { d } { \lambda } .\tag{7.39}
$$

If

$$
N \geq \frac { C } { m h } \log \left( 1 + \frac { m \Phi _ { 0 } ( \lambda , h ) } { \varepsilon ^ { 2 } } \right) ,\tag{7.40}
$$

then $\sqrt { m } W _ { 2 } ( \mu _ { N } , \pi ) \le \varepsilon$ . Equivalently, with the step size (7.38), it is suficient to take

$$
N _ { \mathrm { g r p } } ( \varepsilon ) \leq \frac { C } { m } \left[ L _ { f } + \frac { \tau _ { f } + A _ { \mathrm { g r p } } + d G ^ { 2 } } { \varepsilon ^ { 2 } } \right] \log \left( 1 + \frac { m \Phi _ { 0 } ( \lambda , h ) } { \varepsilon ^ { 2 } } \right) .\tag{7.41}
$$

Proof. The penalty is separable across blocks, so the Moreau envelope and the active trace decompose across blocks. The step size (7.38) implies (7.2). Therefore Lemma 6.6 gives a density bound $2 ^ { q _ { b } } B _ { b }$ for $P _ { b } Y _ { t } ^ { \mathrm { r e f } }$ , uniformly in t.

For the center term in Lemma 7.5,

$$
\frac { q _ { b } } { \lambda } \mathbb { P } \{ \| P _ { b } Y _ { t } ^ { \mathrm { r e f } } \| \le \lambda \gamma _ { b } \} \le \frac { q _ { b } } { \lambda } 2 ^ { q _ { b } } B _ { b } v _ { q _ { b } } ( \lambda \gamma _ { b } ) ^ { q _ { b } } = q _ { b } v _ { q _ { b } } 2 ^ { q _ { b } } B _ { b } \gamma _ { b } ^ { q _ { b } } \lambda ^ { q _ { b } - 1 } .
$$

Because $\lambda \leq 1$ , this is at most the first term in $A _ { \mathrm { g r p } }$ . When $q _ { b } = 1$ , the tangential term is zero. When $q _ { b } \geq 2$ , Corollary 6.9 gives

$$
\mathbb { E } \frac { 1 } { \| P _ { b } Y _ { t } ^ { \mathrm { r e f } } \| } \le C _ { q _ { b } } ( 2 ^ { q _ { b } } B _ { b } ) ^ { 1 / q _ { b } } = 2 C _ { q _ { b } } B _ { b } ^ { 1 / q _ { b } } .
$$

Multiplication by $\gamma _ { b } ( q _ { b } - 1 )$ gives the second term in $A _ { \mathrm { g r p } } .$ . These bounds are uniform in $t ,$ so averaging over $[ 0 , h ]$ proves the reference-trace bound. The estimate $M _ { \lambda } = d / \lambda$ is the global trace bound. The choices (7.37) and (7.38) are Theorem 5.2 with $B _ { \mathrm { r e f } } \leq A _ { \mathrm { g r p } }$ and $M _ { \lambda } = d / \lambda$ , up to universal constants. Since $\overset { \cdot } { d } / ( \varepsilon \lambda ) \overset { \cdot } { = } d \overset { \cdot } { G } ^ { 2 } / ( 2 \varepsilon ^ { 2 } )$ , the complexity display follows. □

Remark 7.7 (Block-size dependence in the group-lasso bound). The preceding proposition is meant to make the ε-dependence transparent: it gives a λ-free upper bound on $B _ { \mathrm { r e f } }$ . The displayed constant should not be read as an optimized estimate in the block dimension. Indeed, for a block of size $q _ { b }$ let

$$
\alpha _ { b } : = 1 - h ( L _ { b } + \lambda ^ { - 1 } ) , \qquad { \overline { { B } } } _ { b } : = B _ { b } \alpha _ { b } ^ { - q _ { b } } , \qquad \kappa _ { b } : = ( v _ { q _ { b } } { \overline { { B } } } _ { b } ) ^ { 1 / q _ { b } } .
$$

Before replacing λ and $\alpha _ { b }$ by crude constants, the contribution of this block satisfies

$$
B _ { \mathrm { r e f } , b } \leq q _ { b } \gamma _ { b } \kappa _ { b } ( \lambda \gamma _ { b } \kappa _ { b } ) ^ { q _ { b } - 1 } + \mathbf { 1 } _ { \{ q _ { b } \geq 2 \} } q _ { b } \gamma _ { b } \kappa _ { b } .
$$

The first term is the contribution of the central ball $\{ \| z _ { b } \| \leq \lambda \gamma _ { b } \}$ , while the second term controls the tangential curvature $\gamma _ { b } ( q _ { b } - 1 ) / \vert \vert z _ { b } \vert \vert$ . Thus, for $q _ { b } \geq 2$ , whenever $\lambda \gamma _ { b } \kappa _ { b } \leq 1$ , the central ball term is no larger than the tangential term, and it is in fact damped by the factor $( \lambda \gamma _ { b } \kappa _ { b } ) ^ { q _ { b } - 1 }$ . The simplified λ-free constant used in the proposition deliberately discards this damping in order to emphasize the ε<sup>−2</sup> $\varepsilon ^ { - 2 }$ complexity.

The quantity $\kappa _ { b }$ is an inverse efective block radius, not an exponentially large density constant. From the definition of $B _ { b }$ , a ball lower bound on the normalizing integral gives

$$
\kappa _ { b } \leq C \alpha _ { b } ^ { - 1 } \left( \sqrt { \frac { L _ { b } } { q _ { b } } } + \frac { \gamma _ { b } } { q _ { b } } \right) ,
$$

with a universal constant C. Consequently the tangential term is at most

$$
C \alpha _ { b } ^ { - 1 } \left( \gamma _ { b } \sqrt { L _ { b } q _ { b } } + \gamma _ { b } ^ { 2 } \right) .
$$

Hence, for bounded $L _ { b }$ and standard group-lasso weights, the active-trace constant grows polynomially, and often linearly or sublinearly, in the block size; it is not exponential in $q _ { b } .$

For the central ball term one may also use a one-dimensional slab bound. For any unit vector u in the block,

$$
\begin{array} { r } { \{ \| Y _ { b , t } ^ { \mathrm { r e f } } \| \le \lambda \gamma _ { b } \} \subseteq \{ | \langle u , Y _ { b , t } ^ { \mathrm { r e f } } \rangle | \le \lambda \gamma _ { b } \} , } \end{array}
$$

so the slab estimate gives

$$
\frac { q _ { b } } { \lambda } \mathbb { P } \{ \| Y _ { b , t } ^ { \mathrm { r e f } } \| \le \lambda \gamma _ { b } \} \le \frac { 2 q _ { b } \gamma _ { b } B _ { u } } { \alpha _ { u } } .
$$

Therefore the central contribution can be bounded by the minimum of the ball estimate and this slab estimate. The slab estimate is sometimes a convenient λ-free way to remove the apparent $\gamma _ { b } ^ { q _ { b } }$ factor. It does not, however, control the tangential term: the inverse-radius estimate still needs the q -dimensional small-ball bound when $q _ { b } \geq 2$

## 7.4 Generalized lasso and anisotropic total variation

Let $D \in \mathbb { R } ^ { J \times d }$ have nonzero rows $d _ { j } ^ { \top }$ , and consider

$$
g ( x ) = \gamma \| D x \| _ { 1 } , \qquad \gamma > 0 .\tag{7.42}
$$

A global Lipschitz constant is

$$
G = \gamma \| D ^ { \top } \| _ { \infty  2 } = \gamma \operatorname* { s u p } _ { \| z \| _ { \infty } \leq 1 } \| D ^ { \top } z \| .\tag{7.43}
$$

For each row define

$$
u _ { j } = \frac { d _ { j } } { \Vert d _ { j } \Vert } ,\tag{7.44}
$$

and

$$
L _ { j } = \operatorname* { s u p } _ { x \in \mathbb { R } ^ { d } } u _ { j } ^ { \top } \nabla ^ { 2 } f ( x ) u _ { j } , \qquad G _ { j } = \gamma \| D u _ { j } \| _ { 1 } ,\tag{7.45}
$$

$$
B _ { j } = \left[ \int _ { \mathbb { R } } \exp \left\{ - \frac { L _ { j } } { 2 } s ^ { 2 } - 2 G _ { j } | s | \right\} ~ \mathrm { d } s \right] ^ { - 1 } .\tag{7.46}
$$

The number $G _ { j }$ is the Lipschitz size of $g$ in direction $u _ { j }$ . Finally, set

$$
R _ { j } = \sum _ { \ell = 1 } ^ { J } | d _ { j } ^ { \top } d _ { \ell } | .\tag{7.47}
$$

Lemma 7.8 (Prox-cell trace bound for generalized lasso). Let $p _ { \lambda } ( x ) = \mathrm { p r o x } _ { \lambda g } ( x )$ . For a point $\boldsymbol { p } \in \mathbb { R } ^ { d }$ , define the active row set

$$
A ( p ) = \{ j : \ d _ { j } ^ { \top } p = 0 \} .\tag{7.48}
$$

For $A \subseteq \{ 1 , \dots , J \}$ , let $D _ { A }$ be the submatrix of D with rows indexed by A, and use the convention $P _ { \mathrm { k e r } D _ { A } } = I$ when $A = \emptyset$

A prox-cell means an open polyhedral region of the x-space on which the active set $A ( p _ { \lambda } ( x ) )$ is fixed and, for every inactive row $j \not \in A ( p _ { \lambda } ( x ) )$ , the sign of $d _ { j } ^ { \top } p _ { \lambda } ( x )$ is fixed. At every diferentiability point x of $p _ { \lambda }$ lying in such a cell, with $A = A ( p _ { \lambda } ( x ) )$ ,

$$
D p _ { \lambda } ( x ) = P _ { \mathrm { k e r } D _ { A } } , \qquad a _ { \lambda } ( x ) = { \frac { \mathrm { r a n k } ( D _ { A } ) } { \lambda } } .\tag{7.49}
$$

Consequently, for a.e. x,

$$
a _ { \lambda } ( \boldsymbol x ) \leq \frac { 1 } { \lambda } \sum _ { j = 1 } ^ { J } \mathbf 1 _ { \{ | d _ { j } ^ { \top } \boldsymbol x | \leq \lambda \gamma R _ { j } \} } .\tag{7.50}
$$

In particular, we can choose

$$
M _ { \lambda } = \frac { \mathrm { r a n k } ( D ) } { \lambda } .\tag{7.51}
$$

Proof. We first explain the cell formula. Fix a prox-cell and write its active set as A. On this cell, the signs

$$
\sigma _ { j } = \mathrm { s i g n } ( d _ { j } ^ { \top } p _ { \lambda } ( x ) ) , \qquad j \notin { \cal A } ,
$$

are fixed, while the active rows satisfy $D _ { A } p _ { \lambda } ( x ) = 0$ . For x in the cell, write $p = p _ { \lambda } ( x )$ . Since $d _ { j } ^ { \top } p \neq 0$ for every $j \not \in A$ , there is a relative neighborhood of $p$ in ker $D _ { A }$ on which

$$
\mathrm { s i g n } ( d _ { j } ^ { \top } y ) = \sigma _ { j } , \qquad j \notin A .
$$

On this neighborhood, the original proximal objective restricted to ker $D _ { A }$ agrees with

$$
y \longmapsto \frac { 1 } { 2 \lambda } \| x - y \| ^ { 2 } + \gamma \sum _ { j \notin { \cal A } } \sigma _ { j } d _ { j } ^ { \top } y .
$$

Hence $p$ is a local minimizer of this function on ker $D _ { A }$ . Since the function is strongly convex on ker $D _ { A }$ , this local minimizer is its unique global minimizer. Completing the square therefore gives

$$
\begin{array} { r l r } {  { p _ { \lambda } ( \boldsymbol x ) = \arg \operatorname* { m i n } _ { \boldsymbol y : D _ { A } \boldsymbol y = 0 } \{ \frac { 1 } { 2 \lambda } \| \boldsymbol x - \boldsymbol y \| ^ { 2 } + \gamma \sum _ { \boldsymbol j \not \in A } \sigma _ { \boldsymbol j } d _ { \boldsymbol j } ^ { \top } \boldsymbol y \} } } \\ & { } & \\ & { } & { = P _ { \mathrm { k e r } D _ { A } } ( \boldsymbol x - \lambda \gamma \sum _ { \boldsymbol j \not \in A } \sigma _ { \boldsymbol j } d _ { \boldsymbol j } ) . } \end{array}\tag{7.52}
$$

Since A and the signs $\sigma _ { j }$ are fixed on the cell, the right-hand side is afine in x. Hence

$$
D p _ { \lambda } ( x ) = P _ { \mathrm { k e r } D _ { A } }
$$

throughout the cell. By Lemma 3.1,

$$
H _ { \lambda } ( x ) = \lambda ^ { - 1 } \big ( I - D p _ { \lambda } ( x ) \big )
$$

at every diferentiability point of $p _ { \lambda }$ . Since $P _ { \mathrm { k e r } D _ { A } }$ is the orthogonal projection onto ker $D _ { A }$ the matrix $I - P _ { \mathrm { k e r } D _ { A } }$ is the orthogonal projection onto (ker $D _ { A } ) ^ { \perp } = \mathrm { r a n g e } ( D _ { A } ^ { \top } )$ , whose trace is rank $( D _ { A } )$ . Therefore

$$
a _ { \lambda } ( x ) = \mathrm { t r } H _ { \lambda } ( x ) = \frac { 1 } { \lambda } \mathrm { t r } ( I - P _ { \mathrm { k e r } D _ { A } } ) = \frac { \mathrm { r a n k } ( D _ { A } ) } { \lambda } .
$$

It remains to locate the points where this curvature can occur. The optimality condition for the original proximal problem gives

$$
x - p _ { \lambda } ( x ) = \lambda \gamma D ^ { \top } z , \qquad z _ { j } \in \partial | ( D p _ { \lambda } ( x ) ) _ { j } | , \qquad | z _ { j } | \le 1 .\tag{7.53}
$$

I ${ \bf \dot { \theta } } _ { j } \in { \cal A } ( p _ { \lambda } ( x ) ) ,$ ), then $d _ { j } ^ { \top } p _ { \lambda } ( x ) = 0$ . Multiplying (7.53) by $d _ { j } ^ { \top }$ gives

$$
| d _ { j } ^ { \top } x | = \lambda \gamma | d _ { j } ^ { \top } D ^ { \top } z | \leq \lambda \gamma \sum _ { \ell = 1 } ^ { J } | d _ { j } ^ { \top } d _ { \ell } | | z _ { \ell } | \leq \lambda \gamma R _ { j } .
$$

Thus every active row $j$ forces x to lie in the row slab

$$
\{ | d _ { j } ^ { \top } x | \leq \lambda \gamma R _ { j } \} .
$$

At diferentiability points in the above cells,

$$
a _ { \lambda } ( \boldsymbol x ) = \frac { \mathrm { r a n k } ( D _ { A ( p _ { \lambda } ( \boldsymbol x ) ) } ) } { \lambda } \leq \frac { | A ( p _ { \lambda } ( \boldsymbol x ) ) | } { \lambda } \leq \frac { 1 } { \lambda } \sum _ { j = 1 } ^ { J } \mathbf 1 _ { \{ | \boldsymbol d _ { j } ^ { \top } \boldsymbol x | \leq \lambda \gamma R _ { j } \} } .
$$

Since $g ( x ) = \gamma \| D x \|$ <sub>1</sub> is polyhedral convex, its proximal map $p _ { \lambda }$ is piecewise afine; see, $\mathrm { { e . g . , } \ [ 2 4 }$ Proposition 12.30]. It follows that, outside a Lebesgue-null set, the active set and all inactive signs are locally constant. Hence almost every x lies in a prox-cell, and the preceding inequality holds for Lebesgue-a.e. x. This proves (7.50). Finally,

$$
a _ { \lambda } ( x ) = { \frac { \operatorname { r a n k } ( D _ { A ( p _ { \lambda } ( x ) ) } ) } { \lambda } } \leq { \frac { \operatorname { r a n k } ( D ) } { \lambda } }
$$

for a.e. x. This gives (7.51).

Proposition 7.9 (Generalized lasso reference trace). Let

$$
A _ { D } : = 4 \gamma \sum _ { j = 1 } ^ { J } \frac { R _ { j } B _ { j } } { \| d _ { j } \| } .\tag{7.54}
$$

There exist universal constants $c , C > 0$ such that the following holds. For $0 < \varepsilon \le 1$ , assume $G > 0$ set

$$
\lambda = \frac { 2 \varepsilon } { G ^ { 2 } } ,\tag{7.55}
$$

and choose

$$
h = c \operatorname* { m i n } \left\{ L _ { f } ^ { - 1 } , \frac { \varepsilon ^ { 2 } } { \tau _ { f } + G ^ { 2 } + A _ { D } } , \frac { \varepsilon \lambda } { \mathrm { r a n k } ( D ) } \right\} .\tag{7.56}
$$

Then

$$
B _ { \mathrm { r e f } } \leq A _ { D } , \qquad M _ { \lambda } = \frac { \mathrm { r a n k } ( D ) } { \lambda } .\tag{7.57}
$$

If

$$
N \geq \frac { C } { m h } \log \left( 1 + \frac { m \Phi _ { 0 } ( \lambda , h ) } { \varepsilon ^ { 2 } } \right) ,\tag{7.58}
$$

then $\sqrt { m } W _ { 2 } ( \mu _ { N } , \pi ) \le \varepsilon$ . Equivalently, with the step size (7.56), it is suficient to take

$$
N _ { D } ( \varepsilon ) \leq \frac { C } { m } \left[ L _ { f } + \frac { \tau _ { f } + A _ { D } + \mathrm { r a n k } ( D ) G ^ { 2 } } { \varepsilon ^ { 2 } } \right] \log \left( 1 + \frac { m \Phi _ { 0 } ( \lambda , h ) } { \varepsilon ^ { 2 } } \right) .\tag{7.59}
$$

Proof. By Lemma 7.8,

$$
B _ { \mathrm { r e f } } \leq \frac { 1 } { \lambda h } \sum _ { j = 1 } ^ { J } \int _ { 0 } ^ { h } \mathbb { P } \{ | d _ { j } ^ { \top } Y _ { t } ^ { \mathrm { r e f } } | \leq \lambda \gamma R _ { j } \} ~ \mathrm { d } t .
$$

The step size (7.56) implies (7.2). By Corollary 6.7,

$$
\mathbb { P } \{ | d _ { j } ^ { \top } Y _ { t } ^ { \mathrm { r e f } } | \leq \lambda \gamma R _ { j } \} \leq \frac { 4 \lambda \gamma R _ { j } B _ { j } } { \| d _ { j } \| } ,
$$

uniformly in t. Substitution proves $B _ { \mathrm { r e f } } \leq A _ { D }$ . The bound on $M _ { \lambda }$ is (7.51). The choices (7.55) and (7.56) are Theorem 5.2 with $B _ { \mathrm { r e f } } \leq A _ { D }$ and $M _ { \lambda } = \mathrm { r a n k } ( D ) / \lambda$ , up to universal constants. Since rank $( D ) / ( \varepsilon \lambda ) = \mathrm { r a n k } ( D ) G ^ { 2 } / ( 2 \varepsilon ^ { 2 } )$ , the complexity display follows. □

Corollary 7.10 (One-dimensional anisotropic total variation). Let $D \in \mathbb { R } ^ { ( d - 1 ) \times d }$ be the firstdiference matrix,

$$
d _ { j } = e _ { j } - e _ { j + 1 } , \qquad j = 1 , \ldots , d - 1 .\tag{7.60}
$$

Then $\| d _ { j } \| = { \sqrt { 2 } }$ , ran $\operatorname { k } ( D ) = d - 1$ , and $R _ { j } \leq 4$ for every row. If $B _ { j } \leq B _ { \mathrm { t v } }$ for every $j$ , set

$$
A _ { \mathrm { t v } } : = 8 \sqrt { 2 } \gamma B _ { \mathrm { t v } } ( d - 1 ) .\tag{7.61}
$$

There exist universal constants $c , C > 0$ such that, for $0 < \varepsilon \le 1$ and $G > 0$ , set

$$
\lambda = \frac { 2 \varepsilon } { G ^ { 2 } } ,\tag{7.62}
$$

and choose

$$
h = c \operatorname * { m i n } \left\{ L _ { f } ^ { - 1 } , \frac { \varepsilon ^ { 2 } } { \tau _ { f } + G ^ { 2 } + A _ { \mathrm { t v } } } , \frac { \varepsilon \lambda } { d - 1 } \right\} .\tag{7.63}
$$

Then

$$
B _ { \mathrm { r e f } } \leq A _ { \mathrm { t v } } , \qquad M _ { \lambda } = \frac { d - 1 } { \lambda } .\tag{7.64}
$$

If

$$
N \geq \frac { C } { m h } \log \left( 1 + \frac { m \Phi _ { 0 } ( \lambda , h ) } { \varepsilon ^ { 2 } } \right) ,\tag{7.65}
$$

then $\sqrt { m } W _ { 2 } ( \mu _ { N } , \pi ) \le \varepsilon$ . In particular,

$$
N _ { \mathrm { t v } } ( \varepsilon ) \leq \frac { C } { m } \left[ L _ { f } + \frac { \tau _ { f } + A _ { \mathrm { t v } } + d G ^ { 2 } } { \varepsilon ^ { 2 } } \right] \log \left( 1 + \frac { m \Phi _ { 0 } ( \lambda , h ) } { \varepsilon ^ { 2 } } \right) .\tag{7.66}
$$

Proof. For first-diference rows, $\| e _ { j } - e _ { j + 1 } \| = \sqrt { 2 }$ . Also $d _ { j } ^ { \top } d _ { j } = 2$ , and the only nonzero crossproducts with other rows are the neighboring ones, equal to −1 when they exist. Thus $R _ { j } \ \leq$ $2 + 1 + 1 = 4$ . Applying (7.54) and using $\begin{array} { r } { \sum _ { j = 1 } ^ { d - 1 } B _ { j } \le B _ { \mathrm { t v } } ( d - 1 ) } \end{array}$ gives $A _ { \mathrm { t v } }$ . The step-size and iteration statements are the specialization of Proposition 7.9 with $\mathrm { r a n k } ( D ) = d - 1$ □

Remark 7.11 (Improved dependence on the target accuracy). The main gain in the examples above is that $B _ { \mathrm { r e f } }$ is bounded independently of λ. If one instead used only the global bound $B _ { \mathrm { r e f } } \leq M _ { \lambda } = O ( \lambda ^ { - 1 } )$ , then the term $B _ { \mathrm { r e f } } / \epsilon ^ { 2 }$ in Theorem 5.2 would scale as $\epsilon ^ { - 3 }$ under the choice $\lambda = 2 \epsilon / G ^ { 2 }$ . The structured active-trace estimates remove this cubic contribution. The remaining term $M _ { \lambda } / \epsilon$ scales as $\epsilon ^ { - 2 }$ , yielding the $\widetilde O ( \epsilon ^ { - 2 } )$ dependence displayed above.

## 8 Conclusion

We developed an active-trace analysis of the classical fixed-λ MYULA kernel for strongly log-concave composite targets. The main bound replaces global Moreau-curvature control by an occupationweighted trace along a reference heat path. Combined with the Moreau-bias estimate, this gives end-to-end Wasserstein guarantees for the original nonsmooth target and, for the structured penalties considered here, $\widetilde { O } ( \varepsilon ^ { - 2 } )$ accuracy dependence without changing the MYULA transition. The same occupation-weighted viewpoint may be useful for other smoothing-based Langevin methods.

## A Real-Analysis Primer for A.e. and Weak Hessians

This appendix fixes the real-analysis conventions underlying the nonsmooth second-order notation used in the main text. We recall only the standard facts needed to interpret the Hessian of a $C ^ { 1 , 1 }$ function as an a.e. derivative, equivalently as an $L ^ { \infty }$ weak Hessian, and to use the corresponding symmetry, positivity, and matrix bounds. Throughout the paper, Hessians of Moreau envelopes are understood in this a.e./weak sense; no everywhere classical $C ^ { 2 }$ regularity is assumed. Standard references for these facts include [9, 16].

## A.1 Rademacher’s theorem

A map $F : \mathbb { R } ^ { d _ { 1 } }  \mathbb { R } ^ { d _ { 2 } }$ is locally Lipschitz continuous if for each compact $K \subset \mathbb { R } ^ { d }$ there is $L _ { K } < \infty$ such that

$$
\begin{array} { r } { \| \boldsymbol { F } ( \boldsymbol { x } ) - \boldsymbol { F } ( \boldsymbol { y } ) \| \le L _ { K } \left\| \boldsymbol { x } - \boldsymbol { y } \right\| , \qquad \boldsymbol { x } , \boldsymbol { y } \in K . } \end{array}
$$

Rademacher’s theorem states that every locally Lipschitz continuous map between Euclidean spaces is diferentiable Lebesgue-a.e [9, Theorem 3.2]. In this paper it is used with $F = \nabla V$ , where $V \in C ^ { 1 , 1 }$ . Thus, although a Lipschitz gradient need not be diferentiable everywhere, its Jacobian exists outside a null set.

## A.2 A.e. Hessians, weak derivatives, and the $L ^ { \infty }$ weak Hessian

Let $V \in C ^ { 1 , 1 } (  { \mathbb { R } } ^ { d } )$ . By definition, $V \in C ^ { 1 }$ and ∇V is Lipschitz. At a point where the Lipschitz map ∇V is diferentiable, we define the classical a.e. Hessian by

$$
\nabla ^ { 2 } V ( x ) : = D ( \nabla V ) ( x ) .
$$

Rademacher’s theorem implies that this definition applies for a.e. x.

There is an equivalent weak-derivative interpretation. For a locally integrable function u, a function w is the weak derivative $\partial _ { j } u$ if

$$
\int _ { \mathbb { R } ^ { d } } u ( x ) \partial _ { j } \varphi ( x ) \mathrm { d } x = - \int _ { \mathbb { R } ^ { d } } w ( x ) \varphi ( x ) \mathrm { d } x
$$

for every smooth compactly supported test function $\varphi .$ When $u = \partial _ { i } V$ , these weak derivatives form a matrix $H _ { V } = \left( H _ { i j } \right)$ . This matrix is called the weak Hessian of V.

Since ∇V is Lipschitz, its a.e. Jacobian is bounded by $\operatorname { L i p } ( \nabla V )$ , and the a.e. Hessian belongs to $L ^ { \infty }$ . The weak Hessian and the a.e. Jacobian $D ( \nabla V )$ agree up to null sets, since the Lipschitz gradient is absolutely continuous on every line, allowing one-dimensional integration by parts. Therefore, in the paper, the phrases “a.e. Hessian” and ${ } ^ { 6 6 } L ^ { \infty }$ weak Hessian” refer to the same matrix-valued object, viewed from two equivalent perspectives.

## A.3 Symmetry, convexity, and matrix bounds

For a $C ^ { 2 }$ function, the Hessian is symmetric because mixed partial derivatives commute. For $V \in C ^ { 1 , 1 }$ , the same statement holds in the weak sense:

$$
H _ { i j } = H _ { j i } \mathrm { ~ a . e . }
$$

One way to see this is to mollify. Let $\rho _ { \varepsilon }$ be a smooth mollifier and set $V _ { \varepsilon } = \rho _ { \varepsilon } * V$ . Then $V _ { \varepsilon } \in C ^ { \infty }$ so $\nabla ^ { 2 } V _ { \varepsilon }$ is symmetric. At every point where the Lipschitz map ∇V is diferentiable, mollification recovers the derivative:

$$
\nabla ^ { 2 } V _ { \varepsilon } ( x ) \to D ( \nabla V ) ( x ) .
$$

Indeed, near such a point, ∇V is well approximated by its linear part, and the mollifier averages over a ball whose radius tends to zero. Since ∇V is diferentiable a.e., the a.e. Hessian is the pointwise a.e. limit of symmetric matrices. Therefore $H _ { i j } = H _ { j i }$ a.e.

If V is convex, then the gradient is monotone:

$$
\langle \nabla V ( x ) - \nabla V ( y ) , x - y \rangle \geq 0 .
$$

At a point where ∇V is diferentiable, take $y = x + t v$ , divide by $t ^ { 2 }$ , and let $t \to 0$ . This gives

$$
v ^ { \top } \nabla ^ { 2 } V ( x ) v \geq 0 ,
$$

so the a.e. Hessian is positive semidefinite.

If, in addition, $\mathrm { L i p } ( \boldsymbol { \nabla } V ) \leq L$ , then at every diferentiability point, for every unit vector $v ,$

$$
\| D ( \nabla V ) ( x ) v \| = \operatorname* { l i m } _ { t \to 0 } \left\| { \frac { \nabla V ( x + t v ) - \nabla V ( x ) } { t } } \right\| \leq L .
$$

Since the Hessian is symmetric and positive semidefinite, all its eigenvalues lie in $[ 0 , L ]$ . Equivalently,

$$
0 \preceq \nabla ^ { 2 } V ( x ) \preceq L I _ { d } \qquad { \mathrm { f o r ~ a . e . ~ } } x .
$$

## B Details for Moreau Weak Second-Order Regularity

Justification of Lemma 3.1. We first quote the standard first-order Moreau–Yosida facts. For a proper, lower semicontinuous, convex function g, the proximal map $p _ { \lambda }$ is single-valued on $\mathbb { R } ^ { d }$ , and the Moreau envelope $g _ { \lambda }$ is finite-valued, convex, and belongs to $C ^ { 1 , 1 } ( \mathbb { R } ^ { d } )$ . Moreover,

$$
\nabla g _ { \lambda } ( x ) = \frac { x - p _ { \lambda } ( x ) } { \lambda } , \qquad \mathrm { L i p } ( \nabla g _ { \lambda } ) \leq \lambda ^ { - 1 } .
$$

See, for example, [1, Chs. 12 and 23] and [24, Ch. 1.G].

It remains to interpret the second-order statements. Since $g _ { \lambda }$ is convex and $C ^ { 1 , 1 }$ , the general facts reviewed in Section A apply with $V = g _ { \lambda }$ and $L = \lambda ^ { - 1 }$ . Hence there exists a set $E _ { \lambda } \subset \mathbb { R } ^ { d }$ whose complement is Lebesgue-null, such that $\nabla g _ { \lambda }$ is diferentiable at every $x \in E _ { \lambda }$ , and, with

$$
\nabla ^ { 2 } g _ { \lambda } ( x ) : = D ( \nabla g _ { \lambda } ) ( x ) ,
$$

the matrix $\nabla ^ { 2 } g _ { \lambda } ( x )$ is symmetric and satisfies

$$
0 \preceq \nabla ^ { 2 } g _ { \lambda } ( x ) \preceq \lambda ^ { - 1 } I , \qquad x \in E _ { \lambda } .
$$

Finally, the gradient formula gives

$$
p _ { \lambda } = \mathrm { I d } - \lambda \nabla g _ { \lambda } .
$$

Therefore $p _ { \lambda }$ is diferentiable at every $x \in E _ { \lambda }$ , and

$$
D p _ { \lambda } ( x ) = I - \lambda \nabla ^ { 2 } g _ { \lambda } ( x ) .
$$

Equivalently,

$$
\nabla ^ { 2 } g _ { \lambda } ( x ) = \lambda ^ { - 1 } \{ I - D p _ { \lambda } ( x ) \} .
$$

Proof of Lemma 3.3. Let $p = p _ { \lambda } ( x )$ . The proximal optimality condition gives $\nabla g _ { \lambda } ( x ) = ( x - p ) / \lambda \in$ $\partial g ( p )$ . It is therefore enough to show that every subgradient of $g$ has norm at most G. If $v \in \partial g ( p )$ and u is a unit vector, the subgradient inequality and Lipschitzness imply, for $t > 0$

$$
g ( p ) + t \left. v , u \right. \leq g ( p + t u ) \leq g ( p ) + G t .
$$

Hence $\langle v , u \rangle \leq G$ . Taking $u = v / \| v \|$ when $v \neq 0$ yields $\| v \| \leq G$ . Therefore $\| \nabla g _ { \lambda } ( x ) \| \leq G$ □

## C Proof of the Heat Identity for $C ^ { 1 , 1 }$ Test Functions

Proof of Lemma 3.4. We use the Itˆo–Krylov formula, i.e. Itˆo’s formula with generalized second derivatives; see [14, Ch. 2, Sec. 10, Thm. 1].

Let $L = \mathrm { L i p } ( \nabla V )$ . Since ∇V is globally L-Lipschitz, the weak Hessian satisfies

$$
\begin{array} { r } { \| H _ { V } ( x ) \| _ { \mathrm { o p } } \leq L \qquad \mathrm { f o r ~ a . e . ~ } x , } \end{array}
$$

and hence

$$
| \Delta V ( x ) | \leq d L \qquad { \mathrm { f o r ~ a . e . ~ } } x .
$$

Moreover,

$$
\| \nabla V ( x ) \| \leq \| \nabla V ( 0 ) \| + L \| x \| ,
$$

and the fundamental theorem of calculus along the segment from 0 to x gives

$$
| V ( x ) | \leq C _ { V } ( 1 + \| x \| ^ { 2 } )
$$

for some constant $C _ { V } < \infty$ . Thus $V ( Y _ { t } )$ is integrable for $t \in [ 0 , h ]$ , since $Y _ { 0 } \in L ^ { 2 }$

For $R > 0$ , let

$$
\tau _ { R } : = \operatorname* { i n f } \{ t \geq 0 : \| Y _ { t } \| \geq R \} .
$$

On the ball $B _ { R } .$ the function V satisfies the hypotheses of the Itˆo–Krylov formula: its first derivatives are continuous and its generalized second derivatives are locally square-integrable. Applying the formula to the stopped process $Y _ { t \wedge \tau _ { R } }$ gives

$$
V ( Y _ { h \wedge \tau _ { R } } ) - V ( Y _ { 0 } ) = \sqrt { 2 } \int _ { 0 } ^ { h \wedge \tau _ { R } } \langle \nabla V ( Y _ { t } ) , d W _ { t } \rangle + \int _ { 0 } ^ { h \wedge \tau _ { R } } \Delta V ( Y _ { t } ) d t .
$$

The stochastic integral has mean zero, since the integrand is bounded on $[ 0 , \tau _ { R } ]$ . Therefore

$$
\mathbb { E } V ( Y _ { h \wedge \tau _ { R } } ) - \mathbb { E } V ( Y _ { 0 } ) = \mathbb { E } \int _ { 0 } ^ { h \wedge \tau _ { R } } \Delta V ( Y _ { t } ) d t .
$$

Letting $R \to \infty$ , we have $\tau _ { R }  \infty \ \mathrm { a . s }$ . The left-hand side converges to $\mathbb { E } V ( Y _ { h } ) - \mathbb { E } V ( Y _ { 0 } )$ by dominated convergence, using the quadratic growth of V and

$$
\mathbb { E } \operatorname* { s u p } _ { 0 \leq t \leq h } \| Y _ { t } \| ^ { 2 } < \infty .
$$

The right-hand side converges to

$$
\int _ { 0 } ^ { h } \mathbb { E } \Delta V ( Y _ { t } ) d t
$$

by dominated convergence, since $\vert \Delta V \vert \leq d L \mathrm { a . e }$ . This proves the identity.

## References

[1] Heinz H Bauschke and Patrick L Combettes. Correction to: convex analysis and monotone operator theory in hilbert spaces. In Convex analysis and monotone operator theory in Hilbert spaces, pages C1–C4. Springer, 2020.

[2] Nicolas Brosse, Alain Durmus, Eric Moulines, and Marcelo Pereyra. Sampling from a log-<sup>´</sup> concave distribution with compact support with proximal langevin monte carlo. In Conference on learning theory, pages 319–342. PMLR, 2017.

[3] Arnak S Dalalyan and Avetik Karagulyan. Improved guarantees for langevin monte carlo with average smoothness. arXiv preprint arXiv:2605.31413, 2026.

[4] Alain Durmus, Szymon Majewski, and B la˙zej Miasojedow. Analysis of langevin monte carlo via convex optimization. Journal of Machine Learning Research, 20(73):1–46, 2019.

[5] Alain Durmus and Eric Moulines. Nonasymptotic convergence analysis for the unadjusted langevin algorithm. 2017.

[6] Alain Durmus, Eric Moulines, and Marcelo Pereyra. Eficient bayesian computation by proximal markov chain monte carlo: when langevin meets moreau. SIAM Journal on Imaging Sciences, 11(1):473–506, 2018.

[7] Armin Eftekhari, Luis Vargas, and Konstantinos C Zygalakis. The forward–backward envelope for sampling with the overdamped langevin algorithm. Statistics and Computing, 33(4):85, 2023.

[8] Matthias J Ehrhardt, Lorenz Kuger, and Carola-Bibiane Sch¨onlieb. Proximal langevin sampling with inexact proximal mapping. SIAM Journal on Imaging Sciences, 17(3):1729–1760, 2024.

[9] Lawrence C Evans. Measure theory and fine properties of functions. Chapman and Hall/CRC, 2025.

[10] Susan Ghaderi, Masoud Ahookhosh, Adam Arany, Alexander Skupin, Panagiotis Patrinos, and Yves Moreau. Smoothing unadjusted langevin algorithms for nonsmooth composite potential functions. Applied Mathematics and Computation, 464:128377, 2024.

[11] Andreas Habring, Alexander Falk, Martin Zach, and Thomas Pock. Difusion at absolute zero: Langevin sampling using successive moreau envelopes. SIAM Journal on Imaging Sciences, 19(1):35–77, 2026.

[12] Andreas Habring, Martin Holler, and Thomas Pock. Subgradient langevin methods for sampling from nonsmooth potentials. SIAM Journal on Mathematics of Data Science, 6(4):897–925, 2024.

[13] Teresa Klatzer, Paul Dobson, Yoann Altmann, Marcelo Pereyra, Jesus Maria Sanz-Serna, and Konstantinos C Zygalakis. Accelerated bayesian imaging by relaxed proximal-point langevin sampling. SIAM Journal on Imaging Sciences, 17(2):1078–1117, 2024.

[14] Nicolai V Krylov. Controlled difusion processes. Springer, 1980.

[15] Tim Tsz-Kit Lau and Han Liu. Bregman proximal langevin monte carlo via bregman-moreau envelopes. In International Conference on Machine Learning, pages 12049–12077. PMLR, 2022.

[16] Giovanni Leoni. A first course in Sobolev spaces. American Mathematical Soc., 2017.

[17] Adrian S Lewis. Active sets, nonsmoothness, and sensitivity. SIAM Journal on Optimization, 13(3):702–725, 2002.

[18] Jingwei Liang, Jalal Fadili, and Gabriel Peyr´e. Activity identification and local linear convergence of forward–backward-type methods. SIAM Journal on Optimization, 27(1):408–437, 2017.

[19] Linghai Liu and Sinho Chewi. A proximal gradient algorithm for composite log-concave sampling. arXiv preprint arXiv:2605.12461, 2026.

[20] Wenlong Mou, Nicolas Flammarion, Martin J Wainwright, and Peter L Bartlett. An eficient sampling algorithm for non-smooth composite potentials. Journal of Machine Learning Research, 23(233):1–50, 2022.

[21] Felix Otto and C´edric Villani. Generalization of an inequality by talagrand and links with the logarithmic sobolev inequality. Journal of Functional Analysis, 173(2):361–400, 2000.

[22] Marcelo Pereyra. Proximal markov chain monte carlo algorithms. Statistics and Computing, 26(4):745–760, 2016.

[23] Marcelo Pereyra, Luis Vargas Mieles, and Konstantinos C Zygalakis. Accelerating proximal markov chain monte carlo by using an explicit stabilized method. SIAM Journal on Imaging Sciences, 13(2):905–935, 2020.

[24] R Tyrrell Rockafellar and Roger JB Wets. Variational analysis. Springer, 1998.

[25] Adil Salim, Dmitry Kovalev, and Peter Richt´arik. Stochastic proximal langevin algorithm: Potential splitting and nonasymptotic rates. Advances in Neural Information Processing Systems, 32, 2019.

[26] Adil Salim and Peter Richtarik. Primal dual interpretation of the proximal stochastic gradient langevin algorithm. Advances in Neural Information Processing Systems, 33:3786–3796, 2020.

[27] Ryan J Tibshirani and Jonathan Taylor. Degrees of freedom in lasso problems. 2012.

[28] Samuel Vaiter, Charles Deledalle, Jalal Fadili, Gabriel Peyr´e, and Charles Dossal. The degrees of freedom of partly smooth regularizers. Annals of the Institute of Statistical Mathematics, 69(4):791–832, 2017.

[29] C´edric Villani et al. Optimal transport: old and new, volume 338. Springer, 2009.