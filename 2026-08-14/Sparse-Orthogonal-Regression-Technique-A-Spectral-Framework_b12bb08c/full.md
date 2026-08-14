# Sparse Orthogonal Regression Technique: A Spectral Framework for Equation Discovery, Approximation, and Integration

Sabin Roman<sup>1</sup>, Ljupčo Todorovski<sup>2,1</sup>, and Sašo Džeroski<sup>1</sup>

<sup>1</sup> Department of Knowledge Technologies, Jožef Stefan Institute sabin.roman@ijs.si

<sup>2</sup> Faculty of Mathematics and Physics, University of Ljubljana

Abstract. We develop the Sparse Orthogonal Regression Technique (SORT), a sparse spectral framework for learning orthonormal-basis expansions from noisy and irregularly sampled data. SORT estimates expansion coeficients directly from observations using L<sup>1</sup>-regularized regression, avoiding explicit quadrature or analytic inner-product evaluation. The central application is data-driven discovery of ordinary diferential equations: vector fields are represented in chosen orthogonal bases and learned as sparse coeficient expansions. This provides a complementary route to symbolic regression, grammar-based discovery, and SINDystyle sparse identification by first recovering a compact spectral representation, which can later guide searches for simpler analytic forms. Across the dynamical-system experiments, SORT matches or improves upon library-based sparse-regression baselines when the basis is well adapted to the problem, and shows more stable degradation under sparse sampling, noisy derivative estimates, and representation mismatch. Specific examples illustrate why this representation is useful: if a finite library misses the problem-specific nonlinearity, the resulting model can fail. SORT is not immune to mismatch, but it shifts the problem away from brittle selection among generic terms to basis design adapted to the problem domain. The experiments also show that dominant low-order coeficients persist as model order increases, supporting order-consistent model growth. Beyond equation discovery, the same learned expansion supports nonlinear approximation and estimation of complex, high-dimensional integrals by coeficient readout. Overall, SORT provides a reusable intermediate representation for system identification, approximation, and integration, while making basis design an explicit part of the scientific modeling problem.

## 1 Introduction

Many problems in scientific computing and machine learning require learning an unknown function from data and reusing that representation for downstream tasks. Examples include surrogate modelling for expensive simulations [28,2,8], numerical integration from sampled observations [3,31], and data-driven modelling or discovery of governing equations from time-series data [5,26,23,24].

Although these tasks are often treated separately, they share a common requirement: reconstructing a functional relationship from finite, noisy, and often irregularly sampled data in a stable and reusable form.

This paper develops the Sparse Orthogonal Regression Technique (SORT) as a practical framework for estimating sparse expansions in prescribed orthonormal bases from sampled observations. SORT combines two standard ingredients: orthonormal basis representations and sparsity-promoting regression. We therefore do not claim algorithmic novelty in sparse regression on orthogonal features. The contribution is application-level and representational: SORT treats the learned sparse orthonormal expansion as a reusable object for data-driven dynamical-system reconstruction, numerical integration by coeficient readout, and nonlinear function approximation with order-consistent model growth.

The central application is dynamical-system reconstruction. Sparse identification methods such as SINDy [5,26] learn governing equations by regressing derivatives onto finite libraries of candidate nonlinearities. This is efective when the correct terms are present in the library, but recovery can become ill-conditioned or brittle when the true dynamics are not sparse in the chosen dictionary [10]. SORT provides a complementary route: each component of the vector field is represented in a fixed orthonormal basis and estimated through sparse regression. The output is not necessarily a compact symbolic expression, but a sparse spectral representation of the vector field. This use of orthonormal expansions for equation discovery remains comparatively underexplored relative to library-based sparse identification and symbolic regression [18].

A second application is numerical integration from data. In an orthonormal expansion, coeficients are inner products between the function and basis elements. If a basis direction is chosen to represent an integration functional, then the corresponding coeficient gives the integral, up to a known normalization factor. Classical spectral methods usually obtain these coeficients by quadrature, exact projection, or structured transforms. SORT instead estimates them from scattered or noisy point samples by sparse regression and then evaluates the desired integral by coeficient readout. This gives a direct route from sampled data to integral estimates in settings where explicit quadrature is dificult, high-dimensional, or unavailable.

A third application is order-consistent model growth. In an orthonormal expansion, increasing the truncation order does not change the meaning of lowerorder basis functions. Low-order coeficients describe coarse structure, while higher-order coeficients refine the approximation. In finite, noisy, and irregularly sampled data this hierarchy is only approximate, because empirical orthogonality and coeficient estimates depend on sampling and regularization. Nevertheless, coeficient persistence across increasing truncation orders provides a practical diagnostic of stable learning. This hierarchy is standard in numerical analysis but remains underused in machine-learning workflows, where increasing model complexity often changes the full parameterization rather than refining an existing coeficient structure. In this sense, SORT also provides a natural way to compare learned models ordered by truncation level, sparsity pattern, coeficient persistence, and predictive accuracy within a common representation [25].

We evaluate SORT across dynamical-system identification, numerical integration, and nonlinear approximation tasks. The experiments compare against dense least-squares, kernel-based, and library-based sparse-regression baselines under noise and subsampling. Across these settings, SORT is used not as a new sparse-regression algorithm, but as a sparse spectral workflow: it estimates orthonormal coeficients from data and reuses them for prediction, integration, vector-field reconstruction, and model refinement.

## 2 Background and related work

SORT builds on several established areas: spectral approximation, sparse recovery in orthogonal systems, polynomial chaos expansions, sparse regression, and data-driven equation discovery. These areas share many mathematical ingredients, but they difer in their home fields, target applications, and intended outputs.

Classical spectral approximation represents functions in global orthogonal bases such as Fourier, Chebyshev, or Legendre systems [3]. Its home field is numerical analysis, where the main goals are accurate approximation of smooth functions and eficient evaluation of derivatives, integrals, or diferential operators. Coeficients are typically computed by projection, quadrature, interpolation, or structured transforms. SORT keeps the spectral representation, but estimates coeficients from finite, noisy, and irregularly sampled data using sparse regression.

Sparse recovery in orthogonal polynomial bases has been studied extensively in approximation theory, compressed sensing, and uncertainty quantification. Early work showed that sparse Legendre expansions can be recovered from random samples via $\ell _ { 1 }$ minimization [22], with later extensions to Chebyshev, Hermite, and other orthogonal systems, including analyses of sampling, conditioning, and high-dimensional stability [14,31,11]. This literature provides the mathematical basis for recovering compressible orthogonal expansions from limited samples. SORT uses this recovery principle as a modelling component, but does not propose new sparse-recovery theory.

Sparse Polynomial Chaos Expansion (PCE) methods are particularly close in methodology [19]. PCE represents the output of a model with uncertain inputs as an expansion in orthogonal polynomials matched to the input probability distribution. Sparse PCE combines this representation with sparse regression, adaptive truncation, or weighted penalties to control the growth of high-dimensional polynomial bases [2,8,21,17]. Its main use is surrogate modelling and uncertainty quantification: once the expansion is learned, coeficients are used to compute statistical quantities such as means, variances, sensitivity indices, and other moments under a prescribed input distribution. SORT overlaps with sparse PCE at the level of orthogonal bases and sparse coeficient estimation, but its goal is diferent. It is geared toward dynamical-system reconstruction, direct integral estimation from sampled functions, and order-consistent approximation, rather than primarily toward uncertainty propagation or statistical post-processing.

At the algorithmic level, SORT relies on classical sparsity-promoting regression. Convex $\ell _ { 1 } .$ -regularized regression [30], path-following algorithms [9], and compressed sensing theory [6] provide the tools and recovery intuition. Related orthogonal forward regression methods also use orthogonalization to reduce collinearity and improve sparse model selection [7,1,29,13,16]. These methods typically begin with a finite candidate library and use orthogonalization or regularization to select a stable subset. SORT instead begins with a prescribed orthonormal coordinate system, so increasing the truncation order refines the representation while preserving the meaning of lower-order coeficients.

Data-driven equation discovery provides the closest application-level comparison. SINDy and related methods identify governing equations by sparse regression over nonlinear libraries constructed from the observed variables [5,26]. Symbolic regression searches more broadly over expression trees, grammars, or probabilistic symbolic structures [12,4]. These approaches can recover compact, human-readable equations when the library or grammar matches the true system. SORT targets the same broad scientific-discovery setting but returns a sparse spectral vector-field model rather than a symbolic formula. This can be advantageous when the correct analytic building blocks are unknown, poorly matched by standard libraries, or too expensive to search over directly.

Overall, SORT is best viewed as a task-oriented recombination of established tools rather than as a new optimization method. This is analogous to the way $\mathrm { S I N D y }$ made sparse regression useful for equation discovery by pairing it with nonlinear libraries. SORT instead places sparse orthonormal coeficients at the centre of the workflow: the same representation can define a spectral vector field, evaluate integrals through linear functionals, approximate nonlinear functions, and compare models across truncation orders.

## 2.1 Mathematical notation

Let $( \varOmega , \mu )$ be the domain and measure with respect to which the basis is orthonormal, with inner product $\begin{array} { r } { { \langle f , g \rangle = \int _ { \Omega } f ( x ) g ( x ) d \mu ( x ) } } \end{array}$ . Let $\{ \phi _ { k } \} _ { k \ge 0 }$ be an orthonormal basis, so that $\langle \phi _ { j } , \phi _ { k } \rangle = \delta _ { j k }$ . The spectral coeficients of $f$ are $c _ { k } = \langle f , \phi _ { k } \rangle$ , and a finite truncation over an index set $\varLambda _ { K }$ is

$$
f _ { K } ( x ) = \sum _ { k \in \varLambda _ { K } } c _ { k } \phi _ { k } ( x ) .
$$

This representation is useful when the coeficient sequence decays rapidly. For example, if f is 2π-periodic and of bounded variation $V .$ , then $| \hat { f } _ { k } | \le V / ( 2 \pi | k | )$ for $k \neq 0$ , and the K-mode Fourier approximation satisfies $\| f - S _ { K } f \| _ { L ^ { 2 } } \le V / \sqrt { \pi K }$ [27]. Thus, many smooth or compressible functions can be represented accurately by relatively few coeficients.

In data-driven settings, the inner products $c _ { k } = \langle f , \phi _ { k } \rangle$ are not available. Instead, we observe samples $y _ { i } = f ( x _ { i } ) + \varepsilon _ { i } , i = 1 , \dots , N$ . SORT forms the design matrix $\varPhi _ { i k } = \phi _ { k } ( x _ { i } )$ and estimates the coeficient vector by

$$
\hat { c } \in \arg \operatorname* { m i n } _ { u \in \mathbb { R } ^ { K } } \frac { 1 } { 2 N } \| y - \varPhi u \| _ { 2 } ^ { 2 } + \lambda \| u \| _ { 1 } .
$$

The learned approximation is $\begin{array} { r } { \hat { f } ( x ) = \sum _ { k \in A _ { K } } \hat { c } _ { k } \phi _ { k } ( x ) } \end{array}$ . The regularization parameter λ suppresses weak or noise-supported modes and is selected empirically, for example by validation error or cross-validation, with preference for the sparsest stable model.

In the ideal empirically orthonormal case, if $( 1 / N ) \varPhi ^ { \top } \varPhi = I _ { K }$ and $\| ( 1 / N ) \varPhi ^ { \top } \varepsilon \| _ { \infty } \leq$ λ, standard LASSO arguments imply

$$
\| \hat { c } - c \| _ { \infty } \leq 2 \lambda
$$

[30,15]. This bound is used only as background intuition: with irregular finite samples, empirical orthogonality is approximate, so coeficient stability must be assessed numerically.

For numerical integration, the relevant fact is that integrals are linear functionals. If $\begin{array} { r } { \begin{array} { r } { \mathcal { T } ( f ) = \int _ { \Omega } f ( x ) d \nu ( x ) } \end{array} } \end{array}$ has Riesz representer $r _ { \mathcal { T } } ~ \in ~ L ^ { 2 } ( \varOmega , \mu )$ , then $\mathcal { I } ( f ) = \langle f , r _ { \mathcal { I } } \rangle$ . If the normalized direction $\phi _ { \mathcal { T } } = r _ { \mathcal { T } } / \Vert r _ { \mathcal { T } } \Vert$ is included as a basis element, its coeficient satisfies

$$
c _ { \mathcal { T } } = \langle f , \phi _ { \mathcal { T } } \rangle , \qquad \mathcal { T } ( f ) = \| r _ { \mathcal { T } } \| c _ { \mathcal { T } } .
$$

SORT estimates $c _ { \mathcal { T } }$ from point samples by sparse regression, after which the integral is obtained by coeficient readout. In the simplest case $d \nu = d \mu$ on a finite-measure domain, $r _ { \mathcal { Z } } ( x ) = 1$ , so the integral is proportional to the coeficient of the normalized constant basis function.

For equation discovery, consider an autonomous system ${ \dot { x } } ( t ) = f ( x ( t ) )$ . SORT represents each vector-field component as

$$
f _ { j } ( x ) = \sum _ { k \in \varLambda _ { K } } c _ { j k } \phi _ { k } ( x ) , \qquad j = 1 , \ldots , d ,
$$

and estimates the coeficients from samples of $( x ( t ) , \dot { x } ( t ) )$ . The result is a sparse spectral vector-field model: explicit and reusable, but not necessarily a compact symbolic expression.

Basis choice should reflect the domain and regularity of the target function: Fourier bases for periodic or oscillatory structure, Legendre or Chebyshev bases on bounded intervals, Hermite or Laguerre families on unbounded or semi-infinite domains, and transformed rational bases when mapping physical domains to canonical intervals. In low dimensions, total-degree truncation is often suficient; in higher dimensions, hyperbolic-cross or anisotropic truncation helps control basis growth. For N samples and K basis functions, constructing the design matrix costs $O ( N K )$ , and coordinate-descent LASSO solvers scale with repeated passes over this matrix. The main bottleneck is the growth of K with dimension and truncation order. SORT does not remove the curse of dimensionality, and severe basis mismatch can degrade performance. When compact human-readable formulas are required, symbolic regression or library-based sparse identification may remain preferable.

![](images/4672321b6cccc0c977ca9a2c1e62bf7ba3c9f9d02441ea5207adc9f7d96d42d2.jpg)  
(a) Dense-trajectory RMSE vs sampling interval.

![](images/8ec63de34dbfbaf70a210ca74a5a2c886cba79c463418f54c9fa90e2a02b9df3.jpg)  
(b) Representative rollouts at $\varDelta t _ { \mathrm { s a m p l e } } =$ 0.75.  
Fig. 1: Sampling degradation under finite-diference derivative estimates. Panel (a) reports dense-rollout RMSE as the sampling interval increases; panel (b) shows representative rollouts at $\varDelta t _ { \mathrm { s a m p l e } } = 0 . 7 5$ . Both methods use the same sampled states and derivative estimates. Polynomial systems use degreethree SINDy and degree-three Legendre SORT; Bar Magnets uses Fourier features. The main efect is not better fine-sampling accuracy, but slower and less catastrophic degradation under coarse sampling.

## 3 Applications

We evaluate SORT on data-driven system reconstruction, numerical integration by coeficient readout, and nonlinear approximation, focusing on data-limited regimes where representation afects stability.

## 3.1 Equation discovery from time-series data

We first study autonomous vector-field recovery from sampled trajectories. For each sampling interval, SINDy and SORT are trained on the same states, with derivatives estimated by finite diferences and no smoothing. The comparison therefore isolates the learned representation, not the derivative estimator.

The examples are selected from Dynobench [20], restricted here to polynomial and smooth trigonometric systems. Polynomial cases use a degree-three SINDy library and a matching degree-three Legendre basis; the trigonometric case uses Fourier features. We exclude the coupled phase oscillator because it is non-autonomous and would require time or forcing terms. Bacterial respiration, predator–prey, glider, and shear flow contain rational, singular, or ratiotrigonometric terms such as 1/x or cot(y), which would require task-specific rational, transformed, or mixed bases. Lorenz is excluded for a diferent reason: its standard polynomial form gives a polynomial SINDy library a representational advantage reflecting our usual notation for the equations, not any principle that natural dynamics are polynomial. Thus, basis design is part of the scientific modeling problem, not an implementation detail, as in the use of Bessel functions for problems with radial or cylindrical symmetry. Diferent bases may be universal in principle, but under finite data, noise, and degraded derivative estimates, the coordinate system matters: some representations make the dynamics sparse and learnable, while others make the same system brittle or unnecessarily high-dimensional.

Figure 1 shows that both methods behave similarly when sampling is fine and derivatives are reliable. As sampling becomes coarse, Lotka–Volterra and Van der Pol show abrupt SINDy rollout failure, while the orthogonal sparse model remains bounded and degrades more gradually. Stuart–Landau is a neutral case, and Bar Magnets tests the Fourier setting. Therefore, with a suitable basis, sparse orthogonal regression is competitive with a library-based sparse model, while showing better behavior under degraded derivative information. Chaotic systems also require attractor-level evaluation rather than long-horizon pointwise trajectory error; we illustrate this issue separately below.

We next evaluate robustness to noise and representation mismatch using two three-dimensional cyclic systems. In these experiments, the trajectories are corrupted with additive Gaussian noise and then subsampled. Because the data are noisy, derivatives are estimated by applying a Savitzky–Golay filter to each observed coordinate and diferentiating the smoothed signal. The same smoothing, derivative estimates, noise model, subsampling strategy, rollout horizon, and evaluation metric are used for SORT and the SINDy-style baseline. This second experiment therefore tests a diferent failure mode from the Lotka–Volterra sampling study: not coarse finite-diference degradation alone, but the interaction of noise, smoothing, and basis mismatch.

We consider two three-dimensional cyclic systems. The first is the Thomas attractor, whose dynamics are explicitly trigonometric. The second replaces the trigonometric coupling with a Bessel function $J _ { 1 } ( \cdot )$ :

Thomas attractor

Bessel-driven variant

$$
\begin{array} { r } { \dot { x } _ { 1 } = \sin ( x _ { 2 } ) - b x _ { 1 } , } \\ { \dot { x } _ { 2 } = \sin ( x _ { 3 } ) - b x _ { 2 } , } \\ { \dot { x } _ { 3 } = \sin ( x _ { 1 } ) - b x _ { 3 } . } \end{array}
$$

$$
\begin{array} { r } { \dot { x } _ { 1 } = J _ { 1 } ( x _ { 2 } ) - b x _ { 1 } , } \\ { \dot { x } _ { 2 } = J _ { 1 } ( x _ { 3 } ) - b x _ { 2 } , } \\ { \dot { x } _ { 3 } = J _ { 1 } ( x _ { 1 } ) - b x _ { 3 } . } \end{array}
$$

Here $J _ { 1 } ( x )$ denotes the Bessel function of the first kind of order one,

$$
\begin{array} { r } { J _ { 1 } ( x ) = \sum _ { k = 0 } ^ { \infty } { \frac { ( - 1 ) ^ { k } } { k ! ( k + 1 ) ! } } \left( { \frac { x } { 2 } } \right) ^ { 2 k + 1 } = { \frac { x } { 2 } } - { \frac { x ^ { 3 } } { 1 6 } } + { \frac { x ^ { 5 } } { 3 8 4 } } - \cdot \cdot \cdot . } \end{array}
$$

This system is deliberately chosen to test representation mismatch: $J _ { 1 } ( \cdot )$ is not sparse in low-degree polynomial or trigonometric libraries. The SINDy baseline uses either a polynomial library or a polynomial library augmented with trigonometric terms, while SORT uses either Legendre or trigonometric tensor bases with sparse regression.

Figure 2 summarizes the cyclic-system results using median short-rollout RMSE with interquartile ranges across random seeds. On the Thomas attractor, both methods perform comparably when the representation is well matched, particularly with trigonometric features. This is expected: the true dynamics align with the candidate representation, so the main diferences are due to numerical conditioning and noise sensitivity.

![](images/76db8718e96de00b17843fe8273519b2ae2fb48212b6525046da343046b24ce8.jpg)  
(a) Thomas: RMSE vs train fraction.

![](images/3136d86f9bb5165993590cc5f51346c712ab784512854732d7f5b03ed692497e.jpg)  
(b) Thomas: RMSE vs noise.

![](images/647f0c0b5fee4ca92be44077689cdadff0e485d00114a5731198e3b3999acbaf.jpg)  
(c) BesselJ1: RMSE vs train fraction.

![](images/e7d65e2727192b83b4fc40530ec8fa219e52269cc70d00dda1103bdea26d7ef6.jpg)  
(d) BesselJ1: RMSE vs noise.  
Fig. 2: Short-rollout prediction error for SINDy and SORT under basis choice. Each panel reports short-horizon rollout RMSE on a log scale, shown as the median with interquartile-range band. Solid lines correspond to polynomial or Legendre bases; dashed lines correspond to trigonometric variants. For SINDy, dashed curves use a polynomial library augmented with trigonometric terms; for SORT, dashed curves use a trigonometric tensor basis. Panels (a)– (b) show the Thomas attractor, where both methods perform comparably and trigonometric features improve accuracy. Panels $\mathrm { ( c ) - ( d ) }$ show the Bessel-driven system with cyclic coupling through $J _ { 1 } ( \cdot )$ , where SORT remains robust across training fractions and noise levels while SINDy degrades under representation mismatch. Panels (a) and (c) vary the training fraction at fixed noise $\sigma = 0 . 1$ panels (b) and (d) vary the noise level at fixed training fraction 20%.

The contrast is stronger for the Bessel-driven system. Here the SINDy-style baseline degrades as the training fraction decreases or the noise level increases, both with a polynomial library and with additional trigonometric terms. The issue is representation mismatch: the true nonlinearity is not sparse in the selected library, making coeficient selection brittle under noise. SORT remains more robust because it does not require the correct analytic building blocks to be anticipated in a finite dictionary. Instead, the vector field is approximated in orthonormal coordinates, where sparsity suppresses unstable modes and orthogonality reduces feature collinearity.

The benchmark study shows that, under sparse temporal sampling, sparse orthogonal regression can degrade more gradually than a SINDy-style library model using the same sampled states and finite-diference derivatives. The Thomas and Bessel experiments test noise, smoothing, and representation mismatch, showing that orthonormal expansions remain useful when the natural basis is uncertain or poorly captured by a small fixed library. Overall, the results support the view that basis choice is part of the modeling problem, not an implementation detail.

## 3.2 Integral estimation

Once an orthonormal expansion is learned, integrals can be evaluated as linear functionals of the estimated coeficients. SORT uses this property in a datadriven way: instead of computing expansion coeficients by quadrature or exact inner products, it estimates them from pointwise samples and then evaluates the integral by coeficient readout.

We work on a canonical domain $t \in [ - 1 , 1 ] ^ { d }$ , related to a bounded physical domain $x \in [ a , b ] ^ { d }$ by the afine map

$$
t _ { j } = \frac { 2 x _ { j } - ( a + b ) } { b - a } , \qquad x _ { j } = \frac { b - a } 2 t _ { j } + \frac { a + b } 2 .\tag{1}
$$

Let

$$
D ( t ) = \prod _ { j = 1 } ^ { d } ( 1 + t _ { j } ^ { 2 } ) ,
$$

and define the transformed target

$$
g ( t ) = \frac { f ( x ( t ) ) } { D ( t ) } .
$$

We expand $g$ in a tensor-product orthonormal basis $\{ \Phi _ { \alpha } \} _ { \alpha \in \Lambda }$ on $[ - 1 , 1 ] ^ { d }$ , choosing the basis element $\varPhi _ { \mathbf { 0 } }$ as the normalized Riesz representer of the integration functional,

$$
\varPhi _ { \bf 0 } ( t ) = \frac { D ( t ) } { \| D \| _ { L ^ { 2 } ( [ - 1 , 1 ] ^ { d } ) } } .
$$

SORT fits a sparse expansion

$$
g ( t ) \approx \sum _ { \alpha \in A _ { N } } c _ { \alpha } \varPhi _ { \alpha } ( t ) .
$$

With this construction, integral estimation reduces to a single-coeficient readout:

$$
\int _ { [ a , b ] ^ { d } } f ( x ) d x \approx \Big ( \frac { b - a } { 2 } \Big ) ^ { d } c _ { 0 } \| D \| _ { L ^ { 2 } ( [ - 1 , 1 ] ^ { d } ) } .\tag{2}
$$

Because $D ( t )$ factorizes across dimensions, its norm has a closed-form expression. Improper integrals can be handled either by mapping the unbounded domain to $[ - 1 , 1 ] ^ { d }$ or by reporting controlled finite-window approximations. In higher dimensions, sparse index sets such as hyperbolic-cross truncations are used to limit basis size and preserve sample eficiency.

![](images/b0808850eab99da233a6ec584207dd69c05d630b790fe12b6d7ac9c0c323efe6.jpg)  
(a) 1D Fresnel-type integral

![](images/67fbffc16ed537cf362e5a79f2d63a7278a535183e9c50bd60de8d6ae6025d69.jpg)  
(b) 1D Gaussian integral

![](images/1c4f1c27b8f730dbc465e225e8a1901ff416fe7a7711dd6100ead80b3a93fb03.jpg)  
(c) High-dimensional Fresnel integral

![](images/9cfbf98c0ea9f7a54ca8b827f8ef5cb06d941d1b137c384821a96cc8d7784a71.jpg)  
(d) High-dimensional Gaussian integral  
Fig. 3: Integral estimation via SORT. SORT learns an orthonormal expansion of a transformed integrand from pointwise samples and estimates integrals by reading of the coeficient associated with the integration functional. In one dimension, SORT matches closed-form values for both oscillatory Fresnel-type integrals and smooth Gaussian integrals across a range of parameters. In higher dimensions, separable test integrands are used so that exact reference values remain available; accuracy decreases gradually with dimension, consistent with the increased sample complexity of sparse polynomial recovery.

Figure 3 reports results for smooth and oscillatory integrands. Panels (a)–(b) show one-dimensional examples: Gaussian integrals $\begin{array} { r l } {  { \int _ { 0 } ^ { L } e ^ { - \alpha x ^ { 2 } } } } \end{array}$ dx and Fresneltype integrals $\begin{array} { r l } {  { \int _ { 0 } ^ { L } \cos ( \alpha x ^ { 2 } ) } } \end{array}$ dx. SORT accurately tracks the closed-form values across a wide range of $\alpha ,$ including strongly oscillatory Fresnel cases where accurate quadrature would require resolving rapid oscillations. Panels (c)–(d) extend the evaluation to higher-dimensional separable integrals. The estimates remain accurate in moderate dimensions, but the error increases with dimension, reflecting the growth of the basis and the sample complexity of sparse coeficient recovery. Overall, these experiments illustrate the second application-level contribution of SORT: once a sparse expansion is recoverable, integration becomes a stable coeficient-readout task rather than a separate quadrature problem.

## 3.3 Spectral recovery and order-consistent approximation

We finally use two controlled approximation examples to illustrate a standard but useful property of orthonormal expansions: increasing the model order refines the representation without changing the meaning of previously learned low-order coeficients. This is not a new fact from approximation theory. The point is that,

![](images/25462e2e1f283a4179b3fb93db928a18c67a76f46c56315ed8b0f7ef5bc36abd.jpg)  
(a) Fourier coeficients for $f ( x ) = x$

![](images/41dfeb8facd08a53cf7417d1923488501035c1c6ecb1e14e6433de769cb11ab0.jpg)  
(b) Held-out reconstruction

![](images/2c5b0317f1791a9c1a0fe850d93a8b3bfc0209211f16e80a66b96c797f8707ac.jpg)

![](images/cff0ae47e8f4edaf044cbc3269c4a1d3ea0cd62764cf95662230ddda1a73d262.jpg)

(c) Test RMSE vs parameter count  
![](images/aed66176587aaa6376ab0101230ce75b33b32cdd843065e03e91778a3043552a.jpg)

(d) SORT coeficients  
![](images/4ee681ab67fa14327941db19028559137e9db08fecf680bc50c101da5060cb24.jpg)

(e) OLS coeficients, monomials  
![](images/2e72a891b4da2c1df4c08e66b3ad30647072395c7c63f31925695a1d46c976d9.jpg)  
(g) RBF ridge coeficients

(f) OLS coeficients, Legendre  
![](images/f08ee8de5020d5f31e0861df4cc86e239a8ca7e8ec08b5de1e10f7afa209cd0c.jpg)  
(h) RFF ridge coeficients

Fig. 4: Spectral recovery and order-consistent approximation. (a)–(b) Recovery of the Fourier sine structure of $f ( x ) ~ = ~ x ~ \mathrm { o n } ~ [ - \pi , \pi ]$ from noisy data using 20% of the samples for training and 80% for testing. OLS amplifies unstable high-frequency modes, while SORT recovers the dominant $1 / k$ coeficient decay and gives a stable reconstruction. (c)–(h) Approximation of $f ( x _ { 1 } , x _ { 2 } ) = \exp ( x _ { 1 } \cos ( 2 x _ { 2 } ) ) { \mathrm { ~ o n ~ } } [ - 1 , 5 ] ^ { 2 }$ using 20% noisy training data $( \sigma = 0 . 1 )$ and 80% clean test data. SORT achieves low test error and a stable low-order coeficient hierarchy, while dense monomial, kernel, and random-feature representations produce coeficients that vary more strongly with model size.

when combined with sparsity, it gives a practical machine-learning diagnostic for stable model growth under noise and subsampling.

The first example has known spectral structure: $f ( x ) = x \mathrm { ~ o n ~ } [ - \pi , \pi ]$ has Fourier sine coeficients $| b _ { k } | = 2 / k$ . Figure 4(a)–(b) shows that, under noise and subsampling, OLS fits unstable high-frequency coeficients that lead to oscillatory test error. SORT instead suppresses unsupported modes and preserves the dominant low-order spectral structure.

The second example considers

$$
f ( x _ { 1 } , x _ { 2 } ) = \exp { \big ( } x _ { 1 } \cos ( 2 x _ { 2 } ) { \big ) } , \qquad ( x _ { 1 } , x _ { 2 } ) \in [ - 1 , 5 ] ^ { 2 } ,
$$

using 20% noisy training data, 80% clean test data, tensor-product Legendre bases with total-degree cutof $D _ { \mathrm { m a x } } = 5 0$ , and parameter counts $k \in \{ 6 0 , 8 0 , 1 0 0 \}$ Figure $4 ( \mathrm { { c } ) - ( \mathrm { { h } ) } }$ compares SORT with dense least-squares, RBF, RFF, and gradientboosting baselines. SORT gives the lowest test error and, more importantly, preserves a stable low-order coeficient hierarchy as model size increases. OLS in a monomial basis changes substantially with order; OLS in a Legendre basis benefits from orthogonality but remains noise-sensitive; and RBF/RFF ridge models produce dense weights without an interpretable hierarchy.

These examples illustrate the approximation-side contribution of SORT. Sparse regression in orthogonal bases is standard, but its order-consistent structure is underused as a machine-learning design principle: model complexity can be increased by adding higher-order terms while retaining the interpretation of previously learned low-order coeficients. This is also the sense in which the name SORT is descriptive: models can be compared across truncation orders by their active coeficient sets, coeficient persistence, sparsity level, and predictive accuracy, because all models are expressed in the same ordered orthonormal coordinate system.

## 4 Conclusion

This paper developed SORT as a sparse spectral representation framework built from orthonormal basis expansions and sparsity-promoting regression. The contribution is application-level: SORT turns a learned coeficient vector into a reusable object for integration, dynamical-system reconstruction, and nonlinear approximation within a common representation.

The dynamical-system experiments show that sparse orthonormal vectorfield representations are useful when derivative information is degraded or when the natural basis is uncertain. In the sampling-degradation study, SORT remains competitive with SINDy-style sparse identification at fine sampling and degrades less abruptly under coarse sampling. In the cyclic-system experiments, it performs comparably on the Thomas attractor when trigonometric structure is supplied to both methods, and is more robust on the Bessel-driven variant, where the dynamics are poorly captured by a small polynomial or trigonometric library. These results support the view that basis design is part of the scientific modeling problem, not an implementation detail.

The integration experiments show that data-driven quadrature can be treated as coeficient readout. Rather than constructing explicit quadrature rules or computing inner products analytically, SORT estimates expansion coeficients from samples and evaluates the integral through the coeficient associated with the integration functional. This works for smooth Gaussian integrals, oscillatory Fresnel-type integrals, and moderate-dimensional separable examples when the expansion remains sparse.

The approximation experiments highlight the value of order-consistent model growth. Because orthonormal expansions provide a stable hierarchy of basis functions, increasing the model order can refine the representation without changing the meaning of lower-order coeficients. In noisy finite data, coeficient persistence across truncation levels gives a practical diagnostic for stable learning.

Overall, SORT provides a compact intermediate representation between raw samples and later symbolic or analytic description. It can first recover a stable expansion in a suitable orthogonal basis; a subsequent symbolic stage may then search for simpler closed-form expressions consistent with that expansion. Future work should develop adaptive and anisotropic basis selection, empirical orthogonalization for irregular samples, and larger-scale applications. The results here suggest that sparse orthonormal expansions learned from data ofer a stable, reusable, and computationally simple route to equation discovery, integration, and approximation.

## Code Availability

Code for reproducing the experiments and figures is available at: https://doi.org/10.5281/zenodo.21707070.

## Disclaimer

Co-funded by the European Union. Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union or European Research Executive Agency. Neither the European Union nor the granting authority can be held responsible for them.

## Acknowledgments.

This publication is supported by the European Union’s Horizon Europe research and innovation programme under the Marie Skłodowska-Curie Postdoctoral Fellowship Programme, SMASH co-funded under the grant agreement No. 101081355. The operation (SMASH project) is co-funded by the Republic of Slovenia and the European Union from the European Regional Development Fund.

The authors acknowledge the financial support of the Slovenian Research Agency via the Gravity project AI for Science, GC-0001 and of the Slovenian Research And Innovation Agency (research core funding No. P1-0188).

## References

1. Billings, S.A., Wei, H.L.: Sparse model identification using a forward orthogonal regression algorithm aided by mutual information. IEEE Transactions on Neural Networks 18(1), 306–310 (2007). https://doi.org/10.1109/tnn.2006.886356

2. Blatman, G., Sudret, B.: Adaptive sparse polynomial chaos expansion based on least angle regression. Journal of Computational Physics 230(6), 2345–2367 (2011). https://doi.org/10.1016/j.jcp.2010.12.021

3. Boyd, J.: Rational chebyshev spectral methods for unbounded solutions on an infinite interval using polynomial-growth special basis functions. Computers & Mathematics with Applications 41(10-11), 1293–1315 (2001), https://doi.org/ 10.1016/s0898-1221(01)00098-0

4. Brence, J., Todorovski, L., Džeroski, S.: Probabilistic grammars for equation discovery. Knowledge-Based Systems 224, 107077 (2021). https://doi.org/10. 1016/j.knosys.2021.107077

5. Brunton, S.L., Proctor, J.L., Kutz, J.N.: Discovering governing equations from data by sparse identification of nonlinear dynamical systems. Proceedings of the National Academy of Sciences 113(15), 3932–3937 (2016). https://doi.org/10. 1073/pnas.1517384113

6. Candès, E.J., Romberg, J.: Robust signal recovery from incomplete observations. In: 2006 IEEE International Conference on Image Processing. pp. 1281–1284. IEEE (2006). https://doi.org/10.1109/ICIP.2006.312579

7. Chen, S., Hong, X., Harris, C., Sharkey, P.: Sparse modeling using orthogonal forward regression with press statistic and regularization. IEEE Transactions on Systems, Man and Cybernetics, Part B (Cybernetics) 34(2), 898–911 (2004). https://doi.org/10.1109/tsmcb.2003.817107

8. Doostan, A., Owhadi, H.: A non-adapted sparse approximation of pdes with stochastic inputs. Journal of Computational Physics 230(8), 3015–3034 (2011). https://doi.org/10.1016/j.jcp.2011.01.002

9. Efron, B., Hastie, T., Johnstone, I., Tibshirani, R.: Least angle regression. The Annals of Statistics 32(2), 407–499 (2004). https://doi.org/10.1214/ 009053604000000067

10. Feng, Y., Mangan, N.M., Jayadharan, M.: Ill-conditioning in dictionary-based dynamic-equation learning: A systems biology case study (2026). https://doi. org/10.48550/arXiv.2603.11330

11. Gilbert, A., Gu, A., Ré, C., Rudra, A., Wootters, M.: Sparse Recovery for Orthogonal Polynomial Transforms. In: 47th International Colloquium on Automata, Languages, and Programming (ICALP 2020). Leibniz International Proceedings in Informatics (LIPIcs), vol. 168, pp. 58:1–58:16 (2020). https://doi.org/10.4230/ LIPIcs.ICALP.2020.58

12. Guimerà, R., Reichardt, I., Aguilar-Mogas, A., Massucci, F.A., Miranda, M., Pallarès, J., Sales-Pardo, M.: A bayesian machine scientist to aid in the solution of challenging scientific problems. Science Advances 6(5), eaav6971 (2020). https://doi.org/10.1126/sciadv.aav6971

13. Guo, Y., Guo, L., Billings, S., Wei, H.L.: Ultra-orthogonal forward regression algorithms for the identification of non-linear dynamic systems. Neurocomputing 173, 715–723 (2016). https://doi.org/10.1016/j.neucom.2015.08.022

14. Hampton, J., Doostan, A.: Compressive sampling of polynomial chaos expansions: Convergence analysis and sampling strategies. Journal of Computational Physics 280, 363–386 (2015). https://doi.org/10.1016/j.jcp.2014.09.019

15. Hastie, T., Tibshirani, R., Wainwright, M.: Statistical Learning with Sparsity: The Lasso and Generalizations, Monographs on Statistics and Applied Probability, vol. 143. CRC Press, Boca Raton (2015). https://doi.org/10.1201/b18401

16. Hong, X., Chen, S.: Elastic net orthogonal forward regression. Neurocomputing 148, 551–560 (2015). https://doi.org/10.1016/j.neucom.2014.07.008

17. Jakeman, J., Eldred, M., Sargsyan, K.: Enhancing $\ell _ { 1 } \cdot$ -minimization estimates of polynomial chaos expansions using basis selection. Journal of Computational Physics 289, 18–34 (2015). https://doi.org/10.1016/j.jcp.2015.02.025

18. Kulkarni, C.S., Gupta, A., Lermusiaux, P.F.J.: Sparse regression and adaptive feature generation for the discovery of dynamical systems. In: Darema, F., Blasch, E., Ravela, S., Aved, A. (eds.) Dynamic Data Driven Applications Systems. Lecture Notes in Computer Science, vol. 12312, pp. 208–216. Springer (2020). https:// doi.org/10.1007/978-3-030-61725-7\_25

19. Lüthen, N., Marelli, S., Sudret, B.: Sparse polynomial chaos expansions: Literature survey and benchmark. SIAM/ASA Journal on Uncertainty Quantification 9(2), 593–649 (2021). https://doi.org/10.1137/20M1315774

20. Omejc, N., Gec, B., Brence, J., Todorovski, L., Džeroski, S.: Probabilistic grammars for modeling dynamical systems from coarse, noisy, and partial data. Machine Learning 113(10), 7689–7721 (2024). https://doi.org/10.1007/ s10994-024-06522-1

21. Peng, J., Hampton, J., Doostan, A.: A weighted ℓ<sub>1</sub>-minimization approach for sparse polynomial chaos expansions. Journal of Computational Physics 267, 92– 111 (2014). https://doi.org/10.1016/j.jcp.2014.02.024

22. Rauhut, H., Ward, R.: Sparse Legendre expansions via ℓ<sub>1</sub>-minimization. Journal of Approximation Theory 164(5), 517–533 (2012). https://doi.org/10.1016/j. jat.2012.01.008

23. Roman, S.: Maximum entropy models for unimodal time series: Case studies of universe 25 and st. matthew island. In: Discovery Science. Lecture Notes in Computer Science, vol. 16090, pp. 32–44. Springer, Cham (2025). https://doi.org/ 10.1007/978-3-032-05461-6\_3

24. Roman, S., Bertolotti, F.: Toward a new AI winter? How difusion of technological innovation on networks leads to chaotic boom-bust cycles. Frontiers in Artificial Intelligence 8, 1671917 (2025). https://doi.org/10.3389/frai.2025.1671917

25. Roman, S., Todorovski, L., Džeroski, S., Skok, G.: Approximating the universal thermal climate index using sparse regression with orthogonal polynomials. Geoscientific Model Development 19(10), 4319–4330 (2026). https://doi.org/10.5194/ gmd-19-4319-2026

26. Rudy, S.H., Brunton, S.L., Proctor, J.L., Kutz, J.N.: Data-driven discovery of partial diferential equations. Science Advances 3(4), e1602614 (2017). https:// doi.org/10.1126/sciadv.1602614

27. Stein, E.M., Shakarchi, R.: Fourier analysis: an introduction, vol. 1. Princeton University Press (2011)

28. Steinmann, P., Verstegen, J., Van Voorn, G., Roman, S., Ligtenberg, A.: Scenario search: finding diverse, plausible and comprehensive scenario sets for complex systems. Socio-Environmental Systems Modelling 7, 18823 (2025). https: //doi.org/10.18174/sesmo.18823

29. Tang, X., Zhang, L.: Stability orthogonal regression for system identification. Systems & Control Letters 117, 30–36 (2018). https://doi.org/10.1016/j. sysconle.2018.05.002

30. Tibshirani, R.: Regression shrinkage and selection via the lasso. Journal of the Royal Statistical Society Series B: Statistical Methodology 58(1), 267–288 (1996). https://doi.org/10.1111/j.2517-6161.1996.tb02080.x

31. Tran, H., Webster, C.: Analysis of sparse recovery for Legendre expansions using envelope bound. Numerical Methods for Partial Diferential Equations 38(6), 2163– 2198 (2022). https://doi.org/10.1002/num.22877