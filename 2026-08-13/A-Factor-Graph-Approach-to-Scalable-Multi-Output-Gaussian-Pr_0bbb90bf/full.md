# A Factor Graph Approach to Scalable Multi-Output Gaussian Process Regression

Wouter W. L. Nuijten w.w.l.nuijten@tue.nl Eindhoven University of Technology & Lazy Dynamics B.V., the Netherlands

Esther G. van Pelt Eindhoven University of Technology, Eindhoven, the Netherlands

e.g.v.pelt@tue.nl

Albert Podusenko   
İsmail Şenöz   
Lazy Dynamics B.V., the Netherlands

albert@lazydynamics.com isenoz@lazydynamics.com

Wouter M. Kouw Eindhoven University of Technology, Eindhoven, the Netherlands

w.m.kouw@tue.nl

Editor: Gustau Camps-Valls, Manuele Leonelli and Gherardo Varando

## Abstract

Multi-output Gaussian process regression scales cubically in the number of observations times outputs, and dense kernel-matrix methods need bespoke handling whenever diferent outputs are observed at diferent inputs. We express multi-output Gaussian process regression as a Forney-style factor graph in which a nearest-neighbor chain orders a fixed candidate set of C inputs into a one-dimensional sequence. Along this chain, latent Matérn processes evolve through linear-Gaussian transition factors, while the linear model of coregionalization mixes L latent processes into D outputs through a deterministic mixing factor and per-output scalar observation factors. Posterior computation reduces to exact Gaussian message passing on the chain at cost O(C(DL<sup>2</sup> + L<sup>3</sup>)) after chain construction, and missing observations omit their local factor without any covariance-matrix restructuring. The formulation therefore scales in the number of data samples and in the rate of missing observations, while remaining best suited to candidate sets in low input dimension. We compare the factor-graph formulation against an exact kernel-matrix baseline, a sparse-variational inducing-point baseline, and a nearest-neighbor baseline on a synthetic input-dimension sweep and on electricity time series forecasting. At low input dimension the factor-graph posterior tracks the exact kernel-matrix posterior closely, and the gap grows gradually as input dimension increases while staying competitive with both approximate baselines. On the electricity time series our factor-graph formulation matches all three baselines in forecast accuracy while scaling linearly in the number of data points, where the exact kernel-matrix method becomes infeasible and the inducing-point baseline remains substantially slower.

Keywords: Factor graphs, Gaussian processes, Message passing, Partial observations

## 1. Introduction

Probabilistic graphical models provide a unified language for inference: belief propagation, Kalman filtering and smoothing, hidden Markov model inference, and variational message passing all arise as instances of message passing on specific factor graphs (Loeliger et al., 2007; Şenöz et al., 2021). Gaussian processes (GPs) are a notable outlier. Defined by a kernel rather than a factorization, GPs are typically treated as monolithic objects, which do not scale well and have poor compositionality. In the multi-output case, fitting a standard GP with D correlated outputs and N training points requires operations on a $D N \times D N$ covariance matrix, i.e., $\mathcal { O } ( ( D N ) ^ { 3 } )$ , which becomes prohibitive as D or N grows (Álvarez et al., 2012). Equally inconvenient, when diferent outputs are observed at diferent inputs, as is the norm in sensor networks, environmental monitoring, and clinical time series, dense kernel-matrix implementations typically handle these partial observations by restructuring or slicing the covariance matrix rather than making a local edit to a graph.

But non-monolithic treatments of GPs do exist. A GP with a Matérn covariance function can be expressed as the solution of a linear stochastic diferential equation (SDE) (Hartikainen and Särkkä, 2010; Särkkä and Solin, 2019), and the resulting linear-Gaussian state-space model has an exact representation as a Forney-style factor graph (FFG) on which Kalman filtering and smoothing reduce to message passing (Loeliger et al., 2007). This bridge has been widely exploited in temporal settings, where the natural ordering of time supplies the one-dimensional chain that the state-space form requires. We use this bridge as the starting point for multi-output regression: we cast low-input-dimensional multi-output GP (MOGP) regression as message passing on an FFG by combining state-space GPs along a greedy nearest-neighbor chain over a fixed candidate set of size C with the Linear Model of Coregionalization (LMC), expressed as a deterministic mixing factor and per-output scalar observation factors. The chain construction is an approximation, since M-dimensional input geometry is compressed onto a one-dimensional Markov sequence. Here we investigate the fidelity of such a compression.

Casting the model in this form has two consequences for inference. First, conditional on the chain, posterior computation is exact Gaussian message passing: the L latent dynamics remain block diagonal in the prior, while the LMC observation factors couple them into dense Gaussian beliefs over the joint latent state. For fixed D and L, inference is therefore linear in the candidate-chain length C, with $\mathcal { O } ( C ( D L ^ { 2 } + L ^ { 3 } ) )$ cost after chain construction. Second, partial observability is native to the factor-graph implementation: outputs that are unobserved at a given input simply omit their local observation message, with no covariancematrix restructuring and no change to the model or to the inference call. Dense kernel-matrix implementations, by contrast, must explicitly build or slice variable-size covariance matrices from the observed (output, input) pairs.

Concretely, our contribution is firstly a chain-induced factor graph formulation of lowinput-dimensional multi-output GP regression, combining state-space GPs (Hartikainen and Särkkä, 2010), FFG message passing (Loeliger et al., 2007), and the LMC (Álvarez et al., 2012), with $\mathcal { O } ( C ( D L ^ { 2 } + L ^ { 3 } ) )$ per-step inference after chain construction. Approximation occurs only through the nearest-neighbor chain; inference on the resulting model is exact. Secondly, partial observability requires no restructuring of the inference procedure, as opposed to dense kernel-matrix treatments that require covariance-matrix slicing.

## 2. Problem Statement

Let $x \in \mathcal { X } \subset \mathbb { R } ^ { M }$ denote an input and $f : \mathcal { X } \to \mathbb { R } ^ { D }$ a vector-valued function with D correlated outputs. Given a training set of N inputs $X \in \mathbb { R } ^ { N \times M }$ and a noisy observation matrix $Y \in \mathbb { R } ^ { N \times D }$ with $y _ { i } = f ( x _ { i } ) + \epsilon _ { i }$ and Gaussian noise $\epsilon _ { i } \sim \mathcal { N } ( 0 , \Sigma _ { \epsilon } )$ , where $\Sigma _ { \epsilon } \in$ $\mathbb { R } ^ { D \times D }$ is a diagonal observation-noise covariance, we wish to infer the posterior predictive distribution of $f$ at unseen inputs under a zero-mean multi-output Gaussian process prior. A binary mask $\dot { O } \in \{ 0 , 1 \} ^ { N \times D }$ records which entries are observed, with arbitrary missingness allowed across $( i , d )$

We target exact Gaussian inference under the chain-induced factor-graph model with two requirements. First, inference cost should grow linearly in the number of training points N for fixed output count D and fixed latent rank. Second, the missingness mask O should be absorbed into the model itself, so that arbitrary missingness patterns produce no additional restructuring cost. Kernel-matrix methods satisfy neither requirement, as vectorised inference costs $\mathcal { O } ( ( N D ) ^ { 3 } )$ and varying missingness forces a variable-size covariance to be rebuilt from the observed entries. Section 4 presents a factor-graph reformulation that meets both requirements, and Section 5 assesses posterior quality empirically.

## 3. Background

## 3.1. Multi-Output Gaussian Processes

Exact multi-output GP inference is costly because covariance matrices couple inputs and outputs. We focus on the Linear Model of Coregionalization (LMC), which mixes L latent $\mathrm { G P s } \ g _ { l } ( x )$ through a mixing matrix $W \in \mathbb { R } ^ { D \times L }$ with induced cross-output covariance

$$
f _ { d } ( x ) = \sum _ { l = 1 } ^ { L } w _ { d l } g _ { l } ( x ) , \qquad \mathrm { C o v } \big [ f _ { d } ( x ) , f _ { d ^ { \prime } } ( x ^ { \prime } ) \big ] = \sum _ { l = 1 } ^ { L } w _ { d l } w _ { d ^ { \prime } l } k _ { l } ( x , x ^ { \prime } ) .\tag{1}
$$

At a fixed input, or with shared latent input covariance, the output-side structure is lowrank plus diagonal noise. For $L \ll D$ , Woodbury applies inverses of $W S W ^ { \top } + \Sigma$ <sub>ϵ</sub> through an $L \times L$ inner system, costing $\mathcal { O } ( D L ^ { 2 } + L ^ { 3 } )$ , or $\mathcal { O } ( D L ^ { 2 } )$ for fixed $L ,$ instead of dense $D \times D$ operations. The special case $L = 1$ is the Intrinsic Coregionalization Model (ICM), which is less expressive than LMC (Alvarez and Lawrence, 2011; Bonilla et al., 2007).

## 3.2. State-Space Gaussian Processes

The cubic complexity in samples has driven extensive work on scalable approximations. Sparse variational methods based on inducing points reduce this cost while preserving a variational lower bound on the marginal likelihood (Smola et al., 2004; Bruinsma et al., 2020). However, these are approximations. Temporal GPs with Matérn covariance functions admit an exact representation as linear SDEs, converting GP regression into Kalman filtering and smoothing at $\mathcal { O } ( N )$ cost (Hartikainen and Särkkä, 2010; Särkkä, 2013; Särkkä and Solin, 2019). For example, a scalar temporal GP with a stationary Matérn $3 / 2$ covariance function $\kappa ( t , t ^ { \prime } )$ , length-scale $\ell ,$ and output scale $\gamma ^ { 2 }$ can be represented exactly as the solution of the linear SDE

$$
\mathrm { d } \left[ \begin{array} { c } { g ( t ) } \\ { \dot { g } ( t ) } \end{array} \right] = \left[ \begin{array} { c c } { 0 } & { 1 } \\ { - \lambda ^ { 2 } } & { - 2 \lambda } \end{array} \right] \left[ \begin{array} { c } { g ( t ) } \\ { \dot { g } ( t ) } \end{array} \right] \mathrm { d } t + \left[ \begin{array} { c } { 0 } \\ { 1 } \end{array} \right] \mathrm { d } w ( t ) ,\tag{2}
$$

for $\lambda = \sqrt { 3 } / \ell$ and white noise $w ( t )$ with spectral density $4 \lambda ^ { 3 } \gamma ^ { 2 }$ , where $\gamma ^ { 2 }$ is the GP output scale (Hartikainen and Särkkä, 2010). We shall refer to the state vector of the $\mathrm { G P } \ g ( t )$ and its derivative in time ${ \dot { g } } ( t )$ as $z ( t )$ . The stationary covariance of the state is $P _ { \infty } =$

diag $\begin{array} { r l } { \left[ \gamma ^ { 2 } \right. } & { { } \left. \lambda ^ { 2 } \gamma ^ { 2 } \right] ) } \end{array}$ . This SDE can be discretized to form a discrete-time state-space model over states $\boldsymbol { z } _ { t } \doteq [ g _ { t } , \dot { g } _ { t } ] ^ { \top }$ . Given the inter-point distance $\Delta _ { i }$ between consecutive points in the chain, the discrete-time transition and process noise matrices are

$$
A _ { i } = \exp \left( \left[ \begin{array} { c c } { { 0 } } & { { 1 } } \\ { { - \lambda ^ { 2 } } } & { { - 2 \lambda } } \end{array} \right] \Delta _ { i } \right) , \qquad Q _ { i } = P _ { \infty } - A _ { i } P _ { \infty } A _ { i } ^ { \top } .\tag{3}
$$

The observation model extracts the function value from the state through the observation vector $h = [ 1 , 0 ] ^ { \top }$ , so that $g _ { t } = h ^ { \top } z _ { t }$ . This yields a linear-Gaussian state-space model equivalent to the GP, with $\mathcal { O } ( N )$ inference in the number of points for a single scalar output via Bayesian filtering and smoothing (Särkkä, 2013). The multi-output cost is treated in Section 4.2.

## 3.3. Factor Graphs and Message Passing

Factor graphs provide a unified language for probabilistic inference, subsuming belief propagation, variational inference and expectation propagation as message computation on a graph (Kschischang et al., 2001; Şenöz et al., 2021; Heess et al., 2013). Many classical algorithms, including Kalman filtering and smoothing, can be written as message passing on particular factor graphs (Loeliger et al., 2007; Palmieri et al., 2022). Take, for example, the below probabilistic model over $x _ { 1 } , x _ { 2 }$ and $x _ { 3 }$ , and assume that we are interested in the marginal distribution for x<sub>3</sub>:

$$
p ( x _ { 1 } , x _ { 2 } , x _ { 3 } ) = p ( x _ { 1 } | x _ { 2 } ) p ( x _ { 2 } | x _ { 3 } ) p ( x _ { 3 } ) , \qquad p ( x _ { 3 } ) = \int p ( x _ { 1 } , x _ { 2 } , x _ { 3 } ) \mathrm { d } ( x _ { 1 } , x _ { 2 } ) .\tag{4}
$$

In a Forney-style factor graph (FFG), factors are nodes and variables are edges; each edge connects at most two factor nodes (Loeliger, 2004). The notation is illustrated in Figure 1: boxes denote factors, edge labels denote variables, and arrows label messages passed along those variables. For a generic node $f ( y , x _ { 1 } , \ldots , x _ { n } )$ , the sum-product message to edge y is

![](images/5d730bbf04379966af8570f63784ae90e4e9224bc6592c7170c90e158ff870ad.jpg)  
Figure 1: Example FFG. Messages along an edge summarize all factors of other edges.

$$
\vec { \nu } ( y ) = \int \cdot \cdot \cdot \int f ( y , x _ { 1 } , \ldots , x _ { n } ) \prod _ { i = 1 } ^ { n } \vec { \nu } ( x _ { i } ) \mathrm { d } x _ { i } ,\tag{5}
$$

where $\vec { \nu } ( x _ { i } )$ are incoming messages to the node. Priors can be represented as terminal factors, likelihoods send observation information back into the graph, and posterior marginals are obtained by multiplying incoming messages on an edge (Şenöz et al., 2021). A key property of FFGs is compositionality: model changes, such as adding or removing observation factors, require only local graph modifications, not a new inference algorithm. Reactive message passing exploits this structure by compiling model declarations into message updates and recomputing only afected messages when data or factors change (Bagaev and De Vries, 2023). This compositionality is central to our treatment of partial observations.

## 4. Message Passing-Based Multi-Output GP Inference

## 4.1. Model Specification

## 4.1.1. Nearest-Neighbor Chain Ordering

Extending state-space GPs beyond one-dimensional inputs is an active research area. In spatiotemporal settings, a natural temporal ordering exists, and the spatial structure can be handled separately via inducing points (Särkkä et al., 2013; Tebbutt et al., 2021; Hamelijnck et al., 2021). An alternative approach is the SPDE method of Lindgren et al. (2011), which discretizes the stochastic partial diferential equation on a mesh to obtain a sparse precision matrix for Matérn fields in arbitrary dimensions. Recently, Li et al. (2025) proposed a universal method for converting any stationary temporal GP into a state-space model via spectral factorization. Our work difers from these spatiotemporal approaches in that we do not assume separable structure. Instead, we construct a one-dimensional chain via nearest-neighbor ordering. This makes state-space inference applicable to non-temporal candidate sets, but it is an approximation whose fidelity depends on how well the chain preserves the relevant geometry. A closely related family also relies on nearest-neighbor orderings: the Vecchia approximation (Vecchia, 1988) conditions each point on a small set of preceding neighbors to obtain a sparse Cholesky factor of the precision matrix, realized as the hierarchical nearest-neighbor GP of Datta et al. (2016) and generalized by Katzfuss and Guinness (2021). These methods produce an approximate sparse precision from variable-size conditioning sets, whereas our Markov-1 chain gives an exact message-passing recursion at the cost of compressing the M-dimensional geometry into a single chain.

For M-dimensional inputs $\boldsymbol { X } \in \mathbb { R } ^ { C \times M }$ , with fixed candidate-set size C, we construct this ordering via a greedy nearest-neighbor heuristic. Let $\pi : \{ 1 , \dots , C \} \to \{ 1 , \dots , C \}$ denote the resulting permutation, so that $x _ { \pi ( i ) }$ is the i-th point visited in the chain: starting from an arbitrary point $( \mathrm { e . g . } , x _ { 1 } )$ , we repeatedly visit the closest unvisited point in Euclidean distance and record the inter-point distances $\Delta _ { i } = \| x _ { \pi ( i ) } - x _ { \pi ( i - 1 ) } \|$ , which serve as the time steps in the state-space model. This ordering defines a chain-induced kernel approximation. Let $s _ { \pi ( 1 ) } = 0$ and $\begin{array} { r } { s _ { \pi ( i ) } = \sum _ { r = 2 } ^ { i } \Delta _ { r } } \end{array}$ be the one-dimensional coordinate induced by the chain. The chain distance is the distance between these coordinates and the chain kernel is the kernel defined by the Matérn kernel covariance function evaluated at the chain distance:

$$
[ K _ { \pi } ] _ { i j } = \kappa \big ( | s _ { \pi ( i ) } - s _ { \pi ( j ) } | \big ) .\tag{6}
$$

A naive greedy approach to constructing the chain will cost $\mathcal { O } ( C ^ { 2 } )$ , but techniques such as kd-trees can speed this up to ${ \mathcal { O } } ( C \log C )$ in low input dimension (Cormen et al., 2022).

To characterize the fidelity of the chain-induced kernel approximation, we derive a bound on the distance in Matérn-kernel GP posterior parameters. First, we tackle the case of singleoutput GP, i.e., for the l-th GP, and then generalize to multi-output, i.e., all L GPs.

Lemma 1 Let $\kappa _ { l } ( \cdot )$ be a Matérn-class kernel with length-scale $\ell _ { l } \ > \ 0$ and output scale $\gamma _ { l } ^ { 2 } > 0$ , and let $\alpha _ { l }$ be its Lipschitz constant on the range of distances induced by $X$ and the chain π. Define $[ K _ { l } ] _ { i j } = \kappa _ { l } \big ( \| x _ { i } - x _ { j } \| \big ) , [ K _ { \pi , l } ] _ { i j } = \kappa _ { l } \big ( | s _ { \pi ( i ) } - s _ { \pi ( j ) } | \big )$ and $\varepsilon _ { \pi } = \operatorname* { m a x } _ { i , j } \left| \| x _ { i } - \right.$ $x _ { j } \| - | s _ { \pi ( i ) } - s _ { \pi ( j ) } | |$ . Then,

$$
\| K _ { l } - K _ { \pi , l } \| _ { 2 } \le C \alpha _ { l } \varepsilon _ { \pi } .\tag{7}
$$

The proof is in Appendix E. Here C is the candidate-set size, so the bound scales with the number of points but vanishes as the chain distortion $\varepsilon _ { \pi } \to 0$ , that is, as the onedimensional ordering better preserves the input geometry. Under scalar observation noise of variance $\sigma _ { n } ^ { 2 }$ , the GP posterior parameters obtained by conditioning on observations y are $\mu = K ( K + \sigma _ { n } ^ { 2 } I ) ^ { - 1 } y$ and $\Sigma = K - K ( K + \sigma _ { n } ^ { 2 } I ) ^ { - 1 } K$ (Rasmussen and Williams, 2006), with $\mu _ { \pi }$ and $\Sigma _ { \pi }$ the same expressions using the chain-induced kernel of Eq. (6) in place of K. A resolvent perturbation argument converts a bound on $\lVert K - K _ { \pi } \rVert _ { 2 }$ into bounds on these parameters; we defer that step (Lemma 2 in $\operatorname { A p p e n d i x } \mathrm { F } )$ and state the multi-output LMC form directly. Writing $w _ { l } : = W _ { : , l } \in \mathbb { R } ^ { D }$ for the l-th column of the mixing matrix, the Euclidean and chain-induced LMC covariance matrices are defined as

$$
K = \sum _ { l = 1 } ^ { L } ( w _ { l } w _ { l } ^ { \top } ) \otimes K _ { l } , \qquad K _ { \pi } = \sum _ { l = 1 } ^ { L } ( w _ { l } w _ { l } ^ { \top } ) \otimes K _ { \pi , l } ,\tag{8}
$$

where $K _ { l } , K _ { \pi , l }$ are as in Lemma 1.

Theorem 1 Let $\eta = \| K - K _ { \pi } \| _ { 2 }$ . Assume scalar observation noise $\sigma _ { n } ^ { 2 } I$ σ<sup>2</sup>I and that each $\kappa _ { l }$ is a Matérn-class kernel with parameters $( \ell _ { l } , \gamma _ { l } ^ { 2 } )$ and Lipschitz constant $\alpha _ { l }$ . Then the multi-output GP posterior parameters obtained by conditioning on output matrix Y satisfy

$$
\| \mu - \mu _ { \pi } \| _ { 2 } \leq \left( \frac { \eta } { \sigma _ { n } ^ { 2 } } + \frac { \left( \| K \| _ { 2 } + \eta \right) \eta } { \sigma _ { n } ^ { 4 } } \right) \| Y \| _ { 2 } ,\tag{9}
$$

$$
\| \Sigma - \Sigma _ { \pi } \| _ { 2 } \leq \eta + \frac { \eta \left( \| K \| _ { 2 } + \| K _ { \pi } \| _ { 2 } \right) } { \sigma _ { n } ^ { 2 } } + \frac { \eta \| K \| _ { 2 } \| K _ { \pi } \| _ { 2 } } { \sigma _ { n } ^ { 4 } } ,\tag{10}
$$

with $\begin{array} { r } { \eta \leq C \varepsilon _ { \pi } \sum _ { l = 1 } ^ { L } \| w _ { l } \| _ { 2 } ^ { 2 } \alpha _ { l } } \end{array}$

The proof is in Appendix G. Thus, the diference between the per-latent bound and the total multi-output GP bound is the weighted sum of Lipschitz constants. For inputs that are roughly uniform in $[ 0 , 1 ] ^ { M }$ , the typical nearest-neighbor distance scales as $C ^ { - 1 / M }$ , so the greedy chain has unnormalised path length $\begin{array} { r } { \sum _ { i } \tilde { \Delta } _ { i } = \Theta ( C ^ { 1 - 1 / M } ) } \end{array}$ . The chain coordinate range $\mathrm { m a x } _ { i , j } \left| s _ { \pi ( i ) } - s _ { \pi ( j ) } \right|$ inherits this scaling, while the Euclidean diameter is bounded by $\sqrt { M }$ . Hence the worst-case distortion satisfies $\varepsilon _ { \pi } = O ( C ^ { 1 - 1 / { M } } )$ for $M \geq 2$ , while at $M { = } 1$ the sorted chain is an isometry of the Euclidean ordering and $\varepsilon _ { \pi } = 0$ . Plugging back into Theorem 1, $\begin{array} { r } { \eta = \mathcal { O } \big ( C ^ { 2 - 1 / M } \sum _ { l } \| w _ { l } \| _ { 2 } ^ { 2 } \alpha _ { l } \big ) } \end{array}$ This is informative for small M, but becomes uninformative as M grows. We stress that both bounds are worst case. $\varepsilon _ { \pi }$ is a maximum over all pairs, and η enters the posterior bounds through $\sigma _ { n } ^ { - 4 }$ . So at practical C and small observation noise they are loose and certify no numerically small posterior error.

Diferent starts of the chain produce diferent permutations π and hence diferent chain coordinate ranges. A classical result on the metric travelling salesman problem bounds this dependence.

Proposition 1 Define $T ( \pi ) : = \operatorname* { m a x } _ { i } s _ { \pi ( i ) }$ and let $\pi _ { 1 } , \pi _ { 2 }$ be two greedy nearest-neighbor chains over the same candidate set X from diferent starting points. Then

$$
T ( \pi _ { 1 } ) \ \leq \ { \frac { 1 } { 2 } } { \big ( } { \lceil \log _ { 2 } C \rceil } + 1 { \big ) } T ( \pi _ { 2 } ) .\tag{11}
$$

The proof is in Appendix H. Since $\varepsilon _ { \pi } \leq T ( \pi )$ for every chain, the bound of Theorem 1 varies by at most an ${ \mathcal { O } } ( \log C )$ factor across starting points. Appendix D measures the variation that actually occurs, and finds it far inside this envelope.

![](images/a3fcc21e465f9d6cdd722c834a0339ca26b64cfff6739f106018aa5b6400b6b9.jpg)  
Figure 2: Example Forney-style factor graph of SS-LMC, for $D = 3$ . The dashed box indicates a plate repeated over all points in the chain. Messages are carried upwards and joined into a total observation message 4, which updates the chain. If $y _ { i , 2 }$ and $y _ { i , 3 }$ are unobserved, the dashed messages (5–11) are absent and do not contribute.

## 4.1.2. State-Space GP with LMC

In state-space form, the full state $z _ { i } \in \mathbb { R } ^ { 2 L }$ is the concatenation of all latent states. The system matrices are block-diagonal:

$$
A _ { i } = \mathrm { b l k } \mathrm { d i a g } \big ( A _ { i } ^ { ( 1 ) } , . . . , A _ { i } ^ { ( L ) } \big ) , Q _ { i } = \mathrm { b l k } \mathrm { d i a g } \big ( Q _ { i } ^ { ( 1 ) } , . . . , Q _ { i } ^ { ( L ) } \big ) , P = \mathrm { b l k } \mathrm { d i a g } \big ( P _ { \infty } ^ { ( 1 ) } , . . , P _ { \infty } ^ { ( L ) } \big ) .\tag{12}
$$

The observation matrix combines the mixing matrix W with the per-latent observation vectors $h ^ { ( l ) } = [ 1 , 0 ] ^ { \top }$ (each picking out the function value from the l-th latent state):

$$
H = \left[ W _ { : , 1 } h ^ { ( 1 ) \top } \quad \cdot \cdot \cdot \quad W _ { : , L } h ^ { ( L ) \top } \right] \in \mathbb { R } ^ { D \times 2 L } ,\tag{13}
$$

so that $H z _ { i }$ yields a D-dimensional vector. Each output dimension is observed independently with per-output scalar noise variance $\tau _ { d } ^ { - 1 }$ . This per-output scalar formulation keeps the model fully linear-Gaussian, enabling exact single-pass inference without variational approximations. The full generative model is expressed as a factor graph for message passing (see Figure 2):

$$
p ( z _ { 0 } ) = \mathcal { N } ( z _ { 0 } | 0 , P ) , ~ p ( z _ { i } | z _ { i - 1 } ) = \mathcal { N } ( z _ { i } \mid A _ { i } z _ { i - 1 } , Q _ { i } ) , ~ p ( y _ { i d } \mid z _ { i } ) = \mathcal { N } ( y _ { i d } \mid e _ { d } ^ { \top } H z _ { i } , \tau _ { d } ^ { - 1 } ) ,\tag{14}
$$

where $e _ { d }$ is a canonical basis vector. Each output at each chain position is an independent scalar observation factor. The observation factor for output d at chain position i is included in the joint only when $O _ { i , d } = 1$ , and entries with $O _ { i , d } = 0$ contribute no factor so that predictions flow through the graph unafected.

## 4.2. Inference

Because all noise precisions $\tau _ { d }$ are fixed scalars, the model is fully linear-Gaussian. Inference is a single forward-backward pass (exact Kalman smoother), automated by RxInfer.jl. After the candidate chain has been constructed, the per-step inference cost is $\mathcal { O } ( C ( D L ^ { 2 } + L ^ { 3 } ) )$ for fixed Matérn state dimension. The block-diagonal structure of $A _ { i }$ and $Q _ { i }$ keeps the latent dynamics sparse, but the LMC observation matrix H couples the L latent processes, so exact inference maintains dense Gaussian beliefs over the joint 2L-dimensional state. The D per-output scalar observation factors contribute $\mathcal { O } ( D L ^ { 2 } )$ work per chain position, and the dense Gaussian state updates contribute the $\mathcal { O } ( L ^ { 3 } )$ term. With partial observations, the efective D at each chain position may be smaller, reducing the cost further; we detail this modularity next.

Partial Observations via Modularity In multi-output settings it is common for different outputs to be observed at diferent inputs. In sensor networks, for instance, sensors at diferent locations measure diferent quantities at diferent times (Liu et al., 2015). The mask O of Section 2 encodes exactly this pattern, with $O _ { i , d } = 0$ marking unobserved entries.

The factor graph formulation handles arbitrary masks natively. Each potential observation $y _ { i , d }$ corresponds to an independent scalar factor, and the local update at chain position i only fires the factors for which $O _ { i , d } = 1$ . No covariance-matrix restructuring is needed and no code path changes, so the same model and the same inference call produce posteriors regardless of O. The local message count drops when O has zeros, although fixed overheads may dominate at moderate D. Figure 2 visualizes this for $O _ { i , 2 } = O _ { i , 3 } = 0$ , where the dashed messages (5–11) are absent and the subgraph drops from 11 messages to 4.

This contrasts with dense kernel-matrix GP baselines, where partial observations require explicitly constructing or slicing a variable-size $P \times P$ kernel matrix from the P observed output–point pairs. The representational model is the same, but the dense covariance implementation has cost $\mathcal { O } ( P ^ { 3 } )$ coupled to the number of observed entries.

## 5. Experiments

We compare the proposed state-space LMC (SS-LMC) against an exact kernel-matrix LMC (KM-LMC), a sparse-variational inducing-point LMC (SVGP-LMC), and a nearest-neighbor LMC (NNGP-LMC) in two studies: an input-dimension sweep on a synthetic sensor network benchmark that probes the chain approximation (Section 5.1), and a real-data forecasting task on ETTh1 that probes the cost of partial-observation updating (Section 5.2). All methods use identical LMC model structure: the same mixing matrix W, length-scales $\ell _ { l } ,$ output scales $\gamma _ { l } ^ { 2 }$ , and fixed diagonal noise. The comparison isolates the efect of the inference method and candidate-chain approximation under controlled hyperparameters, reported through held-out posterior-quality metrics and wall-clock time.

RxInfer.jl is used for a reactive message passing implementation (RMP)<sup>1</sup> (Bagaev et al., 2023). All timings compare CPU implementations in the same pipeline on a MacBook Pro with Apple M1 Pro CPU (8 cores) and 32 GB RAM.

## 5.1. Accuracy as a Function of Input Dimensionality

We probe the nearest-neighbor chain approximation directly by sweeping the input dimension M on a synthetic sensor network benchmark. Each of the M input coordinates is one weather station at a fixed location, and the D=3 outputs are spatially weighted, saturating combinations of the per-station readings, so increasing M enlarges the input geometry that the one-dimensional chain must compress rather than merely duplicating information. We draw a candidate set of size C=2000, observe a uniformly random 50% as training data and hold out the remaining 50% for evaluation, and fit four methods with identical LMC structure $( L { = } 2 , \ell _ { l } \in \{ 1 . 0 , 2 . 0 \} , \gamma _ { l } ^ { 2 } \in \{ 2 . 0 , 1 . 0 \}$ , fixed diagonal noise): the proposed SS-LMC, an exact kernel-matrix KM-LMC, a sparse-variational SVGP-LMC with 64 inducing points per latent, and an NNGP-LMC conditioning each position on its k=20 nearest locations. For each method we report held-out RMSE and mean negative log-likelihood (MNLL) of the held-out observations, alongside the wall clock runtime. We sweep M ∈ {2, 4, 8, 16, 32} over 100 seeds and report mean ± standard deviation.

Appendix A reports the nearest-neighbor chain stretch ∆, i.e., the consecutive interpoint distance between chain-ordered candidates, along with the diference in predictive performance between SS-LMC and KM-LMC.

Results. Figure 3 reports held-out accuracy as M grows, now alongside the exact kernelmatrix posterior, the NNGP-LMC and the SVGP-LMC baseline. At low input dimension the chain preserves the local geometry (mean $\Delta \approx 0 . 1 0$ at M=2) and SS-LMC closely tracks the exact KM-LMC posterior: held-out RMSE is 0.061 versus 0.058 at $M { = } 2$ and 0.055 versus 0.046 at M=4, with SVGP-LMC in the same range. As M increases the chain stretches (mean $\Delta$ rises to 5.6 and max $\Delta$ to 11.2 at M=32) and every method’s error grows, since at fixed C the regression problem itself becomes harder. The two approximations degrade gracefully rather than catastrophically: at M=32 SS-LMC reaches RMSE 0.531 and SVGP-LMC 0.495, against 0.417 for exact KM-LMC and 0.524 for NNGP-LMC, and the SS-LMC–KM-LMC RMSE gap stays $\leq 0 . 1 2$ throughout. SS-LMC’s MNLL lies above the exact baseline across the range and stays close to SVGP-LMC’s (1.29 against 1.26 at M=32). NNGP-LMC shows the sharpest dependence on input dimension: it is indistinguishable from the exact posterior at M=2 (RMSE 0.0584 for both) and close at $M { = } 4$ where k=20 neighboring locations capture nearly all the relevant correlation, but by M=32 its MNLL is the worst of the four (1.50). Cost separates the methods far more sharply than accuracy does: one smoothing sweep takes 0.015–0.017 s and is flat in M, as $\mathcal { O } ( C ( D L ^ { 2 } + L ^ { 3 } ) )$ predicts, against 1.27–1.36 s for exact KM-LMC, 0.07–0.17 s for SVGP-LMC and 0.41–0.51 s for NNGP-LMC.

## 5.2. Cost of Partial Observation Updating

We forecast the ETTh1 dataset (Zhou et al., 2021) in a multidimensional-input regime that exercises the chain approximation on real data. We split the seven channels into an M=3 input (the useful-load measurements HUFL, MUFL, LUFL) and D=4 correlated outputs (the useless-load channels HULL, MULL, LULL and oil temperature OT), and order the inputs jointly with the nearest-neighbor chain so that SS-LMC performs O(N) message passing on three-dimensional inputs. We take a contiguous block of 2N rows, train on the first N with a training mask $O _ { i , d }$ ∼ Bernoulli(1 − p) drawn i.i.d. across (i, d), and forecast the fully held-out next N rows. We compare four methods with identical LMC structure $( L { = } 3 , \ell _ { l } \in \{ 0 . 5 , 1 , 2 \} , \gamma _ { l } ^ { 2 } { = } 1$ , fixed diagonal noise $R { = } 0 . 2 )$ : the proposed SS LMC by reactive message passing, an exact KM-LMC by covariance restructuring, a sparsevariational SVGP-LMC with 64 inducing points per latent, and an NNGP-LMC (Datta et al., 2016; Katzfuss and Guinness, 2021) conditioning each position on its $k { = } 2 0$ nearest locations (kD=80 responses, matched to SVGP-LMC’s budget). KM-LMC forms the full structured LMC kernel over the observed (input, output) pairs and refactorizes; because this dense (DN)-scale factorization is memory-bound, we run it only up to $N { = } 2 0 0 0$ . SS-LMC simply omits the local observation factor of each missing entry, and SVGP-LMC sums per-datum evidence terms over the observed entries. We report held-out forecast MNLL and RMSE and fit+forecast wall-clock time, sweeping the window length $N \in \{ 5 0 0 , 1 0 0 0 , 2 0 0 0 , 4 0 0 0 , 8 0 0 0 \}$ at $\scriptstyle p = 0 . 3$ and the dropout rate $p \in \{ 0 . 1 , 0 . 3 , 0 . 5 , 0 . 7 \}$ at $N { = } 2 0 0 0 .$ , over 10 seeds (mean). The complementary M=1 temporal case, where the natural ordering makes the SS-LMC and KM-LMC posteriors coincide exactly so the comparison is purely computational, is reported in Appendix B.

![](images/3d4ece8525f76ce28a8155e324c486c71053c51e47fab63c48f9adcaf9d545cb.jpg)

![](images/26de11d0fd9cce0b89fdb766e8a31cc52c888969393f15c26b15f6bb7d0d9a28.jpg)

![](images/6ef46fa712c339f32544a2de57d49a06eda988b6e06f197d7c9780db4376f2b8.jpg)  
Figure 3: Input-dimension sweep on the synthetic sensor network (C=2000, D=3, L=2, 50/50 train/test, 100 seeds; mean ± std).

Results. Figure 4 reports the four methods across window length and dropout. Because the input is now three-dimensional, the chain is an approximation and SS-LMC no longer coincides with KM-LMC, but its forecast accuracy stays close: at N=2000 the held-out RMSE is 8.17 for SS-LMC against 7.98 for exact KM-LMC, 8.10 for SVGP-LMC and 7.98 for NNGP-LMC, and at N=8000 (beyond KM-LMC’s reach) SS-LMC reaches RMSE 5.15 against 4.95 for SVGP-LMC and 5.19 for NNGP-LMC, with SVGP-LMC holding a small MNLL advantage throughout and SS-LMC ahead of KM-LMC and NNGP-LMC on MNLL. The cost, however, separates the methods sharply. KM-LMC’s covariance restructuring scales cubically $( 0 . 6 2 \to 2 . 5 3 \to 1 3 . 2 \mathrm { s }$ for $N { = } 5 0 0 \to 2 0 0 0 )$ and is infeasible beyond N=2000, whereas SS-LMC’s single forward–backward sweep grows linearly and reaches 0.16 s at N=8000, 7.6× below the inducing-point SVGP-LMC (1.25 s) and 38× below NNGP-LMC (6.16 s). The dropout sweep shows the complementary efect: SS-LMC’s cost is essentially invariant to the missingness rate (∼0.036 s across p), since missing entries merely omit local factors, while KM-LMC’s cost falls with sparser data (18.5 → 5.7 s as p: $0 . 1  0 . 7 )$ . SVGP-LMC stays near 0.3 s and NNGP-LMC from 1.33 to 0.61 s. Appendix C recasts both views as an accuracy–cost frontier at fixed window length. Handling partial observations therefore requires no covariance-matrix restructuring in SS-LMC, and the chain approximation costs little accuracy relative to an exact, a sparse-variational and a nearest-neighbor LMC baseline.

![](images/74531192efef3883f1a9f1e6dc2dae4e1eb958d0a5858dd447d767a0edc607ca.jpg)

![](images/f8741e8ec7f3e632b3355b494c1c70790257ba0d5753e12c992004d343c290fc.jpg)

![](images/aa99e8deb4a99beb0a741526215b03eba27b08480db480d6892774a74eb279a3.jpg)  
Figure 4: ETTh1 forecasting at M=3, D=4, L=3, 10 seeds. KM-LMC runs only to N=2000. Timings exclude the one-of chain construction (cf. Section 6).

Table 1: Per-step computational cost of the baselines and the proposed SS-LMC. m: SVGP inducing points per latent; k: NNGP conditioning locations, each carrying D outputs; $\textstyle P = \sum _ { i , d } O _ { i , d } .$ observed output–input pairs.
<table><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Naive MOGP</td><td rowspan=1 colspan=1>KM-LMC</td><td rowspan=1 colspan=1>SVGP-LMC</td><td rowspan=1 colspan=1>NNGP-LMC</td><td rowspan=1 colspan=1>SS-LMC (ours)</td></tr><tr><td rowspan=1 colspan=1>Per-step inferenceChain constructionPartial observationsTotal over N steps</td><td rowspan=1 colspan=1>O((DN)3)O((DN)3) $\mathcal { O } ( D ^ { 3 } N ^ { 4 } )$ </td><td rowspan=1 colspan=1> $\mathcal { O } ( L N ^ { 3 } + N D L ^ { 2 } )$  $\mathcal { O } ( P ^ { 3 } )$  $\mathcal { O } ( L N ^ { 4 } { + } N ^ { 2 } D L ^ { 2 } )$ </td><td rowspan=1 colspan=1> $\mathcal { O } ( L ( N m ^ { 2 } + m ^ { 3 } ) )$  $\mathcal { O } ( L ( P m ^ { 2 } + m ^ { 3 } ) )$  $\mathcal { O } ( L N m ^ { 2 } )$ </td><td rowspan=1 colspan=1>O(N(kD)3)O(C2) brute force $\mathcal { O } ( N ( k D ) ^ { 3 } )$  $\mathcal { O } ( C ^ { 2 } { + } \dot { N } ^ { 2 } ( \dot { k } \dot { D } ) ^ { 3 } )$ </td><td rowspan=1 colspan=1> $\mathcal { O } ( C ( D L ^ { 2 } + L ^ { 3 } ) )$ O(C2) brute force $\mathcal { O } ( C ( D L ^ { 2 } + L ^ { 3 } ) )$  $\mathcal { O } ( C ^ { 2 } { + } N C ( D L ^ { 2 } { + } L ^ { 3 } ) )$ </td></tr></table>

## 6. Discussion

SS-LMC has $\mathcal { O } ( C ( D L ^ { 2 } + L ^ { 3 } ) )$ per-step inference after chain construction (Table 1). NNGP-LMC (Datta et al., 2016; Katzfuss and Guinness, 2021) shares the nearest-neighbor ingredient but conditions each prediction on k neighbor locations instead of propagating one state along a chain, and pays the same $\mathcal { O } ( C ^ { 2 } )$ brute-force construction (Finley et al., 2019, §S1). Chain construction is preprocessing that the kernel-matrix and inducing-point baselines avoid, but it depends only on the candidate inputs, and not on Y , the mask O, or the hyperparameters. So it is built once and reused by every later inference call. Writing $c _ { \pi }$ for its cost, R reuses favor SS-LMC once $R > c _ { \pi } / ( t _ { \mathrm { b a s e } } - t _ { \mathrm { S S } } )$ . Since the one-of chain cost is well below the per-fit gap to either baseline, the construction is recovered within a single fit, and repeated queries, hyperparameter-learning iterations, and streaming updates amortize it further. The nearest-neighbor chain replaces Euclidean by path distances along a one-dimensional Markov chain, and the input-dimension sweep (Figure 3) shows the resulting tradeof: fidelity is high at low M and degrades gradually as the chain stretches. The method is therefore strongest for low-input-dimensional candidate sets.

Nguyen et al. (2025) also decomposes a Gaussian process into a factor graph formulation, but utilizes variational approximations instead of the state-space formulation. It is a factor graph treatment of the sparse variational Gaussian process, and has the same computational and memory complexity at similar accuracy in regression and classification tasks.

Hyperparameter Learning. Learning the hyperparameters $( W , \ell _ { l } , \gamma _ { l } ^ { 2 } )$ , which are fixed throughout this work, interacts asymmetrically with the graph. The mixing matrix enters only through H and so touches the C observation blocks alone, whereas $\ell _ { l }$ and $\gamma _ { l } ^ { 2 }$ enter through $A _ { i }$ and $Q _ { i }$ and re-parameterize every transition factor: unlike the missingness mask, a hyperparameter update is not a local graph edit. The ordering is spared, since π and the $\Delta _ { i }$ depend only on the input geometry and are computed once and reused across iterations. Expectation maximization is then natural, alternating one $\mathcal { O } ( C ( D L ^ { 2 } + L ^ { 3 } ) )$ smoothing sweep with automatic-diferentiation gradient steps on the hyperparameters, taking the free energy returned by that same sweep as the objective, with banded linear algebra for the gradients (Durrande et al., 2019); particle MCMC (Svensson et al., 2016) and recursive Bayesian autoregression (Kouw, 2025) are alternatives. Learned length-scales are moreover fit to chain rather than Euclidean distances, so they absorb part of the chain stretch and inflate $\alpha _ { l } .$ and with it the bound of Theorem 1.

Limitations. Firstly, the greedy nearest-neighbor chain is path-dependent. However, the efect is, so far, empirically negligible: held-out RMSE varies by <0.1% across 20 starting points (within the O(log C) bound of Proposition 1), and even adversarially diverse starts leave it unchanged (Appendix D). Secondly, the ETT study (Section 5.2) exercises an M=3 input on real data, but a systematic evaluation at higher input dimension on real data is left to future work. Thirdly, the bounds of Lemma 1 and Theorem 1 are illustrative from a theoretical perspective but do not furnish error estimates one would use in practice. Lastly, the method is currently limited to Matérn-class covariances that admit state-space representations.

## 7. Conclusion

We presented low-input-dimensional multi-output GP inference as message passing on a Forney-style factor graph, combining state-space GPs, LMC, and reactive message passing. On a synthetic sensor-network benchmark it tracks the exact kernel-matrix posterior at low input dimension and degrades gracefully as the chain stretches with input dimension. On real ETTh1 forecasting under random per-(row, output) dropout with a three-dimensional input, SS-LMC matches both baselines in forecast accuracy to within a small margin, while its inference cost grows only linearly in the window length and is invariant to the dropout rate. The factor graph handles partial observations without covariance-matrix restructuring.

## Acknowledgments

This publication is part of the ROBUST project with project number KICH3.LTP.20.006, which is (partly) financed by the Dutch Research Council (NWO), GN Hearing, and the Dutch Ministry of Economic Afairs and Climate Policy (EZK) under the program LTP KIC 2020-2023. This project is also partly financed by Holland High Tech with PPS funding for the AUTO-AR project RVO TKI2112P09.

## References

Mauricio A Alvarez and Neil D Lawrence. Computationally eficient convolved multiple output Gaussian processes. The Journal of Machine Learning Research, 12:1459–1500, 2011.

Mauricio A. Álvarez, Lorenzo Rosasco, and Neil D. Lawrence. Kernels for Vector-Valued Functions: A Review. Foundations and Trends in Machine Learning, 4(3):195–266, June 2012. ISSN 1935-8237. doi: 10.1561/2200000036.

Dmitry Bagaev and Bert De Vries. Reactive Message Passing for Scalable Bayesian Inference. Scientific Programming, 2023:1–26, May 2023. ISSN 1875-919X, 1058-9244. doi: 10.1155/ 2023/6601690.

Dmitry Bagaev, Albert Podusenko, and Bert de Vries. RxInfer: A Julia package for reactive real-time Bayesian inference. Journal of Open Source Software, 8(84):5161, April 2023. ISSN 2475-9066. doi: 10.21105/joss.05161.

Edwin V Bonilla, Kian Ming A. Chai, and Christopher K. I. Williams. Multi-task Gaussian Process Prediction. In Advances in Neural Information Processing Systems, volume 20. Curran Associates, Inc., 2007.

Wessel Bruinsma, Eric Perim, William Tebbutt, Scott Hosking, Arno Solin, and Richard Turner. Scalable Exact Inference in Multi-Output Gaussian Processes. In Proceedings of the 37th International Conference on Machine Learning, pages 1190–1201. PMLR, November 2020.

Thomas H Cormen, Charles E Leiserson, Ronald L Rivest, and Cliford Stein. Introduction to Algorithms. MIT press, 2022.

Abhirup Datta, Sudipto Banerjee, Andrew O. Finley, and Alan E. Gelfand. Hierarchical Nearest-Neighbor Gaussian Process Models for Large Geostatistical Datasets. Journal of the American Statistical Association, 111(514):800–812, April 2016. ISSN 0162-1459. doi: 10.1080/01621459.2015.1044091.

Nicolas Durrande, Vincent Adam, Lucas Bordeaux, Stefanos Eleftheriadis, and James Hensman. Banded Matrix Operators for Gaussian Markov Models in the Automatic Diferentiation Era. In Proceedings of the Twenty-Second International Conference on Artificial Intelligence and Statistics, pages 2780–2789. PMLR, April 2019.

Andrew O. Finley, Abhirup Datta, Bruce D. Cook, Douglas C. Morton, Hans E. Andersen, and Sudipto Banerjee. Eficient Algorithms for Bayesian Nearest Neighbor Gaussian Processes. Journal of Computational and Graphical Statistics, 28(2):401–414, April 2019. ISSN 1061-8600. doi: 10.1080/10618600.2018.1537924.

Oliver Hamelijnck, William Wilkinson, Niki Loppi, Arno Solin, and Theodoros Damoulas. Spatio-Temporal Variational Gaussian Processes. In Advances in Neural Information Processing Systems, volume 34, pages 23621–23633. Curran Associates, Inc., 2021.

Jouni Hartikainen and Simo Särkkä. Kalman filtering and smoothing solutions to temporal Gaussian process regression models. In 2010 IEEE International Workshop on Machine Learning for Signal Processing, pages 379–384, Kittila, Finland, August 2010. IEEE. ISBN 978-1-4244-7875-0. doi: 10.1109/MLSP.2010.5589113.

Nicolas Heess, Daniel Tarlow, and John Winn. Learning to Pass Expectation Propagation Messages. In Advances in Neural Information Processing Systems, volume 26. Curran Associates, Inc., 2013.

Roger A Horn and Charles R Johnson. Matrix Analysis. Cambridge university press, 2012.

Matthias Katzfuss and Joseph Guinness. A General Framework for Vecchia Approximations of Gaussian Processes. Statistical Science, 36(1):124–141, 2021. ISSN 0883-4237.

Wouter M. Kouw. Bayesian autoregression to optimize temporal Matérn kernel Gaussian process hyperparameters. In Proceedings of the First International Conference on Probabilistic Numerics, pages 122–130. PMLR, August 2025.

Frank R. Kschischang, Brendan J. Frey, and H.-A. Loeliger. Factor graphs and the sumproduct algorithm. IEEE Transactions on information theory, 47(2):498–519, 2001. doi: 10.1109/18.910572.

Weihan Li, Yule Wang, Chengrui Li, and Anqi Wu. Learning time-varying multi-region brain communications via scalable markovian gaussian processes, 2025.

Finn Lindgren, Håvard Rue, and Johan Lindström. An explicit link between Gaussian fields and Gaussian Markov random fields: The stochastic partial diferential equation approach. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 73 (4):423–498, 2011. ISSN 1467-9868. doi: 10.1111/j.1467-9868.2011.00777.x.

Qinyuan Liu, Zidong Wang, Xiao He, and D. H. Zhou. Event-Based Recursive Distributed Filtering Over Wireless Sensor Networks. IEEE Transactions on Automatic Control, 60 (9):2470–2475, September 2015. ISSN 1558-2523. doi: 10.1109/TAC.2015.2390554.

H.-A. Loeliger. An introduction to factor graphs. IEEE Signal Processing Magazine, 21(1): 28–41, January 2004. ISSN 1558-0792. doi: 10.1109/MSP.2004.1267047.

Hans-Andrea Loeliger, Justin Dauwels, Junli Hu, Sascha Korl, Li Ping, and Frank R. Kschischang. The Factor Graph Approach to Model-Based Signal Processing. Proceedings of the IEEE, 95(6):1295–1322, June 2007. ISSN 0018-9219. doi: 10.1109/JPROC.2007.896497.

Hoang Minh Huu Nguyen, İsmail Şenöz, and Bert De Vries. A Factor Graph Approach to Variational Sparse Gaussian Processes. IEEE Open Journal of Signal Processing, 6: 815–837, 2025. ISSN 2644-1322. doi: 10.1109/OJSP.2025.3585440.

Francesco A. N. Palmieri, Krishna R. Pattipati, Giovanni Di Gennaro, Giovanni Fioretti, Francesco Verolla, and Amedeo Buonanno. A Unifying View of Estimation and Control Using Belief Propagation With Application to Path Planning. IEEE Access, 10:15193– 15216, 2022. ISSN 2169-3536. doi: 10.1109/ACCESS.2022.3148127.

Carl Edward Rasmussen and Christopher K. I. Williams. Gaussian Processes for Machine Learning. MIT Press, 2006.

Daniel J. Rosenkrantz, Richard E. Stearns, and Philip M. Lewis, II. An Analysis of Several Heuristics for the Traveling Salesman Problem. SIAM Journal on Computing, 6(3):563– 581, September 1977. ISSN 0097-5397. doi: 10.1137/0206041.

Simo Särkkä. Bayesian Filtering and Smoothing. Cambridge University Press, October 2013. ISBN 978-1-107-03065-7.

Simo Särkkä and Arno Solin. Applied Stochastic Diferential Equations. Institute of Mathematical Statistics Textbooks. Cambridge University Press, Cambridge, 2019. ISBN 978- 1-316-51008-7. doi: 10.1017/9781108186735.

Simo Särkkä, Arno Solin, and Jouni Hartikainen. Spatiotemporal Learning via Infinite-Dimensional Bayesian Filtering and Smoothing: A Look at Gaussian Process Regression Through Kalman Filtering. IEEE Signal Processing Magazine, 30(4):51–61, July 2013. ISSN 1053-5888. doi: 10.1109/MSP.2013.2246292.

İsmail Şenöz, Thijs van de Laar, Dmitry Bagaev, and Bert de Vries. Variational message passing and local constraint manipulation in factor graphs. Entropy. An International and Interdisciplinary Journal of Entropy and Information Studies, 23(7):807, 2021.

Alex J. Smola, S. V. N. Vishwanathan, and Eleazar Eskin. Laplace propagation. In NIPS, pages 441–448. The MIT Press Cambridge, MA, USA, 2004.

Andreas Svensson, Arno Solin, Simo Särkkä, and Thomas Schön. Computationally Eficient Bayesian Learning of Gaussian Process State Space Models. In Proceedings of the 19th International Conference on Artificial Intelligence and Statistics, pages 213–221. PMLR, May 2016.

Will Tebbutt, Arno Solin, and Richard E. Turner. Combining pseudo-point and state space approximations for sum-separable Gaussian Processes. In Proceedings of the Thirty-Seventh Conference on Uncertainty in Artificial Intelligence, pages 1607–1617. PMLR, December 2021.

A. V. Vecchia. Estimation and Model Identification for Continuous Spatial Processes. Journal of the Royal Statistical Society: Series B (Methodological), 50(2):297–312, January 1988. ISSN 0035-9246. doi: 10.1111/j.2517-6161.1988.tb01729.x.

Haoyi Zhou, Shanghang Zhang, Jieqi Peng, Shuai Zhang, Jianxin Li, Hui Xiong, and Wancai Zhang. Informer: Beyond eficient transformer for long sequence time-series forecasting. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 35, pages 11106– 11115, 2021.

![](images/f127a8fbce4e4bbc94836eb2660402b7b6bc0872611ef6150a03998033cd77f2.jpg)

![](images/f1eaca13e1b792517b3b1b84ccc94e500b28e4461d52d1b718f02b80701bed29.jpg)

![](images/9520ee5065524830e41abd41b42a3ef2b90e575939ed9bb68abee2a4e91c6527.jpg)  
Figure 5: Chain stretch and approximation gap on the sensor network sweep $( C { = } 2 0 0 0 .$ D=3, L=2, 100 seeds). Left: nearest-neighbor chain stretch $\Delta$ (mean, max) vs. input dim M. Middle and right: held-out RMSE and MNLL gap vs. input dim $M ,$ , as SS-LMC minus exact KM-LMC. Diference is taken within each seed and then averaged $( \mathrm { m e a n } \pm \mathrm { s t d } )$

## Appendix A. Chain Stretch and the Gap to the Exact Posterior

Figure 5 reports the geometry the chain has to compress and what it costs against the exact posterior. The stretch $\Delta$ grows steadily with M (mean $0 . 1 0  5 . 6$ , max $4 . 5  1 1 . 2$ for $M \colon 2 \  \ 3 2 )$ , and the accuracy gap follows it. The diference between SS-LMC and KM-LMC within each seed rises monotonically in RMSE, from $0 . 0 0 3 \pm 0 . 0 0 2$ at M=2 to $0 . 1 1 4 \pm 0 . 0 3 5$ at $M { = } 3 2$ . The MNLL gap rises likewise, from $0 . 0 8 \pm 0 . 0 8$ to a maximum of $0 . 3 7 \pm 0 . 2 9$ at $M { = } 1 6$ , then falls back to $0 . 2 0 \pm 0 . 2 3$ at $M { = } 3 2$ , where the regression problem is hard enough at fixed $C$ that the exact posterior loses part of its advantage too.

## Appendix B. ETT Temporal (M=1) Case

As a complement to the multidimensional-input study of Section 5.2, we also forecast ETTh1 in the purely temporal setting: all seven channels are modeled jointly as D=7 correlated outputs over normalized time, so M=1 and the candidate chain reduces to the natural temporal ordering. Here SS-LMC and KM-LMC realize the same model, so their posteriors coincide to numerical precision at every window length and the comparison is purely computational. Figure 6 shows that SS-LMC’s cost grows linearly in the window length while KM-LMC’s covariance restructuring grows cubically, with SS-LMC reaching a 4.4× speedup at C=4000; SS-LMC’s cost is also invariant to the dropout rate, whereas KM-LMC’s falls with sparser data but stays tied to dense factorization. These timings come from a separate run and are comparable within the figure, not against Figure 4.

## Appendix C. Accuracy–Cost Frontier

Held-out score against fit+forecast wall clock at fixed window length separates the two accuracy measures of Section 5.2. Both windows shown are ones where exact KM-LMC still fits in memory, so every panel carries a complete four-method frontier. On RMSE (Figure $^ { 7 , }$ top row) the mean ordering shifts with the window: at C=1000 SVGP-LMC is on average both cheaper and more accurate than KM-LMC and NNGP-LMC, whereas at C=2000 the four means form a monotone frontier. The accuracy diferences are small against the seed spread, however: at C=2000 the entire 0.18 RMSE span from SS-LMC to KM-LMC is half the ±0.34 standard deviation across seeds, bought at a 367× cost ratio, and NNGP-LMC reaches KM-LMC’s mean RMSE at 15× less time. On MNLL (bottom row) SVGP-LMC and SS-LMC hold the frontier at both windows, with KM-LMC and NNGP-LMC both slower and less well calibrated on average.

![](images/ae17786d257fdec4daf78a26526ff691e0267cb4b537a77331903fb243436663.jpg)

![](images/213cdb89ddf9ac3f9240b3883888b18fb6b7e88e96c6afdf1e3a0fb1e3a6cc52.jpg)

![](images/63dfac5f3b3112d034f56cd5423eade441ac840a25e18b4226dc2527d9f14eb3.jpg)  
Figure 6: ETTh1 temporal forecasting (M=1 natural ordering, D=7, L=4, 3 seeds): train on the first half of a length-C window, forecast the held-out second half under per-(timestamp, output) dropout. At M=1 the SS-LMC and KM-LMC posteriors are identical, so the comparison is computational. Left: fit+forecast wall-clock vs. C at dropout p=0.3 (log scale). Middle: held-out MNLL vs. C. Right: time vs. p at $C { = } 1 0 0 0$ (log scale).

![](images/113ac7d73f946b8a48d6595b9bf1270ec2e4b9337d8ed4049f584b392e6d5cd7.jpg)

![](images/cc48f4fb516010b04402246b8a2d9729958433abb72ddebe7a834c00a94f398d.jpg)

![](images/b7b2f6113ef07dda5fdfafb88384f25632c0c37b24bfae8d6cf78ca5b6758c45.jpg)  
Fit+forecast time (s)

![](images/4a968151a24d49cad6831b86c38c0a89d3af1547c0a6268798d331472a33b23b.jpg)  
Fit+forecast time (s)  
Figure 7: Held-out score against fit+forecast wall clock on ETTh1 (M=3, D=4, L=3, dropout $\mathrm { \ p { = } 0 . 3 } .$ , 10 seeds). Lower-left is better. Top row: RMSE; bottom row: MNLL. Columns are $C { = } 1 0 0 0$ and $C { = } 2 0 0 0$ . Dashed line is Pareto front. Bars are ±1 standard deviation over seeds.

## Appendix D. Sensitivity to the Chain Starting Point

Proposition 1 bounds the worst-case variation across starting points. In this experiment, we measure the variation over 20 starts. Within a seed, the dropout mask and train/test split are fixed on physical points before ordering. W is fixed, so every start sees the same observations. Per start we take the median over 10 seeds, then the standard deviation across starts. Since that is itself an estimated dispersion, the error bars are a bootstrap 95% confidence interval for it (relative standard error $\approx 1 6 \%$ at 20 starts), so only well-separated points difer.

Sensitivity is governed by input dimension, not candidate-set size. Against M (Figure 8, left) the spread is flat at $2 . 5 { - } 4 . 0 \times 1 0 ^ { - 4 }$ for $M \ \leq \ 8$ and steps up roughly threefold at $M \geq 1 6$ mirroring the $\varepsilon _ { \pi } = \mathcal { O } ( C ^ { 1 - 1 / M } )$ compression of M-dimensional geometry into one chain coordinate. Against C (right) the intervals overlap, so the spread is flat in C. The relative efect stays negligible: on ETTh1 the coeficient of variation of held-out RMSE across starts is $\leq 0 . 0 9 \%$ at every C. MNLL, not plotted, is somewhat more sensitive, growing fourto sevenfold over the same range of M. Three checks support reading this as real but harmless path dependence: the permutation-invariant KM-LMC control varies 10–50× less. The even, random and farthest-point start schemes give the same chain diversity and spread, so a bad chain cannot be engineered from a bad start. $\varepsilon _ { \pi }$ does not predict per-start error, its bootstrap correlation intervals remain close to zero. The observed variation therefore sits far inside the envelope of Proposition 1, which holds but is conservative.

![](images/f689418ffded383714d3ea2b480c5a69ac2848e4d2ad33e1bde1bbaf6fd4ce87.jpg)

![](images/d357c1d0aff5364ff0302c240c4300f19af7e218129547db94432939a360dfaf.jpg)  
Figure 8: Spread of SS-LMC’s held-out RMSE across 20 chain starting points (median over 10 seeds per start, then standard deviation across starts). Bars are a bootstrap 95% confidence interval for that standard deviation. Left: vs. input dimension M on the synthetic sensor network (C=2000, D=3, L=2). Right: vs. candidate-set size C on ETTh1 (M=3, D=4, L=3, p=0.3).

## Appendix E. Proof of Lemma 1

Proof Matérn-class kernels are diferentiable away from the origin with bounded derivative on any compact range of distances, hence Lipschitz with a finite constant $\alpha _ { l }$ on the range induced by X and the chain $\pi .$ . For Matérn-3/2 in particular, the maximum of $| \kappa _ { l } ^ { \prime } ( d ) |$ is attained at $d = \ell _ { l } / \sqrt { 3 } .$ , giving $\alpha _ { l } = \sqrt { 3 } \gamma _ { l } ^ { 2 } / ( e \ell _ { l } )$ ; other smoothness orders give analogous closed forms. Lipschitz continuity then upper bounds the diference in kernel entries

$$
\begin{array} { r } { \big | \left[ K _ { l } - K _ { \pi , l } \right] _ { i j } \big | = \big | \kappa _ { l } ( \| x _ { i } - x _ { j } \| ) - \kappa _ { l } ( | s _ { \pi ( i ) } - s _ { \pi ( j ) } | ) \big | \leq \alpha _ { l } \big | \| x _ { i } - x _ { j } \| - | s _ { \pi ( i ) } - s _ { \pi ( j ) } | \big | } \end{array}\tag{15}
$$

This implies $\begin{array} { r } { \| K _ { l } - K _ { \pi , l } \| _ { \operatorname* { m a x } } \leq \alpha _ { l } \operatorname* { m a x } _ { i j } \big | \| x _ { i } - x _ { j } \| - | s _ { \pi ( i ) } - s _ { \pi ( j ) } | \big | = \alpha _ { l } \varepsilon _ { \pi } } \end{array}$ . For any $C \times C$ matrix A, $\| A \| _ { 2 } \leq \sqrt { \| A \| _ { 1 } \| A \| _ { \infty } } \leq C \| A \| _ { \operatorname* { m a x } }$ (Horn and Johnson, 2012). Hence $\lVert K _ { l } - K _ { \pi , l } \rVert _ { 2 } \leq C \alpha _ { l } \varepsilon _ { \pi }$ ■

## Appendix F. Lemma 2 and Its Proof

Lemma 2 Let K, $K _ { \pi } \succeq 0$ and $\eta = \| K - K _ { \pi } \| _ { 2 }$ . Under scalar observation noise $\sigma _ { n } ^ { 2 } I$ , the $G P$ posterior parameters obtained from K and from $K _ { \pi }$ satisfy

$$
\| \mu - \mu _ { \pi } \| _ { 2 } \leq \left( { \frac { \eta } { \sigma _ { n } ^ { 2 } } } + { \frac { ( \| K \| _ { 2 } + \eta ) \eta } { \sigma _ { n } ^ { 4 } } } \right) \| y \| _ { 2 }\tag{16}
$$

$$
\| \Sigma - \Sigma _ { \pi } \| _ { 2 } \leq \eta + \frac { \eta ( \| K \| _ { 2 } + \| K _ { \pi } \| _ { 2 } ) } { \sigma _ { n } ^ { 2 } } + \frac { \eta \| K \| _ { 2 } \| K _ { \pi } \| _ { 2 } } { \sigma _ { n } ^ { 4 } } .\tag{17}
$$

Proof Let $B = K + \sigma _ { n } ^ { 2 } I , B _ { \pi } = K _ { \pi } + \sigma _ { n } ^ { 2 } I$ . Since $K , K _ { \pi } \succeq 0$ , we have B, $B _ { \pi } \succeq \sigma _ { n } ^ { 2 } I$ , so $\| B ^ { - 1 } \| _ { 2 } , \| B _ { \pi } ^ { - 1 } \| _ { 2 } \leq \sigma _ { n } ^ { - 2 }$ . The resolvent identity

$$
B ^ { - 1 } - B _ { \pi } ^ { - 1 } = B ^ { - 1 } ( B _ { \pi } - B ) B _ { \pi } ^ { - 1 } = B ^ { - 1 } ( K _ { \pi } - K ) B _ { \pi } ^ { - 1 }\tag{18}
$$

gives $\| B ^ { - 1 } - B _ { \pi } ^ { - 1 } \| _ { 2 } \leq \eta \sigma _ { n } ^ { - 4 }$ . Together with submultiplicativity gives

$$
\begin{array} { r } { \| B ^ { - 1 } - B _ { \pi } ^ { - 1 } \| _ { 2 } \leq \| B ^ { - 1 } \| _ { 2 } \| K - K _ { \pi } \| _ { 2 } \| B _ { \pi } ^ { - 1 } \| _ { 2 } \leq \eta \sigma _ { n } ^ { - 4 } . } \end{array}\tag{19}
$$

Adding and subtracting $K _ { \pi } B ^ { - 1 } y$ inside $\mu - \mu _ { \pi } = K B ^ { - 1 } y - K _ { \pi } B _ { \pi } ^ { - 1 } y$ splits the diference into a prior-diference term and an inverse-diference term:

$$
\mu - \mu _ { \pi } = \left( K - K _ { \pi } \right) B ^ { - 1 } y \ + \ K _ { \pi } \left( B ^ { - 1 } - B _ { \pi } ^ { - 1 } \right) y .\tag{20}
$$

Taking norms, applying submultiplicativity together with $\| B ^ { - 1 } \| _ { 2 } \leq \sigma _ { n } ^ { - 2 }$ and the resolvent bound above, and using $\| K _ { \pi } \| _ { 2 } \le \| K \| _ { 2 } + \eta$

$$
\| \mu - \mu _ { \pi } \| _ { 2 } \leq \frac { \eta } { \sigma _ { n } ^ { 2 } } \| y \| _ { 2 } + \frac { \left( \| K \| _ { 2 } + \eta \right) \eta } { \sigma _ { n } ^ { 4 } } \| y \| _ { 2 } .\tag{21}
$$

Note that $\Sigma - \Sigma _ { \pi } = ( K - K _ { \pi } ) - \left( K B ^ { - 1 } K - K _ { \pi } B _ { \pi } ^ { - 1 } K _ { \pi } \right)$ . If the second piece is expanded by inserting two telescoping terms, $\pm K _ { \pi } B ^ { - 1 } K$ and $\pm K _ { \pi } B _ { \pi } ^ { - 1 } K$ , then we have:

$$
K B ^ { - 1 } K - K _ { \pi } B _ { \pi } ^ { - 1 } K _ { \pi } = ( K - K _ { \pi } ) B ^ { - 1 } K + K _ { \pi } ( B ^ { - 1 } - B _ { \pi } ^ { - 1 } ) K + K _ { \pi } B _ { \pi } ^ { - 1 } ( K - K _ { \pi } ) .\tag{22}
$$

Taking norms termwise and using the same norm bounds,

$$
\| \Sigma - \Sigma _ { \pi } \| _ { 2 } \leq \eta + \frac { \eta \left( \| K \| _ { 2 } + \| K _ { \pi } \| _ { 2 } \right) } { \sigma _ { n } ^ { 2 } } + \frac { \eta \| K \| _ { 2 } \| K _ { \pi } \| _ { 2 } } { \sigma _ { n } ^ { 4 } } .\tag{23}
$$

## Appendix G. Proof of Theorem 1

Proof Applying Lemma 1 to each Matérn-class latent kernel gives $\| K _ { l } - K _ { \pi , l } \| _ { 2 } \le C \alpha _ { l } \varepsilon _ { \pi }$ By subadditivity of the spectral norm, $\| A \otimes B \| _ { 2 } = \| A \| _ { 2 } \| B \| _ { 2 }$ and $\| w _ { l } w _ { l } ^ { \top } \| _ { 2 } = \| w _ { l } \| _ { 2 } ^ { 2 }$ (Horn and Johnson, 2012). Thus,

$$
\eta = \| K - K _ { \pi } \| _ { 2 } \leq \sum _ { l = 1 } ^ { L } \| w _ { l } w _ { l } ^ { \top } \| _ { 2 } \| K _ { l } - K _ { \pi , l } \| _ { 2 } \leq C \varepsilon _ { \pi } \sum _ { l = 1 } ^ { L } \| w _ { l } \| _ { 2 } ^ { 2 } \alpha _ { l } .\tag{24}
$$

Since $K , K _ { \pi } \succeq 0$ , applying Lemma 2 with this η yields the stated bounds on $\| \mu - \mu _ { \pi } \| _ { 2 }$ and $\| \Sigma - \Sigma _ { \pi } \| _ { 2 }$ ■

## Appendix H. Proof of Proposition 1

Proof Let $\begin{array} { r } { T ( \pi ) : = \operatorname* { m a x } _ { i } s _ { \pi ( i ) } = \sum _ { i = 2 } ^ { C } \Delta _ { i } } \end{array}$ denote the chain coordinate range from starting point $\pi ( 1 )$ . Since $s _ { \pi ( 1 ) } = 0$ and the chain visits each point of X exactly once, $T ( \pi )$ equals the total length of the greedy nearest-neighbor Hamiltonian path on X initiated at $\pi ( 1 )$ . Let $T ^ { \star }$ denote the length of the shortest Hamiltonian path on $X$ under the Euclidean metric.

Rosenkrantz et al. (1977) prove that, for any metric instance on C points, the greedy nearest-neighbor heuristic produces a Hamiltonian path of length at most

$$
T ( \pi ) ~ \le ~ \frac { 1 } { 2 } \bigl ( \lceil \log _ { 2 } C \rceil + 1 \bigr ) T ^ { \star }\tag{25}
$$

regardless of the starting point. Applying this upper bound to $\pi _ { 1 }$ and the trivial lower bound $T ^ { \star } \leq T ( \pi _ { 2 } )$ to $\pi _ { 2 }$ gives the claimed inequality

$$
T ( \pi _ { 1 } ) \ \leq \ { \frac { 1 } { 2 } } { \big ( } { \lceil \log _ { 2 } C \rceil + 1 } { \big ) } T ( \pi _ { 2 } ) .\tag{26}
$$

For any i, j, $\| x _ { i } - x _ { j } \| \leq \mathrm { d i a m } ( X ) \leq T ( \pi )$ , where the last inequality holds because any Hamiltonian path on X is at least as long as the Euclidean diameter. Furthermore, $| s _ { \pi ( i ) } - s _ { \pi ( j ) } | \le T ( \pi )$ and thus, $\varepsilon _ { \pi } \leq T ( \pi )$ . Hence the upper bound of Theorem 1 inherits the same ${ \mathcal { O } } ( \log C )$ variation across choices of starting point. 7