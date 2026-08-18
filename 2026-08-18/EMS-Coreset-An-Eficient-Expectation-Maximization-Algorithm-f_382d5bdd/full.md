# EMS Coreset: An Eficient Expectation-Maximization Algorithm for Sinkhorn Coreset

Haoyun Yin<sup>∗</sup> Chuanhui Liu<sup>∗</sup> Xiao Wang

Department of Statistics, Purdue University West Lafayette, IN 47906, USA yin164@purdue.edu, liu2306@purdue.edu, wangxiao@purdue.edu

## Abstract

Coresets distill large datasets into small, representative subsets for eficient downstream learning. Yet Optimal Transport (OT)–based selection typically requires intensive computation of transport plans, limiting scalability. We introduce a scalable Sinkhorn coreset method that permits closed-form updates of the entropically regularized OT coupling by allowing nonuniform coreset weights. This produces centroids that generalize k-means via soft assignments. We establish asymptotic consistency of the selected measure and Lipschitz stability to data perturbations, providing accuracy and robustness guarantees. Across synthetic and real-world benchmarks, the proposed method achieves competitive or improved approximation quality while substantially reducing runtime compared to Wasserstein- and standard Sinkhorn-based coreset selection, especially at large scale.

Keywords: Data distillation; coreset; optimal transport; Wasserstein distance; representative sampling.

## 1 Introduction

Coreset methods are dataset distillation techniques designed to address the escalating computational and memory demands of large-scale learning by identifying small yet representative subsets of data; see Lei and Tao [2023], Yu et al. [2024], Moser et al. [2025] for comprehensive reviews. By preserving the data geometry with a substantially smaller sample size, downstream tasks achieve performance comparable to the original dataset across various clustering, regression, and inference tasks [Har-Peled and Mazumdar, 2004, Campbell and Broderick, 2018, Mirzasoleiman et al., 2020, Chhaya et al., 2022, Xia et al., 2024]. Coreset methods are becoming increasingly important in frontier applications in AI, including fine-tuning of billion-scale foundation models [San Joaquin et al., 2024, Maekawa et al., 2024], improving eficiency in image classification and object detection [Borsos et al., 2020, Cazenavette et al., 2022], and scaling modern benchmarks across multi-modal and federated learning [Wu et al., 2023, Guo et al., 2024].

Recent work leverages Wasserstein [Villani, 2008] and entropic regularized Wasserstein (Sinkhorn) distance [Cuturi, 2013, Altschuler et al., 2017] in Optimal Transport (OT) theory to construct coresets [Claici et al., 2018, Izzo et al., 2021, Liu et al., 2023, Xiong et al., 2024, Yin et al., 2025]. Unlike the Feldman–Langberg sub-sampling paradigm, which selects representatives from the observed data under a query-space and relative-error setup, the OT-based coresets preserve geometric structure by minimizing transport cost to the full dataset, which connects to classical notions of barycenter [Agueh and Carlier, 2011, Cuturi and Doucet, 2014], representative points [Flury, 1990, Fang and Wang, 1993, Mak and Joseph, 2018] and also aligns with moment-matching herding [Welling, 2009]. In addition, OT-based coreset methods are no longer restricted to uniform sample weights [Braverman et al., 2022], thereby enabling dataset compression eficiency through adaptive weighting schemes. We defer a more detailed comparison to the related work section.

![](images/5b6c050d19a5f539141f175880b948b72230c438afe5605962c03d04ad89aa76.jpg)  
Figure 1: Comparison on 2D manifold datasets. All optimal-transport coresets capture the global geometry. EMS (Ours) spreads smoothly and preserves low-density tails, while gradient-based WCSL can be over-smoothed.

However, existing OT-based coreset methods are computationally expensive. A major bottleneck lies in evaluating the similarity metric—namely, the Sinkhorn distance between a candidate coreset and the full dataset—which requires solving an OT problem. For example, computing the OT plan in Wasserstein distances scales cubically with dataset size [Villani, 2008], and it remains nearly quadratic via Sinkhorn distances [Cuturi, 2013, Altschuler et al., 2017, Genevay et al., 2019]. Moreover, constructing an OT-based coreset necessitates repeated evaluation of this metric until convergence. By contrast, classical coreset selection theory in geometric optimization often provides (1 + ε)-approximations with linear or near-linear construction time [Agarwal et al., 2005, Har-Peled and Mazumdar, 2004]. As a result, OT-based coreset methods sufer limitations from computational complexity, making them impractical for large-scale datasets.

This gap motivates our work to develop an eficient Sinkhorn coreset construction method. Our proposed approach, called EMS, is an EM-style algorithm that replaces explicit OT computation with eficient iterative updates. Unlike classical OT-based coreset that explicitly solves a discrete OT plan, the E-step computes a variational OT plan by taking expectations under a Gibbs distribution in function space, minimizing a variational upper bound on the Sinkhorn distance. In the M-step, the representative locations and weights are updated to maximize the expected assignment likelihood, ensuring a coherent and fully variational update of the discrete measure. Figure 1 highlights the smooth coverage EMS achieves on 2D manifold benchmarks. Our contributions are three-fold:

1. We provide a comprehensive analysis of weighted Sinkhorn coresets, covering standard results on geometry-preserving property, such as finite-atom approximation bounds and asymptotic consistency, to algorithm-specific guarantees, including approximation error, convergence, and stability theorems, ofering a solid foundation for our proposed method. To better highlight the importance of non-uniform weights in the setup, we compared our results with the vanilla k-means method.

2. We provide a detailed derivation of our proposed EMS algorithms. We reinterpret the weighted Sinkhorn coreset optimization problem from a variational perspective, formulating it as a soft assignment of points to coreset atoms, which naturally leads to the Gibbs-form solution (Eq.17) and an equivalent reduced objective that avoids repeated matrix scaling with fixed weights. This approach draws a clear connection to variational inference in function space, treating the coreset as a tractable surrogate that approximates the original measure under entropic regularization.

3. Comprehensive evaluations of our method are conducted in various data structures and also in downstream tasks. Our method improves upon classical baselines in distributional accuracy and stability, and achieves performance comparable to OT-based coresets obtained via explicit gradient-based optimization while being over 100× more eficient, consistent with our theoretical guarantees. For practice, we also propose a mini-batch variant that further improves practical scalability.

## 2 Related Work

Sensitivity-based Coreset Selection Classical coreset methods rely on sensitivity-based sampling from existing data points with respect to a specified query space Feldman and Langberg [2011]. In the case of likelihood-based or Bayesian inference, Bayesian coresets approximate the full-data log-likelihood by a sparse weighted subset of data points, yielding an eficient surrogate posterior for inference Huggins et al. [2016], Campbell and Broderick [2018], Manousakas et al. [2020], Zhang et al. [2021]. In comparison, OT-based coresets are constructed to approximate the discrete empirical data distribution in Wasserstein space. For example, Wasserstein or Sinkhorn coresets [Claici et al., 2018, Xiong et al., 2024, Yin et al., 2025] are distribution-model free in the sense that they do not assume a parametric likelihood and do not approximate a log-likelihood or posterior. That said, their objective are also shown to be equivalent via an integral probability metric (IPM) [Claici et al., 2018] under an ε-uniform approximation capacity framework [Yin et al.,

2025]. Such equivalence follows from a standard application of Kantorovich–Rubinstein duality [Villani, 2008, Arjovsky et al., 2017]. Consequently, OT-based coresets are not limited to finite data points or pointwise-defined target objective and can admit (1 + ϵ)-approximation guarantees over a function class.

Barycenters and Representative Points (RPs) Both barycenters and RPs also aim to compress probability measures, but ofers conceptually complementary views. Wasserstein barycenters minimize the average transport cost to multiple input measures, focusing on optimization problem itself[Agueh and Carlier, 2011], whereas representative points emphasis on asymptotic distributional convergence of its sparse approximation under diference statistical discrepancy (e.g., energy distance) [Flury, 1990, Mak and Joseph, 2018]. OT-based coresets can be viewed as a constrained case of barycenters of single discrete measure or RPs using a OT distance metric. Connecting these dots, we not only advance OT-based coreset theory by showing that Sinkhorn coresets avoid the systematic density approximation bias inherent to uniformly weighted centroids under squared loss, which characterizes classical k-means quantization [Graf and Luschgy, 2007]. Moreover, our proposed EMS algorithm admits smooth, closed-form gradient updates analogous to free-support Wasserstein barycenters [Cuturi and Doucet, 2014], while providing a more rigorous variational and optimization-theoretic derivation.

Variational, Schr¨odinger, and EM views of entropic OT. Entropic OT can be derived as a KL-penalized projection or as iterative Bregman projections [Benamou et al., 2015], and it admits a Schr¨odinger bridge interpretation that links transport plans to Gibbs measures [L´eonard, 2013]. This variational structure underpins diferentiability and stability results for Sinkhorn couplings and costs [Ghosal et al., 2022], and also yields useful statistical interpretations (e.g., as maximum-likelihood deconvolution) [Rigollet and Weed, 2018]. Applying these insights to the OT-based coreset problem, our derivation leads to an EM-style formulation for OT-based coreset construction. Our E-step directly employs the Gibbs form of the entropically regularized coupling, yielding assignments that are stable and largely insensitive to the choice of Sinkhorn regularization hyperparameter λ. The M-step then admits closed-form updates for both representative locations and non-uniform weights. Unlike gradient-based Sinkhorn coreset methods[Yin et al., 2025], our EMS run more eficiently in the small-λ regime and recovers a k-means-like hard-assignment procedure as $\lambda \to 0 ~ [ \mathrm { M i }$ et al., 2018].

## 3 Sinkhorn Coreset

In this section, we formalize the Sinkhorn coreset problem and demonstrate its geometry-preserving properties.

## 3.1 Weighted Coresets via Sinkhorn Loss

Our goal is to compress a given probability measure $\mu \in \mathcal P _ { p } ( \mathbb { R } ^ { d } )$ with finite p-th moment, primarily in the discrete empirical setting relevant to coreset construction, which also natural extends to absolutely continuous measures.

$$
\textstyle \mu = { \frac { 1 } { n } } \sum _ { i = 1 } ^ { n } \delta _ { x _ { i } } { \mathrm { ~ o r ~ } } d \mu ( x ) = p ( x ) d x .\tag{1}
$$

OT-based coreset methods seek to approximate $\mu$ via a discrete measure $\begin{array} { r } { \nu _ { k } ^ { * } : = \sum _ { j = 1 } ^ { k } \omega _ { j } \delta _ { y _ { j } } } \end{array}$ , defined by its locations $Y : = \{ y _ { j } \} _ { j = 1 } ^ { k }$ and weights $W = \{ \omega _ { j } \} _ { j = 1 } ^ { k }$ under OT-based metric $d ( \cdot , \cdot )$ such that,

for $k \ll n ,$

$$
\nu _ { k } ^ { * } = \arg \operatorname* { m i n } _ { \nu _ { k } \in \mathbb { G } _ { k } } d ( \mu , \nu _ { k } ) ,\tag{2}
$$

where $\begin{array} { r } { \mathbb { G } _ { k } = \{ \sum _ { j = 1 } ^ { k } \omega _ { j } \delta _ { y _ { j } } | \sum _ { j } \omega _ { j } = 1 , \omega _ { j } \geq 0 , y _ { j } \in \mathbb { R } ^ { d } \} } \end{array}$ is the set of all discrete measures with at most k atoms.

A canonical example $d ( \cdot , \cdot )$ is the (p-powered) Wasserstein-p distance $\boldsymbol { W } _ { p } ^ { p }$ , which is given by

$$
W _ { p } ^ { p } ( \mu , \nu ) : = \operatorname* { i n f } _ { \pi \in \Pi ( \mu , \nu ) } \int _ { R ^ { d } \times R ^ { d } } \| x - y \| ^ { p } d \pi ( x , y )\tag{3}
$$

where the cost $c ( x , y )$ is p-powered $\ell ^ { p } .$ -norm distance and $\Pi ( \mu , \nu )$ denotes the set of couplings π with marginals being $\mu$ and $\nu .$

When ν is discrete, any coupling $\pi \in \Pi ( \mu , \nu )$ can be represented by a transport plan $T =$ $( T _ { 1 } , \dots , T _ { k } ) : \mathbb { R } ^ { d } \to \Delta _ { k }$ , which induces a semi-discrete coupling $\pi _ { T }$ . The corresponding transport cost is its expected cost under $\pi _ { T }$

$$
\int \| x - y \| ^ { p } d \pi _ { T } ( x , y ) = \sum _ { j = 1 } ^ { k } \int _ { \mathbb { R } ^ { d } } \| x - y _ { j } \| ^ { p } T _ { j } ( x ) d \mu ( x ) .
$$

Define the cost vector $C = ( C _ { 1 } , \ldots , C _ { k } )$ with $C _ { j } ( x ) : = \| x - y _ { j } \| ^ { p } , j \in [ k ]$ , Eq.3 admits the equivalent form

$$
W _ { p } ^ { p } ( \mu , \nu ) = \operatorname* { i n f } _ { T \in \mathbb { T } _ { k } } \sum _ { j = 1 } ^ { k } \int _ { \mathbb { R } ^ { d } } C _ { j } ( x ) T _ { j } ( x ) d \mu ( x ) : = \langle C , T \rangle _ { \mu } ,\tag{4}
$$

where $\mathbb { T } _ { k }$ denotes the function space of admissible plans

$$
\mathbb { T } _ { k } : = \Big \{ T : \mathbb { R } ^ { d } \to \Delta _ { k } \ \big | \ \int _ { \mathbb { R } ^ { d } } T _ { j } ( x ) d \mu ( x ) = \omega _ { j } , \ \forall j \in [ k ] \Big \} .
$$

The Sinkhorn regularization on $W _ { p } ^ { p } ( \mu , \nu )$ [Cuturi, 2013] controls the Kullback–Leibler divergence between π with a trivial and independent coupling $\mu \otimes \nu ,$ which leads to the Sinkhorn distance:

$$
W _ { \lambda , p } ^ { p } ( \mu , \nu ) : = \operatorname* { i n f } _ { T \in \mathbb { T } _ { k } } \left\{ \langle C , T \rangle _ { \mu } + \lambda D _ { \mathrm { K L } } \big ( \pi _ { T } \| \mu \otimes \nu \big ) \right\} ,\tag{5}
$$

where $\lambda > 0$ is the regularization strength. For any $\lambda > 0$ , the regularization ensures convexity in $\pi _ { T }$ , allowing eficient computation of $T$ via Sinkhorn iterations [Sinkhorn and Knopp, 1967]. Though biased, the resulting optimal $\pi \ ( \mathrm { o r } \ T )$ realizes a discrete-time Schr¨odinger bridge [L´eonard, 2013]. Genevay et al. [2019] generalize Eq.5 to a proper distance, interpreting it as an interpolation between Wasserstein and Maximum Mean Discrepancy (MMD) distances.

A Sinkhorn Coreset aims to approximate $\mu$ under this metric

$$
\begin{array} { r l } & { \nu _ { k } ^ { * } : = \underset { \nu _ { k } \in \mathbb { G } _ { k } } { \arg \operatorname* { m i n } } W _ { \lambda , p } ^ { p } ( \mu , \nu _ { k } ) } \\ & { \quad : = \underset { \nu _ { k } ( Y , W ) \in \mathbb { G } _ { k } , \ T \in \mathbb { T } _ { k } } { \arg \operatorname* { m i n } } \mathcal { F } _ { \lambda } ( Y , W , T ) , } \end{array}\tag{6}
$$

where we define the Sinkhorn loss $\mathcal { F } _ { \lambda }$ as

$$
\mathcal F _ { \lambda } ( Y , W , T ) : = \langle C , T \rangle _ { \mu } + \lambda D _ { \mathrm { K L } } \big ( \pi _ { T } \| \mu \otimes \nu \big ) .\tag{7}
$$

Notably, coreset construction via objective $\mathcal { F } _ { \lambda }$ shifts the focus from merely obtaining an optimal coupling $\pi _ { T }$ (or equivalently $T )$ to a distributional geometry-preserving goal as $\nu _ { k } ^ { * }$ summarizes $\mu$ with $k < < n$ , as formalized below.

## 3.2 Geometry-Preserving Guarantees

Theorem 3.1 (Sinkhorn Bound). If the target µ has a bounded density, compact support, and finite $( p + \delta )$ -moment for some $\delta > 0$ and $\lambda > 0$ , for any $k \geq 1$ , then for any global minimizers $\nu _ { k } ^ { * } \in \mathbb { G } _ { k }$ of $E q . 6 ,$

$$
W _ { \lambda , p } ^ { p } ( \mu , \nu _ { k } ^ { * } ) ~ \le ~ C k ^ { - 1 / d } + C ^ { \prime } k ^ { - 2 / d }\tag{8}
$$

where constants $C , C ^ { \prime }$ depend only on $\mu , p , d .$

Proof can be found in Appendix A.1. To obtain this upper bound, we first adapt the classical quantization bound for the Wasserstein distance via deterministic Voronoi assignments. Next, we bound the KL divergence with a tight bound using Taylor expansion, thereby concluding the proof.

Corollary 3.2 (Geometry Persevering Asymptotic Consistency). For any global minimizers of $E q . 6 ,$ we have

$$
\operatorname* { l i m } _ { k \to \infty } W _ { p } ( \mu , \nu _ { k } ^ { * } ) \ = \ 0\tag{9}
$$

See Appendix A.2. Theorem 3.1 provides a finite-sample guarantee for Sinkhorn coresets, which matches the classical quantization rate. This indicates that a coreset of size k approximates $\mu$ eficiently while preserving its geometric structure. Corollary 3.2 further establishes asymptotic consistency: as $k  \infty$ , the coreset distribution converges to the target in Wasserstein distance, ensuring that the underlying geometry of $\mu$ is faithfully recovered in the limit.

## 3.3 On Non-uniform Weights

We contrast our weighted Sinkhorn construction with classical k-means under squared Euclidean loss, whose centers are uniformly weighted in the large-k regime common in high-resolution quantization theory.

Lemma 3.3 (Uniform-weight k-means are not consistent [Gersho, 1979]). Let $\begin{array} { r } { \nu _ { k } ^ { \mathrm { K M } } : = \frac { 1 } { k } \sum _ { j = 1 } ^ { k } \delta _ { y _ { j } } } \end{array}$ be the uniform-weight empirical measure supported on $Y _ { k }$ . Denote weak convergence $a s \Rightarrow$ . Under the high-resolution density assumption established in Appendix A.3, we have

$$
\nu _ { k } ^ { \mathrm { K M } } \Rightarrow \ g ^ { \star } ( y ) d y , \quad w i t h \ g ^ { \star } ( y ) = { \frac { p ( y ) ^ { { \frac { d } { d + 2 } } } } { \int p ^ { { \frac { d } { d + 2 } } } } } ,\tag{10}
$$

As a result, $\nu _ { k } ^ { \mathrm { K M } }$ fails to converge to $\mu$ for any $p \geq 1$ in general, unless the target distribution has uniform density on its support. Formal derivations, regularity assumptions, and the limiting density are provided in Appendix A.3.

Proposition 3.4 (Weighted correction recovers $\mu )$ . For $Y _ { k }$ defined in Appendix A.3. Define weights

$$
\omega _ { j } \propto \frac { p ( y _ { j } ) } { g ^ { \star } ( y _ { j } ) } , \nu _ { k } ^ { w } : = \sum _ { j = 1 } ^ { k } \omega _ { j } \delta _ { y _ { j } } , \sum _ { j = 1 } ^ { k } \omega _ { j } = 1 .\tag{11}
$$

Then $\nu _ { k } ^ { w } \Rightarrow \mu$ as $k \to \infty$

Please refer to Appendix A.4. Proposition 3.4 formalizes how non-uniform weights remove this bias. Consequently, our Sinkhorn coreset formulation leverages this flexibility through adaptive weighting in $\nu _ { k }$ , in contrast to fixed-weight Sinkhorn coreset approaches, as seen in Yin et al. [2025].

## 4 An Eficient Algorithm for Sinkhorn Coresets

In this section, we proposed our EMS algorithm. First, we use the notion of the free energy functionals and its Fenchel identity to derive the closed form update of $T$ . Here, recall that the transport plan $T : \mathbb { R } ^ { d }  \Delta _ { k }$ $\mathrm { E q . 7 }$ for $\mu$ maps each point $\boldsymbol { x } \in \mathbb { R } ^ { d }$ to a $( k - 1 )$ simplex $\Delta _ { k }$ and its admissible set $\mathbb { T } _ { k }$ ensures that $T$ is a valid assignment for every x.

## 4.1 Variational Inference in Function Space

Lemma 4.1 (Fenchel Identity of Free Energy Functional [Fenchel and Blackett, 1953]). Let $( \mathbb { T } _ { k } , \lVert \cdot \rVert )$ denote the Banach space of admissible transport plans $T : \mathbb { R } ^ { d }  \Delta _ { k }$ . Let $P _ { 0 }$ be a reference probability measure of T on $\mathbb { T } _ { k }$ . For a linear cost functional $f ( T )$ , define the free energy functional $F _ { f , P _ { 0 } }$ by

$$
F _ { f , P _ { 0 } } : = \log \mathbb { E } _ { T \sim P _ { 0 } } \big [ e ^ { f ( T ) } \big ] .\tag{12}
$$

Then, $F _ { f , P _ { 0 } }$ is convex in f and admits the variational representation (Fenchel Identity):

$$
F _ { f , P _ { 0 } } = \operatorname* { s u p } _ { q \ll P _ { 0 } } \Big \{ \mathbb { E } _ { T \sim q } [ f ( T ) ] - \mathrm { K L } ( q \| P _ { 0 } ) \Big \} ,\tag{13}
$$

where the supremum is over all probability measures q on $\mathbb { T } _ { k }$ that are absolutely continuous with respect to $P _ { 0 }$ . In particular, the optimizer is the Gibbs measure

$$
q ^ { * } ( T ) \propto e ^ { f ( T ) } P _ { 0 } ( T ) .\tag{14}
$$

The well-known proof is attached in Appendix A.5 for reference. Lemma 4.1 ofers a Gibbs’ variational perspective on the log-partition function defined over the space of transport plans: Minimizing the free energy functional $F _ { f , P _ { 0 } }$ can be interpreted as finding the optimal “trade-of” for a variational distribution $q$ over the admissible set of transport plans, balancing the maximization of the expected cost with proximity to the reference measure $P _ { 0 }$

In what follows, we demonstrate that the Sinkhorn loss, which shares this variational nature, can be recovered from $F _ { f , P _ { 0 } }$ for an appropriate choice of $f ( T )$ and $P _ { 0 } ,$ thereby reformulating Sinkhorn coreset optimization as an alternative variational optimization scheme.

Corollary 4.2 (Sinkhorn Loss as Negative Free Energy Lower Bound). Under Lemma $4 . 1 ,$ , if $f ( T ) : = - \frac { 1 } { \lambda } \langle C , T \rangle _ { \mu }$ with $\lambda > 0$ and $P _ { 0 }$ is chosen so that

$$
D _ { \mathrm { K L } } ( q ^ { * } \Vert P _ { 0 } ) \ \equiv \ \mathbb { E } _ { T \sim q ^ { * } } \Bigl [ D _ { \mathrm { K L } } ( \pi _ { T } \Vert \mu \otimes \nu _ { k } ) \Bigr ] .\tag{15}
$$

Then, by Jensen’s inequality, we obtain

$$
\begin{array} { r l } & { - \lambda F _ { f , P _ { 0 } } = \mathbb { E } _ { T \sim q ^ { * } } [ \langle C , T \rangle _ { \mu } ] + \lambda \mathbb { E } _ { T \sim q ^ { * } } [ D _ { \mathrm { K L } } ( \pi _ { T } \| \mu \otimes \nu _ { k } ) ] } \\ & { \quad \quad \quad \geq \langle C , \bar { T } \rangle _ { \mu } + \lambda D _ { \mathrm { K L } } ( \pi _ { \bar { T } } \| \mu \otimes \nu _ { k } ) } \\ & { \quad \quad \quad = \mathcal { F } _ { \lambda } ( Y , W , \bar { T } ) \geq \underset { T \in \mathbb { T } _ { k } } { \operatorname* { i n f } } \mathcal { F } _ { \lambda } ( Y , W , T ) , } \end{array}\tag{16}
$$

where the optimal variational $\bar { T } : = \mathbb { E } _ { T \sim q ^ { * } } [ T ]$

Proof is deferred to Appendix A.6. In brief, the first inequality in in Eq.16 is tight if and only if the Gibbs measure $q ^ { * }$ collapses to a Dirac measure $\delta _ { T }$ , which occurs either when $P _ { 0 } = \delta ( T - T _ { 0 } )$ or in the zero-temperature limit $\lambda  0 ;$ The second equality further requires that the support point $T _ { 0 }$ of such $P _ { 0 }$ coincides with the global minimizer $T ^ { * }$

In classical theory [Cuturi, 2013], the optimal $T ^ { * }$ is obtained in closed form by solving Lagrange multipliers or KKT conditions. Corollary 4.2, on the other hand, provides the probabilistic perspective of $T ^ { * }$ , interpreted as the expectation under a Dirac measure. Such probabilistic viewpoint is particularly valuable when formulating the weighted Sinkhorn coreset optimization problem to a EM-style iterative algorithms.

Specifically, rather than explicity solving for $T ^ { * }$ that minimizes $\mathcal { F } _ { \lambda } ( Y , W , T )$ for a fixed $Y , W$ we instead minimize the variational objective $- \lambda F _ { f , P _ { 0 } }$ over probability measures q on the function space of transport plans. The obtained expectation, i.e., the averaged transport plan $\bar { T } = \mathbb { E } _ { q ^ { * } } [ T ]$ serves as an efective surrogate for optimization. The formal derivation is provided below.

## 4.2 An EM-algorithm for Sinkhorn (EMS) Coreset

Theorem 4.3 (E-step via mean-field $P _ { 0 } )$ . A suficient condition for $D _ { \mathrm { K L } } ( q ^ { * } \lVert P _ { 0 } ) = \mathbb { E } _ { T \sim q ^ { * } } \Big [ D _ { \mathrm { K L } } ( \pi _ { T } \lVert \mu \otimes$

$\nu _ { k } ) \big |$ is that $P _ { 0 }$ factorizes over $\boldsymbol { x } \in \mathbb { R } ^ { d }$ with weights $\omega = ( \omega _ { 1 } , \ldots , \omega _ { k } )$ . Under such condition, $q ^ { * }$ then also ”factorizes” over x, $i . e .$

$$
q ^ { * } ( T ) = \prod _ { x \in \mathbb { R } ^ { d } } q _ { x } ^ { * } ( T ( x ) ) .
$$

Consequently, the expected transport plan $\bar { T } : = \mathbb { E } _ { T \sim q ^ { * } } [ T ]$ corresponds to the vector-valued softmax

$$
\bar { T } _ { j } ( x ) = \frac { \omega _ { j } \exp \big ( - C _ { j } ( x ) / \lambda \big ) } { \sum _ { l = 1 } ^ { k } \omega _ { l } \exp \big ( - C _ { l } ( x ) / \lambda \big ) } , \quad j = [ k ] .\tag{17}
$$

Please refer to Appendix A.7 for a detailed derivation. At a high level, although the resulting $\bar { T }$ may not correspond to the optimal $T ^ { * }$ in terms of minimizing the Sinkhorn distance Cuturi [2013], it admits a closed-form expression that can be computed eficiently, under appropriate choices of reference measure $P _ { 0 }$ . This flexibility are similar to the E-step in EM algorithms, which replaces exact latent-variable optimization with tractable expected assignments without altering the underlying objective.

Definition 4.4 (M-step). Let $\bar { T } ^ { ( t + 1 ) }$ be the transport plan obtained from the E-step. For $j = [ k ]$ abd at step $t + 1$ , M-step is defined as updating Y, W for $\nu _ { k } ^ { ( t + 1 ) }$ by

$$
y _ { j } ^ { ( t + 1 ) } = \frac { \int _ { \mathbb { R } ^ { d } } x \bar { T } _ { j } ^ { ( t + 1 ) } ( x ) d \mu ( x ) } { \int _ { \mathbb { R } ^ { d } } \bar { T } _ { j } ^ { ( t + 1 ) } ( x ) d \mu ( x ) } ,\tag{18}
$$

$$
\omega _ { j } ^ { ( t + 1 ) } = \int _ { \mathbb { R } ^ { d } } \bar { T } _ { j } ^ { ( t + 1 ) } ( x ) d \mu ( x ) .\tag{19}
$$

Proposition 4.5 (M-step monotonicity and closed-form updates). The updates defined by Definition 4.4 are the minimizer of the free-energy

$$
\mathcal { F } _ { \lambda } ( Y , W , \bar { T } ^ { ( t + 1 ) } )\tag{20}
$$

with respect to the cluster locations $Y = \{ y _ { j } \} _ { j = 1 } ^ { k }$ and weights in $W = \{ \omega _ { j } \} _ { j = 1 } ^ { k }$ . In addition, the functional is non-increasing along the M-step:

$$
\mathcal { F } _ { \lambda } ( Y ^ { ( t + 1 ) } , W ^ { ( t + 1 ) } , \bar { T } ^ { ( t + 1 ) } ) \le \mathcal { F } _ { \lambda } ( Y ^ { ( t ) } , W ^ { ( t ) } , \bar { T } ^ { ( t + 1 ) } ) ,\tag{21}
$$

with equality if and only $i f \left( \boldsymbol { Y } ^ { ( t ) } , W ^ { ( t ) } \right)$ satisfies the first-order optimality conditions for both blocks.

Algorithm 1 EMS Coreset with Mini-batch Data   
Require: dataset $\{ x _ { i } \} _ { i = 1 } ^ { n }$ (or i.i.d. draws from $p ( x ) ) , k \ll n , \lambda > 0 .$ batch size $B ,$ tolerance $\varepsilon > 0$   
1: Initialise $Y ^ { ( 0 ) } , W ^ { 0 } = \bar { \{ y _ { j } ^ { ( 0 ) } , \omega _ { j } ^ { ( 0 ) } \} } _ { j = 1 } ^ { k }$   
2: repeat   
3: reset $N _ { j } \gets 0 , S _ { j } \gets 0$ for $j = 1 , \dots , k$   
4: Partition $\{ i \} _ { i = 1 } ^ { n }$ into batches $\{ B _ { m } \} _ { m = 1 } ^ { n / B }$   
5: for $m = 1$ to $n / B$ do   
6: for $i \in B _ { m }$ and $j \in [ k ]$ do   
7: $\ell _ { i j } \gets \log \omega _ { j } ^ { ( t - 1 ) } - \| x _ { i } - y _ { j } ^ { ( t - 1 ) } \| ^ { 2 } / \lambda$   
8: $\bar { T } _ { i j } ^ { ( t + 1 ) } \gets \mathrm { \ddot { S o f t m a x } } ( l _ { i j } )$ by row   
9: end for   
10: $\begin{array} { r } { N _ { j } \gets N _ { j } + \sum _ { i \in \mathcal { B } _ { m } } \bar { T } _ { i j } ^ { ( k ) } } \end{array}$   
11: $\begin{array} { r } { S _ { j } \gets S _ { j } + \sum _ { i \in \mathcal { B } _ { m } } \bar { T } _ { i j } ^ { ( k ) } x _ { i } } \end{array}$   
12: end for   
13: for $j = 1$ to k do   
14: $\omega _ { j } ^ { ( k ) } \gets N _ { j } / \sum _ { \ell = 1 } ^ { k } N _ { \ell }$   
$y _ { j } ^ { ( t ) } \gets \left\{ { S _ { j } } / N _ { j } \right.$ , if $N _ { j } > 0 ,$   
15:   
otherwise.   
16: end for   
17: until $\| Y ^ { ( t + 1 ) } - Y ^ { ( t ) } \| < \varepsilon$   
18: return $Y ^ { ( t + 1 ) } , W ^ { ( t + \ddot { 1 } ) }$

Proof is shown in Appendix A.8. From a probabilistic viewpoint, the M-step updates the cluster locations and weights to best fit the current soft assignments induced by the transport plan. The location updates correspond to weighted centers of mass, i.e., maximum likelihood estimates of cluster means under the soft assignments, while the weight updates maximize the expected log-likelihood of the mixture weights.

For clarity, a detailed EM-algorithm for Sinkhorn (EMS) coreset with discrete $\mu$ (empirical dataset) is shown below. The complexity analysis and accuracy of our algorithm are discussed in experiments.

## 4.3 Robustness-Fidelity Trade-of

Here, we show that the stability of our algorithm comes from Softmax-activated (Smoothed) T updates. For coreset selection, it is important to ensure the transport plan does not change too wildly when data is perturbed.

Proposition 4.6 (Bounded Gradient). The gradient of the soft assignment from Eq. 17 satisfies

$$
\nabla _ { x } \bar { T } _ { j } ( x ) = \frac { 2 } { \lambda } \bar { T } _ { j } ( x ) \sum _ { \ell = 1 } ^ { k } \bar { T } _ { \ell } ( x ) ( y _ { j } - y _ { \ell } ) .\tag{22}
$$

Consequently, letting $D _ { Y } : = \operatorname* { m a x } _ { i , j \in [ k ] } \| y _ { i } - y _ { j } \|$ , we have $\begin{array} { r } { \left\| \nabla _ { x } \bar { T } _ { j } ( x ) \right\| \leq \frac { 2 D _ { Y } } { \lambda } } \end{array}$

Please refer to Appendix A.9. To conclude, the regularization parameter λ controls the trade-of between the sensitivity and the optimality of the transport plan T<sup>¯</sup>: larger λ tightens the bound and enhances robustness to data perturbations, but introduces bias relative to the true $\mathrm { O T }$ plan, thereby the fidelity of data distribution, as discussed in Corollary 4.2. By the same reasoning, uniform-weighted k-mean does not enjoy this property; for more stability results, please refer to Appendix A.10.

![](images/1961320a1a4133b9a4cff270f7de7c52d1043fdcdf1e25fcf9b14a090354264d.jpg)  
Figure 2: Synthetic Gaussian mixture datasets where each component has unit variance with its mean located at distance R from the origin.

## 5 Numerical Experiments

Methods. 1.EMS: Our proposed function space variational EM algorithm for weighted coreset construction with various λ; 2. WCSL: a gradient-based Sinkhorn coreset learner with uniform weights following Yin et al. [2025]. 3. K-mean centroids: a mean squared error(MSE) coreset with uniform weights[Lloyd, 1982]. Additional experimental details can be found in Appendix B.

## 5.1 Distributional Approximation Accuracy

This section we ask “Does $\nu _ { k }$ obtained by our algorithms approximate $\mu$ well in distributional distance, as suggested by Theorem 3.1 and Corollary 3.2?”

Setup. We report entropic Wasserstein distances between the empirical dataset $\mu$ and $\nu _ { k } \colon ( 1 )$ a $d { = } 1 0 0$ Gaussian mixture benchmark with $n = 1 0 ^ { 4 }$ and $k \in \{ 1 0 , 2 0 , 5 0 , 1 0 0 , 2 0 0 \}$ , and (2) MNIST with the same range of k. We also record wall-clock time to show the quality-cost trade-of.

Table 1 shows, EMS has similar performance in distributional accuracy as WCSL, while WCSL is much slower as it uses gradient descent to minimize the Sinkhorn loss directly. K-means lags on the image task, consistent with the uniform-weight bias in Lem. 3.3. Additional experiments are conducted on real-world benchmark datasets as used in Manousakas et al. [2020] can be found in table 4.

Table 1: Distributional accuracy and total runtime (distance / time). Lower distance is better; bold indicates the best distance for each k. Statistics averaged over five runs. dist / time stands for distance / time.
<table><tr><td rowspan="3">Dataset</td><td rowspan="3">Coreset size</td><td>Random</td><td>K-means</td><td colspan="2">WCSL</td><td colspan="4">EMS</td></tr><tr><td></td><td></td><td> $\lambda = 1 0 ^ { - 2 }$ </td><td></td><td>λ = 0</td><td> $\lambda = 1 0 ^ { - 3 }$ </td><td> $\lambda = 1 0 ^ { - 2 }$ </td><td> $\lambda = 1 0 ^ { - 1 }$ </td></tr><tr><td>dist / time</td><td>dist / time</td><td>dist / time</td><td></td><td>dist / time</td><td>dist / time</td><td>dist / time</td><td>dist / time</td></tr><tr><td rowspan="5">Gaussian</td><td>k = 10</td><td>18.28 /-</td><td>10.87 /0.09</td><td>10.85</td><td>/14.9</td><td>10.87 /0.08</td><td>10.86 /0.15</td><td>10.87 /0.15</td><td>10.87 /0.12</td></tr><tr><td>k = 20</td><td>16.05 /-</td><td>10.14 / 0.14</td><td>10.11</td><td>/ 24.4 10.14</td><td>/0.09</td><td>10.14 /0.10</td><td>10.14 / 0.17</td><td>10.12 /0.22</td></tr><tr><td>k = 50</td><td>14.31</td><td>9.25 / 0.08</td><td>9.19</td><td>56.9 9.24</td><td>0.07</td><td>9.24 / 0.11</td><td>9.23 / 0.18</td><td>9.20 0.39</td></tr><tr><td>k = 100</td><td>12.92 /-</td><td>8.69 /0.06</td><td>8.53 /92.5</td><td>8.63</td><td>/0.06</td><td>8.63 / 0.08</td><td>8.61 / 0.14</td><td>8.56 /0.54</td></tr><tr><td>k = 200</td><td>11.74 / -</td><td>8.18 /0.26</td><td>7.88 165.4</td><td>8.05</td><td>0.11</td><td>8.04 / 0.13</td><td>8.03 / 0.19</td><td>7.97 / 1.2</td></tr><tr><td rowspan="5">MNIST</td><td>k = 10</td><td>778.7 / -</td><td>424.9 /0.11</td><td>417.3 / 28.4</td><td></td><td>417.0 / 0.10</td><td>416.4 / 0.09</td><td>416.2 /0.08</td><td>416.5 /0.08</td></tr><tr><td>k = 20</td><td>727.2</td><td>382.5 / 0.15</td><td>376.0 / 45.1</td><td></td><td>375.4 / 0.14</td><td>374.6 / 0.13</td><td>375.6 /0.13</td><td>374.6 / 0.16</td></tr><tr><td>k = 50</td><td>585.8</td><td>340.2 / 0.29</td><td>329.8 / 87.9</td><td></td><td>329.7 / 0.28</td><td>329.0 / 0.26</td><td>329.5 / 0.38</td><td>329.4 /0.31</td></tr><tr><td>k = 100</td><td>535.3 / -</td><td>315.4 / 0.51</td><td>301.2 / 146.8</td><td></td><td>302.0 / 0.41</td><td>301.5 / 0.51</td><td>302.0 / 0.38</td><td>302.3 / 0.42</td></tr><tr><td>k = 200</td><td>476.6 / -</td><td>294.3 / 0.61</td><td>276.0 /293.2</td><td></td><td>278.7 / 0.60</td><td>278.8 / 0.68</td><td>278.9 / 0.69</td><td>278.6 / 0.76</td></tr><tr><td rowspan="5">Fashion MNIST</td><td>k = 10</td><td>576.7 / -</td><td>297.6 / 0.07</td><td>268.2 / 30.9</td><td></td><td>260.8 / 0.08</td><td>259.7 / 0.05</td><td>261.3 /0.04</td><td>261.4 /0.04</td></tr><tr><td>k = 20</td><td>497.6 / -</td><td>255.2 0.09</td><td>221.7 / 50.4</td><td></td><td>220.6 / 0.11</td><td>219.1 /0.14</td><td>220.8 / 0.10</td><td>220.2 / 0.10</td></tr><tr><td>k = 50</td><td>374.5 / - 326.6 / -</td><td>217.7 /0.22</td><td>188.3 105.5</td><td>187.7</td><td>/0.28</td><td>188.5 / 0.26</td><td>188.9 / 0.29</td><td>187.9 /0.34</td></tr><tr><td>k = 100 k = 200</td><td></td><td>198.1 / 0.42</td><td>169.9 / 201.6</td><td>171.6</td><td>/ 0.41 157.9 / 0.66</td><td>170.9 / 0.46</td><td>170.9 / 0.47</td><td>171.1 / 0.68</td></tr><tr><td></td><td>289.1 / -</td><td>187.5 / 0.65</td><td>155.3 /389.6</td><td></td><td></td><td>158.2 0.58</td><td>158.1 / 0.67</td><td>158.1 /0.91</td></tr></table>

## 5.2 Eficiency and Runtime Scaling

Importantly, we ask: “How fast is the EMS coreset construction compared to existing methods?”

Setup. Gaussian mixture data is simulated as shown in Fig. 2. Baseline is $n = 1 0 ^ { 4 } , d = 1 0 , k =$ 100 and we vary one factor at a time: $n \in \{ 1 0 ^ { 4 } , 1 0 ^ { 5 } , 1 0 ^ { 6 } \} , d \in \{ 1 , 1 0 , 1 0 ^ { 2 } , 1 0 ^ { 3 } \} , k \in \{ 1 0 , 1 0 ^ { 2 } , 1 0 ^ { 3 } \}$ We report mean ± std. wall-clock runtime over 20 repetitions.

Table 2 shows that the EMS family follows the near-linear cost via Alg. 1: even at $n { = } 1 0 ^ { 6 }$ the unregularized EMS run finishes in less than ten seconds and the entropic variants remain in the same single-digit band, while WCSL stretches to thousands of seconds. The time complexity of our algorithm finding OT plan is $O ( n k d )$ , which is linear to the dataset size, coreset size and data dimension.

Increasing the dimension only causes a modest drift (roughly 20 to 200 ms), thanks to the decoupled softmax E-step in Eq. 17. Furthermore, varying k leaves the wall-clock nearly flat, consistent with the closed-form M-step updates of Prop. 4.5. These results validate that EMS achieves fidelity on par with the EM routine, while attaining two to three orders of magnitude faster runtime compared to gradient-based baselines.

## 5.3 Hyper-parameter Selection for λ

We ask: “How sensitive is EMS to the regularization parameter $\lambda ? ^ { \mathfrak { p } }$

Empirically, Tables 1 and 2 show that EMS attains very similar distributional accuracy across a wide range of λ (including $\lambda \in \{ 1 0 ^ { - 3 } , 1 0 ^ { - 2 } , 1 0 ^ { - 1 } \}$ and the unregularized λ=0 variant), while runtime increases smoothly as λ grows.

This behavior is consistent with the EM interpretation: λ acts as a temperature in the softmax E-step (Eq. 17), primarily controlling the sharpness of responsibilities. The M-step (Def. 4.4) re-normalizes these responsibilities and recomputes weighted centroids and mixture weights, so moderate changes in λ tend to leave the converged coreset nearly invariant. In practice, we use $\lambda = 1 0 ^ { - 2 }$ as a default, set $\lambda = 0$ for the fastest hard-assignment variant, and prefer larger λ when additional robustness to perturbations is desired.

Table 2: Wasserstein distance and runtime in seconds. Lower is better. Means over 10 runs. Default setup is $n = 1 0 ^ { 4 } , d = 1 0 , k = 1 0 0$
<table><tr><td rowspan="2">Varying</td><td rowspan="2">Setup</td><td>WCSL</td><td colspan="4">EMS</td></tr><tr><td> $\lambda = 1 0 ^ { - 2 }$  distance / time</td><td>λ = 0 distance time</td><td> $\lambda = 1 0 ^ { - 3 }$  distance / time</td><td> $\lambda = 1 0 ^ { - 2 }$  distance /time</td><td> $\lambda = 1 0 ^ { - 1 }$  distance / time</td></tr><tr><td rowspan="3">Data size n</td><td> $1 0 ^ { 4 }$ </td><td>5.203/9.749</td><td>3.697/0.028</td><td>3.709/0.031</td><td>3.691/0.052</td><td>3.670/0.129</td></tr><tr><td> $1 0 ^ { 5 }$ </td><td>3.634/161.777</td><td>3.764/0.452</td><td>3.765/0.524</td><td>3.756/0.723</td><td>3.755/0.971</td></tr><tr><td> $1 0 ^ { 6 }$ </td><td>0.154/2,006.306</td><td>0.152/7.368</td><td>0.152/9.051</td><td>0.151/7.957</td><td>0.151/9.653</td></tr><tr><td rowspan="4">Dimension d</td><td>1</td><td>0.00791/2.450</td><td>0.00171/0.020</td><td>0.00199/0.024</td><td>0.00244/0.024</td><td>0.00543/0.020</td></tr><tr><td> $1 0$ </td><td>4.891/8.909</td><td>3.730/0.026</td><td>3.730/0.026</td><td>3.727/0.049</td><td>3.688/0.130</td></tr><tr><td> $1 0 ^ { 2 }$ </td><td>78.400/23.061</td><td>59.649/0.046</td><td>59.638/0.055</td><td>59.593/0.064</td><td>59.466/0.186</td></tr><tr><td> $1 0 ^ { 3 }$ </td><td>830.629/61.522</td><td>629.881/0.212</td><td>629.901/0.196</td><td>629.890/0.252</td><td>629.760/0.516</td></tr><tr><td rowspan="3">Coreset size k</td><td>10</td><td>14.639/2.509</td><td>7.191/0.026</td><td>6.520/0.030</td><td>6.262/0.033</td><td>5.622/0.036</td></tr><tr><td> $1 0 ^ { 2 }$ </td><td>4.994/8.866</td><td>3.706/0.026</td><td>3.709/0.029</td><td>3.696/0.053</td><td>3.674/0.121</td></tr><tr><td> $1 0 ^ { 3 }$ </td><td> $2 . 1 4 8 / 2 1 0 . 9 6 2$ </td><td>1.951/0.029</td><td>1.952/0.031</td><td>1.939/0.052</td><td>1.871/0.232</td></tr></table>

Table 3: Shapley reconstruction error on two test datasets (lower is better; best in bold). Results average over five runs.
<table><tr><td colspan="4">Census (n = 6,513)</td></tr><tr><td>Method  $\mathrm { E M S } ( \lambda = 0 . 0 1 )$ </td><td> $_ { \mathrm { M e a n A E } }$   $\mathbf { 5 . 0 7 \times 1 0 ^ { - 3 } }$ </td><td> $\operatorname { M a x A E }$   $5 . 4 0 \times 1 0 ^ { - 2 }$ </td><td>BVE  $3 . 3 1 \times 1 0 ^ { - 2 }$ </td></tr><tr><td> $\mathrm { E M S } ( \lambda = 0 )$  K-means WCSL</td><td> $8 . 1 0 \times 1 0 ^ { - 3 }$   $5 . 7 6 \times 1 0 ^ { - 3 }$   $5 . 5 2 \times 1 0 ^ { - 3 }$ </td><td> $8 . 8 7 \times 1 0 ^ { - 2 }$   $\mathbf { 4 . 3 4 \times 1 0 ^ { - 2 } }$   $4 . 9 2 \times 1 0 ^ { - 2 }$ </td><td> $3 . 7 3 \times 1 0 ^ { - 2 }$   $4 . 3 3 \times 1 0 ^ { - 2 }$   $\mathbf { 1 . 0 1 \times 1 0 ^ { - 2 } }$ </td></tr><tr><td colspan="4">Diabetes (n = 89)</td></tr><tr><td>Method</td><td>MeanAE</td><td>MaxAE</td><td>BVE</td></tr><tr><td> $\mathrm { E M S } ( \lambda = 0 . 0 1 )$ </td><td>1.04</td><td>4.38</td><td>1.38</td></tr><tr><td> $\mathrm { E M S } ( \lambda = 0 )$ </td><td>1.43</td><td>5.10</td><td>2.17</td></tr><tr><td>K-means</td><td>1.39</td><td>7.66</td><td>10.75</td></tr><tr><td>WCSL</td><td>1.44</td><td>6.55</td><td>5.24</td></tr></table>

## 5.4 Stability Across Initializations

Next, we ask “Do the Lipschitz properties of the transport plan and center updates translate into robust coreset constructions?”

Setup. For ten independent seeds, each method fits a coreset with k=64 on four-component Gaussian mixtures (with a small n=1000), as shown in Fig.2. We compute pairwise entropic Wasserstein distances between the resulting $\nu _ { k }$ from seeds and visualize the distances on bar plots.

Fig. 3 shows that EMS with λ = 0 or 0.01 produces concentrated points across seeds in both overlap (R=2) and separated (R=5) datasets. On the contrary, K-means significantly degrades on R = 5 Gaussian mixture dataset, reflecting the inability of uniform weights to reassign mass across empty regions, as discussed Lem. 3.3, where its hard assignments make it more susceptible to poor local minima. WCSL adapts locations but remains seed-sensitive. The low dispersion of $\mathrm { E M S } ( \lambda = 0 . 0 1 )$ aligns with our Lipschitz arguments and the monotone M-step discussed in Prop. 4.5, indicating robust convergence under soft assignments.

## 5.5 Downstream Interpretability via Shapley Values

Shapley-value explainers (SHAP; Shapley, 1953, Lundberg and Lee, 2017) require a background distribution to marginalize missing features, and the quality of this background directly afects feature attributions. To test whether a coreset preserves downstream interpretability, we replace the full training background with a weighted coreset $\nu _ { k }$ (here k=64) and compute test-set SHAP values using shap.TreeExplainer. We treat SHAP computed with the full training background as the reference, and report deviations of coreset-background SHAP from this reference (MeanAE/MaxAE over the SHAP matrix) along with base-value error (BVE); full setup and definitions are in Appendix B. We evaluate our method against k-means as a dataset summarization baseline for downstream tasks.

Table 3 shows that EMS with λ=0.01 achieves the lowest average distortion (MeanAE) on both datasets and is best on all three metrics on Diabetes, indicating that weighted Sinkhorn coresets provide faithful SHAP backgrounds. On Census, K-means attains a slightly smaller worst-case deviation (MaxAE) but has larger mean distortion and base-value drift, consistent with the uniform weight density bias discussed in Lem. 3.3. WCSL yields a strong base-value match on Census, but (as shown in Table 1) incurs substantially higher coreset construction cost. Overall, the improvements from EMS align with our mechanism: adaptive weights better match the data density, while soft assignments reduce high-variance attribution artifacts near low-density regions.

## 6 Conclusion

This work presents a fast and robust approach for constructing weighted coresets via approximate OT plans under a comprehensive theoretical framework on OT-based coresets. Our approach, EMS, combines a closed-form softmax E-step with decoupled M-step updates, enabling fast and stable optimization of the Sinkhorn loss. By maintaining strong fidelity to the original data, the coresets support downstream interpretability, such as Shapley value computation, while running orders of magnitude faster than gradient-based methods. A discussion of limitations and future research directions is provided in Appendix B.4.

## References

Pankaj K Agarwal, Sariel Har-Peled, Kasturi R Varadarajan, et al. Geometric approximation via coresets. Combinatorial and computational geometry, 52(1):1–30, 2005.

Martial Agueh and Guillaume Carlier. Barycenters in the wasserstein space. SIAM Journal on Mathematical Analysis, 43(2):904–924, 2011.

Jason Altschuler, Jonathan Weed, and Philippe Rigollet. Near-linear time approximation algorithms for optimal transport via sinkhorn iteration. In Advances in Neural Information Processing Systems, volume 30, 2017.

Martin Arjovsky, Soumith Chintala, and L´eon Bottou. Wasserstein generative adversarial networks. In International conference on machine learning, pages 214–223. PMLR, 2017.

Jean-David Benamou, Guillaume Carlier, Marco Cuturi, Luca Nenna, and Gabriel Peyr´e. Iterative bregman projections for regularized transportation problems. SIAM Journal on Scientific Computing, 37(2):A1111–A1138, 2015.

Zal´an Borsos, Mojmir Mutny, and Andreas Krause. Coresets via bilevel optimization for continual learning and streaming. Advances in neural information processing systems, 33:14879–14890, 2020.

Vladimir Braverman, Vincent Cohen-Addad, H-C Shaofeng Jiang, Robert Krauthgamer, Chris Schwiegelshohn, Mads Bech Toftrup, and Xuan Wu. The power of uniform sampling for coresets. In 2022 IEEE 63rd Annual Symposium on Foundations of Computer Science (FOCS), pages 462–473. IEEE, 2022.

Trevor Campbell and Tamara Broderick. Bayesian coreset construction via greedy iterative geodesic ascent. In International Conference on Machine Learning, pages 698–706. PMLR, 2018.

George Cazenavette, Tongzhou Wang, Antonio Torralba, Alexei A Efros, and Jun-Yan Zhu. Dataset distillation by matching training trajectories. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4750–4759, 2022.

Rachit Chhaya, Anirban Dasgupta, Jayesh Choudhari, and Supratim Shit. On coresets for fair regression and individually fair clustering. In AISTATS, pages 9603–9625, 2022.

Sebastian Claici, Aude Genevay, and Justin Solomon. Wasserstein measure coresets. arXiv preprint arXiv:1805.07412, 2018.

Marco Cuturi. Sinkhorn distances: Lightspeed computation of optimal transport. In Advances in Neural Information Processing Systems, volume 26, 2013.

Marco Cuturi and Arnaud Doucet. Fast computation of wasserstein barycenters. In International conference on machine learning, pages 685–693. PMLR, 2014.

Kai-Tai Fang and Yuan Wang. Number-theoretic methods in statistics, volume 51. CRC Press, 1993.

Dan Feldman and Michael Langberg. A unified framework for approximating and clustering data. In Proceedings of the forty-third annual ACM symposium on Theory of computing, pages 569–578, 2011.

Werner Fenchel and Donald W Blackett. Convex cones, sets, and functions. Princeton University, Department of Mathematics, Logistics Research Project, 1953.

Bernhard A Flury. Principal points. Biometrika, 77(1):33–41, 1990.

Aude Genevay, Lenaic Chizat, Francis Bach, Marco Cuturi, and Gabriel Peyr´e. Sample complexity of sinkhorn divergences. In Proceedings of the 22nd International Conference on Artificial Intelligence and Statistics (AISTATS), volume 89, pages 1574–1583, 2019.

Allen Gersho. Asymptotically optimal block quantization. IEEE Transactions on information theory, 25(4):373–380, 1979.

Promit Ghosal, Marcel Nutz, and Espen Bernton. Stability of entropic optimal transport and schr¨odinger bridges. Journal of Functional Analysis, 283(9):109622, 2022.

Siegfried Graf and Harald Luschgy. Foundations of quantization for probability distributions. Springer Science & Business Media, 2000.

Siegfried Graf and Harald Luschgy. Foundations of quantization for probability distributions. Springer, 2007.

Hongpeng Guo, Haotian Gu, Xiaoyang Wang, Bo Chen, Eun Kyung Lee, Tamar Eilam, Deming Chen, and Klara Nahrstedt. Fedcore: Straggler-free federated learning with distributed coresets. CoRR, abs/2402.00219, 2024. arXiv:2402.00219.

Sariel Har-Peled and Soham Mazumdar. On coresets for k-means and k-median clustering. In Proceedings of the thirty-sixth annual ACM symposium on Theory of computing, pages 291–300, 2004.

Jonathan Huggins, Trevor Campbell, and Tamara Broderick. Coresets for scalable bayesian logistic regression. Advances in neural information processing systems, 29, 2016.

Zachary Izzo, Sandeep Silwal, and Samson Zhou. Dimensionality reduction for wasserstein barycenter. Advances in neural information processing systems, 34:15582–15594, 2021.

Neil Jethani, Mukund Sudarshan, Ian Covert, Su-In Lee, and Rajesh Ranganath. Fastshap: Real-time shapley value estimation. ICLR 2022, 2022.

Yann LeCun, L´eon Bottou, Yoshua Bengio, and Patrick Hafner. Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11):2278–2324, 2002.

Shiye Lei and Dacheng Tao. A comprehensive survey of dataset distillation. IEEE Transactions on Pattern Analysis and Machine Intelligence, 46(1):17–32, 2023.

Christian L´eonard. A survey of the schr¨odinger problem and some of its connections with optimal transport. Discrete and Continuous Dynamical Systems, 34(4):1533–1574, 2013.

Haoyang Liu, Tiancheng Xing, Luwei Li, Vibhu Dalal, Jingrui He, and Haohan Wang. Dataset distillation via the wasserstein metric. CoRR, abs/2311.18531, 2023. URL https://doi.org/10. 48550/arXiv.2311.18531.

Haoyang Liu, Yijiang Li, Tiancheng Xing, Peiran Wang, Vibhu Dalal, Luwei Li, Jingrui He, and Haohan Wang. Dataset distillation via the wasserstein metric. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 1205–1215, 2025.

Stuart Lloyd. Least squares quantization in pcm. IEEE transactions on information theory, 28(2): 129–137, 1982.

Scott M Lundberg and Su-In Lee. A unified approach to interpreting model predictions. Advances in neural information processing systems, 30, 2017.

Aru Maekawa, Satoshi Kosugi, Kotaro Funakoshi, and Manabu Okumura. Dilm: Distilling dataset into language model for text-level dataset distillation. In Findings of the Association for Computational Linguistics: NAACL 2024, pages 3138–3153, 2024.

Simon Mak and V. Roshan Joseph. Support points. The Annals of statistics, 46(6A):2562–2592, 2018. ISSN 0090-5364.

Dionysis Manousakas, Zuheng Xu, Cecilia Mascolo, and Trevor Campbell. Bayesian pseudocoresets. Advances in Neural Information Processing Systems, 33:14950–14960, 2020.

Liang Mi, Wen Zhang, Xianfeng Gu, and Yalin Wang. Variational wasserstein clustering. In Proceedings of the European Conference on Computer Vision (ECCV), pages 322–337, 2018.

Baharan Mirzasoleiman, Jef Bilmes, and Jure Leskovec. Coresets for data-eficient training of machine learning models. In International Conference on Machine Learning, pages 6950–6960. PMLR, 2020.

Brian B Moser, Arundhati S Shanbhag, Stanislav Frolov, Federico Raue, Joachim Folz, and Andreas Dengel. A coreset selection of coreset selection literature: Introduction and recent advances. arXiv preprint arXiv:2505.17799, 2025.

Philippe Rigollet and Jonathan Weed. Entropic optimal transport is maximum-likelihood deconvolution. Comptes Rendus. Math´ematique, 356(11-12):1228–1235, 2018.

Ayrton San Joaquin, Bin Wang, Zhengyuan Liu, Philippe Muller, Nicholas Asher, Brian Y Lim, and Nancy F Chen. In2core: Leveraging influence functions for coreset selection in instruction finetuning of large language models. In Findings of the Association for Computational Linguistics: EMNLP 2024, pages 10324–10335. Association for Computational Linguistics, 2024.

Lloyd S. Shapley. A value for n-person games. In Harold W. Kuhn and Albert W. Tucker, editors, Contributions to the Theory of Games II, volume 28 of Annals of Mathematics Studies, pages 307–317. Princeton University Press, Princeton, NJ, 1953.

Richard Sinkhorn and Paul Knopp. Concerning nonnegative matrices and doubly stochastic matrices. Pacific Journal of Mathematics, 21(2):343–348, 1967.

C´edric Villani. Optimal transport: old and new, volume 338. Springer, 2008.

Max Welling. Herding dynamical weights to learn. In Proceedings of the 26th annual international conference on machine learning, pages 1121–1128, 2009.

Xindi Wu, Byron Zhang, Zhiwei Deng, and Olga Russakovsky. Vision-language dataset distillation. arXiv preprint arXiv:2308.07545, 2023.

Xiaobo Xia, Jiale Liu, Shaokun Zhang, Qingyun Wu, Hongxin Wei, and Tongliang Liu. Refined coreset selection: Towards minimal coreset size under model performance constraints. In International Conference on Machine Learning, pages 54082–54103. PMLR, 2024.

Zikai Xiong, Niccol\`o Dalmasso, Shubham Sharma, Freddy Lecue, Daniele Magazzeni, Vamsi Potluru, Tucker Balch, and Manuela Veloso. Fair wasserstein coresets. Advances in Neural Information Processing Systems, 37:132–168, 2024.

Haoyun Yin, Yixuan Qiu, and Xiao Wang. Wasserstein coreset via sinkhorn loss. Transactions on Machine Learning Research, 2025.

Ruonan Yu, Songhua Liu, and Xinchao Wang. Dataset Distillation: A Comprehensive Review . IEEE Transactions on Pattern Analysis & Machine Intelligence, 46(01):150–170, January 2024. ISSN 1939-3539. doi: 10.1109/TPAMI.2023.3323376.

Paul L. Zador. Development and Evaluation of Procedures for Quantizing Multivariate Distributions. PhD thesis, Stanford University, Stanford, CA, 1963. Ph.D. Dissertation.

Jacky Zhang, Rajiv Khanna, Anastasios Kyrillidis, and Sanmi Koyejo. Bayesian coresets: Revisiting the nonconvex optimization perspective. In International Conference on Artificial Intelligence and Statistics, pages 2782–2790. PMLR, 2021.

## A Proofs

## A.1 Proof of Theorem 3.1

Proof. Let $\mu = \rho ( x )$ dx be a probability measure on a compact convex domain $\Omega \subset \mathbb { R } ^ { d } .$ , with $\rho \in C ^ { 2 } ( \Omega )$ and $\rho _ { \mathrm { m i n } } > 0$ , and let $\nu _ { k } ^ { * } = \arg \operatorname* { m i n } _ { \nu \in \mathbb { G } _ { k } } W _ { \lambda , p } ( \mu , \nu )$ be the optimal k-point quantizer. Denote by $h _ { k } \sim k ^ { - 1 / d }$ the typical Voronoi diameter.

By classical quantization results from Graf and Luschgy [2000] and Zador’s Theorem [Zador, 1963], the deterministic Voronoi quantizer $\nu _ { k , v o r }$ satisfies

$$
W _ { p } ^ { p } ( \mu , \nu _ { k , v o r } ) = O ( k ^ { - 1 / d } ) ,\tag{23}
$$

where the space is partitioned into Voronoi cells $V _ { j }$ around points $y _ { v o r }$ such that each cell contains all points closer to its center than to any other. This provides an upper bound for $W _ { p } ( \mu , \nu _ { k } ^ { * } )$

To bound the KL divergence, we define a Gibbsized local coupling $\pi _ { \tilde { T } }$ for each Voronoi cell $V _ { j }$

$$
\pi _ { \tilde { T } } ( d x , d y ) = \rho ( x ) q _ { \lambda } ( y | x ) d x \mathrm { ~ w i t h ~ } q _ { \lambda } ( y | x ) = \frac { \exp \bigl ( - \| x - y \| ^ { p } / \lambda \bigr ) } { \sum _ { m = 1 } ^ { k } \exp \bigl ( - \| x - y _ { m } \| ^ { p } / \lambda \bigr ) } .\tag{24}
$$

The logarithm of the partition function is:

$$
\log Z _ { \lambda } ( x ) = \log \sum _ { m = 1 } ^ { k } \exp ( - c _ { m } / \lambda ) , \mathrm { w i t h ~ } c _ { m } : = \| x - y _ { m } \| ^ { p } ,\tag{25}
$$

which acts as a soft-minimum of the local costs $\{ c _ { m } \} _ { m = 1 } ^ { k }$ . For small λ, expanding log $Z _ { \lambda } ( x )$ gives

$$
- \lambda \log \sum _ { m } e ^ { - c _ { m } / \lambda } = \operatorname* { m i n } _ { m } c _ { m } + \frac { \lambda } { 2 } \sum _ { m \neq j ^ { \ast } } e ^ { - ( c _ { m } - c _ { j ^ { \ast } } ) / \lambda } + \cdot \cdot \cdot ,\tag{26}
$$

where $j ^ { * } = \arg \operatorname* { m i n } _ { m } c _ { m }$ . From this expansion, the Gibbs probabilities are dominated by the nearest center:

$$
q _ { \lambda } ( y _ { m } | x ) \approx \left\{ \begin{array} { l l } { 1 - \sum _ { m \neq j ^ { * } } e ^ { - ( c _ { m } - c _ { j ^ { * } } ) / \lambda } , } & { m = j ^ { * } } \\ { e ^ { - 2 ( c _ { m } - c _ { j ^ { * } } ) / \lambda } , } & { m \neq j ^ { * } } \end{array} . \right.\tag{27}
$$

For a single Voronoi cell $V _ { j }$ , the KL divergence w.r.t. the independent measure $\mu \otimes \nu$ is

$$
\mathrm { K L } ( \pi _ { \tilde { T } } \| \mu \otimes \nu ) = \sum _ { m } \int _ { V _ { j } } \pi _ { \tilde { T } } ( x , y _ { m } ) \log \frac { \pi _ { \tilde { T } } ( x , y _ { m } ) } { \mu ( V _ { j } ) \nu ( y _ { m } ) } d x .\tag{28}
$$

Plugging in $\pi _ { \tilde { T } } ( x , y _ { m } ) = \rho ( x ) q _ { \lambda } ( y _ { m } | x )$ and separating the entropy term:

$$
\mathrm { K L } ( \pi _ { \widetilde { T } } \| \mu \otimes \nu ) = \sum _ { m } \int _ { V _ { j } } \rho ( x ) q _ { \lambda } ( y _ { m } | x ) \log q _ { \lambda } ( y _ { m } | x ) d x + \sum _ { m } \int _ { V _ { j } } \rho ( x ) q _ { \lambda } ( y _ { m } | x ) \log \frac { \rho ( x ) } { \mu ( V _ { j } ) \nu ( y _ { m } ) } d x .\tag{29}
$$

The first term dominates for small λ. Using the approximation in 27, we obtain

$$
\lambda \mathrm { K L } ( \pi _ { \tilde { T } } | | \mu \otimes \nu ) = \frac { \lambda } { 2 } \int _ { V _ { j } } \sum _ { m \neq j ^ { * } } \frac { \rho ( x ) } { \mu ( V _ { j } ) } e ^ { - ( c _ { m } - c _ { j ^ { * } } ) / \lambda } d x + O ( \lambda e ^ { - 3 c h _ { k } ^ { p } / \lambda } ) .\tag{30}
$$

Since $c _ { m } ( x ) - c _ { j ^ { * } } ( x ) = \Theta ( h _ { k } ^ { p } )$ within each cell, Laplace’s method gives

$$
\lambda \mathrm { K L } ( \pi _ { \tilde { T } } \| \mu \otimes \nu ) = { \cal O } \big ( \lambda e ^ { - c h _ { k } ^ { p } / \lambda } \big ) + { \cal O } ( \lambda h _ { k } ^ { 2 p } ) .\tag{31}
$$

Choosing $\lambda _ { k } = c h _ { k } ^ { p } = c k ^ { - p / d }$ , the Taylor remainder is $O ( k ^ { - 2 p / d } )$ , so

$$
\mathcal { F } _ { \lambda _ { k } } ( Y _ { v o r } , W _ { v o r } , \tilde { T } ) = O ( k ^ { - p / d } ) + O ( k ^ { - 2 p / d } ) .\tag{32}
$$

By optimality of $\nu _ { k } ^ { * }$ and nonnegativity of KL,

$$
W _ { p } ^ { p } ( \mu , \nu _ { k } ^ { * } ) \leq W _ { \lambda , p } ^ { p } ( \mu , \nu _ { k } ^ { * } ) \leq C k ^ { - p / d } + C ^ { \prime } k ^ { - 2 p / d } .\tag{33}
$$

Taking the $p ^ { \mathrm { t h } }$ root, we conclude

$$
W _ { p } ( \mu , \nu _ { k } ^ { * } ) \leq C k ^ { - 1 / d } + C ^ { \prime } k ^ { - 2 / d } ,\tag{34}
$$

where the first term is the classical quantization rate, and the second arises from the second-order Taylor expansion of the entropic bias. This corresponds to Eq.8 and thus completes the proof.

By the way, if we further assume that the support of $\mu$ lies on a d-dimensional smooth manifold $\mathcal { M } \subset \Omega$ with $d _ { m } = d / 2$ intrinsic manifold dimension, the quantization error $C k ^ { - 1 / d _ { m } }$ in the first term can be absorbed into the second-order term by suitably adjusting constants, under standard manifold quantization argument [Graf and Luschgy, 2000]. The Wasserstein distance admits the simplified bound

$$
W _ { p } ( \mu , \nu _ { k } ^ { * } ) \leq 2 C k ^ { - 2 / d } ,\tag{35}
$$

where C depends on the manifold geometry and smoothness of $\rho , p ,$ and data dimension $d .$ □

## A.2 Proof of Corollary 3.2

Proof. Taking the limit as $k  \infty$ , since $1 / d > 0$ and $2 / d > 0$ , we have

$$
\operatorname * { l i m } _ { k  \infty } C k ^ { - 1 / d } = 0 , \operatorname * { l i m } _ { k  \infty } C ^ { \prime } k ^ { - 2 / d } = 0 .\tag{36}
$$

Thus, combining the vanishing of the classical quantization error and the entropic KL correction, we conclude that, asymptotically,

$$
\operatorname* { l i m } _ { k \to \infty } W _ { p } ( \mu , \nu _ { k } ^ { * } ) = 0 .\tag{37}
$$

In other words, the Sinkhorn coresets converge to the underlying $\mu$ in terms of the Wasserstein distance. □

## A.3 High-resolution asymptotics for uniform k-means

We first describe the regularity assumption and classical density limit that is mentioned Sec. 3.3.   
Lemma 3.3 is an informal version of Theorem A.2.

Assumption A.1 (High-resolution regularity[Gersho, 1979]). Let $\mu$ be absolutely continuous on $\mathbb { R } ^ { d }$ with density p that is bounded and continuous on a compact set. For each $k ,$ , let $Y _ { k } = \{ y _ { 1 } , . . . , y _ { k } \} \subset$ $\mathbb { R } ^ { d }$ be a (near-)minimizer of the distortion

$$
L _ { k } ( Y ) = \int _ { \mathbb { R } ^ { d } } \operatorname* { m i n } _ { 1 \leq j \leq k } \| x - y _ { j } \| ^ { 2 } p ( x ) d x .\tag{38}
$$

Write $\{ V _ { j } \} _ { j = 1 } ^ { k }$ for the Voronoi partition induced by $Y _ { k }$ . We assume the usual high-resolution regularity: cell diameters max<sub>j</sub> diam $( V _ { j } )  0$ as $k  \infty$ , boundary efects are negligible, and local cell shapes are asymptotically regular.

Theorem A.2 (Asymptotic density for k-means[Gersho, 1979]). Under Assumption A.1, any sequence of asymptotically optimal centroids $\left\{ Y _ { k } \right\}$ admits a limiting codepoint density $g ^ { \star }$ on $\mathbb { R } ^ { d }$ such that, for every bounded continuous test function $\varphi$ as $k  \infty$ , we have

$$
\frac { 1 } { k } \sum _ { j = 1 } ^ { k } \varphi ( y _ { j } ) \longrightarrow \int _ { \mathbb { R } ^ { d } } \varphi ( y ) g ^ { \star } ( y ) d y , w i t h \ g ^ { \star } ( y ) \propto p ( y ) ^ { \frac { d } { d + 2 } } .\tag{39}
$$

Equivalently, $g ^ { * } ( x )$ can’t recover underlying true $p ( x )$ and

$$
g ^ { \star } ~ = ~ \frac { p ^ { \frac { d } { d + 2 } } } { \int p ^ { \frac { d } { d + 2 } } } .\tag{40}
$$

Proof. Partition $\mathbb { R } ^ { d }$ into Voronoi cells $\{ V _ { j } \} _ { j = 1 } ^ { k }$ induced by the generator set $Y = \{ y _ { j } \} _ { j = 1 } ^ { k }$ . The expected quantization error can be expressed as

$$
L _ { k } ( Y ) = \sum _ { j = 1 } ^ { k } \int _ { V _ { j } } \| x - y _ { j } \| ^ { 2 } p ( x ) d x .\tag{41}
$$

Under the local uniformity assumption on p within each $V _ { j }$ , we have the asymptotic relation

$$
L _ { k } ( Y ) = \sum _ { j = 1 } ^ { k } p ( y _ { j } ) \int _ { V _ { j } } \| x - y _ { j } \| ^ { 2 } d x + o ( k ^ { - 2 / d } ) , \quad \mathrm { a s } \ k \to \infty .\tag{42}
$$

From high-resolution quantization theory, each cell integral satisfies

$$
\int _ { V _ { j } } | | x - y _ { j } | | ^ { 2 } d x = \alpha _ { d } | V _ { j } | ^ { 1 + \frac { 2 } { d } } + o ( | V _ { j } | ^ { 1 + \frac { 2 } { d } } ) ,\tag{43}
$$

where $\alpha _ { d } > 0$ is a shape-dependent constant. If the empirical locations $\{ y _ { j } \}$ are asymptotically distributed according to a continuous density g, then the cell volumes satisfy

$$
| V _ { j } | g ( y _ { j } ) = k ^ { - 1 } + o ( k ^ { - 1 } ) .\tag{44}
$$

Substituting these expressions yields

$$
L _ { k } ( Y ) = \alpha _ { d } k ^ { - \frac { 2 } { d } } \sum _ { j = 1 } ^ { k } p ( y _ { j } ) g ( y _ { j } ) ^ { - \frac { 2 } { d } } | V _ { j } | + o \bigl ( k ^ { - \frac { 2 } { d } } \bigr ) ,\tag{45}
$$

which in the limit becomes

$$
L _ { k } ( Y ) = \alpha _ { d } k ^ { - { \frac { 2 } { d } } } \int _ { { \mathbb R } ^ { d } } p ( y ) g ( y ) ^ { - { \frac { 2 } { d } } } d y + o ( k ^ { - { \frac { 2 } { d } } } ) .\tag{46}
$$

Minimizing the leading-order term over densities g under the constraint $\textstyle \int g = 1$ and applying H¨older’s inequality yields the unique optimizer

$$
g ^ { \star } ( y ) \propto p ( y ) ^ { \frac { d } { d + 2 } } \neq p ( y ) .\tag{47}
$$

Finally, by a Riemann-sum argument under Assumption A.1, the discrete sum converges weakly to the integral in equation 39. □

## A.4 Proof of Proposition 3.4

Proof. For any bounded continuous $\varphi ,$

$$
\sum _ { j = 1 } ^ { k } \omega _ { j } \varphi ( y _ { j } ) \approx { \frac { \sum _ { j = 1 } ^ { k } { \frac { p ( y _ { j } ) } { g ^ { * } ( y _ { j } ) } } \varphi ( y _ { j } ) } { \sum _ { j = 1 } ^ { k } { \frac { p ( y _ { j } ) } { g ^ { * } ( y _ { j } ) } } } } \longrightarrow { \frac { \int { \frac { p ( y ) } { g ^ { * } ( y ) } } \varphi ( y ) g ^ { \star } ( y ) d y } { \int { \frac { p ( y ) } { g ^ { * } ( y ) } } g ^ { \star } ( y ) d y } } = \int \varphi ( y ) p ( y ) d y ,\tag{48}
$$

using weak convergence of $\{ y _ { j } \}$ to $g ^ { \star }$ and continuity of $y \mapsto p ( y ) / g ^ { \star } ( y )$ . Hence $\nu _ { k } ^ { w }$ weakly converge to $\mu$ as $k  \infty$ □

## A.5 Proof of Lemma 4.1

Proof. For any $f _ { 1 } , f _ { 2 }$ and $\lambda \in [ 0 , 1 ]$ , by H¨older’s inequality

$$
F _ { \lambda f _ { 1 } + ( 1 - \lambda ) f _ { 2 } , P _ { 0 } } = \log \mathbb { E } _ { P _ { 0 } } \Big [ e ^ { \lambda f _ { 1 } ( T ) + ( 1 - \lambda ) f _ { 2 } ( T ) } \Big ] \leq \log \Big ( \mathbb { E } _ { P _ { 0 } } \big [ e ^ { f _ { 1 } ( T ) } \big ] \Big ) ^ { \lambda } \Big ( \mathbb { E } _ { P _ { 0 } } \big [ e ^ { f _ { 2 } ( T ) } \big ] \Big ) ^ { 1 - \lambda } = \lambda F _ { f _ { 1 } , P _ { 0 } } + ( 1 - \lambda ) F _ { f _ { 2 } , P _ { 0 } } .\tag{49}
$$

Hence, $F _ { f , P _ { 0 } }$ is convex in $f .$ . For any $q \ll P _ { 0 }$ , let $\begin{array} { r } { r ( T ) = \frac { d q } { d P _ { 0 } } ( T ) } \end{array}$ , then

$$
\begin{array} { r } { \mathbb { E } _ { q } [ f ( T ) ] = \mathbb { E } _ { P _ { 0 } } [ r ( T ) f ( T ) ] , \quad \mathrm { K L } ( q \| P _ { 0 } ) = \mathbb { E } _ { P _ { 0 } } [ r ( T ) \log r ( T ) ] . } \end{array}\tag{50}
$$

By definition of $F _ { f , P _ { 0 } }$ 2

$$
F _ { f , P _ { 0 } } = \log \mathbb { E } _ { P _ { 0 } } [ e ^ { f ( T ) } ] = \log \mathbb { E } _ { q } \left[ \frac { e ^ { f ( T ) } } { r ( T ) } \right] .\tag{51}
$$

Applying Jensen’s inequality to log yields

$$
F _ { f , P _ { 0 } } \geq \mathbb { E } _ { q } \left[ \log { \frac { e ^ { f ( T ) } } { r ( T ) } } \right] = \mathbb { E } _ { q } [ f ( T ) ] - \mathbb { E } _ { q } [ \log { r ( T ) } ] = \mathbb { E } _ { q } [ f ( T ) ] - \mathrm { K L } ( q \| P _ { 0 } ) .\tag{52}
$$

Taking the supremum over all $q \ll P _ { 0 }$ gives

$$
F _ { f , P _ { 0 } } \geq \operatorname* { s u p } _ { q \ll P _ { 0 } } { \left\{ \mathbb { E } _ { q } [ f ( T ) ] - \mathrm { K L } ( q \| P _ { 0 } ) \right\} } .\tag{53}
$$

To understand the equality condition, define the Gibbs measure

$$
q ^ { * } ( T ) = \frac { e ^ { f ( T ) } P _ { 0 } ( T ) } { \mathbb { E } _ { P _ { 0 } } [ e ^ { f ( T ) } ] } .\tag{54}
$$

Then and

$$
\frac { d q ^ { * } } { d P _ { 0 } } ( T ) = \frac { e ^ { f ( T ) } } { \mathbb { E } _ { P _ { 0 } } \left[ e ^ { f ( T ) } \right] } ; \quad \mathrm { K L } ( q ^ { * } \| P _ { 0 } ) = \mathbb { E } _ { q ^ { * } } \left[ \log \frac { d q ^ { * } } { d P _ { 0 } } \right] = \mathbb { E } _ { q ^ { * } } [ f ( T ) ] - F _ { f , P _ { 0 } } .\tag{55}
$$

Rearranging gives

$$
F _ { f , P _ { 0 } } = \mathbb { E } _ { q ^ { * } } [ f ( T ) ] - \mathrm { K L } ( q ^ { * } | | P _ { 0 } ) ,\tag{56}
$$

where equality holds at $q = q ^ { * }$ . Therefore,

$$
F _ { f , P _ { 0 } } = \operatorname* { s u p } _ { q \ll P _ { 0 } } { \left\{ \mathbb { E } _ { q } [ f ( T ) ] - \mathrm { K L } ( q \| P _ { 0 } ) \right\} } ,\tag{57}
$$

with unique maximizer $q ^ { * } ( T ) \propto e ^ { f ( T ) } P _ { 0 } ( T )$

## A.6 Proof of Corollary 4.2

Proof. Starting from 57 at the optimum $q ^ { * }$

$$
F _ { f , P _ { 0 } } = \mathbb { E } _ { T \sim q ^ { * } } [ f ( T ) ] - \mathrm { K L } ( q ^ { * } | | P _ { 0 } ) .\tag{58}
$$

Let $\begin{array} { r } { f ( T ) = - \frac { 1 } { \lambda } \langle C , T \rangle _ { \mu } } \end{array}$ and multiply both sides by $- \lambda$

$$
- \lambda F _ { f , P _ { 0 } } = \mathbb { E } _ { T \sim q ^ { * } } [ \langle C , T \rangle _ { \mu } ] + \lambda \mathrm { K L } ( q ^ { * } \| P _ { 0 } ) .\tag{59}
$$

By assumption on $P _ { 0 }$ , KL $. ( q ^ { * } \| P _ { 0 } ) = \mathbb { E } _ { T \sim q ^ { * } } \Big [ D _ { \mathrm { K L } } ( \pi _ { T } \| \mu \otimes \nu _ { k } ) \Big ]$ , then substituting into 59 gives

$$
- \lambda F _ { f , P _ { 0 } } = \mathbb { E } _ { T \sim q ^ { * } } [ \langle C , T \rangle _ { \mu } ] + \lambda \mathbb { E } _ { T \sim q ^ { * } } \Bigl [ D _ { \mathrm { K L } } ( \pi _ { T } \| \mu \otimes \nu _ { k } ) \Bigr ] .\tag{60}
$$

Since both $\langle C , T \rangle _ { \mu }$ and $D _ { \mathrm { K L } } ( \pi _ { T } \lVert \boldsymbol { \mu } \otimes \boldsymbol { \nu } _ { k } )$ are convex in $T _ { \mathbf { \delta } }$ , by Jensen’s inequality,

$$
\mathbb { E } _ { T \sim q ^ { * } } [ \langle C , T \rangle _ { \mu } ] \ge \langle C , \bar { T } \rangle _ { \mu } ,\tag{61}
$$

$$
\begin{array} { r } { \mathbb { E } _ { T \sim q ^ { * } } \Big [ D _ { \mathrm { K L } } ( \pi _ { T } \| \mu \otimes \nu _ { k } ) \Big ] \geq D _ { \mathrm { K L } } ( \pi _ { \bar { T } } \| \mu \otimes \nu _ { k } ) , } \end{array}\tag{62}
$$

where $\bar { T } : = \mathbb { E } _ { T \sim q ^ { * } } [ T ]$ . Applying these to 60 yields the inequality

$$
- \lambda F _ { f , P _ { 0 } } \geq \langle C , \bar { T } \rangle _ { \mu } + \lambda D _ { \mathrm { K L } } ( \pi _ { \bar { T } } \| \mu \otimes \nu _ { k } ) .\tag{63}
$$

The right-hand side corresponds to the regularized optimal transport objective,

$$
{ \mathcal F } _ { \lambda } ( Y , W , \bar { T } ) = \langle C , \bar { T } \rangle _ { \mu } + \lambda D _ { \mathrm { K L } } ( \pi _ { \bar { T } } \| \mu \otimes \nu _ { k } ) ,\tag{64}
$$

whose minimum over $\mathbb { T } _ { k }$ defines the Sinkhorn loss. Hence,

$$
- \lambda F _ { f , P _ { 0 } } \ge \mathcal { F } _ { \lambda } ( Y , W , \bar { T } ) \ge \operatorname* { i n f } _ { T \in \mathbb { T } _ { k } } \mathcal { F } _ { \lambda } ( Y , W , T ) .\tag{65}
$$

(A) Equality in the Jensen step. The first inequality in Corollary 4.2 originates from Jensen’s inequality, which is tight if and only if $T$ is almost surely constant under $q ^ { * }$ , i.e. $q ^ { * } = \delta _ { T }$ . This occurs either when $P _ { 0 } = \delta ( T - T _ { 0 } )$ or in the zero-temperature limit $\lambda \to 0$ , where the Gibbs measure collapses to a single deterministic transport plan:

$$
\operatorname* { l i m } _ { \lambda \to 0 } q ^ { \ast } ( T ) = \delta _ { T ^ { \ast } } ( T ) .\tag{66}
$$

In this limit the entropy term $D _ { \mathrm { K L } } ( q ^ { * } \lVert P _ { 0 } )$ vanishes and the free energy reduces to a pure energy functional:

$$
- \lambda F _ { f , P _ { 0 } } = \langle C , T ^ { * } \rangle _ { \mu } .\tag{67}
$$

Geometrically, the collapse of $q ^ { * }$ corresponds to the elimination of stochasticity in the transport plan, while probabilistically it represents the transition from a Boltzmann distribution to a deterministic mapping.

(B) Equality at the global minimum. The second inequality,

$$
\mathcal { F } _ { \lambda } ( Y , W , \bar { T } ) \geq \operatorname* { i n f } _ { T } \mathcal { F } _ { \lambda } ( Y , W , T ) ,\tag{68}
$$

is tight when $\bar { T } = \mathbb { E } _ { T \sim q ^ { * } } [ T ]$ coincides with the global minimizer $T ^ { * }$ of the Sinkhorn functional. Even if $q ^ { * }$ is not a Dirac measure, equality is approximately achieved when $q ^ { * }$ concentrates its mass near $T ^ { * }$ , so that $\mathbb { E } _ { q ^ { * } } [ T ] \approx T ^ { * }$ . In this case, the Gibbs measure $q ^ { * }$ behaves as a Boltzmann distribution over transport plans, assigning higher probability to low-cost couplings. As the temperature λ decreases, $q ^ { * }$ becomes increasingly peaked around $T ^ { * }$ , and the free-energy bound becomes tight:

$$
- \lambda F _ { f , P _ { 0 } } \to \mathcal { F } _ { \lambda } ( Y , W , T ^ { * } ) .\tag{69}
$$

□

## A.7 Proof of Theorem 4.3

Proof. From Lemma 4.1, the optimal variational distribution, when $\begin{array} { r } { f ( T ) = - \frac { 1 } { \lambda } \langle C , T \rangle _ { \mu } . } \end{array}$ , is

$$
q ^ { * } ( T ) = \frac { e ^ { - \frac { 1 } { \lambda } \langle C , T \rangle _ { \mu } } P _ { 0 } ( T ) } { \mathbb { E } _ { T \sim P _ { 0 } } [ e ^ { - \frac { 1 } { \lambda } \langle C , T \rangle _ { \mu } } ] } .\tag{70}
$$

By assumption, $P _ { 0 }$ factorizes over $\boldsymbol { x } \in \mathbb { R } ^ { d }$ with local components $P _ { 0 , x }$ , that is,

$$
P _ { 0 } ( T ) = \prod _ { x \in \mathbb { R } ^ { d } } P _ { 0 , x } ( T ( x ) ) ,\tag{71}
$$

where each $T ( x ) \in \Delta _ { k }$ . Since the exponential term also factorizes across x as

$$
e ^ { - \frac { 1 } { \lambda } \langle C , T \rangle _ { \mu } } = \prod _ { x \in \mathbb { R } ^ { d } } e ^ { - \frac { 1 } { \lambda } \sum _ { j = 1 } ^ { k } C _ { j } ( x ) T _ { j } ( x ) } ,\tag{72}
$$

this together implies that $q ^ { * }$ inherits the same factorization:

$$
\begin{array} { r l } & { \displaystyle q ^ { * } ( T ) = \prod _ { x \in \mathbb { R } ^ { d } } q _ { x } ^ { * } ( T ( x ) ) ; } \\ & { \displaystyle q _ { x } ^ { * } ( T ( x ) ) \propto P _ { 0 , x } ( T ( x ) ) e ^ { - \frac { 1 } { \lambda } \sum _ { j = 1 } ^ { k } C _ { j } ( x ) T _ { j } ( x ) } . } \end{array}\tag{73}
$$

To verify that it is indeed a suficient condition, the Kullback–Leibler divergence decomposes additively:

$$
D _ { \mathrm { K L } } ( q ^ { * } \Vert P _ { 0 } ) = \sum _ { x \in \mathbb { R } ^ { d } } D _ { \mathrm { K L } } ( q _ { x } ^ { * } \Vert P _ { 0 , x } ) = \mathbb { E } _ { q ^ { * } } \Bigl [ D _ { \mathrm { K L } } ( \pi _ { T } \Vert \mu \otimes \nu _ { k } ) \Bigr ] .\tag{74}
$$

Finally, if $P _ { 0 }$ factorizes with weights $\omega = ( \omega _ { 1 } , \ldots , \omega _ { k } )$ such that $\begin{array} { r } { P _ { 0 , x } ( T ( x ) ) = \sum _ { i } \omega _ { j } \delta _ { e _ { j } } ( T ( x ) ) } \end{array}$ , then each local factor $q _ { x } ^ { * }$ is categorical with unnormalized weights $\omega _ { j } e ^ { - C _ { j } ( x ) / \lambda }$ . Normalizing yields the softmax form:

$$
\bar { T } _ { j } ( x ) = \frac { \omega _ { j } e ^ { - C _ { j } ( x ) / \lambda } } { \sum _ { l = 1 } ^ { k } \omega _ { l } e ^ { - C _ { l } ( x ) / \lambda } } , \qquad j = 1 , \ldots , k ,\tag{75}
$$

which matches Equation 17.

The Mean-Field assumption has a clear probabilistic and geometric interpretation:

(A) Mean-field factorization and conditional independence. Assuming $P _ { 0 }$ factorizes over x imposes a mean-field approximation: each spatial location x contributes independently to the transport plan. Consequently, the optimal Gibbs posterior $q ^ { * }$ inherits the same factorized form, decoupling the optimization problem across all x. This drastically reduces complexity and aligns the free-energy functional with a sum of independent local free energies.

(B) Softmax as the variational expectation. The expected plan $\bar { T } ( x )$ emerges as a vectorvalued softmax, analogous to the E-step in the EM algorithm or the soft assignment in Sinkhorn-based clustering. Each $\bar { T } _ { j } ( x )$ represents the posterior probability that mass at x is transported to cluster $j ,$ weighted by the prior $\omega _ { j }$ and penalized by the transport cost $C _ { j } ( x )$ . The temperature λ controls the sharpness of this assignment: as $\lambda  0 , { \bar { T } } ( x )$ approaches a hard assignment, recovering the deterministic optimal transport map.

(C) Relation to variational inference. Under this factorization, minimizing the free-energy functional

$$
- \lambda F _ { f , P _ { 0 } } = \mathbb { E } _ { T \sim q ^ { * } } [ \langle C , T \rangle _ { \mu } ] + \lambda \mathbb { E } _ { T \sim q ^ { * } } [ D _ { \mathrm { K L } } ( \pi _ { T } | | \mu \otimes \nu _ { k } ) ]\tag{76}
$$

corresponds to performing an E-step in a mean-field variational EM scheme: $q ^ { * }$ plays the role of the posterior over transport assignments, and $\bar { T }$ acts as its expected suficient statistic.

Theorem 4.3 provides a rigorous link between the free-energy formulation of regularized optimal transport and the classical mean-field variational inference paradigm. The softmax update in (17) serves as the closed-form E-step, bridging Gibbs variational principles, entropy-regularized transport, and probabilistic clustering.

## A.8 Proof for Proposition 4.5

Proof. This proof elaborates Proposition 4.5 from Section 4. Let $\bar { T } ^ { ( t + 1 ) }$ be fixed from the E-step. Consider Sinkhorn Loss $\mathcal { F } _ { \lambda } ( Y , W , \bar { T } ^ { ( t + 1 ) } )$

For each cluster $j ,$ , the M-step optimizes

$$
\langle c ( x , y _ { j } ) , \bar { T } _ { j } ^ { ( t + 1 ) } \rangle _ { \mu } = \int _ { \mathbb { R } ^ { d } } \bar { T } _ { j } ^ { ( t + 1 ) } ( x ) \| x - y _ { j } \| ^ { 2 } d \mu ( x )\tag{77}
$$

with respect to $y _ { j }$ . Diferentiating and setting the gradient to zero gives

$$
2 \Big ( y _ { j } \int \bar { T } _ { j } ^ { ( t + 1 ) } ( x ) d \mu ( x ) - \int x \bar { T } _ { j } ^ { ( t + 1 ) } ( x ) d \mu ( x ) \Big ) = 0 ,\tag{78}
$$

which yields the update of $Y ^ { ( t + 1 ) }$

$$
y _ { j } ^ { ( t + 1 ) } = \frac { \int x \bar { T } _ { j } ^ { ( t + 1 ) } ( x ) d \mu ( x ) } { \int \bar { T } _ { j } ^ { ( t + 1 ) } ( x ) d \mu ( x ) } .\tag{79}
$$

Similarly, the weight $\omega _ { j }$ appears in the linear term of the $\mathrm { K L }$ divergence between the variational transport plan $\bar { T } ^ { ( t + 1 ) }$ and the discrete measure $\mu \otimes \nu _ { k }$ :

$$
{ \mathrm { K L } } ( \pi _ { \bar { T } ^ { ( t + 1 ) } } \parallel \mu \otimes \nu _ { k } ) = \sum _ { j = 1 } ^ { k } \int _ { \mathbb { R } ^ { d } } \bar { T } _ { j } ^ { ( t + 1 ) } ( x ) \log \frac { \bar { T } _ { j } ^ { ( t + 1 ) } ( x ) } { \omega _ { j } } d \mu ( x ) .\tag{80}
$$

Diferentiating with respect to $\omega _ { j }$ and setting the derivative to zero gives

$$
\frac { \partial } { \partial \omega _ { j } } \mathrm { K L } ( \bar { T } ^ { ( t + 1 ) } \parallel \mu \otimes \nu _ { k } ) = - \frac { 1 } { \omega _ { j } } \int _ { \mathbb { R } ^ { d } } \bar { T } _ { j } ^ { ( t + 1 ) } ( x ) d \mu ( x ) = 0 ,\tag{81}
$$

which yields the closed-form update of $W : = \{ w _ { j } \} _ { j = 1 } ^ { k }$ . The total mass assigned to cluster $j$ for the next step is

$$
\omega _ { j } ^ { ( t + 1 ) } = \int _ { \mathbb { R } ^ { d } } \bar { T } _ { j } ^ { ( t + 1 ) } ( x ) d \mu ( x ) .\tag{82}
$$

By definition, the M-step performs a block coordinate optimization over $Y$ and $W { : }$ :

$$
\mathcal { F } _ { \lambda } ( Y ^ { ( t + 1 ) } , W ^ { ( t ) } , \bar { T } ^ { ( t + 1 ) } ) \leq \mathcal { F } _ { \lambda } ( Y ^ { ( t ) } , W ^ { ( t ) } , \bar { T } ^ { ( t + 1 ) } ) ,\tag{83}
$$

$$
\begin{array} { r } { \mathcal { F } _ { \lambda } ( Y ^ { ( t + 1 ) } , W ^ { ( t + 1 ) } , \bar { T } ^ { ( t + 1 ) } ) \le \mathcal { F } _ { \lambda } ( Y ^ { ( t + 1 ) } , W ^ { ( t ) } , \bar { T } ^ { ( t + 1 ) } ) , } \end{array}\tag{84}
$$

which immediately implies

$$
\mathcal { F } _ { \lambda } ( Y ^ { ( t + 1 ) } , W ^ { ( t + 1 ) } , \bar { T } ^ { ( t + 1 ) } ) \leq \mathcal { F } _ { \lambda } ( Y ^ { ( t ) } , W ^ { ( t ) } , \bar { T } ^ { ( t + 1 ) } ) .\tag{85}
$$

Equality holds if and only if $( Y ^ { ( t ) } , W ^ { ( t ) } )$ are already minimizers along both blocks, i.e., they satisfy the first-order optimality conditions:

$$
\begin{array} { r } { \nabla _ { Y } \mathcal { F } _ { \lambda } ( Y ^ { ( t ) } , W ^ { ( t ) } , \bar { T } ^ { ( t + 1 ) } ) = 0 , } \\ { \nabla _ { W } \mathcal { F } _ { \lambda } ( Y ^ { ( t ) } , W ^ { ( t ) } , \bar { T } ^ { ( t + 1 ) } ) = 0 . } \end{array}\tag{86}
$$

This completes the proof.

## A.9 Proof of Proposition 4.6

Proof. Throughout this proof we use the squared–Euclidean cost $C _ { j } ( x ) = \| x - y _ { j } \| ^ { 2 } ~ ( { \mathrm { i . e . , } } ~ p = 2 )$ 2 consistent with the E–step and Alg. 1. Recall the soft assignment from Eq. (17):

$$
\begin{array} { l } { \displaystyle \bar { T } _ { j } ( x ) = \frac { \omega _ { j } e ^ { - C _ { j } ( x ) / \lambda } } { \sum _ { \ell = 1 } ^ { k } \omega _ { \ell } e ^ { - C _ { \ell } ( x ) / \lambda } } = \frac { a _ { j } ( x ) } { Z ( x ) } , } \\ { \displaystyle a _ { j } ( x ) : = \omega _ { j } e ^ { - \| x - y _ { j } \| ^ { 2 } / \lambda } , \quad Z ( x ) : = \sum _ { \ell = 1 } ^ { k } a _ { \ell } ( x ) . } \end{array}\tag{87}
$$

Since

$$
\begin{array} { l l } { \nabla _ { x } a _ { j } ( x ) = a _ { j } ( x ) \nabla _ { x } \big ( - \| x - y _ { j } \| ^ { 2 } / \lambda \big ) = - ( 2 / \lambda ) a _ { j } ( x ) ( x - y _ { j } ) ; } \\ { \nabla _ { x } Z ( x ) = \displaystyle \sum _ { \ell } \nabla _ { x } a _ { \ell } ( x ) = - ( 2 / \lambda ) \displaystyle \sum _ { \ell } a _ { \ell } ( x ) ( x - y _ { \ell } ) , } \end{array}\tag{88}
$$

Applying the product and chain rules gives

$$
\begin{array} { l } { { \nabla _ { x } \bar { T } _ { j } ( x ) = \displaystyle \frac { \nabla _ { x } a _ { j } } { Z } - \frac { a _ { j } } { Z ^ { 2 } } \nabla _ { x } Z = - \displaystyle \frac { 2 } { \lambda } \frac { a _ { j } } { Z } ( x - y _ { j } ) + \frac { 2 } { \lambda } \frac { a _ { j } } { Z } \sum _ { \ell = 1 } ^ { k } \frac { a _ { \ell } } { Z } ( x - y _ { \ell } ) } } \\ { { \displaystyle \phantom { \frac { 1 } { a _ { j } } } } } \\ { { \displaystyle \phantom { \frac { 1 } { a _ { j } } } = \frac { 2 } { \lambda } \bar { T } _ { j } ( x ) \displaystyle \sum _ { \ell = 1 } ^ { k } \bar { T } _ { \ell } ( x ) \left( y _ { j } - y _ { \ell } \right) } . } \end{array}\tag{89}
$$

This proves the stated gradient identity, i.e. equation 22 in main manuscript. For the norm bound, let $D _ { Y } : = \operatorname* { m a x } _ { r , s } \| y _ { r } - y _ { s } \|$ . Then

$$
\Big \| \sum _ { \ell = 1 } ^ { k } \bar { T } _ { \ell } ( x ) \left( y _ { j } - y _ { \ell } \right) \Big \| \leq \sum _ { \ell = 1 } ^ { k } \bar { T } _ { \ell } ( x ) \left\| y _ { j } - y _ { \ell } \right\| \leq D _ { Y } \sum _ { \ell = 1 } ^ { k } \bar { T } _ { \ell } ( x ) = D _ { Y } ,\tag{90}
$$

and hence $\Vert \nabla _ { x } \bar { T } _ { j } ( x ) \Vert \leq ( 2 / \lambda ) \bar { T } _ { j } ( x ) D _ { Y } \leq 2 D _ { Y } / \lambda$ . Note the weights ω enter only through $\bar { T }$ and cancel in the derivative formula; the bound therefore holds for non-uniform $W$ □

## A.10 Discussion for Stability

In this subsection we collect Lipschitz-type stability properties of the soft assignment map

$$
\bar { T } _ { j } ^ { ( \lambda ) } ( x ; Y , W ) = \frac { \omega _ { j } \exp \big ( - \| x - y _ { j } \| ^ { 2 } / \lambda \big ) } { \sum _ { \ell = 1 } ^ { k } \omega _ { \ell } \exp \big ( - \| x - y _ { \ell } \| ^ { 2 } / \lambda \big ) } .\tag{91}
$$

For brevity we write $\bar { T } _ { j } ( x )$ when the dependence on $( Y , W , \lambda )$ is clear.

Theorem A.3 (Lipschitz continuity in the data). For any $x , x ^ { \prime } \in \mathbb { R } ^ { d }$ and every $j \in [ k ]$ ，

$$
\left| \hat T _ { j } ( x ) - \hat T _ { j } ( x ^ { \prime } ) \right| \ \le \ \frac { 2 D _ { Y } } { \lambda } \left. x - x ^ { \prime } \right. .\tag{92}
$$

Proof. By Proposition 4.6, $\lVert \nabla _ { x } \bar { T } _ { j } ( x ) \rVert \leq 2 D _ { Y } / \lambda$ for all x. Apply the mean-value theorem along the segment $x ( t ) = x + t ( x ^ { \prime } - x ) , t \in [ 0 , 1 ]$ , and integrate over x, which completes the proof. □

Proposition A.4 (Partial derivatives with respect to centers). Fix $\boldsymbol { x } \in \mathbb { R } ^ { d }$ . For any j, $k \in [ k ]$

$$
\partial _ { y _ { k } } { \bar { T } } _ { j } ( x ) = { \frac { 2 } { \lambda } } { \bar { T } } _ { j } ( x ) \Big [ ( x - y _ { j } ) { \bf 1 } \{ j = k \} - { \bar { T } } _ { k } ( x ) ( x - y _ { k } ) \Big ] .\tag{93}
$$

Consequently, if x ranges in a compact set $K \subset \mathbb { R } ^ { d }$ , then for all $x \in K$ 7

$$
\left\| \partial _ { y _ { k } } \bar { T } _ { j } ( x ) \right\| \le \frac { 2 } { \lambda } ( R _ { X } + R _ { Y } ) ,\tag{94}
$$

where $\begin{array} { r } { { \cal R } _ { { \cal X } } : = \operatorname* { s u p } _ { x \in { \cal K } } \| x \| , { \cal R } _ { { \cal Y } } : = \operatorname* { m a x } _ { 1 \leq \ell \leq k } \| y _ { \ell } \| . } \end{array}$

Proof. Let $\bar { T } _ { j } ( x ) = a _ { j } ( x ) / Z ( x )$ , where

$$
a _ { j } ( x ) = \omega _ { j } \exp \big ( - \| x - y _ { j } \| ^ { 2 } / \lambda \big ) , \quad \mathrm { a n d } \quad Z ( x ) = \sum _ { \ell } a _ { \ell } ( x ) .
$$

Observe that only the term $a _ { k }$ depends on $y _ { k }$ . Diferentiating $a _ { k }$ with respect to y<sub>k</sub> gives

$$
\partial _ { y _ { k } } a _ { k } ( x ) = a _ { k } ( x ) \partial _ { y _ { k } } \big ( - \| x - y _ { k } \| ^ { 2 } / \lambda \big ) = \frac { 2 } { \lambda } a _ { k } ( x ) ( x - y _ { k } ) .
$$

Consequently,

$$
\partial _ { y _ { k } } Z ( x ) = { \frac { 2 } { \lambda } } a _ { k } ( x ) ( x - y _ { k } ) .
$$

Using the quotient rule for diferentiation of $\bar { T } _ { j } = a _ { j } / Z$ , we obtain

$$
\partial _ { y _ { k } } \bar { T } _ { j } ( x ) = \frac { Z ( x ) \partial _ { y _ { k } } a _ { j } ( x ) - a _ { j } ( x ) \partial _ { y _ { k } } Z ( x ) } { Z ( x ) ^ { 2 } } .
$$

Since $a _ { j }$ depends on $y _ { k }$ only when $j = k$ , we have

$$
\partial _ { y _ { k } } a _ { j } ( x ) = { \left\{ \begin{array} { l l } { { \displaystyle { \frac { 2 } { \lambda } } a _ { k } ( x ) ( x - y _ { k } ) , } } & { { \mathrm { i f ~ } } j = k , } \\ { { 0 , } } & { { \mathrm { i f ~ } } j \neq k . } \end{array} \right. }
$$

Substituting these expressions yields the stated identity for $\partial _ { y _ { k } } \bar { T } _ { j } ( x )$

Moreover, if $x \in K$ , then $\| x - y _ { \ell } \| \leq R _ { X } + R _ { Y }$ for all $\ell ,$ and by construction each $\bar { T } _ { j } ( x )$ satisfies $0 \leq \bar { T } _ { j } ( x ) \leq 1$ . These bounds, together with the expression for $\partial _ { y _ { k } } \bar { T } _ { j } ( x )$ , yield the claimed uniform estimate. □

Theorem A.5 (Lipschitz continuity in the centers on compact domains). Let $K \subset \mathbb { R } ^ { d }$ be compact, $x \in K$ , and $Y ^ { \prime } = \{ y _ { 1 } ^ { \prime } , \ldots , y _ { k } ^ { \prime } \}$ . Then for every $j \in [ k ]$

$$
\left| \bar { T } _ { j } ( x ; Y , W ) - \bar { T } _ { j } ( x ; Y ^ { \prime } , W ) \right| ~ \leq ~ \frac { 2 k } { \lambda } ~ \left( R _ { X } + R _ { Y } \right) \Delta _ { Y } ,\tag{95}
$$

where $\Delta _ { Y } : = \operatorname* { m a x } _ { 1 \leq \ell \leq k } \| y _ { \ell } - y _ { \ell } ^ { \prime } \|$

Proof. Consider the path $Y ( t ) = \{ y _ { \ell } ( t ) \}$ with $y _ { \ell } ( t ) = y _ { \ell } + t ( y _ { \ell } ^ { \prime } - y _ { \ell } ) , t \in [ 0 , 1 ]$ . By the chain rule and Proposition $\mathrm { A . 4 }$

$$
\frac { d } { d t } \bar { T } _ { j } ( x ; Y ( t ) , W ) = \sum _ { \ell = 1 } ^ { k } \big < \partial _ { y _ { \ell } } \bar { T } _ { j } ( x ; Y ( t ) , W ) , y _ { \ell } ^ { \prime } - y _ { \ell } \big > .\tag{96}
$$

For each $t ,$ the centers satisfy max<sub>ℓ</sub> $\begin{array} { r } { \| y _ { \ell } ( t ) \| \le R _ { Y } , \mathrm { s o } \ \| \partial _ { y _ { \ell } } \bar { T } _ { j } ( x ; Y ( t ) , W ) \| \le \frac 2 \lambda ( R _ { X } + R _ { Y } ) } \end{array}$ . Hence,

$$
\Big | \frac { d } { d t } \bar { T } _ { j } ( x ; Y ( t ) , W ) \Big | \leq \frac { 2 } { \lambda } ( R _ { X } + R _ { Y } ) \sum _ { \ell = 1 } ^ { k } \| y _ { \ell } ^ { \prime } - y _ { \ell } \| \leq \frac { 2 k } { \lambda } ( R _ { X } + R _ { Y } ) \Delta _ { Y } .\tag{97}
$$

Then integrate over $t \in [ 0 , 1 ]$ for both sides of the inequality, which completes the proof. □

Stability of the empirical M–step. Let $X _ { 1 : n } \subset K$ and define the suficient statistics

$$
\bar { T } _ { i j } : = \bar { T } _ { j } ( X _ { i } ; Y , W ) , \qquad S _ { j } : = \sum _ { i = 1 } ^ { n } \bar { T } _ { i j } , \qquad N _ { j } : = \sum _ { i = 1 } ^ { n } \bar { T } _ { i j } X _ { i } .\tag{98}
$$

thus the M–step center update is $y _ { i } ^ { \mathrm { n e w } } : = N _ { j } / S _ { j }$ , as defined in Def. 4.4. Consider perturbing one sample $X _ { k } \mapsto X _ { k } + h$ with $\left\| h \right\| \leq \delta$ . Let $\Delta T _ { k j } : = \bar { T } _ { j } ( X _ { k } + h ) - \bar { T } _ { j } ( X _ { k } ) , \Delta S _ { j } : = \Delta T _ { k j }$ , and $R _ { j , \mathrm { m a x } } : = \operatorname* { m a x } _ { 1 \leq i \leq n } \left\| X _ { i } - y _ { j } ^ { \mathrm { n e w } } \right\|$

Theorem A.6 (Per-sample leverage bound). With the notation above, for any $\delta \geq 0$

$$
\left\| y _ { j } ^ { n e w } ( X _ { k } + h ) - y _ { j } ^ { n e w } ( X _ { k } ) \right\| ~ \leq ~ \frac { \delta + \frac { 2 D _ { Y } } { \lambda } \delta R _ { j , \operatorname* { m a x } } } { S _ { j } - \frac { 2 D _ { Y } } { \lambda } \delta } .\tag{99}
$$

Equivalently, writing $p _ { j } : = S _ { j } / n$

$$
\left\| y _ { j } ^ { n e w } ( X _ { k } + h ) - y _ { j } ^ { n e w } ( X _ { k } ) \right\| \leq \frac { 1 } { n p _ { j } - \frac { 2 D _ { Y } } { \lambda } \delta } \left( 1 + \frac { 2 D _ { Y } } { \lambda } R _ { j , \operatorname* { m a x } } \right) \delta .\tag{100}
$$

In particular, if $\begin{array} { r } { \colon \delta \le \frac { \lambda } { 4 D _ { Y } } S _ { j } } \end{array}$ then

$$
\left. y _ { j } ^ { n e w } ( X _ { k } + h ) - y _ { j } ^ { n e w } ( X _ { k } ) \right. \leq \frac { 2 } { n p _ { j } } \left( 1 + \frac { 2 D _ { Y } } { \lambda } R _ { j , \operatorname* { m a x } } \right) \delta .\tag{101}
$$

Proof. Only the k-th terms change, and

$$
N _ { j } ^ { \prime } - N _ { j } = \bar { T } _ { j } ( X _ { k } + h ) \left( X _ { k } + h \right) - \bar { T } _ { j } ( X _ { k } ) X _ { k } = \bar { T } _ { j } ( X _ { k } + h ) h + \Delta T _ { k j } X _ { k } .\tag{102}
$$

Since $y _ { j } ^ { \mathrm { n e w } } = N _ { j } / S _ { j }$ , one checks

$$
y _ { j } ^ { \mathrm { n e w \prime } } - y _ { j } ^ { \mathrm { n e w } } = \frac { S _ { j } ( N _ { j } ^ { \prime } - N _ { j } ) - ( N _ { j } ) \Delta S _ { j } } { S _ { j } ( S _ { j } + \Delta S _ { j } ) } = \frac { \bar { T } _ { j } ( X _ { k } + h ) h + \Delta T _ { k j } ( X _ { k } - y _ { j } ^ { \mathrm { n e w } } ) } { S _ { j } + \Delta S _ { j } } .\tag{103}
$$

Taking norms, using $\bar { T } _ { j } \leq 1$ , Theorem A.3 to bound $| \Delta T _ { k j } | \le \left( 2 D _ { Y } / \lambda \right) \delta .$ , and $\begin{array} { r } { \| X _ { k } - y _ { j } ^ { \mathrm { n e w } } \| \leq R _ { j , \operatorname* { m a x } } , } \end{array}$ gives the first inequality. The remaining proof follow by substituting $S _ { j } = n p _ { j }$ and the stated using small-δ condition. □

Remark A.7 (Hard k-means lacks global Lipschitz continuity). Let $H _ { j } ( x ; Y ) = \mathbf { 1 } \{ j = \arg \operatorname* { m i n } _ { \ell } \| x -$ $y _ { \ell } \| \}$ be the hard assignment. Across a Voronoi boundary, $H _ { j } ( \cdot ; Y )$ jumps by $\Theta ( 1 )$ under arbitrarily small perturbations of x or $Y ;$ thus no finite global Lipschitz constant exists. By contrast, for any fixed $\lambda > 0$ , the soft assignments $\bar { T } ^ { ( \lambda ) }$ are globally Lipschitz in x (Theorem A.3) and locally Lipschitz in $Y$ (Theorem A.5) on compact data domains.

## B Extended Experimental Setup Details

## B.1 Implementation Details

Hardware environment. Experiments run on a workstation with an Intel Core i7-12700K CPU and an NVIDIA RTX 5090 GPU. Unless noted otherwise, we execute GPU-aware routines on the cuda backend and report wall-clock times measured on this system.

Software environment. All experiments are implemented on Ubuntu 22.04 with Python 3.11.5, PyTorch 2.7.1+cu128 (compiled against CUDA 12.8), and TorchVision 0.22.1+cu128. Supporting libraries include NumPy 1.26.2, SciPy 1.11.4, Pandas 2.1.1, Matplotlib 3.8.0, Seaborn 0.13.0, and scikit-learn 1.2.2.

Hyperparameters. Unless stated otherwise, all methods share a stopping rule of max iter= 1000, a tolerance of $\mathtt { e p s } = 1 0 ^ { - 2 }$ on the Frobenius norm between successive centre updates, and a mini-batch size of batch size= 1000. Centres are initialised by sampling from the dataset via the random initialiser. Gradient-based optimisation in WCSL uses the Adam optimiser with learning rate $1 0 ^ { - 1 }$ and a StepLR scheduler that multiplies the learning rate by 0.99 each epoch. Unless specified otherwise, the Sinkhorn regularisation parameter is fixed to 0.01.

Evaluation metrics. We approximate entropic Wasserstein distances using Sinkhorn loss with entropy parameter 0.01, squared Euclidean costs, numerical tolerance $1 0 ^ { - 6 }$ for simulated data and $1 0 ^ { - 5 }$ for image data, and up to 1000 Sinkhorn iterations.

Repetitions. Each experiment is run with fixed random seeds for reproducibility, and we report the mean over multiple independent runs to mitigate variance.

## B.2 Experiment Details for Sec. 5.1

Synthetic Gaussian data. We generate $n = 1 0 ^ { 4 }$ points in $d = 1 0 0$ from a Gaussian distribution whose covariance matrix is assembled from a randomly sampled orthonormal basis.

MNIST benchmark. We evaluate on the MNIST handwritten digit benchmark [LeCun et al., 2002] using its standard train/test split.

## B.3 Experiment Details for Sec. 5.5

Goal and high-level protocol. This experiment evaluates whether a coreset $\nu _ { k }$ preserves downstream interpretability when used as the background distribution required by SHAP explainers. We train a predictive model $f$ on the training split, and then compute Shapley/SHAP attributions on the test split using either: (i) the full training set as background (reference), or (ii) a coreset of size $k { = } 6 4$ constructed from the same training split (approximation). We then report attribution reconstruction errors between these two SHAP outputs.

Datasets, splits, and preprocessing. We follow the setup of Jethani et al. [2022] using the shap package datasets: Census (tabular classification) and Diabetes (tabular regression). We use an 80/20 train/test split; for classification we use a stratified split, while for regression we use a random split. All features are standardized using StandardScaler fitted on the training split and applied to both splits.

Predictive model and SHAP explainer. On each dataset, we fit a gradient-boosted tree model (scikit-learn) on the 80% training split. Unless otherwise noted, we use 400 estimators, learning rate 0.05, and maximum depth 3. We compute attributions with shap.TreeExplainer [Lundberg and Lee, 2017]. For Census, we explain a scalar output (e.g., the score/probability of the positive class); for Diabetes, we explain the regression output. All methods share the same trained model; only the background distribution difers.

Shapley values with a background distribution. Let $\boldsymbol { x } \in \mathbb { R } ^ { d }$ be a test point and let ν be a background distribution on $\mathbb { R } ^ { d }$ . Define the standard “hybrid” value function for a subset of features $S \subseteq [ d ]$ by

$$
v ( S ; x , \nu ) : = \mathbb { E } _ { y \sim \nu } \Big [ f \big ( x _ { S } , y _ { [ d ] \backslash S } \big ) \Big ] ,\tag{104}
$$

where $( x _ { S } , y _ { [ d ] \backslash S } )$ denotes the vector obtained by taking features in S from x and the remaining features from a background draw $y .$ The Shapley value for feature $j$ is then

$$
\phi _ { j } ( x ; \nu ) = \sum _ { S \subseteq [ d ] \backslash \{ j \} } { \frac { | S | ! ( d - | S | - 1 ) ! } { d ! } } { \Big ( } v ( S \cup \{ j \} ; x , \nu ) - v ( S ; x , \nu ) { \Big ) } ,\tag{105}
$$

which is the unique attribution satisfying eficiency, symmetry, dummy, and additivity [Shapley, 1953]. In practice, directly evaluating (105) is combinatorial, and TreeExplainer provides an eficient computation specialized to tree ensembles [Lundberg and Lee, 2017].

Background distributions: full data vs. coreset. Let the training set be $\{ x _ { i } ^ { \mathrm { t r } } \} _ { i = 1 } ^ { n _ { \mathrm { t r } } }$ and define its empirical measure

$$
\mu _ { \mathrm { t r } } : = \frac { 1 } { n _ { \mathrm { t r } } } \sum _ { i = 1 } ^ { n _ { \mathrm { t r } } } \delta _ { x _ { i } ^ { \mathrm { t r } } } .\tag{106}
$$

Each coreset method constructs a discrete weighted measure supported on k atoms:

$$
\nu _ { k } : = \sum _ { m = 1 } ^ { k } \omega _ { m } \delta _ { y _ { m } } , \qquad \omega _ { m } \geq 0 , \sum _ { m = 1 } ^ { k } \omega _ { m } = 1 .\tag{107}
$$

For EMS, both $\{ y _ { m } \}$ and $\left\{ \omega _ { m } \right\}$ are learned. For K-means and WCSL, the support points are learned but weights are uniform $( \omega _ { m } = 1 / k )$ . For Random, we sample k points from the training set with uniform weights.

With $\nu _ { k }$ as background, the value function becomes a finite weighted average:

$$
v ( S ; x , \nu _ { k } ) = \sum _ { m = 1 } ^ { k } \omega _ { m } f { \big ( } x { \big . } S , y _ { m , [ d ] \backslash S } { \big ) } .\tag{108}
$$

Consequently, Shapley values under $\nu _ { k }$ decompose into weighted contributions from each atom:

$$
\phi _ { j } ( x ; \nu _ { k } ) = \sum _ { m = 1 } ^ { k } \omega _ { m } \phi _ { j } ( x ; y _ { m } ) ,\tag{109}
$$

where $\phi _ { j } ( x ; y _ { m } )$ denotes the Shapley value computed with a singleton background distribution $\delta _ { y _ { m } }$ (equivalently, using $y _ { m }$ to fill missing features in the hybrid evaluation).

SHAP matrices and reconstruction metrics. Let the test set be $\{ x _ { i } ^ { \mathrm { t e } } \} _ { i = 1 } ^ { n _ { \mathrm { t e } } }$ . We form the SHAP attribution matrices

$$
\begin{array} { r } { \Phi ^ { \mathrm { f u l l } } \in \mathbb { R } ^ { n _ { \mathrm { t e } } \times d } , \qquad \Phi ^ { \mathrm { c o r e s e t } } \in \mathbb { R } ^ { n _ { \mathrm { t e } } \times d } , } \end{array}\tag{110}
$$

where $\Phi _ { i j } ^ { \mathrm { f u l l } } : = \phi _ { j } ( x _ { i } ^ { \mathrm { t e } } ; \mu _ { \mathrm { t r } } )$ and $\Phi _ { i j } ^ { \mathrm { c o r e s e t } } : = \phi _ { j } ( x _ { i } ^ { \mathrm { t e } } ; \nu _ { k } )$ . We report three reconstruction errors:

$$
\mathrm { M e a n A E } : = \frac { 1 } { n _ { \mathrm { t e } } d } \big | \big | \Phi ^ { \mathrm { c o r e s e t } } - \Phi ^ { \mathrm { f u l l } } \big | \big | _ { 1 } ,\tag{111}
$$

$$
\begin{array} { r } { \mathrm { M a x A E : } = \left. \Phi ^ { \mathrm { c o r e s e t } } - \Phi ^ { \mathrm { f u l l } } \right. _ { \infty } , } \end{array}\tag{112}
$$

$$
\mathrm { B V E } : = \left| b ( \nu _ { k } ) - b ( \mu _ { \mathrm { t r } } ) \right| ,\tag{113}
$$

where $\| \cdot \| _ { \infty }$ denotes the elementwise maximum absolute deviation and $b ( \nu )$ is the SHAP base value (background expected prediction), estimated empirically as

$$
b ( \mu _ { \mathrm { t r } } ) = \frac { 1 } { n _ { \mathrm { t r } } } \sum _ { i = 1 } ^ { n _ { \mathrm { t r } } } f ( x _ { i } ^ { \mathrm { t r } } ) , \qquad b ( \nu _ { k } ) = \sum _ { m = 1 } ^ { k } \omega _ { m } f ( y _ { m } ) .\tag{114}
$$

This base value is the constant in the SHAP decomposition $\begin{array} { r } { f ( x ) = b ( \nu ) + \sum _ { j = 1 } ^ { d } \phi _ { j } ( x ; \nu ) } \end{array}$ , so BVE captures how well the coreset preserves the explainer’s reference level.

Repetitions and randomness control. All reported numbers average over five runs with independent random seeds. Across runs, randomness may arise from data splitting (when applicable) and/or coreset initialization (e.g., random initialization for K-means and EMS). See table 4 for the results.

## B.4 Limitations and Future Directions: Large-Scale Image Datasets

A limitation of this work is that we do not evaluate EMS on large-scale image datasets or image distillation tasks. This is by design. Our discrepancy is based on squared Euclidean geometry and entropic OT costs, which are well suited to tabular data and controlled synthetic settings, but pixel-wise similarity is generally misaligned with semantic similarity in images. As a result,

Table 4: Performance comparison on real-world datasets. Each entry reports distance / runtime (seconds).
<table><tr><td rowspan="3">Dataset</td><td>k</td><td>Random</td><td>K-means</td><td colspan="2">WCSL</td><td colspan="4">EMS</td></tr><tr><td></td><td></td><td></td><td colspan="2"> $\lambda = 1 0 ^ { - 2 }$ </td><td> $\lambda = 0$ </td><td> $\lambda = 1 0 ^ { - 3 }$ </td><td> $\lambda = 1 0 ^ { - 2 }$ </td><td> $\lambda = 1 0 ^ { - 1 }$ </td></tr><tr><td>distance time</td><td>distance</td><td>/ time</td><td>distance / time</td><td>distance</td><td>/ time distance</td><td>time</td><td>distance / time</td><td>distance time</td></tr><tr><td rowspan="5">Transactions</td><td>20</td><td>74.47</td><td>46.47 1.43</td><td>46.45</td><td>76.57 46.47</td><td>/1.37</td><td>46.48 /1.66</td><td>46.48 /1.45</td><td>46.47 1.85</td></tr><tr><td>50</td><td>68.85</td><td>44.88 1.63</td><td>44.82 148.02</td><td>44.88</td><td>/ 2.08</td><td>44.87 / 1.95</td><td>44.87 / 2.26</td><td>44.84 2.20</td></tr><tr><td>100</td><td>65.65</td><td>43.67 /1.76</td><td>43.56 203.11</td><td>43.66</td><td>2.24</td><td>43.65 / 2.81</td><td>43.65 / 3.00</td><td>43.61 / 3.00</td></tr><tr><td>200</td><td>63.13</td><td>42.48 /3.15</td><td>42.26 307.59</td><td>42.42</td><td>2.86</td><td>42.42 /3.65</td><td>42.40 /6.36</td><td>42.34 /6.46</td></tr><tr><td>500</td><td>59.54</td><td>40.94 / 4.01</td><td>40.44 / 633.87</td><td>40.77</td><td>/3.50</td><td>40.77 /4.21</td><td>40.75 /8.47</td><td>40.65 16.14</td></tr><tr><td rowspan="5">ChemReact100</td><td>20</td><td>167.64</td><td>104.38 /0.065</td><td>94.80 /44.82</td><td>96.71</td><td>/0.079</td><td>97.41 /0.081</td><td>96.93 / 0.108</td><td>97.25 /0.104</td></tr><tr><td>50</td><td>149.77</td><td>110.82 0.111</td><td>93.69 143.95</td><td>98.02</td><td>/0.176</td><td>97.65 / 0.135</td><td>97.34 / 0.125</td><td>97.06 / 0.135</td></tr><tr><td>100</td><td>140.06</td><td>108.89 /0.207</td><td>94.07 224.75</td><td>97.80</td><td>/0.214</td><td>97.83 /0.221</td><td>97.94 / 0.245</td><td>97.95 /0.285</td></tr><tr><td>200</td><td>131.66</td><td>106.79 / 0.460</td><td>93.97 364.31</td><td>98.39</td><td>0.400</td><td>98.32 /0.452</td><td>98.38 /0.434</td><td>98.45 / 0.776</td></tr><tr><td>500</td><td>124.58</td><td>104.74 0.522</td><td>93.67 689.74</td><td>99.09</td><td>0.579</td><td>99.05 0.639</td><td>99.08 / 0.792</td><td>99.15 /1.45</td></tr><tr><td rowspan="5">Music</td><td>20</td><td>467.49</td><td>357.23 /0.048</td><td>219.44 /20.88</td><td>212.53</td><td>/ 0.042</td><td>212.58 0.041</td><td>212.64 /0.047</td><td>212.73 /0.068</td></tr><tr><td>50</td><td>379.75</td><td>403.59 /0.081</td><td>214.71 /51.60</td><td>205.09</td><td>0.071</td><td>205.62 /0.104</td><td>205.77 /0.098</td><td>205.12 /0.125</td></tr><tr><td>100</td><td>352.66</td><td>360.83 /0.147</td><td>213.85 /87.80</td><td>197.09</td><td>/ 0.162</td><td>197.80 0.122</td><td>197.32 /0.126</td><td>197.42 /0.302</td></tr><tr><td>200 500</td><td>333.36 303.75</td><td>405.85 0.254 358.80 0.362</td><td>205.08 /155.39 194.03</td><td>190.86 186.82</td><td>/0.251 /0.323</td><td>191.42 0.200 187.14 / 0.361</td><td>191.50 /0.201</td><td>191.29 /0.560</td></tr><tr><td></td><td></td><td></td><td>364.59</td><td></td><td></td><td></td><td>187.72 /0.444</td><td>187.37 /0.716</td></tr></table>

Wasserstein or Sinkhorn objectives applied directly in pixel space often fail to produce meaningful visual or semantic summaries, even when transport costs are small.

A natural extension is to apply EMS in a learned representation space rather than in pixel space. Let $\psi ( \cdot )$ denote a fixed or lightly tuned feature extractor, such as a pretrained vision encoder, and consider the pushforward measure $\mu _ { \psi } : = \psi _ { \# } \mu$ . Running EMS on $\mu _ { \psi }$ yields a compact weighted measure

$$
\nu _ { k } ^ { \psi } = \sum _ { j = 1 } ^ { k } \omega _ { j } \delta _ { z _ { j } } ,
$$

where the atoms $\{ z _ { j } \}$ summarize the dataset geometry in feature space. This connects closely to feature-space barycenter constructions, moment-matching objectives, and recent dataset distillation approaches based on Wasserstein-type criteria [Liu et al., 2025].

Two practical issues then arise: (i) how to realize feature-space atoms as images $( \mathrm { e . g . }$ , via nearestneighbor selection or by optimizing synthetic images under a perceptual loss with appropriate priors), and (ii) how to evaluate semantic fidelity using downstream performance or representation-level metrics rather than pixel-based distances. Addressing these issues would extend the applicability of EMS to modern vision tasks while preserving its core advantages of fast, stable, EM-style updates.

Pairwise Wasserstein distance between coresets  
![](images/5370416bfcef8dc89b2ca05ee6bdeb70b515543264cdb3a3d8307637203c2ec5.jpg)

Pairwise Wasserstein distance between coresets  
![](images/47a01b9bc844a0825ccd67a337e6240a428e9a61dcd16236d72c86f7f6353486.jpg)  
Figure 3: Pairwise entropic Wasserstein distances across 10 random initializations for Gaussian mixture dataset with R = 2 (top) and R = 5 (bottom).