# Wasserstein Filtering: A Sample Selection Method for Robust Distribution Learning

Yikai Xu<sup>∗</sup>, Zhao Chen<sup>†</sup>, and Jian Huang<sup>‡</sup>

August 14, 2026

## Abstract

Given a dataset where a portion of the samples are contaminated, our goal is to recover the underlying clean population distribution. To this end, we propose Wasserstein Filtering (WF), a novel sample selection framework that discards a fraction of suspicious samples and estimates the target distribution using the empirical measure of the remaining data. The core insight is to select a subset of samples whose empirical distribution maximizes its Wasserstein distance to the fully contaminated empirical distribution, thereby preferentially isolating and removing geometrically influential outliers. To render this optimization computationally tractable, we introduce three algorithms: a marginal screening scheme, SinkMarg, and two joint optimization algorithms, SinkWF and SlicedWF, leveraging entropic optimal transport and sliced Wasserstein approximations, respectively. On the theoretical front, we introduce the Far Exclusion and Local Projection (FELP) contamination model, which characterizes corruptions consisting of well-separated outliers and locally indistinguishable perturbations. Under this model, we prove that the WF estimator achieves minimax optimality over distribution families with bounded covariance. Extensive numerical experiments on synthetic datasets, benchmark anomaly detection suites, and robust generative learning with difusion models demonstrate that WF serves as a highly practical, model-agnostic preprocessing tool. It delivers competitive outlier detection performance and provides substantial downstream benefits for generative modeling under heavy contamination.

Keywords: robust distribution estimation; Wasserstein distance; optimal transport; outlier detection

## 1 Introduction

Let $\mathcal { D } = \{ \mathbf { x } _ { 1 } , \ldots , \mathbf { x } _ { n } \}$ be a set of i.i.d. samples drawn from a population distribution P<sup>∗</sup>. We consider an adversarial corruption model where an adversary can inspect the dataset and modify up to an α-fraction of the samples, yielding a contaminated dataset $\widetilde { \cal D } = \{ \widetilde { \bf x } _ { 1 } , \ldots , \widetilde { \bf x } _ { n } \}$ . The contamination budget $\alpha \in [ 0 , 1 )$ constrains the adversary such that:

$$
\sum _ { i = 1 } ^ { n } \mathbb { I } _ { \left\{ \widetilde { \mathbf { x } } _ { i } \neq \mathbf { x } _ { i } \right\} } \leq \alpha n .
$$

This formulation is commonly referred to as adversarial or Total Variation (TV) contamination, reflecting the fact that the TV distance between the clean and contaminated empirical measures satisfies $\| \widetilde { P } _ { n } - P _ { n } \| _ { \mathrm { T V } } \leq \alpha$ (Diakonikolas et al., 2019; Nietert et al., 2023). Here, $\begin{array} { r } { P _ { n } = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \delta _ { \mathbf { x } _ { i } } } \end{array}$ is the clean empirical measure, $\begin{array} { r } { \widetilde { P } _ { n } = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \delta _ { \widetilde { \mathbf { x } } _ { i } } } \end{array}$ is the contaminated empirical measure, and $\delta _ { \mathbf { x } }$ represents the Dirac delta measure centered at x.

In this work, we address the following fundamental question in robust statistics:

Can we design an accurate estimator and a computationally tractable algorithm to recover the true distribution $P ^ { * }$ under the Wasserstein-1 metric, given samples subject to Total Variation (TV) contamination?

We adopt the Wasserstein-1 distance as our metric of distributional discrepancy because it metrizes weak convergence, naturally tolerates support mismatch, and is highly sensitive to the geometry of outliers (Nietert et al., 2023). Furthermore, by considering the TV contamination model, we impose minimal assumptions on the noise structure and allow for worst-case, adaptive, and malicious corruptions of the data.

While existing methods, such as iterative filtering (Nietert et al., 2024), achieve strong theoretical convergence guarantees under TV contamination, they sufer from practical limitations. Specifically, their algorithms rely heavily on the unrealistic assumption that $P ^ { * }$ is isotropic, and they require prior knowledge of unknown spectral parameters associated with the covariance of $P ^ { * }$

To address these limitations, we propose an algorithm that dispenses with the isotropic assumption and operates entirely independently of the parameters of $P ^ { * }$ . The trade-of for this flexibility is that we establish the theoretical optimality of our estimator under TV contamination subject to mild, physically meaningful structural constraints that make the optimization landscape amenable to analysis.

## 1.1 Key Contributions

In this work, given a contamination proportion α, we formulate robust distribution learning as a constrained sample selection problem. Specifically, we identify and filter out suspicious outliers, using the empirica measure of the remaining samples as a robust representative of the clean population distribution $P ^ { * }$

From this perspective, we introduce the Wasserstein Filtering (WF) estimator. Conceptually, the WF estimator searches over all possible subsets of the contaminated data of a size determined by the contamination rate α. It then selects the subset whose empirical distribution maximizes the Wasserstein-1 distance to the overall contaminated empirical distribution. By framing outlier detection as this targeted optimization problem, the estimator systematically isolates and discards the samples that exert the most severe geometric distortion on the empirical measure.

Our main contributions are both methodological and theoretical, summarized as follows:

1. Novel Algorithmic Frameworks: We propose three distinct algorithms to compute the WF estimator: one marginal scheme (SinkMarg) and two joint optimization schemes (SinkWF and SlicedWF). These algorithms scale flexibly across diferent sample sizes and dimensionalities.

2. Rigorous Theoretical Guarantees: Beyond showing strong empirical performance on par with state-ofthe-art heuristic outlier detection methods, we establish the minimax optimality of the WF estimator under robust distribution estimation. Furthermore, we provide elementary optimization convergence rates for our joint algorithms.

3. Task- and Model-Agnostic Utility: A unique benefit of our sample selection formulation is its taskindependence. It serves as a modular preprocessing filter that cleans the dataset prior to downstream modeling. In Section 5, we demonstrate this versatility by pairing our filter with modern deep generative models (DDPM). Unlike existing approaches that modify the loss function in a model-dependent manner (Balaji et al., 2020; Nietert et al., 2023), our approach is entirely model-agnostic and can safeguard any downstream generative framework.

The rest of this paper is organized as follows. In Section 2, we introduce our Wasserstein Filtering framework, detailing both the mathematical estimator and its three algorithmic realizations. Rigorous theoretical properties, including minimax optimality and optimization convergence, are established in Section 3. In Section 4, we present systematic numerical experiments on synthetic and benchmark datasets to validate and compare the empirical performance of our methods. Section 5 demonstrates the practical utility of our framework as a task-agnostic data-filtering step for modern generative models. Finally, Section 6 concludes the paper.

## 1.2 Related Work

Robust statistics has accumulated a rich body of seminal research, with the fundamental concepts dating back to the pioneering work of Huber (1964). Over the decades, substantial eforts have been devoted to designing robust estimation and inference procedures capable of handling corrupted data; for a comprehensive modern review, see Loh (2025).

In this context, we highlight the literature most relevant to robust distribution learning. The semina work of Donoho and Liu (1988) first formulated the robust distribution estimation problem under the population TV-contamination model and introduced Minimum Distance Estimation (MDE) at the population level. More recently, Zhu et al. (2022) systematically generalized MDE, utilizing it to design mean and covariance estimation procedures that achieve population minimax optimality alongside strong empirical guarantees.

For nonparametric density estimation, Liu and Gao (2019) established the minimax optimal rate of pointwise density estimation under Huber’s contamination model, demonstrating that kernel density estimators (KDE) can achieve this rate when equipped with an optimal bandwidth. However, this optimal bandwidth relies heavily on the smoothness of the underlying clean density and the contamination proportion, both of which are typically unknown in practice. Chao and Dobriban (2023) derived analogous minimax rates under Wasserstein contamination. On the methodological front, several algorithmic variants have been proposed to robustify density estimation, including the integration of Huber-type loss functions with KDE (Kim and Scott, 2012) and the application of the median-of-means principle to kernel frameworks (Humbert et al., 2022).

Partial Optimal Transport (OT), also referred to as unbalanced OT or robust OT, is closely related to our framework. Originally developed within the mathematical analysis community (Cafarelli and McCann, 2010; Figalli, 2010; Liero et al., 2018; Chizat et al., 2018), partial OT was subsequently introduced to the statistics and machine learning communities to relax the strict mass-preserving constraint of standard OT.

Recent methodological developments have leveraged this relaxation to handle noisy data. For instance, Balaji et al. (2020) defined a robust Wasserstein distance by incorporating a $\chi ^ { 2 } .$ -divergence penalty on the marginal distributions. By replacing the standard Wasserstein distance in Wasserstein GANs (WGAN) (Arjovsky et al., 2017) with this robust formulation, they trained generative models that are structurally insensitive to outliers. Mukherjee et al. (2021) substituted the $\chi ^ { 2 } .$ -divergence with a Total Variation (TV) penalty, deriving an equivalent formulation that thresholds the cost matrix, which is significantly more computationally convenient.

Subsequently, the theoretical foundations of this TV-penalized robust Wasserstein distance have been rigorously established, spanning measure-theoretic dualities, optimization landscapes, and statistical convergence properties, particularly its convergence rates in robust distribution estimation when plugged into the MDE framework (Ma et al., 2025; Nietert et al., 2023). Parallel to these theoretical advances, the computational aspects of unbalanced OT (Pham et al., 2020; Nguyen et al., 2023) and its associated MDE optimization schemes (Nietert et al., 2024) have also been extensively investigated.

Another closely related line of research is unsupervised outlier detection. Unlike robust statistics, this field is often heavily empirically driven, focusing on practical heuristic design and computational scalability rather than formal problem abstractions or minimax theoretical guarantees. Classical outlier detection methods typically assign an anomaly score to each sample based on criteria such as local density, distance, recursive isolation, reconstruction error, or marginal extremeness.

For instance, COPOD (Li et al., 2020) leverages empirical copulas and tail probabilities to construct a computationally eficient and highly interpretable outlier score. More recently, Deep Isolation Forest (DIF) (Xu et al., 2023) enhances traditional isolation-based detection by applying random neural network representations prior to isolation-tree partitioning. Large-scale benchmarks, such as ADBench (Han et al., 2022), have systematically evaluated these diverse anomaly detection algorithms across tabular and graph data.

## 2 Methodology

## 2.1 Estimator

To define the proposed estimator, we first recall the definition of the Wasserstein distance (Villani et al., 2009). Let $( \mathcal { X } , d )$ be a Polish metric space, and let $\mathcal { Q } ( \mathcal { X } )$ denote the set of all Borel probability measures on X. For $p \geq 1$ , we define the space of probability measures with finite $p \textmd { - }$ th moment as

$$
\mathcal { Q } _ { p } ( \mathcal { X } ) : = \left\{ \mu \in \mathcal { Q } ( \mathcal { X } ) : \int _ { \mathcal { X } } d ( \mathbf { x } , \mathbf { x } _ { 0 } ) ^ { p } \mathrm { d } \mu ( \mathbf { x } ) < \infty \mathrm { ~ f o r ~ s o m e ~ } \mathbf { x } _ { 0 } \in \mathcal { X } \right\} .\tag{1}
$$

For any $\mu , \nu \in \mathcal { Q } _ { p } ( \mathcal { X } )$ , the Wasserstein-p distance between them is defined as:

$$
W _ { p } ( \mu , \nu ) : = \left( \operatorname* { i n f } _ { \pi \in \Pi ( \mu , \nu ) } \int _ { \mathcal { X } \times \mathcal { X } } d ( \mathbf x , \mathbf y ) ^ { p } \mathrm d \pi ( \mathbf x , \mathbf y ) \right) ^ { \frac { 1 } { p } } ,
$$

where $\Pi ( \mu , \nu )$ denotes the set of all Borel probability measures (or couplings) on $\mathcal { X } \times \mathcal { X }$ having marginal distributions $\mu$ and $\nu ,$ respectively. Throughout this paper, unless specified otherwise, we assume $\boldsymbol { \mathcal { X } } \subseteq \mathbb { R } ^ { d }$ Let d be the standard Euclidean metric $( \mathrm { i . e . , ~ } d ( \mathbf { x } , \mathbf { y } ) = \| \mathbf { x } - \mathbf { y } \| _ { 2 } )$ , and set $p = 1$ . That is, we restrict our focus to the Wasserstein-1 distance on $\mathbb { R } ^ { d }$ , denoted simply as $W _ { 1 }$

Let $\widetilde { \mathcal { D } } = \{ \widetilde { \mathbf { x } } _ { 1 } , \hdots , \widetilde { \mathbf { x } } _ { n } \} \subset \mathcal { X } \subseteq \mathbb { R } ^ { d }$ be a set of possibly contaminated data points, and let $\begin{array} { r } { \widetilde { P } _ { n } = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \delta _ { \widetilde { \mathbf { x } } _ { i } } } \end{array}$ denote the corresponding contaminated empirical measure. For a given contamination fraction $\alpha \in [ 0 , 1 )$ we seek to identify a target distribution within the family of sub-empirical measures obtained by removing αn outliers from $\widetilde { \mathcal { D } } _ { \cdot }$ . Specifically, we look for the sub-measure that maximizes the Wasserstein distance to the contaminated empirical measure ${ \widetilde { P } } _ { n }$

$$
\widehat { P } _ { \mathrm { W F } } = \underset { P \in \mathscr { P } _ { \alpha } } { \mathrm { a r g m a x } } W _ { 1 } ( \widetilde { P } _ { n } , P ) ,\tag{2}
$$

where the candidate family ${ \mathcal P } _ { \alpha }$ of trimmed empirical measures is defined as:

$$
\mathcal { P } _ { \alpha } = \left\{ \mu \in \mathcal { Q } ( \mathcal { X } ) : \mu = \frac { 1 } { ( 1 - \alpha ) n } \sum _ { \tilde { \mathbf { x } } _ { i } \in \tilde { \mathcal { D } } _ { \alpha } } \delta _ { \tilde { \mathbf { x } } _ { i } } , \ \widetilde { \mathcal { D } } _ { \alpha } \subset \widetilde { \mathcal { D } } , \ | \widetilde { \mathcal { D } } _ { \alpha } | = ( 1 - \alpha ) n \right\} .
$$

By reformulating (2) in vector form, our goal is to find an optimal probability mass vector ${ \widehat { \mathbf { p } } } \in \mathbb { R } ^ { n }$ such that:

$$
\widehat { \mathbf { p } } = \underset { \mathbf { p } \in \Delta _ { \alpha } } { \mathrm { a r g m a x } } W _ { 1 } ( \widetilde { P } _ { n } , P _ { \mathbf { p } } ) ,\tag{3}
$$

where $\begin{array} { r } { P _ { \mathbf { p } } = \sum _ { i = 1 } ^ { n } p _ { i } \delta _ { \widetilde { \mathbf { x } } } } \end{array}$ is the discrete measure induced by the weights $\mathbf { p } = ( p _ { 1 } , \ldots , p _ { n } ) ^ { \top }$ . Here, $\Delta _ { \alpha }$ represents

the scaled, discretized simplex:

$$
\Delta _ { \alpha } = \left\{ \mathbf { v } \in \mathbb { R } ^ { n } : \mathbf { v } ^ { \top } \mathbf { 1 } = 1 , \ v _ { i } \in \left\{ 0 , \frac { 1 } { ( 1 - \alpha ) n } \right\} \ \mathrm { f o r ~ a l l } \ i \in [ n ] \right\} .\tag{4}
$$

As the discrete optimization problem (3) is NP-hard, we relax it to its convex hull:

$$
\widehat { \bf p } _ { \mathrm { W F } } = \underset { { \bf p } \in \mathrm { c o n v } ( \Delta _ { \alpha } ) } { \mathrm { a r g m a x } } W _ { 1 } ( \widetilde { P } _ { n } , P _ { \bf p } ) ,\tag{5}
$$

where the convex hull of $\Delta _ { \alpha }$ is given by

$$
\operatorname { c o n v } ( \Delta _ { \alpha } ) = \left\{ \mathbf { v } \in \mathbb { R } ^ { n } : \mathbf { v } ^ { \top } \mathbf { 1 } = 1 , 0 \leq v _ { i } \leq { \frac { 1 } { ( 1 - \alpha ) n } } , \forall i \in [ n ] \right\} .\tag{6}
$$

The optimal solution to (5), denoted by ${ \widehat { \mathbf { p } } } _ { \mathrm { W F } } ,$ , serves as our estimated mass vector. Once $\widehat { \mathbf { p } } _ { \mathrm { W F } }$ is obtained, we filter out contaminated samples based on its entries. Specifically, we estimate the clean distribution $P ^ { * }$ using the empirical distribution of the retained samples:

$$
\widehat { P } ^ { * } = \frac { 1 } { ( 1 - \alpha ) n } \sum _ { i : \widehat { p } _ { \mathrm { W F } , i } > \tau _ { \alpha } } \delta _ { \widetilde { \mathbf { x } } _ { i } } ,
$$

where $\tau _ { \alpha }$ is the α-quantile of the entries of $\widehat { \mathbf { p } } _ { \mathrm { W F } } = ( \widehat { p } _ { \mathrm { W F , 1 } } , \ldots , \widehat { p } _ { \mathrm { W F } , n } ) ^ { \top }$ . Since $\widehat { P } ^ { * } \in { \mathcal { P } } _ { \alpha }$ by construction, it serves as a natural discrete approximation of $\hat { P } _ { \mathrm { W F } }$ . In the following, we present one algorithm to solve the discrete formulation (3) and two algorithms to solve the relaxed formulation (5).

## 2.2 Algorithms

## 2.2.1 SinkMarg

By expressing the Wasserstein-1 distance in its primal form, we can reformulate the discrete optimization problem (3) as a discrete optimal transport problem:

$$
\widehat { \mathbf { p } } = \underset { \mathbf { p } \in \Delta _ { \alpha } } { \mathrm { a r g m a x } } \ \underset { \mathbf { T } \in \Pi \left( \frac { 1 } { n } \mathbf { 1 } , \mathbf { p } \right) } { \mathrm { m i n } } \langle \mathbf { T } , \mathbf { C } \rangle ,\tag{7}
$$

where $\mathbf { C } \in \mathbb { R } _ { + } ^ { n \times n }$ is the cost matrix with entries $C _ { i j } = \| \widetilde { \bf x } _ { i } - \widetilde { \bf x } _ { j } \| _ { 2 }$ representing the Euclidean distance between the i-th and j-th samples, and $\Pi ( \mathbf { u } , \mathbf { v } ) = \{ \mathbf { T } \in \mathbb { R } _ { + } ^ { n \times n } : \mathbf { T 1 } = \mathbf { u } , \mathbf { T } ^ { \top } \mathbf { 1 } = \mathbf { v } \}$ denotes the set of feasible transport plans.

Since computing the exact Wasserstein distance via linear programming is computationally expensive, we instead employ its entropy-regularized approximation (Cuturi, 2013):

$$
W _ { 1 , \lambda } ( \widetilde { P } _ { n } , P _ { \mathbf { p } } ) = \operatorname* { m i n } _ { \mathbf { T } \in \Pi \left( \frac { 1 } { n } \mathbf { 1 } , \mathbf { p } \right) } \langle \mathbf { T } , \mathbf { C } \rangle + \lambda \sum _ { i , j \in [ n ] } T _ { i j } \log ( T _ { i j } ) ,
$$

where $\lambda > 0$ is the regularization parameter, and $T _ { i j }$ is the $( i , j )$ -th entry of the transport plan T.

Let $\mathbf { p } _ { - i } \in \mathbb { R } ^ { n }$ be a probability vector whose i-th entry is 0 and all other entries are equal to $\scriptstyle { \frac { 1 } { n - 1 } }$ . A natural heuristic for estimating $\widehat { \mathbf p }$ is to assign zero mass to the αn points that yield the largest leave-one-out (LOO) scores: $W _ { 1 , \lambda } ( \widetilde { P } _ { n } , P _ { { \bf p } _ { - i } } )$ for $i \in [ n ]$ . The intuition behind this approach is straightforward: if removing a single sample causes a significant shift in the distribution, that sample is highly likely to be a contaminated outlier.

Based on this intuition, we introduce a marginal screening algorithm to solve (3), as outlined in Algorithm 1. As demonstrated in our numerical experiments, this algorithm performs well when the outliers are isolated.

However, due to its purely marginal nature, it fails to capture the group structure of clustered outliers and incurs a heavy computational burden when n is large.

Algorithm 1: Sinkhorn Marginal Screening Algorithm   
Input: $\widetilde { \mathcal { D } } , \alpha \in ( 0 , 1 ) , \lambda \in \mathbb { R } _ { + }$   
1: Normalize $\widetilde { \mathcal { D } }$   
2: for $i = 1 , \ldots , n$ do   
3: Compute $s _ { i } = W _ { 1 , \lambda } ( \widetilde { P } _ { n } , P _ { { \bf p } _ { - i } } )$   
4: end   
5: Thresholding: filter $\widetilde { \mathcal { D } }$ to obtain $S = \{ \widetilde { \mathbf { x } } _ { i } | s _ { i } < \tau _ { 1 - \alpha } , i \in [ n ] \}$ where $\tau _ { 1 - \alpha }$ is the $1 - \alpha$   
quantile of leave-one-out scores $\{ s _ { i } , i \in [ n ] \}$   
Output: S

## 2.2.2 SinkWF

To solve the relaxed formulation (5), we consider its entropy-regularized version:

$$
\widehat { \mathbf { p } } _ { \mathrm { S i n k W F } } = \underset { \mathbf { p } \in \mathrm { c o n v } ( \Delta _ { \alpha } ) } { \mathrm { a r g m a x } } W _ { 1 , \lambda } ( \widetilde { P } _ { n } , P _ { \mathbf { p } } ) ,\tag{8}
$$

where

$$
W _ { 1 , \lambda } ( \widetilde { P } _ { n } , P _ { \mathbf { p } } ) = \operatorname* { m i n } _ { \mathbf { T } \in \Pi \left( \frac { 1 } { n } \mathbf { 1 } , \mathbf { p } \right) } \langle \mathbf { T } , \mathbf { C } \rangle + \lambda \sum _ { i , j \in [ n ] } T _ { i j } \log ( T _ { i j } ) .
$$

Our proposed optimization scheme is detailed in Algorithm 2. To leverage gradient-based optimization, we transform the constrained problem (8) into an unconstrained, smooth formulation.

First, we reparameterize the probability vector p using the softmax function to ensure it lies on the simplex: $\mathbf { p } ( \mathbf { b } ) = \mathrm { s o f t m a x } ( \mathbf { b } )$ for $\mathbf { b } \in \mathbb { R } ^ { n }$ . Second, we incorporate the upper-bound constraint of the capped simplex $\mathrm { c o n v } ( \Delta _ { \alpha } )$ as a penalty term added to the objective:

$$
\mathcal { L } ( \mathbf { b } ) = - W _ { 1 , \lambda } ( \widetilde { P } _ { n } , P _ { \mathrm { s o f t m a x } ( \mathbf { b } ) } ) + \beta \mathbf { 1 } ^ { \top } \operatorname { R e L U } \left( \mathrm { s o f t m a x } ( \mathbf { b } ) - \frac { 1 } { ( 1 - \alpha ) n } \mathbf { 1 } \right) ,
$$

where $\mathrm { R e L U } ( \mathbf { x } ) = \operatorname* { m a x } ( \mathbf { x } , \mathbf { 0 } )$ is applied element-wise, and $\beta > 0$ is a penalty parameter.

To update b via gradient descent, we need to compute the gradient $\mathbf { g } _ { r } = \nabla _ { \mathbf { b } } \mathcal { L } ( \mathbf { b } _ { r } )$ at the r-th iteration. Rather than backpropagating directly through the Sinkhorn iterations using automatic diferentiation (e.g., PyTorch’s AutoGrad), which is highly memory-intensive, we exploit the duality theory of optimal transport. Specifically, the gradient of the entropic Wasserstein loss with respect to the marginal distribution is given by:

$$
\frac { \partial W _ { 1 , \lambda } ( \widetilde { P } _ { n } , P _ { \mathbf { p } } ) } { \partial \mathbf { p } } = \mathbf { v } ^ { * } ,
$$

where $\mathbf { v } ^ { * } \in \mathbb { R } ^ { n }$ is the optimal dual potential associated with p, which is a direct byproduct of the Sinkhorn algorithm (Sinkhorn, 1964; Cuturi, 2013). Using this dual formulation to compute the gradient g<sub>r</sub> bypasses the need to store the computational graph of the Sinkhorn iterations, significantly reducing both time and memory complexity.

After solving the entropic optimal transport problem to obtain the optimal plan $\mathbf { T } _ { r } .$ , one might consider using the unregularized cost $\langle \mathbf { T } _ { r } , \mathbf { C } \rangle$ as a more accurate approximation of the true Wasserstein distance, as suggested by Luise et al. (2018). However, computing the gradient $\partial \langle \mathbf { T } _ { r } , \mathbf { C } \rangle / \partial \mathbf { p }$ requires either backpropa gating through the Sinkhorn steps or solving a system of linear equations with $\mathcal { O } ( n ^ { 3 } )$ complexity, making it computationally prohibitive.

Using the obtained gradient $\mathbf { g } _ { r }$ , we employ the Adam optimizer (Kingma and Ba, 2014) to update ${ \bf b } _ { r }$ for $R$ iterations. To enhance the stability of our estimator, we repeat this optimization process $T$ times with diferent initializations. We then take the average of the $T$ resulting probability vectors, $T ^ { - 1 } \sum _ { t = 1 } ^ { T }$ softmax $( \mathbf { b } _ { R } ^ { ( t ) } )$ , as the final screening score, filtering out the αn samples with the smallest scores as suspected outliers. The complete procedure for SinkWF is detailed in Algorithm 2.

In addition to this penalty-based approach for handling the capped simplex constraint, we also provide a projection-based optimization variant, alongside an automatic selection strategy for the hyperparameter $\alpha .$ These complementary details are discussed in Appendix A.1-A.2.

Algorithm 2: Sinkhorn Wasserstein Filtering Algorithm   
Input: $\widetilde { \mathcal { D } } , \alpha \in ( 0 , 1 ) , \mathbf { b } _ { 0 } , \lambda , \gamma \in \mathbb { R } _ { + } , R , T \in \mathbb { Z } _ { + }$   
1: Normalize $\widetilde { \mathcal { D } }$   
2: for $t = 1 , \dots , T$ do   
3: for $r = 0 , \ldots , R - 1$ do   
4: Solve the following problem:   
$\mathbf { T } _ { r } = \underset { \mathbf { T } \in \Pi \left( \frac { 1 } { n } \mathbf { 1 } , \mathrm { s o f t m a x } ( \mathbf { b } _ { r } ) \right) } { \operatorname { a r g m i n } } \left. \mathbf { T } , \mathbf { C } \right. + \lambda \sum _ { i , j \in [ n ] } t _ { i j } \log ( t _ { i j } )$ (9)   
Compute gradient on $ { \mathbf { b } } _ { r }$ :   
$\mathbf { g } _ { r } = \left\{ \mathbf { v } ^ { * } - \frac { \partial \beta \mathbf { 1 } ^ { \top } \mathrm { R e l u } \left( \mathrm { s o f t m a x } ( \mathbf { b } _ { r } ) - \frac { 1 } { ( 1 - \alpha ) n } \mathbf { 1 } \right) } { \partial \mathrm { s o l f t m a x } ( \mathbf { b } _ { r } ) } \right\} \frac { \partial \mathrm { s o l f t m a x } ( \mathbf { b } _ { r } ) } { \partial \mathbf { b } _ { r } }$ (10)   
Adam optimizer update:   
$\mathbf { b } _ { r + 1 } = \mathrm { A d a m } ( \gamma , \mathbf { b } _ { r } , \mathbf { g } _ { j , j \leq r } )$ (11)   
5: end   
6: Save $\mathbf { h } _ { t } = \mathrm { s o f t m a x } ( \mathbf { b } _ { R } )$   
7: end   
8: Thresholding: filter $\widetilde { \mathcal { D } }$ to obtain $S = \{ \mathbf { x } _ { i } | \bar { h } _ { i } > \tau _ { \alpha } , i \in [ n ] \}$ where $\bar { h } _ { i }$ is the ith entry of   
$\begin{array} { r } { \bar { \mathbf { h } } = \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \mathbf { h } _ { t } , \tau _ { \alpha } } \end{array}$ is the α quantile of $\bar { h } _ { i } , i \in [ n ]$   
Output: S

## 2.2.3 SlicedWF

In the SinkWF formulation (8), the cost matrix C must be explicitly constructed and stored. This incurs an $\mathcal { O } ( n ^ { 2 } )$ space and time complexity, which can be computationally prohibitive for large-scale datasets. To address this limitation, we consider a more memory-eficient alternative to the classical Wasserstein distance: the Sliced Wasserstein (SW) distance (Peyr´e et al., 2019). The SW distance is defined as:

$$
W _ { 1 , S } ( \widetilde { P } _ { n } , P _ { \mathbf { p } } ) = \frac { 1 } { S } \sum _ { i = 1 } ^ { S } W _ { 1 } \left( P _ { \mathbf { u } _ { i } ^ { \top } \widetilde { \mathbf { x } } , \widetilde { \mathbf { x } } \sim \widetilde { P } _ { n } } , P _ { \mathbf { u } _ { i } ^ { \top } \mathbf { y } , \mathbf { y } \sim P _ { \mathbf { p } } } \right) ,
$$

where $\mathbb { S } ^ { d - 1 }$ denotes the d-dimensional unit sphere, and $\mathbf { u } _ { i } \sim \mathbb { S } ^ { d - 1 }$ are random projection directions. The parameter S controls the number of projections used to approximate the continuous Sliced Wasserstein

distance. By projecting the d-dimensional data onto these random 1-dimensional directions, computing the SW distance simplifies to averaging S independent 1-dimensional Wasserstein distances.

The corresponding estimator under this formulation is given by:

$$
\widehat { \mathbf { p } } _ { \mathrm { S l i c e d W F } } = \underset { \mathbf { p } \in \mathrm { c o n v } ( \Delta _ { \alpha } ) } { \mathrm { a r g m a x } } W _ { 1 , S } ( \widetilde { P } _ { n } , P _ { \mathbf { p } } ) ,\tag{12}
$$

where

$$
W _ { 1 , S } ( \widetilde { P } _ { n } , P _ { \mathbf { p } } ) = \frac { 1 } { S } \sum _ { i = 1 } ^ { S } W _ { 1 } \left( P _ { \mathbf { u } _ { i } ^ { \top } \widetilde { \mathbf { x } } , \widetilde { \mathbf { x } } \sim \widetilde { P } _ { n } } , P _ { \mathbf { u } _ { i } ^ { \top } \mathbf { y } , \mathbf { y } \sim P _ { \mathbf { p } } } \right) .
$$

A well-known property of the 1-dimensional Wasserstein-1 distance is that it can be computed in closed form via the cumulative distribution functions (Peyr´e et al., 2019):

$$
W _ { 1 } \left( P _ { \mathbf { u } _ { i } ^ { \top } \widetilde { \mathbf { x } } , \widetilde { \mathbf { x } } \sim \widetilde { P } _ { n } } , P _ { \mathbf { u } _ { i } ^ { \top } \mathbf { y } , \mathbf { y } \sim P _ { \mathbf { p } } } \right) = \int _ { 0 } ^ { 1 } \left| F _ { \mathbf { u } _ { i } ^ { \top } \widetilde { \mathbf { x } } , \widetilde { \mathbf { x } } \sim \widetilde { P } _ { n } } ^ { - 1 } ( t ) - F _ { \mathbf { u } _ { i } ^ { \top } \mathbf { y } , \mathbf { y } \sim P _ { \mathbf { p } } } ^ { - 1 } ( t ) \right| d t ,
$$

where $F ^ { - 1 } ( t )$ denotes the quantile function (inverse CDF) of the projected distributions.

This 1-dimensional distribution matching can be solved sorted-by-index in O(n log n) time and ${ \mathcal { O } } ( n )$ space. Consequently, the Sliced Wasserstein approach bypasses the storage and computation of the $\scriptstyle { \mathcal { O } } ( n ^ { 2 } )$ pairwise cost matrix C, drastically reducing memory requirements and running time. However, to maintain approximation accuracy, the required number of projections S generally scales with the data dimension $d .$

The complete optimization process is detailed in Algorithm 1 in the Appendix A.3. An alternative projection-based variant for enforcing the capped simplex constraint conv $\left( \Delta _ { \alpha } \right)$ is also deferred to Appendix A.1.

## 3 Theoretical Properties

## 3.1 Convergence of the Estimator

In this section, we establish the statistical convergence of $\widehat { P } _ { \mathrm { W F } }$ to the true distribution $P ^ { * }$ . We begin by introducing several essential concepts required to define our data contamination model. Recall that $( \mathcal { X } , d )$ is a Polish metric space $( { \mathrm { i . e . } }$ , a complete and separable metric space), $\mathcal { Q } ( \mathcal { X } )$ denotes the space of all probability measures on $x ,$ and $\mathcal { Q } _ { p } ( \mathcal { X } )$ is defined as in (1). While our theoretical results are established within this general framework, it is helpful to keep in mind the standard setting where $\mathcal { X }$ is a compact subset of $\mathbb { R } ^ { d }$ equipped with the Euclidean metric $d ( \mathbf { x } , \mathbf { y } ) = \| \mathbf { x } - \mathbf { y } \| _ { 2 }$ , and $p = 1$

Definition 1 $( W _ { p }$ Resilience (Nietert et al., 2023, Definition 1)). Let $0 \leq \alpha < 1$ and $\rho \geq 0$ . A distribution $\mu \in \mathcal { Q } _ { p } ( \mathcal { X } )$ is said to be $( \rho , \alpha )$ -resilient with respect to the Wasserstein-p distance if

$$
W _ { p } ( \mu , \mu ^ { \prime } ) \le \rho
$$

for all $\mu ^ { \prime } \in \mathcal { Q } _ { p } ( \mathcal { X } )$ such that $\begin{array} { r } { \mu ^ { \prime } \leq \frac { 1 } { 1 - \alpha } \mu } \end{array}$ . We denote the family of all $( \rho , \alpha )$ -resilient distributions in $\mathcal { Q } _ { p } ( \mathcal { X } )$ by $\mathcal { W } _ { p } ( \rho , \alpha )$

The inequality between the two measures, $\begin{array} { r } { \mu ^ { \prime } \leq \frac { 1 } { 1 - \alpha } \mu . } \end{array}$ is understood pointwise; that is, it holds for any Borel set $B \subseteq { \mathcal { X } }$ . Intuitively, this condition implies that $\mu ^ { \prime }$ is obtained by removing up to an α-fraction of the probability mass from $\mu$ and subsequently normalizing the remaining measure.

Resilience characterizes the structural stability of a distribution: it guarantees that the core of the distribution remains close to the original distribution in the Wasserstein metric, even after an adversary discards an α-fraction of its mass. Originally introduced to study robust mean estimation under adversaria

contamination (Steinhardt et al., 2018), this concept has recently been generalized to a distribution-free setting (Zhu et al., 2022; Nietert et al., 2023).

Definition 2 (Support Total Variation Distance). For two probability distributions $P _ { 1 } , P _ { 2 } \in \mathcal { Q } ( \mathcal { X } )$ , we define the support total variation (STV) distance as:

$$
\begin{array} { r } { \| P _ { 1 } - P _ { 2 } \| \mathrm { s r v } = \operatorname* { i n f } \Big \{ \alpha \in [ 0 , 1 ) : \exists \bar { S } \subseteq \mathrm { s u p p } ( P _ { 1 } ) \cap \mathrm { s u p p } ( P _ { 2 } ) \mathrm { ~ s . t . ~ } P _ { 1 } ( \bar { S } ) = P _ { 2 } ( \bar { S } ) = 1 - \alpha \mathrm { ~ a n d ~ } P _ { 1 } | _ { \bar { S } } = P _ { 2 } | _ { \bar { S } } \Big \} , } \end{array}
$$

where supp(·) denotes the support of a probability measure, and $P | _ { \bar { S } }$ represents the restriction of $P$ to ${ \bar { S } } ,$ defined as $P | _ { \bar { S } } ( B ) = P ( B \cap \bar { S } )$ for any Borel set $B \subseteq { \mathcal { X } }$

The STV distance is a structurally restricted variant of the classical total variation (TV) distance. Recall that if two probability distributions satisfy $\| P _ { 1 } - P _ { 2 } \| _ { \mathrm { T V } } \leq \alpha .$ , then $P _ { 2 }$ can be viewed as being obtained from $P _ { 1 }$ by first removing an arbitrary α-fraction of the mass from $P _ { 1 }$ and replacing it with an arbitrary α-fraction of mass elsewhere (Diakonikolas et al., 2019).

Building upon this deletion-addition decomposition, the STV distance enforces the additional constraint that the mass removal and addition must occur on disjoint supports. In other words, $\mathrm { i f } \ \| P _ { 1 } - P _ { 2 } \| _ { \mathrm { S T V } } \leq \alpha _ { \mathrm { m } }$ there exists a shared support $\bar { S }$ containing 1 − α of the total probability mass, such that $P _ { 2 }$ is obtained by deleting the α-mass on supp( $P _ { 1 } ) \setminus \bar { S }$ and adding a new α-mass restricted to ${ \mathcal { X } } \setminus { \bar { S } }$ . It immediately follows from this definition that $\| P _ { 1 } - P _ { 2 } \| _ { \mathrm { T V } } \leq \| P _ { 1 } - P _ { 2 } \| _ { \mathrm { S T V } }$

## 3.1.1 Population Setting

To isolate the geometric properties of our approach from finite-sample statistical fluctuations, we first analyze the problem in the population setting (corresponding to the infinite-sample limit).

Let the clean distribution $P ^ { * }$ belong to a $( \rho , \alpha )$ -resilient family: $P ^ { * } \in \mathcal G \subseteq \mathcal W _ { 1 } ( \rho , \alpha )$ , and suppose the observed contaminated distribution $\widetilde { P }$ satisfies $\| \widetilde { P } - P ^ { * } \| _ { \mathrm { S T V } } \leq \alpha$ . We define the population-level estimator as:

$$
\widehat { P } _ { \mathrm { W F } } = \widetilde { P } _ { S _ { \mathrm { W F } } } : = \frac { 1 } { 1 - \alpha } \widetilde { P } | _ { S _ { \mathrm { W F } } } ,
$$

where the optimal support $S _ { \mathrm { W F } }$ is given by:

$$
S _ { \mathrm { W F } } = \underset { S \subseteq \mathrm { s u p p } ( \widetilde { P } ) } { \mathrm { a r g m a x } } W _ { 1 } ( \widetilde { P } _ { S } , \widetilde { P } ) ,
$$

and $\begin{array} { r } { { \widetilde P _ { S } } = \frac { 1 } { { \widetilde P ( S ) } } { \widetilde P } { \widetilde  P | _ { S } } } \end{array}$ denotes the normalized restriction of $\widetilde { P }$ to $S$ . This formulation directly mirrors our empirical estimator, where the discrete support points each carry mass $1 / n$

We now introduce our proposed Far Exclusion and Local Projection (FELP) contamination model at the population level:

Definition 3 (Population FELP Contamination). Given $P ^ { * } \in \mathcal { W } _ { 1 } ( \rho , \alpha )$ , we say that the contaminated distribution $\widetilde { P }$ belongs to the FELP contamination family $\mathcal { C } _ { P ^ { * } } ^ { \mathrm { F E L P } } ( \alpha , \rho )$ if the following three conditions hold:

1. Support TV Bound: The total variation of the support satisfies: $\| \widetilde { P } - P ^ { * } \| _ { \mathrm { S T V } } \leq \alpha$ . Let $\bar { S }$ be the shared support of mass $1 - \alpha$ promised by the definition of the STV distance.

2. Far Exclusion (FE): Let $E _ { \rho } : = \{ \mathbf { x } \in \mathrm { s u p p } ( \widetilde { P } ) \setminus \bar { S } \mid$ d(x, supp $\left( P ^ { * } \right) ) > \rho \}$ define the region of “far outliers”. There exists a subset $S _ { + } \subseteq \operatorname { s u p p } ( \tilde { P } )$ with $\widetilde P ( S _ { + } ) = 1 - \alpha$ and $S _ { + } \cap E _ { \rho } = \emptyset$ such that:

$$
\operatorname* { s u p } _ { S \subseteq \operatorname* { s u p p } ( \widetilde { P } ) , \widetilde { P } ( S ) = 1 - \alpha } W _ { 1 } ( \widetilde { P } _ { S } , \widetilde { P } ) < W _ { 1 } ( \widetilde { P } _ { S _ { + } } , \widetilde { P } ) .\tag{13}
$$

3. Local Projection $( \mathrm { L P } ) { \mathrm { : } }$ : Let $N _ { \rho } : = \{ \mathbf x \in \mathrm { s u p p } ( \widetilde P ) \setminus \bar { S } \mid d ( \mathbf x , \mathrm { s u p p } ( P ^ { * } ) ) \le \rho \}$ denote the region of “near outliers”. There exists a measurable projection map $T : N _ { \rho } \to \operatorname { s u p p } ( P ^ { * } )$ such that:

$$
d ( \mathbf { x } , T ( \mathbf { x } ) ) \leq \rho \quad \mathrm { a n d } \quad T _ { \sharp } \widetilde { P } | _ { N _ { \rho } } \leq P ^ { * } ,\tag{14}
$$

where $T _ { \sharp }$ denotes the push-forward operator.

In the FELP contamination model, the STV distance restricts the corruptive mass to support-wise perturbations bound by a budget of $\alpha .$ . The far exclusion and local projection conditions further dictate the precise geometric structure of the contamination under which our estimator achieves minimax optimality:

• Far Outliers $( E _ { \rho } )$ : Must be suficiently far from the clean support that they are guaranteed to be excluded by our Wasserstein-maximization estimator.

• Near Outliers $( N _ { \rho } )$ : Lie close to the clean support. Their impact on the distribution estimation error is negligible even if they are not removed, as they can be modeled as local perturbations from the clean distribution via the mass-non-increasing preimage map $T ^ { - 1 }$

By definition, we have the following nested relationship among the contamination families:

$$
\mathcal { C } _ { P ^ { * } } ^ { \mathrm { F E L P } } ( \alpha , \rho ) \subseteq \big \{ P \in \mathcal { Q } ( \mathcal { X } ) : \| P - P ^ { * } \| _ { \mathrm { S T V } } \leq \alpha \big \} \subseteq \big \{ P \in \mathcal { Q } ( \mathcal { X } ) : \| P - P ^ { * } \| _ { \mathrm { T V } } \leq \alpha \big \} .
$$

This nesting shows that the FELP model is structurally more restrictive than the classical population total variation (TV) contamination model. Nevertheless, FELP remains suficiently rich to capture realistic scenarios. Indeed, adversarial contamination naturally decomposes in this manner: one component consists of far, well-separated outliers, while the other consists of near, indistinguishable inliers. This far-and-local decomposition sharing a similar spirit with the local perturbation and global adversary contamination models of Nietert et al. (2024), as well as the mild and gross contamination frameworks of Clark and McNicholas (2024).

In what follows, we provide a concrete, low-dimensional example to demonstrate when the FELP model conditions are satisfied.

Example (Uniform Cube in the Population Setting). Let $S ^ { * } = [ 0 , 1 ] ^ { d }$ and consider the clean distribution $P ^ { * } = \operatorname { U n i f } ( S ^ { * } )$ . We corrupt the data via a split-and-replace mechanism that deletes clean subregions in $S ^ { * }$ and replaces them with corrupted mass:

$$
\begin{array} { r l } { S _ { \mathrm { l o c } } ^ { \mathrm { i n } } = [ 0 , l ] \times [ 0 , 1 ] ^ { d - 1 } \quad  \quad S _ { \mathrm { l o c } } ^ { \mathrm { o u t } } = [ 1 , 1 + l ] \times [ 0 , 1 ] ^ { d - 1 } , } \\ { S _ { \mathrm { f a r } } ^ { \mathrm { i n } } = [ l , 2 l ] \times [ 0 , 1 ] ^ { d - 1 } \quad  \quad \mathbf { x } _ { \mathrm { f a r } } ^ { \mathrm { o u t } } = ( L , 1 , \ldots , 1 ) ^ { \top } , } \end{array}
$$

where $\begin{array} { r } { 0 < \alpha = 2 l < \frac { 1 } { 2 } } \end{array}$ , and $L > ( 1 + \sqrt { d } ) \lor C _ { l , d }$ for a constant $C _ { l , d }$ depending on d and l. We then define the contaminated probability measure as:

$$
\widetilde { P } = P ^ { * } | _ { S ^ { * } \setminus ( S _ { \mathrm { l o c } } ^ { \mathrm { i n } } \cup S _ { \mathrm { f a r } } ^ { \mathrm { i n } } ) } + l \mathrm { U n i f } ( S _ { \mathrm { l o c } } ^ { \mathrm { o u t } } ) + l \delta _ { \mathbf { x } _ { \mathrm { f a r } } ^ { \mathrm { o u t } } } .
$$

Under this construction, we have $\widetilde { P } \in \mathcal { C } _ { P ^ { * } } ^ { \mathrm { F E L P } } ( \alpha , \sqrt { d } )$

Proof sketch. By construction, $\bar { S } = S ^ { * } \setminus ( S _ { \mathrm { l o c } } ^ { \mathrm { i n } } \cup S _ { \mathrm { f a r } } ^ { \mathrm { i n } } )$ is the valid shared support between $\widetilde { P }$ and $P ^ { * }$ , yielding $\| \widetilde { P } - P ^ { * } \| _ { \mathrm { S T V } } \leq \alpha$

For any $P ^ { \prime } \in \mathcal { Q } _ { 1 } ( \mathcal { X } )$ satisfying $\begin{array} { r } { P ^ { \prime } \leq \frac { 1 } { 1 - \alpha } P ^ { * } } \end{array}$ , we have:

$$
W _ { 1 } ( P ^ { \prime } , P ^ { * } ) \leq \mathrm { d i a m } ( S ^ { * } ) \| P ^ { \prime } - P ^ { * } \| _ { \mathrm { T V } } \leq \alpha \sqrt { d } \leq \sqrt { d } ,
$$

which implies $P ^ { * } \in \mathcal { W } _ { 1 } ( \sqrt { d } , \alpha )$

Setting $\rho = { \sqrt { d } } ,$ and since $L > 1 + { \sqrt { d } }$ , we identify the near and far outlier regions as $N _ { \rho } = S _ { \mathrm { l o c } } ^ { \mathrm { o u t } }$ and $E _ { \rho } = \{ \mathbf { x } _ { \mathrm { f a r } } ^ { \mathrm { o u t } } \}$ , respectively. For any $\mathbf { x } \in S _ { \mathrm { l o c } } ^ { \mathrm { o u t } }$ , we define the projection map $T ( \mathbf { x } ) = \mathbf { x } - \mathbf { e } _ { 1 }$ , where $\mathbf { e } _ { 1 } = ( 1 , 0 , \ldots , 0 ) ^ { \top }$ . This yields:

$$
\begin{array} { r } { d ( \mathbf { x } , T ( \mathbf { x } ) ) = 1 \leq \sqrt { d } = \rho \quad \mathrm { a n d } \quad T _ { \sharp } \widetilde { P } | _ { N _ { \rho } } = T _ { \sharp } \big ( l \mathrm { U n i f } ( S _ { \mathrm { l o c } } ^ { \mathrm { o u t } } ) \big ) = l \mathrm { U n i f } ( S _ { \mathrm { l o c } } ^ { \mathrm { i n } } ) \leq P ^ { * } , } \end{array}
$$

satisfying the local projection condition.

Next, choosing $S _ { + } = S ^ { * } \setminus ( S _ { \mathrm { l o c } } ^ { \mathrm { i n } } \cup S _ { \mathrm { f a r } } ^ { \mathrm { i n } } )$ in the FELP definition, the Wasserstein distance $W _ { 1 } ( \widetilde { P } _ { S _ { + } } , \widetilde { P } )$ is lower bounded by the cost of transporting the mass l at the atom ${ \bf x } _ { \mathrm { f a r } } ^ { \mathrm { o u t } }$ to $S _ { + }$

$$
W _ { 1 } ( \widetilde { P } _ { S _ { + } } , \widetilde { P } ) \geq l \cdot d ( \mathbf { x } _ { \mathrm { f a r } } ^ { \mathrm { o u t } } , S _ { + } ) = l ( L - 1 ) .
$$

Conversely, for any admissible support $S \subseteq \operatorname { s u p p } ( \widetilde { P } )$ with ${ \widetilde { P } } ( S ) = 1 - \alpha$ and $\widetilde P ( S \cap E _ { \rho } ) > 0$ , the set $S$ must contain the atom ${ \bf x } _ { \mathrm { f a r } } ^ { \mathrm { o u t } }$ , meaning $\widetilde { P } _ { S } ( \{ \mathbf { x } _ { \mathrm { f a r } } ^ { \mathrm { o u t } } \} ) = \frac { l } { 1 - \alpha }$ . To upper bound $W _ { 1 } ( \widetilde { P } _ { S } , \widetilde { P } )$ , we construct a feasible transport plan from $\widetilde { P } _ { S }$ to $\widetilde { P }$ as follows:

1. Keep l mass at the atom ${ \bf x } _ { \mathrm { f a r } } ^ { \mathrm { o u t } }$ in place (incurring zero cost).

2. Transport the excess mass $\frac { \alpha l } { 1 - \alpha }$ at ${ \bf x } _ { \mathrm { f a r } } ^ { \mathrm { o u t } }$ to the region $S _ { + } \cup S _ { \mathrm { l o c } } ^ { \mathrm { o u t } } = [ 2 l , 1 + l ] \times [ 0 , 1 ] ^ { d - 1 }$ , incurring a cost of at most $\begin{array} { r } { \frac { \alpha l } { 1 - \alpha } \sqrt { ( L - 2 l ) ^ { 2 } + d - 1 } } \end{array}$

3. Transport the remaining mass $1 - { \frac { l } { 1 - \alpha } }$ within $S _ { + } \cup S _ { \mathrm { l o c } } ^ { \mathrm { o u t } }$ , incurring a cost of at most $\left( 1 - \frac { l } { 1 - \alpha } \right) \sqrt { ( 1 - l ) ^ { 2 } + d - 1 }$ Since this bound holds for any such S, we obtain:

$$
\operatorname* { s u p } _ { S \subseteq \operatorname* { s u p p } _ { \vec { P } ( S \cap E _ { \rho } ) > 0 } } W _ { 1 } ( \widetilde { P } _ { S } , \widetilde { P } ) \leq \frac { \alpha l } { 1 - \alpha } \sqrt { ( L - 2 l ) ^ { 2 } + d - 1 } + \left( 1 - \frac { l } { 1 - \alpha } \right) \sqrt { ( 1 - l ) ^ { 2 } + d - 1 } .
$$

Choosing $L > C _ { l , d }$ with

$$
\begin{array} { l l } { { C _ { l , d } = \displaystyle \frac { a _ { l } - 2 l c _ { l } ^ { 2 } + c _ { l } \sqrt { ( a _ { l } - 2 l ) ^ { 2 } + ( 1 - c _ { l } ^ { 2 } ) ( d - 1 ) } } { 1 - c _ { l } ^ { 2 } } , } } \\ { { a _ { l } = 1 + \displaystyle \frac { 1 - 3 l } { l ( 1 - 2 l ) } \sqrt { ( 1 - l ) ^ { 2 } + d - 1 } , ~ c _ { l } = \displaystyle \frac { 2 l } { 1 - 2 l } } } \end{array}
$$

ensures that the far exclusion condition (13) is satisfied.

We establish the error bounds for the population estimator $\widehat { P } _ { \mathrm { W F } }$ under the FELP contamination model as follows:

Theorem 1 (Population Risk of $\widehat { P } _ { \mathrm { W F } } )$ . For any clean distribution $P ^ { * } \in \mathcal G \subseteq \mathcal W _ { 1 } ( \rho , \alpha )$ and contaminated distribution $\widetilde { P } \in \mathcal { C } _ { P ^ { * } } ^ { \mathrm { F E L P } } ( \alpha , \rho )$ with $\begin{array} { r } { 0 \leq \alpha \leq \frac { 1 } { 2 } } \end{array}$ , the population estimator satisfies:

$$
W _ { 1 } ( \widehat { P } _ { \mathrm { W F } } , P ^ { * } ) \lesssim \rho .
$$

The core intuition behind Theorem 1 is as follows. The STV distance condition in the definition of $\mathcal { C } _ { P ^ { * } } ^ { \mathrm { F E L P } } ( \alpha , \rho )$ guarantees the existence of a shared support $\bar { S }$ of mass $1 - \alpha$ , which yields a reference trimmed distribution $\widetilde { P } _ { \bar { S } }$ . Note that $\widetilde { P } _ { \bar { S } }$ is a feasible (1−α)-truncated candidate of $\widetilde { P }$ that satisfies $\tilde { P } _ { \bar { S } } = P _ { \bar { S } } ^ { * }$ . By the $W _ { 1 } \cdot$ resilience of $P ^ { * }$ , it immediately follows that this reference distribution has a bounded risk: $W _ { 1 } ( \widetilde { P } _ { \bar { S } } , P ^ { * } ) \leq \rho$ Consequently, the optimal estimator $\widehat { P } _ { \mathrm { W F } }$ , which maximizes the Wasserstein distance to the contaminated distribution, is guaranteed to perform at least as well as this reference candidate. The FELP conditions then ensure that $\widehat { P } _ { \mathrm { W F } }$ indeed identifies a support that achieves this optimal risk bound of $\mathcal { O } ( \rho )$

We next establish a minimax lower bound showing that this $\mathcal { O } ( \rho )$ rate is optimal. Let $\mathcal { T } : \mathcal { Q } ( \mathcal { X } )  \mathcal { Q } ( \mathcal { X } )$ denote any population-level estimator, and define its maximum risk over the FELP contamination family as:

$$
\mathcal { R } _ { 1 , P ^ { * } , \mathcal { T } } ( \alpha , \rho ) = \operatorname* { s u p } _ { \widetilde { P } \in \mathcal { C } _ { P ^ { * } } ^ { \mathrm { F E L P } } ( \alpha , \rho ) } W _ { 1 } ( \mathcal { T } ( \widetilde { P } ) , P ^ { * } ) .
$$

We have the following minimax optimality result:

Theorem 2 (Minimax Optimal Population Risk). Let $\mathcal { G } _ { \mathrm { c o v } } ( \sigma ) \subseteq \mathcal { W } _ { 1 } ( \rho _ { \sigma } , \alpha )$ be the class of distributions with bounded covariance: $\begin{array} { r } { \mathcal { G } _ { \mathrm { c o v } } ( \sigma ) = \left\{ \mu \in \mathcal { Q } ( \mathcal { X } ) : \Sigma _ { \mu } \preceq \sigma ^ { 2 } \mathbf { I } \right\} . \ I f \ 0 < \alpha _ { 0 } \leq \alpha \leq \frac { 1 } { 2 } } \end{array}$ and the dimension satisfies $\begin{array} { r } { d \lesssim \frac { 1 } { \alpha ( 1 - \alpha ^ { 2 } ) } } \end{array}$ , then:

$$
\operatorname* { i n f } _ { \mathcal T } \operatorname* { s u p } _ { P ^ { * } \in \mathcal G _ { \mathrm { c o v } } ( \sigma ) } \mathcal R _ { 1 , P ^ { * } , \mathcal T } ( \alpha , \rho _ { \sigma } ) \asymp \rho _ { \sigma } ,
$$

where $\rho _ { \sigma } \asymp \sigma \sqrt { d \alpha }$ is the sharp resilience factor for the family $\mathcal { G } _ { \mathrm { c o v } } ( \sigma )$

In the case where the clean distribution class is restricted to bounded-covariance families, $\operatorname { i . e . , } \mathcal { G } =$ $\mathcal { G } _ { \mathrm { c o v } } ( \sigma )$ , Theorem 1 holds uniformly for any $P ^ { * } \in \mathcal G _ { \mathrm { c o v } } ( \sigma )$ and $\widetilde { P } \in \mathcal { C } _ { P ^ { * } } ^ { \mathrm { F E L P } } ( \alpha , \rho _ { \sigma } )$ . This immediately yields the upper bound:

$$
\operatorname* { s u p } _ { P ^ { * } \in \mathcal { G } _ { \mathrm { c o v } } ( \sigma ) } \mathcal { R } _ { 1 , P ^ { * } , \widehat { P } _ { \mathrm { W F } } } ( \alpha , \rho _ { \sigma } ) \lesssim \rho _ { \sigma } .
$$

In conjunction with the lower bound in Theorem 2, this result establishes that our population estimator $\widehat { P } _ { \mathrm { W F } }$ achieves the minimax optimal risk.

The choice of $\mathcal { G } _ { \mathrm { c o v } } ( \sigma )$ is representative of standard robust statistics settings and imposes minimal assumptions on the clean distribution family G containing $P ^ { * }$ . Indeed, our minimax optimality analysis can be readily extended to other standard distribution classes—such as sub-Gaussian families or families with bounded q-th moments $( q \ge 1 )$ —using analogous arguments based on their respective resilience properties (Nietert et al., 2023).

Furthermore, the conditions on the model parameters are highly practical:

• Contamination Level (α): The condition $\alpha \leq \textstyle { \frac { 1 } { 2 } }$ is mild, indicating that our framework can tolerate a corruption level of up to 50% of the data mass.

• Dimensionality (d): The dimension condition is quite natural, allowing the allowable dimension to grow as the contamination fraction α decreases.

## 3.1.2 Empirical Setting

In this section, we extend our analysis to the finite-sample regime by incorporating stochastic sampling errors. We first introduce the empirical counterpart of our proposed contamination model.

Definition 4 (Empirical FELP Contamination). Let $\mathbf { x } _ { 1 } , \hdots , \mathbf { x } _ { n } \overset { \mathrm { i . i . d . } } { \sim } P ^ { * } \in \mathcal { W } _ { 1 } ( \rho , \alpha )$ be clean samples with empirical measure $\begin{array} { r } { P _ { n } = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \delta _ { \mathbf { x } _ { i } } } \end{array}$ , and let $\widetilde { \mathbf { x } } _ { 1 } , \ldots , \widetilde { \mathbf { x } } _ { n }$ denote the observed contaminated samples. We say that the contaminated empirical measure $\begin{array} { r } { \widetilde { P } _ { n } = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \delta _ { \widetilde { \mathbf { x } } _ { i } } } \end{array}$ belongs to the empirical FELP family $\mathcal { C } _ { P _ { n } } ^ { \mathrm { F E L P } } ( \alpha , \rho )$ if the following three conditions hold:

1. Empirical Support TV Bound: The fraction of corrupted samples is bounded by α: $\begin{array} { r } { \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \mathbb { I } _ { \{ \mathbf { x } _ { i } \neq \widetilde { \mathbf { x } } _ { i } \} } \leq } \end{array}$ α. Let $\bar { S } \subseteq \{ \mathbf { x } _ { j } | j \in [ n ] , \mathbf { x } _ { j } = \widetilde { \mathbf { x } } _ { j } \}$ with $| \bar { S } | = ( 1 - \alpha ) n$ be the uncorrupted shared support.

2. Empirical Far Exclusion (FE): Let $E _ { n , \rho } : = \{ \mathbf { x } \in \mathrm { s u p p } ( \widetilde P _ { n } ) \backslash \bar { S } \mid d ( \mathbf { x } , \mathrm { s u p p } ( P _ { n } ) ) > \rho \}$ define the region of “empirical far outliers”. There exists a subset $S _ { + } \subseteq \operatorname { s u p p } ( \widetilde { P } _ { n } )$ with $\widetilde { P } _ { n } ( S _ { + } ) = 1 - \alpha$ and $S _ { + } \cap E _ { n , \rho } = \emptyset$

such that:

$$
\operatorname* { s u p } _ { S \subseteq \operatorname* { s u p p } ( \widetilde P _ { n } ) , \widetilde P _ { n } ( S ) = 1 - \alpha } W _ { 1 } ( \widetilde P _ { n , S } , \widetilde P _ { n } ) < W _ { 1 } ( \widetilde P _ { n , S _ { + } } , \widetilde P _ { n } ) ,\tag{15}
$$

where $\begin{array} { r } { \widetilde { P } _ { n , S } = \frac { 1 } { \widetilde { P } _ { n } ( S ) } \widetilde { P } _ { n } | _ { S } } \end{array}$ is the normalized restriction of ${ \widetilde { P } } _ { n }$ to S.

3. Empirical Local Projection (LP): Let $N _ { n , \rho } : = \{ \mathbf x \in \mathrm { s u p p } ( \widetilde P _ { n } ) \setminus \bar { S } ~ | ~ d ( \mathbf x , \mathrm { s u p p } ( P _ { n } ) ) \le \rho \}$ denote the region of “empirical near outliers”. There exists a projection map $T _ { n } : N _ { n , \rho } \to \operatorname { s u p p } ( P _ { n } )$ such that:

$$
d ( \mathbf { x } , T _ { n } ( \mathbf { x } ) ) \leq \rho \quad \mathrm { a n d } \quad T _ { n \sharp } \widetilde { P } _ { n } | _ { N _ { n , \rho } } \leq P _ { n } ,\tag{16}
$$

where $T _ { n \sharp }$ denotes the push-forward operator.

Notably, our empirical FELP contamination model is defined directly on the realized sample rather than treating contaminated points as i.i.d. samples from a pre-specified population-level mixture $\widetilde { P } \in \mathcal { C } _ { P ^ { * } } ^ { \mathrm { F E L P } } ( \alpha , \rho )$ The latter framework is commonly referred to as the oblivious contamination model (Diakonikolas et al., 2019; Zhu et al., 2022). Formulating contamination in terms of the realized sample is significantly more realistic because the underlying population distribution is typically unknown in practice.

Furthermore, empirical FELP aligns more closely with the classical TV contamination model discussed in Section 1, encompassing a broader class of adversarial corruptions than the oblivious model.

By construction, any empirical FELP contaminated measure satisfies $\| \widetilde { P } _ { n } - P _ { n } \| _ { \mathrm { S T V } } \leq \alpha$ . Let $\mathbb { P } _ { n }$ denote the joint distribution of the i.i.d. clean samples $( \mathbf { x } _ { 1 } , \ldots , \mathbf { x } _ { n } ) \in { \mathcal { X } } ^ { n }$ , and let ${ \mathcal { D } } _ { n } ^ { \mathrm { F E L P } } ( P ^ { * } , \alpha )$ represent the family of all joint distributions $\widetilde { \mathbb { P } } _ { n } \in \mathcal { Q } ( \mathcal { X } ^ { n } )$ of the contaminated samples $( \widetilde { \mathbf { x } } _ { 1 } , \ldots , \widetilde { \mathbf { x } } _ { n } )$ generated via empirical FELP contamination.

We establish the statistical convergence of our empirical estimator $\widehat { P } _ { \mathrm { W F } }$ (introduced in Section 2.1) under this finite-sample model below:

Theorem 3 (Empirical Risk of $\widehat { P } _ { \mathrm { W F } } )$ . Let $P ^ { * } \in \mathcal { G } \subseteq \mathcal { W } _ { 1 } ( \rho , \alpha )$ and let $0 \leq \alpha \leq \frac { 1 } { 2 }$ . For any empirical contamination $\widetilde { P } _ { n } \in \mathcal { C } _ { P _ { n } } ^ { \mathrm { F E L P } } ( \alpha , \rho ) ~ ( i . e . , ~ \widetilde { \mathbb { P } } _ { n } \in \mathcal { D } _ { n } ^ { \mathrm { F E L P } } ( P ^ { * } , \alpha ) )$ , the empirical estimator satisfies:

$$
\mathbb { E } _ { \widetilde { \mathbb { P } } _ { n } } \left[ W _ { 1 } ( \widehat { P } _ { \mathrm { W F } } , P ^ { * } ) \right] \lesssim \rho + \mathbb { E } _ { \mathbb { P } _ { n } } \left[ W _ { 1 } ( P _ { n } , P ^ { * } ) \right] .
$$

Similar to Theorem 1, the sample corruption constraint $\begin{array} { r } { \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \mathbb { I } _ { \left\{ \widetilde { \mathbf { x } } _ { i } \neq \mathbf { x } _ { i } \right\} } \leq c } \end{array}$ α implies that $\| \widetilde { P } _ { n } - P _ { n } \| _ { \mathrm { S T V } } \leq$ α. This guarantees the existence of a reference bridge estimator $\widetilde { P } _ { n , { \bar { S } } }$ , where $\bar { S } \subseteq \{ \mathbf { x } _ { j } ~ | ~ j \in [ n ] , \mathbf { x } _ { j } = \widetilde { \mathbf { x } } _ { j } \}$ with $| \bar { S } | = ( 1 - \alpha ) n$ . The expected $W _ { \mathrm { 1 ^ { - r i s k } } }$ of this reference estimator is bounded by $\rho + \mathbb { E } _ { \mathbb { P } _ { n } } [ W _ { 1 } ( P _ { n } , P ^ { * } ) ]$ . Since $\tilde { P } _ { n , \bar { S } }$ discards all contaminated points and retains only clean samples, it represents the statistical benchmark (i.e., the best possible performance) for our filtering framework. Theorem 3 establishes that under empirical FELP contamination, the Wasserstein Filtering (WF) estimator indeed matches this optimal benchmark.

To rigorously prove that the WF estimator is minimax optimal in the empirical setting, we introduce a mild condition ensuring that the Far Exclusion property holds with high probability under finite-sample realizations:

Condition 1 (Far Exclusion with High Probability). For every clean distribution $P ^ { * } \in \mathcal G$ and every population contamination $\begin{array} { r } { \widetilde { P } \in \mathcal { C } _ { P ^ { * } } ^ { \mathrm { F E L P } } \left( \frac { \alpha } { 5 } , \rho \right) } \end{array}$ , the following holds. Let $\eta : = P ^ { \ast } | _ { \bar { S } } = \widetilde { P } | _ { \bar { S } }$ represent the shared clean component, and decompose the outliers as $\nu _ { \mathrm { l o c } } : = \widetilde { P } | _ { N _ { \rho } }$ and $\nu _ { \mathrm { f a r } } : = \widetilde { P } | _ { E _ { \rho } }$ . Further, let $\lambda _ { \mathrm { l o c } } : = T _ { \sharp } \nu _ { \mathrm { l o c } }$ denote the projection of local outliers, and define the remaining clean mass as $\lambda _ { \mathrm { f a r } } : = P ^ { * } | _ { \bar { S } ^ { c } } - \lambda _ { \mathrm { l o c } } \geq 0$

For any coupling $\pi _ { \mathrm { f a r } } \in \Pi ( \lambda _ { \mathrm { f a r } } , \nu _ { \mathrm { f a r } } )$ , we construct a joint coupling $\pi \in \Pi ( P ^ { * } , \tilde { P } )$ defined as:

$$
\pi = ( \mathrm { I d } , \mathrm { I d } ) _ { \sharp } \eta + ( T , \mathrm { I d } ) _ { \sharp } \nu _ { \mathrm { l o c } } + \pi _ { \mathrm { f a r } } ,
$$

where Id is the identity map. Under the product measure $( \mathbf { x } _ { i } , \widetilde { \mathbf { x } } _ { i } ) _ { i = 1 } ^ { n } \sim \pi ^ { n }$ , we define the successful exclusion event E as:

$$
\mathcal { E } = \left\{ \begin{array} { l l } { \exists S _ { + } \subseteq \{ \widetilde { \mathbf { x } } _ { 1 } , \dots , \widetilde { \mathbf { x } } _ { n } \} , \ | S _ { + } | = ( 1 - \alpha ) n , \ S _ { + } \cap E _ { n , \rho } = \emptyset \quad s u c h \ t h a t } \\ { \qquad \quad } \\ { \underset { | S | = ( \widetilde { \mathbf { x } } _ { 1 } , \dots , \widetilde { \mathbf { x } } _ { n } ) } { \operatorname* { s u p } } W _ { 1 } \left( \frac { 1 } { | S | } \displaystyle \sum _ { \widetilde { \mathbf { x } } _ { i } \in S } \delta _ { \widetilde { \mathbf { x } } _ { i } } , \widetilde { P } _ { n } \right) < W _ { 1 } \left( \frac { 1 } { | S _ { + } | } \displaystyle \sum _ { \widetilde { \mathbf { x } } _ { i } \in S _ { + } } \delta _ { \widetilde { \mathbf { x } } _ { i } } , \widetilde { P } _ { n } \right) \Bigg \} . } \end{array} \right.
$$

We assume there exists a sequence $\delta _ { n } \leq \frac { 1 } { 2 0 }$ such that $\pi ^ { n } ( { \mathcal { E } } ^ { c } ) \leq \delta _ { n }$

Condition 1 defines a canonical coupling π between the population distribution pair $( P ^ { * } , \tilde { P } )$ under the population-level FELP contamination model. It asserts that under this coupling, the generated contaminated samples $\{ \widetilde { \mathbf { x } } _ { i } \} _ { i = 1 } ^ { n }$ satisfy the empirical far exclusion event E—which is the central requirement of the empirical FELP model—with high probability.

By viewing the observed samples $\{ \widetilde { \mathbf { x } } _ { i } \} _ { i = 1 } ^ { n }$ as being generated from the clean samples $\{ { \bf x } _ { i } \} _ { i = 1 } ^ { n }$ via this coupled process, Condition 1 provides a rigorous bridge from the population setting to the empirical setting. Specifically, for any valid population pair $( P ^ { * } , \tilde { P } )$ , we can construct a sample sequence $( \widetilde { \mathbf { x } } _ { 1 } , \ldots , \widetilde { \mathbf { x } } _ { n } )$ such that their joint distribution $\widetilde { \mathbb { P } } _ { n }$ belongs to the empirical FELP family $\mathcal { D } _ { n } ^ { \mathrm { F E L P } } ( P ^ { * } , \alpha )$ . This construction implies that the empirical FELP contamination family is richer than its population counterpart. Consequently, the minimax risk under empirical FELP contamination is naturally lower bounded by the corresponding population-level minimax risk.

In what follows, we demonstrate that Condition 1 is satisfied in the same uniform cube setting introduced in our population-level example.

Example (Uniform Cube in the Empirical Setting). Consider the d-dimensional uniform cube $S ^ { * } =$ $[ 0 , 1 ] ^ { d }$ with the clean distribution $P ^ { * } = \operatorname { U n i f } ( S ^ { * } )$ . Let $S _ { \mathrm { l o c } } ^ { \mathrm { i n } } , S _ { \mathrm { f a r } } ^ { \mathrm { i n } } , S _ { \mathrm { l o c } } ^ { \mathrm { o u t } }$ , and ${ \bf x } _ { \mathrm { f a r } } ^ { \mathrm { o u t } }$ be defined exactly as in the population setting (with $\begin{array} { r } { l = \frac { \alpha } { 1 0 } ) } \end{array}$ , yielding the population contaminated distribution:

$$
\widetilde { P } = P ^ { * } | _ { S ^ { * } \setminus ( S _ { \mathrm { l o c } } ^ { \mathrm { i n } } \cup S _ { \mathrm { f a r } } ^ { \mathrm { i n } } ) } + l \mathrm { U n i f } ( S _ { \mathrm { l o c } } ^ { \mathrm { o u t } } ) + l \delta _ { \mathbf { x } _ { \mathrm { f a r } } ^ { \mathrm { o u t } } } .
$$

From the population analysis, we have $\begin{array} { r } { \widetilde { P } \in \mathcal { C } _ { P ^ { * } } ^ { \mathrm { F E L P } } \left( \frac { \alpha } { 5 } , \sqrt { d } \right) } \end{array}$ with the projection map $T ( \mathbf { x } ) = \mathbf { x } - \mathbf { e } _ { 1 }$ <sub>1</sub> and the shared support $\bar { S } = S ^ { * } \setminus ( S _ { \mathrm { l o c } } ^ { \mathrm { i n } } \cup S _ { \mathrm { f a r } } ^ { \mathrm { i n } } )$

Let $\mathbf { x } _ { 1 } , . . . , \mathbf { x } _ { n } \stackrel { \mathrm { i . i . d . } } { \sim } P ^ { * }$ . We construct the empirical contaminated samples $\widetilde { \mathbf { x } } _ { i }$ for $i \in [ n ]$ as:

$$
\begin{array} { r } { \widetilde { \mathbf { x } } _ { i } = \left\{ \begin{array} { l l } { \mathbf { x } _ { i } , } & { \mathbf { x } _ { i } \notin S _ { \mathrm { l o c } } ^ { \mathrm { i n } } \cup S _ { \mathrm { f a r } } ^ { \mathrm { i n } } , } \\ { \mathbf { x } _ { i } + \mathbf { e } _ { 1 } , } & { \mathbf { x } _ { i } \in S _ { \mathrm { l o c } } ^ { \mathrm { i n } } , } \\ { \mathbf { x } _ { \mathrm { f a r } } ^ { \mathrm { o u t } } , } & { \mathbf { x } _ { i } \in S _ { \mathrm { f a r } } ^ { \mathrm { i n } } . } \end{array} \right. } \end{array}
$$

Then, the coupled samples satisfy $( { \bf x } _ { i } , \widetilde { { \bf x } } _ { i } ) _ { i = 1 } ^ { n } \sim \pi ^ { n }$ with $\pi = ( \mathrm { I d } , \mathrm { I d } ) _ { \sharp } \eta + ( T , \mathrm { I d } ) _ { \sharp } \nu _ { \mathrm { l o c } } + \pi _ { \mathrm { f a r } }$ , and we have $\pi ^ { n } ( { \mathcal { E } } ^ { c } ) \leq { \frac { 1 } { 2 0 } }$ for suficiently large n.

Proof sketch. Let $M _ { \mathrm { l o c } } = \{ \widetilde { \mathbf { x } } _ { i } \in S _ { \mathrm { l o c } } ^ { \mathrm { o u t } } \ | \ i \in [ n ] \}$ and $M _ { \mathrm { f a r } } = \{ \widetilde { \mathbf { x } } _ { i } = \mathbf { x } _ { \mathrm { f a r } } ^ { \mathrm { o u t } } \ | \ i \in [ n ] \}$ }. Applying Hoefding’s inequality, we define the high-probability concentration event $\mathcal { A } _ { \epsilon }$ as:

$$
\mathbb { P } \{ \mathcal { A } _ { \epsilon } \} : = \mathbb { P } \left\{ \left| \frac { | M _ { \mathrm { l o c } } | } { n } - l \right| \le \epsilon , ~ \left| \frac { | M _ { \mathrm { f a r } } | } { n } - l \right| \le \epsilon \right\} \ge 1 - 4 e ^ { - 2 \epsilon ^ { 2 } n } .
$$

For any $ { \widetilde { \mathbf { x } } } _ { i } \in M _ { \mathrm { l o c } }$ , we have $d ( \widetilde \mathbf { x } _ { i } , \mathrm { s u p p } ( P _ { n } ) ) \leq d ( \widetilde \mathbf { x } _ { i } , T ( \widetilde \mathbf { x } _ { i } ) ) = 1 \leq \sqrt { d } = \rho ,$ which confirms $M _ { \mathrm { l o c } } \subseteq N _ { n , \rho } .$ Under the assumption $L > \sqrt { d } + 1$ , any $\widetilde { \mathbf { x } } _ { i } \in M _ { \mathrm { f a r } }$ satisfies $d ( \widetilde { \mathbf { x } } _ { i } , \mathrm { s u p p } ( P _ { n } ) ) \geq d ( \widetilde { \mathbf { x } } _ { i } , \mathrm { s u p p } ( P ^ { * } ) ) = L - 1 >$ ${ \sqrt { d } } = \rho ,$ , confirming $M _ { \mathrm { f a r } } \subseteq E _ { n , \rho } ,$ . Since $M _ { \mathrm { l o c } } \cup M _ { \mathrm { f a r } } = N _ { n , \rho } \cup E _ { n , \rho }$ and $N _ { n , \rho } \cap E _ { n , \rho } = \emptyset$ , we obtain the partition $M _ { \mathrm { l o c } } = N _ { n , \rho }$ and $M _ { \mathrm { f a r } } = E _ { n , \rho }$

Next, we construct a valid uncorrupted empirical support $S _ { + }$ of size $( 1 - \alpha ) n$ by removing $M _ { \mathrm { f a r } } , M _ { \mathrm { l o c } } ,$ and any remaining αn $- \left| M _ { \mathrm { f a r } } \right| - \left| M _ { \mathrm { l o c } } \right|$ samples from the set of clean inliers $\{ \widetilde { \mathbf { x } } _ { i } \mid \widetilde { \mathbf { x } } _ { i } = \mathbf { x } _ { i } \}$ . This construction is feasible on the event $\boldsymbol { A } _ { \epsilon }$ because: $\begin{array} { r } { | M _ { \mathrm { f a r } } | + | M _ { \mathrm { l o c } } | \le 2 ( l + \epsilon ) n = \left( \frac { \alpha } { 5 } + 2 \epsilon \right) n < } \end{array}$ αn, provided that $\epsilon < \frac { 2 } { 5 } \alpha .$ Consequently, $S _ { + } \cap E _ { n , \rho } = \emptyset$ . Since ${ \widetilde { P } } _ { n }$ places a mass of $\frac { \left| M _ { \mathrm { f a r } } \right| } { n }$ on the atom ${ \bf x } _ { \mathrm { f a r } } ^ { \mathrm { o u t } }$ , we establish the lower bound:

$$
W _ { 1 } ( \widetilde { P } _ { n S _ { + } } , \widetilde { P } _ { n } ) > \frac { | M _ { \mathrm { f a r } } | } { n } ( L - 1 ) .
$$

Conversely, let $S$ be any corrupted subset of size $( 1 - \alpha ) n$ that contains at least one far outlier. Employing the same transportation plan coupling logic as in the population case, we obtain:

$$
\operatorname* { s u p } _ { \widetilde { P } _ { n } ( S ) = 1 - \alpha } W _ { 1 } ( \widetilde { P } _ { n S } , \widetilde { P } _ { n } ) \leq \operatorname* { s u p } _ { 1 \leq | S \cap M _ { \mathrm { f a r } } | \leq | M _ { \mathrm { f a r } } | } \left| \frac { \left| M _ { \mathrm { f a r } } \right| } { n } - \frac { | S \cap M _ { \mathrm { f a r } } | } { n ( 1 - \alpha ) } \right| \sqrt { ( L - 2 l ) ^ { 2 } + d - 1 } + \sqrt { ( 1 - l ) ^ { 2 } + d - 1 } .
$$

Therefore, the far exclusion event $\mathcal { E }$ is guaranteed to hold as long as:

$$
\frac { \left| M _ { \mathrm { f a r } } \right| } { n } ( L - 1 ) > \operatorname* { s u p } _ { 1 \leq | S \cap M _ { \mathrm { f a r } } | \leq | M _ { \mathrm { f a r } } | } \left| \frac { | M _ { \mathrm { f a r } } | } { n } - \frac { | S \cap M _ { \mathrm { f a r } } | } { n ( 1 - \alpha ) } \right| \sqrt { ( L - 2 l ) ^ { 2 } + d - 1 } + \sqrt { ( 1 - l ) ^ { 2 } + d - 1 } .
$$

On the concentration event $\mathcal { A } _ { \epsilon }$ with $\epsilon < l .$ this inequality is satisfied by choosing:

$$
\begin{array} { r l } & { L > \operatorname* { m a x } \left\{ \sqrt { d } + 1 , C _ { l , d } , 2 l , \frac { l + \epsilon + ( l + \epsilon - \Delta _ { \epsilon , n , \alpha } ) \sqrt { d - 1 } + \sqrt { ( 1 - l ) ^ { 2 } + d - 1 } } { \Delta _ { \epsilon , n , \alpha } } \right\} , } \\ & { \quad \quad \quad \quad \quad \mathrm { w h e r e } \quad \Delta _ { \epsilon , n , \alpha } = \operatorname* { m i n } \left\{ \frac { 1 } { ( 1 - \alpha ) n } , \frac { ( l - \epsilon ) ( 1 - 2 \alpha ) } { 1 - \alpha } \right\} . } \end{array}
$$

Asymptotically, this requires $L = \Omega ( n { \sqrt { d } } )$ . This ensures $\mathcal { A } _ { \epsilon } \subseteq \mathcal { E }$ , meaning $\begin{array} { r } { \pi ^ { n } ( \mathcal { E } ^ { c } ) \leq \pi ^ { n } ( \mathcal { A } _ { \epsilon } ^ { c } ) \leq 4 e ^ { - 2 \epsilon ^ { 2 } n } \leq \frac { 1 } { 2 0 } } \end{array}$ for a suficiently large sample size n. 20

We consider the finite-sample empirical risk:

$$
\mathcal { R } _ { 1 , P ^ { * } , \mathcal { T } _ { n } , n } ( \alpha , \rho ) = \operatorname* { s u p } _ { \widetilde { \mathbb { P } } _ { n } \in \mathcal { D } _ { n } ^ { \mathrm { F E L P } } ( P ^ { * } , \alpha ) } \mathbb { E } _ { \widetilde { \mathbb { P } } _ { n } } \Big [ W _ { 1 } ( \mathcal { T } _ { n } ( \widetilde { P } _ { n } ) , P ^ { * } ) \Big ] ,
$$

where $\mathcal { T } _ { n }$ is an empirical distribution estimator, with the subscript n distinguishing it from the populationlevel estimator $\tau$ . The corresponding empirical minimax risk is defined as:

$$
\mathcal { R } _ { 1 , n } ( \alpha , \rho ) = \operatorname* { i n f } _ { \mathcal { T } _ { n } } \operatorname* { s u p } _ { P ^ { * } \in \mathcal { G } } \mathcal { R } _ { 1 , P ^ { * } , \mathcal { T } _ { n } , n } ( \alpha , \rho ) .
$$

We establish the minimax optimality of the empirical risk under the FELP model as follows:

Theorem 4 (Minimax Optimality of Empirical Risk). Let $P ^ { * } \in \mathcal { G } \subseteq \mathcal { W } _ { 1 } ( \rho , \alpha )$ with $0 \leq \alpha \leq \frac { 1 } { 2 }$ . Under Condition 1, the empirical minimax risk satisfies:

$$
\mathscr { R } _ { 1 , \infty } \left( \frac { \alpha } { 5 } , \rho \right) + \operatorname* { i n f } _ { \mathcal { T } _ { n } } \operatorname* { s u p } _ { P ^ { * } \in \mathcal { G } } \mathbb { E } \left[ W _ { 1 } ( \mathcal { T } _ { n } ( P _ { n } ) , P ^ { * } ) \right] \lesssim \operatorname* { i n f } _ { \mathcal { T } _ { n } } \operatorname* { s u p } _ { P ^ { * } \in \mathcal { G } } \mathscr { R } _ { 1 , P ^ { * } , \mathcal { T } _ { n } , n } ( \alpha , \rho ) \lesssim \rho + \operatorname* { s u p } _ { P ^ { * } \in \mathcal { G } } \mathbb { E } \left[ W _ { 1 } ( P _ { n } , P ^ { * } ) \right] ,
$$

where the population-level minimax limit is given by:

$$
\mathcal { R } _ { 1 , \infty } \left( \frac { \alpha } { 5 } , \rho \right) = \operatorname* { i n f } _ { \mathcal { T } } \operatorname* { s u p } _ { P ^ { * } \in \mathcal { G } } \operatorname* { s u p } _ { \widetilde { P } \in { \cal C } _ { P ^ { * } } ^ { \mathrm { F E L P } } \left( \frac { \alpha } { 5 } , \rho \right) } W _ { 1 } ( \mathcal { T } ( \widetilde { P } ) , P ^ { * } ) .
$$

In particular, when $\mathcal { G }$ is the family of distributions with bounded covariance, ${ \mathcal G } _ { \mathrm { c o v } } ( \sigma ) = \{ \mu \in { \mathcal Q } ( \mathcal X ) : \Sigma _ { \mu } \preceq$ $\sigma ^ { 2 } \mathbf { I } \big \}$ , and assuming $\begin{array} { r } { 0 < \alpha _ { 0 } \le \alpha \le \frac { 1 } { 2 } } \end{array}$ alongside the dimension constraint $\begin{array} { r } { d \lesssim \frac { 1 } { \alpha ( 1 - \alpha ^ { 2 } ) } } \end{array}$ , we have:

$$
\operatorname* { i n f } _ { \mathcal T _ { n } } \operatorname* { s u p } _ { P ^ { * } \in \mathcal G _ { \mathrm { c o v } } ( \sigma ) } \mathcal R _ { 1 , P ^ { * } , \mathcal T _ { n } , n } ( \alpha , \rho _ { \sigma } ) \asymp \sigma \sqrt { d \alpha } + \sigma \sqrt { d } n ^ { - \frac { 1 } { d } } .
$$

Furthermore, the Wasserstein Filtering estimator $\widehat { P } _ { \mathrm { W F } }$ achieves this optimal empirical minimax rate.

The first part of Theorem 4 establishes that the empirical minimax risk is lower bounded by the sum of the population-level risk ${ \mathcal { R } } _ { 1 , \infty } \left( { \frac { \alpha } { 5 } } , \rho \right)$ (representing the infinite-sample limit) and the minimax rate of clean distribution estimation under the $W _ { 1 }$ metric. Conversely, the upper bound is characterized by the population-level resilience $\rho$ and the expected estimation error of the empirical measure $P _ { n }$ over the class $\mathcal { G } .$

When we assume bounded second moments $( \mathrm { i . e . , ~ } P ^ { \ast } \in \mathcal G _ { \mathrm { c o v } } ( \sigma ) )$ , we have $\textstyle \mathcal { R } _ { 1 , \infty } \left( { \frac { \alpha } { 5 } } , \rho \right) \asymp \rho \asymp \sigma { \sqrt { d \alpha } }$ by Theorem 2, while the empirical measure $P _ { n }$ achieves the optimal $W _ { 1 }$ -error rate of $\mathcal { O } ( \sigma \sqrt { d } n ^ { - 1 / d } )$ (Lei, 2020; Niles-Weed and Berthet, 2022). This matches the lower and upper bounds up to constant factors therefore shows that our Wasserstein Filtering estimator $\widehat { P } _ { \mathrm { W F } }$ is minimax optimal.

The focus on the bounded-covariance class $\mathcal { G } _ { \mathrm { c o v } } ( \sigma )$ is particularly significant because it allows the clean distribution to be heavy-tailed. Under heavy-tailed distributions, genuine data points can appear far from the mean, which naturally obscures the distinction between clean data and adversarial outliers. This makes outlier detection and robust estimation exceptionally challenging, highlighting the strength of our framework’s guarantees.

## 3.2 Convergence of the Algorithm

In this section, we analyze the algorithmic convergence of the sample weight optimization scheme employed in the SinkWF and SlicedWF algorithms. For simplicity of exposition, we let $\sigma ( \mathbf { b } ) = \mathrm { s o f t m a x } ( \mathbf { b } )$ denote the softmax mapping and define the clipping threshold as $\begin{array} { r } { u _ { \alpha } = \frac { 1 } { ( 1 - \alpha ) n } } \end{array}$ . We consider the idealized objective function with the exact $W _ { 1 }$ distance:

$$
F ( \mathbf { b } ) = W _ { 1 } ( P _ { n } , P _ { \sigma ( \mathbf { b } ) } ) - \beta \sum _ { i = 1 } ^ { n } ( \sigma _ { i } ( \mathbf { b } ) - u _ { \alpha } ) _ { + } ,
$$

where $( t ) _ { + } = \operatorname* { m a x } \{ t , 0 \}$ . We further simplify the optimization dynamics from Adam to standard gradient ascent updates with step size $\gamma > 0 \colon \mathbf { b } _ { r + 1 } = \mathbf { b } _ { r } + \gamma \nabla F ( \mathbf { b } _ { r } )$ . We assume there exists an iteration index $r _ { 0 }$ and a compact set $\mathbb { K } \subset \mathbb { R } ^ { n }$ such that the iterates satisfy $\mathbf { b } _ { r } \in \mathbb { K }$ for all $r \geq r _ { 0 }$ . Without loss of generality, we treat $r _ { 0 }$ as the initial step and analyze the subsequent R iterations. To establish convergence, we introduce the following regularity conditions on the optimization landscape:

Condition 2 (Unique Dual Potential). For every parameter vector b $\in \mathbb { K }$ and its induced probability vector $\mathbf { p } = \sigma ( \mathbf { b } )$ , the exact Optimal Transport $( O T )$ dual problem,

$$
W _ { 1 } ( P _ { n } , P _ { \mathbf { p } } ) = \operatorname* { m a x } _ { { \mathbf { u } } , { \mathbf { v } } \in { \mathbb { R } } ^ { n } \atop { \substack { \mathbf { u } _ { i } + \boldsymbol { v } _ { j } \leq C _ { i j } } } } \left\{ { \frac { 1 } { n } } \sum _ { i = 1 } ^ { n } u _ { i } + \sum _ { j = 1 } ^ { n } p _ { j } v _ { j } \right\} ,
$$

admits a unique optimal target potential $\mathbf { v } ^ { * } ( \mathbf { p } )$ satisfying the normalization constraint ${ \textstyle \sum _ { j = 1 } ^ { n } v _ { j } ^ { * } ( \mathbf { p } ) = 0 }$

Condition 3 (Stable Active Set and Non-boundary Hitting). Let m ${ \bf ( p ) } \in \{ 0 , 1 \} ^ { n }$ define the active-set indicator vector of the penalty term, where $m _ { i } ( \mathbf { p } ) = \mathbb { I } _ { \{ p _ { i } > u _ { \alpha } \} }$ . There exists a constant vector m<sub>0</sub> $\in \{ 0 , 1 \} ^ { n }$ such that ${ \bf m } ( \sigma ( { \bf b } ) ) = { \bf m } _ { 0 }$ for all $\mathbf { b } \in \mathbb { K }$ . Furthermore, the softmax weights do not hit the boundary of the threshold, $i . e . , \sigma _ { i } ( \mathbf { b } ) \neq u _ { \alpha }$ for all $i \in [ n ]$ and b $\in \mathbb { K }$

Condition 4 (Smoothness of the Optimal Dual). The optimal target dual potential $\mathbf { v } ^ { * } ( \cdot )$ is Lipschitz continuous on the simplex image $\sigma ( \mathbb { K } )$ . That is, for any b, $\mathbf { c } \in \mathbb { K }$ with $\mathbf { p } = \sigma ( \mathbf { b } )$ and $\mathbf { q } = \sigma ( \mathbf { c } )$ , there exists a constant $L _ { \mathbf { v } ^ { * } } > 0$ such that: $\| \mathbf { v } ^ { * } ( \mathbf { p } ) - \mathbf { v } ^ { * } ( \mathbf { q } ) \| _ { 2 } \leq L _ { \mathbf { v } ^ { * } } \| \mathbf { p } - \mathbf { q } \| _ { 2 }$

Theorem 5 (Convergence Rate of the WF Algorithm). Under Conditions ${ 2 - 4 }$ , the gradient $\nabla F$ is $L _ { F } .$ Lipschitz continuous on K for some constant $L _ { F } > 0$ . If the step size satisfies $\begin{array} { r } { \gamma < \frac { 2 } { L _ { F } } } \end{array}$ , the iterates of the gradient ascent scheme satisfy:

$$
\operatorname* { m i n } _ { r \in \{ r _ { 0 } , \ldots , r _ { 0 } + R - 1 \} } \| \nabla F ( \mathbf { b } _ { r } ) \| _ { 2 } \leq \sqrt { \frac { \operatorname* { m a x } _ { i , j \in [ n ] } \| \mathbf { x } _ { i } - \mathbf { x } _ { j } \| _ { 2 } - F ( \mathbf { b } _ { r _ { 0 } } ) } { \left( \gamma - \frac { L _ { F } \gamma ^ { 2 } } { 2 } \right) R } } .
$$

Condition 2 and the non-boundary hitting portion of Condition 3 ensure that the objective function $F ( \mathbf { b } )$ is continuously diferentiable on K. This guarantees the existence of a unique, explicit gradient, thereby avoiding the analytical complexities associated with subgradient calculus. Additionally, Condition 4 together with the stable active set property in Condition 3 ensures the $L _ { F } \mathrm { - L i p s c h i t z }$ smoothness of $\nabla F$

Theorem 5 demonstrates that our sample-weight optimization framework achieves a sublinear convergence rate of $\mathcal { O } ( 1 / \sqrt { R } )$ , matching the standard optimal rate for first-order methods in non-concave smooth optimization.

## 4 Numerical Studies

In this section, we evaluate the performance of our proposed methods against several state-of-the-art baselines:

• DIF (Deep Isolation Forest) (Xu et al., 2023).

• MDE (Minimum Distance Estimation) (Nietert et al., 2024). Note that the MDE algorithm evaluated here is a specialized version of the framework in Nietert et al. (2024) designed for TV corruption, which reduces to a standard spectral iterative filtering procedure (see Section 2.3 of Nietert et al. (2024)).

• COPOD (Copula-based Outlier Detection) (Li et al., 2020).

To assess performance on these binary classification (outlier detection) tasks, we employ two widely used metrics: the Area Under the Receiver Operating Characteristic curve (AUC-ROC) and the Area Under the Precision-Recall curve (AUC-PR).

## 4.1 Two-Dimensional Synthetic Data

We first compare our methods with the competing baselines on two-dimensional synthetic datasets to facilitate clear visualization. Specifically, we generate two classic datasets—the two-moons dataset and the concentric circles dataset, using the make moons and make circles functions from the scikit-learn Python library with a noise level of 0.1. For each dataset, we generate $n = 5 0 0$ samples, where a fraction α of the samples is corrupted. The corrupted samples are generated from two diferent Gaussian distributions representing diferent contamination strengths:

$$
\mathbf { W e a k } \mathbf { C a s e } : \mathcal { N } ( [ - 0 . 5 ] , [ 0 . 0 1  \phantom { 0 . 0 1 } 0 ] ) , \quad \mathrm { a n d } \quad \mathbf { S t r o n g } \mathbf { C a s e } : \mathcal { N } ( [ - 2 . 5 ] , [ 0 . 0 1 \phantom { 0 . 0 1 } 0 ] ) .
$$

The “strong” contamination scenario features outliers that are well-separated from the clean data cluster, whereas the ”weak” scenario places outliers much closer to the clean data, as illustrated in Figure A.3 in the Supplementary Material.

We consider two contamination proportions: $\alpha = 0 . 0 5 ~ \mathrm { a n d } ~ \alpha = 0 . 1 5$ . For the algorithms requiring the outlier fraction as an input (the robust WF variants and MDE), we provide the true value of α. For SinkWF and SlicedWF, the optimization parameters are initialized as ${ \bf b } _ { 0 } = { \bf 1 } + 1 0 ^ { - 3 } { \bf z }$ , where $\mathbf { z } \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } )$ , and we set the number of repetitions to T = 10. Other hyperparameters are selected via grid search, with the search spaces detailed in Appendix C.1.

We repeat each experiment 50 times, executing an independent grid search for each run. The average AUC-ROC and AUC-PR, along with their corresponding standard errors, are reported in Table 1. Our implementations of the three WF algorithms leverage the Python Optimal Transport (POT) toolbox (Flamary et al., 2021) for SlicedWF and the GeomLoss library (Feydy et al., 2019) for SinkWF and SinkMarg.

Table 1: Results of 2D data. Weak and strong contamination implies the signal of corruption strength. The mean of 50 times repetition is recorded and the corresponding standard error is recorded in parenthesis right side.
<table><tr><td rowspan="3">Dataset</td><td colspan="5">Weak Contamination</td></tr><tr><td rowspan="2"></td><td colspan="2">AUC-ROC</td><td colspan="2">AUC-PR</td></tr><tr><td>α = 0.05</td><td>α = 0.15</td><td>α = 0.05</td><td>α = 0.15</td></tr><tr><td rowspan="13">Two moons</td><td>COPOD</td><td>0.89(0.02)</td><td>0.80(0.01)</td><td>0.25(0.05)</td><td>0.34(0.02)</td></tr><tr><td>DIF</td><td>0.96(0.02)</td><td>0.32(0.06)</td><td>0.49(0.12)</td><td>0.13(0.02)</td></tr><tr><td>MDE</td><td>0.67(0.05)</td><td>0.65(0.03)</td><td>0.08(0.01)</td><td>0.20(0.02)</td></tr><tr><td>SlicedWF</td><td>0.77(0.12)</td><td>0.85(0.15)</td><td>0.23(0.29)</td><td>0.58(0.32)</td></tr><tr><td>SinkWF</td><td>0.85(0.10)</td><td>1.00(0.00)</td><td>0.31(0.28)</td><td>1.00(0.00)</td></tr><tr><td>SinkMarg</td><td>0.63(0.06)</td><td>0.85(0.03)</td><td>0.11(0.05)</td><td>0.77(0.05)</td></tr><tr><td></td><td></td><td>Strong Contamination</td><td></td><td></td></tr><tr><td></td><td colspan="4">AUC-ROC</td></tr><tr><td></td><td rowspan="5"></td><td colspan="2">α = 0.05 α = 0.15</td><td colspan="2">AUC-PR α = 0.05</td></tr><tr><td>COPOD</td><td>1.00(0.00)</td><td>0.97(0.00)</td><td>0.97(0.01)</td><td>α = 0.15 0.83(0.01)</td></tr><tr><td>DIF</td><td>0.99(0.00)</td><td>0.42(0.10)</td><td>0.85(0.07)</td><td>0.21(0.06)</td></tr><tr><td>MDE</td><td>0.98(0.01)</td><td>1.00(0.00)</td><td>0.66(0.12)</td><td>1.00(0.00)</td></tr><tr><td>SlicedWF</td><td>1.00(0.01)</td><td>0.99(0.02)</td><td>0.99(0.09)</td><td>0.96(0.11)</td></tr><tr><td></td><td>SinkWF</td><td>1.00(0.00)</td><td>1.00(0.00)</td><td>1.00(0.00)</td><td>1.00(0.00)</td></tr><tr><td></td><td>SinkMarg</td><td>0.90(0.02)</td><td>0.91(0.02)</td><td>0.30(0.15)</td><td>0.88(0.03)</td></tr><tr><td></td><td colspan="5">Weak Contamination</td></tr><tr><td rowspan="10">Circles</td><td rowspan="9">COPOD</td><td colspan="3">AUC-ROC</td><td rowspan="9">AUC-PR</td></tr><tr><td>α = 0.05 0.58(0.02)</td><td>α = 0.15</td><td>α = 0.05</td><td>α = 0.15</td></tr><tr><td></td><td>0.48(0.01)</td><td>0.06(0.00)</td><td>0.13(0.00)</td></tr><tr><td>0.24(0.07)</td><td>0.05(0.02) 0.38(0.02)</td><td>0.03(0.00)</td><td>0.08(0.00)</td></tr><tr><td>MDE SlicedWF</td><td>0.42(0.05) 0.80(0.14) 0.82(0.14)</td><td>0.04(0.00) 0.19(0.09)</td><td>0.12(0.00)</td></tr><tr><td>SinkWF</td><td>0.88(0.07) 0.99(0.00)</td><td>0.33(0.17)</td><td>0.41(0.15)</td></tr><tr><td></td><td></td><td>0.16(0.08)</td><td>0.92(0.03) 0.74(0.05)</td></tr><tr><td>SinkMarg</td><td>0.72(0.08) 0.92(0.02)</td><td></td><td></td></tr><tr><td></td><td colspan="2">Strong Contamination</td><td colspan="2"></td></tr><tr><td></td><td colspan="2">AUC-ROC</td><td colspan="2">AUC-PR</td></tr><tr><td rowspan="8"></td><td></td><td>α = 0.05</td><td>α = 0.15</td><td>α = 0.05</td><td>α = 0.15</td></tr><tr><td>COPOD</td><td>1.00(0.00)</td><td>0.96(0.00)</td><td>0.97(0.01)</td><td>0.77(0.01)</td></tr><tr><td>DIF</td><td>0.99(0.00)</td><td>0.49(0.09)</td><td>0.88(0.08)</td><td>0.19(0.04)</td></tr><tr><td>MDE</td><td>1.00(0.00)</td><td>1.00(0.00)</td><td>1.00(0.00)</td><td>1.00(0.00)</td></tr><tr><td>SlicedWF</td><td>1.00(0.00)</td><td>0.99(0.03)</td><td>1.00(0.01)</td><td>0.95(0.12)</td></tr><tr><td>SinkWF</td><td>1.00(0.00)</td><td>1.00(0.00)</td><td>1.00(0.00)</td><td>1.00(0.00)</td></tr><tr><td>SinkMarg</td><td>0.91(0.03)</td><td>0.93(0.02)</td><td>0.64(0.14)</td><td>0.89(0.02)</td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr></table>

The experimental results in Table 1 demonstrate that SinkWF consistently achieves the best performance across most scenarios. Notably, SinkWF exhibits a more pronounced performance advantage over SlicedWF in the weak contamination regime. This behavior is visually illuminated by Figure A.3 in the Supplementary Material, which highlights that the directional signal of contamination is significantly weaker in the weak

case.

As the data dimension d increases, this directional signal becomes even more severely diluted unless the number of random projections S in the sliced distance grows exponentially with d. In contrast, the entropic Optimal Transport used in SinkWF is inherently dimension-free, albeit at the expense of explicitly storing and computing the n × n cost matrix.

Additionally, the two robust joint WF algorithms perform better as the contamination fraction α increases, a trend particularly evident under weak contamination. However, SinkWF remains highly sensitive to variations in this contamination proportion.

Benefiting from the global geometric properties of the Wasserstein distance, our WF algorithms are capable of detecting outliers even when they are not exposed at the extreme edges of the clean data distribution. This capability is clearly illustrated in Figure A.3(a) in the Supplementary Material.

In the two-moons dataset, while baselines like DIF and COPOD can identify some outlying points, the contrast is far more stark in the concentric circles dataset. Here, the outliers concentrate in the region between the inner and outer circles, completely surrounded by normal points. In this challenging scenario, DIF, MDE, and COPOD mistakenly flag points on the outer boundary of the clean distribution as outliers. This limitation stems from their underlying mechanisms, which rely heavily on tail mass or spatial isolation to determine outlier scores.

By contrast, SinkWF and SinkMarg successfully isolate nearly all hidden outliers, while SlicedWF successfully recovers a significant portion of them. This is also reflected quantitatively in Table 1, where all three WF algorithms exhibit a substantial performance lead on the concentric circles dataset, particularly under weak contamination.

## 4.2 Benchmark Datasets

In this section, we systematically validate and compare our proposed framework across several standard benchmark datasets spanning two distinct modalities: tabular and graph-structured.

• Tabular Data: We utilize representative datasets from the comprehensive anomaly detection benchmark ADBench (Han et al., 2022).

• Graph-Structured Data: We evaluate our methods on the Tox21 molecular dataset. To process the graphs, we vectorize the molecular structures using the pretrained Mol2vec model (Jaeger et al., 2018). Both the raw datasets and the pretrained representations are retrieved using the DeepChem ecosystem (Ramsundar et al., 2019).

The empirical results and performance comparisons for the Tabular Data are summarized in Table 2. The results for Graph-Structured Data are given in Table 6 in the Supplementary Material.

## 4.2.1 Discussion of Empirical Results

Tabular Data (Table 2). As shown in Table 2, SinkWF achieves state-of-the-art or near-optimal perfor mance on datasets with higher contamination fractions α (e.g., yeast, cardiotocography, satellite, and shuttle), though its performance degrades in low-α regimes (e.g., pendigits). Conversely, SlicedWF exhibits a clear advantage on low-to-moderate dimensional datasets (such as pendigits, WBC, and satellite). In these settings, the lower-dimensional informative features ensure that random 1D projections preserve the class-separating geometric structure. However, the performance of SlicedWF systematically degrades as the dimensionality increases (e.g., optdigits).

A notable failure mode for all Wasserstein Filtering (WF) variants occurs on the smtp dataset. In scenarios characterized by an extremely large sample size n and a microscopic contamination rate α, the total mass of the contaminated samples becomes statistically negligible. Consequently, the Wasserstein distance between the full empirical distribution and any sub-sample is dominated by the benign intra-class variation of the clean data rather than the geometry of the outliers. In this regime, the core principle of our framework, maximizing distributional discrepancy by withholding an α-fraction of the data, loses its signal. In contrast, marginal-based methods such as COPOD and isolation-based methods like DIF succeed here; their mechanisms (marginal extremeness and recursive isolation, respectively) are designed to identify isolated, rare anomalies in low-dimensional spaces without requiring a population-level geometric signal.

Table 2: Full results of tabular benchmark datasets. SinkMarg on smtp dataset is abandoned since its runtime exceeds 24h.
<table><tr><td rowspan="13">Tabular</td><td colspan="2"></td><td colspan="6">AUC-ROC</td></tr><tr><td>Dataset</td><td>(n, d, α)</td><td>COPOD</td><td>DIF</td><td>MDE</td><td>SlicedWF</td><td>SinkWF</td><td>SinkMarg</td></tr><tr><td>vertebral</td><td>(240,6, 12.50%)</td><td>0.39</td><td>0.39</td><td>0.43</td><td>0.61</td><td>0.74</td><td>0.34</td></tr><tr><td>yeast</td><td>(1484, 8, 34.16%)</td><td>0.41</td><td>0.42</td><td>0.50</td><td>0.50</td><td>0.70</td><td>0.35</td></tr><tr><td>annthyroid breastw</td><td>(7200, 6, 7.42%)</td><td>0.68</td><td>0.71</td><td>0.67</td><td>0.59</td><td>0.61</td><td>0.82</td></tr><tr><td>cardio</td><td>(683, 9, 34.99%)</td><td>0.99</td><td>0.98</td><td>0.99</td><td>0.99 0.80</td><td>0.99</td><td>0.98</td></tr><tr><td></td><td>(1831, 21, 9.61%)</td><td>0.88 0.65</td><td>0.95</td><td>0.82</td><td>0.65</td><td>0.76</td><td>0.56</td></tr><tr><td>tocography</td><td>(2114, 21, 22.04%) (80, 19, 16.25%)</td><td></td><td>0.68 0.81</td><td>0.71 0.60</td><td>0.72</td><td>0.77</td><td>0.46</td></tr><tr><td>Hepatitis Lymph</td><td></td><td>0.80 1.00</td><td>1.00</td><td>0.92</td><td>0.95</td><td>0.68 0.96</td><td>0.63</td></tr><tr><td>shuttle</td><td>(148, 18, 4.05%) (49097, 9, 7.15%)</td><td>0.99</td><td>0.99</td><td>0.96</td><td>0.95</td><td>0.98</td><td>0.99</td></tr><tr><td>pendigits</td><td>(6870, 16, 2.27%)</td><td>0.82</td><td>0.96</td><td>0.76</td><td>0.97</td><td>0.58</td><td>0.75 0.72</td></tr><tr><td>satellite</td><td>(6435, 36, 31.64%)</td><td>0.50</td><td>0.73</td><td>0.50</td><td>0.75</td><td>0.66</td><td>0.66</td></tr><tr><td>WBC</td><td>(223, 9, 4.48%)</td><td>0.99</td><td>0.98</td><td>0.99</td><td>1.00</td><td>0.94</td><td>0.97</td></tr><tr><td>smtp</td><td>(95156, 3, 0.03%)</td><td>0.90</td><td>0.93</td><td>0.83</td><td>0.77</td><td>0.58</td><td></td></tr><tr><td>optdigits</td><td>(5216, 64, 2.88%)</td><td>0.71</td><td>0.57</td><td>0.50</td><td>0.58</td><td>0.67</td><td></td></tr><tr><td>mean wins</td><td></td><td>0.77</td><td>0.79</td><td>0.73</td><td>0.77</td><td>0.76</td><td>0.35</td></tr><tr><td rowspan="13">Tabular</td><td></td><td></td><td>28.57%</td><td>35.71%</td><td>7.14%</td><td>28.57%</td><td>28.57%</td><td>0.66 7.14%</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>AUC-PR</td><td></td><td></td></tr><tr><td>Dataset vertebral</td><td>(n, d, α)</td><td>COPOD</td><td>DIF</td><td>MDE</td><td>SlicedWF</td><td>SinkWF</td><td>SinkMarg</td></tr><tr><td>yeast</td><td>(240,6,12.50%) (1484, 8, 34.16%)</td><td>0.10 0.31</td><td>0.10 0.30</td><td>0.11</td><td>0.15</td><td>0.23</td><td>0.09</td></tr><tr><td>annthyroid</td><td>(7200, 6, 7.42%)</td><td>0.16</td><td>0.27</td><td>0.34</td><td>0.34 0.10</td><td>0.52</td><td>0.27</td></tr><tr><td>breastw</td><td>(683, 9, 34.99%)</td><td>0.98</td><td>0.94</td><td>0.16</td><td>0.99</td><td>0.14</td><td>0.24</td></tr><tr><td>cardio</td><td>(1831, 21, 9.61%)</td><td></td><td></td><td>0.97</td><td>0.52</td><td>0.99</td><td>0.94</td></tr><tr><td>tocography</td><td>(2114, 21, 22.04%)</td><td>0.46</td><td>0.58</td><td>0.24</td><td></td><td>0.31</td><td>0.17</td></tr><tr><td></td><td></td><td>0.37</td><td>0.42</td><td>0.35</td><td>0.41</td><td>0.46</td><td>0.24</td></tr><tr><td>Hepatitis</td><td>(80, 19, 16.25%)</td><td>0.42</td><td>0.47</td><td>0.19</td><td>0.34</td><td>0.36</td><td>0.22</td></tr><tr><td>Lymph</td><td>(148, 18, 4.05%)</td><td>0.87</td><td>0.95</td><td>0.21</td><td>0.31</td><td>0.54</td><td>0.65</td></tr><tr><td>shuttle</td><td>(49097, 9, 7.15%)</td><td>0.91</td><td>0.88</td><td>0.74</td><td>0.77</td><td>0.96</td><td>0.18</td></tr><tr><td>pendigits</td><td>(6870, 16, 2.27%)</td><td>0.11</td><td>0.29</td><td>0.05</td><td>0.38</td><td>0.03</td><td>0.07</td></tr><tr><td>satellite</td><td>(6435, 36, 31.64%)</td><td>0.34</td><td>0.64</td><td>0.32</td><td>0.68</td><td>0.69</td><td>0.49</td></tr><tr><td>WBC</td><td>(223, 9, 4.48%)</td><td>0.90</td><td>0.67</td><td>0.80</td><td>0.96</td><td>0.38</td><td>0.69</td></tr><tr><td>smtp</td><td>(95156, 3, 0.03%)</td><td>0.47</td><td>0.39</td><td>0.06</td><td>0.01</td><td>0.00</td><td></td></tr><tr><td>optdigits</td><td>(5216, 64, 2.88%)</td><td>0.05</td><td>0.03</td><td>0.03</td><td>0.04</td><td>0.28</td><td>0.02</td></tr><tr><td colspan="2">mean</td><td>0.46</td><td>0.50</td><td>0.33</td><td>0.43</td><td>0.42</td><td>0.33</td></tr><tr><td colspan="2">wins</td><td>7.14%</td><td>28.57%</td><td>0.00%</td><td>21.43%</td><td>50.00%</td><td>0.00%</td></tr></table>

For SinkMarg, the best performance is achieved on annthyroid, which is a large dataset (n = 7200) with a low dimension (d = 6) and simple structure, where leave-one-out Sinkhorn scoring is highly efective at identifying well-isolated outliers. However, SinkMarg performs poorly on vertebral (AUC-ROC of 0.34) and yeast (0.35), performing worse than a random classifier. This highlights a fundamental limitation of SinkMarg: it cannot capture group-structured outliers, where contaminated points form tight, coherent clusters rather than isolated Dirac masses.

Finally, MDE is theoretically elegant but practically weak across all tabular datasets. The datasets in ADBench feature highly heterogeneous distributions (comprising medical records, sensor data, and handwritten digits). The spectral filtering mechanism of MDE relies heavily on the assumption that the clean distribution is isotropic, a condition that is severely violated in real-world tabular data.

Graph-Structured Data (Table 6 in the Supplementary Material). In the high-dimensional molecular representation space of Tox21, MDE fails completely. MDE iteratively computes the leading eigenvector of the sample covariance matrix and filters samples along this direction. In d = 300 dimensions with only a small fraction of outliers (∼3–12%), the leading eigenvector of the covariance matrix is entirely dominated by the variance of the clean data, rendering the difuse contamination signal spectrally invisible.

For SlicedWF, using S = 1000 random projections (see Appendix C.2) in a 300-dimensional space is insuficient. Most random 1D projections fail to align with the meaningful directions of molecular variation, making the sliced Wasserstein distance a poor proxy for the true $W _ { 1 }$ metric. This highlights a fundamental statistical-computational tradeof: while SlicedWF enjoys an ${ \mathcal { O } } ( n )$ space complexity, it sufers a severe approximation bottleneck in high dimensions. In contrast, SinkWF incurs an $\scriptstyle { \mathcal { O } } ( n ^ { 2 } )$ space complexity but preserves full geometric fidelity, rendering it robust to high dimensionality.

When the true contamination rate α is unknown, we show in Appendix C.2 that our automatic determination procedure for α incurs negligible performance loss compared to using the ground-truth value.

## 4.3 Robust Generative Learning

In this subsection, we demonstrate how our proposed framework can facilitate downstream tasks, specifically robust generative learning. The goal of generative learning is to synthesize novel samples ${ \bf x } _ { i } \ ( i \geq n + 1 )$ that mimic the target distribution $P ^ { * }$ based on corrupted observed data $\widetilde { \mathbf { x } } _ { 1 } , \ldots , \widetilde { \mathbf { x } } _ { n }$ . For this experiment, we deploy a difusion-based generative model—the Denoising Difusion Probabilistic Model (DDPM) (Ho et al., 2020)—on the MNIST handwritten digit dataset.

We combine the standard training and testing partitions of MNIST to obtain a total sample size of $n = 7 0 , 0 0 0$ . The dataset is subjected to two distinct types of adversarial corruptions:

• Strong Contamination: The top half of each contaminated image is completely masked (replaced with white pixels).

• Weak Contamination: A random one-fourth portion of each contaminated image is masked, as illustrated in Figure 1(a).

We compare the visual quality and distribution of the synthesized images generated by DDPM when trained on: (i) the raw contaminated dataset, and (ii) the clean dataset retrieved after filtering out suspected outliers using our proposed methods.

Performance is evaluated using two sets of metrics: the classification accuracy of outlier detection (AUC-ROC and AUC-PR) and the quality of the generated images, quantified by the Fr´echet Inception Distance (FID) (Heusel et al., 2017). Because the information in the corrupted regions is fundamentally unrecoverable, the FID is computed strictly between the synthesized images and the ground-truth clean (uncorrupted) images. Additionally, we evaluate a baseline scenario with no actual corruption $( \alpha = 0 )$ while still running our algorithms with a nominal input of $\alpha = 0 . 0 5$ . This allows us to quantify the potential performance degradation (or ”robustness $\mathrm { t a x } ^ { \prime \prime } )$ incurred by filtering uncorrupted datasets.

In our implementation, we leverage POT for SlicedWF and the memory-eficient GeomLoss library for SinkWF and SinkMarg. We initialize the parameters as ${ \bf b } _ { 0 } = { \bf 1 } + 1 0 ^ { - 3 } { \bf z }$ , where $\mathbf { z } \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } )$ , and utilize the squared Wasserstein-2 $\left( p = 2 \right)$ distance across all three WF variants. We set the optimization iterations to $T = 1$ for SinkWF and $T = 1 0$ for SlicedWF. All remaining hyperparameter configurations are detailed in Appendix C.3. The empirical results are summarized in Table 3.

Discussion of Generative Results. As reported in Table 3, SlicedWF achieves slightly superior performance compared to SinkWF in outlier classification metrics (AUC-ROC and AUC-PR), whereas SinkWF yields a lower (better) FID score. Notably, SinkMarg proves computationally impractical in this setting;

the extremely large sample size (n = 70, 000) severely slows down the leave-one-out Sinkhorn computations, making it scale poorly to large-scale datasets. Consistent with our previous experiments, the outlier detection capabilities of SinkWF improve as the contamination rate increases from α = 0.05 to α = 0.15.

Regarding image synthesis quality, the α = 0 baseline experiment demonstrates that applying our filtering algorithms to entirely clean data has a negligible impact on downstream generative modeling, indicating that our methods do not discard valuable clean samples. Conversely, the utility of filtering becomes striking under heavy contamination (α = 0.15): pre-filtering the training data with our WF algorithms yields a dramatic 3-to-4-fold reduction in FID score compared to training on the raw contaminated data, highlighting the necessity of robust data sanitization in generative workflows.

![](images/a2f903af1e7fdff2ee9a80fbd6d24bc7d8c455fb01cb07e0d49c2fd333e31ecf.jpg)  
(a) Original and contaminated images. From left to right: clean images, strong contaminated images, and weak contaminated images.

![](images/b7f19632fa60c4e2d3cad39339a1d7c769b57337e70b1ab6410a0b30f483f450.jpg)  
(b) Generated images. First 3 images from the left side: trained on raw data with α = 0.15 weak contamination. The 3 images on the right: trained after outlier removal by SlicedWF.  
Figure 1: Result of generated images without and with outlier removal preprocessing.

Table 3: Results of DDPM training on MNIST. FID evaluate the quality of generated images, AUC-ROC and AUC-PR evaluate the performance of outlier detection, weak and strong contamination are two modes of contamination. SinkMarg is abandoned since its runtime exceeds 24h.
<table><tr><td></td><td colspan="7">Weak contamination</td></tr><tr><td></td><td colspan="3">FID score</td><td colspan="2">AUC-ROC</td><td colspan="2">AUC-PR</td></tr><tr><td>Original data</td><td>α = 0</td><td>α = 0.05</td><td>α = 0.15</td><td>α = 0.05</td><td>α = 0.15</td><td>α = 0.05</td><td>α = 0.15</td></tr><tr><td>SlicedWF</td><td>12.45</td><td>17.23</td><td>21.13</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>11.25</td><td>20.77</td><td>4.83</td><td>1.00</td><td>1.00</td><td>1.00</td><td>1.00</td></tr><tr><td>SinkWF</td><td>10.13</td><td>16.17</td><td>4.83</td><td>0.80</td><td>1.00</td><td>0.71</td><td>1.00</td></tr><tr><td>SinkMarg</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td colspan="7">Strong contamination</td></tr><tr><td></td><td colspan="7">FID score</td></tr><tr><td>Original data</td><td>α = 0</td><td>α = 0.05</td><td>α = 0.15</td><td>α = 0.05</td><td>α = 0.15</td><td>α = 0.05</td><td>α = 0.15</td></tr><tr><td>SlicedWF</td><td>12.45</td><td>12.13</td><td>14.85</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>13.27</td><td>20.77</td><td>4.83</td><td>1.00</td><td>1.00</td><td>1.00</td><td>1.00</td></tr><tr><td>SinkWF</td><td>16.15</td><td>8.98</td><td>5.04</td><td>1.00</td><td>1.00</td><td>0.94</td><td>1.00</td></tr><tr><td>SinkMarg</td><td>一</td><td>1</td><td>一</td><td>一</td><td>一</td><td>1</td><td>一</td></tr></table>

## 5 Conclusion

In this paper, we introduced Wasserstein Filtering (WF), a novel task-agnostic sample selection framework designed to detect and eliminate outliers that cause significant distributional distortion under the Wasserstein distance. By isolating a clean subset of the data, the empirical measure of the remaining samples naturally serves as a robust estimator of the underlying clean distribution.

The core advantages of the WF framework are summarized as follows:

• Task-Agnostic Preprocessing: As a pure sample-selection scheme, WF operates independently of downstream learning tasks, making it a highly versatile preprocessing tool for general data sanitization.

• Minimal Distributional Assumptions: Unlike existing robust estimators, WF does not impose restrictive parametric assumptions on the clean distribution, nor does it require prior knowledge of its specific parameters.

• Computational Tractability and Adaptivity: Harnessing modern deep-learning optimization frameworks (e.g., automatic diferentiation and GPU acceleration), the optimization of sample weights is computationally eficient. Moreover, the framework is equipped with an automated data-driven procedure to estimate the unknown contamination rate α.

• Robustness to Non-Tail and Structured Outliers: By leveraging the global geometric properties of optimal transport, WF successfully identifies structured outliers and hidden anomalies that do not lie on the extreme tails of the clean distribution, surpassing the limitations of isolation- and density-based methods.

• Flexibility across Dimensions: The framework ofers scalable options tailored to diferent data regimes: SlicedWF provides ${ \mathcal { O } } ( n )$ space eficiency in large-sample, low-dimensional settings $( n \gg d )$ , while SinkWF maintains geometric fidelity in high-dimensional, moderate-sample regimes $( d \gg n )$

We systematically validated the empirical performance of the WF framework across tabular and graph data, demonstrating its consistent superiority over state-of-the-art baselines. Furthermore, on the theoretica front, we established the minimax optimality of the empirical risk under the Far Exclusion with Local Projection (FELP) contamination model, confirming that our framework achieves the optimal statistical rate of robust distribution estimation.

## Supplementary Material

This Supplementary Material contains the complete proofs for our theoretical results, additional algorithmic details, and further numerical evaluations.

## A Additional Methodological Details

## A.1 Projection-Based Implementation

In Algorithms 2 and Algorithm 1 below in A.3, the constraint $\mathrm { c o n v } ( \Delta _ { \alpha } )$ is handled by appending a penalty term to the objective function. In this section, we present an alternative approach based on direct projection. Specifically, we solve the projection problem onto the capped simplex conv $\left( \Delta _ { \alpha } \right)$ :

$$
\mathrm { p r o j } _ { \mathrm { c o n v } ( \Delta _ { \alpha } ) } ( \mathbf { b } _ { r } ) = \underset { \mathbf { z } \in \mathrm { c o n v } ( \Delta _ { \alpha } ) } { \mathrm { a r g m i n } } \frac { 1 } { 2 } \| \mathbf { z } - \mathbf { b } _ { r } \| ^ { 2 } .
$$

To solve this, we employ the iterative algorithm proposed by Adam and M´acha (2022), which utilizes a bisection method to find the root of the optimality equation established in their Theorem 3.8. Crucially, the operations involved in this projection are diferentiable almost everywhere, enabling seamless integration into gradient-based optimization via automatic diferentiation (e.g., PyTorch’s AutoGrad).

Specifically, for the Sinkhorn Wasserstein Filtering (SinkWF) algorithm, the optimal transport problem and its associated gradient computation are modified as follows:

$$
\begin{array} { r l } & { \mathbf { T } _ { r } = \underset { \mathbf { T } \in \Pi \left( \frac { 1 } { n } \mathbf { 1 } , \operatorname { p r o j } _ { \mathrm { c o n v } ( \Delta _ { \alpha } ) } ( \mathbf { b } _ { r } ) \right) } { \operatorname { a r g m i n } } \langle \mathbf { T } , \mathbf { C } \rangle + \lambda \underset { i , j \in [ n ] } { \sum } t _ { i j } \log ( t _ { i j } ) , } \\ & { \quad \quad \quad \quad \quad \mathbf { g } _ { r } = \mathbf { v } ^ { * } \frac { \displaystyle \partial \operatorname { p r o j } _ { \mathrm { c o n v } ( \Delta _ { \alpha } ) } ( \mathbf { b } _ { r } ) } { \displaystyle \partial \mathbf { b } _ { r } } . } \end{array}
$$

This adaptation can be applied analogously to the Sliced Wasserstein Filtering (SlicedWF) algorithm:

$$
\begin{array} { c } { { W _ { 1 , S } \left( \widetilde { P } _ { n } , P _ { \mathrm { p r o j } _ { \mathrm { c o n v } ( \Delta _ { \infty } ) } ( \mathbf { b } _ { \mathrm { r } } ) } \right) = \displaystyle \frac { 1 } { S } \sum _ { \mathbf { u } _ { \mathrm { f } } \sim \mathbb { U } ^ { d } , i \in [ S ] } \int _ { 0 } ^ { 1 } \left. F _ { \mathbf { u } _ { \mathrm { f } } ^ { \intercal } \widetilde { \mathbf { x } } , \widetilde { \mathbf { x } } \sim \widetilde { P } _ { n } } ^ { - 1 } ( t ) - F _ { \mathbf { u } _ { \mathrm { f } } ^ { \intercal } \mathbf { y } , \mathbf { y } \sim P _ { \mathrm { p r o j } _ { \mathrm { c o n v } ( \Delta _ { \infty } ) } ( \mathbf { b } _ { \mathrm { r } } ) } } ^ { - 1 } ( t ) \right. d t , } } \\ { { \mathbf { g } _ { r } = \displaystyle \frac { \partial W _ { 1 , S } \left( \widetilde { P } _ { n } , P _ { \mathrm { p r o j } _ { \mathrm { c o n v } ( \Delta _ { \alpha } ) } ( \mathbf { b } _ { \mathrm { r } } ) } \right) } { \partial \mathrm { p r o j } _ { \mathrm { c o n v } ( \Delta _ { \infty } ) } ( \mathbf { b } _ { r } ) } \displaystyle \frac { \partial \mathrm { p r o j } _ { \mathrm { c o n v } ( \Delta _ { \alpha } ) } ( \mathbf { b } _ { r } ) } { \partial \mathbf { b } _ { r } } . } } \end{array}
$$

By eliminating the penalty parameter $\beta ,$ this projection-based implementation bypasses the need for hyperparameter tuning on $\beta ,$ making it user-friendly in practice. We empirically evaluate the performance of this projection-based variant in Appendices C.1-C.3.

## A.2 Automatic Determination of the Contamination Level

In Algorithms 2 and Algorithm 1 below in A.3, the contamination level α is an input parameter that typically must be specified a priori. To address settings where the ground-truth contamination rate is unknown, we introduce an automatic selection strategy that dynamically estimates α during the optimization process.

Taking Algorithm 2 as an example, starting from an initial conservative estimate $\alpha _ { 0 } ~ ( \mathrm { e . g . } , \alpha _ { 0 } = 0 . 1 )$ we update the contamination level iteratively. Let $\mathbf { p } _ { r + 1 } = \mathrm { S o r t } ( \mathrm { s o f t m a x } ( \mathbf { b } _ { r + 1 } ) )$ denote the vector of sample weights sorted in descending order at iteration $r + 1$ . We then update $\alpha _ { r + 1 }$ according to:

$$
\alpha _ { r + 1 } = 1 - { \frac { \operatorname { a r g m a x } _ { i \in [ n - 1 ] } ( p _ { r + 1 , i } - p _ { r + 1 , i + 1 } ) } { n } } .
$$

The intuition behind this strategy is that clean and contaminated samples tend to be separated by a sharp transition in their assigned probability masses. By identifying the index of the largest adjacent gap (i.e., the “elbow” point) in the sorted weights, we can dynamically and adaptively partition the dataset into clean and contaminated subsets. The empirical performance of this adaptive α strategy is evaluated on benchmark datasets in Appendix C.

## A.3 SlicedWF Algorithm

The complete optimization process for the SlicedWF algorithm is given below.

Algorithm 3: Sliced Wasserstein Filtering Algorithm (SlicedWF)   
Input: $\widetilde { \mathcal { D } } , \alpha \in ( 0 , 1 ) , \mathbf { b } _ { 0 } , \gamma \in \mathbb { R } _ { + } , R , T , S \in \mathbb { Z } _ { + }$   
1 Normalize $\widetilde { \mathcal { D } }$   
2 for $t = 1 , \dots , T$ do   
3 for $r = 0 , \ldots , R - 1$ do   
4 Sample S random directions $\{ \mathbf { u } _ { 1 } , \dots , \mathbf { u } _ { S } \}$ uniformly from $\mathbb { S } ^ { d - 1 }$   
5 Compute $W _ { 1 , S } ( \widetilde { P } _ { n } , P _ { \mathrm { s o f t m a x } ( \mathbf { b } _ { r } ) } )$ via:   
$\frac { 1 } { S } \sum _ { i = 1 } ^ { S } \int _ { 0 } ^ { 1 } \left| F _ { \mathbf { u } _ { i } ^ { \top } \tilde { \mathbf { x } } , \tilde { \mathbf { x } } \sim \tilde { P } _ { n } } ^ { - 1 } ( t ) - F _ { \mathbf { u } _ { i } ^ { \top } \mathbf { y } , \mathbf { y } \sim P _ { \mathrm { s o f t m a x } ( \mathbf { b } _ { r } ) } } ^ { - 1 } ( t ) \right|$ dt   
Compute the gradient $\mathbf { g } _ { r } \mathbf { : }$   
$\begin{array} { r } { \partial \left[ W _ { 1 , S } ( \widetilde { P } _ { n } , P _ { \mathrm { s o f t m a x } ( \mathbf { b } _ { r } ) } ) - \beta \mathbf { 1 } ^ { \top } \mathrm { R e L U } \left( \mathrm { s o f t m a x } ( \mathbf { b } _ { r } ) - \frac { 1 } { ( 1 - \alpha ) n } \mathbf { 1 } \right) \right] } \end{array}$   
$\mathbf { g } _ { r } =$   
∂b<sub>r</sub>   
Update parameters using Adam:   
$\mathbf { b } _ { r + 1 } = \mathrm { A d a m } ( \gamma , \mathbf { b } _ { r } , \mathbf { g } _ { j , j \leq r } )$   
6 end   
7 Save $\mathbf { h } _ { t } = \mathrm { s o f t m a x } ( \mathbf { b } _ { R } )$   
8 end   
9 Thresholding: filter $\widetilde { \mathcal { D } }$ to obtain $S = \{ \widetilde { \mathbf { x } } _ { i } \mid \bar { h } _ { i } > \tau _ { \alpha } , i \in [ n ] \}$ where $\bar { h } _ { i }$ is the ith entry of   
$\begin{array} { r } { \bar { \mathbf { h } } = \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \mathbf { h } _ { t } , \tau _ { \alpha } } \end{array}$ is the α quantile of $\bar { h } _ { i } , i \in [ n ]$   
Output: S

## B Additional Details for Section 3: Theoretical Properties

## B.1 Convergence of the Estimator

## B.1.1 Population Setting

Lemma 1 (Complementary Wasserstein Resilience). Let $( \mathcal { X } , d )$ be a metric space, $p \in [ 1 , \infty )$ , and $\alpha \in [ 0 , 1 )$ $I f \mu \in \mathcal { Q } _ { p } ( \mathcal { X } )$ is $( \rho , \alpha )$ -resilient with respect to the $W _ { p }$ distance, then $\mu$ is also $\left( \frac { ( 1 - \alpha ) ^ { 1 / p } } { 1 - ( 1 - \alpha ) ^ { 1 / p } } \rho , 1 - \alpha \right)$ -resilient under $W _ { p }$ . That is, for every probability measure $\nu \in \mathcal { Q } _ { p } ( \mathcal { X } )$ satisfying $\begin{array} { r } { \nu \leq \frac { 1 } { \alpha } \mu . } \end{array}$ , we have:

$$
W _ { p } ( \mu , \nu ) \leq \frac { ( 1 - \alpha ) ^ { 1 / p } } { 1 - ( 1 - \alpha ) ^ { 1 / p } } \rho .
$$

Proof. Following the proof strategy of Lemma 10 in Steinhardt et al. (2018) (which establishes complementary mean resilience), let $\nu \in \mathcal { Q } _ { p } ( \mathcal { X } )$ be any probability measure satisfying $\begin{array} { r } { \nu \leq \frac { 1 } { \alpha } \mu } \end{array}$ , and define

$$
\eta : = { \frac { \mu - \alpha \nu } { 1 - \alpha } } .
$$

It is straightforward to verify that $\eta \in \mathcal { Q } _ { p } ( \mathcal { X } )$ and $\begin{array} { r } { \eta \leq \frac { 1 } { 1 - \alpha } \mu . } \end{array}$

Let π be an optimal coupling between η and ν with respect to the $W _ { p }$ distance. We construct a coupling

![](images/62efde5c6acd478684fab9cd16fab36c066b3f9bfc6f5e62de073b6867a91712.jpg)  
Figure A.2: Roadmap of estimator convergence

$\Gamma _ { 1 }$ defined as:

$$
\Gamma _ { 1 } : = \alpha ( \mathrm { I d } , \mathrm { I d } ) _ { \# } \nu + ( 1 - \alpha ) \pi ,
$$

where Id denotes the identity map. Since the first marginal of $\Gamma _ { 1 }$ is $\alpha \nu + ( 1 - \alpha ) \eta = \mu$ and its second marginal is $\nu ,$ it constitutes a valid coupling between $\mu$ and $\nu .$ Consequently, we obtain:

$$
\begin{array} { l } { { \displaystyle W _ { p } ^ { p } ( \mu , \nu ) \leq \int d ( { \bf x } , { \bf y } ) ^ { p } d \Gamma _ { 1 } ( { \bf x } , { \bf y } ) } } \\ { ~ } \\ { { \displaystyle ~ = ( 1 - \alpha ) \int d ( { \bf x } , { \bf y } ) ^ { p } d \pi ( { \bf x } , { \bf y } ) } } \\ { ~ } \\ { { \displaystyle ~ = ( 1 - \alpha ) W _ { p } ^ { p } ( \nu , \eta ) } . } \end{array}
$$

Taking the $p \textmd { - }$ th root on both sides yields $W _ { p } ( \mu , \nu ) \leq ( 1 - \alpha ) ^ { 1 / p } W _ { p } ( \nu , \eta )$ . By the triangle inequality, we have:

$$
W _ { p } ( \mu , \nu ) \leq ( 1 - \alpha ) ^ { 1 / p } \left( W _ { p } ( \mu , \nu ) + W _ { p } ( \mu , \eta ) \right) .
$$

Rearranging terms, we find:

$$
\begin{array} { r l r } & { \left( 1 - ( 1 - \alpha ) ^ { 1 / p } \right) W _ { p } ( \mu , \nu ) \le ( 1 - \alpha ) ^ { 1 / p } W _ { p } ( \mu , \eta ) } & \\ & { } & \\ & { } & { \qquad \le ( 1 - \alpha ) ^ { 1 / p } \rho , \qquad } \end{array}
$$

where the final inequality follows from the relation $\begin{array} { r } { \eta \leq \frac { 1 } { 1 - \alpha } \mu } \end{array}$ and the assumed $( \rho , \alpha )$ -resilience of $\mu .$ . This completes the proof. □

Lemma 2 (Risk under Support Total Variation Contamination). Assume the true distribution $P ^ { * } \in { \mathcal { G } } \subseteq$ $\mathcal { W } _ { 1 } ( \rho , \alpha )$ and the contaminated distribution $\widetilde { P }$ satisfies $\| \widetilde { P } - P ^ { * } \| _ { \mathrm { S T V } } \leq \alpha$ for some $\textstyle 0 \leq \alpha \leq { \frac { 1 } { 2 } }$ . Then, the

population Wasserstein Filtering estimator $\widehat { P } _ { \mathrm { W F } }$ satisfies:

$$
W _ { 1 } ( \widehat { P } _ { \mathrm { W F } } , P ^ { * } ) \lesssim \kappa _ { \widetilde { P } , P ^ { * } } ( S _ { \mathrm { W F } } ) \vee \rho ,
$$

where the discrepancy term $\kappa _ { \widetilde { P } , P ^ { * } } ( S _ { \mathrm { W F } } )$ is defined as:

$$
\kappa _ { \widetilde { P } , P ^ { * } } ( S _ { \mathrm { W F } } ) = \operatorname* { i n f } _ { \mu \le \frac { 1 } { \widetilde { P } ( S _ { \mathrm { W F } } \backslash \bar { S } ) } P ^ { * } } W _ { 1 } \left( \widetilde { P } _ { S _ { \mathrm { W F } } \backslash \bar { S } } , \mu \right) ,
$$

and $\bar { S }$ is the shared support guaranteed by the definition of the support total variation distance.

Proof. Since $\| \widetilde { P } - P ^ { * } \| _ { \mathrm { S T V } } \leq \alpha$ , by the definition of the Support Total Variation (STV) distance, there exists a measurable subset ${ \bar { S } } \subseteq { \mathcal { X } }$ such that $\widetilde P ( \bar { S } ) = P ^ { * } ( \bar { S } ) = 1 - \alpha$ and $\tilde { P } | _ { \bar { S } } = P ^ { * } | _ { \bar { S } }$ . Consequently, we can write:

$$
\begin{array} { r l } & { W _ { 1 } ( \widehat { P } _ { \mathrm { W F } } , P ^ { * } ) \leq W _ { 1 } ( \widehat { P } _ { \mathrm { W F } } , \widetilde { P } _ { \bar { S } } ) + W _ { 1 } ( \widetilde { P } _ { \bar { S } } , P ^ { * } ) } \\ & { \qquad = W _ { 1 } ( \widehat { P } _ { \mathrm { W F } } , \widetilde { P } _ { \bar { S } } ) + W _ { 1 } ( P _ { \bar { S } } ^ { * } , P ^ { * } ) } \\ & { \qquad \leq W _ { 1 } ( \widehat { P } _ { \mathrm { W F } } , \widetilde { P } _ { \bar { S } } ) + \rho , } \end{array}
$$

where the last inequality follows from the relation $\begin{array} { r } { P _ { \bar { S } } ^ { * } \leq \frac { 1 } { 1 - \alpha } P ^ { * } } \end{array}$ and the assumed $( \rho , \alpha )$ -resilience of $P ^ { * }$

Recall the Kantorovich-Rubinstein dual formulation for $W _ { 1 }$ . Let $\left( \phi ^ { * } , S _ { \mathrm { W F } } \right)$ be an optimal pair realizing $W _ { 1 } ( \widehat { P } _ { \mathrm { W F } } , \widetilde { P } )$ , where $\phi ^ { * }$ is the optimal 1-Lipschitz dual potential. This yields:

$$
\begin{array} { r l } & { W _ { 1 } ( \widehat { P } _ { \mathrm { W F } } , \widetilde { P } ) = \underset { \| \phi \| _ { \mathrm { L i p } } \leq 1 } { \operatorname* { s u p } } \displaystyle \int \phi d \widehat { P } _ { \mathrm { W F } } - \int \phi d \widetilde { P } } \\ & { \quad \quad \quad = \displaystyle \int \phi ^ { * } d \widehat { P } _ { \mathrm { W F } } - \int \phi ^ { * } d \widetilde { P } } \\ & { \quad \quad \quad = \displaystyle \operatorname* { s u p } _ { S \subseteq \operatorname* { s u p } ( \widetilde { P } ) , \widetilde { P } ( S ) = 1 - \alpha } \displaystyle \int \phi ^ { * } d \widetilde { P } _ { S } - \int \phi ^ { * } d \widetilde { P } , } \end{array}
$$

where the third equality follows directly from the definition of the estimator $\widehat { P } _ { \mathrm { W F } }$ . Thus, for any subset $S \subseteq \operatorname { s u p p } ( \widetilde { P } )$ with ${ \widetilde { P } } ( S ) = 1 - \alpha$ , we have:

$$
\int \phi ^ { * } d \widetilde { P } _ { S } \leq \int \phi ^ { * } d \widehat { P } _ { \mathrm { W F } } .\tag{17}
$$

Now, let ψ be any 1-Lipschitz function $( \| \psi \| _ { \mathrm { L i p } } \leq 1 )$ . We observe that:

$$
\begin{array} { r l } { \int \psi d \tilde { R } \mathrm { w } - \int \psi d \tilde { P } _ { 5 } \leq \int \psi d \tilde { R } \mathrm { w } _ { 5 } - \int \psi d \tilde { P } _ { 5 } + \int \phi ^ { * } d \tilde { P } _ { 6 } } & { } \\ & { \qquad = \frac { 1 } { 1 - \alpha } ( \int _ { \mathbb { S } _ { \mathrm { e x } } } ( \psi + \phi ^ { * } ) d \tilde { P } ^ { - } \int _ { \mathbb { S } _ { \mathrm { e x } } } ( \psi + \phi ^ { * } ) d \tilde { P } ) } \\ & { \qquad = \frac { 1 } { 1 - \alpha } ( \int _ { \mathbb { S } _ { \mathrm { e x } } \times \mathbb { S } } ( \psi + \phi ^ { * } ) d \tilde { P } ^ { - } \int _ { \mathbb { S } _ { \mathrm { e x } } \times \mathbb { S } _ { 0 } } ( \psi + \phi ^ { * } ) d \tilde { P } ) } \\ & { \qquad = \frac { \tilde { P } ( \tilde { S } _ { \mathrm { e x } } \times \tilde { S } , \ \xi ) } { ( \int _ { \mathbb { S } _ { \mathrm { e x } } \times \mathbb { S } } \xi ) } ( f ( \psi + \phi ^ { * } ) d \tilde { P } _ { \mathrm { s e x } } ^ { - } \int ( \psi ( + \phi ^ { * } ) d \tilde { P } _ { 5 } \xi _ { 5 , \mathrm { e x } } )  } \\ & { \qquad = \frac { \tilde { \partial } } { 1 - \alpha } ( \int \psi ( \mathrm { w } ) + \psi ^ { * } ( \mathrm { x } ) d \tilde { P } _ { \mathrm { s e x } } ^ { - } \psi ( \mathrm { x } ) - \int ( \psi ( \mathrm { y } ) + \phi ^ { * } ( \mathrm { y } ) ) d \tilde { P } _ { 5 , \mathrm { g x } } \psi ( \mathrm { y } ) ) } \\ &  \qquad = \frac { \delta } { 1 - \alpha } ( \psi ( \mathrm { s } ) - \nu ( \mathrm { y } ) + \delta ^ { * } ( \mathrm { x } ) - \phi ^ { * } ( \mathrm { y } \end{array}
$$

where the first inequality is obtained by setting $\widetilde { P } _ { S } = \widetilde { P } _ { \bar { S } }$ in $( 1 7 )$ ; the third equality follows because $\widetilde { P } ( S _ { \mathrm { W F } } ) =$ ${ \widetilde { P } } ( { \bar { S } } ) = 1 - \alpha$ , which implies $\widetilde { P } ( S _ { \mathrm { W F } } \setminus \bar { S } ) = \widetilde { P } ( \bar { S } \setminus S _ { \mathrm { W F } } ) : = \delta \le \alpha ;$ the fifth equality holds for any arbitrary coupling π of $\widetilde { P } _ { S _ { \mathrm { W F } } \backslash \bar { S } }$ and $\widetilde { P } _ { \bar { S } \backslash S _ { \mathrm { W F } } }$ ; the last inequality is due to the Lipschitz constraints $\| \psi \| _ { \mathrm { L i p } } \leq 1$ and $\| \phi ^ { * } \| _ { \mathrm { L i p } } \leq 1$

Taking the supremum over $\psi$ on the left-hand side and the infimum over all valid couplings $\pi \in$ $\Pi ( \widetilde { P } _ { S _ { \mathrm { W F } } \backslash \bar { S } } , \widetilde { P } _ { \bar { S } \backslash S _ { \mathrm { W F } } } )$ on the right-hand side, we get:

$$
\begin{array} { r l } { \mathbb { W } _ { 1 } \cdot [ \hat { \tilde { \mu } } _ { \mathrm { S U S } } , \hat { \tilde { \mu } } _ { \hat { \mathcal { S } } } ] - } & { \operatorname* { s u p } _ { 0 \leq t \leq T } \int _ { 0 } + \hat { \mu } _ { \mathrm { S U S } } - \int _ { 0 } + \hat { \tilde { \mu } } _ { \mathrm { S U } } } \\ & { \leq \frac { 2 } { 1 - \tilde { \mu } _ { \mathrm { S U S } } } \int _ { 0 } + \hat { \mu } _ { \mathrm { S U S } } \int _ { 0 } + \int _ { 0 } + \hat { \mu } _ { \mathrm { S U S } } \int _ { 0 } + \hat { \mu } _ { \mathrm { S U S } } \int _ { 0 } } \\ & { = \frac { 2 \tilde { \mu } _ { \mathrm { S U S } } } { 1 - \tilde { \mu } _ { \mathrm { S U S } } } \mathbb { W } _ { 1 } ( \hat { \tilde { \mu } } _ { \mathrm { S U S } } , \hat { \mu } _ { \mathrm { S U S } } ) } \\ & { = \frac { 2 \tilde { \mu } _ { \mathrm { S U S } } } { 1 - \tilde { \mu } _ { \mathrm { S U S } } } \mathbb { W } _ { 1 } ( \hat { \tilde { \mu } } _ { \mathrm { S U S } } , \hat { \mu } _ { \mathrm { S U S } } ) } \\ & { \leq \frac { 2 \tilde { \mu } _ { \mathrm { S U S } } } { 1 - \tilde { \mu } _ { \mathrm { S U S } } } \{ \frac { 1 } { \exp ( \tilde { \mu } _ { \mathrm { S U S } } , \hat { \mu } _ { \mathrm { S U S } } ) } + \frac { 2 \tilde { \mu } _ { \mathrm { S U S } } } { 1 - \tilde { \mu } _ { \mathrm { S U S } } } \{ \frac { \tilde { \mu } _ { \mathrm { S U S } } } { 1 - \tilde { \mu } _ { \mathrm { S U S } } } \} \Bigg \} } \\ &  \leq \frac { \tilde { \mu } _ { \mathrm { S U S } } } { 1 - \tilde { \mu } _ { \mathrm { S U S } } }  \end{array}
$$

where the third inequality uses the fact that ${ \widetilde { P } } = P ^ { * }$ on the support ${ \bar { S } } ;$ the fifth inequality follows from Lemma $1 ;$ the final inequality is verified by noting that $\delta \leq \alpha \leq { \frac { 1 } { 2 } }$ (which implies $\textstyle { \frac { 2 \delta } { 1 - \alpha } } \leq 2$ and $\frac { 4 ( 1 - \delta ) } { 1 - \alpha } \leq 8 )$

Combining this result with the initial bound $W _ { 1 } ( \widehat { P } _ { \mathrm { W F } } , P ^ { * } ) \leq W _ { 1 } ( \widehat { P } _ { \mathrm { W F } } , \widetilde { P } _ { \bar { S } } ) + \rho$ completes the proof.

We are now ready to prove establish the population risk bounds.

Proof of Theorem 1. Suppose for contradiction that $\widetilde P ( S _ { \mathrm { W F } } \cap E _ { \rho } ) > 0$ . By (13) in the definition of population FELP contamination, there exists a subset $S _ { + } \subseteq \operatorname { s u p p } ( \widetilde { P } )$ with $\tilde { P } ( S _ { + } ) = 1 - \alpha$ and $S _ { + } \cap E _ { \rho } = \emptyset$ such that:

$$
W _ { 1 } ( \widehat { P } _ { \mathrm { W F } } , \widetilde { P } ) < W _ { 1 } ( \widetilde { P } _ { S _ { + } } , \widetilde { P } ) .
$$

Since $ { \widetilde { P } } _ { S _ { + } }$ is a valid $( 1 - \alpha )$ -truncated version of ${ \widetilde { P } } _ { * }$ , this strictly contradicts the optimality of $S _ { \mathrm { W F } }$ . Therefore, we must have $S _ { \mathrm { W F } } \cap E _ { \rho } = \emptyset$ , which in turn implies $( S _ { \mathrm { W F } } \setminus \bar { S } ) \cap E _ { \rho } = \emptyset$ . Since $S _ { \mathrm { W F } } \setminus \bar { S } \subseteq E _ { \rho } \cup N _ { \rho }$ , it follows that $S _ { \mathrm { W F } } \setminus \bar { S } \subseteq N _ { \rho }$

Let $\tilde { P } ( S _ { \mathrm { W F } } \setminus \bar { S } ) = \delta$ , where $0 \leq \delta \leq \alpha$ . We define the push-forward measure $\mu = T _ { \# } \widetilde { P } _ { S _ { \mathrm { W F } } \backslash \bar { S } }$ . For any measurable set $B \subseteq { \mathcal { X } }$ , we have:

$$
\begin{array} { r l } & { \mu ( B ) = \widetilde { P } _ { S _ { \mathrm { W F } } \backslash \bar { S } } ( T ^ { - 1 } ( B ) ) } \\ & { \qquad \le \displaystyle \frac { 1 } { \delta } \widetilde { P } | _ { N _ { \rho } } ( T ^ { - 1 } ( B ) ) } \\ & { \qquad = \displaystyle \frac { 1 } { \delta } T _ { \# } \widetilde { P } | _ { N _ { \rho } } ( B ) } \\ & { \qquad \le \displaystyle \frac { 1 } { \delta } P ^ { * } ( B ) , } \end{array}
$$

where the last inequality follows from (14), thereby verifying that $\mu \leq { \textstyle \frac { 1 } { \delta } } { \cal P } ^ { * }$

Next, let $\pi _ { 0 } = ( \mathrm { I d } , T ) _ { \# } \widetilde { P } _ { S _ { \mathrm { W F } } \backslash \bar { S } }$ . It is clear that $\pi _ { 0 } \in \Pi ( \widetilde { P } _ { S _ { \mathrm { W F } } \backslash \bar { S } } , \mu )$ represents a valid coupling. Consequently, we obtain:

$$
\begin{array} { l l l } { W _ { 1 } ( \widetilde { P } _ { S _ { \mathrm { W F } } \backslash \bar { S } } , \mu ) = \underset { \pi \in \Pi ( \widetilde { P } _ { S _ { \mathrm { W F } } \backslash \bar { S } } , \mu ) } { \operatorname* { i n f } } \int d ( \mathbf { x } , \mathbf { y } ) d \pi ( \mathbf { x } , \mathbf { y } ) } \\ { \quad \quad \quad \quad \quad \leq \displaystyle \int d ( \mathbf { x } , \mathbf { y } ) d \pi _ { 0 } ( \mathbf { x } , \mathbf { y } ) } \\ { \quad \quad \quad \quad \quad = \displaystyle \int d ( \mathbf { x } , T ( \mathbf { x } ) ) d \widetilde { P } _ { S _ { \mathrm { W F } } \backslash \bar { S } } ( \mathbf { x } ) } \\ { \quad \quad \quad \quad \leq \rho . } \end{array}
$$

This shows there exists a measure $\mu = T _ { \# } \widetilde { P } _ { S _ { \mathrm { W F } } \backslash \bar { S } } \leq \frac { 1 } { \delta } P ^ { * }$ such that $W _ { 1 } ( \mu , \widetilde { P } _ { S _ { \mathrm { W F } } \backslash \bar { S } } ) \leq \rho ;$ , which directly implies that $\kappa _ { \widetilde { P } , P ^ { \ast } } ( S _ { \mathrm { W F } } ) \leq \rho .$ Applying Lemma 2, we conclude that $W _ { 1 } ( \widehat { P } _ { \mathrm { W F } } , P ^ { * } ) \lesssim \rho .$ □

To establish the minimax lower bound in Theorem 2, we first introduce the following auxiliary lemma, which provides a generic lower bound on the minimax risk via the modulus of continuity.

Lemma 3 (Generic Lower Bound for Minimax Risk). Let ${ \mathcal { G } } \subseteq { \mathcal { W } } _ { 1 } ( \rho , \alpha )$ . The minimax estimation risk over $\mathcal { G }$ is lower bounded as:

$$
\operatorname* { i n f } _ { \mathcal { T } } \operatorname* { s u p } _ { P ^ { \ast } \in \mathcal { G } } \mathcal { R } _ { 1 , P ^ { \ast } , \mathcal { T } } ( \alpha , \rho ) \geq \frac { 1 } { 2 } \omega ( \mathcal { G } , \alpha , \rho ) ,
$$

where the modulus of continuity $\omega ( \mathcal { G } , \alpha , \rho )$ is defined as:

$$
\omega ( \mathcal { G } , \alpha , \rho ) = \operatorname* { s u p } \left\{ W _ { 1 } ( P _ { 0 } , P _ { 1 } ) : P _ { 0 } , P _ { 1 } \in \mathcal { G } , \mathcal { C } _ { P _ { 0 } } ^ { \mathrm { F E L P } } ( \alpha , \rho ) \cap \mathcal { C } _ { P _ { 1 } } ^ { \mathrm { F E L P } } ( \alpha , \rho ) \neq \emptyset \right\} .
$$

Proof. Following the standard construction of the modulus of continuity in robust estimation (see, e.g., Donoho and Liu, 1988; Zhu et al., 2022), we fix an arbitrary estimator T . For any $P _ { 0 } , P _ { 1 } \in \mathcal { G }$ sharing a

common contaminated distribution $\widetilde { P } \in \mathcal { C } _ { P _ { 0 } } ^ { \mathrm { F E L P } } ( \alpha , \rho ) \cap \mathcal { C } _ { P _ { 1 } } ^ { \mathrm { F E L P } } ( \alpha , \rho )$ , the triangle inequality yields:

$$
\begin{array} { r l } & { W _ { 1 } ( P _ { 0 } , P _ { 1 } ) \leq W _ { 1 } ( P _ { 0 } , { \mathcal T } ( { \widetilde P } ) ) + W _ { 1 } ( { \mathcal T } ( { \widetilde P } ) , P _ { 1 } ) } \\ & { \qquad \leq 2 \operatorname* { m a x } \Big \{ W _ { 1 } ( P _ { 0 } , { \mathcal T } ( { \widetilde P } ) ) , W _ { 1 } ( P _ { 1 } , { \mathcal T } ( { \widetilde P } ) ) \Big \} . } \end{array}
$$

Taking the supremum over all such valid pairs $( P _ { 0 } , P _ { 1 } )$ and their shared contaminated distributions $\widetilde { P }$ on both sides, we obtain:

$$
\begin{array} { r l } & { \omega ( \mathcal { G } , \alpha , \rho ) = \underset { P _ { 0 } , P _ { 1 } , \tilde { P } } { \operatorname* { s u p } } W _ { 1 } ( P _ { 0 } , P _ { 1 } ) } \\ & { \quad \quad \quad \leq 2 \underset { P _ { 0 } , P _ { 1 } , \tilde { P } } { \operatorname* { s u p } } \operatorname* { m a x } \Big \{ W _ { 1 } ( P _ { 0 } , \mathcal { T } ( \tilde { P } ) ) , W _ { 1 } ( P _ { 1 } , \mathcal { T } ( \tilde { P } ) ) \Big \} } \\ & { \quad \quad \quad \leq 2 \underset { P ^ { * } \in \mathcal { G } } { \operatorname* { s u p } } \underset { \tilde { P } \in \mathcal { C } _ { P ^ { * } \times \mathbb { L } ^ { \mathbb { R } , P } ( \alpha , \rho ) } } { \operatorname* { s u p } } W _ { 1 } ( P ^ { * } , \mathcal { T } ( \tilde { P } ) ) } \\ & { \quad \quad \quad = 2 \underset { P ^ { * } \in \mathcal { G } } { \operatorname* { s u p } } \mathcal { R } _ { 1 , P ^ { * } , \mathcal { T } } ( \alpha , \rho ) . } \end{array}
$$

Taking the infimum over all estimators $\tau$ on both sides completes the proof.

We are now ready to establish the minimax optimality results.

Proof of Theorem 2. We first consider the general setting where the distribution class $\mathcal { G }$ satisfies $\mathcal { G } \subseteq$ $\mathcal { W } _ { 1 } ( \rho , \alpha )$

Let $\mathbf { e } _ { 1 } = ( 1 , 0 , \hdots , 0 ) ^ { \top } \in \mathbb { R } ^ { d }$ denote the first standard basis vector, and let $Q _ { U \mathbf { e } _ { 1 } }$ represent the uniform law on the line segment:

$$
L _ { \frac { \rho } { 4 } } : = \left\{ t \mathbf { e } _ { 1 } : 0 \leq t \leq \frac { \rho } { 4 } \right\} \subset \mathbb { R } ^ { d } ,
$$

where $U \sim \operatorname { U n i f } [ 0 , { \frac { \rho } { 4 } } ]$ . We define the candidate distributions $P _ { 0 } , P _ { 1 }$ and a shared contaminated distribution $\widetilde { P }$ as follows:

$$
P _ { 0 } = ( 1 - \alpha ) Q _ { U { \bf e } _ { 1 } } + \alpha \delta _ { - \frac { \beta } { 2 } { \bf e } _ { 1 } } , \quad P _ { 1 } = ( 1 - \alpha ) Q _ { U { \bf e } _ { 1 } } + \alpha \delta _ { \frac { \beta } { 2 } { \bf e } _ { 1 } } , \quad \tilde { P } = ( 1 - \alpha ) Q _ { U { \bf e } _ { 1 } } + \alpha \delta _ { - \frac { \beta } { 4 } { \bf e } _ { 1 } } .
$$

To apply Lemma $^ { 3 , }$ we verify that $\widetilde { P } \in \mathcal { C } _ { P _ { 0 } } ^ { \mathrm { F E L P } } ( \alpha , \rho ) \cap \mathcal { C } _ { P _ { 1 } } ^ { \mathrm { F E L P } } ( \alpha , \rho )$ . By choosing the shared support $\bar { S } = L _ { \frac { \rho } { 4 } }$ , we have:

$$
P _ { 0 } ( \bar { S } ) = P _ { 1 } ( \bar { S } ) = \widetilde { P } ( \bar { S } ) = 1 - \alpha , \quad \mathrm { a n d } \quad P _ { 0 } | _ { \bar { S } } = P _ { 1 } | _ { \bar { S } } = \widetilde { P } | _ { \bar { S } } = ( 1 - \alpha ) Q _ { U \mathbf { e } _ { 1 } } ,
$$

which immediately implies $\| \widetilde { P } - P _ { i } \| _ { \mathrm { S T V } } \leq \alpha$ for $i = 0 , 1$

Next, we check the Far Exclusion and Local Projection (FELP) conditions. The only contaminated point under ${ \widetilde { P } } { \mathrm { ~ i s ~ } } - { \frac { \rho } { 4 } } \mathbf { e } _ { 1 }$ , which satisfies:

$$
d \left( - { \frac { \rho } { 4 } } \mathbf { e } _ { 1 } , \operatorname { s u p p } ( P _ { 0 } ) \right) = d \left( - { \frac { \rho } { 4 } } \mathbf { e } _ { 1 } , \operatorname { s u p p } ( P _ { 1 } ) \right) = { \frac { \rho } { 4 } } < \rho .
$$

Consequently, the far contamination region $E _ { \rho }$ is empty for both $P _ { 0 }$ and $P _ { 1 }$ . For the local projection condition, we define the transport maps $T _ { 0 }$ and $T _ { 1 }$ by setting $T _ { 0 } \left( - { \textstyle \frac { \rho } { 4 } } \mathbf { e } _ { 1 } \right) = - { \textstyle \frac { \rho } { 2 } } \mathbf { e } _ { 1 }$ and $T _ { 1 } \left( - { \frac { \rho } { 4 } } \mathbf { e } _ { 1 } \right) = { \frac { \rho } { 2 } } \mathbf { e } _ { 1 }$ These maps satisfy:

$$
d \left( - \frac { \rho } { 4 } \mathbf { e } _ { 1 } , T _ { 0 } \left( - \frac { \rho } { 4 } \mathbf { e } _ { 1 } \right) \right) = \frac { \rho } { 4 } < \rho , \quad d \left( - \frac { \rho } { 4 } \mathbf { e } _ { 1 } , T _ { 1 } \left( - \frac { \rho } { 4 } \mathbf { e } _ { 1 } \right) \right) = \frac { 3 \rho } { 4 } < \rho ,
$$

as well as $T _ { 0 \neq } \widetilde { P } | _ { N _ { \rho } } = \alpha \delta _ { - \frac { \rho } { 2 } { \bf e } _ { 1 } } \leq P _ { 0 }$ and $T _ { 1 \# } \widetilde { P } | _ { N _ { \rho } } = \alpha \delta _ { \frac { \rho } { 2 } { \bf e } _ { 1 } } \leq P _ { 1 }$ . This confirms that $\widetilde { P } \in \mathcal { C } _ { P _ { 0 } } ^ { \mathrm { F E L P } } ( \alpha , \rho ) \cap$ $\mathcal { C } _ { P _ { 1 } } ^ { \mathrm { F E L P } } ( \alpha , \rho )$

Assuming $P _ { 0 } , P _ { 1 } \in \mathcal { G }$ , Lemma 3 yields the following minimax lower bound:

$$
\operatorname* { i n f } _ { \mathcal { T } } \operatorname* { s u p } _ { P ^ { * } \in \mathcal { G } } \mathcal { R } _ { 1 , P ^ { * } , \mathcal { T } } ( \alpha , \rho ) \geq \frac { 1 } { 2 } \omega ( \mathcal { G } , \alpha , \rho ) \geq \frac { 1 } { 2 } W _ { 1 } ( P _ { 0 } , P _ { 1 } ) = \frac { 1 } { 2 } \alpha \rho \gtrsim \rho ,
$$

where the final inequality holds since $\alpha \geq \alpha _ { 0 } > 0$ . Combined with the upper bound from Theorem 1, which establishes that

$$
\operatorname* { i n f } _ { T } \operatorname* { s u p } _ { P ^ { * } \in \mathcal { G } } \mathcal { R } _ { 1 , P ^ { * } , \mathcal { T } } ( \alpha , \rho ) \leq \operatorname* { s u p } _ { P ^ { * } \in \mathcal { G } } \mathcal { R } _ { 1 , P ^ { * } , \widehat { P } _ { \mathrm { W F } } } ( \alpha , \rho ) \lesssim \rho ,
$$

we obtain the tight minimax rate of $\Theta ( \rho )$

We now turn to the specialized case where $\mathcal { G } = \mathcal { G } _ { \mathrm { c o v } } ( \sigma )$ . It remains to verify that $P _ { 0 } , P _ { 1 } \in \mathcal { G } _ { \mathrm { c o v } } ( \sigma )$ . For a general distribution of the form $P _ { t , \eta } = ( 1 - \alpha ) Q _ { U \mathbf { e } _ { 1 } } + \alpha \delta _ { t \mathbf { e } _ { 1 } }$ with $U \sim \operatorname { U n i f } [ 0 , \eta ]$ , its covariance matrix is given by:

$$
\pmb { \Sigma } _ { P _ { t , \eta } } = \left( \left( 1 - \alpha \right) \frac { \eta ^ { 2 } } { 1 2 } + \alpha ( 1 - \alpha ) \left( t - \frac { \eta } { 2 } \right) ^ { 2 } \right) \mathbf { e } _ { 1 } \mathbf { e } _ { 1 } ^ { \top } .
$$

For $P _ { 0 } { \mathrm { . } }$ we have $\begin{array} { r } { \eta = \frac { \rho } { 4 } } \end{array}$ and $t = - \textstyle { \frac { \rho } { 2 } }$ , while for $P _ { 1 }$ , we have $\begin{array} { r } { \eta = \frac { \rho } { 4 } } \end{array}$ and $\begin{array} { r } { t = \frac { \rho } { 2 } } \end{array}$ . To ensure that $P _ { 0 } , P _ { 1 } \in \mathcal { G } _ { \mathrm { c o v } } ( \sigma )$ we require the operator norm of the covariance to be bounded by $\sigma ^ { 2 }$

$$
( 1 - \alpha ) \frac { \rho ^ { 2 } } { 1 9 2 } + \alpha ( 1 - \alpha ) \left( \frac { \rho } { 2 } + \frac { \rho } { 8 } \right) ^ { 2 } \leq \sigma ^ { 2 } \implies \rho \lesssim \frac { \sigma } { \sqrt { 1 - \alpha ^ { 2 } } } .
$$

Furthermore, by Theorem 2 of Nietert et al. (2023), the resilience parameter under $\mathcal { G } _ { \mathrm { c o v } } ( \sigma )$ scales as $\rho \asymp$ $\sigma { \sqrt { d \alpha } }$ . Substituting this relation, we find that the covariance constraint is satisfied provided that the dimension satisfies $\begin{array} { r } { d \lesssim \frac { 1 } { \alpha ( 1 - \alpha ^ { 2 } ) } } \end{array}$ , which holds by assumption. This completes the proof. □

## B.1.2 Empirical Setting

Lemma 4 (Upper Bound on Empirical Resilience). Let $P ^ { * } \in { \mathcal { Q } } ( { \mathcal { X } } )$ be a probability measure, let $P _ { n } =$ $\textstyle { \frac { 1 } { n } } \sum _ { i = 1 } ^ { n } \delta _ { \mathbf { x } _ { i } }$ be its empirical measure, and let $0 \leq \alpha < 1$ . Define the population and empirical resilience parameters respectively as:

$$
\rho = \operatorname* { s u p } _ { \mu \leq \frac { 1 } { 1 - \alpha } P ^ { * } } W _ { p } ( P ^ { * } , \mu ) \quad a n d \quad \rho _ { n } = \operatorname* { s u p } _ { \mu \leq \frac { 1 } { 1 - \alpha } P _ { n } } W _ { p } ( P _ { n } , \mu ) .
$$

Then, the empirical resilience parameter satisfies:

$$
\rho _ { n } \leq \rho + \left( 1 + \frac { 1 } { ( 1 - \alpha ) ^ { 1 / p } } \right) W _ { p } ( P _ { n } , P ^ { * } ) .
$$

Proof. For any α-trimmed version of $P _ { n }$ satisfying $\textstyle \mu _ { \alpha } \leq { \frac { 1 } { 1 - \alpha } } P _ { n }$ , the Radon–Nikodym theorem guarantees the existence of a measurable function $\begin{array} { r } { f \colon \mathcal { X } \to \left\lceil 0 , \frac { 1 } { 1 - \alpha } \right\rceil } \end{array}$ with $\textstyle { \int f d P _ { n } = 1 }$ such that:

$$
\mu _ { \alpha } ( A ) = \int _ { A } f d P _ { n }
$$

for all Borel measurable sets $A \subseteq { \mathcal { X } }$

Let $\pi \in \Pi ( P _ { n } , P ^ { * } )$ be an optimal coupling between $P _ { n }$ and $P ^ { * }$ . We construct a new probability measure

$\bar { \pi }$ on $\mathcal { X } ^ { 2 }$ by defining, for any Borel set $B \subseteq \mathcal { X } ^ { 2 }$

$$
\bar { \pi } ( B ) = \int _ { B } f ( { \bf x } ) d \pi ( { \bf x } , { \bf y } ) .
$$

For any Borel set $A \subseteq { \mathcal { X } }$ , the first marginal of ¯π satisfies:

$$
\bar { \pi } ( A \times \mathcal { X } ) = \int _ { A \times \mathcal { X } } f ( \mathbf { x } ) d \pi ( \mathbf { x } , \mathbf { y } ) = \int _ { A } f ( \mathbf { x } ) d P _ { n } ( \mathbf { x } ) = \mu _ { \alpha } ( A ) .
$$

Letting $\nu _ { \alpha }$ denote the second marginal of ¯π, we have for any Borel set $B \subseteq { \mathcal { X } }$

$$
\nu _ { \alpha } ( B ) = \bar { \pi } ( \mathcal { X } \times B ) = \int _ { \mathcal { X } \times B } f ( \mathbf { x } ) d \pi ( \mathbf { x } , \mathbf { y } ) \leq \frac { 1 } { 1 - \alpha } P ^ { * } ( B ) ,
$$

which confirms that $\nu _ { \alpha }$ is indeed a feasible α-trimmed version of $P ^ { * }$

Using $\bar { \pi }$ as a (possibly sub-optimal) coupling between $\mu _ { \alpha }$ and $\nu _ { \alpha }$ , we bound their Wasserstein distance as:

$$
\begin{array} { l } { { \displaystyle W _ { p } ( \mu _ { \alpha } , \nu _ { \alpha } ) = \left( \operatorname* { i n f } _ { \pi _ { 0 } \in \Pi ( \mu _ { \alpha } , \nu _ { \alpha } ) } \int \| { \bf x - y } \| ^ { p } d \pi _ { 0 } ( { \bf x } , { \bf y } ) \right) ^ { 1 / p } } } \\ { { \displaystyle \qquad \leq \left( \int \| { \bf x - y } \| ^ { p } d \bar { \pi } ( { \bf x } , { \bf y } ) \right) ^ { 1 / p } } } \\ { { \displaystyle \qquad \leq \left( \frac { 1 } { 1 - \alpha } \int \| { \bf x - y } \| ^ { p } d \pi ( { \bf x } , { \bf y } ) \right) ^ { 1 / p } } } \\ { { \displaystyle \qquad = \frac { 1 } { ( 1 - \alpha ) ^ { 1 / p } } W _ { p } ( P _ { n } , P ^ { * } ) } . } \end{array}
$$

Applying the triangle inequality for the Wasserstein metric, we obtain:

$$
\begin{array} { l l } { { W _ { p } ( P _ { n } , \mu _ { \alpha } ) \le W _ { p } ( P _ { n } , P ^ { * } ) + W _ { p } ( P ^ { * } , \nu _ { \alpha } ) + W _ { p } ( \nu _ { \alpha } , \mu _ { \alpha } ) } } \\ { { \displaystyle \qquad \le W _ { p } ( P ^ { * } , \nu _ { \alpha } ) + \left( 1 + \frac { 1 } { ( 1 - \alpha ) ^ { 1 / p } } \right) W _ { p } ( P _ { n } , P ^ { * } ) } } \\ { { \displaystyle \qquad \le \operatorname* { s u p } _ { \nu \le \frac { 1 } { 1 - \alpha } P ^ { * } } W _ { p } ( P ^ { * } , \nu ) + \left( 1 + \frac { 1 } { ( 1 - \alpha ) ^ { 1 / p } } \right) W _ { p } ( P _ { n } , P ^ { * } ) } . } \end{array}
$$

Since this bound holds for any arbitrary α-trimmed measure $\textstyle \mu _ { \alpha } \leq { \frac { 1 } { 1 - \alpha } } P _ { n }$ , taking the supremum over all such $\mu _ { \alpha }$ yields:

$$
\operatorname* { s u p } _ { \mu \leq \frac { 1 } { 1 - \alpha } P _ { n } } W _ { p } ( P _ { n } , \mu ) \leq \operatorname* { s u p } _ { \nu \leq \frac { 1 } { 1 - \alpha } P ^ { * } } W _ { p } ( P ^ { * } , \nu ) + \left( 1 + \frac { 1 } { ( 1 - \alpha ) ^ { 1 / p } } \right) W _ { p } ( P _ { n } , P ^ { * } ) ,\tag{18}
$$

which completes the proof.

We are now ready to establish the empirical risk results.

Proof of Theorem 3. By setting $p \ = \ 1$ in Lemma 4 and using the assumption that $\alpha ~ \leq ~ { \frac { 1 } { 2 } }$ , we obtain $P _ { n } \in { \mathcal { W } } _ { 1 } ( { \mathcal { O } } ( \rho + W _ { 1 } ( P _ { n } , P ^ { * } ) ) , \alpha )$

Using the triangle inequality, we write $W _ { 1 } ( \widehat { P } _ { \mathrm { W F } } , P ^ { * } ) \ \leq \ W _ { 1 } ( \widehat { P } _ { \mathrm { W F } } , P _ { n } ) + W _ { 1 } ( P _ { n } , P ^ { * } )$ . Since $\Vert \widetilde { P } _ { n } -$ $P _ { n } \| _ { \mathrm { S T V } } \leq \alpha$ , we can apply Lemma 2 by treating ${ \widetilde { P } } _ { n }$ as the contaminated distribution $\widetilde { P }$ and the empirical

measure $P _ { n }$ as the clean target $P ^ { * }$ . This yields:

$$
W _ { 1 } ( \widehat { P } _ { \mathrm { W F } } , P ^ { * } ) \leq W _ { 1 } ( \widehat { P } _ { \mathrm { W F } } , P _ { n } ) + W _ { 1 } ( P _ { n } , P ^ { * } ) \lesssim \kappa _ { \widetilde { P } _ { n } , P _ { n } } ( S _ { \mathrm { W F } } ) \vee ( \rho + W _ { 1 } ( P _ { n } , P ^ { * } ) ) ,
$$

which can be explicitly written as:

$$
\begin{array} { r } { W _ { 1 } ( \widehat { P } _ { \mathrm { W F } } , P ^ { * } ) \lesssim \operatorname* { m a x } \left\{ \underset { \mu \leq \frac { 1 } { \widetilde { P } _ { n } ( S _ { \mathrm { W F } } \backslash \tilde { S } ) } } { \operatorname* { i n f } } { W _ { 1 } } \left( \widetilde { P } _ { n S _ { \mathrm { W F } } \backslash \tilde { S } } , \mu \right) , \rho + W _ { 1 } ( P _ { n } , P ^ { * } ) \right\} , } \end{array}
$$

where $\bar { S } \subseteq \{ \mathbf { x } _ { j } : j \in [ n ] , \mathbf { x } _ { j } = \widetilde { \mathbf { x } } _ { j } \}$ is the set of uncorrupted samples, satisfying $| \bar { S } | = ( 1 - \alpha ) n$ . Taking the expectation on both sides with respect to the contaminated data generating distribution $\widetilde { \mathbb { P } } _ { n }$ , we get:

$$
\mathbb { E } _ { \widetilde { \mathbb { P } } _ { n } } \left[ W _ { 1 } ( \widehat { P } _ { \mathrm { W F } } , P ^ { * } ) \right] \lesssim \operatorname* { m a x } \left\{ \mathbb { E } _ { \widetilde { \mathbb { P } } _ { n } } \left[ \operatorname* { i n f } _ { \mu \leq \frac { 1 } { P _ { n } ( \mathrm { S W F } \backslash \widetilde { S } ) } P _ { n } } W _ { 1 } \left( \widetilde { P } _ { n S _ { \mathrm { W F } } \backslash \widetilde { S } } , \mu \right) \right] , \ \rho + \mathbb { E } _ { \widetilde { \mathbb { P } } _ { n } } \left[ W _ { 1 } ( P _ { n } , P ^ { * } ) \right] \right\} .\tag{19}
$$

Next, by substituting the efective empirical parameter $\mathcal { O } ( \rho + W _ { 1 } ( P _ { n } , P ^ { * } ) )$ in place of $\rho$ in the definition of the empirical FELP condition, we can apply the exact same conditioning and projection arguments as in the proof of Theorem 1. This directly implies:

$$
\operatorname* { i n f } _ { \mu \leq \frac { 1 } { \tilde { P } _ { n } ( S _ { \mathrm { W F } } \setminus \bar { S } ) } P _ { n } } W _ { 1 } \left( \widetilde { P } _ { n S _ { \mathrm { W F } } \setminus \bar { S } } , \mu \right) \lesssim \rho + W _ { 1 } ( P _ { n } , P ^ { * } ) .
$$

Taking the expectation with respect to $\widetilde { \mathbb { P } } _ { n }$ on both sides and substituting this result back into (19) completes the proof. □

Lemma 5 (Empirical Lifting under FELP Contamination). For any $P ^ { * } \in \mathcal G$ and any $\begin{array} { r } { \widetilde { P } \in \mathcal { C } _ { P ^ { * } } ^ { \mathrm { F E L P } } \left( \frac { \alpha } { 5 } , \rho \right) } \end{array}$ , the following properties hold:

1. First, we have $T _ { \# } \tilde { P } | _ { N _ { \rho } } \leq P ^ { * } | _ { \bar { S } ^ { c } }$

2. Second, let $\begin{array} { r } { \eta : = P ^ { * } | _ { \bar { S } } = \widetilde { P } | _ { \bar { S } } , \nu _ { \mathrm { l o c } } : = \widetilde { P } | _ { N _ { \rho } } , \nu _ { \mathrm { f a r } } : = \widetilde { P } | _ { E _ { \rho } } , \lambda _ { \mathrm { l o c } } : = T _ { \# } \nu _ { \mathrm { l o c } } , a n d \lambda _ { \mathrm { f a r } } : = P ^ { * } | _ { \bar { S } ^ { c } } - \lambda _ { \mathrm { l o c } } \geq 0 } \end{array}$ For any coupling $\pi _ { \mathrm { f a r } } \in \Pi ( \lambda _ { \mathrm { f a r } } , \nu _ { \mathrm { f a r } } )$ , we define the joint coupling $\pi \in \Pi ( P ^ { * } , \widetilde P )$ as:

$$
\pi = ( \mathrm { I d } , \mathrm { I d } ) _ { \# } \eta + ( T , \mathrm { I d } ) _ { \# } \nu _ { \mathrm { l o c } } + \pi _ { \mathrm { f a r } } .
$$

Under the i.i.d. sample generation $( \mathbf { x } _ { i } , \widetilde { \mathbf { x } } _ { i } ) _ { i = 1 } ^ { n } \sim \pi ^ { n }$ , we define the event ${ \mathcal { E } } _ { 2 }$ as:

$$
\mathcal { E } _ { 2 } = \left\{ \begin{array} { l l } { \exists S _ { + } \subseteq \{ \widetilde { \mathbf { x } } _ { 1 } , \dotsc , \widetilde { \mathbf { x } } _ { n } \} , | S _ { + } | = ( 1 - \alpha ) n , S _ { + } \cap E _ { n , \rho } = \emptyset \ s u c h \ t h a t { : } } \\ { \qquad \ \qquad \ \operatorname* { s u p } _ { W _ { 1 } } \left( \displaystyle \frac { 1 } { | S | } \sum _ { \widetilde { \mathbf { x } } _ { i } \in S } \delta _ { \widetilde { \mathbf { x } } _ { i } } , \widetilde { P } _ { n } \right) < W _ { 1 } \left( \displaystyle \frac { 1 } { | S _ { + } | } \sum _ { \widetilde { \mathbf { x } } _ { i } \in S _ { + } } \delta _ { \widetilde { \mathbf { x } } _ { i } } , \widetilde { P } _ { n } \right) \Bigg \} . } \\ { \ | S | = ( 1 - \alpha ) n , S \cap E _ { n , \rho } \neq \emptyset } \end{array} \right.
$$

Then, we have $\textstyle \pi ^ { n } ( { \mathcal { E } } _ { 2 } ^ { c } ) \leq \delta _ { n } \leq { \frac { 1 } { 2 0 } }$

Furthermore, letting $\begin{array} { r } { \mathcal { E } _ { 1 } ~ = ~ \big \{ \sum _ { i = 1 } ^ { n } \mathbb { I } _ { \{ \mathbf { x } _ { i } \neq \widetilde { \mathbf { x } } _ { i } \} } \leq \alpha n \big \} } \end{array}$ , we have $\textstyle \pi ^ { n } ( \mathcal { E } _ { 1 } \cap \mathcal { E } _ { 2 } ) \geq { \frac { 3 } { 4 } }$ , and the second (right) marginal distribution of the conditional joint measure $\pi ^ { n } ( \cdot , \cdot \mid \mathcal { E } _ { 1 } \cap \mathcal { E } _ { 2 } )$ belongs to $\bar { \mathcal { D } } _ { n } ^ { \mathrm { F E L P } } ( P ^ { * } , \alpha )$

Proof. We first bound the probability of the event $\mathcal { E } _ { 1 }$ . By applying Markov’s inequality, we obtain:

$$
\begin{array} { r l } {  { \pi ^ { n } ( \mathcal { E } _ { 1 } ) = 1 - \pi ^ { n } ( \mathcal { E } _ { 1 } ^ { c } ) } } \\ & { \ge 1 - \frac { \mathbb { E } _ { \pi ^ { n } } [ \sum _ { i = 1 } ^ { n } \mathbb { I } _ { \{ \mathbf { x } _ { i } \ne \widetilde { \mathbf { x } } _ { i } \} } ] } { \alpha n } } \\ & { \ge \frac { 4 } { 5 } , } \end{array}
$$

where the second inequality follows from the assumption that $\begin{array} { r } { \widetilde { P } \in \mathcal { C } _ { P ^ { * } } ^ { \mathrm { F E L P } } \left( \frac { \alpha } { 5 } , \rho \right) } \end{array}$ and the construction of the joint coupling π, which ensures that $\textstyle \pi ( \mathbf { x } \neq { \widetilde { \mathbf { x } } } ) \leq { \frac { \alpha } { 5 } }$ . Applying the union bound, we obtain:

$$
\pi ^ { n } ( \mathcal { E } _ { 1 } \cap \mathcal { E } _ { 2 } ) \geq \pi ^ { n } ( \mathcal { E } _ { 1 } ) - \pi ^ { n } ( \mathcal { E } _ { 2 } ^ { c } ) \geq \frac { 4 } { 5 } - \delta _ { n } \geq \frac { 3 } { 4 } .
$$

Next, we show that the second (right) marginal of the conditional distribution $\pi ^ { n } ( \cdot , \cdot \mid \mathcal { E } _ { 1 } \cap \mathcal { E } _ { 2 } )$ belongs to $\mathcal { D } _ { n } ^ { \mathrm { F E L P } } ( P ^ { * } , \alpha )$ . To establish this, it sufices to show that the empirical samples $\widetilde { \mathbf { x } } _ { 1 } , \ldots , \widetilde { \mathbf { x } } _ { n }$ satisfy the empirical FELP contamination conditions, i.e., $\widetilde { P } _ { n } \in \mathcal { C } _ { P _ { n } } ^ { \mathrm { F E L P } } ( \alpha , \rho )$

On the event ${ \mathcal { E } } _ { 1 } \cap { \mathcal { E } } _ { 2 }$ , the first two conditions of FELP contamination (the support total variation bound and the far exclusion condition) are guaranteed by the definitions of $\mathcal { E } _ { 1 }$ and $\mathcal { E } _ { 2 } .$ , respectively. Thus, we only need to verify the third condition, namely the local projection property.

To do this, we partition the contaminated empirical support. Let $S _ { 1 } ~ = ~ \{ \widetilde { \bf x } _ { j } ~ : ~ j ~ \in ~ [ n ] , ~ ( { \bf x } _ { j } , \widetilde { \bf x } _ { j } ) ~ \sim$ $( T , \operatorname { I d } ) _ { \# } \nu _ { \log } \}$ . For any $\widetilde { \mathbf { x } } _ { j } \in S _ { 1 }$ , we have:

$$
d ( \widetilde { \mathbf { x } } _ { j } , \mathrm { s u p p } ( P _ { n } ) ) \leq d ( \widetilde { \mathbf { x } } _ { j } , \mathbf { x } _ { j } ) = d ( \widetilde { \mathbf { x } } _ { j } , T ( \widetilde { \mathbf { x } } _ { j } ) ) \leq \rho ,
$$

which implies $S _ { 1 } \subseteq N _ { n , \rho }$ . Conversely, let $S _ { 2 } = \{ \widetilde { \mathbf { x } } _ { j } : j \in [ n ] , ( \mathbf { x } _ { j } , \widetilde { \mathbf { x } } _ { j } ) \sim \pi _ { \mathrm { f a r } } \}$ . For any $\widetilde { \mathbf { x } } _ { j } \in S _ { 2 }$ , the definition of $\pi _ { \mathrm { f a r } }$ ensures that $\widetilde { \mathbf { x } } _ { j } \in E _ { \rho }$ . Consequently,

$$
d ( \widetilde { \mathbf { x } } _ { j } , \mathrm { s u p p } ( P _ { n } ) ) \geq d ( \widetilde { \mathbf { x } } _ { j } , \mathrm { s u p p } ( P ^ { * } ) ) \geq \rho ,
$$

which verifies that $S _ { 2 } \subseteq E _ { n , \rho }$

Since the contaminated support can be decomposed as

$$
S _ { 1 } \cup S _ { 2 } = \operatorname { s u p p } ( \widetilde { P } _ { n } ) \setminus \{ \widetilde { \mathbf { x } } _ { z } : z \in [ n ] , \widetilde { \mathbf { x } } _ { z } = \mathbf { x } _ { z } \} ,
$$

and since the empirical local and far regions partition the contaminated support such that

$$
\begin{array} { r } { ( N _ { n , \rho } \setminus \{ \widetilde { \bf x } _ { z } : z \in [ n ] , \widetilde { \bf x } _ { z } = { \bf x } _ { z } \} ) \cap E _ { n , \rho } = \emptyset , } \end{array}
$$

we identify $S _ { 1 } = N _ { n , \rho } \setminus \{ \widetilde { \mathbf { x } } _ { z } : z \in [ n ] , \widetilde { \mathbf { x } } _ { z } = \mathbf { x } _ { z } \}$ and $S _ { 2 } = E _ { n , \rho }$

We now construct the empirical local projection map $T _ { n } \colon N _ { n , \rho } \to \operatorname { s u p p } ( P _ { n } )$ by setting:

$$
T _ { n } ( \widetilde { \mathbf { x } } _ { j } ) = \left\{ \begin{array} { l l } { T ( \widetilde { \mathbf { x } } _ { j } ) } & { \mathrm { i f ~ } \widetilde { \mathbf { x } } _ { j } \in S _ { 1 } , } \\ { \widetilde { \mathbf { x } } _ { j } } & { \mathrm { i f ~ } \widetilde { \mathbf { x } } _ { j } \in N _ { n , \rho } \setminus S _ { 1 } . } \end{array} \right.
$$

By construction, this map satisfies $d ( \widetilde { \mathbf { x } } _ { j } , T _ { n } ( \widetilde { \mathbf { x } } _ { j } ) ) \leq \rho$ for all $\widetilde { \mathbf { x } } _ { j } \in N _ { n , \rho } ,$ and $T _ { n \# } \widetilde { P } _ { n } | _ { N _ { n , \rho } } \leq P _ { n }$

This confirms that $\widetilde { P } _ { n } \in { \mathcal { C } _ { P _ { n } } ^ { \mathrm { F E L P } } } ( \alpha , \rho )$ holds on ${ \mathcal { E } } _ { 1 } \cap { \mathcal { E } } _ { 2 }$ . Consequently, the second marginal distribution of the joint conditional measure $\pi ^ { n } ( \cdot , \cdot \mid \ \mathcal { E } _ { 1 } \cap \mathcal { E } _ { 2 } )$ belongs to the empirical FELP family $\mathcal { D } _ { n } ^ { \mathrm { F E L P } } ( P ^ { * } , \alpha )$ , completing the proof. □

We now prove the minimax optimality results in the empirical case.

Proof of Theorem 4. We begin by establishing the upper bound. Taking $\mathcal { T } _ { n } = \widehat { P } _ { \mathrm { W F } }$ in the definition of the empirical minimax risk, we have:

$$
\begin{array} { r l } & { \mathcal { R } _ { 1 , n } ( \alpha , \rho ) = \underset { \mathcal { T } _ { n } } { \operatorname* { i n f } } \underset { P ^ { * } \in \mathcal { G } } { \operatorname* { s u p } } \underset { \widetilde { \mathbb { P } } _ { n } \in \mathcal { D } _ { n } ^ { \mathrm { F E L P } } ( P ^ { * } , \alpha ) } { \operatorname* { s u p } } \mathbb { E } _ { \widetilde { \mathbb { P } } _ { n } } \left[ W _ { 1 } ( \mathcal { T } _ { n } ( \widetilde { P } _ { n } ) , P ^ { * } ) \right] } \\ & { \quad \quad \quad \quad \leq \underset { P ^ { * } \in \mathcal { G } } { \operatorname* { s u p } } \underset { \widetilde { \mathbb { P } } _ { n } \in \mathcal { D } _ { n } ^ { \mathrm { F E L P } } ( P ^ { * } , \alpha ) } { \operatorname* { s u p } } \mathbb { E } _ { \widetilde { \mathbb { P } } _ { n } } \left[ W _ { 1 } ( \widehat { P } _ { \mathrm { W F } } , P ^ { * } ) \right] } \\ & { \quad \quad \quad \lesssim \rho + \underset { P ^ { * } \in \mathcal { G } } { \operatorname* { s u p } } \mathbb { E } _ { \mathbb { P } _ { n } } \left[ W _ { 1 } ( P _ { n } , P ^ { * } ) \right] , } \end{array}
$$

where the last inequality follows directly from Theorem 3, which holds uniformly for any empirical FELP contamination distribution $\widetilde { \mathbb { P } } _ { n } \in \mathcal { D } _ { n } ^ { \mathrm { F E L P } } ( P ^ { * } , \alpha )$ and any $P ^ { * } \in { \mathcal { G } }$

Next, we establish the minimax lower bound. By definition of the empirical FELP family, the uncontaminated empirical measure $P _ { n }$ satisfies $\mathbb { P } _ { n } \in \mathcal { D } _ { n } ^ { \mathrm { F E L P } } ( P ^ { * } , \alpha )$ . Thus, we immediately obtain:

$$
\operatorname* { i n f } _ { \mathcal { T } _ { n } } \operatorname* { s u p } _ { P ^ { * } \in \mathcal { G } } \mathbb { E } _ { \mathbb { P } _ { n } } \left[ W _ { 1 } ( \mathcal { T } _ { n } ( P _ { n } ) , P ^ { * } ) \right] \leq \mathcal { R } _ { 1 , n } ( \alpha , \rho ) .
$$

To establish the tight rate, it remains to prove that the population minimax risk is controlled by the empirica minimax risk, i.e., $\mathcal { R } _ { 1 , \infty } \left( \frac { \alpha } { 5 } , \rho \right) \lesssim \mathcal { R } _ { 1 , n } ( \alpha , \rho )$ . To achieve this, it sufices to construct a population-level estimator $\bar { \tau }$ that satisfies:

$$
\operatorname* { s u p } _ { P ^ { \ast } \in \mathcal { G } } \operatorname* { s u p } _ { \widetilde { P } \in \mathcal { C } _ { P ^ { \ast } } ^ { \mathrm { F E L P } } \left( \frac { \alpha } { 5 } , \rho \right) } W _ { 1 } \left( \bar { T } ( \widetilde { P } ) , P ^ { \ast } \right) \lesssim \mathcal { R } _ { 1 , n } ( \alpha , \rho ) .
$$

Without loss of generality, we assume that the infimum in the definition of $\mathcal { R } _ { 1 , n } ( \alpha , \rho )$ is achieved by an empirical estimator $\bar { \mathcal T } _ { n }$ (otherwise, we can proceed with a standard approximation argument and take the limit):

$$
\operatorname* { s u p } _ { P ^ { * } \in \mathcal { G } _ { \widetilde { \mathbb { P } } _ { n } } \in \mathcal { D } _ { n } ^ { \mathrm { F E L P } } ( P ^ { * } , \alpha ) } \mathbb { E } _ { \widetilde { \mathbb { P } } _ { n } } \left[ W _ { 1 } \left( \bar { \mathcal { T } } _ { n } ( \widetilde { P } _ { n } ) , P ^ { * } \right) \right] \leq \mathcal { R } _ { 1 , n } ( \alpha , \rho ) .
$$

For any fixed $P ^ { * } \in \mathcal G$ and any contaminated population distribution $\begin{array} { r } { \widetilde { P } \in \mathcal { C } _ { P ^ { * } } ^ { \mathrm { F E L P } } \left( \frac { \alpha } { 5 } , \rho \right) } \end{array}$ , the conditions of Lemma 5 are satisfied. Thus, there exists a joint coupling $\pi \in \Pi ( P ^ { * } , \widetilde P )$ such that the second $( \mathrm { r i g h t } )$ marginal distribution of the conditional joint measure $\pi ^ { n } ( \cdot , \cdot \mid \mathcal { E } _ { 1 } \cap \mathcal { E } _ { 2 } )$ , denoted as $\widetilde { \mathbb { P } } _ { n } ^ { \prime }$ , belongs to the empirical FELP class $\mathcal { D } _ { n } ^ { \mathrm { F E L P } } ( P ^ { * } , \alpha )$

By the construction of $\bar { \mathcal T } _ { n }$ , we can bound the probability of the estimation error under the product measure ${ \widetilde { P } } ^ { n }$ as:

$$
\begin{array} { r l r } {  { \widetilde { P } ^ { n } \{ W _ { 1 } ( \mathcal { \hat { T } } _ { n } ( \widetilde { P } _ { n } ) , P ^ { * } ) \le 4 \mathcal { R } _ { 1 , n } ( \alpha , \rho ) \} \sum \pi ^ { n } ( \mathcal { E } _ { 1 } \cap \mathcal { E } _ { 2 } ) \cdot \widetilde { \mathbb { P } } _ { n } ^ { \prime } \{ W _ { 1 } ( \mathcal { \hat { T } } _ { n } ( \widetilde { P } _ { n } ) , P ^ { * } ) \le 4 \mathcal { R } _ { 1 , n } ( \alpha , \rho ) \} } } \\ & { } & { \ge \frac { 3 } { 4 } ( 1 - \widetilde { \mathbb { P } } _ { n } ^ { \prime } \{ W _ { 1 } ( \mathcal { \hat { T } } _ { n } ( \widetilde { P } _ { n } ) , P ^ { * } ) > 4 \mathcal { R } _ { 1 , n } ( \alpha , \rho ) \} ) } \\ & { } & { \ge \frac { 3 } { 4 } ( 1 - \frac { \mathbb { E } _ { \widetilde { \mathbb { P } } _ { n } ^ { \prime } } [ W _ { 1 } ( \mathcal { \hat { T } } _ { n } ( \widetilde { P } _ { n } ) , P ^ { * } ) ] } { 4 \mathcal { R } _ { 1 , n } ( \alpha , \rho ) } ) } \\ & { } & { \ge \frac { 3 } { 4 } ( 1 - \frac { 1 } { 4 } ) = \frac { 9 } { 1 6 } > \frac { 1 } { 2 } , } \end{array}
$$

where the second inequality follows from Lemma $5 ,$ the third from Markov’s inequality, and the fourth from the definition of $\bar { \mathcal T } _ { n }$ and the fact that $\widetilde { \mathbb { P } } _ { n } ^ { \prime } \in \mathcal { D } _ { n } ^ { \mathrm { F E L P } } ( P ^ { * } , \alpha )$

Now, we define the following candidate set of clean distributions:

$$
\Gamma ( \widetilde { P } ) : = \left\{ \nu \in \mathcal { G } : \widetilde { P } ^ { n } \left\{ W _ { 1 } \left( \bar { \mathcal { T } } _ { n } ( \widetilde { P } _ { n } ) , \nu \right) \leq 4 \mathcal { R } _ { 1 , n } ( \alpha , \rho ) \right\} > \frac { 1 } { 2 } \right\} .
$$

The preceding inequality guarantees that $P ^ { * } \in \Gamma ( \widetilde { P } )$ , meaning $\Gamma (  { \widetilde { P } } )$ is non-empty. For the given contami nated population ${ \cal \tilde { P } } ,$ we define our population-level estimator $\bar { \mathcal { T } } ( \widetilde { P } )$ by choosing an arbitrary element from $\Gamma (  { \widetilde { P } } )$ . By definition of $\Gamma (  { \widetilde { P } } )$ , we have:

$$
\widetilde { P } ^ { n } \left\{ W _ { 1 } \left( \hat { \mathcal { T } } _ { n } ( \widetilde { P } _ { n } ) , \hat { \mathcal { T } } ( \widetilde { P } ) \right) \le 4 \mathcal { R } _ { 1 , n } ( \alpha , \rho ) \right\} > \frac { 1 } { 2 } ,
$$

and we already established that:

$$
\widetilde { P } ^ { n } \left\{ W _ { 1 } \left( \mathcal { \bar { T } } _ { n } ( \widetilde { P } _ { n } ) , P ^ { * } \right) \leq 4 \mathcal { R } _ { 1 , n } ( \alpha , \rho ) \right\} > \frac { 1 } { 2 } .
$$

Since both events hold with probability strictly greater than $1 / 2$ under ${ \widetilde { P } } ^ { n }$ , they must have a non-empty intersection. On this intersection, the triangle inequality yields:

$$
W _ { 1 } \left( P ^ { * } , \hat { T } ( \tilde { P } ) \right) \leq W _ { 1 } \left( P ^ { * } , \hat { T } _ { n } ( \tilde { P } _ { n } ) \right) + W _ { 1 } \left( \hat { T } _ { n } ( \tilde { P } _ { n } ) , \hat { T } ( \tilde { P } ) \right) \leq 8 \mathcal { R } _ { 1 , n } ( \alpha , \rho ) .
$$

Since this bound holds uniformly for any $P ^ { * } \in { \mathcal { G } }$ and $\begin{array} { r } { \widetilde { P } \in \mathcal { C } _ { P ^ { * } } ^ { \mathrm { F E L P } } \left( \frac { \alpha } { 5 } , \rho \right) } \end{array}$ , we conclude that:

$$
\frac { 1 } { 8 } \mathcal { R } _ { 1 , \infty } \left( \frac { \alpha } { 5 } , \rho \right) \leq \frac { 1 } { 8 } \operatorname* { s u p } _ { P ^ { * } \in \mathcal { G } } \operatorname* { s u p } _ { \widetilde { P } \in \mathcal { C } _ { P ^ { * } } ^ { \mathrm { F E L P } } \left( \frac { \alpha } { 5 } , \rho \right) } W _ { 1 } \left( P ^ { * } , \bar { T } ( \widetilde { P } ) \right) \leq \mathcal { R } _ { 1 , n } ( \alpha , \rho ) .
$$

Finally, we apply this general result to the specialized class $\mathcal { G } = \mathcal { G } _ { \mathrm { c o v } } ( \sigma )$ . By Theorem 3.1 of Lei (2020), we have $\begin{array} { r } { \operatorname* { s u p } _ { P ^ { * } \in { \mathcal G } _ { \mathrm { c o v } } ( \sigma ) } \mathbb E [ W _ { 1 } ( P _ { n } , P ^ { * } ) ] = { \mathcal O } ( \sqrt { d } n ^ { - 1 / d } ) } \end{array}$ , while Theorem 3 of Niles-Weed and Berthet (2022) provides the corresponding lower bound inf $\mathcal { T } _ { n }$ sup $P ^ { * } \in \mathcal { G } _ { \mathrm { c o v } } ( \sigma ) \mathbb { E } [ W _ { 1 } ( \mathcal { T } _ { n } ( P _ { n } ) , P ^ { * } ) ] \ = \ \Omega ( \sqrt { d } n ^ { - 1 / d } )$ . Furthermore, Theorem 2 guarantees that $\begin{array} { r } { \mathcal { R } _ { 1 , \infty } \left( \frac { \alpha } { 5 } , \rho _ { \sigma } \right) \asymp \rho _ { \sigma } , } \end{array}$ , where the covariance-resilience parameter scales as $\rho _ { \sigma } \asymp \sigma \sqrt { d \alpha }$ (Nietert et al., 2023, Theorem 2). Combining these rates yields the matching minimax rate $\backsimeq \left( \sigma \sqrt { d \alpha } + \sqrt { d } n ^ { - 1 / d } \right)$ , which is achieved by the proposed filter-based estimator $\widehat { P } _ { \mathrm { W F } }$ □

## B.2 Convergence of the Algorithm

In this section, we analyze the convergence properties of the proposed Wasserstein Filter (WF) algorithm and provide the proof of Theorem 5.

Proof of Theorem 5. Let $\mathbf { p } = \sigma ( \mathbf { b } )$ denote the filtered probability vector. We recall the dual formulation of the discrete $W _ { 1 }$ optimal transport problem:

$$
\Psi ( { \bf p } ) : = W _ { 1 } ( P _ { n } , P _ { \bf p } ) = \operatorname* { m a x } _ { { \bf u } , { \bf v } \in \mathbb { R } ^ { n } \atop { u _ { i } + v _ { j } \leq C _ { i j } } } \langle { \bf u } , { \bf 1 } / n \rangle + \langle { \bf v } , { \bf p } \rangle .
$$

Under Condition 2, the optimal dual potential $\mathbf { v } ^ { * } ( \mathbf { p } )$ is unique. Consequently, Danskin’s theorem implies that $\Psi ( \mathbf { p } )$ is diferentiable with gradient:

$$
\nabla \Psi ( \mathbf { p } ) = \mathbf { v } ^ { * } ( \mathbf { p } ) .
$$

For the penalty term, Condition 3 ensures that $p _ { i } \neq u _ { \alpha }$ for all $i \in [ n ]$ , which guarantees diferentiability.

The gradient of the penalty with respect to p is given by:

$$
\nabla _ { \mathbf { p } } \left[ \beta \sum _ { i = 1 } ^ { n } ( p _ { i } - u _ { \alpha } ) _ { + } \right] = \beta \mathbf { m } ( \mathbf { p } ) , \quad \mathrm { w h e r e } \quad \mathbf { m } _ { i } ( \mathbf { p } ) = \mathbb { I } _ { \{ p _ { i } > u _ { \alpha } \} } .
$$

By applying the chain rule, we obtain the gradient of the objective function $F ( \mathbf { b } )$ with respect to the filtered probability vector $\sigma ( \mathbf { b } )$

$$
\nabla _ { \sigma ( \mathbf { b } ) } F ( \mathbf { b } ) = \mathbf { v } ^ { * } ( \sigma ( \mathbf { b } ) ) - \beta \mathbf { m } ( \sigma ( \mathbf { b } ) ) .
$$

Therefore, the full gradient of F with respect to the parameter vector b is:

$$
\nabla F ( { \mathbf b } ) = \frac { \partial F ( { \mathbf b } ) } { \partial \sigma ( { \mathbf b } ) } \frac { \partial \sigma ( { \mathbf b } ) } { \partial { \mathbf b } } = { \mathbf J } _ { \sigma } ( { \mathbf b } ) \left( { \mathbf v } ^ { * } ( \sigma ( { \mathbf b } ) ) - \beta { \mathbf m } ( \sigma ( { \mathbf b } ) ) \right) ,
$$

where $\mathbf { J } _ { \sigma } ( \mathbf { b } ) = \mathrm { d i a g } ( \sigma ( \mathbf { b } ) ) - \sigma ( \mathbf { b } ) \sigma ( \mathbf { b } ) ^ { \top }$ is the Jacobian matrix of the softmax function $\sigma ( \mathbf { b } )$

To analyze the Lipschitz properties of this gradient, we define the following shorthand notation:

$$
\mathbf { A } _ { \sigma } ( \mathbf { b } ) : = \mathbf { v } ^ { * } ( \sigma ( \mathbf { b } ) ) - \beta \mathbf { m } ( \sigma ( \mathbf { b } ) ) ,
$$

and introduce the following supremum and Lipschitz constants over the compact set $\mathbb { K } :$

$$
\begin{array} { r } { L _ { \sigma } : = \mathrm { L i p } _ { \mathbb { K } } ( \sigma ) , \quad L _ { \mathbf { J } } : = \mathrm { L i p } _ { \mathbb { K } } ( \mathbf { J } _ { \sigma } ) , \quad M _ { \mathbf { J } } : = \operatorname* { s u p } _ { \mathbf { b } \in \mathbb { K } } \| \mathbf { J } _ { \sigma } ( \mathbf { b } ) \| , \quad \mathrm { a n d } \quad M _ { \mathbf { A } } : = \operatorname* { s u p } _ { \mathbf { b } \in \mathbb { K } } \| \mathbf { A } _ { \sigma } ( \mathbf { b } ) \| . } \end{array}
$$

Since the softmax function σ is smooth, both σ and its Jacobian $\mathbf { J } _ { \sigma }$ are Lipschitz continuous and bounded on the compact set K. Furthermore, by Conditions 3 and 4, the vector $\mathbf { A } _ { \sigma } ( \mathbf { b } )$ is bounded, ensuring that $M _ { \mathbf { J } } , M _ { \mathbf { A } } , L _ { \sigma }$ , and $L _ { \mathbf { J } }$ are all finite.

Now, for any b, $\mathbf { c } \in \mathbb { K }$ with corresponding filtered vectors $\mathbf { p } = \sigma ( \mathbf { b } )$ and $\mathbf { q } = \sigma ( \mathbf { c } )$ , we bound the gradient diference as:

$$
\begin{array} { r l } { \| \nabla F ( { \mathbf b } ) - \nabla F ( { \mathbf c } ) \| = \| [ \mathbf { J } _ { \sigma } ( { \mathbf b } ) - \mathbf { J } _ { \sigma } ( { \mathbf c } ) ] \mathbf { A } _ { \sigma } ( { \mathbf b } ) + \mathbf { J } _ { \sigma } ( { \mathbf c } ) [ \mathbf { A } _ { \sigma } ( { \mathbf b } ) - \mathbf { A } _ { \sigma } ( { \mathbf c } ) ] \| } & { } \\ { \le \| [ \mathbf { J } _ { \sigma } ( { \mathbf b } ) - \mathbf { J } _ { \sigma } ( { \mathbf c } ) ] \mathbf { A } _ { \sigma } ( { \mathbf b } ) \| + \| \mathbf { J } _ { \sigma } ( { \mathbf c } ) [ { \mathbf v } ^ { * } ( { \mathbf p } ) - { \mathbf v } ^ { * } ( { \mathbf q } ) ] \| } & { } \\ { \le L _ { \mathbf { J } } M _ { \mathbf { A } } \| { \mathbf b } - { \mathbf c } \| + M _ { \mathbf { J } } \| { \mathbf v } ^ { * } ( { \mathbf p } ) - { \mathbf v } ^ { * } ( { \mathbf q } ) \| } & { } \\ { \le \left( L _ { \mathbf { J } } M _ { \mathbf { A } } + M _ { \mathbf { J } } L _ { \sigma } L _ { \mathbf { v } ^ { * } } \right) \| { \mathbf b } - { \mathbf c } \| , } & { } \end{array}
$$

where the first inequality follows from Condition 3 (which guarantees that $\mathbf { m } ( \sigma ( \mathbf { b } ) ) = \mathbf { m } _ { 0 }$ remains locally constant on $\mathbb { K } )$ , and the third inequality follows from the $L _ { { \bf v } ^ { * } } { \mathrm { - L i p s c h i t z } }$ continuity of the optimal dua potential $\mathbf { v } ^ { * }$ assumed in Condition 4.

Thus, the gradient $\nabla F ( \mathbf { b } )$ is $L _ { F ^ { - } } \mathrm { L }$ ipschitz continuous on K with Lipschitz constant $L _ { F } : = L _ { \bf J } M _ { \bf A } + $ $M _ { \mathbf { J } } L _ { \sigma } L _ { \mathbf { v } ^ { * } }$ . By the standard descent (ascent) lemma, this Lipschitz property implies:

$$
\begin{array} { l } { { \displaystyle F ( { \bf b } _ { r + 1 } ) \geq F ( { \bf b } _ { r } ) + \langle \nabla F ( { \bf b } _ { r } ) , { \bf b } _ { r + 1 } - { \bf b } _ { r } \rangle - \frac { L _ { F } } { 2 } \| { \bf b } _ { r + 1 } - { \bf b } _ { r } \| ^ { 2 } } } \\ { { \displaystyle \quad \quad = F ( { \bf b } _ { r } ) + \left( \gamma - \frac { L _ { F } \gamma ^ { 2 } } { 2 } \right) \| \nabla F ( { \bf b } _ { r } ) \| ^ { 2 } } . } \end{array}
$$

Since the step size satisfies $\begin{array} { r } { 0 < \gamma < \frac { 2 } { L _ { F } } } \end{array}$ , the coeficient $\begin{array} { r } { \gamma - \frac { L _ { F } \gamma ^ { 2 } } { 2 } } \end{array}$ is strictly positive, guaranteeing that the objective values $\{ F ( \mathbf { b } _ { r } ) \}$ are non-decreasing. Summing this inequality from iteration $r = r _ { 0 }$ to $r = r _ { 0 } + R - 1$

yields:

$$
\begin{array} { r l r } {  { \sum _ { r = r _ { 0 } } ^ { r _ { 0 } + R - 1 } \| \nabla F ( { \mathbf b } _ { r } ) \| ^ { 2 } \leq \frac { F ( { \mathbf b } _ { r _ { 0 } + R } ) - F ( { \mathbf b } _ { r _ { 0 } } ) } { \gamma - \frac { L _ { F } \gamma ^ { 2 } } { 2 } } } } \\ & { } & { \leq \frac { \operatorname* { m a x } _ { i , j \in [ n ] } \| { \mathbf x } _ { i } - { \mathbf x } _ { j } \| - F ( { \mathbf b } _ { r _ { 0 } } ) } { \gamma - \frac { L _ { F } \gamma ^ { 2 } } { 2 } } , } \end{array}
$$

where we used the fact that the maximum possible value of the $W _ { 1 }$ distance (and thus F(b)) is bounded by the diameter of the empirical support. It follows immediately that:

$$
\operatorname* { m i n } _ { r \in \{ r _ { 0 } , \ldots , r _ { 0 } + R - 1 \} } \| \nabla F ( \mathbf { b } _ { r } ) \| \leq \sqrt { \frac { \operatorname* { m a x } _ { i , j \in [ n ] } \| \mathbf { x } _ { i } - \mathbf { x } _ { j } \| - F ( \mathbf { b } _ { r _ { 0 } } ) } { \left( \gamma - \frac { L _ { F } \gamma ^ { 2 } } { 2 } \right) R } } ,
$$

which completes the proof.

## C Additional Numerical Experiment Results

## C.1 Two-Dimensional Synthetic Data

Figure A.3 shows the two-moon and circle data examples.

Table 4: Results of SinkWF and SlicedWF on 2-D data using projection implementation. Weak and strong contamination implies the signal of corruption strength. The mean of 50 times repetition are recorded and the corresponding standard error is recorded in parenthesis right side.
<table><tr><td rowspan="2"></td><td colspan="4">Weak Contamination</td></tr><tr><td></td><td colspan="2">AUC-ROC α = 0.05</td><td colspan="2">AUC-PR α = 0.15</td></tr><tr><td rowspan="6">Twomoon data</td><td>SlicedWF SinkWF</td><td>0.51(0.04)</td><td>α = 0.15 0.50(0.11)</td><td>α = 0.05 0.06(0.02) 0.18(0.05) 0.18(0.15)</td></tr><tr><td></td><td>0.67(0.12)</td><td>0.90(0.07) Strong Contamination</td><td>0.73(0.16)</td></tr><tr><td></td><td>AUC-ROC</td><td></td><td>AUC-PR α = 0.15</td></tr><tr><td>SlicedWF</td><td>α = 0.05</td><td>α = 0.15</td><td>α = 0.05 0.40(0.45)</td></tr><tr><td>SinkWF</td><td>0.67(0.25)</td><td>0.60(0.27)</td><td>0.42(0.39)</td></tr><tr><td></td><td>0.69(0.05)</td><td>0.79(0.04) 0.24(0.11)</td><td>0.49(0.11)</td></tr><tr><td rowspan="8">Circles data</td><td></td><td>Weak Contamination</td><td></td><td></td></tr><tr><td></td><td>AUC-ROC</td><td></td><td>AUC-PR α = 0.15</td></tr><tr><td rowspan="5">SlicedWF SinkWF</td><td>α = 0.05</td><td>α = 0.15</td><td>α = 0.05</td></tr><tr><td>0.53(0.05)</td><td>0.52(0.11)</td><td>0.07(0.03) 0.19(0.05)</td></tr><tr><td>0.63(0.06)</td><td>0.72(0.03)</td><td>0.14(0.06) 0.35(0.05)</td></tr><tr><td></td><td>Strong Contamination</td><td></td></tr><tr><td>AUC-ROC</td><td></td><td>AUC-PR</td></tr><tr><td></td><td>α = 0.05</td><td>α = 0.15</td><td>α = 0.05</td><td>α = 0.15</td></tr><tr><td>SlicedWF</td><td></td><td></td><td></td><td>0.38(0.34)</td></tr><tr><td></td><td>0.64(0.21)</td><td>0.58(0.25)</td><td>0.31(0.34)</td><td></td></tr><tr><td>SinkWF</td><td>0.69(0.04)</td><td>0.78(0.06)</td><td>0.22(0.09)</td><td>0.45(0.13)</td></tr></table>

![](images/679939a4773ce7126574e7ecaa1b805d3ba529a14fd0787a29752578238d455c.jpg)

![](images/ad68fcf56f4bb48fcadad0463b2831e1618b13684f3230ca0757fcae64897f2d.jpg)

![](images/5b61f91f82f1d77a99b5aa545c3350f084eb70997f9f263d3d5fe3e22811d78a.jpg)

![](images/217fbf8f9433f33eb44f9e4bcac2f834a67297a60fbfa7600cf6a5ae15887d3f.jpg)

![](images/2b63288d9e15c2ee35844b76dfb9d8a366dde8c60703930bfe840fe2d7e92bfc.jpg)

![](images/a48084acf1ef8574e1d3de51b478c0f1e3bf92f1f5a48233b93e9c8ea74107a3.jpg)

![](images/7f1cd5d4634acbaee68ee349ced7660a66e473960a7a07efd26229f599578592.jpg)

![](images/4cc791a59a32ffa18ba93ace330eb95c1d36b05550612055b981e10c9e68019e.jpg)

![](images/2b9a5e112e5fd5024e326cd464395fd700495a72cd3c22037e264a255967fa2a.jpg)

![](images/4b24290f2e8f1209ee3c0a9958836b10ebebc284ecee11d5abd858b34137942d.jpg)

![](images/a14b941ff09f45e3b1e50fd6871d91c078c5c7eb948bb627050508bae08b3995.jpg)

![](images/cd321a9fefd11a85f5253d510f6a7cb826f3816cd293516aab97b0a3e43e7b81.jpg)

![](images/5b94858401f4bfe51b752004766dd865500b24c92562de705778f8873c74b490.jpg)

![](images/e14800ae66f6f235864deee12bd5448a02c86a7ecaf4e48d1719dd38b8677204.jpg)

![](images/06db006dacde728e881c20ac3326f7aee5f3eaed92f0d2c2712d9e93e0444805.jpg)

![](images/65fb3360484536a5dc778f2d2bcc362388edbe03e44089dddeacf710abc1542d.jpg)

![](images/4af8d71461f46d8c171d8c0a93caf2cb401887c569f018c2e821febe1c206164.jpg)

![](images/3f53dbd5a434f84ec64c28d592f246bb03677e87dbf96f53f6e049dfe6664b93.jpg)

![](images/478ac95beb97736f903e98ef7d0a878830408bd9cd9b4b8c829137eb9d9bc4e5.jpg)

![](images/5887c4852d3c54aec8c552d65cf8a49bb0d3928c7596d980b79dea57c3b43ebe.jpg)

![](images/97e75ab06646b6f726bacc516cc2d8caa13b4a906512e302786592089ab36004.jpg)

![](images/673bd475c6fffa92b6ff543fba11211ee398c74f0a7ffa171a8d9c61d174e0be.jpg)  
(a) $\alpha = 0 . 1 5$ with weak contamination, corrupted samples are gathered around (−0.5, −0.5).

![](images/ba887813e564fea2d7be9064133c4c7b2083544539b7d213cb29bcd94abb9297.jpg)

![](images/7ac9ac590d31d7c2b9424b07caab015bdd3855a59292632517d127ad192954db.jpg)  
(b) $\alpha = 0 . 1 5$ with strong contamination, corrupted samples are gathered around (−2.5, −2.5).  
Figure A.3: The two-moon and circle data examples. Green and red points are normal and corrupted samples classified by algorithm.

Table 5: Hyperparameters setting in 2-D data example. (p) means handle constraint with projection implementation while (r) means regularization implementation. In MDE, δ is the threshold for maximum eigenvalue of covariance matrix, µ is the factor that $( \mu \alpha ) n$ samples with largest scores for random filtering. Parameters notation in DIF are consistent with its corresponding paper. COPOD is tuning free.
<table><tr><td>Algorithms</td><td>Parameter Search Scope</td></tr><tr><td>DIF</td><td> $r \in \{ 1 0 , 5 0 \} , t \in \{ 6 , 1 0 \} , n \in \{ 1 2 8 , 2 5 6 \}$ </td></tr><tr><td>MDE</td><td> $\delta \in \{ 0 . 1 , 0 . 4 , 0 . 7 \} , \mu \in \{ 2 , 3 , 4 \}$ </td></tr><tr><td>SlicedWF (p)</td><td> $R \in \{ 1 0 0 , 3 0 0 \} , \gamma \in \{ 1 , 1 0 ^ { - 1 } , 1 0 ^ { - 2 } , 1 0 ^ { - 3 } \} , S = 3 0 0 0$ </td></tr><tr><td>SlicedWF (r)</td><td> $R \in \{ 1 0 0 , 3 0 0 \} , \gamma \in \{ 1 , 1 0 ^ { - 1 } , 1 0 ^ { - 2 } , 1 0 ^ { - 3 } \} , S = 3 0 0 0 , \beta \in \{ 1 , 1 0 , 1 0 0 \}$ </td></tr><tr><td>SinkWF (p)</td><td> $R = 1 0 0 , \lambda \in \{ 1 0 ^ { - 1 } , 1 0 ^ { - 2 } , 1 0 ^ { - 3 } \} , \gamma \in \{ 1 0 ^ { - 1 } , 1 0 ^ { - 2 } , 1 0 ^ { - 3 } , 1 0 ^ { - 4 } \}$ </td></tr><tr><td>SinkWF (r) SinkMarg</td><td> $R = 1 0 0 , \lambda \in \{ 1 0 ^ { - 1 } , 1 0 ^ { - 2 } , 1 0 ^ { - 3 } \} , \gamma \in \{ 1 0 ^ { - 1 } , 1 0 ^ { - 2 } , 1 0 ^ { - 3 } , 1 0 ^ { - 4 } \} , \beta \in \{ 1 , 1 0 , 1 0 0 \}$   $\lambda \in \{ 1 0 ^ { - 1 } , 1 0 ^ { - 2 } , 1 0 ^ { - 3 } \}$ </td></tr></table>

## C.2 Benchmark Datasets

Table 6 below give the full performance results of the methods considered in this work for the graph benchmark datasets.

Table 6: Full results of graph benchmark datasets.
<table><tr><td colspan="3"></td><td colspan="6">AUC-ROC</td></tr><tr><td rowspan="10">Graph</td><td>Dataset</td><td>(n, d, α)</td><td>COPOD</td><td>DIF</td><td>MDE</td><td>SlicedWF</td><td>SinkWF</td><td>SinkMarg</td></tr><tr><td>NR-AR NR-ALBD</td><td>(7823, 300, 3.94%)</td><td>0.71</td><td>0.71</td><td>0.50</td><td>0.56</td><td>0.76</td><td>0.56</td></tr><tr><td>NR-AhR</td><td>(7823, 300, 3.03%) (7823, 300, 9.82%)</td><td>0.75 0.39</td><td>0.75 0.44</td><td>0.50 0.50</td><td>0.51 0.51</td><td>0.77 0.51</td><td>0.54 0.48</td></tr><tr><td>NR-Aro</td><td>(7823, 300, 3.83%)</td><td>0.62</td><td>0.65</td><td>0.50</td><td>0.53</td><td>0.65</td><td>0.58</td></tr><tr><td>NR-ER</td><td>(7823, 300, 10.11%)</td><td>0.47</td><td>0.51</td><td>0.50</td><td>0.56</td><td>0.57</td><td>0.47</td></tr><tr><td>NR-ELBD</td><td>(7823, 300, 4.46%)</td><td>0.55</td><td>0.60</td><td>0.50</td><td>0.56</td><td>0.62</td><td>0.48</td></tr><tr><td>NR-gamma</td><td>(7823, 300, 2.38%)</td><td>0.54</td><td>0.56</td><td>0.50</td><td>0.52</td><td>0.56</td><td>0.58</td></tr><tr><td>SR-ARE</td><td>(7823, 300, 12.04%)</td><td>0.53</td><td>0.57</td><td>0.50</td><td>0.54</td><td>0.56</td><td>0.54</td></tr><tr><td>SR-ATAD5</td><td>(7823, 300, 3.37%)</td><td>0.44</td><td>0.50</td><td>0.50</td><td>0.52</td><td>0.53</td><td>0.52</td></tr><tr><td>SR-HSE</td><td>(7823, 300, 4.76%)</td><td>0.54</td><td>0.57</td><td>0.50</td><td>0.50</td><td>0.55</td><td>0.57</td></tr><tr><td>SR-MMP</td><td>(7823, 300, 11.73%)</td><td>0.51</td><td>0.57</td><td>0.50</td><td>0.59</td><td>0.60</td><td>0.49</td></tr><tr><td>SR-p53</td><td>(7823, 300, 5.41%)</td><td>0.57</td><td>0.61</td><td>0.50</td><td>0.55</td><td>0.61</td><td></td></tr><tr><td>mean</td><td></td><td>0.55</td><td>0.59</td><td>0.50</td><td>0.54</td><td>0.61</td><td>0.62 0.54</td></tr><tr><td colspan="2">wins</td><td>0.00% 25.00%</td><td></td><td>0.00%</td><td>8.33%</td><td>66.67%</td><td>25.00%</td></tr><tr><td rowspan="13">Graph</td><td>Dataset</td><td>(n, d, α)</td><td></td><td></td><td></td><td>AUC-PR</td><td></td><td></td></tr><tr><td>NR-AR</td><td>(7823, 300, 3.94%)</td><td>COPOD 0.09</td><td>DIF 0.09</td><td>MDE</td><td>SlicedWF 0.05</td><td>SinkWF 0.16</td><td>SinkMarg</td></tr><tr><td>NR-ALBD</td><td>(7823, 300, 3.03%)</td><td>0.08</td><td>0.07</td><td>0.04 0.03</td><td>0.03</td><td>0.13</td><td>0.05</td></tr><tr><td>NR-AhR</td><td>(7823, 300, 9.82%)</td><td>0.07</td><td>0.08</td><td></td><td>0.10</td><td>0.10</td><td>0.03</td></tr><tr><td>NR-Aro</td><td>(7823, 300, 3.83%)</td><td></td><td></td><td>0.10</td><td></td><td></td><td>0.09</td></tr><tr><td>NR-ER</td><td>(7823, 300, 10.11%)</td><td>0.06</td><td>0.06</td><td>0.04</td><td>0.04</td><td>0.06</td><td>0.05</td></tr><tr><td>NR-ELBD</td><td>(7823, 300, 4.46%)</td><td>0.10</td><td>0.11</td><td>0.10</td><td>0.11</td><td>0.13</td><td>0.09</td></tr><tr><td>NR-gamma</td><td></td><td>0.06</td><td>0.06</td><td>0.04</td><td>0.05</td><td>0.08</td><td>0.05</td></tr><tr><td>SR-ARE</td><td>(7823, 300, 2.38%)</td><td>0.03</td><td>0.03</td><td>0.02</td><td>0.02</td><td>0.03</td><td>0.03</td></tr><tr><td></td><td>(7823, 300, 12.04%)</td><td>0.13</td><td>0.13</td><td>0.12</td><td>0.13</td><td>0.14</td><td>0.14</td></tr><tr><td>SR-ATAD5</td><td>(7823, 300, 3.37%)</td><td>0.03</td><td>0.03</td><td>0.03</td><td>0.04</td><td>0.04</td><td>0.04</td></tr><tr><td>SR-HSE</td><td>(7823, 300, 4.76%)</td><td>0.05</td><td>0.05</td><td>0.05</td><td>0.05</td><td>0.05</td><td>0.06</td></tr><tr><td>SR-MMP</td><td>(7823, 300, 11.73%)</td><td>0.13</td><td>0.14</td><td>0.12</td><td>0.15</td><td>0.16</td><td>0.12</td></tr><tr><td>SR-p53</td><td>(7823, 300, 5.41%)</td><td>0.08</td><td>0.08</td><td>0.05</td><td>0.07</td><td>0.07</td><td>0.09</td></tr><tr><td colspan="3">mean wins</td><td>0.08 16.67%</td><td>0.08 16.67%</td><td>0.06 8.33%</td><td>0.07 16.67%</td><td>0.10 83.33%</td><td>0.07 41.67%</td></tr></table>

Table 7: Full results of tabular benchmark datasets with projection implementation and adaptive α on penalty implementation.
<table><tr><td colspan="2"></td><td rowspan="2"></td><td colspan="3">AUC-ROC</td></tr><tr><td>Dataset vertebral</td><td>(n, d, α)</td><td>SlicedWF</td><td>SinkWF</td><td>SinkAda</td></tr><tr><td rowspan="10">Tabular</td><td>yeast</td><td>(240, 6, 12.50%)</td><td>0.67</td><td>0.58</td><td>0.70</td></tr><tr><td></td><td>(1484, 8, 34.16%)</td><td>0.62</td><td>0.52</td><td>0.60</td></tr><tr><td>annthyroid</td><td>(7200, 6, 7.42%)</td><td>0.60</td><td>0.55</td><td>0.59</td></tr><tr><td>breastw</td><td>(683, 9, 34.99%)</td><td>0.99</td><td>0.97</td><td>0.75</td></tr><tr><td>cardio tocography</td><td>(1831, 21, 9.61%)</td><td>0.60</td><td>0.62</td><td>0.56</td></tr><tr><td></td><td>(2114, 21, 22.04%)</td><td>0.86</td><td>0.60</td><td>0.78</td></tr><tr><td>Hepatitis</td><td>(80, 19, 16.25%)</td><td>0.57</td><td>0.76</td><td>0.65</td></tr><tr><td>Lymph</td><td>(148, 18, 4.05%)</td><td>0.48</td><td>0.83</td><td>0.65</td></tr><tr><td>shuttle</td><td>(49097, 9, 7.15%)</td><td>0.97</td><td>0.61</td><td>0.87</td></tr><tr><td>pendigits satellite</td><td>(6870, 16, 2.27%)</td><td>0.72</td><td>0.51</td><td>0.85</td></tr><tr><td></td><td>WBC</td><td>(6435, 36, 31.64%) (223, 9, 4.48%)</td><td>0.52 0.77</td><td>0.55 0.89</td><td>0.69</td></tr><tr><td>smtp</td><td></td><td>0.50</td><td>0.50</td><td></td><td>0.87</td></tr><tr><td>optdigits</td><td>(95156, 3, 0.03%)</td><td></td><td></td><td></td><td>0.64</td></tr><tr><td>mean</td><td>(5216, 64, 2.88%)</td><td>0.67</td><td>0.53</td><td></td><td>0.73</td></tr><tr><td colspan="2"></td><td></td><td>0.68</td><td>0.64</td><td>0.71</td></tr><tr><td rowspan="14">Tabular</td><td rowspan="4">Dataset vertebral yeast</td><td></td><td></td><td>AUC-PR</td><td></td><td></td></tr><tr><td>(n, d, α)</td><td>SlicedWF</td><td></td><td>SinkWF</td><td>SinkAda</td></tr><tr><td>(240, 6, 12.50%)</td><td>0.18</td><td></td><td>0.19</td><td>0.25</td></tr><tr><td>(1484, 8,34.16%)</td><td>0.42</td><td>0.36</td><td></td><td>0.45</td></tr><tr><td>annthyroid breastw</td><td>(7200, 6, 7.42%)</td><td>0.14</td><td>0.09</td><td>0.11</td><td></td></tr><tr><td></td><td>(683, 9, 34.99%)</td><td>0.98</td><td>0.93</td><td></td><td>0.51</td></tr><tr><td>cardio</td><td>(1831, 21, 9.61%)</td><td>0.12</td><td>0.22</td><td></td><td>0.15</td></tr><tr><td>tocography</td><td>(2114, 21, 22.04%)</td><td>0.59</td><td>0.39</td><td></td><td>0.54</td></tr><tr><td>Hepatitis</td><td>(80, 19, 16.25%)</td><td>0.22</td><td>0.43</td><td></td><td>0.25</td></tr><tr><td>Lymph</td><td>(148, 18, 4.05%)</td><td>0.04</td><td></td><td>0.20</td><td>0.16</td></tr><tr><td>shuttle</td><td>(49097, 9, 7.15%)</td><td>0.90</td><td></td><td>0.10</td><td>0.26</td></tr><tr><td>pendigits</td><td>(6870, 16, 2.27%)</td><td>0.08</td><td>0.02</td><td></td><td>0.11</td></tr><tr><td>satellite</td><td>(6435, 36, 31.64%)</td><td>0.34</td><td>0.35</td><td></td><td>0.54</td></tr><tr><td>WBC</td><td>(223, 9, 4.48%)</td><td>0.22</td><td>0.47</td><td></td><td>0.29</td></tr><tr><td>smtp</td><td>(95156, 3, 0.03%)</td><td>0.00</td><td>0.00</td><td></td><td>0.00</td></tr><tr><td>optdigits</td><td>(5216, 64, 2.88%)</td><td></td><td>0.04</td><td>0.03</td><td>0.29</td></tr><tr><td>mean</td><td></td><td></td><td>0.31</td><td>0.27</td><td>0.28</td></tr></table>

Table 8: Full Results of graph benchmark datasets with projection implementation and adaptive α on penalty implementation.
<table><tr><td rowspan="2">Dataset NR-AR</td><td rowspan="2">(n, d, α)</td><td colspan="3">AUC-ROC</td></tr><tr><td>SlicedWF</td><td>SinkWF</td><td>SinkAda</td></tr><tr><td rowspan="9">Graph SR-HSE SR-MMP</td><td>NR-ALBD</td><td>(7823, 300, 3.94%) (7823, 300, 3.03%)</td><td>0.66 0.58</td><td>0.50 0.53</td><td>0.77 0.78</td></tr><tr><td>NR-AhR</td><td></td><td>0.58</td><td>0.51</td><td>0.56</td></tr><tr><td>NR-Aro</td><td>(7823, 300, 9.82%)</td><td></td><td></td><td>0.68</td></tr><tr><td>NR-ER NR-ELBD</td><td>(7823, 300, 3.83%) (7823, 300, 10.11%)</td><td>0.64 0.61</td><td>0.51 0.51</td><td>0.57</td></tr><tr><td>NR-gamma</td><td>(7823, 300, 4.46%)</td><td>0.61</td><td>0.53</td><td>0.64</td></tr><tr><td>SR-ARE</td><td>(7823, 300, 2.38%) (7823, 300, 12.04%)</td><td>0.52</td><td>0.50</td><td>0.59</td></tr><tr><td>SR-ATAD5</td><td></td><td>0.57</td><td>0.50</td><td>0.56</td></tr><tr><td>(7823, 300, 3.37%)</td><td>0.55</td><td></td><td>0.50</td><td>0.55</td></tr><tr><td>(7823, 300, 4.76%) (7823, 300, 11.73%)</td><td>0.51 0.54</td><td>0.52</td><td></td><td>0.54</td></tr><tr><td rowspan="9">Graph</td><td>SR-p53</td><td></td><td></td><td>0.52</td><td>0.60</td></tr><tr><td>mean</td><td>(7823, 300, 5.41%)</td><td>0.58</td><td>0.50</td><td>0.62</td></tr><tr><td></td><td></td><td>0.58</td><td>0.51</td><td>0.62</td></tr><tr><td>Dataset NR-AR</td><td>(n, d, α)</td><td>SlicedWF</td><td>AUC-PR SinkWF</td><td>SinkAda</td></tr><tr><td>NR-ALBD</td><td>(7823, 300, 3.94%) (7823, 300, 3.03%)</td><td>0.06 0.04</td><td>0.04</td><td>0.17</td></tr><tr><td>NR-AhR NR-Aro</td><td>(7823, 300, 9.82%)</td><td>0.12</td><td>0.03 0.10</td><td>0.15 0.11</td></tr><tr><td>NR-ER NR-ELBD</td><td>(7823, 300, 3.83%)</td><td>0.06</td><td>0.04</td><td>0.06</td></tr><tr><td></td><td>(7823, 300, 10.11%)</td><td>0.13</td><td>0.10</td><td>0.12</td></tr><tr><td>NR-gamma</td><td>(7823, 300, 4.46%)</td><td>0.06</td><td>0.05</td><td>0.08</td></tr><tr><td>SR-ARE</td><td>(7823, 300, 2.38%)</td><td>0.03</td><td>0.02</td><td>0.04</td></tr><tr><td></td><td>(7823, 300, 12.04%)</td><td>0.14</td><td>0.12</td><td>0.13</td></tr><tr><td>SR-ATAD5</td><td></td><td></td><td></td><td></td></tr><tr><td>SR-HSE</td><td>(7823, 300, 3.37%)</td><td>0.04</td><td>0.03</td><td>0.04</td></tr><tr><td>SR-MMP</td><td>(7823, 300, 4.76%)</td><td>0.05</td><td>0.05</td><td>0.05</td></tr><tr><td></td><td>(7823, 300, 11.73%)</td><td>0.13</td><td>0.12</td><td>0.15</td></tr><tr><td>SR-p53</td><td>(7823, 300, 5.41%)</td><td>0.06</td><td>0.05</td><td>0.08</td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td>mean</td><td></td><td>0.08</td><td>0.06</td><td>0.10</td></tr></table>

Table 9: Full results of time series benchmark datasets with projection implementation and adaptive α on penalty implementation.
<table><tr><td rowspan="2"></td><td rowspan="2">Dataset</td><td colspan="3">AUC-ROC</td></tr><tr><td>(n, d, α) SlicedWF</td><td>SinkWF 0.79</td><td>SinkAda 0.57</td></tr><tr><td>Timeseries</td><td>ECG200 FordA FordB GunPoint Power Surface1 Surface2 TLECG (1162, 320, 50.00%)</td><td>(200, 320, 33.50%) (4921, 320, 48.65%) (4446, 320, 49.15%) (200, 320, 50.00%) (1096, 320, 49.91%) (621, 320, 43.80%) (980, 320, 38.37%)</td><td>0.66 0.55 0.53 0.70 0.80 0.37 0.28</td><td>0.50 0.51 0.49 0.51 0.57 0.59 0.53 0.70 0.54 0.68 0.49 0.54</td></tr><tr><td colspan="3">Dataset ECG200 FordA FordB GunPoint Timeseries Power Surface1 Surface2 TLECG Wafer mean</td><td colspan="3">0.54 0.57 0.63 0.52 0.57 0.61 AUC-PR SlicedWF SinkWF SinkAda 0.45 0.59 0.37 0.50 0.48 0.49 0.51 0.49 0.50 0.77 0.60 0.59</td></tr></table>

Table 10: Hyperparameters setting in benchmark datasets. (p) means handle constraint with projection implementation while (r) means regularization implementation. In MDE, we take the quantile of eigenvalues of covariance matrix as the maximum eigenvalue threshold, so here δ means the quantile level, µ is the factor that (µα)n samples with largest scores for random filtering. Parameters notation in DIF are consistent with its corresponding paper. COPOD is tuning free.  
Algorithms Parameter Search Scope   
DIF $r \in \{ 1 0 , 5 0 , 9 0 \} , t \in \{ 6 , 1 0 , 1 4 \} , n \in \{ 6 4 , 1 2 8 , 2 5 6 \}$   
MDE $\delta \in \{ 0 . 5 , 0 . 6 , 0 . 7 , 0 . 8 \} , \mu \in \{ 2 , 3 , 4 , 5 \}$   
SlicedWF (p) $R = 1 0 0 , \gamma \in \{ 1 0 ^ { - 1 } , 1 0 ^ { - 3 } , 1 0 ^ { - 5 } , 1 0 ^ { - 7 } \} , S = 1 0 0 0$   
SlicedWF (r) $R = 1 0 0 , \gamma \in \{ 1 0 ^ { - 1 } , 1 0 ^ { - 3 } , 1 0 ^ { - 5 } , 1 0 ^ { - 7 } \} , S = 1 0 0 0 , \beta \in \{ 3 0 , 5 0 , 9 0 \}$   
SinkWF (p) $R = 1 0 0 , \lambda \in \{ 1 0 ^ { - 1 } , 1 0 ^ { - 2 } , 1 0 ^ { - 3 } \} , \gamma \in \{ 1 0 ^ { - 1 } , 1 0 ^ { - 3 } , 1 0 ^ { - 5 } , 1 0 ^ { - 7 } \}$   
SinkWF (r) $R = 1 0 0 , \lambda \in \{ 1 0 ^ { - 1 } , 1 0 ^ { - 2 } , 1 0 ^ { - 3 } \} , \gamma \in \{ 1 0 ^ { - 1 } , 1 0 ^ { - 3 } , 1 0 ^ { - 5 } , 1 0 ^ { - 7 } \} , \beta \in \{ 3 0 , 5 0 , 9 0 \}$   
SinkMarg $\lambda \in \{ 1 0 ^ { - 1 } , 1 0 ^ { - 2 } , 1 0 ^ { - 3 } \}$

## C.3 Robust Generative Learning

Table 11: Result of DDPM training on MNIST with projection implementation. $\mathbf b _ { 0 } = \mathbf 1$ for both SinkWF and SlicedWF, other settings are the same as penalty implementation.
<table><tr><td></td><td colspan="7">Weak contamination</td></tr><tr><td></td><td colspan="3">FID score</td><td colspan="2">AUC-ROC</td><td colspan="2">AUC-PR</td></tr><tr><td></td><td> $\alpha = 0$ </td><td> $\alpha = 0 . 0 5$ </td><td> $\alpha = 0 . 1 5$ </td><td> $\alpha = 0 . 0 5$ </td><td> $\alpha = 0 . 1 5$ </td><td> $\alpha = 0 . 0 5$ </td><td> $\overline { { \alpha = 0 . 1 5 } }$ </td></tr><tr><td>SlicedWF</td><td>12.09</td><td>7.17</td><td>7.63</td><td>1.00</td><td>0.98</td><td>0.97</td><td>0.88</td></tr><tr><td>SinkWF</td><td>13.72</td><td>22.99</td><td>23.48</td><td>0.87</td><td>0.71</td><td>0.57</td><td>0.33</td></tr><tr><td rowspan="2"></td><td colspan="7">Strong contamination</td></tr><tr><td></td><td>FID score</td><td></td><td>AUC-ROC</td><td></td><td>AUC-PR</td><td></td></tr><tr><td></td><td> $\alpha = 0$ </td><td> $\overline { { \alpha = 0 . 0 5 } }$ </td><td> $\overline { { \alpha = 0 . 1 5 } }$ </td><td> $\alpha = 0 . 0 5$ </td><td> $\overline { { \alpha = 0 . 1 5 } }$ </td><td> $\alpha = 0 . 0 5$ </td><td> $\overline { { \alpha = 0 . 1 5 } }$ </td></tr><tr><td>SlicedWF</td><td>8.94</td><td>20.77</td><td>4.83</td><td>1.00</td><td>1.00</td><td>1.00</td><td>1.00</td></tr><tr><td>SinkWF</td><td>15.77</td><td>16.74</td><td>4.20</td><td>1.00</td><td>1.00</td><td>1.00</td><td>1.00</td></tr></table>

Table 12: Hyperparameters setting for MNIST data example.
<table><tr><td rowspan="2">parameter</td><td colspan="3">Weak contamination</td></tr><tr><td>SlicedWF (p)  $( R , S , \gamma )$  SlicedWF (r)  $( R , S , \gamma , \beta )$  SinkWF (p)  $( R , \lambda , \gamma )$  SinkWF (r)  $( R , \lambda , \gamma , \beta )$ </td><td> $\alpha = 0$   $( 1 0 0 , 3 0 0 0 , 1 0 ^ { - 4 } )$   $( 1 0 0 , 3 0 0 0 , 1 0 ^ { - 2 } , 1 )$   $( 1 0 0 , 1 0 ^ { - 3 } , 1 0 ^ { - 5 } )$ </td><td> $\alpha = 0 . 0 5$   $( 1 0 0 , 3 0 0 0 , 1 0 ^ { - 4 } )$   $( 1 0 0 , 3 0 0 0 , 1 0 ^ { - 2 } , 1 )$   $( 1 0 0 , 1 0 ^ { - 3 } , 1 0 ^ { - 5 } )$ </td><td> $\alpha = 0 . 1 5$   $( 1 0 0 , 3 0 0 0 , 1 0 ^ { - 4 } )$   $( 1 0 0 , 3 0 0 0 , 1 0 ^ { - 1 } , 1 )$   $( 1 0 0 , 1 0 ^ { - 3 } , 1 0 ^ { - 3 } )$ </td></tr><tr><td>SlicedWF (p)</td><td>(λ) parameter  $\alpha = 0$   $( R , S , \gamma )$   $( 1 0 0 , 3 0 0 0 , 1 0 ^ { - 5 } )$ </td><td>Strong contamination  $\overline { { \alpha = 0 . 0 5 } }$   $( 1 0 0 , 3 0 0 0 , 1 0 ^ { - 5 } )$ </td><td> $\alpha = 0 . 1 5$   $( 1 0 0 , 3 0 0 0 , 1 0 ^ { - 5 } )$ </td></tr></table>

## References

Adam, L. and Macha, V.<sup>´</sup> (2022). Projections onto the canonical simplex with additional linear inequalities. Optimization Methods and Software 37 451–479.

Arjovsky, M., Chintala, S. and Bottou, L. (2017). Wasserstein generative adversarial networks. In International conference on machine learning. Pmlr.

Balaji, Y., Chellappa, R. and Feizi, S. (2020). Robust optimal transport with applications in generative modeling and domain adaptation. Advances in Neural Information Processing Systems 33 12934–12944.

Caffarelli, L. A. and McCann, R. J. (2010). Free boundaries in optimal transport and monge-ampere obstacle problems. Annals of mathematics 673–730.

Chao, P. and Dobriban, E. (2023). Statistical estimation under distribution shift: Wasserstein perturbations and minimax theory. arXiv preprint arXiv:2308.01853 .

Chizat, L., Peyre, G. <sup>´</sup> , Schmitzer, B. and Vialard, F.-X. (2018). Unbalanced optimal transport: Dynamic and kantorovich formulations. Journal of Functional Analysis 274 3090–3123.

Clark, K. M. and McNicholas, P. D. (2024). Finding outliers in gaussian model-based clustering. Journal of Classification 41 313–337.

Cuturi, M. (2013). Sinkhorn distances: Lightspeed computation of optimal transport. Advances in neural information processing systems 26.

Diakonikolas, I., Kamath, G., Kane, D., Li, J., Moitra, A. and Stewart, A. (2019). Robust estimators in high-dimensions without the computational intractability. SIAM Journal on Computing 48 742–864.

Donoho, D. L. and Liu, R. C. (1988). The “automatic” robustness of minimum distance functionals. The Annals of Statistics 552–586.

Feydy, J., Sejourn<sup>´</sup> e, T.<sup>´</sup> , Vialard, F.-X., Amari, S.-i., Trouve, A. and Peyre, G.<sup>´</sup> (2019). Interpolating between optimal transport and mmd using sinkhorn divergences. In The 22nd International Conference on Artificial Intelligence and Statistics.

Figalli, A. (2010). The optimal partial transport problem. Archive for rational mechanics and analysis 195 533–560.

Flamary, R., Courty, N., Gramfort, A., Alaya, M. Z., Boisbunon, A., Chambon, S., Chapel, L., Corenflos, A., Fatras, K., Fournier, N., Gautheron, L., Gayraud, N. T., Janati, H., Rakotomamonjy, A., Redko, I., Rolet, A., Schutz, A., Seguy, V., Sutherland, D. J., Tavenard, R., Tong, A. and Vayer, T. (2021). Pot: Python optimal transport. Journal of Machine Learning Research 22 1–8.

Han, S., Hu, X., Huang, H., Jiang, M. and Zhao, Y. (2022). Adbench: Anomaly detection benchmark. Advances in neural information processing systems 35 32142–32159.

Heusel, M., Ramsauer, H., Unterthiner, T., Nessler, B. and Hochreiter, S. (2017). Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems 30.

Ho, J., Jain, A. and Abbeel, P. (2020). Denoising difusion probabilistic models. Advances in neural information processing systems 33 6840–6851.

Huber, P. J. (1964). Robust estimation of a location parameter. Ann. Math. Statist. 35 73–101.

Humbert, P., Le Bars, B. and Minvielle, L. (2022). Robust kernel density estimation with median-ofmeans principle. In International Conference on Machine Learning. PMLR.

Jaeger, S., Fulle, S. and Turk, S. (2018). Mol2vec: unsupervised machine learning approach with chemical intuition. Journal of chemical information and modeling 58 27–35.

Kim, J. and Scott, C. D. (2012). Robust kernel density estimation. The Journal of Machine Learning Research 13 2529–2565.

Kingma, D. P. and Ba, J. (2014). Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980 .

Lei, J. (2020). Convergence and concentration of empirical measures under wasserstein distance in unbounded functional spaces. Bernoulli 26 767–798.

Li, Z., Zhao, Y., Botta, N., Ionescu, C. and Hu, X. (2020). Copod: copula-based outlier detection. In 2020 IEEE international conference on data mining (ICDM). IEEE.

Liero, M., Mielke, A. and Savare, G. <sup>´</sup> (2018). Optimal entropy-transport problems and a new hellinger– kantorovich distance between positive measures. Inventiones mathematicae 211 969–1117.

Liu, H. and Gao, C. (2019). Density estimation with contamination: minimax rates and theory of adaptation. Electronic Journal of Statistics 13.

Loh, P.-L. (2025). A theoretical review of modern robust statistics. Annual Review of Statistics and Its Application 12 477–496.

Luise, G., Rudi, A., Pontil, M. and Ciliberto, C. (2018). Diferential properties of sinkhorn approximation for learning with wasserstein distance. Advances in Neural Information Processing Systems 31.

Ma, Y., Liu, H., La Vecchia, D. and Lerasle, M. (2025). Inference via robust optimal transportation: theory and methods. International Statistical Review .

Mukherjee, D., Guha, A., Solomon, J. M., Sun, Y. and Yurochkin, M. (2021). Outlier-robust optimal transport. In International Conference on Machine Learning. PMLR.

Nguyen, Q. M., Nguyen, H. H., Zhou, Y. and Nguyen, L. M. (2023). On unbalanced optimal transport: Gradient methods, sparsity and approximation error. Journal of Machine Learning Research 24 1–41.

Nietert, S., Cummings, R. and Goldfeld, Z. (2023). Robust estimation under the wasserstein distance. arXiv preprint arXiv:2302.01237 .

Nietert, S., Goldfeld, Z. and Shafiee, S. (2024). Robust distribution learning with local and global adversarial corruptions. In The Thirty Seventh Annual Conference on Learning Theory. PMLR.

Niles-Weed, J. and Berthet, Q. (2022). Minimax estimation of smooth densities in wasserstein distance. The Annals of Statistics 50 1519–1540.

Peyre, G. <sup>´</sup> , Cuturi, M. et al. (2019). Computational optimal transport: With applications to data science. Foundations and Trends® in Machine Learning 11 355–607.

Pham, K., Le, K., Ho, N., Pham, T. and Bui, H. (2020). On unbalanced optimal transport: An analysis of sinkhorn algorithm. In International Conference on Machine Learning. PMLR.

Ramsundar, B., Eastman, P., Walters, P. and Pande, V. (2019). Deep learning for the life sciences: applying deep learning to genomics, microscopy, drug discovery, and more. O’Reilly Media.

Sinkhorn, R. (1964). A relationship between arbitrary positive matrices and doubly stochastic matrices. The Annals of Mathematical Statistics 35 876–879.

Steinhardt, J., Charikar, M. and Valiant, G. (2018). Resilience: A criterion for learning in the presence of arbitrary outliers. In 9th Innovations in Theoretical Computer Science Conference (ITCS 2018). Schloss Dagstuhl–Leibniz-Zentrum f¨ur Informatik.

Villani, C. et al. (2009). Optimal transport: old and new, vol. 338. Springer.

Xu, H., Pang, G., Wang, Y. and Wang, Y. (2023). Deep isolation forest for anomaly detection. IEEE Transactions on Knowledge and Data Engineering 35 12591–12604.

Zhu, B., Jiao, J. and Steinhardt, J. (2022). Generalized resilience and robust statistics. The Annals of Statistics 50 2256–2283.