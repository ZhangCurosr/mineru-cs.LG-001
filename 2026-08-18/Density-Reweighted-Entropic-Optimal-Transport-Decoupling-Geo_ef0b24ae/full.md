# Density-Reweighted Entropic Optimal Transport: Decoupling Geometry from Sampling Density

Keyi Li<sup>1,∗</sup> Yuval Kluger<sup>1,2</sup> Boris Landa<sup>1,3,∗</sup>

<sup>1</sup>Program of Applied & Computational Mathematics, Yale University <sup>2</sup>Department of Pathology, Yale University <sup>3</sup>Department of Electrical and Computer Engineering, Yale University Corresponding author. Email: keyi.y.li@yale.edu, boris.landa@yale.edu

## Abstract

Dataset alignment is a central step in data analysis across science and engineering, where the goal is to match observations between datasets. Entropic Optimal Transport (EOT) ofers a computationally tractable framework for this task by encoding cross-dataset afinities in a transport plan. However, when two datasets are sampled from geometrically similar low-dimensional structures with substantially diferent sampling densities, the EOT plan may match points by relative sampling density rather than geometric proximity, yielding geometrically misleading correspondences. To address this issue, we propose a density-reweighted EOT framework in which the influence of sampling density on the transport plan can be discounted to a desired degree, ranging from standard EOT to alignment driven purely by underlying geometry. Under suitable regularity conditions, we establish convergence of the reweighted EOT plan to a family of population-level plans whose dependence on sampling density is made explicit. Through simulations, we show that our approach recovers geometrically faithful correspondences, improving over related EOT-based frameworks when datasets exhibit substantial sampling density disparity.

Keywords: Entropic Optimal Transport, dataset alignment, sampling density, manifold learning

## 1 Introduction

Dataset alignment, the problem of establishing correspondences between observations across multiple datasets, is a fundamental task arising throughout the scientific and engineering disciplines [1–3]. In genomics, for instance, datasets collected under heterogeneous conditions, such as diferent experimental batches or measurement technologies, often exhibit similar low-dimensional structures characterizing the underlying biologica process, yet undergo systematic deformations that render direct comparison infeasible [4–6]. Establishing correspondences in a geometrically meaningful way, i.e., identifying for each data point its most similar counterparts in the other dataset with respect to their underlying geometry, is therefore essential for reliable downstream analysis [1].

Optimal Transport (OT) [7] provides a natural mathematical formalization of this task by seeking a transport plan that minimizes the total cost of moving mass between two datasets, where the cost function encodes geometric proximity between data points. However, exact OT scales cubically in the number of samples [8], limiting its applicability to large-scale datasets. Entropic Optimal Transport (EOT) [9], an entropy-regularized convex relaxation of OT, ofers a computationally tractable alternative. Formally, given two finite point clouds $\mathbf { X } = \{ \mathbf { x } _ { i } \} _ { i = } ^ { m }$ and $\mathbf { Y } = \{ \mathbf { y } _ { j } \} _ { j = 1 } ^ { n }$ , a cost matrix $\mathbf { C } \in \mathbb { R } ^ { m \times n }$ with entries $C _ { i j } = c ( \mathbf { x } _ { i } , \mathbf { y } _ { j } )$ denoting the transport cost per unit mass from $\mathbf { x } _ { i }$ to $\mathbf { y } _ { j }$ , and marginal constraints $\mathbf { a } \in \mathbb { R } _ { + } ^ { m }$ , b $\in \mathbb { R } _ { + } ^ { n }$ satisfying $\mathbf { 1 } ^ { \top } \mathbf { a } = \bar { \mathbf { 1 } } ^ { \top } \mathbf { b }$ , EOT seeks the transport plan $\mathbf { W } ^ { * }$ (also referred to as EOT plan hereafter) solving

$$
\begin{array} { r l } { \displaystyle \operatorname* { m i n } _ { { \tilde { \mathbf { W } } } \in \mathbb { R } _ { + } ^ { m \times n } } } & { \displaystyle \sum _ { i = 1 } ^ { m } \sum _ { j = 1 } ^ { n } C _ { i j } { \tilde { W } } _ { i j } + \varepsilon \sum _ { i = 1 } ^ { m } \sum _ { j = 1 } ^ { n } { \tilde { W } } _ { i j } \log { \tilde { W } } _ { i j } } \\ { \mathrm { s . t . ~ } } & { \displaystyle \sum _ { j = 1 } ^ { n } { \tilde { W } } _ { i j } = a _ { i } , \quad \displaystyle \sum _ { i = 1 } ^ { m } { \tilde { W } } _ { i j } = b _ { j } , \quad \forall i \in [ m ] , \forall j \in [ n ] , } \end{array}\tag{1}
$$

![](images/f343aac1549d7813988b5e86b9dc7ea0f9ccc965eaccd367dd3be6270344050c.jpg)  
Figure 1: Sampling density mismatch biases EOT-based dataset alignment. Two datasets—a line segment (bottom) and an arc (top)—share similar underlying geometry but difer substantially in sampling density (yellow/green: high density; blue/purple: low density). Gray arrows indicate maximumcoupling correspondences, where each point is matched to its highest-afinity counterpart in the other dataset. (a) Standard EOT routes correspondences toward high-density regions, producing geometrically misleading matchings. (b) Our approach recovers correspondences that are driven by the underlying geometry. See Supplement A.1 for the experiment details.

where $\varepsilon > 0$ is the entropic regularization parameter controlling the difuseness of the resulting transport plan. The EOT problem in (1) is strictly convex and admits a unique solution of the form

$$
W _ { i j } ^ { * } \ = \ u _ { i } K _ { i j } v _ { j } , \qquad K _ { i j } \ = \ \exp ( - C _ { i j } / \varepsilon ) , \quad \forall i \in [ m ] , \forall j \in [ n ] ,\tag{2}
$$

where $\mathbf { u } \in \mathbb { R } _ { + } ^ { m }$ and $\mathbf { v } \in \mathbb { R } _ { + } ^ { n }$ are positive scaling vectors that can be computed eficiently via the Sinkhorn algorithm [10]. The EOT plan $\mathbf { W } ^ { * }$ reflects pairwise afinities between data points across the two datasets, and thus provides a natural mechanism for establishing correspondence. Beyond its computational eficiency [11, 12], EOT enjoys favorable statistical properties [13], such as robustness to perturbations in the input datasets, making it well-suited for large-scale modern datasets. It has found successful application in domain adaptation [14, 15], generative modeling [16, 17], computer vision [18, 19], and computational biology [20], among others [8].

In dataset alignment tasks, the EOT problem (1) is typically applied with squared Euclidean distances as the cost matrix and uniform empirical marginals reflecting the respective dataset sizes. Concretely, one sets $C _ { i j } = \| \mathbf { x } _ { i } - \mathbf { y } _ { j } \| ^ { 2 } , a _ { i } = n$ for all $i \in [ m ]$ , and $b _ { j } = m$ for all $j \in [ n ]$ . However, this formulation, combined with the inherent mass-conservation requirement of EOT, can lead to biased alignment when datasets are class-imbalanced $^ { \mathrm { o r , } }$ more generally, when they exhibit sampling density mismatch [21, 22]. Specifically, when two datasets are sampled from geometrically similar low-dimensional structures (e.g., manifolds) but with substantially diferent sampling densities, the EOT plan may match points across datasets based on their relative sampling densities rather than their geometric proximity, producing correspondences that are geometrically misleading; see Figure 1(a). Such density mismatches arise naturally in practice: in genomics, for instance, cells profiled across experimental conditions, sequencing batches, or measurement technologies may exhibit artificially skewed sampling densities due to technical biases introduced by the acquisition process itself [4–6]. While Unbalanced Optimal Transport (UOT) [23] (discussed in Section 2) mitigates this limitation by allowing variation in total mass, it requires additional hyperparameter tuning and ofers little intuition on how the resulting transport plan is jointly influenced by sampling density and geometric similarity.

Our Contribution In this work, we address the challenge of obtaining geometrically coherent EOT-based alignments between datasets sampled from manifolds with similar underlying geometry but substantially diferent sampling densities. In this setting, standard EOT can produce correspondences that are strongly distorted by density variation and therefore geometrically misleading; see Figure 1(a). Our goal is to recover the alignment that would arise if both manifolds were sampled uniformly, so that correspondences are driven primarily by geometry rather than by sampling density; see Figure 1(b). More broadly, we aim to provide a principled mechanism for interpolating between purely geometry-driven alignment and the density-sensitive alignment produced by standard EOT.

A density-reweighted EOT framework for geometry-driven alignment. We show that the dependence of the EOT plan on sampling density can be mitigated through density reweighting. Specifically, while standard EOT with uniform empirical marginals amounts to scaling a Gaussian kernel to have constant row and column sums, we propose an approach that instead reweighs the Gaussian kernel by local sampling density and seeks scaling factors that satisfy suitably reweighted marginal constraints. Such density-reweighting assigns greater importance to data points in sparse regions and discounts data points in dense regions, compensating for sampling density variation. The reweighting strength can be controlled by a density discounting factor $\theta \in [ 0 , 1 ] . \ \mathrm { A t } \ \theta = 0$ , the method reduces to standard EOT with uniform empirical marginals; at $\theta = 1$ , the population limit is independent of the sampling densities, so correspondences are driven by the underlying geometry; intermediate values of θ allow a trade-of between the two regimes (see Figure 2). Algorithm 1 outlines our proposed approach, where $\mathbf { W } ^ { ( \theta ) }$ from (6) is the reweighted plan.

We prove that under suitable regularity conditions on the manifold geometry and sampling densities, $\mathbf { W } ^ { ( \theta ) }$ converges with high probability as $m , n  \infty$ , up to a global scaling constant, to a population-level EOT plan whose dependence on sampling density is explicitly governed by $\theta \in [ 0 , 1 ]$ . In particular, at $\theta = 1 , \mathbf { W } ^ { ( \theta ) }$ converges to the population-level EOT plan under uniform measures on both manifolds. This is established in two settings: when the densities are known exactly (Theorem 3.2) and when the density of each dataset is estimated using a kernel density estimator (Theorem 3.4), with explicit convergence rates. Finally, we demonstrate empirically that our approach outperforms standard EOT and UOT across multiple hyperparameter configurations in recovering geometrically faithful correspondences between two datasets with substantial sampling density disparity. In these experiments, we also discuss practical considerations and provide guidance to facilitate deployment of the proposed framework.

## 2 Related Work

Unbalanced Optimal Transport Unbalanced Optimal Transport (UOT) [23] extends EOT by relaxing the hard marginal constraints in (1) through soft penalty terms, allowing the transport plan to deviate from the prescribed marginals and thus accommodate variation in the total transported mass. A popular choice of penalty is the class of f-divergences [23, 24], among which the KL-divergence formulation is widely adopted. Formally, UOT with the KL-divergence penalty seeks a transport plan solving

$$
\begin{array} { r l } { \underset { \bar { \mathbf { W } } \in \mathbb { R } _ { + } ^ { m \times n } } { \operatorname* { m i n } } } & { \displaystyle \sum _ { i = 1 } ^ { m } \sum _ { j = 1 } ^ { n } C _ { i j } \tilde { W } _ { i j } \ + \ \varepsilon \displaystyle \sum _ { i = 1 } ^ { m } \displaystyle \sum _ { j = 1 } ^ { n } \tilde { W } _ { i j } \log \tilde { W } _ { i j } + \ \lambda _ { 1 } \mathrm { K L } \Big ( \tilde { \mathbf { W } } \mathbf { 1 } _ { n } \Big \| \mathbf { a } \Big ) \ + \ \lambda _ { 2 } \mathrm { K L } \Big ( \tilde { \mathbf { W } } ^ { \top } \mathbf { 1 } _ { m } \Big \| \mathbf { b } \Big ) \ , } \end{array}\tag{7}
$$

where $\begin{array} { r } { \mathrm { K L } ( \mathbf { p } \| \mathbf { q } ) = \sum _ { i } \left( p _ { i } \log ( p _ { i } / q _ { i } ) - p _ { i } + q _ { i } \right) } \end{array}$ is the generalized KL divergence for non-negative vectors p, q of the same dimension, and $\lambda _ { 1 } , \lambda _ { 2 } > 0$ control the degree of marginal relaxation. As $\lambda _ { 1 } , \lambda _ { 2 } \to \infty$ , the marginal constraints become increasingly strict and the UOT plan recovers the EOT plan; as $\lambda _ { 1 } , \lambda _ { 2 } \to 0$ , the marginal penalties vanish and the UOT plan reduces to the Gaussian kernel, which encodes cross-data similarities via raw Euclidean distances. UOT has been studied from multiple perspectives; we refer the reader to [23, 7, 8] for comprehensive overviews, to [25] for its statistical properties, to [24, 26] for computational algorithms, and to [20, 27] for applications in machine learning.

In the context of dataset alignment, while relaxing strict mass conservation allows the UOT plan to sufer less from dataset-specific variation in sampling density, the influence of $( \lambda _ { 1 } , \lambda _ { 2 } )$ on the resulting plan is implicit: it is unclear how the choice of $( \lambda _ { 1 } , \lambda _ { 2 } )$ controls the extent to which W<sup>˜</sup> reflects the underlying geometry versus the sampling densities, and to our knowledge this interplay has not been characterized in prior work. Our approach addresses this limitation by making the density-geometry tradeof explicit and easily interpretable via the discounting parameter θ.

Manifold Learning Our approach is conceptually motivated by Difusion Maps [28], which provides a framework for manifold learning with the ability to suppress the influence of the sampling density on the recovered geometry. A key observation underlying Difusion Maps is that the graph Laplacian constructed from a kernel $k ( x , y ) = \exp ( - \| x - y \| ^ { 2 } / h ^ { 2 } )$ depends on the sampling density of the data, mixing geometric structure with the distribution of sample points. To mitigate this, Difusion Maps introduce the following α-normalization of the kernel,

Algorithm 1 Density-reweighted EOT for Geometry-driven   
Alignment   
Input: Datasets $\mathbf { X } = \{ \mathbf { x } _ { i } \} _ { i = 1 } ^ { m } \subset \mathbb { R } ^ { D }$ and $\mathbf { Y } = \{ \mathbf { y } _ { j } \} _ { j = 1 } ^ { n } \subset \mathbb { R } ^ { D } ;$   
positive density estimates $\hat { f } _ { i }$ and ${ \hat { g } } _ { j } ;$ EOT regularization $\varepsilon > 0 ;$   
discounting factor $\theta \in [ 0 , 1 ]$   
1: Compute scaling constant.   
1 m 1   
m Pi=1 ( <sup>ˆ</sup>f<sub>i</sub>) <sup>θ</sup>   
S<sup>(θ)</sup> = 1 n 1 (3)   
n Pj=1 (ˆgj ) θ   
2: Form density-adjusted kernel.   
$M _ { i j } ^ { ( \theta ) } = \frac { \exp \left( - \| \mathbf { x } _ { i } - \mathbf { y } _ { j } \| _ { 2 } ^ { 2 } / \varepsilon \right) } { \left( \hat { f } _ { i } \right) ^ { \theta } \left( \hat { g } _ { j } \right) ^ { \theta } } , \quad \forall i \in [ m ] , j \in [ n ] .$ (4)   
3: Sinkhorn scaling. Find $\pmb { \alpha } \in \mathbb { R } _ { + } ^ { m } , \beta \in \mathbb { R } _ { + } ^ { n }$ such that   
$\frac { 1 } { n } \sum _ { j = 1 } ^ { n } \alpha _ { i } ^ { ( \theta ) } M _ { i j } ^ { ( \theta ) } \beta _ { j } ^ { ( \theta ) } = \frac { 1 } { \sqrt { S ^ { ( \theta ) } } ( \hat { f } _ { i } ) ^ { \theta } } ,$   
(5)   
$\frac { 1 } { m } \sum _ { i = 1 } ^ { m } \alpha _ { i } ^ { ( \theta ) } M _ { i j } ^ { ( \theta ) } \beta _ { j } ^ { ( \theta ) } = \frac { \sqrt { S ^ { ( \theta ) } } } { ( \hat { g } _ { j } ) ^ { \theta } } .$   
4: Recover the density-reweighted EOT plan.   
W<sup>(θ)</sup> ij = α (θ <sup>)</sup> exp −∥x<sub>i</sub> − y<sub>j</sub>∥<sup>2</sup><sub>2</sub>/ε β<sup>(θ)</sup><sub>j</sub> . (6)

![](images/e81eaebd5c3959a773b0f2d9fab87d9b5c7f26cb145c923b5034aaea65f95e49.jpg)  
Figure 2: Discounting the influence of sampling density via θ in Algorithm 1. Each panel shows the maximum-coupling correspondences for a diferent value of θ. At $\theta \ : = \ : 0$ , the method reduces to standard EOT; at $\theta = 1$ , it recovers purely geometry-based alignment.

$$
k _ { \alpha } ( x , y ) = \frac { k ( x , y ) } { d ( x ) ^ { \alpha } d ( y ) ^ { \alpha } } , \quad d ( x ) = \int _ { X } k ( x , y ) d \mu ( y ) ,\tag{8}
$$

where $d ( x )$ is a local measure of volume, and $\alpha \in [ 0 , 1 ]$ controls the degree to which sampling density is discounted. $\mathrm { A s } \ h  0 .$ , the infinitesimal generator of the Markov chain induced by $k _ { \alpha }$ converges to a diferential operator whose dependence on the sampling density is explicitly controlled by α: at $\alpha = 0$ the diferential operator retains full dependence on the sampling density, recovering the random-walk graph Laplacian; at $\alpha = 1$ , the density dependence is completely removed and the diferential operator recovers the Laplace–Beltrami operator on the manifold, independently of the distribution of sample points. We extend this idea to the EOT framework by incorporating an analogous density discounting via a parameter $\theta \in [ 0 , 1 ]$ (Algorithm 1). At $\theta = 0$ , the standard EOT plan is recovered; at $\theta = 1$ , the resulting transport plan is invariant to the sampling measure, with correspondences governed purely by geometry; and intermediate values of θ provide a principled balance between the two regimes.

Our work is also closely related to EOT Eigenmaps [29], which established a connection between the EOT plan and manifold learning. Specifically, [29] showed that the singular vectors of the EOT plan can be used to jointly embed two datasets sampled from a shared low-dimensional manifold, possibly subject to dataset-specific deformations and corruptions. A central result is that, when viewed as a linear operator, the EOT plan converges in the large-sample limit to a kernel operator whose singular vectors encode the intrinsic geometry of the underlying manifold. This analysis assumes that the shared manifold is sampled according to the same density in both datasets. In contrast, our work addresses the setting in which two datasets have similar underlying geometry while exhibiting diferent sampling densities. Therefore, our proposed reweighting can enable joint manifold learning and embedding methods that are robust to dataset-specific variation in sampling density.

## 3 Density-Reweighted EOT and Theoretical Guarantees

Let $\mathbf { X } ~ = ~ \{ \mathbf { x } _ { i } \} _ { i = 1 } ^ { m } ~ \subset ~ \mathbb { R } ^ { D }$ and $\mathbf { Y } ~ = ~ \{ \mathbf { y } _ { j } \} _ { i = 1 } ^ { n } ~ \subset ~ \mathbb { R } ^ { D }$ be two point clouds sampled i.i.d. from probability measures µ dω and $\nu d \eta$ , respectively, where dω and $d \eta$ denote the volume forms on Riemannian manifolds $\mathcal { M } _ { x } , \mathcal { M } _ { y } \subset \mathbb { R } ^ { D }$ with intrinsic dimensions $d _ { 1 } , d _ { 2 }$ , and $\mu ,$ ν are the corresponding sampling density functions. We begin with the following proposition, which motivates Algorithm 1 by connecting it to a suitably modified variant of the optimization problem (1).

Proposition 3.1. The reweighted EOT plan $\mathbf { W } ^ { ( \theta ) }$ in (6) is the unique minimizer of the following optimization problem:

$$
\begin{array} { r l } & { \displaystyle \underset { \mathbf { w } \in \mathbb { R } _ { + } ^ { m } \times n } { \operatorname* { m i n } } \sum _ { i } ^ { m } \displaystyle \sum _ { j } ^ { n } \frac { 1 } { ( \hat { f } _ { i } ) ^ { \theta } ( \hat { g } _ { j } ) ^ { \theta } } \cdot \| \mathbf { x } _ { i } - \mathbf { y } _ { j } \| _ { 2 } ^ { 2 } \cdot \tilde { W } _ { i j } + \varepsilon \displaystyle \sum _ { i } ^ { m } \sum _ { j } ^ { n } \frac { 1 } { ( \hat { f } _ { i } ) ^ { \theta } ( \hat { g } _ { j } ) ^ { \theta } } \cdot \tilde { W } _ { i j } \log \tilde { W } _ { i j } } \\ & { \quad s . t . \quad \displaystyle \frac { 1 } { n } \sum _ { j = 1 } ^ { n } \frac { \tilde { W } _ { i j } } { ( \hat { f } _ { i } ) ^ { \theta } ( \hat { g } _ { j } ) ^ { \theta } } = \frac { 1 } { \sqrt { S ( ^ { \theta } ) } ( \hat { f } _ { i } ) ^ { \theta } } , \quad \displaystyle \frac { 1 } { m } \sum _ { i = 1 } ^ { m } \frac { \tilde { W } _ { i j } } { ( \hat { f } _ { i } ) ^ { \theta } ( \hat { g } _ { j } ) ^ { \theta } } = \frac { \sqrt { S ^ { ( \theta ) } } } { ( \hat { g } _ { j } ) ^ { \theta } } , \quad \forall i \in [ m ] , \forall j \in [ n ] , } \end{array}\tag{9}
$$

where $S ^ { ( \theta ) }$ is defined in (3).

Compared with the standard EOT formulation in (1), the problem in (9) reweights the contribution of each pair $\left( \mathbf { x } _ { i } , \mathbf { y } _ { j } \right)$ in both the transport cost and the entropic regularizer by $1 / ( \hat { f } _ { i } \hat { g } _ { j } ) ^ { \theta }$ . The marginal constraints are reweighted analogously: the mass at each data point is scaled inversely by its local density: $1 / ( \hat { f } _ { i } ) ^ { \theta }$ for each $\mathbf { x } _ { i }$ and $1 / ( \hat { g } _ { j } ) ^ { \theta }$ for each $\mathbf { y } _ { j }$ . This reweighting scheme assigns more mass to points in sparse regions and less mass to points in dense regions, compensating for non-uniform sampling. The scaling factor $\checkmark$ on the right-hand side ensures that total mass is conserved across the two reweighted marginals.

The reweighting in Proposition 3.1 is motivated by the goal of discounting the influence of sampling density from the transport plan, so that the resulting transport plan approximates the plan one would obtain if both datasets were drawn according to density-reweighted reference measures from their respective manifolds. In what follows, we first introduce the family of population-level EOT plans $\mathcal { W } _ { \varepsilon } ^ { ( \theta ) }$ under densityreweighted reference measures. We then show that under mild regularity conditions, $\mathbf { W } ^ { ( \theta ) }$ in (6) computed with fixed $\theta \in [ 0 , 1 ]$ and known sampling densities converges uniformly to ${ \mathcal { W } } _ { \varepsilon } ^ { ( \theta ) }$ , up to a global scaling constant, with high probability as $m , n \to \infty$ . We subsequently address the practically relevant setting in which the sampling densities are unknown and must be estimated from data.

We next formalize our assumptions on the data geometry, sampling densities, and dataset sizes.

Assumption 1. $\mathcal { M } _ { x }$ and $\mathcal { M } _ { y }$ are smooth, compact, boundaryless Riemannian manifolds isometrically embedded in $\mathbb { R } ^ { D }$ , with intrinsic dimension $d _ { 1 }$ and $d _ { 2 }$ , respectively. The sampling densities $\mu$ and ν are strictly positive and belong to ${ \mathcal { C } } ^ { 3 } ( { \mathcal { M } } _ { x } )$ and $\mathcal { C } ^ { 3 } ( \mathcal { M } _ { y } )$ , respectively.

Assumption 1 is a standard regularity condition in manifold learning, with compactness of the manifolds and smoothness of the densities ensuring that finite-sample estimators concentrate around their populationlevel counterparts at a controlled rate.

Assumption 2. $n \geq m \geq n ^ { \gamma }$ for some constant $\gamma \in ( 0 , 1 )$ .

We note that Assumption 2 generalizes to $n \geq m \geq c n ^ { \gamma }$ for any constant $c > 0 ;$ we set $c = 1$ for simplicity. To state our results, we define the θ-reweighted density functions as

$$
\mu ^ { ( \theta ) } ( \mathbf { x } ) : = \frac { \left( \mu ( \mathbf { x } ) \right) ^ { 1 - \theta } } { \int _ { \mathcal { M } _ { x } } { \left( \mu ( \mathbf { x } ) \right) ^ { 1 - \theta } } \ d \omega ( \mathbf { x } ) } , \quad \nu ^ { ( \theta ) } ( \mathbf { y } ) : = \frac { \left( \nu ( \mathbf { y } ) \right) ^ { 1 - \theta } } { \int _ { \mathcal { M } _ { y } } { \left( \nu ( \mathbf { y } ) \right) ^ { 1 - \theta } } \ d \eta ( \mathbf { y } ) } ,\tag{10}
$$

where, as θ increases from 0 to 1, the reweighted densities interpolate between the sampling densities $\mu ,$ ν and the uniform densities on their respective manifolds. We define the family of corresponding population-leve EOT plans $\mathcal { W } _ { \varepsilon } ^ { ( \theta ) } : \mathcal { M } _ { x } \times \mathcal { M } _ { y }  \mathbb { R } _ { + }$ and the Gaussian kernel $K _ { \varepsilon } : { \mathcal { M } } _ { x } \times { \mathcal { M } } _ { y } \to { \mathbb { R } } _ { + }$ as

$$
\mathcal { W } _ { \varepsilon } ^ { ( \theta ) } ( \mathbf { x } , \mathbf { y } ) = u _ { \varepsilon } ^ { ( \theta ) } ( \mathbf { x } ) K _ { \varepsilon } ( \mathbf { x } , \mathbf { y } ) v _ { \varepsilon } ^ { ( \theta ) } ( \mathbf { y } ) , \qquad K _ { \varepsilon } ( \mathbf { x } , \mathbf { y } ) = \exp \left\{ \frac { - \| \mathbf { x } - \mathbf { y } \| _ { 2 } ^ { 2 } } { \varepsilon } \right\} ,\tag{11}
$$

where $u _ { \varepsilon } ^ { ( \theta ) }$ and $v _ { \varepsilon } ^ { ( \theta ) }$ are functions that solve the integral equations

$$
\{ \int _ { \mathcal { M } _ { y } } { \mathcal W } _ { \varepsilon } ^ { ( \theta ) } ( \mathbf { x } , \mathbf { y } ) \nu ^ { ( \theta ) } ( \mathbf { y } ) d \eta ( \mathbf { y } ) = \int _ { \mathcal { M } _ { y } } u _ { \varepsilon } ^ { ( \theta ) } ( \mathbf { x } ) K _ { \varepsilon } ( \mathbf { x } , \mathbf { y } ) v _ { \varepsilon } ^ { ( \theta ) } ( \mathbf { y } ) \nu ^ { ( \theta ) } ( \mathbf { y } ) d \eta ( \mathbf { y } ) = 1 , \quad \forall \mathbf { x } \in \mathcal { M } _ { x ; \mathbf { y } } ,\tag{12}
$$

We note ${ \mathcal { W } } _ { \varepsilon } ^ { ( \theta ) }$ is doubly stochastic with respect to the density-reweighted measures $\mu ^ { ( \theta ) } d \omega$ and $\nu ^ { ( \theta ) } d \eta$ . The scaling functions $u _ { \varepsilon } ^ { ( \theta ) }$ and $v _ { \varepsilon } ^ { ( \theta ) }$ are guaranteed to exist as positive, continuous functions on $\mathcal { M } _ { x }$ and $\mathcal { M } _ { y }$ respectively, and are unique up to a positive multiplicative constant [30].

We now relate the transport plan $\mathbf { \bar { W } } ^ { ( \theta ) }$ in (6), computed with $\theta \in [ 0 , 1 ]$ and known sampling densities, to the population-level EOT map ${ \mathscr W } _ { \varepsilon } ^ { ( \theta ) }$ defined in (11), in Theorem 3.2 below. The proof is in Supplement B.3.

Theorem 3.2. Suppose Algorithm 1 is applied with exact sampling densities $\hat { f } _ { i } = \mu (  { \mathbf { x } } _ { i } )$ and $\hat { g } _ { j } = \nu ( \mathbf { y } _ { j } ) , \forall i \in$ $[ m ] , \forall j \in [ n ]$ , and let $\mathbf { W } ^ { ( \theta ) } \in \mathbb { R } _ { + } ^ { m \times n }$ denote the resulting transport plan in (6). Then, under Assumptions 1 and 2, there exist positive constants $\tau _ { 0 } , n _ { 0 } ( \varepsilon ) , C ^ { \prime } ( \varepsilon )$ (which may additionally depend on $\mathcal { M } _ { x } , \mathcal { M } _ { y } , \mu , \nu , \theta _ { \mathrm { ~ } }$ , and $\gamma )$ such that for all $\tau \geq \tau _ { 0 } , n \geq n _ { 0 } ( \varepsilon , \tau )$ , and m $\geq n ^ { \gamma }$ , with probability at least $1 - n ^ { - \tau }$

$$
\left. C ^ { ( \theta ) } W _ { i j } ^ { ( \theta ) } - \mathcal { W } _ { \varepsilon } ^ { ( \theta ) } ( \mathbf { x } _ { i } , \mathbf { y } _ { j } ) \right. \leq \tau C ^ { \prime } ( \varepsilon ) \sqrt { \frac { \log m } { m } }\tag{13}
$$

holds simultaneously for all $i \in [ m ]$ and $j \in [ n ]$ , where $\begin{array} { r } { C ^ { ( \theta ) } = \sqrt { \int _ { \mathcal { M } _ { x } } \mu ( \mathbf { x } ) ^ { 1 - \theta } d \omega ( \mathbf { x } ) \cdot \int _ { \mathcal { M } _ { y } } \nu ( \mathbf { y } ) ^ { 1 - \theta } d \eta ( \mathbf { y } ) } } \end{array}$

Theorem 3.2 shows that for any fixed ε and suficiently large m and $n ,$ the entries $W _ { i j } ^ { ( \theta ) }$ concentrate around ${ \mathcal { W } _ { \varepsilon } ^ { \left( \theta \right) } ( { \bf x } _ { i } , { \bf y } _ { j } ) } / { C ^ { \left( \theta \right) } }$ simultaneously for all pairs $( i , j )$ with high probability. It further provides an explicit probabilistic bound on the approximation error that depends on the smaller dataset size m and decays at the standard sample-topopulation rate $m ^ { - 1 / 2 }$ up to logarithmic factors. In particular, Theorem 3.2 demonstrates that setting $\theta = 1$ in Algorithm 1 efectively removes the dependence of $\mathbf { W } ^ { ( \theta ) }$ on sampling density, recovering a transport plan that depends solely on the underlying geometry of the manifolds. We numerically demonstrate the convergence rate using the same simulation setup as Figure 1 with $\theta = 1$ . As shown in Figure $^ { 3 , }$ the approximation error in both normalized $\ell _ { 1 }$ and $\ell _ { \infty }$ norms decay approximately like $m ^ { - 1 / 2 }$ . Although this stylized example has boundary and piecewise-smooth sampling density (hence outside the formal assumptions of Theorem 3.2), it illustrates the predicted convergence behavior.

![](images/668f196a1a96dee1f52009608045bb12c0bed6ae0e819666328f349f47dead51.jpg)  
Figure 3: Empirical evaluation of the convergence rate of the reweighted EOT plan. Both error metrics align with (13). Here $C ^ { ( \theta = 1 ) } =$ $\sqrt { \mathrm { V o l } ( \mathcal { M } _ { x } ) \mathrm { V o l } ( \mathcal { M } _ { y } ) }$ , where $\mathrm { V o l } ( \mathcal { M } _ { x } )$ and $\dot { \mathrm { V o l } } ( \mathcal { M } _ { y } )$ denote the Riemannian volumes of $\mathcal { M } _ { x }$ and $\mathcal { M } _ { y } ,$ respectively. See Supplement A.1 for the experiment details.

In practice, the sampling densities $\mu$ and ν are usually unknown and need to be estimated from data. A natural approach is kernel density estimation (KDE): given $N$ points $\{ { \mathbf { z } } _ { i } \} _ { i = 1 } ^ { N }$ in $\mathbb { R } ^ { D }$ , the KDE with bandwidth $h > 0$ is given by

$$
\hat { \rho } ( { \bf z } ) = \frac { 1 } { N h ^ { d / 2 } } \sum _ { i = 1 } ^ { N } K \bigg ( \frac { { \bf z } - { \bf z } _ { i } } { \sqrt { h } } \bigg ) ,\tag{14}
$$

where K is a kernel function. If the points are sampled from a d-dimensional manifold embedded in $\mathbb { R } ^ { D }$ , and $h  0$ suficiently slowly as $N \to \infty$ , then the KDE converges pointwise to the underlying sampling density. Moreover, the bandwidth can be tuned optimally to achieve a mean squared error of $\mathcal { \breve { O } } ( N ^ { - 4 / ( d + 4 ) } )$ [31]. Since the form of the KDE (14) depends explicitly on the intrinsic dimension $d ,$ which is typically unknown in practice, we consider a mean-normalized KDE whose construction does not require knowledge of d.

Definition 3.3. For a fixed bandwidth $h > 0$ and kernel $K _ { h } ( x , y ) = \exp ( - \| x - y \| _ { 2 } ^ { 2 } / h )$ , the mean-normalized density estimator at any $\mathbf { x } _ { i } \in \mathbf { X } = \{ \mathbf { x } _ { i } \} _ { i = 1 } ^ { m } \subset \mathcal { M }$ is defined as

$$
\hat { f } ( { \mathbf x } _ { i } ) : = \frac { \hat { q } ( { \mathbf x } _ { i } ) } { \hat { Z } } , \qquad \hat { q } ( { \mathbf x } _ { i } ) : = \frac { 1 } { m - 1 } \sum _ { j \neq i } ^ { m } K _ { h } ( { \mathbf x } _ { i } , { \mathbf x } _ { j } ) , \qquad \hat { Z } : = \frac { 1 } { m } \sum _ { i = 1 } ^ { m } \hat { q } ( { \mathbf x } _ { i } ) .\tag{15}
$$

This construction is motivated by the observation that the optimal $\mathbf { W } ^ { ( \theta ) }$ in Proposition 3.1 is invariant to per-dataset multiplicative scalings of the density estimates: if $\hat { f } _ { i }$ and ${ \hat { g } } _ { j }$ approximate $\mu ( \mathbf { x } _ { i } )$ and $\nu ( \mathbf { y } _ { j } )$ up to a global constant, the resulting transport plan is rescaled by a positive global constant, leaving the correspondence structure intact.

We now establish that the transport plan $\mathbf { W } ^ { ( \theta ) }$ computed with a fixed $\theta \in [ 0 , 1 ]$ using the mean-normalized KDE converges, up to a global scaling constant, to the population EOT plan ${ \mathcal { W } } _ { \varepsilon } ^ { ( \theta ) }$ from (11). The proof is given in Supplement B.6.

Theorem 3.4. Let $\hat { \bf f }$ and $\hat { \bf g }$ be the mean-normalized density estimators from (15) applied to X and Y with bandwidths $h _ { 1 }$ and $h _ { 2 } .$ respectively. Suppose Algorithm 1 uses $\hat { f } _ { i } = \hat { \mathbf { f } } ( \mathbf { x } _ { i } )$ and $\hat { g } _ { j } = \hat { \mathbf { g } } ( \mathbf { y } _ { j } )$ . Then, under Assumptions 1 and 2, there exist positive constants $\tau _ { 0 }$ and $C ^ { \prime } ( \varepsilon )$ such that the following holds. For every $\tau \geq \tau _ { 0 }$ , one can choose positive constants $n _ { 0 } ( \varepsilon , \tau ) , \kappa _ { 1 } ( \varepsilon , \tau ) , \kappa _ { 2 } ( \varepsilon , \tau ) , h _ { 1 } ^ { \prime } ( \varepsilon , \tau ) \leq 1$ , and $h _ { 2 } ^ { \prime } ( \varepsilon , \tau ) \leq 1$ (which may additionally depend on $\mathcal { M } _ { x } , \mathcal { M } _ { y } , \mu , \nu , \theta _ { \mathrm { i } }$ , and $\gamma )$ such that for all $n \geq n _ { 0 } ( \varepsilon , \tau ) , m \geq n ^ { \gamma }$ (under Assumption 2), and all bandwidths $h _ { 1 } , h _ { 2 } \ s a t i s f y i n g$

$$
\left( \frac { \kappa _ { 1 } \log m } { m } \right) ^ { 1 / d _ { 1 } } \leq h _ { 1 } \leq h _ { 1 } ^ { \prime } , \qquad \left( \frac { \kappa _ { 2 } \log n } { n } \right) ^ { 1 / d _ { 2 } } \leq h _ { 2 } \leq h _ { 2 } ^ { \prime } ,\tag{16}
$$

we have with probability at least $1 - n ^ { - \tau } - 4 n ^ { - 2 \gamma }$ ，

$$
\left. C ^ { ( \theta ) ^ { \prime } } W _ { i j } ^ { ( \theta ) } - \mathcal { W } _ { \varepsilon } ^ { ( \theta ) } ( \mathbf { x } _ { i } , \mathbf { y } _ { j } ) \right. \leq \tau C ^ { \prime } ( \varepsilon ) \left( h _ { 1 } + \sqrt { \frac { \log m } { m h _ { 1 } ^ { d _ { 1 } } } } + h _ { 2 } + \sqrt { \frac { \log n } { n h _ { 2 } ^ { d _ { 2 } } } } \right)\tag{17}
$$

holds simultaneously for all $\begin{array} { r l r l } { i } & { { } \in } & { } & { { } [ m ] } \end{array}$ and $j \in [ n ] ,$ , where $\begin{array} { r l r } { C ^ { ' ( \theta ) } } & { { } = } & { ( \| \mu \| _ { L ^ { 2 } } \| \nu \| _ { L ^ { 2 } } ) ^ { \theta } } \end{array}$ $\begin{array} { r } { \sqrt { \int _ { \mathcal { M } _ { x } } \mu ( \mathbf { x } ) ^ { 1 - \theta } d \omega ( \mathbf { x } ) \cdot \int _ { \mathcal { M } _ { y } } \nu ( \mathbf { y } ) ^ { 1 - \theta } d \eta ( \mathbf { y } ) } } \end{array}$ ; with $\begin{array} { r } { \| \mu \| _ { L ^ { 2 } } = \sqrt { \int _ { \mathcal { M } _ { x } } \mu ^ { 2 } d \omega } } \end{array}$ and $\begin{array} { r } { \| \nu \| _ { L ^ { 2 } } = \sqrt { \int _ { \mathcal { M } _ { y } } \nu ^ { 2 } d \eta } } \end{array}$

Theorem 3.4 extends Theorem 3.2 to the practical setting where the sampling densities $\mu$ and ν are unknown and must be estimated from the data. The probabilistic bound in (17) depends explicitly on the bandwidths $h _ { 1 }$ and $h _ { 2 }$ , the sample sizes m and $n ,$ and the intrinsic dimensions $d _ { 1 }$ and $d _ { 2 }$ . Choosing $\begin{array} { r } { h _ { 1 } \asymp \left( \frac { \log m } { m } \right) ^ { 1 / ( d _ { 1 } \bar { + } 2 ) } } \end{array}$ and $\begin{array} { r l r } {  { h _ { 2 } \asymp \binom { \log { \tilde { n } } } { n } ^ { 1 / ( d _ { 2 } + 2 ) } } } \end{array}$ balances the bias-like and variance-like terms for each density estimator in (17), yielding a convergence rate of $\mathcal { O } \bigg ( \Big ( \frac { \log m } { m } \Big ) ^ { 1 / ( d _ { 1 } + 2 ) } + \Big ( \frac { \log n } { n } \Big ) ^ { 1 / ( d _ { 2 } + 2 ) } \bigg )$ . A data-driven approach to set the bandwidths is described in the next section.

## 4 Experiments

In this section, we consider the task of aligning two datasets generated by sampling from the same underlying manifold up to a constant shift, but with substantially diferent sampling densities. We demonstrate that our density-reweighted EOT framework recovers geometrically faithful correspondences under this setting, while standard EOT and UOT (across multiple hyperparameter configurations) fail systematically.

The experiment involves two manifolds $\mathcal { M } _ { x }$ and $\mathcal { M } _ { y } ,$ , each composed of the union of two circular arcs embedded in $\mathbb { R } ^ { 2 }$ . Specifically, $\mathcal { M } _ { x }$ consists of a full unit circle centered at the origin and a semi-circle of radius 2 centered at (4, 1). The manifold $\mathcal { M } _ { y }$ is obtained by translating $\mathcal { M } _ { x }$ by $( 3 , 5 ) ^ { \top }$ . Datasets $\mathbf { X } = \{ \mathbf { x } _ { i } \} _ { i = 1 } ^ { m }$ and $\mathbf { Y } = \{ \mathbf { y } _ { j } \} _ { j = 1 } ^ { n }$ are sampled from $\mathcal { M } _ { x }$ and $\mathcal { M } _ { y }$ respectively, with opposing sampling densities over their components. Specifically, each $\mathbf { x } _ { i }$ is drawn from the unit circle with probability 0.8 and from the semi-circle with probability 0.2, while each $\mathbf { y } _ { j }$ reverses these weights: drawn from the unit circle with probability 0.2 and from the semi-circle with probability 0.8. Additional density variations are imposed within each component; see Supplement A.3 for details and Figure 4(a) for a visualization of the sampling densities on each manifold.

We compare our approach against standard EOT and UOT, with the latter tested across multiple hyperparameter configurations. Performance is measured by the geometric fidelity of the correspondences encoded in the resulting transport plans, using the $\overline { { R _ { k } } }$ metric:

$$
\overline { { R _ { k } } } = \frac { 1 } { 2 } \left( R _ { k } ( { \bf X } , { \bf Y } ) + R _ { k } ( { \bf Y } , { \bf X } ) \right) ,\tag{18}
$$

where

$$
R _ { k } ( { \mathbf S } , { \mathbf T } ) = \frac { 1 } { C _ { { \mathbf S } } } \sum _ { c = 1 } ^ { C _ { { \mathbf S } } } \frac { 1 } { n _ { c } } \sum _ { i = 1 } ^ { n _ { c } } \frac { \left| { \cal K } _ { k } ^ { ( { \mathbf W } ) } ( { \mathbf s } _ { i } , { \mathbf T } ) \cap { \cal K } _ { k } ^ { ( \mathrm { t r u e } ) } ( { \mathbf s } _ { i } , { \mathbf T } ) \right| } { k } .\tag{19}
$$

Here (S, T) is either (X, Y) or $( \mathbf { Y } , \mathbf { X } )$ , with S and T denoting the source and target datasets, respectively, and $\mathbf { s } _ { i }$ denoting the i-th point in the c-th component of S. Here $C _ { \mathbf { S } }$ is the number of disjoint components of the manifold from which S is sampled $( \mathrm { e . g . , } C _ { \mathbf { X } } = C _ { \mathbf { Y } } = 2 ) , n _ { c }$ is the number of points in S belonging to the c-th component, $\mathcal { K } _ { k } ^ { \mathrm { ( t r u e ) } } ( \mathbf { s } _ { i } , \mathbf { T } )$ is the set of true k nearest neighbors of $\mathbf { s } _ { i }$ in $\mathbf { T } _ { : }$ and ${ \mathcal { K } _ { k } ^ { ( \bf W ) } } ( { \bf s } _ { i } , { \bf T } )$ is the set of k points in T with the highest coupling weights from $\mathbf { s } _ { i }$ under plan $\mathbf { W } ^ { ( \theta ) }$ . The metric macro-averages over components so that each geometric component contributes equally regardless of sampling density imbalances, and averages symmetrically over both transfer directions $( \mathbf { X } \to \mathbf { Y }$ and $\mathbf { Y }  \mathbf { X } )$ so that the score does not favor either direction. In our experiment, ${ \ K } _ { k } ^ { \mathrm { ( t r u e ) } }$ is computed by translating Y back by the known shift to obtain $\mathbf { Y } ^ { \prime }$ , and finding nearest neighbors between X and $\mathbf { Y } ^ { \prime }$ via pairwise Euclidean distances.

We next describe the hyperparameter configurations for each method, beginning with parameters specific to each approach, after which we describe how the entropic regularization parameter ε is selected. Our method requires two kernel bandwidths for the density estimation in the mean-normalized KDE (15), which we select via a data-driven procedure combining bootstrap resampling [32] with Lepski’s method [33]. Lepski’s method considers a geometric grid of candidate bandwidths and selects the largest bandwidth h for which the kernel density estimate does not difer significantly from those at any smaller bandwidth in the grid (i.e., the discrepancy remains within the expected stochastic fluctuation). Intuitively, it chooses the coarsest resolution at which no additional statistically significant structure is revealed at finer scales: for any $\eta < h$ in the grid, the diference of density estimates $\hat { f } _ { h } - \hat { f } _ { \eta }$ is small enough to be explained by stochastic variability rather than systematic bias. Bootstrap resampling is used to empirically quantify this stochastic variability. The full procedure is detailed in Algorithm 2 in Supplement A.3. For UOT, we set $\lambda _ { 1 } = \lambda _ { 2 } = : \lambda$ in (7) and test a grid of values for λ. All three methods require choosing an entropic regularization parameter $\varepsilon ,$ which controls the difusiveness of the resulting transport plan. We select ε for each method (with other hyperparameters fixed) by maximizing the number of mutual k-nearest neighbor (MKNN) pairs in the resulting plan, which serves as a proxy for alignment consistency. Specifically, $\mathbf { x } _ { i }$ and $\mathbf { y } _ { j }$ form an MKNN pair if each is among the top-k maximum-coupling points of the other, reflecting a mutually agreed-upon correspondence. When ε is too small, the transport plan is overly sparse and few MKNN pairs exist; when ε is too large, the plan becomes nearly uniform and MKNN pairs are again scarce as correspondences lose specificity. Selecting the $\varepsilon$ that maximizes the number of MKNN pairs at a prespecified k thus strikes a balance between these two extremes, yielding the greatest alignment consistency. See Supplement A.3 for details.

In Figure 4(b), we report $\overline { { R _ { k } } }$ averaged over 10 independent trials for all methods across $k \in [ 5 , 1 0 0 ]$ Our approach $( \theta = 1 )$ substantially outperforms all other methods at every neighborhood size, with $\overline { { R _ { k } } }$ consistently exceeding that of the best UOT configuration by an approximate factor of 2× or more, and that of standard EOT by nearly an order of magnitude across all examined k. Standard EOT performs worst across all $k ,$ confirming that correspondences derived without accounting for heterogeneous sampling densities are systematically confounded by local data density. UOT partially mitigates this failure by relaxing the marginal constraints, but its performance varies substantially across hyperparameter configurations. Such sensitivity to λ may limit the practical utility of UOT, as dataset alignment is typically unsupervised and provides no natural criterion for choosing the degree of marginal relaxation.

![](images/55149e77c1eb772313cd1751e1b6636f1efa1c3db875c7372e6212a3512d0a84.jpg)

![](images/df7d62739dc7ba7daf8025e7c735e11e29df36b39135122fee4d53949e5098cc.jpg)  
Figure 4: Synthetic alignment under opposing sampling densities. (a) Visualization of $\mathcal { M } _ { x }$ and $\mathcal { M } _ { y }$ colored by sampling density (yellow: high; purple: low). (b) Alignment quality $\overline { { R _ { k } } }$ in (18) as a function of $k \in [ 5 , 1 0 0 ]$ for all methods and hyperparameter configurations examined.

## Acknowledgments

The authors used Anthropic Claude Sonnet and OpenAI ChatGPT to assist with language editing, exposition, and technical consistency checks. This work was supported by NIH grants UM1PA051410, U54AG076043, U54AG079759, U01DA053628, P50CA121974, and R33DA047037.

## Appendix A Details of Experiments and Additional Results

## A.1 Experimental Setup for Figures 1, 2, and 3

We consider a two-dimensional toy example to illustrate the efect of sampling densities on EOT plans. The source dataset $\mathbf { X } = \{ \mathbf { x } _ { i } \} _ { i = 1 } ^ { n } = \{ ( x _ { i 1 } , x _ { i 2 } ) \} _ { i = 1 } ^ { n }$ lies on the line $x _ { i 2 } = x _ { i 1 }$ , and the target dataset $\mathbf { Y } =$ $\{ \mathbf { y } _ { j } \} _ { j = 1 } ^ { m } = \{ ( y _ { j 1 } , y _ { j 2 } ) \} _ { j = 1 } ^ { m }$ lies on the curve $y _ { j 2 } = 2 + y _ { j 1 } + 0 . 5 y _ { j 1 } ^ { 2 }$ . Both datasets share the first-coordinate domain [0, 1] but difer substantially in sampling density. Specifically, for the dataset X, the first coordinates $x _ { i 1 }$ are drawn uniformly from [0, 0.5] with probability 0.9 and uniformly from [0.5, 1] with probability 0.1; the dataset Y is generated analogously with the probabilities reversed. The second coordinate of data point in each dataset is then determined by the respective curve equation applied to its first coordinate.

For computing the transport plan, standard EOT imposes uniform empirical measures as marginal constraints (i.e, the row and columns sums are $n \cdot { \bf 1 } _ { m }$ and $m \cdot { \bf 1 } _ { n } .$ , respectively), whereas our method solves for $\mathbf { W } ^ { ( \theta ) }$ via Algorithm 1 using the ground-truth sampling densities. The entropic regularization parameter is fixed at $\varepsilon = 5 \times 1 0 ^ { - 2 }$ for both methods.

Figure 1. The sample sizes are $n = 3 { , } 0 0 0$ and $m = 2 { , } 0 0 0$ . We set $\theta = 1$ in Algorithm 1.

Figure 2. The sample sizes are $n = 3 { , } 0 0 0$ and $m = 2 { , } 0 0 0$ . We vary θ over a uniform grid on [0, 1] and compute the corresponding transport plan $\mathbf { W } ^ { ( \theta ) }$ for each value via Algorithm 1.

Figure 3. The population-level EOT plan $\mathcal { W } _ { \varepsilon }$ in (11) is computed via the following interpolation procedure. We first construct a fine grid of $N = 1 0 \small { , } 0 0 0$ points placed equally spaced with respect to arc length on each manifold, and solve standard EOT on this $N \times N$ grid via the Sinkhorn–Knopp algorithm, yielding scaling vectors $\hat { \mathbf { u } } , \hat { \mathbf { v } } \in \mathbb { R } ^ { N }$ . These vectors are then extended to continuous scaling functions ˜u<sub>ε</sub> and $\tilde { v } _ { \varepsilon }$ via cubic spline interpolation in the first coordinate, yielding approximations to $u _ { \varepsilon }$ and $v _ { \varepsilon }$ defined in (12). For each dataset, the approximate population-level EOT plan is then calculated entrywise as

$$
\begin{array} { r } { \tilde { \mathcal { W } } _ { \varepsilon } \big ( \mathbf { x } _ { i } , \mathbf { y } _ { j } \big ) = \tilde { u } _ { \varepsilon } \big ( x _ { i 1 } \big ) \exp \Bigl ( - \frac { \| \mathbf { x } _ { i } - \mathbf { y } _ { j } \| ^ { 2 } } { \varepsilon } \Bigr ) \tilde { v } _ { \varepsilon } \big ( y _ { j 1 } \big ) , \qquad i \in [ m ] , \ j \in [ n ] . } \end{array}\tag{20}
$$

To generate the convergence curve, we vary m over a log-uniform grid of 10 values on [100, 10,000] and set $n = 2 m$ . For each value of m, the experiment is repeated 20 times independently and the results are averaged to reduce variability. The two error metrics considered are:

$$
E _ { 1 } ( m ) = \frac { 1 } { m n } \sum _ { i = 1 } ^ { m } \sum _ { j = 1 } ^ { n } \bigg | \sqrt { \mathrm { V o l } ( \mathcal { M } _ { x } ) \mathrm { V o l } ( \mathcal { M } _ { y } ) } W _ { i j } - \mathcal { W } _ { \varepsilon } ( \mathbf { x } _ { i } , \mathbf { y } _ { j } ) \bigg | ,\tag{21}
$$

$$
E _ { \infty } ( m ) = \mathop { \operatorname* { m a x } } _ { i , j } \biggl | \sqrt { \mathrm { V o l } ( \mathcal { M } _ { x } ) \mathrm { V o l } ( \mathcal { M } _ { y } ) } W _ { i j } - \mathcal { W } _ { \varepsilon } ( \mathbf { x } _ { i } , \mathbf { y } _ { j } ) \biggr | ,
$$

where $\mathrm { V o l } ( \mathcal { M } _ { x } )$ and $\mathrm { V o l } ( \mathcal { M } _ { y } )$ are computed analytically via arc-length integration.

## A.2 Estimation Performance of mean-normalized KDE

Lemma A.1. Let $\mathcal { M } _ { x } \subset \mathbb { R } ^ { D }$ be a smooth, compact, boundaryless d-dimensional Riemannian manifold isometrically embedded in $\mathbb { R } ^ { D }$ , let $\mu \in \mathcal { C } ^ { 3 } ( \mathcal { M } _ { x } )$ be a positive probability density on $\mathcal { M } _ { x }$ , and let $\mathbf { X } = \{ \mathbf { x } _ { i } \} _ { i = 1 } ^ { m } \stackrel { \mathrm { i i d } } { \sim } \mu$ Let $\hat { f } ( \mathbf { x } _ { i } )$ be the mean-normalized density estimator defined in (15) with bandwidth h. Then there exist positive constants $h _ { 0 } ( \mathcal { M } _ { x } , \mu ) \leq 1 , \kappa ( \mathcal { M } _ { x } , \mu ) , C _ { 0 } ( \mathcal { M } _ { x } , \mu )$ , and $m _ { 0 } ( h _ { 0 } , \kappa ) \geq 2$ such that for all m $\geq m _ { 0 } ( h _ { 0 } , \kappa )$ and all fixed h with

$$
\left( \frac { \kappa \log m } { m } \right) ^ { 1 / d } \leq h \leq h _ { 0 } ( \mathcal { M } _ { x } , \mu ) ,\tag{22}
$$

with probability at least $1 - 2 m ^ { - 2 }$

$$
\left| \hat { f } ( \mathbf { x } _ { i } ) - \frac { \mu ( \mathbf { x } _ { i } ) } { ( \| \mu \| _ { L ^ { 2 } } ) ^ { 2 } } \right| \leq C _ { 0 } \left( h + \sqrt { \frac { \log m } { m h ^ { d } } } \right) \qquad f o r \ a l l \ i \in [ m ] ,\tag{23}
$$

where $\begin{array} { r } { \| \mu \| _ { L ^ { 2 } } = \left( \int _ { \mathcal { M } _ { x } } \mu ( \mathbf { x } ) ^ { 2 } d \omega ( \mathbf { x } ) \right) ^ { 1 / 2 } } \end{array}$

The proof is in Appendix B.5.

## A.3 Experimental Setup for Figure 4

Sampling Density. The sampling density on each component is parameterized by $\phi$ as follows:

• Unit circle of $\mathcal { M } _ { x } \colon \phi = \tilde { \phi }$ mod 2π, where $\tilde { \phi } \sim 0 . 3 \cdot \mathcal { N } ( 0 . 5 \pi , 0 . 8 ^ { 2 } ) + 0 . 7 \cdot \mathcal { N } ( 1 . 5 \pi , 1 . 0 ^ { 2 } ) ;$

• Semi-circle of $\mathcal { M } _ { x } \colon \phi = \tilde { \phi }$ mod π, where $\tilde { \phi } \sim 0 . 6 \cdot \mathcal { N } ( 0 . 2 5 \pi , 0 . 3 ^ { 2 } ) + 0 . 4 \cdot \mathcal { N } ( 0 . 6 5 \pi , 0 . 8 ^ { 2 } )$ ;

• Unit circle of $\mathcal { M } _ { y } \colon \phi = \tilde { \phi }$ mod 2π, where $\tilde { \phi } \sim 0 . 5 \cdot \mathcal { N } ( 0 . 2 \pi , 0 . 5 ^ { 2 } ) + 0 . 5 \cdot \mathcal { N } ( 1 . 3 \pi , 0 . 9 ^ { 2 } ) ;$

• Semi-circle of $\mathcal { M } _ { y } \colon \phi = \tilde { \phi }$ mod π, where $\tilde { \phi } \sim 0 . 2 5 \cdot \mathcal { N } ( 0 . 1 \pi , 0 . 4 ^ { 2 } ) + 0 . 7 5 \cdot \mathcal { N } ( 0 . 7 \pi , 0 . 6 ^ { 2 } )$

where ${ \mathcal { N } } ( \mu , \sigma ^ { 2 } )$ denotes a Gaussian distribution with mean $\mu$ and variance $\sigma ^ { 2 }$ . Given ϕ, a point on a circle or semi-circle of radius r centered at c is given by $\mathbf { c } + r ( \cos \phi , \sin \phi ) ^ { \top }$ . We note that the densities of $\phi$ on each component are chosen arbitrarily and are not tuned to favor any particular method.

KDE bandwidth Selection. The KDE bandwidth for each dataset X and Y is selected independently via Algorithm 2 with the following settings: $k = 2 0 , 2 h _ { \operatorname* { m i n } } ^ { 2 } = 1 0 ^ { - 2 } , 2 h _ { \operatorname* { m a x } } ^ { 2 } = 1 , m = 5 0 , \alpha = 0 . 0 5 , \mathrm { { a } }$ and $C = 1$

Entropic Regularization Parameter ε. For each method, ε is chosen to maximize the number of MKNN pairs $( K = 5 0 )$ in the resulting transport plan, via golden-section search over [0, 5], a range chosen based on the scale of the data. The search terminates when the active interval shrinks below 0.10 or after at most 20 iterations, whichever comes first.

Figure 4. We fix the sample sizes at $m = n = 2 { , } 0 0 0$ and compute transport plans using standard EOT, our approach with $\theta = 1$ , and UOT with $\lambda = \gamma \varepsilon { \mathrm { ~ f o r ~ } } \gamma \in \{ 0 . 1 , 1 , 1 0 , 1 0 0 \}$ , with all remaining hyperparameters selected as described above. Each plan is evaluated via $\overline { { R _ { k } } }$ as defined in (18), and results are averaged over 10 independent replicates.

Algorithm 2 Bootstrap–Lepski Bandwidth Selection   
Input: Datasets ${ \bf X } = \{ { \bf x } _ { i } \} _ { i = 1 } ^ { n } \subset \mathbb { R } ^ { D } ;$ number of bootstrap resamples $k \in \mathbb { N } ;$ bandwidth range $( h _ { \operatorname* { m i n } } , h _ { \operatorname* { m a x } } )$ ;   
number of candidate bandwidths m ∈ N; quantile level $\alpha \in ( 0 , 1 )$ ; threshold scaling constant $C > 0$   
1: Construct bandwidth grid. Form the geometric grid $\mathcal { H } = \{ h _ { 1 } , \ldots , h _ { m } \}$ with   
$h _ { j } = h _ { \operatorname* { m i n } } \cdot \gamma \ u ^ { j - 1 } , \qquad j \in [ m ] ,$ (24)   
where $\gamma = ( h _ { \operatorname* { m a x } } / h _ { \operatorname* { m i n } } ) ^ { 1 / ( m - 1 ) }$   
2: Bootstrap Resampling. Resample k i.i.d. datasets $\widetilde { \mathbf { X } } ^ { ( 1 ) } , \ldots , \widetilde { \mathbf { X } } ^ { ( k ) }$ of size n uniformly with replacement   
from X.   
3: Compute pairwise squared distance matrices.   
• Self-distances on $\mathbf { X } \colon$   
$D _ { i j } = \| \mathbf { x } _ { i } - \mathbf { x } _ { j } \| ^ { 2 } , \qquad \forall i , j \in [ n ] .$ (25)   
• Cross-distances between X and each resample $\widetilde { \mathbf { X } } ^ { \left( \ell \right) }$   
$D _ { i j } ^ { ( \ell ) } = \big \| \mathbf { x } _ { i } - \tilde { \mathbf { x } } _ { j } ^ { ( \ell ) } \big \| ^ { 2 } , \qquad \forall i , j \in [ n ] , \ell \in [ k ] .$ (26)   
4: Evaluate mean-normalized density estimates. For each $h \in \mathcal { H } .$ compute the mean-normalized   
KDE (Definition 3.3) at each $\mathbf { x } _ { p } \in \mathbf { X }$ as:   
$\hat { f } _ { h } ( \mathbf { x } _ { p } ) = \frac { \displaystyle \sum _ { j = 1 } ^ { n } \exp \left( - \frac { D _ { p j } } { 2 h ^ { 2 } } \right) } { \displaystyle \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \sum _ { j = 1 } ^ { n } \exp \left( - \frac { D _ { i j } } { 2 h ^ { 2 } } \right) } ,$ (27)   
and its bootstrap counterpart on $\widetilde { \mathbf { X } } ^ { \left( \ell \right) }$ , with normalizer retained from (27) as:   
$\hat { f } _ { h } ^ { ( \ell ) } ( \mathbf { x } _ { p } ) = \frac { \displaystyle \sum _ { j = 1 } ^ { n } \exp \left( - \frac { D _ { p j } ^ { ( \ell ) } } { 2 h ^ { 2 } } \right) } { \displaystyle \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \sum _ { j = 1 } ^ { n } \exp \left( - \frac { D _ { i j } } { 2 h ^ { 2 } } \right) } , \qquad \forall l \in [ k ] .$ (28)   
5: Form bootstrap null distributions and pointwise thresholds. For each pair $h _ { i } < h _ { j }$ and each   
point $\mathbf { x } _ { p } \in \mathbf { X }$ , compute the centered absolute bootstrap diferences:   
$\widetilde { V } _ { i j } ^ { ( \ell ) } ( \mathbf { x } _ { p } ) = \Big | \big ( \hat { f } _ { h _ { j } } ^ { ( \ell ) } ( \mathbf { x } _ { p } ) - \hat { f } _ { h _ { i } } ^ { ( \ell ) } ( \mathbf { x } _ { p } ) \big ) - \bar { V } _ { i j } ( \mathbf { x } _ { p } ) \Big | , \quad \ell \in [ k ] ,$ (29)   
where $\begin{array} { r } { \bar { V } _ { i j } ( \mathbf { x } _ { p } ) = \frac { 1 } { k } \sum _ { \ell = 1 } ^ { k } \bigl ( \hat { f } _ { h _ { j } } ^ { ( \ell ) } ( \mathbf { x } _ { p } ) - \hat { f } _ { h _ { i } } ^ { ( \ell ) } ( \mathbf { x } _ { p } ) \bigr ) } \end{array}$ . Set the pointwise threshold as the scaled $( 1 - \alpha )$ -quantile:   
$T _ { i j } ( \mathbf { x } _ { p } ) = C \cdot q _ { 1 - \alpha } \biggl ( \widetilde { V } _ { i j } ^ { ( 1 ) } ( \mathbf { x } _ { p } ) , \dots , \widetilde { V } _ { i j } ^ { ( k ) } ( \mathbf { x } _ { p } ) \biggr ) .$ (30)   
6: Lepski bandwidth selection. Sweep $j = 1 , \ldots , m$ in increasing order and return the largest $h _ { j }$ such   
that for all $i < j ,$   
$\sum _ { p = 1 } ^ { n } \bigl | \hat { f } _ { h _ { j } } ( \mathbf { x } _ { p } ) - \hat { f } _ { h _ { i } } ( \mathbf { x } _ { p } ) \bigr | \ \leq \ \sum _ { p = 1 } ^ { n } T _ { i j } ( \mathbf { x } _ { p } ) .$ (31)

## Appendix B Proofs and Related Background

## B.1 Proof of Proposition 3.1

Proof. The proof proceeds in two steps. We first verify that the feasible set of (9) is nonempty and compact, which, together with the strict convexity and continuity of the objective, guarantees the existence of a unique minimizer. We then show that the matrix $\mathbf { W } ^ { ( \theta ) }$ defined in (6) is feasible and satisfies the first-order optimality conditions, which are suficient to identify it as the unique minimizer.

For notational simplicity, we define

$$
r _ { i } : = \frac { n } { \sqrt { S ^ { ( \theta ) } } ( \hat { f } _ { i } ) ^ { \theta } } , \qquad c _ { j } : = \frac { m \sqrt { S ^ { ( \theta ) } } } { ( \hat { g } _ { j } ) ^ { \theta } } , \qquad C _ { i j } : = \| \mathbf { x } _ { i } - \mathbf { y } _ { j } \| _ { 2 } ^ { 2 } , \qquad h _ { i j } : = ( \hat { f } _ { i } ) ^ { \theta } ( \hat { g } _ { j } ) ^ { \theta } , \qquad \forall i \in [ m ] , \forall j \in [ n ] .\tag{32}
$$

Step 1. A direct calculation using the definition of $S ^ { ( \theta ) }$ in (3) yields $\textstyle \sum _ { i = 1 } ^ { m } r _ { i } = \sum _ { j = 1 } ^ { n } c _ { j }$ , so the feasible set of (9) is nonempty: for instance, the rank-one plan $\begin{array} { r } { W _ { i j } = h _ { i j } r _ { i } c _ { j } / \sum _ { k } r _ { k } } \end{array}$ satisfies both marginal constraints. Moreover, the feasible set is closed (as the intersection of finitely many hyperplanes with the nonnegative orthant) and bounded (since each entry satisfies $0 \leq W _ { i j } \leq h _ { i j }$ min $\{ n r _ { i } , m c _ { j } \} )$ , hence compact.

Step 2. Feasibility. By (6) and the definition of $\mathbf { M } ^ { ( \theta ) } \mathrm { i n } \left( 4 \right)$ , we have

$$
\frac { W _ { i j } ^ { ( \theta ) } } { h _ { i j } } \ : = \ : \alpha _ { i } ^ { ( \theta ) } \ : M _ { i j } ^ { ( \theta ) } \ : \beta _ { j } ^ { ( \theta ) } , \qquad \forall i \in [ m ] , \ : j \in [ n ] .\tag{33}
$$

Substituting into the Sinkhorn scaling conditions in (5) gives exactly the marginal constraints in (9).

First-order conditions. Introducing multipliers $\pmb { \lambda } \in \mathbb { R } ^ { m }$ and $\pmb { \mu } \in \mathbb { R } ^ { n }$ for the row and column constraints in (9) respectively, the Lagrangian reads

$$
\mathcal { L } ( \tilde { \mathbf { W } } , \lambda , \mu ) = \sum _ { i , j } \frac { \tilde { W } _ { i j } } { h _ { i j } } C _ { i j } + \varepsilon \sum _ { i , j } \frac { \tilde { W } _ { i j } } { h _ { i j } } \log { \tilde { W } _ { i j } } - \sum _ { i } \lambda _ { i } \left( \frac { 1 } { n } \sum _ { j } \frac { \tilde { W } _ { i j } } { h _ { i j } } - \frac { 1 } { \sqrt { S ^ { ( \theta ) } } \tilde { f } _ { i } ^ { ( \theta ) } } \right) - \sum _ { j } \mu _ { j } \left( \frac { 1 } { m } \sum _ { i } \frac { \tilde { W } _ { i j } } { h _ { i j } } - \frac { \sqrt { S ^ { ( \theta ) } } } { \left( \tilde { g } _ { j } \right) ^ { \theta } } \right) .
$$

Diferentiating with respect to $\tilde { W } _ { i j }$ , and setting $\partial \mathcal { L } / \partial \tilde { W } _ { i j } = 0$ yields

$$
\tilde { W } _ { i j } \ = \ \exp \left( \frac { \lambda _ { i } } { n \varepsilon } - \frac { 1 } { 2 } \right) \cdot \exp \left( - \frac { C _ { i j } } { \varepsilon } \right) \cdot \exp \left( \frac { \mu _ { j } } { m \varepsilon } - \frac { 1 } { 2 } \right) , \quad \forall i \in [ m ] , \forall j \in [ n ] .\tag{34}
$$

We note $\mathbf { W } ^ { ( \theta ) }$ in (6) factors in the same way.

## B.2 Auxiliary Definition and Lemma for the Proof of Theorem 3.2 and 3.4

We first introduce the matrix scaling problem and state the relevant lemmas, following Section SM5 of [29]. Matrix scaling is the process of diagonally scaling a positive matrix $\mathbf { A } \in \mathbb { R } _ { + + } ^ { m \times n }$ to achieve certain prescribed row sums $\mathbf { r } \in \mathbb { R } _ { + + } ^ { m }$ and column sums $\mathbf { c } \in \mathbb { R } _ { + + } ^ { n }$ , where $\| \mathbf { r } \| _ { 1 } = \| \mathbf { c } \| _ { 1 }$ . We say a pair of vectors α $\in \mathbb { R } _ { + } ^ { m }$ and $\beta \in \mathbb { R } _ { + } ^ { n }$ scales A to row sums r and column sums $\mathbf { c } ,$ if

$$
r _ { i } = \sum _ { j = 1 } ^ { n } \alpha _ { i } A _ { i , j } \beta _ { j } , ~ \mathrm { a n d } ~ c _ { j } = \sum _ { i = 1 } ^ { m } \alpha _ { i } A _ { i , j } \beta _ { j } ,\tag{35}
$$

for all $i \in [ m ]$ and $j \in [ n ]$ . We refer to α and $\beta$ from (35) (or their entries) as scaling factors of A.

Lemma B.1 (Existence and uniqueness [34]). Suppose that $\mathbf { A } \in \mathbb { R } _ { + + } ^ { m \times n } , \ \mathbf { r } \in \mathbb { R } _ { + + } ^ { m }$ , and $\mathbf { c } \in \mathbb { R } _ { + + } ^ { n }$ with $\| \mathbf { r } \| _ { 1 } = \| \mathbf { c } \| _ { 1 }$ . Then, there exists a pair of vectors $\pmb { \alpha } \in \mathbb { R } _ { + + } ^ { m }$ and $\beta \in \mathbb { R } _ { + + } ^ { n }$ that scales A to row sums r and column sums c. Furthermore, the pair of scaling factor $( \alpha , \beta )$ is unique up to a scaler multiple; that is, any other valid scaling pair is of the form $( c \pmb { \alpha } , c ^ { - 1 } \beta )$ for some constant $c > 0$

The following lemma characterizes the stability of the scaling factors to perturbations in the row and column sums. In particular, it states that if there exists a pair of positive vectors $( { \hat { \alpha } } , { \hat { \beta } } )$ that approximately scales A to certain row and column sums, then $( { \hat { \alpha } } , { \hat { \boldsymbol { \beta } } } )$ must be close to a true pair of scaling factors for these row and column sums.

Lemma B.2 (Stability of scaling factors under approximate scaling [29]). Let $\mathbf { A } \in \mathbb { R } _ { + + } ^ { m \times n }$ and denote $\begin{array} { r } { a = \operatorname* { m i n } _ { i , j } A _ { i , j } , b = \operatorname* { m a x } _ { i , j } A _ { i , j } } \end{array}$ . Suppose that there exists $\delta \in ( 0 , 1 )$ and vectors $\hat { \pmb { \alpha } } \in \mathbb { R } _ { + + } ^ { m }$ and $\hat { \beta } \in \mathbb R _ { + + } ^ { n } .$ such that

$$
\left| \frac { 1 } { c _ { j } } \sum _ { i = 1 } ^ { m } \hat { \alpha } _ { i } A _ { i , j } \hat { \beta } _ { j } - 1 \right| \leq \delta , \qquad \left| \frac { 1 } { r _ { i } } \sum _ { j = 1 } ^ { n } \hat { \alpha } _ { i } A _ { i , j } \hat { \beta } _ { j } - 1 \right| \leq \delta ,\tag{36}
$$

for $a l l i \in [ m ]$ and $j \in [ n ]$ . Then, there exists a pair of vectors $\pmb { \alpha } \in \mathbb { R } _ { + + } ^ { m }$ and $\beta \in \mathbb { R } _ { + + } ^ { n }$ that scales A to row sums r and column sums c, satisfying

$$
\frac { \left| \alpha _ { i } - \hat { \alpha } _ { i } \right| } { \hat { \alpha } _ { i } } \leq \delta \left( \frac { 1 } { 1 - \delta } + \frac { 4 \sqrt { b } } { a ^ { 2 } \sqrt { s } C _ { 1 } ^ { 3 / 2 } C _ { 2 } ^ { 3 / 2 } m \operatorname* { m i n } _ { i } r _ { i } } \right) ,\tag{37}
$$

$$
\frac { | \beta _ { j } - \hat { \beta } _ { j } | } { \hat { \beta } _ { j } } \leq \delta \left( \frac { 1 } { 1 - \delta } + \frac { 4 \sqrt { b } } { a ^ { 2 } \sqrt { s } C _ { 1 } ^ { 3 / 2 } C _ { 2 } ^ { 3 / 2 } n \operatorname* { m i n } _ { j } c _ { j } } \right) ,\tag{38}
$$

for all $i \in [ m ]$ and $j \in [ n ]$ , where $\begin{array} { r } { C _ { 1 } = \operatorname* { m i n } _ { i } \{ \hat { \alpha } _ { i } / r _ { i } \} , C _ { 2 } = \operatorname* { m i n } _ { j } \{ \hat { \beta } _ { j } / c _ { j } \} } \end{array}$ , and $s = \| \mathbf { r } \| _ { 1 } = \| \mathbf { c } \| _ { 1 }$

For notational simplicity, we introduce the notation of order in probability, denoted as $\widetilde { \mathcal { O } } _ { m , n } ^ { ( \varepsilon ) }$ .

Definition B.3. For a random variable z, we say $z ~ = ~ \widetilde { \mathcal { O } } _ { m , n } ^ { ( \varepsilon ) } \left( f ( m , n ) \right)$ if there exist global constant $t _ { 0 } , n _ { 0 } ( \varepsilon , t ) , C ( \varepsilon ) > 0$ such that for all $t > t _ { 0 } , n \geq n _ { 0 } ( \varepsilon , t )$ , and $m \geq n ^ { \gamma }$ (under Assumption 2),

$$
| z | \leq t C ( \varepsilon ) f ( m , n ) ,\tag{39}
$$

with probability at least $1 - n ^ { - t }$ . When $n _ { 0 }$ and C can be chosen independently of ε, we write $z =$ $\widetilde { \mathcal { O } } _ { m , n } ( f ( m , n ) )$

Note that under Definition B.3, given a polynomial function $P ( n )$ , if for any $i = 1 , 2 , \ldots , P ( n ) , z _ { i } =$ $\widetilde { \mathcal { O } } _ { m , n } ^ { ( \varepsilon ) } \left( f ( m , n ) \right)$ ), then

$$
\operatorname* { m a x } _ { i = 1 , 2 \ldots , P ( n ) } | z _ { i } | = \widetilde { \mathcal { O } } _ { m , n } ^ { ( \varepsilon ) } \left( f ( m , n ) \right) ,\tag{40}
$$

which can be derived by applying the union bound.

Furthermore, if $Z = \widetilde { \mathcal { O } } _ { m , n } ^ { ( \varepsilon ) } ( f ( m , n ) )$ and $Y = g ^ { ( \varepsilon ) } ( Z )$ , where $g ^ { ( \varepsilon ) }$ is $\mathcal { C } ^ { 1 }$ on a neighborhood of 0 for all $\varepsilon > 0$ and $\begin{array} { r } { \operatorname* { l i m } _ { m , n \to \infty } f ( m , n ) = 0 } \end{array}$ , then by a Taylor expansion of $g ^ { ( \varepsilon ) } ( x )$ around zero,

$$
Y = g ^ { ( \varepsilon ) } ( 0 ) + \widetilde { \mathcal { O } } _ { m , n } ^ { ( \varepsilon ) } ( f ( m , n ) ) .\tag{41}
$$

We will use properties (40) and (41) of Definition B.3 seamlessly throughout the remaining proofs.

## B.3 Proof of Theorem 3.2

For notation simplicity, we define:

$$
Z _ { x } ^ { ( \theta ) } = \int _ { \mathcal { M } _ { x } } \mu ( \mathbf { x } ) ^ { 1 - \theta } d \omega ( \mathbf { x } ) , \qquad Z _ { y } ^ { ( \theta ) } = \int _ { \mathcal { M } _ { y } } \nu ( \mathbf { y } ) ^ { 1 - \theta } d \eta ( \mathbf { y } ) .\tag{42}
$$

Proof. Let $\alpha ^ { ( \theta ) }$ and $\beta ^ { ( \theta ) }$ be the scaling factors in Algorithm 1 that diagonally scale $\mathbf { M } ^ { ( \theta ) }$ , defined in (4), to satisfy the row and column constraints in (5), for fixed $\boldsymbol { \theta } \in [ 0 , 1 ] , \hat { \mathbf { f } } = \boldsymbol { \mu }$ , and $\hat { \mathbf { g } } = \pmb { \nu }$ . Define approximate scaling factors αˆ and $\hat { \boldsymbol \beta }$ as

$$
\hat { \alpha } _ { i } ^ { ( \theta ) } = \frac { 1 } { \sqrt { Z _ { x } ^ { ( \theta ) } } } u _ { \varepsilon } ^ { ( \theta ) } ( \mathbf { x } _ { i } ) , ~ \mathrm { a n d } ~ \hat { \beta } _ { j } ^ { ( \theta ) } = \frac { 1 } { \sqrt { Z _ { y } ^ { ( \theta ) } } } v _ { \varepsilon } ^ { ( \theta ) } ( \mathbf { y } _ { j } ) ,\tag{43}
$$

for any $i \in [ m ]$ and any $j \in [ n ]$ , where $u _ { \varepsilon } ^ { ( \theta ) }$ and $v _ { \varepsilon } ^ { ( \theta ) }$ are defined in (12).

We first derive a bound on $S ^ { ( \theta ) }$ in (3). Applying Hoefding’s inequality, we have:

$$
\frac { 1 } { m } \sum _ { i = 1 } ^ { m } \frac { 1 } { ( \hat { f } _ { i } ) ^ { \theta } } = \frac { 1 } { m } \sum _ { i = 1 } ^ { m } \frac { 1 } { ( \mu ( \mathbf { x } _ { i } ) ) ^ { \theta } } = \int _ { M _ { x } } \frac { 1 } { ( \mu ( \mathbf { x } ) ) ^ { \theta } } \mu ( \mathbf { x } ) d \omega ( \mathbf { x } ) + \widetilde { \mathcal { O } } _ { m } \left( \sqrt { \frac { \log m } { m } } \right) = Z _ { x } ^ { ( \theta ) } + \widetilde { \mathcal { O } } _ { m } \left( \sqrt { \frac { \log m } { m } } \right) ,\tag{44}
$$

where $Z _ { x } ^ { ( \theta ) }$ is defined in (10). Similarly,

$$
\frac { 1 } { n } \sum _ { j = 1 } ^ { n } \frac { 1 } { ( \hat { g } _ { j } ) ^ { \theta } } = Z _ { y } ^ { ( \theta ) } + \widetilde { \mathcal { O } } _ { n } \left( \sqrt { \frac { \log n } { n } } \right) = Z _ { y } ^ { ( \theta ) } + \widetilde { \mathcal { O } } _ { m } \left( \sqrt { \frac { \log m } { m } } \right) .\tag{45}
$$

In the derivation, we use the fact that $\sqrt { \log n / n } \leq 2 \sqrt { \log m / m }$ for all suficiently large m under Assumption 2.

Combining (44) and (45),

$$
S ^ { ( \theta ) } = \frac { Z _ { x } ^ { ( \theta ) } } { Z _ { y } ^ { ( \theta ) } } + \widetilde { \mathcal { O } } _ { m } \left( \sqrt { \frac { \log m } { m } } \right) , \quad \mathrm { ~ a n d ~ } \quad \frac { 1 } { S ^ { ( \theta ) } } = \frac { Z _ { y } ^ { ( \theta ) } } { Z _ { x } ^ { ( \theta ) } } + \widetilde { \mathcal { O } } _ { m } \left( \sqrt { \frac { \log m } { m } } \right) .\tag{46}
$$

We next show that $\hat { \pmb { \alpha } } ^ { ( \theta ) }$ and $\hat { \boldsymbol { \beta } } ^ { ( \theta ) }$ approximately scale $\mathbf { M } ^ { ( \theta ) }$ to have prescribed row sums r and column sums c. Recall that, from (5),

$$
r _ { i } = \frac { n } { ( \hat { f } _ { i } ) ^ { \theta } \sqrt { S ^ { ( \theta ) } } } = \frac { n } { ( \mu ( \mathbf { x } _ { i } ) ) ^ { \theta } \cdot \sqrt { S ^ { ( \theta ) } } } , \quad \mathrm { a n d } \quad c _ { j } = \frac { m \sqrt { S ^ { ( \theta ) } } } { ( \hat { g } _ { j } ) ^ { \theta } } = \frac { m \sqrt { S ^ { ( \theta ) } } } { ( \nu ( \mathbf { y } _ { j } ) ) ^ { \theta } } ,\tag{47}
$$

for all $i \in [ m ]$ and $j \in [ n ]$ , and $S ^ { ( \theta ) }$ is defined in (3). Using (43), (47), K<sub>ε</sub> and $\mathbf { M } ^ { ( \theta ) }$ defined in (11) and (4), we have: for any $i \in [ m ]$

$$
\begin{array} { r l r } & { } & { \displaystyle \frac { 1 } { r _ { i } } \sum _ { j = 1 } ^ { n } \hat { \alpha } _ { i } ^ { ( \theta ) } M _ { i j } ^ { ( \theta ) } \hat { \beta } _ { j } ^ { ( \theta ) } = \frac { ( \mu ( \mathbf { x } _ { i } ) ) ^ { \theta } \cdot \sqrt { S ^ { ( \theta ) } } } { n } \sum _ { j = 1 } ^ { n } \frac { u _ { \varepsilon } ^ { ( \theta ) } ( \mathbf { x } _ { i } ) } { \sqrt { Z _ { x } ^ { ( \theta ) } } } \frac { K _ { \varepsilon } ( \mathbf { x } _ { i } , \mathbf { y } _ { j } ) } { ( \mu ( \mathbf { x } _ { i } ) \nu ( \mathbf { y } _ { j } ) ) ^ { \theta } } \frac { v _ { \varepsilon } ^ { ( \theta ) } ( \mathbf { y } _ { j } ) } { \sqrt { Z _ { y } ^ { ( \theta ) } } } } \\ & { } & { \displaystyle = \frac { \sqrt { S ^ { ( \theta ) } } } { \sqrt { Z _ { x } ^ { ( \theta ) } Z _ { y } ^ { ( \theta ) } } } \frac { 1 } { n } \sum _ { j = 1 } ^ { n } u _ { \varepsilon } ^ { ( \theta ) } ( \mathbf { x } _ { i } ) K _ { \varepsilon } ( \mathbf { x } _ { i } , \mathbf { y } _ { j } ) v _ { \varepsilon } ^ { ( \theta ) } ( \mathbf { y } _ { j } ) \frac { 1 } { ( \nu ( \mathbf { y } _ { j } ) ) ^ { \theta } } . } \end{array}\tag{48}
$$

Conditioning on $\mathbf { x } _ { i }$ and applying Hoefding’s inequality yields

$$
\begin{array} { r l } & { \frac { 1 } { r _ { i } } \displaystyle \sum _ { j = 1 } ^ { n } \hat { \alpha } _ { i } ^ { ( \theta ) } M _ { i j } ^ { ( \theta ) } \hat { \beta } _ { j } ^ { ( \theta ) } = \frac { \sqrt { S ^ { ( \theta ) } } } { \sqrt { Z _ { x } ^ { ( \theta ) } Z _ { y } ^ { ( \theta ) } } } \int _ { \mathcal { M } _ { y } } u _ { \varepsilon } ^ { ( \theta ) } ( \mathbf { x } _ { i } ) \mathcal { K } _ { \varepsilon } ( \mathbf { x } _ { i } , \mathbf { y } ) v _ { \varepsilon } ^ { ( \theta ) } ( \mathbf { y } ) \cdot ( \nu ( \mathbf { y } ) ) ^ { 1 - \theta } d \eta ( \mathbf { y } ) + \widetilde { \mathcal { O } } _ { n } \left( \sqrt { \frac { \log n } { n } } \right) } \\ & { = \frac { \sqrt { S ^ { ( \theta ) } } Z _ { y } ^ { ( \theta ) } } { \sqrt { Z _ { x } ^ { ( \theta ) } Z _ { y } ^ { ( \theta ) } } } + \widetilde { \mathcal { O } } _ { n } \left( \sqrt { \frac { \log n } { n } } \right) = \sqrt { \frac { Z _ { x } ^ { ( \theta ) } } { Z _ { y } ^ { ( \theta ) } } + \widetilde { \mathcal { O } } _ { m } \left( \sqrt { \frac { \log m } { m } } \right) } \sqrt { \frac { Z _ { y } ^ { ( \theta ) } } { Z _ { x } ^ { ( \theta ) } } } + \widetilde { \mathcal { O } } _ { n } \left( \sqrt { \frac { \log n } { n } } \right) } \\ & { = 1 + \widetilde { \mathcal { O } } _ { m } \left( \sqrt { \frac { \log m } { m } } \right) + \widetilde { \mathcal { O } } _ { n } \left( \sqrt { \frac { \log n } { n } } \right) = 1 + \widetilde { \mathcal { O } } _ { m } \left( \sqrt { \frac { \log m } { m } } \right) , } \end{array}\tag{49}
$$

where we use (12) and (46) in the second line. Under Assumption 2, a union bound over all $i \in [ m ]$ shows that (49) holds simultaneously for all $i \in [ m ]$

Similarly, for any $j \in [ n ]$ ，

$$
\frac { 1 } { c _ { j } } \sum _ { i = 1 } ^ { m } \hat { \alpha } _ { i } ^ { ( \theta ) } M _ { i j } ^ { ( \theta ) } \hat { \beta } _ { j } ^ { ( \theta ) } = \frac { 1 } { \sqrt { S ^ { ( \theta ) } } \sqrt { Z _ { x } ^ { ( \theta ) } Z _ { y } ^ { ( \theta ) } } } \frac { 1 } { m } \sum _ { i = 1 } ^ { m } u _ { \varepsilon } ^ { ( \theta ) } ( \mathbf { x } _ { i } ) \mathcal { K } _ { \varepsilon } ( \mathbf { x } _ { i } , \mathbf { y } _ { j } ) v _ { \varepsilon } ^ { ( \theta ) } ( \mathbf { y } _ { j } ) \frac { 1 } { ( \mu ( \mathbf { x } _ { i } ) ) ^ { \theta } } .\tag{50}
$$

Following an analogous derivation of (49),

$$
\frac { 1 } { c _ { j } } \sum _ { i = 1 } ^ { m } \hat { \alpha } _ { i } ^ { ( \theta ) } M _ { i j } ^ { ( \theta ) } \hat { \beta } _ { j } ^ { ( \theta ) } = 1 + \widetilde { \mathcal { O } } _ { m } \left( \sqrt { \frac { \log m } { m } } \right) ,\tag{51}
$$

for all $j \in [ n ]$

We now apply Lemma B.2 to the approximate scaling factor of $\mathbf { M } ^ { ( \theta ) }$ , namely $\hat { \pmb { \alpha } } ^ { ( \theta ) }$ and $\hat { \boldsymbol { \beta } } ^ { ( \theta ) }$ . We first evaluate and bound the diferent quantities appearing in Lemma B.2.

By the definition of s in Lemma B.2 and using (44), (46), and (47):

$$
s = \| \mathbf { r } \| _ { 1 } = \sum _ { i = 1 } ^ { m } { \frac { n } { ( \mu ( \mathbf { x } _ { i } ) ) ^ { \theta } \cdot { \sqrt { S ^ { ( \theta ) } } } } } = { \frac { n m } { \sqrt { S ^ { ( \theta ) } } } } \cdot { \frac { 1 } { m } } \sum _ { i = 1 } ^ { m } { \frac { 1 } { ( \mu ( \mathbf { x } _ { \mathbf { i } } ) ) ^ { \theta } } } = n m \left( { \sqrt { Z _ { x } ^ { ( \theta ) } Z _ { y } ^ { ( \theta ) } } } + { \tilde { \mathcal { O } } } _ { m } \left( { \sqrt { \frac { \log m } { m } } } \right) \right) .\tag{52}
$$

Since $u _ { \varepsilon } ^ { ( \theta ) } , v _ { \varepsilon } ^ { ( \theta ) } , \mu .$ and ν are positive and continuous on the compact manifolds, they are bounded away from zero and infinity. Thus, there exist some positive constants $C _ { 1 } ^ { ' } ( \varepsilon )$ and $C _ { 2 } ^ { ' } ( \varepsilon )$ (which may depend on ε), such that $C _ { 1 }$ and $C _ { 2 }$ defined in Lemma B.2 satisfy:

$$
C _ { 1 } = \operatorname* { m i n } _ { i } \left\{ \frac { \hat { \alpha } _ { i } ^ { ( \theta ) } } { r _ { i } } \right\} \geq \frac { C _ { 1 } ^ { ' } ( \varepsilon ) } { n } , \quad C _ { 2 } = \operatorname* { m i n } _ { j } \left\{ \frac { \hat { \beta } _ { j } ^ { ( \theta ) } } { c _ { j } } \right\} \geq \frac { C _ { 2 } ^ { ' } ( \varepsilon ) } { m } .\tag{53}
$$

Similarly, there exist positive constants $C _ { 3 } ^ { ' } ( \varepsilon )$ and $C _ { 4 } ^ { ' } ( \varepsilon )$ such that

$$
C _ { 3 } ^ { ' } ( \varepsilon ) \leq M _ { i j } ^ { ( \theta ) } = \frac { \exp \left( - \| \mathbf x _ { i } - \mathbf y _ { j } \| _ { 2 } ^ { 2 } / \varepsilon \right) } { ( \mu ( \mathbf x _ { i } ) \nu ( \mathbf y _ { j } ) ) ^ { \theta } } \leq C _ { 4 } ^ { ' } ( \varepsilon ) .\tag{54}
$$

Thus, a and b defined in Lemma B.2 satisfy

$$
C _ { 3 } ^ { ' } ( \varepsilon ) \leq a \leq b \leq C _ { 4 } ^ { ' } ( \varepsilon ) .\tag{55}
$$

Hence, for any $\delta \in ( 0 , 1 / 2 ]$ , using (52), (53), and (55), the coeficients of (37) and (38) are bounded:

$$
\left( \frac { 1 } { 1 - \delta } + \frac { 4 \sqrt { b } } { a ^ { 2 } \sqrt { s } C _ { 1 } ^ { 3 / 2 } C _ { 2 } ^ { 3 / 2 } m \operatorname* { m i n } _ { i } r _ { i } } \right) \le D _ { 1 } ( \varepsilon ) , \mathrm { ~ a n d ~ } \left( \frac { 1 } { 1 - \delta } + \frac { 4 \sqrt { b } } { a ^ { 2 } \sqrt { s } C _ { 1 } ^ { 3 / 2 } C _ { 2 } ^ { 3 / 2 } n \operatorname* { m i n } _ { j } c _ { j } } \right) \le D _ { 2 } ( \varepsilon ) ,\tag{56}
$$

for some positive constants $D _ { 1 } ( \varepsilon )$ and $D _ { 2 } ( \varepsilon )$

Applying Lemma B.2 with $\delta : = \tau C ( \varepsilon ) \sqrt { \log m / m }$ , where $C ( \varepsilon )$ is the constant associated with (49) and (51) in Definition B.3 and $\delta \leq 1 / 2$ upon choosing $n _ { 0 } ( \varepsilon , \tau )$ properly, there exists a pair of positive vectors $( \alpha ^ { ( \theta ) ^ { \prime } } , \dot { \beta } ^ { ( \theta ) ^ { \prime } } )$ that scales $\mathbf { M } ^ { ( \theta ) }$ to row sums r and column sums c, such that

$$
\frac { | \alpha _ { i } ^ { ( \theta ) ^ { \prime } } - \hat { \alpha } _ { i } ^ { ( \theta ) } | } { \hat { \alpha } _ { i } ^ { ( \theta ) } } = \widetilde { \mathcal { O } } _ { m } ^ { ( \varepsilon ) } \left( \sqrt { \frac { \log m } { m } } \right) , \quad \mathrm { ~ a n d ~ } \quad \frac { | \beta _ { j } ^ { ( \theta ) ^ { \prime } } - \hat { \beta } _ { j } ^ { ( \theta ) } | } { \hat { \beta } _ { j } ^ { ( \theta ) } } = \widetilde { \mathcal { O } } _ { m } ^ { ( \varepsilon ) } \left( \sqrt { \frac { \log m } { m } } \right) ,\tag{57}
$$

for any $i \in [ m ]$ and any $j \in [ n ]$

By Lemma B.1 and the definition of scaling factors, the two pairs of scaling factors $( \alpha ^ { ( \theta ) } , \beta ^ { ( \theta ) } )$ and $( \alpha ^ { ( \theta ) ^ { \prime } } , \beta ^ { ( \theta ) ^ { \prime } } )$ satisfy $\alpha _ { i } ^ { ( \theta ) } \beta _ { j } ^ { ( \theta ) } = { \alpha _ { i } ^ { ( \theta ) } } ^ { \prime } \beta _ { j } ^ { ( \theta ) ^ { \prime } }$ , for any $i \in [ m ]$ and $j \in [ n ]$ . Hence, applying a union bound over all $i \in [ m ]$ and $j \in [ n ]$ under Assumption 2, we have: for any $i \in [ m ]$ and $j \in [ n ]$ ,

$$
\begin{array} { l l } { { W _ { i j } ^ { ( \theta ) } = \alpha _ { i } ^ { ( \theta ) } e ^ { \frac { - \| \mathbf { x } _ { i } - \mathbf { y } _ { j } \| _ { 2 } ^ { 2 } } { \varepsilon } } \beta _ { j } ^ { ( \theta ) } = \alpha _ { i } ^ { ( \theta ) ^ { \prime } } e ^ { \frac { - \| \mathbf { x } _ { i } - \mathbf { y } _ { j } \| _ { 2 } ^ { 2 } } { \varepsilon } } \beta _ { j } ^ { ( \theta ) ^ { \prime } } = \hat { \alpha } _ { i } ^ { ( \theta ) } e ^ { \frac { - \| \mathbf { x } _ { i } - \mathbf { y } _ { j } \| _ { 2 } ^ { 2 } } { \varepsilon } } \hat { \beta } _ { j } ^ { ( \theta ) ^ { \prime } } = \hat { \alpha } _ { i } ^ { ( \theta ) } e ^ { \frac { - \| \mathbf { x } _ { i } - \mathbf { y } _ { j } \| _ { 2 } ^ { 2 } } { \varepsilon } } \hat { \beta } _ { j } ^ { ( \theta ) } ( 1 + \widetilde { \mathcal { O } } _ { m } ^ { ( \varepsilon ) } ( \sqrt { \frac { \log m } { m } } ) ) } }  \\   = \frac { u _ { \varepsilon } ^ { ( \theta ) } ( \mathbf { x } _ { i } ) \mathcal { K } _ { \varepsilon } ( \mathbf { x } _ { i } , \mathbf { y } _ { j } ) v _ { \varepsilon } ^ { ( \theta ) } ( \mathbf { y } _ { j } ) } { \sqrt { Z _ { x } ^ { ( \theta ) } Z _ { y } ^ { ( \theta ) } } } ( 1 + \widetilde { \mathcal { O } } _ { m } ^ { ( \varepsilon ) } ( \sqrt { \frac { \log m } { m } } ) ) = \frac { \mathcal { W } _ { \varepsilon } ^ { ( \theta ) } ( \mathbf { x } _ { i } , \mathbf { y } _ { j } ) } { \sqrt { Z _ { x } ^ { ( \theta ) } Z _ { y } ^ { ( \theta ) } } } ( 1 + \widetilde { \mathcal { O } } _ { m } ^ { ( \varepsilon ) } ( \sqrt  \frac { \log m }  \end{array}\tag{58}
$$

Since ${ \mathscr W } _ { \varepsilon } ^ { ( \theta ) }$ is continuous and bounded above on the compact product manifold $\mathcal { M } _ { x } \times \mathcal { M } _ { y }$ , the multiplicative estimate in (58) implies the additive bound in (13), completing the proof.

## B.4 Auxiliary Lemmas for the Proof of Lemma A.1

Lemma B.4. Let $\mathcal { M } \subset \mathbb { R } ^ { D }$ be a smooth, compact, boundaryless d-dimensional Riemannian manifold, and let $f \in \mathcal { C } ^ { 3 } ( \mathcal { M } )$ . Then there exist $h _ { 0 } ( \mathcal { M } ) > 0$ and $C = C ( \mathcal { M } , f ) > 0$ such that for all $0 < h \leq h _ { 0 } ( \mathcal { M } )$

$$
\operatorname* { s u p } _ { x \in { \mathcal { M } } } \left| { \frac { 1 } { ( \pi h ) ^ { d / 2 } } } \int _ { { \mathcal { M } } } \exp \left( - { \frac { \| x - y \| ^ { 2 } } { h } } \right) f ( y ) d V ( y ) - f ( x ) \right| \ \leq \ C h .\tag{59}
$$

Proof. By [28, Lemma 8] applied to the Gaussian kernel $h ( t ) = e ^ { - t }$ , for which $\textstyle \int _ { \mathbb { R } ^ { d } } h ( \| u \| ^ { 2 } ) d u = \pi ^ { d / 2 }$ and $\textstyle \int _ { \mathbb { R } ^ { d } } u _ { 1 } ^ { 2 } h ( \| u \| ^ { 2 } ) d u = { \frac { 1 } { 2 } } \pi ^ { d / 2 }$ , we have

$$
\frac { 1 } { ( \pi h ) ^ { d / 2 } } \int _ { \mathcal { M } } \exp \biggl ( - \frac { \| x - y \| ^ { 2 } } { h } \biggr ) f ( y ) d V ( y ) = f ( x ) + \frac { h } { 4 } \left[ \omega ( x ) f ( x ) + \Delta _ { \mathcal { M } } f ( x ) \right] + O ( h ^ { 2 } ) ,
$$

uniformly in $x \in \mathcal { M }$ , where ω is a potential term depending on the embedding of M and $\Delta _ { { \scriptscriptstyle M } }$ is the Laplace–Beltrami operator. (The estimate in [28] is stated uniformly over points at distance more than $\varepsilon ^ { \gamma }$ from ∂M for any fixed $0 < \gamma < 1 / 2 ;$ since $\partial \mathcal { M } = \emptyset$ this is all of $\mathcal { M } . \big )$ Since ω, f and $\Delta _ { \mathcal { M } } f$ are continuous on the compact manifold M, the first-order term is bounded by a constant multiple of h uniformly in $x _ { \mathrm { { i } } }$ , and the claim follows. □

Lemma B.5. Let $\mathcal { M } \subset \mathbb { R } ^ { D }$ be a smooth, compact, boundaryless d-dimensional Riemannian manifold isometrically embedded in $\mathbb { R } ^ { D }$ , let $\mu \in \mathcal { C } ^ { 3 } ( \mathcal { M } )$ be a positive probability density on M, and let $\mathbf { x } _ { 1 } , \ldots , \mathbf { x } _ { n } \stackrel { \mathrm { i i d } } { \sim } \mu .$ Define

$$
{ \hat { p } } ( x ) : = { \frac { 1 } { n } } \sum _ { j = 1 } ^ { n } { \frac { 1 } { ( \pi h ) ^ { d / 2 } } } \exp \left( - { \frac { \| x - \mathbf { x } _ { j } \| ^ { 2 } } { h } } \right) .\tag{60}
$$

Then there exist $h _ { 0 } ( \mathcal { M } , \mu ) > 0$ and $C = C ( \mathcal { M } , \mu ) > 0$ such that for all $n \geq 2 h _ { 0 } ( \mathcal { M } , \mu ) ^ { - 1 / 2 }$ and all fixed h with $4 n ^ { - 2 } \leq h \leq h _ { 0 } ( \mathcal { M } , \mu )$ , with probability at least $1 - n ^ { - 2 }$ ，

$$
\operatorname* { s u p } _ { x \in \mathcal { M } } \left| \hat { p } ( x ) - \mu ( x ) \right| \ \leq \ C \left( h + \sqrt { \frac { \log n } { n h ^ { d } } } \right) .\tag{61}
$$

Proof. We decompose the error via the triangle inequality:

$$
\operatorname* { s u p } _ { x \in \mathcal { M } } \big | \hat { p } ( x ) - \mu ( x ) \big | \ \leq \ \operatorname* { s u p } _ { x \in \mathcal { M } } \big | \hat { p } ( x ) - \mathbb { E } [ \hat { p } ( x ) ] \big | \ + \ \operatorname* { s u p } _ { x \in \mathcal { M } } \big | \mathbb { E } [ \hat { p } ( x ) ] - \mu ( x ) \big | .\tag{62}
$$

Bias term. Since $\mathbf { x } _ { j } \overset { \mathrm { i i d } } { \sim } \mu$ , the expected value of ${ \hat { p } } ( x )$ is given by

$$
\mathbb { E } [ \hat { p } ( x ) ] = \frac { 1 } { ( \pi h ) ^ { d / 2 } } \int _ { \mathcal { M } } \exp \left( - \frac { \| x - y \| ^ { 2 } } { h } \right) \mu ( y ) d V ( y ) ,
$$

so applying Lemma B.4 with $f = \mu$ , there exist constants $C _ { 1 } = C _ { 1 } ( \mathcal { M } , \mu ) > 0$ and $h _ { 1 } ( \mathcal { M } , \mu ) > 0$ such that for all $0 < h \leq h _ { 1 }$ ，

$$
\operatorname* { s u p } _ { x \in \mathcal { M } } \left| \mathbb { E } [ \hat { p } ( x ) ] - \mu ( x ) \right| \leq C _ { 1 } h .\tag{63}
$$

Since this bound is deterministic, it holds with probability 1.

Variance term. Set $\eta : = \sqrt { h } / 2$ , so that the kernel $k _ { \eta } ( x , y ) = \exp \bigl ( - \| x - y \| ^ { 2 } / ( 4 \eta ^ { 2 } ) \bigr )$ of [35, Section 3.1] coincides with $\exp { \left( - \| x - y \| ^ { 2 } / h \right) } . \mathrm { ~ B y ~ } [ 3 5 ,$ , Proposition 2, equation (64)], applied with bandwidth $\eta = \sqrt { h } / 2$ to the function class ${ \mathcal { K } } _ { \eta } = \{ k _ { \eta } ( x , \cdot ) : x \in { \mathcal { M } } \}$ of [35, Definition $6 ]$ , there exist $h _ { 2 } ( \mathcal { M } , \mu ) > 0$ and $C _ { \mathrm { g c } } > 0$ such that for all $0 < h \leq h _ { 2 } ( { \mathcal { M } } , \mu )$ , with probability at least $1 - n ^ { \bar { - } 2 }$

$$
\operatorname* { s u p } _ { x \in \mathcal { M } } \left| \frac { 1 } { n } \sum _ { j = 1 } ^ { n } \exp \left( - \frac { \| x - \mathbf { x } _ { j } \| ^ { 2 } } { h } \right) - \int _ { \mathcal { M } } \exp \left( - \frac { \| x - y \| ^ { 2 } } { h } \right) \mu ( y ) d V ( y ) \right| \leq \ \frac { C _ { \operatorname { g c } } } { \sqrt { n } } \left( \sqrt { \log ( 2 / \sqrt { h } ) } + \sqrt { \log n } \right) ,\tag{64}
$$

where $C _ { \mathrm { g c } }$ depends only on $d ,$ the diameter of M, and the uniform upper and lower bounds of $\mu ,$ all of which are finite since $\mathcal { M }$ is compact and $\mu$ is positive and continuous.

The condition $h \geq 4 n ^ { - 2 }$ gives $\eta ~ = ~ \sqrt { h } / 2 ~ \ge ~ 1 / n$ , hence $\log ( 2 / { \sqrt { h } } ) = \log ( 1 / \eta ) \leq \log n ;$ , so that the right-hand side of (64) is at most $2 C _ { \mathrm { g c } } \sqrt { \log n / n }$ . Dividing (64) by $( \pi h ) ^ { d / 2 }$ , the left-hand side becomes s $\begin{array} { r } { \operatorname * { l p } _ { x \in \mathcal { M } } \left| \hat { p } ( x ) - \mathbb { E } [ \hat { p } ( x ) ] \right. } \end{array}$ , and we obtain

$$
\operatorname* { s u p } _ { x \in \mathcal { M } } \left. \hat { p } ( x ) - \mathbb { E } [ \hat { p } ( x ) ] \right. \ \leq \ C _ { 3 } \sqrt { \frac { \log n } { n h ^ { d } } } , \qquad C _ { 3 } : = 2 \pi ^ { - d / 2 } C _ { \mathrm { g c } } .\tag{65}
$$

Substituting (63) and (65) into (62) and setting $h _ { 0 } : = \operatorname* { m i n } \{ h _ { 1 } , h _ { 2 } \}$ and $C : = \operatorname* { m a x } \{ C _ { 1 } , C _ { 3 } \}$ yields (61).

## B.5 Proof of Lemma A.1

Proof. Let $h _ { 0 } ^ { \star }$ and $C ^ { \star }$ be the constants in Lemma B.5, both depending only on $\mathcal { M } _ { x }$ and $\mu .$ . For a fixed bandwidth $h ,$ let $\hat { p }$ denote the kernel density estimator in (60) constructed from all m samples $\mathbf { x } _ { 1 } , \ldots , \mathbf { x } _ { m }$ and for each $i \in [ m ]$ , we define

$$
\hat { p } _ { - i } ( \mathbf x _ { i } ) : = \frac { 1 } { m - 1 } \sum _ { j \neq i } ^ { m } \frac { 1 } { ( \pi h ) ^ { d / 2 } } K _ { h } ( \mathbf x _ { i } , \mathbf x _ { j } ) = \frac { \hat { q } ( \mathbf x _ { i } ) } { ( \pi h ) ^ { d / 2 } } , \quad \bar { Z } = \frac { 1 } { m } \sum _ { i = 1 } ^ { m } \hat { p } _ { - i } ( \mathbf x _ { i } ) .\tag{66}
$$

Note by definition of $\hat { f } ( \mathbf { x } _ { i } )$ in (15), $\begin{array} { r } { \hat { f } ( \mathbf x _ { i } ) = { \frac { \hat { p } _ { - i } ( \mathbf x _ { i } ) } { \overline { { Z } } } } } \end{array}$ . We also note that since $\mu$ is positive and continuous on the compact manifold $\mathcal { M } _ { x }$ , there exist $0 < c _ { 1 } \bar { \leq } c _ { 2 } < \infty$ with $c _ { 1 } \leq \mu ( \mathbf { x } ) \leq c _ { 2 } , \forall \mathbf { x } \in \mathcal { M } _ { x }$

We take $m _ { 0 } \geq 2$ large enough that, for all $m \geq m _ { 0 } \colon \mathrm { ( i ) }$ (κ log $m / m ) ^ { 1 / d } \leq h _ { 0 }$ , so that (22) is nonempty; and (ii) (κ log $m / m ) ^ { 1 / d } \geq 4 m ^ { - 2 }$ , so that (22) implies 4 $n ^ { - 2 } \le h \le h _ { 0 } \le h _ { 0 } ^ { \star }$ and Lemma B.5 applies with sample size m. Each of these holds for all suficiently large m. The constants $h _ { 0 } \leq \operatorname* { m i n } \{ 1 , h _ { 0 } ^ { \star } \}$ and κ are fixed at the end of Step ${ 3 ; }$ the constants $C _ { 1 } , C _ { 2 } , C _ { 3 }$ introduced below depend only on $C ^ { \star } , c _ { 1 } , c _ { 2 }$ and $d ,$ and in particular not on $h _ { 0 }$ or $\kappa .$

Step 1: We first bound $\left| \hat { p } _ { - i } ( \mathbf { x } _ { i } ) - \mu ( \mathbf { x } _ { i } ) \right|$

Since $K _ { h } ( \mathbf { x } _ { i } , \mathbf { x } _ { i } ) = 1$ , we note $\hat { p } ( \mathbf { x } _ { i } )$ and $\hat { p } _ { - i } ( { \bf x } _ { i } )$ are related by

$$
\hat { p } _ { - i } ( \mathbf { x } _ { i } ) = \frac { m } { m - 1 } \hat { p } ( \mathbf { x } _ { i } ) - \frac { 1 } { ( m - 1 ) ( \pi h ) ^ { d / 2 } } , \quad \forall i \in [ m ] .\tag{67}
$$

Using (67), we obtain, for any $i \in [ m ]$ :

$$
\begin{array} { r l } & { \displaystyle \left| \hat { p } _ { - i } ( \mathbf { x } _ { i } ) - \mu ( \mathbf { x } _ { i } ) \right| = \left| \frac { m } { m - 1 } \big ( \hat { p } ( \mathbf { x } _ { i } ) - \mu ( \mathbf { x } _ { i } ) \big ) + \frac { \mu ( \mathbf { x } _ { i } ) } { m - 1 } - \frac { 1 } { ( m - 1 ) \pi ^ { d / 2 } h ^ { d / 2 } } \right| } \\ & { \displaystyle \leq \frac { m } { m - 1 } \big | \hat { p } ( \mathbf { x } _ { i } ) - \mu ( \mathbf { x } _ { i } ) \big | + \frac { c _ { 2 } } { m - 1 } + \frac { 1 } { ( m - 1 ) \pi ^ { d / 2 } h ^ { d / 2 } } \leq 2 \big | \hat { p } ( \mathbf { x } _ { i } ) - \mu ( \mathbf { x } _ { i } ) \big | + \frac { 2 c _ { 2 } } { m } + \frac { 2 } { \pi ^ { d / 2 } m h ^ { d / 2 } } } \\ & { \displaystyle \leq 2 \big | \hat { p } ( \mathbf { x } _ { i } ) - \mu ( \mathbf { x } _ { i } ) \big | + ( 2 c _ { 2 } + 2 \pi ^ { - d / 2 } ) \sqrt { \frac { \log m } { m h ^ { d } } } . } \end{array}\tag{68}
$$

where the first line uses (67), the second line follows from the triangle inequality and the bounds $m / ( m - 1 ) \leq$ 2 and $1 / ( m - 1 ) \leq 2 / m$ , both valid for $m \geq m _ { 0 } \geq 2$ , and the third line from $\begin{array} { r } { \frac { 1 } { m h ^ { d / 2 } } \leq \sqrt { \frac { \log m } { m h ^ { d } } } } \end{array}$ and $\begin{array} { r } { \frac { 1 } { m } \leq \sqrt { \frac { \log m } { m h ^ { d } } } } \end{array}$ valid for $m \ge m _ { 0 } \ge 2$ and $h \leq h _ { 0 } \leq 1$

By (ii) the bandwidth h lies in the admissible range of Lemma B.5, so with probability at least $1 - m ^ { - 2 }$

$$
\operatorname* { s u p } _ { \mathbf { x } \in \mathcal { M } _ { x } } \left| \hat { p } ( \mathbf { x } ) - \mu ( \mathbf { x } ) \right| \leq C ^ { \star } \left( h + \sqrt { \frac { \log m } { m h ^ { d } } } \right) .\tag{69}
$$

Call this event A. Since (69) is uniform over $\mathcal { M } _ { x }$ , it holds simultaneously at every sample point $\mathbf { x } _ { i }$ . Combining (68) and (69), we have: with probability at least $1 - m ^ { - 2 }$ ，

$$
\operatorname* { m a x } _ { i \in [ m ] } \Big | \hat { p } _ { - i } \big ( \mathbf { x } _ { i } \big ) - \mu \big ( \mathbf { x } _ { i } \big ) \Big | \leq C _ { 1 } \left( h + \sqrt { \frac { \log m } { m h ^ { d } } } \right) , \qquad C _ { 1 } : = 2 C ^ { \star } + 2 c _ { 2 } + 2 \pi ^ { - d / 2 } .\tag{70}
$$

Step 2: we next bound $\left| \bar { Z } - ( \| \mu \| _ { L ^ { 2 } } ) ^ { 2 } \right|$

$$
\bar { Z } - ( \| \mu \| _ { L ^ { 2 } } ) ^ { 2 } = \frac { 1 } { m } \sum _ { i = 1 } ^ { m } \bigl ( \hat { p } _ { - i } ( \mathbf { x } _ { i } ) - \mu ( \mathbf { x } _ { i } ) \bigr ) + \left( \frac { 1 } { m } \sum _ { i = 1 } ^ { m } \mu ( \mathbf { x } _ { i } ) - ( \| \mu \| _ { L ^ { 2 } } ) ^ { 2 } \right) .\tag{71}
$$

On event A, the first term is bounded by $C _ { 1 } ( h + \sqrt { \log m / ( m h ^ { d } ) } )$ by (70). For the second, since $\mathbb { E } [ \mu ( \mathbf { x } ) ] =$ $\begin{array} { r } { \int _ { \mathcal { M } _ { x } } \mu ( \mathbf { x } ) \cdot \mu ( \mathbf { x } ) d \boldsymbol { \omega } ( \mathbf { x } ) = ( \| \mu \| _ { L ^ { 2 } } ) ^ { 2 } } \end{array}$ and $\mathbf { x } _ { 1 } , \ldots , \mathbf { x } _ { m }$ are i.i.d. with $\mu ( \mathbf { x } _ { i } ) \in [ c _ { 1 } , c _ { 2 } ]$ , Hoefding’s inequality gives, M<sub>for</sub> <sub>any</sub> $t > 0$

$$
\mathbb { P } \left( \left| \frac { 1 } { m } \sum _ { i = 1 } ^ { m } \mu ( \mathbf { x } _ { i } ) - ( \| \mu \| _ { L ^ { 2 } } ) ^ { 2 } \right| \geq t \right) \ \leq \ 2 \exp \left( - \frac { 2 m t ^ { 2 } } { ( c _ { 2 } - c _ { 1 } ) ^ { 2 } } \right) .\tag{72}
$$

Choosing $t = ( c _ { 2 } - c _ { 1 } ) \sqrt { \log ( 2 m ^ { 2 } ) / ( 2 m ) }$ makes the right-hand side of (72) equal to $m ^ { - 2 }$ , and with probability at least $1 - m ^ { - 2 }$

$$
\left| \frac { 1 } { m } \sum _ { i = 1 } ^ { m } \mu ( \mathbf { x } _ { i } ) - ( \| \mu \| _ { L ^ { 2 } } ) ^ { 2 } \right| \leq ( c _ { 2 } - c _ { 1 } ) \sqrt { \frac { \log ( 2 m ^ { 2 } ) } { ( 2 m ) } } \leq 2 ( c _ { 2 } - c _ { 1 } ) \sqrt { \frac { \log m } { m } } \leq 2 ( c _ { 2 } - c _ { 1 } ) \sqrt { \frac { \log m } { m h ^ { d } } } ,\tag{73}
$$

where we use $m \ge m _ { 0 } \ge 2$ , and $h \leq h _ { 0 } \leq 1$ . Call this event B and define $C _ { 2 } : = 2 ( c _ { 2 } - c _ { 1 } )$

By the union bound $\mathbb { P } ( A \cap B ) \geq 1 - 2 m ^ { - 2 }$ , and on ${ \mathcal { A } } \cap B ,$ , combining (71)–(73),

$$
\big | \bar { Z } - ( \| \mu \| _ { L ^ { 2 } } ) ^ { 2 } \big | \le C _ { 3 } \left( h + \sqrt { \frac { \log m } { m h ^ { d } } } \right) , \qquad C _ { 3 } : = C _ { 1 } + C _ { 2 } .\tag{74}
$$

Step 3: We next bound $\bar { Z }$ away from zero. Under (22) we have $h \leq h _ { 0 }$ and $m h ^ { d } \geq$ κ log m, hence

$$
h + \sqrt { \frac { \log m } { m h ^ { d } } } \leq h _ { 0 } + \frac { 1 } { \sqrt { \kappa } } .\tag{75}
$$

Choosing

$$
h _ { 0 } : = \operatorname* { m i n } \left\{ 1 , \ : h _ { 0 } ^ { \star } , \ : \frac { ( \| \mu \| _ { L ^ { 2 } } ) ^ { 2 } } { 4 C _ { 3 } } \right\} , \qquad \kappa : = \left( \frac { 4 C _ { 3 } } { ( \| \mu \| _ { L ^ { 2 } } ) ^ { 2 } } \right) ^ { 2 } ,\tag{76}
$$

both of which depend only on $\mathcal { M } _ { x }$ and $\mu ,$ inequalities (75) and (74) give $| \bar { Z } - ( \| \mu \| _ { L ^ { 2 } } ) ^ { 2 } | \leq ( \| \mu \| _ { L ^ { 2 } } ) ^ { 2 } / 2$ and therefore, on A ∩ B:

$$
\bar { Z } \geq \frac { ( \| \mu \| _ { L ^ { 2 } } ) ^ { 2 } } { 2 } .\tag{77}
$$

$$
\begin{array} { r l } & { \left| \hat { f } ( \mathbf { x } _ { \mathrm { i } } ) - \frac { \mu ( \mathbf { x } _ { \mathrm { i } } ) } { ( \| \mu \| _ { L ^ { 2 } } ) ^ { 2 } } \right| = \left| \frac { \hat { p } - i ( \mathbf { x } _ { \mathrm { i } } ) } { \overline { { Z } } } - \frac { \mu ( \mathbf { x } _ { \mathrm { i } } ) } { ( \| \mu \| _ { L ^ { 2 } } ) ^ { 2 } } \right| = \left| \frac { ( \| \mu \| _ { L ^ { 2 } } ) ^ { 2 } \big ( \hat { p } _ { - \mathrm { i } } ( \mathbf { x } _ { \mathrm { i } } ) - \mu ( \mathbf { x } _ { \mathrm { i } } ) \big ) - \mu ( \mathbf { x } _ { \mathrm { i } } ) \big ( \overline { { Z } } - ( \| \mu \| _ { L ^ { 2 } } ) ^ { 2 } \big ) } { ( \| \mu \| _ { L ^ { 2 } } ) ^ { 2 } \overline { { Z } } } \right| } \\ & { \leq \frac { ( \| \mu \| _ { L ^ { 2 } } ) ^ { 2 } \left| \hat { p } _ { - \mathrm { i } } ( \mathbf { x } _ { \mathrm { i } } ) - \mu ( \mathbf { x } _ { \mathrm { i } } ) \right| + \mu ( \mathbf { x } _ { \mathrm { i } } ) \left| \bar { Z } - ( \| \mu \| _ { L ^ { 2 } } ) ^ { 2 } \right| } { ( \| \mu \| _ { L ^ { 2 } } ) ^ { 2 } \overline { { Z } } } \leq \frac { ( \| \mu \| _ { L ^ { 2 } } ) ^ { 2 } C _ { 1 } \left( h + \sqrt { \frac { \log m } { m h ^ { d } } } \right) + c _ { 2 } C _ { 3 } \left( h + \sqrt { \frac { \log m } { m h ^ { d } } } \right) } { ( \| \mu \| _ { L ^ { 2 } } ) ^ { 2 } \overline { { Z } } } } \\ &  \leq \frac  2 \left[ ( \| \mu \| _ { L ^ { 2 } } ) ^ { 2 } C _ { 1 } + c _ { 2 } C _ { 3 } \right. \end{array}
$$

where we use triangle inequality, (70) and (74) in the second line, and (77) in the last line. This concludes the proof. □

## B.6 Proof of Theorem 3.4

Proof. Let $\alpha ^ { ( \theta ) }$ and $\beta ^ { ( \theta ) }$ be the scaling factors in Algorithm 1 that diagonally scale $\mathbf { M } ^ { ( \theta ) }$ , defined in (4), to satisfy the row and column constraints in (5), for fixed $\theta \in [ 0 , 1 ]$ , where <sup>ˆ</sup>f and gˆ are the mean-normalized density estimates in (15) applied to X and Y with bandwidths $h _ { 1 }$ and $h _ { 2 }$ . Define approximate scaling factors $\hat { \pmb { \alpha } } ^ { ( \theta ) }$ and $\hat { \boldsymbol { \beta } } ^ { ( \theta ) }$ as

$$
\hat { \alpha } _ { i } ^ { ( \theta ) } = \frac { u _ { \varepsilon } ^ { ( \theta ) } ( \mathbf { x } _ { i } ) } { ( \| \mu \| _ { L ^ { 2 } } ) ^ { \theta } \sqrt { Z _ { x } ^ { ( \theta ) } } } , \quad \mathrm { ~ a n d ~ } \quad \hat { \beta } _ { j } ^ { ( \theta ) } = \frac { v _ { \varepsilon } ^ { ( \theta ) } ( \mathbf { y } _ { j } ) } { ( \| \nu \| _ { L ^ { 2 } } ) ^ { \theta } \sqrt { Z _ { y } ^ { ( \theta ) } } } ,\tag{78}
$$

for any $i \in [ m ]$ and any $j \in [ n ]$ , where $u _ { \varepsilon } ^ { ( \theta ) }$ and $v _ { \varepsilon } ^ { ( \theta ) }$ are defined in (12). For brevity we write

$$
\rho _ { 1 } : = h _ { 1 } + \sqrt { \frac { \log m } { m h _ { 1 } ^ { d _ { 1 } } } } , \qquad \rho _ { 2 } : = h _ { 2 } + \sqrt { \frac { \log n } { n h _ { 2 } ^ { d _ { 2 } } } } .\tag{79}
$$

The density-estimation event. Let E denote the event on which the conclusion of Lemma A.1, applied to X with bandwidth $h _ { 1 }$ and to Y with bandwidth $h _ { 2 }$ , holds simultaneously:

$$
\left. \hat { f } ( \mathbf { x } _ { i } ) - \frac { \mu ( \mathbf { x } _ { i } ) } { ( \| \mu \| _ { L ^ { 2 } } ) ^ { 2 } } \right. \leq C _ { 0 } ^ { \alpha } \rho _ { 1 } \quad \mathrm { a n d } \quad \left. \hat { g } ( \mathbf { y } _ { j } ) - \frac { \nu ( \mathbf { y } _ { j } ) } { ( \| \nu \| _ { L ^ { 2 } } ) ^ { 2 } } \right. \leq C _ { 0 } ^ { y } \rho _ { 2 } , \qquad \forall i \in [ m ] , \forall j \in [ n ] ,\tag{80}
$$

where $C _ { 0 } ^ { x } = C _ { 0 } ( \mathcal { M } _ { x } , \mu )$ and $C _ { 0 } ^ { y } = C _ { 0 } ( \mathcal { M } _ { y } , \nu )$ are defined in Lemma A.1. Applying the union bound over each dataset and under Assumption $2 \ ( m \geq n ^ { \gamma }$ and $\gamma \leq 1 )$ ), we have:

$$
\mathbb { P } ( \mathcal { E } ) \geq 1 - 2 m ^ { - 2 } - 2 n ^ { - 2 } \geq 1 - 4 n ^ { - 2 \gamma } .\tag{81}
$$

We next show that on $\boldsymbol { \mathcal { E } } , \hat { f } ( \mathbf { x } )$ and $\hat { g } ( \mathbf { y } )$ are bounded away from zero and infinity for any x and y.

Since $\mu$ and ν are positive and continuous on the compact manifolds $\mathcal { M } _ { x }$ and $\mathcal { M } _ { y }$ , there exist constants $0 < c _ { 1 } \le c _ { 2 } < \infty$ and $0 < c _ { 3 } \le c _ { 4 } <$ ∞ with

$$
c _ { 1 } \leq \mu ( \mathbf { x } ) \leq c _ { 2 } \quad \forall \mathbf { x } \in \mathcal { M } _ { x } , \qquad c _ { 3 } \leq \nu ( \mathbf { y } ) \leq c _ { 4 } \quad \forall \mathbf { y } \in \mathcal { M } _ { y } .\tag{82}
$$

Writing $B _ { \mu } ~ : = ~ ( \| \mu \| _ { L ^ { 2 } } ) ^ { 2 }$ and $B _ { \nu } : = ( \| \nu \| _ { L ^ { 2 } } ) ^ { 2 }$ , we have $\mu ( \mathbf { x } _ { i } ) / B _ { \mu } \ \in \ [ c _ { 1 } / B _ { \mu } , c _ { 2 } / B _ { \mu } ]$ and $\nu ( { \bf y } _ { j } ) / B _ { \nu } \in$ $[ c _ { 3 } / B _ { \nu } , c _ { 4 } / B _ { \nu } ]$ . Under (16), we have $\rho _ { 1 } \leq h _ { 1 } ^ { \prime } + \kappa _ { 1 } ^ { - 1 / 2 }$ , and $\rho _ { 2 } \leq h _ { 2 } ^ { \prime } + \kappa _ { 2 } ^ { - 1 / 2 }$ . Choosing

$$
h _ { 1 } ^ { \prime } \leq \frac { c _ { 1 } } { 4 C _ { 0 } ^ { x } B _ { \mu } } , \quad \kappa _ { 1 } \geq \left( \frac { 4 C _ { 0 } ^ { x } B _ { \mu } } { c _ { 1 } } \right) ^ { 2 } , \qquad h _ { 2 } ^ { \prime } \leq \frac { c _ { 3 } } { 4 C _ { 0 } ^ { y } B _ { \nu } } , \quad \kappa _ { 2 } \geq \left( \frac { 4 C _ { 0 } ^ { y } B _ { \nu } } { c _ { 3 } } \right) ^ { 2 } ,\tag{83}
$$

all of which depend only on $\mathcal { M } _ { x } , \mathcal { M } _ { y } , \mu$ and ν, gives ${ \cal C } _ { 0 } ^ { x } \rho _ { 1 } \le c _ { 1 } / ( 2 B _ { \mu } )$ and $C _ { 0 } ^ { y } \rho _ { 2 } \le c _ { 3 } / ( 2 B _ { \nu } )$ . Combining with (80), on $\mathcal { E }$ we obtain

$$
\hat { f } ( \mathbf { x } _ { i } ) \in I _ { x } : = \left[ \frac { c _ { 1 } } { 2 B _ { \mu } } , \frac { 2 c _ { 2 } } { B _ { \mu } } \right] , \qquad \hat { g } ( \mathbf { y } _ { j } ) \in I _ { y } : = \left[ \frac { c _ { 3 } } { 2 B _ { \nu } } , \frac { 2 c _ { 4 } } { B _ { \nu } } \right] , \qquad \forall i \in [ m ] , \forall j \in [ n ] ,\tag{84}
$$

and the intervals $I _ { x } , I _ { y }$ also contain the targets $\mu ( \mathbf { x } _ { i } ) / B _ { \mu }$ and $\nu ( \mathbf { y } _ { j } ) / B _ { \nu }$ . In particular $I _ { x }$ and $I _ { y }$ are fixed compact subintervals of $( 0 , \infty )$ , independent of $m , n , h _ { 1 }$ and $h _ { 2 }$

The map $\phi ( z ) = z ^ { - \theta }$ is continuously diferentiable on $( 0 , \infty )$ with $\phi ^ { \prime } ( z ) ~ = ~ - \theta z ^ { - \theta - 1 }$ . For $\theta \in [ 0 , 1 ]$ using (84), we have:

$$
L _ { x } : = \operatorname* { s u p } _ { z \in I _ { x } } | \phi ^ { \prime } ( z ) | \leq \left( \frac { 2 B _ { \mu } } { c _ { 1 } } \right) ^ { 2 } , \qquad L _ { y } : = \operatorname* { s u p } _ { z \in I _ { y } } | \phi ^ { \prime } ( z ) | \leq \left( \frac { 2 B _ { \nu } } { c _ { 3 } } \right) ^ { 2 } .\tag{85}
$$

Applying the mean value theorem, we have: on $\mathcal { E } ,$ for any $i \in [ m ]$

$$
\left| \frac { 1 } { ( \hat { f } ( \mathbf { x } _ { i } ) ) ^ { \theta } } - \frac { ( \| \mu \| _ { L ^ { 2 } } ) ^ { 2 \theta } } { ( \mu ( \mathbf { x } _ { i } ) ) ^ { \theta } } \right| = \left| \phi \big ( \hat { f } ( \mathbf { x } _ { i } ) \big ) - \phi \bigg ( \frac { \mu ( \mathbf { x } _ { i } ) } { B _ { \mu } } \bigg ) \right| = \left| \phi ^ { \prime } ( \xi _ { i } ) \right| \cdot \left| \hat { f } ( \mathbf { x } _ { i } ) - \frac { \mu ( \mathbf { x } _ { i } ) } { B _ { \mu } } \right| \leq L _ { x } C _ { 0 } ^ { x } \rho _ { 1 } ,\tag{86}
$$

where $\xi _ { i }$ lies between $\hat { f } ( \mathbf { x } _ { i } )$ and $\mu ( \mathbf { x } _ { i } ) / B _ { \mu }$ , hence in $I _ { x }$ by (84). Equivalently, we have:

$$
\frac { 1 } { ( \hat { f } ( \mathbf x _ { i } ) ) ^ { \theta } } = \frac { ( \| \mu \| _ { L ^ { 2 } } ) ^ { 2 \theta } } { ( \mu ( \mathbf x _ { i } ) ) ^ { \theta } } + \mathcal O ( \rho _ { 1 } ) .\tag{87}
$$

An analogous derivation gives, for any $j \in [ n ]$

$$
\frac { 1 } { ( \hat { g } ( \mathbf y _ { j } ) ) ^ { \theta } } = \frac { ( \| \nu \| _ { L ^ { 2 } } ) ^ { 2 \theta } } { ( \nu ( \mathbf y _ { j } ) ) ^ { \theta } } + \mathcal { O } ( \rho _ { 2 } ) .\tag{88}
$$

Both bounds hold simultaneously for all $i \in [ m ]$ and $j \in [ n ]$ , with absorbed constants depending only on $\mathcal { M } _ { x } , \mathcal { M } _ { y } , \mu , \nu$ and θ.

Bounding $S ^ { ( \theta ) }$ . Applying (87) and then Hoefding’s inequality, we have on $\mathcal { E }$

$$
\begin{array} { l } { \displaystyle \frac { 1 } { m } \sum _ { i = 1 } ^ { m } \frac { 1 } { ( \hat { f } ( \mathbf { x } _ { i } ) ) ^ { \theta } } = \frac { 1 } { m } \sum _ { i = 1 } ^ { m } \frac { ( \| \mu \| _ { L ^ { 2 } } ) ^ { 2 \theta } } { ( \mu ( \mathbf { x } _ { i } ) ) ^ { \theta } } + \mathcal { O } ( \rho _ { 1 } ) = ( \| \mu \| _ { L ^ { 2 } } ) ^ { 2 \theta } \int _ { \mathcal { M } _ { z } } \frac { 1 } { ( \mu ( \mathbf { x } ) ) ^ { \theta } } \mu ( \mathbf { x } ) d \omega ( \mathbf { x } ) + \widetilde { \mathcal { O } } _ { m } \left( \sqrt { \frac { \log m } { m } } \right) + \mathcal { O } ( \rho _ { 1 } ) } \\ { \displaystyle \qquad = \ ( \| \mu \| _ { L ^ { 2 } } ) ^ { 2 \theta } Z _ { _ { z } } ^ { ( \theta ) } + \widetilde { \mathcal { O } } _ { m } ( \rho _ { 1 } ) , } \end{array}\tag{89}
$$

where in the last step we use $\sqrt { \log m / m } \leq \sqrt { \log m / ( m h _ { 1 } ^ { d _ { 1 } } ) } \leq \rho _ { 1 }$ , valid since $h _ { 1 } \leq h _ { 1 } ^ { \prime } \leq 1$ , and absorb the term $\mathcal { O } ( \rho _ { 1 } )$ , a deterministic bound on E, into $\widetilde { \mathcal { O } } _ { m } ( \rho _ { 1 } )$ . Similarly,

$$
\frac { 1 } { n } \sum _ { j = 1 } ^ { n } \frac { 1 } { ( \hat { g } ( \mathbf { y } _ { j } ) ) ^ { \theta } } = ( \| \nu \| _ { L ^ { 2 } } ) ^ { 2 \theta } Z _ { y } ^ { ( \theta ) } + \widetilde { \mathcal { O } } _ { n } ( \rho _ { 2 } ) .\tag{90}
$$

Combining (89) and (90), we have

$$
\begin{array} { r l } & { | S ^ { ( \theta ) } - \frac { ( \| \mu \| _ { L ^ { 2 } } ) ^ { 2 \theta } Z _ { x } ^ { ( \theta ) } } { ( \| \nu \| _ { L ^ { 2 } } ) ^ { 2 \theta } Z _ { y } ^ { ( \theta ) } } | = | \frac { 1 } { n } \sum _ { i = 1 } ^ { m } ( \hat { f } ( \mathbf { x } _ { i } ) ) ^ { - \theta } - \frac { ( \| \mu \| _ { L ^ { 2 } } ) ^ { 2 \theta } Z _ { x } ^ { ( \theta ) } } { ( \| \nu \| _ { L ^ { 2 } } ) ^ { 2 \theta } Z _ { y } ^ { ( \theta ) } } | } \\ &  = | \frac { ( \| \nu \| _ { L ^ { 2 } } ) ^ { 2 \theta } Z _ { y } ^ { ( \theta ) } ( \frac { 1 } { m } \sum _ { i = 1 } ^ { m } ( \hat { f } ( \mathbf { x } _ { i } ) ) ^ { - \theta } - ( \| \mu \| _ { L ^ { 2 } } ) ^ { 2 \theta } Z _ { x } ^ { ( \theta ) } ) - ( \| \mu \| _ { L ^ { 2 } } ) ^ { 2 \theta } Z _ { x } ^ { ( \theta ) } ( \frac { 1 } { n } \sum _ { j = 1 } ^ { n } ( \hat { g } ( \mathbf { y } _ { j } ) ) ^ { - \theta } - ( \| \nu \| _ { L ^ { 2 } } ) ^ { 2 \theta } Z _ { y } ^ { ( \theta ) } ) }  ( \| \nu \| _ { L ^ { 2 } } ) ^ { 2 \theta } Z _ { y } ^ { ( \theta ) } \cdot \frac { 1 } { n } \sum _ { j = 1 } ^ { n } ( \hat { g } ( \mathbf { y } _ { j } ) ) ^ { - \theta } \frac { ( 1 ) } { n } \sum _ { j = 1 } ^ { n } ( \hat { g } ( \mathbf { y } _ { j } ) ) ^ { - \theta } - ( \| \nu \| _ { L ^ { 2 } }  \end{array}\tag{91}
$$

where the third line uses the triangle inequality, and the last uses (89) and (90) together with the fact that on $\mathcal { E }$ the average $\begin{array} { r } { \frac { 1 } { n } \sum _ { j = 1 } ^ { n } ( \hat { g } ( \mathbf { y } _ { j } ) ) ^ { - \theta } } \end{array}$ is bounded away from zero by a constant depending only on $\mathcal { M } _ { y }$ and $\nu ,$ by (84) and (88). An analogous derivation gives

$$
\left| \frac { 1 } { S ^ { ( \theta ) } } - \frac { ( \| \nu \| _ { L ^ { 2 } } ) ^ { 2 \theta } Z _ { y } ^ { ( \theta ) } } { ( \| \mu \| _ { L ^ { 2 } } ) ^ { 2 \theta } Z _ { x } ^ { ( \theta ) } } \right| = \widetilde { \mathcal { O } } _ { m , n } ( \rho _ { 1 } + \rho _ { 2 } ) .\tag{92}
$$

Approximate scaling. We next show that $\hat { \pmb { \alpha } } ^ { ( \theta ) }$ and $\hat { \boldsymbol { \beta } } ^ { ( \theta ) }$ approximately scale $\mathbf { M } ^ { ( \theta ) }$ to have prescribed row sums r and column sums c. Recall that, from (5),

$$
r _ { i } = \frac { n } { ( \hat { f } ( \mathbf { x } _ { i } ) ) ^ { \theta } \sqrt { S ^ { ( \theta ) } } } , \quad \mathrm { a n d } \quad c _ { j } = \frac { m \sqrt { S ^ { ( \theta ) } } } { ( \hat { g } ( \mathbf { y } _ { j } ) ) ^ { \theta } } , \quad \forall i \in [ m ] , \forall j \in [ n ] .\tag{93}
$$

Using (78), (93), $\kappa _ { \varepsilon }$ and $\mathbf { M } ^ { ( \theta ) }$ defined in (11) and (4), we have on $\varepsilon \colon$ for any $i \in [ m ]$

$$
\begin{array} { l } { \displaystyle \frac { 1 } { \tau _ { \mathrm { t } } } \sum _ { s ^ { \prime } = 1 } ^ { n } \widetilde { \alpha } _ { i } ^ { ( s ) } \lambda _ { i } ^ { ( s ) } \widetilde { \beta } _ { s ^ { \prime } } ^ { ( s ) } = \frac { \langle \hat { f } ( { \mathbf { x } } ) \rangle ^ { \beta } \sqrt { \zeta ( { \mathbf { \xi } } ) } ^ { \alpha ^ { \prime } } } { n } \frac { n ^ { \prime } } { \sqrt { ( \| \| \mu \| _ { L ^ { 1 } }  \| \| \boldsymbol { \Sigma } _ { s ^ { \prime } } ( { \mathbf { x } } _ { s } , { \mathbf { y } } ) \| ^ { \alpha _ { 0 } } } } \frac { K _ { \zeta } ( { \mathbf { x } } _ { s , \mathbf { y } } , { \mathbf { \xi } } _ { y } ) } { ( \|  { \Sigma } _ { s ^ { \prime } } \| ) \| \| \boldsymbol { \Sigma } _ { s ^ { \prime } } \| ^ { \alpha _ { 0 } }  } \frac { \nu ^ { \varepsilon ^ { \prime \prime \prime } } | | | \boldsymbol { \eta }  _ { s ^ { \prime } } } { ( \| \| \boldsymbol { \eta } \| _ { L ^ { 2 } }  \|  ^ { \beta } } } \\  \displaystyle = \frac { \sqrt { \zeta ( { \mathbf { \xi } } ) ^ { \beta } } } { ( \| \| \mu \| _ { L ^ { 2 } } \| \| \boldsymbol { \Sigma } _ { s ^ { \prime } } \| ^ { \beta }  \sqrt { \zeta ( { \mathbf { \xi } } ) ^ { \alpha ^ { \prime } } } } \frac { 1 } { n } \frac { \gamma ^ { \prime } } { \sqrt { \alpha } ^ { \beta } - 1 } a _ { s ^ { \prime } } ^ { ( s ) } ( \boldsymbol { \mathbb { \xi } } _ { s , \mathbf { \xi } } ( { \mathbf { x } } _ { s , \mathbf { y } } ) \nu _ { s ^ { \prime } } ^ { ( s ) }  \end{array}\tag{94}
$$

where the third line replaces $( \hat { g } ( \mathbf { y } _ { j } ) ) ^ { - \theta }$ using (88), together with the fact that $u _ { \varepsilon } ^ { ( \theta ) } , v _ { \varepsilon } ^ { ( \theta ) }$ and $\kappa _ { \varepsilon }$ are bounded on the compact manifolds, the fourth line applies Hoefding’s inequality and used (12), and the last line used (91). Under Assumption 2, a union bound over all $i \in [ m ]$ shows that (94) holds for all $i \in [ m ]$

Following an analogous derivation, for all $j \in [ n ]$ ，

$$
\frac { 1 } { c _ { j } } \sum _ { i = 1 } ^ { m } \hat { \alpha } _ { i } ^ { ( \theta ) } M _ { i j } ^ { ( \theta ) } \hat { \beta } _ { j } ^ { ( \theta ) } = 1 + \widetilde { \mathcal { O } } _ { m , n } \big ( \rho _ { 1 } + \rho _ { 2 } \big ) .\tag{95}
$$

Applying the matrix scaling stability lemma. We now apply Lemma B.2 to $\hat { \pmb { \alpha } } ^ { ( \theta ) }$ and $\hat { \boldsymbol { \beta } } ^ { ( \theta ) }$ , and first bound the quantities appearing there. By the definition of s and using (89), and (92),

$$
s = \| \mathbf { r } \| _ { 1 } = \frac { n m } { \sqrt { S ^ { ( \theta ) } } } \cdot \frac { 1 } { m } \sum _ { i = 1 } ^ { m } \frac { 1 } { \left( \hat { f } ( \mathbf { x } _ { i } ) \right) ^ { \theta } } = n m \left( ( \| \mu \| _ { L ^ { 2 } } \| \nu \| _ { L ^ { 2 } } ) ^ { \theta } \sqrt { Z _ { x } ^ { ( \theta ) } Z _ { y } ^ { ( \theta ) } } + \widetilde { \mathcal { O } } _ { m , n } ( \rho _ { 1 } + \rho _ { 2 } ) \right) .\tag{96}
$$

Since $u _ { \varepsilon } ^ { ( \theta ) }$ and $v _ { \varepsilon } ^ { ( \theta ) }$ are positive and continuous on the compact manifolds, and since $\hat { f }$ and $\hat { g }$ are bounded away from zero and infinity on $\mathcal { E } .$ , there exist positive constants $C _ { 1 } ^ { ' } ( \varepsilon )$ and $C _ { 2 } ^ { ' } ( \varepsilon )$ such that $C _ { 1 }$ and $C _ { 2 }$ defined in Lemma B.2 satisfy

$$
C _ { 1 } = \operatorname* { m i n } _ { i } \left\{ \frac { \hat { \alpha } _ { i } ^ { ( \theta ) } } { r _ { i } } \right\} \geq \frac { C _ { 1 } ^ { ' } ( \varepsilon ) } { n } , \quad C _ { 2 } = \operatorname* { m i n } _ { j } \left\{ \frac { \hat { \beta } _ { j } ^ { ( \theta ) } } { c _ { j } } \right\} \geq \frac { C _ { 2 } ^ { ' } ( \varepsilon ) } { m } .\tag{97}
$$

Similarly, there exist positive constants $C _ { 3 } ^ { ' } ( \varepsilon )$ and $C _ { 4 } ^ { ' } ( \varepsilon )$ such that

$$
C _ { 3 } ^ { ' } ( \varepsilon ) \leq M _ { i j } ^ { ( \theta ) } = \frac { \exp \left( - \| \mathbf { x } _ { i } - \mathbf { y } _ { j } \| _ { 2 } ^ { 2 } / \varepsilon \right) } { ( \hat { f } ( \mathbf { x } _ { i } ) \hat { g } ( \mathbf { y } _ { j } ) ) ^ { \theta } } \leq C _ { 4 } ^ { ' } ( \varepsilon ) ,\tag{98}
$$

so that a and b defined in Lemma B.2 satisfy $C _ { 3 } ^ { ' } ( \varepsilon ) \leq a \leq b \leq C _ { 4 } ^ { ' } ( \varepsilon )$ . Hence, for any $\delta \in ( 0 , 1 / 2 ]$ ， using (96), (97), and (98), the coeficients of (37) and (38) are bounded by positive constants $D _ { 1 } ( \varepsilon )$ and $D _ { 2 } ( \varepsilon )$ , exactly as in (56).

Applying Lemma B.2 with $\delta : = \tau C (  { \varepsilon } ) ( \rho _ { 1 } + \rho _ { 2 } )$ , where $C ( \varepsilon )$ is the constant associated with (94) and (95) in Definition B.3 and $\delta \leq 1 / 2$ upon choosing $n _ { 0 } ( \varepsilon , \tau ) , h _ { 1 } ^ { \prime }$ and $h _ { 2 } ^ { \prime }$ properly, there exists a pair of positive vectors $( \alpha ^ { ( \theta ) ^ { \prime } } , \beta ^ { ( \theta ) ^ { \prime } } )$ that scales $\mathbf { M } ^ { ( \theta ) }$ to row sums r and column sums $\mathbf { c } ,$ such that

$$
\frac { | \alpha _ { i } ^ { ( \theta ) ^ { \prime } } - \hat { \alpha } _ { i } ^ { ( \theta ) } | } { \hat { \alpha } _ { i } ^ { ( \theta ) } } = \widetilde { \mathcal { O } } _ { m , n } ^ { ( \varepsilon ) } ( \rho _ { 1 } + \rho _ { 2 } ) , \mathrm { a n d } \frac { | \beta _ { j } ^ { ( \theta ) ^ { \prime } } - \hat { \beta } _ { j } ^ { ( \theta ) } | } { \hat { \beta } _ { j } ^ { ( \theta ) } } = \widetilde { \mathcal { O } } _ { m , n } ^ { ( \varepsilon ) } ( \rho _ { 1 } + \rho _ { 2 } ) ,\tag{99}
$$

for any $i \in [ m ]$ and any $j \in [ n ]$

Conclusion. By Lemma B.1 and the definition of scaling factors, the two pairs $( \alpha ^ { ( \theta ) } , \beta ^ { ( \theta ) } )$ and $( \alpha ^ { ( \theta ) ^ { \prime } } , \beta ^ { ( \theta ) ^ { \prime } } )$ satisfy $\alpha _ { i } ^ { ( \theta ) } \beta _ { j } ^ { ( \theta ) } = { \alpha _ { i } ^ { ( \theta ) } } ^ { \prime } \beta _ { j } ^ { ( \theta ) ^ { \prime } }$ for any $i \in [ m ]$ and $j \in [ n ]$ . Hence, for any $i \in [ m ]$ and $j \in [ n ]$

$$
\begin{array} { l } { { W _ { i j } ^ { ( \theta ) } = \alpha _ { i } ^ { ( \theta ) } e ^ { - \left\| { \bf x } _ { i } - { \bf y } _ { j } \right\| _ { 2 } ^ { 2 } / \varepsilon } \beta _ { j } ^ { ( \theta ) } = \alpha _ { i } ^ { ( \theta ) ^ { \prime } } e ^ { - \left\| { \bf x } _ { i } - { \bf y } _ { j } \right\| _ { 2 } ^ { 2 } / \varepsilon } \beta _ { j } ^ { ( \theta ) ^ { \prime } } = \hat { \alpha } _ { i } ^ { ( \theta ) } e ^ { - \left\| { \bf x } _ { i } - { \bf y } _ { j } \right\| _ { 2 } ^ { 2 } / \varepsilon } \hat { \beta } _ { j } ^ { ( \theta ) } \left( 1 + \widetilde { \mathcal { O } } _ { m , n } ^ { ( \varepsilon ) } ( \rho _ { 1 } + \rho _ { 2 } ) \right) } } \\ { { \ = \frac { \mathcal { W } _ { \varepsilon } ^ { ( \theta ) } ( { \bf x } _ { i } , { \bf y } _ { j } ) } { ( \left\| \mu \right\| _ { L ^ { 2 } } \left\| \nu \right\| _ { L ^ { 2 } } ) ^ { \theta } \sqrt { \mathcal { L } _ { x } ^ { ( \theta ) } \mathcal { L } _ { y } ^ { ( \theta ) } } } \left( 1 + \widetilde { \mathcal { O } } _ { m , n } ^ { ( \varepsilon ) } ( \rho _ { 1 } + \rho _ { 2 } ) \right) . } } \end{array}\tag{100}
$$

Since ${ \mathscr W } _ { \varepsilon } ^ { ( \theta ) }$ is continuous and bounded above on the compact product manifold $\mathcal { M } _ { x } \times \mathcal { M } _ { y }$ , the multiplicative estimate in (100) implies the additive bound in (17). Finally, this estimate holds on the intersection of E with the events underlying (89)–(99); by (81) and the union bound, this intersection has probability at least $1 - n ^ { - \tau } - 4 n ^ { - 2 \gamma }$ , completing the proof. □

## References

[1] Tim Stuart and Rahul Satija. Integrative single-cell analysis. Nature reviews genetics, 20(5):257–272, 2019.

[2] Geofrey Schiebinger, Jian Shu, Marcin Tabaka, Brian Cleary, Vidya Subramanian, Aryeh Solomon, Joshua Gould, Siyan Liu, Stacie Lin, Peter Berube, et al. Optimal-transport analysis of single-cell gene expression identifies developmental trajectories in reprogramming. Cell, 176(4):928–943, 2019.

[3] Ludovic M´etivier, Romain Brossier, Quentin M´erigot, Edouard Oudet, and Jean Virieux. Measuring the misfit between seismograms using an optimal transport distance: Application to full waveform inversion. Geophysical Supplements to the Monthly Notices of the Royal Astronomical Society, 205(1):345–377, 2016.

[4] Ricard Argelaguet, Anna SE Cuomo, Oliver Stegle, and John C Marioni. Computational principles and challenges in single-cell data integration. Nature biotechnology, 39(10):1202–1215, 2021.

[5] Malte D Luecken, Maren B¨uttner, Kridsadakorn Chaichoompu, Anna Danese, Marta Interlandi, Michaela F M¨uller, Daniel C Strobl, Luke Zappia, Martin Dugas, Maria Colom´e-Tatch´e, et al. Benchmarking atlas-level data integration in single-cell genomics. Nature methods, 19(1):41–50, 2022.

[6] Hoa Thi Nhu Tran, Kok Siong Ang, Marion Chevrier, Xiaomeng Zhang, Nicole Yee Shin Lee, Michelle Goh, and Jinmiao Chen. A benchmark of batch-efect correction methods for single-cell rna sequencing data. Genome biology, 21(1):12, 2020.

[7] C´edric Villani. Topics in optimal transportation, volume 58. American Mathematical Soc., 2021.

[8] Gabriel Peyr´e and Marco Cuturi. Computational optimal transport: With applications to data science. Now Foundations and Trends, 2019.

[9] Marco Cuturi. Sinkhorn distances: Lightspeed computation of optimal transport. Advances in neural information processing systems, 26, 2013.

[10] Richard Sinkhorn and Paul Knopp. Concerning nonnegative matrices and doubly stochastic matrices. Pacific Journal of Mathematics, 21(2):343–348, 1967.

[11] Jason Altschuler, Jonathan Niles-Weed, and Philippe Rigollet. Near-linear time approximation algorithms for optimal transport via sinkhorn iteration. Advances in neural information processing systems, 30, 2017.

[12] Tianyi Lin, Nhat Ho, and Michael I Jordan. On the eficiency of entropic regularized algorithms for optimal transport. Journal of Machine Learning Research, 23(137):1–42, 2022.

[13] Gonzalo Mena and Jonathan Niles-Weed. Statistical bounds for entropic optimal transport: sample complexity and the central limit theorem. Advances in neural information processing systems, 32, 2019.

[14] Nicolas Courty, R´emi Flamary, Devis Tuia, and Alain Rakotomamonjy. Optimal transport for domain adaptation. IEEE transactions on pattern analysis and machine intelligence, 39(9):1853–1865, 2016.

[15] Nicolas Courty, R´emi Flamary, Amaury Habrard, and Alain Rakotomamonjy. Joint distribution optimal transportation for domain adaptation. Advances in neural information processing systems, 30, 2017.

[16] Aude Genevay, Gabriel Peyr´e, and Marco Cuturi. Learning generative models with sinkhorn divergences. In International Conference on Artificial Intelligence and Statistics, pages 1608–1617. PMLR, 2018.

[17] Jean Feydy, Thibault S´ejourn´e, Fran¸cois-Xavier Vialard, Shun-ichi Amari, Alain Trouv´e, and Gabriel Peyr´e. Interpolating between optimal transport and mmd using sinkhorn divergences. In The 22nd international conference on artificial intelligence and statistics, pages 2681–2690. PMLR, 2019.

[18] Zhengyang Shen, Jean Feydy, Peirong Liu, Ariel H Curiale, Ruben San Jose Estepar, Raul San Jose Estepar, and Marc Niethammer. Accurate point cloud registration with robust optimal transport. Advances in Neural Information Processing Systems, 34:5373–5389, 2021.

[19] Justin Solomon, Fernando De Goes, Gabriel Peyr´e, Marco Cuturi, Adrian Butscher, Andy Nguyen, Tao Du, and Leonidas Guibas. Convolutional wasserstein distances: Eficient optimal transportation on geometric domains. ACM Transactions on Graphics (ToG), 34(4):1–11, 2015.

[20] Geofrey Schiebinger, Jian Shu, Marcin Tabaka, Brian Cleary, Vidya Subramanian, Aryeh Solomon, Joshua Gould, Siyan Liu, Stacie Lin, Peter Berube, et al. Optimal-transport analysis of single-cell gene expression identifies developmental trajectories in reprogramming. Cell, 176(4):928–943, 2019.

[21] Yogesh Balaji, Rama Chellappa, and Soheil Feizi. Robust optimal transport with applications in generative modeling and domain adaptation. Advances in Neural Information Processing Systems, 33:12934– 12944, 2020.

[22] Milena Gazdieva, Arip Asadulaev, Evgeny Burnaev, and Alexander Korotin. Light unbalanced optimal transport. Advances in Neural Information Processing Systems, 37:93907–93938, 2024.

[23] Thibault S´ejourn´e, Gabriel Peyr´e, and Fran¸cois-Xavier Vialard. Unbalanced optimal transport, from theory to numerics. Handbook of Numerical Analysis, 24:407–471, 2023.

[24] Lenaic Chizat, Gabriel Peyr´e, Bernhard Schmitzer, and Fran¸cois-Xavier Vialard. Scaling algorithms for unbalanced optimal transport problems. Mathematics of computation, 87(314):2563–2609, 2018.

[25] Thibault S´ejourn´e, Jean Feydy, Fran¸cois-Xavier Vialard, Alain Trouv´e, and Gabriel Peyr´e. Sinkhorn divergences for unbalanced optimal transport. arXiv preprint arXiv:1910.12958, 2019.

[26] Thibault S´ejourn´e, Fran¸cois-Xavier Vialard, and Gabriel Peyr´e. Faster unbalanced optimal transport: Translation invariant sinkhorn and 1-d frank-wolfe. In International Conference on Artificial Intelligence and Statistics, pages 4995–5021. PMLR, 2022.

[27] Eduardo Fernandes Montesuma, Fred Maurice Ngole Mboula, and Antoine Souloumiac. Recent advances in optimal transport for machine learning. IEEE Transactions on Pattern Analysis and Machine Intelligence, 47(2):1161–1180, 2024.

[28] Ronald R Coifman and St´ephane Lafon. Difusion maps. Applied and computational harmonic analysis, 21(1):5–30, 2006.

[29] Boris Landa, Yuval Kluger, and Rong Ma. Entropic optimal transport eigenmaps for nonlinear alignment and joint embedding of high-dimensional datasets. arXiv preprint arXiv:2407.01718, 2024.

[30] Paul Knopp and Richard Sinkhorn. A note concerning simultaneous integral equations. Canadian Journal of Mathematics, 20:855–861, 1968.

[31] Bernard W Silverman. Density estimation for statistics and data analysis. Routledge, 2018.

[32] Bradley Efron and Robert J Tibshirani. An introduction to the bootstrap. Chapman and Hall/CRC, 1994.

[33] Oleg V Lepski, Enno Mammen, and Vladimir G Spokoiny. Optimal spatial adaptation to inhomogeneous smoothness: an approach based on kernel estimates with variable bandwidth selectors. The Annals of Statistics, pages 929–947, 1997.

[34] Richard Sinkhorn. Diagonal equivalence to matrices with prescribed row and column sums. The American Mathematical Monthly, 74(4):402–405, 1967.

[35] David B Dunson, Hau-Tieng Wu, and Nan Wu. Spectral convergence of graph laplacian and heat kernel reconstruction in l∞ from random samples. Applied and Computational Harmonic Analysis, 55:282–336, 2021.