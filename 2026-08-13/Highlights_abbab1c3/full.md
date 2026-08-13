## Highlights

## A Local Sinkhorn Framework for Conditional Distribution Reconstruction of Multidimensional Random Fields

Mingtao Xia, Qijing Shen

• We propose a scalable local entropically regularized optimal transport framework to train stochastic neural networks for learning random fields.

• We derive generalization error bounds that reveal how our proposed framework could eficiently learn multidimensional random fields.

• Our proposed approach accurately reconstructs stochastic random fields and dynamical systems and outperforms several machine-learning-based benchmark approaches in uncertainty quantification.

# A Local Sinkhorn Framework for Conditional Distribution Reconstruction of Multidimensional Random Fields

Mingtao Xia<sup>a,b,∗</sup>, Qijing Shen<sup>c</sup>

<sup>a</sup>School ofMathematics, University ofBirmingham, Watson Building, Edgbaston, Birmingham, B15 2TT, United Kingdom

<sup>b</sup>Department ofMathematics, University ofHouston, 3551 Cullen Blvd, Houston, 77204, Texas, United States

<sup>c</sup>Nufield Department ofMedicine, University ofOxford, United Kingdom, Old Road Campus, Oxford, OX3 7BN, United Kingdom

## A R T I C L E I N F O

Keywords: Random Field Reconstruction Sinkhorn Divergence Uncertainty Quantification Stochastic Neural Network Scientific Machine Learning

## A BS T R AC T

In this paper, we propose a local Sinkhorn divergence framework for conditional distribution reconstruction of multidimensional random fields. By utilizing the debiased Sinkhorn divergence, our proposed approach develops a diferentiable and computationally eficient local distribution matching objective to train stochastic neural networks (SNNs). Furthermore, we establish theoretical generalization error estimates for our local Sinkhorn divergence framework, which explicitly characterizes the trade-of between approximation bias and statistical eficiency controlled by the regularization parameter and reveals how our proposed local Sinkhorn divergence loss function can be eficiently applied to learning multidimensional random field models. The proposed framework provides a scalable alternative to exact local optimal transport for conditional distribution reconstruction, ofering a practical compromise between geometric fidelity, statistical eficiency, and computational scalability for uncertainty quantification and probabilistic scientific machine learning. Through various numerical examples, we compare our proposed local Sinkhorn divergence framework with other loss functions to train SNNs and with other machine-learning-based uncertainty quantification frameworks, demonstrating that the proposed local Sinkhorn divergence framework achieves an efective balance between reconstruction accuracy and computational eficiency while maintaining good scalability for multidimensional stochastic systems.

## 1. Introduction

The identification of random fields from observational data has become an increasingly important topic in uncertainty quantification, stochastic modeling, and scientific machine learning. Many physical and engineering systems, including porous media flow, turbulent transport, groundwater dynamics, and biological processes, are inherently afected by uncertain parameters or random forcing, leading to stochastic responses that cannot be adequately characterized by deterministic models alone [13, 33, 25]. Consequently, instead of predicting only the conditional expectation, it is often desirable to reconstruct the entire conditional probability distribution of the stochastic solution, thereby providing a more comprehensive description of uncertainty propagation. However, the primary challenge is usually not learning the conditional mean, but accurately reconstructing the multidimensional conditional distributions in a computationally scalable manner.

Recent advances in machine learning have enabled data-driven approximations of multidimensional stochastic systems without requiring explicit knowledge of the underlying governing equations. Various deep generative models, including conditional variational autoencoders (CVAEs), generative adversarial networks (GANs), conditional normalizing flows (CNFs), and difusion models, have demonstrated remarkable capability in learning complex probability distributions from data [17, 14, 22, 16]. Nevertheless, for stochastic random field reconstruction, the learned distributions should preserve the geometric structure of probability measures associated with nearby physical states. This requirement naturally motivates the use of optimal transport (OT), whose Wasserstein metric provides a physically meaningful distance between probability distributions and has recently attracted increasing attention in scientific machine learning [29, 20]. Compared with divergence-based discrepancies, such as the Kullback–Leibler divergence or maximum mean discrepancy [15], Wasserstein distances remain informative even when two probability measures have disjoint supports and provide meaningful gradients during optimization [2, 20]. These advantages have motivated numerous OT-based machine learning methods, including Wasserstein GANs, gradient-flow formulations, and OT regularization for scientific computing [10, 2]. However, the exact computation of Wasserstein distances requires solving a large-scale linear programming problem whose complexity grows rapidly with sample size, making repeated evaluations during neural network training prohibitively expensive. Our recent work introduced a local squared Wasserstein-2 $( W _ { 2 } )$ framework for random field reconstruction, in which neighboring samples associated with nearby input states were compared through local optimal transport distances. By exploiting the locality of conditional distributions, the method was shown to accurately reconstruct stochastic fields from sparse observations while preserving important geometric structures of the underlying probability measures [31]. However, the local squared $W _ { 2 }$ distance relies on repeatedly solving exact optimal transport problems during neural network training. As the number of neighborhood evaluations and samples increases, the computational cost becomes the dominant bottleneck, significantly limiting scalability for multidimensional stochastic systems and large datasets. Additionally, the generalization error bound may deteriorate when the squared $W _ { 2 }$ distance between two probability measures is estimated using empirical probability measures, particularly in high-dimensional settings.

Recently, Cuturi introduced entropic regularization of optimal transport, leading to the Sinkhorn algorithm that dramatically accelerates OT computations while remaining fully diferentiable [6]. Subsequently, Feydy et al. proposed the Sinkhorn divergence, which removes the entropic bias and interpolates smoothly between the exact Wasserstein distance and kernel-based maximum mean discrepancy [8]. The Sinkhorn divergence has since been successfully applied to generative modeling, Bayesian inference, and scientific machine learning because of its computational eficiency and stable gradient properties [12, 8]. Furthermore, through introducing an extra entropic regularization term, the generalization error when estimating the Sinkhorn divergence using the empirical measures may not deteriorate with the dimensionality of the associated random variables [12].

Motivated by these developments, this work develops a local Sinkhorn-divergence framework for conditional distribution reconstruction of stochastic fields. Building upon our previous local $W _ { 2 }$ formulation, the proposed approach replaces the expensive exact OT computation with Sinkhorn divergence, resulting in substantially improved computational eficiency while preserving the local geometric information of conditional probability distributions. The proposed local Sinkhorn divergence bridges exact optimal transport and kernel-based distribution matching by introducing an entropic regularization that provides a controllable bias while significantly improving computational eficiency and statistical scalability. Owing to the diferentiability and favorable numerical stability of Sinkhorn divergence, the resulting learning framework is considerably more eficient in reconstructing conditional distributions of the to-be-learned random fields. Furthermore, we shall show from both theoretical and empirical aspects that incorporating the regularization term could lead to improved accuracy when reconstructing multidimensional random field models using empirical distributions. Numerical experiments on stochastic Darcy flow and multidimensional stochastic FitzHugh–Nagumo (FHN) systems demonstrate that the proposed approach achieves reconstruction accuracy comparable to that of the exact Wasserstein formulation while significantly reducing the overall training cost.

The main contributions of this work are summarized as follows.

• We extend our recent local optimal transport framework for random field reconstruction from the exact Wasserstein distance to the debiased Sinkhorn divergence, leading to a computationally eficient, scalable, and fully diferentiable local distribution matching method for training SNNs to reconstruct random fields.

• We provide a theoretical analysis of the proposed local Sinkhorn formulation by combining the neighborhood approximation error with the regularization and statistical convergence properties of empirical Sinkhorn divergence. The resulting generalization bounds reveal how the local Sinkhorn divergence may partially alleviate the curse of dimensionality through introducing an entropic regularization term.

• Through various numerical examples, we demonstrate that the proposed local Sinkhorn divergence provides an efective compromise between approximation accuracy and computational eficiency and outperforms several machine-learning-based uncertainty quantification benchmarks, making it particularly suitable for multidimensional uncertainty quantification problems.

The remainder of this paper is organized as follows. Section 2 introduces the proposed local Sinkhorn divergence framework together with its theoretical analysis and generalization error bounds. Section 3 presents three numerical examples, including one-dimensional conditional distribution reconstruction, stochastic Darcy flow, and stochastic FHN systems, where the proposed method is compared with several local distribution matching losses and existing machine-learning benchmark approaches. Finally, Section 4 concludes the paper with a summary of the main findings and discusses several directions for future research. Frequently used notations throughout this paper are summarized in Table 1.

Frequently used notations throughout this paper.
<table><tr><td>Notation</td><td>Description</td></tr><tr><td> $\boldsymbol { x } \in \boldsymbol { D }$ </td><td>Input variable.</td></tr><tr><td> $D \subseteq \mathbb { R } ^ { n }$ </td><td>Input domain.</td></tr><tr><td> $N$ </td><td>Total number of training samples.</td></tr><tr><td> $\omega \left( { \hat { \omega } } \right)$ </td><td>Random variable representing uncertainty.</td></tr><tr><td> $\pmb { y } ( \pmb { x } , \omega ) \in \mathbb { R } ^ { d }$ </td><td>Target random field.</td></tr><tr><td> $\hat { \pmb { y } } ( \pmb { x } , \hat { \omega } ) \in \mathbb { R } ^ { d }$ </td><td>Reconstructed random field.</td></tr><tr><td> $\mu _ { x }$ </td><td>Conditional probability distribution of  ${ \pmb y } ( { \pmb x } , \omega )$ </td></tr><tr><td> $\hat { \mu } _ { x }$ </td><td>Conditional probability distribution of  $\hat { \pmb { y } } ( \pmb { x } , \hat { \omega } ) .$ </td></tr><tr><td> $\mu _ { x , \delta } ^ { \mathrm { e } }$ </td><td>Empirical δ-neighborhood measure associated with  ${ \mathbf y } ( { \mathbf x } , \omega )$  (details given in Section 2).</td></tr><tr><td> $\hat { \mu } _ { x , \delta } ^ { \mathrm { e } }$ </td><td>Empirical δ-neighborhood measure associated with  $\hat { \mathbf { y } } ( \mathbf { x } , \hat { \omega } )$  (details given in Section 2).</td></tr><tr><td> $\delta$ </td><td>Neighborhood radius.</td></tr><tr><td> $\nu$ </td><td>Probability measure on the input domain.</td></tr><tr><td> $\nu ^ { \mathsf { e } }$ </td><td>Empirical measure induced by the observed inputs.</td></tr><tr><td> $W _ { 2 } ^ { 2 } ( \mu , \hat { \mu } )$ </td><td>Squared  $W _ { 2 }$  一 distance between the two probability measures  $\mu , { \hat { \mu } } .$ </td></tr><tr><td> $S _ { \varepsilon } ( \mu , \hat { \mu } )$ </td><td>Debiased Sinkhorn divergence between the two probability measures  $\mu , { \hat { \mu } } .$ </td></tr><tr><td> $\varepsilon$ </td><td>Sinkhorn regularization parameter.</td></tr><tr><td> $\overline { { \mathcal { W } } } _ { 2 }$ </td><td>Averaged squared Wasserstein discrepancy.</td></tr><tr><td> $\overline { { S } } _ { \varepsilon , \delta } ^ { \mathsf { e } }$ </td><td>Averaged empirical local Sinkhorn divergence.</td></tr></table>

## 2. A local Sinkhorn divergence method for conditional distribution reconstruction of random fields

In this section, we formally introduce the random field model we reconstruct and the local Sinkhorn divergence framework we propose. Consider the reconstruction of conditional probability distributions from scattered observations. Let

$$
\begin{array} { r } { \pmb { y } _ { \pmb { x } } = f ( \pmb { x } , \omega ) , \qquad \pmb { x } \in \mathcal { D } \subset \mathbb { R } ^ { n } , } \end{array}\tag{2.1}
$$

be the unknown random field to be reconstructed, where $\omega ~ \in ~ \Omega$ denotes the underlying random variable and $f : \mathcal { D } \times \Omega \to \mathbb { R } ^ { d }$ is an unknown stochastic mapping. For each input location �, only realizations of the random response $y ( \pmb { x } , \omega )$ are available, while the conditional probability measure of ${ \pmb y } ( { \pmb x } ; \omega )$ , denoted as $\mu _ { x } .$ , is unknown.

Our objective is to construct an approximation

$$
\hat { \mathbf { y } } _ { x } = \hat { f } ( \mathbf { x } , \hat { \omega } ) , \qquad \mathbf { x } \in \mathcal { D } ,\tag{2.2}
$$

where $\hat { f }$ is another stochastic mapping, and $\hat { \omega } \in \hat { \Omega }$ is another random variable. The goal is to match the distribution of �(�; �) using the distribution of $\hat { \mathbf { y } } ( \mathbf { x } ; \hat { \omega } )$ , denoted by $\hat { \mu } _ { x }$ for every $\boldsymbol { x } \in \boldsymbol { D }$

We employ a stochastic neural network (SNN) similar to our previous work [31] as the approximate random field model $\hat { f }$ in Eq. (2.2). The structure of the SNN is described in Fig. 1. Unlike deterministic neural networks, the SNN contains random parameters (as the random variable ̂� in Eq. (2.2)) sampled during forward propagation, enabling the model to generate multiple realizations for the same input location. Furthermore, it is proved in [32, Appendix H] that this SNN can approximate any random field model under an averaged squared $W _ { 2 }$ metric, subject to nonrestrictive assumptions.

First, we briefly review the Sinkhorn divergence between two probability measures. Let $\ b { y } , \hat { \ b { y } } \in \mathbb { R } ^ { n }$ be two random variables satisfying:

$$
\begin{array} { r } { \mathbb { E } \big [ \| \pmb { y } \| ^ { 2 } \big ] < \infty , \qquad \mathbb { E } \big [ \| \hat { \pmb { y } } \| ^ { 2 } \big ] < \infty . } \end{array}\tag{2.3}
$$

Throughout this paper, $\| \cdot \|$ denotes the $\ell ^ { 2 }$ norm of a vector. The probability distributions of � and $\hat { \pmb { y } }$ are denoted by $\mu$ and $\hat { \mu } ,$ respectively. Let

$$
c ( y , \hat { y } ) = \| y - \hat { y } \| ^ { 2 }\tag{2.4}
$$

be the quadratic transportation cost. The entropically regularized optimal transport cost is defined as:

$$
W _ { \varepsilon } ^ { 2 } ( \mu , \hat { \mu } ) = \operatorname* { m i n } _ { \pi \in \Pi ( \mu , \hat { \mu } ) } \left\{ \int _ { \mathbb { R } ^ { n } \times \mathbb { R } ^ { n } } c ( { \pmb y } , \hat { \pmb y } ) \mathrm { d } \pi ( { \pmb y } , \hat { \pmb y } ) + \varepsilon \mathrm { K L } ( \pi \parallel \mu \otimes \hat { \mu } ) \right\} ,\tag{2.5}
$$

where $\Pi ( \mu , \hat { \mu } )$ denotes the set of probability measures on $\mathbb { R } ^ { n } \times \mathbb { R } ^ { n }$ whose marginals are $\mu$ and $\hat { \mu } ,$ respectively. In Eq. (2.5),

$$
\operatorname { K L } ( \pi \parallel \mu \otimes { \hat { \mu } } ) = \int _ { \mathbb { R } ^ { n } \times \mathbb { R } ^ { n } } \log \left( { \frac { \mathrm { d } \pi } { \mathrm { d } ( \mu \otimes { \hat { \mu } } ) } } \right) \mathrm { d } \pi
$$

denotes the Kullback–Leibler divergence of � with respect to the product measure $\mu \otimes { \hat { \mu } } .$ , provided that $\pi \ll \mu \otimes { \hat { \mu } } ;$ otherwise, $\operatorname { K L } ( \pi \parallel \mu \otimes { \hat { \mu } } ) = + \infty$ , and $\varepsilon > 0$ is the entropic regularization parameter.

Unlike the classical Wasserstein distance, the regularized transport cost does not vanish when $\mu = \hat { \mu }$ . To remove this entropic bias, the Sinkhorn divergence is defined as

$$
S _ { \varepsilon } ( \mu , \hat { \mu } ) = W _ { \varepsilon } ^ { 2 } ( \mu , \hat { \mu } ) - \frac { 1 } { 2 } W _ { \varepsilon } ^ { 2 } ( \mu , \mu ) - \frac { 1 } { 2 } W _ { \varepsilon } ^ { 2 } ( \hat { \mu } , \hat { \mu } ) .\tag{2.6}
$$

The Sinkhorn divergence in $\operatorname { E q . } \left( 2 . 6 \right)$ is symmetric, non-negative, diferentiable with respect to the empirical samples, and satisfies $S _ { \varepsilon } ( \mu , \mu ) = 0 ;$ ; furthermore, the Sinkhorn divergence converges to the Wasserstein distance as $\varepsilon \ \to \ 0$ while remaining diferentiable and computationally tractable, whereas choosing � → ∞ recovers the maximum mean discrepancy [8]. Yet, the Sinkhorn divergence is more computationally eficient through introducing an additional regularizing term in Eq. (2.5) while preserving the advantage of the squared $W _ { 2 }$ distance for comparing two probability measures.

To simplify the subsequent analysis, we introduce the following assumptions.

Assumption 2.1. Assume that the uncertainty model ${ \pmb y } _ { \pmb x }$ and the approximation model $\hat { y } _ { x }$ in Eqs. (2.1) and (2.2) satisfy the following conditions.

1. Both random fields are uniformly bounded, i.e., there exists $M _ { 0 } > 0$ such that

$$
\| \ b { y } ( \ b { x } , \omega ) \| \le M _ { 0 } , \qquad \| \ b { \hat { y } } ( \ b { x } , \hat { \omega } ) \| \le M _ { 0 } , \qquad \forall \ b { x } \in \ b { D } .\tag{2.7}
$$

2. The mappings � and $\hat { f }$ are uniformly Lipschitz continuous with respect to the input variable. That is, there exists a constant $L > 0$ such that

$$
\| f ( \pmb { x } , \omega ) - f ( \pmb { x } ^ { \prime } , \omega ) \| \leq L \| \pmb { x } - \pmb { x } ^ { \prime } \| , \quad \| \hat { f } ( \pmb { x } , \hat { \omega } ) - \hat { f } ( \pmb { x } ^ { \prime } , \hat { \omega } ) \| \leq L \| \pmb { x } - \pmb { x } ^ { \prime } \| , \ \forall \pmb { x } , \pmb { x } ^ { \prime } \in D .\tag{2.8}
$$

3. The random variables � and ̂� are independent of the input variable �.

Following our previous work on local optimal transport [31], we define the weighted average squared $W _ { 2 }$ metric between the ground-truth and approximate random fields ${ \pmb y } _ { \pmb x }$ and $\hat { y } _ { x }$ in Eqs. (2.1) and (2.2):

$$
\overline { { \mathscr { W } } } _ { 2 } ( y _ { x } , \hat { y } _ { x } ) : = \int _ { D } W _ { 2 } ^ { 2 } ( \mu _ { x } , \hat { \mu } _ { x } ) \nu ( \mathrm { d } x ) .\tag{2.9}
$$

$\overline { { \mathscr { W } } } _ { 2 } ( { \bf y } _ { x } , \hat { \bf y } _ { x } )$ defined in Eq. (2.9) is nonnegative, and if the weighting measure of the input variable � is strictly positive over $D _ { \mathrm { { i } } }$ , then $\overline { { \mathscr { W } } } _ { 2 } ( y _ { x } , \hat { y } _ { x } ) = 0$ holds if and only if

$$
W _ { 2 } ^ { 2 } ( \mu _ { x } , \hat { \mu } _ { x } ) = 0 , \qquad \mathrm { f o r ~ a l m o s t ~ e v e r y ~ } x \in \mathcal { D } ,
$$

which implies that the probability measure of ${ \pmb y } _ { \pmb x }$ can be matched by the probability measure of $\hat { y } _ { x }$ for almost every $\boldsymbol { x } \in \boldsymbol { D }$ . Consequently, minimizing (2.9) is equivalent to matching the conditional distributions $\mu _ { x }$ using $\hat { \mu } _ { x }$ throughout the computational domain. However, in practical applications only a finite set of scattered observations $\{ ( \pmb { x } _ { i } , \pmb { y } _ { \pmb { x } _ { i } } ) \} _ { i = 1 } ^ { N }$ is available. Since the conditional probability measures $\mu _ { x }$ and $\hat { \mu } _ { x }$ cannot be evaluated directly from finite observations, we approximate them by local empirical measures constructed from neighboring samples.

Given a training sample $\pmb { x } _ { i } \in \{ \pmb { x } _ { 1 } , . . . \pmb { x } _ { N } \}$ , we define its neighborhood

$$
B ( \pmb { x } _ { i } , \delta ) = \{ \pmb { x } _ { j } : \lVert \pmb { x } _ { j } - \pmb { x } _ { i } \rVert < \delta , \ j = 1 , . . . , N \} ,\tag{2.10}
$$

where $\delta > 0$ is the neighborhood radius. For implementing the neighborhood technique, we randomly choose a subset

$$
B ( \pmb { x } _ { i } , \delta ) \subseteq B ( \pmb { x } _ { i } , \delta ) .\tag{2.11}
$$

Let $\mu _ { x _ { i } , \delta } ^ { \mathrm { e } }$ and $\hat { \mu } _ { x _ { i } , \delta } ^ { \mathrm { e } }$ denote the corresponding empirical conditional probability measures constructed from the neighboring observations, i.e., $\mu _ { x _ { i } , \delta } ^ { \mathrm { e } }$ is the empirical distribution of $\pmb { y } ( \pmb { x } ; \omega ) , \pmb { x } \in B ( \pmb { x } _ { i } , \delta )$ and $\hat { \mu } _ { x _ { i } , \delta } ^ { \mathrm { e } }$ is the empirical distribution of �̂(�; �), $\pmb { x } \in B ( \pmb { x } _ { i } , \dot { \delta } ) . \mu _ { \pmb { x } _ { i } , \delta } ^ { \mathrm { e } }$ and $\hat { \mu } _ { x _ { i } , \delta } ^ { \mathrm { e } }$ are used to approximate $\mu _ { x }$ and $\hat { \mu } _ { x }$ , respectively. We then introduce the proposed local Sinkhorn divergence loss:

$$
S _ { \varepsilon , \delta } ^ { \mathrm { e } } ( y _ { x } , \hat { y } _ { x } ) : = \int _ { \cal D } S _ { \varepsilon } ( \mu _ { x , \delta } ^ { \mathrm { e } } , \hat { \mu } _ { x , \delta } ^ { \mathrm { e } } ) \nu ^ { \mathrm { e } } ( \mathrm { d } x ) ,\tag{2.12}
$$

where $\nu ^ { \mathrm { e } }$ denotes the empirical measure associated with the observed input variable �. Eq. (2.12) is not a single Sinkhorn divergence between two probability measures. Instead, it computes the average Sinkhorn divergence between the local empirical conditional distributions over the entire input domain. The superscript e indicates empirical probability measures estimated from neighboring observations, while the subscript � denotes the neighborhood radius.

Compared with the $W _ { 2 }$ metrics, the Sinkhorn divergence introduces an entropic regularization, allowing the optimal transport problem to be solved eficiently by iterative matrix scaling rather than network simplex optimization [6, 8]. Consequently, the resulting loss is fully diferentiable and naturally amenable to GPU-parallel computation, while recent theoretical results also demonstrate improved statistical convergence properties for empirical Sinkhorn divergences compared with empirical Wasserstein distances [11]. Specifically, we establish two diferent forms of the generalization error bounds on how the local Sinkhorn divergence Eq. (2.12) approximates the averaged squared Wasserstein discrepancy in Eq. (2.9).

Theorem 2.1 (Generalization bound based on the generalization error of the local squared $W _ { 2 }$ loss). Under Assumptions 2.1, let $\mu _ { x }$ and $\hat { \mu } _ { x }$ denote the true conditional probability measures associated with ${ \pmb y } _ { \pmb x }$ and $\hat { y } _ { x }$ at location $\boldsymbol { x } \in \mathcal { D }$ , respectively. Let $\mu _ { x , \delta } ^ { \mathrm { e } }$ and $\hat { \mu } _ { x , \delta } ^ { \mathrm { e } }$ be the corresponding empirical neighborhood measures built from the sample set $\{ ( \pmb { x } _ { i } , \pmb { y } _ { \pmb { x } _ { i } } ) \} _ { i = 1 } ^ { N }$ with neighborhood radius $\delta > 0$ . We assume $\varepsilon < \frac { 2 M _ { 0 } ^ { 2 } } { \sqrt { d } }$ . For $\overline { { S } } _ { \varepsilon , \delta } ^ { \mathrm { e } } ( \boldsymbol { y } _ { x } , \hat { \boldsymbol { y } } _ { x } )$ and $\overline { { \mathscr { W } } } _ { 2 } ( { \bf y } _ { x } , \hat { \bf y } _ { x } )$ defined in Eqs. (2.9) and (2.12), there exists a constant $C > 0$ such that:

$$
\mathbb { E } \bigg [ \bigg \lvert \overline { { S } } _ { \varepsilon , \delta } ^ { \textnormal { e } } ( y _ { x } , \hat { y } _ { x } ) - \overline { { \mathcal { W } } } _ { 2 } ( y _ { x } , \hat { y } _ { x } ) \bigg \rvert \bigg ] \leq 2 \varepsilon \log \Bigg ( \frac { 2 e ^ { 2 } M _ { 0 } ^ { 2 } } { \sqrt { d } \varepsilon } \Bigg ) + \frac { 4 M _ { 0 } } { \sqrt { N } } + 8 C M _ { 0 } \mathbb { E } \big [ h ( N ( x , \delta ) , d ) \big ] + 1 6 M _ { 0 } ^ { 2 } \delta ,\tag{2.13}
$$

where � is the number of training samples, and

$$
N ( \pmb { x } , \delta ) : = | B ( \pmb { x } , \delta ) |\tag{2.14}
$$

denotes the efective number of samples chosen from the neighborhood $B ( { \pmb x } , \delta )$ centered at � defined in Eq. (2.10), and

$$
\begin{array} { r } { h ( N , d ) : = \left\{ \begin{array} { l l } { 2 N ^ { - 1 / 4 } \log ( 1 + N ) ^ { 1 / 2 } , } & { d \leq 4 , } \\ { 2 N ^ { - \frac { 1 } { 4 } } + \big ( \prod _ { i = 1 } ^ { d } ( \frac { \sigma _ { i , x } } { \sigma _ { 1 , x } } ) N \big ) ^ { - \frac { 1 } { d } } + \big ( \prod _ { i = 1 } ^ { d } ( \frac { \hat { \sigma } _ { i , x } } { \hat { \sigma } _ { 1 , x } } ) N \big ) ^ { - \frac { 1 } { d } } , } & { d > 4 . } \end{array} \right. } \end{array}\tag{2.15}
$$

In Eq. (2.15),

$$
\sigma _ { i , x } : = \big ( \int _ { \mathbb { R } ^ { d } } y _ { i } \mu _ { x } ( \mathrm { d } y ) \big ) ^ { \frac { 1 } { 6 } } , \ \hat { \sigma } _ { i , x } : = \big ( \int _ { \mathbb { R } ^ { d } } y _ { i } \hat { \mu } _ { x } ( \mathrm { d } y ) \big ) ^ { \frac { 1 } { 6 } } .\tag{2.16}
$$

The proof of Theorem 2.1 is mainly based on the generalization error bound of the local squared $W _ { 2 }$ distance ([30, Theorem 2.3]) and is given in Appendix B. Theorem 2.1 provides a generalization error bound on estimating how the empirical Sinkhorn divergence loss function $\bar { S } _ { \varepsilon , \delta } ^ { \mathrm { e } } .$ , when utilizing the neighborhood technique, approximates the averaged squared $W _ { 2 }$ distance between the ground truth random field ${ \pmb y } _ { \pmb x }$ and the approximate model $\hat { y } _ { x }$

On the other hand, if we use the generalization error bound for estimating the Sinkhorn divergence using the empirical distributions, we can obtain another generalization error bound for our local Sinkhorn divergence loss function.

Theorem 2.2 (Generalization bound based on the generalization error of the Sinkhorn divergence). For the averaged local Sinkhorn loss defined in Eq. (2.12) and the corresponding population averaged squared $W _ { 2 }$ distance defined in Eq. (2.9), assume that the hypotheses of [11, Theorem 3] hold, and $\varepsilon < \frac { 2 M _ { 0 } ^ { 2 } } { \sqrt { d } }$ . Then the following generalization error bound holds:

$$
\mathbb { E } \bigg [ \bigg \lvert \overline { { S } } _ { \varepsilon , \delta } ^ { \textnormal { e } } ( \mathbf { y } _ { x } , \hat { y } _ { x } ) - \overline { { \mathcal { W } } } _ { 2 } ( y _ { x } , \hat { y } _ { x } ) \bigg \rvert \bigg ] \leq 2 \varepsilon \log \bigg ( \frac { 2 e ^ { 2 } M _ { 0 } ^ { 2 } } { \sqrt { d } \varepsilon } \bigg ) + \mathcal { O } \bigg ( \frac { e ^ { \frac { S M _ { 0 } ^ { 2 } } { \varepsilon } } } { \sqrt { N ( x , \delta ) } } \bigg ( 1 + \varepsilon ^ { - \lfloor d / 2 \rfloor } \bigg ) \bigg ) + 1 6 M _ { 0 } ^ { 2 } \delta ,\tag{2.17}
$$

where $N ( { \pmb x } , \delta )$ denotes the efective number of samples in each neighborhood (defined in Eq. (2.14)).

The proof of Theorem 2.2 is mainly based on the generalization error bound of using the empirical measures to evaluate the Sinkhorn divergence ([11, Theorem 3]) and is given in Appendix B.

To exemplify how the size of the neighborhood � and the number of samples in each neighborhood $N ( { \pmb x } , \delta )$ influences the generalization error bound, consider the case when the empirical probability measure of the input variable is close to the uniform distribution and the neighborhood size satisfies $N ( { \pmb x } , \delta ) \asymp C N \delta ^ { n }$ (� is the total number of training samples) for the intrinsic dimension � of the input variable �, then using Theorem 2.1, we have:

$$
\mathbb { E } \bigg [ \bigg \lvert \overline { { S } } _ { \varepsilon , \delta } ^ { \mathrm { e } } ( y _ { x } , \hat { y } _ { x } ) - \overline { { \mathcal { W } } } _ { 2 } ( y _ { x } , \hat { y } _ { x } ) \bigg \rvert \bigg ] \lesssim 2 \varepsilon \log \Biggl ( \frac { 2 e ^ { 2 } M _ { 0 } ^ { 2 } } { \sqrt { d } \varepsilon } \Biggr ) + \frac { 4 M _ { 0 } } { \sqrt { N } } + 8 C M _ { 0 } h ( C N \delta ^ { n } , d ) + 1 6 M _ { 0 } ^ { 2 } \delta .\tag{2.18}
$$

On the other hand, using Theorem 2.2 yields:

$$
\mathbb { E } \bigg [ \bigg \lvert \overline { { S } } _ { \varepsilon , \delta } ^ { \textnormal { e } } ( \mathbf { y } _ { x } , \hat { y } _ { x } ) - \overline { { \mathcal { W } } } _ { 2 } ( \mathbf { y } _ { x } , \hat { y } _ { x } ) \bigg \rvert \bigg ] \lesssim 2 \varepsilon \log \Bigg ( \frac { 2 e ^ { 2 } M _ { 0 } ^ { 2 } } { \sqrt { d } \varepsilon } \Bigg ) + \mathcal { O } \Bigg ( \frac { e ^ { \frac { 5 M _ { 0 } ^ { 2 } } { \varepsilon } } } { \sqrt { C N \delta ^ { n } } } \Big ( 1 + \varepsilon ^ { - \lfloor d / 2 \rfloor } \Big ) \Bigg ) + 1 6 M _ { 0 } ^ { 2 } \delta .\tag{2.19}
$$

The two bounds Eqs. (2.18) and (2.19) suggest an important practical implication. When the data are highly heterogeneous, the conditional distributions may vary sharply across the domain. When the approximate random field model can capture such heterogeneity such that $\begin{array} { r } { \prod _ { i = 1 } ^ { d } ( \frac { \sigma _ { i , x } } { \sigma _ { 1 , x } } ) ^ { \frac { 1 } { d } } + \prod _ { i = 1 } ^ { d } ( \frac { \hat { \sigma } _ { i , x } } { \hat { \sigma } _ { 1 , x } } ) ^ { \frac { 1 } { d } } } \end{array}$ is small, the approximation error induced by overly large entropic regularization when � is large can dominate the generalization error. In this regime, it is preferable to choose a relatively small regularization parameter �, so that the local Sinkhorn loss remains close to the local squared $W _ { 2 }$ loss. Then, compared with the local squared $W _ { 2 }$ loss, the additional generalization error induced is essentially only the Sinkhorn regularization bias, while the neighborhood-based approximation and sampling errors remain unchanged.

On the other hand, when the noise in the target variable ${ \pmb y } _ { \pmb x }$ is homogeneous, the conditional distributions vary more smoothly and a moderately large regularization parameter, $\begin{array} { r } { \mathbf { e . g . } \varepsilon = O ( 1 ) } \end{array}$ at the scale of the cost function, can be advantageous, so that the convergence rate as the number of training samples $N \to \infty { \mathrm { ~ i s ~ } } \mathcal { O } \big ( ( N \delta ^ { n } ) ^ { - \frac { 1 } { 2 } } \big )$ . In this case, entropic smoothing improves numerical stability and can reduce the efective sample complexity penalty associated with the Sinkhorn approximation. As a result, by tuning �, the local Sinkhorn loss provides a flexible trade-of between statistical accuracy and computational robustness, and may partially alleviate the curse of dimensionality compared with the exact local squared $W _ { 2 }$ loss. Furthermore, as � decreases, the proposed local Sinkhorn loss approaches the local squared Wasserstein distance, leading to higher geometric fidelity but slower statistical convergence. Conversely, increasing � improves the statistical convergence and computational eficiency of the empirical Sinkhorn divergence while introducing additional regularization bias. Therefore, selecting a moderate regularization parameter also provides a favorable compromise between approximation accuracy and computational eficiency.

Algorithm 1 The pseudocode of the local Sinkhorn approach to train an SNN   
Require: Given � observed data $\{ ( \pmb { x } _ { i } , \pmb { y } _ { \pmb { x } _ { i } } ) \} _ { i = 1 } ^ { N }$ , the neighborhood radius $\delta ,$ the mini-batch size $n _ { b } ,$ the number of   
local samples �, the minimum number of local samples required for each center $n _ { \mathrm { m i n } }$ , the maximal number of local   
samples recorded for each center $n _ { \mathrm { m a x } } ,$ the number of epochs �, and the Sinkhorn regularization parameter �   
1: Initialize the SNN in Fig. 1   
2: For each $x _ { i } ,$ find its neighborhood $B _ { i } : = \{ \pmb { x } _ { j } : \| \pmb { x } _ { j } - \pmb { x } _ { i } \| \leq \delta \}$ . Do not use $\pmb { x } _ { i }$ as an eligible training neighborhood   
center if its neighborhood contains too few samples $( | B _ { i } | < n _ { \operatorname* { m i n } } )$ . If there are too many neighbors such that   
$| B _ { i } | > n _ { \operatorname* { m a x } } ,$ randomly keep $n _ { \mathrm { m a x } }$ neighbors to form the updated $B _ { i }$   
3: Input $\{ \pmb { x } _ { i } \} _ { i = 1 } ^ { N }$ into the SNN model to obtain predictions $\{ \bar { \hat { y } } _ { i } \} _ { i = 1 } ^ { N }$   
4: for epoch ${ \stackrel { } { = } } 1 , \dots , E$ do   
5: Randomly select � neighborhood centers $n _ { b }$   
6: for each selected center $\pmb { x } _ { i }$ do   
7: Randomly choose min(�, $| B _ { i } | )$ samples from $B _ { i }$ to construct a local mini-batch $( B ( \pmb { x } _ { i } , \delta )$ defined in   
Eq. (2.11))   
8: Input the sampled local inputs into the SNN to obtain predictions   
9: Compute the local Sinkhorn loss   
$\mathcal { L } _ { i } = S _ { \varepsilon } \left( \hat { \mu } _ { x _ { i } , \delta } ^ { \mathrm { e } } , \mu _ { x _ { i } , \delta } ^ { \mathrm { e } } \right)$   
10: end for   
11: Calculate the total loss   
$\mathcal { L } = \frac { 1 } { n _ { b } } \sum _ { i = 1 } ^ { n _ { b } } \mathcal { L } _ { i }$   
12: Perform gradient descent to minimize  and update the parameters in the SNN   
13: Resample the stochastic weights in the SNN using the updated parameter distributions   
14: end for   
15: return The trained SNN

Finally, the neighborhood radius � must be chosen to balance two competing sources of error. As shown in Eqs. (2.18) and (2.19), increasing $\delta$ enlarges the number of samples within each neighborhood and therefore reduces the empirical estimation error, namely 8� $M _ { 0 } h ( C N \delta ^ { n } , d )$ and $\mathcal { O } \Bigg ( \frac { \exp \Bigl ( 5 M _ { 0 } ^ { 2 } / \varepsilon \Bigr ) } { \sqrt { C N \delta ^ { n } } } \left( 1 + \varepsilon ^ { - \lfloor d / 2 \rfloor } \right) \Bigg )$ . However, a larger neighborhood inevitably increases the systematic approximation error, quantified by the term $1 6 M _ { 0 } ^ { 2 } \delta$ . Consequently, an appropriate neighborhood radius should strike a balance between statistical accuracy and locality: it should not be too small to provide adequate samples for reliable estimation of the local empirical distribution, while remaining not too large to preserve the local structure of the underlying conditional distribution and avoid excessive approximation bias.

## 3. Numerical experiments

In this section, we carry out numerical experiments to test our proposed local Sinkhorn approach for training SNNs. Test settings and hyperparameters are listed in Table 4. Pseudocode for minimizing our proposed local Sinkhorn to train the SNN in Fig. 1 is given in Algorithm 1.

In Examples 1 and 2, the reconstruction accuracy on the testing set is evaluated by comparing the empirical conditional means and variances of the reconstructed and reference distributions. Let $\{ ( \bar { \mathbf { x } } _ { i } ^ { \mathrm { t e s t } } , \mathbf { y } _ { i } ^ { \mathrm { t e s t } } ) \}$ denote samples in the testing set, and $B ^ { \mathrm { t e s t } } ( { \pmb x } _ { i } ^ { \mathrm { t e s t } } , 0 )$ denote the set of observations with the identical input as $\mathbf { \boldsymbol { x } } _ { i } \mathrm { : }$

$$
\{ \pmb { x } _ { j } ^ { \mathrm { t e s t } } : \pmb { x } _ { j } ^ { \mathrm { t e s t } } = \pmb { x } _ { i } ^ { \mathrm { t e s t } } \} .\tag{3.1}
$$

The ground-truth and predicted means are:

$$
\pmb { m } _ { i } = \frac { 1 } { R } \sum _ { x _ { j } ^ { \mathrm { t e s t } } \in B ( x _ { i } ^ { \mathrm { t e s t } } , 0 ) } \pmb { y } _ { j } ^ { \mathrm { t e s t } } , \qquad \hat { \pmb { m } } _ { i } = \frac { 1 } { R } \sum _ { x _ { j } ^ { \mathrm { t e s t } } \in B ( x _ { i } ^ { \mathrm { t e s t } } , 0 ) } \hat { \pmb { y } } _ { j } ^ { \mathrm { t e s t } }\tag{3.2}
$$

be the empirical conditional mean vectors at the $i ^ { \mathrm { { t h } } }$ testing center, respectively. Similarly, let

$$
v _ { i } = \frac { 1 } { R } \sum _ { x _ { j } ^ { \mathrm { t e s t } } \in B ( x _ { i } ^ { \mathrm { t e s t } } , 0 ) } \left( y _ { j } ^ { \mathrm { t e s t } } - m _ { i } \right) ^ { \circ 2 } , \qquad \hat { v } _ { i } = \frac { 1 } { R } \sum _ { x _ { j } ^ { \mathrm { t e s t } } \in B ( x _ { i } ^ { \mathrm { t e s t } } , 0 ) } \left( \hat { y } _ { j } ^ { \mathrm { t e s t } } - \hat { m } _ { i } \right) ^ { \circ 2 } ,\tag{3.3}
$$

denote the empirical variance vectors at the $i ^ { \mathrm { { t h } } }$ testing center, where $( \cdot ) ^ { \circ 2 }$ denotes the element-wise square. In Eqs. (3.2) and (3.3), $R : = | B ^ { \mathrm { t e s t } } ( x _ { i } ^ { \mathrm { t e s t } } , 0 ) |$ is the number of testing samples at each testing center. The average relative mean and variance reconstruction error is defined as follows:

$$
\mathcal { E } _ { \mathrm { m e a n } } = \frac { 1 } { N _ { c } } \sum _ { i = 1 } ^ { N _ { c } } \frac { \| \hat { \pmb { m } } _ { i } - \pmb { m } _ { i } \| _ { 2 } } { \| \pmb { m } _ { i } \| _ { 2 } + \epsilon _ { 0 } } , ~ \mathcal { E } _ { \mathrm { v a r } } = \frac { 1 } { N _ { c } } \sum _ { i = 1 } ^ { N _ { c } } \frac { \| \hat { \pmb { v } } _ { i } - \pmb { v } _ { i } \| _ { 2 } } { \| \pmb { v } _ { i } \| _ { 2 } + \epsilon _ { 0 } } ,\tag{3.4}
$$

where $N _ { c }$ is the number of testing centers, and $\epsilon _ { 0 }$ is set as $1 0 ^ { - 8 }$ to avoid division by zero.

First, we consider learning a unidimensional random field model to compare with diferent loss functions.

Example 1. We consider a one-dimensional conditional distribution reconstruction problem. The objective is to learn the conditional distribution

$$
p ( y \mid x ) , ~ x \in [ 0 , 1 ] .\tag{3.5}
$$

Unlike conventional regression problems where only the conditional mean is estimated, our objective is to reconstruct the entire conditional distribution from scattered observations. The conditional response is generated from a bimodal Gaussian mixture model,

$$
y = m ( x ) + s d ( x ) + \varepsilon _ { 0 } ,\tag{3.6}
$$

where the random variable $s \in \{ - 1 , + 1 \}$ is sampled with equal probability,

$$
\mathbb { P } ( s = - 1 ) = \mathbb { P } ( s = 1 ) = \frac { 1 } { 2 } ,
$$

and $\varepsilon _ { 0 } \sim \mathcal { N } ( 0 , \sigma ^ { 2 } )$ with $\sigma = 0 . 0 4$ . In Eq. (3.6), the conditional mean function is chosen as

$$
m ( x ) = 0 . 5 0 + 0 . 2 0 x + \exp \left[ - 5 ( x - 0 . 6 0 ) ^ { 2 } \right] + 0 . 4 0 \sin \left( \frac { x } { 2 } \right) ,\tag{3.7}
$$

which produces a smooth nonlinear trend over the computational domain. The separation between the two Gaussian modes is controlled by

$$
d ( x ) = 0 . 3 8 + 0 . 1 0 \exp \left( - \frac { ( x - 0 . 7 0 ) ^ { 2 } } { 2 \cdot ( 0 . 1 4 ) ^ { 2 } } \right) ,\tag{3.8}
$$

which allows the distance between the two modes to vary smoothly with the input variable. Consequently, the conditional distribution is given by

$$
p ( y | x ) = \frac { 1 } { 2 } \mathcal { N } \left( m ( x ) - d ( x ) , \sigma ^ { 2 } \right) + \frac { 1 } { 2 } \mathcal { N } \left( m ( x ) + d ( x ) , \sigma ^ { 2 } \right) .\tag{3.9}
$$

For the training set, $N _ { \mathrm { t r a i n } } = 2 0 0 0$ samples are independently generated by first sampling $\boldsymbol { x } _ { i } \sim \mathcal { V } ( \boldsymbol { 0 } , \boldsymbol { 1 } )$ , followed by drawing $y _ { i } \sim p ( y | x _ { i } )$ . Therefore, each training sample is independent. To generate the test set, 100 uniformly distributed input locations are selected as neighborhood centers. At each testing location, 20 independent realizations are generated from the conditional distribution (Eq. 3.9), providing empirical estimates of the local conditional distributions used for quantitative evaluation.

To investigate the influence of diferent discrepancy measures on conditional distribution learning, six diferent local loss functions are considered, namely local MSE, local MAE, local Energy Distance, local MMD, local squared $W _ { 2 } ,$ , and the proposed local Sinkhorn divergence. The mathematical definitions of these losses are summarized in Appendix D. Fig. 2 summarizes the overall performance of the six loss functions. Fig. 2 (a) compares the total training time. Owing to the entropic regularization, the proposed Local Sinkhorn loss requires significantly less computational efort than the previous local squared $W _ { 2 }$ loss in [31]. Fig. 2 (b) reports the average relative errors of the conditional mean and conditional variance on the testing set. Distribution-based losses consistently outperform pointwise regression losses (Local MSE and Local MAE), indicating that matching local empirical distributions is substantially more efective than matching individual samples. Among all competing losses, the proposed local Sinkhorn loss achieves the smallest overall errors for both the conditional mean and conditional variance. Fig. 2 (c) compares the reconstructed conditional distributions obtained by the proposed Local Sinkhorn model with the reference solution. The predicted mean accurately follows the true nonlinear trend while the predicted standard-deviation band agrees closely with the reference distribution over the entire computational domain. These results demonstrate that the proposed local Sinkhorn formulation is capable of accurately reconstructing both the location and spread of the conditional distribution.

Next, we carry out an experiment in which the target random variable ${ \pmb y } _ { \pmb x }$ is multidimensional and compare our local Sinkhorn approach with other machine-learning uncertainty quantification benchmarks.

Example 2. We consider a stochastic Darcy flow problem on the spatial domain $\pmb { x } \in \mathcal { D } = [ 0 , 1 ] ^ { 2 }$ . For each realization of the uncertain model parameters $\omega \in \Omega$ , we solve:

$$
- \nabla \cdot ( a ( \boldsymbol { x } , \omega ) \nabla u ( \boldsymbol { x } , \omega ) ) = f ( \boldsymbol { x } ) , \qquad \boldsymbol { x } \in D ,\tag{3.10}
$$

subject to homogeneous Dirichlet boundary conditions

$$
u ( \mathbf { x } , \omega ) = 0 , \qquad \mathbf { x } \in \partial D .\tag{3.11}
$$

The permeability field is modeled as:

$$
a ( \mathbf { x } , \omega ) = \exp ( g ( \mathbf { x } , \omega ) ) ,\tag{3.12}
$$

where $g ( \pmb { x } , \omega )$ is a zero-mean Gaussian random field generated by FFT-based spectral synthesis. Specifically, the random field is constructed in the Fourier domain as:

$$
\widehat { g } ( \boldsymbol { k } , \omega ) = \sqrt { c ( \boldsymbol { k } ) } \widehat { \xi } ( \boldsymbol { k } , \omega ) ,\tag{3.13}
$$

where $\widehat { \xi } ( \boldsymbol { k } , \omega )$ are independent complex Gaussian random variables satisfying $\widehat { \xi } ( k , \omega ) \sim \mathcal { N } ( 0 , 1 )$ , and the covariance spectrum is prescribed by

$$
\begin{array} { r } { c ( k ) = \left( 1 + ( 2 \pi \ell ) ^ { 2 } | k | ^ { 2 } \right) ^ { - \alpha / 2 } , } \end{array}\tag{3.14}
$$

with correlation length $\ell$ and smoothness parameter �. The Gaussian random field is then obtained by applying the inverse Fourier transform,

$$
\begin{array} { r } { g ( \pmb { x } , \omega ) = \mathcal { F } ^ { - 1 } \big ( \widehat { g } ( k , \omega ) \big ) , } \end{array}\tag{3.15}
$$

followed by normalization to zero mean and unit variance.

To generate the training and testing set, we establish a uniform 64 × 64 grid on  denoted as $D _ { 6 4 } = \{ \pmb { x } _ { i , j } =$ $( \frac { i } { 6 3 } , \frac { j } { 6 3 } ) \} _ { i , j = 0 } ^ { 6 4 }$ . Then, we randomly sample $\boldsymbol { x } _ { i , j }$ from $D _ { 6 4 }$ satisfying $6 \leq i , j \leq 5 7$ . For the chosen $\boldsymbol { x } _ { i , j }$ , a square window of size $7 \times 7$ is first constructed on the underlying computational grid. A total of 16 grid points $\mathbf x _ { i ^ { \prime } , j ^ { \prime } }$ are then randomly

Comparison of diferent conditional distribution learning methods and loss function used to train the SNN on the Darcy flow benchmark.
<table><tr><td>Method</td><td>Mean Error</td><td>Variance Error</td><td>Training Time (s)</td><td>Memory (MB)</td></tr><tr><td>Heteroscedastic Gaussian Regression</td><td>0.1189</td><td>0.5104</td><td>23.48</td><td>687.38</td></tr><tr><td>Mixture Density Network</td><td>0.0970</td><td>5.3101</td><td>52.06</td><td>686.72</td></tr><tr><td>Conditional VAE</td><td>0.2931</td><td>5.8687</td><td>31.10</td><td>695.09</td></tr><tr><td>Conditional Normalizing Flow</td><td>0.3256</td><td>3.5312</td><td>140.36</td><td>689.73</td></tr><tr><td> $\mathsf { S N N } + \mathsf { L o c a l } \ W _ { 2 }$ </td><td>0.0519</td><td>0.3213</td><td>500.84</td><td>905.14</td></tr><tr><td> $\mathsf { S N N } + \mathsf { S i n k h o r n }$ </td><td>0.0473</td><td>0.2318</td><td>308.34</td><td>791.08</td></tr><tr><td> $\mathsf { S N N } + \mathsf { M M D }$ </td><td>0.0779</td><td>0.7878</td><td>649.60</td><td>733.90</td></tr></table>

sampled without replacement from this window. This sampling strategy is adopted to ensure that most training samples have enough neighbors around them. For each selected grid point $\pmb { x } _ { i ^ { \prime } , j ^ { \prime } } , \mathrm { a } \ 5 \times 5$ solution patch on the grid is extracted with the selected grid point being the center. The corresponding solution patch is defined as:

$$
y ( x _ { i ^ { \prime } , j ^ { \prime } } , \omega _ { i ^ { \prime } , j ^ { \prime } } ) = \left( u ( x _ { i ^ { \prime } - 2 , j ^ { \prime } - 2 } , \omega _ { i ^ { \prime } - 2 , j ^ { \prime } - 2 } ) , u ( x _ { i ^ { \prime } - 2 , j ^ { \prime } - 1 } , \omega _ { i ^ { \prime } - 2 , j ^ { \prime } - 1 } ) , \ldots , u ( x _ { i ^ { \prime } + 2 , j ^ { \prime } + 2 } , \omega _ { i ^ { \prime } + 2 , j ^ { \prime } + 2 } ) \right) ^ { \top } \in \mathbb { R } ^ { 2 5 } .\tag{3.16}
$$

An illustration of the training and testing set sample generation strategy is given in Fig. 3 (a). The vector ${ \pmb y } ( { \pmb x } , \omega )$ serves as the target random field whose conditional distribution is to be reconstructed when � is given. To generate the testing set, 20 testing centers are generated. For each testing center $\pmb { x } _ { \ell } ^ { \mathrm { t e s t } }$ , a large number of independent permeability realizations $\omega _ { 1 } , \ldots , \omega _ { M }$ are generated. The resulting collection

$$
\left\{ \pmb { y } ( \pmb { x } _ { \ell } ^ { \mathrm { t e s t } } , \omega _ { \ell } ) \right\} _ { j = 1 } ^ { M }\tag{3.17}
$$

provides an empirical approximation of the conditional distribution $p { \Big ( } y \mid x _ { \ell } ^ { \mathrm { t e s t } } { \Big ) }$ . The learned model is then evaluated by comparing the predicted conditional distribution with this empirical reference distribution at the test locations. This benchmark naturally produces multidimensional output variables with strong spatial correlations, making it a suitable experiment to test if our local Sinkhorn approach could successfully reconstruct a multidimensional random field in which the noise is strongly heterogeneous or lies in a low-dimensional manifold.

To evaluate the efectiveness of the proposed local Sinkhorn divergence, we compare it with several representative approaches for conditional distribution reconstruction. The compared loss functions include local Maximum Mean Discrepancy (MMD) and the local squared $W _ { 2 }$ loss proposed in our previous work [31], whose definitions are detailed in Appendix D. To further compare with representative probabilistic learning approaches, we also consider Heteroscedastic Gaussian Regression, Mixture Density Networks (MDN) [4], Conditional Variational Autoencoders (CVAE) [24], and Conditional Normalizing Flows (CNF) [22, 7, 19]. These methods represent commonly used likelihood-based and latent-variable models for conditional distribution estimation. All benchmark methods are trained and evaluated using the same datasets and testing protocols to ensure a fair comparison.

Table 2 compares the proposed neighborhood-based distribution learning methods with several representative conditional generative baselines. Among the likelihood-based models, the heteroscedastic Gaussian regression is computationally the most eficient, requiring only 23.48 seconds for training, but its Gaussian assumption limits the reconstruction accuracy, especially for the variance. MDN improves the mean prediction accuracy but still sufers from poor variance reconstruction. CVAE and CNF provide more expressive conditional generative models, although their reconstruction errors remain significantly larger than those of the neighborhood-based approaches.

The neighborhood-based optimal transport methods consistently outperform the global conditional generative models. In particular, training the SNN with the local Sinkhorn divergence achieves the smallest errors in the predicted mean and variance. Furthermore, compared to the previous local squared $W _ { 2 }$ approach, its computational eficiency is improved, and the runtime is shortened. The proposed neighborhood-based Sinkhorn approach to train SNN provides the best overall trade-of between reconstruction accuracy and computational eficiency. These results suggest that once local conditional distributions are exploited with the neighborhood technique, the simple SNN can be eficiently trained using our local Sinkhorn loss, which is suficient to outperform considerably more sophisticated machine-learning conditional generative models. Overall, the experimental results demonstrate that exploiting local neighborhoods when utilizing the OT-based loss functions is considerably more important than increasing the complexity of the conditional generator.

To investigate the robustness of the proposed local Sinkhorn framework with respect to diferent algorithmic parameters, we further perform sensitivity studies on the noise level of the training data, the neighborhood radius, and the Sinkhorn regularization parameter. The corresponding results are presented in Fig. 3. For the noise sensitivity study, we vary the intensity of the underlying Gaussian random field. The permeability field is constructed from

$$
g _ { \sigma } ( \boldsymbol { x } , \omega ) = \sigma g ( \boldsymbol { x } , \omega ) ,\tag{3.18}
$$

where $\sigma > 0$ is referred to as the noise level and $g$ denotes the normalized Gaussian random field generated by the FFT-based spectral synthesis in Eq. (3.15). Consequently, the permeability field is given by

$$
\begin{array} { r } { a ( \mathbf { x } , \omega ) = \exp \left( g _ { \sigma } ( \mathbf { x } , \omega ) \right) . } \end{array}
$$

Increasing � enlarges the variance of the permeability field and therefore increases the intrinsic uncertainty of the corresponding Darcy solutions.

Fig. 3 (b) illustrates the influence of the observation noise level. As expected, increasing the noise level deteriorates the reconstruction accuracy of both the conditional mean and variance, while the training time increases only moderately. This indicates that the proposed method remains computationally stable even under relatively noisy observations. Fig. 3 (c) studies the efect of the neighborhood radius �. A suficiently small neighborhood does not contain enough local samples to accurately approximate the conditional distribution, whereas an excessively large neighborhood violates the locality assumption and mixes samples associated with diferent conditional distributions. Consequently, an intermediate neighborhood radius provides the best trade-of between approximation accuracy and computational cost. Fig. 3 (d) shows the sensitivity of the proposed method to the Sinkhorn regularization parameter $\varepsilon .$ . When $\varepsilon$ is too small, the Sinkhorn divergence approaches the exact Wasserstein distance. In this regime, the reconstructed variance error is large because the empirical squared $W _ { 2 }$ distance converges slowly to the ground truth squared $W _ { 2 }$ distance in multidimensional settings. Furthermore, choosing a too small � leads to longer runtime. Conversely, a large regularization parameter introduces excessive entropic smoothing and leads to larger reconstruction errors. These observations agree well with the theoretical error analysis in Section 2, where the regularization parameter controls the trade-of between approximation bias and computational eficiency, and a moderate regularization parameter � is preferable to reconstruct multidimensional random field models. Overall, the proposed local Sinkhorn framework exhibits good robustness over a reasonably wide range of neighborhood radii and regularization parameters.

Finally, we consider reconstructing a stochastic dynamical system using our proposed local Sinkhorn approach.

Example 3. To further demonstrate the applicability of the proposed method to multidimensional stochastic dynamical systems, we consider a network of coupled nonlinear stochastic FHN oscillators in computational neuroscience [28, 1].

We consider a network consisting of $N _ { n } = 5$ coupled neurons, resulting in a 10-dimensional stochastic diferential equation. Let

$$
\begin{array} { r } { \pmb { x } ( t ; \pmb { x } _ { 0 } ; \omega ) = \left( v _ { 1 } ( t ) , \ldots , v _ { 5 } ( t ) , w _ { 1 } ( t ) , \ldots , w _ { 5 } ( t ) \right) ^ { \top } \in \mathbb { R } ^ { 1 0 } , } \end{array}
$$

where $v _ { i }$ denotes the membrane potential and $w _ { i }$ is the corresponding recovery variable. The governing stochastic system is

$$
\begin{array} { l } { { \displaystyle { \mathrm { d } } v _ { i } ( t ) = \left( v _ { i } - \frac { v _ { i } ^ { 3 } } { 3 } - w _ { i } + \sum _ { j = 1 } ^ { 5 } C _ { i j } ( \omega ) v _ { j } \right) \mathrm { d } t + \sigma _ { i } ( \omega ) { \mathrm { d } } B _ { i } ( t ) , } } \\ { { \displaystyle { \mathrm { d } } w _ { i } ( t ) = \varepsilon _ { i } ( \omega ) \left( v _ { i } + c - d w _ { i } \right) { \mathrm { d } } t , \qquad i = 1 , \dots , 5 , \ ( v _ { 1 } ( 0 ) , \dots , v _ { 5 } ( 0 ) , w _ { 1 } ( 0 ) , \dots , w _ { 5 } ( 0 ) ) = x _ { 0 } , } } \end{array}\tag{3.19}
$$

where $B _ { i } ( t )$ are independent standard Brownian motions, and $c = 0 . 7$ and $d = 0 . 8$ are fixed deterministic parameters. Unlike the classical FHN model, the coupling strengths $C _ { i j } ( \omega )$ , the difusion coeficients $\sigma _ { i } ( \omega )$ , and the recovery time scales $\varepsilon _ { i } ( \omega )$ are all regarded as random variables whose values may vary across diferent realizations, and � denotes the uncertainty in those model parameters. More precisely, $\left( C _ { i j } ( \omega ) , \sigma _ { i } ( \omega ) , \varepsilon _ { i } ( \omega ) \right)$ is independently sampled for every realization, yielding a stochastic drift field $a ( x , \omega )$ and a stochastic difusion field $b ( x , \omega )$ . Consequently, the dynamics can be written compactly as

$$
\mathrm { d } \boldsymbol { x } ( t ; x _ { 0 } ; \omega ) = a ( x ( t ; x _ { 0 } ; \omega ) , \omega ) \mathrm { d } t + b ( x ( t ; x _ { 0 } ; \omega ) , \omega ) \mathrm { d } B _ { t } , \ x ( 0 ; x _ { 0 } ; \omega ) = x _ { 0 } , \ t \in [ 0 , T ] ,\tag{3.20}
$$

where both the drift and difusion are random functions induced by the uncertain parameters, $\pmb { B } _ { t }$ is a five-dimensional standard Brownian motion, and $\mathbf { \boldsymbol { x } } _ { 0 }$ denotes the initial condition, with $\pmb { x } ( t ; \pmb { x } _ { 0 } ; \omega )$ explicitly indicating the dependence of the solution on $\pmb { x } _ { 0 } .$

The reference trajectories and predicted trajectories are generated using the Euler–Maruyama solver implemented in the torchsde package. For the training set, each trajectory is associated with an independently sampled initial condition and an independent realization of the random parameter vector �. The resulting dataset consists of $\left\{ \left( \pmb { x } _ { 0 } ^ { ( i ) } , \pmb { x } ^ { ( i ) } ( t _ { 1 } ; \pmb { x } _ { 0 } ^ { ( i ) } ; \omega ^ { ( i ) } ) , \dots , \pmb { x } ^ { ( i ) } ( t _ { N _ { t } } ; \pmb { x } _ { 0 } ^ { ( i ) } ; \omega ^ { ( i ) } ) \right) \right\} _ { i = 1 } ^ { N _ { \mathrm { t r a i n } } }$ , where $\pmb { x } ^ { ( i ) } ( t _ { k } ) \in \mathbb { R } ^ { 1 0 }$ is the snapshot of the trajectory at time $\begin{array} { r } { t _ { k } = \frac { k T } { N _ { t } } } \end{array}$

For the testing set, we randomly select 20 initial conditions. For each initial condition, 20 independent realizations of the random parameters are generated, producing empirical conditional distributions of the solution trajectories. Therefore, the testing data naturally approximates the conditional distribution $P \left( \boldsymbol { x } ^ { \mathrm { t e s t } } ( t ; \boldsymbol { x } _ { 0 } ; \omega ) \mid \boldsymbol { x } ^ { \mathrm { t e s t } } ( 0 ; \boldsymbol { x } _ { 0 } ; \omega ) = \boldsymbol { x } _ { 0 } \right)$ which serves as the reference distribution for evaluating the reconstructed stochastic dynamics. For both the training set and the testing set, we set the time horizon $T = 1$ and the time steps $N _ { t } = 1 0$ in Eq. (3.20).

To reconstruct the unknown stochastic dynamics, we construct the approximate model:

$$
\begin{array} { r } { \mathrm { d } \hat { \mathbf { x } } ( t ; x _ { 0 } ; \hat { \omega } ) = \hat { a } ( \hat { x } ( t ; x _ { 0 } ; \hat { \omega } ) , \hat { \omega } _ { a } ) \mathrm { d } t + \hat { b } ( \hat { x } ( t ; x _ { 0 } ; \hat { \omega } ) , \hat { \omega } _ { b } ) \mathrm { d } B _ { t } , \hat { \mathbf { x } } ( 0 ; x _ { 0 } ; \hat { \omega } ) = x _ { 0 } , } \end{array}\tag{3.21}
$$

where $\hat { a } ( \pmb { x } ( t ; \pmb { x } _ { 0 } ; \hat { \omega } ) , t ; \hat { \omega } _ { a } )$ and $\hat { b } ( { \pmb x } ( t ; { \pmb x } _ { 0 } ; \hat { \omega } ) , t ; \hat { \omega } _ { b } )$ are two separate SNNs and $\hat { \omega } = ( \hat { \omega } _ { a } , \hat { \omega } _ { b } )$ refers to the union of uncertain weights in ̂� and �<sup>̂</sup>. ̂� and �<sup>̂</sup> in Eq. (3.21) take the 10-dimensional state variable $\hat { \pmb x } ( t ; \pmb x _ { 0 } ; \hat { \omega } )$ and time � as the input and then output 10-dimensional approximate drift and difusion functions to approximate the stochastic drift and difusion coeficients in Eq. (3.20), respectively. Since the dynamics of $w _ { i }$ in $\operatorname { E q . }$ . (3.19) are noise-free, we enforce the last five components of �<sup>̂</sup> in Eq. (3.21) to be 0. During one trajectory simulation, a single realization of the network weights $\hat { \omega }$ is sampled and kept fixed throughout the entire time interval. Consequently, the randomness remains trajectory-wise rather than time-wise, which is consistent with the assumption that the underlying uncertain parameters are realization-dependent. The two SNNs are trained simultaneously by minimizing a temporal-average local Sinkhorn loss:

$$
\begin{array} { r l } & { \displaystyle \frac { 1 } { T } \int _ { 0 } ^ { T } S _ { \varepsilon , \delta } ^ { \mathrm { e } } ( x ( t ; x _ { 0 } ; \omega ) , \hat { x } ( t ; x _ { 0 } ; \hat { \omega } ) ) \mathrm { d } t \approx \frac { 1 } { N _ { T } } \sum _ { i = 1 } ^ { N _ { T } } \overline { { S } } _ { \varepsilon , \delta } ^ { \mathrm { e } } ( x \big ( \frac { i T } { N _ { T } } ; x _ { 0 } ; \omega ) , \hat { x } ( \frac { i T } { N _ { T } } ; x _ { 0 } ; \hat { \omega } ) \big ) } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad = \frac { 1 } { N _ { T } } \displaystyle \sum _ { i = 1 } ^ { N _ { T } } \int _ { D } S _ { \varepsilon } \big ( \mu _ { x _ { 0 } , \delta } ^ { \mathrm { e } } ( \frac { i T } { N _ { T } } ) , \hat { \mu } _ { x _ { 0 } , \delta } ^ { \mathrm { e } } ( \frac { i T } { N _ { T } } ) \big ) \nu ^ { \mathrm { e } } ( \mathrm { d } x _ { 0 } ) , } \end{array}\tag{3.22}
$$

where $\mathbf { \boldsymbol { x } } _ { 0 }$ represents the initial condition, and $\mu _ { x _ { 0 } , \delta } ^ { e } ( t )$ and $\widehat { \mu } _ { x _ { 0 } , \delta } ^ { e } ( t )$ denote the empirical distributions of $\pmb { x } ( t ; \pmb { x } _ { 0 } ; \omega )$ and $\widehat { \pmb x } ( t ; \pmb x _ { 0 } ; \widehat { \omega } )$ , respectively, constructed from trajectories whose initial conditions fall within the selected neighborhood subset $B ( \pmb { x } _ { 0 } , \delta ) \subseteq B ( \pmb { x } _ { 0 } , \delta )$

The proposed temporal-average local Sinkhorn loss Eq. (3.22) compares the empirical distributions of the predicted and reference trajectories at every observation time and then averages the discrepancies over the entire temporal domain. Compared with the temporal-average local squared $W _ { 2 }$ loss introduced in our previous work [31], which replaces the Sinkhorn divergence with the squared $W _ { 2 }$ distance in Eq. (3.22), the proposed Sinkhorn formulation provides a diferentiable approximation with substantially improved computational eficiency while preserving the geometric structure of optimal transport.

To further evaluate the reconstruction quality of the proposed method, Fig. 4 compares the predicted stochastic dynamics with the corresponding reference solution for a representative testing sample. In addition to the stochastic trajectories, the learned drift and difusion functions are also compared with the ground truth. Since the proposed model reconstructs conditional probability distributions rather than individual trajectories, the conditional mean and one-standard-deviation bands estimated from multiple stochastic realizations are reported. As shown in Fig. 4 (a)(b), the predicted trajectories accurately reproduce both the temporal evolution and the associated uncertainty of the stochastic

Comparison of the local Sinkhorn divergence versus the local squared $W _ { 2 }$ loss to train an SNN for reconstructing the FHN model Eq. (3.19).
<table><tr><td>Loss Function</td><td>Training Time  $( \mathsf { s } )$ </td><td>Drift  $W _ { 2 } ^ { 2 }$  Error</td><td>Diffusion  $W _ { 2 } ^ { 2 }$  Error</td><td>Test trajectory  $W _ { 2 } ^ { 2 }$  Error</td></tr><tr><td>Local Sinkhorn</td><td>6772</td><td>0.1110</td><td>0.1090</td><td>0.06415</td></tr><tr><td>Local squared  $W _ { 2 }$ </td><td>7256</td><td>0.1195</td><td>0.1791</td><td>0.06444</td></tr></table>

FHN system. For both the membrane potential variable $v _ { 1 }$ and the recovery variable $w _ { 1 }$ , the predicted conditional means closely follow the reference solution, while the predicted uncertainty bands exhibit good agreement with the corresponding ground-truth distributions. Furthermore, the learned drift functions capture the nonlinear deterministic dynamics with high accuracy (Fig. 4 (c)(d)), and the difusion functions provide reasonable approximations of the stochastic forcing, while larger errors in the predicted difusion functions may be due to error accumulation (shown in Fig. 4 (e)). Overall, these results demonstrate that the proposed local Sinkhorn divergence is capable of simultaneously reconstructing the deterministic and stochastic components of the underlying dynamics from scattered conditional observations.

We also compare minimizing our temporal-average local Sinkhorn divergence Eq. (3.22) with minimizing the previous temporal-average local squared $W _ { 2 }$ loss in [31]. The following error metrics are used to quantify the errors of �̂, ̂�, and $\hat { b }$ of the reconstructed SDE (3.21) on the testing set, respectively:

$$
\begin{array} { r l } & { \mathrm { e r r o r ~ i n ~ } \hat { x } : = \frac { \sum _ { j = 1 } ^ { N _ { \mathrm { t e x } } } \sum _ { i = 1 } ^ { N _ { T } } W _ { 2 } ^ { 2 } ( \mu _ { x _ { 0 , j } ^ { \mathrm { t e x } } } ^ { \mathrm { c } } ( \frac { i T } { N _ { T } } ) , \hat { \mu } _ { x _ { 0 , j } ^ { \mathrm { t e x } } } ^ { \mathrm { c } } ( \frac { i T } { N _ { T } } ) ) } { \sum _ { j = 1 } ^ { N _ { \mathrm { t e x } } } \sum _ { i = 1 } ^ { N _ { T } } [  x ( \frac { i T } { N _ { T } } ; \boldsymbol { \mathbf { x } } _ { 0 , j } ^ { \mathrm { t e x } } ; \boldsymbol { \omega } )  ^ { 2 } ] } , \quad \mathrm { e r r r o r ~ i n ~ } \hat { a } : = \frac { \sum _ { j = 1 } ^ { N _ { \mathrm { t e x } } } \sum _ { i = 1 } ^ { N _ { T } } W _ { 2 } ^ { 2 } ( \eta _ { a , x _ { 0 , j } ^ { \mathrm { t e x } } } ^ { \mathrm { c } } ( \frac { i T } { N _ { T } } ) , \hat { \eta } _ { a , x _ { 0 , j } ^ { \mathrm { t e x } } } ^ { \mathrm { c } } ( \frac { i T } { N _ { T } } ) ) } { \sum _ { j = 1 } ^ { N _ { \mathrm { t e x } } } \sum _ { i = 1 } ^ { N _ { T } } [  a ( { x } ( \frac { i T } { N _ { T } } ; \boldsymbol { \mathbf { x } } _ { 0 , j } ^ { \mathrm { t e x } } ; \boldsymbol { \omega } ) , \boldsymbol { \omega } )  ^ { 2 } ] } , } \\ &  \mathrm { e r r o r ~ i n ~ } \hat { b } : = \frac  \sum _ { j = 1 } ^ { N _ { \mathrm { t e x } } } \sum _ { i = 1 } ^ { N _ { T } } W _ { 2 } ^ { 2 } ( \eta _ { b , x _ { 0 , j } ^ { \mathrm { t e x } } } ^ \end{array}\tag{3.23}
$$

where $\pmb { x } _ { 0 , j } ^ { \mathrm { t e s t } }$ refers to the $j ^ { \mathrm { t e s t } }$ center in the testing set, $\mu _ { x _ { 0 , i } ^ { \mathrm { t e s t } } } ^ { \mathrm { e } } ( \frac { i T } { N _ { T } } ) , \hat { \mu } _ { x _ { 0 , i } ^ { \mathrm { t e s t } } } ^ { \mathrm { e } } ( \frac { i T } { N _ { T } } )$ denotes the empirical distributions of the ground-truth and reconstructed trajectories at time $\frac { i T } { N _ { T } }$ whose initial conditions are $x _ { 0 , j } ^ { \mathrm { t e s t } } , \eta _ { a , x _ { 0 , j } } ^ { \mathrm { e } }$ and $\hat { \eta } _ { a , { \boldsymbol x } _ { 0 , j } } ^ { \mathrm { e } }$ refers to the empirical distribution of $a ( \pmb { x } ( t ; \pmb { x } _ { 0 , j } ^ { \mathrm { t e s t } } ; \omega )$ , �) and $\hat { a } ( \pmb { x } ( t ; \pmb { x } _ { 0 , j } ^ { \mathrm { t e s t . } } ; \hat { \omega } ) , t ; \hat { \omega } )$ , and $\eta _ { b , x _ { 0 , j } } ^ { \mathrm { e } }$ and $\hat { \eta } _ { b , x _ { 0 , j } } ^ { \mathrm { e } }$ refers to the empirica distribution of $b ( \pmb { x } ( t ; \pmb { x } _ { 0 , i } ^ { \mathrm { t e s t } } ; \omega )$ , �) and $\hat { b } ( \pmb { x } ( t ; \pmb { x } _ { 0 , i } ^ { \mathrm { t e s t } } ; \hat { \omega } ) , t ; \hat { \omega } )$ , respectively.

As shown in Table 3, using the Sinkhorn divergence is more computationally eficient, and the learned drift and difusion functions are more accurate compared to those learned when using the temporally decoupled local squared $W _ { 2 }$ as the loss function. The saving in runtime of using the temporal local Sinkhorn divergence Eq. (3.22) is not significant compared to using the temporal local squared $W _ { 2 }$ loss, and the possible reason is that the time needed to numerically solve the SDE is much longer than evaluating either the temporal local Sinkhorn divergence loss or the temporal local $W _ { 2 }$ loss.

## 4. Summary and conclusion

In this paper, we proposed a local Sinkhorn divergence framework for conditional distribution reconstruction of random field models using SNNs. By utilizing the debiased Sinkhorn divergence, the proposed method preserved the geometric structure of optimal transport while introducing entropic regularization, leading to a diferentiable and computationally eficient distribution matching objective. Compared with our previous local squared $W _ { 2 }$ formulation, the proposed local Sinkhorn loss achieved improved computational eficiency and scalability while maintaining high reconstruction accuracy for learning multidimensional random field models. We further established theoretical error bounds for the proposed local Sinkhorn loss. The resulting bounds explicitly characterized the trade-of between approximation bias and statistical eficiency introduced by the entropic regularization parameter and implied that the proposed local Sinkhorn loss could partially alleviate the curse of dimensionality when learning multidimensional random field models by properly choosing the regularization parameter. Numerical experiments on one-dimensional conditional distribution reconstruction, stochastic Darcy flow, and stochastic FHN systems demonstrated that the proposed method accurately reconstructed random field models and outperformed several prevailing machinelearning uncertainty quantification benchmarks. Overall, the proposed local Sinkhorn approach ofered a practical balance between statistical accuracy, computational eficiency, and theoretical interpretability, making it a promising framework for uncertainty quantification and probabilistic scientific machine learning.

Several promising research directions deserve further investigation. First, the neighborhood radius � and the regularization parameter � are currently selected empirically and remain fixed. Developing adaptive strategies for simultaneously choosing and adjusting these two parameters according to the local complexity of the conditional distribution may further improve both accuracy and computational eficiency. Recent advances in stabilized and multiscale Sinkhorn algorithms provide promising directions for adaptive regularization and eficient optimal transport computations [23, 8]. Second, the current framework constructs neighborhoods using Euclidean distances. More general neighborhood constructions, including anisotropic neighborhoods, manifold-based neighborhoods, and graphbased local structures, may significantly improve the approximation of conditional distributions with low-dimensional intrinsic geometry [3, 27]. Such adaptive neighborhood constructions may also lead to improved theoretical convergence guarantees. Another important direction is to extend the proposed local Sinkhorn framework to large-scale scientific machine learning problems, including stochastic partial diferential equations, neural operator learning, probabilistic surrogate modeling, and Bayesian scientific machine learning. Since the proposed local Sinkhorn loss is fully diferentiable and naturally compatible with GPU-based optimization, it is particularly suitable for these large-scale applications [18, 21, 19]. Finally, it would be of considerable interest to further investigate the theoretical properties of local Sinkhorn learning in multidimensional settings.

## Declaration of generative AI and AI-assisted technologies in the manuscript preparation process

During the preparation of this work, the authors used ChatGPT in order to polish the writing and assist in checking the manuscript. After using this tool/service, the authors reviewed and edited the content as needed and take full responsibility for the content of the published article.

## Declaration of competing interest

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

## Data statement

The data used in this study were generated numerically according to the models and experimental settings described in the manuscript. The data and implementation code that support the findings of this study will be made publicly available upon publication of the article.

## CRediT authorship contribution statement

Mingtao Xia: Conceptualization, Methodology, Formal analysis, Investigation, Software, Writing-original draft, Writing-review and editing. Qijing Shen: Methodology, Formal analysis, Software, Writing-original draft, Writingreview and editing.

## A. Proof to Theorem 2.1

Here, we prove Theorem 2.1. We first decompose the generalization error as:

$$
\left. \overline { { S } } _ { \varepsilon , \delta } ^ { \mathrm { e } } ( y _ { x } , \widehat { y } _ { x } ) - \overline { { \mathcal { W } } } _ { 2 } ( y _ { x } , \widehat { y } _ { x } ) \right. \leq \left. \overline { { S } } _ { \varepsilon , \delta } ^ { \mathrm { e } } ( y _ { x } , \widehat { y } _ { x } ) - \overline { { \mathcal { W } } } _ { 2 , \delta } ^ { \mathrm { e } } ( y _ { x } , \widehat { y } _ { x } ) \right. + \left. \overline { { \mathcal { W } } } _ { 2 , \delta } ^ { \mathrm { e } } ( y _ { x } , \widehat { y } _ { x } ) - \overline { { \mathcal { W } } } _ { 2 } ^ { 2 } ( y _ { x } , \widehat { y } _ { x } ) \right. ,\tag{A.1}
$$

where $\overline { { \mathscr { W } } } _ { 2 } ( \pmb { y } _ { x } , \hat { \pmb { y } } _ { x } )$ is defined in Eq. (2.9), and $\overline { { \mathcal { W } } } _ { 2 , \delta } ^ { \mathrm { e } } ( \boldsymbol { y } _ { x } , \hat { \boldsymbol { y } } _ { x } )$ is the local squared $W _ { 2 }$ distance between the two random fields ${ \pmb y } _ { \pmb x }$ and $\hat { y } _ { x }$ defined as:

$$
\overline { { \mathscr { W } } } _ { 2 , \delta } ^ { \mathrm { e } } \bigl ( y _ { x } , \hat { y } _ { x } \bigr ) : = \int _ { D } W _ { 2 } ^ { 2 } \bigl ( \mu _ { x , \delta } ^ { \mathrm { e } } , \hat { \mu } _ { x , \delta } ^ { \mathrm { e } } \bigr ) \nu ( \mathrm { d } x ) .\tag{A.2}
$$

The first term in Eq. (A.1) is the Sinkhorn regularization bias. By [11, Theorem 1] and the boundedness assumption (Assumption 2.1), taking the Lipschitz constant of the quadratic cost function � in Eq. (2.4) as $2 M _ { 0 }$ yields

$$
0 \leq W _ { \varepsilon } ( \alpha , \beta ) - W ( \alpha , \beta ) \leq 2 \varepsilon \log \left( \frac { 2 e ^ { 2 } M _ { 0 } ^ { 2 } } { \sqrt { d } \varepsilon } \right) ,
$$

for every relevant pair of probability measures $( \alpha , \beta )$ . Since the debiased Sinkhorn divergence is obtained by subtracting the self-transport terms, the same bound controls the diference between the local Sinkhorn loss and its $W _ { 2 } ^ { 2 }$ counterpart, namely

$$
\left. \overline { { S } } _ { \varepsilon , \delta } ^ { \mathrm { e } } ( y _ { x } , \hat { y } _ { x } ) - \overline { { \mathcal { W } } } _ { 2 , \delta } ^ { \mathrm { e } } ( y _ { x } , \hat { y } _ { x } ) \right. \leq 2 \varepsilon \log \left( \frac { 2 e ^ { 2 } M _ { 0 } ^ { 2 } } { \sqrt { d } \varepsilon } \right) .\tag{A.3}
$$

The second term in Eq. (A.1) is exactly the local squared $W _ { 2 }$ generalization error. Therefore, applying [30, Theorem 2.3] in our previous work gives

$$
\mathbb { E } \bigg [ \Big \vert \overline { { \mathcal { W } } } _ { 2 , \delta } ^ { \mathrm { e } } ( y _ { x } , \hat { y } _ { x } ) - \overline { { \mathcal { W } } } _ { 2 } ( y _ { x } , \hat { y } _ { x } ) \Big \vert \bigg ] \leq \frac { 4 M _ { 0 } } { \sqrt { N } } + 8 C M _ { 0 } \mathbb { E } \big [ h ( N ( x , \delta ) , d ) \big ] + 1 6 M _ { 0 } ^ { 2 } \delta .
$$

Combining the Sinkhorn bias bound in Eq. (A.3) with the local $W _ { 2 }$ generalization bound above proves Eq. (2.13).

## B. Proof to Theorem 2.2

Here, we prove Theorem 2.2. We decompose the generalization error into three parts:

$$
\begin{array} { r l } { \mathrm { E } \Bigg [ \Bigg | \overline { { S } } _ { \varepsilon , \delta } ^ { \mathrm { e } } ( y _ { x } , \hat { y } _ { x } ) - \overline { { \mathcal { W } } } _ { 2 } ^ { 2 } ( y _ { x } , \hat { y } _ { x } ) \Bigg | \Bigg ] } & { \leq \mathrm { E } \Bigg [ \Bigg | \overline { { S } } _ { \varepsilon , \delta } ^ { \mathrm { e } } ( y _ { x } , \hat { y } _ { x } ) - \overline { { S } } _ { \varepsilon , \delta } ( y _ { x } , \hat { y } _ { x } ) \Bigg | \Bigg ] } \\ & { \qquad + \mathrm { E } \Bigg [ \Bigg | \displaystyle \int _ { D } S _ { \varepsilon } ( \mu _ { x , \delta } , \hat { \mu } _ { x , \delta } ) \nu ^ { \mathrm { e } } ( \mathrm { d } x ) - \int _ { D } W _ { 2 } ^ { 2 } ( \mu _ { x , \delta } , \hat { \mu } _ { x , \delta } ) \nu ^ { \mathrm { e } } ( \mathrm { d } x ) \Bigg | \Bigg ] } \\ & { \qquad + \mathrm { E } \Bigg [ \Bigg | \displaystyle \int _ { D } W _ { 2 } ^ { 2 } ( \mu _ { x , \delta } , \hat { \mu } _ { x , \delta } ) \nu ^ { \mathrm { e } } ( \mathrm { d } x ) - \int _ { D } W _ { 2 } ^ { 2 } ( \mu _ { x } , \hat { \mu } _ { x } ) \nu ^ { \mathrm { e } } ( \mathrm { d } x ) \Bigg | \Bigg ] . } \end{array}\tag{B.1}
$$

The first term is the averaged empirical Sinkhorn error. For each �, by applying [11, Theorem 3], we have:

$$
\mathbb { E } \Big [ \Big | S _ { \varepsilon } ( \mu _ { x , \delta } ^ { \mathrm { e } } , \hat { \mu } _ { x , \delta } ^ { \mathrm { e } } ) - S _ { \varepsilon } ( \mu _ { x , \delta } , \hat { \mu } _ { x , \delta } ) \Big | \Big ] \leq \mathcal { O } \Big ( \frac { e ^ { \frac { 5 M _ { 0 } ^ { 2 } } { \varepsilon } } } { \sqrt { N ( x , \delta ) } } \Big ( 1 + \varepsilon ^ { - \lfloor d / 2 \rfloor } \Big ) \Big ) .\tag{B.2}
$$

The second term is the average regularization bias between the Sinkhorn divergence and the Wasserstein discrepancy. From [11, Theorem 1], for each �, we have

$$
\operatorname { E } \Bigg [ \Big | S _ { \varepsilon } ( \mu _ { x , \delta } , \hat { \mu } _ { x , \delta } ) - \mathcal { W } _ { 2 } ^ { 2 } ( \mu _ { x , \delta } , \hat { \mu } _ { x , \delta } ) \Big | \Bigg ] \leq 2 \varepsilon \log \Bigg ( \frac { 2 e ^ { 2 } M _ { 0 } ^ { 2 } } { \sqrt { d } \varepsilon } \Bigg ) .\tag{B.3}
$$

Therefore,

$$
\mathrm { E } \Bigg [ \Bigg | \int _ { D } S _ { \varepsilon } ( \mu _ { x , \delta } , \hat { \mu } _ { x , \delta } ) \nu ^ { \mathrm { e } } ( \mathrm { d } x ) - \int _ { D } W _ { 2 } ^ { 2 } ( \mu _ { x , \delta } , \hat { \mu } _ { x , \delta } ) \nu ^ { \mathrm { e } } ( \mathrm { d } x ) \Bigg | \Bigg ] \leq 2 \varepsilon \log \Bigg ( \frac { 2 e ^ { 2 } M _ { 0 } ^ { 2 } } { \sqrt { d } \varepsilon } \Bigg ) .\tag{B.4}
$$

Finally, for each �, using the triangular inequality of the $W _ { 2 }$ distance [5], we have:

$$
| W _ { 2 } ( \mu _ { x , \delta } , \hat { \mu } _ { x , \delta } ) - W _ { 2 } ( \mu _ { x } , \hat { \mu } _ { x } ) | \leq W _ { 2 } ( \mu _ { x , \delta } , \mu _ { x } ) + W _ { 2 } ( \hat { \mu } _ { x } , \hat { \mu } _ { x , \delta } ) .\tag{B.5}
$$

Following the proof of [31, Theorem 2.3] and the Lipschitz continuity condition in Assumption 2.1, we have:

$$
W _ { 2 } ( \mu _ { x , \delta } , \mu _ { x } ) + W _ { 2 } ( \hat { \mu } _ { x } , \hat { \mu } _ { x , \delta } ) \leq 4 M _ { 0 } \delta .\tag{B.6}
$$

Therefore,

$$
\begin{array} { r l } & { | W _ { 2 } ^ { 2 } ( \mu _ { x , \delta } , \hat { \mu } _ { x , \delta } ) - W _ { 2 } ^ { 2 } ( \mu _ { x } , \hat { \mu } _ { x } ) | \leq | W _ { 2 } ( \mu _ { x , \delta } , \hat { \mu } _ { x , \delta } ) - W _ { 2 } ( \mu _ { x } , \hat { \mu } _ { x } ) | \cdot | W _ { 2 } ( \mu _ { x , \delta } , \hat { \mu } _ { x , \delta } ) + W _ { 2 } ( \mu _ { x } , \hat { \mu } _ { x } ) | } \\ & { \qquad \leq 2 ( 2 M _ { 0 } ) \delta ( 2 \operatorname* { s u p } \| y \| _ { 2 } + 2 \operatorname* { s u p } \| \hat { y } \| _ { 2 } ) = 1 6 M _ { 0 } ^ { 2 } \delta , } \end{array}\tag{B.7}
$$

which leads to:

$$
\mathrm { E } \Big [ \Big | W _ { 2 } ^ { 2 } ( \mu _ { x , \delta } , \hat { \mu } _ { x , \delta } ) - W _ { 2 } ^ { 2 } ( \mu _ { x } , \hat { \mu } _ { x } ) \Big | \Big ] \leq 1 6 M _ { 0 } ^ { 2 } \delta\tag{B.8}
$$

Combining Eqs. (B.8), (B.1), (B.2), and (B.4) yields Eq. (2.17).

Remark. The error bound induced from using the neighborhood technique Eq. (B.6) only applies to the Wasserstein distance through devising a special coupling of $\mu _ { x , \delta }$ and $\mu _ { x } .$ . However, there is no direct error bound for $S _ { \varepsilon } ( \mu _ { x , \delta } , \mu _ { x } )$ as the Sinkhorn divergence introduces the additional entropic Kullback-Leibler divergence term. Therefore, it is nontrivial to obtain a direct error bound of

$$
\left| \overline { { S } } _ { \varepsilon , \delta } ^ { \mathrm { e } } ( \mathbf { y } _ { \boldsymbol { x } } , \hat { \mathbf { y } } _ { \boldsymbol { x } } ) - \int _ { D } S _ { \varepsilon } ( \mu _ { \boldsymbol { x } } , \hat { \mu } _ { \boldsymbol { x } } ) \nu ( \mathrm { d } \boldsymbol { x } ) \right| ,\tag{B.9}
$$

where $\int _ { D } S _ { \varepsilon } ( \mu _ { x } , \hat { \mu } _ { x } ) \nu ( \mathrm { d } x )$ denotes the average Sinkhorn divergence between ${ \pmb y } _ { \pmb x }$ and $\hat { y } _ { x }$ .

## C. Optimization, training settings and hyperparameters

All numerical experiments are implemented in Python 3.11 using the PyTorch deep learning framework. The proposed Local Sinkhorn divergence is implemented using the GeomLoss package, while the $W _ { 2 }$ distance is computed using the POT package. Unless otherwise specified, all competing methods within each numerical example employ the same neural network architecture, optimizer, initialization strategy, and training settings. All experiments are performed on a desktop equipped with an Intel Core i9-13900KF CPU (24 cores) and 64 GB RAM. The runtime reported in the numerical examples corresponds to the actual wall-clock training time under the same hardware environment. Training settings and hyperparameters are listed in Table 4. In this work, the $W _ { 2 }$ distance is evaluated numerically using the ot.emd2 function in the POT package [9] and the Sinkhorn divergence is evaluated numerically using the SamplesLoss function in the Geomloss package [8].

Unless otherwise specified, all model parameters are initialized using the default PyTorch initialization strategy. The same initialization scheme, optimizer, learning rate, training epochs, and network architecture are adopted for all competing methods within each numerical example to ensure a fair comparison.

## D. Definitions of Diferent Loss Metrics

Here, we summarize the loss functions used in Examples 1 and 2. Let

$$
B ( \pmb { x } _ { i } , \delta ) = \{ \pmb { x } _ { j } : \| \pmb { x } _ { j } - \pmb { x } _ { i } \| \leq \delta \}
$$

denote the neighborhood centered at ${ \pmb x } _ { i } \left( { \bf E q . } \left( 2 . 1 0 \right) \right)$ . If a center $\pmb { x } _ { i }$ is selected when evaluating the loss function within an epoch, a subset $B ( \pmb { x } _ { i } , \delta ) \subseteq B ( \pmb { x } _ { i } , \delta )$ is randomly chosen with $| B ( \pmb { x } _ { i } , \delta ) | = \operatorname* { m i n } ( n , | B ( \pmb { x } _ { i } , \delta ) | )$ . The overall loss is obtained by averaging the corresponding local loss over all randomly selected neighborhood centers.

• 1. Local mean squared error (Local MSE)

$$
\mathrm { M S E } _ { \delta } ( y _ { x } , \hat { y } _ { x } ) = \frac { 1 } { n _ { b } } \sum _ { i = 1 } ^ { n _ { b } } \frac { 1 } { | \mathcal { B } ( x _ { i } , \delta ) | } \sum _ { x _ { j } \in \mathcal { B } ( x _ { i } , \delta ) } \| y ( x _ { j } ; \omega _ { j } ) - \hat { y } ( x _ { j } ; \hat { \omega } _ { j } ) \| ^ { 2 } .\tag{D.1}
$$

Training hyperparameters, neural network settings, and optimization configurations for all numerical examples.
<table><tr><td></td><td>Example 1</td><td>Example 2</td><td>Example 3</td></tr><tr><td>Optimizer</td><td>Adam</td><td>Adam</td><td>Adam</td></tr><tr><td>Learning rate</td><td> $5 \times 1 0 ^ { - 3 }$ </td><td> $5 \times 1 0 ^ { - 4 }$ </td><td> $1 \times 1 0 ^ { - 2 }$ </td></tr><tr><td>Weight decay</td><td>0</td><td>0</td><td>0</td></tr><tr><td>Number of epochs E</td><td>2000</td><td>20000</td><td>200</td></tr><tr><td>Training samples</td><td>2000</td><td>2000</td><td>300</td></tr><tr><td>Number of testing centers  $N _ { c }$ </td><td>100</td><td>20</td><td>20</td></tr><tr><td>Testing Realizations</td><td>20</td><td>100</td><td>20</td></tr><tr><td>Mini-batch size  $n _ { b }$ </td><td>8</td><td>4</td><td>8</td></tr><tr><td>Hidden layers</td><td>2</td><td>3</td><td>2</td></tr><tr><td>Hidden neurons</td><td>32</td><td>128</td><td>32</td></tr><tr><td>Activation function</td><td>ReLU</td><td>ReLU</td><td>GeLU</td></tr><tr><td>Residual connection (ResNet)</td><td>Yes</td><td>Yes</td><td>Yes</td></tr><tr><td>Neighborhood radius δ</td><td>0.05</td><td>0.05</td><td>0.1</td></tr><tr><td>Minimal samples required for each neighborhood  $n _ { \mathrm { m i n } }$ </td><td>4</td><td>4</td><td>4</td></tr><tr><td>Maximal samples recorded in each neighborhood  $n _ { \mathrm { m a x } }$ </td><td>128</td><td>64</td><td>300</td></tr><tr><td>Samples chosen in each neighborhood n</td><td>32</td><td>32</td><td>32</td></tr><tr><td>Regularization coefficient ε</td><td>0.05</td><td>0.03</td><td>0.1</td></tr><tr><td>SNN parameter initialization</td><td></td><td></td><td>N(0, 0.012) N(0, 0.052) N(0, 0.012)</td></tr></table>

• 2. Local mean absolute error (Local MAE)

$$
\mathrm { M A E } _ { \delta } ( \pmb { y } _ { x } , \hat { \pmb { y } } _ { x } ) = \frac { 1 } { n _ { b } } \sum _ { i = 1 } ^ { n _ { b } } \frac { 1 } { | \mathscr { B } ( \pmb { x } _ { i } , \delta ) | } \sum _ { \pmb { x } _ { j } \in \mathscr { B } ( \pmb { x } _ { i } , \delta ) } \| \pmb { y } ( \pmb { x } _ { j } ; \omega _ { j } ) - \hat { \pmb { y } } ( \pmb { x } _ { j } ; \hat { \omega } _ { j } ) \| .\tag{D.2}
$$

## • 3. Local energy distance (Local ED)

The energy distance follows from [26] and is defined locally by

$$
\operatorname { E D } _ { \delta } ( \pmb { y } _ { \pmb { x } } , \hat { \pmb { y } } _ { \pmb { x } } ) = \frac { 1 } { n _ { b } } \sum _ { i = 1 } ^ { n _ { b } } \operatorname { E D } \left( Y _ { i } ^ { \delta } , \hat { Y } _ { i } ^ { \delta } \right) ,\tag{D.3}
$$

where

$$
\begin{array} { r l } & { \mathrm { E D } \big ( Y _ { i } ^ { \delta } , \hat { Y } _ { i } ^ { \delta } \big ) = \displaystyle \frac { 2 } { | { \cal B } ( x _ { i } , \delta ) | ^ { 2 } } \sum _ { y _ { p } \in Y _ { i } ^ { \delta } , \hat { y } _ { q } \in \hat { Y } _ { i } ^ { \delta } } \| y _ { p } - \hat { y } _ { q } \| } \\ & { \quad \quad \quad \quad - \frac { 1 } { | { \cal B } ( x _ { i } , \delta ) | ^ { 2 } } \sum _ { y _ { p } \in Y _ { i } ^ { \delta } , \hat { y } _ { q } \in \hat { Y } _ { i } ^ { \delta } } \| y _ { p } - y _ { q } \| - \frac { 1 } { | { \cal B } ( x _ { i } , \delta ) | ^ { 2 } } \sum _ { y _ { p } \in Y _ { i } ^ { \delta } , \hat { y } _ { q } \in \hat { Y } _ { i } ^ { \delta } } \| \hat { y } _ { p } - \hat { y } _ { q } \| , } \end{array}\tag{D.4}
$$

with

$$
\begin{array} { r } { Y _ { i } ^ { \delta } = \left\{ { \pmb y } ( { \pmb x } _ { j } ; \omega _ { j } ) : { \pmb x } _ { j } \in B ( { \pmb x } _ { i } , \delta ) \right\} , \qquad \hat { Y } _ { i } ^ { \delta } = \left\{ \hat { \pmb y } ( { \pmb x } _ { j } ; \hat { \omega } _ { j } ) : { \pmb x } _ { j } \in B ( { \pmb x } _ { i } , \delta ) \right\} . } \end{array}\tag{D.5}
$$

## • 4. Local maximum mean discrepancy (Local MMD)

The MMD follows from the kernel two-sample test [15] and is defined locally by:

$$
\mathrm { M M D } _ { \delta } ( y _ { x } , \hat { y } _ { x } ) = \frac { 1 } { \left| \Gamma \right| } \sum _ { \gamma \in \Gamma } \frac { 1 } { n _ { b } } \sum _ { i = 1 } ^ { n _ { b } } \mathrm { M M D } _ { \gamma } \left( Y _ { i } ^ { \delta } , \hat { Y } _ { i } ^ { \delta } \right) ,\tag{D.6}
$$

where

$$
\begin{array} { l } { { \mathrm { M M D } _ { \gamma } \bigl ( Y _ { i } ^ { \delta } , \hat { Y } _ { i } ^ { \delta } \bigr ) = \displaystyle \frac { 1 } { | \mathscr { B } ( x _ { i } , \delta ) | ^ { 2 } } \sum _ { \substack { y _ { p } \in Y _ { i } ^ { \delta } , \hat { y } _ { q } \in \hat { Y } _ { i } ^ { \delta } } } K _ { \gamma } ( { \bf y } _ { p } , { \bf y } _ { q } ) + \displaystyle \frac { 1 } { | \mathscr { B } ( x _ { i } , \delta ) | ^ { 2 } } \sum _ { \substack { y _ { p } \in Y _ { i } ^ { \delta } , \hat { y } _ { q } \in \hat { Y } _ { i } ^ { \delta } } } K _ { \gamma } ( \hat { y } _ { p } , \hat { y } _ { q } ) } } \\ { { - \displaystyle \frac { 2 } { | \mathscr { B } ( x _ { i } , \delta ) | ^ { 2 } } \sum _ { \substack { y _ { p } \in Y _ { i } ^ { \delta } , \hat { y } _ { q } \in \hat { Y } _ { i } ^ { \delta } } } K _ { \gamma } ( { y } _ { p } , \hat { y } _ { q } ) , } } \end{array}\tag{D.7}
$$

where $Y _ { i } ^ { \delta }$ and $\hat { Y } _ { i } ^ { \delta }$ are defined in Eq. (D.5), and $K ( \cdot , \cdot )$ denotes the Gaussian kernel:

$$
K _ { \gamma } ( \pmb { y } , \hat { \pmb { y } } ) = \mathrm { e x p } \left( - \frac { \| \pmb { y } - \hat { \pmb { y } } \| ^ { 2 } } { 2 \gamma ^ { 2 } } \right) .
$$

In Examples 1 and $2 , \Gamma = \{ 0 . 5 , 1 . 0 , 2 . 0 , 4 . 0 \}$

• 5. Local squared $W _ { 2 }$ distance

$$
W _ { 2 , \delta } ^ { 2 } ( \pmb { y } _ { \pmb { x } } , \hat { \pmb { y } } _ { \pmb { x } } ) = \frac { 1 } { n _ { b } } \sum _ { i = 1 } ^ { n _ { b } } W _ { 2 } ^ { 2 } ( \mu _ { { \pmb x } _ { i } , \delta } ^ { \mathrm { e } } , \hat { \mu } _ { { \pmb x } _ { i } , \delta } ^ { \mathrm { e } } ) ,\tag{D.8}
$$

where $\mu _ { x _ { i } , \delta } ^ { \mathrm { e } }$ and $\hat { \mu } _ { x _ { i } , \delta } ^ { \mathrm { e } }$ are the empirical distributions associated with the samples ${ \pmb y } ( { \pmb x } _ { j } ; \omega _ { j } )$ in $Y _ { i } ^ { \delta }$ and $\hat { \pmb { y } } ( \pmb { x } _ { j } ; \hat { \omega } _ { j } )$ in $\hat { Y } _ { i } ^ { \delta }$ defined in Eq. (D.5), respectively.

## • 6. Local Sinkhorn divergence

This loss is adopted throughout the numerical experiments unless otherwise specified.

$$
S _ { \varepsilon , \delta } ( \boldsymbol { y } _ { x } , \hat { \boldsymbol { y } } _ { x } ) = \frac { 1 } { n _ { b } } \sum _ { i = 1 } ^ { n _ { b } } S _ { \varepsilon } ( \boldsymbol { \mu } _ { x _ { i } , \delta } ^ { \mathrm { e } } , \hat { \boldsymbol { \mu } } _ { x _ { i } , \delta } ^ { \mathrm { e } } ) ,\tag{D.9}
$$

where $S _ { \varepsilon } ( \mu , \hat { \mu } )$ is the Sinkhorn divergence defined in Eq. (2.6), and $\mu _ { x _ { i } , \delta } ^ { \mathrm { e } }$ and $\hat { \mu } _ { x _ { i } , \delta } ^ { \mathrm { e } }$ are the empirical distributions associated with the samples ${ \bf y } ( { \bf x } _ { j } ; \omega _ { j } )$ in $Y _ { i } ^ { \delta }$ and $\hat { \bf y } ( { \bf x } _ { j } ; \hat { \omega } _ { j } )$ in $\hat { Y } _ { i } ^ { \delta }$ defined in Eq. (D.5), respectively.

## References

[1] Juan A. Acebrón, Adi R. Bulsara, and W.-J. Rappel. Noisy FitzHugh–Nagumo model: From single elements to globally coupled networks. Phys. Rev. E, 69:026202, 2004.

[2] Martin Arjovsky, Soumith Chintala, and Léon Bottou. Wasserstein GAN. 2017.

[3] Mikhail Belkin and Partha Niyogi. Laplacian eigenmaps for dimensionality reduction and data representation. Neural Comput., 15(6):1373– 1396, 2003.

[4] Christopher M. Bishop. Mixture density networks. 1994.

[5] Philippe Clement and Wolfgang Desch. An elementary proof of the triangle inequality for the Wasserstein metric. Proc. Am. Math. Soc., 136(1):333–339, 2008.

[6] Marco Cuturi. Sinkhorn distances: Lightspeed computation of optimal transport. volume 26, 2013.

[7] Laurent Dinh et al. Density estimation using real nvp. In ICLR, 2017.

[8] Jean Feydy, Thibault Séjourné, François-Xavier Vialard, Shun-ichi Amari, Alain Trouvé, and Gabriel Peyré. Interpolating between optimal transport and maximum mean discrepancy using Sinkhorn divergences. In Proceedings of the 22nd International Conference on Artificial Intelligence and Statistics, pages 2681–2690, 2019.

[9] Rémi Flamary, Nicolas Courty, Alexandre Gramfort, Mokhtar Z. Alaya, Aurélie Boisbunon, Stanislas Chambon, Laetitia Chapel, Adrien Corenflos, Kilian Fatras, Nemo Fournier, Léo Gautheron, Nathalie T.H. Gayraud, Hicham Janati, Alain Rakotomamonjy, Ievgen Redko, Antoine Rolet, Antony Schutz, Vivien Seguy, Danica J. Sutherland, Romain Tavenard, Alexander Tong, and Titouan Vayer. POT: Python optimal transport. J. Mach. Learn. Res., 22(78):1–8, 2021.

[10] Charlie Frogner, Chiyuan Zhang, Hossein Mobahi, Mauricio Araya, and Tomaso Poggio. Learning with a Wasserstein loss. volume 28, 2015.

[11] Aude Genevay, Lénaïc Chizat, Francis Bach, Marco Cuturi, and Gabriel Peyré. Sample complexity of Sinkhorn divergences. In Proceedings ofthe 22nd International Conference on Artificial Intelligence and Statistics, pages 1574–1583, 2019.

[12] Aude Genevay, Gabriel Peyré, and Marco Cuturi. Learning generative models with sinkhorn divergences. 2018.

[13] Roger G. Ghanem and Pol D. Spanos. Stochastic Finite Elements: A Spectral Approach. Springer, 1991.

[14] Ian Goodfellow et al. Generative adversarial nets. volume 27, 2014.

[15] Arthur Gretton, Karsten M. Borgwardt, Malte Rasch, Bernhard Schölkopf, and Alexander J. Smola. A kernel two-sample test. J. Mach. Learn. Res., 13:723–773, 2012.

[16] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising difusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.

[17] Diederik P. Kingma and Max Welling. Auto-encoding variational Bayes. 2014.

[18] Nikola Kovachki et al. Neural operator: Learning maps between function spaces. J. Mach. Learn. Res., 24:1–97, 2023.

[19] George Papamakarios, Eric Nalisnick, Danilo Jimenez Rezende, Shakir Mohamed, and Balaji Lakshminarayanan. Normalizing flows for probabilistic modeling and inference. J. Mach. Learn. Res., 22:1–64, 2021.

[20] Gabriel Peyré and Marco Cuturi. Computational Optimal Transport. Now Publishers, 2019.

[21] Maziar Raissi, Paris Perdikaris, and George E. Karniadakis. Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial diferential equations. J. Comput. Phys., 378:686–707, 2019.

[22] Danilo Rezende and Shakir Mohamed. Variational inference with normalizing flows. In ICML, 2015.

[23] Bernhard Schmitzer. Stabilized sparse scaling algorithms for entropy regularized transport problems. SIAM J. Sci. Comput., 41(3):A1443– A1481, 2019.

[24] Kihyuk Sohn et al. Learning structured output representation using deep conditional generative models. In NeurIPS, 2015.

[25] Timothy J. Sullivan. Introduction to Uncertainty Quantification. Springer, 2015.

[26] Gábor J. Székely and Maria L. Rizzo. Energy statistics: A class of statistics based on distances. J. Stat. Plan. Inference, 143(8):1249–1272, 2013.

[27] Joshua B. Tenenbaum, Vin de Silva, and John C. Langford. A global geometric framework for nonlinear dimensionality reduction. Science, 290(5500):2319–2323, 2000.

[28] Henry C. Tuckwell and Roger Rodriguez. Analytical and simulation results for stochastic FitzHugh–Nagumo neurons and neural networks. J. Comput. Neurosci., 5(1):91–113, 1998.

[29] Cédric Villani. Optimal Transport: Old and New. Springer, 2009.

[30] Mingtao Xia and Qijing Shen. Eficient reconstruction of multidimensional random field models with heterogeneous data using stochastic neural networks. arXiv preprint arXiv:2511.13977, 2025.

[31] Mingtao Xia and Qijing Shen. A local squared Wasserstein-2 method for eficient reconstruction of models with uncertainty. Mach. Learn.: Sci. Technol., 7(3):035001, 2026.

[32] Mingtao Xia, Qijing Shen, Philip Maini, Eamonn Gafney, and Alex Mogilner. A new local time-decoupled squared Wasserstein-2 method for training stochastic neural networks to reconstruct uncertain parameters in dynamical systems. Neural Netw., page 107893, 2025.

[33] Dongbin Xiu. Numerical Methodsfor Stochastic Computations. Princeton University Press, 2010.

![](images/54ed27eb02f80a2177361c68fbc899ef0dd1fe0264544293eaa2a1f0e91a5ac5.jpg)

$$
\begin{array} { r l } & { \mathrm { N o r m a l : } \quad g _ { i , k } = \mathrm { R e L U } \Bigl ( \sum _ { j = 1 } ^ { H } w _ { i - 1 , j , k } g _ { i - 1 , j } + b _ { i , k } \Bigr ) } \\ & { \mathrm { R e s N e t : } \quad g _ { i , k } = \mathrm { R e L U } \Bigl ( \sum _ { j = 1 } ^ { H } w _ { i - 1 , j , k } g _ { i - 1 , j } + b _ { i , k } \Bigr ) + g _ { i - 1 , k } \pmod { i } } \\ & { \qquad w _ { i , j , k } \sim \mathcal N ( a _ { i , j , k } , \sigma _ { i , j , k } ^ { 2 } ) } \end{array}
$$

H: the number of neurons per hidden layer ReLU: the ReLU activation function

Figure 1: Illustration of the SNN used for conditional distribution reconstruction. Each linear layer samples its weights from Gaussian distributions whose means and variances are trainable during forward propagation, enabling stochastic realizations to be generated for the same input. Residual connections (ResNet) could be employed to improve optimization stability. The generated samples are used to construct empirical conditional distributions of the reconstructed $\hat { \mathbf { y } } ( \mathbf { x } ; \hat { \omega } )$ in Eq. (2.2). The stochastic weights in the SNN serve as the random variable ̂� in Eq. (2.2). The ReLU activation function may be replaced with other activation functions.  
![](images/c1f69de85353083139fa079e5b9ea0fbeffd16004eaff5e002197c37c9a13777.jpg)

![](images/2adafa3cdf665bfa7a68429c945aa0b48efc69ecf54d6ca75b377ff283de62e1.jpg)

![](images/51cf36d9913704bdb4c674572cd8c57d44153ae75b294b71794c666e40224d3a.jpg)

Figure 2: Comparison of diferent local distribution matching losses for Example 1 (a) Total training time. (b) Average relative errors of the conditional mean and conditional variance on the testing set. (c) Comparison between the true and predicted conditional distributions obtained using the proposed Local Sinkhorn loss. The solid curves denote the conditional means, while the shaded regions represent one standard deviation. All loss functions are minimized to train the same SNN architecture and identical training hyperparameters.  
![](images/8bc83554beffbf6413778609e93ddd69222045f06d7a6917947ea2ad5b3698ab.jpg)

![](images/ce951ce3a523b551608cbaeccd44055186a358282d4625da5975290d60bec7e7.jpg)

![](images/fb58341896373dc9e373014521ef8b1e4e30d2b5190cdd0a67d1527c771b725e.jpg)

![](images/10aab002b2c599da14e36cd94034267de170014959ae45be093efe7fbdad7c73.jpg)  
Figure 3: (a) An illustration of the training and testing sample selection strategy. (b) Influence of the random-field intensity parameter �, where the Gaussian random field is generated as $g _ { \sigma } ( \pmb { x } , \omega ) = \sigma g _ { 0 } ( \pmb { x } , \omega )$ . (c) Influence of the neighborhood radius � used to construct local empirical conditional distributions. (d) Influence of the Sinkhorn regularization parameter �. For all panels, the left axis reports the average relative reconstruction errors of the conditional mean and variance, while the right axis reports the corresponding training time.

(a) Trajectory of v<sub>1</sub>(t)  
![](images/1d4b344be251cc1a0b6160934c0be94fc7ae40f61d96a8e50eeab1bff774d3cf.jpg)

![](images/92a0eed4c5e354479e1c2f346cff7de8ca5713cce78a1276ba4c009d4faeacbf.jpg)  
(d) Drift function of $\mathrm { d } w _ { 1 } ( t )$  
(b) Trajectory of w<sub>1</sub>(t)

(c) Drift function of dv<sub>1</sub>(t)  
![](images/356a39d07cce62d5294cb7f44973dfc7ebba5a8800216ca067df9539798a9fb0.jpg)  
(e) Di,usion function of $\mathrm { d } v _ { 1 } ( t )$

![](images/abbbdc314a380062c2f51feaf22e0149be5fd6352e821a360eda24ce7a235c2a.jpg)

![](images/a77f1fb7463ec8ba89f7f6fef51d549886360a1ca382552693551e9bce7dd3d7.jpg)  
Figure 4: Comparison between the true and predicted stochastic dynamics for a representative testing sample in the stochastic FHN system. Panels $( \mathsf { a } ) ( \mathsf { b } )$ show the mean and standard deviation of trajectories of $v _ { 1 } ( t )$ and $w _ { 1 } ( t ) . ( \mathsf { c } ) ( \mathsf { d } )$ show the mean and standard deviation of the drift functions of $\mathsf { d } v _ { 1 } ( t )$ and $\mathsf { d } w _ { 1 } ( t )$ in Eq. (3.19). (e) shows the mean and standard deviation of the difusion functions of $\mathsf { d } v _ { 1 } ( t )$ in Eq. (3.19). The shaded regions indicate one standard deviation estimated from multiple stochastic realizations.