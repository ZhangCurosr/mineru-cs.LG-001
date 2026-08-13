# Fine-Tuning Generative Models for Extreme Events via CVaR-Penalized Wasserstein Gradient Flows

Thejani Gamage<sup>∗1</sup>, Hyemin Gu<sup>1</sup>, Zhizhen Zhang<sup>1</sup>, Ziyu Chen<sup>2</sup>, Markos Katsoulakis<sup>1</sup>, and Luc Rey-Bellet<sup>1</sup>

<sup>1</sup>Department of Mathematics and Statistics, University of Massachusetts Amherst, Amherst, MA

<sup>2</sup>School of Data and Information Sciences (SDIS) and Department of Mathematics, University of North Carolina at Chapel Hill, Chapel Hill, NC

## Abstract

We propose CVaR-penalized Generative Particle Algorithm (CVaR-GPA), a robust, tail-agnostic algorithm for fine-tuning generative models to learn heavy-tailed distributions and capture extreme events, requiring no prior knowledge or estimation of the target’s tail characteristics. The method is the Wasserstein gradient flow of the Lipschitz-regularized Kullback-Leibler (KL) divergence penalized by a Conditional Value-at-Risk (CVaR) discrepancy term: the Lipschitz-regularized KL divergence enables robust learning under minimal assumptions on the target distribution, while the CVaR penalty restores the velocity that otherwise vanishes prematurely in the under-sampled tails. The penalized flow admits a bounded but non-Lipschitz velocity field. This departs from the Lipschitz transport maps of standard generators, which preserve the tail behavior of a light-tailed source, and enables transport toward heavier-tailed targets. To define this flow on empirical measures, we derive the first-variation subgradients of CVaR from its Rockafellar-Uryasev representation, valid precisely where the classical density-based formula fails. The particle algorithm

CVaR-GPA fine-tunes the output samples of any pre-trained model, without access to its architecture, and runs on an adaptive time horizon set by a kinetic-energy stopping criterion rather than a preset depth. On synthetic isotropic and anisotropic Student-t target distributions, Neal’s funnel distribution, and the real-world high-dimensional Fama-French 25 portfolio dataset, CVaR-GPA dramatically improves global and tail accuracy on heavy-tailed targets over the pre-trained baseline.

Keywords: Extreme events, Wasserstein gradient flows, heavy-tailed distributions, Conditional Value-at-Risk, Lipschitz-regularized divergences, particle neural algorithms, fine-tuning.

## 1 Introduction

Heavy-tailed distributions arise across several high-stakes domains, including finance and insurance [3, 15], catastrophic event forecasting [17], and medicine [10]. In these settings, extreme events can have severe consequences, making accurate simulation particularly important. Yet learning heavy-tailed distributions from finite samples remains a longstanding challenge. Although modern generative models have achieved remarkable success in mapping simple source distributions to complex, high-dimensional targets, learning heavy-tailed distributions remains challenging for two main reasons. The first is a fundamental mathematical obstruction: transport maps in generative models are typically Lipschitz continuous, and since these models are usually initialized from a light-tailed source distribution (e.g., a Gaussian), a Lipschitz transport map necessarily produces a light-tailed output distribution as well [23]. The second is statistical: the inherent scarcity of observations in the tail region can cause training to terminate before the extreme regions are adequately captured, a phenomenon we term premature saturation (or the premature vanishing velocity for flow-based models in particular).

In this work, we propose a novel fine-tuning methodology that addresses the premature saturation exhibited by existing models, enabling them to accurately capture tail behavior without requiring prior knowledge or estimations of the tail decay rates. Let ${ \mathcal { P } } ( \mathbb { R } ^ { d } )$ be the space of probability measures on $\mathbb { R } ^ { d }$ $P ^ { \mathrm { t a r } } \in \mathcal { P } ( \mathbb { R } ^ { d } )$ be the target distribution, and $P ^ { \mathrm { p r e } } \in \mathcal { P } ( \mathbb { R } ^ { d } )$ denote the output distribution of a pre-trained generative model that is available to sample from. We formulate the fine-tuning of a pre-trained model as the

optimization problem

$$
\operatorname* { i n f } _ { \theta \in \Theta } \mathcal { F } ( T _ { \# } ^ { \theta } P ^ { \mathrm { p r e } } ; P ^ { \mathrm { t a r } } ) ,\tag{1.1}
$$

where $\mathcal { F } ( \cdot ; P ^ { \mathrm { t a r } } ) : \mathcal { P } ( \mathbb { R } ^ { d } ) \to [ 0 , \infty )$ is a loss functional whose unique global minimizer is $P ^ { \mathrm { t a r } }$ $T _ { \# } ^ { \theta } P ^ { \mathrm { p r e } }$ is the push-forward measure of the pre-trained measure by a suitably parameterized transport map $T ^ { \theta } : \mathbb { R } ^ { d }  \mathbb { R } ^ { d }$ , where Θ is the parameter space over which (1.1) is optimized. We design a loss functional $\mathcal { F }$ and a transport map $T ^ { \theta }$ that together fine-tune a pre-trained model to learn a heavy-tailed target distribution $P ^ { \mathrm { t a r } }$ more accurately.

We construct the transport map $T ^ { \theta }$ in (1.1) as the discretization of the Wasserstein gradient flow [24, 30] of ${ \mathcal F }$ . If the variational derivative of $\mathcal { F }$ with respect to the generated distribution $Q { \mathrm { . } }$ denoted by $\displaystyle \frac { \delta \mathcal { F } ( Q ; P ^ { \mathrm { t a r } } ) } { \delta Q }$ , exists, then the resulting Wasserstein gradient flow can be formulated as

$$
\partial _ { t } Q _ { t } - \nabla \cdot \Big ( Q _ { t } \nabla _ { x } \Big ( \frac { \delta \mathcal { F } ( Q ; P ^ { \mathrm { t a r } } ) } { \delta Q } ( x ) \Big ) \Big | _ { Q = Q _ { t } } \Big ) = 0 , \quad t > 0 , \quad Q _ { 0 } = P ^ { \mathrm { p r e } } ,\tag{1.2}
$$

where $Q _ { t }$ denotes the evolving distribution at time t. In practice, generative models approximate probability measures via their empirical distributions over finite samples. Thus, both the chosen functional ${ \mathcal { F } } ( Q ; P ^ { \mathrm { t a r } } )$ and its variational derivative $\displaystyle \frac { \delta \mathcal { F } ( Q ; P ^ { \mathrm { t a r } } ) } { \delta Q }$ must be welldefined when $Q$ and $P ^ { \mathrm { t a r } }$ are replaced by their empirical distributions. Such a functional allows us to initialize the fine-tuning (1.1) from any pre-trained model $P ^ { \mathrm { p r e } }$ , whose output samples we have access to. Our use of the Wasserstein gradient flows to construct the transport map $T ^ { \theta }$ is primarily motivated by their ability to initialize the learning directly from samples of $P ^ { \mathrm { p r e } }$ , without access to the internal architecture of the pre-trained model. Beyond this, Wasserstein gradient flows carry several properties that make it a well-suited fine-tuning framework for robust, tail-agnostic learning of heavy-tailed targets. These properties, closely tied to the choice of the loss functional ${ \mathcal F }$ , are discussed in detail in Section 3.

While the KL divergence and the Wasserstein metrics are among the most widely used loss functionals in generative modeling, neither is well suited on its own as the loss functional $\mathcal { F }$ in (1.2). The KL divergence is not well-defined when $Q$ and $P ^ { \mathrm { t a r } }$ are approximated by their empirical measures, while the Wasserstein metrics are not diferentiable in $Q .$ On the other hand, the Lipschitz-regularized KL divergence [14] circumvents both these limitations. As shown in [9], the Lipschitz-regularized KL divergence and its first variational derivative are well defined both at the population level and the finite-sample level. At the population level, the Lipschitz- regularized KL divergence and its first variational derivative are well-defined whenever Q has a finite first moment, with no assumption whatsoever on $P ^ { \mathrm { t a r } }$ Thus, the Lipschitz-regularized KL divergence is a suitable choice for tail-agnostic learning of heavy-tailed targets. However, Wasserstein gradient flows of Lipschitz-regularized divergences, which are referred to as Lipschitz-regularized Wasserstein gradient flows in what follows, exhibit the premature vanishing velocity issue due to the scarcity of data in the tail region (see Figure 1 for illustrations).

We propose the following loss functional that addresses the premature vanishing velocity issue of the Lipschitz-regularized Wasserstein gradient flow by penalizing the Lipschitzregularized KL divergence with a weighted, squared Conditional Value-at-Risk (CVaR) discrepancy term

$$
\begin{array} { r } { \mathcal { F } ^ { \mathrm { C V a R } } ( Q ; P ^ { \mathrm { t a r } } ) = D _ { \mathrm { K L } } ^ { L } ( Q \| P ^ { \mathrm { t a r } } ) + \lambda \Big ( \mathrm { C V a R } _ { \alpha } ^ { P ^ { \mathrm { t a r } } , g } - \mathrm { C V a R } _ { \alpha } ^ { Q , g } \Big ) ^ { 2 } , } \end{array}\tag{1.3}
$$

where $D _ { \mathrm { K L } } ^ { L }$ denotes the Lipschitz-regularized KL divergence in (2.5), $\mathrm { C V a R } _ { \alpha } ^ { Q , g }$ denotes the CVaR at the $\alpha ^ { \mathrm { t h } }$ quantile of a non-negative function g under the distribution Q (similarly for $\mathrm { C V a R } _ { \alpha } ^ { P ^ { \mathrm { t a r } } , g } )$ , and $\lambda > 0$ is a hyperparameter that controls the relative weight of the squared CVaR diference and the Lipschitz-regularized KL divergence. The loss functional (1.3) is referred to as the CVaR-penalized loss functional. But, $\mathrm { C V a R } _ { \alpha } ^ { P ^ { \mathrm { t a r } } , g }$ , and by extension ${ \mathcal { F } } ^ { \operatorname { C V a R } } ( Q ; P ^ { \mathrm { t a r } } )$ , are not diferentiable in Q in general, in particular for empirical measures. Thus, in our work, we consider the first variational subgradients of $\mathrm { C V a R } _ { \alpha } ^ { Q , g }$ and ${ \mathcal { F } } ^ { \mathrm { C V a R } } ( Q ; P ^ { \mathrm { t a r } } )$ defined using Clarke’s generalized gradients [11] (see Appendix A for the formal definition). We note that, while first variational subgradients of $\mathrm { C V a R } _ { \alpha } ^ { Q , g }$ can also be referred to as supergradients due to the concavity of CVaR, in our work we adopt the general terminology first variational subgradient, regardless of the functional being convex, concave, or neither.

The Wasserstein gradient flow of the functional (1.3), referred to as the CVaR-penalized Wasserstein gradient flow, can be viewed as a transport-based variational PDE. The CVaR penalization introduces an additional velocity component to the Lipschitz-regularized Wasserstein gradient flow in the tail region (see Figure 1). Its velocity field is bounded, enabling stable learning, and not Lipschitz continuous. Consequently, the induced transport map is not Lipschitz continuous as well, making the CVaR-penalized Wasserstein gradient flow a suitable framework for heavy-tailed targets. The associated particle algorithm, CVaR-GPA, extends the Lipschitz-regularized Generative Particle Algorithm (Lip-KL-GPA) of [18]. Figure 2 previews the results obtained by fine-tuning the Lip-KL- GPA pre-trained model on the Fama-French 25 monthly portfolio dataset [16]; a stringent test case demonstrating CVaR-GPA performs well on high-dimensional anisotropic targets. We note that CVaR-GPA operates in a tail-agnostic setting and does not require a priori knowledge or estimation of the target distribution’s tail decay rate, nor architectural modifications tailored to it.

![](images/757ba7b8e3bb2e906c983a7537c86fcfbc7b345607715f81b514dd2b098074d9.jpg)

![](images/f076051cf7c0f3017b9db96369e607fc12118f9dc4114ed17beda52ff4ee5f35.jpg)

![](images/ce86e6b23002dac96207a498b0564a01119a1a387c1166bc65b66a101336a751.jpg)  
Figure 1: CVaR-penalization restores Lip-KL-GPA’s premature vanishing velocity. Left: Lip-KL-GPA particles $( L = 1 . 0 , N = 5 , 0 0 0 , 2 0 , 0 0 0$ iterations) on an isotropic 2-d Student-t target $( \nu = 1 . 0 $ , no finite moments); converges but misses the tails. Middle: the velocity $\lVert v \rVert$ vanishes almost everywhere; nonzero corners are NN extrapolation artifacts outside particle support. Right: CVaR-penalized fine-tuning (illustrative hyperparameters: $\alpha =$ 0.9, $\begin{array} { r } { \lambda = \frac { 1 } { 3 2 } } \end{array}$ $L = 0 . 1 2 5 )$ restores velocity, pushing particles outward beyond the radius $\overline { { \mathrm { V a R } } } _ { \alpha }$ (given by (2.13)) towards the tails.

The rest of the paper is organized as follows. Section 2 analyzes the proposed loss functional $\mathcal { F } ^ { \mathrm { C V a R } }$ , defined in (1.3), and its divergence property and computational tractability that make it a well-posed objective for learning heavy-tailed targets. Section 3 derives an explicit formula for the first variational subgradients of $\mathcal { F } ^ { \mathrm { C V a R } }$ , and defines the CVaRpenalized Wasserstein gradient flow. Furthermore, Section 3 rigorously demonstrates how the induced velocity field of this flow remedies the premature vanishing velocity issue. We present the particle algorithm CVaR-GPA in Section 4, and evaluate its performance on a 2-dimensional isotropic Student-t distribution, a 5-dimensional anisotropic Student-t distribution, Neal’s funnel distribution, and the Fama-French 25 dataset in Section 5. Finally, we conclude this paper and discuss future directions in Section 6.

![](images/6b4240db3e715d1fb57b9feda01bfa4d73abf8c9a3edaf327f03fc9deb18ee2a.jpg)

![](images/a5ad784d4314ec5c869faa512efd9446ff8dd904bdc6f2f1b18aa12b3fc1ba75.jpg)  
Figure 2: Comparison of Lip-KL-GPA vs. CVaR-GPA on the Fama-French 25 monthly portfolios dataset via the global $L ^ { 1 }$ error (5.45), denoted by $\mathcal { E } _ { L ^ { 1 } , i }$ for each marginal $i ,$ and the tail error (5.46), denoted by $\mathcal { E } _ { \mathrm { t a i l } , i }$ for each marginal $i ;$ both errors are displayed on a log scale. CVaR-GPA fine-tunes Lip-KL-GPA and decreases the global $L ^ { 1 }$ error and the tail error for each marginal distribution.

## 1.1 Related work

Generative models tailored for learning heavy-tailed targets There are many generative model designs tailored for heavy-tailed distributions [4, 5, 6, 19, 20, 22, 27, 31]. Despite reasonable empirical performance, these generative models still exhibit several limitations. On one hand, certain architectures (including mirror flow matching [19], Pareto GAN [22], Score-based Heavy-tailed Difusion [27], t-EDM [31], and t-Flow [31]) require prior knowledge or accurate estimates of the tail decay rate of the target distribution. Accurate estimation of the tail decay rate is itself a computationally challenging problem. On the other hand, many models, including EV-GAN [4], ExceedGAN [5], Exgan [6], and Tail Transform Flow (TTF) [20], operate over a fixed time horizon. As a result, the efective depth of the neural architecture is treated as a static hyperparameter determined a priori; this potentially limits the model’s capacity to transport light-tailed sources to heavy-tailed targets. Lip-KL-GPA, a particle algorithm introduced in [18] based on Lipschitz-regularized Wasserstein gradient flows, is both tail-agnostic and operates in an adaptive time horizon. As shown in [9], Lip-KL-GPA and other generative models opti mizing Lipschitz-regularized divergences outperform existing generative models on learning heavy-tailed targets, including f-GANs, optimal transport (OT) flow, continuous normalizing flows (CNFs), and score-based generative models (SGMs). However, the generative models optimizing Lipschitz-regularized divergences empirically exhibit premature saturation when learning heavy-tailed targets, the primary drawback that we address by introducing the CVaR penalization.

CVaR for tail-sensitive modeling CVaR is one instance of a broader family of spectral risk measures [2] that assign greater weight to the tail region. CVaR, also known as the Expected Shortfall, is a canonical choice in tail- sensitive modeling, with an extensive literature [1, 12, 26], owing largely to its tractable variational representation via the Rockafellar–Uryasev formula [32]. Incorporating CVaR into the generative objective to improve the accuracy of learning heavy-tailed targets has also been explored by Tail-GAN [12], an adversarial generator for multi-asset financial returns that augments the GAN discriminator loss with Value-at-Risk (VaR) and CVaR, so that the generator learns to reproduce correct tail-risk statistics for benchmark financial portfolios. Unlike our method, however, its generator is a fixed-depth Lipschitz continuous transport map, which may structurally limit its ability to learn a heavy-tailed target distribution. A separate line of work incorporates CVaR into fine-tuning objectives for tail-sensitive reward maximization [8, 13, 33]. This line of research pursues a fundamentally diferent goal from ours: it reweighs the generated distribution to favor extreme rewards, whereas we aim to learn heavy-tailed distributions and capture extreme events. The framework of [33] is closest to ours, where CVaR, its Rockafellar-Uryasev representation, and its first variational derivative are employed for tail-sensitive reward maximization. However, the variational derivative of CVaR utilized in [13, 33] is computed under density assumptions that do not hold for general distributions such as empirical measures supported on finite samples. This necessitates our derivation of the first variational subgradients of $\mathrm { C V a R } _ { \alpha } ^ { Q , g }$

## 2 CVaR-penalized loss functional and its properties

For eficient and stable fine-tuning tailored to learning heavy-tailed distributions, we propose the following loss functional $\mathcal { F } ^ { \mathrm { C V a R } }$

$$
\begin{array} { r } { \mathcal { F } ^ { \mathrm { C V a R } } ( Q ; P ^ { \mathrm { t a r } } ) = D _ { \mathrm { K L } } ^ { L } ( Q \Vert P ^ { \mathrm { t a r } } ) + \lambda \left( \Delta C ( Q ; P ^ { \mathrm { t a r } } ) \right) ^ { 2 } , } \end{array}\tag{2.4}
$$

where $\begin{array} { r } { \Delta C ( Q ; P ^ { \mathrm { t a r } } ) : = \mathrm { C V a R } _ { \alpha } ^ { P ^ { \mathrm { t a r } } , g } - \mathrm { C V a R } _ { \alpha } ^ { Q , g } } \end{array}$ denotes the CVaR discrepancy of $g ( X )$ for $X \sim Q$ and $X \sim P ^ { \mathrm { t a r } }$ , for a given risk function $g : \mathbb { R } ^ { d }  [ 0 , \infty )$ . The risk measure $\mathrm { C V a R } _ { \alpha } ^ { Q , g }$ is defined under the assumption $\mathbb { E } _ { Q } [ g ] < \infty$ , and the Lipschitz-regularized KL divergence is defined for Q with finite first moment. Throughout this paper, we have the following standing assumption:

Assumption 2.1. For the loss functional $\mathcal { F } ^ { \mathrm { C V a R } }$ to be well-defined, we assume the following.

1. Let $\mathcal { Q } ^ { g } \ : = \ \{ Q \ \in \ \mathcal { P } ( \mathbb { R } ^ { d } ) ; \mathbb { E } _ { Q } [ g ] \ < \ \infty \}$ we assume that $Q , P ^ { \mathrm { t a r } } \in \mathcal { Q } ^ { g }$ , so that $\Delta C ( Q ; P ^ { \mathrm { t a r } } )$ is well-defined.

2. We assume that Q has finite first moment so that the Lipschitz-regularized KL divergence (and its variational derivative) is well-defined.

We note that if one were to choose the radial risk function $g ( x ) = \| x \|$ , which is the risk function we use to define the CVaR-penalized Wasserstein gradient flow, upon which our numerical algorithm is built on $Q \in \mathcal { Q } ^ { g }$ implies that Q has finite first moment, and in that case, Assumption 2.1(2) is redundant. We also note that Assumption 2.1 is a densitylevel condition ensuring that $\mathcal { F } ^ { \mathrm { C V a R } }$ and its variational subgradients are well-defined in Section $3 ;$ it imposes no restriction on the algorithm discussed in Section 4. The empirical measures $\widehat { Q }$ and Pdtar on which CVaR-GPA operates (Section 4) are supported on finitely $\widehat { P ^ { \mathrm { t a r } } }$ many samples, so both parts of Assumption 2.1 hold automatically, for any target $P ^ { \mathrm { t a r } }$ including heavy-tailed targets without finite moments, such as the Cauchy $( \nu = 1 )$ target distribution in Section 5.

With Assumption 2.1, the fine-tuning is formulated as the optimization problem

$$
\operatorname* { i n f } _ { Q \in { \mathcal { Q } } ^ { g } } { \mathcal { F } } ^ { \operatorname { C V a R } } ( Q ; P ^ { \operatorname { t a r } } ) .
$$

In this section, we establish some structural properties of $\mathcal { F } ^ { \mathrm { C V a R } }$ that make it a well-posed and tractable objective for optimization. The loss functional ${ \mathcal { F } } ^ { \operatorname { C V a R } } ( Q ; P ^ { \mathrm { t a r } } )$ should satisfy the divergence property, so that the target distribution $P ^ { \mathrm { t a r } }$ is its unique global minimizer; a necessary property which makes minimizing the loss functional ${ \mathcal { F } } ^ { \operatorname { C V a R } } ( Q ; P ^ { \mathrm { t a r } } )$ equivalent to learning the target distribution $P ^ { \mathrm { t a r } }$ . Furthermore, we need both the functional $\mathcal { F } ^ { \mathrm { C V a R } }$ and its variational subgradients to be estimable from finite samples. These properties are inherited from the two components comprising $\mathcal { F } ^ { \mathrm { C V a R } }$ : the Lipschitz- regularized KL divergence and the squared CVaR discrepancy. We first discuss the properties of each component that make $\mathcal { F } ^ { \mathrm { C V a R } }$ a suitable objective, and then present a detailed analysis.

## 2.1 Mathematical preliminaries on components of $\mathcal { F } ^ { \mathrm { C V a R } }$

Lipschitz-regularized KL divergence The Lipschitz-regularized KL divergence was first introduced in [14] and subsequently developed in [7] as a computational tool for generative modeling and other applications involving heavy-tailed and singular data. Lipschitzregularized KL divergence is defined via the infimal convolution

$$
D _ { \mathrm { K L } } ^ { L } ( Q \| P ^ { \mathrm { t a r } } ) = \operatorname* { i n f } _ { \gamma \in \mathcal { P } ( \mathbb { R } ^ { d } ) } \left\{ D _ { K L } ( \gamma \| P ^ { \mathrm { t a r } } ) + L \cdot \mathcal { W } _ { 1 } ( Q , \gamma ) \right\} ,\tag{2.5}
$$

where $D _ { K L }$ and $\mathcal { W } _ { 1 }$ denote the KL divergence and the Wasserstein-1 metric, respectively. By definition, the minimizer $\gamma ^ { * }$ of (2.5) can be viewed as a reweighting of the target measure $P ^ { \mathrm { t a r } }$ through the KL divergence, while the Wasserstein-1 metric measures the cost of transporting mass from the intermediate measure $\gamma ^ { * }$ to $Q$ . The transportation of mass removes the need for an absolute continuity assumption between $Q$ and $P ^ { \mathrm { t a r } }$ and the redistribution makes the Lipschitz-regularized KL divergence suitable for handling heavytailed distributions by redistributing mass from the bulk region to the tail region (see [7, Section 3] for a detailed discussion of mass redistribution/transport interpretation). Furthermore, we recall from [14] that the Lipschitz-regularized KL divergence defined by (2.5) has a variational representation given by

$$
D _ { \mathrm { K L } } ^ { L } ( Q \| P ^ { \mathrm { t a r } } ) = \operatorname* { s u p } _ { \phi \in \Gamma _ { L } } \{ \mathbb { E } _ { Q } [ \phi ] - \log ( \mathbb { E } _ { P ^ { \mathrm { t a r } } } [ e ^ { \phi } ] ) \} ,\tag{2.6}
$$

where $\Gamma _ { L }$ is the space of L-Lipschitz functions.

As proven in [9, Theorem 3], under the Assumption 2.1(2), the optimizer of (2.6), denoted by $\phi ^ { * }$ , exists and is unique on supp $( Q ) \cup \operatorname { s u p p } ( P ^ { \operatorname { t a r } } )$ , the support of $Q$ and $P ^ { \mathrm { t a r } }$ It holds that

$$
\frac { \delta D _ { \mathrm { K L } } ^ { L } ( Q \| P ^ { \mathrm { t a r } } ) } { \delta Q } ( x ) = \phi ^ { * } ( x ) ,\tag{2.7}
$$

for $x \in { \mathrm { s u p p } } ( Q ) \cup { \mathrm { s u p p } } ( P ^ { \mathrm { t a r } } )$

If Q has a finite first moment (i.e. Q satisfies Assumption 2.1(2)), then both $D _ { \mathrm { K L } } ^ { L } ( Q \Vert P ^ { \mathrm { t a r } } )$ and its first variational derivative exist finitely, without any further assumptions on $P ^ { \mathrm { t a r } }$ ， such as absolute continuity between $Q$ and $P ^ { \mathrm { t a r } }$ , or any finite moment assumptions on $P ^ { \mathrm { t a r } }$ (see [9, Theorem 2, Theorem 3]). This property makes the Lipschitz-regularized KL divergence a proper choice over other well-known objectives such as the KL divergence or the Wasserstein metrics for stably learning a broad class of target distributions, including heavy-tailed distributions. However, due to the scarcity of samples in the tail region, algorithms based on Lipschitz-regularized KL divergence exhibit premature saturation. This motivates penalizing the Lipschitz-regularized KL divergence with a tail-sensitive term that mitigates the impact of the sample scarcity in the tail region and remedies the premature vanishing velocity issue by introducing an additional velocity component in the tail region, as we establish in Theorem 3.8.

Conditional Value-at-Risk (CVaR) Using a spectral risk measure that assigns a greater weight to the tail region can be used to design a fine- tuning loss functional that overcomes the limitations due to sample scarcity. While many such measures are admissible for this purpose [2], we adopt the CVaR, which is both tail-sensitive and computationally tractable (due to its variational representation via the Rockafellar–Uryasev formula [32]).

Let $g : \mathbb { R } ^ { d }  [ 0 , \infty )$ be a risk function, and $Q \in \mathcal { Q } ^ { g }$ . Denote the cumulative distribution function of $g$ under $Q$ by $\Psi ^ { Q , g }$ . Given a parameter $0 < \alpha < 1$ , the Value-at-Risk (VaR) of ${ \mathit { g } } ,$ at level α, i.e., the $\alpha ^ { \mathrm { t h } }$ quantile, under Q is given by

$$
\operatorname { V a R } _ { \alpha } ^ { Q , g } : = \operatorname* { i n f } _ { c \in \mathbb { R } } \{ c : \Psi ^ { Q , g } ( c ) \geq \alpha \} .\tag{2.8}
$$

Since $\Psi ^ { Q , g } ( c )$ is a non-decreasing and right-continuous function of $c ,$ the infimum can be attained. We denote the Conditional Value-at-Risk (CVaR) of the risk function g under $Q$ at level $\alpha \in ( 0 , 1 )$ by $\mathrm { C V a R } _ { \alpha } ^ { Q , g }$ . For a random variable $g ( X )$ under a general distribution, including an empirical distribution supported on finite samples, we have

$$
\mathrm { C V a R } _ { \alpha } ^ { Q , g } : = \beta \mathrm { V a R } _ { \alpha } ^ { Q , g } + ( 1 - \beta ) \overline { { \mathrm { C V a R } } } _ { \alpha } ^ { Q , g } ,\tag{2.9}
$$

where $\begin{array} { r } { \beta = \frac { \Psi ^ { Q , g } ( \mathrm { V a R } _ { \alpha } ^ { Q , g } ) - \alpha } { 1 - \alpha } \ \mathrm { a n d } \ \overline { { \mathrm { C V a R } } } _ { \alpha } ^ { Q , g } : = \mathbb { E } _ { Q } [ g ( X ) \ | \ g ( X ) > \mathrm { V a R } _ { \alpha } ^ { Q , g } ] } \end{array}$ . If the random variable $g ( X )$ is continuous, then we have

$$
\mathrm { C V a R } _ { \alpha } ^ { Q , g } = \mathbb { E } _ { Q } [ g ( X ) \mid g ( X ) \geq \mathrm { V a R } _ { \alpha } ^ { Q , g } ] .\tag{2.10}
$$

Thus, CVaR, by definition, focuses on the tail region of the distribution.

Rockafellar-Uryasev formulation CVaR has the following equivalent formulation for a random variable $g ( X )$ under a general distribution $Q _ { i }$ known as the Rockafellar-Uryasev formulation [32]

$$
\mathrm { C V a R } _ { \alpha } ^ { Q , g } = \operatorname* { i n f } _ { y \in \mathbb { R } } F _ { \alpha } ^ { Q , g } ( y ) ,\tag{2.11}
$$

where $F _ { \alpha } ^ { Q , g } ( y )$ is defined as

$$
F _ { \alpha } ^ { Q , g } ( y ) = y + \frac { 1 } { 1 - \alpha } \mathbb { E } _ { Q } [ ( g ( \cdot ) - y ) ^ { + } ] .\tag{2.12}
$$

Consider the quantity

$$
\overline { { \operatorname { V a R } } } _ { \alpha } ^ { Q , g } : = \operatorname* { i n f } \{ c : \Psi ^ { Q , g } ( c ) > \alpha \} ,\tag{2.13}
$$

then the set of minimizers of (2.11), denoted by $T ( Q )$ , is $[ \mathrm { V a R } _ { \alpha } ^ { Q , g } , \overline { { \mathrm { V a R } } } _ { \alpha } ^ { Q , g } ]$ (see [32, Theorem 10] for a proof).

## 2.2 Properties of the loss functional ${ \mathcal { F } } ^ { \mathrm { C V a R } }$

The loss functional $\mathcal { F } ^ { \mathrm { C V a R } }$ has several properties that make it suitable for learning heavytailed distributions. The following proposition establishes that the functional ${ \mathcal { F } } ^ { \operatorname { C V a R } } ( Q ; P ^ { \mathrm { t a r } } )$ is a divergence. Hence, the target distribution $P ^ { \mathrm { t a r } }$ is its unique global minimizer.

Proposition 2.2 (Divergence property). Let Assumption 2.1 hold. Then, the loss functional $\mathcal { F } ^ { \mathrm { C V a R } }$ defined in (2.4) satisfies ${ \mathcal { F } } ^ { \mathrm { C V a R } } ( Q ; P ^ { \mathrm { t a r } } ) \geq 0$ for all $Q \in \mathcal { Q } ^ { g }$ and we have $\mathcal { F } ^ { \mathrm { C V a R } } ( Q ; P ^ { \mathrm { t a r } } ) = 0$ if and only if $Q = P ^ { \mathrm { t a r } }$ . That is, $P ^ { \mathrm { t a r } }$ is the unique global minimizer of the optimization problem

$$
\operatorname* { i n f } _ { Q \in { \mathcal { Q } } ^ { g } } { \mathcal { F } } ^ { \operatorname { C V a R } } ( Q ; P ^ { \operatorname { t a r } } ) .
$$

Proof. The Lipschitz-regularized KL divergence $D _ { K L } ^ { L } ( Q \| P ^ { \mathrm { t a r } } )$ satisfies the divergence property, i.e., $D _ { K L } ^ { L } ( Q \Vert P ^ { \mathrm { t a r } } ) \geq 0$ for all $Q \in \mathcal { P } ( \mathbb { R } ^ { d } )$ and $D _ { K L } ^ { L } ( Q \| P ^ { \mathrm { t a r } } ) = 0$ if and only if $Q = P ^ { \mathrm { t a r } }$ (see [7, Theorem 8] for the proof). This, combined with the fact that $( \Delta C ( Q ; P ^ { \mathrm { t a r } } ) ) ^ { 2 } \ge 0$ for all $Q \in \mathcal { Q } ^ { g }$ , proves that $\mathcal { F } ^ { \mathrm { C V a R } } ( Q ; P ^ { \mathrm { t a r } } ) \ge 0$ for all $Q \in \mathcal { Q } ^ { g }$ and $\mathcal { F } ^ { \mathrm { C V a R } } ( Q ; P ^ { \mathrm { t a r } } ) = 0$ if and only if $Q = P ^ { \mathrm { t a r } }$ . That is, $P ^ { \mathrm { t a r } }$ is the unique global minimizer of the optimization problem $\operatorname* { i n f } _ { Q \in { \mathcal { Q } } ^ { g } } { \mathcal { F } } ^ { \operatorname { C V a R } } ( Q ; P ^ { \operatorname { t a r } } )$ .

We note that the term $( \Delta C ( Q ; P ^ { \mathrm { t a r } } ) ) ^ { 2 }$ alone does not satisfy the divergence property since infinitely many distributions Q share the same CVaR as the target distribution $P ^ { \mathrm { t a r } }$ It is the Lipschitz-regularized KL divergence, $D _ { \mathrm { K L } } ^ { L } ( Q \Vert P ^ { \mathrm { t a r } } )$ , that endows the loss functional (2.4) with the divergence property, and thereby ensures that the minimizer of $\mathcal { F } ^ { \mathrm { C V a R } }$ uniquely identifies the target distribution. Moreover, since $D _ { \mathrm { K L } } ^ { L } ( Q \Vert P ^ { \mathrm { t a r } } )$ directly compares $Q$ and $P ^ { \mathrm { t a r } }$ as a divergence, $D _ { \mathrm { K L } } ^ { L } ( Q \Vert P ^ { \mathrm { t a r } } )$ is capable of capturing structural properties of the target distribution $P ^ { \mathrm { t a r } }$ that may not be reflected by the scalar statistic $\mathrm { C V a R } _ { \alpha } ^ { Q , g }$ . Thus, the two components of the loss functional play distinct and complementary roles:

• The Lipschitz-regularized KL divergence enforces the divergence property and encourages the capture of structural properties of the target distribution $P ^ { \mathrm { t a r } }$

• The CVaR component remedies the issue of premature vanishing velocity by introducing an additional velocity component in the tail region.

Hence, both the Lipschitz-regularized KL divergence and the CVaR component are necessary in the loss functional $\mathcal { F } ^ { \mathrm { C V a R } }$ for accurately learning heavy-tailed targets.

## 3 CVaR-penalized Wasserstein gradient flows

For a given loss functional ${ \mathcal { F } } ,$ if the variational derivative of $\mathcal { F }$ with respect to the generated distribution $Q ,$ denoted by $\displaystyle \frac { \delta \mathcal { F } ( Q ; P ^ { \mathrm { t a r } } ) } { \delta Q }$ , exists, its Wasserstein gradient flow is formulated as

$$
\partial _ { t } Q _ { t } - \nabla \cdot \Big ( Q _ { t } \nabla _ { x } \Big ( \frac { \delta \mathcal { F } ( Q ; P ^ { \mathrm { t a r } } ) } { \delta Q } ( x ) \Big ) \Big | _ { Q = Q _ { t } } \Big ) = 0 , \quad t > 0 , \quad Q _ { 0 } = Q ^ { \mathrm { r e f } }\tag{3.14}
$$

where $Q _ { t }$ denotes the evolving distribution at time t and $Q ^ { \mathrm { r e f } }$ is the initial distribution. In line with our fine-tuning framework, throughout this paper, we take $Q ^ { \mathrm { r e f } } = P ^ { \mathrm { p r e } }$

For $t > 0$ , denote the velocity field of (3.14) by $\begin{array} { r } { v _ { Q _ { t } } : = - \nabla _ { x } \left( \frac { \delta \mathcal { F } \left( Q ; P ^ { \mathrm { t a r } } \right) } { \delta Q } \right) } \end{array}$ . The flow (3.14) terminates if its velocity field dies out, i.e., when $v _ { Q _ { T ^ { * } } } = 0$ for some $T ^ { * } > 0$ . We note that, similar to the result in [18, Theorem 2.5], if the trajectory of distributions $\{ Q _ { t } \} _ { t \ge 0 }$ in (3.14) is suficiently smooth, we formally have the energy dissipation identity

$$
\frac { \mathrm { d } } { d t } \mathcal { F } ( Q ; P ^ { \mathrm { t a r } } ) = - \mathbb { E } _ { Q _ { t } } \big [ \| v _ { Q _ { t } } \| ^ { 2 } \big ] = - 2 { \mathcal { K } } ( t ) ,\tag{3.15}
$$

where $\begin{array} { r } { { \cal K } ( t ) : = \frac { 1 } { 2 } \int \| v _ { Q _ { t } } \| ^ { 2 } \mathrm { d } Q _ { t } } \end{array}$ denotes the kinetic energy. At the terminal time $T ^ { * }$ with $v _ { Q _ { T ^ { * } } } = 0$ , we have $\begin{array} { r } { \kappa ( T ^ { * } ) = 0 } \end{array}$ , and by the energy dissipation identity (3.15) $\begin{array} { r l } { \frac { \mathrm { d } } { \mathrm { d } t } \mathcal { F } ( Q _ { T ^ { * } } ; P ^ { \mathrm { t a r } } ) = } \end{array}$ 0. That is, $Q _ { T }$ ∗ is a critical point of $\mathcal { F } ^ { \mathrm { C V a R } } ( \cdot ; P ^ { \mathrm { t a r } } )$ , motivating the use of Wasserstein gradient flows for minimizing the loss functional ${ \mathcal F } .$ . Wasserstein gradient flows possess several properties, closely tied to the choice of the loss functional ${ \mathcal { F } } _ { : }$ , that make them a well-suited fine- tuning framework for learning heavy-tailed targets:

• Wasserstein gradient flows can be initialized from any reference distribution, unlike frameworks such as CNFs or SGMs that require a specific source distribution class $( \mathrm { e . g . }$ , Gaussian); this lets us fine-tune pre- trained models directly, utilizing the samples of $P ^ { \mathrm { p r e } }$ alone, without access to its internal architecture.

• The velocity field of the Wasserstein gradient flow of a loss functional $\mathcal { F }$ is determined by the variational derivative (or subgradients) of ${ \mathcal F }$ . Thus, the loss functional $\mathcal { F }$ can be designed to embed crucial properties into the velocity function, and thereby into the transport map. In our work, we use this property to design transport maps that are not necessarily Lipschitz continuous, which is crucial for learning heavy-tailed targets.

• The learning time horizon of a Wasserstein gradient flow is not pre-determined. Rather, the time horizon of a Wasserstein gradient flow depends on both the choice of loss functional $\mathcal { F }$ and the target distribution $P ^ { \mathrm { t a r } }$ , since the velocity field of the Wasserstein gradient flow is determined by the variational derivative of the loss functional ${ \mathcal { F } } ( Q ; P ^ { \mathrm { t a r } } )$ .

One can leverage these favorable properties of Wasserstein gradient flows to design a finetuning framework for learning heavy-tailed targets with a suitable loss functional ${ \mathcal F }$ . As discussed in Section 2, we adopt the loss functional $\mathcal { F } ^ { \mathrm { C V a R } }$ in (2.4), resulting in the CVaRpenalized Wasserstein gradient flow. CVaR-penalized Wasserstein gradient flow mitigates the drawback of Lipschitz-regularized Wasserstein gradient flows by introducing an additional velocity component in the tail region that revives the otherwise vanishing velocity field (see Section 3.2 for the explicit derivation of the corresponding velocity field).

## 3.1 Variational subgradients

The velocity field of the Wasserstein gradient flow of a loss functional $\mathcal { F }$ is defined using its variational derivative, when it exists. However, as we prove in Theorem $3 . 7 ( 3 )$ , the variational derivative of ${ \mathcal { F } } ^ { \operatorname { C V a R } } ( Q ; P ^ { \mathrm { t a r } } )$ does not exist, in general. This is an extension of the fact that the variational derivative of $\mathrm { C V a R } _ { \alpha } ^ { Q , g }$ exists if and only if $\mathrm { V a R } _ { \alpha } ^ { Q , g } = \overline { { \mathrm { V a R } } } _ { \alpha } ^ { Q , g }$ (see Theorem 3.5(4) for details). This condition fails for a given α whenever $\Psi ^ { Q , g } ( c ) = \alpha$ on a nonempty interval of $c ;$ for empirical measures, whose CDF $\Psi ^ { Q , g }$ is a step function, such α always exist. Therefore, we instead consider the variational subgradients of both $\mathrm { C V a R } _ { \alpha } ^ { Q , g }$ and therefore ${ \mathcal { F } } ^ { \mathrm { C V a R } } ( Q ; P ^ { \mathrm { t a r } } )$ , defined via Clarke’s generalized gradients [11]. To derive the variational subgradients of $\mathrm { C V a R } _ { \alpha } ^ { Q , g }$ and ${ \mathcal { F } } ^ { \operatorname { C V a R } } ( Q ; P ^ { \mathrm { t a r } } )$ , we consider perturbations of the probability measure $Q$

Let $\rho$ be a signed measure and denote the probability measure $Q$ perturbed by $\epsilon \rho$ as $Q ^ { \epsilon } : = Q + \epsilon \rho \in \mathcal { P } ( \mathbb { R } ^ { d } )$ , for $\epsilon \neq 0$ . A signed measure, $\rho$ is said to be a right-admissible perturbation at $Q$ if $\begin{array} { r } { \int _ { \mathbb { R } ^ { d } } d \rho = 0 , \int _ { \mathbb { R } ^ { d } } d | \rho | , \int \| x \| d | \rho | < \infty } \end{array}$ , and $Q ^ { \epsilon } \in \mathcal { P } ( \mathbb { R } ^ { d } )$ for small $\epsilon > 0$ Similarly, $\rho$ is a left-admissible perturbation at $Q { \mathrm { ~ i f ~ } } - \rho$ is right-admissible. We call $\rho$ an admissible perturbation at $Q$ if $\rho$ is both right and left-admissible at $Q$

Let $Q \mapsto J ( Q )$ be a locally Lipschitz continuous functional (with respect to the Wasserstein-1 metric) on ${ \mathcal { P } } ( \mathbb { R } ^ { d } )$ . Then, the Clarke generalized directional derivative [11] of J at $Q$ for

an admissible perturbation $\rho$ is defined by

$$
J ^ { \circ } ( Q ; \rho ) : = \operatorname* { l i m } _ { Q ^ { \prime } \to Q , \ t \downarrow 0 } \frac { J ( Q ^ { \prime } + t \rho ) - J ( Q ^ { \prime } ) } { t } .
$$

The Clarke subdiferential of $J$ at $Q$ is defined as

$$
\begin{array} { r } { \partial J ( Q ) : = \Bigl \{ \xi : ~ J ^ { \circ } ( Q ; \rho ) \geq \int \xi d \rho ~ \mathrm { f o r ~ a l l ~ a d m i s s i b l e ~ p e r t u r b a t i o n s } ~ \rho \Bigr \} , } \end{array}
$$

and its elements are referred to as the first variational subgradients of $J$ at $Q .$ We note that we adopt the general terminology, variational subgradients, for Clarke’s generalized gradients, whereas for concave functionals such as CVaR, Clarke’s generalized gradients are also referred to as supergradients. We provide definitions and relevant background on Clarke’s generalized gradients in Appendix A.

Locally Lipschitz continuity with respect to the Wasserstein-1 metric To define Clarke’s generalized gradients, the functional J should be locally Lipschitz continuous with respect to some measure in ${ \mathcal { P } } ( \mathbb { R } ^ { d } )$ , such that $J ^ { \circ } ( Q ; \rho )$ is finite. Since the perturbed probability measure $Q ^ { \epsilon } = Q + \epsilon \rho$ corresponds to transporting probability mass, Wasserstein metrics that quantify the cost of such transportation are the natural metrics for this purpose (as opposed to metrics insensitive to spatial displacement, such as the total variation norm). The Wasserstein gradient flow (3.14) is classically formulated in the Wasserstein-2 space $( \mathcal { P } _ { 2 } ( \mathbb { R } ^ { d } ) , \mathcal { W } _ { 2 } )$ [24, 30], so one may establish the Lipschitz continuity with respect to the Wasserstein-2 metric. However, since $\mathcal { W } _ { 1 } \leq \mathcal { W } _ { 2 }$ whenever both are defined, Lipschitz continuity with respect to the Wasserstein-1 metric implies the Lipschitz continuity with respect to the Wasserstein-2 metric. In our work, we adopt the stronger Lipschitz continuity with respect to the Wasserstein-1 metric.

Lemma 3.1. Let $J ( Q )$ be locally Lipschitz continuous with respect to the Wasserstein-1 metric on ${ \mathcal { P } } ( \mathbb { R } ^ { d } )$ . Then for any admissible perturbation $\rho _ { ; }$ , the Clarke generalized directional derivative $J ^ { \circ } ( Q ; \rho )$ exists finitely.

Proof. Let $J ( Q )$ be locally Lipschitz continuous with respect to the Wasserstein-1 metric on ${ \mathcal { P } } ( \mathbb { R } ^ { d } )$ . Let $\operatorname { L i p } ( J , Q )$ denote the Lipschitz constant of the functional J in a neighborhood of $Q { \mathrm { : } }$ , Then for an admissible perturbation $\rho ,$ we have

$$
\begin{array} { r l } & { | P ^ { * } ( Q ; \rho ) | = \left| \left| \underset { \phi \in \mathbb { H } _ { + } } { \operatorname* { m i n } } \cdot \frac { J ( Q ^ { * } + ( \rho ) - J ( Q ^ { * } ) ) } { L } \right. \right. } \\ & { \qquad \leq \left. \underset { \phi \in \mathbb { H } _ { + } , \phi \in \mathcal { G } } { \operatorname* { m i n } } [ \mathrm { i n } ( Q , Q ) ] \underset { \phi \in \mathbb { H } _ { + } } { \operatorname* { m i n } } [ Q \cdot ( \rho , Q ) ] \right. } \\ & { \qquad = \left. \underset { \phi \in \mathbb { H } _ { + } , \phi \in \mathcal { G } } { \operatorname* { m i n } } [ \mathrm { L i p } ( Q ) ] \underset { \phi \in \mathcal { G } } { \operatorname* { m a x } } [ \mathrm { s i g } ( Q ^ { * } + t \rho - Q ) ] \right. } \\ & { \qquad = \left. \mathrm { 1 i s p } ( L , L ) \underset { \phi \in \mathbb { H } _ { + } } { \operatorname* { m i n } } \int _ { \phi \in \mathcal { G } } \int _ { \phi } \int _ { \phi } \langle \mathrm { s i g } ( \mathrm { s i g } ( \mathrm { s i g } ( \mathrm { s i g } ( \mathrm { s i g } ( \mathrm { s i g } ( \mathrm { s i g } ) ) ) ) ) \mathrm { d } \phi \rangle \right. } \\ & { \qquad = \mathrm { 1 i s p } ( L , L ) \underset { \phi \in \mathbb { H } _ { + } } { \operatorname* { m i n } } [ \mathrm { s i g } ( \mathrm { s i g } ( \mathrm { s i g } ) ) ] \underset { \phi \in \mathcal { G } } { \operatorname* { m a x } } [ \mathrm { s i g } ( \mathrm { s i g } ( \mathrm { s i g } ) ) ] \underset { \phi \in \mathcal { G } } { \operatorname* { m a x } } [ \mathrm { d } \mathrm { s i g } ( \mathrm { s i g } ) ] } \\ &  \qquad \leq \mathrm { L i p } ( J , Q ) \underset { \phi \in \mathbb { H } _ { + } } { \operatorname* { m i n } } [ \mathrm { s i g } ( \mathrm { s i g } ) ] \underset { \phi \in \mathcal { G } } { \operatorname* { m a x } } [ \mathrm { s i g } (  \end{array}
$$

where the second equality follows from the Kantorovich-Rubinstein duality [25],

$$
\operatorname * { s u p } _ { \psi \in \Gamma _ { 1 } } \int \psi d ( Q _ { 1 } - Q _ { 2 } ) = { \mathcal W } _ { 1 } ( Q _ { 1 } , Q _ { 2 } ) .
$$

We need $\mathrm { C V a R } _ { \alpha } ^ { Q , g }$ and $Q \mapsto { \mathcal { F } } ^ { \operatorname { C V a R } } ( Q ; P ^ { \operatorname { t a r } } )$ to be locally Lipschitz continuous with respect to the Wasserstein-1 metric, for their variational subgradients to be well-defined. To this end, we have the following proposition, whose proof is given in Appendix B.

Proposition 3.2 (Local Lipschitz continuity). Assume that g is $\operatorname { L i p } ( g )$ -Lipschitz continuous and Assumption 2.1 holds. Then we have:

1. The risk measure $Q \mapsto { \mathrm { C V a R } } _ { \alpha } ^ { Q , g }$ is Lipschitz continuous with respect to the Wasserstein-1 metric, $i . e .$

$$
\left. \mathrm { C V a R } _ { \alpha } ^ { Q _ { 1 } , g } - \mathrm { C V a R } _ { \alpha } ^ { Q _ { 2 } , g } \right. \leq \frac { \mathrm { L i p } ( g ) } { 1 - \alpha } \mathcal { W } _ { 1 } ( Q _ { 1 } , Q _ { 2 } ) ,\tag{3.16}
$$

for all $Q _ { 1 } , Q _ { 2 } \in \mathcal { Q } ^ { g }$

2. The functional $Q \mapsto { \mathcal { F } } ^ { \operatorname { C V a R } } ( Q ; P ^ { \operatorname { t a r } } )$ is locally Lipschitz continuous on $\mathcal { Q } ^ { g }$ with respect to $\mathcal { W } _ { 1 }$

We restrict the choice of the risk function $g$ to Lipschitz continuous functions in what follows. To derive the variational subgradients of $\mathrm { C V a R } _ { \alpha } ^ { Q , g }$ and ${ \mathcal { F } } ^ { \mathrm { C V a R } } ( Q ; P ^ { \mathrm { t a r } } )$ , we need the following lemma, whose proof is given in Appendix A.1.

Lemma 3.3. Let $J ( Q )$ be locally Lipschitz continuous with respect to the Wasserstein-1 metric on ${ \mathcal { P } } ( \mathbb { R } ^ { d } )$ . Define the right/left one-sided directional derivatives of J as:

$$
D _ { \rho } ^ { \pm } J : = \operatorname * { l i m } _ { \epsilon \to 0 ^ { \pm } } \frac { 1 } { \epsilon } \big ( J ( Q ^ { \epsilon } ) - J ( Q ) \big ) .\tag{3.17}
$$

Assume that for every admissible perturbation $\rho$ at $Q$ , the one-sided directional derivatives $D _ { \rho } ^ { \pm } J$ exist and $\int \xi d \rho$ lies in the closed interval with endpoints $D _ { \rho } ^ { - } J$ and $D _ { \rho } ^ { + } J$ . Then $\xi$ is a variational subgradient of J at $Q$

Variational subgradients of $\mathrm { C V a R } _ { \alpha } ^ { Q , g }$ To derive the variational subgradients of the loss functional ${ \mathcal { F } } ^ { \operatorname { C V a R } } ( Q ; P ^ { \mathrm { t a r } } )$ , we first derive those of $\mathrm { C V a R } _ { \alpha } ^ { Q , g }$ . This requires $Q ^ { \epsilon } \in \mathcal { Q } ^ { g }$ , so that $\mathrm { C V a R } _ { \alpha } ^ { Q ^ { \epsilon } , g }$ is well-defined, for all small ϵ of the relevant sign. Since $g$ is non-negative and Lipschitz continuous, $0 \leq g ( x ) \leq g ( 0 ) + \mathrm { L i p } ( g ) \| x \|$ , so for any admissible (or merely right- or left-admissible) $\rho$ at $Q$ ,

$$
\int g d | \rho | \le g ( 0 ) \int d | \rho | + \mathrm { L i p } ( g ) \int \| x \| d | \rho | < \infty ,\tag{3.18}
$$

and $Q ^ { \epsilon } \in \mathcal { Q } ^ { g }$ for all small ϵ such that $Q ^ { \epsilon } \in \mathcal { P } ( \mathbb { R } ^ { d } )$ . Hence no integrability condition on $\rho$ beyond admissibility is needed for $\mathrm { C V a R } _ { \alpha } ^ { Q ^ { \epsilon } , g }$ to be well-defined. We begin with the following lemma, whose proof is provided in Appendix C.

Lemma 3.4. Assume that g is Lipschitz continuous and $Q \in \mathcal { Q } ^ { g }$ . Let $\rho$ be right-admissible at $Q$ . Then there exists $\epsilon _ { 0 } > 0$ such that

1. The function $( \epsilon , y ) \mapsto F _ { \alpha } ^ { Q ^ { \epsilon } , g } ( y )$ is jointly continuous in $[ 0 , \epsilon _ { 0 } ] \times \mathbb { R }$

2. The set $\begin{array} { r } { T : = \bigcup _ { \epsilon \in [ 0 , \epsilon _ { 0 } ] } T ( Q ^ { \epsilon } ) } \end{array}$ is compact.

The following theorem derives the variational subgradients of $\mathrm { C V a R } _ { \alpha } ^ { Q , g }$

Theorem 3.5 (Directional derivatives and variational subgradients of CVaR). Assume that $g$ is Lipschitz continuous and $Q$ has finite first moment (hence $Q \in \mathcal { Q } ^ { g } )$ . For $y \in \mathbb { R }$ 2 define $\begin{array} { r } { h ^ { y } ( x ) : = \frac { ( g ( x ) - y ) ^ { + } } { 1 - \alpha } } \end{array}$ . Then we have the following:

1. (Right derivative) For every right-admissible perturbation $\rho ,$ the right directional derivative exists and is given by

$$
D _ { \rho } ^ { + } \mathrm { C V a R } _ { \alpha } ^ { Q , g } = \operatorname* { l i m } _ { \epsilon \to 0 ^ { + } } \frac { \mathrm { C V a R } _ { \alpha } ^ { Q ^ { \epsilon } , g } - \mathrm { C V a R } _ { \alpha } ^ { Q , g } } { \epsilon } = \operatorname* { m i n } _ { y \in T ( Q ) } \int h ^ { y } d \rho .\tag{3.19}
$$

2. (Left derivative) For every left-admissible perturbation $\rho ,$ , the left directional derivative exists and is given by

$$
D _ { \rho } ^ { - } \mathrm { C V a R } _ { \alpha } ^ { Q , g } = \operatorname* { l i m } _ { \epsilon \to 0 ^ { - } } \frac { \mathrm { C V a R } _ { \alpha } ^ { Q ^ { \epsilon } , g } - \mathrm { C V a R } _ { \alpha } ^ { Q , g } } { \epsilon } = \operatorname* { m a x } _ { y \in T ( Q ) } \int h ^ { y } d \rho .\tag{3.20}
$$

3. (Variational subgradients) Let $y \in T ( Q )$ . Then, for any admissible $\rho ,$ R $h ^ { y } d \rho$ lies in $[ D _ { \rho } ^ { + } \mathrm { C V a R } _ { \alpha } ^ { Q , g } , D _ { \rho } ^ { - } \mathrm { C V a R } _ { \alpha } ^ { Q , g } ]$ . Consequently, for each $y \in T ( Q )$ $h ^ { y }$ is a variational subgradient of $\mathrm { C V a R } _ { \alpha } ^ { Q , g }$

4. (Two-sided derivative) The two-sided (Gateaux) derivative for an admissible $\rho$ exists if and only if $y \mapsto \int h ^ { y }$ dρ is constant on $T ( Q )$ ; this holds for every admissible $\rho$ if and only if $T ( Q )$ is a singleton set, i.e. Va $\begin{array} { r } { \mathrm { R } _ { \alpha } ^ { Q , g } = \overline { { \mathrm { V a R } } } _ { \alpha } ^ { Q , g } } \end{array}$ , in which case $\mathrm { C V a R } _ { \alpha } ^ { Q , g }$ is Gateaux diferentiable at Q with the first variation

$$
\frac { \delta \mathrm { C V a R } _ { \alpha } ^ { Q , g } } { \delta Q } = h ^ { \mathrm { V a R } _ { \alpha } ^ { Q , g } } = \frac { ( g - \mathrm { V a R } _ { \alpha } ^ { Q , g } ) ^ { + } } { 1 - \alpha } .
$$

Proof. For any perturbation $\rho ,$ we define $\begin{array} { r } { \ell _ { \rho } ( y ) : = \frac { 1 } { 1 - \alpha } \int ( g - y ) ^ { + } d \rho = \int h ^ { y } d \rho . } \end{array}$

(1) Let $\rho$ be right-admissible and let $\epsilon _ { 0 } > 0$ be given by Lemma 3.4. From $Q ^ { \epsilon } = Q + \epsilon \rho .$ we have

$$
F _ { \alpha } ^ { Q ^ { \epsilon } , g } ( y ) = F _ { \alpha } ^ { Q , g } ( y ) + \epsilon \ell _ { \rho } ( y ) .\tag{3.21}
$$

Thus, it follows that

$$
\begin{array} { r l } & { \mathrm { { C V a R } } _ { \alpha } ^ { Q ^ { \epsilon } , g } \leq \underset { y \in { \cal T } ( Q ) } { \operatorname* { i n f } } F _ { \alpha } ^ { Q ^ { \epsilon } , g } ( y ) } \\ & { \quad \quad \quad = \underset { y \in { \cal T } ( Q ) } { \operatorname* { i n f } } \{ F _ { \alpha } ^ { Q , g } ( y ) + \epsilon \ell _ { \rho } ( y ) \} } \\ & { \quad \quad \quad = { \mathrm { C V a R } } _ { \alpha } ^ { Q , g } + \epsilon \underset { y \in { \cal T } ( Q ) } { \operatorname* { m i n } } \ell _ { \rho } ( y ) . } \end{array}\tag{3.22}
$$

Hence, we have

$$
\varlimsup _ { \epsilon  0 ^ { + } } \frac { ( \mathrm { C V a R } _ { \alpha } ^ { Q ^ { \epsilon } , g } - \mathrm { C V a R } _ { \alpha } ^ { Q , g } ) } { \epsilon } \leq \operatorname* { m i n } _ { y \in T ( Q ) } \ell _ { \rho } ( y ) = \operatorname* { m i n } _ { y \in T ( Q ) } \int h ^ { y } d \rho .\tag{3.23}
$$

Pick $y _ { \epsilon } \in T ( Q ^ { \epsilon } )$ . Using (3.21) and $F _ { \alpha } ^ { Q , g } ( y _ { \epsilon } ) \ge \mathrm { C V a R } _ { \alpha } ^ { Q , g }$ , we have

$$
\frac { \mathrm { C V a R } _ { \alpha } ^ { Q ^ { \epsilon } , g } - \mathrm { C V a R } _ { \alpha } ^ { Q , g } } { \epsilon } = \frac { F _ { \alpha } ^ { Q , g } ( y _ { \epsilon } ) - \mathrm { C V a R } _ { \alpha } ^ { Q , g } } { \epsilon } + \ell _ { \rho } ( y _ { \epsilon } ) \geq \ell _ { \rho } ( y _ { \epsilon } ) .\tag{3.24}
$$

Fix $\epsilon _ { n } \downarrow 0 ;$ by compactness of $T$ pass to a subsequence with $y _ { \epsilon _ { n } }  \bar { y } \in T$ . Since $y \mapsto \ell _ { \rho } ( y )$ is continuous on $\mathbb { R } .$ , there exists $M > 0$ such that $\textstyle \operatorname* { s u p } _ { y \in T } | \ell _ { \rho } ( y ) | \leq M$ . Thus, for any $y ^ { \ast } \in \mathbb { R }$ we have

$$
\begin{array} { r l } & { F _ { \alpha } ^ { Q , g } ( y _ { \epsilon _ { n } } ) = F _ { \alpha } ^ { Q ^ { \epsilon _ { n } } , g } ( y _ { \epsilon _ { n } } ) - \epsilon _ { n } \ell _ { \rho } ( y _ { \epsilon _ { n } } ) } \\ & { \qquad \leq F _ { \alpha } ^ { Q ^ { \epsilon _ { n } } , g } ( y ^ { * } ) - \epsilon _ { n } \ell _ { \rho } ( y _ { \epsilon _ { n } } ) } \\ & { \qquad = F _ { \alpha } ^ { Q , g } ( y ^ { * } ) + \epsilon _ { n } \ell _ { \rho } ( y ^ { * } ) - \epsilon _ { n } \ell _ { \rho } ( y _ { \epsilon _ { n } } ) } \\ & { \qquad \leq F _ { \alpha } ^ { Q , g } ( y ^ { * } ) + M \epsilon _ { n } + \vert \ell _ { \rho } ( y ^ { * } ) \vert \epsilon _ { n } . } \end{array}
$$

Letting $\epsilon _ { n } \to 0$ , and using the continuity of $F _ { \alpha } ^ { Q , g } ( \cdot )$ , we obtain that $\begin{array} { r l } { \operatorname* { l i m } _ { n \to \infty } F _ { \alpha } ^ { Q , g } ( y _ { \epsilon _ { n } } ) = } \end{array}$ $F _ { \alpha } ^ { Q , g } ( \bar { y } ) \le F _ { \alpha } ^ { Q , g } ( y ^ { * } )$ for all $y ^ { \ast } \in \mathbb { R }$ . Thus $\bar { y } \in T ( Q )$

Thus, by (3.24), and the continuity of $\ell _ { \rho } ( \cdot )$ , we have

$$
\operatorname* { l i m } _ { n \to \infty } \frac { \mathrm { C V a R } _ { \alpha } ^ { Q ^ { \epsilon n } , g } - \mathrm { C V a R } _ { \alpha } ^ { Q , g } } { \epsilon _ { n } } \geq \operatorname* { l i m } _ { n \to \infty } \ell _ { \rho } ( y _ { \epsilon _ { n } } ) = \ell _ { \rho } ( \bar { y } ) \geq \operatorname* { m i n } _ { y \in T ( Q ) } \ell _ { \rho } ( y ) = \operatorname* { m i n } _ { y \in T ( Q ) } \int h ^ { y } d \rho .\tag{3.25}
$$

Thus, we can conclude using (3.23) and (3.25) that for any sequence $\epsilon _ { k } \to 0 ^ { + }$ in the interval $[ 0 , \epsilon _ { 0 } ]$ , there exists a subsequence $\epsilon _ { n _ { k } }$ such that

$$
\operatorname* { l i m } _ { \epsilon _ { n _ { k } } \to 0 ^ { + } } \frac { \mathrm { C V a R } _ { \alpha } ^ { Q ^ { \epsilon _ { n _ { k } } } , g } - \mathrm { C V a R } _ { \alpha } ^ { Q , g } } { \epsilon _ { n _ { k } } } = \operatorname* { m i n } _ { y \in T ( Q ) } \int h ^ { y } d \rho .\tag{3.26}
$$

Hence,

$$
\operatorname* { l i m } _ { \epsilon \to 0 ^ { + } } \frac { \mathrm { C V a R } _ { \alpha } ^ { Q ^ { \epsilon } , g } - \mathrm { C V a R } _ { \alpha } ^ { Q , g } } { \epsilon } = \operatorname* { m i n } _ { y \in T ( Q ) } \int h ^ { y } d \rho ,\tag{3.27}
$$

proving (3.19).

(2) Let $\rho$ be left-admissible. let $\epsilon = - \delta , \delta > 0$ . Thus $Q ^ { \epsilon } = Q + \epsilon \rho = Q - \delta \rho$ . We have

$$
\begin{array} { l } { { \displaystyle D _ { \rho } ^ { - } = \operatorname* { l i m } _ { \epsilon \to 0 ^ { - } } \frac { \mathrm { C V a R } _ { \alpha } ^ { Q \epsilon , g } - \mathrm { C V a R } _ { \alpha } ^ { Q , g } } { \epsilon } } } \\ { { \displaystyle \quad = - \operatorname* { l i m } _ { \delta \to 0 ^ { + } } \frac { \mathrm { C V a R } _ { \alpha } ^ { Q - \delta \rho , g } - \mathrm { C V a R } _ { \alpha } ^ { Q , g } } { \delta } } } \\ { { \displaystyle \quad = - \operatorname* { m i n } _ { y \in T ( Q ) } \int h ^ { y } d ( - \rho ) } } \\ { { \displaystyle \quad = \operatorname* { m a x } _ { y \in T ( Q ) } \int h ^ { y } d \rho } . }  \end{array}\tag{3.28}
$$

(3) Since $y \mapsto \ell _ { \rho } ( y )$ is continuous and $T ( Q )$ is a compact interval, $\begin{array} { r } { \int h ^ { y } \rho = \ell _ { \rho } ( y ) } \end{array}$ ranges over the interval between $D _ { \rho } ^ { + } \mathrm { C V a R } _ { \alpha } ^ { Q , g }$ and $D _ { \rho } ^ { - } \mathrm { C V a R } _ { \alpha } ^ { Q , g }$ , attaining both extremes. Thus, by Lemma 3.3, it follows that the family of functions $\{ h ^ { y } ; y \in T ( Q ) \}$ are variational subgradients of $\mathrm { C V a R } _ { \alpha } ^ { Q , g }$

(4) Let $\rho$ be a perturbation such that $\rho$ is admissible at $Q$ . By parts (1) and (2), both one-sided directional derivatives exist and are given by

$$
D _ { \rho } ^ { + } \mathrm { C V a R } _ { \alpha } ^ { Q , g } = \operatorname* { m i n } _ { y \in { \cal T } ( Q ) } \ell _ { \rho } ( y ) , \qquad D _ { \rho } ^ { - } \mathrm { C V a R } _ { \alpha } ^ { Q , g } = \operatorname* { m a x } _ { y \in { \cal T } ( Q ) } \ell _ { \rho } ( y ) .
$$

The two-sided (Gateaux) derivative in direction $\rho$ exists if and only if these one-sided derivatives coincide, that is, if and only if

$$
\operatorname* { m i n } _ { y \in T ( Q ) } \ell _ { \rho } ( y ) = \operatorname* { m a x } _ { y \in T ( Q ) } \ell _ { \rho } ( y ) .
$$

That ${ \mathrm { i s } } ,$ the two-sided (Gateaux) derivative in direction $\rho$ exists if and only if $y \mapsto \ell _ { \rho } ( y )$ is constant on $T ( Q )$

Assume that $T ( Q )$ is a singleton set, i.e. $\mathrm { V a R } _ { \alpha } ^ { Q , g } = \overline { { \mathrm { V a R } } } _ { \alpha } ^ { Q , g }$ , so that $T ( Q ) = \{ \mathrm { V a R } _ { \alpha } ^ { Q , g } \}$ Then $y \mapsto \ell _ { \rho } ( y )$ is constant on $T ( Q )$ for every admissible $\rho ,$ and

$$
D _ { \rho } ^ { + } \mathrm { C V a R } _ { \alpha } ^ { Q , g } = D _ { \rho } ^ { - } \mathrm { C V a R } _ { \alpha } ^ { Q , g } = \ell _ { \rho } \bigl ( \mathrm { V a R } _ { \alpha } ^ { Q , g } \bigr ) = \int h ^ { \mathrm { V a R } _ { \alpha } ^ { Q , g } } d \rho .
$$

The two-sided derivative therefore exists for every admissible perturbation; that is, $\mathrm { C V a R } _ { \alpha } ^ { Q , g }$ is Gateaux diferentiable at $Q$ with first variation

$$
\frac { \delta \mathrm { C V a R } _ { \alpha } ^ { Q , g } } { \delta Q } = h ^ { \mathrm { V a R } _ { \alpha } ^ { Q , g } } = \frac { ( g - \mathrm { V a R } _ { \alpha } ^ { Q , g } ) ^ { + } } { 1 - \alpha } .
$$

Conversely, assume that Va $\ R _ { \alpha } ^ { Q , g } \ < \ \overline { { \mathrm { V a R } } } _ { \alpha } ^ { Q , g }$ . Define $k ( x ) = \left( g ( x ) - \mathrm { V a R } _ { \alpha } ^ { Q , g } \right) ^ { + } - \left( g ( x ) - \mathrm { V a R } _ { \alpha } ^ { Q , g } \right) ^ { - }$ $\overline { { \mathrm { V a R } } } _ { \alpha } ^ { Q , g } ) ^ { + }$ . Define $\varphi _ { k } : = k - \mathbb { E } _ { Q } [ k ]$ . Since $0 \leq k \leq \overline { { \mathrm { V a R } } } _ { \alpha } ^ { Q , g } - \mathrm { V a R } _ { \alpha } ^ { Q , g }$ , both k and $\varphi _ { k }$ are bounded. Define the perturbation measure $\rho : = \varphi _ { k } Q$ . We verify that $\rho$ is admissible at $Q$ . First, $\begin{array} { r } { \int d \boldsymbol { \rho } = \int \varphi _ { k } d Q = \mathbb { E } _ { Q } [ k ] - \mathbb { E } _ { Q } [ k ] = 0 } \end{array}$ and $\begin{array} { r } { \int d | \rho | = \int | \varphi _ { k } | d Q \le \| \varphi _ { k } \| _ { \infty } < \infty } \end{array}$ Second, for $| \epsilon | \leq \| \varphi _ { k } \| _ { \infty } ^ { - 1 }$ we have $1 + \epsilon \varphi _ { k } \ge 0$ point-wise, so $Q ^ { \epsilon } = Q + \epsilon \rho = \left( 1 + \epsilon \varphi _ { k } \right) Q$ is a non-negative measure of total mass 1, i.e. $Q ^ { \epsilon } \in \mathcal { P } ( \mathbb { R } ^ { d } )$ for all $| \epsilon | \leq \| \varphi _ { k } \| _ { \infty } ^ { - 1 }$ . Third, we have

$$
\int \| x \| d | \rho | = \int \| x \| | \varphi _ { k } | d Q \leq \| \varphi _ { k } \| _ { \infty } \int \| x \| d Q ( x ) ,\tag{3.29}
$$

and since $\varphi _ { k }$ is bounded and Q has a finite first moment, we have $\textstyle \int \| x \| d | \rho | < \infty$ . Hence, $\rho$ is admissible. Since $\rho = \varphi _ { k } Q$

$$
\begin{array} { r l } { \ell _ { \rho } \big ( \mathrm { V a R } _ { \alpha } ^ { Q , g } \big ) - \ell _ { \rho } \big ( \overline { { \mathrm { V a R } } } _ { \alpha } ^ { Q , g } \big ) = \displaystyle \frac { 1 } { 1 - \alpha } \int \Big [ \big ( g - \mathrm { V a R } _ { \alpha } ^ { Q , g } \big ) ^ { + } - \big ( g - \overline { { \mathrm { V a R } } } _ { \alpha } ^ { Q , g } \big ) ^ { + } \Big ] d \rho } & { } \\ { = \displaystyle \frac { 1 } { 1 - \alpha } \int k \varphi _ { k } d Q } & { } \\ { = \displaystyle \frac { 1 } { 1 - \alpha } \mathbb { E } _ { Q } \big [ k \big ( k - \mathbb { E } _ { Q } [ k ] \big ) \big ] } & { } \\ { = \displaystyle \frac { \mathrm { V a r } _ { Q } ( k ) } { 1 - \alpha } . } \end{array}
$$

where $\operatorname { V a r } _ { Q } ( k )$ denotes the variance of k under the probability measure Q. Since $\mathrm { V a R } _ { \alpha } ^ { Q , g } <$ $\overline { { \mathrm { V a R } } } _ { \alpha } ^ { Q , g }$ , the CDF $\Psi ^ { Q , g }$ satisfies $\Psi ^ { Q , g } ( c ) = \alpha$ for all $c \in [ \mathrm { V a R } _ { \alpha } ^ { Q , g } , \overline { { \mathrm { V a R } } } _ { \alpha } ^ { Q , g } )$ . Consequently, $Q ( \{ k = 0 \} ) = Q ( g \leq \mathrm { V a R } _ { \alpha } ^ { Q , g } ) = \Psi ^ { Q , g } ( \mathrm { V a R } _ { \alpha } ^ { Q , g } ) = \alpha$ and $Q ( \{ k > 0 \} ) = 1 - \alpha > 0$ . Since the probability of $k > 0$ and the probability of $k = 0$ are strictly positive under the probability measure $Q , \operatorname { V a r } _ { Q } ( k ) > 0$ . Thus we have,

$$
\ell _ { \rho } \bigl ( \mathrm { V a R } _ { \alpha } ^ { Q , g } \bigr ) - \ell _ { \rho } \bigl ( \overline { { \mathrm { V a R } } } _ { \alpha } ^ { Q , g } \bigr ) = \frac { \mathrm { V a r } _ { Q } ( k ) } { 1 - \alpha } > 0 .\tag{3.30}
$$

Thus $\ell _ { \rho }$ is non-constant on $T ( Q )$ , and hence

$$
D _ { \rho } ^ { + } \mathrm { C V a R } _ { \alpha } ^ { Q , g } = \operatorname* { m i n } _ { y \in T ( Q ) } \ell _ { \rho } ( y ) < \operatorname* { m a x } _ { y \in T ( Q ) } \ell _ { \rho } ( y ) = D _ { \rho } ^ { - } \mathrm { C V a R } _ { \alpha } ^ { Q , g } .
$$

The two-sided derivative fails to exist for the perturbation $\rho .$ . We conclude that the twosided derivative exists for every admissible perturbation if and only if $T ( Q )$ is a singleton set. □

Remark 3.6. We stress that for the radial risk function $g ( x ) = \| x \|$ , Theorem 3.5 holds under the minimal assumption $Q \in \mathcal { Q } ^ { g }$ (which implies that Q has finite first moment), which is precisely what is needed $f o r \mathrm { \ C V a R } _ { \alpha } ^ { Q , g }$ to be well-defined. In our work, we do not enforce any additional hypothesis on the set $T ( Q )$

Part (4) of Theorem 3.5 recovers the classical formula $\begin{array} { r } { \frac { \delta \mathrm { C V a R } _ { \alpha } ^ { Q , g } } { \delta Q } = \frac { ( g - \mathrm { V a R } _ { \alpha } ^ { Q , g } ) ^ { + } } { 1 - \alpha } } \end{array}$ employed in $\begin{array} { r l } { [ 1 3 , } & { { } 3 3 ] . } \end{array}$ it is valid when $T ( Q )$ is a singleton set. The classical formula $\begin{array} { r l } { \frac { \delta \mathrm { C V a R } _ { \alpha } ^ { Q , g } } { \delta Q } = } & { { } } \end{array}$ $\frac { ( g - \mathrm { V a R } _ { \alpha } ^ { Q , g } ) ^ { + } } { 1 - \alpha }$ for the variational derivative of $\mathrm { C V a R } _ { \alpha } ^ { Q , g }$ is derived in [13, 33] under the density assumption on the distribution of $g ( X ) , X \sim Q$ . Such a density assumption, under which $T ( Q )$ is a singleton set, is violated when the probability measures are replaced by their empirical measures. Hence, the subgradient framework of Theorem 3.5 is necessary when the risk measure CVaR and its variational derivative are utilized.

We now consider the variational subgradients of the loss functional $\mathcal { F } ^ { \mathrm { C V a R } }$ itself.

Theorem 3.7 (Directional derivatives and variational subgradients of $\mathcal { F } ^ { \mathrm { C V a R } } )$ . Assume that $g$ is Lipschitz continuous and Assumption 2.1 holds. Let $\phi ^ { * }$ denote the maximizer of (2.6). For each $y \in T ( Q )$ , define the potential function

$$
\Phi _ { Q } ^ { y } : = \phi ^ { * } - 2 \lambda \Delta C ( Q ; P ^ { \mathrm { t a r } } ) h ^ { y } = \phi ^ { * } - \frac { 2 \lambda } { 1 - \alpha } \Delta C ( Q ; P ^ { \mathrm { t a r } } ) ( g - y ) ^ { + } .\tag{3.31}
$$

Then the following hold.

1. For every right-admissible perturbation $\rho ,$ the right directional derivative exists and is given by

$$
D _ { \rho } ^ { + } \mathcal { F } ^ { \mathrm { C V a R } } ( Q ; P ^ { \mathrm { t a r } } ) = \int \phi ^ { * } d \rho - 2 \lambda \Delta C ( Q ; P ^ { \mathrm { t a r } } ) \operatorname* { m i n } _ { y \in T ( Q ) } \int h ^ { y } d \rho .\tag{3.32}
$$

For every left-admissible perturbation $\rho ,$ the left directional derivative exists and is given by

$$
D _ { \rho } ^ { - } \mathcal { F } ^ { \mathrm { C V a R } } ( Q ; P ^ { \mathrm { t a r } } ) = \int \phi ^ { * } d \rho - 2 \lambda \Delta C ( Q ; P ^ { \mathrm { t a r } } ) \operatorname* { m a x } _ { y \in T ( Q ) } \int h ^ { y } d \rho .\tag{3.33}
$$

2. For any admissible $\rho ,$ the range of $\int \Phi _ { Q } ^ { y } d \rho$ for $y \in T ( Q )$ , is the compact interval between $D _ { \rho } ^ { - } { \mathcal { F } } ^ { \mathrm { C V a R } } ( Q ; P ^ { \mathrm { t a r } } )$ and $D _ { \rho } ^ { + } \mathcal { F } ^ { \mathrm { C V a R } } ( Q ; P ^ { \mathrm { t a r } } )$ . Consequently, the family of potential functions $\{ \Phi _ { Q } ^ { y } ; y \in T ( Q ) \}$ are subgradients of ${ \mathcal { F } } ^ { \operatorname { C V a R } } ( Q ; P ^ { \mathrm { t a r } } )$

3. Let $\rho$ be admissible. The two-sided (Gateaux) derivative of $\mathcal { F } ^ { \mathrm { C V a R } }$ for the perturbation $\rho$ exists if and only if $\begin{array} { r } { \Delta C ( Q ; P ^ { \mathrm { t a r } } ) = 0 \ o r \ y \mapsto \int h ^ { y } d \rho } \end{array}$ is constant on $T ( Q )$ . The two-sided (Gateaux) derivative of $\mathcal { F } ^ { \mathrm { C V a R } }$ exists for every admissible perturbation if and only if $\Delta C ( Q ; P ^ { \mathrm { t a r } } ) = 0 ~ o r ~ T ( Q )$ is a singleton set; in that case, $\mathcal { F } ^ { \mathrm { C V a R } } ( \cdot ; P ^ { \mathrm { t a r } } )$ is Gateaux diferentiable at $Q$ with first variational derivative

$$
{ \frac { \delta { \mathcal { F } } ^ { \mathrm { C V a R } } ( Q ; P ^ { \mathrm { t a r } } ) } { \delta Q } } = \Phi _ { Q } ^ { \mathrm { V a R } _ { \alpha } ^ { Q , g } } .
$$

Proof. (1) For a right-admissible perturbation $\rho ,$ we have that

$$
\begin{array} { r l r } { \displaystyle { D _ { \rho } ^ { + } ( \Delta C ( Q ; P ^ { \mathrm { t a r } } ) ) ^ { 2 } = \operatorname* { l i m } _ { \epsilon \to 0 ^ { + } } \frac { ( \Delta C ( Q ^ { \epsilon } ; P ^ { \mathrm { t a r } } ) ) ^ { 2 } - ( \Delta C ( Q ; P ^ { \mathrm { t a r } } ) ) ^ { 2 } } { \epsilon } } } \\ { \displaystyle { \quad \quad } } & { \displaystyle { \quad \quad = \operatorname* { l i m } _ { \epsilon \to 0 ^ { + } } \left[ \mathrm { C V a R } _ { \alpha } ^ { Q ^ { \epsilon } , g } + \mathrm { C V a R } _ { \alpha } ^ { Q , g } - 2 \mathrm { C V a R } _ { \alpha } ^ { P ^ { \mathrm { t a r } } , g } \right] \frac { \mathrm { C V a R } _ { \alpha } ^ { Q ^ { \epsilon } , g } - \mathrm { C V a R } _ { \alpha } ^ { Q , g } } { \epsilon } } } \\ { \displaystyle { \quad \quad } } & { \displaystyle { \quad \quad = - 2 \Delta C ( Q ; P ^ { \mathrm { t a r } } ) D _ { \rho } ^ { + } \mathrm { C V a R } _ { \alpha } ^ { Q , g } , } } \\ { \displaystyle { \quad \quad } } & { \displaystyle { \quad \quad = - 2 \Delta C ( Q ; P ^ { \mathrm { t a r } } ) \operatorname* { m i n } _ { y \in T ( Q ) } \int h ^ { y } d \rho . } } \end{array}\tag{4}
$$

We recall that the variational derivative of the divergence $D _ { \mathrm { K L } } ^ { L } ( Q \Vert P ^ { \mathrm { t a r } } )$ is given by

$$
\frac { \delta D _ { \mathrm { K L } } ^ { L } ( Q \| P ^ { \mathrm { t a r } } ) } { \delta Q } ( x ) = \phi ^ { * } ( x ) .\tag{3.35}
$$

where $\phi ^ { * }$ is the maximizer of (2.6). That is, for any right-admissible $\rho , D _ { \rho } ^ { + } D _ { \mathrm { K L } } ^ { L } ( Q \Vert P ^ { \mathrm { t a r } } ) =$ $\int \phi ^ { * } d \rho$ . Since $D _ { \rho } ^ { + } \mathcal { F } ^ { \mathrm { C V a R } } ( Q ; P ^ { \mathrm { t a r } } ) = D _ { \rho } ^ { + } D _ { \mathrm { K L } } ^ { L } ( Q \| P ^ { \mathrm { t a r } } ) + \lambda D _ { \rho } ^ { + } ( \Delta C ( Q ; P ^ { \mathrm { t a r } } ) ) ^ { 2 }$ , we readily obtain (3.32).

Now let $\rho$ be a left-admissible perturbation. Then we have $\begin{array} { r } { D _ { \rho } ^ { - } D _ { \mathrm { K L } } ^ { L } ( Q \| P ^ { \mathrm { t a r } } ) = \int \phi ^ { * } d \rho } \end{array}$ and $D _ { \rho } ^ { - } ( \Delta C ( Q ; P ^ { \mathrm { t a r } } ) ) ^ { 2 } = - 2 \Delta C ( Q ; P ^ { \mathrm { t a r } } )$ m $\operatorname { a x } _ { y \in T ( Q ) } \int h ^ { y } d \rho$ (similar to the derivation of $( 3 . 3 4 ) $ . Since $D _ { \rho } ^ { - } \mathcal { F } ^ { \mathrm { C V a R } } ( Q ; P ^ { \mathrm { t a r } } ) = D _ { \rho } ^ { - } D _ { \mathrm { K L } } ^ { L } ( Q \| P ^ { \mathrm { t a r } } ) + \lambda D _ { \rho } ^ { - } ( \Delta C ( Q ; P ^ { \mathrm { t a r } } ) ) ^ { 2 }$ , we readily obtain (3.33).

(2) Since $y \mapsto \int \Phi _ { Q } ^ { y } d \rho$ is continuous and $T ( Q )$ is a compact interval, $\int \Phi _ { Q } ^ { y } d \rho$ ranges over the interval between $D _ { \rho } ^ { + } \mathcal { F } ^ { \mathrm { C V a R } } ( Q ; P ^ { \mathrm { t a r } } )$ and $D _ { \rho } ^ { - } { \mathcal { F } } ^ { \mathrm { C V a R } } ( Q ; P ^ { \mathrm { t a r } } )$ , attaining both extremes. Thus, by Lemma 3.3, it follows that the family of potential functions $\{ \Phi _ { Q } ^ { y } ; y \in T ( Q ) \}$ are subgradients of ${ \mathcal { F } } ^ { \operatorname { C V a R } } ( Q ; P ^ { \mathrm { t a r } } )$ .

(3) For any admissible perturbation $\rho ,$ we have

$$
D _ { \rho } ^ { - } \mathcal { F } ^ { \mathrm { C V a R } } ( Q ; P ^ { \mathrm { t a r } } ) - D _ { \rho } ^ { + } \mathcal { F } ^ { \mathrm { C V a R } } ( Q ; P ^ { \mathrm { t a r } } ) = 2 \lambda \Delta C ( Q ; P ^ { \mathrm { t a r } } ) \Big ( \operatorname* { m i n } _ { y \in T ( Q ) } \int h ^ { y } d \rho - \operatorname* { m a x } _ { y \in T ( Q ) } \int h ^ { y } d \rho \Big )
$$

The two-sided derivative for perturbation $\rho$ exists if and only if $\Delta C ( Q ; P ^ { \mathrm { t a r } } ) = 0$ or $\int h ^ { y } d \rho$ is constant on $T ( Q )$ . As proven in Theorem 3.5, R $\mathbf { \dot { \mu } } h ^ { y } d \rho$ is constant on $T ( Q )$ for every admissible $\rho ,$ if and only if $T ( Q ) \ = \ \{ \mathrm { V a R } _ { \alpha } ^ { Q , g } \}$ is a singleton set. If $T ( Q )$ is the singleton set $\{ \mathrm { V a R } _ { \alpha } ^ { Q , g } \}$ , we have

$$
D _ { \rho } ^ { - } \mathcal { F } ^ { \mathrm { C V a R } } ( Q ; P ^ { \mathrm { t a r } } ) = D _ { \rho } ^ { + } \mathcal { F } ^ { \mathrm { C V a R } } ( Q ; P ^ { \mathrm { t a r } } ) = \int \phi ^ { * } d \rho - 2 \lambda \Delta C ( Q ; P ^ { \mathrm { t a r } } ) \int h ^ { \mathrm { V a R } _ { \alpha } ^ { Q , g } } d \rho ,
$$

for every admissible $\rho .$ . That is, $\mathcal { F } ^ { \mathrm { C V a R } } ( \cdot ; P ^ { \mathrm { t a r } } )$ is Gateaux diferentiable at $Q$ with first variational derivative

$$
{ \frac { \delta { \mathcal { F } } ^ { \mathrm { C V a R } } ( Q ; P ^ { \mathrm { t a r } } ) } { \delta Q } } = \phi ^ { * } - 2 \lambda \Delta C ( Q ; P ^ { \mathrm { t a r } } ) h ^ { \mathrm { V a R } _ { \alpha } ^ { Q , g } } = \Phi _ { Q } ^ { \mathrm { V a R } _ { \alpha } ^ { Q , g } } .
$$

## 3.2 Velocity field of the CVaR-penalized Wasserstein gradient flow

The velocity field of the Wasserstein gradient flow of a loss functional $\mathcal { F }$ is defined via the variational derivative of $\mathcal { F }$ . However, by part 3 of Theorem 3.7, the variational derivative of ${ \mathcal { F } } ^ { \operatorname { C V a R } } ( Q ; P ^ { \mathrm { t a r } } )$ does not exist unless $T ( Q )$ is a singleton set or $\Delta C ( Q ; P ^ { \mathrm { t a r } } ) =$ 0. Hence, one may use a variational subgradient of ${ \mathcal { F } } ^ { \operatorname { C V a R } } ( Q ; P ^ { \mathrm { t a r } } )$ as the potential function to define the velocity field of the CVaR-penalized Wasserstein gradient flow. The choice of this subgradient is a genuine degree of freedom in the design of the CVaR-penalized Wasserstein gradient flow. We recall from part 2 of Theorem 3.7 that for all $y \in T ( Q )$ ， $\Phi _ { Q } ^ { y } = \phi ^ { * } - 2 \lambda \Delta C ( Q ; P ^ { \mathrm { t a r } } ) h ^ { y }$ is a variational subgradient of ${ \mathcal { F } } ^ { \operatorname { C V a R } } ( Q ; P ^ { \mathrm { t a r } } )$ . Choosing the family of functions $\{ \Phi _ { Q } ^ { y } ; y \in T ( Q ) \}$ as the potential functions to define the velocity field of the CVaR-penalized Wasserstein gradient flow, we obtain the following result.

Theorem 3.8 (The velocity field of $\mathcal { F } ^ { \mathrm { C V a R } } )$ . Assume that g is Lipschitz continuous, $g \in C ^ { 1 }$ for $g ( x ) \neq 0$ , and Assumption 2.1 holds. For each $y \in T ( Q )$ , choosing the potential function $\Phi _ { Q } ^ { y }$ induces the velocity field $v _ { Q } ^ { y } = - \nabla _ { x } \Phi _ { Q } ^ { y }$ , and we have

$$
\begin{array} { r } { v _ { Q } ^ { y } ( x ) = \left\{ \begin{array} { l l } { \underbrace { - \nabla _ { x } \phi ^ { * } ( x ) } _ { f r o m ~ D _ { \mathrm { K L } } ^ { L } ( Q \| P ^ { \mathrm { t a r } } ) } + \underbrace { \frac { 2 \lambda } { 1 - \alpha } \Delta C ( Q ; P ^ { \mathrm { t a r } } ) \nabla _ { x } g ( x ) } _ { f r o m ~ C V a R - p e n a l i z a t i o n } , } & { g ( x ) > y , } \\ { \underbrace { - \nabla _ { x } \phi ^ { * } ( x ) } _ { f r o m ~ D _ { \mathrm { K L } } ^ { L } ( Q \| P ^ { \mathrm { t a r } } ) } , } & { g ( x ) \le y . } \end{array} \right. } \end{array}\tag{3.36}
$$

$I f T ( Q )$ is a singleton set, the velocity field degenerates to $v _ { Q } ^ { \mathrm { V a R } _ { \alpha } ^ { Q , g } }$

Proof. The result readily follows from taking the gradient of $\Phi _ { Q } ^ { y } ( x )$ with respect to x for $g ( x ) \ > \ y \ \geq \ 0$ . Since $( g ( x ) - y ) ^ { + }$ is not diferentiable at $g ( x ) \ = \ y$ , the velocity field is not defined at $g ( x ) = y$ . One can adopt the convention that $v ^ { y } ( x ) \in \{ - \nabla _ { x } \phi ^ { * } ( x ) +$ $\begin{array} { r } { s \left( \frac { 2 \lambda } { ( 1 - \alpha ) } \Delta C ( Q ; P ^ { \mathrm { t a r } } ) \nabla _ { x } g ( x ) \right) ; s \in [ 0 , 1 ] \} } \end{array}$ for $g ( x ) = y$ . In our work, we define $v ^ { y } ( x ) =$ $- \nabla _ { x } \phi ^ { * } ( x )$ for $g ( x ) = y ,$ , yielding (3.36).

Corollary 3.9 (Boundedness of the velocity field). Assume that g is Lipschitz continuous, $g \in C ^ { 1 } f o r g ( x ) \neq 0$ , and Assumption 2.1 holds. For every $y \in T ( Q )$ , the velocity field $v _ { Q } ^ { y }$ in (3.36) satisfies, for almost every $\boldsymbol { x } \in \mathbb { R } ^ { d }$ we have

$$
\| v _ { Q } ^ { y } ( x ) \| \leq L + \frac { 2 \lambda } { 1 - \alpha } \mathrm { L i p } ( g ) \big | \Delta C ( Q ; P ^ { \mathrm { t a r } } ) \big | .\tag{3.37}
$$

Moreover, the CVaR discrepancy is controlled by the loss ${ \mathcal { F } } ^ { \operatorname { C V a R } } ( Q ; P ^ { \mathrm { t a r } } )$ , so we have

$$
\left\| v _ { Q } ^ { y } ( x ) \right\| \le L + \frac { 2 \operatorname { L i p } ( g ) } { 1 - \alpha } \sqrt { \lambda \mathcal { F } ^ { \mathrm { C V a R } } ( Q ; P ^ { \mathrm { t a r } } ) } ,\tag{3.38}
$$

for almost every $x ~ \in ~ \mathbb { R } ^ { d }$ . In particular, the velocity field is bounded, and the CVaRpenalization component vanishes as $\mathcal { F } ^ { \mathrm { C V a R } } ( Q ; P ^ { \mathrm { t a r } } ) \to 0$ , i.e. as $Q  P ^ { \mathrm { t a r } }$

Proof. By (3.36), for almost every x, we have

$$
\bigl \| v _ { Q } ^ { y } ( x ) \bigr \| \leq \bigl \| \nabla _ { x } \phi ^ { * } ( x ) \bigr \| + \frac { 2 \lambda } { 1 - \alpha } \bigl | \Delta C ( Q ; P ^ { \mathrm { t a r } } ) \bigr | \bigl \| \nabla _ { x } g ( x ) \bigr \| .
$$

Since $\phi ^ { * } \in \Gamma _ { L }$ is L-Lipschitz, $\| \nabla _ { x } \phi ^ { * } \| \leq L$ almost everywhere; since $g$ is $\operatorname { L i p } ( g ) - \operatorname { I }$ Lipschitz, $\| \nabla _ { x } g \| \leq \mathrm { L i p } ( g )$ . This proves (3.37).

Since $D _ { \mathrm { K L } } ^ { L } ( Q \Vert P ^ { \mathrm { t a r } } ) \geq 0$ we have

$$
\lambda \left( \Delta C ( Q ; P ^ { \mathrm { t a r } } ) \right) ^ { 2 } \leq D _ { \mathrm { K L } } ^ { L } ( Q \| P ^ { \mathrm { t a r } } ) + \lambda \left( \Delta C ( Q ; P ^ { \mathrm { t a r } } ) \right) ^ { 2 } = \mathcal { F } ^ { \mathrm { C V a R } } ( Q ; P ^ { \mathrm { t a r } } ) .
$$

That is $| \Delta C ( Q ; P ^ { \mathrm { t a r } } ) | \leq \sqrt { \mathcal { F } ^ { \mathrm { C V a R } } ( Q ; P ^ { \mathrm { t a r } } ) / \lambda } ;$ this combined with (3.37), proves (3.38).

Remark 3.10 (Bounded but non-Lipschitz continuous). Although Theorem 3.9 shows that the velocity field $v _ { Q } ^ { y }$ in (3.36) is bounded, it is not Lipschitz continuous in $x ,$ for two reasons. First, $v _ { Q } ^ { y }$ has a jump of magnitude $\begin{array} { r l } {  { \frac { 2 \lambda } { 1 - \alpha } | \Delta C ( Q ; P ^ { \mathrm { t a r } } ) | \| \nabla _ { x } g \| } } \end{array}$ at the threshold set $\{ x : g ( x ) = y \}$ . Second, $\nabla _ { x } \phi ^ { * }$ is bounded due to $\phi ^ { * }$ being Lipschitz continuous, but $\nabla _ { x } \phi ^ { * }$ is not necessarily continuous.

The bounded yet discontinuous velocity field $v _ { Q } ^ { y }$ in (3.36), distinguishes the CVaR-penalized flow from generative models built on Lipschitz transport maps: the latter preserve the tail behavior of the light-tailed source distribution $[ { \mathcal { Q } } { \mathcal { Y } } ] _ { i }$ , whereas a non-Lipschitz transport map is required to transport the pre-trained distribution $P ^ { \mathrm { p r e } }$ toward the heavier-tailed target distribution $P ^ { \mathrm { t a r } }$ . We do not claim here a rigorous quantitative change of the tail behavior; the resulting gain in tail accuracy is demonstrated empirically in Section 5.

While many risk functions $g$ could be considered in the CVaR-penalized Wasserstein gradient flows, throughout this paper we adopt the natural choice of the radial risk function $g ( x ) = \| x \|$ . The radial risk function $g ( x ) = \| x \|$ is Lipschitz continuous, and $g \in C ^ { 1 }$ for $x \neq 0$ , satisfying the assumptions for $g$ in Theorem 3.8. As discussed below, this choice yields a velocity field with several favorable properties for learning heavy-tailed targets. The following corollary presents the resulting velocity field for such a choice of $g .$

Corollary 3.11. Let Assumption 2.1 holds and let $g ( x ) = \| x \|$ , and $\mathbb { E } _ { Q } [ g ] < \infty$ . For each $y \in T ( Q )$ , choosing the potential function $\Phi _ { Q } ^ { y }$ induces the velocity field $v _ { Q } ^ { y } = - \nabla _ { x } \Phi _ { Q } ^ { y }$ , and

we have

$$
\begin{array} { r } { v _ { Q } ^ { y } ( x ) = \left\{ \begin{array} { l l } { \underbrace { - \nabla _ { x } \phi ^ { * } ( x ) } _ { f r o m ~ D _ { \mathrm { K L } } ^ { L } ( Q \| P ^ { \mathrm { t a r } } ) } + \underbrace { \frac { 2 \lambda } { 1 - \alpha } \Delta C ( Q ; P ^ { \mathrm { t a r } } ) \frac { x } { \| x \| } } _ { f r o m ~ C V a R - p e n a l i z a t i o n } , } & { \| x \| > y , } \\ { \underbrace { - \nabla _ { x } \phi ^ { * } ( x ) } _ { f r o m ~ D _ { \mathrm { K L } } ^ { L } ( Q \| P ^ { \mathrm { t a r } } ) } , } & { \| x \| \le y . } \end{array} \right. } \end{array}\tag{3.39}
$$

Proof. Immediate from Theorem 3.8 with $g ( x ) = \| x \|$ and $\nabla _ { x } \| x \| = x / \| x \|$ for $x \neq 0$

The velocity contribution from CVaR-penalization The CVaR-penalized Wasserstein gradient flow is given by

$$
\partial _ { t } Q _ { t } + \nabla \cdot ( Q _ { t } v _ { Q _ { t } } ^ { y } ) = 0 , \quad Q _ { 0 } = P ^ { \mathrm { p r e } } ,\tag{3.40}
$$

where $Q _ { t }$ denotes the distribution evolving at time t, and the velocity field $v _ { Q _ { t } } ^ { y }$ is given by (3.39). Consistent with our fine-tuning framework, the flow is initialized at $Q _ { 0 } = P ^ { \mathrm { p r e } }$ In line with our fine-tuning goal, the CVaR-penalization results in an additional velocity component, which we denote by $\hat { v } _ { Q } ^ { y } . \ \hat { v } _ { Q } ^ { y } ( x )$ is activated only beyond the threshold $\| x \| > y$ for $y \in T ( Q )$ , and has magnitude $\frac { 2 \lambda | \Delta C ( Q ; P ^ { \mathrm { t a r } } ) | } { ( 1 - \alpha ) }$ . Thus the choice $y \in T ( Q )$ determines the region on which the velocity contribution from CVaR-penalization will be activated, not its magnitude. Two selections are distinguished: the outer endpoint $y = \overline { { \mathrm { V a R } } } _ { \alpha } ^ { Q , g }$ and the inner endpoint $y = \mathrm { V a R } _ { \alpha } ^ { Q , g }$ result in the smallest and the largest activation regions for the velocity contribution from CVaR-penalization.

Furthermore, the magnitude of $\hat { v } _ { Q } ^ { y }$ is strictly nonzero whenever $| \Delta C ( Q ; P ^ { \mathrm { t a r } } ) | > 0$ , regardless of the availability of samples in the tail region. That is, the CVaR-penalization contributes a velocity component that is guaranteed to remain non-vanishing in tail regions, where the velocity contribution from $D _ { \mathrm { K I } } ^ { L }$ may vanish due to sample scarcity. Moreover, since $\hat { v } _ { Q } ^ { y } \propto \Delta C ( Q ; P ^ { \mathrm { t a r } } )$ , this results in a velocity field that self-regulates, stronger when the tail is inadequately captured and naturally tapering of as the generated distribution converges to the target distribution. In essence, the CVaR-penalization acts as a fine-tuning component that contributes an additional velocity component to the Lipschitz-regularized Wasserstein gradient flow, resolving the premature vanishing velocity issue.

## 4 CVaR-penalized Generative Particle Algorithm

The CVaR-penalized Wasserstein gradient flows (3.40) can be viewed as a family of transport-based variational PDEs. This family of PDEs (3.40) motivates a neural particle algorithm, analogous to the Generative Particle Algorithm (GPA) developed in [18], but with an additional CVaR penalty. We refer to such algorithms, built on the development of GPA in [18], as the CVaR-penalized Generative Particle Algorithms, or CVaR-GPA, outlined in Algorithm 2. CVaR-GPA is built by discretizing (3.40) in time and replacing the relevant probability measures by their empirical counterparts.

Let the target distribution $P ^ { \mathrm { t a r } }$ be available through its i.i.d. samples $\{ X ^ { ( j ) } \} _ { j = 1 } ^ { N }$ . At the $k ^ { \mathrm { t h } }$ iteration, we denote by $\widehat { Q } _ { k }$ the empirical measure of the generated particles $\{ Y _ { k } ^ { ( i ) } \} _ { i = 1 } ^ { M }$ We consider the empirical measures $\begin{array} { r } { \widehat { P ^ { \mathrm { t a r } } } = \frac { 1 } { N } \sum _ { j = 1 } ^ { N } \delta _ { X ^ { ( j ) } } } \end{array}$ and $\begin{array} { r } { \widehat { P ^ { \mathrm { p r e } } } = \frac { 1 } { M } \sum _ { i = 1 } ^ { M } \delta _ { Y _ { 0 } ^ { ( i ) } } } \end{array}$ where $\{ Y _ { 0 } ^ { ( i ) } \} _ { i = 1 } ^ { M }$ are samples of the pre-trained distribution $P ^ { \mathrm { p r e } }$ . In contrast to other generative model frameworks such as CNFs, where the learning is typically initialized with a simple class of distributions, such as Gaussian, GPA allows us to initialize the learning with any pre-trained model $P ^ { \mathrm { p r e } }$

Activation region of the velocity contribution from CVaR-penalization The choice of the activation region of the velocity contribution from CVaR-penalization is determined by the choice of $y \in T ( Q )$ , and this is an algorithm design choice. The two endpoints of the $T ( \widehat { Q } _ { k } ) \ \mathrm { V a R } _ { \alpha } ^ { \widehat { Q } _ { k } , g }$ and $\overline { { \mathrm { V a R } } } _ { \alpha } ^ { \widehat { Q } _ { k } , g }$ result in the largest and the smallest activation regions for the velocity contribution from CVaR-penalization, respectively. Furthermore, we can compute the two endpoints of $T ( \widehat { Q } _ { k } )$ exactly via

$$
\mathrm { V a R } _ { \alpha } ^ { \widehat { Q } _ { k } , g } = g _ { ( \lceil \alpha M \rceil ) } , \qquad \overline { { \mathrm { V a R } } } _ { \alpha } ^ { \widehat { Q } _ { k } , g } = g _ { ( \lfloor \alpha M \rfloor + 1 ) } ,\tag{4.41}
$$

where $\{ g _ { ( i ) } \} _ { i = 1 } ^ { M }$ are the order statistics of the risk values. Thus $\mathrm { V a R } _ { \alpha } ^ { \widehat { Q } _ { k } , g } , \overline { { \mathrm { V a R } } } _ { \alpha } ^ { \widehat { Q } _ { k } , g }$ can easily be computed via a simple sorting algorithm, making them a natural choice for designing CVaR-GPA. When αM $\notin \mathbb { Z }$ we obtain the unique empirical quantile; when α $M \in \mathbb { Z }$ $\mathrm { V a R } _ { \alpha } ^ { \widehat { Q } _ { k } , g } \leq \overline { { \mathrm { V a R } } } _ { \alpha } ^ { \widehat { Q } _ { k } , g }$ , with strict inequality precisely when $g _ { ( \alpha M ) } < g _ { ( \alpha M + 1 ) }$ , i.e. when the two adjacent order statistics are not tied.

The update scheme of CVaR-GPA Because $\widehat { Q } _ { k }$ evolves across iterations, at each step the function $\phi _ { k } ^ { * }$ is re-obtained by solving the variational problem (2.6) over an approximation of the function space $\Gamma _ { L }$ . Thus, the $k ^ { \mathrm { t h } }$ iteration of the update scheme is

$$
\begin{array} { r l } & { Y _ { k + 1 } ^ { ( i ) } = Y _ { k } ^ { ( i ) } + v _ { k } ^ { ( i ) } \Delta t , \quad v _ { k } ^ { ( i ) } : = - \nabla _ { x } \phi _ { k } ^ { * } ( Y _ { k } ^ { ( i ) } ) + \lambda b _ { k } ^ { ( i ) } , \quad Y _ { 0 } ^ { ( i ) } \sim P ^ { \mathrm { p r e } } , \quad i = 1 , \dots , M , } \\ & { \phi _ { k } ^ { * } = \arg \underset { \phi \in \Gamma _ { k } ^ { \mathrm { N R } } } { \mathrm { m a x } } \left\{ \frac { 1 } { M } \sum _ { i = 1 } ^ { M } \phi ( Y _ { k } ^ { ( i ) } ) - \log \left( \frac { 1 } { N } \sum _ { j = 1 } ^ { N } e ^ { \phi ( X ^ { ( j ) } ) } \right) \right\} } \\ & { \theta _ { k } ^ { ( i ) } = \frac { 2 \Delta _ { k } } { ( 1 - \alpha ) } \frac { Y _ { k } ^ { ( i ) } } { \| Y _ { k } ^ { ( i ) } \| } 1 _ { \{ \| Y _ { k } ^ { ( i ) } \| > y _ { k } \} } , \quad y _ { k } \in \{ \mathrm { V a R } _ { \alpha } ^ { \hat { Q } _ { k } , g } , \overline { { \mathrm { V a R } } } _ { \alpha } ^ { \hat { Q } _ { k } , g } \} } \\ & { \Delta _ { k } = \mathrm { C V a R } _ { \alpha } ^ { \widetilde { P } _ { k } ^ { \mathrm { a r } } , g } - \mathrm { C V a R } _ { \alpha } ^ { \hat { Q } _ { k } , g } , } \end{array}\tag{4.42}
$$

where $\Gamma _ { L } ^ { \mathrm { N N } }$ is a neural network approximation of the function space $\Gamma _ { L }$ of L-Lipschitz functions and $b _ { k } ^ { ( i ) }$ is the velocity component induced by the CVaR-penalization in the $k ^ { \mathrm { t h } }$ iteration of the algorithm. The necessary tail statistics for the $k ^ { \mathrm { t h } }$ update, namely $\mathrm { V a R } _ { \alpha } ^ { \widehat { Q } _ { k } , g } , \widehat { \mathrm { V a R } } _ { \alpha } ^ { \widehat { Q } _ { k } , g } , \mathrm { C V a R } _ { \alpha } ^ { \widehat { P } ^ { \mathrm { t a r } } , g }$ , and $\mathrm { C V a R } _ { \alpha } ^ { \widehat { Q } _ { k } , g }$ are computed via Algorithm 1.

There are two crucial properties of the components utilized in designing the loss functional $\mathcal { F } ^ { \mathrm { C V a R } }$ that enable us to develop our numerical algorithm.

1. The variational representation (2.6) is used to approximate its optimizer $\phi ^ { * } \colon$ the function space $\Gamma _ { L }$ of L-Lipschitz functions is approximated by a class of neural networks $\Gamma _ { L } ^ { \mathrm { N N } }$ with spectral normalization [28], and $\phi _ { k } ^ { * }$ is obtained at each iteration by solving the finite-dimensional maximization problem (4.42) over the network parameters. The velocity contribution $- \nabla _ { x } \phi _ { k } ^ { * } ( Y _ { k } ^ { ( i ) } )$ from the Lipschitz-regularized KL divergence is then evaluated by automatic diferentiation of the network at the particle positions.

2. In contrast to the function $\phi ^ { * }$ , whose approximation requires neural networks, the tail statistics $\mathrm { V a R } _ { \alpha } ^ { \widehat { Q } _ { k } , g } , \overline { { \mathrm { V a R } } } _ { \alpha } ^ { \widehat { Q } _ { k } , g }$ and $\mathrm { C V a R } _ { \alpha } ^ { \widehat { Q } _ { k } , g }$ for the velocity field induced by the CVaR-penalization can be directly computed: $\mathrm { V a R } _ { \alpha } ^ { \widehat { Q } _ { k } , g }$ and $\overline { { \mathrm { V a R } } } _ { \alpha } ^ { \widehat { Q } _ { k } , g }$ can be exactly computed by sorting the samples, and $\mathrm { C V a R } _ { \alpha } ^ { \widehat { Q } _ { k } , g }$ can be computed using the empirical version of the Rockafellar-Uryasev formula

$$
\mathrm { C V a R } _ { \alpha } ^ { \widehat { Q } _ { k } , g } = \mathrm { V a R } _ { \alpha } ^ { \widehat { Q } _ { k } , g } + \frac { 1 } { ( 1 - \alpha ) M } \sum _ { i = 1 } ^ { M } ( g ( Y _ { k } ^ { ( i ) } ) - \mathrm { V a R } _ { \alpha } ^ { \widehat { Q } _ { k } , g } ) ^ { + }\tag{4.43}
$$

which results in more accurate and eficient computations.

The stopping criterion of CVaR-GPA For a given $y \in T ( Q )$ , the CVaR-penalized gradient flow (3.40) terminates when its velocity field dies out, i.e., when $v _ { Q _ { T ^ { * } } } ^ { y } = 0$ for some

$T ^ { * } > 0$ . Hence, one can use the empirical estimate of the velocity field, and by extension the empirical estimate of the kinetic energy (at the $k ^ { \mathrm { t h } }$ iteration) given by

$$
\mathcal { K } _ { k } : = \frac { 1 } { 2 M } \sum _ { i = 1 } ^ { M } \| v _ { k } ^ { ( i ) } \| _ { 2 } ^ { 2 } ,
$$

to define a stopping criterion for CVaR-GPA. We terminate the algorithm when ${ { \kappa } _ { k } } < \epsilon$ for a given threshold ϵ, resulting in an algorithm whose architecture depth is implicitly determined by the target distribution $P ^ { \mathrm { t a r } }$

Algorithm 1 Empirical tail statistics   
Require: samples $\{ Z _ { i } \} _ { i = 1 } ^ { m } , g ( \cdot ) = \| \cdot \| \colon$ the risk function, α: parameter of the quantile   
1: $g _ { i }  g ( Z _ { i } )$ ; sort ascending $g _ { ( 1 ) } \leq \cdot \cdot \cdot \leq g _ { ( m ) }$   
2: VaR $ g _ { ( \lceil \alpha m \rceil ) } , \quad \overline { { \mathrm { V a R } } }  g _ { ( \lfloor \alpha m \rfloor + 1 ) }$   
m   
3: ${ \mathrm { C V a R } } \gets \mathrm { V a R } + \frac { 1 } { \left( 1 - \alpha \right) m } \sum _ { i = \lceil \alpha m \rceil + 1 } ^ { \cdots } \left( g _ { ( i ) } - \mathrm { V a R } \right)$ ▷ from (4.43)   
4: return (VaR, VaR, CVaR)

Tail agnosticism The only target-dependent inputs utilized in Algorithm 2 are the target samples $\{ X ^ { ( j ) } \}$ feeding the Lipschitz-regularized KL divergence and the scalar value $\mathrm { C V a R } _ { \alpha } ^ { P ^ { \mathrm { t a r } } , g }$ computed using the target samples $\{ X ^ { ( j ) } \}$ . This contrasts with tail-specialized generators that require Hill/peaks-over-threshold estimation of the tail exponent.

## 5 Numerical experiments

We evaluate the performance of CVaR-GPA across a range of synthetic and real-world benchmarks to characterize its ability to learn complex tail structures. We first demonstrate the tail-agnostic capability of CVaR-GPA by showing that the same fine-tuning procedure improves distributions with diverse tail indices without prior knowledge of the target tail behavior. Next, we assess its scalability and robustness on anisotropic multivariate distributions, including high-dimensional real-world data and synthetic targets with heterogeneous marginal tail behaviors.

While CVaR-GPA can be used to fine-tune any pre-trained model, in this work we choose Lip-KL-GPA [9, 18] as the pre-trained model, based on both its theoretical properties and its demonstrated empirical performance in learning heavy-tailed distributions [9].

```latex
Algorithm 2 CVaR-GPA
Require: $g ( \cdot ) = \| \cdot \| \cdot$ the risk function, α: parameter of the quantile, λ: weight parameter,
$L \colon$ Lipschitz constant, $\Delta t \colon$ step size, M: number of pre-trained particles, N: number
of target particles, ϵ: a chosen error threshold.
Require: $W = \{ W ^ { l } \} _ { l = 1 } ^ { D }$ : parameters for the neural network $\phi ( \because W ) : \mathbb { R } ^ { d } \to \mathbb { R } ,$ D: depth
of the NN, $N _ { \phi } \colon$ number of updates for the NN.
Result: Fine-tuned particles $Y _ { K } = \{ Y _ { K } ^ { ( i ) } \} _ { i = 1 } ^ { M }$
1: Sample $\{ X ^ { ( j ) } \} _ { j = 1 } ^ { N } \sim P ^ { \mathrm { t a r } }$ , a batch of samples from the target distribution.
2: Sample $\{ Y _ { 0 } ^ { ( i ) } \} _ { i = 1 } ^ { M } \sim P ^ { \mathrm { p r e } }$ , a batch of samples from the pre-trained distribution.
3: Compute the target tail statistic $\mathrm { C V a R } _ { \alpha } ^ { \widehat { P ^ { \mathrm { t a r } } } , g }$ via Algorithm 1 using $\{ X ^ { ( j ) } \} _ { j = 1 } ^ { N } \sim P ^ { \mathrm { t a r } }$
4: $k  0 , \ K  2 \epsilon$
5: while $\kappa > \epsilon$ do
6: $\begin{array} { r } { \phi _ { k } ^ { * }  \arg \operatorname* { m a x } _ { \phi \in \Gamma _ { r } ^ { \mathrm { N N } } } \Big \{ \frac { 1 } { M } \sum _ { i } \phi \big ( Y _ { k } ^ { ( i ) } \big ) - \log \big ( \frac { 1 } { N } \sum _ { j } e ^ { \phi ( X ^ { ( j ) } ) } \big ) \Big \} } \end{array}$ ▷ $N _ { \phi }$ steps
7: $( \mathrm { V a R } _ { \alpha } ^ { \widehat { Q } , g } , \overline { { \mathrm { V a R } } } _ { \alpha } ^ { \widehat { Q } , g } , \mathrm { C V a R } _ { \alpha } ^ { \widehat { Q } , g } )$ via Algorithm 1 using $\{ Y _ { k } ^ { ( i ) } \} _ { i = 1 } ^ { M }$
8: $y _ { k } \gets \mathrm { V a R } _ { \alpha } ^ { \widehat { Q } , g } \ \mathrm { o r } \ \overline { { \mathrm { V a R } } } _ { \alpha } ^ { \widehat { Q } , g }$
9: $\Delta _ { k } \gets \mathrm { C V a R } _ { \alpha } ^ { \widehat { P ^ { \mathrm { t a r } } } , g } - \mathrm { C V a R } _ { \alpha } ^ { \widehat { Q } , g }$
10: for $i = 1 , \dots , M$ do
11: $v _ { k } ^ { ( i ) } \gets - \nabla _ { x } \phi _ { k } ^ { * } ( Y _ { k } ^ { ( i ) } ) + \frac { 2 \lambda } { 1 - \alpha } \Delta _ { k } \frac { Y _ { k } ^ { ( i ) } } { \| Y _ { k } ^ { ( i ) } \| } \mathbb { 1 } \{ \| Y _ { k } ^ { ( i ) } \| > y _ { k } \}$
12: end for
13: $\begin{array} { r } { Y _ { k + 1 } ^ { ( i ) }  Y _ { k } ^ { ( i ) } + \Delta t v _ { k } ^ { ( i ) } ; \quad \mathcal { K }  \frac { 1 } { 2 M } \sum _ { i } \| v _ { k } ^ { ( i ) } \| ^ { 2 } } \end{array}$
14: $k \gets k + 1$
15: end while
16: return $\{ Y _ { K } ^ { ( i ) } \} _ { i = 1 } ^ { M }$
```

Error metrics for evaluating the accuracy of the learned target distributions Standard error metrics, which weigh all the regions of the domain equally, alone are not suficient for evaluating the learned heavy-tailed distributions. The tail region contributes negligible mass to such an error metric. Thus, a learned distribution may have a low error value even when it is underestimating the tail region or missing extreme events entirely. This necessitates tail-sensitive metrics. We therefore use both a standard metric and a tail-sensitive metric to evaluate the learned distributions.

For N empirical samples $\{ Z _ { Q } ^ { ( i ) } \} _ { i = 1 } ^ { N }$ drawn from a distribution $Q \in \mathcal { P } ( \mathbb { R } ^ { d } )$ , we denote the empirical Cumulative Distribution Function (CDF) and the Complementary CDF (CCDF) of $\{ g ( Z _ { Q } ^ { ( i ) } ) \} _ { i = 1 } ^ { N }$ <sub>1</sub> by $\widehat { \Psi } ^ { Q , g }$ and $\bar { \Psi } ^ { Q , g }$ . We fix $g ( x ) = \| x \|$ and define

$$
\widehat \Psi ^ { Q , g } ( r ) = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \mathbb { 1 } _ { \{ \| Z _ { Q } ^ { ( i ) } \| \leq r \} } , \qquad \bar { \Psi } ^ { Q , g } ( r ) = 1 - \widehat \Psi ^ { Q , g } ( r ) .\tag{5.44}
$$

We use the following metrics.

1. Global $L ^ { 1 }$ error: The $L ^ { 1 }$ distance between the CCDFs of the learned distribution $Q$ and the target distribution $P ^ { \mathrm { t a r } }$ over the domain $\operatorname { s u p p } ( Q ) \cup \operatorname { s u p p } ( P ^ { \operatorname { t a r } } )$

$$
\mathcal { E } _ { L ^ { 1 } } ( Q ; P ^ { \mathrm { t a r } } ) = \int _ { 0 } ^ { r _ { \mathrm { m a x } } } \big | \bar { \Psi } ^ { Q , g } ( r ) - \bar { \Psi } ^ { P ^ { \mathrm { t a r } } , g } ( r ) \big | d r .\tag{5.45}
$$

where $r _ { \mathrm { m a x } } = \operatorname* { m a x } \{ \operatorname* { m a x } _ { i } \| Z _ { Q } ^ { ( i ) } \|$ , max $_ j \parallel Z _ { P \mathrm { t a r } } ^ { ( j ) } \parallel \}$ and the integral is evaluated by the trapezoidal rule on a uniform 2000-point grid of $[ 0 , { r _ { \mathrm { m a x } } } ]$

2. Tail error: In the tail region, the absolute diferences in CCDF are not informative because $\bar { \Psi } ^ { Q , g }$ is small. Hence, we use the log-CCDF discrepancy

$$
\mathcal { E } _ { \mathrm { t a i l } } ( Q ; P ^ { \mathrm { t a r } } ) = \frac { 1 } { | T _ { \mathrm { t a i l } } | } \sum _ { t \in T _ { \mathrm { t a i l } } } \big | \log ( \bar { \Psi } ^ { Q , g } ( t ) + \varepsilon ) - \log ( \bar { \Psi } ^ { P ^ { \mathrm { t a r } } , g } ( t ) + \varepsilon ) \big | ,\tag{5.46}
$$

where $T _ { \mathrm { t a i l } } : = \{ t : t = ( \widehat { \Psi } ^ { P ^ { \mathrm { t a r } } , g } ) ^ { - 1 } ( p )$ for 20 equi-spaced points $p \in [ 0 . 9 5 , 0 . 9 9 9 ] \}$ and $\varepsilon = 1 0 ^ { - 1 0 }$ is added for numerical stability.

Throughout this section, we fix the hyperparameters as $( \lambda , \alpha ) \ = \ ( 0 . 0 2 , \ 0 . 9 9 9 )$ , and choose the outer radius $\overline { { \mathrm { V a R } } }$ for the activation region of the CVaR velocity component.

## 5.1 Learning tail behavior without prior tail knowledge

We first provide a calibrated example demonstrating the tail-agnostic learning capability of CVaR-GPA. Specifically, we consider two-dimensional isotropic Student-t distributions with varying tail indices, $\nu \in \{ 1 , 1 . 2 , 1 . 5 , 1 . 8 \}$ , where decreasing ν corresponds to increasingly heavy polynomial tails. This family provides a controlled calibration of tail decay rates, as a smaller ν corresponds to heavier polynomial tail decay, with $\nu = 1$ recovering the Cauchy distribution. Although our theoretical analysis assumes $\mathbb { E } _ { P ^ { \mathrm { t a r } } } [ g ] < \infty$ , this condition is automatically satisfied for the empirical target measure $\widehat { P ^ { \mathrm { t a r } } }$ , which has a finite support, allowing us to evaluate CVaR-GPA on the full range of considered tail indices.

CVaR-GPA is applied to all target distributions using the same hyperparameters, without any target-specific tuning or prior knowledge of the tail index ν. As shown in Figure 3, CVaR-GPA consistently reduces both the global $L ^ { 1 }$ error $\mathcal { E } _ { L ^ { 1 } }$ and the tail error $\mathcal { E } _ { \mathrm { t a i l } }$ of the pre-trained model across all tail indices. This demonstrates that CVaR-GPA can adaptively improve tail learning across a range of tail behaviors without requiring prior specification of the target tail decay rate.

![](images/b4592ce8b1295fb2e04f26363ac910ca9b9b69753114233792cdcfb2b322759c.jpg)  
Figure 3: Comparison of the pre-trained model (Lip-KL-GPA) vs. the fine-tuned model (CVaR-GPA) on a 2-dimensional isotropic Student-t distribution via the global $L ^ { 1 }$ error and the tail error (both shown on a log scale) for the joint distribution: CVaR-GPA finetunes the Lip-KL-GPA and improves the accuracy of learning the 2-dimensional isotropic Student-t distribution for each tail index $\nu \in \{ 1 , 1 . 2 , 1 . 5 , 1 . 8 \}$ with no modifications curated to the tail index ν.

The same no-prior-tail-knowledge setting is maintained throughout the subsequent experiments, where we further investigate the ability of CVaR-GPA to learn multivariate distributions with anisotropic tails.

## 5.2 Learning Multivariate Anisotropic Heavy-Tailed Distributions

Anisotropic multivariate distributions introduce two important challenges for tail-aware generative modeling: scalability with increasing dimension and robustness to heterogeneous tail behaviors across dimensions. We investigate these challenges through a series of examples, including high-dimensional real-world distributions and synthetic targets with varying degrees of tail heterogeneity. Across these experiments, CVaR-GPA improves tail learning without requiring prior knowledge of the target tail behavior.

Scalability to high-dimensional anisotropic distributions. We first evaluate the scalability of CVaR-GPA on the Fama-French 25 monthly portfolios [16], a real-world 25-dimensional anisotropic dataset. The marginal tail indices estimated using the Hill estimator [21] range from 2.21 to 3.18, representing a realistic heavy-tailed regime with moderate variation across dimensions. As shown in Figure 2, CVaR-GPA consistently decreases both the global $L ^ { 1 }$ error and the tail error across all dimensions compared to the pre-trained model. These results demonstrate the scalability of CVaR-GPA to highdimensional anisotropic target distributions while efectively improving tail learning across dimensions.

Robustness to mixed heavy- and light-tailed marginals. We next examine whether CVaR-GPA remains efective when the target distribution contains both heavy- and lighttailed marginals. For this purpose, we consider Neal’s funnel distribution [29], a canonical two-dimensional benchmark with a Gaussian marginal and a heavy-tailed marginal:

$$
x \sim { \mathcal { N } } ( 0 , 9 ) ,\tag{5.47}
$$

$$
y | x \sim { \mathcal { N } } ( 0 , e ^ { x } ) .\tag{5.48}
$$

As shown in Figure 4, CVaR-GPA improves both marginals by reducing the global $L ^ { 1 }$ error and the tail error relative to the pre-trained model. This demonstrates that the presence of both heavy- and light-tailed components alone does not prevent efective tail-aware fine-tuning. In this setting, the diference in tail behavior between the two marginals is moderate, allowing the CVaR velocity correction to remain efective across both components.

![](images/f2a0018c6016fa38b0370b16499ea45c1d398be38770ab4102bf77a286f4463e.jpg)  
Figure 4: Comparison of the pre-trained model (Lip-KL-GPA) vs. the fine-tuned model (CVaR-GPA) on Neal’s funnel distribution via the global $L ^ { 1 }$ error and the tail error (both shown on a log scale) for the marginal distributions: CVaR-GPA fine-tunes the Lip-KL-GPA and improves the accuracy of learning each of the marginal distributions.

Learning under extreme tail heterogeneity. Finally, we investigate a more challenging synthetic setting where marginal tail behaviors vary across a wide range. We consider a 5-dimensional anisotropic Student-t distribution with independent marginals and tail indices $\nu = ( 1 , 1 . 5 , 3 , 1 0 , 3 0 )$ . This construction provides a controlled evaluation of whether CVaR-GPA can distinguish heterogeneous tail behaviors across dimensions without prior knowledge of the target tail indices, ranging from extremely heavy-tailed $( \nu = 1 )$ to nearly Gaussian $( \nu = 3 0 )$ marginals.

As shown in Figure 5, CVaR-GPA consistently reduces both the global $L ^ { 1 }$ error and the tail error for the heavy-tailed marginals with $\nu = ( 1 , 1 . 5 , 3 , 1 0 )$ . For the nearly Gaussian marginal with $\nu = 3 0$ , however, both errors increase relative to the pre-trained model. These results suggest that while CVaR-GPA efectively handles a broad range of heterogeneous heavy-tailed behaviors, an extreme disparity between heavy-tailed and nearly Gaussian marginals can introduce calibration challenges. In such regimes, the global CVaR correction may become dominated by the heaviest-tailed directions, leaving the correction less calibrated for nearly Gaussian components. Addressing this regime provides an opportunity for further refinement of the current formulation. We defer such a refinement to our future work, see Section 6.

![](images/e5d958085d28536e3895df2c26e0220387485a0482b5fdce93d2c7c2bac2fbe1.jpg)  
Figure 5: Comparison of the pre-trained model (Lip-KL-GPA) vs. the fine-tuned model (CVaR-GPA) on a 5-dimensional anisotropic Student-t distribution $( \nu = ( 1 , 1 . 5 , 3 , 1 0 , 3 0 ) )$ ), evaluated via the global $L ^ { 1 }$ error and the tail error (both shown on a log scale) for each marginal distribution. CVaR-GPA actively decreases the global $L ^ { 1 }$ error and the tail error for heavy-tailed marginals (with $\nu = ( 1 , 1 . 5 , 3 , 1 0 ) )$ ), while increasing the global $L ^ { 1 }$ error and the tail error on the nearly Gaussian marginal with $\nu = 3 0$

## 6 Conclusions and discussions

We propose a tail-agnostic algorithm, CVaR-GPA, to fine-tune pre-trained models to learn multivariate heavy-tailed distributions with both anisotropic and isotropic tails. The CVaR penalization mitigates the efects of data scarcity in the tail region, remedying the premature saturation issue exhibited by pre-trained models. We use the kinetic energy of the particle system as the stopping criterion for CVaR-GPA, thus creating a fine-tuning model architecture whose depth is implicitly determined by the target data distribution, as opposed to a pre-determined feature of the model architecture. In our work, we compare the performance of the proposed algorithm, CVaR-GPA, with the Lip-KL-GPA [18] that has been shown to outperform many existing generative models, including f-GANs, OT flows, CNFs, and SGMs for learning heavy-tailed distributions [9]. The simulation results show that our proposed CVaR-GPA demonstrates even more accurate learning of heavy-tailed targets.

Future directions Although we adopted the natural choice of the radial risk function in our work, the theoretical analysis was carried out for a general risk function g, thereby opening several natural directions for future work. For instance, one can replace the radial risk function with a per-coordinate risk function that adapts the velocity contribution from

CVaR-penalization and its activation region to each marginal distribution. Such a risk function may potentially be more efective on anisotropic targets with tail indices that span a wide range that include both light-tailed and extremely heavy-tailed marginals. CVaR-GPA can also be modified by utilizing other spectral risk measures in place of CVaR to penalize the Lipschitz-regularized KL divergence. For instance, a spectral risk measure that weighs extreme regions with monotonically increasing weights may potentially capture the tail regions of extremely heavy-tailed targets better. One may also define an alternate, but related spectral risk measure to CVaR by smoothing the function $( g - y ) ^ { + } \mathrm { a t } g ( \cdot ) = y ,$ so that its first variational derivative is well-defined. We defer such extensions to our future work.

## Funding

This work was supported in part by the Air Force Ofice of Scientific Research (AFOSR grant FA9550-21-1-0354) (T.G., H.G., Z.C., M.K., L.R.-B.) and by the National Science Foundation (NSF grants DMS-2307115 and DMS-2606221 (M.K., L.R.-B.); NSF grant DMS-2606084 (Z.C.)).

## References

[1] C. Acerbi, C. Nordio, and C. Sirtori. Expected shortfall as a tool for financial risk management. arXiv preprint cond-mat/0102304, 2001.

[2] C. Acerbi and P. Simonetti. Portfolio optimization with spectral measures of risk. arXiv preprint cond-mat/0203607, 2002.

[3] H. Albrecher and S. Asmussen. Ruin probabilities and aggregrate claims distributions for shot noise cox processes. Scandinavian Actuarial Journal, 2006(2):86–110, 2006.

[4] M. Allouche, S. Girard, and E. Gobet. EV-GAN: Simulation of extreme events with ReLU neural networks. Journal of Machine Learning Research, 23(150):1–39, 2022.

[5] M. Allouche, S. Girard, and E. Gobet. Exceedgan: simulation above extreme thresholds using generative adversarial networks. Extremes, pages 1–23, 2026.

[6] S. Bhatia, A. Jain, and B. Hooi. Exgan: Adversarial generation of extreme samples. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 35, pages 6750–6758, 2021.

[7] J. Birrell, P. Dupuis, M. A. Katsoulakis, Y. Pantazis, and L. Rey-Bellet. (f,gamma)- divergences: Interpolating between f-divergences and integral probability metrics. Journal of Machine Learning Research, 23(39):1–70, 2022.

[8] S. Chaudhary, U. Dinesha, D. Kalathil, and S. Shakkottai. Risk-averse fine-tuning of large language models. Advances in Neural Information Processing Systems, 37:107003–107038, 2024.

[9] Z. Chen, H. Gu, M. A. Katsoulakis, L. Rey-Bellet, and W. Zhu. Robust generative learning with lipschitz-regularized α-divergences allows minimal assumptions on target distributions. Information and Inference: A Journal of the IMA, 14(4):iaaf028, 2025.

[10] P. Cirillo and N. N. Taleb. Tail risk of contagious diseases. Nature Physics, 16(6):606– 613, 2020.

[11] F. H. Clarke. Optimization and nonsmooth analysis. SIAM, 1990.

[12] R. Cont, M. Cucuringu, R. Xu, and C. Zhang. Tail-gan: Learning to simulate tail risk scenarios. Management Science, 72(4):2917–2936, 2026.

[13] R. De Santi, M. Vlastelica, Y.-P. Hsieh, Z. Shen, N. He, and A. Krause. Flow density control: Generative optimization beyond entropy-regularized fine-tuning. Advances in neural information processing systems, 38:11056–11088, 2026.

[14] P. Dupuis and Y. Mao. Formulation and properties of a divergence used to compare probability measures without absolute continuity. ESAIM: Control, Optimisation and Calculus of Variations, 28:10, 2022.

[15] P. Embrechts and N. Veraverbeke. Estimates for the probability of ruin with special emphasis on the possibility of large claims. Insurance: Mathematics and Economics, 1(1):55–72, 1982.

[16] E. F. Fama and K. R. French. Common risk factors in the returns on stocks and bonds. Journal of Financial Economics, 33(1):3–56, 1993.

[17] P. Grossi, H. Kunreuther, and C. C. Patel. Catastrophe modeling: a new approach to managing risk, volume 25. Springer Science & Business Media, 2005.

[18] H. Gu, P. Birmpa, Y. Pantazis, L. Rey-Bellet, and M. A. Katsoulakis. Lipschitzregularized gradient flows and generative particle algorithms for high-dimensional scarce data. SIAM Journal on Mathematics of Data Science, 6(4):1205–1235, 2024.

[19] Y. Guan, K. Balasubramanian, and S. Ma. Mirror flow matching with heavy-tailed priors for generative modeling on convex domains. In International Conference on Learning Representations, volume 2026, pages 130098–130124, 2026.

[20] T. Hickling and D. Prangle. Flexible tails for normalizing flows. arXiv preprint arXiv:2406.16971, 2024.

[21] B. M. Hill. A Simple General Approach to Inference About the Tail of a Distribution. The Annals of Statistics, 3(5):1163 – 1174, 1975.

[22] T. Huster, J. Cohen, Z. Lin, K. Chan, C. Kamhoua, N. O. Leslie, C.-Y. J. Chiang, and V. Sekar. Pareto gan: Extending the representational power of gans to heavy-tailed distributions. In International Conference on Machine Learning, pages 4523–4532. PMLR, 2021.

[23] P. Jaini, I. Kobyzev, Y. Yu, and M. Brubaker. Tails of lipschitz triangular flows. In International Conference on Machine Learning, pages 4673–4681. PMLR, 2020.

[24] R. Jordan, D. Kinderlehrer, and F. Otto. The variational formulation of the Fokker– Planck equation. SIAM Journal on Mathematical Analysis, 29(1):1–17, 1998.

[25] L. V. Kantorovich and S. Rubinshtein. On a space of totally additive functions. Vestnik of the St. Petersburg University: Mathematics, 13(7):52–59, 1958.

[26] M. Kishida. Risk-aware stability, ultimate boundedness, and positive invariance. IEEE Transactions on Automatic Control, 69(1):681–688, 2023.

[27] H. Liu, T. Zhu, N. Jia, J. He, and Z. Zheng. Learning to simulate from heavy-tailed distribution via difusion model. Available at SSRN 4975931, 2024.

[28] T. Miyato, T. Kataoka, M. Koyama, and Y. Yoshida. Spectral normalization for generative adversarial networks. arXiv preprint arXiv:1802.05957, 2018.

[29] R. M. Neal. Slice sampling. The Annals of Statistics, 31(3):705–767, 2003.

[30] F. Otto. The geometry of dissipative evolution equations: The porous medium equation. Communications in Partial Diferential Equations, 26(1-2):101–174, 2001.

[31] K. Pandey, J. Pathak, Y. Xu, S. Mandt, M. Pritchard, A. Vahdat, and M. Mardani. Heavy-tailed difusion models. arXiv preprint arXiv:2410.14171, 2024.

[32] R. T. Rockafellar and S. Uryasev. Conditional value-at-risk for general loss distributions. Journal of banking & finance, 26(7):1443–1471, 2002.

[33] Z. Wang, R. De Santi, X. Mo, M. M. Zavlanos, A. Krause, and K. H. Johansson. Eficient tail-aware generative optimization via flow model fine-tuning. arXiv preprint arXiv:2602.16796, 2026.

## Appendix A: Clarke’s generalized subgradients

Let $Q \mapsto J ( Q )$ be a locally Lipschitz continuous (with respect to the Wasserstein-1 metric $\mathcal { W } _ { 1 }$ on ${ \mathcal { P } } ( \mathbb { R } ^ { d } ) )$ real-valued functional, defined on ${ \mathcal { P } } ( \mathbb { R } ^ { d } )$ . The Clarke generalized directional derivative [11] of J at $Q$ for an admissible perturbation $\rho$ is defined by

$$
J ^ { \circ } ( Q ; \rho ) : = \operatorname* { l i m } _ { Q ^ { \prime } \to Q , \ t \downarrow 0 } \frac { J ( Q ^ { \prime } + t \rho ) - J ( Q ^ { \prime } ) } { t } .\tag{A1}
$$

The Clarke subdiferential of J at $Q$ is defined as

$$
\begin{array} { r } { \partial J ( Q ) : = \Bigl \{ \xi : ~ J ^ { \circ } ( Q ; \rho ) \geq \int \xi d \rho ~ \mathrm { f o r ~ a l l ~ a d m i s s i b l e ~ p e r t u r b a t i o n s } ~ \rho \Bigr \} , } \end{array}\tag{A2}
$$

and its elements are referred to as the first variational subgradients of J at $Q$

Convex and concave cases When J is convex, its Clarke subdiferential coincides with the classical subdiferential of convex analysis

$$
\partial J ( Q ) : = \Bigl \{ \xi : ~ J ( Q ^ { \prime } ) \geq J ( Q ) + \int \xi d ( Q ^ { \prime } - Q ) ~ \forall Q ^ { \prime } \in \mathcal { P } ( \mathbb { R } ^ { d } ) \Bigr \} .\tag{A3}
$$

When $J$ is concave, the Clarke subdiferential instead coincides with the superdiferential

$$
\partial J ( Q ) : = \Bigl \{ \xi : ~ J ( Q ^ { \prime } ) \leq J ( Q ) + \int \xi d ( Q ^ { \prime } - Q ) ~ \forall Q ^ { \prime } \in \mathcal { P } ( \mathbb { R } ^ { d } ) \Bigr \} ,\tag{A4}
$$

and its elements are correspondingly called supergradients rather than subgradients.

The risk measure $\mathrm { C V a R } _ { \alpha } ^ { Q , g }$ is an infimum of a family of functions afine in $Q { \mathrm { . } }$ by the Rockafellar-Uryasev formulation (2.11)-(2.12). Thus $\mathrm { C V a R } _ { \alpha } ^ { \cdot , g }$ is concave in $Q .$ However, in this paper we use the single term variational subgradients throughout to refer to the elements of $\partial J$ for any given functional $^ { J , }$ regardless of whether $J$ is convex, concave, or neither.

## Appendix A.1: Proof of Lemma 3.3

Lemma A.1. Let $J ( Q )$ be locally Lipschitz continuous with respect to the Wasserstein-1 metric on ${ \mathcal { P } } ( \mathbb { R } ^ { d } )$ . Assume that for every admissible perturbation $\rho$ at $Q ,$ the one-sided derivatives $D _ { \rho } ^ { \pm } J$ exist and $\int \xi d \rho$ lies in the closed interval with endpoints $D _ { \rho } ^ { - } J$ and $D _ { \rho } ^ { + } J$ Then $\xi$ is a variational subgradient of J at $Q$

Proof. Let $\rho$ be an admissible perturbation at $Q .$ . The right/left one-sided derivatives of J at $Q$ are defined by

$$
D _ { \rho } ^ { \pm } J : = \operatorname * { l i m } _ { \epsilon \to 0 ^ { \pm } } \frac { 1 } { \epsilon } \big ( J ( Q ^ { \epsilon } ) - J ( Q ) \big ) .\tag{A5}
$$

Restricting the limit superior in the definition of $J ^ { \circ } ( Q ; \rho )$ to the constant family $Q ^ { \prime } = Q$ gives

$$
J ^ { \circ } ( Q ; \rho ) ~ \geq ~ \operatorname* { l i m } _ { t \downarrow 0 } \frac { J ( Q + t \rho ) - J ( Q ) } { t } ~ = ~ D _ { \rho } ^ { + } J .
$$

Similarly, we may instead take the family $Q _ { t } ^ { \prime } : = Q - t \rho .$ which lies in ${ \mathcal { P } } ( \mathbb { R } ^ { d } )$ for all small $t > 0$ by the admissibility of $- \rho$ and converges to $Q$ in total variation as $t \downarrow 0$ . Along this family

$$
\frac { J ( Q _ { t } ^ { \prime } + t \rho ) - J ( Q _ { t } ^ { \prime } ) } { t } = \frac { J ( Q ) - J ( Q - t \rho ) } { t } = \left. \frac { J ( Q + s \rho ) - J ( Q ) } { s } \right| _ { s = - t } \ \underset { t \downarrow 0 } { \longrightarrow } \ D _ { \rho } ^ { - } J _ { \uparrow }
$$

whence $J ^ { \circ } ( Q ; \rho ) ~ \ge ~ D _ { \rho } ^ { - } J$ as well. Therefore $J ^ { \circ } ( Q ; \rho ) \geq \operatorname* { m a x } \{ D _ { \rho } ^ { + } J , D _ { \rho } ^ { - } J \} \geq \int \xi d \rho$ , and $\xi \in \partial J ( Q )$ , i.e. $\xi$ is a variational subgradient of $J ( Q )$

## Appendix B: Proof of Proposition 3.2

The following proposition proves that both $\mathrm { C V a R } _ { \alpha } ^ { Q , g }$ and ${ \mathcal { F } } ^ { \operatorname { C V a R } } ( Q ; P ^ { \mathrm { t a r } } )$ are locally Lipschitz continuous with respect to $\mathcal { W } _ { 1 }$ , which is the regularity hypothesis under which the Clarke subdiferential framework is classically defined [11].

Proposition B.1 (Local Lipschitz continuity). Assume that $g$ is $\operatorname { L i p } ( g )$ -Lipschitz continuous and Assumption 2.1 holds. Then we have the following:

1. The CVaR $Q \mapsto { \mathrm { C V a R } } _ { \alpha } ^ { Q , g }$ , is Lipschitz continuous with respect to the Wasserstein-1 metric, i.e.

$$
\left. \mathrm { C V a R } _ { \alpha } ^ { Q _ { 1 } , g } - \mathrm { C V a R } _ { \alpha } ^ { Q _ { 2 } , g } \right. \leq \frac { \mathrm { L i p } ( g ) } { 1 - \alpha } \mathcal { W } _ { 1 } ( Q _ { 1 } , Q _ { 2 } ) ,\tag{B6}
$$

for all $Q _ { 1 } , Q _ { 2 } \in \mathcal { Q } ^ { g }$

2. The functional $Q \mapsto { \mathcal { F } } ^ { \operatorname { C V a R } } ( Q ; P ^ { \operatorname { t a r } } )$ is locally Lipschitz continuous on $\mathcal { Q } ^ { g }$ with respect to $\mathcal { W } _ { 1 }$

Proof. (1) Fix $y \in \mathbb { R }$ . Since $t \mapsto t ^ { + }$ is 1-Lipschitz and $g$ is $\operatorname { L i p } ( g ) - \operatorname { I }$ ipschitz, $x \mapsto ( g ( x ) - y ) ^ { + }$ is Lip(g)-Lipschitz, uniformly in y. Thus we have

$$
\begin{array} { r l r } {  { \big \vert F _ { \alpha } ^ { Q _ { 1 } , g } ( y ) - F _ { \alpha } ^ { Q _ { 2 } , g } ( y ) \big \vert = \frac { 1 } { 1 - \alpha } \big \vert \mathbb { E } _ { Q _ { 1 } } [ ( g - y ) ^ { + } ] - \mathbb { E } _ { Q _ { 2 } } [ ( g - y ) ^ { + } ] \big \vert } } \\ & { } & { \leq \frac { 1 } { 1 - \alpha } \operatorname* { s u p } _ { \phi \in \Gamma _ { \mathrm { L i p } ( g ) } } \int \phi d ( Q _ { 1 } - Q _ { 2 } ) , \quad \quad } \end{array}
$$

$$
\begin{array} { l l } { { { \displaystyle = \frac { \mathrm { L i p } ( g ) } { 1 - \alpha } \operatorname* { s u p } _ { \psi \in \Gamma _ { 1 } } \int \psi d ( Q _ { 1 } - Q _ { 2 } ) } } } \\ { { { \displaystyle = \frac { \mathrm { L i p } ( g ) } { 1 - \alpha } \mathcal { W } _ { 1 } ( Q _ { 1 } , Q _ { 2 } ) , } } } \end{array}
$$

for every y, uniformly. Here the last equality follows from the Kantorovich-Rubinstein duality [25]

$$
\operatorname * { s u p } _ { \psi \in \Gamma _ { 1 } } \int \psi d ( Q _ { 1 } - Q _ { 2 } ) = { \mathcal W } _ { 1 } ( Q _ { 1 } , Q _ { 2 } ) .
$$

Thus it follows

$$
\begin{array} { r l } { \bigl | \mathrm { C V a R } _ { \alpha } ^ { Q _ { 1 } , g } - \mathrm { C V a R } _ { \alpha } ^ { Q _ { 2 } , g } \bigr | = \big | \underset { y \in \mathbb { R } } { \operatorname* { i n f } } F _ { \alpha } ^ { Q _ { 1 } , g } ( y ) - \underset { y \in \mathbb { R } } { \operatorname* { i n f } } F _ { \alpha } ^ { Q _ { 2 } , g } ( y ) \big | } & { } \\ { \quad } & { \leq \underset { y \in \mathbb { R } } { \operatorname* { s u p } } \big | F _ { \alpha } ^ { Q _ { 1 } , g } ( y ) - F _ { \alpha } ^ { Q _ { 2 } , g } ( y ) \big | } \\ { \quad } & { \leq \frac { \mathrm { L i p } ( g ) } { 1 - \alpha } \mathcal { W } _ { 1 } ( Q _ { 1 } , Q _ { 2 } ) , } \end{array}
$$

proving (3.16).

(2) Fix $Q _ { 0 } \in \mathcal { Q } ^ { g }$ and $R > 0$ , and let $Q _ { 1 } , Q _ { 2 } \in B _ { R } ( Q _ { 0 } )$ , so that ${ \mathcal { W } } _ { 1 } ( Q _ { 1 } , Q _ { 0 } ) \leq R$ and ${ \mathcal { W } } _ { 1 } ( Q _ { 2 } , Q _ { 0 } ) \leq R$ . By (3.16), for $i = 1$ , 2

$$
\left| \Delta C ( Q _ { i } ; P ^ { \mathrm { t a r } } ) - \Delta C ( Q _ { 0 } ; P ^ { \mathrm { t a r } } ) \right| \ \leq \ \frac { \mathrm { L i p } ( g ) } { 1 - \alpha } \mathcal W _ { 1 } ( Q _ { i } , Q _ { 0 } ) \ \leq \ \frac { \mathrm { L i p } ( g ) } { 1 - \alpha } R .\tag{B7}
$$

Hence

$$
\begin{array} { l } { { | \Delta C ( Q _ { i } ; P ^ { \mathrm { t a r } } ) | \le | \Delta C ( Q _ { 0 } ; P ^ { \mathrm { t a r } } ) | + \left| \Delta C ( Q _ { i } ; P ^ { \mathrm { t a r } } ) - \Delta C ( Q _ { 0 } ; P ^ { \mathrm { t a r } } ) \right| } } \\ { { \displaystyle \qquad \le | \Delta C ( Q _ { 0 } ; P ^ { \mathrm { t a r } } ) | + \frac { \mathrm { L i p } ( g ) } { 1 - \alpha } R : = M _ { R } } } \end{array}
$$

Then it readily follows that

$$
\begin{array} { r l } & { \quad \big | \left( \Delta C ( Q _ { 1 } ; P ^ { \mathrm { t a r } } ) \right) ^ { 2 } - \left( \Delta C ( Q _ { 2 } ; P ^ { \mathrm { t a r } } ) \right) ^ { 2 } \big | } \\ & { { = } \big | \Delta C ( Q _ { 1 } ; P ^ { \mathrm { t a r } } ) + \Delta C ( Q _ { 2 } ; P ^ { \mathrm { t a r } } ) \big | \big | \Delta C ( Q _ { 1 } ; P ^ { \mathrm { t a r } } ) - \Delta C ( Q _ { 2 } ; P ^ { \mathrm { t a r } } ) \big | } \\ & { { = } \left( \big | \Delta C ( Q _ { 1 } ; P ^ { \mathrm { t a r } } ) \big | + \big | \Delta C ( Q _ { 2 } ; P ^ { \mathrm { t a r } } ) \big | \right) \big | \Delta C ( Q _ { 1 } ; P ^ { \mathrm { t a r } } ) - \Delta C ( Q _ { 2 } ; P ^ { \mathrm { t a r } } ) } \\ & { { \leq } 2 M _ { R } \left| \Delta C ( Q _ { 1 } ; P ^ { \mathrm { t a r } } ) - \Delta C ( Q _ { 2 } ; P ^ { \mathrm { t a r } } ) \right| } \\ & { { \leq } 2 M _ { R } \displaystyle \frac { \mathrm { L i p } ( g ) } { 1 - \alpha } \mathcal { W } _ { 1 } ( Q _ { 1 } , Q _ { 2 } ) ; \quad \mathrm { b y ~ ( 3 . 1 6 ) } . } \end{array}
$$

Let $A ^ { Q _ { 1 } } ( \phi ) : = \mathbb { E } _ { Q _ { 1 } } [ \phi ] - \log \mathbb { E } _ { P ^ { \mathrm { t a r } } } [ e ^ { \phi } ]$ and $A ^ { Q _ { 2 } } ( \phi ) : = \mathbb { E } _ { Q _ { 2 } } [ \phi ] - \log \mathbb { E } _ { P ^ { \mathrm { t a r } } } [ e ^ { \phi } ]$ for $\phi \in \Gamma _ { L }$

Then, by the variational representation (2.6), we have

$$
\begin{array} { r l } { D _ { \mathrm { K L } } ^ { \vec { L } } ( Q _ { 1 } \lVert P ^ { \mathrm { f a r } } ) - D _ { \mathrm { K L } } ^ { \vec { L } } ( Q _ { 2 } \lVert P ^ { \mathrm { f a r } } ) = } & { \underset { \phi \in \mathbb { T } _ { L } } { \operatorname* { s u p } } A ^ { Q _ { 1 } } ( \vec { \phi } ) - \underset { \phi \in \mathbb { T } _ { L } } { \operatorname* { s u p } } A ^ { Q _ { 2 } } ( \vec { \phi } ) } \\ & { \leq \underset { \phi \in \mathbb { T } _ { L } } { \operatorname* { s u p } } A ^ { Q _ { 1 } } ( \vec { \phi } ) - A ^ { Q _ { 2 } } ( \vec { \phi } ) } \\ & { = \underset { \phi \in \mathbb { T } _ { L } } { \operatorname* { s u p } } \left\{ \mathbb { E } _ { Q _ { 1 } } [ \phi ] - \mathbb { E } _ { Q _ { 2 } } [ \vec { \phi } ] \right\} } \\ & { = \underset { \phi \in \mathbb { T } _ { L } } { \operatorname* { s u p } } \int \phi d ( Q _ { 1 } - Q _ { 2 } ) , } \\ & { = L \underset { \psi \in \mathbb { T } _ { L } } { \operatorname* { s u p } } \int \psi d ( Q _ { 1 } - Q _ { 2 } ) } \\ & { = L \underset { \psi \in \mathbb { T } _ { 1 } } { \operatorname* { s u p } } \int \psi d ( Q _ { 1 } - Q _ { 2 } ) } \\ & { = L \mathcal { W } _ { i } ( Q _ { 1 } , Q _ { 2 } ) , } \end{array}
$$

where the last equality follows from the Kantorovich-Rubinstein duality. A similar argument yields $D _ { \mathrm { K L } } ^ { L } ( Q _ { 2 } \| P ^ { \mathrm { t a r } } ) - D _ { \mathrm { K L } } ^ { L } ( Q _ { 1 } \| P ^ { \mathrm { t a r } } ) \leq L \mathcal { W } _ { 1 } ( Q _ { 1 } , Q _ { 2 } )$ . Hence we obtain

$$
\begin{array} { r } { \left| D _ { \mathrm { K L } } ^ { L } ( Q _ { 1 } \| P ^ { \mathrm { t a r } } ) - D _ { \mathrm { K L } } ^ { L } ( Q _ { 2 } \| P ^ { \mathrm { t a r } } ) \right| \leq L \mathcal { W } _ { 1 } ( Q _ { 1 } , Q _ { 2 } ) . } \end{array}\tag{B8}
$$

Then we have

$$
\begin{array} { r l } & { \quad \left| \mathcal { F } ^ { \mathrm { C V a R } } ( Q _ { 1 } ; P ^ { \mathrm { t a r } } ) - \mathcal { F } ^ { \mathrm { C V a R } } ( Q _ { 2 } ; P ^ { \mathrm { t a r } } ) \right| } \\ & { \leq \left| D _ { \mathrm { K L } } ^ { L } ( Q _ { 1 } \| P ^ { \mathrm { t a r } } ) - D _ { \mathrm { K L } } ^ { L } ( Q _ { 2 } \| P ^ { \mathrm { t a r } } ) \right| + \lambda \big | ( \Delta C ( Q _ { 1 } ; P ^ { \mathrm { t a r } } ) ) ^ { 2 } - ( \Delta C ( Q _ { 2 } ; P ^ { \mathrm { t a r } } ) ) ^ { 2 } \big | } \\ & { \leq L \mathcal { W } _ { 1 } ( Q _ { 1 } , Q _ { 2 } ) + 2 \lambda M _ { R } \frac { \mathrm { L i p } ( g ) } { 1 - \alpha } \mathcal { W } _ { 1 } ( Q _ { 1 } , Q _ { 2 } ) } \\ & { = K _ { R } \mathcal { W } _ { 1 } ( Q _ { 1 } , Q _ { 2 } ) } \end{array}
$$

where $\begin{array} { r } { K _ { R } = L + 2 \lambda M _ { R } \frac { \operatorname { L i p } ( g ) } { 1 - \alpha } } \end{array}$ . Since $Q _ { 0 } \in \mathcal { Q } ^ { g }$ and $R > 0$ were arbitrary, every $Q \in \mathcal { Q } ^ { g }$ has a W<sub>1</sub>-neighborhood on which $\mathcal { F } ^ { \mathrm { C V a R } } ( \cdot ; P ^ { \mathrm { t a r } } )$ is Lipschitz, i.e., $\mathcal { F } ^ { \mathrm { C V a R } } ( \cdot ; P ^ { \mathrm { t a r } } )$ is locally Lipschitz continuous with respect to $\mathcal { W } _ { 1 }$ ■

## Appendix C: Proof of Lemma 3.4

Lemma C.1. Assume that g is Lipschitz continuous and $Q \in \mathcal { Q } ^ { g }$ . Let $\rho$ be right-admissible at $Q$ . Then there exists $\epsilon _ { 0 } > 0$ such that

1. The function $( \epsilon , y ) \mapsto F _ { \alpha } ^ { Q ^ { \epsilon } , g } ( y )$ is jointly continuous in $[ 0 , \epsilon _ { 0 } ] \times \mathbb { R }$

2. The set $\begin{array} { r } { T : = \bigcup _ { \epsilon \in [ 0 , \epsilon _ { 0 } ] } T ( Q ^ { \epsilon } ) } \end{array}$ is compact.

Proof. (1) Denote by $\begin{array} { r } { \ell _ { \rho } ( y ) : = \frac { 1 } { 1 - \alpha } \int ( g ( x ) - y ) ^ { + } d \rho ( x ) } \end{array}$ for any $y \in \mathbb { R }$ and $\textstyle \| \rho \| = \int d | \rho |$ . For all $\begin{array} { r } { y \in \mathbb { R } , | \ell _ { \rho } ( y ) | \leq \frac { 1 } { 1 - \alpha } \left[ \int g d | \rho | + | y | \left. \rho \right. \right] < \infty } \end{array}$ since $\rho$ is right-admissible and $\textstyle \int g d | \rho | < \infty$ by (3.18). Let $\epsilon _ { \mathrm { 0 } }$ be a given threshold and let $( \epsilon , y ) \in [ 0 , \epsilon _ { 0 } ] \times \mathbb { R }$ be arbitrary. Let $\tilde { \epsilon } > 0$ and $\begin{array} { r } { \delta < \frac { \tilde { \epsilon } } { E _ { 0 } } } \end{array}$ where $\begin{array} { r } { E _ { 0 } : = \frac { 2 - \alpha } { 1 - \alpha } + | \ell _ { \rho } ( y ) | + \frac { 6 0 \| \rho \| } { 1 - \alpha } } \end{array}$ . For $\epsilon ^ { \prime } \in [ 0 , \epsilon _ { 0 } ]$ let $\| ( y , \epsilon ) - ( y ^ { \prime } , \epsilon ^ { \prime } ) \| < \delta$ . Since the function $t \mapsto t ^ { + }$ is 1-Lipschitz continuous, $| ( g ( x ) - y ) ^ { + } - ( g ( x ) - y ^ { \prime } ) ^ { + } | \leq | y - y ^ { \prime } |$ for all $\boldsymbol { x } \in \mathbb { R } ^ { d }$ . Hence, for any signed measure π, we have the following

$$
\left| \int ( g ( x ) - y ) ^ { + } - ( g ( x ) - y ^ { \prime } ) ^ { + } d \pi \right| \leq \int | y - y ^ { \prime } | d | \pi | = | y - y ^ { \prime } | \int d | \pi |\tag{C9}
$$

$$
\begin{array} { r l } & { \quad | F _ { \gamma } ^ { ( 2 ) } ( e _ { 2 \theta } ^ { ( 1 ) } , e _ { 3 \theta } ^ { ( 2 ) } , \psi _ { 1 } ^ { ( 2 ) } | ) } \\ & { = | \nu - \nu ^ { \prime } - \frac { 1 } { 1 - \sigma } \int [ \| \xi ( x ^ { ( 1 ) } - \psi _ { 1 } ^ { ( 1 ) } - \bar { \nu } _ { 2 \theta } ^ { ( 2 ) } ) - \bar { \nu } _ { 1 } \| ] d \theta + d _ { \theta } ^ { \prime } ( x ) \psi _ { 1 } - d _ { \theta } ^ { \prime } ( x ) \| } \\ & { \quad - \| \nu - \nu ^ { \prime } \| + \frac { 1 } { 1 - \sigma } \| \frac { \psi _ { 1 } } { \mu } \| \psi _ { 1 } - y _ { \theta } ^ { \prime } \| , } \\ & { \quad - \frac { 2 - \sigma \kappa } { 1 - \sigma } \langle \nu - \psi _ { 1 } ^ { ( 1 ) } - \bar { \nu } _ { 1 } ^ { ( 1 ) } | + \bar { \nu } _ { 2 \theta } ^ { ( 1 ) } - d _ { \theta } ^ { \prime } ( x ) \| } \\ & { \quad \times \frac { 1 - \sigma \kappa } { 1 - \sigma } \| \psi _ { 1 } - \bar { \nu } _ { 1 } ^ { ( 1 ) } | + \bar { \nu } _ { 4 \theta } ^ { ( 1 ) } - \bar { \nu } _ { 4 \theta } ^ { ( 1 ) } \| } \\ & { \quad - \frac { 2 - \sigma \kappa } { 1 - \sigma } \| \psi _ { 1 } ^ { ( 1 ) } - \bar { \nu } _ { 1 } ^ { ( 1 ) } | - \bar { \nu } _ { 1 } ^ { ( 1 ) } | \bar { \xi } | \bar { \nu } _ { 1 } ^ { ( 1 ) } - d _ { \theta } ^ { \prime } ( x ) \| } \\ &  \quad \times \frac { 2 - \sigma \kappa } { 1 - \sigma } \| \psi _ { 1 } ^ { ( 1 ) } - \bar { \nu } _ { 1 } ^ { ( 1 ) } | - \bar { \nu } _  2 \theta \end{array}
$$

Hence, $F _ { \alpha } ^ { Q ^ { \epsilon } , g } ( y )$ is jointly continuous in $[ 0 , \epsilon _ { 0 } ] \times \mathbb { R }$

(2) For any $c \in \mathbb { R } , | \Psi ^ { Q ^ { \epsilon } , g } ( c ) - \Psi ^ { Q , g } ( c ) | \leq \epsilon \| \rho \|$ . In the trivial case $\| \rho \| = 0 , T = T ( Q )$ and the result readily follows. Assume $\| \rho \| > 0$ . Fix $\zeta \in ( 0 , \frac { 1 } { 2 } \operatorname* { m i n } \{ \alpha , 1 - \alpha \} )$ and $a < b$ with $\Psi ^ { Q , g } ( a ) \leq \alpha - 2 \zeta , \Psi ^ { Q , g } ( b ) \geq \alpha + 2 \zeta$ ; set $\epsilon _ { 0 } = \operatorname* { m i n } \{ \zeta / \| \rho \| , \bar { \epsilon } \}$ , where $\bar { \epsilon } > 0$ is an admissibility bound $( Q ^ { \epsilon } \in \mathcal { P } ( \mathbb { R } ^ { d } )$ for $\epsilon \in [ 0 , \bar { \epsilon } ] )$ . Then for all $\epsilon \in [ 0 , \epsilon _ { 0 } ]$ we have

$$
\Psi ^ { Q ^ { \epsilon } , g } ( a ) \leq \alpha - \zeta < \alpha , \qquad \Psi ^ { Q ^ { \epsilon } , g } ( b ) \geq \alpha + \zeta > \alpha .\tag{C10}
$$

Thus, by definition of $\mathrm { V a R } _ { \alpha } ^ { Q ^ { \epsilon } , g }$ and $\overline { { \mathrm { V a R } } } _ { \alpha } ^ { Q ^ { \epsilon } , g }$ we have that $a ~ \leq ~ \mathrm { V a R } _ { \alpha } ^ { Q ^ { \epsilon } , g } ~ \leq ~ \overline { { \mathrm { V a R } } } _ { \alpha } ^ { Q ^ { \epsilon } , g } ~ \leq$ $b .$ Therefore, $T ( Q ^ { \epsilon } ) \ = \ [ \mathrm { V a R } _ { \alpha } ^ { Q ^ { \epsilon } , g } , \overline { { \mathrm { V a R } } } _ { \alpha } ^ { Q ^ { \epsilon } , g } ] \ \subseteq \ [ a , b ]$ for every $\epsilon \in \mathsf { \Gamma } [ 0 , \epsilon _ { 0 } ]$ . Thus $T : =$ $\textstyle \bigcup _ { \epsilon \in [ 0 , \epsilon _ { 0 } ] } T ( Q ^ { \epsilon } )$ is a bounded set.

Let $y _ { n } \in T$ be a sequence that converges to $\hat { y }$ . For each $n ,$ there exists $\epsilon _ { n } \in [ 0 , \epsilon _ { 0 } ]$ such that $y _ { n } \in T ( Q ^ { \epsilon _ { n } } )$ . By the compactness of the interval $[ 0 , \epsilon _ { 0 } ]$ , there exists a subsequence $\epsilon _ { n _ { k } }$ that converges to $\epsilon ^ { * } \in [ 0 , \epsilon _ { 0 } ]$ . Since $( \epsilon , y ) \mapsto F _ { \alpha } ^ { Q ^ { \epsilon } , g } ( y )$ is jointly continuous in $[ 0 , \epsilon _ { 0 } ] \times \mathbb { R }$ for any threshold $\epsilon _ { \mathrm { 0 } }$ , we have the following for any $z \in \mathbb { R }$

$$
F _ { \alpha } ^ { { Q ^ { \epsilon } } ^ { * } , g } ( \hat { y } ) = \operatorname* { l i m } _ { k \to \infty } F _ { \alpha } ^ { { Q ^ { \epsilon } } n _ { k } , g } ( y _ { n _ { k } } ) \leq \operatorname* { l i m } _ { k \to \infty } F _ { \alpha } ^ { { Q ^ { \epsilon } } n _ { k } , g } ( z ) = F _ { \alpha } ^ { { Q ^ { \epsilon } } ^ { * } , g } ( z ) .\tag{C11}
$$

That is $\hat { y } \in T ( Q ^ { \epsilon ^ { * } } ) \subseteq T$ . Thus, $T$ is closed. Hence, the result.