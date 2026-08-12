# Generator-Guided Inverse Sampling for Levy-Driven Generative Models´

Tianfu Qi, Graduate Student Member, IEEE, Jun Wang, Senior Member, IEEE, Jun Zhang, Fellow, IEEE

Abstract—This paper studies inverse sampling for Levy-driven´ generative models from the perspective of Markov generators. Unlike conventional diffusion models, Levy-driven dynamics´ involve infinite jump activities, which makes their reverse process nonlocal and difficult to characterize using score information alone. We address this challenge by analyzing the forward and reversed generators. It is derived that the reversed jump component generally becomes a state-dependent Markov jump process governed by a nonlocal density ratio. This observation motivates a structured reverse sampler that decomposes the dynamics into diffusion, small jump, and large jump components. Based on this characterization, we develop a computationally tractable sampler for a class of isotropic linear Levy SDEs with symmetric´ α- stable jump components. For the jump component, the neural network is used only to amortize the rate of large jump activities, while jump amplitudes are generated from analytically derived conditional distributions, which improves interpretability and controllability. Efficient implementation techniques are further introduced under this setting to avoid expensive high-dimensional integration and sampling. The sampler is further adapted to approximate observation-guided sampling and applied to OFDM-SISO channel estimation under mixed Gaussian and impulsive noise. Simulations show robust estimation performance with a favorable tradeoff between complexity and performance.

Index Terms—Inverse sampling, Levy process, stable distribu-´ tion, generative model

## I. INTRODUCTION

Diffusion models based on score matching have emerged as a powerful class of generative models [1]–[4]. They have also attracted increasing attention in signal processing applications, including inverse problem solving [5]–[10], image restoration [11]–[14], and wireless channel estimation [15]–[17]. By gradually perturbing data with Gaussian noise and learning the reverse dynamics through score matching, these models provide a stochastic framework for sampling from complex data distributions.

Despite their success, conventional diffusion models are mainly built on the Wiener process. Their reverse process is characterized by local score information and infinitesimal Gaussian perturbations [1], [3]. As a result, long-range probability transport is usually achieved through a sequence of small reverse steps. This local update mechanism may affect sampling efficiency and modeling flexibility [18], especially when the target distribution has heavy tails, impulsive components, or highly multimodal structures [19], [20].

Levy-driven generative models provide a natural extension´ by incorporating jump components into the forward dynamics [21]. The resulting nonlocal transitions are well suited for modeling heavy-tailed perturbations. They can also provide more flexible distribution transport than standard diffusion dynamics [19]. However, these advantages bring new challenges for inverse sampling. Unlike the case without jumps, the time reversal of a Levy-driven process cannot be fully characterized´ by local score information alone [19], [20]. Its reversed jump component depends on a nonlocal density ratio. It generally becomes a Markov jump process with state-dependent kernels, which makes the inverse sampler difficult to design.

It has been shown in [19] that the denoising score matching (DSM) framework can still be applied to inverse sampling for Levy processes. Under this formulation, the neural net-´ work is required to learn denoising targets associated with impulsive perturbations, such as those induced by α-stable distributions. However, different from white Gaussian noise (WGN), impulse noise (IN) contains multiple outliers with large amplitudes. Compared with WGN, which has a relatively stable envelope over a long time interval, IN is much more impulsive and chaotic. Therefore, it is difficult to learn the statistical characteristics of impulse noise at different moments. In this case, full neural parameterization makes the sampler less interpretable and controllable.

Motivated by these observations, we aim to establish a generator-guided inverse sampling approach for Levy-driven´ generative models. The generator provides a fundamental characterization of the infinitesimal statistics of Markov processes and naturally reveals the nonlocal structure of the reversed jump dynamics. Based on this structural characterization, we design a practical sampler under isotropic linear α-stable assumptions, so that the main components of the reverse dynamics can be implemented in a statistically interpretable and computationally tractable manner. The main contributions of this paper are summarized as follows.

• First, we provide a generator-based characterization of the time-reversed dynamics of Levy-driven Markov´ processes. The analysis shows that the reversed jump component is generally a state-dependent Markov jump process whose kernel involves a nonlocal density ratio. This result clarifies why reverse sampling for Levy-driven´ processes cannot, in general, be reduced to score-based local diffusion updates.

• Second, motivated by this characterization, we develop a practical inverse sampler for isotropic linear

Levy SDEs with symmetric´ α-stable jump components. The reverse dynamics are decomposed into diffusion, small jump, and large jump parts. The small jump contribution is approximated through a Gaussian surrogate, while large jumps are handled explicitly through a rate-and-amplitude decomposition. The neural network is used to amortize the large jump rate, whereas jump amplitudes are sampled from analytically derived conditional distributions.

• Third, we adapt the proposed sampler to approximate observation-guided sampling under observation models and demonstrate its use in OFDM-SISO channel estimation with mixed Gaussian and impulsive noise. In particular, the prior-trained jump rate network is retained as a proposal mechanism and the observation likelihood is incorporated through reweighted distributions. The resulting algorithm provides a robust estimator under heavy-tailed perturbations and offers an interpretable alternative to fully neural reverse-transition parameterizations.

The remainder of this paper is organized as follows. In Section II, we describe the Levy process considered through-´ out this paper and present the corresponding assumptions. Statistical analyses from the generator perspective are given in Section III. Based on the above theoretical results, an efficient inverse sampling algorithm is designed in Section IV. Section V provides simulation results for channel estimation under mixed channel noise to validate the proposed inverse sampling framework. Section VI concludes the paper.

Notations: Vectors are denoted by bold letters and are assumed to be column vectors by default. Superscripts and subscripts are also used to denote vectors and vector sequences. For example, given an m-dimensional vector $^ { x , }$ we have $\pmb { x } = [ x _ { 1 } , x _ { 2 } , \cdots , x _ { m } ] ^ { \top }$ and $\pmb { x } _ { 1 } ^ { p } = [ \pmb { x } _ { 1 } , \cdots , \pmb { x } _ { p } ] ^ { \top }$ Uppercase letters denote random variables $\mathrm { ( R V s ) }$ , and lowercase letters denote their realizations, e.g., X and x. The operators $\Delta , \nabla$ , and ∇· denote the Laplace operator, gradient operator, and divergence operator, respectively. The notation di $\arg ( a _ { 1 } , a _ { 2 } , \cdot \cdot \cdot , a _ { L } )$ represents a diagonal matrix with diagonal elements $a _ { 1 } , a _ { 2 } , \cdot \cdot \cdot , a _ { L } . \ \langle x , y \rangle$ denotes the inner product of vectors x and $y . \ 1 _ { \mathcal { A } } ( x )$ denotes the indicator function, which equals 1 if $x \in A .$ . The real number domain is denoted by R. The notation $X \sim$ · indicates that X follows a certain distribution.

## II. PRELIMINARIES

In this paper, we consider a Levy process with drift and´ diffusion components, which can be generally written as

$$
\begin{array} { r l } & { d { \boldsymbol { { X } } _ { t } } = b ( { \boldsymbol { { X } } _ { t } } , t ) d t + \Phi _ { G } ( t ) d { \boldsymbol { W } _ { t } } + \Phi _ { S } ( t ) d L _ { t } , } \\ & { \qquad t : \mathrm { ~ } 0 \to T , \boldsymbol { { X } } _ { t } \in \mathbb { R } ^ { D } } \end{array}\tag{1}
$$

where D denotes the dimension of the SDE. The term $\pmb { b } ( X _ { t } , t )$ is the drift coefficient of the dynamics, which depends on both the time index t and the system state $X _ { t }$ . The processes $W _ { t }$ and $\scriptstyle { L _ { t } }$ denote the Wiener process and the α-stable process, respectively. The matrices $\Phi _ { G } ( t )$ and $\Phi _ { S } ( t )$ denote the scaling matrices for $\mathbf { } W _ { t }$ and $\scriptstyle L _ { t } .$ , respectively. For isotropic cases, $\Phi _ { G } ( t )$ and $\Phi _ { S } ( t )$ reduce to scaled identity matrices. In the sequel, we focus on the SDE under the following assumptions.

Assumption 1: The Wiener process $\mathbf { } W _ { t }$ and the Levy process´ $\scriptstyle { \mathbf { L } } _ { t }$ are mutually independent.

Assumption 2: $\scriptstyle { L _ { t } }$ is a D-dimensional symmetric isotropic α-stable Levy process with stability index´ $\alpha \in ( 0 , 2 )$

Our ultimate goal is to investigate the characteristics of the reverse process corresponding to the SDE in (1) and design an efficient reverse sampler. Unfortunately, unlike a diffusion process driven only by Brownian motion, (1) is a Levy process´ with infinite jump activities. Its reverse process cannot be described using only local information. In other words, if we set $\Phi _ { S } ( t ) \equiv \mathbf { 0 }$ , the backward process of (1) can be determined by the score information, i.e., $\nabla \log p ( \pmb { x } _ { t } )$ . This information is local since it only requires the gradient of the density. This property can be obtained by analyzing the forward and backward Kolmogorov equations, which are special cases of the Kramers-Moyal (KM) expansion. The core idea of the KM expansion is to express the time evolution of the probability density as an infinite series of state-space changes.

For diffusion processes driven by Brownian motion, this series can be truncated after the first two terms, which leads to the forward Kolmogorov equation. However, if $\Phi _ { S } ( t ) \neq \mathbf { 0 }$ large jumps may appear in (1) due to $d { \pmb { L } } _ { t }$ . As a result, $\| X _ { t + \Delta t } - X _ { t } \|$ can be quite large. Moreover, the KM expansion involves the $j \cdot$ -th moment of $X _ { t } ,$ , where $j \in \mathbb { Z } ^ { + } \cup \{ 0 \}$ For a Levy process driven by an ´ α-stable process, finite $p \textmd { - }$ th moments with $p \geq \alpha$ do not exist. Therefore, the reverse process becomes much more complicated. In the following, we use the generator to analyze the statistical properties of (1).

Given the forward dynamics in (1), its generator is given by [21]

$$
\begin{array} { l } { \displaystyle \left( \mathcal { L } _ { F } f \right) ( \pmb { x } ) = \pmb { b } ( \pmb { x } , t ) \cdot \nabla f ( \pmb { x } ) + \frac { 1 } { 2 } \mathrm { T r } \left( \Sigma ( t ) \nabla ^ { 2 } f ( \pmb { x } ) \right) } \\ { \displaystyle ~ + \left( \mathcal { L } _ { J , F } f \right) ( \pmb { x } ) } \end{array}\tag{2}
$$

where $\Sigma ( t ) ~ = ~ \Phi _ { G } ( t ) \Phi _ { G } ( t ) ^ { \top }$ and $( \mathcal { L } _ { J , F } f ) ( { \pmb x } )$ denotes the forward generator of the process $\Phi _ { S } ( t ) d L _ { t }$ . The backward generator of the reversed version of (1) will be characterized in the next section. Before ending this section, we introduce the following assumptions, which ensure the validity of the subsequent analysis.

Assumption $3 ;$ The forward SDE admits a non-explosive Markov solution with transition density $p ( { \pmb x } _ { t } )$ with respect to the Lebesgue measure.

Assumption $4 { : }$ The density $p ( { \pmb x } _ { t } )$ is strictly positive and sufficiently smooth so that $\nabla \log p ( \pmb { x } _ { t } )$ is well defined.

## III. GENERATOR-BASED REVERSE DYNAMICS

In this section, we first derive the backward generator corresponding to the forward process in (1). Then, we theoretically analyze how to decompose the whole inverse sampling process and generate each component.

## A. Backward generator

Throughout this paper, the term “backward generator” refers to the infinitesimal generator of the time-reversed process.

It should not be confused with the backward Kolmogorov operator associated with the forward process. For notational clarity, we write $p ( { \pmb x } _ { t } )$ for the marginal density of $X _ { t }$

Based on (1) and its forward generator (2), the backward generator is given in the following proposition.

Proposition 1 (Backward generator): Consider the forward SDE in (1) with Levy measure´ ν. Let $\nu _ { t } ~ = ~ ( \Phi _ { S } ( t ) ) _ { \# } \nu$ be the push-forward Levy measure induced by the jump scaling´ matrix $\Phi _ { S } ( t )$ . Suppose that the forward process admits a strictly positive marginal density $p ( { \pmb x } _ { t } )$ , and that $p ( { \pmb x } _ { t } )$ and ${ \mathbf { } } b ( { \mathbf { } } x , t )$ are sufficiently smooth and integrable so that the following integrations by parts and principal-value integrals are well defined. If the scaled Levy measure ´ $\nu _ { t }$ is symmetric, then for any test function $f \in C _ { c } ^ { 2 } ( \mathbb { R } ^ { D } )$ , the generator of the reversed process can be written as

$$
\begin{array} { l } { \displaystyle ( L _ { B } f ) ( { \pmb x } _ { t } ) = [ - { \pmb b } ( { \pmb x } _ { t } , t ) + \Sigma ( t ) \nabla \log p ( { \pmb x } _ { t } ) ] \cdot \nabla f ( { \pmb x } _ { t } ) } \\ { \displaystyle + \frac { 1 } { 2 } \mathrm { T r } \left( \Sigma ( t ) \nabla ^ { 2 } f ( { \pmb x } _ { t } ) \right) + \big ( L _ { J , B } f ) ( { \pmb x } _ { t } ) } \end{array}\tag{3}
$$

$$
\begin{array} { r } { ( L _ { J , B } f ) ( \pmb { x } _ { t } ) = \mathrm { p . v . } \displaystyle \int _ { \mathbb { R } ^ { D } } \left[ f ( \pmb { x } _ { t } + \pmb { v } ) - f ( \pmb { x } _ { t } ) \right] } \\ { \times \displaystyle \frac { p ( \pmb { x } _ { t } + \pmb { v } ) } { p ( \pmb { x } _ { t } ) } \nu _ { t } ( d \pmb { v } ) . \qquad } \end{array}\tag{4}
$$

where p. v. denotes the Cauchy principal-value interpretation.

Proof: The proof is relegated to Appendix A.

Remark 1: The jump integral in Proposition 1 is understood as

$$
\begin{array} { r l } & { \quad \mathrm { p . v . } \displaystyle \int _ { \mathbb R ^ { D } } \left[ f ( { \pmb x } _ { t } + { \pmb v } ) - f ( { \pmb x } _ { t } ) \right] \frac { p ( { \pmb x } _ { t } + { \pmb v } ) } { p ( { \pmb x } _ { t } ) } \nu _ { t } ( d { \pmb v } ) } \\ & { = \displaystyle \operatorname* { l i m } _ { \epsilon \to 0 } \int _ { \| { \pmb v } \| > \epsilon } \left[ f ( { \pmb x } _ { t } + { \pmb v } ) - f ( { \pmb x } _ { t } ) \right] \frac { p ( { \pmb x } _ { t } + { \pmb v } ) } { p ( { \pmb x } _ { t } ) } \nu _ { t } ( d { \pmb v } ) . } \end{array}
$$

For $\alpha \ < \ 1$ , the integral is locally absolutely integrable under standard smoothness assumptions. For $\alpha ~ \geq ~ 1$ , the uncompensated integral is generally not absolutely integrable near the origin and should be interpreted in the Cauchy principal-value sense. To see this, for smooth f and $p ( { \pmb x } _ { t } )$ we have $f ( { \pmb x } _ { t } + { \pmb v } ) - f ( { \pmb x } _ { t } ) = \nabla f ( { \pmb x } _ { t } ) ^ { \top } { \pmb v } + O ( \| { \pmb v } \| ^ { 2 } )$ and $\begin{array} { r } { \frac { p ( { \boldsymbol x } _ { t } + { \boldsymbol v } ) } { p ( { \boldsymbol x } _ { t } ) } = 1 + { \boldsymbol v } ^ { \top } \nabla \log p ( \bar { \boldsymbol x } _ { t } ) + O ( \| { \boldsymbol v } \| ^ { 2 } ) } \end{array}$ . The leading firstorder term is odd in v and is canceled by the symmetry of $\nu _ { t }$ in the principal-value sense. The remaining second-order terms are locally integrable for $\alpha < 2$ . This justifies the principalvalue form used in Proposition 1.

Remark 2: We specify the explicit expression of the scaled Levy measure´ $\nu _ { t } ( d \pmb { v } )$ to facilitate the following analysis. For the D-dimensional case, the standard Levy measure is´ defined as $\nu ( d \pmb { v } ) \ = \ C _ { D , \alpha } \| \pmb { v } \| ^ { - D - \alpha }$ [21], where ${ \mathrm { \it C } } _ { D , \alpha } \ =$ $\frac { \alpha 2 ^ { \alpha - 1 } \Gamma ( \frac { D + \alpha } { 2 } ) } { \pi ^ { \frac { D } { 2 } } \Gamma ( 1 - \frac { \alpha } { 2 } ) }$ . Combined with the effect of $\Phi _ { S } ( t ) = \sigma _ { S } ( t ) I _ { D }$ we have $\dot { \nu _ { t } ( d \pmb { v } ) } = | \sigma _ { S } ( t ) | ^ { \alpha } \nu ( d \pmb { v } )$ . In the following, we denote $C _ { D , \alpha , \sigma _ { S } } \triangleq | \sigma _ { S } ( t ) | ^ { \alpha } C _ { D , \alpha }$ to avoid redundancy.

From Proposition 1, we can see that the backward kernel is related to the density ratio $\frac { p ( \pmb { x } _ { t } + \pmb { v } ) } { p ( \pmb { x } _ { t } ) }$ , which contains global information. Therefore, the jump part only belongs to a Markovtype jump process rather than a Levy process. This process´ is much more general and is not as mathematically tractable as the forward process. Consequently, the total reversed SDE corresponding to (1) cannot be expressed as the superposition of a modified drift term, a diffusion term, and a stable process. This makes it challenging to design an efficient sampler directly, as in the diffusion case.

Besides guiding the design of the reverse sampler, the backward generator is also useful for understanding discrete diffusion processes. For example, in a standard DDPM driven by a Gaussian transition kernel, $p ( \pmb { x } _ { t - \Delta t } | \pmb { x } _ { t } )$ can be modeled by a Gaussian distribution for the following reason. First, with sufficient diffusion, $\mathbf { \nabla } _ { \mathbf { \mathcal { X } } \mathcal { T } }$ can be approximated by a Gaussian distribution. Next, based on the backward Kolmogorov equation, the drift of the reversed SDE is $- ( b ( { \pmb x } _ { t } , t ) \ -$ $g ( t ) ^ { 2 } \nabla \log p ( { \pmb x } _ { t } , t ) ) d t$ , where $g ( t )$ is the coefficient of the Wiener process in the forward SDE. ${ \mathrm { ~ \bf ~ A ~ t ~ } } t \ = \ T$ , we have $\nabla \log p ( { \pmb x } _ { T } , T ) \propto - { \pmb x } _ { T }$ . Then, we discretize the continuous time interval with step size $\Delta t .$ In this case, $\begin{array} { r l } { \pmb { x } _ { T - \Delta t } } & { { } = } \end{array}$ $\pmb { x } _ { T } - ( \pmb { b } ( \pmb { x } _ { T } , T ) + \pmb { x } _ { T } ) \Delta t + g ( t ) \sqrt { \Delta t } \pmb { w }$ , where ${ \pmb w } \sim \mathcal { N } ( { \bf 0 } , { \pmb I } _ { D } )$ and $\pmb { b } ( \pmb { x } _ { t } , t )$ is usually set as an affine function of $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { t } }$ . This recursive formula is equivalent to the summation of Gaussiandistributed RVs. Therefore, $p ( \pmb { x } _ { t - \Delta t } | \pmb { x } _ { t } )$ can be treated as a Gaussian distribution, and the objective loss function can be significantly simplified by regressing the expectation of $\mathbf { \Delta } x _ { t } , t : T  0$

Unfortunately, according to Proposition 1, this simplification does not hold for the process in (1). In the sequel, we will design a simple loss function that is more suitable for neural network learning.

## B. Reverse sampling decomposition

From the definition of the generator, given the jump kernel $K ( d \pmb { v } )$ , the term $\begin{array} { r } { \int _ { \mathbb { R } ^ { D } } ( f ( \pmb { x } + \pmb { v } ) - f ( \pmb { x } ) ) K ( d \pmb { v } ) } \end{array}$ indicates that the jump amplitude follows the distribution $\begin{array} { r } { K ( d \pmb { v } ) / \int _ { \mathbb { R } ^ { D } } K ( d \pmb { v } ) } \end{array}$ . However, the number of small jumps with $\lVert \boldsymbol { v } \rVert \leq \epsilon$ is usually infinite due to the singularity of the Levy measure at the origin. Therefore, we handle small and´ large jumps separately. Specifically, we decompose the original integral by truncation with threshold ϵ:

$$
\begin{array} { c } { { \displaystyle \int _ { \mathbb { R } ^ { D } } \displaystyle \big ( f ( { \boldsymbol x } + { \boldsymbol v } ) - f ( { \boldsymbol x } ) \big ) K ( d { \boldsymbol v } ) } } \\ { { = \displaystyle \int _ { \| { \boldsymbol v } \| \leq \epsilon } \displaystyle \big ( f ( { \boldsymbol x } + { \boldsymbol v } ) - f ( { \boldsymbol x } ) \big ) K ( d { \boldsymbol v } ) } } \\ { { + \displaystyle \int _ { \| { \boldsymbol v } \| > \epsilon } \displaystyle \big ( f ( { \boldsymbol x } + { \boldsymbol v } ) - f ( { \boldsymbol x } ) \big ) K ( d { \boldsymbol v } ) } } \end{array}\tag{5}
$$

In other words, jumps with amplitude larger than ϵ are treated as large jumps. For large jumps, the occurrence frequency is finite. The number of large jumps follows a Poisson distribution with rate $\begin{array} { r } { \lambda \triangleq \int _ { \| v \| > \epsilon } \dot { K ( d v ) } } \end{array}$ . For a sufficiently small time interval $\Delta t$ , the occurrence of a large jump can also be approximated by a Bernoulli distribution with probability $\lambda \Delta t$ . If a large jump occurs, we need to obtain one jump sample from the normalized distribution $\frac { K ( d \pmb { v } ) } { \lambda }$ . Otherwise, no large jump occurs within the interval $\Delta t .$ , and the reversed process reduces to the standard diffusion process.

For the case with $\| \pmb { v } \| \ \le \ \epsilon$ , the $\operatorname { L e v y }$ measure is not integrable, and there are infinitely many small jumps. Unlike large jump activities, these small jumps cannot be enumerated explicitly. The distribution of small jumps is also complicated, which does not follow a Gaussian or stable distribution due to amplitude truncation. A simple method is to directly ignore small jump activities when the threshold ϵ is small enough. However, this may introduce considerable approximation error, especially when the small jumps have a nonzero mean. Meanwhile, this effect accumulates when ∆t is small.

Here, we adopt a more reasonable approach. After truncation, the small jumps have a finite second moment. Thus, we replace the small jump generator by a Gaussian generator matching the first two local moments. This is a weak approximation whose generator error is controlled by the third local moment of the truncated Levy measure. Namely, the small´ jump part within $[ t , t + \Delta t ]$ , denoted by $\boldsymbol { X } _ { t , \Delta t } ^ { \le \epsilon } ,$ can be written as

$$
\begin{array} { r } { \pmb { X } _ { t , \Delta t } ^ { \le \epsilon } \approx \pmb { b } _ { t , \Delta t } ^ { \le \epsilon } \Delta t + \sqrt { \Delta t } ( \pmb { \Sigma } _ { t , \Delta t } ^ { \le \epsilon } ) ^ { \frac { 1 } { 2 } } \pmb { W } } \end{array}\tag{6}
$$

where W is the standard Gaussian RV and

$$
b _ { t , \Delta t } ^ { \leq \epsilon } = \mathrm { p . v . } \int _ { \| v \| \leq \epsilon } v K ( d v )\tag{7}
$$

Substituting the state-dependent backward jump kernel into the small jump drift, we have

$$
\begin{array} { r l } & { b _ { t , \Delta t } ^ { \leq \epsilon } = \mathrm { p . v . } \displaystyle \int _ { \| v \| \leq \epsilon } v \frac { p \left( x _ { t } + v \right) } { p \left( x _ { t } \right) } \nu _ { t } ( d v ) } \\ & { \quad \quad \quad = \displaystyle \int _ { \| v \| \leq \epsilon } v \left( \frac { p \left( x _ { t } + v \right) } { p \left( x _ { t } \right) } - 1 \right) \nu _ { t } ( d v ) } \\ & { \quad \quad \quad = \displaystyle \int _ { \| v \| \leq \epsilon } v \left( v ^ { \top } \nabla \log p ( x _ { t } ) + R _ { p } ( x _ { t } , v , t ) \right) \nu _ { t } ( d v ) } \\ & { \quad \quad \quad = A _ { \nu _ { t } } \nabla \log p ( x _ { t } ) + R _ { b , \epsilon } ( x _ { t } , t ) , } \end{array}\tag{8}
$$

where the truncated second-order moment matrix is defined as $\begin{array} { r } { A _ { \nu _ { t } } \triangleq \int _ { \| v \| < \epsilon } { v v ^ { \top } \nu _ { t } ( d v ) } } \end{array}$ and the Taylor remainder is $\begin{array} { r } { R _ { b , \epsilon } ( \pmb { x } _ { t } , t ) \triangleq \int _ { \| \pmb { v } \| < \epsilon } \pmb { v } R _ { p } ( \pmb { x } _ { t } , \pmb { v } , t ) \nu _ { t } ( d \pmb { v } ) } \end{array}$ . Here, the second equality follows from the symmetry of $\nu _ { t }$ , which gives

$$
\mathrm { p . v . } \int _ { \lVert v \rVert \leq \epsilon } { v \nu _ { t } ( d v ) } = \mathbf { 0 } .\tag{9}
$$

If $p ( { \pmb x } _ { t } )$ is locally smooth and strictly positive, then

$$
\begin{array} { r } { \frac { p ( { \pmb x } _ { t } + { \pmb v } ) } { p ( { \pmb x } _ { t } ) } = 1 + { \pmb v } ^ { \top } \nabla \log p ( { \pmb x } _ { t } ) + R _ { p } ( { \pmb x } _ { t } , { \pmb v } , t ) , } \\ { | R _ { p } ( { \pmb x } _ { t } , { \pmb v } , t ) | \leq C \| { \pmb v } \| ^ { 2 } . } \end{array}\tag{10}
$$

For the isotropic α-stable Levy measure ´ $\begin{array} { r l } { \nu _ { t } ( d \pmb { v } ) } & { { } = } \end{array}$ $C \| \pmb { v } \| ^ { - D - \alpha } d \pmb { v }$ , the drift remainder satisfies

$$
\begin{array} { r l } { \| R _ { b , \epsilon } ( { \pmb x } _ { t } , t ) \| \le C \displaystyle \int _ { \| { \pmb v } \| \le \epsilon } \| { \pmb v } \| ^ { 3 } \nu _ { t } ( d { \pmb v } ) } & { } \\ { = C \epsilon ^ { 3 - \alpha } . } \end{array}\tag{11}
$$

Therefore, by replacing the unknown score ∇ log p(x<sub>t</sub>) with the score network $s ^ { \theta _ { 1 } } ( \pmb { x } _ { t } , t )$ , the practical approximation becomes

$$
b _ { t , \Delta t } ^ { \le \epsilon } \approx A _ { \nu _ { t } } s ^ { \theta _ { 1 } } ( { \pmb x } _ { t } , t ) .\tag{12}
$$

Similarly, the covariance matrix of the small jump component is given by

$$
\begin{array} { r l r } {  { \Sigma _ { t , \Delta t } ^ { \leq \epsilon } } } \\ & { = \int _ { \| v \| \leq \epsilon } { v v ^ { \top } ( 1 + v ^ { \top } \nabla \log p ( { \pmb x } _ { t } ) + R _ { p } ( { \pmb x } _ { t } , { \pmb v } , t ) ) \nu _ { t } ( d v ) } } \\ & { = A _ { { \nu } _ { t } } + R _ { \Sigma , \epsilon } ( { \pmb x } _ { t } , t ) . } & { ( 1 3 ) } \end{array}
$$

where the remaining Taylor term satisfies

$$
\| R _ { \Sigma , \epsilon } ( { \boldsymbol x } _ { t } , t ) \| \leq C \int _ { \| { \boldsymbol v } \| \leq \epsilon } \| { \boldsymbol v } \| ^ { 4 } \nu _ { t } ( d { \boldsymbol v } ) = C \epsilon ^ { 4 - \alpha } .\tag{14}
$$

Thus, the covariance matrix can be approximated as

$$
\begin{array} { r } { \Sigma _ { t , \Delta t } ^ { \le \epsilon } \approx A _ { \nu _ { t } } . } \end{array}\tag{15}
$$

Both (8) and (15) admit closed-form expressions based on the polar coordinate transform. For instance,

$$
\begin{array} { l } { \displaystyle { A _ { \nu _ { t } } = \int _ { \| v \| \le \epsilon } v v ^ { \top } \nu _ { t } ( d v ) } } \\ { \displaystyle { \quad = \frac { | \mathbb { S } _ { D - 1 } | C _ { D , \alpha , \sigma _ { s } } I _ { D } } { D } \int _ { 0 } ^ { \epsilon } r ^ { 1 - \alpha } d r } } \\ { \displaystyle { \quad = \frac { 2 \pi ^ { \frac { D } { 2 } } C _ { D , \alpha , \sigma _ { s } } \epsilon ^ { 2 - \alpha } } { D \Gamma ( \frac { D } { 2 } ) ( 2 - \alpha ) } I _ { D } , } } \end{array}\tag{16}
$$

where $I _ { D }$ denotes the D-dimentional identity matrix and $\mathbb { S } _ { D - 1 }$ represents the $D - 1$ -dimensional sphere. Note that this is a generator-level weak approximation rather than an exact Gaussian representation of the small jump sum. Denote the effect of large jumps on $X _ { t }$ by $X _ { t , \Delta t } ^ { > \epsilon }$ . Then, the total jump influence during the reverse sampling process is $X _ { t , \Delta t } ^ { \le \epsilon } + X _ { t , \Delta t } ^ { > \epsilon }$ . Finally, by combining the externally applied drift term and the reversed diffusion part, the recursive relation between $X _ { t - \Delta t }$ and $X _ { t }$ can be constructed.

## IV. INVERSE SAMPLING

According to the above analysis, the main challenge of inverse sampling comes from large jump sampling. Therefore, this section mainly focuses on how to generate large jump samples based on the current system state and time instant. First, we explain how to obtain large jump samples. We also show that, from a theoretical perspective, a neural network is not necessary for reverse sampling. These observations help determine the learning target and design the corresponding network structure. Then, two techniques are introduced to effectively reduce the complexity. Finally, since additional conditions are often required to control the generation process in practical applications, we also discuss an approximate observation-guided version for inverse problems.

Before proceeding, we first describe a special family of functions that is important for efficient learning.

In many scenarios, we may want to train a network to regress a target $f ( { \pmb x } )$ that is not directly accessible. However, we can efficiently evaluate only its conditional version $f ( { \pmb x } | { \pmb x } _ { 0 } )$ . Thus, we aim to establish the condition under which optimizing the neural network based on $f ( { \pmb x } | { \pmb x } _ { 0 } )$ is equivalent to optimizing it based on $f ( { \pmb x } )$ . This is the core of the following theorem.

Theorem 1 (Marginal trainable functions): Define the accurate target and conditional target as $l ( { \pmb x } ) : \mathbb { R } ^ { \hat { D } }  \mathbb { R } ^ { \tilde { D } }$ and $l ( \pmb { x } | \pmb { x } _ { 0 } ) : \mathbb { R } ^ { D }  \mathbb { R } ^ { \tilde { D } }$ , respectively. Denote the neural network by $f ^ { \theta } ( { \pmb x } )$ , which is used to regress $l ( { \pmb x } )$ . Then, the optimization problem arg min<sub>θ</sub> $\mathbb { E } _ { { \pmb x } \sim p ( { \pmb x } ) } [ \| { \dot { \pmb f } } ^ { \theta } ( { \pmb x } ) - l ( { \pmb x } ) \| ^ { 2 } ]$ is equivalent to arg min<sub>θ</sub> $\mathbb { E } _ { \pmb { x } , \pmb { x } _ { 0 } \sim p ( \pmb { x } , \pmb { x } _ { 0 } ) } [ \| \hat { f } ^ { \theta } ( \pmb { x } ) - l ( \pmb { x } | \pmb { x } _ { 0 } ) \| ^ { 2 } ] \ \ i$ if

$$
l ( { \pmb x } ) = \int l ( { \pmb x } | { \pmb x } _ { 0 } ) p ( { \pmb x } _ { 0 } | { \pmb x } ) d { \pmb x } _ { 0 }\tag{17}
$$

and the functions $l ( { \pmb x } )$ satisfying (17) are referred to as marginally trainable functions.

Proof: The original optimization problem can be reformulated as follows:

$$
\begin{array} { r l } & { \quad \mathbb { E } _ { \pmb { x } \sim p ( \pmb { x } ) } [ \| f ^ { \theta } ( \pmb { x } ) - l ( \pmb { x } ) \| ^ { 2 } ] } \\ & { = \mathbb { E } _ { \pmb { x } \sim p ( \pmb { x } ) } [ \| f ^ { \theta } ( \pmb { x } ) \| ^ { 2 } ] - 2 \mathbb { E } _ { \pmb { x } \sim p ( \pmb { x } ) } [ f ^ { \theta } ( \pmb { x } ) l ( \pmb { x } ) ] } \\ & { \quad + \mathbb { E } _ { \pmb { x } \sim p ( \pmb { x } ) } [ \| l ( \pmb { x } ) \| ^ { 2 } ] } \end{array}\tag{18}
$$

The third term in (18) is independent of θ and can be ignored. For the second term,

$$
\begin{array} { r l r } {  { \mathbb { E } _ { { \pmb x } \sim p ( { \pmb x } ) } [ f ^ { \theta } ( { \pmb x } ) l ( { \pmb x } ) ] = \int f ^ { \theta } ( { \pmb x } ) \int l ( { \pmb x } | { \pmb x } _ { 0 } ) p ( { \pmb x } _ { 0 } | { \pmb x } ) d { \pmb x } _ { 0 } p ( { \pmb x } ) d { \pmb x } } } \\ & { } & { = \int \int f ^ { \theta } ( { \pmb x } ) l ( { \pmb x } | { \pmb x } _ { 0 } ) p ( { \pmb x } _ { 0 } | { \pmb x } ) p ( { \pmb x } ) d { \pmb x } _ { 0 } d { \pmb x } } \\ & { } & { = \mathbb { E } _ { { \pmb x } , { \pmb x } _ { 0 } \sim p ( { \pmb x } , { \pmb x } _ { 0 } ) } [ f ^ { \theta } ( { \pmb x } ) l ( { \pmb x } | { \pmb x } _ { 0 } ) ] \qquad ( 1 9 ) } \end{array}
$$

Plugging (19) into (18) completes the proof.

Theorem 1 provides sufficient conditions under which a wide range of functions can be used for network training. In fact, Theorem 1 is general since it includes conventional training frameworks such as score matching and flow matching. For example, the target of score matching is the score function $\nabla \log p ( \pmb { x } _ { t } )$ , which is not accessible in practice. Therefore, the conditional score ∇ log $p ( \pmb { x } _ { t } | \pmb { x } _ { 0 } )$ is usually used, together with the simple underlying relationship between $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { t } }$ and $\scriptstyle { \mathbf { { \mathit { x } } } } _ { 0 }$ . It can be verified that

$$
\begin{array} { l } { \displaystyle \nabla \log p ( { \pmb x } _ { t } ) = \frac { 1 } { p ( { \pmb x } _ { t } ) } \nabla \int p ( { \pmb x } _ { t } | { \pmb x } _ { 0 } ) p ( { \pmb x } _ { 0 } ) d { \pmb x } _ { 0 } } \\ { \displaystyle = \int \frac { \nabla p ( { \pmb x } _ { t } | { \pmb x } _ { 0 } ) } { p ( { \pmb x } _ { t } ) p ( { \pmb x } _ { t } | { \pmb x } _ { 0 } ) } p ( { \pmb x } _ { 0 } ) p ( { \pmb x } _ { t } | { \pmb x } _ { 0 } ) d { \pmb x } _ { 0 } } \\ { \displaystyle = \int \frac { \nabla p ( { \pmb x } _ { t } | { \pmb x } _ { 0 } ) } { p ( { \pmb x } _ { t } | { \pmb x } _ { 0 } ) } p ( { \pmb x } _ { 0 } | { \pmb x } _ { t } ) d { \pmb x } _ { 0 } } \\ { \displaystyle = \int \nabla \log p ( { \pmb x } _ { t } | { \pmb x } _ { 0 } ) p ( { \pmb x } _ { 0 } | { \pmb x } _ { t } ) d { \pmb x } _ { 0 } } \end{array}\tag{20}
$$

Hence, $\nabla \log p ( \pmb { x } _ { t } )$ belongs to the class of marginally trainable functions. Similar derivations apply to the training of the conditional vector field in flow matching [22].

## A. Sampling of large jumps

Suppose that the dataset $\pmb { x } _ { 0 , j } , j = 1 , \cdots , N$ is available. According to the forward SDE, the expressions of $p ( \pmb { x } _ { t } | \pmb { x } _ { 0 } )$ )

and $\frac { p ( \pmb { x } _ { t } + \pmb { v } | \pmb { x } _ { 0 } ) } { p ( \pmb { x } _ { t } | \pmb { x } _ { 0 } ) }$ can usually be efficiently obtained. Consequently,

$$
\begin{array} { r l } { \displaystyle \frac { p ( x _ { t } + v ) } { p ( x _ { t } ) } = \int \frac { p ( x _ { t } + v | x _ { 0 } ) } { p ( x _ { t } | x _ { 0 } ) } p ( x _ { 0 } | x _ { t } ) d x _ { 0 } \ ~ } & { } \\ { \displaystyle } & { = \int \frac { p ( x _ { t } + v | x _ { 0 } ) } { p ( x _ { t } | x _ { 0 } ) } \frac { p ( x _ { 0 } ) p ( x _ { t } | x _ { 0 } ) } { p ( x _ { t } ) } d x _ { 0 } } \\ { \displaystyle } & { = \int \frac { p ( x _ { t } + v | x _ { 0 } ) } { p ( x _ { t } | x _ { 0 } ) } w ( x _ { t } , x _ { 0 } ) p ( x _ { 0 } ) d x _ { 0 } } \\ { \displaystyle } & { \approx \frac { 1 } { N } \sum _ { j = 1 } ^ { N } \frac { p ( x _ { t } + v | x _ { 0 , j } ) } { p ( x _ { t } | x _ { 0 , j } ) } w ( x _ { t } , x _ { 0 , j } ) } \end{array}\tag{21}
$$

where we define

$$
\begin{array} { r l } & { w ( \pmb { x } _ { t } , \pmb { x } _ { 0 } ) \triangleq \frac { p ( \pmb { x } _ { t } | \pmb { x } _ { 0 } ) } { p ( \pmb { x } _ { t } ) } = \frac { p ( \pmb { x } _ { t } | \pmb { x } _ { 0 } ) } { \int p ( \pmb { x } _ { t } | \pmb { x } _ { 0 } ) p ( \pmb { x } _ { 0 } ) d \pmb { x } _ { 0 } } } \\ & { \qquad \approx \frac { p ( \pmb { x } _ { t } | \pmb { x } _ { 0 } ) } { \frac { 1 } { N } \sum _ { j = 1 } ^ { N } p ( \pmb { x } _ { t } | \pmb { x } _ { 0 , j } ) } } \end{array}\tag{22}
$$

As for the overall jump rate, we have

$$
\begin{array} { l } { \displaystyle \lambda ( \pmb { x } _ { t } ) \triangleq \int _ { \| \pmb { v } \| > \epsilon } \frac { p ( \pmb { x } _ { t } + \pmb { v } ) } { p ( \pmb { x } _ { t } ) } \nu _ { t } ( d \pmb { v } ) } \\ { \displaystyle = \frac { 1 } { \sum _ { j = 1 } ^ { N } p ( \pmb { x } _ { t } | \pmb { x } _ { 0 , j } ) } \sum _ { j = 1 } ^ { N } \int _ { \| \pmb { v } \| > \epsilon } p ( \pmb { x } _ { t } + \pmb { v } | \pmb { x } _ { 0 , j } ) \nu _ { t } ( d \pmb { v } ) } \end{array}\tag{23}
$$

Finally, the unnormalized jump distribution is written as follows,

$$
\begin{array} { c l } { q ( { \pmb x } _ { t } , { \pmb v } ) \triangleq \displaystyle \frac { p ( { \pmb x } _ { t } + { \pmb v } ) } { p ( { \pmb x } _ { t } ) } \nu _ { t } ( d { \pmb v } ) } \\ { \approx \displaystyle \frac { \sum _ { j = 1 } ^ { N } p ( { \pmb x } _ { t } + { \pmb v } | { \pmb x } _ { 0 , j } ) } { \sum _ { j = 1 } ^ { N } p ( { \pmb x } _ { t } | { \pmb x } _ { 0 , j } ) } \nu _ { t } ( d { \pmb v } ) , \| { \pmb v } \| > \epsilon } \end{array}\tag{24}
$$

The standard score function can also be directly approximated from the dataset based on (20) as follows:

$$
\begin{array} { r } { \nabla \log p ( { \pmb x } _ { t } ) = \displaystyle \int \nabla \log p ( { \pmb x } _ { t } | { \pmb x } _ { 0 } ) p ( { \pmb x } _ { 0 } | { \pmb x } _ { t } ) d { \pmb x } _ { 0 } } \\ { \approx \displaystyle \frac { 1 } { N } \sum _ { j = 1 } ^ { N } \frac { w ( { \pmb x } _ { t } , { \pmb x } _ { 0 , j } ) \nabla p ( { \pmb x } _ { t } | { \pmb x } _ { 0 , j } ) } { p ( { \pmb x } _ { t } | { \pmb x } _ { 0 , j } ) } } \end{array}\tag{25}
$$

With the above procedure, reverse sampling can be performed using the distribution $p ( \pmb { x } _ { t } | \pmb { x } _ { 0 } )$ . However, this introduces an unacceptable computational burden during inference. For example, $w ( \pmb { x } _ { t } , \pmb { x } _ { 0 } )$ and $\lambda ( \pmb { x } _ { t } )$ need to be calculated at every iteration. This remains time-consuming, even though the explicit expression of $p ( \pmb { x } _ { t } | \pmb { x } _ { 0 } )$ and the integral in (23) are available.

## B. Network amortization

To make large jump sampling more efficient, we use a neural network to amortize part of the computational burden and accelerate inference.

Based on the previous analysis, there are several possible learnable targets, including the density ratio $\begin{array} { r } { \frac { p ( \bar { \mathbf { x } } _ { t } + \pmb { v } ) } { p ( \pmb { x } _ { t } ) } , } \end{array}$ the prior distribution $p ( { \pmb x } _ { t } )$ , and the logarithmic density ratio $\begin{array} { r } { \log { \frac { p ( { \pmb x } _ { t } + { \pmb v } ) } { p ( { \pmb x } _ { t } ) } } } \end{array}$ . However, these choices lead to numerical challenges during both learning and sampling. For example, consider learning $\frac { p ( \pmb { x } _ { t } + \pmb { v } ) } { p ( \pmb { x } _ { t } ) }$ . Its advantage is that it is marginally trainable, so the learning target can be easily computed from its conditional version. However, large jump sampling requires numerical integration with respect to dv to obtain the overall jump rate. This is infeasible for high-dimensional data, since the integrand value must be obtained from the network output. Even though analytical expressions of $p ( \pmb { x } _ { t } | \pmb { x } _ { 0 , j } )$ and $\begin{array} { r } { \int _ { \| \pmb { v } \| > \epsilon } p ( \pmb { x } _ { t } + \pmb { v } | \pmb { x } _ { 0 , j } ) \nu _ { t } ( d \pmb { v } ) } \end{array}$ can be derived, 2N additions are still required at each iteration. In addition, numerical stability is not guaranteed when $p ( { \pmb x } _ { t } )$ is very small but $p ( \pmb { x } _ { t } + \pmb { v } )$ is relatively large. In this case, $\frac { p ( \pmb { x } _ { t } + \pmb { v } ) } { p ( \pmb { x } _ { t } ) }$ behaves like an outlier with a large amplitude.

Although this issue can be alleviated by applying the logarithm, log $\frac { p ( { \pmb x } _ { t } + { \pmb v } ) } { p ( { \pmb x } _ { t } ) }$ does not satisfy the conditions in Theorem 1. Moreover, the integration difficulty still remains. As for $p ( { \pmb x } _ { t } )$ , it requires the network to learn the absolute values of the global probability distribution, which is difficult, especially when D is large. Note that the method in [19] is equivalent to directly learning reverse sampling, and its large jump part is generated by the target distribution. This approach places a heavier burden on the network, making it more like a black box and difficult to control.

Therefore, we choose to learn the large jump rate, i.e., $\lambda ( \pmb { x } _ { t } )$ . This task is similar to parameter estimation for a specialized distribution. From this perspective, it is more learnable than $p ( { \pmb x } _ { t } )$ and $\frac { p ( \pmb { x } _ { t } + \pmb { v } ) } { p ( \pmb { x } _ { t } ) }$ . Moreover, although $\lambda ( \pmb { x } _ { t } )$ is the integral of the density ratio with respect to the Levy´ measure, it still belongs to the class of marginally trainable functions. Therefore, it can be efficiently trained using $\begin{array} { r } { \lambda ( \pmb { x } _ { t } | \pmb { x } _ { 0 } ) \triangleq \int _ { \| \pmb { v } \| > \epsilon } \frac { p \left( \pmb { x } _ { t } + \pmb { v } | \pmb { x } _ { 0 } \right) } { p \left( \pmb { x } _ { t } | \pmb { x } _ { 0 } \right) } \nu _ { t } ( d \pmb { v } ) } \end{array}$

Based on the above discussion, the next proposition provides the loss function for the jump part. For the diffusion part, the standard score matching framework can be directly used.

Proposition 2 (Loss function for jump activities): The loss function of the neural network for learning the overall jump rate is given as follows:

$$
\begin{array} { r l } & { \mathcal { L } ( \theta ) = \mathbb { E } _ { t \sim \mathcal { U } [ 0 , 1 ] , { \boldsymbol { x } } _ { t } \sim p ( { \boldsymbol { x } } _ { t } \mid { \boldsymbol { x } } _ { 0 } ) , { \boldsymbol { x } } _ { 0 } \sim p ( { \boldsymbol { x } } _ { 0 } ) } \left[ \| g ^ { \theta _ { 2 } } ( { \boldsymbol { x } } _ { t } , t ) \right. } \\ & { \qquad \left. - \ \lambda ( { \boldsymbol { x } } _ { t } | { \boldsymbol { x } } _ { 0 } ) \| ^ { 2 } \right] } \end{array}\tag{26}
$$

## C. Efficient calculation of large jump rate

The remaining two main challenges in bridging the gap between theoretical analysis and practical applications are how to calculate the target value in (26) and how to rapidly sample a large jump from the distribution $q ( \pmb { x } _ { t } , \pmb { v } )$ . Here, we first focus on the first problem.

So far, we have not imposed specific assumptions on the coefficients in (1), except that the drift coefficient depends on both the system state and the time index, while the scaling matrices of the Wiener process and the stable process depend only on the time index. For the design of a specific reverse sampling algorithm, its expressions need to be further specified.The principle is to make the learning process as simple as possible without sacrificing much generality. Similar to conventional diffusion models and SDEs driven only by the Wiener process [1], [3], we make the following assumptions.

Assumption 5: $\pmb { b } ( X _ { t } , t )$ is an affine function of $X _ { t } ,$ i.e., $\begin{array} { r c l } { { b ( X _ { t } , t ) } } & { { = } } & { { R X _ { t } \ + \ s , } } \end{array}$ which is similar to the Ornstein-Uhlenbeck (OU) process. Since $\boldsymbol { b } ( X _ { t } , t )$ is a timehomogeneous drift, we rewrite it as $\pmb { b } ( \pmb { X } _ { t } )$ in the sequel.

Assumption 6: For $\Phi _ { G } ( t )$ and $\Phi _ { S } ( t )$ , we restrict them to diagonal matrices, i.e., $\Phi _ { G } ( t ) = $ diag $( \sigma _ { G , 1 } ( t ) , \cdot \cdot \cdot , \sigma _ { G , D } ( t ) )$ and $\Phi _ { S } ( t ) = \mathrm { d i a g } ( \sigma _ { S , 1 } ( t ) , \cdot \cdot \cdot , \sigma _ { S , D } ( t ) )$

In the following, we rewrite $\boldsymbol { b } ( X _ { t } , t )$ as $ { \mathbf { \delta } } _ { b \left(  { \boldsymbol { X } } _ { t } \right) }$ because it is directly related only to $X _ { t } .$ Under these configurations, the distribution of $X _ { t }$ generated from a given $X _ { 0 }$ can be explicitly determined.

Proposition 3 (Distribution of $X _ { t } ) \colon$ Consider the forward SDE in (1). Let $b ( X _ { t } ) , \ \Phi _ { G } ( t ) ,$ , and $\Phi _ { S } ( t )$ satisfy Assumptions 5 and 6. Then, $X _ { t } = \mu ( t , \pmb { x } _ { 0 } ) + \pmb { G } ( t ) + \pmb { S } ( t )$ , where $G ( t ) \sim \mathcal { N } ( \mathbf { 0 } , \Sigma _ { G } )$ and $S ( t )$ follows an α-stable distribution with characteristic function $\phi _ { S } ( t )$ . Moreover,

$$
\pmb { \mu } ( t , \pmb { x } _ { 0 } ) = \exp ( R t ) \pmb { x } _ { 0 } + \exp ( R t ) \int _ { 0 } ^ { t } \exp ( - R l ) s d l\tag{27}
$$

$$
\Sigma _ { G } ( t ) = \int _ { 0 } ^ { t } \exp ( ( t - l ) R ) \Phi _ { G } ( l ) \Phi _ { G } ( l ) ^ { \top } \exp ( ( t - l ) R ^ { \top } ) d l\tag{28}
$$

$$
\phi _ { S } ( \pmb { u } , t ) = \mathrm { e x p } \bigg ( - \int _ { 0 } ^ { t } \left\| \Phi _ { S } ( \boldsymbol { l } ) ^ { \top } \mathrm { e x p } ( ( t - \boldsymbol { l } ) \boldsymbol { R } ^ { \top } ) \pmb { u } \right\| _ { 2 } ^ { \alpha } d \boldsymbol { l } \bigg ) _ { \Omega }
$$

Proof: The proof is relegated to Appendix B.

(29)

Corollary 1 (Distribution of $X _ { t }$ for special cases): Assume that R, $\Phi _ { G } ( t ) ,$ , and $\Phi _ { S } ( t )$ are all scaled identity matrices, i.e., $\begin{array} { c c c c c c } { { R } } & { { = } } & { { R _ { 0 } I _ { D } , } } & { { \Phi _ { G } ( t ) } } & { { = } } & { { \sigma _ { G } ( t ) I _ { D } } } \end{array}$ and $\Phi _ { S } ( t ) ~ = ~ \sigma _ { S } ( t ) I _ { D }$ . The other settings are the same as those in Proposition 3. Then, $\begin{array} { r c l } { \Sigma _ { G } ( t ) } & { = } & { \gamma _ { G } ( t ) ^ { 2 } I _ { D } } \end{array}$ where $\begin{array} { r l r } { \gamma _ { G } ( t ) } & { { } = } & { \sqrt { \int _ { 0 } ^ { t } \exp ( 2 R _ { 0 } ( t - l ) ) \sigma _ { G } ( l ) ^ { 2 } d l } . } \end{array}$ Moreover, $\begin{array} { r c l } { \phi _ { S } ( { \bf u } , t ) } & { = } & { \displaystyle \exp ( - \gamma _ { \alpha } ( t ) ^ { \alpha } \| { \bf u } \| _ { 2 } ^ { \alpha } ) } \end{array}$ , where $\gamma _ { \alpha } ( t ) \quad = $ $\begin{array} { r } { ( \int _ { 0 } ^ { t } | \sigma _ { S } ( l ) | ^ { \alpha } \exp ( \alpha R _ { 0 } ( t - l ) ) d l ) ^ { \frac { 1 } { \alpha } } } \end{array}$

Remark 3: From Proposition 3, it can be observed that the final distribution of $X _ { T }$ is still related to $X _ { 0 }$ . In practice, we can choose T and $R < 0$ such that $\exp ( R T ) X _ { 0 } \approx 0 .$ In this case, the original distribution of the inverse sampling can be set to the mixed distribution composed of Gaussian distribution and α-stable distribution.

Proposition 3 and Corollary 1 imply that if $R , ~ \Phi _ { G } ( t )$ and $\Phi _ { S } ( t )$ are diagonal matrices with different diagonal elements, the isotropic property of the distribution of $X _ { t }$ will be destroyed. This makes the following derivations much more cumbersome. In the remainder of this paper, we use the assumptions in Corollary 1 to facilitate the design of the reverse sampler.

Recall that the first challenge comes from the integral operation. During training, performing high-dimensional numerical integration for every data sample is time-consuming. Our goal is to derive an analytical expression for $\int _ { \| \pmb { v } \| > \epsilon } p ( \pmb { x } _ { t } +$ ${ \pmb v } | { \pmb x } _ { 0 } ) \nu _ { t } ( d { \pmb v } )$ . According to Proposition 3, $p ( \pmb { x } _ { t } | \ddot { \pmb { x } } _ { 0 } )$ is the distribution of an RV composed of a bias term, a Gaussian RV, and a stable RV, which are mutually independent. Due to the general lack of an analytical PDF for the α-stable distribution, it is challenging to obtain an explicit expression for $p ( \pmb { x } _ { t } | \pmb { x } _ { 0 } )$ . Therefore, PDF approximation is needed to facilitate the following analysis.

Let $X _ { B } \sim \mathcal { N } ( 0 , 2 \gamma _ { q } ^ { 2 } )$ and $X _ { S } \sim \mathcal { S } ( \alpha , 0 , \gamma _ { s } , 0 )$ be mutually independent. Denote the PDFs of $X _ { B }$ and $X _ { S }$ by $f _ { X _ { B } } ( x )$ and $f _ { X _ { S } } ( x )$ , respectively. Then, the exact PDF of $X _ { M } \triangleq X _ { B } + X _ { S }$ equals the convolution of $f _ { X _ { B } } ( x )$ and $f _ { X _ { S } } ( x )$ . Unfortunately, there is no general explicit expression for $f _ { X _ { M } } ( x )$ . In [23], an approximate PDF is proposed to describe the statistical model of $X _ { M }$ . It can be regarded as a weighted sum of a Gaussian kernel and a tail component. Specifically,

$$
f _ { X _ { M } } ( x ) \approx \tilde { f } _ { X _ { M } } ( x ) = \frac { 1 } { k _ { 1 } } \biggl [ c _ { 1 } g _ { 0 } \exp { \biggl ( - \frac { x ^ { 2 } } { 4 \gamma _ { g } ^ { 2 } } \biggr ) } + \frac { \alpha \gamma _ { s } C _ { \alpha } } { c _ { 2 } + | x | ^ { \alpha + 1 } } \biggr ] ,\tag{30}
$$

where $k _ { 1 }$ is the normalization factor. The parameter $c _ { 2 } \geq 0$ is used to avoid singularity at the origin and control the shape of the tail component. The parameters $\gamma _ { g }$ and $\gamma _ { s }$ represent the scaling parameters of the Gaussian kernel and the tail component, respectively. The quantities $g _ { 0 }$ and $C _ { \alpha }$ are expressed by the above parameters in closed form. It has been shown in [23] that $\tilde { f } _ { X _ { M } } ( { \pmb x } )$ can accurately model $X _ { M }$ under various scenarios.

The tail component in (30) is derived from the asymptotic behavior of the α-stable distribution [23], [24]. Following a similar procedure, (30) can also be extended to highdimensional cases [25]. For example, let $\pmb { X } _ { B } \sim \mathcal { N } ( \mathbf { 0 } , \sigma ^ { 2 } \pmb { I } )$ $\ b X _ { S } = \sqrt { \ b A } \ b X _ { G }$ , and $X _ { M } = X _ { B } + X _ { S }$ . Here, we consider the sub-Gaussian representation of the multivariate isotropic stable distribution, where A follows a right-skewed α-stable distribution [24]. Note that the characteristic function of $X _ { M }$ is $\phi _ { X _ { M } } ( l ) = \exp ( - \gamma _ { g } ^ { 2 } \| l \| ^ { 2 } - \gamma _ { s } ^ { \alpha } \| l \| ^ { \alpha } )$ . Then, the PDF of $X _ { M }$ can be similarly approximated as [25]

$$
\begin{array} { l } { { \displaystyle { \widetilde { f } } _ { X _ { M } } ( \pmb { x } ) = \frac { 1 } { k _ { 2 } } \biggl [ c _ { 1 } g _ { 0 } \exp \bigg ( - \frac { \| \pmb { x } \| ^ { 2 } } { 4 \gamma _ { g } ^ { 2 } } \bigg ) } } \\ { { \displaystyle ~ + \frac { \alpha 2 ^ { \alpha - 1 } \Gamma ( \frac { \alpha + D } { 2 } ) \gamma _ { s } ^ { \alpha } } { \pi ^ { D / 2 } \Gamma ( 1 - \frac { \alpha } { 2 } ) } ( c _ { 2 } + \| \pmb { x } \| ^ { 2 } ) ^ { - \frac { \alpha + D } { 2 } } \biggr ] } } \end{array}\tag{31}
$$

where $k _ { 2 }$ denotes the normalization factor.

With $( 3 1 )$ , the target in (26) can be further derived. Since $p ( \pmb { x } _ { t } | \pmb { x } _ { 0 , j } )$ is independent of dv, we focus only on $\begin{array} { r } { \int _ { \| \pmb { v } \| > \epsilon } p ( \pmb { x } _ { t } \ + \ \pmb { v } | \pmb { x } _ { 0 } ) \nu _ { t } ( d \pmb { v } ) } \end{array}$ in the sequel. For notational simplicity, we derive explicit expressions for the following integrals:

$$
P _ { 1 } ( \epsilon ) \triangleq \int _ { \| v \| > \epsilon } \exp ( - \| v + \mu \| ^ { 2 } ) \nu _ { t } ( d v )\tag{32}
$$

$$
P _ { 2 } ( \epsilon ) \triangleq \int _ { \| v \| > \epsilon } ( c _ { 2 } + \| v + \pmb { \mu } \| ^ { 2 } ) ^ { - \frac { \alpha + D } { 2 } } \nu _ { t } ( d v )\tag{33}
$$

Note that $\pmb { \mu }$ is introduced only to characterize the mismatch between $p ( \pmb { x } _ { t } + \pmb { v } | \pmb { x } _ { 0 } )$ and the Levy measure. Moreover, the´ scaling parameter $\gamma _ { g }$ is omitted in (32) for brevity. It does not denote the mean of $X _ { t } .$ Direct numerical integration is infeasible in high-dimensional settings. To address this issue, we transform $P _ { 1 } ( \epsilon )$ and $P _ { 2 } ( \epsilon )$ into polar coordinates as follows:

$$
\begin{array} { c } { { P _ { 1 } ( \epsilon ) = \displaystyle \frac { 2 \pi ^ { \frac { D - 1 } { 2 } } C _ { D , \alpha , \sigma _ { S } } } { \Gamma ( \frac { D - 1 } { 2 } ) } \int _ { \epsilon } ^ { + \infty } \int _ { - 1 } ^ { 1 } \exp ( - ( r ^ { 2 } + \| \pmb { \mu } \| ^ { 2 } } } \\ { { + 2 r z \| \pmb { \mu } \| ) ) r ^ { - \alpha - 1 } ( 1 - z ^ { 2 } ) ^ { \frac { D - 3 } { 2 } } d z d r } } \end{array}\tag{34}
$$

$$
\begin{array} { c } { { P _ { 2 } ( \epsilon ) = \displaystyle \frac { 2 \pi ^ { \frac { D - 1 } { 2 } } C _ { D , \alpha , \sigma _ { S } } } { \Gamma ( \frac { D - 1 } { 2 } ) } \int _ { \epsilon } ^ { + \infty } \int _ { - 1 } ^ { 1 } ( c _ { 2 } + r ^ { 2 } + \| \pmb { \mu } \| ^ { 2 } } } \\ { { + 2 r z \| \pmb { \mu } \| ) ^ { - \frac { \alpha + D } { 2 } } r ^ { - \alpha - 1 } ( 1 - z ^ { 2 } ) ^ { \frac { D - 3 } { 2 } } d z d r } } \end{array}\tag{35}
$$

which does not admit a closed-form expression in general. However, the original integral has been significantly simplified into a two-dimensional integral. In fact, the inner integral with respect to z in (32) and (33) can be further simplified. First, we need the following two integral identities [26]:

$$
\begin{array} { l } { \displaystyle \int _ { - 1 } ^ { 1 } \exp ( a x ) ( 1 - x ^ { 2 } ) ^ { p } d x } \\ { \displaystyle \qquad = \sqrt { \pi } \Gamma ( p + 1 ) ( 2 a ^ { - 1 } ) ^ { \frac { p + 1 } { 2 } } I _ { \frac { p + 1 } { 2 } } ( a ) , } \end{array}\tag{36}
$$

$$
\begin{array} { c } { { \displaystyle \int _ { 0 } ^ { 1 } x ^ { a - 1 } ( 1 - x ) ^ { b - a - 1 } ( 1 - z x ) ^ { - c } d x } } \\ { { = B ( a , b - a ) { } _ { 2 } F _ { 1 } ( c , a ; b ; z ) , b > a > 0 } } \end{array}\tag{37}
$$

where $I _ { a } ( x )$ denotes the modified Bessel function of the first kind and $B ( \cdot )$ denotes the Beta function. For $P _ { 1 } ( \epsilon )$ , applying (36) yields

$$
\begin{array} { l } { { \displaystyle P _ { 1 } ( \epsilon ) = 2 \pi ^ { \frac { D } { 2 } } C _ { D , \alpha , \sigma _ { S } } \| { \pmb { \mu } } \| ^ { \frac { 2 - D } { 2 } } \exp ( - \| { \pmb { \mu } } \| ^ { 2 } ) } } \\ { { \displaystyle ~ \times \int _ { \epsilon } ^ { + \infty } \exp ( - r ^ { 2 } ) r ^ { - \alpha - \frac { D } { 2 } } I _ { \frac { D - 2 } { 2 } } ( 2 r \| { \pmb { \mu } } \| ) d r } } \end{array}\tag{38}
$$

For $P _ { 2 } ( \epsilon )$ , we use the variable transformation $x = 2 t - 1$ After some manipulations, (39) can be obtained. In this way, calculating the total jump rate $\lambda ( \pmb { x } _ { t } | \pmb { x } _ { 0 , j } )$ reduces to evaluating a one-dimensional integral, which significantly decreases the computational complexity compared to D-dimensional integrals.

## D. High-dimensional sampling

As for the second problem, numerical sampling algorithms such as Hamiltonian Monte Carlo (HMC) can be applied, as the PDF and its gradient information are available. However, reverse sampling of the SDE is still slow due to the cold-start problem. Specifically, for each reverse sampling iteration with sufficiently small $\Delta t .$ , only one sample is needed for a large jump if it occurs. However, the target sampling distribution varies with t. In this case, the HMC sampler should be “retrained” repeatedly to reach the steady state of the dynamic system, which is quite inefficient.

Then, we design the fast sampling method for the large jump. Recall that the normalized target distribution is

$$
\begin{array} { c l } { \displaystyle \tilde { q } ( \pmb { x } _ { t } , \pmb { v } ) \triangleq \frac { p ( \pmb { x } _ { t } + \pmb { v } ) \nu _ { t } ( d \pmb { v } ) } { Q _ { t , \epsilon } } } \\ { \displaystyle \approx \frac { \sum _ { j = 1 } ^ { N } p ( \pmb { x } _ { t } + \pmb { v } | \pmb { x } _ { 0 , j } ) \nu _ { t } ( d \pmb { v } ) } { N Q _ { t , \epsilon } } } \end{array}\tag{40}
$$

where we define $\begin{array} { r } { Q _ { t , \epsilon } = \int _ { \| v \| > \epsilon } p ( \pmb { x } _ { t } + \pmb { v } ) \nu _ { t } ( d \pmb { v } ) } \end{array}$ . Note that $p ( { \pmb x } _ { t } )$ is omitted because it is not related to v, and our objective is to generate a sample from the distribution with respect to v.

Different from the one-dimensional case, we sample from $\tilde { q } ( \boldsymbol { x } _ { t } , \boldsymbol { v } )$ under polar coordinates. First, we rewrite $\tilde { q } ( \boldsymbol { x } _ { t } , \boldsymbol { v } )$

$$
P _ { 2 } ( \epsilon ) = \frac { C _ { D , \alpha , \sigma _ { S } } \pi ^ { \frac { D - 1 } { 2 } } 2 ^ { D - 1 } B ( \frac { D - 1 } { 2 } , \frac { D - 1 } { 2 } ) } { \Gamma ( \frac { D - 1 } { 2 } ) } \int _ { \epsilon } ^ { + \infty } \frac { r ^ { - \alpha - 1 } { _ 2 F _ { 1 } } ( \frac { \alpha + D } { 2 } , \frac { D - 1 } { 2 } ; D - 1 ; - \frac { 4 r \| \mu \| } { 2 ( r ^ { 2 } + r ^ { 2 } + \| \mu \| ) ^ { 2 } } - 2 r \| \mu \| ) } { ( c _ { 2 } + r ^ { 2 } + \| \mu \| ^ { 2 } - 2 r \| \mu \| ) ^ { \frac { \alpha + D } { 2 } } } d r\tag{39}
$$

as $\tilde { q } ( \pmb { x } _ { t } , r , z , \theta )$ , where $r \triangleq \| \pmb { v } \| , z \triangleq \frac { \langle \pmb { \mu } , \pmb { v } \rangle } { \| \pmb { \mu } \| \| \pmb { v } \| }$ , and $\theta \in \mathcal { R } ^ { D - 2 }$ represents the remaining angular variables. Then, $\tilde { q } ( \boldsymbol { x } _ { t } , \boldsymbol { v } )$ can be rewritten as in (44), where $\pmb { \mu }$ is defined as before and incorporates the information of $\mathbf { \nabla } _ { \mathbf { x } _ { t } . }$ We then need to sample $r ,$ $z ,$ and θ from the joint distribution $\tilde { q } ( \pmb { x } _ { t } , r , z , \theta )$ . Given r and $z , \theta$ can be generated uniformly on the sphere orthogonal to the direction represented by z. Therefore, θ does not explicitly appear in the PDF. Based on this observation, we omit θ in $\tilde { q } ( \pmb { x } _ { t } , r , z , \theta )$ for simplicity.

Note that the derivations of the marginal distributions are the same as those of $P _ { 1 } ( \epsilon )$ and $P _ { 2 } ( \epsilon )$ . Hence, the previous lookup tables can be directly used for fast sampling based on inverse CDF methods. In other words, we can first sample r from the marginal distribution $\tilde { q } ( \boldsymbol { x } _ { t } , \boldsymbol { r } )$ and then sample z from $\tilde { q } ( \boldsymbol { z } | \boldsymbol { x } _ { t } , \boldsymbol { r } )$ . To distinguish the distribution under polar coordinates from $\tilde { q } ( \boldsymbol { x } _ { t } , \boldsymbol { v } )$ , we denote the CDFs of $\tilde { q } ( { \boldsymbol x } _ { t } , { \boldsymbol r } )$ and $\tilde { q } ( \boldsymbol { z } | \boldsymbol { x } _ { t } , \boldsymbol { r } )$ by $F _ { \tilde { q } , ( r , t ) } ( \cdot )$ and $F _ { \tilde { q } , z | ( r , t ) } ( \cdot )$ , respectively, in the sequel.

However, even if $P _ { 1 } ( x )$ and $P _ { 2 } ( x )$ with $x \in [ \epsilon , M ]$ and a large constant M can be precomputed and stored in the lookup table, the inference stage is still time-consuming. This is because, for each pair $( r , F _ { \tilde { q } , ( r , t ) } ( \cdot ) )$ , the corresponding conditional pair $( r , F _ { \tilde { q } ( | ) , ( r , t ) } ( \cdot ) )$ needs to be obtained N times. This complexity can be considerably reduced by another simple sampling step. Indeed, $Q _ { t , \cdot }$ <sub>ϵ</sub> can be decomposed as

$$
Q _ { t , \epsilon } = \frac { 1 } { N } \sum _ { j = 1 } ^ { N } \int _ { \| v \| > \epsilon } p ( x _ { t } + v | x _ { 0 , j } ) \nu _ { t } ( d v ) \triangleq \frac { 1 } { N } \sum _ { j = 1 } ^ { N } Q _ { t , \epsilon , j }\tag{41}
$$

Based on $( 4 0 ) , \tilde { q } ( \pmb { x } _ { t } , \pmb { v } )$ can be rewritten as follows:

$$
\begin{array} { r l } & { \tilde { q } ( { \pmb x } _ { t } , { \pmb v } ) = \displaystyle \frac { \sum _ { j = 1 } ^ { N } p ( { \pmb x } _ { t } + { \pmb v } | { \pmb x } _ { 0 , j } ) \nu _ { t } ( d { \pmb v } ) } { \displaystyle \sum _ { j = 1 } ^ { N } Q _ { t , \epsilon , j } } } \\ & { ~ = \displaystyle \frac { \sum _ { j = 1 } ^ { N } Q _ { t , \epsilon , j } \tilde { q } ( { \pmb x } _ { t } , { \pmb v } | { \pmb x } _ { 0 , j } ) } { \displaystyle \sum _ { j = 1 } ^ { N } Q _ { t , \epsilon , j } } } \\ & { ~ \triangleq \displaystyle \sum _ { j = 1 } ^ { N } g _ { Q } ( { \pmb x } _ { 0 , j } | { \pmb x } _ { t } , t ) \tilde { q } ( { \pmb x } _ { t } , { \pmb v } | { \pmb x } _ { 0 , j } ) } \end{array}\tag{42}
$$

where we define the weighted distribution $g _ { Q } ( \pmb { x } | \pmb { x } _ { t } , t )$ as

$$
g _ { Q } ( { \pmb x } | { \pmb x } _ { t } , t ) = \sum _ { j = 1 } ^ { N } \frac { 1 _ { \{ { \pmb x } _ { 0 , j } \} } ( { \pmb x } ) Q _ { t , { \epsilon } , j } } { \sum _ { j = 1 } ^ { N } Q _ { t , { \epsilon } , j } }\tag{43}
$$

In this case, $\tilde { q } ( \boldsymbol { x } _ { t } , \boldsymbol { v } )$ can be regarded as a mixture distribution of the conditional distributions $\tilde { q } ( \boldsymbol { x } _ { t } , \boldsymbol { v } | \boldsymbol { x } _ { 0 , j } )$ . Then, $\tilde { q } ( \boldsymbol { x } _ { t } , \boldsymbol { v } )$ is equivalent to a distribution controlled by an underlying Markov process. Based on the above observation, we first randomly choose a data sample $\pmb { x } _ { 0 } ^ { * } \in \{ \pmb { x } _ { 0 , j } , j = 1 , \cdots , N \}$ according to the distribution $g _ { Q } ( \pmb { x } | \pmb { x } _ { t } , t )$ . Then, we generate a sample from the distribution conditioned on $\pmb { x } _ { 0 } ^ { * }$ . In this way, we only need to focus on the conditional distribution $\tilde { q } ( \boldsymbol { x } _ { t } | \boldsymbol { x } _ { 0 } ^ { * } )$ and obtain the pair $( r , F _ { \tilde { q } ( | ) , ( r , t ) } ( \cdot ) )$ only once. This is much faster than directly sampling from $\tilde { q } ( \boldsymbol { x } _ { t } , \boldsymbol { v } )$

To obtain the complete $g _ { Q } ( \pmb { x } | \pmb { x } _ { t } , t )$ , all $Q _ { t , \epsilon , j }$ with $j ~ =$ $1 , \cdots , N$ should be computed. This is equivalent to searching the lookup table N times. In fact, many $Q _ { t , \epsilon , j }$ may be rather small, and their influence can be neglected. Therefore, instead of constructing the complete $g _ { Q } ( \pmb { x } | \pmb { x } _ { t } , t )$ , we truncate $g _ { Q } ( \pmb { x } | \pmb { x } _ { t } , t )$ by choosing the K samples with the largest approximate mixture weights, or equivalently the largest $Q _ { t , \epsilon , j }$ in the prior sampler. This operation is reasonable because the contribution of $\pmb { x } _ { 0 , j }$ is determined by the overlap between the conditional density $p ( \pmb { x } _ { t } + \pmb { v } | \pmb { x } _ { 0 , j } )$ and the truncated Levy mea-´ sure. In the isotropic case, $\nu _ { t } ( \dot { d } v ) \propto \lVert v \rVert ^ { - D - \alpha } d v$ . After truncation, it favors jumps with $\lVert \boldsymbol { v } \rVert$ close to ϵ. Therefore, $Q _ { t , \epsilon , j }$ is large only when the high-density region of $p ( \pmb { x } _ { t } + \pmb { v } | \pmb { x } _ { 0 , j } )$ intersects the Levy-favored shell around´ $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { t } }$ . This motivates a top-K approximation that keeps only the candidates with the largest approximate $Q _ { t , \epsilon , j }$ and renormalizes their weights. For convenience, we denote the index set by $\kappa$ and the truncated version of $g _ { Q } ( \pmb { x } | \pmb { x } _ { t } , t )$ by $\tilde { g } _ { Q } ( { \pmb x } )$ .

Remark 4: If we choose the largest K elements from all $Q _ { t , \epsilon , j }$ via the trivial traversal, the complexity still remains $O ( N )$ . To solve this problem, more advanced and efficient ordering algorithms can be utilized, such as approximate nearest-neighbor (ANN) methods.

In addition to the inverse CDF method, rejection sampling can also be efficient for sampling the radial variable r based on (44). This converts the original D-dimensional sampling problem into two one-dimensional sampling steps and one remaining direction sampling step. Since (44) has an analytical PDF expression with explicit asymptotic behavior, the proposal distribution for sampling r can be chosen as a Student’s t-distribution with $2 \alpha + D$ degrees of freedom. This choice matches the tail behavior of the marginal distribution induced by (44). Note that marginalization with respect to z does not change the tail order of the distribution in terms of r. Its closed-form expression has been analyzed in $( 3 4 ) { \sim } ( 3 9 )$ For z, rejection sampling is not necessary, since the proposal distribution is not straightforward to design due to the bounded domain, and the corresponding lookup table is quite small.

Finally, the overall training procedure, the large jump sampling method, and the complete reverse sampling algorithm are summarized in Algorithm 1, Algorithm 2, and Algorithm 3, respectively. Note that complete reverse sampling starts from samples generated from the terminal prior distribution $p _ { t p } ( \pmb { x } )$ . In our setting, $p _ { t p } ( \pmb { x } )$ is the convolution of a multivariate Gaussian distribution and a stable distribution. The corresponding samples can be generated separately from these two distributions. Here, if we set $\mathbf { \boldsymbol { s } } \neq \mathbf { \boldsymbol { 0 } }$ , the mean of $p _ { t p } ( { \pmb x } )$ is $- \frac { \pmb { s } } { R _ { 0 } }$ . Without loss of generality, we can also set $\mathbf { \boldsymbol { s } } = \mathbf { 0 }$ so that $\bar { p _ { t p } } ( { \pmb x } )$ is symmetric about the origin.

$$
\begin{array} { r l } & { \widetilde { q } ( x _ { t } , r , z , \theta ) \propto \left( \frac { c _ { 1 } g _ { 0 } } { k _ { 2 } } \exp \left( - \frac { 1 } { 4 \gamma _ { g } ^ { 2 } } ( r ^ { 2 } + \| \mu \| ^ { 2 } + 2 r z \| \mu \| ) \right) + \frac { \alpha 2 ^ { \alpha - 1 } \Gamma \left( \frac { \alpha + D } { 2 } \right) \gamma _ { s } ^ { \alpha } } { k _ { 2 } \pi ^ { D / 2 } \Gamma \left( 1 - \frac { \alpha } { 2 } \right) } ( c _ { 2 } + r ^ { 2 } + \| \mu \| ^ { 2 } + 2 r z \| \mu \| ) ^ { - \frac { \alpha + D } { 2 } } \right) } \\ & { \qquad \times r ^ { - \alpha - 1 } ( 1 - z ^ { 2 } ) ^ { \frac { D - 3 } { 2 } } , r > \epsilon , z \in [ - 1 , 1 ] , } \end{array}\tag{44}
$$

Remark 5: In Algorithm 2, we only consider the case where $\mu _ { t } \ \neq \ \mathbf { 0 }$ and $\begin{array} { l } { { D } } \end{array} \geq \ 3 .$ . If ${ \pmb \mu } _ { t } \ = \ { \bf 0 } .$ , the conditional density is radially symmetric. Therefore, we only need to sample r. When $D = 1 , 2$ , the sampling procedure can be significantly simplified. For example, when $D = 1$ , the orthogonal direction does not exist, and only r needs to be sampled.

Remark 6: We emphasize that the generator-based reverse characterization is stated at the level of Levy-driven Markov´ processes, whereas the computational sampler developed in this paper is derived under additional structural assumptions. In particular, the practical algorithm focuses on isotropic linear Levy SDEs with symmetric ´ α-stable jump measures, for which the conditional transition density, jump rate approximation, and polar coordinate sampling procedure can be implemented efficiently. This distinction allows us to retain the general insight provided by the reversed generator while avoiding an overstatement of the scope of the proposed numerical sampler.

Algorithm 1 Training algorithm   
Input: Dataset $\pmb { x } _ { 0 , j } , j \ = \ 1 , \cdots , N$ and the training epoch   
$N _ { \mathrm { e p o c h } } .$   
Output: Well-trained networks $s ^ { \theta _ { 1 } } ( \cdot )$ and $g ^ { \theta _ { 2 } } ( \cdot )$   
1: # Training of $s ^ { \theta _ { 1 } } ( \cdot ) \colon$   
2: Use the dataset to optimize $\theta _ { 1 }$ via standard denoising   
score matching.   
3: # Training of $g ^ { \theta _ { 2 } } ( \cdot ) \colon$   
4: $k \gets 1 .$   
5: while $k \leq N _ { \mathrm { e p o c h } }$ do   
6: Sample $t \sim \mathcal { U } ( 0 , 1 )$   
7: Calculate the bias and scaling parameters of the   
Gaussian and stable RVs used to describe $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { t } }$ based on   
(27)∼(29).   
8: Calculate the target based on (26), (38), and (39).   
9: Optimize $\theta _ { 2 }$ by stochastic gradient descent based on   
(26).   
10: $k  k + 1 .$   
11: end while

It should be remarked that for the large jump component, we should theoretically sample the number of jumps from $N _ { t } ^ { > \epsilon } \sim$ Poisson $( g ^ { \theta _ { 2 } } ( \mathbf { x } _ { t } , t ) \Delta t )$ . If $N _ { t } ^ { > \epsilon } > 0$ , independently generate jump amplitudes $V _ { t , \ell } ^ { > \epsilon } , \ell = 1 , \dots , N _ { t } ^ { > \epsilon }$ . Then the large jump contribution is $\begin{array} { r } { X _ { t , \Delta t } ^ { > \epsilon } = \sum _ { \ell = 1 } ^ { N _ { t } ^ { > \epsilon } } V _ { t , \ell } ^ { > \epsilon } } \end{array}$ . In practice, one may further approximate the compound Poisson update by allowing at most one large jump within each time step. Let $\Lambda ( { \pmb x } _ { t } , t ) =$ $\lambda ( \pmb { x } _ { t } , t ) \Delta t$ . The probability of two or more large jumps in the same interval is

$$
\begin{array} { r l r } {  { \operatorname* { P r } ( N _ { t } ^ { > \epsilon } \geq 2 ) = 1 - e ^ { - \Lambda ( { \pmb x } _ { t } , t ) } ( 1 + \Lambda ( { \pmb x } _ { t } , t ) ) } } \\ & { } & { \leq \frac { \Lambda ( { \pmb x } _ { t } , t ) ^ { 2 } } { 2 } } \end{array}\tag{45}
$$

Algorithm 2 Sampling algorithm for large jumps   
Input: ${ \mathbf { } } x _ { t } , \ x _ { 0 } ^ { * } ,$ , lookup tables containing $( r , F _ { \tilde { q } , ( r , t ) } ( \cdot ) )$ and   
$( z , { \cal F } _ { \tilde { q } , z | ( r , t ) } ( \cdot ) )$   
Output: Large jump sample $\omega .$   
1: $\pmb { \mu } _ { t } \gets \pmb { x } _ { t } - \pmb { \mu } ( t , \pmb { x } _ { 0 } ^ { * } )$ , where $\mu ( t , \pmb { x } _ { 0 } ^ { * } )$ is given in (27).   
2: $\begin{array} { r } { \breve { \pmb { \mu } } _ { t } \gets \frac { \pmb { \mu } _ { t } } { \lVert \pmb { \mu } _ { t } \rVert } . } \end{array}$   
3: Sample $u _ { 1 } , u _ { 2 } \sim \mathcal { U } ( 0 , 1 )$ and ${ \pmb u } _ { 3 } \sim \mathcal { N } ( { \bf 0 } , { \pmb I } _ { D } )$   
4: $r  F _ { \tilde { q } , ( r , t ) } ^ { - 1 } ( u _ { 1 } )$ , or equivalently, apply rejection sampling   
based on $\tilde { q } ( { \boldsymbol x } _ { t } , { \boldsymbol r } )$ with Student’s t-distribution as the   
proposal distribution.   
5: $z \gets F _ { \tilde { q } , z | ( r , t ) } ^ { - 1 } ( u _ { 2 } ) .$   
6: $\begin{array} { r } { \pmb { u } _ { 3 }  \pmb { u } _ { 3 } - \langle \check { \pmb { \mu } } _ { t } , \pmb { u } _ { 3 } \rangle \breve { \pmb { \mu } } _ { t } . } \end{array}$   
7: $\begin{array} { r } { \breve { \pmb u } _ { 3 } \gets \frac { \pmb u _ { 3 } } { \parallel \pmb u _ { 3 } \parallel } . } \end{array}$   
8: $\omega \gets r z \breve { \pmb { \mu } } _ { t } + r \sqrt { 1 - z ^ { 2 } } \breve { \pmb { u } } _ { 3 }$

Therefore, the single jump approximation introduces an additional error of order

$$
E _ { \mathrm { m u l t i } } = O \left( \operatorname* { s u p } _ { t , { \boldsymbol { \mathbf { x } } _ { t } } } \lambda ( \boldsymbol { \mathbf { x } } _ { t } , t ) ^ { 2 } \Delta t ^ { 2 } \right) ,\tag{46}
$$

provided that $\lambda ( \pmb { x } _ { t } , t ) \Delta t$ is uniformly small.

## E. Approximate observation-guided sampler

Posterior sampling for the diffusion process can be derived straightforwardly based on Bayes’ rule, i.e., ∇ log $p ( \pmb { x } _ { t } | \pmb { y } ) =$ $\nabla \log p ( \pmb { y } | \pmb { x } _ { t } ) + \nabla \log p ( \pmb { x } _ { t } )$ . The score function is approximated by the output of the neural network, and the likelihood is obtained by approximating $p ( \pmb { y } | \pmb { x } _ { t } )$ with $p ( \pmb { y } | E [ \hat { \pmb { x } } _ { 0 } | \pmb { x } _ { t } ] )$ [27].

However, a similar strategy is not directly applicable to the jump component because the reversed jump kernel involves a nonlocal density ratio. Exact posterior sampling for Levy-´ driven SDEs is therefore considerably more challenging. In principle, the large jump rate should be replaced by the posterior large jump rate, which requires likelihood-reweighted integration over the empirical mixture at each reverse step. This computation is rather expensive, especially because the integral cannot be simplified in general.

In this work, we therefore use the prior-trained rate network as an amortized proposal rate for large jump events in the posterior sampler. This approximation means that the prior rate network is not interpreted as the exact posterior rate. Instead, it provides a computationally efficient proposal mechanism for deciding when nonlocal transitions are proposed. The observation information is then incorporated into the large jump amplitude and direction through likelihood-reweighted empirical mixture weights. This design prioritizes the posterior correction of the jump destination, which has a direct impact on the nonlocal transition once a large jump is triggered, while avoiding the high cost of recomputing the exact posterior jump rate at every reverse step.

Algorithm 3 Total reverse sampling algorithm   
Input: $\begin{array} { r l r } { { \bf x } _ { T , j } } & { { } \sim } & { p _ { t p } ( { \bf x } ) , j \mathrm { ~  ~ \Omega ~ } = \mathrm { ~  ~ \Omega ~ } 1 , \cdots , N . } \end{array}$ well-trained   
networks $s ^ { \theta _ { 1 } } ( \cdot )$ and $g ^ { \theta _ { 2 } } ( \cdot )$ , weighted distribution   
$g _ { Q } ( \pmb { x } | \pmb { x } _ { t } , t )$ , lookup tables containing $( r , F _ { \tilde { q } , ( r , t ) } ( \cdot ) )$ and   
$( z , { \cal F } _ { \tilde { q } , z | ( r , t ) } ( \cdot ) )$ , and time step size $\Delta t$   
Output: Reversed samples $\pmb { x } _ { 0 , j } , j = 1 , \cdots , N .$   
1: $j  1 .$   
2: while $j \leq N$ do   
3: $t \gets T .$   
4: while $t > 0$ do   
5: # Score-based update:   
6: Sample $\pmb { \xi } _ { 1 } , \pmb { \xi } _ { 2 } \sim \mathcal { N } ( \mathbf { 0 } , \pmb { I } _ { D } )$   
7: $\tilde { x } _ { t , j } \gets x _ { t , j } + ( - b ( x _ { t , j } ) + \Sigma ( t ) s ^ { \theta _ { 1 } } ( x _ { t , j } , t ) ) \Delta t +$   
$\Phi _ { G } ( t ) \sqrt { \Delta t } \xi _ { 1 }$   
8: # Large jump generation:   
9: $\lambda ( \pmb { x } _ { t , j } )  g ^ { \theta _ { 2 } } ( \pmb { x } _ { t , j } , t ) .$   
10: Sample $\beta \sim \mathcal { U } ( 0 , 1 ) .$   
11: if $\beta < 1 - \exp ( - \lambda ( \mathbf { x } _ { t , j } ) \Delta t )$ then   
12: Randomly choose a sample $\pmb { x } _ { 0 } ^ { * }$ from the dataset   
according to the truncated weighted distribution $\tilde { g } _ { Q } ( \pmb { x } )$   
13: Obtain a jump sample ω from Algorithm 2 using   
$\boldsymbol { \mathbf { \mathit { x } } } _ { t , j } , \boldsymbol { \mathbf { \mathit { x } } } _ { 0 } ^ { * } ,$ , and the lookup tables.   
14: else   
15: $\omega  0 .$   
16: end if   
17: $\tilde { \mathbf { x } } _ { t , j } \gets \tilde { \mathbf { x } } _ { t , j } + \omega .$   
18: # Small jump generation:   
19: Calculate the drift $\pmb { b } _ { t , \Delta t } ^ { \le \epsilon }$ in (8) and the scaling   
parameter $\Sigma _ { t , \Delta t } ^ { \le \epsilon }$ in (15).   
20: $\begin{array} { r } { \pmb { x } _ { t - \Delta t , j }  \tilde { \pmb { x } } _ { t , j } + \pmb { b } _ { t , \Delta t } ^ { \le \epsilon } \Delta t + \sqrt { \Delta t } ( \Sigma _ { t , \Delta t } ^ { \le \epsilon } ) ^ { \frac { 1 } { 2 } } \pmb { \xi } _ { 2 } . } \end{array}$   
21: $t \gets t - \Delta t .$   
22: end while   
23: $j  j + 1$   
24: end while

In the experiments, we will evaluate its effect by comparing the prior proposal rate with the likelihood-reweighted posterior rate approximation and by examining the sensitivity of the final estimation performance to the jump-rate choice. In the following proposition, we analyze how the observation modifies the empirical mixture weights used for large jump amplitude sampling.

Proposition 4: Let y be the observations used for posterior sampling of the forward SDE (1). Assume that the observation depends on the clean state $X _ { 0 }$ only, i.e., $p _ { { \pmb Y } | { \pmb X } _ { 0 } , { \pmb X } _ { t } } ( { \pmb y } | { \pmb x } _ { 0 } , { \pmb x } _ { t } ) = p _ { { \pmb Y } | { \pmb X } _ { 0 } } ( { \pmb y } | { \pmb x } _ { 0 } )$ . Then, the weighted distribution $g _ { Q } ( \pmb { x } | \pmb { x } _ { t } , t )$ used for generating large jump samples should be replaced by

$$
g _ { p o s t e r i o r , Q } ( \pmb { x } ) = \sum _ { j = 1 } ^ { N } \frac { 1 _ { \{ \pmb { x } _ { 0 , j } \} } ( \pmb { x } ) p _ { \pmb { Y } | \pmb { X } _ { 0 } } ( \pmb { y } | \pmb { x } _ { 0 , j } ) Q _ { t , \epsilon , j } } { \sum _ { j = 1 } ^ { N } p _ { \pmb { Y } | \pmb { X } _ { 0 } } ( \pmb { y } | \pmb { x } _ { 0 , j } ) Q _ { t , \epsilon , j } }\tag{47}
$$

Proof: The proof is relegated to Appendix C.

Note that the likelihood can usually be efficiently obtained from the transition function between $\scriptstyle { \mathbf { { \vec { x } } } } _ { 0 }$ and $\mathbf { \pmb { y } } .$ For instance, this transition function is the channel model for communication tasks. Proposition 4 shows that the large jump amplitude sampler can be adapted to the observation-guided setting by modifying the empirical mixture weights. Similarly, g<sub>posterior,Q</sub>(x) can also be truncated to reduce the inference latency.

## V. NUMERICAL RESULTS

In this section, several numerical experiments are provided to validate the advantages of the proposed inverse sampler. We consider channel estimation for OFDM-SISO systems under mixed noise.

For the channel model, we consider a frequency-selective SISO-OFDM channel modeled by a tapped-delay-line (TDL) response. The path gains are modeled as independent circularly symmetric complex Gaussian random variables with prescribed average powers. Moreover, the delay-power profiles from 3GPP TR 38.901 are used. Due to space limitations, we choose the TDL-A, TDL-C, and TDL-D delay-power profiles as representative scenarios. The mixed channel noise consists of WGN and IN described by the SαS distribution. They are mutually independent and memoryless. Let their scaling parameters be $\gamma _ { g }$ and $\gamma _ { s } ,$ respectively. Then, the generalized signal-to-noise ratio (GSNR) is defined as follows:

$$
\mathrm { G S N R } ( \mathrm { d } \mathbf { B } ) = 1 0 \log _ { 1 0 } \frac { P _ { s } } { 2 ( \gamma _ { g } ^ { 2 } + \gamma _ { s } ^ { 2 } ) } ,\tag{48}
$$

where $P _ { s }$ is the transmit signal power.

For the baselines, we consider existing model-based algorithms and generative methods, including:

• Linear minimum mean square error (LMMSE): LMMSE is an efficient channel estimation method under WGN. However, its performance and stability may deteriorate considerably in mixed-noise scenarios due to impulsive outliers.

• Clipped LMMSE: Clipped LMMSE first suppresses large-amplitude pilot observations through clipping and then applies the conventional LMMSE estimator. This simple preprocessing improves the robustness of LMMSE against IN while retaining low computational complexity.

• Clipped orthogonal matching pursuit (OMP) [28]: OMP iteratively selects dominant delay atoms from a common candidate dictionary and reconstructs the frequency-domain channel. Clipped OMP is a clipped version of OMP for scenarios with mixed noise.

• Clipped sparse Bayesian learning (SBL): SBL estimates the delay-domain sparse channel by learning the hyperparameters that reflect the sparsity. Clipped SBL is a clipped version of conventional SBL.

• Outlier-aware SBL [29], [30]: Different from Clipped SBL, outlier-aware SBL explicitly models the received pilot observations as the sum of a sparse delay-domain channel component, a sparse outlier component, and Gaussian background noise. By jointly estimating the channel and the impulsive outliers, this method is designed for mixed Gaussian and IN environments.

• Diffusion method [15]: This method uses a generative model based on the diffusion forward process and posterior inverse inference. Different from the proposed framework, it does not introduce the jump process and uses only the local score function to recover the target distribution.

![](images/aa40c8cc0b505639a7285346f319f9e1360d149669087e7c24ef194c4044c78a.jpg)  
(a)

![](images/87f646a64eab28a85754c8a30d0f27c7a694fab00823285f54d1f0f5c936ffc6.jpg)  
(b)

![](images/f108ca0f079d3867fec9b09b73e328d39c99afc0376bdde9d531d8755d7dfe95.jpg)  
(c)  
Fig. 1. Training loss, test loss, and accuracy validation of jump rate prediction using the proposed lightweight neural network under different values of α. (a) Comparison of training and test losses; (b) Comparison of jump rates for α = 1.2; (c) Comparison of jump rates for $\alpha = 1 . 8 .$

• Levy-DSM [´ 19]: This method is based on the inverse sampler designed for the Levy process under the DSM´ framework. It shows that the network can be trained to regress noise following α-stable distributions to achieve inverse sampling.

• Benchmark: The benchmark is a genie-aided LMMSE estimator under WGN with known true delay support. It uses ideal prior delay information that is unavailable to practical estimators. Therefore, it serves as an NMSE reference for the achievable estimation performance.

Finally, the other parameter configurations are as follows: the number of subcarriers is $N _ { c } = 6 4 ;$ the number of OFDM symbols per frame is $N _ { t } = 1 6 ;$ the pilot pattern is “comb”; the pilot spacing is 4 or 8; the modulation scheme is QPSK; and the number of paths, path delays, and path powers follow TDL-A/C/D. The pilot spacing is set to either 4 or 8 to validate the performance under different pilot densities.

## A. Structure and training of network $g ^ { \theta _ { 2 } }$

Since the large jump rate is a scalar statistical quantity, a lightweight neural network is sufficient for its amortized estimation. Specifically, we parameterize $g _ { \boldsymbol { \theta } _ { 2 } } ( \mathbf { x } _ { t } , t )$ by a convolutional neural network (CNN) and a multilayer perceptron (MLP). Its input is the normalized state $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { t } }$ concatenated with a time embedding of $t .$ To ensure the nonnegativity of the estimated jump rate, the final output is passed through a softplus activation. Compared with ReLU, softplus provides a smoother parameterization and avoids permanently inactive zero-rate outputs. In our implementation, the network is trained by minimizing the mean square error (MSE) between its output and $\lambda ( \pmb { x } _ { t } | \pmb { x } _ { 0 } )$ . Note that the original sample $\scriptstyle { \mathbf { { \mathit { x } } } } _ { 0 }$ is only used to construct the conditional training target. It is not fed into the network as input because an accurate $\scriptstyle { \mathbf { { \mathit { x } } } } _ { 0 }$ is not available at the inference stage.

For the OFDM channel estimation task, the complex-valued state is first represented by its real and imaginary parts.

For example, when one frame contains $N _ { t }$ OFDM symbols and $N _ { c }$ subcarriers, we reshape the state into a two-channel tensor ${ \bf X } _ { t } ~ \in ~ \mathbb { R } ^ { 2 \times N _ { t } \times N _ { c } }$ . The rate network adopts a small convolutional architecture to exploit the local time-frequency structure while keeping the number of trainable parameters low. Specifically, it consists of four $3 \times 3$ convolutional layers with channel widths {16, 16, 32, 32}, respectively, followed by a global average pooling layer. The pooled feature is concatenated with the time embedding and then passed to a two-layer MLP head to produce a scalar output. The estimated large jump rate is given by

$$
\begin{array} { r l r } & { } & { g ^ { \theta _ { 2 } } ( { \bf x } _ { t } , t ) = \mathrm { s o f t p l u s } \big ( \mathrm { M L P } ^ { \theta _ { 2 } } ( [ \mathrm { G A P } ( \mathrm { C N N } ^ { \theta _ { 2 } } ( { \bf X } _ { t } ) ) , \mathrm { E m b } ( t ) ) ) } \\ & { } & { + \lambda _ { \epsilon } , \ ~ ( 4 9 ) } \end{array}
$$

where $\mathrm { G A P } ( \cdot )$ denotes global average pooling, Emb(t) is the time embedding, and $\lambda _ { \epsilon } ~ > ~ 0$ is a small constant for numerical stability. For $N _ { t } ~ = ~ 1 6$ and $N _ { c } = 6 4$ , this small CNN contains approximately $2 \times 1 0 ^ { 4 }$ trainable parameters. This is much smaller than directly parameterizing the whole reverse transition with a neural network.

The MSE loss between the accurate rate and the network output is shown in Fig. 1. Fig. 1b and Fig. 1c present the results of 1000 experiments, where the time instant and data sample $\scriptstyle { \mathbf { { \mathit { x } } } } _ { 0 }$ are chosen randomly. These 1000 tests are then sorted according to their accurate jump rates. For validation, the reference large jump rate is computed using the empirical marginal expression in (23), rather than a single conditional target $\lambda ( \pmb { x } _ { t } | \pmb { x } _ { 0 } )$ . From Fig. 1a, it can be observed that the neural network converges very fast and only requires about 60 epochs with different values of $\alpha .$ . Moreover, the network can accurately estimate the jump rate under different impulsiveness levels.

## B. Various channel configurations

Here, we investigate the channel estimation performance under various α. First, the channel estimation accuracy and the corresponding detection performance are compared. We use normalized mean square error (NMSE) and bit error rate (BER) as the criteria. The experimental results are presented in Fig. 2 and Fig. 3.

![](images/0f84a62c56fc3ca3c15a4d10158b1aab6341c4d00f52a6dabf17d1470ee0d19a.jpg)  
(a) NMSE, TDL-A, α = 1.2

![](images/1bd598b55bb70105f7ad92aca5a5ffb2825576995c70bb0a62058b1d805f58c7.jpg)  
(b) BER, TDL-A, α = 1.2

![](images/3b7fa6f62c1da154dd7065edd5be0d0ad36ce34c4672ed55725fb93027da8192.jpg)  
(c) NMSE, TDL-C, α = 1.2

![](images/99cc818d4df01b9c1895b021603b0d15fac172f0262b6d725ff3d94245566a13.jpg)  
(d) BER, TDL-C, α = 1.2

![](images/ff0358c8ed10bdec6cf50e9c9e207edf5a649ac9b883f16654f5700e2e8f35b8.jpg)  
(e) NMSE, TDL-D, α = 1.2

![](images/aacac7231b5bc00b9009174a04c07a1ef44e17613b1c835d0aedf0cfb6956c78.jpg)  
(f) BER, TDL-D, α = 1.2  
Fig. 2. NMSE and BER comparisons of channel estimation under different scenarios (TDL-A/C/D) with α = 1.2.

In general, the proposed approach achieves the best performance among the baselines in terms of both NMSE and BER. For “Clipped LMMSE”, “Clipped SBL”, and “Clipped OMP”, the clipping operation suppresses most IN samples and thus improves robustness. However, clipping also distorts the transmitted signals, especially in the high-GSNR regime. Therefore, these three methods show a lower NMSE decay rate as the GSNR continues to increase. “Outlier-aware SBL” has better robustness under mixed channel noise because it explicitly considers the influence of IN samples. However, it can only achieve an accurate estimation at the pilot locations. For learning-based methods, both “Score-based” and “Proposed” learn the statistical structure of the channel model and thus achieve better estimation accuracy in various scenarios. Compared with the “Score-based” algorithm, the proposed method is more likely to reach a better performance because it allows large jumps to enable nonlocal transitions between distant high-probability regions. In contrast, “Score-based” only uses gradient information and small step sizes for optimization. As a result, it may get trapped in local optima, especially when the channel model is more complicated. Finally, the proposed method is also more robust and stable compared to the “Levy-´ DSM”, which is consistent with the above analyses.

Fig. 3 presents the estimation results under different impulsiveness levels. When $\alpha = 1 . 2$ , there are multiple outliers with large amplitudes, and the channel becomes much more complex. In this case, the proposed approach is not only robust but also achieves a larger performance gain compared with the cases with larger α. However, when α is close to 2 and the impulsiveness is very weak, the existing baselines also show comparable performance. This is consistent with the above analysis.

![](images/1bc6385656cc4d366b525f3a3413428640b7586d15055e17762b3f5d15d43117.jpg)  
(a) NMSE, TDL-C, $\alpha = 1 . 8$

![](images/713e262831ddc254f7c3daf86e3887de598f03a2b1b6c28a6c6b240410fcc835.jpg)  
(b) BER, TDL-C, α = 1.8  
Fig. 3. NMSE and BER comparisons of channel estimation with TDL-C as the delay-power profile and $\alpha = 1 . 8$

## C. Ablations

Ablation studies are conducted to explore the influence from the approximation of the forward noise distribution, the top-K truncation and the approximated posterior jump rate.

During the design of the inverse sampler, the distribution of the forward mixed noise at arbitrary time instants is required. However, there is no exact closed-form PDF expression for the impulsive component or the mixed noise. To avoid numerical integration, we use the approximated PDF of the mixed noise given in (30) and (31). The following experiments demonstrate the effectiveness of this approximation. We still use channel estimation as the application.

In Fig. 4, Fig. 5 and Fig. 6, we separately focus on the effects of noise distribution approximation, top-K truncation and large jump rate approximation. The time consumption is shown in Table I, using a CPU configuration of ‘12th Gen Intel(R) Core(TM) i9-12900H’. It can be observed that using the accurate PDF does not bring a considerable performance gain, while the complexity of “Accurate PDF” is unacceptable. This is because the distribution needs to be calculated by numerical integration at every step. More importantly, even though the high-dimensional PDF can be transformed into polar coordinates, the integral is still challenging to further simplify. This remains time-consuming when a high resolution is required to ensure accuracy.

Then, the approximation error introduced by the top-K truncation is examined. For the benchmark, we set K equal to the total number of samples used for training. Here, for the truncation scheme, we set K as 10% of the total number of training samples. Similar to the noise model approximation, the truncation error is also not significant. This is because, for a given time instant and sample ${ \mathbf { } } x _ { t } ,$ the probability that most original data samples generate $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { t } }$ is quite small. Therefore, the influence of these samples on jump generation is negligible. As a result, K can be much smaller than the total number of samples, which reduces redundancy and improves efficiency.

Finally, Fig. 6 implies that the posterior inverse sampling based on the exact large jump rate achieves a better NMSE. However, its computational cost significantly increases since the jump rate calculation needs to rely on the numerical integral. Moreover, in terms of detection error probability, the performance gain is less considerable than that with respect to NMSE. This result supports that using prior large jump rate is an efficient scheme.

In summary, the approximation operations in the inverse sampling algorithm achieve a tradeoff between performance and complexity.

![](images/06a9e50c541c990c3752dfe41fede44e513965db2d5cc0daa61b497190ffd425.jpg)  
(a) NMSE

![](images/b638751599f3471699267007f419f0869555060fd496bc851435e8990c967ea7.jpg)  
(b) BER  
Fig. 4. Ablation results compared with the accurate mixed noise PDF in terms of NMSE and BER.

![](images/b3b3812c89b7e1f667e0686034cafada8d84f3d647d5da39efa954afc982b1c8.jpg)  
(a) NMSE

![](images/15dd230708c025a571d6cf4d7d24b6a82c599e60ba3e99c3790e3d58c3ed3aa3.jpg)  
(b) BER  
Fig. 5. Ablation results compared with the no-truncation case in terms of NMSE and BER.

![](images/9c12fd8429f0c096cb355e0079afcfa150703d61ecbe57ad1f8d9c9d5c328d80.jpg)  
(a) NMSE

![](images/294b43749f1906d069613c65dc6f566caca84e1ca97c6abd8939c22b9b0b6a03.jpg)  
(b) BER  
Fig. 6. Ablation results compared with the accurate posterior large jump rate in terms of NMSE and BER.

TABLE I  
TIME CONSUMPTION COMPARISON OF SIMULATIONS IN FIG. 4∼6 UNDER DIFFERENT VALUES OF α.
<table><tr><td>Algorithms/Scenarios</td><td> $\alpha = 1 . 2$ </td><td> $\alpha = 1 . 8$ </td></tr><tr><td>Proposed</td><td>5.6s</td><td>5.5s</td></tr><tr><td>Accurate PDF</td><td>1717.6s</td><td>1698.4s</td></tr><tr><td>No truncation</td><td>52.7s</td><td>54.1s</td></tr><tr><td>Accurate rate</td><td>6125.3s</td><td>6379.6s</td></tr></table>

## VI. CONCLUSIONS

In this paper, we proposed a generator-guided inverse sampling algorithm for a class of isotropic linear Levy-driven´ SDEs with symmetric α-stable jump components. We first derived the expression of the backward generator and used it to decompose the full reverse sampling process into the diffusion part and the jump components. Then, by theoretically analyzing the inverse sampling process from a statistical perspective, we determined the objective that can be learned by a lightweight network. We also designed an efficient reverse sampling method by introducing several efficient approximation operations. As an application, the proposed inverse sampling algorithm outperformed the considered baselines in channel estimation under mixed channel noise across different scenarios. Moreover, numerical results showed that the proposed approach achieves a good balance between performance and complexity.

## APPENDIX A PROOF OF PROPOSITION 1

For a Levy process with infinite activity, the generator´ associated with the test function f is given by

$$
\begin{array} { r l } {  { ( \mathcal { L } _ { J , F } f ) ( { \pmb x } _ { t } ) = { \pmb b } _ { J , F } ^ { \epsilon } ( { \pmb x } _ { t } , t ) \cdot \nabla f ( { \pmb x } _ { t } ) } } \\ & { + \displaystyle \int _ { R ^ { D } } ( f ( { \pmb x } _ { t } + { \pmb v } ) - f ( { \pmb x } _ { t } ) - 1 _ { \{ { \pmb v } : \| { \pmb v } \| \leq \epsilon \} } { \pmb v } \cdot \nabla f ( { \pmb x } _ { t } ) ) \nu _ { t } ( d { \pmb v } ) } \end{array}\tag{A.1}
$$

where ${ b } _ { J , F } ^ { \epsilon } ( { \pmb x } _ { t } , t )$ is the canonical drift induced by $d { \pmb { L } } _ { t }$ . For the isotropic case, we have $b _ { J , F } ^ { \epsilon } ( { \pmb x } _ { t } , t ) = 0$ . By substituting (A.1) into (2), the complete forward generator is obtained as

$$
\left( \mathcal { L } _ { F } f \right) ( \mathbf { \boldsymbol { x } } _ { t } ) = \pmb { b } _ { F } ^ { \epsilon } ( \mathbf { \boldsymbol { x } } _ { t } , t ) \cdot \nabla f ( \mathbf { \boldsymbol { x } } _ { t } ) + \frac { 1 } { 2 } \mathrm { T r } \left( \Sigma ( t ) \nabla ^ { 2 } f ( \pmb { x } _ { t } ) \right)
$$

$$
+ \int _ { R ^ { D } } \left( f ( { \pmb x } _ { t } + { \pmb v } ) - f ( { \pmb x } _ { t } ) - 1 _ { \{ { \pmb v } : \| { \pmb v } \| \leq \epsilon \} } { \pmb v } \cdot \nabla f ( { \pmb x } _ { t } ) \right) \nu _ { t } ( d { \pmb v } )\tag{A.2}
$$

with $b _ { F } ^ { \epsilon } ( { \pmb x } _ { t } , t ) = { b } ( { \pmb x } _ { t } , t ) + { b } _ { J , F } ^ { \epsilon } ( { \pmb x } _ { t } , t )$ . Next, time reversal is applied to derive the backward generator. Based on [31], the forward and backward jump kernels satisfy the flux equation

$$
p ( \pmb { x } _ { t } ) \overleftarrow { K } _ { t , \pmb { x } _ { t } } ( d \pmb { v } ) = p ( \pmb { x } _ { t } + \pmb { v } ) \overrightarrow { K } _ { t , \pmb { x } _ { t } + \pmb { v } } ( d ( - \pmb { v } ) )\tag{A.3}
$$

which means that the jump density from $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { t } }$ to $\mathbf { \boldsymbol { x } } _ { t } + \mathbf { \boldsymbol { v } }$ should be equal to that from $\pmb { x } _ { t } + \pmb { v }$ back to $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { t } }$ . Moreover, the backward canonical drift satisfies [31]

$$
\begin{array} { l } { b _ { J , F } ^ { \epsilon } ( { \pmb x } _ { t } , t ) + b _ { J , B } ^ { \epsilon } ( { \pmb x } _ { t } , t ) } \\ { \displaystyle = \int _ { R ^ { D } } 1 _ { \{ { \pmb v } : \| { \pmb v } \| \leq \epsilon \} } { \pmb v } \left( \vec { K } _ { t , { \pmb x } _ { t } } ( d { \pmb v } ) + \overleftarrow { K } _ { t , { \pmb x } _ { t } } ( d { \pmb v } ) \right) } \end{array}\tag{A.4}
$$

Therefore, we have

$$
\begin{array} { l } { \displaystyle { \overleftarrow { K } _ { t , \mathbf { x } _ { t } } ( d v ) = \frac { p ( \mathbf { x } _ { t } + v ) } { p ( \mathbf { x } _ { t } ) } \vec { K } _ { t , \mathbf { x } _ { t } + v } \left( d \left( - v \right) \right) } } \\ { \displaystyle { = \frac { p ( \mathbf { x } _ { t } + v ) } { p ( \mathbf { x } _ { t } ) } \nu _ { t } ( d ( - v ) ) } } \end{array}\tag{A.5}
$$

In this case, the backward generator of the Levy process is´ given by

$$
\begin{array} { r l } & { \quad \langle \mathcal { L } _ { i , B } f \rangle ( \alpha _ { i } ) , } \\ & { = \delta _ { i , B } ^ { \ast } ( x _ { i } , t ) \cdot \nabla f ( x _ { t } ) + \int _ { n ^ { \mathcal { D } } } \big ( f ( x _ { t } + v ) - f ( x _ { t } ) \big .  } \\ & { \quad - 1   ( \mathrm { e } _ { 1 } | \mathrm { e } _ { 2 } | \mathrm { e } _ { 3 } \cdot \nabla f ( x _ { t } ) ) \hat { X } _ { i , \alpha _ { i } , ( d ) } ( d v )  } \\ &  = \bigg ( - b \delta _ { i , F } ^ { \ast } ( x _ { i } , t ) + \int _ { n ^ { \mathcal { D } } } 1 _ { \{ w \} = 1 \} | \mathrm { e } _ { 4 } \mathrm { e } _ { 3 } \cdot \big ( P _ { \mathrm { r } } ( \mathrm { d } w ) } \\ & { \quad +  \hat { K } _ { i , \alpha _ { i } , ( d ) } \big ) ) \cdot \nabla f ( x _ { t } ) \big ) + \int _ { n ^ { \mathcal { D } } } \big ( f ( x _ { t } + v ) - f ( x _ { t } ) \big .  } \\ & { \quad  - 1 _ { \{ v \in [ \mathrm { e } ] \} \leq 0 } \cdot \nabla f ( x _ { t } ) \big ) \hat { K } _ { i , \alpha _ { i } , ( d ) } ( \mathrm { e } ) } \\ & { = \bigg ( - b \delta _ { i , F } ^ { \ast } ( x _ { i } , t ) + \mathrm { P r } \cdot \int _ { n ^ { \mathcal { D } } } 1 _ { \{ w \} = 1 } | v _ { i } | \mathrm { e } _ { 4 } \mathrm { e } _ { 3 } \mathrm { e } _ { i } ( v ) \Big ) \cdot \nabla f ( x _ { t } ) } \\ &  \quad + \mathrm { p } . \mathrm { s } . \int _ { n _ { \mathcal { D } } } \big ( f ( x _ { t } + v ) - f ( x _ { t } ) \big ) \frac { p ( x _ { t } + v ) } { p ( x _ { t } ) } \nu _ { i , ( d ) } ( \mathrm  e \end{array}
$$

Combining (A.2) with the time reversal of the diffusion part, the complete backward generator is given by

$$
\begin{array} { r l }  \left( \mathcal { L } _ { B } f \right) ( \pmb { x } _ { t } ) = \left( \begin{array} { l } { - b ( \pmb { x } _ { t } , t ) + \Sigma ( t ) \nabla \log p ( \pmb { x } _ { t } ) \right) \cdot \nabla f ( \pmb { x } _ { t } ) } \\ { + \displaystyle \frac { 1 } { 2 } \mathrm { T r } \left( \Sigma ( t ) \nabla ^ { 2 } f ( \pmb { x } _ { t } ) \right) + ( \mathcal { L } _ { J , B } f ) ( \pmb { x } _ { t } ) ( \pmb { x } _ { t } , t ) } \end{array} \end{array}\tag{A.7}
$$

Finally, if the scaled Levy measure´ $\nu _ { t }$ is symmetric, we have $\nu _ { t } ( d ( - pmb { v } ) ) = \nu _ { t } ( d \pmb { v } )$ . For the symmetric α-stable case, the canonical jump drift vanishes. This further simplifies the backward generator and completes the proof.

## APPENDIX B PROOF OF PROPOSITION 3

We first use a variable transformation to absorb the drift term. Let $Y _ { t } = \exp ( - R t ) X _ { t }$ . Based on Ito’s formula,

$$
\begin{array} { r l } & { d \pmb { Y } _ { t } = \exp ( - R t ) d \pmb { X } _ { t } + d ( \exp ( - R t ) ) \pmb { X } _ { t } } \\ & { \quad \quad = \exp ( - R t ) d \pmb { X } _ { t } - R \exp ( - R t ) \pmb { X } _ { t } } \\ & { \quad \quad = \exp ( - R t ) ( s d t + \Phi _ { G } ( t ) d \pmb { W } _ { t } + \Phi _ { S } ( t ) d \pmb { L } _ { t } ) } \end{array}\tag{B.1}
$$

According to $Y _ { t } = \exp ( - R t ) X _ { t }$

$$
\begin{array} { r l } & { X _ { t } = \underbrace { \exp ( R t ) X _ { 0 } + \exp ( R t ) \int _ { 0 } ^ { t } \exp ( - R l ) s d l } _ { \mu ( t , x _ { 0 } ) } } \\ & { \quad + \underbrace { \int _ { 0 } ^ { t } \exp ( ( t - l ) R ) \Phi _ { G } ( l ) d W _ { l } } _ { G ( t ) } } \\ & { \quad + \underbrace { \int _ { 0 } ^ { t } \exp ( ( t - l ) R ) \Phi _ { S } ( l ) d L _ { l } } _ { S ( t ) } } \end{array}\tag{B.2}
$$

Then, based on the linearity of Gaussian RV and α-stable RV, the proof is completed.

## APPENDIX C PROOF OF PROPOSITION 4

First, we have $\begin{array} { r } { \frac { p ( { \pmb x } _ { t } + { \pmb v } | { \pmb y } ) } { p ( { \pmb x } _ { t } | { \pmb y } ) } \nu _ { t } ( d { \pmb v } ) = \frac { p ( { \pmb x } _ { t } + { \pmb v } , { \pmb y } ) } { p ( { \pmb x } _ { t } , { \pmb y } ) } \nu _ { t } ( d { \pmb v } ) } \end{array}$ . Since the denominator is independent of v, it can be omitted during sampling. For clarity, we add subscripts to indicate the arguments of the distributions. The target sampling distribution is given by

$$
\begin{array} { r l } & { \int _ { \mathrm { R e } } \eta _ { \alpha } \log ( \alpha + \eta _ { 1 } ) \log ( k \Theta ) } \\ & { = \int _ { \mathrm { R e } } \eta _ { \alpha } \log ( \alpha + \eta _ { 1 } ) \log ( k \Theta ) + ( \alpha + \eta _ { 1 } ) ( k \Theta ) } \\ & { = \int _ { \mathrm { R e } } \eta _ { \alpha } \log ( k \Theta ) \log ( \alpha + \eta _ { 1 } ) ( k \Theta ) - ( \alpha + \eta _ { 1 } ) ( k \Theta ) } \\ & { = \int _ { \mathrm { R e } } \eta _ { \alpha } \log ( k \Theta ) \log ( k \Theta ) - ( \alpha + \eta _ { 1 } ) ( k \Theta ) } \\ & { = \int _ { \mathrm { R e } } \eta _ { \alpha + \pi / \alpha } \int _ { \mathrm { R e } } \eta _ { \alpha + \pi / \alpha } \log ( k \Theta ) \log ( \alpha + \eta _ { 1 } ) ( k \Theta ) \log ( k \Theta ) } \\ & { \quad \times | \alpha + \eta _ { 1 } \Theta ( k ) | ^ { \alpha - 2 } d k } \\ & { = \int _ { \mathrm { R e } } \eta _ { \alpha + \pi / \alpha } \int _ { \mathrm { R e } } \eta _ { \alpha + \pi / \alpha } \log ( k \Theta ) \log ( 1 , k \Theta ) \log ( \alpha ) d k } \\ & { \quad \times | \alpha + \eta _ { 1 } \Theta ( k ) | ^ { \alpha - 2 } d k } \\ & { = \int _ { \mathrm { R e } } \eta _ { \alpha + \pi / \alpha } \log ( 1 , k \Theta ) \log ( 1 , k \Theta ) \log ( 1 , k \Theta ) \log ( 1 , \alpha ) } \\ & { \quad \times | \alpha + \eta _ { 1 } \Theta ( k ) | ^ { \alpha - 2 } d k } \\ & { = \int _ { \mathrm { R e } } \eta _ { \alpha + \pi / \alpha } \prod _ { \alpha } \sum _ { \beta = \pi / \alpha } ( k \Theta ) \log ( 1 , k \Theta ) \log ( 1 , k \Theta ) } \\ & { \quad \times | \alpha + \eta _ { 1 } \Theta ( k ) | ^ { \alpha - 2 } d k } \\ &  \quad \times | \end{array}\tag{C.1}
$$

By introducing the normalization factors and following procedures similar to those in (41) and (42), we can deduce that the target sampling distribution can also be decomposed into the conditional distribution and the weighted distribution $g _ { \mathrm { p o s t e r i o r } , Q } ( \pmb { x } )$ , where

$$
g _ { \mathrm { p o s t e r i o r } , Q } ( \pmb { x } ) = \sum _ { j = 1 } ^ { N } \frac { 1 _ { \pmb { x } _ { 0 , j } } ( \pmb { x } ) p _ { \pmb { Y } | \pmb { X } _ { 0 } } ( \pmb { y } | \pmb { x } _ { 0 , j } ) Q _ { t , \epsilon , j } } { \sum _ { j = 1 } ^ { N } p _ { \pmb { Y } | \pmb { X } _ { 0 } } ( \pmb { y } | \pmb { x } _ { 0 , j } ) Q _ { t , \epsilon , j } }\tag{C.2}
$$

[1] Y. Song and S. Ermon, “Generative modeling by estimating gradients of the data distribution,” Advances in neural information processing systems, vol. 32, 2019.

[2] J. Ho, A. Jain, and P. Abbeel, “Denoising diffusion probabilistic models,” Advances in neural information processing systems, vol. 33, pp. 6840– 6851, 2020.

[3] Y. Song, J. Sohl-Dickstein, D. P. Kingma, A. Kumar, S. Ermon, and B. Poole, “Score-based generative modeling through stochastic differential equations,” International Conference on Learning Representations, vol. 9, 2021. [Online]. Available: https://mlanthology.org/iclr/2021/song2021iclr-scorebased/

[4] P. Vincent, “A connection between score matching and denoising autoencoders,” Neural Computation, vol. 23, no. 7, pp. 1661–1674, Jul. 2011.

[5] J. Li and C. Wang, “Efficient diffusion posterior sampling for noisy inverse problems,” SIAM Journal on Imaging Sciences, vol. 18, no. 2, pp. 1468–1492, 2025.

[6] B. Kawar, M. Elad, S. Ermon, and J. Song, “Denoising diffusion restoration models,” Advances in neural information processing systems, vol. 35, pp. 23 593–23 606, 2022.

[7] H. Chung, B. Sim, D. Ryu, and J. C. Ye, “Improving diffusion models for inverse problems using manifold constraints,” Advances in Neural Information Processing Systems, vol. 35, pp. 25 683–25 696, 2022.

[8] B. Song, S. M. Kwon, Z. Zhang, X. Hu, Q. Qu, and L. Shen, “Solving inverse problems with latent diffusion models via hard data consistency,” in International Conference on Learning Representations, vol. 2024, 2024, pp. 7624–7654.

[9] N. Zilberstein, A. Sabharwal, and S. Segarra, “Solving linear inverse problems using higher-order annealed langevin diffusion,” IEEE Transactions on Signal Processing, vol. 72, pp. 492–505, Jan. 2024.

[10] Y. Hu, E. Bell, G. Wang, and Y. Sun, “Prism: Probabilistic and robust inverse solver with measurement-conditioned diffusion prior for blind inverse problems,” in ICASSP 2026 - 2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2026, pp. 11 432–11 436.

[11] B. Xia, Y. Zhang, S. Wang, Y. Wang, X. Wu, Y. Tian, W. Yang, and L. Van Gool, “Diffir: Efficient diffusion model for image restoration,” in Proceedings of the IEEE/CVF international conference on computer vision, 2023, pp. 13 095–13 105.

[12] B. Fei, Z. Lyu, L. Pan, J. Zhang, W. Yang, T. Luo, B. Zhang, and B. Dai, “Generative diffusion prior for unified image restoration and enhancement,” in 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023, pp. 9935–9946.

[13] Y. Chen and T. Pock, “Trainable nonlinear reaction diffusion: A flexible framework for fast and effective image restoration,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 39, no. 6, pp. 1256– 1272, Jun. 2017.

[14] O. Ozdenizci and R. Legenstein, “Restoring vision in adverse weather<sup>¨</sup> conditions with patch-based denoising diffusion models,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 45, no. 8, pp. 10 346–10 357, Aug. 2023.

[15] X. Zhou, L. Liang, J. Zhang, P. Jiang, Y. Li, and S. Jin, “Generative diffusion models for high dimensional channel estimation,” IEEE Transactions on Wireless Communications, vol. 24, no. 7, pp. 5840–5854, Jul. 2025.

[16] M. Arvinte and J. I. Tamir, “Mimo channel estimation using score-based generative models,” IEEE Transactions on Wireless Communications, vol. 22, no. 6, pp. 3698–3713, Jun. 2023.

[17] R. Li, J. Sun, and J. Xue, “Generative diffusion-based bayesian modeling for universal channel estimation,” IEEE Journal on Selected Areas in Communications, vol. 44, pp. 3104–3119, Dec. 2026.

[18] T. Karras, M. Aittala, T. Aila, and S. Laine, “Elucidating the design space of diffusion-based generative models,” Advances in neural information processing systems, vol. 35, pp. 26 565–26 577, 2022.

[19] E. B. Yoon, K. Park, S. Kim, and S. Lim, “Score-based generative models with levy processes,” ´ Advances in neural information processing systems, vol. 36, pp. 40 694–40 707, 2023.

[20] D. Shariatian, U. Simsekli, and A. Oliviero Durmus, “Heavy-tailed diffusion with denoising levy probabilistic models,” in ´ International Conference on Learning Representations, vol. 2025, 2025, pp. 96 991– 97 024.

[21] D. Applebaum, Levy Processes and Stochastic Calculus´ , ser. Cambridge Studies in Advanced Mathematics. Cambridge: Cambridge University Press, 2004.

[22] Y. Lipman, R. T. Q. Chen, H. Ben-Hamu, M. Nickel, and M. Le, “Flow matching for generative modeling,” in The Eleventh International Conference on Learning Representations, Kigali, Rwanda, 2023.

[23] G. Sureka and K. Kiasaleh, “Sub-optimum receiver architecture for awgn channel with symmetric alpha-stable interference,” IEEE Trans Commun., vol. 61, no. 5, pp. 1926–1935, May. 2013.

[24] G. Samorodnitsky, M. S. T. Chapman, and Hall, Stable non-Gaussian random processes: Stochastic models with infinite variance. New York: Chapman & Hall, 1994.

[25] T. Qi, J. Zhang, J. Wang, and Y. Zhu, “Bursty mixed gaussian-impulsive noise model and parameter estimation,” IEEE Trans. Commun., vol. 73, no. 9, pp. 8274–8288, Sept. 2025.

[26] D. Zwillinger, V. Moll, I. Gradshteyn, and I. Ryzhik, Eds., Table of Integrals, Series, and Products (Eighth Edition). Boston: Academic Press, 2014.

[27] H. Chung, J. Kim, M. T. McCann, M. L. Klasky, and J. C. Ye, “Diffusion posterior sampling for general noisy inverse problems,” in The Eleventh International Conference on Learning Representations (ICLR 2023), Kigali, Rwanda, 2023.

[28] Z. Wang, Y. Li, C. Wang, D. Ouyang, and Y. Huang, “A-omp: An adaptive omp algorithm for underwater acoustic ofdm channel estimation,” IEEE Wireless Communications Letters, vol. 10, no. 8, pp. 1761–1765, Aug. 2021.

[29] X. Feng, J. Wang, X. Kuai, M. Zhou, H. Sun, and J. Li, “Message passing-based impulsive noise mitigation and channel estimation for underwater acoustic ofdm communications,” IEEE Transactions on Vehicular Technology, vol. 71, no. 1, pp. 611–625, Jan. 2022.

[30] Z. Zhang, X. Han, W. Li, L. Wei, and J. Yin, “Robust time-correlated sparse bayesian channel estimation for short-block underwater acoustic communications under impulsive noise,” IEEE Wireless Communications Letters, vol. 15, pp. 3194–3198, May. 2026.

[31] G. Conforti and C. Leonard, “Time reversal of markov processes with´ jumps under a finite entropy condition,” Stochastic Processes and their Applications, vol. 144, pp. 85–124, Feb. 2022.