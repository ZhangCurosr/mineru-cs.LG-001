# Multi-stage neural operator learning with application for convolutions

Zhiping Mao<sup>a</sup>, Zhenye Wen<sup>b</sup>, Yong Zhang<sup>c,d</sup>, Xiaofei Zhao<sup>b,e</sup>

<sup>a</sup>School of Mathematical Sciences, Eastern Institute of Technology, Ningbo, 315200 Ningbo, China <sup>b</sup>School of Mathematics and Statistics, Wuhan University, 430072 Wuhan, China <sup>c</sup>Center for Applied Mathematics and KL-AAGDM, Tianjin University, 300072 Tianjin, China <sup>d</sup>State Key Laboratory of Synthetic Biology, Tianjin University, Tianjin, 300072, China <sup>e</sup>Computational Sciences Hubei Key Laboratory, Wuhan University, 430072 Wuhan, China

## Abstract

Convolution integrals widely exist in applications, and to enable fast and accurate computations, this paper introduces two general multi-stage neural operator learning frameworks. The first, Deep Collocation Neural Operator (DCNO), is a supervised approach that iteratively refines the operator approximation by learning residuals from input-output data pairs. The second, Deep Galerkin Neural Operator (DGNO), is an unsupervised framework applicable when the target operator can be represented by a PDE, leveraging the weak form of the PDE residual for training. Both methods progressively construct basis operators through multiple training stages to enrich the approximation space, leading to significantly improved accuracy over standard one-shot operator learning. We provide theoretical analysis for their approximation capabilities and implement them for learning convolutions. Extensive numerical experiments demonstrate that both DCNO and DGNO achieve high accuracy, approaching machine precision under single float for convolution problems, and ofer substantial eficiency gains for numerous queries or parametric variations compared to traditional solvers. We also extend these frameworks to handle multi-input operator learning scenarios involving variations in both the density and kernel of a convolution.

Keywords: Operator learning, Convolution, Residual-driven approximation, Multiple stages, Deep operator network, High accuracy

## 1. Introduction

Convolution computations arise naturally in a wide range of scientific and engineering applications, such as potential evaluation in electrostatics, gravitational particle simulations and signal processing [19, 48, 38, 17, 23], playing a crucial role in eficiently capturing long-range interactions. This paper focuses on the accurate and eficient evaluation of convolution. Let G denote the target convolution operator mapping the density $\rho ( \mathbf { x } )$ to the potential ϕ(x), i.e.,

$$
\phi ( \mathbf { x } ) = \mathcal { G } ( \rho ) ( \mathbf { x } ) = \int _ { \mathcal { R } } K ( \mathbf { x } , \tilde { \mathbf { x } } ) \rho ( \tilde { \mathbf { x } } ) d \tilde { \mathbf { x } } , \quad \mathbf { x } \in \mathcal { R } \subset \mathbb { R } ^ { d }\tag{1.1}
$$

where $d \in \mathbb { N } _ { + }$ is the spatial dimension, R denotes a region, and $K ( \mathbf { x } , \tilde { \mathbf { x } } )$ denotes a kernel.

Despite the linearity of the mapping, the nonlocal nature of convolution (1.1) renders its evaluation computationally demanding under direct discretizations. Fast solvers such as fast multipole methods, hierarchical matrix techniques, Gaussian-Summation method, and kernel truncation method have therefore been extensively developed to enable eficient computation [19, 48, 20, 15, 42] of convolution for an input ρ. They are able to produce predicted ϕ on $N \in \mathbb { N } .$ <sub>+</sub> prescribed grids at cost O(N log N). Although efective, repeated calls of the traditional solver for new density inputs in scenarios like parametric variations or manyquery settings may still incur substantial costs. This forms a natural circumstance for operator learning to apply, where neural networks are set to learn the mapping from the input function space (e.g., density and/or kernel) to the output solution space (e.g., potential). The trained operator network will be able to ofer a fast prediction of the solution function for a wide range of input functions at arbitrary points of the region [34, 9, 43, 28]. In contrast, traditional solvers produce the predicted solution only on the prescribed grid points, and values elsewhere will need to be obtained posteriorly via proper interpolations to maintain accuracy, e.g., [18].

In the literature, a wide range of neural operator learning methods have been developed, with representative approaches including Fourier neural operators [28, 27], deep operator networks (DeepONet) [34, 24, 35, 44], graph kernel networks [5], Orthogonal Polynomial Neural Operator [32], and random feature-based models [37]. Successful applications have already been made in a broad range of scientific computing problems [10, 11, 14, 26, 29, 36], and for the particular studies on the convolution problem (1.1), we refer to [34, 8]. Despite the remarkable progress, the existing operator learning works are based on direct one-shot training, whose accuracy is often quite limited. The resulting operator network for the convolution problem (1.1) would therefore be less competitive compared to traditional fast solvers. In fact, such a one-shot training accuracy limitation was noted earlier in deep learning of fixed-configuration PDE problems [2, 46, 45], and to address this issue, several residual-based approaches reaching high accuracy have been developed. They progressively seek new bases to improve solution accuracy by minimizing the current residual of the PDE. The Galerkin Neural Networks (GNN) [2] employs a Galerkin framework based on the weak formulation of PDEs to iteratively enrich the solution approximation, particularly for the problems involving singularities. Its extension, Extended Galerkin Neural Networks [3], further generalizes this framework to a broader class of boundary value problems. Under the strong formulation, the Multi-level Neural Networks [4] and the Multi-stage Neural Networks [45] progressively improve the approximation by sequentially introducing new neural network components to approximate the residual, whereas the Multigrade Neural Networks [47] achieves iterative refinement through progressive network deepening. More recently, the Deep Collocation Method [46] adopts a collocation-based least-squares framework to improve training eficiency while maintaining high accuracy. These high-accuracy PDE solvers ofer valuable insights for developing high-accuracy operator learning frameworks, which will be one of our goals.

To this end, in this work we shall first propose two high-accuracy operator learning frameworks that enrich the approximation space by progressively constructing neural operator bases as DeepONets through multiple residual-based training stages guided by the residuals. Specifically, a deep collocation neural operator (DCNO) method is designed for settings where input-output data pairs are available, falling within a supervised learning paradigm. On the other hand, when the task operator can be represented as a PDE, a deep Galerkin neural operator (DGNO) method is proposed, whose training uses the residual of the PDE in the weak form, enabling fully unsupervised operator learning. Theoretical analysis is done for each of them to support their approximation ability and convergence of the iterative frameworks. Afterwards, we shall apply the two proposed methods to the convolution operator learning task (1.1), and demonstrate through a series of numerical experiments and comparisons the accuracy and eficiency of the proposed algorithms. Extensions to the case with multiple inputs are implemented in the end using MIONet [24]. The main contributions of this work are summarized as follows.

1) Two general multi-stage operator learning frameworks are proposed, including a supervised one based on data pairs and an unsupervised one based on PDE. They are implemented using DeepONet and analyzed for accuracy and convergence.

2) The proposed algorithms are applied to convolution operator learning with extensions made using MIONet. Experimental results demonstrate eficient predictions with accuracy that approaches machine precision under single-precision floating-point arithmetic, making the proposed methods a competitive addition to the state-of-the-art solvers for convolution.

The rest of this paper is organized as follows. Section 2 provides the necessary preliminaries on residualbased neural network methods and operator networks. Sections 3 and 4 present and analyze the DCNO and

DGNO methods, respectively. Applications to convolution operator learning problems are given in Section 5. Some conclusions are drawn in Section 6.

## 2. Preliminary

In this section, we briefly review some preliminary knowledge that is fundamental for proposing our method. The first part presents a class of residual-based neural network methods for high-accuracy PDE solving, and the second part presents the neural network for operator learning.

## 2.1. Residual-based neural network methods for high-accuracy PDE solving

Although deep neural networks are capable of representing intricate solution structures of PDEs [40], direct one-shot training often faces limitations in accuracy. Residual-based neural network methods in the literature have addressed this challenge by iteratively constructing adaptive approximation spaces using multiple neural networks, thereby progressively improving approximations for PDEs. For illustration here, we briefly present a representative approach, namely the Galerkin neural network (GNN) [2].

Consider the following PDE problem in a bounded domain $\Omega \subset \mathbb { R } ^ { d }$ with boundary ∂Ω:

$$
\left\{ \begin{array} { l l } { \mathscr { L } u ( x ) = f ( x ) , } & { x \in \Omega , } \\ { \mathrm { b o u n d a r y ~ c o n d i t i o n s , } } & { x \in \partial \Omega , } \end{array} \right.\tag{2.1}
$$

where $\mathcal { L }$ is a linear diferential operator and $f ( x )$ is a source term. The bounded setup here is natural for computations. Assume that (2.1) can be reformulated into a variational (weak) form: find

$$
u \in X \quad { \mathrm { s u c h ~ t h a t } } \quad a ( u , v ) = F ( f , v ) , \qquad \forall v \in X .\tag{2.2}
$$

Here, X denotes the solution space, $F ( f , \cdot ) : X \to \mathbb { R }$ is a bounded linear functional on X induced by the source term $f ,$ and $a ( \cdot , \cdot ) : X \times X \to \mathbb { R }$ is a bounded, symmetric, bilinear form defined on X satisfying

$$
a ( v , v ) \geq 0 , { \mathrm { ~ a n d ~ } } a ( v , v ) = 0 \iff v = 0 .
$$

Under this assumption, $a ( \cdot , \cdot )$ induces an inner product on X, with the corresponding norm defined by $\| v \| _ { X } = \sqrt { a ( v , v ) }$ . For example, in (2.1), when $\mathcal { L } = - \Delta$ with homogeneous Dirichlet boundary conditions, one takes $X \ = \ H _ { 0 } ^ { 1 } ( \Omega )$ , and the operator a and F associated with (2.1) read $\begin{array} { r } { a ( u , v ) \ = \ \int _ { \Omega } \nabla u \cdot \nabla v d \mathbf { x } } \end{array}$ $\begin{array} { r } { F ( f , v ) \ = \ \int _ { \Omega } f v d \mathbf { x } . } \end{array}$ The Galerkin subspace within GNN is constructed as follows. Starting from an initial basis function $\psi _ { 0 } ~ = ~ 0$ , at stage $s \geq 1$ , let $\Psi _ { s - 1 } = \operatorname { s p a n } \{ \psi _ { 0 } , \dots , \psi _ { s - 1 } \} \subset X$ denote the current approximation space spanned by the basis functions $\{ \psi _ { j } \} _ { j = 0 } ^ { s - 1 } ,$ , and let $u _ { s - 1 } \in \Psi _ { s - 1 }$ be the corresponding Galerkin approximation. In principle, the most efective enrichment direction is given by the error $e _ { s - 1 } : =$ $u - u _ { s - 1 }$ , since adding this direction to the approximation subspace would recover the exact solution in a single step. Thus, GNN seeks a neural-network basis function $\psi _ { s }$ that approximates the normalized error $\varphi _ { s - 1 } = ( u - u _ { s - 1 } ) / \| u - u _ { s - 1 } \| _ { X }$ . Since the exact solution u is not accessible, the residual functional $R _ { s - 1 } ( f , v ) : = F ( f , v ) - a ( u _ { s - 1 } , v )$ can be used as a substitute due to the fact that the normalized error $\varphi _ { s - 1 }$ mathematically maximizes $R _ { s - 1 } ( f , v )$ over the unit ball in X [2]:

$$
\varphi _ { s - 1 } \in \underset { \Vert v \Vert _ { X } = 1 } { \arg \operatorname* { m a x } } R _ { s - 1 } ( f , v ) , \quad R _ { s - 1 } ( f , \varphi _ { s - 1 } ) = \Vert u - u _ { s - 1 } \Vert _ { X } .\tag{2.3}
$$

This characterization suggests the construction of the new basis function:

$$
\psi _ { s } = \underset { v \neq 0 } { \arg \operatorname* { m a x } } \frac { R _ { s - 1 } ( f , v ) } { \| v \| _ { X } } ,\tag{2.4}
$$

and with the enriched finite-dimensional subspace $\Psi _ { s } = \operatorname { s p a n } \{ \psi _ { 0 } , \dots , \psi _ { s } \} \subset X$ , the Galerkin approximation $u _ { s } \in \Psi _ { s }$ is determined by solving a $( s + 1 ) \times ( s + 1 )$ linear system

$$
u _ { s } \in \Psi _ { s } : \quad a ( u _ { s } , \psi _ { j } ) = F ( v _ { j } ) \quad 0 \leq j \leq s ,\tag{2.5}
$$

for the linear expansion coeficients of $u _ { s }$ on the basis of $\Psi _ { s } .$

In addition to GNN, residual-based neural network methods based on the strong form of PDE (2.1) have also been proposed, such as Multi-level Neural Networks [4] and the Deep Collocation Method [46]. We shall develop some high-accuracy neural operator learning methods by leveraging the residual-based framework.

## 2.2. Deep operator network

In this work, we adopt DeepONet as the neural architecture for operator approximation, designed to learn mappings between infinite-dimensional function spaces [34]. Consider an operator ${ \mathcal { G } } : { \mathcal { W } } \to X$ , mapping an input function $\rho \in \mathcal W$ to an output function $\phi \in X$ , where W and X both are (for simplicity) function spaces defined on $\Omega \subset \mathbb { R } ^ { d }$ with $d \in \mathbb { N } _ { + }$ . For any $\mathbf { x } \in \Omega$ , DeepONet approximates $\mathcal { G } ( \boldsymbol { \rho } )$ via a trunk-branch decomposition:

$$
\mathcal { G } ( \rho ) ( \mathbf { x } ) \approx \mathcal { G } _ { \theta } ( \rho ) ( \mathbf { x } ) = \sum _ { k = 1 } ^ { p } \mathfrak { b } _ { k } ( \rho ( \mathbf { s } _ { 1 } ) , \dots , \rho ( \mathbf { s } _ { m } ) ; \theta _ { \mathfrak { b } } ) \ \mathfrak { t } _ { k } ( \mathbf { x } ; \theta _ { \mathfrak { t } } ) ,\tag{2.6}
$$

where $\theta = \{ \theta _ { \mathfrak { b } } , \theta _ { \mathfrak { t } } \}$ denotes the trainable parameters, with $\theta _ { \mathfrak { b } }$ and $\theta _ { \mathrm { { t } } }$ representing the parameters of the branch and trunk networks, respectively. Here, $\{ \mathbf { s } _ { l } \} _ { l = 1 } ^ { m } \subset \Omega$ are sensor locations, and latent dimension $p \in  { \mathbb { N } } _ { + }$ is the output width of branch and trunk networks.

The branch network is responsible for extracting a p-dimensional latent representation $\mathfrak { b } = ( \mathfrak { b } _ { 1 } , \mathfrak { b } _ { 2 } , \dots , \mathfrak { b } _ { p } )$ $\in \mathbb { R } ^ { p }$ of the input function $\rho$ and is typically implemented as a multilayer perceptron (MLP) [41]. In the standard setting, the branch input consists of the evaluations of $\rho$ at fixed sensors $\{ \mathbf { s } _ { l } \} _ { l = 1 } ^ { m }$ . More generally, the branch input can also be taken as the expansion coeficients of $\rho$ with respect to a prescribed basis of W [12, 34, 35]. The trunk network captures the dependence of the output function on the coordinate $\mathbf { x } \in \Omega$ It is implemented as another MLP that maps x to a latent vector $\mathfrak { t } = ( \mathfrak { t } _ { 1 } , \mathfrak { t } _ { 2 } , \ldots , \mathfrak { t } _ { p } ) \in \mathbb { R } ^ { p } ;$ which can be interpreted as a set of learned basis functions over the output domain. The operator approximation is then constructed through the inner product of b and t, resulting in a continuous representation of G.

Since its introduction, DeepONet has been successfully applied to a wide range of problems [10, 11, 26, 29]. However, its standard supervised training paradigm, which relies on labeled data and one-shot optimization, often sufers from limited approximation accuracy. The aim of this work, on the one hand, is to develop a general multi-stage framework for high-accuracy operator learning by progressively constructing basis operators guided by residuals and improving the accuracy of operator approximation. On the other hand, the generation of high-fidelity training data for labels typically requires repeated calls to traditional numerical solvers, leading to substantial computational cost. Thus, in addition to the multi-stage framework, we also work for an unsupervised operator learning manner that eliminates the need for labeled data.

## 3. Deep collocation neural operator (DCNO) method

In this section, we introduce the deep collocation neural operator (DCNO) method, a supervised framework for high-accuracy operator learning. Instead of training a single, monolithic neural operator to approximate a target operator ${ \mathcal { G } } ( \rho ) = \phi$ directly, DCNO constructs a sequence of basis operators and then approximates the target operator as a linear combination of trained basis operator networks. The following first presents the method and then provides some analysis to support its approximation validity.

## 3.1. Neural network basis operators and collocation scheme

Let us first outline the basic framework. We construct the basis operators through a greedy iterative strategy similarly as DCM and GNN [2, 46] (see also [4, 45, 47]), where each basis operator is implemented as DeepONet (2.6) to approximate the solution residual generated at the current iteration. Beginning with an initial operator $\mathcal { G } _ { 0 } : = \mathcal { G } _ { \theta _ { 0 } }$ , which is chosen as the zero operator in this work, the initial residual is defined as $e _ { 0 } = \mathcal { G } - \mathcal { G } _ { \theta _ { 0 } } = \mathcal { G }$ . The first basis operator $\mathcal { G } _ { \theta _ { 1 } }$ is then trained to approximate the initial residual mapping, i.e., $\mathcal { G } _ { \theta _ { 1 } }$ ≈ e<sub>0</sub>. The resulting approximation for $\mathcal { G }$ is then updated to $\mathcal { G } _ { 1 } = \mathcal { G } _ { \theta _ { 0 } } + \mathcal { G } _ { \theta _ { 1 } }$ . Assume that we are now at some stage $s \geq 2$ with the approximation from the previous stage $\begin{array} { r } { \mathcal { G } _ { s - 1 } = \sum _ { j = 0 } ^ { s - 1 } \mathcal { G } _ { \theta _ { j } } } \end{array}$ , then the corresponding residual operator is defined as $e _ { s - 1 } = \mathcal { G } - \mathcal { G } _ { s - 1 }$ . By training a new basis operator $\mathcal { G } _ { \theta _ { s } }$ to approximate the residual mapping $\rho \mapsto e _ { s - 1 }$ , the update $\begin{array} { r } { \mathcal { G } _ { s } = \sum _ { j = 0 } ^ { s } \mathcal { G } _ { \theta } } \end{array}$ thereby incrementally improves the overall approximation for $\mathcal { G }$

Now, we detail the implementation. We consider here for DCNO in the particular case that solution data $\{ ( \rho ^ { ( b ) } , \phi ^ { ( b ) } ) \} _ { b = 1 } ^ { N _ { b } }$ are available, where $\rho ^ { ( b ) }$ are sampled inputs with $N _ { b } \in \mathbb { N } _ { + }$ and $\phi ^ { ( b ) } = \mathcal { G } ( \rho ^ { ( b ) } )$ are possibly obtained from numerical solvers. Then, each stage can be trained in a supervised manner, as detailed below. Let Ω denote the computational (bounded) domain in practice. The trunk network in (2.6) takes the spatial coordinate $\mathbf { x } \in \Omega$ as input. A set of collocation points is then generated through discretization: $\{ \mathbf { x } _ { k } \} _ { k = 1 , . . . } ^ { N _ { \mathbf { x } } } \subset$ Ω, where $N _ { \mathbf { x } } \in \mathbb { N } _ { + }$ denotes the number of collocation points. Accordingly, for each input sample $\rho ^ { ( b ) }$ $b = 1 , \dots , N _ { b }$ , with corresponding output $\phi ^ { ( b ) }$ , the training data are constructed as $\{ ( \rho ^ { ( b ) } ( { \bf x } _ { k } ) , \phi ^ { ( \bar { b } ) } ( { \bf x } _ { k } ) )$ $k = 1 , \ldots , N _ { \mathbf { x } } \big \} _ { b = 1 } ^ { N _ { b } }$ . Let $\mathcal { G } _ { 0 } = 0$ . For $s \geq 1$ , the residual corresponding to the b-th training sample at stage $s - 1$ is given by

$$
e _ { s - 1 } ( \rho ^ { ( b ) } ) ( \mathbf { x } ) : = \phi ^ { ( b ) } ( \mathbf { x } ) - \mathcal { G } _ { s - 1 } ( \rho ^ { ( b ) } ) ( \mathbf { x } ) , \qquad b = 1 , \dots , N _ { b } .\tag{3.1}
$$

We measure it under a discrete norm:

$$
\| e _ { s - 1 } \| = \sqrt  \frac { 1 } { N _ { b } \times N _ { \mathbf { x } } } \sum _ { b = 1 } ^ { N _ { b } } \sum _ { k = 1 } ^ { N _ { \mathbf { x } } } | e _ { s - 1 } ( \rho ^ { ( b ) } ) ( \mathbf { x } _ { k } ) | ^ { 2 } = : \beta _ { s - 1 } , \quad s \geq 1 .\tag{3.2}
$$

As long as $\beta _ { s - 1 } > 0$ , DCNO could in principle proceed to the next stage and the DeepONet $\mathcal { G } _ { \theta _ { s } }$ is determined by minimizing the empirical loss

$$
\mathrm { L o s s } _ { s } ( \theta ) : = \frac { 1 } { N _ { b } \times N _ { \mathbf { x } } } \sum _ { b = 1 } ^ { N _ { b } } \sum _ { k = 1 } ^ { N _ { \mathbf { x } } } \left| \frac { 1 } { \beta _ { s - 1 } } e _ { s - 1 } ( \rho ^ { ( b ) } ) ( \mathbf { x } _ { k } ) - \mathcal { G } _ { \theta } ( \rho ^ { ( b ) } ) ( \mathbf { x } _ { k } ) \right| ^ { 2 } , \quad \mathrm { a n d } \quad \theta _ { s } \in \arg \operatorname* { m i n } _ { \theta } \left\{ \mathrm { L o s s } _ { s } ( \theta ) \right\}\tag{3.3}
$$

Here, the residual is scaled to order one for practical training eficiency $[ 4 ] .$ , and the approximate operator for $\mathcal { G }$ at stage s reads:

$$
\mathcal { G } _ { s } ( { \boldsymbol \rho } ) : = \sum _ { j = 1 } ^ { s } \beta _ { j - 1 } \mathcal { G } _ { \theta _ { j } } ( { \boldsymbol \rho } ) , \quad s \geq 1 .\tag{3.4}
$$

In our numerical experiments later, the unconstrained optimization problem in (3.3) for every $s \geq 1$ will be solved by Adam optimizer [25] with the Xavier method [16] for parameter initialization. It is common for the optimizer to produce a practical minimizer for (3.3) with Loss $\left( \theta _ { s } \right) > 0$ , which is in fact fine here since DCNO will pass the error as some residual to the next stage for refinement.

This process can be terminated once $\beta _ { s } = \| e _ { s } \| \le \epsilon$ for some $s ,$ where $\epsilon > 0$ is a prescribed tolerance. This means that the target operator has been approximated to the desired accuracy on the training set. If the training data are sampled suficiently large and efective, the final DCNO approximation $\mathcal { G } ( \boldsymbol { \rho } ) \approx \mathcal { G } _ { s } ( \boldsymbol { \rho } )$ can therefore become valid for any input function $\rho$ (inside or outside the training set) from a desired functional space. Note that by construction, $e _ { s } = \mathcal { G } - \mathcal { G } _ { s } = e _ { s - 1 } - \beta _ { s - 1 } \mathcal { G } _ { \theta _ { s } }$ , and so the stopping criterion $\| e _ { s } \| \le \epsilon$ can be replaced equivalently as

$$
\beta _ { s - 1 } \sqrt { \mathrm { L o s s } _ { s } ( \theta _ { s } ) } \le \epsilon .
$$

The whole process of the DCNO method is summarized in Algorithm 1. Note that DCNO itself does not require the target operator $\mathcal { G }$ to be linear, despite that we will apply it later to convolutions.

## 3.2. Numerical analysis

Here we present some numerical analysis for the proposed DCNO method. Let $\tilde { \mathcal { G } } ( \rho )$ and $\mathcal { G } ( \boldsymbol { \rho } )$ denote the exact operator and the reference operator, respectively, where the latter serves as the training label in our algorithm and may deviate from the exact one. They both map $\mathcal { W }$ to $X \subset L ^ { 2 } ( \Omega )$ . With the numerical solution $\begin{array} { r } { \mathcal { G } _ { s } = \sum _ { j = 1 } ^ { s } \beta _ { j - 1 } \mathcal { G } _ { \theta _ { j } } } \end{array}$ at the s-th stage, the stage-wise error is defined by

$$
e _ { s } ( \rho ) : = \mathcal { G } ( \rho ) - \mathcal { G } _ { s } ( \rho ) ,\tag{3.5}
$$

```latex
Algorithm 1 DCNO method
Parameters: tolerance ϵ; maximum step $N _ { e }$ for optimizer
Input: Training dataset $\{ ( \rho ^ { ( b ) } , \phi ^ { ( b ) } ) \} _ { b = 1 } ^ { \hat { N } _ { b } } ;$ collocation points $\{ \mathbf { x } _ { k } \} _ { k = 1 } ^ { N _ { \mathbf { x } } }$
Output: Basis operator set $\{ \mathcal { G } _ { \boldsymbol { \theta } _ { j } } \} _ { j = 1 } ^ { s } ;$ Solution operator $\mathcal { G } _ { s } ( { \boldsymbol \rho } ) \approx \mathcal { G } ( { \boldsymbol \rho } )$
1: Initialize: $s = 0 ; \mathcal { G } _ { 0 } = 0 ;$
2: Part I: Construction of neural operator basis
3: do
4: Set $s \gets s + 1$
5: Compute $e _ { s - 1 } ( \rho ^ { ( b ) } )$ (3.1) for every b and compute (3.2) for $\beta _ { s - 1 }$
6: Initialize DeepONet parameters $\theta _ { s }$
7: for epoch = 1 to epoch = N do
8: compute the loss function by (3.3)
9: update $\theta _ { s }$ via the optimizer
10: end for
11: Update approximation $\begin{array} { r } { \mathcal { G } _ { s } = \sum _ { j = 1 } ^ { s } \beta _ { j - 1 } \mathcal { G } _ { \theta _ { j } } } \end{array}$
12: while $\beta _ { s - 1 } \sqrt { \mathrm { L o s s } _ { s } ( \theta _ { s } ) } > \epsilon$
13: Part II: Inference for new input $\rho$
14: Given $\rho ,$ evaluate $\{ \mathcal { G } _ { \theta _ { j } } ( \rho ) \} _ { j = 1 } ^ { s }$
15: Form the prediction $\begin{array} { r } { \mathcal { \bar { G } } _ { s } ( \rho ) \bar { = } \sum _ { j = 1 } ^ { s } \beta _ { j - 1 } \mathcal { G } _ { \theta _ { j } } ( \rho ) } \end{array}$
```

and by taking the data error $\tilde { e } ( \rho ) : = \tilde { \mathcal { G } } ( \rho ) - \mathcal { G } ( \rho )$ into account, we further define

$$
\tilde { e } _ { s } ( \rho ) : = \tilde { \mathcal { G } } ( \rho ) - \mathcal { G } _ { s } ( \rho ) .\tag{3.6}
$$

We consider the DeepONets for stage-wise construction of DCNO to be from

$$
\mathcal { D } : = \{ \mathcal { G } _ { \theta } ( \rho ) ( \mathbf { x } ) = \mathfrak { b } ( \rho ) \cdot \mathbf { t } ( \mathbf { x } ) \ | \ \mathfrak { b } \in \mathcal { N } _ { b r a n c h } , \mathbf { t } \in \mathcal { N } _ { t r u n k } \} ,\tag{3.7}
$$

where $\mathcal { N } _ { b r a n c h }$ and $\mathcal { N } _ { t r u n k }$ are classes of continuous neural network functions $( \mathrm { e . g . }$ , the MLP class), and each picked ${ \mathfrak { b } } , { \mathfrak { t } }$ are two vectors of the same size so that their inner product makes sense. To quantify the approximation performance over the input space $\mathcal { W } _ { : }$ , we consider a finite measure $\mu$ supported on $\mathcal { W }$ and introduce the Bochner space $L ^ { 2 } ( \mathcal { W } ; L ^ { 2 } ( \overline { { \Omega } } ) )$ ) equipped with the norm [1]

$$
\| v \| _ { L ^ { 2 } ( \mathcal { W } ; L ^ { 2 } ( \Omega ) ) } ^ { 2 } : = \int _ { \mathcal { W } } \| v ( \rho ) \| _ { L ^ { 2 } ( \Omega ) } ^ { 2 } d \mu ( \rho ) , \quad v : \mathcal { W } \to L ^ { 2 } ( \Omega ) .\tag{3.8}
$$

Note that (3.2) can be interpreted as a realization/discretization of (3.8), so we will still denote $\beta _ { s } ~ =$ $\| e _ { s } \| _ { L ^ { 2 } ( \mathcal { W } ; L ^ { 2 } ( \Omega ) ) }$ for simplicity in the following. For the error estimate of the DCNO construction process, we make the following basic assumption.

Assumption 3.1. Suppose that $\mathcal { G } : \mathcal { W } \to X$ is a continuous operator, where W is a compact subset of $C ( { \overline { { \Omega } } } )$ and $X = C ( \overline { { \Omega } } )$ with $0 < | \Omega | < \infty$ and $\textstyle 0 < C : = \int _ { \mathcal { W } } d \mu ( \rho ) < \infty$ . Before the construction process of DCNO in Algorithm 1 stops at some $S \in \mathbb { N } _ { + }$ , the residual operator is non-zero, i.e.,

$$
\beta _ { s } = \| e _ { s } \| _ { L ^ { 2 } ( \mathcal { W } ; L ^ { 2 } ( \Omega ) ) } > 0 , \quad s = 0 , 1 , \dots S - 1 .
$$

We assume that the training and sampling within DCNO are suficient at each stage so that the resulting DeepONet basis $\mathcal { G } _ { \theta _ { s } } ~ f o r ~ s = 1 , \dots S$ do not pollute the universal approximation ability of D up to a prescribed threshold $\tau / \sqrt { C | \Omega | }$

Under Assumption 3.1, we can directly have the following error estimate result.

Theorem 3.2. Fix one $0 < \tau < 1$ and let Assumption 3.1 hold. Then, the error of DCNO is strictly decreasing during construction of each stage, i.e., $\lVert e _ { s } \rVert _ { L ^ { 2 } ( \mathcal { W } ; L ^ { 2 } ( \Omega ) ) } < \lVert e _ { s - 1 } \rVert _ { L ^ { 2 } ( \mathcal { W } ; L ^ { 2 } ( \Omega ) ) }$ , and the error at the s-stage satisfies the bound

$$
\| e _ { s } \| _ { L ^ { 2 } ( \mathcal { W } ; L ^ { 2 } ( \Omega ) ) } \leq \tau ^ { s } \left\| e _ { 0 } \right\| _ { L ^ { 2 } ( \mathcal { W } ; L ^ { 2 } ( \Omega ) ) } , \quad s = 1 , \ldots , S .\tag{3.9}
$$

Taking into account the data error ${ \tilde { e } } ,$ we further have

$$
\| \tilde { e } _ { s } \| _ { L ^ { 2 } ( \mathcal { W } ; L ^ { 2 } ( \Omega ) ) } \leq \| \tilde { e } \| _ { L ^ { 2 } ( \mathcal { W } ; L ^ { 2 } ( \Omega ) ) } + \tau ^ { s } \| e _ { 0 } \| _ { L ^ { 2 } ( \mathcal { W } ; L ^ { 2 } ( \Omega ) ) } , \quad s = 1 , \ldots , S .\tag{3.10}
$$

Proof. Since each $\mathcal { G } _ { \boldsymbol { \theta } _ { i } }$ is taken from $\mathcal { D } _ { : }$ , we have $\mathcal { G } _ { s }$ as defined in (3.4) is a continuous operator and so does $e _ { s } ( \rho )$ given by (3.5) for each s. By the universal approximation theorem for DeepONet [34], the basis operator network generated within DCNO at stage s, i.e., $\mathcal { G } _ { \theta _ { s } } \in \mathcal { D }$ for $1 \leq s \leq S$ , can approximate the last scaled residue $e _ { s - 1 } / \beta _ { s - 1 }$ to a range:

$$
\left| \frac { 1 } { \beta _ { s - 1 } } e _ { s - 1 } ( \rho ) ( \mathbf { x } ) - \mathcal { G } _ { \theta _ { s } } ( \rho ) ( \mathbf { x } ) \right| \leq \frac { \tau } { \sqrt { C | \Omega | } } , \quad \forall \rho \in \mathcal { W } , \ \mathbf { x } \in \Omega .\tag{3.11}
$$

By taking the square on both sides of (3.11) and then integrating with respect to $\mathbf { x } \in \Omega$ , we have

$$
\left\| \frac { 1 } { \beta _ { s - 1 } } e _ { s - 1 } ( \rho ) - \mathcal { G } _ { \theta _ { s } } ( \rho ) \right\| _ { L ^ { 2 } ( \Omega ) } ^ { 2 } = \int _ { \Omega } \left| \frac { 1 } { \beta _ { s - 1 } } e _ { s - 1 } ( \rho ) ( \mathbf { x } ) - \mathcal { G } _ { \theta _ { s } } ( \rho ) ( \mathbf { x } ) \right| ^ { 2 } d \mathbf { x } \leq \frac { \tau ^ { 2 } } { C } , \quad \forall \rho \in \mathcal { W } .
$$

Further integrating the above with respect to $\rho \in \mathcal W$ under the measure $\mu$ and then taking the square root, we are led to $\begin{array} { r } { \mathopen { } \mathclose \bgroup \left\| \frac { 1 } { \beta _ { s - 1 } } e _ { s - 1 } - \mathcal G _ { \theta _ { s } } \aftergroup \egroup \right\| _ { L ^ { 2 } ( \mathcal W ; L ^ { 2 } ( \Omega ) ) } \leq \tau . } \end{array}$ . By the construction of DCNO and the definition (3.5), we have $e _ { s } = e _ { s - 1 } - \beta _ { s - 1 } \mathcal { G } _ { \theta _ { s } }$ , and so

$$
\begin{array} { r } { \| e _ { s } \| _ { L ^ { 2 } ( \mathcal W ; L ^ { 2 } ( \Omega ) ) } \leq \tau \| e _ { s - 1 } \| _ { L ^ { 2 } ( \mathcal W ; L ^ { 2 } ( \Omega ) ) } , \quad s = 1 , \ldots , S . } \end{array}\tag{3.12}
$$

As $\tau < 1$ , the error strictly decreases. Applying the estimate (3.12) recursively, we obtain (3.9).

Then, by subtracting (3.6) from (3.5), we see that

$$
\tilde { e } _ { s } ( \rho ) = \tilde { e } ( \rho ) + e _ { s } ( \rho ) .
$$

Applying the triangle inequality to it yields $\| \tilde { e } _ { s } \| _ { L ^ { 2 } ( \mathcal { W } ; L ^ { 2 } ( \Omega ) ) } \leq \| e _ { s } \| _ { L ^ { 2 } ( \mathcal { W } ; L ^ { 2 } ( \Omega ) ) } + \| \tilde { e } \| _ { L ^ { 2 } ( \mathcal { W } ; L ^ { 2 } ( \Omega ) ) }$ . Using (3.9) further gives (3.10). □

## 4. Deep Galerkin neural operator (DGNO) method

Alternative to DCNO, in this section, we develop an unsupervised high-accuracy operator learning framework, termed the deep Galerkin neural operator (DGNO) method, for approximating the solution operator of a PDE. For illustration, the target PDE here is set as (2.1) with $\mathcal { G } : f \mapsto u$ , the mapping of the input source term to the corresponding solution. Similarly as DCNO, DGNO also builds a finite sum of basis operator networks for approximation, but the training is fully based on equation residue without requiring labeled data.

## 4.1. Constructing neural network basis operator in an unsupervised manner

As we are working with a PDE, to reduce the dependence of regularity, we choose to build DGNO for (2.1) upon its variational formulation (2.2), via a framework analogous to GNN. The progressive construction procedure now is to find DeepONets as basis neural operators based on residuals to enrich the approximation space.

The iterative procedure starts with a zero initial operator $\mathcal { G } _ { 0 } = \mathcal { G } _ { \theta _ { 0 } } = 0 .$ , then the initial normalized error under an input $f$ is $\varphi _ { 0 } ( f ) = ( \mathcal { G } ( f ) - \mathcal { G } _ { 0 } ( f ) ) / \| \mathcal { G } ( f ) - \mathcal { G } _ { 0 } ( f ) \| _ { X } = \mathcal { G } ( f ) / \| \mathcal { G } ( f ) \| _ { X }$ . The first basis operator $\mathcal { G } _ { \theta _ { 1 } }$ is trained to approximate the initial error mapping, i.e., $\mathcal { G } _ { \theta _ { 1 } } \approx \varphi _ { 0 }$ , and the resulting approximation for $\mathcal { G }$ is then updated to $\mathcal { G } _ { 1 } : = \beta _ { 1 } \mathcal { G } _ { \theta _ { 1 } }$ , where the coeficient $\beta _ { 1 }$ will be determined via a Galerkin projection detailed later. Assume that we are now at some stage $s \geq 2$ with the approximation from the previous stage $\begin{array} { r } { \mathcal G _ { s - 1 } = \sum _ { j = 1 } ^ { s - 1 } \beta _ { j } \mathcal G _ { \theta _ { j } } } \end{array}$ , the new basis operator $\mathcal { G } _ { \theta _ { s } }$ is trained to approximate the current optimal update direction, namely the normalized error operator read under an input $f$ as

$$
\varphi _ { s - 1 } ( f ) : = { \frac { { \mathcal { G } } ( f ) - { \mathcal { G } } _ { s - 1 } ( f ) } { \| { \mathcal { G } } ( f ) - { \mathcal { G } } _ { s - 1 } ( f ) \| _ { X } } } .\tag{4.1}
$$

For fixed $f ,$ by (2.3), $\varphi _ { s - 1 } ( f )$ can be achieved by maximizing the normalized PDE-residual functional:

$$
\varphi _ { s - 1 } ( f ) \in \underset { \Vert v \Vert _ { X } = 1 } { \arg \operatorname* { m a x } } R _ { s - 1 } ( f , v ) , \quad \mathrm { w h e r e } \quad R _ { s - 1 } ( f , v ) : = F ( f , v ) - a ( \mathcal { G } _ { s - 1 } ( f ) , v ) .
$$

To extend this pointwise characterization to the operator level, we consider the expectation of the normalized residual functional with respect to a measure $\mu$ for input $f ,$ and further motivated from the GNN strategy (2.4), we are led to the following operator-level optimization problem:

$$
\mathcal { G } _ { \theta _ { s } } \in \mathop { \operatorname { a r g m a x } } _ { \mathcal { G } _ { \theta } \in \mathcal { D } } \mathbb { E } _ { f \sim \mu } \left[ \frac { R _ { s - 1 } ( f , \mathcal { G } _ { \theta } ( f ) ) } { \| \mathcal { G } _ { \theta } ( f ) \| _ { X } } \right] , \quad s \ge 1 .\tag{4.2}
$$

Here, D denotes the class of DeepONets defined in (3.7). The target function of the optimization problem (4.2) is purely based on the residue of PDE, without calling for labeled data, and thus provides an unsupervised way for training the basis operators.

Now we give some details on implementation. At stage $s \geq 1$ , given the set of constructed operators $\{ \mathcal { G } _ { \boldsymbol { \theta } _ { j } } \} _ { j = 0 } ^ { s - 1 }$ , the training set of input $\{ f ^ { ( b ) } \} _ { b = 1 } ^ { N _ { b } }$ , and the approximate solution $\begin{array} { r } { \mathcal { G } _ { s - 1 } ( { f } ^ { ( b ) } ) = \sum _ { i = 0 } ^ { s - 1 } \beta _ { j } ^ { ( b ) } \mathcal { G } _ { \theta _ { j } } ( { f } ^ { ( b ) } ) } \end{array}$ We look for the basis operator $\mathcal { G } _ { \theta _ { s } }$ ≈ $\varphi _ { s - 1 }$ based on (4.2). Here, the expectation is discretized as the empirical average over the training samples in practice, and then we are motivated to train $\theta _ { s }$ of $\mathcal { G } _ { \theta _ { i } }$ by minimizing the loss

$$
\operatorname { L o s s } _ { s } ( \theta ) = - \frac { 1 } { N _ { b } } \sum _ { b = 1 } ^ { N _ { b } } \frac { R _ { s - 1 } ( f ^ { ( b ) } , \mathcal { G } _ { \theta } ( f ^ { ( b ) } ) ) } { \| \mathcal { G } _ { \theta } ( f ^ { ( b ) } ) \| _ { X } } , \quad \theta _ { s } \in \arg \operatorname* { m i n } _ { \theta } \left\{ \operatorname { L o s s } _ { s } ( \theta ) \right\} .\tag{4.3}
$$

To ensure high accuracy, both the residual functional $R _ { s - 1 } ( \cdot , \cdot )$ and the energy norm $\left\| \cdot \right\| _ { X }$ are computed using a Gauss–Legendre quadrature rule on the quadrature points $\left\{ \mathbf { x } _ { k } \right\} _ { k = 1 } ^ { N _ { \mathbf { x } } }$ . With the obtained basis operator set $\{ \mathcal { G } _ { \boldsymbol { \theta } _ { j } } \} _ { j = 1 } ^ { s }$ and an input $f _ { : }$ we seek a Galerkin approximation $\mathcal { G } _ { s } ( f )$ for the solution of (2.2) in span $\{ \mathcal { G } _ { \theta _ { j } } ( f ) \} _ { j = 1 } ^ { s }$ i.e., $\begin{array} { r } { \dot { \mathcal { G } _ { s } } ( f ) ( \mathbf { x } ) = \sum _ { j = 1 } ^ { s } \beta _ { j } ( f ) \mathcal { G } _ { \theta _ { j } } ( f ) ( \mathbf { x } ) } \end{array}$ , such that

$$
a ( \mathcal { G } _ { s } ( f ) , \mathcal { G } _ { \theta _ { j } } ( f ) ) = F ( f , \mathcal { G } _ { \theta _ { j } } ( f ) ) , \quad 1 \leq j \leq s .
$$

The coeficient vector $\beta ( f ) = [ \beta _ { 0 } ( f ) , \dots , \beta _ { s } ( f ) ] ^ { T }$ is thus obtained by solving the linear system

$$
\mathbf { K } ( f ) { \boldsymbol { \beta } } ( f ) = \mathbf { F } ( f ) ,\tag{4.4}
$$

where $\mathbf { K } = ( \mathbf { K } _ { k , j } ) _ { s \times \varepsilon }$ <sub>s</sub> and $\mathbf { F } = ( \mathbf { F } _ { j } ) _ { s \times 1 }$ with entries

$$
\mathbf { K } _ { k , j } ( f ) = a { \big ( } { \mathcal { G } } _ { \theta _ { k } } ( f ) , { \mathcal { G } } _ { \theta _ { j } } ( f ) { \big ) } , \qquad \mathbf { F } _ { j } ( f ) = F { \big ( } f , { \mathcal { G } } _ { \theta _ { j } } ( f ) { \big ) } .
$$

The Galerkin projection scheme (4.4) ensures that for each input $f ,$ it provides the optimal approximation to the solution of the variational problem within the current approximation subspace given by the basis operators. The construction process of DGNO can be stopped when a prescribed maximum stage number $S \geq 1$ is reached. The overall framework of DGNO is summarized in Algorithm 2.

Remark 4.1. The current DGNO is designed based on the bilinear form (2.2). Nevertheless, by incorporating linearization techniques such as fixed-point iteration, the proposed framework could be extended to general nonlinear case, which will be investigated in our future work.

Algorithm 2 DGNO method   
Parameters: maximum stage number $S ;$ maximum step $N _ { e }$ for optimizer   
Input: Training dataset $\{ \check { f } ^ { ( b ) } \} _ { b = 1 } ^ { N _ { b } } ;$ Gaussian quadrature points $\left\{ \mathbf { x } _ { k } \right\} _ { k = 1 } ^ { N _ { x } }$   
Output: Basis operator set $\{ \mathcal { G } _ { \boldsymbol { \theta } _ { j } } \} _ { j = 1 } ^ { S } ;$ Solution $\mathcal G _ { S } \approx \mathcal G ( f )$   
1: Initialize: $\mathcal { G } _ { 0 }  0 ; \Psi _ { s }  \emptyset$   
2: Part I: Greedy construction of neural operator basis   
3: for $s = 1$ to S do   
4: Initialize DeepONet parameters $\theta _ { s }$   
5: for epoch = 1 to $e p o c h = N _ { e }$ do   
6: compute the loss function by (4.3)   
7: update $\theta _ { s }$ via the optimizer   
8: end for   
9: Update basis set: $\Psi _ { s + 1 }  \Psi _ { s } \cup \{ \mathcal { G } _ { \theta _ { s } } \}$   
10: Compute coeficients $\beta ( f ^ { ( b ) } )$ for every b via (4.4)   
11: Update approximation $\begin{array} { r } { \ddot { \mathcal { G } } _ { s } ( \dot { f } ^ { ( b ) } ) = \sum _ { j = 1 } ^ { \check { s } } \beta _ { j } ( \dot { f } ^ { ( b ) } ) \mathcal { G } _ { \theta _ { j } } ( f ^ { ( b ) } ) } \end{array}$ for $b = 1 , \cdots , N _ { b }$   
12: end for   
13: Part II: Inference for new input $f$   
14: Given $f ,$ evaluate $\{ \mathcal { G } _ { \theta _ { j } } ( f ) \} _ { j = 1 } ^ { S }$   
15: Compute coeficient $\ddot { \beta ( f ) }$ via solving linear system (4.4)   
16: Form the prediction $\begin{array} { r } { \mathcal { G } _ { S } ( f ) ( \mathbf { x } ) = \sum _ { j = 1 } ^ { S } \beta _ { j } ( f ) \mathcal { G } _ { \theta _ { j } } ( f ) ( \mathbf { x } ) } \end{array}$

## 4.2. Numerical analysis

Here we perform some numerical analysis for the proposed DGNO method. Our analysis considers a continuous projection neglecting the quadrature errors. To quantify the overall approximation performance over the input space $\mathcal { W } _ { : }$ , we consider a probability measure $\mu$ supported on W and introduce the Bochner space $L ^ { 1 } ( { \boldsymbol { \ W } } ; X )$ equipped with the norm [1]

$$
\| v \| _ { L ^ { 1 } ( \mathcal W ; X ) } : = \int _ { \mathcal W } \| v ( f ) \| _ { X } d \mu ( f ) , \quad v \in L ^ { 1 } ( \mathcal W ; X ) .\tag{4.5}
$$

In the context of solving PDE, it is natural to consider the solution space X as some Sobolev space and we make the following basic assumption.

Assumption 4.2. For some $0 \leq q _ { 1 } , q _ { 2 } < \infty$ , let the target operator ${ \mathcal { G } } : { \mathcal { W } } \to X$ be a continuous operator, where W is a compact subset $o f H ^ { q _ { 1 } } ( \Omega )$ and X is subspace of $H ^ { q _ { 2 } } ( \Omega )$ with the energy norm $| | \cdot | | _ { X }$ continuously controlled by $\| \cdot \| _ { H ^ { q _ { 2 } } ( \Omega ) }$ and Ω a bounded open domain.

We first establish a universal approximation result for DeepONet in the Sobolev space. To do this, we begin with the following lemma.

Lemma 4.3 ([22]). Suppose $0 \leq q < \infty$ and $\Omega \subset \mathbb { R } ^ { d }$ is a bounded open domain. $I f \sigma \in C ^ { q } ( \mathbb { R } )$ is non-constant and bounded, then the space

$$
V ^ { \sigma } = \bigcup _ { N = 1 } ^ { \infty } \{ f _ { N } : \mathbb { R } ^ { d }  \mathbb { R } \middle | f _ { N } ( \mathbf { x } ) = \sum _ { j = 1 } ^ { N } \alpha _ { j } \sigma ( w _ { j } \cdot \mathbf { x } + z _ { j } ) , \forall \alpha _ { j } , z _ { j } \in \mathbb { R } , w _ { j } \in \mathbb { R } ^ { d } \} .
$$

is dense in the Sobolev space $H ^ { q } ( \Omega )$

Based on the neural-network approximation for functions in Sobolev spaces in Lemma 4.3, we establish the following universal approximation in Sobolev space for operator networks.

Lemma 4.4 (Universal approximation for stacked DeepONet in Sobolev norm). Let Assumption 4.2 hold. $I f \sigma \in C ^ { q _ { 2 } } ( \mathbb { R } )$ is non-constant and bounded, then for any $\epsilon > 0$ , there exists a stacked DeepONet $\mathcal { G } _ { \theta } ( f ) ( \mathbf { x } ) : =$ $\begin{array} { r } { \sum _ { l = 1 } ^ { p } \mathfrak { b } _ { l , \theta _ { \mathfrak { b } } } \left( \mathbf { c } _ { K } ( f ) \right) \mathfrak { t } _ { l , \theta _ { \mathfrak { t } } } ( \mathbf { x } ) } \end{array}$ , such that:

$$
\operatorname* { s u p } _ { f \in \mathcal { W } } \lVert \mathcal { G } ( f ) - \mathcal { G } _ { \theta } ( f ) \rVert _ { H ^ { q _ { 2 } } ( \Omega ) } < \epsilon .
$$

Here, $\begin{array} { r } { \mathbf { x } \in \Omega , \ \mathfrak { b } _ { l , \theta _ { \mathrm { b } } } ( \mathbf { c } _ { K } ( f ) ) : = \sum _ { k = 1 } ^ { N _ { l } } \alpha _ { k } ^ { l } \sigma ( w _ { k } ^ { l } \cdot \mathbf { c } _ { K } ( f ) + z _ { k } ^ { l } ) , \ \mathfrak { t } _ { l , \theta _ { \mathrm { t } } } ( \mathbf { x } ) : = \sum _ { j = 1 } ^ { M _ { l } } \alpha _ { j } ^ { l } \sigma ( w _ { j } ^ { l } \cdot \mathbf { x } + z _ { j } ^ { l } ) } \end{array}$ . The coeficient vector $\mathbf { c } _ { K } ( f ) = ( \langle f , \zeta _ { 1 } \rangle _ { H ^ { q _ { 1 } } ( \Omega ) } , \dots , \langle f , \zeta _ { K } \rangle _ { H ^ { q _ { 1 } } ( \Omega ) } ) \in \mathbb { R } ^ { K }$ , where $\{ \zeta _ { k } \} _ { k = 1 } ^ { \infty }$ is an orthonormal basis of $H ^ { q _ { 1 } } ( \Omega )$

Proof. Define the projection operator $P _ { K } : H ^ { q _ { 1 } } ( \Omega )  H ^ { q _ { 1 } } ( \Omega )$ by $\begin{array} { r } { P _ { K } f : = \sum _ { k = 1 } ^ { K } \langle f , \zeta _ { k } \rangle _ { H ^ { q _ { 1 } } } \zeta _ { k } . } \end{array}$ . Since $w \subset$ $H ^ { q _ { 1 } } ( \Omega )$ is compact and $\{ P _ { K } \} _ { K \in \mathbb { N } }$ is a sequence of uniformly bounded projections, we have sup $ \operatorname { \partial } _ { f \in \mathcal { W } } \| P _ { K } f -$ $f \| _ { H ^ { q _ { 1 } } ( \Omega ) }  0$ as $K  \infty$ . Moreover, since $\mathcal { G }$ is continuous on the compact set $\mathcal { W } _ { : }$ it is uniformly continuous. Therefore, for any $\epsilon > 0$ , there exists $\delta > 0$ such that $\lVert f - P _ { K } f \rVert _ { H ^ { q _ { 1 } } ( \Omega ) } < \delta \Longrightarrow \lVert \mathcal { G } ( f ) - \mathcal { G } ( P _ { K } f ) \rVert _ { H ^ { q _ { 2 } } ( \Omega ) } < \epsilon / 4$ Choosing K suficiently large so that s $\begin{array} { r } { \displaystyle \operatorname * { l p } _ { f \in \mathcal { W } } \| P _ { K } f - f \| _ { H ^ { q _ { 1 } } ( \Omega ) } < \delta , } \end{array}$ , we obtain

$$
\operatorname* { s u p } _ { f \in \mathcal { W } } \| \mathcal { G } ( f ) - \mathcal { G } ( P _ { K } f ) \| _ { H ^ { q _ { 2 } } ( \Omega ) } < \frac { \epsilon } { 4 } .
$$

Let $\{ \eta _ { l } \} _ { l = 1 } ^ { \infty }$ be an unit orthonormal basis of $H ^ { q _ { 2 } } ( \Omega )$ . Since $P _ { K } ( \mathcal { W } )$ is compact in $H ^ { q _ { 1 } } ( \Omega )$ and $\mathcal { G }$ is continuous, the set $\mathcal { G } ( P _ { K } ( \mathcal { W } ) )$ is compact in $H ^ { q _ { 2 } } ( \Omega )$ . Similarly, there exists a positive integer $p$ such that

$$
\operatorname* { s u p } _ { f \in \mathcal { W } } \left\| \mathcal { G } ( P _ { K } f ) - \sum _ { l = 1 } ^ { p } \langle \mathcal { G } ( P _ { K } f ) , \eta _ { l } \rangle _ { H ^ { q _ { 2 } } } \eta _ { l } \right\| _ { H ^ { q _ { 2 } } ( \Omega ) } < \frac { \epsilon } { 4 } .
$$

For each $l = 1 , \ldots , p .$ define $\mathcal { B } _ { l } ( \mathbf { c } _ { K } ( f ) ) : = \langle \mathcal { G } ( P _ { K } f ) , \eta _ { l } \rangle _ { H ^ { q _ { 2 } } } , \ f \in \mathcal { W }$ . Since $P _ { K } f$ is uniquely determined by $\mathbf { c } _ { K } ( f ) = \left( \langle f , \zeta _ { 1 } \rangle _ { H ^ { q _ { 1 } } } , \dots , \langle f , \zeta _ { K } \rangle _ { H ^ { q _ { 1 } } } \right)$ , the above definition is well posed. Moreover, $\boldsymbol { B } _ { l }$ is continuous on ${ \bf c } _ { K } ( \mathcal { W } )$ as it is the composition of continuous mappings. Since W is compact in $H ^ { q _ { 1 } } ( \Omega ) , \mathbf { c } _ { K } ( \mathcal { W } )$ is a compact subset of $\mathbb { R } ^ { K }$ . By the universal approximation theorem for continuous functions [13], for each $l = 1 , \ldots , p .$ there exists a neural network $\begin{array} { r } { \mathfrak { b } _ { l , \theta _ { \mathfrak { b } } } ( \mathbf { c } _ { K } ( f ) ) = \sum _ { k = 1 } ^ { N _ { l } } \alpha _ { k } ^ { l } \sigma ( \pmb { w } _ { k } ^ { l } \cdot \mathbf { c } _ { K } ( f ) + z _ { k } ^ { l } ) } \end{array}$ , such that

$$
\operatorname* { s u p } _ { f \in \mathcal { W } } | \mathcal { B } _ { l } ( \mathbf { c } _ { K } ( f ) ) - \mathfrak { h } _ { l , \theta _ { \mathfrak { b } } } ( \mathbf { c } _ { K } ( f ) ) | < \frac { \epsilon } { 4 \sqrt { p } } .
$$

Since $p$ is finite, these approximations can be chosen simultaneously for all $l = 1 , \ldots , p$ . Then,

$$
\begin{array} { r l } { \displaystyle \operatorname* { s u p } _ { f \in \mathcal { W } } \Big \| \displaystyle \sum _ { l = 1 } ^ { p } \big ( \mathcal { B } _ { l } ( \mathbf { c } _ { K } ( f ) ) - \mathfrak { b } _ { l , \theta _ { \mathfrak { b } } } ( \mathbf { c } _ { K } ( f ) ) \big ) \eta _ { l } \Big \| _ { H ^ { \alpha _ { 2 } } ( \Omega ) } = \displaystyle \operatorname* { s u p } _ { f \in \mathcal { W } } \left( \displaystyle \sum _ { l = 1 } ^ { p } | \mathcal { B } _ { l } ( \mathbf { c } _ { K } ( f ) ) - \mathfrak { b } _ { l , \theta _ { \mathfrak { b } } } ( \mathbf { c } _ { K } ( f ) ) | ^ { 2 } \right) ^ { 1 / 2 } } & { } \\ { \displaystyle } & { \leq \left( \displaystyle \sum _ { l = 1 } ^ { p } \displaystyle \operatorname* { s u p } _ { f \in \mathcal { W } } | \mathcal { B } _ { l } ( \mathbf { c } _ { K } ( f ) ) - \mathfrak { b } _ { l , \theta _ { \mathfrak { b } } } ( \mathbf { c } _ { K } ( f ) ) | ^ { 2 } \right) ^ { 1 / 2 } < \frac { \epsilon } { 4 } . } \end{array}
$$

Now, define the network $\begin{array} { r } { \mathbf { t } _ { l , \theta _ { \mathrm { t } } } ( \mathbf { x } ) : = \sum _ { i = 1 } ^ { M _ { l } } \alpha _ { j } ^ { l } \sigma ( w _ { i } ^ { l } \cdot \mathbf { x } + z _ { j } ^ { l } ) , 1 \le l \le p . } \end{array}$ . Since p is finite, for each $1 \leq l \leq p _ { ; }$ by the universal approximation from Lemma $4 . { \overset { } { 3 } } .$ , there exists $M _ { l }$ such that $\lVert \eta _ { l } - \mathfrak { t } _ { l , \theta _ { \mathfrak { t } } } \rVert _ { H ^ { q _ { 2 } } ( \Omega ) } < \delta _ { 0 }$ . Moreover, since ${ \bf c } _ { K } ( \mathcal { W } )$ is compact and $\mathfrak { b } _ { l , \theta _ { \mathfrak { b } } }$ is continuous, there exists a constant $C _ { 0 } > 0$ such that $\vert \mathfrak { b } _ { l , \theta _ { \mathfrak { b } } } ( \mathbf { c } _ { K } ( f ) ) \vert \le$ $C _ { 0 } , \ \forall f \in \mathcal { W } , \ 1 \leq l \leq p .$ Defining $\begin{array} { r } { \mathcal { G } _ { \theta } ( f ) ( \mathbf { x } ) : = \sum _ { l = 1 } ^ { p } \mathfrak { b } _ { l , \theta _ { \mathfrak { b } } } ( \mathbf { c } _ { K } ( f ) ) \mathfrak { t } _ { l , \theta _ { \mathfrak { t } } } ( \mathbf { x } ) } \end{array}$ and by the triangle inequality, we have

$$
\begin{array} { r l } { \displaystyle \operatorname* { s u p } _ { f \in \mathcal { W } } \left\| \displaystyle \sum _ { l = 1 } ^ { p } \mathsf { b } _ { l , \theta _ { \mathfrak { b } } } ( \mathbf { c } _ { K } ( f ) ) \eta _ { l } - \mathcal { G } _ { \theta } ( f ) \right\| _ { H ^ { q _ { 2 } } ( \Omega ) } \leq \displaystyle \operatorname* { s u p } _ { f \in \mathcal { W } } \Big ( \displaystyle \sum _ { l = 1 } ^ { p } | \mathsf { b } _ { l , \theta _ { \mathfrak { b } } } ( \mathbf { c } _ { K } ( f ) ) | \cdot \| \eta _ { l } - \mathfrak { t } _ { l , \theta _ { \mathfrak { t } } } \| _ { H ^ { q _ { 2 } } ( \Omega ) } \Big ) } & { } \\ { \leq C _ { 0 } \displaystyle \sum _ { l = 1 } ^ { p } \| \eta _ { l } - \mathfrak { t } _ { l , \theta _ { \mathfrak { t } } } \| _ { H ^ { q _ { 2 } } ( \Omega ) } \leq C _ { 0 } p \delta _ { 0 } . } \end{array}
$$

By choosing $\delta _ { 0 } = \epsilon / ( 4 C _ { 0 } p )$ , we obtain $\left\| \sum _ { l = 1 } ^ { p } \mathfrak { h } _ { l , \theta _ { \mathfrak { b } } } ( \mathbf { c } _ { K } ( f ) ) \eta _ { l } - \mathcal { G } _ { \theta } ( f ) \right\| _ { H ^ { q _ { 2 } } ( \Omega ) } < \epsilon / 4$ . Then we have

$$
\begin{array} { r l } { \displaystyle \operatorname* { s u p } _ { f \in \mathcal { W } } \| \mathcal { G } ( f ) - \mathcal { G } _ { \theta } ( f ) \| _ { H ^ { q _ { 2 } } ( \Omega ) } \le \displaystyle \operatorname* { s u p } _ { f \in \mathcal { W } } \Big ( \| \mathcal { G } ( f ) - \mathcal { G } ( P _ { K } f ) \| _ { H ^ { q _ { 2 } } ( \Omega ) } + \left\| \mathcal { G } ( P _ { K } f ) - \displaystyle \sum _ { l = 1 } ^ { p } \mathcal { B } _ { l } ( \mathbf { c } _ { K } ( f ) ) \eta _ { l } \right\| _ { H ^ { q _ { 2 } } ( \Omega ) } } & { } \\ { + \displaystyle \Big \| \displaystyle \sum _ { l = 1 } ^ { p } [ \mathcal { B } _ { l } ( \mathbf { c } _ { K } ( f ) ) - \mathfrak { b } _ { l , \theta _ { \mathfrak { h } } } ( \mathbf { c } _ { K } ( f ) ) ] \eta _ { l } \Big \| _ { H ^ { q _ { 2 } } ( \Omega ) } + \left\| \displaystyle \sum _ { l = 1 } ^ { p } \mathfrak { b } _ { l , \theta _ { \mathfrak { h } } } ( \mathbf { c } _ { K } ( f ) ) \eta _ { l } - \mathcal { G } _ { \theta } ( f ) \right\| _ { H ^ { q _ { 2 } } ( \Omega ) } \Big ) } & { } \\ { < \epsilon . } \end{array}
$$

Lemma 4.4 only considers DeepONet constructed from stacked shallow networks. The following extends the result by allowing for more general branch and trunk networks.

Lemma 4.5 (Generalized universal approximation for DeepONet in Sobolev norms). Under Assumption 4.2, for any $\epsilon > 0$ , there exist positive integers $K , p _ { ; }$ , continuous functionals $c _ { 1 } , c _ { 2 } , \ldots , c _ { K } : H ^ { q _ { 1 } } ( \Omega ) \to \mathbb { R }$ and continuous vector functions $\mathfrak { b } : \mathbb { R } ^ { K } \to \bar { \mathbb { R } ^ { p } } , \mathfrak { t } : \mathbb { R } ^ { \bar { d } } \to \mathbb { R } ^ { p }$ , such that

$$
\operatorname* { s u p } _ { f \in \mathcal { W } } \lVert \mathcal { G } ( f ) - \underbrace { \mathfrak { b } ( c _ { 1 } ( f ) , c _ { 2 } ( f ) , \ldots , c _ { K } ( f ) ) } _ { b r a n c h } \cdot \underbrace { \mathfrak { t } } _ { t r u n k } \rVert _ { H ^ { q _ { 2 } } ( \Omega ) } < \epsilon .\tag{4.6}
$$

The functions b and t can be chosen as diverse classes of neural networks which satisfy the classical universal approximation theorem of functions, including (stacked/unstacked) fully connected neural networks and convolutional neural networks.

Proof. Let ${ \sf b } = ( { \sf b } _ { 1 } , \cdots , { \sf b } _ { p } ) ^ { T }$ and $\mathbf { t } = ( \mathbf { t } _ { 1 } , \cdots , \mathbf { t } _ { p } ) ^ { T }$ . We choose $\begin{array} { r } { \mathfrak { b } _ { l } \big ( \mathbf { c } _ { K } \big ( f \big ) \big ) = \sum _ { k = 1 } ^ { N _ { l } } \alpha _ { k } ^ { l } \sigma \big ( \pmb { w } _ { k } ^ { l } \cdot \mathbf { c } _ { K } \big ( f \big ) + z _ { k } ^ { l } \big ) } \end{array}$ , and $\begin{array} { r } { \mathfrak { t } _ { l } ( \mathbf { x } ) = \sum _ { j = 1 } ^ { M _ { l } } \alpha _ { j } ^ { l } \sigma ( \pmb { w } _ { j } ^ { l } \cdot \mathbf { x } + z _ { j } ^ { l } ) } \end{array}$ for $\mathbf { x } \in \mathbb { R } ^ { d }$ , where all the notations involved are defined in Lemma 4.4. Then, this result holds immediately following Lemma 4.4.

Note that the domains of b and t can be restricted on compact sets. In fact, t is defined on Ω, and b can be defined on $[ - Z , Z ] ^ { K }$ , where $Z > 0$ is a uniform bound of the functions in W. Therefore, b and t can be approximated by neural networks according to the universal approximation of functions [13]. □

Now, we are ready to present a convergence result for the proposed DGNO method.

Theorem 4.6. Under Assumption 4.2, fix one $0 < \tau < 1$ and assume that the training and sampling within DGNO are suficient at each stage so that the resulting DeepONet basis $\mathcal { G } _ { \theta _ { s } } ~ f o r ~ s = 1 , 2 , . . . S$ can meet (4.6) with $\epsilon = \tau$ . Then, the following error estimates hold.

1) The error of DGNO is strictly decreasing during construction of each stage, i.e., $\left\| e _ { s } \right\| _ { L ^ { 1 } ( \mathcal { W } ; \boldsymbol { X } ) } <$ $\| e _ { s - 1 } \| _ { L ^ { 1 } ( \mathcal { W } ; X ) }$ , and the error at the s-stage satisfies the bound

$$
\| e _ { s } \| _ { L ^ { 1 } ( \mathcal { W } ;  { \boldsymbol { X } } ) } < \tau ^ { s } \left\| e _ { 0 } \right\| _ { L ^ { 1 } ( \mathcal { W } ;  { \boldsymbol { X } } ) } .\tag{4.7}
$$

2) Define $\begin{array} { r } { \mathcal { T } _ { s - 1 } ( \mathcal { G } _ { \theta _ { s } } ) : = \int _ { \mathcal { W } } R _ { s - 1 } ( f , \mathcal { G } _ { \theta _ { s } } ( f ) ) d \mu ( f ) } \end{array}$ , then the error at stage $s - 1$ of DGNO satisfies the bound

$$
\frac { 1 } { 1 + \tau } \mathcal { I } _ { s - 1 } ( \mathcal { G } _ { \theta _ { s } } ) < \| e _ { s - 1 } \| _ { L ^ { 1 } ( \mathcal { W } ; \mathit { X } ) } < \frac { 1 } { 1 - \tau } \mathcal { I } _ { s - 1 } ( \mathcal { G } _ { \theta _ { s } } ) .\tag{4.8}
$$

Proof. By the definition (4.1), $\varphi _ { s - 1 }$ is a continuous operator from $\mathcal { W } \subset H ^ { q _ { 1 } } ( \Omega )$ to $X \subset H ^ { q _ { 2 } } ( \Omega )$ . Under the assumption and also by Lemma 4.5, the basis operator network generated within DGNO at stage s, i.e., $\mathcal { G } _ { \theta _ { s } } \in \mathcal { D }$ for $1 \leq s \leq S$ , can approximate the $\varphi _ { s - 1 }$ to a range:

$$
\| \mathcal { G } _ { \theta _ { s } } ( f ) - \varphi _ { s - 1 } ( f ) \| _ { X } < \tau , \quad \forall f \in \mathcal { W } .\tag{4.9}
$$

For any fixed $f \in \mathcal W$ , note that $\mathcal { G } _ { s - 1 } + \Vert e _ { s - 1 } ( f ) \Vert _ { X } \mathcal { G } _ { \theta _ { s } } ( f ) \in \mathrm { s p a n } \{ \mathcal { G } _ { \theta _ { j } } ( f ) \} _ { j = 1 } ^ { s }$ . Hence, by the property of Galerkin projection (see also [2]), we have

$$
\begin{array} { r l } & { \| e _ { s } ( f ) \| _ { X } = \| \mathcal { G } ( f ) - \mathcal { G } _ { s } ( f ) \| _ { X } \le \| \mathcal { G } ( f ) - \mathcal { G } _ { s - 1 } ( f ) - \| e _ { s - 1 } ( f ) \| _ { X } \mathcal { G } _ { \theta _ { s } } ( f ) \| _ { X } } \\ & { \qquad = \| \| e _ { s - 1 } ( f ) \| _ { X } \varphi _ { s - 1 } ( f ) - \| e _ { s - 1 } ( f ) \| _ { X } \mathcal { G } _ { \theta _ { s } } ( f ) \| _ { X } } \\ & { \qquad = \| e _ { s - 1 } ( f ) \| _ { X } \| \varphi _ { s - 1 } ( f ) - \mathcal { G } _ { \theta _ { s } } ( f ) \| _ { X } } \\ & { \qquad < \tau \| e _ { s - 1 } ( f ) \| _ { X } . } \end{array}\tag{4.10}
$$

where the final step follows (4.9). Integrating over W with respect $f$ under the measure $\mu$ immediately yields $\| e _ { s } \| _ { L ^ { 1 } ( \mathcal { W } ; X ) } < \tau \left\| e _ { s - 1 } \right\| _ { L ^ { 1 } ( \mathcal { W } ; X ) }$ . As $\tau < 1$ , the error strictly decreases. Applying this estimate recursively, we obtain (4.7).

By the definition of $R _ { s - 1 }$ , for every $f \in \mathcal { W } , R _ { s - 1 } ( f , \varphi _ { s - 1 } ( f ) ) = \| e _ { s - 1 } ( f ) \| _ { X }$ . It is then direct to deduce that

$$
\begin{array} { r l } & { \quad \big | R _ { s - 1 } \big ( f , \mathcal { G } _ { \theta _ { s } } ( f ) \big ) - \| e _ { s - 1 } ( f ) \| _ { X } \big | = \big | R _ { s - 1 } \big ( f , \mathcal { G } _ { \theta _ { s } } ( f ) \big ) - R _ { s - 1 } \big ( f , \varphi _ { s - 1 } ( f ) \big ) \big | = \big | R _ { s - 1 } \big ( f , \mathcal { G } _ { \theta _ { s } } ( f ) - \varphi _ { s - 1 } ( f ) \big ) \big | } \\ & { = \big | F \big ( f , \mathcal { G } _ { \theta _ { s } } ( f ) - \varphi _ { s - 1 } ( f ) \big ) - a \big ( \mathcal { G } _ { \theta _ { s - 1 } } ( f ) , \mathcal { G } _ { \theta _ { s } } ( f ) - \varphi _ { s - 1 } ( f ) \big ) \big | } \\ & { = \big | a \big ( \mathcal { G } ( f ) , \mathcal { G } _ { \theta _ { s } } ( f ) - \varphi _ { s - 1 } ( f ) \big ) - a \big ( \mathcal { G } _ { \theta _ { s - 1 } } ( f ) , \mathcal { G } _ { \theta _ { s } } ( f ) - \varphi _ { s - 1 } ( f ) \big ) \big | = \big | a \big ( e _ { s - 1 } ( f ) , \mathcal { G } _ { \theta _ { s } } ( f ) - \varphi _ { s - 1 } ( f ) \big ) \big | . } \end{array}
$$

By the Cauchy–Schwarz inequality and (4.10), we find

$$
\begin{array} { r l } { \big | R _ { s - 1 } \big ( f , \mathcal { G } _ { \theta _ { s } } ( f ) \big ) - \| e _ { s - 1 } ( f ) \| _ { X } \big | = \big | a \big ( e _ { s - 1 } ( f ) , \mathcal { G } _ { \theta _ { s } } ( f ) - \varphi _ { s - 1 } ( f ) \big ) \big | } & { } \\ { \leq \| e _ { s - 1 } ( f ) \| _ { X } \| \mathcal { G } _ { \theta _ { s } } ( f ) - \varphi _ { s - 1 } ( f ) \| _ { X } < \tau \| e _ { s - 1 } ( f ) \| _ { X } , } & { } \end{array}
$$

Thus,

$$
( 1 - \tau ) \| e _ { s - 1 } ( f ) \| _ { X } < R _ { s - 1 } \big ( f , \mathcal { G } _ { \theta _ { s } } ( f ) \big ) < ( 1 + \tau ) \| e _ { s - 1 } ( f ) \| _ { X } ,
$$

and integrating over W immediately yields (4.8).

## 5. Application to convolution problems

In this section, we apply the proposed DCNO and DGNO methods to solve the convolution problem (1.1). We first detail the implementation of the methods for the convolution operator, followed by a series of numerical experiments to validate and compare accuracy and eficiency. Extensions to the multi-input case are made in the end. All experiments<sup>1</sup> are conducted using a single CPU with 32 GB RAM for CPU-based methods and a single NVIDIA A800-PCIE-80GB GPU for GPU-based methods.

## 5.1. Implementation of DCNO & DGNO on convolution

Focusing on the convolution problem (1.1): $\phi ( \mathbf { x } ) = { \mathcal { G } } ( \rho ) ( \mathbf { x } ) = ( K * \rho ) ( \mathbf { x } )$ , the target operator $\mathcal { G }$ maps the input density $\rho$ to the solution potential ϕ. The generality of G can separate the use of DCNO from DGNO. When $\mathcal { G }$ does not have a precise (local) PDE formulation, then DGNO might not be applicable. In such a case, we would employ DCNO for supervised training using data pairs $\{ ( \check { \rho } ^ { ( b ) } , \phi ^ { ( b ) } ) \} _ { b = 1 } ^ { \tilde { N _ { b } } }$ that could be generated by traditional numerical convolution solvers. By sampling the input density function $\rho ^ { ( b ) }$ , the solver produces accurate enough $\phi ^ { ( b ) } = \mathcal { G } ( \rho ^ { ( b ) } )$ . After the training dataset is constructed, the learning of the convolution operator via DCNO can be performed as in Algorithm 1.

When $\mathcal { G }$ can be expressed through a proper PDE, ${ \mathrm { i . e . , ~ } } ( 2 . 1 )$ , DGNO becomes available and its unsupervised training can certainly be rewarded with more eficiency. The task of learning the operator $f \mapsto u$ in (2.1) then follows the DGNO procedure in Algorithm 2. If the original convolution problem (1.1) is imposed in the whole space: $\begin{array} { r } { \phi ( \mathbf { x } ) = \int _ { \mathbb { R } ^ { d } } K ( \mathbf { x } , \tilde { \mathbf { x } } ) \rho ( \tilde { \mathbf { x } } ) d \tilde { \mathbf { x } } } \end{array}$ , the resulting equivalent whole PDE problem, for computational reasons, usually needs to be truncated onto a bounded domain Ω with appropriate/approximate boundary conditions. Careless truncations may cause loss of accuracy or require a very large domain size. To enlarge computational eficiency in such a case, we here combine with a moment-matching strategy [31] briefly presented below, which can accurately approximate the whole-space convolution as a local PDE with zero boundary conditions in a bounded domain.

The moment-matching strategy begins with a decomposition [7, 31] on the density: $\rho ~ = ~ \rho _ { \mathrm { m m } } + f$ with $\rho _ { \mathrm { m m } }$ constructed for moment-matching and f the residual. Here, $\rho _ { \mathrm { m m } }$ is assumed to be $\rho _ { \mathrm { m m } } ( \mathbf { x } ) =$ $\scriptstyle \sum _ { | \alpha | = 0 } ^ { m } \gamma _ { \alpha } \partial ^ { \alpha } G ( \mathbf { x } )$ , with $\begin{array} { r } { G ( \mathbf { x } ) = ( 2 \pi \gamma ^ { 2 } ) ^ { - d / 2 } \exp ( - \frac { | \mathbf { x } | ^ { 2 } } { 2 \gamma ^ { 2 } } ) } \end{array}$ , where α the multi-index and the coeficients $\{ \gamma _ { \alpha } \}$ are determined by imposing the moment-matching condition up to order m:

$$
\int _ { \mathbb { R } ^ { d } } \rho _ { \mathrm { m m } } ( \mathbf { x } ) \mathbf { x } ^ { \alpha } d \mathbf { x } = \int _ { \mathbb { R } ^ { d } } \rho ( \mathbf { x } ) \mathbf { x } ^ { \alpha } d \mathbf { x } , \quad { \mathrm { f o r ~ } } \vert \alpha \vert = 0 , 1 , \ldots , m .\tag{5.1}
$$

Conditions in (5.1) read equivalently as a linear system: $\begin{array} { r } { \sum _ { | \alpha | = 0 } ^ { m } \gamma _ { \alpha } \int _ { \mathbb { R } ^ { d } } \partial ^ { \alpha } G ( \mathbf { x } ) \mathbf { x } ^ { \beta } d \mathbf { x } = \int _ { \mathbb { R } ^ { d } } \rho ( \mathbf { x } ) \mathbf { x } ^ { \beta } } \end{array}$ dx, for $| \beta | = 0 , 1 , \ldots , m ,$ which admits an analytical solution [31]. The advantage of this construction is that $\phi _ { \mathrm { m m } } = K * \rho _ { \mathrm { m m } }$ can be analytically evaluated and for the residual $u = \phi - \phi _ { \mathrm { m m } } = K * f .$ , the condition $\left( 5 . 1 \right)$ ensures a rapid decay of $f$ at infinity. This in turn can lead to a rapid far-field decay also on u for K satisfying suitable far-field asymptotic behavior [31], which is the case for Green’s functions of common elliptic operators like the Laplacian. Then, the unbounded-domain problem can be truncated to a Ω not too large and u can be accurately approximated as the solution of

$$
\left\{ \begin{array} { l l } { \mathcal { L } u ( \mathbf { x } ) = f ( \mathbf { x } ) , } & { \mathbf { x } \in \Omega , } \\ { u ( \mathbf { x } ) = 0 , } & { \mathbf { x } \in \partial \Omega . } \end{array} \right.\tag{5.2}
$$

Coming back to the convolution operator learning task, with the sampled training set $\{ \rho ^ { ( b ) } \} _ { b = 1 } ^ { N _ { b } }$ , now under this strategy, we first compute the corresponding $\{ f ^ { ( b ) } = \rho ^ { ( b ) } - \rho _ { \mathrm { m m } } ^ { ( b ) } \} _ { b = 1 } ^ { N _ { b } }$ as inputs associated with (5.2) and then DGNO in Algorithm 2 can be applied for $\mathcal { G } _ { S } ( f ) \approx u .$ . Thus, for any given $\rho ,$ the final prediction for the convolution outcome ϕ under our method is $\phi _ { \mathrm { m m } } + \mathcal { G } _ { S } ( f )$

By the structure of DeepONet (2.6), the trained $\mathcal { G } _ { S }$ from DCNO or DGNO can give the predicted solution at $N \in  { \mathbb { N } } _ { + }$ arbitrary spatial points with computational complexity $\mathcal { O } ( N )$ . In comparison, most existing traditional solvers such as the hierarchical matrix technique [21], the Gaussian-Summation method [15] and the kernel truncation method [30, 42] cost O(N log N) for the prescribed N grid points. For a new spatial point outside the prescribed grid, traditional solvers will need to do extra delicate interpolations, $\mathrm { e . g . }$ , the NUFFT [18], if one wants to maintain high accuracy. For the traditional method SOE [50], which is specially designed to achieve $\mathcal { O } ( N )$ cost for one-dimensional convolution, we will provide a numerical comparison below.

## 5.2. Numerical experiments

We now carry out some numerical experiments to illustrate the performance of DCNO and DGNO. The involved DeepONet (2.6) uses identical depth L and width W for both branch and trunk networks, with the same number of neurons in each hidden layer for simplicity.

Example 5.1 (Poisson potential). We begin with a 2D Poisson potential in $( 1 . 1 ) , i . e . , d = 2 , \mathbf { x } = ( x , y ) \in \mathbb { R } ^ { 2 }$ and the convolution kernel $\begin{array} { r } { K ( x , y ) = - \frac { 1 } { 2 \pi } \ln { \sqrt { x ^ { 2 } + y ^ { 2 } } } } \end{array}$ . We consider the density function of form $\rho ( x , y ) =$ $\mathrm { e } ^ { - ( x ^ { 2 } + y ^ { 2 } ) / \alpha ^ { 2 } }$ with $\alpha > 0 _ { : }$ , and the corresponding analytical solution of this problem is given by

$$
\phi ( x , y ) = - \frac { \alpha ^ { 2 } } 4 \left[ - \mathrm { E i } \left( - \frac { x ^ { 2 } + y ^ { 2 } } { \alpha ^ { 2 } } \right) + \ln ( x ^ { 2 } + y ^ { 2 } ) \right] , \quad s a t i f y i n g \ - \ \Delta \phi = \rho \ o r \phi = K \ast \rho \ o n \mathbb { R } ^ { 2 } .
$$

Here, $\begin{array} { r } { \mathrm { E i } ( r ) = \int _ { - \infty } ^ { r } t ^ { - 1 } \mathrm { e } ^ { t } \mathrm { d } t } \end{array}$ . Note that the limiting value at the origin is finite: $\begin{array} { r } { \phi ( 0 , 0 ) = \frac { \alpha ^ { 2 } } { 4 } \left( \gamma _ { E } - 2 \ln \alpha \right) } \end{array}$ with γ ≈ 0.5772156649 the Euler–Mascheroni constant. Our goal is to learn the operator $\mathcal { G } : \rho \mapsto \phi$ and the computational domain is set as $\Omega = ( - I , I ) ^ { 2 }$ with $I = 1 2$

![](images/c563c7ae7855c93fbbf21a1218d6cc4c285e9b52e1d8598bd2d50c0332a3d46f.jpg)  
Figure 1: Results of DCNO in Algorithm 1 for Example 5.1: (a) Change of Loss= $\beta _ { s - 1 } \sqrt { \mathrm { L o s s } _ { s } }$ during the training; (b) Mean relative $L ^ { 2 }$ error with respect to the number of stages on test set.

![](images/a26c60482e8347bfb9bf41a25f075ed6a5694a084b4f86ca8954504fdb695d91.jpg)

Table 1: Relative $L ^ { 2 } .$ -errors obtained by DCNO and DGNO under diferent network architectures.
<table><tr><td rowspan="2"></td><td colspan="4">DCNO</td><td colspan="4">DGNO</td></tr><tr><td> $\overline { { W _ { 0 } = 1 0 } }$ </td><td> $\overline { { W _ { 0 } = 3 0 } }$ </td><td> $\overline { { W _ { 0 } = 5 0 } }$ </td><td> $\overline { { W _ { 0 } = 7 0 } }$ </td><td> $\overline { { W _ { 0 } = 1 0 } }$ </td><td> $\overline { { W _ { 0 } = 3 0 } }$ </td><td> $\overline { { W _ { 0 } = 5 0 } }$ </td><td> $\overline { { W _ { 0 } = 7 0 } }$ </td></tr><tr><td> $\overline { { L = 1 } }$ </td><td> $\overline { { 1 . 7 5 ~ e - 3 } }$ </td><td> $\overline { { 1 . 8 1 e - 3 } }$ </td><td> $\overline { { 1 . 2 0 e - 3 } }$ </td><td> $5 . 3 7 e - 4$ </td><td> $\overline { { 2 . 9 9 e - 3 } }$ </td><td> $2 . 1 5 ~ e { \mathrm { - } } 3$ </td><td> $\overline { { 1 . 2 7 e - 3 } }$ </td><td> $6 . 7 0 e { - 4 }$ </td></tr><tr><td> $L = 2$ </td><td> $1 . 9 4 \ : e - 5$ </td><td> $1 . 8 0 e { - 5 }$ </td><td> $1 . 1 5 e { - 5 }$ </td><td> $9 . 4 6 \ : e - 6$ </td><td> $5 . 8 0 e { - 5 }$ </td><td> $3 . 1 7 e - 5$ </td><td> $2 . 3 8 e - 5$ </td><td> $1 . 6 2 e { - 5 }$ </td></tr><tr><td> $L = 3$ </td><td> $4 . 0 1 e { - 6 }$ </td><td> $3 . 1 7 e - 6$ </td><td> $1 . 4 8 e { - 6 }$ </td><td> $8 . 2 9 e - 7$ </td><td> $1 . 6 8 e { - 5 }$ </td><td> $6 . 0 3 e { - 6 }$ </td><td> $3 . 4 7 e - 6$ </td><td> $9 . 2 0 e - 7$ </td></tr><tr><td> $L = 4$ </td><td> $1 . 6 1 e { - 6 }$ </td><td> $1 . 2 4 e { - 6 }$ </td><td> $7 . 0 4 e - 7$ </td><td> $5 . 4 7 e - 7$ </td><td> $1 . 4 3 e - 5$ </td><td> $3 . 1 0 e { - 6 }$ </td><td> $1 . 8 9 e { - 6 }$ </td><td> $6 . 1 7 e - 7$ </td></tr></table>

![](images/7c69d01c1186cf00bb998540e0630e94f944fcc51c7bd2a9efe3f5fe910af666.jpg)

![](images/cf021123dd1a4ca55feb77ed4da56cb45957cc9848fb54cb40f9040e4ae2b250.jpg)

![](images/08bc4e014951f4a15bbd36a5f2b49bc8d72b2a0fb80fb1cc3f99aa8a67eec6bd.jpg)

![](images/598c8023519b9e3d33af1a54da63993c462b2a3d6cd2e7c1d83ba176cb502c25.jpg)

![](images/042fab027ad7c82f1bb20b1873ace165ac653338b43d1a9336d5bf4e05fd8d6e.jpg)

![](images/1f80be937980465a40c7deb704f74cdd1b3a0e6a4071b30d2e1afe05c3118273.jpg)  
 -  
Figure 2: Numerical solution (1st row) and pointwise error (2nd row) of DCNO for Example 5.1 on three test samples (a) α = 1; (b) α = 3; (c) α = 5.

We first test the performance of DCNO in the supervised scenario by applying Algorithm 1 to Example $5 . 1$ . For training, a dataset $\{ ( \rho ^ { ( b ) } , \phi ^ { ( b ) } ) \} _ { b = 1 } ^ { N _ { b } }$ is generated with samples $N _ { b } = 1 0 0$ , where each parameter $\alpha ^ { ( b ) } { \mathrm { ~ o f ~ } } \rho ^ { ( b ) }$ is independently sampled from the uniform distribution U[1, 5]. Motivated by the strategy in $[ 2 , 4 , 4 5 , 4 6 ]$ , we take the width parameters $W$ and $p$ of the DeepONet (2.6) in a stage-dependent manner: $W = W _ { 0 } + 2 0 \left( s - 1 \right)$ and $p = 1 2 0 + 2 0 ( s - 1 )$ , where s denotes the stage index. The base width is selected as $W _ { 0 } \in \{ 1 0 , 3 0 , 5 0 , 7 0 \}$ , and the network depth is chosen as $L \in \{ 1 , 2 , 3 , 4 \}$ The input coordinates of the trunk network are sampled from a uniform spatial grid defined by $x _ { k } = - I + 2 k I / N _ { x } , k = 1 , \ldots , N _ { x } .$ $y _ { k } = - I + 2 k I / N _ { y } , k = 1 , \ldots , N _ { y } ,$ where $N _ { x } = N _ { y } = 1 0 0$ is used during training and a finer grid with $N _ { x } = N _ { y } = 1 5 0$ is adopted for testing error. The activation function is chosen as $\sigma = \sin$ , and the learning rate is set to $5 \times 1 0 ^ { - 4 }$ With $L = 4$ and $W _ { 0 } = 7 0$ fixed, the stopping criterion in Algorithm 1 is set to $\epsilon = 2 \times 1 0 ^ { - 6 }$ . Fig. 1(a) shows the evolution of $\beta _ { s - 1 } \sqrt { \mathrm { L o s s } _ { s } }$ during training at each stage, with a decay observed as the number of training iterations increases.

![](images/75b6e95983b31886632dd2032563e30e355ac0fc342c25a9daba477b476360af.jpg)  
Figure 3: Basis functions generated by the basis operators of DCNO method.

For testing, 50 samples disjoint from the training set are randomly selected. The corresponding solutions are predicted by DCNO, and the performance is evaluated by the mean relative $L ^ { 2 }$ error over all samples in the test set. The errors of the numerical solution from DCNO under diferent depth and width are presented in Table 1. It can be observed that increasing the network width and depth consistently reduces the error, indicating improved approximation of the numerical solution. With $L = 4$ and $W _ { 0 } = 7 0$ fixed, Fig. 1(b) illustrates a clear decrease in the test error as the number of basis operators increases, reaching eventually $5 . 4 7 \times 1 0 ^ { - 7 }$ which is near the machine precision under single-precision floating-point arithmetic. Notably, without the progressive construction of basis operators employed in DCNO, the accuracy at stage 1 corresponds to that of the standard DeepONet, which is only $8 . 1 7 \times 1 0 ^ { - 3 }$ . This demonstrates the significant advantage of DCNO in improving the accuracy of operator learning. Fig. 2 shows the DCNO numerical solutions for several test samples, along with the corresponding pointwise error distributions. As observed, DCNO method can accurately predict the potential ϕ across the entire computational domain. For a fixed $\rho ( x , y ) = \mathrm { e } ^ { - ( x ^ { 2 } + y ^ { 2 } ) / 4 }$ , Fig. 3 visualizes the basis functions $\{ \mathcal { G } _ { \theta _ { j } } ( \rho ) \} _ { j = 1 } ^ { 5 }$ generated by the basis operators. It is observed that, as the stage index s increases, the basis functions tend to capture higher-frequency components.

Also, we test the robustness of DCNO in the presence of noisy training data here to validate the theoretical result (3.10). Independent Gaussian noises with relative noise levels ranging from $1 0 ^ { - 6 } \ \mathrm { t o } \ 1 0 ^ { - 1 }$ are added to the exact solution data, which serve as training labels. With $L = 4$ and $W _ { 0 } = 7 0$ fixed and the total number of stages set to $S = 5$ , all other training settings remain the same as before. The left plot of Fig. 4 presents the test error of DCNO versus the noise level. The results indicate that the prediction error increases monotonically with the noise amplitude, while remaining consistently below the noise level across all tested regimes spanning several orders. This demonstrates that DCNO achieves robust operator approximation under noisy supervision. For illustration, a Gaussian noise with level 0.001 is shown in the middle plot of Fig. 4, together with the corresponding pointwise error of the predicted solution for the test sample $\mathsf { \bar { \rho } } ( x , y ) = \mathsf { e } ^ { - ( x ^ { 2 } + y ^ { 2 } ) }$ in the right plot. The outcomes in total support Theorem 3.2.

![](images/9c3188daa5dfe50858d2d9d65f0de76eca723e0db5be48187da4fc728c31aa8a.jpg)

![](images/4ead639ce111b6405a6521885539e442420e34cd011a3440047faaa665a82496.jpg)

![](images/1fb702427aaf9e9adec12b3bc912e7f91a03d9d102ce7ea2a8ab2a3339a7797a.jpg)

Figure 4: Relative $L ^ { 2 }$ error of DCNO under varying noise levels (left); A Gaussian noise sample with noise level 0.001 (middle); Pointwise error of DCNO for Example 5.1 on a test sample with α = 1 (right).  
![](images/c837e8627e4f2ee98314518ccc7156f1986b95a6ca402281813eaa37b143e22a.jpg)

![](images/8d78939526a55c29d1cbf2ed5528b484413813524b893f6591b5f398d0534eb5.jpg)  
Figure 5: Results of DGNO in Algorithm 2 for Example 5.1: (a) Change of Loss during the iterations; (b) Mean relative $L ^ { 2 }$ error with respect to the number of stages.

Table 2: Number of stages, epochs per stage, training time and error for DCNO and DGNO.
<table><tr><td></td><td>Stages</td><td>Epochs per stage</td><td>Training time(s)</td><td>Mean relative  $\overline { { L ^ { 2 } } }$  error</td></tr><tr><td>supervised DCNO</td><td>5</td><td>3000</td><td>251.23s</td><td>1.31E-6</td></tr><tr><td>unsupervised DGNO</td><td>5</td><td>3000</td><td>491.60s</td><td>6.17E-7</td></tr></table>

We move on to test DGNO on Example 5.1. The unsupervised DGNO needs only the input functions as the training set, which is adopted as the same $\{ \rho ^ { ( b ) } \} _ { b = 1 } ^ { N _ { b } }$ with $N _ { b } = 1 0 0$ for training in DCNO. We set $\gamma = 2$ and $m = 4$ in the combined moment-matching strategy, and zero boundary conditions are encoded into DeepONet via $\begin{array} { r } { \mathcal { G } _ { \theta _ { s } } ( f , x , y ) = ( I ^ { 2 } - x ^ { 2 } ) ( I ^ { 2 } - y ^ { 2 } ) \sum _ { k = 1 } ^ { p } \mathfrak { b } _ { k } ( f ) \mathfrak { t } _ { k } ( x , y ) } \end{array}$ . The trunk input is discretized on Gauss–Legendre nodes with $N _ { x } = N _ { y } = 1 1 0$ in each spatial direction. The total number of stages set to $S = 5 { \mathrm { ; } }$ and all other training settings are kept the same as for DCNO above. Fig. 5(a) shows the evolution of the loss function during training. In each stage, the loss decreases rapidly as the training iterations proceed. In the testing phase, the same test set as that used for DCNO is adopted. Table 1 presents the errors obtained by DGNO for various network depths and widths. The results indicate a clear reduction in error with increasing network depth and width. With $L = 4$ and $W _ { 0 } = 7 0$ fixed, Fig. 5(b) demonstrates the reduction in the test error as the stage increases, with the final error $6 . 1 7 \times 1 0 ^ { - 7 }$ also approaching the machine precision of single float. These results support Theorem 4.6.

Lastly, we compare the performance of DCNO and DGNO on Example 5.1. The results are summarized in Table 2, which presents the number of stages, epochs per stage, training time, and mean relative $L ^ { 2 }$ error for both approaches. It can be observed that, with the same number of stages and training epochs, DGNO achieves a lower generalization error, albeit at the cost of longer training time, which is due to the additional computational overhead of evaluating derivatives in its loss function. Overall, DGNO benefiting from its unsupervised feature is certainly more eficient. Nevertheless, there are convolution problems that do not enjoy a PDE formulation, where DGNO is not applicable and DCNO becomes the solution. The following example is presented to illustrate this.

Example 5.2 (Multiquadrics). Consider the multiquadric kernels $K ( x ) = 1 / { \sqrt { x ^ { 2 } + a ^ { 2 } } }$ with $0 < a \ll 1$ commonly used in electrostatic computation [50, 51, 52], for a 1D convolution:

$$
\phi ( x ) = \int _ { 0 } ^ { 1 } K ( x - \tilde { x } ) \rho ( \tilde { x } ) d \tilde { x } , \quad x \in \Omega = ( 0 , 1 ) .
$$

It does not have a PDE formulation $I 5 1 , \ 5 2 ] ,$ and our input is the Gaussian-type density function $\rho ( x ) =$ $\mathrm { e } ^ { - \beta \left( x - \frac { 1 } { 2 } \right) ^ { 2 } }$ with $\beta > 0$

In this example, the parameter $a > 0$ controls the regularization scale of the kernel, and as $a  0 ,$ , the kernel approaches a weakly singular form. For a fixed $\bar { \rho ( x ) } = \mathrm { e } ^ { - 2 \left( x - \frac { 1 } { 2 } \right) ^ { 2 } }$ , the corresponding solutions ϕ(x) for diferent a are shown in the left plot of Fig. 6. As a decreases, the kernel becomes more singular at $0 ,$ resulting in solutions with steeper variations near the boundary. We then set $a = 0 . 0 0 0 1$ to examine DCNO in this near-singular regime. The training data is provided by the traditional method SOE [50], generating a training set $\{ ( \rho ^ { ( b ) } , \bar { \phi } ^ { ( b ) } ) \} _ { b = 1 } ^ { N _ { b } }$ containing $N _ { b } = 5 0$ samples, where $\rho ^ { ( b ) }$ is generated by uniformly sampling $\beta \sim \mathcal { U } [ 1 , 5 ]$ . The trunk network is evaluated at collocation points defined on the uniform grid $x _ { k } = k / N _ { x } , k = 0 , . . . , N _ { x }$ , with $N _ { x } = 2 0 0$ for training and $N _ { x } = 3 0 0$ for testing. The total number of stages is set to $S = 4$ , with $\sigma =$ tanh and the depth of DeepONet fixed at $L = 4$ . The network width and $p$ are increased with the stage index s, i.e., $W = 7 0 + 1 0 ( s - 1 )$ and $p = 1 2 0 + 1 0 ( s - 1 )$ . For test, 100 samples that are disjoint from the training set are randomly selected to form the test set. The right plot of Fig. 6 shows the mean relative $L ^ { 2 }$ error on the test set, which decreases steadily as the number of basis operators increases with the stage index $s ,$ and eventually reaches $1 . 1 6 \times 1 0 ^ { - 6 }$ . For the test sample $\rho ( x ) = \mathrm { e } ^ { - 2 \left( x - \frac { 1 } { 2 } \right) ^ { 2 } }$ Fig. 7(a) presents the exact and numerical solutions, while the corresponding pointwise error is shown in Fig. 7(b). The results demonstrate that DCNO can achieve high accuracy even for this near-singular setting. The mean relative $L ^ { 2 }$ errors of the approximate solution $\phi$ over the range of a values are presented in Table 3 to evaluate the robustness of DCNO. The results show that DCNO maintains consistently high accuracy across all tested values of $a ,$ , indicating that DCNO is uniformly efective from the near-singular $( a = 1 0 ^ { - 4 } )$ regime to the regular $( a = 1 0 ^ { - 1 } )$ regime.

![](images/865a30f93d0b530eaf2820dbd5f883db7c4ef1bcf53ed2740fe70739e320e306.jpg)

![](images/768ef172b7f80ce45759cc793e1fadf90142a49c340955252e4211d9a95d5be6.jpg)  
Figure 6: Solutions for diferent values of the kernel parameter a (left); Mean relative $L ^ { 2 }$ error with respect to the number of stages for $a = 0 . 0 0 0 1$ (right).

By Example 5.2, we assess the computational eficiency of our operator learning by comparing the traditional SOE method with DCNO on a test set of 1000 samples. Both methods are evaluated on the same uniform grid with 64 spatial points for $a = 0 . 0 0 0 1$ . The results in Table 4 indicate that while DCNO incurs additional training overhead, it drastically reduces the evaluation time compared to the traditional approach, achieving several orders of magnitude speedup during inference and demonstrating clear advantages for large-scale evaluations. Fig. 7(c) further reports the total computational time of SOE and DCNO as the number of test cases increases. DCNO demonstrates a much slower growth rate, highlighting its superior computational eficiency at scale. So will DGNO if applicable.

![](images/4ca6d39876968bb72ce1ad57b2c5540d7bf371cad61349c5d6a2d0fb450faa7a.jpg)

![](images/a2d46746c5db940e2a0050a0cee4d98dd28d9d545d705c5a70a6abdde834d7ae.jpg)

![](images/693a67b9916c2f1720e5a653cf59440abfbdf8571bba3e7bfbe89d0e9c231020.jpg)  
Figure 7: Results of DCNO for Example 5.2: (a) Exact and numerical solutions; (b) Error of the numerical solution; (c) Computational time of the traditional solver and DCNO against the number of computational cases.

Table 3: Mean relative $L ^ { 2 }$ error of numerical solutions from DCNO for varying a.
<table><tr><td>a</td><td>0.0001</td><td>0.001</td><td>0.001</td><td>0.1</td></tr><tr><td>Error</td><td> $\overline { { 1 . 2 0 e - 6 } }$ </td><td> $3 . 0 8 e { - 6 }$ </td><td> $2 . 7 1 e { - 6 }$ </td><td> $3 . 4 8 e { - 6 }$ </td></tr></table>

Table 4: Comparison between DCNO and the traditional method SOE on test set.
<table><tr><td>Method</td><td>Evaluation time (s)</td><td>Training time (s)</td></tr><tr><td>Traditional method DCNO method</td><td>670.44 0.021</td><td>202.14</td></tr></table>

Example 5.3 (Dynamics of Schrödinger-Poisson system). For quantum mechanics taking into account selfgravitating interaction, one considers the nonlinear Schrödinger-Newton/Poisson equations that reads in dimensionless form $/ 3 9 , \ 4 9 ] .$

$$
i \partial _ { t } \psi ( \mathbf { x } , t ) = - \frac { 1 } { 2 } \Delta \psi + V ( \mathbf { x } ) \psi + \phi ( \mathbf { x } , t ) \psi + | \psi | ^ { 2 } \psi , \quad \mathbf { x } \in \Omega \subset \mathbb { R } ^ { 2 } , \ t > 0 ,\tag{5.3a}
$$

$$
- \Delta \phi ( { \bf x } , t ) = | \psi | ^ { 2 } , ~ { \bf x } \in \Omega , ~ t > 0 ,\tag{5.3b}
$$

$$
\psi ( \mathbf { x } , 0 ) = \psi _ { 0 } ( \mathbf { x } ) , \quad \mathbf { x } \in \overline { { \Omega } } ; \qquad \psi ( \mathbf { x } , t ) = 0 , \ \phi ( \mathbf { x } , t ) = 0 , \quad \mathbf { x } \in \partial \Omega , \ t \geq 0 .\tag{5.3c}
$$

Here, ψ is a complex-valued wave function, ϕ is the real-valued gravitational potential, $\begin{array} { r } { V ( \mathbf { x } ) = \frac { 1 } { 2 } ( x ^ { 2 } + y ^ { 2 } ) } \end{array}$ denotes an external harmonic oscillator trapping potential. We consider the circular computational domain $\Omega = \{ \mathbf { x } \in \mathbb { R } ^ { 2 } : | \mathbf { x } | < I \}$ with radius $I = 8$ and solve its dynamics for $t \in [ 0 , 1 ]$ . The initial condition is given by $\textstyle \psi _ { 0 } ( x , y ) = { \frac { 1 } { \sqrt { 2 \pi } } } \mathrm { e } ^ { - ( x ^ { \frac { 5 } { 2 } } + y ^ { 2 } ) / 4 }$

A popular numerical solver for the dynamics of (5.3) on a circular domain is the time-splitting finite element method, briefly described below as a reference. The spatial domain is discretized by a conforming triangular mesh with $N _ { h }$ interior nodes, where continuous piecewise linear $( P _ { 1 } )$ finite elements are employed. Let $\textcircled { \vartheta _ { j } } \rbrace _ { j = 1 } ^ { N _ { h } }$ denote the corresponding nodal basis functions. The finite element approximation of $\psi ( \cdot , t _ { n } )$ is given by $\begin{array} { r } { \psi _ { h } ( \cdot , t _ { n } ) = \sum _ { j = 1 } ^ { N _ { h } } \Psi _ { j } ^ { n } \vartheta _ { j } } \end{array}$ , with $\Psi ^ { n } = ( \Psi _ { 1 } ^ { n } , \dots , \Psi _ { { N _ { h } } } ^ { n } ) ^ { T } \in \mathbb { C } ^ { N _ { h } }$ . Given the time discretization $t _ { n } = n \Delta t , \ n \geq 0$ , the Strang splitting Crank–Nicolson finite element scheme [6] reads

![](images/870b350097701b8bff9431ab221f2ba602edb2fb456b00f82109f275f616bd7e.jpg)

![](images/87da64a3835519f9cc50e2a8740ec099d959c9484d590c99ddfe4237d56adbbe.jpg)

![](images/e16789c35e7afba8267d450ecb61b753412836accc4162e8b0269564b86e0356.jpg)

Figure 8: Results of Example 5.3: (a) Relative $L ^ { 2 }$ error of ψ over time; (b) Total evaluation time of the traditional solver and DGNO for solving the Poisson equation (5.3b) during the simulation; (c) Evolution of the mass over time.  
![](images/36ca99f158d3e3c946f910c61f6e28ee9de85215a5f7cddde87f0a4d8a8acc37.jpg)

![](images/935d472bd2d38b1a81042649747b9bc889e6c67b08015421f97560d2189bec08.jpg)

![](images/5175ac737b7a5f923c24b1bc85062f79f67f47dea456551e2af77c26e9748de5.jpg)  
Figure 9: $| \psi ( \cdot , t = 0 ) | ^ { 2 }$ (left); $| \psi ( \cdot , t = 1 ) | ^ { 2 }$ by DGNO (middle); Pointwise absolute error at t = 1 (right) for Example 5.3.

$$
\left( M _ { h } + \frac { i \Delta t } { 8 } K _ { h } \right) \Psi ^ { * } = \left( M _ { h } - \frac { i \Delta t } { 8 } K _ { h } \right) \Psi ^ { n } ,\tag{5.4a}
$$

$$
\Psi ^ { * * } = \exp \left[ - i \Delta t \left( V + \Phi ^ { * } + | \Psi ^ { * } | ^ { 2 } \right) \right] \Psi ^ { * } ,\tag{5.4b}
$$

$$
\left( M _ { h } + \frac { i \Delta t } { 8 } K _ { h } \right) \Psi ^ { n + 1 } = \left( M _ { h } - \frac { i \Delta t } { 8 } K _ { h } \right) \Psi ^ { * * } .\tag{5.4c}
$$

Here, $M _ { h }$ and $K _ { h }$ denote the finite element mass and stifness matrices, respectively, with entries $( M _ { h } ) _ { j k } =$ $\langle \vartheta _ { j } , \vartheta _ { k } \rangle _ { L ^ { 2 } ( \Omega ) } , ( K _ { h } ) _ { j k } = \langle \nabla \vartheta _ { j } , \nabla \vartheta _ { k } \rangle _ { L ^ { 2 } ( \Omega ) }$ . The numerical potential $\Phi ^ { * } \in \mathbb { R } ^ { N _ { h } }$ is obtained from the finite element discretization of the Poisson equation (5.3b), leading to the linear system $K _ { h } \Phi ^ { * } \ = \ F _ { h } ( | \Psi ^ { * } | ^ { 2 } )$ where $F _ { h } ( | \Psi ^ { * } | ^ { 2 } )$ denotes the assembled load vector. Since the stifness matrix $K _ { h }$ is time-independent, its sparse LU factorization is performed only once before time stepping, while each Poisson solve requires only triangular substitutions. Nevertheless, solving the Poisson equation repeatedly at every time step remains computationally demanding, especially for small $\Delta t$ or long-time simulations.

Here, we employ our DGNO method to replace the evaluation process for $\Phi ^ { * }$ in (5.4). The training of DGNO is performed similarly to that in Example 5.1. After training, only a single neural operator inference is required at each time step to eficiently obtain an approximation of the gravitational potential. We set $\Delta t = 0 . 0 0 1 , N _ { h } = 8 0 0 0 0$ for the experiment. Fig. 8(a) illustrates the relative $L ^ { 2 }$ error of the solution field $\psi$ over time. The linear drift in the solution error with respect to time comes from the natural accumulation of the approximation error. Fig. 8(b) compares the accumulated evaluation time of the Poisson solver based on the conventional finite element method and DGNO throughout the simulation. The evolution of the mass $\int _ { \Omega } | \psi ( \mathbf { x } , t ) | ^ { 2 }$ dx over time is presented in Fig. 8(c). Fig. 9 further presents $| \psi | ^ { 2 }$ at $t = 0$ and $t = 1$ , together with the corresponding pointwise absolute error at $t = 1$ . As can be seen, the proposed method remains stable and accurate throughout the evolution, provides an accelerated solver for (5.3), and exhibits excellent

mass conservation.

## 5.3. Extension to multiple inputs

We generalize our methods to further study the convolution problem (1.1) in which both the density function and the convolution kernel vary, forming a multi-input operator learning task. In fact, this can be quite straightforward under our approach by adding additional branches to (2.6), replacing DeepONet with the Multiple-Input Operator Network (MIONet) [24].

Example 5.4 (DCNO with MIONet). Consider the Gaussian kernel family $\begin{array} { r } { K _ { \alpha } ( \mathbf { x } ) = \frac { 1 } { 2 \pi \alpha ^ { 2 } } \mathrm { e } ^ { - \frac { | \mathbf { x } | ^ { 2 } } { 2 \alpha ^ { 2 } } } , \mathbf { x } \in \Omega = } \end{array}$ $( - 8 , 8 ) ^ { 2 } , \ \alpha > 0$ . Define the convolution operator

$$
\mathcal { G } : ( \rho , K _ { \alpha } ) \mapsto \boldsymbol \phi , \qquad \boldsymbol \phi ( \mathbf { x } ) = \big ( K _ { \alpha } \ast \rho \big ) ( \mathbf { x } ) = \int _ { \Omega } K _ { \alpha } ( \mathbf { x } - \tilde { \mathbf { x } } ) \rho ( \tilde { \mathbf { x } } ) d \tilde { \mathbf { x } } .
$$

Such a convolution is repeatedly needed in constructing the scale-space of the scale-invariant feature transform algorithm [33]. Here we take ρ from a family of Gaussian densities $\begin{array} { r } { \rho \big ( \mathbf { { x } } \big ) = \frac { 1 } { 2 \pi \tau ^ { 2 } } \mathrm { e } ^ { - | \mathbf { x } | ^ { 2 } / ( 2 \tau ^ { 2 } ) } } \end{array}$ with parameter $\tau > 0$ , and then the convolution admits the analytical solution $\begin{array} { r } { \phi ( \mathbf { x } ) = \frac { 1 } { 2 \pi ( \alpha ^ { 2 } + \tau ^ { 2 } ) } \mathrm { e } ^ { - | \mathbf { x } | ^ { 2 } / ( 2 \alpha ^ { 2 } + 2 \tau ^ { 2 } ) } } \end{array}$ as reference.

To solve Example 5.4, we apply the DCNO algorithm with the basis operator network constructed using the MIONet architecture [24] to accommodate multiple input functions. For training, a dataset $\{ ( \rho ^ { ( b ) } , K _ { \alpha } ^ { ( l ) } , \phi ^ { ( b , l ) } ) \} _ { b = 1 , l = 1 } ^ { N _ { b } , N _ { l } }$ with $N _ { b } = N _ { l } = 2 0$ samples is generated, with each parameter $\alpha ^ { ( b ) }$ and $\tau ^ { ( l ) }$ is independently drawn from U[1, 2]. All other training settings are the same as those in Example 5.1, except that $N _ { x } = N _ { y } = 6 4$ are used during training, while a finer grid with $N _ { x } = N _ { y } = 1 0 0$ is adopted for error evaluation. At each stage s, the practical loss function for DCNO in the multiple-input case is extended to

$$
\mathrm { L o s s } _ { s } ( \theta ) = \frac { 1 } { N _ { b } \times N _ { l } \times N _ { \mathbf { x } } } \sum _ { b = 1 } ^ { N _ { b } } \sum _ { l = 1 } ^ { N _ { l } } \sum _ { k = 1 } ^ { N _ { \mathbf { x } } } \left| \frac { 1 } { \beta _ { s - 1 } } e _ { s - 1 } ( \rho ^ { ( b ) } , K _ { \alpha } ^ { ( l ) } ) ( \mathbf { x } _ { k } ) - \mathcal { G } _ { \theta } ( \rho ^ { ( b ) } , K _ { \alpha } ^ { ( l ) } , \mathbf { x } _ { k } ) \right| ^ { 2 } .\tag{5.5}
$$

In the testing phase, 200 samples disjoint from the training set are randomly selected. Fig. 10(a) illustrates a clear decrease in the test error as the number of basis operators increases, and ultimately reaches an accuracy of $1 . 4 0 \times 1 0 ^ { - 5 }$ . Notably, without the progressive construction of basis operators employed in DCNO, the accuracy at stage 1 corresponds to that of the standard MIONet, which is only $4 . 8 4 \times 1 0 ^ { - 2 }$ This demonstrates the significant advantage of DCNO in improving the accuracy of multi-input operator learning. Fig. 11 presents the numerical solutions on the test set for diferent combinations of density functions and convolution kernels, together with the corresponding pointwise error distributions. These results demonstrate that DCNO maintains strong performance in the multiple-input setting for generalizing simultaneously over the density and the convolution kernel.

![](images/32672df1e59f03a7402212e7519a0f27fa83d9be28f17776495981cb80798f55.jpg)

![](images/5f621d774fbcf63012a3cbcd61f43a6fa4f3a60cb98539f14d2cb1ca250e9cf1.jpg)  
Figure 10: Mean relative $L ^ { 2 }$ error with respect to the number of stages: (a) Example 5.4; (b) Example 5.5 .

![](images/674d716bcd627d3fce451d8a9cd552925781390ed783424b83ea352c9cac5d62.jpg)

![](images/f1263593570e13f84ec66e2721c313354a5795cba0edbae463320f497d758eb4.jpg)

![](images/7fb55eecdaf13fe5667cd1c0cef807d4c919884f27ec97bdecb00df92f3fd0b4.jpg)

![](images/36bc34d7f268f90450758f8284b15a6b79765d158e134630cd759aa227e1457e.jpg)

![](images/4a0bad16b87d0c36b4bba639749d8d12a896825ad392d34ffb463720ba3c228d.jpg)

![](images/436cf77ed248c763fe172e7ce9a6c888d5a7ae3725677b2c9566281eb04f5701.jpg)  
 -  
Figure 11: Numerical solution (1st row) and pointwise error (2nd row) of DCNO for Example 5.4 on three test samples: (a) $\alpha = \tau = 2 ;$ (b) α = τ = 1.5; (c) α = τ = 1.

Example 5.5 (DGNO with MIONet). We turn to the unsupervised scenario by considering the following general elliptic boundary value problem

$$
\left\{ \begin{array} { l l } { - \nabla \cdot \left( \kappa ( \mathbf { x } ) \nabla \phi ( \mathbf { x } ) \right) = \rho ( \mathbf { x } ) , } & { \mathbf { x } \in \Omega , } \\ { \phi ( \mathbf { x } ) = 0 , } & { \mathbf { x } \in \partial \Omega , } \end{array} \right.\tag{5.6}
$$

where the computational domain is set as $\Omega = ( 0 , 1 ) ^ { 2 }$

In (5.6), diferent choices of $\kappa ( { \bf x } )$ correspond to diferent integral kernels in the convolution (1.1). Our goal is to learn the mapping $\mathcal { G } : ( \rho , \kappa ) \mapsto \phi$ with $\rho$ and κ jointly serving as inputs, which constitutes a multiple-input operator learning problem. The diferential formulation in which the problem is posed enables an unsupervised training strategy, and we apply the DGNO algorithm with two branch networks to input $\rho$ and $\kappa .$ To obtain reference solutions for tests, we set $\kappa ( \mathbf { x } ) = 1 + \alpha \sin ( 2 \pi x ) \sin ( 2 \pi y )$ with $| \alpha | < 1$ and

$$
\begin{array} { r l r }   { \rho ( \mathbf { x } ) = 2 \pi ^ { 2 } \alpha \kappa ( \mathbf { x } ) \Big [ \gamma \sin ( \pi x ) \sin ( \pi y ) + 4 \sin ( 2 \pi x ) \sin ( 2 \pi y ) \Big ] - 2 \pi ^ { 2 } \alpha ^ { 2 } \Big [ \gamma \big ( \cos ( 2 \pi x ) \sin ( 2 \pi y ) \cos ( \pi x ) \sin ( \pi y ) } \\ & { } & { \qquad + \sin ( 2 \pi x ) \cos ( 2 \pi y ) \sin ( \pi x ) \cos ( \pi y ) \big ) + 2 \big ( \cos ^ { 2 } ( 2 \pi x ) \sin ^ { 2 } ( 2 \pi y ) + \sin ^ { 2 } ( 2 \pi x ) \cos ^ { 2 } ( 2 \pi y ) \big ) \Big ] , } \end{array}
$$

where $\phi ( \mathbf { x } ) = \alpha \gamma \sin ( \pi x ) \sin ( \pi y ) + \alpha \sin ( 2 \pi x ) \sin ( 2 \pi y )$ is the exact solution to (5.6). For the training data, a set $\{ ( \kappa ^ { ( b ) } , \rho ^ { ( l ) } ) \} _ { b = 1 , l = } ^ { N _ { b } , N _ { l } }$ with $N _ { b } = N _ { l } = 2 0$ is generated, resulting in a total of 400 samples. The parameters $\alpha ^ { ( b ) }$ and $\tau ^ { ( l ) }$ are independently sampled from $\mathcal { U } ( 0 . 5 , 1 )$ and $\mathcal { U } [ 2 , 4 ]$ , respectively. All other training settings are the same as those in Example 5.1, except that $N _ { x } = N _ { y } = 1 1 0$ are used during training, while a finer grid with $N _ { x } = N _ { y } = 1 5 0$ is adopted for error evaluation. In view of (4.3), the extended practical loss for DGNO in the multiple-input case reads

$$
\mathrm { L o s s } _ { s } ( \theta ) = \frac { - 1 } { N _ { b } \times N _ { l } } \sum _ { b = 1 } ^ { N _ { b } } \sum _ { l = 1 } ^ { N _ { l } } \frac { R _ { s - 1 } ( \kappa ^ { ( b ) } , \rho ^ { ( l ) } , \mathcal { G } _ { \theta _ { s } } ( \kappa ^ { ( b ) } , \rho ^ { ( l ) } ) ) } { \| \mathcal { G } _ { \theta _ { s } } ( \kappa ^ { ( b ) } , \rho ^ { ( l ) } ) \| _ { X } } .\tag{5.7}
$$

For testing, 200 samples that are disjoint from the training set are randomly selected to form the test set. As shown in Fig. 10(b), the mean relative $L ^ { 2 }$ error on the test set decreases progressively with the stage

![](images/0cb3872cafed18630b7aba062d342e2dc7d13ba6728b30b140007e5f71c9b969.jpg)  
 -

![](images/eb5940a3528a068ad8d7f6ddf20ecc372002c378bef81748187328c7bf568e65.jpg)  
 -

![](images/99bb68a38cd527e8c455c65aba693af81249bbf7bbd5d178b222b049a0195bc9.jpg)  
 -

![](images/6e977cdd6bf639850ab0d50cd03f0bb1873e96a4fdb7f881db3c548e6b8b2d5e.jpg)  
 -

![](images/eba358c1a0bfb6ef25d48bad0a369a7fe5374d225ca1d44f01b388f838ba174c.jpg)  
 -

![](images/b3429ae7d01d201abd8b2e6df4b91caa2f05caa68a4f234ff83a6194a3aa7ed3.jpg)  
 -  
Figure 12: Numerical solution (1st row) and pointwise error (2nd row) of DGNO for Example 5.5 on three test samples: (a) α = 0.5, γ = 2; (b) α = 0.6, γ = 3; (c) α = 0.7, γ = 4.

index s, and ultimately reaches an accuracy of $7 . 0 8 \times 1 0 ^ { - 5 }$ . We also select some representative samples from the test set to visually assess the approximation capability of the trained basis operators in Fig. 12. These results demonstrate that DGNO as the unsupervised method also performs well for the multiple-input case.

## 6. Conclusion

In this work, we have introduced two novel and general multi-stage neural operator learning frameworks, the Deep Collocation Neural Operator (DCNO) and the Deep Galerkin Neural Operator (DGNO), designed to significantly enhance the accuracy of operator approximation, particularly applied for convolutions. DCNO ofers a supervised approach that progressively builds a richer approximation space by training basis operators on the residuals of the target operator, leveraging available input-output data. Complementing this, DGNO provides an unsupervised framework that learns operator bases by minimizing the residual of a governing PDE in its weak form, thus eliminating the need for labeled data.

We have demonstrated the eficacy of these methods through extensive numerical experiments. Both DCNO and DGNO achieved high accuracy in learning convolution operators, often approaching machine precision of single-precision floating-point arithmetic. Notably, for convolution problems lacking a straightforward PDE formulation, DCNO proved to be a robust supervised solution. When a PDE formulation is available, DGNO ofers an eficient unsupervised alternative. Furthermore, we have extended these frameworks to handle multi-input operator learning scenarios, encompassing variations in both the density and the kernel for convolutions.

## Acknowledgements

Z. Wen and X. Zhao are supported by National Key Research and Development Program of China, National MCF Energy R&D Program (No. 2024YFE03240400), National Natural Science Foundation of China (Nos. 42450275, 12271413). Z. Mao is supported by the Zhejiang Provincial Natural Science Foundation of China under Grant No. LR26A010001 and the National Natural Science Foundation of China under Grant No. 12531015. Y. Zhang is supported by the National Natural Science Foundation of China (No. 12271400), the National Key R&D Program of China (No. 2024YFA1012803) and basic research fund of Tianjin University under Grant 2025XJ21-0010.

## References

[1] R. A. Adams and J. J. Fournier, Sobolev Spaces, vol. 140, Elsevier, 2003.

[2] M. Ainsworth and J. Dong, Galerkin neural networks: A framework for approximating variational equations with error control, SIAM J. Sci. Comput., 43 (2021), pp. A2474–A2501.

[3] M. Ainsworth and J. Dong, Extended Galerkin neural network approximation of singular variational problems with error control, SIAM J. Sci. Comput., 47 (2025), pp. C738–C768.

[4] Z. Aldirany, R. Cottereau, M. Laforest, and S. Prudhomme, Multi-level neural networks for accurate solutions of boundary-value problems, Comput. Methods Appl. Mech. Engrg., 419 (2024), p. 116666.

[5] A. Anandkumar, K. Azizzadenesheli, K. Bhattacharya, N. Kovachki, Z. Li, B. Liu, and A. Stuart, Neural operator: Graph kernel network for partial diferential equations, ICLR, (2020).

[6] W. Auzinger, T. Kassebacher, O. Koch, and M. Thalhammer, Convergence ofa Strang splitting finite element discretization for the Schrödinger–Poisson equation, ESAIM Math. Model. Numer. Anal., 51 (2017), pp. 1245–1278.

[7] W. Bao, S. Jiang, Q. Tang, and Y. Zhang, Computing the ground state and dynamics of the nonlinear Schrödinger equation with nonlocal interactions via the nonuniform FFT, J. Comput. Phys., 296 (2015), pp. 72–89.

[8] H. Bassi, Y. Zhu, S. Liang, J. Yin, C. C. Reeves, V. Vlček, and C. Yang, Learning nonlinear integral operators via recurrent neural networks and its application in solving integro-diferential equations, Mach. Learn. Appl., 15 (2024), p. 100524.

[9] S. L. Brunton and J. N. Kutz, Promising directions of machine learning for partial diferential equations, Nat. Comput. Sci., 4 (2024), pp. 483–494.

[10] S. Cai, Z. Wang, L. Lu, T. A. Zaki, and G. E. Karniadakis, DeepM&Mnet: Inferring the electroconvection multiphysics fields based on operator approximation by neural networks, J. Comput. Phys., 436 (2021), p. 110296.

[11] Z. Chang, B. Gao, R. Yin, and X. Zhao, DLTM: A deep learning method for tearing mode simulation and prediction, Nucl. Fusion, 65 (2025), p. 086009.

[12] Z. Chang, Z. Wen, and X. Zhao, Unsupervised operator learning approach for dissipative equations via Onsager principle, SIAM J. Sci. Comput., 48 (2026), pp. C1060–C1085.

[13] T. Chen and H. Chen, Universal approximation to nonlinear operators by neural networks with arbitrary activation functions and its application to dynamical systems, IEEE Trans. Neural Netw., 6 (1995), pp. 911–917.

[14] P. C. Di Leoni, L. Lu, C. Meneveau, G. E. Karniadakis, and T. A. Zaki, Neural operator prediction of linear instability waves in high-speed boundary layers, J. Comput. Phys., 474 (2023), p. 111793.

[15] L. Exl, N. J. Mauser, and Y. Zhang, Accurate and eficient computation of nonlocal potentials based on Gaussian-sum approximation, J. Comput. Phys., 327 (2016), pp. 629–642.

[16] X. Glorot and Y. Bengio, Understanding the dificulty of training deep feedforward neural networks, in AISTATS, JMLR Workshop and Conference Proceedings, 2010, pp. 249–256.

[17] R. C. Gonzalez, Digital Image Processing, Pearson Education India, 2009.

[18] J. Y. Greengard, L.and Lee, Accelerating the nonuniform fast Fourier transform, SIAM Rev., 46 (2004), p. 443–454.

[19] L. Greengard, The Rapid Evaluation of Potential Fields in Particle Systems, MIT Press, 1988.

[20] W. Hackbusch, A sparse matrix arithmetic based on H-matrices. Part I: Introduction to H-matrices, Computing, 62 (1999), pp. 89–108.

[21] W. Hackbusch et al., Hierarchical Matrices: Algorithms and Analysis, vol. 49, Springer, 2015.

[22] K. Hornik, Approximation capabilities of multilayer feedforward networks, Neural Netw., 4 (1991), pp. 251–257.

[23] Y. Ji, J. Liang, and Z. Xu, Machine-learning interatomic potentials for long-range systems, Phys. Rev. Lett., 135 (2025), p. 178001.

[24] P. Jin, S. Meng, and L. Lu, MIONet: Learning multiple-input operators via tensor product, SIAM J. Sci. Comput., 44 (2022), pp. A3490–A3514.

[25] D. P. Kingma and J. Ba, Adam: A method for stochastic optimization, ICLR, (2015), p. 13.

[26] H. Li and M. Shatarah, Operator learning for urban water clarification hydrodynamics and particulate matter transport with physics-informed neural networks, Water Res., 251 (2024), p. 121123.

[27] Z. Li, D. Z. Huang, B. Liu, and A. Anandkumar, Fourier neural operator with learned deformations for PDEs on general geometries, J. Mach. Learn. Res., 24 (2023), pp. 1–26.

[28] Z. Li, N. Kovachki, K. Azizzadenesheli, B. Liu, K. Bhattacharya, A. Stuart, and A. Anandkumar, Fourier neural operator for parametric partial diferential equations, ICLR, (2021).

[29] C. Lin, Z. Li, L. Lu, S. Cai, M. Maxey, and G. E. Karniadakis, Operator learning for predicting multiscale bubble growth dynamics, J. Chem. Phys., 154 (2021).

[30] X. Liu, Q. Tang, S. Zhang, and Y. Zhang, On optimal zero-padding of kernel truncation method, SIAM J. Sci. Comput., 46 (2024), pp. A23–A49.

[31] X. Liu, Q. Tang, and Y. Zhang, Fast convolution solvers using moment-matching, arXiv preprint arXiv:2602.12850, (2026).

[32] Z. Liu, H. Wang, H. Zhang, K. Bao, X. Qian, and S. Song, Render unto numerics: Orthogonal polynomial neural operator for PDEs with nonperiodic boundary conditions, SIAM J. Sci. Comput., 46 (2024), pp. C323–C348.

[33] D. G. Lowe, Distinctive image features from scale-invariant keypoints, Int. J. Comput. Vis., 60 (2004), pp. 91–110.

[34] L. Lu, P. Jin, G. Pang, Z. Zhang, and G. E. Karniadakis, Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators, Nat. Mach. Intell., 3 (2021), pp. 218–229.

[35] L. Lu, X. Meng, S. Cai, Z. Mao, S. Goswami, Z. Zhang, and G. E. Karniadakis, A comprehensive and fair comparison of two neural operators (with practical extensions) based on fair data, Comput. Methods Appl. Mech. Engrg., 393 (2022), p. 114778.

[36] Z. Mao, L. Lu, O. Marxen, T. A. Zaki, and G. E. Karniadakis, DeepM&Mnet for hypersonics: Predicting the coupled flow and finite-rate chemistry behind a normal shock using neural-network approximation of operators, J. Comput. Phys., 447 (2021), p. 110698.

[37] N. H. Nelsen and A. M. Stuart, The random feature model for input-output maps between Banach spaces, SIAM J. Sci. Comput., 43 (2021), pp. A3212–A3243.

[38] A. V. Oppenheim, Discrete-Time Signal Processing, Pearson Education India, 1999.

[39] R. Penrose, On the gravitization of quantum mechanics 1: Quantum state reduction, Found. Phys., 44 (2014), pp. 557–575.

[40] M. Raissi, P. Perdikaris, and G. E. Karniadakis, Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial diferential equations, J. Comput. Phys., 378 (2019), pp. 686–707.

[41] F. Rosenblat, The perceptron: A probabilistic model for information storage and organization in the brain, Psychol. Rev., 65 (1958), pp. 386–408.

[42] F. Vico, L. Greengard, and M. Ferrando, Fast convolution with free-space Green’s functions, J. Comput. Phys., 323 (2016), pp. 191–203.

[43] H. Wang, T. Fu, Y. Du, W. Gao, K. Huang, Z. Liu, P. Chandak, S. Liu, P. Van Katwyk, A. Deac, et al., Scientific discovery in the age of artificial intelligence, Nature, 620 (2023), pp. 47–60.

[44] S. Wang, H. Wang, and P. Perdikaris, Learning the solution operator of parametric partial diferential equations with physics-informed DeepONets, Sci. Adv., 7 (2021), p. eabi8605.

[45] Y. Wang and C. Lai, Multi-stage neural networks: Function approximator of machine precision, J. Comput. Phys., 504 (2024), p. 112865.

[46] M. Weng, Z. Mao, and J. Shen, Deep collocation method: A framework for solving PDEs using neural networks with error control, SIAM J. Sci. Comput., 48 (2026), pp. C77–C102.

[47] Y. Xu, Multi-grade deep learning, Communications on Appl. Math. Comput., 8 (2026), pp. 778–829.

[48] L. Ying, G. Biros, and D. Zorin, A kernel-independent adaptive fast multipole algorithm in two and three dimensions, J. Comput. Phys., 196 (2004), pp. 591–626.

[49] Y. Zhang and X. Dong, On the computation of ground state and dynamics of Schrödinger–Poisson– Slater system, J. Comput. Phys., 230 (2011), pp. 2660–2676.

[50] Y. Zhang, C. Zhuang, and S. Jiang, Fast one-dimensional convolution with general kernels using sum-of-exponential approximation, Commun. Comput. Phys., 29 (2021), pp. 1570–1582.

[51] C. Zhuang and R. Zeng, A local discontinuous Galerkin method for 1.5-dimensional streamer discharge simulations, Appl. Math. Comput., 219 (2013), pp. 9925–9934.

[52] C. Zhuang, Y. Zhang, X. Zhou, R. Zeng, J. He, and L. Liu, A fast tree algorithm for electric field calculation in electrical discharge simulations, IEEE Trans. Magn., 54 (2018), pp. 1–4.