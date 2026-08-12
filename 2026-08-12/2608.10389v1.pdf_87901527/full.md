# Eficient Weak-Entropy PINN for Solving Hyperbolic Conservation Laws

Qi Gao<sup>1</sup>

Kuang Huang<sup>2</sup>

Xuan Di<sup>1,3∗</sup>

<sup>1</sup>Department of Civil Engineering and Engineering Mechanics, Columbia University <sup>2</sup>Department of Mathematics, The Chinese University of Hong Kong <sup>3</sup>Center for Smart Cities, Data Science Institute, Columbia University

## Abstract

In recent years, neural networks have significantly advanced numerical solutions of partial diferential equations (PDEs). However, solving PDEs with discontinuous solutions, such as hyperbolic conservation laws, remains challenging for neural network-based methods such as physics-informed neural networks (PINNs). Existing methods often rely on strong prior assumptions such as knowledge of discontinuity locations, or they introduce artificial smoothing terms that degrade accuracy. However, accurately solving these conservation laws and predicting the formation and propagation of discontinuities in solutions is crucial in many practical applications, including gas dynamics and trafic flow modeling. In this paper, we introduce a novel Weak-Entropy PINN (WEPINN) framework for hyperbolic conservation laws with discontinuous solutions. The method enforces the governing equations in their weak (integral) formulation and incorporates the entropy condition to select the physically admissible solution, while employing the discrete fast Fourier transform (DFFT) for eficient numerical integration. Our method is tested through extensive numerical experiments on a variety of scalar conservation laws and systems of conservation laws in one and two dimensional spaces. These experiments demonstrate that our method can accurately resolve sharp discontinuities while efectively capturing interactions between multiple shock and rarefaction waves.

Keywords: Hyperbolic conservation laws, shock waves, physics-informed neural networks, weak formulation, entropy condition.

## 1 Introduction

Recent years have seen an exponential growth in the use of neural networks for solving partial diferential equations (PDEs), e.g., physics-informed neural networks (PINNs) (Raissi et al., 2019). Despite their success in a variety of applications, PINNs require that solutions to the PDEs possess some level of smoothness and face significant challenges with discontinuous solutions, which often appear in hyperbolic conservation laws:

$$
\partial _ { t } { \bf { U } } ( t , { \bf { x } } ) + \nabla _ { \bf { x } } \cdot { \bf { F } } ( { \bf { U } } ( t , { \bf { x } } ) ) = 0 ,\tag{1}
$$

where U : $( 0 , \infty ) \times \mathbb { R } ^ { d } \to \mathbb { R } ^ { m }$ denotes the vector of conserved quantities, $t > 0$ is time, $\textbf { x } =$ $( x _ { 1 } , \ldots , x _ { d } ) ^ { \top } \in \mathbb { R } ^ { d }$ is the spatial coordinate in d dimensions, and $\mathbf { F } : \mathbb { R } ^ { m }  \mathbb { R } ^ { m \times d }$ is the flux function with columns $F _ { 1 } ( \mathbf { U } ) , \ldots , F _ { d } ( \mathbf { U } )$ , where each column $F _ { j } : \mathbb { R } ^ { m }  \mathbb { R } ^ { m }$ corresponds to the j-th flux component. The divergence operator acts as $\begin{array} { r } { \nabla _ { \mathbf { x } } \cdot \mathbf { F } ( \mathbf { U } ) = \sum _ { i = 1 } ^ { d } \partial _ { x _ { j } } F _ { j } ( \mathbf { U } ) } \end{array}$

These conservation laws are widely used in various scientific and engineering disciplines, such as (inviscid) Burgers’ equation and Euler equations for gas dynamics (Dafermos, 1983), or Lighthill-Whitham-Richards (LWR) model for trafic flow (Lighthill and Whitham, 1955; Richards, 1956). Solutions to these equations often develop discontinuities even from smooth initial conditions, giving rise to critical phenomena such as shock waves in gas propagation or trafic congestion. Trafic congestion can be understood as a discontinuity in vehicle density when vehicles upstream brake abruptly, while entering a jammed region downstream.

The standard PINN framework (Raissi et al., 2019) embeds governing equations in their strong form directly into the loss function by minimizing pointwise PDE residuals, e.g.,

$$
\mathcal { L } _ { \mathrm { P D E } } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \left. \partial _ { t } \hat { \mathbf { U } } ( t _ { i } , \mathbf { x } _ { i } ) + \nabla _ { \mathbf { x } } \cdot \mathbf { F } ( \hat { \mathbf { U } } ( t _ { i } , \mathbf { x } _ { i } ) ) \right. ^ { 2 } ,\tag{2}
$$

computed via automatic diferentiation, where $\hat { \textbf { U } }$ is the neural network approximation and $\{ ( t _ { i } , \mathbf { x } _ { i } ) \} _ { i = 1 } ^ { N }$ is a set of collocation points. Because the residual is evaluated pointwise through diferentiation, we refer to this method as Dif-PINN. This approach inherently relies on solution smoothness and therefore struggles with discontinuous solutions that arise in hyperbolic conservation laws, typically producing overly smooth profiles or spurious oscillations near discontinuities (Huang and Agarwal, 2023).

When solving hyperbolic conservation laws with discontinuous solutions, one workaround to enable the use of pointwise residuals in the Dif-PINN is to incorporate artificial viscosity for regularization, i.e., to consider

$$
\partial _ { t } { \bf U } ( t , { \bf x } ) + \nabla _ { \bf x } \cdot { \bf F } ( { \bf U } ( t , { \bf x } ) ) = \varepsilon \Delta _ { \bf x } { \bf U } ( t , { \bf x } ) ,
$$

for a suitable regularization parameter $\varepsilon > 0$ , where $\begin{array} { r } { \Delta _ { \mathbf { x } } = \sum _ { j = 1 } ^ { d } \partial _ { x _ { j } } ^ { 2 } } \end{array}$ is the Laplacian operator. Although this makes solutions smooth, it also permanently smooths out the discontinuities, producing smeared shock profiles whose width is controlled by $\varepsilon ,$ rather than reproducing the sharp discontinuities present in the true solution. In contrast, classical numerical methods such as the Lax-Friedrichs and Godunov schemes also introduce numerical viscosity, but this viscosity vanishes as the mesh is refined, and convergence to the physically correct solution is guaranteed by the mathematical framework of weak formulations and entropy conditions (LeVeque, 2002). This well-established framework has inspired several weak-form PINN approaches.

Weak PINN (WPINN) (De Ryck et al., 2024) incorporates residuals computed from the weak formulation and the entropy condition into the loss function. The key feature of WPINN is that it adopts an adversarial framework in which two neural networks are trained simultaneously: one to approximate the solution and the other to generate test functions. These networks are trained with opposite objectives: the solution network minimizes the residual loss, while the test function network maximizes it. This adversarial setup is designed to identify the most informative test functions without exhaustively searching the test function space, thereby improving eficiency. Chaumet and Giesselmann (2022) proposed several modifications to the WPINN framework, including dual norm computation for training eficiency, as well as extensions to general boundary conditions and systems of conservation laws. Zhou and Shi (2025) further extended WPINN to complex geometries. However, as noted in De Ryck et al. (2024); Chaumet and Giesselmann (2022), the adversarial training of WPINN faces stability challenges in practice. The the test function network may require frequent re-initialization during training, and the predictions can be sensitive to hyperparameter choices.

Variational PINN (VPINN) (Kharazmi et al., 2019, 2021) is another variant of PINNs that leverages integration by parts to transfer derivatives from the predicted solution to test functions. The VPINN formulates its loss function as the sum of residuals from the resulting integral equations. Subsequent work analyzed the role of quadrature rules and test function selection in VPINNs (Berrone et al., 2021), and proposed a robust variant based on discrete dual norm minimization (Rojas et al., 2023). Although it is not specifically developed for hyperbolic conservation laws, one of its residual terms naturally corresponds to a weak formulation. However, VPINN does not enforce the entropy condition, which is essential for selecting physically admissible solutions among multiple weak solutions.

Beyond weak-form approaches, a variety of other strategies have been proposed to address discontinuities when using PINNs. For instance, Jagtap and Karniadakis (2020); Jagtap et al. (2020); Lorin and Novruzi (2024) adopt domain decomposition by dividing the solution domain into subregions separated at the shock locations. However, this approach requires prior knowledge of shock locations, limiting its applicability. More complex training pipelines (Wang et al., 2025) have also been explored, in which a PINN is iteratively trained to identify shock locations, followed by domain decomposition into subdomains that contain no discontinuities. Independent PINNs are then trained within each subdomain, but this approach becomes increasingly impractical when multiple discontinuities interact or merge over time, as the subdomain configuration must be continuously updated. Gradient-weighting strategies (Liu et al., 2024; Liang et al., 2024; Ghoreishi and Naderan, 2026) suppress large residual gradients near shocks to mitigate training instability. While efective for problems with stationary or isolated discontinuities, their performance degrades when shocks are moving or multiple discontinuities interact. Network architecture improvements (Lei et al., 2025a,b) provide another approach by incorporating Kolmogorov-Arnold network (KAN) layers with explicit high-frequency feature embeddings to better capture shock waves. However, these methods still require additional regularization such as adaptive artificial viscosity.

Alternative reformulations include a lift-and-embed method (Sun et al., 2024) that lifts the problem to a higher dimension where solutions become smooth, though it relies on known initial locations of discontinuities. In Zhang et al. (2022), an implicit form is used to design the loss function, thereby avoiding diferentiation of the predicted solution. Subsequent work (Zeng et al., 2025) extends this framework with solution bounds and Rankine-Hugoniot jump conditions to improve shock detection and the accuracy of predicted shock speeds. However, the implicit form does not enforce the entropy condition that is necessary for selecting physically admissible solutions. In Su et al. (2025), a Stable Physics-Informed Kernel Evolution (SPIKE) method is proposed. This method employs reproducing kernel representations of solutions, which are equivalent to two-layer neural networks, together with Tikhonov regularization. However, its applicability to multidimensional problems remains to be explored. Cai et al. (2022, 2023) proposed a least-squares ReLU neural network (LSNN) that leverages finite-volume discretization of conservation laws, though its extension to systems of conservation laws has not been demonstrated.

The aforementioned methods either require prior knowledge of discontinuity locations, lack essential constraints such as the entropy condition, introduce artificial smoothing that degrades accuracy near discontinuities, or sufer from training instability and ineficiency. These challenges become particularly severe in regimes involving complex nonlinear wave dynamics, including shock formation from initially smooth profiles, shock merging, and shock-rarefaction interactions.

To overcome these challenges, we propose a novel PINN framework for hyperbolic conservation laws that rigorously integrates weak formulations and entropy conditions while ensuring stable and eficient training. This mathematically grounded foundation mirrors the classical theory and numerical methods for hyperbolic conservation laws, making our approach applicable to general systems of conservation laws in multiple space dimensions. We refer to this method as the Weak-

Entropy PINN (WEPINN), which addresses the challenges of discontinuous solutions through three key components:

• the weak formulation of the conservation laws as integral equations, which transfers all spatial and temporal derivatives onto smooth test functions (e.g., trigonometric polynomials) and inherently accommodates discontinuities without artificial smoothing or prior knowledge of their locations;

• enforcement of the entropy condition as integral inequalities to select the physically admissible solution, enabling the method to correctly determine whether an initial discontinuity evolves into a shock or a rarefaction wave;

• use of the discrete fast Fourier transform (DFFT) for eficient and accurate evaluation of the integrals appearing in both the weak formulation and the entropy condition, significantly reducing computational cost compared to standard numerical integration.

A key feature of the proposed WEPINN is that it pre-selects a set of test functions and minimizes the residuals over all of them simultaneously via least-squares optimization, avoiding the adversarial training employed in WPINN and leading to stable and eficient training. We demonstrate the efectiveness of WEPINN through extensive numerical experiments covering one-dimensional scalar conservation laws (the linear advection equation, Burgers’ equation, and the LWR trafic model), one-dimensional system of conservation laws (the compressible Euler equations), and twodimensional scalar conservation laws (the 2D Burgers’ equation), under a variety of initial and boundary conditions. Our experiments systematically evaluate a wide range of nonlinear wave phenomena, including shock formation from smooth initial conditions, propagation of shock and rarefaction waves, shock merging, and shock-rarefaction interactions. Furthermore, we introduce two shock-aware evaluation metrics, the shock detection rate and shock position accuracy, which complement traditional $\mathrm { L } ^ { p }$ error measures by directly assessing the presence and location of discontinuities predicted by a model.

The remainder of this paper is organized as follows. Section 2 introduces the mathematical framework of weak entropy solutions and presents the detailed design of the proposed Weak-Entropy PINN (WEPINN). Section 3 provides extensive numerical experiments and performance evaluations. Section 4 presents ablation studies that examine the role of the entropy condition and the choice of test functions in WEPINN. Section 5 concludes the paper and discusses potential future directions.

## 2 Methods

In this section, we present the Weak-Entropy PINN (WEPINN) framework. We begin by reviewing the theory of weak entropy solutions to hyperbolic conservation laws, which provides the mathematical foundation of our method, and then describe the design of the WEPINN.

## 2.1 Weak entropy solutions to hyperbolic conservation laws

We consider the hyperbolic conservation law (1) on a hypercube $\Omega = [ a , b ] ^ { d } \subset \mathbb { R } ^ { d }$ for some $a < b _ { \mathrm { : } }$ equipped with the initial condition

$$
\begin{array} { r } { { \mathbf U } ( 0 , { \mathbf x } ) = { \mathbf U } _ { 0 } ( { \mathbf x } ) \quad \mathrm { f o r } \quad { \mathbf x } \in { \Omega } , } \end{array}\tag{3}
$$

and periodic boundary conditions. In this work, we address non-periodic boundary conditions only for one-dimensional conservation laws (cf. Section 2.4).

For discontinuous solutions to (1), the derivatives $\partial _ { t } \mathbf { U }$ and $\nabla _ { \mathbf { x } } \cdot \mathbf { F } ( \mathbf { U } )$ in (1) do not exist at the discontinuities, so that the equation fails to describe the propagation of the discontinuities. This necessitates considering weak solutions defined by the following integral equation:

$$
\begin{array} { r l } { \mathrm { [ W e a k ] } } & { \displaystyle \int _ { 0 } ^ { \infty } \int _ { \Omega } \mathbf { U } ( t , \mathbf { x } ) \partial _ { t } \varphi ( t , \mathbf { x } ) + \mathbf { F } ( \mathbf { U } ( t , \mathbf { x } ) ) \cdot \nabla _ { \mathbf { x } } \varphi ( t , \mathbf { x } ) d \mathbf { x } d t + \int _ { \Omega } \mathbf { U } _ { 0 } ( \mathbf { x } ) \varphi ( 0 , \mathbf { x } ) d \mathbf { x } = 0 , } \end{array}\tag{4}
$$

where $\varphi : [ 0 , \infty ) \times \Omega \to \mathbb { R }$ is any spatially periodic, temporally compactly supported, and continuously diferentiable test function.

The integral equation (4), known as the weak formulation of equation (1), is obtained by multiplying (1) by the test function $\varphi$ and integrating by parts. The weak formulation (4) imposes no smoothness condition on U and yields the Rankine-Hugoniot jump condition (Dafermos, 1983) that capture the propagation of discontinuities. However, weak solutions defined by (4) are generally non-unique; in particular, multiple weak solutions can correspond to the same initial data, difering in how discontinuities form. To select physically admissible solutions, one crucial criterion is the entropy condition:

$$
\begin{array} { r l } { \mathrm { [ E n t r o p y ] } } & { \displaystyle \int _ { 0 } ^ { \infty } \int _ { \Omega } \eta ( { \mathbf U } ( t , { \mathbf x } ) ) \partial _ { t } \varphi ( t , { \mathbf x } ) + { \mathbf q } ( { \mathbf U } ( t , { \mathbf x } ) ) \cdot \nabla _ { { \mathbf x } } \varphi ( t , { \mathbf x } ) d { \mathbf x } d t + \int _ { \Omega } \eta ( { \mathbf U } _ { 0 } ( { \mathbf x } ) ) \varphi ( 0 , { \mathbf x } ) d { \mathbf x } \ge 0 , } \end{array}\tag{5}
$$

where $\eta : \mathbb { R } ^ { m }  \mathbb { R }$ and $\mathbf { q } : \mathbb { R } ^ { m }  \mathbb { R } ^ { d }$ form an entropy-entropy flux pair satisfying

$$
\frac { \partial q _ { j } } { \partial u _ { k } } = \sum _ { l = 1 } ^ { m } \frac { \partial \eta } { \partial u _ { l } } \frac { \partial F _ { j , l } } { \partial u _ { k } } \quad \mathrm { f o r } \quad j = 1 , \cdots , d , \quad k = 1 , \cdots , m ,\tag{6}
$$

where $u _ { k }$ denotes the k-th component of U, $q _ { j }$ the $j \cdot$ th component of ${ \bf q } ,$ and $F _ { j , l }$ the l-th component of the j-th flux $F _ { j }$ of $\mathbf { F }$ . When the solution U is smooth, the chain rule and (6) yield a conservation law for the entropy:

$$
\begin{array} { r } { \partial _ { t } \eta ( \mathbf { U } ) + \nabla _ { \mathbf { x } } \cdot \mathbf { q } ( \mathbf { U } ) = 0 . } \end{array}
$$

When discontinuities form, however, this equation no longer holds. The entropy condition (5) replaces it with integral inequalities over all nonnegative test functions $\varphi _ { : }$ , characterizing the dissipation of the entropy $\eta ( \mathbf { U } )$ and the irreversibility of the time evolution of U.

The weak formulation (4) and the entropy condition (5) together give the following definition of weak entropy solutions.

Definition 1. Suppose that E is a family of entropy-entropy flux pairs $( \eta , \mathbf { q } )$ that satisfy condition (6), and let T denote the space of functions $\varphi \in \mathrm { C } ^ { 1 } ( [ 0 , \infty ) \times \Omega )$ that are spatially periodic on Ω and compactly supported in time. A function $\mathbf { U } \in \mathrm { L } ^ { \infty } ( ( 0 , \infty ) \times \Omega ) \cap \mathrm { C } ( ( 0 , \infty ) ; \mathrm { L } ^ { 1 } ( \Omega ) )$ is called a weak entropy solution to the hyperbolic conservation law (1) with the initial condition (3) if it satisfies the weak formulation (4) for all test functions $\varphi \in { \mathcal { T } }$ and the entropy condition (5) for all nonnegative test functions $\varphi \in { \mathcal { T } }$ and every $( \eta , \mathbf { q } ) \in \mathcal { E }$

For scalar conservation laws $( m = 1 )$ , any $\mathrm { C ^ { 1 } }$ convex function $\eta : \mathbb { R }  \mathbb { R }$ serves as an entropy, with the corresponding entropy flux $\mathbf { q } : \mathbb { R }  \mathbb { R } ^ { d }$ defined component-wise by $q _ { j } ^ { \prime } ( \xi ) = \eta ^ { \prime } ( \xi ) F _ { j } ^ { \prime } ( \xi )$ for $j = 1 , \ldots , d .$ The existence and uniqueness of weak entropy solutions in this case are well established for any spatial dimension $d ,$ following the classical theory of Kruˇzkov (Kruˇzkov, 1970).

Theorem 2. Suppose that $m = 1$ . For any initial data $\mathbf { U } _ { 0 } \in \mathrm { L } ^ { \infty } ( \Omega )$ , there exists a unique weak entropy solution $\mathbf { U } \in \mathrm { L } ^ { \infty } ( ( 0 , \infty ) \times \Omega ) \cap \mathrm { C } ( ( 0 , \infty ) ; \mathrm { L } ^ { 1 } ( \Omega ) )$ to the scalar conservation law (1) in the sense of Definition 1, where the family E consists of all entropy-entropy flux pairs $( \eta , \mathbf { q } )$ with $\mathrm { C ^ { 1 } }$ convex η.

For systems of conservation laws $( m > 1 )$ , no generic construction of entropy-entropy flux pairs is available. However, certain systems arising from gas and fluid dynamics admit natural entropy structures. For example, the compressible Euler equations possess entropy pairs derived from thermodynamic entropy, as detailed in Section 3.3. The existence and uniqueness of weak entropy solutions for general systems remain open problems (Dafermos, 1983, 2023) and are beyond the scope of this work.

![](images/c3bf0e6f9dce67b3b6663798f0d661f61fc2cf4c877a0e9e39e968aebe8327aa.jpg)  
Figure 1: Pipeline of the proposed WEPINN

## 2.2 Neural network approximation to solutions

We approximate solutions to the hyperbolic conservation law (1) using a neural network

$$
\begin{array} { r } { ( t , \mathbf { x } ) \in \mathbb { R } ^ { d + 1 } \longmapsto \hat { \mathbf { U } } ( t , \mathbf { x } ; \theta ) \in \mathbb { R } ^ { m } , } \end{array}\tag{7}
$$

where θ denotes the collection of all trainable parameters (weights and biases) of the network. The solutions may be discontinuous but remain of bounded variation, a regularity class that neura networks can approximate to arbitrary accuracy (Ryck and Mishra, 2024).

In the Dif-PINN framework, the neural network is trained by minimizing a loss function that enforces the governing equation (1) in its strong form and the initial and boundary conditions. The loss function is defined as

$$
\mathcal { L } _ { \mathrm { S t r o n g } } = \mathcal { L } _ { \mathrm { P D E } } + \mathcal { L } _ { \mathrm { I C } } + \mathcal { L } _ { \mathrm { B C } } ,
$$

where $\mathcal { L } _ { \mathrm { P D E } }$ denotes the pointwise PDE residual as in (2), and $\mathcal { L } _ { \mathrm { I C } }$ and $\mathcal { L } _ { \mathrm { B C } }$ penalize deviations from the prescribed initial and boundary conditions at collocation points sampled on the initial time slice and along the domain boundary, respectively.

For hyperbolic conservation laws, the true solution U may develop discontinuities, making $\mathcal { L } _ { \mathrm { S t r o n g } }$ an inappropriate training objective. In our WEPINN framework, we replace ${ \mathcal { L } } _ { \mathrm { S t r o n g } }$ with a weak-entropy loss based on (4)–(5), as described in the following subsection.

## 2.3 Weak-Entropy PINN

The proposed WEPINN is built on the notion of weak entropy solutions defined in Definition 1. As illustrated in Figure 1, we employ a standard PINN architecture to represent the approximate solution as a neural network: $( t , \mathbf { x } ) \mapsto { \hat { \mathbf { U } } } ( t , \mathbf { x } ; \theta )$ , and design a weak-entropy loss by discretizing (4)–(5) to train the neural network.

We discretize (4)–(5) on a finite time horizon [0, T] where $T > 0$ is the terminal time of interest, using a spatial mesh $\{ \mathbf { x } _ { j } \} _ { j \in \Omega _ { d } }$ on Ω and a temporal mesh $\{ t _ { k } \} _ { k = 0 } ^ { N _ { t } }$ on [0, T], with $\{ \beta _ { j } \} _ { j \in \Omega _ { d } }$ and $\{ \alpha _ { k } \} _ { k = 0 } ^ { N _ { t } }$ the corresponding quadrature weights. That is, for any function $h : [ 0 , T ] \times \Omega  \mathbb { R }$ , the integral $\begin{array} { r } { \int _ { 0 } ^ { T } \int _ { \Omega } h ( t , \mathbf { x } ) } \end{array}$ dxdt is discretized as

$$
\sum _ { j \in \Omega _ { d } } \sum _ { k = 0 } ^ { N _ { t } } \alpha _ { k } \beta _ { j } h ( t _ { k } , x _ { j } ) .
$$

In our implementation, both $\{ \alpha _ { k } \} _ { k = 0 } ^ { N _ { t } }$ and $\{ \beta _ { j } \} _ { j \in \Omega _ { d } }$ are taken as trapezoidal-rule quadrature weights on the uniform grid.

For any test function $\varphi ,$ we discretize the weak formulation (4) as

$$
\mathcal { T } _ { t , { \bf x } } ^ { \varphi } + \mathcal { T } _ { 0 , { \bf x } } ^ { \varphi } = 0 ,
$$

through numerical quadratures:

$$
\mathcal { Z } _ { t , \mathbf { x } } ^ { \varphi } = \sum _ { j \in \Omega _ { d } } \sum _ { k = 0 } ^ { N _ { t } } \alpha _ { k } \beta _ { j } \big ( \hat { \mathbf { U } } ( t _ { k } , \mathbf { x } _ { j } ; \theta ) \partial _ { t } \varphi ( t _ { k } , \mathbf { x } _ { j } ) + \mathbf { F } ( \hat { \mathbf { U } } ( t _ { k } , \mathbf { x } _ { j } ; \theta ) ) \cdot \nabla _ { \mathbf { x } } \varphi ( t _ { k } , \mathbf { x } _ { j } ) \big ) ,\tag{8}
$$

$$
\mathcal { T } _ { 0 , \mathbf { x } } ^ { \varphi } = \sum _ { j \in \Omega _ { d } } \beta _ { j } \mathbf { U } _ { 0 } ( \mathbf { x } _ { j } ) \varphi ( 0 , \mathbf { x } _ { j } ) ,\tag{9}
$$

Then, for any nonnegative test function $\varphi ,$ we discretize the entropy condition (5) as

$$
\mathcal { T } _ { t , x } ^ { \varphi } + \mathcal { T } _ { 0 , x } ^ { \varphi } \geq 0 ,
$$

through numerical quadratures:

$$
\mathcal { I } _ { t , x } ^ { \varphi } = \sum _ { j \in \Omega _ { d } } \sum _ { k = 0 } ^ { N _ { t } } \alpha _ { k } \beta _ { j } \big ( \eta ( \hat { \mathbf { U } } ( t _ { k } , \mathbf { x } _ { j } ; \theta ) ) \partial _ { t } \varphi ( t _ { k } , \mathbf { x } _ { j } ) + \mathbf { q } ( \hat { \mathbf { U } } ( t _ { k } , \mathbf { x } _ { j } ; \theta ) ) \cdot \nabla _ { \mathbf { x } } \varphi ( t _ { k } , \mathbf { x } _ { j } ) \big ) ,\tag{10}
$$

$$
\mathcal { T } _ { 0 , \mathbf { x } } ^ { \varphi } = \sum _ { j \in \Omega _ { d } } \beta _ { j } \eta ( \mathbf { U } _ { 0 } ( \mathbf { x } _ { j } ) ) \varphi ( 0 , \mathbf { x } _ { j } ) .\tag{11}
$$

With the numerical discretization of the weak formulation and the entropy condition above, our weak-entropy loss is defined as

$$
\mathcal { L } _ { \mathrm { W e a k - E n t r o p y } } = \mathcal { L } _ { \mathrm { W e a k } } + \mathcal { L } _ { \mathrm { E n t r o p y } } + \mathcal { L } _ { \mathrm { I C } } ,
$$

where

$$
\mathcal { L } _ { \mathrm { W e a k } } = \frac { 1 } { N } \sum _ { n = 1 } ^ { N } \Big \| \mathcal { T } _ { t , { \bf x } } ^ { \varphi _ { n } } + \mathcal { T } _ { 0 , { \bf x } } ^ { \varphi _ { n } } \Big \| ,
$$

$$
\mathcal { L } _ { \mathrm { E n t r o p y } } = \frac { 1 } { N } \sum _ { n = 1 } ^ { N } \mathrm { R e L U } \left( - \mathcal { T } _ { t , \mathbf { x } } ^ { \varphi _ { n } } - \mathcal { T } _ { 0 , \mathbf { x } } ^ { \varphi _ { n } } , 0 \right) ,
$$

$$
\mathcal { L } _ { \mathrm { I C } } = \frac { 1 } { N _ { 0 } } \sum _ { i = 1 } ^ { N _ { 0 } } \left. \hat { \mathbf { U } } ( 0 , \mathbf { x } _ { i } ^ { 0 } ; \boldsymbol { \theta } ) - \mathbf { U } _ { 0 } ( \mathbf { x } _ { i } ^ { 0 } ) \right. ^ { 2 } ,
$$

with $\mathcal { T } _ { t , \mathbf { x } } ^ { \varphi _ { n } } , \mathcal { T } _ { 0 , \mathbf { x } } ^ { \varphi _ { n } } , \mathcal { T } _ { t , \mathbf { x } } ^ { \varphi _ { n } } , \mathcal { T } _ { 0 , \mathbf { x } } ^ { \varphi _ { n } }$ as defined in equations (8)–(11), $\{ \varphi _ { n } \} _ { n = 1 } ^ { N }$ is a set of pre-selected test functions, and $\{ \mathbf { x } _ { i } ^ { 0 } \} _ { i = 1 } ^ { N _ { 0 } }$ a set of collocation points sampled on the initial time slice $t = 0$

Test functions.

The weak formulation in principle requires evaluating the residuals in (4) against all test functions $\varphi \in { \mathcal { T } }$ as in Definition 1. Since this is computationally infeasible, we approximate T by a finitedimensional subspace spanned by a representative set of test functions and evaluate the residuals only on this subset.

We adopt a trigonometric basis on a uniform grid, with a tensor-product structure in space and time. The tensor-product structure allows the high-dimensional summations in the weakentropy loss to factorize along each coordinate direction, and combined with a uniform grid and trigonometric basis functions, enables an eficient implementation through the discrete fast Fourier transform (DFFT), as described below. The choice of the trigonometric basis over alternatives such as orthogonal polynomials is supported by an ablation study in Section 4.2, with additional discussion in D.

Concretely, each test function in our finite set $\{ \varphi _ { n } \} _ { n = 1 } ^ { N }$ is indexed by a multi-index

$$
\nu ( n ) = ( \nu _ { 0 } ( n ) , \nu _ { 1 } ( n ) , \ldots , \nu _ { d } ( n ) ) ,
$$

where $\nu _ { 0 } ( n ) \in \{ 0 , \ldots , N _ { q } \}$ selects the temporal basis and $\nu _ { r } ( n ) \in \{ 0 , \ldots , N _ { p } \}$ selects the spatial basis in the r-th spatial dimension. The total number of test functions is $N \overset { \cdot } { = } ( N _ { q } + 1 ) ( N _ { p } + 1 ) ^ { d }$ and each test function factorizes as

$$
\varphi _ { n } ( t , \mathbf { x } ) = q _ { \nu _ { 0 } ( n ) } ( t ) \prod _ { r = 1 } ^ { d } p _ { r , \nu _ { r } ( n ) } ( x _ { r } ) .\tag{12}
$$

We use two sets of basis functions: one for the weak formulation, where test functions only need to be smooth, and one for the entropy condition, where test functions must additionally satisfy $\varphi \geq 0$

Suppose that the spatial domain $\Omega = [ 0 , 1 ] ^ { d }$ and the time horizon is $[ 0 , T ]$ . For the weak formulation, we adopt standard trigonometric bases. The spatial basis per dimension consists of sine-cosine pairs with frequencies $m = 1 , \ldots , N _ { p } / 2$ , where $N _ { p }$ is assumed even. For each spatial dimension $r = 1 , \ldots , d \colon$

$$
p _ { r , 0 } ( x _ { r } ) = 1 , \quad p _ { r , 2 m - 1 } ( x _ { r } ) = \sin ( 2 \pi m x _ { r } ) , \quad p _ { r , 2 m } ( x _ { r } ) = \cos ( 2 \pi m x _ { r } ) , \quad m = 1 , \ldots , N _ { p } / 2 ;\tag{13}
$$

For the temporal basis, we introduce a window function $w ( t ) = T - t$ to ensure $\varphi ( T , \mathbf { x } ) = 0$ The temporal basis uses sine-cosine pairs with frequencies $m / 2$ for $m = 1 , \ldots , N _ { q } / 2$ , where $N _ { q }$ is assumed even:

$$
q _ { 0 } ( t ) = w ( t ) , \quad q _ { 2 m - 1 } ( t ) = w ( t ) \sin ( \pi m t ) , \quad q _ { 2 m } ( t ) = w ( t ) \cos ( \pi m t ) , \quad m = 1 , \dots , N _ { q } / 2 .\tag{14}
$$

For the entropy condition, we modify the above bases to guarantee non-negativity. The spatial basis uses four nonnegative combinations per frequency $m = 1 , \ldots , N _ { p } / 4$ , where $N _ { p }$ is assumed divisible by 4:

$$
p _ { r , 0 } ( x _ { r } ) = 1 , \quad p _ { r , 4 m - 3 } ( x _ { r } ) = 1 + \sin ( 2 \pi m x _ { r } ) , \quad p _ { r , 4 m - 2 } ( x _ { r } ) = 1 + \cos ( 2 \pi m x _ { r } ) ,\tag{15}
$$

$$
p _ { r , 4 m - 1 } ( x _ { r } ) = 1 - \sin ( 2 \pi m x _ { r } ) , \quad p _ { r , 4 m } ( x _ { r } ) = 1 - \cos ( 2 \pi m x _ { r } ) , \quad m = 1 , \ldots , N _ { p } / 4 ,\tag{16}
$$

and the temporal basis uses four nonnegative combinations per frequency $m / 2$ for $m = 1 , \ldots , N _ { q } / 4 .$ where $N _ { q }$ is assumed divisible by 4, with the window function $w ( t )$

$$
q _ { 0 } ( t ) = w ( t ) , \quad q _ { 4 m - 3 } ( t ) = w ( t ) \bigl ( 1 + \sin ( \pi m t ) \bigr ) , \quad q _ { 4 m - 2 } ( t ) = w ( t ) \bigl ( 1 + \cos ( \pi m t ) \bigr ) ,\tag{17}
$$

$$
q _ { 4 m - 1 } ( t ) = w ( t ) \bigl ( 1 - \sin ( \pi m t ) \bigr ) , \quad q _ { 4 m } ( t ) = w ( t ) \bigl ( 1 - \cos ( \pi m t ) \bigr ) , \quad m = 1 , \ldots , N _ { q } / 4 .\tag{18}
$$

DFFT implementation.

The dominant computational cost of evaluating the weak-entropy loss lies in the residuals $\mathcal { T } _ { t , \mathbf { x } } ^ { \varphi _ { n } }$ and $\mathcal { I } _ { t , { \bf x } } ^ { \varphi _ { n } }$ , which involve summations over the full space-time grid. For high-frequency test functions, direct evaluation of these summations becomes both time- and memory-intensive.

Our choice of trigonometric test functions on a uniform grid allows these summations to be evaluated via the DFFT. The resulting computational complexity is reduced from $O ( N _ { t } ^ { 2 } N _ { x } ^ { 2 d } )$ for direct numerical quadrature to $O ( N _ { t } N _ { x } ^ { d }$ log $N _ { t } \log ^ { d } { N _ { x } } )$ , where $N _ { t }$ is the number of temporal grid points and $N _ { x }$ is the number of grid points per spatial dimension. Implementation details and empirical validation of the speedup are provided in E.

## 2.4 Boundary conditions

The weak-entropy loss defined in Section 2.3 assumes periodic boundary conditions. With the chosen periodic test functions, the boundary terms in the weak formulation (4) and the entropy condition (5) vanish identically, and no additional boundary loss is required in WEPINN.

For one-dimensional problems, the boundary of the domain $\Omega = [ a , b ]$ consists of two points $x = a$ and $x = b$ . In this case, we also consider the Dirichlet boundary condition:

$$
\mathbf { U } ( t , a ) = \mathbf { U } _ { 0 } ( a ) \quad { \mathrm { a n d } } \quad \mathbf { U } ( t , b ) = \mathbf { U } _ { 0 } ( b ) \quad { \mathrm { f o r ~ } } t \in [ 0 , T ] .
$$

This boundary condition is interpreted as extending the problem to the real line R by the following constant extension of solutions:

$$
\tilde { \mathbf { U } } _ { 0 } ( x ) = \left\{ \begin{array} { l l } { \mathbf { U } _ { 0 } ( a ) , \quad } & { x \leq a ; } \\ { \mathbf { U } _ { 0 } ( x ) , \quad } & { a < x < b ; } \\ { \mathbf { U } _ { 0 } ( b ) , \quad } & { x \geq b ; } \end{array} \right. \quad \tilde { \mathbf { U } } ( t , x ) = \left\{ \begin{array} { l l } { \mathbf { U } _ { 0 } ( a ) , \quad } & { x \leq a ; } \\ { \mathbf { U } ( t , x ) , \quad } & { a < x < b ; } \\ { \mathbf { U } _ { 0 } ( b ) , \quad } & { x \geq b ; } \end{array} \right.
$$

Then, we can study the initial value problem (1) for $\tilde { \textbf { U } }$ with the initial condition $\tilde { \mathbf { U } } ( 0 , x ) = \tilde { \mathbf { U } } _ { 0 } ( x )$ for $x \in \mathbb { R }$ . Provided that all waves propagate within the domain $[ a , b ]$ over the time horizon $[ 0 , T ]$ the extended solution coincides with the original one on $[ a , b ]$ , i.e., $\mathbf { U } ( t , x ) = \tilde { \mathbf { U } } ( t , x )$ for all $t \in [ 0 , T ]$ and $x \in [ a , b ]$ . This allows us to consider Riemann problems in one dimension.

For the Dirichlet boundary condition, the weak formulation and the entropy condition become

$$
\begin{array} { r l } { \mathrm { [ W e a k ] } } & { \displaystyle \int _ { 0 } ^ { T } \int _ { a } ^ { b } \mathbf { U } ( t , x ) \partial _ { t } \varphi ( t , x ) + \mathbf { F } ( \mathbf { U } ( t , x ) ) \partial _ { x } \varphi ( t , x ) d x d t + \int _ { a } ^ { b } \mathbf { U } _ { 0 } ( x ) \varphi ( 0 , x ) d x } \\ & { \displaystyle + \int _ { 0 } ^ { T } \mathbf { F } ( \mathbf { U } _ { 0 } ( a ) ) \varphi ( t , a ) d t - \int _ { 0 } ^ { T } \mathbf { F } ( \mathbf { U } _ { 0 } ( b ) ) \varphi ( t , b ) d t = 0 , } \\ { \mathrm { [ E n t r o p y ] } } & { \displaystyle \int _ { a } ^ { T } \int _ { a } ^ { b } \eta ( \mathbf { U } ( t , x ) ) \partial _ { t } \varphi ( t , x ) + \mathbf { q } ( \mathbf { U } ( t , x ) ) \partial _ { x } \varphi ( t , x ) d x d t + \int _ { a } ^ { b } \eta ( \mathbf { U } _ { 0 } ( x ) ) \varphi ( 0 , x ) d x } \\ & { \displaystyle + \int _ { 0 } ^ { T } \mathbf { F } ( \mathbf { U } _ { 0 } ( a ) ) \varphi ( t , a ) d t - \int _ { 0 } ^ { T } \mathbf { F } ( \mathbf { U } _ { 0 } ( b ) ) \varphi ( t , b ) d t \ge 0 , } \end{array}
$$

respectively. Correspondingly, the weak and entropy losses are updated as follows:

$$
\mathcal { L } _ { \mathrm { W e a k } } = \frac { 1 } { N } \sum _ { n = 1 } ^ { N } \Big \| \mathcal { T } _ { t , x } ^ { \varphi _ { n } } + \mathcal { T } _ { 0 , x } ^ { \varphi _ { n } } + \mathcal { T } _ { \mathrm { B C } } ^ { \varphi _ { n } } \Big \| ,
$$

$$
\mathcal { L } _ { \mathrm { E n t r o p y } } = \frac { 1 } { N } \sum _ { n = 1 } ^ { N } \mathrm { R e L U } \left( - \mathcal { I } _ { t , x } ^ { \varphi _ { n } } - \mathcal { I } _ { 0 , x } ^ { \varphi _ { n } } - \mathcal { I } _ { \mathrm { B C } } ^ { \varphi _ { n } } , 0 \right) ,
$$

where ${ \mathcal { T } } _ { t , x } ^ { \varphi _ { n } } , { \mathcal { T } } _ { 0 , x } ^ { \varphi _ { n } } , { \mathcal { T } } _ { t , x } ^ { \varphi _ { n } } , { \mathcal { T } } _ { 0 , x } ^ { \varphi _ { n } }$ are as in Section 2.3, and the boundary terms $\mathcal { I } _ { \mathrm { B C } } ^ { \varphi _ { n } }$ and $\mathcal { I } _ { \mathrm { B C } } ^ { \varphi _ { n } }$ are defined as:

$$
\begin{array} { r l } & { \mathcal { T } _ { \mathrm { B C } } ^ { \varphi _ { n } } = \displaystyle \sum _ { k = 0 } ^ { N _ { t } } \alpha _ { k } \mathbf { F } ( \mathbf { U } _ { 0 } ( a ) ) \varphi _ { n } ( t _ { k } , a ) - \displaystyle \sum _ { k = 0 } ^ { N _ { t } } \alpha _ { k } \mathbf { F } ( \mathbf { U } _ { 0 } ( b ) ) \varphi _ { n } ( t _ { k } , b ) , } \\ & { \mathcal { T } _ { \mathrm { B C } } ^ { \varphi _ { n } } = \displaystyle \sum _ { k = 0 } ^ { N _ { t } } \alpha _ { k } \mathbf { q } ( \mathbf { U } _ { 0 } ( a ) ) \varphi _ { n } ( t _ { k } , a ) - \displaystyle \sum _ { k = 0 } ^ { N _ { t } } \alpha _ { k } \mathbf { q } ( \mathbf { U } _ { 0 } ( b ) ) \varphi _ { n } ( t _ { k } , b ) . } \end{array}
$$

## 3 Numerical Experiments and Results

In this section, we evaluate the performance of the proposed method through comprehensive numerical experiments. Specifically, we consider scalar conservation laws and systems of conservation laws in one and two spatial dimensions, as summarized in Table 1.

<table><tr><td>Problem class</td><td>One-dimension</td><td>Two-dimension</td></tr><tr><td>Scalar conservation laws</td><td>Linear advection equation Burgers&#x27; equation LWR traffic model (Section 3.2)</td><td>Burgers&#x27; equation (Section 3.4)</td></tr><tr><td>Systems of conservation laws</td><td>Compressible Euler equations (Section 3.3)</td><td></td></tr></table>

Table 1: Benchmark problems and corresponding subsections, covering both scalar conservation laws and systems of conservation laws in one and two spatial dimensions.

## 3.1 Experimental setup

We compare the proposed WEPINN against three representative baselines that require only the governing PDEs, without prior knowledge of discontinuity locations: Dif-PINN, VPINN, and WPINN. Auxiliary techniques such as adaptive collocation sampling and domain decomposition are orthogonal to this comparison and could be combined with any of the formulations considered; therefore, we do not include them as separate baselines.

For Dif-PINN and VPINN, we adopt the same network architecture as in WEPINN, except that ReLU activations are replaced with Tanh for better performance, and we use the same number of uniformly sampled collocation points as in WEPINN. VPINN as originally proposed (Kharazmi et al., 2019, 2021) is not specifically designed for hyperbolic conservation laws. We adapt it by combining our weak-formulation loss with the pointwise PDE residual loss used in Dif-PINN, which corresponds to the core idea of VPINN applied to the present setting. All three methods use Adam for optimization. Full network architecture and training details are provided in Appendix C.

For WPINN, we adopt the implementation of Chaumet and Giesselmann (2022) rather than the original version of De Ryck et al. (2024), as it provides a TensorFlow implementation more compatible with our training environment. We import the initial and boundary conditions from our experimental setup into their code and evaluate the resulting solutions on a uniform mesh for downstream metric computation. The released code supports only Dirichlet boundary conditions and does not include an implementation for the compressible Euler equations. Accordingly, we report WPINN results only for the Dirichlet-boundary scalar cases, and for the Euler case we use the numbers reported in Chaumet and Giesselmann (2022).

To assess model performance, we evaluate both global solution accuracy and shock (discontinuity) resolution. Global accuracy is measured by the relative $\mathrm { L ^ { 2 } }$ error against a high-resolution reference solution computed with the Lax-Friedrichs scheme. To assess shock resolution, we first extract the shock locations from the predicted solution at each time step using a gradient-based peak detector, and then compare them against the ground-truth shock locations. This comparison yields two shock-aware metrics:

• the shock detection rate (S-Rate), defined as the fraction of ground-truth shocks correctly identified by the model;

• the shock position accuracy (S-Acc), defined as the mean positional error between the matched predicted and ground-truth shocks.

Together, the two metrics quantify whether a model captures the presence of shocks and how accurately it predicts their positions over a time horizon. The detailed procedure for computing these metrics is given in Algorithm 1 (Appendix A).

## 3.2 One-dimensional scalar conservation laws

In this subsection, we consider the one-dimensional scalar conservation law

$$
\partial _ { t } u ( t , x ) + \partial _ { x } f ( u ( t , x ) ) = 0 , \quad ( t , x ) \in [ 0 , 1 ] \times [ 0 , 1 ] ,\tag{19}
$$

subject to the initial condition $u ( 0 , x ) = u _ { 0 } ( x )$ for $x \in [ 0 , 1 ]$ and either the Dirichlet boundary condition described in Section 2.4 or periodic boundary conditions.

We consider three representative examples with diferent flux functions $f \colon$

• Linear advection equation, with $f ( u ) = c u$ for a constant advection speed c.

• (inviscid) Burgers’ equation, with $\begin{array} { r } { f ( u ) = \frac { 1 } { 2 } u ^ { 2 } } \end{array}$ , a classical benchmark in gas dynamics.

• LWR trafic model, with $f ( u ) = u ( 1 - u )$ , where u denotes trafic density.

For each combination of governing equation and boundary condition, we randomly generate 30 initial conditions drawn from several initial condition classes designed to cover a range of solution behaviors. Under the Dirichlet boundary condition, the classes are Sigmoid (smooth transitions between two states) and Riemann (piecewise-constant two- or three-state profiles). Under the periodic boundary condition, the classes are Fourier (smooth trigonometric series), Trig (highfrequency sinusoidal wave), Bell (smooth single-peaked profiles), and PWC (piecewise-constant profiles). Full generation procedures are given in B. Each model is trained independently on each initial condition, and the reported metrics are averaged across all trials.

Tables 2 and 3 summarize the results under Dirichlet and periodic boundary conditions, respectively. For the linear advection equation, no shocks form from the Sigmoid initial conditions, so we omit the shock-resolution metrics. In addition, since the entropy condition plays no role in solution selection for linear problems (Dafermos, 1983), we report only WEPINN without the entropy loss (WEPINN $\mathrm { w / o }$ Entropy) for the linear advection equation.

Under the Dirichlet boundary condition, WEPINN achieves substantially lower relative $\mathrm { L ^ { 2 } }$ errors than the baselines on Burgers’ equation and the LWR model, together with near-perfect S-Rate and the smallest S-Acc. These gains are consistent across both Sigmoid and Riemann initial conditions.

<table><tr><td rowspan="2">Problem</td><td rowspan="2">Method</td><td colspan="3">Riemann</td><td colspan="3">Sigmoid</td></tr><tr><td> $\overline { { \mathrm { ~ L } ^ { 2 } \downarrow } }$ </td><td>S-Rate↑</td><td>S-Acc↓</td><td> $\overline { { \mathrm { ~ L } ^ { 2 } \downarrow } }$ </td><td>S-Rate↑</td><td>S-Acc↓</td></tr><tr><td rowspan="4">IIiear</td><td>Diff-PINN</td><td>0.87</td><td>100</td><td>0.02</td><td>0.45</td><td>一</td><td>1</td></tr><tr><td>VPINN</td><td>0.91</td><td>100</td><td>0.03</td><td>0.63</td><td>一</td><td></td></tr><tr><td>WPINN</td><td>1.13</td><td>75.34</td><td>0.28</td><td>0.92</td><td>一</td><td></td></tr><tr><td>WEPINN w/o Entropy</td><td>0.93</td><td>100</td><td>0.03</td><td>0.87</td><td>=</td><td>一</td></tr><tr><td rowspan="6">Burrs</td><td>Diff-PINN</td><td>8.62</td><td>40.75</td><td>0.79</td><td>6.23</td><td>40.00</td><td>0.01</td></tr><tr><td>VPINN</td><td>5.34</td><td>56.13</td><td>0.43</td><td>5.89</td><td>48.90</td><td>0.05</td></tr><tr><td>WPINN</td><td>9.74</td><td>92.20</td><td>0.62</td><td>7.36</td><td>60.41</td><td>0.19</td></tr><tr><td>WEPINN w/o Entropy</td><td>3.68</td><td>82.42</td><td>0.06</td><td>3.12</td><td>99.36</td><td>0.07</td></tr><tr><td>WEPINN</td><td>1.84</td><td>98.58</td><td>0.05</td><td>1.53</td><td>99.41</td><td>0.04</td></tr><tr><td>Diff-PINN</td><td>6.12</td><td>41.60</td><td>1.49</td><td>3.37</td><td>32.07</td><td>0.81</td></tr><tr><td rowspan="5">TR</td><td>VPINN</td><td>6.53</td><td>47.32</td><td>0.95</td><td>3.15</td><td>31.76</td><td>0.15</td></tr><tr><td>WPINN</td><td>9.72</td><td>41.45</td><td>1.48</td><td>6.36</td><td>46.23</td><td></td></tr><tr><td>WEPINN</td><td>2.54</td><td>77.22</td><td>0.19</td><td>2.50</td><td>96.45</td><td>0.78</td></tr><tr><td>w/o Entropy</td><td></td><td></td><td></td><td></td><td></td><td>0.28</td></tr><tr><td>WEPINN</td><td>1.89</td><td>93.90</td><td>0.15</td><td>2.05</td><td>96.71</td><td>0.23</td></tr></table>

Table 2: Performance comparison under the Dirichlet boundary condition across problems and initial conditions. Reported metrics are the relative $\mathrm { L } ^ { 2 }$ error, the shock detection rate (S-Rate), and the shock position accuracy (S-Acc). Arrows indicate whether lower (↓) or higher (↑) is better. All numbers are reported in $\mathrm { e - 2 } .$

Both Dif-PINN and VPINN degrade sharply on Burgers’ equation and the LWR model, where shocks form. VPINN improves slightly on Dif-PINN in shock localization owing to the use of the weak formulation, but its S-Rate remains too low for S-Acc to be meaningful. WPINN, despite using the weak formulation with the entropy condition, also underperforms on these problems. By contrast, on the linear advection equation all methods achieve comparable accuracy. In particular, Dif-PINN achieves the lowest relative $\mathrm { L } ^ { 2 }$ error on the smooth Sigmoid initial conditions. This suggests that the failure of Dif-PINN on nonlinear problems stems from the ill-posed pointwise residual near discontinuities rather than from any limitation of the neural network’s approximation capacity. These findings show that the proposed WEPINN is the most accurate, particularly in shock resolution, among all methods for nonlinear problems involving shocks.

Under the periodic boundary condition, WEPINN continues to outperform the baselines on all problems. In Table 3, the best result in every cell, across the relative $\mathrm { L ^ { 2 } }$ error, S-Rate, and S-Acc metrics and across all problems and initial conditions, is obtained by either WEPINN or WEPINN $\mathrm { w / o }$ Entropy. The two variants alternate in achieving the best S-Rate and S-Acc, depending on the initial condition class. This suggests that the weak formulation already captures most of the shock structure accurately, while the entropy loss mainly serves to select the physically admissible solution. Among the baselines, Dif-PINN and VPINN again degrade sharply once shocks form, with S-Rate values below 20 across most initial conditions on Burgers’ equation and the LWR model. These findings indicate that the proposed WEPINN is also the most accurate method under the periodic boundary condition.

To further illustrate the qualitative diferences among the methods, we present a representative Burgers’ equation example in Figure 2. The example is chosen so that, within a single solution, one observes the formation and propagation of shocks and rarefactions, shock–shock merging, and shock–rarefaction interaction. The columns show the solution at diferent time snapshots from $t = 0$ $t = 1$ , while the rows correspond to diferent methods. The ground-truth solution is shown as a dashed line for comparison, and the ground-truth shock locations are indicated by solid vertical lines.

![](images/f90b72f782d6e69bbf97eee1788bf3f3bb3982c7d662f779eb61ec870b80a0c1.jpg)  
Figure 2: Qualitative comparison on a Burgers’ equation example whose solution contains rich phenomena including formation of shocks and rarefactions as well as their interactions. Columns show the solution at $t = 0 , 0 . 2 5 , 0 . 5 , 0 . 7 5 , 1 . 0$ . Rows correspond to Dif-PINN, VPINN, WPINN, WEPINN without the entropy loss, and the full WEPINN, together with the ground truth solution in dashed line. Solid vertical lines mark the ground-truth shock locations.

<table><tr><td rowspan="2">Problem</td><td rowspan="2">Method</td><td colspan="3"> $\overline { { { \bf { T r i g } } } }$ </td><td colspan="3">Fourier</td><td colspan="3">Bell</td><td colspan="3">PWC</td></tr><tr><td> $\overline { { \mathrm { ~ L ~ } ^ { 2 } } }$  ↓</td><td>S-Rate↑</td><td> $\overline { { \mathrm { S - A c c } \downarrow } }$ </td><td> $\overline { { \mathrm { ~ L ~ } ^ { 2 } \downarrow } }$ </td><td>S-Rate↑</td><td> $\overline { { \mathrm { S - A c c } \downarrow } }$ </td><td> $\overline { { \mathrm { ~ L ~ } ^ { 2 } \downarrow } }$ </td><td>S-Rate↑</td><td> $_ { \overline { { \mathrm { S - A c c } } } }$ </td><td> $\overline { { \mathrm { ~ L ~ } ^ { 2 } \downarrow } }$ </td><td>S-Rate↑</td><td> $\overline { { \mathrm { S - A c c } \downarrow } }$ </td></tr><tr><td></td><td>Diff-PINN</td><td>2.05</td><td></td><td></td><td>1.33</td><td></td><td></td><td>1.55</td><td></td><td></td><td>2.99</td><td>86.09</td><td>0.30</td></tr><tr><td>Iinear</td><td>VPINN</td><td>2.14</td><td></td><td></td><td>1.30</td><td></td><td></td><td>1.36</td><td></td><td></td><td>1.58</td><td>85.99</td><td>0.31</td></tr><tr><td></td><td>WEPINN w/o Entropy</td><td>1.89</td><td></td><td></td><td>0.53</td><td></td><td></td><td>0.95</td><td></td><td></td><td>0.96</td><td>99.19</td><td>0.20</td></tr><tr><td rowspan="4">Burrs</td><td>Diff-PINN</td><td>8.22</td><td>1.04</td><td>0.12</td><td>5.51</td><td>1.66</td><td>0.18</td><td>6.06</td><td>9.69</td><td>0.45</td><td>6.36</td><td>12.87</td><td>0.55</td></tr><tr><td>VPINN</td><td>8.47</td><td>0.45</td><td>0.05</td><td>5.26</td><td>2.57</td><td>0.10</td><td>7.31</td><td>5.86</td><td>0.08</td><td>6.15</td><td>15.78</td><td>0.52</td></tr><tr><td>WEPINN</td><td>2.15</td><td>46.57</td><td>0.08</td><td>0.89</td><td>95.88</td><td>0.25</td><td>0.91</td><td>99.85</td><td>0.23</td><td>1.64</td><td>97.94</td><td>0.27</td></tr><tr><td>w/o Entropy WEPINN</td><td>1.80</td><td>57.83</td><td>0.10</td><td>0.72</td><td>94.25</td><td>0.11</td><td>0.82</td><td>99.40</td><td>0.15</td><td>1.28</td><td>95.16</td><td>0.22</td></tr><tr><td rowspan="4">IR</td><td>Diff-PINN</td><td>9.39</td><td>0.60</td><td>0.00</td><td>6.12</td><td>0.00</td><td>0.00</td><td>9.27</td><td>3.58</td><td>0.22</td><td>11.15</td><td>16.19</td><td>0.62</td></tr><tr><td>VPINN</td><td>7.50</td><td>2.32</td><td>0.14</td><td>4.57</td><td>2.94</td><td>0.40</td><td>12.00</td><td>0.95</td><td>0.23</td><td>10.96</td><td>30.59</td><td>0.75</td></tr><tr><td>WEPINN</td><td>3.36</td><td>76.39</td><td>0.40</td><td>1.63</td><td>83.07</td><td>0.61</td><td>2.58</td><td>91.94</td><td>0.32</td><td>4.35</td><td>98.30</td><td>0.32</td></tr><tr><td>w/o Entropy WEPINN</td><td>2.98</td><td>71.58</td><td>0.13</td><td>0.80</td><td>88.71</td><td>0.15</td><td>0.95</td><td>99.21</td><td>0.18</td><td>1.85</td><td>97.78</td><td>0.21</td></tr></table>

Table 3: Performance comparison under the periodic boundary condition across problems and initial conditions. Reported metrics are the relative $\mathrm { L } ^ { 2 }$ error, the shock detection rate (S-Rate), and the shock position accuracy (S-Acc). Arrows indicate whether lower (↓) or higher (↑) is better. All numbers are reported in $\mathrm { e - 2 }$

We observe from Figure 2 that Dif-PINN smooths out the solution throughout the temporal evolution. A rarefaction from the left-hand initial discontinuity is captured with noticeable error in magnitude, but the shock formation and propagation, shock-shock merging, and shock-rarefaction interaction are all mispredicted. VPINN ofers only a marginal improvement and additionally introduces small spurious oscillations near the boundary. The WPINN prediction shows only a single traveling wave from $t = 0 . 2 5$ onward, missing most of the wave formation and interaction phenomena. Using our weak-formulation loss alone (WEPINN $\mathrm { w / o }$ Entropy), the model accurately captures shock formation and propagation as well as shock–shock merging, but fails to capture the rarefaction. Only the full WEPINN simultaneously captures all the key phenomena in the nonlinear dynamics, including the formation and propagation of shocks and rarefactions, shock–shock merging, and shock–rarefaction interaction, in close agreement with the ground truth. These observations are consistent with the quantitative results in Tables 2 and 3, and underscore the role of the entropy condition: without it, the model cannot distinguish a physical rarefaction from a spurious shock satisfying the weak formulation. This point is further examined in the ablation study (Section 4.1), where a simpler example isolates the efect of the entropy loss.

## 3.3 One-dimensional system of conservation laws

In this subsection, we consider the one-dimensional compressible Euler equations,

$$
\partial _ { t } \mathbf { U } + \partial _ { x } \mathbf { F } ( \mathbf { U } ) = 0 ,\tag{20}
$$

where $\mathbf { U } = ( \rho , \rho u , E ) ^ { \top }$ denotes the vector of conserved quantities, corresponding to mass, momentum, and energy conservation, respectively. The physical unknowns are the mass density $\rho ,$ the velocity $u ,$ and the pressure $p ;$ the energy density E is determined by the ideal-gas equation of state:

$$
E = \frac { p } { \gamma - 1 } + \frac { 1 } { 2 } \rho u ^ { 2 }
$$

with an adiabatic index $\gamma > 1$ . The flux function is

$$
\mathbf { F } ( \mathbf { U } ) = \left( \begin{array} { c } { \rho u } \\ { \rho u ^ { 2 } + p } \\ { u ( E + p ) } \end{array} \right) .
$$

The Euler system is a hyperbolic system of conservation laws and admits rich wave phenomena, including shocks, contact discontinuities, and rarefactions. As with the scalar case, an entropy condition is required to select the physically admissible solution. For the Euler system, the natural entropy-entropy flux pair is derived from the thermodynamic entropy. Following the classical construction (Dafermos, 1983), we take

$$
\eta ( \mathbf { U } ) = - \rho s , \quad q ( \mathbf { U } ) = - \rho u s \quad { \mathrm { w h e r e } } \quad s = \ln \left( { \frac { p } { \rho ^ { \gamma } } } \right) .
$$

The pair $( \eta , q )$ satisfies the compatibility condition (6), and $\eta$ is convex on the physically admissible region $p , \rho > 0$

We evaluate the proposed method on the Sod shock tube problem, a standard benchmark for the compressible Euler equations. From a piecewise-constant initial condition separating two constant states, the solution develops three characteristic wave structures: a rarefaction, a contact discontinuity, and a shock, each propagating at a diferent speed. Dirichlet boundary conditions are imposed on the domain $\Omega = [ 0 , 1 ]$

<table><tr><td rowspan="2">Method</td><td colspan="2">density  $\rho$ </td><td colspan="3">speed u</td><td colspan="2">pressure  $p$ </td></tr><tr><td> $\overline { { \mathrm { ~ L ~ } ^ { 2 } \downarrow } }$  S-Rate↑</td><td>S-Acc↓</td><td> $\overline { { \mathrm { L } ^ { 2 } \downarrow } }$ </td><td>S-Rate↑ S-Acc↓</td><td> $\overline { { \mathrm { L } ^ { 2 } \downarrow } }$ </td><td>S-Rate↑</td><td>S-Acc↓</td></tr><tr><td>Diff-PINN</td><td>6.93 37</td><td>2.45</td><td>32.2</td><td>0 N. A.</td><td>3.46</td><td>25</td><td>0.35</td></tr><tr><td>VPINN</td><td>6.53 43</td><td>1.88</td><td>33.7</td><td>5 1.95</td><td>3.15</td><td>29</td><td>0.32</td></tr><tr><td>WPINN</td><td>1.38 -</td><td>1</td><td>3.77</td><td>-</td><td>1.47</td><td>-</td><td>一</td></tr><tr><td>WEPINN w/o Entropy</td><td>0.90 79</td><td>0.52</td><td>2.87</td><td>99</td><td>0.51 1.13</td><td>100</td><td>0.36</td></tr><tr><td>WEPINN</td><td>0.60 99</td><td>0.30</td><td>2.34</td><td>100</td><td>0.37</td><td>0.61 100</td><td>0.29</td></tr></table>

Table 4: Performance comparison on the Sod shock tube problem for the compressible Euler equations. Reported metrics are the relative $\mathrm { L ^ { 2 } }$ error, the shock detection rate (S-Rate), and the shock position accuracy (S-Acc). Arrows indicate whether lower (↓) or higher (↑) is better. All numbers are reported in e-2.

Table 4 reports the relative $\mathrm { L } ^ { 2 }$ error, S-Rate, and S-Acc of each method on the density $\rho ,$ velocity $u ,$ and pressure $p$ predicted by the models, and Figure 3 visualizes the predicted solutions at time $t = 0 . 2$ . For WPINN, we quote the numbers reported in Chaumet and Giesselmann (2022); as discussed in Section 3.1, no implementation is available to us for the Euler equations, so S-Rate, $\mathrm { S - A c c } .$ , and the solution visualization are absent.

We observe from Table 4 that WEPINN achieves the lowest $\mathrm { L } ^ { 2 }$ error on all three variables, together with the highest S-Rate and lowest S-Acc. Figure 3 shows that WEPINN accurately reproduces all three waves: the rarefaction on the left, the contact discontinuity in the middle, and the shock on the right. Removing the entropy loss (WEPINN $\mathrm { w / o }$ Entropy) preserves the shock and the contact discontinuity but yields a less accurate rarefaction, consistent with the behavior observed on the scalar problems in Section 3.2. Dif-PINN and VPINN perform comparably, with VPINN ofering only a marginal improvement, and both methods yield smeared profiles across the spatial domain. These results confirm the efectiveness of WEPINN on this benchmark problem of compressible Euler equations, and that entropy enforcement remains essential in the presence of more complex wave structures.

![](images/a0d4efacfa7cf97885c532e51ea3131f0326ba4d4bec87712ecb1e002d41b651.jpg)  
Figure 3: Comparison of predicted solutions (density $\rho ,$ speed $u ,$ and pressure p in diferent colors) at t = 0.2 for the Sod shock tube problem for the compressible Euler equations. Solid lines indicate predictions and dashed lines indicate the ground truth. Each panel shows a diferent method.

## 3.4 Two-dimensional scalar conservation laws

In this subsection, we consider a scalar Burgers’ equation in two space dimensions, obtained by taking $\begin{array} { r } { f ( u ) = \frac { 1 } { 2 } u ^ { 2 } } \end{array}$ as the flux in each spatial direction:

$$
\partial _ { t } \boldsymbol { u } + \nabla _ { \mathbf { x } } \cdot \mathbf { F } ( \boldsymbol { u } ) = 0 \quad \mathrm { w i t h } \quad \mathbf { F } ( \boldsymbol { u } ) = \left( \frac { u ^ { 2 } } { 2 } , \frac { u ^ { 2 } } { 2 } \right) ,\tag{21}
$$

or equivalently

$$
\partial _ { t } u + \partial _ { x } \left( \frac { u ^ { 2 } } { 2 } \right) + \partial _ { y } \left( \frac { u ^ { 2 } } { 2 } \right) = 0 .\tag{22}
$$

This is a special case of the two-dimensional scalar conservation laws studied in Zhang and Zheng (1989), and a natural two-dimensional extension of the one-dimensional Burgers’ equation in Section 3.2.

We consider the following two initial conditions:

• a disk profile

$$
u _ { 0 } ( x , y ) = \left\{ \begin{array} { l l } { { 0 . 5 , } } & { { ( x - 0 . 5 ) ^ { 2 } + ( y - 0 . 5 ) ^ { 2 } < 0 . 0 4 ; } } \\ { { 0 , } } & { { \mathrm { o t h e r w i s e } } } \end{array} \right.
$$

which is piecewise constant with a circular discontinuity of radius 0.2 centered at (0.5, 0.5);

• a smooth trigonometric profile

$$
u _ { 0 } ( x , y ) = \frac { 1 } { 4 } \sin ( 2 \pi x ) \sin ( 2 \pi y ) .
$$

The two initial conditions test complementary regimes: the disk one contains a circular initial discontinuity whose propagation needs to be tracked, while the trigonometric one is smooth and tests the model’s ability to capture shock formation from smooth data. We impose periodic boundary conditions in both spatial directions on the domain $\Omega = [ 0 , 1 ] ^ { 2 }$ . WPINN is not included in the comparison because no implementation for two-dimensional problems is available, as discussed in Section 3.1.

<table><tr><td rowspan=1 colspan=1>Method</td><td rowspan=1 colspan=1>Disk</td><td rowspan=1 colspan=1>Trigonometric</td></tr><tr><td rowspan=1 colspan=1>Diff-PINN</td><td rowspan=1 colspan=1>0.28</td><td rowspan=1 colspan=1>0.17</td></tr><tr><td rowspan=1 colspan=1>VPINN</td><td rowspan=1 colspan=1>0.27</td><td rowspan=1 colspan=1>0.21</td></tr><tr><td rowspan=1 colspan=1>WEPINNw/o Entropy</td><td rowspan=1 colspan=1>0.96</td><td rowspan=1 colspan=1>0.57</td></tr><tr><td rowspan=1 colspan=1>WEPINN</td><td rowspan=1 colspan=1>0.12</td><td rowspan=1 colspan=1>0.14</td></tr></table>

Table 5: Relative $\mathrm { L ^ { 2 } }$ errors on the two-dimensional scalar Burgers’ equation under periodic boundary conditions.

![](images/2263bd1cfd1578e7653722c0acbc373d60e5621d5db3553001b9756739e12fdb.jpg)

![](images/f2622fdcfbe33c0193aa5f29c68157af372fbacb0a252037314a56e1dfdd6862.jpg)  
Figure 4: Predicted solutions of the two-dimensional scalar Burgers’ equation under periodic boundary conditions. (Left) Disk initial condition; (Right) Trigonometric initial condition. Within each panel, rows correspond to the ground truth and diferent methods. Columns show the solution at $t = 0 , t = 0 . 5$ , and $t = 1$

Table 5 reports the relative $\mathrm { L ^ { 2 } }$ error of each method on the two initial conditions, and Figure 4 visualizes the predicted solutions at $t = 0 , t = 0 . 5$ , and $t = 1$ . Because shocks in two dimensions are curves rather than isolated points, the S-Rate and S-Acc metrics from Section 3.1 do not directly extend to this setting; we therefore report only the $\mathrm { L ^ { 2 } }$ errors.

Across both initial conditions, WEPINN achieves the lowest $\mathrm { L ^ { 2 } }$ error. Moreover, Figure 4 shows that WEPINN accurately tracks the propagation of the circular discontinuity in the disk case and captures shock formation from the smooth trigonometric profile. By contrast, Dif-PINN and VPINN smooth out the solution on both initial conditions, particularly at the intermediate time steps, and in some cases distort the initial condition itself.

The behavior of WEPINN without the entropy loss (WEPINN $\mathrm { w / o }$ Entropy) reveals a notable diference from the experiments on the one-dimensional scalar problems (Section 3.2), where WEPINN $\mathrm { w / o }$ Entropy is already competitive with the full WEPINN. In the two-dimensional setting, however, it produces the largest $\mathrm { L ^ { 2 } }$ error among all methods and fails to reproduce shock formation and propagation. This contrast suggests that the entropy loss becomes increasingly important in higher-dimensional problems.

## 4 Ablation Studies

## 4.1 Role of the entropy loss

The entropy condition (5) is essential for eliminating unphysical weak solutions to the hyperbolic conservation law (1) that satisfy the weak formulation (4) but violate physical admissibility. A classical illustrative example is the Riemann problem for the one-dimensional Burgers’ equation

$$
\partial _ { t } u + \partial _ { x } \left( \frac { u ^ { 2 } } { 2 } \right) = 0 , \qquad u ( 0 , x ) = \left\{ u _ { L } , \quad x < x _ { 0 } \right.\tag{23}
$$

with $u _ { L } < u _ { R }$ . For this problem, both a shock and a rarefaction satisfy the weak formulation, but only the rarefaction is admissible under the entropy condition. In WEPINN, the entropy loss enforces the entropy condition during training, so that the model converges to the weak entropy solution. Figure 5 illustrates this on the above Riemann problem with $x _ { 0 } = 0 . 5 , u _ { L } = - 0 . 2$ , and $u _ { R } = 0$ , under the Dirichlet boundary condition. Without the entropy loss, WEPINN converges to a propagating shock, while adding the entropy loss recovers the correct rarefaction. This sheds light on why the full WEPINN outperforms WEPINN without the entropy loss in most cases across the numerical experiments in Section 3, particularly on initial conditions where rarefactions develop.

![](images/e61113fce6950750747d19f6ec66a505aec10d134862f0c7fd56ba94ac1d22d2.jpg)

![](images/a518fa0d1d848b22c725481dd9cbaaadf00c6a7a258e58e1ddb1a75cc05f1a58.jpg)  
Figure 5: Role of the entropy loss in WEPINN, illustrated on the Riemann problem for Burgers equation (23) with $x _ { 0 } = 0 . 5 , u _ { L } = - 0 . 2 $ , and $u _ { R } = 0$ , under the Dirichlet boundary condition. (Left) WEPINN without the entropy loss; (Right) Full WEPINN. In each panel, the horizontal axis is the spatial coordinate x and the vertical axis is time t.

## 4.2 Test function selection

The choice of test functions used in the weak-entropy loss is critical to the performance of WEPINN. This includes both the function family (e.g., trigonometric functions or orthogonal polynomials such as Chebyshev polynomials) and the maximum degree of the functions, which controls their expressive power.

To determine the optimal maximum degree (i.e., maximum frequency) of the trigonometric test functions chosen in Section 2.3, we conduct an ablation study by varying the degree from 4 to 32. The results are presented in Table 6. We observe that degrees 16 and 32 yield the best performance. Moreover, degree 16 performs better on high-frequency data, while degree 32, which is used in the WEPINN implementation for the numerical experiments in Section 3, yields better results in scenarios with sharp discontinuities.

<table><tr><td rowspan="2">Problem</td><td rowspan="2">Degree</td><td colspan="3">Trig</td><td colspan="3">Fourier</td><td colspan="3">Bell</td><td colspan="3">PWC</td></tr><tr><td> $\overline { { \mathrm { ~ L ~ } ^ { 2 } \downarrow } }$ </td><td>S-Rate↑</td><td>S-Acc↓</td><td>L2↓</td><td>S-Rate↑</td><td>S-Acc↓</td><td>L2↓</td><td>S-Rate↑</td><td>S-Acc↓</td><td> $\overline { { \mathrm { ~ L ~ } ^ { 2 } \downarrow } }$ </td><td>S-Rate↑</td><td>S-Acc↓</td></tr><tr><td>Burrs</td><td>4</td><td>9.27</td><td>16.35</td><td>0.23</td><td>4.52</td><td>24.11</td><td>0.97</td><td>3.32</td><td>73.88</td><td>1.03</td><td>3.80</td><td>82.98</td><td>0.85</td></tr><tr><td></td><td>8</td><td>2.91</td><td>40.22</td><td>0.16</td><td>1.85</td><td>52.76</td><td>0.33</td><td>2.43</td><td>96.43</td><td>0.42</td><td>2.06</td><td>87.51</td><td>0.24</td></tr><tr><td></td><td>16</td><td>1.43</td><td>54.69</td><td>0.16</td><td>0.96</td><td>65.84</td><td>0.14</td><td>1.27</td><td>98.94</td><td>0.18</td><td>1.21</td><td>95.06</td><td>0.23</td></tr><tr><td></td><td>32</td><td>1.80</td><td>57.83</td><td>0.10</td><td>0.72</td><td>94.25</td><td>0.11</td><td>0.82</td><td>99.40</td><td>0.15</td><td>1.28</td><td>95.16</td><td>0.22</td></tr><tr><td rowspan="4">TR</td><td>4</td><td>10.14</td><td>22.26</td><td>0.20</td><td>3.15</td><td>47.77</td><td>0.60</td><td>4.90</td><td>59.72</td><td>0.63</td><td>5.09</td><td>90.04</td><td>0.39</td></tr><tr><td>8</td><td>4.38</td><td>56.08</td><td>0.68</td><td>1.53</td><td>76.42</td><td>0.39</td><td>2.27</td><td>79.71</td><td>0.41</td><td>3.05</td><td>91.70</td><td>0.32</td></tr><tr><td>16</td><td>3.06</td><td>63.93</td><td>0.31</td><td>0.97</td><td>83.69</td><td>0.24</td><td>1.94</td><td>91.43</td><td>0.19</td><td>2.24</td><td>96.73</td><td>0.24</td></tr><tr><td>32</td><td>2.98</td><td>71.58</td><td>0.13</td><td>0.80</td><td>88.71</td><td>0.15</td><td>0.95</td><td>99.21</td><td>0.18</td><td>1.85</td><td>97.78</td><td>0.21</td></tr></table>

Table 6: Performance comparison between diferent maximum degrees of trigonometric test functions in WEPINN across diferent problems and initial conditions under the periodic boundary condition. Reported metrics are the relative $\mathrm { L ^ { 2 } }$ error, the shock detection rate (S-Rate), and the shock position accuracy (S-Acc). Arrows indicate whether lower (↓) or higher (↑) is better. Al numbers are reported in e-2.

<table><tr><td rowspan="2">Problem</td><td rowspan="2">Test function</td><td colspan="3">Riemann</td><td colspan="3">Sigmoid</td></tr><tr><td> $\overline { { \mathrm { ~ L ~ } ^ { 2 } \downarrow } }$ </td><td>S-Rate↑</td><td>↑ S-Acc↓</td><td> $\overline { { \mathrm { ~ L ~ } ^ { 2 } \downarrow } }$ </td><td>S-Rate↑ S-Acc↓</td><td></td></tr><tr><td>Linear</td><td>Cheby Trig</td><td>4.58 0.93</td><td>88.48 100</td><td>1.11 0.03</td><td>5.29 0.87</td><td>- 一</td><td>- -</td></tr><tr><td>Burgers</td><td>Cheby Trig</td><td>44.11 1.84</td><td>77.47 98.58</td><td>1.35 0.05</td><td>46.29 1.53</td><td>73.86 99.41</td><td>1.43 0.04</td></tr><tr><td>LWR</td><td>Cheby Trig</td><td>43.21 1.89</td><td>83.48 93.90</td><td>1.07 0.15</td><td>43.80 2.05</td><td>79.19 96.71</td><td>1.43 0.23</td></tr></table>

Table 7: Performance comparison between trigonometric functions and Chebeshev polynomials as test functions in WEPINN across diferent problems and initial conditions under the periodic boundary condition. Reported metrics are the relative L<sup>2</sup> error, the shock detection rate (S-Rate), and the shock position accuracy (S-Acc). Arrows indicate whether lower (↓) or higher (↑) is better. All numbers are reported in e-2.

To assess the efectiveness of trigonometric test functions, we implemented WEPINN using Chebyshev polynomials of the first kind to construct test functions. For a fair comparison, the maximum degree of the Chebyshev polynomials is set to 32, matching the trigonometric case. Due to the non-periodic nature of Chebyshev polynomials, this comparison is performed under the Dirichlet boundary condition. The results are presented in Table 7.

We observe from Table 7 that using Chebyshev polynomials for test functions results in worse performance than trigonometric functions across all metrics. In particular, the S-Acc degrades more significantly than S-Rate, suggesting that the model with Chebyshev polynomials struggles to track shocks accurately. This may due to the increasing challenge to evaluate integrals in the weak formulation and the entropy condition with Chebeshev polynomials that exhibit rapid oscillations near the domain boundary.

## 5 Conclusion

In this work, we introduce a novel physics-informed learning framework that leverages weak entropy conditions to efectively solve hyperbolic conservation laws, where solutions may develop discontinuities. By rigorously adhering to weak formulation and entropy conditions, and employing the discrete fast Fourier transform (DFFT) for eficient numerical quadrature, our method avoids common pitfalls such as unphysical dissipation and oscillations observed in existing approaches. Our extensive numerical experiments demonstrate that the proposed method consistently outperforms existing approaches across a variety of scalar conservation laws and systems of conservation laws in one and two dimensions. It notably excels in accurately resolving complex nonlinear wave phenomena, including formation and propagation of shocks, shock-shock merging, and shock-rarefaction interaction, under diverse initial and boundary conditions. The enforcement of entropy conditions significantly enhances the reliability of our method, particularly by ensuring physical admissibility of the solution. Future work will explore extending this methodology to design neural operators for solving hyperbolic conservation laws and related PDEs involving discontinuities and moving interfaces.

## Acknowledgments

This work was supported by the National Science Foundation (NSF) under Grant No. CPS-2038984.

## A Evaluation Metrics

To quantitatively evaluate the performance of diferent models in solving hyperbolic conservation laws, we adopt several metrics that assess both global solution accuracy and shock-capturing capability. A reference solution is generated using the Lax–Friedrichs numerical scheme on an ultra-fine grid to serves as the ground truth for all subsequent comparisons.

We compute the relative L<sup>2</sup> error between the predicted solution ˆu and the ground truth u over the entire spatiotemporal domain: Relative $\mathrm { L } ^ { 2 } = \Vert \hat { u } - u \Vert _ { 2 } / \Vert u \Vert _ { 2 }$ .This metric captures the overall discrepancy between the predicted and true solutions.

To evaluate the model’s ability to identify shocks, we define the shock detection rate (S-Rate) as the proportion of ground truth shock locations that are successfully identified by the model within a specified tolerance band. A shock is considered detected, if a high-gradient region in the predicted solution is located within a small neighborhood (10% of spatial domain) of the ground truth shock position.

We also compute the shock position accuracy (S-Acc), defined as the average distance between detected shock locations and their corresponding ground truth positions. This metric captures the spatial precision of shock prediction and penalizes shifts or smearing of the shock front.

The detailed procedure for calculate both newly proposed metric are as shown in Algorithm 1.

Algorithm 1 Computing Shock Detection Rate and Shock Accuracy   
1: function ShockError $( u , \hat { u } , d x , t _ { \mathrm { m a x } } , h )$   
2: Compute gradient: $g _ { u } \gets | \partial _ { x } u | , g _ { \hat { u } } \gets | \partial _ { x } \hat { u } |$   
3: Compute total variation $T V ( t )$ over time   
4: Compute average total variation: $\begin{array} { r } { T V _ { \mathrm { m e a n } } \gets \frac { 1 } { t _ { \mathrm { m a x } } } \sum _ { t } T V ( t ) } \end{array}$   
5: Initialize counters: $D \gets 0 , M \gets 0 , N \gets 0$   
6: for $t = 0$ to $t _ { \mathrm { m a x } } - 1$ do   
7: Detect shocks: $S _ { u } \gets \mathrm { p e a k s } ( g _ { u } [ t ] , \mathrm { h e i g h t } = h \cdot T V _ { \mathrm { m e a n } } )$   
8: Detect shocks: $S _ { \hat { u } } \gets \mathrm { p e a k s } ( g _ { \hat { u } } [ t ] , \mathrm { h e i g h t } = h \cdot T V _ { \mathrm { m e a n } } )$   
9: Compute distance matrix $d _ { i j } = \operatorname* { m i n } ( | S _ { u } [ i ] - S _ { \hat { u } } [ j ] |$ , 128 − |S<sub>u</sub>[i] − S<sub>uˆ</sub>[j]|)   
10: for each i in $S _ { u }$ do   
11: $j ^ { * } \gets \mathrm { a r g }$ min<sub>j</sub> $d _ { i j }$   
12: if $d _ { i j ^ { * } } < d _ { \operatorname* { m a x } }$ then   
13: $D \gets D + d _ { i j ^ { * } }$ ▷ Accumulate distance   
14: $M \gets M + 1$ ▷ Count matched shocks   
15: end if   
16: $N \gets N + 1$ ▷ Count total true shocks   
17: end for   
18: end for   
19: Detection rate: $\mathrm { S - R a t e }  M / N$   
20: Accuracy (distance): $\mathrm { S - A c c }  ( D / M )$ · dx   
21: return S-Rate, S-Acc   
22: end function

## B Initial and Boundary conditions

## B.1 Initial Conditions

To ensure robustness to diverse solution behaviors, we consider multiple classes of initial conditions reflecting varying smoothness and discontinuity.

For Dirichlet boundary conditions, two types of initial conditions are considered:

• Sigmoid-type profiles: Smooth transitions between prescribed left and right boundary states, with adjustable steepness and transition location. These profiles test the model’s ability to resolve sharp but continuous gradients while respecting fixed boundary values.

• Riemann-type profiles: Piecewise-constant initial states composed of two or three constant regions. This setting enables systematic evaluation of shock, rarefaction, and their interactions under Dirichlet boundary conditions, including shock–shock and shock–rarefaction configurations.

For periodic boundary conditions, we consider four types of initial conditions are used:

• Fourier series: Smooth functions constructed from sine and cosine modes with randomly sampled coeficients up to mode 4.

• Trigonometric functions: High-frequency sinusoidal waves (up to frequency 4) with random phase shifts, testing sensitivity to oscillatory dynamics.

• Bell-shaped functions: Smooth, single-peaked profiles with random centers and widths, designed to assess shock formation from localized features.

• Piecewise constant functions: Discontinuous profiles with randomly sampled segment values and breakpoints, modeling shock-like initial states.

Periodic boundary conditions enforce continuity across the domain boundaries,

$$
u ( t , 0 ) = u ( t , 1 ) ,
$$

modeling a spatially repeating (closed-loop) system. This setting eliminates artificial boundary effects and enables clean analysis of wave propagation and interaction, making it particularly suitable for Fourier-based representations and operator learning methods.

## C Implementation Detail

## C.1 Neural Network Architecture

To better capture complex shock interactions, we employ a neural network architecture with residual (skip) connections and ReLU activation. The network uses a hidden dimension of 128 and consists of 9 residual layers. The detailed architecture is presented in Algorithms 2 and 3. For baseline comparisons, we replace the ReLU activation with Tanh, as ReLU can negatively afect the automatic diferentiation required for computing the diferential form loss in PINN.

Algorithm 2 Residual Block (ResBlock)   
1: function ResBlock(x)   
2: residual ← x   
3: x ← fc<sub>1</sub>(x)   
4: x ← activation(x)   
5: x ← fc<sub>2</sub>(x)   
6: return x + residual   
7: end function

Algorithm 3 PINN with Residual Connections (PINNRes)   
1: function PINNRes(x, t)   
2: input ← concat(x, t)   
3: x ← fc<sub>in</sub>(input)   
4: x ← activation(x)   
5: for i = 1 to L do   
6: x ← ResBlock(x)   
7: end for   
8: output ← fc<sub>out</sub>(x)   
9: return output   
10: end function

## C.2 Training

The model is trained using the Adam optimizer with an initial learning rate of $1 \times 1 0 ^ { - 3 }$ . A scheduled learning rate decay is applied, reducing the learning rate by a factor of 0.3 at steps

![](images/fbaf1f3d7a0b5e18282d3081af7923bff13aa68c5da48d2b08ec0e70a91297fa.jpg)

![](images/d522649bc892a7fb73c186788fe98936a71d6cdf8a7b13993cfcaeb792a79c5d.jpg)  
Figure 6: Comparison of evaluations of a Legendre polynomial of degree 32, a Chebyshev polynomial of degree 32, and a trigonometric function of frequency 32 on uniform grid and random collocation points.

10,000 and 20,000. The total number of training steps is 30,000. All reported numbers in the table are average across 30 samples.

## D Compatibility of diferent families of test functions with uniform grids

In prior PINN research that incorporates test functions—such as VPINN or WPINN—Legendre or Chebyshev polynomials are commonly used due to their orthogonality properties. While any complete basis that spans the desired test function space is theoretically suficient, we opt for trigonometric functions instead of polynomials. The key motivation lies in their uniform wavelength and better behavior under discretizations. At higher frequencies, polynomial bases become increasingly dificult to resolve accurately on finite grids, leading to numerical artifacts or instability. In contrast, trigonometric functions maintain consistent oscillatory behavior across frequencies, making them more suitable for capturing fine-scale solution structures. This diference is as illustrated in Figures 6 left. To validate this design choice, we additionally implemented our proposed weak and entropy losses using Chebyshev polynomial test functions for comparison in the section 4.2.

Similarly, we opt for a uniform grid rather than a random grid that is commonly used in prior PINN works, as a uniform grid has a better capacity to capture high frequency test functions, whereas a random grid could miss peaks in the waves of test functions, as shown in Figure 6 right.

## E Eficiency

The main computational bottleneck of the proposed method lies in evaluating $\mathcal { T } _ { t , \mathbf { x } } ^ { \varphi _ { n } }$ and $\mathcal { I } _ { t , { \bf x } } ^ { \varphi _ { n } }$ , which involve summations over the full space-time grid. Here we take one dimensional scalar case as an example:

$$
I _ { t , x } ^ { \varphi } = \sum _ { j = 0 } ^ { n _ { x } } \sum _ { k = 0 } ^ { n _ { t } } \alpha _ { k } \beta _ { j } \big ( u \partial _ { t } \varphi + f ( u ) \partial _ { x } \varphi \big ) ( t _ { k } , x _ { j } ) ,\tag{24}
$$

$$
J _ { t , x } ^ { \varphi } = \sum _ { j = 0 } ^ { n _ { x } } \sum _ { k = 0 } ^ { n _ { t } } \alpha _ { k } \beta _ { j } \bigl ( \eta ( u ) \partial _ { t } \varphi + q ( u ) \partial _ { x } \varphi \bigr ) ( t _ { k } , x _ { j } ) .\tag{25}
$$

Direct evaluation of these summations incurs a computational cost of $O ( N ^ { 2 d } )$ , which becomes prohibitive for fine grids.

## DFFT acceleration.

Given choosing trigonometric test functions on a uniform grid, the above summations can be interpreted as discrete Fourier transforms of the weighted solution. This enables eficient computation using the discrete fast Fourier transform (DFFT).

To illustrate, consider the first term in (24) with a test function

$$
\varphi ( t , x ) = \sin ( 2 \pi n t ) \sin ( 2 \pi m x ) , \quad n , m \in \mathbb { Z } .
$$

Then,

$$
\sum _ { j = 0 } ^ { n _ { x } } \sum _ { k = 0 } ^ { n _ { t } } \alpha _ { k } \beta _ { j } u ( t _ { k } , x _ { j } ) \partial _ { t } \varphi ( t _ { k } , x _ { j } ) = \sum _ { j = 0 } ^ { n _ { x } } \sum _ { k = 0 } ^ { n _ { t } } \alpha _ { k } \beta _ { j } u ( t _ { k } , x _ { j } ) \left( 2 \pi n \cos ( 2 \pi n t _ { k } ) \sin ( 2 \pi m x _ { j } ) \right) .
$$

Using Euler’s identity, the product of trigonometric functions can be expressed as a linear combination of complex exponentials:

$$
\cos ( 2 \pi n t _ { k } ) \sin ( 2 \pi m x _ { j } ) = { \frac { 1 } { 2 } } \Im \left( e ^ { 2 \pi i ( n t _ { k } + m x _ { j } ) } + e ^ { 2 \pi i ( - n t _ { k } + m x _ { j } ) } \right) .
$$

Substituting this into the summation yields

$$
\sum _ { j = 0 } ^ { n _ { x } } \sum _ { k = 0 } ^ { n _ { t } } \alpha _ { k } \beta _ { j } u ( t _ { k } , x _ { j } ) \partial _ { t } \varphi ( t _ { k } , x _ { j } ) = \pi n \sum _ { j = 0 } ^ { n _ { x } } \sum _ { k = 0 } ^ { n _ { t } } \alpha _ { k } \beta _ { j } u ( t _ { k } , x _ { j } ) \mathfrak { H } \left( e ^ { 2 \pi i ( n t _ { k } + m x _ { j } ) } + e ^ { 2 \pi i ( - n t _ { k } + m x _ { j } ) } \right) .
$$

Each term corresponds to a Fourier coeficient of the weighted field $\alpha _ { k } \beta _ { j } u ( t _ { k } , x _ { j } )$ . Therefore, all such terms can be computed simultaneously using DFFT.

## Computational complexity.

Using DFFT, the computational cost is reduced to

$$
O ( n _ { t } n _ { x } \log ( n _ { t } n _ { x } ) ) ,
$$

compared to the quadratic cost $O ( n _ { t } ^ { 2 } n _ { x } ^ { 2 } )$ of direct evaluation. In d spatial dimensions, this generalizes to

$$
O ( N _ { t } N _ { x } ^ { d } \log N _ { t } \log ^ { d } N _ { x } ) ,
$$

which is significantly more eficient than the $O ( N _ { t } ^ { 2 } N _ { x } ^ { 2 d } )$ complexity of standard numerical integration.

## Practical eficiency.

This acceleration is particularly important when using high-frequency test functions, which are necessary to capture fine-scale structures in the solution. Without DFFT, evaluating the weak and entropy residuals becomes both time- and memory-intensive.

We validate the eficiency of the proposed implementation through an ablation study on runtime and memory consumption across diferent grid resolutions, as shown in Figure 7. Due to GPU memory limitations, direct numerical integration could not be evaluated at larger grid sizes.

![](images/7be0b6c9161bbd7c0d3c4532eb2877e8327ef9c9b6b2c78c0817f59e8d2f34ca.jpg)

![](images/152dcee0ae8067181ed8063065c56c9e9ba307a917fc042c4c02053a57d5f1d2.jpg)  
Figure 7: Time and memory eficiency of the proposed method compared against the Dif-PINN and our proposed method with and without DFFT. Left: Runtime (in seconds) for 500 training epochs across varying grid sizes on single Nvidia RTX4090. Right: Total memory consumption by the model during training. The DFFT-based implementation demonstrates superior scalability and resource eficiency.

## References

Stefano Berrone, Claudio Canuto, and Moreno Pintore. Variational physics informed neural networks: the role of quadratures and test functions. Journal of Scientific Computing, 92, 2021. URL https://api.semanticscholar.org/CorpusID:237420756.

Zhiqiang Cai, Jingshuang Chen, and Min Liu. Least-squares relu neural network (lsnn) method for scalar nonlinear hyperbolic conservation law. Applied Numerical Mathematics, 174:163–176, 2022.

Zhiqiang Cai, Jingshuang Chen, and Min Liu. Least-squares neural network (lsnn) method for scalar nonlinear hyperbolic conservation laws: Discrete divergence operator. Journal of Computational and Applied Mathematics, 433:115298, 2023.

Aidan Chaumet and Jan Giesselmann. Improving weak pinns for hyperbolic conservation laws: Dual norm computation, boundary conditions and systems. arXiv preprint arXiv:2211.12393, 2022.

Constantine M Dafermos. Hyperbolic systems of conservation laws. In Systems of nonlinear partial diferential equations, pages 25–70. Springer, 1983.

Constantine M. Dafermos. Hyperbolic Conservation Laws: Past, Present, Future, pages 479– 486. Springer International Publishing, Cham, 2023. ISBN 978-3-031-12244-6. doi: 10.1007/ 978-3-031-12244-6 33. URL https://doi.org/10.1007/978-3-031-12244-6\_33.

Tim De Ryck, Siddhartha Mishra, and Roberto Molinaro. wpinns: Weak physics informed neural networks for approximating entropy solutions of hyperbolic conservation laws. SIAM Journal on Numerical Analysis, 62(2):811–841, 2024.

Mahshid Sadat Ghoreishi and Hamid Naderan. Physics-informed neural network with weighted loss and hard constraints for hyperbolic conservation laws. Scientific Reports, 2026.

Archie J. Huang and Shaurya Agarwal. On the limitations of physics-informed deep learning: Illustrations using first-order hyperbolic conservation law-based trafic flow models. IEEE Open Journal of Intelligent Transportation Systems, 4:279–293, 2023. URL https://api. semanticscholar.org/CorpusID:257205943.

Ameya D Jagtap and George Em Karniadakis. Extended physics-informed neural networks (xpinns): A generalized space-time domain decomposition based deep learning framework for nonlinear partial diferential equations. Communications in Computational Physics, 28(5), 2020.

Ameya D Jagtap, Ehsan Kharazmi, and George Em Karniadakis. Conservative physics-informed neural networks on discrete domains for conservation laws: Applications to forward and inverse problems. Computer Methods in Applied Mechanics and Engineering, 365:113028, 2020.

Ehsan Kharazmi, Zhongqiang Zhang, and George Em Karniadakis. Variational physics-informed neural networks for solving partial diferential equations. arXiv preprint arXiv:1912.00873, 2019.

Ehsan Kharazmi, Zhongqiang Zhang, and George Em Karniadakis. hp-vpinns: Variational physicsinformed neural networks with domain decomposition. Computer Methods in Applied Mechanics and Engineering, 374:113547, 2021.

Stanislav N Kruˇzkov. First order quasilinear equations in several independent variables. Mathematics of the USSR-Sbornik, 10(2):217–243, 1970.

Guoqiang Lei, D Exposito, and Xuerui Mao. Discontinuity-aware kan-based physics-informed neural networks. arXiv preprint arXiv:2507.08338, 2025a.

Guoqiang Lei, Zhihua Wang, Lijing Zhou, D Exposito, and Xuerui Mao. Discontinuity-aware physics-informed neural networks for phase-field method in three-phase flows. arXiv preprint arXiv:2511.23102, 2025b.

Randall J LeVeque. Finite volume methods for hyperbolic problems, volume 31. Cambridge university press, 2002.

Hong Liang, Zilong Song, Chong Zhao, and Xin Bian. Continuous and discontinuous compressible flows in a converging–diverging channel solved by physics-informed neural networks without exogenous data. Scientific Reports, 14(1):3822, 2024.

Michael James Lighthill and Gerald Beresford Whitham. On kinematic waves ii. a theory of trafic flow on long crowded roads. Proceedings of the royal society of london. series a. mathematical and physical sciences, 229(1178):317–345, 1955.

Li Liu, Shengping Liu, Hui Xie, Fansheng Xiong, Tengchao Yu, Mengjuan Xiao, Lufeng Liu, and Heng Yong. Discontinuity computing using physics-informed neural networks. Journal of Scientific Computing, 98(1):22, 2024.

Emmanuel Lorin and Arian Novruzi. Non-difusive neural network method for hyperbolic conservation laws. Journal of Computational Physics, 513:113161, 2024.

Maziar Raissi, Paris Perdikaris, and George E Karniadakis. Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial diferential equations. Journal of Computational physics, 378:686–707, 2019.

Paul I. Richards. Shock waves on the highway. Operations Res., 4:42–51, 1956. ISSN 0030-364X. doi: 10.1287/opre.4.1.42. URL https://doi.org/10.1287/opre.4.1.42.

Sergio Rojas, Pawel Maczuga, Judit Mu˜noz-Matute, David Pardo, and Maciej Paszy´nski. Robust variational physics-informed neural networks. ArXiv, abs/2308.16910, 2023. URL https://api. semanticscholar.org/CorpusID:261394719.

Tim De Ryck and Siddhartha Mishra. Numerical analysis of physics-informed neural networks and related models in physics-informed machine learning. Acta Numerica, 33:633 – 713, 2024. URL https://api.semanticscholar.org/CorpusID:267750765.

Hua Su, Lei Zhang, and Jin Zhao. Spike: Stable physics-informed kernel evolution method for solving hyperbolic conservation laws. arXiv preprint arXiv:2510.18266, 2025.

Qi Sun, Zhenjiang Liu, Lili Ju, and Xuejun Xu. Lift-and-embed learning methods for solving scalar hyperbolic equations with discontinuous solutions. arXiv preprint arXiv:2411.05382, 2024.

Chuanxing Wang, Hui Luo, Kai Wang, Guohuai Zhu, and Mingxing Luo. Solving euler equations with multiple discontinuities via separation-transfer physics-informed neural networks. arXiv preprint arXiv:2505.20361, 2025.

Weiheng Zeng, Kun Wang, Ruoxi Lu, and Tiegang Liu. Clinn: Conservation law informed neural network for approximating discontinuous solutions. arXiv preprint arXiv:2509.02091, 2025.

Tong Zhang and Yu Xi Zheng. Two-dimensional riemann problem for a single conservation law. Transactions of the American Mathematical Society, 312(2):589–619, 1989.

Xiaoping Zhang, Tao Cheng, and Lili Ju. Implicit form neural network for learning scalar hyperbolic conservation laws. In Mathematical and Scientific Machine Learning, pages 1082–1098. PMLR, 2022.

Hanfei Zhou and Lei Shi. Weak physics informed neural networks for geometry compatible hyperbolic conservation laws on manifolds. arXiv preprint arXiv:2505.19036, 2025.