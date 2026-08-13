# TESLA: Taylor Expansion of Sinusoidal Learnable Activations

Daehwa Ko

Jaehyeon Kim Seunghyun Ham Korea Aerospace University, Goyang, Republic of Korea

Jay Hoon Jung

## Abstract

The parity problem—deciding whether the number of ones in a binary vector is odd or even—remains challenging for standard neural networks due to linear inseparability and the need for global interactions. We propose TESLA, an activation defined as a learnable combination of sine and cosine terms, enabling explicit control over polynomial degree and selective amplification of high-order components. Theoretically, we show that constraining TESLA’s coeficients yields Lipschitz/Rademacher complexity bounds and shapes the training dynamics to emphasize higher-frequency structure. Empirically, on parity with input length n = 32, TESLA attains strong generalization with 100K training samples (≈ 0.002% of the 2<sup>32</sup> input space) and remains robust under heavy corruption, retaining high accuracy with up to 30% label noise. We also compare against periodic and frequency-based baselines (SIREN, SNAKE, and Fourier feature embeddings) on parity and Forrelation. Beyond synthetic structure, TESLA delivers comparable performance on ImageNet-100, indicating that activationlevel degree control transfers to more general vision workloads. Code: https://github. com/KAU-QuantumAILab/TESLA

![](images/0acb2b3088a703f0470378f8ab00cb4c09db42bca0e1327bd55a6da75910a618.jpg)  
Figure 1: Visualization of the decision boundaries for TESLA(K = 8) and baseline activations on a 2D feature space. The solid lines separate the predicted classes.

## 1 INTRODUCTION

Modern neural networks largely build complex functions by stacking local nonlinearities such as ReLU (Nair and Hinton, 2010) and GeLU (Hendrycks and Gimpel, 2016) with linear maps. These designs excel at capturing local structure in images and language, but they can be ineficient for global or longrange interaction tasks that depend on high-order combinations of many input coordinates, global parity/phase, or other global symmetries (Daniely and Malach, 2020). Empirically and theoretically, traditional activation functions and vanilla MLPs exhibit a spectral bias toward low-frequency (low-degree) components, so representing strong high-order or global structure often requires substantially greater depth, width, or very large weight norms (Rahaman et al., 2019; Zhi-Qin John Xu et al., 2020; Tancik et al., 2020;

Sitzmann et al., 2020).

To enable direct and learnable control over the polynomial order at the activation level, we propose a simple parametric activation function based on a finite sine and cosine basis. This formulation allows the network to learn Fourier-like coeficients that explicitly control the efective polynomial order of activations. We refer to this activation as Taylor Expansion of Sinusoidal Learnable Activation (TESLA). Our main contributions are as follows:

• We derive tight Lipschitz bounds, establish Rademacher complexity–based generalization guarantees, and characterize learning dynamics, showing that TESLA’s inductive bias naturally favors recovery of low- and mid-frequency structure.

• We present constructive approximation results for ridge- and interaction-type functions, together with complementary lower bounds showing that standard piecewise-linear activations require substantially more resources to match TESLA’s ability to represent oscillatory structure.

• On synthetic benchmarks, TESLA generalizes a 32-bit parity task with 100K training samples and remains robust under label noise up to 30%. It also improves performance on Forrelation and LPN tasks and remains usable in ImageNet-100- scale models while incurring negligible throughput overhead (≈ 1%).

## 2 RELATED WORK

Recent research has explored various ways to enrich neural network representations through activation design. Several works, such as PReLU (He et al., 2015) and APL (Agostinelli et al., 2015), introduce parametric nonlinear activations with a small number of learnable parameters to improve expressiveness and facilitate optimization. While these methods directly modify the pointwise nonlinearity, they do not provide explicit mechanisms for controlling polynomial or spectral degree. In contrast, approaches that learn per-edge or per-unit activations—such as KANs (Liu et al., 2024) and spline-based methods (Apicella et al., 2021; Scardapane et al., 2019)—ofer substantially greater expressivity and can closely approximate complex mathematical or scientific functions. However, prior studies and surveys have noted that this increased flexibility can lead to overfitting or instability in low-data regimes unless it is carefully regulated through techniques such as regularization or parameter sharing.

Building on these insights, a complementary line of work employs sinusoidal and Fourier-based parameterizations to address the spectral limitations of standard activations and embeddings. These approaches aim to expand the spectral capacity of neural networks and reshape their training dynamics, motivating the development of periodic activations and Fourier feature mappings. Periodic activations (e.g., SIREN (Sitzmann et al., 2020)) and positional or Fourier feature encodings (Tancik et al., 2020; Rahimi and Recht, 2007) mitigate spectral bias by introducing high-frequency components into the network—either by expanding its efective spectral support or by modifying the spectrum of the initial Neural Tangent Kernel (NTK, (Jacot et al., 2018)). The NTK is a theoretical construct that characterizes the training dynamics of infinitely wide neural networks, showing that such networks evolve like linear models governed by a fixed kernel, which in turn guarantees convergence to a global minimum.

In parallel, a substantial body of theoretical work has emerged to analyze spectral bias (or the frequency principle) and NTK dynamics, aiming to explain why standard networks prioritize low-frequency components and how modifying kernels or embeddings can alter this behavior (Rahaman et al., 2019; Zhi Qin John Xu et al., 2020; Jacot et al., 2018). Classi cal approximation-theoretic results—such as Barrontype bounds (Barron, 1993, 1994), depth–width separation theorems (Eldan and Shamir, 2016; Telgarsky, 2016), and lower bounds for piecewise-linear approximations (Yarotsky, 2017; Telgarsky, 2016)—provide a rigorous framework for comparing diferent param eterizations. These results motivate our constructive ridge-approximation upper bounds and lower-bound sketches, which characterize when piecewise-linear activations incur high resource costs to emulate oscillatory global features (Barron, 1993; Telgarsky, 2016).

## 3 PROPOSED METHOD

## 3.1 Activation Function Definition

We define a parametric activation function, shared across all neurons within a layer, built from a finite sine and cosine basis:

$$
\phi _ { K } ( z ) = \sum _ { k = 1 } ^ { K } \left( \frac { a _ { k } } { k } \sin ( k z ) + \frac { b _ { k } } { k } \cos ( k z ) \right) ,\tag{1}
$$

where $K \in \mathbb N$ controls the maximum frequency, and $\{ a _ { k } , b _ { k } \} _ { k = 1 } ^ { K }$ are learnable scalar coeficients. For a neuron receiving input $z \ = \ v ^ { \top } x$ , the neuron output is $f ( x ) = \phi _ { K } ( v ^ { \top } x )$ . We also define the coeficient bud-

get:

$$
A _ { K } : = \sum _ { k = 1 } ^ { K } { \big ( } | a _ { k } | + | b _ { k } | { \big ) } ,\tag{2}
$$

which appears in stability and complexity bounds.

## 3.2 Taylor (Maclaurin) Expansion and Polynomial Coeficients

Expanding ϕ<sub>K</sub> via the Taylor series of sine and cosine gives:

$$
\begin{array} { r l r } & { } & { \phi _ { K } ( z ) = \displaystyle \sum _ { k = 1 } ^ { K } \frac { b _ { k } } { k } + \sum _ { m = 0 } ^ { \infty } \left( \frac { ( - 1 ) ^ { m } } { ( 2 m + 1 ) ! } \sum _ { k = 1 } ^ { K } a _ { k } k ^ { 2 m } \right) z ^ { 2 m + 1 } } \\ & { } & { + \sum _ { m = 1 } ^ { \infty } \left( \frac { ( - 1 ) ^ { m } } { ( 2 m ) ! } \sum _ { k = 1 } ^ { K } b _ { k } k ^ { 2 m - 1 } \right) z ^ { 2 m } . \qquad ( 3 ) } \end{array}
$$

Eq. (3) shows that each polynomial order is an explicit linear functional of the learned coeficients $\{ a _ { k } , b _ { k } \}$ giving direct control over degree components at the activation level.

Interpretation vs. Implementation. TESLA is implemented as the finite trigonometric expansion in Eq. (1); we do not evaluate a polynomial series during the forward pass. We use Eq. (3) as an interpretation that links learned trigonometric coeficients to efective polynomial degree and motivates the coeficient budget in our analysis.

Input Spectralization vs. Activation Spectralization. Input-side spectralization (e.g., Fourier features; Tancik et al., 2020) maps inputs to highfrequency coordinates and leaves mode selection to later layers. TESLA instead spectralizes the activation itself: coeficients $\left\{ a _ { k } , b _ { k } \right\}$ directly shape Maclaurinorder terms, enabling selective amplification or attenuation of target orders.

In practice, this design reduces the need for handcrafted frequency mappings at the input stage and moves spectral control into a small set of learnable activation coeficients. As a result, practitioners can tune inductive bias with fewer architectural changes while preserving compatibility with standard training pipelines. This makes TESLA particularly attractive when transferring across tasks with diferent frequency characteristics.

## 4 THEORETICAL ANALYSIS

In this section, we provide a concise theoretical foundation for TESLA. We first bound derivatives and Lipschitz constants to show how the per-harmonic $\ell _ { 1 }$ budget $A _ { K }$ controls gradient magnitudes. We then present

Rademacher-complexity generalization bounds, analyze NTK-style mode-wise learning dynamics, and derive a practical rule for selecting K as a function of the efective order $m _ { \mathrm { e f f } }$ , sample size $N _ { : }$ , and budget $A _ { K }$

## 4.1 Derivative Bound and Stability

Diferentiation of Eq. (1) gives:

$$
\phi _ { K } ^ { \prime } ( z ) \ = \ \sum _ { k = 1 } ^ { K } { \big ( } a _ { k } \cos ( k z ) - b _ { k } \sin ( k z ) { \big ) } .
$$

Since $| \sin ( \cdot ) | ~ \leq ~ 1$ and $| \cos ( \cdot ) | ~ \le ~ 1$ , each term is bounded by its amplitude, $| a _ { k } \cos ( k z ) - b _ { k } \sin ( k z ) | \le$ $\sqrt { a _ { k } ^ { 2 } + b _ { k } ^ { 2 } }$ ; therefore,

$$
\| \phi _ { K } ^ { \prime } \| _ { \infty } \leq \sum _ { k = 1 } ^ { K } { \sqrt { a _ { k } ^ { 2 } + b _ { k } ^ { 2 } } } \leq \sum _ { k = 1 } ^ { K } ( | a _ { k } | + | b _ { k } | ) = A _ { K } .
$$

Composition Lipschitz Bounds. Let $\begin{array} { r l } { h ( x ) } & { { } = } \end{array}$ $\phi _ { K } ( W x )$ , where $\phi _ { K }$ is applied elementwise. By the chain rule,

$$
\mathrm { L i p } ( h ) \leq \| \phi _ { K } ^ { \prime } \| _ { \infty } \| W \| _ { \mathrm { o p } } \leq A _ { K } \| W \| _ { \mathrm { o p } } .
$$

Here, $\| \cdot \| _ { \mathrm { o p } }$ denotes the spectral norm, and $\| \cdot \| _ { \infty }$ denotes the $L _ { \infty }$ norm.

Network-level Lipschitz and Degree Control. For an L-layer network

$$
f ( x ) = W _ { L } \phi _ { K } ( \cdot \cdot \cdot \phi _ { K } ( W _ { 1 } x ) ) , \qquad \| \phi _ { K } ^ { \prime } \| _ { \infty } \leq A _ { K } ,
$$

the Lipschitz constant satisfies

$$
\mathrm { L i p } ( f ) \leq \left( \prod _ { \ell = 1 } ^ { L } \| W _ { \ell } \| _ { \mathrm { o p } } \right) A _ { K } ^ { L - 1 } .
$$

Beyond Lipschitz control, the same composition also governs the growth of polynomial complexity: under the Maclaurin-coeficient view induced by Eq. (3), composing $L - 1$ TESLA layers activates terms up to order $O ( \bar { K } ^ { L - 1 } )$ . Moreover, the $\ell _ { 1 }$ norm of the corresponding coeficient vector is bounded as

$$
\| \mathrm { c o e f } ( f ) \| _ { 1 } \le A _ { K } ^ { L - 1 } \prod _ { \ell = 1 } ^ { L } \| W _ { \ell } \| _ { 1 \to 1 } .
$$

Thus, a single per-layer budget $A _ { K }$ stabilizes gradients, while $K$ controls the accessible frequency range. In particular, for an $L _ { \mathrm { l o s s } ^ { - 1 } }$ ipschitz loss ${ \mathcal { L } } ,$

$$
\| \nabla _ { W _ { \ell } } \mathcal { L } ( f ( x ) , y ) \| _ { F } \leq L _ { \mathrm { l o s s } } \| x \| _ { 2 } \left( \prod _ { j = 1 } ^ { L } \| W _ { j } \| _ { \mathrm { o p } } \right) A _ { K } ^ { L - 1 } .
$$

This highlights the trade-of: increasing K expands representable high frequencies, while controlling $A _ { K }$ keeps optimization stable.

## 4.2 Rademacher Complexity and Generalization

Theorem 1. Assume $\| x _ { i } \| _ { 2 } \leq R$ for all $i = 1 , \ldots , N .$ Define

$$
{ \mathcal { F } } _ { K } = \{ x \mapsto \phi _ { K } ( v ^ { \top } x ) ~ | ~ \| v \| _ { 2 } \leq W , ~ A _ { K } \leq A \} ,
$$

and

$$
B _ { 0 } ( A ) : = \operatorname* { s u p } _ { \phi _ { K } : A _ { K } \leq A } | \phi _ { K } ( 0 ) | .
$$

Then, for any sample of size N, the empirical Rademacher complexity satisfies

$$
{ \hat { \mathfrak { R } } } _ { N } ( { \mathcal { F } } _ { K } ) \leq { \frac { A W R } { \sqrt { N } } } + { \frac { B _ { 0 } ( A ) } { \sqrt { N } } } = { \frac { A W R + B _ { 0 } ( A ) } { \sqrt { N } } } .
$$

Consequently, for any L<sub>loss</sub>-Lipschitz loss L, the generalization gap scales as $O \Big ( L _ { \mathrm { l o s s } } ( A W R + B _ { 0 } ( A ) ) / \sqrt { N } \Big )$

The bound recovers the familiar linear-class rate $\Theta ( W R / \sqrt { N } )$ up to the multiplicative factor A due to the activation Lipschitz constant. In deep architectures, the same efect accumulates multiplicatively via layer-wise operator norms and activation budgets.

Proof. Let

$$
{ \mathcal { F } } _ { \mathrm { l i n } } : = \{ x \mapsto v ^ { \top } x : \| v \| _ { 2 } \leq W \} .
$$

By a standard argument,

$$
\hat { \mathfrak { R } } _ { N } ( \mathcal { F } _ { \mathrm { l i n } } ) = \frac { 1 } { N } \mathbb { E } \bigg [ \operatorname* { s u p } _ { \| v \| _ { 2 } \leq W } \sum _ { i = 1 } ^ { N } \sigma _ { i } v ^ { \top } x _ { i } \bigg ] \leq \frac { W R } { \sqrt { N } } .
$$

For any $\phi _ { K }$ with $A _ { K } \ \leq \ A _ { \mathsf { i } }$ write $\phi _ { K } ( t ) = \tilde { \phi } _ { K } ( t ) +$ $\phi _ { K } ( 0 )$ with $\tilde { \phi } _ { K } ( 0 ) = 0$ and $\mathrm { L i p } ( \tilde { \phi } _ { K } ) \leq A$ . Also, since $\begin{array} { r } { \phi _ { K } ( 0 ) = \sum _ { k = 1 } ^ { K } \frac { b _ { k } } { k } } \end{array}$ and $A _ { K } \le A$ , we have $B _ { 0 } ( A ) \leq A$ By the Ledoux–Talagrand contraction lemma (Ledoux and Talagrand, 2013),

$$
\hat { \Re } _ { N } \big ( \{ \tilde { \phi } _ { K } ( v ^ { \top } x ) \} \big ) \leq A \hat { \Re } _ { N } ( \mathcal { F } _ { \mathrm { l i n } } ) \leq \frac { A W R } { \sqrt { N } } .
$$

For the constants,

$$
\begin{array} { r l r } {  { \operatorname* { s u p } _ { | c | \leq B _ { 0 } ( A ) } \frac { 1 } { N } \mathbb { E } \Big | \displaystyle \sum _ { i = 1 } ^ { N } \sigma _ { i } c \Big | = \frac { B _ { 0 } ( A ) } { N } \mathbb { E } \Big | \displaystyle \sum _ { i = 1 } ^ { N } \sigma _ { i } \Big | } } \\ & { } & { \leq \frac { B _ { 0 } ( A ) } { N } \sqrt { \mathbb { E } \Big ( \displaystyle \sum _ { i = 1 } ^ { N } \sigma _ { i } \Big ) ^ { 2 } } } \\ & { } & { = \frac { B _ { 0 } ( A ) } { \sqrt { N } } . } \end{array}
$$

Combine the two parts to get the stated bound.

![](images/d5a6e26171d275185efc16941eb7120e611047d608667a575de311a17f9b8eb1.jpg)  
Figure 2: Mode-wise empirical spectra: normalized eigenvalues $\tilde { \lambda } _ { m } = \lambda _ { m } / \lambda _ { 1 }$ vs. mode index m. Lines: TESLA analytic $\left( { \frac { 1 } { 2 m ^ { 2 } } } \right)$ , TESLA empirical, RF-ReLU, RF-GeLU. RF denotes finite random-feature approximations of the infinite-width ReLU/GeLU kernels.

## 4.3 Mode-Wise Learning Dynamics

We analyze learning dynamics on the circle $\mathbb { T } = [ 0 , 2 \pi )$ in the Fourier basis, where a mode refers to a trigonometric component (e.g., sin(mt) or cos(mt)) with frequency index m. We equip T with the inner product

$$
\langle f , g \rangle = { \frac { 1 } { 2 \pi } } \int _ { 0 } ^ { 2 \pi } f ( t ) g ( t ) d t .
$$

Under squared loss, the linearized gradient flow around initialization is

$$
\partial _ { \tau } ( f _ { \tau } - f ^ { \star } ) = - \eta K _ { \phi } ( f _ { \tau } - f ^ { \star } ) ,
$$

where $\eta > 0$ is the learning-rate scale and $K _ { \phi }$ is the kernel operator

$$
\begin{array} { r l } & { K _ { \phi } ( t , t ^ { \prime } ) = \nabla _ { \theta } f _ { \theta } ( t ) ^ { \top } \nabla _ { \theta } f _ { \theta } ( t ^ { \prime } ) , } \\ & { ( K _ { \phi } g ) ( t ) = \displaystyle \int _ { 0 } ^ { 2 \pi } K _ { \phi } ( t , t ^ { \prime } ) g ( t ^ { \prime } ) \frac { d t ^ { \prime } } { 2 \pi } . } \end{array}
$$

With TESLA coeficients $\{ a _ { k } , b _ { k } \} _ { k = 1 } ^ { K }$ , the Jacobian features satisfy

$$
\frac { \partial f _ { \theta } } { \partial a _ { k } } ( t ) = \frac { 1 } { k } \sin ( k t ) , \qquad \frac { \partial f _ { \theta } } { \partial b _ { k } } ( t ) = \frac { 1 } { k } \cos ( k t ) ,
$$

so $K _ { \phi }$ is translation-invariant with kernel

$$
K _ { \phi } ( t , t ^ { \prime } ) = \sum _ { k = 1 } ^ { K } \frac { 1 } { k ^ { 2 } } \cos \bigl ( k ( t - t ^ { \prime } ) \bigr ) .
$$

The orthonormal Fourier basis $\psi _ { m , s } ( t ) = \sqrt { 2 } \sin ( m t )$ and $\psi _ { m , c } ( t ) = \sqrt { 2 } \cos ( m t )$ diagonalizes $K _ { \phi } \colon$

$$
\begin{array} { c } { { K _ { \phi } [ { \psi _ { m , a } } ] = \lambda _ { m } ^ { ( \phi ) } \psi _ { m , a } , } } \\ { { \lambda _ { m } ^ { ( \phi ) } = \frac { 1 } { 2 m ^ { 2 } } , \quad a \in \{ s , c \} , ~ 1 \leq m \leq K . } } \end{array}
$$

![](images/9ec9b82aeb6dc608b560ec14f76e2480b09be4c9aa94ada139353cedb2fb6e63.jpg)  
Figure 3: Validation accuracy (%) for the parity task. Each panel shows a diferent noise condition: Clean, LPN (noise level = 10%) and noisy-parity with 5%, 10%, 20%, 30% label noise. While most baseline activations remain near chance (≈ 50%), TESLA maintains substantially higher accuracy across bit-lengths and noise levels.

and $\lambda _ { m } ^ { ( \phi ) } = 0$ for $m > K$ . Writing the modal coeficients

$$
e _ { m , a } ( \tau ) = \langle f _ { \tau } - f ^ { \star } , \psi _ { m , a } \rangle ,
$$

the dynamics decouple as

$$
\begin{array} { l } { { \displaystyle e _ { m , a } ( \tau ) = e _ { m , a } ( 0 ) \exp \{ - \eta \lambda _ { m } ^ { ( \phi ) } \tau \} , } } \\ { { \displaystyle \| f _ { \tau } - f ^ { \star } \| _ { L ^ { 2 } } ^ { 2 } = \sum _ { m , a } e _ { m , a } ( \tau ) ^ { 2 } . } } \end{array}
$$

Thus, larger $\lambda _ { m } ^ { ( \phi ) }$ yields faster decay of the corresponding mode. Analytically, TESLA gives $\lambda _ { m } ^ { ( \phi ) } \propto m ^ { - 2 } ;$ empirically (Fig. 2), its spectrum is flatter than RF proxies, allocating relatively larger eigenvalues to medium– high modes and accelerating recovery of higher-order structure.

## 4.4 Theory-Guided Choice of the Harmonic Count K

We give a compact, operational rule for how the harmonic count K should scale with task complexity and sample size by balancing approximation and estimation. Throughout this subsection, N denotes the training sample size and $f ^ { \star }$ the target. For approximation statements, we consider Boolean targets of the form $f ^ { \star } : \{ 0 , 1 \} ^ { d } \to \{ \pm 1 \}$ , while using a continuous periodic proxy $f ^ { \star } : \mathbb { T } $ R to interpret mode-wise NTK behavior. Constants $c , C , C _ { 0 } , . . .$ . are absolute and may change value between displays.

Task Complexity via Efective Order. Write the Walsh expansion (Walsh, 1923)

$$
f ^ { \star } ( x ) = \sum _ { T \subseteq [ d ] } { \hat { f } } _ { T } \chi _ { T } ( x ) .
$$

For $\varepsilon \in ( 0 , \frac { 1 } { 2 } )$ , define the efective interaction order

$$
m _ { \mathrm { e f f } } : = \operatorname* { i n f } { \Big \{ } m \in \{ 0 , \ldots , d \} : \sum _ { | T | \leq m } { \hat { f } } _ { T } ^ { 2 } \geq 1 - \varepsilon { \Big \} } .
$$

$m _ { \mathrm { e f f } }$ is the interaction order needed to capture most of the target energy (e.g., parity of order m has $m _ { \mathrm { e f f } } =$ m).

Approximation Behavior of K-harmonic Activations. Using Eq. (1), let the atoms $\left\{ \sigma _ { k } \right\}$ be normalized by $\| \sigma _ { k } \| _ { \infty } \leq 1$ with $1 / k$ scaling. Assume the best-in-class $L ^ { 2 }$ approximation error over $\mathcal { F } _ { K }$ satisfies

$$
\begin{array} { r l r } & { } & { \underset { f \in \mathcal { F } _ { K } } { \operatorname* { i n f } } \ \mathbb { E } \big [ ( f ( x ) - f ^ { \star } ( x ) ) ^ { 2 } \big ] \ \leq \ B \big ( \frac { K } { m _ { \mathrm { e f f } } } \big ) , } \\ & { } & { B ( z ) \in \{ c _ { 1 } e ^ { - \gamma z } , \ c _ { 1 } z ^ { - p } \} , } \end{array}\tag{4}
$$

for some $c _ { 1 } , \gamma , p > 0 ;$ i.e., once $K = \Theta ( m _ { \mathrm { e f f } } )$ , the approximation error decays monotonically.

Estimation Term for ℓ<sub>1</sub>-Budgeted Mixtures. Let $\hat { \Re } _ { N }$ denote empirical Rademacher complexity on N samples. By the $\ell _ { 1 }$ -budgeted mixture bound (budget $A _ { K } )$

$$
\Re _ { N } ( \mathcal { F } _ { K } ) \ \leq \ C _ { 0 } \frac { A _ { K } } { \sqrt { N } } \sqrt { \log ( 2 K ) } .\tag{5}
$$

![](images/495a9c2b682a70aa8068ada028c4b43bad7970453544e173684756fc0136f6e3.jpg)  
Figure 4: Validation accuracy (%) for the Forrelation task with d = 12 using a two-layer MLP. Each panel shows a diferent hidden size $H \in \{ 1 2 8 , 5 1 2 , 1 0 2 4 \}$ . Curves compare TESLA with $K \in \{ 1 , 2 , 4 , 1 6 \}$ against standard activations. TESLA converges faster and achieves higher accuracy on this global-correlation task.

Consequently, for any 1-Lipschitz surrogate loss ${ \mathcal { L } } ,$ with probability at least $1 - \delta .$

$$
\begin{array} { r l r } {  { \operatorname* { s u p } _ { f \in \mathcal { F } _ { K } } \Big | \mathbb { E } [ \mathcal { L } ( f ( x ) , y ) ] - \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \mathcal { L } ( f ( x _ { i } ) , y _ { i } ) \Big | } } \\ & { } & { \leq \ C _ { 1 } \frac { A _ { K } } { \sqrt { N } } \sqrt { \log ( 2 K ) } + C _ { 2 } \sqrt { \frac { \log ( 1 / \delta ) } { N } } . } \end{array}\tag{6}
$$

Excess-risk Bound and Optimal Scale. Let $\hat { f } _ { K } \in \mathcal { F } _ { K }$ minimize empirical surrogate risk. Com bining Eq. (4)–(6) yields

$$
\begin{array} { r } { \mathcal { E } ( \hat { f } _ { K } ) \lesssim \underbrace { B \Big ( \frac { K } { m _ { \mathrm { e f f } } } \Big ) } _ { \mathrm { a p p r o x i m a t i o n } } + \underbrace { C \frac { A _ { K } } { \sqrt { N } } \sqrt { \log ( 2 K ) } } _ { \mathrm { e s t i m a t i o n } } + C \sqrt { \frac { \log ( 1 / \delta ) } { N } } . } \end{array}\tag{7}
$$

Balancing the first two terms gives

$$
K ^ { \star } ~ = ~ \Theta \left( m _ { \mathrm { e f f } } \right) \cdot \kappa ( N , A _ { K } ) ,
$$

where κ varies slowly with N and $A _ { K } ;$ : for $B ( z ) =$ $c _ { 1 } e ^ { - \gamma z } , ~ \kappa ~ = ~ \Theta ( \log N )$ ; for $B ( z ) ~ = ~ c _ { 1 } z ^ { - p } , ~ \kappa ~ = ~$ $\Theta \big ( N ^ { 1 / ( 2 p ) } ( \log N ) ^ { 1 / ( 2 \bar { p } ) } \big )$ up to constants. Intuitively, taking K much larger than $m _ { \mathrm { e f f } }$ dilutes the perharmonic budget (average amplitude $\sim A _ { K } / K )$ and increases estimation error.

In d-bit Parity, $m _ { \mathrm { e f f } } = d ,$ hence the theory predicts $K ^ { \star } ~ = ~ \Theta ( d )$ up to slowly varying factors. Empirically (Sec. 5), we find a stable range $\kappa \in [ 0 . 2 , 0 . 4 ]$ at $N = 1 0 ^ { 5 }$ , placing the optimum near $K \approx d / 4 .$ This aligns with the NTK view: the analytic TESLA kernel on T has eigenvalues $\begin{array} { r } { \lambda _ { m } ^ { ( \phi ) } = \frac { 1 } { 2 m ^ { 2 } } } \end{array}$ , but finite-feature and finite-sample efects flatten the empirical spectrum and emphasize medium–high modes. Thus, once $K \gtrsim m _ { \mathrm { e f f } } ,$ marginal approximation gains are ofset by the estimation term in Eq. (7).

## 5 EXPERIMENTS

## 5.1 Comparisons with Periodic and Frequency-based Baselines

To provide a balanced comparison with periodic and Fourier-style activations, we include SIREN, SNAKE, and a Fourier-feature embedding baseline in both parity and Forrelation experiments, using matched architectures and training protocols unless otherwise noted. For the Fourier-feature embedding baseline, we use a 64-dimensional input mapping with sin $. ( 2 ^ { k } \pi x )$ and cos $( 2 ^ { k } \pi x )$ for $k = 0 { : } 3 1$ , followed by the same MLP architecture with ReLU.

## 5.2 Continuous-domain Evaluation

Sinusoidal activations are often efective in continuousdomain learning problems. We therefore evaluate TESLA on three representative settings: (i) physics-informed neural networks (PINNs) for PDE solving, (ii) implicit neural representations (INRs) for coordinate-based image reconstruction, and (iii) mixed-frequency regression for explicit high-frequency signal modeling.

Physics-informed neural networks (PINNs). We train a PINN for the 1D viscous Burgers’ equation. The total loss is

$$
\mathcal { L } _ { \mathrm { t o t a l } } = \mathcal { L } _ { \mathrm { P D E } } + 1 0 0 \mathcal { L } _ { \mathrm { I C } } + 1 0 0 \mathcal { L } _ { \mathrm { B C } } ,\tag{8}
$$

where $\mathcal { L } _ { \mathrm { P D E } } , ~ \mathcal { L } _ { \mathrm { I C } }$ , and $\mathcal { L } _ { \mathrm { B C } }$ are mean-squared errors of the PDE residual, initial condition, and boundary conditions, respectively.

As shown in Table 1, TESLA achieves the lowest final loss on the Burgers’ equation PINN among the compared activations.

Table 1: 1D Burgers’ equation PINN: final losses after 20,000 epochs (MSE).
<table><tr><td>Activation</td><td> $\mathcal { L } _ { \mathrm { t o t a l } }$ </td><td> $\mathcal { L } _ { \mathrm { P D E } }$ </td><td> $\mathcal { L } _ { \mathrm { I C } }$ </td><td> $\mathcal { L } _ { \mathrm { B C } }$ </td></tr><tr><td>ReLU</td><td> $7 . 1 3 \times 1 0 ^ { - 1 }$ </td><td> $5 . 5 2 \times 1 0 ^ { - 1 }$ </td><td> $1 . 3 1 \times 1 0 ^ { - 3 }$ </td><td> $3 . 0 3 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Tanh</td><td> $1 . 1 9 \times 1 0 ^ { - 2 }$ </td><td> $3 . 8 1 \times 1 0 ^ { - 3 }$ </td><td> $2 . 9 3 \times 1 0 ^ { - 6 }$ </td><td> $7 . 7 7 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>SIREN</td><td> $1 . 0 4 \times 1 0 ^ { - 2 }$ </td><td> $6 . 0 0 \times 1 0 ^ { - 3 }$ </td><td> $1 . 0 6 \times 1 0 ^ { - 5 }$ </td><td> $3 . 3 8 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>TESLA</td><td> $\mathbf { 1 . 2 1 \times 1 0 ^ { - 3 } }$ </td><td> $\mathbf { 5 . 0 1 \times 1 0 ^ { - 4 } }$ </td><td> $\mathbf { 1 . 1 1 \times 1 0 ^ { - 6 } }$ </td><td> $\mathbf { 6 . 0 0 \times 1 0 ^ { - 6 } }$ </td></tr></table>

Implicit neural representations (INRs). We evaluate INRs on Kodak24 and DIV2K, where a 6- layer MLP (512 hidden units) is trained per image for 3000 epochs under an identical training budget.

Table 2: Implicit neural representations on Kodak24 and DIV2K. We report reconstruction quality (PSNR/SSIM) and the number of epochs needed to reach 25 dB (Epochs@25dB).
<table><tr><td>Dataset</td><td>Activation</td><td>PSNR ↑</td><td>SSIM ↑</td><td>Epochs@25dB↓</td></tr><tr><td rowspan="4">Kodak24</td><td>TESLA</td><td>27.72</td><td>0.9671</td><td>1595.8</td></tr><tr><td>SIREN</td><td>25.33</td><td>0.9404</td><td>1887.5</td></tr><tr><td>ReLU</td><td>21.22</td><td>0.8389</td><td>3000.0</td></tr><tr><td>Tanh</td><td>20.17</td><td>0.7818</td><td>3000.0</td></tr><tr><td rowspan="4">DIV2K</td><td>TESLA</td><td>24.73</td><td>0.9522</td><td>2485.0</td></tr><tr><td>SIREN</td><td>22.77</td><td>0.9301</td><td>2695.0</td></tr><tr><td>ReLU</td><td>18.21</td><td>0.8030</td><td>3000.0</td></tr><tr><td>Tanh</td><td>17.94</td><td>0.7819</td><td>3000.0</td></tr></table>

Mixed-frequency regression. We regress a target signal composed of mixed sine/cosine components from 3.29 Hz to 79.90 Hz. Table 3 reports the dominant frequencies in the target (left) and train/test MSE of diferent activations (right).

Table 3: Left: top 5 dominant frequencies of the target synthetic function. Right: train/test MSE comparison across activation functions for mixed-frequency signal modeling.
<table><tr><td>Rank</td><td>Freq (Hz)</td><td>Amp</td><td>Phase</td><td>Model</td><td>Train ↓</td><td>Test↓</td></tr><tr><td>1</td><td>79.19</td><td>0.98</td><td>1.79</td><td>ReLU</td><td>0.0682</td><td>0.0681</td></tr><tr><td>2</td><td>79.23</td><td>0.96</td><td>4.41</td><td>GeLU</td><td>0.0752</td><td>0.0756</td></tr><tr><td>3</td><td>9.78</td><td>0.95</td><td>5.97</td><td>SiLU</td><td>0.0753</td><td>0.0756</td></tr><tr><td>4</td><td>18.02</td><td>0.92</td><td>1.42</td><td>SIREN</td><td>0.0236</td><td>0.0248</td></tr><tr><td>5</td><td>3.29</td><td>0.89</td><td>4.72</td><td>TESLA</td><td>0.0097</td><td>0.0100</td></tr></table>

In this mixed-frequency setting, TESLA is the only method with test MSE below 0.011.

## 5.3 Parity with Label Noise: Task, Metrics, and Interpretation

Task. We evaluate on Parity and global-statistics problems where the target depends on a high-order global interaction of the input bits. Let $x \in \{ 0 , 1 \} ^ { d }$

and $S \subseteq [ d ]$ with $| S | = m$ . The clean label is

$$
y _ { \mathrm { c l e a n } } = \bigoplus _ { i \in S } x _ { i } \in \{ 0 , 1 \} ,
$$

i.e., the parity of the selected bits.<sup>1</sup> This benchmark is intentionally adversarial to local, piecewise-linear activations, as the Bayes-optimal classifier is a global rule with a single nonzero Fourier coeficient at frequency S.

Data Generation, Splits, and Noise Settings. For each run, we sample i.i.d. inputs uniformly from $\{ 0 , 1 \} ^ { d }$ and assign splits deterministically via a fixed hash partition $h ( x )$ mod 3, ensuring disjoint and reproducible train/validation/test sets (i.e., no leakage). We then sample without replacement within each split, using 100,000 training samples and 20,000 validation samples for all d. We evaluate three settings: (i) clean parity, (ii) noisy parity with independent label flips $p \in \{ 0 . 0 5 , 0 . 1 0 , 0 . 2 0 , 0 . 3 0 \}$ , and (iii) LPN; in the noisy setting, train and test use the same flip rate p.

LPN. We also evaluate the classical LPN (Learning Parity with Noise) variant, where the relevant subset is unknown and must be inferred from data. Fix a hidden secret vector $s \in \{ 0 , 1 \} ^ { d }$ and draw query vectors $a \sim$ Unif $( \{ 0 , 1 \} ^ { d } )$ i.i.d. The observed label is

$$
y ~ = ~ \langle a , s \rangle { \mathrm { ~ m o d ~ 2 ~ } } \oplus ~ e , ~ e \sim \mathrm { B e r n o u l l i } ( \eta ) .
$$

Compared to the noisy parity setting in Sec. 5.3— where the subset S is fixed per run and implicitly known to the data generator—LPN additionally hides S (equivalently s), making both identification and optimization more challenging at larger d.

Metrics and Bayes Limit. With symmetric label noise (flip rate $p < \frac { 1 } { 2 } )$ on the test set, perfect recovery of the underlying rule cannot yield 100% measured accuracy. The Bayes-optimal expected test accuracy under symmetric, input-independent label flips at rate p is $A _ { \operatorname* { m a x } } ( p ) = 1 - p$

## 5.4 Forrelation

Forrelation (Aaronson and Ambainis, 2015) is a decision problem on pairs of Boolean functions $f , g \colon$ $\{ 0 , 1 \} ^ { n } \to \{ \pm 1 \}$ . Let $M = 2 ^ { n }$ . The goal is to decide whether the correlation

$$
\Phi _ { f , g } = { \frac { 1 } { \sqrt { M } } } \langle H f , g \rangle
$$

is high or low under a standard promise. Here H is the Hadamard matrix and $\langle \cdot , \cdot \rangle$ denotes the inner product.

We treat $\Phi _ { f , g } \ge 0 . 6$ as positive and $| \Phi _ { f , g } | \le 0 . 0 1$ as negative. The property of being forrelated is global. A classifier must integrate information spread across the entire pair of truth tables. We use a two-layer MLP that takes as input the concatenation of the truth tables of $f$ and $g$ with length $2 ^ { n + 1 }$ and outputs a single logit for binary prediction. We train and evaluate on a synthetic dataset with $n = 1 2 .$ , consisting of 10,000 training examples and 10,000 test examples, where labels are assigned as $y = 1$ if $\Phi _ { f , g } ~ \ge ~ 0 . 6$ and $y = 0$ if $| \Phi _ { f , g } | \leq 0 . 0 1$ , excluding samples with intermediate values.

## 5.5 ImageNet-100 Classification

To empirically validate the performance and eficiency of our proposed activation function, TESLA, we conducted experiments on the ImageNet-100 dataset (Russakovsky et al., 2015), a standard benchmark for image classification. To demonstrate TESLA’s general applicability, our evaluation includes both Transformer-based models (ViT (Dosovitskiy et al., 2021), MLP-Mixer (Tolstikhin et al., 2021)) and CNN-based models (ResNet (He et al., 2016), MobileNetV3 (Howard et al., 2019)). We compare TESLA’s performance against conventional activation functions, namely ReLU (Nair and Hinton, 2010), GeLU (Hendrycks and Gimpel, 2016), and SiLU (Elfwing et al., 2018). TESLA uses $K = 2$ for ViT-T/16, ResNet-18, and MobileNetV3, and $K = 6$ for MLP-Mixer. The evaluation is based on a comprehensive analysis of not only the classification performance (Top-1 Accuracy) but also key computational eficiency metrics, including FLOPs, throughput (imgs/s), and the number of parameters. Table 4 shows these results. The top-1 accuracy is reported as the mean and standard deviation over three seeds and both the measurement of FLOPs and throughput measurement are conducted on a single GPU.

## 5.6 Result Analysis

Parity. Figure 3 shows the performance of TESLA and baseline models. For the parity experiments we use a fully-connected MLP with 2 hidden layers and hidden dimension H = 128. Across the plotted settings, TESLA matches the Bayes limit across noise levels while remaining near 100% on the clean task. For $p \in \{ 0 . 0 5 , 0 . 1 0 , 0 . 2 0 , 0 . 3 0 \}$ , TESLA’s final accuracy concentrates near {95%, 90%, 80%, 70%}. In contrast, common activations always collapse toward 50% as the number of bits grows, which is no better than random guessing of parity. In the LPN setting, TESLA sustains high accuracy up to 27-bits before degrading in the hardest regimes.

Table 4: ImageNet-100 results across architectures comparing TESLA with common activations. Top-1 accuracy is mean ± std (3 seeds). Bold indicates best per architecture. FLOPs reported in GigaFLOPs, throughput (imgs/s) measured on a single GPU, and parameters (Params) in millions.
<table><tr><td>Model</td><td>Act</td><td> $\mathrm { { T o p } \mathrm { { - } 1 \ ( \% ) } }$ </td><td>FLOPs</td><td>imgs/s</td><td>Params</td></tr><tr><td rowspan="4">ViT-T/16</td><td>ReLU</td><td> $6 2 . 4 9 \pm 0 . 7 3$ </td><td>1.08</td><td>7,512</td><td>5.7</td></tr><tr><td>GeLU</td><td> ${ \bf 6 3 . 6 7 \pm 0 . 7 6 }$ </td><td>1.08</td><td>7,491</td><td>5.7</td></tr><tr><td>SiLU</td><td> $6 3 . 6 0 \pm 0 . 2 6$ </td><td>1.08</td><td>7,482</td><td>5.7</td></tr><tr><td>TESLA</td><td> $6 2 . 5 3 \pm 0 . 7 1$ </td><td>1.09</td><td>7,509</td><td>5.7</td></tr><tr><td rowspan="4">MLP-Mixer-b16</td><td>ReLU</td><td> $5 7 . 8 1 \pm 0 . 1 3$ </td><td>12.62</td><td>1,151</td><td>59.9</td></tr><tr><td>GeLU</td><td> $5 7 . 4 2 \pm 0 . 0 1$ </td><td>12.62</td><td>1,144</td><td>59.9</td></tr><tr><td>SiLU</td><td> $5 8 . 3 6 \pm 0 . 9 1$ </td><td>12.62</td><td>1,146</td><td>59.9</td></tr><tr><td>TESLA</td><td> ${ \bf 5 8 . 7 9 \pm 0 . 0 2 }$ </td><td>12.75</td><td>1,139</td><td>59.9</td></tr><tr><td rowspan="4">ResNet-18</td><td>ReLU</td><td> $7 3 . 1 4 \pm 0 . 3 1$ </td><td>1.82</td><td>8,447</td><td>11.7</td></tr><tr><td>GeLU</td><td> $7 3 . 9 3 \pm 0 . 4 7$ </td><td>1.82</td><td>8,307</td><td>11.7</td></tr><tr><td>SiLU</td><td> $7 3 . 9 9 \pm 0 . 6 6$ </td><td>1.82</td><td>8,326</td><td>11.7</td></tr><tr><td>TESLA</td><td> ${ \bf 7 4 . 0 1 \pm 0 . 3 3 }$ </td><td>1.82</td><td>8,433</td><td>11.7</td></tr><tr><td rowspan="4">MobileNetV3-S</td><td>ReLU</td><td> $6 8 . 2 0 \pm 0 . 1 6$ </td><td>0.05</td><td>35,607</td><td>2.0</td></tr><tr><td>GeLU</td><td> ${ \bf 6 8 . 2 3 \pm 0 . 9 4 }$ </td><td>0.05</td><td>35,082</td><td>2.0</td></tr><tr><td>SiLU</td><td> $6 7 . 9 2 \pm 0 . 1 5$ </td><td>0.05</td><td>35,056</td><td>2.0</td></tr><tr><td>TESLA</td><td> $6 7 . 8 2 \pm 0 . 1 7$ </td><td>0.05</td><td>35,483</td><td>2.0</td></tr></table>

Periodic baselines on parity. Table 5 compares TESLA against periodic and frequency-based baselines. For easier settings (16–20 bits), TESLA and SIREN both reach 100% accuracy under no noise. However, as the bit-length and noise increase, SIREN collapses to random guessing (≈ 50%) while TESLA remains stable (e.g., at 24 bits with 30% noise, TESLA achieves 65.53% vs. 49.47% for SIREN; at 32 bits with no noise, TESLA achieves 95.87% vs. 50.10% for SIREN). SNAKE remains near chance across all settings, and Fourier-Emb similarly fails to capture the global interaction signal in these regimes.

Forrelation. We evaluate a two-layer MLP on the n = 12 Forrelation dataset. Figure 4 reports validation accuracy over epochs for hidden sizes H = {128, 512, 1024}. TESLA consistently learns faster and reaches higher final accuracy than standard activations across all K settings. Larger hidden sizes improve performance for every method. The gap in favor of TESLA remains, which indicates that TESLA captures the global correlation signal more efectively.

ImageNet-100. Table 4 reports results on ViT-T/16, MLP-Mixer, ResNet-18, and MobileNetV3. To ensure architecture-aware comparison, we keep activations inside convolutional blocks unchanged for CNNs and replace only non-convolutional activations (e.g., MLP heads, projection/FFN layers) with TESLA; for ViT and MLP-Mixer, we replace all pointwise activations. TESLA is trained with an explicit $\ell _ { 1 }$ penalty enforcing a layer-wise coeficient budget $A _ { K }$ for stability and generalization control. Across models, TESLA remains competitive in Top-1 accuracy while keeping FLOPs and throughput within about 1% of standard activations. These results indicate that TESLA extends beyond synthetic parity-style benchmarks as a practical drop-in replacement for standard activations in realistic vision workloads.

<table><tr><td rowspan="2">Bits</td><td rowspan="2">Activation</td><td colspan="4">Noise probability</td></tr><tr><td>0.0</td><td>0.1</td><td>0.2</td><td>0.3</td></tr><tr><td rowspan="4">16</td><td>TESLA(K = 8)</td><td>100.00</td><td>89.80</td><td>79.22</td><td>68.00</td></tr><tr><td>SIREN(w0 = 30.0)</td><td>100.00</td><td>89.83</td><td>79.61</td><td>69.53</td></tr><tr><td> $\mathrm { S N A K E } ( \alpha = 1 . 0 )$ </td><td>49.95</td><td>49.97</td><td>49.58</td><td>49.89</td></tr><tr><td>Fourier-Emb</td><td>73.21</td><td>50.19</td><td>50.09</td><td>50.34</td></tr><tr><td rowspan="4">20</td><td>TESLA(K = 8)</td><td>100.00</td><td>89.52</td><td>78.78</td><td>67.14</td></tr><tr><td> $\mathrm { S I R E N } ( w _ { 0 } = 3 0 . 0 )$ </td><td>100.00</td><td>89.74</td><td>79.42</td><td>69.67</td></tr><tr><td> $\mathrm { S N A K E } ( \alpha = 1 . 0 )$ </td><td>50.01</td><td>50.02</td><td>50.06</td><td>50.07</td></tr><tr><td>Fourier-Emb</td><td>52.13</td><td>49.68</td><td>50.12</td><td>50.33</td></tr><tr><td rowspan="4">24</td><td>TESLA(K = 8)</td><td>100.00</td><td>89.75</td><td>78.50</td><td>65.53</td></tr><tr><td> $\mathrm { S I R E N } ( w _ { 0 } = 3 0 . 0 )$ </td><td>100.00</td><td>90.08</td><td>80.06</td><td>49.47</td></tr><tr><td> $\mathrm { S N A K E } ( \alpha = 1 . 0 )$ </td><td>50.24</td><td>49.90</td><td>50.52</td><td>49.97</td></tr><tr><td>Fourier-Emb</td><td>50.55</td><td>49.20</td><td>50.20</td><td>50.00</td></tr><tr><td rowspan="4">28</td><td>TESLA(K = 8)</td><td>96.97</td><td>89.61</td><td>78.10</td><td>68.14</td></tr><tr><td>SIREN(w0 = 30.0)</td><td>50.48</td><td>50.52</td><td>49.58</td><td>49.78</td></tr><tr><td> $\mathrm { S N A K E } ( \alpha = 1 . 0 )$ </td><td>49.95</td><td>49.36</td><td>50.02</td><td>50.19</td></tr><tr><td>Fourier-Emb</td><td>49.74</td><td>50.45</td><td>50.27</td><td>49.81</td></tr><tr><td rowspan="4">32</td><td>TESLA(K = 8)</td><td>95.87</td><td>85.78</td><td>78.05</td><td>68.78</td></tr><tr><td> $\mathrm { S I R E N } ( w _ { 0 } = 3 0 . 0 )$ </td><td>50.10</td><td>49.89</td><td>50.02</td><td>49.95</td></tr><tr><td> $\mathrm { S N A K E } ( \alpha = 1 . 0 ) $ </td><td>50.24</td><td>50.37</td><td>50.34</td><td>50.10</td></tr><tr><td>Fourier-Emb</td><td>50.06</td><td>49.42</td><td>50.12</td><td>50.48</td></tr></table>

Table 5: Parity with long bit-length and label noise: validation accuracy(%) (20 epochs) under matched architectures and training settings.

Stability vs. sinusoidal baselines. To assess whether TESLA remains numerically stable in practical architectures, we also compare against SIREN under the same ImageNet-100 training settings. In these runs, SIREN exhibits severe optimization instability and much lower accuracy, whereas TESLA trains stably and reaches standard accuracy levels (see Table 6).

## 6 FUTURE WORK

TESLA opens several promising avenues for future research. Theoretically, we plan to extend our continuous Fourier and NTK analysis to discrete Boolean domains to establish rigorous approximation bounds and explain the empirical gains observed on paritytype tasks. On the systems side, we will investigate sparse factorizations, low-bit quantization, and hardware-aware implementations to preserve TESLA’s spectral benefits while minimizing inference latency. Finally, we aim to integrate TESLA into large-scale Transformers and LLMs—such as augmenting FFN activations or positional encodings—to evaluate its im pact on long-range reasoning, compositional generalization, and sample eficiency.

<table><tr><td>Model</td><td>Act</td><td>Top-1 (%)</td></tr><tr><td rowspan="2">ViT-T/16</td><td>SIREN</td><td> $4 . 1 1 \pm 0 . 6 7$ </td></tr><tr><td>TESLA</td><td> ${ \bf 6 2 . 5 3 \pm 0 . 7 1 }$ </td></tr><tr><td rowspan="2">MLP-Mixer-b16</td><td>SIREN</td><td> $3 2 . 4 1 \pm 1 0 . 4 5$ </td></tr><tr><td>TESLA</td><td> ${ \bf 5 8 . 7 9 \pm 0 . 0 2 }$ </td></tr><tr><td rowspan="2">ResNet-18</td><td>SIREN</td><td> $3 7 . 4 1 \pm 1 . 3 4$ </td></tr><tr><td>TESLA</td><td> ${ \bf 7 4 . 0 1 \pm 0 . 3 3 }$ </td></tr><tr><td rowspan="2">MobileNetV3-S</td><td>SIREN</td><td> $3 9 . 5 3 \pm 0 . 8 5$ </td></tr><tr><td>TESLA</td><td> ${ \bf 6 7 . 8 2 \pm 0 . 1 7 }$ </td></tr></table>

Table 6: ImageNet-100 comparison between TESLA and SIREN under identical training settings. Top-1 accuracy is mean ± std.

## 7 CONCLUSION

In this work, we introduced TESLA, a learnable sinusoidal activation that enables explicit degree/frequency control while preserving stable optimization through coeficient budgeting. We provided Lipschitz and Rademacher-complexity bounds and analyzed mode-wise learning dynamics to connect the parameterization to frequency-selective behavior.

Empirically, TESLA consistently improves tasks that require global, high-order interactions. On parity with label noise, it remains well above chance at larger bit-lengths and heavy corruption, and is more stable than periodic/frequency-based baselines (SIREN, SNAKE, and Fourier feature embeddings) in the hard est regimes. On Forrelation, TESLA achieves the high est accuracy across widths, while periodic and Fourierstyle baselines remain near chance. On continuousdomain tasks (PINNs, INRs, and mixed-frequency regression), TESLA is competitive with or better than both standard and sinusoidal activations. On ImageNet-100, TESLA remains competitive across architectures with modest overhead and is substantially more stable than SIREN. Overall, TESLA suggests that activation-level degree control is a robust, practical alternative to input-only spectralization.

## References

Aaronson, S. and Ambainis, A. (2015). Forrelation: A problem that optimally separates quantum from classical computing. In Proceedings of the 47th Annual ACM Symposium on Theory of Computing (STOC 2015), page 307–316.

Agostinelli, F., Hofman, M. D., Sadowski, P. J., and Baldi, P. (2015). Learning activation functions to improve deep neural networks. In 3rd International Conference on Learning Representations (ICLR 2015), Workshop Track Proceedings.

Apicella, A., Donnarumma, F., Isgr\`o, F., and Prevete, R. (2021). A survey on modern trainable activation functions. Neural Networks, 138:14–32.

Barron, A. R. (1993). Universal approximation bounds for superpositions of a sigmoidal function. IEEE Transactions on Information Theory.

Barron, A. R. (1994). Approximation and estimation bounds for artificial neural networks. Machine learning.

Daniely, A. and Malach, E. (2020). Learning parities with neural networks. In Advances in Neural Information Processing Systems 33 (NeurIPS 2020), pages 20356–20365.

Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J., and Houlsby, N. (2021). An image is worth 16x16 words: Transformers for image recognition at scale. In 9th International Conference on Learning Representations (ICLR 2021).

Eldan, R. and Shamir, O. (2016). The power of depth for feedforward neural networks. In 29th Annual Conference on Learning Theory (COLT 2016), pages 907–940.

Elfwing, S., Uchibe, E., and Doya, K. (2018). Sigmoidweighted linear units for neural network function approximation in reinforcement learning. Neural Networks, 107:3–11.

He, K., Zhang, X., Ren, S., and Sun, J. (2015). Delving deep into rectifiers: Surpassing human-level performance on imagenet classification. In Proceedings of the IEEE International Conference on Computer Vision (ICCV 2015), pages 1026–1034.

He, K., Zhang, X., Ren, S., and Sun, J. (2016). Deep residual learning for image recognition. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR 2016), pages 770– 778.

Hendrycks, D. and Gimpel, K. (2016). Gaussian error linear units (gelus). arXiv preprint arXiv:1606.08415.

Howard, A., Sandler, M., Chu, G., Chen, L.-C., Chen, B., Tan, M., Wang, W., Zhu, Y., Pang, R., Vasudevan, V., Le, Q. V., and Adam, H. (2019). Searching for mobilenetv3. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV 2019), pages 1314–1324.

Jacot, A., Gabriel, F., and Hongler, C. (2018). Neural tangent kernel: Convergence and generalization in neural networks. In Advances in Neural Information Processing Systems 31 (NeurIPS 2018), pages 8571– 8580.

Ledoux, M. and Talagrand, M. (2013). Probability in Banach Spaces: isoperimetry and processes. Springer Science & Business Media.

Liu, Z., Wang, Y., Vaidya, S., et al. (2024). Kan: Kolmogorov–arnold networks. arXiv preprint arXiv:2404.19756.

Nair, V. and Hinton, G. E. (2010). Rectified linear units improve restricted boltzmann machines. In Proceedings of the 27th international conference on machine learning (ICML 2010), pages 807–814.

Rahaman, N., Baratin, A., Arpit, D., Draxler, F., Lin, M., Hamprecht, F., Bengio, Y., and Courville, A. (2019). On the spectral bias of neural networks. In Proceedings of the 36th International Conference on Machine Learning (ICML 2019), pages 5301–5310.

Rahimi, A. and Recht, B. (2007). Random features for large-scale kernel machines. In Advances in Neural Information Processing Systems 20 (NIPS 2007), pages 1177–1184.

Russakovsky, O., Deng, J., Su, H., Krause, J., Satheesh, S., Ma, S., Huang, Z., Karpathy, A., Khosla, A., Bernstein, M., et al. (2015). Imagenet large scale visual recognition challenge. International journal of computer vision.

Scardapane, S., Scarpiniti, M., Comminiello, D., and Uncini, A. (2019). Learning Activation Functions from Data Using Cubic Spline Interpolation. Springer International Publishing.

Sitzmann, V., Martel, J., Bergman, A., Lindell, D., and Wetzstein, G. (2020). Implicit neural representations with periodic activation functions. In Advances in Neural Information Processing Systems 33 (NeurIPS 2020), pages 7462–7473.

Tancik, M., Srinivasan, P., Mildenhall, B., Fridovich-Keil, S., Raghavan, N., Singhal, U., Ramamoorthi, R., Barron, J., and Ng, R. (2020). Fourier features let networks learn high frequency functions in low dimensional domains. In Advances in Neural Information Processing Systems 33 (NeurIPS 2020), pages 7537–7547.

Telgarsky, M. (2016). benefits of depth in neural networks. In 29th Annual Conference on Learning Theory (COLT 2016), pages 1517–1539.

Tolstikhin, I. O., Houlsby, N., Kolesnikov, A., Beyer, L., Zhai, X., Unterthiner, T., Yung, J., Steiner, A., Keysers, D., Uszkoreit, J., Lucic, M., and Dosovitskiy, A. (2021). Mlp-mixer: An all-mlp architecture for vision. In Advances in Neural Information Processing Systems 33 (NeurIPS 2020), pages 24261– 24272.

Walsh, J. L. (1923). A closed set of normal orthogonal functions. American Journal of Mathematics.

Yarotsky, D. (2017). Error bounds for approximations with deep relu networks. Neural Networks, 94:103– 114.

Zhi-Qin John Xu, Z.-Q. J. X., Yaoyu Zhang, Y. Z., Tao Luo, T. L., Yanyang Xiao, Y. X., and Zheng Ma, Z. M. (2020). Frequency principle: Fourier analysis sheds light on deep neural networks. Communications in Computational Physics.

## TESLA: Taylor Expansion of Sinusoidal Learnable Activations Supplementary Materials

## 1 Experiment environment summary

This appendix provides details, such as hyperparameter, computer hardware and software environment information, necessary to reproduce the experiments reported in the main text. K denotes the number of sinusoida harmonics in each TESLA layer, and $A _ { K }$ is the $l _ { 1 } .$ -style coeficient budget for controlling their amplitude.

Table 1: Hyperparameter summary for main experiments. Each reported experiment in the paper uses the settings below unless noted. All task were conducted on backbone with MLP, 2 layers, and optimizer with Adam and $A _ { k } = 0$
<table><tr><td>Task</td><td>Width</td><td>K</td><td>LR</td><td>Batch Size</td><td>Epochs</td></tr><tr><td>Parity (d = 16 − 32, n = 100k)</td><td>128</td><td>d/4</td><td>1e-3</td><td>1024</td><td>30</td></tr><tr><td>Forrelation (d = 12)</td><td>[128, 512, 1024] [1, 2, 4, 16]</td><td></td><td>1e-3</td><td>128</td><td>20</td></tr><tr><td>LPN (noise = 0.1)</td><td>128</td><td>d/4</td><td>5e-4</td><td>1024</td><td>30</td></tr></table>

Table 2: Hyperparameter summary for ImageNet-100. All models are trained with AdamW for 50 epochs using cosine LR decay. Warmup uses a linear schedule; Warmup ratio is the LR multiplier at the first warmup step. All backbones fix the batch size to 128, set the learning rate to 1e-3, perform warm-up for 5 epochs, and set the warm-up ratio to 0.85.
<table><tr><td>Backbone</td><td>K</td><td> $A _ { K }$ </td></tr><tr><td>ViT-T/16</td><td>2</td><td>1e-3</td></tr><tr><td>MLP-Mixer-b16</td><td>6</td><td>1e-1</td></tr><tr><td>ResNet-18</td><td>2</td><td>1e-3</td></tr><tr><td>MobileNetV3-S</td><td>2</td><td>1e-3</td></tr></table>

Table 3: Compute and software environment used for all experiments unless otherwise noted.
<table><tr><td>Hardware</td><td>NVIDIA A100 80GB (1x), AMD EPYC 7543 32-Core Processor, 256GB RAM</td></tr><tr><td>OS</td><td>Ubuntu 22.04 LTS</td></tr><tr><td>Python / Framework</td><td>Python 3.10, PyTorch 2.0+</td></tr><tr><td>CUDA / cuDNN / Driver</td><td>CUDA 11.8, cuDNN 8.x, NVIDIA Driver 535+</td></tr><tr><td>Key Python deps</td><td>numpy 1.26, scipy 1.14, triton 3.1.0, tqdm 4.x</td></tr><tr><td>Reproducibility</td><td>Fixed seeds {0, 1, 2}; all results reported as mean ± std over seeds.</td></tr></table>

## 2 Representations and Generalization on Toy Tasks: Synthetic Data & Parity

This section complements the theory by visualizing how TESLA difers from standard activations on synthetic 2D tasks and on the 10-bit parity mapping.

![](images/1092dc63f36ccf6eda05e0914a391dff2ef92c56c9b16e744666c8410a29eb3d.jpg)  
Figure 1: Decision-boundary comparison on two synthetic datasets. A two-layer MLP with 128 hidden units is trained for 200 epochs on 1,000 samples. TESLA is evaluated with multiple K and compared against KAN and standard activations. Subpanel titles report training accuracy in percent. All subpanels share the same color and legend scale.

![](images/0822e0abe224e8af926be9f7800dc292b30cd950ec1aa0cc061181d4431de992.jpg)  
Figure 2: Decision-boundary comparison on two additional datasets. Training setup matches Figure 1. TESLA consistently recovers global or oscillatory structure, whereas baselines often miss these patterns. Subpanel titles report training accuracy in percent. Color and legend scales are shared across subpanels.

![](images/f6df1d520e99b9717092e5a72b6d53f8f86b2862867ba2003253bffd2a16a419.jpg)  
Figure 3: Parity decision surfaces on 10-bit inputs. A two-layer MLP $( \mathrm { h i d d e n } = 1 2 8 , \mathrm { l r } = 1 \mathrm { e } \mathrm { - } 3$ , 50 epochs) is trained on 350 samples with diferent activations. Each 10-bit string is split into two 5-bit halves mapped to the x- and y-axes as binary integers (0–31), yielding a $3 2 { \times } 3 2 ~ \mathrm { g r i d }$ . Color shows predicted $P ( y = 1 )$ (blue = 0, red = 1). Black dots mark training points. The top-left panel is ground truth. Panel titles report training accuracy (%).