# Demystifying Oversmoothing in Sheaf Neural Networks: An Index-Theoretic Criterion

Junwen Dong<sup>∗</sup> Chern Institute of Mathematics Nankai University Tianjin, China

Yuhan Peng<sup>∗</sup> School of Physical and Mathematical Sciences Nanyang Technological University Singapore

Hao Li<sup>∗</sup> Department of Mathematics, Faculty of Science National University of Singapore Singapore

Huitao Feng Chern Institute of Mathematics Nankai University Tianjin, China

Kelin Xia<sup>†</sup> School of Physical and Mathematical Sciences Nanyang Technological University Singapore xiakelin@ntu.edu.sg

## Abstract

To combat oversmoothing in Graph Convolutional Networks, Sheaf Neural Networks (SNNs) were proposed as a generalization by equipping the graph with a sheaf structure and replacing the graph Laplacian with a sheaf Laplacian L. Existing analyses connect sheaf diffusion to oversmoothing via the harmonic space (ker L), taking its absolute dimension as an indicator of anti-oversmoothing capacity. However, absolute dimension alone is not a reliable measure: certain sheaf configurations inflate dim ker L while their harmonic sections remain entirely constant, without enriching discriminative capacity. We instead introduce the first relative, geometric approach, yielding a precise characterisation of anti-oversmoothing capacity. Under natural conditions on stalk transportation and global sheaf structure, we establish an index-theoretic comparison criterion showing that one sheaf’s harmonic space genuinely contains another’s beyond trivial inflation. We illustrate this with a concrete instance and further introduce GyroSheaf, a sheaf with curved gyrovector-space stalks, extending the criterion to the non-linear setting via local tangent-space linearization. Experiments across ten models confirm the theoretical criterion: sheaf models violating the criterion collapse despite possessing index jumps, while compliant models maintain depth-stable representations.

## 1 Introduction

Oversmoothing is the phenomenon where node representations become increasingly similar under repeated message passing, reducing their ability to separate node classes Li et al. [2018]. In the idealized diffusion view, this corresponds to the asymptotic projection of node signals onto the harmonic space of the graph Laplacian. For a connected graph, harmonic space consists of constant signals, so the diffusion limit collapses all node features onto a single global mode, precisely the classical oversmoothing phenomenon. Sheaf Neural Networks (SNNs) Hansen and Gebhart [2020] generalize graph diffusion by equipping a graph with stalk spaces and restriction maps (setting all stalks to R with identity maps recovers the graph Laplacian of GCN Bodnar et al. [2022]), yielding a sheaf Laplacian whose harmonic space can retain nontrivial sections that preserve node-level differences even in the deep limit. Bodnar et al. Bodnar et al. [2022] further studied the relationship between the harmonic space of the sheaf Laplacian and oversmoothing in SNNs. These results highlight the central role of the harmonic space in understanding oversmoothing. However, for a fixed sheaf, the dimension of the harmonic space alone does not provide a universal measure of anti-oversmoothing capacity. Certain sheaves may possess a large harmonic space whose sections are nonetheless constant; conversely, a small harmonic space may consist entirely of nonconstant, class-preserving sections. Thus anti-oversmoothing cannot be characterised merely by the rank or dimension of the harmonic space of a single sheaf. This motivates a relative, geometric comparison: rather than measuring the harmonic space of a single sheaf in isolation, we ask what additiona harmonic structure a target sheaf $\mathcal { F }$ admits beyond that of a reference sheaf $\xi .$

Our approach. A first natural indicator of this relative capacity is the index jump $\Delta _ { i n d } ( \mathcal { F } , \xi ) : =$ $i n d ( D _ { \mathcal { F } } ^ { + } ) - i n d ( D _ { \xi } ^ { + } )$ ) which yields the harmonic-capacity difference dim $H ^ { 0 } ( { \mathcal { F } } ) \mathrm { - } \mathrm { d i m } H ^ { 0 } ( { \xi } )$ (Proposition 4). However, a positive raw index jump alone does not guarantee genuine anti-oversmoothing: trivial configurations $( \mathrm { e . g . }$ , IdentitySheaf or InverseSheaf; see Example 1) can inflate dim $H ^ { 0 }$ via channel-wise constant sections that still collapse. Thus, the raw index cannot distinguish meaningful harmonic enlargement from trivial stalk dimension inflation.

This led us to propose a more detailed and structurally informed set of criterion. Our main result (Theorem 5) establishes that the index jump must be combined with a degree-one heat trace correction $\Delta _ { S t r } ^ { ( 1 ) }$ to properly control $H ^ { 0 }$ . Specifically, when the restriction maps induce non-trivial holonomy (preventing full channel decoupling), and the corrected index balance satisfies our capacity conditions, we prove that $H ^ { 0 } ( { \mathcal { F } } ) / \iota _ { * } H ^ { 0 } ( { \dot { \xi } } ) \not = 0$ . Rather than a mere numerical comparison, this establishes a genuine geometric strict inclusion: the harmonic space of $\mathcal { F }$ strictly contains that of $\xi$ embedded via $\iota _ { * }$ . This excess harmonic capacity precisely measures $\mathcal { F } \mathrm { { s } }$ structural resistance to oversmoothing. Furthermore, Section 5 extends this criterion to sheaves with non-linear stalks.

Contributions. Our main contributions are threefold: (1) A precise mathematical characterisation for anti-oversmoothing capacity. We establish (Theorem 5) the first algebraic-topological inclusion criterion that enables a rigorous comparison of anti-oversmoothing capacity between arbitrary sheaf frameworks, validated by oversmoothing experiments across ten models on three datasets. (2) Extension to non-linear stalk structures. We generalise the index-theoretic framework beyond the Euclidean setting to sheaves whose stalks carry non-linear geometry (e.g. SPD manifolds), broadening the scope of the theory to sheaf architectures that operate on curved feature spaces. (3) GyroSheaf: a concrete non-linear instantiation. We construct GyroSheaf, a sheaf model whose stalks carry gyrovector-space structure, and prove that it admits a tangent Hodge decomposition. This construction demonstrates the design principles of our criterion on curved stalks. We also provide a direct comparison between the nonlinear and linear sheaf frameworks on curved stalks.

## 2 Related Work

Oversmoothing in GNNs. The oversmoothing behavior of message-passing GNNs—GCN Kipf and Welling [2017], GAT Velickovic et al. [2018], GraphSAGE Hamilton et al. [2017]—has been characterized both as Laplacian smoothing Li et al. [2018] and as exponential convergence to a low-dimensional subspace determined by the graph’s connected components and node degrees Oono and Suzuki [2020]; see Rusch et al. [2023] for a survey. Existing remedies fall into three broad families: normalization/regularization approaches that suppress feature collapse layer-wise Zhao and Akoglu [2020], Zhou et al. [2021]; residual/skip-connection variants that preserve earlier layers signals Xu et al. [2018], Chen et al. [2020]; energy orflow-based methods that recast propagation as a continuous dynamical system and control its limit Chamberlain et al. [2021], Giovanni et al. [2022]. These methods do not, however, directly change the harmonic subspace that determines the infinite-depth limit of the underlying graph diffusion. This motivates a complementary line of work that changes the diffusion operator itself; sheaf Laplacians Hansen and Gebhart [2020], Bodnar et al. [2022], Hansen and Ghrist [2019] provide such a mechanism by making the limiting harmonic space depend on stalks and restriction maps rather than only on graph connectivity.

Sheaf neural networks and geometric extensions. Sheaf neural networks Hansen and Gebhart [2020], Bodnar et al. [2022] replace graph diffusion with sheaf diffusion, enabling a richer harmonic space that can preserve non-constant signals at the diffusion limit, with subsequent extensions to hypergraphs and cell complexes Duta et al. [2023], Barbero et al. [2022], Battiloro et al. [2024]. However, existing sheaf-theoretic analyses characterize anti-oversmoothing capacity through the absolute dimension dim $H ^ { 0 }$ of a single sheaf, without a relative criterion for comparing two sheaf frameworks. On the geometric side, hyperbolic GNNs Liu et al. [2019] and SPD-valued GNNs Zhao et al. [2023], Wang and Chang [2026] perform message passing in non-Euclidean spaces, with Peng et al. [2026] further extend SPD geometry to the sheaf setting. Gyrovector algebras Ungar [2009] have been adopted in hyperbolic Liu et al. [2024] and SPD López et al. [2021], Nguyen [2022] graph networks, but have not been used to construct sheaf coboundary operators. Neither line of work provides a relative criterion for strict harmonic inclusion—our index comparison theorem fills this gap, and GyroSheaf extends it to curved gyrovector stalks as a non-linear instantiation.

## 3 A Sheaf-Theoretic Diffusion Framework on Graphs

Preliminaries. This section develops the framework of sheaf diffusion on graphs, which forms the basis for the index-theoretic analysis in Section 4. Our discrete construction is guided by continuous heat diffusion: on a compact Riemannian manifold $M ,$ , under the positive-semidefinite convention for the Laplace–Beltrami operator $\Delta .$ , the heat equation $\partial _ { t } f = - \dot { \Delta } f$ is the $L ^ { 2 }$ –gradient flow of the Dirichlet energy $\begin{array} { r } { E ( f ) = \frac { 1 } { 2 } { \stackrel { \cdot } { \int } } _ { M } \| \nabla f \| ^ { 2 } } \end{array}$ . Along this flow, the energy is monotonically dissipated, and the solution converges as $t \to \infty$ to its projection onto the harmonic space ker $\Delta$ . By Hodge theory, harmonic spaces are isomorphic to de Rham cohomology groups, connecting diffusion equilibria with the topology of the underlying space.

We reproduce this structure in the discrete sheaf setting. The core algebraic objects are well established for vector-space stalks Hansen and Gebhart [2020], Bodnar et al. [2022], Hansen and Ghrist [2019]. We restate them in a unified form extending to general stalk spaces (e.g., SPD matrices). Definition 1, 2 and Theorem 2 recover the classical formulations of Bodnar et al. [2022], Hansen and Ghrist [2019] for linear stalks. Remark 2 notes that the algebraic definitions requiring addition and scalar multiplication (Definition 3, Proposition 1) generalize to SPD stalks either via the Lie group operation Peng et al. [2026] required for Section 4 or via tangent-space linearization required for Section 5 when no group is assumed.

## 3.1 Sheaves on Graphs

Definition 1 (Discrete sheaf). A discrete sheaf F on an undirected graph $G = ( V , E )$ consists of:

• A stalk space $\mathcal { F } _ { v }$ to each node $v \in V$ , and a stalk space $\mathcal { F } _ { e }$ to each edge $e \in E { \mathrm { : } }$

• For every incident pair $v \in e ,$ , a restriction map $\mathcal { F } _ { v \to e } : \mathcal { F } _ { v } \to \mathcal { F } _ { e }$ , acting as a discrete parallel transport. The restriction maps encode the compatibility relations between adjacent stalks.

In the linear setting, each stalk is a finite-dimensional vector space $( \boldsymbol { \mathrm { e . g . } } \mathcal { F } _ { v } = \mathbb { R } ^ { n } )$ and each restriction map is a linear transformation. In the nonlinear setting, each stalk is a Riemannian manifold (e.g. $\mathcal { F } _ { v } = \mathrm { S P D } _ { n } , n \times n$ SPD matrices) and each restriction map is a smooth map between manifolds. The 0-cochain space is $\begin{array} { r } { \mathcal { C } ^ { 0 } ( G , \mathcal { F } ) : = \prod _ { v \in V } \mathcal { F } _ { v } } \end{array}$ and the 1-cochain space is $\begin{array} { r } { \mathcal { C } ^ { 1 } \hat { ( } G , \mathcal { F } ) : = \prod _ { e \in E } \mathcal { F } _ { e } } \end{array}$

## 3.2 Coboundary Operator, Admissible Pairing, and Adjoint Operator

Definition 2 (Discrete coboundary). Let $\mathcal { F }$ be a sheaf on a graph $G .$ . The discrete coboundary operator $\delta : \mathcal { C } ^ { 0 } ( G , \mathcal { F } )  \mathcal { C } ^ { 1 } ( G , \mathcal { F } )$ maps node-wise assignments to edge-wise discrepancies. For $\dot { X } \in { \mathcal { C } } ^ { 0 } ( G , { \mathcal { F } } )$ and an oriented edge $e = ( u , v ) , ( \delta X ) _ { e } \in \mathcal { F } _ { e }$ compares the transported endpoint features $\mathcal { F } _ { u  e } X _ { u }$ and $\mathcal { F } _ { v  e } X _ { v }$ in the common edge stalk $\mathcal { F } _ { e }$

Remark 1. The concrete form of δ depends on the geometry of the stalks. For vector-space stalks, it is given by the signed difference: $( \delta \bar { X _ { ) _ { e } } } = \mathcal { F } _ { u  e } \bar { X _ { u } } - \mathcal { F } _ { v  e } \bar { X _ { v } }$ . For general geometric stalks, this subtraction is replaced by an appropriate discrepancy map on $\mathcal { F } _ { e }$

To define the adjoint $\delta ^ { * }$ of the coboundary operator $\delta : { \mathcal { C } } ^ { 0 } \to { \mathcal { C } } ^ { 1 }$ , we need pairings on the cochain spaces. Unlike a smooth manifold equipped with a Riemannian metric, where the $L ^ { 2 }$ pairing is induced by the metric and volume form, a discrete sheaf lacks a canonical global cochain pairing. We therefore construct the global cochain pairings explicitly by summing the local pairings on the stalks: $\begin{array} { r } { \langle X , X ^ { \prime } \rangle _ { 0 } : = \sum _ { v \in V } \langle \tilde { X _ { v } } , X _ { v } ^ { \prime } \rangle _ { \mathcal { F } _ { v } } } \end{array}$ for $\dot { X } , X ^ { \prime } \in \mathcal { C } ^ { 0 }$ and $\begin{array} { r } { \langle \bar { Y } , \bar { Y ^ { \prime } } \rangle _ { 1 } : = \breve { \sum _ { e \in E } } \langle Y _ { e } , \bar { Y ^ { \prime } } \rangle _ { \mathcal { F } _ { e } } } \end{array}$ for $Y , Y ^ { \prime } \in { \mathcal { C } } ^ { 1 }$ Definition 3 (Admissible pairing). We say that a global cochain pairing is admissible if for $i \in \{ 0 , 1 \}$ the pairing $\langle \cdot , \cdot \rangle _ { i }$ is bilinear, symmetric, and positive definite. That is, for $X , Y , Z \in { \mathcal { C } } ^ { i }$ and $\alpha , \beta \in \mathbb { R }$

$$
\langle \alpha X + \beta Y , Z \rangle _ { i } = \alpha \langle X , Z \rangle _ { i } + \beta \langle Y , Z \rangle _ { i } , \qquad \langle X , Z \rangle _ { i } = \langle Z , X \rangle _ { i } ,
$$

and $\langle X , X \rangle _ { i } > 0$ for all $X \neq 0$

Definition 4 (Adjoint operator). With respect to these pairings, the adjoint operator $\delta ^ { * }$ satisfies: $\langle \delta X , Y \rangle = \langle X , \delta ^ { * } Y \rangle$ , for all $X ^ { ^ { \prime } } \in \mathcal { C } ^ { 0 } , \ Y \in \overset { ^ { \prime } } { \mathcal { C } ^ { 1 } }$

## 3.3 Sheaf Laplacian and Associated Dirichlet Energy

Definition 5 (Sheaf Laplacian and Dirichlet Energy). The sheaf Laplacian on $\mathcal { C } ^ { 0 }$ is defined as $\mathcal { L } : = \delta ^ { * } \delta$ . and the associated Dirichlet energy is defined as $\begin{array} { r } { \mathcal { E } ( X ) : = \frac { 1 } { 2 } \| \delta X \| ^ { 2 } } \end{array}$

Analogous to the Laplace–Beltrami operator $\Delta$ on Riemannian manifolds, L induces the sheaf heat equation $\dot { X } ( t ) = - \mathcal { L } X ( t )$ , whose solution converges to ker L as $t \to \infty$ . An explicit Euler discretization yields the sheaf diffusion layer $X _ { k + 1 } = \sigma ( \Breve { ( } I - \mathcal { L } ) X _ { k } )$ . And the following proposition shows the energy-Laplacian relationship mirrors the continuous setting.

Remark 2. These operators admit two natural nonlinear extensions: when a Lie-group structure is available, δ and $\delta ^ { * }$ lift to group operations (Section 4.3); when no such structure is assumed, one instead linearises at each cochain and works in the local tangent space (Section 5).

Proposition 1 (Energy-Laplacian correspondence). The Laplacian $\mathcal { L } = \delta ^ { * } \delta$ is tied to the discrete energy $\mathcal { E } ( f )$ through thefollowing properties:

• L is the gradient of the Dirichlet energy: $\nabla { \mathcal { E } } ( X ) = { \mathcal { L } } X .$

• The quadraticform ofL equals the total edge-wise sheafinconsistency: $\langle \mathcal { L } X , X \rangle _ { 0 } = \| \delta X \| _ { 1 } ^ { 2 } =$ $2 \mathcal { E } ( \hat { X } )$

• The positive semidefiniteness ofL ensures stability ofthe continuous diffusion process $\partial _ { t } X _ { t } =$ $- \mathcal { L } \bar { X } _ { t }$ : it dissipates energy and stabilizes at harmonic equilibria.

## 3.4 Discrete Hodge Theory

As noted above, discrete sheaf diffusion drives features toward ker ${ \mathcal { L } } ,$ so over-smoothing is governed by the structure of this limiting subspace. Following Bodnar et al. [2022], Hansen and Ghrist [2019], Proposition 3 characterizes ker $\mathcal { L }$ in the general stalk setting of Definition 1 and formalizes anti-oversmoothing as a structural property of $\mathcal { H } ^ { \overline { { { 0 } } } }$

Theorem 2 (0th-order Discrete Hodge Decomposition). Let $\mathcal { L } = \delta ^ { * } \delta$ be the sheaf Laplacian. Then

$$
\ker { \mathcal { L } } \cong \ker \delta \cong H ^ { 0 } ( G , { \mathcal { F } } ) = \{ X \in C ^ { 0 } ( G , { \mathcal { F } } ) : \delta X = 0 \} .
$$

Thus the diffusion equilibrium is canonically isomorphic to the zeroth sheaf cohomology, i.e., the space of global sections (defined as ker δ) satisfying strict edgewise consistency. Thus over-smoothing reduces to a structural question: how large is the harmonic capacity dim $\mathcal { H } ^ { 0 }$ of the underlying sheaf?

Proposition 3 (Harmonic Characterisation of Over-Smoothing; proof in Appendix B.5). Let $P _ { \mathcal { H } ^ { 0 } }$ $\mathcal { C } ^ { 0 } \to \mathcal { H } ^ { 0 }$ be the projection onto harmonic space $\mathcal { H } ^ { 0 } = \ker \mathcal { L } .$ . For any initialfeature assignment $f ^ { ( 0 ) }$ the deep limit ofsheafdiffusion satisfies $f ^ { ( \infty ) } = P _ { \mathcal { H } ^ { 0 } } f ^ { ( 0 ) }$ . The resistance of $\mathcal { F }$ to over-smoothing is governed by the representational capacity of ${ \mathcal { H } } ^ { 0 } \cong { \dot { H } } ^ { 0 } ( G , { \mathcal { F } } )$

(1) Flat collapse: Ifdim $( \mathcal { H } ^ { 0 } )$ is minimal $\mathcal { H } ^ { 0 }$ lacks the representational capacity to preserve nodelevel distinctions, and $f ^ { ( \infty ) }$ collapses to a uniform state.

(2) Geometric preservation: A richer $\mathcal { H } ^ { 0 }$ provides sufficient representational capacity so that $f ^ { ( \infty ) }$ preserves node-level distinctions, preventing collapse.

## 4 An Index-Theoretic Analysis of Over-Smoothing

As established previously, mitigating operator-induced over-smoothing requires a structural enlargement of the zeroth harmonic space. A natural analytic invariant is the "raw" index, but this quantity is Euler-balanced: it records the difference between 0th and 1st discrete sheaf cohomology, not the 0th harmonic capacity alone. We instead combine the index jump with a degree-one heat-supertrace correction and state the resulting criterion in the main text and defer the full derivation to Appendix $C ,$ including the holonomy-fixed sufficient condition.

## 4.1 Index jump

We now connect the cohomological objects to the dynamics of deep diffusion. The continuous-depth forward propagation obeys $\bar { \partial _ { t } f } = - \bar { \mathcal { L } } _ { 0 } f$ , with solution $f ( t ) = \dot { e } ^ { - t \mathcal { L } _ { 0 } } f ( 0 )$ , where ${ \mathcal L } _ { 0 } = \delta _ { { \mathcal F } } ^ { * } \delta _ { { \mathcal F } }$ is the 0-cochain sheaf Laplacian. $\mathrm { A s } ~ { \mathcal { L } } _ { 0 }$ is non-negative and self-adjoint, the heat operator satisfies lim $_ { \ t  \infty } e ^ { - t \mathcal { L } _ { 0 } }  P _ { \ker \mathcal { L } _ { 0 } }$ , where $P _ { \mathrm { k e r } \ : \mathcal { L } _ { 0 } }$ is the projection onto the harmonic space. The topologically stable quantity is the heat supertrace of the Hodge–Dirac pair, which is related to the analysis index of Dirac operator $D _ { \mathcal { F } }$ (see Appendix C.1 for its full construction.) Let $D _ { \mathcal { F } } = \left( \begin{array} { c c } { { 0 } } & { { \delta _ { \mathcal { F } } ^ { * } } } \\ { { \delta _ { \mathcal { F } } } } & { { 0 } } \end{array} \right)$ denote the Dirac operator on $\mathcal { C } ^ { 0 } ( G ; \mathcal { F } ) \oplus \mathcal { C } ^ { 1 } ( G ; \mathcal { F } )$ , with chiral components

$$
D _ { \mathcal { F } } ^ { + } = \delta _ { \mathcal { F } } : \mathcal { C } ^ { 0 } ( G ; \mathcal { F } ) \to \mathcal { C } ^ { 1 } ( G ; \mathcal { F } ) , \qquad D _ { \mathcal { F } } ^ { - } = \delta _ { \mathcal { F } } ^ { * } : \mathcal { C } ^ { 1 } ( G ; \mathcal { F } ) \to \mathcal { C } ^ { 0 } ( G ; \mathcal { F } ) .
$$

The "raw" analytic index is defined by

$$
\mathrm { i n d } ( D _ { \mathcal { F } } ^ { + } ) : = \mathrm { d i m } \ker D _ { \mathcal { F } } ^ { + } - \mathrm { d i m } \ker D _ { \mathcal { F } } ^ { - } = \mathrm { d i m } \ker \delta _ { \mathcal { F } } - \mathrm { d i m } \ker \delta _ { \mathcal { F } } ^ { * } .
$$

By the discrete Hodge decomposition, this becomes ind $( D _ { \mathcal { F } } ^ { + } ) = \dim H ^ { 0 } ( \mathcal { F } ) - \dim H ^ { 1 } ( \mathcal { F } )$

Proposition 4 (Topological Index and Index Jump; proof in Appendix C.2). Let $\mathcal { F }$ be a rank-r sheaf on a finite graph $G = ( V , E )$ . Then the topological index is:

$$
\begin{array} { r } { \mathrm { i n d } ( D _ { \mathcal { F } } ^ { + } ) = \mathrm { d i m } \mathcal { C } ^ { 0 } ( \mathcal { F } ) - \mathrm { d i m } \mathcal { C } ^ { 1 } ( \mathcal { F } ) = r \chi ( G ) . } \end{array}
$$

depending only on the rank and the Euler characteristic $\chi ( G )$ . In particular, $i f \xi$ is the Euclidean sheafofrank $n ,$ ind $( D _ { \xi } ^ { + } ) = n \chi ( G )$

Consequently, The indexjump ind $( D _ { \mathcal { F } } ^ { + } ) - \mathrm { i n d } ( D _ { \xi } ^ { + } )$ ) between the two sheaves then gives the harmoniccapacity difference: dim $H ^ { 0 } ( { \mathcal { F } } ) -$ dim $H ^ { 0 } ( \xi ) \stackrel { , } { = } \dim H ^ { 1 } ( { \mathcal F } ) - \dim H ^ { 1 } ( \xi ) + ( r - n ) \chi ( G )$

Remark 3 (Index jump alone is insufficient). Proposition 4 implies $\Delta h ^ { 0 } ( { \mathcal F } , \xi ) = \Delta _ { \mathrm { i n d } } ( { \mathcal F } , \xi ) +$ $\Delta h ^ { 1 } ( { \mathcal { F } } , \xi )$ , where $\Delta h ^ { i } : = \cdot$ dim $H ^ { i } ( { \mathcal { F } } ) -$ dim $H ^ { i } ( \xi )$ . The raw index jump $\Delta _ { \mathrm { i n d } }$ alone cannot control $H ^ { 0 }$ because the Euler term $( r - n ) \chi ( G )$ is often negative on cyclic graphs. Thus, the degree-one correction $\Delta h ^ { 1 }$ is essential.

Example 1 (Negative control). The Euclidean sheaf ξ with trivial transport $( \mathcal { F } _ { u  e } ~ = ~ \mathcal { F } _ { v  e } )$ yields $\mathbf { \bar { \mathit { H } } } ^ { 0 } ( \xi ) = \mathbf { \bar { \ s p a n } } \{ \mathbf { 1 } _ { V } \} \otimes \mathbb { R } ^ { n }$ . Although dim $H ^ { 0 } ( \xi ) = n$ scales with feature dimension, its harmonic modes are merely channel-wise constant signals that still collapse node distinctions. Both IdentitySheaf and InverseSheaf exhibit this trivial inflation (see Section 6, Figure 1).

As shown, large stalks or raw index jumps often merely reflect trivial channel-wise replication rather than genuine preservation of node-level geometry. The truly anti-oversmoothing requires our corrected framework: combining the index jump with the degree-one heat trace correction $\bar { \Delta } h ^ { 1 }$ , and measuring excess capacity via the relative quotient $H ^ { 0 } ( \mathcal { F } ) \breve { / } \iota _ { * } H ^ { 0 } ( \xi )$ . We formalize this geometric criterion next.

## 4.2 From Index Comparison to Genuine Harmonic Inclusion

We now state the core result of this work. The detailed heat trace correction derivation and holonomyfixed lower bound are deferred to Appendix C.4.

Theorem 5 (Genuine harmonic inclusion). Let G be a connected graph. Let ξ be the Euclidean sheaf of rank n and $\mathcal { F }$ a rank-r sheaf, both constructed within the framework of Section 3. Suppose a sheaf homomorphism $\iota : \xi \to \mathcal { F }$ induces an inclusion on cohomology, i.e. $\iota _ { * } : H ^ { 0 } ( \xi ) \hookrightarrow H ^ { \hat { 0 } } ( \mathcal { F } )$

Let $\Delta _ { \mathrm { i n d } } ( \mathcal { F } , \xi ) : = \mathrm { i n d } ( D _ { \mathcal { F } } ^ { + } ) - \mathrm { i n d } ( D _ { \xi } ^ { + } )$ be the indexjump and $\begin{array} { r } { \Delta _ { \mathrm { S t r } } ^ { ( 1 ) } ( \mathcal { F } , \xi ) : = \operatorname* { l i m } _ { t  \infty } [ \mathrm { T r } ( e ^ { - t L _ { \mathcal { F } } ^ { 1 } } ) -  } \end{array}$ $\operatorname { T r } ( e ^ { - t L _ { \xi } ^ { 1 } } ) ]$ the degree-one heat trace correction. Assume thefollowing conditions on $\mathcal { F } \colon$

• (Dimensional control). The restriction maps determine a holonomy representation $\rho :$ $\pi _ { 1 } ( G , v _ { 0 } ) \to \operatorname { A u t } ( { \mathcal { F } } ( v _ { 0 } ) )$ , where $\pi _ { 1 } ( G , v _ { 0 } )$ is the fundamental group of G at basepoint v and $\mathrm { A u t } ( \mathcal { F } ( v _ { 0 } ) )$ is the group oflinear automorphisms ofthe stalk $\mathcal { F } ( v _ { 0 } )$ . Itsfixed subspace is

$$
W : = \bigcap _ { \gamma \in \pi _ { 1 } ( G , v _ { 0 } ) } \ker ( \rho ( \gamma ) - I ) \subseteq { \mathcal { F } } ( v _ { 0 } ) , \qquad \dim W = k .
$$

Moreover, the holonomy is non-trivial on thefull stalk, i.e $k < \mathrm { r a n k } ( \mathcal { F } )$

• (Capacity control). At least one ofthefollowing certificates holds:

$$
( a ) \Delta _ { \mathrm { i n d } } ( { \mathcal F } , \xi ) + \Delta _ { \mathrm { S t r } } ^ { ( 1 ) } ( { \mathcal F } , \xi ) \geq 1 ; \qquad ( b ) \dim W > \dim H ^ { 0 } ( \xi ) .
$$

Then dim $H ^ { 0 } ( { \mathcal { F } } ) > \dim H ^ { 0 } ( \xi )$ and the harmonic quotient is nontrivial: $H ^ { 0 } ( { \mathcal F } ) / \iota _ { * } H ^ { 0 } ( { \xi } ) \not = 0 .$

That $i s , \mathcal { F }$ exhibits strictly greater non-dissipative harmonic capacity than the Euclidean baseline. This excess originatesfrom non-trivial holonomy rather than trivial higher-rank channel enlargement, ensuring F provably resists Euclidean-scale over-smoothing.

Remark 4. For small $\beta _ { 1 } ( G )$ , after adding the "local system conditions" in Appendix B.5, certificate (a) admits a simplified explicit form: $\Delta _ { \mathrm { i n d } } ( \mathcal { F } , \xi ) \geq ( n - k ) \beta _ { 1 } ( G ) + 1$ . Given its limited attainability, future iterations should refine $\beta _ { 1 } ( G )$ by synthesizing global topological invariants intrinsic to the graph geometry.

Computability establishes certificate practicability. Specifically, $\Delta _ { \mathrm { i n d } }$ follows by definition, $\Delta _ { \mathrm { { S t } 1 } }$ reduces to the discrete Laplacian trace, and (b) just requires cocycle traversal across all vertices in a simple structured graph.

## 4.3 Case Study: the SPDSheaf

We now illustrate Theorem 5 through a concrete comparison between an SPD sheaf and the Euclidean baseline (details in Appendix C.5).

Recall that we require a sheaf $\mathcal { F }$ equipped with stalks, restriction maps, and an admissible pairing on cochains that defines an adjoint $\delta _ { \mathcal { F } } ^ { * }$ and the Laplacian $\mathcal { L } _ { 0 , \mathcal { F } } : = \delta _ { \mathcal { F } } ^ { * } \delta _ { \mathcal { F } } ,$ with the Hodge decomposition $H ^ { 0 } ( \mathcal { F } ) \cong \ker \mathcal { L } _ { 0 , \mathcal { F } }$ . Moreover, if there exists a sheaf homomorphism $\iota : \xi \to \mathcal { F }$ , the holonomy induced by the restriction maps is non-trivial and the index jump admits an explicit lower bound, Theorem 5 then asserts that the strict harmonic inclusion $H ^ { 0 } \bar { ( } \mathcal { F } ) \bar { / } \iota _ { * } H ^ { 0 } ( \xi ) \not = 0$ holds.

Consider the SPD sheaf $\mathcal { F } _ { \mathrm { S P D } }$ Peng et al. [2026], whose stalks are $\mathrm { S P D } _ { n }$ with congruence-type restriction maps. These determine a coboundary operator, admissible pairing, adjoint, and Hodge Laplacian satisfying the Hodge decomposition Peng et al. [2026]. It remains to verify the three conditions of Theorem 5:

(1) Global section preserving embedding. The Euclidean-to-SPD embedding provides a sheaf morphism $\iota : \xi \to \mathcal { F } _ { \mathrm { S P D } }$ that induces an injection on global sections: $\iota _ { * } : H ^ { 0 } \overline { { ( } } \xi \dot { ) } \hookrightarrow H ^ { 0 } ( \mathcal { F } _ { \mathrm { S P D } } )$ Euclidean global sections are thus preserved in the SPD harmonic space through a concrete embedding, making the subsequent comparison structural rather than merely dimensional.

(2) Non-trivial holonomy. The restriction maps induce congruence-type transports, and it generates non-trivial holonomy representation, ensuring the holonomy-fixed subspace $W \subsetneq { \mathcal { F } } _ { \mathrm { S P D } } ( v _ { 0 } )$

(3) Index comparison. Since $\mathcal { F } _ { \mathrm { S P D } }$ has a finite-dimensional Laplacian, the heat trace $\mathrm { T r } ( e ^ { - t L _ { \mathcal { F } _ { \mathrm { S P D } } } ^ { 1 } } )$ and its correction $\Delta _ { \mathrm { S t r } } ^ { ( 1 ) }$ are computable and we show in Appendix C.5 that $\Delta _ { \mathrm { i n d } } ( \mathcal { F } , \xi ) + \Delta _ { \mathrm { S t r } } ^ { ( 1 ) } ( \mathcal { F } , \xi ) >$ 0, satisfying the capacity control condition of Theorem 5. The strict enlargement is then measured by $H ^ { 0 } ( \mathcal { F } _ { \mathrm { S P D } } ) \big / \iota _ { * } H ^ { 0 } ( \dot { \xi } )$

All three conditions are satisfied, so Theorem 5 yields $H ^ { 0 } ( { \mathcal { F } } _ { \mathrm { S P D } } ) / \iota _ { * } H ^ { 0 } ( \xi ) \neq 0$ . This is exactly the harmonic enlargement established in Theorem 5: the structural excess is measured as a quotient beyond the embedded image, confirming that Theorem 5 detects genuine geometric anti-oversmoothing capacity, stronger than simply observing that a larger stalk gives a larger kernel (see Section 6.2 for empirical confirmation).

Remark 5 (Role of the SPD case). The SPD sheaf clarifies the boundary between raw index comparison and harmonic-capacity comparison. Despite non-Euclidean stalks, the logarithmic and congruence-based structure yields a finite-dimensional Hodge complex after linearization, making it a concrete non-Euclidean application of Theorem 5.

## 5 Nonlinear Generalization

## 5.1 Index Jump beyond Linearity

Let the preceding analysis relies on the linearity of the cochain complex. When the stalks are nonlinear manifolds $\dot { M _ { v } }$ , the global tools break down: $\begin{array} { r } { \mathcal { C } ^ { 0 } = \prod _ { v } \boldsymbol { M _ { \iota } } } \end{array}$ is a product manifold and δ is no longer linear. The global section condition $\delta f = \mathbf { 0 }$ defines a nonlinear harmonic set ${ \mathcal { H } } : = \{ f \in { \bar { \mathcal { C } } } ^ { 0 } : \delta f = \mathbf { 0 } \}$ rather than a linear kernel ker δ. Correspondingly, the diffusion flow $\partial _ { t } f = - L _ { 0 } f$ generalises to a nonlinear gradient flow on $\mathcal { C } ^ { 0 }$ that is no longer generated by a traceclass operator. As a result, the global spectral tools underpinning the linear index theorem no longer apply (details in Appendix D).

The correct replacement is local: rather than requiring the entire cochain complex to be linear, one linearises the problem at each harmonic state $f \in \mathcal H$ by passing to the tangent space $T _ { f } \mathcal { C } ^ { 0 }$ , a vector space. In this sense, the nonlinear framework is a locally linearised version of the index-jump theory developed in the preceding sections. Concretely, around a harmonic state $f \in \mathcal { H }$ , the nonlinear compatibility map $\delta$ can be linearised. Its differential $d \delta _ { f } : T _ { f } \mathcal { C } ^ { 0 } \longrightarrow T _ { 0 } \mathcal { C } ^ { 1 }$ serves as the tangentspace analogue of the linear coboundary operator, giving rise to the tangent complex at $f .$ . The global linear cohomology groups are thus replaced by their local tangent analogues:

$$
H ^ { 0 } ( { \mathcal { F } } ) \ \sim \ H _ { f } ^ { 0 } : = \ker d \delta _ { f } = T _ { f } { \mathcal { H } } , \qquad H ^ { 1 } ( { \mathcal { F } } ) \  \ H _ { f } ^ { 1 } : = \operatorname { c o k e r } d \delta _ { f } .
$$

Correspondingly, The nonlinear analogue of the index becomes:

$$
\mathrm { i n d } ( D _ { \mathcal { F } } ^ { + } ) \  \ \mathrm { i n d } _ { f } ( d \delta ) : = \mathrm { d i m } \mathrm { k e r } d \delta _ { f } - \mathrm { d i m } \mathrm { c o k e r } d \delta _ { f } .
$$

Thus the nonlinear global-section capacity is no longer measured by a global kernel dimension, but by the local dimension of the harmonic submanifold dim $T _ { f } { \mathcal { H } } =$ dim ker $d \delta _ { f }$ . Accordingly, harmonic-capacity enlargement becomes a local increase of this tangent dimension rather than a global dimension jump of vector spaces; in this sense, the nonlinear framework is a locally linearised version of the index-jump theory developed in the preceding sections.

Therefore, the index–heat trace criterion applies locally after tangent linearization. At a harmonic state $f \in \mathcal H$ , the nonlinear coboundary δ has differential $d \delta _ { f } : T _ { f } { \overline { { C } } } ^ { 0 } \to T _ { \delta f } C ^ { 1 }$ with local analytic index ind $_ f ( d \delta )$ := dim ker dδ<sub>f</sub> − dim coker $d \delta _ { f }$ . The local analogue of Theorem 5 is $\Delta$ dim ${ \dot { H _ { f } ^ { 0 } } } =$ $\Delta _ { \mathrm { i n d } , f } + \Delta _ { \mathrm { S t r } , f } ^ { ( 1 ) }$ , where $\Delta _ { \mathrm { S t r } , f } ^ { ( 1 ) }$ involves calculation of trace for analytic operator which we have discussed before. Thus, for a nonlinear sheaf, it suffices to construct $d \delta _ { f } ,$ equip the tangent cochains with admissible pairings, define the adjoint and tangent Hodge Laplacians, and verify the tangent Hodge decomposition. We realize this program through the SPD-based GyroSheaf framework below.

## 5.2 Case Study: the GyroSheaf

From SPD matrices to the Poincaré ball. We first map each SPD feature $P _ { v } \in \mathrm { S P D } _ { n }$ to the Poincaré ball $B _ { n }$ via the Cayley transformation:

$$
X _ { v } : = \Psi ( P _ { v } ) = ( P _ { v } - I ) ( P _ { v } + I ) ^ { - 1 } \in \mathcal { B } _ { n } : = \{ X = X ^ { \top } : \| X \| < 1 \} .
$$

This yields a bounded gyrovector domain in which edge-wise inconsistencies can be measured through gyro-algebraic operations while preserving SPD geometry.

Definition 6 (Restriction maps). A gyro-valued sheaf $( G , { \mathcal { F } } )$ on a graph $G = ( V , E )$ assigns the Cayley ball $\mathcal { F } ( v ) = B _ { n }$ to each vertex v and the symmetric-matrix space $\mathbf { \bar { \mathcal { F } } } ( e ) = \mathrm { S y m } _ { n }$ to each edge $e .$ For each incident pair $v \in e ,$ the restriction map $\mathcal { F } _ { v \to e } : \mathcal { F } ( v ) \stackrel { \cdot } { \to } \mathcal { F } ( e )$ is defined as:

$$
X _ { v e } : = \mathcal { F } _ { v \to e } ( X _ { v } ) = M _ { v e } X _ { v } M _ { v e } ^ { \top } , \qquad M _ { v e } \in O ( n ) .
$$

The differential $d \mathcal { F } _ { v \to e } : T _ { X _ { v } } \mathcal { F } ( v ) \to T _ { X _ { v e } } \mathcal { F } ( e )$ where the tangent spaces both canonically identified with $\operatorname { S y m } _ { n }$ and the Frobenius adjoint act as:

$$
d \mathscr { F } _ { v \to e } ( G ) = M _ { v e } G M _ { v e } ^ { \intercal } \quad G \in T _ { X _ { v } } \mathscr { F } ( v ) ; \qquad ( d \mathscr { F } _ { v \to e } ) ^ { \ast } ( G ^ { \prime } ) = M _ { v e } ^ { \intercal } G ^ { \prime } M _ { v e } \quad G ^ { \prime } \in T _ { X _ { v } } \mathscr { F } ( e ) .
$$

Gyro-Coboundary operator. Since matrix multiplication is non-commutative, we adopt the symmetrised gyro-subtraction to measure edge-wise inconsistency:

$$
A \ominus _ { s y m } B : = ( I - A B ) ^ { - 1 / 2 } ( A - B ) ( I - B A ) ^ { - 1 / 2 } \in \mathrm { S y m } _ { n } , \quad A , B \in \mathcal { B } _ { n } .
$$

Definition 7 (Gyro-compatible Coboundary Operator). The coboundary operator $\delta : C ^ { 0 } ( G ; { \mathcal { F } } ) \longrightarrow$ $C ^ { 1 } ( G ; { \mathcal { F } } )$ , with $\begin{array} { r } { C ^ { 0 } ( G ; \mathcal { F } ) \ : = \ : \prod _ { v \in V } \ d B _ { n } } \end{array}$ and $C ^ { 1 } ( G ; { \mathcal { F } } ) = \bigoplus _ { e \in E } { \dot { \mathrm { S y m } } } ( n )$ , is defined on each oriented edge $\boldsymbol { e } = ( u , v )$ as:

$$
\delta ( X ) _ { e } = { \mathcal F } _ { v \to e } ( X _ { v } ) \ominus { \mathcal F } _ { u \to e } ( X _ { u } ) = X _ { v e } \ominus _ { s y m } X _ { u e } = S _ { v e } ^ { 1 / 2 } ( X _ { v e } - X _ { u e } ) S _ { u e } ^ { 1 / 2 } \in \mathrm { S y m } _ { n } = { \mathcal F } ( e ) ,
$$

where we denote $S _ { v e } : = ( I - X _ { v e } X _ { u e } ) ^ { - 1 }$ and $S _ { u e } : = ( I - X _ { u e } X _ { v e } ) ^ { - 1 } = S _ { v e } ^ { \top }$ for notational brevity.

Differential and adjoint. Since δ is nonlinear, the adjoint appearing in the gyro-sheaf Laplacian is the Frobenius adjoint of the linearised coboundary at the current state X. Its differential and Frobenius adjoint have types $D \delta ( X ) : \bigoplus _ { v \in V } T _ { X _ { v } } { \mathcal { B } } _ { n } { \stackrel { \cdot } { \longrightarrow } } C ^ { 1 } ( G ; { \mathcal { F } } )$ and $D \delta ( X ) ^ { * } : C ^ { 1 } ( G ; { \mathcal { F } } ) \longrightarrow$ $\textstyle \bigoplus _ { v \in V } T _ { X _ { v } } B _ { n }$ . Their closed-form expressions involve the Fréchet derivative of the principal matrix inverse square root, and are deferred to Appendix D.2. In implementation, we compute the gyro-sheaf Laplacian directly as the gradient of the Dirichlet energy.

Definition 8 (Dirichlet energy). Let $( \delta X ) _ { e } = \Delta _ { e }$ . The energy on edge e and the total energy are defined as $\begin{array} { r } { \mathcal { E } _ { e } ( X ) : = \frac { 1 } { 2 } \| \Delta _ { e } \| _ { F } ^ { 2 } ; \mathcal { E } ( X ) : = \sum _ { e \in E } \mathcal { E } _ { e } ( X ) } \end{array}$

Definition 9 (Gyro-sheaf Laplacian). For a directed edge $\boldsymbol { e } = \left( u , v \right)$ with v as the head, the gyrosheaf Laplacian at vertex v is the gradient of the total inconsistency energy $( \mathcal { L } X ) _ { v } : = \nabla _ { X _ { v } } \mathcal { E } ( \bar { X } ) \in$ $T _ { X _ { v } } B _ { n } \stackrel { . } { = } \mathrm { S y m } ( n )$ . Recall that in the linear sheaf setting the coboundary δ is a linear and the sheaf Laplacian equals ${ \dot { \cal C } } = \delta ^ { * } \delta .$ . In our setting, δ is nonlinear, we replace δ with its tangent map $D \delta ( X )$ and the Laplacian generalises accordingly to $\mathcal { L } X = \left( D \delta ( X ) \right) ^ { * } D \delta ( X )$ . More explicitly,

$$
( \mathcal { L } X ) _ { v } = \sum _ { e = ( u , v ) } ( d \mathcal { F } _ { v  e } ) ^ { \ast } ( d \delta _ { v e } ) ^ { \ast } \delta ( X ) _ { e } + \sum _ { e = ( v , w ) } ( d \mathcal { F } _ { v  e } ) ^ { \ast } ( d \delta _ { v e } ) ^ { \ast } \delta ( X ) _ { e }
$$

Proposition 6 (Hodge-type Decomposition for the Nonlinear GyroSheaf). Let $\begin{array} { r l } { T _ { X } { \mathcal { C } } ^ { 0 } } & { { } = } \end{array}$ $\textstyle \bigoplus _ { v \in V } T _ { X _ { v } } B _ { \iota }$ and $T _ { \delta ( X ) } \bar { \mathcal { C } } ^ { \mathrm { { i } } } = \bigoplus _ { e \in E } \mathbf { \bar { \mathcal { T } } } _ { \delta ( X ) _ { e } } \mathcal { B } _ { \epsilon }$ be the tangent-space direct sums, and let $D \delta _ { X }$ $T _ { X } \bar { \mathcal { C } } ^ { 0 } \to T _ { \delta ( X ) } \mathcal { C } ^ { 1 }$ be the linearised coboundary. Under the Frobenius pairing $\langle A , B \rangle _ { F } = \operatorname { t r } ( A ^ { \top } B )$ $D \delta _ { X }$ is a linear operator between Hilbert spaces. Then:

$$
T _ { \delta ( X ) } { \mathcal C } ^ { 1 } = \mathrm { i m } ( D \delta _ { X } ) \oplus \mathrm { k e r } ( D \delta _ { X } ^ { * } ) , \qquad T _ { X } { \mathcal C } ^ { 0 } = \mathrm { i m } ( D \delta _ { X } ^ { * } ) \oplus \mathrm { k e r } ( D \delta _ { X } ) .
$$

(detalis in Appendix D.3)

Remark 6. The GyroSheaf verifies the local Hodge-theoretic structure required by the nonlinear index–heat trace framework: it constructs the tangent complex, defines the tangent Hodge Laplacian, and proves the corresponding tangent Hodge decomposition, yielding a well-defined local harmonic space. Moreover, since its restriction maps act by $O ( n )$ -congruence on $\operatorname { S y m } _ { n }$ (Definition 6), the tangent complex at any $f \in \mathcal H$ inherits the holonomy and index data of the linearised SPD complex (Section 4.3), so the conditions of Theorem 5 are satisfied by identical verification. Experiments in Section 6 and Appendix F confirm the predicted anti-oversmoothing behavior.

## 6 Experiments

Our experiments empirically validate the predictions of the index-theoretic framework. The design is structured as a progressive verification of Theorem 5: we move from GCNs, which correspond to a trivial sheaf (Baseline), through sheaf models that violate the theorem’s conditions (Tier 1), to sheaf models that satisfy them with increasing index jump (Tiers 2–3).This progression directly tests whether the theorem govern anti-oversmoothing capacity as predicted. Datasets, models, training protocol, and oversmoothing measures are detailed in Appendix E. Code is available here.

![](images/cb17853a6256185a5341bd52cd3232a6122970501d56e3dbda574a2be41f730a.jpg)  
Figure 1: Normalized Dirichlet energy (top) and MAD (bottom) on Cora, Citeseer, and Texas in the untrained regime. Curves and shaded bands show per-seed mean and min–max range over five seeds.

## 6.1 Models

Baseline: rank-1 trivial sheaf. GCN Kipf and Welling [2017], GAT Velickovic et al. [2018], Graph-SAGE Hamilton et al. [2017]. Standard GNN propagation by the graph Laplacian $L _ { G }$ , corresponding to the rank-1 trivial sheaf (scalar stalk $\mathcal { F } ( v ) = \mathbb { R }$ , identity restriction maps). No index jump, no sheaf structure. Oversmoothing is the default.

Tier 1: sheafstructure with index jump, but conditions violated. Two negative controls that possess sheaf structure with stalk dimension d and a nonzero index jump $\Delta \mathrm { i n } \mathbf { \bar { d } } = ( d - 1 ) \chi ( G )$ over the baseline, as exemplified after Prop 4, are predicted by Theorem 5 to fail at harmonic enlargement:

• IdentitySheaf Caralt et al. [2026]: Consider the sheaf in which every restriction map is the identity $I _ { d } .$ The parallel transport along any edge is therefore also the identity, so the holonomy representation is trivial: $\rho ( \gamma ) ~ = ~ I _ { d }$ for every $\gamma ~ \in ~ \pi _ { 1 } ( G , v _ { 0 } )$ Consequently, $\begin{array} { r } { W = \bigcap _ { \gamma } \ker ( \rho ( \gamma ) - I ) = \mathcal { F } ( v _ { 0 } ) } \end{array}$ , i.e. the holonomy-fixed subspace equals the entire stalk. Moreover, the trivial holonomy yields a global trivialisation of sheaf structure, so the harmonic space reduces to a rank-d tensor copy of the scalar Euclidean baseline (channel-wise constant signals). The hypotheses of Theorem 5 are therefore not satisfied: Condition 1 (Dimensional control) requires non-trivial holonomy.

• InverseSheaf: For each edge e, the restriction maps from the two endpoints to e are the same learned invertible matrix $M _ { e }$ . Although the restriction maps appear non-trivial, the parallel transport from one endpoint to the other is $M _ { e } ^ { - 1 } M _ { e } = I _ { d }$ , which renders the holonomy around every cycle trivial. There situation thus reduces to the IdentitySheaf case above.

Tier 2: conditions satisfied, Euclidean stalks. DiagSheaf, BundleSheaf, and GeneralSheaf Bodnar et al. [2022], with diagonal, orthogonal, and general restriction maps respectively. Same stalk dimension d and index jump as Tier 1, but their learned restriction maps can develop holonomy in the middle ground required by Condition 1, non-trivial yet preserving enough fixed directions, while also satisfying Condition 2.

Tier 3: conditions satisfied, higher-dimensional non-Euclidean stalks. SPDSheaf Peng et al. [2026] and our GyroSheaf, whose manifold-valued stalks have effective dimension $r = d ( d { + } 1 ) / 2 > d .$ which enlarges the ambient space in which holonomy-fixed directions may occur. On cyclic graphs, this yields a strict enlarge of relevant quantity $\Delta h ^ { 0 } = \Delta _ { \mathrm { i n d } } + \Delta _ { \mathrm { S t r } } ^ { ( 1 ) }$ . Both conditions are satisfied.

## 6.2 Untrained Regime: Structural Diagnostic

Figure 1 reports normalized Dirichlet energy and MAD on Cora, Citeseer, and Texas. The trajectories confirm the four-level ordering predicted by Theorem 5. Baseline and both Tier 1 negative controls collapse by six to eight orders of magnitude within 128 layers—despite Tier 1’s enlarged stalks and nontrivial sheaf-like structure, they decay to the same level as the rank-1 baselines, confirming that index jump without the holonomy conditions is insufficient. The Tier 2 learnable sheaves substantially delay the decay (with BundleSheaf slightly faster, consistent with the orthogonal constraint limiting holonomy flexibility), while the Tier 3 geometric sheaves maintain both metrics essentially flat across all depths and datasets, reflecting the combined effect of satisfied conditions and a larger index jump.

![](images/1348212a86c9664cdbebb9cdcc19ef317ca064a01ad7996e47ff859dfb38ce07.jpg)

![](images/bb22c621ea3bf2462177c432ff340cb5fc5daa632e6d8cbf9a920f419cbdd526.jpg)

![](images/4c60ed6d80b9f98b81ddece4754ec1b27d047af10b0394a8c9e22e83158095b4.jpg)  
Figure 2: Normalized Dirichlet energy (left), MAD (middle), and test accuracy (right) on Cora in trained regime. Legend and shading as in Figure 1.

## 6.3 Trained Regime: End-to-End Confirmation

Figure 2 reports normalized Dirichlet energy, MAD, and test accuracy on Cora across depths. Baseline accuracy drops sharply beyond L=8, while all sheaf models, including Tier 1 controls, maintain ∼80% accuracy, confirming that the enlarged harmonic capacity of Tiers 2–3 does not come at the cost of finite-depth performance. The structural distinctions between Tiers 1 and 2, clearly separated in the untrained regime, are masked by finite-depth optimization in the trained regime, which is expected: the operator-level criterion of Theorem 5 characterizes intrinsic capacity in the infinite-depth limit, complementary to finite-depth trained behavior. We also note that GCN’s Dirichlet energy does not decay substantially despite its accuracy degradation, consistent with the bias-induced energy plateau in Rusch et al. [2023]’s Figure 3, further illustrating that trained-regime energy trajectories do not directly reflect operator-level oversmoothing behavior. Together, these observations confirm that the untrained diagnostic is the appropriate probe for the operator-level harmonic capacity of Theorem 5.

## 7 Conclusion

We presented the first relative criterion for anti-oversmoothing capacity in Sheaf Neural Networks. Theorem 5 establishes that a positive index jump, combined with natural conditions on holonomy and cohomological structure, guarantees genuine harmonic enlargement beyond the Euclidean baseline. We extended this framework to nonlinear stalks via local linearisation and instantiated it with GyroSheaf. Experiments across ten models confirm the criterion.

## References

Federico Barbero, Cristian Bodnar, Haitz Sáez de Ocáriz Borde, Michael M. Bronstein, Petar Velickovic, and Pietro Liò. Sheaf neural networks with connection laplacians. In Alexander Cloninger, Timothy Doster, Tegan Emerson, Manohar Kaul, Ira Ktena, Henry Kvinge, Nina Miolane, Bastian Rice, Sarah Tymochko, and Guy Wolf, editors, Topological, Algebraic and Geometric Learning Workshops 2022, 25- 22 July 2022, Virtual, Proceedings of Machine Learning Research, pages 28–36. PMLR, 2022. URL https://proceedings.mlr.press/v196/barbero22a.html.

Claudio Battiloro, Lucia Testa, Lorenzo Giusti, Stefania Sardellitti, Paolo Di Lorenzo, and Sergio Barbarossa. Generalized simplicial attention neural networks. IEEE Trans. Signal Inf. Process. over Networks, 10:833–850, 2024. doi: 10.1109/TSIPN.2024.3485473. URL https://doi.org/10.1109/TSIPN.2024.3485473.

Nicole Berline, Ezra Getzler, and Michele Vergne. Heat Kernels and Dirac Operators. Grundlehren Text Editions. Springer-Verlag Berlin Heidelberg, 2004. ISBN 978-3-540-20062-8. doi: 10.1007/978-3-540-20062-8.

Cristian Bodnar, Francesco Di Giovanni, Benjamin Paul Chamberlain, Pietro Lió, and Michael M. Bronstein. Neural sheaf diffusion: A topological perspective on heterophily and oversmoothing in gnns. In Sanmi Koyejo, S. Mohamed, A. Agarwal, Danielle Belgrave, K. Cho, and A. Oh, editors, Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022,

New Orleans, LA, USA, November 28 - December 9, 2022, 2022. URL http://papers.nips.cc/paper\_ files/paper/2022/hash/75c45fca2aa416ada062b26cc4fb7641-Abstract-Conference.html.

Ferran Hernandez Caralt, Mar Gonzàlez i Català, Adrián Bazaga, and Pietro Liò. On the necessity of learnable sheaf laplacians. CoRR, abs/2603.05395, 2026. doi: 10.48550/ARXIV.2603.05395. URL https://doi. org/10.48550/arXiv.2603.05395.

Ben Chamberlain, James Rowbottom, Maria I. Gorinova, Michael M. Bronstein, Stefan Webb, and Emanuele Rossi. GRAND: graph neural diffusion. In Marina Meila and Tong Zhang, editors, Proceedings ofthe 38th International Conference on Machine Learning, ICML 2021, 18-24 July 2021, Virtual Event, Proceedings of Machine Learning Research, pages 1407–1418. PMLR, 2021. URL http://proceedings.mlr.press/ v139/chamberlain21a.html.

Ming Chen, Zhewei Wei, Zengfeng Huang, Bolin Ding, and Yaliang Li. Simple and deep graph convolutional networks. In Proceedings ofthe 37th International Conference on Machine Learning, ICML 2020, 13-18 July 2020, Virtual Event, Proceedings of Machine Learning Research, pages 1725–1735. PMLR, 2020. URL http://proceedings.mlr.press/v119/chen20v.html.

Iulia Duta, Giulia Cassarà, Fabrizio Silvestri, and Pietro Lió. Sheaf hypergraph networks. In Alice Oh, Tristan Naumann, Amir Globerson, Kate Saenko, Moritz Hardt, and Sergey Levine, editors, Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023, 2023. URL http://papers.nips.cc/paper\_ files/paper/2023/hash/27f243af2887d7f248f518d9b967a882-Abstract-Conference.html.

Francesco Di Giovanni, James Rowbottom, Benjamin Paul Chamberlain, Thomas Markovich, and Michael M. Bronstein. Graph neural networks as gradient flows. CoRR, abs/2206.10991, 2022. doi: 10.48550/ARXIV. 2206.10991. URL https://doi.org/10.48550/arXiv.2206.10991.

William L. Hamilton, Zhitao Ying, and Jure Leskovec. Inductive representation learning on large graphs. In Isabelle Guyon, Ulrike von Luxburg, Samy Bengio, Hanna M. Wallach, Rob Fergus, S. V. N. Vishwanathan, and Roman Garnett, editors, Advances in Neural Information Processing Systems 30: Annual Conference on Neural Information Processing Systems 2017, December 4-9, 2017, Long Beach, CA, USA, pages 1024–1034, 2017. URL https://proceedings.neurips.cc/paper/2017/hash/ 5dd9db5e033da9c6fb5ba83c7a7ebea9-Abstract.html.

Jakob Hansen and Thomas Gebhart. Sheaf neural networks. CoRR, abs/2012.06333, 2020. URL https: //arxiv.org/abs/2012.06333.

Jakob Hansen and Robert Ghrist. Toward a spectral theory of cellular sheaves. J. Appl. Comput. Topol., 3(4):315–358, 2019. doi: 10.1007/S41468-019-00038-7. URL https://doi.org/10.1007/ s41468-019-00038-7.

Thomas N. Kipf and Max Welling. Semi-supervised classification with graph convolutional networks. In 5th International Conference on Learning Representations, ICLR 2017, Toulon, France, April 24-26, 2017, Conference Track Proceedings. OpenReview.net, 2017. URL https://openreview.net/forum?id= SJU4ayYgl.

Qimai Li, Zhichao Han, and Xiao-Ming Wu. Deeper insights into graph convolutional networks for semisupervised learning. In Sheila A. McIlraith and Kilian Q. Weinberger, editors, Proceedings of the Thirty-Second AAAI Conference on Artificial Intelligence, (AAAI-18), the 30th innovative Applications ofArtificial Intelligence (IAAI-18), and the 8th AAAI Symposium on Educational Advances in Artificial Intelligence (EAAI-18), New Orleans, Louisiana, USA, February 2-7, 2018, pages 3538–3545. AAAI Press, 2018. doi: 10.1609/AAAI.V32I1.11604. URL https://doi.org/10.1609/aaai.v32i1.11604.

Jiaxu Liu, Xinping Yi, and Xiaowei Huang. Deephgcn: Toward deeper hyperbolic graph convolutional networks. IEEE Trans. Artif. Intell., 5(12):6172–6185, 2024. doi: 10.1109/TAI.2024.3440223. URL https://doi.org/10.1109/TAI.2024.3440223.

Qi Liu, Maximilian Nickel, and Douwe Kiela. Hyperbolic graph neural networks. In Hanna M. Wallach, Hugo Larochelle, Alina Beygelzimer, Florence d’Alché-Buc, Emily B. Fox, and Roman Garnett, editors, Advances in Neural Information Processing Systems 32: Annual Conference on Neural Information Processing Systems 2019, NeurIPS 2019, December 8-14, 2019, Vancouver, BC, Canada, pages 8228–8239, 2019. URL https://proceedings.neurips.cc/paper/2019/hash/ 103303dd56a731e377d01f6a37badae3-Abstract.html.

Federico López, Beatrice Pozzetti, Steve Trettel, Michael Strube, and Anna Wienhard. Vector-valued distance and gyrocalculus on the space of symmetric positive definite matrices. In Marc’Aurelio Ranzato, Alina Beygelzimer, Yann N. Dauphin, Percy Liang, and Jennifer Wortman Vaughan, editors, Advances in Neural Information Processing Systems 34: Annual Conference on Neural Information Processing Systems 2021, NeurIPS 2021, December 6-14, 2021, virtual, pages 18350–18366, 2021. URL https://proceedings. neurips.cc/paper/2021/hash/98c39996bf1543e974747a2549b3107c-Abstract.html.

Xuan Son Nguyen. The gyro-structure of some matrix manifolds. In Sanmi Koyejo, S. Mohamed, A. Agarwal, Danielle Belgrave, K. Cho, and A. Oh, editors, Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 - December 9, 2022, 2022. URL http://papers.nips.cc/paper\_files/paper/2022/ hash/a9ad92a81748a31ef6f2ef68d775da46-Abstract-Conference.html.

Kenta Oono and Taiji Suzuki. Graph neural networks exponentially lose expressive power for node classification. In 8th International Conference on Learning Representations, ICLR 2020, Addis Ababa, Ethiopia, April 26-30, 2020. OpenReview.net, 2020. URL https://openreview.net/forum?id=S1ldO2EFPr.

Yuhan Peng, Junwen Dong, Yuzhi Zeng, Hao Li, Ce Ju, Huitao Feng, Diaaeldin Taha, Anna Wienhard, and Kelin Xia. Sheaf neural networks on spd manifolds: Second-order geometric representation learning, 2026. URL https://arxiv.org/abs/2604.20308.

T. Konstantin Rusch, Michael M. Bronstein, and Siddhartha Mishra. A survey on oversmoothing in graph neural networks. CoRR, abs/2303.10993, 2023. doi: 10.48550/ARXIV.2303.10993. URL https://doi.org/10. 48550/arXiv.2303.10993.

Abraham Albert Ungar. A Gyrovector Space Approach to Hyperbolic Geometry. Synthesis Lectures on Mathematics & Statistics. Morgan & Claypool Publishers, 2009. ISBN 978-3-031-01268-6. doi: 10.2200/ S00175ED1V01Y200901MAS004. URL https://doi.org/10.2200/S00175ED1V01Y200901MAS004.

Petar Velickovic, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Liò, and Yoshua Bengio. Graph attention networks. In 6th International Conference on Learning Representations, ICLR 2018, Vancouver, BC, Canada, April 30 - May 3, 2018, Conference Track Proceedings. OpenReview.net, 2018. URL https://openreview.net/forum?id=rJXMpikCZ.

Yu Wang and Yi Chang. Enhancing graph neural networks on SPD manifolds via cholesky decomposition. Pattern Recognit., 172:112763, 2026. doi: 10.1016/J.PATCOG.2025.112763. URL https://doi.org/10. 1016/j.patcog.2025.112763.

Keyulu Xu, Chengtao Li, Yonglong Tian, Tomohiro Sonobe, Ken-ichi Kawarabayashi, and Stefanie Jegelka. Representation learning on graphs with jumping knowledge networks. In Jennifer G. Dy and Andreas Krause, editors, Proceedings of the 35th International Conference on Machine Learning, ICML 2018, Stockholmsmässan, Stockholm, Sweden, July 10-15, 2018, Proceedings of Machine Learning Research, pages 5449–5458. PMLR, 2018. URL http://proceedings.mlr.press/v80/xu18c.html.

Kaicheng Zhang, Piero Deidda, Desmond J. Higham, and Francesco Tudisco. Are we measuring oversmoothing in graph neural networks correctly? In International Conference on Learning Representations (ICLR), 2026.

Lingxiao Zhao and Leman Akoglu. Pairnorm: Tackling oversmoothing in gnns. In 8th International Conference on Learning Representations, ICLR 2020, Addis Ababa, Ethiopia, April 26-30, 2020. OpenReview.net, 2020. URL https://openreview.net/forum?id=rkecl1rtwB.

Wei Zhao, Federico López, J. Maxwell Riestenberg, Michael Strube, Diaaeldin Taha, and Steve Trettel. Modeling graphs beyond hyperbolic: Graph neural networks in symmetric positive definite matrices. In Danai Koutra, Claudia Plant, Manuel Gomez Rodriguez, Elena Baralis, and Francesco Bonchi, editors, Machine Learning and Knowledge Discovery in Databases: Research Track - European Conference, ECML PKDD 2023, Turin, Italy, September 18-22, 2023, Proceedings, Part III, Lecture Notes in Computer Science, pages 122–139. Springer, 2023. doi: 10.1007/978-3-031-43418-1\_8. URL https://doi.org/10.1007/978-3-031-43418-1\_8.

Kaixiong Zhou, Xiao Huang, Daochen Zha, Rui Chen, Li Li, Soo-Hyun Choi, and Xia Hu. Dirichlet energy constrained learning for deep graph neural networks. In Marc’Aurelio Ranzato, Alina Beygelzimer, Yann N. Dauphin, Percy Liang, and Jennifer Wortman Vaughan, editors, Advances in Neural Information Processing Systems 34: Annual Conference on Neural Information Processing Systems 2021, NeurIPS 2021, December 6-14, 2021, virtual, pages 21834–21846, 2021. URL https://proceedings.neurips.cc/paper/2021/ hash/b6417f112bd27848533e54885b66c288-Abstract.html.

## A Limitations

While our work introduces the first relative, index-theoretic criterion for anti-oversmoothing capacity in Sheaf Neural Networks and validates it across ten models on both homophilic and heterophilic benchmarks, it presents some limitations and areas for future exploration. Our theoretical framework and experiments focus on standard graph structures; extending the index-theoretic criterion to more general combinatorial domains such as hypergraphs and cell complexes, where the coboundary oper ators and Hodge decomposition have known analogues, is a natural direction, though verifying the holonomy and cohomological conditions of Theorem 5 in these settings would require careful adaptations. Additionally, as our analysis demonstrates, the operator-level criterion of Theorem 5 precisely characterises intrinsic harmonic capacity in the infinite-depth limit, which serves a complementary role to finite-depth trained behaviour—the structural distinctions clearly separated in the untrained regime can be partially masked by finite-depth optimisation, as expected from an operator-theoretic guarantee. Investigating how the index-theoretic criterion interacts with training dynamics to further inform architectural choices in practical finite-depth settings is a promising direction for future work.

## B Details in Section 3

## B.1 Details of 3.1 (Sheaves on Graphs)

To bridge the gap between the continuous fiber bundle $\pi : \mathcal { E }  M$ and the graph-based construction, we formalize the following process.

Description of stalks and maps. The graph $G = ( V , E )$ is viewed as a 1-dimensional cell complex sampled from the manifold M. The node stalks $\mathcal { F } _ { v }$ are the fibers $\pi ^ { - 1 } ( v )$ of the continuous bundle. For an edge $\boldsymbol { e } = \left( u , v \right)$ , the edge stalk $\mathcal { F } _ { e }$ is the fiber over the midpoint of e or a shared representative space. The restriction maps $\mathcal { F } _ { u  e }$ and $\mathcal { F } _ { v  e }$ are the discrete realizations of the connection form ω defined in Section 2.1. Specifically, $\mathcal { F } _ { v  e }$ represents the parallel transport from v onto the edge e.

Cochains and operator grounding. The space of 0-cochains ${ \mathcal { C } } ^ { 0 } ( G , { \mathcal { F } } )$ is the discrete analog of the space of smooth sections $\Gamma ( M , { \bar { \varepsilon } } )$ . A 0-cochain $f \in { \mathcal { C } } ^ { 0 }$ assigns a local feature $f ( v ) \in \bar { \mathcal { F } _ { v } }$ to each vertex. The 1-cochains ${ \mathcal { C } } ^ { 1 } ( { \dot { G } } , { \mathcal { F } } )$ capture the "difference" of these features across edges, which corresponds to $\Omega ^ { 1 } ( M , s ^ { * } \mathcal { V } )$ . The discrete coboundary operator $\delta : { \mathcal { C } } ^ { 0 } \to { \mathcal { C } } ^ { 1 }$ measures the failure of a 0-cochain to be a global parallel section, directly simulating the continuous operator $d _ { 0 } s = s ^ { * } \omega$

## B.2 Details of 3.2 (Coboundary Operator)

This section provides the rigorous formulation of the discrete coboundary operator introduced in Definition 2, emphasizing its role as a measure of local incompatibility on non-linear stalks.

Geometric interpretation of $\delta .$ The discrete coboundary operator $\delta$ serves as the discrete analogue of the connection-induced differential $d _ { 0 } s = s ^ { * } \omega$ . For a 0-cochain $f \in { \mathcal { C } } ^ { 0 } ( G , { \mathcal { F } } )$ (which assigns a local feature $f ( v ) \in \mathcal { F } _ { v }$ to each node), the operator evaluates the failure of $f$ to form a globally parallel section. Along any edge $\boldsymbol { e } = \left( u , v \right)$ , the maps $\mathcal { F } _ { u  e }$ and $\mathcal { F } _ { v  e }$ transport the node features into the common edge stalk $\mathcal { F } _ { e }$ . The quantity $( \delta f ) _ { \epsilon }$ must represent the intrinsic geometric difference between these transported features.

As in the remark after Definition 2, the output $( \delta f ) ,$ technically lies in a general type space with no natural algebra structure on it, the key point of defining δ is how to formulate the “difference” in terms of specific sheaf types. One typical approach is to try to identify the algebraic structure inherent in the space, and another way is utilizing local linearize technique to identify algebraic structure on tangent space.

Example 2 (Linear sheaf coboundary). One can verify above discussion by its behavior on classical linear sheaves. Suppose the stalks $\mathcal { F } _ { v }$ and $\mathcal { F } _ { e }$ are Euclidean vector spaces, and the restriction maps are linear transformations. The difference natively becomes the standard vector difference. In this case, the operator reduces exactly to:

$$
( \delta f ) _ { u  v } = \mathcal { F } _ { v  e } f ( v ) - \mathcal { F } _ { u  e } f ( u )
$$

which is precisely the classical definition of the degree-0 coboundary operator in Euclidian sheaf and other previous work Hansen and Gebhart [2020], Bodnar et al. [2022], Hansen and Ghrist [2019].

## B.3 Proof of Proposition 1 (Energy-Laplacian Correspondence)

We prove the three properties connecting the sheaf Laplacian $\mathcal { L } = \delta ^ { * } \delta$ to the Dirichlet energy and diffusion dynamics.

Proof. • Step 1: Gradient of the discrete energy. Given the admissible pairing $\langle \cdot , \cdot \rangle$ ⟩ on the cochain spaces, the discrete energy is $\begin{array} { r } { \mathcal { E } ( x ) = \frac { 1 } { 2 } \langle \delta \dot { x } , \delta x \rangle } \end{array}$ . We consider the first variation of this energy along an arbitrary test 0-cochain v:

$$
\delta { \mathcal E } ( x ) [ v ] = \left. \frac { d } { d \epsilon } \left( \frac { 1 } { 2 } \langle \delta ( x + \epsilon v ) , \delta ( x + \epsilon v ) \rangle \right) \right| _ { \epsilon = 0 } = \langle \delta x , \delta v \rangle
$$

By the defining property of the adjoint operator $\delta ^ { * }$ , we can move the coboundary operator to the left side of the pairing:

$$
\langle \delta x , \delta v \rangle = \langle \delta ^ { * } \delta x , v \rangle = \langle \mathcal { L } x , v \rangle
$$

Since $\delta { \mathcal E } ( x ) [ v ] = \langle \nabla { \mathcal E } ( x ) , v \rangle$ holds for all variations $v ,$ we conclude that the gradient of the energy functional is exactly $\dot { \nabla } \mathcal { E } ( x ) = \mathcal { L } x$

• Step 2: Measurement of local inconsistency (ii). To see how $\mathcal { L }$ encodes the total inconsistency, we evaluate the inner product of the Laplacian applied to x with x itself:

$$
\langle \mathcal { L } x , x \rangle = \langle \delta ^ { * } \delta x , x \rangle
$$

Applying the adjoint relation again, we obtain:

$$
\langle \delta ^ { * } \delta x , x \rangle = \langle \delta x , \delta x \rangle = \| \delta x \| ^ { 2 }
$$

This confirms that the quadratic form of the Laplacian exactly recovers the accumulated edgewise geometric discrepancies.

• Step 3: Positivity and diffusion stability (iii). From Step 2, we have $\langle \mathcal { L } x , x \rangle = \| \delta x \| ^ { 2 } \ge 0$ for any $x \in \mathcal { C } ^ { 0 }$ . Thus, $\mathcal { L }$ is inherently positive semi-definite. When the system evolves according to the diffusion equation $\partial _ { t } x _ { t } = - \mathcal { L } x _ { t }$ , the time evolution of the energy is given by:

$$
\frac { d } { d t } \mathcal { E } ( x _ { t } ) = \langle \nabla \mathcal { E } ( x _ { t } ) , \partial _ { t } x _ { t } \rangle = \langle \mathcal { L } x _ { t } , - \mathcal { L } x _ { t } \rangle = - \| \mathcal { L } x _ { t } \| ^ { 2 } \leq 0
$$

This strictly non-positive derivative guarantees that the diffusion process dissipates the inconsistency energy monotonically, driving the signal toward a stable harmonic state where $\mathcal { L } x _ { t } = 0$

## B.4 Proof of Theorem 2 (0th-order Discrete Hodge Decomposition)

We state the Hodge decomposition for 0-cochains on discrete sheaves, where the cochain pairing and its adjoint are already defined. This proof establishes the identity between the discrete harmonic space and the space of global sections by demonstrating the equivalence of their kernels.

Proof. • Step 1: The identity ker $\begin{array} { r } { \mathcal { L } = \ker \delta } \end{array}$ . We prove the equivalence through double inclusion based on the positive definiteness of the admissible pairing.

– ⊇: Let $f \in$ ker $\delta ,$ such that $\delta f = 0 . \mathrm { B y }$ the definition of the discrete Laplacian ${ \mathcal { L } } = \delta ^ { * } \delta \colon$

$$
\mathcal { L } f = \delta ^ { * } ( \delta f ) = \delta ^ { * } ( 0 ) = 0
$$

This implies $f \in$ ker $\mathcal { L } .$

– ⊆: Conversely, let $f \in$ ker ${ \mathcal { L } } ,$ meaning $\mathcal { L } f = 0$ . We evaluate the quadratic form:

$$
0 = \langle \mathcal { L } f , f \rangle = \langle \delta ^ { * } \delta f , f \rangle
$$

Applying the definition of the adjoint operator $\delta ^ { * }$ :

$$
\langle \delta ^ { * } \delta f , f \rangle = \langle \delta f , \delta f \rangle
$$

Since the pairing is admissible, $\langle \delta f , \delta f \rangle = 0$ implies δf = 0 exactly. Thus, $f \in$ ker δ.

• Step 2: Connection to Sheaf Cohomology. By the definition of the discrete sheaf structure, a 0-cochain f is a global section if it is compatible with all restriction maps, which is precisely the condition $\bar { ( \delta f ) } _ { e } \bar { = } 0$ for all $e \in E$ . In the language of sheaf cohomology, the 0-th cohomology group is defined as the kernel of the first differential:

$$
H ^ { 0 } ( G , { \mathcal { F } } ) = \ker \delta
$$

Combining Step 1 and Step 2, we obtain the isomorphism ker ${ \mathcal { L } } \cong H ^ { 0 } ( G , { \mathcal { F } } )$ . This result ensures that the steady states of the discrete diffusion process $\mathbf { \bar { \partial } } \partial _ { t } x = - \mathcal { L } x$ are precisely the global sections of the sheaf. □

## B.5 Technical Details for Proposition 3 (Harmonic Characterisation of OverSmoothing)

The principle stated in Proposition 3 provides an operator-theoretic analysis of representation collapse. The discrete diffusion process is governed by the linear system $\partial _ { t } f ( t ) = - \mathscr { L } f ( t )$ . Given that $\mathcal { L } = \delta ^ { * } \delta$ is a symmetric positive semi-definite operator, let $\{ ( \lambda _ { i } , \phi _ { i } ) \} _ { i = 1 } ^ { n }$ be its eigenpairs, where $0 = \lambda _ { 1 } = \cdot \cdot \cdot = \lambda _ { k } < \lambda _ { k + 1 } \leq \cdot \cdot \cdot \leq \lambda _ { n }$ . The general solution for the feature evolution is:

$$
f ( t ) = \sum _ { i = 1 } ^ { k } \langle f ^ { ( 0 ) } , \phi _ { i } \rangle \phi _ { i } + \sum _ { j = k + 1 } ^ { n } e ^ { - \lambda _ { j } t } \langle f ^ { ( 0 ) } , \phi _ { j } \rangle \phi _ { j }
$$

As $t  \infty .$ , the terms associated with strictly positive eigenvalues vanish. The long-time limit is exactly the projection of the initial state onto the kernel of L, which coincides with previous discussion of Hodge decomposition:

$$
f ^ { ( \infty ) } = \operatorname* { l i m } _ { t \to \infty } f ( t ) = \sum _ { i = 1 } ^ { k } \langle f ^ { ( 0 ) } , \phi _ { i } \rangle \phi _ { i } = P _ { \mathcal H ^ { 0 } } f ^ { ( 0 ) }
$$

When the input signal is centered or nearly centered $( \mu \approx 0 )$ , the ratio simplifies to $\mathbb { E } [ \mathcal { R } ( \mathcal { F } ) ] \approx 1 / N$ This provides a rigorous quantitative characterization of operator-induced oversmoothing: as the graph size N increases, the portion of discriminative information preserved by standard diffusion vanishes at a rate of $\mathcal { O } ( 1 / \bar { N } )$ ). Our framework aims to mitigate this by enriching $\mathcal { H } ^ { 0 }$ such that dim $( \mathcal { H } ^ { 0 } ) \gg 1$ , effectively lifting the preservation ratio away f rom this zero-limit.

![](images/b3b0b02e65afcf81e8971e0ec1f037469bd53264e5cd1bf2d81af0faf48a1a34.jpg)  
Figure 3: Asymptotic decay of $\mathcal { R } ( \mathcal { F } )$ in trivial sheaves $( d i m ( \mathcal { H } ^ { 0 } ) = 1 )$

![](images/b25e8c810fdbc2c5c86ce98dc43c7f1eba684a461c686991e61dcad38058c636.jpg)  
Figure 4: Mitigation of representation collapse through intermediate harmonic enrichment.

![](images/8947d7292524b00692f642842dd252abba9843bd4f578f722b9e78a89d7f673b.jpg)  
Figure 5: Robust information preservation in high-capacity geometric sheaf frameworks.

Two senses of harmonic enrichment. The capacity of a sheaf framework to resist oversmoothing is determined by the size of its harmonic space. We formalize this enrichment in two mathematically distinct senses:

1. Weak Enrichment (Dimensional Capacity): A sheaf $\mathcal { F } _ { 2 }$ is weakly richer than $\mathcal { F } _ { 1 }$ if dim $( \mathcal { H } _ { 2 } ^ { 0 } ) >$ $\dim ( { \mathcal { H } } _ { 1 } ^ { 0 } )$ ).

2. Strong Enrichment (Subspace Inclusion): A sheaf $\mathcal { F } _ { 2 }$ is strongly richer if $\mathcal { H } _ { 1 } ^ { 0 } \subsetneq \mathcal { H } _ { 2 } ^ { 0 }$ . Under this structural condition, we obtain a deterministic, point-wise guarantee. By the decomposition of subspaces $( \mathcal { H } _ { 2 } ^ { 0 } = \mathcal { H } _ { 1 } ^ { 0 } \oplus W )$ , we have $\| P _ { \mathcal { H } _ { 2 } ^ { 0 } } f ^ { ( 0 ) } \| ^ { 2 } \geq \| P _ { \mathcal { H } _ { 1 } ^ { 0 } } \bar { f } ^ { ( 0 ) } \| ^ { 2 }$ for any arbitrary input $f ^ { ( 0 ) }$

Standard graph diffusion restricts $\mathcal { H } ^ { 0 }$ to the 1-dimensional constant subspace. By employing geometry-aware restriction maps, our framework seeks to achieve strict structural enrichment (strong enrichment), lifting the preservation ratio deterministically and unconditionally. This sets the stage for the rigorous index improvement theory developed in Section 4.1.

## C An Index-Theoretic Analysis of oversmoothing

## C.1 Algebraic Foundations of the Discrete Index 4.1

The discrete Hodge–Dirac framework. This appendix provides the rigorous algebraic proofs for the discrete sheaf index and its holonomy-induced harmonic expansion.

Recall the 0-th order discrete Hodge decomposition $\mathcal { C } ^ { 0 } =$ ker $\mathcal { L } \oplus \mathrm { I m } \delta ^ { * }$ . The isomorphism $H ^ { 0 } ( G , { \mathcal { F } } ) \cong$ ker L identifies the harmonic space with the 0-th cohomology group. Consequently, the discrete index explicitly constrains the harmonic dimension:

$$
\dim \ker { \mathcal { L } } = \operatorname { I n d } ( { \mathcal { F } } ) + \dim \ker \delta ^ { * } .\tag{1}
$$

This formula establishes the structural link between the analytic capacity of the diffusion operator and the topological invariants of the sheaf.

The stability of ${ \mathrm { I n d } } ( { \mathcal { F } } )$ under the diffusion flow $\partial _ { t } f = - L f$ is rigorously verified via the discrete heat kernel supertrace (see, e.g., Berline et al. [2004] for general supertrace properties). Let $\mathcal { L } _ { 0 } = \delta ^ { * } \delta$ and $\mathcal { L } _ { 1 } = \delta \delta ^ { * }$ . For any $t > 0$ , the supertrace is defined as:

$$
\mathrm { S t r } ( e ^ { - t L } ) = \mathrm { T r } ( e ^ { - t L _ { 0 } } ) - \mathrm { T r } ( e ^ { - t L _ { 1 } } ) .\tag{2}
$$

Let $\{ \lambda _ { i } \}$ be the strictly positive eigenvalues of ${ \mathcal { L } } _ { 0 } .$ The coboundary operator $\delta$ acts as an isomorphism between the positive eigenspaces $\dot { E } _ { \lambda _ { i } } ( \mathcal { L } _ { 0 } )$ and $E _ { \lambda _ { i } } ( \mathcal { L } _ { 1 } )$ , since the intertwining relation $\delta \mathcal { L } _ { 0 } \overset { = } { = } \mathcal { L } _ { 1 } \delta$ holds. Consequently, the spectral contributions from all strictly positive eigenvalues cancel exactly in the alternating sum:

$$
\mathrm { S t r } ( e ^ { - t \mathcal { L } } ) = \left( \mathrm { d i m } \mathrm { k e r } \mathcal { L } _ { 0 } + \sum _ { \lambda _ { i } > 0 } e ^ { - t \lambda _ { i } } \right) - \left( \mathrm { d i m } \mathrm { k e r } \mathcal { L } _ { 1 } + \sum _ { \lambda _ { i } > 0 } e ^ { - t \lambda _ { i } } \right)\tag{3}
$$

$$
= \dim \ker { \mathcal { L } } _ { 0 } - \dim \ker { \mathcal { L } } _ { 1 } .\tag{4}
$$

Since ker $\mathcal { L } _ { 0 } = \ker \delta$ and ker $\mathcal { L } _ { 1 } = \ker \delta ^ { * }$ , this yields $\mathrm { S t r } ( e ^ { - t \mathcal { L } } ) = \mathrm { I n d } ( \mathcal { F } )$ . Taking the limit $t \to \infty$ confirms that the deep-layer representational capacity is intrinsically locked to the topological index, invariant to the diffusion time.

In deep graph neural networks, oversmoothing refers to the exponential decay of node features into a trivial subspace that carries no discriminative signal. Modeling forward propagation as continuoustime heat diffusion $\partial _ { t } f = - \mathcal { L } f$ with graph Laplacian ${ \mathcal { L } } ,$ , the asymptotic representation capacity is fully characterized by dim ker $\mathcal { L }$

To analyze this capacity, we lift graph signals to a cellular sheaf $\mathcal { F }$ over a graph $G = ( V , E )$ . Denote by $\begin{array} { r } { \mathcal { C } ^ { 0 } \dot { = } \oplus _ { v \in V } \dot { \mathcal { F } } ( v ) } \end{array}$ and $\mathcal C ^ { 1 } = \mathbf { \bar { \bigoplus _ { e \in E } } } \breve { \mathcal F } ( e )$ the spaces of 0-cochains (node features) and 1-cochains (edge features), respectively, and let $\operatorname { r a n k } ( { \mathcal { F } } ) = r$

Definition 10 (Hodge–Dirac operator). The coboundary operator $\delta \colon { \mathcal { C } } ^ { 0 } \to { \mathcal { C } } ^ { 1 }$ measures local inconsistency across edges. Given an admissible inner product, its adjoint $\delta ^ { * } \colon { \mathcal { C } } ^ { 1 } \to { \mathcal { C } } ^ { 0 }$ acts as the discrete divergence, and the 0-Laplacian is $\mathcal { L } _ { 0 } = \delta ^ { * } \delta$ . The discrete Dirac operator on $\mathcal C ^ { 0 } \oplus \mathcal C ^ { 1 }$ is $D _ { \mathcal { F } } = \delta + \delta ^ { * }$ , with chiral component $D _ { \mathcal { F } } ^ { + } = \delta \colon \mathcal { C } ^ { 0 } \to \mathcal { C } ^ { 1 }$

By discrete Hodge theory, ker $L _ { 0 } = \ker \delta \cong H ^ { 0 } ( { \mathcal { F } } )$ , which identifies the space of global sections, i.e., parallel node signals. The first cohomology $H ^ { 1 } ( { \mathcal { F } } )$ captures harmonic edge flows. In the graphlevel case where $\bar { \mathcal { C } } ^ { 2 } = 0$ , this reduces to ker $\bar { \delta } ^ { * }$ : divergence-free edge flows representing conserved cycle currents that cannot be expressed as gradients of node potentials. In sheaves with nontrivial transport, holonomy around fundamental cycles introduces global compatibility obstructions, giving rise to geometric frustration.

Heat dynamics and the analysis index. We state more details of the analysis index. The continuousdepth forward propagation of a diffusion-type GNN can be written as

$$
\partial _ { t } f = - \mathcal { L } _ { 0 } f , \qquad f ( t ) = e ^ { - t \mathcal { L } _ { 0 } } f ( 0 ) ,
$$

where $\mathcal { L } _ { 0 } = \delta _ { \mathcal { F } } ^ { * } \delta _ { \mathcal { F } }$ is the 0-cochain sheaf Laplacian. Since $\mathcal { L } _ { 0 }$ is non-negative and self-adjoint, the heat operator satisfies $\begin{array} { r } { \operatorname* { l i m } _ { t  \infty } e ^ { - t \mathcal { L } _ { 0 } }  P _ { \mathrm { k e r } \mathcal { L } _ { 0 } } } \end{array}$ , where $P _ { \mathrm { k e r } \ : \mathcal { L } _ { 0 } }$ is the projection onto the harmonic space. Consequently, lim $\scriptstyle { \mathrm { 1 } } _ { t \to \infty } \operatorname { T r } ( e ^ { - t { \mathcal { L } } _ { 0 } } ) =$ dim ker $\mathcal { L } _ { 0 }$ . Thus the deep-limit survival space of node features is precisely the space of global sections.

However, the heat trace $\operatorname { T r } ( e ^ { - t { \mathcal { L } } _ { 0 } } )$ itself is not a topological invariant at finite depth. The topologically stable quantity is the heat traceassociated with the Hodge–Dirac pair. Let $D _ { \mathcal { F } }$ act on $\dot { \mathcal { C } } ^ { 0 } ( G ; \mathcal { F } )$ ⊕ $\mathcal { C } ^ { 1 } ( G ; \dot { \mathcal { F } } )$ defined by

$$
D _ { \mathcal { F } } = \left( \begin{array} { c c } { { 0 } } & { { \delta _ { \mathcal { F } } ^ { * } } } \\ { { \delta _ { \mathcal { F } } } } & { { 0 } } \end{array} \right)
$$

with chiral components

$$
D _ { \mathcal { F } } ^ { + } = \delta _ { \mathcal { F } } : \mathcal { C } ^ { 0 } ( G ; \mathcal { F } ) \to \mathcal { C } ^ { 1 } ( G ; \mathcal { F } ) , \qquad D _ { \mathcal { F } } ^ { - } = \delta _ { \mathcal { F } } ^ { * } : \mathcal { C } ^ { 1 } ( G ; \mathcal { F } ) \to \mathcal { C } ^ { 0 } ( G ; \mathcal { F } ) .
$$

The analytic index is defined by

$$
\mathrm { i n d } ( D _ { \mathcal { F } } ^ { + } ) : = \mathrm { d i m } \ker D _ { \mathcal { F } } ^ { + } - \mathrm { d i m } \ker D _ { \mathcal { F } } ^ { - } = \mathrm { d i m } \ker \delta _ { \mathcal { F } } - \mathrm { d i m } \ker \delta _ { \mathcal { F } } ^ { * } .
$$

By the discrete Hodge decomposition, this becomes ind $( D _ { \mathcal { F } } ^ { + } ) = \dim H ^ { 0 } ( \mathcal { F } ) - \dim H ^ { 1 } ( \mathcal { F } )$

## C.2 Proof of Proposition 4 (Topological Index and Index Jump)

Proof of Prop 4. By the Euler–Poincaré formula applied to the sheaf cochain complex,

$$
\dim H ^ { 0 } ( { \mathcal { F } } ) - \dim H ^ { 1 } ( { \mathcal { F } } ) = \dim { \mathcal { C } } ^ { 0 } ( G ; { \mathcal { F } } ) - \dim { \mathcal { C } } ^ { 1 } ( G ; { \mathcal { F } } ) .
$$

Since $\mathcal { F }$ has rank $r ,$

$$
\dim { \mathcal { C } } ^ { 0 } ( G ; { \mathcal { F } } ) = r | V | , \qquad \dim { \mathcal { C } } ^ { 1 } ( G ; { \mathcal { F } } ) = r | E | .
$$

Therefore

$$
\mathrm { i n d } ( D _ { \mathcal { F } } ^ { + } ) = r | V | - r | E | = r \chi ( G ) .
$$

The same argument for the rank-n Euclidean sheaf $\xi$ gives

$$
\operatorname { i n d } ( D _ { \xi } ^ { + } ) = n \chi ( G ) .
$$

Subtracting the two index identities gives

$$
\mathrm { i n d } ( D _ { \mathcal { F } } ^ { + } ) - \mathrm { i n d } ( D _ { \xi } ^ { + } ) = ( r - n ) \chi ( G ) .
$$

## C.3 Details of Remark 3 (Index Jump Alone is Insufficient)

The index comparison exposes a fundamental geometric tension:

$$
\dim { H ^ { 0 } ( \mathcal { F } ) } - \dim { H ^ { 0 } ( \xi ) } = \dim { H ^ { 1 } ( \mathcal { F } ) } - \dim { H ^ { 1 } ( \xi ) } + ( r - n ) \chi ( G ) .
$$

For any connected graph with nontrivial cycle rank, one has $| E | \ge | V |$ and $\chi ( G ) \leq 0$ . Thus increasing the stalk dimension from n to r does not automatically increase the harmonic capacity. When $r > n$ and $\chi ( G ) < 0 .$ , the rank term $( r - n ) \chi ( G )$ is strictly negative. Hence any strict increase of $H ^ { 0 } ( { \mathcal { F } } )$ over the Euclidean baseline must be supported by sufficient growth of the frustration space $H ^ { 1 } ( { \mathcal { F } } )$ , which is interpreted as the divergence free space.

At the same time, the absolute dimension of the harmonic space should not be interpreted as a complete criterion for avoiding oversmoothing. One more promising approach is to focus on the harmonic space relation compares to simple baseline sheaf. More specifically, if there is a comparison map

$$
\iota _ { * } : H ^ { 0 } ( \xi ) \longrightarrow H ^ { 0 } ( \mathcal { F } ) ,
$$

one should measure the additional non-dissipative capacity by the excess quotient $H ^ { 0 } ( \mathcal { F } ) / \iota _ { * } H ^ { 0 } ( \xi )$ or at the level of dimensions, by dim $H ^ { 0 } ( { \mathcal { F } } ) - \dim { \dot { H } } ^ { 0 } ( { \dot { \xi } } ) . { \dot { \mathrm { A } } }$ positive excess dimension means that $\mathcal { F }$ contains harmonic modes beyond the Euclidean constant-channel subspace. If these additional modes are node-separating, they provide a structural mechanism against oversmoothing.

Thus the index jump should be understood as a relative harmonic-capacity certificate, rather than as a pointwise oversmoothing theorem by itself. It does not merely count the size of a single harmonic space; it compares the geometric sheaf against the Euclidean baseline and identifies when additional non-dissipative modes must appear. In the dimension-matched case $r = n$ , the rank penalty disappears and the comparison reduces to

$$
\dim H ^ { 0 } ( { \mathcal { F } } ) - \dim H ^ { 0 } ( \xi ) = \dim H ^ { 1 } ( { \mathcal { F } } ) - \dim H ^ { 1 } ( \xi ) .
$$

Hence, in this case, growth of the frustration space $H ^ { 1 } ( { \mathcal { F } } )$ is exactly reflected as excess harmonic capacity beyond the Euclidean oversmoothing baseline. In this sense, resistance to oversmoothing is mediated by the growth of geometric frustration.

## C.4 Details of Theorem 5 (Genuine Harmonic Inclusion)

Theorem 5 in the main text condenses the results developed in full detail throughout this appendix.   
We investigate when the inclusion $\iota _ { * } : H ^ { 0 } ( \xi )  H ^ { 0 } ( \mathcal { F } )$ holds, beginning with the simplest case.

Example 3 (Trivial holonomy as channel replication). Assume that the sheaf connection has trivial holonomy. In this totally flat regime, the rank-r sheaf has global trivialization and decomposes into r independent scalar Euclidean channels:

$$
\mathcal { F } \cong \xi _ { \mathrm { s c } } \otimes \mathbb { R } ^ { r } ,
$$

where $\xi _ { \mathrm { s c } }$ denotes the rank-1 scalar Euclidean sheaf with stalk R. Hence

$$
\dim H ^ { k } ( { \mathcal { F } } ) = r \beta _ { k } ( G ) , \qquad \beta _ { k } ( G ) : = \dim H ^ { k } ( \xi _ { \mathrm { s c } } ) .
$$

On the other hand, the Euclidean baseline $\xi$ has rank n and satisfies

$$
\xi = \xi _ { \mathrm { s c } } \otimes \mathbb { R } ^ { n } , \qquad \mathrm { d i m } H ^ { k } ( \xi ) = n \beta _ { k } ( G ) .
$$

Therefore, one has dim $H ^ { 0 } ( { \mathcal { F } } ) - \dim H ^ { 0 } ( { \xi } ) = ( r - n ) \beta _ { 0 } ( G )$ . If G is connected, then $\beta _ { 0 } ( G ) = 1$ and hence

$$
\dim H ^ { 0 } ( { \mathcal { F } } ) - \dim H ^ { 0 } ( \xi ) = r - n .
$$

Thus the totally flat case gives a strict capacity gain only by increasing the number of trivial channels, i.e. only when $r > n$ . In the dimension-matched case $r = n ,$ , trivial holonomy gives no excess harmonic capacity over the Euclidean baseline.

Remark 7 (Decoupling under trivial holonomy). The decoupling follows from the existence of a global parallel frame. Fix a base vertex $v _ { 0 } \in V$ . For each vertex $v \in V$ , choose a path $\gamma _ { v _ { 0 } v }$ from v<sub>0</sub> to v and let

$$
Q _ { v } : \mathcal { F } ( v _ { 0 } )  \mathcal { F } ( v )
$$

denote the parallel transport along $\gamma _ { v _ { 0 } v }$ . Since the holonomy around every closed cycle is trivial, $Q _ { v }$ is independent of the chosen path. Hence the maps $\{ Q _ { v } \} _ { v \in V }$ give a global trivialization of the sheaf. In this trivialization, every edge transport becomes the identity:

$$
Q _ { v } ^ { - 1 } P _ { v u } Q _ { u } = I _ { r } ,
$$

for each oriented edge $u  v$ . Consequently, the coboundary operator splits as $\delta _ { \mathcal { F } } = \delta _ { \xi _ { \mathrm { s c } } } \otimes I _ { r }$ where $\xi _ { \mathrm { s c } }$ is the rank-1 scalar Euclidean sheaf with stalk R. Thus

$$
\mathcal { C } ^ { i } ( G ; \mathcal { F } ) \cong \mathcal { C } ^ { i } ( G ; \xi _ { \mathrm { s c } } ) \otimes \mathbb { R } ^ { r } \qquad i = 0 , 1 .
$$

Taking cohomology gives $H ^ { k } ( { \mathcal { F } } ) \cong H ^ { k } ( \xi _ { \mathrm { s c } } ) \otimes \mathbb { R } ^ { r }$ , and hence

$$
\dim H ^ { k } ( { \mathcal { F } } ) = r \dim H ^ { k } ( \xi _ { \mathrm { s c } } ) .
$$

Since $\xi = \xi _ { \mathrm { s c } } \otimes \mathbb { R } ^ { n }$ , one also has

$$
\dim H ^ { k } ( \xi ) = n \dim H ^ { k } ( \xi _ { \mathrm { s c } } ) .
$$

Therefore the totally flat regime does not create genuinely new geometric cycle modes; it only replicates the scalar Euclidean cohomology in independent channels. This motivates the non-trivial holonomy case, where additional harmonic capacity must come from geometric frustration rather than from trivial channel replication.

Non-trivial holonomy. To make more intrinsic complexity of structure for anti-oversmoothing, one should not expect all cocycle-closed transports (holonomy) to be simultaneously trivialized. Instead, the Bochner–Weitzenböck-type formula inspires to separate the cochains into two effect parts: the rough discrepancy of edge-level modes and the curvature obstruction carried by the connection. We analysis the curvature influnced part, which is represented on the sheaf framework by the cocycle transport difference induced from restriction maps.

The inclusion of cochains means that part of the Euclidean baseline is preserved inside the geometric sheaf. In representation-theoretic terms, instead of requiring $R = 0$ or equivalently that the stalk transport is trivial on the whole stalk, we require the restriction maps induces well-defined stalk transport, and the transport’s holonomy representation has a nontrivial fixed subspace. That ${ \mathrm { i s } } ,$ there is a cocycle transport-fixed subspace

$$
W : = \bigcap _ { \gamma \in \pi _ { 1 } ( G , v _ { 0 } ) } \ker ( \rho ( \gamma ) - I ) \subseteq { \mathcal { F } } ( v _ { 0 } ) , \qquad \dim W = k .
$$

$W \neq \mathcal { F } ( v _ { 0 } )$ sites that the excess is not a full trivial-channel replication of the Euclidean baseline; the complementary directions carry nontrivial holonomy.

Every vector in W is fixed by parallel transport around all fundamental cycles. Hence it generates a global parallel section of $\mathcal { F }$ . In particular, when G is connected,

$$
H ^ { 0 } ( { \mathcal { F } } ) \cong \operatorname { F i x } ( \rho ) , \qquad \dim H ^ { 0 } ( { \mathcal { F } } ) = k .
$$

Relative to the rank-n Euclidean baseline $\xi ,$ this gives a strict increase of $H ^ { 0 }$ only when $k \mathrm { ~ > ~ }$ dim $H ^ { 0 } ( \xi ) = n$ for connected G. If one compares with the scalar Euclidean sheaf $\xi _ { \mathrm { s c } }$ , the corresponding condition is $k > 1$

Recall in Berline et al. [2004] that the heat traceis

$$
\mathrm { S t r } ( e ^ { - t D _ { \mathcal { F } } ^ { 2 } } ) = \mathrm { T r } ( e ^ { - t L _ { \mathcal { F } } ^ { 0 } } ) - \mathrm { T r } ( e ^ { - t L _ { \mathcal { F } } ^ { 1 } } ) .
$$

By the finite-dimensional Hodge decomposition, the nonzero spectra of $L _ { \mathcal { F } } ^ { 0 }$ and $L _ { \mathcal { F } } ^ { 1 }$ agree, so their nonzero heat contributions cancel in the supertrace. Hence

$$
\mathrm { S t r } ( e ^ { - t D _ { \mathcal { F } } ^ { 2 } } ) = \dim H ^ { 0 } ( \mathcal { F } ) - \dim H ^ { 1 } ( \mathcal { F } ) = \operatorname { i n d } ( D _ { \mathcal { F } } ^ { + } )
$$

which is just the interpretation of Atiyah-Singer index theorem combined with hodge theory. Equiva lently, $\mathrm { T r } ( e ^ { - t L _ { \mathcal { F } } ^ { 0 } } ) = \mathrm { i n d } ( D _ { \mathcal { F } } ^ { + } ) + \mathrm { T r } ( e ^ { - t L _ { \mathcal { F } } ^ { 1 } } )$ . Applying and subtracting this to $\mathcal { F }$ and ξ gives

$$
\mathrm { T r } ( e ^ { - t L _ { \mathcal { F } } ^ { 0 } } ) - \mathrm { T r } ( e ^ { - t L _ { \xi } ^ { 0 } } ) = \Delta _ { \mathrm { i n d } } ( \mathcal { F } , \xi ) + \left[ \mathrm { T r } ( e ^ { - t L _ { \mathcal { F } } ^ { 1 } } ) - \mathrm { T r } ( e ^ { - t L _ { \xi } ^ { 1 } } ) \right] .
$$

Taking the large-time limit, using

$$
\operatorname* { l i m } _ { t \to \infty } \operatorname { T r } ( e ^ { - t L _ { \mathcal { F } } ^ { i } } ) = \dim H ^ { i } ( \mathcal { F } ) ,
$$

we obtain the corrected relative capacity identity, which is equivalent with the index jump equation Proposition 4:

$$
\dim H ^ { 0 } ( { \mathcal F } ) - \dim H ^ { 0 } ( { \boldsymbol { \xi } } ) = \Delta _ { \mathrm { i n d } } ( { \mathcal F } , { \boldsymbol { \xi } } ) + \Delta _ { \mathrm { S t r } } ^ { ( 1 ) } ( { \mathcal F } , { \boldsymbol { \xi } } ) ,
$$

where

$$
\Delta _ { \operatorname { S t r } } ^ { ( 1 ) } ( { \mathcal F } , \xi ) : = \operatorname* { l i m } _ { t \to \infty } \left[ \operatorname { T r } ( e ^ { - t L _ { { \mathcal F } } ^ { 1 } } ) - \operatorname { T r } ( e ^ { - t L _ { \xi } ^ { 1 } } ) \right] = \dim H ^ { 1 } ( { \mathcal F } ) - \dim H ^ { 1 } ( \xi ) .
$$

Thus the corrected capacity statement is not a replacement of the index idea, but a completion of it: the raw index gives the Euler-balanced part, while the degree-one heat trace supplies the missing first-cohomology contribution needed to recover $H ^ { 0 }$

For 1-cochains $H ^ { 1 }$ , the holonomy fixed decouple in 0-cochain case above requires a stronger condition, we call it together with previous condition of $W$ the "local system condition". Considering that a fixed subspace at the base stalk controls global sections, but it does not automatically imply that the first cohomology of the scalar Euclidean sheaf injects into $H ^ { 1 } ( { \mathcal { F } } )$ . To obtain a decoupled $H ^ { 1 }$ -contribution, we assume that

• W extends to a rank-k trivial sub-sheaf, i.e. the W-channel defines a subcomplex

$$
\mathcal { C } ^ { i } ( G ; \xi _ { \mathrm { s c } } ) \otimes W \longrightarrow \mathcal { C } ^ { i } ( G ; \mathcal { F } ) , \qquad i = 0 , 1
$$

whose induced map on first cohomology is injective:

$$
H ^ { 1 } ( \xi _ { \mathrm { s c } } ) \otimes W \hookrightarrow H ^ { 1 } ( \mathcal { F } ) .
$$

In the local-system case, the corrected index–heat trace balance reduces exactly to the holonomy-fixed excess:

$$
\dim H ^ { 0 } ( { \mathcal { F } } ) - \dim H ^ { 0 } ( \xi ) = k - n .
$$

This explains how nontrivial holonomy can compensate the negative raw index on cyclic graphs. When $\bar { \beta } _ { 1 } ( G )$ is large, the raw index jump $\Delta _ { \mathrm { i n d } } ( \mathcal { F } , \xi ) = ( r - \bar { n } ) ( 1 - \beta _ { 1 } ( G ) )$ may be negative, and the 1-cochain heat trace correction records the cycle-level geometric frustration and restores the correct zeroth harmonic capacity. Consequently, the strict capacity condition can be certified either by the corrected analyst balance $\Delta _ { \mathrm { i n d } } ( \mathcal { F } , \xi ) + \Delta _ { \mathrm { S t r } } ^ { ( 1 ) } ( \mathcal { F } , \xi ) \geq 1$ or by the algebraic setting dim $W > \dim H ^ { 0 } ( \xi )$

We can proceed with the derivation under local system. By analysing the decoupling, one has dim $H ^ { 1 } ( { \mathcal { F } } ) \geq k$ dim $H ^ { 1 } ( \xi _ { \mathrm { s c } } )$ . Since the rank-n Euclidean baseline satisfies

$$
\xi = \xi _ { \mathrm { s c } } \otimes \mathbb { R } ^ { n } , \qquad \mathrm { d i m } H ^ { 1 } ( \xi ) = n \mathrm { d i m } H ^ { 1 } ( \xi _ { \mathrm { s c } } ) ,
$$

we obtain the comparison lower bound

$$
\dim H ^ { 1 } ( { \mathcal { F } } ) - \dim H ^ { 1 } ( { \boldsymbol { \xi } } ) \geq ( k - n ) \dim H ^ { 1 } ( { \boldsymbol { \xi } } _ { \mathrm { s c } } ) .
$$

Equivalently, if G is connected and $\beta _ { 1 } ( G ) = \dim { H ^ { 1 } } ( \xi _ { \mathrm { s c } } )$ , then

$$
\dim H ^ { 1 } ( { \mathcal { F } } ) - \dim H ^ { 1 } ( { \boldsymbol { \xi } } ) \geq ( k - n ) \beta _ { 1 } ( G ) .
$$

Combining this with the index jump, we obtain the sufficient estimate

$$
\dim H ^ { 0 } ( { \mathcal { F } } ) - \dim H ^ { 0 } ( \xi ) \geq ( k - n ) \beta _ { 1 } ( G ) + \Delta _ { \operatorname { i n d } } ( { \mathcal { F } } , \xi )
$$

for connected $G .$ . This reduces to dim H<sup>0</sup>(F) − dim $H ^ { 0 } ( \xi ) \geq ( k - n ) \beta _ { 1 } ( G )$ , which gives the sufficient condition:

$$
\Delta _ { \mathrm { i n d } } ( { \mathcal { F } } , \xi ) \geq ( n - k ) \beta _ { 1 } ( G ) + 1\tag{5}
$$

This inspires an adjustment to increase anti-over smoothing capacity relative to the baseline,by adjusting restriction maps to make sufficiently k forces excess harmonic capacity over the Euclidean baseline.

Full trivial holonomy is the special case $W = \mathcal { F } ( v _ { 0 } )$ , so $k = r$ . In that case the whole sheaf complex decouples into scalar Euclidean channels. The present condition is weaker: the connection may have nontrivial holonomy on the complementary directions, while a rank-k holonomy-fixed channel still contributes Euclidean cohomology to the sheaf. In this sense, the relevant condition is not global flatness of the entire connection, but the existence of a holonomy-fixed subcomplex supporting non-dissipative global and cycle-level modes.

One important note is that condition (5) only fits the case that $\beta _ { 1 }$ is not very large. Under local system condition, the equation is equivalent to $r - n - 1 \geq ( r - k ) \beta _ { 1 } ( G )$ , which cannot be achieved when $\beta _ { 1 } ( G )$ is very large, since we cannot arbitrarily increase the rank of sheaf. One way to solve this problem is to replace the graph topological term $\beta _ { 1 } ( G )$ by some other intrinsic structure’s topological term. For example, For graphs with well-defined geometric embedding properties, replacing the graph Betti number $\beta _ { 1 } ( \bar { G } )$ with the first-order Betti number $\beta _ { 1 } ( M )$ of the embedding base space $M$ is a more reasonable choice, as it directly reflects the truly decisive topological information that might otherwise be obscured by the graph structure. Constructing the replacement topological feature based on the graph and its intrinsic geometry that applies to more general scenarios, and deriving estimation inequalities with improved adaptability inspires one future work related here.

## C.5 Details of the SPD Sheaf Case Study 4.3

We illustrate more details between SPDSheaf $\mathcal { F } _ { \mathrm { S P D } }$ Peng et al. [2026] and our framework of applying Theorem 5. Recall that the SPD sheaf is equipped with stalks modeled on $\operatorname { S P D } _ { n } ,$ congruencetype restriction maps, an SPD coboundary operator, an admissible pairing, an adjoint, and Hodge Laplacians satisfying the harmonic/global-section correspondence. Through the logarithm map

$$
\log : \mathrm { S P D } _ { n } \to \mathrm { S y m } _ { n } ,
$$

the SPD cochains are identified with symmetric-matrix-valued cochains of effective dimension

$$
r = \dim \operatorname { S y m } _ { n } = { \frac { n ( n + 1 ) } { 2 } } .
$$

Under this identification, restriction maps of the form

$$
P \mapsto M P M ^ { \top } , \qquad M \in O ( n ) ,
$$

become linear isometric transports

$$
S \mapsto M S M ^ { \top } .
$$

Thus the SPD sheaf provides a concrete non-Euclidean instance where the finite-dimensional indexheat traceframework applies. We verify the required comparison ingredients as follows:

• Baseline-preserving embedding. The Euclidean-to-SPD lifting

$$
\iota _ { \varepsilon } ( x ) = x x ^ { \top } + \varepsilon I
$$

preserves Euclidean global sections at the harmonic-set level. Hence the Euclidean baseline is embedded into the SPD harmonic space:

$$
\iota _ { * } H ^ { 0 } ( \xi ) \subseteq H ^ { 0 } ( { \mathcal { F } } _ { \mathrm { S P D } } ) .
$$

• Geometric transport. The congruence-type transports generate nontrivial SPD-compatible parallel transport on $\operatorname { S y m } _ { n }$ after logarithmic linearization. The corresponding holonomy-fixed sector gives the geometric source of invariant harmonic directions, while nontrivial transport prevents the model from reducing to a full identity-sheaf tensor replication.

• Index comparision. The SPD Hodge complex satisfies the corrected relative capacity identity

$$
\Delta \dim H ^ { 0 } = \Delta _ { \mathrm { i n d } } + \Delta _ { \mathrm { S t r } } ^ { ( 1 ) } .
$$

Therefore the strict enlargement of SPD global sections should not be interpreted as a positive raw Euler-index jump on cyclic graphs. Instead, it is measured by the relative quotient

$$
H ^ { 0 } ( { \mathcal { F } } _ { \mathrm { S P D } } ) / { \Phi _ { * } H ^ { 0 } ( \xi ) } .
$$

Since the SPD sheaf admits global sections beyond the lifted Euclidean image, the quotient

$$
H ^ { 0 } ( \mathcal { F } _ { \mathrm { S P D } } ) / \iota _ { * } H ^ { 0 } ( \xi )
$$

is nontrivial. This confirms that Theorem 4.2 detects genuine relative harmonic capacity, stronger than simply observing a larger ambient stalk dimension.

Remark 8. The purpose of this discussion is not to claim that the previous SPD sheaf work already provides a complete anti-oversmoothing theory or experimental validation. Rather, it shows that the structural ingredients required by the index-comparison theorem can occur in a concrete non-Euclidean sheaf model.

First, the SPD sheaf provides geometric stalks modeled by $\mathrm { S P D } _ { n }$ , together with restriction maps induced by congruence-type transports and logarithmic SPD geometry [SPD sheaf, Definition/Construction of SPD sheaf; Proposition on SPD restriction maps]. Together with an admissible pairing, these data define a coboundary operator

$$
\delta _ { \mathcal { F } _ { \mathrm { S P D } } } : \mathcal { C } ^ { 0 } ( G ; \mathcal { F } _ { \mathrm { S P D } } ) \to \mathcal { C } ^ { 1 } ( G ; \mathcal { F } _ { \mathrm { S P D } } ) ,
$$

its adjoint $\delta _ { \mathcal { F } _ { \mathrm { S P D } } } ^ { * }$ , and the associated Hodge Laplacian

$$
L _ { 0 , \mathrm { S P D } } = \delta _ { \mathcal { F } _ { \mathrm { S P D } } } ^ { * } \delta _ { \mathcal { F } _ { \mathrm { S P D } } }
$$

By Definition 4.7 in Peng et al. [2026], hence the SPD sheaf realizes the Hodge-compatible operator package required by the general framework.

Second, the Euclidean-to-SPD embedding gives the baseline comparison map

$$
\iota : \xi \to \mathcal { F } _ { \mathrm { S P D } }
$$

By the Euclidean-to-SPD embedding defined in Definition 4.10 of Peng et al. [2026], Euclidean global sections are preserved inside the SPD sheaf, giving an induced injection

$$
\iota _ { * } : H ^ { 0 } ( \xi ) \hookrightarrow H ^ { 0 } ( \mathcal { F } _ { \mathrm { S P D } } )
$$

By Proposition 4.13 in Peng et al. [2026], the SPD sheaf thus does not merely have a larger ambient stalk; it contains the Euclidean harmonic baseline as a comparable subspace of global sections.

Third, the strict enlargement theorem for the SPD sheaf shows that

$$
\dim H ^ { 0 } ( { \mathcal { F } } _ { \mathrm { S P D } } ) > \dim H ^ { 0 } ( \xi )
$$

By Theorem 4.15 in Peng et al. [2026]– or equivalently, since the Euclidean baseline injects into the SPD global-section space, the quotient

$$
H ^ { 0 } ( \mathcal { F } _ { \mathrm { S P D } } ) / \iota _ { * } H ^ { 0 } ( \xi )
$$

is nontrivial. That is, this quotient measures the excess harmonic capacity beyond the Euclidean constant-channel baseline. Moreover, Appendix.B in Peng et al. [2026] gives the index jump, and by calculating the discrete trace of $L _ { \mathrm { S P D } }$ one derives the certification $\Delta _ { \mathrm { i n d } } + \Delta _ { \mathrm { i n d } } > 0$ . In practice, it is enough for us to interpret that it holds Theorem 5.

This reinterpretation is important for two reasons. First, it shows that the comparison theorem is not only formal: a concrete geometric sheaf can satisfy the baseline-preserving injection and strict harmonic enlargement required by the framework. Second, it separates the present contribution from the previous SPD-specific construction. The earlier SPD sheaf work established a strict global-section enlargement result, but it was not formulated as a complete oversmoothing theory and did not provide anti-oversmoothing experiments for SPD sheaves. The present paper extracts the general comparison principle behind that result and later evaluates the anti-oversmoothing behavior experimentally for SPD-type and nonlinear variants.

Finally, the pairing and the index comparison play different roles. The pairing determines the analytic energy

$$
\begin{array} { r } { \mathcal { E } _ { \mathcal { F } } ( x ) = \langle \delta _ { \mathcal { F } } x , \delta _ { \mathcal { F } } x \rangle , } \end{array}
$$

and hence the diffusion trajectory toward the harmonic space. The SPD prototype illustrates the central message of the linear theory: anti-oversmoothing capacity should be compared relative to the Euclidean baseline through an excess harmonic quotient, rather than read from the absolute size of a single kernel.

## D Nonlinear Generalization

## D.1 Details of Non-Linear Index Jump 5

Prior work assumes that the stalks $\mathcal { F } ( v )$ are vector spaces, in which case the cochain spaces are:

$$
{ \mathcal { C } } ^ { 0 } ( G ; { \mathcal { F } } ) = \bigoplus _ { v \in V } { \mathcal { F } } ( v ) , \qquad { \mathcal { C } } ^ { 1 } ( G ; { \mathcal { F } } ) = \bigoplus _ { e \in E } { \mathcal { F } } ( e )
$$

are linear spaces, or at least carry a compatible linear structure $( \mathrm { e . g . }$ , a Lie group structure on SPD manifolds). The coboundary operator δ : $\mathcal { C } ^ { 0 } ( G ; \mathcal { F } ) \to \mathcal { C } ^ { 1 } ( G ; \mathcal { F } )$ is linear, and given an admissible inner product, one can define its adjoint $\delta ^ { * }$ and the 0-Laplacian $\mathcal { L } _ { 0 } = \delta ^ { * } \delta$ , which is self-adjoint by construction. This linear structure is essential: it underpins the heat kernel, the McKean–Singer supertrace, and the resulting global index formula.

A fundamentally different setting arises when the stalks are nonlinear manifolds: each node feature is no longer a vector but a point on a manifold. The 0-cochain space then becomes the product manifold

$$
\displaystyle { \mathcal { C } } ^ { 0 } = \prod _ { v \in V } M _ { v } ,
$$

where each $M _ { v }$ is a (possibly distinct) nonlinear stalk manifold. Correspondingly, the coboundary operator generalizes to a nonlinear map $\delta \colon { \mathcal { C } } ^ { 0 } \to { \mathcal { C } } ^ { 1 }$ , with $\mathcal { C } ^ { 1 }$ denoting the space of edge constraints. The global section condition remains $\delta f = \mathbf { 0 } ,$ , but the linear kernel ker δ is now replaced by the nonlinear harmonic set

$$
{ \mathcal { H } } : = \{ f \in { \mathcal { C } } ^ { 0 } : \delta f = \mathbf { 0 } \} .
$$

When $\mathcal { H }$ is a smooth submanifold, it serves as the natural nonlinear analogue of the space of global sections.

In this nonlinear regime, the classical heat kernel machinery breaks down. While the resulting flow can still be interpreted as a nonlinear gradient flow, it is no longer generated by a linear traceclass operator on a fixed cochain vector space. As a result, the trace $\breve { \mathrm { T r } } ( e ^ { - t \mathcal { L } _ { 0 } } )$ loses its spectral interpretation, and the McKean–Singer supertrace does not extend directly to this setting.

Local tangent laplacian. The correct replacement is local. Around a compatible state $f \in \mathcal H$ , the compatibility map admits a differential

$$
d \delta _ { f } \colon T _ { f } \mathcal { C } ^ { 0 } \longrightarrow T _ { 0 } \mathcal { C } ^ { 1 } ,
$$

which serves as the tangent-space analogue of the coboundary operator. Given admissible inner products on the tangent spaces, we define its adjoint $( d \delta _ { f } ) ^ { * } \colon \bar { T } _ { \mathbf { 0 } } \mathcal { C } ^ { 1 } \to T _ { f } \mathcal { C } ^ { 0 }$ and the local tangent Laplacian

$$
\mathcal { L } _ { 0 , f } : = ( d \delta _ { f } ) ^ { * } d \delta _ { f } .
$$

By construction, $\mathcal { L } _ { 0 , f }$ is a self-adjoint linear operator on $T _ { f } \mathcal { C } ^ { 0 }$ . Although no global linear heat kernel exists in the nonlinear model, the local tangent heat operator $e ^ { - t \mathcal { L } _ { 0 , f } }$ is well-defined, and its large-time limit projects onto

$$
\ker { \mathcal { L } } _ { 0 , f } = \ker d \delta _ { f } .
$$

That is, the surviving directions of the nonlinear diffusion are precisely those in the kernel of the linearized constraint.

When 0 is a regular value of $\delta$ near $f ,$ , the implicit function theorem yields

$$
\begin{array} { r } { T _ { f } \mathcal { H } = \ker d \delta _ { f } , \qquad \mathrm { d i m } T _ { f } \mathcal { H } = \mathrm { d i m } T _ { f } \mathcal { C } ^ { 0 } - \mathrm { r a n k } ( d \delta _ { f } ) . } \end{array}
$$

The nonlinear global-section capacity is therefore determined not by a single global kernel dimension, but by the local dimension of the harmonic submanifold.

From global index to local analytic index. This also gives the correct nonlinear analogue of the index. The linearised operator $d \delta _ { f } : T _ { f } \mathcal { C } ^ { 0 }  T _ { 0 } \mathcal { C } ^ { 1 }$ induces a local analytic index:

$$
\mathrm { i n d } _ { f } ( d \delta ) : = \dim \ker d \delta _ { f } - \dim \mathrm { c o k e r } d \delta _ { f } = \dim T _ { f } { \mathcal C } ^ { 0 } - \dim T _ { 0 } { \mathcal C } ^ { 1 } ,
$$

where the second equality holds because the spaces are finite-dimensional. Hence

$$
\dim T _ { f } { \mathcal { H } } = \operatorname { i n d } _ { f } ( d \delta ) + \dim \operatorname { c o k e r } d \delta _ { f } ;
$$

If $d \delta _ { f }$ is surjective, then dim $T _ { f } \mathcal { H } = \operatorname { i n d } _ { f } ( d \delta )$ , the local replacement of the global index formula.

From global trace formula to local McKean–Singer cancellation. Define Laplacian $\mathcal { L } _ { 1 , f } : =$ $d \delta _ { f } ( d \delta _ { f } ) ^ { * }$ on $T _ { \mathbf { 0 } } \mathcal { C } ^ { 1 }$ . Since the nonzero spectra of $\mathcal { L } _ { 0 , f }$ and $\mathcal { L } _ { 1 , f }$ coincide,

$$
\mathrm { T r } ( e ^ { - t \mathcal { L } _ { 0 , f } } ) - \mathrm { T r } ( e ^ { - t \mathcal { L } _ { 1 , f } } ) = \mathrm { d i m } \mathrm { k e r } d \delta _ { f } - \mathrm { d i m } \mathrm { k e r } ( d \delta _ { f } ) ^ { \ast } = \mathrm { i n d } _ { f } ( d \delta ) .
$$

Thus, the classical McKean–Singer cancellation survives locally after linearisation, even though the original nonlinear model carries no global trace formula.

From global holonomy to local tangent holonomy. When the stalks are manifolds, the differentials of the nonlinear restriction maps around $f \in \mathcal H$ define linear maps between tangent spaces, inducing a tangent sheaf:

$$
T _ { f } \mathcal { F } ( v ) : = T _ { f _ { v } } M _ { v } .
$$

Along a closed cycle $\gamma ,$ the composition of these tangent restriction maps gives a tangent holonomy operator $d \rho _ { f } ( \gamma ) : T _ { f _ { v _ { 0 } } } M _ { v _ { 0 } } \to T _ { f _ { v _ { 0 } } } M _ { v _ { 0 } }$ , whose local parallel directions are controlled by

$$
\mathrm { F i x } ( d \rho _ { f } ) = \bigcap _ { \gamma \in \pi _ { 1 } ( G , v _ { 0 } ) } \ker \bigl ( d \rho _ { f } ( \gamma ) - I \bigr ) .
$$

Thus, The nonlinear holonomy acts through the tangent representation and controls ker $d \delta _ { f }$

## D.2 The Calculation of Dδ and $( D \delta ) ^ { * }$ of 5.2

For an oriented edge $\boldsymbol { e } = ( u , v )$ , the gyro-coboundary is defined as:

$$
\delta ( X ) _ { e } = X _ { v e } \ominus _ { s y m } X _ { u e } = S _ { v e } ^ { 1 / 2 } ( X _ { v e } - X _ { u e } ) S _ { u e } ^ { 1 / 2 } \in \mathrm { S y m } _ { n } = \mathcal { F } ( e ) .
$$

Fixing the u-side representative $X _ { u e } ,$ , we define the v-side local gyro-coboundary map:

$$
\delta _ { v e } ( \cdot ) : = \delta ( \cdot , X _ { u e } ) = \cdot \ominus _ { \mathrm { s y m } } X _ { u e } ,
$$

whose explicit form is:

$$
\delta _ { v e } ( X _ { v e } ) = X _ { v e } \ominus _ { \mathrm { s y m } } X _ { u e } = ( I - X _ { v e } X _ { u e } ) ^ { - 1 / 2 } ( X _ { v e } - X _ { u e } ) ( I - X _ { u e } X _ { v e } ) ^ { - 1 / 2 } .
$$

For notational brevity, set

$$
D _ { e } : = X _ { v e } - X _ { u e } , \qquad L _ { e } : = ( I - X _ { v e } X _ { u e } ) ^ { - 1 / 2 } , \qquad R _ { e } : = ( I - X _ { u e } X _ { v e } ) ^ { - 1 / 2 } .
$$

Then

$$
\delta _ { v e } ( X _ { v e } ) = L _ { e } D _ { e } R _ { e } .
$$

Tangent map. Let $\mathcal { S } _ { K } [ H ] : = D ( K ^ { - 1 / 2 } ) [ H ]$ denote the Fréchet derivative of the principal matrix inverse square root at K in the direction H. The tangent map of $\delta _ { v e }$ at $X _ { v \epsilon }$ e

$$
d \delta _ { v e } : T _ { X _ { v e } } \mathrm { S y m } ( n ) \longrightarrow T _ { \delta _ { v e } ( X _ { v e } ) } \mathrm { S y m } ( n ) ,
$$

is given by

$$
d \delta _ { v e } ( P ) \ = \ L _ { e } P R _ { e } \ + \ \mathcal { V } _ { v e } ( P ) , \qquad P \in T _ { X _ { v e } } \mathrm { S y m } ( n ) ,
$$

where

$$
\begin{array} { r } { \gamma _ { v e } ( P ) : = \mathcal { S } _ { I - X _ { v e } X _ { u e } } \bigl [ - P X _ { u e } \bigr ] D _ { e } R _ { e } + L _ { e } D _ { e } \mathcal { S } _ { I - X _ { u e } X _ { v e } } \bigl [ - X _ { u e } P \bigr ] . } \end{array}
$$

denote the Fréchet derivative of the principal matrix inverse square root. Then the tangent map of $\delta _ { v \epsilon }$ at $X _ { v e }$ expressed as:

$$
d \delta _ { v e } : T _ { X _ { v e } } \mathrm { S y m } ( n ) \to T _ { X _ { v e } \ominus X _ { u e } } \mathrm { S y m } ( n )
$$

$$
d \delta _ { v e } ( P ) = L _ { e } P R _ { e } + \mathcal { V } _ { v e } ( P ) , \qquad P \in T _ { X _ { v e } } \mathrm { S y m } ( n ) ,
$$

where

$$
\begin{array} { r } { \mathcal { V } _ { v e } ( P ) : = \mathcal { S } _ { I - X _ { v e } X _ { u e } } \bigl [ { - P X _ { u e } } \bigr ] D _ { e } R _ { e } + L _ { e } D _ { e } \mathcal { S } _ { I - X _ { u e } X _ { v e } } [ - X _ { u e } P ] . } \end{array}
$$

Frobenius adjoint. Let $\langle A , B \rangle _ { F } : = \operatorname { t r } ( A ^ { \top } B )$ denote the Frobenius inner product. The adjoint of $d \delta _ { v e }$ with respect to this pairing is

$$
( d \delta _ { v e } ) ^ { * } ( E ) \ : = \ : \mathrm { s y m } \Big ( L _ { e } ^ { \top } E R _ { e } ^ { \top } \ : + \ : \gamma _ { v e } ^ { * } ( E ) \Big ) , \quad \quad E \in T _ { \delta _ { v e } ( X _ { v e } ) } \mathrm { S y m } ( n ) ,
$$

where $\operatorname { s y m } ( Z ) : = \frac { 1 } { 2 } ( Z + Z ^ { \top } )$ ) and

$$
\begin{array} { r } { \mathcal { V } _ { v e } ^ { * } ( E ) : = - \mathcal { S } _ { I - X _ { v e } X _ { u e } } ^ { * } \big ( E R _ { e } ^ { \top } D _ { e } ^ { \top } \big ) X _ { u e } ^ { \top } - X _ { u e } ^ { \top } \mathcal { S } _ { I - X _ { u e } X _ { v e } } ^ { * } \big ( D _ { e } ^ { \top } L _ { e } ^ { \top } E \big ) . } \end{array}
$$

Here $\mathcal { S } _ { K } ^ { * }$ denotes the Frobenius adjoint of the linear map $\mathcal { S } _ { K }$

Remark 9. While the closed-form expressions for $d \delta _ { v e }$ and $( d \delta _ { v e } ) ^ { * }$ above may appear involved, we stress that they need not be implemented explicitly for gradient computation during training. Because the gyro-Laplacian $\mathcal { L }$ satisfies $( \mathcal { L } X ) _ { v } : = \nabla _ { X _ { v } } \mathcal { E } ( X ) = \big ( D \delta ( X ) \big ) ^ { * } D \delta ( X )$ , differentiating gyro-Laplacian end-to-end with automatic differentiation yields exactly the same gradient as composing the tangent maps and their adjoints derived here. In all our experiments we therefore simply backpropagate through $\Delta _ { \mathrm { g y r } }$ directly, leveraging the autograd machinery of PyTorch. The explicit formulate are presented for theoretical completeness: they make the gradient structure transparent.

## D.3 Proof of Proposition 6 (Hodge-type Decomposition for the Nonlinear GyroSheaf)

Proof. This construction is in the same spirit as the classical Hodge decomposition on a compact, oriented Riemannian manifold M without boundary. In the continuous case, one starts from the space $A ^ { k } ( M )$ of smooth k-forms. For any two such forms $\alpha , \beta \in A ^ { k } ( M )$ , the Riemannian metric together with the Hodge star operator ∗ determines an $L ^ { 2 } .$ -inner product:

$$
\langle \alpha , \beta \rangle _ { L ^ { 2 } } = \int _ { M } \alpha \wedge { * \beta } .
$$

Thus, after $L ^ { 2 } .$ -completion, $A ^ { k } ( M )$ can be viewed inside a Hilbert space. The exterior derivative

$$
d \colon A ^ { k - 1 } ( M ) \to A ^ { k } ( M )
$$

is a linear operator. Its adjoint

$$
d ^ { * } : A ^ { k } ( M ) \to A ^ { k - 1 } ( M )
$$

is defined by

$$
\langle d \alpha , \beta \rangle _ { L ^ { 2 } } = \langle \alpha , d ^ { * } \beta \rangle _ { L ^ { 2 } } .
$$

Then the Hilbert space identity

$$
\mathrm { i m } ( d ) = \ker ( d ^ { * } ) ^ { \perp }
$$

gives the decomposition

$$
A ^ { k } ( M ) = \operatorname { i m } ( d ) \oplus ^ { \bot } \ker ( d ^ { * } ) .
$$

Hence the direct-sum decomposition is obtained by combining three ingredients: a Hilbert structure, an inner product, and an adjoint operator.

In our discrete setting the same three ingredients are present. The role of the space of k-forms is played by the tangent spaces $T _ { X } { \mathcal { C } } ^ { 0 }$ and $\Breve { T _ { \delta ( X ) } } \Breve { C ^ { 1 } }$ , both of which are finite-dimensional inner-product spaces and hence, in particular, Hilbert spaces. The linearized boundary map

$$
D \delta _ { X } : T _ { X } { \mathcal { C } } ^ { 0 } \longrightarrow T _ { \delta ( X ) } C ^ { 1 }
$$

takes the place of the exterior derivative $d ,$ and its adjoint

$$
D \delta _ { X } ^ { * } : T _ { \delta ( X ) } \mathcal { C } ^ { 1 } \longrightarrow T _ { X } \mathcal { C } ^ { 0 }
$$

takes the place of the codifferential $d ^ { * }$ . We now derive the corresponding complement relation. Observe that

$$
Y \in \mathrm { i m } ( D \delta _ { X } ) ^ { \perp }
$$

if and only if

$$
\langle D \delta _ { X } h , Y \rangle = 0 , \qquad \forall h \in T _ { X } \mathcal { C } ^ { 0 } .
$$

By the definition of the adjoint,

$$
\langle D \delta _ { X } h , Y \rangle = \langle h , D \delta _ { X } ^ { * } Y \rangle .
$$

Therefore,

$$
\langle D \delta _ { X } h , Y \rangle = 0 , \quad \forall h
$$

if and only if

$$
D \delta _ { X } ^ { * } Y = 0 .
$$

Hence

$$
\mathrm { i m } ( D \delta _ { X } ) ^ { \perp } = \mathrm { k e r } ( D \delta _ { X } ^ { * } ) ,
$$

which yields the decomposition

$$
T _ { \delta ( X ) } \mathcal { C } ^ { 1 } = \mathrm { i m } ( D \delta _ { X } ) \oplus ^ { \perp } \mathrm { k e r } ( D \delta _ { X } ^ { * } ) .
$$

Similarly, on the node-level tangent space, we obtain

$$
T _ { X } { \mathcal { C } } ^ { 0 } = \operatorname { i m } ( D \delta _ { X } ^ { * } ) \oplus ^ { \perp } \ker ( D \delta _ { X } ) .
$$

## E Experimental Setup

This appendix details the implementation, datasets, hyperparameters, and runtime environment for experiments reported in Section 6. Two regimes are evaluated and configured separately: an untrained regime that isolates intrinsic propagation dynamics (Sec. E.4), and a trained regime that verifies these properties survive end-to-end optimization (Sec. E.5). Model implementations are described in Sec. E.2, and the compute environment details in Sec. E.6.

## E.1 Datasets

Datasets. We use Cora, Citeseer and Texas since they span both homophilic citation graphs (Cora, Citeseer) and a heterophilic web graph (Texas), so that the layer-wise propagation behavior is evaluated under qualitatively different connectivity regimes.

## E.2 Model Implementations

Linear sheaf diffusion. We use the original NSD discrete sheaf diffusion implementation Bodnar et al. [2022] (the neural-sheaf-diffusion reference repository) for DiagSheaf, BundleSheaf and GeneralSheaf. IdentitySheaf instantiates the same NSD backbone and Laplacian builder as DiagSheaf, but the restriction maps are hard-coded to the identity matrix on every edge. For InverseSheaf, for each undirected edge $e = ( i , j )$ we predict an invertible matrix $M _ { e }$ from the concatenated endpoint features through a single linear layer with a nonlinearity, parameterized via an LU factorization $M _ { e } = L _ { e } U _ { e }$ with strict-triangular off-diagonals and softplus-rectified diagonals to ensure $M _ { e }$ is invertible throughout training. Edge transports in the both $i  j$ and $j  i$ direction use the same $M _ { e }$

SPDSheaf. SPDSheaf is a from-scratch PyTorch implementation following the SPD-valued sheaf construction of Peng et al. [2026], simplified for the oversmoothing benchmark: we retain the core SPD geometry and sheaf Laplacian, but omit the molecule-specific dual-stream architecture and the coordinate-to-SPD lifting. Each node carries an SPD state $P _ { v } \in \mathrm { S P D } _ { n }$ . For each edge $e = ( i , j )$ , the restriction map is parameterized by feeding the tangent-space vectorizations [vec(log $P _ { i } ) \parallel \operatorname { v e c } ( \log P _ { j } ) ]$ through a single-layer local-concat sheaf learner Bodnar et al. [2022] to get an orthogonal matrix $M _ { i e } \in O ( \overline { { n } } )$ . The restriction map acts by orthogonal congruence: $\mathcal { F } _ { i  e } ( P ) = M _ { i e } P M _ { i e } ^ { \top }$ . Each layer applies the SPD sheaf Laplacian $\mathcal { L } _ { \mathrm { S P D } } ^ { ( \ell ) }$ (Definition 4.7 of Peng et al. [2026]), and then an Lie-group update is applied at the identity:

$$
P _ { v } ^ { ( \ell + 1 ) } = P _ { v } ^ { ( \ell ) } \odot \mathcal { L } _ { \mathrm { S P D } } ^ { ( \ell ) } ( P _ { v } ^ { ( \ell ) } ) .
$$

GyroSheaf (ours). GyroSheaf is also implemented from scratch in PyTorch. Each node carries an SPD state $P _ { v } \in \mathrm { S P D } _ { n }$ , which is mapped to the symmetric unit ball $\dot { \boldsymbol { B } }$ via the Cayley transform $\Psi ( P ) = ( P - I ) ( P + I ) ^ { - 1 }$ before propagation. Restriction maps are parameterized identically to SPDSheaf so that SPDSheaf and GyroSheaf differ only in how information is propagated across edges, isolating the effect of intrinsic curved propagation. The diffusion update is the raw Laplacian increment $\boldsymbol { X _ { k + 1 } } = \boldsymbol { X _ { k } } + \boldsymbol { \mathcal { L } } ( \boldsymbol { X _ { k } } )$ based on definition 9.

## E.3 Oversmoothing Measures

We adopt a Dirichlet energy measure and the mean-average-distance as primary oversmoothing measures. The pairwise summation form of both follows the survey of Rusch et al.Rusch et al. [2023] (Eqs. (2) and (4) therein). For normalized Dirichlet energy, we additionally divide by $\| X \| _ { F } ^ { 2 }$ to make the measure invariant to global feature rescaling, in line with Zhang et al. [2026].

Where the measure is evaluated. All oversmoothing measures are computed on the backbone hidden representation immediately after the last propagation layer, before any classification head. In the untrained regime this is the only representation produced by the model. In the trained regime this means we explicitly do not measure on the classifier logits, so that metrics are insensitive to head-specific design choices and remain directly comparable to the untrained trajectories.

Neighborhood. For all measures and all models, the neighborhood used in the pairwise summation is the symmetrized 1-hop neighborhood without self-loops,

$$
{ \mathcal { N } } _ { i } = \{ j \in V : ( i , j ) \in E \} ,
$$

applied identically across the Euclidean baselines, the linear sheaves, and the SPD-/gyro-valued models.

Euclidean and linear-sheaf models. For models whose final-layer representation is a node feature matrix $X = \{ x _ { v } \} _ { v \in V } \subset \mathbb { R } ^ { d }$ (GCN, GAT, GraphSAGE, DiagSheaf, BundleSheaf, GeneralSheaf, IdentitySheaf, InverseSheaf), both measures use the standard $\ell _ { 2 }$ inner product on $\mathbb { R } ^ { d }$

$$
\mu _ { \mathrm { N o r m D i r } } ( X ) = \frac { \frac { 1 } { | V | } \sum _ { i \in V } \sum _ { j \in \mathcal { N } _ { i } } \| x _ { i } - x _ { j } \| _ { 2 } ^ { 2 } } { \sum _ { v \in V } \| x _ { v } \| _ { 2 } ^ { 2 } } ,
$$

$$
\mu _ { \mathrm { M A D } } ( X ) = \frac { 1 } { | V | } \sum _ { i \in V } \sum _ { j \in { \cal N } _ { i } } \left( 1 - \frac { \langle x _ { i } , x _ { j } \rangle } { \| x _ { i } \| _ { 2 } \| x _ { j } \| _ { 2 } } \right) .
$$

SPD-valued models. For models whose final-layer representation is SPD-valued, $P = \{ P _ { v } \} _ { v \in V } \subset$ $\mathrm { S P D } _ { n } ,$ the natural distance is not the ambient $\ell _ { 2 }$ distance but the Log-Euclidean distance, which is intrinsic to the SPD manifold. We therefore replace each $P _ { v }$ by its image $S _ { v } = \log ( P _ { v } ) \in \mathrm { S y m } _ { n } ,$ the $n \times n$ symmetric matrix, under the matrix logarithm and compute the corresponding measures

with the Frobenius inner product on $\operatorname { S y m } _ { n }$

$$
\begin{array} { r l r } { \mu _ { \mathrm { N o r m D i r } } ^ { \mathrm { S P D } } ( P ) } & { = } & { \frac { \frac { 1 } { | V | } \sum _ { i \in V } \sum _ { j \in N _ { i } } \| S _ { i } - S _ { j } \| _ { F } ^ { 2 } } { \sum _ { v \in V } \| S _ { v } \| _ { F } ^ { 2 } } , } \\ { \mu _ { \mathrm { M A D } } ^ { \mathrm { S P D } } ( P ) } & { = } & { \frac { 1 } { | V | } \displaystyle \sum _ { i \in V } \displaystyle \sum _ { j \in \cal N _ { i } } \left( 1 - \frac { \mathrm { t r } ( S _ { i } S _ { j } ) } { \| S _ { i } \| _ { F } \| S _ { j } \| _ { F } } \right) . } \end{array}
$$

The Log-Euclidean measures admit an equivalent coordinate representation under any linear isomorphism $\mathrm { S y m } _ { n }  \mathbb { R } ^ { n ( n + 1 ) / 2 }$ , but we formulate them at the matrix level to make the underlying Riemannian structure on $\mathrm { S P D } _ { n }$ explicit and to avoid committing to an arbitrary coordinate chart.

GyroSheaf. GyroSheaf’s final-layer representation lies in the symmetric unit ball $B \subset \operatorname { S y m } _ { n } .$ equipped with a gyrovector-space geometry that is distinct from the Log-Euclidean structure used for SPDSheaf. To enable a controlled comparison between SPDSheaf and GyroSheaf, we deliberately evaluate both under a common metric: we recover an SPD representation via the inverse Cayley transform $\Psi ^ { - 1 } ( X ) = ( I + X ) ( I - X ) ^ { - 1 }$ , and apply the SPD measures above to $\Psi ^ { - 1 } ( X ) \in \bar { \mathrm { S P D } _ { n } }$ This is an explicit trade-off: it sacrifices alignment with GyroSheaf’s native gyro-geometry in exchange for commensurability with SPDSheaf, so that the SPDSheaf-vs-GyroSheaf comparison in Appendix F.1 isolates the effect of the propagation rule rather than confounding it with the choice of measurement geometry.

Consistency across geometries. All four measures share identical functional form and normalization; the only difference is the underlying inner product $( \ell _ { 2 }$ on $\mathbb { R } ^ { d }$ versus Frobenius on $\operatorname { S y m } _ { n } )$ . Since the feature dimensions are matched $( n \bar { ( } n + 1 \bar { ) } / 2 = 1 2 0 = d )$ and all measures are scale-invariant, the trajectory shapes are directly comparable across all ten models.

Aggregation across seeds and depths. For each (model, depth) configuration, the measures are computed independently for each of the five seeds and aggregated into a per-depth mean and min–max range. In all trajectory plots the solid curves show the per-seed mean and the shaded bands show the per-seed min–max range; we do not report standard deviations because at 5 seeds the min–max range is a more honest summary of the variation actually observed.

## E.4 Untrained Regime

Following Rusch et al. Rusch et al. [2023], both node features and model parameters are randomly initialized, and we record layer-wise representations under purely forward propagation; no optimizer, loss, or training step is involved. Depths $L \in \{ 1 , 2 , 4 , 8 , 1 6 , \bar { 3 } 2 , \bar { 6 4 } , 1 2 8 \}$ are evaluated over five seeds $\{ 4 2 , 4 3 , 4 4 , 4 5 , 4 6 \}$ ; trajectory plots report the mean and min–max range across seeds. All Euclidean and linear-sheaf models use a hidden width of 120. For SPDSheaf and GyroSheaf, the SPD stalk dimension is $n = 1 5$ , giving $n ( n { + } 1 ) / 2 = 1 2 0$ independent parameters per node, matched to the Euclidean width. For all linear sheaf models from Bodnar et al. [2022], the auxiliary weights are disabled. These are feature-space transformations that do not enter the sheaf definition and disabling them isolates the intrinsic sheaf-geometric propagation characterized by Theorem 5. SPDSheaf and GyroSheaf do not have these auxiliary weights by construction.

## E.5 Trained Regime

We train every model end-to-end on Cora for $L \in \{ 2 , 4 , 8 , 1 6 , 3 2 \}$ , with five seeds {42, 43, 44, 45, 46} per configuration. All oversmoothing curves are computed on the trained backbone hidden represen tations (not on the classifier logits) so that they are directly comparable to the untrained trajectories of App. E.4. To isolate the effect of architecture on oversmoothing, we use a single shared training recipe across all model families, summarized in Table 1; all models are trained using cross-entropy loss with the label smoothing value listed therein, and early stopping is triggered when validation accuracy fails to improve for Patience consecutive epochs.

## E.6 Compute Environment

All experiments were conducted on a single NVIDIA GeForce RTX 3090 (24GB) GPU with Intel Xeon Gold 5218R CPUs and 376GB system memory. The total compute for all reported results,

Table 1: Training hyperparameters (shared across all model families).
<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>Learning rate</td><td>0.004</td></tr><tr><td>Weight decay</td><td> $1 0 ^ { - 4 }$ </td></tr><tr><td>Epochs</td><td>220</td></tr><tr><td>Patience</td><td>50</td></tr><tr><td>Label smoothing</td><td>0.05</td></tr><tr><td>LR scheduler</td><td>ReduceLROnPlateau</td></tr><tr><td>Stalk dim. d (sheaf models only)</td><td>8</td></tr></table>

covering ten models, three datasets, five seeds, and eight depth configurations in both untrained and trained regimes, is modest and reproducible on a single consumer-grade GPU.

## F Additional Experiments

## F.1 Within-Tier Discrimination: SPDSheaf vs. GyroSheaf under Nonlinear Distortion

Motivation. Sections 6.2 and 6.3 verify the between-tier ordering predicted by Theorem 5. Within Tier 3, however, SPDSheaf and GyroSheaf share the same stalk dimension and restriction maps, therefore the same index jump $\Delta _ { \mathrm { i n d } }$ and holonomy fixed subspace W, so the index criterion alone cannot separate them. What actually distinguishes the two models is how each realizes the tangent complex of Section $5 ,$ and the goal of this experiment is to expose the regime in which this difference becomes observable.

Two linearization strategies. SPDSheaf exploits the associative Lie-group structure of $\mathrm { S P D } _ { n }$ to linearize the cochain complex through a single global chart, namely the matrix logarithm (cf. Appendix C.5). The resulting linear Hodge complex falls within the finite-dimensional index–heat trace framework of Section 4. This shortcut is faithful when the observation map respects the global chart, but its fidelity degrades once the data departs from the globally-linearizable regime. GyroSheaf, by contrast, has no associative group operation on its gyrovector-space stalks (Section 5.2); it therefore follows the pointwise linearization program of Section 5, building the tangent coboundary $d \delta _ { X }$ and the tangent Hodge Laplacian afresh at each harmonic state X. This state-dependent construction makes no global-chart assumption and is expected to remain faithful when the observation manifold is nonlinearly distorted.

Setup. Latent coordinates $\{ z _ { v } \} \subset \mathbb { R } ^ { 2 }$ are drawn from a 3-arm pinwheel distribution, and node labels are assigned by the angular coordinate of $z _ { v }$ so that supervision is not linearly separable. The graph is the symmetrized 8-nearest-neighbor graph in latent space, held fixed independently of the observation map. Observed features $x _ { v } \in \mathbb { R } ^ { 1 6 }$ are produced by a fixed nonlinear basis expansion of $z _ { v }$ followed by a random linear projection. Splits are stratified by label with 20 train and 20 validation nodes per class.

To progressively distort the observation manifold, we inject isotropic Gaussian noise:

$$
\tilde { x } \ = \ \frac { ( x + \eta \varepsilon ) - \mu } { \sigma } , \qquad \varepsilon \sim \mathcal { N } ( 0 , I ) , \quad \eta \in [ 0 , 0 . 8 ] ,
$$

where $( \mu , \sigma )$ are per-dimension statistics of the perturbed features. Re-standardization keeps scales comparable across noise levels. Latent coordinates, labels, and graph are held fixed; only η varies.

Quantifying global-chart breakdown. To compare the two strategies meaningfully, we need a model-agnostic measure of how far the observation map departs from local affinity, i.e. how stressed a fixed global chart should be. For each node $i ,$ let ${ \mathcal { N } } _ { i }$ denote its 8 latent-space nearest neighbors; fit a local affine map $( A _ { i } , b _ { i } )$ by least squares and define the normalized residual

$$
\operatorname { A f f E r r } ( X ) \ = \ { \frac { 1 } { | V | } } \sum _ { i \in V } { \frac { { \frac { 1 } { | \mathcal { N } _ { i } | } } \sum _ { j \in \mathcal { N } _ { i } } | | x _ { j } - ( A _ { i } z _ { j } + b _ { i } ) | | _ { 2 } ^ { 2 } } { \operatorname { V a r } ( X ) } } .
$$

$\mathrm { A f f E r r } = 0$ when the observation map is globally affine; larger values mark the breakdown of local-affine approximability. This is precisely the regime in which the fixed global-chart linearization (SPDSheaf) is expected to lose fidelity while the state-dependent tangent-complex construction (GyroSheaf) should remain stable.

![](images/2db87c0c885c5480c3c9bc07076123993c05a7ae25f42a3a6fb8ae19b050739f.jpg)  
Figure 6: PCA projections of input features (left), GyroSheaf (middle), and SPDSheaf (right) at the clean (η = 0, top) and strongly-perturbed $( \eta = 0 . 8$ , bottom) endpoints; ellipses show per-class covariance. Under strong perturbation, SPDSheaf collapses into a narrow band while GyroSheaf retains separable class structure. Right: test accuracy versus AfErr(X). SPDSheaf degrades monotonically as AfErr grows; GyroSheaf remains flat.

Results. Figure 6 reports test accuracy against AfErr. At AfErr = 0.002, where local linearity essentially holds, the two models result similar: when the global chart is faithful, both linearization strategies succeed. As AfErr grows to 0.273, SPDSheaf trends downward to 0.73, while GyroSheaf remains essentially flat at 0.83. The two models become distinguishable here: GyroSheaf’s pointwise tangent-complex construction tolerates nonlinear distortion that defeats SPDSheaf’s global-chart linearization. In short, the index criterion governs harmonic capacity, while the linearization strategy provides a complementary axis of robustness once the data leaves the globally-linearizable regime.