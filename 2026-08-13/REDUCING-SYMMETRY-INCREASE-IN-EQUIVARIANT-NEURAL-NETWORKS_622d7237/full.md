# REDUCING SYMMETRY INCREASE IN EQUIVARIANT NEURAL NETWORKS

Ning Lin, Jiacheng Cen, Anyi Li, Wenbing Huang<sup>∗</sup>, Hao Sun<sup>∗</sup>   
Gaoling School of Artificial Intelligence, Renmin University of China, Beijing, China   
{ninglin00, jiacc.cn, li\_anyi}@outlook.com,   
{hwenbing, haosun}@ruc.edu.cn

## ABSTRACT

Equivariant Neural Networks (ENNs) have empowered numerous applications in scientific fields. Despite their remarkable capacity for representing geometric structures, ENNs suffer from degraded expressivity when processing symmetric inputs: the output representations are invariant to transformations that extend beyond the input’s symmetries. The mathematical essence of this phenomenon is that a symmetric input, after being processed by an equivariant map, experiences an increase in symmetry. While prior research has documented symmetry increase in specific cases, a rigorous understanding of its underlying causes and general reduction strategies remains lacking. In this paper, we provide a detailed and in-depth characterization of symmetry increase together with a principled framework for its reduction: (i) For any given feature space and input symmetry group, we prove that the increased symmetry admits an infimum determined by the structure of the feature space; (ii) Building on this foundation, we develop a computable algorithm to derive this infimum, and propose practical guidelines for feature design to prevent harmful symmetry increases. (iii) Under standard regularity assumptions, we demonstrate that for most equivariant maps, our guidelines effectively reduce symmetry increase. To complement our theoretical findings, we provide visualizations and experiments on both synthetic datasets and the real-world QM9 dataset. The results validate our theoretical predictions.

## 1 INTRODUCTION

Equivariant Neural Networks (ENNs) have become a cornerstone of modern machine learning, empowering numerous applications in scientific fields ranging from molecular dynamics to materials design (Bronstein et al., 2021; Huang & Cen, 2026). By building in physical symmetries, these models achieve remarkable data efficiency and generalization capabilities when representing complex geometric structures.

![](images/0f7a46bf53aaa58d8e0a432cc2cfce3f655b306f674853236a7158f75dcdd2fe.jpg)  
(a) 3-fold

![](images/9e72a695a92f3670d6f9e9af616cc02d44b7e693667445f952553b98de26e7f6.jpg)  
(b) 4-fold  
Figure 1: k-fold structures.

Despite their success, ENNs exhibit a critical vulnerability when processing symmetric inputs: their expressivity can degrade, leading to a loss of information. This phenomenon, which we term symmetry increase, occurs when the output representation becomes invariant to transformations that are not symmetries of the original input itself. A canonical example arises when processing k-fold symmetric structures. These objects, visualized in Fig. 1, possess a specific dihedral symmetry, yet an ENN will map their distinct rotated versions to an identical feature, erasing their orientation.

This type of degradation has been documented in previous work. Empirically, it has been observed that the degradation depends on the feature space, particularly for symmetric inputs (Joshi et al., 2023). Theoretically, research on ENNs has shown that for k-fold symmetries, selecting only low-degree features can cause all rotated inputs to collapse into a single representation (Cen et al., 2024). This ENN-specific issue is a modern manifestation of a general phenomenon observed in other fields. It has been linked to physical principles such as Curie’s Principle (Smidt et al., 2021) and described using the concept of orbit types (Kaba & Ravanbakhsh, 2023).

However, existing analyses provide an incomplete picture and lack a predictive framework. In our analysis, this degradation of k-fold caused by symmetry increase can be categorized into three distinct types: full degeneration, axial degeneration, and halfdegeneration (see Fig. 2)<sup>1</sup>. The work of Joshi et al. (2023), while empirically important, does not theoretically explore the cause of the degradation. The collapse-to-zero theory proposed by Cen et al. (2024) addresses only full degeneration, which is the most extreme case identified by our analysis. The broader principles discussed by Smidt et al. (2021) and Kaba & Ravanbakhsh (2023) lack a rigorous mathematical description, and the solutions proposed, such as in Kaba & Ravanbakhsh (2023), often involve relaxing the equivariance constraint itself, rather than providing a solution within the equivariant framework.

In this paper, we fill this gap by providing a comprehensive mathematical characterization of symmetry increase. Our main contributions are briefly listed as follows:

• In $\ S \ 3 ,$ we prove for any given feature space and input symmetry group, that the increased symmetry is bounded from below by a unique symmetry infimum, which is determined entirely by the algebraic structure of the feature space.

• In $\ S 4 ,$ , we develop a computable algorithm to derive this infimum by analyzing the orbit types. This provides practical guidelines for predicting and controlling potential symmetry increases, thereby preventing harmful symmetry increases in feature design.

• In $\ S \ : 5 ,$ we demonstrate that under regularity conditions, such as the manifold hypothesis for data, our method can fully reduce symmetry increase. Specifically, for most equivariant maps or for ENNs with sufficient approximation capabilities, the output symmetry will be precisely this predictable infimum, preventing orientational information loss.

• In $\ S \ S ,$ we complement our theoretical findings with empirical evidence. We provide visualizations to illustrate the proposed concepts and present experimental results on both synthetic datasets and the real-world QM9 dataset, which validate our theoretical predictions and demon strate the practical effectiveness of our framework.

## 2 PRELIMINARIES

Group action and representation. Consider the action of a group $G$ on a set $X$ , denoted by $\rho _ { X }$ This action is a map that assigns to each element $g \in G$ a transformation $\rho _ { X } ( g ) : X \to X$ , such that $\rho _ { X } ( g _ { 1 } g _ { 2 } ) = \rho _ { X } ( g _ { 1 } ) \rho _ { X } ( g _ { 2 } )$ . We call such a set X a G-set. In particular, if X is a vector space and $\rho _ { X } ( g )$ is a linear transformation for all $g \in G$ , we call X a G-representation<sup>2</sup>.

Equivariant map. For maps between two G-sets X and $Y _ { \pm }$ , an equivariant map is one that respects the group action, meaning that the output transforms accordingly when the input is transformed. Formally, a map $f : X \to Y$ is equivariant if for all $g \in G$ and $x \in X$

$$
f ( \rho _ { X } ( g ) ( x ) ) = \rho _ { Y } ( g ) ( f ( x ) ) .\tag{1}
$$

Example 2.1 (Equivariant Encoding of Point Clouds). The symmetry group is $G = H \times S _ { n } ,$ where H is typically the special orthogonal group $S O ( 3 )$ or the orthogonal group $O ( 3 )$ , and $S _ { n }$ is the permutation group. We consider features that are invariant to both permutation and translation. To achieve translation invariance, the input representation X is the space ofcentered point clouds, where H acts on the coordinate of each point and $S _ { n }$ permutes the points. To achieve permutation invariance, thefinal output are designed to be invariant with respect to $S _ { n }$ . This means thefeature representation ofinterest is a direct sum ofspecified irreducible representations ofH. The task of equivariant encoding is then to learn an equivariant map $f : X \to Y$

Data symmetry. For a G-set X, the action partitions the space into orbits, $G ( x ) : = \left\{ g ( x ) \mid g \in G \right\}$ The subgroup that fixes a point is its isotropy subgroup, ${ \bar { G _ { x } } } : = \{ g \in G \mid g ( x ) = x \}$ . The conjugacy class $\left( G _ { x } \right)$ of this subgroup is the orbit type of x. The set of all points with orbit type $( H )$ is $X _ { ( H ) }$ and the set of points fixed by all elements of a subgroup H is the fixed-point set $X ^ { H }$

These concepts distinguish between the global symmetry of the space and the intrinsic symmetry of an object. The group G represents the global symmetry, transforming the object between different reference frames. An orbit $G ( x )$ thus represents a single physical object in all of its possible orientations. This object possesses its own intrinsic symmetry, mathematically defined by the isotropy subgroup $G _ { x }$ for any point x on its orbit. While the specific subgroup is frame-dependent, its structure up to conjugation is constant. The orbit type $\left( G _ { x } \right)$ therefore serves as a reference-frame independent identifier that precisely describes the physical object’s intrinsic symmetry. The set of all possible orbit types, $\bar { \mathcal { O } _ { G } } ( X )$ , thus catalogs all distinct symmetries that objects in the space can possess.

Example 2.2. Now apply the setup from $E x . ~ 2 . 1 .$ Consider the action of $G = { \cal O } ( 3 ) \times { \cal S } _ { k + 1 }$ on centered point cloud space $X f o r k > 2 .$ . Let x  X be the set of vertices of a k-fold in the xOy-plane:

$$
x = ( x _ { 0 } , x _ { 1 } , . . . , x _ { k } ) , \quad w h e r e x _ { i } = ( \cos ( 2 i \pi / k ) , \sin ( 2 i \pi / k ) , 0 ) f o r i > 0 .\tag{2}
$$

with $x _ { 0 }$ at the origin. The generators of $G _ { x }$ include:(1) A rotation about the z-axis combined with a cyclic permutation $o f x _ { 1 } , \ldots , x _ { k } . \ ( 2 )$ A reflection across the xOz-plane combined with a product of transpositions. (3) A reflection across the xOy-plane combined with the identity.

Considering the projection map $\pi _ { \boldsymbol { X } } ( ( g , \sigma ) ) = g ,$ where $( g , \sigma )$ is a pair consisting of a geometric transformation $g \in O ( 3 )$ and a permutation $\sigma \in S _ { k + 1 }$ , we find that geometric symmetry $\pi _ { X } ( G _ { x } )$ is the dihedral group with horizontal reflection of order 4k, denoted by the Schoenflies symbol $D _ { k h }$

The symmetry of data can be altered by an equivariant map. The following theorem shows that an equivariant map does not decrease symmetry.

Theorem 2.3 (Curie’s principle, Kaba & Ravanbakhsh (2023), Thm. 1). Let $f : X \to Y$ be a G-equivariant map. For $\bar { \boldsymbol { x } } \in \bar { \boldsymbol { X } }$ , the isotropy subgroup ofx is contained in that ofits image $f ( x )$ , i.e.,

$$
G _ { x } \subseteq G _ { f ( x ) } .\tag{3}
$$

Such an increase in symmetry becomes unavoidable if the feature space Y cannot support the input’s symmetry. If the orbit type $\left( G _ { x } \right)$ is not present in $Y ~ ( \mathrm { i . e . , ~ } \bar { ( } G _ { x } ) \not \in ~ { \mathcal { O } } _ { G } ( Y ) )$ , then the equality $G _ { x } = G _ { f ( x ) }$ is impossible (otherwise, $G _ { f ( x ) }$ becomes an isotropy subgroup, implying $( G _ { x } ) \in { \mathcal { O } } _ { G } ( Y )$ , which leads to a contradiction), and the symmetry must therefore strictly increase. This increase leads to a degeneration, as any transformation g in the larger group $G _ { f ( x ) }$ that is not in $G _ { x }$ will map the distinct inputs x and $g ( x )$ to the same output, since $f ( g ( x ) ) = g ( \dot { f } ( x ) ) = f ( x )$ Fig. 2 illustrates three possible types of such degenerations for the k-fold structure from Ex. 2.2.

## 3 THE INFIMUM OF SYMMETRY

This section establishes that symmetry increase is governed by an infimum determined by the feature space. We prove the existence of this infimum and show that its coincidence with the input symmetry is a necessary condition for an equivariant map to preserve symmetry.

![](images/076bd79efe08dee7ae30a5e8362d6d23bdf09d74cba18a19a7c88b74eb07f6d8.jpg)

## 3.1 THE INFIMUM OF SYMMETRY

Figure 2: Three types of degeneration of k-fold.

The symmetry of data can increase after being transformed by an equivariant map. This increase can be an intentional design choice, or it can arise from subtle properties of the feature space. For instance, in the task from Ex. 2.1, the requirement for permutation-invariant features means that the permutation group $S _ { n }$ is naturally introduced into the isotropy subgroup of any output. This type of designed, unavoidable symmetry increase is formalized by the kernel of the group action on the feature space. We define the kernel of the action $\rho _ { X }$ as the set of group elements that fix every point in X: ker $\rho _ { X } : = \{ g \in G \mid g ( x ) = x , \forall x \in X \}$

Distinguishing between intentional and unintended symmetry increase is crucial. We begin by assuming that the group action is faithful, i.e., it has a trivial kernel with ke $: \rho _ { Y } = \{ e \} \quad$ . The case of a nontrivial kernel will be discussed in the next subsection.

To characterize the behavior of symmetry increase within a representation X, we first establish a partial order to compare different symmetries. An orbit type $\left( H _ { 1 } \right)$ is considered greater than or equal to another, $\left( H _ { 2 } \right)$ , written as $\left( H _ { 1 } \right) \overset { \cdot } { \geq } \left( H _ { 2 } \right)$ , if $H _ { 1 }$ contains a conjugate of $H _ { 2 }$ . This ordering reflects that a larger orbit type corresponds to a higher symmetry.

With this framework, we are interested in the lower bound of orbit types that can be reached from a point x with a specific isotropy group $H = G _ { x }$ . The analysis can be framed around any closed subgroup H, as the set of all possible isotropy subgroups is precisely the set of all closed subgroups of G (see, e.g., Field (2007); Mostow (1957)). The fixed-point space $\mathbf { \dot { X } } ^ { H }$ contains points of all higher orbit types. This leads to the following powerful theorem, which guarantees that the lower bound on symmetry increase is unique, corresponding to an isotropy subgroup that is unique up to conjugation.

Theorem 3.1 (Uniqueness of Minimal Type). Let X be a representation of a compact Lie group G. For any closed subgroup $H \subseteq G ,$ , a unique minimal orbit type exists among the points in the fixed-point subspace $\Breve { X ^ { H } }$ . In particular, $i f \left( H \right) \in \mathcal { O } _ { G } ( X ^ { H } )$ , then (H) is the minimal orbit type within that subspace.

The uniqueness guaranteed by this theorem allows us to define the symmetry infimum, denoted by $I _ { G } ( X , H )$ , as this unique minimal orbit type. In the context of symmetry increase, we are concerned with the relationship between $I ( Y , G _ { x } )$ and $\left( G _ { f ( x ) } \right)$ . An unexpected symmetry increase occurs if $( G _ { f ( x ) } ) > I _ { G } ( Y , G _ { x } )$ . The desired behavior for an equivariant map is captured by the following definition. For a map between G-sets X and Y, we define an isovariant map as one that strictly preserves symmetry for all $x \in X$

$$
G _ { x } = G _ { f ( x ) } .\tag{4}
$$

Using the concept of the symmetry infimum, we can provide a necessary condition for the existence of isovariant maps. In $\ S 5 . 1$ , we will see that this condition is in fact not sufficient for equivariant maps between representations, even when we assume a trivial kernel ker $\rho _ { Y } = \{ e \}$

Theorem 3.2. A necessary conditionfor the existence ofan isovariant map between G-sets X and Y is that $\mathcal { O } _ { G } ( X ) \subseteq \mathcal { O } _ { G } ( Y )$ . When X and Y are representations ofa compact Lie group G, this is equivalent to the condition that $I _ { G } ( Y , H ) = ( H )$ for all $( H ) \in { \mathcal { O } } _ { G } { \dot { ( X ) } }$ .

## 3.2 EQUIVARIANCE WITH NON-TRIVIAL KERNELS

When the feature space Y is restricted to have a non-trivial kernel, i.e., ker $\rho _ { Y } \neq \{ e \}$ , the definition of an isovariant map becomes too restrictive. Since every isotropy group $G _ { y }$ in $Y$ contains the kernel, any subgroup H not containing ker $\rho _ { Y }$ cannot occur as an isotropy subgroup. Consequently, for any map $f : X \to Y$ , an input isotropy subgroup $G _ { x }$ that does not contain the kernel must increase.

To formalize this unavoidable symmetry increase, we introduce an operator $p _ { Y }$ . In the discussion following Ex. 2.2, the presence of the $S _ { k + 1 }$ kernel forces the input isotropy subgroup $G _ { x }$ to become at least $D _ { k h } \times S _ { k + 1 }$ in the feature space. We generalize this observation. The operator $p _ { Y } ( H )$ is defined as the smallest subgroup containing H that is compatible with the action on $Y ,$ , given by the projection $p _ { Y } = \pi _ { Y } ^ { - 1 } \circ \pi _ { Y }$ , where $\pi _ { Y } : G \to G / \ker \rho _ { Y }$ is the natural projection. This operator is idempotent $( p _ { Y } ^ { 2 } = p _ { Y } )$ and maps any isotropy subgroup in $Y$ to itself. These properties reveal why increasing is unavoidable. For any equivariant map $f ,$ the relation $G _ { x } \subseteq G _ { f ( x ) }$ must hold. Applying the $p _ { Y }$ operator to this inclusion gives:

$$
G _ { x } \subseteq p _ { Y } ( G _ { x } ) \subseteq p _ { Y } ( G _ { f ( x ) } ) = G _ { f ( x ) } .\tag{5}
$$

This unavoidable increasing from $G _ { x } ~ { \mathrm { t o } } ~ p _ { Y } ( G _ { x } )$ means our goal is not to preserve $G _ { x }$ itself, but to ensure no additional symmetry is introduced beyond $p _ { Y } ( G _ { x } )$ . This leads to a more practical definition. We say a map ${ \dot { f } } : X \to Y$ is an isovariant map relative to $Y$ if for all $x \in X$ it satisfies

$$
p _ { Y } ( G _ { x } ) = G _ { f ( x ) } \quad \iff \quad \rho _ { Y } ( G _ { x } ) = \rho _ { Y } ( G _ { f ( x ) } ) .\tag{6}
$$

When the kernel is trivial, this definition reduces to that of a standard isovariant map. With this refined goal, we can state a necessary condition for the existence of such maps.

Theorem 3.3. A necessary condition for the existence of an isovariant map relative to Y from a G-set X to a G-set Y is that $( p _ { Y } ( H ) ) \in { \mathcal { O } } _ { G } ( Y )$ for every $( H ) \in { \mathcal { O } } _ { G } ( X )$ . When X and Y are representations, this is equivalent to the condition that $I _ { G } ( Y , \dot { H } ) \stackrel { \cdot } { = } ( p _ { Y } ( \dot { H } ) )$ for all $( H ) \in { \mathcal { O } } _ { G } ( X )$

![](images/e1c51bd39bdffc0e873a590306ec26fe1a5b5dfea9b10b1f78974a1c30274858.jpg)  
Figure 3: Visualization of representation spaces. (a) A k-fold structure is reoriented onto multiple planes. (b) Each is further rotated about the perpendicular axis. (c) All structures are embedded and projected into 2D. Marker shapes denote rotation axes, and colors denote rotation rates. Full degeneration appears at $l = 0 , 1$ , and axial degeneration at $l = 2 , 4$

## 4 COMPUTATION OF ORBIT TYPES

The computation of orbit types is a classical problem, with most established results on general representations, often in the context of bifurcation theory. In representation learning tasks, however, we utilize feature spaces containing high multiplicities of these representations, which requires us to supplement the existing computational frameworks.

## 4.1 ORBIT TYPES OF HIGH-MULTIPLICITY REPRESENTATIONS

For a compact Lie group $G ,$ , any representation space X can be uniquely decomposed into a direct sum of its irreducible components. For $G = S O ( 3 )$ , this decomposition is written as

$$
\begin{array} { r } { X \cong \bigoplus _ { l _ { 0 } = 0 } ^ { \infty } V _ { l = l _ { 0 } } ^ { \oplus m ( X , V _ { l = l _ { 0 } } ) } , } \end{array}\tag{7}
$$

where $V _ { l = l _ { 0 } }$ is the irreducible representation corresponding to the space of spherical harmonics of degree $l _ { 0 } ,$ , and $m ( X , V _ { l = l _ { 0 } } )$ is its multiplicity. For $\bar { G } = O ( \bar { 3 } )$ , the decomposition is similar, but the irreducible representations must also be distinguished by parity, denoted ${ \bar { V } } _ { l = l _ { 0 } ^ { + } }$ and $V _ { l = l _ { 0 } ^ { - } }$

We begin with the foundational criterion for identifying isotropy subgroups, first established as a necessary condition by Michel (1980)

Theorem 4.1 (Michel’s Criterion, Michel (1980), App. A). Let V be a representation of a group G. A necessary condition for a closed subgroup H to be an isotropy subgroup in V is that for any adjacent closed subgroup $\mathbf { \bar { \boldsymbol { H } } ^ { \prime } } \supset \mathbf { \boldsymbol { H } }$ , the dimension ofthefixed-point subspace strictly decreases:

$$
\dim V ^ { H ^ { \prime } } < \dim V ^ { H } .\tag{8}
$$

While this condition is not sufficient for all representations, its sufficiency can be guaranteed under certain common conditions. We define a representation V of a group G as a high-multiplicity representation if for every non-zero isotypic component corresponding to an irreducible representation $V _ { i } ,$ , its multiplicity $m ( V , V _ { i } )$ is greater than dim G.

Proposition 4.2. For a high-multiplicity representation V , the necessary condition stated in Thm. 4.1 is also sufficient.

This criterion is particularly powerful for two reasons. First, it offers a computationally convenient method in the form of a chain recursion, which only requires checking adjacent subgroups. The dimensions of the necessary fixed-point spaces can be calculated via the trace formula (Golubitsky et al., 1988). Second, the sufficiency condition is frequently met in our applications, as it holds for all finite groups and for feature spaces with a high number of channels. Based on the result of Prop. 4.2, we design an orbit type test Algo. 1 and a symmetry infimum calculation Algo. 2. Using the two algorithms described above, we have characterized all instances of symmetry increase for the closed subgroups of $S O ( 3 )$ or $O ( 3 )$ in the representations $V _ { l = l _ { 0 } } ^ { \oplus r }$ and $V _ { l = l _ { 0 } ^ { \pm } } ^ { \oplus r }$ for $r > \bar { 3 } .$ , respectively, see § C.4. We now illustrate our algorithms with a simple example.

Algorithm 1: Orbit Type Test for High- Algorithm 2: Symmetry Infimum Cal-  
Multiplicity Representations culation   
Data: Symmetry group $\overline { { G ; } }$ Data: Symmetry group $G ;$   
Closed subgroup $H \subset G ;$ Closed subgroup $H \subset G ;$   
High-Multiplicity Rep. V of $G$ Rep. $V$ of $G .$   
Result: $\mathrm { i } \mathsf { s \_ i n } ( ( \mathsf { \bar { H } } ) , \mathsf { \bar { O } } _ { G } ( \bar { V } ) )$ Result: Symmetry inf. $I _ { G } ( V , H )$   
1 Let set $S = \{ H _ { i } \} \subset G$ to be all adjacent 1 if $\mathrm { i } \mathsf { s \_ i n } \mathsf { \bar { ( } } ( H ) , \dot { \mathcal { O } } _ { G } ( V ) )$ then   
closed supergroups of H in $G ;$ 2 return (H);   
2 $\mathcal { O }  \mathcal { O } ;$ 3 end   
4 Let set $S = \{ H _ { i } \}$ to be all closed   
3 d dim $V ^ { H }$ supergroups of H in $G ;$   
4 for $H _ { i }$ in S do ${ \mathfrak { s } } \ { \mathcal { O } } \gets \emptyset ;$   
5 $d _ { H _ { i } } \gets$ dim $V ^ { H _ { i } } ;$ 6 for $H _ { i }$ in S do   
6 if $\dot { \boldsymbol d } _ { H } - \boldsymbol d _ { H _ { i } } = 0$ then 7 $\mathbf { i f } \ i \mathbf { s } \_ { - } \mathrm { i n } ( ( H _ { i } ) , { \mathcal { O } } _ { G } ( V ) )$ then   
7 return False; 8 Add $\left( \dot { H } _ { i } \right)$ to ;   
8 end 9 end   
9 end 10 end   
10 return True; 11 return min( );

Example 4.3. We illustrate our algorithms by calculating the orbit type and symmetry infimumfor the geometric symmetry $D _ { k h } \left( k > 2 \right)$ ofthe k-foldfrom Ex. 2.2, considered as a subgroup of $O ( 3 )$ The calculation is performed in the high-multiplicity representation space $Y = V _ { l = l _ { 0 } } ^ { \oplus r } \left( r > 3 , l _ { 0 } > 0 \right)$ . Here we provide only a sketch ofthe derivation, thefull procedure is provided in $\ S \ C . 3 .$

First, we apply the orbit type test from Algo. 1. This involves comparing the dimension of the fixed-point space of $D _ { k h }$ with that of its adjacent supergroups $( e . g . , D _ { p k , h }$ and,for $k = 4 , O _ { h } ) .$ . The analysis shows that $( D _ { k h } )$ is an orbit type ifand only $i f l _ { 0 } \ge$ k and $l _ { 0 } .$ , k have the same parity. Next, we apply the symmetry infimum calculation from Algo. 2. This requires identifying the minimal orbit type among all supergroups of $D _ { k h }$ , including non-adjacent ones like $D _ { \infty h }$ and $O ( 3 )$ . The final results are summarized in Table 1.

Table 1: The symmetry infimum $I _ { O ( 3 ) } ( V _ { l = l _ { 0 } } ^ { \oplus r } , D _ { k h } )$ for $k > 2 , r > 3 , l _ { 0 } > 0 .$
<table><tr><td></td><td colspan="2"> $l _ { 0 } < k$ </td><td colspan="2"> $k \leq l _ { 0 } < 2 k$ </td><td colspan="2"> $l _ { 0 } \geq 2 k$ </td></tr><tr><td></td><td> $l _ { 0 }$  is even</td><td> $l _ { 0 }$  is odd</td><td> $l _ { 0 }$  is even</td><td> $l _ { 0 }$  is odd</td><td>is even</td><td> $l _ { 0 }$  is odd</td></tr><tr><td>k is even</td><td> $( D _ { \infty h } )$ </td><td>(0(3))</td><td> $( D _ { k h } )$ </td><td>(0(3))</td><td> $( D _ { k h } )$ </td><td> $\left( O ( 3 ) \right)$ </td></tr><tr><td>k is odd</td><td> $( D _ { \infty h } )$ </td><td>(0(3))</td><td> $( D _ { \infty h } )$ </td><td> $( D _ { k h } )$ </td><td> $\left( D _ { 2 k h } \right)$ </td><td> $( D _ { k h } )$ </td></tr></table>

The analysis in Ex. 4.3 predicts three types of degeneration for the k-fold inputs from Ex. 2.1:

• Half Degeneration: The symmetry infimum of $G _ { x }$ is $\left( D _ { 2 k h } \times S _ { k + 1 } \right)$ . The feature cannot distinguish the k-fold from itself rotated by $\pi / k$ around the ${ z \mathbf { - } } \mathbf { a x i s }$

• Axial Degeneration: The symmetry infimum of $G _ { x }$ is $( D _ { \infty h } \times S _ { k + 1 } )$ . The feature cannot distinguish the k-fold from itself rotated by any angle around the z-axis.

• Full Degeneration: The symmetry infimum of $G _ { x }$ is $( O ( 3 ) \times S _ { k + 1 } )$ . The feature cannot distinguish the k-fold from itself rotated by any angle around any axis.

We consider encoding k-fold point clouds using equivariant neural networks and visualize the resulting embeddings. The three degenerations are experimentally verified in our visualizations, with Fig. 3 showing full and axial degeneration, and Fig. 4 showing axial and half degeneration. Although derived assuming high multiplicity $( r > 3 )$ in the feature representation, these predictions are identical for the single representation case $( r = 1 )$ , see $\ S \ C . 4$

a

c

![](images/96b87b9ffe30babfdf84c52dd28556135f76d9dda6931bd588bbde00df07744b.jpg)  
Figure 4: Visualization of representation spaces. (a) A k-fold (k is odd) structure is rotated $\pi / k$ about the perpendicular axis. (b) The path of symmetry increase. (c) Half degeneration appears at $l = 1 0$ At res = 98 and res = 49, the overall shape is identical, but the yellow data points (i.e. the second half of the rotation) completely cover the blue data points.

## 4.2 GUIDELINES FOR MANAGING SYMMETRY INCREASE

It is a general property of representations that the orbit types of a direct sum are related to those of its components by $\mathcal { O } _ { G } ( V _ { 1 } ) \cup \mathcal { O } _ { G } ( V _ { 2 } ) \subseteq \mathcal { O } _ { G } ( V _ { 1 } \oplus V _ { 2 } )$ , and $I _ { G } ( V _ { 1 } \oplus V _ { 2 } , H ) \le I _ { G } ( V _ { i } , H )$ for $i = 1 , 2$ , with equality conditions discussed in $\ S \dot { \mathrm { ~ C ~ } } . 5$ . These properties provide a direct mechanism for controlling the symmetry increase of an equivariant feature, that is to choose components whose symmetry infimum (computed as described in § C.4 for $G = S O ( 3 ) \mathrm { o r } O ( 3 ) )$ align with the desired behavior for task-relevant symmetries. Regarding the selection of the feature space $Y$ in equivariant representation learning, this principle translates into two guidelines.

For orientation-dependent tasks (e.g., § 6.2), when considering the kernel of feature space, it is crucial to avoid non-trivial symmetry increase (i.e. ensuring the map is relative isovariant) since such increases can lead to the accidental loss of orientational information. Therefore, for a given input symmetry (H), one should select feature components that contain the orbit type $\left( p _ { Y } ( H ) \right)$ .

For general tasks (e.g., § 6.3), certain forms of symmetry increase must be avoided, as the output symmetry reflects the dimensionality of the fixed-point subspace where the equivariant features lie (see Remark of Prop. C.2). In this context, one should generally avoid components where the symmetry infimum indicates a severe compression of the fixed-point subspace. Specifically, one must be cautious of components corresponding to non-trivial representations where the symmetry increases to the full group, as this causes the component to be annihilated and lose all discriminative power.

## 5 DENSITY OF (ALMOST) ISOVARIANT MAPS

We now connect the preceding theory to a practical machine learning context by introducing models for the data distribution and for the parameterized map. We show that the necessary conditions for isovariance established previously become sufficient under a relaxed definition of isovariance.

## 5.1 THE MANIFOLD HYPOTHESIS

Motivated by the manifold hypothesis and the broader considerations summarized in $\ S \ \mathrm { A } . 3 .$ , we model the data distribution as being supported on a finite union of smooth, compact submanifolds $M = \textstyle \bigcup _ { j } M _ { j }$ embedded in X. When a group G acts on X, this action equips each $M _ { j }$ with a natural G-manifold structure.

The central question is: when does an isovariant map $f : M \to Y$ exist? The existence is a non-trivial issue. The counterexample in Cex. D.3 demonstrates that the necessary condition of orbit type inclusion is not sufficient. Specifically, it shows that an isovariant map can fail to exist precisely because the multiplicities of the irreducible representations in the feature space are insufficient.

The non-existence of perfectly isovariant maps motivates a more practical, relaxed definition: a map that is isovariant almost everywhere. To formalize this, we equip M with the d-dimensional Hausdorff measure $\mu _ { M }$ , where $d { \overset { \cdot } { = } } \operatorname* { m a x } _ { j } \{ \dim M _ { j } \}$ . This allows us to identify subsets of measure zero as negligible. Note that $M _ { ( H ) }$ is a finite union of submanifolds, then $f$ is almost isovariant relative to $Y$ if for every orbit type (H) in the data support, the isovariance condition

$$
\rho _ { Y } ( G _ { x } ) = \rho _ { Y } ( G _ { f ( x ) } ) ,\tag{9}
$$

holds for all points $x \in M _ { ( H ) }$ except for a subset of $\mu _ { M _ { ( H ) } }$ -measure zero. This ensures that any undesired increase in symmetry occurs only on a negligible portion of the data.

## 5.2 GENERICITY OF (ALMOST) ISOVARIANT MAPS

For Ex. 2.1, we select TFN (Thomas et al., 2018), a classic ENN based on tensor products, as our parameterized model. We provide the complete formulation to § D.2. An important property of TFN parameterizations $\mathcal { F } _ { \mathrm { T F N } }$ is that they satisfy a universal approximation theorem (Dym & Maron, 2021). In topology, this is equivalent to $\mathcal { F } _ { \mathrm { T F N } }$ being dense in equivariant function space $C _ { G } ( X , Y )$ with respect to the $C ^ { 0 }$ topology. In fact, we can establish a stronger approximation theorem.

Theorem 5.1. In Ex. 2.1, thefunctionfamilies $\mathcal { F } _ { \mathrm { T F N } }$ with smooth non-polynomial activation function are $C ^ { \infty }$ -dense in the space ofsmooth equivariant maps $C _ { G } ^ { \infty } ( X , Y )$ . That is,for any integer $r \geq 0 ,$ any map $f \in C _ { G } ^ { \infty } ( X , \bar { Y } )$ , any compact set $K \subset X$ , and any $\epsilon > 0 ,$ , there exists $g \in \mathcal { F } _ { \mathrm { T F N } }$ such that

$$
\begin{array} { r } { \operatorname* { m a x } _ { x \in K } \left\| D ^ { k } f ( x ) - D ^ { k } g ( x ) \right\| < \epsilon , k \le r . } \end{array}\tag{10}
$$

Here, $D ^ { k }$ denotes the k-th order total derivative operator. A significant portion of maps within a dense parameterization reflects the generic properties of the mapping space. For equivariant maps, a key generic property, closely related to almost isovariance, is that the dimension of the set of points where the orbit type is increase from $\left( H \right) \mathsf { t o } \left( H ^ { \prime } \right)$ by a map f is constrained for a generic map. The following theorem shows that for expressive models with $C ^ { \infty }$ approximation capabilities, such as the TFN discussed, almost isovariance is a generic property, and full relative isovariance can be achieved by increasing representation multiplicity. As shown in Cex. D.3, this requirement is tight.

Theorem 5.2. Let be a equivariant parametrization with $C ^ { \infty }$ approximation capability. Iffor every $( H ) \in { \mathcal { O } } _ { G } ( M )$ we have $( p _ { Y } ( H ) ) \in { \mathcal { O } } _ { G } ( Y )$ , then for any finite union of compact, smooth G-submanifolds $M \subset X , a n y f \in C _ { G } ^ { \infty } ( X , Y )$ , any integer $r \geq 0 ,$ and any $\epsilon > 0 ,$ , there exists a map $g \in { \mathcal { F } }$ such that

$$
\begin{array} { r } { \operatorname* { m a x } _ { x \in M } \lVert D ^ { k } f ( x ) - D ^ { k } g ( x ) \rVert < \epsilon , k \le r , } \end{array}\tag{11}
$$

and $g | _ { M }$ is almost isovariant relative to $Y .$ Furthermore, ifthefeature space Y contains a representation $\widetilde { Y } ^ { \oplus r }$ for an integer $r > \operatorname* { m a x } _ { j } \{ \dim M _ { j } \}$ , where Y satisfies the condition $( p _ { Y } ( H ) ) \in { \mathcal { O } } _ { G } ( Y )$ then the approximating map g can be chosen to be isovariant relative to $Y .$

## 6 EXPERIMENT

We validate our theoretical analysis through three experiments: representation-space visualizations in $\ S \ G . 1$ , a geometric graph discrimination task in $\ S \ 6 . 2 .$ , and a molecular isotropic polarizability prediction task on QM9 (Ramakrishnan et al., 2014) in § 6.3. Across all experiments, we consider two equivariant architectures, TFN (Thomas et al., 2018) and HEGNN (Cen et al., 2024): TFN is used in the first two experiments, whereas HEGNN is employed in the last two. All detailed experimental settings are provided in § F.

## 6.1 VISUALIZATION OF REPRESENTATION SPACE

To provide a clearer illustration of our theory, we present visualizations of the representation spaces of different degrees, obtained from the 3-fold structure.

Dataset. We first construct a k-fold structure lying in the xOy plane (here $k = 3 )$ , and then apply random rotations to place it on m distinct planes (here $m = 6 )$ , as illustrated in Fig. 4(a).

Subsequently, as shown in Fig. 4(b), each structure is further rotated about the axis perpendicular to its plane. The rotation angle $\theta \in [ 0 , 2 \pi / k )$ is uniformly discretized into res candidate values, defined as $\{ 2 \pi i / ( k \cdot \mathrm { r e s } ) \} _ { i = 0 } ^ { \mathrm { r e s } - 1 }$ . Unless otherwise stated, we use res = 49 to verify the half-degeneration. We also consider the doubled resolution $\mathrm { r e s } = 9 8 .$

Embeddings. For the resulting m res candidate structures, we compute the graph-level features via randomly initialized single-layer TFN (Thomas et al., 2018) with detailed setting in § F.1. For $l _ { 0 } = 0$ the feature dimension is 1; for visualization we set the second coordinate to zero. For $l _ { 0 } \geq 1$ the feature dimension is $2 l _ { 0 } + 1 > 2 ;$ we reduce dimensionality via random projection and then rescale all features to a common range so that visualizations are comparable. Data points are plotted in ascending order of plane index; for structures on the same plane they are ordered by increasing rotation angle about the axis.

Results. Detailed experimental results are presented in Figs. 3 and 4. The former shows the input symmetry with $l _ { 0 } \le \bar { 1 } 1$ 1 increase to $( O ( 3 ) \bar { \times } S _ { k + 1 } )$ (Full degeneration, $l _ { 0 } = 0 , 1 ) , ( D _ { \infty h } \times S _ { k + 1 } )$ (Axial degeneration, $l _ { 0 } = 2 , 4 )$ , or remains non-degenerate. The latter shows the symmetry increase to $\left( D _ { 2 k h } \times S _ { k + 1 } \right)$ at $l _ { 0 } = 1 0$ . These experimental results are consistent with Ex. 4.3.

## 6.2 EXPRESSIVITY ON SYMMETRIC GRAPHS

To experimentally validate our theoretical conclusion established , we design a more comprehensive experiment following Joshi et al. (2023).

Dataset. We construct four symmetric k-fold structures $( k \in 2 , 3 , 4 , 6 )$ , each centered at the origin. For each structure $\mathcal { G } _ { 0 }$ we apply a random rotation to obtain $\mathcal { G } _ { 1 }$ , ensuring that $\mathcal { G } _ { 1 }$ does not coincide with the original $\mathcal { G } _ { 0 }$ . The goal is to evaluate whether different ENNs can distinguish $\mathcal { G } _ { 0 }$ from $\mathcal { G } _ { 1 }$ . To probe different aspects of our theory, we treat 2D and 3D rotations separately; in the 3D setting we additionally require that $\mathcal { G } _ { 1 }$ is not coplanar with $\mathcal { G } _ { 0 }$

![](images/fec32f6b50b42eff5a6dd39897a31e04793489df172edd1ca10ec44336f06354.jpg)  
Figure 5: Heatmap of Emb. Diff. Norm.

Embeddings. We employed both TFN (Thomas et al., 2018) and HEGNN (Cen et al., 2024) to compute the norm of the embedding difference across 12 configurations for each, varying by the number of irrep channels (1, 4, 16) and layers (1-4). The extracted $l _ { 0 } .$ -degree embeddings are evaluated via the norm of the difference between the embeddings of $\mathcal { G } _ { 0 }$ and $\mathcal { G } _ { 1 }$ as in Cen et al. (2024) with detailed setting in § F.2.1. When this norm approaches zero, the two embeddings are numerically indistinguishable, and hence the corresponding geometric figures cannot be told apart by the model <sup>3</sup>.

Results. The maximum value was selected for each configuration and visualized in a heatmap, as shown in Fig. 5. The results exhibit a clear binary pattern: the values are either greater than $1 0 ^ { - 3 }$ or less than $1 0 ^ { - 6 }$ (due to numerical error), with a difference of more than $1 0 ^ { 3 }$ times. This suggests that values exceeding $1 0 ^ { - 3 }$ indicate distinguishable structures, while those below $1 0 ^ { - 6 }$ correspond to indistinguishable structures. These findings align precisely with our theoretical predictions. Furthermore, as the maximum value was chosen, with all norms being less than $1 0 ^ { - 6 }$ this phenomenon is shown to be independent of the model choice, the number of channels or layers.

## 6.3 MOLECULE PROPERTY PREDICTION WITH PRETRAINED EQUIVARIANT FEATURES

To illustrate the guiding significance of the theory presented in this paper for practical applications, we designed experiments on the QM9 dataset (Ramakrishnan et al., 2014) for verification.

Dataset. We choose to predict the molecular isotropic polarizability α. It is worth noting that the QM9 dataset (Ramakrishnan et al., 2014) contains many highly symmetric structures spanning 22

![](images/e2697da730379ad3802ac799385492ab2be570478ef293a49e1452574e0cf642.jpg)  
Figure 6: MAE loss (in units of $a _ { 0 } ^ { 3 } )$ for isotropic polarizability prediction with degree $l = l _ { 0 }$ across molecules from the top-16 point groups by molecular count. Each boxplot shows the distribution of errors at a given degree, while diamond markers denote the corresponding mean MAE.

molecular symmetry groups <sup>4</sup>, with Fig. 6 reporting sample counts for the 16 most frequent groups;   
the remaining seven are $D _ { 3 }$ with 2 samples and $C _ { 4 } ^ { \bar { ( } } , D _ { 2 } , \mathsf { ^ { - } D _ { 6 h } } , O _ { h } , S _ { 4 }$ , each appearing only once.

Embeddings. We adopt HEGNN (Cen et al., 2024) as our backbone. Specifically, we first pretrain on features of $l \leq 1 1$ to obtain a shared equivariant feature encoder, ensuring that all subsequent configurations operate on the same embedding space. We then consider two fine-tuning strategies: (a) using only features of $l = l _ { 0 }$ , and (b) using all features of $l \leq l _ { 0 } .$ , yielding 12 distinct prediction heads for each. The detailed experimental setup is given in § F.3.1.

Results. The results are shown in Fig. 6 and § F.3.2, with a summary of all symmetry increases in Table 22. We note that for the majority of samples, different feature components contribute similarly to the prediction. However, as illustrated in Fig. 6, for non-trivial feature components where molecular symmetry increase to $O ( 3 )$ , the prediction loss is substantially higher. Furthermore, the results in § F.3.2 show that for symmetries causing full degeneration in 1-degree features, including additional 1-degree features may not provide significant improvement to the model’s prediction performance. This validates the design guidelines in § 4.2. Detailed case studies are provided in § F.3.2.

## 7 CONCLUSION

In this work, we presented a rigorous mathematical framework to address the critical issue of symmetry increase in ENNs. We introduced the concept of the symmetry infimum, a computable lower bound for any increase in symmetry determined by the feature space. Our central contribution is to show that this infimum can be used to precisely predict and control the expressive degradation of ENNs. The framework successfully explains phenomena in settings like those of Joshi et al. (2023), which could not be fully accounted for by prior theories such as the collapse-to-zero model from Cen et al. (2024). Our findings provide both a robust theoretical understanding and practical guidelines for designing more reliable ENNs.

## ACKNOWLEDGMENTS

This work was supported by the National Natural Science Foundation of China (Nos. 62276269, 92270118, and 62376276), the Beijing Natural Science Foundation (No. 1232009), and the Beijing Nova Program (No. 20230484278).

## AUTHOR CONTRIBUTIONS

Ning Lin organized this project. Ning Lin led the theoretical development in § 2–§ 5 and was responsible for the theoretical proofs of the corresponding part. Ning Lin and Jiacheng Cen jointly led the experimental studies in § 6. Jiacheng Cen was responsible for model implementation and code development. Anyi Li was responsible for data processing. Jiacheng Cen and Anyi Li jointly contributed to figure preparation and visualization. Wenbing Huang and Hao Sun jointly supervised and guided the project. All authors participated in writing and revising the manuscript.

## USAGE OF LARGE LANGUAGE MODELS

We only use Large Language Models to polish our writing.

## ETHICS STATEMENT

This work is a theoretical contribution in the domain of equivariant neural networks, focusing on mathematical properties of symmetry preservation and transformation under equivariant mappings. The research does not involve human subjects, personal data, or real-world deployments, and therefore does not raise concerns related to privacy, fairness, bias, or potential misuse.

We affirm that this work adheres to the ICLR Code of Ethics. In particular, we have ensured honesty in representing our contributions, accuracy in reporting our findings, and proper attribution of prior work. As this is a theoretical study, there are no conflicts of interest, sponsorship influences, or applications with foreseeable societal harms to disclose. We support responsible stewardship of machine learning research and believe that foundational advances such as ours contribute positively to the scientific community by enhancing understanding and enabling future trustworthy systems.

## REPRODUCIBILITY STATEMENT

We are committed to ensuring the reproducibility of both the theoretical and experimental components of our work. To support the reproducibility of our theoretical results, we provide complete and self-contained proofs for all main theorems and propositions in the appendix, including detailed derivations and necessary mathematical background. These proofs clarify all assumptions and logical steps required to verify our claims. For the experimental component, we have made our implementation code publicly available at https://github.com/GLAD-RUC/SymInc. The codebase contains clear documentation and scripts that fully reproduce the reported results. All experimental settings, hyperparameters, and data generation procedures are described in detail within the supplementary materials. Together, the comprehensive theoretical appendices and open-sourced code ensure that our findings can be rigorously verified and built upon by the research community.

## REFERENCES

V.I. Arnold, S.M. Gusein-Zade, and A.N. Varchenko. Singularities of Differentiable Maps: Classification of Critical Points, Caustics and Wave Fronts, volume 1. Birkhäuser Boston, 2012. doi: 10.1007/978-0-8176-8340-5. URL https://link.springer.com/10.1007/ 978-0-8176-8340-5.

M. I. Aroyo. International Tables for Crystallography: Space-group symmetry, volume A. International Union of Crystallography, 2 edition, 2016. ISBN 978-0-470-97423-0. doi: 10.1107/97809553602060000114. URL https://it.iucr.org/Ac/.

Sarp Aykent and Tian Xia. GotenNet: Rethinking Efficient 3D Equivariant Graph Neural Networks. In The Thirteenth International Conference on LearningRepresentations, 2025.

Perla Azzi, Rodrigue Desmorat, Julien Grivaux, and Boris Kolev. Rationality of normal forms of isotropy strata of a representation of a compact lie group. arXiv preprint arXiv:2301.08599v1, 2023.

Perla Azzi, Rodrigue Desmorat, Julien Grivaux, and Boris Kolev. On the isotropy stratification of a real representation of a compact lie group. arXiv preprint arXiv:2301.08599v2, 2026.

Edward Bierstone. General position of equivariant maps. Transactions ofthe American Mathematical Society, 234(2):447–466, 1977.

Glen E. Bredon. Introduction to compact transformation groups, volume 46. Academic Press, 1972.

Michael M. Bronstein, Joan Bruna, Taco Cohen, and Petar Velickoviˇ c. Geometric deep learning:´ Grids, groups, graphs, geodesics, and gauges, 2021.

Bradley CA Brown, Anthony L Caterini, Brendan Leigh Ross, Jesse C Cresswell, and Gabriel Loaiza-Ganem. Verifying the union of manifolds hypothesis for image data. arXiv preprint arXiv:2207.02862, 2022.

Bin Cao, Yang Liu, Longhan Zhang, Yifan Wu, Zhixun Li, Yuyu Luo, Hong Cheng, Yang Ren, and Tongyi ZHANG. Beyond structure: Invariant crystal property prediction with pseudo-particle ray diffraction. In The Fourteenth International Conference on Learning Representations, 2026.

Abel Carreras. pointgroup: Python library to determine the point symmetry group of molecular geometries, 2025. URL https://github.com/abelcarreras/pointgroup.

Jiacheng Cen, Anyi Li, Ning Lin, Yuxiang Ren, Zihe Wang, and Wenbing Huang. Are high-degree representations really unnecessary in equivariant graph neural networks? In Advances in Neural Information Processing Systems, volume 37, pp. 26238–26266, 2024.

Jiacheng Cen, Anyi Li, Ning Lin, Tingyang Xu, Yu Rong, Deli Zhao, Zihe Wang, and Wenbing Huang. Universally invariant learning in equivariant GNNs. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025.

Alexandre Agm Duval, Victor Schmidt, Alex Hernández-Garcıa, Santiago Miret, Fragkiskos D Malliaros, Yoshua Bengio, and David Rolnick. Faenet: Frame averaging equivariant gnn for materials modeling. In International Conference on Machine Learning, pp. 9013–9033. PMLR, 2023.

Nadav Dym and Haggai Maron. On the universality of rotation equivariant point cloud networks. In International Conference on Learning Representations, 2021.

Jinjia Feng, Zhewei Wei, Taifeng Wang, and Zongyang Qiu. TetraGT: Tetrahedral geometry-driven explicit token interactions with graph transformer for molecular representation learning. In The Fourteenth International Conference on Learning Representations, 2026.

Matthias Fey and Jan Eric Lenssen. Fast graph representation learning with pytorch geometric. arXiv preprint arXiv:1903.02428, 2019.

Michael Field. Dynamics and symmetry, volume 3. World Scientific, 2007.

Mario Geiger and Tess Smidt. e3nn: Euclidean neural networks. arXiv preprint arXiv:2207.09453, 2022.

Martin Golubitsky and Victor Guillemin. Stable mappings and their singularities. Number 14 in Graduate texts in mathematics. Springer, 1973. ISBN 978-0-387-90073-5. doi: 10.1007/ 978-1-4615-7904-5.

Martin Golubitsky, Ian Stewart, and David G. Schaeffer. Singularities and Groups in Bifurcation Theory, volume 2, number 69 of Applied Mathematical Sciences. Springer New York, 1988. ISBN 978-1-4612-8929-6 978-1-4612-4574-2. doi: 10.1007/978-1-4612-4574-2.

Ian Goodfellow, Yoshua Bengio, and Aaron Courville. Deep Learning, volume 1. MIT Press, 2016.

Victor Guillemin and Alan Pollack. Differential topology. Prentice Hall, 1974. ISBN 978-0-13- 212605-2.

Jiaqi Han, Jiacheng Cen, Liming Wu, Zongzhao Li, Xiangzhe Kong, Rui Jiao, Ziyang Yu, Tingyang Xu, Fandi Wu, Zihe Wang, et al. A survey of geometric graph neural networks: Data structures, models and applications. Frontiers of Computer Science, 19(11):1911375, 2025.

Morris W. Hirsch. Differential topology. Number 33 in Graduate texts in mathematics. Springer-Verlag, 1976. ISBN 978-1-4684-9451-8. doi: 10.1007/978-1-4684-9449-5.

Wenbing Huang and Jiacheng Cen. Geometric graph learning for drug design. Deep Learning in Drug Design, pp. 133–151, 2026.

E Ihrig and Martin Golubitsky. Pattern selection with O(3) symmetry. Physica D: Nonlinear Phenomena, 13(1-2):1–33, 1984.

Chaitanya K Joshi, Cristian Bodnar, Simon V Mathis, Taco Cohen, and Pietro Lio. On the expressive power of geometric graph neural networks. In International conference on machine learning, pp. 15330–15355. PMLR, 2023.

Sékou-Oumar Kaba and Siamak Ravanbakhsh. Symmetry breaking and equivariant neural networks. In NeurIPS 2023 Workshop on Symmetry and Geometry in Neural Representations, 2023.

John M. Lee. Introduction to Smooth Manifolds, volume 218 of Graduate Texts in Mathematics. Springer, 2012. doi: 10.1007/978-1-4419-9982-5.

Anyi Li, Jiacheng Cen, Songyou Li, Mingze Li, Yang Yu, and Wenbing Huang. Geometric mixture models for electrolyte conductivity prediction. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025a.

Qi Li, Rui Jiao, Liming Wu, Tiannian Zhu, Wenbing Huang, Shifeng Jin, Yang Liu, Hongming Weng, and Xiaolong Chen. Powder diffraction crystal structure determination using generative models. Nature Communications, 16(1):7428, 2025b.

Zongzhao Li, Jiacheng Cen, Wenbing Huang, Taifeng Wang, and Le Song. Size-generalizable RNA structure evaluation by exploring hierarchical geometries. In The Thirteenth International Conference on Learning Representations, 2025c.

M. J. Linehan and G. E. Stedman. Little groups of irreps of O(3), SO(3), and the infinite axial subgroups. Journal ofPhysics A: Mathematical and General, 34(34):6663–6688, 2001.

Eckhard Meinrenken. Group actions on manifolds, 2003. URL https://www.math.toronto. edu/mein/teaching/LectureNotes/action.pdf.

Louis Michel. Symmetry defects and broken symmetry. configurations hidden symmetry. Reviews of Modern Physics, 52(3):617–651, 1980. doi: 10.1103/RevModPhys.52.617.

John Willard Milnor. Topology from the differentiable viewpoint. Univ. Pr. of Virginia, 8. print edition, 1990. ISBN 978-0-8139-0181-7.

G. D. Mostow. Equivariant embeddings in euclidean space. The Annals of Mathematics, 65(3):432, 1957. doi: 10.2307/1970055.

Ikumitsu Nagasaki. The weak isovariant borsuk-ulam theorem for compact lie groups. Archiv der Mathematik, 81(3):348–359, 2003. doi: 10.1007/s00013-003-4693-1.

Allan Pinkus. Approximation theory of the mlp model in neural networks. Acta numerica, 8:143–195, 1999.

Omri Puny, Matan Atzmon, Edward J. Smith, Ishan Misra, Aditya Grover, Heli Ben-Hamu, and Yaron Lipman. Frame averaging for invariant and equivariant network design. In International Conference on Learning Representations, 2022.

Raghunathan Ramakrishnan, Pavlo O Dral, Matthias Rupp, and O Anatole Von Lilienfeld. Quantum chemistry structures and properties of 134 kilo molecules. Scientific data, 1(1):1–7, 2014.

Vıctor Garcia Satorras, Emiel Hoogeboom, and Max Welling. E (n) equivariant graph neural networks. In International Conference on Machine Learning. PMLR, 2021.

Tess E Smidt, Mario Geiger, and Benjamin Kurt Miller. Finding symmetry breaking order parameters with euclidean neural networks. Physical Review Research, 3(1):L012002, 2021.

Nathaniel Thomas, Tess Smidt, Steven Kearnes, Lusann Yang, Li Li, Kai Kohlhoff, and Patrick Riley. Tensor field networks: Rotation-and translation-equivariant neural networks for 3d point clouds. arXiv preprint arXiv:1802.08219, 2018.

D. J. A. Trotman. Counterexamples in stratification theory: two discordant horns. Real and complex singularities, Oslo, pp. 679–686, 1976.

Stephen Willard. General topology. Addison-Wesley series in mathematics. Addison-Wesley, 1970. ISBN 978-0-201-08707-9.

Liming Wu, Zhichao Hou, Jirui Yuan, Yu Rong, and Wenbing Huang. Equivariant spatio-temporal attentive graph networks to simulate physical dynamics. Advances in Neural Information Processing Systems, 36, 2023.

Liming Wu, Wenbing Huang, Rui Jiao, Jianxing Huang, Liwei Liu, Yipeng Zhou, Hao Sun, Yang Liu, Fuchun Sun, Yuxiang Ren, et al. Siamese foundation models for crystal structure prediction. arXiv preprint arXiv:2503.10471, 2025.

Liming Wu, Rui Jiao, Qi Li, Mingze Li, Songyou Li, Shifeng Jin, and Wenbing Huang. Dmflow: Disordered materials generation by flow matching. arXiv preprint arXiv:2602.04734, 2026.

Siyuan Zeng, Kuanping Gong, Yongquan Jiang, and Yan Yang. A method for predicting molecular point group based on graph neural networks. Artificial Intelligence Chemistry, pp. 100097, 2025.

Yuelin Zhang, Jiacheng Cen, Jiaqi Han, Zhiqiang Zhang, JUN ZHOU, and Wenbing Huang. Improving equivariant graph neural networks on large geometric graphs via virtual nodes learning. In Forty-first International Conference on Machine Learning, 2024.

Yuelin Zhang, Jiacheng Cen, Jiaqi Han, and Wenbing Huang. Fast and distributed equivariant graph neural networks by virtual node learning. arXiv preprint arXiv:2506.19482, 2025.

## CONTENTS OF APPENDIX

A Backgrounds 16   
A.1 Equivariant Neural Networks 16   
A.2 Closed Subgroups of SO(3) or O(3) 16   
A.3 Manifolds and Basic Data Assumption 17   
B Proofs of The Infimum of Symmetry (§ 3) 19   
B.1 Proof of Thm. 3.1 19   
B.2 Proof of Thm. 3.3 20   
C Symmetry Increase for G = SO(3) or O(3) in § 4 21   
C.1 General Orbit Type Criterion 21   
C.2 Proof of Prop. 4.2 21   
C.3 Detailed Calculation in Ex. 4.3 22   
C.4 Calculation of Symmetry Infimum 23   
C.5 Composition Property of High-Multiplicity Representation 24   
D Proofs of Density of (Almost) Isovariant Maps (§ 5) 27   
D.1 Counterexample in § 5.1 27   
D.2 Proof of Thm. 5.1 27   
D.3 Some Results on Topology 31   
D.4 Generic Equivariant Mappings 33   
D.5 Proof of Thm. 5.2 38   
E Tables 39   
E.1 Minimal Proper Supergroups in SO(3) or O(3) 39   
E.2 Dimensions of Fixed-point Subspaces for Subgroups of SO(3) or O(3) 40   
E.3 Symmetry Infimum for Subgroups of SO(3) 41   
E.4 Symmetry Infimum for Subgroups of O(3) 42   
F Detailed Experiment 45   
F.1 Visualization of Representation Space 45   
F.2 Expressivity on Symmetric Graphs 45   
F.2.1 Embedding Difference Norm Experiment 45   
F.2.2 Original GWL-Test on Symmetric Graphs 45   
F.3 Molecule Property Prediction with Pretrained Equivariant Features 46   
F.3.1 Detailed Experimental Setup 46   
F.3.2 Case Studies 46

## A BACKGROUNDS

## A.1 EQUIVARIANT NEURAL NETWORKS

Equivariant Neural Networks (ENNs) have emerged as a cornerstone of modern machine learning, enabling a wide range of applications across the sciences (Han et al., 2025; Wu et al., 2023; Li et al., 2025b; Wu et al., 2025; Li et al., 2025c; Feng et al., 2026; Cao et al., 2026; Wu et al., 2026; Li et al., 2025a). Most mainstream ENNs adopt tensor product operators to design the message passing architecture (Thomas et al., 2018). In particular, since scalarization operations $( e . g .$ , norm and inner product) can substantially reduce computational cost, Cartesian vector-based networks (Satorras et al., 2021) and spherical scalarization networks (Cen et al., 2024; Aykent & Xia, 2025) have become particularly popular. More specifically for asymmetric graph structures, Cen et al. (2025) point out that networks employing scalarization operations already possess sufficient expressive power, i.e., universal approximation.

However, additional subtleties arise in the presence of input symmetries. The interaction between architectural components and symmetric structures can lead to nontrivial representational effects. For example, techniques such as global virtual nodes (Zhang et al., 2024; 2025) and reference frames (Puny et al., 2022; Duval et al., 2023), though effective in improving model capacity, may exhibit unintended behaviors when applied to symmetric inputs. In particular, their use can potentially induce symmetry increase. In this work, we systematically analyze these phenomena under symmetric structures for general equivariant neural network architectures.

## A.2 CLOSED SUBGROUPS OF $S O ( 3 )$ OR O(3)

Before proceeding, we briefly review the classification of the closed subgroups of $S O ( 3 )$ or $O ( 3 )$ The closed subgroups of $O ( 3 )$ , also known as point groups, are classified up to conjugacy as follows. Throughout this paper, we use the Schoenflies notation to denote these groups.

• Finite Subgroups: These are divided into axial and polyhedral groups. The axial groups include the Abelian subgroups $( C _ { k } , S _ { 2 k } , C _ { k h } )$ and the non-Abelian subgroups $( C _ { k v } , D _ { k } , D _ { k h } , D _ { k d } )$ . The polyhedral groups are all non-Abelian and comprise seven families $( T , T _ { d } , T _ { h } , O , O _ { h } , I , I _ { h } )$ . Among these, the groups $C _ { k } , D _ { k } , T , O , I$ consist solely of pure rotations and are subgroups of $S O ( 3 )$

• Infinite Subgroups: These include the cylindrical groups $( C _ { \infty } , C _ { \infty h } , C _ { \infty v } , D _ { \infty } , D _ { \infty h } )$ which arise as limits of the axial groups, and the spherical groups, which are $K = S O ( 3 )$ and $K _ { h } = O ( 3 )$

To facilitate the identification of input symmetries, we now provide a brief overview of the elemental structure of the key finite subgroups. The identification of cylindrical symmetries, which are infinite, can be achieved by taking the limits of these finite axial groups.

## ABELIAN AXIAL GROUPS

• $C _ { k }$ (Cyclic Group): Generated by a rotation $c _ { k }$ of angle $2 \pi / k$ , with k elements:

(1) Rotations $c _ { k } ^ { j }$ about the principal axis.

$S _ { 2 k }$ (Rotation-Reflection Group): Generated by adding a rotation-reflection element $c _ { 2 k } \sigma _ { h }$ to $C _ { k } .$ , with 2k elements:

(1) Rotations $c _ { k } ^ { j }$ about the principal axis.

(2) Rotation-reflections $c _ { 2 k } ^ { 2 j + 1 } \sigma _ { h }$ about the principal axis.

$C _ { k h }$ (Cyclic Groups with Horizontal Reflection): Generated by adding a horizontal reflection plane $\sigma _ { h }$ to $C _ { k }$ , with 2k elements:

(1) Rotations $c _ { k } ^ { j }$ about the principal axis.

(2) Rotation-reflections $c _ { k } ^ { j } \sigma _ { h }$ about the principal axis.

## NON-ABELIAN AXIAL GROUPS

$C _ { k v }$ (Cyclic Groups with Vertical Reflections): Generated by adding a vertical reflection plane $\sigma _ { v }$ to $C _ { k }$ , with 2k elements:

(1) Rotations $c _ { k } ^ { j }$ about the principal axis.

(2) Reflections $c _ { k } ^ { j } \sigma _ { v }$ about the vertical plane.

$D _ { k }$ (Dihedral Groups): Generated by adding a 2-fold rotation axis $u _ { 2 }$ perpendicular to the principal axis of $C _ { k }$ , with 2k elements:

(1) Rotations $c _ { k } ^ { j }$ about the principal axis.

(2) 2-fold rotations $c _ { k } ^ { j } u _ { 2 }$ about horizontal axes.

$D _ { k h }$ (Dihedral Groups with Horizontal Reflection): Generated by adding a horizontal reflection plane $\sigma _ { h }$ to $D _ { k }$ , with 4k elements:

(1) Rotations $c _ { k } ^ { j }$ about the principal axis.

(2) 2-fold rotations $c _ { k } ^ { j } u _ { 2 }$ about horizontal axes.

(3) Rotation-reflections $c _ { k } ^ { j } \sigma _ { h }$ about the principal axis.

(4) Reflections $c _ { k } ^ { j } \sigma _ { v }$ about the vertical plane.

$D _ { k d }$ (Dihedral Groups with Dihedral Reflections): Generated by adding a diagonal reflection plane $\sigma _ { d } = c _ { 2 k } \sigma _ { v }$ to $D _ { k }$ , with 4k elements:

(1) Rotations $c _ { k } ^ { j }$ about the principal axis.

(2) 2-fold rotations $c _ { k } ^ { j } u _ { 2 }$ about horizontal axes.

(3) Rotation-reflections $c _ { 2 k } ^ { 2 j + 1 } \sigma _ { h }$ about the principal axis.

(4) Reflections $c _ { 2 k } ^ { 2 j + 1 } \sigma _ { v }$ about the diagonal plane.

## POLYHEDRAL GROUPS

• T (Tetrahedral Group): The rotational symmetry group of a tetrahedron, with 12 elements.

• $T _ { d }$ (Full Tetrahedral Group): The full symmetry group of a tetrahedron, with 24 elements.

• $T _ { h }$ (Pyritohedral Group): Generated by adding an inversion center to $T ,$ , with 24 elements.

• O (Octahedral Group): The rotational symmetry group of a cube or octahedron, with 24 elements.

$O _ { h }$ (Full Octahedral Group): The full symmetry group of a cube or octahedron, generated by adding an inversion center to O, with 48 elements.

• I (Icosahedral Group): The rotational symmetry group of an icosahedron, with 60 elements.

$I _ { h }$ (Full Icosahedral Group): The full symmetry group of an icosahedron, generated by adding an inversion center to I, with 120 elements.

The subgroup relations among point groups, as well as the dimensions of fixed-point spaces of O(3) representations, can be readily determined. In particular, they can be derived from the minimal subgroup relations provided in § E.1 and from the dimension table for fixed-point spaces of irreducible O(3) representations established in § E.2.

## A.3 MANIFOLDS AND BASIC DATA ASSUMPTION

Following the definition in Milnor (1990), $\mathbf { a } ~ C ^ { r }$ manifold M is mathematically defined as a subset of some linear space $\mathbb { R } ^ { d }$ . For every point x in this subset, there must exist a neighborhood W in $\mathbb { R } ^ { d }$ such that $W \cap M$ is $C ^ { r }$ -diffeomorphic to an open set in another linear space $\mathbb { R } ^ { l }$ . The integer l is known as the dimension of M. An alternative, intrinsic definition that does not depend on an embedding space can also be found in Hirsch (1976).

Machine learning often assumes that the data distribution is supported on a submanifold of the input space. The dimension of this manifold characterizes the directions in which the data can vary within the input space. However, the term manifold is often used loosely in this context and does not perfectly align with its mathematical counterpart (Goodfellow et al., 2016). First, the data manifold may have self-intersections, causing the local dimension of data variation to differ at various points. Second, data belonging to different classes or clusters may possess different structures and potentially different dimensions. Lastly, stochastic factors such as observational noise can prevent the data from forming a strict surface in the input space. We will not address this last factor, instead, we assume that the interference from noise is minimal and ignorable.

To address the first two issues, we can relax the manifold hypothesis by assuming that the data is supported on a finite union of submanifolds within the input space. Since a manifold with multiple connected components can itself be viewed as a disjoint union of connected manifolds, we can assume, without loss of generality, that each of these submanifolds is connected. For theoretical convenience, we further assume that these manifolds are bounded and closed. By the Weierstrass theorem, this is equivalent to assuming they are compact. We briefly outline the justification for these assumptions below:

• Finite Union of Manifolds: This assumption has been partially verified experimentally on computer vision datasets (Brown et al., 2022). Theoretically, it encompasses classic cases of self-intersection. A data manifold may arise as the image of a map. However, the image of an immersion or submersion of a manifold can have self-intersections. The image of such a map from a compact manifold is, however, a finite union of compact manifolds.

• Boundedness of Manifolds: This assumption stems from the natural assumption that the data distribution itself is bounded.

• Closedness of Manifolds: This assumption covers scenarios where the data is defined by a set of well-behaved constraints. For example, the set of points satisfying n independent, differentiable constraint equations in the input space forms a closed submanifold of codimension n.

In equivariant representation learning, the data possesses symmetries, meaning that data corresponding to the same physical object can be transformed by symmetry operations to represent different reference frames. These symmetry transformations typically form a group, and since the transformed data is still valid data, the data manifold must be closed under these group transformations, that is, the data manifold must be invariant under the group action. Considering the group action for any given input, we summarize the preceding discussions into the following assumptions about the data:

• Lie Group Assumption: The transformation group G is a compact Lie group.

• Manifold Assumption: The input space X is a linear space equipped with a linear action $\rho _ { X }$ of the group G. That is, there exists a map $\rho _ { X } : G \to G L ( X )$ such that $\rho _ { X } ( g _ { 1 } g _ { 2 } ) = \rho _ { X } ( g _ { 1 } ) \rho _ { X } ( g _ { 2 } )$ , and the identity element of the group is mapped to the identity transformation. The data manifold M is a finite union of compact, connected, smooth, and G-invariant submanifolds of X.

The smoothness assumption is added for theoretical convenience and the connectedness assumption does not impose additional restrictions, since each compact smooth submanifold admits only finitely many connected components.

## B PROOFS OF THE INFIMUM OF SYMMETRY (§ 3)

## B.1 PROOF OF THM. 3.1

Lemma B.1 (Azzi et al. (2023), Cor. 2.3). For a compact Lie group G and a representation $V ,$ , let $v \in V$ be any point. There exists a neighborhood U of v such that for any point $u \in U$ , we have $\left( G _ { u } \right) \leq \left( G _ { v } \right)$

Lemma B.2 (Azzi et al. (2023), Cor. 2.20). For a reductive group G and an affine algebraic variety $X ,$ let $x \in X$ be a point with a closed orbit $G ( x )$ . There exists a Zariski neighborhood U ofx such that for any point $u \in U$ , we have $( G _ { u } ) \leq ( \dot { G _ { x } } )$

We use the definition of the complexification from Prop. 3.3 of Azzi et al. (2023). For a real vector space $V ,$ , its complexification is ${ \dot { V } } ^ { \mathbb { C } } : = \mathbb { C } \otimes V$ . Considering an orthogonal action of G on $V ,$ , with g being the Lie algebra of $G ,$ the complexification of the group is $G ^ { \mathbb { C } } : = \{ g \exp ( i X ) \mid g \in G , X \in { \mathfrak { g } } \}$ By the above complexification, we obtain the following lemma.

Lemma B.3. For a compact Lie group $G ,$ consider the inclusion map $\iota : G \hookrightarrow G ^ { \mathbb { C } }$ into its complexification. Let H and K be closed subgroups of G. Then

$$
( H ) \leq ( K ) \quad \Longleftrightarrow \quad ( H ^ { \mathbb { C } } ) \leq ( K ^ { \mathbb { C } } ) ,\tag{12}
$$

where the two partial orders are taken with respect to conjugation in G and $G ^ { \mathbb { C } }$ , respectively.

Proof. The reverse implication follows from Prop. 4.9 of Azzi et al. (2026). For the forward direction, let $g _ { 0 } \in G$ be an element such that $g _ { 0 } H g _ { 0 } ^ { - 1 } \subseteq { \bf K }$ by $( H ) \leq ( K )$ . Let h and k be the Lie algebras of H and K, respectively. Any element of $H ^ { \mathbb { C } }$ can be expressed as $h _ { 0 } \exp ( i X _ { 0 } )$ for $h _ { 0 } \in H$ and $X _ { 0 } \in { \mathfrak { h } }$ . Its conjugation by $g _ { 0 }$ is

$$
g _ { 0 } \big ( h _ { 0 } \exp ( i X _ { 0 } ) \big ) g _ { 0 } ^ { - 1 } = \big ( g _ { 0 } h _ { 0 } g _ { 0 } ^ { - 1 } \big ) \exp ( i \mathrm { A d } _ { g _ { 0 } } X _ { 0 } ) .\tag{13}
$$

Since $g _ { 0 } H g _ { 0 } ^ { - 1 } \subseteq K$ , the term $g _ { 0 } h _ { 0 } g _ { 0 } ^ { - 1 }$ belongs to $K .$ . The expression thus belongs to $K ^ { \mathbb { C } }$ provided that $\operatorname { A d } _ { g _ { 0 } } X _ { 0 } \in { \mathfrak { k } } .$

This condition on the Lie algebra is obtained by differentiating the subgroup inclusion $g _ { 0 } H g _ { 0 } ^ { - 1 } \subseteq K$ at the identity element, which gives $\operatorname { A d } _ { g _ { 0 } } ( { \mathfrak { h } } ) \subseteq { \mathfrak { k } }$ . As $X _ { 0 } \in { \mathfrak { h } }$ , it follows immediately that $\mathrm { A d } _ { g _ { 0 } } X _ { 0 } \in$ k. Then,

$$
\iota ( g _ { 0 } ) H ^ { \mathbb { C } } \iota ( g _ { 0 } ) ^ { - 1 } \subseteq K ^ { \mathbb { C } } ,\tag{14}
$$

which complete the proof.

Proposition B.4. Let V be a representation of a compact Lie group G. For any closed subgroup H of G, the set of points of minimal orbit type in the fixed-point subspace $V ^ { H }$ is a dense and open subset of $\cdot \ : V ^ { H }$

Proof. The proof strategy is similar to that of Prop. 2.10 in Azzi et al. (2023). We first show that the set of points with a minimal orbit type is open, then use complexification to show it is Zariski-open, which implies density.

First, to prove openness, let $( K )$ be a minimal orbit type in $V ^ { H }$ and let $x \in V ^ { H }$ be a point with $( G _ { x } ) = { \bar { ( K ) } }$ . By Lem. B.1, there exists a neighborhood U of x such that for any $u \in U , ( \bar { G _ { u } } ) \leq ( K )$ Since $( K )$ is minimal in $\dot { V } ^ { H }$ , any point in the open set $U \cap V ^ { H }$ must have orbit type $( K )$ . Thus, openness is established.

Next, we consider the complexifications $G ^ { \mathbb { C } }$ and $V ^ { \mathbb { C } }$ . For a point $x \in V$ with orbit type $( K )$ , by Lem. 3.13 of Azzi et al. (2023), we have $( G ^ { \mathbb { C } } ) _ { x } = ( G _ { x } ) ^ { \mathbb { C } } = { \dot { K ^ { \mathbb { C } } } }$ . By Prop. 3.14 of Azzi et al. $( 2 0 2 3 )$ the orbit $G ^ { \mathbb { C } } ( x )$ is closed. Therefore, by Lem. B.2, there exists a Zariski neighborhood U of x in $V ^ { \mathbb { C } }$ such that any point in this neighborhood has an isotropy type less than or equal to $( K ^ { \mathbb { C } } )$ . The set $U ^ { \prime } = U \cap ( V ^ { \mathbb { C } } ) ^ { H ^ { \mathbb { C } } }$ is Zariski-open in $( V ^ { \mathbb { C } } ) ^ { H ^ { \mathbb { C } } }$ . By Cor. $\mathrm { A } . 7$ of Azzi et al. (2023), its real part $U ^ { \prime \prime } = U ^ { \prime } \cap V$ contains a non-empty real Zariski-open subset of $V ^ { H }$

We now show that for any $y \in U ^ { \prime \prime }$ , its orbit type is $( G _ { y } ) = ( K )$ . Since $y \in U$ , Lem. B.2 gives $( ( G ^ { \mathbb { C } } ) _ { y } ) \leq ( K ^ { \mathbb { C } } )$ . By Lem. 3.13 of Azzi et al. (2023), we have $( G ^ { \mathbb { C } } ) _ { y } = ( G _ { y } ) ^ { \mathbb { C } }$ . Therefore, the reverse implication of Lem. B.3 yields $( G _ { y } ) \leq ( K )$

Since $y \in V ^ { H }$ , the orbit type $\left( G _ { y } \right)$ occurs among the orbit types in $V ^ { H }$ . For Lem. B.3, by the minimality of $( K )$ , the preceding inequality must be an equality $( G _ { y } ) = ( K )$ . Since $U ^ { \prime \prime }$ contains a nonempty real Zariski-open subset of $V ^ { H }$ , the set of points of orbit type (K) is dense in $V ^ { H }$ Together with the openness proved above, this completes the proof.

Theorem 3.1 (Uniqueness of Minimal Type). Let X be a representation of a compact Lie group G. For any closed subgroup $H \subseteq G ,$ , a unique minimal orbit type exists among the points in the fixed-point subspace $\Breve { X ^ { H } }$ . In particular, $i f \in \mathsf { \Gamma } ( H ) \in { \mathcal { O } } _ { G } ( X ^ { H } )$ , then (H) is the minimal orbit type within that subspace.

Proof of Thm. 3.1. The density and openness of the set of points corresponding to any minimal orbit type is established by Prop. B.4. The uniqueness then follows from the fact that two distinct open dense sets must have a non-empty intersection.

## B.2 PROOF OF THM. 3.3

Theorem 3.3. A necessary condition for the existence of an isovariant map relative to Y from a G-set X to a G-set Y is that $( p _ { Y } ( H ) ) \in { \mathcal { O } } _ { G } ( Y )$ for every $( H ) \in { \mathcal { O } } _ { G } ( X )$ . When X and Y are representations, this is equivalent to the condition that $I _ { G } ( \dot { Y } , \dot { H } ) \stackrel { } { = } ( p _ { Y } \bar { ( H ) } ) ^ { \prime } f o r a l l \left( H \right) \in \mathcal { O } _ { G } ( X )$

Proof of Thm. 3.3. The necessity for general G-sets follows directly from the definition of relative isovariance. Now suppose that X and Y are representations. For any closed subgroup $H \subseteq G$

$$
( p _ { Y } ( H ) ) \leq I _ { G } ( Y , H ) ,\tag{15}
$$

since every isotropy subgroup occurring in $Y ^ { H }$ contains both H and ker $\rho _ { Y }$ . If $( p _ { Y } ( H ) ) \in O _ { G } ( Y )$ then this orbit type is realized in $Y ^ { H }$ , because $H \subseteq p _ { Y } ( H )$ . Hence, by the minimality of $I _ { G } ( Y , \overrightharpoon { H } )$ ,

$$
I _ { G } ( Y , H ) \leq ( p _ { Y } ( H ) ) .\tag{16}
$$

Therefore $I _ { G } ( Y , H ) = ( p _ { Y } ( H ) )$ . The converse is immediate since $I _ { G } ( Y , H )$ is itself an orbit type occurring in Y .

## C SYMMETRY INCREASE FOR $G = S O ( 3 )$ OR $O ( 3 )$ IN § 4

## C.1 GENERAL ORBIT TYPE CRITERION

To derive Prop. 4.2, we need more general results require criteria for identifying orbit types of general representation. Given the sufficiency of the Ihrig-Golubitsky Criterion (Prop. C.1), we demonstrate that for high-multiplicity representations, the conditions of the Michel Criterion imply the conditions of the Ihrig-Golubitsky Criterion. This, in turn, establishes the sufficiency of the Michel Criterion.

Let G be a compact Lie group and X be the input space. The normalizer of a subgroup $H \subseteq G$ is $N _ { G } ( H ) : = \{ \stackrel { \cdot } { g } \in G \ | \ \stackrel { \cdot } { g } H g ^ { - 1 } = H \}$ . The normalizer of H relative to a supergroup $\bar { H ^ { \prime } } \supset H$ is $N _ { G } ( H , \dot { H } ^ { \prime } ) : = \{ g \in \dot { G } \ \bar { | } \ H ^ { - } g H ^ { \prime } g ^ { - 1 } \}$ . Based on the above definitions, we recall the following general criterion.

Proposition C.1 (Ihrig-Golubitsky Criterion, Ihrig & Golubitsky (1984), Prop. 5.3). Let V be a faithful representation ofa group G. A sufficient conditionfor a closed subgroup H to be an isotropy subgroup in V is that for every orbit type $( H ^ { \prime } )$ in $V$ with $\left( \dot { H } ^ { \prime } \right) > \left( H \right)$ , thefollowing inequality holds:

$$
\dim V ^ { H ^ { \prime } } + \alpha _ { G } ( H , H ^ { \prime } ) : = \dim V ^ { H ^ { \prime } } + \dim N _ { G } ( H , H ^ { \prime } ) - \dim N _ { G } ( H ^ { \prime } ) < \dim V ^ { H } ,\tag{17}
$$

where $\alpha _ { G } ( H , H ^ { \prime } )$ is the Ihrig-Golubitsky correction term.

Remark. Ihrig & Golubitsky (1984) states that the condition is also necessary for $G = S O ( 3 ) , O ( 3 )$

Compared to Prop. C.1, the Linehan-Stedman Criterion (Linehan & Stedman, 2001) is more convenient due to its structure, which only requires checking adjacent subgroups. Since the associated theoretical results are not needed in our proofs, we do not elaborate on them here.

## C.2 PROOF OF PROP. 4.2

Proposition 4.2. For a high-multiplicity representation V, the necessary condition stated in Thm. 4.1 is also sufficient.

Proof of Prop. 4.2. We show that for representations where each irreducible component has multiplicity $r >$ dim G, the necessary condition from Michel’s Criterion (Thm. 4.1) becomes sufficient by proving it implies the condition in Prop. C.1. We may assume without loss of generality that the representation is faithful. For unfaithful representations, we proceed by factoring out the kernel from G and invoking Prop. C.1, as the procedure remains unchanged.

It follows from the condition that for any closed subgroup $H ^ { \prime }$ of G containing H, we have dim $V ^ { H ^ { \prime } } <$ dim $V ^ { H } .$ . The condition implies that for at least one irreducible component $V _ { i } ,$ we must have dim $V _ { i } ^ { H ^ { \prime } } < \dim V _ { i } ^ { H }$ . Since dimensions are integers, this is equivalent to dim $V _ { i } ^ { H ^ { \prime } } + 1 \leq \dim V _ { i } ^ { H }$ Given that the multiplicity $m ( V _ { i } , V ) > \dim G ,$ , it follows that

$$
m ( V _ { i } , V ) \dim V _ { i } ^ { H ^ { \prime } } + \dim G < m ( V _ { i } , V ) \dim V _ { i } ^ { H } .\tag{18}
$$

Summing this strict inequality for one such component with the non-strict inequalities for all other components yields

$$
\dim V ^ { H ^ { \prime } } + \dim G < \dim V ^ { H } .\tag{19}
$$

The term from Prop. C.1 is bounded by

$$
0 \leq \alpha _ { G } ( H , H ^ { \prime } ) = \dim N _ { G } ( H , H ^ { \prime } ) - \dim N _ { G } ( H ^ { \prime } ) \leq \dim G .\tag{20}
$$

The sufficiency of Michel’s condition follows from combining these two results:

$$
\dim V ^ { H ^ { \prime } } < \dim V ^ { H } \quad \Longrightarrow \quad \dim V ^ { H ^ { \prime } } + \dim G < \dim V ^ { H }\tag{21}
$$

$$
\begin{array} { r l } { \Longrightarrow } & { { } \dim V ^ { H ^ { \prime } } + \dim N _ { G } ( H , H ^ { \prime } ) - \dim N _ { G } ( H ^ { \prime } ) < \dim V ^ { H } , } \end{array}\tag{22}
$$

where the final implication uses the upper bound from Eq. (20). This shows that Michel’s condition implies the sufficient condition from Prop. C.1. □

## C.3 DETAILED CALCULATION IN EX. 4.3

We now conduct the orbit-type test and symmetry infimum calculation for the geometric symmetry $D _ { k h } \left( k > 2 \right)$ of the k-fold from Ex. 2.2. Consider $D _ { k h }$ as a closed subgroups of $O ( 3 )$ , the calculation is performed in the representation space $Y = V _ { l = l _ { 0 } } ^ { \oplus r }$ for $r > 3$ and $l _ { 0 } > 0 .$ . We will use the precomputed subgroup relations from Table $^ { 6 , }$ the dimensions of the fixed-point spaces for $D _ { k h }$ and $O _ { h }$ in $V _ { l = l _ { 0 } }$ from Table 8, and the orbit type test results for $( D _ { \infty h } )$ and $( \bar { O _ { h } } )$ in $\dot { V } _ { l = l _ { 0 } } ^ { \oplus r }$

First, we perform the orbit type test. Consider the case where $k \neq 4 .$ . In this situation, the adjacent supergroups of $D _ { k h }$ are of the form $D _ { p k , h }$ , where p is a prime number. We classify the discussion into 12 cases based on the parity of k and $l _ { 0 }$ , and the ranges $l _ { 0 } < k , k \le l _ { 0 } < \dot { 2 k }$ , and $l _ { 0 } \geq 2 k$ and calculate the dimensions of the fixed-point spaces of $V _ { l = l _ { 0 } }$ for $D _ { k h }$ and $D _ { p k , h }$ . The calculated dimensions are shown in Table 2. For the case $k = 4$ , we must additionally consider the supergroup $O _ { h }$ . Here, dim $V _ { l = l _ { 0 } } ^ { D _ { 4 h } } =$ dim $V _ { l = l _ { 0 } } ^ { O _ { h } }$ only when $l _ { 0 }$ is odd, which aligns with the results from Table 2. In all other cases, $O _ { h }$ does not affect the orbit type test of $( D _ { k h } )$ . Thus, we find that $( D _ { k h } )$ is an orbit type when $l _ { 0 } \geq k$ and when $l _ { 0 }$ and k have the same parity.

Table 2: Dimension of fixed-point space for supergroups of $D _ { k h } \left( k > 2 \right)$ in $V _ { l = l _ { 0 } } ^ { \oplus r } \left( r > 3 \right)$ , organized by the parity of k and $l _ { 0 }$
<table><tr><td> $l _ { 0 }$   $l _ { 0 }$ </td><td> $l _ { 0 } < k$  is even</td><td>is odd</td><td> $k \leq l _ { 0 } < 2 k$   $l _ { 0 }$  is even</td><td> $l _ { 0 }$  is odd</td><td> $l _ { 0 } \geq 2 k$   $l _ { 0 }$  is even</td><td> $l _ { 0 }$  is odd</td></tr><tr><td colspan="7">k is even</td></tr><tr><td> $D _ { k h }$ </td><td>1</td><td>0</td><td>2</td><td>0</td><td> $d$ </td><td>0</td></tr><tr><td> $D _ { p k h }$ </td><td>1</td><td>0</td><td>1</td><td>0</td><td> $< d$ </td><td>0</td></tr><tr><td colspan="7">k is odd</td></tr><tr><td> $D _ { k h }$ </td><td>1</td><td>0</td><td>1</td><td>1</td><td>d</td><td>d</td></tr><tr><td> $D _ { 2 k h }$ </td><td>1</td><td>0</td><td>1</td><td>0</td><td>d</td><td>0</td></tr><tr><td> $D _ { p ^ { * } k h }$ </td><td>1</td><td>0</td><td>1</td><td>0</td><td> $< d$ </td><td> $< d$ </td></tr></table>

Next, we proceed with the symmetry infimum calculation. Again, we first consider the case where $k \neq 4$ . In this situation, the supergroups of $D _ { k h }$ are $D _ { p k , h } , D _ { \infty h }$ , and $O ( 3 )$ , where $O ( 3 )$ is always an orbit type. Using the orbit type test results just calculated and those pre-computed, we analyze the cases based on the parity of k and $l _ { 0 }$ as before, with the results summarized in Table 3. For $k = 4 .$ the additional supergroup $O _ { h }$ does not affect the final result. This is because the supergroup $O _ { h }$ is never the minimal isotropy supergroup, as it only becomes an orbit type when $( D _ { 4 h } )$ already is. This leads us to the symmetry infimums shown in Table 1.

Table 3: Isotropy conditions for supergroups of $D _ { k h } \left( k > 2 \right) \mathrm { i n } V _ { l = l _ { 0 } } ^ { \oplus r } \left( r > 3 \right)$ , organized by the parity of k and $l _ { 0 } . \mathrm { ~ A ~ } \checkmark$ indicates the condition is satisfied, and ✗ that it is not. In the subgroup notation, p denotes a prime number and $p ^ { * }$ denotes an odd prime number.
<table><tr><td colspan="3"> $l _ { 0 } < k$   $l _ { 0 }$  is even  $l _ { 0 }$  is odd</td><td colspan="3"> $k \leq l _ { 0 } < 2 k$   $l _ { 0 }$  is even  $l _ { 0 }$  is odd</td><td colspan="3"> $l _ { 0 } \geq 2 k$  is even  $l _ { 0 }$  is odd</td></tr><tr><td colspan="3"></td><td colspan="3">k is even</td><td colspan="3"> $l _ { 0 }$ </td></tr><tr><td> $D _ { k h }$ </td><td colspan="3">x X</td><td></td><td>x</td><td></td><td>√</td><td>x</td></tr><tr><td> $D _ { p k h }$ </td><td colspan="3">X X</td><td>√</td><td>x</td><td></td><td></td><td>X</td></tr><tr><td> $D _ { \infty h }$ </td><td colspan="3">√</td><td>- –</td><td>X</td><td></td><td>- -</td><td>X</td></tr><tr><td></td><td colspan="3">X</td><td>k is odd</td><td></td><td></td><td></td><td></td></tr><tr><td colspan="3"> $D _ { k h }$  x</td><td colspan="3"></td><td></td><td>x</td><td>√</td></tr><tr><td> $D _ { 2 k h }$ </td><td colspan="3">X</td><td>x X</td><td>x X</td><td>√</td><td>√</td><td>-</td></tr><tr><td> $D _ { p ^ { * } k h }$ </td><td colspan="3">x</td><td>X</td><td>1 1</td><td>-</td><td></td><td>1</td></tr><tr><td> $D _ { \infty h }$ </td><td colspan="3">√</td><td>x X √</td><td>-</td><td>-</td><td></td><td>一</td></tr></table>

## C.4 CALCULATION OF SYMMETRY INFIMUM

We calculate the orbit types for the SO(3) representation $V _ { l = l _ { 0 } } ^ { \oplus r }$ and the $O ( 3 )$ representation $V _ { l = l _ { 0 } ^ { \pm } } ^ { \oplus r }$ As the calculation procedure is highly similar to that in Ex. $4 . 3 ,$ , we omit the detailed steps here. For the case of $r = 1$ , the results can be found in Table B.1 and Table B.2 of Linehan & Stedman (2001).

The orbit types are calculated using the procedure in Algo. 1. This calculation requires the dimensions of the fixed-point spaces for the closed subgroups of $S O ( 3 )$ or O(3) in the representations $V _ { l = l _ { 0 } }$ and $V _ { l = l _ { 0 } ^ { \pm } }$ , respectively, which are provided in § E.2.

According to Prop. 6.2 from Ihrig & Golubitsky (1984), the Ihrig-Golubitsky correction term $\alpha ( H , H ^ { \prime } ) ~ = ~ 0$ for all subgroups except $H ~ = ~ C _ { k }$ in $S O ( 3 )$ , and for all subgroups except $H = C _ { k } , S _ { 2 k } , C _ { k h }$ in $O ( 3 )$ . Therefore, for all subgroups other than $H = C _ { k } , S _ { 2 k } , \bar { C _ { k h } }$ , our results are identical to those calculated by Linehan & Stedman (2001) for $r = 1$ . We present only the results of whether $( C _ { k } ) \in \mathscr { O } _ { S O ( 3 ) } ( V _ { l = l _ { 0 } } ^ { \oplus r } )$ on Table 4 and whether $( C _ { k } ) , ( S _ { 2 k } ) , ( C _ { k h } ^ { \bullet } ) \in \mathcal { O } _ { O ( 3 ) } \mathbf { \bar { ( } } V _ { l = l _ { 0 } ^ { - } } ^ { \oplus r } )$ on Table 5.

The reason for omitting the discussion of $V _ { l = l _ { 0 } ^ { + } } ^ { \oplus r }$ is consistent with the explanation in the header of Table B.2 in Linehan & Stedman (2001). This is because the corresponding conclusions can be found in the orbit type table for the SO(3) representation $V _ { l = l _ { 0 } } ^ { \oplus r }$ via the mappings $D _ { \infty } \to D _ { \infty h }$ $C _ { \infty }  C _ { \infty h } , I  \stackrel {  } { I _ { h } } , O  O _ { h } , T  T _ { h }$ , as well as $D _ { k } \to \tilde { D _ { k h } }$ and $C _ { k } \to C _ { k h }$ for even $k ,$ , and $D _ { k } \to D _ { k d }$ and $C _ { k } \to S _ { 2 k }$ for odd k.

Table 4: Modified isotropy subgroups for the multiple irreducible representation $V _ { l = l _ { 0 } } ^ { \oplus r } , r > 3$ of SO(3) obtained via the Michel criterion.
<table><tr><td></td><td>H Condition of  $l _ { 0 }$ </td></tr><tr><td> $C _ { k }$ </td><td> $l _ { 0 } \geq k$ </td></tr></table>

Table 5: Modified isotropy subgroups for the multiple irreducible representation $V _ { l = l _ { 0 } ^ { - } } ^ { \oplus r } , r > 3$ of O(3) with nontrivial reflection, obtained via the Michel criterion.
<table><tr><td>H</td><td>Condition of  $l _ { 0 }$ </td></tr><tr><td> $C _ { k }$ </td><td> $l _ { 0 } \geq k$ </td></tr><tr><td> $S _ { 2 k }$  (k is even)</td><td> $l _ { 0 } \geq k$ </td></tr><tr><td> $C _ { k h }$  (k is odd)</td><td> $l _ { 0 } \geq k$ </td></tr></table>

The complete set of orbit type calculation results can also be found in the tables for the symmetry infimum. This is because (H) is an orbit type of a representation V if and only if the symmetry infimum of H in V is (H) itself. These non-degenerate cases are highlighted in green in the tables.

Using the calculated orbit types for the $S O ( 3 )$ representation $V _ { l = l _ { 0 } } ^ { \oplus r }$ and the O(3) representation $V _ { l = l _ { 0 } ^ { \pm } } ^ { \oplus r }$ , we can now compute the symmetry infimum for all subgroups of $S O ( 3 )$ or $O ( 3 )$ in these representations. When calculating the symmetry infimum for all subgroups, the algorithm differs slightly from that in Ex. 4.3. We reduce the need to enumerate all supergroups by leveraging the symmetry infimums of adjacent supergroups. An improved version of Algo. 2 is detailed in Algo. 3.

To reduce the computational load, this algorithm employs a top-down calculation strategy. First, we compute the results for the infinite groups, followed by the polyhedral groups. Finally, we calculate the results for the axial groups in the order $D _ { k h } , D _ { k d } , \bar { D } _ { k } , \bar { C _ { k v } } , \bar { C } _ { k h } , S _ { 2 k } , \bar { C _ { k } }$ . Due to the exceptional subgroup relations shown in Table 6, special consideration for additional supergroups is required for certain cases where $k \in \{ 1 , 2 , 3 , 4 , 5 \}$ . These cases are handled last.

The results for $S O ( 3 )$ are presented in $\ S \operatorname { E } . 3 ,$ and the results for $O ( 3 )$ are in § E.4. Here, we provide a general classification for the observed symmetry increases. For a closed subgroup H, an increase to the full group $O ( 3 )$ or $S O ( 3 )$ is termed full degeneration and marked in red in the tables. An increase to a supergroup of higher dimension than H is termed continuous degeneration and marked in blue . An increase to a supergroup of the same dimension as H is termed discrete degeneration and marked in yellow or light green . No symmetry increase is no degeneration and marked in green

Algorithm 3: Symmetry Infimum Calculation With Precomputed Results   
Data: A symmetry group $G ;$   
a closed subgroup $H \subset G ;$   
a Rep. V of $G ;$   
a map M of previously computed symmetry infs.   
Result: Symmetry inf. $I _ { G } ( \dot { V } , H )$   
1 i $\mathbf { f } \mathrm { i } \mathrm { s } \_ \mathrm { i n } \dot { ( } ( H ) , \dot { \mathcal { O } } _ { G } ( V ) )$ then   
2 return (H);   
3 end   
4 Let $S _ { 0 } = \{ H _ { i } \} \subset G$ to be all adjacent closed supergroups of H in $G ;$   
5 $\mathcal { O }  \mathcal { O } ;$   
6 for $H _ { i }$ in $S _ { 0 }$ do   
7 if $\mathrm { i } \mathsf { s \_ i n } ( ( H _ { i } ) , { \mathcal { O } } _ { G } ( V ) )$ then   
8 Add $( \dot { H } _ { i } )$ to the set ;   
9 end   
10 else if $H _ { i }$ is a key in the map M then   
11 Add $M [ H _ { i } ] \stackrel { \cdot } { = } I _ { G } ( V , \dot { H _ { i } } )$ to the set $\mathcal { O } ;$   
12 end   
13 else   
14 Let $S _ { i } = \{ K _ { j } \}$ to be all adjacent closed supergroups of $H _ { i }$ in $G ;$   
15 for $K _ { j }$ in $\mathbf { \bar { \boldsymbol { S } } } _ { i }$ do   
16 if $\operatorname { \ ' } ( K _ { j } ) \in { \mathcal { O } } _ { G } ( V )$ then   
17 Add $( K _ { j } )$ to the set $\mathcal { O } ;$   
18 end   
19 end   
20 end   
21 end   
22 return min $( { \mathcal { O } } ) ;$

For the subgroup $D _ { k h } \subset O ( 3 )$ , the classification of degeneration behaviors given after Ex. 4.3 is a special case of this general framework. Specifically, full degeneration is identical to the definition here, axial degeneration corresponds to continuous degeneration, and half degeneration correspond to discrete degeneration.

We note that for the representation $Y = V _ { l = l _ { 0 } ^ { + } } ^ { \oplus r }$ , the action of G has a non-trivial kernel, and thus symmetry increase is inevitable. Let $\pi _ { Y }$ be the natural projection to the quotient $G / \mathrm { k e r } \rho _ { Y }$ In this context, the light green marks an increase to $\pi _ { Y } ^ { - 1 } ( \pi _ { Y } ( H ) )$ ), which, as explained in $\ S \ 3 . 2 .$ , is the lowest possible symmetry that H can be reduced to when a non-trivial kernel is present. This is therefore a predictable behavior. The yellow marks other exceptional cases within discrete degeneration.

## C.5 COMPOSITION PROPERTY OF HIGH-MULTIPLICITY REPRESENTATION

In the tables of § E.3 and § E.4, there exists a special class of subgroups H for which any non-trivial symmetry increase always results in an infimum $\left( H _ { 0 } \right)$ where $H _ { 0 }$ is an adjacent supergroup. We call $\left( H _ { 0 } \right)$ the bottleneck of H, denoted by $B _ { G } ( H )$ , and we say that a subgroup H that possesses a bottleneck satisfies the bottleneck condition. These groups that act as bottlenecks satisfy some elegant properties. To demonstrate this, we first prove a structure theorem for the symmetry infimum of high-multiplicity representations.

Proposition C.2. For a high-multiplicity representation $V ,$ the lowest orbit type in thefixed-point subspace $V ^ { H } i s \left( G _ { V ^ { H } } \right)$ , where $\begin{array} { r } { G _ { V ^ { H } } = \bigcap _ { x \in V ^ { H } } G _ { x } } \end{array}$ is the largest subgroup that leaves $V ^ { \check { H } }$ invariant. This shows that $I _ { G } ( V , H ) = \left( G _ { V ^ { H } } \right)$

Proof. Since any element in $V ^ { H }$ has at least the symmetry $G _ { V ^ { H } }$ , we only need to prove that $G _ { V ^ { H } }$ is an isotropy subgroup. Assume, for the sake of contradiction, that $G _ { V ^ { H } }$ is not an isotropy subgroup. By the sufficiency of the Michel Criterion, there exists a supergroup $K$ such that $G _ { V ^ { H } } \subsetneq K$ and

$$
\dim V ^ { H } = \dim V ^ { G _ { V ^ { H } } } = \dim V ^ { K } \implies V ^ { H } = V ^ { G _ { V ^ { H } } } = V ^ { K } .\tag{23}
$$

This means that K leaves all elements of $V ^ { H }$ fixed. From this, we derive a contradiction:

$$
K \subset \bigcap _ { x \in V ^ { H } } G _ { x } = G _ { V ^ { H } } .\tag{24}
$$

Remark. This result implies that for a high-multiplicity representation, the symmetry increase from (H) to $I _ { G } ( V , H )$ does not alter the dimension of the fixed-point space. Therefore, for high-multiplicity representations, the dimension of the fixed-point subspace corresponding to the input symmetry group equals that corresponding to the symmetry infimum. Since the fixed-point subspace dimensions for certain closed subgroups exhibit distinct regularities, the behavior of symmetry increase toward these subgroups serves as an indicator of the underlying subspace dimension. For instance, in the cases of $\bar { S } { \cal O } ( 3 )$ or $O ( 3 )$ , for non-trivial representations, full degeneration corresponds to a 0-dimensional fixed-point subspace, whereas for finite input subgroups, continuous degeneration corresponds to a 1-dimensional subspace. However, we must emphasize that when a quantitative assessment of the equivariant feature’s expressive capacity is required, relying solely on the symmetry infimum is insufficient, as it yields only coarse, qualitative insights. In such cases, one should directly compute the fixed-point subspace dimension. For calculations related to $S O ( 3 )$ or $O ( 3 )$ , see $\ S \operatorname { E } . 2 \AA$

We can prove that these groups acting as bottlenecks satisfy the property that for any irreducible representation $V _ { 0 } ,$ if dim $V _ { 0 } ^ { H } >$ dim $V _ { 0 } ^ { \breve { H } _ { 0 } }$ , then dim $V _ { 0 } ^ { H } >$ dim $V _ { 0 } ^ { H ^ { \prime } }$ holds for all adjacent supergroups $H ^ { \prime }$ of $H$ . This is because if we assume there exists an $H ^ { \prime }$ such that equality holds, then by the Michel Criterion, $( H )$ must undergo a non-trivial symmetry increase to $\bar { ( G _ { V H } ) }$ in the high-multiplicity representation $V _ { 0 } ^ { \oplus r }$ . Since these increases always have the bottleneck $\left( H _ { 0 } \right)$ as their infimum, the inequalities

$$
\dim V _ { 0 } ^ { H } \geq \dim V _ { 0 } ^ { H _ { 0 } } \geq \dim V _ { 0 } ^ { G _ { V ^ { H } } }\tag{25}
$$

must in fact collapse to dim $V _ { 0 } ^ { H } = \dim V _ { 0 } ^ { H _ { 0 } }$ , which leads to a contradiction.

For any high-multiplicity representation V that satisfies dim $V ^ { H } > \dim V ^ { H _ { 0 } }$ , there must exist a component corresponding to an irreducible representation $V _ { 0 }$ in V such that dim $V _ { 0 } ^ { H } >$ dim $V _ { 0 } ^ { H _ { 0 } }$ It follows that dim $V _ { 0 } ^ { H } >$ dim $V _ { 0 } ^ { H ^ { \prime } }$ for all adjacent supergroups $H ^ { \prime }$ of $H$ , which in turn shows that dim $V ^ { H } >$ dim $V ^ { H ^ { \prime } }$ holds for all such $H ^ { \prime }$ . Therefore, for high-multiplicity representations, $H _ { 0 }$ controls the dimension gap of the fixed-point spaces between $\check { H }$ and its other adjacent supergroups.

This property allows us to prove a theorem regarding the direct sum of high-multiplicity representations, which in turn establishes a property for the orbit types of such direct sums.

Theorem C.3. Let a subgroup H satisfy the bottleneck condition. For two high-multiplicity representations $V _ { 1 }$ and $V _ { 2 }$ , we have $( \mathbf { \bar { { H } } } ) \in { \mathcal { O } } _ { G } ( V _ { 1 } \oplus V _ { 2 } )$ ifand only $i f ( H ) \in \mathcal { O } _ { G } ( \bar { V _ { 1 } } ) o r ( \bar { H } ) \in \mathcal { O } _ { G } ( V _ { 2 } )$

Proof. We have already shown that for any high-multiplicity representation V , if dim $V ^ { H } ~ >$ dim $V ^ { H _ { 0 } }$ , then dim $V ^ { H } >$ dim $V ^ { H ^ { \prime } }$ holds for all adjacent supergroups of H. Therefore, assuming $( H ) \in { \mathcal { O } } _ { G } ( V _ { 1 } \oplus V _ { 2 } )$ , the Michel Criterion gives us

$$
\dim V _ { 1 } ^ { H } + \dim V _ { 2 } ^ { H } > \dim V _ { 1 } ^ { H ^ { \prime } } + \dim V _ { 2 } ^ { H ^ { \prime } } .\tag{26}
$$

By taking $H ^ { \prime } = H _ { 0 } = B _ { G } ( H )$ , we see that at least one of the following two conditions must be true:

$$
\dim V _ { i } ^ { H } > \dim V _ { i } ^ { H _ { 0 } } , \quad i = 1 , 2 .\tag{27}
$$

This implies that for all adjacent supergroups of H, at least one of the following two conditions must hold:

$$
\dim V _ { i } ^ { H } > \dim V _ { i } ^ { H ^ { \prime } } , \quad i = 1 , 2 .\tag{28}
$$

Therefore, (H) is an orbit type of either $V _ { 1 }$ or $V _ { 2 }$ .

The above theorem shows that when we construct a high-multiplicity representation for which (H) is an orbit type using high-multiplicity components, the only way is to find a high-multiplicity component that already contains $( { \bar { H } } )$ as an orbit type.

For $S O ( 3 )$ , all closed subgroups satisfy the bottleneck condition. Consequently, for high-multiplicity representations, we have $\bar { \mathcal { O } } _ { G } ( \bar { V } _ { 1 } ) \cup \bar { \mathcal { O } _ { G } } ( V _ { 2 } ) = \mathcal { O } _ { G } ( V _ { 1 } \oplus V _ { 2 } )$ . This means that $I _ { G } ( V _ { 1 } \oplus V _ { 2 } , H )$ will be the minimum of $I _ { G } ( V _ { 1 } , H )$ and $I _ { G } ( V _ { 2 } , H )$ . Furthermore, the set of representations $\{ V _ { l = l _ { 0 } } ^ { \oplus r } \}$ is sufficient to generate all closed subgroups as orbit types, because for any closed subgroup, there always exists a high-multiplicity representation for which it is an orbit type.

For $O ( 3 )$ , the bottleneck condition does not necessarily hold. For example, for $H = C _ { \infty }$ , a symmetry increase can result in either $D _ { \infty }$ or $C _ { \infty v }$ , which shows that no bottleneck group exists. The fact that $C _ { \infty }$ never appears as an orbit type in any representation $V _ { l = l _ { 0 } ^ { \pm } } ^ { \oplus r }$ demonstrates this point precisely.

This introduces a subtle issue when we apply the guideline on § 4.2: for certain orbit types, a representation exhibiting the target orbit type cannot be constructed simply by including the component $V _ { l = l _ { 0 } ^ { \pm } } ^ { \oplus r }$ associated with it. Fortunately, $C _ { \infty }$ is the sole instance of this phenomenon encountered when $G = O ( 3 )$ . Regarding the construction of a $O ( 3 )$ representation containing $C _ { \infty }$ , following Prop. 4.2, it suffices to simultaneously select components $V _ { l = l _ { 0 } ^ { - } } ^ { \oplus \bar { r } }$ with both odd and even degrees $l _ { 0 } > 0$

## D PROOFS OF DENSITY OF (ALMOST) ISOVARIANT MAPS (§ 5)

## D.1 COUNTEREXAMPLE IN § 5.1

Lemma D.1 (Borsuk-Ulam Theorem, Guillemin & Pollack (1974), Chap. 2, Sec. 6). For any continuous oddfunction $g : S ^ { n } \to { \mathbb { R } } ^ { n }$ , there exists a point $x \in S ^ { n }$ such that $g ( x ) = 0$

Lemma D.2 (Weak Borsuk-Ulam Theorem, Nagasaki (2003), Thm. A). Let M and N be G-spheres in a representation spacefor a compact group G. Ifthere exists an equivariant mapfrom M to $N ,$ then thefollowing dimensional inequality holds:

$$
\varphi _ { G } ( \dim { \cal M } - \dim { \cal M } ^ { G } ) \leq \dim { \cal N } - \dim { \cal N } ^ { G } ,\tag{29}
$$

where $\varphi _ { G } : \mathbb { N }  \mathbb { N }$ is a non-decreasing function that diverges to infinity.

Counterexample D.3. For a compact Lie group $G ,$ consider a representation $\tilde { Y }$ with no trivial component, i.e. $\tilde { Y } ^ { G } = \{ 0 \}$ . Let $Y = { \tilde { Y } } ^ { \oplus r }$ and $X = Y ^ { \oplus ( n _ { 0 } + 1 ) }$ for some $r , n _ { 0 } \ >$ dim G. By Prop. 4.2, we have $\mathcal { O } _ { G } ( X ) \stackrel { \cdot } { = } \mathcal { O } _ { G } ( Y )$ , yet no isovariant map existsfrom the unit sphere in X to Y for a sufficiently large integer n .

In particular,for $G = \mathbb { Z } _ { 2 } , i f { \tilde { Y } }$ is the non-trivial irreducible representation, thenfor any $X = \tilde { Y } ^ { \oplus r _ { 1 } }$ and $Y = \tilde { Y } ^ { \oplus r _ { 2 } }$ with $r _ { 1 } > r _ { 2 } ,$ , no isovariant map existsfrom the unit sphere ofX to Y.

Proof. For any compact Lie group $G ,$ consider an equivariant map $f$ from a G-sphere M in a vector space X to a G-representation $Y$ . We assume that the multiplicities of the trivial representation components in both $X$ and Y are zero, i.e., $X ^ { G } = \{ 0 \}$ and $Y ^ { \ ' G } = \{ 0 \}$ . Therefore, only the origin in X and Y has the orbit type (G), and the sphere M does not contain any point of orbit type (G). Consequently, if $f$ is an isovariant map, it must have no zeros.

From such a zero-free map $f ,$ we can define a map $\tilde { f }$ to the G-sphere N in $Y \colon$

$$
\tilde { f } : M \to N , \quad \tilde { f } ( x ) = f ( x ) / \| f ( x ) \| _ { 2 } .\tag{30}
$$

Since scaling in a vector space does not change the orbit type, $\tilde { f }$ is also an isovariant map. Therefore, by Lem. D.2, we obtain the dimensional relation:

$$
\varphi _ { G } ( \dim M - \dim M ^ { G } ) = \varphi _ { G } ( \dim X - 1 ) \leq \dim N - \dim N ^ { G } = \dim Y - 1 .\tag{31}
$$

Since $\varphi _ { G }$ is a non-decreasing function that diverges to infinity, there must exist an integer $n _ { 0 } >$ dimG such that $\varphi _ { G } ( n _ { 0 } ) > \dim Y - 1$ . This implies that

$$
\varphi _ { G } ( \dim ( Y ^ { \oplus ( n _ { 0 } + 1 ) } ) - 1 ) > \dim Y - 1 .\tag{32}
$$

This shows that no isovariant map can exist from the unit G-sphere M in the space $X = Y ^ { \oplus ( n _ { 0 } + 1 ) }$ to the space $Y .$ To complete the counterexample, let $Y = { \tilde { Y } } ^ { \oplus r }$ , where $r > \dim G$ . By Prop. 4.2, we have ${ \dot { \mathcal { O } _ { G } } } ( Y ) = { \mathcal { O } } _ { G } ( X ) { \dot { \mathbf { \mu } } }$ , yet no isovariant map exists between them.

For the special case of $G = \mathbb { Z } _ { 2 }$ , let $\tilde { Y }$ be the non-trivial irreducible representation. Let $X = \tilde { Y } ^ { \oplus r _ { 1 } }$ and $\boldsymbol { Y } = \tilde { Y } ^ { \oplus r _ { 2 } }$ with $r _ { 1 } > r _ { 2 }$ . A map $f$ from the unit sphere in X to Y is equivariant if and only if it is an odd function. Since the representations have no trivial components, an isovariant map is equivalent to an odd function that has no zeros. However, by Lem. D.1, any odd map from a sphere in a higher-dimensional space to a lower-dimensional space must have a zero. Thus, no such isovariant map exists. □

## D.2 PROOF OF THM. 5.1

TFN is an ENN based on tensor products of hidden features. The TFN architecture is composed of a feature lifting map followed by an equivariant pooling stage. The feature lifting map contains equivariant convolutional layers that update node features by aggregating information from neighbors. These convolutions employ filters built from learnable radial functions parameterized by a Multi-Layer Perceptron (MLP) and real spherical harmonics $Y _ { l m }$

Mathematically, the parameterized map $f _ { \mathrm { T F N } } \in \mathcal { F } _ { \mathrm { T F N } }$ is defined as a sum over feature channels. Each term in the sum is a composition $f _ { \mathrm { p o o l } } \circ f _ { \mathrm { f e a t } }$ , consisting of a feature lifting map followed by equivariant linear pooling. Let $Z$ represent the hidden features associated with each node. The feature lifting map $f _ { \mathrm { f e a t } }$ takes the input point cloud X to node features $Z \otimes \mathbb { R } ^ { n }$ . This map is constructed as a composition of layers:

$$
f _ { \mathrm { f e a t } } = \pi _ { Z \otimes \mathbb { R } ^ { n } } \circ \left( f ^ { ( L ) } , \mathrm { i d } \right) \circ \cdot \cdot \cdot \circ \left( f ^ { ( 1 ) } , \mathrm { i d } \right) \circ \mathrm { e x t } \circ C ,\tag{33}
$$

where $C$ is a centering operation, ext is a constant extension map, and each layer $f ^ { ( k ) }$ updates the node features $\mathbf { \chi } _ { v } ( k - 1 ) \mathbf { \chi } _ { \mathrm { t o } } ^ { \mathbf { - } } \hat { v ( k ) }$ according to the rule:

$$
\begin{array} { r } { v _ { i l _ { 3 } m _ { 3 } } ^ { ( k ) } = \theta v _ { i l _ { 3 } m _ { 3 } } ^ { ( k - 1 ) } + \sum _ { j \neq i } \sum _ { l _ { 1 } , m _ { 1 } , l _ { 2 } , m _ { 2 } } C _ { ( l _ { 2 } , m _ { 2 } ) , ( l _ { 1 } , m _ { 1 } ) } ^ { ( l _ { 3 } , m _ { 3 } ) } F _ { m _ { 2 } } ^ { ( l _ { 2 } ) } ( x _ { i } - x _ { j } ) v _ { j l _ { 1 } m _ { 1 } } ^ { ( k - 1 ) } . } \end{array}\tag{34}
$$

Here, $v _ { i l _ { 3 } m _ { 3 } }$ corresponds to the feature of type $l _ { 3 }$ at node $i ,$ and $C _ { ( l _ { 2 } , m _ { 2 } ) , ( l _ { 1 } , m _ { 1 } ) } ^ { ( l _ { 3 } , m _ { 3 } ) }$ are the Clebsch-Gordan coefficients. For $G = O ( 3 ) \times S _ { n }$ , the parities of $\cdot _ { l _ { 1 } }$ and $l _ { 3 }$ must also be considered.

For each degree l and order $m ,$ , let ${ \mathcal { V } } _ { l m }$ denote the corresponding solid spherical harmonic. It satisfies

$$
\mathcal { Y } _ { l m } ( \boldsymbol { x } ) = \| \boldsymbol { x } \| _ { 2 } ^ { l } Y _ { l m } \left( \boldsymbol { x } / \| \boldsymbol { x } \| _ { 2 } \right) , \qquad \boldsymbol { x } \neq 0 ,\tag{35}
$$

and extends to $x = 0$ as a homogeneous harmonic polynomial of degree l. We define the filter function by

$$
F _ { m } ^ { ( l ) } ( x ) = h _ { l } ( \| x \| _ { 2 } ^ { 2 } ) \mathcal { V } _ { l m } ( x ) .\tag{36}
$$

This parameterization is a mild modification of the commonly used implementation

$$
\widetilde { F } _ { m } ^ { ( l ) } ( \boldsymbol { x } ) = \widetilde { h } _ { l } ( \| \boldsymbol { x } \| _ { 2 } ) Y _ { l m } \left( \boldsymbol { x } / \| \boldsymbol { x } \| _ { 2 } \right) .\tag{37}
$$

Indeed, away from the origin, the factor $\| \boldsymbol { x } \| _ { 2 } ^ { l }$ can be absorbed into the radial function by setting $\widetilde { h } _ { l } ( r ) = r ^ { l } h _ { l } ( r ^ { 2 } )$ , which can in turn be approximated by a radial MLP. We adopt the solid-harmonic form because it yields a globally well-defined filter, including at $x = 0$ . Moreover, this adaptation does not affect the universality result of Dym & Maron (2021), since the polynomial filters used in their construction admit an equivalent representation in the solid-harmonic form above.

When the radial function $h _ { l } : \mathbb { R } _ { > 0 } $ R is parameterized by a Multi-Layer Perceptron (MLP), we denote by $\mathcal { F } _ { \mathrm { T F N } }$ the resulting family of TFN filters under MLP-parameterized radial functions. In contrast, when $h _ { l }$ is chosen as polynomial parameterization, we denote the corresponding parameteri zation by $\mathcal { F } _ { \mathrm { T F N } } ^ { \mathrm { p o l y } }$ . Notice that we use $\mathcal { F } _ { \mathrm { T F N } }$ and $\mathcal { F } _ { \mathrm { T F N } } ^ { \mathrm { p o l y } }$ denotes the union of these architectures over the number of layers, feature channels, and maximal representation degree.

We use $L ( X , Y )$ to denote the vector space of linear maps $X  Y$ , and $P ( X , Y )$ the vector space of polynomial maps $X  Y$ . Given the G-actions on $\bar { X }$ and $Y$ , we write $L _ { G } ( \dot { X } , Y ) \subseteq L ( \dot { X } , Y )$ and ${ \dot { P _ { G } } } ( X , Y ) \subseteq { \bar { P } } ( X , Y )$ for the subspaces of G-equivariant maps, i.e., $f ( g ( x ) ) = g { \bigl ( } f ( x ) { \bigr ) }$ for all $g \in G$ and $x \in X$ . If Y carries the trivial action, equivariance reduces to invariance.

Lemma D.4. Let $X , Y , Z$ be representations of a group G. Consider a subset of G-equivariant polynomial maps $S \subseteq P _ { G } ( X , Z )$ that satisfies the spanning condition:

$$
P ( X , Y ) \subseteq \operatorname { s p a n } ( L ( Z , Y ) \circ S ) : = \operatorname { s p a n } ( \{ A \circ p | A \in L ( Z , Y ) , p \in S \} ) .\tag{38}
$$

Then, itfollows that $P _ { G } ( X , Y ) = \operatorname { s p a n } ( L _ { G } ( Z , Y ) \circ S )$

In the special case where $G = H _ { 1 } \times H _ { 2 } ,$ , the spanning condition can be relaxed to

$$
P _ { H _ { 2 } } ( X , Y ) \subseteq \operatorname { s p a n } ( L ( Z , Y ) \circ S ) .\tag{39}
$$

Remark. The case for $G = S O ( 3 ) \times S _ { n }$ is from Thm. 1 of Dym & Maron (2021). In this context, the action of $S _ { n }$ on the representations is faithful, while we are concerned with features that are invariant under $S _ { n }$

Proof. Let $f \in P _ { G } ( X , Y )$ . Since $P _ { G } ( X , Y ) \subseteq P ( X , Y )$ , by the spanning condition, $f$ can be written as a linear combination of compositions:

$$
f = \sum _ { i = 1 } ^ { N } A _ { i } \circ p _ { i } ,\tag{40}
$$

where $p _ { i } \in S$ and $A _ { i } \in L ( Z , Y )$ . Here, we have absorbed the expansion coefficients into the linear maps $A _ { i }$

Since f is G-equivariant, it is a fixed point of the group averaging operator. Applying this operator to both sides of the equation gives:

$$
f ( x ) = \int _ { G } \rho _ { Y } { \bigl ( } g ^ { - 1 } { \bigr ) } f { \bigl ( } \rho _ { X } { \bigl ( } g { \bigr ) } ( x ) { \bigr ) } \mathrm { d } g\tag{41}
$$

$$
= \int _ { G } \rho _ { Y } ( g ^ { - 1 } ) \left( \sum _ { i = 1 } ^ { N } A _ { i } ( \rho _ { X } ( g ) ( x ) ) \right) \mathrm { d } g\tag{42}
$$

$$
= \sum _ { i = 1 } ^ { N } \left( \int _ { G } \rho _ { Y } ( g ^ { - 1 } ) A _ { i } \rho _ { Z } ( g ) \mathrm { d } g \right) p _ { i } ( x ) .\tag{43}
$$

In the last step, we used the fact that the maps $p _ { i } \in S$ are themselves G-equivariant and moved the integral inward. Let us define the averaged linear maps as

$$
( A _ { G } ) _ { i } : = \int _ { G } \rho _ { Y } ( g ^ { - 1 } ) A _ { i } \rho _ { Z } ( g ) \mathrm { d } g .\tag{44}
$$

By construction, each $( A _ { G } ) _ { i }$ <sub>i</sub> is a G-equivariant linear map, i.e., $( A _ { G } ) _ { i } \in L _ { G } ( Z , Y )$ . The expression for $f ( x )$ can now be written as $\begin{array} { r } { f ( x ) = \sum _ { i = 1 } ^ { N } ( A _ { G } ) _ { i } \circ p _ { i } ( x ) } \end{array}$ . This shows that any map in $P _ { G } ( X , Y )$ ) can be expressed as a linear combination of compositions of maps from $S$ and equivariant linear maps from $L _ { G } ( Z , Y )$ ). Therefore,

$$
P _ { G } ( X , Y ) \subseteq \operatorname { s p a n } ( L _ { G } ( Z , Y ) \circ S ) .\tag{45}
$$

The reverse inclusion, span $( L _ { G } ( Z , Y ) \circ S ) \subseteq P _ { G } ( X , Y )$ , is true by definition, since the composition of two equivariant maps is equivariant. Thus, the equality $P _ { G } ( \dot { X } , Y ) = \mathrm { s p a n } ( L _ { G } ( Z , Y ) \dot { \circ } S )$ is established.

The proof for the relaxed condition when $G = H _ { 1 } \times H _ { 2 }$ follows the exact same logic, by taking an $f \in \mathsf { ^ { \prime } } P _ { G } ( X , Y ) \subseteq P _ { H _ { 2 } } ( X , Y )$ and averaging over the group $H _ { 1 }$ □

Lemma D.5. Consider the TFN model with polynomial parametric radialfunction, a input space X, a lifted representation space $Z ,$ and a final output space Y. Suppose the final output space $Y = { \tilde { Y } } \otimes \mathbb { R } ^ { n }$ , where $\tilde { Y }$ is a representation of $S O ( 3 ) \ o r \ O ( 3 )$ , and the symmetric group $S _ { n }$ acts on Y by permuting the coordinates in the $\mathbb { R } ^ { n }$ factor. Then the family ofpolynomial feature maps, ${ \mathcal { F } } _ { \mathrm { f e a t } } ^ { \mathrm { p o l y } } \subseteq P _ { G } ( X , Z )$ , used in TFN satisfies the relaxed spanning condition:

$$
P _ { S _ { n } } ( X , Y ) \subseteq \operatorname { s p a n } ( L ( Z , Y ) \circ \mathcal { F } _ { \mathrm { f e a t } } ^ { \mathrm { p o l y } } ) .\tag{46}
$$

Remark. Lem. 4 in Dym & Maron (2021) proves the case for $G = S O ( 3 ) \times S _ { n }$ . The proof for $O ( 3 )$ is similar to that for $S O ( 3 )$ and is therefore not repeated here. It is worth noting that in the $O ( 3 )$ case, the spherical harmonics map can still be used to construct the component-wise map from representation $\dot { V } _ { l = 1 ^ { - } } \cong \mathbb { R } ^ { 3 }$ to symmetric algebra $\mathrm { S y m } _ { k } ( V _ { l = 1 ^ { - } } )$ , which in turn is used to build the lifted representation.

Lemma D.6. In Ex. 2.1, the function families $\mathcal { F } _ { \mathrm { T F N } } ^ { \mathrm { p o l y } }$ contain all equivariant polynomial maps.

Proof. According to Lem. D.4, to prove that a family of equivariant maps contains all equivariant polynomials, it is sufficient to show that it satisfies the (potentially relaxed) spanning condition. We verify this for $\mathcal { F } _ { \mathrm { T F N } } ^ { \mathrm { p o l y } }$

By Lem. D.5, for $Y = { \tilde { Y } } \otimes \mathbb { R } ^ { n }$ , the family $\mathcal { F } _ { \mathrm { f e a t } } ^ { \mathrm { p o l y } }$ satisfies the spanning condition

$$
P _ { S _ { n } } ( X , Y ) \subseteq \operatorname { s p a n } ( L ( Z , Y ) \circ \mathcal { F } _ { \mathrm { f e a t } } ^ { \mathrm { p o l y } } ) .\tag{47}
$$

We now only need to verify the following spanning condition:

$$
P _ { S _ { n } } ( X , \tilde { Y } ) \subseteq \mathrm { s p a n } ( L ( Z , \tilde { Y } ) \circ \mathcal { F } _ { \mathrm { f e a t } } ^ { \mathrm { p o l y } } ) .\tag{48}
$$

For any $g \in P _ { S _ { n } } ( X , \tilde { Y } )$ , we construct an $S _ { n }$ -equivariant polynomial map to $Y$ as

$$
{ \tilde { g } } : X \to Y , \quad { \tilde { g } } ( x _ { 1 } , \dots , x _ { n } ) _ { j } = g ( x _ { 1 } , \dots , x _ { n } ) \quad { \mathrm { f o r ~ } } j = 1 , \dots , n .\tag{49}
$$

This function is clearly an $S _ { n }$ -equivariant polynomial because all n components of its output are identical. Thus, $\tilde { g } \in \dot { P _ { S _ { n } } } ( X , Y )$ . By the spanning condition, we can write

$$
\tilde { g } = \sum _ { i } A _ { i } \circ p _ { i } , \quad \mathrm { w h e r e ~ } A _ { i } \in L ( Z , Y ) \mathrm { ~ a n d ~ } p _ { i } \in \mathcal { F } _ { \mathrm { f e a t } } ^ { \mathrm { p o l y } } .\tag{50}
$$

Applying the averaging operator to both sides yields

$$
{ \frac { 1 } { | S _ { n } | } } \sum _ { \sigma \in S _ { n } } { ( \tilde { g } ( \sigma ( x ) ) ) _ { \sigma ( j ) } } = g ( x ) = \sum _ { i } ( A _ { S _ { n } } ) _ { i } \circ p _ { i } ( x ) ,\tag{51}
$$

where

$$
( A _ { S _ { n } } ) _ { i } = \frac { 1 } { | S _ { n } | } \sum _ { \sigma \in S _ { n } } \pi _ { \tilde { Y } } \circ \rho _ { Y } ( \sigma ^ { - 1 } ) \circ A _ { i } \circ \rho _ { Z } ( \sigma ) \in L _ { G } ( Z , \tilde { Y } ) .\tag{52}
$$

This establishes the spanning condition

$$
P _ { S _ { n } } ( X , \tilde { Y } ) \subseteq \mathrm { s p a n } ( L _ { G } ( Z , \tilde { Y } ) \circ \mathcal { F } _ { \mathrm { f e a t } } ^ { \mathrm { p o l y } } ) .\tag{53}
$$

We consider the higher-order approximation theorem of MLPs, and obtain $C ^ { \infty }$ -density via higherorder approximations based on polynomial parameterization.

Lemma D.7 (Higher-Order Approximation Theorem for MLPs, Pinkus (1999), Thm. 4.1). For any compact set $K \subset \mathbb { R } ^ { n }$ , any function $f \in C ^ { m } ( \mathbb { R } ^ { n } )$ , and any non-polynomial activation function $\sigma \in C ^ { m } ( \mathbb { R } )$ , let $\epsilon > 0$ . Then there exists an MLP parameterized map g with activationfunction σ such that

$$
\operatorname* { m a x } _ { x \in K } | \partial _ { x _ { 1 } } ^ { k _ { 1 } } \cdot \cdot \cdot \partial _ { x _ { n } } ^ { k _ { n } } f ( x ) - \partial _ { x _ { 1 } } ^ { k _ { 1 } } \cdot \cdot \cdot \partial _ { x _ { n } } ^ { k _ { n } } g _ { \theta } ( x ) | < \epsilon\tag{54}
$$

for all non-negative integers $k _ { 1 } , \ldots , k _ { n }$ with $k _ { 1 } + \cdots + k _ { n } < m $

Theorem 5.1. In Ex. 2.1, thefunctionfamilies $\mathcal { F } _ { \mathrm { T F N } }$ with smooth non-polynomial activation function are $C ^ { \infty }$ -dense in the space ofsmooth equivariant maps $C _ { G } ^ { \infty } ( X , Y )$ . That is,for any integer $r \geq 0$ any map $f \in C _ { G } ^ { \infty } ( X , \bar { Y } )$ , any compact set $K \subset X$ , and any $\epsilon > 0 ,$ , there exists $g \in \mathcal { F } _ { \mathrm { T F N } }$ such that

$$
\begin{array} { r } { \operatorname* { m a x } _ { x \in K } \left\| D ^ { k } f ( x ) - D ^ { k } g ( x ) \right\| < \epsilon , k \le r . } \end{array}\tag{10}
$$

Proof of Thm. 5.1. We first extend the polynomial approximation argument of Lem. 1 in Dym & Maron (2021) to the $C ^ { r }$ topology. Let $K _ { G } = \{ \rho _ { X } ( g ) x : g \in G , x \in K \}$ , which is compact. By the $C ^ { r }$ -density of polynomials on finite-dimensional Euclidean spaces, for any $\epsilon > 0$ there exists a polynomial map $p : X \to Y$ such that

$$
\operatorname* { m a x } _ { x \in K _ { G } } \| D ^ { k } f ( x ) - D ^ { k } p ( x ) \| < \epsilon , \qquad k \le r .\tag{55}
$$

Define its equivariant symmetrization by

$$
p _ { G } ( x ) : = \int _ { G } \rho _ { Y } ( g ^ { - 1 } ) p ( \rho _ { X } ( g ) x ) d g .\tag{56}
$$

Since the actions are linear, $p _ { G }$ is an equivariant polynomial. Moreover, differentiation commutes with the integral, and compactness of $G$ implies that $p _ { G }$ converges to $f$ in the $C ^ { r }$ norm on $K$ as $\delta  0$ . Hence equivariant polynomials are $C ^ { \infty }$ -dense in $C _ { G } ^ { \infty } ( X , \bar { Y } )$

It remains to show that every map in $\mathcal { F } _ { \mathrm { T F N } } ^ { \mathrm { p o l y } }$ can be approximated in the $C ^ { r }$ topology by a TFN whose radial functions are parameterized by MLPs. Fix a polynomial TFN $p _ { \mathrm { T F N } }$ and a compact set K. Only finitely many polynomial radial functions occur in this network. By Lem. D.7, for each such radial function and every $\epsilon > 0$ , there exists an MLP radial function whose derivatives up to order r differ by at most ϵ on the compact interval of pairwise distances attained on $K$

Each TFN layer is obtained from these radial functions and the previous-layer features through finitely many sums, tensor products, Clebsch–Gordan projections, and linear maps. By the Leibniz and chain rules, these operations are continuous in the $C ^ { \hat { r } }$ topology on compact sets. Induction over the finitely many layers therefore shows that, for every $\epsilon > 0$ , the polynomial TFN can be approximated by an MLP-parameterized TFN $g$ such that

$$
\operatorname* { m a x } _ { x \in K } \| D ^ { k } p _ { \mathrm { T F N } } ( x ) - D ^ { k } g ( x ) \| < \epsilon , \qquad k \leq r .\tag{57}
$$

Together with Lem. D.6 and the $C ^ { r }$ -density of equivariant polynomials proved above, this establishes the theorem.

## D.3 SOME RESULTS ON TOPOLOGY

Here we adopt the definition of $N _ { G } ( H )$ and $N _ { G } ( H , H ^ { \prime } )$ from $\ S \mathrm { { C . 1 } }$ , and consider the twisted product between G-sets as defined in Chap. 2, Sec. 2 of Bredon (1972). We denote $X _ { H } : = \{ x \in X \ \mathsf { \bar { l } } G _ { x } =$ $H \}$ , from which we obtain the following lemma on topology.

Lemma D.8. When the action ofa group G on a G-manifold M isfaithful, thefollowing decompositions hold:

$$
M _ { ( H ^ { \prime } ) } = M _ { H ^ { \prime } } \times _ { N _ { G } ( H ^ { \prime } ) / H ^ { \prime } } G / H ^ { \prime }\tag{58}
$$

and

$$
M _ { ( H ^ { \prime } ) } ^ { H } = M _ { H ^ { \prime } } \times _ { N _ { G } ( H ^ { \prime } ) / H ^ { \prime } } N _ { G } ( H , H ^ { \prime } ) / H ^ { \prime } .\tag{59}
$$

Proof. The first identity is derived from Thm. 1.31 in Meinrenken (2003), which states the existence of a homeomorphism

$$
\sigma : M _ { H ^ { \prime } } \times _ { N _ { G } ( H ^ { \prime } ) / H ^ { \prime } } G / H ^ { \prime }  M _ { ( H ^ { \prime } ) } , \mathrm { g i v e n b y } \sigma ( [ x , g H ^ { \prime } ] ) = g ( x ) .\tag{60}
$$

We now prove that $\sigma ( M _ { H ^ { \prime } } \times _ { N _ { G } ( H ^ { \prime } ) / H ^ { \prime } } ( N _ { G } ( H , H ^ { \prime } ) / H ^ { \prime } ) ) = M _ { ( H ^ { \prime } ) } ^ { H }$ . The second identity then follows by restricting this homeomorphism.

( ) Take an element [x, gH<sup>′</sup>] where $x \in M _ { H ^ { \prime } }$ and $g \in N _ { G } ( H , H ^ { \prime } )$ . Its image under σ is $g ( x )$ . The isotropy subgroup is $\stackrel {  } { G _ { g ( x ) } } = g G _ { x } g ^ { - 1 } = g H ^ { \prime } g ^ { - 1 }$ . Since $\dot { g } \in \dot { N _ { G } } ( H , H ^ { \prime } )$ , we have $H \subseteq { \dot { g } } { \dot { H ^ { \prime } } } g ^ { - 1 }$ This implies $g ( x ) \in M ^ { H }$ , and since its orbit type is $( H ^ { \prime } )$ , we have $g ( x ) \in M _ { ( H ^ { \prime } ) } ^ { H } .$

( ) Take any $y \in M _ { ( H ^ { \prime } ) } ^ { H }$ . This means $H \subseteq G _ { y }$ and $( G _ { y } ) = ( H ^ { \prime } )$ . The latter implies there exists a $g _ { 0 } \in G$ such that $G _ { y } = g _ { 0 } H ^ { \prime } g _ { 0 } ^ { - 1 }$ . The condition $H \subseteq g _ { 0 } H ^ { \prime } g _ { 0 } ^ { - 1 }$ implies that $g _ { 0 } \in N _ { G } ( H , H ^ { \prime } )$ . Let $x = g _ { 0 } ^ { - 1 } ( y )$ . Then $G _ { x } = H ^ { \prime }$ , so x $\in { \cal M } _ { H ^ { \prime } }$ . We can then write y as

$$
y = g _ { 0 } ( g _ { 0 } ^ { - 1 } ( y ) ) = g _ { 0 } ( x ) = \sigma ( [ x , g _ { 0 } H ^ { \prime } ] ) .\tag{61}
$$

Since $x \in M _ { H ^ { \prime } }$ and $g _ { 0 } \in N _ { G } ( H , H ^ { \prime } )$ , this shows that y is in the image of the restricted domain. The inclusion is thus proven. □

From the following proposition onward, we need to invoke stratification theory. For the definitions of Whitney conditions and stratifications, we refer to Chap. 3, Sec. 9 of Field (2007). Going forward, unless otherwise specified, all function spaces in this paper are equipped with the (weak) $C ^ { \infty }$ topology. For the topology on function spaces, we refer to Chap. 2, Sec. 1 of Hirsch (1976).

Proposition D.9. Consider smooth manifolds M and N. Let $( S _ { \alpha } , S _ { \beta } )$ be a pair of submanifolds in N that satisfies Whitney’s condition (a). If a map $f : M \to N$ is transverse to $S _ { \alpha }$ at a point $x \in f ^ { - 1 } ( S _ { \alpha } )$ , then there exists a neighborhood U off in $C ^ { \infty } ( M , N )$ and a neighborhood V ofx in M, such thatfor any map $g \in U$ and any point $y \in V ,$ , g is transverse to both $S _ { \alpha }$ and $S _ { \beta }$ at y.

Proof. The proof is adapted from Prop. 1.3 of Trotman (1976). We proceed by contradiction. Assume the conclusion is false. Then for any neighborhood V of x and any neighborhood $U$ of $f _ { i }$ , there exists a map $g \in U$ for which the condition ${ g \cap S _ { \alpha } , S _ { \beta } }$ on V does not hold.

For a neighborhood $V _ { 1 }$ of x with radius less than $\epsilon _ { 1 }$ , we can construct a sequence of maps $\{ g _ { n } ^ { ( 1 ) } \}$ converging to f such that transversality to $S _ { \alpha }$ or $S _ { \beta }$ fails on $V _ { 1 }$ . Let the sequence of non-transverse points be $\{ y _ { n } ^ { ( 1 ) } \}$ . We can similarly construct, for neighborhoods $V _ { t }$ of x with radius less than $\epsilon _ { t }$ sequences of maps $\{ g _ { n } ^ { ( t ) } \}$ converging to f and corresponding sequences of non-transverse points $\{ y _ { n } ^ { ( t ) } \}$

Now, consider the diagonal sequence of maps $\{ g _ { n } ^ { ( n ) } \}$ and the corresponding sequence of nontransverse points $\{ y _ { n } ^ { ( n ) } \}$ . We have $g _ { n } ^ { ( n ) }  f$ and $y _ { n } ^ { ( n ) } \to x$ . We can partition the sequence $\{ y _ { n } ^ { ( n ) } \}$ into a subsequence lying in $S _ { \alpha }$ and another lying in $S _ { \beta }$ . At least one of these two subsequences must be infinite. It is therefore sufficient to negate the following proposition: There exists a sequence of maps $\{ g _ { n } \}  f$ and a sequence of points $\{ y _ { n } \}  x$ with either $\{ y _ { n } \} \subset S _ { \alpha } { \mathrm { o r } } \{ y _ { n } \} \subset S _ { \beta }$ , such that for each n, $g _ { n }$ is not transverse to $\dot { S } _ { \alpha }$ or $S _ { \beta }$ at $y _ { n }$

This is impossible. Note that for a sufficiently small $\epsilon _ { 1 }$ , the closure of the neighborhood $V _ { 1 }$ is compact by local compactness of the manifold, and we can consider the control of the neighborhood $U$ over functions on this compact closure in the $C ^ { \infty }$ topology of functioin space. Also note that $\mathrm { d } ( g _ { n } ) _ { y _ { n } } ( T _ { y _ { n } } M )$ converges to $\mathrm { d } f _ { x } ( T _ { x } M )$ . For the case of $S _ { \alpha } .$ , by the smoothness of the manifold, $T _ { g _ { n } ( y _ { n } ) } S _ { \alpha }$ converges to $T _ { f ( x ) } S _ { \alpha }$ . For the case of $S _ { \beta }$ , by Whitney $\mathrm { ^ { , } s }$ condition (a), the limit of $\dot { T _ { g _ { n } \left( y _ { n } \right) } } S _ { \beta }$ contains $T _ { f ( x ) } \dot { S _ { \alpha } }$ . However, $f$ is transverse to $S _ { \alpha }$ at x, meaning

$$
\mathrm { d } f _ { x } ( T _ { x } M ) + T _ { f ( x ) } S _ { \alpha } = T _ { f ( x ) } N .\tag{62}
$$

Therefore, due to convergence, the transversality condition for $S _ { \alpha }$ or $S _ { \beta }$ must hold for $g _ { n }$ at $y _ { n }$ for sufficiently large $n ,$ which is a contradiction. □

Corollary D.10. Under the conditions of the previous proposition, assume that the stratification $S = \{ S _ { \alpha } \}$ satisfies Whitney’s condition (a). If a map f is transverse to a stratum $S _ { \alpha }$ at a point $x \in f ^ { - 1 } ( \overbar { S } _ { \alpha } )$ , then there exists a neighborhood $V _ { x }$ ofx and a neighborhood $U _ { x }$ off in $C ^ { \infty } ( M , N )$ such that for any $g \in U _ { x }$ and any $y \in V _ { x } ,$ , g is transverse to the entire stratification $S$ in the neighborhood $V _ { x }$

Proof. Let x be a point where $f$ is transverse to $S _ { \alpha }$ . We consider all strata $S _ { \beta }$ that satisfies ${ \overline { { S _ { \beta } } } } \cap S _ { \alpha } \neq$ ∅. Since a stratification is locally finite, there are only a finite number of such adjacent strata. For each such adjacent stratum $S _ { \beta }$ , the pair $( S _ { \alpha } , S _ { \beta } )$ satisfies Whitney’s condition (a). By Prop. D.9, there exist neighborhoods $U _ { x } ^ { \left( \alpha , \beta \right) }$ of $f$ and $V _ { x } ^ { \left( \alpha , \beta \right) }$ of $x$ where transversality to both $S _ { \alpha }$ and $S _ { \beta }$ holds. We can then construct the desired neighborhoods by taking the finite intersection of these individual neighborhoods labelled by $\beta \colon$

$$
U _ { x } : = \bigcap _ { \overline { { S _ { \beta } } } \cap S _ { \alpha } \neq \emptyset } U _ { x } ^ { ( \alpha , \beta ) } \quad \mathrm { a n d } \quad V _ { x } : = \bigcap _ { \overline { { S _ { \beta } } } \cap S _ { \alpha } \neq \emptyset } V _ { x } ^ { ( \alpha , \beta ) } .\tag{63}
$$

Since the intersection is finite, $U _ { x }$ and $V _ { x }$ are still open neighborhoods of $f$ and $x ,$ respectively. For any $g \in U _ { x }$ and $y \in V _ { x } , g$ is transverse to $S _ { \alpha }$ and all its adjacent strata $S _ { \beta }$ at $y$ . Therefore, it is transverse to the entire stratification $S$ within the local neighborhood $V _ { x }$

Lemma D.11. For a G-manifold M and a G-representation $Y _ { i }$ , the collection of submanifolds

$$
\{ ( M _ { ( H ) } \times Y _ { ( H ^ { \prime } ) } ) _ { ( H ) } \} _ { ( H ^ { \prime } ) \geq ( H ) }\tag{64}
$$

forms a Whitney stratification of $( M \times Y ) _ { ( H ) }$

Proof. By Prop. 3.7.1 of Field (2007), we obtain that these point sets in the collection are smooth G-submanifolds. By Lem. D.8, we have

$$
( M _ { ( H ) } \times Y _ { ( H ^ { \prime } ) } ) _ { ( H ) } = ( M _ { ( H ) } \times Y _ { ( H ^ { \prime } ) } ) _ { H } \times _ { N _ { G } ( H ) / H } G / H\tag{65}
$$

$$
= \left( M _ { ( H ) } \times Y _ { ( H ^ { \prime } ) } \right) _ { ( H ) } ^ { H } \times _ { N _ { G } ( H ) / H } G / H\tag{66}
$$

$$
= ( M _ { ( H ) } ^ { H } \times Y _ { ( H ^ { \prime } ) } ^ { H } ) \times _ { N _ { G } ( H ) / H } G / H .\tag{67}
$$

By local trivialization, similar to the proof of Prop. 3.9.2 of Field (2007), it is sufficient to prove that the pair $( Y _ { ( H ^ { \prime } ) } ^ { H } , Y _ { ( K ) } ^ { H } )$ satisfies Whitney’s condition (b). By Prop. 3.9.2 of Field (2007), specialized

to a representation space, the orbit-type stratification of Y is Whitney regular. Hence the pair $( Y _ { ( H ^ { \prime } ) } , \bar { Y } _ { ( K ) } )$ for $( H ^ { \bar { \prime } } ) > ( K )$ satisfies Whitney’s condition (b). We show that the condition is preserved after restriction to the fixed-point space $Y ^ { H }$

Let $x \in Y _ { ( H ^ { \prime } ) } ^ { H } \cap { \overline { { Y _ { ( K ) } ^ { H } } } } ,$ and consider sequences $\{ x _ { n } \} \subset Y _ { ( H ^ { \prime } ) } ^ { H }$ and $\{ y _ { n } \} \subset Y _ { ( K ) } ^ { H }$ converging to x, such that the secant lines $\overline { { x _ { n } y _ { n } } }$ converge to a line l and

$$
T _ { y _ { n } } Y _ { ( K ) } ^ { H } \longrightarrow F .\tag{68}
$$

Passing to a subsequence, we may further assume that

$$
T _ { y _ { n } } Y _ { ( K ) } \longrightarrow E .\tag{69}
$$

Since $( Y _ { ( H ^ { \prime } ) } , Y _ { ( K ) } )$ satisfies Whitney’s condition (b), we have $l \subseteq E$ . Moreover, since $x _ { n } , y _ { n } \in Y ^ { H }$ and $Y ^ { H }$ is a linear subspace of Y, we also have $l \subseteq Y ^ { H }$

It remains to identify F. Since $Y _ { ( K ) }$ is H-invariant and $y _ { n } \in \mathsf { Y } ^ { H }$ , we obtain $T _ { y _ { n } } Y _ { ( K ) } ^ { H } = $ $\left( T _ { y _ { n } } Y _ { \left( K \right) } \right) ^ { H }$ . Choose an H-invariant inner product on $Y$ , and let

$$
P _ { H } = \int _ { H } \rho ( h ) d h\tag{70}
$$

be the orthogonal projection onto $Y ^ { H }$ . Let $Q _ { n }$ and $Q$ denote the orthogonal projections onto $T _ { y _ { n } } Y _ { ( K ) }$ and $E ,$ respectively. Then $Q _ { n } \to Q$ . Since $T _ { y _ { n } } Y _ { ( K ) }$ is H-invariant, $Q _ { n }$ commutes with $\dot { P _ { H } }$ , and hence $Q _ { n } P _ { H }$ is the orthogonal projection onto $\left( T _ { y _ { n } } Y _ { ( K ) } \right) ^ { H }$ . Therefore, $Q _ { n } P _ { H } \longrightarrow Q P _ { H }$ , which implies $F = E ^ { H } = E \cap Y ^ { H }$ . Combining $l \subseteq E$ and $l \subseteq Y ^ { H }$ , we obtain $l \subseteq E \cap Y ^ { H } = F$ . Thus the pair $( Y _ { ( H ^ { \prime } ) } ^ { H } , Y _ { ( K ) } ^ { H } )$ satisfies Whitney’s condition (b).

Lemma D.12. Let M and N be topological spaces, with M being compact. Consider a continuous map $f : M \to N$ . Let C be a closed set in $\bar { N } ,$ , and suppose there exists a neighborhood V of the preimage $f ^ { - 1 } ( C )$

Then there exists a neighborhood U off in $C ( M , N )$ with the $C ^ { 0 }$ topology (compact-open topology) such that for any map ${ \dot { \boldsymbol { g } } } \in U _ { : }$ , we have $y ( M \setminus V ) \cap C = \emptyset$

Proof. Note that $M \backslash V$ is a compact set, and $N \backslash C$ is an open set. By the definition of the compact-open topology, e.g. Definition 43.1 of Willard (1970), the set

$$
U : = \{ h \in C ( M , N ) \mid h ( M \setminus V ) \subseteq N \setminus C \}\tag{71}
$$

is an open set.

Since V is a neighborhood of $f ^ { - 1 } ( C )$ , we have $f ^ { - 1 } ( C ) \subseteq V$ , which implies $f ( M \setminus V ) \subseteq N \setminus C$ Therefore, the map f itself is an element of $U ,$ meaning U is an open neighborhood of $f .$ For any map $g \in U$ , its definition directly implies that $g ( { \bar { M } } \setminus V ) \subseteq { \bar { N } } \setminus C$ , which is equivalent to $g ( \mathcal { \bar { M } } \setminus \bar { V } ) \cap C = \emptyset$ . Thus, this set U is the desired neighborhood. □

## D.4 GENERIC EQUIVARIANT MAPPINGS

Proposition D.13 (Equivariant Smooth Extension). Let M be a smooth G-manifold and let $S \subset M$ be a smooth and compact G-submanifold. For any smooth equivariant map $f \in { \dot { C } } _ { G } ^ { \infty } ( S , Y )$ defined on S that maps into a representation space Y, there exists a smooth equivariant extension $\tilde { f } \in C _ { G } ^ { \infty } ( M , Y )$ such that its restriction to $S i s f , i . e . , \tilde { f } | _ { S } = f ,$

Proof. The proof strategy is the same as that for the Tietze-Gleason Theorem (see Chap. 1, Thm. 2.3 of Bredon (1972)).

First, we can view the map $f$ as a collection of dim $Y$ smooth real-valued functions defined on $S .$ By the standard smooth extension theorem, $\mathrm { e . g }$ ., Lem. 5.34 in Lee (2012), there exists a smooth (but not necessarily equivariant) extension $\varphi \in C ^ { \bar { \infty } } ( M , Y )$ such that $\varphi | _ { S } = f$

We then construct an equivariant map from $\varphi$ using the group averaging operator. Define the map $\psi : M \to Y$ by

$$
\psi ( x ) = \int _ { G } \rho _ { Y } { \bigl ( } g ^ { - 1 } { \bigr ) } \varphi { \bigl ( } \rho _ { X } { \bigl ( } g { \bigr ) } ( x ) { \bigr ) } \mathrm { d } g .\tag{72}
$$

Due to the linearity of the integral and the properties of the Haar measure, this map is equivariant. For any $h \in G \colon$

$$
\psi ( h ( x ) ) = \int _ { G } \rho _ { Y } ( g ^ { - 1 } ) \varphi ( g ( h ( x ) ) ) \mathrm { d } g\tag{73}
$$

$$
= \int _ { G } \rho _ { Y } ( ( g ^ { \prime } h ^ { - 1 } ) ^ { - 1 } ) \varphi ( g ^ { \prime } ( x ) ) \mathrm { d } g ^ { \prime }\tag{74}
$$

$$
= \rho _ { Y } ( h ) \int _ { G } \rho _ { Y } ( ( g ^ { \prime } ) ^ { - 1 } ) \varphi ( g ^ { \prime } ( x ) ) \mathrm { d } g ^ { \prime }\tag{75}
$$

$$
= \rho _ { Y } ( h ) \psi ( x ) .\tag{76}
$$

Next, we verify that the restriction of $\psi$ to the submanifold S is equal to the original map $f .$ For any $s \in S ;$

$$
\psi ( s ) = \int _ { G } \rho _ { Y } ( g ^ { - 1 } ) \varphi ( g ( s ) ) \mathrm { d } g\tag{77}
$$

$$
= \int _ { G } \rho _ { Y } ( g ^ { - 1 } ) f ( g ( s ) ) \mathrm { d } g\tag{78}
$$

$$
= \int _ { G } \rho _ { Y } ( g ^ { - 1 } ) \rho _ { Y } ( g ) f ( s ) \mathrm { d } g\tag{79}
$$

$$
= \int _ { G } f ( s ) \mathrm { d } g = f ( s ) \int _ { G } 1 \mathrm { d } g = f ( s ) .\tag{80}
$$

Finally, the smoothness of ψ follows from the smoothness of $\varphi$ and the properties of integration over a compact group. This can be verified by a local coordinate analysis. Thus, ψ is the desired smooth equivariant extension $\tilde { f } .$ □

Corollary D.14. Consider maps from a G-manifold X to a representation space $Y ,$ where X contains a compact, smooth G-submanifold M. Let S be a subset of $C _ { G } ^ { \infty } ( M , Y )$ that contains an open dense set. Then the set of maps $f \in C _ { G } ^ { \infty } ( X , Y )$ whose restriction to M lies in $S ( i . e . , f | _ { M } \bar { \in } S )$ also contains an open dense subset of $\overleftarrow { C } _ { G } ^ { \infty } ( X , \overleftarrow { Y } )$ .

Proof. We recall that a set contains an open dense subset if and only if for any non-empty open set, its intersection with the set contains a non-empty open subset. Let $\dot { A ^ { . } } { = } \{ f \in \dot { C } _ { G } ^ { \infty } ( X , \dot { Y } ) \operatorname { | } \dot { f } | _ { M } \in S \}$ Our goal is to show that for any $f \in C _ { G } ^ { \infty } ( \dot { X } , \dot { Y } )$ and any of its open neighborhoods $U ,$ there exists a non-empty open set $V \subseteq U \cap A$

Consider the restriction map Res ${ } _ { M } : C _ { G } ^ { \infty } ( X , Y ) \to C _ { G } ^ { \infty } ( M , Y )$ , by Prop. D.13, this map is surjective.   
Furthermore, this map is continuous and open.

Let U be an open neighborhood of an arbitrary map $f .$ Since ${ \mathrm { R e s } } _ { M }$ is an open map, its image, ${ \mathrm { R e s } } _ { M } ( U )$ , is an open neighborhood of $f \vert _ { M }$ . By our hypothesis on $S _ { \ i }$ , its intersection with the open set Res $_ M ( U )$ must contain a non-empty open subset. Let us call this non-empty open set $V ^ { \prime }$ . We have

$$
V ^ { \prime } \subseteq \operatorname { R e s } _ { M } ( U ) \cap S .\tag{81}
$$

Now, let’s consider the preimage $V : = \operatorname { R e s } _ { M } ^ { - 1 } ( V ^ { \prime } )$ . Since ${ \mathrm { R e s } } _ { M }$ is continuous and $V ^ { \prime }$ is open, $V$ is an open set. Since ${ \mathrm { R e s } } _ { M }$ is surjective, the preimage $V$ must also be non-empty. Therefore $\bar { U } \cap V$ is a non-empty open subset of $U \cap A$ , which proves the corollary. □

For $C ^ { r }$ manifolds X and $Y , C ^ { r } ( X )$ is the set of $C ^ { r }$ real-valued functions on $X ,$ , and $C ^ { r } ( X , Y )$ is the set of $C ^ { r }$ maps from $X$ to $\dot { Y }$ . For manifolds with a $C ^ { r }$ G-action, $C _ { G } ^ { r } ( X , Y )$ is the set of $C ^ { r }$ equivariant maps from X to Y . We assume these function spaces are endowed with the $C ^ { r }$ topology; in our proofs, we always consider the $C ^ { \infty }$ topology. For a map $f \in C ^ { 1 } ( X , Y )$ and a submanifold $A \subseteq { \bar { Y } } , f \pitchfork A$ denotes that f is transverse to A. For G-manifolds, $f \pitchfork _ { G }$ A denotes that $f$ is in equivariant general position with respect to the G-submanifold A defined in Bierstone (1977). For $f \in C ^ { r } ( X , \bar { Y } )$ , the jet map $j ^ { r } f : X \stackrel { \cdot } {  } J ^ { r } ( X , Y )$ maps a point x to the equivalence class of the first r derivatives of f at x. We obtain the following results.

Proposition D.15 (Bierstone (1977), Thm. 1.3). Let M and N be smooth G-manifolds. IfP is a closed G-submanifold ofN and K is a compact subset ofM, then the set ofmaps $\bar { f } \in C _ { \bar { G } } ^ { \infty } ( M , N )$ satisfying f ⋔<sub>G</sub> P on K forms an open subset of $C _ { G } ^ { \infty } ( M , N )$ (in the Whitney $C ^ { \infty }$ topology).

Proposition D.16 (Bierstone (1977), Thm. 1.4). Let M, N be smooth G-manifolds and $P$ be a G-submanifold ofN. The set ofmaps $f \in C _ { G } ^ { \infty } ( M , N )$ satisfying f ⋔ P forms a residual subset of $C _ { G } ^ { \infty } ( M , N )$ , i.e., a countable intersection of open dense sets (in the Whitney $C ^ { \infty }$ topology).

Remark. For compact $M$ , note that the Whitney $C ^ { \infty }$ topology coincides with the $C ^ { \infty }$ topology.

Proposition D.17 (Stratumwise Transversality Theorem). For any orbit type $( H ) \in { \mathcal { O } } _ { G } ( M )$ , a map $f \in { \overline { { C _ { G } ^ { \infty } ( M , N ) } } }$ with f ⋔ P satisfies the stratumwise transversality property:

$$
f | _ { M _ { H } } : M _ { H } \mapsto N ^ { H } , f | _ { M _ { H } } \oplus P ^ { H } .\tag{82}
$$

Alternatively, this can be expressed in the language ofjets as

$$
j ^ { 0 } f | _ { M _ { ( H ) } } : M _ { ( H ) } \mapsto ( M _ { ( H ) } \times N ) _ { ( H ) } , j ^ { 0 } f | _ { M _ { ( H ) } } \oplus ( M _ { ( H ) } \times P ) _ { ( H ) }\tag{83}
$$

Remark. The fact that $f \pitchfork _ { G } P$ implies stratumwise transversality on the fixed-point sets is a fundamental conclusion derived from Prop. 6.4 of Bierstone (1977). Our proof below is a straightforward corollary of this.

Proof. From the discussion in Prop. 6.4 of Bierstone (1977), we have

$$
j ^ { 0 } ( f | _ { M _ { ( H ) } } ) : M _ { ( H ) } \mapsto ( M \times N ) _ { ( H ) } , \quad j ^ { 0 } ( f | _ { M _ { ( H ) } } ) \pitchfork ( M \times P ) _ { ( H ) } .\tag{84}
$$

Note that the image of the map $j ^ { 0 } ( f | _ { M _ { ( H ) } } )$ is contained within $M _ { ( H ) } \times N$ . The transversality condition is an equality of tangent spaces:

$$
\mathrm { d } ( j ^ { 0 } ( f | _ { M _ { ( H ) } } ) ) _ { x } ( T _ { x } M _ { ( H ) } ) + T _ { y } ( ( M \times P ) _ { ( H ) } ) = T _ { y } ( ( M \times N ) _ { ( H ) } ) .\tag{85}
$$

Intersecting both sides of this equation with $T _ { y } ( ( M _ { ( H ) } \times N ) _ { ( H ) } )$ yields

$$
\mathrm { d } ( j ^ { 0 } ( f | _ { M _ { ( H ) } } ) ) _ { x } ( T _ { x } M _ { ( H ) } ) + T _ { y } ( ( M _ { ( H ) } \times P ) _ { ( H ) } ) = T _ { y } ( ( M _ { ( H ) } \times N ) _ { ( H ) } ) .\tag{86}
$$

Therefore, we have

$$
j ^ { 0 } ( f | _ { M _ { ( H ) } } ) : M _ { ( H ) } \to ( M _ { ( H ) } \times N ) _ { ( H ) } , \quad j ^ { 0 } ( f | _ { M _ { ( H ) } } ) \pitchfork ( M _ { ( H ) } \times P ) _ { ( H ) } .\tag{87}
$$

Proposition D.18 (Bierstone (1977), Sec. 7). Consider smooth G-manifolds M and N. Let P be a smooth G-submanifold ofN. Ifan equivariant map f satisfies f ⋔<sub>G</sub> P at a point x $\in f ^ { - 1 } ( P )$ , then there exists a neighborhood U off in $C _ { G } ^ { \infty } ( M , N )$ and a G-invariant neighborhood V ofthe orbit $G ( x )$ such that for any map $g \in U$ and any point $y \in V$ , it holds that g ⋔<sub>G</sub> P at y.

Remark. The proposition above can also be derived from the properties of stratifications given in Prop. D.9 by the definition of equivariant general position in Bierstone (1977).

Proposition D.19. Let G be a compact Lie group, M be a compact, smooth G-manifold, and Y be a G-representation space. The set ofsmooth equivariant maps $f \in C _ { G } ^ { \infty } ( M , Y )$ that satisfy the transversality condition

$$
j ^ { 0 } ( f | _ { M _ { ( H ) } } ) \pitchfork ( M _ { ( H ) } \times Y _ { ( H ^ { \prime } ) } ) _ { ( H ) }\tag{88}
$$

for all pairs oforbit types $( H )$ and $( H ^ { \prime } )$ such that $( H )$ is present in M and $\left( H ^ { \prime } \right) \geq \left( H \right)$ , contains an open dense subset of $C _ { G } ^ { \infty } ( \dot { M } , Y )$

Remark. The proof of density is straightforward. However, the openness of this property does not generally hold. A counterexample can be constructed following Ex. 2.1 of Bierstone (1977).

Proof. Since M is compact, by Proposition 3.7.2 of Field (2007), the orbit types are finite. For any given symmetry type $( \bar { H } )$ , by Prop. D.16 the set of maps satisfying f ⋔<sub>G</sub> $Y _ { ( H ) }$ is an intersection of a finite number of residual sets, and is therefore itself a residual set. By Prop. D.17 the set of maps satisfying the transversality condition stated in the theorem is dense.

Instead of directly proving openness property, we prove a related proposition: for a map f that satisfies f ⋔<sub>G</sub> $Y _ { ( H ) }$ for all orbit types (H), there exists a neighborhood U of f such that for any $g \in U$

$$
\begin{array} { r } { j ^ { 0 } ( g | _ { M _ { ( H ) } } ) \Uparrow ( M _ { ( H ) } \times Y _ { ( H ^ { \prime } ) } ) _ { ( H ) } \quad \mathrm { f o r a l l } \ ( H ^ { \prime } ) \geq ( H ) . } \end{array}\tag{89}
$$

The proof proceeds by induction on the dimension of the strata in the orbit type stratification of $Y$ ordered from lowest to highest (or equivalently, from highest to lowest symmetry). We discuss the fixed orbit type $( H )$ in $M .$ . Then we can construct neighborhoods that hold for all (H) by taking the intersection of the neighborhoods corresponding to each orbit type (H).

We start with the lowest-dimensional strata. Let $Y _ { ( H _ { 1 } ) }$ be a stratum of minimal dimension. Such a stratum is unique, which corresponds to a maximal symmetry type (G) and is a closed G-submanifold $Y ^ { G }$ . By Prop. D.15, the set of maps in general position to $Y _ { ( H _ { 1 } ) }$ is open in $C _ { G } ^ { \infty } ( M , Y )$ . Thus, there exists a neighborhood $U _ { 1 }$ of f such that for any $g \in U _ { 1 } , g$ ⋔<sub>G</sub> $Y _ { ( H _ { 1 } ) }$ <sub>)</sub>. By Prop. D.17, this implies that for all (H),

$$
j ^ { 0 } ( g | _ { M _ { ( H ) } } ) \pitchfork ( M _ { ( H ) } \times Y _ { ( H _ { 1 } ) } ) _ { ( H ) } .\tag{90}
$$

Assume the proposition holds for all strata of dimension up to $k - 1$ . Let $U _ { k - 1 }$ be the neighborhood of $f$ found from the inductive hypothesis. For any map $g \in U _ { k - 1 }$ and for any stratum $Y _ { \left( H _ { i } \right) }$ with dim $Y _ { ( H _ { i } ) } \le k - 1$ , we have

$$
j ^ { 0 } ( g | _ { M _ { ( H ) } } ) \pitchfork ( M _ { ( H ) } \times Y _ { ( H _ { i } ) } ) _ { ( H ) } \quad \mathrm { f o r a l l } \ ( H _ { i } ) \ge ( H ) .\tag{91}
$$

Note that by Lem. D.11 the collection $S _ { ( H ) } = \{ ( M _ { ( H ) } \times Y _ { ( H ^ { \prime } ) } ) _ { ( H ) } \} _ { ( H ^ { \prime } ) \geq ( H ) }$ is a Whitney stratification and satisfies the frontier condition. Therefore, by Cor. D.10, for any $x \in M _ { ( H ) }$ with $f ( x ) \in Y _ { ( H _ { i } ) }$ , there exists a neighborhood $V _ { x }$ of x and a neighborhood $U _ { x }$ of $j ^ { 0 } ( f | _ { M _ { ( H ) } } )$ such that for any map in $U _ { x } ,$ it is transverse to the stratification $S _ { ( H ) }$

Next, we need to obtain an open set in $C _ { G } ^ { \infty } ( M , Y )$ . We use the following sequence of maps between function spaces:

$$
C _ { G } ^ { \infty } ( M , Y ) \stackrel { \mathrm { R e s } _ { M _ { ( H ) } } } { \mapsto } C _ { G } ^ { \infty } ( M _ { ( H ) } , Y )\tag{92}
$$

$$
\stackrel { j ^ { 0 } } { \mapsto } C _ { G } ^ { \infty } ( M _ { ( H ) } , ( M _ { ( H ) } \times Y ) _ { ( H ) } )\tag{93}
$$

$$
\hookrightarrow C ^ { \infty } ( M _ { ( H ) } , ( M _ { ( H ) } \times Y ) _ { ( H ) } ) .\tag{94}
$$

The maps between these function spaces are continuous in $C ^ { \infty }$ topology. For example, the 0-jet map is a restriction of the continuous 0-jet map on the general function space, which can be obtained from the continuity of jet map in the Whitney $C ^ { \infty }$ topology established in Chap. 2, Prop. 3.4 of Golubitsky & Guillemin (1973). Since the topology on the equivariant function space is induced from the general function space, the restricted map is also continuous. Similarly, Res $M _ { ( H ) }$ and the inclusion are continuous. From this analysis, we can pull back the neighborhood $U _ { x }$ to obtain a neighborhood $U _ { x } ^ { \prime }$ of f in $C _ { G } ^ { \infty } ( M , Y )$ where the transversality holds.

We deal with the global result. For any i with dim $Y _ { ( H _ { i } ) } \le k - 1$ , the collection of neighborhoods $\{ V _ { x } \} _ { x \in f ^ { - 1 } ( Y _ { ( H _ { i } ) } ) }$ for all i with dim $Y _ { \left( H _ { i } \right) } \le k { - } 1$ forms an open cover of $f ^ { - 1 } ( Y _ { ( H _ { i } ) } )$ . By the frontier condition, any stratum that intersects the closure $\overline { { Y _ { ( H _ { i } ) } } }$ has strictly smaller dimension than $Y _ { \left( H _ { i } \right) }$ Hence $\cup _ { i } Y _ { ( H _ { i } ) }$ is closed. Since M is compact and f is continuous, the preimage $f ^ { - 1 } ( \bigcup _ { i } Y _ { ( H _ { i } ) } )$ is compact, so there exists a finite subcover $\{ V _ { x _ { n } } \} _ { n = 1 } ^ { N }$ of the preimage. We construct the neighborhood $\textstyle V _ { k } : = \bigcup _ { n = 1 } ^ { N } V _ { x _ { \tau } }$ of the preimage in $M$ , and the neighborhood $\begin{array} { r } { U _ { k } ^ { ( 1 ) } : = \bigcap _ { n = 1 } ^ { N } U _ { x _ { r } } ^ { \prime } } \end{array}$ of $f$ in the function space.

In the $C ^ { 0 }$ topology, by Lem. D.12, there exists a neighborhood $U _ { k } ^ { ( 2 ) }$ of f in C(M, Y ) such that for any $g \in U _ { k } ^ { ( 2 ) }$

$$
g ( M \setminus V _ { k } ) \cap \bigcup _ { i } Y _ { ( H _ { i } ) } = \emptyset .\tag{95}
$$

By pulling this back through the inclusions $C _ { \cal G } ^ { \infty } ( M , Y ) \hookrightarrow C ^ { \infty } ( M , Y ) \hookrightarrow C ( M , Y )$ , we obtain a neighborhood $( U _ { k } ^ { ( 2 ) } ) ^ { \prime }$ in $C _ { G } ^ { \infty } ( M , Y )$ . The neighborhood is open in the $C ^ { 0 }$ topology, so it also open in the $C ^ { \infty }$ topology. Let $U _ { k } ^ { \prime } = U _ { k - 1 } \cap U _ { k } ^ { ( 1 ) } \cap ( U _ { k } ^ { ( 2 ) } ) ^ { \prime }$ . For any map $g \in U _ { k } ^ { \prime }$ , we have

$$
\left\{ \begin{array} { l l } { j ^ { 0 } ( g | _ { M _ { ( H ) } } ) \cap S _ { ( H ) } } & { \mathrm { f o r } x \in V _ { k } } \\ { g ( x ) \notin Y _ { ( H _ { i } ) } } & { \mathrm { f o r } x \in M \setminus V _ { k } } \end{array} \right.\tag{96}
$$

for all i with dim $Y _ { ( H _ { i } ) } \le k - 1$

Now we show by contradiction that there exists a neighborhood $U _ { k } \subset U _ { k } ^ { \prime }$ of $f$ where the transversality condition holds for strata of dimension k. By the induction hypothesis, the transversality condition holds for all strata of dimension at most $k - 1$ . Therefore, it also holds for all strata of dimension at most k. The proof idea is from the proof of openness of equivariant general position in Sec. 7 of Bierstone $( 1 9 7 \dot { 7 } )$ , where the closedness of $P$ is replaced by the condition that $f$ does not intersect low-dimensional $Y _ { \left( H _ { i } \right) }$ outside of $V _ { k }$

Assume openness does not hold. Then there exists a sequence $\{ g _ { n } \}  f$ in $U _ { k } ^ { \prime }$ such that each $g _ { n }$ fails the stratumwise transversality condition for some stratum of dimension k. Since the stratumwise transversality condition fails, it follows from Prop. D.17 that $g _ { n }$ ⋔<sub>G</sub> $Y _ { ( H _ { j } ) }$ for all j with dim $Y _ { ( H _ { j } ) } =$ k also does not hold. The points of non-transversality for these maps, $\left\{ y _ { n } \right\}$ , can only occur in $\dot { M ^ { \setminus } } V _ { k }$ and $g _ { n } ( y _ { Y } ) \in \bigcup _ { j } Y _ { ( H _ { j } ) }$ . Since $M \backslash V _ { k }$ is compact, there is a convergent subsequence $\{ y _ { n _ { i } } \}$ with limit x. Then $g _ { n } { \dot { ( \boldsymbol { y } _ { Y } ) } }  f ( \boldsymbol { x } )$ . By construction of $U _ { k } ^ { \prime } , f ( M \setminus V _ { k } )$ does not intersect $Y _ { \left( H _ { i } \right) }$ for any dim $Y _ { ( H _ { i } ) } \le k - 1$ . Thus, we must have $f ( x ) \in Y _ { ( H _ { j } ) }$ for some j.

We claim that $g _ { n _ { i } } ( y _ { n _ { i } } ) \in Y _ { ( H _ { j } ) }$ for all sufficiently large i. Otherwise, there exists a further subsequence $\{ g _ { n _ { i _ { \alpha } } } ( y _ { n _ { i _ { \alpha } } } ) \}$ such that $g _ { n _ { i _ { \alpha } } } ( y _ { n _ { i _ { \alpha } } } ) ~ \in ~ Y _ { ( H _ { j ^ { \prime } } ) }$ for some $j ^ { \prime } \ne j$ and all α, while still $g _ { n _ { i _ { \alpha } } } ( y _ { n _ { i _ { \alpha } } } )  f ( x ) \in Y _ { ( H _ { j } ) }$ . Hence $Y _ { ( H _ { j } ) } \cap { \overline { { Y _ { ( H _ { j ^ { \prime } } ) } } } } \neq \emptyset$ . By the frontier condition of stratification this implies dim $Y _ { ( H _ { j } ) } < \dim Y _ { ( H _ { j ^ { \prime } } ) }$ . However, $Y _ { ( H _ { j } ) }$ and $Y _ { ( H _ { i ^ { \prime } } ) }$ have the same dimension, yielding a contradiction. Therefore, for i sufficiently large, $g _ { n _ { i } } ( y _ { n _ { i } } ) \in Y _ { ( H _ { j } ) }$

We now show that this contradicts the assumption that f ⋔<sub>G</sub> $Y _ { ( H _ { i } ) }$ at x. If f were to satisfy $f \pitchfork _ { G } Y _ { ( H _ { j } ) }$ at x, then by Prop. D.18, there would exist a neighborhood U of $f$ and a G-invariant neighborhood $V$ of $G ( x )$ such that any $g \in U$ satisfies $g \pitchfork _ { G } Y _ { ( H _ { j } ) }$ at any $y \in V$ . This contradicts the existence of sequence $\left\{ g _ { n } \right\}$ and points $\left\{ y _ { n } \right\}$ where transversality fails. This completes the proof.

Proposition D.20. Let  be a $C ^ { \infty }$ -dense family of smooth parameterized maps in $C _ { G } ^ { \infty } ( X , Y )$ and $\{ M _ { j } \}$ be a finite collection of compact, connected and smooth G-submanifolds of X. Let

$$
S _ { ( H )  ( H ^ { \prime } ) } ( f ) = \{ x \in X \mid ( G _ { x } ) = ( H ) , ( G _ { f ( x ) } ) = ( H ^ { \prime } ) \} .\tag{97}
$$

There is a $C ^ { \infty }$ -dense subset $\mathcal { G } \subset \mathcal { F } , f o r g \in \mathcal { G } _ { }$ , the set $S _ { ( H )  ( H ^ { \prime } ) } ( g | _ { M _ { j } } )$ is a disjoint union ofsmooth G-submanifolds of $X ,$ , and its dimension satisfies

$$
\dim S _ { ( H ) \to ( H ^ { \prime } ) } ( g | _ { M _ { j } } ) = \dim ( M _ { j } ) _ { ( H ) } - ( \dim Y ^ { H } - \dim Y _ { ( H ^ { \prime } ) } ^ { H } )\tag{98}
$$

if the right-hand side of the equation is not smaller than dim G  dim H. Otherwise, the set $S _ { ( H )  ( H ^ { \prime } ) } ( g | _ { M _ { j } } )$ is empty.

Proof. By Prop. D.19, for each $M _ { j }$ , there exists an open dense set in $C _ { G } ^ { \infty } ( M _ { j } , Y )$ such that for any map g in this set,

$$
j ^ { 0 } ( g | _ { ( M _ { j } ) _ { ( H ) } } ) \cap ( ( M _ { j } ) _ { ( H ) } \times Y _ { ( H ^ { \prime } ) } ) _ { ( H ) } \quad \mathrm { f o r ~ a l l ~ } ( H ^ { \prime } ) \geq ( H ) .\tag{99}
$$

Therefore, writing codim $x Y : =$ dim $X -$ dim $Y$ for the codimension of $Y$ in X, the dimension theorem for transverse maps (see, e.g., Sec. 2.3 of Arnold et al. (2012)) yields

$$
\mathrm { c o d i m } _ { ( M _ { j } ) _ { ( H ) } } S _ { ( H )  ( H ^ { \prime } ) } = \mathrm { c o d i m } _ { ( ( M _ { j } ) _ { ( H ) } \times Y ) _ { ( H ) } } ( ( M _ { j } ) _ { ( H ) } \times Y _ { ( H ^ { \prime } ) } ) _ { ( H ) } .\tag{100}
$$

Furthermore, from the proof of Lem. D.11, we have

$$
( ( M _ { j } ) _ { ( H ) } \times Y _ { ( H ^ { \prime } ) } ) _ { ( H ) } = ( ( M _ { j } ) _ { ( H ) } ^ { H } \times Y _ { ( H ^ { \prime } ) } ^ { H } ) \times _ { N _ { G } ( H ) / H } G / H .\tag{101}
$$

Thus, we obtain

$$
\mathrm { c o d i m } _ { ( M _ { j } ) _ { ( H ) } } S _ { ( H )  ( H ^ { \prime } ) } = \mathrm { c o d i m } _ { Y ^ { H } } Y _ { ( H ^ { \prime } ) } ^ { H } .\tag{102}
$$

Moreover, since the dimension of an orbit with orbit type $( H )$ in a G-manifold is dim G dim $H$ if the dimension of $S _ { ( H )  ( H ^ { \prime } ) }$ calculated above is less than dim G dim H, then $S _ { ( H )  ( H ^ { \prime } ) }$ is an empty set.

For each $M _ { j } ,$ , we take the corresponding open dense set $U _ { j } \subset C _ { G } ^ { \infty } ( M _ { j } , Y )$ on which the maps satisfy the dimension theorem. Then, by Cor. D.14, the set of functions $f ^ { ^ { \prime } } \in { \dot { C } } _ { G } ^ { \infty } ( X , Y )$ that satisfy $f | _ { M _ { j } } \in U _ { j }$ contains an open dense set $U _ { j } ^ { \prime }$ . Since $\{ M _ { j } \}$ is a finite collection, the intersection $U = \textstyle \bigcap _ { j } U _ { j } ^ { \prime }$ is also an open dense set. Therefore, the intersection of $U$ with the $C ^ { \infty }$ -dense set $\mathcal { F }$ is a dense set . Thus, this intersection is the set of maps in $\mathcal { F }$ that we sought to construct, which is dense and whose elements satisfy the dimension theorem. □

## D.5 PROOF OF THM. 5.2

Theorem 5.2. Let be a equivariant parametrization with $C ^ { \infty }$ approximation capability. Iffor every $( H ) \in { \mathcal { O } } _ { G } ( M )$ we have $( p _ { Y } ( H ) ) \in { \mathcal { O } } _ { G } ( Y )$ , then for any finite union of compact, smooth G-submanifolds ${ \dot { M } } \subset X$ , any $f \in { \dot { C } } _ { G } ^ { \infty } ( X , Y )$ , any integer $r \geq 0 ,$ and any $\epsilon > 0 ,$ , there exists a map $g \in { \mathcal { F } }$ such that

$$
\begin{array} { r } { \operatorname* { m a x } _ { x \in M } \lVert D ^ { k } f ( x ) - D ^ { k } g ( x ) \rVert < \epsilon , k \le r , } \end{array}\tag{11}
$$

and $g | _ { M }$ is almost isovariant relative to $Y .$ . Furthermore, ifthefeature space Y contains a representation $\tilde { Y } ^ { \oplus r }$ for an integer $r > \mathrm { m a x } _ { j }$ dim $M _ { j } \}$ , where Y satisfies the condition $( p _ { Y } ( H ) ) \in { \mathcal { O } } _ { G } ( Y )$ then the approximating map $g | _ { M }$ can be chosen to be isovariant relative to Y.

ProofofThm. 5.2. By Lem. D.8, we consider the relation

$$
\dim N _ { G } ( H , H ^ { \prime } ) - \dim N _ { G } ( H ^ { \prime } ) = \alpha ( H , H ^ { \prime } ) = \dim Y _ { ( H ^ { \prime } ) } ^ { H } - \dim Y ^ { H ^ { \prime } } .\tag{103}
$$

Regarding the G-representation as a faithful representation of $G / \mathrm { k e r } \rho _ { Y }$ , for orbit types satisfying $( p _ { Y } \mathbf { \bar { ( } } H ^ { \prime } ) ) { \bf \bar { \theta } } > I ( H , { \bf \bar { Y } } )$ , the dense and open property of the minimal orbit type by Prop. B.4 in the fixed-point space implies that for each $M _ { j }$

$$
\dim Y ^ { H } - \dim Y _ { ( H ^ { \prime } ) } ^ { H } = \dim Y ^ { H } - \dim Y ^ { H ^ { \prime } } - \alpha _ { G } ( H , H ^ { \prime } ) > 0 .\tag{104}
$$

By Prop. D.20, it implies

$$
\dim ( M _ { j } ) _ { ( H ) } > \dim S _ { ( H ) \to ( H ^ { \prime } ) } ( g | _ { M _ { j } } ) .\tag{105}
$$

With respect to the Hausdorff measure $\mathcal { H } ^ { d }$ , since the dimension of $S _ { ( H )  ( H ^ { \prime } ) } ( g | _ { M _ { j } } )$ is strictly less than the dimension of the manifold it lies in, its measure is zero. Then, by the finiteness of the set of orbit types for a compact manifold and the finiteness of the collection $\{ M _ { j } \}$ , it follows that $g | _ { M }$ is an almost isovariant map relative to $Y$

Furthermore, if we require g to be isovariant relative to $Y _ { \textrm { \scriptsize i } }$ , we need

$$
S _ { ( H )  ( H ^ { \prime } ) } ( g | _ { M _ { j } } ) = \emptyset .\tag{106}
$$

The condition for this set to be empty is related to its codimension. Since the inequality

$$
\dim Y ^ { H } - \dim Y ^ { H ^ { \prime } } - \alpha _ { G } ( H , H ^ { \prime } ) \geq 1\tag{107}
$$

scales with multiplicity r to become

$$
r ( \dim Y ^ { H } - \dim Y ^ { H ^ { \prime } } ) - \alpha _ { G } ( H , H ^ { \prime } ) \geq r ,\tag{108}
$$

it is sufficient to choose the multiplicity for the representation $Y ^ { \oplus r }$ such that $r > \operatorname* { m a x } _ { j } \{ \dim M _ { j } \}$ 口

## E TABLES

All closed subgroups of $O ( 3 )$ are denoted using Schoenflies notation, where it should be noted that $K = S O ( 3 )$ and $\bar { K _ { h } } = O ( 3 )$ . When interpreting the tables, care should be taken to recognize the low-dimensional equivalences: $C _ { s } = C _ { 1 h } = C _ { 1 v } , D _ { 1 } = C _ { 2 } , D _ { 1 h } = C _ { 2 v }$ , and $D _ { 1 d } = C _ { 2 h }$ . In the tables of symmetry infimum, we list a representative subgroup from each conjugacy class and omit the class notation $( \mathrm { e } . \mathrm { g } . , C _ { 2 }$ instead of $( C _ { 2 } ) )$ for clarity.

## E.1 MINIMAL PROPER SUPERGROUPS IN $S O ( 3 )$ OR $O ( 3 )$

Minimal proper supergroups table of $S O ( 3 )$ or $O ( 3 )$ . Some of the results can be found from Fig. 3.2.1.6 of Aroyo (2016). We only present the results for $O ( 3 )$ . The discussion for $S O ( 3 )$ can be obtained by removing the subgroups that are not subgroups of $S O ( 3 )$ (i.e., the subgroups of the first kind), namely $C _ { k } , \bar { D _ { k } } , \bar { T } , O , \bar { I } , C _ { \infty } ^ { \mathrm { ~ \scriptsize ~ \cdot ~ } } D _ { \infty }$ , and $K$

Table 6: Table of minimal proper supergroups H of axial closed subgroups of $O ( 3 )$ . In the supergroup notation, p denotes a prime number and $p ^ { * }$ denotes an odd prime number.
<table><tr><td>G</td><td>H for General k</td><td>H for Special k</td></tr><tr><td> $C _ { k }$ </td><td> $C _ { p k } , S _ { 2 k } , C _ { k h } , C _ { k v } , D _ { k }$ </td><td> $D _ { p ^ { * } } \ ( k = 2 ) ; T \ ( k = 3 )$ </td></tr><tr><td> $S _ { 2 k }$ </td><td> $S _ { 2 p ^ { * } k } , C _ { 2 k , h } , D _ { k d }$ </td><td> $T _ { h } \left( k = 3 \right)$ </td></tr><tr><td> $C _ { k h }$ </td><td> $C _ { p k , h } , D _ { k h }$ </td><td> $C _ { p v } ( k = \dot { 1 } ) ; D _ { p ^ { * } d } ( k = 2 )$ </td></tr><tr><td> $C _ { k v } \left( k > 1 \right)$ </td><td> $\dot { C _ { p k , v } } , D _ { k h } , D _ { k d }$ </td><td> $T _ { d } ( k = 3 ) ; D _ { p h } ^ { \ast } ( k = 2 )$ </td></tr><tr><td> $D _ { k } \left( k > 1 \right)$ </td><td> $\bar { D _ { p k } } , \bar { D _ { k h } } , \bar { D _ { k d } }$ </td><td> $T \ ( k = 2 ) ; O \ ( k = 3 , 4 ) ; I \ ( k = 3 , 5 )$ </td></tr><tr><td> $D _ { k h } \left( k > 1 \right)$ </td><td> $D _ { p k , h }$ </td><td> $T _ { h } ~ ( k = 2 ) ; O _ { h } ~ ( k = 4 )$ </td></tr><tr><td> $D _ { k d } \left( k > 1 \right)$ </td><td> $D _ { p ^ { * } k , d } , D _ { 2 k , h }$ </td><td> $T _ { d } ( k = 2 ) ; O _ { h } ( k = 3 ) ; I _ { h } ( k = 3 , 5 )$ </td></tr></table>

Table 7: Table of minimal proper supergroups H of other closed subgroups $O ( 3 )$
<table><tr><td>G</td><td>H</td></tr><tr><td> $T$ </td><td> $T _ { d } , T _ { h } , O , I$ </td></tr><tr><td> $T _ { d }$ </td><td> $O _ { h }$ </td></tr><tr><td> $T _ { h }$ </td><td> $O _ { h } , I _ { h }$ </td></tr><tr><td> $O$ </td><td> $O _ { h } , K$ </td></tr><tr><td> $O _ { h }$ </td><td> $K _ { h }$   $I _ { h } , K$ </td></tr><tr><td> $I$   $I _ { h }$ </td><td> $K _ { h }$ </td></tr><tr><td> $C _ { \infty }$ </td><td> $C _ { \infty h } , C _ { \infty , v } , D _ { \infty }$ </td></tr><tr><td></td><td> $D _ { \infty , h }$ </td></tr><tr><td> $C _ { \infty h }$   $C _ { \infty v }$ </td><td> $D _ { \infty , h }$ </td></tr><tr><td></td><td></td></tr><tr><td> $D _ { \infty }$ </td><td> $D _ { \infty , h } , K$ </td></tr><tr><td> $D _ { \infty h }$ </td><td> $K _ { h }$ </td></tr><tr><td> $K$ </td><td> $K _ { h }$ </td></tr></table>

## E.2 DIMENSIONS OF FIXED-POINT SUBSPACES FOR SUBGROUPS OF SO(3) OR O(3)

Dimensions table of fixed-point subspaces for subgroups of SO(3) or O(3). Some of the results can be found from Table B.1 and Table B.2 of Linehan & Stedman (2001). We only present the results for O(3). The discussion for $S O ( 3 )$ can be obtained directly from the table.

Table 8: Dimensions of fixed-point subspaces for closed subgroups of O(3) acting on the irreducible representations $V _ { l = l _ { 0 } ^ { \pm } }$ , where $\bar { a } _ { k } ( l ) = \lfloor \bar { l } / k \rfloor$ and $b _ { k } ( l ) = \lfloor ( \overline { { l } } + \dot { k } ) / ( 2 k ) \rfloor$ .
<table><tr><td rowspan=2 colspan=2>Subgroup</td><td rowspan=1 colspan=2> $l = l _ { 0 } ^ { - }$ </td><td rowspan=1 colspan=2> $l = l _ { 0 } ^ { + }$ </td></tr><tr><td rowspan=1 colspan=1> $l _ { 0 }$ even</td><td rowspan=1 colspan=1> $l _ { 0 }$ odd</td><td rowspan=1 colspan=1> $l _ { 0 }$ even</td><td rowspan=1 colspan=1> $l _ { \mathrm { 0 } } ~ \mathrm { o d d }$ </td></tr><tr><td rowspan=1 colspan=1> $C _ { k }$ </td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=4> $2 a _ { k } ( l _ { 0 } ) + 1$ </td></tr><tr><td rowspan=2 colspan=1> $S _ { 2 k }$ </td><td rowspan=1 colspan=1>k even</td><td rowspan=1 colspan=2> $2 b _ { k } ( l _ { 0 } )$ </td><td rowspan=1 colspan=2> $2 a _ { 2 k } ( l _ { 0 } ) + 1$ </td></tr><tr><td rowspan=1 colspan=1>k odd</td><td rowspan=1 colspan=2>0</td><td rowspan=1 colspan=2> $2 a _ { k } ( l _ { 0 } ) + 1$ </td></tr><tr><td rowspan=2 colspan=1> $C _ { k h }$ </td><td rowspan=1 colspan=1>k even</td><td rowspan=1 colspan=2>0</td><td rowspan=1 colspan=2> $2 a _ { k } ( l _ { 0 } ) + 1$ </td></tr><tr><td rowspan=1 colspan=1>k odd</td><td rowspan=1 colspan=2> $2 b _ { k } ( l _ { 0 } )$ </td><td rowspan=1 colspan=2> $2 a _ { 2 k } ( l _ { 0 } ) + 1$ </td></tr><tr><td rowspan=1 colspan=1> $C _ { k v }$ </td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1> $a _ { k } ( l _ { 0 } )$ </td><td rowspan=1 colspan=1> $a _ { k } ( l _ { 0 } ) + 1$ </td><td rowspan=1 colspan=1> $a _ { k } ( l _ { 0 } ) + 1$ </td><td rowspan=1 colspan=1> $a _ { k } ( l _ { 0 } )$ </td></tr><tr><td rowspan=1 colspan=1> $D _ { k }$ </td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1> $a _ { k } ( l _ { 0 } ) + 1$ </td><td rowspan=1 colspan=1> $a _ { k } ( l _ { 0 } )$ </td><td rowspan=1 colspan=1> $a _ { k } ( l _ { 0 } ) + 1$ </td><td rowspan=1 colspan=1> $a _ { k } ( l _ { 0 } )$ </td></tr><tr><td rowspan=2 colspan=1> $D _ { k h }$ </td><td rowspan=1 colspan=1>k even</td><td rowspan=1 colspan=2>0</td><td rowspan=1 colspan=1> $a _ { k } ( l _ { 0 } ) + 1$ </td><td rowspan=1 colspan=1> $a _ { k } ( l _ { 0 } )$ </td></tr><tr><td rowspan=1 colspan=1>k odd</td><td rowspan=1 colspan=2> $b _ { k } ( l _ { 0 } )$ </td><td rowspan=1 colspan=1> $a _ { 2 k } ( l _ { 0 } ) + 1$ </td><td rowspan=1 colspan=1> $a _ { 2 k } ( l _ { 0 } )$ </td></tr><tr><td rowspan=2 colspan=1> $D _ { k d }$ </td><td rowspan=1 colspan=1>k even</td><td rowspan=1 colspan=2> $b _ { k } ( l _ { 0 } )$ </td><td rowspan=1 colspan=1> $a _ { 2 k } ( l _ { 0 } ) + 1$ </td><td rowspan=1 colspan=1> $a _ { 2 k } ( l _ { 0 } )$ </td></tr><tr><td rowspan=1 colspan=1>k odd</td><td rowspan=1 colspan=2>0</td><td rowspan=1 colspan=1> $a _ { k } ( l _ { 0 } ) + 1$ </td><td rowspan=1 colspan=1> $a _ { k } ( l _ { 0 } )$ </td></tr><tr><td rowspan=1 colspan=1> $T$ </td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=4> $2 a _ { 3 } ( l _ { 0 } ) + a _ { 2 } ( l _ { 0 } ) - l _ { 0 } + 1$ </td></tr><tr><td rowspan=1 colspan=1> $T _ { h }$ </td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>0</td><td rowspan=1 colspan=2> $2 a _ { 3 } ( l _ { 0 } ) + a _ { 2 } ( l _ { 0 } ) - l _ { 0 } + 1$ </td></tr><tr><td rowspan=1 colspan=1> $T _ { d }$ </td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2> $a _ { 3 } ( l _ { 0 } ) + b _ { 2 } ( l _ { 0 } ) + b _ { 1 } ( l _ { 0 } ) - l _ { 0 }$ </td><td rowspan=1 colspan=2> $a _ { 4 } ( l _ { 0 } ) + a _ { 3 } ( l _ { 0 } ) + a _ { 2 } ( l _ { 0 } ) - l _ { 0 } + 1$ </td></tr><tr><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2> $a _ { 4 } ( l _ { 0 } ) + a _ { 3 } ( l</td><td rowspan=1 colspan=2>_ { 0 } ) + a _ { 2 } ( l _ { 0 } ) - l _ { 0 } + 1$ </td></tr><tr><td rowspan=1 colspan=1> $O _ { h }$ </td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>0</td><td rowspan=1 colspan=2> $a _ { 4 } ( l _ { 0 } ) + a _ { 3 } ( l _ { 0 } ) + a _ { 2 } ( l _ { 0 } ) - l _ { 0 } + 1$ </td></tr><tr><td rowspan=1 colspan=1>I</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2> $a _ { 5 } ( l _ { 0 } ) + a _ { 3 } ( l</td><td rowspan=1 colspan=2>_ { 0 } ) + a _ { 2 } ( l _ { 0 } ) - l _ { 0 } + 1$ </td></tr><tr><td rowspan=1 colspan=1> $I _ { h }$ </td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>0</td><td rowspan=1 colspan=2> $a _ { 5 } ( l _ { 0 } ) + a _ { 3 } ( l _ { 0 } ) + a _ { 2 } ( l _ { 0 } ) - l _ { 0 } + 1$ </td></tr><tr><td rowspan=1 colspan=1> $C _ { \infty }$ </td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=2>1</td></tr><tr><td rowspan=1 colspan=1> $C _ { \infty h }$ </td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>0</td><td rowspan=1 colspan=2>1</td></tr><tr><td rowspan=1 colspan=1> $C _ { \infty v }$ </td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>0</td></tr><tr><td rowspan=1 colspan=1> $D _ { \infty }$ </td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>0</td></tr><tr><td rowspan=1 colspan=1> $D _ { \infty h }$ </td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>0</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>0</td></tr></table>

## E.3 SYMMETRY INFIMUM FOR SUBGROUPS OF $S O ( 3 )$

Symmetry infimum table for subgroups of $S O ( 3 )$ . In the following tables, we use a color-coding scheme to classify the types of symmetry increase. An increase to the full group is termed full degeneration and is marked in red . An increase to a supergroup of a strictly higher dimension is termed continuous degeneration and is marked in blue . An increase to a supergroup of the same dimension is termed discrete degeneration, and is marked in yellow . No increase is termed no degeneration, and is marked in green . All subgroups increase to to $K$ when $l = 0$

Table 9: Symmetry infimum of general axis subgroup of SO(3) on $V _ { l = l _ { 0 } } ^ { \oplus r } , r > 3 , l _ { 0 } > 0 .$
<table><tr><td rowspan=2 colspan=1></td><td rowspan=1 colspan=2> $l _ { 0 } < k$ </td><td rowspan=2 colspan=1> $l _ { 0 } \geq k$ </td></tr><tr><td rowspan=1 colspan=1> $l _ { 0 }$ even</td><td rowspan=1 colspan=1> $l _ { 0 }$ odd</td></tr><tr><td rowspan=1 colspan=1> $C _ { k }$ </td><td rowspan=1 colspan=1> $D _ { \infty }$ </td><td rowspan=1 colspan=1> $C _ { \infty }$ </td><td rowspan=1 colspan=1> $C _ { k }$ </td></tr><tr><td rowspan=1 colspan=1> $D _ { k } ( k > 2 )$ </td><td rowspan=1 colspan=1> $D _ { \infty }$ </td><td rowspan=1 colspan=1>K</td><td rowspan=1 colspan=1> $D _ { k }$ </td></tr></table>

Table 10: Symmetry infimum of special axis subgroup of $S O ( 3 )$ on $V _ { l = l _ { 0 } } ^ { \oplus r } , r > 3 , l _ { 0 } > 0$
<table><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1> $l _ { 0 } = 1$ </td><td rowspan=1 colspan=1> $l _ { 0 } = 2$ </td><td rowspan=1 colspan=1> $l _ { 0 } = 3$ </td><td rowspan=1 colspan=1> $l _ { 0 } \geq 4$ </td></tr><tr><td rowspan=1 colspan=1> $D _ { 2 }$ </td><td rowspan=1 colspan=1>K</td><td rowspan=1 colspan=1> $D _ { 2 }$ </td><td rowspan=1 colspan=1>T</td><td rowspan=1 colspan=1> $D _ { 2 }$ </td></tr></table>

Table 11: Symmetry infimum of polyhedral subgroup of $S O ( 3 )$ on $V _ { l = l _ { 0 } } ^ { \oplus r } , r > 3 , l _ { 0 } > 0$ . The "Caption" in the table takes the value $l _ { 0 } = 6 ,$ 10, 12, 15, 16, 18, 20 22, 24 28.
<table><tr><td rowspan="2"> $T$ </td><td> $l _ { 0 } = 6 , 9 , 1 0$ </td><td> $l _ { 0 } = 3 , 7$ </td><td> $l _ { 0 } = 4 , 8$ </td><td> $l _ { 0 } \geq 1 2$ </td><td>other</td></tr><tr><td>T</td><td>T</td><td>O</td><td> $T$  </td><td>K</td></tr><tr><td rowspan="2"> $O$ </td><td> $l _ { 0 } = 4 , 6 , 8 , 9 , 1 0$ </td><td> $l _ { 0 } \geq 1 2$ </td><td></td><td>other</td><td></td></tr><tr><td>0</td><td>0</td><td></td><td>K</td><td></td></tr><tr><td rowspan="2">I</td><td>Caption</td><td> $l _ { 0 } \geq 3 0$ </td><td></td><td>other</td><td></td></tr><tr><td>I</td><td>I</td><td></td><td>K</td><td></td></tr></table>

Table 12: Symmetry infimum of infinite subgroup of $S O ( 3 )$ on $V _ { l = l _ { 0 } } ^ { \oplus r } , r > 3 , l _ { 0 } > 0$
<table><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1> $l _ { 0 }$ even</td><td rowspan=1 colspan=1> $l _ { 0 }$ odd</td></tr><tr><td rowspan=1 colspan=1> $C _ { \infty }$ </td><td rowspan=1 colspan=1> $D _ { \infty }$ </td><td rowspan=1 colspan=1> $C _ { \infty }$ </td></tr><tr><td rowspan=1 colspan=1> $D _ { \infty }$ </td><td rowspan=1 colspan=1> $D _ { \infty }$ </td><td rowspan=1 colspan=1>K</td></tr><tr><td rowspan=1 colspan=1> $K$ </td><td rowspan=1 colspan=1> $K$ </td><td rowspan=1 colspan=1>K</td></tr></table>

## E.4 SYMMETRY INFIMUM FOR SUBGROUPS OF O(3)

Symmetry Infimum table for Subgroups of O(3). The color scheme for full, continuous, and no degeneration is the same as for SO(3). For cases of discrete degeneration of $l _ { 0 } = l ^ { + }$ , we distinguish between (a) light Green , which indicates the predictable increase to $\pi ^ { - 1 } ( \pi ( H ) )$ for the projection map $\pi : O ( 3 )  S O ( 3 )$ , and (b) yellow , which indicates all other exceptional cases of discrete degeneration. In the following table, the first type of subgroups degenerate to K and the others to $K _ { h }$ when $l _ { 0 } = 0 ^ { - }$ . All subgroups increase to to $\dot { K _ { h } }$ when $\bar { l _ { 0 } } = \bar { 0 } ^ { + }$

Table 13: Symmetry infimum of general axis subgroup of O(3) on $V _ { l = l _ { 0 } ^ { - } } ^ { \oplus r } , r > 3 , l _ { 0 } > 0$
<table><tr><td rowspan=2 colspan=2>Subgroup</td><td rowspan=1 colspan=2> $0 < l _ { 0 } < k$ </td><td rowspan=1 colspan=2> $k \leq l _ { 0 } < 2 k$ </td><td rowspan=1 colspan=2> $l _ { 0 } \geq 2 k$ </td></tr><tr><td rowspan=1 colspan=1> $l _ { 0 }$ even</td><td rowspan=1 colspan=1> $l _ { 0 }$ odd</td><td rowspan=1 colspan=1> $l _ { 0 }$ even</td><td rowspan=1 colspan=1> $l _ { 0 }$ odd</td><td rowspan=1 colspan=1> $l _ { 0 }$ even</td><td rowspan=1 colspan=1> $l _ { 0 }$ odd</td></tr><tr><td rowspan=2 colspan=1> $C _ { k }$ </td><td rowspan=1 colspan=1>k even</td><td rowspan=1 colspan=1> $D _ { \infty }$ </td><td rowspan=1 colspan=1> $C _ { \infty v }$ </td><td rowspan=1 colspan=1> $C _ { k }$ </td><td rowspan=1 colspan=1> $C _ { k }$ </td><td rowspan=1 colspan=1> $C _ { k }$ </td><td rowspan=1 colspan=1> $C _ { k }$ </td></tr><tr><td rowspan=1 colspan=1>k odd</td><td rowspan=1 colspan=1> $D _ { \infty }$ </td><td rowspan=1 colspan=1> $C _ { \infty v }$ </td><td rowspan=1 colspan=1> $C _ { k }$ </td><td rowspan=1 colspan=1> $C _ { k }$ </td><td rowspan=1 colspan=1> $C _ { k }$ </td><td rowspan=1 colspan=1> $C _ { k }$ </td></tr><tr><td rowspan=2 colspan=1> $S _ { 2 k }$ </td><td rowspan=1 colspan=1>k even</td><td rowspan=1 colspan=1> $K _ { h }$ </td><td rowspan=1 colspan=1> $K _ { h }$ </td><td rowspan=1 colspan=1> $S _ { 2 k }$ </td><td rowspan=1 colspan=1> $S _ { 2 k }$ </td><td rowspan=1 colspan=1> $S _ { 2 k }$ </td><td rowspan=1 colspan=1> $S _ { 2 k }$ </td></tr><tr><td rowspan=1 colspan=1>k odd</td><td rowspan=1 colspan=1> $K _ { h }$ </td><td rowspan=1 colspan=1> $K _ { h }$ </td><td rowspan=1 colspan=1> $K _ { h }$ </td><td rowspan=1 colspan=1> $K _ { h }$ </td><td rowspan=1 colspan=1> $K _ { h }$ </td><td rowspan=1 colspan=1> $K _ { h }$ </td></tr><tr><td rowspan=2 colspan=1> $C _ { k h }$ </td><td rowspan=1 colspan=1>k even</td><td rowspan=1 colspan=1> $K _ { h }$ </td><td rowspan=1 colspan=1> $K _ { h }$ </td><td rowspan=1 colspan=1> $K _ { h }$ </td><td rowspan=1 colspan=1> $K _ { h }$ </td><td rowspan=1 colspan=1> $K _ { h }$ </td><td rowspan=1 colspan=1> $K _ { h }$ </td></tr><tr><td rowspan=1 colspan=1>k odd</td><td rowspan=1 colspan=1> $K _ { h }$ </td><td rowspan=1 colspan=1> $K _ { h }$ </td><td rowspan=1 colspan=1> $C _ { k h }$ </td><td rowspan=1 colspan=1> $C _ { k h }$ </td><td rowspan=1 colspan=1> $C _ { k h }$ </td><td rowspan=1 colspan=1> $C _ { k h }$ </td></tr><tr><td rowspan=2 colspan=1> $C _ { k v } ( k > 2 )$ </td><td rowspan=1 colspan=1>k even</td><td rowspan=2 colspan=1>Kh $K _ { h }$ </td><td rowspan=2 colspan=1> $C _ { \infty v }$  $C _ { \infty v }$ </td><td rowspan=2 colspan=1> $D _ { k d }$  $D _ { k h }$ </td><td rowspan=2 colspan=1> $C _ { k v }$  $C _ { k v }$ </td><td rowspan=2 colspan=1> $C _ { k v }$  $C _ { k v }$ </td><td rowspan=2 colspan=1> $C _ { k v }$  $C _ { k v }$ </td></tr><tr><td rowspan=1 colspan=1>k odd</td></tr><tr><td rowspan=2 colspan=1> $D _ { k } ( k > 2 )$ </td><td rowspan=1 colspan=1>k even</td><td rowspan=1 colspan=1> $D _ { \infty }$ </td><td rowspan=1 colspan=1> $K _ { h }$ </td><td rowspan=1 colspan=1> $D _ { k }$ </td><td rowspan=1 colspan=1> $D _ { k d }$ </td><td rowspan=1 colspan=1> $D _ { k }$ </td><td rowspan=1 colspan=1> $D _ { k }$ </td></tr><tr><td rowspan=1 colspan=1>k odd</td><td rowspan=1 colspan=1> $D _ { \infty }$ </td><td rowspan=1 colspan=1> $K _ { h }$ </td><td rowspan=1 colspan=1> $D _ { k }$ </td><td rowspan=1 colspan=1> $D _ { k h }$ </td><td rowspan=1 colspan=1> $D _ { k }$ </td><td rowspan=1 colspan=1> $D _ { k }$ </td></tr><tr><td rowspan=2 colspan=1> $D _ { k h } ( k > 2 )$ </td><td rowspan=1 colspan=1>k even</td><td rowspan=2 colspan=1> $K _ { h }$  $K _ { h }$ </td><td rowspan=2 colspan=1> $K _ { h }$  $K _ { h }$ </td><td rowspan=1 colspan=1> $K _ { h }$ </td><td rowspan=2 colspan=1> $K _ { h }$  $D _ { k h }$ </td><td rowspan=2 colspan=1> $K _ { h }$  $D _ { k h }$ </td><td rowspan=2 colspan=1> $K _ { h }$  $D _ { k h }$ </td></tr><tr><td rowspan=1 colspan=1>k odd</td><td rowspan=1 colspan=1> $D _ { k h }$ </td></tr><tr><td rowspan=2 colspan=1> $D _ { k d } ( k > 2 )$ </td><td rowspan=1 colspan=1>k even</td><td rowspan=1 colspan=1> $K _ { h }$ </td><td rowspan=1 colspan=1> $K _ { h }$ </td><td rowspan=1 colspan=1> $D _ { k d }$ </td><td rowspan=1 colspan=1> $D _ { k d }$ </td><td rowspan=1 colspan=1> $D _ { k d }$ </td><td rowspan=1 colspan=1> $D _ { k d }$ </td></tr><tr><td rowspan=1 colspan=1>k odd</td><td rowspan=1 colspan=1> $K _ { h }$ </td><td rowspan=1 colspan=1> $K _ { h }$ </td><td rowspan=1 colspan=1> $K _ { h }$ </td><td rowspan=1 colspan=1> $K _ { h }$ </td><td rowspan=1 colspan=1> $K _ { h }$ </td><td rowspan=1 colspan=1> $K _ { h }$ </td></tr></table>

Table 14: Symmetry infimum of general axis subgroup of $O ( 3 )$ on $V _ { l = l _ { 0 } ^ { + } } ^ { \oplus r } ,$ r > 3, $l _ { 0 } > 0$
<table><tr><td rowspan=2 colspan=2>Subgroup</td><td rowspan=1 colspan=2> $0 < l _ { 0 } < k$ </td><td rowspan=1 colspan=2> $k \leq l _ { 0 } < 2 k$ </td><td rowspan=1 colspan=2> $l _ { 0 } \geq 2 k$ </td></tr><tr><td rowspan=1 colspan=1> $l _ { 0 }$ even</td><td rowspan=1 colspan=1> $l _ { 0 }$ odd</td><td rowspan=1 colspan=1> $l _ { 0 }$ even</td><td rowspan=1 colspan=1> $l _ { 0 }$ odd</td><td rowspan=1 colspan=1> $l _ { 0 }$ even</td><td rowspan=1 colspan=1> $l _ { 0 }$ odd</td></tr><tr><td rowspan=2 colspan=1> $C _ { k }$ </td><td rowspan=1 colspan=1>k even</td><td rowspan=1 colspan=1> $D _ { \infty h }$ </td><td rowspan=1 colspan=1> $C _ { \infty h }$ </td><td rowspan=2 colspan=1> $C _ { k h }$  $S _ { 2 k }$ </td><td rowspan=2 colspan=1> $C _ { k h }$  $S _ { 2 k }$ </td><td rowspan=2 colspan=1> $C _ { k h }$  $S _ { 2 k }$ </td><td rowspan=2 colspan=1> $C _ { k h }$  $S _ { 2 k }$ </td></tr><tr><td rowspan=1 colspan=1>k odd</td><td rowspan=1 colspan=1> $D _ { \infty h }$ </td><td rowspan=1 colspan=1> $C _ { \infty h }$ </td></tr><tr><td rowspan=2 colspan=1> $S _ { 2 k }$ </td><td rowspan=1 colspan=1>k even</td><td rowspan=1 colspan=1> $D _ { \infty h }$ </td><td rowspan=1 colspan=1> $C _ { \infty h }$ </td><td rowspan=1 colspan=1> $D _ { \infty h }$ </td><td rowspan=1 colspan=1> $C _ { \infty h }$ </td><td rowspan=1 colspan=1> $C _ { 2 k h }$ </td><td rowspan=1 colspan=1> $C _ { 2 k h }$ </td></tr><tr><td rowspan=1 colspan=1>k odd</td><td rowspan=1 colspan=1> $D _ { \infty h }$ </td><td rowspan=1 colspan=1> $C _ { \infty h }$ </td><td rowspan=1 colspan=1> $S _ { 2 k }$ </td><td rowspan=1 colspan=1> $S _ { 2 k }$ </td><td rowspan=1 colspan=1> $S _ { 2 k }$ </td><td rowspan=1 colspan=1> $S _ { 2 k }$ </td></tr><tr><td rowspan=2 colspan=1> $C _ { k h }$ </td><td rowspan=1 colspan=1>k even</td><td rowspan=1 colspan=1> $D _ { \infty h }$ </td><td rowspan=1 colspan=1> $C _ { \infty h }$ </td><td rowspan=1 colspan=1> $C _ { k h }$ </td><td rowspan=1 colspan=1> $C _ { k h }$ </td><td rowspan=1 colspan=1> $C _ { k h }$ </td><td rowspan=1 colspan=1> $C _ { k h }$ </td></tr><tr><td rowspan=1 colspan=1>k odd</td><td rowspan=1 colspan=1> $D _ { \infty h }$ </td><td rowspan=1 colspan=1> $C _ { \infty h }$ </td><td rowspan=1 colspan=1> $D _ { \infty h }$ </td><td rowspan=1 colspan=1> $C _ { \infty h }$ </td><td rowspan=1 colspan=1> $C _ { 2 k h }$ </td><td rowspan=1 colspan=1> $C _ { 2 k h }$ </td></tr><tr><td rowspan=2 colspan=1> $C _ { k v } ( k > 2 )$ </td><td rowspan=1 colspan=1>k even</td><td rowspan=1 colspan=1> $D _ { \infty h }$ </td><td rowspan=1 colspan=1> $K _ { h }$ </td><td rowspan=1 colspan=1> $D _ { k h }$ </td><td rowspan=1 colspan=1> $D _ { k h }$ </td><td rowspan=1 colspan=1> $D _ { k h }$ </td><td rowspan=1 colspan=1> $D _ { k h }$ </td></tr><tr><td rowspan=1 colspan=1>k odd</td><td rowspan=1 colspan=1> $D _ { \infty h }$ </td><td rowspan=1 colspan=1> $K _ { h }$ </td><td rowspan=1 colspan=1> $D _ { k d }$ </td><td rowspan=1 colspan=1> $D _ { k d }$ </td><td rowspan=1 colspan=1> $D _ { k d }$ </td><td rowspan=1 colspan=1> $D _ { k d }$ </td></tr><tr><td rowspan=2 colspan=1> $D _ { k } ( k > 2 )$ </td><td rowspan=1 colspan=1>k even</td><td rowspan=1 colspan=1> $D _ { \infty h }$ </td><td rowspan=2 colspan=1> $K _ { h }$  $K _ { h }$ </td><td rowspan=2 colspan=1> $D _ { k h }$  $D _ { k d }$ </td><td rowspan=2 colspan=1> $D _ { k h }$  $D _ { k d }$ </td><td rowspan=2 colspan=1> $D _ { k h }$  $D _ { k d }$ </td><td rowspan=2 colspan=1> $D _ { k h }$  $D _ { k d }$ </td></tr><tr><td rowspan=1 colspan=1>k odd</td><td rowspan=1 colspan=1> $D _ { \infty h }$ </td></tr><tr><td rowspan=2 colspan=1> $D _ { k h } ( k > 2 )$ </td><td rowspan=1 colspan=1>k even</td><td rowspan=1 colspan=1> $D _ { \infty h }$ </td><td rowspan=1 colspan=1> $K _ { h }$ </td><td rowspan=1 colspan=1> $D _ { k h }$ </td><td rowspan=1 colspan=1> $D _ { k h }$ </td><td rowspan=1 colspan=1> $D _ { k h }$ </td><td rowspan=1 colspan=1> $D _ { k h }$ </td></tr><tr><td rowspan=1 colspan=1>k odd</td><td rowspan=1 colspan=1> $D _ { \infty h }$ </td><td rowspan=1 colspan=1> $K _ { h }$ </td><td rowspan=1 colspan=1> $D _ { \infty h }$ </td><td rowspan=1 colspan=1> $K _ { h }$ </td><td rowspan=1 colspan=1> $D _ { 2 k h }$ </td><td rowspan=1 colspan=1> $D _ { 2 k h }$ </td></tr><tr><td rowspan=2 colspan=1> $D _ { k d } ( k > 2 )$ </td><td rowspan=1 colspan=1>k even</td><td rowspan=1 colspan=1> $D _ { \infty h }$ </td><td rowspan=1 colspan=1> $K _ { h }$ </td><td rowspan=1 colspan=1> $D _ { \infty h }$ </td><td rowspan=1 colspan=1> $K _ { h }$ </td><td rowspan=1 colspan=1> $D _ { 2 k h }$ </td><td rowspan=1 colspan=1> $D _ { 2 k h }$ </td></tr><tr><td rowspan=1 colspan=1>k odd</td><td rowspan=1 colspan=1> $D _ { \infty h }$ </td><td rowspan=1 colspan=1> $K _ { h }$ </td><td rowspan=1 colspan=1> $D _ { k d }$ </td><td rowspan=1 colspan=1> $D _ { k d }$ </td><td rowspan=1 colspan=1> $D _ { k d }$ </td><td rowspan=1 colspan=1> $D _ { k d }$ </td></tr></table>

Table 15: Symmetry infimum of special axis subgroup of O(3) on $V _ { l = l _ { \mathrm { o } } ^ { - } } ^ { \oplus r } , r > 3 , l _ { 0 } > 0$
<table><tr><td rowspan=2 colspan=1></td><td rowspan=2 colspan=1> $l _ { 0 } = 1$ </td><td rowspan=2 colspan=1> $l _ { 0 } = 2$ </td><td rowspan=2 colspan=1> $l _ { 0 } = 3$ </td><td rowspan=1 colspan=2> $l _ { 0 } \geq 4$ </td></tr><tr><td rowspan=1 colspan=1> $l _ { 0 }$ even</td><td rowspan=1 colspan=1> $l _ { 0 }$ odd</td></tr><tr><td rowspan=1 colspan=1> $C _ { 2 v }$ </td><td rowspan=1 colspan=1> $C _ { \infty v }$ </td><td rowspan=1 colspan=1> $D _ { 2 d }$ </td><td rowspan=1 colspan=1> $C _ { 2 v }$ </td><td rowspan=1 colspan=1> $C _ { 2 v }$ </td><td rowspan=1 colspan=1> $C _ { 2 v }$ </td></tr><tr><td rowspan=1 colspan=1> $D _ { 2 }$ </td><td rowspan=1 colspan=1> $K _ { h }$ </td><td rowspan=1 colspan=1> $D _ { 2 }$ </td><td rowspan=1 colspan=1> $T _ { d }$ </td><td rowspan=1 colspan=1> $D _ { 2 }$ </td><td rowspan=1 colspan=1> $D _ { 2 }$ </td></tr><tr><td rowspan=1 colspan=1> $D _ { 2 h }$ </td><td rowspan=1 colspan=1> $K _ { h }$ </td><td rowspan=1 colspan=1> $K _ { h }$ </td><td rowspan=1 colspan=1> $K _ { h }$ </td><td rowspan=1 colspan=1> $K _ { h }$ </td><td rowspan=1 colspan=1> $K _ { h }$ </td></tr><tr><td rowspan=1 colspan=1> $D _ { 2 d }$ </td><td rowspan=1 colspan=1> $K _ { h }$ </td><td rowspan=1 colspan=1> $D _ { 2 d }$ </td><td rowspan=1 colspan=1> $T _ { d }$ </td><td rowspan=1 colspan=1> $D _ { 2 d }$ </td><td rowspan=1 colspan=1> $D _ { 2 d }$ </td></tr></table>

Table 16: Symmetry infimum of special axis subgroup of $O ( 3 )$ on $V _ { l = l _ { 0 } ^ { + } } ^ { \oplus r } , r > 3 , l _ { 0 } > 0$
<table><tr><td rowspan=2 colspan=1></td><td rowspan=2 colspan=1> $l _ { 0 } = 1$ </td><td rowspan=2 colspan=1> $l _ { 0 } = 2$ </td><td rowspan=2 colspan=1> $l _ { 0 } = 3$ </td><td rowspan=1 colspan=2> $l _ { 0 } \geq 4$ </td></tr><tr><td rowspan=1 colspan=1> $l _ { 0 }$ even</td><td rowspan=1 colspan=1> $l _ { 0 }$ odd</td></tr><tr><td rowspan=1 colspan=1> $C _ { 2 v }$ </td><td rowspan=1 colspan=1> $K _ { h }$ </td><td rowspan=1 colspan=1> $D _ { 2 h }$ </td><td rowspan=1 colspan=1> $T _ { h }$ </td><td rowspan=1 colspan=1> $D _ { 2 h }$ </td><td rowspan=1 colspan=1> $D _ { 2 h }$ </td></tr><tr><td rowspan=1 colspan=1> $D _ { 2 }$ </td><td rowspan=1 colspan=1> $K _ { h }$ </td><td rowspan=1 colspan=1> $D _ { 2 h }$ </td><td rowspan=1 colspan=1> $T _ { h }$ </td><td rowspan=1 colspan=1> $D _ { 2 h }$ </td><td rowspan=1 colspan=1> $D _ { 2 h }$ </td></tr><tr><td rowspan=1 colspan=1> $D _ { 2 h }$ </td><td rowspan=1 colspan=1> $K _ { h }$ </td><td rowspan=1 colspan=1> $D _ { 2 h }$ </td><td rowspan=1 colspan=1> $T _ { h }$ </td><td rowspan=1 colspan=1> $D _ { 2 h }$ </td><td rowspan=1 colspan=1> $D _ { 2 h }$ </td></tr><tr><td rowspan=1 colspan=1> $D _ { 2 d }$ </td><td rowspan=1 colspan=1> $K _ { h }$ </td><td rowspan=1 colspan=1> $D _ { \infty h }$ </td><td rowspan=1 colspan=1> $K _ { h }$ </td><td rowspan=1 colspan=1> $D _ { 4 h }$ </td><td rowspan=1 colspan=1> $D _ { 4 h }$ </td></tr></table>

Table 17: Symmetry infimum of polyhedral subgroup of $O ( 3 )$ on $V _ { l = l _ { 0 } ^ { - } } ^ { \oplus r } , r > 3 , l _ { 0 } > 0$ . The "Caption" in the table takes the value $l _ { 0 }$ = 6, 10, 12, 15, 16, 18, 20  22, 24  28.
<table><tr><td rowspan=2 colspan=1> $T$ </td><td rowspan=1 colspan=1> $l _ { 0 } = 6 , 9 , 1 0$ </td><td rowspan=1 colspan=1> $l _ { 0 } = 3 , 7$ </td><td rowspan=1 colspan=1> $l _ { 0 } = 4 , 8$ </td><td rowspan=1 colspan=1> $l _ { 0 } \geq 1 2$ </td><td rowspan=1 colspan=1>other</td></tr><tr><td rowspan=1 colspan=1> $T$ </td><td rowspan=1 colspan=1> $T _ { d }$ </td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1> $T$ </td><td rowspan=1 colspan=1> $K _ { h }$ </td></tr><tr><td rowspan=2 colspan=1> $T _ { d }$ </td><td rowspan=1 colspan=1> $l _ { 0 } = 3 , 6 , 7$ </td><td rowspan=1 colspan=1> $l _ { 0 } \geq 9$ </td><td rowspan=1 colspan=3>other</td></tr><tr><td rowspan=1 colspan=1> $T _ { d }$ </td><td rowspan=1 colspan=1> $T _ { d }$ </td><td rowspan=1 colspan=3> $K _ { h }$ </td></tr><tr><td rowspan=1 colspan=1> $T _ { h }$ </td><td rowspan=1 colspan=5>all $l _ { 0 }$  $K _ { h }$ </td></tr><tr><td rowspan=2 colspan=1> $O$ </td><td rowspan=1 colspan=1> $l _ { 0 } = 4 , 6 , 8 , 9 , 1 0$ </td><td rowspan=1 colspan=1> $l _ { 0 } \geq 1 2$ </td><td rowspan=1 colspan=3>other</td></tr><tr><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=3> $K _ { h }$ </td></tr><tr><td rowspan=1 colspan=1> $O _ { h }$ </td><td rowspan=1 colspan=5>all l{0 $K _ { h }$ </td></tr><tr><td rowspan=2 colspan=1> $I$ </td><td rowspan=2 colspan=1>Caption $I _ { h }$ </td><td rowspan=1 colspan=1> $l _ { 0 } \geq 3 0$ </td><td rowspan=1 colspan=3>other</td></tr><tr><td rowspan=1 colspan=1> $I _ { h }$ </td><td rowspan=1 colspan=3> $K _ { h }$ </td></tr><tr><td rowspan=1 colspan=1> $I _ { h }$ </td><td rowspan=1 colspan=5>all $l _ { 0 }$  $K _ { h }$ </td></tr></table>

Table 18: Symmetry infimum of polyhedral subgroup of $O ( 3 )$ on V<sup>⊕r</sup> <sub>l=l+</sub> , r > 3, $l _ { 0 } > 0 .$ .The "Caption" in the table takes the value $l _ { 0 } = 6 ,$ , 10, 12, 15, 16, 18, 20 22, 24 28.
<table><tr><td rowspan=2 colspan=1> $T$ </td><td rowspan=1 colspan=1> $l _ { 0 } = 3 , 6 , 7 , 9 , 1 0$ </td><td rowspan=1 colspan=1> $l _ { 0 } = 4 , 8$ </td><td rowspan=1 colspan=1> $l _ { 0 } \geq 1 2$ </td><td rowspan=1 colspan=1>other</td></tr><tr><td rowspan=1 colspan=1> $T _ { h }$ </td><td rowspan=1 colspan=1> $O _ { h }$ </td><td rowspan=1 colspan=1> $T _ { h }$ </td><td rowspan=1 colspan=1> $K _ { h }$ </td></tr><tr><td rowspan=2 colspan=1> $T _ { d }$ </td><td rowspan=2 colspan=1> $l _ { 0 } = 4 , 6 , 8 , 9 , 1 0$  $O _ { h }$ </td><td rowspan=1 colspan=1> $l _ { 0 } \geq 1 2$ </td><td rowspan=1 colspan=2>other</td></tr><tr><td rowspan=1 colspan=1> $O _ { h }$ </td><td rowspan=1 colspan=2> $K _ { h }$ </td></tr><tr><td rowspan=2 colspan=1> $T _ { h }$ </td><td rowspan=1 colspan=1> $l _ { 0 } = 3 , 6 , 7 , 9 , 1 0$ </td><td rowspan=1 colspan=1> $l _ { 0 } = 4 , 8$ </td><td rowspan=1 colspan=1> $l _ { 0 } \geq 1 2$ </td><td rowspan=1 colspan=1>other</td></tr><tr><td rowspan=1 colspan=1> $T _ { h }$ </td><td rowspan=1 colspan=1> $O _ { h }$ </td><td rowspan=1 colspan=1> $T _ { h }$ </td><td rowspan=1 colspan=1> $K _ { h }$ </td></tr><tr><td rowspan=2 colspan=1> $O$ </td><td rowspan=2 colspan=1> $l _ { 0 } = 4 , 6 , 8 , 9 , 1 0$  $O _ { h }$ </td><td rowspan=1 colspan=1> $l _ { 0 } \geq 1 2$ </td><td rowspan=1 colspan=2>other</td></tr><tr><td rowspan=1 colspan=1> $O _ { h }$ </td><td rowspan=1 colspan=2> $K _ { h }$ </td></tr><tr><td rowspan=2 colspan=1> $O _ { h }$ </td><td rowspan=2 colspan=1> $l _ { 0 } = 4 , 6 , 8 , 9 , 1 0$  $O _ { h }$ </td><td rowspan=1 colspan=1> $l _ { 0 } \geq 1 2$ </td><td rowspan=1 colspan=2>other</td></tr><tr><td rowspan=1 colspan=1> $O _ { h }$ </td><td rowspan=1 colspan=2> $K _ { h }$ </td></tr><tr><td rowspan=2 colspan=1> $I$ </td><td rowspan=2 colspan=1> $\mathrm { C a p t i o n }$  $I _ { h }$ </td><td rowspan=1 colspan=1> $l _ { 0 } \geq 1 2$ </td><td rowspan=1 colspan=2>other</td></tr><tr><td rowspan=1 colspan=1> $I _ { h }$ </td><td rowspan=1 colspan=2> $K _ { h }$ </td></tr><tr><td rowspan=2 colspan=1> $I _ { h }$ </td><td rowspan=1 colspan=1> $\mathrm { C a p t i o n }$ </td><td rowspan=1 colspan=1> $l _ { 0 } \geq 1 2$ </td><td rowspan=1 colspan=2>other</td></tr><tr><td rowspan=1 colspan=1> $I _ { h }$ </td><td rowspan=1 colspan=1> $I _ { h }$ </td><td rowspan=1 colspan=2> $K _ { h }$ </td></tr></table>

Table 19: Symmetry infimum of infinite subgroup of $O ( 3 )$ on $V _ { l = l _ { 0 } ^ { - } } ^ { \oplus r } , r > 3 , l _ { 0 } > 0$
<table><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1> $l _ { 0 }$ even</td><td rowspan=1 colspan=1> $l _ { 0 }$ odd</td></tr><tr><td rowspan=1 colspan=1> $C _ { \infty }$ </td><td rowspan=1 colspan=1> $D _ { \infty }$ </td><td rowspan=1 colspan=1> $C _ { \infty v }$ </td></tr><tr><td rowspan=1 colspan=1> $C _ { \infty h } = S _ { 2 \infty }$ </td><td rowspan=1 colspan=1> $K _ { h }$ </td><td rowspan=1 colspan=1> $K _ { h }$ </td></tr><tr><td rowspan=1 colspan=1> $C _ { \infty v }$ </td><td rowspan=1 colspan=1> $K _ { h }$ </td><td rowspan=1 colspan=1> $C _ { \infty v }$ </td></tr><tr><td rowspan=1 colspan=1> $D _ { \infty }$ </td><td rowspan=1 colspan=1> $D _ { \infty }$ </td><td rowspan=1 colspan=1> $K _ { h }$ </td></tr><tr><td rowspan=1 colspan=1> $D _ { \infty h } = D _ { \infty d }$ </td><td rowspan=1 colspan=1> $K _ { h }$ </td><td rowspan=1 colspan=1> $K _ { h }$ </td></tr><tr><td rowspan=1 colspan=1> $K$ </td><td rowspan=1 colspan=1> $K _ { h }$ </td><td rowspan=1 colspan=1> $K _ { h }$ </td></tr><tr><td rowspan=1 colspan=1> $K _ { h }$ </td><td rowspan=1 colspan=1> $K _ { h }$ </td><td rowspan=1 colspan=1> $K _ { h }$ </td></tr></table>

Table 20: Symmetry infimum of infinite subgroup of $O ( 3 )$ on $V _ { l = l _ { 0 } ^ { + } } ^ { \oplus r } , r > 3 , l _ { 0 } > 0$
<table><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1> $l _ { 0 }$ even</td><td rowspan=1 colspan=1> $l _ { 0 }$ odd</td></tr><tr><td rowspan=1 colspan=1> $C _ { \infty }$ </td><td rowspan=1 colspan=1> $D _ { \infty h }$ </td><td rowspan=1 colspan=1> $C _ { \infty h }$ </td></tr><tr><td rowspan=1 colspan=1> $C _ { \infty h } = S _ { 2 \infty }$ </td><td rowspan=1 colspan=1> $D _ { \infty h }$ </td><td rowspan=1 colspan=1> $C _ { \infty h }$ </td></tr><tr><td rowspan=1 colspan=1> $C _ { \infty v }$ </td><td rowspan=1 colspan=1> $D _ { \infty h }$ </td><td rowspan=1 colspan=1> $K _ { h }$ </td></tr><tr><td rowspan=1 colspan=1> $D _ { \infty }$ </td><td rowspan=1 colspan=1> $D _ { \infty h }$ </td><td rowspan=1 colspan=1> $K _ { h }$ </td></tr><tr><td rowspan=1 colspan=1> $D _ { \infty h } { = } D _ { \infty d }$ </td><td rowspan=1 colspan=1> $D _ { \infty h }$ </td><td rowspan=1 colspan=1> $K _ { h }$ </td></tr><tr><td rowspan=1 colspan=1> $K$ </td><td rowspan=1 colspan=1> $K _ { h }$ </td><td rowspan=1 colspan=1> $K _ { h }$ </td></tr><tr><td rowspan=1 colspan=1> $K _ { h }$ </td><td rowspan=1 colspan=1> $K _ { h }$ </td><td rowspan=1 colspan=1> $K _ { h }$ </td></tr></table>

## F DETAILED EXPERIMENT

Experiments in § 6.1 and 6.2 are conducted with randomly initialized TFN (Thomas et al., 2018) and HEGNN (Cen et al., 2024) architectures as the underlying equivariant models, whereas the experiments in § 6.3 first pretrain a HEGNN encoder to obtain molecular embeddings and then fine-tune separate MLP heads for the final prediction task under different settings.

For TFN (Thomas et al., 2018), we use the implementation from GWL-test (Joshi et al., 2023), which relies on irreducible-representation features with alternating parity. For HEGNN, we extend the original design to a multi-channel variant. In particular, the spherical scalarized features computed for nodes j and i are defined as

$$
z _ { i j , c } ^ { ( l ) } = 1 / \sqrt { C } \cdot \langle \tilde { \pmb { v } } _ { i , c } ^ { ( l ) } , \tilde { \pmb { v } } _ { j , c } ^ { ( l ) } \rangle ,\tag{109}
$$

where $\tilde { v } _ { i , c } ^ { ( l ) }$ and $\tilde { \mathbf { v } } _ { j , c } ^ { ( l ) }$ denote the l-th degree steerable features of channel c among a total of C channels. To obtain the degree $\boldsymbol { - l _ { 0 } }$ graph-level representation, we apply global mean pooling for each graph $\mathcal { G } ( \nu , \mathcal { E } )$

$$
\begin{array} { r } { \tilde { \mathbf { v } } _ { \mathcal { G } , c } ^ { ( l ) } = 1 / | \mathcal { V } | \cdot \sum _ { i } \tilde { \mathbf { v } } _ { i , c } ^ { ( l ) } . } \end{array}\tag{110}
$$

We will detail, in this section, how each experimental task processes these graph-level features.

## F.1 VISUALIZATION OF REPRESENTATION SPACE

We use a TFN with single-layer to calculate the embedding. After that, to extract features of degree $l _ { 0 } ,$ we append an $_ { \odot 3 }$ .Linear layer from e3nn (Geiger & Smidt, 2022), denoting this setup as $\mathrm { T F N } _ { l = l _ { 0 } }$ in our experiments.

## F.2 EXPRESSIVITY ON SYMMETRIC GRAPHS

This section first introduces the detailed settings of § 6.2 in § F.2.1, and then introduces the reproduction of the original experiment of GWL-test (Joshi et al., 2023) in § F.2.2.

## F.2.1 EMBEDDING DIFFERENCE NORM EXPERIMENT

We employed both TFN (Thomas et al., 2018) and HEGNN (Cen et al., 2024) to compute the norm of the embedding difference across 12 configurations for each model, varying the number of channels (1, 4, 16) and layers (1, 2, 3, 4). The degree-l discrepancy between two graphs $\mathcal { G } _ { 0 }$ and $\mathcal { G } _ { 1 }$ is defined as

$$
\begin{array} { r } { \Delta ^ { ( l ) } = 1 / | C | \cdot \sum _ { c = 1 } ^ { C } \| \tilde { \pmb { v } } _ { { \mathcal { G } } _ { 0 } , c } ^ { ( l ) } - \tilde { \pmb { v } } _ { { \mathcal { G } } _ { 1 } , c } ^ { ( l ) } \| . } \end{array}\tag{111}
$$

These choices give rise to $2 \times 3 \times 4 = 2 4$ distinct $\Delta ^ { ( l ) }$ . In Fig. 5, we report the maximum norm across all $\Delta ^ { ( l ) }$ . Since every norm is strictly positive, a maximal value below $1 0 ^ { - 6 }$ indicates that all corresponding norms fall below $1 0 ^ { - 6 }$ , meaning none of them can distinguish $\mathcal { G } _ { 0 }$ from $\mathcal { G } _ { 1 }$

## F.2.2 ORIGINAL GWL-TEST ON SYMMETRIC GRAPHS

Dataset. Same as the setting in (Joshi et al., 2023), we construct four symmetric k-fold structures $( k \in \{ 2 , 3 , 4 , 6 \} )$ ), each centered at the origin. For each structure $\mathcal { G } _ { 0 }$ we apply a random rotation to produce $\mathcal { G } _ { 1 }$ which ensures $\mathcal { G } _ { 1 }$ does not coincide with the original $\mathcal { G } _ { 0 } .$ . The goal is to evaluate whether different equivariant neural network architectures can distinguish $\mathcal { G } _ { 0 }$ from $\mathcal { G } _ { 1 }$ . To validate distinct aspects of our theory, we consider rotations separately in 2D and 3D; in the 3D experiments we additionally ensure that $\mathcal { G } _ { 1 }$ is not coplanar with $\mathcal { G } _ { 0 }$

Embeddings. The extracted $l _ { 0 }$ -degree embeddings from TFN are fed into a vanilla classifier for the classification task. The model was trained for 300 epochs to ensure the classifier had sufficient capacity to discriminate the classes.

Results. The detailed experimental results are presented in Table 21, which can be observed that the color blocks in this table are completely consistent with Fig. 5. It demonstrates that our theoretical predictions in Table 1 are in complete agreement with the empirical findings obtained from the model.

Such remarkable consistency not only confirms the correctness of our theoretical analysis, but also highlights the importance of constructing mappings with appropriately chosen features.

Table 21: Results of distinguishing k-fold structures rotated in 2D/3D space.
<table><tr><td colspan="5">2D Rotational Symmetry</td><td colspan="4">3D Rotational Symmetry</td></tr><tr><td>GNN Layer</td><td>2 fold</td><td>3 fold</td><td>4 fold</td><td>6 fold</td><td>2 fold</td><td>3 fold</td><td>4 fold</td><td>6 fold</td></tr><tr><td> $\mathrm { T F N } _ { l = 0 }$ </td><td> $5 0 . 0 \pm 0 . 0$ </td><td> $5 0 . 0 \pm 0 . 0$ </td><td> $5 0 . 0 \pm 0 . 0$ </td><td> $5 0 . 0 \pm 0 . 0$ </td><td> $5 0 . 0 \pm 0 . 0$ </td><td> $5 0 . 0 \pm 0 . 0$ </td><td> $5 0 . 0 \pm 0 . 0$ </td><td> $5 0 . 0 \pm 0 . 0$ </td></tr><tr><td>TFNl=1</td><td>50.0 ± 0.0</td><td> $5 0 . 0 \pm 0 . 0$ </td><td> $5 0 . 0 \pm 0 . 0$ </td><td> $5 0 . 0 \pm 0 . 0$ </td><td> $5 0 . 0 \pm 0 . 0$ </td><td> $5 0 . 0 \pm 0 . 0$ </td><td> $5 0 . 0 \pm 0 . 0$ </td><td>50.0 ± 0.0</td></tr><tr><td>TFNl=2</td><td> $\mathbf { 1 0 0 . 0 \pm 0 . 0 }$ </td><td> $5 0 . 0 \pm 0 . 0$ </td><td> $5 0 . 0 \pm 0 . 0$ </td><td> $5 0 . 0 \pm 0 . 0$ </td><td> $\mathbf { 1 0 0 . 0 \pm 0 . 0 }$ </td><td> $\mathbf { 1 0 0 . 0 \pm 0 . 0 }$ </td><td> $\mathbf { 1 0 0 . 0 \pm 0 . 0 }$ </td><td> $\mathbf { 1 0 0 . 0 \pm 0 . 0 }$ </td></tr><tr><td>TFNl=3</td><td> $5 0 . 0 \pm 0 . 0$ </td><td>100.0 ± 0.0</td><td> $5 0 . 0 \pm 0 . 0$ </td><td> $5 0 . 0 \pm 0 . 0$ </td><td> $5 0 . 0 \pm 0 . 0$ </td><td> $\mathbf { 1 0 0 . 0 \pm 0 . 0 }$ </td><td> $5 0 . 0 \pm 0 . 0$ </td><td> $5 0 . 0 \pm 0 . 0$ </td></tr><tr><td> $\mathrm { T F N } _ { l = 4 }$ </td><td> $\mathbf { 1 0 0 . 0 \pm 0 . 0 }$ </td><td> $5 0 . 0 \pm 0 . 0$ </td><td> $\mathbf { 1 0 0 . 0 \pm 0 . 0 }$ </td><td> $5 0 . 0 \pm 0 . 0$ </td><td> $\mathbf { 1 0 0 . 0 \pm 0 . 0 }$ </td><td> $\mathbf { 1 0 0 . 0 \pm 0 . 0 }$ </td><td> $\mathbf { 1 0 0 . 0 \pm 0 . 0 }$ </td><td> $\mathbf { 1 0 0 . 0 \pm 0 . 0 }$ </td></tr><tr><td> $\mathrm { T F N } _ { l = 5 }$ </td><td> $5 0 . 0 \pm 0 . 0$ </td><td> $\mathbf { 1 0 0 . 0 \pm 0 . 0 }$ </td><td> $5 0 . 0 \pm 0 . 0$ </td><td>50.0 ± 0.0</td><td> $5 0 . 0 \pm 0 . 0$ </td><td> $\mathbf { 1 0 0 . 0 \pm 0 . 0 }$ </td><td> $5 0 . 0 \pm 0 . 0$ </td><td> $5 0 . 0 \pm 0 . 0$ </td></tr><tr><td> $\mathrm { T F N } _ { l = 6 }$ </td><td> $\mathbf { 1 0 0 . 0 \pm 0 . 0 }$ </td><td> $\mathbf { 1 0 0 . 0 \pm 0 . 0 }$ </td><td> $\mathbf { 1 0 0 . 0 \pm 0 . 0 }$ </td><td>100.0 ± 0.0</td><td> $\mathbf { 1 0 0 . 0 \pm 0 . 0 }$ </td><td> $\mathbf { 1 0 0 . 0 \pm 0 . 0 }$ </td><td> $\mathbf { 1 0 0 . 0 \pm 0 . 0 }$ </td><td> $\mathbf { 1 0 0 . 0 \pm 0 . 0 }$ </td></tr><tr><td> $\mathrm { T F N } _ { l = 7 }$ </td><td> $5 0 . 0 \pm 0 . 0$ </td><td> $\mathbf { 1 0 0 . 0 \pm 0 . 0 }$ </td><td> $5 0 . 0 \pm 0 . 0$ </td><td> $5 0 . 0 \pm 0 . 0$ </td><td> $5 0 . 0 \pm 0 . 0$ </td><td> $\mathbf { 1 0 0 . 0 \pm 0 . 0 }$ </td><td> $5 0 . 0 \pm 0 . 0$ </td><td> $5 0 . 0 \pm 0 . 0$ </td></tr><tr><td> $\mathrm { T F N } _ { l = 8 }$ </td><td> $\mathbf { 1 0 0 . 0 \pm 0 . 0 }$ </td><td> $\mathbf { 1 0 0 . 0 \pm 0 . 0 }$ </td><td> $\mathbf { 1 0 0 . 0 \pm 0 . 0 }$ </td><td> $\mathbf { 1 0 0 . 0 \pm 0 . 0 }$ </td><td> $\mathbf { 1 0 0 . 0 \pm 0 . 0 }$ </td><td> $\mathbf { 1 0 0 . 0 \pm 0 . 0 }$ </td><td> $\mathbf { 1 0 0 . 0 \pm 0 . 0 }$ </td><td> $\mathbf { 1 0 0 . 0 \pm 0 . 0 }$ </td></tr><tr><td>TFNl=9</td><td> $5 0 . 0 \pm 0 . 0$ </td><td> $\mathbf { 1 0 0 . 0 \pm 0 . 0 }$ </td><td> $5 0 . 0 \pm 0 . 0$ </td><td> $5 0 . 0 \pm 0 . 0$ </td><td> $5 0 . 0 \pm 0 . 0$ </td><td> $\mathbf { 1 0 0 . 0 \pm 0 . 0 }$ </td><td> $5 0 . 0 \pm 0 . 0$ </td><td> $5 0 . 0 \pm 0 . 0$ </td></tr><tr><td> $\mathrm { T F N } _ { l = 1 0 }$ </td><td>100.0 ± 0.0</td><td> $\mathbf { 1 0 0 . 0 \pm 0 . 0 }$ </td><td>100.0 ± 0.0</td><td> $\mathbf { 1 0 0 . 0 \pm 0 . 0 }$ </td><td> $\mathbf { 1 0 0 . 0 \pm 0 . 0 }$ </td><td> $\mathbf { 1 0 0 . 0 \pm 0 . 0 }$ </td><td>100.0 ± 0.0</td><td> $\mathbf { 1 0 0 . 0 \pm 0 . 0 }$ </td></tr><tr><td> $\mathrm { T F N } _ { l = 1 1 }$ </td><td> $5 0 . 0 \pm 0 . 0$ </td><td> $\mathbf { 1 0 0 . 0 \pm 0 . 0 }$ </td><td> $5 0 . 0 \pm 0 . 0$ </td><td> $5 0 . 0 \pm 0 . 0$ </td><td> $5 0 . 0 \pm 0 . 0$ </td><td> $\mathbf { 1 0 0 . 0 \pm 0 . 0 }$ </td><td> $5 0 . 0 \pm 0 . 0$ </td><td> $5 0 . 0 \pm 0 . 0$ </td></tr></table>

## F.3 MOLECULE PROPERTY PREDICTION WITH PRETRAINED EQUIVARIANT FEATURES

## F.3.1 DETAILED EXPERIMENTAL SETUP

We employ HEGNN as the encoder in our experiments. TFN is computationally prohibitive in this setting: its tensor-product operator has a complexity of $\mathcal { O } ( L ^ { 6 } )$ , and when the maximum degree is set to $L = 1 1$ (the upper limit supported by e3nn), training a four-layer TFN with only four irreducible-representation channels on QM9 requires roughly 10 A100 GPU-hours per epoch. This far exceeds any practical budget. In contrast, HEGNN adopts spherical scalarization, where interactions across different degrees are mediated solely through scalars, reducing the complexity to $\mathcal { O } ( L ^ { 2 } )$ . For this reason, we choose HEGNN as our encoder.

Concretely, we use a four-layer HEGNN with 16 irreducible-representation channels and a hidden dimension of 64. The resulting features are passed through a scalarization layer and then into a two-layer MLP to predict the molecular isotropic polarizability α on QM9. During fine-tuning, we freeze the encoder, apply a mask to selectively remove information, and train a separate two-layer MLP for each setting. Specifically, our designed mask multiplies the features of the unselected degrees by 0, followed by scalarization calculation, that is:

$$
\tilde { \mathbf { v } } _ { c } ^ { ( l ) } = \mathrm { H E G N N } ( \mathcal { G } ) ,\tag{112}
$$

$$
\begin{array} { r } { \pmb { s } ^ { ( l ) } = \mathrm { v e c } \big ( \big [ \langle \tilde { \pmb { v } } _ { c _ { 1 } } ^ { ( l ) } , \tilde { \pmb { v } } _ { c _ { 2 } } ^ { ( l ) } \rangle \big ] \big ) _ { C \times C } , l = 1 , \dots , 1 1 , } \end{array}\tag{113}
$$

$$
\begin{array} { r } { \hat { \alpha } = \mathtt { M L P } _ { \mathtt { f i n e t u n e } } ( \tilde { \pmb { v } } ^ { ( 0 ) } , \pmb { s } ^ { ( 1 ) } , \dots , \pmb { s } ^ { ( 1 1 ) } ) , } \end{array}\tag{114}
$$

Following the standard protocol, we train on the first 110k molecules for 300 epochs and fine-tune for an additional 30 epochs. Notably, although the final visualizations are produced by running the trained models on the entire QM9 dataset<sup>5</sup>, this does not affect our theoretical conclusions, as our analysis relies solely on horizontal comparisons rather than absolute predictive performance.

## F.3.2 CASE STUDIES

To further validate our theory, we analyze three prominent point groups $C _ { 2 h } , C _ { 3 h }$ , and $T _ { d }$ (see § F.3.2). Each exhibiting a representative pattern of symmetry increase. These groups display a symmetry increase to the full group O(3) under specific conditions: for $C _ { 2 h }$ when $l _ { 0 }$ is odd, for $C _ { 3 h }$ when $l _ { 0 } = 1$ , and for $T _ { d }$ when $l _ { 0 } = 1 , 2 , 5$ . This increase corresponds to full degeneration, making the features non-discriminative. The impact of this degeneration is evident in our experiments. Due to the full degeneration of 1-degree features, introducing them paradoxically decreases predictive performance for molecules with $C _ { 2 h } , C _ { 3 h }$ , and $T _ { d }$ symmetry. Following the introduction of 2-degree features, a marked improvement in performance is observed for $C _ { 2 h }$ and $C _ { 3 h }$ . Conversely, for $T _ { d }$ performance again decreases, a result directly attributable to the fact that its 2-degree features also undergo full degeneration.

![](images/bc68fd654f6c93425a15a5c47b0f0df7c4126ca3f11d0b8ade57fef4ed73a07c.jpg)

![](images/e2d04e91b4ae57b4a86143583596109abc34ebbf1f979cb97a5973e481948ee6.jpg)

![](images/f2498a62c78a87456e09afe3dbe888c95b9adada096f24329ec501a6521a9899.jpg)

![](images/1f693275e539751e849fbc1ec46e0232f880e4201975bdeb248760c9546d862c.jpg)

![](images/03af399d7539060ddcfe7ae08f246a0b1ce7ef75f5f3046ad264c1d450972f7e.jpg)

![](images/efa70cf9fcb2561d10fb997df2b3ac0e65743eb19d8f94f390b1daa86efc3c9f.jpg)

![](images/a2bd5535ef19947913c6a38a68353df3a491edf07cb14b71c9f8fa7e40f4e198.jpg)

![](images/216faf2b8d7ab4495cc8eff0c53a900ce2606281227c97bf5db3d374bf752ba4.jpg)

![](images/3221f776890273fedf8973ba765dd7fcc8b84cc3e480a2ef8dc3ab9366ad27e7.jpg)

![](images/39134ce1ae1fbd81c5dd2ba8ecbea94928a66f0a2b1c5e6c8ec366221d68837d.jpg)

![](images/b24607072c0990c9857ed2c7ff278f9bfbdbcd56ee73277011dd3f89af1336a6.jpg)

![](images/2487bfda8ff0a8c7a7b2f1430206d8ec8ede9dfa563f739c3e450392dbbebd4f.jpg)

![](images/62cc798e35d14a3ef20ed7fe277bedfb75e6c692400d0ac66761131cfd776c77.jpg)

![](images/e810e3675cf5b983c214994062b7bc86ccee6956531b611f24f11d2d998b1aa3.jpg)

![](images/0261b5847322dd383b7ce44e68f128284a420451a11008095a2300a23a890fa5.jpg)

![](images/8eac1d8235ee1bdccd9127f554bde501aa6bccc2a5ee5bb4a24661913f71d210.jpg)

Table 22: Symmetry infimum of point group symmetry on the QM9 dataset.
<table><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>10</td><td rowspan=1 colspan=1>11</td></tr><tr><td rowspan=1 colspan=1> $C _ { 1 }$ </td><td rowspan=1 colspan=1> $C _ { 1 }$ </td><td rowspan=1 colspan=1> $C _ { s }$ </td><td rowspan=1 colspan=1> $C _ { 1 }$ </td><td rowspan=1 colspan=1> $C _ { s }$ </td><td rowspan=1 colspan=1> $C _ { 1 }$ </td><td rowspan=1 colspan=1> $C _ { s }$ </td><td rowspan=1 colspan=1> $C _ { 1 }$ </td><td rowspan=1 colspan=1> $C _ { s }$ </td><td rowspan=1 colspan=1> $C _ { 1 }$ </td><td rowspan=1 colspan=1> $C _ { s }$ </td><td rowspan=1 colspan=1> $C _ { 1 }$ </td></tr><tr><td rowspan=1 colspan=1> $C _ { s }$ </td><td rowspan=1 colspan=1> $C _ { s }$ </td><td rowspan=1 colspan=1> $C _ { 2 h }$ </td><td rowspan=1 colspan=1> $C _ { s }$ </td><td rowspan=1 colspan=1> $C _ { 2 h }$ </td><td rowspan=1 colspan=1> $C _ { s }$ </td><td rowspan=1 colspan=1> $C _ { 2 h }$ </td><td rowspan=1 colspan=1> $C _ { s }$ </td><td rowspan=1 colspan=1> $C _ { 2 h }$ </td><td rowspan=1 colspan=1> $C _ { s }$ </td><td rowspan=1 colspan=1> $C _ { 2 h }$ </td><td rowspan=1 colspan=1> $C _ { s }$ </td></tr><tr><td rowspan=1 colspan=1> $C _ { 2 }$ </td><td rowspan=1 colspan=1> $C _ { \infty v }$ </td><td rowspan=1 colspan=1> $C _ { 2 h }$ </td><td rowspan=1 colspan=1> $C _ { 2 }$ </td><td rowspan=1 colspan=1> $C _ { 2 h }$ </td><td rowspan=1 colspan=1> $C _ { 2 }$ </td><td rowspan=1 colspan=1> $C _ { 2 h }$ </td><td rowspan=1 colspan=1> $C _ { 2 }$ </td><td rowspan=1 colspan=1> $C _ { 2 h }$ </td><td rowspan=1 colspan=1> $C _ { 2 }$ </td><td rowspan=1 colspan=1> $C _ { 2 h }$ </td><td rowspan=1 colspan=1> $C _ { 2 }$ </td></tr><tr><td rowspan=1 colspan=1> $C _ { 2 v }$ </td><td rowspan=1 colspan=1> $C _ { \infty v }$ </td><td rowspan=1 colspan=1> $D _ { 2 h }$ </td><td rowspan=1 colspan=1> $C _ { 2 v }$ </td><td rowspan=1 colspan=1> $D _ { 2 h }$ </td><td rowspan=1 colspan=1> $C _ { 2 v }$ </td><td rowspan=1 colspan=1> $D _ { 2 h }$ </td><td rowspan=1 colspan=1> $C _ { 2 v }$ </td><td rowspan=1 colspan=1> $D _ { 2 h }$ </td><td rowspan=1 colspan=1> $C _ { 2 v }$ </td><td rowspan=1 colspan=1> $D _ { 2 h }$ </td><td rowspan=1 colspan=1> $C _ { 2 v }$ </td></tr><tr><td rowspan=1 colspan=1> $C _ { 3 v }$ </td><td rowspan=1 colspan=1> $D _ { \infty h }$ </td><td rowspan=1 colspan=1> $C _ { \infty v }$ </td><td rowspan=1 colspan=1> $C _ { 3 v }$ </td><td rowspan=1 colspan=1> $D _ { 3 d }$ </td><td rowspan=1 colspan=1> $C _ { 3 v }$ </td><td rowspan=1 colspan=1> $D _ { 3 d }$ </td><td rowspan=1 colspan=1> $C _ { 3 v }$ </td><td rowspan=1 colspan=1> $D _ { 3 d }$ </td><td rowspan=1 colspan=1> $C _ { 3 v }$ </td><td rowspan=1 colspan=1> $D _ { 3 d }$ </td><td rowspan=1 colspan=1> $C _ { 3 v }$ </td></tr><tr><td rowspan=1 colspan=1> $C _ { 2 h }$ </td><td rowspan=1 colspan=1> $K _ { h }$ </td><td rowspan=1 colspan=1> $C _ { 2 h }$ </td><td rowspan=1 colspan=1> $K _ { h }$ </td><td rowspan=1 colspan=1> $C _ { 2 h }$ </td><td rowspan=1 colspan=1> $K _ { h }$ </td><td rowspan=1 colspan=1> $C _ { 2 h }$ </td><td rowspan=1 colspan=1> $K _ { h }$ </td><td rowspan=1 colspan=1> $C _ { 2 h }$ </td><td rowspan=1 colspan=1> $K _ { h }$ </td><td rowspan=1 colspan=1> $C _ { 2 h }$ </td><td rowspan=1 colspan=1> $K _ { h }$ </td></tr><tr><td rowspan=1 colspan=1> $C _ { i } = S _ { 2 }$ </td><td rowspan=1 colspan=1> $K _ { h }$ </td><td rowspan=1 colspan=1> $C _ { i }$ </td><td rowspan=1 colspan=1> $K _ { h }$ </td><td rowspan=1 colspan=1> $C _ { i }$ </td><td rowspan=1 colspan=1> $K _ { h }$ </td><td rowspan=1 colspan=1> $C _ { i }$ </td><td rowspan=1 colspan=1> $K _ { h }$ </td><td rowspan=1 colspan=1> $C _ { i }$ </td><td rowspan=1 colspan=1> $K _ { h }$ </td><td rowspan=1 colspan=1> $C _ { i }$ </td><td rowspan=1 colspan=1> $K _ { h }$ </td></tr><tr><td rowspan=1 colspan=1> $D _ { 3 h }$ </td><td rowspan=1 colspan=1> $K _ { h }$ </td><td rowspan=1 colspan=1> $D _ { \infty h }$ </td><td rowspan=1 colspan=1> $D _ { 3 h }$ </td><td rowspan=1 colspan=1> $D _ { \infty h }$ </td><td rowspan=1 colspan=1> $D _ { 3 h }$ </td><td rowspan=1 colspan=1> $D _ { 6 h }$ </td><td rowspan=1 colspan=1> $D _ { 3 h }$ </td><td rowspan=1 colspan=1> $D _ { 6 h }$ </td><td rowspan=1 colspan=1> $D _ { 3 h }$ </td><td rowspan=1 colspan=1> $D _ { 6 h }$ </td><td rowspan=1 colspan=1> $D _ { 3 h }$ </td></tr><tr><td rowspan=1 colspan=1> $D _ { 2 d }$ </td><td rowspan=1 colspan=1> $K _ { h }$ </td><td rowspan=1 colspan=1> $D _ { \infty h }$ </td><td rowspan=1 colspan=1> $T _ { d }$ </td><td rowspan=1 colspan=1> $D _ { 4 h }$ </td><td rowspan=1 colspan=1> $D _ { 2 d }$ </td><td rowspan=1 colspan=1> $D _ { 4 h }$ </td><td rowspan=1 colspan=1> $D _ { 2 d }$ </td><td rowspan=1 colspan=1> $D _ { 4 h }$ </td><td rowspan=1 colspan=1> $D _ { 2 d }$ </td><td rowspan=1 colspan=1> $D _ { 4 h }$ </td><td rowspan=1 colspan=1> $D _ { 2 d }$ </td></tr><tr><td rowspan=1 colspan=1> $C _ { 3 }$ </td><td rowspan=1 colspan=1> $C _ { \infty v }$ </td><td rowspan=1 colspan=1> $D _ { \infty h }$ </td><td rowspan=1 colspan=1> $C _ { 3 }$ </td><td rowspan=1 colspan=1> $C _ { 3 h }$ </td><td rowspan=1 colspan=1> $C _ { 3 }$ </td><td rowspan=1 colspan=1> $C _ { 3 h }$ </td><td rowspan=1 colspan=1> $C _ { 3 }$ </td><td rowspan=1 colspan=1> $C _ { 3 h }$ </td><td rowspan=1 colspan=1> $C _ { 3 }$ </td><td rowspan=1 colspan=1> $C _ { 3 h }$ </td><td rowspan=1 colspan=1> $C _ { 3 }$ </td></tr><tr><td rowspan=1 colspan=1> $D _ { \infty h }$ </td><td rowspan=1 colspan=1> $K _ { h }$ </td><td rowspan=1 colspan=1> $D _ { \infty h }$ </td><td rowspan=1 colspan=1> $K _ { h }$ </td><td rowspan=1 colspan=1> $D _ { \infty h }$ </td><td rowspan=1 colspan=1> $K _ { h }$ </td><td rowspan=1 colspan=1> $D _ { \infty h }$ </td><td rowspan=1 colspan=1> $K _ { h }$ </td><td rowspan=1 colspan=1> $D _ { \infty h }$ </td><td rowspan=1 colspan=1> $K _ { h }$ </td><td rowspan=1 colspan=1> $D _ { \infty h }$ </td><td rowspan=1 colspan=1> $K _ { h }$ </td></tr><tr><td rowspan=1 colspan=1> $C _ { \infty v }$ </td><td rowspan=1 colspan=1> $C _ { \infty v }$ </td><td rowspan=1 colspan=1> $D _ { \infty h }$ </td><td rowspan=1 colspan=1> $C _ { \infty v }$ </td><td rowspan=1 colspan=1> $D _ { \infty h }$ </td><td rowspan=1 colspan=1> $C _ { \infty v }$ </td><td rowspan=1 colspan=1> $D _ { \infty h }$ </td><td rowspan=1 colspan=1> $C _ { \infty v }$ </td><td rowspan=1 colspan=1> $D _ { \infty h }$ </td><td rowspan=1 colspan=1> $C _ { \infty v }$ </td><td rowspan=1 colspan=1> $D _ { \infty h }$ </td><td rowspan=1 colspan=1> $C _ { \infty v }$ </td></tr><tr><td rowspan=1 colspan=1> $D _ { 3 d }$ </td><td rowspan=1 colspan=1> $K _ { h }$ </td><td rowspan=1 colspan=1> $D _ { \infty h }$ </td><td rowspan=1 colspan=1> $K _ { h }$ </td><td rowspan=1 colspan=1> $D _ { 3 d }$ </td><td rowspan=1 colspan=1> $K _ { h }$ </td><td rowspan=1 colspan=1> $D _ { 3 d }$ </td><td rowspan=1 colspan=1> $K _ { h }$ </td><td rowspan=1 colspan=1> $D _ { 3 d }$ </td><td rowspan=1 colspan=1> $K _ { h }$ </td><td rowspan=1 colspan=1> $D _ { 3 d }$ </td><td rowspan=1 colspan=1> $K _ { h }$ </td></tr><tr><td rowspan=1 colspan=1> $D _ { 2 h }$ </td><td rowspan=1 colspan=1> $K _ { h }$ </td><td rowspan=1 colspan=1> $D _ { 2 h }$ </td><td rowspan=1 colspan=1> $K _ { h }$ </td><td rowspan=1 colspan=1> $D _ { 2 h }$ </td><td rowspan=1 colspan=1> $K _ { h }$ </td><td rowspan=1 colspan=1>D2h</td><td rowspan=1 colspan=1> $K _ { h }$ </td><td rowspan=1 colspan=1> $D _ { 2 h }$ </td><td rowspan=1 colspan=1> $K _ { h }$ </td><td rowspan=1 colspan=1> $D _ { 2 h }$ </td><td rowspan=1 colspan=1> $K _ { h }$ </td></tr><tr><td rowspan=1 colspan=1> $T _ { d }$ </td><td rowspan=1 colspan=1> $K _ { h }$ </td><td rowspan=1 colspan=1> $K _ { h }$ </td><td rowspan=1 colspan=1> $T _ { d }$ </td><td rowspan=1 colspan=1> $O _ { h }$ </td><td rowspan=1 colspan=1> $K _ { h }$ </td><td rowspan=1 colspan=1> $O _ { h }$ </td><td rowspan=1 colspan=1> $T _ { d }$ </td><td rowspan=1 colspan=1> $O _ { h }$ </td><td rowspan=1 colspan=1> $T _ { d }$ </td><td rowspan=1 colspan=1> $O _ { h }$ </td><td rowspan=1 colspan=1> $T _ { d }$ </td></tr><tr><td rowspan=1 colspan=1> $C _ { 3 h }$ </td><td rowspan=1 colspan=1> $K _ { h }$ </td><td rowspan=1 colspan=1> $D _ { \infty h }$ </td><td rowspan=1 colspan=1> $C _ { 3 h }$ </td><td rowspan=1 colspan=1> $D _ { \infty h }$ </td><td rowspan=1 colspan=1> $C _ { 3 h }$ </td><td rowspan=1 colspan=1> $C _ { 6 h }$ </td><td rowspan=1 colspan=1> $C _ { 3 h }$ </td><td rowspan=1 colspan=1> $C _ { 6 h }$ </td><td rowspan=1 colspan=1> $C _ { 3 h }$ </td><td rowspan=1 colspan=1> $C _ { 6 h }$ </td><td rowspan=1 colspan=1> $C _ { 3 h }$ </td></tr></table>

The experiment demonstrates a model design principle: not only should one avoid building a model entirely from fully degenerate features, but one should also avoid including individual feature components that undergo full degeneration, as they can be harmful to predictive performance.