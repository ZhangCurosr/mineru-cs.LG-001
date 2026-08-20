# Learning Topological Features of Zb-invariants

Brandon Robinson<sup>†,∗</sup> Shimal Harichurn<sup>†</sup> Fabian Ruehle<sub>c,d,e</sub> Sergei Gukov<sub>f</sub> Rak-Kyeong Seong<sub>g</sub> Miranda C. N. Cheng <sub>,h,</sub>

<sup>a</sup>Institute of Physics, University of Amsterdam, Science Park 904, 1098 XH Amsterdam, Netherlands <sup>b</sup>School of Mathematics, Statistics and Computer Science, University of KwaZulu-Natal, South Africa <sup>c</sup>Department of Physics, Northeastern University, Boston, MA 02115, USA

<sup>d</sup>Department of Mathematics, Northeastern University, Boston, MA 02115, USA

<sup>e</sup>The NSF AI Institute for Artificial Intelligence and Fundamental Interactions

<sup>f</sup> Richard N. Merkin Center for Pure and Applied Mathematics, California Institute of Technology, Pasadena, CA 91125, USA

<sup>g</sup>Department of Mathematical Sciences, and Department of Physics, Ulsan National Institute of Science and Technology, 50 UNIST-gil, Ulsan 44919, South Korea

<sup>h</sup>Institute for Mathematics, Academia Sinica, Taipei 106319, Taiwan

<sup>i</sup>Korteweg-de Vries Institute for Mathematics, University of Amsterdam, Science Park 105-107, 1098 XG Amsterdam, Netherlands

## Abstract

Machine learning and data analysis techniques have recently emerged as powerful tools for identifying patterns and formulating conjectures in mathematical research, most notably in the field of low-dimensional topology. In this paper, we initiate a systematic approach to handling mathematical data structured as (truncated) infinite q-series, or equivalently, infinite series of integers. To apply this data analysis pipeline, we construct a comprehensive dataset of Zb-invariants (homological blocks) for plumbed 3-manifolds. We demonstrate that neural networks can reliably extract essential topological information, such as homology class and underlying graph structure, directly from the q-series coefficients. A central feature of our methodology is a focus on interpretability; by contrasting local gradient sensitivity with global feature relevance, we reveal that the networks learn to bypass complex topological rules in favor of specific spectral and geometric proxies. Finally, we apply this pipeline to probe homology cobordism, discovering a high-accuracy predictive relationship between the Zb-invariant exponents and the Heegaard Floer d-invariant (correction term). These results suggest that Zb-invariants capture subtle geometric information regarding cobordism equivalences, warranting a new direction for the study of quantum invariants.

## Contents

1 Introduction 2   
2 Glossary 6   
3 Data 7   
3.1 Datasets 7   
3.2 Data Representations 12   
4 Topology Experiments 13   
4.1 Task Description 14   
4.2 Full Sequence Representation 15   
4.3 Value-Index Pair Representation 24   
4.4 Latent Space Representation 27   
4.5 Control Experiment: Derived Features 32   
5 Cobordism Experiments 34   
5.1 Learning d from $\widehat { Z }$ 34   
6 Other Experiments 39   
6.1 Multi-class classification: Singular fibers 39   
6.2 Learning $S L _ { 2 } ( \mathbb { Z } )$ orbit 40   
7 Conclusions and outlook 42   
A Data Analysis Tools and Concepts 44   
A.1 Model Performance Metrics 44   
A.2 Model Explainability Techniques 47   
A.3 Contrastive Learning 51   
B Interpretability Robustness 52   
B.1 Multi-seed robustness protocol 52   
B.2 Normalized Margin Reconstruction protocol 54   
C Spectral bound on $| \varepsilon _ { \mathrm { m i n } } |$ 56   
D Bound involving minτ 56   
E Splice diagrams and integral homology Spheres 58   
E.1 An algorithm to produce integral homology Spheres 59   
E.2 Algorithms and Pseudocode 60   
References 61

## 1 Introduction

The interaction between artificial intelligence and pure mathematics has historically focused on automated theorem proving or symbolic computation. In recent years, however, machine learning (ML) has emerged as a distinct and powerful partner in the theoretical research pipeline, expanding the scope of neural network applications from verification to discovery. This paradigm leverages the capacity of neural networks to detect high-dimensional correlations that may elude human intuition, thereby aiding researchers towards finding new, mathematically rigorous conjectures. So far, this strategy has proven particularly fruitful in the field of lowdimensional topology, where vast tabulations of knot invariants provide rich, structured datasets suitable for data-driven analysis. Notable examples include discovering novel relationships between the geometric structure of knots and their signature [1], as well as predicting invariants such as the Jones polynomial and hyperbolic volume [2, 3, 4]. Partially as a result of the availability of knot datasets, the majority of this existing literature focuses on knots and knot complements rather than closed 3-manifolds, and typically targets scalar or polynomial invariants [5, 6, 7, 8].

In this work, we turn to data in the form of infinite q-series, or equivalently infinite series of integers, appropriately truncated. This is in part motivated by the so-called Zb-invariants of closed 3-manifolds, $\widehat { Z } ( M _ { 3 } ; q )$ . The $\widehat { Z }$ -invariants, often referred to as homological blocks, constitute a family of infinite q-series invariants, $\widehat { Z } _ { b } ( q ) \in q ^ { \Delta _ { b } } \mathbb { Z } [ [ q ] ]$ , characterized as power series with integer coefficients that converge within the unit disk $\vert q \vert < 1 [ 9 , 1 0 ]$ . These invariants have their origin in M-theory, arising from the physics of M5-branes wrapped on 3-manifolds, where they capture the 3d half-index (the supersymmetric partition function on a cigar background multiplied by the temporal circle) of the 3-dimensional ${ \mathcal { N } } = 2$ theory resulting from the compactification.

Much of the study of the ${ \widehat { Z } } .$ -invariants, including the present work, focuses on a class of 3- manifolds known as the plumbed 3-manifolds. Given a weighted tree $\Gamma ,$ one can build a framed link $L ( \Gamma ) \subseteq S ^ { 3 }$ consisting of copies of unknots chained together whenever vertices are connected by an edge in Γ and with framing numbers provided by the weights on the vertices. One can then perform Dehn surgery on the resulting framed link to obtain a closed 3-manifold Y(Γ), called a plumbed manifold. We call Γ a plumbing graph for the corresponding plumbed manifold. The data of the weighted tree can equivalently be encoded in a matrix, referred to as the plumbing matrix. For a so-called weakly negative plumbed 3-manifolds, the invariants are defined mathematically through a principal value contour integral of a lattice theta function, appropriately decorated by the contribution from the edges of the plumbing graph. A plumbed manifold is said to be weakly negative if its plumbing matrix inverse, $M ^ { - 1 }$ , is negative definite when restricted to the subspace spanned by vertices of degree three or greater. We refer to these vertices as the high-degree vertices. Concretely, the ${ \widehat { Z } } .$ -invariants for these manifolds are given by

$$
\widehat Z _ { b } ( M _ { 3 } ; \tau ) : = q ^ { \Delta ^ { \prime } } \oint \prod _ { \nu \in V } { \frac { d z _ { \nu } } { 2 \pi i z _ { \nu } } } \left( z _ { \nu } - { \frac { 1 } { z _ { \nu } } } \right) ^ { 2 - \mathrm { d e g } ( \nu ) } \Theta _ { b } ^ { M } ( \tau ; { \bf z } )\tag{1}
$$

for some $\Delta ^ { \prime } \in \mathbb { Q }$ , and some choice of vector $b$ in the lattice, where the theta function is given by

$$
\Theta _ { b } ^ { M } ( \tau ; \mathbf { z } ) : = \sum _ { \ell \in 2 M \mathbb { Z } ^ { | V | } \pm b } q ^ { - \ell ^ { T } M ^ { - 1 } \ell / 4 } \mathbf { z } ^ { \ell } .\tag{2}
$$

These invariants are significant across the realms of mathematics and physics, functioning simultaneously as topological invariants and BPS invariants, and they are believed to be closely related to quantum modular forms and (non-rational) vertex operator algebras [11, 12, 13, 14, 15]. They are known to extend the Witten–Reshetikhin–Turaev (WRT) invariants away from the roots of unity, and it is expected that their categorification will yield powerful smooth 4- manifold invariants.

While various properties of the $\widehat { Z }$ -invariants have been intensively studied in recent years, many remain mysterious. For instance, we would like to ask

What type of information about the 3-manifold $M _ { 3 }$ does $\widehat { Z } ( M _ { 3 } )$ encode? And how?

Given the recent advances in data analysis and machine learning methods, it is natural to probe these questions using these new tools. Such an approach was initiated in the preliminary work [16], where the authors performed Principal Component Analysis (PCA) of ${ \widehat { Z } } ,$ truncated to ten terms, for a small dataset of 20 plumbed 3-manifolds. To greatly scale up such a data-based study of Zb-invariants, in this paper we first build various datasets of size $1 0 ^ { 4 }$ of plumbed 3- manifolds, each with ${ \widehat { Z } } .$ -invariants computed up to the first $1 0 ^ { 4 }$ terms. Subsequently, we build an ML-assisted data analysis pipeline to extract information from this type of data.

Beyond standard prediction networks, we place special emphasis on interpreting the mechanisms driving the neural network’s decisions. To achieve this, drawing on the strategy of [17], we train a separate regression network to map a comprehensive set of mathematically motivated features directly to the logits of the classifier. We then employ data analysis tools, including feature saliency (FS), permutation importance (PI), and Shapley values (SHAP), to decode the information exploited by the classification model. Figure 1 summarizes the integrated pipeline developed in this work.

![](images/799ad40f8824cf59a18bbfc38cccd30cf7958cb775d6f4001ab3d041d91270dc.jpg)  
Figure 1: Schematic of the ML-assisted data analysis pipeline.

Recall that the ${ \widehat { Z } } .$ -invariants in our database are computed using the plumbing graph, or equivalently the plumbing matrix, of a closed 3-manifold $M _ { 3 }$ . To test the interpretability methodology, we first choose classification tasks which are straightforward using the plumbing graph/matrix but obscure when just given the resulting infinite $\widetilde { \widehat { Z } }$ q -series. Examples include the identification of integral homology spheres, which correspond to plumbed manifolds with unimodular plumbing matrices, and the determination of the underlying graph topology, such as distinguishing star-shaped graphs from H-shaped configurations.

To ensure the robustness of the interpretations, we perform many of the experiments in three different data representations — Full Sequence (including all zero coefficients), Value-Index Pair (excluding zeros), and Latent Space representations trained using contrastive learning with labeled datasets. We find that our method can indeed lead to human-understandable insights into how the classification network decodes specific topological information from the infinite q-series.

These insights reveal that the neural network sometimes employs simpler “surrogate” features that are merely correlated with the defining mathematical properties, rather than learning the exact theoretical rules. For instance, we observe that the network effectively uses the minimummagnitude eigenvalue of the plumbing matrix as a proxy for the determinant in some of the classification tasks. This reliance on proxies provides a clear and quantifiable explanation for the classifier’s imperfect generalization, as we discuss in detail in $\ S 4$ and §6.

Having validated our methodology on basic topological questions, we turn to the exploration of the connection between the q-series invariants and the three-dimensional homology cobordism group $\Theta _ { \mathbb { Z } } ^ { 3 } .$ , a subject of prime interest in low-dimensional topology. Indeed, while $\Theta _ { \mathbb { Z } } ^ { 3 }$ is related to many important questions about smooth structures on 4-manifolds and other exciting questions, its structure — for instance, the existence of nontrivial torsion — remains quite mysterious

## despite decades of study.

The definition of $\Theta _ { \mathbb { Z } } ^ { 3 }$ is deceptively simple: it is the set of oriented integer homology spheres modulo homology cobordism, where two spheres $Y _ { 0 }$ and $Y _ { 1 }$ are equivalent if they cobound a smooth 4-manifold W, $\partial W = Y _ { 0 } \sqcup ( - Y _ { 1 } )$ , such that the inclusion maps $Y _ { i } \hookrightarrow W$ induce isomorphisms on integer homology. Under the connected sum operation, these equivalence classes form an abelian group, with the identity element represented by the standard sphere $S ^ { 3 }$ Some of the basic properties of the homology cobordism group come from Rokhlin’s signature theorem, which yields the homomorphism $\mu \colon \Theta _ { \mathbb { Z } } ^ { 3 } \to \mathbb { Z } / 2 [ 1 8 ]$ , and one has the associated short exact sequence

$$
0 \longrightarrow \ker ( \mu ) \longrightarrow \Theta _ { \mathbb { Z } } ^ { 3 } \stackrel { \mu } { \longrightarrow } \mathbb { Z } / 2 \longrightarrow 0 .
$$

Galewski–Stern and Matumoto showed that the question of whether this sequence splits is equivalent to the existence of simplicial triangulations for all manifolds of dimension $\ge ~ 5$ [19, 20]. Manolescu proved that it does not split using Pin(2)-equivariant Seiberg–Witten Floer theory, thereby resolving the high-dimensional Triangulation Conjecture [21]. This illustrates how the integer homology cobordism group has important applications in low-dimensional topology, albeit through rather advanced tools like gauge theory and Floer homology.

Important developments in understanding the structure of $\Theta _ { \mathbb { Z } } ^ { 3 }$ include the construction of explicit infinite-order elements via instanton Floer theory by Fintushel–Stern [22], Furuta’s proof that $\Theta _ { \mathbb { Z } } ^ { 3 }$ contains a free abelian subgroup of infinite rank [23], and Frøyshov’s surjective homomorphism h: $\Theta _ { \mathbb { Z } } ^ { 3 } \to \mathbb { Z }$ (hence a <sup>Z</sup>-summand) [24]. Heegaard Floer theory provides computable homology cobordism invariants such as the correction term d [25] and its involutive refinements [26], and Dai–Hom–Stoffregen–Truong showed that $\Theta _ { \mathbb { Z } } ^ { 3 }$ admits an infinite-rank direct summand [27]. Despite these advances, key questions remain open, so the group continues to serve as a central testing ground for new topological and quantum invariants [28], particularly their ability to detect non-trivial cobordism classes that classical invariants fail to distinguish.

After having tested our data analysis methodology on basic topological questions with the $\widehat { Z }$ database we constructed, we turn to the question of probing cobordism information contained in Zb-invariants. A homology cobordism invariant is a group homomorphism $\lambda : \Theta _ { \mathbb { Z } } ^ { 3 } \to A$ where A is an abelian group. A modern example of such an invariant is the Heegaard Floer correction term, or d-invariant, introduced by Ozsváth and Szabó [25]. Originating from the grading of the Heegaard Floer homology group $H F ^ { + } ( Y )$ , the d-invariant measures the “spectral shift” of the infinite rank tower in the homology relative to that of the 3-sphere. For integer homology spheres it takes values in the even integers and provides a powerful obstruction to slicing knots and embedding manifolds. Partially inspired by previous work [29, 30, 31], which studied the relationship of d to the leading q-power of the Zb-invariant (the ∆-invariant) and to specializations of the Zb-invariant, in this work we explore the relation between the q-powers of the Zb-invariant and the d invariant.

To our surprise, our experiment yielded the following result.

On datasets of Seifert integer homology spheres with r singular fibers (Brieskorn spheres for   
$r = 3 )$ , we found that regression models are able to learn to fit d from the leading exponents of $\widehat { Z }$ and explain > 99.9% of the target variance. By including symmetrized Dedekind sums of the   
Seifert invariants in the training data, exact integral predictions of d reach nearly 90% accuracy from quadratic polynomials of the important exponents.

Through various diagnostics, we were able to provide a heuristic understanding of how such a positive result, which might be surprising at first glance, can be plausibly obtained by machine learning. Nevertheless, we believe that the result indicates the possibility of some new relation between $\widehat { Z }$ -invariants and the d-invariant, which in turn suggests an interesting connection between the homology cobordism group and the ${ \widehat { Z } } .$ -invariants.

The remainder of this paper is organized as follows. Section 2 summarizes the notation used throughout this work. Section 3 describes the generation of $\widehat { Z }$ datasets for manifolds plumbed by star- and H-shaped graphs and introduces our input representations. Section 4 details the topological classification experiments, the isolation of driving features, and a control study utilizing the derived features. Section 5 investigates the prediction of the Heegaard-Floer d-invariant from q-series coefficients. Section 6 presents additional experiments regarding singular fiber counts and $S L _ { 2 } ( \mathbb { Z } )$ orbit sizes. Finally, Section 7 provides a summary of our findings and future directions, followed by technical appendices on data-analysis tools, the multi-seed robustness protocol, spectral and min τ bounds, and splice diagrams.

## 2 Glossary

$c _ { i }$ The integer coefficients appearing in the q-series expansion.

$c _ { \mathrm { m a x } }$ The maximum coefficient value found in the truncated q-series expansion used as input data.

$\Delta$ The minimal q-power of $\widehat { Z } = q ^ { \Delta } ( 1 + O ( q ) )$

M The plumbing matrix associated with the plumbing graph of the manifold

det The determinant of M.

$\varepsilon _ { \mathrm { { m i n } } }$ The eigenvalue of M with the smallest magnitude.

$\varepsilon _ { \mathrm { m a x } }$ The eigenvalue of M with the largest magnitude.

$\mu _ { \varepsilon }$ The mean value of the eigenvalues of M.

$\sigma _ { \varepsilon }$ The standard deviation of the eigenvalues of M.

$\gamma _ { 0 }$ The first gap between exponents with nonzero coefficients in the q-series.

$\mu _ { \gamma }$ The mean of the gaps between exponents of nonzero coefficients in the q-series.

$\sigma _ { \gamma }$ The standard deviation of the gaps between exponents of nonzero coefficients in the qseries.

d The Heegaard-Floer homology cobordism invariant, the “correction term".

α The size of the $S L _ { 2 } ( \mathbb { Z } )$ orbit associated with the invariant.

BAS Balanced Accuracy Score: The average recall across all classes, used to evaluate classification performance on imbalanced datasets.

PCC Pearson Correlation Coefficient: A metric measuring the linear correlation between predicted and ground truth values, ranging from -1 to 1.

$R ^ { 2 }$ Coefficient of Determination: A regression metric quantifying the proportion of variance in the observed data explained by the model.

ρ Spearman Rank Correlation: A metric assessing the monotonic relationship between ground truth and predictions by calculating the correlation of their rank vectors.

ϑ Sign Accuracy: A metric evaluating whether a regression output falls on the correct side of a decision boundary (typically zero), effectively treating it as a binary classifier.

$S _ { \mathrm { M A E } }$ Complement Normalized MAE Score: A metric derived from the Mean Absolute Error, normalized by the range and inverted so that higher values indicate better performance.

Q<sub>composite</sub> A weighted composite score combining $R ^ { 2 } , \rho , \vartheta _ { \mathrm { i } }$ , and $S _ { \mathrm { M A E } }$ to assess how well a regressor characterizes a decision boundary.

FS Feature Saliency: A gradient-based interpretability method that weights input features by their gradient contribution to the network’s output.

PI Permutation Importance: A model-agnostic method that quantifies feature importance by measuring the drop in model performance when a feature’s values are randomly shuffled.

SHAP Shapley Additive Explanations: A game-theoretic approach to interpretability that distributes the model’s prediction deviation from the average among features based on their marginal contributions.

SHTL Semi-Hard Triplet Loss: A loss function used in contrastive learning that focuses on "semihard" negatives (negatives farther from the anchor than the positive but within a margin) to improve latent space clustering.

## 3 Data

This section details the data generation process, the resulting datasets, and the input representations used in our experiments. We name the datasets based on the topology of their underlying plumbing graphs and homological property. Each sample consists of the q-series expansion of the $\widehat { Z } _ { b } ( M _ { 3 } )$ invariant, truncated (for the topology datasets of Sec. 4) to the first ${ \bar { 1 } } 0 ^ { 4 }$ coefficients — the cobordism and orbit-size datasets of later sections use shorter windows described there, and the orbit-size analysis additionally assumes square-free $b _ { 1 } b _ { 2 } b _ { 3 } -$ , for a plumbed 3- manifold $M _ { 3 }$ equipped with a fixed Spin<sup>c</sup>-structure $b ^ { 1 }$ . In total, our aggregate dataset comprises 63,603 such truncated series, derived from plumbing graphs containing one or two high-degree vertices, which we refer to as the star-shaped and H-shaped graphs, respectively. To ensure the quality of the graph topology labels in the dataset, all of the generated graphs have been passed through an automated algorithm to check that they are in minimal form.

## 3.1 Datasets

We begin by outlining the datasets constructed for our analysis. A key constraint is the computational cost of generating data for graphs with multiple high-degree nodes; consequently, our dataset is predominantly composed of star-shaped graphs (roughly 71%), namely weighted trees with a single high-degree vertex. Nevertheless, with sample sizes of $O ( 1 0 ^ { 4 } )$ for each category, the data remains robust enough to support the machine learning tasks described in this paper.

<table><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Homology Sphere |</td><td rowspan=1 colspan=1>Non-Sphere</td></tr><tr><td rowspan=1 colspan=1>Star</td><td rowspan=1 colspan=1>13228</td><td rowspan=1 colspan=1>31706</td></tr><tr><td rowspan=1 colspan=1>H-shaped</td><td rowspan=1 colspan=1>7952</td><td rowspan=1 colspan=1>10717</td></tr></table>

Table 1: Summary of datasets split between homology and graph topology labels used for binary classification tasks.

<table><tr><td rowspan=3 colspan=1># singular</td><td rowspan=3 colspan=1>fibe</td><td></td><td></td><td></td><td></td></tr><tr><td rowspan=2 colspan=1>rs | 3</td><td rowspan=2 colspan=1>| 4 |</td><td></td><td></td></tr><tr><td rowspan=1 colspan=1>5 | 6</td><td rowspan=1 colspan=1>| 7</td></tr><tr><td rowspan=1 colspan=1># samples | 6</td><td rowspan=1 colspan=1>000 | 5</td><td rowspan=1 colspan=1>998 | 5</td><td rowspan=1 colspan=1>988 |</td><td rowspan=1 colspan=1>5984 |</td><td rowspan=1 colspan=1>5978</td></tr></table>

Table 2: Data for multi-class experiments learning the number of singular fibers of Seifert manifolds from $\widehat { Z } .$

## 3.1.1 Star-shaped (Seifert) Datasets

Star-shaped plumbing graphs correspond to Seifert fibered 3-manifolds. In this paper, we restrict our attention to those satisfying the weakly negative condition allowing for straightforward computation of the ${ \widehat { Z } } .$ -invariants.

Specifically, for a star-shaped graph as shown in Figure 2, we require that on each of the r legs (chains of degree-2 vertices terminating in a degree-1 vertex), the weights satisfy the continued fraction decomposition:

$$
\begin{array} { r } { w _ { i , 1 } - \displaystyle \frac { 1 } { w _ { i , 2 } - \frac { 1 } { \cdots - \frac { 1 } { w _ { i , l _ { i } } } } } = \frac { b _ { i } } { a _ { i } } } \\ { \displaystyle \qquad \cdots - \frac { 1 } { w _ { i , l _ { i } } } } \end{array}
$$

for coprime integers $a _ { i } , b _ { i }$ satisfying $0 < a _ { i } < b _ { i }$ . Additionally, we impose the condition that the weight of the central vertex (of degree r) satisfies $b < 0 .$ . Under these constraints, the resulting plumbed manifold corresponds to the Seifert fibered space $M ( b ; \frac { a _ { 1 } } { b _ { 1 } } , \ldots , \frac { a _ { r } } { b _ { r } } )$ . We refer to the pairs $( a _ { i } , b _ { i } )$ as the Seifert invariants of the manifold. Note that the orbifold Euler number $e \in \mathbb { Q }$ is given by:

$$
e = b + \sum _ { i = 1 } ^ { r } { \frac { a _ { i } } { b _ { i } } } .\tag{3}
$$

Crucially, the condition $b < 0$ typically ensures that the plumbing matrix is weakly negative, a requirement for the convergence of the Zb-invariant.

The normalization condition $0 < a _ { i } < b _ { i }$ serves to fix the ambiguity arising from 3D Neumann moves. By imposing these constraints, we ensure that each Seifert manifold in our dataset is represented by a unique ’canonical’ plumbing graph, eliminating redundancy in the input data.

![](images/6c82391ead69c8e4f631b5ce2f1d0d0691ae803ea407ef4162d16eed1b4843e3.jpg)  
Figure 2: A star-shaped graph representing a Seifert manifold. Vertex labels have been omitted.

## Star-Non-Sphere Dataset

In the case of star-shaped graphs with three legs, we perform a brute-force search over the $b _ { i }$ in the range $1 < b _ { i } < 6 0$ for $i = { 1 , 2 , 3 }$ and find the corresponding $a _ { i } s$ such that the pairs $( b _ { i } , a _ { i } )$ are Seifert invariants for some Seifert manifold.

When generating star-shaped graphs with more than three legs, we adopt a streamlined sampling strategy. We arbitrarily select a central weight $b \in [ - 1 0 , - 1 ]$ and base weights $b _ { i }$ in the range $1 < b _ { i } < 5 0$ , while fixing all numerators $a _ { i } = 1$ . Since $1 < b _ { i }$ , the pairs $( b _ { i } , 1 )$ automatically satisfy the condition for valid Seifert invariants $( 0 < a _ { i } < b _ { i } )$ . Geometrically, setting $a _ { i } = 1$ corresponds to legs of length one; consequently, the resulting plumbing graphs contain no degree-2 vertices. Exploiting this strategy, we populate the Star-Non-Sphere dataset. In cases where the determinant of the resulting plumbing matrix is ±1, this strategy produces a homology sphere, and we instead add it to the Star-Homology Sphere Dataset.

## Star-Homology Sphere Dataset

Recall that the manifold corresponding to the plumbing graph in Figure 2 is an integral homology sphere if and only if the determinant of the corresponding plumbing matrix is ±1. In the Seifert case, this condition implies that the orbifold Euler number e and the base weights $b _ { i }$ must satisfy:

$$
e \prod _ { i = 1 } ^ { r } b _ { i } = \pm 1 .\tag{4}
$$

While the general topological definition allows for either sign, the weakly negative condition required for the ${ \widehat { Z } } .$ -invariants restricts us to the case where the orbifold Euler number is negative. Consequently, we enforce the condition e $\prod b _ { i } = - 1$ . Crucially, with the sign fixed $\mathrm { t o } - 1$ , for any set of pairwise coprime integers $\{ b _ { 1 } , . . . , b _ { r } \}$ , the numerators $a _ { i }$ are uniquely determined by this equation combined with the normalization conditions $b \in \mathbb { Z }$ and $0 < a _ { i } < b _ { i }$ . We generate datasets for central vertex degree $r = 4 , 5 ,$ 6 by performing a brute-force search over possible tuples $( b _ { 1 } , \ldots , b _ { r } )$ that admit such solutions. To ensure uniqueness and reduce the search space, we impose the ordering $2 \leq b _ { 1 } < b _ { 2 } < \cdots < b _ { r }$ . Specifically, for $r = 4$ , we search for configurations where the largest weight satisfies $b _ { r } \leq 2 9$ . For $r = 5$ and $r = 6 _ { \mathrm { : } }$ , we restrict the search to $b _ { r } \leq 2 3$ . For each valid tuple found, we compute the unique corresponding $a _ { i }$ values to reconstruct the full plumbing graph.

The case of three singular fibers admits an alternative description:

$$
\Sigma ( b _ { 1 } , b _ { 2 } , b _ { 3 } ) = \{ ( x _ { 1 } , x _ { 2 } , x _ { 3 } ) \in { \mathbb { C } } ^ { 3 } \mid x _ { 1 } ^ { b _ { 1 } } + x _ { 2 } ^ { b _ { 2 } } + x _ { 3 } ^ { b _ { 3 } } = 0 \} \cap S ^ { 5 }
$$

and in this case the ${ \widehat { Z } } .$ -invariant admits a particularly simple expression, allowing for faster computations.

Define $m : = { b _ { 1 } b _ { 2 } b _ { 3 } }$ and

$$
\alpha _ { \epsilon _ { 1 } , \epsilon _ { 2 } , \epsilon _ { 3 } } = b _ { 1 } b _ { 2 } b _ { 3 } \left( 1 - \sum _ { i = 1 } ^ { 3 } ( - 1 ) ^ { \epsilon _ { i } } \frac { 1 } { b _ { i } } \right) , \epsilon _ { i } \in \{ 0 , 1 \} .
$$

We then define false-theta functions

$$
\widetilde { \vartheta } _ { m , a } ( q ) : = \sum _ { n = 0 } ^ { \infty } \psi _ { 2 m } ^ { ( a ) } ( n ) q ^ { \frac { n ^ { 2 } } { 4 m } } , \qquad \widetilde { \vartheta } _ { m , a } ( q ) \in q ^ { \frac { a ^ { 2 } } { 4 m } } \mathbb { Z } [ [ q ] ] ,\tag{5}
$$

where

$$
\psi _ { 2 m } ^ { ( a ) } ( n ) = \left\{ \begin{array} { c } { { \pm 1 \mathrm { i f } n = \pm a \bmod 2 m } } \\ { { 0 \mathrm { o t h e r w i s e . } } } \end{array} \right.\tag{6}
$$

Then for $M _ { 3 } = \Sigma ( b _ { 1 } , b _ { 2 } , b _ { 3 } )$ we have

$$
\widehat { Z } ( \boldsymbol { q } ) = \boldsymbol { q } ^ { \tilde { \Delta } } \cdot \left[ \boldsymbol { q } ^ { - \frac { \alpha _ { 0 , 0 , 0 } ^ { 2 } } { 4 m } } \sum _ { \epsilon _ { i } \in \{ 0 , 1 \} \atop \sum _ { i = 1 } ^ { 3 } \epsilon _ { i } \equiv 0 ( 2 ) } \widetilde { \vartheta } _ { m , \alpha _ { \epsilon _ { 1 } , \epsilon _ { 2 } , \epsilon _ { 3 } } } ( \boldsymbol { q } ) \right] + p ( \boldsymbol { q } )\tag{7}
$$

where $p ( q ) = 0$ for $( b _ { 1 } , b _ { 2 } , b _ { 3 } ) \neq ( 2 , 3 , 5 )$ and is a simple q-monomial when $( b _ { 1 } , b _ { 2 } , b _ { 3 } ) = ( 2 , 3 , 5 )$ The factor $\tilde { \Delta }$ can be computed

$$
\tilde { \Delta } = \frac { 1 } { 4 } \left( \sum _ { i = 1 } ^ { 3 } h _ { i } - 3 \nu - \mathrm { T r } ( M ) - \frac { b _ { 2 } b _ { 3 } } { b _ { 1 } } - \frac { b _ { 1 } b _ { 3 } } { b _ { 2 } } - \frac { b _ { 1 } b _ { 2 } } { b _ { 3 } } \right) + \frac { \alpha _ { 0 , 0 , 0 } ^ { 2 } } { 4 m }\tag{8}
$$

where M is the plumbing matrix for $\Sigma ( b _ { 1 } , b _ { 2 } , b _ { 3 } )$ , and $h _ { i }$ is the absolute value of the determinant of the plumbing matrix of the graph obtained by deleting the terminal vertex (a vertex with degree 1) on the i-th leg from the plumbing graph Γ of $\Sigma ( b _ { 1 } , b _ { 2 } , b _ { 3 } )$ , and ν is the number of vertices in Γ.

Using this approach, we compute $\widehat { Z }$ for a subset of the Brieskorn spheres $\Sigma ( b _ { 1 } , b _ { 2 } , b _ { 3 } )$ with pairwise coprime $2 \leq b _ { 1 } < b _ { 2 } < b _ { 3 }$ satisfying $b _ { 1 } \leq 4 1 , b _ { 2 } \leq 9 9$ and $b _ { 3 } \leq 1 1 5$ . In particular a subset of manifolds $\Sigma ( b _ { 1 } , b _ { 2 } , b _ { 3 } )$ , will be used in the $S L _ { 2 } ( \mathbb { Z } )$ orbit learning process discussed in §6.2 and further specifics will be discussed therein.

Using the methods described above to find $\Sigma ( b _ { 1 } , \ldots , b _ { r } )$ for $3 \leq r \leq 6$ , we build the Star-Homology-Sphere dataset.

## 3.1.2 H-shaped Datasets

Next we turn to the datasets of H-shaped graphs, which are, as the name suggests, weighted graphs with two degree three vertices.

![](images/e9344c72d5c8af3d3bc978196ad95f4ffdc349176bd05c25431c48819da75f3c.jpg)  
Figure 3: A generic H-shaped plumbing graph.

H-shaped non-sphere dataset. The generic H-shaped dataset is generated from integer tuples $( p _ { 1 } , p _ { 2 } , p _ { 3 } , p _ { 4 } , \mathfrak { a } , \mathfrak { b } , w _ { c , 1 } , w _ { c , J } )$ satisfying the following constraints. We take

$$
2 \leq p _ { 1 } , p _ { 3 } \leq 9 , \quad p _ { 1 } + 1 \leq p _ { 2 } \leq 1 1 , \quad p _ { 3 } + 1 \leq p _ { 4 } \leq 1 1 ,
$$

with gcd $( p _ { 1 } , p _ { 2 } ) = \ * \ * { g c d } ( p _ { 3 } , p _ { 4 } ) = 1$ . The parameters a and b are chosen in the range $1 \leq { \mathfrak { a } } , { \mathfrak { b } } \leq 4 9$ subject to $\operatorname* { g c d } ( { \mathfrak { a } } , p _ { 1 } p _ { 2 } ) = 1$ and $\operatorname* { g c d } ( \mathfrak { b } , p _ { 3 } p _ { 4 } ) = 1$ , and the two central weights satisf $\mathrm { \check { y } } - 5 \le w _ { c , 1 } , w _ { c , J } \le - 2$ Finally, we impose the positivity condition ab $- p _ { 1 } p _ { 2 } p _ { 3 } p _ { 4 } > 0$

Given such a tuple, we construct an H-shaped plumbing tree of the form shown in Figure 3. The four arms are determined by the residues

$$
\beta _ { 1 } \equiv p _ { 2 } \ \mathfrak { a } \ \pmod p _ { 1 } ) , \quad \beta _ { 2 } \equiv p _ { 1 } \mathfrak { a } \ \pmod p _ { 2 } ) , \quad \beta _ { 3 } \equiv p _ { 4 } \mathfrak { b } \ \pmod p _ { 3 } ) , \quad \beta _ { 4 } \equiv p _ { 3 } \mathfrak { b } \ \pmod p _ { 4 } ) .
$$

For each $i = 1 , \ldots , 4 .$ , the weights along the $i ^ { \mathrm { t h } }$ arm are obtained from the negative continuedfraction expansion of ${ \cal { p } } _ { i } / \beta _ { i }$ . Thus, if

$$
\frac { p _ { i } } { \beta _ { i } } = [ - w _ { i , 1 } , . . . , - w _ { i , l _ { i } } ] ,
$$

then the corresponding arm carries vertex weights $w _ { i , 1 } , \ldots , w _ { i , l _ { i } }$

The two high-valence vertices carry weights $w _ { c , 1 }$ and $w _ { c , J }$ . Between them we insert a linear chain of $k - 2$ vertices, each of weight −2, where

$$
J = 2 + \operatorname* { m a x } \left\{ 1 , \left\lfloor \log \left( \mathfrak { a } \mathfrak { b } - p _ { 1 } p _ { 2 } p _ { 3 } p _ { 4 } + 1 \right) \right\rfloor \right\} .
$$

This produces an H-shaped plumbing tree with two Seifert-type ends joined by a variable-length central chain. We only retain plumbing trees which are in minimal form (under Kirby–Neumann moves) and whose plumbing matrices are strictly negative-definite and satisfy absolute determinant < 1000.

## H-shaped Homology Sphere Dataset from Splice Diagrams

As mentioned before, prior work has performed PCA analysis on a small dataset (20 instances) of Zb-invariants of homology spheres with H-shaped plumbing graphs [16]. This limitation arose from the difficulty of generating random integer matrices with determinant ±1, a necessary condition for the plumbed manifold to be an integral homology sphere. In this work, we overcome this bottleneck by leveraging an algorithm based on splice diagrams. This approach allows us to efficiently generate a large, diverse dataset of H-shaped homology spheres for ML analysis.

A splice diagram is a finite tree graph with finitely many vertices. It contains only vertices of degree 1 (referred to as leaves) or degree at least 3 (referred to as nodes). Furthermore, to each edge departing a node we associate an integer weight. In what follows, we use $( \nu , u )$ to denote the edge from vertex v to vertex u, and $w _ { \nu u }$ the corresponding weight in case v is a node.

For any edge connecting two nodes, we define the edge determinant as the product of the two weights on that edge minus the product of the weights on all other edges incident to those nodes. For the edge $( \nu _ { 1 } , \nu _ { 4 } )$ in Figure 4, the edge determinant is given by:

$$
\mathrm { D e t } _ { ( v _ { 1 } , v _ { 4 } ) } = w _ { v _ { 1 } v _ { 4 } } w _ { v _ { 4 } v _ { 1 } } - \left( \prod _ { u \in N ( v _ { 1 } ) \backslash \{ v _ { 4 } \} } w _ { v _ { 1 } u } \right) \left( \prod _ { z \in N ( v _ { 4 } ) \backslash \{ v _ { 1 } \} } w _ { v _ { 4 } z } \right) = w _ { v _ { 1 } v _ { 4 } } w _ { v _ { 4 } v _ { 1 } } - w _ { v _ { 4 } v _ { 5 } } w _ { v _ { 4 } v _ { 6 } } w _ { v _ { 1 } v _ { 2 } } w _ { v _ { 1 } v _ { 3 } } ,
$$

where N(v) denotes the set of neighbors of v.

The utility of splice diagrams lies in the following result: if a splice diagram satisfies (a) the weights around any node are positive and pairwise coprime, (b) the weight on any edge ending on a leaf is > 1, and (c) all edge determinants are positive, then there exists an explicit algorithm to convert the splice diagram into a plumbing graph for a negative-definite integral homology sphere. This algorithm, detailed in [32, Section 9], is reviewed in Appendix E.

![](images/9c89f56524d6e73a9b312bacf11e65af63d24c9a55bcb92e17cc3dab22e04f91.jpg)  
Figure 4: A splice diagram with two nodes and four leaves.

The H-shaped homology spheres in Tab. 1 were generated as follows. Let ${ \mathcal P } = \{ P _ { \alpha } \} _ { \alpha = 1 } ^ { \infty }$ denote the sequence of prime numbers $\{ 2 , 3 , 5 , \hdots \}$ . We initialize an H-shaped splice diagram as in Figure 4 and assign weights in two steps. First, for the central edge connecting the two nodes, we set $w _ { \nu _ { 1 } \nu _ { 4 } } = P _ { \alpha }$ and $w _ { \nu _ { 4 } \nu _ { 1 } } = P _ { \beta }$ , where the prime indices α and $\beta$ are sampled uniformly from integers in [11, 51] and [11, 31] respectively, subject to the constraint $\alpha \neq \beta$ . Next, for the external edges connecting a node v to a leaf $u ,$ we sample the weights $w _ { \nu u }$ uniformly from the set {2, 3, 5, 7, 11, 13}. To satisfy the conditions for the resulting manifold to be a homology sphere, we compute the edge determinant for each generated configuration and retain only those instances where it remains strictly positive. Finally, we apply the conversion algorithm to transform these valid splice diagrams into standard H-shaped plumbing graphs.

## 3.2 Data Representations

To ensure the robustness of our interpretation analysis, in our experiments we examine the impact of varying input formats on the learned features. In this subsection we introduce the different data representations used in this paper.

In this paper, we fix a cutoff, given by $N = 1 0 , 0 0 0 _ { \cdot }$ , in the $q \cdot$ -series expansion: we write an infinite q-series as

$$
q ^ { \Delta } \left( \sum _ { n = 0 } ^ { N - 1 } c _ { n } q ^ { n } + { \cal O } ( q ^ { 1 0 , 0 0 0 } ) \right) .
$$

The data is normalized by a multiplication of C-number such that the first coefficient is $c _ { 0 } = 1$

This fixed cutoff imposes a potentially significant limitation. For sufficiently sparse q-series, particularly those corresponding to star-shaped graphs, the first gap $\gamma _ { 0 }$ may exceed the cutoff $N$ . In extreme cases, the truncation discards all non-trivial coefficients, resulting in a degenerate input vector $[ 1 , 0 , \ldots , 0 ]$ . Without auxiliary features such as $\Delta$ to resolve these ambiguities, the resulting data collisions may confound the network, most notably in the binary classification of homology spheres. We discuss this and other potential issues posed by this data preprocessing in Section 4.

## 3.2.1 Full Sequence Representation

In the full sequence representation, we encode the input data as arrays of shape $( N , )$ , containing the coefficients $( c _ { 0 } , \ldots , c _ { N - 1 } )$ . The primary advantage of this representation is its simplicity: a single array of fixed shape fully captures the truncated data, where the array indices map directly to the q-series exponents.

## 3.2.2 Value-Index Pair Representation

In this representation, we compress the data by storing only the nonzero coefficients. Specifically, the input is defined as a sequence of tuples $( c _ { i } , n _ { i } )$ , where each tuple represents a nonvanishing term $c _ { i } q ^ { n _ { i } }$ (with $c _ { i } \neq 0 )$ . To ensure a uniform input dimension across the dataset, we identify the maximum sequence length and pad all shorter sequences to the right with zeros, resulting in a fixed dataspace dimension N = 10,474. This approach allows us to process inputs of varying sparsity.

## 3.2.3 Latent Space Representation

As an alternative to the direct sequence inputs described above, we investigate the impact of data compression through a learned, structured latent space. Specifically, we employ a network trained using contrastive learning to map the high-dimensional input space onto a lowdimensional (typically two-dimensional) manifold. This learned embedding allows us to examine whether the topological features driving the classification tasks can be condensed into a compact geometry without loss of interpretability. See Appendix A.3 for the details of the contrastive learning techniques used.

## 4 Topology Experiments

As illustrated in the data analysis pipeline schematic (Figure 1), the goal of our experiments is to extract interpretable topological features from the Zb-invariant q-series through a sequence of modular stages. Following the generation of raw data in §3, the preprocessing stage converts these series into one of three input representations: full sequence, value-index pair, or latent space representations.

Depending on the choice of representation, the pipeline proceeds along two distinct paths:

• Direct Classification (§4.2 & §4.3): The preprocessed full sequence or value-index pair vectors are fed directly into the classifier network.

• Latent Space Classification (§4.4): The data first passes through a Latent Space Learning module to generate low-dimensional embeddings, which then serve as inputs for the classifier.

The pipeline concludes with a dedicated interpretability phase. Here, we train a regression network to approximate the classifier’s decision values (logits). Crucially, this regressor does not see the raw q-series; instead, its inputs are restricted to a set of candidate features we selected based on their mathematical plausibility (see Glossary). By determining which of these humanunderstandable candidates are necessary to reconstruct the decision boundary, we isolate the specific topological information driving the classifier’s performance.

To validate the insights gained from this analysis, we include a set of control experiments in §4.5. In one experiment, we bypass the q-series inputs entirely and train classifiers directly on the derived topological features, allowing us to verify whether the explicitly learned decision boundaries align with those reconstructed by the regression pipeline. In another experiment, described in §4.2.3, we utilize a teacher–student (margin-reconstruction) protocol in which the teacher is trained on q-series data and student classifiers, trained only on the derived topological features, are asked to reproduce its decision boundary.

At the end of the pipeline, the data analysis stage assesses feature importance using three complementary metrics: Feature Saliency (FS), Permutation Importance (PI), and Shapley Additive

Explanations (SHAP). The patterns of consensus and discrepancies observed among these metrics constitute the primary findings of this section.

## 4.1 Task Description

In this section, we define the supervised learning tasks probing the global topological information encoded in the ${ \widehat { Z } } .$ -invariants: distinguishing homology spheres from non-spheres (HOM), and identifying the underlying plumbing graph topology (TOP). We selected these two classification tasks because they represent topological questions that are both theoretically significant and well-understood.

Homology Classification The Homology Classification (HOM) task evaluates the network’s capacity to distinguish integral homology spheres from non-spherical manifolds based on their truncated q-series. The ground truth labels separate manifolds satisfying | det M| = 1 (homology spheres) from those with | det $M | \neq 1$ , where M denotes the plumbing matrix of the plumbed manifold. This distinction is significant because it probes the network’s sensitivity to a global property that is often considered obfuscated in the coefficients of the $q \cdot$ -series invariant. To disentangle the influence of the plumbing graph structure from the homological properties, we first restrict the experiment to datasets with fixed graph topologies: HOM-Star (star-shaped graphs) and HOM-H (H-shaped graphs). We then assess the robustness of the learned features by performing the classification on the full, heterogeneous dataset (HOM-All).

Graph Topology Classification The Graph Topology Classification (TOP) task investigates whether the truncated ${ \widehat { Z } } .$ -invariant retains sufficient information to recover the underlying plumb ing graph structure. Specifically, we train the network to distinguish between Star-shaped and H-shaped graphs, testing its sensitivity to the complexity of the plumbing graph. To determine if the network’s learning strategy depends on the homological properties of the manifold, we partition the experiments into three distinct cases. The first two experiments restrict the dataset based on the homology class label: TOP-Sph (restricted to homology spheres) and TOP-Non (restricted to non-spheres). Finally, we perform the same classification experiment on the full, heterogeneous dataset (TOP-All).

Each experiment is conducted across all three input representations: full sequence, value-index pair, and latent space.

Multi-class Classification To probe the non-linear geometry of the latent space, we combine the graph topology and homology labels into a single four-class classification problem (HOM-Star, HOM-H, TOP-Sph, TOP-H). Unlike the binary tasks, which admitted linear solutions, distinguishing these four clusters requires a non-linear classifier. We therefore employ a RBF-Kernel Support Vector Machine (SVM) trained on the latent embeddings derived from the value-index pair representation, which yielded the highest performance.

To interpret these non-linear boundaries, we train a Random Forest (RF) regressor to reconstruct the SVM decision values using our standard set of topological features. We assess the fidelity of this reconstruction using the composite score defined in § A.1.

Experimental protocol and reporting. All results in this section are reported using the multiseed robustness protocol described in Appendix B. For each experiment we aggregate over classifier seeds, regressor seeds, and interpretability seeds, and report mean values with standard deviations. Classifier performance is measured by balanced accuracy score (BAS). The featureto-logit reconstruction estimator is evaluated by both $R ^ { 2 }$ and the Pearson correlation coefficient (PCC) on the train and validation splits: $R ^ { 2 }$ measures calibrated reconstruction of the scaled logits, while the PCC measures alignment with the learned logit geometry up to affine rescaling. Feature saliency, permutation importance, and SHAP are summarized by average feature rank across the interpretability ensemble.

The supervised classifier is trained on one of three truncated $\widehat { Z }$ representations: the full coefficient sequence, the serialized value-index representation, or a low-dimensional contrastive latent representation. In the full-sequence and value-index cases, the classifier is a fully connected ReLU network with one hidden layer of width $6 4 ,$ , trained with cross-entropy loss using Adam with learning rate $1 0 ^ { - 3 }$ , batch size 128, and early stopping on balanced accuracy. In the contrastive-learning case, the high-dimensional input is first mapped to a five-dimensional latent space by a ReLU encoder trained with semi-hard triplet loss; the downstream classifier is then trained on this latent representation using the same validation criterion, with hidden layer width 32.

For interpretability, we extract the pre-softmax logits of the trained classifier and use them as a dataset-level coordinate system for the learned decision geometry. A second ReLU network maps ten abstract features derived from the plumbing matrix and the q-series to the scaled classifier logits. This reconstruction estimator has architecture $( d , w ) = ( 4 , 1 2 8 )$ , where d is the number of affine layers and w is the hidden-layer width. It is trained with mean-squared error using Adam with learning rate $1 0 ^ { - 4 }$ , batch size $6 4 ,$ and an epoch cap of 3000. We use the validation-selected checkpoint for reporting and interpretation. The train metrics measure the signal strength of the ten-feature reconstruction of the learned logit geometry, while the validation metrics measure the stability of this reconstruction under the regressor split.

<table><tr><td rowspan=1 colspan=1>Stage</td><td rowspan=1 colspan=1>Input</td><td rowspan=1 colspan=1>Architecture</td><td rowspan=1 colspan=1>|Batch</td><td rowspan=1 colspan=1>Learning rate</td></tr><tr><td rowspan=1 colspan=1>Classifier</td><td rowspan=1 colspan=1>Full Sequence</td><td rowspan=1 colspan=1>(2,64)</td><td rowspan=1 colspan=1>128</td><td rowspan=1 colspan=1> $1 0 ^ { - 3 }$ </td></tr><tr><td rowspan=1 colspan=1>Classifier</td><td rowspan=1 colspan=1>Value-Index</td><td rowspan=1 colspan=1>(2,64)</td><td rowspan=1 colspan=1>128</td><td rowspan=1 colspan=1> $\overline { { 1 0 ^ { - 3 } } }$ </td></tr><tr><td rowspan=1 colspan=1>Contrastive encoder</td><td rowspan=1 colspan=1>Full Sequence</td><td rowspan=1 colspan=1>(4,128)</td><td rowspan=1 colspan=1>128</td><td rowspan=1 colspan=1> $\overline { { 1 0 ^ { - 3 } } }$ </td></tr><tr><td rowspan=1 colspan=1>CL classifier</td><td rowspan=1 colspan=1>Latent Space</td><td rowspan=1 colspan=1>(2,32)</td><td rowspan=1 colspan=1>128</td><td rowspan=1 colspan=1> $\overline { { 1 0 ^ { - 3 } } }$ </td></tr><tr><td rowspan=1 colspan=1>Regressor</td><td rowspan=1 colspan=1>Abstract features</td><td rowspan=1 colspan=1>(4,128)</td><td rowspan=1 colspan=1>64</td><td rowspan=1 colspan=1> $\overline { { 1 0 ^ { - 4 } } }$ </td></tr></table>

Table 3: Summary of the network architectures and hyperparameters used in the seed-robust topology and homology experiments. The architecture column records $( d , w )$ , where d is the number of affine layers and w is the hidden-layer width. Classifiers are trained with crossentropy loss and selected by balanced accuracy. The contrastive encoder is trained with semihard triplet loss, and the regressor is trained with mean-squared error to reconstruct the scaled classifier logits from the ten abstract features.

## 4.2 Full Sequence Representation

In this subsection, we evaluate the classification tasks defined in §4.1 using the full sequence representation. We present results for both the stratified experiments (e.g., HOM-Star, TOP-Sph) and the combined datasets (HOM-All, TOP-All).

## 4.2.1 Homology Classification

## HOM-Star

The HOM-Star dataset comprises 44,934 samples, consisting of 13,228 homology spheres and 31,706 non-spheres. Given the approximate 2.4:1 class imbalance, to remove any population effects we compared models trained with a resampled class-balanced dataset against those trained on the raw, imbalanced distribution. We observed no significant performance deviation; consequently, we report results derived from the full, imbalanced dataset.

<table><tr><td>Experiment</td><td>BAS</td><td>R2 一</td><td>PCC</td><td>一 FS</td><td>PI</td><td></td><td>SHAP</td></tr><tr><td>HOM-Star</td><td> $0 . 9 2 8 \pm 0 . 0 0 3$ </td><td> $\operatorname { t r } : 0 . 4 3 0 \pm 0 . 2 5 3$   $\mathrm { v a l } : 0 . 1 9 3 \pm 0 . 2 9 0$ </td><td> $\operatorname { t r } : 0 . 8 8 5 \pm 0 . 1 2 3$   $\mathrm { v a l } : 0 . 5 6 5 \pm 0 . 4 7 6$ </td><td> $\varepsilon _ { \operatorname* { m i n } _ { 1 . 8 } , \sigma _ { \gamma } \atop 4 . 1 } \Delta$ </td><td> $\varepsilon _ { \operatorname* { m i n } } \sigma _ { \varepsilon } \varepsilon _ { \operatorname* { m a x } }$ </td><td></td><td> $\varepsilon _ { \operatorname* { m i n } } \sigma _ { \varepsilon } \varepsilon _ { \operatorname* { m a x } }$ </td></tr><tr><td>HOM-H</td><td> $0 . 9 2 3 \pm 0 . 0 0 9$ </td><td> $\operatorname { t r } : 0 . 7 6 9 \pm 0 . 0 2 3$   $\mathrm { v a l } : 0 . 3 9 6 \pm 0 . 0 3 9$ </td><td> $\overline { { \operatorname { t r } : 0 . 8 7 9 \pm 0 . 0 1 4 } }$   $\mathrm { v a l } : 0 . 7 4 2 \pm 0 . 0 1 3$ </td><td> $\mu _ { \gamma } \ : \sigma _ { \gamma } \ : \varepsilon _ { \mathrm { m i n } }$ </td><td> $\sigma _ { \gamma } ~ \mu _ { \gamma } ~ \varepsilon _ { \mathrm { { m i n } } }$  1.7&#x27;3.4’3.8</td><td></td><td> $\operatorname { \sigma } _ { \gamma } , \varepsilon _ { \mathrm { m i n } } , \mu _ { \gamma }$ </td></tr><tr><td>HOM-All</td><td> $0 . 9 5 2 \pm 0 . 0 0 3$ </td><td> $\mathrm { t r } : 0 . 6 9 8 \pm 0 . 0 3 5$   $\mathrm { v a l } : 0 . 5 7 2 \pm 0 . 0 2 9$ </td><td> $\mathbf { \overline { { t r } } } : 0 . 8 5 2 \pm 0 . 0 2 5$   $\mathrm { v a l } : 0 . 7 9 8 \pm 0 . 0 2 6$ </td><td> $\operatorname* { d e t } _ { 1 . 5 } \varepsilon _ { \mathrm { m i n } } c _ { \mathrm { m a x } }$ </td><td> $_ { 2 . 5 } \sigma _ { \varepsilon } \mathrm { , } \varepsilon _ { \mathrm { m a x } } \mathrm { , } \varepsilon _ { \mathrm { m i n } }$ </td><td></td><td>εmin det σγ  $_ { 1 . 4 } ^ { \mathrm { ~ m i n } } , _ { 2 . 0 } , _ { 3 . 1 }$ </td></tr><tr><td> $\mathrm { T O P - } S \mathrm { p h }$ </td><td> $0 . 9 6 4 \pm 0 . 0 0 4$ </td><td> $\mathbf { \bar { t r } } : 0 . 8 3 7 \pm 0 . 0 1 2$   $\mathrm { v a l } : 0 . 7 2 5 \pm 0 . 0 1 7$ </td><td> $\operatorname { t r } : 0 . 9 1 6 \pm 0 . 0 0 7$   $\mathsf { v a l } : 0 . 8 6 3 \pm 0 . 0 1 0$ </td><td> $\mu _ { \gamma } \ : c _ { \mathrm { m a x } } \ : \sigma _ { \gamma }$  2.5 3.6</td><td> $\mu _ { \gamma } ~ \sigma _ { \gamma } ~ \Delta$   $_ { 1 . 0 } \dot { ^ { 2 } } \dot { \ } _ { 2 . 0 } \dot { ^ { 2 } } \ 3 . 4$ </td><td> $\mu _ { \gamma } \textit { \textbf { \sigma } } _ { \gamma }$ </td><td>det 1.0&#x27;2.0’3.6</td></tr><tr><td>TOP-Non</td><td> $0 . 9 7 1 \pm 0 . 0 0 9$ </td><td> $\operatorname { t r } : 0 . 7 5 9 \pm 0 . 0 1 9$   $\mathrm { v a l } : 0 . 6 0 0 \pm 0 . 0 2 9$ </td><td> $\overline { { \mathrm { t r } : 0 . 8 7 6 \pm 0 . 0 1 0 } }$   $\mathsf { v a l } : 0 . 8 1 9 \pm 0 . 0 1 2$ </td><td> $\operatorname * { d e t } _ { 1 . 0 } , \varepsilon _ { \mathrm { m i n } } , \sigma _ { \gamma }$ </td><td> $\varepsilon _ { \mathrm { m i n } } \operatorname* { d e t } _ { 2 . 1 } { \mu _ { \varepsilon } }$ </td><td></td><td> $\begin{array} { l c r } { { \operatorname { d e t } \ \varepsilon _ { \mathrm { m i n } } \ \sigma _ { \gamma } } } \\ { { 1 . 0 \ ^ { 3 } \ \mathrm { ~ } 2 . 4 \ ^ { 2 } \ \mathrm { ~ } 3 . 8 } } \end{array}$ </td></tr><tr><td>TOP-All</td><td> $0 . 9 7 8 \pm 0 . 0 0 4$ </td><td> $\operatorname { t r } : 0 . 7 6 1 \pm 0 . 0 1 8$   $\mathrm { v a l } : 0 . 6 1 6 \pm 0 . 0 1 3$ </td><td> $\overline { { \mathrm { t r } : 0 . 8 7 6 \pm 0 . 0 1 0 } }$   $\mathsf { v a l } : 0 . 8 1 6 \pm 0 . 0 1 0$ </td><td> $\begin{array} { r l } { \operatorname* { d e t } \ \varepsilon _ { \mathrm { m i n } } \ \sigma _ { \gamma } } & { { } } \\ { 1 . 0 \ ^ { 3 } \ \varepsilon _ { 2 . 2 } \ \cdot \ 3 . 0 } \end{array}$ </td><td> $\varepsilon _ { \operatorname* { m i n } } ~ \sigma _ { \gamma } ~ \sigma _ { \varepsilon }$ </td><td></td><td> ${ \sigma _ { \gamma } } ^ { \mathrm { ~ } } , { \varepsilon _ { \mathrm { { m i n } } } } ^ { \mathrm { ~ d e t } }$ </td></tr></table>

Table 4: Multi-seed full-sequence results for the topology and homology experiments. BAS is reported as mean ± standard deviation over classifier seeds. The $R ^ { 2 }$ and Pearson correlation (PCC) columns report train and validation reconstruction quality of the feature-to-logit regressor, stacked within each cell and averaged over the classifier–regressor seed ensemble. The final columns report the three lowest mean feature ranks for feature saliency (FS), permutation importance (PI), and SHAP, where the subscript gives the average rank over the normalized interpretability ensemble.

Naively, this classification task appears non-trivial because the q-series for homology spheres and non-spheres often share identical asymptotic growth. This is particularly apparent for weakly negative plumbed manifolds with three singular fibers in our dataset, as all of them have q-series invariants with coefficients $c _ { 0 } , c _ { 1 } , \ldots$ . that are either <sup>−</sup>1, +1, or 0. While one might expect the network to rely on explicit features like coefficient gaps to resolve this ambiguity, our subsequent analysis reveals that it exploits more abstract derived features.

We trained the classifier using the architecture specified in the first row of Tab. 3, with a 90/10 train-test split. The model achieved a mean Balanced Accuracy Score (BAS) of $0 . 9 2 8 \pm 0 . 0 0 3$ with the confusion matrix shown in Figure 13. Figure 14 visualizes the classifier’s output logits for a single representative seed; these values serve as the target variables for the regressionbased interpretability analysis discussed below.

Discussion of Features Mathematically, integral homology spheres are defined by the condition | det $M | = 1$ . However, the non-vanishing variance observed along the principal axis of the classifier logits suggests that the network does not rely on this discrete invariant alone; if it did, the homology sphere samples would collapse to a single point rather than spanning a segment. To isolate the actual decision drivers, we trained a regression network on the full set of 10 interpretable features, achieving a mean fit of

$$
( R _ { \mathrm { t r } } ^ { 2 } , R _ { \mathrm { v a l } } ^ { 2 } , \mathrm { P C C _ { \mathrm { t r } } , \mathrm { P C C _ { \mathrm { v a l } } } } ) = ( 0 . 4 3 0 \pm 0 . 2 5 3 , 0 . 1 9 3 \pm 0 . 2 9 0 , 0 . 8 8 5 \pm 0 . 1 2 3 , 0 . 5 6 5 \pm 0 . 4 7 6 ) .
$$

These performance metrics suggest that the classifier is robust, but the ten-feature logit reconstruction is weak and high variance. The feature rankings obtained from the regressor network alone are thus only suggestive, but will be corroborated by further experiments.

The interpretability analysis of this regressor reveals a divergence between metrics. Beyond the minimum magnitude eigenvalue $\varepsilon _ { \mathrm { { m i n } } } ,$ Feature Saliency (FS) prioritizes spectral information about the gaps between non-trivial terms in the q-series, i.e. their standard deviation $\sigma _ { \gamma } ,$ and the unseen leading power $\Delta _ { z }$ , while the statistical metrics (PI and SHAP) consistently identify spectral properties—specifically $\varepsilon _ { \mathrm { { m i n } } }$ , the standard deviation of the eigenvalues $\sigma _ { \varepsilon }$ , and maximum magnitude eigenvalue $\varepsilon _ { \mathrm { m a x } } .$ —as the dominant predictors.

The prominence of $\varepsilon _ { \mathrm { m i n } }$ is notable and allows for a clear explanation: for the plumbing matrices in our dataset, we observe that the condition $| \mathsf { d e t } M | = 1$ is typically achieved by a spectrum where one eigenvalue is significantly smaller than the others. This empirical observation can be placed on rigorous footing. By combining Cramer’s rule for the diagonal entries of $M ^ { - 1 }$ with the Rayleigh quotient bound on the spectral radius, one obtains a strict upper bound on $| \varepsilon _ { \mathrm { m i n } } |$ in terms of the Seifert data (see Appendix C for the full derivation):

Lemma 1 Let M be the plumbing matrix of a three-legged star-shaped graph associated to a Seifert manifold with singular fiber orders $b _ { 1 } , b _ { 2 } , b _ { 3 }$ . Then

$$
\vert \varepsilon _ { \mathrm { m i n } } \vert \leq \frac { \vert \operatorname* { d e t } M \vert } { b _ { 1 } b _ { 2 } b _ { 3 } } .\tag{9}
$$

For Brieskorn spheres, | det $M | = 1$ and the $b _ { i }$ are pairwise coprime integers $\geq 2$ , so $b _ { 1 } b _ { 2 } b _ { 3 } \geq 3 0$ forcing $\vert \varepsilon _ { \mathrm { m i n } } \vert \leq 1 / 3 0 $ —typically much smaller for generic Seifert homology spheres. The bound thus provides a structural explanation for the network’s reliance on $\varepsilon _ { \mathrm { { m i n } } } \mathrm { { : } }$ rather than learning the determinant directly, the network exploits a spectral consequence of the homology sphere condition that is guaranteed by the underlying geometry.

The caveat is that the relatively poor average performance of the regressor networks places any definitive claim that the full-sequence networks robustly learn $\varepsilon _ { \mathrm { m i n } }$ as a meaningful surrogate discriminating feature on weak footing. We will see in further experiments with much stronger regressor signals that $\varepsilon _ { \mathrm { { m i n } } }$ is frequently identified as a leading regressor surrogate to explain the decision boundary separating homology spheres from non-spheres.

The secondary feature, $\sigma _ { \varepsilon } ,$ captures the spread of the spectrum. In particular, neither $\varepsilon _ { \mathrm { { m i n } } }$ nor $\sigma _ { \varepsilon }$ directly captures the determinant. This indicates that the network solves the classification task by augmenting the primary proxy $( \varepsilon _ { \mathrm { m i n } } )$ with independent spectral information $( \sigma _ { \varepsilon } )$ , rather than using the determinant or relying solely on a direct proxy for the determinant.

This reliance on eigenvalue statistics highlights a tension between machine learning efficacy and topological invariance. Unlike the determinant, individual eigenvalues are not invariant under Kirby–Neumann moves; as exemplified in Figure 5, equivalent plumbing graphs can yield distinct spectra, while still satisfying the above lemma. The network effectively ignores the invariant determinant in favor of spectral data, identifying a statistical shortcut that is valid for most, but not all, of the dataset. This suggests the model is exploiting distribution-specific correlations rather than learning the exact topological definition, resulting in high but imperfect classification performance and failure to learn the crucial invariance property.

In particular, it is important to note that the spectral bound of Lemma 1 is one-directional: it guarantees that Brieskorn spheres must have a small $| \varepsilon _ { \mathrm { m i n } } |$ , but does not exclude the possibility that certain non-spheres also possess a small minimum-magnitude eigenvalue. This asymmetry—where the network mistakes a necessary condition for being a sphere for a sufficient one—provides a structural explanation for the observed generalization error of the classifier: the network’s reliance on $\varepsilon _ { \mathrm { m i n } }$ as a proxy is mathematically well-motivated but inherently incomplete.

## HOM-H.

The dataset for the HOM-H experiment consists of plumbed manifolds defined by H-shaped plumbing graphs. It contains 18,669 samples, comprising 7,952 homology spheres and $1 0 { , } 7 1 7$

![](images/46f44a3e5433b0368615109b87a8b583eda31e387a80df145cd30e3f93ddf2a5.jpg)  
Figure 5: Two plumbings of Σ(2, 3, 5) equivalent under Kirby–Neumann moves.

non-spheres.

We trained the classifier detailed in Tab. 3 using a 90/10 train-test split, achieving a mean BAS of $0 . 9 2 3 \pm 0 . 0 0 9$ on the test set. The resulting confusion matrix and representative single-seed logit distribution are visualized in Figs. 13 and 14, respectively. To interpret these decisions, we trained a regression network to reconstruct the classifier logits from the full set of ten derived features. The regressor achieves a substantially more stable fit than in HOM-Star, with

$$
( R _ { \mathrm { t r } } ^ { 2 } , R _ { \mathrm { v a l } } ^ { 2 } , \mathrm { P C C _ { \mathrm { t r } } , \mathrm { P C C _ { \mathrm { v a l } } } } ) = ( 0 . 7 6 9 \pm 0 . 0 2 3 , 0 . 3 9 6 \pm 0 . 0 3 9 , 0 . 8 7 9 \pm 0 . 0 1 4 , 0 . 7 4 2 \pm 0 . 0 1 3 ) .
$$

Thus, while the abstract features do not fully reconstruct the calibrated two-logit geometry, they preserve a stable monotone relation with the classifier’s decision function.

Although the classification objective is the same as in HOM-Star—distinguishing homology spheres from non-spheres—the H-graph sector changes the feature geometry of the problem. In the star sector, the leading recurring feature is $\varepsilon _ { \mathrm { { m i n } } }$ , suggesting a low-end spectral proxy for the determinant condition. In the H-graph sector, the attribution pattern shifts toward gap statistics. The full-sequence interpretability metrics in Tab. 4 consistently rank $\mu _ { \gamma } , \sigma _ { \gamma } ,$ and $\varepsilon _ { \mathrm { m i n } }$ among the leading features. Feature saliency ranks $\mu _ { \gamma }$ and $\sigma _ { \gamma }$ first and second, with average ranks 1.3 and 1.7, followed by $\varepsilon _ { \mathrm { m i n } }$ at rank 3.5. Permutation importance places $\sigma _ { \gamma }$ first and retains $\mu _ { \gamma }$ and $\varepsilon _ { \mathrm { m i n } }$ among the top three. SHAP similarly keeps the same trio at the top, with $\sigma _ { \gamma } , \varepsilon _ { \mathrm { { m i n } } } ,$ and $\mu _ { \gamma }$ all closely ranked.

This indicates that the HOM-H classifier does not rely on a single low-end eigenvalue proxy. Instead, it recruits gap statistics to resolve examples whose coarse spectral behavior is ambiguous. The H-graph dataset contains non-spherical examples whose low-end spectrum can partially resemble that of homology spheres, so $\varepsilon _ { \mathrm { m i n } }$ alone is not a sufficiently stable discriminator. The gap statistics provide complementary information about the sparsity profile of the q-series, allowing the classifier to separate examples whose eigenvalue summaries are similar but whose coefficient support differs.

Geometrically, this suggests that in the H-graph sector the homology-sphere decision is mediated not only by the determinant constraint, but also by how the quadratic form $\ell ^ { T } M ^ { - 1 } \ell$ organizes the support of the $q \mathrm { . }$ series. Changes in the spectrum of M, especially near small eigenvalues, affect the directions along which this quadratic form grows slowly. The resulting changes in coefficient spacing and support density are summarized by the gap features $\mu _ { \gamma }$ and $\sigma _ { \gamma }$ . The interpretability results therefore point to a mixed spectral–sparsity mechanism: $\varepsilon _ { \mathrm { { m i n } } }$ remains relevant as a low-end spectral feature, but the more stable HOM-H signal is carried by the gap profile of the q-series.

## HOM-All.

Finally, we generalize the homology-sphere task by training on the combined heterogeneous dataset, asking the network to identify homology spheres without first conditioning on the plumbing graph topology. The classifier achieves a mean BAS of $0 . 9 5 2 { \scriptstyle \pm 0 . 0 0 3 }$ , improving over both HOM-Star and HOM-H individually. A representative single-seed logit distribution for the combined dataset is shown in Fig. 14. This performance is notable because the combined dataset contains graph-topology heterogeneity: the classifier must learn a homology-sphere decision rule that is stable across both star-shaped and H-shaped plumbing graphs.

The logit-regression analysis indicates that this heterogeneous task is substantially better captured by the abstract feature dictionary than either HOM-Star or HOM-H alone. The regressor achieves

$$
( R _ { \mathrm { t r } } ^ { 2 } , R _ { \mathrm { v a l } } ^ { 2 } , \mathrm { P C C _ { \mathrm { t r } } , \mathrm { P C C _ { \mathrm { v a l } } } } ) = ( 0 . 6 9 8 \pm 0 . 0 3 5 , 0 . 5 7 2 \pm 0 . 0 2 9 , 0 . 8 5 2 \pm 0 . 0 2 5 , 0 . 7 9 8 \pm 0 . 0 2 6 ) .
$$

Thus, in the pooled homology task, the derived spectral and support features recover a meaningful fraction of the classifier’s two-logit geometry, not merely its hard labels.

The feature rankings in Tab. 4 show that the combined task does not reduce to a single spectral proxy. Feature saliency ranks det first, followed by $\varepsilon _ { \mathrm { { m i n } } }$ and $c _ { \mathrm { m a x } }$ . SHAP similarly ranks $\varepsilon _ { \mathrm { { m i n } } }$ and det among the leading features, with $\sigma _ { \gamma }$ also appearing near the top. Permutation importance emphasizes eigenvalue dispersion and scale, ranking $\sigma _ { \varepsilon } , \varepsilon _ { \mathrm { m a x } } .$ , and $\varepsilon _ { \mathrm { { m i n } } }$ highest. Taken together, the attribution methods suggest that the classifier combines the explicit determinant signal with low-end spectral information and distributional eigenvalue statistics.

This differs from the individual HOM-Star and HOM-H sectors. HOM-Star is dominated by low-end eigenvalue information but is poorly reconstructed by the ten-feature logit regressor, indicating that the present abstract dictionary captures only part of the star-sector decision geometry. HOM-H has a more stable logit reconstruction, but its leading attributions shift toward gap statistics $( \mu _ { \gamma } , \sigma _ { \gamma } )$ together with $\varepsilon _ { \mathrm { { m i n } } }$ . In HOM-All, the pooled task stabilizes around features that are robust across both graph families: det supplies the direct homological invariant, while $\varepsilon _ { \operatorname* { m i n } } , \sigma _ { \varepsilon }$ , and $\varepsilon _ { \mathrm { m a x } }$ supply spectral proxies that remain informative when the graph topology varies. The appearance of $\sigma _ { \gamma }$ in SHAP further indicates that support-gap information still contributes to the combined decision, but it is no longer the sole organizing feature.

We therefore interpret HOM-All as evidence that the full-sequence classifier learns a hybrid determinant–spectral rule. The determinant provides the coarse homology-sphere signal, while low-end and dispersion statistics of the spectrum help resolve topology-dependent confounders across the star and H-graph sectors. The improved validation $R ^ { 2 }$ and PCC indicate that this hybrid rule is more faithfully represented in the ten-feature dictionary than the corresponding sector-specific HOM-Star rule.

## 4.2.2 Graph Topology Classification

We now turn from homology-sphere detection to a different classification objective: identifying the underlying plumbing graph topology from the $q \cdot$ series. In these experiments the labels distinguish star-shaped and H-shaped plumbing graphs. The results show that the classifier again achieves high balanced accuracy, but the features used to reconstruct its logits differ from those in the HOM experiments. In particular, the topology classifiers use a sector-dependent mixture of gap statistics, determinant information, and low-end spectral data.

TOP-Sph. We first isolate the homology-sphere subset, containing N = 21,180 samples, and ask whether the network can distinguish the underlying plumbing graph topology—star-shaped versus H-shaped—solely from the q-series. The classifier achieves a high and stable mean BAS of $0 . 9 6 4 \pm 0 . 0 0 4$ . The corresponding confusion matrix in Fig. 13 shows a mild asymmetry, with a slight tendency to classify some H-graphs as star-shaped graphs.

The logit-regression stage gives one of the clearest interpretable reconstructions in the fullsequence experiments:

$$
\begin{array} { r } { ( R _ { \mathrm { t r } } ^ { 2 } , R _ { \mathrm { v a l } } ^ { 2 } , \mathrm { P C C } _ { \mathrm { t r } } , \mathrm { P C C } _ { \mathrm { v a l } } ) = ( 0 . 8 3 7 \pm 0 . 0 1 2 , 0 . 7 2 5 \pm 0 . 0 1 7 , 0 . 9 1 6 \pm 0 . 0 0 7 , 0 . 8 6 3 \pm 0 . 0 1 0 ) . } \end{array}
$$

Thus, for homology spheres, the ten-dimensional abstract feature dictionary reconstructs a substantial fraction of the teacher’s two-logit geometry.

The attribution results show a decisive shift away from the determinant/eigenvalue proxies emphasized in the homology classification tasks. In TOP-Sph, the leading feature is the mean gap $\mu _ { \gamma }$ , which is ranked first by FS, PI, and SHAP. The next stable signal is $\sigma _ { \gamma } .$ , which PI and SHAP rank second, while FS also includes $c _ { \mathrm { m a x } }$ and $\sigma _ { \gamma }$ among the top-ranked features. This indicates that, once | det $M | = 1$ is fixed, the graph-topology decision is primarily encoded in the sparsity and spacing structure of the q-series rather than in determinant variation.

We interpret this reliance on gap statistics through the analytic structure of the $\widehat { Z }$ expansion. The exponents are governed by values of the quadratic form $\ell ^ { T } M ^ { - 1 } \ell$ for lattice vectors $\ell ,$ as in (1). Since all samples in TOP-Sph are homology spheres, determinant information cannot distinguish the two graph topologies. Instead, differences in the spectrum and anisotropy of $M ^ { - 1 }$ can change the rate at which $\bar { \ell ^ { T } } M ^ { - 1 } \ell$ grows along different lattice directions. This changes the spacing and density of accessible exponents in the q-series. The strong ranking of $\mu _ { \gamma }$ and $\sigma _ { \gamma } s \mathrm { u g . }$ gests that the full-sequence classifier detects topology through this induced gap profile.

TOP-Non. We next restrict to non-homology spheres and again distinguish star-shaped from H-shaped plumbing graphs. The classifier achieves a high, stable mean BAS of $0 . 9 7 1 \pm 0 . 0 0 9$ The logit-regression reconstruction remains strong, although weaker than in TOP-Sph:

$$
( R _ { \mathrm { t r } } ^ { 2 } , R _ { \mathrm { v a l } } ^ { 2 } , \mathrm { P C C _ { \mathrm { t r } } , \mathrm { P C C _ { \mathrm { v a l } } } } ) = ( 0 . 7 5 9 \pm 0 . 0 1 9 , 0 . 6 0 0 \pm 0 . 0 2 9 , 0 . 8 7 6 \pm 0 . 0 1 0 , 0 . 8 1 9 \pm 0 . 0 1 2 ) .
$$

The attribution pattern changes substantially from TOP-Sph. In the homology-sphere sector, the fixed determinant forces the classifier to rely on gap statistics. In the non-spherical sector, determinant variation becomes available again, and the classifier uses a hybrid spectral signature. Feature saliency ranks det first, followed by $\varepsilon _ { \mathrm { { m i n } } }$ and $\sigma _ { \gamma }$ . Permutation importance instead places $\varepsilon _ { \mathrm { m i n } }$ first, followed by det and $\mu _ { \varepsilon }$ . SHAP again ranks det first and $\varepsilon _ { \mathrm { { m i n } } }$ second, with $\sigma _ { \gamma }$ also appearing among the leading features. Across methods, the stable signal is therefore not a pure gap statistic: it is the combination of determinant information, low-end spectral information, and a secondary gap contribution.

This should be interpreted as a sector-dependent shortcut rather than a failure mode. In TOP-Sph, topology must be inferred from the q-series spacing structure because the determinant is fixed. In TOP-Non, the graph topology is correlated with broader spectral properties of the plumbing matrix, including det and $\varepsilon _ { \mathrm { { m i n } } }$ . The classifier exploits these available invariants, while still retaining some sensitivity to gap structure through $\sigma _ { \gamma }$ . Thus the non-spherical topology decision combines coarse determinant information with low-end spectral data and residual sparsity information.

TOP-All. Finally, we train on the combined topology dataset, asking the classifier to distinguish the plumbing graph topology without filtering by homology class. The classifier achieves the strongest topology performance, with mean $\mathrm { B A S } 0 . 9 7 8 { \pm } 0 . 0 0 4$ . The confusion matrix in Fig. 13 and the representative logit distribution in Fig. 14 show performance consistent with the high accuracy observed in the sector-specific topology experiments.

The logit-regression stage reflects the mixture of signals seen in TOP-Sph and TOP-Non:

$$
( R _ { \mathrm { t r } } ^ { 2 } , R _ { \mathrm { v a l } } ^ { 2 } , \mathrm { P C C _ { \mathrm { t r } } , \mathrm { P C C _ { \mathrm { v a l } } } } ) = ( 0 . 7 6 1 \pm 0 . 0 1 8 , 0 . 6 1 6 \pm 0 . 0 1 3 , 0 . 8 7 6 \pm 0 . 0 1 0 , 0 . 8 1 6 \pm 0 . 0 1 0 ) .
$$

The feature rankings confirm that the pooled topology classifier does not rely on a single mechanism. Feature saliency ranks det first, followed by $\varepsilon _ { \mathrm { { m i n } } }$ and $\sigma _ { \gamma }$ . Permutation importance ranks $\varepsilon _ { \operatorname* { m i n } } , \sigma _ { \gamma }$ , and $\sigma _ { \varepsilon }$ among the leading features. SHAP ranks $\sigma _ { \gamma } , \varepsilon _ { \mathrm { { m i n } } } ,$ and det highest. Thus the combined topology task retains both components: determinant and low-end eigenvalue information inherited from the non-spherical sector, together with gap statistics inherited from the homology-sphere topology task.

We therefore interpret TOP-All as a hybrid topology classifier. In the homology-sphere sector, topology is most cleanly expressed through the gap profile of the q-series, especially $\mu _ { \gamma }$ and $\sigma _ { \gamma }$ In the non-spherical sector, determinant variation and low-end spectral information become available and provide strong graph-topology proxies. When both sectors are pooled, the fullsequence classifier combines these mechanisms: det and $\varepsilon _ { \mathrm { m i n } }$ capture coarse spectral differences between graph families, while $\sigma _ { \gamma }$ preserves the sparsity/gap signal that remains informative even after the homology class is no longer fixed.

## 4.2.3 Discussion of Full Sequence Results

Synthesizing the results across the full sequence representation experiments, distinct patterns emerge regarding the network’s learning strategy and the interpretability of its decisions.

<table><tr><td rowspan=1 colspan=1>Feature</td><td rowspan=1 colspan=1>Skewness</td><td rowspan=1 colspan=1>Kurtosis</td></tr><tr><td rowspan=1 colspan=1>det</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>17.73</td></tr><tr><td rowspan=1 colspan=1> $\varepsilon _ { \mathrm { m i n } }$ </td><td rowspan=1 colspan=1>1.65</td><td rowspan=1 colspan=1>9.98</td></tr><tr><td rowspan=1 colspan=1> $\varepsilon _ { \mathrm { m a x } }$ </td><td rowspan=1 colspan=1>5.15</td><td rowspan=1 colspan=1>45.54</td></tr><tr><td rowspan=1 colspan=1> $\mu _ { \varepsilon }$ </td><td rowspan=1 colspan=1>2.01</td><td rowspan=1 colspan=1>10.48</td></tr><tr><td rowspan=1 colspan=1> $\sigma _ { \varepsilon }$ </td><td rowspan=1 colspan=1>5.40</td><td rowspan=1 colspan=1>50.47</td></tr><tr><td rowspan=1 colspan=1> $\Delta$ </td><td rowspan=1 colspan=1>10.93</td><td rowspan=1 colspan=1>148.67</td></tr><tr><td rowspan=1 colspan=1> $\gamma _ { 0 }$ </td><td rowspan=1 colspan=1>7.58</td><td rowspan=1 colspan=1>77.67</td></tr><tr><td rowspan=1 colspan=1> $\mu _ { \gamma }$ </td><td rowspan=1 colspan=1>3.90</td><td rowspan=1 colspan=1>27.89</td></tr><tr><td rowspan=1 colspan=1> $\sigma _ { \gamma }$ </td><td rowspan=1 colspan=1>2.64</td><td rowspan=1 colspan=1>13.35</td></tr><tr><td rowspan=1 colspan=1> $c _ { \mathrm { m a x } }$ </td><td rowspan=1 colspan=1>49.44</td><td rowspan=1 colspan=1>3052.40</td></tr></table>

Table 5: Higher moments of the ten-feature distribution. Skewness and kurtosis are computed on the raw feature matrix used in the topology experiments, before the interpretability scores are ranked. Large values indicate heavy-tailed or strongly non-normal feature marginals.

Sensitivity vs. Relevance. A consistent contrast emerges between interpretability metrics measuring local sensitivity and those measuring global relevance. FS measures the local gradient of the model output with respect to the standardized input features, whereas PI and SHAP probe the effect of feature perturbations or feature coalitions on predictive performance. Consequently, FS should be interpreted as a local sensitivity diagnostic rather than a direct measure of global explanatory power.

In our experiments, this distinction is most visible in the behavior of $c _ { \mathrm { m a x } }$ . FS assigns relatively high importance to $c _ { \mathrm { m a x } }$ in cases where PI and SHAP do not give it comparable scores and where direct ablation or single-feature predictive tests do not identify it as a comparably strong explanatory feature. We therefore do not interpret the FS ranking alone as evidence, in those cases, that $c _ { \mathrm { m a x } }$ controls the decision of the classifier network. Rather, the discrepancy indicates that FS is more prone than PI or SHAP to select features that induce local output sensitivity without necessarily carrying global predictive relevance.

However, this should not be read as a claim that FS always selects high-skewness or high-kurtosis features<sup>2</sup>. The feature-moment analysis in Tab. 5 shows that the situation is more nuanced. For example, $\mu _ { \gamma }$ is not distinguished by unusually large moments compared with other inputs. The robust empirical statement is instead that, among the three interpretability metrics, FS is the most local and hence the most susceptible to selection effects. PI and SHAP provide more stable characterization of global relevance, and in the main experiments, they point away from raw coefficient scale features toward spectral invariants of the plumbing matrix or gap data.

We quantified this effect by computing a bias score,

$$
\mathrm { b i a s } = \frac { 1 } { 2 } \left( \rho _ { \mathrm { s k e w } } + \rho _ { \mathrm { k u r t } } \right) ,\tag{10}
$$

obtained by averaging the Spearman rank correlations $( \rho _ { \mathrm { s k e w } } , \rho _ { \mathrm { k u r t } } )$ between a metric’s featureimportance magnitudes and the feature |skewness| and kurtosis values, so that positive scores indicate a preference for high-moment features. This analysis reveals that all three metrics are on average anti-biased (TOP-Sph excepted), preferring features with stable marginals (Tab. 6). The contrast is therefore relative: FS is consistently the least anti-biased, attaining the highest bias score in five of the six experiments, while SHAP is the most strongly anti-biased.

<table><tr><td rowspan=1 colspan=1>Experiment</td><td rowspan=1 colspan=1>FS        —</td><td rowspan=1 colspan=1>PI</td><td rowspan=1 colspan=1>SHAP</td></tr><tr><td rowspan=1 colspan=1>HOM-Star</td><td rowspan=1 colspan=1> $- 0 . 2 0 0 \pm 0 . 4 1 6$ </td><td rowspan=1 colspan=1> $- 0 . 3 1 5 \pm 0 . 1 8 4$ </td><td rowspan=1 colspan=1> $- 0 . 4 2 5 \pm 0 . 1 9 0$ </td></tr><tr><td rowspan=1 colspan=1>HOM-H</td><td rowspan=1 colspan=1> $- 0 . 4 8 1 \pm 0 . 1 0 0$ </td><td rowspan=1 colspan=1> $- 0 . 4 5 6 \pm 0 . 1 3 7$ </td><td rowspan=1 colspan=1> $\overline { { - 0 . 5 7 7 \pm 0 . 1 1 7 } }$ 1</td></tr><tr><td rowspan=1 colspan=1>HOM-All</td><td rowspan=1 colspan=1> $\overline { { - 0 . 1 9 4 \pm 0 . 1 5 7 } }$ </td><td rowspan=1 colspan=1> $- 0 . 5 0 5 \pm 0 . 1 4 2$ </td><td rowspan=1 colspan=1> $- 0 . 7 0 9 \pm 0 . 1 1 5$ 一</td></tr><tr><td rowspan=1 colspan=1>TOP-Sph</td><td rowspan=1 colspan=1> $\overline { { 0 . 3 0 4 \pm 0 . 2 0 0 } }$ </td><td rowspan=1 colspan=1> $\overline { { 0 . 2 0 1 \pm 0 . 1 2 2 } }$ </td><td rowspan=1 colspan=1> $- 0 . 0 1 7 \pm 0 . 1 7 6$ 1</td></tr><tr><td rowspan=1 colspan=1>TOP-Non</td><td rowspan=1 colspan=1> $\overline { { - 0 . 3 9 1 \pm 0 . 2 5 3 } }$ </td><td rowspan=1 colspan=1> $- 0 . 7 6 9 \pm 0 . 1 1 1$ </td><td rowspan=1 colspan=1> $- 0 . 7 6 1 \pm 0 . 0 5 8$ 1</td></tr><tr><td rowspan=1 colspan=1>TOP-All</td><td rowspan=1 colspan=1> $- 0 . 3 6 2 \pm 0 . 0 6 4$ </td><td rowspan=1 colspan=1> $- 0 . 6 6 6 \pm 0 . 1 7 4$ T</td><td rowspan=1 colspan=1> $- 0 . 7 6 9 \pm 0 . 0 2 8$ 1</td></tr></table>

Table 6: Multi-seed higher-moment bias scores for the full-sequence interpretability metrics. Entries are reported as mean ± standard deviation over the classifier–regressor–interpretability seed ensemble. Positive values indicate that an attribution method preferentially ranks features with large skewness or kurtosis, while negative values indicate preference for features with more stable marginals. Scores are computed via (10) from Spearman correlations between each metric’s feature-importance vector and the feature-moment rankings of Tab. 5.

Margin Reconstruction of the full-sequence teachers. The logit-regression experiments above test whether the continuous two-logit geometry of the full-sequence classifiers can be reconstructed from the abstract feature dictionary. This is a stringent requirement: a low-dimensional feature set may fail to reproduce calibrated logits while still capturing the teacher’s hard decision boundary. We therefore introduce a normalized Margin Reconstruction (MR) protocol. In this protocol the source teachers are not retrained. Instead, the teachers, train/validation splits, and teacher BAS values are inherited directly from the multi-seed full-sequence experiment. Student classifiers are then trained only on prescribed abstract feature subsets and are asked to reproduce the source teacher’s hard decisions on the same held-out split.

Table 7 reports the resulting margin profile. The column “Full” gives balanced agreement with the source teacher on the full held-out split. The columns $B _ { 5 } , B _ { 1 0 } ,$ and $B _ { 2 0 }$ restrict this agreement to the 5%, 10%, and 20% smallest-margin validation samples, where the source-teacher margin is $| \ell _ { 1 } - \ell _ { 0 } |$ . The column $E _ { 5 0 }$ reports agreement on the 50% largest-margin samples. Thus the table separates easy-region agreement from genuine boundary-local reconstruction. A surrogate that is accurate only on $E _ { 5 0 }$ captures coarse global structure; a surrogate that remains accurate on $B _ { 5 }$ and $B _ { 1 0 }$ captures the teacher’s local decision geometry.

<table><tr><td rowspan=1 colspan=1>Task| St</td><td rowspan=1 colspan=1>udent feats. |</td><td rowspan=1 colspan=1> $B _ { 1 0 } { \mathrm { ~ f e a t s . } }$ </td><td rowspan=1 colspan=1>—  Full   1</td><td rowspan=1 colspan=1> $B _ { 5 }$       一</td><td rowspan=1 colspan=1> $B _ { 1 0 }$     一</td><td rowspan=1 colspan=1> $B _ { 2 0 }$ </td><td rowspan=1 colspan=1> $E _ { 5 0 }$ </td></tr><tr><td rowspan=1 colspan=1>HOM-Star</td><td rowspan=1 colspan=1> $\varepsilon , \log .$ </td><td rowspan=1 colspan=1> $\varepsilon _ { \operatorname* { m i n } } , \sigma _ { \varepsilon }$ </td><td rowspan=1 colspan=1> $. 7 2 6 \pm . 2 5 8$ </td><td rowspan=1 colspan=1> $0 . 5 8 4 \pm 0 . 0 7 5$ </td><td rowspan=1 colspan=1> $. 6 0 4 \pm . 1 2 7$ </td><td rowspan=1 colspan=1> $. 6 2 3 \pm . 1 7 7$ </td><td rowspan=1 colspan=1> $. 7 6 2 \pm . 2 3 7$ </td></tr><tr><td rowspan=1 colspan=1>HOM-H</td><td rowspan=1 colspan=1> $\operatorname* { d e t } , \mathrm { R F }$ </td><td rowspan=1 colspan=1>det</td><td rowspan=1 colspan=1> $\overline { { . 9 2 0 \pm . 0 0 7 } }$ </td><td rowspan=1 colspan=1> $\overline { { 0 . 5 7 6 \pm 0 . 0 3 7 } }$ </td><td rowspan=1 colspan=1> $\overline { { . 6 3 3 \pm . 0 1 5 } }$ </td><td rowspan=1 colspan=1> $\overline { { . 7 1 6 \pm . 0 3 2 } }$ </td><td rowspan=1 colspan=1> $\overline { { . 9 7 3 \pm . 0 1 1 } }$ </td></tr><tr><td rowspan=1 colspan=1>HOM-All</td><td rowspan=1 colspan=1> $\operatorname* { d e t } , \mathrm { R F }$ </td><td rowspan=1 colspan=1> $\operatorname { d e t }$ </td><td rowspan=1 colspan=1> $9 5 3 \pm . 0 0 5$ </td><td rowspan=1 colspan=1> $\overline { { 0 . 7 3 7 \pm 0 . 0 2 1 } }$ </td><td rowspan=1 colspan=1> $\overline { { . 8 0 9 \pm . 0 1 4 } }$ </td><td rowspan=1 colspan=1> $\overline { { . 8 6 9 \pm . 0 1 7 } }$ </td><td rowspan=1 colspan=1> $\overline { { . 9 7 0 \pm . 0 0 5 } }$ </td></tr><tr><td rowspan=1 colspan=1>TOP-Sph</td><td rowspan=1 colspan=1>γ, log.</td><td rowspan=1 colspan=1> $\mu _ { \gamma } , \gamma _ { 0 }$ </td><td rowspan=1 colspan=1> $. 9 6 9 \pm . 0 0 5$ </td><td rowspan=1 colspan=1> $0 . 6 8 2 \pm 0 . 0 4 0$ </td><td rowspan=1 colspan=1> $. 7 6 2 \pm . 0 3 0$ </td><td rowspan=1 colspan=1> $. 8 5 0 \pm . 0 2 6$ </td><td rowspan=1 colspan=1> $. 9 9 0 \pm . 0 0 3$ </td></tr><tr><td rowspan=1 colspan=1>TOP-Non</td><td rowspan=1 colspan=1> $\operatorname* { d e t } { + \gamma } , \mathrm { M L P }$ </td><td rowspan=1 colspan=1> $\sigma _ { \gamma } , \mu _ { \gamma }$ </td><td rowspan=1 colspan=1> $. 9 8 6 \pm . 0 0 3$ </td><td rowspan=1 colspan=1> $0 . 8 6 1 \pm 0 . 0 3 5$ </td><td rowspan=1 colspan=1> $. 9 0 9 \pm . 0 2 9$ </td><td rowspan=1 colspan=1> $. 9 3 2 \pm . 0 2 8$ </td><td rowspan=1 colspan=1> $. 9 8 5 \pm . 0 0 5$ </td></tr><tr><td rowspan=1 colspan=1>TOP-All</td><td rowspan=1 colspan=1> $\gamma , \log .$ </td><td rowspan=1 colspan=1> $\sigma _ { \gamma } , \mu _ { \gamma }$ </td><td rowspan=1 colspan=1> $. 9 7 7 \pm . 0 0 2$ </td><td rowspan=1 colspan=1> $0 . 7 8 4 \pm 0 . 0 1 3$ </td><td rowspan=1 colspan=1> $. 8 5 6 \pm . 0 1 1$ </td><td rowspan=1 colspan=1> $. 9 0 5 \pm . 0 1 1$ </td><td rowspan=1 colspan=1> $. 9 8 5 \pm . 0 0 3$ </td></tr></table>

Table 7: Normalized margin reconstruction for the full-sequence teachers, with feature attribution on the $B _ { 1 0 }$ teacher-boundary slice. The source teachers and train/validation splits are inherited directly from the full-sequence experiments. Students are trained to reproduce the source teacher’s hard decisions using only the indicated abstract feature set. $\displaystyle { } ^ { \mathfrak { s c } } { } \mathrm { F u l l } { } ^ { \mathfrak { s } }$ reports balanced agreement with the source teacher on the held-out split. $B _ { 5 } , \ B _ { 1 0 }$ and $B _ { 2 0 }$ restrict agreement to the 5%, 10%, and 20% smallest-margin held-out samples, where the margin is $| \ell _ { 1 } - \ell _ { 0 } | . ~ E _ { 5 0 }$ reports agreement on the 50% largest-margin held-out samples. $^ { * } B _ { 1 0 }$ feats.” lists the leading features under the average of normalized saliency, SHAP, and permutation-drop attribution scores. Here ϵ-stats denotes $( \varepsilon _ { \operatorname* { m i n } } , \varepsilon _ { \operatorname* { m a x } } , \mu _ { \varepsilon } , \sigma _ { \varepsilon } )$ , and γ-stats denotes $( \gamma _ { 0 } , \mu _ { \gamma } , \sigma _ { \gamma } )$

The homology tasks show two different behaviors. For HOM-H and HOM-All, determinantbased students recover the full-sequence teachers with high global agreement. HOM-H has full teacher agreement $0 . 9 2 0 \pm 0 . 0 0 7$ and true-label BAS $0 . 9 9 7 \pm 0 . 0 0 1$ , but its agreement drops to $0 . 5 7 6 \pm 0 . 0 3 7$ on $B _ { 5 }$ . This indicates that det captures the dominant global rule but not all boundary-local refinements of the teacher. HOM-All is stronger: the determinant-only student retains substantial agreement throughout the boundary profile, with $0 . 7 3 7 { \pm } 0 . 0 2 1$ on $B _ { 5 }$ $0 . 8 0 9 \pm 0 . 0 1 4$ on $B _ { 1 0 } ,$ and $0 . 8 6 9 \pm 0 . 0 1 7$ on $B _ { 2 0 }$

The topology tasks are more consistently explained by gap statistics. TOP-Sph and TOP-All select the γ-statistics feature set $( \gamma _ { 0 } , \mu _ { \gamma } , \sigma _ { \gamma } )$ , while TOP-Non selects det +γ-statistics. Their margin profiles show monotone improvement from the hardest boundary slice to the easy region, with strong full-split agreement in all cases. TOP-Non is the strongest boundary reconstruction result: the selected det +γ-statistics MLP achieves $0 . 8 6 1 \pm 0 . 0 3 5$ on $B _ { 5 } , 0 . 9 0 9 \pm 0 . 0 2 9$ on $B _ { 1 0 } ,$ $0 . 9 3 2 \pm 0 . 0 2 8$ on $B _ { 2 0 } { } ;$ , and $0 . 9 8 6 \pm 0 . 0 0 3$ on the full held-out split. TOP-All is also robust, with $\sigma _ { \gamma }$ and $\mu _ { \gamma }$ dominating the $B _ { 1 0 }$ -slice attribution and agreement rising from $0 . 7 8 4 \pm 0 . 0 1 3$ on $B _ { 5 }$ to $0 . 9 0 5 \pm 0 . 0 1 1$ on $B _ { 2 0 }$

The attribution-augmented column resolves the selected feature groups into active components. For HOM-Star, the selected feature set is the eigenvalue-statistic tuple $( \varepsilon _ { \mathrm { m i n } } , \varepsilon _ { \mathrm { m a x } } , \mu _ { \varepsilon } , \sigma _ { \varepsilon } )$ , with $\varepsilon _ { \mathrm { m i n } }$ and $\sigma _ { \varepsilon }$ dominating the $B _ { 1 0 }$ -slice attribution. However, the corresponding surrogate remains weak and unstable: full teacher agreement is only $0 . 7 2 6 { \pm } 0 . 2 5 8$ , with $0 . 5 8 4 \pm 0 . 0 7 5$ on $B _ { 5 }$ and $0 . 6 0 4 \pm 0 . 1 2 7$ on $B _ { 1 0 } .$ . We therefore treat HOM-Star as the hard case rather than as a successful low-dimensional reconstruction. This agrees with the poor multi-feature logit-regression result and suggests that the HOM-Star full-sequence teacher uses structure not captured by the current determinant/eigenvalue $/ \mathrm { g a p } / c _ { \mathrm { m a x } }$ abstract dictionary.

Overall, MR gives a more refined interpretation than either classifier accuracy or logit regression alone. Determinant information explains much of the HOM-H and HOM-All decision structure, gap statistics explain most of the topology decision structure, and HOM-Star remains the principal unresolved feature-discovery case.

Spectral Reconstruction. The signals identified by the interpretability metrics suggest that the network learns to recover spectral information about the plumbing matrix M from the q-series, but that the relevant spectral summary depends on the classification objective and on the sector of the dataset.

In the HOM experiments, the classifier is asked to distinguish homology spheres, where $| \mathsf { d e t } M | = 1$ from non-spheres. The full-sequence interpretability results show that the network does not rely only on the determinant itself. Instead, it repeatedly uses low-end and dispersion information in the spectrum, especially $\varepsilon _ { \operatorname* { m i n } } , \varepsilon _ { \operatorname* { m a x } } ,$ and $\sigma _ { \varepsilon } ,$ together with determinant information in the pooled HOM-All task. These features act as spectral proxies for the determinant constraint: they help distinguish genuine unit-determinant examples from plumbing matrices whose $q \mathrm { - }$ series statistics partially mimic the homology-sphere sector. The normalized MR analysis is consistent with this interpretation. For HOM-H and HOM-All, determinant-based students reconstruct the global teacher decisions well, while HOM-Star remains only weakly reconstructed by the current eigenvalue-statistic dictionary, indicating that the star-sector homology decision contains additional structure not captured by these coarse abstract features alone.

In the TOP experiments, the mechanism is more sector-dependent. For homology spheres, where the determinant is fixed, the dominant signal shifts toward gap statistics of the q-series. In TOP-Sph, $\mu _ { \gamma }$ is ranked first by FS, PI, and SHAP, with $\sigma _ { \gamma }$ appearing as the next stable feature. This supports the geometric picture that differences between H-graphs and star graphs affect the anisotropy of the quadratic form $\ell ^ { T } M ^ { - 1 } \ell$ in (1). Greater anisotropy can produce slower growth along selected directions and hence a different sparsity/gap profile in the resulting q-series. The classifier detects this topology-dependent anisotropy through the gap statistics, rather than through determinant variation.

For non-spheres, however, the determinant is no longer fixed, and the topology decision is not a pure gap test. In TOP-Non, the full-sequence attribution table repeatedly identifies det and $\varepsilon _ { \mathrm { { m i n } } }$ as leading features, with $\sigma _ { \gamma }$ appearing as a secondary gap statistic. Thus the non-spherical topology classifier combines determinant information, low-end spectral information, and residual gap structure. The pooled TOP-All task reflects the same hybrid mechanism: det, $\varepsilon _ { \mathrm { { m i n } } } .$ and $\sigma _ { \gamma }$ all appear across the attribution methods. In short, gap statistics dominate the fixeddeterminant homology-sphere topology task, while determinant and eigenvalue summaries become essential once the non-spherical sector is included.

## 4.3 Value-Index Pair Representation

In order to test the effect of data representation, we repeat the six classification tasks using the value-index pair representation described in §3.2.2. In this representation, the network receives both coefficient values and their exponent indices, rather than a fixed full-sequence array in which the exponent is encoded by position. This provides a useful robustness check: if the same abstract features remain predictive across the full-sequence and value-index representations, then the learned signal is unlikely to be an artifact of the dense array indexing.

For these experiments we use a fully connected MLP classifier with (depth, width) = (2, 64), and the same feature-to-logit interpretability pipeline with a regression network of size (4, 128).<sup>3</sup> The results are summarized in Tab. 8. Across all six tasks, the value-index representation improves the classifier performance and substantially improves the feature-to-logit reconstruction quality. In particular, the validation $R ^ { 2 }$ values are uniformly high, ranging from 0.851 ± 0.030 in HOM-All to 0.987 ± 0.004 in TOP-Sph, and the validation PCC values are at least 0.927 in every experiment. Thus, in the value-index representation, the ten derived features recover not

only the hard decisions but also a large fraction of the learned logit geometry.
<table><tr><td>Experiment |</td><td>BAS</td><td> $\mathbb { R } ^ { 2 }$ </td><td>PCC</td><td>FS</td><td>PI</td><td>SHAP</td></tr><tr><td>HOM-Star</td><td> $0 . 9 9 9 \pm 0 . 0 0 1$ </td><td> $\operatorname { t r } : 0 . 9 4 0 \pm 0 . 0 1 3$   $\mathrm { v a l } : 0 . 9 2 0 \pm 0 . 0 1 8$ </td><td> $\operatorname { t r } : 0 . 9 6 9 \pm 0 . 0 0 7$   $\mathrm { v a l } : 0 . 9 5 8 \pm 0 . 0 1 1$ </td><td> $\begin{array} { l } { \Delta { \ \mu _ { \gamma } \ } { \sf d e t } } \\ { { 1 . 3 ^ { \mathrm { , ~ } } \mu _ { 1 . 7 } ^ { \mathrm { ~ , ~ } } 3 . 7 } } \end{array}$ </td><td> $\mu _ { \gamma } \ \varepsilon _ { \mathrm { m i n } } \ \mathrm { d e t }$   $_ { 1 . 2 } \cdot _ { \mathrm { ~ 2 . 0 ~ } } \cdot _ { \mathrm { ~ 3 . 5 ~ } }$ </td><td> $\varepsilon _ { \operatorname* { m i n } } \ \mu _ { \gamma }$  det  $_ { 1 . 4 } \ \cdot \ _ { 1 . 6 } \cdot _ { 3 . 0 }$ </td></tr><tr><td>HOM-H</td><td> $0 . 9 6 7 \pm 0 . 0 0 4$ </td><td> $\mathbf { \overline { { t r } } } : 0 . 9 7 1 \pm 0 . 0 2 3$   $\mathrm { v a l } : 0 . 9 5 1 \pm 0 . 0 3 1$ </td><td> $\operatorname { t r } : 0 . 9 8 6 \pm 0 . 0 1 1$   $\mathrm { v a l } : 0 . 9 7 6 \pm 0 . 0 1 5$ </td><td> $\mu _ { \gamma } ~ \sigma _ { \gamma } ~ \Delta$   $_ { 1 . 0 } \cdot _ { \ 2 . 0 } \cdot _ { \ 3 . 0 }$ </td><td> $\mu _ { \gamma } ~ \triangle ~ \sigma _ { \gamma }$   $_ { 1 . 7 } \cdot _ { 2 . 5 ^ { 2 } 2 . 8 }$ </td><td> $\mu _ { \gamma } \textit { \textbf { \em { \alpha } } } $  det  $_ { 1 . 0 } \cdot _ { 2 . 0 } \cdot _ { 3 . 3 }$ </td></tr><tr><td>HOM-All</td><td> $0 . 9 8 5 \pm 0 . 0 0 2$ </td><td> $\mathbf { \overline { { t r } } } : 0 . 8 9 2 \pm 0 . 0 2 9$   $\mathrm { v a l } : 0 . 8 5 1 \pm 0 . 0 3 0$   $\mathbf { \bar { t r } } : 0 . 9 9 0 \pm 0 . 0 0 2$ </td><td> $\mathrm { ~ t r : 0 . 9 4 8 \pm 0 . 0 1 4 ~ }$   $\mathsf { v a l } : 0 . 9 2 7 \pm 0 . 0 1 3$ </td><td> $\operatorname* { d e t } \ \varepsilon _ { \mathrm { m i n } } \ \sigma _ { \gamma }$   $_ { 1 . 0 } \cdot _ { \mathrm { ~ 2 . 0 ~ } } \cdot _ { \mathrm { ~ 3 . 5 ~ } }$ </td><td> $\sigma _ { \varepsilon } \varepsilon _ { \mathrm { m i n } } \mu _ { \gamma }$   $_ { 2 . 2 } > _ { 2 . 4 } > _ { 3 . 3 }$ </td><td> $\varepsilon _ { \mathrm { m i n } } ~ \mathrm { d e t } ~ \mu _ { \gamma }$   $_ { 1 . 5 } ~ , _ { _ { 2 . 3 } } \cdot _ { _ { 2 . 6 } }$ </td></tr><tr><td> $\mathrm { T O P - } \mathrm { S p h }$ </td><td> $1 . 0 0 0 \pm 0 . 0 0 0$ </td><td> $\mathrm { v a l } : 0 . 9 8 7 \pm 0 . 0 0 4$   $\mathbf { \overline { { t r } } } : 0 . 9 7 9 \pm 0 . 0 0 3$ </td><td> $\mathbf { \overline { { t r } } } : 0 . 9 9 5 \pm 0 . 0 0 1$   $\mathsf { v a l } : 0 . 9 9 4 \pm 0 . 0 0 2$   $\operatorname { t r } : 0 . 9 9 0 \pm 0 . 0 0 2$ </td><td> $\mu _ { \gamma } \ : \sigma _ { \gamma } \ : \ : \gamma _ { 0 }$ </td><td> $\mu _ { \gamma } \textit { \textbf { \em { \alpha } } } ^ { \pi } \gamma _ { 0 }$   $_ { 1 . 1 } , \ _ { 2 . 0 } , \ _ { 3 . 1 }$ </td><td> $\mu _ { \gamma } ~ \sigma _ { \gamma } ~ \Delta$   $_ { 1 . 0 } , \ _ { 2 . 1 } , \ _ { 3 . 0 }$ </td></tr><tr><td>TOP-Non</td><td> $1 . 0 0 0 \pm 0 . 0 0 0$ </td><td> $\mathrm { v a l } : 0 . 9 7 5 \pm 0 . 0 0 4$   $\operatorname { t r } : 0 . 9 6 0 \pm 0 . 0 0 3$ </td><td> $\mathrm { v a l } : 0 . 9 8 8 \pm 0 . 0 0 2$ </td><td> $\mu _ { \gamma } ~ \Delta { \sf { d e t } }$   $_ { 1 . 4 } \cdot _ { 3 . 2 } \cdot _ { 3 . 6 }$ </td><td> $\mu _ { \gamma } \sigma _ { \gamma } \varepsilon _ { \mathrm { { m i n } } }$   $_ { 1 . 4 } \dot { } ^ { \ 3 } 2 . \dot { 3 } ^ { \ 3 } 3 . 7$ </td><td> $\mu _ { \gamma } \varepsilon _ { \mathrm { m i n } } \sigma _ { \gamma }$   $_ { 1 . 0 } \cdot _ { \ 2 . 2 } \cdot _ { \ 3 . 3 }$ </td></tr><tr><td>TOP-All</td><td> $1 . 0 0 0 \pm 0 . 0 0 0$ </td><td> $\mathrm { v a l } : 0 . 9 5 2 \pm 0 . 0 0 5$ </td><td> $\mathbf { \bar { t r } } : 0 . 9 8 0 \pm 0 . 0 0 2$   $\mathrm { v a l } : 0 . 9 7 7 \pm 0 . 0 0 2$ </td><td> $\Delta \ \sigma _ { \gamma } \ \varepsilon _ { \mathrm { { m i n } } }$   $_ { 1 . 8 } \mathbf { ^ { 3 } } \mathbf { ^ { 2 . 5 } } \mathbf { ^ { 3 } }$  3.8</td><td> $\varepsilon _ { \mathrm { m i n } } ~ \mu _ { \gamma } ~ \sigma _ { \gamma }$   $_ { 2 . 3 } > _ { 2 . 6 } > _ { 3 . 1 }$ </td><td> ${ \sigma _ { \gamma } } \operatorname { \mathrm { ~ } } _ { { { \mathrm { ~ } \alpha } ^ { } } , { { \mathrm { ~ } \alpha } ^ { } } , { { } ^ { } \varepsilon _ { \operatorname { m i n } } } } ^ { \mathrm { ~ } }$ </td></tr></table>

Table 8: Multi-seed value-index results for the topology and homology experiments. BAS is reported as mean ± standard deviation over classifier seeds. The $R ^ { 2 }$ and Pearson correlation (PCC) columns report train and validation reconstruction quality of the feature-to-logit regressor, stacked within each cell and averaged over the classifier–regressor seed ensemble. The final columns report the three lowest mean feature ranks for feature saliency (FS), permutation importance (PI), and SHAP, where the subscript gives the average rank over the normalized interpretability ensemble.

## 4.3.1 Homology Classification

HOM-Star. We begin by training the value-index pair classifier to distinguish homology spheres from non-spheres within the star-shaped graph dataset. The classifier is trained on 10,474- dimensional value-index inputs and achieves nearly perfect classification, with mean BAS 0.999±0.001. This is a substantial improvement over the corresponding full-sequence HOM-Star experiment, where the classifier remained accurate but the logit geometry was poorly reconstructed by the ten-feature regressor.

In the value-index representation, the feature-to-logit regressor achieves

$$
( R _ { \mathrm { t r } } ^ { 2 } , R _ { \mathrm { v a l } } ^ { 2 } , { \mathrm { P C C } } _ { \mathrm { t r } } , { \mathrm { P C C } } _ { \mathrm { v a l } } ) = ( 0 . 9 4 0 \pm 0 . 0 1 3 , 0 . 9 2 0 \pm 0 . 0 1 8 , 0 . 9 6 9 \pm 0 . 0 0 7 , 0 . 9 5 8 \pm 0 . 0 1 1 ) .
$$

Thus the ten derived features now reconstruct the value-index classifier logits extremely well. This does not mean that the raw network uses no information beyond the ten features; rather, it means that the part of the learned decision function relevant to the classifier logits is accurately summarized by these spectral and support statistics.

The attribution metrics indicate that the value-index HOM-Star classifier uses a mixed determinant– gap–spectral signature. Feature saliency ranks $\Delta$ and $\mu _ { \gamma }$ highest, followed by det. Permutation importance ranks $\mu _ { \gamma }$ first, then $\varepsilon _ { \mathrm { { m i n } } } ,$ then det. SHAP ranks $\varepsilon _ { \mathrm { { m i n } } }$ first, with $\mu _ { \gamma }$ and det close behind. Thus, unlike the full-sequence HOM-Star case, the value-index representation does not isolate a single low-end eigenvalue proxy. Instead, it makes the HOM-Star decision highly linearly reconstructible from a combination of first-gap information, mean gap information, determinant information, and $\varepsilon _ { \mathrm { { m i n } } } .$ The robust appearance of $\mu _ { \gamma }$ across all three methods suggests that exposing the exponent indices directly allows the classifier to use the support spacing of the q-series more efficiently than in the full-sequence representation.

HOM-H. Next, we evaluate the value-index pair classifier on the H-graph homology task. The model achieves $\mathrm { B A S } \ 0 . 9 6 7 \pm 0 . 0 0 4$ , improving over the full-sequence HOM-H classifier. The

regressor also becomes highly accurate:

$$
( R _ { \mathrm { t r } } ^ { 2 } , R _ { \mathrm { v a l } } ^ { 2 } , \mathrm { P C C _ { \mathrm { t r } } , \mathrm { P C C _ { \mathrm { v a l } } } } ) = ( 0 . 9 7 1 \pm 0 . 0 2 3 , 0 . 9 5 1 \pm 0 . 0 3 1 , 0 . 9 8 6 \pm 0 . 0 1 1 , 0 . 9 7 6 \pm 0 . 0 1 5 ) .
$$

Thus, for H-graphs, the value-index classifier’s logits are almost completely reconstructed by the ten abstract features.

The feature rankings are also more coherent than in the full-sequence case. FS, PI, and SHAP all identify $\mu _ { \gamma }$ as a leading feature, with average ranks 1.0, 1.7, and 1.0, respectively. The gap dispersion $\sigma _ { \gamma }$ is also consistently important, appearing second under FS and SHAP and third under PI. The unseen leading power $\Delta$ is emphasized by FS and PI, while SHAP includes det as the third-ranked feature. This pattern strengthens the interpretation developed in the fullsequence HOM-H analysis: in the H-graph sector, homology-sphere detection is not governed solely by a low-end eigenvalue proxy. Instead, the classifier uses the support-spacing structure of the q-series, especially $\mu _ { \gamma }$ and $\sigma _ { \gamma } ,$ , together with residual determinant information.

HOM-All. We next train on the combined homology dataset, asking the value-index classifier to distinguish homology spheres from non-spheres without conditioning on the graph topology. The classifier achieves BAS $0 . 9 8 5 \pm 0 . 0 0 2$ , again improving over the full-sequence representation. The feature-to-logit regressor gives

$$
\begin{array} { r } { ( R _ { \mathrm { t r } } ^ { 2 } , R _ { \mathrm { v a l } } ^ { 2 } , \mathrm { P C C } _ { \mathrm { t r } } , \mathrm { P C C } _ { \mathrm { v a l } } ) = ( 0 . 8 9 2 \pm 0 . 0 2 9 , 0 . 8 5 1 \pm 0 . 0 3 0 , 0 . 9 4 8 \pm 0 . 0 1 4 , 0 . 9 2 7 \pm 0 . 0 1 3 ) . } \end{array}
$$

The ten-feature dictionary therefore remains highly predictive even in the pooled, heterogeneous homology task.

The attribution metrics show a hybrid rule. Feature saliency ranks det first and $\varepsilon _ { \mathrm { { m i n } } }$ second, with $\sigma _ { \gamma }$ third. SHAP similarly ranks $\varepsilon _ { \mathrm { { m i n } } } .$ , det, and $\mu _ { \gamma }$ as the leading features. Permutation importance emphasizes eigenvalue dispersion, ranking $\sigma _ { \varepsilon } , \varepsilon _ { \mathrm { m i n } }$ , and $\mu _ { \gamma }$ highest. Hence the value-index HOM-All classifier combines the explicit determinant signal with low-end spectral information and gap information. This is consistent with the pooled full-sequence HOM-All experiment, but the value-index representation makes the relationship between the classifier logits and the abstract features much sharper.

## 4.3.2 Graph Topology Classification

We next ask the value-index classifier to distinguish star-shaped from H-shaped plumbing graphs. This is a particularly useful representation check. In the full-sequence representation, exponent information is encoded implicitly by the array position. In the value-index representation, the coefficients and exponent indices are supplied as paired data. If graph topology were detected only through artifacts of the dense array coordinate system, one would expect a collapse in performance or a substantial change in the reconstructed features. Instead, the topology tasks become almost perfectly classified, and the feature-to-logit regressors achieve very high validation $R ^ { 2 }$ and PCC.

TOP-Sph. For the homology-sphere subset, the value-index topology classifier achieves perfect separation, with BAS $1 . 0 0 0 \pm 0 . 0 0 0$ . The logit regressor also becomes nearly exact:

$$
( R _ { \mathrm { t r } } ^ { 2 } , R _ { \mathrm { v a l } } ^ { 2 } , \mathrm { P C C _ { \mathrm { t r } } , \mathrm { P C C _ { \mathrm { v a l } } } } ) = ( 0 . 9 9 0 \pm 0 . 0 0 2 , 0 . 9 8 7 \pm 0 . 0 0 4 , 0 . 9 9 5 \pm 0 . 0 0 1 , 0 . 9 9 4 \pm 0 . 0 0 2 ) .
$$

The attribution metrics are essentially unanimous. FS, PI, and SHAP all rank $\mu _ { \gamma }$ first, and all identify $\sigma _ { \gamma }$ as the next stable feature. The third feature varies between $\gamma _ { 0 }$ and $\Delta _ { \perp }$ , but the dominant signal is clearly the gap profile.

This reproduces and strengthens the full-sequence TOP-Sph interpretation. Once the determinant is fixed by restricting to homology spheres, topology is encoded primarily in the spacing

structure of the q-series. The value-index representation makes this signal even cleaner: the classifier can access exponent locations explicitly, and its logit geometry is almost completely reconstructed by the gap statistics $\mu _ { \gamma } , \sigma _ { \gamma } .$ , and related first-gap information.

TOP-Non. For non-homology spheres, the value-index topology classifier again achieves perfect separation, with BAS $1 . 0 0 0 \pm 0 . 0 0 0$ . The regressor remains highly accurate:

$$
( R _ { \mathrm { t r } } ^ { 2 } , R _ { \mathrm { v a l } } ^ { 2 } , \mathrm { P C C _ { \mathrm { t r } } , \mathrm { P C C _ { \mathrm { v a l } } } } ) = ( 0 . 9 7 9 \pm 0 . 0 0 3 , 0 . 9 7 5 \pm 0 . 0 0 4 , 0 . 9 9 0 \pm 0 . 0 0 2 , 0 . 9 8 8 \pm 0 . 0 0 2 ) .
$$

Unlike the full-sequence TOP-Non experiment, where determinant and $\varepsilon _ { \mathrm { { m i n } } }$ were the most stable recurring features, the value-index representation shifts the dominant reconstructed signal toward gap statistics. All three attribution methods rank $\mu _ { \gamma }$ first or near first: FS and PI give it average rank 1.4, while SHAP ranks it first. PI and SHAP also retain spectral information through $\varepsilon _ { \mathrm { { m i n } } }$ , and FS includes det, but these appear as secondary rather than primary signals.

Thus TOP-Non and TOP-Sph are no longer aligned with the full-sequence distinction between a gap-driven sphere sector and a determinant/eigenvalue-driven non-sphere sector. In the valueindex representation, the non-spherical topology classifier can still use determinant and lowend spectral information, but its logit geometry is reconstructed most cleanly by the q-series gap profile. This suggests that explicitly providing exponent indices makes topology-dependent support spacing accessible even when determinant variation is present.

TOP-All. Finally, in the pooled topology task, the value-index classifier again achieves BAS 1.000 ± 0.000. The regressor gives

$$
( R _ { \mathrm { t r } } ^ { 2 } , R _ { \mathrm { v a l } } ^ { 2 } , \mathrm { P C C _ { \mathrm { t r } } , \mathrm { P C C _ { \mathrm { v a l } } } } ) = ( 0 . 9 6 0 \pm 0 . 0 0 3 , 0 . 9 5 2 \pm 0 . 0 0 5 , 0 . 9 8 0 \pm 0 . 0 0 2 , 0 . 9 7 7 \pm 0 . 0 0 2 ) .
$$

The pooled value-index topology task therefore remains almost perfectly reconstructible from the abstract features.

The feature rankings show a mixed but gap-centered rule. FS ranks $\Delta , \sigma _ { \gamma }$ , and $\varepsilon _ { \mathrm { { m i n } } }$ highest. PI ranks $\varepsilon _ { \operatorname* { m i n } } , \mu _ { \gamma } ,$ , and $\sigma _ { \gamma }$ highest. SHAP ranks $\sigma _ { \gamma }$ first, followed by $\mu _ { \gamma }$ and $\varepsilon _ { \mathrm { { m i n } } } .$ . Thus the stable features across methods are $\sigma _ { \gamma } , \mu _ { \gamma } ,$ , and $\varepsilon _ { \mathrm { { m i n } } } ,$ with the unseen leading power $\Delta$ prominent in FS.

We interpret this as a representation-dependent strengthening of the gap signal. In the fullsequence TOP-All experiment, the reconstructed topology signal was hybrid, involving det, $\varepsilon _ { \mathrm { { m i n } } } ,$ and $\sigma _ { \gamma }$ . In the value-index representation, determinant information becomes less central in the pooled topology attribution, while gap statistics become the dominant reconstructed features. This is consistent with the purpose of the value-index input: by supplying exponent positions explicitly, it makes the support-spacing structure of the q-series easier for the classifier to exploit. The topology signal is therefore not lost when the dense array coordinate system is removed; instead, it becomes more cleanly associated with the gap profile of the q-series.

## 4.4 Latent Space Representation

## 4.4.1 Binary Classification

We first evaluate classifiers trained on contrastive latent embeddings. In these experiments, the encoder is trained from the $^ { 1 0 , 0 0 0 }$ -dimensional full-sequence representation using semihard triplet loss and outputs a 5-dimensional embedding. The classifier is then trained on the standardized 5-dimensional latent coordinates. This creates a severe information bottleneck, compressing the original full-sequence input by a factor of 2000. The purpose of this experiment is to test whether the contrastive objective distills the q-series into a lower-dimensional representation whose classifier logits remain reconstructible from the ten derived spectral and support features.

Homology Classification (Star, H, and Combined). For the homology classification tasks, the latent-space classifiers achieve strong performance, although they generally do not match the nearly perfect value-index classifiers. The BAS values are $0 . 9 4 0 \pm 0 . 0 0 5$ for HOM-Star, $0 . 9 6 0 \pm 0 . 0 0 8$ for HOM-H, and $0 . 9 5 7 \pm 0 . 0 0 4$ for HOM-All. Thus the contrastive bottleneck preserves most of the discriminative information needed to identify homology spheres, despite reducing the input dimension from 10,000 to 5.

The main advantage of the latent representation is not raw classifier accuracy, but the stability and interpretability of the learned logit geometry. In HOM-H, for example, the latent classifier slightly underperforms the value-index classifier, but its logits are highly reconstructible from the ten-feature dictionary:

$$
\begin{array} { r } { ( R _ { \mathrm { t r } } ^ { 2 } , R _ { \mathrm { v a l } } ^ { 2 } , \mathrm { P C C _ { \mathrm { t r } } , \mathrm { P C C _ { \mathrm { v a l } } } } ) = ( 0 . 9 9 1 \pm 0 . 0 0 2 , 0 . 8 9 9 \pm 0 . 0 2 1 , 0 . 9 9 6 \pm 0 . 0 0 1 , 0 . 9 5 2 \pm 0 . 0 0 9 ) . } \end{array}
$$

The same pattern holds across the homology tasks. HOM-Star improves dramatically relative to the full-sequence regressor, reaching $R _ { \mathrm { v a l } } ^ { 2 } = 0 . 7 8 4 \pm 0 . 0 3 2$ , while HOM-All reaches $R _ { \mathrm { v a l } } ^ { 2 } = 0 . 8 1 8 \pm 0 . 0 1 6$ . The contrastive encoder therefore appears to smooth or regularize the decision geometry: the resulting low-dimensional classifier remains accurate, and its logits are much more faithfully reconstructed by the abstract feature dictionary than in the full-sequence representation.

We interpret this as a form of semantic filtering. In the full-sequence and value-index representations, the classifier can route information through many high-dimensional directions, making gradient-based and perturbation-based explanations sensitive to representation-specific artifacts. By contrast, the semi-hard triplet objective forces samples with the same label to cluster and samples with different labels to separate in a low-dimensional metric space. Nondiscriminative high-variance directions provide little margin benefit but can inject noise into near-boundary triplets. The encoder is therefore pressured to suppress such directions and retain coordinates that are stable for the class geometry. This does not prove that the latent classifier uses a single universal feature across all homology tasks; rather, it shows that after contrastive compression, the learned logits are consistently well explained by the ten derived determinant, spectral, and support statistics.

Graph Topology Classification (Spheres, Non-Spheres, and Combined). The topology tasks show that the latent representation preserves the graph-topology signal even more strongly. The classifiers achieve BAS 0.979±0.004 for TOP-Sph, 0.998±0.001 for TOP-Non, and 0.992±0.001 for TOP-All. Thus, even after compression to five latent coordinates, the encoder retains enough information to distinguish star-shaped and H-shaped plumbing graphs with high accuracy.

The feature-to-logit regressors are also strong. For TOP-Sph, the regressor achieves

$$
( R _ { \mathrm { t r } } ^ { 2 } , R _ { \mathrm { v a l } } ^ { 2 } , \mathrm { P C C _ { \mathrm { t r } } , \mathrm { P C C _ { \mathrm { v a l } } } } ) = ( 0 . 9 9 1 \pm 0 . 0 0 3 , 0 . 9 4 4 \pm 0 . 0 0 4 , 0 . 9 9 5 \pm 0 . 0 0 2 , 0 . 9 7 3 \pm 0 . 0 0 2 ) .
$$

For TOP-Non, the reconstruction is even stronger:

$$
( R _ { \mathrm { t r } } ^ { 2 } , R _ { \mathrm { v a l } } ^ { 2 } , \mathrm { P C C _ { \mathrm { t r } } , \mathrm { P C C _ { \mathrm { v a l } } } } ) = ( 0 . 9 9 7 \pm 0 . 0 0 2 , 0 . 9 8 0 \pm 0 . 0 1 8 , 0 . 9 9 8 \pm 0 . 0 0 1 , 0 . 9 9 0 \pm 0 . 0 0 9 ) .
$$

The pooled TOP-All task remains highly accurate at the classifier level, although the regressor is less stable across seeds, with $R _ { \mathrm { v a l } } ^ { 2 } = 0 . 8 3 8 \pm 0 . 3 3 2$ and $\mathrm { P C C } _ { \mathrm { v a l } } = 0 . 9 4 7 \pm 0 . 0 9 9$ . This larger variance indicates that some latent TOP-All classifiers learn logit geometries that are less uniformly captured by the ten-feature regressor, even though their hard-label performance remains strong.

These results refine the picture from the full-sequence and value-index experiments. The contrastive bottleneck does not destroy either the gap-based topology signal visible in TOP-Sph or the determinant/eigenvalue information available in the non-spherical sector. Instead, it compresses these signals into a five-dimensional metric representation whose logits are usually highly reconstructible from the abstract feature dictionary. The strongest statement supported by the latent-space results is therefore not that the encoder eliminates all spectral shortcuts, but that it organizes the topology decision into a low-dimensional geometry aligned with the same determinant, spectral, and gap features identified in the higher-dimensional representations.
<table><tr><td>Feature</td><td></td><td>TOP coefficient HOM coefficient</td></tr><tr><td>det</td><td> $0 . 0 3 \pm 0 . 0 1$ </td><td> $2 . 1 7 \pm 0 . 0 4$ </td></tr><tr><td> $\varepsilon _ { \mathrm { m i n } }$ </td><td> $0 . 3 0 \pm 0 . 0 1$ </td><td> $4 . 4 3 \pm 0 . 1 6$ </td></tr><tr><td> $\varepsilon _ { \mathrm { m a x } }$ </td><td> $- 0 . 2 7 \pm 0 . 0 4$ </td><td> $- 0 . 6 2 \pm 0 . 0 3$ </td></tr><tr><td> $\mu _ { \varepsilon }$ </td><td> $0 . 1 6 \pm 0 . 0 2$ </td><td> $- 0 . 2 6 \pm 0 . 0 2$ </td></tr><tr><td> $\sigma _ { \varepsilon }$ </td><td> $0 . 1 9 \pm 0 . 0 6$ </td><td> $0 . 6 3 \pm 0 . 0 4$ </td></tr><tr><td> $\Delta$ </td><td> $1 . 1 3 \pm 0 . 0 3$ </td><td> $- 0 . 3 5 \pm 0 . 0 4$ </td></tr><tr><td> $\gamma _ { 0 }$ </td><td> $\overline { { 0 . 0 6 \pm 0 . 0 6 } }$ </td><td> $0 . 0 5 \pm 0 . 0 2$ </td></tr><tr><td> $\mu _ { \gamma }$ </td><td> $2 . 2 3 \pm 0 . 0 7$ </td><td> $0 . 4 1 \pm 0 . 1 6$ </td></tr><tr><td> $\sigma _ { \gamma }$ </td><td> $5 . 4 1 \pm 0 . 0 4$ </td><td> $- 1 . 0 3 \pm 0 . 0 5$ </td></tr><tr><td> $c _ { \mathrm { m a x } }$ </td><td> $0 . 2 1 \pm 0 . 0 1$ </td><td> $0 . 0 4 \pm 0 . 0 1$ </td></tr></table>

Table 9: Multi-seed normalized coefficients $\kappa _ { i }$ for the linear abstract-feature boundary approximation $\begin{array} { r } { \sum _ { i } \kappa _ { i } f _ { i } = 0 } \end{array}$ . The TOP column is obtained from a class-balanced linear classifier separating graph topology classes, while the HOM column is obtained from a class-balanced linear classifier separating homology spheres from non-spheres. For each seed, the ten abstract features are standardized on the training split, the linear classifier is fit, and the coefficient vector is rescaled so that $\begin{array} { r } { \frac { 1 } { 1 0 } \sum _ { i } | \kappa _ { i } | = 1 } \end{array}$ . Entries report mean ± standard deviation over the five-seed ensemble. Coefficient signs follow the binary label encoding, with positive coefficients pointing toward the star-shaped class (TOP row) and the non-sphere class (HOM row); coefficient magnitudes should be interpreted as robust linear feature strengths.

## 4.4.2 Multi-class Classification

Finally, we study the geometry of the latent-space decision boundaries in a four-class classification problem. We use the complete dataset described in Tab. 1, organizing it into four classes according to both plumbing graph topology and homology type: STAR-Non, STAR-Sph, H-Sph, H-Non. Preliminary experiments showed that the value-index pair representation gave the strongest classifier performance in the binary tasks. We therefore use the value-index pair format as the input representation for generating the contrastive latent embeddings in this analysis.

For each seed, the encoder is trained on the 10,474-dimensional value-index input using semihard triplet loss. The encoder is an MLP with three hidden layers of width 512, followed by a 5-dimensional latent output. Training uses margin $m = 1$ , learning rate $1 0 ^ { - 3 }$ , batch size 256, and early stopping with patience 10. All downstream classifiers and reconstruction models are trained using standard scikit-learn implementations unless otherwise specified. The analysis is repeated over the seed ensemble {42, 123, 456, 789, 1337}.

Although the learned embedding is 5-dimensional, PCA shows that the first two principal components capture approximately 99.5% of the latent variance across seeds. The resulting twodimensional projection is therefore not merely a visualization device: it captures the dominant geometry of the contrastive representation. This collapse is consistent with the label structure, since the four classes are the intersections of two binary attributes, namely homology class and graph topology.

Working in this two-dimensional PCA subspace, we first train binary logistic regression classifiers for the two induced binary labels. The topology classifier, separating star-shaped graphs from H-graphs, achieves near-perfect accuracy across seeds, $0 . 9 9 8 \pm 0 . 0 0 2$ . The homology classifier, separating spheres from non-spheres, is also strong but more seed-dependent, with accuracy $0 . 9 4 3 \pm 0 . 0 5 0$ . Thus the latent plane is organized by two coarse axes corresponding approximately to graph topology and homology type, with the topology direction more stable across seeds.

As a first linear diagnostic, we fit class-balanced logistic regressions from the ten standardized abstract features directly to the two induced binary labels: graph topology and homology type. This gives the linear boundary approximation

$$
\sum _ { i } \kappa _ { i } f _ { i } = 0 ,
$$

with normalized coefficients reported in Tab. 9. The linear topology classifier achieves held-out $\mathrm { B A S } \ 0 . 9 8 8 \pm 0 . 0 0 1$ , while the linear homology classifier achieves held-out BAS 0.975 ± 0.002. The coefficient magnitudes show a clean separation of mechanisms: the TOP boundary is dominated by the gap statistics, especially $\sigma _ { \gamma }$ and $\mu _ { \gamma } { . }$ , with $\Delta$ also contributing; the HOM boundary is dominated by $\varepsilon _ { \mathrm { { m i n } } }$ and det, with a smaller contribution from $\sigma _ { \varepsilon }$ . Thus, before analyzing the nonlinear one-vs-rest SVM surfaces, the abstract-feature linear model already recovers the two coarse axes suggested by the contrastive latent geometry.

To capture the nonlinear geometry of the four-class problem, we then train one-vs-rest RBF-SVM classifiers in the two-dimensional latent plane. The resulting decision functions are shown in Fig. 6. The top row displays the seed-42 latent projection and the corresponding one-vs-rest SVM decision values. The dashed curves mark the zero-level decision boundaries. These curves show that the four-class separation is not globally linear: the binary topology and homology axes provide a useful coarse coordinate system, but the individual class boundaries bend around the local cluster geometry.

To relate these nonlinear SVM boundaries to interpretable quantities, we reconstruct each onevs-rest decision function from the ten abstract features. For each class and each seed, we scan a small family of reconstruction models, including random forests, gradient boosting, linear regression, ridge regression, and a simple MLP. Model selection is performed on the validation split. Reconstruction quality is summarized by the composite score $Q _ { \mathrm { c o m p o s i t e } } ,$ which combines the coefficient of determination $R ^ { 2 }$ , Spearman rank correlation $\rho ,$ sign accuracy $\vartheta ,$ and the normalized MAE-complement boundary score $S _ { \mathrm { M A E } } ,$ as defined in (32). The resulting multiseed reconstruction summary is reported in Tab. 10.

The reconstruction results show that the one-vs-rest boundaries are strongly, but not uniformly, captured by the ten-feature dictionary. STAR-Sph, H-Sph, and H-Non have high composite scores,

$$
Q _ { \mathrm { c o m p o s i t e } } = 0 . 9 7 5 \pm 0 . 0 1 3 , \qquad 0 . 9 8 1 \pm 0 . 0 0 9 , \qquad 0 . 9 6 3 \pm 0 . 0 2 3 ,
$$

respectively. STAR-Non is weaker and more seed-dependent, with

$$
Q _ { \mathrm { c o m p o s i t e } } = 0 . 8 9 8 \pm 0 . 2 0 1 .
$$

This is consistent with the bottom row of Fig. 6, where the STAR-Non reconstruction diagnostics have the largest error bars.

![](images/751f14348a8818b67763b16a132802d508cd024a656912511116a3d37dc1a575.jpg)

![](images/c46587766f10c815618be14e8627a505819df582cfbee8f8725fab44ba7b4da2.jpg)

![](images/d3bea31ea4fe471981b3d30098b90782d33b61ab3b63e30b3b6de92cfe786fa5.jpg)

![](images/e00c75d344b42e1c3f3ac8c68bd31dfbdd08817784b2cdae0de0852bc56256ab.jpg)

![](images/1001e29f32f9abcabe2083d78092d03163cfec2ffc22fd98ab5ab8858c861415.jpg)

![](images/72e95c7df5f06b4d81cea04792d4e993fd704e925e72a7ad2341d6bbf63fffd1.jpg)

![](images/5f889c98142f748afd9b3c339592e72c4c7597045047aa403d103cd3d38565a1.jpg)

![](images/bdef53da89b3d6fdb4a47c1d70f6669ffafbb3239782b2f36e2042b5b4270f2a.jpg)

![](images/a169cbd663e406d742e897db958a2aa34425e3b13bdc5bb6e238f6e0b72b3cc3.jpg)

![](images/9d7a180aafa7c8cd8997bbcf4c3d021f6ea4897e8526d81b517312e23ce9427b.jpg)

![](images/26b8bfb7a85563d149d33013b907a5529ec23f35aaebb58da4d97f89087cb3ff.jpg)

![](images/3be48ef16dfc1021cf789b5056a2b3858cbf586d6c1ce0a8abee669797e1ec46.jpg)

Figure 6: One-vs-rest decision-boundary reconstruction in the four-class contrastive latent representation. Columns correspond to the four classes STAR-Non, STAR-Sph, H-Sph, and H-Non. The top row shows the seed-42 latent-space projection with an RBF-SVM one-vs-rest decision function: colored points are the positive class, gray points are the complementary classes, filled contours show the SVM decision value, and the dashed black curve is the zero-level decision boundary. The middle row reports the multi-seed mean feature importances of the cross-validation-selected reconstruction model for the corresponding boundary, with error bars showing seed-to-seed standard deviation. The bottom row reports multi-seed reconstruction quality using $R ^ { 2 } { } _ { ; }$ , Spearman correlation $\rho ,$ sign accuracy $\vartheta ,$ and boundary agreement $ { \mathcal { S } } _ { \mathrm { M A E } }$ . Together, the panels show that the latent one-vs-rest boundaries are largely reconstructible from the abstract features, with STAR-side boundaries emphasizing $\varepsilon _ { \mathrm { m i n } }$ and det, and H-side boundaries emphasizing gap-statistic features such as $\mu _ { \gamma }$ and $\sigma _ { \gamma } .$ The STAR-Non boundary is the least stable across seeds and accounts for most of the variance in the reconstruction diagnostics.
<table><tr><td rowspan=1 colspan=1>OvR Boundary</td><td rowspan=1 colspan=1>Best Model</td><td rowspan=1 colspan=1>Best FI</td><td rowspan=1 colspan=1> $\mathbf { Q } _ { \mathrm { c o m p o s i t e } }$ </td></tr><tr><td rowspan=1 colspan=1>STAR-Non</td><td rowspan=1 colspan=1>RandomForest</td><td rowspan=1 colspan=1> $\varepsilon _ { \mathrm { m i n } }$      det       $\mu _ { \gamma }$ 0.342± 0.0490.260± 0.0270.147± 0.037</td><td rowspan=1 colspan=1> $0 . 8 9 8 \pm 0 . 2 0 1$ </td></tr><tr><td rowspan=1 colspan=1>STAR-Sph</td><td rowspan=1 colspan=1>GradBoost</td><td rowspan=1 colspan=1> $\varepsilon _ { \mathrm { m i n } }$        $\mu _ { \gamma }$        det0.304± 0.241&#x27;0.220± 0.168&#x27;0.211 ± 0.142</td><td rowspan=1 colspan=1> $0 . 9 7 5 \pm 0 . 0 1 3$ </td></tr><tr><td rowspan=1 colspan=1>H-Sph</td><td rowspan=1 colspan=1>RandomForest</td><td rowspan=1 colspan=1> $\mu _ { \gamma }$         $\sigma _ { \gamma }$         $\gamma _ { 0 }$ 0.462± 0.276&#x27;0.218± 0.1250.113± 0.065</td><td rowspan=1 colspan=1> $0 . 9 8 1 \pm 0 . 0 0 9$ </td></tr><tr><td rowspan=1 colspan=1>H-Non</td><td rowspan=1 colspan=1>GradBoost</td><td rowspan=1 colspan=1> $\mu _ { \gamma }$        $\varepsilon _ { \mathrm { m i n } }$        $\sigma _ { \gamma }$ 0.569± 0.337&#x27;0.182± 0.306&#x27;0.108± 0.131</td><td rowspan=1 colspan=1> $0 . 9 6 3 \pm 0 . 0 2 3$ </td></tr></table>

Table 10: Multi-seed reconstruction summary for the one-vs-rest decision boundaries in the four-class contrastive latent space. The top-feature column reports the three largest feature importances of the cross-validation-selected reconstruction model, with mean ± standard deviation over seeds displayed below each feature. The composite score $Q _ { \mathrm { c o m p o s i t e } }$ is the weighted combination of the four reconstruction diagnostics defined in Appendix B. All entries are averaged over the seed ensemble.

The feature importances reveal a clear class-dependent structure. The STAR-side one-vs-rest boundaries emphasize low-end spectral and determinant information: STAR-Non is reconstructed primarily by $\varepsilon _ { \mathrm { { m i n } } } .$ , det, and $\mu _ { \gamma }$ , while STAR-Sph is reconstructed by $\varepsilon _ { \mathrm { m i n } } , \mu _ { \gamma }$ , and det. By contrast, the H-side boundaries emphasize gap statistics. H-Sph is dominated by $\mu _ { \gamma }$ , followed by $\sigma _ { \gamma }$ and $\gamma _ { 0 } ,$ while H-Non is dominated by $\mu _ { \gamma } ,$ with $\varepsilon _ { \mathrm { m i n } }$ and $\sigma _ { \gamma }$ appearing as secondary features.

Synthesizing the binary and one-vs-rest analyses, the multi-class latent space is organized by two coarse physical directions but resolved by class-specific nonlinear boundaries. At the binary level, topology is separated very cleanly and is primarily associated with gap statistics, while homology is associated with determinant and low-end spectral data. At the one-vs-rest level, STAR boundaries retain $\varepsilon _ { \mathrm { m i n } }$ and det, whereas H-graph boundaries are governed primarily by $\mu _ { \gamma }$ and related gap statistics. Thus the contrastive latent representation does not merely memorize the four labels: it arranges the classes in a low-dimensional geometry whose boundaries are largely reconstructible from the same determinant, spectral, and q-series support features identified throughout the preceding experiments.

## 4.4.3 Discussion of Latent Space Results

Synthesizing the results across the latent space experiments, the most defining characteristic is the convergence of interpretability. By filtering out high-variance noise, the contrastive learning objective aligns gradient sensitivity with global relevance, revealing a "dual-channel" mechanism where the network disentangles the problem into two orthogonal physical axes: a homology axis defined by $\varepsilon _ { \mathrm { { m i n } } }$ and a topology axis defined by $\mu _ { \gamma }$ . Crucially, the spontaneous emergence of this clean separation within a universal two-dimensional latent manifold, regardless of the initial bottleneck size, provides strong evidence that these two variables capture the intrinsic, irreducible coordinates of the ${ \widehat { Z } } .$ -invariant’s topological phase space.

## 4.5 Control Experiment: Derived Features

In the preceding experiments, the classifiers received raw or learned representations of the truncated $\widehat { Z }$ series, and the interpretability pipeline was used to infer which abstract quantities were reconstructed by the network. We now perform a direct control experiment: instead of giving the classifier the q-series, we give it only the ten derived features used throughout the regression and attribution analyses,

$$
( \operatorname* { d e t } , \varepsilon _ { \operatorname* { m i n } } , \varepsilon _ { \operatorname* { m a x } } , \mu _ { \varepsilon } , \sigma _ { \varepsilon } , \Delta , \gamma _ { 0 } , \mu _ { \gamma } , \sigma _ { \gamma } , c _ { \operatorname* { m a x } } ) .
$$

This asks whether these features are sufficient to solve the same homology and topology tasks, and whether the resulting classifiers rely on the same features identified indirectly in the rawq-series experiments.

For each task, we train an MLP classifier with architecture $\mathrm { ( d e p t h , w i d t h ) } = \left( 4 , 1 2 8 \right)$ , using the same train/test protocol and early-stopping criteria as in the preceding experiments. Since the classifier inputs are already the interpretable features, there is no separate feature-to-logit regression stage in this control. Instead, feature saliency, permutation importance, and SHAP are applied directly to the trained feature classifier. The multi-seed results are summarized in Tab. 11.

The feature controls solve all six tasks with very high accuracy. All topology controls reach perfect balanced accuracy within numerical precision, and the homology controls achieve BAS between $0 . 9 9 5 \pm 0 . 0 0 1$ and $1 . 0 0 0 \pm 0 . 0 0 0$ . This confirms that the ten derived quantities are sufficient to recover the class labels in the present datasets. The more informative question is therefore not whether the features contain enough information, but which features the direct classifier uses once the extraction problem has been removed.

<table><tr><td>Experiment</td><td>BAS</td><td>FS</td><td>PI</td><td>SHAP</td></tr><tr><td>HOM-Star</td><td> $1 . 0 0 0 \pm 0 . 0 0 0$ </td><td> $\varepsilon _ { \operatorname* { m i n } } ~ \mu _ { \gamma } ~ \Delta \nonumber$ </td><td>det  $\varepsilon _ { \operatorname* { m i n } } \ \mu _ { \gamma }$   $_ { 1 . 0 } ^ { 1 } , \ \dot { } _ { 2 . 0 } ^ { \prime } , \ 3 . 6$ </td><td> $\varepsilon _ { \mathrm { m i n } } ~ \mu _ { \gamma } ~ \sigma _ { \varepsilon }$   $1 . 0 ^ { \cdot } , _ { 2 . 0 ^ { \cdot } 3 . \bar { 3 } }$ </td></tr><tr><td>HOM-H</td><td> $0 . 9 9 8 \pm 0 . 0 0 1$ </td><td> $\varepsilon _ { \mathrm { m i n } } ~ \mathrm { d e t } ~ \sigma _ { \gamma }$   $1 . 2 ^ { \prime } \ 1 . 8 ^ { \prime } \ 3 . 0$ </td><td>det σγ  $\varepsilon _ { \mathrm { m i n } }$   $1 . 0 ^ { \cdot } , \ 2 . 0 ^ { \cdot } \ 3 . 4 ^ { \cdot }$ </td><td> $\varepsilon _ { \mathrm { m i n } } ~ \mathrm { d e t } ~ \sigma _ { \gamma }$   $1 . 4 ^ { \prime } , _ { 1 . 6 ^ { \prime } 3 . 2 }$ </td></tr><tr><td>HOM-All</td><td> $0 . 9 9 5 \pm 0 . 0 0 1$ </td><td> $\varepsilon _ { \mathrm { m i n } } ~ \Delta ~ \sigma _ { \gamma }$   $1 . 0 ^ { \cdot } , \ 2 . 2 ^ { \cdot } \ 2 . 8$ </td><td> $\varepsilon _ { \mathrm { m i n } } ~ \varepsilon _ { \mathrm { m a x } } ~ \sigma _ { \gamma }$  1.0’2.5’3.6</td><td> $\varepsilon _ { \mathrm { m i n } } ~ \sigma _ { \gamma } ~ \varepsilon _ { \mathrm { m a x } }$  1.0 ’2.0’4.1</td></tr><tr><td>TOP-Sph</td><td> $1 . 0 0 0 \pm 0 . 0 0 0$ </td><td> $\begin{array}{c} \mu _ { \gamma } \ : \sigma _ { \gamma } \ : \Delta  \\ { 1 . 0 ^ { \prime } \ : 2 . 0 ^ { \prime } \ : 3 . 4 } \end{array}$ </td><td> $\mu _ { \gamma } \ : , \ : \sigma _ { \gamma } \ : , \ : c _ { \mathrm { m a x } }$ </td><td> $\mu _ { \gamma } \ : \sigma _ { \gamma } \ : , \ : \varepsilon _ { \mathrm { m a x } }$ </td></tr><tr><td>TOP-Non</td><td> $1 . 0 0 0 \pm 0 . 0 0 0$ </td><td> $\varepsilon _ { \operatorname* { m i n } _ { 1 . 0 } , \ Y _ { 0 } , \ S _ { 3 . 4 } }$ </td><td> $\varepsilon _ { \operatorname* { m i n } } \varepsilon _ { \operatorname* { m a x } } \sigma _ { \gamma }$ </td><td> $\varepsilon _ { \operatorname* { m i n } } ~ \sigma _ { \varepsilon } ~ \sigma _ { \gamma }$   $1 . 0 \cdot 2 . { \overset { \cdot } { 4 } } ^ { , } 2 . { \overset { \cdot } { 8 } }$ </td></tr><tr><td>TOP-All</td><td> $1 . 0 0 0 \pm 0 . 0 0 0$ </td><td> ${ } _ { 1 . 0 } ^ { \sigma _ { \gamma } } , { \ Y _ { 0 } } _ { 3 . 2 } ^ { \mu _ { \gamma } }$ </td><td> ${ } _ { 1 . 0 } ^ { \sigma _ { \gamma } } , { } _ { 2 . 0 } ^ { \mu _ { \gamma } } , { } _ { 3 . 0 } ^ { \varepsilon _ { \mathrm { m i n } } }$ </td><td> $\operatorname { \varepsilon } _ { 1 . 0 } ^ { \sigma _ { \gamma } } , \mu _ { \gamma } , \varepsilon _ { \mathrm { m i n } }$ </td></tr></table>

Table 11: Multi-seed direct feature-classifier results. The classifier input is the ten-feature vector $( \operatorname* { d e t } , \varepsilon _ { \mathrm { m i n } } , \varepsilon _ { \mathrm { m a x } } , \mu _ { \varepsilon } , \sigma _ { \varepsilon } , \Delta , \gamma _ { 0 } , \mu _ { \gamma } , \sigma _ { \gamma } , c _ { \mathrm { m a x } } )$ rather than the truncated $\widehat { Z }$ sequence. BAS is reported as mean ± standard deviation over classifier seeds. The final columns report the three lowest mean feature ranks for feature saliency (FS), permutation importance (PI), and SHAP, with the average rank displayed below each feature.

For the homology tasks, the direct classifiers are dominated by the low-end spectrum. In HOM-Star, HOM-H, and HOM-All, $\varepsilon _ { \mathrm { m i n } }$ is ranked first by all three attribution methods, up to small seed fluctuations. The determinant appears as a secondary feature most clearly in HOM-H, while HOM-Star and HOM-All also use gap quantities such as $\mu _ { \gamma } , \Delta$ , and $\sigma _ { \gamma }$ . Thus the direct-feature homology classifier does not simply implement the exact rule | det $M | = 1$ . Instead, it learns a low-dimensional spectral proxy in which $\varepsilon _ { \mathrm { { m i n } } }$ is the most stable feature, with determinant and support-spacing information providing secondary corrections.

For topology classification, the control experiment separates the homology-sphere and nonsphere sectors. In TOP-Sph, where the determinant is fixed, the classifier relies almost entirely on the q-series gap profile: $\mu _ { \gamma }$ is ranked first and $\sigma _ { \gamma }$ second by FS, PI, and SHAP. This directly supports the interpretation that, in the fixed-determinant topology task, star-shaped and Hshaped graphs are distinguished through the spacing structure of the q-series.

The TOP-Non control behaves differently. Although the classifier again reaches perfect BAS, all three attribution methods rank $\varepsilon _ { \mathrm { { m i n } } }$ first. Gap features remain present, with $\sigma _ { \gamma } , \gamma _ { 0 } ,$ or ∆ appearing among the secondary features, but the dominant control signal is low-end spectral rather than purely gap-based. This is consistent with the full-sequence TOP-Non analysis, where determinant and eigenvalue information remained important once determinant variation was no longer fixed.

Finally, in TOP-All, the pooled topology control returns to a gap-dominated rule. All three attribution methods rank $\sigma _ { \gamma }$ first, and PI and SHAP rank $\mu _ { \gamma }$ second. The feature $\varepsilon _ { \mathrm { m i n } }$ remains visible as a secondary predictor, but the dominant pooled topology signal is the dispersion of the q-series gaps. This suggests that, once the derived features are provided explicitly, the most stable topology discriminator across the full dataset is the gap profile, even though the nonspherical sector alone can be solved by a low-end spectral shortcut.

Comparison with q-series models. The direct-feature controls clarify the role of the interpretability pipeline. The raw q-series classifiers must first extract relevant determinant, spectral, and support-spacing information from the truncated series. Depending on representation and sector, they sometimes emphasize high-variance spectral proxies, especially in the non-spherical topology tasks. When the ten derived features are supplied directly, however, the learned rules become sharper and more task-specific: homology classification is organized primarily by $\varepsilon _ { \mathrm { { m i n } } } ,$ with determinant information secondary, while topology classification is organized by gap statistics in the homology-sphere and pooled sectors, with a low-end spectral shortcut persisting in TOP-Non. The control experiment therefore supports the main conclusion of Sec. 4.3: the classifiers are not using opaque artifacts of the raw input representation, but are largely driven by a small set of determinant, spectral, and q-series support features.

## 5 Cobordism Experiments

## 5.1 Learning d from $\widehat { Z }$

## 5.1.1 Theoretical Background and Estimates

To set the stage for our results, we first review the known connections between $\widehat { Z }$ and the Heegaard Floer correction term d. One implicit relation discovered in [29] links the radial limit of $\widehat { Z }$ at $q \to i$ to the invariant d (mod 4):

$$
\frac { 1 } { 2 \pi } \arg \widehat { Z } \big | _ { q  i } \equiv \frac { 1 } { 4 } + \frac { 3 } { 4 } d \pmod { 1 } .\tag{11}
$$

Among the connections between $\widehat { Z }$ and d studied so far, the relation between $\Delta$ and d has received the most attention. For integer homology spheres [29] showed that:

$$
\Delta \equiv \frac 1 2 + d { \pmod { 1 } } .
$$

The interested reader may also find a $\mathsf { S p i n } ^ { c }$ -dependent version of this relation for rational homology spheres in [29]. Thereafter, [30] gave an explicit formula for $\Delta$ for Brieskorn spheres and compared $\Delta$ and d across various families of Brieskorn spheres, including some homology cobordant to $S ^ { 3 }$ . Most recently [31] pushed this comparison further, showing that for Seifert homology spheres $Y = \Sigma ( b _ { 1 } , \dots , b _ { r } )$

$$
\Delta ( Y ) = \frac { 1 } { 2 } - \frac { \kappa ( Y ) } { 4 }\tag{12}
$$

where $\kappa ( Y )$ , written $K ^ { 2 } + s$ in the prior literature [33], is the invariant recalled below. Additionally [31] further developed the comparisons between d and $\Delta .$ . We build on this story in this section.

We first introduce Dedekind sums following the notation in [33] as they will be useful in the formulas to appear. The generalized Dedekind sum is defined for $p \in \mathbb { Z } \setminus \{ 0 \} , q \in \mathbb { Z } _ { \geq 1 }$ , and $x , y \in \mathbb { R }$ by:

$$
s \left( p , q ; x , y \right) = \sum _ { i = 0 } ^ { q - 1 } \biggl \langle \frac { i + y } { q } \biggr \rangle \biggl \langle \frac { p ( i + y ) } { q } + x \biggr \rangle\tag{13}
$$

with the classical Dedekind sum being the specialization $s ( p , q ) : = s ( p , q ; 0 , 0 )$ . In the above $\langle \cdot \rangle$ is the sawtooth function defined for $x \in \mathbb { R }$ by $\begin{array} { r } { \langle x \rangle = x - \lfloor x \rfloor - { \frac { 1 } { 2 } } { \mathrm { ~ i f ~ } } x \not \in \mathbb { Z } } \end{array}$ and 0 otherwise. Expressing the manifold Y as the Seifert manifold $Y = M ( b ; { \frac { a _ { 1 } } { b _ { 1 } } } , \ldots , { \overline { { { \frac { \ l } { b _ { r } } } } } } )$ , we have the following expression for κ:

$$
\kappa ( Y ) = \left( 2 - r + \sum _ { i = 1 } ^ { r } \frac { 1 } { b _ { i } } \right) ^ { 2 } \frac { 1 } { e } + e + 5 - 1 2 \sum _ { i = 1 } ^ { r } s ( a _ { i } , b _ { i } )\tag{14}
$$

where e is the orbifold Euler number of Y. In [33] Borodzik and Némethi showed:

$$
d ( Y ) = { \frac { \kappa ( Y ) } { 4 } } - 2 \operatorname* { m i n } _ { k \geq 0 } \tau ( k )\tag{15}
$$

which together with (12) implies:

$$
d = - \Delta + \frac { 1 } { 2 } - 2 \operatorname* { m i n } _ { k \geq 0 } \tau ( k ) .\tag{16}
$$

In the above, $\tau$ is defined as

$$
\tau ( k ) = \sum _ { j = 0 } ^ { k - 1 } \delta _ { j } \quad \mathrm { a n d } \quad \delta _ { j } : = 1 - j b - \sum _ { i = 1 } ^ { r } \biggl \lceil \frac { j a _ { i } } { b _ { i } } \biggr \rceil .\tag{17}
$$

Let $P = b _ { 1 } \cdots b _ { r }$ and $\hat { b } _ { i } = P / b _ { i } ;$ ; there is an alternative formula for $\tau$ (cf. [33, eq. 3.6]) that we find instructive:

$$
\begin{array} { l } { \displaystyle \tau ( k ) = \frac { k ( k - 1 ) } { 2 P } + k \left( 1 - \frac { r } { 2 } \right) + \frac { r } { 2 } } \\ { \displaystyle \quad + \sum _ { i = 1 } ^ { r } \left( - \frac { 1 } { 2 } \left. \frac { k a _ { i } } { b _ { i } } \right. + \frac { 1 } { 2 } \left\lfloor \frac { k - 1 } { b _ { i } } \right\rfloor - s \left( \hat { b } _ { i } , b _ { i } ; \frac { k } { b _ { i } } , 0 \right) + s \left( \hat { b } _ { i } , b _ { i } \right) \right) . } \end{array}\tag{18}
$$

We now fix $r = 3$ for the remainder of this subsubsection. By inspecting (18) one can show, as we do in Appendix D, that:

$$
\left| \operatorname* { m i n } _ { k \ge 0 } \tau ( k ) + \frac { P } { 8 } \left( 1 - \sum _ { i = 1 } ^ { 3 } \frac { 1 } { b _ { i } } \right) ^ { 2 } \right| \le \sum _ { i = 1 } ^ { 3 } b _ { i }\tag{19}
$$

Combining (16) with the behavior of min τ we see:

$$
\left| d + \Delta - \frac { 1 } { 2 } - \frac { P } { 4 } \left( 1 - \sum _ { i = 1 } ^ { 3 } \frac { 1 } { b _ { i } } \right) ^ { 2 } \right| \leq 2 \sum _ { i = 1 } ^ { 3 } b _ { i } ,\tag{20}
$$

which, as far as we know, is a new observation in the $\widehat { Z }$ literature. Interestingly, by comparing the expression for $\Delta$ in [31, eq. 7.1] with (19), one sees that $\Delta$ and ${ \textstyle | 2 \operatorname* { m i n } _ { k \geq 0 } \tau ( k ) | }$ | are similar in magnitude. In comparison, d is much smaller than either quantity (cf. (49)) and varies erratically as a function of the $b _ { i }$ due to the presence of Dedekind sums as seen in (18).

## 5.1.2 Regression of d from $\widehat { Z }$ Exponents

## The Experiment

Given the relation (16), we design the regression task around $d + \Delta - { \textstyle \frac { 1 } { 2 } } = - 2 \operatorname* { m i n } _ { k \ge 0 } \tau ( k )$ Writing

$$
{ \widehat { Z } } ( Y ; q ) = q ^ { \Delta } \left( 1 + c _ { n _ { 1 } } q ^ { n _ { 1 } } + c _ { n _ { 2 } } q ^ { n _ { 2 } } + \cdots \right)\tag{21}
$$

where $c _ { n _ { i } } \in \mathbb { Z }$ are all nonzero and $0 < n _ { 1 } < n _ { 2 } < \cdots$ with $n _ { i } \in \mathbb { N } ,$ we ask whether the leading exponent $\Delta$ and the ordered nonzero exponent offsets $n _ { i }$ contain enough information to predict the τ-function contributions. This formulation is mathematically equivalent to predicting $d ,$ since d is recovered from (16). We emphasize, however, that this is not the same as asking a generic smooth regressor to learn the full correction term formula directly from the raw Seifert parameters $( b _ { 1 } , b _ { 2 } , b _ { 3 } )$ : the latter involves floor, ceiling, and Dedekind-sum corrections, and is not well approximated by the simple smooth models tested here.

Two families of datasets enter this section. The main three-fiber dataset $\mathcal { D } _ { 3 } ^ { 2 0 }$ starts from the pairwise-coprime triples $2 \leq b _ { 1 } < b _ { 2 } < b _ { 3 } \leq 4 4$ and keeps the 685 manifolds whose first twenty nonzero ${ \widehat { Z } } .$ -coefficients occur at exponents at most $1 . 2 8 \times 1 0 ^ { 4 }$ , yielding the leading exponent $\Delta$ and nineteen subsequent offsets per manifold; in effect this restricts $b _ { 1 } b _ { 2 } b _ { 3 } \leq 2 4 4 2$ The per-fiber-count sets $\mathcal { D } _ { r } , r = 3 , 4 , 5 ,$ consist of the Seifert fibered integer homology spheres with pairwise-coprime multiplicities $2 \leq b _ { 1 } < \cdots < b _ { r } \leq 4 5 , 3 0$ , 20 for $r = 3 , 4 , 5$ respectively, oriented so that the orbifold Euler number is negative, generated independently under a deeper extraction (first one hundred nonzero coefficients at exponents below $3 . 3 \times 1 0 ^ { 6 }$ ; see the reproduction notebook [34]), giving $n = 3 , 0 2 0 , 1 , 7 4 3$ , and 306 manifolds respectively.

All results in this section and the following subsection use $\mathcal { D } _ { 3 } ^ { 2 0 }$ , except where a $\mathcal { D } _ { \iota }$ is named explicitly: Table 12, the Kernel-Ridge ∆-augmentation, the cross-r transfer, and the prefixscaling analysis. Throughout, a model’s input is a prefix of this offset sequence, the first L offsets $( n _ { 1 } , \ldots , n _ { L } )$ , optionally augmented by $\Delta$ and further features as stated; we call L the prefix length.

## Results

We began with a sweep over regressor classes and input dimensions. Throughout, each reported $R ^ { 2 }$ is the mean ± standard deviation over ten random seeds; for each seed we average the heldout $R ^ { 2 }$ over a freshly shuffled five-fold partition. For the deterministic models reported here, the fold partition is the only seed-dependent quantity. Exact-integer accuracies follow the same scheme, scored once per seed on the pooled out-of-fold predictions (one held-out prediction per manifold).

On the three-fiber Brieskorn dataset, using the first four exponent offsets $( n _ { 1 } , \ldots , n _ { 4 } )$ , the strongest models were a degree-three polynomial regression and Kernel Ridge regression with an RBF kernel, achieving respectively $R ^ { 2 } = 0 . 9 9 9 5 \pm 0 . 0 0 0 0$ and $R ^ { 2 } = 0 . 9 9 9 3 \pm 0 . 0 0 0 2$ , followed closely by degree-two polynomial regression with $R ^ { 2 } = 0 . 9 9 9 0 \pm 0 . 0 0 0 0$ . Linear regression remained lower, at $R ^ { 2 } = 0 . 9 4 5 3 \pm 0 . 0 0 0 4$ . Adding more exponent offsets did not systematically improve these scores, and in some cases degraded performance, indicating that the smooth, large-scale variation o $\mathrm { f } - 2 \operatorname* { m i n } _ { k \geq 0 } \tau ( k )$ is already captured by a short exponent prefix. The performance of the degree-two model suggests that the dominant trend is essentially quadratic in these leading offsets, although, as discussed below, the exact integer-valued correction contains finer numbertheoretic structure not captured by such a smooth approximation alone.

Feature-importance analysis with three attribution methods (permutation importance, mean |SHAP|, and MLP gradient saliency (FS), each recomputed over ten re-drawn train/test splits) yields the consensus hierarchy $n _ { 3 } > n _ { 1 } > n _ { 4 } > n _ { 2 }$ (mean ranks 1.50, 2.07, 2.43, 4.00). Permutation importance and |SHAP| reproduce this ordering on every split, while gradient saliency instead ranks $n _ { 4 }$ first with $n _ { 1 }$ and $n _ { 3 }$ statistically indistinguishable, as the FS bars of Figure 24 show (per-split rank tables in [34]). The dominance of $n _ { 3 }$ is consistent with the arithmetic structure of the Brieskorn false-theta expansion: among the early offsets, $n _ { 3 }$ typically mixes several Seifert parameters, heuristically $n _ { 3 } \approx ( b _ { 1 } - 1 ) ( b _ { 2 } + b _ { 3 } )$ on much of the dataset. Separately, degree-two polynomial regression reconstructs $\kappa ( Y ) / 4$ from $( n _ { 1 } , \ldots , n _ { 4 } )$ with $R ^ { 2 } = 0 . 9 9 7 9 \pm 0 . 0 0 0 0$ and $- 2 \mathrm { m i n } _ { k } \tau ( k )$ with $0 . 9 9 9 0 \pm 0 . 0 0 0 0$

To test whether this phenomenon persists beyond three singular fibers, we expanded the dataset to Seifert fibered integer homology spheres with $r = 4$ and $r = 5$ exceptional fibers. The same qualitative pattern persists: low-degree polynomial and kernel models achieve very high $R ^ { 2 }$ from short exponent prefixes, and adding $\Delta$ further improves the reconstruction. The number of exponent offsets required to reach a fixed high-accuracy threshold grows with r. Under the Kernel Ridge RBF protocol, the smallest prefix length reaching $R ^ { 2 } \geq 0 . 9 9 9$ is ${ \cal L } _ { 9 9 . 9 \% } = 4 , 8 $ , 15 for $r = 3 , 4 , 5 ,$ , identical across all ten seeds. This is consistent with the quadratic count $r ^ { 2 } - 2 r$ ， which matches exactly at $r = 4 , 5$ and is off by one at $r = 3$ (empirically 4, not 3). We treat this as an empirical scaling law rather than a theorem: the higher-r regimes have fewer samples and larger candidate feature spaces, and the threshold estimates depend on the model class and available prefix length.

Importantly, the polynomial degree need not grow with r in these experiments. A degree-two polynomial remains highly effective across all tested fiber counts, especially when $\Delta$ is included as an additional feature. Higher-degree polynomials, by contrast, are more prone to overfitting because the number of cross-terms grows rapidly relative to the available sample size.
<table><tr><td>1</td><td> $r = 3$  一</td><td> $r = 4$ </td><td>一  $r = 5$ </td></tr><tr><td>Poly(2), offsets only (best over L)</td><td> $1 . 0 0 0 0 \pm 0 . 0 0 0 0$ </td><td> $1 . 0 0 0 0 \pm 0 . 0 0 0 0$ </td><td> $0 . 9 9 5 0 \pm 0 . 0 0 1 9$ </td></tr><tr><td> $\mathrm { P o l y } ( 2 )$   $\mathrm { o f f s e t s } + \Delta$  2</td><td> $1 . 0 0 0 0 \pm 0 . 0 0 0 0$ </td><td> $1 . 0 0 0 0 \pm 0 . 0 0 0 0$ </td><td> $1 . 0 0 0 0 \pm 0 . 0 0 0 0$ </td></tr></table>

Table 12: Cross-validated $R ^ { 2 }$ for a degree-2 polynomial regressing $\begin{array} { r } { d + \Delta - \frac { 1 } { 2 } } \end{array}$ on the $\widehat { Z }$ exponent offsets of Seifert fibered integer homology spheres with r exceptional fibers $( n = 3 0 2 0 , 1 7 4 3 , 3 0 6$ for $r = 3 , 4 , 5 )$ . The best-over-L scores in the offsets-only row are attained at $L = 6 ,$ 12, and 16 for $r = 3 , 4 , 5$ respectively. The $+ \Delta$ entries are evaluated at fixed base prefix lengths $L = 4 , 7 , 1 0$ for $r = 3 , 4 , 5 \colon$ : the empirical Poly(2) $R ^ { 2 } \geq 0 . 9 9$ crossings for $r = 3 , 4 ,$ , with $L = 1 0$ chosen for $r = 5 ,$ whose crossing is unstable at $n = 3 0 6$ . These evaluation defaults are distinct from the 99.9% thresholds $L _ { 9 9 . 9 \% }$ discussed in the text.

The leading exponent $\Delta$ carries additional nontrivial information. On the $\mathcal { D } _ { r }$ datasets, at the base prefix lengths of Table 12, adding $\Delta$ to the offset array improves the Kernel Ridge reconstruction of $d + \Delta - { \frac { 1 } { 2 } }$ : in the $r = 3$ case the score increases from 0.9998 using offsets alone to 1.0000 at the reported precision, while for $r = 4$ it increases from 0.9943 to 0.9999, and for $r = 5$ from 0.9928 to 0.9994. On $\mathcal { D } _ { 3 } ^ { 2 0 }$ at $L = 4 .$ , adding $\Delta$ also raises the degree-two polynomial reconstruction of $\kappa ( Y ) / 4$ to $1 . 0 0 0 0 \pm 0 . 0 0 0 0$ , and the reconstruction of $- 2 \mathrm { m i n } _ { k } \tau ( k )$ to $0 . 9 9 9 7 \pm 0 . 0 0 0 0$

The exact coefficients of the quadratic map are fiber-count-dependent. A polynomial trained on three-fiber manifolds does not transfer directly to four-fiber manifolds, giving $R ^ { 2 } < 0$ . However, transfer between adjacent higher fiber counts can succeed: an $r = 4$ model evaluated on $r = 5$ gives $R ^ { 2 } = 0 . 6 4 8 5$ . This asymmetry is consistent with the possibility that lower-fiber polynomial relations arise as specializations of a higher-dimensional arithmetic structure, although we regard this as suggestive rather than conclusive.

The exponent offsets encode more than the scalar correction term alone. In the three-fiber experiments, from the first four offsets, mi $\mathsf { \Omega } _ { \mathsf { l } _ { k } } \tau ( k )$ is reconstructed with $R ^ { 2 } = 0 . 9 9 9 0$ , the range of τ with $R ^ { 2 } = 0 . 9 9 9 8$ , and the location of the minimum $k ^ { * } = \arg \operatorname* { m i n } _ { k \geq 0 } \tau ( k )$ with $R ^ { 2 } = 0 . 9 9 5 1$ Even after projecting out the variance linearly correlated with the d-invariant, the exponent offsets still explain 96% of the variance in $k ^ { * }$ , indicating that the exponent spectrum carries information about the shape of the τ-function beyond the value of $d$ itself.

In summary, for the Seifert fibered integer homology spheres in our datasets, the macroscopic shape of $\begin{array} { r } { d + \Delta - \frac { 1 } { 2 } } \end{array}$ is predominantly captured by a degree-two polynomial in a short prefix of the exponent support, supplemented by the leading exponent $\Delta$ . The exponent spectrum of $\widehat { Z }$ also encodes several related geometric quantities, including the minimum, argmin, and range of the τ-function, as well as $\kappa ( Y ) / 4$

These $\mathrm { \ h i g h } { - R ^ { 2 } }$ results should be interpreted as smooth approximation results rather than exact arithmetic reconstruction. The success of the exponent-offset features is nevertheless consistent with a more rigid arithmetic mechanism. In the three-fiber Brieskorn case, the false-theta sector shifts contain pairwise combinations of the Seifert multiplicities, such as

$$
b _ { i } b _ { j } - b _ { i } - b _ { j } ,
$$

and once the three pairwise offsets are identified one can recover $( b _ { 1 } , b _ { 2 } , b _ { 3 } )$ by elementary square-root formulas. The ordered exponent offsets are not labeled by sector, and the pairwise offsets need not coincide with the first three ordered offsets because other sector contributions can interleave. However, direct enumeration on the finite three-fiber dataset verifies that the prefix $( n _ { 1 } , n _ { 2 } , n _ { 3 } , n _ { 4 } )$ is injective in the Brieskorn triple. For $r > 3$ , we treat the analogous injectivity statement as empirical: the observed scaling in the number of required offsets suggests that a small prefix of the exponent support captures enough Seifert arithmetic to approximate the τ-function contribution to $d _ { z }$ , but we do not claim a closed-form inverse in general.

## 5.1.3 Towards Exact Integer Predictions

Despite the near-perfect $R ^ { 2 }$ scores, the polynomial predictions are continuous, whereas the target

$$
t = - 2 \operatorname* { m i n } _ { k \geq 0 } \tau ( k )
$$

is integer-valued, and in this setting lies in 2Z. Directly rounding the degree-two polynomial prediction for t from the four offsets alone gives exact agreement on only $1 7 . 9 \% \pm 0 . 7 \%$ of the dataset. The residuals are tightly bounded, with in-sample standard deviation approximately $2 . 3 0 ,$ maximum 8.5, and cross-validated MAE 1.86, but they are not well explained by the tested polynomial functions of the exponent offsets. This is consistent with the structure of $\tau ( k )$ whose floor and ceiling terms, together with the Dedekind-sum contributions in the Seifert formula, introduce arithmetic corrections beyond the smooth quadratic trend. Appending $\Delta$ as a fifth input feature doubles the exact-recovery rate to $3 6 . 1 \% \pm 0 . 8 \%$ , the leftmost bar of Figure 25.

To isolate the finer arithmetic correction, we modified the target and augmented the feature set. First, we trained the regressor on $t / 2 \in \mathbb { Z }$ rather than $t \in 2  { \mathbb { Z } }$ , so that rounding errors are measured on the natural integer scale of the half-target, lifting exact recovery to $6 1 . 5 \% \pm 1 . 0 \%$ $( \mathrm { t h e } ^ { \mathfrak { a } } \div 2 ^ { \mathfrak { n } }$ bar). Separately, widening the input from the first four offsets to all nineteen lifts this further to $7 3 . 8 \% \pm 0 . 5 \%$ (“+all offsets”), recovering nearly half of what the Dedekind augmentation below achieves from the first four alone. Second, returning to the first four offsets, we supplemented the exponent features with symmetrized Dedekind-sum features derived from the Seifert data: writing $P = b _ { 1 } b _ { 2 } b _ { 3 }$ , these are $D _ { i } = s ( P / b _ { i } , b _ { i } )$ for $i = { 1 , 2 , 3 }$ (each a symmetric function of the two complementary multiplicities; up to sign, the sums $s ( a _ { i } , b _ { i } )$ of (14)) together with their total $D _ { 1 } + D _ { 2 } + D _ { 3 }$ . The symmetrization is important: raw, unsymmetrized Dedekind sums do not respect the permutation symmetry of the Brieskorn triple and empirically degrade the exact recovery rate. Using the nine-dimensional feature set consisting of $\Delta _ { : }$ , four exponent offsets $( n _ { 1 } , . . . , n _ { 4 } )$ , and four symmetrized Dedekind-sum features, a degree-two polynomial expansion followed by linear regression gives exact integer recovery of d on $8 8 . 6 \% \pm 0 . 6 \%$ of the Brieskorn dataset.

The remaining failures are not catastrophic. In the failed cases, the predicted $t / 2$ differs from the true value by exactly $\pm 1$ . These errors concentrate among manifolds with large Seifert multiplicities, large product $b _ { 1 } b _ { 2 } b _ { 3 }$ , and large leading exponent $\Delta$

We also tested whether the Dedekind-sum features used in this arithmetic augmentation are themselves recoverable from the ${ \widehat { Z } } .$ -exponent data. This auxiliary experiment separates three notions: finite identifiability, smooth regression, and arithmetic reconstruction. First, the short prefix $( n _ { 1 } , n _ { 2 } , n _ { 3 } , n _ { 4 } )$ has no collisions in the finite three-fiber Brieskorn dataset, either in the Brieskorn triple $( b _ { 1 } , b _ { 2 } , b _ { 3 } )$ or in the associated Dedekind-feature vector. Thus, on this dataset, the Dedekind features are not independent external information; they are determined by the leading exponent support. Second, we trained regressors from longer prefixes $( \Delta , n _ { 1 } , . . . , n _ { L } )$ to the Dedekind features. Smooth recovery improves with $L ,$ , but remains far from exact for the full Dedekind vector: the aggregate $R ^ { 2 }$ rises from approximately 0.19 for $L = 4$ to roughly 0.41 for $L \ge 1 2$ . The total Dedekind contribution is much more predictable than the individual symmetrized components, indicating that the arithmetic combination relevant for d is smoother than the full vector of Dedekind data. Third, nonparametric and parameter-recovery tests show that injectivity does not automatically imply train/test interpolation: nearest-neighbor recovery of held-out Brieskorn triples is poor even though full-catalog collisions vanish, reflecting the arithmetic, non-Euclidean nature of the map. We therefore interpret the Dedekind features as structured Seifert arithmetic latent in the ${ \widehat { Z } } .$ -exponent support, but not as information that the tested low-degree regressors extract reliably from short prefixes alone. Efficient recovery appears to require either longer exponent prefixes together with richer models, or an explicit arithmetic decoding step from exponent support to Seifert data.

## 6 Other Experiments

## 6.1 Multi-class classification: Singular fibers

We first extend the binary classification framework to a genuinely multi-class task. The goal is to determine the number of singular fibers in a Seifert manifold, equivalently the degree of the central node in the corresponding star-shaped plumbing graph. We generated a balanced dataset of 30,000 samples, with 6,000 examples in each of the five classes corresponding to 3, 4, 5, 6, and 7 singular fibers. The classifier was trained on the value-index pair representation and achieved balanced accuracy 0.9426 on the test set.

This task is substantially more geometrically complex than the binary experiments in Sec. 4.3. In the binary setting, the classifier logits can often be interpreted through a single separating direction, and the feature-to-logit regression problem reduces to reconstructing a comparatively simple decision geometry. Here, the classifier output is five-dimensional, and the logit cloud is organized by multiple pairwise class separations. Visualizing the logits reveals a piecewise structure resembling intersecting affine regions rather than a single dominant separating direction. This increased nonlinearity is reflected in the interpretability pipeline: the regression network reconstructs the logits with reduced quality, achieving $R ^ { 2 } = 0 . 8 0 0 6$ . Thus the ten-feature dictionary still captures a substantial part of the learned representation, but the multi-class geometry is not as cleanly reducible to a single linear feature mechanism.

Despite this reduction in regression quality, the feature analysis remains informative. The most stable signals are spectral and support-spacing quantities that vary systematically with the number of arms in the star-shaped graph. In particular, the eigenvalue dispersion of the plumbing matrix and the gap statistics of the q-series provide natural summaries of the increasing graph complexity. As the number of singular fibers changes, both the spectrum of M and the distribution of exponents in the $\widehat { Z }$ series are altered. The classifier appears to exploit this combined information: spectral-spread features capture changes in the quadratic form determined by $M ^ { - 1 }$ , while gap statistics capture the induced changes in the support structure of the truncated q-series.

The singular-fiber experiment therefore supports two conclusions. First, the value-index representation remains effective beyond binary tasks, achieving high accuracy even when the output space contains five ordered graph-complexity classes. Second, the interpretability picture becomes less one-dimensional. Unlike the binary homology and topology tasks, where determinant, low-end spectral data, or gap statistics often dominate a single separating direction, the singular-fiber task requires a collection of class-dependent boundaries. The resulting feature analysis should therefore be interpreted as identifying the dominant coordinates of a multiclass decision geometry, not as isolating one universal scalar invariant for the number of singular fibers.

## 6.2 Learning $S L _ { 2 } ( \mathbb { Z } )$ orbit

In the final experiment, we move from classification to a regression task motivated by the modular properties of ${ \widehat { Z } } .$ . The goal is to test whether a network can predict the size of the $S L _ { 2 } ( \mathbb { Z } )$ orbit associated with a $\widehat { Z }$ -invariant directly from truncated $q \cdot$ series data. See [11, 13] for discussions of the $S L _ { 2 } ( \mathbb { Z } )$ orbits underlying these invariants.

For a general plumbed 3-manifold there is no simple closed formula for the orbit size. A useful controlled family is provided by Brieskorn spheres $\Sigma ( b _ { 1 } , b _ { 2 } , b _ { 3 } )$ , where the fiber orders $b _ { i }$ are pairwise coprime and square-free. In this case, the orbit size α is determined by the Milnor number:

$$
\alpha \left( \Sigma ( b _ { 1 } , b _ { 2 } , b _ { 3 } ) \right) = \frac { ( b _ { 1 } - 1 ) ( b _ { 2 } - 1 ) ( b _ { 3 } - 1 ) } { 4 } = \frac { \mu } { 4 } .\tag{22}
$$

We use this family as an in-distribution benchmark for learning the orbit size from the corresponding truncated q-series.

The dataset consists of q-series expansions for Brieskorn spheres satisfying the coprime and square-free conditions, ranging from Σ(2, 3, 5) to Σ(19, 67, 115). Since these q-series are sparse, we computed expansions to order $O ( 1 0 ^ { 8 } )$ , yielding samples with between 64 and 4619 nonzero coefficients. We fixed the input length to 64 terms. The input is a serialized value-index representation in which the coefficient and exponent arrays are concatenated:

$$
[ c _ { 0 } , \ldots , c _ { 6 3 } , e _ { 0 } , \ldots , e _ { 6 3 } ] .
$$

This gives a 128-dimensional input vector. We focus on this ablated representation, without appending the leading exponent $\Delta ,$ , so that the network must infer the orbit size from the coefficient–exponent sequence itself rather than from an explicitly supplied leading shift.

We trained a transformer encoder on this flat-block representation using a projection width of 32, two encoder layers, four attention heads, and one-dimensional adaptive average pooling. The input is ordered as

$$
[ c _ { 0 } , \ldots , c _ { 6 3 } , e _ { 0 } , \ldots , e _ { 6 3 } ] ,
$$

with scalar token shape (128, 1). The regression target was standardized during training and inverse-transformed for reporting on the original orbit-size scale. The experiment was repeated over a 30-seed ensemble. On held-out in-distribution Brieskorn-sphere samples, the transformer achieves

$$
R _ { \mathrm { p r e d } } ^ { 2 } = 0 . 9 9 2 9 \pm 0 . 0 0 8 3 ,
$$

where the mean and standard deviation are taken over seeds. Aggregating held-out predictions across the ensemble gives an ensemble-mean prediction score

$$
R _ { \mathrm { p r e d } } ^ { 2 } = 0 . 9 9 6 7 ,
$$

with linear calibration

$$
R _ { \mathrm { l i n } } ^ { 2 } = 0 . 9 9 6 8 , ~ \widehat { \alpha } = 0 . 9 8 8 3 \alpha + 7 0 0 . 5 3 .
$$

Thus the flat-block transformer learns a highly accurate in-distribution interpolation rule for this Brieskorn-sphere family. The fit is not exact, but the seed-to-seed variation is small: the per-seed calibration slopes are $0 . 9 8 9 1 { \scriptstyle \pm 0 . 0 1 8 3 }$ , and the per-seed prediction scores concentrate around $R _ { \mathrm { p r e d } } ^ { 2 } = 0 . 9 9 2 9 \pm 0 . 0 0 8 3$

Averaging predictions over the unique held-out samples appearing across the seed ensemble yields the calibration shown in Fig. 23a. The ensemble-mean prediction achieves

$$
\begin{array} { r } { R _ { \mathrm { { s e e d } } } ^ { 2 } = 0 . 9 9 2 9 \pm 0 . 0 0 8 , \quad R _ { \mathrm { { p r e d } } } ^ { 2 } = 0 . 9 9 6 7 , \quad R _ { \mathrm { { l i n } } } ^ { 2 } = 0 . 9 9 6 8 , \quad \widehat { \alpha } = 0 . 9 8 8 3 \alpha + 7 0 0 . 5 3 , } \end{array}
$$

over 5329 unique held-out samples. The calibration slope remains close to one, confirming that the model captures the global scale of the orbit size, while the small positive intercept and slope below one indicate mild regression toward the mean at the largest orbit sizes. The scatter in Fig. 23a shows that the largest residuals are concentrated among a small number of high-orbit examples and lower-α outliers, but the overall ensemble calibration is substantially tighter than in the earlier robustness run. We therefore interpret the flat-block transformer as learning a robust in-distribution interpolation rule within the square-free Brieskorn-sphere family, rather than an exact closed-form formula for the $S L _ { 2 } ( \mathbb { Z } )$ -orbit size.

To probe what the transformer uses in the serialized input, we apply gradient-based attribution to the trained models: saliency, integrated gradients, gradient×input, and attention rollout. Because the input dimension is large and the task is a regression problem, we restrict the diagnostics to these attribution methods rather than running the full FS/PI/SHAP pipeline. The attributions concentrate on the active coefficient–exponent region and displays an approximately periodic modulation across the serialized terms, as shown in Fig. 7. This structure is compatible with the partial-theta decomposition of the Brieskorn-sphere Zb-series discussed below. Importantly, this should be interpreted as evidence for a structured in-family interpolation mechanism, not as evidence that the transformer has learned a general algorithm for $S L _ { 2 } ( \mathbb { Z } )$ - orbit size.

To understand this, note that the ${ \widehat { Z } } .$ -invariants for Brieskorn spheres $\Sigma ( b _ { 1 } , b _ { 2 } , b _ { 3 } )$ take the following form<sup>4</sup>:

$$
\begin{array} { c } { { \widehat { Z } ( \Sigma ( b _ { 1 } , b _ { 2 } , b _ { 3 } ) ; \tau ) = c q ^ { \tilde { \Delta } } \displaystyle \sum _ { ( \epsilon _ { 1 } , \epsilon _ { 2 } , \epsilon _ { 3 } ) \in ( { \mathbb Z } / 2 ) ^ { 3 } } ( - 1 ) ^ { \sum _ { i } \epsilon _ { i } } \widetilde { P } _ { b _ { 1 } b _ { 2 } b _ { 3 } , \chi _ { \epsilon _ { 1 } , \epsilon _ { 2 } , \epsilon _ { 3 } } } ( q ) } } \\ { { \chi _ { \epsilon _ { 1 } , \epsilon _ { 2 } , \epsilon _ { 3 } } : = b _ { 1 } b _ { 2 } b _ { 3 } - ( - 1 ) ^ { \epsilon _ { 1 } } b _ { 1 } b _ { 3 } - ( - 1 ) ^ { \epsilon _ { 2 } } b _ { 1 } b _ { 2 } - ( - 1 ) ^ { \epsilon _ { 3 } } b _ { 2 } b _ { 3 } } } \end{array}\tag{23}
$$

for some $\tilde { \Delta } \in \mathbb { Q }$ and $c \in \mathbb { C }$ . In the above, $\tilde { P } _ { m , r } ( q )$ denotes the partial theta function

$$
\tilde { P } _ { m , r } ( q ) = \sum _ { k \geq 0 } q ^ { \frac { ( 2 m k + r ) ^ { 2 } } { 4 m } } .
$$

From this we see that the ordered set $\left\{ \boldsymbol { e } _ { i } \right\}$ is naturally split into the following set:

$$
e _ { 8 n + i } = { \frac { ( 2 b _ { 1 } b _ { 2 } b _ { 3 } n + r _ { i } ) ^ { 2 } - r _ { 1 } ^ { 2 } } { 4 b _ { 1 } b _ { 2 } b _ { 3 } } }
$$

for $i \in \left\{ 1 , 2 , . . . , 8 \right\}$ . The quasi-periodicity simply indicates that the attribution profiles assign comparable weight to terms belonging to a specific partial theta function $\tilde { P } _ { b _ { 1 } b _ { 2 } b _ { 3 } , r _ { i } }$

To test generalization, we evaluated the trained networks on out-of-distribution q-series whose $S L _ { 2 } ( \mathbb { Z } )$ orbit sizes were computed in [11, Tab. 14]. These examples lie outside the squarefree Brieskorn-sphere family used for training. Both the MLP and transformer fail this test:

SL2(Z) orbit transformer attribution: seed-normalized ensemble  
![](images/9332f4cc1959b984f0397c833d6d31440ec0e7dcd0c63213bbb571a11e986a53.jpg)  
Figure 7: $S L _ { 2 } ( \mathbb { Z } )$ -orbit regression: attribution diagnostics for the flat-block transformer. The input is ordered as $[ c _ { 0 } , \ldots , c _ { 6 3 } , e _ { 0 } , \ldots , e _ { 6 3 } ]$ . The four panels show gradient saliency, integrated gradients, gradient-times-input, and attention rollout, averaged over the 30-seed robustness ensemble after normalizing each seed by its total attribution mass. The dashed red line separates the coefficient block from the exponent block. Bar color denotes the within-block index modulo $^ { 8 , }$ reflecting positional organization associated with the eight interleaved partial-theta sectors in the Brieskorn-sphere expression.

neither architecture correctly predicts the OOD orbit sizes, and the OOD regression quality is worse than a baseline expectation, with $R ^ { 2 } < 0$ . This failure is important. It shows that the high in-distribution performance should not be interpreted as learning the general $S L _ { 2 } ( \mathbb { Z } )$ orbit problem. Rather, the networks learn a highly accurate interpolation rule within a restricted Brieskorn-sphere family, where the orbit size is tightly correlated with the growth pattern of the exponent sequence.

## 7 Conclusions and outlook

In this work, we have demonstrated that finite truncations of ${ \widehat { Z } } .$ -invariants for weakly negative plumbed 3-manifolds carry significant topological information that can be effectively extracted and interpreted using machine learning. Below we discuss the key conclusions.

• Generalizable Data Pipeline. While we focused on the ${ \widehat { Z } } .$ -invariants of plumbed 3- manifolds in this work, we have established a systematic pipeline for handling mathematical data structured as (truncated) infinite q-series. This framework is readily applicable to series originating from diverse sources, such as Nahm sums, q-expansions of modular forms, partition functions of other types of quantum field theories, or other quantum invariants.

• Novel Connections to Cobordism. Our experiments revealed a relationship between the q-series data of the ${ \widehat { Z } } .$ -invariant and three-dimensional homology cobordism invariants: regression models fit $d { + } \Delta { - } \frac { 1 } { 2 }$ , equivalently the τ-function contribution in d $= - \Delta + { \textstyle \frac { 1 } { 2 } } - 2 \operatorname* { m i n } _ { k } \tau ( k )$ with $> 9 9 . 9 \%$ explained variance from a small number of nonzero exponent offsets of ${ \widehat { Z } } .$ Supplementing these exponent features with symmetrized Dedekind sums yields a learned quadratic approximation to $- 2 \mathrm { m i n } _ { k } \tau ( k )$ giving an accurate $( \sim 9 0 \% )$ integer prediction for d. For three-fiber Brieskorn spheres, this success is partially explained by the arithmetic structure of the false-theta expansion: the partial-theta sector shifts contain low-order combinations of the Seifert multiplicities $b _ { i } ,$ and in the finite threefiber dataset the leading exponent offsets are injective in the Brieskorn triple $( b _ { 1 } , b _ { 2 } , b _ { 3 } )$ ; once these multiplicities are known, the normalized Seifert numerators $a _ { i }$ are fixed by the homology-sphere condition, so the quantities entering the known formula for $d ,$ including the Dedekind sums and the τ-function, become arithmetic functions of the same underlying Seifert data. Thus the high regression accuracy should be interpreted as evidence that, in this Brieskorn family, the exponent support carries enough Seifert arithmetic for the network to approximate the correction-term formula; the related $S L _ { 2 } ( \mathbb { Z } )$ -orbit experiment supports the same sector-arithmetic interpretation, while also cautioning against assuming out-of-family generalization.

• Interpretability Beyond “Black Boxes”. Finally, our methodology—specifically the use of an interpretability regression network to map mathematical features to classifier logits— provides a window into the network’s predictive mechanism. By identifying the specific arithmetic and geometric data used by the model, we highlight the fact that machine learning methods do not necessarily lead only to black-box results. Instead, suitably designed, explainable learning tasks serve to elucidate complex mathematical structures, guiding researchers toward the discovery of human-understandable conjectures. A concrete illustration is the network’s use of $\varepsilon _ { \mathrm { { m i n } } }$ as a proxy for the determinant condition in the homology classification tasks. The spectral bound of Lemma 1 provides a justification for this strategy, and at the same time offers a precise structural account of the classifier’s residual generalization error (see §4.2.1).

Acknowledgements. We thank Giorgi Butbaia, Mrunmay Jagadale, Davide Passaro and Josef Svoboda for useful conversations. SG was supported by the Simons Collaboration Grant on "New Structures in Low-Dimensional Topology", by the NSF grant DMS-2245099, and by the U.S. Department of Energy, Office of Science, Office of High Energy Physics, under Award No. DE-SC0011632. R.-K. S. is supported by an Outstanding Young Scientist Grant (RS-2025-00516583) of the National Research Foundation of Korea (NRF). He is also partly supported by the BK21 Program ("Next Generation Education Program for Mathematical Sciences", 4299990414089) funded by the Ministry of Education in Korea and the National Research Foundation of Korea (NRF). SH was supported by the FirstRand FNB 2020 Fund Education Scholarship and the University of KwaZulu-Natal’s 2024 Vincent Maphai Scholarship Award. The research of MC and BR is supported by the Vici grant (number VI.C.232.117) from the Dutch Research Council (NWO). MC is also supported by the AS-IAIA-114 grant from Academia Sinica Taiwan. The work of FR is supported by the NSF grants PHY-2210333, PHY-2609835, and PHY-2019786 (The NSF AI Institute for Artificial Intelligence and Fundamental Interactions), as well as by startup funding from Northeastern University.

## A Data Analysis Tools and Concepts

For the document to be self-contained, in the appendix we introduce and review various data analysis tools and concepts relevant for the experiments in the paper.

## A.1 Model Performance Metrics

In this appendix, we briefly review the methods used to assess network performance across each of the classification and regression tasks.

## Balanced Accuracy Score (BAS)

To evaluate the performance of the trained network on the classification tasks, we utilize the Balanced Accuracy Score (BAS). We selected this metric to account for the significant population imbalances between labels in the dataset (see Tab. 1). Instead of truncating the dataset to enforce parity, we compute the average recall across all classes. For $n _ { c }$ classes, the BAS is defined as:

$$
{ \mathrm { r e c a l l } } = { \frac { T P } { T P + F N } } , \mathrm { B A S } = { \frac { 1 } { n _ { c } } } \sum _ { i = 1 } ^ { n _ { c } } { \mathrm { r e c a l l } } _ { i }\tag{24}
$$

where $T P _ { i }$ is the number of true positives and $F N _ { i }$ is the number of false negatives for the $i ^ { \mathrm { t h } }$ class.

## Pearson Correlation Coefficient (PCC)

The first metric we use to evaluate network performance on the regression task is the Pearson Correlation Coefficient (PCC). The PCC measures the strength and direction of the linear relationship between the ground truth and the predicted values. It is normalized such that PCC $\in [ - 1 , 1 ]$ , where $\mathsf { P C C } = + 1 \ ( - 1 )$ indicates perfect correlation (anti-correlation). Consider the set of n ground truth samples $\{ y _ { i } \}$ and predicted values $\left\{ \hat { y } _ { i } \right\}$ . The PCC is defined as:

$$
\mathrm { P C C } = \frac { \sum _ { i = 1 } ^ { n } ( y _ { i } - \bar { y } ) ( \hat { y } _ { i } - \bar { \hat { y } } ) } { \sqrt { \sum _ { i = 1 } ^ { n } ( y _ { i } - \bar { y } ) ^ { 2 } } \sqrt { \sum _ { i = 1 } ^ { n } ( \hat { y } _ { i } - \bar { \hat { y } } ) ^ { 2 } } }\tag{25}
$$

where $\bar { y }$ and $\bar { \hat { y } }$ denote the sample means of the ground truth and predictions, respectively. In our use case, $y$ corresponds to the logits of the classifier network and $\hat { y }$ to the outputs of the regression network. Thus, the more accurate the regression network, the closer the resulting PCC score will be to 1.

## Coefficient of Determination $( R ^ { 2 }$ Score)

The second metric used to assess the performance of the regression network is the coefficient of determination, or $R ^ { 2 }$ score. Complementary to the $\mathrm { P C C } ,$ the $R ^ { 2 }$ score evaluates the goodness of fit by quantifying the proportion of variance in the observed data explained by the model. For a set of n ground truth samples $\left\{ y _ { i } \right\}$ with mean $\bar { y }$ and n predicted values $\{ \hat { y } _ { i } \}$ , the score is defined as:

$$
R ^ { 2 } = 1 - \frac { \sum _ { i = 1 } ^ { n } ( y _ { i } - \hat { y } _ { i } ) ^ { 2 } } { \sum _ { i = 1 } ^ { n } ( y _ { i } - \bar { y } ) ^ { 2 } } .\tag{26}
$$

Values of $R ^ { 2 } < 0$ indicate that the model performs worse than a constant baseline predicting the sample mean. An $R ^ { 2 } = 0$ implies the model explains none of the variance around the mean. As

$R ^ { 2 }$ approaches 1, the approximation improves; $R ^ { 2 } = 1$ indicates that the predictions perfectly match the ground truth.

## Spearman Rank Correlation $( \rho )$

We employ a regressor, such as an RF regressor, to characterize the non-linear decision boundaries learned by the SVM, in the latent space learned using a multi-class contrastive learning network. In this context, the standard $R ^ { 2 }$ score is insufficient because it primarily measures how well the model captures variance and linear trends. However, the decision boundaries in the latent space are often complex and non-linear. As a remedy, we use a complementary metric–the Spearman rank correlation $( \rho )$ –that assesses the topological quality of the fit rather than just the magnitude of the error.

Unlike the standard PCC score, which assesses linearity, Spearman rank correlation $\rho$ evaluates the monotonic relationship between the ground truth and the predictions. A relationship is monotonic if an increase in one variable consistently corresponds to an increase (or decrease) in the other, even if that rate of change is not constant.

Let $y$ and $\hat { y }$ denote the vectors of ground truth and predicted values, respectively. We convert these values into rank vectors $\mathcal { R }$ and $\hat { \mathcal { R } }$ , where $\mathcal { R } _ { i }$ represents the rank of the i-th sample within the set. The Spearman rank correlation is defined as the PCC of these rank vectors:

$$
\rho = \frac { \mathrm { c o v } ( \mathcal { R } , \hat { \mathcal { R } } ) } { \sigma _ { \mathcal { R } } \sigma _ { \hat { \mathcal { R } } } }\tag{27}
$$

where $\sigma _ { \mathcal { R } }$ and $\sigma _ { \hat { \mathcal { R } } }$ represent the standard deviations of the rank variables. Note that, for a sample size $n ,$ in the absence of tied ranks we have

$$
\begin{array} { r } { \sigma _ { \mathcal { R } } = \sigma _ { \hat { \mathcal { R } } } = \sqrt { \frac { n ^ { 2 } - 1 } { 1 2 } } , } \end{array}
$$

allowing the formula to simplify to:

$$
\rho = 1 - \frac { 6 \sum _ { i = 1 } ^ { n } ( \mathcal { R } _ { i } - \hat { \mathcal { R } } _ { i } ) ^ { 2 } } { n ( n ^ { 2 } - 1 ) } .\tag{28}
$$

The coefficient ranges over $\rho \in [ - 1 , 1 ]$ . A value of +1 indicates a strictly increasing monotonic relationship (perfect ordering), while −1 indicates a strictly decreasing one.

The utility of this metric is that it verifies whether the predicted boundary preserves the ordinal structure of the true decision function. That is, $\rho = 1$ implies that if the ground truth implies $y _ { i } > y _ { j }$ , the model correctly predicts $\hat { y } _ { i } > \hat { y } _ { j }$ , regardless of the exact magnitude of the values. This makes $\rho$ more robust to outliers and non-linearities than the $R ^ { 2 }$ score. However, $\rho$ should not be used in isolation; a high $\rho$ score guarantees correct ranking, but the model might still suffer from significant scaling errors (poor $R ^ { 2 }$ score), as $\rho$ is insensitive to even large magnitude errors.

## Sign Accuracy (ϑ)

While the PCC and $R ^ { 2 }$ metrics evaluate the continuous regression performance, we also require a metric to assess the implied classification performance. In the context of SVM decision boundaries, the hyperplane is defined at $y = 0$ . Thus, we employ the Sign Accuracy (ϑ), which effectively treats the regression output as a binary classifier. It is defined as the fraction of samples where the predicted decision value falls on the same side of the decision boundary as the ground truth:

$$
\vartheta = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \mathbb { I } \left[ \mathsf { s g n } ( y _ { i } ) = \mathsf { s g n } ( \hat { y } _ { i } ) \right]\tag{29}
$$

where $y$ and $\hat { y }$ denote the true and predicted decision values (logits), and $\mathbb { I } [ \cdot ]$ is the indicator function. This is equivalent to the standard classification accuracy metric:

$$
\vartheta = { \frac { T P + T N } { T P + T N + F P + F N } }\tag{30}
$$

where T P and T N represent the true positives and true negatives, respectively.

This metric is crucial for determining classification consistency near the boundary. Notably, because $\vartheta$ relies solely on the sign, it is magnitude agnostic. Unlike $R ^ { 2 }$ , it does not penalize the model for predicting a value close to the boundary when the truth is far, provided both are on the correct side.

## Complement Normalized MAE Score $( S _ { \mathrm { M A E } } )$

The final individual metric used to evaluate the regression fit is derived from the Mean Absolute Error (MAE). To ensure consistency with PCC and $R ^ { 2 }$ (where higher values indicate better performance), we define a complement score $S _ { \mathrm { M A E } }$ . Given the ground truth and the predicted vector y and $\hat { y }$ , the score is defined as:

$$
S _ { \mathrm { M A E } } = 1 - \frac { 1 } { n ( \operatorname* { m a x } ( y ) - \operatorname* { m i n } ( y ) ) } \sum _ { i = 1 } ^ { n } \lvert y _ { i } - \hat { y } _ { i } \rvert .\tag{31}
$$

For typical models, this yields a score $S _ { \mathrm { M A E } } \in [ 0 , 1 ]$ . The primary advantage of this metric over squared-error metrics (like $R ^ { 2 } )$ is that it imposes a linear penalty on errors, making it less sensitive to moderate outliers. Furthermore, normalizing by the range renders the metric scale-invariant, allowing us to incorporate it into the composite score described below.

## Composite Score $( Q _ { \mathrm { c o m p o s i t e } } )$

We now assess the performance of the various regressors trained to approximate the decision boundaries around the clusters labeled by the four classes (STAR-Sph, STAR-Non, H-Sph, H-Non) that the multi-class SVM classifier learned in the latent space. That is, we focus on a single cluster, extract the decision values that define the boundary region from the SVM classifier, and train a regressor network from the input space of q-series and plumbing matrix features to the decision values. To assess the performance of the regressor network, we construct a composite score using the four metrics described above. A conservative approach would be to utilize the minimum score, $Q _ { \operatorname* { m i n } } = \operatorname* { m i n } \{ R ^ { 2 } , \rho , \vartheta , S _ { \mathrm { M A E } } \}$ . While this ensures no single metric is poor, it does not leverage the specific complementary strengths of each measure. Instead, we define a weighted composite score to emphasize the most descriptive metrics for this topology:

$$
Q _ { \mathrm { c o m p o s i t e } } = 0 . 3 R ^ { 2 } + 0 . 2 5 \rho + 0 . 3 \vartheta + 0 . 1 5 S _ { \mathrm { M A E } } .\tag{32}
$$

We justify this specific weighting by examining the complementary nature of the chosen metrics:

$R ^ { 2 }$ vs. $\rho \colon$ These metrics capture distinct aspects of the fit. A perfect rank correlation $( \rho = 1 )$ can be achieved even with a poor $R ^ { 2 }$ score if the predictions preserve the ordinal structure but suffer from systematic scaling errors. $\rho$ validates the topology, while $R ^ { 2 }$ validates the magnitude.

$R ^ { 2 }$ vs. ϑ: These metrics emphasize different performance regimes. The $R ^ { 2 }$ score is dominated by outliers, measuring global regression fidelity. In contrast, $\vartheta$ is magnitudeinvariant and focuses solely on the consistency of the classification.

$R ^ { 2 } \ { \bf v s } . S _ { \mathrm { M A E } } \colon$ Both metrics measure absolute errors but apply different penalties. $R ^ { 2 }$ scales quadratically with error, making it highly sensitive to outliers. $S _ { \mathrm { M A E } }$ applies a linear penalty, providing a measure of the "typical" error that is less skewed by extreme outliers.

$\vartheta$ vs. $S _ { \mathrm { { M A E } } } { \mathrm { : } }$ These metrics are largely independent. It is possible to achieve $\vartheta = 1$ (perfect classification) while having a poor $S _ { \mathrm { M A E } }$ if the model places points on the correct side of the boundary but with incorrect magnitudes (e.g., predicting a value close to 0 when the truth is large).

Tab. 13 compares reconstruction models for the one-vs-rest SVM decision functions in the four-class contrastive latent space. Random Forest and Gradient Boosting are consistently the strongest methods, with nearly identical composite scores on all four boundaries. Random Forest is nevertheless the top model for each individual boundary, achieving $0 . 9 6 6 \pm 0 . 0 2 0$ on STAR-Sph, $0 . 9 7 7 \pm 0 . 0 0 9$ on STAR-Non, $0 . 9 7 0 \pm 0 . 0 1 5$ on H-Sph, and $0 . 9 5 2 \pm 0 . 0 3 5$ on H-Non, with the best overall average score $0 . 9 6 6 \pm 0 . 0 1 5$ . Gradient Boosting is only marginally lower, with average score $0 . 9 6 4 \pm 0 . 0 1 5$ , while the linear baselines are substantially weaker and the neural network reconstruction, although competitive, does not match the tree ensembles. The per-boundary reconstructions and feature-importance summaries of Tab. 10 use the crossvalidation-selected model for each boundary and seed; given the near-degeneracy of the two tree ensembles, this selection alternates between Random Forest and Gradient Boosting across boundaries, consistent with the out-of-fold comparison here.
<table><tr><td>Model</td><td>STAR-Sph</td><td>STAR-Non</td><td>H-Sph</td><td>H-Non</td><td>Average</td></tr><tr><td>Random Forest</td><td> $\mathbf { 0 . 9 6 6 \pm 0 . 0 2 0 }$ </td><td> $\mathbf { 0 . 9 7 7 \pm 0 . 0 0 9 }$ </td><td> $\mathbf { 0 . 9 7 0 \pm 0 . 0 1 5 }$ </td><td> $\mathbf { 0 . 9 5 2 \pm 0 . 0 3 5 }$ </td><td> $\mathbf { 0 . 9 6 6 \pm 0 . 0 1 5 }$ </td></tr><tr><td>Gradient Boost</td><td> $\overline { { 0 . 9 6 4 \pm 0 . 0 2 1 } }$ </td><td> $\overline { { 0 . 9 7 5 \pm 0 . 0 1 0 } }$ </td><td> $\overline { { 0 . 9 6 8 \pm 0 . 0 1 6 } }$ </td><td> $\overline { { 0 . 9 4 9 \pm 0 . 0 3 6 } }$ </td><td> $\overline { { 0 . 9 6 4 \pm 0 . 0 1 5 } }$ </td></tr><tr><td>Ridge Regression</td><td> $\overline { { 0 . 5 2 1 \pm 0 . 0 5 5 } }$ </td><td> $\overline { { 0 . 7 8 9 \pm 0 . 0 1 7 } }$ </td><td> $\overline { { 0 . 6 6 3 \pm 0 . 0 7 4 } }$ </td><td> $\overline { { 0 . 5 8 8 \pm 0 . 0 9 9 } }$ </td><td> $\overline { { 0 . 6 4 0 \pm 0 . 0 3 2 } }$  一</td></tr><tr><td>Linear Regression</td><td> $\overline { { 0 . 5 2 1 \pm 0 . 0 5 5 } }$ </td><td> $\overline { { 0 . 7 8 9 \pm 0 . 0 1 7 } }$ </td><td> $\overline { { 0 . 6 6 3 \pm 0 . 0 7 4 } }$ </td><td> $\overline { { 0 . 5 8 8 \pm 0 . 0 9 9 } }$ </td><td> $\overline { { 0 . 6 4 0 \pm 0 . 0 3 2 } }$ </td></tr><tr><td>Neural Network</td><td> $\overline { { 0 . 9 5 1 \pm 0 . 0 2 6 } }$ </td><td> $\overline { { 0 . 9 6 5 \pm 0 . 0 1 1 } }$ </td><td> $\overline { { 0 . 9 4 3 \pm 0 . 0 1 9 } }$ </td><td> $\overline { { 0 . 9 0 8 \pm 0 . 0 3 8 } }$ </td><td> $\overline { { 0 . 9 4 2 \pm 0 . 0 1 6 } }$ </td></tr></table>

Table 13: Multi-seed composite model-selection scores for reconstructing the one-vs-rest SVM decision functions in the four-class contrastive latent space. Each entry reports the mean ± standard deviation of $Q _ { \mathrm { c o m p o s i t e } }$ over seeds, computed from out-of-fold predictions of the corresponding reconstruction model. The composite score combines calibrated reconstruction $R ^ { 2 }$ Spearman rank correlation $\rho ,$ sign accuracy $\vartheta ,$ and the range-normalized MAE-complement score $S _ { \mathrm { M A E } }$ . Larger values indicate better reconstruction. Bold entries mark the best model for each boundary and the best average across boundaries.

## A.2 Model Explainability Techniques

This appendix outlines the interpretability methods used to determine feature importance. We categorize these tools into two distinct classes based on the nature of the information they extract from the trained model:

1. Local Sensitivity (Feature Saliency): These gradient-based methods measure the rate of change of the network output with respect to the input. They answer the question: “Which feature change causes the steepest immediate change in the output?” High sensitivity implies the network is locally responsive to a feature, but does not necessarily imply the feature is critical for global accuracy.

2. Global Relevance (PI and SHAP): These perturbation-based and game-theoretic methods measure the contribution of a feature to the model’s predictive performance. They answer the question: “How much does the model rely on this feature to correctly classify samples?”

Distinguishing between these two perspectives is crucial for analyzing the results in §4, as features with high local sensitivity often differ from those with high global relevance.

Feature Saliency (FS) To interpret the regression model, we employ a gradient-based saliency method inspired by Class Activation Mapping (CAM) techniques [35]. While originally designed for Convolutional Neural Networks (CNNs) to identify spatial regions of interest, the underlying principle – weighting feature activations by their gradient contribution to the output – can be adapted to the hidden layers of an MLP.

We define the “importance weight” of a hidden unit $h _ { j }$ for a given class logit c (or regression target) as the gradient of the output $y ^ { c }$ with respect to that unit:

$$
\alpha _ { j } ^ { c } = \frac { \partial y ^ { c } } { \partial h _ { j } } .\tag{33}
$$

This weight quantifies how strongly the j-th hidden unit influences the final prediction. Analogous to the Grad-CAM localization map, we compute a weighted activation map for the hidden layer:

$$
L _ { \mathrm { M L P } } ^ { c } = \mathrm { R e L U } \left( \sum _ { j } \alpha _ { j } ^ { c } h _ { j } \right) .\tag{34}
$$

The ReLU function ensures we only consider features that have a positive influence on the class of interest. To project this importance back to the input space (feature attribution), we estimate the contribution of the i-th input feature $x _ { i }$ as:

$$
S _ { i } ^ { c } = \sum _ { j } \alpha _ { j } ^ { c } h _ { j } W _ { j i } \ ,\tag{35}
$$

where $W _ { j i }$ are the weights connecting input i to hidden unit j. This formulation effectively decomposes the network’s decision, highlighting which input variables maximize the activation of the most critical hidden neurons.

## Permutation Importance (PI)

To complement the gradient-based internal analysis, we employ Permutation Importance (PI), a model-agnostic statistical method that quantifies feature contribution based on predictive performance. The fundamental premise is that if a feature is important, corrupting its information content should significantly degrade the model’s score. Conversely, if a feature is irrelevant, the model’s performance should remain unchanged.

To implement this, we take the dataset and randomly shuffle (permute) the values of a single feature i while keeping all other feature columns fixed. This breaks the relationship between feature i and the target while preserving the marginal distribution of the features. For a trained model f and dataset D, the importance of feature i, denoted $I _ { i } ,$ is defined as the expected drop in the scoring metric S:

$$
I _ { i } = S \big ( f , \mathcal { D } \big ) - \mathbb { E } _ { \pi } \big [ S \big ( f , \mathcal { D } ^ { \pi ( i ) } \big ) \big ] ,\tag{36}
$$

where S is a chosen utility metric (where higher values indicate better performance, such as $R ^ { 2 }$ or accuracy) and $\mathcal { D } ^ { \pi ( i ) }$ is the dataset with the i-th feature randomly permuted. The expectation value $\mathbb { E } _ { \pi }$ is estimated by repeating the shuffling process numerous times to ensure statistical robustness. For the regression tasks in this analysis, we utilize the $R ^ { 2 }$ score as the metric S.

This approach provides an external, performance-based measure of reliance on features, independent of the model’s internal weights or gradients. Therefore, it serves as a valuable crosscheck to the gradient-based saliency maps described above. However, a crucial caveat is that PI can underestimate the importance of correlated features; if two features are highly collinear, permuting one may not severely impact performance because the model can extract similar information from the other. We implement this analysis using the built-in function available in sklearn.inspection.

## Shapley Additive Explanations (SHAP)

We employ SHAP (SHapley Additive exPlanations) [36] to interpret the model’s predictions. This framework adapts Shapley values [37] from cooperative game theory, where they are used to fairly distribute a “payout” among a set of “players” based on their marginal contributions. In the context of machine learning, the game is the prediction task, the players are the input features, and the payout is the model’s output relative to the baseline expectation. This approach unifies several explanation methods (e.g., LIME [38], DeepLIFT [39]) under the class of additive feature attribution methods. We implement this analysis using the standard shap Python package.

Formally, let F be the set of all features with cardinality $| F | = M$ . For a given datapoint x, the Shapley value $\phi _ { i }$ for a specific feature $i \in F$ is calculated as the weighted average of its marginal contribution across all possible feature coalitions $S \subseteq F \setminus \{ i \}$

$$
\phi _ { i } ( x ) = \sum _ { S \subseteq F \backslash \{ i \} } { \frac { | S | ! ( M - | S | - 1 ) ! } { M ! } } \left[ f _ { S \cup \{ i \} } ( x _ { S \cup \{ i \} } ) - f _ { S } ( x _ { S } ) \right] ,\tag{37}
$$

where $f _ { S } ( x _ { S } )$ represents the model’s prediction given $x _ { S } .$ , the subset of features S, marginalized over the absent features with the whole dataset. A key property of SHAP is local accuracy, or additivity, meaning the sum of the feature attributions equals the deviation of the prediction from the dataset average:

$$
f ( { \pmb x } ) = \overline { { f } } + \sum _ { i = 1 } ^ { M } \phi _ { i } ( { \pmb x } ) .\tag{38}
$$

Normalized Feature Importance For visualization and comparison across attribution methods, we convert raw feature scores into normalized feature-importance profiles. Given an attribution vector ${ \cal S } = ( S _ { 1 } , \ldots , S _ { M } )$ , we define

$$
\widehat { S } _ { i } = \frac { S _ { i } } { \sum _ { j = 1 } ^ { M } | S _ { j } | + \varepsilon _ { 0 } } ,\tag{39}
$$

where $ { \varepsilon } _ { 0 } > 0$ is a small numerical stabilizer. Thus the denominator is the total absolute attribution mass. This normalization removes arbitrary scale differences between seeds, models, and attribution methods while preserving the sign of methods that can produce signed effects, such as permutation importance. In figures such as Fig. 15, we plot these normalized scores over the ordered feature coordinates; for coefficient–exponent representations, the horizontal axis records the retained coefficient positions followed by the corresponding exponent positions. Peaks in absolute normalized score indicate coordinates on which the trained model concentrates a disproportionate fraction of its attribution mass.

Other Gradient-Based and Attention-Based Diagnostics In addition to the feature-saliency score above, we compute three input-level diagnostics for the $S L _ { 2 } ( \mathbb { Z } )$ -orbit regression experiment: gradient-times-input, integrated gradients, and attention rollout. These diagnostics are applied to the scalar regression output $f ( x )$ , where the input is the flat coefficient–exponent block

$$
\boldsymbol { x } = [ c _ { 0 } , \ldots , c _ { L _ { \mathrm { e f f } } - 1 } , e _ { 0 } , \ldots , e _ { L _ { \mathrm { e f f } } - 1 } ] .
$$

Thus each attribution coordinate corresponds either to a retained coefficient position or to the corresponding retained exponent position.

The gradient-times-input score is defined by

$$
S _ { i } ^ { \mathrm { g x } } ( x ) = \left| x _ { i } { \frac { \partial f ( x ) } { \partial x _ { i } } } \right| .
$$

Unlike raw gradient saliency, this quantity weights local sensitivity by the actual input value. It therefore emphasizes coordinates that are both locally influential and active in the represented sample.

We also compute integrated gradients. Given a baseline input $x ^ { \prime }$ , taken to be the zero input in standardized coordinates, the integrated-gradient attribution is

$$
S _ { i } ^ { \mathrm { I G } } ( x ) = \left| ( x _ { i } - x _ { i } ^ { \prime } ) \int _ { 0 } ^ { 1 } { \frac { \partial f { \big ( } x ^ { \prime } + \alpha ( x - x ^ { \prime } ) { \big ) } } { \partial x _ { i } } } d \alpha \right| .
$$

Numerically, this integral is approximated by a finite Riemann sum along the straight-line interpolation from $x ^ { \prime }$ to x. Integrated gradients probe sensitivity accumulated along this path rather than only at the endpoint $x ,$ and hence provide a complementary diagnostic to both raw feature saliency and gradient-times-input.

For transformer regressors, we further compute attention rollout. Let $A ^ { ( \ell , h ) }$ denote the attention matrix for head h in layer ℓ. We first average over heads,

$$
\hat { A } ^ { ( \ell ) } = \frac { 1 } { H } \sum _ { h = 1 } ^ { H } A ^ { ( \ell , h ) } ,
$$

and include the residual connection by forming

$$
\tilde { A } ^ { ( \ell ) } = \frac { \bar { A } ^ { ( \ell ) } + I } { \mathrm { r o w } { - } \mathrm { s u m } ( \bar { A } ^ { ( \ell ) } + I ) } .
$$

The rollout matrix is then the layerwise product

$$
R = \tilde { A } ^ { ( L ) } \tilde { A } ^ { ( L - 1 ) } \cdots \tilde { A } ^ { ( 1 ) } .
$$

We use the resulting token-level mass as an attention-based attribution profile over the coefficient and exponent positions. Unlike the gradient-based diagnostics, attention rollout does not measure sensitivity of the output with respect to infinitesimal input perturbations; rather, it records how information is routed through the self-attention layers. It is therefore used as a complementary structural diagnostic rather than as a substitute for gradient attribution.

For the multi-seed robustness analysis, all attribution profiles are computed separately for each trained seed. Before averaging over seeds, each profile is normalized by its total attribution mass on the represented feature block. This prevents a small number of high-magnitude seeds from dominating the ensemble attribution profile. We interpret the resulting averaged diagnostics as evidence for which coefficient and exponent positions are used or routed through by the trained regression model, not as a closed-form symbolic rule for the $S L _ { 2 } ( \mathbb { Z } )$ -orbit size.

## A.3 Contrastive Learning

We use a contrastive learning algorithm based on semi-hard triplet loss (SHTL) to perform clustering-based data compression. The rationale for this choice is the ability of the network to learn meaningful, clustered embeddings in a low-dimensional space even in the presence of limited or imbalanced data, which is often the case in our analysis. Compared to other contrastive losses, SHTL offers a good trade-off between computational efficiency and representational robustness.

Contrastive learning originated in computer vision as a method for image verification and facial recognition. Given a labeled dataset P with class labels $c _ { i } ,$ the goal is to learn a mapping $f : P $ M from the input space to a latent space M (typically $M \subset \mathbb { R } ^ { d } )$ such that similar samples are embedded close together and dissimilar samples are far apart. A commonly used contrastive loss for binary-labeled pairs $( x _ { 1 , i } , x _ { 2 , i } )$ with a pair relationship label $y ^ { i } \in \{ 0 , 1 \}$ is given by [40]:

$$
\mathcal { L } _ { \mathrm { c o n } } = \frac { 1 } { 2 } \sum _ { \mathrm { p a i r s } i } \left( ( 1 - y ^ { i } ) \| f ( x _ { 1 , i } ) - f ( x _ { 2 , i } ) \| ^ { 2 } + y ^ { i } \operatorname* { m a x } \left[ 0 , \kappa - \| f ( x _ { 1 , i } ) - f ( x _ { 2 , i } ) \| \right] ^ { 2 } \right)
$$

Here, κ defines a margin in the latent space, and $\| \cdot \|$ denotes the $L _ { 2 } { \mathrm { - } } \mathrm { n o r m }$ . We adopt the convention where $y = 0$ denotes a similar pair (same class) and $y = 1$ denotes a dissimilar pair. Consequently, pairs with $y = 0$ contribute to the first term and are pulled together. Negative pairs $( y = 1 )$ contribute to the second term only if they fall within the margin $\kappa ,$ effectively pushing them apart.

Triplet loss generalizes this approach by evaluating triplets $( A , P , N )$ consisting of an anchor $A ,$ a positive P (same class), and a negative N (different class) example. The loss encourages the distance between the anchor and positive to be smaller than the distance between the anchor and negative by at least a margin κ:

$$
\begin{array} { c } { \displaystyle { \mathcal { L } _ { \mathrm { t r i p } } = \sum _ { i } \mathrm { m a x } \left[ 0 , D ( A _ { i } , P _ { i } , N _ { i } ) \right] } } \\ { \displaystyle { D ( A _ { i } , P _ { i } , N _ { i } ) = | | f ( A _ { i } ) - f ( P _ { i } ) | | ^ { 2 } - | | f ( A _ { i } ) - f ( N _ { i } ) | | ^ { 2 } + \kappa } } \end{array}
$$

Training with all possible triplets is computationally expensive and often inefficient, as many triplets satisfy the margin condition $( \mathrm { i . e . , }$ they are already correctly clustered) and contribute zero gradient. Conversely, “hard” negatives (where N is very close to A) can destabilize training in the early stages.

To address these issues, semi-hard triplet loss (SHTL) restricts training to semi-hard triplets– those where the negative is farther from the anchor than the positive, but still within the margin:

$$
\| f ( A _ { i } ) - f ( P _ { i } ) \| ^ { 2 } < \| f ( A _ { i } ) - f ( N _ { i } ) \| ^ { 2 } < \| f ( A _ { i } ) - f ( P _ { i } ) \| ^ { 2 } + \kappa
$$

This constraint defines a shell in the latent space: the inner radius is determined by the distance to the positive, and the shell thickness is fixed by κ. Negatives within this shell are considered semi-hard and contribute to the loss. Negatives within the inner radius are labeled hard, while those outside the shell are easy. By focusing training on semi-hard cases, SHTL improves convergence speed and embedding quality while avoiding the instability associated with hard negatives.

While more complex formulations such as N-tuple contrastive losses exist [41], we find that given the relatively small number of classes in our experimental datasets, SHTL strikes an effective balance between simplicity and performance. It encourages well-separated clusters in the latent space without requiring extensive sampling or computational overhead.

## B Interpretability Robustness

In this appendix, we detail the protocol of the multi-seed robustness runs that underlie the binary full- sequence and value-index classification experiments of Sec. 4. The single-seed results reported for those experiments in the main body are taken from this robustness experiment. The goal is to establish whether the results are robust to the choice of random seed, with all other pipeline stages held fixed.

## B.1 Multi-seed robustness protocol

This subsection describes the multi-seed robustness experiment underlying the Sec. 4 classifier, logit-regression, and interpretability results. The purpose of this experiment is to separate stable structural effects from artifacts of a particular random initialization, train/validation split, or attribution pass. All seed-averaged Sec. 4 quantities are computed from this experiment, and the normalized Margin Reconstruction analysis in the next subsection uses the same source teachers and stored splits.

Seed bank and stages. All stages draw from the fixed seed bank

$$
\mathcal { S } = \{ 4 2 , 1 2 3 , 4 5 6 , 7 8 9 , 1 3 3 7 \} .
$$

For each task and representation, we train one classifier for each seed in S. For each trained classifier, we then train logit regressors over the same seed bank, and interpretability passes are likewise repeated over the same seed set. Thus the robustness experiment is organized into three nested components: source classifiers, abstract-feature logit regressors, and attribution diagnostics.

Tasks. The six binary tasks are

$$
\mathrm { H O M - S t a r , } \quad \mathrm { H O M - H , } \quad \mathrm { H O M - A l l , } \quad \mathrm { T O P - S p h , } \quad \mathrm { T O P - N o n , } \quad \mathrm { T O P - A l l . }
$$

The HOM tasks distinguish homology-sphere from non-homology-sphere examples on different graph sectors, while the TOP tasks distinguish H-graph from STAR examples on different topological sectors. Each task is evaluated as a binary classification problem with balanced accuracy as the primary classifier metric.

Representations. For each task we evaluate three input representations. The full-sequence or sparse representation embeds each normalized $\widehat { Z }$ coefficient sequence into a fixed 10,000- dimensional sparse vector indexed by exponent. The dense representation serializes each sequence as a value-index pair vector, with coefficient entries followed by exponent entries, and applies an asinh scaling. The contrastive-learning representation starts from the full-sequence input and passes it through a learned encoder trained with a semi-hard triplet objective. The main full-sequence results in Sec. 4 use the sparse representation; the dense and contrastivelearning representations are reported as robustness comparisons.

Abstract feature dictionary. The logit-regression and interpretability stages use the ten-dimensional abstract feature vector

$$
\Phi ( x ) = \bigl ( \operatorname* { d e t } , \varepsilon _ { \mathrm { m i n } } , \varepsilon _ { \mathrm { m a x } } , \mu _ { \varepsilon } , \sigma _ { \varepsilon } , \Delta , \gamma _ { 0 } , \mu _ { \gamma } , \sigma _ { \gamma } , c _ { \mathrm { m a x } } \bigr ) .
$$

Here $( \varepsilon _ { \operatorname* { m i n } } , \varepsilon _ { \operatorname* { m a x } } , \mu _ { \varepsilon } , \sigma _ { \varepsilon } )$ are eigenvalue summary statistics, while $( \gamma _ { 0 } , \mu _ { \gamma } , \sigma _ { \gamma } )$ are gap summary statistics. The abstract features are standardized before being used by the regression and interpretability models.

Classifier training. For each task, representation, and classifier seed $s \in S ,$ , the data are split into train and held-out validation sets using the stored split associated to that seed. A shallow ReLU classifier is then trained with cross-entropy loss, Adam optimization, early stopping on held-out balanced accuracy, hidden width 64, learning rate $1 0 ^ { - 3 }$ , batch size 128, and patience 20. The best validation checkpoint is retained. The classifier records store the trained model, the fitted scaler, the label encoder, the train/validation indices, the validation inputs, and the train/validation labels. The training design matrix is intentionally omitted from the stored classifier records to avoid unnecessary memory growth; when needed, it is reconstructed from the stored split indices and fitted scaler.

The primary classifier metric is validation balanced accuracy. As a reproducibility check, all reported classifier means and standard deviations are recomputed directly from the released models and validation inputs, rather than read from cached summary tables; the recomputed balanced accuracies agree with the recorded checkpoint values, with a largest observed deviation of $2 . 2 \times 1 0 ^ { - 4 }$ (one HOM-Star seed). In particular, the full-sequence HOM-Star result used in the main text is $0 . 9 2 8 \pm 0 . 0 0 3$

Logit-regression stage. After training the source classifiers, the next stage asks how much of the teacher’s two-logit geometry can be reconstructed from the abstract feature dictionary. For each source classifier, the classifier logits are extracted on the train and validation splits. A regressor is then trained from the standardized abstract features to the two-dimensional teacher logits. This is a deliberately stricter test than reproducing the hard class label: it requires the abstract features to reconstruct the continuous pre-softmax geometry of the teacher. Regressor performance is reported by train and validation $R ^ { 2 }$ and Pearson correlation coefficient. The checkpoint selected by validation performance is used for attribution.

Attribution diagnostics. For each trained logit regressor, we compute feature-level attribution diagnostics on the abstract feature dictionary. The reported diagnostics include gradient saliency, grouped permutation importance, and SHAP-style attribution. These methods are not interpreted as causal interventions on the original $\widehat { Z }$ sequence. Rather, they quantify which abstract features are most responsible for reconstructing the trained teacher’s logit geometry within the abstract-feature regression model. Rankings are aggregated over the seed ensemble and reported as mean ranks or seed-averaged attribution summaries.

Reporting and figure generation. The final Sec. 4 tables and figures are generated directly from the trained models and stored data of this multi-seed experiment. Classifier confusion matrices and logit-space plots are recomputed directly from the stored full-sequence classifiers and held-out inputs. The logit-space visualization standardizes logits within each panel for display and applies central quantile filtering so that a small number of extreme logits does not determine the plotting range. These display transformations are used only for visualization and do not affect the reported balanced accuracy values.

Interpretation. The multi-seed experiment distinguishes three levels of robustness. First, classifier robustness asks whether the full-sequence, dense, and contrastive-learning teachers achieve stable validation balanced accuracy across seeds. Second, logit-regression robustness asks whether the abstract feature dictionary reconstructs the continuous teacher logits in a stable way. Third, attribution robustness asks whether the same abstract features remain important across seeds and attribution methods. The normalized Margin Reconstruction experiment in the next subsection complements this pipeline by relaxing the logit-regression target: instead of asking whether abstract features reconstruct calibrated logits, it asks whether they reconstruct the source teacher’s hard decision boundary.

## B.2 Normalized Margin Reconstruction protocol

This appendix describes the normalized Margin Reconstruction (MR) protocol used in Table 7. The purpose of the protocol is to test whether low-dimensional abstract features reproduce the hard decision boundary of the full-sequence classifiers, rather than their calibrated logits. This distinction matters because the logit-regression experiment tests reconstruction of the continuous two-logit geometry, while MR tests reconstruction of the teacher’s induced classification rule.

Source teachers. For each task T ∈ {HOM-Star, HOM-H, HOM-All, TOP-Sph, TOP-Non, TOP-All}, the source teacher is the full-sequence classifier from the multi-seed robustness experiment; no separate MR teacher is trained. For each teacher seed $s \in \{ 4 2 , 1 2 3 , 4 5 6 , 7 8 9 , 1 3 3 7 \}$ , we use that experiment’s stored train/validation split, fitted input scaler, trained full-sequence classifier, and validation labels. The training inputs are reconstructed from the stored split and scaler, and the reconstructed teacher predictions are checked against the recorded validation BAS for consistency.

Student targets. Let $f _ { T , s }$ denote the source full-sequence teacher for task T and seed $s ,$ and let

$$
\ell _ { T , s } ( x ) = ( \ell _ { 0 } ( x ) , \ell _ { 1 } ( x ) )
$$

be its two-logit output on a validation sample x. The MR student does not regress $\ell _ { T , s } ( x )$ Instead, it is trained to reproduce the hard teacher label

$$
\hat { y } _ { T , s } ( x ) = \arg \operatorname* { m a x } _ { a \in \{ 0 , 1 \} } \ell _ { a } ( x ) .
$$

Thus the primary target is balanced agreement with $\hat { y } _ { T , s } ,$ not balanced accuracy against the original ground-truth label. True-label BAS is reported only as a reference.

Abstract feature dictionary. Students receive only subsets of the abstract feature vector

$$
\Phi ( x ) = \bigl ( \operatorname* { d e t } , \varepsilon _ { \mathrm { m i n } } , \varepsilon _ { \mathrm { m a x } } , \mu _ { \varepsilon } , \sigma _ { \varepsilon } , \Delta , \gamma _ { 0 } , \mu _ { \gamma } , \sigma _ { \gamma } , c _ { \mathrm { m a x } } \bigr ) .
$$

Here $( \varepsilon _ { \operatorname* { m i n } } , \varepsilon _ { \operatorname* { m a x } } , \mu _ { \varepsilon } , \sigma _ { \varepsilon } )$ are the eigenvalue statistics, and $( \gamma _ { 0 } , \mu _ { \gamma } , \sigma _ { \gamma } )$ are the gap statistics. The candidate feature subsets include det-only, eigenvalue statistics, gap statistics, $c _ { \mathrm { m a x } } { \mathrm { - o n l y } }$ and several combined sets such as det +γ-statistics and det +ϵ-statistics.

Student model families. For each task, teacher seed, feature subset, and student seed, we train students from three simple model families:

logistic regression, random forest, small ReLU MLP.

Logistic regression and the MLP use standardized abstract features. The random forest is trained on the corresponding raw abstract-feature subset. The MLP has a single hidden layer matching the small classifier family used elsewhere in the robustness pipeline and is selected by an internal validation split within the teacher-training split. The final evaluation is always performed on the held-out split inherited from the source teacher.

Margin slices. MR evaluates not only full held-out teacher agreement but also agreement on margin-defined subsets. For each validation sample x, define the source-teacher margin

$$
m ( x ) = | \ell _ { 1 } ( x ) - \ell _ { 0 } ( x ) | .
$$

The boundary slices $B _ { 5 } , B _ { 1 0 }$ , and $B _ { 2 0 }$ are the 5%, 10%, and 20% of validation samples with smallest $m ( x )$ , respectively. The easy slice $E _ { 5 0 }$ is the 50% of validation samples with largest $m ( x )$ . Balanced teacher agreement on these slices distinguishes true boundary reconstruction from easy-region agreement.

Selection criterion. For each task, a single MR student is selected by a margin-aware score combining full teacher agreement, boundary-slice teacher agreement, easy-region teacher agreement, true-label BAS, and a small feature-count penalty. The score used for the selected rows is

$$
S = 0 . 3 0 A _ { \mathrm { F u l l } } + 0 . 2 0 A _ { B _ { 5 } } + 0 . 2 0 A _ { B _ { 1 0 } } + 0 . 1 5 A _ { B _ { 2 0 } } + 0 . 0 5 A _ { E _ { 5 0 } } + 0 . 1 0 \mathrm { B A } S _ { \mathrm { t r u e } } - 0 . 0 0 2 d ,
$$

where $A _ { \mathrm { F u l l } }$ is balanced agreement with the source teacher on the full held-out split, $A _ { B _ { n } }$ is balanced agreement on the p% smallest-margin slice, $A _ { E _ { 5 0 } }$ is balanced agreement on the 50% largest-margin slice, ${ \mathrm { B A S } } _ { \mathrm { t r u e } }$ is balanced accuracy against the original labels, and d is the number of abstract features used by the student. The selected row for each task is the candidate with largest mean score over the teacher/student seed ensemble.

Feature attribution. For the selected MR student in each task, we compute feature-level attributions on the $B _ { 1 0 }$ slice. Three diagnostics are used: saliency, SHAP attribution, and permutation drop in teacher agreement. For logistic students, saliency is the absolute standardized coefficient; for random forests, saliency is the impurity-based feature importance; and for MLP students, saliency is the mean absolute input gradient of the class-margin score. SHAP values are computed with model-appropriate explainers when available, with the logistic case using the standard linear coefficient-times-input contribution. Permutation importance is computed by permuting one selected abstract feature at a time within the evaluation slice and recording the drop in balanced agreement with the source teacher. The “Dominant $B _ { 1 0 }$ features” column in Table 7 reports the leading one or two features under the average of normalized saliency, SHAP, and permutation-drop scores.

Interpretation. MR is not intended to replace the full-sequence classifier. Instead, it asks whether a low-dimensional abstract feature subset gives a faithful surrogate for the teacher’s hard decision boundary. High true-label BAS alone is insufficient: a student can classify the original labels well while failing to reproduce the teacher near its boundary. The margin-slice profile is therefore essential. In the present experiments, determinant and gap statistics explain most of the HOM-H/HOM-All and topology teacher geometry, respectively, while HOM-Star remains poorly reconstructed by the current abstract feature dictionary.

## C Spectral bound on $| \varepsilon _ { \mathrm { m i n } } |$

In this appendix we give the full derivation of Lemma 1. Let M be the plumbing matrix of a three-legged star-shaped graph associated to a Seifert manifold $M ( b ; a _ { 1 } / b _ { 1 } , a _ { 2 } / b _ { 2 } , a _ { 3 } / b _ { 3 } )$ with singular fiber orders $b _ { 1 } , b _ { 2 } , b _ { 3 }$ . Being an intersection form, M is a nonsingular symmetric integer matrix; we write $\varepsilon _ { \mathrm { { m i n } } }$ for its eigenvalue of smallest absolute value.

Because M is symmetric, its inverse $M ^ { - 1 }$ is also symmetric. The spectral radius of $M ^ { - 1 }$ is $1 / \vert \varepsilon _ { \mathrm { m i n } } \vert .$ , and by the Rayleigh quotient (taking $x = e _ { \nu } )$ , the absolute value of any diagonal entry is bounded by the spectral radius:

$$
{ \frac { 1 } { \vert \varepsilon _ { \mathrm { m i n } } \vert } } \geq \vert ( M ^ { - 1 } ) _ { \nu \nu } \vert \quad { \mathrm { f o r ~ a l l ~ } } \nu \in V .\tag{40}
$$

By Cramer’s rule, the diagonal entries of the inverse are $( M ^ { - 1 } ) _ { \nu \nu } = \operatorname * { d e t } M _ { \nu } /$ det M, where $M _ { \nu }$ is the submatrix obtained by deleting vertex v.

Consider the central node $\nu _ { 0 } .$ . Deleting $\nu _ { 0 }$ disconnects the star into three independent paths, so $M _ { \nu _ { 0 } }$ is block-diagonal and | det $\begin{array} { r } { M _ { \nu _ { 0 } } | = \prod _ { i = 1 } ^ { 3 } | } \end{array}$ det $L _ { i } | _ { \ r { \ r } }$ , where $L _ { i }$ is the tridiagonal intersection matrix of the i-th leg. The absolute determinant of each leg satisfies a recurrence relation

$$
\Delta _ { k } = w _ { k } \Delta _ { k - 1 } - \Delta _ { k - 2 } , \quad \mathrm { w i t h } \ \Delta _ { 0 } = 1 , \ \Delta _ { - 1 } = 0 ,
$$

which coincides with the numerator recurrence for the Hirzebruch–Jung continued fraction $[ w _ { 1 } , . . . , w _ { n } ] = b _ { i } / a _ { i }$ . Hence | det $\begin{array} { r } { L _ { i } | = b _ { i } . } \end{array}$ , giving | det $M _ { \nu _ { 0 } } | = b _ { 1 } b _ { 2 } b _ { 3 }$ . See Appendix A of [12], in particular equations $( \mathsf { A } . 2 4 )$ , for more details.

Combining with Cramer’s rule, $| ( M ^ { - 1 } ) _ { \nu _ { 0 } \nu _ { 0 } } | = b _ { 1 } b _ { 2 } b _ { 3 } / |$ det M|, and the spectral bound (40) becomes:

$$
\vert \varepsilon _ { \mathrm { m i n } } \vert \leq \frac { \vert \operatorname* { d e t } M \vert } { b _ { 1 } b _ { 2 } b _ { 3 } } ,\tag{41}
$$

completing the proof of Lemma 1.

The bound reveals two independent mechanisms that conspire to make $| \varepsilon _ { \mathrm { m i n } } |$ small for Brieskorn spheres. First, the homology sphere condition | det $M | = 1$ fixes the numerator to its minimal possible value, whereas for non-spheres | det M| can be arbitrarily large, directly loosening the bound. Second, the pairwise coprimality of the $b _ { i }$ forces a large denominator: since the smallest pairwise coprime integers $\geq 2$ are $2 , 3 , 5 ,$ one has $b _ { 1 } b _ { 2 } b _ { 3 } \geq 3 0$ , and typically much larger for generic Seifert homology spheres. Together, these two factors guarantee that a Brieskorn sphere must possess a small $| \varepsilon _ { \mathrm { m i n } } |$ , bounded by $1 / 3 0$ from above, for all three-fiber Brieskorn homology spheres.

## D Bound involving minτ

In this section we give a proof that for every pairwise coprime triple $2 \leq b _ { 1 } < b _ { 2 } < b _ { 3 }$

$$
\biggl | \operatorname* { m i n } _ { k \geq 0 } \tau ( k ) + \frac { P } { 8 } \Bigl ( 1 - \sum _ { i = 1 } ^ { 3 } \frac { 1 } { b _ { i } } \Bigr ) ^ { 2 } \biggr | \leq \sum _ { i = 1 } ^ { 3 } b _ { i } .
$$

In what follows we use the notation of Section 5.1.1. Additionally we introduce the notation {x} to denote $x - \lfloor x \rfloor$ for $x \in \mathbb { R }$ . It is clear that $\{ x \} \in [ 0 , 1 )$ . Similarly, note the sawtooth

function satisfies $\begin{array} { r } { | \langle x \rangle | \leq \frac { 1 } { 2 } } \end{array}$ for every $x \in \mathbb { R }$ . Recall the definition of the generalized and classical Dedekind sums defined respectively as

$$
s ( p , q ; x , y ) = \sum _ { l = 0 } ^ { q - 1 } \biggl < \frac { l + y } { q } \biggr > \biggl < \frac { p ( l + y ) } { q } + x \biggr > , \qquad s ( p , q ) : = s ( p , q ; 0 , 0 ) .
$$

We will use the result (cf. [42]) that, for $\operatorname* { g c d } ( p , q ) = 1$ ，

$$
| s ( p , q ) | \leq s ( 1 , q ) = \frac { ( q - 1 ) ( q - 2 ) } { 1 2 q } .\tag{42}
$$

We begin by writing $\begin{array} { r } { S = 1 - \sum _ { i = 1 } ^ { 3 } \frac { 1 } { b _ { i } } } \end{array}$ . Apart from the case $( b _ { 1 } , b _ { 2 } , b _ { 3 } ) = ( 2 , 3 , 5 )$ where $S < 0 _ { ; }$ ， in all other cases one has $0 < S < 1$ . Substituting $\lfloor ( k - 1 ) / b _ { i } \rfloor = ( k - 1 ) / b _ { i } - \{ ( k - 1 ) / b _ { i } \}$ into (18), allows us to write for every integer $k \geq 0$

$$
\tau ( k ) = g ( k ) + R ( k ) , \qquad g ( k ) : = \frac { k ( k - 1 ) } { 2 P } - \frac { k S } { 2 } ,\tag{43}
$$

with

$$
R ( k ) : = \frac { 3 } { 2 } - \frac { 1 } { 2 } \sum _ { i = 1 } ^ { 3 } \Biggl [ \frac { 1 } { b _ { i } } + \Bigl \{ \frac { k - 1 } { b _ { i } } \Bigr \} + \Biggl \langle \frac { k a _ { i } } { b _ { i } } \Biggr \rangle - 2 \Bigl ( s ( \hat { b } _ { i } , b _ { i } ) - s \Bigl ( \hat { b } _ { i } , b _ { i } ; \frac { k } { b _ { i } } , 0 \Bigr ) \Bigr ) \Biggr ] .\tag{44}
$$

Let us now establish a bound for $R ( k )$ . Each generalized Dedekind sum appearing in R has $y = 0$ so its $l = 0$ summand contains the factor $\left. 0 \right. = 0$ and vanishes; the remaining $b _ { i } - 1$ summands are each a product of two sawtooth values, thus for every x we find $| s ( \hat { b } _ { i } , b _ { i } ; x , 0 ) | \leq ( b _ { i } - 1 ) / 4$ Finally, pairwise coprimality gives $\textstyle \sum _ { i = 1 } ^ { 3 } 1 / b _ { i } \leq { \frac { 1 } { 2 } } + { \frac { 1 } { 3 } } + { \frac { 1 } { 5 } } = { \frac { 3 1 } { 3 0 } }$ . Therefore, after noting that $s ( \hat { b } _ { i } , b _ { i } )$ satisfies (42), for every $k \geq 0$ we obtain,

$$
| R ( k ) | \leq \frac { 3 } { 2 } + \frac { 3 1 } { 6 0 } + \frac { 3 } { 2 } + \frac { 3 } { 4 } + \sum _ { i = 1 } ^ { 3 } \frac { ( b _ { i } - 1 ) ( b _ { i } - 2 ) } { 1 2 b _ { i } } + \frac { 1 } { 4 } \sum _ { i = 1 } ^ { 3 } ( b _ { i } - 1 ) \leq \frac { 5 2 9 } { 1 8 0 } + \frac { 1 } { 3 } \sum _ { i = 1 } ^ { 3 } b _ { i } .\tag{45}
$$

Since $g ( k )$ grows quadratically and $| R |$ is bounded we see $\tau ( k ) \to \infty$ as $k \to \infty$ , so min $k { \geq } 0 \tau ( k )$ exists. Define $\begin{array} { r } { B : = { \frac { 5 2 9 } { 1 8 0 } } + { \frac { 1 } { 3 } } \sum _ { i = 1 } ^ { 3 } b _ { i } } \end{array}$ . We then, moreover, see from (43) that $| \tau ( k ) - g ( k ) | \leq B$ for every integer $k \geq 0$

We now turn to analyzing $g .$ . Extended to a real argument, $g$ is a parabola with leading coefficient $\scriptstyle { \frac { 1 } { 2 P } }$ , vertex $\begin{array} { r } { x ^ { \ast } = \frac { 1 + \tilde { P S } } { 2 } } \end{array}$ , and minimal value,

$$
g ( x ^ { * } ) = - { \frac { ( 1 + P S ) ^ { 2 } } { 8 P } } = - \biggl [ { \frac { P S ^ { 2 } } { 8 } } + { \frac { S } { 4 } } + { \frac { 1 } { 8 P } } \biggr ] .
$$

Let $k _ { 0 }$ denote a nearest integer to $x ^ { * }$ , then mi $\begin{array} { r } { \mathfrak { n } _ { k \in \mathbb { Z } _ { > 0 } } g ( k ) = g ( k _ { 0 } ) } \end{array}$ . If PS is odd, then $x ^ { * } \in \mathbb { Z }$ and $k _ { 0 } = x ^ { * }$ . On the other hand if $P S$ is even then $\begin{array} { r } { k _ { 0 } = x ^ { \ast } \pm \frac { 1 } { 2 } } \end{array}$ . Note that $\begin{array} { r } { g ( x ^ { * } \pm \frac { 1 } { 2 } ) = g ( x ^ { * } ) + \frac { 1 } { 8 P } } \end{array}$ In all cases we thus obtain

$$
0 \leq \operatorname* { m i n } _ { k \in \mathbb { Z } _ { \geq 0 } } g ( k ) - g ( x ^ { * } ) \leq \frac { 1 } { 8 P } .\tag{46}
$$

This can be in turn used to show

$$
\left| \operatorname* { m i n } _ { k \in \mathbb { Z } _ { \geq 0 } } g ( k ) + \frac { P S ^ { 2 } } { 8 } \right| \leq \frac { S } { 4 } + \frac { 1 } { 8 } \leq \frac { 3 } { 8 }\tag{47}
$$

where in the last inequality we use the assumption that $0 < S < 1$ . However (47) holds even when $( b _ { 1 } , b _ { 2 } , b _ { 3 } ) = ( 2 , 3 , 5 )$ and thus it holds for all cases of S. Since $\begin{array} { r } { | \operatorname* { m i n } _ { k \geq 0 } \tau ( k ) - \operatorname* { m i n } _ { k \in \mathbb { Z } _ { > 0 } } g ( k ) | \leq B } \end{array}$ a further calculation, after noting that $\textstyle \sum _ { i = 1 } ^ { 3 } b _ { i } \geq 2 + 3 + 5 = 1 0$ , shows

$$
\left| \operatorname* { m i n } _ { k \geq 0 } \tau ( k ) + \frac { P S ^ { 2 } } { 8 } \right| \leq B + \frac { 3 } { 8 } = \frac { 1 1 9 3 } { 3 6 0 } + \frac { 1 } { 3 } \sum _ { i = 1 } ^ { 3 } b _ { i } \leq \sum _ { i = 1 } ^ { 3 } b _ { i } .\tag{48}
$$

This completes the proof. This result also implies

$$
d ( \Sigma ( b _ { 1 } , b _ { 2 } , b _ { 3 } ) ) \leq 2 ( b _ { 1 } + b _ { 2 } + b _ { 3 } ) .\tag{49}
$$

To prove this, first note that the orbifold Euler number of $\Sigma ( b _ { 1 } , b _ { 2 } , b _ { 3 } )$ is given by $e = - 1 / P$ After substituting this into (14) and comparing with (15) one obtains

$$
d ( \Sigma ( b _ { 1 } , b _ { 2 } , b _ { 3 } ) ) = { \frac { 5 } { 4 } } - { \frac { 1 } { 4 P } } - 3 \sum _ { i = 1 } ^ { 3 } s ( a _ { i } , b _ { i } ) - 2 \operatorname* { m i n } _ { k \geq 0 } \tau ( k ) - { \frac { P S ^ { 2 } } { 4 } }\tag{50}
$$

The middle expression in (48) implies $\begin{array} { r } { 2 \big | \operatorname* { m i n } _ { k \geq 0 } \tau ( k ) + \frac { P S ^ { 2 } } { 8 } \big | \ \leq \ \frac { 1 1 9 3 } { 1 8 0 } + \frac { 2 } { 3 } \sum _ { i = 1 } ^ { 3 } b _ { i } } \end{array}$ . An application of (42) together with $\textstyle \sum _ { i = 1 } ^ { 3 } 1 / b _ { i } \leq { \frac { 3 1 } { 3 0 } }$ yields the estimate

$$
3 \sum _ { i = 1 } ^ { 3 } \lvert s ( a _ { i } , b _ { i } ) \rvert \leq \frac { 1 } { 4 } \sum _ { i = 1 } ^ { 3 } b _ { i } - \frac { 9 } { 4 } + \frac { 3 1 } { 6 0 } .
$$

With these facts at hand, applying the triangle inequality to (50) gives

$$
d ( \Sigma ( b _ { 1 } , b _ { 2 } , b _ { 3 } ) ) \leq \frac { 1 1 } { 1 2 } \sum _ { i = 1 } ^ { 3 } b _ { i } + \frac { 5 5 3 } { 9 0 } + \frac { 1 } { 4 P } .
$$

and since $\textstyle \sum _ { i = 1 } ^ { 3 } b _ { i } \geq 1 0$ and $P \geq 3 0$ we see that $\begin{array} { r } { \frac { 5 5 3 } { 9 0 } + \frac { 1 } { 4 P } \leq \frac { 1 3 } { 1 2 } \sum _ { i = 1 } ^ { 3 } b _ { i } } \end{array}$ . From this (49) follows.   
It is possible to obtain tighter estimates of $d ,$ , but we do not pursue this here.

## E Splice diagrams and integral homology Spheres

In this section we give a self-contained overview of splice diagrams following [32]. The main goal will be to outline an algorithm from [32, Section 9] which allows us to produce integral homology spheres from a given splice diagram.

Recall that earlier in the paper we defined a splice diagram as a tree satisfying some conditions. For clarity, it is helpful to instead define a splice diagram to be a digraph (see Figure 8), whose underlying graph structure has finitely many vertices which have deg = 1 or deg <sup>≥</sup> 3. For each of the outgoing edges of a node an integer weight is assigned arbitrarily. On the other hand, for each of the outgoing edges of a leaf, a weight is assigned as per the "Splicing Leaf Repair" procedure (see [32, Theorem 9.1]) outlined below.

Suppose $\nu _ { i }$ is a leaf of a splice diagram $\widetilde { \Gamma }$ and $E = ( \nu _ { i } , \nu _ { j } )$ is an outgoing edge connecting $\nu _ { i }$ to the node $\nu _ { j }$ . To assign the weight $w _ { \nu _ { i } \nu _ { j } }$ on $E \colon$ we first let $b = w _ { \nu _ { i } \nu _ { i } }$ be the weight on the edge $( \nu _ { j } , \nu _ { i } )$ . Let a be the product of the weights on edges $( \nu _ { j } , \nu _ { k } )$ where $\nu _ { k } \neq \nu _ { i }$ is a vertex adjacent to the node $\nu _ { j }$ , then define

$$
w _ { \nu _ { i } \nu _ { j } } : = { \bigg \lceil } { \frac { a } { b } } { \bigg \rceil } .\tag{51}
$$

We call this procedure "Splicing Leaf Repair" as it assigns a weight to each of the outgoing edges from a leaf. It is summarized in Algorithm 1.

In what follows we will always assume that a splice diagram $\widetilde { \Gamma }$ has the Splicing Leaf Repair procedure applied to it and satisfies the edge-determinant and pairwise-coprimality hypotheses of the splice-diagram definition; the constructions below are stated under, and must preserve, these hypotheses.

![](images/65f69637ce3a143021b0c1da77d17307d6e10cac3c7560d3159f30590883ae94.jpg)

![](images/0ec519b02b8671c5ab736dbcbf5ef45c46b7b5c36cbbad94299dd68329f19e68.jpg)  
Figure 8: A splice diagram represented by a digraph. The weights on outgoing edges from nodes are depicted as $w _ { \nu _ { i } \nu _ { j } }$ and the weights on outgoing edges from leaves are omitted.

## E.1 An algorithm to produce integral homology Spheres

We now turn to the main goal of this section, which is to outline the algorithm to produce negative-definite plumbed manifolds which are integral homology spheres from splice diagrams.

At a high level the algorithm is as follows: one starts off with a splice diagram $\widetilde { \Gamma }$ and then constructs what is known as a maximal splice diagram $\widetilde { \Gamma } _ { \mathrm { m a x } }$ from it. The maximal splice diagram has the same underlying graph structure as a plumbing graph Γ . From $\widetilde { \Gamma } _ { \mathrm { m a x } }$ one can then construct the plumbing matrix of an associated plumbed manifold and by [32, Thm 1.1] one sees that this plumbed manifold is an integral homology sphere. The reader can find the relevant pseudocode in Algorithm 3.

## Maximal Splice Diagram from Splice Diagram

We first outline Algorithm 2 which generates the maximal splice diagram $\widetilde { \Gamma } _ { \mathrm { m a x } }$ from a splice diagram $\widetilde { \Gamma } .$

It works as follows: let $E = ( \nu _ { i } , \nu _ { j } )$ be an edge between two nodes $\nu _ { i }$ and $\nu _ { j }$ (see Figure 9).

![](images/1f21018561740e2e65ab06bba6aa987d75c30b25c3145ffa01d0deefc54a38d0.jpg)  
Figure 9: Starting point for algorithm

Set $\textstyle a = \prod _ { i = 1 } ^ { r } a _ { i }$ and $\begin{array} { r } { d = \prod _ { i = 1 } ^ { s } d _ { i } } \end{array}$ . In the case that either $\nu _ { i }$ or $\nu _ { j }$ is a leaf instead of a node, then we set a = 1 or d = 1 respectively. Consider the infinite linear graph in Figure 10.

x J On the infinite linear graph in Figure 10, vertices ● are ordered by the size of $x / y$ The algorithm proceeds by modifying the infinite graph via the process outlined in Figure 11

![](images/54722f4b92c12d840f3873b2c60a7a23a66087ccd03b6e292dcdd3ec2c66503a.jpg)  
Figure 10: The infinite linear graph used as a building block within the algorithm.

![](images/e0a3d3525d863c6ddc6384db1b4a977101286f05d653c3faf267c24f79ffa81e.jpg)  
Figure 11: Graph Refinement

The algorithm then replaces the edge $( \nu _ { i } , \nu _ { j } )$ by the extracted portion of the modified linear graph: the two endpoint vertices are identified with $\nu _ { i }$ and $\nu _ { j }$ (whose weights along this edge are unchanged), and the interior vertices subdivide the edge. This process is repeated for every edge of $\widetilde { \Gamma }$ and we call the resulting graph $\widetilde { \Gamma } _ { \mathrm { m a x } }$ the maximal splice diagram.

## Plumbing Graph from Maximal Splice Diagram

To retrieve the plumbing graph Γ from a maximal splice diagram $\widetilde { \Gamma } _ { \mathrm { m a x } }$ , recall that the underlying graph structure of $\widetilde { \Gamma } _ { \mathrm { m a x } }$ (i.e. the vertices and undirected edges) is the same as the graph structure of Γ. In particular this allows one to read off the adjacency matrix of Γ from the underlying graph structure of $\widetilde { \Gamma } _ { \mathrm { m a x } }$ . To find the plumbing matrix M associated with Γ , we are further required to find the weights $w _ { i }$ associated with the vertices $\nu _ { i }$ of Γ. In what follows, (similar to [32]), we use the notation $\nu _ { i } - \nu _ { j }$ to mean the vertices $\nu _ { i }$ and $\nu _ { j }$ are connected by an edge. If $\nu _ { k }$ is any vertex adjacent to $\nu _ { i }$ we obtain:

$$
w _ { i } = \frac { - 1 } { \ell _ { \nu _ { i } \nu _ { k } } } \left( \sum _ { \{ \nu _ { j } \in \mathrm { V e r t } ( \widetilde { \Gamma } _ { \mathrm { m a x } } ) | \nu _ { i } - \nu _ { j } \} } \ell _ { \nu _ { j } \nu _ { k } } \right)\tag{52}
$$

where the notation $\ell _ { \nu _ { i } \nu _ { i } }$ is used to denote the product of the weights adjacent $^ { \mathrm { t o , } }$ but not on, the shortest path from $\nu _ { i }$ to $\nu _ { j }$ (with $\ell _ { \nu _ { i } \nu _ { i } }$ denoting the product of the weights adjacent to $\nu _ { i } )$ in $\widetilde { \Gamma } _ { \mathrm { m a x } } .$

## E.2 Algorithms and Pseudocode

In this subsection we list the algorithms in the form of pseudocode for the interested reader.

## Algorithm 1: Splicing Leaf Repair

Data: splice diagram $\widetilde { \Gamma }$   
Result: splice diagram $\widetilde { \Gamma } _ { r }$ with weights on edges adjacent to leaves   
set $\widetilde { \Gamma } _ { r } = \widetilde { \Gamma }$   
for vertices $\nu _ { i }$ in $\widetilde { \Gamma }$ with deg $( \nu _ { i } ) = 1$ do   
let $\nu _ { j }$ be the node adjacent to the vertex/leaf $\nu _ { i }$   
set $b = w _ { \nu _ { j } \nu _ { i } }$ the weight on the edge $( \nu _ { j } , \nu _ { i } )$   
set $a = 1$   
for vertices $\nu _ { k }$ adjacent to $\nu _ { j }$ do   
if $\nu _ { k }$ ̸= $\nu _ { i }$ then   
1 $\boldsymbol { a } = \boldsymbol { a } \times \boldsymbol { w } _ { \nu _ { j } \nu _ { k } }$ the weight on the edge $( \nu _ { j } , \nu _ { k } )$   
set the weight on the edge $( \nu _ { i } , \nu _ { j } )$ to be equal to $\textstyle { \left\lceil { \frac { a } { b } } \right\rceil }$   
return $\widetilde { \Gamma } _ { r }$

Algorithm 2: Generate maximal splice diagram from splice diagram

```latex
Data: splice diagram $\widetilde { \Gamma }$
Result: Maximal splice diagram $\widetilde { \Gamma } _ { \mathrm { m a x } }$
set $\underset { ^ { \sim } } { \widetilde { \Gamma } } { } _ { r }$ equal to Splicing Leaf Repair of $\widetilde { \Gamma }$ by Algorithm 1
set $\widetilde { \Gamma } _ { \mathrm { m a x } } = \widetilde { \Gamma } _ { r }$
for all edges $( \nu _ { i } , \nu _ { j } )$ of $\widetilde { \Gamma } _ { r }$ do
set $b = w _ { \nu _ { i } \nu _ { j } }$ the weight on the edge $( \nu _ { i } , \nu _ { j } )$
set $c = w _ { \nu _ { j } \nu _ { i } }$ the weight on the edge $( \nu _ { j } , \nu _ { i } )$
set $a = 1$
Set $d = 1$
for $\boldsymbol { \nu } \in \{ \nu _ { i } , \nu _ { j } \}$ do
if v is a node then
for vertices $\nu _ { k }$ adjacent to v do
if edge $( \nu , \nu _ { k } ) \notin \{ ( \nu _ { i } , \nu _ { j } ) , ( \nu _ { j } , \nu _ { i } ) \}$ then
set $w _ { \nu \nu _ { k } }$ equal to the weight on the edge $( \nu , \nu _ { k } )$
if $\nu = \nu _ { i }$ then
$a = \bar { a } \times w _ { \nu \nu _ { k } }$
else
$d = d \times w _ { \nu \nu _ { k } }$
Let I be the infinite graph in Figure 10
Add edges to I by following the procedure in Figure 11 until vertices with
a b c d
and appear in I
a b
Replace the edge $( \nu _ { i } , \nu _ { j } )$ of $\widetilde { \Gamma } _ { \mathrm { m a x } }$ by the portion of I between and
c d
, identifying its endpoint vertices with $\nu _ { i }$ and $\nu _ { j }$
return $\widetilde { \Gamma } _ { \mathrm { m a x } }$
```

Algorithm 3: Convert a splice diagram to a plumbing graph

$$
\widetilde { \Gamma }
$$

Data: splice diagram   
Result: Plumbing matrix M of the plumbing graph Γ corresponding to $\widetilde { \Gamma }$   
convert splice diagram $\widetilde { \Gamma }$ to maximal splice diagram $\widetilde { \Gamma } _ { \mathrm { m a x } }$ using Algorithm 2   
set M equal to the adjacency matrix of underlying graph of $\widetilde { \Gamma } _ { \mathrm { m a x } }$   
for each vertex $\nu _ { i }$ in $\widetilde { \Gamma } _ { \mathrm { m a x } }$ do   
pick a vertex $\nu _ { k }$ adjacent to $\nu _ { i }$   
compute $w _ { i }$ via (52)   
set $M _ { i i } = w _ { i }$   
return M

As mentioned earlier, the resulting plumbing graph Γ produced from the splice diagram $\widetilde { \Gamma }$ is an integral homology sphere by [32, Thm 1.1].

## References

[1] A. Davies, P. Veliˇckovi´c, L. Buesing, S. Blackwell, D. Zheng, N. Tomašev, R. Tanburn, P. Battaglia, C. Blundell, A. Juhász, M. Lackenby, G. Williamson, D. Hassabis, and P. Kohli, “Advancing mathematics by guiding human intuition with ai,” Nature 600

no. 7887, (Dec, 2021) 70–74. https://doi.org/10.1038/s41586-021-04086-x.

[2] V. Jejjala, A. Kar, and O. Parrikar, “Deep Learning the Hyperbolic Volume of a Knot,” Phys. Lett. B 799 (2019) 135033, arXiv:1902.05547 [hep-th].

[3] J. Craven, V. Jejjala, and A. Kar, “Disentangling a deep learned volume formula,” Journal of High Energy Physics 2021 no. 6, (Jun, 2021) 40. https://doi.org/10.1007/JHEP06(2021)040.

[4] J. Craven, M. Hughes, V. Jejjala, and A. Kar, “Illuminating new and known relations between knot invariants,” Machine Learning: Science and Technology 5 no. 4, (Dec, 2024) 045061. https://doi.org/10.1088/2632-2153/ad95d9.

[5] S. Gukov, J. Halverson, F. Ruehle, and P. Sułkowski, “Learning to Unknot,” Mach. Learn. Sci. Tech. 2 no. 2, (2021) 025035, arXiv:2010.16263 [math.GT].

[6] M. C. Hughes, “A neural network approach to predicting and computing knot invariants,” Journal of Knot Theory and Its Ramifications 29 no. 03, (2020) 2050005, arXiv:1610.05744.

[7] J. S. F. Levitt, M. Hajij, and R. Sazdanovic, “Big data approaches to knot theory: Understanding the structure of the Jones polynomial,” Journal of Knot Theory and Its Ramifications 31 no. 13, (2022) 2250095, arXiv:1912.10086.

[8] A. Lindsay and F. Ruehle, “On the Learnability of Knot Invariants: Representation, Predictability, and Neural Similarity,” Advances in Theoretical and Mathematical Physics (2026) , arXiv:2502.12243 [math.GT]. To appear.

[9] S. Gukov, P. Putrov, and C. Vafa, “Fivebranes and 3-manifold homology,” JHEP 07 (2017) 071, arXiv:1602.05302 [hep-th].

[10] S. Gukov, D. Pei, P. Putrov, and C. Vafa, “BPS spectra and 3-manifold invariants,” J. Knot Theor. Ramifications 29 no. 02, (2020) 2040003, arXiv:1701.06567 [hep-th].

[11] M. C. N. Cheng, S. Chun, F. Ferrari, S. Gukov, and S. M. Harrison, “3d Modularity,” JHEP 10 (2019) 010, arXiv:1809.10148 [hep-th].

[12] M. C. N. Cheng, S. Chun, B. Feigin, F. Ferrari, S. Gukov, S. M. Harrison, and D. Passaro, “3-Manifolds and VOA Characters,” Commun. Math. Phys. 405 no. 2, (2024) 44, arXiv:2201.04640 [hep-th].

[13] M. C. N. Cheng, I. Coman, P. Kucharski, D. Passaro, and G. Sgroi, “3d Modularity Revisited,” arXiv:2403.14920 [hep-th].

[14] K. Bringmann, K. Mahlburg, and A. Milas, “Quantum modular forms and plumbing graphs of 3-manifolds,” 2019. https://arxiv.org/abs/1810.05612.

[15] K. Bringmann, K. Mahlburg, and A. Milas, “Higher depth quantum modular forms and plumbed 3-manifolds,” Letters in Mathematical Physics 110 no. 10, (July, 2020) 2675–2702. http://dx.doi.org/10.1007/s11005-020-01310-z.

[16] S. Gukov and R.-K. Seong, “Machine learning BPS spectra and the gap conjecture,” Phys. Rev. D 110 (Aug, 2024) 046016. https://link.aps.org/doi/10.1103/PhysRevD.110.046016.

[17] J. Halverson and F. Ruehle, “Learning Topological Invariance.” arXiv preprint, https://arxiv.org/abs/2504.12390, 2025.

[18] V. A. Rokhlin, “New results in the theory of four-dimensional manifolds,” Doklady Akademii Nauk SSSR (N.S.) 84 no. 2, (1952) 221–224. https://webhomes.maths.ed.ac.uk/\~v1ranick/papers/rohlin.pdf.

[19] D. Galewski and R. J. Stern, “Classification of simplicial triangulations of topological manifolds,” Annals of Mathematics 111 no. 1, (1980) 1–34. https://www.jstor.org/stable/1971215.

[20] T. Matumoto, “Triangulation of manifolds,” Algebraic and Geometric Topology 3 (1978) 1–24.

[21] C. Manolescu, “Pin(2)-equivariant Seiberg–Witten Floer homology and the triangulation conjecture,” Journal of the American Mathematical Society 29 no. 1, (2016) 147–176, arXiv:1303.2354.

[22] R. Fintushel and R. J. Stern, “Instanton homology of seifert fibred homology three spheres,” Proceedings of the London Mathematical Society 61 no. 1, (1990) 109–137. https://doi.org/10.1112/plms/s3-61.1.109.

[23] M. Furuta, “Homology cobordism group of homology 3-spheres,” Inventiones mathematicae 100 no. 2, (1990) 339–355. https://doi.org/10.1007/BF01231190.

[24] K. A. Frøyshov, “Equivariant aspects of Yang–Mills Floer theory,” Topology 41 no. 3, (2002) 525–552.

[25] P. S. Ozsváth and Z. Szabó, “Absolutely graded Floer homologies and intersection forms for four-manifolds with boundary,” Advances in Mathematics 173 no. 2, (2003) 179–261. https://arxiv.org/abs/math/0110170.

[26] K. Hendricks and C. Manolescu, “Involutive heegaard floer homology,” Duke Mathematical Journal 166 no. 7, (2017) 1211–1299. https://arxiv.org/abs/1507.00383.

[27] I. Dai, J. Hom, M. Stoffregen, and L. Truong, “An infinite-rank summand of the homology cobordism group,” Duke Mathematical Journal 172 no. 12, (2023) 2365–2432. https://arxiv.org/abs/1810.06145.

[28] O. ¸Savk, “A survey of the homology cobordism group,” Bulletin of the American Mathematical Society 61 no. 1, (2024) 119–157. https://arxiv.org/abs/2209.10735.

[29] S. Gukov, S. Park, and P. Putrov, “Cobordism Invariants from BPS q-Series,” Annales Henri Poincaré 22 no. 12, (July, 2021) 4173–4203. http://dx.doi.org/10.1007/s00023-021-01089-2.

[30] S. Harichurn, “On the $\Delta _ { a }$ invariants in non-perturbative complex Chern–Simons theory,” Letters in Mathematical Physics 115 (Nov., 2025) 136. https://doi.org/10.1007/s11005-025-01991-4.

[31] S. Harichurn, A. Némethi, and J. Svoboda, “∆ Invariants of Plumbed Manifolds,” Symmetry, Integrability and Geometry: Methods and Applications (Oct., 2025) 1–30. http://dx.doi.org/10.3842/SIGMA.2025.091.

[32] W. D. Neumann and J. Wahl, “Complex surface singularities with integral homology sphere links,” Geometry & Topology 9 no. 2, (2005) 757–811.

[33] M. Borodzik and A. Némethi, “Heegaard–floer homologies of (+1) surgeries on torus knots,” Acta Mathematica Hungarica 139 no. 4, (Jun, 2013) 303–319. https://doi.org/10.1007/s10474-012-0280-x.

[34] B. Robinson, S. Harichurn, F. Ruehle, S. Gukov, R.-K. Seong, and M. C. N. Cheng, “Learning Topological Features of Zb Invariants: Supplementary Material.” Zenodo dataset, Aug., 2026. https://doi.org/10.5281/zenodo.18734903.

[35] R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh, and D. Batra, “Grad-cam: Visual explanations from deep networks via gradient-based localization,” International Journal of Computer Vision 128 no. 2, (Oct., 2019) 336âA¸S359<sup>˘</sup> . http://dx.doi.org/10.1007/s11263-019-01228-7.

[36] S. M. Lundberg and S.-I. Lee, “A unified approach to interpreting model predictions,” Advances in neural information processing systems 30 (2017) 4768–4777.

[37] L. S. Shapley, “A value for n-person games,” in Contributions to the Theory of Games II, H. W. Kuhn and A. W. Tucker, eds., pp. 307–317. Princeton University Press, Princeton, 1953.

[38] M. T. Ribeiro, S. Singh, and C. Guestrin, ““Why should I trust you?” Explaining the predictions of any classifier,” in Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pp. 1135–1144. 2016. https://doi.org/10.1145/2939672.2939778.

[39] A. Shrikumar, P. Greenside, A. Shcherbina, and A. Kundaje, “Not just a black box: Learning important features through propagating activation differences.” arXiv preprint, https://arxiv.org/abs/1605.01713, 2016.

[40] S. Chopra, R. Hadsell, and Y. LeCun, “Learning a similarity metric discriminatively, with application to face verification,” in 2005 IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR’05), vol. 1, pp. 539–546. 2005. https://doi.org/10.1109/CVPR.2005.202.

[41] K. Sohn, “Improved deep metric learning with multi-class n-pair loss objective,” in Advances in Neural Information Processing Systems, D. Lee, M. Sugiyama, U. Luxburg, I. Guyon, and R. Garnett, eds., vol. 29. Curran Associates, Inc., 2016. https://proceedings.neurips.cc/paper\_files/paper/2016/file/ 6b180037abbebea991d8b1232f8a8ca9-Paper.pdf.

[42] H. Rademacher and E. Grosswald, Dedekind Sums. No. 16 in The Carus Mathematical Monographs. The Mathematical Association of America, Washington, D.C., 1972.

Figures  
![](images/9c432cebc82baba975090a08f2e21075c8aa8e407a280991c9ce3a8c17b0ea27.jpg)  
Figure 12: Heatmap of correlation matrix of the features used as input to regression networks and their binary labels.

## Confusion Matrices —Full sequence

HOM-All BAS=0.952±0.003  
![](images/3702eb06dac3a5a61be46a0385ca048ad9363bfe553ca038d1963cc79d3a5bd5.jpg)

![](images/6fe53078dc5d4dd150fa93e289dc192230ba756be21fba5f169dcf92d6ee24a6.jpg)

![](images/992216cf68179cfec3e1a58c5aff28be2ec9f53c0e1196d7165d22dec705b5fd.jpg)

![](images/0ed9e7558d8796fbd7c8999bb33c6d9aaf3fdcfc678fb7d7f9e19402d36b1206.jpg)

![](images/379bca67b2ff7bbb9b5e869d5a4a175d5977d79fac94fd55246290e655afaccd.jpg)

![](images/d7cfd12b9d63207166162e0489078c39758e79e4c8e06c63333ebcded9c74401.jpg)

![](images/202a390f0e08d469826601afca2e4b5bac879acd62c2ddb262901a047b047290.jpg)  
Figure 13: Full Sequence. Mean confusion matrices across five seeds for all HOM and TOP experiments.

![](images/2708759dee907b15da72e10a627a98b1c3c842ff8f5bf90c25830144d6e1af4d.jpg)  
Figure 14: Full Sequence. Logit space visualization for a representative seed (seed-456) for all HOM and TOP experiments. For display only, the two logits are standardized within each panel and central quantile filtering is used to avoid axis ranges being determined by a small number of extreme logits.

![](images/017ece1ef6baed1c9bc9e60eab38906260c2a421883bed37e56961575fdce678.jpg)  
Figure 15: Full Sequence. Mean normalized feature importance scores for feature saliency, permutation importance, and SHAP; averaged over all five seeds for all HOM and TOP experiments.

![](images/68e9d79444a822a1c6860ae7cb05eadba6b98f7c68c4bd178c5cce759d2b6051.jpg)

![](images/d8b5d6ae0cff28d28680a96f944184c0c3145ac056a023794dcd02f190e70b45.jpg)

![](images/56758ee994a52676a54f6f9c2426b0890ae81e274cbe0f314e0fbe67c005d4cc.jpg)

![](images/41c0471e24aada3f995a0009e340cd3b3e51d64b4e554f83bba3943152087a38.jpg)

TOP-Non BAS=1.000±0.000  
![](images/07944e7f0d925d41d7ba6b32564fe934cbc27f2b33fe2214abc73b78fcc32026.jpg)

TOP-AII BAS=1.000±0.000  
![](images/77ec7c952930cd8612b79ab7c4a489cd570bd76248e9bfc4df66f5321aa729cd.jpg)  
Figure 16: Value-Index. Mean confusion matrices across five seeds for all HOM and TOP experiments.

![](images/2b5797ed1143e93a5eb182bdbc97f4c1d706a1b13eb2dd7ff44bc594c2da8fe1.jpg)

![](images/cbbce5c154309affcf51ad3ce852945452ef97cdc15ce29de20630102501308b.jpg)

![](images/796bed05dbc72aa6635cabf62fd551e1c017a940c3374c57ca0aa5aca90acce3.jpg)

![](images/a9be5b5e78f069bccea1f6a4ed1830705b05ad7ecbe2532db5ecd5a4c53e061c.jpg)

![](images/97739f807390edc9cf368bde905c454ce4abf54076cb3b99bdb2bb872f3f5ae0.jpg)

![](images/7f90ba74d0e62301f93ba7178f8a5d783a78cf98857a7673f5abe27602e174e3.jpg)  
Figure 17: Value-Index. Logit space visualization for a representative seed (seed-42) for all HOM and TOP experiments.

![](images/b0cabba390cc0dc890e0d2d47628474ed733d25a596f0169b586ac79b24d0c7f.jpg)  
Figure 18: Value-Index. Mean normalized feature importance scores for feature saliency, permutation importance, and SHAP; averaged over all five seeds for all HOM and TOP experiments.

![](images/ce9fa3d6fcc60405c4f84dd2de5c31b4e7d8206a781ed1fe934103c7442f5ae2.jpg)

![](images/cb0dbed1dc82c92d1bacc4034812a0d557420a3b78574602261334f96631bfa6.jpg)

![](images/1c067a446e333b71da13992b244e667a7ddd59d06dbf6e8218f007b4e87ec1a8.jpg)

![](images/b8f9666882cde5980a92456954aa16902a90d2a8375886a983bf1e32be709538.jpg)

![](images/59f494eaf05e39f370117d727bdd0a48795ca4c1cb1c6916569b7a42c1642fca.jpg)

TOP-All BAS=0.992±0.001  
![](images/35a74cb72b09cf18a9e08e9d95a52cbc1b73ae22d2cb80b1fed1cb0e2345ec53.jpg)  
Figure 19: Latent Space. Mean confusion matrices across five seeds for all HOM and TOP experiments.

![](images/a4202df9f02a8e6653aed402dd84ca5adf00729b00bb969a44ca0dc51df34d18.jpg)  
Figure 20: Latent Space. Logit space visualization for a representative seed (seed-42) for all HOM and TOP experiments.

![](images/9ee5331eb4f9ed7ee1ada977addfe40d56385eef39f96d45ebb5faae94b88ba8.jpg)  
Figure 21: Latent Space. Mean normalized feature importance scores for feature saliency, permutation importance, and SHAP; averaged over all five seeds for all HOM and TOP experiments.

![](images/d3d5122db395c23c33f201ce24d00ea05ccd6cf38c56d28db6ade94221421148.jpg)

![](images/59f99bccb0cdfbdc5c9bc3495cbe086ae83b6deff0337576c3dcb36bc0220d62.jpg)

![](images/74266a464035abeeab9c844a41ce05b821d7b7e056234b91eb374816f5910ba6.jpg)

![](images/6b1b44c9dd5692d3d0b0df4e36fd63109cdb70fa65a311bf3edf23b76be5b236.jpg)  
Figure 22: Multi-class Latent Space; seed-42. Top Left: SVM decision boundaries in latent space separating STAR-Sph, STAR-Non, H-Sph, and H-Non classes. TOP Right: SVM prediction confidence map. Bottom Left: SVM support vectors. Bottom Right: SVM and linear, binary decision boundaries overlaid with seed-42 binary boundaries displayed as thick dashed lines and the linear boundaries for other seeds rendered as faint, dashed lines.

SL₂(Z) orbit regression: ensemble calibration  
![](images/e97290951482007e4e31217c082b93a2aa8d3582c3e7683b306ced00838ce139.jpg)  
(a)

SL2(Z) per-seed calibration overlay  
![](images/9e70655dd1e7dc02b062a6d1318ee0b38270cf103d1c1771f5f97c992edcc032.jpg)  
(b)  
Figure 23: Learning $S L _ { 2 } ( \mathbb { Z } )$ orbit. (a) In-distribution $S L _ { 2 } ( \mathbb { Z } )$ -orbit regression. Points show the ensemble-mean transformer prediction α against the true orbit size α over the unique heldout samples appearing across the 30-seed robustness run. Color indicates the standard deviation of the prediction across seeds. The dashed line is the ideal relation ${ \widehat { \alpha } } = \alpha ,$ , while the solid line is the fitted calibration. The ensemble prediction shows strong in-distribution interpolation but visible seed sensitivity and degradation for some large-orbit examples. (b) Seed overlay where each point is a held-out prediction from one of the 30 transformer seeds. The dashed line is the ideal relation $\widehat { \alpha } = \alpha$ . The overlay shows that most seeds learn the same global linear trend, but the spread increases for large orbit sizes and for a small set of lower-α outliers.

![](images/a0911a485017ed55bd3bebcb453b388a71b5d0e34666f0242145d30716d5a89e.jpg)  
Figure 24: Learning d from ${ \widehat { Z } } .$ . Normalized mean exponent feature importance.

Integer prediction progression (10-seed mean ± std)

![](images/5acd9570317d3786729e498af74867f8c4c51e961c0aa4ac7d5623b636b5cfe2.jpg)

Errors cluster at rounding boundary  
![](images/ba616f35efbc58c2486bf92928d865f3f5102fe1a32e3d9f96150c1ac66e6f0e.jpg)

![](images/2949dddb551d7667d3bce98f3057c7ae2cba08cfdefe52e93855740335ce02e4.jpg)

![](images/af5dc706321ac3ba09d9001787344d9a0335369431d227ea6ad7dee44a1bd978.jpg)  
Figure 25: Error profile for learning d exactly, on the $n = 6 8 5$ three-fiber dataset $\mathcal { D } _ { 3 } ^ { 2 0 }$ . Top left: exact-integer accuracy at each stage (from left: direct rounding with ∆ appended, the halved target $t / 2 ,$ all nineteen offsets, the symmetrized Dedekind features), mean ± one standard deviation over the ten seeds; the offsets-only starting point (17.9%) is discussed in the text. The remaining panels show out-of-fold predictions under the seed-42 fold assignment, for which the best model is exact on 89.2% (611/685), every miss off by exactly ±2 in $t .$ Top right: predicted vs. true $\begin{array} { r } { t = d + \Delta - \frac { 1 } { 2 } } \end{array}$ . Bottom left: residual histograms of the best model (halved target, MAE 0.277) and of the first bar’s rounding model (full target, MAE 0.950) under the same folds; each is plotted in its own rounding variable, so $\begin{array} { r } { | \varepsilon | < { \frac { 1 } { 2 } } } \end{array}$ marks a correctly rounded prediction in both, and the axis is truncated to [<sup>−</sup>2, 2]. Bottom right: best-model |ϵ| against $\Delta _ { : }$ , with the misses marked.