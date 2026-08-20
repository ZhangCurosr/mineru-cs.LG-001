# A Unifying Relational Perspective on Expressive Lottery Tickets

Lorenz Kummer <sup>1</sup> <sup>2</sup> Samir Moustafa <sup>1</sup> <sup>2</sup> <sup>3</sup> Anatol Ehrlich <sup>1</sup> <sup>2</sup> Franka Bause <sup>4</sup> Marco Nennstiel <sup>1</sup> Przemysław Andrzej Wał˛ega <sup>5</sup> Nils Morten Kriege <sup>1</sup> <sup>6</sup>

## Abstract

Graph neural networks (GNNs) are widely used, but how parameter sparsity affects the expressivity of relational (RGNNs) and temporal (TGNNs) variants is poorly understood. The Strong Expressive Lottery Ticket Hypothesis (SELTH) posits the existence of sparse GNNs that preserve Weisfeiler-Leman (WL) expressivity on static graphs. We generalize this existence result to a probabilistic statement for multi-relational and temporal domains via the relational WL (RWL). We prove that sufficiently parameterized RGNNs contain sparse subnetworks that maintain 1-RWL expressivity and derive a lower bound on the probability that a random pruning yields such a subnetwork. We show that common TGNNs and crossgraph message passing schemes admit RGNN reformulations such that they inherit these guarantees and, moreover, that the expressivity of a sparse RGNN is connected to its optimization behavior under common update regimes. Experiments instantiate the bound, compare it to empirical probabilities on synthetic data, and study how pre-training expressivity relates to optimization and prediction quality metrics on temporal and molecular benchmarks.

## 1. Introduction

GNNs are powerful models for learning on graph-structured data, where complex objects such as molecules, proteins, or social networks are represented as graphs with featureenriched nodes and edges. They extend deep learning to domains including finance, social networks, medical data, and chem-/bioinformatics (Lu & Uddin, 2021; Cheung & Moura, 2020; Sun et al., 2021; Gao et al., 2022; Wu et al., 2018; Xiong et al., 2021). A central research direction concerns expressivity, i.e., the ability to distinguish nonisomorphic graphs, typically benchmarked against the WL test, with 1-WL serving as a standard baseline. Foundational advances such as Deep Sets (Zaheer et al., 2017) and Graph Isomorphism Network (GIN) (Xu et al., 2019) established 1-WL as the reference point for evaluating GNN expressivity, spurring efforts to surpass its limitations (Morris et al., 2023). Nonetheless, 1-WL remains sufficient for distinguishing most graphs in widely used benchmarks (Morris et al., 2021; Zopf, 2022), underscoring its practical relevance. These advances, however, do not address the multirelational setting and focus exclusively on static graphs.

For multi-relational graphs, Barceló et al. (2022) and Huang et al. (2023b) introduce the relational WL (1-RWL) and relate it to the expressivity of RGNNs, as proposed by, e.g., Schlichtkrull et al. (2018); Vashishth et al. (2020), and present k-relational GNNs that provably surpass 1-RWL. For temporal graphs, Wał˛ega & Rawson (2025) analyze temporal message passing (MP) by distinguishing global and local TGNNs, provide RWL characterizations for both, and show that their expressivities are incomparable in general but that local models are more powerful on color-persistent temporal graphs. In contrast, Heeg et al. (2025) study temporal isomorphism via time-respecting paths and propose an RWL test on augmented event graphs, grounding expressivity in the temporal causal topology rather than in the global/local MP distinction.

The Lottery Ticket Hypothesis (LTH), introduced by Frankle & Carbin (2019), posits that large, randomly initialized neural networks contain smaller subnetworks, termed winning tickets, that can be trained in isolation to match the full model’s prediction quality. LTH has also been extended to GNNs via pruning of both graph structures and GNN parameters, resulting in Graph Lottery Tickets (GLTs) that retain prediction quality while lowering computational demands (Chen et al., 2021; Tsitsulin & Perozzi, 2023). Recently, Kummer et al. (2025b) formalized the link between the LTH and GNN expressivity for static, uni-relational graphs: at a minimum, a winning ticket must allow the model to learn to distinguish any non-isomorphic graphs (or nodes) associated with different targets in the downstream task. Under certain circumstances, a sparsely initialized GNN cannot recover from a loss of expressivity. Kummer et al. (2025b) conclude that preserving expressivity under sparsity is crucial for successfully identifying winning lottery tickets and show that sparse subnetworks exist which preserve 1-WL expressivity, thereby introducing the SELTH. Their analysis is purely existential, restricted to static unirelational graphs and the corresponding 1-WL variant, and does not extend to temporal or multi-relational settings, which can be parameter heavy (Ehrlich et al., 2026).

To address this gap, we leverage Wał˛ega & Rawson (2025)’s characterization of local and global MP schemes as instances of 1-RWL on suitable abstractions. Together with the RGNN framework, this enables us to extend the SELTH beyond static graphs, proving that sparsely initialized subnetworks can preserve 1-RWL expressivity in both temporal and multi-relational settings, thereby establishing SELTH as a general principle for expressive and efficient RGNNs and TGNNs. We further apply the framework to cross-graph and hierarchical MP (Fey et al., 2020; Ehrlich et al., 2026).

Related work. Building on Frankle & Carbin (2019)’s LTH, Frankle et al. (2019) prune early in training rather than strictly at initialization, and obtain sparse subnetworks without loss of prediction quality or stability. Malach et al. (2020); da Cunha et al. (2022) extend to the Strong Lottery Ticket Hypothesis (SLTH), proving that over-parameterized networks contain subnetworks that achieve high prediction quality without training. Zhang et al. (2021a) explain LTH’s generalization benefits: pruning enlarges a convex region of the loss landscape, enabling faster convergence with fewer samples. Zhang et al. (2021b) further validate that pruned subnetworks can match dense networks without repeated prune-retrain cycles.

In GNNs, LTH approaches typically co-prune graph structure and model parameters. Chen et al. (2021); Wang et al. (2023) and Sui et al. (2023) co-prune adjacency matrices and weights to recover GLTs that retain prediction quality and Zhang et al. (2024) automate adaptive pruning to find GLTs without manual tuning. Hui et al. (2023) and Yuxin et al. (2024) consider co-pruning from an adversarial robustness perspective. Tsitsulin & Perozzi (2023) posit that any graph contains a sparse substructure preserving downstream prediction quality. Yan et al. (2024) leverage SLTH to improve memory efficiency. Kummer et al. (2025b) formally and empirically linked LTH and GNN expressivity via SELTH. In line with Zhang et al. (2021b), Kummer et al. (2025b) show that expressive sparse initializations can speed convergence and improve generalization.

Relating LTH to TGNN expressivity is challenging, as the very notion of isomorphism on temporal graphs is still being clarified. Wał˛ega & Rawson (2025) lay out two seminal notions of temporal isomorphism: pointwise isomorphism (Beddar-Wiesing et al., 2024), which compares snapshots independently, and timewise isomorphism, which additionally preserves inter-snapshot time gaps. They formalize global (Longa et al., 2023; Xu et al., 2020; Luo & Li, 2022) and local (Qu et al., 2020; Rossi et al., 2020) MP TGNNs via knowledge graphs and prove tight 1-RWL characterizations. Moreover, they note that event-based temporal graphs can be transformed into the snapshot representation (Gao & Ribeiro, 2022; Longa et al., 2023), concluding that analyzing snapshots subsumes event-based inputs. Hence, our work focuses on this snapshot representation to maximize its generality. Complementing these snapshot-oriented notions, Heeg et al. (2025) introduce consistent event-graph isomorphism and prove it is equivalent to static isomorphism on an augmented event graph. Other notions of temporal graph isomorphism are provided by Gao & Ribeiro (2022); Souza et al. (2022). Furthermore Souza et al. (2022) show that TGNNs with injective aggregation/updates are as expressive as a variant of 1-WL that refines colors using timestamped interactions, whereby walk-aggregating and MP TGNNs are incomparable.

Similarly, the expressivity of cross-graph and hierarchical MP architectures has, to the best of our knowledge, not yet been formally characterized or discussed under sparsity. While Ehrlich et al. (2026) formally show that their proposed architecture (XIMP) subsumes Fey et al. (2020) (HIMP), which, in turn, can distinguish some exemplary molecular graphs indistinguishable to 1-WL-bounded GNNs, neither provides formal guarantees.

Contributions. We generalize SELTH to (i) multirelational RGNNs, (ii) local and global TGNNs and (iii) crossgraph and hierarchical MP architectures. We formally show that sparse subnetworks in these architectural classes exist which preserve expressivity under the 1-RWL framework. That is, we prove Relational SELTH (RSELTH), which subsumes the original SELTH, and derive an explicit lower bound on the probability that a random pruning mask yields a maximally expressive subnetwork. We further show that every local or global TGNN can be rewritten as an RGNN, to which RSELTH applies, and that RSELTH similarly applies to certain cross-graph and hierarchical MP patterns, for which our 1-RWL characterization also marks the first formal characterization of their expressivity. For TGNNs, we connect RSELTH with different notions of temporal isomorphism. Finally, we analyze the optimization of sparse RGNNs with the minimal expressivity required to realize all task-relevant distinctions, and show that all surviving parameters remain trainable under small-step first-order methods.

## 2. Preliminaries

An undirected graph G is a pair $( V , E )$ of a finite set of nodes V and edges $E \subseteq \{ \{ u , v \} \mid u , v \in V \}$ . The node and edge sets of G are denoted by $V ( G )$ and $E ( G )$ , respectively. The neighborhood of a node $v ~ \in ~ V ( G )$ is $N ( v ) = \{ u \in V ( G ) \mid \{ u , v \} \in E ( G ) \}$ . For a node labeled graph G, we write $G = ( V , E , \lambda )$ , where $\lambda : V ( G ) \to \mathcal { X }$ maps to a label set X. For an edge labeled graph G, we write $G = ( V , E , \tau )$ , where $\tau : E  \mathcal { R }$ maps to a label set R. If G is a directed graph, we write ${ \dot { E } } \subseteq V \times V$ and $\dot { \tau } : \dot { E }  \mathcal { R }$ We define the outgoing and incoming neighborhoods of $v \in V$ as $N ^ { - } ( v ) = \{ u \in V \mid ( v , u ) \in \dot { E } \} , N ^ { + } ( v ) =$ $\{ u \in V \mid ( u , v ) \in \dot { E } \}$ and $N ^ { \pm } ( v ) = N ^ { - } ( v ) \cup N ^ { + } ( v )$ If G is directed and R is a set of relations (edge types), we refer to G as directed multi-relational graph and write $\dot { E } _ { r } \ = \ \{ ( u , v ) \in \dot { E } | \dot { \tau } ( u , v ) \ = \ r \}$ for $r \in \mathcal { R }$ and $N _ { r } ^ { - } ( v ) \ = \ \{ u \ \in \ V \ | \ ( v , u ) \in \dot { E } _ { r } \} , \ N _ { r } ^ { + } ( v ) \ = \ \{ u \ \in$ $V ~ \mid ~ ( u , v ) \in ~ \dot { E } _ { r } \}$ and $N _ { r } ^ { \pm } ( v ) = N _ { r } ^ { - } ( v ) \cup N _ { r } ^ { + } ( v )$ . If G is undirected multi-relational with $\tau : E ( G )  \mathcal { R }$ we write $E _ { r } ( G ) \ = \ \{ \{ u , v \} \in { \cal E } ( G ) \ | \ \tau ( \{ u , v \} ) \ = \ r \}$ and $N _ { r } ( v ) \ : = \ : \{ u \in \ : V ( G ) \mid \{ u , v \} \in { \cal E } _ { r } ( G ) \}$ . If a bijection $\varphi \colon V ( G ) \to V ( H )$ with $\{ u , v \} \in E ( G ) \iff$ $\{ \varphi ( u ) , \varphi ( v ) \} ~ \in ~ E ( H )$ for all $u , v \in V ( G )$ exists, we call the two undirected unlabeled graphs G and H isomorphic and write $G \simeq H$ For node labeled graphs $H , G ,$ , the bijection must satisfy $\lambda _ { H } ( \varphi ( v ) ) = \lambda _ { G } ( v )$ for all $v \in G .$ For undirected edge labeled graphs $H , G ,$ the bijection must satisfy $\tau _ { H } ( \{ \varphi ( u ) , \varphi ( v ) \} ) = \tau _ { G } ( \{ u , v \} )$ for all $\{ u , v \} \ \in \ E ( G )$ and, for directed edge labeled graphs, $\dot { \tau } _ { H } ( \varphi ( u ) , \varphi ( v ) ) = \dot { \tau } _ { G } ( u , v )$ for all $( u , v ) \in { \dot { E } } ( G )$ For directed multi-relational graphs $H , G ,$ , the bijection must satisfy $( u , v ) \ \in \ \dot { E } _ { r } ( G ) \quad \Longleftrightarrow \quad ( \varphi ( u ) , \varphi ( v ) ) \ \in$ $\dot { E } _ { r } ( H )$ , where $\dot { E } _ { r } ( G ) = \{ ( u , v ) \in \dot { E } ( G ) \mid \dot { \tau } _ { G } ( u , v ) = r \}$ for all $r \in \mathcal { R }$ . For undirected multi-relational graphs $H , G ,$ , the bijection must satisfy $\{ u , v \} \in E _ { r } ( G ) \iff$ $\{ \varphi ( u ) , \varphi ( v ) \} \ \in \ E _ { r } ( H )$ , where $\dot { E } _ { r } ( G ) ~ = ~ \{ \{ u , v \} ~ \in$ $\dot { E } ( G ) \mid \tau _ { G } ( u , v ) = r \}$ for all $r \in \mathcal { R }$

Relational Weisfeiler-Leman. Let $G = ( V , \dot { E } , \lambda , \dot { \tau } )$ be a directed, node-labeled, multi-relational graph with relation set R. The (directed) 1-RWL test maintains a node coloring $c _ { G } ^ { ( k ) } : V ( G ) \to { \mathcal { C } }$ over iterations $k \geq 0 ,$ , initialized as $c _ { G } ^ { ( 0 ) } = \lambda .$ . Given $c _ { G } ^ { ( k ) }$ , the next coloring is obtained by aggregating, for each relation $r \in \mathcal { R }$ , the multisets of colors in the relation-specific outgoing and incoming neighborhoods $N _ { r } ^ { - } ( v )$ and $N _ { r } ^ { + } ( v )$ , and then applying an injective hashing scheme to obtain new, unused colors. Concretely, similar to (Huang et al., 2023b), we write the update rule as

$$
\begin{array} { r l r } & { } & { c _ { G } ^ { ( k + 1 ) } ( v ) = \Upsilon _ { 3 } \Big ( c _ { G } ^ { ( k ) } ( v ) , ( \left\{ c _ { G } ^ { ( k ) } ( u ) \vert u \in N _ { r } ^ { - } ( v ) \right\} \big ) _ { r \in { \mathcal R } } , } \\ & { } & { ( \left\{ c _ { G } ^ { ( k ) } ( u ) \vert u \in N _ { r } ^ { + } ( v ) \right\} \big ) _ { r \in { \mathcal R } } \Big ) , } \end{array}
$$

where $\Upsilon _ { 3 }$ is any fixed injective encoding of its arguments into a new color in C. Let $C _ { \mathrm { r w l } } ^ { ( k ) } ( G ) \ = \ \Downarrow \ c _ { G } ^ { ( k ) } ( v ) \ \mid \ v \ \in$ $V ( G ) \mathbb { J }$ denote the multiset of node colors at iteration k. The refinement stabilizes once $\vert C _ { \mathrm { r w l } } ^ { ( k ) } ( G ) \vert = ~ \vert C _ { \mathrm { r w l } } ^ { ( k + 1 ) } ( G ) \vert$ For two input graphs $G , H$ , if there exists $k \geq 0$ such that $C _ { \mathrm { r w l } } ^ { ( k ) } ( G ) \bar { \neq } C _ { \mathrm { r w l } } ^ { ( \bar { k } ) } ( \bar { H } )$ , then $G \not \simeq _ { \mathrm { R W L } } ( k )$ H. Finally, 1-RWL reduces to classical 1-WL if $| \mathcal { R } | = 1$ and G is undirected. Therefore, it can be seen as the relational counterpart to 1-WL, serving as an expressivity baseline for RGNNs.

Relational GNNs. Let $G = ( V , \dot { E } , \lambda , \dot { \tau } )$ be a directed, node-labeled, multi-relational graph with relation set $\mathcal { R } .$ We initialize node embeddings from labels via an encoder $\Lambda : \mathcal { X } \to \mathbb { R } ^ { d } , \mathrm { i . e . , } h _ { v } ^ { ( 0 ) } = \Lambda ( \lambda ( v ) )$ . A relational GNN (RGNN) layer produces $\smash { h ^ { ( l + 1 ) } }$ from $\it { h ^ { ( l ) } }$ at layer $l \leq L$ by first aggregating messages per relation and direction, and then combining the resulting summaries in a permutationinvariant fashion (Barceló et al., 2022).

Representative of common RGNN architectures such as (Schlichtkrull et al., 2018; Vashishth et al., 2020; Huang et al., 2023b), we define an RGNN layer that mirrors directed 1-RWL refinement $G = ( V , \dot { E } , \lambda , \dot { \tau } )$ as

$$
m _ { r , - } ^ { ( l ) } ( v ) = \Omega _ { a } ( \{ \Phi _ { r , - } ^ { ( l ) } ( h _ { u } ^ { ( l ) } ) , u \in N _ { r } ^ { - } ( v ) \} ) ,\tag{1}
$$

$$
m _ { r , + } ^ { ( l ) } ( v ) = \Omega _ { a } ( \{ \Phi _ { r , + } ^ { ( l ) } ( h _ { u } ^ { ( l ) } ) , u \in N _ { r } ^ { + } ( v ) \} )\tag{2}
$$

for each $r \in \mathcal { R }$ and

$$
h _ { v } ^ { ( l + 1 ) } = \Gamma ^ { ( l ) } \Big ( \Omega _ { c } \big ( \{ h _ { v } ^ { ( l ) } \} \cup \bigcup _ { \mathfrak { p } \in \{ + , - \} } \{ \{ m _ { r , \triangleright } ^ { ( l ) } ( v ) , r \in \mathcal { R } \} \} \big ) \Big ) .\tag{3}
$$

Here, each $\Phi _ { r \pm } ^ { ( l ) } : \mathbb { R } ^ { d }  \mathbb { R } ^ { d }$ and $\Gamma ^ { ( l ) } : \mathbb { R } ^ { d }  \mathbb { R } ^ { d }$ are learnable functions and ⊎ is the additive multiset union: if $A , B$ are multisets with multiplicity functions $\mu _ { A } , \mu _ { B }$ then $\mu _ { A \uplus B } ( x ) = \mu _ { A } ( x ) + \mu _ { B } ( x )$ . Using $\displaystyle \sum$ as multiset aggregator $\Omega _ { a }$ and combinator $\Omega _ { c }$ and injective $\Phi _ { r } ^ { ( l ) }$ and $\Gamma ^ { \overline { { ( l ) } } } \left( \mathrm { { e . g } } \right.$ ., MLPs with suitable activations and width (Amir et al., 2023; Puthawala et al., 2022; Zaheer et al., 2017)) yields the same distinguishing power as 1-RWL:

Remark 2.1. Using $\displaystyle \sum$ aggregation as a permutationinvariant multiset encoder over the elements of a multiset D alone is injective if the elements of D are linearly independent (Kummer et al., 2025a). Hence $\Phi _ { r , \pm } ^ { ( l ) }$ and Λ must be chosen accordingly.

Note that normalizations (e.g., degree-based scalings) can be inserted into the sums without changing the 1-RWL upper bound (Barceló et al., 2022). We denote the L-layer RGNN architecture stacking layers of the form $( 1 ) – ( 3 )$ as $\Psi ^ { ( L ) }$ and write $\Psi _ { \Theta } ^ { ( L ) }$ for a concrete parameterization Θ.

Temporal graphs. A temporal graph in the snapshot model is a finite sequence $T G = \left( ( G _ { 1 } , t _ { 1 } ) , \dots , ( G _ { n } , t _ { n } ) \right)$ of undirected node labeled graphs with strictly increasing (real-valued) timestamps $t _ { 1 } < \cdots < t _ { n }$ , where each snapshot $G _ { i } = ( V , E _ { i } , \lambda _ { i } )$ shares the same node set V (node labels may vary with i) (Wał˛ega & Rawson, 2025). For $T G ,$ timestamped nodes are defined as $V ^ { \mathrm { t i m e } } = \{ ( v , t _ { i } ) \ |$ $v ~ \in ~ V , ~ i ~ \in ~ \{ 1 , \dots , n \} \} ~ = ~ \operatorname { t - n o d e s } ( T G )$ with timeaware labels $\lambda ^ { \mathrm { t i m e } } ( v , t _ { i } ) : = \lambda _ { i } ( v )$ For a timestamped node $( v , t )$ , we follow Souza et al. (2022) and define its temporal neighborhood as $N ( v , t _ { j } ) ~ = ~ \{ ( u , t _ { i } )$ $\{ u , v \} \ \in \ E _ { i } .$ , for some $( G _ { i } , t _ { i } ) ~ \in ~ { \cal T } \bar { G }$ with $t _ { i } ~ \leq ~ t _ { j } \}$ We overload $N ( \cdot )$ by context: when the second argument is a time $t _ { j } , \ N ( v , t _ { j } )$ denotes the multiset/set of timestamped neighbors up to (and including) time $t _ { j }$ , whereas $N ( v )$ denotes the static neighborhood in a single snapshot. According to Wał˛ega & Rawson (2025), two temporal graphs $T G _ { a } = ( ( G _ { 1 } ^ { a } , t _ { 1 } ^ { a } ) , \dots , ( G _ { n } ^ { a } , t _ { n } ^ { a } ) )$ and $T G _ { b } =$ $( ( G _ { 1 } ^ { b } , t _ { 1 } ^ { b } ) , \dots , ( G _ { m } ^ { b } , t _ { m } ^ { b } ) )$ are timewise isomorphic if $n =$ m, $t _ { i + 1 } ^ { a } - t _ { i } ^ { a } = t _ { i + 1 } ^ { b } - t _ { i } ^ { b }$ for all $i \in \{ 1 , \ldots , n - 1 \}$ , and there exists a bijection ϕ that is a graph isomorphism between $G _ { i } ^ { a }$ and $G _ { i } ^ { b }$ for every $i \in \{ 1 , \ldots , n \}$ . If this is the case, we write $T G _ { a } \simeq _ { \mathrm { t i m e } } T G _ { b }$ . The timestamped nodes $( v , t _ { i } ^ { a } )$ and $( u , t _ { i } ^ { b } )$ are timewise isomorphic if $\phi ( v ) = u$

Temporal GNNs. Let $\delta ( ( u , t _ { i } ) , ( v , t _ { j } ) ) : = t _ { j } - t _ { i }$ denote the real valued time gaps between timestamped nodes and write $\delta _ { i j }$ as shorthand. Then, following (Wał˛ega & Rawson, 2025), for a $T G$ as above, a general TGNN is defined as

$$
m _ { t _ { j } } ^ { ( l ) } ( v ) = \Omega _ { a } ( \{ \Phi ^ { ( l ) } ( \Xi ( h _ { \star } ^ { ( l ) } , \zeta ( \delta _ { i j } ) ) ) , ( u , t _ { i } ) \in N ( v , t _ { j } ) \} )\tag{4}
$$

whereby ${ \star } = ( u , t _ { i } )$ yields global and $\star = ( u , t _ { j } )$ yields local $\mathsf { M P } , \zeta$ denotes some mapping of the scalar $\delta _ { i j }$ to a scalar or a vector and Ξ denotes a function combining $h _ { ( u , t _ { i } ) } ^ { ( l ) }$ or $h _ { ( u , t _ { j } ) } ^ { ( l ) }$ with $\zeta ( \delta _ { i j } )$ , whereby Wał˛ega & Rawson (2025) chose concatenation ∥ for $\Xi \mathrm { a n d } \zeta ( \delta _ { i j } ) = \delta _ { i j }$ . This followed by a combinator

$$
h _ { ( v , t _ { j } ) } ^ { ( l + 1 ) } = \Gamma ^ { ( l ) } \Big ( \Omega _ { c } ( \{ \{ h _ { v , t _ { j } } ^ { ( l ) } \} \} \cup \{ \{ m _ { t _ { j } } ^ { ( l ) } ( v ) \} \} ) \Big ) .\tag{5}
$$

By Rossi et al. (2020); Xu et al. (2019), summation $\displaystyle \sum$ or concatenation ∥ are common choices for $\Omega _ { a } , \Omega _ { c }$ . Similar to RGNNs, we denote the L-layer TGNN stacking layers of the form $( 4 ) \AA - \textcircled { 5 }$ as Ψ or $\Psi _ { \Theta } ^ { ( \bar { L } ) }$ . Where ambiguous, we specify whether $\Psi _ { \Theta } ^ { ( L ) }$ denotes an RGNN or a TGNN.

Expressivity of TGNNs. To bridge the MP schemes of TGNNs with the 1-RWL characterization of expressivity, Wał˛ega & Rawson (2025) construct from $T G$ two directed, multi-relational knowledge graphs $K _ { g l o b }$ and $K _ { l o c }$ over the timestamped nodes $V ^ { \mathrm { t i m e } }$ and a finite relation set of time lags $\mathcal { R } = \{ 0 , 1 , \ldots , n - 1 \}$ , whereby the $r \in \mathcal { R }$ is obtained from the real valued time stamps via the orderpreserving bijection $\iota : \{ t _ { 1 } , \ldots , t _ { n } \} \to \{ 1 , \ldots , n \}$ with $\iota ( t _ { i } ) ~ = ~ i , ~ { \mathrm { i . e . , } } ~ r ~ = ~ \iota ( t _ { j } ) ~ - ~ \iota ( t _ { i } ) ~ = ~ j ~ - ~ i .$ Then, for $K _ { g l o b }$ , the relational edge sets for each $r \in \mathcal { R }$ can be constructed over the directional edge sets $\dot { E } ^ { g l o b } =$ $\{ ( j - i , ( v , t _ { i } ) , ( u , t _ { j } ) ) ~ | ~ i \le j , ~ \{ u , v \} ~ \in ~ E _ { i } \} ~ \subseteq ~ \mathcal { R } ~ \times ~$ $V ^ { \mathrm { t i m e } } \times V ^ { \mathrm { t i m e } } \mathrm { a s } \dot { E } _ { r } ^ { g l o b } = \{ e \in \dot { E } ^ { g l o b } \mid \dot { \tau } ^ { \star } ( e ) = r \}$ with $\dot { \tau } ^ { \star } ( ( j - i , ( v , t _ { i } ) , ( u , t _ { j } ) ) ) = r = j - i \ ( \mathrm { w i t h } \ ^ { \star }$ indicating the adaptation to the triplet domain). For $K _ { l o c } ,$ they are constructed via $\dot { E } ^ { l o c } = \{ ( j - i , ( v , t _ { j } ) , ( u , t _ { j } ) ) \mid i \leq$ $j , \ \{ u , v \} \in E _ { i } \} \subseteq { \mathcal { R } } \times V ^ { \mathrm { t i m e } } \times V ^ { \mathrm { t i m e } }$ and $\dot { E } _ { r } ^ { l o c } = \{ e \in$ $\dot { E } ^ { l o c } \mid \dot { \tau } ^ { \star } ( e ) = \stackrel { . } { r } \}$

For these $K _ { l o c }$ and $K _ { g l o b }$ , the following trivially follows from Wał˛ega & Rawson (2025):

Corollary 2.2. Let T G be a temporal graph and let $x =$ $( v , t _ { i } ) , x ^ { \prime } = ( u , t _ { j } ) \ \in \ V ^ { \mathrm { t i m e } }$ . For any $k \in \mathbb N ,$ for any local or global TGNN, and 1-RWL instantiated with the perrelation, per-direction neighborhoods $N _ { r } ^ { + } ( \cdot ) , N _ { r } ^ { - } ( \cdot )$ ofthe corresponding $K _ { \star } ( T G )$ , with $\star \in \mathrm  \{ g l o b $ , loc}, there exist parametersfor that TGNN such that

$$
c _ { K _ { \star } ( T G ) } ^ { ( k ) } ( x ) = c _ { K _ { \star } ( T G ) } ^ { ( k ) } ( x ^ { \prime } ) \Longleftrightarrow h _ { x } ^ { ( k ) } = h _ { x ^ { \prime } } ^ { ( k ) } .
$$

The proofs of the above and all subsequent theoretical claims are provided in Appendix D.

## 3. Relational Expressivity and Lottery Tickets

We now extend SELTH to RGNNs. We assume that all binary pruning masks M with sparsity ratio $\rho \in ( 0 , 1 )$ are constructed such that each entry is an independent Bernoulli random variable, and we write $M \sim ^ { i . i . d . } \mathcal { B } ( 1 - \rho )$ . Furthermore, we assume a parameter initialization $\Theta _ { 0 }$ for $\Psi ^ { ( L ) }$ where each entry is independently drawn from a continuous, bounded uniform distribution on an interval $[ c , d ] \in \mathbb { R }$ with $c < d ,$ and we write $\Theta _ { 0 } \sim \mathcal { U } _ { c } ^ { d } . \ \mathbf { A }$ sparse initialization of $\Psi ^ { ( L ) }$ is then given by the Hadamard product $\widehat { \Theta } _ { 0 } = M \odot \Theta _ { 0 }$ We also assume, for all MLPs $( \mathrm { e . g . , } \Gamma ^ { ( l ) } )$ , a class of realanalytic, injective, continuously differentiable zero-fixing activations σ with a nowhere-zero derivative. The assumptions on σ are merely technical and simplify the analysis. Extending our results to other activations is straightforward but requires case distinctions. We assume that initial node labels are encoded by an injective map Λ. Appendix H details all assumptions and limitations.

Theorem 3.1 (Relational SELTH). Let D be any finite collection of finite graphs, where each element of D is a directed node-labeled multi-relational graph $G \ =$ $( V , \dot { E } , \lambda , \dot { \tau } )$ . Let $\Psi ^ { ( L ) }$ be a corresponding sufficiently overparametrized depth-L RGNN. Then, for all $G _ { a } , G _ { b } \in \mathcal { D }$

$$
G _ { a } \not \simeq \not \simeq \not \simeq \delta _ { \mathrm { R W L } ^ { ( L ) } } G _ { b } \longleftrightarrow \widehat { \Psi } _ { \widehat { \Theta } _ { 0 } } ^ { ( L ) } ( G _ { a } ) \not = \widehat { \Psi } _ { \widehat { \Theta } _ { 0 } } ^ { ( L ) } ( G _ { b } )
$$

with a probability at least γ<sub>RGNN</sub> $> 0$

Intuitively, γ<sub>RGNN</sub> is the probability that all branch and combine MLPs remain injective on the finite input sets they witness on D. The proof of Theorem 3.1 (Appendix D) yields an explicit lower bound in terms of: the maximum number $N _ { \mathrm { m a x } }$ of distinct inputs witnessed by any $\Phi _ { r \pm } ^ { ( l ) }$ or $\Gamma ^ { ( l ) }$ , the minimum $\ell _ { 0 }$ -separation $s _ { \mathrm { m i n } }$ between such inputs, the minimum hidden width $m _ { \mathrm { m i n } }$ across these MLPs, the number of branches $| B |$ per layer (tied to the relations and directions), the MLP depth $M ,$ message-passing depth $L ,$ and the pruning probability $\rho .$ Appendix D derives γ<sub>RGNN</sub> by bounding per-block non-injectivity on D, union-bounding within each layer, and composing over L layers under independent masks. A compact (slightly looser) bound is

$$
\widetilde { \gamma } _ { \mathrm { R G N N } } \geq \bigg ( [ 1 - \binom { \widetilde { N } _ { \operatorname* { m a x } } } { 2 } \rho ^ { \widetilde { s } _ { \operatorname* { m i n } } \widetilde { m } _ { \operatorname* { m i n } } } ] + \bigg ) ^ { L ( M | \mathcal { B } | + 1 ) } ,
$$

with $\begin{array} { r l r } { [ x ] _ { + } } & { { } : = } & { \operatorname* { m a x } \{ x , 0 \} } \end{array}$ where $\widetilde { N } _ { \mathrm { m a x } } \quad : =$ max $\{ N _ { \mathrm { m a x } } , N _ { \mathrm { m a x , c o m b } } \} , \ : \widetilde { s } _ { \mathrm { m i n } } : = \mathrm { m i n } \{ s _ { \mathrm { m i n } } , s _ { \mathrm { m i n , c o m b } } \}$ and $\widetilde { m } _ { \mathrm { m i n } } \ : = \ \operatorname* { m i n } \{ m _ { \mathrm { m i n } } , m _ { \mathrm { m i n , c o m b } } \}$ aggregate worstcase input-count, separation, and width across branch and combine MLPs; the full expression is given in Appendix D. For any target $\gamma _ { t a r g e t } \in ( 0 , 1 )$ , it suffices to choose $\widetilde { m } _ { \mathrm { m i n } }$ large enough so that the (clamped) bound $( \left[ 1 ~ - ~ \left( { \widetilde { N } } _ { \mathrm { m a x } } \right) \rho ^ { { \widetilde { s } } _ { \mathrm { m i n } } { \widetilde { m } } _ { \mathrm { m i n } } } \right] _ { + } ) ^ { L ( M | { \widetilde { B } } | + 1 ) }$ exceeds γ<sub>target</sub>. Provided $1 - \binom { \widetilde { N } _ { \mathrm { m a x } } } { 2 } \rho ^ { \widetilde { s } _ { \mathrm { m i n } } \widetilde { m } _ { \mathrm { m i n } } } > 0$ , a sufficient width to certify γeRGNN $\geq \gamma _ { t a r g e t }$ is

$$
\widetilde { m } _ { \mathrm { m i n } } \ge \frac { 1 } { \widetilde { s } _ { \mathrm { m i n } } \ln \rho } \ln \left( \frac { 1 - \left( \gamma _ { t a r g e t } \right) ^ { 1 / ( L ( M | \mathcal { B } | + 1 ) ) } } { \binom { \widetilde { N } _ { \mathrm { m a x } } } { 2 } } \right) .
$$

This threshold is obtained by inverting a conservative lower bound and may be pessimistic. We therefore call $\Psi ^ { ( L ) }$ sufficiently overparameterized if its smallest hidden width $\widetilde { m } _ { \mathrm { m i n } }$ satisfies the above inequality, which certifies $\gamma _ { t a r g e t } .$

Theorem 3.1 is not a restatement of Kummer et al. (2025b)’s SELTH, which is an existence result for 1-WL bounded GNNs. Theorem 3.1 gives a probabilistic guarantee: under a random sparse initialization, directed 1-RWL distinguishability on directed multi-relational graphs is preserved with probability at least $\gamma _ { \mathrm { R G N N } } > 0$ . This requires |B| relationdirection branches plus a combine MLP and makes γ<sub>RGNN</sub> depend on worst-case input statistics across all branch and combine blocks. With $| \mathcal { R } | = 1$ and no directions, Theorem 3.1 reduces to Kummer et al. (2025b)’s SELTH.

Temporal expressivity and lottery tickets. To bring RSELTH to TGNNs, we first derive the following graphlevel statement from the node-level formulation of Corollary 2.2 to match the setup of Theorem 3.1:

Lemma 3.2. Let $T G _ { a } , T G _ { b }$ be arbitrary elements of a finite collection D oftemporal graphs and $\dot { \Psi } ^ { ( L ) }$ a depth-L local or global TGNN. Then, with $\star \in \{ g l o b , l o c \}$ being $\Psi ^ { ( L ) } { \mathbf { \bar { \rho } } } _ { S }$ MP paradigm, there exist $\Theta f o r \Psi _ { \star } ^ { ( L ) }$ such that

$$
K _ { \star } ( T G _ { a } ) \not \subsetneq _ { \mathrm { R W L } ^ { ( L ) } } K _ { \star } ( T G _ { b } ) \longleftrightarrow\tag{6}
$$

$$
\Psi _ { \Theta , \star } ^ { ( L ) } ( T G _ { a } ) \neq \Psi _ { \Theta , \star } ^ { ( L ) } ( T G _ { b } ) .\tag{7}
$$

To show that Theorem 3.1 also applies to TGNNs, we show that Lemma 3.2 also holds under pruning. To achieve this, we consider several different routes. Any global or local TGNN can be rewritten as an RGNN operating on an augmented $\smash { \widetilde { K } _ { \star } ( T G ) }$ (either encoding the real valued time gaps $\delta _ { i j }$ as edge features or using a refined relation alphabet) to which Theorem 3.1 applies. We discuss the case of a $\smash { \widetilde { K } _ { \star } ( T G ) }$ encoding $\delta _ { i j }$ as edge features here and the alternative route along its shortcomings in Appendix A.

To this purpose, we construct an RGNN operating on an augmented $\smash { \widetilde { K } _ { \star } ( T G ) }$ that is algebraically identical to the corresponding TGNN operating on TG: Fix $\begin{array} { r l } { \star } & { { } \in } \end{array}$ {glob, loc} and a TGNN layer as in Eqs. $( 4 ) – ( 5 )$ with aggregation/combination $\Omega _ { a } , \Omega _ { c } ,$ , message map $\Phi ^ { ( l ) }$ , combiner $\Gamma ^ { ( l ) }$ , gap encoder $\zeta ,$ edge-time gaps $\delta ( y , x )$ and a corresponding $K _ { \star } ( T G )$ . We can realize the same update with an RGNN on $\smash { \widetilde { K } _ { \star } ( T G ) }$ by attaching to each directed edge $e \ = \ ( y  x )$ the edge feature $\xi _ { e } \ : = \ \zeta ( \delta ( y , x ) )$ and using a shared message map $\Phi _ { r , \pm } ^ { ( l ) } \equiv \Phi ^ { ( l ) }$ , writing $\widetilde { \Phi } ^ { ( l ) } ( h , \xi ) : = \Phi ^ { ( l ) } ( \Xi ( h , \xi ) ) $ , for an appropriate $\Xi \left( \mathrm { e . g . } \right.$ ∥). With the same $\Omega _ { a } , \Omega _ { c } , \Gamma ^ { ( l ) }$ as in the TGNN, the RGNN update at x becomes

$$
\begin{array} { r l } & { m _ { r , \pm } ( x ) = \Omega _ { a } ( \{ \widetilde { \Phi } ^ { ( l ) } ( h _ { y } ^ { ( l ) } , \xi _ { ( y  x ) } ) \mid y \in N _ { r } ^ { \pm } ( x ) \} ] , } \\ &  \quad h _ { x } ^ { ( l + 1 ) } = \Gamma ^ { ( l ) } \Big ( \Omega _ { c } ( \{ \ S h _ { x } ^ { ( l ) } \} ) \ { \stackrel {  } {  } } \ \underbrace { \lfloor \mathbf { f } \{ m _ { r , \pm } ( x ) \} \} ) , } \end{array}
$$

which by construction is algebraically identical to Eqs. (4)- (5) for the chosen (global/local) neighborhoods:

Lemma 3.3. Fix $\star \in \{ \mathrm { g l o b } , \mathrm { l o c } \}$ and a TGNN layer with maps $( \Omega _ { a } , \Omega _ { c } , \Phi ^ { ( l ) } , \Gamma ^ { ( \tilde { l } ) } , \zeta , \Xi )$ . Instantiate an RGNN on $\smash { \widetilde { K } _ { \star } ( T G ) }$ with $\widetilde { \Phi } ^ { ( l ) } ( h , \xi ) : = \Phi ^ { ( l ) } ( \Xi ( h , \xi ) )$ , using the same $( \Omega _ { a } , \Omega _ { c } , \Gamma ^ { ( l ) } )$ and strict weight tying $\Phi _ { r , \pm } ^ { ( l ) } \equiv \Phi ^ { ( l ) }$ across $( r , \pm )$ . Then for every $T G$ , all $x \in V ^ { \mathrm { t i m e } }$ and all $l \geq 0$

$$
h _ { x } ^ { ( l ) } \left( T G N N o n T G \right) = \widetilde { h } _ { x } ^ { ( l ) } \left( R G N N o n \widetilde { K } _ { \star } ( T G ) \right) .
$$

Together with Corollary 2.2, Lemma 3.3 implies that the RGNN operating on $\smash { \widetilde { K } _ { \star } ( T G ) }$ has exactly the same nodelevel expressivity as 1-RWL operating $K _ { \star } ( T G )$ , which transfers to the graph level by the same argument as in Lemma 3.2. As D is finite and each $T G \in { \mathcal { D } }$ is finite, only finitely many time gaps $\delta$ appear as edge features on $\tilde { K } _ { \star } ( T G )$ . Hence Theorem 3.1 applies verbatim to this RGNN via the same finite-domain injectivity argument. Moreover, as we tie parameters $( \boldsymbol { \Phi } _ { r , \pm } ^ { ( l ) } \equiv \boldsymbol { \Phi } ^ { ( \bar { l } ) } )$ and choose $\Omega _ { a } , \Omega _ { c } , \Gamma ^ { ( l ) }$ to match the TGNN exactly, parameter matrices correspond one-to-one and pruning masks transfer verbatim from the RGNN back to the TGNN. Consequently, RSELTH, up to minor modifications, also holds for sufficiently overparameterized global and local TGNNs. We state the corresponding TGNN reformulation of Theorem 3.1 in Appendix B, and, for completeness, prove it directly using Eqs. (4)-(5). Moreover, Appendix C extends the result to the node level, matching the node-level focus of Wał˛ega & Rawson (2025).

We now relate Theorem 3.1 applied to TGNNs to (Wał˛ega & Rawson, 2025)’s notion of temporal isomorphism, which is taken to the graph level by the following Lemma.

Lemma 3.4. Let $T G _ { a } \ = \ ( ( G _ { 1 } ^ { a } , t _ { 1 } ^ { a } ) , \dots , ( G _ { n } ^ { a } , t _ { n } ^ { a } ) )$ and $T G _ { b } = ( ( G _ { 1 } ^ { b } , t _ { 1 } ^ { b } ) , \dots , ( G _ { n } ^ { b } , t _ { n } ^ { b } ) )$ be temporal graphs. Fix $L \in \mathbb { N }$ and ⋆ ∈ {glob, loc}. Thenfor sufficient L:

$$
T G _ { a } \simeq _ { \mathrm { t i m e } } T G _ { b } \Longrightarrow K _ { \star } ( T G _ { a } ) \simeq _ { \mathrm { R W L } ^ { ( L ) } } K _ { \star } ( T G _ { b } ) .
$$

Therefore, combining Lemma 3.4 with Theorem 3.1 via Lemma 3.3, if a sparse parameterization of a (local or global) TGNN distinguishes two temporal graphs, then those graphs are not timewise isomorphic. This links RSELTH to other notions of isomorphism of temporal graphs. By Wał˛ega & Rawson (2025), (Beddar-Wiesing et al., 2024)’s pointwise isomorphism is less strict than timewise isomorphism. Heeg et al. (2025)’s notion is also more relaxed than timewise isomorphism: they prove it lies between time-aggregated and time-concatenated isomorphism, with the latter being equivalent to the notions proposed by Gao & Ribeiro (2022); Souza et al. (2022). For unlabeled snapshot graphs, Wał˛ega & Rawson (2025)’s timewise isomorphism coincides with time-concatenated isomorphism, hence any timewise-isomorphic pair is consistent event-graph isomorphic (equivalently, their augmented event graphs are isomorphic). The converse does not hold in general; equivalence with timewise isomorphism holds only under constraints (Heeg et al., 2025).

Cross-graph MP (XIMP/HIMP). Let $G = ( V , \dot { E } , \lambda , \dot { \tau } )$ be a directed, node-labeled, (multi-)relational graph $( \mathrm { e . g . , a }$ molecular graph) and let $\{ T _ { i } = ( V _ { i } , \dot { E } _ { i } , \lambda _ { i } , \dot { \tau } _ { i } ) \} _ { i = 1 } ^ { n }$ be abstractions of G (e.g., junction tree (Jin et al., 2018), extended reduced graph (Stiefl et al., 2006)). Analogous to Ehrlich et al. (2026) (XIMP) and Fey et al. (2020) (HIMP), we encode atom-abstraction incidence by an assignment matrix $S _ { i } ~ \in ~ \{ 0 , 1 \} ^ { | V | \times | V _ { i } | }$ , where $S _ { i } [ v , u ] = 1$ iff $v \in V$ contributes to $u \in V _ { i }$ . XIMP takes G and $\{ T _ { i } , S _ { i } \}$ as input and computes $\Psi _ { \Theta } ^ { ( L ) } ( G , \{ T _ { i } , S _ { i } \} )$ to embed G. It performs MP on $G$ and each $T _ { i }$ and two-way inter-graph MP both indirectly $( \mathrm { I } ^ { 2 } \mathrm { M P } , G  T _ { i } )$ via $S _ { i }$ and directly (DIMP, $T _ { i }  T _ { k } )$ via the on-the-fly computed abstraction-abstraction incidence matrix $\bar { S _ { i k } ^ { - } } ~ \in ~ \bar { 0 , } 1 ^ { | V _ { i } | \times | V _ { k } | }$ , where $S _ { i k } [ u , w ] = 1 $ iff there exists $v ~ \in ~ V$ with $S _ { i } [ v , u ] \ = \ S _ { k } [ v , w ] \ = \ 1$ While I<sup>2</sup>MP and DIMP both employ a single-layer postaggregation transformation (a linear map followed by a pointwise nonlinearity), XIMP and HIMP can employ arbitrary standard GNNs for local MP on G and $T _ { i }$ . Moreover, XIMP uses mean aggregation and HIMP uses concatenation or sum. Hence, to standardize the analysis, we adopt sum aggregation to leverage its multiset discriminative power (Xu et al., 2019) and assume a single-layer post-aggregation transformation for I<sup>2</sup>MP, DIMP and MP.

We define the compound node set ${ \widetilde { V } } : = V \ \not  \ \not \in \Sigma _ { i = 1 } ^ { n } V _ { i }$ and construct a directed multi-relational compound graph $G ^ { \oplus } =$ $( \widetilde { V } , \dot { E } ^ { \oplus } , \widetilde { \lambda } , \dot { \tau } ^ { \oplus } )$ by taking the disjoint union of all intragraph edges and adding typed cross-graph edges: (i) intra edges $\dot { E }$ on $V$ and $\dot { E } _ { i }$ on $V _ { i }$ retain their original relation labels; (ii) for each i and each $( v , u )$ with $S _ { i } [ v , u ] = 1$ , add $v  u$ with relation type $\alpha _ { i }$ and $u \to v$ with type $\bar { \alpha } _ { i } ;$ (iii) for $i \neq k$ , add $u \in V _ { i } \to w \in V _ { k }$ with type $\beta _ { i k }$ whenever there exists $v \in V$ such that $S _ { i } [ v , u ] = S _ { k } [ v , w ] = 1$ and the corresponding reverse edge with type $\beta _ { k i }$ . Up to notation, our definition is the same as Ehrlich et al. (2026), except that edge types are made explicit. As XIMP applies non-linear transformation post-aggregation, we initialize embeddings by $\widetilde { h } _ { x } ^ { ( 0 ) } = \Lambda ( \widetilde { \lambda } ( x ) )$ for $x \in \widetilde { V }$ with an injective Λ as in the relational case (Section 3) but with the additional constraint that $\Lambda ( \mathcal { X } ) \subset \mathbb { R } ^ { d }$ is linearly independent (Remark 2.1). A single cross-graph layer on $G ^ { \oplus }$ is then a relational update as in Eqs. (1)-(3), over $\mathcal { R } ^ { \oplus }$ , the union of all intra-graph relation types and the cross-graph types $\alpha _ { i } , \bar { \alpha } _ { i } , \beta _ { i k } , \bar { \beta } _ { k i }$ . By tying parameters per relation type (i.e., assigning linear $\dot { \Phi } _ { r } ^ { ( l ) }$ to all cross-graph types and one parameter tied $\Phi _ { r } ^ { ( l ) }$ to each set of intra-graph relations per $G$ and $T _ { i } )$ and using $\begin{array} { r } { \Omega _ { a } = \sigma \circ \sum , \Omega _ { c } = \sum } \end{array}$ with $\Gamma ^ { ( l ) } = \mathrm { i d }$ , this single layer is, except DIMP’s normalization, by construction algebraically identical to XIMP’s combination of MP with I<sup>2</sup>MP and DIMP. As DIMP’s normalization would not change the 1- RWL bound on $G ^ { \oplus }$ (Section 2), we omit it in the analysis.

Corollary 3.5. Let D be a finite collection of inputs $( G , \{ T _ { i } , S _ { i } \} _ { i = 1 } ^ { n } )$ and $G ^ { \oplus }$ be constructed as abovefor each element of D. Then there exists a parameterization of an RGNN on $G ^ { \oplus }$ such that,for all pairs in D and all $L \in \mathbb { N } ,$

$$
\begin{array} { r l } & { G _ { a } ^ { \oplus } \operatorname { \mathcal { H } } _ { \mathrm { R W L } ^ { ( L ) } } G _ { b } ^ { \oplus } \iff } \\ & { \Psi _ { \Theta } ^ { ( L ) } ( G _ { a } , \{ T _ { i } ^ { a } , S _ { i } ^ { a } \} ) \neq \Psi _ { \Theta } ^ { ( L ) } ( G _ { b } , \{ T _ { i } ^ { b } , S _ { i } ^ { b } \} ) . } \end{array}\tag{8}
$$

As the reduction to a relational layer on the compound graph $G ^ { \oplus }$ yields a one-to-one correspondence with XIMP’s parameters by construction, the pruning masks align. It suffices to directly apply Theorem 3.1 to the corresponding RGNN operating on $\{ G ^ { \oplus } \}$ : sparse subnetworks that preserve 1-RWL distinguishing power on $G ^ { \oplus }$ transfer to XIMP (and hence HIMP, as setting $n { = } 1$ with $T _ { 1 }$ the junction tree recovers HIMP’s inter-MP), yielding SELTH for cross-graph MP.

![](images/f61ee79c42496fad291225ad343f654d6a41bb8153714f1f31b3a9fbd9afa9f0.jpg)

![](images/23f7556307624331ec9a19fb20553894718068a57b0cf685eebc002609504d6b.jpg)  
Figure 1. (a) Landscape of γ<sub>RGNN</sub> over $N _ { \mathrm { m a x } }$ (log-scale, x) as task complexity proxy and width $m _ { \mathrm { { m i n } } } \mathrm { { ( y ) } }$ at fixed sparsity $\rho = 0 . 7 0$ Black iso-curves show $e f f i c i e n c y = \gamma _ { \mathrm { R G N N } } / ( \mathrm { M A D D s } \cdot 1 0 ^ { - 6 } )$ . (b) γ<sub>RGNN</sub> versus $s _ { \mathrm { m i n } }$ (x) and m<sub>min</sub> (y) at $\rho = 0 . 7 0 ;$ contours at $\gamma _ { \mathrm { R G N N } } = 0 . 9 9$ and $1 0 ^ { - 9 }$ illustrate that increasing $s _ { \mathrm { m i n } }$ expands the high-γ region, reducing the width needed to hit the target. (c) MADDs cost landscape over $( \rho , m _ { \mathrm { m i n } } )$ with $\gamma _ { \mathrm { R G N N } } = 0 . 9 9$ and $1 0 ^ { - 9 }$ frontiers overlaid, showing that achieving γ ≈ 1 typically requires either lower pruning (smaller ρ) or larger width. (d) γ<sub>RGNN</sub> = 0.99 frontiers in the $( \rho , m _ { \mathrm { m i n } } )$ plane for multiple $N _ { \mathrm { m a x } }$ (color) and branch counts $\boldsymbol { B } ^ { \prime } = | \boldsymbol { B } |$ (line style): larger $N _ { \mathrm { m a x } }$ or B shifts the frontier toward denser or wider regimes. Unless varied per panel: $L = 5 ,$ $B = 4$ (varied only in d), $M = 2 ,$ $s _ { \operatorname* { m i n } } = 2 ,$ and $N _ { m a x } = 1 0 0 0$

Optimization of expressive lottery tickets. The optimization of sparse RGNNs is subtle, so we defer technical details to Appendix E and summarize the main results here. We call a sparse initialization with mask M optimizable on a finite dataset D if, when the output loss gradient is nonzero at least for one element of D, every surviving parameter is gradient-connected to the loss and receives a nonzero update at least once over D. We show for RGNNs that irreducible task-expressive masks (minimal masks whose surviving parameters suffice to distinguish exactly those non-isomorphic graphs in D that carry different labels) are optimizable in this sense. In contrast, arbitrarily sparsified RGNNs at the same sparsity need not be optimizable: they may both fail to realize label-relevant distinctions and allocate disconnected parameters receiving zero gradients. Finally, we prove that optimizability under irreducible task-expressive masks persists under standard small-step first-order updates for sufficiently small step sizes.

## 4. Numerical Illustrations and Empirics

Our evaluation has three parts: (i) we instantiate Theorem 3.1’s bound γ<sub>RGNN</sub> to visualize width–sparsity– compute trade-offs as its governing quantities vary (Figure 1); (ii) on a synthetic multi-relational dataset, we instantiate γ<sub>RGNN</sub> using architecture/dataset-specific input statistics, compare it to an empirical mask success rate γ<sub>empirical</sub>, and relate pre-training graph separability (i.e., whether observed graphs receive distinct embeddings) as empirical expressivity proxy to gradient flow and training loss/accuracy under fixed masks as a proxies for optimizability (Figures 2, 3 and 4); (iii) on real temporal and molecular benchmarks, we examine how pre-training separability of pruned models correlates with compression across pruning rates and predicts test performance (Figure 5).

Setup. We instantiate Theorem 3.1’s $\gamma _ { \mathrm { R G N N } }$ to visualize how expressivity and sparsity interact with width and computational cost under varying $N _ { \mathrm { m a x } } , s _ { \mathrm { m i n } } , m _ { \mathrm { m i n } } .$ , |B|, M, L and $\rho$ in Figure 1. We treat γ<sub>RGNN</sub> as a worst-case probability that a random mask preserves the L-round 1-RWL expressivity on a finite dataset, not as a predictor of downstream prediction quality. We focus on the RGNN of Section 2 and estimate the multiply-adds operations (MADDs) of an MLP similar to Kummer et al. (2023); see Appendix F.

Figure 2(a) illustrates the tightness of γ<sub>RGNN</sub>. We first generate a synthetic dataset $\mathcal { D } _ { { s y n } }$ of 30 multi-relational graphs. Then, we randomly initialize and prune a simple RGNN using $\displaystyle \sum$ for $\Omega _ { a } , \Omega _ { c }$ and readout. Next, we compute graph embeddings over $\mathcal { D } _ { s y n }$ . A pruning mask at sparsity $\rho$ is counted as successful iff the resulting embeddings are pairwise distinct across all graphs. Sampling 100 masks for each $\rho$ on a uniform grid of 50 values in [0.01, 0.99] yields an empirical success rate $\gamma _ { \mathrm { e m p i r i c a l } }$ , which we plot against the mean $\overline { { \gamma } } _ { \mathrm { R G N N } }$ obtained by instantiating γ<sub>RGNN</sub> on the configuration-specific input statistics of each sample (details Appendix G). Moreover, Figure 2(b) shows how a simple pre-training separability proxy is associated with optimization under pruning. For each $\rho ,$ we sample 20 masks, apply each mask to a freshly initialized RGNN with a linear classifier, and evaluate thefraction ofseparable graphs before training, i.e., the number of distinct embeddings divided by $| \mathcal { D } _ { s y n } |$ . We then train the model for 10 epochs on a graph-ID classification task over $\mathcal { D } _ { s y n } .$ , masking both weights and gradients. We record (i) the final training loss, (ii) the final training accuracy, and (iii) the global gradient-flow (Evci et al., 2022) given by the global $\ell _ { 2 }$ -norm of the gradients, averaged over epochs. We average metrics across trials and report Spearman correlations between pre-training separability and post-training metrics. While Figure 2(b) illustrates a general trend that separability is associated with optimization under pruning over a large interval of pruning rates, it does not fully disentangle expressivity from model size. As subnetworks with lower sparsity naturally retain more parameters, it remains unclear whether the observed improvement in gradient flow, loss, or accuracy comes from preserving task-relevant expressivity or simply from having larger subnetworks. Figure 3 hence shows a fixed-sparsity control experiment: using the same RGNN configuration and $\mathcal { D } _ { s y n } .$ for four sparsity levels $\rho \in \{ 0 . 9 1 2 5 , 0 . 9 2 5 , 0 . 9 3 7 5 , 0 . 9 5 \}$ we sampled 100 random masks with identical parameter counts and compared optimization behavior across different pre-training separability levels, using 30 training epochs.

![](images/d05b06cf67e68a41f49545b69b219a5944964186eb94d8777b1eecef6cd7c263.jpg)

(b)  
![](images/730535b2084be18a7dd96002927f997cbdaf70ce669d40db1adef13934acc8fa.jpg)  
Figure 2. (a) γ<sub>empirical</sub> (green) vs. mean $\overline { { \gamma } } _ { \mathrm { R G N N } }$ (red) for randomly sparsely initialized untrained simple RGNNs on $\mathcal { D } _ { s y n }$ . Orange segments visualize $\Delta _ { \gamma } = \gamma _ { \mathrm { e m p i r i c a l } } - \overline { { \gamma } } _ { \mathrm { R G N N } } .$ (b) Spearman correlation between the fraction of separable graphs before training and (i) mean gradient flow, (ii) final training loss, and (iii) final training accuracy. For $\rho ,$ we average results across trials and compute Spearman’s $\rho _ { s m }$ and its p-value; points plot $( \rho _ { s m } , - \log _ { 1 0 } p )$ with the dashed lines indicating $p = 0 . 0 5$ and $\rho _ { s m } = 0 $

To complement Figure 2(a), in Figure 4, we perform a per-factor study of bound tightness using the same synthetic multi-relational graph generator and the same simple, randomly initialized, untrained RGNN as above. Unless varied, we fix the common hidden width and remaining dataset/model parameters to $( m , | \mathcal { D } _ { \mathrm { s y n } } | , n , L , | \mathcal { R } | ) =$ (16, 30, 6, 3, 3), and sweep $n , n , | D _ { \mathrm { s y n } } | \in \{ 4 , 8 , 1 6 , 3 2 , 6 4 \}$ , and $L , | \mathcal { R } | \in \{ 2 , 4 , 6 , 8 , 1 0 \}$ one at a time. For each configuration, we evaluate 50 uniformly spaced sparsity levels $\rho \in$ [0.01, 0.99]. At each $\rho ,$ we estimate γ<sub>empirical</sub> and γ<sub>RGNN</sub> from 10 independently sampled pruning masks and associated statistics, respectively. We summarize tightness by the integrated absolute gap R|γ<sub>empirical</sub> $( \rho ) - \overline { { \gamma } } _ { \mathrm { R G N N } } ( \rho )$ | dρ ≈ $\Sigma _ { \rho } \Delta _ { \gamma } ( \rho )$ (see Figure 2(a)), so smaller values indicate closer agreement between γ<sub>RGNN</sub> and γ<sub>empirical</sub>.

Figure 5(a) relates compression ratio, test performance and separability ratio of the untrained, pruned model, measured (i) at the node level after the final message-passing step for local and global TGNNs on the node-classification benchmark tgbn-trade (Huang et al., 2023a), and (ii) at the graph level for XIMP on one polaris ADMET (HLM) and potency (pIC50 MERS-CoV Mpro) regression benchmark (ASAP Discovery x OpenADMET, 2025a;b) each. Motivated by Figure 2(a), we sweep pruning ratios ρ over 200 uniformly spaced values in [0.0, 0.995], with 5 samples per $\rho .$ We train XIMP for 50 epochs and the TGNNs for 10 epochs, using width 16 for all learnable transformations. Figure 5(b) shows the same setup but from a probabilistic perspective. We estimate, for each model-dataset pair, the probability that a pruned model achieves test performance within a small margin of its dense counterpart, conditioned on its pre-training separability. Separability values are grouped into 25 bins, and for each bin we report P(WT | Sep.), the probability of a model being a winning ticket (WT) <sup>1</sup>.

![](images/df79e0d3f5c59088e84d09584d941650941c8dff57f695455ba08fde8d251f8d.jpg)  
Figure 3. Spearman correlation between the fraction of separable graphs before training and post-training metrics under fixed sparsity (simple $\mathbf { R G N N } , \mathcal { D } _ { \mathrm { s y n } } )$ . For each $\rho ,$ we compute Spearman’s $\rho _ { s m }$ (and p-value) across masks between pre-training separability and (i) mean gradient flow, (ii) final training loss, and (iii) final training accuracy. Points plot $( \rho _ { s m } , - \log _ { 1 0 } p )$ , dashed lines indicate $\rho _ { s m } = 0$ and $p = 0 . 0 5$

Results and practical implications. The numerical behavior of γ<sub>RGNN</sub> suggests practical design rules for sparse RGNN initializations. Figure 1(d) shows that as $N _ { \mathrm { m a x } }$ and B increase, the width–sparsity trade-off shifts, so larger and more complex benchmarks typically tolerate less aggressive pruning (or require wider layers) to maintain a fixed γ . Moreover, Figure 1(b) shows that increases in $s _ { \mathrm { m i n } }$ (which could be achieved via input encodings (Kummer et al., 2025a) or by avoiding pruning schemes (Deng et al., 2020) inherently zeroing activations) can increase the sparsity attainable at a given width for the same target γ<sub>RGNN</sub>. This also suggests a link to over-smoothing (Chuang et al., 2025; Rusch et al., 2023), which can similarly reduce input separability. The γ<sub>RGNN</sub> contours in Figure 1(c) highlight a width–pruning trade-off: along a fixed target level (e.g., γ<sub>RGNN</sub> ≈ 0.99), width can be traded for sparsity up to moderate $\rho$ while keeping the approximate MADDs budget roughly constant. At high sparsity, however, the width required to maintain the target makes cost grow exponentially. Consistently, Figure 1(a) shows diminishing cost-efficiency from further widening at fixed benchmark complexity and sparsity as γ<sub>RGNN</sub> → 1. Figures 1(a)–(c) show that the band between $\gamma _ { \mathrm { R G N N } } ~ = ~ 1 0 ^ { - 9 }$ and $\gamma _ { \mathrm { R G N N } } = 0 . 9 9$ is narrow, suggesting that most configurations are either severely underparameterized or inefficiently overparameterized. Only a small region supports a cost-effective, highly expressive subnetwork, where careful tuning of width and sparsity relative to dataset and architectural properties affects γ<sub>RGNN</sub>.

Figure 2(a) compares mean γ<sub>RGNN</sub> to γ<sub>empirical</sub>. The bound is tight when expressivity holds with high probability (roughly $\rho \lesssim 0 . 6 5$ here), becomes more conservative in the transition region where expressivity fails (approx.

![](images/717abf3456f62116ab019f1f76790076c5c026fac1d3b75e580e3310e41fa60e.jpg)  
Figure 4. Tightness of $\gamma _ { \mathrm { R G N N } }$ as model and dataset characteristics vary. Each panel reports the absolute area between $\gamma _ { \mathrm { e m p i r i c a l } }$ and mean γ over a sparsity sweep, compare Figure 2 (a), so smaller area indicates a tighter bound. Panels vary (a) width m, (b) number of graphs $| \mathcal { D } _ { \mathrm { s y n } } |$ , (c) number of nodes per graph n, (d) number of message-passing layers L, and (e) number of relations |R|. For each ρ, γ<sub>empirical</sub> is estimated from 10 pruning-mask trials, while γ<sub>RGNN</sub> is instantiated using model-specific statistics and averaged over 10 sparse-mask samples. Unless varied per panel: $m = 1 6 , | \mathcal { D } _ { \mathrm { s y n } } | = 3 0$ , 6 nodes per graph, $L = 3 ,$ , and $| \mathcal { R } | = 3$

![](images/9752c023ba2f17a49803495fe4609515c36e4320d4e84c069c5f9fa810a267e3.jpg)  
Figure 5. (a) For each model–dataset variant, we compute Spearman’s $\rho _ { s m }$ (and p-value) across runs between (i) separability (Sep.) and normalized test performance (Perf.), (ii) Sep. and compression (Compr.), and (iii) Perf. and Compr.; points plot $( \rho _ { s m } , - \log _ { 1 0 } p )$ (log-y), with colors encoding the variable pair and markers the model–dataset group; dashed lines indicate $\rho _ { s m } = 0 \mathrm { a n d } p = 0 . 0 5$ (b) P(WT | Sep.) for pruned runs, shown per model–dataset variant over separability bins (midpoints on x).

$\rho \in [ 0 . 6 5 , 0 . 9 5 ] )$ , and tightens again as success vanishes for $\rho  1$ . The observed variations of $\overline { { \gamma } } _ { \mathrm { R G N N } }$ in Figure $2 ( \mathrm { a } )$ reflect the dependency of γ on statistics obtained from concrete model and mask instantiations. This matches the standard deviation (0.18) of the empirical fraction of distinguishable graphs over $\rho \in [ 0 . 6 5 , 0 . 9 5 ]$ ]. Figure 4 studies bound tightness through the integrating the gap $\Delta _ { \gamma }$ . The clearest effect is width: as m increases, the gap decreases, indicating that once the model is sufficiently overparameterized, the lower bound tracks the empirical success curve more closely. Increasing $| \mathcal { D } _ { \mathrm { s y n } } |$ has the opposite effect and makes the guarantee more conservative, which is consistent with the need to control more witnessed inputs and distinctions on the finite $\mathcal { D } _ { \mathrm { s y n } }$ . The trends with n and |R| require more caution. In our setup, varying n and |R| does not only change a single quantity in the bound; it also changes the synthetic graphs generated (see Appendix G), and, therefore, the witnessed statistics used to instantiate γ<sub>RGNN</sub>. Hence, a smaller gap in panels (c) and (e) should not be read as saying that larger graphs or more relations intrinsically improve the lower bound itself. Rather, these changes move the sparsity regimes in which both $\gamma _ { \mathrm { e m p } }$ <sub>.</sub> and $\overline { { \gamma } } _ { \mathrm { R G N N } }$ curves are closer (compare Figure 2(a)). Thus, Figure 4 complements rather than contradicts Figure 1: Figure 1 analyzes how the bound varies as a function of its governing quantities, whereas Figure 4 measures how well the instantiated bound matches empirical behavior for a specific construction.

Figure 2(b) shows that higher pre-training separability is associated with stronger gradient flow, lower final training loss, and higher final training accuracy. Figure 3 strengthens this conclusion. Even at fixed sparsity, higher pre-training separability remains significantly associated with stronger gradient flow and lower final loss across all tested sparsity levels, and with higher accuracy at the higher sparsity levels. Thus, the optimization trend in Figure 2(b) is not merely a byproduct of larger subnetworks, but persists in a within-sparsity comparison as well. This matches Section 3’s optimization discussion: masks that preserve more label-relevant distinctions at initialization retain more gradient-connected paths, so loss signals reach more surviving weights. Moreover, this aligns with the hypothesis that more expressive sparse RGNN initializations are easier to optimize, which is consistent with Kummer et al. (2025b)’s empirical findings and theoretical Gradient Diversity (Yin et al., 2018) analysis for classic GNNs. Figure 5 further supports this on real-world benchmarks and practical architectures: pre-training separability correlates with and predicts test-set performance and is anticorrelated with pruning compression. In practice, this highlights the importance of expressivity-aware sparsification for finding winning lottery tickets.

## 5. Conclusion

We introduce RSELTH, proving that sufficiently wide RGNNs contain sparse subnetworks that match 1-RWL expressivity on any finite dataset and deriving an explicit lower bound γ<sub>RGNN</sub> on the probability that a random pruning mask yields such a subnetwork. We show the theory extends to TGNNs and XIMP/HIMP-type architectures. Our results indicate that (i) γ <sub>G</sub> is tight in the high- and low-success regimes and becomes most conservative near the critical sparsity threshold where expressivity breaks down, (ii) only a narrow band of width–sparsity configurations yields costefficient, highly expressive sparse initializations, (iii) the attainable expressivity at fixed width and sparsity depends strongly on dataset complexity and (iv) more expressive sparse initializations can be easier to optimize. Together, our results position RSELTH as a unifying perspective on sparse, expressive RGNNs and TGNNs and beyond.

## Acknowledgements

This work was supported in part by the Vienna Science and Technology Fund (WWTF) and the City of Vienna through grants [10.47379/VRG19009] and [10.47379/ICT22059].

## Impact Statement

This work advances the theoretical understanding of how parameter sparsity interacts with expressivity in relational and temporal graph neural networks. By clarifying when random pruning can preserve discriminative power, the results may support the design of more compute- and memory-efficient graph models, which can reduce deployment costs and energy use in applications where graph learning is already standard.

The methods studied here are general and could be applied to sensitive relational or temporal data (e.g., social, financial, or behavioral graphs). Improved efficiency can lower the barrier to deploying such models at scale, which may amplify risks around privacy, profiling, or biased decision-making if used without appropriate safeguards.

Our contributions do not introduce new data collection or new capabilities targeted at a specific domain, and they do not address fairness, privacy, or security directly. Practitioners should treat the theory as one component in a broader responsible pipeline, including dataset governance, privacy-preserving training where appropriate, and domain-specific evaluations for bias and misuse.

## References

Amir, T., Gortler, S., Avni, I., Ravina, R., and Dym, N. Neural injective functions for multisets, measures and graphs via a finite witness theorem. In Advances in Neural Information Processing Systems (NeurIPS), pp. 42516–42551, 2023.

ASAP Discovery x OpenADMET. Antiviral ADMET 2025. Data set, 2025a. URL https://polarishub. io/competitions/asap-discovery/ antiviral-admet-2025. License: CC0-1.0.

ASAP Discovery x OpenADMET. Antiviral potency 2025. Data set, 2025b. URL https://polarishub. io/competitions/asap-discovery/ antiviral-potency-2025. License: CC-BY-4.0.

Barceló, P., Galkin, M., Morris, C., and Orth, M. R. Weisfeiler and Leman go relational. In Rieck, B. and Pascanu, R. (eds.), Proceedings of the First Learning on Graphs Conference, volume 198 of Proceedings of Machine Learning Research, pp. 46:1–46:26. PMLR, 09–12 Dec 2022. URL https://proceedings.mlr.press/ v198/barcelo22a.html.

Beddar-Wiesing, S., D’Inverno, G. A., Graziani, C., Lachi, V., Moallemy-Oureh, A., Scarselli, F., and Thomas, J. M. Weisfeiler-Lehman goes dynamic: An analysis of the expressive power of graph neural networks for attributed and dynamic graphs. Neural Networks, 173:106213, 2024.

Chen, T., Sui, Y., Chen, X., Zhang, A., and Wang, Z. A unified lottery ticket hypothesis for graph neural networks. In International Conference on Machine Learning (ICML), pp. 1695–1706, 2021.

Cheung, M. and Moura, J. M. Graph neural networks for covid-19 drug discovery. In IEEE International Conference on Big Data, pp. 5646–5648, 2020.

Chuang, Y.-N., Lai, K.-H., Tang, R., Du, M., Chang, C.-Y., Zou, N., and Hu, X. Fair-rgnn: Mitigating relational bias on knowledge graphs. ACM Trans. Knowl. Discov. Data, 19(2), February 2025. ISSN 1556-4681. doi: 10.1145/3681792. URL https: //doi.org/10.1145/3681792.

da Cunha, A., Natale, E., and Viennot, L. Proving the strong lottery ticket hypothesis for convolutional neural networks. In International Conference on Learning Representations (ICLR), 2022.

Deng, L., Li, G., Han, S., Shi, L., and Xie, Y. Model compression and hardware acceleration for neural networks: A comprehensive survey. Proceedings ofthe IEEE, 108(4):485–532, 2020.

Ehrlich, A., Kummer, L., Voracek, V., Bause, F., and Kriege, N. M. XIMP: Cross graph inter-message passing for molecular property prediction. CoRR, abs/2601.19037, 2026.

Evci, U., Ioannou, Y., Keskin, C., and Dauphin, Y. Gradient flow in sparse neural networks and how lottery tickets win. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 36, pp. 6577–6586, 2022.

Fey, M., Yuen, J.-G., and Weichert, F. Hierarchical inter-message passing for learning on molecular graphs. In ICML Graph Representation Learning and Beyond (GRL+) Workshop, 2020.

Frankle, J. and Carbin, M. The lottery ticket hypothesis: Finding sparse, trainable neural networks. In International Conference on Learning Representations (ICLR), 2019.

Frankle, J., Dziugaite, G. K., Roy, D. M., and Carbin, M. Stabilizing the lottery ticket hypothesis. CoRR, abs/1903.01611, 2019.

Gao, J. and Ribeiro, B. On the equivalence between temporal and static equivariant graph representations. In International Conference on Machine Learning (ICML), pp. 7052–7076. PMLR, 2022.

Gao, J., Lyu, T., Xiong, F., Wang, J., Ke, W., and Li, Z. Predicting the survival of cancer patients with multimodal graph neural network. IEEE/ACM Transactions on Computational Biology and Bioinformatics, 19(2):699–709, 2022.

Heeg, F., Sauer, J., Mutzel, P., and Scholtes, I. Weisfeiler and Leman follow the arrow of time: Expressive power of message passing in temporal event graphs. arXiv preprint arXiv:2505.24438, 2025.

Huang, S., Poursafaei, F., Danovitch, J., Fey, M., Hu, W., Rossi, E., Leskovec, J., Bronstein, M., Rabusseau, G., and Rabbany, R. Temporal graph benchmark for machine learning on temporal graphs. Advances in Neural Information Processing Systems (NeurIPS), 36:2056–2073, 2023a.

Huang, X., Orth, M. R., Ceylan, <sup>˙</sup>I. <sup>˙</sup>I., and Barceló, P. A theory of link prediction via relational weisfeiler-Leman on knowledge graphs. In Advances in Neural Information Processing Systems (NeurIPS), 2023b. URL https://openreview. net/forum?id=7hLlZNrkt5.

Hui, B., Yan, D., Ma, X., and Ku, W.-S. Rethinking graph lottery tickets: Graph sparsity matters. In International Conference on Learning Representations (ICLR), 2023.

Jin, W., Barzilay, R., and Jaakkola, T. Junction tree variational autoencoder for molecular graph generation. In International Conference on Machine Learning (ICML), volume 80 of PMLR, 2018. URL https://proceedings.mlr.press/v80/ jin18a.html.

Kummer, L., Sidak, K., Reichmann, T., and Gansterer, W. N. Adaptive precision training (AdaPT): A dynamic quantized training approach for DNNs. In SIAM International Conference on Data Mining (SDM), pp. 559–567, 2023.

Kummer, L., Gansterer, W. N., and Kriege, N. M. On the relationship between robustness and expressivity of graph neural networks. In International Conference on Artificial Intelligence and Statistics (AISTATS), 2025a.

Kummer, L., Moustafa, S., Ehrlich, A., Bause, F., Suess, N., Gansterer, W. N., and Kriege, N. M. Weisfeiler and Leman go gambling: Why expressive lottery tickets win. In International Conference on Machine Learning (ICML), 2025b. URL https://openreview.net/forum?id= 7EtP9u7JNw.

Longa, A., Lachi, V., Santin, G., Bianchini, M., Lepri, B., Lio, P., Scarselli, F., Passerini, A., et al. Graph neural networks for temporal graphs: State of the art, open challenges, and opportunities. Trans. Mach. Learn. Res., 2023:1–24, 2023.

Lu, H. and Uddin, S. A weighted patient network-based framework for predicting chronic diseases using graph neural networks. Scientific reports, 11(1):22607, 2021.

Luo, Y. and Li, P. Neighborhood-aware scalable temporal network representation learning. In Learning on Graphs Conference, pp. 1–1. PMLR, 2022.

Malach, E., Yehudai, G., Shalev-Schwartz, S., and Shamir, O. Proving the lottery ticket hypothesis: Pruning is all you need. In International Conference on Machine Learning (ICML), pp. 6682–6691, 2020.

Morris, C., Fey, M., and Kriege, N. The power of the Weisfeiler-Leman algorithm for machine learning with graphs. In International Joint Conference on Artificial Intelligence (IJCAI), pp. 4543–4550, 2021.

Morris, C., Lipman, Y., Maron, H., Rieck, B., Kriege, N. M., Grohe, M., Fey, M., and Borgwardt, K. Weisfeiler and Leman go machine learning: The story so far. Journal of Machine Learning Research, 24(333):1–59, 2023.

Puthawala, M., Kothari, K., Lassas, M., Dokmanic, I., and´ De Hoop, M. Globally injective ReLU networks. Journal ofMachine Learning Research, 23(1):4544–4598, 2022.

Qu, L., Zhu, H., Duan, Q., and Shi, Y. Continuous-time link prediction via temporal dependent graph neural network. In Proceedings ofthe Web Conference 2020, pp. 3026–3032, 2020.

Rossi, E., Chamberlain, B., Frasca, F., Eynard, D., Monti, F., and Bronstein, M. Temporal graph networks for deep learning on dynamic graphs. arXiv preprint arXiv:2006.10637, 2020.

Rusch, T. K., Bronstein, M. M., and Mishra, S. A survey on oversmoothing in graph neural networks. arXiv preprint arXiv:2303.10993, 2023.

Schlichtkrull, M., Kipf, T. N., Bloem, P., Van Den Berg, R., Titov, I., and Welling, M. Modeling relational data with graph convolutional networks. In European Semantic Web Conference, pp. 593–607. Springer, 2018.

Souza, A., Mesquita, D., Kaski, S., and Garg, V. Provably expressive temporal graph networks. Advances in Neural Information Processing Systems (NeurIPS), 35:32257–32269, 2022.

Stiefl, N., Watson, I., Baumann, K., and Zaliani, A. ErG: 2d pharmacophore descriptions for scaffold hopping. Journal of Chemical Information and Modeling, 46:208–20, 2006. doi: 10.1021/ci050457y.

Sui, Y., Wang, X., Chen, T., Wang, M., He, X., and Chua, T.- S. Inductive lottery ticket learning for graph neural networks. Journal of Computer Science and Technology, 2023.

Sun, Z., Yin, H., Chen, H., Chen, T., Cui, L., and Yang, F. Disease prediction via graph neural networks. IEEE Journal of Biomedical and Health Informatics, 25(3):818–826, 2021.

Tsitsulin, A. and Perozzi, B. The graph lottery ticket hypothesis: Finding sparse, informative graph structure. CoRR, abs/2312.04762, 2023.

Vashishth, S., Sanyal, S., Nitin, V., and Talukdar, P. P. Compositionbased multi-relational graph convolutional networks. In International Conference on Learning Representations (ICLR), 2020.

Wał˛ega, P. A. and Rawson, M. Expressive power of temporal message passing. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pp. 21000–21008, 2025.

Wang, K., Liang, Y., Wang, P., Wang, X., Gu, P., Fang, J., and Wang, Y. Searching lottery tickets in graph neural networks: A dual perspective. In International Conference on Learning Representations (ICLR), 2023.

Wu, Z., Ramsundar, B., Feinberg, E. N., Gomes, J., Geniesse, C., Pappu, A. S., Leswing, K., and Pande, V. Moleculenet: a benchmark for molecular machine learning. Chemical science, 9(2):513–530, 2018.

Xiong, J., Xiong, Z., Chen, K., Jiang, H., and Zheng, M. Graph neural networks for automated de novo drug design. Drug Discovery Today, 26(6):1382–1393, 2021.

Xu, D., Ruan, C., Korpeoglu, E., Kumar, S., and Achan, K. Inductive representation learning on temporal graphs. In International Conference on Learning Representations, 2020. URL https: //openreview.net/forum?id=rJeW1yHYwH.

Xu, K., Hu, W., Leskovec, J., and Jegelka, S. How powerful are graph neural networks? In International Conference on Learning Representations (ICLR), 2019.

Yan, J., Ito, H., García-Arias, Á. L., Okoshi, Y., Otsuka, H., Kawamura, K., Van Chu, T., and Motomura, M. Multicoated and folded graph neural networks with strong lottery tickets. In Learning on Graphs Conference (LoG), pp. 11–1, 2024.

Yin, D., Pananjady, A., Lam, M., Papailiopoulos, D., Ramchandran, K., and Bartlett, P. Gradient diversity: a key ingredient for scalable distributed learning. In International Conference on Artificial Intelligence and Statistics (AISTATS), pp. 1998–2007, 2018.

Yuxin, W., Xiannian, H., Jiaqing, X., Zhangyue, Y., Yunhua, Z., Xipeng, Q., and Xuanjing, H. Graph structure learning via lottery hypothesis at scale. In Asian Conference on Machine Learning, pp. 1401–1416, 2024.

Zaheer, M., Kottur, S., Ravanbakhsh, S., Póczos, B., Salakhutdinov, R., and Smola, A. J. Deep sets. In Advances in Neural Information Processing Systems (NeurIPS), pp. 3391–3401, 2017.

Zhang, G., Wang, K., Huang, W., Yue, Y., Wang, Y., Zimmermann, R., Zhou, A., Cheng, D., Zeng, J., and Liang, Y. Graph lottery ticket automated. In International Conference on Learning Representations (ICLR), 2024.

Zhang, S., Wang, M., Liu, S., Chen, P.-Y., and Xiong, J. Why lottery ticket wins? a theoretical perspective of sample complexity on sparse neural networks. In Advances in Neural Information Processing Systems (NeurIPS), pp. 2707–2720, 2021a.

Zhang, Z., Jin, J., Zhang, Z., Zhou, Y., Zhao, X., Ren, J., Liu, J., Wu, L., Jin, R., and Dou, D. Validating the lottery ticket hypothesis with inertial manifold theory. In Advances in Neural Information Processing Systems (NeurIPS), pp. 30196–30210, 2021b.

Zopf, M. 1-WL expressiveness is (almost) all you need. In International Joint Conference on Neural Networks (IJCNN), pp. 1–8, 2022.

## A. TGNN → RGNN via refined relational alphabet

The construction of a $\smash { \widetilde { K } _ { \star } ( T G ) }$ using an augmented relation alphabet is a little more intricate. For a fixed temporal graph $T G = ( ( G _ { 1 } , t _ { 1 } ) , \dots , ( G _ { n } , t _ { n } ) )$ in D with strictly increasing timestamps, let the finite set of realized gaps as $\Delta ( T G ) ~ : = ~ \{ t _ { j } - t _ { i } ~ | ~ 1 \leq i \leq j \leq n _ { }$ , and there is a corresponding edge in $K _ { \star } ( T G ) \rfloor$ . Furthermore, let $\sigma : \Delta ( T G ) $ $\{ 1 , \ldots , B \}$ be any injective enumeration of these gaps (with $B : = | \Delta ( T G ) | )$ ). We define the refined relation alphabet as $\begin{array} { r l r } { \widetilde { \mathscr R } } & { : = } & { \{ ( r , b ) \ | \ r \in \ \{ 0 , \ldots , n - 1 \} , \ b \in \ \{ 1 , \ldots , B \} } \end{array}$ , and $\exists i \ \leq \ j : \ j - i \ = r , \ \sigma ( t _ { j } - t _ { i } ) \ = \ b \}$ . We then form a refined temporal knowledge graph $\widetilde { K } _ { \star } ( T G ) = ( V ^ { \mathrm { t i m e } } , \dot { E } ^ { \star } , \lambda ^ { \mathrm { t i m e } } , \tilde { \tau } ^ { \star } )$ by keeping the same node set and directed edge set as in $K _ { \star } ( T G )$ , but labeling each edge $( ( v , t _ { i } )  ( u , t _ { j } ) )$ by $\widetilde { \tau } ^ { \star } ( ( v , t _ { i } ) , ( u , t _ { j } ) ) ~ : = ~ ( j - i , ~ \sigma ( t _ { j } - t _ { i } ) ) ~ \in ~ \widetilde { \mathcal { R } }$ . Then, we write the per-relation, per-direction neighborhoods on $\smash { \widetilde { K } _ { \star } ( T G ) }$ as $\begin{array} { r l r } { N _ { ( r , b ) } ^ { - } ( x ) } & { { } = } & { \{ y ~ \mid ~ \tilde { \tau } ^ { \star } ( x , y ) ~ = ~ ( r , b ) \} } \end{array}$ and $N _ { ( r , b ) } ^ { + } ( x ) = \{ y \mid \tilde { \tau } ^ { \star } ( y , x ) = ( r , b ) \}$

Lemma A.1. $L e t \star \in \{ \mathrm { g l o b } , \mathrm { l o c } \}$ and a TGNN layer as in Eqs. (4)-(5) with maps $\Omega _ { a } , \Omega _ { c } , \Phi ^ { ( l ) } , \Xi , \zeta$ . Consider the RGNN instantiated on the refined temporal knowledge graph $\smash { \widetilde { K } _ { \star } ( T G ) }$ with relation alphabet Re. There exist per-relation message maps $\Phi _ { ( r , b ) , \pm } ^ { ( l ) } : \mathbb { R } ^ { d }  \mathbb { R } ^ { d }$ and a combiner $\Gamma ^ { ( l ) }$ (same as in Eq. (5)) such that,for all $x \in V ^ { \mathrm { t i m e } }$

$$
\begin{array} { r l } & { m _ { ( r , b ) , - } ^ { ( l ) } ( x ) = \Omega _ { a } \Big ( \big \{ \big \{ \Phi _ { ( r , b ) , - } ^ { ( l ) } ( h _ { y } ^ { ( l ) } ) \big \vert y \in N _ { ( r , b ) } ^ { - } ( x ) \big \} \Big \} \Big ) , } \\ & { m _ { ( r , b ) , + } ^ { ( l ) } ( x ) = \Omega _ { a } \Big ( \big \{ \big \{ \Phi _ { ( r , b ) , + } ^ { ( l ) } ( h _ { y } ^ { ( l ) } ) \big \vert y \in N _ { ( r , b ) } ^ { + } ( x ) \big \} \Big \} \Big ) , } \\ & { \qquad h _ { x } ^ { ( l + 1 ) } = \Gamma ^ { ( l ) } \Big ( \Omega _ { c } ( \big \{ \big \{ h _ { x } ^ { ( l ) } \big \} \forall \ \sharp \ \mathcal { M } ^ { ( l ) } ( x ) \big ) \Big ) . } \end{array}
$$

with $\mathcal { M } ^ { ( l ) } ( x ) : = \lvert \pmb { \mathscr { + } } \rvert _ { s \in \widetilde { \mathcal { R } } } \left\{ m _ { s , - } ^ { ( l ) } ( x ) , m _ { s , + } ^ { ( l ) } ( x ) \right\}$ and these updates exactly match those of the original TGNN on $T G \left( g l o b a l \right.$ or local, respectively).

Proof. Let $x = ( v , t _ { j } ) \in V ^ { \mathrm { t i m e } }$ and $\star \in \{ \mathrm { g l o b } , \mathrm { l o c } \}$ . Let $\mathcal { N } _ { \star } ( x )$ denote the multiset of neighbors used by the TGNN aggregation at x in Eq. (4): for $\star = \mathrm { g l o b } , \tilde { \mathcal { N } } _ { \star } ( x ) = \{ ( u , t _ { i } ) \mid \{ u , v \}  \in E _ { i } , i \leq j \} ; \mathrm { f o r } \star = \mathrm { l o c } , \mathcal { N } _ { \star } ( x ) = \{ ( u , t _ { j } ) \mid \exists i \leq j \} .$ $j : \{ u , v \} \in E _ { i } \}$

Each $y \in \mathcal { N } _ { \star } ( x )$ is associated with a unique index lag $r = j - i$ and a realized time gap $\Delta = t _ { j } - t _ { i }$ . By construction of the augmented relation alphabet $\mathcal { \widetilde { R } } .$ , we assign a bucket $b = \sigma ( \Delta )$ to ∆. Hence every neighbor y determines a unique triple $( r , b , \triangleright )$ , whereby $\mathsf { D } \in \{ + , - \}$ indicates whether the corresponding directed edge in $\smash { \widetilde { K } _ { \star } ( T G ) }$ enters or leaves x.

This induces a disjoint partition of $\mathcal { N } _ { \star } ( x )$ into the per-relation/per-direction neighborhoods

$$
\mathcal { N } _ { \star } ( x ) ~ = ~ \bigsqcup _ { ( r , b ) \in \widetilde { \mathcal { R } } } \Big ( N _ { ( r , b ) } ^ { + } ( x ) ~ \sharp ~ N _ { ( r , b ) } ^ { - } ( x ) \Big ) ,
$$

where, by definition of $\widetilde { K } _ { \star } ( T G ) , N _ { ( r , b ) } ^ { \pm } ( x ) = \{ y \in V ^ { \mathrm { t i m e } } \mid ( r , b )$ is the edge type and $y \stackrel { \pm } { \to } x \}$

For each bucket b fix the (unique) gap value realized on that bucket at x and denote it by $\Delta _ { b }$ . Define the per-relation message maps

$$
\Phi _ { ( r , b ) , \pm } ^ { ( l ) } ( h ) : = \Phi ^ { ( l ) } ( \Xi ( h , \zeta ( \Delta _ { b } ) ) ) .
$$

Because every $y \in N _ { ( r , b ) } ^ { \pm } ( x )$ has the same index lag r and the same realized gap $\Delta _ { b }$ , we obtain

$$
\Omega _ { a } \Big ( \{ \{ \Phi _ { ( r , b ) , \pm } ^ { ( l ) } ( h _ { y } ^ { ( l ) } ) \ : | \ : \ : y \in N _ { ( r , b ) } ^ { \pm } ( x ) \ : \} \} \Big ) = \Omega _ { a } \Big ( \{ \{ \Phi ^ { ( l ) } ( \Xi ( h _ { y } ^ { ( l ) } , \zeta ( \Delta _ { b } ) ) ) \ : | \ : \ : y \in N _ { ( r , b ) } ^ { \pm } ( x ) \} \Big \} \Big )
$$

for parameterizations of $\Phi _ { ( r , b ) , \pm } ^ { ( l ) }$ suitable to the choice of $\Xi , \zeta \left( \mathbf { e } . \mathbf { g } . , \right\| , \Delta _ { b }$ , respectively, consistent with Wał˛ega & Rawson (2025).

Given $\zeta ( \Delta _ { b } ) = \zeta ( t _ { j } - t _ { i } )$ for all $y \in N _ { ( r , b ) } ^ { \pm } ( x )$ , the right-hand side is exactly the contribution that the original TGNN aggregation, Eq. (4), would collect from the subset $N _ { ( r , b ) } ^ { \pm } ( x )$ of neighbors.

Since $\{ N _ { ( r , b ) } ^ { + } ( x ) , N _ { ( r , b ) } ^ { - } ( x ) \} _ { ( r , b ) }$ forms a disjoint partition of the TGNN’s neighborhood $\mathcal { N } _ { \star } ( x )$ , aggregating within each block by $\dot { \Omega } _ { a }$ and then feeding the family of block summaries to the combiner reproduces the TGNN’s total message: the multiset

$$
\{ \Phi ^ { ( l ) } ( \Xi ( h _ { y } ^ { ( l ) } , \zeta ( t _ { j } - t _ { i } ) ) ) ~ | ~ y \in \mathcal { N } _ { \star } ( x ) \}
$$

is the disjoint multiset union of the blockwise multisets $\{ \{ \Phi _ { ( r , b ) , \pm } ^ { ( l ) } ( h _ { y } ^ { ( l ) } ) \mid y \in N _ { ( r , b ) } ^ { \pm } ( x ) \} \}$ . Therefore, with the same $\Omega _ { c }$ and $\Gamma ^ { ( l ) }$ as in (5), the update

$$
h _ { x } ^ { ( l + 1 ) } \ = \ \Gamma ^ { ( l ) } \Big ( \Omega _ { c } \big ( \{ h _ { x } ^ { ( l ) } \} \big \forall \ \vdash \ \bigsqcup _ { ( r , b ) , - } \{ x \} , m _ { ( r , b ) , + } ^ { ( l ) } ( x ) \} \big ) \Big )
$$

coincides with applying $\Gamma ^ { ( l ) }$ to the combine of ${ h } _ { x } ^ { ( l ) }$ and the TGNN’s one-shot aggregate $m _ { t _ { i } } ^ { ( l ) } ( v )$ from (4). Hence the relational layer above reproduces exactly the original TGNN layer on T G (for both glob and loc). □

While correct and allowing for the unifying perspective through RSELTH similar as Lemma 3.3, as Theorem 3.1 applies verbatim to the above RGNN, this reduction has a drawback: the resulting sparsity (pruning) masks for the RGNN do not, in general, trivially transfer back to the original TGNN, as handling the $\delta _ { i j }$ present in Eqs. (4)-(5) in the RGNN framework given by Eqs. (1)-(3) requires a refinement of the relation alphabet along with an associated extended parameterization.

## B. Temporal SELTH (TSELTH)

While Temporal SELTH as stated below in Theorem B.1 directly follows from the RGNN reformulation discussed in the main paper, we provide a reformulation and standalone proof for completeness.

Theorem B.1 (Temporal SELTH). Let D be a finite collection offinite temporal graphs. Let $\Psi ^ { ( L ) }$ be a corresponding sufficiently overparameterized depth-L TGNN ofthe pattern described either in Eqs. (4)-(5).Then,for $M \sim ^ { i . i . d . } \mathcal { B } ( 1 - \rho )$ $\Theta _ { 0 } \sim \mathcal { U } _ { c } ^ { d }$ and corresponding sparse initialization $\widehat { \Theta } _ { 0 } = M \odot \Theta _ { 0 } o f \Psi ^ { ( L ) }$ , thefollowing holdsfor al $T G _ { a } , T G _ { b } \in \mathcal { D }$

$$
K _ { \star } ( T G _ { a } ) \not \subsetneq _ { \mathrm { R W L } ^ { ( L ) } } K _ { \star } ( T G _ { b } ) \longleftrightarrow\tag{9}
$$

$$
\widehat { \Psi } _ { \widehat { \Theta } _ { 0 } , \star } ^ { ( L ) } ( T G _ { a } ) \neq \widehat { \Psi } _ { \widehat { \Theta } _ { 0 } , \star } ^ { ( L ) } ( T G _ { b } )\tag{10}
$$

with a probability at least $\gamma _ { \mathrm { T G N N } } > 0$

Proof. For any local or global TGNN, Lemma A.1 implies that, at layer l, the set of inputs that the message MLP $\Phi ^ { ( l ) }$ actually receives across D is given by

$$
S ^ { ( l ) } : = \bigcup _ { T G \in \mathcal { D } } \bigcup _ { x \in V ^ { \mathrm { t i m e } } ( T G ) } \bigcup _ { ( r , b , \triangleright ) } \Big \{ ⨏ _ { \mathbb { { Z } } } ( h _ { y } ^ { ( l ) } , \zeta ( \Delta ( y , x ) ) ) \Big | y \in N _ { ( r , b ) } ^ { \triangleright } ( x ) \Big \} ,
$$

where $( r , b , \triangleright )$ ranges over the per-lag, per-bucket, per-direction neighborhoods $N _ { ( r , b ) } ^ { \triangleright } ( \cdot )$ on $\tilde { K } _ { \star } ( T G )$ (see Lemma A.1), and $\Delta ( y , x )$ is the time gap used by the TGNN.

On a finite $\mathcal { D }$ over finite temporal graphs, $N _ { l } : = | S ^ { ( l ) } |$ is likewise finite. Let the minimum ℓ<sub>0</sub>-separation across distinct inputs be

$$
s _ { l } : = \operatorname* { m i n } _ { z \neq z ^ { \prime } \atop z , z ^ { \prime } \in S ^ { ( l ) } } \| z - z ^ { \prime } \| _ { 0 } \ ( \geq 1 ) .
$$

Consider the MLP with width $m _ { l }$ and a Bernoulli pruning mask whose entries are 0 independently with probability $\rho \in ( 0 , 1 )$ (sparsity ratio). By Lemma A.1 in Kummer et al. (2025b), the masked map is injective on $S _ { l }$ with probability at least

$$
\begin{array} { r } { \gamma _ { l } \ge 1 - \binom { N _ { l } } { 2 } \rho ^ { s _ { l } m _ { l } } . } \end{array}
$$

and the layer is injective with probability

$$
\gamma _ { l } ^ { \mathrm { l a y e r } } \ \ge \ 1 - \left( \frac { N _ { l } } { 2 } \right) \rho ^ { s _ { l } m _ { l } } - \left( \begin{array} { c } { { N _ { l , \mathrm { c o m b } } } } \\ { { 2 } } \end{array} \right) \rho ^ { s _ { l , \mathrm { c o m b } } m _ { l , \mathrm { c o m b } } } ,
$$

where $( \cdot ) _ { \mathrm { c o m b } }$ refers to the combine MLP. As in Theorem 3.1, we define for the combine MLP the input multiset $M _ { l }$ (which here is similarly the finite multiset of possible branch summaries), width $m _ { l , \mathrm { c o m b } } , N _ { l , \mathrm { c o m b } } = | M _ { l } |$ and

$$
s _ { l , \mathrm { c o m b } } : = \operatorname* { m i n } _ { z \neq z ^ { \prime } \atop z , z ^ { \prime } \in M _ { l } } \| z - z ^ { \prime } \| _ { 0 } ( \geq 1 ) .
$$

If we use independent masks across all MLP blocks, and take a conservative worst-case bound with

$$
N _ { \operatorname* { m a x } } : = \operatorname* { m a x } _ { l } N _ { l } , \quad s _ { \operatorname* { m i n } } : = \operatorname* { m i n } _ { l } s _ { l } , \quad m _ { \operatorname* { m i n } } : = \operatorname* { m i n } _ { l } m _ { l } ,
$$

$$
N _ { \mathrm { m a x , c o m b } } : = \operatorname* { m a x } _ { l } N _ { l , c o m b } , \quad s _ { \mathrm { m i n } } , : = \operatorname* { m i n } _ { l } s _ { l , c o m b } , \quad m _ { \mathrm { m i n , c o m b } } : = \operatorname* { m i n } _ { l } m _ { l , \mathrm { c o m b } } ,
$$

then each branch is injective with probability at least $1 - \binom { N _ { m a x } } { 2 } \rho ^ { s _ { \mathrm { m i n } } m }$ <sup>min</sup> and each combination with probability at least $1 - \binom { N _ { m a x , \mathrm { c o m b } } } { 2 } \rho ^ { s }$ <sup>min,combmmin,comb</sup> . For $M = 1 { \bf M } { \bf L } { \bf P }$ sublayers per MP layer and L MP layers, without relational branch MLPs as in Theorem 3.1, the crude product lower bound for TGNNs simplifies to

$$
\gamma _ { \mathrm { T G N N } } \geq \left( [ \left( 1 - \binom { N _ { m a x } } { 2 } \rho ^ { s _ { \mathrm { m i n } } m _ { \mathrm { m i n } } } ] _ { + } \right) \left( [ 1 - \binom { N _ { \mathrm { m a x } , \mathrm { c o m b } } } { 2 } \rho ^ { s _ { \mathrm { m i n } , \mathrm { c o m b } } m _ { \mathrm { m i n } , \mathrm { c o m b } } } ] _ { + } \right) \right) ^ { L }
$$

with $[ x ] _ { + } : = \operatorname* { m a x } \{ x , 0 \}$ and furthermore to

$$
\widetilde { \gamma } _ { \mathrm { T G N N } } \geq \left( [ 1 - \binom { \widetilde { N } _ { m a x } } { 2 } \rho ^ { \widetilde { s } _ { \operatorname* { m i n } } \widetilde { m } _ { \operatorname* { m i n } } } ] _ { + } \right) ^ { L } ,
$$

with

$$
\begin{array} { r } { \widetilde { N } _ { \operatorname* { m a x } } : = \operatorname* { m a x } \{ N _ { \operatorname* { m a x } } , N _ { \operatorname* { m a x } , \coth } \} , \quad \widetilde { s } _ { \operatorname* { m i n } } : = \operatorname* { m i n } \{ s _ { \operatorname* { m i n } } , s _ { \operatorname* { m i n } , \coth } \} , \quad \widetilde { m } _ { \operatorname* { m i n } } : = \operatorname* { m i n } \{ m _ { \operatorname* { m i n } } , m _ { \operatorname* { m i n } , \coth } \} . } \end{array}
$$

By the same argument as in Theorem 3.1, we argue this estimate is conservative; so for fixed $N _ { m a x } , s _ { \mathrm { m i n } }$ and their combine $\mathrm { o r } ^ { \sim }$ counterparts, one can pick m such that $\gamma _ { \mathrm { T G N N } } > 0$ □

## C. TSELTH Node Level Transfer

The transfer is possible because 1-RWL (and its temporal variants via $K _ { \star } ( T G ) )$ operates fundamentally through iterative node color refinement: nodes with distinct k-hop relational or temporal neighborhoods receive distinct colors (Corollary 2.2), encoding their structural roles. In sparse subnetworks under SELTH, the preservation of injective aggregation and combi nation functions ensures that node embeddings $h _ { x } ^ { ( L ) }$ remain distinct for such nodes, providing the discriminative features needed for node-level predictions.

Lemma C.1. $F i x \star \in \left\{ \mathrm { g l o b } \right.$ , loc} and depth L. Consider a TGNN oftheform Eqs. $( 4 ) \AA - \mathrm { ( 5 ) }$ with (i) an injective label encoder, (ii) injective $\Phi ^ { ( l ) }$ and $\dot { \Gamma } ^ { ( l ) }$ on the finite input sets they actually witness on a finite dataset ${ \mathcal { D } } ,$ and (iii) $\Omega _ { a } , \Omega _ { c }$ chosen as injective multiset encoders on those finite domains $( e . g . , \sum$ with a linearly independent codebook (see Remark 2.1)). Then for every $T G \in D$ and every $k \leq L$ there exists an injective $\eta _ { k }$ such thatfor all timestamped nodes x

$$
h _ { x } ^ { ( k ) } \ = \ \eta _ { k } ( c _ { K _ { \star } ( T G ) } ^ { ( k ) } ( x ) ) .
$$

In particular, $i f c _ { K _ { \star } ( T G ) } ^ { ( k ) } ( x ) \neq c _ { K _ { \star } ( T G ) } ^ { ( k ) } ( x ^ { \prime } )$ , then $h _ { x } ^ { ( k ) } \neq h _ { x ^ { \prime } } ^ { ( k ) }$ under the same parameters.

Proof. We proceed by induction on $k .$

Base case $( k = 0 )$ . By definition of directed 1-RWL on $K _ { \star } ( T G )$ , the initial color is $c _ { K _ { \star } ( T G ) } ^ { ( 0 ) } ( x ) = \lambda ^ { \mathrm { t i m e } } ( x )$ . The TGNN initializes $h _ { x } ^ { ( 0 ) } = \Lambda ( \lambda ^ { \mathrm { t i m e } } ( x ) )$ ) with an injective label encoder Λ. Hence setting $\eta _ { 0 } : = \Lambda$ gives $h _ { x } ^ { ( 0 ) } = \eta _ { 0 } ( c _ { K _ { \star } ( T G ) } ^ { ( 0 ) } ( x ) )$ ), and $\eta _ { 0 }$ is injective.

Inductive step. Assume for some $k \geq 0$ there exists an injective $\eta _ { k }$ such that $h _ { y } ^ { ( k ) } = \eta _ { k } ( c _ { K _ { \star } ( T G ) } ^ { ( k ) } ( y ) )$ for all timestamped nodes y. Fix $x = ( v , t _ { j } ) \in V ^ { \mathrm { t i m e } }$ . In the global/local TGNN, aggregation ranges over the temporal neighborhood $N ( v , t _ { j } )$

as in Eq. (4). By construction of $K _ { \star } ( T G )$ , each $( u , t _ { i } ) \in N ( v , t _ { j } )$ ) with $i \leq j$ appears as an incoming edge $( u , t _ { i } ) \to ( v , t _ { j } )$ labeled by the unique index lag $r = j - i$ . Writing

$$
N _ { r } ^ { + } ( x ) \ : = \ \{ \ ( u , t _ { i } ) | \ ( u , t _ { i } ) {  } ( v , t _ { j } ) \ \mathrm { i n } \ K _ { \star } ( T G ) , \ j { - } i = r \ \} ,
$$

we obtain a disjoint partition of the temporal neighborhood by lag:

$$
N ( v , t _ { j } ) \ = \ \biguplus _ { r = 0 } ^ { j - 1 } N _ { r } ^ { + } ( x ) ,
$$

and, moreover, $\delta ( ( u , t _ { i } ) , x ) = t _ { j } - t _ { i }$ is constant on each slice $N _ { r } ^ { + } ( x )$ (namely $t _ { j } - t _ { i }$ with $j - i = r )$ . Consequently, we may equivalently index the aggregation by the per-lag, per-direction neighborhoods $N _ { r } ^ { \triangleright } ( x )$ of $K _ { \star } ( T G )$ , which matches the directed 1-RWL bookkeeping; the same reasoning applies to the outgoing branch $\triangleright = - \mathrm { i f }$ present.

For each relation r and direction $\triangleright \in \{ + , - \}$ on $K _ { \star } ( T G )$ , define

$$
\begin{array} { r } { \mathcal { A } _ { r , \mathrm { s } } ^ { ( k ) } ( x ) : = \{ \{ c _ { K _ { \star } ( T G ) } ^ { ( k ) } ( y ) ~ | ~ y \in N _ { r } ^ { \mathrm { \flat } } ( x ) ~ \} \quad \mathrm { a n d } \quad \mathcal { H } _ { r , \mathrm { s } } ^ { ( k ) } ( x ) : = \{ \{ h _ { y } ^ { ( k ) } ~ | ~ y \in N _ { r } ^ { \mathrm { \flat } } ( x ) ~ \} \} . } \end{array}
$$

By the induction hypothesis, elementwise $h _ { y } ^ { ( k ) } = \eta _ { k } ( c _ { K _ { \star } ( T G ) } ^ { ( k ) } ( y ) )$ on the finite dataset ${ \mathcal { D } } ,$ so (as multisets)

$$
\mathcal { H } _ { r , \triangleright } ^ { ( k ) } ( x ) \ : = \ : \eta _ { k } [ \mathcal { A } _ { r , \triangleright } ^ { ( k ) } ( x ) ] ,
$$

where $\eta _ { k } [ \cdot ]$ acts elementwise.

Apply the layer’s elementwise message map and relation/direction aggregator:

$$
m _ { r , \mathrm { b } } ^ { ( k ) } ( x ) = \Omega _ { a } ( \{ \int \Phi ^ { ( k ) } ( h _ { y } ^ { ( k ) } ) ~ \vert ~ y \in N _ { r } ^ { \mathrm { b } } ( x ) ~ \} \} ) = \underbrace { \Omega _ { a } ( \{ \{ \Phi ^ { ( k ) } ( \eta _ { k } ( a ) ) ~ \vert ~ a \in \mathcal { A } _ { r , \mathrm { b } } ^ { ( k ) } ( x ) ~ \} \} ) } _ { : = ~ F _ { r , \mathrm { b } } ^ { ( k ) } ( \mathcal { A } _ { r , \mathrm { b } } ^ { ( k ) } ( x ) ) } .
$$

By assumption, $\Phi ^ { ( k ) }$ is injective on the finite set $\eta _ { k } ( \operatorname { I m } c ^ { ( k ) } )$ actually witnessed on $\mathcal { D } ,$ and $\Omega _ { a }$ is an injective multiset encoder on the corresponding finite domain $( \mathrm { e . g . }$ , sum over a linearly independent codebook; cf. Remark 2.1). Therefore $F _ { r , \triangleright } ^ { ( k ) }$ is injective on the finite family of multisets $A _ { r , \triangleright } ^ { ( k ) } ( x )$ realized on $\mathcal { D } .$

Next, combine the self-embedding and all $\mathrm { p e r } { - } ( r , \mathsf { D } )$ messages:

$$
h _ { x } ^ { ( k + 1 ) } = \Gamma ^ { ( k ) } \Big ( \Omega _ { c } ( \{ \{ h _ { x } ^ { ( k ) } \} \} \ \sharp \ \sharp | \{ m _ { r , s } ^ { ( k ) } ( x ) \} \} ) \Big ) = \Gamma ^ { ( k ) } \Big ( \Omega _ { c } ( \{ \{ \eta _ { k } ( c ^ { ( k ) } ( x ) ) \} \} \ \sharp \ \sharp \ | \{ F _ { r , s } ^ { ( k ) } ( A _ { r , s } ^ { ( k ) } ( x ) ) \} \} ) \Big ) .
$$

By assumption, $\Omega _ { c }$ is injective on the finite domain of inputs realized on $\mathcal { D } ,$ and $\Gamma ^ { ( k ) }$ is injective there as well. Moreover, since each $F _ { r , \triangleright } ^ { ( k ) }$ is injective and we retain the (r, ▷) tags, the map from the refinement input

$$
\left( c ^ { ( k ) } ( x ) , \ ( \mathcal { A } _ { r , - } ^ { ( k ) } ( x ) ) _ { r } , \ ( \mathcal { A } _ { r , + } ^ { ( k ) } ( x ) ) _ { r } \right)
$$

to all per-relation/per-direction messages $m _ { r , \triangleright } ^ { ( k ) } ( x )$ (for $r \in \mathcal { R }$ and $\textsf { D } \in \{ + , - \} )$ is injective. Applying the injective $\Omega _ { c }$ to this indexed tuple and then the injective $\Gamma ^ { ( k ) }$ preserves injectivity, so the right-hand side is an injectivefunction of the above refinement input, which is precisely the directed 1-RWL refinement input at x on $K _ { \star } ( T G )$ . As the 1-RWL color update

$$
c _ { K _ { \star } ( T G ) } ^ { ( k + 1 ) } ( x ) \ = \ \Upsilon _ { 3 } \Big ( c ^ { ( k ) } ( x ) , \ ( A _ { r , - } ^ { ( k ) } ( x ) ) _ { r } , \ ( A _ { r , + } ^ { ( k ) } ( x ) ) _ { r } \Big )
$$

uses an injective $\Upsilon _ { 3 }$ on this finite domain, we can define $\eta _ { k + 1 }$ on the set $o f ( k { + } 1 )$ -colors actually realized on $\mathcal { D }$ by

$$
\eta _ { k + 1 } ( c _ { K _ { \star } ( T G ) } ^ { ( k + 1 ) } ( x ) ) \ : = \ \Gamma ^ { ( k ) } \Big ( \Omega _ { c } \big ( \{ \{ \eta _ { k } ( c ^ { ( k ) } ( x ) ) \} \} \} \ \sharp \ \sharp \ \big \vert _ { r , \mathtt { p } } \{ \{ f _ { r , \mathtt { p } } ^ { ( k ) } ( \mathcal { A } _ { r , \mathtt { p } } ^ { ( k ) } ( x ) ) \} \} \} \big ) \Big ) .
$$

This is well-defined (equal colors correspond to equal refinement inputs) and injective (distinct colors correspond to distinct refinement inputs and the outer map is injective), and we have $h _ { x } ^ { ( k + 1 ) } = \eta _ { k + 1 } \bar { ( c _ { K _ { \star } ( T G ) } ^ { ( k + 1 ) } ( x ) ) }$

By induction, for every $k \leq L$ there exists an injective $\eta _ { k }$ with $h _ { x } ^ { ( k ) } = \eta _ { k } ( c _ { K _ { \star } ( T G ) } ^ { ( k ) } ( x ) )$ for all timestamped nodes x. In particular, for any $x , x ^ { \prime } ,$ if $c _ { K _ { \star } ( T G ) } ^ { ( k ) } ( x ) \neq c _ { K _ { \star } ( T G ) } ^ { ( k ) } ( x ^ { \prime } )$ , then $h _ { x } ^ { ( k ) } \neq h _ { x ^ { \prime } } ^ { ( k ) }$ under the same parameters. □

By Theorem B.1, on any finite D there exists a sparse mask M that preserves the layerwise injectivity conditions above.   
Therefore, Lemma C.1 continues to hold under the same sparse support M.

## D. Proofs (Main Paper)

Complete proofs for all theoretical claims made in the main paper are given below. The proofs for theoretical claims made in the appendix are given directly below the respective claim.

## D.1. Proof of Corollary 2.2

Corollary 2.2 is effectively a concise summary of Theorems 2 to 6 proven by Wał˛ega & Rawson (2025).

Proof. Let a temporal graph T G and timestamped nodes $x = ( v , t _ { i } ) , x ^ { \prime } = ( u , t _ { j } ) \in V ^ { \mathrm { t i m e } }$ , and let $\star \in \{ \mathrm { g l o b } , \mathrm { l o c } \}$ match the TGNN MP type.

(⇒) By Theorem 2 (global) and Theorem 5 (local) of Wał˛ega & Rawson (2025), directed 1-RWL on $K _ { \star } ( T G )$ upper-bounds any MP-TGNN of the corresponding type. In our notation this reads: if $c _ { K _ { \star } ( T G ) } ^ { ( k ) } ( x ) = c _ { K _ { \star } ( T G ) } ^ { ( k ) } ( x ^ { \prime } )$ , then there exists a TGNN layer of the form (4) with combiner (5) for which we have $h _ { x } ^ { ( k ) } = h _ { x ^ { \prime } } ^ { ( k ) }$ , which implies the existence of corresponding parameters.

(⇐) By Theorem 3 (global) and Theorem 6 (local) of Wał˛ega & Rawson (2025), there exists a TGNN of the corresponding architectural form such that, layerwise, embeddings can be mapped 1-to-1 to the 1-RWL colors on $K _ { \star } ( T G )$ . Instantiating our layer class (4), (5) with $\Omega _ { a } = \Sigma$ and choosing $\Phi ^ { ( l ) } , \Gamma ^ { ( l ) }$ as in the referred constructions, hence implies the existence of parameters for which

$$
h _ { x } ^ { ( k ) } = h _ { x ^ { \prime } } ^ { ( k ) } \quad \Longleftrightarrow \quad c _ { K _ { \star } ( T G ) } ^ { ( k ) } ( x ) = c _ { K _ { \star } ( T G ) } ^ { ( k ) } ( x ^ { \prime } ) .
$$

This proves the bi-implication for some parametrization of any TGNN of the examined class.

## D.2. Proof of Lemma 3.2

Proof. Let $\star \in \{ \mathrm { g l o b } , \mathrm { l o c } \}$ and $T G _ { a } , T G _ { b } \in \mathcal { D }$ and assume $K _ { \star } ( T G _ { a } ) \not \simeq \not \simeq \not \simeq \ W \mathrm { L } ^ { ( L ) } \ K _ { \star } ( T G _ { b } )$ . Then their L-step 1-RWL color multisets on $K _ { \star }$ differ:

$$
\sharp c _ { K _ { \star } ( T G _ { a } ) } ^ { ( L ) } ( x ) \mid x \in V ^ { \mathrm { t i m e } } ( T G _ { a } ) \mathbb { J } \neq \{ \{ c _ { K _ { \star } ( T G _ { b } ) } ^ { ( L ) } ( x ) \mid x \in V ^ { \mathrm { t i m e } } ( T G _ { b } ) \} \} .
$$

By Corollary 2.2, there exists a parametrization of the (local/global) TGNN such that, layerwise, $h _ { x } ^ { ( L ) } = \psi _ { L } ( c _ { K _ { \star } ( T G ) } ^ { ( L ) } ( x ) )$ for some injective $\psi _ { L }$ . Consequently, the two multisets of node embeddings differ.

Let the TGNN’s graph readout $\chi$ be permutation-invariant and injective on finite multisets; for instance, $\chi ( \{ \{ h _ { x } \} \} ) =$ $\alpha ( \sum x _ { } h _ { x } )$ , where α is injective (e.g., a suitable MLP (Amir et al., 2023; Puthawala et al., 2022)) and the codebook (see Remark $2 . 1 ) \left\{ \psi _ { L } ( \cdot ) \right\}$ is chosen linearly independent, which makes the sum an injective multiset encoder (Zaheer et al., 2017). Then

$$
\Psi _ { \Theta , * } ^ { ( L ) } ( T G _ { a } ) = \chi \{ \{ h _ { x } ^ { ( L ) } \mid x \in V ^ { \mathrm { t i m e } } ( T G _ { a } ) \} \} ) \neq \chi \{ \{ h _ { x } ^ { ( L ) } \mid x \in V ^ { \mathrm { t i m e } } ( T G _ { b } ) \} \} = \Psi _ { \Theta , * } ^ { ( L ) } ( T G _ { b } ) ,
$$

for suitable Θ as claimed.

Conversely, assume $\Psi _ { \Theta , \star } ^ { ( L ) } ( T G _ { a } ) \neq \Psi _ { \Theta , \star } ^ { ( L ) } ( T G _ { b } )$ . By injectivity of $\chi .$ , the underlying multisets of node embeddings must differ:

$$
\{ \{ h _ { x } ^ { ( L ) } ~ | ~ x \in V ^ { \mathrm { t i m e } } ( T G _ { a } ) \} \} ~ \neq ~ \{ \{ h _ { x } ^ { ( L ) } ~ | ~ x \in V ^ { \mathrm { t i m e } } ( T G _ { b } ) \} \} .
$$

As $h _ { x } ^ { ( L ) } = \psi _ { L } ( c _ { K _ { \star } ( T G ) } ^ { ( L ) } ( x ) )$ ) with injective $\psi _ { L }$ , the two color multisets at depth $L$ must differ, hence $K _ { \star } ( T G _ { a } ) \nsubseteq _ { \mathrm { R W L } ^ { ( L ) } }$ $K _ { \star } ( T G _ { b } )$

This establishes the claimed bi-implication for the chosen parameterization Θ.

## D.3. Proof of Theorem 3.1

Proof. In R-GNN $\Psi ^ { ( L ) }$ , if each layer injectively aggregates separately per relation and per direction (Eq. (1)) and then applies an injective combine to the resulting tuple together with the node’s own embedding (Eq. (3)), the layer realizes exactly the 1- RWL refinement step at the next iteration. This is the relational analogue of the criterion used in the static case (Kummer et al., 2025b), Lemma 2.1, which is based on the correspondence between 1-WL and GNN aggregation/combine injectivity (Xu et al., 2019). Stacking L such RGNN layers matches 1-RWL to depth L.

Hence, we need to first show injectivity can be preserved under pruning. Let a finite dataset D (cardinality |D|) and a depth-L RGNN $\Psi ^ { ( L ) }$ . At any layer l and for any branch b (one per relation $r \in \mathcal { R }$ in the undirected case, or per relation/direction in the directed case), let $X _ { l , b } \subset \mathbb { R } ^ { n _ { l , i } }$ <sup>b</sup> denote the finite set of inputs that actually occur at branch b when propagating D up to layer l. We write $N _ { l , b } = | X _ { l , b } |$ and let

$$
s _ { l , b } : = \operatorname* { m i n } _ { \stackrel { x \neq x ^ { \prime } } { x , x ^ { \prime } \in X _ { l , b } } } \| x - x ^ { \prime } \| _ { 0 } ( \geq 1 ) ,
$$

be the minimum $\ell _ { 0 } .$ -separation across distinct inputs in $X _ { l , b }$ . Consider the branch MLP $\Phi _ { b } ^ { ( l ) }$ with width $m _ { l , b }$ and a Bernoulli pruning mask whose entries are 0 independently with probability $\rho \in ( 0 , 1 )$ (sparsity ratio). By Lemma A.1 in Kummer et al. (2025b), which has the same distributional assumptions on parameters and pruning masks, the masked branch map is injective on $X _ { l , b }$ with probability at least

$$
\gamma _ { l , b } \geq 1 - \binom { N _ { l , b } } { 2 } \rho ^ { s _ { l , b } m _ { l , b } } .
$$

Apply the same argument to every branch of layer l and to the combine MLP $\Gamma ^ { ( l ) }$ (whose input multiset $M _ { l }$ is the finite multiset of possible branch summaries) with width $m _ { l , \mathrm { c o m b } } , N _ { l , \mathrm { c o m b } } = | M _ { l } |$ and

$$
s _ { l , \mathrm { c o m b } } : = \operatorname* { m i n } _ { \begin{array} { c } { x \neq x ^ { \prime } } \\ { x , x ^ { \prime } \in M _ { l } } \end{array} } \| x - x ^ { \prime } \| _ { 0 } ( \geq 1 ) .
$$

Then, by a union bound one obtains a layer-level lower bound

$$
\gamma _ { l } ^ { \mathrm { l a y e r } } \ \geq \ 1 - \sum _ { b \in \mathcal { B } _ { t } } \binom { N _ { l , b } } { 2 } \rho ^ { s _ { l , b } m _ { l , b } } - \binom { N _ { l , \mathrm { c o m b } } } { 2 } \rho ^ { s _ { l , \mathrm { c o m b } } m _ { l , \mathrm { c o m b } } } ,
$$

where $\boldsymbol { B } _ { l }$ is the set of branches at layer l and $( \cdot ) _ { \mathrm { c o m b } }$ refers to the combine MLP.

If we use independent masks across all MLP blocks, and take a conservative worst-case bound with

$$
N _ { \operatorname* { m a x } } : = \operatorname* { m a x } _ { l , b } N _ { l , b } , \quad s _ { \operatorname* { m i n } } : = \operatorname* { m i n } _ { l , b } s _ { l , b } , \quad m _ { \operatorname* { m i n } } : = \operatorname* { m i n } _ { l , b } m _ { l , b }
$$

$$
N _ { \mathrm { m a x , c o m b } } : = \operatorname* { m a x } _ { l } N _ { l , c o m b } , \quad s _ { \mathrm { m i n } } , : = \operatorname* { m i n } _ { l } s _ { l , c o m b } , \quad m _ { \mathrm { m i n , c o m b } } : = \operatorname* { m i n } _ { l } m _ { l , \mathrm { c o m b } } ,
$$

then each branch is injective with probability at least $1 - \binom { N _ { m a x } } { 2 } \rho ^ { s _ { \mathrm { m i n } } m _ { \mathrm { m i n } } }$ and each combination with probability at least $1 - \binom { N _ { m a x , \mathrm { c o m b } } } { 2 } \rho ^ { s }$ <sup>min,combmmin,comb</sup> . Counting M MLP sublayers per MP layer and L MP layers, with $| B _ { l } | = | \mathcal { R } |$ (undirected) or $| B _ { l } | = \bar { 2 } | \mathcal { R } |$ (directed), a crude product lower bound is

$$
\gamma _ { \mathrm { R G N N } } \geq \left( \left( \left[ 1 - \binom { N _ { \operatorname* { m a x } } } { 2 } \rho ^ { s _ { \operatorname* { m i n } } m _ { \operatorname* { m i n } } } \right] _ { + } \right) ^ { | \mathcal { B } | M } \left( \left[ 1 - \binom { N _ { m a x , \cosh } } { 2 } \rho ^ { s _ { \operatorname* { m i n } , \cosh } m _ { \operatorname* { m i n } , \cosh } } \right] _ { + } \right) \right) ^ { L } ,
$$

with $[ x ] _ { + } : = \operatorname* { m a x } \{ x , 0 \}$ , where $| B | = | { \mathcal { R } } | \ \mathrm { o r 2 } | { \mathcal { R } } |$ , which further simplifies to

$$
\widetilde { \gamma } _ { \mathrm { R G N N } } \geq \left( [ 1 - \binom { \widetilde { N } _ { m a x } } { 2 } \rho ^ { \widetilde { s } _ { \operatorname* { m i n } } \widetilde { m } _ { \operatorname* { m i n } } } ] _ { + } \right) ^ { L ( M | \mathcal { B } | + 1 ) } ,
$$

with

$$
\begin{array} { r } { \widetilde { N } _ { \operatorname* { m a x } } : = \operatorname* { m a x } \{ N _ { \operatorname* { m a x } } , N _ { \operatorname* { m a x } , \coth } \} , \quad \widetilde { s } _ { \operatorname* { m i n } } : = \operatorname* { m i n } \{ s _ { \operatorname* { m i n } } , s _ { \operatorname* { m i n } , \coth } \} , \quad \widetilde { m } _ { \operatorname* { m i n } } : = \operatorname* { m i n } \{ m _ { \operatorname* { m i n } } , m _ { \operatorname* { m i n } , \coth } \} . } \end{array}
$$

This is conservative, as (i) earlier non-injectivity would shrink the finite input sets faced by later blocks, increasing their success probability and (ii) any increase of width m drives $\rho ^ { s m }$ down exponentially, so for fixed $N _ { m a x } , s _ { \mathrm { m i n } }$ and their combine or · counterparts, one can pick the width m of the pruned branch and combine MLPs such that $\gamma _ { \mathrm { R G N N } } > 0$ or $\widetilde { \gamma } _ { \mathrm { R G N N } } > 0$

Consequentially, the random pruning of $\Psi ^ { ( L ) }$ can preserve 1-RWL expressivity under mild assumptions<sup>2</sup>.

## D.4. Proof of Lemma 3.4

Proof. We begin by first showing the following auxiliary Corollary:

Corollary D.1. Let $f : V ( G _ { i } ^ { a } ) {  } V ( G _ { i } ^ { b } )$ be the timewise isomorphism’s snapshot-wise bijection and define $f ^ { \prime } : \left( v , t _ { i } ^ { a } \right) \mapsto$ $( f ( v ) , t _ { i } ^ { b } )$ . Then for all $l \in \{ 0 , \ldots , L \}$ and all $x = ( v , t _ { i } ^ { a } )$ we have $c _ { K _ { \star } ( T G _ { a } ) } ^ { ( l ) } ( x ) = c _ { K _ { \star } ( T G _ { b } ) } ^ { ( l ) } ( f ^ { \prime } ( x ) )$ .

Proof. By Wał˛ega & Rawson (2025)’s Theorem 10, timewise-isomorphic timestamped nodes produce identical embeddings in any TGNN at every layer. Instantiate the specific local/global TGNN that (by the standard WL-characterisation used in our Corollary 2.2) computes embeddings $h ^ { ( l ) } = \psi _ { l } ( c _ { K _ { \star } ( \cdot ) } ^ { ( l ) } )$ with ψ<sub>l</sub> injective at each layer l. Applying Wał˛ega & Rawson (2025)’s Theorem 10 to this TGNN yields $h _ { x } ^ { ( l ) } = h _ { f ^ { \prime } ( x ) } ^ { ( l ) }$ , hence injectivity of ψ gives $c _ { K _ { \star } ( T G _ { a } ) } ^ { ( l ) } ( x ) = c _ { K _ { \star } ( T G _ { b } ) } ^ { ( l ) } ( f ^ { \prime } ( x ) )$ for all l. □

The map $f ^ { \prime }$ is a bijection between $V ^ { \mathrm { t i m e } } ( T G _ { a } )$ and $V ^ { \mathrm { t i m e } } ( T G _ { b } )$ , so Corollary D.1 implies equality of the entire L-step color multisets, as stated. In particular, if one employs a permutation-invariant, multiset-injective graph readout $\chi$ for the final TGNN layer $( \mathrm { e . g . }$ , a sum followed by an injective MLP), then applying $\chi$ to equal multisets yields equal graph-level outputs $\Psi _ { \Theta , \star } ^ { ( L ) } ( T G _ { a } ) = \Psi _ { \Theta , \star } ^ { ( L ) } ( T G _ { b } )$ for suitable Θ. □

## D.5. Proof of Corollary 3.5

Proof. For each $( G , \{ T _ { i } , S _ { i } \} ) \in \mathcal { D }$ , we build $G ^ { \oplus }$ as in Section 3. Then, by construction, stacking L relational layers on $G ^ { \oplus }$ (with the appropriate parameter choices) simulates L XIMP layers. As in the relational setup, we pick parameters so that at every layer l there exists an injective map $\psi _ { l }$ with $\widetilde { h } ^ { ( l ) } = \psi _ { l } ( c ^ { ( l ) } )$ , where $c ^ { ( l ) }$ is the l-round 1-RWL color on $G ^ { \oplus }$ ; this is feasible on finite inputs using sum aggregation with a linearly independent codebook (see Remark 2.1) and injective $\Phi _ { r , + } ^ { ( l ) } , \Phi _ { r , - } ^ { ( l ) } , \Gamma ^ { ( l ) }$

If $G _ { a } ^ { \oplus } \not \simeq _ { \mathrm { R W L } ^ { ( L ) } } G _ { b } ^ { \oplus }$ , then their L-round color multisets differ. Injectivity of $\psi _ { L }$ implies the final-layer node-embedding multisets differ. With any permutation-invariant readout $\chi$ that is injective on finite multisets $( \mathrm { e . g . }$ , sum followed by an injective MLP), the graph-level outputs differ:

$$
\Psi _ { \Theta } ^ { ( L ) } ( G _ { a } , \{ T _ { i } ^ { a } , S _ { i } ^ { a } \} ) \ \neq \ \Psi _ { \Theta } ^ { ( L ) } ( G _ { b } , \{ T _ { i } ^ { b } , S _ { i } ^ { b } \} ) .
$$

Conversely, if $\Psi _ { \Theta } ^ { ( L ) } ( G _ { a } , \{ T _ { i } ^ { a } , S _ { i } ^ { a } \} ) \neq \Psi _ { \Theta } ^ { ( L ) } ( G _ { b } , \{ T _ { i } ^ { b } , S _ { i } ^ { b } \} )$ under an injective $\chi ,$ then the final-layer node-embedding multisets must differ; by injectivity of $\psi _ { L }$ , the L-round 1-RWL color multisets differ, hence $G _ { a } ^ { \oplus } \operatorname * { \mathcal { L } } _ { \mathrm { R W L } ( L ) } G _ { b } ^ { \oplus }$

Therefore, for suitable parameters, 1-RWL on $G ^ { \oplus }$ exactly characterizes the L-round distinguishing power of XIMP/HIMP on the finite collection $\mathcal { D } .$ □

## D.6. Proof of Lemma 3.3

Proof. Fix $\star \in \{ \mathrm { g l o b } , \mathrm { l o c } \}$ and a temporal node $x = ( v , t _ { j } ) \in V ^ { \mathrm { t i m e } }$ . Recall the TGNN aggregation and update from Eqs. (4)-(5):

$$
\begin{array} { r l } { \mathrm { ( g l o b a l ) } } & { m _ { t _ { j } } ^ { ( l ) } ( v ) = \Omega _ { a } \Big ( \{ \mathfrak { E } ^ { ( l ) } ( \Xi ( h _ { ( u , t _ { i } ) } ^ { ( l ) } , \zeta ( \delta _ { i j } ) ) ) \ \textbf { | } \ ( u , t _ { i } ) \in N ( v , t _ { j } ) \} \Big ) , } \\ { \mathrm { ( l o c a l ) } } & { m _ { t _ { j } } ^ { ( l ) } ( v ) = \Omega _ { a } \Big ( \{ \mathfrak { E } ^ { ( l ) } ( \Xi ( h _ { ( u , t _ { j } ) } ^ { ( l ) } , \zeta ( \delta _ { i j } ) ) ) \ \textbf { | } \ ( u , t _ { i } ) \in N ( v , t _ { j } ) \} \Big ) , } \\ & { \ h _ { ( v , t _ { j } ) } ^ { ( l + 1 ) } = \Gamma ^ { ( l ) } \Big ( \Omega _ { c } ( \{ \mathfrak { h } _ { ( v , t _ { j } ) } ^ { ( l ) } \mathbb { \ } \sharp \textbf {  } \{ \mathfrak { f } _ { i j } ( v ) \} \} ) , } \end{array}
$$

where $\delta _ { i j } = t _ { j } - t _ { i }$ and $N ( v , t _ { j } )$ is the temporal neighborhood $N ( v , t _ { j } ) = \{ ( u , t _ { i } ) \mid \{ u , v \} \in E _ { i } , i \leq j \}$

On the knowledge-graph side, let $K _ { \star } ( T G )$ be the (directed, typed) temporal knowledge graph constructed for ⋆ as in the preliminaries, and define its edge-feature augmentation $\xi _ { ( y  x ) } : = \zeta ( \delta ( y , x ) )$ ), where for $\boldsymbol { y } = \left( \boldsymbol { u } , t _ { i } \right)$ and $\boldsymbol { x } = ( v , t _ { j } )$ we have $\delta ( y , x ) = t _ { j } - t _ { i }$ . As in the statement, instantiate the RGNN on $\smash { \widetilde { K } _ { \star } ( T G ) }$ with

$$
\widetilde { \Phi } ^ { ( l ) } ( h , \xi ) = \Phi ^ { ( l ) } ( \Xi ( h , \xi ) ) ,
$$

strict tying $\Phi _ { r , \pm } ^ { ( l ) } \equiv \Phi ^ { ( l ) }$ across $( r , \pm )$ , and the same $( \Omega _ { a } , \Omega _ { c } , \Gamma ^ { ( l ) } )$

We now prove by induction on l that for all $\boldsymbol { x } = ( v , t _ { j } )$

$$
h _ { ( v , t _ { j } ) } ^ { ( l ) } \left( \mathrm { T G N N } \right) \ : = \ : \widetilde { h } _ { ( v , t _ { j } ) } ^ { ( l ) } \left( \mathrm { R G N N } \right) .
$$

Base case $( l = 0 )$ . Both models use the same initialization $h _ { ( v , t _ { j } ) } ^ { ( 0 ) } = \Lambda ( \lambda ^ { \mathrm { t i m e } } ( v , t _ { j } ) )$ , so the claim holds.

Induction step. Assume $h _ { ( . ) } ^ { ( l ) } = \widetilde { h } _ { ( . ) } ^ { ( l ) }$ for all timestamped nodes. Fix $\boldsymbol { x } = ( v , t _ { j } )$ and consider the multiset of incoming neighbors to x in $K _ { \star } ( T G )$ , grouped by index lag $r = j - i \colon$

$$
\bigcup _ { r } \ N _ { r } ^ { + } ( x ) \quad \mathrm { w i t h } \quad N _ { r } ^ { + } ( x ) = \{ ( u , t _ { i } ) \mid ( u , t _ { i } )  ( v , t _ { j } ) \mathrm { i n } K _ { \star } ( T G ) , j - i = r \} .
$$

By construction of $K _ { \star } ( T G )$ , for every $( u , t _ { i } ) \in N ( v , t _ { j } )$ there is a unique index lag $r = j { - } i \in \{ 0 , \ldots , n { - } 1 \}$ and a corresponding incoming edge of type r into $\boldsymbol { x } = ( v , t _ { j } )$ : for $\star = \mathrm { g l o b }$ the edge is $( u , t _ { i } ) \to ( v , t _ { j } )$ , and for $\star =$ loc the edge is $( u , t _ { j } ) \to ( v , t _ { j } )$ but it is labeled by the witness snapshot i via $r = j - i$ . Hence each temporal neighbor $( u , t _ { i } )$ belongs to exactly one lag class r, and the per-lag incoming neighborhoods form a disjoint (multi)set partition of the TGNN temporal neighborhood:

$$
N ( v , t _ { j } ) \ = \ \biguplus _ { r } N _ { r } ^ { + } ( x ) .
$$

Moreover, if $( u , t _ { i } ) \in N _ { r } ^ { + } ( x )$ is the (unique) witness for lag $r = j - i$ , then the realized time gap used by the TGNN is fixed on that block:

$$
\delta ( ( u , t _ { i } ) , x ) = t _ { j } - t _ { i } = \delta _ { i j } .
$$

Define the RGNN’s per-lag incoming messages as

$$
m _ { r , + } ^ { ( l ) } ( x ) = \Omega _ { a } \Bigl ( \{ \widetilde { \Phi } ^ { ( l ) } ( \widetilde { h } _ { y } ^ { ( l ) } , \xi _ { ( y  x ) } ) \ | \ y \in N _ { r } ^ { + } ( x ) \mathbb  \}  \Bigr ) .
$$

Using $\xi _ { ( y  x ) } = \zeta ( \delta ( y , x ) ) = \zeta ( \delta _ { i j } )$ and the induction hypothesis $\widetilde { h } _ { y } ^ { ( l ) } = h _ { y } ^ { ( l ) }$ , we obtain, elementwise inside each $N _ { r } ^ { + } ( x )$

$$
\widetilde { \Phi } ^ { ( l ) } ( \widetilde { h } _ { y } ^ { ( l ) } , \xi _ { ( y  x ) } ) = \Phi ^ { ( l ) } ( \Xi ( h _ { y } ^ { ( l ) } , \zeta ( \delta _ { i j } ) ) ) .
$$

Now distinguish the two TGNN cases:

• Global (Eq. (4)). Here $\boldsymbol { y } = \left( \boldsymbol { u } , t _ { i } \right)$ is precisely the embedding source used by the TGNN, so

$$
m _ { r , + } ^ { ( l ) } ( \boldsymbol { x } ) = \Omega _ { a } \Bigl ( \bigl \{ \Phi ^ { ( l ) } ( \Xi ( h _ { ( u , t _ { i } ) } ^ { ( l ) } , \zeta ( \delta _ { i j } ) ) ) \ \mathrm { ~ \Bigl | ~ \ } ( u , t _ { i } ) \in N _ { r } ^ { + } ( x ) \bigl \} \Bigr ) .
$$

• Local $( E q . \ ( 4 ) )$ . By construction of $K _ { \mathrm { l o c } } ( T G ) , N _ { r } ^ { + } ( x )$ contains the current-time counterparts $( u , t _ { j } )$ ; hence

$$
\begin{array} { r } { m _ { r , + } ^ { ( l ) } ( x ) = \Omega _ { a } \Big ( \big \{ \updownarrow \Phi ^ { ( l ) } ( \Xi ( h _ { ( u , t _ { j } ) } ^ { ( l ) } , \zeta ( \delta _ { i j } ) ) ) ~ \Big | ~ ( u , t _ { i } ) \in N _ { r } ^ { + } ( x ) \big \} \Big ) . } \end{array}
$$

Summing over lags and using the disjoint union $N ( v , t _ { j } ) = \lvert \pm \rvert _ { r } N _ { r } ^ { + } ( x )$ yields

$$
\underbrace  \Omega _ { a } \Big ( \big \{ \phi ^ { ( l ) } ( \Xi ( \cdot , \zeta ( \delta _ { i j } ) ) ) \big \} \Big | ( u , t _ { i } ) \in N ( v , t _ { j } ) \big \} _ { \mathrm { T G N N } } = \underbrace { \Omega _ { a } \Big ( \big \lvert \dddot { \xi } \big \{ m _ { r , + } ^ { ( l ) } ( x ) \big \} \Big ) } _ { \mathrm { R G N N } \mathrm { c o m b i n e d i n c o m i n g } } ,
$$

i.e., the TGNN aggregate $m _ { t _ { j } } ^ { ( l ) } ( v )$ equals the RGNN’s combined incoming per-lag message at $x .$

Finally, both models apply the same $( \Omega _ { c } , \Gamma ^ { ( l ) } )$ to $\{ \} h _ { ( v , t _ { j } ) } ^ { ( l ) } \}$ and the above aggregate, hence

$$
h _ { ( v , t _ { j } ) } ^ { ( l + 1 ) } = \Gamma ^ { ( l ) } \Big ( \Omega _ { c } ( \{ \{ h _ { ( v , t _ { j } ) } ^ { ( l ) } \} \} \cup \{ \{ m _ { t _ { j } } ^ { ( l ) } ( v ) \} \} ) \Big ) = \Gamma ^ { ( l ) } \Big ( \Omega _ { c } ( \{ \{ h _ { ( v , t _ { j } ) } ^ { ( l ) } \} \} \cup \bigcup \{ + \} \{ \{ m _ { r , + } ^ { ( l ) } ( x ) \} \} ) \Big ) = \widetilde { h } _ { ( v , t _ { j } ) } ^ { ( l + 1 ) } .
$$

This proves the induction step and concludes the proof.

## E. Optimization of Expressive Lottery Tickets

We now discuss the impact of the expressivity of sparse RGNN initializations with fixed pruning masks on a fixed and finite dataset on their optimization behavior assuming standard small-step weight updates. Throughout, we use the directed multi-relational RGNN layer from Eqs. (1)–(3) with: (i) single-layer linear maps (i.e., without bias) plus nonlinearity for all message and combine functions $\Phi _ { r , \pm } ^ { ( l ) }$ and $\Gamma ^ { ( l ) }$ , (ii) a real-analytic activation σ that is continuously differentiable, injective, satisfies $\sigma ( 0 ) = 0$ , and $\sigma ^ { \prime } ( x ) \neq 0$ for all $x ,$ (iii) sum aggregation/combination as in Section 2, (iv) a fixed finite dataset $\mathcal { D }$ of finite input graphs with labels $y .$ We focus on graph classification for clarity; node-level tasks are analogous.

Let $\Theta$ denote the collection of all trainable RGNN parameters, including all weights of the message maps $\Phi _ { r , \pm } ^ { ( l ) }$ and the combine maps $\Gamma ^ { ( l ) }$ . Let $\Theta _ { k }$ denote the collection at optimization step k. We index individual parameters by $\Theta _ { k , i }$ . A mask M is a binary tensor of the same shape as Θ, and a sparse initialization is given by

$$
\begin{array} { r } { \Theta _ { 0 } = M \odot \widetilde { \Theta } _ { 0 } , } \end{array}
$$

for some base initialization $\widetilde { \Theta } _ { 0 } .$ , with surviving parameter indices supp $( M ) = \{ i \mid M _ { i } = 1 \}$ . Let $L \in \mathbb { N }$ denote the network depth. A mask M is expressive for D up to depth L if there exists a parameter setting Θb with $\operatorname { s u p p } ( { \widehat { \Theta } } ) \subseteq \operatorname { s u p p } ( M )$ such that for all $G _ { a } , G _ { b } \in \mathcal { D }$

$$
G _ { a } \not \to _ { \mathrm { R W L } ^ { ( L ) } } G _ { b } \Longrightarrow \Psi _ { \widehat { \Theta } } ^ { ( L ) } ( G _ { a } ) \not \ = \Psi _ { \widehat { \Theta } } ^ { ( L ) } ( G _ { b } ) ,
$$

i.e., the masked RGNN matches the L-round 1-RWL distinguishing power on D. An expressive mask M is irreducible if for every $i \in \operatorname { s u p p } ( M )$ , the mask $M ^ { ( - i ) }$ obtained by setting $M _ { i } = 0$ is not expressive for D up to depth L. Irreducibility captures the intuition of a “minimal expressive ticket”: every surviving parameter is needed (on D) to maintain the desired expressivity.

Let $y ( G )$ denote the label of graph $G \in { \mathcal { D } }$ . Relational 1-WL induces an equivalence relation $\equiv _ { \mathrm { R W L } } ( L )$ on graphs via $L$ refinement steps. Within our RGNN class, $\equiv _ { \mathrm { R W L } } ( L )$ is the fundamental upper bound on what can be distinguished. We are interested only in those WL-distinctions that are label-relevant on D. A pair $( G _ { a } , G _ { b } ) \in \mathcal { D } ^ { 2 }$ with $G _ { a } \neq G _ { b }$ is task-relevant (for depth L) if

$$
G _ { a } \not \equiv _ { \mathrm { R W L } ^ { ( L ) } } G _ { b } \quad \mathrm { a n d } \quad y ( G _ { a } ) \not = y ( G _ { b } ) .
$$

That is, the architecture can distinguish $G _ { a } , G _ { b }$ at depth $L ,$ and the labels require it.

Let M be a binary mask over the parameters of an L-layer RGNN. We call M task-expressive for $( \mathcal { D } , y )$ up to depth L if there exists a parameter setting $\widehat { \Theta }$ with $\operatorname { s u p p } ( { \widehat { \Theta } } ) \subseteq \operatorname { s u p p } ( M )$ such that for all task-relevant pairs $\left( G _ { a } , G _ { b } \right)$ ,

$$
\Psi _ { \widehat { \Theta } } ^ { ( L ) } ( G _ { a } ) \neq \Psi _ { \widehat { \Theta } } ^ { ( L ) } ( G _ { b } ) .
$$

In other words: the masked network realizes all label-relevant distinctions that are representable by L-round 1-RWL. A task-expressive mask M is irreducible if for every $i \in \operatorname { s u p p } ( M )$ , the mask $M ^ { ( - i ) }$ obtained by setting $M _ { i } = 0$ is not task-expressive for $( \mathcal { D } , y )$ up to depth L. Irreducibility in this context means: every remaining parameter is needed to realize some required label-relevant distinction on $\mathcal { D }$ (given the architectural 1-RWL limit).

We next formalize when a surviving parameter structurally participates in the supervised computation. Let $\mathcal { L } ( \Theta )$ be the empirical loss on $( \mathcal { D } , y )$ under parameters $\Theta ;$

$$
\mathcal { L } ( \Theta ) \ : = \ : \frac { 1 } { | D | } \sum _ { G \in \mathcal { D } } \ell ( \Psi _ { \Theta } ^ { ( L ) } ( G ) , y ( G ) ) ,
$$

with some per-graph loss function $\ell ( \cdot , \cdot ) ( \mathrm { e . g . }$ ., cross-entropy).

Let M be a mask and consider an RGNN with parameters supported on M and loss $\mathcal { L }$ on $( \mathcal { D } , y )$ . A parameter index $i \in$ supp(M) is gradient-connected (w.r.t. $( \mathcal { D } , y ) )$ if there exists some $G \in { \mathcal { D } }$ and some output coordinate j such that:

(i) the forward computation from $G$ to that output depends on $\Theta _ { k , i }$ via a path of operations through the weights of some $\Phi _ { r , \pm } ^ { ( l ) }$ and/or $\Gamma ^ { ( l ) }$

(ii) along this path, all local derivatives (including $\sigma ^ { \prime } )$ are nonzero on the realized inputs.

Equivalently, $\partial \ell ( \Psi _ { \Theta _ { k } } ^ { ( L ) } ( G ) , y ( G ) ) / \partial \Psi _ { \Theta _ { k , i } } ^ { ( L ) } ( G ) _ { j } \neq 0$ for some $G \in { \mathcal { D } }$ and $j .$ . A non-connected parameter is structurally dead for this task on $\mathcal { D } \colon$ changing it never affects any prediction on D.

Using the RGNN equations and the chain rule, we now spell out the explicit gradients for the linear $+ \sigma$ blocks $\Gamma ^ { ( l ) }$ and $\Phi _ { r , \pm } ^ { ( l ) }$ in the directed multi-relational case (Eqs. (1)–(3)), with sum aggregation/combination<sup>3</sup>.

We parameterize

$$
\Phi _ { r , \pm } ^ { ( l ) } ( h ) = \sigma ( W _ { r , \pm } ^ { ( l ) } h ) , \qquad \Gamma ^ { ( l ) } ( x ) = \sigma ( W _ { \Gamma } ^ { ( l ) } x ) ,\tag{11}
$$

where $\sigma$ is injective, continuously differentiable, satisfies $\sigma ( 0 ) = 0$ and $\sigma ^ { \prime } ( x ) \neq 0$ for all x, and $W _ { r , \pm } ^ { ( l ) } , W _ { \Gamma } ^ { ( l ) } \in \mathbb { R } ^ { d \times d }$ Instantiating $\Omega _ { a }$ as the sum, for each $v \in V$ and $r \in \mathcal { R }$ , the per-relation messages are

$$
m _ { r , - } ^ { ( l ) } ( v ) = \sum _ { u \in N _ { r } ^ { - } ( v ) } \Phi _ { r , - } ^ { ( l ) } ( h _ { u } ^ { ( l ) } ) , \qquad m _ { r , + } ^ { ( l ) } ( v ) = \sum _ { u \in N _ { r } ^ { + } ( v ) } \Phi _ { r , + } ^ { ( l ) } ( h _ { u } ^ { ( l ) } ) .\tag{12}
$$

Instantiating $\Omega _ { c }$ as the sum, the input to $\Gamma ^ { ( l ) }$ is

$$
a _ { v } ^ { ( l ) } = h _ { v } ^ { ( l ) } + \sum _ { r \in \mathcal { R } } m _ { r , - } ^ { ( l ) } ( v ) + \sum _ { r \in \mathcal { R } } m _ { r , + } ^ { ( l ) } ( v ) ,\tag{13}
$$

and

$$
z _ { v } ^ { ( l + 1 ) } = W _ { \Gamma } ^ { ( l ) } a _ { v } ^ { ( l ) } , \qquad h _ { v } ^ { ( l + 1 ) } = \sigma ( z _ { v } ^ { ( l + 1 ) } ) .\tag{14}
$$

Let $\mathcal { L }$ be the training loss on the fixed finite dataset D. All sums below range over all graphs in $\mathcal { D }$ and their nodes.

We define the local backpropagation factor at layer l+1:

$$
\delta _ { v } ^ { ( l + 1 ) } : = \frac { \partial \mathcal { L } } { \partial z _ { v } ^ { ( l + 1 ) } } = \frac { \partial \mathcal { L } } { \partial h _ { v } ^ { ( l + 1 ) } } \odot \sigma ^ { \prime } ( z _ { v } ^ { ( l + 1 ) } ) .\tag{15}
$$

Gradients for $\Gamma ^ { ( l ) }$ . By the chain rule

$$
\frac { \partial \mathcal { L } } { \partial W _ { \Gamma } ^ { ( l ) } } = \sum _ { v } \delta _ { v } ^ { ( l + 1 ) } \left( a _ { v } ^ { ( l ) } \right) ^ { \top } .\tag{16}
$$

The gradient w.r.t. $a _ { v } ^ { ( l ) }$ is

$$
\frac { \partial \mathcal { L } } { \partial a _ { v } ^ { ( l ) } } = { ( W _ { \Gamma } ^ { ( l ) } ) } ^ { \top } \delta _ { v } ^ { ( l + 1 ) } \ = : \ g _ { v } ^ { ( l ) } .\tag{17}
$$

Since $a _ { v } ^ { ( l ) }$ is a sum of $h _ { v } ^ { ( l ) }$ and all $m _ { r , \pm } ^ { ( l ) } ( v )$ (see (13)), each summand receives the same contribution:

$$
\frac { \partial \mathcal { L } } { \partial m _ { r , + } ^ { ( l ) } ( v ) } = g _ { v } ^ { ( l ) } , \qquad \frac { \partial \mathcal { L } } { \partial m _ { r , - } ^ { ( l ) } ( v ) } = g _ { v } ^ { ( l ) } ,\tag{18}
$$

and $g _ { v } ^ { ( l ) }$ also contributes to $\partial \mathcal { L } / \partial h _ { v } ^ { ( l ) }$ via the self-term.

Gradients for $\Phi _ { r , + } ^ { ( l ) }$ . For each r and u, define

$$
z _ { r , + } ^ { ( l ) } ( u ) = W _ { r , + } ^ { ( l ) } h _ { u } ^ { ( l ) } , \qquad \phi _ { r , + } ^ { ( l ) } ( u ) = \Phi _ { r , + } ^ { ( l ) } ( h _ { u } ^ { ( l ) } ) = \sigma ( z _ { r , + } ^ { ( l ) } ( u ) ) .\tag{19}
$$

Since $\begin{array} { r } { m _ { r , + } ^ { ( l ) } ( v ) = \sum _ { u \in N _ { r } ^ { + } ( v ) } \phi _ { r , + } ^ { ( l ) } ( u ) } \end{array}$ , we obtain

$$
\frac { \partial \mathcal { L } } { \partial \phi _ { r , + } ^ { ( l ) } ( u ) } = \sum _ { v : u \in N _ { r } ^ { + } ( v ) } \frac { \partial \mathcal { L } } { \partial m _ { r , + } ^ { ( l ) } ( v ) } = \sum _ { v : u \in N _ { r } ^ { + } ( v ) } g _ { v } ^ { ( l ) } .\tag{20}
$$

We define

$$
\delta _ { r , + } ^ { ( l ) } ( u ) : = \frac { \partial \mathcal { L } } { \partial z _ { r , + } ^ { ( l ) } ( u ) } = \frac { \partial \mathcal { L } } { \partial \phi _ { r , + } ^ { ( l ) } ( u ) } \odot \sigma ^ { \prime } ( z _ { r , + } ^ { ( l ) } ( u ) ) ,\tag{21}
$$

which is nonzero whenever the upstream terms are nonzero, since $\sigma ^ { \prime } ( x ) \neq 0$ . Then

$$
\frac { \partial \mathcal { L } } { \partial W _ { r , + } ^ { ( l ) } } = \sum _ { u } \delta _ { r , + } ^ { ( l ) } ( u ) \left( h _ { u } ^ { ( l ) } \right) ^ { \top } .\tag{22}
$$

Gradients for $\Phi _ { r , - } ^ { ( l ) }$ . The derivation is analogous for the incoming-direction block. For each r and $u ,$

$$
z _ { r , - } ^ { ( l ) } ( u ) = W _ { r , - } ^ { ( l ) } h _ { u } ^ { ( l ) } , \phi _ { r , - } ^ { ( l ) } ( u ) = \sigma ( z _ { r , - } ^ { ( l ) } ( u ) ) ,\tag{23}
$$

and since $\begin{array} { r } { m _ { r , - } ^ { ( l ) } ( v ) = \sum _ { u \in N _ { r } ^ { - } ( v ) } \phi _ { r , - } ^ { ( l ) } ( u ) } \end{array}$

$$
\frac { \partial \mathcal { L } } { \partial \phi _ { r , - } ^ { ( l ) } ( u ) } = \sum _ { v : u \in N _ { r } ^ { - } ( v ) } \frac { \partial \mathcal { L } } { \partial m _ { r , - } ^ { ( l ) } ( v ) } = \sum _ { v : u \in N _ { r } ^ { - } ( v ) } g _ { v } ^ { ( l ) } ,\tag{24}
$$

$$
\delta _ { r , - } ^ { ( l ) } ( u ) : = \frac { \partial \mathcal { L } } { \partial z _ { r , - } ^ { ( l ) } ( u ) } = \frac { \partial \mathcal { L } } { \partial \phi _ { r , - } ^ { ( l ) } ( u ) } \odot \sigma ^ { \prime } ( z _ { r , - } ^ { ( l ) } ( u ) ) ,\tag{25}
$$

and thus

$$
\frac { \partial \mathcal { L } } { \partial W _ { r , - } ^ { ( l ) } } = \sum _ { u } \delta _ { r , - } ^ { ( l ) } ( u ) \left( h _ { u } ^ { ( l ) } \right) ^ { \top } .\tag{26}
$$

These explicit expressions specialize the backpropagation rules to the RGNN blocks $\Gamma ^ { ( l ) }$ and $\Phi _ { r , \pm } ^ { ( l ) }$ with sum aggregation, and are exactly the gradients used in our optimizability analysis.

Lemma E.1. Let M be a mask and let $i \in \operatorname { s u p p } ( M )$ be gradient-connectedfor the RGNN described in $E q s .$ . (11)–(26). Then, for almost all initializations $\Theta _ { 0 }$ supported on M, andfor any nonzero loss gradient at the network outputs on some $G \in { \mathcal { D } }$ , we have

$$
\frac { \partial \mathcal { L } } { \partial \Theta _ { 0 , i } } ( \Theta _ { 0 } ) \neq 0 .
$$

Proof. By construction of the RGNN layer, every trainable parameter in Θ belongs to one of the matrices $W _ { \Gamma } ^ { ( l ) }$ or $W _ { r , \pm } ^ { ( l ) }$ We show that for a gradient-connected parameter, the corresponding partial derivative is a nontrivial smooth function of $\Theta$ hence nonzero almost everywhere.

Case 1: $\Theta _ { i }$ is an entry of $W _ { \Gamma } ^ { ( l ) }$

From (16), each entry of $\partial \mathcal { L } / \partial W _ { \Gamma } ^ { ( l ) }$ is of the form

$$
\Bigl [ \frac { \partial \mathcal { L } } { \partial W _ { \Gamma } ^ { ( l ) } } \Bigr ] _ { p q } = \sum _ { v } \delta _ { v } ^ { ( l + 1 ) } [ p ] a _ { v } ^ { ( l ) } [ q ] ,
$$

where $\delta _ { v } ^ { ( l + 1 ) }$ is defined in (15) and $a _ { v } ^ { ( l ) }$ in (13). Gradient-connectedness of $\Theta _ { i }$ means there exists at least one node v and output coordinate such that: $\mathrm { ( i ) } a _ { v } ^ { ( l ) } [ q ]$ ] depends (through the RGNN computation) on earlier features along a path using $\Theta _ { i }$ and (ii) the upstream factor $\delta _ { v } ^ { ( l + 1 ) } [ \bar { p } ]$ receives nonzero signal from the loss whenever the output loss gradient is nonzero. All involved mappings are compositions of linear transformations and $\sigma$ with $\sigma ^ { \prime } ( x ) \neq 0$ , hence are smooth and locally invertible along the path.

Consequently, the function $\Theta \mapsto [ \partial \mathcal { L } / \partial W _ { \Gamma } ^ { ( l ) } ] _ { p q }$ is not identically zero on the parameter space restricted by M: at some admissible parameter setting, the corresponding summand $\delta _ { v } ^ { ( l + 1 ) } [ p ] a _ { v } ^ { ( l ) } [ q ]$ is nonzero. A nontrivial real-analytic (indeed smooth) function has a zero set of Lebesgue measure zero. Thus for almost all $\Theta _ { 0 }$ supported on $M , [ \partial \mathcal { L } / \partial W _ { \Gamma } ^ { ( l ) } ] _ { p q } ( \Theta _ { 0 } ) \neq 0$

Case 2: $\Theta _ { i }$ is an entry of $W _ { r , + } ^ { ( l ) }$

From (22),

$$
\Big [ \frac { \partial \mathcal { L } } { \partial W _ { r , + } ^ { ( l ) } } \Big ] _ { p q } = \sum _ { u } \delta _ { r , + } ^ { ( l ) } ( u ) [ p ] h _ { u } ^ { ( l ) } [ q ] ,
$$

with $\delta _ { r , + } ^ { ( l ) } ( u )$ given by (21). Each $\delta _ { r , + } ^ { ( l ) } ( u )$ is obtained from upstream gradients $g _ { v } ^ { ( l ) }$ (see (17), (20)) multiplied elementwise by $\sigma ^ { \prime } ( z _ { r , + } ^ { ( l ) } ( u ) )$ ), which is never zero. If $\Theta _ { 0 , \ast }$ <sub>i</sub> is gradient-connected, there exists at least one path from $\Theta _ { 0 , i }$ to some output coordinate through $h _ { u } ^ { ( l ) } \mapsto z _ { r , + } ^ { ( l ) } ( u ) \mapsto \phi _ { r , + } ^ { ( l ) } ( u ) \mapsto m _ { r , + } ^ { ( l ) } ( v ) \mapsto a _ { v } ^ { ( l ) } \mapsto h ^ { ( l + 1 ) } \mapsto \cdot \cdot . .$ that is used in the computation on some $G \in { \mathcal { D } }$ . Along this path, all local derivatives are nonzero by assumption, and the loss provides a nonzero upstream signal. Therefore the corresponding summand $\delta _ { r , + } ^ { ( l ) } ( u ) [ p ] h _ { u } ^ { ( l ) } [ q ]$ is a non-constant smooth function of Θ and is not identically zero. As in Case 1, its zero set has measure zero, hence for almost all $\Theta _ { 0 }$

$$
\Big [ \frac { \partial \mathcal { L } } { \partial W _ { r , + } ^ { ( l ) } } \Big ] _ { p q } ( \Theta _ { 0 } ) \neq 0 .
$$

Case 3: $\Theta _ { i }$ is an entry of $W _ { r , - } ^ { ( l ) }$

This is identical to Case 2, using Eqs. (23)–(26): for a gradient-connected entry, at least one term $\delta _ { r , - } ^ { ( l ) } ( u ) [ p ] h _ { u } ^ { ( l ) } [ q ]$ is a nontrivial smooth function of Θ, hence nonzero for almost all $\Theta _ { 0 }$

In all cases, if i is gradient-connected, the map $\Theta \mapsto \partial \mathcal { L } / \partial \Theta _ { \mathrm { i } }$ <sub>i</sub> is not identically zero on the parameter space compatible with M. Thus its zero set is a measure-zero subset, and for almost all initializations $\Theta _ { 0 }$ supported on M we obtain $\partial \mathcal { L } / \partial \Theta _ { 0 , i } ( \Theta _ { 0 } ) \neq 0$ whenever the loss gradient at the outputs is nonzero. □

A sparse initialization $\Theta _ { 0 }$ with mask M is optimizable on $( \mathcal { D } , y )$ if:

(i) every $i \in \operatorname { s u p p } ( M )$ is gradient-connected w.r.t. $( \mathcal { D } , y )$

(ii) for almost all $\Theta _ { 0 }$ supported on M, all surviving parameters receive nonzero gradient at initialization, i.e., $\frac { \partial \mathcal { L } } { \partial \Theta _ { 0 , i } } ( \Theta _ { 0 } ) \neq 0$ whenever the loss gradient at the outputs is nonzero,

(iii) thus small gradient steps from $\Theta _ { 0 }$ can affect all task-relevant directions permitted by M (no structurally dead subblocks).

Proposition E.2. Let M be an irreducible task-expressive mask for $( \mathcal { D } , y )$ up to depth L in the RGNN architecture of Eqs. $( 1 1 ) ‐ ( 2 6 )$ . Then every $i \in \operatorname { s u p p } ( M )$ is gradient-connected w.r.t. $( \mathcal { D } , y )$ . Consequently, for almost all initializations $\Theta _ { 0 }$ supported on $M , \Theta _ { 0 }$ is optimizable on $( \mathcal { D } , y )$ .

Proof. Suppose, for contradiction, that there exists $i \in \mathrm { s u p p } ( M )$ that is not gradient-connected. By definition, “not gradient-connected” is equivalent to

$$
{ \frac { \partial \Psi _ { \Theta } ^ { ( L ) } ( G ) } { \partial \Theta _ { 0 , i } } } \equiv 0 \quad { \mathrm { f o r ~ a l l ~ } } G \in { \mathcal { D } } ,
$$

i.e., the RGNN outputs on $\mathcal { D }$ are independent of $\Theta _ { 0 , i }$ . Because the loss $\mathcal { L }$ is computed from these outputs, the chain rule together with the explicit gradients (16), (22), (26) implies

$$
\frac { \partial \mathcal { L } } { \partial \Theta _ { 0 , i } } ( \Theta ) \equiv 0 \quad \mathrm { f o r ~ a l l ~ p a r a m e t e r ~ s e t t i n g s } \Theta \ : \mathrm { s u p p o r t e d ~ o n } \ : M .
$$

Equivalently, varying $\Theta _ { 0 , i }$ while keeping all other coordinates fixed never changes any prediction on D.

Now consider the reduced mask $M ^ { ( - i ) }$ obtained from M by setting its i-th entry to zero. For any $\Theta$ supported on $M ,$ define $\Theta ^ { \prime }$ supported on $M ^ { ( - i ) }$ by

$$
\Theta _ { 0 , j } ^ { \prime } = { \left\{ \begin{array} { l l } { \Theta _ { 0 , j } , } & { j \neq i , } \\ { 0 , } & { j = i . } \end{array} \right. }
$$

Since the outputs on D are independent of $\Theta _ { i } ,$ , we have

$$
\Psi _ { \Theta } ^ { ( L ) } ( G ) = \Psi _ { \Theta ^ { \prime } } ^ { ( L ) } ( G ) \quad \mathrm { f o r \ a l l } \ G \in { \mathcal { D } } .
$$

Thus $M ^ { ( - i ) }$ can realize exactly the same functions on $( \mathcal { D } , y )$ as $M .$

Because M is task-expressive, there exists some $\widehat { \Theta }$ supported on M that realizes all required label-relevant distinctions By the above argument, zeroing out $\widehat { \Theta } _ { 0 , i }$ yields a parameter ${ \widehat { \Theta } } ^ { \prime }$ supported on $M ^ { ( - i ) }$ that induces the same predictions on $\mathcal { D }$ and hence $\bar { M } ^ { ( - i ) }$ is also task-expressive. This contradicts irreducibility , which requires that removing any $i \in \operatorname { s u p p } ( M )$ destroys task-expressivity.

Therefore, every $i \in \operatorname { s u p p } ( M )$ must be gradient-connected.

Finally, by Lemma E.1, for each gradient-connected $i \in \operatorname { s u p p } ( M )$ the partial derivative $\partial \mathcal { L } / \partial \Theta _ { 0 , i }$ is nonzero for almost all initializations $\Theta _ { 0 }$ supported on $M$ whenever there is a nonzero loss gradient at the outputs on some $G \in { \mathcal { D } }$ . Since this holds for all surviving parameters, $\Theta _ { 0 }$ is optimizable on $( \mathcal { D } , y )$ □

Proposition E.3. Fix depth L and a sparsity level s. There exist masks $M ^ { \prime }$ with $| \mathrm { s u p p } ( M ^ { \prime } ) | = s$ such that:

(a) $M ^ { \prime }$ is not task-expressive $f o r \left( \mathcal { D } , y \right)$ up to depth L, i.e., it fails to separate some task-relevant pair $\left( G _ { a } , G _ { b } \right)$

(b) some $i \in \operatorname { s u p p } ( M ^ { \prime } )$ is not gradient-connected w.r.t. $( \mathcal { D } , y )$ , so $\frac { \partial \mathcal { L } } { \partial \Theta _ { k , i } } ( \Theta ) = 0$ for all Θ and all training steps.

Thus $M ^ { \prime }$ is strictly dominated by an irreducible task-expressive mask: it is worse both in terms ofexpressivityfor the task and in terms of optimizability.

Proof. Start from an irreducible task-expressive mask M and:

(i) remove some nonzeros on paths (through parameters of $\Phi _ { r , \pm } ^ { ( l ) } \mathrm { o r } \Gamma ^ { ( l ) } )$ needed to separate a task-relevant pair $\left( G _ { a } , G _ { b } \right)$ and

(ii) reassign the same number of nonzeros to parameters that only affect intermediate features which are structurally disconnected from the loss. Concretely, pick a feature channel (or neuron) whose entire set of outgoing weights to subsequent layers and to the readout is pruned by $M ^ { \prime } ,$ so that no computational path from this channel can reach a labeled output on D. Place the new nonzeros on the incoming weights into this channel. By construction, these parameters lie in supp(M<sup>′</sup>) but any change to them can never influence the network outputs on $\mathcal { D } ,$ hence $\frac { \partial \mathcal { L } } { \partial \Theta _ { k , i } } ( \Theta ) = 0$ for all Θ.

The resulting mask $M ^ { \prime }$ has the same sparsity but cannot realize all label-relevant distinctions anymore, so it is not taskexpressive. By construction, the newly added parameters never lie on a path to the loss and hence are not gradient-connected, with identically zero gradients. □

## E.1. Persistence Under Training

We now extend Proposition E.2 from initialization (k=0) to the whole optimization trajectory under small-step gradient-based updates with a fixed mask M. We write a generic first-order update as

$$
\Theta _ { k + 1 } = \Theta _ { k } + U _ { k } ( \Theta _ { k } , \nabla { \mathcal { L } } ( \Theta _ { k } ) ) ,\tag{27}
$$

where $U _ { k }$ is continuous in its arguments and satisfies $\| U _ { k } ( \Theta _ { k } , \nabla { \mathcal { L } } ( \Theta _ { k } ) ) \| \le \kappa _ { k } \| \nabla { \mathcal { L } } ( \Theta _ { k } ) \|$ for some stepsize $\kappa _ { k } > 0 ( { \mathrm e . g } .$ , gradient descent, SGD, momentum, Adam with bounded step). The mask M remains fixed throughout.

We begin by showing that, under our construction, parameter gradients that are non-zero at a point $\Theta _ { k }$ remain non-zero in a local neighbourhood of $\Theta _ { k }$ in parameter space:

Lemma E.4. $F i x \left( \mathcal { D } , y \right)$ and the RGNN block structure in Eqs. (11)–(26). Suppose that at iteration k the output loss gradient is nonzero and for all $i \in \operatorname { s u p p } ( M )$ we have $\frac { \partial \mathcal { L } } { \partial \Theta _ { k , i } } ( \Theta _ { k } ) \overline { { { \Big ) } } } \neq 0 .$ . Then there exists a radius $\omega _ { k } > 0$ such that for every Θ with $\lVert \Theta - \Theta _ { k } \rVert < \omega _ { k }$

$$
\frac { \partial \mathcal { L } } { \partial \Theta _ { i } } ( \Theta ) \neq 0 f o r a l l i \in \mathrm { s u p p } ( M ) .
$$

Proof. For each surviving parameter index $i \in \operatorname { s u p p } ( M )$ define the scalar function

$$
g _ { i } ( \Theta ) : = \frac { \partial \mathcal { L } } { \partial \Theta _ { i } } ( \Theta ) .
$$

By construction of the RGNN layer, all forward computations (14), (19), (23) are obtained from the parameters Θ by finitely many compositions of

• linear maps (matrix–vector products, sums), and

• the activation σ, which is real-analytic and hence $C ^ { 1 }$

together with finite sums over nodes and relations. The explicit gradient expressions (16), (17), (22), (26) are built from these quantities using again only sums, products, and $\sigma ^ { \prime } .$ All of these operations are continuous, and finite compositions and sums of continuous functions are continuous. Therefore, for each i, the map $g _ { i } : \Theta \mapsto \partial \mathcal { L } / \partial \Theta _ { i } ( \Theta )$ is continuous on the parameter space (restricted by the mask M).

Next, fix some $i \in \operatorname { s u p p } ( M )$ . By assumption, $\begin{array} { r } { g _ { i } ( \Theta _ { k } ) = \frac { \partial \mathcal { L } } { \partial \Theta _ { i } } ( \Theta _ { k } ) \neq 0 } \end{array}$ . Define

$$
\varepsilon _ { i } : = \frac { 1 } { 2 } | g _ { i } ( \Theta _ { k } ) | > 0 .
$$

Using the standard definition of the continuity of a function, by continuity of $g _ { i }$ at $\Theta _ { k }$ , there exists a radius $\omega _ { k , i } > 0$ such that

$$
\begin{array} { r } { \| \Theta - \Theta _ { k } \| < \omega _ { k , i } \quad \Longrightarrow \quad | g _ { i } ( \Theta ) - g _ { i } ( \Theta _ { k } ) | < \varepsilon _ { i } . } \end{array}
$$

For any such Θ we have

$$
\begin{array} { r } { \lvert g _ { i } ( \Theta ) \rvert \ge \lvert g _ { i } ( \Theta _ { k } ) \rvert - \lvert g _ { i } ( \Theta ) - g _ { i } ( \Theta _ { k } ) \rvert > \lvert g _ { i } ( \Theta _ { k } ) \rvert - \varepsilon _ { i } = \varepsilon _ { i } > 0 . } \end{array}
$$

Hence $g _ { i } ( \Theta ) \neq 0$ for all Θ in the ball $B ( \Theta _ { k } , \omega _ { k , i } )$

The mask support $\operatorname { s u p p } ( M )$ is finite, so we have only finitely many radii $\{ \omega _ { k , i } \} _ { i \in \mathrm { s u p p } ( M ) }$ . Define

$$
\omega _ { k } : = \operatorname* { m i n } _ { i \in \mathrm { s u p p } ( M ) } \omega _ { k , i } > 0 .
$$

$\mathrm { I f ~ } \| \Theta - \Theta _ { k } \| < \omega _ { k }$ , then $\lVert \Theta - \Theta _ { k } \rVert < \omega _ { k , i }$ <sub>i</sub> for every $i \in \mathrm { s u p p } ( M )$ , and by the previous step each $g _ { i } ( \Theta )$ is nonzero. Equivalently,

$$
\frac { \partial \mathcal { L } } { \partial \Theta _ { i } } ( \Theta ) = g _ { i } ( \Theta ) \neq 0 \quad \mathrm { f o r } \mathrm { a l l } i \in \mathrm { s u p p } ( M ) ,
$$

whenever $\lVert \Theta - \Theta _ { k } \rVert < \omega _ { k }$ . This is exactly the desired statement.

Proposition E.5. Let M be an irreducible task-expressive mask for $( \mathcal { D } , y )$ up to depth L for the RGNN of Eqs. (11)–(26). Assume trainingfollows (27) with afixed mask M. Then,for almost all initializations $\Theta _ { 0 }$ supported on M, thefollowing holds:

(a) (Base case) At $k { = } 0 , \Theta _ { 0 }$ is optimizable on $( \mathcal { D } , y )$ (Proposition E.2).

(b) (Inductive step) Suppose at some $k \geq 0$ the output loss gradient is nonzero on the current batch and $\Theta _ { k }$ is optimizable. Let $\omega _ { k } > 0$ be as in Lemma E.4. Ifthe update satisfies $\lVert \Theta _ { k + 1 } - \Theta _ { k } \rVert < \omega _ { k }$ , then $\Theta _ { k + 1 }$ is also optimizable.

Consequently,for anyfinite horizon $K \in \mathbb N$ there exists a stepsize schedule $\left\{ \eta _ { 0 } , \dots , \eta _ { K - 1 } \right\}$ such that $\Theta _ { k }$ is optimizable for all $k \leq K ,$ , provided the output loss gradient is nonzero at those iterations.

Proof. (a) is exactly Proposition E.2.

For (b), assume $\Theta _ { k }$ is optimizable. Then, by the definition of optimizability and the fixed-mask architecture, every $i \in \mathrm { s u p p } ( M )$ remains gradient-connected at all iterations: gradient-connectedness depends only on the existence of structural paths through $\Phi _ { r , \pm } ^ { ( l ) }$ and $\Gamma ^ { ( l ) }$ together with $\sigma ^ { \prime } ( x ) \neq 0$ , not on the specific numerical values of $\Theta _ { k }$ . By the definition of optimizability and Lemma $\begin{array} { r } { \mathrm { E } . 1 , \frac { \partial \mathcal { L } } { \partial \Theta _ { k , i } } ( \Theta _ { k } ) \neq 0 } \end{array}$ for all surviving i, given a nonzero output loss gradient. Lemma E.4 yields a radius $\omega _ { k } > 0$ within which these partial derivatives remain nonzero for all $i \in \operatorname { s u p p } ( M )$

If the update obeys $\lVert \Theta _ { k + 1 } - \Theta _ { k } \rVert < \omega _ { k }$ , then $\frac { \partial \mathcal { L } } { \partial \Theta _ { k + 1 , i } } \big ( \Theta _ { k + 1 } \big ) \neq 0$ for all surviving i whenever the output loss gradient is nonzero at iteration k+1. Thus items (i) and (ii) of the definition of optimizability hold at $k { \pm } 1$ , so $\Theta _ { k + 1 }$ is optimizable. By induction, repeating this argument up to any finite K proves the consequence. □

## F. Computational cost model

For an MLP of depth M (number of hidden layers), width m (hidden dimension), and unstructured sparsity level $\rho \in$ $[ 0 , 1 ]$ applied to $n _ { \mathrm { i n } }$ distinct inputs (e.g., nodes or branch inputs), we roughly approximate the per-forward MADDS as $\mathrm { M A D D } _ { \mathrm { M L P } } \left( n _ { \mathrm { i n } } , m , M , \rho \right) = n _ { \mathrm { i n } } \cdot M \cdot \left( 1 - \rho \right) m ^ { 2 } \approx N _ { \mathrm { m a x } } \cdot M \cdot \left( 1 - \rho \right) m _ { \mathrm { m i n } } ^ { 2 } .$ Here $m ^ { 2 }$ counts the weights in a dense $m \times m$ layer, $( 1 - \rho ) m ^ { 2 }$ is the number of surviving weights after pruning, and the factor $n _ { \mathrm { i n } }$ accounts for applying the MLP to all inputs.

## G. Toy dataset and RGNN architecture

This appendix details the exact RGNN architecture and the synthetic multi-relational dataset generator used for Figure 2.

Synthetic multi-relational dataset. Each graph is a directed multi-relational graph $G = ( V , \{ E _ { r } \} _ { r \in \mathcal { R } } )$ , with $| V | = N$ nodes and $| { \mathcal { R } } | = R$ relations. We construct $E _ { r }$ by first sampling an undirected Erdos–Rényi edge set˝ $\tilde { E } _ { r } \subseteq \{ \{ u , v \} \mid$ $u < v , u , v \in V \}$ where each unordered pair is included independently with probability $p _ { r }$ . To avoid degenerate empty relations, if $\tilde { E } _ { r } ~ = ~ \varnothing$ we add one random undirected edge $\{ u , v \}$ with $u \ne v .$ . We then convert each undirected edge to two directed edges, i.e., $E _ { r } \ : = \ \{ ( u , v ) , ( v , u ) \ | \ \{ u , v \} \in \tilde { E _ { r } } \}$ . To encourage structural diversity among generated graphs, we reject duplicates under a cheap signature heuristic: for each relation r, we compute the (out-)degree sequence $( \deg _ { r } ( v ) ) _ { v \in V } , \quad \deg _ { r } ( v ) : = | \{ u \in V \mid ( v , u ) \in E _ { r } \} |$ , sort it, and concatenate it with the undirected edge count $| \tilde { E } _ { r } | . \mathrm { ~ A ~ }$ candidate graph is accepted only if its signature has not been seen before. Unless stated otherwise, we generate $G _ { \mathrm { s e t } } = 3 0$ graphs with $N = 6$ nodes and $R = 3$ relations using probabilities $( p _ { r } ) _ { r = 1 } ^ { R } = ( 0 . 1 2 , 0 . 2 0 , 0 . 2 8 )$ and a fixed random seed.

Node features. We use simple structural node features derived from degrees:

$$
x _ { v } : = [ 1 , \deg _ { r _ { 1 } } ( v ) , \deg _ { r _ { 2 } } ( v ) , \ldots , \deg _ { r _ { R } } ( v ) ] ^ { \top } \in \mathbb { R } ^ { 1 + R } .\tag{28}
$$

Thus the input dimension is $d _ { \mathrm { i n } } = 1 + R .$

RGNN architecture. We instantiate a minimal directed multi-relational RGNN with L layers, hidden width D, and activation $\sigma = \operatorname { t a n h }$ . All linear maps are bias-free. First, we project node features to hidden states:

$$
h _ { v } ^ { ( 0 ) } = \sigma ( W _ { \mathrm { i n } } x _ { v } ) , \qquad W _ { \mathrm { i n } } \in \mathbb { R } ^ { d \times d _ { \mathrm { i n } } } .\tag{29}
$$

For each layer $\ell \in \{ 0 , \ldots , L - 1 \}$ and relation $r \in \mathcal { R }$ , we compute a (directed) sum aggregation

$$
a _ { v , r } ^ { ( \ell ) } = \sum _ { \left( u , v \right) \in E _ { r } } h _ { u } ^ { ( \ell ) } \in \mathbb { R } ^ { d } ,\tag{30}
$$

apply a relation-specific linear map, and pass through tanh:

$$
m _ { v , r } ^ { ( \ell ) } = \sigma \mathopen { } \mathclose \bgroup \left( W _ { r } ^ { ( \ell ) } a _ { v , r } ^ { ( \ell ) } \aftergroup \egroup \right) , \qquad W _ { r } ^ { ( \ell ) } \in \mathbb { R } ^ { d \times d } .\tag{31}
$$

We then combine the current state with all relation messages via summation (rather than concatenation) and apply a shared combine transform:

$$
s _ { v } ^ { ( \ell ) } = h _ { v } ^ { ( \ell ) } + \sum _ { r \in \mathcal { R } } m _ { v , r } ^ { ( \ell ) } , \qquad h _ { v } ^ { ( \ell + 1 ) } = \sigma \Big ( W _ { \mathrm { c } } ^ { ( \ell ) } s _ { v } ^ { ( \ell ) } \Big ) ,\tag{32}
$$

with $W _ { \mathrm { c } } ^ { ( \ell ) } \in \mathbb { R } ^ { d \times d }$ . Finally, we produce a graph-level embedding by sum readout:

$$
e ( G ) = \sum _ { v \in V } h _ { v } ^ { ( L ) } \in \mathbb { R } ^ { d } .\tag{33}
$$

In our experiments we use $d = 1 6 \mathrm { o r } 8$ and $L = 3$ unless stated otherwise, and we evaluate embeddings without training (random initialization as in standard nn.Linear defaults).

## H. Assumptions and Limitations

Assumptions. Our results are stated and proved under the following assumptions, which are either inherent to the setting (finite datasets) or technically convenient for establishing dataset-conditional injectivity guarantees under random pruning.

All expressivity and probability guarantees are dataset-conditional: we fix a finite dataset D of finite (relational or temporal) graphs. This makes the sets of MLP inputs actually witnessed on D finite and allows us to bound non-injectivity events via combinatorial arguments.

The expressivity target is L-round 1-RWL (and its temporal lift via the corresponding knowledge-graph construction). Concretely, the theorems guarantee that, with probability at least $\gamma > 0$ over a random mask, a pruned network preserves the ability to separate all pairs in D that are separated by L rounds of the corresponding 1-RWL refinement.

We analyze random pruning at initialization via a Bernoulli mask $M \sim B _ { \rho }$ (entries independently set to zero with probability ρ). We additionally assume a bounded base initialization $( \mathbf { e . g . } , \Theta _ { 0 } \sim \mathcal { U } _ { c } ^ { d } )$ and define the sparse initialization by $\widehat { \Theta } _ { 0 } = M \odot \Theta _ { 0 }$

Initial node labels are encoded by an injective map Λ, ensuring label collisions are not introduced at input.

The RWL-to-RGNN correspondence relies on injectivity of the multiset encoders and update maps on the finite domain witnessed on D. The probability lower bounds are expressed in terms of (i) the maximum number $N _ { \mathrm { m a x } }$ of distinct witnessed

MLP inputs, (ii) the minimum $\ell _ { 0 } { \mathrm { - s e p a r a t i o l } }$ n $s _ { \mathrm { m i n } }$ between distinct witnessed inputs, and (iii) the minimum hidden width $m _ { \mathrm { m i n } }$ across the relevant MLP blocks.

For all $\mathbf { M L P s } \left( \mathbf { e . g . , } \Gamma ^ { ( l ) } \right.$ and, in the optimization discussion, also $\Phi _ { r , \pm } ^ { ( l ) } )$ , we assume a real-analytic, injective, continuously differentiable, zero-fixing activation σ with a nowhere-zero derivative $( \mathrm { i . e . , } \sigma ( 0 ) = 0$ and $\sigma ^ { \prime } ( x ) \neq 0$ for all x). These assumptions are technical and simplify the analysis. They hold for smooth monotone activations such as tanh and shifted sigmoids $( \mathbf { e . g . , } x \mapsto \mathrm { s i g m o i d } ( x ) - \mathrm { s i g m o i d } ( 0 )$ , which enforces $\sigma ( 0 ) = 0 )$ but exclude non-smooth activations such as ReLU. Extending the theory to non-smooth activations is conceptually possible, but would require different lemmata and additional case distinctions, which we consider future work beyond the scope of the current paper.

In Appendix E we focus on the RGNN block structure induced by Eqs. (1)–(3) with single-layer linear maps (no bias) plus nonlinearity for all message and combine functions, sum aggregation/combination, a fixed finite dataset $( \mathcal { D } , y )$ , and small-step first-order updates with a fixed mask.

Limitations and scope. The above assumptions enable clean, explicit probability bounds and a unifying reduction viewpoint, but they also delimit what we claim.

Theorems certify preservation of RWL-separations on a fixed finite dataset; they do not constitute a population-level or distributional generalization guarantee.

The guarantee is with respect to L-round 1-RWL (and the temporal analogue). It does not address distinctions beyond this baseline (e.g., higher-order WL tests) and does not, by itself, guarantee the best possible expected test-set performance for a given task.

The probability lower bounds are derived for independent Bernoulli pruning at initialization (and independence across MLP blocks when composing across layers). Structured pruning (e.g., magnitude-based, channel-wise) is outside the formal guarantee unless separately analyzed.

The lower bound γ<sub>RGNN</sub> is derived via per-block non-injectivity bounds and union-bounding; as a result it can be conservative, particularly near the critical sparsity/width regime where expressivity transitions from holding to failing. Empirically, it tightens again in the high- and low-success regimes.

The certificate depends on $N _ { \mathrm { m a x } }$ and $s _ { \mathrm { m i n } }$ (witnessed input counts and separations). These quantities capture dataset/architecture interaction, but they can be large or difficult to control a priori; consequently, the bound may be pessimistic even when expressive subnetworks exist.

Appendix E establishes non-degeneracy/gradient-connectivity properties and local persistence of nonzero gradients under small-step updates for fixed masks; it is not a global convergence result.