# Pair-Centric Graph Rewiring for Over-Squashing via Optimal Transport-Guided Communication Alignment

Yan Wang and Chuan-Xian Ren

Abstract—Message-passing neural networks (MPNNs) often struggle when task-relevant information is distributed across distant regions of a graph, since local propagation must compress remote signals through limited structural interfaces. Graph rewiring provides a structural response to over-squashing. Most existing methods rely on edge-level bottleneck scores or graphlevel connectivity surrogates. With a limited rewiring budget, the key question is which pairwise communications most need structural support. This paper proposes PairAlign, a paircentric graph rewiring framework that makes this question explicit through demand-support shortage. Specifically, PairAlign combines original-graph structural demand with current-graph finite-hop propagation support; their ratio highlights interactions whose communication demand is poorly supported by topology, and our theory shows that this score provides a computable proxy for the corresponding Jacobian-based shortage with a pair-level interpretation of over-squashing. Our theory reveals a two-sided effect of edge insertion: a new edge can create useful walks and simultaneously dilute existing normalized transition mass. Guided by this observation, PairAlign optimizes shortage to favor edge additions that alleviate over-squashing. Beyond selecting useful additions, PairAlign further introduces an Optimal Transport-guided rewiring mechanism to coordinate the finite edge budget for pair-level structural compatibility and shortage-target coverage. It formulates communication alignment between the candidate edge budget and the shortage targets, and the theory shows that this allocation covers shortage targets more broadly and effectively than a greedy-local assignment. Experiments on standard graph benchmarks show PairAlign’s improvement across message-passing backbones, validating pairlevel repair as an effective route for alleviating over-squashing.

Index Terms—Graph rewiring; Over-squashing; Pairwise communication alignment; Optimal Transport

## I. INTRODUCTION

M <sup>PNNS</sup> <sup>have</sup> <sup>become</sup> <sup>a</sup> <sup>standard</sup> <sup>framework</sup> <sup>for</sup> <sup>learning</sup>on graph-structured data because they represent nodes, on graph-structured data because they represent nodes, edges, and whole graphs through iterative local propagation over graph connectivity [1], [2]. The effectiveness of an MPNN depends on whether the input topology can support the transfer of task-relevant information. When useful evidence is distributed across distant regions of a graph, it must pass through successive local neighborhoods and be compressed into fixed-width hidden states. The graph can then become a communication bottleneck, with many remote signals forced through a small number of structural interfaces and consequently attenuated or lost. This over-squashing phenomenon has made graph rewiring a natural structural response, since the difficulty is governed not only by model depth but also by how the graph distributes communication support across node interactions [3]–[5].

![](images/e5a7b248f6fb339ebc9bcb60b09e7fbcce28e9f52ffa11badfbe0cb3210538e7.jpg)  
Fig. 1. Different rewiring targets in graph rewiring. Under the same graph structure, graph-level rewiring targets global connectivity, edge-level rewiring focuses on local bottleneck repair, and PairAlign assigns rewiring support to communication-limited node pairs.

Existing rewiring methods intervene on this problem from several directions. Curvature-based and edge-centered methods use local geometric or topological cues to locate bottlenecks and modify structurally unfavorable regions [4], [6]. Related local or path-aware variants enlarge the propagation structure through positional or shortest-path-based modifications [7], [8]. A second group evaluates rewiring through graph-level communication surrogates such as spectral expansion, effective resistance, graph expansion, or differentiable global objectives [5], [9]–[11]. Fig. 1 summarizes these two rewiring targets: graph-level methods improve aggregate connectivity, while edge-level methods repair local bottleneck structures. More recent formulations further incorporate structural priors such as locality, community structure, feature similarity, or homophily guidance into the rewiring process [12]–[16]. These directions have broadened the design space of graph rewiring, yet they also leave a practical allocation problem under limited budgets: structural support must be assigned to the node-pair communications that need it most. This problem is not fully resolved by graph-level connectivity measures or edge-level bottleneck scores, because the scarce budget must ultimately serve interactions between node pairs. Without an explicit pairlevel notion of communication shortage, rewiring can improve aggregate connectivity while leaving the most constrained pairwise communications only indirectly supported. This can weaken the effect of rewiring on the pairwise interactions most affected by over-squashing.

To address this gap, we propose PairAlign, a graph rewiring framework that makes pairwise communication shortage the basic object of structural repair. PairAlign follows the pairlevel view in Fig. 1 and builds on the Jacobian view by prioritizing node pairs whose structural communication demand is high while their finite-hop propagation support remains limited. To make this view operational, PairAlign defines shortage as the ratio between original-graph structural demand and current-graph finite-hop propagation support. This score highlights interactions whose communication demand is poorly supported by the current topology. Our theory shows that this shortage score provides a computable proxy for the corresponding Jacobian-based shortage with a pair-level interpretation of over-squashing.

To turn pair-level diagnosis into edge additions, PairAlign scores candidate edges by their support effect on high-shortage pairs. A useful candidate edge need not directly connect the target pair; its value depends on the support improvement it provides to shortage pairs. Our theory shows that the gain from edge insertion is not always positive: a new edge can create useful walks, while normalization reduces the transition mass assigned to existing neighbors. Thus, PairAlign recomputes shortage during rewiring and selects edges by their current shortage reduction. To coordinate the limited budget, PairAlign introduces an OT-guided rewiring mechanism to allocate candidate-edge mass across shortage targets. This keeps the budget from concentrating on a few locally cheap pairs and coordinates allocation according to the shortage distribution over graph-wide targets. Accordingly, the main contributions of this work are summarized as follows.

The paper introduces a pair-centric formulation of graph rewiring based on communication shortage. The shortage score ranks node pairs by the lack of current-graph support relative to the original-graph structural demand. The theory shows the formulation makes Jacobian-based shortage tractable through graph structure and yields an interpretable pair-level diagnostic for over-squashing.

• PairAlign is proposed as an OT-guided graph rewiring framework for shortage-aware edge addition. Candidate edges are scored by current shortage reduction, and the OT-guided allocation coordinates the limited edge budget across shortage targets. This yields a globally coordinated rewiring strategy and avoids concentrating additions on only a few targets.

• We validate PairAlign on standard graph benchmarks across message-passing backbones. The results and mechanism studies show that PairAlign can effectively repair the communication shortage and improve performance. This supports pair-level repair as an effective route for alleviating over-squashing.

The remainder of the paper is organized as follows. Section II reviews graph rewiring methods for over-squashing. Section III develops PairAlign from the shortage formulation to OT-guided edge allocation. Section IV evaluates the method and analyzes its rewiring mechanism. Section V concludes the paper and discusses future directions.

## II. RELATED WORK

Graph rewiring has become a central structural strategy for mitigating over-squashing in MPNNs. Over-squashing arises when the input topology concentrates multiple information flows through restricted structural interfaces, causing useful signals to be compressed into fixed-width node representations [3], [4]. This phenomenon links the effectiveness of message passing to graph topology, including curvature, expansion, information contraction, and effective resistance properties [5], [10], [12]. Expander-based propagation further shows that bottleneck-resistant computational structures can improve message passing without relying only on the original input topology [17]. Recent surveys further organize over-squashing mitigation into rewiring, spectral, curvaturebased, and propagation-oriented families, confirming graph rewiring as a sustained research direction in MPNN design [18]. Related spatial and path-aware interventions expand the computational neighborhood through positional encodings, shortest-path message passing, or discrete geometric rewiring, providing additional ways to improve propagation beyond immediate graph adjacency [7], [8], [19]. In this setting, the main question is how a modified computational graph should distribute propagation support under a finite structural budget.

A major line of rewiring methods diagnoses over-squashing through local geometry and edge-level bottlenecks. SDRF uses graph curvature to locate structurally critical regions for rewiring, making curvature one of the earliest operational proxies for topology-induced over-squashing [4]. BORF extends this perspective through Ollivier-Ricci curvature and connects negative curvature with over-squashing and positive curvature with over-smoothing, yielding a unified localgeometric rewriting principle [6]. Recent studies have refined this view by examining the empirical behavior of curvatureguided rewiring and by proposing more expressive curvature variants for detecting over-squashed edges [20], [21]. This family establishes a strong local diagnostic foundation for graph rewiring, and it motivates a complementary question concerning which node-pair communications should receive structural support after local bottlenecks have been detected.

Another line of work formulates rewiring through graphlevel communication surrogates. FoSR improves spectral expansion by adding sparse edges that increase the spectral gap, thereby strengthening global connectivity for message passing [9]. Effective resistance-based methods use resistance as a communication proxy and guide edge modification through graph-wide improvements in connectivity [5], [22]. DiffWire introduces a differentiable rewiring objective based on the Lovasz bound [11], and LASER studies the trade-off´ among connectivity, locality, and sparsity through localityaware sequential rewiring [12]. More recently, spectrumpreserving sparsification has strengthened this family by improving connectivity while retaining sparsity and Laplacian spectral structure [23]. Cayley graph propagation extends the expander-based perspective by propagating over complete Cayley graph structures to obtain bottleneck-free computational templates [24]. These methods provide mature graph level principles for structural improvement, and they set the stage for rewiring objectives that explicitly organize the budget around a distribution of underserved node-pair interactions.

Recent rewiring studies have also incorporated task-relevant signals beyond purely topological criteria. ComFy uses community structure and feature similarity to guide rewiring toward stronger label-community alignment [13]. JDR jointly rewires the graph and denoises node features by aligning the leading spectral spaces of graph and feature matrices [14]. In heterophilic graphs, DHGR modifies neighborhoods by adding homophilic edges and pruning heterophilic ones, and labelguided rewiring further uses label information to improve the compatibility between graph structure and downstream classification [15], [16]. These studies show that rewiring objectives benefit from structural signals that better match the role of the graph in downstream prediction.

PairAlign keeps the criterion topology-driven and uses topology to define a pair-level shortage signal that brings node-pair communication needs into the rewiring objective. It frames rewiring as pair-centric structural repair, where the target is a communication-deficient node pair whose demand exceeds the support available under the current topology. Its OT-guided allocation assigns candidate additions to highshortage pairs under the finite budget, turning graph rewiring into a budgeted allocation problem for the pairwise communications most constrained by over-squashing.

## III. METHODS

## A. Preliminaries

Let $G = ( V , E , \mathbf { H } )$ be a graph with node set V, edge set $E ,$ and node feature matrix $\mathbf { H } \in \mathbb { R } ^ { n \times d }$ , where $n = | V |$ . Set $\mathbf { H } ^ { ( 0 ) } = \mathbf { H } .$ and let $h _ { u } ^ { ( 0 ) }$ denote its u-th row. The adjacency matrix of G is denoted by $\mathbf { A } \in \{ 0 , 1 \} ^ { n \times n }$ . For any $u , v \in V$ let $d _ { G } ( u , v )$ denote the shortest-path distance on the original graph. We consider an add-only rewiring setting. Let E<sup>¯</sup> denote the set of candidate non-edges of G. A rewiring solution is represented by a matrix $\mathbf { B } \in \mathbb { R } _ { \geq 0 } ^ { n \times n }$ . Its feasible set is

$$
{ \mathcal { B } } : = \left\{ \mathbf { B } : \mathbf { B } = \mathbf { B } ^ { \top } , ~ \mathrm { d i a g } ( \mathbf { B } ) = \mathbf { 0 } , ~ \mathrm { s u p p } ( \mathbf { B } ) \subseteq { \bar { E } } \right\} .\tag{1}
$$

For budgeted allocation, let $\tau$ be the finite target-pair set, let $\mathcal { E } \subseteq \bar { E }$ be the finite candidate-edge pool, and let $\mathbf { p } \in \Delta ( \mathcal T )$ be the shortage-target distribution used by the allocation. The added-budget distribution induced by the edge-addition variables is denoted by $\mathbf { q } ( \mathbf { B } ) \in \Delta ( \mathcal { E } )$ and is used as the source marginal over candidate added edges.

The rewired graph is encoded by $\mathbf { W } = \mathbf { A } + \mathbf { B }$ . Let D be the degree matrix of W. The row-normalized propagation matrix P on the rewired graph is defined as $\mathbf { P } = \mathbf { D } ^ { - 1 } \mathbf { W }$ Let $\mathcal { P } ~ = ~ \{ ( u , v ) ~ \in ~ V \times V ~ : ~ u ~ \neq ~ v \}$ denote the set of ordered node pairs. Unless explicitly specified, all pairwise quantities in the sequel are indexed over P, and all distancebased constraints are defined on the original graph G. At layer $r ,$ a row-normalized MPNN updates node $j$ by

$$
h _ { j } ^ { ( r ) } = \phi _ { r } \left( h _ { j } ^ { ( r - 1 ) } , \sum _ { i } P _ { i j } \psi _ { r } ( h _ { i } ^ { ( r - 1 ) } ) \right) ,\tag{2}
$$

where $\psi _ { r }$ is the message map and $\phi _ { r }$ is the update map.

B. Pairwise Communication Shortage as a Demand–Support View of Over-Squashing

Motivation. Existing graph rewiring studies usually evaluate structural modifications through edge-level bottlenecks, local geometric indicators, or graph-level connectivity surrogates. These criteria provide useful diagnoses of unfavorable graph structures. Under a limited rewiring budget, the central decision is which node-pair interactions should receive structural support. From the perspective of over-squashing, the constrained object is the communication between distant node pairs whose information exchange must pass through limited multi-hop propagation pathways. This observation motivates a pair-centric view that allocates structural budget to node pairs whose communication demand is poorly supported by the current graph. We introduce pairwise communication shortage as a demand-support diagnostic for over-squashing risk and use it to prioritize the node-pair interactions that should be supported by rewiring.

Formulation. To make the pair-centric view operational, PairAlign scores over-squashing risk by comparing structural demand with propagation support for each node pair. The demand term is fixed on the original graph and reflects how much structural communication the pair requires. The support term is measured on the current computational graph and records how much finite-hop propagation is available to that pair. A large shortage marks a pair whose communication demand receives limited support from the current topology. We use this score as a rewiring diagnostic for over-squashing risk and first define the two quantities that enter it: pairwise communication demand and propagation support.

Definition 1 (Pairwise Communication Demand). Let $\mathcal { P }$ denote the set of ordered node pairs. The pairwise communication demand is the function $\omega : \mathcal { P }  \mathbb { R } _ { > 0 }$ defined by

$$
\omega ( u , v ) = d _ { G } ( u , v ) ^ { p } , \qquad ( u , v ) \in \mathcal { P } ,\tag{3}
$$

where $d _ { G } ( u , v )$ is the shortest-path distance between u and v on G, and $p > 0$ controls the growth rate of communication demand with respect to distance.

This definition captures the structural side of pairwise communication demand. By assigning greater demand weights to distant node pairs, it brings to the foreground those interactions that rely more heavily on long-range information exchange; under multi-hop propagation, such interactions typically traverse longer transmission chains and are more vulnerable to compression caused by bottlenecks and information attenuation [3], [4]. In this sense, the shortest-path distance provides a natural topology-based prior for long-range pairwise communication demand.

We now define the support side of the demand-support diagnostic. For a finite-depth message passing model, communication from u to v can only use propagation walks whose length is within the effective depth of the model. PairAlign measures support by the finite-hop propagation mass induced by the current computational graph.

Definition 2 (Pairwise Propagation Support). Let W be a graph with propagation matrix P, let $\mathcal { P }$ denote the set of ordered node pairs, and let $\alpha _ { 1 } , \ldots , \alpha _ { K } \in \mathbb { R } _ { \ge 0 }$ be nonnegative hop weights satisfying $\textstyle \sum _ { \ell = 1 } ^ { K } \alpha _ { \ell } = 1$ . The finite-hop pairwise propagation support on W is the function $s ( \mathbf { \cdot } ; \mathbf { W } ) : \mathcal { P } \to$ $\mathbb { R } _ { \geq 0 }$ defined by

$$
s ( u , v ; \mathbf { W } ) = \sum _ { \ell = 1 } ^ { K } \alpha _ { \ell } \left[ \mathbf { P } ^ { \ell } \right] _ { u v } , \qquad ( u , v ) \in \mathcal { P } .\tag{4}
$$

In the main method, PairAlign uses the uniform hop average $\alpha _ { \ell } = 1 / K$ . Under this convention, $s ( u , v ; \mathbf { W } )$ measures the average propagation mass that can travel from u to v within K message-passing steps. The following lemma gives the corresponding Jacobian bound.

Lemma 1 (Support bounds normalized Jacobian influence). Consider the K-layer MPNN defined as Eq. (2), and let $h _ { v } ^ { ( \acute { \ell } ) }$ be the representation of node v at layer $\ell = 0 , \ldots , K .$ Assume that $\psi _ { r }$ is Lipschitz and $\phi _ { r }$ is Lipschitz in its propagatedmessage argument at each layer $r = 1 , \ldots , K$ , and let $L _ { r }$ denote the corresponding layer Lipschitz constant. For every ordered pair $( u , v ) \in \mathcal { P }$ , the support $s ( u , v ; \mathbf { W } )$ upper bounds the normalized finite-depth Jacobian influence $\scriptstyle { \mathcal { T } } ( u , v )$ as

$$
\mathcal { T } ( u , v ) = \frac { 1 } { K } \sum _ { \ell = 1 } ^ { K } \frac { \Big \| \partial h _ { v } ^ { ( \ell ) } / \partial h _ { u } ^ { ( 0 ) } \Big \| } { \prod _ { r = 1 } ^ { \ell } L _ { r } } \leq s ( u , v ; \mathbf { W } ) .
$$

At the mechanism level, over-squashing appears as weak influence between distant nodes after finite-depth message passing. This influence reflects the graph structure together with the MPNN model functions. When the model functions are Lipschitz, Lemma 1 isolates the structural side and shows that the finite-hop propagation mass $s ( u , v ; \mathbf { W } )$ is a structural upper bound on the normalized Jacobian influence.

For a pair with large structural demand $\omega ( u , v )$ , a small value of $s ( u , v ; \mathbf { W } )$ means that the current graph leaves little finite-depth influence available for this pairwise message passing, which indicates a higher risk of over-squashing. The shortage score serves as a diagnostic to record this mismatch between structural demand and support on the graph topology. Overall, PairAlign ranks the resulting communication-deficient pairs by the ratio below.

Definition 3 (Pairwise Communication Shortage). For each ordered node pair $( u , v ) \in \mathcal { P } _ { : }$ , let $\boldsymbol { \omega } ( u , v )$ denote its pairwise communication demand in Eq. (3) and let $s ( u , v ; \mathbf { W } )$ denote its finite-hop pairwise propagation support in Eq. (4). The pairwise communication shortage on W is defined as the function $\begin{array} { r } { S ( \cdot ; \mathbf { W } ) : \mathcal { P }  \mathbb { R } _ { \ge 0 } } \end{array}$ given by

$$
\mathcal { S } ( u , v ; \mathbf { W } ) = \frac { \omega ( u , v ) } { s ( u , v ; \mathbf { W } ) + \varepsilon } , \qquad ( u , v ) \in \mathcal { P } ,\tag{5}
$$

where $\varepsilon > 0$ is a small positive constant.

For fixed $( u , v )$ , we abbreviate $s ( \mathbf { W } ) \ = \ s ( u , v ; \mathbf { W } )$ and ${ \cal S } ( { \bf W } ) \ : = \ : { \cal S } ( u , v ; { \bf W } )$ . Under a fixed demand specification and propagation protocol, S(W) ranks node pairs by how much structural demand remains unsupported by finite-hop propagation; larger values indicate pairs that should receive higher priority in shortage-aware rewiring. The Jacobian view of over-squashing relates the problem to weak finite-depth influence between nodes. In graph rewiring, weak influence alone is not enough to rank repair priorities, since some weakly influenced pairs may require only limited structural communication. A pair with large structural demand and weak normalized Jacobian influence reflects a more severe communication shortage. We measure this shortage by the ratio between structural demand and normalized Jacobian influence, defined below as the Jacobian-based shortage. Proposition 1 shows that the computable shortage used by PairAlign serves as a reasonable proxy for this Jacobian-based ratio.

Proposition 1 (Proxy of Jacobian-based shortage). For a node pair $( u , v )$ , suppose that the assumptions of Lemma 1 hold. Define the Jacobian-based communication shortage by

$$
\mathcal { S } ^ { \mathrm { J a c } } ( \mathbf { W } ) = \frac { \omega ( u , v ) } {  { \mathcal { T } } ( u , v ) + \varepsilon } ,
$$

where $\scriptstyle { \mathcal { T } } ( u , v )$ is the normalized Jacobian influence in Lemma 1. Then the following statements hold.

(i) Bottleneck lower bound. $\begin{array} { r } { \mathcal { S } ( \mathbf { W } ) \leq { S } ^ { \mathrm { J a c } } ( \mathbf { W } ) } \end{array}$

(ii) Constant-factor equivalence in the non-degenerate regime. Assume that the normalized Jacobian influence from u to v retains a c fraction of $s ( \mathbf { W } )$ , i.e., ${ \mathcal { T } } ( u , v ) \geq c s ( \mathbf { W } )$ with $c \in \mathsf { \Gamma } ( 0 , 1 ] . \mathsf { \Gamma } S ( \mathbf { W } )$ and ${ \mathcal { S } } ^ { \mathrm { J a c } } ( \mathbf { W } )$ satisfy the following constant-factor equivalence:

$$
\mathcal { S } ( \mathbf { W } ) \leq \mathcal { S } ^ { \mathrm { J a c } } ( \mathbf { W } ) \leq c ^ { - 1 } \mathcal { S } ( \mathbf { W } ) .
$$

The non-degenerate regime gives a sufficient setting for analyzing graph rewiring as a structural way to alleviate oversquashing. It assumes that the finite-hop support supplied by the graph is not erased by the model functions, so topology changes can still improve the communication bottlenecks. With this condition, $\mathcal { S } ( \mathbf { W } )$ and $S ^ { \mathrm { J a c } } ( \mathbf { W } )$ are equivalent up to a constant factor. A large $\mathcal { S } ( \mathbf { W } )$ points to a node pair whose communication demand is high while the current graph provides limited finite-hop support. Thus, reducing the computable score $\mathcal { S } ( \mathbf { W } )$ also reduces the corresponding Jacobianbased shortage. This makes the rewiring objective focus on the pairs most vulnerable to over-squashing.

Once the shortage score selects demand–support deficient pairs, we now consider how an edge insertion changes the support of those targets. PairAlign acts on the graph through candidate edges and each edge is evaluated by its support change for a shortage target under the row-normalized propagation matrix. This effect is not monotone. Adding a new neighbor creates walks through the new edge, while row normalization reallocates probability mass away from the original neighbors of the same row. To make this local effect explicit, we consider the row update $e \ = \ a \  \ b$ induced by the undirected candidate edge $\{ a , b \}$ . For $\beta \geq 0 .$ , let $\mathbf { P } _ { \beta }$ denote the row-normalized propagation matrix obtained after adding edge weight $\beta$ to this row update, with $\mathbf { P } _ { 0 } = \mathbf { P }$ . The first-order support change $g _ { e } ( u , v )$ at the uninserted graph is

$$
g _ { e } ( u , v ) = \left. \frac { d } { d \beta } \frac { 1 } { K } \sum _ { \ell = 1 } ^ { K } \left[ \mathbf { P } _ { \beta } ^ { \ell } \right] _ { u v } \right| _ { \beta = 0 ^ { + } } .
$$

Proposition 2 (First-order support change from edge insertion). Assume that $b \notin N ( a )$ before insertion and let $d _ { a }$ be

the original row degree of a. For $e = a  b ,$ the first-order finite-hop support change of $( u , v )$ is

$$
g _ { e } ( u , v ) = { \cal B } _ { e } ( u , v ) - { \cal D } _ { e } ( u , v ) ,
$$

where the added-path contribution is

$$
B _ { e } ( u , v ) = \frac { 1 } { K } \sum _ { \ell = 1 } ^ { K } \sum _ { i = 1 } ^ { \ell } [ \mathbf { P } ^ { i } ] _ { u a } \frac { [ \mathbf { P } ^ { \ell - i } ] _ { b v } } { d _ { a } } ,
$$

and the normalization loss over the original neighbors of a is

$$
D _ { e } ( u , v ) = \frac { 1 } { K } \sum _ { \ell = 1 } ^ { K } \sum _ { i = 1 } ^ { \ell } [ \mathbf { P } ^ { i } ] _ { u a } \frac { \sum _ { z \in N ( a ) } \mathbf { P } _ { a z } [ \mathbf { P } ^ { \ell - i } ] _ { z v } } { d _ { a } } .
$$

Here $B _ { e } ( u , v )$ sums walks that reach $^ { a , }$ use the new step $a \ \to \ b ,$ and continue to v, while $D _ { e } ( u , v )$ sums the mass removed from walks that would have continued through the original neighbors of $a .$ For the undirected insertion $\{ a , b \}$ the first-order support change is the symmetrized quantity $g _ { \{ a , b \} } ( u , v ) = g _ { a  b } ( u , v ) + g _ { b  a } ( u , v )$

Proposition 2 shows that an inserted edge does not simply add propagation support to a target pair. The support increases only when the added-path contribution exceeds the loss, namely $B _ { e } ( u , v ) \ > \ D _ { e } ( u , v )$ . An edge is useful only when it lowers shortage on the rewiring graph, not merely when it increases connectivity. This view is consistent with evidence that more connectivity is not always preferable [25]. PairAlign keeps optimizing shortage during rewiring to ensure the effectiveness of edge insertions.

## C. PairAlign with OT-Guided Communication Alignment for Shortage Reduction

Under a finite edge addition budget, graph rewiring needs a global allocation rule for assigning candidate edges to highshortage pairs. This allocation must address two coupled issues. High-shortage pairs are not independent targets. Several targets may rely on overlapping communication regions, and one candidate edge may affect multiple targets at once. If edges are selected independently, as in Greedy-Local, the budget can concentrate on targets that are easier to improve locally, leaving other shortage regions undercovered. The allocation also needs to account for how well a candidate edge can serve the target pairs it is meant to support. A candidate edge should be favored only when it can provide useful propagation support for the corresponding shortage pairs.

To establish a pair-level rewiring mechanism that captures both budget coverage and structural compatibility, PairAlign formulates the communication alignment between the candi date edge budget and the shortage targets as an Optimal Transport problem, as shown in Fig. 2. Drawing on the coupling view of Optimal Transport [26], [27], OT-guided alignment treats the candidate edge budget as structural resources to be assigned and pairs with high shortage as targets that require support. The transport cost measures how suitable a candidate edge is for supporting a given shortage target. The optimal coupling gives the allocation solution by matching the finite candidate-edge budget to the shortage targets with minimum total transport cost. As a result, structural resources are preferentially assigned to node pairs that both need support and can be effectively served by the selected candidates.

Formulation of OT-guided Alignment. Under this view, PairAlign aligns the added-budget distribution ${ \bf q } ( { \bf B } )$ with the shortage distribution p on the original graph under a transport cost C that encodes their structural compatibility. The corresponding OT problem is defined as Eq. (6) and the optimal coupling $\mathbf { T } ^ { \star }$ characterizes the cost-compatible allocation,

$$
\mathbf { \Gamma } \mathbf { \Gamma } ^ { \star } = \arg \operatorname* { m i n } _ { \mathbf { \Gamma } \Gamma \in \Pi ( \mathbf { q } ( \mathbf { B } ) , \mathbf { p } ) } \mathbf { \langle \mathbf { \Gamma } \mathbf { \Gamma } \mathbf { C } \rangle } - \varepsilon H ( \mathbf { \Gamma } \mathbf { \Gamma } ) ,\tag{6}
$$

where $\Pi ( \mathbf { q } ( \mathbf { B } ) , \mathbf { p } ) = \{ \Gamma _ { i j } \geq 0 , \forall i , j \mid \Gamma \mathbf { 1 } = \mathbf { q } ( \mathbf { B } ) , \ \Gamma ^ { \top } \mathbf { 1 } =$ $\mathbf { p } \}$ , and $\begin{array} { r } { H ( \mathbf { r } ) ~ = ~ - \sum _ { i , j } \Gamma _ { i j } ( \log \Gamma _ { i j } - 1 ) } \end{array}$ is the entropy regularizer. The construction of PairAlign involves three key components, namely the added-budget distribution ${ \bf q } ( { \bf B } )$ , the shortage distribution p, and the transport geometry C.

Here, q(B) denotes the distribution over candidate added edges induced by the current edge-addition variables B. It represents how the finite added structural budget is placed over candidate non-edges before alignment and is used as the source marginal in the OT problem.

To make the shortage target more concentrated, PairAlign starts from the pairwise shortage of the original graph and derives a target distribution p by centering it against the graph-wide baseline. This centering gives more weight to severe communication shortages. Specifically, the base shortage $\{ S _ { u v } \} _ { ( u , v ) \in \mathcal { P } }$ on the original graph is obtained from Eq. (5) with $\mathbf { \hat { W } } = \mathbf { A } $ . This pattern contains both graphwide background shortage and genuinely prominent shortage mass. Directly using the raw shortage pattern would spread the budget over many mildly deficient pairs and make the severe regions less visible. We use a simple construction adapted to the graph in which the original shortage is centered relative to a graph-level baseline and only the positive excess $\begin{array} { r } { \tilde { p } _ { u v } \ = \ \operatorname* { m a x } ( S _ { u v } - | \mathcal { P } | ^ { - 1 } \sum _ { ( a , b ) \in \mathcal { P } } S _ { a b } , 0 ) } \end{array}$ is retained. The corresponding shortage target distribution is then

defined by

$$
\mathbf { p } = ( p _ { u v } ) _ { ( u , v ) \in \mathcal { P } } , \quad \mathrm { w h e r e } \quad p _ { u v } = \frac { \tilde { p } _ { u v } } { \sum _ { ( a , b ) \in \mathcal { P } } \tilde { p } _ { a b } } .\tag{7}
$$

In PairAlign, the transport cost matrix C captures the structural match between candidate added edges and target shortage pairs. For any candidate added edge $e = ( a , b )$ and target pair $( u , v )$ , the cost $C _ { e , ( u , v ) }$ has two parts. Endpoint alignment measures whether the edge can connect the source side of the target pair to its target side, while bridge suitability measures whether the span of the candidate edge is commensurate with the scale of the shortage pair. Based on this design, the transport cost is defined by

$$
\begin{array} { r } { C _ { ( a , b ) , ( u , v ) } = \underbrace { \operatorname* { m i n } \{ d ( a , u ) + d ( b , v ) , d ( a , v ) + d ( b , u ) \} } _ { \mathrm { e n d p o i n t ~ a l i g n m e n t } } } \\ { - \underbrace { \lambda \psi \biggl ( \frac { d ( a , b ) } { d ( u , v ) + \varepsilon } \biggr ) } _ { \mathrm { b r i d g e ~ s u i t a b i l i t y } } , } \end{array}\tag{8}
$$

where $\lambda ~ > ~ 0$ controls the strength of the bridge reward and $\psi ( \cdot )$ is a monotone increasing function. The endpointalignment term measures how well a feasible candidate edge is positioned with respect to the two sides of a target pair. For a single target pair $( u , v )$ , the direct shortcut $( u , v )$ , if feasible, is naturally one of the most strongly aligned candidates. PairAlign does not exclude this case. At the graph level, the goal is not to add one direct shortcut for each pair with high shortage, but to select a finite set of added edges that can support many shortage targets under a limited budget. The transport cost is used to evaluate the structural fit between each feasible candidate edge and each shortage target. The bridgesuitability term complements the endpoint score by considering the scale of the candidate edge. Among candidates with similar endpoint alignment, an edge whose span covers a larger normalized portion of the original pairwise distance receives a lower cost. In this way, the matrix C provides a supportaware geometry for OT allocation. It favors candidate edges that are well positioned for a shortage pair and have a span suitable for supporting that pair under the finite edge budget. If each unit of candidate-edge budget is assigned independently to its cheapest target, the induced target marginal follows the cost-induced preference instead of the shortage distribution p. Part of the high-shortage target set can then remain uncovered. Proposition 3 formalizes the coverage advantage of shortage alignment. OT alignment addresses it by solving a minimumcost global allocation with prescribed added edge and shortage target marginals. By keeping the allocation aligned with pairs with high shortage, the selected edges provide more effective structural repair under the same budget.

![](images/d0ef0d253fbe0af6cd79ad6bea08b05bb0277b0397593a975b62dee46345b450.jpg)  
Fig. 2. PairAlign rewiring mechanism. PairAlign rewires a graph by aligning limited edge additions with pairwise communication shortages. (a) The method constructs a shortage-target distribution over communication-deficient node pairs, and defines a candidate-edge budget to represent the limited structural resources available for repair. (b) OT-guided alignment solves for a coupling between candidate edges and shortage targets under a transport cost that balances structural compatibility and target coverage. (c) The optimization loop updates candidate-edge scores, re-evaluates shortage on the current rewired graph, and feeds this signal back into edge selection. After discrete projection, the selected set $E _ { \mathrm { a d d } }$ forms $\mathbf { W } = \mathbf { A } + \mathbf { B }$ . By assigning limited rewiring capacity to communication-deficient node pairs, PairAlign strengthens message passing and alleviates over-squashing.

Proposition 3 (Coverage advantage over greedy target assignment). Let $\mathcal { E } = \{ e _ { 1 } , \dots , e _ { M } \}$ be the candidate-edge set and $\mathcal { T } = \{ t _ { 1 } , \ldots , t _ { m } \}$ be the set of shortage target pairs; write $t \in \tau$ for one target pair. Let $\mathbf { p } \in \Delta ( \mathcal { T } )$ be the shortage-target distribution, $\mathbf { q } ( \mathbf { B } ) \in \Delta ( \mathcal { E } )$ be the added-budget distribution. Let $E _ { 1 } , \ldots , E _ { n }$ be drawn independently from q(B).

For each $e \in { \mathcal { E } } ,$ define the greedy target map $g _ { C } : \mathcal { E }  \mathcal { T }$ and its induced target distribution by

$$
g _ { C } ( e ) = \arg \operatorname* { m i n } _ { t \in \mathcal { T } } C _ { e , t } , \qquad \pi _ { C } ^ { \mathbf { B } } ( t ) = \sum _ { e \in \mathcal { E } } q _ { e } ( \mathbf { B } ) \mathbf { 1 } \{ g _ { C } ( e ) = t \} .
$$

Let $\mathbf { T } ^ { g } \in \mathbb { R } _ { + } ^ { n \times m }$ be the empirical greedy coupling over the realized edge budgets, where $\Gamma _ { i , t } ^ { g } = n ^ { - 1 } \mathbf { 1 } \{ g _ { C } ( E _ { i } ) = t \}$ . Let $r _ { \mathbf { T } ^ { g } }$ be its target marginal. Define shortage-target coverage by

$$
\mathrm { C o v } _ { \mathbf { p } } ( \Gamma ) = \sum _ { t \in \mathcal { T } } \operatorname* { m i n } \{ r _ { \Gamma } ( t ) , p _ { t } \} .
$$

Let $\mathbf { \Gamma } \mathbf { \Gamma } ^ { \mathrm { { C T } } }$ be any OT coupling over the same realized budget units with target marginal $\mathbf { p } ,$ namely $\Gamma ^ { \mathrm { O T } } \in \Pi _ { n } ( \mathbf { p } )$ , where

$$
\Pi _ { n } ( \mathbf { p } ) = \{ \mathbf { r } \geq 0 : \mathbf { \Gamma } \mathbf { 1 } = n ^ { - 1 } \mathbf { 1 } , \ \mathbf { \Gamma } \mathbf { \Gamma } ^ { \top } \mathbf { 1 } = \mathbf { p } \} .
$$

Then $\mathrm { C o v } _ { \mathbf { p } } ( \Gamma ^ { \mathrm { O T } } ) = 1$ . Moreover, for any $\delta \in ( 0 , 1 )$ , with probability at least $1 - \delta$

$$
\mathrm { C o v } _ { \mathbf { p } } ( \Gamma ^ { \mathrm { O T } } ) - \mathrm { C o v } _ { \mathbf { p } } ( \Gamma ^ { g } ) \geq \mathrm { T V } ( \pi _ { C } ^ { \mathbf { B } } , \mathbf { p } ) - \frac { | \mathcal { T } | } { 2 } \sqrt { \frac { \log ( 2 | \mathcal { T } | / \delta ) } { 2 n } } .
$$

Proposition 3 explains the necessity of OT-guided allocation. Under the randomized finite-sample view, Greedy-Local maps each sampled candidate edge to its lowest cost target, so the induced target distribution $\pi _ { C } ^ { \mathbf { B } }$ may drift away from the p and leave some shortage mass uncovered. Proposition 3 shows that, with high probability, the coverage gain of OT over Greedy-Local is controlled by $\mathrm { T V } ( \pi _ { C } ^ { \mathbf { B } } , \mathbf { p } )$ up to a finitesample error term. OT introduces coordination across edge budgets by matching the shortage marginal, preventing the finite budget from collapsing onto a few targets. This coordinated allocation helps PairAlign turn limited edge additions into targeted shortage reduction and alleviate over-squashing more effectively.

Total Rewiring Objective of PairAlign. The goal of rewiring in PairAlign is to reduce pairwise communication shortage under a limited structural budget. OT-guided allocation controls how structural additions are deployed. The overall objective combines two terms. The first term measures the focused pairwise shortage on the rewired graph. Following Proposition 2, this term checks whether the selected insertions increase finite-hop support after row normalization, and encourages rewiring to reduce the residual demand–support mismatch of high-shortage pairs,

Algorithm 1 The PairAlign Rewiring   
Require: Original graph $G = ( V , E , \mathbf { H } )$ , candidate non-edge   
set ${ \bar { E } } ,$ add budget $k ,$ propagation depth K, temperature τ ,   
optimization steps $T _ { \mathrm { o p t } }$   
Ensure: Discrete added-edge set $E _ { \mathrm { a d d } }$   
1: Phase I: Construct pairwise shortage-aware targets.   
2: Compute pairwise demand $\omega _ { u v }$ for $( u , v ) \in { \mathcal { P } } ;$   
3: Compute original pairwise shortage $\boldsymbol { S _ { u v } }$ on A and con  
struct target weights $p _ { u v } ;$   
4: Initialize candidate logits $\{ \theta _ { e } \} _ { e \in \bar { E } } ;$   
5: Phase II: Optimize budgeted near-discrete selection.   
6: for $t = 1$ to $T _ { \mathrm { o p t } }$ do   
7: Compute soft edge scores $z _ { e } =$ softmax $( \pmb \theta / \tau ) _ { e } ;$   
8: Select the top-k candidates by a hard mask in the   
forward pass   
9: Form $\bar { \mathbf { W } } ^ { ( t ) } = \mathbf { A } + \mathbf { B } ^ { ( t ) } ;$   
10: Evaluate pairwise supports $s ( \mathbf { W } ^ { ( t ) } ) ;$   
11: Compute focused shortage $\dot { L _ { \mathrm { s h o r t a g e } } } ( \mathbf { W } ^ { ( t ) } ) ;$   
12: Construct ${ \bf q } _ { \mathrm { a d d } } ( { \bf B } ^ { ( t ) } )$ and evaluate $\dot { L _ { \mathrm { O T } } } ( \mathbf { B } ^ { ( t ) } ) ;$   
13: Update θ by minimizing Eq. (11) using a straight  
through style gradient approximation;   
14: end for   
15: Phase III: Project to the final discrete rewiring   
16: Rank candidate non-edges by the optimized scores;   
17: Form a projection pool from the top-ranked candidates;   
18: Select $E _ { \mathrm { a d d } }$ from the pool by re-evaluating Eq. (11) under   
the add budget;   
19: return $E _ { \mathrm { a d d } }$

$$
L _ { \mathrm { s h o r t a g e } } ( \mathbf { W } ) = \sum _ { ( u , v ) \in \mathcal { P } } p _ { u v } S _ { u v } ( \mathbf { W } ) .\tag{9}
$$

The second term is the OT-guided rewiring term introduced above. The optimal coupling $\mathbf { T } ^ { \star }$ represents how candidate added edges should be matched to shortage pairs. Minimizing the OT-guided alignment term yields an allocation with global structural compatibility, defined by

$$
L _ { \mathrm { O T } } ( { \bf B } ) = \operatorname* { m i n } _ { \Gamma \in \Gamma ( { \bf q } _ { \mathrm { a d d } } ( { \bf B } ) , { \bf p } ) } \langle \Gamma , { \bf C } \rangle - \varepsilon H ( \Gamma ) .\tag{10}
$$

In all, the total objective of PairAlign is defined by combining shortage reduction with OT-guided alignment,

$$
\operatorname* { m i n } _ { \mathbf { B } } \quad L _ { \mathrm { s h o r t a g e } } ( \mathbf { A } + \mathbf { B } ) + \lambda _ { \mathrm { O T } } L _ { \mathrm { O T } } ( \mathbf { B } ) .\tag{11}
$$

## D. Optimization and Computational Complexity

Optimization. The PairAlign rewiring induced in Eq. (11) amounts to choosing a finite subset of added edges from the candidate non-edge set E<sup>¯</sup>, which makes the optimization combinatorial. The candidate non-edge set $\bar { E }$ specifies the feasible locations for edge addition, while B is the decision matrix supported on ${ \bar { E } } ,$ with each nonzero entry $B _ { u v }$ representing the strength or selection of the added edge $( u , v )$ . Direct optimization over such discrete edge subsets is intractable in general. To address this, we introduce a continuous parameterization that yields a differentiable approximation while remaining consistent with the final discrete rewiring. In all, the PairAlign procedure in Alg. 1 consists of three stages: shortage-aware pairwise target construction, budgeted near-discrete rewiring, and final projection to the discrete added-edge set.

Specifically, Optimization is carried out over continuous logits on candidate non-edges. Let $\pmb { \theta } = \{ \theta _ { e } \} _ { e \in \bar { E } }$ denote the collection of all logits. The logits are initialized uniformly, $\mathrm { e . g . }$ $\theta _ { e } = 0$ for all candidate edges. Let $z _ { e } = \mathrm { s o f t m a x } ( \pmb { \theta } / \tau )$ <sub>e</sub> be the corresponding soft selection probability under temperature $\tau \ > \ 0$ . During optimization, under an add-only rewiring budget of k edges, a top-k hard mask is applied in the forward pass so that only the highest-scoring candidate edges contribute to the current rewiring matrix, with reduced forward computation. In the backward pass, gradients are propagated through the corresponding soft variables by a straight-through style approximation [28]. This yields a near-discrete training procedure that remains compatible with gradient-based optimization while staying close to the target discrete add-only rewiring problem. This objective implements the feedback loop in Fig. 2. Updates to B change the rewired graph $\mathbf { W } = \mathbf { A } + \mathbf { B }$ , the shortage term re-evaluates finite-hop support on the current graph, and the OT term keeps the edge budget aligned with the shortage targets.

After optimization, the learned candidate scores are converted into a discrete rewiring set through a projection step. Candidate non-edges are first ranked by their optimized scores, and a projection pool is formed from the top-ranked entries. The final rewiring set is then selected from this pool under the prescribed add budget by re-evaluating the main objective Eq. (11) after optimization. In this way, the final output is a discrete added-edge set, while the training and projection stages remain aligned under a common objective.

Computational Complexity. The computational cost of PairAlign mainly comes from truncated propagation evaluation, OT-based alignment, and candidate ranking. Let M be the number of focused target pairs. The one-time construction of the shortage target distribution is a preprocessing cost before the optimization loop. The full sorting of candidate nonedges costs $\mathcal { O } ( | \bar { E } | \log | \bar { E } | )$ per step. Dense Sinkhorn alignment over candidate edges and target pairs costs $\mathcal { O } ( T _ { \mathrm { O T } } | \bar { E } | M )$ per step [27], [29]. The overall optimization complexity is $\mathcal { O } \big ( T _ { \mathrm { o p t } } \big ( | \bar { E } | \log | \bar { E } | + T _ { \mathrm { O T } } | \bar { E } | M \big ) \big )$ . If the implementation restricts Sinkhorn alignment to the candidates retained by the hard top-k mask, the factor |E<sup>¯</sup>| in the OT term is replaced by at most k. After optimization, the final candidate scores are ranked once more, adding another $\mathcal { O } ( | \bar { E } | \log | \bar { E } | )$ cost. The final projection is performed on a small projection pool, so the exact reevaluation of feasible k-subsets remains a low-cost post-processing step. PairAlign is used as an offline rewiring preprocessor, so this cost is incurred once before downstream message-passing model training. A broader comparison with representative rewiring methods is provided in the supplementary appendix.

TABLE I  
NODE CLASSIFICATION ACCURACY (%) ON SIX DATASETS UNDER DIFFERENT REWIRING METHODS AND BACKBONES.THE AVERAGE RANKING (AR) REFLECTS THE MEAN POSITION OF EACH METHOD ACROSS ALL DATASETS.
<table><tr><td>Backbone</td><td>Rewiring</td><td>Cora</td><td>Citeseer</td><td>Texas</td><td>Cornell</td><td>Wisconsin</td><td>Chameleon</td><td></td><td>AR</td></tr><tr><td rowspan="6">GCN</td><td>None [2]</td><td> $8 6 . 7 \pm 0 . 3$ </td><td> $7 2 . 3 \pm 0 . 3$ </td><td> $4 4 . 2 \pm 1 . 5$ </td><td> $4 1 . 5 \pm 1 . 8$ </td><td> $4 4 . 6 \pm 1 . 4$ </td><td></td><td> $5 9 . 2 \pm 0 . 6$ </td><td>6.9</td></tr><tr><td>SDRF [4]</td><td> $8 6 . 3 \pm 0 . 3$ </td><td> $7 2 . 6 \pm 0 . 3$ </td><td> $4 3 . 9 \pm 1 . 6$ </td><td> $4 2 . 2 \pm 1 . 5$ </td><td> $4 6 . 2 \pm 1 . 2$ </td><td></td><td> $5 9 . 4 \pm 0 . 5$ </td><td>5.9</td></tr><tr><td>FoSR [9]</td><td> $8 5 . 9 \pm 0 . 3$ </td><td> $7 2 . 3 \pm 0 . 3$ </td><td> $4 6 . 0 \pm 1 . 6$ </td><td> $4 0 . 2 \pm 1 . 6$ </td><td> $4 8 . 3 \pm 1 . 3$ </td><td></td><td> $5 9 . 3 \pm 0 . 6$ </td><td>6.1</td></tr><tr><td>BORF [6]</td><td> $8 7 . 5 \pm 0 . 2$ </td><td> $7 3 . 8 \pm 0 . 2$ </td><td> $4 9 . 4 \pm 1 . 2$ </td><td> $5 0 . 8 \pm 1 . 1$ </td><td> $5 0 . 3 \pm 0 . 9$ </td><td></td><td> $6 1 . 5 \pm 0 . 4$ </td><td>2.8</td></tr><tr><td>GTR [5]</td><td> $\overline { { 8 7 . 3 \pm 0 . 4 } }$ </td><td> $7 2 . 4 \pm 0 . 3$ </td><td> $4 5 . 9 \pm 1 . 9$ </td><td> $5 0 . 8 \pm 1 . 6$ </td><td> $4 6 . 7 \pm 1 . 5$ </td><td></td><td> $5 7 . 6 \pm 0 . 8$ </td><td>5.0</td></tr><tr><td>LASER [12]</td><td> $8 6 . 9 \pm 1 . 1$ </td><td> $7 2 . 6 \pm 0 . 6$ </td><td> $4 5 . 9 \pm 2 . 6$ </td><td> $4 2 . 7 \pm 2 . 6$ </td><td> $4 6 . 0 \pm 2 . 6$ </td><td></td><td> $5 3 . 7 \pm 1 . 4$ </td><td>5.7</td></tr><tr><td></td><td>GOKU [23]</td><td> $8 6 . 8 \pm 0 . 3$ </td><td> $7 3 . 6 \pm 0 . 2$ </td><td> ${ \bf 6 4 . 7 \pm 2 . 4 }$ </td><td> ${ \bf 5 8 . 9 \pm 1 . 6 }$ </td><td> ${ \pm } 9 . 8 \pm 2 . 4$ </td><td> $6 3 . 2 \pm 0 . 4$ </td><td></td><td>2.2</td></tr><tr><td></td><td>PAR (Ours)</td><td> ${ \bf 8 9 . 1 \pm 0 . 4 }$ </td><td> ${ \bf 7 6 . 8 \pm 0 . 4 }$ </td><td> $5 8 . 7 \pm 1 . 6 $ </td><td> $5 1 . 7 \pm 2 . 8$ </td><td> ${ \underline { { 5 2 . 4 } } } \pm 1 . 7$ </td><td></td><td> ${ \bf 6 5 . 1 \pm 0 . 6 }$ </td><td>1.5</td></tr><tr><td rowspan="6">GIN</td><td>None [30]</td><td> $7 6 . 0 \pm 0 . 6$ </td><td> $5 9 . 3 \pm 0 . 9$ </td><td> $5 3 . 5 \pm 3 . 1$ </td><td> $3 6 . 5 \pm 2 . 2$ </td><td> $4 8 . 5 \pm 2 . 2$ </td><td></td><td> $5 8 . 1 \pm 2 . 1 $ </td><td>6.1</td></tr><tr><td>SDRF [4]</td><td> $7 4 . 9 \pm 0 . 1$ </td><td> $6 0 . 3 \pm 0 . 8$ </td><td> $5 0 . 3 \pm { 3 . 7 }$ </td><td> $4 0 . 0 \pm 2 . 1$ </td><td> $4 8 . 8 \pm 1 . 9$ </td><td></td><td> $5 8 . 4 \pm 2 . 1$ </td><td>5.7</td></tr><tr><td>FoSR [9]</td><td> $7 5 . 1 \pm 0 . 8$ </td><td> $6 1 . 7 \pm 0 . 7$ </td><td> $4 7 . 0 \pm 3 . 7$ </td><td> $3 5 . 6 \pm 2 . 4$ </td><td> $4 8 . 5 \pm 2 . 1 $ </td><td></td><td> $5 6 . 3 \pm 2 . 2$ </td><td>6.9</td></tr><tr><td>BORF [6]</td><td> $7 8 . 4 \pm 0 . 4$ </td><td> $6 3 . 1 \pm 0 . 8$ </td><td> $\underline { { 6 3 . 1 \pm 1 . 7 } }$ </td><td> $4 8 . 6 \pm 1 . 2$ </td><td> $5 4 . 9 \pm 1 . 2$ </td><td></td><td> $6 5 . 3 \pm 0 . 8$ </td><td>3.1</td></tr><tr><td>GTR [5]</td><td> $7 8 . 6 \pm 1 . 3$ </td><td> $6 2 . 6 \pm 1 . 9$ </td><td> $\overline { { 4 9 . 5 \pm 2 . 9 } }$ </td><td> $3 9 . 4 \pm 2 . 3$ </td><td> $4 4 . 2 \pm 2 . 4$ </td><td></td><td> $\overline { { 5 7 . 1 \pm 1 . 2 } }$ </td><td>5.8</td></tr><tr><td>LASER [12]</td><td> $7 9 . 1 \pm 1 . 0$ </td><td> ${ \bf 6 6 . 5 \pm 1 . 3 }$ </td><td> $4 6 . 5 \pm 4 . 5$ </td><td> $4 4 . 5 \pm 3 . 8$ </td><td> $4 6 . 1 \pm 4 . 6$ </td><td></td><td> $5 9 . 8 \pm 2 . 2$ </td><td>4.5</td></tr><tr><td></td><td>GOKU [23]</td><td> $\overline { { 7 8 . 4 \pm 0 . 5 } }$ </td><td> $6 3 . 6 \pm 1 . 3$ </td><td> $6 0 . 2 \pm 2 . 6$ </td><td> $4 9 . 5 \pm 3 . 5$ </td><td> $5 7 . 6 \pm 3 . 1$ </td><td> $6 2 . 1 \pm 0 . 6$ </td><td></td><td>2.9</td></tr><tr><td></td><td>PAR (Ours)</td><td> ${ \bf 8 0 . 6 \pm 0 . 6 }$ </td><td> $6 5 . 9 \pm 1 . 1$ </td><td> ${ \bf 6 8 . 8 \pm 2 . 8 }$ </td><td> $\overline { { { \bf 5 1 . 0 } ~ \pm ~ 2 . 9 } }$ </td><td> $\overline { { 5 8 . 2 \pm 1 . 9 } }$ </td><td></td><td> ${ \bf 6 5 . 5 \pm 0 . 7 }$ </td><td>1.0</td></tr></table>

## IV. EXPERIMENTS

## A. Experimental Setup

The experimental study compares PairAlign (PAR) with representative preprocessing-based graph rewiring baselines covering the main design families in the literature. Specifically, we include curvature-driven methods, namely SDRF [4] and BORF [6]; spectral or global-connectivity methods, including FoSR [9], the effective resistance-based rewiring method GTR [5], and GOKU [23]; and the locality-aware sparse rewiring method LASER [12]. For the heterophilic benchmarks, we further compare with two recent variants that extend rewiring beyond purely structural criteria, namely JDR [14], which couples graph rewiring with feature denoising, and ComFy [13], which guides rewiring using community structure and feature similarity. For each backbone, we also report the corresponding model on the original graph as the no-rewiring baseline. Across all tables, boldface denotes the best result and underline denotes the second-best result.

The benchmarks are organized into three groups. The first group contains widely used real-world node classification datasets: the citation networks Cora and CiteSeer [31], the web graphs Texas, Cornell, and Wisconsin [32], and the Wikipedia-network dataset Chameleon [33]. The second group contains the standard TUDataset graph classification benchmarks ENZYMES, IMDB-BINARY, MUTAG, PRO-TEINS, REDDIT-BINARY, and COLLAB [34]. The third group contains five heterophilic node classification benchmarks from Platonov et al. [35]: Roman-Empire, Amazon-Ratings, Minesweeper, Tolokers, and Questions. Following the standard evaluation protocols of these benchmarks, performance is reported in classification accuracy for the first group and for Roman-Empire and Amazon-Ratings, and in ROC AUC for Minesweeper, Tolokers, and Questions. Detailed dataset statistics, split protocols, and implementation settings are provided in the appendix.

For the standard node classification and graph classification benchmarks, we consider two widely used message-passing backbones, namely GCN [2] and GIN [30]. For the heterophilic node classification benchmarks, we use GCN [2] and GAT [36]. Within each benchmark family, all rewiring methods are evaluated under the same backbone, data split, and training protocol unless otherwise stated, so that performance differences can be attributed to the rewiring strategy rather than to backbone-specific tuning. Rewiring is treated as a structural preprocessing step, after which the downstream message-passing model is trained on the rewired graph using an otherwise identical optimization pipeline across competing methods. This protocol is designed to isolate the contribution of rewiring to both graph structure and downstream prediction.

## B. Evaluation Protocol

We evaluate rewiring from two complementary perspectives, downstream prediction performance and structural rewiring quality. For downstream prediction performance, the primary metric follows the native evaluation protocol of each benchmark family. Classification accuracy is reported on node classification and graph classification datasets. We state the exact evaluation implementation in Appendix and report results under that protocol for reproducibility.

Beyond downstream accuracy, we evaluate how rewiring changes the communication structure of the graph. We report two pair-centric diagnostics, ∆Shortage and Coverage@10, to inspect whether the added edges repair pairs with high shortage. Effective resistance proxies provide external references for pairwise and global communication bottlenecks.

Based on the pair-shortage $\boldsymbol { S _ { u v } }$ in Eq. (5) and the normalized shortage weight $p _ { u v }$ in Eq. (7) on the computational graph W, we measure the relative reduction of shortage on the rewired graph $\mathbf { W } ^ { \prime }$ by

$$
\Delta \mathrm { S h o r t a g e } = \frac { \sum _ { ( u , v ) \in \mathcal { T } } p _ { u v } \left[ S _ { u v } ( \mathbf { W } ) - S _ { u v } ( \mathbf { W } ^ { \prime } ) \right] } { \sum _ { ( u , v ) \in \mathcal { T } } p _ { u v } S _ { u v } ( \mathbf { W } ) + \varepsilon } .
$$

A larger value indicates stronger repair of the originally underserved pairs. Coverage@10 measures how broadly this

TABLE II  
GRAPH CLASSIFICATION ACCURACY (%) ON SIX DATASETS UNDER DIFFERENT REWIRING METHODS AND BACKBONES. THE AVERAGE RANKING (AR) REFLECTS THE MEAN POSITION OF EACH METHOD ACROSS ALL DATASETS.
<table><tr><td>Backbone</td><td>Rewiring</td><td>ENZYMES</td><td>IMDB</td><td>MUTAG</td><td>PROTEINS</td><td>REDDIT</td><td>COLLAB</td><td>AR</td></tr><tr><td rowspan="6">GCN</td><td>None [2]</td><td> $2 5 . 5 \pm 1 . 3$ </td><td> $4 9 . 3 \pm 1 . 0$ </td><td> $6 8 . 8 \pm 2 . 1$ </td><td> $7 0 . 6 \pm 1 . 0$ </td><td> $8 1 . 4 \pm 1 . 7$ </td><td> $6 7 . 2 \pm 1 . 2$ </td><td>7.1</td></tr><tr><td>SDRF [4]</td><td> $2 6 . 1 \pm 1 . 1$ </td><td> $4 9 . 1 \pm 0 . 9$ </td><td> $7 0 . 5 \pm 2 . 1$ </td><td> $7 1 . 4 \pm 0 . 8$ </td><td> $8 5 . 4 \pm 1 . 4$ </td><td> $6 7 . 2 \pm 1 . 4$ </td><td>6.0</td></tr><tr><td>FoSR [9]</td><td> $2 7 . 4 \pm 1 . 1$ </td><td> $4 9 . 6 \pm 0 . 8$ </td><td> $7 5 . 6 \pm 1 . 7$ </td><td> $7 2 . 3 \pm 0 . 9$ </td><td> $8 6 . 2 \pm 0 . 9$ </td><td> $6 5 . 3 \pm 1 . 2$ </td><td>4.4</td></tr><tr><td>BORF [6]</td><td> $2 4 . 7 \pm 1 . 0$ </td><td> $5 0 . 1 \pm 0 . 9$ </td><td> $7 5 . 8 \pm 1 . 9$ </td><td> $7 1 . 0 \pm 0 . 8$ </td><td> $\overline { { 8 5 . 9 \pm 2 . 1 } }$ </td><td> $\mathrm { T I M E O U T }$ </td><td>5.3</td></tr><tr><td>GTR [5]</td><td> $2 7 . 4 \pm 1 . 1$ </td><td> $\overline { { 4 9 . 5 \pm 1 . 0 } }$ </td><td> $7 8 . 9 \pm 1 . 8$ </td><td> $7 2 . 4 \pm 1 . 2$ </td><td> $8 4 . 8 \pm 1 . 1$ </td><td> $6 7 . 0 \pm 1 . 0$ </td><td>4.7</td></tr><tr><td>LASER [12]</td><td> $2 7 . 7 \pm 2 . 6$ </td><td> $4 9 . 9 \pm 1 . 6$ </td><td> $7 5 . 2 \pm 3 . 6$ </td><td> $7 2 . 7 \pm 1 . 7$ </td><td> $8 4 . 8 \pm 1 . 9$ </td><td> $6 7 . 6 \pm 1 . 2$ </td><td>3.4</td></tr><tr><td></td><td>GOKU [23]</td><td> $2 7 . 6 \pm 1 . 2$ </td><td> $4 9 . 8 \pm 0 . 7$ </td><td> $8 1 . 0 \pm 2 . 0$ </td><td> $7 1 . 9 \pm 0 . 8$ </td><td> $8 5 . 4 \pm 1 . 2$ </td><td> $6 7 . 5 \pm 1 . 0$ </td><td>3.5</td></tr><tr><td></td><td>PAR (Ours)</td><td> ${ \bf 3 1 . 5 \pm 3 . 8 }$ </td><td> ${ \bf 5 0 . 3 \pm 2 . 5 }$ </td><td> $\overline { { { \bf 8 3 . 3 \pm 0 . 8 } } }$ </td><td> $7 1 . 9 \pm 1 . 6$ </td><td> ${ \bf 8 8 . 9 \pm 1 . 7 }$ </td><td> ${ \bf 6 8 . 2 \pm 1 . 7 }$ </td><td>1.6</td></tr><tr><td rowspan="7">GIN</td><td>None [30]</td><td> $3 1 . 3 \pm 1 . 2$ </td><td> $6 9 . 0 \pm 1 . 3$ </td><td> $7 5 . 5 \pm 2 . 9$ </td><td> $6 9 . 7 \pm 1 . 0$ </td><td> $8 7 . 6 \pm 3 . 5$ </td><td> $7 3 . 1 \pm 0 . 7$ </td><td>6.8</td></tr><tr><td>SDRF [4]</td><td> $3 3 . 5 \pm 1 . 3$ </td><td> $6 8 . 6 \pm 1 . 2$ </td><td> $7 7 . 3 \pm 2 . 3$ </td><td> $7 2 . 2 \pm 0 . 9$ </td><td> $8 9 . 0 \pm 1 . 1$ </td><td> $7 3 . 6 \pm 0 . 7$ </td><td>4.9</td></tr><tr><td>FoSR [9]</td><td> $2 5 . 3 \pm 1 . 2$ </td><td> $6 9 . 5 \pm 1 . 1$ </td><td> $7 5 . 2 \pm 3 . 0$ </td><td> ${ \bf 7 4 . 2 \pm 0 . 8 }$ </td><td> $8 9 . 4 \pm 0 . 9$ </td><td> $7 4 . 0 \pm 0 . 9$ </td><td>4.3</td></tr><tr><td>BORF [6]</td><td> $\underline { { 3 5 . 5 } } \pm 1 . 2$ </td><td> $7 1 . 3 \pm 1 . 5$ </td><td> $8 0 . 8 \pm 2 . 5$ </td><td> $7 1 . 3 \pm 1 . 0$ </td><td> $\overline { { { \bf 8 9 . 9 \pm 0 . 9 } } }$ </td><td> $\overline { { \mathrm { T I M E O U T } } }$ </td><td>3.8</td></tr><tr><td>GTR [5]</td><td> $2 8 . 4 \pm 1 . 8$ </td><td> $\overline { { 7 0 . 1 \pm 1 . 2 } }$ </td><td> $\overline { { 7 8 . 5 \pm 3 . 5 } }$ </td><td> $7 3 . 3 \pm 0 . 9$ </td><td> $8 8 . 3 \pm 0 . 9$ </td><td> $7 3 . 3 \pm 0 . 8$ </td><td>4.7</td></tr><tr><td>LASER [12]</td><td> $3 5 . 3 \pm 1 . 3$ </td><td> $6 8 . 6 \pm 1 . 2$ </td><td> $7 6 . 1 \pm 2 . 4$ </td><td> $7 2 . 1 \pm 0 . 7$ </td><td> $8 7 . 1 \pm 0 . 6$ </td><td> $7 3 . 1 \pm 0 . 9$ </td><td>6.0</td></tr><tr><td>GOKU [23]</td><td> $3 3 . 8 \pm 1 . 2$ </td><td> $7 1 . 3 \pm 0 . 9$ </td><td> $7 8 . 4 \pm 2 . 5$ </td><td> $7 3 . 9 \pm 1 . 0$ </td><td> $8 8 . 8 \pm 0 . 9$ </td><td> $7 3 . 7 \pm 0 . 8$ </td><td>3.4</td></tr><tr><td></td><td>PAR (Ours)</td><td> ${ \pm 4 . 5 \pm 2 . 6 }$ </td><td> $\overline { { 7 1 . 5 \pm 1 . 7 } }$ </td><td> ${ \bf 8 2 . 6 \pm 3 . 0 }$ </td><td> ${ \overline { { 7 1 . 7 \pm 1 . 8 } } }$ </td><td> $8 9 . 2 \pm 1 . 5$ </td><td> ${ \bf 7 4 . 4 \pm 1 . 9 }$ </td><td>2.2</td></tr></table>

repair reaches the most severe shortage region. Let $\mathcal { T } _ { 1 0 } \subset \mathcal { T }$ be the top 10% pairs ranked by $S _ { u v } ( \mathbf { W } )$ . We compute

$$
\mathrm { C o v e r a g e @ 1 0 } = \frac { 1 } { | \mathcal T _ { 1 0 } | } \sum _ { ( u , v ) \in \mathcal T _ { 1 0 } } \mathbb { I } \left[ s _ { u v } ( \mathbf { W } ^ { \prime } ) - s _ { u v } ( \mathbf { W } ) > \eta \right] ,
$$

where $\eta$ is a small numerical tolerance. The value ∆Shortage measures the amount of targeted repair, and Coverage@10 measures the fraction of top shortage pairs that receive nontrivial support.

We further report effective resistance proxies to relate our diagnostics to established bottleneck measures [5]. For a pair $( u , v )$ , the pairwise effective resistance is

$$
\mathrm { P E R } _ { G } ( u , v ) = ( \mathbf { e } _ { u } - \mathbf { e } _ { v } ) ^ { \top } L _ { G } ^ { \dag } ( \mathbf { e } _ { u } - \mathbf { e } _ { v } ) ,
$$

where $L _ { G } ^ { \dagger }$ is the Moore–Penrose pseudoinverse of the graph Laplacian, and $\mathbf { e } _ { u } , \mathbf { e } _ { v } \in \mathbb { R } ^ { | V | }$ are the standard basis vectors associated with nodes u and v, respectively. At the graph level, we use total effective resistance,

$$
\mathrm { T E R } ( G ) = \sum _ { u < v } \mathrm { P E R } _ { G } ( u , v ) = | V | \sum _ { i = 2 } ^ { | V | } \frac { 1 } { \lambda _ { i } ( L _ { G } ) } .
$$

comes from the repaired topology used for message passing. PAR improves the pairwise support available before backbone training, which is especially useful when useful information must move beyond immediate neighborhoods.

Here, $\lambda _ { i } ( L _ { G } )$ is the i-th eigenvalue of $L _ { G }$ . Lower PER or TER indicates smaller effective resistance and stronger communication connectivity. We report the relative reductions of PER on top shortage pairs and TER on the whole graph as external bottleneck proxies. Rewiring is applied as a preprocessing step, and downstream results are averaged over repeated seeds.

## C. Experiment analysis

Node Classification. PAR gives the strongest node classification results under both backbones, with the best average rank under GCN and GIN, reaching 1.50 and 1.00. The gains are steady on citation graphs and become larger on topologylimited datasets. Under GIN, Texas rises from 53.5 to 68.8 and Cornell from 36.5 to 51.0; under GCN, PAR also improves the no-rewiring baseline on both Cora and CiteSeer. The consistent gains across GCN and GIN suggest that the benefit

Graph Classification. PAR also performs strongly on the TUD graph classification benchmarks, with the best average rank under both GCN and GIN. Under GCN, it attains the top result on five of the six datasets, with clear gains on ENZYMES and MUTAG. Under GIN, it remains strongest on ENZYMES, IMDB, MUTAG, and COLLAB. The gains are larger when graph labels depend on information distributed across separated or weakly connected substructures. They are narrower on PROTEINS and in some REDDIT settings, where local motifs or sufficient original structure may already carry much of the label signal. This pattern supports the intended role of PairAlign. Pairwise support can improve whole-graph representations when labels require information to move across separated parts of the graph.

Heterophilous Node Classification. The heterophilous benchmarks are a stricter test of rewiring quality because useful evidence is often nonlocal, but arbitrary shortcuts can damage class-relevant structure. PAR obtains the best average rank under both GCN and GAT, improving over the closest competitor ComFy in each case. With GCN, it gives the strongest results on Roman-Empire and Amazon-Ratings and remains close to the top on Minesweeper and Tolokers. With GAT, it leads on Tolokers and Questions and stays within a small margin of the best result on the remaining datasets. This stability is more informative than isolated wins. It shows that PAR can add useful nonlocal support without turning rewiring into broad long-range mixing, which is important for heterophilous graphs where distant evidence and structural specificity must both be preserved.

The benchmark groups offer three important observations about PairAlign.

• On node classification, PAR improves both GCN and GIN across the tested datasets and keeps the best average rank under both backbones. This consistent gain across backbones supports a graph-side explanation. PairAlign repairs pairwise support for message passing.

TABLE III  
RESULTS ON HETEROPHILIC NODE CLASSIFICATION DATASETS. ROMAN-EMPIRE AND AMAZON-RATINGS ARE REPORTED IN ACCURACY (%), WHILE MINESWEEPER, TOLOKERS, AND QUESTIONS ARE REPORTED IN ROC AUC (%). THE AVERAGE RANKING (AR) REFLECTS THE MEAN POSITION OF EACH METHOD ACROSS ALL DATASETS.
<table><tr><td>Backbone</td><td>Rewiring</td><td>Roman-Empire</td><td>Amazon-Ratings</td><td>Minesweeper</td><td>Tolokers</td><td>Questions</td><td>AR</td></tr><tr><td rowspan="7">GCN</td><td>None [2]</td><td> $7 2 . 6 \pm 0 . 6$ </td><td> $4 7 . 9 \pm 0 . 5$ </td><td> $8 9 . 2 \pm 0 . 7$ </td><td> $8 3 . 2 \pm 1 . 0$ </td><td> $7 6 . 0 \pm 1 . 1$ </td><td>7.3</td></tr><tr><td>SDRF [4]</td><td> $7 3 . 7 \pm 0 . 6$ </td><td> $4 8 . 1 \pm 0 . 5$ </td><td> $8 9 . 3 \pm 0 . 4$ </td><td> $8 3 . 5 \pm 0 . 6$ </td><td> $7 6 . 0 \pm 1 . 0$ </td><td>5.7</td></tr><tr><td>FoSR [9]</td><td> $7 3 . 7 \pm 0 . 7$ </td><td> $4 8 . 6 \pm 0 . 5$ </td><td> ${ \bf 8 9 . 5 \pm 0 . 4 }$ </td><td> $8 3 . 6 \pm 0 . 6$ </td><td> $7 6 . 3 \pm 0 . 9$ </td><td>3.4</td></tr><tr><td>BORF [6]</td><td> $7 3 . 8 \pm 0 . 6$ </td><td> $4 8 . 8 \pm 0 . 5$ </td><td> $8 9 . 3 \pm 0 . 6$ </td><td> $\mathrm { T I M E O U T }$ </td><td> $\mathrm { T I M E O U T }$ </td><td>5.2</td></tr><tr><td>GTR [5]</td><td> $7 3 . 2 \pm 0 . 6$ </td><td> $4 8 . 2 \pm 0 . 4$ </td><td> $8 9 . 3 \pm 0 . 5$ </td><td> $8 3 . 6 \pm 0 . 8$ </td><td> $7 6 . 2 \pm 1 . 0$ </td><td>4.9</td></tr><tr><td>JDR [14]</td><td> $7 2 . 9 \pm 0 . 7$ </td><td> $4 8 . 2 \pm 0 . 5$ </td><td> $8 9 . 1 \pm 0 . 5$ </td><td> $8 3 . 8 \pm 0 . 8$ </td><td> ${ \bf 7 6 . 6 \pm 0 . 9 }$ </td><td>4.9</td></tr><tr><td>ComFy [13]</td><td> $7 3 . 8 \pm 0 . 5$ </td><td> $4 9 . 2 \pm 0 . 4$ </td><td> $8 9 . 3 \pm 0 . 5$ </td><td> ${ \bf 8 5 . 1 \pm 0 . 7 }$ </td><td> $7 6 . 4 \pm 0 . 9$ </td><td>2.4</td></tr><tr><td></td><td>PAR (Ours)</td><td> $\overline { { 7 4 . 1 \pm 0 . 7 } }$ </td><td> $\overline { { { \bf 4 9 . 6 \pm 0 . 3 } } }$ </td><td> $\underline { { 8 9 . 4 \pm 0 . 6 } }$ </td><td> $\underline { { 8 4 . 7 \pm 0 . 7 } }$  </td><td> $\overline { { 7 6 . 1 \pm 1 . 3 } }$ </td><td>2.2</td></tr><tr><td rowspan="7">GAT</td><td>None [36]</td><td> $8 0 . 3 \pm 0 . 7$ </td><td> $4 7 . 8 \pm 0 . 7$ </td><td> $9 1 . 3 \pm 0 . 6$ </td><td> $8 1 . 9 \pm 0 . 7$ </td><td> $7 7 . 1 \pm 1 . 0$ </td><td>7.3</td></tr><tr><td>SDRF [4]</td><td> $8 0 . 8 \pm 0 . 4$ </td><td> $4 8 . 3 \pm 0 . 5$ </td><td> $9 1 . 3 \pm 0 . 4$ </td><td> $8 2 . 5 \pm 0 . 5$ </td><td> $7 7 . 4 \pm 0 . 8$ </td><td>5.3</td></tr><tr><td>FoSR [9]</td><td> $8 0 . 9 \pm 0 . 4$ </td><td> $4 9 . 0 \pm 0 . 5$ </td><td> $9 1 . 5 \pm 0 . 6$ </td><td> $8 2 . 6 \pm 0 . 5$ </td><td> $7 7 . 4 \pm 0 . 8$ </td><td>3.6</td></tr><tr><td>BORF [6]</td><td> $8 0 . 8 \pm 0 . 6$ </td><td> ${ \bf 4 9 . 3 \pm 0 . 4 }$ </td><td> $9 1 . 1 \pm 0 . 4$ </td><td> $\mathrm { T I M E O U T }$ </td><td> $\mathrm { T I M E O U T }$ </td><td>6.0</td></tr><tr><td>GTR [5]</td><td> $8 0 . 7 \pm 0 . 5$ </td><td> $4 8 . 4 \pm 0 . 5$ </td><td> $9 1 . 7 \pm 0 . 4$ </td><td> $8 2 . 1 \pm 0 . 6$ </td><td> $7 7 . 3 \pm 0 . 8$ </td><td>5.6</td></tr><tr><td>JDR [14]</td><td> $8 0 . 8 \pm 0 . 5$ </td><td> $4 8 . 5 \pm 0 . 5$ </td><td> $9 1 . 7 \pm 0 . 6$ </td><td> $8 2 . 1 \pm 0 . 7$ </td><td> $7 7 . 4 \pm 0 . 8$ </td><td>4.6</td></tr><tr><td>ComFy [13]</td><td> ${ \bf 8 1 . 1 \pm 0 . 5 }$ </td><td> $4 8 . 8 \pm 0 . 7$ </td><td> ${ \bf 9 2 . 3 \pm 0 . 5 }$ </td><td> $8 2 . 7 \pm 0 . 6$ </td><td> $7 7 . 5 \pm 0 . 8$ </td><td>2.0</td></tr><tr><td></td><td>PAR (Ours)</td><td> $8 1 . 0 \pm 0 . 4$ </td><td> $\underline { { 4 9 . 1 \pm 0 . 5 } }$ </td><td> $9 2 . 1 \pm 0 . 4$ </td><td> $\overline { { { \bf 8 3 . 8 \pm 0 . 6 } } }$  </td><td> $\overline { { 7 7 . 6 \pm 0 . 7 } }$ </td><td>1.6</td></tr></table>

• On graph classification, the larger gains appear on datasets where labels depend on information spread across separated or weakly connected substructures. This extends the effect of pairwise repair beyond node prediction, showing that better pairwise support can improve whole-graph representations.

• On heterophilous graphs, PAR is more stable across datasets than baselines that peak only in a few cases. This stability matters because heterophily needs nonlocal evidence but can be harmed by indiscriminate mixing. PAR adds nonlocal support through shortage targets, keeping long-range rewiring selective.

## D. Pair-Shortage Validity.

![](images/143f03f18902a20230e4003dcaa13c688c2bd781e6e4edcb8f4f5f572a570ea1.jpg)  
(a) Pair-shortage validity

![](images/67a0e44b148cc1e2aae9fddf65c3e564b89d28ac82d200b723deca659c368b99.jpg)  
(b) Structural reduction  
Fig. 3. Pair-level shortage analysis and structural reduction comparison.

We first relate the proposed pair-shortage score to an external pair-level bottleneck proxy before analyzing the effect of rewiring. For each graph in MUTAG, we compute the pair-shortage score on the original graph and compare it with pairwise effective resistance (PER). We then group node pairs into five shortage quantiles and report the normalized mean PER in each bin. As shown in Fig. 3(a), higher shortage quantiles consistently correspond to larger PER. The normalized mean PER increases from 0.4527 in the lowest shortage bin to 1.8161 in the highest bin, and the graphlevel Spearman correlation between shortage and PER reaches 0.9304 on average. This monotone pattern indicates that pairs with high shortage are also pairwise bottlenecks under an effective-resistance view. Hence, the proposed pair-shortage can provide a sensible pair-level diagnostic that is aligned with an established communication bottleneck proxy.

As shown in Fig. 3(b), PAR yields larger reductions in both the shortage score and the pairwise effective resistance of the top 10% pairs with high shortage. ∆PER@T10 increases from 0.4826 with FOSR to 0.6278 with PAR, and ∆Shortage increases from 0.3795 to 0.5030. This indicates that the pair-centric objective of PAR improves aggregate graph connectivity and guides the limited rewiring budget toward communication-deficient node pairs. PAR makes the structural modification more aligned with the pairwise interactions relevant to downstream prediction.

We next evaluate structural reduction on the same target pairs fixed before rewiring. On MUTAG, PAR is compared with FOSR under the same edge budget. Since FOSR directly optimizes a spectral graph-connectivity surrogate, it provides a relevant baseline for resistance-based evaluation.

## E. Main Ablations.

The ablation study in Table IV isolates the components that turn shortage diagnosis into targeted repair. We compare the full model with Greedy-Local and variants that remove focused target selection, remove OT allocation, or drop one term in the transport cost. Task performance is reported together with ∆Shortage and Coverage@10.

The full model gives the clearest evidence for this mech anism. PAR obtains the best task performance on Texas, Roman-Empire, and MUTAG, while also producing the largest ∆Shortage and Coverage@10. For example, on Texas, PAR reaches ∆Shortage of 0.2350 and Coverage@10 of 0.2113, both higher than all ablated variants. On MUTAG, PAR achieves the best task score of 0.8263, with the strongest structural diagnostics as well. The consistent improvement in both prediction and structural repair supports the alignment between PAR’s downstream gains and its targeted repair of pairs with high shortage.

TABLE IV  
MECHANISM-ORIENTED ABLATIONS ON THREE REPRESENTATIVE BENCHMARKS. HIGHER VALUES MEAN BETTER TASK PERFORMANCE, SHORTAGE REDUCTION, AND COVERAGE@10.
<table><tr><td rowspan="2">Method</td><td colspan="3">Texas</td><td colspan="3">Roman-Empire</td><td colspan="3">MUTAG</td></tr><tr><td>Task</td><td>∆Shortage</td><td>Coverage@10</td><td>Task</td><td>∆Shortage</td><td>Coverage@10</td><td>Task</td><td>∆Shortage</td><td>Coverage@10</td></tr><tr><td>Greedy-Local</td><td>0.5622</td><td>0.0452</td><td>0.0351</td><td>0.7317</td><td>0.2795</td><td>0.2143</td><td>0.7833</td><td>0.1620</td><td>0.0467</td></tr><tr><td>w/o Focus</td><td>0.5568</td><td>0.2208</td><td>0.1939</td><td>0.7293</td><td>0.5610</td><td>0.5857</td><td>0.7961</td><td>0.4288</td><td>0.5479</td></tr><tr><td>w/o OT</td><td>0.5351</td><td>0.1741</td><td>0.1471</td><td>0.7305</td><td>0.5709</td><td>0.6143</td><td>0.7767</td><td>0.4826</td><td>0.6432</td></tr><tr><td>w/o Bridge</td><td>0.5528</td><td>0.1586</td><td>0.1668</td><td>0.7346</td><td>0.5487</td><td>0.6465</td><td>0.8078</td><td>0.4415</td><td>0.6036</td></tr><tr><td>w/o Endpoint</td><td>0.5459</td><td>0.1235</td><td>0.1036</td><td>0.7295</td><td>0.5216</td><td>0.4952</td><td>0.8030</td><td>0.4186</td><td>0.5368</td></tr><tr><td>PAR</td><td>0.5877</td><td>0.2350</td><td>0.2113</td><td>0.7412</td><td>0.6220</td><td>0.7048</td><td>0.8263</td><td>0.4855</td><td>0.6447</td></tr></table>

The focused target set is the first ingredient behind this behavior. PAR does not start from candidate edges alone; it first selects pairs with high shortage on the original graph as the pairs to be repaired by rewiring. Removing the focus mechanism still preserves part of the shortage signal, so the variant does not collapse. Yet w/o Focus is consistently below PAR in task performance, ∆Shortage, and Coverage@10 across the three datasets. On Roman-Empire, Coverage@10 decreases from 0.7048 to 0.5857, and on MUTAG it drops from 0.6447 to 0.5479. This gap means that the focused target set gives rewiring a sharper pairwise target. Greedy-Local further shows why local edge ranking is not enough. It selects edges by local improvement but does not coordinate the budget across shortage targets, so its Coverage@10 is only 0.0351 on Texas and 0.0467 on MUTAG, far below PAR. The focus mechanism does more than filter candidates by specifying the pairwise interactions that should receive the limited rewiring budget.

The transport term becomes important once the target set has been fixed. The w/o OT variant keeps the focus mechanism, continuous optimization, and projection pipeline, but sets the transport weight to zero. It yields nontrivial ∆Shortage, reaching 0.5709 on Roman-Empire and 0.4826 on MUTAG. This confirms that the shortage term can expose communication-deficient regions. The w/o OT variant remains below PAR in task performance and targeted coverage. On Texas, Coverage@10 decreases from 0.2113 to 0.1471, and on Roman-Empire it decreases from 0.7048 to 0.6143. These gaps clarify the role of OT: shortage reveals where support is lacking, and OT coordinates the limited edge budget across targets with high shortage to produce an organized allocation.

The remaining variants separate the roles of the two geometric terms in the transport cost. In w/o Bridge, endpoint matching is retained, so the model can still cover a moderate fraction of pairs with high shortage. For instance, on Roman-Empire, its Coverage@10 is 0.6465, close to w/o OT at 0.6143 and clearly above w/o Endpoint at 0.4952. Its ∆Shortage remains lower than PAR, decreasing from 0.2350 to 0.1586 on Texas and from 0.4855 to 0.4415 on MUTAG. Endpoint proximity alone does not guarantee effective repair; the bridge reward helps select edges that actually increase pairwise support. In contrast, w/o Endpoint keeps the bridge reward but suffers a larger drop in Coverage@10, especially on Texas, where it decreases from 0.2113 to 0.1036. Bridge-like edges still need to be aligned with the intended target pairs. The two cost terms play distinct roles. Endpoint matching determines the pairwise targets an added edge should support, and the bridge reward favors edges that can produce effective communication repair.

![](images/e2dece25b9a5e817c321080a201bc0597846e684fdf00aa69ca473227a4478ed.jpg)

![](images/f0195d89a6ba2adc53f8457a3ce5a92bb45689dae766ddfe5463b28b914c2ec7.jpg)  
Fig. 4. Visualization of how added edges are allocated to target pairs with high shortage. Each row is normalized to show how one added edge distributes its allocation over the selected target pairs. Darker colors indicate a larger allocated fraction for the corresponding target pair.

By reformulating over-squashing around communicationdeficient node pairs, PAR recasts graph rewiring as a budgeted structural repair problem, yielding stronger targeted repair and better downstream performance across the datasets.

## F. Analysis of OT-guided Budget Allocation

Necessity of OT. To observe how the OT-guided rewiring rule changes the allocation of a limited edge addition budget, we visualize the assignment from the final added edges to the top shortage target pairs. As shown in Fig. 4, each row corresponds to the added edge and each column corresponds to a target pair with high shortage. Darker entries indicate that a larger fraction of the allocation associated with an added edge is assigned to the corresponding target pair.

The heatmap shows that our PAR distributes its added edges across a broader set of targets with high shortage while retaining clear assignments. Full PAR achieves higher ∆Shortage and Coverage@10. Greedy-Local exhibits a more concentrated allocation pattern, with several added edges assigned mainly to a small subset of target columns. These results indicate that OT-guided allocation spreads the budget over a wider target set and produces stronger pair-centric structural repair.

Stability of OT. We further analyze how the weight $\lambda _ { \mathrm { O T } }$ of the OT-guided rewiring affects the rewiring objective. The MUTAG-GCN setting is kept fixed and only $\lambda _ { \mathrm { O T } }$ is varied. As shown in Fig. 5(a), OT-guided alignment improves Task and structure proxies in the main operating range compared with $\lambda _ { \mathrm { O T } } ~ = ~ 0 .$ . This indicates that the OT-guided rule provides a proper global coordination signal for structural repair and helps added edges cover targets with high shortage more effectively. When $\lambda _ { \mathrm { O T } }$ is further increased, ∆Shortage remains high and even continues to increase, but Task performance starts to decrease. This suggests that an overly strong transport preference mainly emphasizes the structural repair objective and does not necessarily translate into better prediction. The trend supports the role of OT as a budget-coordination mechanism. With a suitable weight, it organizes the limited edge addition budget across shortage targets and helps convert structural repair into downstream task gains.

![](images/854743455d9edb7e420cfa2e48f9d712b28f60418a55daa02aa95e2132cd8a6d.jpg)  
(a) OT weight

![](images/d232869557fca48857a97a60f5eb29f073d4c2d404519cf8b2d65639a91518bf.jpg)  
(b) Rewiring budget  
Fig. 5. Visualization of OT-weight stability and sensitivity analysis.

![](images/e07094cc545027133a3b700152109a7d99b922c8a1f840629500a2409b9fbc7d.jpg)  
(c) Propagation depth

## G. Sensitivity Analysis.

Rewiring budget. The budget sensitivity on MUTAG is shown in Fig. 5(b). We set $k _ { 0 } ~ = ~ 3$ as the integer PAR budget anchor compatible with the main GIN setting. The task curve shows a high-variance plateau; the highest mean appears at $0 . 2 5 k _ { 0 }$ , and its error band largely overlaps with the results from $0 . 5 k _ { 0 }$ to $k _ { 0 } .$ . In this compact budget range, PAR already achieves competitive task performance, and ∆Shortage increases from 0.0620 to 0.1860. When the budget is enlarged beyond $k _ { 0 } ,$ ∆Shortage continues to rise and reaches 0.2506 at $2 . 0 k _ { 0 }$ . The task mean moves to a lower range.

This trend indicates that PAR obtains its main predictive benefit within a limited rewiring budget, with additional edges mainly contributing further targeted shortage repair.

Propagation depth. We further evaluate the sensitivity to the propagation depth K in the pair-support term. To avoid the bias that larger K accumulates more hop contributions under uniform weighting, this experiment uses the same MUTAG-GIN setting with a fixed decayed-support configuration. As shown in Fig. 5(c), K = 1 gives the weakest structural repair, with ∆Shortage of only 0.0218. This indicates that a purely local propagation range is insufficient to capture pairwise communication shortage. Increasing K to 2 raises the task score to 0.8172 and improves ∆Shortage to 0.2389, showing that a moderate propagation range provides a more effective support estimate. When K increases to 3 and 4, ∆Shortage continues to grow and reaches 0.3652 and 0.6398. The task means stay in a similar range under larger variance.

The result suggests that K controls the scale at which PairAlign evaluates pairwise communication shortage. Taskrelevant rewiring benefits are strongest when this scale matches the effective range of downstream message passing.

## V. CONCLUSION

This work proposes PairAlign (PAR), an OT-guided graph rewiring framework that treats over-squashing as a pairlevel communication shortage. PAR centers rewiring on node pairs whose structural demand is high but whose propagation support remains limited. The shortage score turns this pairlevel bottleneck into an interpretable quantity as the ratio of current-graph support relative to original-graph demand, with theoretical consistency to the Jacobian-based shortage. PairAlign turns this diagnosis into an OT-guided allocation strategy for rewiring. Candidate additions are scored by current shortage reduction, with theoretical support for positive shortage relief on the rewired graph. OT-guided allocation then coordinates the limited budget across the shortage profile, prevents repair from concentrating on only a few targets, and obtains a coverage advantage over Greedy-Local under the coverage theory. Across standard graph benchmarks and different backbones, PAR shows consistent effectiveness, and its agreement with PER/TER supports pairwise shortage as a valid over-squashing bottleneck diagnostic.

How to explore task-aware demand modeling and explicit locality constraints for improving scalability on large graphs is our future work.

## REFERENCES

[1] J. Gilmer, S. S. Schoenholz, P. F. Riley, O. Vinyals, and G. E. Dahl, “Neural message passing for quantum chemistry,” in International conference on machine learning. Pmlr, 2017, pp. 1263–1272.

[2] T. N. Kipf and M. Welling, “Semi-supervised classification with graph convolutional networks,” arXiv preprint arXiv:1609.02907, 2016.

[3] U. Alon, “On the bottleneck of graph neural networks and its practical implications,” in International Conference on Learning Representations, 2021.

[4] J. Topping, F. Di Giovanni, B. P. Chamberlain, X. Dong, and M. M. Bronstein, “Understanding over-squashing and bottlenecks on graphs via curvature,” in International Conference on Learning Representations, 2022.

[5] M. Black, Z. Wan, A. Nayyeri, and Y. Wang, “Understanding oversquashing in gnns through the lens of effective resistance,” in International Conference on Machine Learning. PMLR, 2023, pp. 2528–2547.

[6] K. Nguyen, N. M. Hieu, V. D. Nguyen, N. Ho, S. Osher, and T. M. Nguyen, “Revisiting over-smoothing and over-squashing using ollivierricci curvature,” in International Conference on Machine Learning. PMLR, 2023, pp. 25 956–25 979.

[7] R. B. Gabrielsson, M. Yurochkin, and J. Solomon, “Rewiring with positional encodings for graph neural networks,” Transactions on Machine Learning Research, 2023.

[8] R. Abboud, R. Dimitrov, and I. I. Ceylan, “Shortest path networks for graph property prediction,” in Learning on graphs conference. PMLR, 2022.

[9] K. Karhadkar, P. Banerjee, and G. Montufar, “Fosr: First-order spectral rewiring for addressing oversquashing in gnns,” in International Conference on Learning Representations, 2023.

[10] P. K. Banerjee, K. Karhadkar, Y. G. Wang, U. Alon, and G. Montufar,´ “Oversquashing in gnns through the lens of information contraction and graph expansion,” in 58th Annual Allerton Conference on Communication, Control, and Computing (Allerton), 2022, pp. 1–8.

[35] O. Platonov, D. Kuznedelev, M. Diskin, A. Babenko, and L. Prokhorenkova, “A critical look at the evaluation of gnns under heterophily: Are we really making progress?” in Advances in Neural Information Processing Systems, 2023.

[36] P. Velickovi ˇ c, G. Cucurull, A. Casanova, A. Romero, P. Li ´ o, and\` Y. Bengio, “Graph attention networks,” in International Conference on Learning Representations, 2018.

[11] A. Arnaiz-Rodr´ıguez, A. Begga, F. Escolano, and N. M. Oliver, “Diffwire: Inductive graph rewiring via the lovasz bound,” in´ Proceedings of the First Learning on Graphs Conference, 2022.

[12] F. Barbero, A. Velingker, A. Saberi, M. M. Bronstein, and F. Di Giovanni, “Locality-aware graph rewiring in gnns,” in International Conference on Learning Representations, 2024.

[13] C. Rubio-Madrigal, A. Jamadandi, and R. Burkholz, “Gnns getting comfy: Community and feature similarity guided rewiring,” in International Conference on Learning Representations, 2025.

[14] J. Linkerhagner, C. Shi, and I. Dokmani ¨ c, “Joint graph rewiring and´ feature denoising via spectral resonance,” in International Conference on Learning Representations, 2025.

[15] W. Bi, L. Du, Q. Fu, Y. Wang, S. Han, and D. Zhang, “Make heterophilic graphs better fit gnn: A graph rewiring approach,” IEEE Transactions on Knowledge and Data Engineering, vol. 36, no. 12, pp. 8744–8757, 2024.

[16] K. Bose, S. Banerjee, and S. Das, “Can graph neural networks tackle heterophily? yes, with a label-guided graph rewiring approach!” IEEE Transactions on Neural Networks and Learning Systems, 2025.

[17] A. Deac, M. Lackenby, and P. Velickovi ˇ c, “Expander graph propagation,”´ in Learning on Graphs Conference. PMLR, 2022, pp. 38–1.

[18] S. Akansha, “Over-squashing in graph neural networks: A comprehensive survey,” Neurocomputing, vol. 642, 2025.

[19] J. Bober, A. Monod, E. Saucan, and K. N. Webster, “Rewiring networks for graph neural network training using discrete geometry,” in International Conference On Complex Networks And Their Applications. Springer, 2023, pp. 225–236.

[20] F. Tori, V. Holst, and V. Ginis, “The effectiveness of curvature-based rewiring and the role of hyperparameters in gnns revisited,” in International Conference on Learning Representations, 2025.

[21] J. Chen, B. Deng, Z. Zheng, and C. Chen, “Rethinking the gold standard: Why discrete curvature fails to fully capture over-squashing in gnns?” in International Conference on Learning Representations, 2026.

[22] X. Shen, P. Lio, L. Yang, R. Yuan, Y. Zhang, and C. Peng, “Graph rewiring and preprocessing for graph neural networks based on effective resistance,” IEEE Transactions on Knowledge and Data Engineering, vol. 36, no. 11, pp. 6330–6343, 2024.

[23] L. Liang, F. Bu, Z. Song, Z. Xu, S. Pan, and K. Shin, “Mitigating over-squashing in graph neural networks by spectrum-preserving sparsification,” in International Conference on Machine Learning, 2025.

[24] J. Wilson, M. Bechler-Speicher, and P. Velickovi ˇ c, “Cayley graph prop-´ agation,” in Proceedings of the Third Learning on Graphs Conference. PMLR, 2025.

[25] A. Jamadandi, C. Rubio-Madrigal, and R. Burkholz, “Spectral graph pruning against over-squashing and over-smoothing,” in Advances in Neural Information Processing Systems, vol. 37, 2024.

[26] C. Villani et al., Optimal transport: old and new. Springer, 2009, vol. 338.

[27] M. Cuturi, “Sinkhorn distances: Lightspeed computation of optimal transport,” in Advances in Neural Information Processing Systems, 2013.

[28] Y. Bengio, N. Leonard, and A. Courville, “Estimating or propagating´ gradients through stochastic neurons for conditional computation,” arXiv preprint arXiv:1308.3432, 2013.

[29] J. Altschuler, J. Weed, and P. Rigollet, “Near-linear time approximation algorithms for optimal transport via sinkhorn iteration,” in Advances in Neural Information Processing Systems, 2017.

[30] K. Xu, W. Hu, J. Leskovec, and S. Jegelka, “How powerful are graph neural networks?” arXiv preprint arXiv:1810.00826, 2018.

[31] Z. Yang, W. Cohen, and R. Salakhudinov, “Revisiting semi-supervised learning with graph embeddings,” in International conference on machine learning. PMLR, 2016, pp. 40–48.

[32] H. Pei, B. Wei, K. C. C. Chang, Y. Lei, and B. Yang, “Geom-gcn: Geometric graph convolutional networks,” in International Conference on Learning Representations, 2020.

[33] B. Rozemberczki, C. Allen, and R. Sarkar, “Multi-scale attributed node embedding,” Journal of Complex Networks, vol. 9, no. 2, pp. 1–22, 2021.

[34] C. Morris, N. M. Kriege, F. Bause, K. Kersting, P. Mutzel, and M. Neumann, “Tudataset: A collection of benchmark datasets for learning with graphs,” arXiv preprint arXiv:2007.08663, 2020.