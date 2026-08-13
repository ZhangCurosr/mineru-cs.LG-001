# CHAIN-OF-THOUGHT SHOWS THE PATH TO A TREE: REALIZING BRANCHING COMPLEXITY

Debanjan Dutta Indian Statistical Institute Kolkata, India

Anish Chakrabarty LTCI, Télécom Paris Palaiseau, France

Swagatam Das<sup>†</sup> Indian Statistical Institute Kolkata, India

## ABSTRACT

Chain of Thought (CoT) lifts the expressive ceiling of bounded-depth Transformers, with characterizations tying the number of CoT steps to circuit complexity classes. What remains largely missing are concrete instantiations with explicit, depth-bounded constructions, and the traversal procedures such characterizations presuppose. We close this gap for branching complexity. We give CoT realizations of depth-first search (DFS) and of Dijkstra algorithm, the latter subsuming breadth-first search, by unique hard-attention decoders of at most two layers, and use them as a shared computational substrate: reusing the DFS decoder yields the Strahler number of an n-vertex tree in 2n − 1 steps with four layers, and reusing the Dijkstra decoder yields its width in n − 1 steps with three. Since computing the Strahler number of a binary tree given as a term is NC<sup>1</sup>-complete, and our constructions handle arbitrary n-ary trees without layer normalization or positional encodings, this is a non-trivial witness for the linear-step regime of the CoT hierarchy. Exploiting the classical bijection between ordered trees and Dyck paths, itself realized by our DFS construction, which emits the path as it traverses, we give independent constructions for both measures on the path representation.

## 1 Introduction

Chain-of-thought (CoT) prompting has turned autoregressive Transformers into systems that appear to execute procedures rather than merely predict tokens (Wei et al., 2022). Going beyond empirical curiosity, Merrill and Sabharwal (2024) show that CoT lets bounded-depth Transformers escape the complexity-class ceiling of a single forward pass, and Barceló et al. (2025) go further, giving an exact fixed-step characterization in terms of the Ehrenfeucht–Haussler (EH) rank of a function. Such possibility results still allow for explicit, minimal-depth CoT construction solving non-trivial problems exhibiting the traversal or recursion, which are only known to exist. Barceló et al. (2025) themselves note that their bound is achieved through exhaustive tree traversal, but they do not construct the traversal. Zhu et al. (2026) study whether continuous CoT can solve reachability, but only empirically. Our work begins with this fact in mind.

Branching complexity of a tree —measured classically by the Strahler number (Strahler, 1957) and, independently, by the width of a tree —is one of the oldest quantitative questions about hierarchical structure, with consequences ranging from the minimum number of registers needed to evaluate an arithmetic expression (Flajolet et al., 1979) to bounds on random tree shape (Addario-Berry et al., 2013) to the size of the largest antichain (Dilworth, 1950). Moreover, it turns out to be readily tied to the CoT lore. The Strahler number of binary trees and the EH rank on binary decision trees satisfy the same recursive update (Dahiya and Mahajan, 2021), and Ganardi and Lohrey (2026) have just shown that computing the Strahler number of a binary tree is NC<sup>1</sup>-complete – precisely the class Merrill and Sabharwal (2024) associate with a linear number of CoT steps. In simpler terms, we now know of a rank measure known to sit at the boundary of what constant (or, linear) CoT depth can express, and a general theorem stating that such measures should be expressible with the right traversal machinery. In this work, we introduce the machinery.

We give explicit constructions of depth-first search (DFS), breadth-first search (BFS, as a special case of Dijkstra’s algorithm), and Dijkstra, realized by Transformer decoders with two layers and two attention heads under CoT. The findings do not emerge as isolated expressivity results but are based on a computational substrate on which branching complexity becomes CoT-accessible. For example, reusing the same DFS realization that produces the traversal order, we compute the Strahler number in $2 n - 1$ steps for an n-vertex tree (Theorem 8), and reusing the Dijkstra realization, we compute the width of a tree in n − 1 steps (Theorem 13). Because computing the Strahler number of a general n-ary tree is a strict generalization of the binary, tree setting in which the ${ \mathsf { N C } } ^ { 1 }$ -completeness result was established, this is a non-trivial concrete witness for the Merrill and Sabharwal framework, obtained without appealing to auxiliary primitives such as layer normalization that prior CoT constructions have relied on.

The traversal machinery gives us a second, independent route to the same measures. The classical bijection between ordered trees and Dyck paths (Goldman and Sundquist, 1992) allows us to flatten a tree onto a one-dimensional lattice path while providing an algorithmic witness of the bijection. The fact that the number of full binary trees $G _ { T }$ with n internal vertices and Strahler number $\mathsf { s t } ( G _ { T } )$ turns out to be the same as the number of Dyck paths of length 2n whose height h satisfies $\lfloor \log _ { 2 } ( 1 + h ) \rfloor = { \mathsf { s t } } ( G _ { T } )$ (Flajolet et al., 1979; Addario-Berry et al., 2024) becomes crucial in context. The DFS realization of Theorem 5 is the map from tree to path, since a traverse step is exactly an up-step and a backtrack step is exactly a down-step. This raises a natural question that, to our knowledge, has not been asked in the CoT-expressivity literature: given two representations of the same combinatorial object related by an explicit, algorithmically realized bijection, do CoT realizations of a derived quantity (here, Strahler number and width) transfer across the bijection, or must they be constructed independently on each side? We answer this by proving identifiability results and associated constructions for both representations (Theorems 8, 11 for Strahler number; Theorems 13, 16 for width). The constructions, following Rizvi-Martel et al. (2024), involve bilinear maps. While we are interested in proving existence, an interesting open question emerges out of our inquiry: Is CoT realizability closed under bijective changes of representation, and hence, in general, under composition?

Below we summarize our main contributions.

• We give the first CoT realizations of general graph traversal (DFS and Dijkstra) by hard-attention Transformer decoders, each requiring at most two layers and two heads, and show that Dijkstra realization yields a step-count advantage over comparison-based implementations by replacing the search for a minimum-distance vertex with a single constant-time attention operation.

• We use these traversal realizations as a shared computational substrate to give explicit, small-depth CoT constructions for two independent notions of branching complexity, the Strahler number and width of a tree, providing a concrete, non-trivial instantiation of the CoT-rank framework after Barceló et al. (2025) on a measure already known to be NC<sup>1</sup>-complete in the binary case (Ganardi and Lohrey, 2026), generalized here to arbitrary ordered trees.

• Exploiting the classical bijection between trees and Dyck paths, realized algorithmically by our DFS construction, we give independent CoT constructions for both measures on the path representation, and show that the resulting constructions do not transfer readily across the bijection without change in mechanism or layer count.

![](images/934a2eb4ce5ca51ffc808fec6c7b31e02180eb08a40726522e4946a039de940e.jpg)  
Figure 1: Overview of the paper’s constructions and their relationships. The unfolding, ϕ, maps a tree $G _ { T }$ to its Dyck path by recording the DFS traversal used in Theorem 5, and the reconstruction, ψ, is realized via Algorithm 1 (Theorem 7). Besides identifying st and wd across this bijection, we provide standalone CoTrealization pathways for each directly on both representations: $G _ { T }$ (Theorems 8 and 13, respectively) and paths (Theorems 11 and 16).

Organization. section 2 introduces the necessary preliminaries on graph traversal and branching complexity. section 3 formalizes the CoT realization framework used throughout. section 4 contains the main results: traversal constructions (DFS and Dijkstra) (subsection 4.1) and realization of the Strahler number and tree width on both trees and Dyck paths along with the path-reconstruction algorithm (subsection 4.2). section 5 concludes with a comprehensive discussion. Full proofs are deferred to the Appendix.

Related Works. The expressivity of Transformers is well studied, from single-pass characterizations of encoders and decoders (Strobl et al., 2024) to step-indexed accounts of chain-of-thought decoders placing logarithmic and linear reasoning budgets in L and ${ \mathsf { N C } } ^ { 1 }$ respectively (Merrill and Sabharwal, 2024; Li et al., 2024). In parallel, graph-algorithmic constructions realize DFS, BFS, and Dijkstra via looped Transformers (Giannou et al., 2023) with graph-interacting heads (De Luca and Fountoulakis, 2024; Sanford et al., 2024), and empirical CoT studies address reachability without materializing traversal (Zhu et al., 2026); each lies outside the autoregressive step-counted CoT setting, leaving traversal yet to be explicitly realized as CoT steps. The concrete problems that inhabit the linear-step regime remain few. The Strahler–EH correspondence (Dahiya and Mahajan, 2021), together with the ${ \mathsf { N C } } ^ { 1 }$ membership of Strahler number computation (Ganardi and Lohrey, 2026), motivates our focus on branching complexity. Appendix A presents a comprehensive discussion.

## 2 Branching Complexity and Graphs

## 2.1 Graphs and their traversal techniques

Let $G = ( V , E , A )$ be a simple connected directed graph with vertex set $V = \{ v _ { 0 } , \ldots , v _ { n - 1 } \}$ , edge set $E \subseteq V \times V$ and adjacency matrix $A : \bar { V \times V }  \mathsf { A }$ over an arbitrary domain A, where $a _ { i j } = A ( v _ { i } , v _ { j } )$ denotes the weight of edge $e _ { i j } = ( v _ { i } , v _ { j } )$

Dijkstra Algorithm. Restricting to $\mathsf { A } = \mathbb { R } _ { > 0 }$ and fixing a source $s = v _ { p } \in V$ , the Dijkstra algorithm computes single-source shortest paths. The instantaneous description of the dynamic at step t is the tuple $\mathsf { D } \mathsf { k } \mathsf { s t } ^ { ( t ) } =$ $( \mathsf { D i s t } ^ { ( t ) } , c u r ^ { ( t ) } , v s t ^ { ( t ) } )$ , with $\mathsf { D i s t } ^ { ( t ) } \in ( \mathbb { R } _ { > 0 } \cup \{ \infty \} ) ^ { n }$ being the tentative distance vector, $c u r ^ { ( t ) } \in V$ the current vertex, and $v s t ^ { ( t ) } : V \to \{ 0 , 1 \}$ the visited indicator. The system is initialized by $c u r ^ { ( 0 ) } = v _ { p } , \mathsf { D i s t } ^ { ( 0 ) } [ p ] = 0 , \mathsf { D i s t } ^ { ( 0 ) } [ i ] = \infty$ for all $v _ { i } \neq v _ { p }$ , and $v s t ^ { ( 0 ) } ( v _ { p } ) = 1$ with $v s t ^ { ( 0 ) } ( v _ { i } ) = 0$ otherwise. At each step, given $c u r ^ { ( t ) } = v _ { i }$ , the transition relaxes all outgoing edges by setting $\mathsf { D i s t } ^ { ( t + 1 ) } [ j ] = \operatorname* { m i n } ( \mathsf { D i s t } ^ { ( t ) } [ j ] , \mathsf { D i s t } ^ { ( t ) } [ i ] + a _ { i j } )$ for every $v _ { j } \in V$ , selects the next vertex $c u r ^ { ( t + 1 ) } = v _ { j ^ { \prime } }$ where j<sup>′</sup> = argmin $\{ \mathsf { D i s t } ^ { ( t + 1 ) } [ j ] \mid v s t ^ { ( t ) } ( v _ { j } ) = 0 \}$ , and marks $v s t ^ { ( t + 1 ) } ( v _ { j ^ { \prime } } ) = 1$ while preserving all prior visits. At termination, $\begin{array} { r } { \mathsf { D i s t } ^ { ( n - 1 ) } [ i ] = \operatorname* { m i n } _ { P \in \mathcal { P } ( s , v _ { i } ) } w ( P ) } \end{array}$ for every $v _ { i } \in V$ , where $\mathcal { P } ( s , v _ { i } )$ denotes the set of directed $( s , v _ { i } )$ -paths and $\begin{array} { r } { w ( P ) = \sum _ { ( v _ { k } , v _ { l } ) \in P } a _ { k l } } \end{array}$

Observe that Dijkstra algorithm can be interpreted as a weighted generalization of the BFS traversal. In this context, we redefine $\mathsf { A } = \{ 1 , \infty \}$ to indicate the presence or absence of an edge $e _ { i j }$ , where a value 1 denotes that the edge $e _ { i j }$ exists and ∞ implies that it does not.

Depth-First Traversal. For DFS traversal (Aho and Hopcroft, 1974) DFS, we choose $\mathsf { A } = \{ 0 , 1 \}$ . Then, ${ \mathcal { N } } ( v _ { i } ) =$ $\{ v _ { j } \ | \ a _ { i j } = 1 \}$ and let $n = | V |$ . The instantaneous description of the dynamic ${ \sf D F S } ^ { ( t ) }$ is denoted by the tuple $( \mathsf { d } \mathsf { f s } ^ { ( t ) } , p a r ^ { ( t ) } , v s t ^ { ( t ) } )$ , such that $\mathsf { d } \mathsf { f s } ^ { ( t ) } : V \to V , p a r ^ { ( t ) } : V \to V \cup \{ \phi \}$ and $v s t ^ { ( t ) } : V \to \{ 0 , 1 \}$ returns an spanning tree T from any basic diagraph (Gessel and Wang, 1979) G in $O ( | V | + | E | )$ ) moves. $\mathbf { A } \mathbf { t } t = 0$ , given a starting vertex $s = v _ { p } \in V , v s t ^ { ( 0 ) } ( s ) = 1$ and $v s t ^ { ( 0 ) } ( v _ { i } ) = 0$ for all $v _ { i } \in V \setminus \{ s \}$ and $p a r ^ { ( 0 ) } ( v _ { i } ) = \phi$ for all $v _ { i } .$ . Then the $t ^ { \mathrm { { t h } } }$ dynamic is given by

$$
\mathsf { d } \mathsf { f } \mathsf { s } ^ { ( t ) } ( s ) = \left\{ \begin{array} { l l } { v _ { j } \quad \mathrm { i f } v _ { j } \in \mathcal { N } ( \mathsf { d } \mathsf { f } \mathsf { s } ^ { ( t - 1 ) } ( s ) ) } & { \mathrm { w i t h } v s t ^ { ( t - 1 ) } ( v _ { j } ) = 0 , } \\ { p a r ^ { ( t - 1 ) } ( \mathsf { d } \mathsf { f } \mathsf { s } ^ { ( t - 1 ) } ( s ) ) , } & { \mathrm { o t h e r w i s e . } } \end{array} \right.\tag{1}
$$

In the former case, traverse, $p a r ^ { ( t ) } ( v _ { j } ) = \mathsf { d f } \mathsf { s } ^ { ( t - 1 ) } ( s )$ and $v s t ^ { ( t ) } ( v _ { j } ) = 1 ;$ all remaining entries of $p a r ^ { ( t ) }$ and $v s t ^ { ( t ) }$ are inherited from step $t - 1$ . In the latter case, backtrack, the mappings remain unchanged. The algorithm terminates after backtracking from s.

Throughout the remainder, we write $n = | V |$ unless otherwise stated, with n also denoting the length of a lattice path where relevant. Furthermore, whenever the edge set A is omitted in the definition of a graph, we refer to its unweighted variant.

## 2.2 Trees, Paths, and Branching Complexity

If in a graph $G _ { T } = ( V , E )$ there is exactly one path between any two distinct vertices $v _ { i } , v _ { j } \in V$ , then $G _ { T }$ is a tree, and $| E | = \bar { | V | } - 1$ . For a finite rooted tree $G _ { T } ~ ( \mathrm { D i e s t e l } , 2 0 1 7 )$ , let depth(v) denote the number of edges on the unique path from the root r to vertex v. The height of $G _ { T }$ is defined as $\begin{array} { r } { \mathrm { h t } ( \bar { G _ { T } } ) \mathrel { \mathop : } = \operatorname* { m a x } _ { v \in V } \mathrm { d e p t h } ( v ) } \end{array}$ , and width as

$$
\operatorname { w d } ( G _ { T } ) : = \operatorname* { m a x } _ { k \geq 0 } | \{ v \in V : \operatorname { d e p t h } ( v ) = k \} | .
$$

Observation 1 (Determining wd while performing $B F S ) .$ . To algorithmically determine w $\operatorname { T d } ( G _ { T } )$ , the width of tree $G _ { T }$ , we employ BFS as a special version of the Dijkstra algorithm. Let $v _ { j }$ and $v _ { j ^ { \prime } }$ be the vertices such that $v s t ^ { ( t ) } ( v _ { j } ) = v s t ^ { ( t ) } ( v _ { j ^ { \prime } } ) = 0 ; v s t ^ { ( t + 1 ) } ( v _ { j } ) = 1 , v s t ^ { ( t + 1 ) } ( v _ { j ^ { \prime } } ) = 0 ;$ and $v s t ^ { ( t + 2 ) } ( v _ { j } ) \stackrel {  } { = } v s t ^ { ( t + 2 ) } ( v _ { j ^ { \prime } } ) = 1$ . Note that, D $\mathsf { i } \mathsf { s t } ^ { ( t + 2 ) } [ j ^ { \prime } ] - \mathsf { D i } \mathsf { s t } ^ { ( t + 2 ) } [ j ] \in \{ 0 , 1 \}$ . Clearly, when $\mathsf { D i s t } ^ { ( t + 2 ) } [ j ^ { \prime } ] = \mathsf { D i s t } ^ { ( t + 1 ) } [ j ]$ , de $\mathrm {  ~ \ t h } ( v ^ { \prime } ) = \mathrm { d e p t h } ( v )$ ; and depth $( v ^ { \prime } ) = \mathrm { d e p t h } ( v ) + 1$ , otherwise. To capture the width during $\mathsf { D } \mathsf { k } \mathsf { s t } ^ { ( t ) }$ , we augment it with scalars $t w d ^ { ( t ) }$ and $m w d ^ { ( t ) } \in \mathbb { Z }$ so that the former stores $\left| \left\{ v _ { i } : v s t ^ { ( t ) } ( i ) = 1 \mathrm { { a n d } } \nexists j \left( \mathrm { D i s t } ^ { ( t ) } [ j ] \neq \infty \land \mathrm { D i s t } ^ { ( t ) } [ j ] > \mathrm { D i s t } ^ { ( t ) } [ i ] \right) \right\} \right|$ while the latter stores ma $\mathrm { x } _ { t ^ { \prime } < t } t w d ^ { ( t ^ { \prime } ) }$ . Unless $G _ { T }$ is a null graph, $t w d ^ { ( t ) }$ and $ m w d ^ { ( t ) }$ for all t must be at least 1. Then initialized by $t w d ^ { ( 0 ) } = m w d ^ { ( 0 ) } = 1$ indicating the root is at depth 1, $m w d ^ { ( n - 1 ) } = \mathrm { w d } ( G _ { T } )$

![](images/c40a230734a3c3fca88b328e9c3d393870de776fe21570f88986566223517d5e.jpg)  
Figure 2: Following the DFS dynamic, the map ϕ on the tree $G _ { T }$ (left) generates the Dyck path (middle). We define the reconstruction ψ of the tree from the Dyck word (or path) via Algorithm 1, with a demonstrative (partial) reconstruction illustrated on the right. The edge pairs (11, 3), (12, 4) and (12, 4), (13, 3) are folded into a single tree edge $( v _ { 1 } , v _ { 2 } )$ The highlighted region on the left is formed by folding the region between (10, 2) and (16, 2).

Although this is a very natural notion of branching complexity, the Strahler number st, on the other hand, is calculated recursively. For leaf node $v , { \mathsf { s t } } ( v ) = 0$ and a node with subtrees $\tau _ { 1 } , \ldots , \tau _ { k }$ , let $M = \operatorname* { m a x } _ { i } { \ s t ( \tau _ { i } ) }$ . Then,

$$
\mathsf { s t } ( G _ { T } ) = { \left\{ \begin{array} { l l } { M + 1 } & { { \mathrm { i f ~ } } c = | \{ i : \mathsf { s t } ( \tau _ { i } ) = M \} | \geq 2 } \\ { M } & { { \mathrm { o t h e r w i s e } } . } \end{array} \right. }
$$

For binary trees, this reduces to $\mathsf { s t } ( G _ { T } ) = \operatorname* { m a x } ( \operatorname* { m i n } ( \mathsf { s t } ( \tau _ { 1 } ) , \mathsf { s t } ( \tau _ { 2 } ) ) + 1 , \operatorname* { m a x } ( \mathsf { s t } ( \tau _ { 1 } ) , \mathsf { s t } ( \tau _ { 2 } ) ) )$ , which, interestingly, coincides with the expression of the Ehrenfeucht-Haussler rank for binary decision trees.

Observation 2 (Determining M and c whiler performing DFS). We reuse the dynamic formulation (1), which is evaluated in a bottom-up manner. Specifically, when the dynamic process is in a traverse step, we assign $t _ { \mathsf { d f s } ^ { ( t ) } ( s ) }$ , the tree rooted at the vertex $\mathsf { d f s } ^ { ( t ) } ( s )$ , the value 0. Note that in this situation, $\tau _ { \mathsf { d f s } } ( t ) _ { \bigl ( s \bigr ) }$ consists of a single vertex. During a backtrack step, let $v = \mathsf { d f s } ^ { ( t ) } ( s )$ and $u = \mathsf { d } \mathsf { f } \mathsf { s } ^ { ( t + 1 ) } ( s )$ . At this point, we record s $\mathfrak { t } ( \tau _ { u } )$ for the tree rooted at the vertex u that has been explored up to step t. We let $M _ { u } ^ { ( t ) }$ and $c _ { u } ^ { ( t ) }$ denote, respectively, the maximum Strahler number observed among the already explored children of u other than $v ^ { 1 }$ , and the number of children attaining this maximum value. Analogously, $M _ { v } ^ { ( t ) }$ and $c _ { v } ^ { ( t ) }$ denote the maximum Strahler number observed among the children of v and the corresponding multiplicity of this maximum.Then st $\left. \tau _ { v } \right. = M _ { v } ^ { ( t ) } + \mathbb { 1 } [ c _ { v } ^ { ( t ) } \geq 2 ]$ . Let $D = \mathsf { s t } ( v ) - M _ { u } ^ { ( t ) }$

$$
D \left\{ \begin{array} { l l } { { < 0 } } & { { M _ { v } ^ { ( t + 1 ) } = M _ { u } ^ { ( t ) } , c _ { v } ^ { ( t + 1 ) } = c _ { u } ^ { ( t ) } } } \\ { { = 0 } } & { { \left\{ M _ { v } ^ { ( t + 1 ) } \begin{array} { l l } { { = M _ { u } ^ { ( t ) } + \mathbb { 1 } [ c _ { u } ^ { ( t ) } \geq 1 ] } } \\ { { c _ { v } ^ { ( t + 1 ) } } } \end{array} \right. } } \\ { { > 0 } } & { { M _ { v } ^ { ( t + 1 ) } = \mathfrak { s t } ( \tau _ { v } ) , c _ { v } ^ { ( t + 1 ) } = 1 , } } \end{array} \right.\tag{2}
$$

which reduces to

$$
\begin{array} { r l } & { M _ { v } ^ { ( t + 1 ) } = \operatorname* { m a x } ( \mathsf { s t } ( \tau _ { v } ) , M _ { u } ^ { ( t ) } ) + \mathbb { 1 } [ D = 0 ] \mathbb { 1 } [ c _ { u } ^ { ( t ) } > 0 ] } \\ & { ~ c _ { v } ^ { ( t + 1 ) } = \mathbb { 1 } [ D < 0 ] c _ { u } ^ { ( t ) } + \mathbb { 1 } [ D = 0 ] \mathbb { 1 } [ c _ { u } ^ { ( t ) } = 0 ] + \mathbb { 1 } [ D > 0 ] \mathbb { 1 } . } \end{array}
$$

A lattice path of length n with step set ${ \mathcal { S } } \subseteq \mathbb { Z }$ is a sequence $( 0 , s _ { 0 } ) , ( 1 , s _ { 1 } ) , \ldots , ( n , s _ { n } )$ with $s _ { 0 } = 0$ and $s _ { i } - s _ { i - 1 } \in S$ for all $i \geq 1$ ; it is an excursion Flajolet and Sedgewick (2009) if $s _ { i } \geq 0$ throughout and additionally $s _ { n } = 0$ . An excursion with step-set $\begin{array} { r } { S = \{ + 1 , - 1 \} } \end{array}$ is famously known as a Dyck path. By writing U for +1 and D for −1, Dyck paths are presented as Dyck words over {U, D}. The bijection between trees and these paths is well known Goldman and Sundquist (1992). Given $G _ { T }$ , we run a DFS starting from root r and record a U each time the first case of the dynamic dfs occurs, and $\mathbf { a } \ \mathbb { D } ,$ otherwise. Since every one of the $n - 1$ edges is traversed once in each direction, the resulting word $\phi ( G _ { T } )$ has length $2 ( n - 1 )$ . Going the other way, given an excursion w of length $2 m ,$ , we build the tree $\psi ( w ) \colon$ a U creates a new node attached as the next ordered child of the current node and moves the pointer down to it, while a D returns to the parent (see Figure 2). Therefore,

Lemma 1. There exists maps ϕ and ψ, mutually inverse bijections between the set $\mathcal { T } _ { n }$ ofall finite ordered trees and the set $\mathcal { D } _ { n - 1 }$ ofDyck words oflength $2 ( n - 1 )$

The bijection $\phi$ reduces the combinatorial structure of $G _ { T }$ to the geometry of $\phi ( G _ { T } )$ . We make this precise for both ht and wd. Throughout, for $G _ { T } \in { \mathcal { T } } _ { n }$ we let $\left( s _ { 0 } , s _ { 1 } , \ldots , s _ { 2 \left( n - 1 \right) } \right)$ denote the ordinate sequence of the lattice path induced in $\phi ( G _ { T } ) \colon$ concretely, $s _ { 0 } = 0$ and $s _ { i } - s _ { i - 1 } = + 1 \mathrm { o r } - 1$ according to the i-th letter of $\phi ( G _ { T } )$ being U or D, so that $\begin{array} { r } { s _ { i } = \sum _ { j = 1 } ^ { i } q _ { j } } \end{array}$ where $q _ { j } \in \{ + 1 , - 1 \}$ is the signed value of the j-th step.

Proposition 2. For every $G _ { T } \in { \mathcal { T } } _ { n } , \operatorname { h t } ( G _ { T } ) = \operatorname* { m a x } _ { 0 \leq i \leq 2 ( n - 1 ) } s _ { i } .$

Proposition 3. For every $\begin{array} { r } { G _ { T } \in \mathcal { T } _ { n } \mathrm { ~ } w i t h \mathrm { ~ } n \geq 2 , \mathrm { w d } ( G _ { T } ) = \operatorname* { m a x } _ { 0 < l \leq \mathrm { h t } ( G _ { T } ) } | \{ i : s _ { i } = l , s _ { i } - s _ { i - 1 } = 1 \} | . } \end{array}$

Remark 3.1 (Regularity and Relation between the Metric Measure Spaces). The existence and identifiability results above become even more important given the scarcity of similar evidence in a geometric setting, let alone their algorithmic realizability. Plane trees with n edges induce a deterministic bijection with Dyck paths of length 2n via the contour walk (Viennot, 2002) and with Łukasiewicz paths via the depth-first walk (Gall, 2005). However, the regularity of the same is not trivially observed. One can only ensure at the continuum for a continuous excursion $g : [ 0 , 1 ] \to \mathbb { R } _ { + }$ into the metric measure space of trees $\mathcal { T } _ { g } ^ { 2 }$ the 2-Lipschitz-regularity of the forawrd map (in terms of the Gromov-Hausdorff distance) (Evans (2008), Gall (2005), Lemma 2.2.3). Then, the inverse embedding (tree → path) also fails to be Lipschitz, and thus admits no isometric isomorphism at the continuum level. A deterministic isomorphism of lattices (and hence a graph isometry of the associated metric spaces) can only be realized under triangulation (both endowed with the Tamari partial order) (Bernardi and Bonichon, 2009). While we leave checking the regularity of $( \psi , \phi )$ as future work, it can be noted as a first attempt at scrutinizing their CoT realizability.

To construct $\psi ( w )$ from w, we introduce Algorithm 1 that returns the adjacency matrix of the tree on receiving a Dyck path as input (Appendix B provides the proof of correctness).

```perl
Algorithm 1. Reconstruction: Path → Tree (ψ)
Input: A Dyck word w corresponding to lattice path $\left\{ \left( t , s _ { t } \right) \right\} _ { t = 0 } ^ { 2 ( m - 1 ) }$
1 Let $A _ { w }$ be the adjacency matrix of $\psi ( w )$ initialized to 0.
2 nl $ 0 , i  0 .$
3 for $c \in w$
4 if c is U <sup>▷</sup> i.e. $s _ { t + 1 } - s _ { t } = 1$
5 nl $ n l + 1 .$
6 $A _ { w } [ i , n l ] \gets 1 , A _ { w } [ n l , i ] \gets 1 .$
7 $i \gets n l .$
8 else
9 Search $j <$ i s.t. $A _ { w } [ j , i ] = 1$
10 $i  j .$
11 return $A _ { w } .$
```

## 3 Transformer Decoder and CoT

The CoT steps formalize, following Merrill and Sabharwal (2024); Li et al. (2024), the autoregressive nature of the Transformer decoder. We first describe the unique hard-attention layer and then the full decoder architecture. Our framework largely follows Barceló et al. (2025), with two necessary modifications that we identify in our exposition.

Unique Hard-Attention. Given an input sequence $( x _ { 0 } , \ldots , x _ { m - 1 } ) \in ( \mathbb { R } ^ { d } ) ^ { m }$ , let $Q , K \in \mathbb { R } ^ { d ^ { \prime } \times d }$ denote the query and key projection matrices, respectively. The attention score between the position i and the query position $m - 1$ is defined as $a _ { i , m - 1 } = \langle K x _ { i } , Q x _ { m - 1 } \rangle$ . With respect to the m-th query $Q x _ { m - 1 }$ , the scalar $a _ { i , m - 1 }$ quantifies the affinity of the key $K x _ { i } . \operatorname { L e t } j _ { 0 } < j _ { 1 } < \cdots < j _ { p } \in \{ 0 , \ldots , m - 1 \}$ denote the indices, arranged in ascending order, that jointly attain the maximum attention score. Unique hard attention selects the smallest such index $j _ { 0 }$ . Departing from Barceló et al. (2025), we retain the standard value matrix<sup>3</sup> $V \in \mathbb { R } ^ { d ^ { \prime \prime } \times d }$ , so that the attention produces $V x _ { j _ { 0 } }$

Multi-Head Attention. A multi-head extension uses H attention heads, where head h $, ( h \in [ H ] )$ is parameterized by $( Q _ { h } , K _ { h } , V _ { h } )$ . Let ${ j _ { 0 } ^ { ( h ) } }$ be the index selected by head $h ;$ the combined output is

$$
\alpha ^ { \prime \prime } = W _ { O } \cdot \left[ \left( V _ { 1 } x _ { j _ { 0 } ^ { ( 1 ) } } \right) ^ { \top } \Big \Vert \mathbf { \Sigma } \cdot \cdot \Big \Vert \mathbf { \Sigma } \left( V _ { H } x _ { j _ { 0 } ^ { ( H ) } } \right) ^ { \top } \right] ^ { \top } ,
$$

where ∥ denotes vector concatenation, $W _ { O }$ is the output projection matrix, and $\boldsymbol { \alpha } ^ { \prime \prime } \in \mathbb { R } ^ { d }$

Single-Layer Decoder. A position-wise feed-forward network is applied to the residual $\beta + x _ { m - 1 }$ via two successive linear projections interleaved with a ReLU nonlinearity, where ReLU $( r ) : = \operatorname* { m a x } ( 0 , r )$ for $r \in \mathbb { R } .$ , applied coordinatewise for $r \in \mathbb { R } ^ { d }$ . The single-layer decoder thus computes

$$
L ( x _ { 0 } , \ldots , x _ { m - 1 } ) = W _ { 2 } \mathrm { \ R e L U } ( W _ { 1 } ( \beta + x _ { m - 1 } ) ) ,
$$

for weight matrices $W _ { 1 } \in \mathbb { R } ^ { d ^ { \prime \prime \prime } \times d }$ and $W _ { 2 } \in \mathbb { R } ^ { d \times d ^ { \prime \prime \prime } }$ . In our constructions, we will often use $\alpha ^ { \prime }$ to denote $V x$ . Each input vector $x _ { i }$ is constructed from a word embedding $\mathrm { W E } ( \sigma _ { i } )$ and a positional encoding $\mathrm { p o s } ( i )$ via a composition $x _ { i } =$ $g ( \mathrm { \mathrm { W E } } ( \sigma _ { i } ) , \mathrm { p o s } ( i ) )$ . Unlike Barceló et al. (2025), we adopt the simplification $g ( \mathrm { W E } ( \tilde { \sigma _ { i } } ) , \tilde { \mathrm { p o s } ( i ) } ) = \mathrm { W E } ( \sigma _ { i } ) = : x _ { i } . \ \mathrm { A }$ single-layer Transformer decoder, upon receiving input $( x _ { 0 } , \ldots , x _ { m - 1 } ) \in ( \mathbb { R } ^ { d } ) ^ { m }$ and an initial eos token $y _ { 0 } \in \mathbb { R } ^ { d }$ generates output sequence $\{ y _ { t } \in \mathbb { R } ^ { d } \} _ { t = 1 } ^ { \infty }$ via the recurrence

$$
y _ { t } = L ( x _ { 0 } , \ldots , x _ { m - 1 } , y _ { 0 } , \ldots , y _ { t - 1 } ) , \quad t \geq 1 .
$$

That is, at each step t, the decoder appends its most recent output $y _ { t - 1 }$ to the running context and applies L to the resulting extended sequence.

Multi-Layer Extension. A depth-ℓ Transformer decoder is obtained by composing ℓ such single-layer decoders $L ^ { ( 1 ) } , \ldots , L ^ { ( \ell ) }$ , each with its own independent parameter set $\{ Q _ { h } ^ { ( k ) } , K _ { h } ^ { ( k ) } , V _ { h } ^ { ( k ) } , \bar { W } _ { O } ^ { ( k ) } , \bar { W } _ { 1 } ^ { ( k ) } , W _ { 2 } ^ { ( k ) } \} _ { h = 1 } ^ { - }$ for layer k $\in$ [ℓ]. Given the input context $( x _ { 0 } , \ldots , x _ { m - 1 } )$ , the layers process the sequence in depth-first order: at generation step t, the intermediate representations are computed as

$$
z _ { m } ^ { ( 0 ) } : = x _ { m - 1 } , z _ { m } ^ { ( k ) } : = L ^ { ( k ) } \Bigl ( z _ { 1 } ^ { ( k - 1 ) } , \ldots , z _ { m } ^ { ( k - 1 ) } \Bigr ) ,
$$

for $k \in [ \ell ]$ and the final output at step t is $y _ { t } : = z _ { m _ { t } } ^ { ( \ell ) }$ , where $m _ { t } = m + t$ denotes the current sequence length. The newly produced token $y _ { t }$ is appended to the context, after which the full ℓ-layer computation is repeated for step $t + 1$

Definition 3.1 (CoT Realization). Let A be an algorithm that, on input, I say, produces a sequence of states $\{ \mathsf { s } _ { t } \} _ { t = 1 } ^ { T } .$ where T denotes the number of iterations until termination. Let $\pi : \mathbb { R } ^ { d }  \mathbb { R } ^ { d _ { s } }$ denote the projection onto the designated output coordinates. We say that a decoder realizes A via CoT if there exists a parameter assignment such that, for every valid input and every ${ \bf \dot { \tau } } t \in [ T ] , \pi ( y _ { t } ) = \mathrm { e n c } ( s _ { t } )$ , where enc is a fixed encoding of algorithmic states into $\mathbb { R } ^ { d _ { s } }$ . The generation runs for exactly T steps; the step count T is determined a priori, and hence termination follows immediately.

## 4 Main Results

This section is organized as follows. First, we present the construction of the graph traversal techniques (see section 4.1). Section 4.2 then introduces the theorems necessary to establish the main results, as summarized in Figure 1. The bilinear maps employed in our constructions, following Rizvi-Martel et al. (2024), are in general not unique, as can be seen by comparing Theorem 8 and 11. However, substituting these bilinear maps with FFN blocks from a Transformer layer would, in several cases where merging independent operations is infeasible, alter the required number of layers.

Bilinear maps have been employed in the study of Transformers in both theoretical analyses (Rizvi-Martel et al., 2024) and empirical investigations, particularly within the contexts of multi-modal and multi-view learning (Li et al., 2017; Gao et al., 2016). In our work, we adopt analogous bilinear projection mechanisms as fundamental components of our constructions. We apply linear, affine, and bilinear maps to independent resultant blocks using a single transformation. In Appendix C, we show that all of these can be written as a single bilinear map. Also, any scalar–vector operation is applied elementwise to the vector.

## 4.1 Results on Graph Traversal

Theorem 5. A two-layer, two-head Transformer decoder can simulate the depth-first traversal DFS on any simple directed graph $G = ( { \dot { V } } , E , A )$ in $O ( | V | + | E | )$ chain-of-thought steps.

ProofSketch. The construction realizes the dynamic $\mathsf { d f s } ^ { ( t ) }$ as in (1) and proceeds in two layers. The first layer uses a trivial attention mechanism and determines, via its feed-forward block, whether the current vertex has an unvisited neighbor — distinguishing a traverse from a backtrack. The second layer executes the chosen operation through two mutually exclusive attention heads, one resolving the next unvisited neighbor and the other retrieving the parent vertex (see Appendix D). □

When the underlying graph is a tree $G _ { T }$ , the number of required CoT steps is $2 ( n - 1 )$ . The DFS realization plays a central role in two distinct respects: i. The mapping ϕ coincides exactly with the execution of the procedure DFS on the tree $G _ { T }$ . To obtain the associated Dyck path (or Dyck word), it suffices to extend the token embedding by one additional component that records +1 (or U) and −1 (or D) exclusively, in accordance with the traverse and backtrack operations, in the final layer. ii. The same DFS realization is also reused in the implementation of (2), thereby yielding the Strahler number of a tree as stated in Theorem 8.

Theorem 6. A two-layer, single-head Transformer decoder can simulate the Dijkstra algorithm Dkst on any simple connected directed graph $G = \overline { { ( V , E , A ) } }$ with $\mathsf { A } = \mathbb { R } _ { > 0 }$ in exactly $| V | - 1$ chain-of-thought steps.

ProofSketch. The proof proceeds by implementing the dynamics of $\mathsf { D } \mathsf { k } \mathsf { s t } ^ { ( t ) }$ as described in subsection 2.1. In the first layer, the attention collects the edge weights by identifying the current vertex from input tokens. Since $\mathsf { A } \in \mathbb { R } _ { > 0 }$ , its FFN block updates the distance vector Dist by applying the relaxation operation to all neighboring vertices. The second layer then uses its attention mechanism to select the unvisited vertex with the smallest Dist, updates its representation, and marks the selected vertex as visited (see Appendix E). □

The classical implementation of Dijkstra algorithm requires $O ( n ^ { 2 } )$ time, whereas efficient variant runs in $O ( | E | +$ n log n). In contrast, our implementation requires at most n − 1 CoT steps. This yields a substantial computational advantage: the Transformer avoids explicitly searching for the minimum-distance unvisited vertex by using constant-time attention. We use this theorem, along with Observation 1, to compute wd in Theorem 13.

## 4.2 CoT Realization of Complexity Measures

After proving Algorithm 1 in Theorem 7 (see Appendix F), Theorem 8 and 11 present the proofs for computing the Strahler number treating tree and Dyck path as inputs, respectively, under CoT (Appendix G). Theorem 13 and 16 provide the proofs for computing the measure wd (Appendix H).

Theorem 7. A single-layer, two-head Transformer decoder can simulate Algorithm 1 in the $2 ( m - 1 ) \mathrm { C o T } \mathrm { - s t e p s } ^ { 4 }$

Proof Sketch. Each step executes one iteration of the loop on line 3, reading the symbol $w _ { t }$ and updating $( i , n l , A _ { w } )$ in place. A single attention head reads $w _ { t }$ by matching the cursor pos against the input tokens, returning its signed increment $q _ { t } \in \{ + 1 , - 1 \}$ ; a second head, querying on the current vertex, retrieves from that vertex’s creation token the parent needed on a D step (line 9). The feed-forward block then branches on the sign of $q _ { t }$ entirely within a single $\mathrm { R e L U }$ , via $\mathrm { R e L U } ( v + q _ { t } - 1 ) = \mathbb { 1 } [ q _ { t } = + 1 ] v \colon$ on a U it increments nl, writes the symmetric edge $\{ i , n l \}$ as the one quadratic feature of $W _ { 1 }$ , and advances i to the new vertex (lines 5–6); on a D it moves i to the retrieved parent. The cursor advances each step, and the chain halts when the path exhausts. Induction on t shows $\pi _ { A } ( y _ { t } )$ equals the adjacency matrix of $\psi ( w )$ after reading $w _ { 0 } \cdot \cdot \cdot w _ { t - 1 }$ □

Theorem 8. A four-layer, two-head Transformer decoder can compute ${ \sf s t } ( G _ { T } )$ , the Strahler number of a tree $G _ { T }$ during the simulation of DFS in exactly $2 n - 1$ CoT steps, where n denotes the number of vertices in $G _ { T }$

ProofSketch. The decoder of Theorem 5 traverses $G _ { T }$ . We extend its embedding alone, giving each token a pair $( \pi _ { M } , \pi _ { c } ) ;$ : the running maximum of the Strahler numbers of the current vertex’s finished children, and the count attaining it. A timestamp $\pi _ { f l g _ { 4 } }$ turns the backtrack head’s leftmost selection into a rightmost one, so it resolves the parent as last left and reads the accumulator it then carried. The feed-forward blocks of two further layers fold the finished child into this accumulator. The comparison against the parent’s maximum is decided through indicators of $D = \mathsf { s t } ( v ) - M _ { u }$ and of the count, laid down as differences of rectified blocks. A carry lifts the maximum when the count saturates, keeping the fold bilinear. The accumulator is never finalized in place; the final backtrack folds the source under the same invariant, giving $\pi _ { M } ( y _ { 2 n - 1 } ) = { \mathsf { s t } } ( G _ { T } )$ in $2 n - 1$ steps. □

Theorem 11. A four-layer, single-head Transformer decoder can compute $\mathsf { s t } ( \psi ( w ) )$ , the Strahler number of a tree corresponding to the Dyck word w in exactly n CoT steps, where $n = | w |$

The proof of the above theorem builds on the same techniques as before, in particular the implementation of (2) from Theorem 8. For this reason, it is only outlined in Appendix G, where we provide the additional arguments needed to handle paths as input. Note that while Theorem 8 directly computes the st number of the tree $G _ { T }$ in its final CoT step, the above theorem provides the values M and c in the designated positions as specified by $\pi _ { M } ( y _ { n } )$ and $\pi _ { c } ( y _ { n } )$ so that $\mathsf { s t } ( \psi ( w ) ) = \pi _ { M } ( \bar { y _ { n } } ) + \mathbb { 1 } \left[ \pi _ { c } ( y _ { n } ) \geq 2 \right]$

Theorem 13. A three-layer single-head Transformer decoder can find the width of a tree wd $. ( G _ { T } )$ during the simulation of Dkst in exactly $n - 1$ steps, where n denotes the number of vertices in $G _ { T }$

ProofSketch. We augment the embeddings of the tokens of Theorem 6 by appending three scalar coordinates $( \pi _ { d i f } , \pi _ { t w d } , \pi _ { m w d } )$ , initialized to (0, 1, 1) at the root, tracking a BFS level-change indicator, the running node count at the current level, and the global maximum width. Layers 1-2 replicate the Dijkstra construction, with the sole modification that $W _ { 2 } ^ { ( 2 ) }$ additionally writes $\begin{array} { r } { \pi _ { d i f } ( y _ { t } ^ { ( 2 ) } ) \gets \sum _ { i } \pi _ { b u f } ( y _ { t } ^ { ( 1 ) \prime } ) [ i ] - \pi _ { c r d } ( y _ { t } ) } \end{array}$ , the difference between the BFS depths of the current and preceding decoded nodes. Since $G _ { T }$ carries weights in $\{ 1 , \infty \}$ and nodes are decoded in non-decreasing BFS-depth order, Observation 1 guarantees $\pi _ { d i f } \in 0 , 1$ , with $\pi _ { d i f } = 1$ signalling a transition to a strictly deeper level. Layer 3 employs a trivial attention sub-layer followed by an FFN that, when $\pi _ { d i f } = 0$ increments $\pi _ { t w d }$ and updates $\pi _ { m w d }$ with ma $\mathrm { x } ( \pi _ { m w d } , \pi _ { t w d } )$ , and when $\pi _ { d i f } = 1$ , commits the current count to π<sub>mwd</sub> and resets $\pi _ { t w d }$ to 1. After $n - 1$ decoding steps, each BFS level’s count has been committed to $\pi _ { m w d }$ and thus $\pi _ { m w d } ( y _ { n - 1 } ) = \mathrm { w d } ( G _ { T } )$ □

Theorem 16. A two-layer single-head Transformer decoder can find wd $( \psi ( w ) )$ for a Dyck word w in exactly n CoT steps, where $| w | = n$

ProofSketch. The embedding dimension $\begin{array} { r } { d = \frac { 3 n } { 2 } + 4 } \end{array}$ is partitioned into a one-hot position block, a step scalar $q _ { t } , \mathbf { a }$ cumulative height scalar ht, a count block $p \in \mathbb { R } ^ { \frac { n } { 2 } + 1 }$ where $p _ { i }$ tallies up-steps reaching height i, and a width scalar wd. In the first Layer, attention retrieves the current step $q _ { \tau }$ via a shift matrix, and the FFN applies a discrete pulse function (Lemma 14) to increment exactly the $p _ { i }$ entry indexed by the current height ht. In the final layer, the FFN performs a global max-aggregation (Lemma 15) updating wd to max $p _ { i }$ while zeroing the step scalar. Correctness follows since each $p _ { i }$ counts nodes at depth i in $\psi ( w )$ , so the final wd equals wd $( \psi ( w ) )$ as claimed in Proposition 3. □

A natural question arises regarding the optimal use of layers: given a tree $G _ { T } ,$ , is there an advantage to directly computing its width $\mathrm { w d } ( G _ { T } )$ using a three-layer Transformer (as in Theorem 13), rather than breaking the process down? For instance, an alternative composite setup might combine a two-layer, two-head Transformer to implement the map ϕ (in Theorem 5) with a two-layer, single-head Transformer to compute width on the corresponding Dyck path (in Theorem 16). If CoT is closed under composition and layer counts add linearly, the first approach is strictly more economical in layer count. Moreover, directly employing the three-layer Transformer offers another benefit: it reduces the required CoT steps significantly compared to the composite method.

Remark 3.2 (Closure under Composition). Given the fact that CoT-Transformers fail to solve Compositional Reasoning Questions (CRQ) unless CoT tokens are allowed to grow accommodating compositional depth (Yehudai et al., 2025), it is clear that the composition does not replicate itself. However, the same constructive realizability witness result for composition over tree-structured tasks hints towards a linear complexity class. Notably, our construction preserves linear token growth. On the other hand, since ensuring CoT-learnability essentially boils down to the finiteness of the VCdimension of the base classes $\mathcal { F } _ { i }$ (Joshi et al., 2025), it would be sufficient for them to satisfy a VC upper bound showing linear aggregation in the spirit of Alon et al. (2021), Proposition 2: $\begin{array} { r } { \mathrm { V C } ( G ( \mathcal { F } _ { 1 } , \ldots , \mathcal { F } _ { k } ) ) \lesssim \mathrm { V C } ( G ) + \sum _ { i = 1 } ^ { k } \mathrm { V C } ( \mathcal { F } _ { i } ) } \end{array}$ to prove closure under composition along those lines, where G specifies composition, $k > 0$

## 5 Conclusion

To our knowledge, the CoT realizations developed here are the first of their kind in the literature. This immediately invites the question of optimality: identifying constructions that minimize embedding dimension, layer depth, and number of attention heads constitutes a natural and important direction for future work. A second open question concerns the closure properties of CoT under the tree–path bijection. The CoT realizations of the Strahler number for path and tree inputs exhibit remarkably little structural synergy — beyond their shared adherence to (2) — despite the bijective correspondence between the two input representations via ϕ and ψ. A further observation concerns the number of heads required in Theorem 8 and Theorem 11. Intuitively, a single head can scarcely distinguish between a traverse and a backtrack along the same edge, whereas the linear progression of U and D in a Dyck path makes this distinction straightforward, so one head suffices. This also explains the two heads in Theorem 7: reconstructing the tree from the path re-encodes each D with its corresponding U, folding the information of both the traverse and backtrack steps back onto a single edge – and it is this folding that necessitates the second head.

## Limitations

The constructions presented herein are not necessarily unique, and we acknowledge that alternative formulations satisfying the same theoretical requirements may well exist. As one illustration, Theorem 8 introduces bilinear gating at the final layer to suppress redundant operations on the M and c blocks during a traverse step; Lemma 10, however, demonstrates that bilinearity can be entirely circumvented to achieve the same effect in the context of Theorem 11. As one limitation, several of our constructions invoke the bilinear operation. Following Rizvi-Martel et al. (2024), however, this usage stays within the established conventions of this line of work rather than constituting a departure from them. In this regard, unlike Merrill and Sabharwal (2024); Qiu et al. (2025), our constructions forgo supplementary computational primitives such as layer normalization. Rather than a shortcoming, this economy proved crucial for constructing CoT Transformers that reach NC<sup>1</sup>, showing that such augmentations are not a necessary architectural ingredient for CoT constructions. Additionally, we note that Theorem 11 concludes with the final values M and c, the closing operation $M + \mathbb { 1 } [ c \geq 2 ]$ that fixes the Strahler number sits marginally outside the CoT iteration. Furthermore, there is also a classical correspondence with plane trees and Łukasiewicz paths. We have not studied CoT realizations for these representations, which may yield additional architectural insights in the broader theory of computation.

## References

Louigi Addario-Berry, Marie Albenque, Serte Donderwinkel, and Robin Khanfir. 2024. Refined Horton-Strahler numbers I: a discrete bijection. Preprint, arXiv:2406.03025.

Louigi Addario-Berry, Luc Devroye, and Svante Janson. 2013. Sub-Gaussian tail bounds for the width and height of conditioned Galton–Watson trees. The Annals ofProbability, 41(2):1072 – 1087.

Alfred V Aho and John E Hopcroft. 1974. The Design and Analysis of Computer Algorithms. Pearson Education India.

Noga Alon, Alon Gonen, Elad Hazan, and Shay Moran. 2021. Boosting simple learners. In Proceedings ofthe 53rd Annual ACM SIGACT Symposium on Theory of Computing, STOC 2021, page 481–489, New York, NY, USA. Association for Computing Machinery.

Pablo Barceló, Alexander Kozachinskiy, Anthony Widjaja Lin, and Vladimir Podolskii. 2024. Logical Languages Accepted by Transformer Encoders with Hard Attention. In The Twelfth International Conference on Learning Representations.

Pablo Barceló, Alexander Kozachinskiy, and Tomasz Steifer. 2025. Ehrenfeucht-Haussler Rank and Chain of Thought. In Forty-second International Conference on Machine Learning.

Olivier Bernardi and Nicolas Bonichon. 2009. Intervals in Catalan lattices and realizers of triangulations. Journal of Combinatorial Theory, Series A, 116(1):55–75.

David Chiang, Peter Cholak, and Anand Pillay. 2023. Tighter Bounds on the Expressivity of Transformer Encoders. In Proceedings of the 40th International Conference on Machine Learning, ICML’23. JMLR.org.

Yogesh Dahiya and Meena Mahajan. 2021. On (Simple) Decision Tree Rank. In 41st IARCS Annual Conference on Foundations of Software Technology and Theoretical Computer Science (FSTTCS 2021), volume 213 of Leibniz International Proceedings in Informatics (LIPIcs), pages 15:1–15:16, Dagstuhl, Germany. Schloss Dagstuhl – Leibniz-Zentrum für Informatik.

Artur Back De Luca and Kimon Fountoulakis. 2024. Simulation of Graph Algorithms with Looped Transformers. In International Conference on Machine Learning, pages 2319–2363. PMLR.

Reinhard Diestel. 2017. Graph Theory, 5 edition, volume 173 of Graduate Texts in Mathematics. Springer, Heidelberg.

E. W. Dijkstra. 1959. A Note on Two Problems in Connexion with Graphs, 1 edition, page 287–290. Association for Computing Machinery, New York, NY, USA.

R. P. Dilworth. 1950. A Decomposition Theorem for Partially Ordered Sets. Annals of Mathematics, 51(1):161–166.

Debanjan Dutta, Anish Chakrabarty, Faizanuddin Ansari, and Swagatam Das. 2025. On the Existence of Universa Simulators of Attention. arXiv preprint arXiv:2506.18739.

Steven Neil Evans. 2008. Probability and Real Trees: École D’Été de Probabilités de Saint-Flour XXXV-2005. Springer.

P. Flajolet, J.C. Raoult, and J. Vuillemin. 1979. The Number of Registers Required for Evaluating Arithmetic Expressions. Theoretical Computer Science, 9(1):99–125.

Philippe Flajolet and Robert Sedgewick. 2009. Analytic Combinatorics. Cambridge University Press, Cambridge.

Jean-François Le Gall. 2005. Random trees and applications. Probability Surveys, 2(none):245 – 311.

Moses Ganardi and Markus Lohrey. 2026. On the Complexity of Computing Strahler Numbers. In 43rd International Symposium on Theoretical Aspects ofComputer Science, STACS 2026, Grenoble, France, March 9-13, 2026, LIPIcs, pages 41:1–41:22. Schloss Dagstuhl - Leibniz-Zentrum für Informatik.

Yang Gao, Oscar Beijbom, Ning Zhang, and Trevor Darrell. 2016. Compact Bilinear Pooling. In Proceedings ofthe IEEE Conference on Computer Vision and Pattern Recognition (CVPR).

Ira Gessel and Da-Lun Wang. 1979. Depth-first Search as a Combinatorial Correspondence. Journal ofCombinatorial Theory, Series A, 26(3):308–313.

Angeliki Giannou, Shashank Rajput, Jy-Yong Sohn, Kangwook Lee, Jason D. Lee, and Dimitris Papailiopoulos. 2023. Looped Transformers as Programmable Computers. In Proceedings ofthe 40th International Conference on Machine Learning, pages 11398–11442. PMLR.

Jay R Goldman and Thomas Sundquist. 1992. Lattice Path Enumeration by Formal Schema. Advances in Applied Mathematics, 13(2):216–251.

Michael Hahn. 2020. Theoretical Limitations of Self-Attention in Neural Sequence Models. Transactions of the Associationfor Computational Linguistics, 8:156–171.

Yiding Hao, Dana Angluin, and Robert Frank. 2022. Formal Language Recognition by Hard Attention Transformers: Perspectives from Circuit Complexity. Transactions ofthe Associationfor Computational Linguistics, 10:800–810.

Nirmit Joshi, Gal Vardi, Adam Block, Surbhi Goel, Zhiyuan Li, Theodor Misiakiewicz, and Nathan Srebro. 2025. A Theory of Learning with Autoregressive Chain of Thought. arXiv preprint arXiv:2503.07932.

Yanghao Li, Naiyan Wang, Jiaying Liu, and Xiaodi Hou. 2017. Factorized Bilinear Models for Image Recognition. In 2017 IEEE International Conference on Computer Vision (ICCV), pages 2098–2106.

Zhiyuan Li, Hong Liu, Denny Zhou, and Tengyu Ma. 2024. Chain of Thought Empowers Transformers to Solve Inherently Serial Problems. In The Twelfth International Conference on Learning Representations.

Charles London and Varun Kanade. 2026. Pause Tokens Strictly Increase the Expressivity of Constant-Depth Transformers. Advances in Neural Information Processing Systems, 38:89765–89795.

Will Merrill and Ashish Sabharwal. 2026. A Little Depth Goes a Long Way: the Expressive Power of Log-Depth Transformers. Advances in Neural Information Processing Systems, 38:95315–95339.

William Merrill and Ashish Sabharwal. 2024. The Expressive Power of Transformers with Chain of Thought. In The Twelfth International Conference on Learning Representations.

Franz Nowak, Anej Svete, Alexandra Butoi, and Ryan Cotterell. 2024. On the Representational Capacity of Neural Language Models with Chain-of-Thought Reasoning. In Proceedings ofthe 62nd Annual Meeting ofthe Association for Computational Linguistics (Volume 1: Long Papers), pages 12510–12548.

Binghui Peng, Srini Narayanan, and Christos Papadimitriou. 2024. On Limitations of the Transformer Architecture. In First Conference on Language Modeling.

Jorge Pérez, Javier Marinkovic, and Pablo Barceló. 2019. On the Turing Completeness of Modern Neural Network´ Architectures. arXiv preprint arXiv:1901.03429.

Jorge Pérez, Pablo Barceló, and Javier Marinkovic. 2021. Attention is Turing-Complete. Journal of Machine Learning Research, 22(75):1–35.

Ruizhong Qiu, Zhe Xu, Wenxuan Bao, and Hanghang Tong. 2025. Ask, and it shall be given: On the turing completeness of prompting. In International Conference on Learning Representations, volume 2025, pages 6286–6309.

Michael Rizvi-Martel, Maude Lizaire, Clara Lacroce, and Guillaume Rabusseau. 2024. Simulating Weighted Automata over Sequences and Trees with Transformers. In Proceedings of The 27th International Conference on Artificial Intelligence and Statistics, volume 238, pages 2368–2376. PMLR.

Clayton Sanford, Bahare Fatemi, Ethan Hall, Anton Tsitsulin, Mehran Kazemi, Jonathan Halcrow, Bryan Perozzi, and Vahab Mirrokni. 2024. Understanding Transformer Reasoning Capabilities via Graph Algorithms. Advances in Neural Information Processing Systems, 37:78320–78370.

Clayton Sanford, Daniel J Hsu, and Matus Telgarsky. 2023. Representational strengths and limitations of transformers. In Advances in Neural Information Processing Systems, volume 36, pages 36677–36707.

Arthur N. Strahler. 1957. Quantitative Analysis of Watershed Geomorphology. Transactions, American Geophysical Union, 38(6):913–920.

Lena Strobl, William Merrill, Gail Weiss, David Chiang, and Dana Angluin. 2024. What Formal Languages Can Transformers Express? A Survey. Transactions ofthe Associationfor Computational Linguistics, 12:543–561.

Xavier Gérard Viennot. 2002. A Strahler bijection between Dyck paths and planar trees. Discrete Mathematics, 246(1-3):317–329.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, brian ichter, Fei Xia, Ed H. Chi, Quoc V Le, and Denny Zhou. 2022. Chain of Thought Prompting Elicits Reasoning in Large Language Models. In Advances in Neural Information Processing Systems.

Gail Weiss, Yoav Goldberg, and Eran Yahav. 2021. Thinking Like Transformers. In Proceedings ofthe 38th International Conference on Machine Learning, volume 139 of Proceedings ofMachine Learning Research, pages 11080–11090. PMLR.

Gilad Yehudai, Noah Amsel, and Joan Bruna. 2025. Compositional Reasoning with Transformers, RNNs, and Chain of Thought. In The Thirty-ninth Annual Conference on Neural Information Processing Systems.

Haoyu Zhao, Yiming Duan, Junfeng Hao, and Sanjeev Arora. 2023. Do Transformers Parse while Predicting the Masked Word? In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 16513–16542. Association for Computational Linguistics.

Hanlin Zhu, Shibo Hao, Zhiting Hu, Jiantao Jiao, Stuart Russell, and Yuandong Tian. 2026. Emergence of Superposition: Unveiling the Training Dynamics of Chain of Continuous Thought. In The Fourteenth International Conference on Learning Representations.

## A Related Works

A substantial body of work delineates what Transformers can compute in a single forward pass, for encoders (Hahn, 2020; Pérez et al., 2021; Weiss et al., 2021; Hao et al., 2022; Chiang et al., 2023; Sanford et al., 2023; Barceló et al., 2024) and decoders alike (Pérez et al., 2021; Merrill and Sabharwal, 2024; Peng et al., 2024; Barceló et al., 2025), surveyed by Strobl et al. (2024). The recurring conclusion is a ceiling: unique and generalized hard-attention encoders recognize only languages in $\mathsf { A C } ^ { 0 }$ (Hao et al., 2022), which already excludes the Dyck languages. Later work sharpens the picture to exact automata-theoretic characterizations (Rizvi-Martel et al., 2024) and depth hierarchies (Merrill and Sabharwal, 2026). As such, it is fair to say that we work in the most restrictive of these regimes (unique hard attention, no layer normalization) and the expressive power we exhibit is attributable to CoT alone.

Building on Pérez et al.’s result that hard-attention decoders with unboundedly many steps compute any decidable language, Merrill and Sabharwal refine this into a step-counted hierarchy (log steps → L, linear steps ${ \dot {  } } { \mathsf { N C } } ^ { 1 } ) ;$ Li et al. (2024) obtain constant-depth constructions for P/poly, Nowak et al. (2024) characterize CoT-augmented language models, and related gains are known for pause and filler tokens (London and Kanade, 2026) and continuous thought (Zhu et al., 2026). While these characterize classes, concrete problems instantiating them remain scarce. In this context, we reiterate Barceló et al.’s contribution in showing that the EH rank is exactly the minimum number of CoT steps for a single-layer hard-attention decoder. Since the EH rank of a binary decision tree obeys the same recurrence as its Strahler number (Dahiya and Mahajan, 2021), and computing the latter from a term representation is ${ \mathsf { N C } } ^ { 1 }$ -complete (Ganardi and Lohrey, 2026), branching complexity probes precisely the linear-step regime rather than being an arbitrary target.

A parallel thread constructs Transformers that execute procedures: looped architectures as programmable computers (Giannou et al., 2023), parsing during masked prediction (Zhao et al., 2023), universal simulation of attention (Dutta et al., 2025), and simulation of weighted automata over sequences and trees by bilinear maps (Rizvi-Martel et al., 2024), which we adopt as a primitive. For graph algorithms specifically, De Luca and Fountoulakis (2024) simulate DFS, BFS, Dijkstra, and Kosaraju by looped Transformers with graph-interacting heads and graph-size-independent parameter count; Sanford et al. (2024) chart architecture tradeoffs for graph reasoning. However, all of the above evade the CoT decoder model. Looping is not autoregressive generation, and neither line counts CoT steps. Within CoT, Zhu et al. (2026) study reachability (to identify which nodes in a graph can be reached from a specified source node) empirically without realizing the traversal, and Barceló et al. (2025) invoke exhaustive tree traversal as a proof device without constructing it or handling weights. We close this gap with explicit decoder constructions, making traversal an object of study rather than scaffolding. Notably, Ganardi and Lohrey’s own ${ \mathsf { N C } } ^ { 1 }$ upper bound proceeds by a heavy-subtree-first DFS, so a CoT realization of DFS is the natural bridge to their result.

On the tree–path correspondence end, while it admits several classical realizations (Viennot, 2002; Gall, 2005), its regularity is delicate even at the continuum (Evans, 2008), and none has previously been examined for CoT realizability.

## B Missing Proofs of Lemmas and Propositions

Proof of Lemma 1. Suppose $\phi ( T )$ is not a Dyck word for some $T \in \mathcal T _ { n }$ . Then for some edge $( ( i , s _ { i } ) , ( i + 1 , s _ { i + 1 } ) )$ on ϕ(T) labeled with D $s _ { i + 1 } < 0 \mathrm { o r } s _ { 2 ( n - 1 ) } \ne \bar { 0 }$ . If the former is the case, there must be more −1 steps than +1 steps. If this is the case, some edges in T are backtracked more than they are traversed, which is absurd. On the other hand, had $s _ { 2 ( n - 1 ) } \neq 0$ , specifically $s _ { 2 ( n - 1 ) } > 0$ , then we would not have backtracked some edges. This would, by definition, violate the DFS traversal scheme.

We verify $\psi ( \phi ( T ) ) = T$ by structural induction on n. The case $n = 1$ is clear: $\phi ( T ) = \varepsilon$ and $\psi ( \varepsilon )$ is a lone root r. If the root has ordered children subtending subtrees $t _ { 1 } , \ldots , t _ { k }$ , then ${ \phi } ( t ) = \mathbb { U } { \phi } ( t _ { 1 } ) \mathbb { D } \cdot \cdot \cdot \mathbb { U } { \phi } ( t _ { k } ) \mathbb { D }$ . Reading this with $\psi ,$ the j-th block $\mathtt { U } \phi ( t _ { i } ) \mathtt { D }$ creates a new child of the root and reconstructs $t _ { j }$ beneath it by the inductive hypothesis. The identity $\phi ( \psi ( w ) ) = w$ follows by the same argument on |w|. □

ProofofProposition 2. Since $G _ { T }$ and $\phi ( G _ { T } )$ are both finite, both $\mathrm { h t } ( G _ { T } )$ and ma $\mathbb { X } _ { 0 \leq i \leq 2 ( n - 1 ) } s _ { i }$ are finite. Let $u _ { i } \in V ( G _ { T } )$ denote the node at which the DFS pointer rests after the processing step i, so $u _ { 0 } = r$ . We claim

$$
s _ { i } = \mathrm { d e p t h } ( u _ { i } ) \quad ( 0 \leq i \leq 2 ( n - 1 ) ) , \quad \mathrm { ~ a n d ~ } \quad s _ { i } - s _ { i - 1 } = 1 \quad ( 1 \leq i \leq 2 ( n - 1 ) ) .\tag{3}
$$

The base case $s _ { 0 } = 0 = \mathrm { d e p t h } ( r )$ is immediate. For the inductive step, a U at position i moves the pointer to a child of $u _ { i - 1 }$ , giving dept $\operatorname { \mathrm { \Omega } } _ { 1 } ( u _ { i } ) = \operatorname { d e p t h } ( u _ { i - 1 } ) + 1 = s _ { i - 1 } + 1 = s _ { i } .$ . A D at position i moves it to the parent of $u _ { i - 1 }$ , giving dept $\mathrm { h } ( u _ { i } ) = \mathrm { d e p t h } ( u _ { i - 1 } ) - 1 = s _ { i - 1 } - 1 = s _ { i }$ . This establishes (3).

Since DFS visits every node of $G _ { T }$ , the map $i \mapsto u _ { i }$ is surjective onto $V ( G _ { T } )$ . Together with (3),

$$
\operatorname* { m a x } _ { 0 \leq i \leq 2 ( n - 1 ) } s _ { i } = \operatorname* { m a x } _ { 0 \leq i \leq 2 ( n - 1 ) } \operatorname* { d e p t h } ( u _ { i } ) = \operatorname* { m a x } _ { v \in V ( G _ { T } ) } \operatorname* { d e p t h } ( v ) = \operatorname { h t } ( G _ { T } ) .
$$

ProofofProposition 3. Consider l with $1 \leq l \leq \mathrm { h t } ( G _ { T } )$ , and define

$$
\begin{array} { r l } & { A _ { l } : = \{ v \in V ( G _ { T } ) : \mathrm { d e p t h } ( v ) = l \} , } \\ & { B _ { l } : = \{ i : s _ { i } = l , \ s _ { i } - s _ { i - 1 } = 1 \} . } \end{array}
$$

We construct a bijection $\varphi _ { l } : A _ { l } \to B _ { l }$ . For each $v \in A _ { l }$ , DFS first arrives at v from its parent via a unique U step; let $i _ { v }$ denote the index of this step. By Proposition $2 , s _ { i _ { v } } = \mathrm { d e p t h } ( u _ { i _ { v } } ) = \mathrm { d e p t h } ( v ) = \bar { l } $ , and by construction $s _ { i _ { v } } - s _ { i _ { v } - 1 } = 1 , \mathrm { s o } i _ { v } \in B _ { l }$ . Set $\varphi _ { l } ( v ) : = i _ { v }$

Injectivity. DFS visits each node for the first time at a distinct step, so $v \ne$ w implies $i _ { v } \neq i _ { w }$

Surjectivity. Every $i \in B _ { l }$ is a U step satisfying $s _ { i } = l _ { : }$ , so the pointer moves to $u _ { i }$ with $\mathrm { d e p t h } ( u _ { i } ) = l$ by Proposition 2. Since DFS first enters any node via a downward step, and step i is the first occasion $u _ { i }$ is reached (a second descent to $u _ { i }$ would require a prior ascent from it, which would have been recorded as the unique D step terminating that visit), we have $i = i _ { u _ { i } } = \varphi _ { l } ( u _ { i } )$

Hence $\left. A _ { l } \right. = \left. B _ { l } \right.$ for every $1 \leq l \leq \mathrm { h t } ( G _ { T } )$ . It remains to verify that excluding the level $l = 0$ does not alter the maximum. The root is the unique node at depth 0, so $| A _ { 0 } | = 1$ . If $\mathrm { w d } ( G _ { T } ) = \mathrm { i }$ , then $| A _ { l } | = 1$ for all l, and since h $_ { \star } ( G _ { T } ) \geq 1$ for $n \geq 2$ the set $\{ A _ { l } : 1 \le l \le \mathrm { { h t } } ( G _ { T } ) \}$ is non-empty with maximum $1 = \mathrm { w d } ( G _ { T } )$ . If wd $. ( G _ { T } ) \ge 2$ the maximum is attained at some $l \geq 1$ and $| A _ { 0 } | \overset { \cdot } { = } 1$ is strictly dominated. In both cases,

$$
\mathrm { w d } ( G _ { T } ) = \operatorname* { m a x } _ { 0 \leq l \leq \mathrm { h t } ( G _ { T } ) } | A _ { l } | = \operatorname* { m a x } _ { 1 \leq l \leq \mathrm { h t } ( G _ { T } ) } | A _ { l } | = \operatorname* { m a x } _ { 0 < l \leq \mathrm { h t } ( G _ { T } ) } | B _ { l } | .
$$

ProofofCorrectness ofAlgorithm 1. We first address two implicit invariants. First, the branch on line 8 is unreachable when $i = 0 \colon$ if the current character were D at depth zero, the lattice path would descend below zero, contradicting the non-negativity condition of a Dyck path. Second, whenever the algorithm reaches line $^ { 8 , }$ it is guaranteed that a unique $j < n l$ satisfying $A _ { w } [ j , n l ] \stackrel { \cdot } { = } 1$ exists. Indeed, a step D at position t implies $s _ { t } - s _ { t - 1 } = - 1$ , so the path is closing a previously opened step U; the corresponding upward transition was handled by line 6 at the time that U was processed, which set $\dot { \boldsymbol { A } } _ { w } [ i , n l ] \dot {  } \boldsymbol { 1 }$ for the unique parent i of nl at that moment. The uniqueness of such $j$ follows from the observation that node nl is introduced exactly once, at the single U step that executes line 6 for nl, so no index $\boldsymbol { j ^ { \prime } } \neq \boldsymbol { j }$ can satisfy $A _ { w } [ j ^ { \prime } , n l ] = 1$ at the time line 8, since every previous step U operates on a strictly lower value of nl. Finally, ψ is injective: if $\psi ( w ) = \psi ( w ^ { \prime } )$ for two Dyck words w, $w ^ { \prime } \in \mathcal { D } _ { m - 1 }$ , then applying $\phi \mathrm { t o }$ both sides and invoking the Lemma 1 yields $\dot { w } \dot { = } \phi ( \dot { \psi } ( w ) \dot { ) } = \phi ( \psi ( w ^ { \dot { \prime } } ) ) = w ^ { \prime }$ . Hence, the algorithm terminates correctly on every valid Dyck word, the output $A _ { w }$ is the adjacency matrix of $\psi ( w )$ , and ψ is a well-defined injection in $\mathcal { D } _ { m - 1 }$ □

## C Preliminaries

Fact 1. Let $v \in \mathbb { R } ^ { n }$ be an input vector, $W \in \mathbb { R } ^ { m \times n }$ be a weight matrix, and $c \in \mathbb { R } ^ { m }$ be a bias vector. An affine transformation defined by $v ^ { \prime } = W v + c$ can be rewritten strictly as a linear transformation $\boldsymbol { v ^ { \prime } } = \boldsymbol { W ^ { \prime } } \boldsymbol { \tilde { v } }$ by concatenating the input vector v with c.

Proof. If we explicitly want to define our new input vector by concatenating v and c, we can write $\tilde { \boldsymbol { v } } = [ \boldsymbol { v } ^ { \top }  \boldsymbol { c } ^ { \top } ] ^ { \top } \in$ $\mathbb { R } ^ { n + m }$

Define $W ^ { \prime } \in \mathbb { R } ^ { m \times ( n + m ) }$ as $W ^ { \prime } = \left[ W \parallel I _ { m } \right]$ . Then

$$
\begin{array} { c } { { W ^ { \prime } ( \tilde { v } ) = [ W  I _ { m }  [ \begin{array} { l } { { v } } \\ { { c } } \end{array} ] = W v + I _ { m } c } } \\ { { = W v + c = v ^ { \prime } . } } \end{array}
$$

Fact 2. Let X and U be vector spaces over a field R. A linear transformation $L : X \to U$ parameterized by a matrix $W \in \mathbb { R } ^ { m \times n }$ such that $L ( v ) = W$ v can be represented as a bilinear operation.

Proof. To represent the linear transformation as a bilinear-style operation where the matrix weights are treated as parameters and v is the input, we consider the second argument as R.

We define a mapping $B : X \times \mathbb { R } \to U ,$ as $B ( v , c ) = W v \cdot c ,$ where $v \in V , c \in \mathbb { R }$ . To show this behaves linearly with respect to the input vector space $X$ , let $v _ { 1 } , v _ { 2 } \in X$ be input vectors, and let $\alpha , \beta \in \mathbb { R }$ be scalars.

$$
\begin{array} { r l } & { B ( \alpha v _ { 1 } + \beta v _ { 2 } , c ) = W ( \alpha v _ { 1 } + \beta v _ { 2 } ) \cdot c } \\ & { \qquad = ( \alpha W v _ { 1 } + \beta W v _ { 2 } ) \cdot c } \\ & { \qquad = \alpha ( W v _ { 1 } \cdot c ) + \beta ( W v _ { 2 } \cdot c ) } \\ & { \qquad = \alpha B ( v _ { 1 } , c ) + \beta B ( v _ { 2 } , c ) . } \end{array}
$$

This satisfies linearity in the first argument. Similarly for the second coordinate

$$
\begin{array} { r l } & { B ( v , \alpha c _ { 1 } + \beta c _ { 2 } ) = W v ( \alpha c _ { 1 } + \beta c _ { 2 } ) } \\ & { \qquad = W v \alpha c _ { 1 } + W v \beta c _ { 2 } } \\ & { \qquad = \alpha W v \cdot c _ { 1 } + \beta W v \cdot c _ { 2 } } \\ & { \qquad = \alpha B ( v , c _ { 1 } ) + \beta B ( v , c _ { 2 } ) . } \end{array}
$$

Lemma 4. Let U be a vector space and let $U ^ { \prime } , U ^ { \prime \prime }$ be vector spaces with $W = U ^ { \prime } \oplus U ^ { \prime \prime }$ . Let $B _ { 1 } : U \times U \to U ^ { \prime }$ and $B _ { 2 } : U \times U \to U ^ { \prime \prime }$ be bilinear maps, and define $B : U \times U \stackrel { \textstyle \cdot } { \to } W$ by

$$
B ( v , w ) = { \binom { B _ { 1 } ( v , w ) } { B _ { 2 } ( v , w ) } } .
$$

Then B is bilinear.

Proof. We verify linearity in the first argument; the second is identical. Let $v , \tilde { v } , w \in U$ and let c be a scalar. By the bilinearity of $B _ { 1 }$ and $B _ { 2 }$ in their first argument,

$$
B _ { i } ( v + \tilde { v } , w ) = B _ { i } ( v , w ) + B _ { i } ( \tilde { v } , w ) , \qquad i \in \{ 1 , 2 \} .
$$

Since addition in $W = U ^ { \prime } \oplus U ^ { \prime \prime }$ is taken componentwise,

$$
\begin{array} { r l } & { \boldsymbol { B } ( v + \tilde { v } , w ) = \binom { \boldsymbol { B } _ { 1 } ( v , w ) + \boldsymbol { B } _ { 1 } ( \tilde { v } , w ) } { \boldsymbol { B } _ { 2 } ( v , w ) + \boldsymbol { B } _ { 2 } ( \tilde { v } , w ) } } \\ & { \qquad = \binom { \boldsymbol { B } _ { 1 } ( v , w ) } { \boldsymbol { B } _ { 2 } ( v , w ) } + \binom { \boldsymbol { B } _ { 1 } ( \tilde { v } , w ) } { \boldsymbol { B } _ { 2 } ( \tilde { v } , w ) } } \\ & { \qquad = \boldsymbol { B } ( v , w ) + \boldsymbol { B } ( \tilde { v } , w ) . } \end{array}
$$

By the homogeneity of $B _ { 1 }$ and $B _ { 2 }$ in their first argument, $B _ { i } ( c v , w ) = c B _ { i } ( v , w )$ , and since scalar multiplication in W is componentwise,

$$
\begin{array} { c } { { B ( c v , w ) = { \binom { c B _ { 1 } ( v , w ) } { c B _ { 2 } ( v , w ) } } = c \left( { B _ { 1 } ( v , w ) } \right) } } \\ { { = c B ( v , w ) . } } \end{array}
$$

The second argument is treated identically, using the bilinearity of each $B _ { i }$ there. Hence B is bilinear.

Before we present the explicit constructions we substantiate the Definition 3.1.

Realization of Algorithms via Chain-of-Thought. The construction of Transformers that implement functions and algorithmic dynamics has been extensively studied in the expressivity literature (Giannou et al., 2023; Zhao et al., 2023; Dutta et al., 2025). To ground the constructions presented in section 2, Definition 3.1 outline the general scheme we adopt. The input sequence $( x _ { 0 } , \ldots , x _ { m - 1 } ) \in ( \mathbb { R } ^ { \bar { d } } ) ^ { m }$ encodes the graph structure, where each $x _ { i }$ may additionally carry a fixed number of auxiliary coordinates reserved for intermediate computation. The initial token $y _ { 0 } \in \mathbb { R } ^ { d }$ serves as the designated start symbol for each algorithm; for instance, in graph traversal, a fixed block of coordinates within $y _ { 0 }$ encodes the identity of the source vertex. However, during the design of individual algorithms, including those that take the lattice path as input, we will specify the encoding required to realize particular algorithms via CoT.

We call the realization uniform if the map from n to the family of resultant decoders is logspace-computable. In our work, all constructions follow uniformity in this sense. Note that the encoding enc represents an interpretation of the state of the algorithm A. While enc is not spelled out in each construction, its role is implicit and can be read from context. For instance, in the Dijkstra construction the block $\pi _ { \mathrm { D i s t } }$ carries the distance Dist, which serves as $\operatorname { e n c } ( \mathsf { s } _ { t } ) ;$ by contrast, in the DFS construction only the order of traversal matters, and no such $\mathrm { e n c } ( \mathsf { s } _ { t } )$ is present. The remaining cases follow the same pattern.

## D DFS

Theorem 5. A two-layer, two-head Transformer decoder can simulate the depth-first traversal DFS on any simple directed graph $G = ( { \dot { V } } , E , A )$ in $O ( | V | + | E | )$ chain-of-thought steps.

Proof. We show that the $t ^ { \mathrm { t h } }$ step of a two-layer, two-head Transformer decoder implements dfs $^ { ( t ) } ( s )$ . Let $G = ( V , E , A )$ be a graph with adjacency matrix $A \in \{ 0 , { \dot { 1 } } \} ^ { n \times n }$ , and denote by $e _ { i } \in \{ 0 , 1 \} ^ { n }$ the standard basis vector corresponding to vertex $v _ { i } \in V$

The initial token sequence consists of embedded vectors $x _ { 0 } , \ldots , x _ { n - 1 } \in \mathbb { R } ^ { d }$ with $d = 6 n + 3$ , preceded by a sentinel token $x _ { \perp } = 0 ^ { d }$ . Each embedding is a concatenation of nine blocks, that together capture the instantaneous description of the traversal: the current vertex, the parent vertex, the visited marker, the neighborhood, a temporary register, a buffer register, two state-control flags, and a type flag. We define the node embedding $x _ { i } \in \mathbb { R } ^ { 6 n + 3 }$ for each $v _ { i } \in V$ as

$$
x _ { i } = [ e _ { i } \parallel 0 ^ { n } \parallel 0 ^ { n } \parallel A _ { i , : } \parallel 0 ^ { n } \parallel 0 ^ { n } \parallel - 1 \parallel - 1 \parallel 0 ] ^ { \top } ,
$$

where the block $A _ { i , : }$ is the $i ^ { \mathrm { { t h } } }$ row of the adjacency matrix, encoding the neighborhood $\mathcal { N } ( v _ { i } )$ . Note that the collection $\{ x _ { i } \} _ { i = 0 } ^ { n - 1 }$ uniquely identifies G. The embedding carries three scalar flags. A state-control flag $f l g _ { 1 } \in \{ 1 , 0 , - 1 \}$ indicates whether the current step is a traverse, backtrack, or idle operation, respectively. An auxiliary flag $f l g _ { 2 }$ supports the final computation of $f l g _ { 1 } . \mathrm { \bf A }$ type flag $f l g _ { 3 } \in \{ 0 , 1 \}$ distinguishes x<sub>i</sub>s $( f l g _ { 3 } = 0 )$ from $y _ { t } \mathbf { s } \left( f l g _ { 3 } = 1 \right)$ The dynamic DFS at step t is maintained in vector $\bar { y _ { t } } \in \mathbb { R } ^ { \bar { 6 } n + 3 }$ , where $t \leq O ( | V | + \bar { | E | } )$ . Given a source vertex $s = v _ { p } \in V$ , the initial state $y _ { 0 }$ is

$$
y _ { 0 } = \left[ e _ { p } \parallel 0 ^ { n } \parallel e _ { p } \parallel A _ { p , : } \parallel 0 ^ { n } \parallel 0 ^ { n } \parallel - 1 \parallel - 1 \parallel 1 \right] ^ { \top } .
$$

The two embeddings differ in precisely two blocks: $y _ { 0 }$ records $e _ { p }$ in the visited restriction and sets $f l g _ { 3 } = 1$ , reflecting that s has been visited and that $y _ { 0 }$ is a traversal token.

Let ${ \mathcal { E } } \subset \{ 0 , 1 \} ^ { n }$ denote the set of standard basis vectors in $\mathbb { R } ^ { n }$ . For notational convenience, we define the following selectors from $\mathbb { R } ^ { d }$ to access individual blocks of an embedding. The maps $\pi _ { c u r } , \pi _ { p a r } : \mathbb { R } ^ { d }  \mathcal { E }$ extract the current and parent node indicators<sup>5</sup>; the maps $\pi _ { v i s } , \pi _ { n b r } : \mathbb { R } ^ { d } \to \{ 0 , 1 \} ^ { n }$ extract the visited and neighborhood vectors; the maps $\pi _ { t m p } , \pi _ { b u f } : \mathbb { R } ^ { d } \to \mathbb { R } ^ { n }$ extract the temporary and buffer registers; and $\pi _ { f l g _ { k } } : \mathbb { R } ^ { d }  \mathbb { R }$ for $k \in \{ 1 , 2 , 3 \}$ extracts the corresponding control flag.

We proceed inductively. Suppose $y _ { t }$ encodes the instantaneous description of G immediately before ${ \mathsf { D F S } } ^ { ( t + 1 ) } ( s )$ The first operation updates the control flags $\pi _ { f l g _ { 1 } } ( y _ { t } )$ and $\pi _ { f l g _ { 2 } } ( y _ { t } ) ;$ ; the attention in this layer is trivial, with $W _ { \mathcal { O } } ^ { ( 1 ) } = \mathbf { 0 } ^ { d \times d }$ . The update is carried out by the FFN block. The bilinear map $W _ { 1 } ^ { ( 1 ) }$ computes the inner product $\gamma _ { t } = ( { \bf 1 } - \pi _ { v i s } ( y _ { t } ) ) ^ { \top } \pi _ { n b r } ( y _ { t } )$ , which counts the number of unvisited neighbors of the current vertex, together with its unit-shifted counterpart $\gamma _ { t } - 1$ , placing these into the $f l g _ { 1 }$ and $f l g _ { 2 }$ positions respectively. After the ReLU activation, the subsequent linear map $\bar { W } _ { 2 } ^ { ( 1 ) }$ writes $\pi _ { f l g _ { 1 } } ( y _ { t } ) = \mathrm { R e L U } ( \gamma _ { t } ) - \mathrm { R e L U } ( \gamma _ { t } - 1 ) , \pi _ { f l g _ { 2 } } ( y _ { t } ) = 0$ , and copies $\pi _ { c u r } ( y _ { t } )$ into the buffer register. We denote the resulting vector by $y _ { t } ^ { ( 1 ) }$ . Observe that $\pi _ { f l g _ { 1 } } ( y _ { t } ^ { ( 1 ) } )$ is 1 precisely when $\gamma _ { t } \geq 1$ and 0 otherwise, encoding the traverse-or-backtrack decision. After this step, all preceding vectors $\{ x _ { \perp } ^ { ( 1 ) } , x _ { 0 } ^ { ( 1 ) } , \ldots , x _ { n - 1 } ^ { ( 1 ) } , y _ { 0 } ^ { ( 1 ) } , \ldots , y _ { t - 1 } ^ { ( 1 ) } \}$ differ from their initial values in the $f l g _ { 1 } , f l g _ { 2 }$ , and buffer positions. Because none of these positions of the aforementioned vector interact with any operation in the subsequent layer, this modification does not affect the computations.

The second layer comprises two mutually exclusive attention heads. The first head executes the operation traverse. The query $Q _ { T } ^ { ( 2 ) } y _ { t } ^ { ( \bar { 1 } ) }$ is a bilinear map (Rizvi-Martel et al., 2024) implementing the Hadamard product $( \mathbf { 1 } - \pi _ { v i s } ( y _ { t } ^ { ( 1 ) } ) ) \odot$ $\pi _ { n b r } ( y _ { t } ^ { ( 1 ) } )$ , which masks out all previously visited vertices from the neighborhood. Evaluated against keys $K _ { T } ^ { ( 2 ) } \alpha =$ $\pi _ { c u r } ( \alpha )$ for all $\alpha \in \{ x _ { \bot } ^ { ( 1 ) } , x _ { 0 } ^ { ( 1 ) } , \dotsc , x _ { n - 1 } ^ { ( 1 ) } , y _ { 0 } ^ { ( 1 ) } , \dotsc , y _ { t } ^ { ( 1 ) } \}$ , the UHA resolves the lowest-indexed token $\alpha _ { T } ^ { \prime }$ (among $\{ x _ { i } ^ { ( 1 ) } \} )$ whose current-node encoding $\pi _ { c u r } ( \alpha _ { T } ^ { \prime } )$ corresponds to ${ \mathsf { d f s } } ^ { ( t + 1 ) } ( s )$ . When $\pi _ { f l g _ { 1 } } ( y _ { t } ^ { ( 1 ) } ) = 0 , \alpha _ { T } ^ { \prime }$ defaults to $x _ { \perp } ^ { ( \mathrm { i } ) }$

The second head implements the operation backtrack. The query $Q _ { B } ^ { ( 2 ) } y _ { t } ^ { ( 1 ) }$ is a bilinear map computing $( 1 - \pi _ { f l g _ { 1 } } ( y _ { t } ^ { ( 1 ) } ) )$ $[ \pi _ { p a r } ( y _ { t } ^ { ( 1 ) } ) \parallel \pi _ { f l g _ { 3 } } ( y _ { t } ^ { ( 1 ) } ) ]$ ], which activates only when $\pi _ { f l g _ { 1 } } ( y _ { t } ^ { ( 1 ) } ) = 0$ . Evaluated against keys $K _ { B } ^ { ( 2 ) } \alpha = [ \pi _ { c u r } ( \alpha )$ ∥ $\pi _ { f l g _ { 3 } } ( \alpha ) ]$ for all $\alpha \in \{ x _ { \bot } ^ { ( 1 ) } , x _ { 0 } ^ { ( 1 ) } , \ldots , x _ { n - 1 } ^ { ( 1 ) } , y _ { 0 } ^ { ( \bar { 1 } ) } , \ldots , y _ { t } ^ { ( \bar { 1 } ) } \}$ , the UHA resolves the lowest-indexed token $\alpha _ { B } ^ { \prime }$ (among $\{ y _ { i } ^ { ( 1 ) } \} )$ such that $\pi _ { c u r } ( \alpha _ { B } ^ { \prime } )$ corresponds to the vertex first visited during dfs $^ { ( t ^ { \prime } ) } ( s )$ for some $0 \leq t ^ { \prime } < t$ . When $\pi _ { f l g _ { 1 } } ( y _ { t } ^ { ( 1 ) } ) = 1$ , the query vanishes and $\alpha _ { B } ^ { \prime }$ defaults to $x _ { \perp } ^ { ( 1 ) }$ . The linear map $W _ { O } ^ { ( 2 ) } \in \{ - 1 , 0 , 1 \} ^ { d \times 2 d }$ produces the combined update $\alpha ^ { \prime \prime } \overset { \cdot } { = } \alpha _ { T } ^ { \prime \prime } + \alpha _ { B } ^ { \prime \prime }$ , where

$$
\alpha _ { T } ^ { \prime \prime } = \left[ - \pi _ { c u r } ( \alpha _ { T } ^ { \prime } ) \parallel - \pi _ { p a r } ( \alpha _ { T } ^ { \prime } ) \parallel \pi _ { c u r } ( \alpha _ { T } ^ { \prime } ) \parallel 0 ^ { n } \parallel \pi _ { n b r } ( \alpha _ { T } ^ { \prime } ) \parallel 0 ^ { n } \parallel 0 ^ { 3 } \right] ^ { \top }
$$

$$
\alpha _ { B } ^ { \prime \prime } = \left[ - \pi _ { c u r } ( \alpha _ { B } ^ { \prime } ) \parallel - \pi _ { p a r } ( \alpha _ { B } ^ { \prime } ) \parallel 0 ^ { 2 n } \parallel \pi _ { n b r } ( \alpha _ { B } ^ { \prime } ) \parallel 0 ^ { n } \parallel 0 ^ { 3 } \right] ^ { \top } .
$$

Since the two heads are mutually exclusive – governed by $\pi _ { f l g _ { 1 } } ( y _ { t } ^ { ( 1 ) } )$ ) – exactly one of $\alpha _ { T } ^ { \prime \prime }$ and $\alpha _ { B } ^ { \prime \prime }$ is $0 ^ { 6 n + 3 }$

Since G is simple, during traverse the current vertex and any unvisited neighbor are necessarily distinct, whence $\pi _ { c u r } ( y _ { t } ^ { ( 1 ) } ) ^ { \top } \pi _ { c u r } ( \alpha _ { T } ^ { \prime \prime } ) = 0$ . Similarly, during backtrack, $\pi _ { p a r } ( y _ { t } ^ { ( 1 ) } ) ^ { \top } \pi _ { p a r } ( \alpha _ { B } ^ { \prime \prime } ) = 0$ holds strictly: were it otherwise, we would have $\pi _ { c u r } ( \alpha _ { B } ^ { \prime \prime } ) = \pi _ { p a r } ( \alpha _ { B } ^ { \prime \prime } )$ , implying a self-loop and contradicting the assumption that G is simple.

The linear map $W _ { 1 } ^ { ( 2 ) }$ then negates the current and parent restrictions of $\alpha ^ { \prime \prime } + y _ { t } ^ { ( 1 ) }$ , resets the temporary register, and copies the (previous) temporary register into the neighborhood position, leaving all other positions intact. The subsequent ReLU activation discards the negated priors: $- \pi _ { c u r } ( y _ { t } ^ { ( 1 ) } )$ is eliminated from the current restriction in both traversal and backtrack, while $- \pi _ { p a r } ( y _ { t } ^ { ( 1 ) } )$ is eliminated from the parent restriction during backtrack. Denote the result by $\beta .$ . At this stage, $\beta$ agrees with the desired output $y _ { t + 1 }$ of the $( t + 1 ) \cdot \mathrm { t h }$ DFS dynamic in every restriction except the parent restriction during traverse. To resolve this, the bilinear map $W _ { 2 } ^ { ( 2 ) }$ sets $\pi _ { p a r } ( \beta )$ to value $\pi _ { f l g _ { 1 } } ( \beta ) \pi _ { b u f } ( \bar { \beta } ) + ( \bar { 1 } - \pi _ { f l g _ { 1 } } ( \beta ) ) \pi _ { p a r } ( \beta )$ , which selects the buffered vertex (the pre-traversal current node) when $\pi _ { f l g _ { 1 } } ( y _ { t } ) = 1$ and retains the existing parent otherwise. The chain-of-thought terminates when $\pi _ { c u r } ( y _ { t + 1 } ) = 0 ^ { n }$ .□

## E Dijkstra Algorithm

Theorem 6. A two-layer, single-head Transformer decoder can simulate the Dijkstra algorithm Dkst on any simple connected directed graph $G \overset { \vartriangle } { = } ( V , E , A )$ with $\mathsf { A } = \mathbb { R } _ { > 0 }$ in exactly $| V | - 1$ chain-of-thought steps.

Proof. We show that the $t ^ { \mathrm { { t h } } }$ step of a two-layer single-head Transformer decoder implements $\mathsf { D } \mathsf { k } \mathsf { s t } ^ { ( t ) }$ . Let $G = ( V , E , A )$ be a graph with adjacency matrix $A \in \mathbb { R } _ { > 0 } ^ { \bar { n } \times n }$ , fix a source $s = v _ { p } \in V$ , and write $e _ { i } \in \{ 0 , 1 \} ^ { n }$ for the standard basis vector corresponding to $v _ { i }$ . The input token sequence consists of embeddings $x _ { 0 } , \ldots , x _ { n - 1 } \in \mathbb { R } ^ { d }$ with $d = 5 n + 1$ that together encode the instantaneous description of the traversal: the identity of the current vertex, the tentative distance from s to that vertex, the visited indicator, the outgoing edge weights, the global distance vector Dist, and a buffer. Concretely, the embedding of vertex $v _ { i }$ is

$$
\begin{array} { r } { \boldsymbol { x } _ { i } = \left[ \boldsymbol { e } _ { i } \parallel - \lambda \parallel 0 ^ { n } \parallel A _ { i , : } \parallel \lambda ^ { n } \parallel 0 ^ { n } \right] ^ { \top } , } \end{array}
$$

where $A _ { i , : }$ is the $i ^ { \mathrm { { t h } } }$ row of the adjacency matrix encoding the outgoing weights from $v _ { i } .$ , and λ is a sufficiently large constant. The scalar −λ in the second block serves as a sentinel indicating that no finite shortest-path estimate has yet been assigned; the fourth block will be repurposed during output token generation, and the final block acts as a

scratchpad. Note that the collection $\{ x _ { i } \} _ { i = 0 } ^ { n - 1 }$ uniquely determines G. The dynamic Dkst at step $t \in \{ 0 , \ldots , n - 1 \}$ is maintained in a vector $y _ { t } \in \mathbb { R } ^ { d } ;$ given source $s = v _ { p }$ , the initial state is

$$
y _ { 0 } = \left[ e _ { p } \parallel 0 \parallel e _ { p } \parallel 0 ^ { n } \parallel \lambda ^ { n } \parallel 0 ^ { n } \right] ^ { \top } .
$$

That is, $y _ { 0 }$ starts the CoT with the initialized state $\mathsf { D } \mathsf { k } \mathsf { s t } ^ { ( 0 ) }$ – source visited, distance zero – whereas the embeddings $\{ x _ { i } \}$ encode the graph structure. Let $\mathcal { E } \subset \{ 0 , 1 \} ^ { n }$ denote the set of standard basis vectors in $\mathbb { R } ^ { n }$ . For notational convenience, we define the following selectors that extract individual blocks from any embedding in $\mathbb { R } ^ { d } \colon \pi _ { c u r } : \mathbb { R } ^ { d } \to \mathcal { E } , \pi _ { c r d }$ $\mathbb { R } ^ { d } \to \mathbb { R } _ { > 0 } \cup \{ - \lambda \} , \bar { \pi _ { v i s } } : \mathbb { R } ^ { d } \to \{ 0 , 1 \} ^ { n } , \pi _ { w e i } , \pi _ { d i s } : \mathbb { R } ^ { d } \to \mathbb { R } ^ { n }$ , and $\dot { \pi _ { b u f } } : \mathbb { R } ^ { d }  \mathbf { \bar { \mathbb { R } } } ^ { n }$ . In the generated sequence $\{ y _ { t + 1 } \}$ , the scalar $\pi _ { c r d } ( y _ { t + 1 } )$ records $\mathsf { D i s t } ^ { ( t + 1 ) } [ j ^ { \prime } ]$ for the uniquely newly visited vertex, i.e., the vertex $v _ { j ^ { \prime } }$ satisfying vs $t ^ { ( t + 1 ) } ( v _ { j ^ { \prime } } ) = 1$ and $v s t ^ { ( t ) } ( v _ { j ^ { \prime } } ) = 0$ . The block $\pi _ { w e i } ( y _ { t + 1 } ) = 0 ^ { n }$ is not used to carry adjacency information in the output tokens and is instead repurposed as another buffer for storing intermediate results between layers. Note that the codomain $\mathbb { R } _ { \geq 0 } \cup \{ - \lambda \}$ reflects a dual role: on input tokens $\{ x _ { i } \}$ the selector $\pi _ { c r d }$ returns the sentinel $- \lambda$ , while on generated tokens $\{ y _ { t } \}$ it returns a non-negative shortest-path distance.

We proceed inductively. Suppose $y _ { t }$ encodes the instantaneous description of G immediately prior to $\mathsf { D } \mathsf { k } \mathsf { s t } ^ { ( t + 1 ) }$ ; the first layer updates the distance vector from $\mathsf { D i s t } ^ { ( t ) }$ to $\mathsf { D i s t } ^ { ( t + 1 ) }$ . The query $Q ^ { ( 1 ) } y _ { t }$ extracts $\pi _ { c u r } ( y _ { t } )$ , the one-hot identity of the current vertex, and is evaluated against keys $K ^ { ( 1 ) } \alpha = \pi _ { c u r } ( \alpha )$ for every token $\alpha \in \{ x _ { 0 } , \ldots , x _ { n - 1 } , y _ { 0 } , \ldots , y _ { t } \} ;$ the UHA selects the lowest-indexed match, which is necessarily the input embedding $x _ { i }$ corresponding to $c u r ^ { ( t ) } = v _ { i }$ thereby getting access of the adjacency row $A _ { i , : }$ . Taking $V ^ { ( 1 ) }$ as identity, the output projection $W _ { O } ^ { ( \bar { 1 } ) }$ zeroes out all blocks except $\pi _ { w e i } ,$ , so that after the residual connection the weight block carries $\bar { A _ { i , : } } - \bar { \mathrm { i n } }$ particular, this is where $\pi _ { w e i }$ transitions from its role as a zero buffer on output tokens to carrying adjacency data within the layer. Let this vector be $\alpha ^ { \prime \prime }$ The first linear map of the feed-forward network, $W _ { 1 } ^ { ( 1 ) }$ , then computes $\pi _ { d i s } ( \alpha ^ { \prime \prime } + y _ { t } ) [ i ] - \pi _ { c r d } ( \alpha ^ { \prime \prime } + y _ { t } ) - \pi _ { w e i } ( \alpha ^ { \prime \prime } + y _ { t } ) [ i ]$ for each $i \in \{ 0 , \ldots , n - 1 \}$ ; after the ReLU activation, $\pi _ { w e i }$ stores the non-negative updates corresponding to $\mathsf { D i s t } ^ { ( t + 1 ) }$ with all other positions intact. Denote the result by $\beta$ and observe that $\pi _ { w e i } ( \beta ) \geq 0 ^ { n }$ . The second linear map $W _ { 2 } ^ { ( 1 ) }$ replaces $\pi _ { d i s } ( \beta )$ with $\pi _ { d i s } ( \beta ) - \pi _ { w e i } ( \beta )$ and resets $\pi _ { w e i } ( \beta )$ to $0 ^ { n }$ . We denote the resulting vector by $y _ { t } ^ { ( 1 ) }$

The second layer selects the unvisited vertex of minimum tentative distance. The query $Q ^ { ( 2 ) }$ is a bilinear map (Rizvi-Martel et al., 2024) realizing the Hadamard product $( 1 - \pi _ { v i s } ( y _ { t } ^ { ( 1 ) } ) ) \odot ( \lambda - \pi _ { d i s } ( y _ { t } ^ { ( 1 ) } ) )$ , which assigns each unvisited vertex $v _ { j }$ a score of $\lambda - \pi _ { d i s } ( y _ { t } ^ { ( 1 ) } ) [ j ] > 0$ . Evaluated against keys $K ^ { ( 2 ) } \alpha = \pi _ { c u r } ( \alpha )$ for all $\alpha \in$ $\{ x _ { 0 } ^ { ( 1 ) } , \ldots , x _ { n - 1 } ^ { ( 1 ) } , y _ { 0 } ^ { ( 1 ) } , \ldots , y _ { t } ^ { ( 1 ) } \}$ , UHA selects the lowest-indexed token $\alpha ^ { \prime }$ among $\{ x _ { i } ^ { ( 1 ) } \} _ { i = 0 } ^ { n - 1 }$ at which $- \pi _ { d i s } ( y _ { t } ^ { ( 1 ) } )$ is maximized over unvisited vertices – equivalently, the vertex $v _ { j ^ { \prime } }$ such that j<sup>′</sup> = argmin $\{ \mathsf { D i s t } ^ { ( t + 1 ) } [ j ] \mid v s t ^ { ( t ) } ( v _ { j } ) = 0 \}$ The output projection $W _ { O } ^ { ( 2 ) }$ produces

$$
\alpha ^ { \prime \prime } = [ - \pi _ { c u r } ( \alpha ^ { \prime } ) \parallel 0 \parallel \pi _ { c u r } ( \alpha ^ { \prime } ) \parallel 0 ^ { n } \parallel 0 ^ { n } \parallel \pi _ { c u r } ( \alpha ^ { \prime } ) ] ^ { \top } ,
$$

so that the residual connection yields $\pi _ { c u r } ( \alpha ^ { \prime \prime } + y _ { t } ^ { ( 1 ) } ) = \pi _ { c u r } ( y _ { t } ^ { ( 1 ) } ) - \pi _ { c u r } ( \alpha ^ { \prime } )$ and $\pi _ { v i s } ( \alpha ^ { \prime \prime } + y _ { t } ^ { ( 1 ) } ) = \pi _ { v i s } ( y _ { t } ^ { ( 1 ) } ) +$ $\pi _ { c u r } ( \alpha ^ { \prime } )$ , while $\pi _ { b u f }$ now holds $\pi _ { c u r } ( \alpha ^ { \prime } )$ . We claim $\pi _ { c u r } ( \alpha ^ { \prime } ) ^ { \top } \pi _ { c u r } ( y _ { t } ^ { ( 1 ) } ) = 0 :$ : were this not $\mathbf { s o } ,$ the current vertex would be attending to itself, implying a self-loop and contradicting the assumption that G is simple. Hence, the residual updates the current block to the difference of two orthogonal one-hot vectors and augments the visited set by exactly one element. The feed-forward network of the second layer completes the transition. The linear map $W _ { 1 } ^ { ( 2 ) }$ acts on $\alpha ^ { \prime \prime } + y _ { t } ^ { ( 1 ) }$ at two blocks while leaving all others intact: it negates the current block and sets the buffer to $\gamma \in \mathbb { R } ^ { n }$ defined coordinate-wise by

$$
\gamma [ i ] = \pi _ { d i s } ( { \alpha ^ { \prime \prime } } + y _ { t } ^ { ( 1 ) } ) [ i ] + \lambda \pi _ { b u f } ( { \alpha ^ { \prime \prime } } + y _ { t } ^ { ( 1 ) } ) [ i ] - \lambda \sum _ { j = 0 } ^ { n - 1 } \pi _ { b u f } ( { \alpha ^ { \prime \prime } } + y _ { t } ^ { ( 1 ) } ) [ j ] ,
$$

for each $i \in \{ 0 , \ldots , n - 1 \}$ . Since

$$
\pi _ { b u f } ( \alpha ^ { \prime \prime } + y _ { t } ^ { ( 1 ) } ) = \pi _ { c u r } ( \alpha ^ { \prime } ) = e _ { j ^ { \prime } } ,
$$

the sum equals unity and the expression simplifies: at coordinate $j ^ { \prime }$ the λ terms cancel, yielding $\gamma [ j ^ { \prime } ] = \mathsf { D i s t } ^ { ( t + 1 ) } [ j ^ { \prime } ]$ while at every other coordinate $\gamma [ i ] = { \sf D i s t } ^ { ( t + 1 ) } [ i ] - \lambda < 0$ for λ sufficiently large. The subsequent ReLU therefore retains $\mathsf { D i s t } ^ { ( i + 1 ) } [ j ^ { \prime } ]$ at the $j ^ { \prime \mathrm { t h } }$ coordinate of the buffer and zeros out all others, while simultaneously eliminating $- \pi _ { c u r } ( y _ { t } ^ { ( 1 ) } )$ from the current block, leaving $\pi _ { c u r } ( \alpha ^ { \prime } )$ as the sole survivor. Denote the result by $y _ { t } ^ { ( 1 ) ^ { \prime } }$ . Finally, $W _ { 2 } ^ { ( 2 ) }$ sets $\begin{array} { r } { \pi _ { c r d } ( y _ { t + 1 } ) = \sum _ { i = 0 } ^ { n - 1 } { \pi _ { b u f } ( y _ { t } ^ { ( 1 ) } ) [ i ] } = \mathsf { D i s t } ^ { ( t + 1 ) } [ j ^ { \prime } ] } \end{array}$ and resets $\pi _ { b u f } ( y _ { t + 1 } ) = 0 ^ { n }$ , resulting $y _ { t + 1 }$ . After exactly $n - 1$ transitions the visited indicator satisfies $\pi _ { v i s } ( y _ { n - 1 } ) = 1 ^ { n }$ , which serves as the halting criterion. □

## F Realizing Algorithm 1

Theorem 7. A single-layer, two-head Transformer decoder can simulate the Algorithm 1 in the $2 ( m - 1 )$ CoT-steps.

Proof. The $t ^ { \mathrm { { t h } } }$ decoding step reconstructs the partial tree built after reading the lattice path $\{ ( \tau , s _ { \tau } ) \} _ { \tau = 0 } ^ { t } .$ , which is one iteration of the for loop on line 3. Let $\mathcal { E } _ { m } = \{ \bar { e } _ { 0 } , \dots , e _ { m - 1 } \}$ and $\mathcal { E } _ { 2 m - 2 } = \{ \hat { e } _ { 0 } , \dots , \hat { e } _ { 2 m - 3 } \}$ denote the standard bases of $\mathbb { R } ^ { m }$ and $\mathbb { R } ^ { 2 m - 2 }$ . The construction maintains $i , n l ,$ , and $A _ { w }$ across iterations, encoded in eight blocks of dimension $d = m ^ { 2 } +$ 6m: the current vertex cur (the iterator i of line 2), the label nl of the most recently created vertex, the parent par recorded when an U creates a vertex, the flattened adjacency matrix $A$ of the partial tree, the cursor pos marking the position read so far, two scratch blocks $s t p$ holding $q _ { \tau } \in \{ + 1 , - 1 \}$ } for U resp. D and on a CoT token $y _ { t }$ is $0 ,$ taking a transient value in $\{ + 1 , - 1 \}$ only while a step is being read; and buf holding the parent encoding retrieved on a D (so that line 9 is realized), and a flag $f l g$ separating input tokens x $( f l g = 0 )$ from CoT tokens $y \left( f l g = 1 \right)$ ).

The input tokens $x _ { 0 } , \ldots , x _ { 2 m - 3 } \in \mathbb { R } ^ { d }$ encode the Dyck word $w = w _ { 0 } \cdot \cdot \cdot w _ { 2 m - 3 }$ of the path $\{ ( \tau , s _ { \tau } ) \} _ { \tau = 0 } ^ { 2 m - 2 } $

$$
\boldsymbol { x } _ { \tau } = [ 0 ^ { m } \parallel 0 ^ { m } \parallel 0 ^ { m } \parallel 0 ^ { m ^ { 2 } } \parallel 0 ^ { m ^ { 2 } } \parallel \hat { e } _ { \tau } \parallel q _ { \tau } \parallel 0 ^ { m } \parallel 0 ] ^ { \top } ,
$$

$$
q _ { \tau } = s _ { \tau + 1 } - s _ { \tau } .
$$

The dynamic is carried out by tokens $y _ { t } \in \mathbb R ^ { d }$ , with

$$
\boldsymbol { y } _ { 0 } = [ e _ { 0 } \parallel e _ { 0 } \parallel 0 ^ { m } \parallel 0 ^ { m ^ { 2 } } \parallel \hat { e } _ { 0 } \parallel 0 \parallel 0 ^ { m } \parallel 1 ] ^ { \top } ,
$$

so that $i = n l = 0 ( c u r _ { ; }$ , nl at e<sub>0</sub>) and par, $A _ { w }$ are empty.

For $v \in \{ 0 , 1 \}$ and $q \in \{ + 1 , - 1 \}$ ,

$$
\mathrm { R e L U } ( v + q - 1 ) = \mathbb { 1 } [ q = + 1 ] v , \quad \mathrm { R e L U } ( v - q - 1 ) = \mathbb { 1 } [ q = - 1 ] v ,\tag{4}
$$

since ${ \mathrm { R e L U } } ( v + 1 - 1 ) = v$ while ${ \mathrm { R e L U } } ( v - 1 - 1 ) = { \mathrm { R e L U } } ( v - 2 ) = 0$ , and symmetrically. Hence, one ReLU selects a branch by the sign of $q _ { t }$ and passes a binary v unchanged on the selected branch, zeroing it on the other. Like other constructions, we use the selectors

$$
\begin{array} { r l } & { \pi _ { c u r } , \pi _ { n l } : \mathbb { R } ^ { d }  \mathcal { E } _ { m } , } \\ & { \pi _ { p a r } , \pi _ { b u f } : \mathbb { R } ^ { d }  \mathcal { E } _ { m } \cup \{ 0 ^ { m } \} , } \\ & { \pi _ { A } : \mathbb { R } ^ { d }  \{ 0 , 1 \} ^ { m ^ { 2 } } , } \\ & { \pi _ { p o s } : \mathbb { R } ^ { d }  \mathcal { E } _ { 2 m - 2 } , } \\ & { \pi _ { s t p } : \mathbb { R } ^ { d }  \{ + 1 , - 1 , 0 \} , } \\ & { \mathrm { a n d ~ } \pi _ { f l q } : \mathbb { R } ^ { d }  \{ 0 , 1 \} , } \end{array}
$$

each extracting its block from an embedding in $\mathbb { R } ^ { d }$

We argue by induction, with the hypothesis that $\pi _ { A } ( y _ { t } )$ holds the adjacency matrix of the partial tree built after reading $x _ { 0 } , \ldots , x _ { t - 1 }$ . Step t is driven by two heads, one per branch of line 4: the U head is selected when $\pi _ { s t p } ( x _ { t } ) = + 1$ , the D head when $\pi _ { s t p } ( x _ { t } ) = - 1$

The U head reads the current symbol. With query $Q _ { \tt U } y _ { t } ~ = ~ \pi _ { p o s } ( y _ { t } )$ and keys $K _ { \tt U } \alpha \ = \ \pi _ { p o s } ( \alpha )$ over $\alpha \in$ $\left\{ x _ { 0 } , \ldots , x _ { 2 m - 3 } , y _ { 0 } , \ldots , y _ { t } \right\}$ , the score $\pi _ { p o s } ( y _ { t } ) ^ { \top } \pi _ { p o s } ( \alpha )$ is 1 exactly on the tokens whose block pos is $\hat { e } _ { t }$ , namely $x _ { t }$ and $y _ { t } ;$ the lowest-indexed of these is the input token $x _ { t } = : \alpha _ { \mathbb { U } } ^ { \prime }$ . Its value $\alpha _ { \mathrm { U } } ^ { \prime \prime } : = V _ { \mathrm { U } } \alpha _ { \mathrm { U } } ^ { \prime }$ copies $\pi _ { s t p } ( \alpha _ { \mathrm { U } } ^ { \prime } ) = q _ { t }$ into the stp block and is 0 elsewhere.

The D head retrieves a parent. With query $Q _ { \mathsf { D } } y _ { t } = [ \pi _ { c u r } ( y _ { t } ) \parallel \pi _ { f l g } ( y _ { t } ) ]$ and keys $K _ { \tt D } \alpha = [ \pi _ { c u r } ( \alpha ) \parallel \pi _ { f l g } ( \alpha ) ]$ , the score is 2 on the $y$ tokens whose current block equals $\pi _ { c u r } ( y _ { t } )$ and at most 1 on every other token; among these the lowest-indexed is the creation token of the current vertex $i _ { t }$ (the $y$ token that first set $\pi _ { c u r } = \pi _ { c u r } ( y _ { t } )$ on a U step), which precedes $y _ { t }$ and is thus selected over it. Call it $\alpha _ { \mathrm { { D } } } ^ { \prime }$ . Its value $\alpha _ { \tt D } ^ { \prime \prime } : = V _ { \tt D } \alpha _ { \tt D } ^ { \prime }$ copies the recorded parent $\pi _ { p a r } ( \alpha _ { \tt D } ^ { \prime } )$ into the buf block and is 0 elsewhere.

The projection $W _ { O } \in \{ 0 , 1 \} ^ { d \times 2 d }$ sums the two contributions, $\alpha ^ { \prime \prime } = \alpha _ { \mathrm { U } } ^ { \prime \prime } + \alpha _ { \mathrm { D } } ^ { \prime \prime } .$

We now construct the FFN. Write $\beta = \alpha ^ { \prime \prime } + y _ { t }$ for its input; since the heads write only the $s t p$ and $b u f$ blocks, $\beta$ agrees with $y _ { t }$ on every other block, so $\pi _ { c u r } ( \beta ) = \pi _ { c u r } ( y _ { t } ) , \pi _ { n l } ( \beta ) = \pi _ { n l } ( y _ { t } ) , \pi _ { A } ( \beta ) = \pi _ { A } ( y _ { t } )$ , while $\pi _ { s t p } ( \beta ) = q _ { t }$ and $\pi _ { b u f } ( \beta ) = \pi _ { p a r } ( \alpha _ { \mathtt { D } } ^ { \prime } )$ ). Let $h = W _ { 1 } \beta \in \mathbb { R } ^ { d }$ with $d ^ { \prime } = 2 m ^ { 2 } + 6 m - 2 > d ,$ , and overload $\pi _ { . } ^ { \prime }$ for the blocks of h:

$$
h = [ \pi _ { c u r } ^ { \prime } \parallel \pi _ { n l } ^ { \prime } \parallel \pi _ { p a r } ^ { \prime } \parallel \pi _ { b u f } ^ { \prime } \parallel \pi _ { A } ^ { \prime } \parallel \pi _ { p o s } ^ { \prime } \parallel \pi _ { A ^ { \prime } } ^ { \prime } ] ,
$$

which drops the $f l g$ and stp blocks and adds $A ^ { \prime } ,$ , the indicator of the edge inserted on a U (line 6). Let $S \in \{ 0 , 1 \}$ m×m be the shift $s _ { i j } \stackrel { \cdot } { = } \mathbb { 1 } [ j = i - 1 ]$ (so $S e _ { k } = e _ { k + 1 }$ and $S e _ { m - 1 } = 0 )$ ; addition of a scalar to a vector is broadcast, i.e. added to every coordinate. By (4), W sets each block so that the activation ReLU realizes the branch of line 4:

$$
\begin{array} { r l } & { \pi _ { c u r } ^ { \prime } ( h ) = S \pi _ { n l } ( \beta ) + \pi _ { s t p } ( \beta ) - 1 } \\ & { \pi _ { n l } ^ { \prime } ( h ) = \pi _ { n l } ( \beta ) - \pi _ { s t p } ( \beta ) - 1 } \\ & { \pi _ { p a r } ^ { \prime } ( h ) = \pi _ { c u r } ( \beta ) + \pi _ { s t p } ( \beta ) - 1 } \\ & { \pi _ { b u f } ^ { \prime } ( h ) = \pi _ { b u f } ( \beta ) - \pi _ { s t p } ( \beta ) - 1 } \\ & { \pi _ { A } ^ { \prime } ( h ) = \pi _ { A } ( \beta ) } \\ & { \pi _ { A ^ { \prime } } ^ { \prime } ( h ) _ { a m + b } = \beta ^ { \top } M ^ { ( a , b ) } \beta + \pi _ { s t p } ( \beta ) - 1 } \\ & { \quad = \pi _ { c u r } ( \beta ) _ { a } \cdot \pi _ { n l } ( \beta ) _ { b - 1 } + \pi _ { s t p } ( \beta ) - 1 } \\ & { \pi _ { p o s } ^ { \prime } ( h ) = \pi _ { p o s } ( \beta ) , } \end{array}
$$

for $( a , b ) \in \{ 0 , \ldots , m - 1 \} ^ { 2 }$ , where $M ^ { ( a , b ) } \in \mathbb { R } ^ { d \times d }$ is the constant matrix with $\beta ^ { \top } M ^ { ( a , b ) } \beta = \pi _ { c u r } ( \beta ) _ { a } \cdot \pi _ { n l } ( \beta ) _ { b - 1 }$ and $\pi _ { n l } ( \beta ) _ { b - 1 }$ is read as 0 when $b = 0$

Recall $\pi _ { s t p } ( \beta ) = q _ { t } \in \{ + 1 , - 1 \}$ . By (4), after activation $\pi _ { c u r } ^ { \prime } ( h ^ { \prime } )$ equals the shifted label $S \pi _ { n l } ( \beta )$ on a U and 0 on a D, while $\pi _ { b u f } ^ { \prime } ( h ^ { \prime } )$ equals the retrieved parent on a D and 0 on a $\mathrm { U } ;$ the remaining blocks gate analogously. Let $S ^ { \prime } \in \{ 0 , 1 \}$ }<sup>(2m−2)×(2m−2)</sup> be the shift $S ^ { \prime } { \hat { e } } _ { \tau } ~ = ~ { \hat { e } } _ { \tau + 1 }$ with $S ^ { \prime } { \hat { e } } _ { 2 m - 3 } = 0$ . With $h ^ { \prime } \ = \ \mathrm { R e L U } ( h )$ , the map $W _ { 2 } \in \{ 0 , 1 \} ^ { d \times d ^ { \prime } }$ produces $y _ { t + 1 } = W _ { 2 } h ^ { \prime }$ by summing the gated blocks:

$$
\begin{array} { r l } & { \pi _ { c u r } ( y _ { t + 1 } ) = \pi _ { c u r } ^ { \prime } ( h ^ { \prime } ) + \pi _ { b u f } ^ { \prime } ( h ^ { \prime } ) , } \\ & { \pi _ { n l } ( y _ { t + 1 } ) = \pi _ { c u r } ^ { \prime } ( h ^ { \prime } ) + \pi _ { n l } ^ { \prime } ( h ^ { \prime } ) , } \\ & { \pi _ { p a r } ( y _ { t + 1 } ) = \pi _ { p a r } ^ { \prime } ( h ^ { \prime } ) , } \\ & { \pi _ { A } ( y _ { t + 1 } ) [ a m + b ] = \pi _ { A } ^ { \prime } ( h ^ { \prime } ) [ a m + b ] } \\ & { \quad \quad + \pi _ { A ^ { \prime } } ^ { \prime } ( h ^ { \prime } ) [ a m + b ] + \pi _ { A ^ { \prime } } ^ { \prime } ( h ^ { \prime } ) [ b m + a ] , } \\ & { \pi _ { p a s } ( y _ { t + 1 } ) = S ^ { \prime } \pi _ { p o s } ^ { \prime } ( h ^ { \prime } ) , \quad \pi _ { s t p } ( y _ { t + 1 } ) = 0 , } \\ & { \pi _ { b u f } ( y _ { t + 1 } ) = 0 ^ { m } , \quad \pi _ { f l } ( y _ { t + 1 } ) = 1 . } \end{array}
$$

On a U step this gives $\pi _ { c u r } ( y _ { t + 1 } ) = \pi _ { n l } ( y _ { t + 1 } ) = e _ { n l _ { t } + 1 }$ and sets $A [ i _ { t } , n l _ { t } + 1 ] = A [ n l _ { t } + 1 , i _ { t } ] = 1$ with $\pi _ { p a r } ( y _ { t + 1 } ) =$ $e _ { i _ { t } }$ , realizing lines 5–6; on a D step it gives $\pi _ { c u r } ( y _ { t + 1 } ) = \pi _ { b u f } ( h ^ { \prime } ) = e _ { \mathrm { p a r } ( i _ { t } ) }$ while $\pi _ { n l }$ and π are unchanged, realizing line 9. In both cases the cursor advances, $\pi _ { p o s } ( y _ { t + 1 } ) = \hat { e } _ { t + 1 }$ □

## G Realizing st on Trees and Paths

Theorem 8. A four-layer, two-head Transformer decoder can compute st $\left( G _ { T } \right)$ , the Strahler number of a tree $G _ { T }$ during the simulation of DFS in exactly 2n − 1 CoT steps, where n denotes the number of vertices in $G _ { T }$

Proof. We show that the $t ^ { \mathrm { t h } }$ step of a four-layer, two-head decoder, extending that of Theorem 5 in its embedding and by two further layers, computes the Strahler number st $\left( G _ { T } \right)$ of the tree $G _ { T }$ during DFS traversal. We will follow Observation 2.

We extend the embedding from $\mathbb { R } ^ { 6 n + 3 } \ \mathrm { t o } \ \mathbb { R } ^ { 6 n + 1 4 }$ by appending ten scalar blocks, written $[ \cdots \parallel M \parallel c \parallel M _ { b u f } \parallel$ $c _ { b u f } \parallel f l g _ { 4 } \parallel t m p _ { D - 1 } \parallel \bar { t } m p _ { D } \parallel t m p _ { D + 1 } \parallel t m p _ { c _ { b u f } - 1 } \parallel t m p _ { c _ { b u f } } ^ { - } \parallel t m p _ { c _ { b u f } } \parallel t m p _ { c _ { b u f } + 1 } ]$ . In a generated token $y _ { t } .$ , the block M holds the running maximum of the Strahler numbers of the already-finished children of the current vertex, and $c \in \{ 0 , 1 , 2 \}$ counts how many of those children attain M. Within a backtrack step these blocks are transformed by the layers below before the invariant is restored in $y _ { t + 1 }$ . The blocks $M _ { b u f }$ and $c _ { b u f }$ are transient buffers, zero on every input and generated token and nonzero only within a step during a backtrack. The flag $f l g _ { 4 } \in \{ 0 , 1 , \ldots , 2 n - 1 \}$ is a timestamp, equal to 0 on every input token and on $y _ { 0 }$ and to t on the generated token $y _ { t }$ , so that the generated tokens are linearly ordered by $f l g _ { 4 }$ . The five blocks tmp<sub>·</sub> are scratch positions used by the later layers; their role is given where they first arise. Whereas the type flag $f l g _ { 3 }$ separates inputs from generated tokens by value in {0, 1}, the flag $f l g _ { 4 }$ records their order; supplied to the backtrack head of Theorem 5, it converts that head’s leftmost selection into a rightmost one, so that the head resolves the most recent rather than the first occurrence of the parent vertex, as made precise below.

The selectors of Theorem $^ { 5 }$ are retained, and we add $\pi _ { M } , \pi _ { c } , \pi _ { M _ { b u f } } , \pi _ { c _ { b u f } } , \pi _ { f l g _ { 4 } }$ and $\pi _ { t m p . } : \mathbb { R } ^ { d } $ R with $d = 6 n + 1 4$ The input tokens are those of the traversal with the new blocks set to zero,

$$
x _ { i } = \left[ e _ { i } \parallel 0 ^ { n } \parallel 0 ^ { n } \parallel A _ { i , : } \parallel 0 ^ { n } \parallel 0 ^ { n } \parallel - 1 \parallel - 1 \parallel 0 \parallel 0 ^ { 1 1 } \right] ^ { \top } ,
$$

so that $\pi _ { M } ( x _ { i } ) = \pi _ { c } ( x _ { i } ) = \pi _ { M _ { b u f } } ( x _ { i } ) = \pi _ { c _ { b u f } } ( x _ { i } ) = \pi _ { f l g _ { 4 } } ( x _ { i } ) = 0$ for every $v _ { i } \in V$ , and the sentinel remains $x _ { \perp } = 0 ^ { d }$ . Given the source $s = v _ { p } ,$ , the initial state is

$$
y _ { 0 } = \left[ e _ { p } \parallel 0 ^ { n } \parallel e _ { p } \parallel A _ { p , : } \parallel 0 ^ { n } \parallel 0 ^ { n } \parallel - 1 \parallel - 1 \parallel 1 \parallel 0 ^ { 1 1 } \right] ^ { \top } ,
$$

which agrees with the traversal’s initial state on the inherited blocks and carries an empty accumulator $\pi _ { M } ( y _ { 0 } ) =$ $\pi _ { c } ( y _ { 0 } ) = 0$ , empty buffers $\pi _ { M _ { b u f } } ( y _ { 0 } ) = \pi _ { c _ { b u f } } ( y _ { 0 } ) = 0$ , and timestamp $\pi _ { f l g _ { 4 } } ( y _ { 0 } ) = 0$ . We take the Strahler number of a leaf to be zero.

The selection effected by the backtrack head is the following. With respect to the $t ^ { \mathrm { t h } }$ query $Q y _ { t }$ , the scalar $a _ { i , t }$ is the attention score of the key against the token indexed i among $\{ x _ { \perp } , x _ { 0 } , \ldots , x _ { n - 1 } , y _ { 0 } , \ldots , y _ { t } \}$ . Let $j _ { 0 } < j _ { 1 } < \cdots < j _ { p }$ denote the indices, in ascending order, that jointly attain the maximum score. Where unique hard attention selects the smallest such index $j _ { 0 } ,$ , the inclusion of $f l g _ { 4 }$ in the key makes the head select the largest, $j _ { p } ,$ , the most recently generated token among those of maximal score.

The induction follows that of Theorem $^ { 5 , }$ recalled only to the extent the Strahler computation requires. Suppose $y _ { t }$ encodes the instantaneous description immediately before ${ \mathsf { D F S } } ^ { ( t + 1 ) } ( s )$ . The first layer leaves its attention trivial and, through the feed-forward block, writes into $\pi _ { f l g _ { 1 } } ( y _ { t } )$ the indicator ReL $\mathrm { \dot { U } } ( \gamma _ { t } ) - \mathrm { R e } \dot { \mathrm { L U } } ( \gamma _ { t } - 1 )$ ) of whether the current vertex has an unvisited neighbour, where $\gamma _ { t } = ( { \bf 1 } - \pi _ { v i s } ( y _ { t } ) ) ^ { \top } \pi _ { n b r } ( y _ { t } )$ , so that $\pi _ { f l g _ { 1 } } ( y _ { t } ^ { ( 1 ) } )$ is 1 on a traverse and 0 on a backtrack; it clears $\pi _ { f l g _ { 2 } }$ and copies $\pi _ { c u r } ( y _ { t } )$ into the buffer. The eleven new blocks are untouched by this layer. We denote the result $y _ { t } ^ { ( 1 ) }$

In the second layer the traverse head acts exactly as in Theorem $5 ,$ resolving the next unvisited neighbour on a traverse and defaulting to $x _ { \mid } ^ { ( 1 ) }$ on a backtrack. The backtrack head retains its query but is keyed to select the parent’s most recent occurrence rather than its first. Its query is the bilinear map

$$
Q _ { B } ^ { ( 2 ) } y _ { t } ^ { ( 1 ) } = ( 1 - \pi _ { f l g _ { 1 } } ( y _ { t } ^ { ( 1 ) } ) ) \cdot [ \pi _ { p a r } ( y _ { t } ^ { ( 1 ) } ) \parallel \pi _ { f l g _ { 3 } } ( y _ { t } ^ { ( 1 ) } ) ] ,
$$

which vanishes on a traverse and otherwise presents the parent indicator together with the scalar $\pi _ { f l g _ { 3 } } ( y _ { t } ^ { ( 1 ) } ) = 1$ . It is evaluated against the keys

$$
K _ { B } ^ { ( 2 ) } \alpha = [ \lambda \cdot \pi _ { c u r } ( \alpha ) \parallel \pi _ { f l g _ { 4 } } ( \alpha ) ] ,
$$

for $\alpha \in \{ x _ { \bot } ^ { ( 1 ) } , x _ { 0 } ^ { ( 1 ) } , \ldots , x _ { n - 1 } ^ { ( 1 ) } , y _ { 0 } ^ { ( 1 ) } , \ldots , y _ { t } ^ { ( 1 ) } \}$ , where the constant $\lambda \geq 2 n - 1$ exceeds every attainable timestamp. The resulting score $\lambda \cdot \pi _ { p a r } ( y _ { t } ^ { ( 1 ) } ) ^ { \top } \pi _ { c u r } ( \alpha ) + \pi _ { f l g _ { 4 } } ( \alpha )$ is lexicographic: the term $\lambda \cdot \pi _ { p a r } ( y _ { t } ^ { ( 1 ) } ) ^ { \top } \pi _ { c u r } ( \alpha )$ places every occurrence of the parent strictly above every non-occurrence, since λ alone exceeds any timestamp, while the constant $\pi _ { f l g _ { 3 } } ( y _ { t } ^ { ( 1 ) } ) = 1$ admits $\pi _ { f l g _ { 4 } } ( \alpha )$ into the score with unit weight. The head thus resolves the token $\alpha _ { B } ^ { \prime }$ of greatest timestamp among the occurrences of the parent, that is, the parent as it was last left. The query supplies π ${ \bf \nabla } _ { f l g _ { 3 } } ,$ fixed at 1, rather than its own $\pi _ { f l g _ { 4 } } = t .$ , which is common to all keys and so cannot order the parent occurrences.

The initial token $y _ { 0 }$ does not reach the backtrack head. Unless G is the null graph, the root s has an unvisited neighbor, so $\gamma _ { 0 } \geq 1$ and $\pi _ { f l g _ { 1 } } ( y _ { 0 } ^ { ( 1 ) } ) = 1$ ; the backtrack query vanishes and $y _ { 0 }$ is served by the traverse head.

We now specify the value carried by the two heads. On the inherited blocks they act as in Theorem $5 ;$ we extend only the backtrack value, and only on the blocks M and c. Let $\alpha _ { B } ^ { \prime }$ be the token resolved by the backtrack head. Its value map $V _ { B } ^ { ( 2 ) }$ moves the accumulator into the buffers and clears the originals,

$$
\begin{array} { r l } & { \pi _ { M _ { b u f } } ( V _ { B } ^ { ( 2 ) } \alpha _ { B } ^ { \prime } ) = \pi _ { M } ( \alpha _ { B } ^ { \prime } ) , \pi _ { c _ { b u f } } ( V _ { B } ^ { ( 2 ) } \alpha _ { B } ^ { \prime } ) = \pi _ { c } ( \alpha _ { B } ^ { \prime } ) , } \\ & { \pi _ { M } ( V _ { B } ^ { ( 2 ) } \alpha _ { B } ^ { \prime } ) = 0 , \pi _ { c } ( V _ { B } ^ { ( 2 ) } \alpha _ { B } ^ { \prime } ) = 0 , } \end{array}
$$

and is the identity on every other coordinate. The traverse value $\alpha _ { T } ^ { \prime \prime }$ is zero on all ten new blocks, so the buffers are written by the backtrack head alone, and $M , c , f l g _ { 4 }$ by neither head.

Form $\beta = \alpha _ { T } ^ { \prime \prime } + \alpha _ { B } ^ { \prime \prime } + y _ { t } ^ { ( 1 ) }$ . On a backtrack the backtrack query does not vanish; the buffers receive the parent accumulator while the head contributes zero to $M , c { \mathrm { : } }$

$$
\pi _ { M _ { b u f } } ( \beta ) = \pi _ { M } ( \alpha _ { B } ^ { \prime } ) , \quad \pi _ { c _ { b u f } } ( \beta ) = \pi _ { c } ( \alpha _ { B } ^ { \prime } ) ,
$$

$$
\pi _ { M } ( \beta ) = \pi _ { M } ( y _ { t } ^ { ( 1 ) } ) , \quad \pi _ { c } ( \beta ) = \pi _ { c } ( y _ { t } ^ { ( 1 ) } ) , \quad \pi _ { f l g _ { 4 } } ( \beta ) = t .
$$

On a traverse the query vanishes, $\alpha _ { B } ^ { \prime }$ defaults to $x _ { \perp } ^ { ( 1 ) } = 0 ^ { d }$ , and the buffers stay empty, $\pi _ { M _ { b u f } } ( \beta ) = \pi _ { c _ { b u f } } ( \beta ) = 0$ while $M ,$ c pass through from $y _ { t } ^ { ( 1 ) }$ . On a backtrack the pair $( \pi _ { M } ( \beta ) , \pi _ { c } ( \beta ) )$ is the accumulator of the vertex being left, v say, and the buffers hold the accumulator of its parent, u say.

Prior to define the parameters $W _ { 1 } ^ { ( 2 ) }$ , and $W _ { 2 } ^ { ( 2 ) }$ along with the subsequent layers, we set up the proof for defining the tmp blocks. Let $D = \mathsf { s t } ( v ) - \bar { M } _ { u }$ where $M _ { u } = \pi _ { M _ { b u f } } ( \beta )$ . Similarly, $c _ { v } = \pi _ { c } ( \beta )$ and $c _ { u } = \pi _ { c _ { b u f } } ( \beta )$ . With this we want

$$
\pi _ { M } ( y _ { t + 1 } ) = M _ { u } + \mathrm { R e L U } ( \mathsf { s t } ( v ) - M _ { u } ) + \mathbb { 1 } [ D = 0 ] \mathbb { 1 } [ c _ { u } > 0 ]\tag{5}
$$

$$
\pi _ { c } ( y _ { t + 1 } ) = \mathbb { 1 } [ D > 0 ] c _ { u } + \mathbb { 1 } [ D = 0 ] \mathbb { 1 } [ c _ { u } = 0 ] + \mathbb { 1 } [ D < 0 ] \mathbb { 1 }\tag{6}
$$

during backtrack.

The linear map $W _ { 1 } ^ { ( 2 ) }$ sets

$$
\begin{array} { r l } & { \pi _ { c } ( W _ { 1 } ^ { ( 2 ) } \beta ) = \pi _ { c } ( \beta ) - 1 , } \\ & { \pi _ { f l g _ { 4 } } ( W _ { 1 } ^ { ( 2 ) } \beta ) = \pi _ { f l g _ { 4 } } ( \beta ) + 1 , } \\ & { \pi _ { t m p _ { c _ { b u f } - 1 } } ( W _ { 1 } ^ { ( 2 ) } \beta ) = \pi _ { c _ { b u f } } ( \beta ) - 1 , } \end{array}
$$

leaving the other blocks intact; after the activation we write the result $\beta ^ { \prime }$ . Since $c _ { v } \in \{ 0 , 1 , 2 \}$ , the rectified count is $\pi _ { c } ( \beta ^ { \prime } ) = \mathrm { R e L U } ( c _ { v } - 1 ) = \mathbb { 1 } [ c _ { v } \geq 2 ]$ . The linear map $W _ { 2 } ^ { ( 2 ) }$ then forms $y _ { t } ^ { ( 2 ) }$ by

$$
\pi _ { M } ( \boldsymbol { y } _ { t } ^ { ( 2 ) } ) = \pi _ { M } ( \beta ^ { \prime } ) + \pi _ { c } ( \beta ^ { \prime } ) = M _ { v } + \mathbb { 1 } [ c _ { v } \geq 2 ] = \mathsf { s t } ( v ) ,
$$

and, writing $\pi _ { M } ( \beta ^ { \prime } ) + \pi _ { c } ( \beta ^ { \prime } ) - \pi _ { M _ { b u f } } ( \beta ^ { \prime } ) = \mathsf { s t } ( v ) - M _ { u } = D ,$

$$
\begin{array} { r l } & { \pi _ { t m p _ { D - 1 } } ( y _ { t } ^ { ( 2 ) } ) = D - 1 , } \\ & { \pi _ { t m p _ { D } } ( y _ { t } ^ { ( 2 ) } ) = D , } \\ & { \pi _ { t m p _ { D + 1 } } ( y _ { t } ^ { ( 2 ) } ) = D + 1 . } \end{array}
$$

After the second layer the block M carries $\mathsf { s t } ( v )$ , and the blocks t $m p _ { D - 1 } , t m p _ { D } , t m p _ { D + 1 } \mathrm { c a r r y } D - 1 , D , D + 1$

The third and fourth layers leave the attention trivial, $W _ { O } ^ { ( 3 ) } = W _ { O } ^ { ( 4 ) } = \mathbf { 0 }$ , and act through their feed-forward blocks alone. The linear map $W _ { 1 } ^ { ( 3 ) }$ extends $y _ { t } ^ { ( 2 ) }$ by six scalar blocks, $1 - D , \ - D , \ - D - 1 , \ 1 - c _ { u } , \ - c _ { u }$ , and $- c _ { u } - 1$ The activation reads off indicators through the integer identities

$$
\begin{array} { r l } & { \mathbb { 1 } [ v = 0 ] = \mathrm { R e L U } ( v + 1 ) - 2 \mathrm { R e L U } ( v ) + \mathrm { R e L U } ( v - 1 ) } \\ & { \mathbb { 1 } [ v > 0 ] = \mathrm { R e L U } ( v ) - \mathrm { R e L U } ( v - 1 ) . } \end{array}
$$

write the rectified vector $\beta ^ { \prime \prime }$ . The linear map $W _ { 2 } ^ { ( 3 ) }$ then produces $y _ { t } ^ { ( 3 ) } \in \mathbb { R } ^ { d }$ , returning the indicators to the tmp blocks:

$$
\begin{array} { r l } & { \pi _ { M } ( y _ { t } ^ { ( 3 ) } ) = \pi _ { M } ( \beta ^ { \prime \prime } ) + \pi _ { - D } ( \beta ^ { \prime \prime } ) } \\ & { \qquad = \operatorname* { m a x } ( s \in [ \boldsymbol { v } ] , M _ { u } ) , } \\ & { \pi _ { t m p _ { D - 1 } } ( y _ { t } ^ { ( 3 ) } ) = \mathbb { 1 } [ D > 0 ] , } \\ & { \pi _ { t m p _ { D } } ( y _ { t } ^ { ( 3 ) } ) = \mathbb { 1 } [ D = 0 ] , } \\ & { \pi _ { t m p _ { D - 1 } } ( y _ { t } ^ { ( 3 ) } ) = \mathbb { 1 } [ D < 0 ] , } \\ & { \pi _ { t m p _ { D + 1 } } ( y _ { t } ^ { ( 3 ) } ) = \mathbb { 1 } [ c _ { u } > 0 ] , } \\ & { \pi _ { t m p _ { E _ { k n f } } } ( y _ { t } ^ { ( 3 ) } ) = \mathbb { 1 } [ c _ { u } > 0 ] , } \\ & { \pi _ { t m p _ { E _ { k n f - 1 } } } ( y _ { t } ^ { ( 3 ) } ) = \mathbb { 1 } [ c _ { u } = 0 ] . } \end{array}
$$

The 1s the indicators require are supplied by $\pi _ { f l g _ { 3 } } = 1$ . The fourth layer carries out the accumulation. Recall that after the third layer the blocks $t m p _ { D - 1 } , t m p _ { D } , t m p _ { D + 1 }$ <sub>1</sub> hold $\mathbb { 1 } [ D > 0 ] , \mathbb { 1 } [ D = 0 ] , \mathbb { 1 } [ D < 0 ]$ and the blocks $t m p _ { c _ { b u f } } , t m p _ { c _ { b u f } - 1 }$ hold $\mathbb { 1 } [ c _ { u } > 0 ] , \mathbb { 1 } [ c _ { u } = 0 ]$ . The bilinear map $W _ { 1 } ^ { ( 4 ) }$ sets

$$
\pi _ { M } ( W _ { 1 } ^ { ( 4 ) } y _ { t } ^ { ( 3 ) } ) = \pi _ { M } ( y _ { t } ^ { ( 3 ) } ) + \pi _ { t m p _ { D } } ( y _ { t } ^ { ( 3 ) } ) \pi _ { t m p _ { c _ { b u f } } } ( y _ { t } ^ { ( 3 ) } ) ,
$$

$$
\pi _ { c } ( W _ { 1 } ^ { ( 4 ) } y _ { t } ^ { ( 3 ) } ) = \pi _ { t m p D - 1 } ( y _ { t } ^ { ( 3 ) } ) 1 + \pi _ { t m p D } ( y _ { t } ^ { ( 3 ) } ) \pi _ { t m p _ { c _ { b u f } - 1 } } ( y _ { t } ^ { ( 3 ) } ) + \pi _ { t m p D + 1 } ( y _ { t } ^ { ( 3 ) } ) \pi _ { c _ { b u f } } ( y _ { t } ^ { ( 3 ) } ) ,
$$

which, by the contents of those blocks, are $\pi _ { M } ( y _ { t } ^ { ( 3 ) } ) + \mathbb { 1 } [ D = 0 ] \mathbb { 1 } [ c _ { u } > 0 ]$ and $\mathbb { 1 } [ D > 0 ] c _ { u } + \mathbb { 1 } [ D = 0 ] \mathbb { 1 } [ c _ { u } =$ $0 ] + \mathbb { 1 } [ D < 0 ] 1$ , realizing (5) and (6). Each is a product of two blocks of $y _ { t } ^ { ( 3 ) }$ , hence of degree two. The map further resets $M _ { b u f } , c _ { b u f }$ , and the tmp blocks to zero, leaving the remaining coordinates intact; write the result $\beta ^ { \prime \prime \prime }$

The bilinear map $W _ { 2 } ^ { ( 4 ) }$ corrects M and c for the traverse case, assigning

$$
\pi _ { M } ( y _ { t + 1 } ) = ( 1 - \pi _ { f l g _ { 1 } } ( y _ { t } ^ { ( 3 ) } ) ) \pi _ { M } ( \beta ^ { \prime \prime \prime } ) ,\tag{7}
$$

$$
\pi _ { c } ( y _ { t + 1 } ) = ( 1 - \pi _ { f l g _ { 1 } } ( y _ { t } ^ { ( 3 ) } ) ) \pi _ { c } ( \beta ^ { \prime \prime \prime } ) ,\tag{8}
$$

and sets $f l g _ { 2 } = 0 , f l g _ { 3 } = 1$ . On a backtrack the gate is 1 and the accumulator commits; on a traverse it is 0, so the freshly visited vertex receives the empty accumulator $( 0 , 0 )$ . Each coordinate is a product of two blocks, hence of degree two. This is $y _ { t + 1 }$

At the $( 2 n - 1 ) ^ { \mathrm { t h } }$ step the current restriction becomes $0 ^ { n }$ ; this final backtrack folds s under the same invariant, giving $\pi _ { M } ( y _ { 2 n - 1 } ) = { \mathsf { s t } } ( s ) = { \mathsf { s t } } ( G _ { T } )$ 口

Lemma 9. Let $d \geq 2 ,$ , and let $v = [ m \parallel h \parallel \mathbf { 0 } _ { d } ] ^ { \top } \in \mathbb { R } ^ { 2 d + 1 }$ be an input vector where m $\in \{ - 1 , + 1 \} , h \in \{ 0 , 1 \} ^ { d }$ is a one-hot vector, and $\mathbf { 0 } _ { d }$ is the zero vector in $\mathbb { R } ^ { d } . L e t h ^ { \prime } \in \{ 0 , 1 \} ^ { d }$ denote the zero-padded, one-position shift ofh directed by the sign ofm (right-shifted $i f m = + 1$ , left-shifted $i f m = - 1 )$ . There exists an FFN with ReLU activation $W _ { 2 } \mathrm { R e L U } ( W _ { 1 } v + b _ { 1 } ) + b _ { 2 }$ , that produces $[ m \parallel h \parallel \mathbf { \bar { \Sigma } } h ^ { \prime } ] ^ { \top }$

Proof. Let $I _ { d }$ be the $d \times d$ identity matrix, $\mathbf { 1 } _ { d }$ be the d-dimensional all-ones vector, and $\mathbf { 0 } _ { r \times c }$ be the $r \times c$ zero matrix. Define the right-shift matrix $S _ { R } \in \mathbf { \mathbb { R } } ^ { d \times d }$ such that $( S _ { R } ) _ { i , j } = 1 { \mathrm { i f } } i - j = 1$ and 0 otherwise, and let the left-shift matrix be $S _ { L } = S _ { R } ^ { \top }$

Construct the first linear projection $W _ { 1 } \in \mathbb { R } ^ { ( 3 d + 2 ) \times ( 2 d + 1 ) }$ and bias $b _ { 1 } \in \mathbb { R } ^ { 3 d + 2 }$ to partition the hidden representation into five blocks:

$$
W _ { 1 } = \left( \begin{array} { c c c } { { 1 } } & { { { \bf 0 } _ { d } ^ { \top } } } & { { { \bf 0 } _ { d } ^ { \top } } } \\ { { - 1 } } & { { { \bf 0 } _ { d } ^ { \top } } } & { { { \bf 0 } _ { d } ^ { \top } } } \\ { { { \bf 0 } _ { d } } } & { { I _ { d } } } & { { { \bf 0 } _ { d \times d } } } \\ { { 0 . 5 { \bf 1 } _ { d } } } & { { S _ { R } } } & { { { \bf 0 } _ { d \times d } } } \\ { { - 0 . 5 { \bf 1 } _ { d } } } & { { S _ { L } } } & { { { \bf 0 } _ { d \times d } } } \end{array} \right) , b _ { 1 } = \left[ \begin{array} { c } { { 0 } } \\ { { 0 } } \\ { { { \bf 0 } _ { d } } } \\ { { - 0 . 5 { \bf 1 } _ { d } } } \\ { { - 0 . 5 { \bf 1 } _ { d } } } \end{array} \right]
$$

Applying ReL $. \mathrm { U } ( W _ { 1 } v + b _ { 1 } )$ yields the hidden vector $z = [ z _ { m , + } , z _ { m , - } , z _ { h } , z _ { + } , z _ { - } ] ^ { \top }$

• Scalars $z _ { m , + } = \mathrm { R e L U } ( m )$ and $z _ { m , - } = \mathrm { R e L U } ( - m )$ isolate the sign of $m$

• Since $h \in \{ 0 , 1 \} ^ { d }$ is non-negative, $z _ { h } = \mathrm { R e L U } ( h ) = h$

• The shift gates resolve to mutually exclusive activations: if $m = + 1 , z _ { + } = S _ { R } h$ and $z _ { - } = 0 _ { d } ;$ if $m = - 1$ $z _ { + } = \mathbf { 0 } _ { d }$ and $z _ { - } = S _ { L } h$

Construct the second linear projection $W _ { 2 } \in \mathbb { R } ^ { ( 2 d + 1 ) \times ( 3 d + 2 ) }$ and bias $b _ { 2 } = { \bf 0 } _ { 2 d + 1 }$ as:

$$
W _ { 2 } = \left( \begin{array} { c c c c c } { { 1 } } & { { - 1 } } & { { \mathbf { 0 } _ { d } ^ { \top } } } & { { \mathbf { 0 } _ { d } ^ { \top } } } & { { \mathbf { 0 } _ { d } ^ { \top } } } \\ { { \mathbf { 0 } _ { d } } } & { { \mathbf { 0 } _ { d } } } & { { I _ { d } } } & { { \mathbf { 0 } _ { d \times d } } } & { { \mathbf { 0 } _ { d \times d } } } \\ { { \mathbf { 0 } _ { d } } } & { { \mathbf { 0 } _ { d } } } & { { \mathbf { 0 } _ { d \times d } } } & { { I _ { d } } } & { { I _ { d } } } \end{array} \right)
$$

Multiplying $W _ { 2 } z + b _ { 2 }$ sums the mutually exclusive shift gates and reconstructs the input blocks:

$$
v ^ { \prime } = \left[ { \begin{array} { c } { { z _ { m , + } - z _ { m , - } } } \\ { { I _ { d } z _ { h } } } \\ { { I _ { d } z _ { + } + I _ { d } z _ { - } } } \end{array} } \right] = \left[ { \begin{array} { c } { { m } } \\ { { h } } \\ { { h ^ { \prime } } } \end{array} } \right] .
$$

Lemma 10. For a real bound $\lambda \geq 0 ,$ and let $v = [ m \parallel M \parallel c ] ^ { \top } \in \{ - 1 , 1 \} \times [ 0 , \lambda ] \times [ 0 , \lambda ]$ be the input vector. There exists an FFN with ReLU activation $W _ { 2 } \mathrm { R e L U } ( \dot { W } _ { 1 } v ^ { ' } + b _ { 1 } ) + \dot { b } _ { 2 } ,$ , such that the output vector satisfies the following:

$$
\begin{array} { r } { v ^ { \prime } = \left\{ \begin{array} { l l } { [ 1 \parallel 0 \parallel 0 ] ^ { \top } } & { i f m = 1 , } \\ { [ - 1 \parallel M \parallel c ] ^ { \top } } & { i f m = - 1 . } \end{array} \right. } \end{array}
$$

Proof. For any scalar $x \in [ 0 , \lambda ]$ and a variable $m \in \{ - 1 , 1 \}$ , define the scalar gating function:

$$
g ( m , x ) = \mathrm { R e L U } ( x ) - \mathrm { R e L U } \left( x + { \frac { \lambda } { 2 } } m - { \frac { \lambda } { 2 } } \right) .
$$

When $m = 1$ , the second term simplifies to $\scriptstyle { \mathrm { R e L U } } ( x )$ , yielding $g ( 1 \parallel x ) = \mathrm { R e L U } ( x ) - \mathrm { R e L U } ( x ) = 0$ . When $m = - 1$ the second term becomes ReL $. \mathrm { U } ( x - \lambda ) = 0$ because $x \leq \lambda .$ , leaving $g ( - 1 \parallel \dot { x } ) = \mathrm { R e L U } ( x ) = x$ since $x \ge 0$ Furthermore, the identity mapping for the binary variable m can be expressed exactly as $m = { \mathrm { R e L U } } ( m ) - { \mathrm { R e L U } } ( - m )$

We construct $h = \mathrm { R e L U } ( W _ { 1 } v + b _ { 1 } )$ by defining the first layer’s parameters as:

$$
W _ { 1 } = \left( \begin{array} { c c c } { { 1 } } & { { 0 } } & { { 0 } } \\ { { - 1 } } & { { 0 } } & { { 0 } } \\ { { 0 } } & { { 1 } } & { { 0 } } \\ { { \frac { \lambda } { 2 } } } & { { 1 } } & { { 0 } } \\ { { 0 } } & { { 0 } } & { { 1 } } \\ { { \frac { \lambda } { 2 } } } & { { 0 } } & { { 1 } } \end{array} \right) , \quad b _ { 1 } = \left[ \begin{array} { c } { { 0 } } \\ { { 0 } } \\ { { 0 } } \\ { { - \frac { \lambda } { 2 } } } \\ { { 0 } } \\ { { - \frac { \lambda } { 2 } } } \end{array} \right] .
$$

Applying the ReLU activation yields the hidden state vector:

$$
\begin{array} { r } { h = \mathrm { R e L U } \left[ \begin{array} { c } { m } \\ { - m } \\ { M } \\ { M + \frac { \lambda } { 2 } m - \frac { \lambda } { 2 } } \\ { c } \\ { c + \frac { \lambda } { 2 } m - \frac { \lambda } { 2 } } \end{array} \right] . } \end{array}
$$

Setting the output bias to zero $( b _ { 2 } = \mathbf { 0 } )$ and defining the output projection matrix $W _ { 2 }$ as:

$$
W _ { 2 } = \left( { \small \begin{array} { c c c c c c } { 1 } & { - 1 } & { 0 } & { 0 } & { 0 } & { 0 } \\ { 0 } & { 0 } & { 1 } & { - 1 } & { 0 } & { 0 } \\ { 0 } & { 0 } & { 0 } & { 0 } & { 1 } & { - 1 } \end{array} } \right) , \ 
$$

we obtain the network output:

$$
\mathcal { N } ( v ) = \left[ \begin{array} { c } { \mathrm { R e L U } ( m ) - \mathrm { R e L U } ( - m ) } \\ { g ( m \parallel M ) } \\ { g ( m \parallel c ) } \end{array} \right] = \left[ \begin{array} { c } { m } \\ { g ( m \parallel M ) } \\ { g ( m \parallel c ) } \end{array} \right] .
$$

Evaluating this vector at $m = 1$ yields $[ 1 \parallel 0 \parallel 0 ] ^ { \top }$ , and evaluating at $m = - 1$ yields $[ - 1 \parallel M \parallel c ] ^ { \top }$

Theorem 11. A four-layer, single-head Transformer decoder can compute $\mathsf { s t } ( \psi ( w ) )$ , the Strahler number of a tree corresponding to the Dyck word w in exactly n CoT steps, where $n = | w |$

ProofSketch. Like other proofs, we show that the $t ^ { \mathrm { { t h } } }$ step of a four-layer single head decoder can calculate $\mathsf { s t } ( \psi ( w ) )$ .   
The proof will rely on the following observation and Observation 2.

Observation 3. Let the Dyck word $w = w _ { 0 } \cdot \cdot \cdot w _ { n - 1 }$ encodes the lattice path $\{ ( \tau , s _ { \tau } ) \} _ { \tau = 0 } ^ { n }$ . Let $q _ { \tau } = s _ { \tau } - s _ { \tau - 1 }$ . For a time step $t , \mathsf { s t } ( \psi ( w _ { 0 } \cdot \cdot \cdot w _ { t } ) )$ ) depends on two $( M , c )$ -pairs (see Observation 2) if $q _ { t } = - 1$ . The first is the $( M _ { v } , c _ { v } )$ -pair of the vertex v with co-ordinate $( t , s _ { t } )$ , while the second $( M _ { u } , c _ { u } )$ is associated with the vertex u with co-ordinate $( t ^ { \prime } , s _ { t ^ { \prime } } )$ such that $\nexists t ^ { \prime \prime } , t ^ { \prime } < t ^ { \prime \prime } < t$ and $s _ { t ^ { \prime \prime } } = s _ { t ^ { \prime } }$

Based on this, we shall provide the construction up to the second layer alone and the last layer partially. The proof thereon follows Theorem 8. Assume $\mathcal { E } _ { n } = \{ e _ { 0 } , . . . , \bar { e } _ { n - 1 } \}$ denotes the standard basis vector on $\mathbb { R } ^ { n }$ . Let $\dot { S } \dot { \in } \{ 0 , 1 \dot { \} } ^ { n \times n }$ be the shift $s _ { i j } = \mathbb { 1 } [ j = i - 1 ]$ (so $S e _ { k } = e _ { k + 1 }$ and $S e _ { n - 1 } = 0 )$ . The input tokens $x _ { 0 } , \ldots , x _ { n - 1 } \in \mathbb { R } ^ { d }$ where $d = 3 n + 1 5$ encodes the lattice paths $\{ ( \tau , s _ { \tau } ) \} _ { \tau = 0 } ^ { n }$ as follows:

$$
\begin{array} { r } { x _ { \tau } = [ e _ { \tau } \parallel \tau + 1 \parallel 0 \parallel q _ { \tau } \parallel 0 \parallel e _ { s _ { \tau } } \parallel 0 ^ { n } \parallel 0 ^ { 4 } \parallel 0 \parallel 0 ^ { 6 } ] ^ { \top } , } \end{array}
$$

for $\tau = 0 \mathrm { t o } n - 1$ . The Dynamic is carried out by tokens $y _ { t } \in \mathbb R ^ { d }$ with

$$
y _ { 0 } = [ e _ { 0 } \parallel 1 \parallel 0 \parallel 1 \parallel 0 \parallel e _ { 0 } \parallel 0 ^ { n } \parallel 0 ^ { 4 } \parallel 1 \parallel 0 ^ { 6 } ] ^ { \top } .
$$

The selectors

$$
\begin{array} { r l } & { \pi _ { p o s } : \mathbb { R } ^ { d } \to \mathbb { R } ^ { n } , } \\ & { \pi _ { p o s _ { s } } : \pi _ { t r m p _ { p o s _ { e } } } , \pi _ { t r m p _ { q } } : \mathbb { R } ^ { d } \to \mathbb { R } , } \\ & { \pi _ { q } : \mathbb { R } ^ { d } \to \{ - 1 , + 1 \} , } \\ & { \pi _ { t h } , \pi _ { t r m p _ { h i } } : \mathbb { R } ^ { d } \to \mathbb { R } ^ { n } , } \\ & { \pi _ { M } , \pi _ { c } , \pi _ { M _ { b a f } , \pi _ { c b a f } } : \mathbb { R } ^ { d } \to \mathbb { R } , } \\ & { \pi _ { f l } : \mathbb { R } ^ { d } \to \{ 0 , 1 \} , } \\ &  \pi _ { t m p _ { D - 1 } , \pi _ { t m p _ { D } } , \pi _ { t m p _ { D + 1 } } , \pi _ { t m p _ { c _ { b a f } - 1 } } , \pi _ { t m p _ { c _ { b a f } } } , } \\ & { \mathrm { a n d } \pi _ { t m p _ { c _ { b a f } + 1 } } : \mathbb { R } ^ { d } \to \mathbb { R } , } \end{array}
$$

extract the relevant blocks from an embedding in $\mathbb { R } ^ { d }$ . The first three blocks correspond to the one-hot position $\tau ,$ its decimal equivalent, and a temporary block. The blocks denoted by $q$ and $t m p _ { q }$ register $q _ { \tau }$ and its temporary computation trace. The following two store the one-hot equivalent of $s _ { \tau }$ and an intermediate buffer. The blocks M and c correspond to the $M _ { v }$ and $c _ { v }$ of $( 2 )$ . Similarly, $M _ { b u f }$ and $c _ { b u f }$ correspond to $M _ { u }$ and $c _ { u } .$ . The block $f l g$ distinguishes the input tokens $x _ { i } \mathrm { s } \left( f l g = 0 \right)$ from the generated tokens y s $( { \dot { f } } l g = 1 )$ , while tmp $f l g$ is for temporary calculation. The remaining blocks continue to serve the purpose as in Theorem 8. We argue that st $\mathbf { \dot { \rho } } ( \dot { \psi } ( w ) ) = \pi _ { M } ^ { \mathbf { \dot { \rho } } } ( y _ { n } ) + \mathbb { 1 } [ \pi _ { c } ( y _ { n } ) ]$

The first layer will update the blocks pos, q, and ht. The query exacts the block pos and shifts the one $Q ^ { ( 1 ) } y _ { t } =$ $S \pi _ { p o s } ( y _ { t } )$ and evaluated against keys $K \alpha = \pi _ { p o s } ( \alpha )$ for all $\alpha \in \left\{ x _ { 0 } , x _ { 1 } , \ldots , x _ { n - 1 } , y _ { 0 } , \ldots , y _ { t } \right\}$ . The UHA selects the lowest-indexed match from $x _ { i } { \ ' } _ { \mathbf { S } } .$ , say $\alpha ^ { \prime }$ . The vector after value projection is $\alpha ^ { \prime \prime }$ . Then $\pi _ { p o s } ( \alpha ^ { \prime \prime } ) = - \pi _ { p o s } ( \alpha ^ { \prime } )$ $\pi _ { t m p _ { p o s s c } } ( \alpha ^ { \prime \prime } ) = \pi _ { p o s _ { s c } } ( \alpha ^ { \prime } ) , \pi _ { t m p _ { q } } ( \alpha ^ { \prime \prime } ) = \pi _ { q } ( \alpha ^ { \prime } )$ , and, leaving rest to 0. Consider $W _ { O } ^ { ( 1 ) }$ being the identity, note that the residual connection in $\alpha ^ { \prime \prime } + y _ { t }$ does not affect the non-zero values of $y _ { t }$ in any of its blocks. The linear projection $W _ { 1 } ^ { ( 1 ) }$ performs two operations. It transforms the block into $- \pi _ { p o s } ( \alpha ^ { \prime \prime } + y _ { t } )$ and fulfills the role of $W _ { 1 }$ in Lemma 9, where the blocks tmp , h, and tmp are treated as placeholders for $m , h ,$ and $h ^ { \prime } .$ , respectively. After activation ReLU, the block pos will register $S \pi _ { p o s } ( y _ { t } )$ . Followed by ${ W } _ { 2 } ^ { ( 1 ) }$ , the block $t m p _ { h }$ stores the one-hot encoding of $s _ { \tau + 1 }$ . Let the resulting representation be denoted by $y _ { t } ^ { ( 1 ) }$ . It is important to emphasize that, with the exception of the block $f l g ,$ , all updates and value assignments within the constituent blocks of $y _ { t } ^ { ( 1 ) }$ are carried out in auxiliary blocks. This ensures that the actual blocks associated with the preceding tokens in the layered computation remain unchanged throughout the intermediate computational stages. Since $f l g$ does not participate in both the query and key computations of the remaining layers, it remains unaffected by these operations.

The subsequent layers work by guessing that $q _ { \tau + 1 }$ is a $\textsf { D } ( - 1 )$ step. However, the last layer applies Lemma $1 0 ^ { 6 }$ to perform the final correction in M and c blocks depending on the value of the block $q$ in $y _ { t } ^ { ( 4 ) }$ . The second layer will realize the observation we made earlier in this proof – finding the appropriate token (among $y _ { i } s \ i < t )$ such that $\pi _ { h t } ( y _ { i } ) = \pi _ { t m p _ { h t } } ( y _ { t } ^ { ( 1 ) } )$ and $\nexists i ^ { \prime } , i < i ^ { \prime } < t$ and $\pi _ { h t } ( y _ { i ^ { \prime } } ) = \pi _ { t m p _ { h t } } ( y _ { t } ^ { ( 1 ) } )$ ). Let the query be $Q ^ { ( 2 ) } \bar { y } _ { t } ^ { ( \bar { 1 } ) } = [ \pi _ { t m p _ { h t } } ( y _ { t } ^ { ( 1 ) } )$ ∥ $\pi _ { f l g } ( y _ { t } ^ { ( 1 ) } ) ]$ and the key be $K ^ { ( 2 ) } \alpha = \left[ \lambda \pi _ { h t } ( \alpha ) \parallel \right] \pi _ { p o s _ { s c } } ( \alpha ) ]$ for all $\alpha \in \{ x _ { 0 } ^ { ( 1 ) } , x _ { 1 } ^ { ( 1 ) } , \ldots , x _ { n - 1 } ^ { ( 1 ) } , y _ { 0 } ^ { ( 1 ) } , \ldots , y _ { t } ^ { ( 1 ) } \}$ and $\lambda > n + 1$ . Let this vector be $\alpha ^ { \prime }$ . Clearly $\pi _ { h t } ( y _ { t } ^ { ( 1 ) } ) ^ { \top } \pi _ { t m p _ { h t } } ( y _ { t } ^ { ( 1 ) } ) = 0$ ensures that we obtain a right-most $i < t .$ . The value projection $V ^ { ( 2 ) }$ then

$$
\begin{array} { r l } & { \pi _ { M _ { b u f } } ( V ^ { ( 2 ) } \alpha ^ { \prime } ) = \pi _ { M } ( \alpha ^ { \prime } ) , \ \pi _ { c _ { b u f } } ( V ^ { ( 2 ) } \alpha ^ { \prime } ) = \pi _ { c } ( \alpha ^ { \prime } ) , } \\ & { \pi _ { M } ( V ^ { ( 2 ) } \alpha ^ { \prime } ) = 0 , \quad \pi _ { c } ( V ^ { ( 2 ) } \alpha ^ { \prime } ) = 0 . } \end{array}
$$

The subsequent operations will be analogous to that of Theorem 8 that realizes Observation 2 using the blocks M, $c , M _ { b u f } , c _ { b u f } , t m p _ { D - 1 } , t m p _ { D } , t m p _ { D + 1 } , t m p _ { c _ { b u f } - 1 } , t m p _ { c _ { b u f } } , t m p _ { c _ { b u f } + 1 }$ , and $f l g$ in place of $f l g _ { 3 }$ of Theorem 8. However, in the fourth layer we employ Lemma 10 that in its FFN block by dissolving the gating (7) in $W _ { 2 } ^ { ( 4 ) }$ . This final projection $W _ { 2 } ^ { ( 4 ) }$ produces $y _ { t + 1 }$ by nullifying all temporary positions along with the buffers $M _ { b u f }$ and $c _ { b u f }$ . Finally, $\mathsf { s t } ( \bar { \psi } ( w ) ) = \bar { \pi } _ { M } ( \bar { y } _ { n } ) + \mathbb { 1 } [ \bar { \pi } _ { c } ( y _ { n } ) \geq 2 ]$

It should be noted that the pair $( t ^ { \prime } , s _ { t ^ { \prime } } )$ in Observation 3 plays a pivotal role in transmitting the Strahler rank to the current vertex. In the absence of this contribution, the pair $( M _ { v } , c _ { v } )$ alone would not suffice to determine the Strahler rank at the current vertex (see Figure 3).

![](images/b4101eeb2e69f452f839ddddade8c1dca3ba324e389452c63b95e34e63ab9102.jpg)  
Figure 3: As illustrated in Figure 2, calculating the Strahler number at the point (24, 2) without incorporating the specific pair $( t ^ { \prime } , s _ { t ^ { \prime } } ) = ( 1 6 , 2 )$ (from Observation 3) yields a value equal to that evaluated at (20, 2) in the above path. This would incorrectly suggest a generation-invariant property of the Strahler number for previously explored children.

## H Realizing wd on Trees and Paths

Lemma 12. Let $v = [ v _ { 1 } \parallel v _ { 2 } \parallel v _ { 3 } ] ^ { \top } \in \mathbb { R } ^ { 3 }$ be an input vector such that $v _ { 1 } \in \{ 0 , 1 \}$ , and $v _ { 2 } , v _ { 3 } \in \mathbb { Z } ^ { + }$ are bounded such that $v _ { 2 } \le \lambda f o r c$ a large λ. Let the target transformation $f : \mathbb { R } ^ { 3 } \to \mathbb { R } ^ { 3 }$ be defined piecewise as:

$$
f ( v ) = \left\{ \begin{array} { l l } { [ 0 \parallel v _ { 2 } + 1 \parallel \operatorname* { m a x } ( v _ { 3 } , v _ { 2 } + 1 ) ] ^ { \top } i f v _ { 1 } = 0 } \\ { [ 1 \parallel 1 \parallel \operatorname* { m a x } ( v _ { 3 } , 1 ) ] ^ { \top } i f v _ { 1 } = 1 } \end{array} \right.
$$

It can be exactly computed by a single hidden-layer FFN utilizing a ReLU activationfunction with a hidden dimension $o f 4 ,$ taking the form $f ( v ) = W _ { 2 } \mathrm { R e L U } ( W _ { 1 } v + b _ { 1 } ) + b _ { 2 }$

Proof. To construct the exact FFN, let $\lambda \in \mathbb { R }$ be a strictly positive large scalar.

Define the weight matrix $W _ { 1 } \in \mathbb { R } ^ { 4 \times 3 }$ and bias vector $b _ { 1 } \in \mathbb { R } ^ { 4 }$

$$
W _ { 1 } = \left( { \begin{array} { c c c } { 1 } & { 0 } & { 0 } \\ { - \lambda } & { 1 } & { 0 } \\ { 0 } & { 0 } & { 1 } \\ { - \lambda } & { 1 } & { - 1 } \end{array} } \right) , \quad b _ { 1 } = [ 0 \parallel 0 \parallel 0 \parallel 1 ] ^ { \intercal }
$$

Evaluating the pre-activation state $z = W _ { 1 } v + b _ { 1 }$ yields $z = [ v _ { 1 } \parallel v _ { 2 } - \lambda v _ { 1 } \parallel v _ { 3 } \parallel v _ { 2 } - v _ { 3 } - \lambda v _ { 1 } + 1 ] ^ { \top }$

Let $h = \mathrm { R e L U } ( z ) = \operatorname* { m a x } ( 0 , z )$ . Because $v _ { 1 } \in \{ 0 , 1 \}$ and $v _ { 3 } \geq 1$ , the first and third components are strictly nonnegative and pass through unchanged $( h _ { 1 } = v _ { 1 } , h _ { 3 } = v _ { 3 } )$ . To evaluate the conditional components for both valid states of $v _ { 1 } .$

Case $v _ { 1 } = 0 ;$ The second term is $v _ { 2 }$ , and the fourth term is $v _ { 2 } - v _ { 3 } + 1$ . Thus, the hidden state is $h = [ 0 \parallel v _ { 2 } \parallel v _ { 3 }$ ∥ max $\mathbf { [ 0 , } v _ { 2 } - v _ { 3 } + 1 ) ] ^ { \top }$

Case $v _ { 1 } = 1 \mathrm { : }$ : The second term becomes $v _ { 2 } - \lambda$ . Since $\lambda > \lambda \ge v _ { 2 }$ , this is strictly negative and clamps to zero. The fourth term becomes $v _ { 2 } - v _ { 3 } - \lambda + 1$ . Since $v _ { 2 } - \lambda + 1 \leq 0$ and $v _ { 3 } \geq 1$ , this entire expression is strictly negative and also clamps to zero. Thus, the hidden state is $\boldsymbol { \overline { { h } } } = [ 1 \ \lVert \ 0 \ \rVert \ \overline { { v } } _ { 3 } \ \lVert \ 0 \rVert ^ { \top }$

The final projection $W _ { 2 } \in \mathbb { R } ^ { 3 \times 4 }$ and output bias $b _ { 2 } \in \mathbb { R } ^ { 3 }$

$$
W _ { 2 } = { \binom { 1 } { 0 } } \quad { \stackrel { 0 } { 1 } } \quad { \stackrel { 0 } { 0 } } \quad { \stackrel { 0 } { 0 } } \quad { \stackrel { 1 } { 0 } } = [ 0 \parallel 1 \parallel 0 ] ^ { \top }
$$

Applying this final projection verifies the transformation exactly maps to the conditions of $f ( v )$ :

• For $v _ { 1 } = 0 \colon$

$$
\begin{array} { r l r } & { } & { f ( \boldsymbol { v } ) = \left( \begin{array} { c c c c } { 1 } & { 0 } & { 0 } & { 0 } \\ { 0 } & { 1 } & { 0 } & { 0 } \\ { 0 } & { 0 } & { 1 } & { 1 } \end{array} \right) \left[ \begin{array} { c } { 0 } \\ { v _ { 2 } } \\ { \operatorname* { m a x } ( 0 , v _ { 2 } - v _ { 3 } + 1 ) } \end{array} \right] + \left[ \begin{array} { c } { 0 } \\ { 1 } \\ { 0 } \end{array} \right] } \\ & { } & { = \left[ \begin{array} { c } { 0 } \\ { v _ { 2 } + 1 } \\ { v _ { 3 } + \operatorname* { m a x } ( 0 , v _ { 2 } - v _ { 3 } + 1 ) } \end{array} \right] = \left[ \begin{array} { c } { 0 } \\ { v _ { 2 } + 1 } \\ { \operatorname* { m a x } ( v _ { 3 } , v _ { 2 } + 1 ) } \end{array} \right] . } \end{array}
$$

• For $v _ { 1 } = 1$

$$
\begin{array}{c} f ( v ) = { \binom { 1 } { 0 } } \ 1 \ 0 \ 0 \ 0  \\ { 0 } & { 0 } & { 1 } \ 1 ^ { 1 }  \end{array} \left[ \begin{array} { l } { 1 } \\ { 0 } \\ { v _ { 3 } } \\ { 0 } \end{array} \right] + { \binom { 0 } { 1 } } = { \left[ \begin{array} { l } { 1 } \\ { 0 + 1 } \\ { v _ { 3 } + 0 } \end{array} \right] } = { \left[ \begin{array} { l } { 1 } \\ { 1 } \\ { v _ { 3 } } \end{array} \right] }
$$

Since $v _ { 3 } \in \mathbb { Z } ^ { + } \implies v _ { 3 } \geq 1$ , we can substitute $v _ { 3 } = \operatorname* { m a x } ( v _ { 3 } , 1 )$ , yielding $[ 1 \parallel 1 \parallel \operatorname* { m a x } ( v _ { 3 } , 1 ) ] ^ { \top }$

Note that, the first component of $f ( v )$ can be even assigned to 0, when $v _ { 1 } = 1$ , by changing $W _ { 2 } [ 0 , 0 ] = 0 .$ □

Theorem 13. A three-layer single-head Transformer decoder can find the width of a tree wd $\left( G _ { T } \right)$ during the simulation ofDkst in exactly $n - 1$ steps, where n denotes the number ofvertices in $G _ { T }$

Proof. To accommodate the computation of $\mathrm { w d } ( G _ { T } )$ , we augment the vectors $\tilde { x } _ { 0 } , \ldots , \tilde { x } _ { n - 1 } , \tilde { y } _ { 0 } , \ldots , \tilde { y } _ { t }$ as shown in Theorem 6 with three scalar coordinates such that

$$
\boldsymbol { x } _ { i } = [ \tilde { \boldsymbol { x } } _ { i } ^ { \top } \parallel 0 \parallel 0 \parallel 0 ] ^ { \top } \mathrm { ~ a n d ~ } \boldsymbol { y } _ { 0 } = [ \tilde { \boldsymbol { y } } _ { 0 } ^ { \top } \parallel 0 \parallel 1 \parallel 1 ] ^ { \top }
$$

$\mathbf { \Lambda } \in \mathbb { R } ^ { d } .$ , where $d = 5 n + 4 $ . The appended positions store, in order, the difference in distances $\pi _ { c r d } ( \tilde { y } _ { t } )$ and $\pi _ { c r d } ( \tilde { y } _ { t - 1 } )$ maps $t w d ^ { ( t ) }$ and $ m w d ^ { ( t ) }$ . We will use the selectors $\pi _ { d i f } , \pi _ { t w d }$ , and $\pi _ { m w d } : \mathbb { R } ^ { d }  \mathbb { R }$ to select these positions along with reusing the previous selectors as in Theorem 6 as and when necessary.

The two layers of Theorem 6 remain identical except for $W _ { 2 } ^ { ( 2 ) }$ ; the dimensional extension does not affect the previous computation once the parameters are adapted accordingly. In addition to the designated changes $W _ { 2 } ^ { ( 2 ) }$ does in $y _ { t } ^ { ( 1 ) ^ { \prime } }$ (as in $\tilde { y } _ { t } ^ { ( 1 ) ^ { \prime } } )$ , it will set $\begin{array} { r } { \pi _ { d i f } ( y _ { t } ^ { ( 1 ) ^ { \prime } } ) \operatorname { t o } \sum _ { i = 0 } ^ { n - 1 } \pi _ { b u f } ( y _ { t } ^ { ( 1 ) ^ { \prime } } ) [ i ] - \pi _ { c r d } ( y _ { t } ^ { ( 1 ) ^ { \prime } } ) } \end{array}$ to obtain $y _ { t } ^ { ( 2 ) }$ . Note that, $\pi _ { c r d } ( y _ { t } ^ { ( 1 ) ^ { \prime } } ) = \pi _ { c r d } ( y _ { t } )$

The third layer contains a trivial attention layer that takes $W _ { O } ^ { ( 3 ) }$ as empty. Since the tree $G _ { T }$ is (un)weighted taking weights in $\{ 1 , \infty \}$ , and as we have already observed $\pi _ { d i f } \in \{ 0 , 1 \}$ } in Observation 1, the subsequent FFN layer can be developed analogus to Lemma 12 considering $V _ { m a x } = n$ . Consider this as the resultant vector $y _ { t + 1 }$ . Clearly, $\pi _ { m w d } ( y _ { n - 1 } ) = \mathrm { w d } ( G _ { T } )$ □

Lemma 14. Let $v \in \mathbb { R } ^ { H + 2 }$ be a vector defined as the concatenation $v = [ m \parallel h \parallel p _ { 1 } \parallel p _ { 2 } \parallel \cdots \parallel p _ { H } ] ^ { \intercal }$ , where $H > h \geq 1$ is a strictly bounding integer constant, $p _ { i } \ge 1 f o r a l l i \in \{ 1 , . . . , \stackrel { . } { H } \}$ , and $m \in \{ - 1 , 1 \}$ . There exists an FFN with ReLU activation, such that:

$$
v ^ { \prime } = { \left\{ \begin{array} { l l } { [ m \parallel h + 1 \parallel p _ { 1 } \parallel \cdots \parallel p _ { h + 1 } + 1 \parallel \cdots \parallel p _ { H } ] ^ { \top } } & { i f m = 1 } \\ { \left[ m \parallel h - 1 \parallel p _ { 1 } \parallel p _ { 2 } \parallel \cdots \parallel p _ { H } \right] ^ { \top } } & { i f m = - 1 } \end{array} \right. }\tag{9}
$$

Proof. Let the input dimension be $d _ { i n } = H + 2$ . We construct a hidden layer of dimension $d _ { h } = 4 H$ to evaluate the conditional mapping. We partition the hidden dimension into two functional blocks: state preservation mappings and discrete pulse functions. The weight matrices $W _ { 1 } \in \mathbb { R } ^ { 4 H \times ( H + 2 ) }$ and $W _ { 2 } \in \mathbb { R } ^ { ( H + 2 ) \times \hat { 4 } H }$ are constructed by concatenating their respective sub-matrices.

Since $m \in \{ - 1 , 1 \}$ , we map it to the positive domain using two nodes: max $\mathbf { \Omega } _ { : ( 0 , m ) }$ and $\operatorname* { m a x } ( 0 , - m )$ . The state preservation sub-matrix $\boldsymbol { W _ { \mathrm { 1 , i d } } } \in \mathbb { R } ^ { ( H + 3 ) ^ { \cdot } \times ( H + 2 ) }$ is defined as:

$$
W _ { \mathrm { 1 , i d } } = \left( \begin{array} { c c c } { { 1 } } & { { 0 } } & { { { \bf 0 } _ { \mathrm { 1 } \times H } } } \\ { { - 1 } } & { { 0 } } & { { { \bf 0 } _ { \mathrm { 1 } \times H } } } \\ { { 0 } } & { { 1 } } & { { { \bf 0 } _ { \mathrm { 1 } \times H } } } \\ { { { \bf 0 } _ { H \times 1 } } } & { { { \bf 0 } _ { H \times 1 } } } & { { I _ { H } } } \end{array} \right)
$$

where $I _ { H }$ is the identity matrix. Since $h \geq 1$ and $p _ { i } \geq 1$ , the ReLU acts as the identity function for all these dimensions. To selectively increment $p _ { h + 1 }$ when $m = 1$ , we construct an indicator for each $k \in \{ 1 , \ldots , H - 1 \}$ that evaluates to 1 when $m = 1 \land h = k ,$ and 0 otherwise. This requires 3 nodes per index k to form a piecewise linear basis. We stack $H - 1$ blocks of size $3 \times ( H + 2 )$ to form $W _ { \mathrm { 1 , p l s } }$ . The k-th block is:

$$
\begin{array} { r } { W _ { \mathrm { 1 , p l s } } ^ { ( k ) } = \binom { - ( k - 1 ) } { - k } ~ 1 ~ \mathbf { 0 } _ { 1 \times H } } \\ { - ( k + 1 ) ~ 1 ~ \mathbf { 0 } _ { 1 \times H } } \end{array}
$$

The second layer matrix $W _ { 2 } = [ W _ { \mathrm { 2 , i d } } \quad \parallel \quad W _ { \mathrm { 2 , p l s } } ]$ linearly combines the hidden representations to yield $v ^ { \prime } .$ First, $W _ { \mathrm { 2 , i d } } \in \mathbb { R } ^ { ( H + 2 ) \times ( H + 3 ) }$ recovers the unmodified states and structurally applies the m increment to h:

$$
W _ { \mathrm { 2 , i d } } = \left( \begin{array} { c c c c } { { 1 } } & { { - 1 } } & { { 0 } } & { { { \bf 0 } _ { 1 \times H } } } \\ { { 1 } } & { { - 1 } } & { { 1 } } & { { { \bf 0 } _ { 1 \times H } } } \\ { { { \bf 0 } _ { H \times 1 } } } & { { { \bf 0 } _ { H \times 1 } } } & { { { \bf 0 } _ { H \times 1 } } } & { { I _ { H } } } \end{array} \right)
$$

Finally, $W _ { \mathrm { 2 , p l s } } ~ \in ~ \mathbb { R } ^ { ( H + 2 ) \times 3 ( H - 1 ) }$ routes the linear coefficients $[ 1 , - 2 , 1 ]$ to the corresponding coordinates for $p _ { 2 }$ through $p _ { H }$ (noting $p _ { 1 }$ is never modified since $h \geq 1 \implies h + 1 \geq 2 )$

$$
W _ { 2 , \mathrm { p l s } } = \left( \begin{array} { c c c c c } { \mathbf { 0 } _ { 3 \times 3 } } & { \mathbf { 0 } _ { 3 \times 3 } } & { . . . } & { \mathbf { 0 } _ { 3 \times 3 } } \\ { \left[ 1 \mid \mathbf { - } 2 \mid 1 \right] } & { \mathbf { 0 } _ { 1 \times 3 } } & { . . . } & { \mathbf { 0 } _ { 1 \times 3 } } \\ { \mathbf { 0 } _ { 1 \times 3 } } & { \left[ 1 \mid \mathbf { - } 2 \mid 1 \right] } & { . . . } & { \mathbf { 0 } _ { 1 \times 3 } } \\ { \vdots } & { \vdots } & { \ddots } & { \vdots } \\ { \mathbf { 0 } _ { 1 \times 3 } } & { \mathbf { 0 } _ { 1 \times 3 } } & { . . . } & { \left[ 1 \mid \mathbf { - } 2 \mid 1 \right] } \end{array} \right)
$$

By construction, if $m = 1$ , the k-th discrete pulse evaluates to max $( 0 , h - k + 1 ) - 2 \operatorname* { m a x } ( 0 , h - k ) + \operatorname* { m a x } ( 0 , h - k - 1 )$ Because h is integral, this yields exactly 1 at $h = k$ and 0 otherwise. Conversely, if $m = - 1$ , the arguments to the ReLUs become $\bar { h + k } - 1 , \bar { h + k } .$ , and $h + k + 1$ . Because $h \geq 1$ and $k \geq 1$ , we are guaranteed that $h + \bar { k } - 1 \geq 1 > 0$ Consequently, all arguments are strictly positive, placing the ReLUs purely in their linear regime. The combination perfectly collapses to $( h + k - 1 ) - \dot { 2 } ( \bar { h } + k ) + \bar { ( h + k + 1 ) } = 0$ . Thus, no pointer $p _ { i }$ is affected when $m = - 1$ satisfying the lemma. 口

Lemma 15. Let $v \in \mathbb { R } ^ { H + 1 }$ be a vector defined as the concatenation $v = [ p _ { 1 } \parallel p _ { 2 } \parallel \cdots \parallel p _ { H } \parallel w d ] ^ { \top }$ , where wd $\geq 0$ and $p _ { i } \geq 0 f o r$ all $i \in \{ 1 , \ldots , H \}$ . Assume that $p _ { i } \leq w d + 1 $ for all i, and that $p _ { i } = w d + 1$ for at most one index i. There exists an FFN with ReLU activation, such that:

$$
v ^ { \prime } = { \left\{ p _ { 1 } \parallel p _ { 2 } \parallel \cdots \parallel p _ { H } \parallel w d + 1 \right\} } ^ { \intercal } \quad i f \exists i \ s . t . \ p _ { i } = w d + 1\tag{10}
$$

Proof. Let the input dimension be $d _ { i n } = H + 1$ . We construct a hidden layer of dimension $d _ { h } = 2 H + 1$ . We partition the hidden layer into two sets of nodes: an identity mapping to preserve the input state, and a set of H temporary indicator nodes to detect the increment condition.

The first weight matrix $W _ { 1 } \in \mathbb { R } ^ { ( 2 H + 1 ) \times ( H + 1 ) }$ is constructed as a block matrix:

$$
W _ { 1 } = \binom { I _ { H + 1 } } { I _ { H } - \mathbf { 1 } _ { H } }
$$

where $I _ { k }$ denotes the $k \times k$ identity matrix, and $\mathbf { 1 } _ { H } \in \mathbb { R } ^ { H \times 1 }$ is the all-ones column vector.

Applying the ReLU activation yields the hidden state $h = \operatorname* { m a x } ( 0 , W _ { 1 } v ) \in \mathbb { R } ^ { 2 H + 1 }$ . Because $p _ { i } \geq 0$ and wd $\geq 0$ the first $H + 1$ dimensions pass through the ReLU unchanged, preserving the original vector v. The remaining H dimensions compute the temporary variables ${ \mathrm { : } } m p _ { i } = \operatorname* { m a x } ( { \bar { 0 , } } p _ { i } - w d ) { \mathrm { ~ f o r ~ } } { \bar { i } } \in \{ 1 , \ldots , H \}$

Given the constraint $p _ { i } \leq w d + 1$ , the difference $p _ { i } - w d$ is bounded above by 1. Consequently, the temporary variables evaluate to:

$$
t m p _ { i } = \left\{ { \begin{array} { l l } { 1 } & { { \mathrm { i f } } p _ { i } = w d + 1 } \\ { 0 } & { { \mathrm { i f } } p _ { i } \leq w d } \end{array} } \right.
$$

Since $p _ { i } = w d + 1 $ holds for at most one index, the vector $t m p = [ t m p _ { 1 } \parallel \cdot \cdot \cdot \parallel t m p _ { H } ] ^ { \intercal }$ is guaranteed to be either a one-hot vector or the zero vector.

The second weight matrix $W _ { 2 } \in \mathbb { R } ^ { ( H + 1 ) \times ( 2 H + 1 ) }$ reconstructs the output dimension and aggregates the temporary variables into the wd coordinate:

$$
W _ { 2 } = \left( \begin{array} { c c c } { { I _ { H } } } & { { { \bf 0 } _ { H \times 1 } } } & { { { \bf 0 } _ { H \times H } } } \\ { { { \bf 0 } _ { 1 \times H } } } & { { 1 } } & { { { \bf 1 } _ { H } ^ { \top } } } \end{array} \right)
$$

Multiplying h by $W _ { 2 }$ leaves the components $p _ { 1 } \ldots p _ { H }$ strictly isolated and unmodified. The final coordinate is computed as $\begin{array} { r } { w d ^ { \prime } = w d + \sum _ { i = 1 } ^ { H } t m p _ { i } } \end{array}$ . Because tmp contains at most a single 1 corresponding to the condition $p _ { i } = w d + 1$ , the sum evaluates to 1 if the condition is met, and 0 otherwise. □

Theorem 16. A two-layer single-head Transformer decoder canfind wd $. ( \psi ( w ) )$ for a Dyck word w in exactly n CoT steps, where $| w | = n$

Proof. Given a Dyck word w of length $n , \operatorname { h t } ( \psi ( w ) )$ is bounded above by ${ \frac { n } { 2 } } + 1$ , that is, when $\psi ( w )$ is a path. The $t ^ { \mathrm { { t h } } }$ decoding step records the width $\mathrm { h t } ( \psi ( w ) )$ of the partial tree obtained after reading the prefix $w _ { 0 } \cdot \cdot \cdot w _ { t - 1 }$ , equivalently the lattice path $\{ ( \tau , s _ { \tau } ) \} _ { \tau = 0 } ^ { t }$ . The embedding dimension is $\begin{array} { r } { d = \frac { 3 n } { 2 } + 4 . } \end{array}$ , split into five blocks: the position $p o s \in \mathbb { R } ^ { n }$ in one-hot form; the next position is a scalar holding the step $q _ { t } \in \{ - 1 , + 1 \}$ on an input token; the next is another scaler storing the cumulative sum of all $q _ { \tau }$ see till the current CoT step. From the remaining block of length $\frac { n } { 2 } + 1 , p _ { i }$ stores $| \{ \tau \mid s _ { \tau } = i$ and $g _ { \tau } = s _ { \tau } - s _ { \tau - 1 } = + 1 \}$ | for $0 \leq \tau \leq t$ and $i \in \{ 1 , \ldots , \frac { n } { 2 } \}$ . And the final block wd is a scalar that stores the maximum of all $p _ { i }$ , which turns out to be the width, as claimed in Proposition 3.

Assume $\mathcal { E } _ { n } = \{ e _ { 0 } , . . . , e _ { n - 1 } \}$ denotes the standard basis vector on $\mathbb { R } ^ { n }$ . The input tokens $x _ { 0 } , \ldots , x _ { n - 1 } \in R ^ { d }$ encodes the lattice path $\{ \tau , s _ { \tau } \} _ { \tau = 0 } ^ { n }$ as follows:

$$
\begin{array} { r } { \boldsymbol { x } _ { \tau } = [ e _ { \tau } \parallel \boldsymbol { q } _ { \tau } \parallel 0 \parallel 0 ^ { \frac { n } { 2 } + 1 } \parallel 0 ] ^ { \top } . } \end{array}
$$

The Dynamic is carried out by tokens $y _ { t } \in \mathbb R ^ { d }$ with

$$
y _ { 0 } = [ e _ { 0 } \parallel 0 \parallel 0 \parallel 0 \perp + 1 \parallel 0 ] ^ { \top } .
$$

The selectors $\pi _ { p o s } : \mathbb { R } ^ { d }  \mathbb { R } ^ { n }$ selects the first one-hot block that corresponds to the current position, $\pi _ { q } : \mathbb { R } ^ { d } $ $\{ - 1 , 0 , 1 \}$ selects the block that records the corresponding move taken at the position for input tokens or 0 for generated tokens, $\pi _ { h t } : \mathbb { R } ^ { d }  \mathbb { R } _ { > 0 }$ records the height, $\pi _ { p _ { i } } : \mathbb { R } ^ { d }  \mathbb { R } _ { \geq 0 }$ records $p _ { i }$ as discussed above. Finally, $\pi _ { p o s } : \mathbb { R } ^ { d }  \mathbb { R } _ { > 0 }$ records the wd. Analogous to Theorem 11, the attention mechanism of the first layer employs a shift matrix $S$ to find the $\pi _ { p o s }$ and records the corresponding $\pi _ { q }$ from the input tokens. Let $\alpha ^ { \prime }$ be the vector after multiplication by $W _ { O } ^ { ( 1 ) }$ where every block is 0 except $\pi _ { q } ( \alpha ^ { \prime } ) = q _ { \tau }$ . After the residual connection with $y _ { t }$ , let this vector be $\tilde { y } _ { t }$ . We employ $\bar { W } _ { 1 } ^ { ( 1 ) }$ and $W _ { 2 } ^ { ( 1 ) }$ of the following FFN layer similar to Lemma 14 so that $[ \pi _ { \boldsymbol { q } } ( \tilde { y } _ { t } ) \parallel \pi _ { h t } ( \tilde { y } _ { t } ) \parallel \pi _ { p _ { 1 } } ( \tilde { y } _ { t } ) \parallel \cdot \cdot \cdot \parallel \pi _ { \frac { n } { 2 } + 1 } ( \tilde { y } _ { t } ) ] ^ { \intercal }$ serves the vector v and undergoes an update as in (9). The remaining positions remain unchanged.

Note that the above operation would impact only one position of all $p _ { 1 } , \ldots , p _ { \frac { n } { 2 } + 1 }$ . For the second layer, we will utilize the FFN only. We will consider we $w _ { O } ^ { ( 1 ) }$ as 0. Let the vector obtained after applying the residual connection be $\tilde { y } _ { t } ^ { ( 1 ) }$ . The FFN layer performs updates on $[ \overline { { \pi } } _ { p _ { 1 } } ( \tilde { y } _ { t } ^ { ( 1 ) } ) \parallel \cdots \parallel \pi _ { \frac { n } { 2 } + 1 } ( \tilde { y } _ { t } ^ { ( 1 ) } ) \parallel w d ] ^ { \top }$ similar to (10) as shown in Lemma 15. The linear projections do not affect any other positions of $\tilde { y } _ { t } ^ { ( 1 ) }$ except that $W _ { 2 } ^ { ( 2 ) }$ updates the the block $q \tan 0$ . The resultant vector is $y _ { t + 1 }$ and $\pi _ { w d } ( y _ { n } ) = \mathrm { w d } \bar { ( } \psi ( w ) )$ ). □