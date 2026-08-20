# Enhancing Distance-Based Graph Autoencoders with Structural Penalties for Dynamic Graph Embedding

Aleksandar Tomˇci´c, Miloˇs Savi´c, and Miloˇs Radovanovi´c

Department of Mathematics and Informatics, Faculty of Sciences,

University of Novi Sad, Serbia

Trg Dositeja Obradovi´ca 4, 21000 Novi Sad, Serbia

{atomic, svc, radacha}@dmi.uns.ac.rs

Abstract. Graph autoencoders (GAEs) are widely used for learning representations of dynamic graphs. However, their optimisation objectives typically do not take structural heterogeneity across nodes into account. We propose three distance-based GAE variants that incorporate structural penalties into the reconstruction loss. All variants share a twolayer Graph Convolutional Network encoder and a Euclidean-distance decoder trained with distance-based reconstruction objectives. We extend sparsity-corrected loss with two node-level regularization terms: (i) a hub penalty based on degree centrality, and (ii) a penalty based on Natural Community Local Intrinsic Dimensionality (NC-LID). The paper is motivated by prior evidence linking high NC-LID to reduced embedding quality. The proposed methods are designed to emphasize reconstruction errors for structurally ambiguous nodes. Experiments on multiple dynamic graph data sets show that incorporating NC-LID-based regularization consistently improves reconstruction performance over the baseline without structural regularization and the method using hub-aware regularization. These findings highlight NC-LID as a useful structural signal for enhancing distance-based graph autoencoders in dynamic settings.

Keywords: Dynamic graph embedding, Graph autoencoder, Hubness, Local intrinsic dimensionality (LID), Euclidean distance decoder, Structural regularization

## 1 Introduction

Many real-world complex systems are inherently dynamic. The networks that represent them evolve over time through the addition or removal of nodes and edges [3]. Learning useful representations of such evolving structures is a core challenge in machine learning on graphs [1]. Graph embedding methods ofer a task-agnostic solution, as they map nodes to low dimensional vectors that preserve structural properties, and can subsequently be used for a wide range of downstream tasks including link prediction, node classification, and anomaly detection [4].

Among dynamic graph embedding approaches, two main families exist. Methods based on random walks [9, 10, 12] generate node contexts by traversing the graph and learn embeddings from the resulting sequences, following the paradigm of static walk-based methods such as node2vec [2] and DeepWalk [13]. Methods based on autoencoders [1] encode the graph structure directly through a neural network and reconstruct it from a low-dimensional bottleneck. While walk-based methods have received more research attention, autoencoder-based methods remain comparatively underexplored in terms of structural signals that could improve their training objectives.

Recent lines of work are particularly relevant here. First, it has been shown that hub-awareness, incorporating node degree centrality into walk transition probabilities, substantially improves dynamic graph embedding [18]. Second, NC-LID, a graph-adapted measure of local intrinsic dimensionality [15, 16], has been shown to negatively correlate with node embedding quality in dynamic graphs. Nodes with high NC-LID (complex, irregularly-shaped natural communities) tend to be systematically underrepresented in walk-based dynamic embeddings [7]. Both findings point to structural properties that the standard training objectives of autoencoder-based methods do not take into account.

This paper investigates whether these structural signals can be incorporated directly into the loss function of a distance-based graph autoencoder. We make the following contributions:

– We design a Euclidean distance decoder for dynamic graph autoencoders that directly aligns the training objective with the Euclidean distance-based graph reconstruction criterion used in evaluation, correcting the geometric mismatch present in the standard inner product decoder.

– We introduce two structurally penalised loss functions, one based on the degree outer product (hub penalty) and one based on the NC-LID outer product (LID penalty), each amplifying reconstruction errors for structurally significant node pairs.

We provide a systematic empirical evaluation on nine dynamic networks, showing that the LID-penalised variant achieves better reconstruction F<sub>1</sub> score in six of nine datasets while adding negligible computational overhead through a precomputation strategy for NC-LID scores.

## 2 Related Work

Surveys by Barros et al. [1] and Khoshraftar and An [4] provide comprehensive overviews of dynamic graph embedding methods. Among walk-based methods, Dynnode2vec [9] initialises embeddings via node2vec [2] and incrementally updates only nodes whose ego networks change; CTDNE [10] enforces temporal ordering within walks, and STWalk [12] combines spatial and historical walks. Autoencoder-based methods are surveyed in [1], but remain comparatively underexplored in terms of structural inductive biases.

In the static setting, hub-aware random walk methods [17] that adjust transition probabilities by node degree improve node classification performance; Deep-Hub [18] extends this to the dynamic setting, showing that inverse-degree-biased walks outperform Dynnode2vec on graph reconstruction while narrowing the hub/non-hub embedding quality gap.

Recent work [15, 16] introduced NC-LID, a local intrinsic dimensionality measure defined over natural communities [8]: low NC-LID indicates compact, welldefined communities, while high NC-LID indicates structurally ambiguous nodes spanning multiple structural directions. LID-elastic node2vec [16] improved both intrinsic embedding quality and downstream performance in the static setting, and [7] adapted NC-LID to the dynamic setting, showing significant negative Spearman correlations between NC-LID and embedding $F _ { 1 }$ under Dynnode2vec, establishing NC-LID as a reliable indicator of weakly embedded nodes.

Kipf and Welling [6] introduced the GAE framework, combining GCN encoders [5] with inner product decoders. This decoder has become standard, but its implicit assumption, that co-directional vectors correspond to connected nodes, is inconsistent with Euclidean distance-based evaluation. Distance-based decoders have been explored for hyperbolic embeddings [11] and link prediction [19], but not, to our knowledge, systematically applied to dynamic graph autoencoders with structurally informed losses.

## 3 Proposed Methods

## 3.1 Background: Graph Autoencoders and GCNs

An autoencoder is a neural network trained to reconstruct its own input through a low-dimensional bottleneck. The encoder $f _ { \theta } : \mathcal { X }  \mathcal { Z }$ maps an input $x \in \mathcal { X }$ to a latent code $z \in { \mathcal { Z } }$ , and the decoder $g _ { \phi } : \mathcal { Z }  \mathcal { X }$ attempts to recover x from z:

$$
\operatorname* { m i n } _ { \theta , \phi } \ \mathbb { E } _ { x \sim p ( x ) } [ \mathcal { L } ( x , g _ { \phi } ( f _ { \theta } ( x ) ) ) ] .\tag{1}
$$

Since $| \mathcal { Z } | \ll | \mathcal { X } |$ , the model must discard noise and retain only the most salient structure of the input.

A Graph Autoencoder (GAE) [6] instantiates this with a GCN encoder [5]. At layer ℓ, node representations are updated by:

$$
\mathbf { H } ^ { ( \ell ) } = \sigma \Big ( \tilde { \mathbf { A } } \mathbf { H } ^ { ( \ell - 1 ) } \mathbf { W } ^ { ( \ell ) } \Big ) ,\tag{2}
$$

where $\tilde { \mathbf { A } } = \hat { \mathbf { D } } ^ { - 1 / 2 } ( \mathbf { A } + \mathbf { I } ) \hat { \mathbf { D } } ^ { - 1 / 2 }$ is the symmetrically normalised adjacency with self-loops and $\mathbf { W } ^ { ( \ell ) }$ is a learnable weight matrix. After L layers, the embedding of node $v _ { i }$ integrates the structure of all nodes within graph distance $L , \mathrm { e . g . }$ ., two layers encode both immediate and two-hop neighbourhood structure.

## 3.2 Dynamic Graph Setting

Let $\mathcal { G } = \{ G _ { 1 } , . . . , G _ { T } \}$ denote a discrete-time dynamic graph, where each snapshot $G _ { t } = ( V _ { t } , E _ { t } )$ is a static graph on $n _ { t } = | V _ { t } |$ nodes, and let $\textstyle \mathcal { V } = \bigcup _ { t } V _ { t }$ be the global vocabulary of $N = | \nu |$ distinct nodes. Since these datasets carry no semantic node attributes, each node is represented by a one-hot identity vector keyed by its global vocabulary index, giving feature matrix $\mathbf { X } ^ { ( t ) } \in \{ 0 , 1 \} ^ { n _ { t } \times N }$ with $X _ { i , \ i \mathbf { d x } ( v _ { i } ) } ^ { ( t ) } = 1$ elsewhere zero. This global vocabulary keeps the encoder input dimension consistent across snapshots without padding or masking, regardless of node dynamics.

## 3.3 Euclidean Distance Decoder

Classical GAEs decode via the inner product $\hat { A } _ { i j } = \sigma ( \mathbf { z } _ { i } ^ { \top } \mathbf { z } _ { j } )$ , assigning high scores to co-directional rather than geometrically close vectors. Since graph reconstruction evaluation connects the $| E _ { t } |$ closest node pairs by Euclidean distance [18, 7], training with an inner product decoder mismatches the objective and the evaluation metric.

Our EuclideanGAEModel replaces the inner product decoder with one based on pairwise squared Euclidean distances. The encoder uses tanh in the hidden layer to bound embedding coordinates and prevent distance explosion:

$$
\mathbf { H } = \operatorname { t a n h } \left( \tilde { \mathbf { A } } \mathbf { X } \mathbf { W } ^ { ( 1 ) } \right) ,\tag{3}
$$

$$
\begin{array} { r } { \mathbf { Z } = \tilde { \mathbf { A } } \mathbf { H } \mathbf { W } ^ { ( 2 ) } , } \end{array}\tag{4}
$$

with ${ \bf W } ^ { ( 1 ) } \in \mathbb { R } ^ { N \times d _ { h } } , { \bf W } ^ { ( 2 ) } \in \mathbb { R } ^ { d _ { h } \times d }$ , and hidden dimension $d _ { h } = 2 d$ . The decoder forms the full pairwise distance matrix using the numerically stable expansion:

$$
\begin{array} { r } { D _ { i j } = \operatorname* { m a x } \ ( 0 , \ s _ { i } + s _ { j } - 2 [ { \mathbf Z } { \mathbf Z } ^ { \top } ] _ { i j } ) , \quad s _ { i } = \sum _ { k } Z _ { i k } ^ { 2 } , } \end{array}\tag{5}
$$

and converts distances to adjacency logits via a learnable scalar bias b (initialised to zero):

$$
\hat { A } _ { i j } = b - D _ { i j } .\tag{6}
$$

Node pairs with $D _ { i j } ~ < ~ b$ receive positive logits (edge predicted); pairs with $D _ { i j } ~ > ~ b$ receive negative logits (no edge predicted). The bias adapts to the average connectivity of each snapshot. The computational cost of the decoder is $O ( n ^ { 2 } d )$ , identical to the inner product decoder, since the dominant operation is $\mathbf { Z } \mathbf { Z } ^ { \top }$

## 3.4 Sparsity-Corrected Baseline Loss

Real-world networks are highly sparse, the edge density $| E _ { t } | / n _ { t } ^ { 2 }$ is below 1% for most datasets studied here. A na¨ıve binary cross entropy (BCE) loss is dominated by the abundant negative pairs, biasing the model toward predicting no edges.

Following standard practice, we up-weight positive pairs by the class imbalance ratio:

$$
w ^ { + } = \frac { n ^ { 2 } - | E | } { | E | + \varepsilon } , \qquad \varepsilon = 1 0 ^ { - 9 } ,\tag{7}
$$

yielding the sparsity corrected baseline loss used by Distance Basic variant:

$$
\mathcal { L } _ { \mathrm { b a s e } } = \frac { 1 } { n ^ { 2 } } \sum _ { i , j } w _ { i j } \ \ell _ { \mathrm { B C E } } \Big ( \hat { A } _ { i j } , A _ { i j } \Big ) , \quad w _ { i j } = \left\{ \begin{array} { l l } { w ^ { + } } & { A _ { i j } = 1 } \\ { 1 } & { A _ { i j } = 0 . } \end{array} \right.\tag{8}
$$

where $\ell _ { \mathrm { B C E } } ( \hat { a } , a ) = - a \log \sigma ( \hat { a } ) - ( 1 - a ) \log ( 1 - \sigma ( \hat { a } ) )$ and σ is the sigmoid function.

## 3.5 Hub-Penalty Variant

High-degree nodes (hubs) occupy structurally critical positions [14, 17, 18], acting as bridges between communities and appearing disproportionately in shortest paths, so incorrectly placing a hub propagates reconstruction error to all deg(v) incident pairs. The Distance Hub variant amplifies the reconstruction loss for hub-involved pairs in proportion to the product of their degrees:

$$
\delta _ { i j } = \deg ( v _ { i } ) \cdot \deg ( v _ { j } ) , \qquad \tilde { \delta } _ { i j } = \frac { \delta _ { i j } } { \operatorname* { m a x } _ { k , l } \delta _ { k l } + \varepsilon } .\tag{9}
$$

$$
\mathcal { L } _ { \mathrm { h u b } } = \frac { 1 } { n ^ { 2 } } \sum _ { i , j } w _ { i j } \ell _ { \mathrm { B C E } } \Big ( \hat { A } _ { i j } , A _ { i j } \Big ) \cdot \big ( 1 + \alpha _ { h } \tilde { \delta } _ { i j } \big ) ,\tag{10}
$$

where $\alpha _ { h } \ \geq \ 0$ controls penalty strength (set to 1.0 in all experiments). The multiplier $( 1 + \alpha _ { h } \tilde { \delta } _ { i j } ) \in [ 1 , 1 + \alpha _ { h } ]$ ensures the baseline loss is only amplified, never suppressed.

## 3.6 LID-Penalty Variant

The Distance LID variant replaces the degree-based signal with NC-LID, a topological measure of structural ambiguity. Following work presented in [16], the NC-LID of a node v is defined over its natural community S (recovered by the fitness-based algorithm [8] with v as seed) and the largest shortest-path distance k from v to any node in S:

$$
\mathrm { N C - L I D } ( v ) = - \ln \left( \frac { | S | } { D ( v , k ) } \right) ,\tag{11}
$$

where $D ( v , k )$ is the number of nodes at shortest-path distance at most k from v. A value of zero indicates a perfectly compact community; higher values indicate that the shortest-path distance struggles to separate S from the rest of the graph, reflecting structural ambiguity. As shown in [7], NC-LID correlates negatively with embedding quality in dynamic graphs for the majority of datasets.

Table 1. Experimental datasets. Max nodes and max edges refer to the largest individual snapshot.
<table><tr><td>Dataset</td><td>Res.</td><td>Snaps.</td><td>Max nodes</td><td>Max edges</td></tr><tr><td>radoslaw-email</td><td>month</td><td>9</td><td>151</td><td>1,675</td></tr><tr><td>ia-hospital-ward-proximity-attr</td><td>day</td><td>5</td><td>54</td><td>492</td></tr><tr><td>ia-contacts_hypertext2009</td><td>day</td><td>3</td><td>102</td><td>1,062</td></tr><tr><td>ia-enron-employees</td><td>3-months</td><td>13</td><td>140</td><td>823</td></tr><tr><td>ia-primary-school-proximity-attr</td><td>day</td><td>2</td><td>241</td><td>5,923</td></tr><tr><td>fb-forum</td><td>month</td><td>6</td><td>815</td><td>5,838</td></tr><tr><td>email-Eu-core-temporal</td><td>month</td><td>18</td><td>762</td><td>4,518</td></tr><tr><td>CollegeMsg</td><td>month</td><td>7</td><td>1,371</td><td>6,865</td></tr><tr><td>fb-messages</td><td>month</td><td>7</td><td>1,394</td><td>9,425</td></tr></table>

Amplifying the reconstruction penalty for boundary nodes (high NC-LID on both endpoints) forces the encoder to resolve their positions precisely rather than collapsing them into the interior of the nearest dominant community. The loss is:

$$
\lambda _ { i j } = \mathrm { N C - L I D } ( v _ { i } ) \cdot \mathrm { N C - L I D } ( v _ { j } ) , \qquad \tilde { \lambda } _ { i j } = \frac { \lambda _ { i j } } { \operatorname* { m a x } _ { k , l } \lambda _ { k l } + \varepsilon } ,\tag{12}
$$

$$
\mathcal { L } _ { \mathrm { l i d } } = \frac { 1 } { n ^ { 2 } } \sum _ { i , j } w _ { i j } \ell _ { \mathrm { B C E } } \Big ( \hat { A } _ { i j } , A _ { i j } \Big ) \cdot \underbrace { \big ( 1 + \alpha _ { \ell } \tilde { \lambda } _ { i j } \big ) } _ { \mathrm { L I D ~ m u l t i p l i e r } } ,\tag{13}
$$

with $\alpha _ { \ell } = 1 . 0$ throughout. The LID multiplier lies in $[ 1 , 1 + \alpha _ { \ell } ]$ , matching the hub variant’s guarantee that the baseline loss is never suppressed.

NC-LID estimation scales as $O ( n \cdot k _ { s } )$ per snapshot $( k _ { s } { \mathrm { : } }$ average natural community size), which would be prohibitive if repeated for every model and dimension. We instead estimate NC-LID once per dataset, cache it as tensors, and load the cached values at training time; the per-epoch loss then adds a single $O ( n ^ { 2 } )$ outer product. This keeps Distance LID’s per-epoch training cost within ±10% of the baseline across all datasets.

The complete training procedure is summarised in Algorithm 1.

## 4 Experimental Setup

The proposed methods are evaluated on nine publicly available temporal networks from $\mathrm { S N A P ^ { 1 } }$ and Network Repository,<sup>2</sup> summarised in Table 1 (institutional email, physical proximity, and online social messaging networks; 54–1,394 nodes per snapshot, 2–18 snapshots).

Efective graph embeddings should allow reconstruction of the original graph: computing the Euclidean distance between every pair of embedding vectors and connecting the closest |E| pairs, where |E| is the number of edges in the original graph. Let n denote an arbitrary node and C the number of correctly reconstructed links incident to n. We use:

Algorithm 1 Dynamic graph embedding with structural loss penalty   
Require: Dynamic graph $\overline { { \mathcal { G } = \{ G _ { 1 } , \dots , G _ { T } \} } }$ , model variant M, embedding dimension   
d, epochs $T _ { e } ,$ learning rate η   
Ensure: Per-snapshot embeddings $\{ \mathbf { Z } ^ { ( 1 ) } , \ldots , \mathbf { Z } ^ { ( T ) } \}$   
1: Build global vocabulary $\nu ;$ set $N \gets | \nu |$   
2: if M = Distance LID then   
3: for t ← 1 to T do   
4: Estimate NC-LID scores for all nodes in $G _ { t }$ via Eq. (11)   
5: Cache as torch.Tensor at index t   
6: end for   
7: end if   
8: for t ← 1 to T do   
9: if $| V _ { t } | < 2$ then   
10: continue   
11: end if   
12: Compute A<sup>˜</sup> <sup>(t)</sup>, X<sup>(t)</sup>, A<sup>(t)</sup>   
13: Initialise EuclideanGAE(N, 2d, d); Adam(η)   
14: if M = Distance LID then   
15: Load cached NC-LID tensor for snapshot t   
16: end if   
17: for epoch ← 1 to $T _ { e }$ do   
18: Forward: $\hat { \mathbf { A } } ^ { ( t ) } , \mathbf { Z } ^ { ( t ) } \gets \mathrm { E }$ uclideanGAE(X<sup>(t)</sup>, A<sup>˜</sup> <sup>(t)</sup>)   
19: L ← compute loss $( \hat { \mathbf { A } } ^ { ( t ) } , \mathbf { A } ^ { ( t ) } , G _ { t } )$ ▷ Eq. (8), (10), or (13)   
20: Backpropagate; update weights   
21: end for   
22: Store $\mathbf { Z } ^ { ( t ) }$   
23: end for

Precision – C divided by the number of links n has in the reconstructed graph.

${ \mathrm { R e c a l l } } - C$ divided by the number of links n has in the original graph.

$- \ F _ { 1 }$ score – The harmonic mean of precision and recall.

Higher values of precision, recall, and $F _ { 1 }$ indicate fewer link reconstruction errors for node n. At the graph level, precision, recall, and $F _ { 1 }$ scores can be obtained by macro-averaging over all nodes. We report mean $F _ { 1 }$ averaged across all temporal snapshots of each dataset.

Five embedding dimensions are evaluated: $d \in \{ 1 0 , 2 5 , 5 0 , 1 0 0 , 2 0 0 \}$ with hidden dimension $d _ { h } = 2 d .$ . All models are trained for $T _ { e } = 3 0 0$ epochs with Adam at $\eta = 0 . 0 1$ . Each model dimension combination is trained from random initialisation without parameter sharing across snapshots. Penalty strengths are fixed at $\alpha _ { h } = \alpha _ { \ell } = 1 . 0$ for all penalised variants.

Table 2. Mean reconstruction $F _ { 1 }$ for radoslaw-email (top) and ia-hospital-wardproximity-attr (bottom).
<table><tr><td> $d$ </td><td>BASIC</td><td>HUB</td><td>LID</td></tr><tr><td>radoslaw-email 10</td><td>0.4597</td><td>0.4565</td><td>0.4777</td></tr><tr><td>25</td><td>0.5567</td><td>0.5408</td><td>0.5633 0.6016</td></tr><tr><td>50 100</td><td>0.5917 0.6099</td><td>0.5763 0.6000</td><td>0.6229*</td></tr><tr><td>200</td><td>0.6045</td><td>0.5933</td><td>0.6124</td></tr><tr><td>0.8166 0.7920 100 0.8155 0.7968</td><td></td><td>ia-hospital-ward-proximity-attr</td><td></td></tr><tr><td>10 25 50</td><td>0.6819 0.7729</td><td>0.6893 0.7628</td><td>0.6932 0.7792</td></tr></table>

## 5 Results and Discussion

## 5.1 Per-Dataset Reconstruction ${ \pmb F } _ { 1 }$

Tables 2–4 report mean reconstruction $F _ { 1 }$ for all nine datasets across the five embedding dimensions. Bold entries mark the best model for each dimensiondataset pair. The symbol ⋆ marks the single best configuration per dataset.

Table 5 condenses the results to the single best $F _ { 1 }$ per model per dataset.

## 5.2 Discussion

Reconstruction $F _ { 1 }$ increases consistently as d grows from 10 to 100 across all models and datasets, then plateaus or declines at d = 200, consistent with a capacity-generalisation trade-of: at $d = 1 0$ the bottleneck is too tight for structurally distinct nodes, while at d = 200 the encoder (N×2d parameters) begins to memorise snapshot-specific pairs rather than learning generalisable codes. The d = 100 optimum holds for seven of nine datasets; the exceptions (fb-forum, email-Eu-core) are the largest, most variable datasets, where extra capacity still helps.

Distance LID wins on six of nine datasets and records the top score in 26 of 45 dimension-dataset configurations, most visibly on communication and forum networks with heterogeneous community structure: radoslaw-email (+1.3 pp), fb-forum (+1.0 pp), and enron-employees (+0.4 pp) at d=100. This matches the NC-LID analysis of [7]: the datasets where NC-LID correlates most negatively with embedding quality under Dynnode2vec are those where the LID penalty helps most, since pairs with high NC-LID on both endpoints receive the largest loss amplification, preventing boundary nodes from collapsing into the interior of the nearest dominant community.

Table 3. Mean reconstruction $F _ { 1 }$ for ia-contacts hypertext2009 (top), ia-enronemployees (middle), and ia-primary-school-proximity-attr (bottom).
<table><tr><td> $d$ </td><td>BASIC</td><td>HUB</td><td>LID</td></tr><tr><td colspan="4">ia-contacts_hypertext2009</td></tr><tr><td>10</td><td>0.5273</td><td>0.5120</td><td>0.5179</td></tr><tr><td>25</td><td>0.6373</td><td>0.6335</td><td>0.6417</td></tr><tr><td>50</td><td>0.6892</td><td>0.6727</td><td>0.6873</td></tr><tr><td>100</td><td>0.7290*</td><td>0.7197</td><td>0.7214</td></tr><tr><td>200</td><td>0.7136</td><td>0.7026</td><td>0.7130</td></tr><tr><td colspan="4">ia-enron-employees</td></tr><tr><td>10 25</td><td>0.7359</td><td>0.7247</td><td>0.7238</td></tr><tr><td>50</td><td>0.7942</td><td>0.7672</td><td>0.7972</td></tr><tr><td>100</td><td>0.8033</td><td>0.7938</td><td>0.8083</td></tr><tr><td>200</td><td>0.8098</td><td>0.7999</td><td>0.8140*</td></tr><tr><td></td><td>0.8084</td><td>0.7945</td><td>0.8069</td></tr><tr><td colspan="4">ia-primary-school-proximity-attr</td></tr><tr><td>10</td><td>0.7571</td><td>0.7588</td><td>0.7490</td></tr><tr><td>25</td><td>0.7644</td><td>0.7651</td><td>0.7635</td></tr><tr><td>50</td><td>0.7684</td><td>0.7669</td><td>0.7636</td></tr><tr><td>100</td><td>0.7694</td><td>0.7686</td><td>0.7706</td></tr><tr><td>200</td><td>0.7718*</td><td>0.7706</td><td>0.7698</td></tr></table>

On homogeneous proximity networks (primary-school, hospital), NC-LID scores are uniformly low, so the LID multiplier approaches 1 everywhere and the LID loss nears the baseline, explaining the near-parity there and confirming the signal is most valuable under structural ambiguity.

Distance Hub wins zero of nine datasets and only 8 of 45 cells. The failure mode is the long-tailed degree-outer-product distribution on scale-free networks: a handful of extreme hub pairs dominate the gradient, sacrificing non-hub representations and distorting the embedding geometry, echoing [18]’s finding that hub-dominant walks hurt non-hub quality. NC-LID avoids this because its scores are bounded and roughly unimodal. Replacing $\delta _ { i j }$ with $\log ( 1 + \delta _ { i j } )$ would dampen the tails while preserving rank order, as in DeepHub’s logarithmic scaling [18].

CollegeMsg and fb-messages, whose snapshots vary by up to a factor of 12 in node count, show a sharp $F _ { 1 }$ drop at $d = 2 0 0$ not seen elsewhere, plausibly because the one-hot feature matrix becomes extremely sparse for small snapshots when N is large, poorly conditioning $\mathbf { W } ^ { ( 1 ) } \in \mathbb { R } ^ { \breve { N } \times \hat { 2 } d }$ when $n _ { t } \ll N$ . No variant dominates both datasets at $d \ : = \ : 1 0 0$ : Distance Basic remains best on CollegeMsg (0.4641, vs. 0.4482 for Distance LID, −1.59 pp), while Distance LID leads on fb-messages (0.4471 vs. 0.4320, +1.51 pp; Table 4). This points to structural node features (degree, clustering coeficient) as a promising alternative to one-hot initialisation for highly variable-size graphs.

Table 4. Mean reconstruction $F _ { 1 }$ for fb-forum, email-Eu-core-temporal, CollegeMsg, and fb-messages.
<table><tr><td>d</td><td>BASIC</td><td>HuB</td><td>LID</td></tr><tr><td colspan="4">fb-forum</td></tr><tr><td>10</td><td>0.3754</td><td>0.3562</td><td>0.4098</td></tr><tr><td>25</td><td>0.5797</td><td>0.5708</td><td>0.5925</td></tr><tr><td>50</td><td>0.6374</td><td>0.6277</td><td>0.6431</td></tr><tr><td>100</td><td>0.6522</td><td>0.6439</td><td>0.6622</td></tr><tr><td>200</td><td>0.6638</td><td>0.6600</td><td>0.6681*</td></tr><tr><td colspan="4">email-Eu-core-temporal</td></tr><tr><td>10</td><td>0.3462</td><td>0.3387</td><td>0.3436</td></tr><tr><td>25</td><td>0.4364</td><td>0.4336</td><td>0.4415</td></tr><tr><td>50</td><td>0.4654</td><td>0.4598</td><td>0.4663</td></tr><tr><td>100</td><td>0.4773</td><td>0.4680</td><td>0.4789</td></tr><tr><td>200</td><td>0.4912</td><td>0.4767</td><td>0.4914*</td></tr><tr><td colspan="4">CollegeMsg</td></tr><tr><td>10</td><td>0.2062</td><td>0.2231</td><td>0.2242</td></tr><tr><td>25</td><td>0.3476</td><td>0.3785</td><td>0.3736</td></tr><tr><td>50</td><td>0.4350</td><td>0.4261</td><td>0.4375</td></tr><tr><td>100</td><td>0.4641*</td><td>0.4238</td><td>0.4482</td></tr><tr><td>200</td><td>0.3758</td><td>0.3604</td><td>0.3613</td></tr><tr><td colspan="4">fb-messages</td></tr><tr><td>10</td><td>0.1840</td><td>0.2033</td><td>0.2030</td></tr><tr><td>25</td><td>0.3648</td><td>0.3397</td><td>0.3377</td></tr><tr><td>50</td><td>0.4213</td><td>0.4132</td><td>0.4147</td></tr><tr><td>100</td><td>0.4320</td><td>0.4392</td><td>0.4471*</td></tr><tr><td>200</td><td>0.3594</td><td>0.3516</td><td>0.3818</td></tr></table>

## 6 Conclusions

We introduced three distance-based graph autoencoder variants for dynamic graph embedding, sharing a Euclidean distance decoder that aligns the training objective with the evaluation metric. Augmenting the sparsity-corrected BCE loss with an NC-LID-based penalty, which amplifies errors for structurally ambiguous boundary nodes, consistently improves reconstruction $F _ { 1 }$ across nine dynamic networks: Distance LID achieves the best $F _ { 1 }$ in six of nine datasets and wins 26 of 45 dimension-dataset comparisons, confirming NC-LID, previously shown useful for walk-based methods [7], as an equally valuable signal within the autoencoder loss.

Distance Hub consistently underperforms due to a gradient-concentration pathology from the long-tailed degree distribution, illustrating that structural penalties must be bounded to avoid distorting the embedding geometry, a requirement NC-LID satisfies naturally.

Table 5. Best mean reconstruction $F _ { 1 }$ per dataset and model. $d ^ { \star }$ is the optimal dimension. Bold marks the winner per dataset. Win count tallies best $F _ { 1 }$ victories across all 9 datasets.
<table><tr><td rowspan="2">Dataset</td><td colspan="2">BASIC</td><td colspan="2"> $\mathrm { H U B }$ </td><td colspan="2">LID</td></tr><tr><td> $F _ { 1 }$ </td><td> $d ^ { \star }$ </td><td> $F _ { 1 }$ </td><td> $d ^ { \star }$ </td><td> $F _ { 1 }$ </td><td> $d ^ { \star }$ </td></tr><tr><td>radoslaw-email</td><td>0.6099</td><td>100</td><td>0.6000</td><td>100</td><td>0.6229</td><td>100</td></tr><tr><td>hospital-ward</td><td>0.8207</td><td>200</td><td>0.7968</td><td>100</td><td>0.8222</td><td>200</td></tr><tr><td>hypertext2009</td><td>0.7290</td><td>100</td><td>0.7197</td><td>100</td><td>0.7214</td><td>100</td></tr><tr><td>enron-employees</td><td>0.8098</td><td>100</td><td>0.7999</td><td>100</td><td>0.8140</td><td>100</td></tr><tr><td>primary-school</td><td>0.7718</td><td>200</td><td>0.7706</td><td>200</td><td>0.7706</td><td>100</td></tr><tr><td>fb-forum</td><td>0.6638</td><td>200</td><td>0.6600</td><td>200</td><td>0.6681</td><td>200</td></tr><tr><td>email-Eu-core</td><td>0.4912</td><td>200</td><td>0.4767</td><td>200</td><td>0.4914</td><td>200</td></tr><tr><td>CollegeMsg</td><td>0.4641</td><td>100</td><td>0.4238</td><td>100</td><td>0.4482</td><td>100</td></tr><tr><td>fb-messages</td><td>0.4320</td><td>100</td><td>0.4392</td><td>100</td><td>0.4471</td><td>100</td></tr><tr><td>Win count</td><td>3</td><td></td><td>0</td><td></td><td>6</td><td></td></tr></table>

Future work includes (i) extending the LID penalty to walk-based methods, (ii) logarithmic compression of the hub penalty to recover its motivation without the gradient pathology, (iii) replacing one-hot initialisation with structural node features for highly variable-size graphs, (iv) combining LID-aware embeddings with hub-aware hybrid random walkers, and (v) evaluating the proposed variants on downstream tasks such as link prediction.

## Acknowledgements

This research is supported by the Science Fund of the Republic of Serbia, #7462, Graphs in Space and Time: Graph Embeddings for Machine Learning in Complex Dynamical Systems – TIGRA.

## References

1. Barros, C.D., Mendon¸ca, M.R., Vieira, A.B., Ziviani, A.: A survey on embedding dynamic graphs. ACM Computing Surveys (CSUR) 55(1), 1–37 (2021)

2. Grover, A., Leskovec, J.: node2vec: Scalable feature learning for networks. In: Proceedings of the 22nd ACM SIGKDD international conference on Knowledge discovery and data mining, pp. 855–864 (2016)

3. Holme, P., Saram¨aki, J.: Temporal networks. Physics reports 519(3), 97–125 (2012)

4. Khoshraftar, S., An, A.: A survey on graph representation learning methods. ACM Transactions on Intelligent Systems and Technology 15(1), 1–55 (2024)

5. Kipf, T.N., Welling, M.: Semi-supervised classification with graph convolutional networks. arXiv preprint arXiv:1609.02907 (2016)

6. Kipf, T.N., Welling, M.: Variational graph auto-encoders. arXiv preprint arXiv:1611.07308 (2016)

7. Kneˇzevi´c, D., Savi´c, M., Radovanovi´c, M.: Local intrinsic dimensionality for dynamic graph embeddings. In: International Conference on Complex Networks and Their Applications, pp. 374–385. Springer (2024)

8. Lancichinetti, A., Fortunato, S., Kert´esz, J.: Detecting the overlapping and hierarchical community structure in complex networks. New journal of physics 11(3), 033,015 (2009)

9. Mahdavi, S., Khoshraftar, S., An, A.: dynnode2vec: Scalable dynamic network embedding. In: 2018 IEEE international conference on big data (Big Data), pp. 3762–3765. IEEE (2018)

10. Nguyen, G.H., Lee, J.B., Rossi, R.A., Ahmed, N.K., Koh, E., Kim, S.: Continuoustime dynamic network embeddings. In: Companion proceedings of the the web conference 2018, pp. 969–976 (2018)

11. Nickel, M., Kiela, D.: Poincar´e embeddings for learning hierarchical representations. Advances in neural information processing systems 30 (2017)

12. Pandhre, S., Mittal, H., Gupta, M., Balasubramanian, V.N.: Stwalk: learning trajectory representations in temporal graphs. In: Proceedings of the ACM India joint international conference on data science and management of data, pp. 210– 219 (2018)

13. Perozzi, B., Al-Rfou, R., Skiena, S.: Deepwalk: Online learning of social representations. In: Proceedings of the 20th ACM SIGKDD international conference on Knowledge discovery and data mining, pp. 701–710 (2014)

14. Radovanovic, M., Nanopoulos, A., Ivanovic, M.: Hubs in space: Popular nearest neighbors in high-dimensional data. Journal of machine learning research 11(sept), 2487–2531 (2010)

15. Savi´c, M., Kurbalija, V., Radovanovi´c, M.: Local intrinsic dimensionality and graphs: towards lid-aware graph embedding algorithms. In: International Conference on Similarity Search and Applications, pp. 159–172. Springer (2021)

16. Savi´c, M., Kurbalija, V., Radovanovi´c, M.: Local intrinsic dimensionality measures for graphs, with applications to graph embeddings. Information Systems 119, 102,272 (2023)

17. Tomˇci´c, A., Savi´c, M., Radovanovi´c, M.: Hub-aware random walk graph embedding methods for classification. Statistical Analysis and Data Mining: The ASA Data Science Journal 17(2), e11,676 (2024)

18. Tomˇci´c, A., Savi´c, M., Simi´c, D., Radovanovi´c, M.: Dynamic graph embedding through hub-aware random walks. In: International Conference on Similarity Search and Applications, pp. 315–329. Springer (2025)

19. Zhang, M., Chen, Y.: Link prediction based on graph neural networks. Advances in neural information processing systems 31 (2018)