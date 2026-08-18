# POI Recommendation with LLM-Augmented Multi-Graph Learning and Contrastive Alignment

Burak Tamer, Wolfram Hopken and Zehui Wang¨

Abstract Point-of-interest (POI) recommendation models based on graph neural networks achieve strong performance by propagating collaborative signals over user-item interactions, yet they struggle with the cold-start problem, where items with few or no interactions are not represented. In this paper, we propose LLM-augmented Multi-Graph Contrastive Learning (LLM-MGCL), a multi-graph neural network that uses semantic and spatial information about items to extend the LightGCN backbone with two auxiliary item-item graphs: a semantic graph constructed from sentence embeddings of LLM-generated photo summaries and keywords, and a geographic graph derived from Haversine distances between business locations. Item embeddings are propagated over all three graphs in parallel, fused additively, and aligned across views through a bidirectional InfoNCE contrastive objective that connects behavioral, semantic, and spatial representations of the same items. Experiments on the Yelp Multimodal Recommendation Dataset show that LLM-MGCL outperforms classical collaborative filtering, matrix factorization, and interaction-only graph neural network baselines. It improves Recall@20 by 52.0% and NDCG@20 by 64.8% over LightGCN while performing on par with the strongest contrastive baseline, Self-supervised Graph Learning (SGL), which is also afected by the cold-start problem. An ablation study reveals that the cross-view contrastive alignment (CA) is the primary driver of these gains, with the best performance achieved when all three graphs are combined. Our results suggest that externally grounded, LLM-derived item knowledge can efectively compensate for missing collaborative signal and mitigate the item cold-start problem in POI recommendation.

Keywords Graph Neural Network · Recommender System · POI Recommendation · Multi-Graph Learning · Item-Item Graph · Large Language Model

## 1 Introduction

Point-of-interest (POI) recommendation has become a central component of locationbased services, helping users navigate and discover businesses that match their preference and location [16]. Graph neural networks (GNNs), and Light Graph Convolution Network (LightGCN) [2] in particular, have emerged as a strong backbone for this task by propagating collaborative signals across user and item interactions, becoming a powerful paradigm enabling state-of-the-art performance [13, 2, 9]. However, purely collaborative approaches rely exclusively on observed interaction patterns and therefore struggle with the cold-start problem [7]: businesses with few or no interactions by users receive little to no useful signal during training and are therefore not recommended. At the same time, semantic and spatial information about businesses, such as what they look like, what services they ofer, and where they are located, is available but rarely exploited by these models. Since this information does not depend on the user or the interaction, but rather on the businesses, it could directly compensate for the missing collaborative signal that limits recommendation in the cold-start problem.

In this work, we propose to leverage LLM-generated textual descriptions of businesses from multimodal data (images and metadata), as well as geographic proximity to enhance POI recommendation. The central research question is whether incorporating LLM-generated textual information such as summaries, keywords and spatial relationships between businesses into a multi-graph neural network with contrastive alignment can improve recommendation performance compared to state-of-the-art approaches that rely solely on user-item interactions. The main contributions of this work are as follows:

• Construction of a semantic item-item graph based on textual embeddings derived from LLM-generated descriptions and keywords, capturing semantic similarity between businesses in the Yelp Multimodal Recommendation Dataset [10].

• Construction of a geographic item-item graph based on the Haversine distance between business locations, capturing spatial proximity that is particularly relevant for POI recommendation.

• Design of a multi-graph model (LLM-MGCL) that integrates user-item interactions with both item-item graphs, and its evaluation against collaborative filtering (CF), matrix factorization (MF) and interaction-only graph baselines.

The remainder of this paper is structured as follows. Section 2 reviews related work in recommendation systems, graph neural networks and the use of textual information for recommendation. In Sect. 3 we describe the dataset used, the proposed methodology and model architecture, including graph construction and the GNN training procedure. Lastly, Sect. 4 presents the experimental results compared to other state-of-the-art models and Sect. 5 concludes the paper and discusses potential future work.

## 2 Related Work

Recommender systems have evolved from modeling user-item interactions to increasingly incorporating richer sources of information, such as auxiliary graph structure, self-supervised training signals, and external content beyond the interaction history itself [13]. Each development addresses a specific limitation of purely interaction-based collaborative filtering, and together they motivate the design choices underlying LLM-MGCL.

GNN-based Recommendation: Graph neural networks (GNNs) have become a dominant paradigm for collaborative filtering by modeling user-item interactions in a bipartite graph structure [13]. NGCF [9] introduced standard graph convolution while LightGCN [2] showed that a simplified propagation scheme of only neighborhood aggregation achieves better performance. Related work such as GC-MC [1] and PinSage [14] similarly demonstrated the efectiveness of graph-based propagation for recommendation systems. These graph-based methods achieve strong performance, but inherit the same dependence on interaction data discussed in Sect. 1.

Contrastive Learning: To improve the robustness of learned representations, selfsupervised contrastive learning has recently been introduced into graph-based recommendation [13]. SGL [11] provides an additional training signal beyond the ranking objective by generating two augmented views of the interaction graph through random edge or node dropout and trains the model to align representations of the same node across diferent views. Recent work such as SimGCL [15] and NCL [4] refined this idea by replacing structural augmentation with simpler embedding level perturbations. A common characteristic of these methods is that the contrasted views are perturbations of the same interaction graph, meaning the additional signal is derived entirely from the same source of information the model already used [11].

Content and Geographic Signals: A separate line of work incorporates external information, such as text, images, or knowledge graphs, to complement collaborative signals. KGAT [8], for example, integrates a knowledge graph of item attributes into the propagation process, while multimodal recommenders combine visual and textual item features into the propagation process [17]. With the emergence of large language models (LLMs), recent work has begun incorporating LLM-generated content as a richer source of item semantics, reflecting a trend towards using LLMs for recommender systems [12]. Geographic proximity has similarly been used as an auxiliary signal in POI recommendation, reflecting the intuition that users are more likely to interact with nearby businesses [16].

This work builds on the LightGCN backbone and extends it with LLM-generated semantic content and geographic proximity, aligned through a contrastive learning objective. Unlike SGL and its successors, LLM-MGCL contrasts collaborative views against semantic and geographic representations, rather than randomly perturbed views.

## 3 Methodology

The presented approach combines collaborative filtering signals with LLM-derived semantic and geographic information through a multi-graph architecture. LLM-MGCL extends the LightGCN backbone with two parallel propagation paths over item-item graphs constructed from LLM-generated semantic similarity and geographic proximity and aligns the resulting item representations through contrastive learning objectives.

As illustrated in Fig. 1, LLM-MGCL uses three graphs as input, a user-item graph �<sub>��</sub> encoding collaborative behavior between users and businesses, an item-item similarity graph $G _ { S E M }$ derived from LLM-generated photo summaries and an item-item geographic graph $G _ { G E O }$ derived from Haversine distances between business locations. All three graphs share the same item node set, allowing semantic, behavioral and geographic signals to influence the same underlying item representation.

![](images/70dd04431051598201418aad6af941a3c20610209f02a3968e2fd4dd5e13dc2b.jpg)  
Fig. 1 Overview of the LLM-MGCL methodology.

The following sections describe the dataset (Sect. 3.1), the construction of each graph (Sect. 3.2), the model architecture (Sect. 3.3) and the training procedure (Sect. 3.4).

## 3.1 Dataset and Split

Experiments are conducted on the Yelp Multimodal Recommendation Dataset, which is derived from the online business review platform Yelp and widely adopted in research on collaborative filtering and POI recommendation [13]. Each interaction consists of a user ID, a business ID and a star rating; in addition, the dataset provides multimodal information about businesses such as images and metadata.

We use the processed and extended version provided by Wang et al. [10] on Hugging Face<sup>1</sup>, in which the LLMs ChatGPT and DeepSeek transform this multimodal information into natural language summaries and keywords, providing a high-level semantic representation of each business [10].

Train-Test-Validation Split: We apply a per-user random split, retaining only users with at least three interactions so that every user can contribute to all partitions. Per user, interactions are shufled with a fixed random seed and divided into training (80%), validation (10%) and test (10%) sets.

## 3.2 Graph construction

LLM-MGCL operates on three complementary graphs that capture diferent types of relationships between users and businesses. The user-item graph $G _ { U I }$ encodes observed collaborative behavior, while the two item-item graphs $G _ { S E M }$ and $G _ { G E O }$ provide auxiliary structural information, with $G _ { S E M }$ derived from LLM-generated semantic content and $G _ { G E O }$ from geographic locations.

## 3.2.1 User-Item Graph

The user-item graph $G _ { U I } = ( \mathcal { U } \cup \mathcal { I } , \mathcal { E } _ { U I } )$ is a bipartite graph where U denotes the set of users, I the set of businesses and $\mathcal { E } _ { U I }$ the set of observed user-business interactions. Each interaction corresponds to a review submitted by a user � $\in \mathcal { U }$ for a business $i \in \mathcal { I }$ accompanied by a star rating $r _ { u i } \in \{ 1 , 2 , 3 , 4 , 5 \}$ . To incorporate star rating information into the graph structure, each edge $( u , i ) \in \mathcal { E } _ { U I }$ is assigned a weight $w _ { u i }$ derived from the user’s star rating.

$$
w _ { u i } = \operatorname* { m a x } \left( \frac { r _ { u i } - 1 } { 4 } , 0 . 1 \right)\tag{1}
$$

This transformation (Eq. 1) maps 5-star ratings to a maximum weight of 1.0 and 1-star ratings to a minimum weight of 0.1, which maps $r _ { u i } ~ \in ~ \{ 1 , 2 , 3 , 4 , 5 \}$ to $w _ { u i } \in \{ 0 . 1 , 0 . 2 5 , 0 . 5 , 0 . 7 5 , 1 . 0 \}$ . The lower clamp of 0.1 ensures that low-rated interactions still contribute to message passing, preventing complete elimination of low-rated reviews while emphasizing strong positive ratings. The resulting graph is converted into a symmetric adjacency matrix $\mathbf { A } _ { U I } \in \mathbb { R } ^ { ( | \mathcal { U } | + | \mathcal { I } | ) \times ( | \mathcal { U } | + | \mathcal { I } | ) }$ where $\mathbf { R } \in \mathbb { R } ^ { | \mathcal { U } | \times | \bar { \mathcal { I } } | }$ contains the weighted interactions. The matrix is then normalized using the standard LightGCN formulation [2, Eq. 8]. This symmetric normalization stabilizes message passing by counteracting the efect of high-degree nodes during propagation [2, p. 4-5]. After preprocessing, $G _ { U I }$ contains $| \mathcal { U } | = 2 2 , 2 0 7$ users, $| \bar { J } | = 1 6 { , } 7 0 2$ businesses and 1,247,788 bidirectional weighted edges, corresponding to 623,894 unique user-business interactions in the training split.

## 3.2.2 SEM-Item-Item Graph

The SEM-item-item graph $G _ { S E M } = \left( \bar { I } , \mathcal { E } _ { S E M } \right)$ encodes semantic similarity between businesses based on LLM-generated textual content. For each business, the LLMgenerated photo summary and keywords [10] are concatenated into a single textual representation $t _ { i }$ and encoded with the pretrained Sentence-Transformer model all-MiniLM-L6-v2 into an L2-normalized embedding $\mathbf { s } _ { i } \in \mathbb { R } ^ { 3 8 4 }$ , so that the cosine similarity sim $( i , j )$ between two businesses reduces to the inner product $\mathbf { s } _ { i } \cdot \mathbf { s } _ { j } .$

$$
\mathrm { s i m } ( i , j ) = \frac { \mathbf { s } _ { i } \cdot \mathbf { s } _ { j } } { \| \mathbf { s } _ { i } \| \cdot \| \mathbf { s } _ { j } \| }\tag{2}
$$

To construct a sparse and informative graph, top-k filtering is applied: for each business only the $k = 1 0$ most similar businesses are retained as neighbors. This bounds the graph density at $O ( k \cdot | \tau | )$ edges and prevents items with broadly applicable descriptions from dominating the neighborhood structure. The fixed neighborhood size also guarantees that every item, including those with few or no observed user-item interactions, receives a consistent semantic signal during propagation, directly addressing the item cold-start problem inherent to collaborative filtering [7]. Edges are added bidirectionally with the similarity score as edge weight, and the resulting adjacency matrix is normalized just like for $G _ { U I }$ . The final graph contains 265,780 bidirectional weighted edges with a mean raw similarity of 0.87, indicating that retained neighbors are semantically close to their target items.

## 3.2.3 GEO-Item-Item Graph

The geographic-item-item graph $G _ { G E O } = ( { \cal I } , { \cal E } _ { G E O } )$ encodes spatial proximity between businesses based on their physical locations. While $G _ { S E M }$ captures semantic similarity, $G _ { G E O }$ captures locational relationships, which are relevant for POI recommendations, where users typically interact with businesses in geographic proximity.

Coordinate Extraction: Each business $i \in \mathcal { I }$ is associated with a geographic location represented by latitude $\phi _ { i }$ and longitude $\lambda _ { i }$ , both extracted from the Yelp business metadata.

Distance Computation: The geographic distance between two businesses � and $j$ is computed using the Haversine great-circle formula:

$$
d ( i , j ) = 2 R \cdot \arcsin \left( \sqrt { \sin ^ { 2 } \left( \frac { \Delta \phi } { 2 } \right) + \cos ( \phi _ { i } ) \cos ( \phi _ { j } ) \sin ^ { 2 } \left( \frac { \Delta \lambda } { 2 } \right) } \right)\tag{3}
$$

The result $d ( i , j )$ represents the great-circle distance in kilometers between the two compared businesses.

Edgefiltering: Analogous to $G _ { S E M }$ , only pairs within a distance threshold of $d _ { m a x } = 5 0$ km are considered, and for each business the $k _ { g e o } = 1 0$ nearest neighbors are retained, ensuring a balanced neighborhood structure regardless of regional business density.

Edge Weighting: Unlike the SEM-item-item graph, where higher cosine similarity directly corresponds to stronger edges, the geographic edge weights are defined by the calculated distances of the Haversine formula. Closer businesses should have stronger connections; therefore, the edge weights are defined as:

$$
w _ { i j } ^ { G E O } = \frac { 1 } { 1 + d ( i , j ) }\tag{4}
$$

This formulation ensures that weights decrease smoothly as distance grows and stay bounded between 0 and 1, matching the scale of the cosine-based weights in $G _ { S E M }$ to keep the signal stable during message passing.

Graph Assembly: To form an undirected graph, edges are added bidirectionally. Like every graph, the matrix is symmetrically normalized following the LightGCN scheme.

The final GEO-item-item graph contains 334,040 bidirectional weighted edges. Before normalization, the raw edge weights have a mean of 0.74, reflecting the high spatial density of businesses in the same area, which is particularly well suited for POI recommendations, since users typically choose between alternatives in close proximity.

## 3.3 LLM-MGCL Architecture

LLM-MGCL builds on LightGCN, which learns user and item representations through simple neighborhood aggregation without learned weight matrices or nonlinear activations [2]. Instead of propagating only over the user-item graph, LLM-MGCL runs three propagation paths in parallel and combines the output into a unified item representation. Embedding Layer: Before any graph propagation takes place, each user and item is assigned a trainable embedding vector of the dimension 64. These vectors are the representation of the users and items in a continuous space where similar entities end up close to each other in training. Each matrix is randomly initialized from a zero-mean normal distribution with a standard deviation of $\sigma = 0 . 0 1$ and is updated end-to-end during the training. The item embedding matrix is hereby shared across all propagation paths. This means that when the model updates item representations based on semantic similarity, those changes also afect the same embeddings used in the other paths.

Multi-Graph Propagation: The heart of LLM-MGCL is a propagation mechanism that spreads information through each graph. The mechanism is straightforward: if two items are connected in a graph — because users interact with both, because they are semantically similar or because they are geographically close, they should influence each other’s representation. While LightGCN applies this mechanism exclusively to collaborative filtering propagation, LLM-MGCL spreads information through multiple graphs simultaneously. Expanding this mechanism by stacking multiple propagation layers, the model can aggregate information not just from direct neighbors but also from secondand third-order connections. All three paths use the same propagation formula, following the LightGCN neighborhood aggregation scheme [2]. At each layer, a node’s new representation is a weighted sum of its neighbors’ current representations, where the weights are given by the pre-computed normalized adjacency matrix. After all layers are computed, the representations from each layer, including the initial embedding, are averaged to produce the final output. This layer-wise mean pooling ensures that not only local (one-hop) but also global (multi-hop) neighborhood information contribute to the final representation [2, p. 4]. The collaborative path propagates the users and item embeddings over $G _ { U I }$ for the recommended three layers [2]. Since $G _ { S E M }$ and $G _ { G E O }$ contain no user nodes, the semantic and geographic paths propagate only the item embeddings, for two layers each. A shallower depth for these two paths is used because each item is connected only to its k-nearest neighbors. A deeper propagation would mix in loosely related items and dilute rather than sharpen the signal. This results in three separate item representations: a collaborative view ${ \bf e } _ { i } ^ { \scriptscriptstyle \hat { C } F }$ , a semantic view ${ \bf e } _ { i } ^ { S E M }$ and a geographic view $\mathbf { e } _ { i } ^ { G E O }$

Fusion and Prediction: To combine the three views into a single representation for ranking, LLM-MGCL uses additive fusion, which lets all views contribute equally and keeps the model free of additional learned parameters:

$$
{ \bf e } _ { i } ^ { \mathrm { f i n a l } } = { \bf e } _ { i } ^ { C F } + { \bf e } _ { i } ^ { S E M } + { \bf e } _ { i } ^ { G E O }\tag{5}
$$

The interaction score between a user � and an item � is the dot product $\hat { y } _ { u i } = \mathbf { e } _ { u } ^ { C F }$ $\mathbf { e } _ { i } ^ { \mathrm { { f i n a l } } }$ ; during inference, all candidate items are scored and ranked to produce the top-k recommendation list. A higher score means the model rates the item as more relevant. Contrastive Learning Objective: Without an additional training signal, the three propagation paths are only combined in the fusion step and are not forced to produce consistent representations of the same business. LLM-MGCL addresses this with a contrastive learning objective that explicitly pulls together representations of the same item across diferent views while pushing apart representations of diferent items.

In contrast to SGL [11], which contrasts two randomly perturbed views of the same interaction graph, LLM-MGCL contrasts fundamentally diferent sources of information: behavioral co-occurrence on one side (user-item) and externally grounded semantic (SEM-item-item) or geographic knowledge (geo-item-item) on the other. This makes the alignment not merely a noise-robustness signal, but more meaningful as an explicit link between what users do and what the content or location says about an item.

For each training batch, the contrastive loss is computed over the set of positive items ${ \mathcal { B } } ^ { + }$ sampled for the ranking objective described in Sect. 3.4. For each auxiliary view $X \in \{ S E M , G E O \}$ , the loss pulls the collaborative embedding ${ \bf e } _ { i } ^ { C F }$ and the auxiliary embedding $\mathbf { e } _ { i } ^ { X }$ of the same item � closer together, while treating all other items in the batch as negatives. Following the bidirectional InfoNCE formulation used in SGL [11, p. 729], we define:

$$
\begin{array} { r } { \mathcal { L } _ { C L } ^ { X } = - \displaystyle \frac { 1 } { 2 | \mathcal { B } ^ { + } | } \sum _ { i \in \mathcal { B } ^ { + } } \left[ \log \frac { \exp { \left( \sin ( \mathbf { e } _ { i } ^ { C F } , \mathbf { e } _ { i } ^ { X } ) / \tau \right) } } { \sum _ { j \in \mathcal { B } ^ { + } } \exp { \left( \sin ( \mathbf { e } _ { i } ^ { C F } , \mathbf { e } _ { j } ^ { X } ) / \tau \right) } } \right. } \\ { \quad \left. + \log \frac { \exp { \left( \sin ( \mathbf { e } _ { i } ^ { X } , \mathbf { e } _ { i } ^ { C F } ) / \tau \right) } } { \sum _ { j \in \mathcal { B } ^ { + } } \exp { \left( \sin ( \mathbf { e } _ { i } ^ { X } , \mathbf { e } _ { j } ^ { C F } ) / \tau \right) } } \right] } \end{array}\tag{6}
$$

where sim(·, ·) denotes cosine similarity between $L _ { 2 }$ -normalized embeddings and $\tau =$ 0.2 controls how sharply the loss distinguishes between similar and dissimilar pairs. The loss is computed symmetrically in both directions so that neither view dominates. The total contrastive loss averages both views with equal weight.

## 3.4 LLM-MGCL Training

During training three signals need to be balanced: the model must rank observed interactions above unobserved ones, item embeddings must stay aligned across views, and its parameters must not grow too large. LLM-MGCL combines these three goals into a single loss:

$$
\mathcal { L } = \mathcal { L } _ { B P R } + \lambda \mathcal { L } _ { C L } + \gamma \mathcal { L } _ { L 2 }\tag{7}
$$

The ranking term $\mathcal { L } _ { B P R }$ follows the Bayesian Personalized Ranking formulation [6], as also adopted in LightGCN [2, Eq. 15], and encourages the predicted score of an observed user-item pair to exceed that of a sampled unobserved pair. The regularization term $\mathcal { L } _ { L 2 }$ penalizes the squared norm of the initial user and item embeddings involved in each batch. The contrastive term $\mathcal { L } _ { C L }$ is the dual-view alignment loss (Eq. 6) introduced in Sect. 3.3. The two auxiliary terms are scaled by $\lambda = 0 . 0 5$ and $\gamma = 1 0 ^ { - 3 }$ respectively. A small � keeps the contrastive term acting as a regularizer rather than dominating the ranking objective it is meant to support.

Negative Sampling: Since the dataset only records observed interactions between users and items, a review tells us a user visited and rated a specific business, but says nothing about the other businesses the user did not interact with. Negative sampling addresses this gap by artificially constructing negative examples from an unobserved set. For every positive interaction observed, the BPR loss requires a contrasting item which the user has not interacted with. For each training step, one positive item is drawn uniformly at random from a user’s training interactions, and one negative item from the full item set. If the sampled negative happens to also be a positive item for that user it is discarded and re-sampled. This uniform strategy is computationally cheap and forms the standard baseline for negative sampling in implicit-feedback recommendation.

## 4 Empirical Results

In this section, we evaluate LLM-MGCL against a range of established recommendation approaches and isolate the contribution of the components through an ablation study. Model Comparison: Table 1 reports Recall@10, NDCG@10, Recall@20, and NDCG@20 for LLM-MGCL against eight baseline methods, including classical collaborative filtering (ItemKNN, UserKNN), matrix factorization (MF-BPR [5], NeuMF [3]) and graph neural network (LightGCN [2], NGCF [9], GC-MC [1], SGL [11]) methods. All stochastic models are trained under the same seeds and constraints from Sect. 3.4.

Table 1 Performance comparison on the Yelp dataset. Results are averaged over three runs (mean ± std). Deterministic methods (ItemKNN, UserKNN) are reported without standard deviation. Best results are shown in bold, second-best are underlined.
<table><tr><td>Type</td><td>Model</td><td>Recall@10</td><td>NDCG@10</td><td>Recall@20</td><td>NDCG@20</td></tr><tr><td>GNN</td><td>NGCF</td><td> $0 . 0 3 9 2 \pm 0 . 0 0 2 6$ </td><td> $0 . 0 2 8 1 \pm 0 . 0 0 2 2$ </td><td> $0 . 0 6 5 5 \pm 0 . 0 0 3 7$ </td><td> $0 . 0 3 8 2 \pm 0 . 0 0 1 0$ </td></tr><tr><td>MF</td><td>NeuMF</td><td> $0 . 0 4 6 4 \pm 0 . 0 0 3 5$ </td><td> $0 . 0 3 3 7 \pm 0 . 0 0 0 6$ </td><td> $0 . 0 7 5 4 \pm 0 . 0 0 2 3$ </td><td> $0 . 0 4 1 9 \pm 0 . 0 0 1 6$ </td></tr><tr><td>GNN</td><td>GC-MC</td><td> $0 . 0 4 7 9 \pm 0 . 0 0 2 3$ </td><td> $0 . 0 3 3 8 \pm 0 . 0 0 1 3$ </td><td> $0 . 0 7 6 0 \pm 0 . 0 0 5 1$ </td><td> $0 . 0 4 2 8 \pm 0 . 0 0 1 6$ </td></tr><tr><td>GNN</td><td>LightGCN</td><td> $0 . 0 4 6 8 \pm 0 . 0 0 4 1$ </td><td> $0 . 0 3 3 1 \pm 0 . 0 0 3 7$ </td><td> $0 . 0 7 7 3 \pm 0 . 0 0 1 6$ </td><td> $0 . 0 4 3 8 \pm 0 . 0 0 3 0$ </td></tr><tr><td>MF</td><td>MF-BPR</td><td> $0 . 0 5 4 0 \pm 0 . 0 0 3 5$ </td><td> $0 . 0 4 1 0 \pm 0 . 0 0 3 7$ </td><td> $0 . 0 8 9 3 \pm 0 . 0 0 3 8$ </td><td> $0 . 0 5 1 8 \pm 0 . 0 0 2 6$ </td></tr><tr><td>CF</td><td>ItemKNN</td><td>0.0568</td><td>0.0439</td><td>0.0909</td><td>0.0555</td></tr><tr><td>CF</td><td>UserKNN</td><td>0.0666</td><td>0.0507</td><td>0.1093</td><td>0.0652</td></tr><tr><td>GNN</td><td>SGL</td><td> $\mathbf { 0 . 0 7 0 3 \pm 0 . 0 0 3 0 }$ </td><td> $\mathbf { 0 . 0 5 3 2 \pm 0 . 0 0 1 8 }$ </td><td> $\mathbf { 0 . 1 1 8 7 \pm 0 . 0 0 4 1 }$ </td><td> $\underline { { 0 . 0 7 1 2 } } \pm 0 . 0 0 1 2$ </td></tr><tr><td>GNN</td><td>LLM-MGCL</td><td> $\underline { { 0 . 0 6 9 9 } } \pm 0 . 0 0 1 9$ </td><td> $\underline { { 0 . 0 5 1 1 } } \pm 0 . 0 0 2 7$ </td><td> $\underline { { 0 . 1 1 7 5 } } \pm 0 . 0 0 5 1$ </td><td> $\mathbf { 0 . 0 7 2 2 \pm 0 . 0 0 2 3 }$ </td></tr></table>

LLM-MGCL outperforms all classical and matrix factorization methods, as well as the GNN baselines that do not incorporate any auxiliary item-item structure (LightGCN, NGCF, GC-MC) and rely only on collaborative filtering. Compared to LightGCN, the direct backbone LLM-MGCL builds on, Recall@20 improves from 0.0773 to 0.1175, a relative gain of over 50%, and NDCG@20 improves from 0.0438 to 0.0722, a gain of roughly 65%. This indicates that the combination of LLM-generated semantic and geographic item-item graphs with contrastive alignment adds substantial value beyond the collaborative filtering backbone. Against SGL, the strongest baseline overall, the picture is nuanced. SGL achieves higher Recall@10, NDCG@10, and Recall@20, while LLM-MGCL achieves a higher NDCG@20 (0.0722 vs. 0.0712). Across the remaining three metrics, the gap to SGL is small relative to the standard deviations. This suggests that the two models perform comparably overall, while LLM-MGCL additionally incorporates externally grounded semantic and geographic information, which, unlike SGL, also reaches items with few or no observed user-item interactions via semantic and geographic edges, mitigating the cold-start problem.

Ablation Study: To isolate the contribution of individual model components, we evaluate the four ablation variants alongside the full model. Each variant disables one or more components while keeping all other hyperparameters fixed. This allows a direct comparison of the performance attributed to the target component. The LightGCN baseline serves as the lower bound, using only collaborative filtering between users and items. It provides the reference point against which all other variant improvements are measured. The variant w/o CL sets the contrastive loss weight � to zero, removing the alignment signal between the collaborative view (user-item) and the auxiliary views. This isolates the efect of view alignment. Without it, the three propagation paths contribute through additive fusion, but are no longer encouraged to produce consistent representations. The variant w/o $G _ { S E M }$ replaces the SEM-item-item graph with an empty graph, ignoring the semantic propagation of the item embeddings. A performance diference in this variant compared to the full model indicates that the SEM-item-item graph carries information which is crucial for recommendation performance. The variant w/o $G _ { G E O }$ removes the geographic propagation path. This brings collaborative (user-item) and semantic (item-item) signals in the foreground. Since POI recommendation is inherently location-sensitive, this variant tests how much explicit geographic modeling adds beyond what collaborative behavior implicitly encodes.

The ablation study (Table 2) reveals that contrastive learning is the primary driver of performance gains. Removing CL from the full model causes a substantial drop in Recall@20 from 0.1175 to 0.0906 (a relative decrease of 22.9%), while keeping both item-item graphs intact. In contrast, removing either the LLM-augmented SEM item-item graph $G _ { S E M }$ or the geographic graph $G _ { G E O }$ results in only minor performance diferences (Recall@20 of 0.1157 and 0.1172 respectively), suggesting that the two auxiliary graphs encode partially overlapping information. We hypothesize this is because businesses in the same geographic area often share similar semantic characteristics, leading to redundancy between the two views. Nevertheless, the highest performance across all metrics is achieved only when both graphs are combined with contrastive learning (+52.0% Recall@20 and +64.8% NDCG@20 over the LightGCN baseline), demonstrating that LLM-generated semantic edges, geographic proximity and contrastive learning provide complementary supervision signals.

Table 2 Ablation study of LLM-MGCL components on the Yelp dataset. Single-component ablations (w/o CL, w/o $G _ { S E M } ,$ w/o $G _ { G E O } )$ are reported for a single run with seed 42 due to computational constraints. For the full model, results are averaged over three random seeds (42, 123, 2024) and reported as mean. Δ denotes relative improvement over LightGCN baseline.
<table><tr><td>Model Variant</td><td> $G _ { S E M }$ </td><td> $G _ { G E O }$ </td><td>CL</td><td>R@20</td><td>∆R@20</td><td>N@20</td><td>∆N@20</td></tr><tr><td>LightGCN</td><td>×</td><td>×</td><td>×</td><td>0.0773</td><td></td><td>0.0438</td><td></td></tr><tr><td>LLM-MGCL w/o CL</td><td>√</td><td>√</td><td>×</td><td>0.0906</td><td>+17.25%</td><td>0.0539</td><td>+22.99%</td></tr><tr><td>LLM-MGCL w/o GSEM</td><td>×</td><td>√</td><td>√</td><td>0.1157</td><td>+49.67%</td><td>0.0699</td><td>+59.52%</td></tr><tr><td>LLM-MGCL w/o GGEO</td><td>√</td><td>×</td><td>√</td><td>0.1172</td><td>+51.60%</td><td>0.0714</td><td>+62.94%</td></tr><tr><td>LLM-MGCL (full)</td><td>√</td><td>√</td><td>√</td><td>0.1175</td><td>+52.00%</td><td>0.0722</td><td>+64.80%</td></tr></table>

## 5 Conclusion

In this paper, we propose LLM-MGCL, a multi-graph neural network for POI recommendation that extends the LightGCN backbone with two auxiliary item-item graphs. The item-item graphs consist of a semantic graph constructed from LLM-generated photo summaries and keywords, and a geographic graph derived from Haversine distances between business locations. By propagating item embeddings over all three graphs in parallel and aligning the collaborative view with the semantic and geographic views through a bidirectional InfoNCE contrastive objective, the model injects contentand location-based signals directly into the item representations, addressing the item cold-start problem inherent to purely interaction-based methods.

Experiments on the Yelp Multimodal Recommendation Dataset demonstrate that LLM-MGCL substantially outperforms classical collaborative filtering, matrix factorization, and interaction-only graph neural network baselines. LLM-MGCL shows improvements over the LightGCN backbone on Recall@20 by 52.0% and NDCG@20 by 64.8%, while performing on par with SGL, the strongest contrastive baseline, which also sufers from the cold-start problem. The ablation study revealed that the contrastive alignment between the collaborative and auxiliary views is the primary driver of these gains. Removing it reduces Recall@20 by 22.9%, whereas removing either individual item-item graph causes only minor degradation. Nevertheless, the best results across all metrics are achieved only when both graphs and the contrastive objective are combined, confirming that LLM-generated semantic edges, spatial proximity, and cross-view alignment provide complementary supervision beyond the collaborative signal.

These findings suggest that externally grounded, LLM-derived item information can serve as an efective substitute for missing collaborative signal that models like SGL explicitly rely on, ofering a path towards mitigating cold-start problems in POI recommendation. Regarding future research driven by our work, three directions appear promising. First, the additive fusion could be replaced by a learned weighting mechanism to adaptively balance the three views. Second, the individual contribution of each graph could be examined more closely to better understand how their information overlaps. Third, a dedicated cold-start evaluation could assess the model’s benefit for businesses with few interactions more directly.

## References

1. Berg, R., Kipf, T., Welling, M.: Graph convolutional matrix completion. arXiv preprint arXiv:1706.02263 (2017)

2. He, X., Deng, K., Wang, X., Li, Y., Zhang, Y., Wang, M.: LightGCN: Simplifying and powering graph convolution network for recommendation. In: Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval, SIGIR ’20, pp. 639– 648 (2020)

3. He, X., Liao, L., Zhang, H., Nie, L., Hu, X., Chua, T.S.: Neural collaborative filtering. In: Proceedings of the 26th International Conference on World Wide Web, WWW ’17, pp. 173–182. International World Wide Web Conferences Steering Committee (2017)

4. Lin, Z., Tian, C., Hou, Y., Zhao, W.X.: Improving graph collaborative filtering with neighborhoodenriched contrastive learning. In: Proceedings of the ACM Web Conference 2022, WWW ’22, pp. 2320–2329 (2022)

5. Loni, B., Pagano, R., Larson, M., Hanjalic, A.: Bayesian personalized ranking with multi-channel user feedback. In: Proceedings of the 10th ACM Conference on Recommender Systems, RecSys ’16, pp. 361–364 (2016)

6. Rendle, S., Freudenthaler, C., Gantner, Z., Schmidt-Thieme, L.: BPR: Bayesian personalized ranking from implicit feedback. In: Proceedings of the Twenty-Fifth Conference on Uncertainty in Artificial Intelligence, UAI ’09, pp. 452–461 (2009)

7. Schein, A.I., Popescul, A., Ungar, L.H., Pennock, D.M.: Methods and metrics for cold-start recommendations. In: Proceedings of the 25th Annual International ACM SIGIR Conference on Research and Development in Information Retrieval, SIGIR ’02, pp. 253–260 (2002)

8. Wang, X., He, X., Cao, Y., Liu, M., Chua, T.S.: KGAT: Knowledge graph attention network for recommendation. In: Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, KDD ’19, pp. 950–958 (2019)

9. Wang, X., He, X., Wang, M., Feng, F., Chua, T.S.: Neural graph collaborative filtering. In: Proceedings of the 42nd International ACM SIGIR Conference on Research and Development in Information Retrieval, SIGIR’19, pp. 165–174 (2019)

10. Wang, Z., Hopken, W., Jannach, D.: Beyond visit trajectories: Enhancing POI recommendation¨ via LLM-augmented text and image representations. In: Proceedings of the Nineteenth ACM Conference on Recommender Systems, RecSys ’25, pp. 521–526 (2025)

11. Wu, J., Wang, X., Feng, F., He, X., Chen, L., Lian, J., Xie, X.: Self-supervised graph learning for recommendation. In: Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval, SIGIR ’21, pp. 726–735 (2021)

12. Wu, L., Zheng, Z., Qiu, Z., Wang, H., Gu, H., Shen, T., Qin, C., Zhu, C., Zhu, H., Liu, Q., Xiong, H., Chen, E.: A survey on large language models for recommendation. World Wide Web 27 (5) (2024)

13. Wu, S., Sun, F., Zhang, W., Xie, X., Cui, B.: Graph neural networks in recommender systems: A survey. ACM Comput. Surv. 55(5) (2022)

14. Ying, R., He, R., Chen, K., Eksombatchai, P., Hamilton, W.L., Leskovec, J.: Graph convolutional neural networks for web-scale recommender systems. In: Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, KDD ’18, pp. 974–983 (2018)

15. Yu, J., Yin, H., Xia, X., Chen, T., Cui, L., Nguyen, Q.V.H.: Are graph augmentations necessary? Simple graph contrastive learning for recommendation. In: Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval, SIGIR ’22, pp. 1294–1303 (2022)

16. Zhang, Q., Yang, P., Yu, J., Wang, H., He, X., Yiu, S.M., Yin, H.: A survey on point-of-interest recommendation: Models, architectures, and security. IEEE Trans. on Knowl. and Data Eng. 37(6), 3153–3172 (2025)

17. Zhou, X., Shen, Z.: A tale of two graphs: Freezing and denoising graph structures for multimodal recommendation. In: Proceedings of the 31st ACM International Conference on Multimedia, MM ’23, pp. 935–943 (2023)