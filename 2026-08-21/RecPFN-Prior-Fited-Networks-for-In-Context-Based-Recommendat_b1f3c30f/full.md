# RecPFN: Prior-Fited Networks for In-Context-Based Recommendations

En Zhi Tan   
SAP SE   
Singapore   
en.zhi.tan@sap.com   
Jia Xiang Lim   
SAP SE   
Singapore   
jia.xiang.lim@sap.com   
Tze Minh Ng   
SAP SE   
Singapore   
tze.minh.ng@sap.com   
Bryan Lijie Chew   
SAP SE   
Singapore   
bryan.chew@sap.com   
Benjamin Yan Han Yap   
SAP SE   
Singapore   
benjamin.yap@sap.com

## Abstract

We introduce RecPFN, a prior-fitted network that brings in-context learning to sequential recommendation. RecPFN is pretrained entirely on synthetic clickstream environments sampled from a broad structural causal prior, enabling it to amortize Bayesian-style inference from a small support set. At inference, a lightweight decoderonly transformer conditions on a handful of domain sequences and produces next-item predictions for queries in a single forward pass, without any weight updates. Across eight public benchmarks, RecPFN achives state-of-the-art zero-shot performance while remaining strongly competitive with supervised methods in lowcompute and low-data regimes. It is deployment-eficient and robust to domain shift, outperforming strong zero-shot baselines that rely on large real-interaction corpora. RecPFN provides a practical path toward generalizable, data-eficient recommenders and opens avenues for richer priors, longer-context ICL, and multimodal extensions. Code for training and evaluation is publicly available at github.com/SAP-samples/tabular-ai-recpfn.

## CCS Concepts

• Information systems → Recommender systems.

## Keywords

In-context-learning, Recommender systems, Prior-fitted networks

## ACM Reference Format:

En Zhi Tan, Jia Xiang Lim, Bryan Lijie Chew, Tze Minh Ng, and Benjamin Yan Han Yap. 2026. RecPFN: Prior-Fitted Networks for In-Context-Based Recommendations. In Proceedings ofthe 49th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR ’26), July 20–24, 2026, Melbourne, VIC, Australia. ACM, New York, NY, USA, 12 pages. https://doi.org/10.1145/3805712.3809696

## 1 Introduction

Recommender systems personalize large information spaces to drive engagement and business outcomes [17, 39, 45, 64]. The strongest sequential models (e.g., self-attention/transformers) typically require large in-domain datasets, extensive training, and significant online-learning compute [6, 11, 20, 42, 53, 56, 58, 62].

Inspired by foundation models in other domains [1, 27, 31, 36], recent work explores generalist recommenders via LLM prompting [28, 33, 49] or by pretraining recommendation-specific architectures on diverse user datasets [8, 19, 40]. However, these approaches struggle with true zero-shot transfer: preferences and interaction patterns are highly setting-dependent, and mixed-domain pretraining often fails to capture the nuances of a new target environment.

This motivates in-context learning (ICL), where a model adapts at inference by conditioning on a small set of examples without weight updates [9]. Beyond LLMs, ICL has proven efective on structured data [3, 34]. Prior-Fitted Networks (PFNs) go further by training on diverse synthetic distributions to approximate Bayesian inference, enabling accurate predictions from few-shot context in a single forward pass [13, 14, 35].

We introduce RecPFN, to our knowledge the first embeddingbased PFN-style ICL approach for sequential recommendation. Our contributions are:

• A synthetic framework for generating clickstream embedding sequences under a broad, expressive prior.

• A lightweight transformer that consumes example sequences from the target domain and predicts the next item for query sequences in a single pass.

Pretraining solely on synthetic data yields strong performance, outperforming zero-shot baselines and rivaling supervised methods with far less compute. RecPFN enables data-eficient, pure incontext adaptation for new domains, ofering a practical path toward generalizable recommenders.

## 2 Related work

Sequential recommendation has progressed from RNNs and CNNs to transformers [20, 42–44, 62, 65]. While classical systems are largely ID-based, text and multimodal embeddings improve generalization and cold-start robustness [21, 23, 30, 40, 63]. Recent approaches integrate pretrained language models, either frozen or fine-tuned for recommendation [4, 22, 24, 29].

The push for foundational recommenders spans two directions. LLM prompting methods format histories and candidates as text, but often face higher latency and dificulty capturing latent collaborative signals [26, 28, 33, 49]. Pretraining recommendation-specific architectures on large, aggregated datasets has also been explored $[ 8 , 1 9 , 2 4 , 4 0 , 4 7 , 4 8 , 6 1 ] ,$ , including domain-invariant learning and cross-domain embedding alignment; yet these domain-agnostic strategies often underperform in new settings due to environmentspecific behavioral logic.

ICL ofers adaptation without weight updates. Beyond prompting LLMs for ranking/prediction [49, 55], PFNs achieve ICL on structured data by training on diverse synthetic tasks and amortizing Bayesian inference [13]. Our work is the first to extend PFN-style ICL to sequential recommendation.

Synthetic data supports augmentation, privacy, and evaluation in recommenders [2, 41, 57], via autoencoders, clickstream statistics, or latent preference modeling [5, 38, 50, 54, 59]. We adopt a causal, latent-factor perspective: a broad prior over a structural causal model defines item properties and transitions, enabling sampling of diverse environments. Training on this distribution teaches RecPFN a general sequential learning algorithm that adapts via in-context examples rather than overfitting to domain-specific idiosyncrasies.

## 3 RecPFN: In-Context Learning for Sequential Recommendation

In this section, we describe the Bayesian framework underlying prior-fitted networks, the synthetic data generation procedure used for pre-training, and the architecture that enables single-pass nextitem prediction conditioned on in-context examples. An overview is shown in Figure 1.

## 3.1 Bayesian view of prior-fitted networks for sequential recommendation

Let the item catalog be $\textit { I } = \left\{ 1 , \ldots , N \right\}$ and a user history be $S _ { 1 : t } = \left( s _ { 1 } , \ldots , s _ { t } \right)$ with $s _ { \tau } \in \mathcal { I }$ . The task is to predict the next item �<sub>�+1</sub> ∈ I given $S _ { 1 : t }$ . An environment (hypothesis) $h \in { \mathcal { H } }$ specifies the sequence-evolution law via $p ( j \mid S _ { 1 : t } , h )$ for $j \in \mathcal { I } . \mathsf { A }$ prior $\boldsymbol { p } ( h )$ over H encodes our beliefs about plausible environments. In RecPFN, ℎ is instantiated by a structural causal model (SCM) and its hyperparameters (Section 3.2). Training data are sampled by first drawing an environment and then sequences under its dynamics:

$$
h \sim p ( h ) , \mathcal { D } = \{ ( S _ { 1 : t } ^ { ( n ) } , y ^ { ( n ) } ) \} _ { n = 1 } ^ { M } \sim p ( \mathcal { D } \mid h ) ,
$$

where $y ^ { ( n ) } = s _ { t + 1 } ^ { ( n ) }$ is the next item following $S _ { 1 : t } ^ { ( n ) }$ . At test time in a fixed (but unknown) environment, given an observed dataset D (e.g., the train split) and a query prefix $S _ { 1 : t } { } _ { : }$ , the Bayesian posterior predictive distribution over the next item is

$$
p ( \boldsymbol j | S _ { 1 : t } , \mathcal D ) = \int _ { \mathcal H } p ( \boldsymbol j | S _ { 1 : t } , \boldsymbol h ) p ( \boldsymbol h | \mathcal D ) d \boldsymbol h , \quad \boldsymbol j \in \boldsymbol Z ,\tag{1}
$$

$$
p ( h \mid \mathcal { D } ) = \frac { p ( \mathcal { D } \mid h ) p ( h ) } { \int _ { \mathcal { H } } p ( \mathcal { D } \mid h ^ { \prime } ) p ( h ^ { \prime } ) d h ^ { \prime } } .\tag{2}
$$

Equations (1)–(2) yield the gold-standard target distribution for the next item. We approximate the PPD in (1) with a prior-fitted network �<sub>�</sub> that takes $( S _ { 1 : t } , \mathcal { D } )$ and outputs a predictive distribution $\pi _ { \theta } ( \cdot \mid S _ { 1 : t } , { \mathcal { D } } )$ over $\boldsymbol { \mathcal { T } }$ . We optimize � by minimizing the expected cross-entropy under the joint prior over environments and data:

$$
\begin{array} { r l } & { \mathcal { L } ( \theta ) = \mathbb { E } _ { ( h , \mathcal { D } _ { c } ) \sim p ( h ) p ( \mathcal { D } | h ) } \mathbb { E } _ { ( S _ { 1 : t } , y ) \sim p ( S _ { 1 : t } , y | h ) } } \\ & { \quad \quad \quad \big [ - \log \pi _ { \theta } ( y \mid S _ { 1 : t } , \mathcal { D } _ { c } ) \big ] . } \end{array}
$$

Algorithm 1 Synthetic Environment and Sequence Generation   
Require: Prior ranges for hyperparameters $N , d , \omega , \alpha _ { \mathrm { p o p } } , . . . ;$ em  
bedding bank E   
1: Sample hyperparameters: $N , d , \omega , \alpha _ { \mathrm { p o p } } , \hdots \sim p ( h )$   
2: Sample � item embeddings $\{ \mathbf { e } _ { j } \} _ { j = 1 } ^ { N }$ from $\varepsilon$   
3: Construct transition matrix $\mathbf { T } \in \mathbb { R } ^ { N \times N }$ via random-graph or   
latent-factor prior (Section 3.2.2)   
4: Sample popularity $P _ { \mathrm { p o p } }$ ∼ Softmax $( N ( 0 , \sigma ^ { 2 } ) )$   
5: for each sequence � in the batch do   
6: $s _ { 1 } \sim \mathrm { U n i f o r m } ( \tau )$   
7: for $t = 2 , \ldots , L$ do   
8: $\begin{array} { r } { \mathbf { p } _ { \mathrm { t r a n s } } ^ { \prime }  \sum _ { k = \operatorname* { m a x } ( 1 , t - d ) } ^ { t - 1 } \omega ^ { t - 1 - k } \mathbf { T } _ { s _ { k } , : } } \end{array}$   
9: p<sub>next</sub> $\propto \left( 1 - \alpha _ { \mathrm { p o p } } \right) \mathbf { p } _ { \mathrm { t r a n s } } ^ { \prime } + \alpha _ { \mathrm { p o p } } P _ { \mathrm { p o p } }$   
10: $s _ { t } \sim \mathbf { p } _ { \mathrm { n e x t } }$ (excluding previously seen items)   
11: end for   
12: end for   
13: return sequences, item embeddings $\{ \mathbf { e } _ { j } \}$

where $\mathcal { D } _ { c }$ is the context (support) set. In practice, the expectations are approximated via Monte Carlo using our synthetic generator by sampling $h \sim p ( h )$ , then $\mathscr { D } \sim p ( \mathscr { D } \mid h )$ .

## 3.2 Synthetic Sequential Data Generation via a Causal Prior

We pre-train RecPFN on synthetic environments sampled from a broad prior over a structural causal model (SCM) of user behavior. At a high level, each environment defines a catalog of � items with embeddings and a transition matrix T encoding item-to-item afinities. A user sequence is generated by starting from a random item and, at each step, sampling the next item from a mixture of recency-weighted transition scores (capturing short-term sequential patterns) and a global popularity distribution (capturing long-term static appeal). By sampling environments from a broad prior over the SCM hyperparameters, we expose RecPFN to a wide variety of sequence dynamics during pre-training. Algorithm 1 summarizes the full procedure.

3.2.1 Structural Causal Modelfor User Behavior. Let the item catalog of � items be $\mathcal { I } = \{ 1 , . . . , N \}$ . Each item � has an embedding $\mathbf { e } _ { j } \in \mathbb { R } ^ { D _ { \mathrm { i t e m } } }$ and a global popularity $P _ { \mathrm { p o p } , j }$ with $\begin{array} { r } { \sum _ { j } P _ { \mathrm { p o p } , j } = 1 } \end{array}$ . Let $\dot { \mathbf { T } } \in \mathbb { R } ^ { N \times N }$ be an (unnormalized) transition score matrix; $T _ { j k }$ scores the tendency to visit � after �.

Given a sequence $S = \left( s _ { 1 } , \ldots , s _ { L } \right)$ , we generate:

• Initialization: �<sub>1</sub> ∼ Uniform(I).

• For $t > 1 ,$ , form a recency-weighted transition vector from the last � items:

$$
\mathbf { p } _ { \mathrm { t r a n s } } ^ { \prime } = \sum _ { k = \operatorname* { m a x } \left( 1 , t - d \right) } ^ { t - 1 } w _ { t - 1 - k } \mathbf { T } _ { s _ { k } , : } ,\tag{3}
$$

$$
{ \bf p } _ { \mathrm { t r a n s } } = \frac { { \bf p } _ { \mathrm { t r a n s } } ^ { \prime } } { \sum _ { j } p _ { \mathrm { t r a n s } , j } ^ { \prime } } ,\tag{4}
$$

with weights $w _ { \tau } \geq 0 .$ . The next item is sampled from

$$
\begin{array} { r } { P ( s _ { t } = j \mid s _ { < t } ) \propto ( 1 - \alpha _ { \mathrm { p o p } } ) \rlap { / } P \mathrm { t r a n s } , j + \alpha _ { \mathrm { p o p } } P _ { \mathrm { p o p } , j } , } \end{array}\tag{5}
$$

![](images/ebbc745df73ad8abb73a2c15f10c1615a28d25ff44f3416f27a1617e4b73f354.jpg)  
Figure 1: Overall framework of RecPFN: (From left to right) Synthetic data-driven pre-training on next-item prediction task; RecPFN model architecture comprising of alternating Type-A and Type-B ICL modules; Inference based on on-the-fly adaptation to support set retrieved via a recency-weighted overlap

with $\alpha _ { \mathrm { p o p } } \in [ 0 , 1 ]$ blending short-term sequential context (transition) and long-term static appeal (popularity). The distribution is normalized before sampling.

3.2.2 Transition-Matrix Priors. The sequence dynamics depend critically on the structure of T. We introduce two complementary priors that cover qualitatively diferent real-world patterns: a random-graph prior for sparse, topology-driven transitions (modeling settings where items have a fixed set of natural successors, e.g., series or bundles), and a latent-factor prior for factor-structured transitions (modeling settings where co-consumption is driven by shared latent attributes, $\mathrm { e . g . }$ , genre or category preferences).

Random-Graph Prior. This prior models environments where each item has a small set of likely successors defined independently of embedding similarity—capturing niche co-purchase or playliststyle sequences. For each $j \in \mathcal { I }$

• Sample a neighbor set $\mathcal { K } _ { j } \subset \mathcal { I } \setminus \{ j \}$ of size $k _ { \mathrm { r e l } } ;$ set $T _ { j k } = 1$ for $k \in \mathcal K _ { j }$

• Set $T _ { j j } = 1$ with probability $\scriptstyle { \mathcal { P } } \mathrm { s e l f } ;$ , else 0.

All other entries are 0, yielding a sparse, feature-agnostic transition structure.

Latent-Factor Prior. This prior models environments where transitions are mediated by shared latent concepts—capturing settings such as genre afinity in media or category co-consumption in retail. Let $D _ { \mathrm { l a t e n t } }$ be the latent dimension.

• Sample latent embeddings $\mathbf { L } _ { \mathrm { e m b e d } } \in \mathbb { R } ^ { D _ { \mathrm { l a t e n t } } \times D _ { \mathrm { i t e m } } }$ and a latent transition matrix T<sub>latent</sub> $\in \ \mathbb { R } ^ { D _ { \mathrm { l a t e n t } } \times D _ { \mathrm { l } } }$ atent (via the randomgraph construction).

• Sample a sparse item–factor map $\mathbf { F } \in \mathbb { R } ^ { N \times D _ { \mathrm { l a t e n } } }$ t with entries in $\{ - 1 , 1 \}$ and about $k _ { \mathrm { f a c } }$ nonzeros per row.

• Construct item embeddings and of-diagonal transitions:

$$
\mathbf { E } = \mathbf { F } \mathbf { L } _ { \mathrm { e m b e d } } \in \mathbb { R } ^ { N \times D _ { \mathrm { i t e m } } } ,\tag{6}
$$

$$
{ \bf T } ^ { \prime } = { \bf F } { \bf T } _ { \mathrm { l a t e n t } } { \bf F } ^ { \top } \in \mathbb { R } ^ { N \times N } .\tag{7}
$$

Set $T _ { j k } = T _ { j k } ^ { \prime }$ $j \neq k ,$ and �<sub>��</sub> = 1 w.p. �<sub>self</sub> (else 0).

Since $\mathbf { L _ { \mathrm { e m b e d } } }$ and $\mathbf { T } _ { \mathrm { l a t e n t } }$ are sampled independently, this decouples feature similarity from sequential transitions. Matrix factorization is a special case corresponding to $\mathbf { T } _ { \mathrm { l a t e n t } } = \mathbf { I }$

3.2.3 Prior over Environments. We place broad priors over SCM hyperparameters $( \mathrm { e . g . } , N , d , w , \alpha _ { \mathrm { p o p } } , k _ { \mathrm { r e l } } , p _ { \mathrm { s e l f } } , D _ { \mathrm { l a t e n t } } , k _ { \mathrm { f a c } } )$ . Sampling $\begin{array} { r } { \begin{array} { r } { h \sim \ p ( h ) } \end{array} } \end{array}$ yields diverse environments with varying popularity strength, locality, and factor structure.

3.2.4 Sampling Embedding Vectors. We build a fixed corpus of 800,000 sentence embeddings from BEIR datasets (Quora, FIQA, FEVER, MS MARCO), encoded with the target LLM. For the randomgraph prior, we sample � item embeddings $\{ \mathbf { e } _ { j } \}$ from this bank. For the latent-factor prior, we sample $D _ { \mathrm { l a t e n t } }$ latent vectors for $\mathbf { L _ { \mathrm { e m b e d } } }$ and derive item embeddings via $\mathbf { E } = \mathbf { F } \mathbf { L } _ { \mathrm { e m b e d } }$ . This aligns synthetic embeddings with downstream usage.

3.2.5 Sampling Procedure. To draw a training batch: (1) sample an environment $h \sim p ( h )$ ; (2) instantiate item embeddings and T; (3) generate support and query sequences via the SCM. This yields an efectively infinite stream of diverse sequential tasks for pre-training.

## 3.3 RecPFN Architecture and In-Context Learning

RecPFN is a lightweight decoder-only transformer that performs in-context learning by conditioning query sequences on a small set of support sequences from the target domain. The model stacks � custom ICL blocks that first build intra-sequence representations (causal self-attention) and then integrate support information into queries (context-to-query cross-attention). This design enables single-pass next-item prediction without weight updates.

3.3.1 InputFormulationforIn-ContextLearning. A prompt consists of $N _ { c }$ support sequences and $N _ { q }$ query sequences, each padded to length � with embedding dimension �. The model input can thus be expressed as $\mathbf { X } = ( \mathbf { X } _ { c } , \mathbf { X } _ { q } )$ with $\mathbf { X } _ { c } \in \mathbb { R } ^ { N _ { c } \times L \times D }$ and $\bar { \mathbf { X } } _ { q } \in \mathbb { R } ^ { N _ { q } \times L \times D }$

## 3.3.2 Model Architecture. Each ICL block comprises:

• Intra-sequence self-attention on $\mathbf { X } _ { c }$ and $\mathbf { X } _ { q }$ (with causal masking), producing per-sequence representations.

• Context-to-query cross-attention where query tokens attend over the flattened support sequences, enabling queryconditioned retrieval of relevant support information.

<table><tr><td>Dataset</td><td>Users</td><td>Items</td><td>Interactions</td><td>Ave. Length</td></tr><tr><td colspan="5">Evaluation</td></tr><tr><td>Appliances</td><td>703</td><td>30,252</td><td>6,875</td><td>9.78</td></tr><tr><td>Arts</td><td>1,579,230</td><td>302,809</td><td>2,875,917</td><td>1.82</td></tr><tr><td>Dianping</td><td>208,596</td><td>243,247</td><td>4,422,473</td><td>21.20</td></tr><tr><td>Games</td><td>100,955</td><td>71,982</td><td>733,447</td><td>7.27</td></tr><tr><td>Movies</td><td>450,757</td><td>182,032</td><td>4,169,631</td><td>9.25</td></tr><tr><td>Pantry</td><td>247,659</td><td>10,814</td><td>471,614</td><td>1.90</td></tr><tr><td>Scientific</td><td>28,096</td><td>165,764</td><td>199,798</td><td>7.11</td></tr><tr><td>Yelp</td><td>371,243</td><td>150,346</td><td>4,728,381</td><td>12.74</td></tr><tr><td colspan="5">Pretraining for ablation experiment</td></tr><tr><td>Fashion</td><td>8,886</td><td>186,189</td><td>44,769</td><td>5.04</td></tr><tr><td>Grocery</td><td>248,194</td><td>283,507</td><td>1,841,452</td><td>7.42</td></tr><tr><td>Movies</td><td>450,757</td><td>182,032</td><td>4,169,631</td><td>9.25</td></tr><tr><td>Sports</td><td>6,703,391</td><td>957,764</td><td>12,980,837</td><td>1.94</td></tr></table>

Let $\dim ( \ Q , \ K , \mathbf { V } _ { \iota }$ ) denote a multi-head attention operator. Selfand cross-attention on $\mathbf { X } = ( \mathbf { X } _ { c } , \mathbf { X } _ { q } )$ can then be denoted as:

$$
\mathrm { S e l f A t t e n t i o n } ( \mathbf { X } ) = \left( \mathrm { A t t n } ( \mathbf { X } _ { c } , \mathbf { X } _ { c } , \mathbf { X } _ { c } ) , \mathrm { ~ A t t n } ( \mathbf { X } _ { q } , \mathbf { X } _ { q } , \mathbf { X } _ { q } ) \right)
$$

CrossAttention(X) = (X<sub>�</sub>, Attn(X<sub>�</sub>, flatten $( \mathbf { X } _ { c } )$ , flatten(X<sub>�</sub>)))

The �-th ICL block updates $( \mathbf { X } _ { c } ^ { k - 1 } , \mathbf { X } _ { q } ^ { k - 1 } )$ to $( \mathbf { X } _ { c } ^ { k } , \mathbf { X } _ { q } ^ { k } )$ by:

$$
\mathbf { H } ^ { ( k ) } = \mathrm { I C L } ( \mathbf { H } ^ { ( k - 1 ) } ) = \mathrm { C r o s s A t t e n t i o n } \big ( \mathrm { S e l f A t t e n t i o n } ( \mathbf { H } ^ { ( k - 1 ) } ) \big )
$$

3.3.3 Alternating Atention Mechanisms. To balance copying-style operations and feature extraction, RecPFN alternates two attention types per block.

Type-A (summed-head attention, no FFN). The value vector for each head is projected to the full embedding dimension �. We then specially initialize $\mathbf { W } _ { i } ^ { V }$ to be the identity matrix. Here, we take inspiration from [18] who demonstrate powerful and eficient sequence copying abilities with such a formulation. We replace the MLP with a summation across the heads since it has been demonstrated that attention-only transformers are capable of information movement along the sequence [10] and defer the learning of nonlinear latent features to Type-B attention.

$$
\mathbf { Q } _ { i } = \mathbf { Z } \mathbf { W } _ { i } ^ { Q } , \quad \mathbf { K } _ { i } = \mathbf { Z } \mathbf { W } _ { i } ^ { K } , \quad \mathbf { V } _ { i } = \mathbf { Z } \mathbf { W } _ { i } ^ { V }\tag{8}
$$

$$
\bf { \Delta } \mathrm { { w h e r e } } \bf { W } _ { \it { i } } ^ { \it Q } , \bf { W } _ { \it { i } } ^ { K } \in \mathbb { R } ^ { D \times d _ { k } } , \ b u t { W } _ { \it { i } } ^ { V } \in \mathbb { R } ^ { D \times D }
$$

$$
{ \mathrm { h e a d } } _ { i } = { \mathrm { s o f t m a x } } \left( { \frac { \mathbf { Q } _ { i } \mathbf { K } _ { i } ^ { \top } } { \sqrt { d _ { k } } } } + \mathbf { M } \right) \mathbf { V } _ { i } \in \mathbb { R } ^ { L ^ { \prime } \times D }\tag{9}
$$

$$
{ \mathrm { A t t n } } _ { \mathrm { A } } ( { \mathbf Z } ) = { \mathrm { L a y e r N o r m } } \left( { \mathbf Z } + \sum _ { i = 1 } ^ { N _ { h } } { \mathrm { h e a d } } _ { i } \right)\tag{10}
$$

Type-B (standard multi-head attention with FFN). This is the conventional transformer attention mechanism. The value vector is split across heads, and their outputs are concatenated and passed through a final linear layer and a feed-forward network.

$$
\mathbf { Q } _ { i } = \mathbf { Z } \mathbf { W } _ { i } ^ { Q } , \quad \mathbf { K } _ { i } = \mathbf { Z } \mathbf { W } _ { i } ^ { K } , \quad \mathbf { V } _ { i } = \mathbf { Z } \mathbf { W } _ { i } ^ { V }\tag{11}
$$

$$
\mathrm { w h e r e } \ \mathbf { W } _ { i } ^ { Q } , \mathbf { W } _ { i } ^ { K } , \mathbf { W } _ { i } ^ { V } \in \mathbb { R } ^ { D \times d _ { k } } \ \mathrm { a n d } \ d _ { k } = D / N _ { h }
$$

$$
{ \mathrm { h e a d } } _ { i } = { \mathrm { s o f t m a x } } \left( { \frac { \mathbf { Q } _ { i } \mathbf { K } _ { i } ^ { \top } } { \sqrt { d _ { k } } } } + \mathbf { M } \right) \mathbf { V } _ { i } \in \mathbb { R } ^ { L ^ { \times } \times d _ { k } }\tag{12}
$$

$$
\mathbf { H } _ { \mathrm { c a t } } = { \mathrm { C o n c a t } } ( \mathrm { h e a d } _ { 1 } , \dots , \mathrm { h e a d } _ { N _ { h } } ) \mathbf { W } ^ { O }\tag{13}
$$

$$
\mathbf { Z ^ { \prime } } = \mathrm { L a y e r N o r m } ( \mathbf { Z } + \mathbf { H } _ { \mathrm { c a t } } )\tag{14}
$$

$$
\mathrm { A t t n } _ { \mathrm { B } } ( \mathbf { Z } ) = \mathrm { L a y e r N o r m } ( \mathbf { Z } ^ { \prime } + \mathrm { F F N } ( \mathbf { Z } ^ { \prime } ) )\tag{15}
$$

Hard-Alibi Masking. The mask term M in the attention computation implements hard-alibi masking [18]. Given $N _ { h }$ attention heads, for the first half of the heads $( i < N _ { h } / 2 )$ , the mask M is constructed to be highly restrictive, allowing each token to attend only to itself and the preceding � tokens. For the remaining heads $( i \geq N _ { h } / 2 )$ , M applies a standard causal mask.

3.3.4 Training Objective. During pre-training on the synthetic data, the model is trained autoregressively to predict the next item’s embedding for each position in the query sequences using a sampled softmax loss [51]. This forces the model to learn a general algorithm for sequential pattern inference from the provided context.

Table 1: Dataset statistics

3.3.5 Inference on a New Domain. RecPFN adapts in-context without weight updates. The context selector is a structural component of this framework: the quality of in-context adaptation depends directly on the relevance of the retrieved support set, since it is through these sequences that the model conditions on the target domain’s behavioral dynamics. From a Bayesian perspective, higherquality retrieval yields a support set that better approximates a draw from the posterior over environments, directly improving inference accuracy. For a query $\boldsymbol { S _ { q } } = ( s _ { q , 1 } , \ldots , s _ { q , L _ { q } } )$ , we select $N _ { c }$ support sequences from the target domain dataset via a recency-weighted overlap (i.e. sequences containing recent items from the query are prioritized):

$$
\mathrm { S i m } ( S _ { q } , S _ { j } ) = \sum _ { i \in S _ { q } \cap S _ { j } } w ( i , S _ { q } ) , \quad w ( i , S _ { q } ) = \lambda ^ { L _ { q } - t }\tag{16}
$$

Candidate sequences are retrieved eficiently via an inverted index (built over item-to-sequence mappings) starting from the most recent items in the query until some predefined size $\eta \times N _ { c }$ is reached (we set $\eta \ : = \ : 1 0 )$ , then reranked by $\mathrm { S i m } ( \cdot , \cdot )$ . The top $N _ { c }$ sequences are kept to form $\mathbf { X } c , \mathbf { A }$ single forward pass yields a nextitem embedding $\hat { \mathbf { e } } _ { p r e d }$ per query, and final items are retrieved by eˆ<sub>����</sub> ·e� cosine similarity: Top-K = argmax� | eˆ <sub>�</sub> | | e |

## 4 Experiments

We conduct comprehensive experiments to evaluate RecPFN’s efectiveness. Given its single-pass inference and pretraining on varied synthetic data, we expect RecPFN to perform well in low-compute and low-data regimes. Consequently, we evaluate RecPFN against:

(1) State-of-the-art zero-shot recommendation methods.

(2) Supervised methods trained on the target domain under limited compute.

(3) Supervised methods trained on the target domain with limited data.

Table 2: RecPFN performance against SOTA zero-shot baselines
<table><tr><td>Dataset</td><td colspan="2">RecPFN</td><td colspan="2">RecFormer</td><td colspan="2">RecGPT</td><td colspan="2">UniSRec</td><td colspan="2"> ${ \mathrm { V Q } } { \mathrm { - R e c } }$ </td><td colspan="2">EmbKNN</td><td colspan="2">Improv.</td></tr><tr><td>@10</td><td>HR</td><td>MRR</td><td>HR</td><td>MRR</td><td>HR</td><td>MRR</td><td>HR</td><td>MRR</td><td>HR</td><td>MRR</td><td>HR</td><td>MRR</td><td>HR</td><td>MRR</td></tr><tr><td>Appliances</td><td>.1915</td><td>.1471</td><td>.1250</td><td>.0808</td><td>.1064</td><td>.0993</td><td>.0709</td><td>.0340</td><td>.1206</td><td>.0844</td><td>.1064</td><td>.0483</td><td>+53.2%</td><td>+48.1%</td></tr><tr><td>Arts</td><td>.2077</td><td>.1485</td><td>.1851</td><td>.1393</td><td>.0390</td><td>.0372</td><td>.1469</td><td>.1124</td><td>.1293</td><td>.1199</td><td>.1918</td><td>.1301</td><td>+8.3%</td><td>+6.6%</td></tr><tr><td>Games</td><td>.0673</td><td>.0417</td><td>.0585</td><td>.0283</td><td>.0355</td><td>.0330</td><td>.0239</td><td>.0116</td><td>.0346</td><td>.0302</td><td>.0471</td><td>.0161</td><td>+15.0%</td><td>+26.6%</td></tr><tr><td>Movies</td><td>.1253</td><td>.0708</td><td>.0875</td><td>.0431</td><td>.0416</td><td>.0367</td><td>.0354</td><td>.0212</td><td>.0428</td><td>.0366</td><td>.0774</td><td>.0268</td><td>+43.2%</td><td>+64.2%</td></tr><tr><td>Pantry</td><td>.2491</td><td>.2031</td><td>.1534</td><td>.1004</td><td>.0599</td><td>.0579</td><td>.2069</td><td>.1780</td><td>.1975</td><td>.1853</td><td>.2377</td><td>.1913</td><td>+4.8%</td><td>+6.2%</td></tr><tr><td>Scientific</td><td>.1007</td><td>.0730</td><td>.0827</td><td>.0542</td><td>.0655</td><td>.0595</td><td>.0431</td><td>.0226</td><td>.0735</td><td>.0482</td><td>.0826</td><td>.0480</td><td>+21.7%</td><td>+22.6%</td></tr><tr><td>Software</td><td>.1882</td><td>.1038</td><td>1418</td><td>.0923</td><td>.1096</td><td>.0912</td><td>.1313</td><td>.0906</td><td>.1229</td><td>.0930</td><td>.1448</td><td>.0419</td><td>+30.0%</td><td>+11.6%</td></tr><tr><td>Dianping</td><td>.0574</td><td>.0269</td><td>.0103</td><td>.0088</td><td>.0007</td><td>.0002</td><td>.0016</td><td>.0005</td><td>.0003</td><td>.0002</td><td>.0220</td><td>.0040</td><td>+160.9%</td><td>+204.7%</td></tr><tr><td>Yelp</td><td>.0245</td><td>.0165</td><td>.0163</td><td>.0092</td><td>.0179</td><td>.0171</td><td>.0073</td><td>.0044</td><td>.0186</td><td>.0172</td><td>.0293</td><td>.0152</td><td>-16.4%</td><td>-4.2%</td></tr></table>

Notes: Bold = best, Underlined = second-best, Dotted underline denotes third-best.

## 4.1 Experimental Setup

4.1.1 Datasets. We evaluate on eight public benchmarks for sequential recommendation (Table 1).

The evaluation suite spans: (i) Amazon Review categories in e-commerce (Appliances; Arts, Crafts and Sewing; Movies; Video Games; Industrial and Scientific; Software), (ii) Yelp for local businesses and services (reviews and check-ins), and (iii) Dianping for O2O services in the Chinese market, which difers linguistically and culturally from Yelp and Amazon. Together these datasets cover distinct platforms (marketplaces vs. local services), heterogeneous item semantics, and a wide range of scales and sparsity levels.

For the pre-training-on-real-data ablation (Section 5.1.2), we replace the synthetic prior with four Amazon categories—Fashion, Grocery and Gourmet Food, Sports and Outdoors, and Movies—used only for pre-training. Movies does not appear among the evaluation datasets, avoiding leakage.

4.1.2 Baselines. We compare RecPFN with strong supervised and zero-shot baselines. The supervised set includes GRU4Rec [12], SAS-Rec [20], and BERT4Rec [42] (ID-based), FDSA [60] (embeddingbased), semantic variants of the three ID-based models, and pretrained recommenders UniSRec [16] and VQ-Rec [15] fine-tuned on the target domain. Semantic variants replace the discrete item-ID inputs of ID-based models with the same sentence embeddings used by RecPFN, enabling a direct comparison of the ID-based and embedding-based paradigms under identical training conditions. The foundation/zero-shot set includes RecGPT [19] as well as zero-shot variants of UniSRec, VQ-Rec, and RecFormer [23]. We additionally include EmbKNN as a zero-shot baseline: given a query sequence, EmbKNN retrieves the top-� items by cosine similarity between the query’s most recent item embedding and all catalog item embeddings, without any learned model. This suite spans IDonly, metadata-aware, semantic, LLM-prompted, and pre-trained paradigms.

4.1.3 Evaluation Setings. We report HR@10 and MRR@10 using the full item catalog as the candidate set. Data are split 70/10/20 into train/validation/test; validation and test are each capped at 10,000 samples, with the remainder assigned to training. All methods use identical splits. For fine-tuning of supervised baselines, we repeat each experiment across 4 random seeds and average the results.

## 4.1.4 Implementation Details.

Architecture. RecPFN uses �=6 stacked ICL modules (Section 3.3.2) with alternating Type-A and Type-B attention blocks (Section 3.3.3), $N _ { h } { = } 8$ attention heads, and dropout of 0.2 on attention and residual connections.

Training. Pre-training is performed on a single A100 GPU for 36 hours in two stages: first on the random-graph prior, then on an equal mix of random-graph and latent-factor priors. Each stage runs up to 120 epochs; per epoch we execute 500 training and 200 validation steps. Each step samples one environment and constructs a batch with $N _ { c } { = } 1 2 8$ support sequences and $N _ { q } { = } 1 6$ query sequences. We use AdamW [32] with a learning rate of $1 \times 1 0 ^ { - 4 }$ , gradient accumulation of 2, and 6 warm-up epochs. Early stopping (patience 20) monitors the ratio $\mathrm { M R R } @ 1 0 _ { \mathrm { R e c P F N } } / \mathrm { M R R } @ 1 0 _ { \mathrm { E m b K N N } }$ on the synthetic validation stream to normalize for environment dificulty. Item embeddings are produced by gte-Qwen2-1.5B-instruct [25]. Additional details on synthetic data generation appear in Appendix A.

Evaluation. For evaluation, we use a batch size of 1 with a context size of 8. We set context selector hyperparameters � = � and � = 10.

Baselines. SASRec, GRU4Rec, BERT4Rec, and FDSA are implemented via RecBole [52] with its optimal hyperparameters. All other baselines use their oficial repositories and default hyperparameters for fine-tuning (where applicable) and inference.

## 4.2 Main Results

## We evaluate in three settings:

4.2.1 Zero-shot (Table 2). RecPFN exceeds all zero-shot baselines on all but one dataset, despite being pre-trained solely on synthetic data, whereas competing methods rely on large real interaction corpora. Notably, EmbKNN is often competitive—sometimes matching or beating pre-trained baselines—which underscores the strong context dependence of recommendation and calls into question the efectiveness of domain-agnostic pretraining. RecPFN addresses this by performing single-pass inference while adapting on-the-fly via a small, relevant support set, without weight updates.

Table 3: RecPFN performance against supervised baselines trained for 3 epochs on the full target domain dataset, illustrating deployment eficiency under a per-domain training budget. <sup>∧</sup> denotes the semantic embedding variant of the ID-based models. <sup>∗</sup> indicates statistically significant improvements (p<0.05).
<table><tr><td>Dataset</td><td>@10</td><td>RecPFN</td><td>BERT</td><td>BERT^</td><td>FDSA</td><td>GRU</td><td>GRU^</td><td>SAS</td><td>SAS^</td><td>UniSRec</td><td>VQ-Rec</td><td>Improv.</td><td>Rank</td></tr><tr><td rowspan="2">Appliances</td><td>HR</td><td>.1915*</td><td>.0391</td><td>.0938</td><td>.0836</td><td>.0240</td><td>.0000</td><td>.0697</td><td>.1192</td><td>.1259</td><td>.1242</td><td>+52.1%</td><td>1</td></tr><tr><td>MRR</td><td>.1471*</td><td>.0095</td><td>.0392</td><td>.0235</td><td>.0113</td><td>.0000</td><td>.0655</td><td>.0831</td><td>.0937</td><td>.0982</td><td>+49.8%</td><td>1</td></tr><tr><td rowspan="2">Arts</td><td>HR</td><td>.2077*</td><td>.0396</td><td>.0163</td><td>.0661</td><td>.0264</td><td>.0149</td><td>.0229</td><td>.0484</td><td>.1936</td><td>.1280</td><td>+7.3%</td><td>1</td></tr><tr><td>MRR</td><td>.1485</td><td>.0181</td><td>.0076</td><td>.0458</td><td>.0135</td><td>.0090</td><td>.0094</td><td>.0343</td><td>.1531</td><td>.1183</td><td>-3.0%</td><td>2</td></tr><tr><td rowspan="2">Games</td><td>HR</td><td>.0673*</td><td>.0372</td><td>.0122</td><td>.0396</td><td>.0244</td><td>.0152</td><td>.0169</td><td>.0352</td><td>.0636</td><td>.0343</td><td>+5.9%</td><td>1</td></tr><tr><td>MRR</td><td>.0417*</td><td>.0144</td><td>.0053</td><td>.0165</td><td>.0093</td><td>.0099</td><td>.0070</td><td>.0283</td><td>.0346</td><td>.0300</td><td>+20.5%</td><td>1</td></tr><tr><td rowspan="2">Movies</td><td>HR</td><td>.1253*</td><td>.1022</td><td>.0191</td><td>.1078</td><td>.0984</td><td>.0262</td><td>.0227</td><td>.0064</td><td>.1167</td><td>.0425</td><td>+7.4%</td><td>1</td></tr><tr><td>MRR</td><td>.0708</td><td>.0674</td><td>.0065</td><td>.0814</td><td>.0638</td><td>.0166</td><td>.0083</td><td>.0038</td><td>.0743</td><td>.0365</td><td>-13.0%</td><td>3</td></tr><tr><td rowspan="2">Pantry</td><td>HR</td><td>.2491*</td><td>.0292</td><td>.0340</td><td>.0488</td><td>.0299</td><td>.0577</td><td>.0369</td><td>.0243</td><td>.2360</td><td>.1994</td><td>+5.5%</td><td>1</td></tr><tr><td>MRR</td><td>.2031*</td><td>.0096</td><td>.0140</td><td>.0197</td><td>.0107</td><td>.0346</td><td>.0149</td><td>.0132</td><td>.2006</td><td>1879</td><td>+1.3%</td><td>1</td></tr><tr><td rowspan="2">Scientific</td><td>HR</td><td>.1007*</td><td>.0491</td><td>.0425</td><td>.0527</td><td>.0414</td><td>.0352</td><td>.0488</td><td>.0441</td><td>.0819</td><td>.0719</td><td>+23.0%</td><td>1</td></tr><tr><td>MRR</td><td>.0730*</td><td>.0275</td><td>.0298</td><td>.0298</td><td>.0212</td><td>.0250</td><td>.0265</td><td>.0325</td><td>.0505</td><td>.0488</td><td>+44.5%</td><td>1</td></tr><tr><td rowspan="2">Software</td><td>HR</td><td>.1882</td><td>.0488</td><td>.0596</td><td>.0763</td><td>.0461</td><td>.0509</td><td>.1018</td><td>.1160</td><td>.1816</td><td></td><td>+3.6%</td><td>1</td></tr><tr><td>MRR</td><td>.1038</td><td>.0146</td><td>.0229</td><td>.0344</td><td>.0144</td><td>.0403</td><td>.0815</td><td>.0740</td><td>.1042</td><td>1211 .0920</td><td>-0.4%</td><td>2</td></tr><tr><td rowspan="2">Dianping</td><td>HR</td><td>.0574*</td><td>.0154</td><td>.0004</td><td>.0039</td><td>.0106</td><td>.0001</td><td>.0011</td><td>.0000</td><td>.0068</td><td>.0003</td><td>+273.3%</td><td></td></tr><tr><td>MRR</td><td>.0269*</td><td>.0049</td><td>.0001</td><td>.0012</td><td>.0033</td><td>.0000</td><td>.0003</td><td>.0000</td><td>.0021</td><td>.0002</td><td>+448.0%</td><td>1 1</td></tr><tr><td rowspan="2">Yelp</td><td>HR</td><td>.0245</td><td>.0358</td><td>.0066</td><td>.0293</td><td>.0356</td><td>.0049</td><td>.0191</td><td>.0019</td><td>.0272</td><td>.0181</td><td>-31.5%</td><td></td></tr><tr><td>MRR</td><td>.0165</td><td>.0122</td><td>.0023</td><td>.0105</td><td>.0124</td><td>.0019</td><td>.0066</td><td>.0008</td><td>.0124</td><td>.0168</td><td>-2.1%</td><td>5 2</td></tr></table>

Notes: Bold = best, Underlined = second-best, Dotted underline denotes third-best. BERT = BERT4Rec, SAS = SASRec, GRU = GRU4Rec.

Table 4: RecPFN performance against supervised baselines trained for 50 epochs on 10% of the target domain dataset. <sup>∧</sup> denotes the semantic embedding variant of the ID-based models. <sup>∗</sup> indicates statistically significant improvements (p<0.05)
<table><tr><td>Dataset</td><td>@10</td><td>RecPFN</td><td>BERT</td><td>BERT^</td><td>FDSA</td><td>GRU</td><td>GRU^</td><td>SAS</td><td>SAS^</td><td>UniSRec</td><td>VQ-Rec</td><td>Rank</td></tr><tr><td rowspan="2">Appliances</td><td>HR</td><td>.1348</td><td>.0868</td><td>.0834</td><td>.0960</td><td>.0295</td><td>.0428</td><td>.1221</td><td>.1342</td><td>.1277</td><td>.1277</td><td>1</td></tr><tr><td>MRR</td><td>.1016*</td><td>.0225</td><td>.0513</td><td>.0314</td><td>.0100</td><td>.0214</td><td>.0632</td><td>.0829</td><td>.0931</td><td>.0887</td><td>1</td></tr><tr><td rowspan="2">Arts</td><td>HR</td><td>.1947</td><td>.0531</td><td>.0173</td><td>.1102</td><td>.0720</td><td>.0157</td><td>.0973</td><td>.0658</td><td>.1990</td><td>.1834</td><td>2</td></tr><tr><td>MRR</td><td>.1424</td><td>.0360</td><td>.0060</td><td>.0920</td><td>.0544</td><td>.0096</td><td>.0716</td><td>.0534</td><td>.1610</td><td>.1656</td><td>3</td></tr><tr><td rowspan="2">Pantry</td><td>HR</td><td>.2364</td><td>.0258</td><td>.0296</td><td>.0755</td><td>.0398</td><td>.0405</td><td>.0688</td><td>.0594</td><td>.2357</td><td>.2187</td><td>1</td></tr><tr><td>MRR</td><td>.1990</td><td>.0170</td><td>.0169</td><td>.0583</td><td>.0145</td><td>.0254</td><td>.0506</td><td>.0503</td><td>.1993</td><td>1944</td><td>2</td></tr><tr><td rowspan="2">Scientific</td><td>HR</td><td>.0865</td><td>.0408</td><td>.0606</td><td>.0785</td><td>.0309</td><td>.0322</td><td>.0626</td><td>.0748</td><td>.0936</td><td>.0832</td><td>2</td></tr><tr><td>MRR</td><td>.0650</td><td>.0279</td><td>.0488</td><td>.0653</td><td>.0226</td><td>.0237</td><td>.0368</td><td>.0646</td><td>.0630</td><td>.0551</td><td>2</td></tr><tr><td rowspan="2">Dianping</td><td>HR</td><td>.0275*</td><td>.0054</td><td>.0003</td><td>.0136</td><td>.0058</td><td>.0001</td><td>.0086</td><td>.0001</td><td>.0069</td><td>.0001</td><td>1</td></tr><tr><td>MRR</td><td>.0074*</td><td>.0018</td><td>.0001</td><td>.0055</td><td>.0021</td><td>.0000</td><td>.0022</td><td>.0000</td><td>.0019</td><td>.0000</td><td>1</td></tr><tr><td rowspan="2">Yelp</td><td>HR</td><td>.0191</td><td>.0137</td><td>.0068</td><td>.0324</td><td>.0124</td><td>.0041</td><td>.0258</td><td>.0025</td><td>.0224</td><td>.0363</td><td>5</td></tr><tr><td>MRR</td><td>.0107</td><td>.0044</td><td>.0023</td><td>.0114</td><td>.0041</td><td>.0016</td><td>.0122</td><td>.0006</td><td>.0148</td><td>.0199</td><td>5</td></tr></table>

Notes: Bold = best, Underlined = second-best, Dotted underline denotes third-best. BERT = BERT4Rec, SAS = SASRec, GRU = GRU4Rec.

4.2.2 Deployment Eficiency (Table 3). We evaluate under a budget of 3 training epochs, where supervised baselines receive multiple full passes over the target-domain data. RecPFN, which never updates weights and performs a single forward pass conditioned on a small support set drawn from the train split, still outperforms most baselines across datasets, including pre-trained UniSRec and

VQ-Rec after brief fine-tuning. This comparison reflects RecPFN’s core deployment advantage: it trades a one-time pretraining cost (36 hours on a single A100 GPU) for zero per-domain retraining cost across all future target domains. This amortization becomes increasingly favorable as the number of deployment domains grows.

![](images/b5cde154d89d4bb96f5508864bb7af39e762ae7fb8178fab1d460997daef25cf.jpg)  
Figure 2: MRR@10 versus training epochs (up to 100) on four datasets for SASRec, FDSA, VQ-Rec, and RecPFN. RecPFN requires no target-domain training and is shown as a horizontal line.

4.2.3 Low-data (Table 4). When the train split is truncated to 10%, both the supervised training signal and RecPFN’s context pool shrink. Pre-trained methods become strongest, while ID-based approaches degrade most. Even after 50 epochs of supervised finetuning, RecPFN remains competitive, indicating that its in-context learning more efectively exploits severely limited target data for adaptation than conventional fine-tuning.

## 4.3 Performance versus Training Compute

We measure MRR@10 versus epochs for SASRec, FDSA, and VQ Rec on four representative benchmarks, training each supervised baseline on the full split for up to 100 epochs under identical settings, and evaluating after every epoch. RecPFN needs no target-domain training and thus appears as a horizontal line.

Figure 2 shows RecPFN remains competitive across the entire compute range. Even after 100 epochs, no supervised method outperforms RecPFN on more than two of the four datasets, and match ing RecPFN typically requires substantial per-domain compute. In the early training regime (first 10–20 epochs), RecPFN often leads, making it attractive for rapid deployment and settings with frequent domain shifts where per-domain retraining is costly. These observations confirm that RecPFN achieves strong performance without domain-specific training while supervised methods require significant per-domain compute to match it.

## 5 Analysis and Discussion

## 5.1 Ablation Studies

We analyze RecPFN across four axes: (1) model design, (2) synthetic data priors, (3) inference-time batching, and (4) context construction. Overall, the ablations support our design choices and highlight where further gains are possible.

## 5.1.1 Model Ablations.

Model size. We evaluate two reduced-capacity variants—RecPFNsmall (4 ICL blocks) and RecPFN-mini (2 ICL blocks). Both remain competitive overall but show regressions on some datasets, suggesting that depth helps RecPFN learn a stronger in-context algorithm. Increasing the dificulty and diversity of the synthetic prior (e.g., longer-range dependencies, multi-step motifs, hierarchical patterns) should better saturate capacity and further separate the larger model from smaller variants.

Language model. We train RecPFN with BGE M3 [7], Embedding Gemma-300M [46], and paraphrase-multilingual-mpnet-base-v2 [37]. The multilingual model is included to probe cross-lingual robustness, given that our evaluation suite includes Dianping, a Chinese-market dataset. Smaller encoders cause noticeable regressions on some datasets, but overall performance remains solid, indicating robustness to moderate embedding shifts. Stronger embedding geometry directly benefits recommendation quality, since inference relies on cosine retrieval over the item catalog (Section 3.3.5).

Attention mechanisms. Variants using only Type-A or only Type-B attention (Section 3.3.3) underperform the alternating design. Type-B-only is markedly worse, supporting that Type-A is crucial for eficient copying and sequence-conditioned retrieval over flattened context. Alternation balances feature extraction (Type-B) and copying-style operations (Type-A), stabilizing training and yielding the best generalization.

## 5.1.2 Synthetic Data .

Pre-training on real-world data. Replacing the synthetic SCM prior with large-scale real datasets (Amazon Fashion, Grocery, Movies, Sports) significantly lowers performance across evaluation domains, consistent with our claim that recommendation is highly context-dependent: domain-agnostic pre-training on real interactions fails to generalize zero-shot, whereas the synthetic prior trains amortized inference rather than memorizing domain idiosyncrasies.

Prior composition. Training exclusively on the random-graph prior is generally poorer than the mixed-prior setting, indicating that the latent-factor prior is essential for capturing factor-driven coconsumption and non-trivial item–item dependencies. Conversely, training solely on the latent-factor prior did not converge reliably due to its higher dificulty, motivating our two-stage curriculum: learn robust sequence-processing primitives on the random-graph prior, then introduce latent-factor dynamics without destabilizing training.

5.1.3 Inference: Batch size. Scaling batch size and context proportionally degrades performance, suggesting that cross-attention over broad contexts dilutes attention budgets and increases retrieval noise. Improving robustness at larger contexts is a promising direction—e.g., hierarchical context indexing, locality-aware routing, gating, or continued pre-training on longer-context regimes.

## 5.1.4 Context.

Context size. Removing the support set causes drastic drops across all datasets, confirming that RecPFN’s gains stem from genuine in-context adaptation. Random context also causes drastic regression, showing context does not merely have a regularization efect. Small context sizes remain close to peak, while larger contexts introduce slight degradations, likely from reduced signalto-noise.

Table 5: Relative performance of various ablation experiments for RecPFN (expressed as percentage change)
<table><tr><td>Experiment</td><td colspan="2">Appliances</td><td colspan="2">Arts</td><td colspan="2">Pantry</td><td colspan="2">Scientific</td><td colspan="2">Dianping</td><td colspan="2">Yelp</td></tr><tr><td>@10</td><td>HR</td><td>MRR</td><td>HR</td><td>MRR</td><td>HR</td><td>MRR</td><td>HR</td><td>MRR</td><td>HR</td><td>MRR</td><td>HR</td><td>MRR</td></tr><tr><td>RecPFN-base</td><td>.1915</td><td>.1471</td><td>.2077</td><td>.1485</td><td>.2491</td><td>.2031</td><td>.1007</td><td>.0730</td><td>.0574</td><td>.0269</td><td>.0245</td><td>.0165</td></tr><tr><td colspan="9">Model</td><td></td><td></td><td></td><td></td></tr><tr><td>RecPFN-small RecPFN-mini</td><td>+3.7 +3.7</td><td>-26.6 -12.8</td><td>+0.5 +0.5</td><td>+0.7 -0.8</td><td>-1.0 -1.3</td><td>+0.3 +0.2</td><td>+1.9 +4.8</td><td>-1.0 -14.7</td><td>-17.8 -25.1</td><td>-29.2 -40.7</td><td>-9.0 +2.0</td><td>-10.0 +4.5</td></tr><tr><td>baai/bge-m3</td><td>-14.8</td><td>-20.9</td><td>+2.5</td><td>+0.4</td><td>+1.2</td><td>+1.1</td><td>+4.4</td><td>-4.6</td><td>+17.6</td><td>+16.2</td><td>+2.0</td><td>+0.1</td></tr><tr><td>embeddinggemma-300m multilingual-mpnet-base-v2</td><td>+3.7 -11.1</td><td>-18.0 -32.8</td><td>-2.0 -6.6</td><td>-0.9 -2.0</td><td>-0.7 -3.4</td><td>+0.4 -1.1</td><td>+4.2 -15.4</td><td>-12.9 -27.0</td><td>+24.0 -69.7</td><td>+31.0 -88.3</td><td>-64.9 -44.9</td><td>-72.8 -52.3</td></tr><tr><td>only type-A attention</td><td>+0.0</td><td>-19.7</td><td>+0.5</td><td>+0.8</td><td>-0.3</td><td>+0.5</td><td>+0.0</td><td>-3.0</td><td>-7.0</td><td>-12.3</td><td>-5.3</td><td>-3.7</td></tr><tr><td>only type-B attention</td><td>-18.5</td><td>-22.4</td><td>-5.8</td><td>-5.7</td><td>-0.4</td><td>-1.4</td><td>+6.5</td><td>-12.6</td><td>-61.0</td><td>-83.0</td><td>+8.6</td><td>+18.4</td></tr><tr><td colspan="9">Synthetic Data</td><td></td><td></td><td></td></tr><tr><td>amazon dataset</td><td>-14.8</td><td>-37.5</td><td>-2.5</td><td>-3.0</td><td>-0.3</td><td>-0.1</td><td>-7.1</td><td>-26.0</td><td>-42.9</td><td>-59.3</td><td>-13.9</td><td>-25.7</td></tr><tr><td>only random graph prior</td><td>+0.0</td><td>-11.3</td><td>+0.8</td><td>+1.5</td><td>-0.4</td><td>+0.1</td><td>-0.2</td><td>-1.5</td><td>-19.5</td><td>-32.2</td><td>-12.2</td><td>-10.7</td></tr><tr><td colspan="9">Inference</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>-4.4</td><td>-15.0</td><td>-14.1</td><td>-11.6</td><td>-11.2</td><td>-9.4</td><td>-8.1</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="9">batch size = 2 batch size = 4</td><td>-43.6 -64.0</td><td></td><td>-18.8</td><td>-19.8</td></tr><tr><td></td><td>-20.2</td><td>-21.3</td><td>-28.4</td><td>-24.6</td><td>-17.9</td><td>-14.4</td><td>-14.1</td><td>-6.5 -11.8</td><td>-59.8</td><td>-78.2</td><td>-29.4</td><td>-33.8</td></tr><tr><td>batch size = 8</td><td>-16.6</td><td>-18.0</td><td>-30.9</td><td>-27.4</td><td>-16.3</td><td>-12.0</td><td>-20.4</td><td>-17.1</td><td>-65.5</td><td>-84.0</td><td>-39.6</td><td>-43.4</td></tr><tr><td colspan="9">Context</td><td></td><td></td><td></td><td></td></tr><tr><td>no context</td><td>-74.1</td><td>-86.3</td><td>-69.3</td><td>-71.5</td><td>-38.2</td><td>-44.4</td><td>-82.0</td><td>-85.2</td><td>-69.3</td><td>-87.9</td><td>-86.9</td><td>-89.8</td></tr><tr><td>random context</td><td>-59.3</td><td>-70.8</td><td>-64.9</td><td>-72.7</td><td>-39.4</td><td>-47.1</td><td>-67.5</td><td>-66.9</td><td>-71.3</td><td>-89.8</td><td>-88.2</td><td>-94.5</td></tr><tr><td>context size = 2</td><td>+0.0</td><td>+4.2</td><td>+0.5</td><td>+0.2</td><td>-0.7</td><td>+0.2</td><td>+3.4</td><td>-4.2</td><td>+1.7</td><td>+5.2</td><td>+5.7</td><td>+2.7</td></tr><tr><td>context size = 4</td><td>+3.7</td><td>+0.5</td><td>+0.8</td><td>+0.1</td><td>-0.1</td><td>+0.1</td><td>+2.8</td><td>-0.6</td><td>+5.2</td><td>+10.8</td><td>+2.0</td><td>+5.2</td></tr><tr><td>context size = 16 context size = 32</td><td>+3.7</td><td>-4.4</td><td>+0.0</td><td>-0.5</td><td>-0.4</td><td>-0.1</td><td>-1.6</td><td>-2.3</td><td>-15.5</td><td>-25.0</td><td>-2.4</td><td>-6.8</td></tr><tr><td></td><td>-3.7</td><td>-9.1</td><td>-0.9</td><td>-1.0</td><td>-0.7</td><td>-0.2</td><td>-1.6</td><td>-2.5</td><td>-39.2</td><td>-57.7</td><td>-7.8</td><td>-17.1</td></tr><tr><td>10% context selection dataset</td><td>-29.6</td><td>-30.9</td><td>-6.3</td><td>-4.1</td><td>-5.1</td><td>-2.0</td><td>-14.1</td><td>-10.9</td><td>-52.1</td><td>-72.3</td><td>-22.0</td><td>-35.1</td></tr><tr><td>λ = 1 (unweighted similarity score) λ = e2</td><td>+0.0 -3.7</td><td>-7.8</td><td>-0.5</td><td>-0.3</td><td>-0.8</td><td>-0.3</td><td>-1.1</td><td>-0.4</td><td>-36.8</td><td>-52.1</td><td>-5.7</td><td>-11.4</td></tr><tr><td>η=3</td><td></td><td>+0.2</td><td>-0.5</td><td>-0.2</td><td>+0.3</td><td>+0.0</td><td>+0.9</td><td>-0.2</td><td>+1.2</td><td>-0.8</td><td>+0.0</td><td>+1.2</td></tr><tr><td>η= 50</td><td>-3.7 +0.0</td><td>-0.6</td><td>-0.6</td><td>-0.3</td><td>-0.3</td><td>+0.1</td><td>-0.4 +0.0</td><td>-1.5 +0.2</td><td>-36.9 +12.4</td><td>-52.7 +23.5</td><td>-9.4</td><td>-11.9 +2.3</td></tr><tr><td></td><td></td><td>+1.0</td><td>-0.2</td><td>-0.1</td><td>+0.1</td><td>+0.0</td><td></td><td></td><td></td><td></td><td>+2.4</td><td></td></tr></table>

Notes: Bold = best, Underlined = second-best, Dotted underline denotes third-best.

Limited context-selection pool. Constraining the candidate pool for context selection causes significant degradation, consistent with the need for a suficiently large pool to find high-similarity exemplars. Nevertheless, as in our low-data experiments (Section 4.2.3), RecPFN remains competitive against fully trained supervised baselines when both training data and context pool are limited.

Context selector. Replacing the temporally weighted similarity with an unweighted Jaccard-style overlap degrades performance across all datasets, underscoring the importance of recency and order - consistent with our sequence-centric synthetic generation (Section 3.2) and hard-alibi masking (Section 3.3.3). Increasing the decay rate does not significantly impact performance. However, reducing the candidate pool degrades performance, particularly for the largest datasets Dianping and Yelp while increasing it yields significant gains for these datasets.

Remarks on context ablations. Together, these ablations establish the context selector as a structural bottleneck: removing context, degrading retrieval quality, or shrinking the candidate pool all cause substantial drops, while the model is insensitive to moderate retrieval noise. This asymmetry reflects the Bayesian framing: the model exploits whatever signal is in the support set, but cannot compensate for its absence.

Table 6: RecPFN runtimes in seconds against baselines for Pantry and Yelp. Times reported for single epoch training, inference per user (averaged across full test split)
<table><tr><td>Method</td><td>Preprocessing (s)</td><td>Training (s) Inference (ms)</td></tr><tr><td>Yelp</td><td></td><td></td></tr><tr><td>RecPFN</td><td>46.56</td><td>6.58</td></tr><tr><td>RecGPT</td><td></td><td>40.81</td></tr><tr><td>VQ-Rec</td><td>691.20</td><td>380.69 0.44</td></tr><tr><td>FDSA</td><td>一</td><td>1016.30 2.49</td></tr><tr><td>Scientific</td><td></td><td></td></tr><tr><td>RecPFN</td><td>10.05</td><td>5.59</td></tr><tr><td>RecGPT</td><td></td><td>37.35</td></tr><tr><td>VQ-Rec</td><td>401.94</td><td>199.32 0.46</td></tr><tr><td>FDSA</td><td></td><td>44.17 1.61</td></tr></table>

## 5.2 Eficiency Analysis

Table 6 shows the single epoch training runtimes as well as the inference time per user for the best performing baselines from the diferent categories. For inference, the runtimes are obtained by averaging across the full test split with batch size of 96 for the baselines and 8 for RecPFN. We additionally record preprocessing runtimes for RecPFN (building context selector) and VQ-Rec (building item codes). We omit embedding generation for all algorithms.

RecPFN requires a single pre-training phase and no domainspecific fine-tuning, so deployment compute is dominated by inference. While its inference time is noticeably higher than that of compact supervised baselines, it remains far more eficient than competitive zero-shot methods. This makes RecPFN particularly attractive in settings with frequent distribution shifts where retraining would otherwise be required.

## 5.3 Qualitative Analysis: In-Context Adaptation in Action

5.3.1 Case Study on Amazon Movies example. We illustrate how RecPFN leverages in-context examples to make semantically aligned and sequence-consistent predictions. Table 7 presents a representative case from Amazon Movies where the ground-truth next item is a faith/family film. RecPFN, conditioning on a small set of retrieved support sequences (Table 8), correctly predicts the next item; ablating the context or comparing against EmbKNN demonstrates the importance of in-context adaptation.<sup>1</sup>

This case highlights the mechanism by which RecPFN adapts: the query attends over a compact, high-similarity support set that encodes domain-specific sequential logic (e.g., consistent co-consumption of faith/family titles and the observed transition from In the Arms ofAngels to A Pioneer Miracle). Baselines that lack such on-the-fly conditioning either drift toward globally popular titles or produce semantically plausible but context-mismatched recommendations.

5.3.2 Embedding-space visualization (UMAP). We visualize the label, predictions, recent history, and context tokens in UMAP space.

Table 7: Qualitative example (Amazon Movies). RecPFN correctly predicts the ground-truth next item when conditioned on the retrieved support sequences; removing the context or using EmbKNN yields plausible but incorrect recommendations.
<table><tr><td>Query history</td><td>Last Vegas; The Big Wedding (Digital); Life of Pi; Poverty, Inc.; Intolerable Cruelty; In the Arms of Angels</td></tr><tr><td>Ground truth</td><td>A Pioneer Miracle</td></tr><tr><td>RecPFN Top-3</td><td>A Pioneer Miracle; The Teacher; Savior</td></tr><tr><td>RecPFN (no con- text) Top-3</td><td>Panic in the Streets (VHS); Frightworld; A Tale of Two Cities</td></tr><tr><td>EmbKNN Top-3</td><td>Silver Linings Playbook; Movies He&#x27;ll Love; City by the Sea</td></tr><tr><td>Rationale</td><td>RecPFN observes support sequences containing the transition: In the Arms of Angels → A Pioneer Miracle (Table 8). This in-context signal steers the model toward the correct next item. Without context, predictions revert to broadly similar or popular items.</td></tr></table>

Table 8: Most influential support sequences retrieved by the context selector (titles shown). These sequences contain the co-occurrence and transition indicative of the ground-truth next item.
<table><tr><td>1</td><td>Secrets of Archeology: The Lost Cities of the Maya; Seven Girlfriends; In the Arms of Angels; A Pioneer Miracle</td></tr><tr><td>2</td><td>About Miracles; Doctor Thorne (Season 1); In the Arms of Angels; A Pioneer Miracle</td></tr><tr><td>3</td><td>In the Arms of Angels; A Pioneer Miracle; Touched By Grace; Before All Others</td></tr><tr><td>4</td><td>Dying of the Light; In the Arms of Angels; Finding Neverland; A Pioneer Miracle</td></tr></table>

![](images/91c08c9f2d27615f8857d7c971546d742c6c01ddd4176d60eef5440d3316cf46.jpg)  
Figure 3: UMAP projection for the Amazon Movies case in Table 7. Dense, relevant context near the label (star) pulls RecPFN’s prediction (blue diamond) toward the ground truth, as indicated by the arrow from the no-context prediction. Scattered context is ignored. Axes: UMAP-1 and UMAP-2.

For clarity, we 1) plot user history with emphasis on more recent items, 2) jitter context sequence points to avoid occlusion and 3) zoom in to focus on the neighborhood around the predictions and ground truth point, omitting some outlying context and user points.

![](images/fac0036d20347c22cdb31730e576c6d184ea1128f81faba73f11ebe1d32e50b9.jpg)  
Figure 4: MRR@10 relative to the in-range median as each SDG hyperparameter is swept inside and outside its training range (shaded). Results are averaged over 50 synthetic environments per value; shaded bands show ±1 std. RG = random-graph prior; LF = latent-factor prior.

In this case, multiple context sequences encode the transition In the Arms ofAngels → A Pioneer Miracle. This creates a a dense cluster of context points aligns with the ground-truth label. We see that this steers RecPFN toward the ground truth label (green arrow), while RecPFN correctly ignores scattered context points elsewhere. Without context, the prediction drifts to semantically plausible but non-specific regions; EmbKNN similarly favors broadly popular titles.

## 5.4 Prior Hyperparameter Sensitivity

A natural question is whether RecPFN’s performance is brittle to the choice of synthetic prior: does it degrade sharply when evaluated on environments whose hyperparameters fall outside the training distribution? To assess this, we fix the trained model and sweep each SDG hyperparameter across a range spanning both inside and outside its training values, generating 50 independent synthetic environments per setting and reporting MRR@10.

Figure 4 summarises the results. Performance is stable across all parameters within the training range. Outside it, degradation is generally gradual. Two parameters deserve particular mention. For popularity bias $( \alpha _ { \mathrm { p o p } } )$ , out-of-range values actually increase MRR@10: a high popularity bias concentrates sequences onto a small set of popular items, making next-item prediction trivially easier — this reflects a ceiling efect in the synthetic environments rather than a genuine robustness gain. For graph out-degree $( k _ { \mathrm { r e l } } ) _ { \mathrm { : } }$ the sharpest drop occurs at large values, where two efects compound: the transition structure becomes harder to infer from limited context, and the next-item distribution is inherently more difuse, reducing the maximum achievable MRR@10 regardless of model quality. Overall, the results confirm that the prior provides adequate coverage for the evaluation domains, and that degradation outside it is principled rather than catastrophic.

## 6 Conclusion

We presented RecPFN, a prior-fitted, in-context learner for sequential recommendation that performs next-item prediction in a single forward pass while adapting to a new domain purely through a small set of support sequences. By pre-training on a broad synthetic prior defined by structural causal models, RecPFN amortizes Bayesian-style inference over diverse environments, avoiding domain-specific fine-tuning and large-scale real-data pre-training. Across eight public benchmarks, RecPFN achieves state-of-the-art zero-shot results and remains competitive with supervised methods in low-compute and low-data regimes, despite relying only on concise in-context adaptation at inference.

Our analysis highlights several design choices that enable these gains. Alternating attention blocks balance copying-style retrieval and feature extraction; hard-alibi masking focuses computation on recent context; and a two-stage curriculum over synthetic priors stabilizes training while capturing factor-driven dynamics. Ablations further show that replacing the synthetic prior with real interactions reduces transfer, underscoring the importance of training for generalizable inference rather than memorization.

Limitations point to clear avenues for improvement. Performance degrades when contexts become large or poorly selected, suggesting hierarchical or locality-aware routing, improved retrieval, and longer-context pre-training. Embedding quality also matters, motivating stronger or multimodal encoders and tighter alignment between training and deployment representations. Future work includes (i) richer synthetic priors with long-range and hierarchical motifs and slate/session structure; (ii) structured retrieval with jointly trained selectors to preserve signal-to-noise at larger contexts; (iii) longer-context pre-training and ICL scaling laws; (iv) stronger or multimodal encoders with contrastive alignment; and (v) analyses of amortized posterior quality, calibration, and failure modes under catalog churn and domain shift, alongside real-world A/B tests.

## A Synthetic data generation hyperparameters

This appendix details the sampling ranges of the environment hyperparameters for synthetic data pre-training.

Generic hyperparameters. We sample the popularity mixture $\alpha _ { \mathrm { p o p } } \sim \mathcal { U } ( 0 , 0 . 5 )$ , the context window size � $\in \ \{ 1 , 2 , 3 \}$ , and set recency weights $w _ { t } = \omega ^ { t }$ with $\omega \sim \mathcal { U } ( 0 . 5 , 0 . 8 )$ . The catalog size is �=500 in stage 1 and �=1000 in stage 2.

Random-graph prior. We sample $k _ { \mathrm { r e l } } ~ \in ~ \{ 1 , 3 , 5 , 7 \}$ and $\mathit { p } _ { \mathrm { s e l f } } \sim$ U(0.1, 0.3), and construct T as in Section 3.2.2.

Latent-factor prior. We sample $D _ { \mathrm { l a t e n t } } \in \{ 2 0 , 4 0 , 6 0 \} , k _ { \mathrm { f a c } } \in \{ 2 , 3 , 5 \}$ and $p _ { \mathrm { s e l f } } \sim \mathcal { U } ( 0 . 1 , 0 . 3 )$ . We then build F, L<sub>embed</sub>, and T<sub>latent</sub> (Section 3.2.2), and project to item-level $\mathbf { E } = \mathbf { F } \mathbf { L } _ { \mathrm { e m b e d } }$ and $\mathbf { T } = \mathbf { F T } _ { \mathrm { l a t e n t } } \mathbf { F } ^ { \top }$ (with diagonal set by $ { p _ { \mathrm { s e l f } } } )$

## References

[1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. 2023. Gpt-4 technical report. arXiv preprint arXiv:2303.08774 (2023).

[2] Nino Antulov-Fantulin, Matko Bošnjak, Vinko Zlatić, Miha Grčar, and Tomislav Šmuc. 2014. Synthetic sequence generator for recommender systems–memory biased random walk on a sequence multilayer network. In International Conference on Discovery Science. Springer, 25–36.

[3] Alan Arazi, Eilam Shapira, and Roi Reichart. [n. d.]. TabSTAR: A Tabular Foundation Model for Tabular Data with Text Fields. In The Thirty-ninth Annual Conference on Neural Information Processing Systems.

[4] Keqin Bao, Jizhi Zhang, Wenjie Wang, Yang Zhang, Zhengyi Yang, Yanchen Luo, Chong Chen, Fuli Feng, and Qi Tian. 2025. A bi-step grounding paradigm for large language models in recommendation systems. ACM Transactions on Recommender Systems 3, 4 (2025), 1–27.

[5] Francois Belletti, Karthik Lakshmanan, Walid Krichene, Yi-Fan Chen, and John Anderson. 2019. Scalable realistic recommendation datasets through fracta expansions. arXiv preprint arXiv:1901.08910 (2019).

[6] Hao Chen, Yuanchen Bei, Qijie Shen, Yue Xu, Sheng Zhou, Wenbing Huang, Feiran Huang, Senzhang Wang, and Xiao Huang. 2024. Macro graph neural networks for online billion-scale recommender systems. In Proceedings of the ACM web conference 2024. 3598–3608.

[7] Jianlv Chen, Shitao Xiao, Peitian Zhang, Kun Luo, Defu Lian, and Zheng Liu. 2024. Bge m3-embedding: Multi-lingual, multi-functionality, multi-granularity text embeddings through self-knowledge distillation. arXiv preprint arXiv:2402.03216 (2024).

[8] HAO DING, Anoop Deoras, Bernie Wang, and Hao Wang. 2022. Zero-Shot Recommender Systems. In ICLR Workshop on Deep Generative Models for Highly Structured Data.

[9] Qingxiu Dong, Lei Li, Damai Dai, Ce Zheng, Jingyuan Ma, Rui Li, Heming Xia, Jingjing Xu, Zhiyong Wu, Baobao Chang, et al. 2024. A survey on in-context learning. In Proceedings ofthe 2024 conference on empirical methods in natural language processing. 1107–1128.

[10] Nelson Elhage, Neel Nanda, Catherine Olsson, Tom Henighan, Nicholas Joseph, Ben Mann, Amanda Askell, Yuntao Bai, Anna Chen, Tom Conerly, et al. 2021. A mathematical framework for transformer circuits. Transformer Circuits Thread 1, 1 (2021), 12.

[11] Liam Hebert, Marialena Kyriakidi, Hubert Pham, Krishna Sayana, James Pine, Sukhdeep Sodhi, and Ambarish Jash. 2025. FLARE: Fusing Language Models and Collaborative Architectures for Recommender Enhancement. In Companion Proceedings ofthe ACM on Web Conference 2025. 2235–2244.

[12] Balázs Hidasi, Alexandros Karatzoglou, Linas Baltrunas, and Domonkos Tikk. 2015. Session-based recommendations with recurrent neural networks. arXiv preprint arXiv:1511.06939 (2015).

[13] Noah Hollmann, Samuel Müller, Katharina Eggensperger, and Frank Hutter. 2022. Tabpfn: A transformer that solves small tabular classification problems in a second. arXiv preprint arXiv:2207.01848 (2022).

[14] Noah Hollmann, Samuel Müller, Lennart Purucker, Arjun Krishnakumar, Max Körfer, Shi Bin Hoo, Robin Tibor Schirrmeister, and Frank Hutter. 2025. Accurate predictions on small data with a tabular foundation model. Nature 637, 8045 (2025), 319–326.

[15] Yupeng Hou, Zhankui He, Julian McAuley, and Wayne Xin Zhao. 2023. Learning vector-quantized item representation for transferable sequential recommenders. In Proceedings ofthe ACM Web Conference 2023. 1162–1171.

[16] Yupeng Hou, Shanlei Mu, Wayne Xin Zhao, Yaliang Li, Bolin Ding, and Ji-Rong Wen. 2022. Towards universal sequence representation learning for recommender systems. In Proceedings ofthe 28th ACMSIGKDD conference on knowledge discovery and data mining. 585–593.

[17] Andreea Iana, Mehwish Alam, and Heiko Paulheim. 2024. A survey on knowledgeaware news recommender systems. Semantic Web 15, 1 (2024), 21–82.

[18] Samy Jelassi, David Brandfonbrener, Sham M Kakade, and Eran Malach. 2024. Repeat After Me: Transformers are Better than State Space Models at Copying. In International Conference on Machine Learning. PMLR, 21502–21521.

[19] Yangqin Jiang, Xubin Ren, Lianghao Xia, Da Luo, Kangyi Lin, and Chao Huang. 2025. RecGPT: A Foundation Model for Sequential Recommendation. arXiv preprint arXiv:2506.06270 (2025).

[20] Wang-Cheng Kang and Julian McAuley. 2018. Self-attentive sequential recommendation. In 2018 IEEE international conference on data mining (ICDM). IEEE, 197–206.

[21] Safia Kanwal, Sidra Nawaz, Muhammad Kamran Malik, and Zubair Nawaz. 2021. A review of text-based recommendation systems. IEEE access 9 (2021), 31638– 31661.

[22] Sein Kim, Hongseok Kang, Seungyoon Choi, Donghyun Kim, Minchul Yang, and Chanyoung Park. 2024. Large language models meet collaborative filtering: An eficient all-round llm-based recommender system. In Proceedings ofthe 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining. 1395–1406.

[23] Jiacheng Li, Ming Wang, Jin Li, Jinmiao Fu, Xin Shen, Jingbo Shang, and Julian McAuley. 2023. Text is all you need: Learning language representations for sequential recommendation. In Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining. 1258–1267.

[24] Yunzhe Li, Junting Wang, Hari Sundaram, and Zhining Liu. 2025. LLM-RecG: A Semantic Bias-Aware Framework for Zero-Shot Sequential Recommendation. In Proceedings ofthe Nineteenth ACM Conference on Recommender Systems. 237–246.

[25] Zehan Li, Xin Zhang, Yanzhao Zhang, Dingkun Long, Pengjun Xie, and Meishan Zhang. 2023. Towards general text embeddings with multi-stage contrastive learning. arXiv preprint arXiv:2308.03281 (2023).

[26] Yueqing Liang, Liangwei Yang, Chen Wang, Xiongxiao Xu, Philip S Yu, and Kai Shu. 2025. Taxonomy-guided zero-shot recommendations with llms. In Proceedings of the 31st International Conference on Computational Linguistics.

1520–1530.

[27] Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, et al. 2024. Deepseek-v3 technical report. arXiv preprint arXiv:2412.19437 (2024).

[28] Jiahao Liu, Xueshuo Yan, Dongsheng Li, Guangping Zhang, Hansu Gu, Peng Zhang, Tun Lu, Li Shang, and Ning Gu. 2025. Improving llm-powered recommendations with personalized information. In Proceedings of the 48th International ACM SIGIR Conference on Research and Development in Information Retrieval. 2560–2565.

[29] Qidong Liu, Xian Wu, Wanyu Wang, Yejing Wang, Yuanshao Zhu, Xiangyu Zhao, Feng Tian, and Yefeng Zheng. 2025. Llmemb: Large language model can be a good embedding generator for sequential recommendation. In Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 39. 12183–12191.

[30] Qijiong Liu, Jieming Zhu, Yanting Yang, Quanyu Dai, Zhaocheng Du, Xiao-Ming Wu, Zhou Zhao, Rui Zhang, and Zhenhua Dong. 2024. Multimodal pretraining, adaptation, and generation for recommendation: A survey. In Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining. 6566–6576.

[31] Yixin Liu, Kai Zhang, Yuan Li, Zhiling Yan, Chujie Gao, Ruoxi Chen, Zhengqing Yuan, Yue Huang, Hanchi Sun, Jianfeng Gao, et al. 2024. Sora: A review on background, technology, limitations, and opportunities of large vision models. arXiv preprint arXiv:2402.17177 (2024).

[32] Ilya Loshchilov, Frank Hutter, et al. 2017. Fixing weight decay regularization in adam. arXiv preprint arXiv:1711.05101 5, 5 (2017), 5.

[33] Hanjia Lyu, Song Jiang, Hanqing Zeng, Yinglong Xia, Qifan Wang, Si Zhang, Ren Chen, Chris Leung, Jiajie Tang, and Jiebo Luo. 2024. Llm-rec: Personalized recommendation via prompting large language models. In Findings ofthe Association for Computational Linguistics: NAACL 2024. 583–612.

[34] Junwei Ma, Valentin Thomas, Rasa Hosseinzadeh, Hamidreza Kamkari, Alex Labach, Jesse C Cresswell, Keyvan Golestan, Guangwei Yu, Maksims Volkovs, and Anthony L Caterini. 2024. Tabdpt: Scaling tabular foundation models. arXiv preprint arXiv:2410.18164 (2024).

[35] Jingang Qu, David Holzmüller, Gaël Varoquaux, and Marine Le Morvan. 2025. Tabicl: A tabular foundation model for in-context learning on large data. arXiv preprint arXiv:2502.05564 (2025).

[36] Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen, and Ilya Sutskever. 2021. Zero-shot text-to-image generation. In International conference on machine learning. Pmlr, 8821–8831.

[37] Nils Reimers and Iryna Gurevych. 2019. Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. In Proceedings ofthe 2019 Conference on Empirical Methods in Natural Language Processing. Association for Computational Linguistics. http://arxiv.org/abs/1908.10084

[38] Wafa Shafqat and Yung-Cheol Byun. 2022. A hybrid GAN-based approach to solve imbalanced data problem in recommendation systems. IEEE access 10 (2022), 11036–11047.

[39] Kartik Sharma, Yeon-Chang Lee, Sivagami Nambi, Aditya Salian, Shlok Shah, Sang-Wook Kim, and Srijan Kumar. 2024. A survey of graph neural networks for social recommender systems. Comput. Surveys 56, 10 (2024), 1–34.

[40] Leheng Sheng, An Zhang, Yi Zhang, Yuxin Chen, Xiang Wang, and Tat-Seng Chua. 2025. Language Representations Can be What Recommenders Need: Findings and Potentials. In ICLR.

[41] Elizaveta Stavinova, Alexander Grigorievskiy, Anna Volodkevich, Petr Chunaev, Klavdiya Bochenina, and Dmitry Bugaychenko. 2022. Synthetic data-based simulators for recommender systems: A survey. arXiv preprint arXiv:2206.11338 (2022).

[42] Fei Sun, Jun Liu, Jian Wu, Changhua Pei, Xiao Lin, Wenwu Ou, and Peng Jiang. 2019. BERT4Rec: Sequential recommendation with bidirectional encoder rep resentations from transformer. In Proceedings of the 28th ACM international conference on information and knowledge management. 1441–1450.

[43] Yong Kiam Tan, Xinxing Xu, and Yong Liu. 2016. Improved recurrent neural networks for session-based recommendations. In Proceedings ofthe 1st workshop on deep learning for recommender systems. 17–22.

[44] Jiaxi Tang and Ke Wang. 2018. Personalized top-n sequential recommendation via convolutional sequence embedding. In Proceedings ofthe eleventh ACM international conference on web search and data mining. 565–573.

[45] Alejandro Valencia-Arias, Hernán Uribe-Bedoya, Juan David González-Ruiz, Gustavo Sánchez Santos, Edgard Chapoñan Ramírez, and Ezequiel Martínez Rojas. 2024. Artificial intelligence and recommender systems in e-commerce. Trends and research agenda. Intelligent Systems with Applications 24 (2024), 200435.

[46] Henrique Schechter Vera, Sahil Dua, Biao Zhang, Daniel Salz, Ryan Mullins, Sindhu Raghuram Panyam, Sara Smoot, Iftekhar Naim, Joe Zou, Feiyang Chen, et al. 2025. Embeddinggemma: Powerful and lightweight text representations. arXiv preprint arXiv:2509.20354 (2025).

[47] Junting Wang, Adit Krishnan, Hari Sundaram, and Yunzhe Li. 2023. Pre-trained neural recommenders: A transferable zero-shot framework for recommendation systems. arXiv preprint arXiv:2309.01188 (2023).

[48] Junting Wang, Praneet Rathi, and Hari Sundaram. 2024. A pre-trained zero-shot sequential recommendation framework via popularity dynamics. In Proceedings ofthe 18th ACM Conference on Recommender Systems. 433–443.

[49] Lei Wang and Ee-Peng Lim. 2024. The whole is better than the sum: Using aggregated demonstrations in in-context learning for sequential recommendation. arXiv preprint arXiv:2403.10135 (2024).

[50] Wenjie Wang, Xinyu Lin, Fuli Feng, Xiangnan He, Min Lin, and Tat-Seng Chua. 2022. Causal representation learning for out-of-distribution recommendation. In Proceedings ofthe ACM Web Conference 2022. 3562–3571.

[51] Jiancan Wu, Xiang Wang, Xingyu Gao, Jiawei Chen, Hongcheng Fu, and Tianyu Qiu. 2024. On the efectiveness of sampled softmax loss for item recommendation. ACM Transactions on Information Systems 42, 4 (2024), 1–26.

[52] Lanling Xu, Zhen Tian, Gaowei Zhang, Junjie Zhang, Lei Wang, Bowen Zheng, Yifan Li, Jiakai Tang, Zeyu Zhang, Yupeng Hou, et al. 2023. Towards a more user-friendly and easy-to-use benchmark library for recommender systems. In Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval. 2837–2847.

[53] Bencheng Yan, Shilei Liu, Zhiyuan Zeng, Zihao Wang, Yizhen Zhang, Yujin Yuan, Langming Liu, Jiaqi Liu, Di Wang, Wenbo Su, et al. 2025. Unlocking Scaling Law in Industrial Recommendation Systems with a Three-step Paradigm based Large User Model. CoRR (2025).

[54] Jian’en Yan, Haihui Huang, Kairan Yang, Haiyan Xu, and Yanling Li. 2025. Synthetic data for enhanced privacy: A VAE-GAN approach against membership inference attacks. Knowledge-Based Systems 309 (2025), 112899.

[55] Haowei Yang, Yushang Zhao, Sitao Min, Bo Su, Chao Yao, and Wei Xu. 2025. Instructional Prompt Optimization for Few-Shot LLM-Based Recommendations on Cold-Start Users. arXiv preprint arXiv:2509.09066 (2025).

[56] Zhengyi Yang, Xiangnan He, Jizhi Zhang, Jiancan Wu, Xin Xin, Jiawei Chen, and Xiang Wang. 2023. A generic learning framework for sequential recommenda tion with distribution shifts. In Proceedings ofthe 46th International ACM SIGIR Conference on Research and Development in Information Retrieval. 331–340.

[57] Mingjia Yin, Hao Wang, Wei Guo, Yong Liu, Suojuan Zhang, Sirui Zhao, Defu Lian, and Enhong Chen. 2024. Dataset regeneration for sequential recommendation. In Proceedings ofthe 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining. 3954–3965.

[58] Jiaqi Zhai, Lucy Liao, Xing Liu, Yueming Wang, Rui Li, Xuan Cao, Leon Gao, Zhaojie Gong, Fangda Gu, Michael He, et al. 2024. Actions speak louder than words: Trillion-parameter sequential transducers for generative recommendations. arXiv preprint arXiv:2402.17152 (2024).

[59] Shengyu Zhang, Dong Yao, Zhou Zhao, Tat-Seng Chua, and Fei Wu. 2021. Causerec: Counterfactual user sequence synthesis for sequential recommendation. In Proceedings ofthe 44th international ACM SIGIR conference on research and development in information retrieval. 367–377.

[60] Tingting Zhang, Pengpeng Zhao, Yanchi Liu, Victor S Sheng, Jiajie Xu, Deqing Wang, Guanfeng Liu, Xiaofang Zhou, et al. 2019. Feature-level deeper selfattention network for sequential recommendation.. In IJCAI. 4320–4326.

[61] Zeyu Zhang, Heyang Gao, Hao Yang, and Xu Chen. 2023. Hierarchical invariant learning for domain generalization recommendation. In Proceedings ofthe 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining. 3470–3479.

[62] Kun Zhou, Hui Wang, Wayne Xin Zhao, Yutao Zhu, Sirui Wang, Fuzheng Zhang, Zhongyuan Wang, and Ji-Rong Wen. 2020. S3-rec: Self-supervised learning for sequential recommendation with mutual information maximization. In Proceedings of the 29th ACM international conference on information & knowledge management. 1893–1902.

[63] Xin Zhou. 2023. Mmrec: Simplifying multimodal recommendation. In Proceedings ofthe 5th ACM International Conference on Multimedia in Asia Workshops. 1–2.

[64] Shichao Zhu, Mufan Li, Guangmou Pan, and Xixun Lin. 2025. TTGL: large-scale multi-scenario universal graph learning at TikTok. In Proceedings ofthe 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 2. 5249–5259.

[65] Yu Zhu, Hao Li, Yikang Liao, Beidou Wang, Ziyu Guan, Haifeng Liu, and Deng Cai. 2017. What to do next: Modeling user behaviors by time-LSTM.. In IJCAI, Vol. 17. Melbourne, VIC, 3602–3608.