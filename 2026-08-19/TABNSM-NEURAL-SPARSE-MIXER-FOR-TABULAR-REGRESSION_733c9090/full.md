# TABNSM: NEURAL SPARSE MIXER FOR TABULAR REGRESSION

Ali Eslamian Department of Computer Science University of Kentucky Lexington, KY 40506

Qiang Cheng

Department of Computer Science   
Institute for Biomedical Informatics University of Kentucky Lexington, KY 40506

## ABSTRACT

Large-scale, high-dimensional tabular regression remains challenging. Tree-based models are often robust but do not support end-to-end representation learning, while deep models offer flexible feature learning but can incur costly interaction modeling and remain sensitive to noisy or redundant features. We propose TabNSM, a scalable regression framework that extends our earlier sparse-attention and mixer architectures. TabNSM adapts these components within an Adaptive Sparse Interaction Module (ASIM), which combines foreground feature discovery, sparse local interaction encoding, and Feature-Token Mixing with near-linear complexity under fixed sparse configurations. The regression-specific contributions are a Multi-Stage Regression Head, GridLoss, an ordinal-aware soft-binning objective, and RISE (Reweighted Instance Sampling by Error), a difficulty-aware sampling strategy based on loss-quantile bins. Across nine real-world regression benchmarks, TabNSM achieves strong predictive performance and practical scalability, with consistent gains on high-dimensional and heterogeneous datasets.

## 1 Introduction

Modern applications in healthcare, finance, and industrial monitoring depend on large-scale tabular regression to forecast continuous outcomes, such as clinical risk scores or sensor-based targets. These tasks involve high-dimensional, heterogeneous feature spaces comprising numerical signals, categorical attributes, and text-derived embeddings [1]. A defining characteristic of such data is that predictive signals are rarely distributed evenly across the feature space; instead, a small subset of features may be decision-relevant for each instance, while the majority are noisy, redundant, or weakly informative [2]. This sparse and instance-specific structure contrasts with image and text modalities, where informative content is organized into coherent spatial or sequential regions.

This structure motivates a foreground-background view of high-dimensional tabular regression. For each sample, the feature space can be viewed as containing a small foreground of decision-relevant features and a much larger background of uninformative or noisy features. Crucially, this partition is not globally fixed: the foreground shifts across samples, so no single feature subset characterizes the entire dataset. A model that cannot dynamically isolate the foreground may conflate signal with background variation, degrading both accuracy and robustness.

The empirical strength of tree-based models on tabular data can be partly understood from this perspective. Models such as CatBoost perform recursive, axis-aligned feature selection, progressively isolating sparse predictive signals through hierarchical splitting [3; 4; 5]. This mechanism provides a structural prior well aligned with tabular sparsity. However, tree-based models are less suited to end-to-end differentiable representation learning, seamless integration of high-dimensional semantic embeddings, and joint optimization with learned feature encoders. These limitations become increasingly important in modern text-enriched tabular settings.

Deep tabular models offer greater representational flexibility but face complementary challenges. Transformer-style architectures typically treat tabular features as tokens and use self-attention to model cross-feature dependencies, bu dense self-attention computes all pairwise interactions, resulting in quadratic cost with respect to feature dimension and potentially mixing informative features with large numbers of noisy or redundant background features. Recent state-space models (SSMs) [6; 7; 8] reduce this computational burden through linear-time sequential scanning, but their reliance on an imposed feature ordering does not directly provide sparse, instance-adaptive feature selection. Existing deep tabular models therefore face a persistent trade-off among expressive interaction modeling, robust foreground isolation, and computational scalability.

Building on our earlier TabNSA sparse-attention architecture and TabMixer mixing block [2; 9], we propose Tabular Neural Sparse Mixer (TabNSM), a scalable framework for large-scale, high-dimensional tabular regression. TabNSM adapts these prior components within an Adaptive Sparse Interaction Module (ASIM), which combines instance-adaptive foreground discovery, sparse local interaction encoding, and Feature-Token Mixing (FTM). By selecting structured subsets of feature interactions, ASIM provides a differentiable analogue of tree-like selective feature isolation while preserving the expressiveness and end-to-end trainability of neural networks. FTM further promotes information propagation beyond the sparse attention support, enabling the model to balance selective feature interaction with broader feature-level context without incurring dense quadratic attention cost.

To support robust training with continuous targets, TabNSM further introduces two complementary training components. GridLoss aligns predictions and targets in an ordinal grid space, providing a smooth, structure-aware complement to standard pointwise regression losses. RISE (Reweighted Instance Sampling by Error) periodically reweights training examples according to per-sample loss, encouraging the model to focus on difficult or underfit regions of the target distribution. Together, ASIM, GridLoss, and RISE provide a unified framework for scalable and robust high-dimensional tabular regression.

In summary, the methodological contributions of this regression-focused extension are as follows:

• Regression-oriented ASIM integration: an adaptation of the TabNSA sparse-attention pattern and TabMixerstyle feature-token mixing within a unified high-dimensional regression pipeline.

• Multi-Stage Regression Head: a progressively refined prediction head with residual coupling and feature fusion for coarse-to-fine regression from the ASIM-enhanced representation.

• GridLoss: a differentiable soft-binning objective that aligns predictions and continuous targets in an ordinal grid space, complementing standard pointwise regression losses.

• RISE: a difficulty-aware sampling strategy that periodically reweights training instances using loss-quantile bins to improve robustness under non-uniform error distributions.

We evaluate TabNSM on nine real-world regression benchmarks across diverse domains and data regimes. Results show strong predictive performance and practical scalability, while ablations validate the complementary roles of ASIM, the Multi-Stage Regression Head, GridLoss, and RISE.

## 2 Related Work

Deep Learning versus Tree Ensembles for Regression. Gradient Boosted Decision Trees (GBDTs), including XGBoost [10], LightGBM [11], and CatBoost [5], remain strong baselines for tabular regression and frequently outperform deep architectures on medium-scale datasets with limited feature complexity [12; 13]. However, GBDTs are not naturally end-to-end differentiable and are less suited to joint representation learning with high-dimensional learned features, limiting their flexibility in modern multimodal or text-enriched tabular settings. While tree-based ensembles remain competitive, their training cost can increase substantially with dataset size, number of trees, and tree depth, which may limit scalability in large-scale or high-dimensional regimes (Appendix K.4).

Deep tabular models have progressed beyond standard multilayer perceptrons (MLPs) to include regularization-focused architectures [14], logic-aware networks [15], and hybrid tree-neural models [16]. Recent tabular foundation models (TFMs), notably TabPFN [17] and TabPFNv2 [18], achieve strong generalization through in-context learning. However, their transformer-based designs can incur substantial computational and memory costs due to attention over samples and/or features, often scaling quadratically in the relevant dimension. In addition, inference may require conditioning on training examples, which can limit scalability on large datasets. As a result, practical use of such models is often constrained by dataset size, feature dimensionality, and memory budget [19; 20; 21].

Feature Interaction and Scalability. Transformer-based tabular models, such as TabTransformer [22] and FT-Transformer [23], enable end-to-end representation learning by using dense self-attention to capture global feature interactions. However, their dense attention mechanisms can be sensitive to feature noise in high-dimensional settings [24; 23]. While expressive, dense attention incurs $\mathcal { O } ( D ^ { 2 } )$ computational and memory costs with respect to feature dimension D, which becomes expensive for wide tables with hundreds of features. Recent sparse attention variants [2; 25] reduce this burden through block-diagonal masks, hierarchical patterns, or selective attention mechanisms. Concurrently, State-Space Models (SSMs), such as Mamba [26], provide linear-time sequence mixing along the feature axis through selective scan operations, improving scalability and parameter efficiency [27]. However, SSM-based tabular models typically rely on imposed feature ordering and do not explicitly model sparse pairwise feature interactions.

Handling Heterogeneous Data and Text Features. Modern tabular datasets incorporate high-dimensional textual embeddings alongside structured numerical and categorical fields [28; 29]. While preprocessing relies on ordinal, label, or one-hot encoding, recent work has shown that pretrained language models can provide semantic representations for text-rich tabular data [30]. Integrating semantic text embeddings, such as those from Sentence-BERT, can improve performance on benchmarks containing textual attributes [18; 31]. TabICL [32] performs in-context learning at inference time but is designed for classification rather than large-scale regression. BiSHop [33] models column- and row-wise dependencies via bi-directional sparse Hopfield layers but scales poorly to high-dimensional datasets. MITRA [34] pretrains on a mixture of synthetic priors for in-context learning but does not consistently outperform TabPFNv2 on high-dimensional regression.

Regression Objectives and Difficulty-Aware Learning. Beyond standard MSE and MAE, robust losses such as Huber and quantile losses can improve resistance to outliers and heterogeneous noise [35]. Distribution-aware and soft-label approaches improve regression calibration by discretizing continuous targets or representing them through label distributions [36; 37]. Curriculum learning and hard-example mining are widely used in classification, but remain less developed in regression, particularly for long-tailed targets and heterogeneous error distributions. These settings motivate training strategies that emphasize difficult samples without destabilizing optimization.

Positioning of TabNSM. Prior work suggests that scalable tabular regression requires more than reducing the cost of feature mixing. GBDTs provide strong feature selectivity but lack differentiable representation learning; dense-attention models learn flexible interactions but scale poorly with feature dimension; and SSM-based models improve efficiency but do not explicitly impose sparse, instance-adaptive feature interaction structure [38; 39]. Our earlier TabNSA model adapted compressed-block selection and local sparse attention to tabular features, while TabMixer introduced parallel feature- and token-mixing branches [2; 9]. TabNSM is a follow-on regression framework: ASIM reuses and adapts those principles rather than claiming a wholly new sparse-attention primitive. The new contribution is their regression-specific integration with a Multi-Stage Regression Head, GridLoss, and RISE. These components jointly address selective feature interaction, ordered continuous targets, and heterogeneous training difficulty.

## 3 Methodology

## 3.1 Problem Formulation and Preprocessing

Let $\{ ( x _ { i } , y _ { i } ) \} _ { i = 1 } ^ { N }$ be a supervised regression dataset where $x _ { i } \in \mathbb { R } ^ { D }$ represents a sample of D heterogeneous features and $y _ { i } \in \mathbb { R }$ is the continuous target. Our objective is to learn a mapping $f _ { \Theta } : \mathbb { R } ^ { D } $ R that minimizes a robust regression objective while maintaining scalability in the high-dimensional regime $( D \gg 1 0 ^ { 2 } )$ .

To address the heterogeneity of tabular data, we unify numerical, categorical, and textual features into a dense representation. Following standard practice [40], we categorize columns into numerical, low-cardinality categorical, and high-cardinality or free-text features.

Numerical features are standardized to zero mean and unit variance. Low-cardinality categorical variables (fewer than 10 unique values) are integer-encoded; this avoids the dimensionality expansion of one-hot encoding, while allowing downstream layers to learn category-dependent interactions. Crucially, all transformation parameters are fitted solely on the training split to prevent data leakage.

For high-cardinality and free-text fields $( \mathrm { e . g . }$ ., descriptions or addresses), we use semantic embeddings [41; 42]. Text columns are concatenated per row and encoded with a frozen all-MiniLM-L6-v2 Transformer [43; 44]. A fixed vector $E _ { i } \in \mathbb { R } ^ { d _ { \mathrm { t e x t } } }$ is obtained via masked mean pooling over last-layer states $H _ { i }$ as $\begin{array} { r } { E _ { i } = \frac { \sum _ { t } M _ { i , t } H _ { i , t } } { \sum _ { t } M _ { i , t } + \epsilon } } \end{array}$ , where $M _ { i , t } \in \{ 0 , 1 \}$ Embeddings are ℓ<sub>2</sub>-normalized. The final input $x _ { i }$ concatenates numerical, categorical, and text features. The approach is zero-shot but compatible with larger LLM encoders (e.g., Gemma) for stronger text–target alignment [29].

## 3.2 Architecture Overview

Building on the published TabNSA–TabMixer backbone [2; 9], TabNSM makes two regression-specific architectural changes: (i) ASIM couples the inherited structured sparse-attention and FTM pathways through learnable residual blending; and (ii) a Multi-Stage Regression Head applies residually coupled MLP branches followed by a compact fusion head to produce a continuous scalar prediction. Figure 1 summarizes the resulting architecture.

![](images/74525ef9d3f25f789da5092dd8d6fa0e296070b1b5773743c7999dd8d3bb6831.jpg)  
Figure 1: Overview of the TabNSM architecture. TabNSM extends the published TabNSA–TabMixer representation backbone with regression-specific residual integration and a Multi-Stage Regression Head. Within ASIM, the inherited structured sparse-attention operator provides instance-adaptive feature selection, while the inherited FTM operator prop agates broader feature-level context. Learnable residual blending couples these streams, and the resulting representation is passed to progressively narrowed MLP branches with residual coupling and final feature fusion to produce yˆ.

## 3.3 Adaptive Sparse Interaction Module (ASIM)

To model cross-feature interactions at scale, we project each scalar feature into a $d _ { \mathrm { t o k } }$ -dimensional embedding space. Given an input vector x $\in \mathbb { R } ^ { D }$ , each feature x is mapped via a learnable projection to obtain the tokenized representation $X _ { \mathrm { t o k } } = \mathrm { T o \dot { k } } ( x ) \in \mathbb { R } ^ { D \times d _ { \mathrm { t o k } } }$ , where $X _ { \mathrm { t o k } } [ d , : ] \in \mathbb { R } ^ { i _ { \mathrm { t o k } } }$ denotes the token vector for the d-th feature. (Batch dimensions are omitted for clarity).

Instance-Adaptive Sparse Feature-wise Attention. Following the feature-wise adaptation introduced in TabNSA [2], we use a sparse attention mechanism that treats the D input features as a sequence, where each feature token serves as a query for the current sample. Foreground feature discovery is implemented through three components: (i) a local sliding window of size s that captures local interactions under the chosen feature ordering; (ii) compressed block representations, which aggregate blocks of size b with stride r into summary keys using a learnable MLP, whose resulting attention scores identify the most relevant feature groups for each query; and (iii) retrieval of the top-m highest-scoring blocks at full resolution as theforeground. Because the attention scores are computed from sample-specific query tokens, the selected blocks vary across inputs, allowing each sample to dynamically isolate its own foreground feature groups:

$$
\begin{array} { r } { \tilde { X } = \mathcal { A } _ { \mathrm { s p a r s e } } ( X _ { \mathrm { t o k } } ; \ s , b , r , m , H , d _ { h } ) \in \mathbb { R } ^ { D \times d _ { \mathrm { t o k } } } . } \end{array}\tag{1}
$$

Here H is the number of attention heads and $d _ { h }$ is the per-head dimension, with $d _ { \mathrm { t o k } } = H d _ { h }$ . This mechanism enables instance-adaptive focus on informative feature-to-feature dependencies and can improve robustness to background noise when the sparse support excludes irrelevant dimensions (Proposition 3.2; see Appendix I for the full formulation).

Feature-Token Mixing (FTM) and Residual Blending. To capture global dependencies that might fall outside the sparse attention mask, we adapt the dual-axis mixing design of TabMixer [9] as an FTM operator M(·). It performs per-instance mixing across the embedding and feature dimensions using lightweight MLPs with parallel branches (Appendix J). To stabilize optimization, we combine the sparse and original token streams through learnable scalars $w _ { 1 } , w _ { 2 } , w _ { 3 } \in \mathbb { R }$ . All intermediate tensors share the same shape R $D \times d _ { \mathrm { t o k } } ^ { \mathrm { } }$ as $X _ { \mathrm { t o k } }$

$$
U = w _ { 1 } X _ { \mathrm { t o k } } + \tilde { X } ,\tag{2}
$$

$$
Z = { \mathcal { M } } ( U ) ,\tag{3}
$$

$$
Y = w _ { 2 } U + Z + w _ { 3 } X _ { \mathrm { t o k } } .\tag{4}
$$

Finally, we aggregate across the token axis to return to the feature space via $y _ { \mathrm { f e a t } } = \mathrm { m e a n } _ { \mathrm { t o k } } ( Y ) \in \mathbb { R } ^ { D }$ . We refer to this module as the Adaptive Sparse Interaction Module (ASIM); it yields a feature-level representation enriched by adaptive selectivity and global mixing while maintaining linear memory complexity in D under fixed sparse settings.

Proposition 3.1 (ASIM Complexity (Informal)). Under the notation above, let the sparse attention operator attend to at most $L = L ( s , b , r , m )$ keys per queryfeature. Forfixed token dimension $d _ { \mathrm { t o k } } = H d _ { h } ,$ , number ofheads H, and head dimension $d _ { h }$ , one ASIM layer has sparse-attention complexity $\mathcal { O } ( D H d _ { h } L ) = \mathcal { O } ( D d _ { \mathrm { t o k } } L )$ and activation memory $\mathcal { O } ( D ( d _ { \mathrm { t o k } } + L ) )$ . Underfixed sparse hyperparameters,fixed token dimension, andfixedfeature mini-batch size $B _ { f }$ both compute and memory scale linearly with thefeature dimension D.

The formal statement and proof are in Appendix B.1.

The following proposition gives a simplified conditional analysis: if the sparse support covers the informative features while excluding most background dimensions, then sparse aggregation improves the signal-to-noise ratio relative to dense aggregation.

Proposition 3.2 (Conditional SNR Improvement under Sparse Feature Selection). Let $\mathbf { x } = [ \mathbf { x } _ { S } ; \mathbf { x } _ { N } ] \in \mathbb { R } ^ { D }$ , where S indexes s informative features with average signal µ and N indexes i.i.d. zero-mean noise features with variance $\sigma ^ { 2 }$ . Consider dense aggregation over all D features and sparse aggregation over a support T satisfying $\mathcal { S } \subseteq \mathcal { T }$ and $| { \mathcal { T } } | = m b .$ Then,for mb $> s ,$

$$
\mathrm { S N R } _ { \mathrm { d e n s e } } = \frac { s | \mu | } { \sigma \sqrt { D - s } } , \qquad \mathrm { S N R } _ { \mathrm { s p a r s e } } = \frac { s | \mu | } { \sigma \sqrt { m b - s } } .
$$

Thus, when the sparse support covers the informativefeatures while excluding most background dimensions, sparse aggregation improves SNR by

$$
{ \frac { \mathrm { S N R } _ { \mathrm { s p a r s e } } } { \mathrm { S N R } _ { \mathrm { d e n s e } } } } = { \sqrt { \frac { D - s } { m b - s } } } .
$$

In the ideal case m $\phantom { } _ { l } b = s ,$ , the sparse support contains no background noise, so the noise contribution vanishes under this simplified model.

The full statement and proof are provided in Appendix B.2.

## 3.4 Multi-Stage Feature Refinement

The TabNSM head processes the enriched representation $y _ { \mathrm { f e a t } } \in \mathbb { R } ^ { D }$ from the ASIM through a multi-scale residual pipeline. This design enables the model to iteratively refine predictions via coarse-to-fine pathways.

Progressive Branches. Rather than using a purely sequential bottleneck, TabNSM employs residually coupled branches that repeatedly access the full representation $y _ { \mathrm { f e a t } }$ while progressively reducing dimensionality:

$$
y _ { 1 } = \mathrm { D e c } _ { 1 } ( \mathrm { M L P } _ { 1 } ( y _ { \mathrm { f e a t } } ) + y _ { \mathrm { f e a t } } ) \in \mathbb { R } ^ { D / 2 } ,\tag{5}
$$

$$
y _ { 2 } = \mathrm { D e c } _ { 2 } ( \mathrm { M L P } _ { 2 } ( y _ { \mathrm { f e a t } } ) + y _ { 1 } ) \in \mathbb { R } ^ { 1 } ,\tag{6}
$$

$$
y _ { 3 } = \mathrm { M L P } _ { 3 } ( y _ { \mathrm { f e a t } } ) + y _ { 2 } \in \mathbb { R } ^ { 1 } ,\tag{7}
$$

where $\mathrm { M L P } _ { 1 } : \mathbb { R } ^ { D }  \mathbb { R } ^ { D }$ $\mathrm { M L P _ { 2 } } : \mathbb { R } ^ { D } \to \mathbb { R } ^ { D / 2 }$ , and $\mathrm { M L P _ { 3 } } : \mathbb { R } ^ { D } \to \mathbb { R } ^ { 1 }$ are shallow networks with GELU activations. The decoders $\mathrm { D e c _ { 1 } }$ and $\mathrm { D e c _ { 2 } }$ facilitate dimensionality reduction. In this configuration, Eq. (6) performs residual learning in the intermediate $D / 2$ space by combining the coarse representation $y _ { 1 }$ with a secondary featureinformed projection, while Eq. (7) provides the final scalar refinement. This architecture ensures that even the final prediction remains grounded in the high-dimensional feature interactions encoded by $y _ { \mathrm { f e a t } }$

Concatenation and Feature Fusion Head. In tandem with these coupled branches, the multi-stage outputs are concatenated and processed by a compact fusion head to produce the final prediction:

$$
\begin{array} { r } { \hat { y } = h \left( \left[ y _ { 1 } , y _ { 2 } , y _ { 3 } \right] \right) + y _ { 2 } + y _ { 3 } , } \end{array}\tag{8}
$$

where $[ \cdot ]$ denotes concatenation and $h : \mathbb { R } ^ { D / 2 + 2 }  \mathbb { R }$ is a shallow fusion MLP. This additive fusion couples the progressively narrowed signals from the final branches $( y _ { 2 }$ and $y _ { 3 } )$ with a multi-scale summary of the entire cascade. By processing the concatenated vector $[ y _ { 1 } , y _ { 2 } , y _ { 3 } ]$ , the fusion head learns residual corrections that integrate coarse and fine-grained information, improving both predictive stability and expressivity at negligible computational cost.

Complexity Remarks. For fixed token dimension $d _ { \mathrm { t o k } }$ , number of heads H, head dimension $d _ { h }$ , and sparse hyperparameters $( s , b , r , m )$ , the computational complexity of TabNSM is dominated by three terms: (i) Sparse Attention: $\mathcal { O } ( D d _ { \mathrm { t o k } } L )$ , where $L$ is the number of attended keys per feature token, ensuring sub-quadratic scaling in D; (ii) FTM: $O ( D d _ { \mathrm { t o k } } + D B _ { f } )$ under fixed token dimension and feature mini-batch size $B _ { f } ;$ and (iii) TabNSM Head: $\mathcal { O } ( D )$ through linear-time projections and shallow MLPs. Empirically, this architecture achieves near-linear scaling with respect to the feature dimension $D$ under practical sparse mixer configurations, making it suitable for high-dimensional tabular datasets.

## 3.5 GridLoss: Differentiable Soft-Binning for Ordinal Alignment

Regression losses such as $\ell _ { 1 }$ and Huber penalize pointwise errors but do not explicitly enforce ordinal alignment between predictions and targets. To preserve relative positions along the target scale, we introduce GridLoss, a differentiable soft-binning objective that aligns predictions and targets in a shared discretized index space.

Batch-Adaptive Grid Construction. For a mini-batch of size $N ,$ let $\hat { y } , y \in \mathbb { R } ^ { N }$ denote the predictions and targets. We define a batch-adaptive interval $[ \alpha , \beta ]$ as:

$$
\alpha = \operatorname* { m i n } ( \operatorname* { m i n } ( \hat { y } ) , \operatorname* { m i n } ( y ) ) , \beta = \operatorname* { m a x } ( \operatorname* { m a x } ( \hat { y } ) , \operatorname* { m a x } ( y ) ) .\tag{9}
$$

We construct a uniform grid of $K + 1$ anchor points $\begin{array} { r } { e _ { k } = \alpha + k \cdot \frac { \beta - \alpha } { K } } \end{array}$ for $k \in \{ 0 , \ldots , K \}$ . A small constant $\epsilon > 0$ $( \mathbf { e } . \mathbf { g } . , 1 0 ^ { - 8 } )$ is added to $\beta - \alpha$ to prevent degenerate ranges. In implementation, α and $\beta$ are computed per mini-batch and detached from the computation graph.

Soft Assignment via Temperature-Scaled Distances. We compute the distances of each sample to all anchor points: $d _ { k } ^ { \mathrm { p r e d } } = \left| \hat { y } - e _ { k } \right|$ and $d _ { k } ^ { \mathrm { t r u e } } = | y - e _ { k } |$ . These are converted into soft assignments via a temperature-scaled softmax:

$$
w _ { k } ^ { \mathrm { p r e d } } = \frac { \exp ( - d _ { k } ^ { \mathrm { p r e d } } / \tau ) } { \sum _ { j = 0 } ^ { K } \exp ( - d _ { j } ^ { \mathrm { p r e d } } / \tau ) } , \quad w _ { k } ^ { \mathrm { t r u e } } = \frac { \exp ( - d _ { k } ^ { \mathrm { t r u e } } / \tau ) } { \sum _ { j = 0 } ^ { K } \exp ( - d _ { j } ^ { \mathrm { t r u e } } / \tau ) } ,
$$

where $\tau > 0$ controls the assignment softness. We then compute soft indices as expectations over the anchors: $\begin{array} { r } { b ^ { \mathrm { p r e d } } = \sum _ { k = 0 } ^ { K } k w _ { k } ^ { \mathrm { p r e d } } } \end{array}$ and $\begin{array} { r } { b ^ { \mathrm { t r u e } } = \sum _ { k = 0 } ^ { K } k w _ { k } ^ { \mathrm { t r u e } } } \end{array}$

Objective Function. The GridLoss objective is the mean $\ell _ { 1 }$ distance between the soft indices:

$$
\mathcal { L } _ { \mathrm { G r i d } } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } | b _ { i } ^ { \mathrm { p r e d } } - b _ { i } ^ { \mathrm { t r u e } } | .\tag{10}
$$

GridLoss operates in an ordinal index space, encouraging predictions to fall into the correct relative region of the target distribution. As formally analyzed in Theorem B.9 and Proposition B.13 (Appendix B.4), under the high-resolution and low-temperature limit $( K \to \infty , \tau \to 0 ^ { + } )$ , the normalized objective approaches an MAE-like ordinal distance. Thus, GridLoss provides a smooth, structure-aware approximation to robust pointwise regression losses while preserving ordinal information in the target space.

GridLoss has three useful properties: its soft index is monotone in the prediction, its gradient is bounded by $K / ( 2 \tau )$ and it is Lipschitz continuous. For finite τ, these properties imply stable gradients, while the high-resolution, lowtemperature limit yields an MAE-like behavior; formal statements and proofs are provided in Appendix B.3.

## 3.6 Training Strategy with RISE Sampling

Standard regression pipelines typically construct mini-batches via uniform sampling, implicitly treating all training instances as equally informative. In high-dimensional tasks with long-tailed error distributions, this can underrepresent “hard” samples that often correspond to minority subpopulations or extreme target tails, leading to suboptimal general ization. To mitigate this, we adopt a difficulty-aware sampling strategy termed RISE (Reweighted Instance Sampling by Error). RISE periodically estimates per-sample prediction errors and dynamically biases the sampling distribution toward harder examples. It is conceptually related to the non-uniform sampling used by GRIP [45], but RISE operates on regression examples and assigns bounded weights from empirical loss-quantile bins rather than sampling policy trajectories.

The RISE-enhanced training procedure adapts the model’s focus as convergence progresses. Every k epochs, we refresh the sampling distribution via the following steps:

1. Loss Estimation: Compute per-sample losses $\ell _ { i }$ for the entire training set using the current parameters $\theta _ { e }$

2. Quantile Partitioning: Partition samples into G difficulty bins based on the loss quantiles.

3. Weight Assignment: Assign a sampling weight $\begin{array} { r } { w _ { i } = 1 + g _ { i } \cdot \frac { \beta - 1 } { G - 1 } } \end{array}$ , where $g _ { i } \in \{ 0 , \ldots , G - 1 \}$ is the bin index and $\beta \geq 1$ is the maximum boost factor.

4. Weighted Sampling: Draw mini-batches using a weighted random sampler with probabilities $p _ { i } \propto w _ { i }$

By upweighting bins with larger errors, RISE implements a dynamic curriculum that prioritizes samples currently underfit by the model or lying in difficult regions of the target distribution. We provide a theoretical interpretation of this mechanism below; see Appendix B.5 for the formal statement and proof.

Proposition 3.3 (RISE as a Tail-aware Objective (Informal)). Let $\ell _ { i } ( \theta )$ denote the per-sample loss. At refresh step e, RISE assigns weights $w _ { i } ^ { ( e ) } = \varphi ( q _ { i } ^ { ( e ) } )$ based on the loss-quantile bin $q _ { i } ^ { ( e ) }$ . If mini-batches are drawn by weighted sampling with probabilities $p _ { i } ^ { ( e ) } \propto w _ { i } ^ { ( e ) }$ and updates are performed using sampled gradients without importance-weight correction, the expected SGD step is equivalent to descending the reweighted empirical risk:

$$
\mathcal { L } _ { \mathrm { R I S E } } ^ { ( e ) } ( \theta ) = \frac { \sum _ { i = 1 } ^ { N } w _ { i } ^ { ( e ) } \ell _ { i } ( \theta ) } { \sum _ { j = 1 } ^ { N } w _ { j } ^ { ( e ) } } .\tag{11}
$$

Thus, a monotonically increasing φ(·) ensures the optimization emphasizes the tail ofthe error distribution.

Remark. Standard Importance Sampling applies an inverse-probability correction $( 1 / p _ { i } )$ to keep the gradient estimator unbiased. RISE intentionally omits this correction to prioritize high-loss samples, thus minimizing a risk functional that favors robustness in difficult feature-space regions.

## 4 Experiments and Results

We evaluate TabNSM on nine real-world regression benchmarks spanning robotics, transportation, public health, criminology, real estate, air quality, and neuroimaging. Dataset statistics are reported in Table 3, with detailed descriptions in Appendix F. We use a 72/8/20 train/validation/test split with fixed random partitions across methods. Samples with missing targets are removed, missing feature values are imputed with zero, and continuous features are standardized using training-set statistics. Models are trained for up to 100 epochs with early stopping on validation loss; test performance is reported from the best validation checkpoint. Additional seed-variation analysis is provided in Appendix G.

Training protocol. The ASIM hyperparameters are selected via Optuna [46] on the validation set and fixed thereafter. The sparse operator $\mathcal { A } _ { \mathrm { s p a r s e } }$ is implemented using structured sparse attention (Eq. 1), while M is instantiated as an FTM operator for global feature interaction. Unless otherwise stated, TabNSM is trained with AdamW, validation-based scheduling, mixed precision for first-order optimizers, and RISE-based sampling.

Main results. Table 1 summarizes RMSE results against 38 baselines grouped into classical methods, tree ensembles, MLP-style models, transformer-based models, retrieval-based methods, other deep tabular architectures, and tabular foundation models (TFMs). Baseline implementation details are provided in Appendix D. To keep the main table compact, we report the best-performing baseline within each model family; the full per-model comparison is provided in Appendix C.1.

TabNSM achieves the lowest RMSE on seven of nine benchmarks (SA, TO, AQ, SC, ChD, EVP, RES). Pairwise Wilcoxon signed-rank tests over the benchmark suite show statistically significant aggregate gains over the baselines after correction $( p \leq 0 . 0 2 ;$ Appendix C.2). The largest margins appear on TO (+22%, 0.021 vs. 0.027 for all competing models) and EVP (+21%, 1.199 vs. 1.507 for FT-Transformer), two benchmarks with high feature dimensionality and structured or complex targets. On SA, TabNSM surpasses DNNR (0.149→0.137), the strongest deep learning-based baseline. On AQ, TabNSM outperforms DCN-v2 (3.999 vs. 4.081) despite DCN-v2’s explicit cross-network interaction design. On CD, TabNSM ranks second to Extratrees (0.007 vs. 0.004); on CP, a medium-sized, low-dimensional benchmark, tree-based ensembles dominate, making it the only dataset where TabNSM does not place in the top two.

Result interpretation. TabNSM is most effective on high-dimensional, interaction-rich datasets. On EVP (392 features) and ChD (405 features), ASIM’s instance-adaptive foreground selection helps isolate informative subsets from a large noisy feature background, while FTM propagates global context without dense quadratic cost. On AQ, sparse selection also improves over DCN-v2’s explicit cross-network mixing, suggesting that instance-specific interaction structure is important. The gains on TO are consistent with the role of GridLoss in preserving ordinal target alignment, while the strong performance on SA and RES suggests that RISE can help stabilize learning under heterogeneous target distributions.

Table 1: RMSE on nine regression benchmarks. Lower is better. Best results are shown in bold, and second-best results are underlined. We report the best-performing baseline within each model family; full per-model results are provided in Appendix C.1.
<table><tr><td>Model</td><td>SA</td><td>CP</td><td>AQ</td><td>TO</td><td>SC</td><td>CD</td><td>ChD</td><td>EVP</td><td>RES</td></tr><tr><td>Best Classical (6 models)</td><td>0.175</td><td>3.719</td><td>12.816</td><td>0.027</td><td>3.602</td><td>0.236</td><td>8717</td><td>26.104</td><td>3163</td></tr><tr><td>Best Tree-based (5 models)</td><td>0.171</td><td>2.617</td><td>5.869</td><td>0.027</td><td>3.455</td><td>0.004</td><td>17525</td><td>2.799</td><td>3135</td></tr><tr><td>Best MLP-style (5 models)</td><td>0.167</td><td>2.920</td><td>4.578</td><td>0.028</td><td>3.920</td><td>0.028</td><td>4963</td><td>1.556</td><td>3161</td></tr><tr><td>Best Transformer (6 models)</td><td>0.156</td><td>2.901</td><td>4.081</td><td>0.028</td><td>3.667</td><td>0.007</td><td>4475</td><td>1.507</td><td>3163</td></tr><tr><td>Best Retrieval (4 models)</td><td>0.150</td><td>2.833</td><td>5.062</td><td>0.028</td><td>3.454</td><td>0.052</td><td>22661</td><td>2.653</td><td>3187</td></tr><tr><td>Best Other (8 models)</td><td>0.149</td><td>2.991</td><td>4.690</td><td>0.028</td><td>3.570</td><td>0.136</td><td>4808</td><td>3.219</td><td>3163</td></tr><tr><td>Best Foundation (4 models)</td><td>0.168</td><td>2.740</td><td>6.748</td><td>0.027</td><td>3.522</td><td>5.253</td><td>17858</td><td>65.471</td><td>4171</td></tr><tr><td>TabNSM (Ours)</td><td>0.137</td><td>2.842</td><td>3.999</td><td>0.021</td><td>3.322</td><td>0.007</td><td>4460</td><td>1.199</td><td>3103</td></tr></table>

Per-family best is the minimum RMSE among runnable models in that family. Full per-model results are provided in Table 2 in Appendix C.1.

![](images/7b89afea4525a8e93270dbe07f1657e58d3083021b8f9a929b2fe91db0da470b.jpg)  
(a) Ablation of sparse interaction modeling.

![](images/f45b736878a100b3a7fc432d50f4eece479cf5daf03eb537f448f1d35032c865.jpg)  
(b) Parameter scaling with feature dimensionality.  
Figure 2: Ablation and scalability analyses. (a) Full attention, Mamba, and ASIM within TabNSM; ASIM achieves the lowest RMSE. (b) Parameter count grows approximately linearly with retained feature fraction, consistent with the ASIM complexity analysis.

## 4.1 Ablation of Sparse Interaction Modeling

We ablate ASIM by replacing its sparse feature-wise attention with either full self-attention or a Mamba block operating along the feature axis, while keeping the downstream head and training protocol fixed. As shown in Figure 2a, full attention is unstable in high-dimensional settings, while Mamba improves efficiency but still underperforms ASIM. This indicates that linear-time mixing alone is insufficient: instance-adaptive sparse feature selection is important for robust high-dimensional regression. These findings are consistent with Proposition 3.2, which shows that sparse selection can improve SNR when the selected support covers informative features and excludes many background dimensions. Per-instance feature attribution heatmaps further confirm that TabNSM activates distinct feature subsets across samples (Appendix E).

## 4.2 Scalability Analysis

To complement the empirical accuracy comparison, we examine whether the ASIM component exhibits the nearlinear scaling predicted by Proposition 3.1. We vary the retained feature fraction while keeping the sparse-attention configuration fixed and measure the resulting parameter count. As shown in Figure 2b, the number of parameters grows approximately linearly with the input feature dimension across datasets. This behavior supports the intended design of ASIM: sparse feature-wise interaction modeling avoids dense all-pair feature mixing while retaining sufficient capacity for high-dimensional tabular regression. Additional runtime and memory measurements are provided in Appendix H.

Additional component checks. Appendix K further evaluates FTM, GridLoss, and RISE. Removing FTM degrades performance on CP, TO, SA, and EVP, especially on EVP, supporting its role in propagating information beyond the sparse attention support. GridLoss matches or improves over standard pointwise losses under heterogeneous errors, while RISE reduces validation variance by prioritizing difficult samples. Together, these results show that TabNSM’s gains arise from sparse feature selection, global mixing, structure-aware supervision, and difficulty-aware sampling.

## 5 Conclusion and Discussion

We introduced TabNSM, a regression-focused extension of our earlier TabNSA and TabMixer architectures that combines an adapted sparse interaction module with a Multi-Stage Regression Head, ordinal grid supervision (GridLoss), and difficulty-aware reweighting (RISE). ASIM replaces dense quadratic interaction with sample-specific foreground selection, achieving near-linear complexity while preserving adaptive sparsity. Across nine regression benchmarks, TabNSM achieves the best performance on seven datasets and ranks second on one, outperforming GBDT baselines on most tasks. Ablations show that instance-adaptive selectivity, rather than efficiency alone, is a major performance driver, while the regression-specific head, GridLoss, and RISE provide complementary gains. Although TabNSM requires GPU training and can incur higher overhead than tree-based methods, it achieves substantially lower error than most deep learning baselines with favorable training efficiency. Future work will pursue compression and distillation, extend GridLoss to multi-target settings, and explore foreground–background interactions for multimodal tabular learning.

## 6 Acknowledgements

We acknowledge the contributors who made the publicly available datasets and baseline models accessible for research purposes. We also extend our sincere thanks to Brian Gold, PhD, and the Gold Lab at the University of Kentucky for their continued support and for providing the facilities required to conduct this study. This work was partially supported by the National Science Foundation (NSF) under Grants IIS-2327113 and ITE-2433190, the National Institutes of Health (NIH) under Grants R21AG070909 and P30AG072946, and by the National Artificial Intelligence Research Resource (NAIRR) Pilot through NSF OAC-240219, as well as the Jetstream2, Bridges2, and Neocortex computing resources. Data for the SC benchmark were obtained through NACC/SCAN under the NACC Data.

## References

[1] Jintai Chen, Kuanlun Liao, Yao Wan, Danny Z. Chen, and Jian Wu. DANets: Deep abstract networks for tabular data classification and regression. Proceedings ofthe AAAI Conference on Artificial Intelligence, 36(4):3930–3938, Jun. 2022.

[2] Ali Eslamian and Qiang Cheng. TabNSA: Native sparse attention for efficient tabular data learning. Neurocomputing, page 132928, 2026.

[3] Tony Duan, Avati Anand, Daisy Yi Ding, Khanh K Thai, Sanjay Basu, Andrew Ng, and Alejandro Schuler. NGBoost: Natural gradient boosting for probabilistic prediction. In International conference on machine learning, pages 2690–2700. PMLR, 2020.

[4] Sergei Popov, Stanislav Morozov, and Artem Babenko. Neural oblivious decision ensembles for deep learning on tabular data. In International Conference on Learning Representations, 2020.

[5] Liudmila Prokhorenkova, Gleb Gusev, Aleksandr Vorobev, Anna Veronika Dorogush, and Andrey Gulin. CatBoost: unbiased boosting with categorical features. In Advances in Neural Information Processing Systems, 2018.

[6] Md Atik Ahamed and Qiang Cheng. MambaTab: A plug-and-play model for learning tabular data. In 2024 IEEE 7th International Conference on Multimedia Information Processing and Retrieval (MIPR), pages 369–375. IEEE, 2024.

[7] Yongchang Chen, Yinning Liu, Guomin Chen, Qian Zhang, Yilin Wu, and Jinxin Ruan. FT-Mamba: A novel deep learning model for efficient tabular regression. In 2024 IEEE International Conference on Bioinformatics and Biomedicine (BIBM), pages 3096–3100, 2024.

[8] Z Wang and W Zhang. Mambular: A sequential model for tabular deep learning. arXiv preprint arXiv:2408.06291, 2024.

[9] Ali Eslamian and Qiang Cheng. TabMixer: advancing tabular data analysis with an enhanced mlp-mixer approach. Pattern Analysis and Applications, 28(2):1–17, 2025.

[10] Tianqi Chen and Carlos Guestrin. XGBoost: A scalable tree boosting system. In Proceedings ofthe 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 2016.

[11] Guolin Ke, Qi Meng, Thomas Finley, Taifeng Wang, Wei Chen, Weidong Ma, Qiwei Ye, and Tie-Yan Liu. LightGBM: A highly efficient gradient boosting decision tree. In Advances in Neural Information Processing Systems, 2017.

[12] Léo Grinsztajn, Edouard Oyallon, and Gaël Varoquaux. Why do tree-based models still outperform deep learning on typical tabular data? Advances in Neural Information Processing Systems, 35:507–520, 2022.

[13] Vadim Borisov, Tobias Leemann, Kathrin Seßler, Johannes Haug, Martin Pawelczyk, and Gjergji Kasneci. Deep neural networks and tabular data: A survey. IEEE Transactions on Neural Networks and Learning Systems, 2022.

[14] Arlind Kadra, Marius Lindauer, Frank Hutter, and Josif Grabocka. Well-tuned simple nets excel on tabular datasets. In Advances in Neural Information Processing Systems, 2023.

[15] Liran Katzir, Gal Elidan, and Ran El-Yaniv. Net-DNF: Effective deep modeling of tabular data. In International conference on learning representations, 2020.

[16] Chenwei Xu and Yifan Wang. NCART: Neural classification and regression tree for tabular data. Pattern Recognition, 148:110–125, 2024.

[17] Noah Hollmann, Samuel Müller, Katharina Eggensperger, and Frank Hutter. TabPFN: A transformer that solve small tabular classification problems in a second. In NeurIPS 2022 First Table Representation Workshop, 2022.

[18] Noah Hollmann, Samuel Müller, Lennart Purucker, Arjun Krishnakumar, Max Körfer, Shi Bin Hoo, Robin Tibor Schirrmeister, and Frank Hutter. Accurate predictions on small data with a tabular foundation model. Nature, 637 (8045):319–326, 2025.

[19] Andreas C Mueller, Carlo A Curino, and Raghu Ramakrishnan. MotherNet: Fast training and inference via hyper-network transformers. In The Thirteenth International Conference on Learning Representations, 2025.

[20] Han-Jia Ye, Si-Yang Liu, and Wei-Lun Chao. A closer look at tabpfn v2: Strength, limitation, and extension. CoRR, abs/2502.17361, February 2025.

[21] Léo Grinsztajn, Klemens Flöge, Oscar Key, Felix Birkel, Philipp Jund, Brendan Roof, Benjamin Jäger, Dominik Safaric, Simone Alessi, Adrian Hayler, et al. TabPFN-2.5: Advancing the state of the art in tabular foundation models. arXiv preprint arXiv:2511.08667, 2025.

[22] Xin Huang, Ashish Khetan, Milan Cvitkovic, and Zohar Karnin. TabTransformer: Tabular data modeling using contextual embeddings. arXiv preprint arXiv:2012.06678, 2020.

[23] Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov, and Artem Babenko. Revisiting deep learning models for tabular data. In Advances in Neural Information Processing Systems, volume 34, pages 18932–18943, 2021.

[24] Wenji Li. Tabnet for high-dimensional tabular data: advancing interpretability and performance with feature fusion. IET Conference Proceedings, 2025:168–173, 2025.

[25] Jiho Park and Min-Soo Kim. Optimizing FT-Transformer: Sparse attention for improved performance on tabular data. Pattern Recognition Letters, 178, 2024.

[26] Albert Gu and Tri Dao. Mamba: Linear-time sequence modeling with selective state spaces. In First conference on language modeling, 2024.

[27] Yifan Wang, J Sun, et al. Representation learning for tabular data: A comprehensive survey. Journal ofMachine Learning Research, 2025.

[28] Pasan Dissanayake and Sanghamitra Dutta. Distilling transformers into neural nets for few-shot tabular classification, 2025.

[29] Stefan Hegselmann, Alejandro Buendia, Hunter Lang, Monica Agrawal, Xiaoyi Jiang, and David Sontag. TabLLM: Few-shot classification of tabular data with large language models. In Francisco Ruiz, Jennifer Dy, and Jan-Willem van de Meent, editors, Proceedings ofThe 26th International Conference on Artificial Intelligence and Statistics, volume 206 of Proceedings ofMachine Learning Research, pages 5549–5581. PMLR, 25–27 Apr 2023.

[30] L Zhang and J Wu. Embedding strategies for categorical and text features in tabular deep learning. Computational Statistics, 2023.

[31] Manan Roy Choudhury, Anirudh Iyengar Kaniyar Narayana Iyengar, Shikhhar Siingh, Sugeeth Puranam, and Vivek Gupta. TABARD: A novel benchmark for tabular anomaly analysis, reasoning and detection. In Findings of the Associationfor Computational Linguistics: EMNLP 2025, pages 21783–21817, 2025.

[32] QU Jingang, David Holzmüller, Gaël Varoquaux, and Marine Le Morvan. TabICL: A tabular foundation model for in-context learning on large data. In Forty-second International Conference on Machine Learning, 2025.

[33] Chenwei Xu, Yu-Chao Huang, Jerry Yao-Chieh Hu, Weijian Li, Ammar Gilani, Hsi-Sheng Goan, and Han Liu. BiSHop: Bi-directional cellular learning for tabular data with generalized sparse modern hopfield model. In Forty-first International Conference on Machine Learning (ICML), 2024.

[34] Xiyuan Zhang, Danielle C. Maddix, Junming Yin, Nick Erickson, Abdul Fatir Ansari, Boran Han, Shuai Zhang, Leman Akoglu, Christos Faloutsos, Michael W. Mahoney, Cuixiong Hu, Huzefa Rangwala, George Karypis, and Bernie Wang. Mitra: Mixed synthetic priors for enhancing tabular foundation models. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2026.

[35] Sedigheh Khaki et al. A methods guideline for deep learning for tabular data in agriculture. Computers and Electronics in Agriculture, 205, 2023.

[36] X Li et al. Rethinking soft labels for regression. In International Conference on Machine Learning, 2022.

[37] Qianxi Qiu and Han Liu. Numerical embedding of categorical features in tabular data: A survey. In 2023 International Conference on Machine Learning and Cybernetics (ICMLC), pages 446–451, 2023.

[38] Weijieying Ren, Tianxiang Zhao, Yuqing Huang, and Vasant Honavar. Deep learning within tabular data: Foundations, challenges, advances and future directions. arXiv preprint arXiv:2501.03540, 2025.

[39] Badri Narayana Patro and Vijay Srinivas Agneeswaran. Mamba-360: Survey of state space models as transformer alternative for long sequence modelling: Methods, applications, and challenges. Engineering Applications of Artificial Intelligence, 159:111279, 2025.

[40] Fabian Pedregosa, Gaël Varoquaux, Alexandre Gramfort, Vincent Michel, Bertrand Thirion, Olivier Grisel, Mathieu Blondel, Peter Prettenhofer, Ron Weiss, Vincent Dubourg, et al. Scikit-learn: Machine learning in Python. Journal ofMachine Learning Research, 12:2825–2830, 2011.

[41] Sebastian Bordt, Harsha Nori, and Rich Caruana. TAB2TEXT: A framework for deep learning with tabular data. In Findings ofthe Associationfor Computational Linguistics: EMNLP 2024, 2024.

[42] Boshko Koloski, Andrei Margeloiu, Xiangjian Jiang, Blaž Škrlj, Nikola Simidjievski, and Mateja Jamnik. LLM embeddings for deep learning on tabular data. arXiv preprint arXiv:2502.11596, 2025.

[43] Nils Reimers and Iryna Gurevych. Sentence-BERT: Sentence embeddings using siamese bert-networks. In EMNLP, 2019.

[44] Wenhui Wang, Hangbo Bao, Shaohan Huang, et al. MiniLM: Deep self-attention distillation for task-agnostic compression of pre-trained transformers. In NeurIPS, 2020.

[45] Yucong Luo, Yitong Zhou, Mingyue Cheng, Jiahao Wang, Daoyu Wang, Tingyue Pan, and Jintao Zhang. Time series forecasting as reasoning: A slow-thinking approach with reinforced llms. arXiv preprint arXiv:2506.10630, 2025.

[46] Takuya Akiba, Shotaro Sano, Toshihiko Yanase, Takeru Ohta, and Masanori Koyama. Optuna: A next-generation hyperparameter optimization framework. In Proceedings ofthe 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, pages 2623–2631, 2019.

[47] Leo Breiman. Random forests. Machine Learning, 45(1):5–32, 2001.

[48] Pierre Geurts, Damien Ernst, and Louis Wehenkel. Extremely randomized trees. Machine Learning, 63(1):3–42, 2006.

[49] Yury Gorishniy, Ivan Rubachev, and Artem Babenko. On embeddings for numerical features in tabular deep learning. In Advances in Neural Information Processing Systems, volume 35, pages 24991–25004, 2022.

[50] Yury Gorishniy, Akim Kotelnikov, and Artem Babenko. TabM: Advancing tabular deep learning with parameterefficient ensembling. In International Conference on Learning Representations, 2025. arXiv:2410.24210.

[51] Jiahuan Yan, Jintai Chen, Jian Wu, et al. T2G-Former: Organizing tabular features into relation graphs promotes heterogeneous feature interaction. In Proceedings ofthe AAAI Conference on Artificial Intelligence, 2023.

[52] Gowthami Somepalli, Micah Goldblum, Avi Schwarzschild, C.Bayan Bruss, and Tom Goldstein. SAINT:<sup>˜</sup> Improved neural networks for tabular data via row attention and contrastive pre-training. arXiv preprint arXiv:2106.01342, 2021.

[53] Weiping Song, Chence Shi, Zhiping Xiao, Zhijian Duan, Yewen Xu, Ming Zhang, and Jian Tang. AutoInt: Automatic feature interaction learning via self-attentive neural networks. In Proceedings of the 28th ACM International Conference on Information and Knowledge Management, pages 1161–1170, 2019.

[54] Ruoxi Wang, Rakesh Shivanna, Derek Z. Cheng, Sagar Jain, Dong Lin, Lichan Hong, and Ed H. Chi. DCN V2: Improved deep & cross network and practical lessons for web-scale learning to rank systems. In Proceedings of the Web Conference 2021, pages 1785–1797, 2021.

[55] Jintai Chen, Jiahuan Yan, Qiyuan Chen, Danny Ziyi Chen, Jian Wu, and Jimeng Sun. ExcelFormer: A neural network surpassing GBDTs on tabular data. arXiv preprint arXiv:2301.02819, 2023.

[56] Yury Gorishniy, Ivan Rubachev, Nikolay Kartashev, Daniil Shlenskii, Akim Kotelnikov, and Artem Babenko. TabR: Tabular deep learning meets nearest neighbors in 2023. In International Conference on Learning Representations, 2024.

[57] Han-Jia Ye et al. Modern neighborhood components analysis. arXiv preprint arXiv:2407.03257, 2024.

[58] Sercan Ö. Arik and Tomas Pfister. TabNet: Attentive interpretable tabular learning. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 35, pages 6679–6687, 2021.

[59] Günter Klambauer, Thomas Unterthiner, Andreas Mayr, and Sepp Hochreiter. Self-normalizing neural networks. In Advances in Neural Information Processing Systems, volume 30, 2017.

[60] Alan Jeffares, Tennison Liu, Jonathan Crabbé, Fergus Imrie, and Mihaela van der Schaar. TANGOS: Regularizing tabular neural networks through gradient orthogonalization and specialization. In International Conference on Learning Representations, 2023.

[61] Chao Ye et al. PTaRL: Prototype-based tabular representation learning via space calibration. In International Conference on Learning Representations, 2024.

[62] Sarkhan Badirli, Xuanqing Liu, Zhengming Xing, Avradeep Bhowmik, Khoa Doan, and Sathiya S. Keerthi. Gradient boosting neural networks: GrowNet. arXiv preprint arXiv:2002.07971, 2020.

[63] Youssef Nader, Leon Sixt, and Tim Landgraf. DNNR: Differential nearest neighbors regression. In Proceeding of the 39th International Conference on Machine Learning, pages 16296–16317, 2022.

[64] Jing Wu, Suiyao Chen, Qi Zhao, Renat Sergazinov, Chen Li, Shengjie Liu, Chongchao Zhao, Tianpei Xie, Hanqing Guo, Cheng Ji, Daniel Cociorva, and Hakan Brunzell. SwitchTab: Switched autoencoders are effective tabular learners. In Proceedings ofthe AAAI Conference on Artificial Intelligence, pages 15924–15933, 2024.

[65] Sascha Marton, Stefan Lüdtke, Christian Bartelt, and Heiner Stuckenschmidt. GRANDE: Gradient-based decision tree ensembles. In International Conference on Learning Representations, 2024.

[66] Shiyu Wang et al. AMFormer: Arithmetic multihead attention for tabular data. In arXiv preprint arXiv:2302.08970, 2023.

[67] Bingzhao Liu et al. TabPTM: Pre-trained model for tabular data via meta-representation. arXiv preprint arXiv:2303.09884, 2023.

[68] Andrei Margeloiu, Xiangjian Jiang, Nikola Simidjievski, and Mateja Jamnik. TabEBM: A tabular data augmentation method with distinct class-specific energy-based models. Advances in Neural Information Processing Systems, 37:72094–72144, 2024.

[69] Janez Demšar. Statistical comparisons of classifiers over multiple data sets. Journal of Machine Learning Research, 7:1–30, 2006.

[70] Yoav Benjamini and Yosef Hochberg. Controlling the false discovery rate: a practical and powerful approach to multiple testing. Journal ofthe Royal statistical society: series B (Methodological), 57(1):289–300, 1995.

[71] Jingyang Yuan, Huazuo Gao, Damai Dai, Junyu Luo, Liang Zhao, Zhengyan Zhang, Zhenda Xie, Yuxing Wei, Lean Wang, Zhiping Xiao, Yuqing Wang, Chong Ruan, Ming Zhang, Wenfeng Liang, and Wangding Zeng. Native sparse attention: Hardware-aligned and natively trainable sparse attention. In Wanxiang Che, Joyce Nabende, Ekaterina Shutova, and Mohammad Taher Pilehvar, editors, Proceedings of the 63rd Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 23078–23097, Vienna, Austria, July 2025. Association for Computational Linguistics. ISBN 979-8-89176-251-0.

## A Theoretical Intuition of GridLoss

Gradients and Monotonicity. GridLoss maps a scalar value x to a soft grid index

$$
b ( \boldsymbol { x } ) = \sum _ { k = 0 } ^ { K } k w _ { k } ( \boldsymbol { x } ) , \qquad w _ { k } ( \boldsymbol { x } ) = \frac { \exp ( - | \boldsymbol { x } - \boldsymbol { e } _ { k } | / \tau ) } { \sum _ { j = 0 } ^ { K } \exp ( - | \boldsymbol { x } - \boldsymbol { e } _ { j } | / \tau ) } .
$$

This mapping can be viewed as a smooth approximation to hard binning: for finite temperature τ, the weights concentrate around nearby grid anchors, so $b ( x )$ behaves like a differentiable coordinate on the target grid. The derivative $\partial b ( x ) / \partial x$ is controlled by the temperature and grid resolution, yielding bounded gradients for finite τ. As x increases, mass shifts from lower-index anchors to higher-index anchors, making $b ( x )$ nondecreasing on an ordered grid. Thus, GridLoss provides an ordinal training signal: it encourages predictions not only to reduce pointwise error, but also to occupy the correct relative position along the target scale.

Why Not KL/Wasserstein Distribution Matching. Distributional objectives such as KL divergence or Wasserstein distance typically require constructing normalized distributions, such as histograms or kernel-smoothed densities, over predicted and true targets. In per-sample regression, this introduces additional design choices, including binning, smoothing, regularization, and, for optimal transport, potentially iterative solvers. Moreover, batch-level distribution matching can encourage agreement between marginal target distributions without directly enforcing alignment between each prediction–target pair. GridLoss instead performs a sample-wise comparison by mapping each $( \bar { y } _ { i } , y _ { i } )$ pair onto a shared grid and penalizing their soft-index distance. This captures ordinal geometry with low overhead, controlled mainly by the grid resolution K and temperature τ .

Why Pointwise Regression Losses Alone Can Be Insufficient. Standard pointwise losses optimize residual magnitudes but do not explicitly encode ordinal placement on the target scale. MSE and RMSE emphasize large residuals and can over-focus on outliers; MAE and robust variants, such as Huber and log-cosh, improve robustness but still compare predictions and targets only through pointwise deviations. MAPE can be unstable near zero targets and may overweight small values, while quantile loss targets conditional quantiles rather than the conditional mean. GridLoss complements these objectives by providing an ordinal, calibration-oriented signal in grid space.

Ablation Protocol. We ablate two aspects of GridLoss. First, we compare different regression objectives, including MSE, MAE, Huber, log-cosh, GridLoss, and mixed objectives

$$
\lambda \mathcal { L } _ { \mathrm { H u b e r } } + ( 1 - \lambda ) \mathcal { L } _ { \mathrm { G r i d } } , \qquad \lambda \in \{ 1 . 0 , 0 . 9 , 0 . 7 , 0 . 5 , 0 . 3 , 0 . 0 \} .
$$

Second, we study the sensitivity of GridLoss to its hyperparameters, using

$$
K \in \{ 5 0 , 1 0 0 , 2 0 0 , 5 0 0 , 1 0 0 0 \} , \qquad \tau \in \{ 0 . 1 , 0 . 3 , 1 . 0 , 3 . 0 , 1 0 . 0 \} .
$$

In addition to RMSE, MAE, and $R ^ { 2 }$ , we report ordinal and calibration-oriented metrics, including Spearman’s $\rho$ and bin accuracy within ±1 bin, to quantify improvements in target-scale alignment.

## B Proofs of Theoretical Results

## B.1 ASIM Complexity

The full statement of Proposition 3.1 is as follows.

Proposition B.1 (Time and Memory Complexity of ASIM). Let D be the number ofscalarfeatures and let $d _ { \mathrm { t o k } }$ be the token dimension perfeature. Let $\dot { X _ { \mathrm { t o k } } } \in \dot { \mathbb { R } } ^ { D \times \bar { d } _ { \mathrm { t o k } } }$ denote the tokenized feature representation. ASIM computes

$$
\begin{array} { r } { \tilde { X } = \mathcal { A } _ { \mathrm { s p a r s e } } ( X _ { \mathrm { t o k } } ; s , b , r , m , H , d _ { h } ) \in \mathbb { R } ^ { D \times d _ { \mathrm { t o k } } } , } \end{array}\tag{12}
$$

$$
U = w _ { 1 } X _ { \mathrm { t o k } } + \tilde { X } \in \mathbb { R } ^ { D \times d _ { \mathrm { t o k } } } ,\tag{13}
$$

$$
Z = \mathcal { M } ( U ) \in \mathbb { R } ^ { D \times d _ { \mathrm { t o k } } } ,\tag{14}
$$

$$
Y = w _ { 2 } U + Z + w _ { 3 } X _ { \mathrm { t o k } } \in \mathbb { R } ^ { D \times d _ { \mathrm { t o k } } } ,\tag{15}
$$

followed by $y _ { \mathrm { f e a t } } = \mathrm { m e a n } _ { \mathrm { t o k } } ( Y ) \in \mathbb { R } ^ { D }$ . Assume the sparse attention operator attends, for each query feature, to at most

$$
L : = L ( s , b , r , m )
$$

feature positions, where L is determined by the fixed sparse configuration. $H d _ { \mathrm { t o k } } = H d _ { h }$ , then the sparse interaction part of one ASIM layer has complexity

$$
\mathcal { O } ( D H d _ { h } L ) = \mathcal { O } ( D d _ { \mathrm { t o k } } L )
$$

and activation memory

$$
\mathcal { O } ( D ( d _ { \mathrm { t o k } } + L ) ) .
$$

Including the token projections, FTM operator, residual blending, and token-wise aggregation, the overall ASIM computation remains linear in D underfixed d<sub>tok</sub>, H, d<sub>h</sub>,fixedfeature mini-batch size $B _ { f }$ , and sparse hyperparameters $( s , b , r , m )$

Proof. The sparse attention path restricts each query feature to at most L attended feature positions. Across D query features and H attention heads with head dimension $d _ { h }$ , the sparse dot-product and value-aggregation cost is

$$
\mathcal { O } ( D H d _ { h } L ) = \mathcal { O } ( D d _ { \mathrm { t o k } } L ) ,
$$

using $d _ { \mathrm { t o k } } = H d _ { h }$ . If sparse attention weights or indices are materialized, the sparse attention pattern contributes $\mathcal { O } ( \bar { D } L )$ storage, while the token representations contribute $\mathcal { O } ( D d _ { \mathrm { t o k } } )$ storage. Thus, the sparse attention activation memory is $\mathcal { O } \bar { ( } D ( d _ { \mathrm { t o k } } + L ) )$ ).

The token projections contribute $\mathcal { O } ( D d _ { \mathrm { t o k } } ^ { 2 } )$ computation under standard linear projections, which is linear in D for fixed $d _ { \mathrm { t o k } }$ . The FTM operator $\mathcal { M } ( \cdot )$ applies lightweight embedding-wise mixing and feature-wise mini-batch mixing to $U \in \mathbb { R } ^ { D \times d _ { \mathrm { t o k } } }$ , contributing $O ( D d _ { \mathrm { t o k } } + D B _ { f } )$ computation under fixed token dimension and fixed feature mini-batch size $B _ { f }$ . The residual blending operations and the token-wise mean projection also require $O ( D d _ { \mathrm { t o k } } )$ computation and storage. Therefore, under fixed sparse hyperparameters, fixed token dimension, and fixed feature mini-batch size, both compute and activation memory scale linearly with the feature dimension $D .$ □

## B.2 Conditional SNR Improvement under Sparse Feature Selection

We provide the formal statement and proof of Proposition 3.2. The result is a simplified conditional analysis: if the sparse support covers the informative features while excluding most background dimensions, then sparse aggregation improves the signal-to-noise ratio relative to dense aggregation.

Proposition B.2 (Conditional SNR Improvement under Sparse Feature Selection). Let $\mathbf { x } = [ \mathbf { x } _ { S } ; \mathbf { x } _ { \mathcal { N } } ] \in \mathbb { R } ^ { D }$ be partitioned into informative features indexed by S, with $| S | = s ,$ , and background noise features indexed by ${ \mathcal { N } } =$ $\left\{ 1 , \ldots , D \right\} \backslash S .$ Assume that the noisefeatures are independent with mean zero and variance $\sigma ^ { 2 }$ . Let

$$
\mu _ { \mathcal { S } } = \frac { 1 } { s } \sum _ { d \in \mathcal { S } } x _ { d }
$$

denote the average signal over informativefeatures. Define

$$
A _ { \mathrm { d e n s e } } ( \mathbf { x } ) = \frac { 1 } { D } \sum _ { d = 1 } ^ { D } x _ { d } , \qquad A _ { \mathrm { s p a r s e } } ( \mathbf { x } ; \mathcal { T } ) = \frac { 1 } { | \mathcal { T } | } \sum _ { d \in \mathcal { T } } x _ { d } ,
$$

where $\mathcal { T } \subseteq \{ 1 , \ldots , D \}$ is a sparse support satisfying $s \subseteq \tau$ and $| \mathcal { T } | = m b$

The noise variance contributions are

$$
\mathrm { V a r } _ { \mathcal { N } } [ A _ { \mathrm { d e n s e } } ] = \frac { D - s } { D ^ { 2 } } \sigma ^ { 2 } , \qquad \mathrm { V a r } _ { \mathcal { N } } [ A _ { \mathrm { s p a r s e } } ] = \frac { m b - s } { ( m b ) ^ { 2 } } \sigma ^ { 2 } .
$$

Moreover, the corresponding signal-to-noise ratios are

$$
\mathrm { S N R } _ { \mathrm { d e n s e } } = \frac { s | \mu _ { S } | } { \sigma \sqrt { D - s } } ,
$$

and, for mb $> s ,$

$$
\mathrm { S N R } _ { \mathrm { s p a r s e } } = \frac { s | \mu _ { S } | } { \sigma \sqrt { m b - s } } .
$$

Thus, when $\tau$ covers the informative features while excluding most background dimensions, sparse aggregation improves SNR by afactor of

$$
{ \frac { \mathrm { S N R } _ { \mathrm { s p a r s e } } } { \mathrm { S N R } _ { \mathrm { d e n s e } } } } = { \sqrt { \frac { D - s } { m b - s } } } , \qquad m b > s .
$$

In the ideal case $m b = s ,$ the sparse support contains no background noise, so the noise contribution vanishes under this simplified model.

Proof. For dense aggregation,

$$
A _ { \mathrm { d e n s e } } ( \mathbf { x } ) = \frac { 1 } { D } \sum _ { d \in \cal { S } } x _ { d } + \frac { 1 } { D } \sum _ { d \in \cal { N } } x _ { d } .
$$

Treating the informative features as fixed, the variance comes only from the noise term:

$$
\operatorname { V a r } _ { \mathcal { N } } [ A _ { \mathrm { d e n s e } } ] = \operatorname { V a r } \left[ { \frac { 1 } { D } } \sum _ { d \in { \mathcal { N } } } x _ { d } \right] = { \frac { 1 } { D ^ { 2 } } } \sum _ { d \in { \mathcal { N } } } \operatorname { V a r } ( x _ { d } ) = { \frac { D - s } { D ^ { 2 } } } \sigma ^ { 2 } .
$$

The signal component of dense aggregation is

$$
\mathbb { E } _ { \mathcal { N } } [ A _ { \mathrm { d e n s e } } ] = \frac { 1 } { D } \sum _ { d \in \mathcal { S } } x _ { d } = \frac { s } { D } \mu _ { \mathcal { S } } .
$$

Hence

$$
\mathrm { S N R } _ { \mathrm { d e n s e } } = { \frac { | s \mu _ { S } / D | } { \sigma { \sqrt { D - s } } / D } } = { \frac { s | \mu _ { S } | } { \sigma { \sqrt { D - s } } } } .
$$

For sparse aggregation, since $s \subseteq \tau$ and $| { \mathcal { T } } | = m b ,$ the number of background noise features included in $\tau$ is $m b - s .$ Therefore,

$$
\operatorname { V a r } _ { \mathcal { N } } [ A _ { \mathrm { s p a r s e } } ] = \operatorname { V a r } \left[ \frac { 1 } { m b } \sum _ { d \in \mathcal { T } \cap \mathcal { N } } x _ { d } \right] = \frac { m b - s } { ( m b ) ^ { 2 } } \sigma ^ { 2 } .
$$

The signal component is

$$
\mathbb { E } _ { \mathcal { N } } [ A _ { \mathrm { s p a r s e } } ] = \frac { 1 } { m b } \sum _ { d \in \mathcal { S } } x _ { d } = \frac { s } { m b } \mu _ { \mathcal { S } } .
$$

For $m b > s ,$ , this yields

$$
\mathrm { S N R } _ { \mathrm { s p a r s e } } = { \frac { | s \mu _ { S } / m b | } { \sigma { \sqrt { m b - s } } / m b } } = { \frac { s | \mu _ { S } | } { \sigma { \sqrt { m b - s } } } } .
$$

Taking the ratio gives

$$
{ \frac { \mathrm { S N R } _ { \mathrm { s p a r s e } } } { \mathrm { S N R } _ { \mathrm { d e n s e } } } } = { \sqrt { \frac { D - s } { m b - s } } } .
$$

If $m b = s ,$ then $\tau$ contains no background noise features, so the noise variance of $A _ { \mathrm { s p a r s e } }$ is zero. This completes the proof. □

Corollary B.3 (Scaling with Feature Dimension). Fix the number of informative features s and let the sparse budget be $m b = s + \delta ,$ for afixed $\delta \geq 0 . \ A s \ D  \infty ,$ , the dense aggregation SNR satisfies

$$
\mathrm { S N R } _ { \mathrm { d e n s e } } = \frac { s | \mu _ { S } | } { \sigma \sqrt { D - s } } = \mathcal { O } ( D ^ { - 1 / 2 } ) ,
$$

whereas, $f o r \delta > 0 ,$

$$
\mathrm { S N R } _ { \mathrm { s p a r s e } } = { \frac { s | \mu _ { S } | } { \sigma \sqrt { \delta } } } = \Theta ( 1 ) .
$$

$H \delta = 0 ,$ , the sparse support contains no background noise and the noise contribution vanishes under the simplified model. Thus, dense aggregation suffersfrom signal dilution as the ambientfeature dimension grows, whereas sparse aggregation preserves a dimension-independent SNR when the sparse support remains tight.

Proof. The dense SNR expression follows directly from Proposition B.2:

$$
\mathrm { S N R } _ { \mathrm { d e n s e } } = { \frac { s | \mu _ { S } | } { \sigma { \sqrt { D - s } } } } = \mathcal { O } ( D ^ { - 1 / 2 } )
$$

for fixed $s , \mu s$ , and σ. For sparse aggregation with m $\boldsymbol { b } = \boldsymbol { s } + \boldsymbol { \delta }$ and $\delta > 0 ,$

$$
\mathrm { S N R } _ { \mathrm { s p a r s e } } = { \frac { s | \mu _ { S } | } { \sigma { \sqrt { m b - s } } } } = { \frac { s | \mu _ { S } | } { \sigma { \sqrt { \delta } } } } ,
$$

which is independent of D. When $\delta = 0 _ { : }$ , the sparse aggregation contains no background noise, so its noise variance is zero under the assumptions of the proposition. □

Remark B.4 (Extension to Softmax Attention Weights). The uniform-weight analysis provides a tractable baseline. For softmax attention with learned queries and keys, the SNR behavior depends on the model’s ability to assign small weights to background features and larger weights to informative features. If the attention mechanism concentrates most mass on S, then the effective support

$$
\mathcal { T } _ { \mathrm { e f f } } = \{ d : \alpha _ { d } > \varepsilon \}
$$

can approach the idealized sparse case. This motivates structurally sparse mechanisms, which encourage restricted feature interaction patterns rather than relying solely on learned dense softmax weights to become sparse in highdimensional noisy settings.

Remark B.5 (Connection to Empirical Results). The conditional analysis in Proposition B.2 and Corollary B.3 provides intuition for the empirical findings in Section 4.1 and Figure 2a:

1. Dense attention. In high-dimensional settings, dense attention mixes informative and background features through all-pair interactions. The analysis suggests that, when only a small subset of features is informative, dense aggregation can dilute the useful signal as D grows.

2. Mamba. SSM-based mixing provides linear-time propagation along the feature axis, but it does not explicitly enforce sparse feature selection. As a result, it may propagate both informative and background dimensions, yielding intermediate performance.

3. ASIM. Structured sparse selection encourages the model to restrict feature interactions to a smaller support. When this support covers informative features while excluding many irrelevant dimensions, the analysis predicts improved SNR. This provides theoretical intuition for why the ASIM design can improve robustness in high-dimensional tabular regression.

## B.3 Properties of GridLoss

Lemma B.6 (Monotonicity of the GridLoss soft index). Fix ordered grid anchors $e _ { 0 } < e _ { 1 } < \dots < e _ { K }$ and temperature $\tau > 0 .$ . For any scalar $x \in \mathbb { R }$ , define

$$
w _ { k } ( x ) ~ = ~ { \frac { \exp ( - | x - e _ { k } | / \tau ) } { \sum _ { j = 0 } ^ { K } \exp ( - | x - e _ { j } | / \tau ) } } , \qquad b ( x ) ~ = ~ \sum _ { k = 0 } ^ { K } k w _ { k } ( x ) .
$$

Then $b ( x )$ is nondecreasing in x; equivalently, $x _ { 1 } \leq x _ { 2 }$ implies $b ( x _ { 1 } ) \leq b ( x _ { 2 } )$

Proof. Let $z _ { k } ( x ) = - | x - e _ { k } | / \tau$ and $w _ { k } ( x ) = \mathrm { s o f t m a x } ( z ( x ) ) _ { k }$ . For $x \notin \{ e _ { 0 } , \ldots , e _ { K } \}$ , we have

$$
z _ { k } ^ { \prime } ( x ) = - { \frac { 1 } { \tau } } \mathrm { s i g n } ( x - e _ { k } ) .
$$

The softmax derivative gives

$$
w _ { k } ^ { \prime } ( x ) = w _ { k } ( x ) \left( z _ { k } ^ { \prime } ( x ) - \sum _ { j = 0 } ^ { K } w _ { j } ( x ) z _ { j } ^ { \prime } ( x ) \right) .
$$

Therefore,

$$
b ^ { \prime } ( x ) = \sum _ { k = 0 } ^ { K } k w _ { k } ^ { \prime } ( x ) = - \frac { 1 } { \tau } \left( \mathbb { E } _ { w } [ k s _ { k } ] - \mathbb { E } _ { w } [ k ] \mathbb { E } _ { w } [ s _ { k } ] \right) = - \frac { 1 } { \tau } \operatorname { C o v } _ { w } ( k , s _ { k } ) ,
$$

where $s _ { k } = \mathrm { s i g n } ( x - e _ { k } )$ and $\mathbb { E } _ { w } [ \cdot ]$ denotes expectation under the weights $\{ w _ { k } ( x ) \} _ { k = 0 } ^ { K } .$ . Since e increases with $e _ { k }$ $k ,$ the sequence $\{ s _ { k } \} _ { k = 0 } ^ { K }$ is nonincreasing in k. Hence $\mathrm { C o v } _ { w } ( k , s _ { k } ) \le 0$ , which implies $b ^ { \prime } ( x ) \geq 0$ wherever the derivative exists. At anchor locations $x = e _ { k }$ , monotonicity follows by continuity and one-sided derivatives. □

Lemma B.7 (Bounded slope and stable GridLoss gradients). Under the same definitions as Lemma $B . 6 ,$ for all x $\not \in \{ e _ { 0 } , \dotsc , e _ { K } \}$

$$
0 \leq b ^ { \prime } ( x ) \leq { \frac { K } { 2 \tau } } .
$$

Consequently, for the per-sample GridLoss term

$$
\ell _ { \mathrm { G r i d } } ( \hat { y } , y ) = \big | b ( \hat { y } ) - b ( y ) \big | ,
$$

any subgradient $g \in \partial _ { \hat { y } } \ell _ { \mathrm { G r i d } } ( \hat { y } , y )$ satisfies

$$
| g | \leq { \frac { K } { 2 \tau } } .
$$

Proof. From the proof of Lemma B.6,

$$
b ^ { \prime } ( x ) = - { \frac { 1 } { \tau } } \operatorname { C o v } _ { w } ( k , s _ { k } ) , \qquad s _ { k } \in \{ - 1 , + 1 \} .
$$

By Cauchy–Schwarz,

$$
| \operatorname { C o v } _ { w } ( k , s _ { k } ) | \leq \sqrt { \operatorname { V a r } _ { w } ( k ) \operatorname { V a r } _ { w } ( s _ { k } ) } \leq \sqrt { \operatorname { V a r } _ { w } ( k ) } ,
$$

since $\mathrm { V a r } _ { w } ( s _ { k } ) \le 1$ . Because $k \in \{ 0 , 1 , \ldots , K \}$ , Popoviciu’s inequality gives

$$
\mathrm { V a r } _ { w } ( k ) \leq \frac { K ^ { 2 } } { 4 } .
$$

Therefore,

$$
0 \leq b ^ { \prime } ( x ) \leq { \frac { K } { 2 \tau } } .
$$

For $\ell _ { \mathrm { G r i d } } ( \hat { y } , y ) = | b ( \hat { y } ) - b ( y ) |$ |, the chain rule for subgradients yields

$$
\begin{array} { r } { \partial _ { \hat { y } } \ell _ { \mathrm { G r i d } } ( \hat { y } , y ) \subseteq \mathrm { s i g n } \big ( b ( \hat { y } ) - b ( y ) \big ) \cdot \partial _ { \hat { y } } b ( \hat { y } ) , } \end{array}
$$

so every subgradient satisfies $| g | \le K / ( 2 \tau )$

Lemma B.8 (Lipschitz continuity of the soft index). Under the conditions of Lemma B.7, for all $x _ { 1 } , x _ { 2 } \in \mathbb { R }$

$$
| b ( x _ { 1 } ) - b ( x _ { 2 } ) | \leq \frac { K } { 2 \tau } | x _ { 1 } - x _ { 2 } | .
$$

Proof. On intervals not containing grid anchors, the result follows from the mean value theorem and the slope bound in Lemma B.7. Since b is continuous at the anchors, the bound extends to arbitrary $x _ { 1 } , x _ { 2 } \in \mathbb { R }$ by partitioning the interval between $x _ { 1 }$ and $x _ { 2 }$ at any anchors it contains and summing the bounds over the subintervals. □

## B.4 Asymptotic Consistency of GridLoss

We show that normalized GridLoss recovers the mean absolute error in a high-resolution, low-temperature limit. This provides a link between GridLoss and classical robust regression losses.

Fixed-interval analysis versus batch-adaptive implementation. For theoretical clarity, the analysis assumes a fixed interval $[ \alpha , \beta ]$ . In implementation, the grid interval is batch-adaptive: α and $\beta$ are computed from the current mini-batch, for example using the batch extrema of y and yˆ, and are detached from the computation graph. The analysis applies conditionally within each batch-defined interval because the key arguments depend on the local grid geometry and the relative distances to nearby anchors.

Theorem B.9 (GridLoss Consistency). Let $\mathcal { L } _ { \mathrm { G r i d } } ^ { ( K , \tau ) } ( \hat { y } , y )$ denote GridLoss with K bins and temperature τ on a fixed interval $[ \alpha , \beta ]$ . Define the normalized GridLoss

$$
\widetilde { \mathcal { L } } _ { \mathrm { G r i d } } ^ { ( K , \tau ) } ( \boldsymbol { \hat { y } } , \boldsymbol { y } ) = \frac { \beta - \alpha } { K } \mathcal { L } _ { \mathrm { G r i d } } ^ { ( K , \tau ) } ( \boldsymbol { \hat { y } } , \boldsymbol { y } ) .
$$

Then, for any $\hat { y } , y \in [ \alpha , \beta ]$

$$
\operatorname* { l i m } _ { \tau  0 ^ { + } } \operatorname* { l i m } _ { K  \infty } \widetilde { \mathcal { L } } _ { \mathrm { G r i d } } ^ { ( K , \tau ) } ( \hat { y } , y ) = | \hat { y } - y | .
$$

That $i s ,$ normalized GridLoss converges to the $\ell _ { 1 }$ loss as the grid becomes infinitely fine and the soft assignments become sharp.

Proof. Let

$$
\Delta = \frac { \beta - \alpha } { K } , \qquad e _ { k } = \alpha + k \Delta , \qquad k = 0 , \ldots , K .
$$

For a scalar $x \in [ \alpha , \beta ]$ , define the soft index

$$
b ^ { ( K , \tau ) } ( x ) = \sum _ { k = 0 } ^ { K } k w _ { k } ( x ) , \qquad w _ { k } ( x ) = \frac { \exp ( - | x - e _ { k } | / \tau ) } { \sum _ { j = 0 } ^ { K } \exp ( - | x - e _ { j } | / \tau ) } .
$$

The normalized soft coordinate is $\Delta b ^ { ( K , \tau ) } ( x )$ . As $K  \infty$ with τ fixed, the discrete grid becomes dense and the weighted sum converges to the expectation of a truncated Laplace-type density on $[ \alpha , \beta ]$ centered at x:

$$
\Delta b ^ { ( K , \tau ) } ( x ) \longrightarrow B _ { \tau } ( x ) : = \frac { \int _ { \alpha } ^ { \beta } u \exp ( - | u - x | / \tau ) d u } { \int _ { \alpha } ^ { \beta } \exp ( - | u - x | / \tau ) d u } .
$$

The quantity $B _ { \tau } ( x )$ is a smoothed version of $x .$ Since the exponential kernel concentrates around x as $\tau \to 0 ^ { + }$ , we have

$$
B _ { \tau } ( x )  x \qquad \mathrm { f o r ~ e v e r y ~ } x \in [ \alpha , \beta ] .
$$

Therefore,

$$
\operatorname* { l i m } _ { \tau  0 ^ { + } } \operatorname* { l i m } _ { K  \infty } \Delta b ^ { ( K , \tau ) } ( x ) = x .
$$

Applying this to $\hat { y }$ and y gives

$$
\operatorname* { l i m } _ { \tau \to 0 ^ { + } } \operatorname* { l i m } _ { K \to \infty } \widetilde { \mathcal { L } } _ { \mathrm { G r i d } } ^ { ( K , \tau ) } ( \hat { y } , y ) = \operatorname* { l i m } _ { \tau \to 0 ^ { + } } | B _ { \tau } ( \hat { y } ) - B _ { \tau } ( y ) | = | \hat { y } - y | ,
$$

which completes the proof.

Corollary B.10 (Approximation Behavior). On a fixed interval $[ \alpha , \beta ] _ { ; }$ , normalized GridLoss approaches an MAE-like ordinal distance as the grid resolution increases and the temperature decreases. Finer grids reduce discretization error, whilefinite temperature smooths the hard binning operation and yields bounded gradients. Thus, K controls target-scale resolution, whereas τ controls the smoothness–sharpness trade-off.

Proof. The convergence to MAE follows from Theorem B.9. The role of finite temperature follows from Lemma B.7, which bounds the gradient magnitude by $K / ( 2 \tau )$ for fixed K and τ. Thus, larger K gives a finer discretization of the target interval, while τ controls how sharply each scalar value is assigned to nearby grid anchors. □

Remark B.11 (Relationship to Soft Label Regression). Theorem B.9 positions GridLoss within a broader family of soft-binning regression losses. Related approaches include:

• Label Distribution Learning (LDL): LDL represents targets as distributions over discrete labels; GridLoss can be viewed as using Gibbs-type soft assignments centered at yˆ and y.

• Ordinal Regression: Ordinal regression methods often enforce rank consistency through hard or probabilistic class decompositions. GridLoss instead uses soft indices to provide a differentiable relaxation of ordinal positioning for continuous targets.

• Histogram or Distributional Losses: Histogram losses and Earth Mover’s Distance compare target distributions. GridLoss compares each prediction–target pair through its soft grid index, yielding a sample-wise ordinal distance.

The consistency result shows that GridLoss approaches MAE in the high-resolution, low-temperature limit while maintaining smooth gradients for finite τ.

Remark B.12 (Practical Implications). Theorem B.9 has several practical implications:

1. Hyperparameter guidance: Increasing K improves target-scale resolution, but very large K with very small τ can increase gradient magnitudes, as Lemma B.7 shows. Thus, K and τ should be selected jointly to balance resolution and stability.

2. MAE-like behavior: Since normalized GridLoss approaches MAE in the high-resolution, low-temperature limit, it inherits an MAE-like robustness profile while retaining smooth gradients at finite temperature.

3. Ordinal structure: Unlike standard pointwise losses, GridLoss explicitly operates in a discretized ordinal space, encouraging predictions to respect relative placement along the target range.

Proposition B.13 (GridLoss as Interpolation). For fixed K and varying τ , GridLoss interpolates between:

$$
\begin{array} { r } { I . \ A s \ \tau  \infty , w _ { k } ( x )  1 / ( K + 1 ) f o r \ a l l \ k , s o \ b ( x )  K / 2 a n d \ \mathcal { L } _ { \mathrm { G r i d } } ^ { ( K , \tau ) } ( \hat { y } , y )  0 f o r \ a l l \ \hat { y } , y . } \end{array}
$$

2. $A s \tau \to 0 ^ { + }$ , the weights concentrate on the nearest anchor, so $\mathcal { L } _ { \mathrm { G r i d } } ^ { ( K , \tau ) } ( \hat { y } , y )$ converges to a hard ordinal bin distance.

As K increases, the grid provides a finer discretization of the target interval. Under the high-resolution and lowtemperature limit in Theorem B.9, the normalized GridLoss converges to MAE.

Proof. $\mathbf { A s } \tau \to \infty ,$ each exponential term satisfies $\exp ( - | x - e _ { k } | / \tau ) \to 1 , \operatorname { s o } w _ { k } ( x ) \to 1 / ( K + 1 )$ for all k. Hence

$$
b ( \boldsymbol { x } )  \frac { 1 } { K + 1 } \sum _ { k = 0 } ^ { K } k = \frac { K } { 2 }
$$

for all $x ,$ and the index distance between any $\hat { y }$ and y converges to zero.

As $\tau \to 0 ^ { + }$ , the weights concentrate on the nearest grid anchor. Thus $b ( x )$ converges to the nearest-anchor index, and GridLoss becomes the absolute distance between hard ordinal bin indices. Finally, the convergence of normalized GridLoss to MAE under joint high-resolution and low-temperature behavior follows from Theorem B.9. □

## B.5 Objective Interpretation

Proposition B.14 (Objective Interpretation of RISE as Reweighted Empirical Risk). Let $\{ ( x _ { i } , y _ { i } ) \} _ { i = 1 } ^ { N }$ be the training set and let $\ell _ { i } ( \theta )$ denote the per-sample loss at parameters $\theta , e . g . , \ell _ { i } ( \theta ) = \ell ( f _ { \theta } ( x _ { i } ) , y _ { i } )$ . At RISE refresh step e, define a positive weight

$$
w _ { i } ^ { ( e ) } = \varphi ( q _ { i } ^ { ( e ) } ) > 0 ,
$$

where $q _ { i } ^ { ( e ) } \in \{ 0 , \ldots , G - 1 \}$ is the empirical loss-quantile bin index of sample i, computed from $\{ \ell _ { j } ( \theta _ { e } ) \} _ { j = 1 } ^ { N }$ , and $\varphi ( \cdot )$ is a monotone increasing bin-to-weight map. Consider SGD where each mini-batch is drawn by weighted sampling using probabilities

$$
p _ { i } ^ { ( e ) } = \frac { w _ { i } ^ { ( e ) } } { \sum _ { j = 1 } ^ { N } w _ { j } ^ { ( e ) } } ,
$$

and the update uses the uncorrected stochastic gradient $\nabla _ { \boldsymbol { \theta } } \ell _ { i } ( \boldsymbol { \theta } )$ ofthe sampled example, without inverse-probability correction. Conditional on the weights $\{ w _ { i } ^ { ( e ) } \}$ , the expected stochastic gradient equals the gradient ofthe reweighted empirical risk

$$
\mathcal { L } _ { \mathrm { R I S E } } ^ { ( e ) } ( \theta ) = \sum _ { i = 1 } ^ { N } p _ { i } ^ { ( e ) } \ell _ { i } ( \theta ) = \frac { \sum _ { i = 1 } ^ { N } w _ { i } ^ { ( e ) } \ell _ { i } ( \theta ) } { \sum _ { j = 1 } ^ { N } w _ { j } ^ { ( e ) } } .
$$

That $i s ,$

$$
\begin{array} { r } { \mathbb { E } _ { i \sim p ^ { ( e ) } } [ \nabla _ { \theta } \ell _ { i } ( \theta ) ] = \nabla _ { \theta } \mathcal { L } _ { \mathrm { R I S E } } ^ { ( e ) } ( \theta ) , } \end{array}
$$

where the weights are treated as fixed between refresh steps. Therefore, RISE can be interpreted as following stochastic gradients of a sequence of reweighted empirical risks that emphasize higher-loss quantile bins when $\varphi ( \cdot )$ is increasing.

Proof. Fix a refresh step e and condition on the weights $\{ w _ { i } ^ { ( e ) } \}$ and probabilities $\{ p _ { i } ^ { ( e ) } \}$ . A single-sample stochastic gradient drawn by weighted sampling satisfies

$$
\mathbb { E } _ { i \sim p ^ { ( e ) } } [ \nabla _ { \theta } \ell _ { i } ( \theta ) ] = \sum _ { i = 1 } ^ { N } p _ { i } ^ { ( e ) } \nabla _ { \theta } \ell _ { i } ( \theta ) .
$$

Between refresh steps, the sampling weights are fixed with respect to the current optimization variable θ. Hence

$$
\sum _ { i = 1 } ^ { N } p _ { i } ^ { ( e ) } \nabla _ { \theta } \ell _ { i } ( \theta ) = \nabla _ { \theta } \left( \sum _ { i = 1 } ^ { N } p _ { i } ^ { ( e ) } \ell _ { i } ( \theta ) \right) = \nabla _ { \theta } \mathcal { L } _ { \mathrm { R I S E } } ^ { ( e ) } ( \theta ) .
$$

Substituting

$$
p _ { i } ^ { ( e ) } = \frac { w _ { i } ^ { ( e ) } } { \sum _ { j = 1 } ^ { N } w _ { j } ^ { ( e ) } }
$$

gives the normalized reweighted empirical risk. If φ(·) is increasing in the loss-quantile bin, higher-loss samples receive larger sampling probabilities, so the resulting stochastic gradients emphasize the tail of the empirical loss distribution. □

## C Full Benchmark Results and Statistical Analyses

## C.1 Full Benchmark Results

Table 2 extends the summary in Table 1 with per-model RMSE across all nine benchmarks. This table compares against classical baselines [40], tree-based ensembles [47; 5; 11; 10; 48], MLP-style architectures [23; 49; 50; 51], transformer-based models [22; 52; 53; 54; 55], retrieval-based methods [56; 57; 4; 58], other deep tabular architectures [59; 60; 61; 1; 62; 63; 64; 65], and tabular foundation models (either pretrained parametric models or non-parametric inferences) [66; 17; 18; 67; 68]. Symbols indicate runs that could not be completed due to time constraints (†), GPU memory limitations (‡), training divergence (§), hard-coded dataset size restrictions (∗), or prohibitive algorithm complexity (♢). TabNSM achieves the lowest RMSE on seven of nine benchmarks, with competitive performance on CP where LightGBM remains the strongest baseline.

## C.2 Statistical Significance of Results

We assess whether TabNSM’s improvements over baselines are statistically significant using two complementary non-parametric tests [69].

Friedman Test We first test the global null hypothesis that all models perform equally across benchmarks. For each benchmark we assign integer ranks $( 1 = \mathrm { l o w e s t R M S E = b e s t } )$ , then apply the Friedman test to the resulting k × N rank matrix. We use the $k = 3 7$ models that completed all five benchmarks with reliable results (SA, CP, AQ, TO, SC), giving $N = 5 ;$ the two excluded models (SAINT, DNNR) diverged or ran out of memory on three of the five. The Friedman statistic is $\chi ^ { 2 } ( 3 6 ) = 7 9 . 3 , p = 4 . 2 5 \times 1 0 ^ { - 5 }$ , rejecting the null hypothesis and confirming that performance differences across models are not due to chance. Figure 3 shows the resulting Critical Difference diagram.

![](images/cccbc01ec4db5b908b132091a7c0ec754e81ed68cd2ecbb1f7628b7e0749ee26.jpg)  
Figure 3: Critical Difference diagram (Nemenyi post-hoc test, $\alpha = 0 . 0 5 )$ . Each dot is a model coloured by architectural family; its horizontal position is its average rank across five benchmarks (lower = better). The CD bracket shows the minimum rank difference required for statistical significance $\mathrm { ( C D } = 2 6 . 8 )$ . Models marked with + (GrowNet, SVR-Linear) fall outside TabNSM’s critical difference band and are significantly worse. The large CD is expected when $k \gg N ;$ ; the Friedman test nonetheless confirms a significant overall difference $( p = 4 . 2 5 \times \bar { 1 0 } ^ { - 5 } )$ .

The Nemenyi post-hoc test is known to be conservative when $k \gg N :$ with 37 models and only 5 benchmarks, the critical difference is $\mathrm { C D } = 2 6 . 8$ rank units, leaving most pairwise comparisons inconclusive regardless of the true effect. This is a known limitation of the Friedman–Nemenyi procedure in wide-but-short rank matrices. The Wilcoxon signed-rank test sidesteps this by testing each pair directly across all nine benchmarks, providing substantially more statistical power. The two tests are therefore not contradictory: Nemenyi confirms only the largest rank gaps with high confidence given $N = 5$ , while Wilcoxon leverages all nine benchmarks to establish significance for all baselines.

Pairwise Wilcoxon Signed-Rank Test Following the rejection of the global null, we conduct 38 one-sided pairwise Wilcoxon signed-rank tests comparing TabNSM against each baseline (H<sub>1</sub>: TabNSM rank < baseline rank) across all nine benchmarks, imputing missing values as worst rank. p-values are corrected for multiple comparisons using the Benjamini–Hochberg procedure [70] at $\alpha = 0 . 0 5$ . TabNSM achieves a significantly lower rank than every baseline after correction (all corrected $p \leq 0 . 0 2 )$ . The closest competitors by average rank are ResNet $( \bar { r } = 9 . 7 2 , p = 0 . 0 0 2 )$ Extratrees $( \bar { r } = 1 1 . 4 4 , p = 0 . \bar { 0 } 2 0 )$ , MLP $( \bar { r } = 1 1 . 7 8 , p = 0 . 0 0 2 )$ , CatBoost $( \bar { r } = 1 2 . 4 4 , p = 0 . 0 1 0 )$ , and LightGBM $( \bar { r } = 1 2 . 7 8 , p = 0 . 0 1 0 )$ , compared to TabNSM’s average rank of $\bar { r } = 1 . 8 9$ across nine benchmarks. Figure 4 summarises the average ranks and significance results for all baselines.

We note that worst-rank imputation is conservative for TabNSM (which has no missing runs) and liberal for baselines with many incomplete runs; results should therefore be interpreted as an aggregate trend rather than a precise pairwise comparison for models with many incomplete entries.

## D Brief Summary of Baseline Model Setup

All baseline models were implemented in Python using standard libraries. Data were loaded from ARFF files into pandas DataFrames, categorical strings decoded to UTF-8 and converted to numeric where possible, missing values filled with zeros, and the target variable removed before training. The data were split into 80% training+validation and 20% test, with the training+validation portion further split into 90% training and 10% validation. Neural and hybrid models (DeepGBM, TabNet, FT-Transformer) used z-score normalization, while tree-based models used raw values.

Performance was evaluated on the held-out test set using RMSE (primary), MSE, MAE, and $R ^ { 2 }$ . Hyperparameter optimization was performed with Optuna (3-fold CV on training+validation, 20 trials, maximizing negative RMSE). The best configuration was used to train a final model on the training set (using the validation set when supported) and evaluated on the held-out test set.

The code for models marked with <sup>∗</sup> was not provided for the regression task. Models marked with ⋆ were not tuned for most large-dataset benchmarks, often resulting in non-convergence or outlier performance. Therefore, we do not report their results in Table 2.

## E Instance-adaptive feature selectivity

To examine whether TabNSM learns instance-adaptive feature selectivity, we visualize per-instance feature attributions on the test set. Figure 5 shows a feature attribution heatmap, where each row corresponds to a test instance and each column corresponds to an input feature. The attribution score is computed using gradient–embedding attribution and normalized to [0, 1] within each instance. Brighter values indicate features with stronger influence on the model’s regression output for that sample

The heatmap reveals two patterns. First, some features appear as consistently bright columns across many instances, suggesting globally influential predictors. Second, many instances exhibit distinct sparse attribution patterns, indicating that TabNSM dynamically emphasizes different feature subsets depending on the input context. This behavior is consistent with the design of the Adaptive Sparse Interaction Module (ASIM), which encourages instance-adaptive sparse feature interaction rather than relying on a fixed global feature weighting.

## F Dataset Descriptions

This section provides detailed descriptions of the datasets used in our experiments, along with the associated prediction tasks.

topo\_2\_1. The Topo dataset was used, with 1,143 molecular descriptors computed using the Adriana. Molecular structures and response values were taken directly from the original studies. The final attribute in each file was used as the prediction target; for the Topo dataset, this target corresponds to oz267.

Sarcos Robot Arm Dataset. The Sarcos dataset consists of measurements from a 7-degree-of-freedom robotic arm performing various motions. Each instance includes joint positions, velocities, and accelerations, and the prediction task is to estimate the torque of a specific joint (V22). This dataset is commonly used to evaluate regression models for learning inverse dynamics.

Electric Vehicle Population Dataset. This dataset contains records of battery electric and plug-in hybrid vehicle registrations in Washington State. Features include vehicle attributes and registration information, and the task is to predict the electric driving range.

Air Quality The dataset consists of New York City air quality surveillance indicators measured across neighborhoods and time. Features describe pollutant exposure and related environmental health metrics, while the prediction target, Data Value, represents the quantitative measurement of each air quality indicator used for supervised learning.

U.S. Chronic Disease Dataset. The Chronic Disease dataset provides standardized public health indicators collected across U.S. states and territories. The prediction task involves estimating numeric values associated with chronic disease metrics.

Crime Dataset. The Crime dataset consists of crime incident reports from Los Angeles since 2020. Each record includes temporal, categorical, and spatial features, and the task is to predict the geographic area associated with each incident.

Real Estate Sales Dataset. This dataset contains property transaction records reported by municipalities in Connecticut. Given property and transaction features, the task is to predict the sales ratio, a commonly used indicator in real estate valuation and property tax assessment.

Standardized Centralized Alzheimer’s Neuroimaging (SCAN) Dataset. The SC benchmark was constructed from controlled-access multimodal data obtained through the NACC data-request process under the NACC Data Use Agreement. The data include neuroimaging, fluid biomarkers, and demographic variables collected across Alzheimer’s Disease Research Centers. The prediction task is to estimate Mini-Mental State Examination (MMSE) scores.

## F.1 Dataset Links

Eight benchmark datasets are publicly available; NACC/SCAN data are available to researchers through NACC’s data-request process under the NACC Data Use Agreement. Table 4 provides the public source or access page for each dataset.

## G Seed Variation

To assess the robustness of the proposed approach with respect to random initialization and training stochasticity, we repeat each experiment 20 times using different random seeds. Figure 6 reports the distribution of test Root Mean Squared Error (RMSE) across runs using boxplots. The results show that TabNSM maintains stable performance across seeds, with relatively small interquartile ranges and limited variation across runs. This suggests that the reported performance is not driven by a favorable initialization, but reflects consistent behavior under repeated training. Overall, the limited variance indicates that the proposed method is robust to randomness introduced during training.

## H Efficiency and Memory Footprint

Figure 7 and Table 5 compare the practical training time of CatBoost and TabNSM across datasets with varying sample sizes n and feature dimensions d. Runtime is measured under a fixed validation-loss threshold protocol, where training time is recorded until each method reaches a predefined validation-loss level. This threshold-based criterion provides a practical comparison of convergence speed across models with different optimization behavior, although the methods use different computational backends: CatBoost is evaluated on CPU, whereas TabNSM uses GPU acceleration.

As shown in Table 5, CatBoost runtime increases as dataset size and feature dimensionality grow, consistent with the scaling behavior of gradient-boosted decision trees on large tabular datasets. In contrast, TabNSM maintains low runtime across the evaluated settings. This reflects the mini-batch training procedure and the use of sparse feature interaction modeling, which avoids dense all-pair feature mixing and limits the dependence on feature dimensionality. These results suggest that TabNSM provides favorable practical efficiency in large-scale, high-dimensional regression settings.

Tables 6 and 7 further report memory usage on representative large datasets. Since CatBoost is run on CPU, GPU memory is reported only for TabNSM. CPU memory increase is reported for both methods. TabNSM shows moderate GPU memory usage and lower CPU memory increase than CatBoost on the evaluated large datasets. Overall, the results highlight complementary scalability profiles: CatBoost remains a strong CPU-based tree ensemble, while TabNSM offers efficient GPU-accelerated training with favorable memory behavior on large, high-dimensional datasets.

Additional runtime and memory measurements are provided in Appendix H; Figure 8 summarises the accuracy– efficiency trade-off across all evaluated methods.

## I Sparse Feature-wise Attention: Full Formulation

Within ASIM, we treat the D input features as a sequence of token embeddings $X _ { \mathrm { t o k } } \in \mathbb { R } ^ { D \times d _ { \mathrm { t o k } } }$ and apply a structured sparse self-attention operator $\bar { \mathcal { A } } _ { \mathrm { s p a r s e } }$ [2; 71]. Rather than computing all $D ^ { 2 }$ pairwise feature interactions, each feature token attends only to a compact, instance-adaptive subset of other features. This subset is constructed from three complementary components designed to jointly cover local, global, and instance-specific interaction patterns.

Local neighborhood. Features adjacent in the input ordering often encode correlated measurements (e.g., related embedding dimensions or grouped sensor readings). For feature d, we retain the s nearest neighbors as a sliding window:

$$
\mathcal { K } _ { d } ^ { \mathrm { w i n } } = \mathbf { k } _ { d - s : d + s } , \quad \mathcal { V } _ { d } ^ { \mathrm { w i n } } = \mathbf { v } _ { d - s : d + s } .\tag{16}
$$

Compressed global summary. To give each query feature coarse awareness of the full feature set, we partition the D features into blocks of size b (stride $r \leq b )$ and compress each block to a single representative key–value pair via a learnable MLP ϕ:

$$
\begin{array} { r } { \mathcal { K } _ { d } ^ { \mathrm { c m p } } = \left\{ \phi ( \mathbf { k } _ { i b + 1 : i b + b } ) ~ \middle | ~ 0 \leq i \leq \left\lfloor \frac { D - b } { r } \right\rfloor \right\} , } \end{array}\tag{17}
$$

with an analogous $\mathcal { V } _ { d } ^ { \mathrm { c m p } }$ . This reduces global context to $\mathcal { O } ( D / r )$ keys.

Instance-adaptive block selection. The compressed attention scores in Eq. (17) provide a low-cost proxy for identifying which feature groups are most relevant to the current query. We compute:

$$
\mathbf { p } _ { d } ^ { \mathrm { s l c } } = \mathrm { S o f t m a x } \big ( \mathbf { q } _ { d } ^ { \intercal } \mathcal { K } _ { d } ^ { \mathrm { c m p } } \big ) ,\tag{18}
$$

rank all blocks by $\mathbf { p } _ { d } ^ { \mathrm { s l c } }$ , and retrieve the original full-resolution keys and values for the top-m blocks:

$$
\mathcal { T } _ { d } = \{ i \mid \mathrm { r a n k } ( \mathbf { p } _ { d } ^ { \mathrm { s l c } } [ i ] ) \leq m \} , \quad \mathcal { K } _ { d } ^ { \mathrm { s l c } } = \mathrm { C a t } \big [ \{ \mathbf { k } _ { i b + 1 : ( i + 1 ) b } \mid i \in \mathcal { T } _ { d } \} \big ] .\tag{19}
$$

This recovers fine-grained interactions for the most relevant feature groups with a fixed budget of mb keys per query.

Gated aggregation. The three branch outputs are merged via input-dependent gating scalars $g _ { d } ^ { c } \in [ 0 , 1 ]$ , produced by a small MLP with sigmoid activation applied to the query embedding:

$$
\tilde { X } _ { d } = \sum _ { c \in \{ \mathrm { w i n } , \mathrm { c m p } , \mathrm { s l c } \} } g _ { d } ^ { c } \cdot \mathrm { A t t n } ( \mathbf { q } _ { d } , \mathcal { K } _ { d } ^ { c } , \mathcal { V } _ { d } ^ { c } ) .\tag{20}
$$

The total attended context per feature is $N = 2 s + \lfloor ( D - b ) / r \rfloor + m b \ll D$ , giving overall complexity $\mathcal { O } ( D d _ { \mathrm { t o k } } N )$ sub-quadratic in D for fixed sparse hyperparameters $( s , b , r , m )$

## J Feature-Token Mixing (FTM): Full Formulation

Adapting the dual-axis mixing block introduced in TabMixer [9], the FTM operator $\mathcal { M } ( \cdot )$ acts on the residual sparse representation $U = w _ { 1 } X _ { \mathrm { t o k } } + \tilde { X } \in \mathbb { R } ^ { D \times d _ { \mathrm { t o k } } }$ , where D feature tokens each carry a $d _ { \mathrm { t o k } }$ -dimensional embedding.

The operator applies two parallel mixing branches whose outputs are fused by element-wise gating:

$$
U _ { 1 } = \mathrm { G e L U } \big ( W _ { 2 } \big [ \mathrm { R e L U } ( W _ { 1 } \mathrm { L N } ( U ) ^ { \top } + b _ { 1 } ) \big ] + b _ { 2 } \big ) ^ { \top }\tag{21}
$$

$$
U _ { 2 } = W _ { 4 } \big ( \mathrm { R e L U } ( W _ { 3 } \cdot \mathrm { L N } ( U ) + b _ { 3 } ) \big ) + b _ { 4 }\tag{22}
$$

$$
U _ { 3 } = \mathrm { S i L U } ( \mathrm { L N } ( U _ { 1 } \odot U _ { 2 } ) )\tag{23}
$$

$$
Z = U + U _ { 3 } ,\tag{24}
$$

where W<sub>1</sub>, $W _ { 2 } \in \mathbb { R } ^ { d _ { \mathrm { t o k } } \times d _ { \mathrm { t o k } } }$ and $W _ { 3 } , W _ { 4 } \in \mathbb { R } ^ { B _ { f } \times B _ { f } }$ are learnable weights, with $B _ { f }$ a constant feature mini-batch size $( B _ { f } \ll D ) ; \mathrm { L N } ( \cdot )$ denotes Layer Normalization; and $\odot$ is element-wise multiplication.

The two branches serve distinct mixing roles. The embedding-wise branch (Eq. (21)) transposes the input so that the MLP operates across the $d _ { \mathrm { t o k } }$ embedding dimensions for each feature independently, capturing intra-token structure. Thefeature-wise branch (Eq. (22)) mixes along the token axis. To avoid a quadratic cost in D, we process the $D$ feature tokens in disjoint mini-batches of size $B _ { f }$ (padding the last batch if needed). The weight matrices $W _ { 3 } , W _ { 4 }$ are applied per batch, leading to a total complexity of $\bar { \mathcal { O } } ( D \cdot \bar { B } _ { f } )$ for this branch—linear in $D .$ . In our experiments, $B _ { f } = 6 4$ and the number of feature tokens $D$ ranges from 128 to 512, so the feature-wise mixing overhead remains small.

The outputs of the two branches are combined via element-wise multiplication, followed by Layer Normalization and a SiLU activation (Eq. (23)). A residual connection adds the result back to the original representation (Eq. (24)), ensuring stable gradient flow while enriching the representation with globally mixed context.

## K Additional Ablation Studies and Sensitivity Analyses

## K.1 Ablation of the FTM Pathway

We ablate the FTM pathway by removing it while keeping the sparse attention backbone and training protocol fixed. Table 8 shows that ablating FTM degrades RMSE on CP, TO, SA, and EVP, with particularly large drops on EVP (570.09 vs. 1.199). This indicates that the adapted FTM pathway contributes to the overall model performance, while sparse selection remains the primary architectural performance driver.

## K.2 Ablation on Loss Function

We analyze the effect of different regression losses to justify the proposed GridLoss and its use in a blended objective. We compare standard pointwise losses, including MSE, MAE, and Huber, a logarithmic robust variant, LogCosh, and the proposed GridLoss across three representative datasets with different scales and noise characteristics. Figure 9 reports RMSE on a log scale, with relative changes annotated with respect to GridLoss.

Across the evaluated benchmarks, GridLoss achieves competitive or superior RMSE relative to standard regression losses. Compared with MAE and LogCosh, GridLoss is less sensitive to scale variation and large residuals, while remaining competitive with Huber loss. The gains are most pronounced on datasets with heterogeneous error distributions, where purely pointwise losses may not adequately capture ordinal alignment along the target scale.

These results motivate the use of a blended objective in the full model,

$$
\begin{array} { r } { \mathcal { L } = \alpha \mathcal { L } _ { \mathrm { H u b e r } } + ( 1 - \alpha ) \mathcal { L } _ { \mathrm { G r i d } } , } \end{array}
$$

which combines the local robustness of Huber loss with the structure-aware ordinal supervision of GridLoss. Empirically, this combination yields stable optimization and strong predictive performance, particularly in high-dimensional regression settings. We fix α across all experiments and observe limited sensitivity to this choice; additional analysis is provided in Appendix B.4.

## K.3 Ablation on effect of RISE

We evaluate the impact of RISE by disabling it and reverting to uniform sampling. As shown in Figure 10, removing RISE increases both the mean validation error and the run-to-run variance. When RISE is enabled, validation performance is more consistent across runs, indicating improved optimization stability. Although RISE introduces additional sampling overhead, this cost is modest relative to the observed gains in validation performance. The box plot summarizes the distribution of final validation metrics across multiple random seeds, showing that RISE achieves a lower median error and reduced variability compared to uniform sampling.

## K.4 Sensitivity to Feature Dimensionality and Sample Size

We evaluate robustness to changing feature dimensionality and training-set size. For feature-dimensionality analysis, we vary the featurefraction, defined as the proportion of retained input features for each dataset, while keeping the number of rows fixed. For each fraction, we subsample feature columns using a fixed random seed and keep all other training settings identical to the main experimental protocol. For sample-size analysis, we vary the fraction of training samples while keeping the full feature set fixed.

Figure 11 reports MSE for TabNSM and baseline models on the TO and SC datasets. In the feature-fraction setting, TabNSM maintains stable performance across increasing dimensionality, whereas the baseline exhibits higher error and more pronounced non-monotonic behavior. In the sample-fraction setting, TabNSM remains competitive under reduced data availability. These trends suggest that TabNSM is robust to both high-dimensional feature spaces and limited training data. GridLoss and RISE further contribute to stability by providing structure-aware gradients and emphasizing difficult examples during training.

## K.5 Sensitivity to GridLoss Hyperparameters

We analyze the sensitivity of GridLoss to the temperature parameter τ and the blending coefficient α in the combined objective:

$$
\begin{array} { r } { \mathcal { L } = \alpha \mathcal { L } _ { \mathrm { H u b e r } } + ( 1 - \alpha ) \mathcal { L } _ { \mathrm { G r i d } } . } \end{array}
$$

Figure 12 reports RMSE as a function of $\tau$ across several datasets (CP, TO, and SCAN-2, abbreviated as SC). Performance varies smoothly over multiple orders of magnitude, indicating that GridLoss is not overly sensitive to the

choice of τ within a broad range. In the main experiments, we use a fixed default setting for both τ and α, while tuning the number of grid bins during training based on dataset-specific target statistics. The effects of varying τ and α are analyzed separately below.

## K.6 Sensitivity of τ to Target Distribution Properties

Table 9 summarizes the relationship between target distribution characteristics and the observed sensitivity of RMSE to the regularization parameter τ. We find that the optimal choice of τ is primarily governed by the concentration of the bulk of the target distribution, as quantified by the ratio IQR/Std, rather than by tail heaviness alone.

For CP, the target distribution exhibits a moderately dispersed bulk with heavy tails. Increasing τ consistently improves robustness, resulting in a monotonic decrease in RMSE. In contrast, TO has a tightly concentrated bulk with rare but extreme outliers, favoring small τ values that preserve precision on the dominant mass while avoiding excessive smoothing. Finally, SCAN-2 is characterized by a degenerate bulk (IQR = 0) and extremely rare but large outliers, leading to a highly non-monotonic RMSE–τ relationship in which very small τ values perform best.

These observations suggest that τ should be adapted to the effective bulk structure of the target distribution. Relying solely on tail statistics such as skewness or kurtosis is insufficient for selecting an appropriate regularization strength.

## K.7 Effect of the Blending Coefficient α

We next study the impact of the blending coefficient α, which controls the trade-off between the Huber loss and the grid-based loss. As shown in Figure 13, relying on either loss alone $( \alpha = 0 \mathrm { o r } \alpha = 1 )$ leads to degraded performance In contrast, intermediate values of α consistently achieve lower normalized RMSE across all datasets, with optimal performance observed for $\alpha \in [ 0 . 4 , 0 . 6 ]$ . These results indicate that the two loss terms are complementary and that balancing robustness with structural regularization is critical for optimal performance.

## K.8 Component-wise Ablation of Sparse Attention Parameters

We conduct a component-wise ablation study to assess the contribution of each sparse attention hyperparameter to model performance (Figure 14). In each experiment, a single hyperparameter is varied while all others are held fixed, and performance is evaluated using average validation RMSE across four validation sets. Lower RMSE indicates better performance. For the compress block size, a value of 10 achieves the best average validation RMSE, with performance degrading for both smaller and larger values. The number ofselected blocks performs best at 3, while deviations in either direction reduce effectiveness. Similarly, a selection block size of 10 yields the lowest validation RMSE among the tested values. Finally, a sliding window size of 6 provides the best performance, suggesting the importance of selecting an appropriate local context size. Based on these findings, we restrict the hyperparameter search space in the first-stage optimization to improve search efficiency. Nevertheless, joint hyperparameter optimization remains necessary to achieve optimal performance across datasets.

Table 2: Evaluation of different models on regression tasks. Results report RMSE (lower is better).
<table><tr><td>Model</td><td></td><td>SA</td><td>CP</td><td>AQ</td><td>TO</td><td>SC</td><td>CD</td><td>ChD</td><td>EVP</td><td>RES</td></tr><tr><td rowspan="10">Cassical</td><td>LinearRegression</td><td>0.402</td><td>9.277</td><td>15.062</td><td>0.028</td><td>3.997</td><td>0.236</td><td>8717.544</td><td>35.156</td><td>3163</td></tr><tr><td>KNN-5</td><td>0.238</td><td>3.719</td><td>12.816</td><td>0.029</td><td>3.602</td><td>1.347</td><td>15860.523</td><td>26.104</td><td>3212</td></tr><tr><td>Lasso</td><td>0.402</td><td>9.277</td><td>15.064</td><td>0.028</td><td>3.964</td><td>0.251</td><td>8717.439</td><td>35.155</td><td>3163</td></tr><tr><td>Ridge</td><td>0.402</td><td>9.277</td><td>15.062</td><td>0.027</td><td>3.982</td><td>0.236</td><td>8717.479</td><td>35.156</td><td>3163</td></tr><tr><td>SVR-Linear</td><td>0.406</td><td>11.663</td><td>17.854</td><td>0.042</td><td>3.974</td><td>,†</td><td>,t</td><td>,t</td><td>,†</td></tr><tr><td>SVR-RBF</td><td>0.175</td><td>7.491</td><td>19.416</td><td>0.038</td><td>3.964</td><td>*</td><td>*</td><td>*</td><td>*</td></tr><tr><td>Random Forest</td><td>0.201</td><td>2.842</td><td>9.095</td><td>0.027</td><td>4.092</td><td>0.733</td><td>17525</td><td>†</td><td>†</td></tr><tr><td>CatBoost</td><td>0.171</td><td>2.739</td><td>7.692</td><td>0.027</td><td>3.971</td><td>0.009</td><td>19000</td><td>3.291</td><td>3144</td></tr><tr><td>LightGBM</td><td>0.174</td><td>2.680</td><td>7.902</td><td>0.027</td><td>4.639</td><td>0.043</td><td>18688</td><td>2.799</td><td>3135</td></tr><tr><td>XGBoost</td><td>0.197</td><td>2.690</td><td>11.583</td><td>0.028</td><td>4.116</td><td>0.040</td><td>18188</td><td>2.894</td><td>3186</td></tr><tr><td>Extratrees</td><td>0.182</td><td>2.617</td><td>5.869</td><td>0.027</td><td>3.455</td><td>0.004</td><td>22882</td><td>2.809</td><td>46267</td></tr><tr><td>MLP ML-sye</td><td></td><td>2.939</td><td>8.967</td><td>0.028</td><td>3.920</td><td>0.046</td><td>4963</td><td>1.847</td><td>3161</td></tr><tr><td>ResNet</td><td>0.181 0.167</td><td>2.920</td><td>4.578</td><td>0.028</td><td>3.948</td><td>0.028</td><td>13022</td><td>1.556</td><td>3163</td></tr><tr><td>MLP-PLR</td><td>0.192</td><td>2.925</td><td>5.296</td><td>0.028</td><td>3.966</td><td>0.143</td><td>18476</td><td>3.222</td><td>40947</td></tr><tr><td>TabM</td><td>0.269</td><td>3.139</td><td>8.862</td><td>0.028</td><td>3.966</td><td>0.095</td><td>16032</td><td>2.101</td><td>40941</td></tr><tr><td>T2G-Former</td><td>0.192</td><td>2.937</td><td>5.358</td><td>0.028</td><td>3.962</td><td>‡</td><td>1</td><td>1.867</td><td>21058</td></tr><tr><td rowspan="9">Traomr</td><td>TabTransformer</td><td>0.298</td><td>8.605</td><td>4.189</td><td>0.028</td><td>3.866</td><td>0.046</td><td>7188</td><td>2.338</td><td>3936</td></tr><tr><td>SAINT</td><td>0.156</td><td>2.901</td><td>‡</td><td>丰</td><td>‡</td><td>‡</td><td></td><td></td><td>書</td></tr><tr><td>FT-Transformer</td><td>0.192</td><td>2.924</td><td>7.330</td><td>0.029</td><td>3.967</td><td>0.014</td><td>8940</td><td>1.507</td><td>3163</td></tr><tr><td>AutoInt</td><td>0.213</td><td>3.050</td><td>6.969</td><td>0.028</td><td>3.919</td><td>0.007</td><td>4999</td><td>2.587</td><td>3163</td></tr><tr><td>DCN-v2</td><td>0.165</td><td>8.133</td><td>4.081</td><td>0.038</td><td>3.751</td><td></td><td>4475</td><td>1.992</td><td>6667</td></tr><tr><td>ExcelFormer</td><td>0.165</td><td>2.901</td><td>8.088</td><td>0.028</td><td>3.667</td><td>s++</td><td>十</td><td>1.608</td><td>1</td></tr><tr><td>TabR</td><td>0.150</td><td>2.867</td><td>5.062</td><td>0.029</td><td>3.967</td><td></td><td></td><td></td><td></td></tr><tr><td>ModernNCA</td><td>0.167</td><td>2.833</td><td>12.186</td><td>0.028</td><td>3.730</td><td>十十</td><td>十</td><td></td><td>十十</td></tr><tr><td>NODE</td><td>0.270</td><td>7.150</td><td>15.575</td><td>0.028</td><td>3.966</td><td>0.052</td><td>22661</td><td>5.157</td><td>3187</td></tr><tr><td rowspan="9">Retival</td><td>TabNet</td><td>0.279</td><td>3.346</td><td>8.615</td><td>0.028</td><td>3.454</td><td>0.284</td><td>24271</td><td>2.653</td><td>4152</td></tr><tr><td>SNN</td><td>2.453</td><td>14.809</td><td>17.833</td><td>0.028</td><td>3.932</td><td>5.276</td><td>18759</td><td>71.000</td><td>3163</td></tr><tr><td>TANGOS</td><td>0.181</td><td>2.991</td><td>4.690</td><td>0.028</td><td>3.953</td><td>0.136</td><td>7298</td><td>3.219</td><td>4094</td></tr><tr><td>PTaRL</td><td>0.251</td><td>3.183</td><td>5.201</td><td>0.028</td><td>3.570</td><td>0.196</td><td>7865</td><td>4.157</td><td>4082</td></tr><tr><td>DANets</td><td>0.289</td><td>3.256</td><td>5.148</td><td>0.028</td><td>3.970</td><td>0.606</td><td>4808</td><td>9.367</td><td>41321</td></tr><tr><td>GrowNet</td><td>2.141</td><td>16.855</td><td>20.089</td><td>0.028</td><td>3.982</td><td>sst</td><td>25264</td><td>13.448</td><td>124527</td></tr><tr><td>DNNR</td><td>0.149</td><td>3.065</td><td>§</td><td>§</td><td>§</td><td></td><td>†</td><td>†</td><td>†</td></tr><tr><td>SwitchTab</td><td>0.457</td><td>3.884</td><td>6.384</td><td>0.029</td><td>3.969</td><td>0.285</td><td>17978</td><td>6.227</td><td>4083</td></tr><tr><td>GRANDE</td><td>0.347</td><td>11.716</td><td>23.815</td><td>0.028</td><td>3.964</td><td>4.353</td><td>25440</td><td>69.129</td><td>25620</td></tr><tr><td></td><td>AMFormer</td><td>0.205</td><td>2.980</td><td>16.363</td><td>0.028</td><td>3.964</td><td>++</td><td>17858</td><td></td><td>21149 ◇</td></tr><tr><td>Founatton</td><td>TabEBM</td><td>0.178</td><td>2.740</td><td>6.748</td><td>0.027</td><td>3.522 3.965</td><td></td><td>◇</td><td>++◇ *</td><td></td></tr><tr><td></td><td>TabPFN-v2</td><td>0.168</td><td>3.006</td><td>10.609</td><td>0.027</td><td></td><td>*</td><td>*</td><td></td><td>*</td></tr><tr><td></td><td>TabPTM</td><td>0.579</td><td>4.015</td><td>18.627</td><td>0.029</td><td>3.827</td><td>5.253</td><td>25630</td><td>65.471</td><td>4171</td></tr><tr><td>Ours</td><td></td><td>0.137</td><td>2.842</td><td>3.999</td><td>0.021</td><td>3.322</td><td>0.007</td><td>4460</td><td>1.199</td><td>3103</td></tr></table>

† Not reported due to time constraints (24-hour limit); runs exceeding this duration were terminated.  
‡ Not reported due to GPU memory limitations (GPU capacity: 79.21 GiB); models exceeding this requirement could not be executed.  
<sup>§</sup> Training diverged; the model produced losses orders of magnitude outside the comparable range and is treated as non-convergent. ∗ Applicable only to limited numbers of features and samples (hard-coded).  
♢ Infeasible at dataset scale (algorithm complexity prohibitive).

Pairwise Wilcoxon Signed-Rank Test: TabNSM vs. each baseline (one-sided, H : TabNSM rank < baseline rank; BH-corrected at =0.05) Missing values worst rank imputed

![](images/911ad543b4798e67dfd3a326eeeb3f284e9eb2445779dae80d2d62d06b1a2139.jpg)  
Figure 4: Average rank across nine benchmarks for each model (lower is better), coloured by architectural family. All baselines are statistically significantly worse than TabNSM (one-sided Wilcoxon signed-rank test, Benjamini–Hochberg corrected [70], $\alpha = 0 . 0 5 ;$ all corrected $p \leq 0 . 0 2 )$ . Missing values are imputed as worst rank.

![](images/06bbdff1278fd0245c9567e9cd6c1c2dac0035486f847f2d002d9666e0e9ff7f.jpg)  
Figure 5: Per-instance feature attribution heatmap for TabNSM. Each row corresponds to a test instance and each column corresponds to an input feature. Cell intensity represents the normalized attribution score, computed using gradient–embedding attribution and scaled to [0, 1] within each instance. Brighter cells indicate features with stronger influence on the prediction. The sparse and instance-varying attribution patterns suggest that TabNSM adaptively focuses on different feature subsets across samples.

Table 3: Dataset Statistics
<table><tr><td>Dataset</td><td>Abbreviation</td><td># Samples</td><td># Features</td></tr><tr><td>Sarcos</td><td>SA</td><td>48,933</td><td>27</td></tr><tr><td>CPU</td><td>CP</td><td>8,192</td><td>12</td></tr><tr><td>Topo</td><td>TO</td><td>8,885</td><td>266</td></tr><tr><td>SCAN</td><td>SC</td><td>4,355</td><td>386</td></tr><tr><td>Crime Data</td><td>CD</td><td>1,004,991</td><td>401</td></tr><tr><td>U.S. Chronic Disease</td><td>ChD</td><td>309,215</td><td>405</td></tr><tr><td>Electric Vehicle Population Data</td><td>EVP</td><td>264,628</td><td>392</td></tr><tr><td>Air Quality</td><td>AQ</td><td>18,862</td><td>390</td></tr><tr><td>Real Estate Sales</td><td>RES</td><td>1,048,575</td><td>389</td></tr></table>

Table 4: Benchmark dataset sources.
<table><tr><td>Dataset</td><td>URL</td></tr><tr><td>CPU</td><td>https://www.openml.org/search?type= data&amp;status=active&amp;id=42841</td></tr><tr><td>Topo</td><td>https://www.openml.org/search?type= data&amp;status=active&amp;id=422</td></tr><tr><td>Sarcos Robot Arm Data</td><td>https://www.openml.org/search?type= data&amp;status=active&amp;id=43873&amp;sort=runs</td></tr><tr><td>Electric Vehicle Population Data</td><td>https://catalog.data.gov/dataset/ electric-vehicle-population-data</td></tr><tr><td>U.S. Chronic Disease Data</td><td>https://catalog.data.gov/dataset/ u-s-chronic-disease-indicators</td></tr><tr><td>Air Quality</td><td>https: //catalog.data.gov/dataset/air-quality</td></tr><tr><td>Crime Data from 2020 to Present</td><td>https://catalog.data.gov/dataset/ crime-data-from-2020-to-present</td></tr><tr><td>Real Estate Sales Data</td><td>https://catalog.data.gov/dataset/ real-estate-sales-2001-2018</td></tr><tr><td>Standardized Centralized Alzheimer&#x27;s Neuroimaging (SCAN) Data</td><td>https://scan.naccdata.org/</td></tr></table>

Note: All datasets were accessed by August 2025; NACC/SCAN data require an approved request and Data Use Agreement.

![](images/6620339cb6d07fc6d7f8c34c36711b2c5f53fa05567355e44b7d9d42ffa4162b.jpg)  
Figure 6: Sensitivity analysis with respect to random seed initialization. The boxplot shows the distribution of test Root Mean Squared Error (RMSE) over 20 independent runs with different random seeds. Boxes indicate the interquartile range (IQR), the center line denotes the median, and whiskers extend to $1 . 5 \times \mathrm { I Q R }$ . Lower values indicate better performance.

Table 5: Runtime in seconds for CatBoost and TabNSM under the validation-loss threshold protocol. Lower is better. CatBoost is evaluated on CPU, whereas TabNSM uses GPU acceleration.
<table><tr><td>Dataset</td><td>n</td><td>d</td><td>CatBoost Time (s)</td><td>TabNSM Time (s)</td></tr><tr><td>cpu</td><td>8,192</td><td>12</td><td>1.042</td><td>0.346</td></tr><tr><td>topo</td><td>8,885</td><td>266</td><td>10.788</td><td>0.333</td></tr><tr><td>sarcos</td><td>48,933</td><td>27</td><td>2.111</td><td>0.333</td></tr><tr><td>EVP</td><td>264,628</td><td>392</td><td>14.986</td><td>0.346</td></tr><tr><td>ChD</td><td>309,215</td><td>405</td><td>17.252</td><td>0.349</td></tr><tr><td>CD</td><td>1,004,991</td><td>401</td><td>45.476</td><td>0.398</td></tr></table>

Table 6: Peak GPU memory usage in MB for TabNSM. CatBoost is run on CPU, so GPU memory is not applicable.
<table><tr><td>Dataset</td><td>CatBoost</td><td>TabNSM</td></tr><tr><td>EVP</td><td>N/A</td><td>477</td></tr><tr><td>ChD</td><td>N/A</td><td>529</td></tr><tr><td>CD</td><td>N/A</td><td>1621</td></tr></table>

Table 7: CPU memory increase in MB during training. Lower is better.
<table><tr><td>Dataset</td><td>CatBoost</td><td>TabNSM</td><td></td></tr><tr><td>EVP</td><td>3763.77</td><td>3006.73</td><td rowspan="4"></td></tr><tr><td>ChD</td><td>4325.94</td><td>3408.70</td></tr><tr><td>CD</td><td>12080.61</td><td>8524.16</td></tr></table>

These large-scale settings are not evaluated with TabPFN because of its configuration limits on the number of samples and features.

Table 8: Ablation of the FTM pathway. Lower RMSE is better.
<table><tr><td></td><td>CP</td><td>TO</td><td>SA</td><td>EVP</td></tr><tr><td>w/o FTM</td><td>3.966</td><td>0.034</td><td>0.176</td><td>570.09</td></tr><tr><td>Full TabNSM</td><td>2.842</td><td>0.021</td><td>0.137</td><td>1.199</td></tr></table>

![](images/e237b92d99df9fa8e769d7bbc70dc58dac025d4ea29ac09270e87f1b7e4a2e54.jpg)  
Figure 7: Training time versus sample size. Runtime is measured under the validation-loss threshold protocol. CatBoost runtime increases with dataset size, whereas TabNSM maintains low runtime across the evaluated settings, reflecting mini-batch GPU training and sparse feature interaction modeling.

![](images/473f84e23b5e241165f6969ae8ad3c71c8288e98ab8300a09123253a4e146d13.jpg)  
Figure 8: Accuracy–efficiency trade-off for tabular learning methods. The figure summarizes predictive accuracy and computational efficiency across representative tabular learning models.

![](images/b073b1646f87522a90c45edc72c201da3247adc1cbd6a000cd8e77510b142944.jpg)  
Figure 9: Loss-function ablation across three datasets. RMSE is reported on a log scale. Values above bars indicate relative change with respect to GridLoss. GridLoss achieves the lowest or comparable RMSE across the evaluated datasets, while pointwise losses such as MAE and LogCosh can degrade substantially in high-error regimes.

![](images/85f42b26710fbbe01bdeced52206690b6f76758b0de885ebef01d98a15f5f187.jpg)  
Figure 10: Final validation metric distributions across multiple runs for models trained with and without RISE.

Table 9: Target distribution statistics and their relationship to τ sensitivity.
<table><tr><td>Dataset</td><td>Std</td><td>IQR</td><td>IQR/Std</td><td>Tail Type</td><td>Preferred τ</td><td>RMSE-τ Behavior</td></tr><tr><td>CP</td><td>18.40</td><td>13.00</td><td>0.71</td><td>Heavy-tailed</td><td>Large</td><td>Monotonic decrease</td></tr><tr><td>TO</td><td>0.03</td><td>0.02</td><td>0.67</td><td>Rare extremes</td><td>Small</td><td>Sharp peak</td></tr><tr><td>SCAN-2</td><td>4.99</td><td>0.00</td><td>0.00</td><td>Degenerate bulk</td><td>Very small</td><td>Highly non-monotonic</td></tr></table>

![](images/97db81591db507975f81fb7d3250ab40e15c90c97c820be3d2028a5218aa7721.jpg)

SC Dataset  
![](images/bf57ba761a6f2aab438cc07ea0ca7f09c31a87d23bc9bf6aa919097bddd39bba.jpg)

(a) MSE versus feature fraction on the TO and SC datasets.  
EVP Dataset  
![](images/6e8a0f16ecab5c20c0b7b5ef928ce303d036b6671a8ba76f9d8e9b6683b1c289.jpg)

SC Dataset  
![](images/c7dfc52b32d5dd8c66bba8e32ff39505eeff479bb426116ef98e4c8d7502b05d.jpg)  
(b) MSE versus training sample fraction on the TO and SC datasets.  
Figure 11: Sensitivity to feature dimensionality and sample size. The top panel evaluates robustness under varying numbers of retained input features, while the bottom panel evaluates robustness under varying fractions of training samples. Separate y-axes are used to accommodate differences in error scale.

![](images/575cee5e90821c17e21d39cd6f9cd24367645d3619e3ea9475eece3810187455.jpg)  
Figure 12: Normalized RMSE as a function of τ across datasets.

![](images/ff015bc5f331e026f0a64f2f1b04136da94ea5b03c2199edb90d1a9b0e762eef.jpg)  
Figure 13: Normalized RMSE as a function of α across datasets.

![](images/978eed6ee43d1af6f7881b9a101fc524046deb1e477482c1cccc9554793c82af.jpg)

![](images/c36ba7b7b24d60a55b42d105129e06709d6a9f18ce2e0368b8f74333da8e886b.jpg)

![](images/f375ae7f5a5cb81960d81b0d52b783416f04ca8473b25addfc8ebd0f61aca24b.jpg)

![](images/5e1f232827eaeab0eb6095e97056c41f0565b24bb08fd6b3cec1fc4d4d57a3f7.jpg)  
Figure 14: Component-wise ablation of sparse attention parameters. Each plot shows the effect of varying one parameter: number\_of\_selected\_blocks (top-left), selection\_block\_size (top-right), compress\_block\_size (bottom-left), and sliding\_window\_size (bottom-right) on the average validation RMSE across four datasets.