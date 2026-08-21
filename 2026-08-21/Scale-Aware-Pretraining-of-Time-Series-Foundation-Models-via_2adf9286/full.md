# Scale-Aware Pretraining of Time Series Foundation Models via Multi-Patch Token Alignment and Hybrid Masking

Taihua Chen<sup>1,2</sup>, Xiang Ma<sup>1</sup>, Yixin Zhang<sup>3</sup>, Tailin Zhan<sup>1</sup>,

Manyu Sun<sup>1</sup>, and Lizhen Cui<sup>1,2,\*</sup>

<sup>1</sup>School of Software, Shandong University, Jinan 250101, China

<sup>2</sup>Joint SDU–NTU Centre for Artificial Intelligence Research (C-FAIR),

Shandong University, Jinan 250101, China

<sup>3</sup>Nanyang Technological University, Singapore

cfair-cth@mail.sdu.edu.cn, xiangma@sdu.edu.cn, zhangyixin9610@gmail.com, tailinzhan@mail.sdu.edu.cn, sjxyctcsusud6383@gmail.com, clz@sdu.edu.cn

Abstract—Pretraining time series foundation models across heterogeneous datasets necessitates effective handling of varying sampling frequencies. Current methods either employ datasetspecific patch sizes and separate FFNs, leading to fragmented representations, or enforce a fixed patch size that neglects inherent temporal variations. To address this, we propose SATS, featuring a scale-aware token alignment mechanism that treats patch size as an explicit notion of scale. By incorporating a contrastive-inspired alignment regularizer, SATS aligns representation spaces across scales while preserving distinct modeling capacities. Furthermore, a hybrid masking strategy combining random and contiguous masking is introduced to capture multi-scale temporal structures. Experimental results on LSTF benchmarks demonstrate that SATS achieves a 9.2% improvement in MSE and an 8.3% gain in GIFT-Eval MASE compared to competitive baselines. Notably, SATS consistently delivers SOTA performance while achieving a 65.6% increase in model efficiency over advanced baselines, highlighting its effectiveness and scalability in time series pretraining.

Index Terms—Time series foundation models, Time series forecasting, Heterogeneous time series, Zero-shot forecasting.

## I. INTRODUCTION

The recent emergence of foundation models has significantly advanced various domains such as natural language processing [1], [2], computer vision [3], [4], and speech understanding [5], [6]. Inspired by their success, growing efforts have been devoted to developing foundation models for time series, aiming to produce general-purpose representations transferable across diverse downstream tasks. An early line of work adapts pretrained language models to time series tasks, leveraging their sequence modeling capabilities in hopes of achieving strong generalization [7]–[9]. However, the modality gap often hinders their performance on temporally structured data, resulting in suboptimal generalization across diverse time series tasks. Moreover, their black-box nature further exacerbates the issue, raising concerns about interpretability and the lack of alignment with intrinsic temporal characteristics [10]. To address these challenges, a second line of work has emerged that trains foundation models from scratch on large-scale, heterogeneous time series datasets [11]–[13]. These models aim to capture universal temporal dynamics in a data-driven and domain-adaptive manner, thereby enhancing robustness to distribution shifts and improving transferability across domains with varying sampling rates, modalities, and sequence lengths (e.g., finance, healthcare, meteorology, IoT).

Despite the promise of the latter direction, it presents unique challenges— particularly in how to effectively segment and tokenize continuous signals for cross-dataset pretraining. Unlike language, where discrete word units naturally serve as stable tokens [14], or vision, where uniform patch sizes are viable due to consistent spatial resolution and semantic robustness [15], [16], time series data exhibit irregular sampling and variable sequence lengths, making fixed-size downsampling ineffective. These characteristics necessitate the use of small, adaptive patch sizes to preserve fine-grained temporal patterns.

As shown in Figure 1, recent studies have explored two main strategies for time series tokenization, each with inherent limitations. (1) Dataset-specific patching adopts variable patch sizes tailored to local sampling rates, combined with independent FFNs for token projection [12], [17]. While this design aligns well with the granularity of each dataset, it results in fragmented token spaces that hinder the learning of generalizable temporal patterns and compromise training stability [18]. (2) Uniform patching enforces a globally small patch size across datasets to promote representational consistency [19], [20]. However, this strategy introduces information bottlenecks and often misaligns local dynamics, as it fails to accommodate the diverse temporal structures inherent in different datasets [21]. Both strategies, therefore, face a tradeoff between dataset adaptability and representational generality, limiting their effectiveness in scalable pretraining.

To bridge the gap between fragmented token spaces introduced by adaptive patching and the representational rigidity of fixed segmentation, we propose a scale-aware token alignment mechanism tailored for time series pretraining. By treating the patch size as an explicit notion of scale, our method aligns the representation spaces induced by scale-specific FFNs. This is achieved by minimizing the distance between mean token embeddings across scales to encourage semantic alignment, while simultaneously maximizing the distance between their maximal embeddings to preserve the scalespecific modeling capacity. The resulting token space offers a unified yet expressive foundation for downstream tasks.

![](images/c2f84215a602a1af47dacb18e51575ad865042aa13d3c8b151a109306503953b.jpg)  
Fig. 1. (a) Dataset-specific patch sizes and independent FFNs for varying sampling rates lead to fragmented token spaces. (b) Using a unified patch size and FFN risks information bottlenecks and misaligned local dynamics. (c) SATS adopts dataset-specific patch sizes and enforces scale-aware alignment across FFN-projected spaces, yielding semantically rich and consistent representations.

Building on this aligned representation space, a remaining challenge lies in the diverse temporal structures inherent to different datasets. Even with aligned embeddings, temporal variations may manifest within individual tokens or span across multiple tokens, depending on the dynamics of the underlying sequence. To capture such variability, we introduce a hybrid masking strategy that enhances multi-scale temporal modeling during masked reconstruction. This strategy combines random masking, which promotes fine-grained inference, with contiguous masking, which facilitates the modeling of long-range dependencies. By jointly optimizing across these complementary patterns, the model learns to recover temporal structures at varying resolutions, improving its robustness and generalization.

Our main contributions are summarized as follows:

• We propose SATS, a Scale-Aware foundation model for Time Series, which achieves superior generalization across heterogeneous time series datasets.

• We introduce a scale-aware alignment mechanism based on scale-specific FFNs, unifying token spaces across patch scales while preserving scale-specific expressiveness.

• We design a hybrid masking strategy that enables the model to capture both fine-grained and long-range temporal dependencies across multiple resolutions.

• Extensive experiments demonstrate the effectiveness of SATS in both zero-shot and in-distribution forecasting settings, establishing its potential as a strong pretraining paradigm for time series foundation models.

## II. RELATED WORK

## A. Time Series Foundation Models

Traditional forecasting models are typically optimized for specific datasets, horizons, and sampling resolutions, so their scale handling is often encoded through architectural or hyperparameter choices. Multi-scale models such as Pathformer [22] and TimeMixer [23] strengthen this line by modeling temporal patterns at multiple resolutions through scale-specific pathways, decomposition, pooling, patching, or dilation. While these methods capture multi-resolution patterns within individual datasets, foundation-model pretraining mixes datasets with different sampling frequencies and patch scales, making crossfrequency representation compatibility necessary for transferable temporal representations. Time series foundation models have therefore gained increasing attention. To meet forecastingspecific demands, decoder-only models such as Timer [24] and Lag-Llama [25] adopt causal architectures, while sparse MoE variants such as Time-MoE [11] and Moirai-MoE [20] improve scalability. Encoder-decoder models such as LightGTS [19] and Chronos [13] use parallel decoding or discretized objectives. Encoder-only architectures remain less explored for time series foundation models, although recent experimental analyses [26] suggest stronger representational capacity under limited compute, motivating further investigation into their architecture and pretraining strategies [12], [27]. Following this encoder-only pretraining direction, SATS addresses the overlooked need to align scale-specific token spaces induced by heterogeneous patch sizes.

## B. Contrastive Learning in Pretraining

Contrastive learning has emerged as a powerful paradigm in large-scale pretraining across various domains. In NLP, methods such as SimCSE [28] leverage contrastive objectives to learn semantically meaningful sentence embeddings without supervision. In computer vision, CLIP [4] and ALIGN [29] jointly embed images and texts by maximizing the similarity of paired modalities while contrasting unpaired ones, achieving impressive zero-shot performance. While contrastive learning in time series remains relatively underexplored, recent works like TS-TCC [30] and CoST [31] demonstrate its potential in learning transferable representations by aligning augmented views of temporal data. These methods generally seek a balance between aligning semantically related representations and maintaining sufficient dispersion in the embedding space. Such an attraction–repulsion structure not only improves feature discriminability but also helps prevent representations from collapsing into an uninformative configuration. A key advantage of contrastive learning lies in its ability to preserve embedding diversity—by pulling semantically similar instances closer and pushing dissimilar ones apart, it structures the latent space in a discriminative and robust manner. Inspired by contrastive learning’s structured divergence, we adopt an InfoNCE-motivated objective to enhance distinctiveness among multi-scale features—without explicit negative samples—thus inheriting its regularization benefits.

![](images/12460da88873988e8feb12032182405d93f8c83904cfc70bc82ce1d3a6d7b50c.jpg)  
Fig. 2. Overview of the SATS framework. Tokens from multiple patch sizes are projected via separate FFNs. SATS employs Scale-aware Alignment mechanism to promote proximity of mean-pooled representations within each scale, while enforcing separation of max-pooled representations across scales—balancing consistency and scale-specific expressiveness. Hybrid masking strategy, integrating Random Masking and Continuous Masking, is further applied to capture both fine-grained and long-range temporal dependencies.

## III. METHODOLOGY

## A. Problem Formulation

Let $\boldsymbol { S } = \{ ( \mathbf { X } ^ { ( i ) } , \mathbf { C } ^ { ( i ) } ) \} _ { i = 1 } ^ { N }$ denote a dataset of multivariate time series, where $\mathbf { X } ^ { ( i ) } \in \mathbb { R } ^ { d _ { x } \times T _ { i } }$ are target sequences and $\mathbf { C } ^ { ( i ) } \in \mathbb { R } ^ { d _ { c } \times T _ { i } }$ are associated covariates. Given the unmasked observations $\mathbf { X } _ { \mathrm { o b s } }$ and the corresponding covariates C, the objective is to learn model parameters θ such that the model $f _ { \theta }$ predicts the distribution parameters $\hat { \psi }$ for the masked subset $\mathbf { X } _ { \mathcal { M } }$ of the target sequence.

This leads to the following optimization problem:

$$
\begin{array} { r l } & { \underset { \theta } { \operatorname* { m i n } } \ : \mathbb { E } _ { ( \mathbf { X } , \mathbf { C } ) \sim p ( S ) } \mathbb { E } _ { \mathcal { M } \sim p ( \mathcal { T } | S ) } \left[ \mathcal { L } _ { \mathrm { n l l } } \left( \mathbf { X } _ { \mathcal { M } } , \boldsymbol { \hat { \psi } } \right) \right] } \\ & { ~ \mathrm { s . t . } ~ \boldsymbol { \hat { \psi } } = f _ { \theta } ( \mathbf { X } _ { \mathrm { o b s } } , \mathbf { C } ) } \end{array}\tag{1}
$$

Here, ${ \mathcal L } _ { \mathrm { n l l } }$ denotes the negative log-likelihood loss:

$$
\mathcal { L } _ { \mathrm { n l l } } ( \mathbf { X } _ { \mathcal { M } } , \boldsymbol { \hat { \psi } } ) = - \log p ( \mathbf { X } _ { \mathcal { M } } \mid \boldsymbol { \hat { \psi } } )\tag{2}
$$

where $p ( S )$ is the data-generating distribution over time series instances $( \mathbf { X } , \mathbf { C } )$ , and $p ( \mathcal T \ \mid S )$ defines the task sampling distribution that governs the selection of masked positions $\mathcal { M } \subset$ $\{ 1 , \ldots , T \}$ for prediction. Classical forecasting corresponds to the special case where the masked region M is located at the end of the sequence.

## B. Model Architecture

As shown in Figure 2, SATS adopts a non-overlapping patchbased, encoder-only Transformer [32]. The multivariate time series is first flattened and, following Moirai [12], mapped into patches of varying sizes based on the dataset. To improve efficiency, we adopt packing as a default setting [2], [33], enabling tokens with different patch sizes from multiple datasets to be packed into a single sequence. This multi-scale design introduces inconsistencies in the token space; while packing is not the direct cause, it is an indispensable component of modern scalable training, making it both practical and necessary to develop solutions within this paradigm.

To mitigate such inconsistencies while embracing the packing paradigm, SATS employs a scale-aware alignment mechanism: it pulls closer the mean-pooled representations within the same scale, while pushing apart the max-pooled ones across scales, ensuring consistency while preserving scale-specific expressiveness. Based on this aligned space, a hybrid masking strategy combining random and contiguous patterns is applied to capture both fine-grained and long-range dependencies.

Although not shown, the encoder incorporates key techniques from foundational model pretraining—such as RoPE [34], SwiGLU [35], and RMSNorm [36]—as well as inductive biases specific to time-series pretraining, including Any-Variate Bias, Mixture Distribution Output [12] and RevIN [37] for modeling inter-variable dependencies and normalization under distribution shifts.

1) Scale-aware Alignment: To enhance the effectiveness of universal temporal modeling, especially when dealing with subsequences of varying scales, it is crucial to design a robust embedding space alignment strategy. Given token sequences $\boldsymbol { \mathcal { I } } \in \mathbb { R } ^ { L \times \mathbf { \smile } }$ , where L represents the maximum input length during training and D is the hidden layer dimension of the encoder, the challenge arises from the coexistence of tokens originating from $n \leq N$ different patch sizes, where N denotes the total number of distinct patch sizes. A direct approach could be to minimize the feature space distance, such as cosine similarity, between subsequences, encouraging their proximity. However, this approach faces several challenges: first, the varying lengths of subsequences make it difficult to quantify alignment; second, different samples within the same batch may contain different numbers of subsequences, complicating the application of proximity constraints both within and across samples. Furthermore, a structured information constraint is necessary to discourage scale-wise anchor collapse and maintain diverse temporal representations.

In response to these challenges, we propose the Scaleaware Alignment (SA) method, which integrates two key components. First, we introduce a pooling mechanism to address the issues of variable subsequence lengths and differing numbers of subsequences across samples. Specifically, we pool tokens based on their patch sizes to generate the scalewise embedding representation $Y \in \mathbb { R } ^ { N \times \mathbf { \overline { { D } } } }$ . In cases where a patch size is absent in a given sample, the corresponding embedding position $Y _ { i }$ is set to zero $( i \leq N )$ to maintain fixed tensor dimensions for batching. Crucially, a binary validity mask is applied during the computation of the alignment loss to explicitly exclude these zeroed positions from the similarity summation, thereby preventing gradient propagation from missing patch sizes. Second, inspired by the principles of contrastive learning, we design a structured information constraint: the mean embeddings from different patch sizes are pulled closer to establish neighboring centers in the token space, while the maximal embeddings are repelled to encode scale-specific information, ensuring richer and more diverse token semantics. More theoretical analysis is provided in Section IV-A. To operationalize this constraint, we adopt the InfoNCE framework, as detailed in Equation 3 and Equation $^ { 4 , }$ where $\cos ( \cdot )$ denotes the cosine similarity function and $\tau$ is the temperature parameter.

$$
{ \mathcal { L } } _ { \mathrm { c l o s e } } = - \mathbb { E } \left[ \log { \frac { \sum _ { j \neq i } \exp { ( \cos ( Y _ { i } \cdot Y _ { j } ) / \tau ) } } { \sum _ { j = 1 } ^ { N } \exp { ( \cos ( Y _ { i } \cdot Y _ { j } ) / \tau ) } } } \right]\tag{3}
$$

$$
\mathcal { L } _ { \mathrm { f a r } } = - \mathbb { E } \left[ 1 - \log \left( \sum _ { j = 1 } ^ { N } \exp { ( \cos ( Y _ { i } \cdot Y _ { j } ) / \tau ) } \right) \right]\tag{4}
$$

In practice, $Y _ { i } ~ \in ~ Y ^ { \mathrm { m e a n } }$ is sequentially substituted into Equation 3, while $Y _ { i } \in Y ^ { \operatorname* { m a x } }$ is substituted into Equation $4 .$ Although both equations follow the InfoNCE form, they do not involve true negative samples. We therefore combine these two losses to form the final scale-aware alignment constraint in Equation 5. This design ensures that the loss function conforms to the geometric structure of contrastive learning, as detailed in Section IV-B. Consequently, it provides structured regularization that aligns feature representations across different patch sizes, enhances cross-scale consistency, and discourages scale-wise anchor collapse. The hyperparameter $\beta$ controls the relative weight of the maximal embedding pull-away term, balancing the overall objective.

$$
\mathcal { L } _ { \mathrm { s a } } = \mathcal { L } _ { \mathrm { c l o s e } } + \beta \mathcal { L } _ { \mathrm { f a r } }\tag{5}
$$

2) Hybrid Masking Strategy: On top of the aligned token space, the intrinsic heterogeneity and complexity of temporal dynamics across datasets continue to challenge effective representation learning. Although alignment mitigates certain variations, temporal dependencies inherently span multiple scales: some manifest as fine-grained, localized fluctuations within individual tokens, while others emerge as extended, structured patterns across contiguous token segments. To com prehensively capture these diverse temporal scales and improve the robustness of learned representations, we therefore propose a Hybrid Masking Strategy (HM) that synergistically combines Random Masking (RM) with Continuous Masking (CM), implemented as contiguous-span masking during pretraining.

Concretely, given each token subsequence $\bar { \mathcal { T } } _ { j } ^ { \mathsf { ^ { - } } } \in \mathbb { R } ^ { L _ { j } \times \bar { D } }$ extracted from the full sequence I, where $L _ { j }$ denotes the length of the j-th subsequence, a masking ratio $r \in [ 0 . 1 5 , 0 . 5 ]$ is applied. For each subsequence, a predefined probability $p \in$ [0, 1] determines whether RM or CM is used. With probability $p ,$

RM uniformly selects $m _ { j }$ token positions, where $m _ { j } = \lceil r \cdot L _ { j } \rceil$ and $\begin{array} { r } { \sum _ { i = 0 } ^ { L _ { j } - 1 } \dot { \mathcal { M } } _ { r } ^ { ( j ) } ( i ) = \dot { m } _ { j } } \end{array}$ , producing a binary mask $\mathcal { M } _ { r } ^ { ( j ) } \{$

$$
\mathcal { M } _ { r } ^ { ( j ) } ( i ) = \left\{ \begin{array} { l l } { 1 , } & { \mathrm { i f ~ t o k e n ~ } i \mathrm { ~ i s ~ r a n d o m l y ~ s e l e c t e d } } \\ { 0 , } & { \mathrm { o t h e r w i s e } } \end{array} \right. .\tag{6}
$$

Note that $m _ { j } ~ \leq ~ L _ { j }$ is inherently satisfied given $r ~ \leq ~ 0 . 5$ Alternatively, with probability $1 - p ,$ CM samples a start index $s _ { j } \in \{ 0 , \ldots , L _ { j } - m _ { j } \}$ , masking a continuous block of tokens:

$$
\mathcal { M } _ { c } ^ { ( j ) } ( i ) = \left\{ \begin{array} { l l } { 1 , } & { s _ { j } \le i < s _ { j } + m _ { j } } \\ { 0 , } & { \mathrm { o t h e r w i s e } . } \end{array} \right.\tag{7}
$$

The final mask $\mathcal { M } ^ { \left( j \right) }$ applied to each subsequence $\mathcal { T } _ { j }$ is sampled as

$$
\mathcal { M } ^ { ( j ) } = \left\{ \mathcal { M } _ { r } ^ { ( j ) } , \mathrm { \quad w i t h ~ p r o b a b i l i t y ~ } p \right.\tag{8}
$$

By guiding the model to recover masked tokens across both RM and CM patterns, HM balances fine-grained local inference and long-range dependency learning. Consequently, it enhances the robustness and generalizability of learned representations for diverse temporal modeling tasks.

## C. Model Training

1) Unified Learning Objective: The proposed alignment and masking strategies introduce no additional learnable parameters, enabling their integration into a unified learning objective without parameter overhead. In practice, the mask $\mathcal { M }$ obtained from Equation 8 is applied to Equation 2 to compute the primary training loss. Simultaneously, Equation 5 is employed as an auxiliary training loss to enforce SA. We combine them into the total loss function as follows:

$$
\mathcal { L } = \mathcal { L } _ { \mathrm { n l l } } + \alpha \mathcal { L } _ { \mathrm { s a } }\tag{9}
$$

where $\alpha$ is a weighting coefficient balancing the two objectives. 2) SATS Setup: We pretrain the SATS models on the LOTSA dataset [12] in two configurations—small and base—with detailed model specifications provided in Table I. The small model is trained for 100,000 steps with a batch size of 256, while the base model is trained for 200,000 steps with a batch size of 128. All experiments use the following fixed hyperparameters unless otherwise specified:

• Optimizer: AdamW with learning rate $1 \times 1 0 ^ { - 3 }$ , weight decay $1 \times 1 0 ^ { - 1 } , \beta _ { 1 } = 0 . 9 , \beta _ { 2 } = 0 . 9 8$

$\operatorname { s A } \colon$ Temperature $\tau _ { \mathrm { m e a n } } = 0 . 1 ~ ( \mathrm { E q . } ~ 3 ) , \tau _ { \mathrm { m a x } } = 0 . 2 ~ ( \mathrm { E q . } ~ 4 )$

• HM: Masking probability $p = 0 . 5$ for balanced RM and CM.

• Loss Weights: Primary objective weight $\alpha = 0 . 1$ , auxiliary objective weight $\beta = 0 . 3$

Due to limited computational resources and empirical evidence suggesting that large-scale language model pretraining is relatively robust to hyperparameter choices within reasonable ranges — as performance is primarily governed by scale rather than fine-tuned hyperparameters [38], [39] — no further hyperparameter tuning was performed beyond the values listed above.

TABLE I  
KEY PARAMETER DETAILS OF SATS MODEL SIZES.
<table><tr><td></td><td>Layers</td><td>dmodel</td><td>dff</td><td>Heads</td><td>Params</td></tr><tr><td>1  $\overline { { \mathbf { S } \mathbf { A } \mathbf { T } \mathbf { S } _ { \mathbf { S } } } }$ </td><td>6</td><td>384</td><td>1536</td><td>6</td><td>14M</td></tr><tr><td> $\mathrm { S A T S _ { B } }$ </td><td>9</td><td>768</td><td>3072</td><td>12</td><td>70M</td></tr></table>

## IV. THEORETICAL ANALYSIS

## A. Mean Consistency and Max Discriminability

We provide a concise justification for the two pooling choices in SA: mean-pooled representations are pulled together to encourage cross-scale semantic consistency, whereas maxpooled representations are pushed apart to preserve scalespecific information.

Consider a time series patch decomposed as

$$
x [ n ] = \ell [ n ] + h [ n ] ,
$$

where $\ell [ n ]$ is a low-frequency trend satisfying the Lipschitz condition $| \ell [ n ] - \ell [ m ] | \leq K | n - m | \ [ 4 0 ]$ , [41], and $h [ n ]$ is a zero-mean high-frequency component with bounded variance and short-range dependence. For a patch of length $P ,$ define

$$
\mu _ { P } = \frac { 1 } { P } \sum _ { n = 1 } ^ { P } x [ n ] , \qquad M _ { P } = \operatorname * { m a x } _ { 1 \leq n \leq P } x [ n ] .
$$

1) Mean consistency: The mean statistic suppresses highfrequency fluctuations. Under short-range dependence,

$$
\mathrm { V a r } ( \mu _ { P } ) = \frac { 1 } { P ^ { 2 } } \sum _ { n , m = 1 } ^ { P } \mathrm { C o v } ( h [ n ] , h [ m ] ) \leq \frac { C _ { \mathrm { c o v } } \sigma _ { \operatorname* { m a x } } ^ { 2 } } { P } ,
$$

which shows that the variance of the high-frequency component decays as $O ( 1 / P )$ . Moreover, since the trend component is Lipschitz smooth, the expectation gap between two patch lengths is controlled by their scale difference:

$$
\big | \mathbb { E } [ \mu _ { P _ { 1 } } ] - \mathbb { E } [ \mu _ { P _ { 2 } } ] \big | \le \frac { K } { 2 } | P _ { 1 } - P _ { 2 } | .
$$

Thus, mean pooling yields stable scale-wise prototypes that mainly retain shared low-frequency semantics.

2) Max discriminability: In contrast, the max statistic remains sensitive to local high-frequency responses. Under mild tail conditions for the high-frequency component, classical extreme value theory [42] gives

$$
\mathbb { E } \left[ \operatorname* { m a x } _ { n \leq P } h [ n ] \right] \asymp \sigma _ { \operatorname* { m a x } } \sqrt { 2 \log P } ,
$$

up to lower-order terms. Therefore, max-pooled anchors emphasize scale-dependent peaks whose magnitude varies with the patch length. This makes them suitable for preserving scale-specific information that would be weakened if all scale representations were only pulled together.

3) Summary: These complementary properties motivate the design of SA. Pulling mean-pooled embeddings together promotes a coherent cross-scale semantic space, while pushing max-pooled embeddings apart preserves scale-sensitive anchor diversity and mitigates scale-wise representation collapse.

## B. Geometric Correspondence with Contrastive Learning

We further examine the intrinsic connection between SA and contrastive representation learning. Although our objective does not rely on conventional instance-level negative sampling, it preserves the core contrastive geometry: aligning related representations while dispersing informative anchors. This perspective explains why mean-attraction and max-repulsion are consistent with contrastive learning principles and can promote cross-scale consistency while reducing collapse risk.

1) Structured contrastive geometry: We make this geometric prior explicit through a normalized L2 analytic surrogate:

$$
\mathcal { L } _ { \mathrm { s a } } ^ { \mathrm { a n a l y t i c } } = \underbrace { \sum _ { i = 1 } ^ { N } \| h _ { i } - \bar { h } \| ^ { 2 } } _ { \mathcal { L } _ { \mathrm { a l i g n } } } - \underbrace { \beta \sum _ { i \neq t } \| m _ { i } - m _ { t } \| ^ { 2 } } _ { \mathcal { L } _ { \mathrm { s e p } } } ,
$$

where $\{ m _ { i } \} _ { i = 1 } ^ { N } \subset \mathbb { S } ^ { D - 1 }$ are normalized scale-wise max anchors, $h _ { i }$ are scale-wise mean prototypes, and h<sup>¯</sup> is their centroid. For normalized embeddings, $\| x - y \| ^ { 2 } = 2 - 2 x ^ { \top } y ;$ hence the first term increases similarity between mean prototypes and a shared semantic center, while the second decreases similarity among max anchors. The surrogate does not replace the InfoNCE-style training loss; it exposes the same normalized attraction–repulsion geometry. Equations 3–4 implement this geometry in smooth softmax form: mean-pooled prototypes are aligned, and scale-dominant max anchors are dispersed. This correspondence yields a structured alignment–uniformity variant [43], where uniformity acts only on the compact anchor set $\{ m _ { i } \}$ rather than all tokens.

2) Mitigation of Representation Collapse: The separation term provides an anti-collapse bias for scale-wise max anchors. Since these anchors capture scale-sensitive high-variation responses, angular separation preserves diversity in the scaleanchor subspace. This connects to dimensional collapse, commonly characterized by degeneration of the embedding covariance spectrum [44].

Formally, let $\{ m _ { i } \} _ { i = 1 } ^ { N } \subset \mathbb { S } ^ { D - 1 }$ denote the normalized maxpooled scale anchors, with $\bar { m } = N ^ { - 1 } \sum _ { i } m _ { i }$ and

$$
C _ { m } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } ( m _ { i } - \bar { m } ) ( m _ { i } - \bar { m } ) ^ { \top } .
$$

Then

$$
\mathrm { t r } ( C _ { m } ) = 1 - \| \bar { m } \| ^ { 2 } = \frac { 1 } { 2 N ^ { 2 } } \sum _ { i , j = 1 } ^ { N } \| m _ { i } - m _ { j } \| ^ { 2 } .
$$

Thus, increasing pairwise separation directly increases total anchor variance. Consequently, $\mathcal { L } _ { \mathrm { s e p } }$ complements $\mathcal { L } _ { \mathrm { a l i g n } }$ by discouraging scale-wise point collapse and preserving non-zero variance in the anchor subspace.

3) Summary: Overall, $\mathcal { L } _ { \mathrm { s a } }$ is a lightweight anchor-level contrastive regularizer. It aligns mean-pooled prototypes for cross-scale consistency, disperses max-pooled anchors for scalespecific diversity, and reduces scale-wise collapse risk without large-batch negative sampling.

TABLE II  
FULL RESULTS OF ZERO-SHOT FORECASTING ACROSS ALL EVALUATED MODELS. LOWER VALUES OF MSE AND MAE INDICATE SUPERIOR PERFORMANCE. AS TIMESFM INCORPORATES WEATHER DATA DURING PRETRAINING, IT IS EXCLUDED FROM EVALUATION ON THIS DATASET (DENOTED BY “–”). RED HIGHLIGHTS THE BEST RESULT, WHILE BLUE MARKS THE SECOND BEST.
<table><tr><td colspan="2">Models</td><td colspan="2">SATSS</td><td colspan="2">SATSB</td><td colspan="2">Timer-XL</td><td colspan="2">Time-MoEB</td><td colspan="2">MoiraiB</td><td colspan="2">ChronosL</td><td colspan="2">Moment</td><td colspan="2">TimesFM</td></tr><tr><td colspan="2">Metrics 96</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td></tr><tr><td colspan="2"></td><td>0.375</td><td>0.393 0.415</td><td>0.360</td><td>0.387</td><td>0.369</td><td>0.391</td><td>0.357</td><td>0.381</td><td>0.383</td><td>0.402</td><td>0.441</td><td>0.390</td><td>0.688</td><td>0.557</td><td>0.414</td><td>0.404</td></tr><tr><td rowspan="8">ETh1</td><td>192</td><td>0.412</td><td></td><td>0.395</td><td>0.409</td><td>0.405</td><td>0.413</td><td>0.384</td><td>0.404</td><td>0.425</td><td>0.429</td><td>0.502</td><td>0.424</td><td>0.688</td><td>0.560</td><td>0.465</td><td>0.434</td></tr><tr><td>336</td><td>0.423</td><td>0.425</td><td>0.413</td><td>0.422</td><td>0.418</td><td>0.423</td><td>0.411</td><td>0.434</td><td>0.456</td><td>0.450</td><td>0.576</td><td>0.467</td><td>0.675</td><td>0.563</td><td>0.503</td><td>0.456</td></tr><tr><td>720</td><td>0.418</td><td>0.441</td><td>0.413</td><td>0.438</td><td>0.423</td><td>0.441</td><td>0.449</td><td>0.477</td><td>0.470</td><td>0.473</td><td>0.835</td><td>0.583</td><td>0.683</td><td>0.585</td><td>0.511</td><td>0.481</td></tr><tr><td>AVG</td><td>0.407</td><td>0.418</td><td>0.395</td><td>0.414</td><td>0.404</td><td>0.417</td><td>0.400</td><td>0.424</td><td>0.433</td><td>0.438</td><td>0.589</td><td>0.466</td><td>0.684</td><td>0.566</td><td>0.473</td><td>0.444</td></tr><tr><td>96</td><td>0.283</td><td>0.328</td><td>0.273</td><td>0.331</td><td>0.283</td><td>0.342</td><td>0.305</td><td>0.359</td><td>0.277</td><td>0.327</td><td>0.320</td><td>0.345</td><td>0.342</td><td>0.396</td><td>0.315</td><td>0.349</td></tr><tr><td>192</td><td>0.343</td><td>0.369</td><td>0.330</td><td>0.372</td><td>0.340</td><td>0.379</td><td>0.351</td><td>0.386</td><td>0.340</td><td>0.374</td><td>0.406</td><td>0.399</td><td>0.354</td><td>0.402</td><td>0.388</td><td>0.395</td></tr><tr><td>336</td><td>0.365</td><td>0.391</td><td>0.353</td><td>0.396</td><td>0.366</td><td>0.400</td><td>0.391</td><td>0.418</td><td>0.371</td><td>0.401</td><td>0.492</td><td>0.453</td><td>0.356</td><td>0.407</td><td>0.422</td><td>0.427</td></tr><tr><td>720</td><td>0.404</td><td>0.424 0.380</td><td></td><td>0.409</td><td>0.397</td><td>0.431</td><td>0.419</td><td>0.454</td><td>0.394</td><td>0.426</td><td>0.603</td><td>0.511</td><td>0.395</td><td>0.434</td><td>0.443</td><td>0.454</td></tr><tr><td rowspan="8">ET2 ETm1</td><td>AVG</td><td>0.349</td><td>0.378 0.334</td><td></td><td>0.377</td><td>0.347</td><td>0.388</td><td>0.367</td><td>0.404</td><td>0.345</td><td>0.382</td><td>0.455</td><td>0.427</td><td>0.362</td><td>0.410</td><td>0.392</td><td>0.406</td></tr><tr><td>96</td><td>0.325</td><td>0.353</td><td>0.323</td><td>0.345</td><td>0.317</td><td>0.356</td><td>0.338</td><td>0.368</td><td>0.396</td><td>0.382</td><td>0.457</td><td>0.403</td><td>0.654</td><td>0.527</td><td>0.361</td><td>0.370</td></tr><tr><td>192</td><td>0.352</td><td>0.372</td><td>0.352</td><td>0.364</td><td>0.358</td><td>0.381</td><td>0.353</td><td>0.388</td><td>0.425</td><td>0.402</td><td>0.530</td><td>0.450</td><td>0.662</td><td>0.532</td><td>0.414</td><td>0.405</td></tr><tr><td>336</td><td>0.372</td><td>0.387</td><td>0.371</td><td>0.379</td><td>0.386</td><td>0.401</td><td>0.381</td><td>0.413</td><td>0.452</td><td>0.415</td><td>0.577</td><td>0.481</td><td>0.672</td><td>0.537</td><td>0.445</td><td>0.429</td></tr><tr><td>720</td><td>0.405</td><td>0.410</td><td>0.401</td><td>0.403</td><td>0.430</td><td>0.431</td><td>0.504</td><td>0.493</td><td>0.477</td><td>0.431</td><td>0.660</td><td>0.526</td><td>0.692</td><td>0.551</td><td>0.512</td><td>0.471</td></tr><tr><td>AVG</td><td>0.364</td><td>0.380</td><td>0.362</td><td>0.373</td><td>0.373</td><td>0.392</td><td>0.394</td><td>0.416</td><td>0.437</td><td>0.407</td><td>0.556</td><td>0.465</td><td>0.670</td><td>0.537</td><td>0.433</td><td>0.419</td></tr><tr><td>96</td><td>0.172</td><td>0.255</td><td>0.167</td><td>0.251</td><td>0.189</td><td>0.277</td><td>0.201</td><td>0.291</td><td>0.195</td><td>0.269</td><td>0.197</td><td>0.271</td><td>0.260</td><td>0.335</td><td>0.202</td><td>0.270</td></tr><tr><td></td><td>0.226</td><td>0.292 0.327</td><td>0.222</td><td>0.290</td><td>0.241</td><td>0.315</td><td>0.258</td><td>0.334</td><td>0.247</td><td>0.303</td><td>0.254</td><td>0.314</td><td>0.289</td><td>0.350</td><td>0.289</td><td>0.321</td></tr><tr><td rowspan="7">ETm32</td><td>192 336</td><td>0.279</td><td></td><td>0.269</td><td>0.323</td><td>0.286</td><td>0.348</td><td>0.324</td><td>0.373</td><td>0.291</td><td>0.333</td><td>0.313</td><td>0.353</td><td>0.324</td><td>0.369</td><td>0.360</td><td>0.366</td></tr><tr><td>720</td><td>0.369</td><td>0.385</td><td>0.343</td><td>0.374</td><td>0.375</td><td>0.402</td><td>0.488</td><td>0.464</td><td>0.355</td><td>0.377</td><td>0.416</td><td>0.415</td><td>0.394</td><td>0.409</td><td>0.462</td><td>0.430</td></tr><tr><td>AVG</td><td>0.262</td><td>0.315</td><td>0.250</td><td>0.309</td><td>0.273</td><td>0.336</td><td>0.318</td><td>0.366</td><td>0.272</td><td>0.321</td><td>0.295</td><td>0.338</td><td>0.317</td><td>0.366</td><td>0.328</td><td>0.347</td></tr><tr><td>96</td><td>0.180</td><td>0.236</td><td>0.162</td><td>0.217</td><td>0.171</td><td>0.225</td><td>0.160</td><td>0.214</td><td>0.176</td><td>0.210</td><td>0.194</td><td>0.235</td><td>0.243</td><td>0.255</td><td></td><td></td></tr><tr><td>192</td><td>0.226</td><td>0.280</td><td>0.210</td><td>0.265</td><td>0.221</td><td>0.271</td><td>0.210</td><td>0.260</td><td>0.218</td><td>0.251</td><td>0.249</td><td>0.285 0.327</td><td>0.278</td><td>0.329 0.346</td><td></td><td></td></tr><tr><td>336</td><td>0.274 0.341</td><td>0.316</td><td>0.258 0.325</td></table>

## V. EXPERIMENTS

## A. Benchmark Description

We evaluate SATS on three complementary benchmark collections: LSTF [45], GIFT-Eval [46], and Monash [47]. Together, they cover long-horizon forecasting, zero-shot crossdataset generalization, and in-distribution evaluation across diverse domains, frequencies, and horizons.

• LSTF: A collection of widely used multivariate datasets from domains such as electricity, traffic, and weather, used here to evaluate long-term forecasting.

• GIFT-Eval: A general forecasting benchmark for evaluating transfer to unseen heterogeneous datasets. It spans diverse domains, frequencies, variable types, and prediction horizons, and reports both point and probabilistic forecasting quality through metrics such as MASE and CRPS.

• Monash: A large-scale collection of univariate real-world datasets, used here for in-distribution evaluation under heterogeneous data distributions.

## B. Benchmarking Setup

Baselines. We conduct extensive comparisons with widely adopted time series foundation models, including Timer-XL [48], Time-MoE [11], Moirai [12], Chronos [13], Moment [27], TimesFM [49], and TTM-R2 [50]. To broaden the comparative scope, we additionally incorporate two methods adapted from foundation models of other modalities: VisionTS [51] and LLMTime [52]. Furthermore, following the recommendations of Bergmeir [53], we extend our indistribution evaluation to encompass traditional baselines, including Naive, ETS [54], and DeepAR [55].

Evaluation Setup. Baselines follow their original configurations. For SATS, we follow Moirai, search context lengths in {1000, 2000, 3000, 4000, 5000}, and use Woo et al.’s frequency-specific patch candidates [12]:

• Yearly/Quarterly: 8

• Monthly: 8, 16, 32

• Weekly/Daily: 16, 32

• Hourly: 32, 64

• Minute-level: 32, 64, 128

• Second-level: 64, 128

Empirically, we choose the largest feasible patch size with a lookback of at least 3000. Metrics use 100 predictive samples and report results based on the mean prediction. As pretrained baselines often impose model-specific input lengths, such as Time-MoE’s four-horizon input [11] and Timer-XL’s datasetdependent lengths [48], we follow each model’s recommended setting for fairness.

## C. Zero-shot Forecasting

We evaluate zero-shot forecasting on LSTF and GIFT-Eval. LSTF tests long-horizon forecasting on datasets excluded from pretraining, while GIFT-Eval tests cross-dataset generalization under a leakage-controlled protocol.

1) LSTF Benchmark: We evaluate five LSTF datasets excluded from LOTSA, using horizons {96, 192, 336, 720} and reporting MSE and MAE. For baselines with multiple variants, we exclude models above 1B parameters and report the variant with the best average performance. Electricity, Traffic, and PEMS are excluded because they appear in the pretraining corpora of most models [11], [12], [48].

![](images/4e4a0a2717396a307ec6d8105bfbbcc8cb383f21b9590b2d1bcd00091fd4a98f.jpg)

![](images/59c6bf1f85db3597c120f4253273ac9ce8847483f79832e71cd3cbb284bbf123.jpg)  
Fig. 3. Zero-shot forecasting performance evaluated on 23 datasets from GIFT-Eval benchmark [46]. Lower values of MASE and CRPS indicate superior performance. Methods trained with access to these evaluation datasets during pretraining are denoted with asterisks (\*). Results were aggregated in accordance with the standard method of GIFT-Eval.

Result. The detailed zero-shot results are presented in Table II, where ${ \bf S A T S _ { B } }$ consistently achieves state-of-the-art per formance. Compared to $\mathbf { M o i r a i } _ { \mathrm { B } }$ (encoder-only), the strongest multi-scale FFNs-embedded baseline, ${ \bf S A T S _ { B } }$ achieves a 9.2% improvement in MSE. It also outperforms Timer-XL (decoderonly) and Chronos<sub>L</sub> (encoder-decoder) with MSE improvements of 4.2% and 27.4%, respectively. Notably, SATS<sub>B</sub> contains only 70M parameters, which is substantially fewer than those of the compared baselines. Moreover, even the lightweight $\mathrm { S A T S } _ { \mathrm { S } }$ with 14M parameters surpasses all other baselines in overall average performance, highlighting its efficiency. In addition, SATS exhibits a clear performance gain as model size increases, revealing strong scalability. This trend contrasts with models such as Time-MoE [11] and Moirai [12], whose performance plateaus or even degrades with larger model configurations.

2) GIFT-Eval Benchmark: We follow the standard GIFT-Eval protocol on 23 datasets spanning economics, energy, healthcare, nature, sales, transportation, and cloud operations. To enforce a strict zero-shot setting, we remove all GIFT-Eval overlaps from LOTSA and retrain SATS. Performance is aggregated by the official protocol, covering both point and probabilistic forecasting metrics.

Result. The detailed zero-shot results are shown in Figure 3. Overall, ${ \bf S A T S _ { B } }$ delivers consistently state-of-the-art performance across metrics. Against the strongest encoderonly baseline, $\mathrm { { M o i r a i } _ { L } , }$ it yields substantial gains—8.3% in MASE and 8.2% in CRPS. Notably, even when set against the runner-up, Chronos- $\mathbf { \nabla \cdot B _ { B } }$ , SATS maintains a clear lead in CRPS under a strictly leakage-free evaluation protocol while requiring only 35% of its parameters, underscoring its ability to learn highly transferable temporal representations during pretraining. Beyond encoder-based models, SATS also surpasses VisionTS—which converts sequences into images for training—with consistently more stable improvements on both MASE and CRPS. This suggests that SATS captures the underlying data distribution more faithfully, rather than merely improving point forecasts as VisionTS tends to do. Together, these findings highlight SATS’s robust cross-dataset generalization under both point and probabilistic forecasting metrics.

## D. In-distribution Forecasting

We evaluate in-distribution forecasting on 29 Monash datasets [47]. Only training portions are included in LOTSA, while test sets are reserved for evaluation. Results are reported as normalized MAE relative to a naive forecast and aggregated by the geometric mean across datasets.

Result. As shown in Figure 4, SATS consistently outperforms all competing methods. Compared to $\mathbf { M o i r a i } _ { \mathrm { L } }$ , the best baseline trained on clean data, ${ \bf S A T S _ { B } }$ achieves a 6.9% improvement while using only 22.6% of its parameters. Similarly, against Chronos<sub>S</sub>, the strongest baseline under data contamination, ${ \mathrm { S A T S } } _ { \mathrm { S } }$ achieves superior performance with just 30.4% of its parameter count. Notably, the gain from ${ \mathrm { S A T S } } _ { \mathrm { S } }$ to ${ \bf S A T S _ { B } }$ is modest, likely because in-distribution forecasting involves limited temporal complexity, where increasing model size yields diminishing returns.

## E. Ablation Studies<sup>1</sup>

1) Module Design: We begin by performing ablation studies on the components of ${ \bf S A T S _ { B } }$ , with results summarized in Table III. On ETT, removing SA consistently degrades performance, while discarding either HM component causes larger drops in most cases; Weather shows a similar MSE pattern, with relatively small MAE differences. These results indicate that HM is central to long-term forecasting, where random and contiguous masking jointly encourage robust temporal representations, while SA provides complementary gains by regularizing cross-scale token spaces. On GIFT-Eval, removing SA yields the worst results, whereas removing individual HM components has a smaller impact. This contrast suggests that HM mainly strengthens long-horizon dependency learning on LSTF, while SA is more critical for cross-dataset generalization. Together with the t-SNE visualizations in Section V-F1, these findings show that SA forms a stable, structured embedding space and that the two components provide complementary benefits.

![](images/424cb3ae4aefa6658f4a2e2f01e3f3b24afe5958d7f6dee09094dfadf82c6aaa.jpg)  
Fig. 4. In-distribution forecasting performance evaluated on 29 datasets from the Monash benchmark [47]. Methods trained with access to these evaluation datasets during pretraining are denoted with asterisks (\*). Results are normalized using the naive forecast and summarized with the geometric mean.

TABLE III  
MODULE ABLATION UNDER ZERO-SHOT EVALUATION. RED AND BLUE INDICATE THE BEST AND SECOND-BEST RESULTS.
<table><tr><td rowspan="2">Model variants Metrics</td><td colspan="2">ETTh</td><td colspan="2">ETTm</td><td colspan="2">Weather</td><td colspan="2">GIFT-Eval</td></tr><tr><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MASE</td><td>CRPS</td></tr><tr><td>SATSB</td><td>0.365</td><td>0.396</td><td>0.306</td><td>0.341</td><td>0.239</td><td>0.283</td><td>0.802</td><td>0.550</td></tr><tr><td>w/o SA</td><td>0.370</td><td>0.404</td><td>0.311</td><td>0.344</td><td>0.244</td><td>0.281</td><td>0.866</td><td>0.607</td></tr><tr><td>w/o CM</td><td>0.398</td><td>0.409</td><td>0.317</td><td>0.351</td><td>0.259</td><td>0.293</td><td>0.819</td><td>0.569</td></tr><tr><td>w/o RM</td><td>0.381</td><td>0.400</td><td>0.324</td><td>0.354</td><td>0.251</td><td>0.269</td><td>0.833</td><td>0.583</td></tr></table>

2) Alignment Mechanism: The key design of SA is to align mean embeddings while separating maximal embeddings, thereby preserving both cross-scale consistency and scaleanchor diversity. We therefore vary the pooling choices in Table IV to examine whether these two roles are necessary. Removing the max-embedding repulsion term causes a clear performance drop, confirming the importance of scale-anchor separation. Replacing max pooling with min or random pooling generally yields degraded or less stable results, indicating that maximal embeddings better capture scale-specific information. Applying alignment only to maximal embeddings also remains suboptimal in most cases, suggesting that this constraint is too weak to prevent scale-wise anchor collapse. Overall, bringing mean embeddings closer while pushing maximal embeddings apart is a robust design choice, consistent with the theoretical analysis in Section IV-A.

3) Transformer Types: We further assess architectural generality by integrating the proposed modules into three Transformer variants, omitting HM for the decoder-only model because its pre-training objective is incompatible with hierarchical masking. As shown in Table V, SA and HM bring consistent gains across applicable architectures: HM mainly improves long-horizon forecasting, while SA benefits crossdataset generalization. Under the same setting, the decoderonly model performs best, likely due to denser training signals [39], though its incompatibility with HM limits further gains. By contrast, the larger encoder-decoder variant often trails the encoder-only model, possibly because additional crossattention layers introduce optimization instability and gradient dilution [56]. Overall, these results suggest that SA and HM provide architecture-agnostic improvements.

TABLE IV  
ALIGNMENT ABLATION UNDER ZERO-SHOT EVALUATION. A DASH DENOTES A REMOVED OBJECTIVE; RED AND BLUE INDICATE THE BEST AND SECOND-BEST RESULTS.
<table><tr><td colspan="2">SATSB</td><td colspan="2">ETTh</td><td colspan="2">ETTm</td><td colspan="2">Weather</td><td colspan="2">GIFT-Eval</td></tr><tr><td>Close</td><td>Far</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MASE</td><td>CRPS</td></tr><tr><td>Mean</td><td>Max</td><td>0.365</td><td>0.396</td><td>0.306</td><td>0.341</td><td>0.239</td><td>0.283</td><td>0.802</td><td>0.550</td></tr><tr><td>Mean</td><td></td><td>0.388</td><td>0.410</td><td>0.353</td><td>0.368</td><td>0.269</td><td>0.294</td><td>0.945</td><td>0.846</td></tr><tr><td>Mean</td><td>Min</td><td>0.386</td><td>0.414</td><td>0.332</td><td>0.361</td><td>0.248</td><td>0.287</td><td>0.892</td><td>0.637</td></tr><tr><td>Mean</td><td>Random</td><td>0.372</td><td>0.403</td><td>0.312</td><td>0.352</td><td>0.243</td><td>0.288</td><td>0.821</td><td>0.567</td></tr><tr><td>Max</td><td></td><td>0.368</td><td>0.398</td><td>0.316</td><td>0.348</td><td>0.237</td><td>0.277</td><td>0.817</td><td>0.573</td></tr></table>

TABLE V

TRANSFORMER-TYPE ABLATION UNDER ZERO-SHOT EVALUATION. VALUES ARE AVERAGE MSE OR MASE; “-” DENOTES AN INAPPLICABLE SETTING, AND RED MARKS THE BEST RESULT.
<table><tr><td>Transformer</td><td colspan="3">Encoder-Only</td><td colspan="2">Decoder-Only</td><td colspan="2">Encoder-Decoder</td></tr><tr><td>Variants</td><td>RAW</td><td>+SA</td><td>+HM</td><td>RAW +SA</td><td>+HM</td><td>RAW +SA</td><td>+HM</td></tr><tr><td>ETTh</td><td>0.389</td><td>0.383</td><td>0.370</td><td>0.382 0.376</td><td></td><td>0.392 0.376</td><td>0.368</td></tr><tr><td>ETTm</td><td>0.355</td><td>0.320</td><td>0.311</td><td>0.353 0.325</td><td></td><td>0.366 0.334</td><td>0.323</td></tr><tr><td>Weather</td><td>0.250</td><td>0.258</td><td>0.244</td><td>0.247 0.247</td><td></td><td>0.253 0.250</td><td>0.244</td></tr><tr><td>GIFT-Eval</td><td>0.901</td><td>0.839</td><td>0.866</td><td>0.892 0.834</td><td></td><td>0.912 0.842</td><td>0.870</td></tr></table>

## F. Model Analysis

1) Mechanism Analysis: To understand what SA learns beyond forecasting gains, we examine whether it organizes multi-resolution tokens into a coherent yet scale-discriminative embedding space. We consider both a sufficient-scale regime with representative tokens at each scale and a scarce-scale regime with underrepresented patch sizes. In Figure 5, SATS forms compact yet separable clusters across patch sizes, even when size-8 tokens are scarce, whereas w/o SA exhibits confusion between patch sizes 16 and 32 and nearly absorbs scarce size-8 tokens. This visual pattern is consistent with

![](images/2bf718da7a7141af988adc3ecf9861b6be6e596b9a45d3307d32162ce11dde69.jpg)

![](images/adf24d3603c20814bc66d7f1a55cd41f378e05b92c174eec24fda89bb87437e0.jpg)

![](images/9f3f86d0457907ab9d8f9b4df8b0f854b78e9430a8aa1c789632596df6ce05a6.jpg)  
Fig. 5. T-SNE token distributions in sufficient- and scarce-scale regimes. Colors denote patch-size origins.

TABLE VI  
COSINE SIMILARITY BETWEEN MULTI-RESOLUTION PATCH EMBEDDINGS ({8, 16, 32, 64}) AND SIZE-128 REPRESENTATIONS UNDER MEAN AND MAX POOLING.
<table><tr><td></td><td colspan="4">Mean Pooling</td><td colspan="4">Max Pooling</td></tr><tr><td>PatchSize</td><td>8</td><td>16</td><td>32</td><td>64</td><td>8</td><td>16</td><td>32</td><td>64</td></tr><tr><td>SATSB</td><td>0.979</td><td>0.988</td><td>0.998</td><td>0.992</td><td>0.639</td><td>0.424</td><td>0.437</td><td>0.509</td></tr><tr><td>w/o SA</td><td>0.189</td><td>-0.032</td><td>-0.054</td><td>0.293</td><td>0.752</td><td>0.721</td><td>0.723</td><td>0.889</td></tr></table>

Table VI, where downsampled patches at resolutions {8, 16, 32, 64} are compared with the original size-128 embeddings. SATS achieves very high mean-pooling similarity (0.979–0.998), indicating cross-resolution alignment, while retaining lower max-pooling similarity (0.424–0.639), indicating preserved scale-specific anchors. Conversely, w/o SA shows weak or negative mean similarity (-0.054–0.293) but high max similarity (0.721–0.889), revealing fragmented global semantics and a tendency toward local-anchor collapse. Thus, the qualitative clusters and quantitative geometry tell a consistent story: SA pulls semantically matched resolutions toward a shared center while separating scale-specific maxima, producing coherent representations that ease Transformer pretraining and support generalization.

2) Patch Protocol Sensitivity: To further understand the mechanism of SATS and guide patch-size selection when sampling-frequency metadata is noisy or unavailable, we evaluate its sensitivity to patch protocols on the LSTF benchmark. As shown in Figure 6, the prescribed protocol consistently yields the best performance, while deviations exhibit an asymmetric effect: using patch sizes smaller than the protocol causes substantial degradation, whereas using larger patch sizes leads to only mild performance loss. This suggests that excessively small patches fragment temporal semantics and fail to capture complete patterns [32], while larger patches can still retain macroscopic trends through effective downsampling [57]. Therefore, in zero-shot scenarios with unreliable metadata, a conservative strategy is to prefer larger patch sizes within overlapping protocol ranges; when metadata is absent, spectral analysis can be used to select the closest frequency-based protocol.

![](images/c48cf0f39620db027c390d5f7590672d42349dbc4c93f9c0d0715589ea468b0d.jpg)

![](images/ffcd0337f218c1dfa62c27a32b9455c142e49ae411a7feace897a0b639a10167.jpg)

![](images/d95e3cc7dbe905977f30ee7f7e4c21dc8027e07a5c10f27c1e6820fc74d313c7.jpg)

Fig. 6. Sensitivity analysis of patch protocols on the LSTF benchmark. The charts report average MSE (lower is better) across different patch sizes.  
![](images/b71209f73441d0edc756429926d45e00597c86db3f6257e8b17729a49571b6d9.jpg)  
Fig. 7. Model efficiency under zero-shot LSTF evaluation, measured as MS $\textstyle { \frac { 1 } { 3 \cdot \log ( N + 1 ) } }$ . Higher values indicate better accuracy-parameter trade-offs for foundation models, whose parameter count meaningfully proxies capacity.

3) Model Efficiency: To examine whether the gains of SATS translate into a practical accuracy-parameter trade-off, we evaluate model efficiency under the zero-shot LSTF setting. Since the proposed alignment and masking strategies introduce no additional learnable parameters to the multi-FFNs backbone, we measure efficiency as the inverse of the product between zero-shot error and the logarithm of model size. As shown in Figure 7, SATS achieves a strong balance between accuracy and compactness: ${ \bf S A T S _ { B } }$ attains SOTA accuracy and surpasses the runner-up model, Timer-XL, by 8.9% in efficiency, while the lightweight $\mathrm { S A T S } _ { \mathrm { S } }$ delivers a $\mathbf { 6 5 . 6 }$ % efficiency improvement despite only slightly outperforming Timer-XL in accuracy. These results indicate that SATS improves predictive performance without relying on parameter expansion, making it suitable for resource-constrained or real-time deployment.

## VI. CONCLUSION

This paper presents SATS, a Scale-Aware foundation model for Time Series that addresses the challenge of fragmented token spaces and misaligned representations in time series pretraining. SA is introduced to unify representations across patch sizes by jointly minimizing inter-scale embedding discrepancies and preserving scale-specific modeling capacity. Furthermore, HM combines RM and CM to capture temporal dependencies at multiple resolutions. Extensive experiments demonstrate that SATS achieves superior generalization and robustness while maintaining high efficiency, as its proposed alignment and masking strategies introduce no additional learnable parameters.

[1] T. Brown, B. Mann, N. Ryder, M. Subbiah, J. D. Kaplan, P. Dhariwal, A. Neelakantan, P. Shyam, G. Sastry, A. Askell et al., “Language models are few-shot learners,” Advances in Neural Information Processing Systems, vol. 33, pp. 1877–1901, 2020.

[2] A. Dubey, A. Jauhri, A. Pandey, A. Kadian, A. Al-Dahle, A. Letman, A. Mathur, A. Schelten, A. Yang, A. Fan et al., “The llama 3 herd of models,” 2024, arXiv:2407.21783.

[3] M. Oquab, T. Darcet, T. Moutakanni, H. Vo, M. Szafraniec, V. Khalidov, P. Fernandez, D. Haziza, F. Massa, A. El-Nouby, M. Assran, N. Ballas, W. Galuba, R. Howes, P.-Y. Huang, S.-W. Li, I. Misra, M. Rabbat, V. Sharma, G. Synnaeve, H. Xu, H. Jegou, J. Mairal, P. Labatut, A. Joulin, and P. Bojanowski, “Dinov2: Learning robust visual features without supervision,” Transactions on Machine Learning Research, 2024.

[4] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, G. Krueger, and I. Sutskever, “Learning transferable visual models from natural language supervision,” in Proceedings of the 38th International Conference on Machine Learning, vol. 139, 2021, pp. 8748–8763.

[5] A. Baevski, Y. Zhou, A. Mohamed, and M. Auli, “wav2vec 2.0: A framework for self-supervised learning of speech representations,” Advances in Neural Information Processing Systems, vol. 33, pp. 12 449– 12 460, 2020.

[6] A. Radford, J. W. Kim, T. Xu, G. Brockman, C. Mcleavey, and I. Sutskever, “Robust speech recognition via large-scale weak supervision,” in Proceedings of the 40th International Conference on Machine Learning, vol. 202, 2023, pp. 28 492–28 518.

[7] D. Cao, F. Jia, S. O. Arik, T. Pfister, Y. Zheng, W. Ye, and Y. Liu, “Tempo: Prompt-based generative pre-trained transformer for time series forecasting,” in International Conference on Learning Representations, 2024.

[8] M. Jin, S. Wang, L. Ma, Z. Chu, J. Y. Zhang, X. Shi, P.-Y. Chen, Y. Liang, Y.-F. Li, S. Pan, and Q. Wen, “Time-llm: Time series forecasting by reprogramming large language models,” in International Conference on Learning Representations, 2024.

[9] Z. Pan, Y. Jiang, S. Garg, A. Schneider, Y. Nevmyvaka, and D. Song, “s<sup>2</sup>IP-LLM: Semantic space informed prompt learning with LLM for time series forecasting,” in Proceedings of the 41st International Conference on Machine Learning, vol. 235, 2024, pp. 39 135–39 153.

[10] M. Tan, M. Merrill, V. Gupta, T. Althoff, and T. Hartvigsen, “Are language models actually useful for time series forecasting?” Advances in Neural Information Processing Systems, vol. 37, pp. 60 162–60 191, 2024.

[11] X. Shi, S. Wang, Y. Nie, D. Li, Z. Ye, Q. Wen, and M. Jin, “Time-moe: Billion-scale time series foundation models with mixture of experts,” in International Conference on Learning Representations, 2025.

[12] G. Woo, C. Liu, A. Kumar, C. Xiong, S. Savarese, and D. Sahoo, “Unified training of universal time series forecasting transformers,” in Proceedings of the 41st International Conference on Machine Learning, vol. 235, 2024, pp. 53 140–53 164.

[13] A. F. Ansari, L. Stella, C. Turkmen, X. Zhang, P. Mercado, H. Shen, O. Shchur, S. S. Rangapuram, S. P. Arango, S. Kapoor, J. Zschiegner, D. C. Maddix, H. Wang, M. W. Mahoney, K. Torkkola, A. G. Wilson, M. Bohlke-Schneider, and Y. Wang, “Chronos: Learning the language of time series,” Transactions on Machine Learning Research, 2024.

[14] R. Sennrich, B. Haddow, and A. Birch, “Neural machine translation of rare words with subword units,” in Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 2016, pp. 1715–1725.

[15] H. Touvron, M. Cord, M. Douze, F. Massa, A. Sablayrolles, and H. Jegou, “Training data-efficient image transformers & distillation through attention,” in Proceedings of the 38th International Conference on Machine Learning, vol. 139, 2021, pp. 10 347–10 357.

[16] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, J. Uszkoreit, and N. Houlsby, “An image is worth 16x16 words: Transformers for image recognition at scale,” in International Conference on Learning Representations, 2021.

[17] J. Zhang, S. Zheng, X. Wen, X. Zhou, J. Bian, and J. Li, “Elastst: Towards robust varied-horizon forecasting with elastic time-series transformer,” Advances in Neural Information Processing Systems, vol. 37, pp. 119 174– 119 197, 2024.

[18] T. Ronen, O. Levy, and A. Golbert, “Vision transformers with mixedresolution tokenization,” in Proceedings of the IEEE/CVF Conference on

Computer Vision and Pattern Recognition Workshops, 2023, pp. 4613– 4622.

[19] Y. Wang, Y. Qiu, P. Chen, Y. Shu, Z. Rao, L. Pan, B. Yang, and C. Guo, “LightGTS: A lightweight general time series forecasting model,” in Proceedings of the 42nd International Conference on Machine Learning, vol. 267, 2025, pp. 64 109–64 126.

[20] X. Liu, J. Liu, G. Woo, T. Aksu, Y. Liang, R. Zimmermann, C. Liu, S. Savarese, C. Xiong, and D. Sahoo, “Moirai-moe: Empowering time series foundation models with sparse mixture of experts,” in Proceedings of the 42nd International Conference on Machine Learning, vol. 267, 2025, pp. 38 940–38 962.

[21] K. Ding, F. Fan, C. Hou, Z. Wang, L. Wang, Z. Yang, and J. Zhan, “Timemosaic: Temporal heterogeneity guided time series forecasting via adaptive granularity patch and segment-wise decoding,” Proceedings of the AAAI Conference on Artificial Intelligence, vol. 40, no. 25, pp. 20 790–20 798, 2026.

[22] P. Chen, Y. Zhang, Y. Cheng, Y. Shu, Y. Wang, Q. Wen, B. Yang, and C. Guo, “Pathformer: Multi-scale transformers with adaptive pathways for time series forecasting,” in International Conference on Learning Representations, 2024.

[23] S. Wang, H. Wu, X. Shi, T. Hu, H. Luo, L. Ma, J. Y. Zhang, and J. Zhou, “Timemixer: Decomposable multiscale mixing for time series forecasting,” in International Conference on Learning Representations, 2024.

[24] Y. Liu, H. Zhang, C. Li, X. Huang, J. Wang, and M. Long, “Timer: Generative pre-trained transformers are large time series models,” in Proceedings of the 41st International Conference on Machine Learning, vol. 235, 2024, pp. 32 369–32 399.

[25] K. Rasul, A. Ashok, A. R. Williams, H. Ghonia, R. Bhagwatkar, A. Khorasani, M. J. D. Bayazi, G. Adamopoulos, R. Riachi, N. Hassen, M. Bilos, S. Garg, A. Schneider, N. Chapados, A. Drouin, V. Zantedeschi,ˇ Y. Nevmyvaka, and I. Rish, “Lag-llama: Towards foundation models for probabilistic time series forecasting,” 2024, arXiv:2310.08278.

[26] Q. Yao, C.-H. H. Yang, R. Jiang, Y. Liang, M. Jin, and S. Pan, “Towards neural scaling laws for time series foundation models,” 2025, arXiv:2410.12360.

[27] M. Goswami, K. Szafer, A. Choudhry, Y. Cai, S. Li, and A. Dubrawski, “Moment: A family of open time-series foundation models,” in Proceedings of the 41st International Conference on Machine Learning, vol. 235, 2024, pp. 16 115–16 152.

[28] T. Gao, X. Yao, and D. Chen, “SimCSE: Simple contrastive learning of sentence embeddings,” in Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, 2021, pp. 6894– 6910.

[29] C. Jia, Y. Yang, Y. Xia, Y.-T. Chen, Z. Parekh, H. Pham, Q. Le, Y.-H. Sung, Z. Li, and T. Duerig, “Scaling up visual and vision-language representation learning with noisy text supervision,” in Proceedings of the 38th International Conference on Machine Learning, vol. 139, 2021, pp. 4904–4916.

[30] E. Eldele, M. Ragab, Z. Chen, M. Wu, C. K. Kwoh, X. Li, and C. Guan, “Time-series representation learning via temporal and contextual contrasting,” in Proceedings of the Thirtieth International Joint Conference on Artificial Intelligence, IJCAI-21, 2021, pp. 2352–2359.

[31] G. Woo, C. Liu, D. Sahoo, A. Kumar, and S. Hoi, “Cost: Contrastive learning of disentangled seasonal-trend representations for time series forecasting,” in International Conference on Learning Representations, 2022.

[32] Y. Nie, N. H. Nguyen, P. Sinthong, and J. Kalagnanam, “A time series is worth 64 words: Long-term forecasting with transformers,” in International Conference on Learning Representations, 2023.

[33] M. M. Krell, M. Kosec, S. P. Perez, and A. Fitzgibbon, “Efficient sequence packing without cross-contamination: Accelerating large language models without impacting performance,” 2022, arXiv:2107.02027.

[34] J. Su, M. Ahmed, Y. Lu, S. Pan, W. Bo, and Y. Liu, “RoFormer: Enhanced transformer with rotary position embedding,” Neurocomputing, vol. 568, p. 127063, 2024.

[35] N. Shazeer, “Glu variants improve transformer,” 2020, arXiv:2002.05202.

[36] B. Zhang and R. Sennrich, “Root mean square layer normalization,” Advances in Neural Information Processing Systems, vol. 32, 2019.

[37] T. Kim, J. Kim, Y. Tae, C. Park, J.-H. Choi, and J. Choo, “Reversible instance normalization for accurate time-series forecasting against distribution shift,” in International Conference on Learning Representations, 2022.

[38] Y. Liu, M. Ott, N. Goyal, J. Du, M. Joshi, D. Chen, O. Levy, M. Lewis, L. Zettlemoyer, and V. Stoyanov, “Roberta: A robustly optimized bert pretraining approach,” 2019, arXiv:1907.11692.

[39] J. Kaplan, S. McCandlish, T. Henighan, T. B. Brown, B. Chess, R. Child, S. Gray, A. Radford, J. Wu, and D. Amodei, “Scaling laws for neural language models,” 2020, arXiv:2001.08361.

[40] C. Giraud, F. Roueff, and A. Sanchez-Perez, “Aggregation of predictors for nonstationary sub-linear processes and online adaptive forecasting of time varying autoregressive processes,” The Annals of Statistics, pp. 2412–2450, 2015.

[41] S. Yakowitz, L. Gyorfi, J. Kieffer, and G. Morvai, “Strongly consis-¨ tent nonparametric forecasting and regression for stationary ergodic sequences,” Journal of multivariate analysis, vol. 71, no. 1, pp. 24–41, 1999.

[42] M. R. Leadbetter, G. Lindgren, and H. Rootzen,´ Extremes and related properties of random sequences and processes. Springer Science & Business Media, 2012.

[43] T. Wang and P. Isola, “Understanding contrastive representation learning through alignment and uniformity on the hypersphere,” in Proceedings of the 37th International Conference on Machine Learning, vol. 119, 2020, pp. 9929–9939.

[44] L. Jing, P. Vincent, Y. LeCun, and Y. Tian, “Understanding dimensional collapse in contrastive self-supervised learning,” 2022, arXiv:2110.09348.

[45] H. Zhou, S. Zhang, J. Peng, S. Zhang, J. Li, H. Xiong, and W. Zhang, “Informer: Beyond efficient transformer for long sequence time-series forecasting,” in Proceedings of the AAAI conference on artificial intelligence, vol. 35, 2021, pp. 11 106–11 115.

[46] T. Aksu, G. Woo, J. Liu, X. Liu, C. Liu, S. Savarese, C. Xiong, and D. Sahoo, “Gift-eval: A benchmark for general time series forecasting model evaluation,” 2024, arXiv:2410.10393.

[47] R. Godahewa, C. Bergmeir, G. I. Webb, R. J. Hyndman, and P. Montero-Manso, “Monash time series forecasting archive,” 2021, arXiv:2105.06643.

[48] Y. Liu, G. Qin, X. Huang, J. Wang, and M. Long, “Timer-xl: Longcontext transformers for unified time series forecasting,” in International Conference on Learning Representations, 2025.

[49] A. Das, W. Kong, R. Sen, and Y. Zhou, “A decoder-only foundation model for time-series forecasting,” in Proceedings of the 41st International Conference on Machine Learning, vol. 235, 2024, pp. 10 148–10 167.

[50] V. Ekambaram, A. Jati, P. Dayama, S. Mukherjee, N. H. Nguyen, W. M. Gifford, C. Reddy, and J. Kalagnanam, “Tiny time mixers (TTMs): Fast pre-trained models for enhanced zero/few-shot forecasting of multivariate time series,” Advances in Neural Information Processing Systems, vol. 37, pp. 74 147–74 181, 2024.

[51] M. Chen, L. Shen, Z. Li, X. J. Wang, J. Sun, and C. Liu, “VisionTS: Visual masked autoencoders are free-lunch zero-shot time series forecasters,” in Proceedings of the 42nd International Conference on Machine Learning, vol. 267, 2025, pp. 8979–9007.

[52] N. Gruver, M. Finzi, S. Qiu, and A. G. Wilson, “Large language models are zero-shot time series forecasters,” Advances in Neural Information Processing Systems, vol. 36, pp. 19 622–19 635, 2023.

[53] C. Bergmeir, “Fundamental limitations of foundational forecasting models: The need for multimodality and rigorous evaluation,” Vancouver, Canada, Dec. 2024. [Online]. Available: https://cbergmeir.com/talks/ neurips2024/

[54] R. Hyndman, A. Koehler, K. Ord, and R. Snyder, Forecasting with Exponential Smoothing: The State Space Approach. Springer, 2008.

[55] D. Salinas, V. Flunkert, J. Gasthaus, and T. Januschowski, “DeepAR: Probabilistic forecasting with autoregressive recurrent networks,” International journal offorecasting, vol. 36, no. 3, pp. 1181–1191, 2020.

[56] J. Hong and S. Lee, “Variance sensitivity induces attention entropy collapse and instability in transformers,” in Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, 2025, pp. 8360–8378.

[57] H. Wang, J. Peng, F. Huang, J. Wang, J. Chen, and Y. Xiao, “MICN: Multi scale local and global context modeling for long-term series forecasting,” in International Conference on Learning Representations, 2023.