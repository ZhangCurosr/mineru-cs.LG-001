# MAG: MAnifold Guided Semi-Supervised Multi-modal In-Context Learning

Zirui Cheng<sup>1,2</sup>, Xun Xu<sup>2</sup>, Tiankai Chen<sup>2</sup>, Fady Rezk<sup>2</sup>, Bowen Zheng<sup>1</sup>, Xiaodong Shi<sup>1</sup>, Shijie Li<sup>2</sup>, Kangkang Lu<sup>2</sup>, Bharadwaj Veeravalli<sup>1</sup>, Nancy F. Chen<sup>2</sup>

<sup>1</sup>College of Design and Engineering, National University of Singapore <sup>2</sup>Institute for Infocomm Research, ASTAR, Singapore zirui\_c@u.nus.edu

## Abstract

Few-shot in-context learning (ICL) with multi-modal large language models (MLLMs) enables task adaptation without parameter updates, but its performance is highly sensitive to the quality and coverage of the selected demonstrations. While unlabeled multi-modal data is abundant, it remains elusive how to exploit them for ICL. We propose MAG (MAnifold-Guided semi-supervised in-context demonstration selection), an efficient framework that leverages unlabeled data to improve multi-modal ICL. MAG formulates demonstration selection as a semi-supervised propagation problem on a multi-modal graph and adopts a two-stage strategy: (i) relevance score propagation identifies a compact set of high-impact unlabeled samples for pseudo-labeling, reducing MLLM inference cost; (ii) multi-modal relevance is used to select the final demonstrations. We show that textual representations are more effective for relevance propagation, while both visual and textual modalities are crucial for high-quality demonstration selection. Experiments on eight multi-modal benchmarks demonstrate that MAG consistently outperforms strong baselines in label-scarce regimes, achieving significant gains with a limited pseudo-labeling budget.

## 1 Introduction

Large language models (LLMs) and their multi-modal extensions (MLLMs) have shown strong in-context learning (ICL) abilities, allowing adaptation to new tasks by conditioning on a small set of input–output demonstrations, e.g., visual samples, textual queries, and answer tuples, without parameter updates [Brown et al., 2020, Alayrac et al., 2022, Liu et al., 2023, OpenAI, 2023]. In multimodal settings, ICL is especially attractive, as it enables MLLMs to address diverse vision–language tasks through prompt construction alone. However, the effectiveness of few-shot multi-modal ICL critically depends on the quality, relevance, and coverage of the demonstrations, making their construction both central and challenging [Zhou et al., 2024, Wu et al., 2023, Chen et al., 2025b,a].

In practice, such high-quality input–output demonstrations, i.e., labeled data, are often scarce. For instance, new users may have limited historical interactions, or insufficient labeled data may be available to bootstrap a new downstream task. This inherent label scarcity substantially limits the effectiveness of ICL [Levy et al., 2023, Rahman et al., 2025]. In contrast, it is typically much easier to acquire large amounts of unlabeled multi-modal data, such as visual samples paired with textual queries but without answers. Yet, in the absence of ground-truth outputs, these unlabeled samples are difficult to exploit using standard retrieval-based ICL methods [Wu et al., 2023, Rubin et al., 2022]. This tension naturally motivates semi-supervised learning which has long shown that unlabeled data can significantly improve generalization by capturing intrinsic data structure that is inaccessible from limited labeled examples [Zhu et al., 2003, Tarvainen and Valpola, 2017, Iscen et al., 2019, Sohn et al.,

2020]. These insights suggest that abundant unlabeled multi-modal data, if properly utilized, could serve as a powerful resource for constructing more informative and robust in-context demonstrations.

A central challenge is how to effectively propagate information from labeled data to unlabeled data in a manner compatible with ICL. We address this challenge through a two-stage framework designed to balance scalability with demonstration quality.

First of all, given the pool of unlabeled multi-modal samples and the high inference cost of MLLMs, directly pseudolabeling all unlabeled data is too expensive. Instead, we employ a graph-based relevance score propagation mechanism [Zhu et al., 2002, Zhou et al., 2004] to identify a compact candidate set of unlabeled samples that are most relevant to the labeled data. This step propagates supervision along the intrinsic data manifold, en-

![](images/88b9ab243a9c74622d476411ba1a2fb4fae21f1b8d020fc982644dbe56fc9f4e.jpg)  
Figure 1: Comparison of our proposed semi-supervised in-context learning (ICL) against vanilla ICL.

abling efficient filtering of unlabeled samples that are likely to provide informative and transferable context. Importantly, the procedure is formulated as label propagation on the graph which is both highly computational efficient with closed-form solution and offers principled mathematical explana tion. The approach is also highly scalable to very large unlabeled data pool due to the sparsity of the manifold graph. Eventually, only selected unlabeled candidates are then forwarded to the MLLM for pseudo-labeling, reducing inference cost while preserving semantic relevance. In the second stage, we further select the most effective in-context demonstrations from the combined labeled and pseudo-labeled candidate pool. Specifically, we apply score propagation again by treating the query sample as positive node and propagate relevance score to all candidates for demonstration selection.

Critically, we reveal that relying on textual descriptions of visual content is a stable option for relevance propagation in the first stage for coarse filtering. In contrast, incorporating both visual and textual modalities is essential for high-quality demonstration selection in the second stage to ensure fine-grained alignment between the demonstrations and the query. In particular, late fusion between visual and textual modalities proves more effective in the second stage, as it better synergizes complementary modalities while decoupling modality-specific features.

To reflect the MAnifold-Guided nature of our semi-supervised in-context demonstration selection, we term the proposed method MAG. Unlike learning-based approaches [Deng et al., 2024, Li et al., 2025] that rely on fine-tuning or training auxiliary models, MAG is learning-free, enabling rapid deployment across diverse tasks. Across eight benchmarks spanning visual emotion recognition [Yang et al., 2023, Panda et al., 2018], scene text understanding [Zong et al., 2024], visual reasoning [Chen et al., 2024], and visual question answering [Hudson and Manning, 2019, Marino et al., 2019], MAG consistently outperforms methods retrieving from labeled data alone. These results demonstrate that principled incorporation of unlabeled data through relevance-guided pseudo-labeling and multi-modal graph propagation substantially improves ICL performance under label-scarce conditions, with particularly pronounced gains on tasks requiring complex cross-modal reasoning.

We summarize the contributions as follows.

• We identify label scarcity as a key bottleneck in few-shot multi-modal ICL and propose to leverage abundant unlabeled multi-modal data via a semi-supervised formulation, enabling more robust and informative demonstration construction under limited labeled data.

• We introduce a two-stage, training-free framework that employs graph-based relevance score propagation to efficiently select a compact, query-relevant subset of unlabeled samples for pseudo-labeling and ICL, substantially reducing MLLM inference cost while preserving semantic relevance.

• Across 8 diverse benchmarks, we show consistent and substantial gains with particularly large improvements on reasoning-intensive tasks, demonstrating that unlabeled data can be effectively leveraged for MAG.

## 2 Related Work

In-Context Learning (ICL): ICL enables large language models to solve new tasks by conditioning on demonstrations without parameter updates [Brown et al., 2020]. Research has explored ICL mechanisms as implicit gradient descent [Peebles et al., 2022, Garg et al., 2022] or Bayesian inference [Xie et al., 2021], while other work examines how demonstration selection, ordering, and formatting affect performance [Lu et al., 2022, Rubin et al., 2022, An et al., 2023]. Methods for improving ICL fall into learning-free and learning-based categories. Learning-based approaches optimize prompts or selection strategies through training. Examples include automatic prompt engineering [Zhou et al., 2022], gradient-based prompt optimization [Pryzant et al., 2023], and parameter-efficient adaptation [Liu et al., 2022]. CEIL [Ye et al., 2023] learns selection policies using determinant point processes. While effective, these methods require additional training or task-specific supervision, unlike our learning-free approach. Learning-free methods use heuristics for demonstration selection: TopK+MDL uses content similarity and minimum description length [Wu et al., 2023], while Cover-LS emphasizes structural similarity and diversity [Levy et al., 2023]. MAPLE [Chen et al., 2025d] introduces graph-based influence propagation to select demonstrations and incorporates unlabeled data through adaptive pseudo-labeling for text-only ICL. Unlike MAPLE, which addresses many-shot pseudo-labeling for in-context learning with large context windows, our work targets few-shot multi-modal ICL by introducing a two-stage, modality-aware score propagation that efficiently filters unlabeled data and selects high-quality demonstrations under strict context length and inference cost constraints. The introduced score propagation presents a more computational efficient and mathematically principled perspective. We also identify effective multi-modal fusion practices in the context of ICL.

Multi-Modal In-Context Learning: Vision-language models [Radford et al., 2021, Jia et al., 2021, Li and et al., 2022] established shared representation spaces, enabling multi-modal ICL with models like Flamingo [Alayrac et al., 2022], GPT-4V [OpenAI, 2023], and LLaVA [Liu et al., 2023]. However, MM-ICL performance remains highly sensitive to demonstration selection [Li et al., 2024b, Chen et al., 2025b]. Learning-free MM-ICL methods address this through improved retrieval. MMICES [Chen et al., 2025b] and VICL [Zhou et al., 2024] combine visual concept consistency with textual similarity, while ICCG [Li et al., 2024a] uses diversity-coverage matching. MPCAR [Rahman et al., 2025] incorporates multi-perspective demonstrations, and Cola [Chen et al., 2023] coordinates multiple VLMs. However, these methods operate only on labeled demonstrations and cannot leverage abundant unlabeled data, limiting their effectiveness in label-scarce settings. Learning-based methods adapt MLLMs through training. STIC [Deng et al., 2024] uses self-training for image comprehension, TACO [Li et al., 2025] employs task-aware attention, CVR-LLM [Li et al., 2024c] generates context-aware textual descriptions, and CONTEXTNA [Li et al., 2024c] proposes an agentic retrieval framework. These methods require training or complex pipelines, while our approach is learning-free and directly applicable across tasks.

Semi-Supervised Learning and Label Propagation: Label propagation diffuses label information from labeled to unlabeled samples over similarity graphs [Zhu et al., 2002, Zhou et al., 2004], assuming nearby samples share labels. Modern variants leverage deep representations: Iscen et al. [Iscen et al., 2017] proposed efficient diffusion on region manifolds, while subsequent work integrated pseudo-labeling into deep learning [Iscen et al., 2019, Basak and Yin, 2023, Zheng et al., 2023]. More expressive approaches use hypergraph structures [Zhang et al., 2020] to model higherorder relations. Chen et al. [Chen et al., 2025c] propose graph-based pseudo-labeling for text-only ICL, propagating labels over example-query relations. To the best of our knowledge, no prior work has applied label propagation paradigms to multi-modal ICL. We build on these advances by extending influence-guided pseudo-labeling to multi-modal ICL and demonstrating that visual and textual graphs capture complementary structures requiring late fusion.

## 3 Methodology

Problem Setup. We are given a labeled dataset $\mathcal { D } _ { L } = \{ ( x _ { i } , q _ { i } , a _ { i } ) \} _ { i = 1 } ^ { N _ { L } }$ and an unlabeled dataset $\mathcal { D } _ { U } = \{ ( x _ { j } , q _ { j } ) \} _ { j = 1 } ^ { N _ { U } }$ , where $x _ { i } , q _ { i }$ , and $a _ { i }$ denote the image, text (e.g., a question in VQA), and answer, respectively. In typical semi-supervised settings, $N _ { L } \ll N _ { U }$ . Our goal is to select effective in-context demonstrations for each query q under label-scarce conditions.

![](images/c62ad92ff2c8279e3cb796e2d5049abd7d43155df4dbcaa4527195d3372cd79e.jpg)  
Figure 2: Overview of MAG. Stage 1 (top): We generate textual descriptions of all images using the MLLM, construct a textual relationship graph, identify high-relevance unlabeled samples, and pseudo-label them. Stage 2 (bottom): We construct visual and textual graphs over the expanded candidate pool and perform query-conditioned relevance propagation in both modalities to select demonstrations for each query.

MAG addresses this problem through two components: (1) manifold-guided pseudo-labeling, which identifies and pseudo-labels high-relevance unlabeled samples to expand the candidate pool while controlling cost; and (2) query-conditioned multi-modal graph-based demonstration selection, which performs relevance propagation on complementary visual and textual graphs to select most relevant demonstrations from the candidate pool.

## 3.1 Manifold-Guided Pseudo-Labeling

In the first stage, we identify unlabeled samples that are strongly connected to labeled data and pseudolabel them to augment the limited labeled pool. To this end, we construct a textual relationship graph that captures semantic structure shared across labeled and unlabeled samples.

Textual Description Generation. Contrary to the practice in MAPLE Chen et al. [2025d] which directly uses question q to compute affinity, we generate textual descriptions of image using the MLLM, $d _ { i } = \mathbf { \bar { \mathcal { M } } } _ { \mathrm { d e s c } } ( x _ { i } )$ , where ${ \mathcal { M } } _ { \mathrm { d e s c } }$ produces a detailed description of the visual content. These descriptions, combined with any existing textual information (e.g., task-specific questions), form the textual representation $t _ { i } = [ d _ { i } , q _ { i } ]$ . This design ensures more visual information is captured in the construction of graph affinity and enables handling the tasks where questions are identical across samples (e.g. many VQA tasks have an identical question for different images).

Manifold Graph Construction. We construct a textual relationship graph $\mathcal { G } ^ { ( t ) } = ( \mathcal { V } ^ { ( t ) } , \mathcal { E } ^ { ( t ) } )$ over $\mathcal { D } _ { L } \cup \mathcal { D } _ { U }$ . Each node corresponds to a text embedding $h _ { i } ^ { ( t ) } = f _ { t } ( t _ { i } )$ , where $f _ { t }$ is a pre-trained text encoder (e.g., Contriver [Izacard et al., 2021]). We build a k-NN graph and symmetrize it to capture the underlying manifold, with edge weights defined as

$$
w _ { i j } ^ { ( t ) } = \left\{ \begin{array} { l l } { s _ { i j } ^ { ( t ) } , } & { \mathrm { i f ~ } j \in \mathrm { N N } _ { k } ( i ) , } \\ { 0 , } & { \mathrm { o t h e r w i s e } , } \end{array} \right. \quad s _ { i j } ^ { ( t ) } = \langle h _ { i } ^ { ( t ) } , h _ { j } ^ { ( t ) } \rangle .\tag{1}
$$

Relevance Score Propagation. To identify candidate unlabeled samples for pseudo-labeling, we propagate relevance scores from labeled nodes to unlabeled ones. We initialize an relevance score vector $I _ { ( 0 ) } ^ { ( t ) } \in \{ 0 , 1 \} ^ { N _ { L } + N _ { U } }$ by assigning value 1 to labeled samples and 0 to unlabeled samples. Relevance is iteratively propagated over $\mathcal { G } ^ { ( t ) }$ via

$$
I _ { ( k ) } ^ { ( t ) } = \alpha \hat { W } ^ { ( t ) } I _ { ( k - 1 ) } ^ { ( t ) } + ( 1 - \alpha ) I _ { ( 0 ) } ^ { ( t ) } ,\tag{2}
$$

where k indicates the propagation step, $\alpha \in [ 0 , 1 ]$ controls the propagation strength and $\hat { W } ^ { ( t ) } =$ $( D ^ { ( t ) } ) ^ { - 1 / 2 } W ^ { ( t ) } ( D ^ { ( t ) } ) ^ { - 1 / 2 }$ is the normalized adjacency matrix with $D ^ { ( t ) }$ being the degree matrix. The iteration converges to the closed-form solution

$$
\begin{array} { r } { I _ { ( \infty ) } ^ { ( t ) } = ( 1 - \alpha ) ( \mathbf { I } - \alpha \hat { W } ^ { ( t ) } ) ^ { - 1 } I _ { ( 0 ) } ^ { ( t ) } . } \end{array}\tag{3}
$$

We select the top-K unlabeled samples with the highest relevance scores and pseudo-label them using the MLLM conditioned on labeled demonstrations:

$$
\hat { a } _ { j } = \mathcal { M } ( x _ { j } , q _ { j } ; \mathcal { D } _ { L } ) .\tag{4}
$$

The pseudo-labeled set $\mathcal { D } _ { P }$ and labeled set $\mathcal { D } _ { L }$ together form the expanded candidate pool $\mathcal { D } _ { E } =$ $\mathcal { D } _ { L } \cup \mathcal { D } _ { P }$ . The above score propagation is easily scalable to very large unlabeled data pool since W is sparse which allows efficiency solution to Eq. 3, e.g. through conjugate gradient or Cholesky decomposition.

## 3.2 Multi-Modal Graph Construction & Query-Conditioned Propagation

In the second stage, we select the most relevant demonstrations for ICL from the expanded candidate pool $\mathcal { D } _ { E }$ . We construct separate relationship graphs in the visual and textual embedding spaces. These practices ensure most relevant demonstrations are selected for ICL while the inference cost is manageable.

Multi-modal graph construction & Score Propagation. For each sample in $\mathcal { D } _ { E }$ , we compute both visual and textual embeddings,

$$
\begin{array} { r } { \mathbf { \Phi } ^ { \mathsf { ^ { \prime } } } h _ { i } ^ { ( v ) } = f _ { v } ( x _ { i } ) , \quad h _ { i } ^ { ( t ) } = f _ { t } ( t _ { i } ) , } \end{array}\tag{5}
$$

where $f _ { v }$ is a visual encoder (e.g., CLIP ViT-L/14 [Radford et al., 2021]). We then construct modalityspecific graphs $\mathcal { G } ^ { ( v ) }$ and $\mathcal { G } ^ { ( t ) }$ using k-NN with cosine similarity following a similar fashion in Eq. 1. This results in two graphs featuring both visual $\mathcal { G } ^ { ( v ) }$ and textual $\mathcal { G } ^ { ( t ) }$ modalities.

Given a query $q ,$ we initialize modality-specific relevance vectors $I _ { 0 } ^ { ( m ) } ( q )$ by assigning value 1 to the k nearest neighbors of $q$ in modality $m \in \{ v , t \}$ and 0 to all other nodes. Relevance is then propagated following the closed-form solution:

$$
I _ { ( \infty ) } ^ { ( m ) } = ( 1 - \alpha ) ( { \mathbf I } - \alpha \hat { W } ^ { ( m ) } ) ^ { - 1 } I _ { ( 0 ) } ^ { ( m ) }\tag{6}
$$

Demonstration Selection and Inference. After convergence, we apply late fusion to combine the modality-specific relevance scores via,

$$
I = \beta I _ { ( \infty ) } ^ { ( v ) } + ( 1 - \beta ) I _ { ( \infty ) } ^ { ( t ) } ,\tag{7}
$$

where $\beta \in [ 0 , 1 ]$ balances the visual and textual modalities. For each query q, we rank all candidates in $\mathcal { D } _ { E }$ by the fused relevance score I and select the top-k demonstrations:

$$
\begin{array} { r } { D S ( q ) = \mathrm { T o p } { - } k \left( \{ I _ { i } \} _ { x _ { i } \in \mathcal { D } _ { E } } \right) . } \end{array}\tag{8}
$$

The selected demonstrations are then provided to the MLLM for inference:

$$
\hat { a } _ { q } = \mathcal { M } ( x _ { q } , q _ { q } ; D S ( q ) ) .\tag{9}
$$

Efficient Incremental Inference. The static graph construction described above requires attaching query samples to the graph. Without seeing all query samples, a naive way to incrementally do inference for a new query sample may need to rebuild the graph from scratch. However, for a graph with $N = | \mathcal { D } _ { E } | + 1$ nodes, this incurs additional computational cost. To enable efficient inference while still exploiting the manifold structure, we introduce a simple graph update mechanism. Given an existing modality-specific graph $\mathcal { G } ^ { ( m ) } = ( \mathcal { V } ^ { ( m ) } , \mathcal { E } ^ { ( m ) } )$ ) with adjacency matrix $W ^ { ( m ) } \in \mathbb { R } ^ { N \times N }$ we insert a new query sample $x _ { q }$ by measuring its similarity to all existing nodes:

$$
s _ { q i } ^ { ( m ) } = \langle h _ { q } ^ { ( m ) } , h _ { i } ^ { ( m ) } \rangle , \quad \forall i \in \{ 1 , \dots , | \mathcal { D } _ { E } | \} .\tag{10}
$$

Rather than recomputing k-nearest neighbors for all nodes, we only remove the previous query sample from the graph and create edges for the top K nodes and connect them to the new query sample. This operation only requires calculating the inner product between query and $| \mathcal { D } _ { E } |$ samples when each new query arrives.

## 4 Experiments

## 4.1 Experimental Setup

Datasets. We evaluate on eight benchmarks spanning visual emotion recognition, scene text understanding, visual reasoning, and visual question answering. These tasks require diverse visual-textual understanding capabilities, from recognizing subtle emotional cues to complex spatial reasoning. Visual Emotion Recognition: EmoSet [Yang et al., 2023] contains images annotated with eight emotion categories for emotion classification, while Emotion6 [Panda et al., 2018] provides six basic emotion labels collected via controlled user studies. Scene Text Understanding: TextOCR [Zong et al., 2024] evaluates text detection and recognition in natural scene images. Visual Reasoning: MMStar [Chen et al., 2024] contains 1,500 challenge samples testing multi-modal reasoning with reduced language bias. MatchingMI [Zong et al., 2024] requires cross-image matching and correspon dence. CLEVR [Zong et al., 2024] is a synthetic diagnostic benchmark for compositional reasoning including counting, comparison, and logical operations. Visual Question Answering: GQA [Hudson and Manning, 2019] features compositional questions grounded in scene graphs. OK-VQA [Marino et al., 2019] evaluates open-domain VQA requiring external commonsense knowledge. For all datasets, we report accuracy as an evaluation metric.

Competing Methods. We compare against seven baselines spanning both zero-shot approaches and few-shot retrieval methods. Few-shot methods: Few-shot Brown et al. [2020] randomly samples labeled demonstrations. VICL [Zhou et al., 2024] uses retrieval and reranking with language-based demonstrations. Top-K+MDL [Wu et al., 2023] ranks candidates by content similarity and minimum description length. MMICES [Chen et al., 2025b] performs two-stage selection via visual concept consistency and textual similarity. CVR-LLM [Li et al., 2024c] generates context-aware visual descriptions through iterative refinement and conducts demonstration selection using multi-modal features. MAPLE [Chen et al., 2025d] performs pseudo-labeling on unlabeled data and selects demonstrations from a graph constructed using textual questions only. Zero-shot methods: Zeroshot uses task instructions without demonstrations. Cola-Zero [Chen et al., 2023] coordinates multiple VLMs via an LLM.

## 4.2 Implementation Details

ICL Configurations. For each dataset, we randomly sample 15 labeled examples, 585 unlabeled examples as the candidate pool, and 300 test samples. From the unlabeled pool, we pseudo-label the top 45 high-relevance samples identified in Stage 1, forming an expanded candidate pool of 60 samples $( | \mathcal { D } _ { E } | = 1 5 + 4 5 )$ ). For few-shot methods, both MAPLE and MAG select 10 demonstrations from 15 labeled + 585 unlabeled examples while all others select 10 demonstrations from 15 labeled ones.

Text Description Generation. For all samples, we generate image descriptions using Gemini-2.0- Flash [Comanici et al., 2025] with the following prompt: “Generate a description for the image, taking the question as guidance, elucidating both the visual content and the underlying purpose or intention depicted. Craft a clear and concise description that integrates detailsfrom the image, highlighting visual cues and semantic meaning which may be useful to get the answer.” These descriptions are used to construct textual representations in both Stage 1 and Stage 2.

Models and hyperparameters. We adopt Gemini-2.0-Flash as the base MLLM. Contriever [Izacard et al., 2021] is used for text encoding $( f _ { t } )$ , and CLIP ViT-L/14 [Radford et al., 2021] for visual encoding (f<sub>v</sub>). To investigate the generalization of method, we also evaluate against alternative MLLMs, including Gemini-3.0-Flash, GPT-4o, GPT-4.1, GPT-5 and Qwen models in the appendix. For graph construction and propagation, we set the number of nearest neighbors to $k _ { n n } = 5 .$ , the propagation strength to $\alpha = 0 . 9 ,$ and the modality fusion weight to $\beta = 0 . 5$ . These hyperparameters are fixed across all experiments for consistency.

## 4.3 Quantitave Evaluations

Table 1 compares MAG against baselines under label-scarce conditions, reported with average and standard deviation over 3 random runs. We make the following observations from the results.

Unlabeled data provides substantial gains under label scarcity. MAG consistently outperforms all baselines across eight benchmarks using only 15 labeled samples. Compared with the strongest retrieval baseline (MMICES), MAG achieves notable improvements on challenging tasks, including +30.5% on CLEVR, +7.1% on MMStar, and +25.1% on TextOCR. Relative to standard few-shot prompting, MAG further improves TextOCR (+23.8%), CLEVR (+30.8%), and EmoSet (+15.3%). These results demonstrate the effectiveness of leveraging unlabeled data through relevance-guided pseudo-labeling and graph propagation.

Table 1: Main results on multi-modal benchmarks under label-scarce settings with Gemini-2.0-Flash (mean accuracy ± standard dev % across 3 runs).
<table><tr><td>METHOD</td><td>EMOSET</td><td>EMOTION6</td><td>TEXTOCR</td><td>MMSTAR</td><td>CLEVR</td><td>MATCHINGMI</td><td>GQA</td><td>OKVQA</td></tr><tr><td>Zero-shot Methods</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>ZERO-SHOT</td><td>60.1±4.2</td><td>60.6±0.6</td><td>56.1±1.2</td><td>23.9±2.9</td><td>59.8±2.5</td><td>73.2±3.0</td><td></td><td>57.6±0.5 55.2±1.1</td></tr><tr><td>COLA-ZERO CHEN ET AL. [2023]</td><td>47.7±2.3</td><td>49.4±0.5</td><td>25.0±1.0</td><td>34.2±2.5</td><td>24.9±1.6</td><td>18.2±3.3</td><td>51.4±2.8</td><td>49.6±1.3</td></tr><tr><td>Few-shot Methods</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>FEW-SHOT BROWN ET AL. [2020]</td><td>61.0±2.9</td><td>63.0±1.9</td><td>67.6±3.9</td><td>65.8±4.8</td><td>61.3±3.9</td><td>96.7±0.7</td><td></td><td>54.2±0.7 54.8±2.3</td></tr><tr><td>VICL ZHOU ET AL. [2024]</td><td>62.6±1.6</td><td>61.5±4.5</td><td>67.0±3.7</td><td>64.1±3.1</td><td>50.6±1.5</td><td>92.0±3.7</td><td></td><td>55.1±1.0 56.3±1.5</td></tr><tr><td>TOP-K+MDL WU ET AL. [2023]</td><td>59.6±1.3</td><td>62.8±1.7</td><td>64.3±0.9</td><td>62.3±2.0</td><td>57.8±2.4</td><td>94.1±0.8</td><td></td><td>56.2±1.1 54.7±1.4</td></tr><tr><td>MMICES CHEN ET AL. [2025B])</td><td>60.4±3.7</td><td>62.9±0.9</td><td>66.3±1.4</td><td>60.8±2.6</td><td>61.6±2.2</td><td>94.2±1.5</td><td></td><td>53.1±1.3 54.9±1.8</td></tr><tr><td>CVR-LLM LI ET AL. [2024C]</td><td>74.8±1.5</td><td>62.5±2.3</td><td>87.9±1.2</td><td>64.4±2.5</td><td>90.2±1.8</td><td>97.6±0.9</td><td></td><td>52.3±1.4 46.7±2.0</td></tr><tr><td>MAPLE CHEN ET AL. [2025D]</td><td>72.4±1.2</td><td>61.3±1.3</td><td>84.0±0.8</td><td>64.8±1.7</td><td>89.3±0.6</td><td>96.2±0.4</td><td></td><td>55.7±2.1 54.3±2.3</td></tr><tr><td>MAG (OURS)</td><td>76.3±1.4</td><td>64.3±1.4</td><td>91.4±0.6</td><td>67.9±2.1</td><td>92.1±0.5</td><td>99.8±0.2</td><td></td><td>59.0±3.9 57.6±2.1</td></tr></table>

Graph-based propagation outperforms similarity-based retrieval. Compared with similaritybased retrieval methods such as VICL and Top-K+MDL, MAG consistently achieves stronger performance by modeling global relational structure among samples. For instance, MAG surpasses Top-K+MDL by +34.3% on CLEVR and +27.1% on TextOCR, validating that graph connectivity provides more reliable supervision than local similarity alone under limited labeled data.

Our method complements description-based approaches. Although CVR-LLM achieves competitive performance through iterative description refinement, MAG still improves results on most benchmarks, including +3.5% on EmoSet, +3.5% on TextOCR, +6.7% on GQA, and +10.9% on OKVQA. Unlike CVR-LLM, which shows unstable cross-task generalization, MAG maintains consistently strong performance, suggesting that pseudo-label expansion and graph propagation provide a more robust way to exploit unlabeled data.

Stronger performance than single modality manifold method. Compared with the single modality manifold based method (MAPLE), we observe consistent improvement across all datasets. Despite both methods selecting demonstrations from the same labeled + unlabeled examples, our method excels by exploiting multiple modalities and principled selection strategy.

Overall, these results confirm that MAG enables effective multi-modal ICL by incorporating unlabeled data, achieving strong and stable performance. Qualitative examples demonstrating improved demonstration selection are provided in Appendix C.

## 4.4 Ablation Study

We conduct ablation studies to validate the key components of MAG. Table 2 evaluates three design choices: pseudo-labeling strategy, demonstration selection method, and modality usage. We report mean and standard deviation across 3 runs and draw the following observations.

Manifold-guided pseudo-labeling is critical, and naive pseudo-labeling can be harmful. Comparing three strategies, we observe a clear trend: (1) removing pseudo-labeling yields 72.0% on EmoSet, (2) adding randomly selected pseudo-labels provides only marginal gains on EmoSet (73.1%) and even degrades performance on some tasks (e.g., GQA: 54.7% vs. 56.4%), while (3) our manifoldguided pseudo-labeling significantly improves performance across all benchmarks (76.3% on EmoSet, 91.4% on TextOCR, 67.9% on MMStar, 58.0% on GQA). This indicates that pseudo-label quality, rather than quantity, is the determining factor. Naively introducing pseudo-labels can inject noise and offset potential gains, whereas our relevance-guided selection effectively identifies high-value samples, leading to consistent improvements.

Graph-based demonstration selection consistently outperforms heuristic strategies. Compared to random selection, graph-based selection yields substantial gains (+3.5% on EmoSet, +2.5% on TextOCR, +1.8% on GQA). Replacing graph propagation with TopK similarity leads to smaller improvements and consistently underperforms the full method. This suggests that selecting demonstrations purely based on query-example similarity is insufficient. Instead, graph propagation captures higher-order relationships among samples, enabling retrieval of globally relevant and representative demonstrations. This advantage becomes more pronounced in low-resource settings where labeled examples are sparse and potentially biased.

Both modalities are necessary and complementary. Removing either modality results in significant performance degradation. Text-only graphs (–img) particularly hurt visually grounded reasoning (GQA: 44.2% vs. 58.0%), while visual-only graphs (–desc) fail on semantically intensive tasks (EmoSet: 60.2% vs. 76.3%, TextOCR: 67.0% vs. 91.4%). Notably, visual-only selection introduces misleading context due to superficial visual similarity, leading to large drops on language-heavy tasks. These results highlight that visual and textual modalities encode complementary structures, and effective demonstration selection requires jointly modeling both.

Table 2: Ablation study. We evaluate the contribution of (1) relevance-guided pseudo-labeling (MAPLE vs. Random vs. None), (2) graph-based propagation (Graph vs. TopK vs. Random), and (3) multi-modal representations (with/without images and descriptions).
<table><tr><td>METHOD VARIANT</td><td>PSEUDO-LABEL</td><td>DEMO. SELECTION</td><td>EMOSET</td><td>TEXTOCR MMSTAR</td><td></td><td>GQA</td></tr><tr><td colspan="7">Ablating pseudo-labeling strategy</td></tr><tr><td>OURS (-PSEUDO)</td><td></td><td>MMGRAPH</td><td>72.0±0.3</td><td>88.1±2.7</td><td></td><td>66.5±0.7 56.4±1.4</td></tr><tr><td>OURS (+RAND)</td><td>RANDOM</td><td>MMGRAPH</td><td>73.1±2.7</td><td>88.0±3.6</td><td></td><td>66.4±1.6 54.7±2.1</td></tr><tr><td>OURS (FULL)</td><td>MANIFOLD</td><td>MMGRAPH</td><td>76.3±1.4</td><td>91.4±0.6</td><td></td><td>67.9±2.1 58.0±3.9</td></tr><tr><td colspan="7">Ablating demonstration selection</td></tr><tr><td>OURS (RAND)</td><td>MANIFOLD</td><td>RANDOM</td><td>72.8±2.0</td><td>88.9±1.9</td><td></td><td>65.1±4.3 56.2±1.9</td></tr><tr><td>OURS (TOPK)</td><td>MANIFOLD</td><td>TOPK</td><td>74.1±1.9</td><td>89.0±1.0</td><td></td><td>67.6±3.1 56.1±1.9</td></tr><tr><td>OURS (FULL)</td><td>MANIFOLD</td><td>MMGRAPH</td><td>76.3±1.4</td><td>91.4±0.6</td><td>67.9±2.1</td><td>58.0±3.9</td></tr><tr><td colspan="7">Ablating modalities</td></tr><tr><td>OURS (-IMG)</td><td>MANIFOLD</td><td>GRAPH (TEXT ONLY)</td><td>75.2±1.0</td><td>90.4±0.5</td><td></td><td>67.4±2.5 44.2±2.6</td></tr><tr><td>OURS (-DESC)</td><td>MANIFOLD</td><td>GRAPH (VISUAL ONLY)</td><td>60.2±3.6</td><td>67.0±2.4</td><td></td><td>65.9±2.9 52.4±1.4</td></tr><tr><td>OURS (FULL)</td><td>MANIFOLD</td><td>MMGRAPH (BOTH)</td><td>76.3±1.4</td><td>91.4±0.6</td><td>67.9±2.1 58.0±3.9</td><td></td></tr></table>

## 4.5 Modality Analysis for Stage 1

We further evaluate MAG using different modality representations for Stage 1 pseudo-label selection.   
Table 3 shows results for textual-only, visual-only, and combined representations.

Textual representations provide robust cross-task performance. Textual-only achieves best performance on 6 of 8 datasets, particularly excelling on emotion recognition (EmoSet: 77.0%, Emotion6: 65.0%), scene text understanding (TextOCR: 91.7%), and visual reasoning (MMStar: 70.3%, GQA: 62.3%). This suggests that MLLM-generated image descriptions capture semantic concepts that are well-aligned with these task requirements.

Visual representations excel slightly on spatial reasoning. Visual-only achieves marginally higher accuracy on spatially-intensive tasks (CLEVR: 92.3% vs. 92.0%, OKVQA: 58.0% vs. 56.7%), where geometric relationships are critical. However, the performance gap is small (+0.3% to +1.3%).

Choose textual representation only for Stage 1. Given that textual representations provide strong and stable performance across diverse task types, we use textual-only for Stage 1 in all experiments. Importantly, even on datasets where visual representations provide small gains in Stage 1 isolation, our full method with textual Stage 1 still substantially outperforms all baselines (e.g., 92.0% vs. 90.3% baseline on CLEVR, 56.7% vs. 54.0% baseline on OKVQA in Table 1). This demonstrates that Stage 2’s multi-modal propagation effectively compensates for Stage 1 modality choice, as it incorporates both visual and textual information through complementary graphs.

Table 3: Performance using different Stage 1 modalities for pseudo-label selection (accuracy %). Textual representations provide robust performance across tasks. Visual representations achieve slightly higher accuracy on spatially-intensive tasks (CLEVR, OKVQA) but the gap is small (+0.3% to +1.3%). We use textual representations as default for consistency.
<table><tr><td>STAGE 1 MODALITY</td><td></td><td>EMOSET EMOTION6</td><td></td><td></td><td></td><td>TEXTOCR MMSTAR CLEVR MATCHINGMI GQA OKVQA</td><td></td><td></td></tr><tr><td>TEXTUAL ONLY</td><td>77.0</td><td>65.0</td><td>91.7</td><td>70.3</td><td>92.0</td><td>99.7</td><td>62.3</td><td>56.7</td></tr><tr><td>VISUAL ONLY</td><td>75.0</td><td>60.7</td><td>85.3</td><td>62.3</td><td>92.3</td><td>99.7</td><td>57.3</td><td>58.0</td></tr><tr><td>TEXTUAL &amp; VISUAL</td><td>72.7</td><td>59.7</td><td>87.0</td><td>61.7</td><td>91.3</td><td>99.7</td><td>57.7</td><td>54.7</td></tr></table>

## 4.6 Multi-Modal Fusion Strategy

We compare three modality fusion strategies at Stage 2 for computing relevance score: early fusion (concatenate the textual embeddings and visual embeddings before graph construction), mid fusion (utilize the weighted sum of adjacency matrix for relevance propagation), and late fusion (do relevance propagation of two modalities separately, then average these two relevance scores). Table 4 shows that late fusion consistently achieves the best performance across all benchmarks, with particularly large gains on reasoning-intensive tasks (MMStar: +6.6% over early fusion, GQA: +2.3%). Early fusion performs worst, suggesting that directly mixing heterogeneous visual and textual features before relational modeling weakens modality-specific structure learning. Late fusion preserves complementary relational signals by allowing each modality’s graph to capture its own semantic structure before combining scores, which is particularly beneficial for tasks requiring cross-modal reasoning.

![](images/fa517b1d9c359d8eb61fbc0570a1d5fcfdd1ba2e740731b62390a5ae7122d912.jpg)  
Figure 3: Sensitivity analysis across three datasets (EmoSet, Emotion6, MMStar). Performance is stable across wide ranges of α, $\beta , k _ { n n } .$ , pool size, and label rate, with clear optimal configurations: $\alpha \approx 0 . 9 , k _ { n n } = 5 , \beta = 0 . 5 ,$ , pool size of 60, label rate of 25%, and 10 demonstrations.

Table 4: Performance using different fusion strategies for combining visual and textual relevance scores (accuracy %). Late fusion consistently outperforms early and mid fusion by preserving modality-specific graph structures during propagation.
<table><tr><td>FUSION STRATEGY EMOSET EMOTION6 TEXTOCR MMSTAR CLEVR MATCHINGMI</td><td></td><td></td><td></td><td></td><td></td><td></td><td>GQA</td><td>OKVQA</td></tr><tr><td>EARLY FUSION</td><td>76.0</td><td>60.0</td><td>86.3</td><td>63.7</td><td>89.3</td><td>99.3</td><td>60.0</td><td>53.7</td></tr><tr><td>MID FUSION</td><td>77.0</td><td>60.7</td><td>88.0</td><td>64.3</td><td>91.0</td><td>99.3</td><td>56.7</td><td>52.0</td></tr><tr><td>LATE FUSION</td><td>77.0</td><td>65.0</td><td>91.7</td><td>70.3</td><td>92.0</td><td>99.7</td><td>62.3</td><td>56.7</td></tr></table>

## 4.7 Hyperparameter Sensitivity

We evaluate MAG’s sensitivity to key hyperparameters. Figure 3 shows results across EmoSet, Emotion6, and MMStar for (a) propagation strength (α), (b) modality fusion weight (β), (c) nearest neighbors $( k _ { n n } ) .$ , (d) candidate pool size $( | \mathcal { D } _ { E } | )$ , (e) label rate within candidate pool size $( | \mathcal { D } _ { L } | / | \mathcal { D } _ { E } | )$ and (f) number of demonstrations $( | D S ( q ) | )$ ). Performance remains stable across wide ranges for most parameters, with clear optimal configurations emerging: α ≈ 0.9 (emphasizing propagation over initialization), $k _ { n n } = 5$ (moderate neighborhood size), $\beta = 0 . 5$ (balanced modality fusion), pool size of 60 (15 labeled + 45 pseudo-labeled), label rate of $2 5 \%$ (demonstrating label efficiency), and 10 demonstrations (beyond which performance plateaus or degrades). Notably, the label rate analysis shows that performance peaks at only 25% labeled data, validating that our relevance-guided pseudolabeling effectively leverages unlabeled samples. The framework demonstrates strong robustness to hyperparameter choices while maintaining consistent optimal configurations across diverse tasks.

## 5 Conclusion

We introduced MAG, a semi-supervised framework for multi-modal in-context learning that effectively leverages unlabeled data under label-scarce settings. By modeling demonstration selection as relevance propagation on multi-modal graphs, MAG identifies a compact set of high-impact unlabeled samples for pseudo-labeling and selects query-conditioned demonstrations that are globally relevant across visual and textual modalities. This design enables efficient use of unlabeled data while respecting inference-time and context-length constraints. Extensive experiments across eight diverse multi-modal benchmarks show that MAG consistently outperforms strong few-shot and semi-supervised baselines, achieving substantial gains with a limited pseudo-labeling budget. Our analysis highlights the complementary roles of textual and visual manifolds in ICL and demonstrates that principled graph-based selection is key to scalable multi-modal prompting. We hope MAG provides a practical foundation for exploiting unlabeled data in future multi-modal ICL systems.

## References

Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al. Flamingo: a visual language model for few-shot learning. In Advances in Neural Information Processing Systems, 2022.

Shengnan An, Zeqi Lin, Qiang Fu, Bei Chen, Nanning Zheng, Jian-Guang Lou, and Dongmei Zhang. How do in-context examples affect compositional generalization? In Annual Meeting of the Association for Computational Linguistics, 2023.

Hritam Basak and Zhaozheng Yin. Pseudo-label guided contrastive learning for semi-supervised medical image segmentation. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023.

Tom Brown, Benjamin Mann, Nick Ryder, et al. Language models are few-shot learners. In Advances in Neural Information Processing Systems, 2020.

Cheng Chen, Yunpeng Zhai, Yifan Zhao, Jinyang Gao, Bolin Ding, and Jia Li. Provoking multimodal few-shot lvlm via exploration-exploitation in-context learning. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025a.

Liangyu Chen, Bo Li, Sheng Shen, Jingkang Yang, Chunyuan Li, Kurt Keutzer, Trevor Darrell, and Ziwei Liu. Large language models are visual reasoning coordinators. In Advances in Neural Information Processing Systems, 2023.

Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Jiaqi Wang, Yu Qiao, Dahua Lin, et al. Are we on the right way for evaluating large vision-language models? In Advances in Neural Information Processing Systems, 2024.

Shuo Chen, Zhen Han, Bailan He, Jianzhe Liu, Mark Buckley, Yao Qin, Philip Torr, Volker Tresp, and Jindong Gu. Can multimodal large language models truly perform multimodal in-context learning? In IEEE/CVF Winter Conference on Applications of Computer Vision, 2025b.

Zihan Chen, Song Wang, Xingbo Fu, Chengshuai Shi, Zhenyu Lei, Cong Shen, and Jundong Li. From cross-task examples to in-task prompts: A graph-based pseudo-labeling framework for in-context learning. In Conference on Empirical Methods in Natural Language Processing, 2025c.

Zihan Chen, Song Wang, Zhen Tan, Jundong Li, and Cong Shen. Maple: Many-shot adaptive pseudo-labeling for in-context learning. In International Conference on Machine Learning, 2025d.

Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025.

Yihe Deng, Pan Lu, Fan Yin, Ziniu Hu, Sheng Shen, Quanquan Gu, James Y Zou, Kai-Wei Chang, and Wei Wang. Enhancing large vision language models with self-training on image comprehension. Advances in Neural Information Processing Systems, 2024.

Shivam Garg, Dimitris Tsipras, Percy S Liang, and Gregory Valiant. What can transformers learn in-context? a case study of simple function classes. In Advances in Neural Information Processing Systems, 2022.

Drew A Hudson and Christopher D Manning. Gqa: A new dataset for real-world visual reasoning and compositional question answering. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2019.

Ahmet Iscen, Giorgos Tolias, Yannis Avrithis, Teddy Furon, and Ondrej Chum. Efficient diffusion on region manifolds: Recovering small objects with compact cnn representations. In IEEE conference on computer vision and pattern recognition, 2017.

Ahmet Iscen, Giorgos Tolias, Yannis Avrithis, and Ondrej Chum. Label propagation for deep semisupervised learning. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2019.

Gautier Izacard, Mathilde Caron, Lucas Hosseini, Sebastian Riedel, Piotr Bojanowski, Armand Joulin, and Edouard Grave. Unsupervised dense information retrieval with contrastive learning. arXiv preprint arXiv:2112.09118, 2021.

Chao Jia, Yinfei Yang, Ye Xia, Yi-Ting Chen, Zarana Parekh, Hieu Pham, Quoc Le, Yun-Hsuan Sung, Zhen Li, and Tom Duerig. Scaling up visual and vision-language representation learning with noisy text supervision. In International Conference on Machine Learning, 2021.

Itay Levy, Ben Bogin, and Jonathan Berant. Diverse demonstrations improve in-context compositional generalization. In Annual Meeting of the Association for Computational Linguistics, 2023.

Chuanhao Li, Chenchen Jing, Zhen Li, Mingliang Zhai, Yuwei Wu, and Yunde Jia. In-context compositional generalization for large vision-language models. In Conference on Empirical Methods in Natural Language Processing, 2024a.

Junnan Li and et al. Blip: Bootstrapped language-image pre-training for unified vision-language understanding and generation. In International Conference on Machine Learning, 2022.

Li Li, Jiawei Peng, Huiyi Chen, Chongyang Gao, and Xu Yang. How to configure good in-context sequence for visual question answering. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024b.

Yanshu Li, Jianjiang Yang, Tian Yun, Pinyuan Feng, Jinfa Huang, and Ruixiang Tang. Taco: Enhancing multimodal in-context learning via task mapping-guided sequence configuration. In Conference on Empirical Methods in Natural Language Processing, 2025.

Zhiyuan Li, Dongnan Liu, Chaoyi Zhang, Heng Wang, Tengfei Xue, and Weidong Cai. Enhancing advanced visual reasoning ability of large language models. In Conference on Empirical Methods in Natural Language Processing, 2024c.

Haokun Liu, Derek Tam, Mohammed Muqeeth, Jay Mohta, Tenghao Huang, Mohit Bansal, and Colin A Raffel. Few-shot parameter-efficient fine-tuning is better and cheaper than in-context learning. In Advances in Neural Information Processing Systems, 2022.

Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. In Advances in Neural Information Processing Systems, 2023.

Yao Lu, Max Bartolo, Alastair Moore, Sebastian Riedel, and Pontus Stenetorp. Fantastically ordered prompts and where to find them: Overcoming few-shot prompt order sensitivity. In Annual Meeting of the Association for Computational Linguistics, 2022.

Kenneth Marino, Mohammad Rastegari, Ali Farhadi, and Roozbeh Mottaghi. Ok-vqa: A visual question answering benchmark requiring external knowledge. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2019.

OpenAI. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.

Rameswar Panda, Jianming Zhang, Haoxiang Li, Joon-Young Lee, Xin Lu, and Amit K Roy-Chowdhury. Contemplating visual emotions: Understanding and overcoming dataset bias. In European Conference on Computer Vision, 2018.

William Peebles, Ilija Radosavovic, Tim Brooks, Alexei A Efros, and Jitendra Malik. Learning to learn with generative models of neural network checkpoints. arXiv preprint arXiv:2209.12892, 2022.

Reid Pryzant, Dan Iter, Jerry Li, Yin Tat Lee, Chenguang Zhu, and Michael Zeng. Automatic prompt optimization with" gradient descent" and beam search. In Annual Meeting of the Association for Computational Linguistics, 2023.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International Conference on Machine Learning, 2021.

Amirul Rahman, Qiang Xu, and Xueying Huang. Mpcar: Multi-perspective contextual augmentation for enhanced visual reasoning in large vision-language models. arXiv preprint arXiv:2508.12400, 2025.

Ohad Rubin, Jonathan Herzig, and Jonathan Berant. Learning to retrieve prompts for in-context learning. In Annual Conference ofthe North American Chapter ofthe Associationfor Computational Linguistics, 2022.

Kihyuk Sohn, David Berthelot, Nicholas Carlini, Zizhao Zhang, Han Zhang, Colin A Raffel, Ekin Dogus Cubuk, Alexey Kurakin, and Chun-Liang Li. Fixmatch: Simplifying semi-supervised learning with consistency and confidence. In Advances in Neural Information Processing Systems, 2020.

Antti Tarvainen and Harri Valpola. Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results. In Advances in Neural Information Processing Systems, 2017.

Zhiyong Wu, Yaoxiang Wang, Jiacheng Ye, and Lingpeng Kong. Self-adaptive in-context learning: An information compression perspective for in-context example selection and ordering. In Annual Meeting ofthe Associationfor Computational Linguistics, 2023.

Sang Michael Xie, Aditi Raghunathan, Percy Liang, and Tengyu Ma. An explanation of in-context learning as implicit bayesian inference. arXiv preprint arXiv:2111.02080, 2021.

Jingyuan Yang, Qirui Huang, Tingting Ding, Dani Lischinski, Danny Cohen-Or, and Hui Huang. A large-scale visual emotion dataset with rich attributes. In IEEE/CVF International Conference on Computer Vision, 2023.

Jiacheng Ye, Zhiyong Wu, Jiangtao Feng, Tao Yu, and Lingpeng Kong. Compositional exemplars for in-context learning. In International Conference on Machine Learning, 2023.

Yubo Zhang, Nan Wang, Yufeng Chen, Changqing Zou, Hai Wan, Xinbin Zhao, and Yue Gao. Hypergraph label propagation network. In AAAI Conference on Artificial Intelligence, 2020.

Mingkai Zheng, Shan You, Lang Huang, Chen Luo, Fei Wang, Chen Qian, and Chang Xu. Simmatchv2: Semi-supervised learning with graph consistency. In IEEE/CVF International Conference on Computer Vision, 2023.

Dengyong Zhou, Olivier Bousquet, Thomas Lal, Jason Weston, and Bernhard Schölkopf. Learning with local and global consistency. In Advances in Neural Information Processing Systems, 2004.

Yongchao Zhou, Andrei Ioan Muresanu, Ziwen Han, Keiran Paster, Silviu Pitis, Harris Chan, and Jimmy Ba. Large language models are human-level prompt engineers. In International Conference on Learning Representations, 2022.

Yucheng Zhou, Xiang Li, Qianning Wang, and Jianbing Shen. Visual in-context learning for large vision-language models. arXiv preprint arXiv:2402.11574, 2024.

Xiaojin Zhu, Zoubin Ghahramani, and John Lafferty. Learning from labeled and unlabeled data with label propagation. In International Conference on Machine Learning, 2002.

Xiaojin Zhu, Zoubin Ghahramani, and John D Lafferty. Semi-supervised learning using gaussian fields and harmonic functions. In International Conference on Machine Learning, 2003.

Yongshuo Zong, Ondrej Bohdal, and Timothy Hospedales. Vl-icl bench: The devil in the details of multimodal in-context learning. arXiv preprint arXiv:2403.13164, 2024.

## Appendix

## A Algorithm Psuedo Code

Algorithm 1 summarizes the overall procedure of MAG, including the generation of the construction of the expanded pool, the final query-conditioned selection and prediction process.

Algorithm 1 MAG   
1: Input: Labeled data $\mathcal { D } _ { L }$ , unlabeled data $\mathcal { D } _ { U }$ , query $x _ { q } , q _ { q }$   
2: Output: Predicted answer $\hat { a } _ { q }$   
3:   
4: // Generate textual descriptions for all images   
5: for $x _ { i } \in \mathcal { D } _ { L } \cup \mathcal { D } _ { U }$ do   
6: $d _ { i } \gets { \mathcal { M } } _ { \mathrm { d e s c } } ( x _ { i } )$ // Generate image description   
7: end for   
8:   
9: // Stage 1: Relevance-guided pseudo-labeling   
10: Build textual graph $\mathcal { G } ^ { ( \aleph ) }$ over $\mathsf { \bar { D } } _ { L } \cup \mathcal { D } _ { U }$ using text embeddings   
11: Compute relevance scores $I _ { j }$ for $x _ { j } \in \mathcal { D } _ { U }$ via Eq. (4)   
12: for $x _ { j } \in$ Top-K $\left( \{ I _ { j } \} _ { x _ { j } \in \mathcal { D } _ { U } } \right)$ do   
13: Pseudo-label: $\dot { \hat { a } } _ { j } \gets \dot { \mathcal { M } } ( x _ { j } , q _ { j } ; \mathcal { D } _ { L } )$   
14: Add $( x _ { j } , \hat { a } _ { j } )$ to $\mathsf { \Pi } _ { \mathcal { D } _ { P } } ^ { }$   
15: end for   
16: Expand pool: $\mathcal { D } _ { E }  \mathcal { D } _ { L } \cup \mathcal { D } _ { P }$   
17:   
18: // Stage 2: Multi-modal query-conditioned selection   
19: Build visual graph $\mathcal { G } ^ { ( v ) }$ (image embeddings) and textual graph $\mathcal { G } ^ { ( t ) }$ (text embeddings) over $\mathcal { D } _ { E }$   
20: Propagate score $I _ { ( \infty ) } ^ { ( m ) }$ via Eq (7)   
21: Late Fusion: $I  \dot { \beta I ^ { ( v ) } } + ( 1 - \beta ) I ^ { ( t ) }$   
22: Select: ${ D S } ( q ) \gets \mathrm { T o p } { - k } \left( \{ I _ { i } \} _ { i = 1 } ^ { | \mathcal { D } _ { E } | } \right)$   
23: Predict: $\hat { a } _ { q } \gets \mathcal { M } ( x _ { q } , q _ { q } ; \mathit { D S } ( q ) )$   
24: return $\hat { a } _ { q }$

## B Additional Experimental Analysis

## B.1 Multi-Modal Backbone Analysis

We evaluate MAG across seven state-of-the-art multi-modal foundation models to assess performance variation with backbone choice. Table 5 reports results across all eight benchmarks.

Table 5: Performance of MAG with different multi-modal backbones (accuracy %). Gemini-2.0-Flash provides the most balanced performance across task types.
<table><tr><td>MODEL</td><td>EMOSET</td><td>EMOTION6</td><td>TEXTOCR</td><td>MMSTAR</td><td></td><td>CLEVR MATCHINGMI</td><td>GQA</td><td>OKVQA</td></tr><tr><td>GEMINI-2.0-FLASH</td><td>77.0</td><td>65.0</td><td>91.7</td><td>70.3</td><td>92.0</td><td>99.7</td><td>62.3</td><td>56.7</td></tr><tr><td>GEMINI-3.0-FLASH</td><td>84.7</td><td>59.3</td><td>55.7</td><td>92.3</td><td>96.0</td><td>87.3</td><td>68.0</td><td>57.3</td></tr><tr><td>GPT-40</td><td>73.7</td><td>59.3</td><td>34.7</td><td>19.3</td><td>91.3</td><td>89.3</td><td>26.7</td><td>42.0</td></tr><tr><td>GPT-4.1</td><td>76.0</td><td>60.7</td><td>50.3</td><td>33.3</td><td>93.7</td><td>79.3</td><td>37.3</td><td>52.0</td></tr><tr><td>GPT-5</td><td>78.7</td><td>60.3</td><td>89.0</td><td>45.7</td><td>96.3</td><td>99.7</td><td>71.3</td><td>68.3</td></tr><tr><td>QWEN-VL-PLUS</td><td>71.7</td><td>23.0</td><td>89.3</td><td>30.0</td><td>76.0</td><td>96.3</td><td>68.0</td><td>61.3</td></tr><tr><td>QWEN-VL-MAX</td><td>73.7</td><td>60.0</td><td>79.7</td><td>50.0</td><td>97.3</td><td>90.0</td><td>71.0</td><td>64.0</td></tr></table>

Overall trends. We observe substantial performance variance across backbones, indicating that MAG is sensitive to the underlying multi-modal reasoning and perception capabilities. While all models benefit from MAG, their relative strengths differ significantly across task types.

Balanced vs. specialized performance. Gemini-2.0-Flash achieves the most balanced performance, maintaining consistently strong results across both perception-heavy tasks (e.g., TextOCR, CLEVR) and reasoning-intensive benchmarks (e.g., GQA, OKVQA). In contrast, Gemini-3.0-Flash exhibits more polarized behavior, achieving state-of-the-art results on MMStar and CLEVR, but suffering a notable drop on TextOCR, suggesting potential trade-offs between visual reasoning and text recognition.

Effect of backbone capability. Stronger general-purpose models such as GPT-5 demonstrate clear advantages on high-level reasoning tasks (e.g., GQA, OKVQA), while also maintaining competitive performance on structured reasoning benchmarks like CLEVR and MatchingMI. However, earlier generations (e.g., GPT-4o, GPT-4.1) show clear limitations on multi-modal understanding tasks such as MMStar and TextOCR, highlighting the importance of improved visual-text alignment in newer models.

Open-source vs. proprietary models. Qwen-VL variants exhibit competitive performance on several benchmarks, particularly MatchingMI and GQA, but show instability on tasks requiring fine-grained semantic understanding (e.g., Emotion6). Notably, Qwen-VL-Plus suffers a significant drop on Emotion6, indicating that emotion recognition remains challenging for some open-source models.

Key takeaway. These results suggest that MAG is robust across a wide range of backbones, but its effectiveness is amplified by models with strong cross-modal alignment and reasoning capabilities. In particular, backbones that achieve a better balance between perception and reasoning (e.g., Gemini-2.0-Flash, GPT-5) tend to yield the most consistent gains across diverse benchmarks.

## B.2 Scalability Analysis

We further analyze the scalability of our method in terms of both computational and memory complexity.

As shown in Eq. (4) of the main paper, the steady-state solution of graph propagation is:

$$
\begin{array} { r } { I _ { ( \infty ) } ^ { ( m ) } = ( 1 - \alpha ) ( \mathbf { I } - \alpha \hat { W } ^ { ( m ) } ) ^ { - 1 } I _ { ( 0 ) } ^ { ( m ) } . } \end{array}
$$

The main computational cost arises from solving the linear system involving $( \mathbf { I } - \alpha \hat { W } ^ { ( m ) } )$ . In practice, $\hat { W } ^ { ( m ) }$ is a sparse affinity matrix constructed via k-nearest neighbors. Therefore, this reduces to solving a sparse linear system, rather than performing dense matrix inversion.

Such systems can be efficiently solved using iterative methods (e.g., Conjugate Gradient, Lanczos) or sparse Cholesky decomposition, with complexity typically ranging from O(n log n) to $O ( n ^ { 1 . 5 } )$ , significantly lower than the $O ( n ^ { 3 } )$ complexity of dense inversion. Memory consumption scales linearly as ${ \dot { O } } ( n k )$ , since only non-zero edges are stored.

Empirical Validation. We validate the above analysis with empirical measurements shown in Table 6.

Table 6: Latency and memory usage of graph propagation.
<table><tr><td>Sample Number</td><td>600</td><td>800</td><td>1000</td><td>1200</td></tr><tr><td>Edge Number</td><td>6000</td><td>8000</td><td>10000</td><td>12000</td></tr><tr><td>Latency (ms)</td><td>5.7</td><td>7.6</td><td>9.5</td><td>11.4</td></tr><tr><td>Memory Usage (MB)</td><td>12.8</td><td>20.1</td><td>28.0</td><td>38.5</td></tr></table>

As the number of samples increases, the number of edges grows linearly, consistent with the $O ( n k )$ complexity. The latency increases approximately linearly (doubling the samples roughly doubles runtime), confirming that our method avoids cubic complexity. Memory usage also exhibits nearlinear growth, with minor deviations due to implementation overhead.

Overall, both theoretical and empirical results demonstrate that our approach remains efficient and scalable for larger unlabeled datasets.

## B.3 Pseudo-Label Accuracy Analysis

We present a quantitative comparison between different pseudo-label selection strategies in Table 7.

Table 7: Pseudo-label accuracy (p-label acc) and final performance.
<table><tr><td>Metric</td><td>EmoSet</td><td>Emotion6</td><td>TextOCR</td><td>MMStar</td><td>GQA</td></tr><tr><td>Random (p-label acc)</td><td>0.667 0.778</td><td>0.556</td><td>0.833</td><td>0.756</td><td>0.489</td></tr><tr><td>Ours (p-label acc)</td><td></td><td>0.400</td><td>0.933</td><td>0.800</td><td>0.578</td></tr><tr><td>Random (final) Ours (final)</td><td>0.733 0.770</td><td>0.657</td><td>0.870 0.917</td><td>0.687 0.703</td><td>0.570</td></tr><tr><td></td><td></td><td>0.650</td><td></td><td></td><td>0.623</td></tr></table>

We observe a strong positive correlation between pseudo-label accuracy and final performance. In 4 out of 5 datasets, our method outperforms random selection in both pseudo-label quality and downstream performance, validating the effectiveness of our selection strategy.

Robustness to Real-World Noise. To address real-world challenges such as cross-modal inconsistencies, our method adopts modality-specific disentanglement. Each modality is processed independently, preventing noise in one modality (e.g., low-quality images) from contaminating the other during propagation.

As demonstrated in Section 4.6, the proposed late-fusion strategy consistently outperforms alternative designs across seven datasets. By selectively integrating reliable steady-state representations from each modality, our method exhibits strong robustness to semantic gaps and noisy inputs.

## C Qualitative Analysis

## C.1 Comparison with Baselines

Figure 4 compares MAG with strong baselines across emotion recognition (EmoSet), visual QA (GQA), reasoning (MMStar), and OCR (TextOCR). Our method consistently produces more accurate predictions. In emotion recognition, we correctly identify subtle affective cues (disgust) while baselines produce generic interpretations. For reasoning tasks, graph-based selection helps distinguish linguistically plausible but visually inconsistent options. In OCR, we accurately recognize scene text (REINICIO) where baselines hallucinate. These examples demonstrate that multi-modal graph propagation enables selection of globally relevant demonstrations beyond local similarity.

![](images/5d83aeafa5fdab926cd45016aabf860b7251d0efa41bba5a49e99ab9fa3e7a9a.jpg)  
Figure 4: Qualitative comparison between MAG and baselines (VICL, MMICES, CVR-LLM) across four tasks. Our method consistently produces more accurate predictions by selecting semantically relevant demonstrations via graph-based propagation.

## C.2 Ablation Visualizations

Figures 5–7 present qualitative ablations comparing the full method with randomized Stage 1 (pseudolabel selection) and randomized Stage 2 (demonstration selection).

When Stage 1 selection is randomized (Figure 6), the retrieved demonstrations are less semantically aligned with the queries, which in turn yields less stable emotion recognition and weaker objectcentric reasoning. Moreover, random selection increases the incidence of corrupted OCR text outputs, suggesting that the model is less likely to incorporate informative unlabeled samples under this setting.

Replacing Stage 2 with random demonstration selection (Figure 7) further degrades performance. Irrelevant demonstrations introduce noisy visual-text associations, harming multi-choice reasoning and scene-text understanding.

In contrast, the full model (Figure 5) consistently selects semantically coherent demonstrations, producing accurate predictions across all task types. These qualitative results verify that both relevance-guided pseudo-labeling (Stage 1) and graph-based demonstration selection (Stage 2) are essential for robust multi-modal reasoning.

![](images/65943d7b84302b36ea81831375d13f74ef3ce56b0daf786096e263e7f39aef58.jpg)  
Figure 5: Visualization of the full MAG method across EmoSet, GQA, MMStar, and OCR tasks.

![](images/f29ca81e93789980da40b89ec442eb4510a0df1c9cf3d819a7ffe9733781b1bd.jpg)  
Figure 6: MAG with random Stage 1 pseudo-label selection (vs. relevance-guided). Random selection leads to weaker semantic alignment and less accurate predictions.

![](images/c2f3fb6ea651bef95cf7fbd24def9c379839a990ed18fb9f1e78b8959afd5321.jpg)  
Figure 7: MAG with random Stage 2 demonstration selection (vs. graph-based). Random selection introduces noisy demonstrations, harming reasoning and OCR tasks.

## D Evaluation Protocol

Following prior multi-modal in-context learning works, we adopt an exact containment-based evaluation criterion for all benchmarks. Specifically, given the model response r and the ground-truth answer y, a prediction is considered correct if the ground-truth text appears in the generated response:

$$
{ \mathrm { C o r r e c t } } ( r , y ) = { \left\{ \begin{array} { l l } { 1 , } & { { \mathrm { i f ~ } } y \subseteq r , } \\ { 0 , } & { { \mathrm { o t h e r w i s e } } . } \end{array} \right. }
$$

That is, if the response contains the ground-truth answer as a substring, the prediction is counted as correct; otherwise, it is treated as incorrect. We report the average accuracy over all evaluation samples. This evaluation protocol is particularly suitable for open-ended multi-modal generation settings, where models may produce additional explanatory text while still containing the correct answer.

## E Prompts for ICL

The prompts we use for in-context learning on different tasks are as follows.

• For visual emotion recognition task(EmoSet and Emotion6), the prompt we use in our method is "What is the emotion category of this image? Give only the emotion category, and no extra commentary, formatting, or chattiness. You can only make prediction from the following categories: [Label List]. Given an image and its description, please predict the emotion category of this image. Here are several examples. Image: image-1, Description: description-1, Question: question-1, Emotion category: label-1. Image: image-2, Description: description-2, Question: question-2, Emotion category: label-2... Image: image-N, Description: description-N, Question: question-N, Emotion category: label-N. Image: image-q, Description: description-q, Question: question-q, Emotion category:".

• For TextOCR, MMStar, CLEVR, GQA and OKVQA, the prompt is "Given an image and its description, please answer a question about the image. Here are several examples. Image: image-1, Description: description-1, Question: question-1, Answer: label-1. Image: image-2, Description: description-2, Question: question-2, Answer: label-2... Image: image-N, Description: description-N, Question: question-N, Answer: label-N. Image: image-q, Description: description-q, Question: question-q, Answer:".

• For MatchingMI, the prompt is "Given two images and their description, please answer a question about the two images. Here are several examples. Images: image-1-Aimage-1-B, Description: description-1, Question: question-1, Answer: label-1. Image: image-2- Aimage-2-B, Description: description-2, Question: question-2, Answer: label-2... Image: image-N-Aimage-N-B, Description: description-N, Question: question-N, Answer: label-N. Image: image-q-Aimage-q-B, Description: description-q, Question: question-q, Answer:".