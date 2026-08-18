# Task-Anchored Representation Shaping for Pre-Trained Model-Based Continual Learning

Zhiming Xu<sup>1,2</sup>, Huiyu Yi<sup>1,2</sup>, Zhen-Hao Xie<sup>1,2</sup>, Baile Xu<sup>1,2</sup>, Furao Shen<sup>1,2,†</sup>, Jian Zhao<sup>4</sup>, Suorong Yang<sup>1,3</sup>

## Abstract

Pre-trained models (PTMs) provide a strong foundation for continual learning by ofering stable representations that facilitate lightweight adaptation to new tasks. However, adapting well to each task does not ensure reliable inference over all learned tasks. Since task boundaries are often artificial and semantically entangled, an input from an unknown task can remain ambiguous even with strong PTM features, mak ing cross-task prediction a key bottleneck. We propose Task-Anchored Inference Latent Shaping (TAILS), a lightweight post-PTM module that can be integrated into diverse continual learners and optimized through a decoupled step. TAILS uses fixed task anchors as persistent references to accumulated knowledge. It interprets each sample’s feature representation relative to these references, then composes relevant evidence across tasks into latent recall. Rather than selecting a task-specific path or adjusting classifier outputs, TAILS uses latent recall to directly correct the feature representation before prediction. It therefore resolves cross-task ambiguity at the representation level, while leaving the original PTM, method-specific modules, and classifier unchanged. Extensive experiments across multiple PTM-based continual learning paradigms show that TAILS can improve classification and task-inference performance with modest parameter overhead and negligible inference cost.

## 1 Introduction

Modern deep models are increasingly deployed in nonstationary environments, where data arrive sequentially, and their distributions evolve over time (Gomes et al. 2017; Ye et al. 2019; Chen et al. 2021, 2022; Yang, Shen, and Zhao 2024). To remain efective in such environments, models must continually incorporate new knowledge without erasing what has already been learned, which is the central objective of continual learning (CL) (Liang and Li 2024; Yang et al. 2025). However, sequential updates can cause catastrophic forgetting (De Lange et al. 2021), whereby learning new data overwrites previously acquired knowledge. Although conventional CL methods alleviate this problem (Zheng et al. 2025; Aghasanli, Li, and Angelov 2025; Li et al. 2025; Nori, Kim, and Wang 2025), they still learn representations primarily from limited incremental, task-specific data, restricting their ability to maintain a stable and discriminative feature space across tasks. Pre-trained models (PTMs) ofer a stronger foundation for CL by supplying representations learned from large-scale data (Wang et al. 2022a; Zhou et al. 2025a; Wang, Zhou, and Ye 2025; Sun et al. 2025). Rather than learning representations from scratch on scarce incremental data, PTM-based methods adapt to incoming tasks through lightweight components while keeping the backbone frozen or lightly updated, yielding stable and discriminative features throughout continual learning.

![](images/10d74f5e8700dea2b31346933d803c9a63873df7ab49747b51b858e61dd0fb39.jpg)  
(a) Task-Inference Bottleneck

![](images/e7a0078d368e4a6648517d9ad2fbaaaa3c8df54b2ada3c383d7fc2e4b3eef11a.jpg)  
(b) Recall-Guided Reshaping  
Figure 1: Motivation of TAILS. Left: Accuracy under the original and task-ID-oracle settings throughout continual learning. Right: For an ambiguous input, TAILS interprets its representation against accumulated task structures, composes sample-specific latent recall, and uses it to condition a feature correction before prediction by the original classifier.

Despite these advantages, data arriving at each learning stage are conventionally organized as distinct tasks (Zhou et al. 2024a), whereas inference must be performed over all accumulated tasks without task identity. As powerful PTMs and task-wise adaptation have substantially reduced intratask prediction errors, the primary challenge has shifted from learning individual tasks to identifying and exploiting relevant cross-task knowledge, making this identification and use increasingly important. For example, in continual classification (Zhou et al. 2024c), classes are typically assigned to tasks in a random order so that the resulting task boundaries may exhibit little meaningful semantic separation. Consequently, inputs from diferent tasks may remain highly overlapping in the feature space, making task inference inherently ambiguous. This ambiguity becomes increasingly severe as more tasks are accumulated, since the learner must distinguish among a growing number of semantically entangled task distributions. Existing methods mainly address this problem through task selection (Wang et al. 2022a; Xu et al. 2025; Sun et al. 2025) or model ensembling (Zhou et al. 2024b; Wang, Zhou, and Ye 2025). These strategies organize task-specific components through routing or aggregation, but their efectiveness depends heavily on accurately estimating task relevance. When task boundaries lack clear semantic separation, inaccurate task inference inevitably leads to selecting or weighting inappropriate task-specific components, while leaving the underlying feature ambiguity unresolved. Fig. 1a illustrates that under a Task-ID Oracle setting, where ground-truth task identity is supplied at inference, accuracy for existing PTM-based CL methods remains nearly flat as tasks accumulate; without this oracle, task-prediction accuracy drops sharply and converges with overall accuracy in later tasks. This convergence suggests that existing methods already perform reliably once the correct task is identified, whereas errors in task inference increasingly dominate final prediction performance, making task inference a major bottleneck for continual learning. This naturally raises a new and pressing question for PTM-based CL: how can a PTMbased CL learner efectively acquire and preserve task-level structuresfor reliable cross-task inference?

To address the cross-task inference bottleneck, we internalize accumulated task structures into the current representation before prediction. Specifically, we propose Task-Anchored Inference Latent Shaping (TAILS), a lightweight plug-and-play module inserted after the PTM feature encoder and before the classifier. Unlike selection and ensembling, TAILS does not route samples to a discrete task-specific component; instead, it composes a soft, input-conditioned summary of evidence across all learned tasks and uses it to correct the feature itself, leaving the original classifier and prediction mechanism unchanged. Concretely, given a PTMderived feature, TAILS forms a compact query, matches it against task slots built from accumulated task anchors, composes the retrieved evidence into a sample-specific latent recall, and uses this recall to condition a correction of the feature before classification (Fig. 1b). TAILS is shared across all learned tasks and trained in a decoupled step after the base learner’s standard training completes, so it augments inference without altering or risking the base learner’s original training pipeline. To keep this recall process stable as new tasks arrive, we construct fixed task anchors from class prototypes and support features, which serve as compact, persistent references both for preserving the recall space during optimization and for retrieving accumulated evidence at inference. Because TAILS is a plug-in, post-hoc module rather than a replacement for any existing component, it can be attached to a wide range of PTM-based continual learners with modest parameter overhead and negligible inference cost. It consistently elevates diverse state-of-the-art baselines such as TUNA and CL-LoRA by up to 5.05 percentage points while incurring as little as 3.0% additional storage overhead when added to storage-heavy base learners, highlighting both its eficiency and efectiveness. The main contributions are summarized as follows: (1) We propose TAILS, a plug-in post-PTM module that performs assisted inference with task-anchored recall, enabling base learners to shape representations without modifying their original prediction mechanisms. (2) We construct task anchors from class prototypes and support features. These anchors provide compact references for preserving and accessing accumulated task structures during optimization and inference. (3) We conduct extensive experiments across diverse PTM-based CL paradigms, showing that TAILS consistently improves performance with modest parameter overhead and negligible inference cost.

## 2 Related Work

## 2.1 Continual Learning

Continual learning (CL) aims to continuously acquire knowledge from non-stationary data streams while preserving previously learned capabilities. A broad body of work has addressed this objective across diverse continual settings through complementary mechanisms (Zhu et al. 2025; Zhang et al. 2025; Zhou et al. 2025b; Liu et al. 2025; Gao, Zhao, and Fu 2026). Replay-based methods retain past information by storing representative samples or reconstructing compact historical distributions from prototypes and features (Zhu et al. 2025; Zhang et al. 2025). Regularization- and distillationbased approaches constrain parameter updates or model responses to remain consistent with previously learned functions (Gao et al. 2025; Zang et al. 2026). Prototype- and feature-based methods preserve class-level statistics, refine historical prototypes, or compensate for representation drift across increments (Zhu et al. 2025; Gao, Zhao, and Fu 2026). Architecture- and ensemble-based methods instead isolate or combine knowledge by expanding model capacity, organizing related classes, or aggregating multiple model states (Lai et al. 2025; Lee, Hayat, and Yun 2025). However, these methods still construct and maintain their knowledge from the limited incremental stream, which constrains the quality and generalizability of the learned feature space. This limitation has motivated the growing adoption of large-scale pre-trained models as a stronger representational foundation for continual learning.

## 2.2 Pre-Trained Model-based CL

Recent studies have increasingly incorporated pre-trained models into continual learning. Early approaches adapt frozen PTMs with learnable prompts, including L2P (Wang et al. 2022b), DualPrompt (Wang et al. 2022a), CODA-Prompt (Smith et al. 2023), and APT (Chen et al. 2025). Adapter-based methods insert lightweight modules for efficient task adaptation. EASE (Zhou et al. 2024b) builds task-specific adapter subspaces and ensembles their representations, while MOS (Sun et al. 2025) trains taskspecific adapters with improved retrieval to reduce forgetting.

![](images/07232b7aa7572113256f0deb951fe724ea671de86214aef72d551376561c4291.jpg)  
Figure 2: Illustration of TAILS. After each task is learned by the base learner, its class prototypes and features are retained as task anchors. During continual optimization, the accumulated anchor pool maintains a coherent latent-recall space. At inference, a query derived from the current representation addresses the accumulated task structures and composes sample-specific latent recall, which conditions feature shaping before prediction by the original classifier.

SEMA (Wang et al. 2025) dynamically expands modularized adapters under distribution shifts, whereas TUNA (Wang, Zhou, and Ye 2025) combines task-specific and universal adapters for specialization and knowledge sharing. In addition, MOS and TUNA preserve class-alignment statistics from previous tasks, which can be viewed as a form of non-sample replay. Such retained statistics mainly support adapter retrieval and classifier calibration, rather than directly resolving sample-specific ambiguity at the representation level before prediction. LoRA-based methods perform adaptation through low-rank updates. SD-LoRA (Wu et al. 2025) decouples LoRA direction and magnitude, while CL-LoRA (He, Duan, and Zhu 2025) combines task-shared and task-specific low-rank adapters. Other works revisit PTMbased CIL through prototype learning (Zhou et al. 2025a) and analytic classifiers (Yi et al. 2026), further demonstrating the benefits of strong PTM representations for continual recognition.

## 3 Preliminaries

A continual learning (CL) model learns from a sequence of evolving data streams $D _ { 1 } , D _ { 2 } , \cdots , D _ { t }$ , where each stream $D _ { i } \ : = \ : \{ ( \mathbf { x } _ { j } , y _ { j } ) \} _ { j = 1 } ^ { n _ { i } }$ contains $n _ { i }$ training samples. For any stage �, the current stream $D _ { t }$ is treated as a new task, with its label space denoted by $Y _ { t }$ . The model has encountered the data streams $D _ { 1 : t } = D _ { 1 } \cup D _ { 2 } \cup \cdot \cdot \cdot \cup D _ { t }$ and the accumulated label space $y _ { 1 : t } = Y _ { 1 } \cup Y _ { 2 } \cup \cdot \cdot \cdot \cup Y _ { t }$ . The goal is to learn a CL model $f _ { t } : X \to y _ { 1 : t }$ that performs well on all learned data. The performance of $f _ { t }$ is evaluated by its classification error on the accumulated test set:

$$
\begin{array} { r } { \mathrm { E r r } ( f _ { t } ) = \mathbb { E } _ { ( \mathbf { x } , \mathbf { y } ) \in D _ { 1 : t } ^ { \mathrm { t e s t } } } \left[ \mathbb { I } \left( f _ { t } ( \mathbf { x } ) \neq \mathbf { y } \right) \right] , } \end{array}\tag{1}
$$

where I(·) is the indicator function. We follow the commonly adopted class-incremental continual learning setting, where task IDs are unavailable at inference. The model must make correct predictions among all classes using only the sample.

This paper focuses on PTM-based CL, where the learner adopts a powerful pre-trained encoder as its feature extractor, therefore providing a stronger foundation for continual adaptation. Given an input sample x, a PTM-based learner first extracts its representation through the pre-trained encoder $\phi _ { \mathrm { p t m } }$ and then predicts the label with a classifier �:

$$
\mathbf { h } = \phi _ { \mathrm { p t m } } ( \mathbf { x } ) , \quad \hat { y } = G ( \mathbf { h } ) .\tag{2}
$$

To better adapt the PTM to downstream continual tasks, recent PTM-based CL methods usually freeze the PTM backbone and introduce lightweight fine-tuning modules. The PTM-based learner is trained only on the current task data $D _ { t }$ at task $t ;$ once it proceeds to the next task, previous training images are inaccessible.

## 4 Method

TAILS introduces task-anchored assisted inference at the feature level. Unlike task routing, which selects a task-specific path, or output calibration, which adjusts logit scores from the classifier, TAILS performs auxiliary inference directly within the feature space. Specifically, a shared post-PTM module uses the retrieved latent recall to reposition an ambiguous input in the accumulated class space, while leaving the base learner and its original prediction mechanism unchanged, as illustrated in Fig. 2. In the following, we describe the TAILS module, its training objective, and its integration with diferent PTM-based learners.

## 4.1 Task-Anchored Latent Recall and Shaping

TAILS is a unified post-PTM module $\mathcal { F } _ { \mathrm { T A I L S } } ( \mathbf { h } ; \mathcal { A } )$ shared across all learning tasks. It interprets the current representation against task structures retained in the accumulated task anchors, composes relevant historical evidence into latent recall, and internalizes the recalled evidence as a feature correction. After the base learner finishes task �, we construct an anchor subset $\mathcal { A } _ { i }$ from its post-update sample features. For each class, we retain its class prototype and uniformly sample � support features without replacement, resulting in a class-balanced anchor set. If fewer than � samples are available, all features of that class are retained. The prototypes capture class-level semantics, while the support anchors preserve local feature variations. Once constructed, each $\mathcal { A } _ { i }$ remains fixed, and the accumulated task anchor pool $\mathcal { A } = \{ \mathcal { A } _ { 1 } , \ldots , \mathcal { A } _ { t } \}$ provides a persistent reference for crosstask optimization and inference. Since continual tasks may overlap semantically, a representation that is discriminative within its local task can remain ambiguous among classes learned at diferent tasks. TAILS addresses this ambiguity through soft, input-dependent access to the accumulated anchor structures. To establish this access, TAILS represents the current input and each task-specific anchor structure in the same compact space. Given a representation h $\epsilon \mathbb { R } ^ { d }$ extracted from the input sample, we first derive a compact normalized query $\mathbf { q } \in \mathbb { R } ^ { d _ { q } }$ , where $d _ { q } = 6 4$

$$
\mathbf { q } = \mathrm { N o r m } \left( \mathrm { L N } ( W _ { q } \mathbf { h } ) \right) ,\tag{3}
$$

where LN(·) denotes layer normalization (Ba, Kiros, and Hinton 2016), and Norm(·) denotes $\ell _ { 2 }$ normalization. The query provides an input-conditioned address for accessing the accumulated anchor structures. Let $\mathbf { A } _ { i } ~ \in ~ \mathbb { R } ^ { | \mathcal { A } _ { i } | \times d }$ denote the matrix obtained by stacking all anchors in $\mathcal { A } _ { i }$ . To make the stored anchors directly comparable with the query ${ \bf q } ,$ TAILS projects every anchor of task � from the original feature space into the same $d _ { q }$ -dimensional query space:

$$
\overline { { \mathbf { A } } } _ { i } = \mathrm { N o r m } \left( \mathrm { L N } \left( \mathbf { A } _ { i } W _ { a } ^ { \top } \right) \right) .\tag{4}
$$

Then TAILS summarizes the embedded anchor structure into � task slots:

$$
{ { \bf { Z } } _ { i } } = \mathrm { { N o r m } } \left[ { \mathrm { s o f t m a x } \left( { \frac { { \bf { S } } { { \bf { \overline { { A } } } } _ { i } ^ { \top } } } { \sqrt { d _ { q } } } } \right) { \bf { \overline { { A } } } } _ { i } } \right] ,\tag{5}
$$

where the softmax is applied row-wise over the anchors, S ∈ $\mathbb { R } ^ { K \times d _ { q } }$ contains � learnable slot vectors shared across tasks, and $\mathbf { Z } _ { i } \ = \ [ \mathbf { z } _ { i , 1 } , \ldots , \mathbf { z } _ { i , K } ] ^ { \top }$ contains the task slots derived from $\mathcal { A } _ { i }$ . Thus, all tasks share the same slot construction, while the content of $\mathbf { Z } _ { i }$ is determined by the corresponding anchor set $\mathcal { A } _ { i }$ . The normalized mean of the task slots defines a coarse address for task �:

$$
\mathbf { T } _ { c } ^ { i } = \mathrm { N o r m } \left( \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \mathbf { z } _ { i , k } \right) .\tag{6}
$$

Given a sample query ${ \bf q } ,$ TAILS first evaluates the relevance of each accumulated task structure and then the relevance of its individual task slots:

$$
\begin{array} { r } { \pi _ { i } ( \mathbf { q } ) = \mathrm { s o f t m a x } _ { i } \left( \mathbf { q } ^ { \top } \mathbf { T } _ { c } ^ { i } \right) , } \end{array}\tag{7}
$$

$$
\rho _ { i , k } ( { \bf q } ) = \mathrm { s o f t m a x } _ { k } \left( { \bf q } ^ { \top } { \bf z } _ { i , k } \right) .\tag{8}
$$

$$
\mathbf { r } = \sum _ { i = 1 } ^ { t } \pi _ { i } ( \mathbf { q } ) \sum _ { k = 1 } ^ { K } \rho _ { i , k } ( \mathbf { q } ) \mathbf { z } _ { i , k } .\tag{9}
$$

We refer to r as latent recall: an input-conditioned summary of accumulated feature structures that provides auxiliary evidence for interpreting the current representation. The task organization acts only as an indexing scafold rather than a hard task decision. Since $\pi _ { i } ( \mathbf { q } )$ is soft, latent recall may combine relevant task slots from multiple tasks when semantic neighborhoods cross the procedural boundaries of continual training. The resulting r provides the historical context that TAILS internalizes through a conditioned feature displacement. Specifically, the current feature determines a candidate correction direction, while latent recall modulates which dimensions of that correction are expressed:

$$
\mathbf { h } ^ { \prime } = \mathcal { F } _ { \mathrm { T A I L S } } ( \mathbf { h } ; \mathcal { A } ) = \mathrm { N o r m } ( \mathrm { L N } ( W _ { b } \mathbf { h } ) + \Delta \mathbf { h } ( \mathbf { h } , \mathbf { r } ) ) ,\tag{10}
$$

$$
\Delta \mathbf { h } ( \mathbf { h } , \mathbf { r } ) = W _ { o } \mathrm { G E L U } ( W _ { h } \mathbf { L } \mathbf { N } ( \mathbf { h } ) \odot \sigma ( W _ { r } \mathbf { L } \mathbf { N } ( \mathbf { r } ) ) )\tag{11}
$$

Here, $W _ { b }$ is an identity-initialized linear mapping, allowing this branch to start from the original PTM feature and avoid disturbing pre-trained semantics at the beginning of TAILS training (Houlsby et al. 2019). Unlike conventional residual feature adaptation conditioned only on the current representation, TAILS conditions its correction on latent recall derived from accumulated task structures.

## 4.2 Latent-Recall-Preserving Optimization

The TAILS module is incrementally updated after each task. Once the base learner completes task $t ,$ we freeze its representation learner and classifier and optimize only TAILS in the feature space. Let $\mathcal { F } _ { t } ^ { \mathrm { c u r } }$ denote the set of post-update representations extracted from the current-task samples. The optimization uses ${ \mathcal { F } } _ { t } ^ { \mathrm { c u r } }$ together with the historical anchor set $\begin{array} { r } { \mathcal { A } _ { 1 : t - 1 } = \bigcup _ { i = 1 } ^ { t - 1 } \mathcal { A } _ { i } . } \end{array}$ . These two sources play complementary roles: current-task representations incorporate newly acquired structures into the shared TAILS mapping, whereas historical anchors provide stable reference features for preserving previously established classifier compatibility, recall addresses, and local feature geometry without retaining any previous training samples. Since the anchors remain fixed and the base learner is never revisited, this procedure updates only how TAILS interprets the accumulated feature space without modifying the original continual-learning model. Accordingly, each training feature $\mathbf { h } _ { \mathrm { t r a i n } } \in \mathcal { F } _ { t } ^ { \mathrm { c u r } } \cup \mathcal { A } _ { 1 : t - 1 }$ is shaped as $\dot { \mathbf { h } } _ { \mathrm { t r a i n } } ^ { \prime } = \mathcal { F } _ { \mathrm { T A I L S } } ( \mathbf { h } _ { \mathrm { t r a i n } } ; \mathcal { A } )$ , and its corresponding query q is computed from $\mathbf { h } _ { \mathrm { t r a i n } }$ using Eq. (3). We impose three complementary constraints:

$$
{ \mathcal { L } } _ { \mathrm { T A I L S } } = { \mathcal { L } } _ { \mathrm { c l s } } + { \mathcal { L } } _ { \mathrm { t a s k } } + { \mathcal { L } } _ { \mathrm { s t a b } } .\tag{12}
$$

The classifier-compatibility constraint ensures that the shaped feature remains meaningful under the frozen decision rule of base learner B:

$$
\mathcal { L } _ { \mathrm { c l s } } = \mathrm { C E } \left( G _ { \mathcal { B } } ( \mathbf { h } _ { \mathrm { t r a i n } } ^ { \prime } ) , y \right) .\tag{13}
$$

Gradients pass through the fixed classifier only to TAILS. Thus, the objective does not learn a new classifier; it teaches latent recall to produce feature corrections that are compatible with the base learner’s established semantic space.

The recall-addressing constraint organizes where the current query should seek evidence. Let $\tau ( y )$ denote the task

ID associated with the class �. We supervise the similarities between the query and accumulated task centers:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { t a s k } } = \mathrm { C E } \Big ( \big \{ \mathrm { s i m } _ { \mathrm { c o s } } \big ( \mathbf { q } _ { \mathrm { t r a i n } } , \mathbf { T } _ { c } ^ { i } \big ) \big \} _ { i = 1 } ^ { t } , \tau ( y ) \Big ) . } \end{array}\tag{14}
$$

This signal learns the addressing behavior of latent recall without requiring a task ID at inference. The supervised center response establishes a reliable address, whereas Eq. (9) still performs soft composition of task slots for the actual feature correction.

The semantic-preservation constraint limits excessive deviation from the original PTM representation:

$$
\mathcal { L } _ { \mathrm { s t a b } } = 1 - \frac { \mathbf { h } _ { \mathrm { t r a i n } } ^ { \prime } ^ { \top } \mathbf { h } _ { \mathrm { t r a i n } } } { \Vert \mathbf { h } _ { \mathrm { t r a i n } } ^ { \prime } \Vert _ { 2 } \Vert \mathbf { h } _ { \mathrm { t r a i n } } \Vert _ { 2 } } .\tag{15}
$$

Together, the three terms preserve classifier compatibility, recall-address consistency, and continuity of PTM semantics, allowing TAILS to incorporate new task structures while maintaining a coherent latent-recall across all tasks.

## 4.3 Integrating TAILS with PTM Learners

TAILS operates at the representation interface shared by otherwise heterogeneous PTM-based learners. This placement allows the same unified module to assist inference while preserving each base learner’s adaptation mechanism and original decision rule.

For methods that ultimately produce a single representation for prediction, including raw PTM, prompt-based methods, and adapter-based methods that fuse multiple adapter outputs into one (Wang, Zhou, and Ye 2025), TAILS applies $\mathcal { F } _ { \mathrm { T A I L S } } ( \cdot ; \mathcal { A } )$ to the resulting feature h and feeds the shaped representation h<sup>′</sup> from Eq. (10) to the original classifier. For methods that expose task-specific candidate features before prediction, such as task-specific adapters (Sun et al. 2025) or task-specific LoRA branches (He, Duan, and Zhu 2025), the same TAILS module is applied independently to each candidate feature:

$$
\mathbf { h } ^ { \prime } { } ^ { i } = \mathcal { F } _ { \mathrm { T A I L S } } \left( \mathbf { h } ^ { i } ; \mathcal { A } \right) , \quad i = 1 , \cdots , t .\tag{16}
$$

The base learner then follows its original candidate selection, aggregation, or prediction procedure. Thus, every candidate is interpreted through the same accumulated recall space without changing the semantics of the base method.

TAILS is also compatible with diferent classifier types. For a parametric classifier, such as a linear head, the shaped representation is passed directly to the original classifier:

$$
\hat { y } = \arg \operatorname* { m a x } _ { c } G _ { \mathcal { B } } \left( \mathbf { h } ^ { \prime } \right) _ { c } .\tag{17}
$$

For prototype-based cosine classifiers (Sun et al. 2025; Zhou et al. 2025a), the classifier is parameterized by stored class prototypes. Since TAILS changes the representation used for prediction, each class prototype is transformed by the same unified mapping to preserve sample–prototype consistency:

$$
\mu _ { c } ^ { \prime } = \mathcal { F } _ { \mathrm { T A I L S } } \left( \mu _ { c } ; \mathcal { A } \right) .\tag{18}
$$

The prediction rule is then:

$$
\hat { y } = \arg \operatorname* { m a x } _ { c } \sin _ { \cos } \left( \mathbf { h } ^ { \prime } , \mu _ { c } ^ { \prime } \right) .\tag{19}
$$

Consequently, deploying TAILS requires no structural modification to the original method. Its prediction mechanism remains intact, while task-anchored latent recall only reshapes the representation supplied to it.

## 5 Experiments

This section conducts extensive experiments to evaluate TAILS. Sec. 5.2 compares TAILS with representative PTMbased CL methods across diferent adaptation paradigms. Sec. 5.3 analyzes the training objectives, underlying mechanism, inference eficiency, and parameter overhead. Sec. 5.4 studies the sensitivity of TAILS to the number of support anchors and task slots. T-SNE visualizations of the feature spaces are provided in the supplementary material.

## 5.1 Implementation Details

Dataset and splits. We select four benchmark datasets that exhibit notable domain gaps (Zhou et al. 2024b) relative to the ImageNet-21K pretraining distribution, including Stanford Cars (Krause et al. 2013), ImageNet-A (Hendrycks et al. 2021b), ImageNet-R (Hendrycks et al. 2021a), and VTAB (Zhai et al. 2019). Following the “B/Base-m, Inc-n” protocol (Zhou et al. 2025a), datasets are split into CARS B16 Inc20, ImageNet-A B20 Inc20, VTAB B10 Inc10, and ImageNet-R B5 Inc5. ImageNet-R with B5 Inc5 forms a 40-task benchmark, allowing us to evaluate whether TAILS remains effective when task-level ambiguity accumulates over longsequence CL.

Baselines. We deploy TAILS on baseline or SOTA PTMbased CL approaches that cover diferent paradigms: promptbased method DualPrompt (Wang et al. 2022a), raw PTM and prototype classification method SimpleCIL (Zhou et al. 2025a), multi-task adapter methods MOS (Sun et al. 2025) and SEMA (Wang et al. 2025), adapter-based integration method TUNA (Wang, Zhou, and Ye 2025), and task-specific LoRA method CL-LoRA (He, Duan, and Zhu 2025). Following the original settings of each method, DualPrompt and SEMA use ViT-B/16-224, while the remaining methods adopt ViT-B/16-IN21K as the PTM backbone.

Programming and hyperparameters. All experiments are implemented in PyTorch 2.4.1 and conducted on NVIDIA 3090 GPUs. Baseline settings follow the configurations in original papers or LAMDA-Pilot (Sun et al. 2023). For TAILS, the number of task slots � is mainly set to 5, and the number of support anchors for each class is mainly set to 20. TAILS adopts the training configuration of the corresponding base method, including the optimizer, initial learning rate, batch size, number of epochs, and learning-rate schedule.

Evaluation metrics. We use $A _ { T }$ to evaluate the final classification accuracy on the joint test set of all tasks after training on the last task, and use $A _ { T } ^ { \mathrm { t a s k } }$ to measure whether the task of the predicted class matches the ground-truth task.

## 5.2 Benchmark Comparison

This subsection evaluates TAILS by deploying it on various PTM-based CL methods. Table 1 reports the mean and standard deviation of $A _ { T } ^ { \mathrm { t a s k } }$ and $A _ { T }$ over five seeds on four benchmark datasets. For high-performing baselines, $A _ { T } ^ { \mathrm { t a s k } }$ is often close to $A _ { T } ,$ , indicating that their within-task prediction is already strong, while cross-task inference becomes the main factor limiting final performance. TAILS efectively improves both cross-task prediction accuracy and final classification accuracy, enabling the enhanced learners to surpass existing state-of-the-art PTM-based CL methods on multiple benchmarks. The improvements are also consistent on the 40-task ImageNet-R dataset, demonstrating that TAILS remains efective on the long-sequence CL benchmark. These results demonstrate the broad applicability of TAILS to diverse PTM-based CL paradigms. In particular, TAILS still yields notable improvements when applied to strong SOTA baselines such as TUNA and CL-LoRA. We further report the accuracy trajectories throughout continual learning in $\mathrm { F i g . }$ . 3. TAILS consistently performs best in all evaluated settings, and the gains remain evident as more classes are introduced. The complete curves for all methods and datasets are provided in the supplementary material.

<table><tr><td rowspan="2">Method</td><td colspan="2">CARS-196 B16 Inc20</td><td colspan="2">ImageNetA-200 B20 Inc20</td><td colspan="2">ImageNetR-200 B5 Inc5</td><td colspan="2">VTAB-50 B10 Inc10</td></tr><tr><td> $A _ { T } ^ { \mathrm { t a s k } }$ </td><td> $A _ { T }$ </td><td> $A _ { T } ^ { \mathrm { t a s k } }$ </td><td> $A _ { T }$ </td><td> $A _ { T } ^ { \mathrm { t a s k } }$ </td><td> $A _ { T }$ </td><td> $A _ { T } ^ { \mathrm { t a s k } }$ </td><td> $A _ { T }$ </td></tr><tr><td>SimpleCIL</td><td> $4 3 . 5 3 _ { \pm 0 . 0 1 }$ </td><td> $3 7 . 6 8 _ { \pm 0 . 0 1 }$ </td><td> $5 2 . 9 8 _ { \pm 0 . 0 1 }$ </td><td> $4 8 . 6 7 _ { \pm 0 . 0 1 }$ </td><td> $5 5 . 1 9 _ { \pm 0 . 0 1 }$ </td><td> $5 4 . 5 9 _ { \pm 0 . 0 2 }$ </td><td> $8 7 . 1 1 _ { \pm 0 . 0 1 }$ </td><td> $8 4 . 3 3 _ { \pm 0 . 0 1 }$ </td></tr><tr><td>SimpleCIL-TAILS</td><td> $6 8 . 0 3 _ { \pm 0 . 9 7 }$ </td><td> $6 4 . 3 9 _ { \pm 0 . 9 5 }$ </td><td> $5 8 . 7 0 { \scriptstyle \pm 0 . 6 6 }$ </td><td> $5 4 . 8 6 _ { \pm 0 . 6 0 }$ </td><td> $6 8 . 6 9 _ { \pm 0 . 9 9 }$ </td><td> $6 7 . 9 4 _ { \pm 0 . 9 3 }$ </td><td> $9 1 . 9 8 _ { \pm 0 . 0 8 }$ </td><td> $9 0 . 1 8 _ { \pm 0 . 0 4 }$ </td></tr><tr><td>DualPrompt</td><td> $3 4 . 9 0 { \scriptstyle \pm 0 . 9 2 }$ </td><td> $3 2 . 7 3 { \scriptstyle \pm 0 . 9 6 }$ </td><td> $5 9 . 4 8 _ { \pm 0 . 8 0 }$ </td><td> $5 6 . 7 1 _ { \pm 0 . 8 4 }$ </td><td> $6 5 . 6 9 _ { \pm 0 . 9 4 }$ </td><td> $6 5 . 4 9 _ { \pm 0 . 9 8 }$ </td><td> $7 9 . 7 5 _ { \pm 2 . 8 4 }$ </td><td> $7 9 . 0 7 _ { \pm 2 . 9 0 }$ </td></tr><tr><td>DualPrompt-TAILS</td><td> $6 1 . 7 0 { \scriptstyle \pm 0 . 9 5 }$ </td><td> $5 9 . 7 6 { \scriptstyle \pm 0 . 9 3 }$ </td><td> $6 5 . 0 7 _ { \pm 0 . 8 3 }$ </td><td> $6 0 . 8 6 _ { \pm 0 . 8 1 }$ </td><td> $7 4 . 3 0 { \scriptstyle \pm 0 . 9 5 }$ </td><td> $7 3 . 8 9 _ { \pm 0 . 9 7 }$ </td><td> $9 0 . 1 6 { \scriptstyle \pm 2 . 8 5 }$ </td><td> $8 8 . 3 0 { \scriptstyle \pm 2 . 8 9 }$ </td></tr><tr><td>SEMA</td><td> $5 7 . 0 5 _ { \pm 0 . 8 5 }$ </td><td> $5 5 . 5 4 _ { \pm 0 . 9 1 }$ </td><td>64.82±0.35</td><td> $5 8 . 6 2 _ { \pm 0 . 4 1 }$ </td><td> $6 9 . 3 1 _ { \pm 0 . 8 5 }$ </td><td> $6 9 . 1 1 _ { \pm 0 . 9 1 }$ </td><td> $9 4 . 9 3 _ { \pm 0 . 8 6 }$ </td><td> $8 9 . 5 1 _ { \pm 0 . 9 2 }$ </td></tr><tr><td>SEMA-TAILS</td><td> $6 7 . 2 3 _ { \pm 0 . 9 0 }$ </td><td> $6 5 . 7 2 _ { \pm 0 . 8 6 }$ </td><td> $6 7 . 7 1 _ { \pm 0 . 4 0 }$ </td><td> $6 1 . 1 2 _ { \pm 0 . 3 6 }$ </td><td> $7 0 . 8 2 \substack { \pm 0 . 8 6 }$ </td><td> $7 0 . 5 6 { \scriptstyle \pm 0 . 9 0 }$ </td><td> $9 8 . 4 8 { \scriptstyle \pm 0 . 8 8 }$ </td><td> $9 2 . 0 3 { \scriptstyle \pm 0 . 9 0 }$ </td></tr><tr><td>MOS</td><td> $7 3 . 5 2 \substack { \pm 0 . 8 7 }$ </td><td> $7 0 . 6 6 _ { \pm 0 . 8 9 }$ </td><td> $5 8 . 8 1 _ { \pm 0 . 3 9 }$ </td><td> $5 5 . 2 0 { \scriptstyle \pm 0 . 3 7 }$ </td><td> $7 1 . 1 7 _ { \pm 0 . 8 9 }$ </td><td> $7 0 . 5 5 _ { \pm 0 . 8 7 }$ </td><td> $9 9 . 5 4 _ { \pm 0 . 9 1 }$ </td><td> $9 2 . 2 5 _ { \pm 0 . 8 7 }$ </td></tr><tr><td>MOS-TAILS</td><td> $7 5 . 4 8 _ { \pm 0 . 8 9 }$ </td><td> $7 2 . 7 5 _ { \pm 0 . 8 7 }$ </td><td> $6 0 . 9 2 _ { \pm 0 . 3 6 }$ </td><td> $5 7 . 9 6 _ { \pm 0 . 4 0 }$ </td><td> $7 3 . 9 0 { \scriptstyle \pm 0 . 8 5 }$ </td><td> $7 3 . 3 2 _ { \pm 0 . 9 1 }$ </td><td> $\mathbf { 9 9 . 6 5 } _ { \pm 0 . 8 5 }$ </td><td> $9 2 . 9 4 { \scriptstyle \pm 0 . 9 3 }$ </td></tr><tr><td>TUNA</td><td> $7 7 . 6 6 _ { \pm 0 . 9 0 }$ </td><td> $7 6 . 6 1 _ { \pm 0 . 8 6 }$ </td><td> $6 6 . 2 1 _ { \pm 0 . 3 7 }$ </td><td> $6 4 . 1 8 _ { \pm 0 . 3 9 }$ </td><td> $7 5 . 7 4 _ { \pm 0 . 8 7 }$ </td><td> $7 5 . 3 2 _ { \pm 0 . 8 9 }$ </td><td> $9 8 . 2 8 { \scriptstyle \pm 0 . 9 0 }$ </td><td> $9 1 . 8 0 { \scriptstyle \pm 0 . 8 8 }$ </td></tr><tr><td>TUNA-TAILS</td><td> $\mathbf { 7 9 . 0 6 } _ { \pm 0 . 8 7 }$ </td><td> $\mathbf { 7 8 . 0 9 } _ { \pm 0 . 8 9 }$ </td><td> ${ \bf 6 7 . 7 3 } _ { \pm 0 . 4 1 }$ </td><td> ${ \bf 6 5 . 3 0 { \scriptstyle \pm 0 . 3 5 } }$ </td><td> $7 7 . 1 6 { \scriptstyle \pm 0 . 8 9 }$ </td><td> ${ \bf 7 6 . 7 4 } _ { \pm 0 . 8 7 }$ </td><td> $9 8 . 4 4 _ { \pm 0 . 9 1 }$ </td><td> $9 2 . 2 7 _ { \pm 0 . 8 7 }$ </td></tr><tr><td>CL-LoRA</td><td> $6 5 . 4 9 _ { \pm 0 . 5 1 }$ </td><td> $6 4 . 3 6 { \scriptstyle \pm 0 . 5 5 }$ </td><td> $5 9 . 4 1 _ { \pm 0 . 1 2 }$ </td><td> $5 7 . 6 2 _ { \pm 0 . 1 6 }$ </td><td> $7 3 . 6 2 _ { \pm 0 . 6 1 }$ </td><td> $7 3 . 4 9 _ { \pm 0 . 6 5 }$ </td><td> $9 7 . 6 7 _ { \pm 0 . 3 2 }$ </td><td> $9 4 . 2 2 { \scriptstyle \pm 0 . 3 6 }$ </td></tr><tr><td> $\mathrm { C L \mathrm { - } L o R A \mathrm { - } T A I L S }$ </td><td> $7 0 . 1 2 { \scriptstyle \pm 0 . 5 3 }$ </td><td> $6 9 . 4 1 _ { \pm 0 . 5 6 }$ </td><td> $6 1 . 0 6 _ { \pm 0 . 1 3 }$ </td><td> $5 9 . 3 3 _ { \pm 0 . 1 5 }$ </td><td> $7 4 . 1 5 _ { \pm 0 . 6 6 }$ </td><td> $7 4 . 0 1 _ { \pm 0 . 6 2 }$ </td><td> $9 8 . 1 8 _ { \pm 0 . 3 3 }$ </td><td> ${ \bf 9 4 . 7 1 { \bf _ { \pm 0 . 3 5 } } }$ </td></tr></table>

Table 1: Comparison of task prediction accuracy $A _ { T } ^ { \mathrm { t a s k } }$ and final classification accuracy $A _ { T } .$ . ImageNet-R uses the B5 Inc5 split with 200 classes, resulting in a 40-task long-sequence benchmark. The best results are highlighted in bold.

![](images/b9c930d57df77729d8504a9f2ac62e6066d2853093f229fb9f8e42e78d730492.jpg)  
(a) ImageNet-A

![](images/a89324d9004632312eb6433c5d89086ada479c6e39305617f094782b596615ee.jpg)  
(b) ImageNet-R

<table><tr><td> $\overline { { \mathcal { L } _ { \mathrm { c l s } } } }$ </td><td> $\overline { { \mathcal { L } _ { \mathrm { t a s k } } } }$ </td><td> $\overline { { \mathcal { L } _ { \mathrm { { s t a b } } } } }$ </td><td>SimpleCIL-TAILS</td><td>TUNA-TAILS</td></tr><tr><td>×</td><td>×</td><td>×</td><td>48.67</td><td>64.18</td></tr><tr><td>×</td><td>√</td><td>√</td><td>50.51</td><td>65.17</td></tr><tr><td>√</td><td>×</td><td>√</td><td>54.12</td><td>64.91</td></tr><tr><td>√</td><td>√</td><td>×</td><td>54.78</td><td>65.02</td></tr><tr><td>√</td><td>√</td><td>√</td><td>54.86</td><td>65.30</td></tr></table>

Table 2: Ablation study of the training objectives in TAILS. All results are averaged over five seeds.

Figure 3: Accuracy trajectories of diferent methods on ImageNet-A and ImageNet-R.
<table><tr><td rowspan="2">Variant</td><td colspan="4">CARS</td><td colspan="4">ImageNet-R</td></tr><tr><td>SimpleCIL TUNA MOS CL-LoRA SimpleCIL TUNA MOS </td><td></td><td></td><td></td><td></td><td></td><td></td><td>CL-LoRA</td></tr><tr><td>Anchor-MLP</td><td>58.81</td><td></td><td>77.32 71.34</td><td>65.92</td><td>62.43</td><td></td><td>75.91 72.15</td><td>73.57</td></tr><tr><td>Anchor Retrieval</td><td>56.66</td><td></td><td>77.04 71.03</td><td>66.02</td><td>60.47</td><td></td><td>75.51 71.37</td><td>73.48</td></tr><tr><td>Single-Center</td><td>62.23</td><td></td><td>77.72 70.54</td><td>67.52</td><td>65.95</td><td></td><td>76.23 72.55</td><td>73.91</td></tr><tr><td>Logit Calibration</td><td>45.22</td><td>76.63</td><td>70.81</td><td>64.61</td><td>56.82</td><td></td><td>75.33 70.42</td><td>73.31</td></tr><tr><td>TAILS</td><td>64.39</td><td></td><td>78.09 72.75</td><td>69.41</td><td>67.94</td><td></td><td>76.74 73.32</td><td>74.01</td></tr></table>

Table 3: Mechanism comparison on CARS and ImageNet-R. All results are averaged over five seeds. All variants use the same anchor and optimization budgets.

## 5.3 Ablation Study

This subsection conducts a series of ablation studies on TAILS. We first investigate the efectiveness of the three training objectives in TAILS on the ImageNetA dataset, as shown in Table 2. The results show that $\bar { \mathcal { L } } _ { \mathrm { c l s } } , \mathcal { L } _ { \mathrm { t a s k } }$ , and $\mathcal { L } _ { \mathrm { s t a b } }$ all contribute to the final performance. The relative importance of ${ \mathcal { L } } _ { \mathrm { c l s } }$ and $\mathcal { L } _ { \mathrm { t a s k } }$ varies across diferent base learners.

For methods such as SimpleCIL, where classification is not suficiently optimized during continual learning, ${ \mathcal { L } } _ { \mathrm { c l s } }$ plays a more important role in directly improving classification accuracy. In contrast, TUNA has already learned task-specific classifiers through task-wise adaptation, so $\mathcal { L } _ { \mathrm { t a s k } }$ becomes critical for organizing task-level responses and improving task inference. $\mathcal { L } _ { \mathrm { s t a b } }$ further provides a consistent gain by preventing the shaped representation from drifting excessively away from the original PTM feature.

Moreover, we compare TAILS with four competitive variants under the same anchor and optimization budgets, as shown in Table 3. All results are averaged over five random seeds. For each seed, the same sampled anchors are shared by TAILS and all mechanism variants. Anchor-MLP removes the TAILS architecture and uses anchors only to assist the incremental update of an MLP. Anchor Retrieval directly computes attention between the input feature and all stored anchors, and uses their weighted sum for feature shaping. Single-Center just represents each task with one center rather than multiple slots. Logit Calibration applies TAILS to the logits instead of feature shaping. TAILS consistently performs best across all evaluated settings. The consistent gains over Anchor-MLP, Anchor Retrieval, and Single-Center validate the TAILS module design. Direct logit calibration is brittle because even small corrections can change class rankings and destabilize the final prediction. Moreover, cross-task ambiguity is a representation-level problem that is dificult to resolve by modifying the classifier alone.

<table><tr><td rowspan="2">Method</td><td colspan="2">ImageNet-A</td><td colspan="2">ImageNet-R</td></tr><tr><td> $A _ { T }$ </td><td> $\overline { { t } } _ { \mathrm { i n f } }$  (ms)</td><td> $A _ { T }$ </td><td> $\overline { { t } } _ { \mathrm { i n f } }$  (ms)</td></tr><tr><td>SimpleCIL</td><td>48.67</td><td>4.73</td><td>54.59</td><td>4.82</td></tr><tr><td>SimpleČIL-TAILS</td><td>54.86</td><td>4.92</td><td>67.94</td><td>5.10</td></tr><tr><td>ACMap</td><td>54.49</td><td>4.93</td><td>69.33</td><td>5.05</td></tr><tr><td>TUNÁ</td><td>64.18</td><td>62.78</td><td>75.32</td><td>145.04</td></tr><tr><td>TUNA-TAILS</td><td>65.30</td><td>66.09</td><td>76.74</td><td>150.42</td></tr></table>

Table 4: Comparison of average final accuracy and the persample inference time measured over the test set.

Furthermore, we investigate the inference-time overhead introduced by TAILS. For a more comprehensive comparison, we include ACMap (Fukuda, Kera, and Kawamoto 2025), a PTM-based method specifically optimized for inference speed as an eficiency-oriented reference. We report the mean final accuracy and the average per-sample inference time on the test set over multiple runs with five seeds, as shown in Table 4. TAILS introduces only marginal inference overhead over the original method. Notably, applying TAILS to SimpleCIL directly yields a learner that is competitive with state-of-the-art eficiency-oriented methods in both accuracy and inference speed.

<table><tr><td>Methods</td><td>Model Params.</td><td>Auxiliary Storage</td><td>Total Storage</td></tr><tr><td>ACMap</td><td>1.19</td><td>47.59</td><td>48.78</td></tr><tr><td>MOS</td><td>12.17</td><td>130.29</td><td>142.46</td></tr><tr><td>TUNA</td><td>12.48</td><td>118.12</td><td>130.60</td></tr><tr><td>CL-LoRA</td><td>7.56</td><td>0.00</td><td>7.56</td></tr><tr><td>TAILS</td><td>1.14</td><td>3.07</td><td>4.21</td></tr></table>

Table 5: Comparison of additional storage overhead on the ImageNet-R dataset, measured in millions of scalar values (M), excluding the PTM.

In addition, Table 5 compares the additional storage required by diferent methods under the ImageNet-R protocol. For a consistent comparison, we count both the model parameters and all auxiliary tensors that must be retained for subsequent learning or inference. The auxiliary storage includes the task anchors maintained by TAILS, the class-alignment statistics used by MOS and TUNA, and the task-specific adapter checkpoints retained by ACMap. For CL-LoRA, all task-expanded LoRA weights are included in the model parameters, with no separately stored auxiliary tensors. TAILS is lightweight on its own, and its plug-in overhead remains below the learner-specific storage introduced by every compared method. Relative to the storage of the corresponding base learner, adding TAILS incurs only 3.0% overhead for MOS, 3.2% for TUNA, and 8.6% for ACMap; even for the more compact CL-LoRA, the additional storage remains below its original task-expanded parameter footprint.

## 5.4 Sensitivity Analysis of Hyperparameters

We study how key hyperparameters in TAILS afect performance, with the results shown in Fig. 4. The results indicate that even when no support anchors are used, TAILS can still improve the base learner by relying solely on class prototypes as task anchors. Increasing the number of support anchors generally improves final classification accuracy, as additional anchors capture more within-class variation and support richer task-slot representations for composing latent recall. However, the gains diminish as additional anchors become increasingly redundant. Increasing the number of task slots further improves performance by capturing finer structures within each task. Too many slots, however, can fragment the task representation and make query-slot matching less reliable, thereby degrading latent-recall composition.

![](images/39b2aa014cf37f61d56f01f32a95102770f69da0eb081eb574463520d65f5c8a.jpg)  
(a) Support Anchors

![](images/2a122c16ee13febe497030310b6d67ab1397a45d6ec8dc636303723123788a7b.jpg)  
(b) Task Slots  
Figure 4: Impact of support anchor and task slot counts on the results across diferent PTM-based CL methods.

## 6 Conclusion

This paper introduces TAILS, a task-anchored latent-recallassisted inference module for PTM-based continual learning. TAILS interprets each current representation against fixed accumulated task structures, composes input-conditioned latent recall, and internalizes the recalled evidence as a recallconditioned feature correction. It is shared across all learned tasks and incrementally updated in a decoupled manner, so accumulated task-level information can refine current representations while preserving the original training pipeline of the base learner. We further store class prototypes and support features as fixed task anchors in the feature space, providing stable references for training and inference. Extensive experiments across various continual learning methods demonstrate that TAILS consistently improves baseline performance with modest parameter overhead and negligible inference cost. Future work will explore more expressive anchor construction strategies and stronger shaping objectives to further enhance task-level discriminability.

## References

Aghasanli, A.; Li, Y.; and Angelov, P. 2025. Prototype-Based Continual Learning with Label-free Replay Bufer and Cluster Preservation Loss. In Proceedings of the Computer Vision and Pattern Recognition Conference, 6545–6554.

Ba, J. L.; Kiros, J. R.; and Hinton, G. E. 2016. Layer Normalization. arXiv preprint arXiv:1607.06450.

Chen, H.; Wang, P.; Zhou, Z.; Zhang, X.; Wu, Z.; and Jiang, Y.-G. 2025. Achieving More with Less: Additive Prompt Tuning for Rehearsal-Free Class-Incremental Learning. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 340–349.

Chen, S.; Gong, C.; Li, J.; Yang, J.; Niu, G.; and Sugiyama, M. 2022. Learning contrastive embedding in low-dimensional space. Advances in Neural Information Processing Systems, 35: 6345–6357.

Chen, S.; Niu, G.; Gong, C.; Li, J.; Yang, J.; and Sugiyama, M. 2021. Large-margin contrastive learning with distance polarization regularizer. In International Conference on Machine Learning, 1673–1683. PMLR.

De Lange, M.; Aljundi, R.; Masana, M.; Parisot, S.; Jia, X.; Leonardis, A.; Slabaugh, G.; and Tuytelaars, T. 2021. A continual learning survey: Defying forgetting in classification tasks. IEEE transactions on pattern analysis and machine intelligence, 44(7): 3366–3385.

Fukuda, T.; Kera, H.; and Kawamoto, K. 2025. Adapter Merging with Centroid Prototype Mapping for Scalable Class-Incremental Learning. In Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR), 4884–4893.

Gao, R.; Zhao, Q.; and Fu, K. 2026. DiAPR: Dimensionally-Allocated Prototype Refinement for Non-Exemplar Class Incremental Learning. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 40, 21189–21197.

Gao, Z.; Han, S.; Zhang, X.; Xu, K.; Zhou, D.; Mao, X.; Dou, Y.; and Wang, H. 2025. Maintaining fairness in logit-based knowledge distillation for class-incremental learning. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, 16763–16771.

Gomes, H. M.; Barddal, J. P.; Enembreck, F.; and Bifet, A. 2017. A survey on ensemble learning for data stream classification. ACM Computing Surveys (CSUR), 50(2): 1– 36.

He, J.; Duan, Z.; and Zhu, F. 2025. CL-LoRA: Continua Low-Rank Adaptation for Rehearsal-Free Class-Incremental Learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 30534– 30544.

Hendrycks, D.; Basart, S.; Mu, N.; Kadavath, S.; Wang, F.; Dorundo, E.; Desai, R.; Zhu, T.; Parajuli, S.; Guo, M.; et al. 2021a. The many faces of robustness: A critical analysis of out-of-distribution generalization. In Proceedings of the IEEE/CVF international conference on computer vision, 8340–8349.

Hendrycks, D.; Zhao, K.; Basart, S.; Steinhardt, J.; and Song, D. 2021b. Natural adversarial examples. In Proceedings of

the IEEE/CVF conference on computer vision and pattern recognition, 15262–15271.

Houlsby, N.; Giurgiu, A.; Jastrzebski, S.; Morrone, B.; De Laroussilhe, Q.; Gesmundo, A.; Attariyan, M.; and Gelly, S. 2019. Parameter-eficient transfer learning for NLP. In International conference on machine learning, 2790–2799. PMLR.

Krause, J.; Stark, M.; Deng, J.; and Fei-Fei, L. 2013. 3D Object Representations for Fine-Grained Categorization. In Proceedings ofthe IEEE International Conference on Computer Vision Workshops, 554–561.

Lai, G.; Li, Y.; Wang, X.; Zhang, J.; Li, T.; and Yang, X. 2025. Order-Robust Class Incremental Learning: Graph-Driven Dynamic Similarity Grouping. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).

Lee, J.; Hayat, M.; and Yun, S. 2025. Tripartite Weight-Space Ensemble for Few-Shot Class-Incremental Learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 15329–15338.

Li, Y.; Wang, H.; Qi, Y.; Liu, W.; and Li, R. 2025. Re-fed+: A better replay strategy for federated incremental learning. IEEE Transactions on Pattern Analysis and Machine Intelligence.

Liang, Y.-S.; and Li, W.-J. 2024. Loss decoupling for taskagnostic continual learning. Advances in Neural Information Processing Systems, 36: 11151–11167.

Liu, S.; et al. 2025. Enhancing Online Continual Learning with Plug-and-Play State Space Model and Class-Conditional Mixture of Discretization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).

Nori, M. K.; Kim, I.-M.; and Wang, G. 2025. Federated classincremental learning: A hybrid approach using latent exemplars and data-free techniques to address local and global forgetting. arXiv preprint arXiv:2501.15356.

Smith, J. S.; Karlinsky, L.; Gutta, V.; Cascante-Bonilla, P.; Kim, D.; Arbelle, A.; Panda, R.; Feris, R.; and Kira, Z. 2023. Coda-prompt: Continual decomposed attention-based prompting for rehearsal-free continual learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 11909–11919.

Sun, H.-L.; Zhou, D.-W.; Ye, H.-J.; and Zhan, D.-C. 2023. Pilot: A pre-trained model-based continual learning toolbox. arXiv preprint arXiv:2309.07117.

Sun, H.-L.; Zhou, D.-W.; Zhao, H.; Gan, L.; Zhan, D.-C.; and Ye, H.-J. 2025. Mos: Model surgery for pre-trained model-based class-incremental learning. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, 20699–20707.

Wang, H.; Lu, H.; Yao, L.; and Gong, D. 2025. Self-Expansion of Pre-trained Models with Mixture of Adapters for Continual Learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 10087–10098.

Wang, Y.; Zhou, D.-W.; and Ye, H.-J. 2025. Integrating Task-Specific and Universal Adapters for Pre-Trained Modelbased Class-Incremental Learning. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 806–816.

Wang, Z.; Zhang, Z.; Ebrahimi, S.; Sun, R.; Zhang, H.; Lee, C.-Y.; Ren, X.; Su, G.; Perot, V.; Dy, J.; et al. 2022a. Dualprompt: Complementary prompting for rehearsal-free continual learning. In European Conference on Computer Vision, 631–648. Springer.

Wang, Z.; Zhang, Z.; Lee, C.-Y.; Zhang, H.; Sun, R.; Ren, X.; Su, G.; Perot, V.; Dy, J.; and Pfister, T. 2022b. Learning to prompt for continual learning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 139–149.

Wu, Y.; Piao, H.; Huang, L.-K.; Wang, R.; Li, W.; Pfister, H.; Meng, D.; Ma, K.; and Wei, Y. 2025. SD-LoRA: Scalable Decoupled Low-Rank Adaptation for Class Incremental Learning. In Proceedings of the International Conference on Learning Representations (ICLR).

Xu, Z.; Yang, S.; Xu, B.; Shen, F.; and Zhao, J. 2025. Dual Prototypes for Adaptive Pre-Trained Model in Class-Incremental Learning. Neural Networks, 108389.

Yang, S.; Shen, F.; and Zhao, J. 2024. EntAugment: Entropy-Driven Adaptive Data Augmentation Framework for Image Classification. In European Conference on Computer Vision, 197–214. Springer.

Yang, S.; Zhang, T.; Xu, Z.; Li, P.; Xu, B.; Shen, F.; and Zhao, J. 2025. Supervised contrastive learning with prototype distillation for data incremental learning. Neural Networks, 107651.

Ye, H.-J.; Zhan, D.-C.; Li, N.; and Jiang, Y. 2019. Learning multiple local metrics: Global consideration helps. IEEE transactions on pattern analysis and machine intelligence, 42(7): 1698–1712.

Yi, H.; Xu, Z.; Tu, D.; Wang, Z.; Xu, B.; and Shen, F. 2026. Beyond Point-wise Neural Collapse: A Topology-Aware Hierarchical Classifier for Class-Incremental Learning. arXiv preprint arXiv:2605.11904.

Zang, H.; Dong, Y.; Li, L.; Yang, L.; and Wang, Y. 2026. Topology-aware Knowledge Preservation for Class-Incremental Learning. In Proceedings of the AAAI Conference on Artificial Intelligence.

Zhai, X.; Puigcerver, J.; Kolesnikov, A.; Ruyssen, P.; Riquelme, C.; Lucic, M.; Djolonga, J.; Pinto, A. S.; Neumann, M.; Dosovitskiy, A.; et al. 2019. A large-scale study of representation learning with the visual task adaptation benchmark. arXiv preprint arXiv:1910.04867.

Zhang, S.; Lv, X.; Xing, Y.; Wu, Q.; Xu, D.; and Zhang, Y. 2025. Revisiting Generative Replay for Class Incremental Object Detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 20340–20349.

Zheng, B.; Zhou, D.-W.; Ye, H.-J.; and Zhan, D.-C. 2025. Task-Agnostic Guided Feature Expansion for Class-Incremental Learning. In Proceedings of the Computer Vision and Pattern Recognition Conference, 10099–10109.

Zhou, D.-W.; Cai, Z.-W.; Ye, H.-J.; Zhan, D.-C.; and Liu, Z. 2025a. Revisiting Class-Incremental Learning with Pre-Trained Models: Generalizability and Adaptivity Are All You Need. International Journal of Computer Vision.

Zhou, D.-W.; Sun, H.-L.; Ning, J.; Ye, H.-J.; and Zhan, D.- C. 2024a. Continual learning with pre-trained models: A survey. arXiv preprint arXiv:2401.16386.

Zhou, D.-W.; Sun, H.-L.; Ye, H.-J.; and Zhan, D.-C. 2024b. Expandable Subspace Ensemble for Pre-Trained Model-Based Class-Incremental Learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 23554–23564.

Zhou, D.-W.; Wang, Q.-W.; Qi, Z.-H.; Ye, H.-J.; Zhan, D.- C.; and Liu, Z. 2024c. Class-incremental learning: A survey. IEEE Transactions on Pattern Analysis and Machine Intelligence.

Zhou, D.-W.; et al. 2025b. Dual Consolidation for Pre-Trained Model-Based Domain-Incremental Learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).

Zhu, G.; Wu, D.; Gao, C.; Wang, R.; Yang, W.; and Sang, N. 2025. Adaptive Prototype Replay for Class Incremental Semantic Segmentation. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, 10932–10940.