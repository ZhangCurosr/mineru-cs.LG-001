# GEAR: Generative Expansion and Real Anchoring for Two-Stage Distillation of Tabular Foundation Models

Qi Qin<sup>1,2∗†</sup>, Jiajie Zhu<sup>2∗</sup>, Dali Chen<sup>3</sup>, Yuzhao Zhang<sup>2</sup>, Jia-Xing Han<sup>2</sup>, Yu Su<sup>2</sup>, Peng Zhang<sup>2</sup>, Ying Yan<sup>2</sup>, Yifan Sun<sup>1‡</sup>

<sup>1</sup>Renmin University of China, Center for Applied Statistics and School of Statistics, Beijing, China <sup>2</sup>Ant Digital Technologies, Ant Group, Hangzhou, China <sup>3</sup>Nanjing University, Nanjing, China

## Abstract

Tabular foundation models (TFMs) achieve strong performance through in-context learning, but context-dependent inference imposes substantial latency and memory costs, hindering large-scale deployment. We propose GEAR (Generative Expansion and Real Anchoring), a modular two-stage framework that distills TFMs into lightweight MLP or treebased predictors that can be deployed on commodity CPUs. Stage 1 uses synthetic covariates solely as teacher-query locations and trains the student on soft TFM targets, expanding coverage beyond observed rows. Stage 2 re-anchors the student to the target distribution using real labels and out-of-fold teacher predictions, which avoids self-labeling leakage. We further derive a risk certificate characterizing the trade-of between generated-query volume and generator fidelity. Experiments on TALENT and TabArena demonstrate the broad applicability of GEAR. Two-stage MLPs outperform supervised MLPs by 1.81–2.00 AUC points on binary tasks and 1.19–1.35 points on multiclass tasks, with additional gains over real-data-only distillation of 1.76–2.19 and 2.09–2.40 points, respectively. On binary tasks, the gains also transfer to LightGBM and XGBoost, and all three student families outperform CatBoost, the strongest non-TFM baseline, in mean AUC. Ablations show gains beyond longer training or alternative warm starts, greater stability from staged than mixed optimization, and generator-dependent diminishing returns as query volume increases. Finally, GEAR reduces median inference time by 57–2866× and peak prediction memory by 1.9–3.3×, while retaining higher AUC than matched supervised baselines.

## 1 Introduction

Tabular data are central to machine learning in finance, marketing, healthcare, and other high-stakes domains (Grinsztajn, Oyallon, and Varoquaux 2022). Unlike natural language and images, tables are mixed-type, heterogeneous, and dataset-specific, making reusable representations dificult to learn and deploy (Jiang et al. 2026a). Practical tabular learning has therefore relied on task-specific preprocessing and model selection; gradient-boosted decision trees such as XGBoost (Chen and Guestrin 2016), LightGBM (Ke et al. 2017), and CatBoost (Prokhorenkova et al. 2018) remain highly competitive with more complex neural architectures (Van Breugel and Van Der Schaar 2024). Recent tabular foundation models (TFMs), including TabPFN (Hollmann et al. 2022, 2025), TabICL (Qu et al. 2025, 2026), and TabDPT (Ma et al. 2024b), challenge this paradigm through transformer-based in-context learning (ICL). Given a labeled table as context, they predict query rows without task-specific gradient updates. By learning across large collections of synthetic tabular tasks, these models achieve strong few-shot performance.

![](images/bf66f4caae3255ec65debee891ae24beaa7b8977b6a623d3a9bddc8612e1a31a.jpg)  
Figure 1: Accuracy–eficiency trade-of on TALENT. Mean AUC is plotted against median prediction time; bubble size denotes peak RSS. Shapes denote TFM families, colors denote model types, crosses mark supervised baselines, light fills indicate real-only distillation, and dark fills indicate GEAR. TabM real-only distillation reaches only ≈ 0.76 AUC and is omitted; GEAR students retain teacher accuracy at much lower cost.

This in-context interface, however, creates a deployment bottleneck (Tanna et al. 2026b; Zabërgja et al. 2026). During inference, each query batch requires executing a large teacher while conditioning on the labeled context table, so latency and memory consumption can grow sharply with the context size and query workload. Large-context inference is therefore expensive, prone to out-of-memory failures, and requires training records to remain available at prediction time, raising both eficiency and privacy concerns.

![](images/772c577df3d6b97f9cc0b8af5f11467fcdfcbcf81607c1fc5fca408e06b8a33e.jpg)  
Figure 2: Overview of the proposed two-stage generative distillation protocol.

Recent work mitigates this cost by modifying the context used at inference time, including retrieval or nearestneighbor context selection (Thomas et al. 2024; Liu and Ye 2025), mixture-of-experts routing (Xu et al. 2025), divideand-conquer inference (Ye, Liu, and Chao 2026), and optimal transport-based distillation (Lin and Li 2026). Although these approaches improve scalability, the predictor still depends on an inference-time context and a large TFM. Consequently, they still incur substantial memory overhead and do not provide the fixed, millisecond-scale latency of a compact, context-free model.

Knowledge distillation (KD) ofers a complementary route (Hinton, Vinyals, and Dean 2015; Tanna et al. 2026b): the teacher’s task-adaptive behavior can be amortized into a lightweight MLP or tree-based student. Once distilled, the student has fixed per-row inference cost, runs on commodity CPUs, and no longer requires the labeled context at test time. The main challenge is query coverage: the real table provides only finitely many teacher-query locations, which is especially restrictive in small-sample regimes. Moreover, teacher predictions for rows included in the teacher’s own context may sufer from self-labeling leakage, motivating out-of-fold (OOF) real-data anchoring.

To address these challenges, we introduce GEAR, a twostage protocol that performs Generative ExpAnsion and Real Anchoring for in-context TFMs, as summarized in Figure 2. Stage 1 queries a frozen teacher on generated covariates and trains the student by teacher-only imitation. Stage 2 initializes from this generated-imitation student and returns to the real table. It uses OOF teacher predictions to avoid selflabeling leakage and optimizes a mixed real-row objective that combines ground-truth labels with teacher soft targets. Thus, synthetic queries expand teacher coverage, whereas real rows provide target-domain anchoring. GEAR is modular and agnostic to generator, teacher, and student architectures, combining diverse tabular generators and TFM teachers with deployable students such as MLPs and tree models.

We evaluate GEAR on TALENT and TabArena with TabICL, TabPFN, and TabDPT teachers, four representative generators, multiple generated-to-real ratios K, and MLP or tree-based students. Across teachers, GEAR MLP students outperform supervised MLPs by 1.81–2.00 AUC points on binary tasks and 1.19–1.35 points on multiclass tasks. On binary tasks, Stage 2 improves LightGBM and XGBoost for every teacher, and all three two-stage student families achieve higher mean AUC than CatBoost, the strongest non-TFM baseline. Relative to direct TFM inference, GEAR MLPs retain higher AUC than matched supervised baselines while reducing median prediction time by 57–2866× and peak prediction memory by 1.9–3.3× (Figure 1). Volume studies, mechanism ablations, and the risk certificate support the separation of query expansion from real-data anchoring.

Our contributions are summarized as follows:

1. We introduce GEAR, a modular two-stage framework that combines generative query expansion with real-data anchoring to transfer in-context TFMs into eficient predictors while reducing self-labeling leakage and supporting diverse generator families, teacher models, and student architectures.

2. We conduct extensive experiments showing that GEAR improves over supervised learning and real-data-only distillation, while substantially reducing TFM inference costs and preserving competitive predictive performance.

3. We theoretically link student risk to generated-query volume and fidelity. Our analysis explains the diminishing returns from synthetic expansion, the efects of distribution mismatch, and the role of real-data anchoring in the two-stage design.

## 2 Methodology

## 2.1 Setup

Let P be the joint distribution of (X, Y) on $\mathcal { X } \times [ L ]$ with covariate marginal $P _ { X } .$ , where X is a covariate vector and $Y \in [ L ] = \overline { { \{ 1 , \dots , L \} } }$ is the class label. We observe a training dataset $D _ { n } = \mathsf { \bar { \{ ( X _ { i } , Y _ { i } ) \} } } _ { i = 1 } ^ { n }$ drawn independently from P. Let $\begin{array} { r } { \Delta _ { L - 1 } = \{ p \in [ 0 , 1 ] ^ { L } : \sum _ { a = 1 } ^ { L } p _ { a } = 1 \} } \end{array}$ denote the probability simplex.

In standard supervised learning, a probabilistic classifier $p _ { \theta } : \mathcal X  \Delta _ { L - 1 }$ , parameterized by $\theta \in \Theta$ , maps each covariate vector to a predictive distribution. The training data are used only to estimate θ; after fitting, the predictor evaluates new queries independently of the training table. For a supervised loss $\ell : \Delta _ { L - 1 } \times [ \bar { L } ] \to \mathbb { R } _ { + }$ , such as crossentropy, the population risk is $R \dot { ( \theta ) } = \operatorname { E } _ { P } \{ \ell ( p _ { \theta } ( X ) , Y ) \}$ , and empirical risk minimization estimates θ by minimizing its empirical counterpart.

Tabular foundation models (Hollmann et al. 2025) use a diferent, context-dependent interface. They are pretrained across many synthetic tabular tasks and avoid task-specific parameter updates at deployment. Given a labeled context C and a query x, the teacher returns $f _ { \mathrm { I C L } } ( x \mid C ) \in \Delta _ { L - 1 }$ Thus, unlike $p _ { \theta } ( x )$ , whose predictions are fixed once training is complete, $f _ { \mathrm { I C L } } ( x \mid C )$ remains coupled to the size, composition, and labels of the context table.

This context dependence creates a practical deployment burden: each prediction invokes the large teacher and requires access to the labeled table. To amortize this cost while preserving the teacher’s task-adaptive behavior, we distill the frozen teacher $f _ { T } ( x \mid C ) = f _ { \mathrm { I C L } } ( x \mid C )$ into a lightweight student $p _ { \theta } ( x ) = f _ { S } ( x ; \theta )$ . The student can be instantiated as an MLP, a tree model, or another deployable tabular model.

## 2.2 Real-Data Mixed Distillation

To distill the TFM teacher’s behavior into a lightweight parametric student, we use the standard teacher–student formulation with soft teacher targets (Hinton, Vinyals, and Dean 2015), adapted to tabular TFM distillation (Tanna et al. 2026b). For a student distribution $p$ and a teacher distribution q in $\Delta _ { L - 1 }$ , the soft-label loss is $\ell _ { T } ( p , q ) \ =$ $\begin{array} { r } { - \sum _ { a = 1 } ^ { L } q _ { a } \log p _ { a } ^ { 1 } } \end{array}$ . With mixing weight $\lambda \in [ 0 , 1 ]$ , the realrow mixed loss is

$$
\ell _ { \lambda } ( p , y , q ) = ( 1 - \lambda ) \ell ( p , y ) + \lambda \ell _ { T } ( p , q ) .
$$

With a single fixed teacher context $C ,$ , the population loss would be $\operatorname { E } _ { P } \ell _ { \lambda } \{ p _ { \boldsymbol { \theta } } ( \boldsymbol { X } ) , \boldsymbol { Y } , f _ { T } ( \boldsymbol { X } \mid \boldsymbol { C } ) \}$ . On real training rows, however, asking the teacher to predict a row that is also inside its context can leak that row’s label into the distillation target. We therefore use out-of-fold (OOF) teacher predictions, following the real-data TFM distillation baseline of Tanna et al. (2026b).

Specifically, we partition $D _ { n }$ into $K _ { \mathrm { c f } }$ equal folds $I _ { 1 } , \ldots , I _ { K _ { \mathrm { c f } } }$ of size $m = n / K _ { \mathrm { c f } }$ . For fold $I _ { k } ,$ let $D _ { n } ^ { ( - k ) } =$ $D _ { n } \ \backslash \{ ( X _ { i } , Y _ { i } ) : i \in I _ { k } \}$ and define the fold teacher map $T _ { k } ( \cdot ) = f _ { T } ( \cdot \mid D _ { n } ^ { ( - k ) } )$ . The fold-averaged OOF population loss is

$$
L _ { \lambda } ^ { P } ( \theta ) = \frac { 1 } { K _ { \mathrm { c f } } } \sum _ { k = 1 } ^ { K _ { \mathrm { c f } } } \mathrm { E } _ { ( X , Y ) \sim P } \ell _ { \lambda } \{ p _ { \theta } ( X ) , Y , T _ { k } ( X ) \} .\tag{1}
$$

The matching empirical OOF objective is

$$
\widehat { L } _ { \lambda , n } ( \theta ) = \frac { 1 } { K _ { \mathrm { c f } } } \sum _ { k = 1 } ^ { K _ { \mathrm { c f } } } \frac { 1 } { m } \sum _ { i \in I _ { k } } \ell _ { \lambda } \{ p _ { \theta } ( X _ { i } ) , Y _ { i } , T _ { k } ( X _ { i } ) \} .\tag{2}
$$

The OOF construction prevents self-labeling by ensuring that the teacher target for a real row is produced from a context excluding that row. Unless stated otherwise, all realdata teacher predictions below use this OOF construction. The resulting real-only OOF mixed baseline, which uses hard labels and OOF teacher predictions on real rows only, is

$$
\widehat { \theta } _ { R } \in \mathop { \mathrm { a r g } } \operatorname* { m i n } _ { \theta \in \Theta } \widehat { L } _ { \lambda , n } ( \theta ) .\tag{3}
$$

## 2.3 Synthetic Covariate Augmentation

Synthetic data ofer a natural means of alleviating data scarcity and can improve learning through pretraining and fine-tuning (Bühler, Purucker, and Hutter 2026). GEAR uses generated rows more conservatively: they provide additional covariate locations for querying the frozen teacher. To avoid bias from generator-provided labels, Stage 1 discards generated labels and uses only generated covariates with teacher soft targets. Concretely, we draw covariates from a tabular generator, such as a GAN or difusion model (Xu et al. 2019; Shi et al. 2025a), and never treat them as labeled target examples.

Let $\widetilde { D } _ { M } = \{ \widetilde { X } _ { j } \} _ { i = 1 } ^ { M }$ , where $\widetilde { X } _ { j } \stackrel { \mathrm { i . i . d . } } { \sim } Q _ { X }$ and $Q _ { X }$ denotes the generated covariate law. For generated queries, we write $T _ { s } ( \bar { x } ) = f _ { T } ( x \mid D _ { n } )$ for the teacher response. The corresponding population and empirical teacher-imitation objectives are

$$
\begin{array} { r l } & { { \cal L } _ { s } ^ { Q _ { \cal X } } ( \theta ) = { \cal { E } } _ { \widetilde { { \cal X } } \sim Q _ { \cal X } } \ell _ { T } \{ p _ { \theta } ( \widetilde { { \cal X } } ) , T _ { s } ( \widetilde { { \cal X } } ) \} , } \\ & { \widehat { \cal L } _ { s , M } ^ { Q _ { \cal X } } ( \theta ) = \displaystyle \frac { 1 } { M } \sum _ { j = 1 } ^ { M } \ell _ { T } \{ p _ { \theta } ( \widetilde { { \cal X } } _ { j } ) , T _ { s } ( \widetilde { { \cal X } } _ { j } ) \} . } \end{array}\tag{4}
$$

This stage is most useful when generated volume and fidelity are balanced. A larger M provides more teacher-query locations, whereas a low-fidelity $Q _ { X }$ can shift imitation away from the real covariate law ${ \dot { P } } _ { X }$ . Section 3 formalizes this trade-of as a finite-query term plus a generator-fidelity floor.

## 2.4 Two-Stage Distillation

To exploit broad synthetic coverage while preserving realdata anchoring, GEAR adopts a two-stage protocol. Stage 1 pretrains the student using generated covariates and teacher soft targets. Stage 2 then returns to the real table and optimizes an OOF mixed objective that combines hard labels with OOF teacher predictions, correcting the Stage 1 student toward the target joint distribution $\breve { P } .$ . From a transferlearning perspective, Stage 1 acquires knowledge from auxiliary generated data, whereas Stage 2 performs real-data anchoring in the target domain. Abundant synthetic queries reduce the estimation burden, whereas real samples mitigate distribution shift and align the student with the target distribution (Zhuang et al. 2020; Li, Cai, and Li 2022).

Given a center $\theta _ { 0 } ,$ let $\Theta _ { 2 } ( \theta _ { 0 } ) \subseteq \Theta$ denote the implementation’s Stage 2 search region around $\theta _ { 0 } .$ This region may be induced by initialization, early stopping, or regularization. Algorithm 1 summarizes the complete pipeline from generated teacher queries to OOF real-data anchoring.

Stage 1 performs pure teacher imitation on generated covariates:

$$
\widehat { \theta } _ { 1 } \in \mathop { \mathrm { a r g } } _ { \theta \in \Theta } \operatorname* { m i n } _ { } \widehat { L } _ { s , M } ^ { Q _ { X } } ( \theta ) .\tag{5}
$$

Stage 2 starts from $\widehat { \theta } _ { 1 }$ and returns to the real labeled sample. It optimizes the OOF mixed objective in (2), anchoring the Stage 1 student to real labels and OOF teacher predictions within the implementation’s Stage 2 search region:

$$
\widehat { \theta _ { 2 } } \in \mathop { \mathrm { a r g } } _ { \theta \in \Theta _ { 2 } ( \widehat { \theta } _ { 1 } ) } \widehat { L } _ { \lambda , n } ( \theta ) .\tag{6}
$$

The restricted Stage 2 search region does not guarantee a lower empirical OOF loss than an unrestricted real-only fit. Instead, initializing at $\widehat { \theta } _ { 1 }$ provides a teacher-informed starting point that can facilitate optimization toward a favorable target-domain solution relative to random initialization.

Algorithm 1 Two-stage generative distillation with OOF   
real-data anchoring.   
Input: real table $D _ { n } = \{ ( X _ { i } , Y _ { i } ) \} _ { i = 1 } ^ { n }$ , frozen teacher   
$f _ { T } ,$ , student $f _ { S } ( \cdot ; \theta )$   
Input: generated pool size M, fold count $K _ { \mathrm { c f } } .$ , losses   
$\ell , \ell _ { T } ,$ , weight λ, Stage 2 search rule $\Theta _ { 2 } .$   
Output: two-stage student $\widehat { \theta } _ { 2 }$   
Stage 1: generated teacher imitation.   
1: Draw generated covariates $\widetilde { X } _ { 1 } , \dots , \widetilde { X } _ { M } \sim Q _ { X } .$   
2: for $j \doteq 1 , \ldots , M$ do   
Query $f _ { T } ( \widetilde { X } _ { j } \mid D _ { n } ) .$   
4: end for   
5: $\widehat { \theta } _ { 1 } \gets \arg \operatorname* { m i n } _ { \theta \in \Theta } \widehat { L } _ { s , M } ^ { Q _ { X } } ( \theta ) .$   
Stage 2: OOF real-data anchoring.   
6: Partition $D _ { n }$ into folds $I _ { 1 } , \ldots , I _ { K _ { \mathrm { c f } } } .$   
7: for $k = 1 , \ldots , K _ { \mathrm { c f } }$ do   
8: Let $D _ { n } ^ { ( - k ) } = D _ { n } \setminus \{ ( X _ { i } , Y _ { i } ) : i \in I _ { k } \}$   
9: for each $i \in I _ { k }$ do   
10: Query $f _ { T } ( X _ { i } \mid D _ { n } ^ { ( - k ) } )$   
11: end for   
12: end for   
13: Define $\widehat { L } _ { \lambda , n }$ by mixing real-label and OOF teacher terms   
as in (2).   
14: Initialize at $\widehat { \theta } _ { 1 }$ and optimize the OOF mixed objective   
within $\Theta _ { 2 } ( \widehat { \theta } _ { 1 } ) \colon$   
15: $\widehat { \theta } _ { 2 } \gets \mathrm { a r g m i n } _ { \theta \in \Theta _ { 2 } ( \widehat { \theta } _ { 1 } ) } \widehat { L } _ { \lambda , n } ( \theta )$   
16: return ${ \widehat { \theta } } _ { 2 } .$

## 3 Theoretical Guarantee

This section presents a compact risk certificate for the generated-imitation part of GEAR. The baseline is the realonly OOF mixed estimator ${ \widehat { \theta } } _ { R }$ in (3), and the target criterion is the population risk $R ( \theta ) = \operatorname { E } _ { P } \{ \ell ( p _ { \theta } ( X ) , Y ) \}$ . The certificate characterizes when generated-only teacher imitation can outperform the real-only baseline.

We use the empirical objectives $\widehat { L } _ { s , M } ^ { Q _ { X } }$ and $\widehat { L } _ { \lambda , n }$ for synthetic teacher imitation and real-data OOF distillation, respectively. The conversion gaps quantify surrogate-to-target bias. The terms $\epsilon _ { s }$ and $\epsilon _ { \lambda } ^ { \mathrm { o o f } }$ are finite-query and cross-fitting radii. Here, TV denotes total variation, $\mathring { B } \mathrm { T V } ( P _ { X } , Q _ { X } )$ is the generator-fidelity floor, and $\underline { { R } } _ { R }$ is a conservative lower certificate for the real-only baseline. Formal definitions appear in the supplementary material.

Theorem 1 (Generated Stage 1 certificate). Under the regularity conditions stated in the supplementary material, with probability at least $1 - \delta ,$

$$
\begin{array} { r l } & { R ( \widehat { \theta } _ { 1 } ) - R ( \widehat { \theta } _ { R } ) \leq \widehat { L } _ { s , M } ^ { Q _ { X } } ( \widehat { \theta } _ { 1 } ) - \widehat { L } _ { \lambda , n } ( \widehat { \theta } _ { R } ) } \\ & { \qquad + \epsilon _ { s } ( \Theta , M , \delta / 2 ) + \epsilon _ { \lambda } ^ { \mathrm { o o f } } ( \Theta , n , \delta / 2 ) } \\ & { \qquad + \mathfrak { b } _ { s } ( \Theta ) + \mathfrak { b } _ { \lambda } ( \Theta ) . } \end{array}\tag{7}
$$

When the right-hand side of (7) is negative, the certificate guarantees that generated-only Stage 1 outperforms the realonly baseline in target risk. The theorem separates two factors that determine whether additional generated rows help.

More queries reduce the finite-query price, but they do not remove a persistent mismatch between generated and real covariates. The following specialization makes this volume– fidelity trade-of explicit.

Corollary 1 (Fidelity–volume specialization). Suppose the generated-query radius satisfies

$$
\boldsymbol { \epsilon } _ { s } ( \Theta , M , \delta ) \leq \boldsymbol { \mathfrak { C } } _ { s } ( \Theta , \delta ) M ^ { - 1 / 2 }\tag{8}
$$

for a class-dependent constant ${ \mathfrak { C } } _ { s } ( \Theta , \delta )$ . Using the TV control in the supplementary material, a suficient condition for $R ( { \widehat { \theta } } _ { 1 } ) < R ( { \widehat { \theta } } _ { R } )$ is

$$
\begin{array} { r l r } & { } & { \boldsymbol { B } \mathrm { T V } ( \boldsymbol { P } _ { X } , \boldsymbol { Q } _ { X } ) + 2 \mathfrak { C } _ { s } ( \Theta , \delta / 2 ) \boldsymbol { M } ^ { - 1 / 2 } } \\ & { } & { < \underline { { R } } _ { R } ( \delta / 2 ) - \displaystyle \operatorname* { i n f } _ { \theta \in \Theta } L _ { s } ^ { Q _ { X } } ( \theta ) - \beta _ { s } ( \Theta ) . } \end{array}\tag{9}
$$

Corollary 1 isolates the fidelity–volume price

$$
H _ { 1 } ( M ) = B \mathrm { T V } ( P _ { X } , Q _ { X } ) + 2 \mathfrak { C } _ { s } ( \Theta , \delta / 2 ) M ^ { - 1 / 2 } .\tag{10}
$$

The $M ^ { - 1 / 2 }$ term decays with generated volume, whereas $B \mathrm { T V } ( P _ { X } , Q _ { X } )$ is an irreducible fidelity floor intrinsic to the generator. When $Q _ { X }$ closely matches $P _ { X }$ so that $\mathrm { T V } ( P _ { X } , Q _ { X } ) \approx 0 .$ , increasing M progressively tightens the certificate until it reaches this fidelity floor. This yields a scaling law for synthetic expansion: the certifiable finitequery error decays as $M ^ { - 1 / 2 }$ only while sampling error is the primary bottleneck. When $Q _ { X }$ is poorly matched to $P _ { X }$ additional queries cannot mitigate the resulting distribution shift.

Stage 2 addresses the source-induced bias inherited from generated-query imitation. In transfer-learning terms, Stage 1 uses source-like synthetic coverage to reduce the estimation burden and provide an informed initialization, whereas Stage 2 uses real labels and OOF teacher predictions to debias the student under the target distribution. This correction is essential for controlling target risk when $Q _ { X } \neq P _ { X }$ (Cortes and Mohri 2014; Li, Cai, and Li 2022).

## 4 Experiments

## 4.1 Experimental Setup

We evaluate on the TALENT (Liu et al. 2025) and TabArena (Erickson et al. 2026) benchmarks. For all settings, we use ROC-AUC for binary classification and macro one-vs-rest AUC for multiclass classification; reported values are averaged over five random runs. The generated-toreal sample ratio for Stage 1 is swept over $\breve { K } = M / n \in$ {1, 5, 10, 20, 30, 40}, where M is the number of generated covariates and n is the real training size.

We use TabICL v2 (Qu et al. 2026), TabPFN v3 (Grinsztajn et al. 2026)<sup>2</sup>, and TabDPT v1.2 (Ma et al. 2024b) as teacher models. The default student is an MLP, and the studentfamily ablation evaluates LightGBM and XGBoost instead. We additionally benchmark against three strong supervised tabular baselines: CatBoost, RealMLP (Holzmüller, Grinsztajn, and Steinwart 2024), and TabM (Gorishniy, Kotelnikov, and Babenko 2025). The primary Stage 1 generator is TabPFGen (Ma et al. 2024a), an energy-based in-context tabular generator that conditions on the observed table and produces synthetic covariates without fitting a separate taskspecific generator. We also evaluate Copula, CTGAN (Xu et al. 2019), and TabDif (Shi et al. 2025a), covering a dependence-preserving statistical generator, a GAN-based generator, and a difusion-based generator.

We compare the following procedures. The teacher model is the frozen TFM evaluated directly with the real training table as context. The supervised student is trained only on real data with hard labels. The synthetic-pretraining student is trained on generated covariates with soft teacher targets in Stage 1. The real-data distillation baseline uses only real rows with out-of-fold teacher predictions and hard labels (Tanna et al. 2026b). The two-stage student (GEAR) in Section 2.4 starts from synthetic pretraining and then applies the real-data anchoring step.

Implementation and eficiency. The default student is a residual MLP with LayerNorm and SiLU activations, hidden width 512, four blocks, and dropout 0.2; Stage 2 uses a softlabel weight of $\lambda = 0 . 7$ . Timing experiments use a server with a single NVIDIA A800-SXM GPU and two Intel Xeon Platinum 8369B CPUs. We time direct TFM inference on the GPU and student inference on the CPU and, where supported, on the same GPU. Detailed optimization settings and implementation are provided in the supplementary material.

## 4.2 Main Synthetic-Volume Study

Q1: Does distillation improve a deployable MLP? Figure 3 compares four generators on matched datasets and reports AUC-point gains relative to the supervised MLP; thus, zero denotes the supervised baseline, whereas the real-data distillation line represents the performance attainable without generated covariates. Across teacher–task panels, the trajectories exhibit a consistent pattern: increasing K improves Stage 1 until the gains plateau at a generator-dependent level, while Stage 2 generally improves the corresponding Stage 1 solution through real-data anchoring.

For TabPFGen, the two-stage MLP at K = 40 achieves mean AUC gains of 1.91 points on binary tasks and 1.27 points on multiclass tasks over the supervised MLP. As K increases, the marginal benefit of additional generated rows diminishes. From K = 1 to K = 10, mean Stage 1 AUC rises by 0.98 points on binary tasks and 0.86 points on multiclass tasks, whereas the increase from K = 10 to K = 40 is only 0.29 and 0.17 points, respectively. This saturation is consistent with the volume–fidelity trade-of: additional teacher queries reduce finite-sample error, whereas real-data anchoring is required to mitigate the generated–real distribution mismatch.

Q2 (Stage 1 Analysis): How much does the generator matter? Generator choice materially afects both the Stage 1 plateau and the benefit available from Stage 2.

TabPFGen and TabDif produce richer energy-based and difusion-based query distributions; their stronger Stage 1 performance is often accompanied by larger two-stage gains. Copula is a useful low-capacity stress test because it preserves a simple dependence structure, but it is not designed to learn complex structural dependencies or causal relationships. CTGAN typically lies between these extremes. As K grows, Stage 1 with higher-fidelity generators can surpass the real-only reference, although weaker generators need more generated rows to reach the same range. By contrast, in the TabICL and TabPFN panels, Copula often remains below real-only distillation even with larger generated pools. This pattern matches the fidelity floor in Corollary 1.

<table><tr><td>Task</td><td>TabPFGen</td><td>Copula</td><td>CTGAN</td><td>TabDiff</td></tr><tr><td>Bin.</td><td></td><td>+0.87 / 0.04 +0.81 /0.19 +0.71 / 0.10 +1.18/</td><td></td><td rowspan="4">&lt; 0.01</td></tr><tr><td></td><td></td><td>Multi. +0.26 / 0.05 −0.05/ 0.08 −0.17/ 0.28</td><td>+0.18/0.02</td></tr><tr><td></td><td></td><td></td><td></td></tr></table>

Table 1: GEAR MLP versus CatBoost at $K \ = \ 4 0 .$ . Cells report mean $\Delta _ { \mathrm { G E A R - C B } } .$ / two-sided Wilcoxon p-value.

Stage 2 usually improves its corresponding Stage 1 curve because it re-optimizes on real rows, but it is not an automatic repair step. If synthetic teacher imitation pushes the student toward a low-performing region, OOF correction on real rows has limited room to recover. Generator quality therefore affects both Stage 1 accuracy and the usefulness of subsequent real-data anchoring. At $K = 4 0$ , Table 1 reports mean paired AUC-point diferences and two-sided Wilcoxon p-values after averaging available teacher results per task. The fractions of positive paired diferences are 57.7–69.8% (binary) and 60.9–64.4% (multiclass); TabDif shows the broadest positive shift, whereas the Copula and CTGAN multiclass means are negative despite positive medians.

Q3 (Stage 1 Analysis): Are generated teacher queries really helpful? One possibility is that Stage 2 benefits only from the initialization supplied by pretraining or from a longer training schedule. We therefore retain the same twostage protocol and compare GEAR with five alternatives: real-only mixed distillation, synthetic-only soft distillation, real-only soft distillation, supervised MLP initialization, and real-only soft initialization. The first three use, respectively, real rows with mixed soft and hard targets, generated rows with soft teacher targets only, and real rows with soft teacher targets only. The final two replace synthetic pretraining with a supervised MLP warm start or a real-only soft-distillation warm start before the same real-data second stage. If synthetic pretraining merely provided an arbitrary initialization or if the gain came solely from a longer schedule, these alternatives should perform comparably. Figure 4 shows a representative comparison with TabPFN as the teacher. The full result appears in the supplementary material.

The results answer Q3 afirmatively. For TabDif with a TabPFN teacher on binary tasks, GEAR improves over realonly mixed distillation by 2.61 AUC points and wins on 92.6% of the paired datasets. It also exceeds supervised MLP initialization by 1.46 points. On multiclass tasks, GEAR gains 2.38 points over real-only mixed distillation and wins on 94.2% of the paired datasets. Comparisons with supervised MLP initialization and real-only soft initialization show that neither a generic warm start nor additional training explains the gains. Synthetic pretraining yields stronger Stage 2 performance under comparable budgets. Moreover, real-only soft distillation underperforms synthetic-only soft distillation, indicating that generated covariates expand teacherquery coverage rather than merely repeating distillation on observed rows. Thus, the generated stage supplies informative query locations, while the real stage restores targetdomain alignment.

![](images/76e2f1c180df45495e99e1773eec68532b6e22561b09b9da6a52ad2a975f870c.jpg)  
Figure 3: Generator trajectories on TALENT. Columns correspond to TabICL, TabPFN, and TabDPT, and rows to binary and multiclass tasks. Curves show mean gains in AUC points over the supervised MLP across generated-to-real ratios K. Colors denote generators; dashed circles and solid squares indicate Stage 1 and Stage 2, respectively. Horizontal lines mark the teacher, supervised-MLP, and real-only-distillation references.

![](images/942f9ebfcbf0efab536efe58a2393bdcc385c67d8b1f53ec7bf03ba57c333314.jpg)

![](images/976349abf72c248367a255d9c6ad66ad9a4d1815659fba353b9c1942a5b502e5.jpg)  
Figure 4: Stage 1 replacement ablations using TabPFN as the teacher. Boxplots report paired gains in AUC points of GEAR over five alternatives, ordered from real-only mixed distillation to real-only soft initialization.

Q4: Does GEAR generalize across student families? This study replaces the MLP student with LightGBM and XGBoost while retaining the same two-stage logic. The treebased student study uses TabPFGen as its generator: Stage 1 first trains the tree student on generated teacher targets, and Stage 2 continues training from that Stage 1 student using real-data anchoring. Table 2 separates three paths: Stage 1- only synthetic teacher imitation, real-data distillation from scratch, and the full two-stage route.

<table><tr><td>Teacher Task</td><td>Student</td><td>S1 ∆</td><td>Real ∆</td><td>S2∆</td></tr><tr><td rowspan="2">TabICL</td><td>LGBM Bin. XGB</td><td> $+ 0 . 7 9 _ { ( 0 . 2 5 ) }$   $+ 0 . 9 6 _ { ( 0 . 2 1 ) }$ </td><td> $+ 0 . 9 2 _ { ( 0 . 2 4 ) } + 1 . 5 3 _ { ( 0 . 2 0 ) }$   $+ 1 . 1 1 _ { ( 0 . 1 5 ) } + 1 . 3 7 _ { ( 0 . 1 8 ) }$ </td><td rowspan="2"></td></tr><tr><td>LGBM Multi.</td><td> $- 6 . 3 4 _ { ( 1 . 1 4 ) }$ </td><td> $- 8 . 6 9 _ { ( 1 . 0 4 ) } - 5 . 9 3 _ { ( 1 . 1 4 ) }$ </td></tr><tr><td rowspan="4">TabPFN</td><td>Bin.</td><td>XGB LGBM</td><td> $+ 0 . 2 2 \dot { ( 0 . 2 2 ) }$   $- 2 . 5 7 _ { ( 0 . 7 6 ) }$ </td><td> $+ 0 . 4 0 _ { ( 0 . 1 3 ) } + 0 . 5 1 _ { ( 0 . 1 2 ) }$   $+ 0 . 9 1 _ { ( 0 . 2 4 ) } + 1 . 3 2 _ { ( 0 . 2 0 ) }$ </td></tr><tr><td></td><td>XGB LGBM</td><td> $- 2 . 7 1 _ { ( 0 . 9 0 ) } \ + 1 . 1 4 _ { ( 0 . 1 6 ) } + 1 . 3 8 _ { ( 0 . 1 8 ) }$   $- 1 0 . 3 9 _ { ( 1 . 6 2 ) }$ </td><td></td></tr><tr><td>Multi.</td><td>XGB</td><td> $- 5 . 2 8 _ { ( 1 . 4 5 ) }$ </td><td> $- 8 . 9 6 _ { ( 1 . 1 2 ) } - 6 . 0 2 _ { ( 1 . 2 2 ) }$   $+ 0 . 1 5 _ { ( 0 . 2 5 ) } + 0 . 4 4 _ { ( 0 . 1 2 ) }$ </td></tr><tr><td>LGBM</td><td> $+ 0 . 9 0 _ { ( 0 . 2 4 ) }$ </td><td></td><td></td></tr><tr><td>TabDPT Multi.</td><td>Bin.</td><td>XGB LGBM</td><td></td><td> $+ 0 . 6 5 _ { ( 0 . 2 7 ) } + 1 . 1 1 _ { ( 0 . 2 3 ) }$   $+ 0 . 6 9 _ { ( 0 . 2 0 ) } \ + 0 . 8 2 _ { ( 0 . 2 0 ) } + 0 . 9 8 _ { ( 0 . 1 8 ) }$ </td><td> $- 2 . 5 0 _ { ( 0 . 5 5 ) } - 1 . 7 7 _ { ( 0 . 4 2 ) } \ - 2 . 1 7 _ { ( 0 . 4 8 ) }$   $- 0 . 3 6 _ { ( 0 . 2 9 ) } ^ { } ~ - 0 . 2 0 _ { ( 0 . 2 6 ) } ^ { } - 0 . 1 1 _ { ( 0 . 2 2 ) } ^ { }$ </td></tr></table>

Table 2: Tree-student distillation with TabPFGen. Values are paired gains in AUC points over supervised training; subscripts denote standard errors.

Because tree models are non-diferentiable and poorly suited to fine-tuning, distillation is challenging; nevertheless, two-stage training improves LightGBM by 1.11–1.53 AUC points and XGBoost by 0.98–1.38 points over supervised training across all three teachers. Aggregated across datasets, these students also surpass CatBoost, the strongest non-TFM baseline, by 0.74 and 0.84 AUC points, respectively, showing that the gains extend beyond matched supervised comparisons. XGBoost is especially stable on multiclass tasks, with positive gains for TabICL and TabPFN and near parity for TabDPT. LightGBM is less stable on multiclass tasks, where all three distillation paths remain below supervised training, consistent with Tanna et al. (2026a). Nevertheless, for TabICL and TabPFN, two-stage training is less harmful than real-only distillation. Overall, the benefit is student-dependent. Generated teacher queries can guide a useful learning trajectory, whereas the benefit of real-data anchoring depends strongly on the student family and task type. We also evaluate TabM and MLPs with one to three blocks; shallower MLPs show a larger two-stage advantage. Complete results are provided in the supplementary material.

![](images/33e2c8a894f576121b5a0066765b9dac1e7785cc3814800eee035287be80620f.jpg)  
Figure 5: Two-stage versus mixed real–generated distillation on TALENT binary tasks. Rows denote TFM teachers, and columns denote TabPFGen and TabDif. Solid curves show paired median gains in AUC points over Stage 1 with interquartile bands; dashed hollow-marker curves show nondegradation rates relative to Stage 1 (right axis).

## 4.3 Ablation Studies

Staged optimization versus mixed training. This ablation tests whether the generated and real objectives can be optimized jointly. The mixed real–generated baseline combines generated teacher targets with OOF real-data anchoring but does not separate query expansion from real-data anchoring. Figure 5 reports paired results across K on TAL-ENT binary tasks, including median AUC gains and nondegradation rates relative to Stage 1; each rate is the fraction of paired tasks where the corresponding sequential or mixed method is at least as accurate as Stage 1.

Figure 5 shows representative TabPFGen and TabDif comparisons; the full sweep appears in the supplementary material. With TabPFGen, two-stage training has higher median gains in AUC points than mixed training for both teachers at every K, while the mixed approach approaches or falls below Stage 1 as K grows. Mixed training can be competitive for other generators, but two-stage training achieves nondegeneration rates that are 4.2–48.7 percentage points higher in every comparison. Separating query expansion from realdata anchoring therefore improves stability.

Soft-label mixing. We examine GEAR’s sensitivity to the weighting parameter λ, with full results reported in the supplementary material. In Stage 1, pure or near-pure teacher supervision outperforms training with generated hard labels, confirming that generated rows are more useful as teacherquery locations than as labeled examples. In Stage 2, sensitivity to λ is weaker but non-negligible: teacher soft targets remain beneficial when balanced with real labels, thereby anchoring the objective to the target distribution.

## 5 Related Work

Tabular foundation models. Gradient-boosted trees and tuned neural architectures remain strong tabular baselines (Chen and Guestrin 2016; Ke et al. 2017; Gorishniy et al. 2021; Grinsztajn, Oyallon, and Varoquaux 2022). TabPFN introduced context-based prediction from a labeled table (Hollmann et al. 2022); subsequent TFMs improve pretraining strategies, extend context length, or incorporate more structured-data modeling (Ma et al. 2024b; Qu et al. 2026; Grinsztajn et al. 2026; Wang et al. 2026). Scalability methods retrieve, adapt, chunk, or optimize the context supplied to a TFM (Thomas et al. 2024; Liu and Ye 2025; Sergazinov and Yin 2025; Chen, Ding, and Akoglu 2026).

Distilling tabular foundation models. Knowledge distillation compresses powerful predictors into deployable models (Bucilua, Caruana, and Niculescu-Mizil 2006; Hinton,ˇ Vinyals, and Dean 2015). TFM-specific approaches include in-context data distillation, MotherNet, TabDistill, end-toend compression, real-data-only distillation, and interaction distillation (Ma et al. 2024c; Müller, Curino, and Ramakrishnan 2025; Dissanayake and Dutta 2025; Zabërgja et al. 2026; Tanna et al. 2026b; Jia et al. 2026). GEAR difers by separating generated teacher-query expansion from real-data anchoring, which connects it to sample-bias correction under distribution shift (Cortes and Mohri 2014).

Synthetic tabular data generation. Synthetic tabular generators include CTGAN, difusion models, and TFM-based generators (Xu et al. 2019; Kotelnikov et al. 2023; Zhang et al. 2024; Shi et al. 2025a; Ma et al. 2024a; Jiang et al. 2026b). Their utility depends on generator quality, task type, and evaluation protocol (Shi et al. 2025b; Challagundla et al. 2025); accordingly, GEAR uses synthetic rows as teacherquery locations rather than as labeled target examples.

## 6 Conclusion

We present GEAR, a modular two-stage framework that distills in-context tabular foundation models into eficient, context-free predictors through generative query expansion and real-data anchoring. In the main MLP experiments, GEAR improves over supervised learning and real-data-only distillation while substantially reducing TFM inference costs. Our risk certificate further explains the trade-of between query volume and generator fidelity.

Although this work focuses on classification and singleteacher distillation, GEAR could be extended to regression and multi-teacher distillation. Promising directions include fidelity-aware query generation and combining heterogeneous generators. Another direction is to introduce adversarial distribution alignment during distillation, encouraging generated queries to expand teacher coverage while remaining close to the real-data distribution.

## References

Bucilua, C.; Caruana, R.; and Niculescu-Mizil, A. 2006.ˇ Model compression. In Proceedings of the 12th ACM SIGKDD international conference on Knowledge discovery and data mining, 535–541.

Bühler, M.; Purucker, L.; and Hutter, F. 2026. Causal Data Augmentation for Robust Fine-Tuning of Tabular Foundation Models. arXiv preprint arXiv:2601.04110.

Challagundla, R.; Dorodchi, M.; Wang, P.; and Lee, M. 2025. Synthetic tabular data generation: A comparative survey for modern techniques. arXiv preprint arXiv:2507.11590.

Chen, T.; and Guestrin, C. 2016. Xgboost: A scalable tree boosting system. In Proceedings of the 22nd acm sigkdd international conference on knowledge discovery and data mining, 785–794.

Chen, Y.; Ding, X.; and Akoglu, L. 2026. VIP-COP: Context Optimization for Tabular Foundation Models. arXiv preprint arXiv:2605.12904.

Cortes, C.; and Mohri, M. 2014. Domain adaptation and sample bias correction theory and algorithm for regression. Theoretical Computer Science, 519: 103–126.

Dissanayake, P.; and Dutta, S. 2025. TabDistill: Distilling Transformers into Neural Nets for Few-Shot Tabular Classification. arXiv preprint arXiv:2511.05704.

Erickson, N.; Purucker, L.; Tschalzev, A.; Holzmüller, D.; Desai, P.; Salinas, D.; and Hutter, F. 2026. Tabarena: A living benchmark for machine learning on tabular data. Advances in Neural Information Processing Systems, 38.

Gorishniy, Y.; Kotelnikov, A.; and Babenko, A. 2025. Tabm: Advancing tabular deep learning with parameter-eficient ensembling. In International Conference on Learning Representations, volume 2025, 77899–77935.

Gorishniy, Y.; Rubachev, I.; Khrulkov, V.; and Babenko, A. 2021. Revisiting deep learning models for tabular data. Advances in neural informationprocessing systems, 34: 18932– 18943.

Grinsztajn, L.; Flöge, K.; Key, O.; Birkel, F.; Jund, P.; Roof, B.; Manium, M.; Bin, S.; Bühler, M.; Garg, A.; et al. 2026. TabPFN-3: Technical Report. arXiv preprint arXiv:2605.13986.

Grinsztajn, L.; Oyallon, E.; and Varoquaux, G. 2022. Why do tree-based models still outperform deep learning on typical tabular data? Advances in neural information processing systems, 35: 507–520.

Hinton, G.; Vinyals, O.; and Dean, J. 2015. Distilling the knowledge in a neural network. arXiv preprint arXiv:1503.02531.

Hollmann, N.; Müller, S.; Eggensperger, K.; and Hutter, F. 2022. Tabpfn: A transformer that solves small tabular classification problems in a second. arXiv preprint arXiv:2207.01848.

Hollmann, N.; Müller, S.; Purucker, L.; Krishnakumar, A.; Körfer, M.; Hoo, S. B.; Schirrmeister, R. T.; and Hutter, F. 2025. Accurate predictions on small data with a tabular foundation model. Nature, 637(8045): 319–326.

Holzmüller, D.; Grinsztajn, L.; and Steinwart, I. 2024. Better by default: Strong pre-tuned mlps and boosted trees on tabular data. Advances in Neural Information Processing Systems, 37: 26577–26658.

Jia, J.; Singh, C.; Caruana, R.; and Lengerich, B. 2026. Selecting Feature Interactions for Generalized Additive Models by Distilling Foundation Models. arXiv preprint arXiv:2604.13332.

Jiang, J.-P.; Liu, S.-Y.; Cai, H.-R.; Zhou, Q.-L.; and Ye, H.-J. 2026a. Representation learning for tabular data: A comprehensive survey. IEEE Transactions on Pattern Analysis and Machine Intelligence.

Jiang, X.; Liu, M.; Simidjievski, N.; Klein, T.; and Jamnik, M. 2026b. Tabular Foundation Model for Generative Modelling. arXiv preprint arXiv:2605.09424.

Ke, G.; Meng, Q.; Finley, T.; Wang, T.; Chen, W.; Ma, W.; Ye, Q.; and Liu, T.-Y. 2017. Lightgbm: A highly eficient gradient boosting decision tree. Advances in neural information processing systems, 30.

Kotelnikov, A.; Baranchuk, D.; Rubachev, I.; and Babenko, A. 2023. Tabddpm: Modelling tabular data with difusion models. In International conference on machine learning, 17564–17579. PMLR.

Li, S.; Cai, T. T.; and Li, H. 2022. Transfer learning for high-dimensional linear regression: Prediction, estimation and minimax optimality. Journal of the Royal Statistical Society Series B: Statistical Methodology, 84(1): 149–173.

Lin, Y.; and Li, S. 2026. Context-Constrained Transfer Learning for Tabular Foundation Models via Data Distillation. arXiv preprint arXiv:2607.04809.

Liu, S.-Y.; Cai, H.-R.; Zhou, Q.-L.; Yin, H.-H.; Zhou, T.; Jiang, J.-P.; and Ye, H.-J. 2025. TALENT: A tabular analytics and learning toolbox. Journal of Machine Learning Research, 26(226): 1–16.

Liu, S.-Y.; and Ye, H.-J. 2025. Tabpfn unleashed: A scalable and efective solution to tabular classification problems. arXiv preprint arXiv:2502.02527.

Ma, J.; Dankar, A.; Stein, G.; Yu, G.; and Caterini, A. 2024a. TabPFGen–Tabular Data Generation with TabPFN. arXiv preprint arXiv:2406.05216.

Ma, J.; Thomas, V.; Hosseinzadeh, R.; Labach, A.; Kamkari, H.; Cresswell, J. C.; Golestan, K.; Yu, G.; Caterini, A. L.; and Volkovs, M. 2024b. Tabdpt: Scaling tabular foundation models on real data. arXiv preprint arXiv:2410.18164.

Ma, J.; Thomas, V.; Yu, G.; and Caterini, A. 2024c. Incontext data distillation with TabPFN. arXiv preprint arXiv:2402.06971.

Müller, A.; Curino, C.; and Ramakrishnan, R. 2025. Mothernet: Fast training and inference via hyper-network transformers. In International Conference on Learning Representations, volume 2025, 76666–76686.

Prokhorenkova, L.; Gusev, G.; Vorobev, A.; Dorogush, A. V.; and Gulin, A. 2018. CatBoost: unbiased boosting with categorical features. Advances in neural information processing systems, 31.

Qu, J.; Holzmüller, D.; Varoquaux, G.; and Le Morvan, M. 2025. TabICL: A Tabular Foundation Model for In-Context Learning on Large Data. In International Conference on Machine Learning.

Qu, J.; Holzmüller, D.; Varoquaux, G.; and Morvan, M. L. 2026. TabICLv2: A better, faster, scalable, and open tabular foundation model. arXiv preprint arXiv:2602.11139.

Shi, J.; Xu, M.; Hua, H.; Zhang, H.; Ermon, S.; and Leskovec, J. 2025a. Tabdif: a mixed-type difusion model for tabular data generation. In International Conference on Learning Representations, volume 2025, 37353–37375.

Sergazinov, R.; and Yin, S.-A. 2025. Chunked TabPFN: Exact Training-Free In-Context Learning for Long-Context Tabular Data. arXiv preprint arXiv:2509.00326.

Shi, R.; Wang, Y.; Du, M.; Shen, X.; Chang, Y.; and Wang, X. 2025b. A comprehensive survey of synthetic tabular data generation. arXiv preprint arXiv:2504.16506.

Tanna, A.; Bouarour, N.; Bouadi, M.; Sankarapu, V. K.; and Seth, P. 2026a. Distilling Tabular Foundation Models for Structured Health Data. arXiv preprint arXiv:2605.18702.

Tanna, A.; Bouarour, N.; Bouadi, M.; Seth, P.; et al. 2026b. Pocket Foundation Models: Distilling TFMs into CPU-Ready Gradient-Boosted Trees. arXiv preprint arXiv:2605.18654.

Thomas, V.; Ma, J.; Hosseinzadeh, R.; Golestan, K.; Yu, G.; Volkovs, M.; and Caterini, A. 2024. Retrieval & finetuning for in-context tabular models. Advances in Neural Information Processing Systems, 37: 108439–108467.

Van Breugel, B.; and Van Der Schaar, M. 2024. Why tabular foundation models should be a research priority. arXiv preprint arXiv:2405.01147.

Wang, Y.; Zhang, X.; Yu, H.; Ming, M.; Ren, G.; Yuan, H.; Mao, L.; Zhang, Y.; Yuan, C.; and Cui, P. 2026. LimiX-2M: Mitigating Low-Rank Collapse and Attention Bottlenecks in Tabular Foundation Models. arXiv preprint arXiv:2606.04485.

Xu, D.; Cirit, O.; Asadi, R.; Sun, Y.; and Wang, W. 2025. Mixture of in-context prompters for tabular pfns. In International Conference on Learning Representations, volume 2025, 24630–24665.

Xu, L.; Skoularidou, M.; Cuesta-Infante, A.; and Veeramachaneni, K. 2019. Modeling tabular data using conditional gan. Advances in neural information processing systems, 32.

Ye, H.-J.; Liu, S.-Y.; and Chao, W.-L. H. 2026. A closer look at TabPFN v2: Understanding its strengths and extending its capabilities. Advances in Neural Information Processing Systems, 38: 135605–135637.

Zabërgja, G.; Kamel, R.; Kadra, A.; Frey, C. M.; and Grabocka, J. 2026. End-to-End Compression for Tabular Foundation Models. arXiv preprint arXiv:2602.05649.

Zhang, H.; Zhang, J.; Shen, Z.; Srinivasan, B.; Qin, X.; Faloutsos, C.; Rangwala, H.; and Karypis, G. 2024. Mixedtype tabular data synthesis with score-based difusion in latent space. In International Conference on Learning Representations, volume 2024, 52829–52857.

Zhuang, F.; Qi, Z.; Duan, K.; Xi, D.; Zhu, Y.; Zhu, H.; Xiong, H.; and He, Q. 2020. A comprehensive survey on transfer learning. Proceedings ofthe IEEE, 109(1): 43–76.