# Post-Calibration Reliability Reranking of Relevance Decisions via Label-wise Monotone Projection

Inwoo Tae

vitainu0104@unist.ac.kr

UNIST

Ulsan, Republic of Korea

Yongjae Lee   
yongjaelee@unist.ac.kr   
UNIST   
Ulsan, Republic of Korea   
LinqAlpha   
New York, NY, USA

## Abstract

Web search, product search, and question-answering retrieval systems often assign a relevance label and confidence score to each query-candidate pair. The relevance label describes how well a page, product, or passage matches the query, while the confidence often guides downstream use or fallback decisions. Post-hoc calibration is therefore needed because misaligned confidence can make systems over-trust wrong predictions or unnecessarily defer correct ones. However, calibration mainly aligns confidence with average correctness, and does not remove predicted-label-dependent reliability diferences that remain within the same calibrated confidence level. We address this gap with Label-wise Monotone Reliability Projection (MRP), which learns label-wise monotone functions that map calibrated confidence to correctness reliability while preserving the original predicted labels and class probabilities. The resulting reliability score reranks fixed predictions according to residual risk. Across six information access relevance datasets and multiple posthoc calibrators, MRP improves reliability reranking and average fallback utility while preserving full-coverage accuracy and ECE. Structural ablations show that the main gains come from label-wise residual reliability rather than from global confidence remapping. We further analyze when MRP reliability scores can be embedded back into top-label probability geometry, showing that this projection is useful as a compatibility analysis but is distinct from the main reliability-reranking objective. The implementation will be made publicly available.

## Keywords

relevance prediction, confidence calibration, reliability reranking, selective prediction, information access systems, fallback routing

## 1 Introduction

Information access systems such as web search, product search, document reranking, and question answering (QA) retrieval have become central interfaces through which users discover and use information in digital services. Users search for web pages through search engines, compare products in online marketplaces, and retrieve documents or passages to support question answering. Although these services difer in form, they share a common need to evaluate multiple candidate results for a user input. In this setting, candidate evaluation can be viewed as a relevance prediction prob lem, where relevance refers to how well a candidate matches the user’s input. These predictions then shape how candidate results are handled by the downstream system [24, 30, 33].

When such predictions are used downstream, the confidence attached to each decision becomes important. It may determine whether the prediction is used directly or routed to review or fallback [6, 7]. We use reliability for a diferent question from relevance: relevance is the predicted query-candidate match, whereas reliability is whether that fixed relevance decision is likely to be correct. A natural response is to apply post-hoc calibration so that confidence values better match empirical correctness frequencies [9, 37]. However, calibration is usually applied as an average condition over predictions with similar confidence, without directly distin guishing which relevance decision was made. It can therefore leave heterogeneous reliability within the same confidence level. As a result, a calibrated confidence group can contain diferent relevance decisions whose empirical correctness rates difer [11]. Even after calibration, the system must still rerank fixed relevance decisions by residual risk among predictions with similar confidence. We view this as a post-calibration residual reliability problem, where the goal is not to produce another relevance score, but to estimate the residual correctness structure that remains after calibration.

This setting suggests a method that preserves the calibrated prediction while providing a separate score for reranking fixed decisions by expected correctness. We propose Label-wise Monotone Reliability Projection (MRP), a post-calibration reliability reranking method. Given a calibrated confidence � and a fixed predicted relevance decision �, MRP learns a label-wise monotone function �<sub>�</sub> (�) that estimates the probability that the decision is correct. Monotonicity ensures that, within the same predicted relevance decision, higher calibrated confidence does not imply lower correctness reliability. The predicted label and calibrated class probabilities remain unchanged. Only the reliability score used for reranking, review, or fallback is replaced by �<sub>�</sub> (�). This turns calibrated confidence into a decision-conditioned correctness reliability score without turning MRP into another relevance scorer or class-probability calibrator.

We evaluate MRP on relevance prediction benchmarks covering product search, web search, question answering retrieval, ecommerce reranking, and scientific retrieval. Each experiment fixes a base relevance predictor and a post-hoc calibrator before applying MRP. The evaluation tests whether MRP improves reliability reranking while leaving class probabilities, predicted labels, full-coverage accuracy, and expected calibration error (ECE) unchanged. Across datasets and calibrators, MRP broadly improves correctness negative log-likelihood (NLL) and area under the risk-coverage curve (AURC), with AUPR-Error gains concentrated in settings where label-conditioned residual signal is present. Budgeted fallback experiments further show positive average gains in the accuracy of retained automatic predictions. This suggests that probability-scale calibration and reliability reranking capture distinct reliability aspects. Structural ablations show that this reranking gain mainly comes from residual reliability structure associated with predicted relevance decisions, rather than from confidence-only ranking. The contributions of this paper are as follows:

![](images/e5635112828d338d8fda98e774f14d73ee0d036d52d4ae79d65f173fb1846d83.jpg)  
Figure 1: Overview of MRP for post-calibration reliability reranking. Each row is a fixed query-candidate relevance prediction. Post-hoc calibration updates the confidence of the predicted label, while MRP maps this calibrated confidence to label-wise correctness reliability and reranks the same predictions by estimated risk. The bottom comparison shows that confidence based reranking and MRP reliability reranking can difer. This is reliability reranking of predictions, not retrieval ranking of candidates.

• We define post-calibration residual reliability as a reliability reranking problem over fixed relevance decisions. This separates reliability reranking of predicted decisions from both class-probability recalibration and candidate ranking by relevance.

• We propose MRP, which projects calibrated confidence to correctness reliability through label-wise monotone functions. MRP preserves the calibrated class probabilities and predicted labels, and changes only the score used to rerank predictions for review, fallback, or selective use.

• We evaluate MRP across relevance benchmarks and post-hoc calibrators. The results show broad improvements in correctness NLL and AURC, positive average fallback gains, and AUPR-Error gains when label-conditioned residual signal is present, while preserving full-coverage accuracy and ECE.

• We provide structural ablations that clarify why MRP works. The main gain comes from label-conditioned residual reliability, while label-wise monotone curves provide a confidencedependent generalization of a label-only reliability baseline.

## 2 Related Work

## 2.1 Probability Calibration

Confidence calibration began as the problem of making model scores interpretable as probabilities consistent with empirical correctness frequencies. Early post-hoc methods kept the trained predictor fixed and learned a score-to-probability mapping on a separate validation set. For example, Platt scaling uses a parametric sigmoid mapping, while histogram binning, Bayesian binning, and isotonic regression provide more flexible non-parametric mappings from prediction scores to calibrated probabilities [28, 29, 31, 37].

The problem became especially prominent for deep neural networks after it was observed that accurate models can still produce overconfident probabilities. In this setting, temperature scaling became a simple yet strong post-hoc baseline that adjusts logits using a single scalar temperature [9]. Subsequent work has moved beyond scalar temperature by allowing more flexible transformations of class-wise logits or the full probability vector. Other lines of work directly optimize or evaluate calibration error, or distinguish full-vector calibration from top-label confidence calibration for decisions [11, 19–21, 36].

The common goal of these methods is to make the model’s class probabilities themselves more reliable as probabilities. Accordingly, they are often evaluated using calibration metrics such as expected calibration error (ECE). Our work does not replace these calibration methods. Instead, we use them as base calibrators. In our experiments, we apply the same MRP layer on top of several base calibrators, including temperature scaling, diagonal order-preserving calibration, spline-based top-label confidence calibration, h-calibration, and SMART-style calibration. Our central question is therefore not only how well a calibrator aligns the probability scale, but whether, after calibration, predictions with similar confidence can be reranked more reliably by their correctness risk.

Although MRP uses label-wise monotone functions, it is not a label-wise class-probability calibrator. A label-wise calibrator modifies the probability assigned to a class, whereas MRP keeps the calibrated probability vector fixed and learns a separate correctnessreliability score used only for reranking fixed predictions.

## 2.2 Selective Prediction and Reliability Reranking

If calibration concerns the numerical meaning of predicted probabilities, selective prediction concerns how such probabilities or uncertainty scores decide which predictions should be accepted. This line of work studies settings in which a system abstains from, rejects, or defers predictions in order to trade coverage for lower risk [4, 6– 8]. Related learning-to-defer work further studies how predictions can be routed to a human or expert decision maker [25, 27]. Errordetection work similarly uses scores such as maximum softmax probability to identify misclassified or out-of-distribution examples [15]. These studies make confidence and uncertainty operational because they are not only reported as probabilities, but used as selection scores for accepting or withholding predictions.

This perspective is closely related to downstream use of calibrated probabilities, but it leaves a distinct post-calibration question. Selective prediction typically asks how much error is reduced after removing low-confidence or high-uncertainty examples. In contrast, after top-label confidence has already been calibrated, predictions with the same nominal confidence may still have dif ferent residual correctness risk. Reliability reranking focuses on this remaining problem. Before calibrated predictions are used for selection, fallback, or review, their residual risk must be estimated and used to rerank predictions without necessarily changing the calibrated class probabilities themselves.

## 2.3 Relevance Prediction in Information Access Systems

Relevance prediction is a central component of information access systems. Classical probabilistic retrieval and learning-to-rank methods use query–document relevance signals to optimize the ranking of retrieved items [24, 34], while neural rerankers and dense retrievers learn semantic matching signals between queries and candidate documents or passages [5, 18, 30]. Similar relevance prediction problems arise in product search, where query–product pairs are annotated with graded relevance labels [1, 33], and in retrieval-augmented question answering, where retrieved passages afect downstream answer generation [13, 17, 23]. Across these settings, relevance scores are typically evaluated by their ability to improve candidate ranking, retrieval, or classification quality.

In deployed information access systems, however, relevance scores are also used as decision signals. A system may need to decide whether to show a result directly, send a candidate to a stronger reranker, use a passage as evidence for answer generation, or defer an uncertain case for additional processing. These decisions depend not only on relative candidate ranking quality, but also on whether a relevance confidence score is reliable as a numerical signal. Existing relevance prediction work provides the scoring and candidate-ranking machinery for these systems, while calibration and selective prediction provide tools for probability interpretation and acceptance decisions. The remaining gap is the post-calibration setting in which a relevance confidence has already been assigned, but predictions with the same nominal confidence may still difer in residual correctness risk.

## 3 Method

This section formalizes MRP as a post-calibration reliability reranking layer. We first define a fixed-decision protocol in which the base relevance decision is kept unchanged and a post-hoc calibrator assigns calibrated confidence to that decision. We then define the correctness reliability target and show why calibrated confidence alone can be insuficient when reliability depends on the predicted relevance label. Finally, we introduce the label-wise monotone reliability map, its lattice implementation, and the ablation variants used to isolate label conditioning and decision separation.

## 3.1 Setup and Top-label Calibration

Let � be an input example and $Y \in \{ 1 , \ldots , K \}$ its class label. Superscript 0 denotes the uncalibrated base predictor, and superscript A denotes outputs after applying calibrator ${ \mathcal { A } } .$ . The base relevance predictor outputs $\hat { \mathbf { p } } ^ { 0 } ( x ) \overset { - } { \in } [ 0 , 1 ] ^ { \overset { \textstyle } { K } }$ , with $\begin{array} { r } { \sum _ { k = 1 } ^ { K } \hat { p } _ { k } ^ { 0 } ( x ) = 1 } \end{array}$

The top-label decision of the base predictor is fixed as

$$
d ( x ) = \arg \operatorname* { m a x } _ { k } \hat { p } _ { k } ^ { 0 } ( x ) , \qquad Z ( x ) = 1 \{ Y = d ( x ) \} .
$$

Here, $Z ( x )$ indicates whether the fixed decision is correct.

A post-hoc calibrator $\mathcal { A }$ produces $\hat { \mathbf { p } } ^ { \mathcal { A } } ( x ) \in [ 0 , 1 ] ^ { K }$ , with $\begin{array} { r } { \sum _ { k = 1 } ^ { K } \hat { p } _ { k } ^ { \mathcal { A } } ( x ) = } \end{array}$ 1. Our protocol keeps � (�) fixed rather than re-evaluating the calibrated top label, so calibration cannot change decision accuracy. This isolates reliability reranking from changes in the underlying prediction. The confidence assigned by A to this decision is

$$
\hat { c } ^ { \mathcal { A } } ( x ) = [ \hat { \mathbf { p } } ^ { \mathcal { A } } ( x ) ] _ { d ( x ) }
$$

where $[ \mathbf { v } ] _ { k }$ denotes the �-th component of v. Here $\mathcal { A }$ can be any probability-producing calibrator, including the identity map. The evaluated event is always $Y = d ( x )$

Let $\hat { Y }$ be a predicted top label and let $\hat { C }$ be the confidence assigned to that prediction. The standard top-label calibration condition can be written as

$$
\mathbb { P } ( Y = \hat { Y } \mid \hat { C } = c ) = c .
$$

Under our fixed-decision protocol, with $\hat { C } ^ { \mathcal { A } } = \hat { c } ^ { \mathcal { A } } ( X )$ , this becomes

$$
\mathbb { P } ( Z = 1 \mid \hat { C } ^ { \mathcal { A } } = c ) = c .
$$

Expected calibration error (ECE) approximates this condition over confidence bins or regions. It evaluates whether the confidence scale is correct on average, but does not imply that all predictions with the same confidence have the same correctness reliability.

## 3.2 Correctness Reliability After Calibration

The projection target is not a new class probability vector, but the probability that the selected top-label event is correct. The simplest baseline uses the calibrated score itself as the correctness reliability estimate:

$$
\hat { q } _ { \varnothing } ^ { \mathcal { A } } ( x ) = \hat { c } ^ { \mathcal { A } } ( x ) .
$$

We use ∅ to denote the case without a reliability projection. Let $D = d ( X )$ denote the random variable corresponding to the decision label selected by the uncalibrated predictor. This baseline is suficient only if conditioning on � does not change the correctness probability beyond calibrated confidence:

$$
H _ { 0 } : \quad \mathbb { P } ( Z = 1 \mid \hat { C } ^ { \mathcal { A } } = c , D = k ) = \mathbb { P } ( Z = 1 \mid \hat { C } ^ { \mathcal { A } } = c ) \quad \forall c , k .
$$

When this equality fails, calibrated confidence alone misses labeldependent reliability diferences that matter for reliability reranking. This motivates a label-wise monotone reliability map.

## 3.3 Label-wise Monotone Reliability Projection

Label-wise monotone reliability projection (MRP) estimates correctness reliability with a separate monotone function for each predicted label. For an input whose decision is $d ( x )$ , MRP applies the reliability function associated with that label:

$$
\hat { q } _ { \mathrm { M R P } } ^ { \mathcal { R } } ( x ) = T _ { d ( x ) } ( \hat { c } ^ { \mathcal { A } } ( x ) ) ,
$$

where $T _ { k } : [ 0 , 1 ] \ \to \ [ 0 , 1 ]$ is the confidence-to-reliability function for label �. Each function satisfies the following monotonicity constraint:

$$
c \leq c ^ { \prime } \quad \implies \quad T _ { k } ( c ) \leq T _ { k } ( c ^ { \prime } ) \qquad \forall k .
$$

This constraint prevents MRP from assigning lower reliability to a higher-confidence prediction within the same label. Thus, MRP preserves the confidence ranking within each label while learning label-specific reliability curves.

On the projection-fit set, we learn $T _ { 1 } , \dots , T _ { K }$ with the following objective. Let $S _ { \mathrm { f i f } } ^ { \mathcal { A } } = \{ ( \hat { c } _ { i } ^ { \mathcal { A } } , d _ { i } , Z _ { i } ) \} _ { i = 1 } ^ { m }$ be the triples produced by calibrator ${ \mathcal { A } } ,$ where $d _ { i } = d ( x _ { i } )$ and $Z _ { i } = Z ( x _ { i } )$ . Let M be the class of label-wise functions satisfying $T _ { k } : [ 0 , 1 ] \to [ 0 , 1 ]$ and monotonicity in �. We use $\mathcal { R } _ { \Delta ^ { 2 } } ( T )$ for a second-diference regularizer specified in the lattice implementation. For $T = \{ T _ { k } \} _ { k = 1 } ^ { K } \in \mathcal { M }$ and nonnegative weight $\rho ,$ define

$$
\mathcal { L } _ { \rho } ( T ; S _ { \mathrm { f i } } ^ { \mathcal { R } } ) = \frac { 1 } { m } \sum _ { i = 1 } ^ { m } \mathrm { B C E } \left( Z _ { i } , T _ { d _ { i } } ( \hat { c } _ { i } ^ { \mathcal { R } } ) \right) + \rho \mathcal { R } _ { \Delta ^ { 2 } } ( T ) .\tag{1}
$$

Here, $\mathrm { B C E } ( z , q ) ~ = ~ - z \log q - ( 1 - z ) \log ( 1 - q )$ is binary cross entropy. The BCE term fits correctness, while $\mathcal { R } _ { \Delta ^ { 2 } } ( T )$ stabilizes finite-sample label-wise curves. MRP therefore learns a separate reliability reranking score rather than replacing the class-probability calibrator.

Although $\hat { q } _ { \mathrm { M R P } } ^ { \mathcal { R } } ( x )$ is probability-valued, it is not a calibrated class-probability vector. It estimates only the probability that the fixed event $Y = d ( x )$ is correct. Turning it into a multiclass predictor would require allocating $1 - \hat { q } _ { \mathrm { M R P } } ^ { \mathcal { A } } ( x )$ over the non-selected labels, so we use it only as a correctness reliability score.

This distinction matters because $\hat { p } _ { d ( x ) } ^ { \mathcal { A } } ( x )$ remains the base calibrator’s class-probability estimate, while $\hat { q } _ { \mathrm { M R P } } ^ { \mathcal { R } } ( x )$ is a decision-level reliability score for reranking, fallback, and review.

For fallback, selective use, and reliability reranking, predictions are ranked by decreasing estimated error probability,

$$
1 - \hat { q } _ { \mathrm { M R P } } ^ { \mathcal { A } } ( x ) = 1 - T _ { d ( x ) } ( \hat { c } ^ { \mathcal { A } } ( x ) ) .
$$

Larger values indicate predictions that should be handled later, routed to fallback, or reviewed first. Importantly, MRP does not change $\hat { \mathbf { p } } ^ { \mathcal { A } } ( x ) , d ( x ) , \hat { c } ^ { \mathcal { A } } ( x ) , \mathrm { o r } Z ( x )$ . Therefore, full-coverage accuracy and the calibration metrics of the base calibrator remain unchanged before and after applying MRP. What changes is only how calibrated predictions are ranked for use.

## 3.4 Monotone Lattice Implementation

We do not optimize over arbitrary continuous functions $T _ { k }$ . Instead, we represent each $T _ { k }$ as a finite one-dimensional monotone lattice on the confidence axis and optimize its lattice parameters. Let $0 = u _ { 1 } < \dots < u _ { I } = 1$ be knots on the confidence axis, and let $a _ { k , 1 } , \ldots , a _ { k , J }$ be the logit-scale lattice values for label �. We enforce monotonicity using a nonnegative increment parameterization:

$$
a _ { k , j } = b _ { k } + \sum _ { r < j } \mathrm { s o f t p l u s } ( \eta _ { k , r } ) .
$$

Here, softplus $( t ) = \log ( 1 + \exp ( t ) )$ , so this parameterization guarantees $a _ { k , 1 } \leq \dots \leq a _ { k , J }$ . For $c \in \left[ u _ { j } , u _ { j + 1 } \right]$ , we interpolate on the logit scale:

$$
\tilde { a } _ { k } ( c ) = \frac { u _ { j + 1 } - c } { u _ { j + 1 } - u _ { j } } a _ { k , j } + \frac { c - u _ { j } } { u _ { j + 1 } - u _ { j } } a _ { k , j + 1 } .
$$

The reliability map is then $T _ { k } ( c ) = \sigma ( \tilde { a } _ { k } ( c ) )$ , where $\sigma ( t ) = 1 / ( 1 +$ $\exp ( - t ) )$ . Thus, the parameterization satisfies both $0 \leq T _ { k } ( c ) \leq 1$ and monotonicity throughout optimization. The second-diference regularizer in Eq. (1) is

$$
\mathcal { R } _ { \Delta ^ { 2 } } ( T ) = \frac { 1 } { K ( J - 2 ) } \sum _ { k = 1 } ^ { K } \sum _ { j = 2 } ^ { J - 1 } \left( a _ { k , j + 1 } - 2 a _ { k , j } + a _ { k , j - 1 } \right) ^ { 2 } ,
$$

which penalizes highly oscillatory label-wise curves.

At test time, each fixed-decision pair $( \hat { c } _ { i } ^ { \mathcal { A } } , d _ { i } )$ receives

$$
\begin{array} { r } { \hat { q } _ { \mathrm { M R P } , i } ^ { \mathcal { A } } = T _ { d _ { i } } ( \hat { c } _ { i } ^ { \mathcal { A } } ) , } \end{array}
$$

and predictions are ranked by $1 - \hat { q } _ { \mathrm { M R P } , i } ^ { \mathcal { A } } .$

## 3.5 Ablation Variants

The main MRP model above, the label-wise 1D variant, learns one monotone curve $T _ { k } ( c )$ for each predicted label. We compare it with variants that remove or add one structural component. Together with the confidence baseline $\hat { q } _ { \emptyset } ^ { \mathcal { A } }$ , these variants separate three effects, namely using confidence alone, conditioning on the predicted label, and allowing the label efect to vary with confidence.

Shared 1D � (�) removes label conditioning and learns a single global confidence-to-reliability curve. Throughout this subsection, $\mathrm { l o g i t } ( p ) = \log ( p / ( 1 - p ) )$ ). The label-only intercept baseline removes confidence-dependent curves but keeps a learned label intercept:

$$
\hat { q } _ { \mathrm { L I } } ^ { \mathcal { R } } ( x ) = \sigma \left( \log \mathrm { i t } ( \hat { c } ^ { \mathcal { A } } ( x ) ) + \alpha _ { d ( x ) } \right)
$$

where $\alpha _ { k }$ is learned for each predicted label. This diagnostic variant tests whether the observed gain can be explained by a fixed labelwise intercept. It is a special case of $T _ { k } ( c )$ where each label applies a fixed logit-scale shift rather than a confidence-dependent reliability curve. We also include per-label isotonic projection, which fits an independent monotone confidence-to-reliability map for each predicted label without the smooth lattice parameterization used by MRP.

Table 1: Datasets used in the relevance prediction experiments.
<table><tr><td>Dataset</td><td>Domain</td><td>K</td><td>Labels</td><td>Task</td></tr><tr><td>Amazon ESCI [33]</td><td>E-commerce search</td><td>4</td><td>Exact/Substitute/Complement/Irrelevant</td><td>Query-product relevance classification</td></tr><tr><td>MSLR-WEB10K [26]</td><td>Web search</td><td>5</td><td>Relevance grades 0-4</td><td>Learning-to-rank relevance prediction</td></tr><tr><td>Alloprof-Rerank [22]</td><td>QA retrieval</td><td>2</td><td>Relevant/Non-relevant</td><td>Question-document relevance reranking</td></tr><tr><td>ESCI-Rerank-US [33]</td><td>E-commerce reranking</td><td>2</td><td>Relevant/Non-relevant</td><td>Query-product candidate reranking</td></tr><tr><td>WANDS [1]</td><td>E-commerce search</td><td>3</td><td>Exact/Partial/Irrelevant</td><td>Product search relevance prediction</td></tr><tr><td>SciDocs [2, 35]</td><td>Scientific retrieval</td><td>2</td><td>Relevant/Non-relevant</td><td>Candidate document reranking</td></tr></table>

Label-wise 2D adds decision separation to the main label-wise model. Let the uncalibrated runner-up label be

$$
\ell ( x ) = \arg \operatorname* { m a x } _ { k \neq d ( x ) } \hat { \mathcal { P } } _ { k } ^ { 0 } ( x )
$$

and define the calibrated top-runner logit gap as

$$
\begin{array} { r } { g ^ { \mathcal { R } } ( x ) = \log \mathrm { i t } ( [ \hat { \boldsymbol { \mathsf { p } } } ^ { \mathcal { R } } ( x ) ] _ { d ( x ) } ) - \log \mathrm { i t } ( [ \hat { \boldsymbol { \mathsf { p } } } ^ { \mathcal { R } } ( x ) ] _ { \ell ( x ) } ) . } \end{array}
$$

The 2D variant learns

$$
\hat { q } _ { \mathrm { 2 D } } ^ { \mathcal { A } } ( x ) = T _ { d ( x ) } ( \hat { c } ^ { \mathcal { A } } ( x ) , g ^ { \mathcal { A } } ( x ) )
$$

where $T _ { k } ( c , g )$ is constrained to be nondecreasing in both � and �. This variant tests whether top-runner separation provides additional reliability information beyond confidence and the fixed decision label.

## 4 Experiments

MRP is evaluated on relevance prediction tasks spanning product search, web search, QA retrieval, e-commerce reranking, and scientific document retrieval. All experiments follow the fixed-decision protocol from Section 3, where base decisions are held fixed, posthoc calibrators only change the confidence assigned to those decisions, and MRP only reranks calibrated predictions by reliability. The main question is whether this improves correctness reliability reranking while preserving full-coverage predictions and the calibrated probabilities.

## 4.1 Datasets

The experiments use six datasets that cover both graded and binary relevance decisions. Amazon ESCI and WANDS represent product search, MSLR-WEB10K represents graded web relevance, Alloprof-Rerank and SciDocs represent retrieval reranking, and ESCI-Rerank-US provides a binary e-commerce reranking setting. These datasets difer in domain and label granularity, but in all cases the predicted relevance label can afect whether a candidate is trusted for downstream use. Table 1 summarizes the datasets used in our experiments. � denotes the number of relevance classes.

## 4.2 Base Predictors and Calibrators

For each dataset, we first train a base relevance predictor and then fit post-hoc calibrators on validation data. For text-based query– candidate datasets, the predictor is a lightweight linear classifier over the concatenated query and candidate text. For feature-based learning-to-rank data, we use the provided ranking features. The goal of this setup is not to maximize the base relevance model itself, but to create a fixed prediction source on which calibration and post-calibration reliability reranking can be compared cleanly.

The main comparison starts from a single fixed base predictor. Concretely, for each dataset we use the saved logits from model seed 0 and define the fixed decision �(�) from this uncalibrated predictor. All calibrators are then applied to the same saved logits and only change the probability or confidence assigned to this fixed decision. This fixed-decision protocol ensures that calibration methods and MRP share the same decision event $Y = d ( x )$ , so full-coverage accuracy is identical within each dataset.

The comparison includes six calibration outputs. Uncal. uses the identity map and keeps softmax probabilities. TS [9] fits a scalar temperature to rescale logits. DIAG [32] follows the diagonal intraorder-preserving calibration family. For fixed-decision evaluation, it rescales sorted adjacent logit gaps with positive factors to preserve the within-sample class order. Spline [12] applies spline-based top-label confidence calibration. h-cal [16] uses the h-calibration objective. SMART [10] follows sample margin-aware recalibration of temperature scaling and predicts a sample-wise temperature from the top-two logit gap. For all outputs, MRP leaves $\hat { \mathbf { p } } ^ { \mathcal { A } } ( x ) , d ( x )$ , and $\hat { c } ^ { \mathcal { A } } ( x )$ unchanged and estimates only the correctness reliability of the fixed decision.

## 4.3 Implementation Details

For each protocol seed, we split validation data into calibrator-fit, projection-fit, and projection-selection subsets. The calibrator-fit subset fits the base calibrator A. The projection-fit subset is used to learn $T _ { k }$ or an ablation variant, and the projection-selection subset is held out for model selection in variants that require it. For Uncal., the base calibrator is the identity map and therefore has no fitted calibration parameters, but we still repeat the projection-fit and projection-selection split to estimate MRP under the same threeseed protocol. The test set is evaluated only after the calibrator, reliability projection, and hyperparameters are fixed.

The main MRP model is implemented as the label-wise monotone 1D lattice $T _ { k } ( \hat { c } ^ { \mathcal { A } } )$ described in Section 3. Unless otherwise stated, we use $J = 8$ confidence knots, second-diference coeficient $\rho = 1 0 ^ { - 4 }$ , the Adam optimizer with learning rate 0.03, and a maximum of 400 epochs. To keep computation comparable across datasets, projection-fit and projection-selection are each capped at 8000 samples. The base predictor is trained or loaded once with model seed 0, and the calibrator and MRP procedures are repeated with protocol seeds 1, 2, 3. For fitted calibrators, this repeats both the calibration fit and the MRP fit. For Uncal., only the MRP fit varies because the identity calibrator is deterministic. For every run, we verify that the reported full-coverage accuracy matches the seed-0 uncalibrated decision accuracy, confirming that calibration and MRP do not change the evaluated decision event. We report means and standard deviations over the protocol seeds.

Table 2: Post-calibration reliability reranking results under the fixed-decision protocol. Acc. and ECE are base-calibrator values and are unchanged by MRP. Reliability metrics compare the confidence baseline $\hat { q } = \hat { c } ^ { \mathcal { A } }$ with MRP. Blue bold Δ values indicate improvements in the direction of each metric.
<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Calibrator</td><td rowspan="2">Acc.</td><td rowspan="2">ECE</td><td colspan="3"> $\begin{array} { r } { { \mathrm { N L L } } _ { \mathrm { c o r r e c t } } \downarrow } \\ { { \cal M R P } } \end{array}$ </td><td colspan="3">AUPR-Error↑</td><td colspan="3">AURC↓</td></tr><tr><td>Conf.</td><td></td><td>Δ</td><td>Conf.</td><td>MRP</td><td>Δ</td><td>Conf.</td><td>MRP</td><td>Δ</td></tr><tr><td rowspan="6">ESCI-Rerank US</td><td>Uncal.</td><td>0.571</td><td>0.183</td><td>0.832± 0.000</td><td>0.647± 0.000</td><td>-0.186</td><td>0.462± 0.000</td><td>0.558± 0.000</td><td>+0.097</td><td>0.385± 0.000</td><td>0.331± 0.000</td><td>-0.053</td></tr><tr><td>TS</td><td>0.571</td><td>0.013</td><td>0.679± 0.000</td><td>0.646± 0.000</td><td>-0.033</td><td>0.462± 0.000</td><td>0.558± 0.000</td><td>+0.097</td><td>0.385± 0.000</td><td>0.331± 0.000</td><td>-0.053</td></tr><tr><td>DIAG</td><td>0.571</td><td>0.011</td><td>0.679± 0.000</td><td>0.646± 0.000</td><td>-0.033</td><td>0.462± 0.000</td><td>0.558± 0.000</td><td>+0.097</td><td>0.385± 0.000</td><td>0.332± 0.000</td><td>-0.053</td></tr><tr><td>Spline</td><td>0.571</td><td>0.015</td><td>0.680± 0.000</td><td>0.646± 0.000</td><td>-0.034</td><td>0.466± 0.003</td><td>0.563± 0.002</td><td>+0.097</td><td>0.387± 0.001</td><td>0.332± 0.001</td><td>-0.055</td></tr><tr><td>h-cal</td><td>0.571</td><td>0.112</td><td>0.734± 0.000</td><td>0.646± 0.000</td><td>-0.087</td><td>0.462± 0.000</td><td>0.558± 0.000</td><td>+0.097</td><td>0.385± 0.000</td><td>0.331± 0.000</td><td>-0.053</td></tr><tr><td>SMART</td><td>0.571</td><td>0.007</td><td>0.680± 0.001</td><td>0.647± 0.000</td><td>-0.034</td><td>0.462± 0.000</td><td>0.558± 0.000</td><td>+0.097</td><td>0.385± 0.000</td><td>0.332± 0.000</td><td>-0.054</td></tr><tr><td rowspan="6">MSLR- WEB10K</td><td>Uncal.</td><td>0.394</td><td>0.292</td><td>1.387± 0.000</td><td>0.450± 0.001</td><td>-0.937</td><td>0.653± 0.000</td><td>0.906± 0.000</td><td>+0.252</td><td>0.605± 0.000</td><td>0.344± 0.000</td><td>-0.261</td></tr><tr><td>TS</td><td>0.394</td><td>0.070</td><td>0.684± 0.000</td><td>0.448± 0.001</td><td>-0.236</td><td>0.678± 0.000</td><td>0.906± 0.000</td><td>+0.229</td><td>0.557± 0.000</td><td>0.343± 0.000</td><td>-0.214</td></tr><tr><td>DIAG</td><td>0.394</td><td>0.096</td><td>0.651± 0.000</td><td>0.453± 0.001</td><td>-0.198</td><td>0.743± 0.001</td><td>0.905± 0.000</td><td>+0.162</td><td>0.466± 0.002</td><td>0.345± 0.000</td><td>-0.120</td></tr><tr><td>Spline</td><td>0.394</td><td>0.011</td><td>0.663± 0.000</td><td>0.469± 0.000</td><td>-0.193</td><td>0.673± 0.001</td><td>0.889± 0.001</td><td>+0.216</td><td>0.547± 0.000</td><td>0.380± 0.001</td><td>-0.167</td></tr><tr><td>h-cal</td><td>0.394</td><td>0.055</td><td>0.685± 0.000</td><td>0.448± 0.001</td><td>-0.237</td><td>0.667± 0.000</td><td>0.906± 0.000</td><td>+0.239</td><td>0.555± 0.000</td><td>0.343± 0.000</td><td>-0.212</td></tr><tr><td>SMART</td><td>0.394</td><td>0.114</td><td>0.637± 0.001</td><td>0.456± 0.002</td><td>-0.180</td><td>0.723± 0.004</td><td>0.902± 0.002</td><td>+0.179</td><td>0.457± 0.002</td><td>0.357± 0.003</td><td>-0.099</td></tr><tr><td rowspan="6">Amazon ESCI</td><td>Uncal.</td><td>0.764</td><td>0.047</td><td>0.528± 0.000</td><td>0.468± 0.001</td><td>-0.060</td><td>0.391± 0.000</td><td>0.525± 0.002</td><td>+0.133</td><td>0.161± 0.000</td><td>0.151± 0.000</td><td>-0.010</td></tr><tr><td>TS</td><td>0.764</td><td>0.047</td><td>0.526± 0.000</td><td>0.468± 0.001</td><td>-0.058</td><td>0.391± 0.000</td><td>0.524± 0.002</td><td>+0.133</td><td>0.161± 0.000</td><td>0.151± 0.000</td><td>-0.010</td></tr><tr><td>DIAG</td><td>0.764</td><td>0.045</td><td>0.525± 0.000</td><td>0.468± 0.001</td><td>-0.057</td><td>0.391± 0.000</td><td>0.524± 0.002</td><td>+0.133</td><td>0.161± 0.000</td><td>0.151± 0.000</td><td>-0.010</td></tr><tr><td>Spline</td><td>0.764</td><td>0.019</td><td>0.516± 0.001</td><td>0.468± 0.001</td><td>-0.048</td><td>0.391± 0.000</td><td>0.523± 0.003</td><td>+0.132</td><td>0.161± 0.001</td><td>0.151± 0.001</td><td>-0.010</td></tr><tr><td>h-cal</td><td>0.764</td><td>0.040</td><td>0.528± 0.003</td><td>0.468± 0.001</td><td>-0.060</td><td>0.391± 0.000</td><td>0.524± 0.001</td><td>+0.133</td><td>0.161± 0.000</td><td>0.151± 0.000</td><td>-0.010</td></tr><tr><td>SMART</td><td>0.764</td><td>0.021</td><td>0.516± 0.000</td><td>0.468± 0.001</td><td>-0.048</td><td>0.392± 0.000</td><td>0.525± 0.001</td><td>+0.133</td><td>0.162± 0.001</td><td>0.152± 0.000</td><td>-0.011</td></tr><tr><td rowspan="6">SciDocs</td><td>Uncal.</td><td>0.562</td><td>0.210</td><td>0.913± 0.000</td><td>0.679± 0.001</td><td>-0.234</td><td>0.480± 0.000</td><td>0.477± 0.004</td><td>-0.002</td><td>0.398± 0.000</td><td>0.381± 0.001</td><td>-0.017</td></tr><tr><td>TS</td><td>0.562</td><td>0.014</td><td>0.682± 0.000</td><td>0.679± 0.001</td><td>-0.003</td><td>0.480± 0.000</td><td>0.476± 0.006</td><td>-0.003</td><td>0.398± 0.000</td><td>0.382± 0.002</td><td>-0.016</td></tr><tr><td>DIAG</td><td>0.562</td><td>0.008</td><td>0.682± 0.000</td><td>0.679± 0.001</td><td>-0.003</td><td>0.480± 0.000</td><td>0.476± 0.007</td><td>-0.004</td><td>0.398± 0.000</td><td>0.382± 0.003</td><td>-0.016</td></tr><tr><td>Spline</td><td>0.562</td><td>0.015</td><td>0.682± 0.000</td><td>0.680± 0.000</td><td>-0.002</td><td>0.478± 0.002</td><td>0.474± 0.004</td><td>-0.005</td><td>0.397± 0.002</td><td>0.385± 0.001</td><td>-0.012</td></tr><tr><td>h-cal</td><td>0.562</td><td>0.142</td><td>0.776± 0.000</td><td>0.679± 0.001</td><td>-0.096</td><td>0.480± 0.000</td><td>0.477± 0.006</td><td>-0.003</td><td>0.398± 0.000</td><td>0.383± 0.002</td><td>-0.015</td></tr><tr><td>SMART</td><td>0.562</td><td>0.016</td><td>0.681± 0.000</td><td>0.680± 0.001</td><td>-0.001</td><td>0.480± 0.000</td><td>0.477± 0.006</td><td>-0.002</td><td>0.398± 0.000</td><td>0.381± 0.001</td><td>-0.017</td></tr><tr><td rowspan="6">WANDS</td><td>Uncal.</td><td>0.777</td><td>0.006</td><td>0.442± 0.000</td><td>0.436± 0.000</td><td>-0.005</td><td>0.463± 0.000</td><td>0.490± 0.001</td><td>+0.026</td><td>0.091± 0.000</td><td>0.088± 0.000</td><td>-0.003</td></tr><tr><td>TS</td><td>0.777</td><td>0.006</td><td>0.442± 0.000</td><td>0.436± 0.000</td><td>-0.005</td><td>0.463± 0.000</td><td>0.490± 0.001</td><td>+0.026</td><td>0.091± 0.000</td><td>0.088± 0.000</td><td>-0.003</td></tr><tr><td>DIAG</td><td>0.777</td><td>0.006</td><td>0.441± 0.000</td><td>0.436± 0.000</td><td>-0.005</td><td>0.464± 0.000</td><td>0.490± 0.001</td><td>+0.026</td><td>0.091± 0.000</td><td>0.088± 0.000</td><td>-0.003</td></tr><tr><td>Spline</td><td>0.777</td><td>0.009</td><td>0.442± 0.000</td><td>0.437± 0.001</td><td>-0.005</td><td>0.463± 0.001</td><td>0.489± 0.002</td><td>+0.025</td><td>0.091± 0.000</td><td>0.088± 0.000</td><td>-0.003</td></tr><tr><td>h-cal</td><td>0.777</td><td>0.009</td><td>0.442± 0.000</td><td>0.437± 0.001</td><td>-0.005</td><td>0.464± 0.000</td><td>0.490± 0.001</td><td>+0.026</td><td>0.091± 0.000</td><td>0.088± 0.000</td><td>-0.003</td></tr><tr><td>SMART</td><td>0.777</td><td>0.006</td><td>0.441± 0.000</td><td>0.437± 0.001</td><td>-0.005</td><td>0.463± 0.000</td><td>0.489± 0.001</td><td>+0.026</td><td>0.091± 0.000</td><td>0.088± 0.000</td><td>-0.003</td></tr><tr><td rowspan="6">Alloprof-Rerank</td><td>Uncal.</td><td>0.757</td><td>0.150</td><td>0.662± 0.000</td><td>0.306± 0.001</td><td>-0.357</td><td>0.443± 0.000</td><td>0.761± 0.000</td><td>+0.318</td><td>0.099± 0.000</td><td>0.059± 0.000</td><td>-0.040</td></table>

The ablation variants defined in Section 3 use the same data split, hyperparameter-selection protocol, and evaluation procedure. All experiments are implemented in PyTorch 2.1.0 with CUDA 12.1. GPU runs use a single NVIDIA RTX A5000 24GB GPU. Calibration and MRP fitting are performed on saved logits, so every calibrator and MRP comparison shares the same prediction source unless the base relevance model is explicitly retrained.

## 4.4 Evaluation Metrics

We evaluate the class-probability quality of the base calibrator with full-coverage accuracy, ECE, multiclass NLL, and Brier score. Since MRP does not modify $\hat { \mathbf { p } } ^ { \mathcal { A } }$ , these quantities must remain identical before and after applying MRP. The main table therefore first reports that Acc. and ECE are preserved, and then evaluates MRP with metrics for correctness-reliability reranking.

MRP is evaluated through the correctness reliability estimate $\hat { q } _ { i } ^ { \mathcal { A } }$ and the corresponding estimated error probability $1 - \hat { q } _ { i } ^ { \mathcal { A } }$ . Correctness negative log-likelihood, denoted as $\mathrm { N L L } _ { \mathrm { c o r r e c t } }$ , is defined

as

$$
\mathrm { N L L } _ { \mathrm { c o r r e c t } } = - \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \left[ Z _ { i } \log \hat { q } _ { i } ^ { \mathcal { A } } + \left( 1 - Z _ { i } \right) \log \bigl ( 1 - \hat { q } _ { i } ^ { \mathcal { A } } \bigr ) \right]
$$

and measures how well $\hat { q } _ { i } ^ { \mathcal { A } }$ predicts the probability that the fixed decision is correct. Lower values are better.

AUPR-Error follows failure-prediction and misclassification-detection evaluations [3, 15]. It is the standard area under the precision–recall curve (AUPR), computed for detecting incorrect predictions rather than relevant items. Let $W _ { i } = 1 - Z _ { i }$ be the wrong-prediction indicator, let $s _ { i } = 1 - \hat { q } _ { i } ^ { \mathcal { A } }$ be the estimated error score, and let � rank examples by decreasing �<sub>�</sub>. We compute

$$
{ \mathrm { A U P R - E r r o r } } = { \frac { 1 } { \sum _ { i } W _ { i } } } \sum _ { j = 1 } ^ { n } W _ { \pi _ { j } } { \frac { \sum _ { t = 1 } ^ { j } W _ { \pi _ { t } } } { j } } .
$$

It measures whether predictions flagged as high risk are actually wrong.

AUPR-Error is important for selective prediction and fallback routing, where the operational goal is often to prioritize likely failures for review or fallback rather than to change every predicted probability. Higher values are better.

Area under the risk-coverage curve, or AURC, measures the area under the coverage-risk curve obtained by accepting predictions from low risk to high risk. Given a ranking � by increasing

estimated risk,

$$
{ \mathrm { A U R C } } = { \frac { 1 } { n } } \sum _ { j = 1 } ^ { n } { \frac { 1 } { j } } \sum _ { t = 1 } ^ { j } ( 1 - Z _ { \pi _ { t } } )
$$

and lower values are better.

Finally, we report selective accuracy to simulate a budgeted fallback setting. At coverage �, the system automatically handles the lowest-risk �� predictions and routes the remaining 1−� fraction to fallback or review. We define

$$
\mathrm { S e l A c c } @ \tau = \frac { 1 } { | \mathcal { R } _ { \tau } | } \sum _ { i \in \mathcal { R } _ { \tau } } Z _ { i }
$$

where $\mathcal { R } _ { \tau }$ is the set ofretained predictions with the lowest estimated risk. This metric tests whether MRP places usable predictions earlier under a limited fallback budget. We use $\tau \in \{ 0 . 1 , 0 . 5 , 0 . 7 , 0 . 9 \}$ to examine whether the reranking gain remains stable across diferent fallback budgets.

## 5 Results and Analysis

## 5.1 Main Reliability Results

The first analysis asks whether MRP addresses the intended reliability task after calibration. The evaluation has two questions. First, does MRP preserve the class probabilities and full-coverage predictions produced by the base calibrator? Second, without changing these probabilities, does it rerank calibrated predictions by correctness reliability more efectively? Table 2 reports both aspects.

For any base calibrator A, MRP leaves $\hat { \mathbf { p } } ^ { \mathcal { \bar { A } } } ( x ) , d ( x ) , \hat { c } ^ { \mathcal { \bar { A } } } ( x )$ , and �(�) unchanged. Therefore, Acc. and ECE must be identical before and after applying MRP, and the table reports these quantities once for each base calibrator. In contrast, $\mathrm { N L L } _ { \mathrm { c o r r e c t : } }$ , AUPR-Error, and AURC compare the no-projection confidence baseline $\hat { q } _ { \emptyset } ^ { \mathcal { A } }$ with the MRP reliability score $\hat { q } _ { \mathrm { M R P } } ^ { \mathcal { A } }$

Across the evaluated settings, MRP improves the main correctnessreliability metrics on average while leaving Acc. and ECE unchanged. The largest gains appear when calibrated confidence alone is a weak reranking signal, as in MSLR-WEB10K and Alloprof Rerank. These improvements are not changes in prediction accuracy. They mean that the same fixed predictions are ranked more usefully for identifying likely errors.

The table also shows why this task is diferent from ordinary calibration. Some calibrators substantially improve ECE but leave almost the same confidence-baseline reliability reranking. ESCI-Rerank-US is a clear example. TS, DIAG, Spline, and SMART sharply reduce ECE, but their confidence-baseline AUPR-Error remains nearly unchanged. After MRP, AUPR-Error improves for the same fixed decisions. Thus, probability calibration improves the scale of confidence, whereas MRP improves how calibrated predictions are prioritized by residual correctness risk.

The gains are not uniform in every metric. SciDocs improves in $\mathrm { N L L } _ { \mathrm { c o r r e c t } }$ and AURC, but AUPR-Error is nearly unchanged or slightly lower. This exception clarifies the scope of the method. MRP does not uniformly improve every risk metric. It is most efective when label-conditioned residual reliability provides a usable reranking signal. Even here, the reduction in AURC indicates that the retained low-risk prefix becomes better ranked, while error detection by AUPR remains dificult.

Table 3: Budgeted fallback simulation. At automatic coverage $\tau ,$ each reliability score keeps the lowest-risk � fraction for automatic use and routes the remaining highest-risk predictions to fallback or review. Values compare MRP with the no-projection confidence baseline $\hat { q } _ { \varnothing } ^ { \mathcal { R } } = \hat { c } ^ { \mathcal { R } }$ . Positive values indicate a better reliability reranking.

<table><tr><td rowspan="2">Dataset</td><td colspan="4">∆SelAcc@τ</td></tr><tr><td>10%</td><td>50%</td><td>70%</td><td>90%</td></tr><tr><td>ESCI-Rerank-US</td><td>+0.043</td><td>+0.082</td><td>+0.056</td><td>+0.016</td></tr><tr><td>MSLR-WEB10K</td><td>+0.331</td><td>+0.174</td><td>+0.113</td><td>+0.028</td></tr><tr><td>Amazon ESCI</td><td>+0.003</td><td>+0.007</td><td>+0.014</td><td>+0.030</td></tr><tr><td>SciDocs</td><td>+0.074</td><td>-0.003</td><td>-0.003</td><td>-0.001</td></tr><tr><td>WANDS</td><td>+0.005</td><td>+0.002</td><td>+0.004</td><td>+0.004</td></tr><tr><td>Alloprof-Rerank</td><td>+0.000</td><td>+0.050</td><td>+0.101</td><td>+0.041</td></tr><tr><td>Average</td><td>+0.076</td><td>+0.052</td><td>+0.047</td><td>+0.020</td></tr></table>

## 5.2 Budgeted Fallback Performance

Table 3 gives a more operational view of the same reliability reranking problem. It reports selective-accuracy gains over the confidence baseline at coverage �. Here, a system routes the highest-risk predictions to fallback or review and automatically handles only the retained lowest-risk predictions. The goal is not to change the predicted probabilities, but to decide which calibrated predictions should remain in automatic use under a limited fallback budget.

Positive values mean that MRP keeps a more accurate retained set than the confidence baseline. Average gains are positive at every coverage level and largest when few predictions can be used automatically. This is where reliability reranking matters most, because few can be accepted without fallback.

The dataset-level pattern is also informative. MSLR-WEB10K shows the strongest gains, while ESCI-Rerank-US and Alloprof-Rerank also improve at moderate coverages. Amazon ESCI and WANDS show smaller gains, and SciDocs is close to neutral after the most selective setting. This matches Table 2, where SciDocs had little AUPR-Error gain.

Overall, the fallback simulation gives MRP an operational interpretation. The method does not claim that class probabilities change or that the model becomes more accurate at full coverage. It shows that, when only a fraction can be handled automatically, the same calibrated predictions can be ranked more usefully.

## 5.3 Evidence for Label-wise Residual Reliability

We next test whether confidence alone is suficient for reliability. If true, residual reliability would be similar across predicted labels within each confidence region.

To diagnose this efect, we group predictions by confidence and measure how much the average residual $Z - \hat { c } ^ { \mathcal { A } }$ varies across predicted labels. For a confidence group $B ,$ we define

$$
\begin{array} { r l } & { \operatorname { S p r e a d } ( B ) = \underset { k } { \operatorname* { m a x } } \mathbb { E } [ Z - \hat { c } ^ { \mathcal { A } } \mid X \in B , D = k ] } \\ & { \qquad - \underset { k } { \operatorname* { m i n } } \mathbb { E } [ Z - \hat { c } ^ { \mathcal { A } } \mid X \in B , D = k ] . } \end{array}
$$

If this quantity is close to zero, then the residual reliability is nearly the same across predicted labels within the same confidence group.

![](images/a8f8a4a23ac3ee619527d16bd8277c354e19278dbe7e2e6ac040571bebf591b7.jpg)  
Figure 2: Learned label-wise reliability curves $T _ { k } ( c ) _ { \cdot }$ , shown under SMART calibration as a representative base calibrator. The dashed line is the identity $q = c .$ The curves show that equal calibrated confidence can correspond to diferent correctness reliability depending on the predicted relevance label.

Table 4: Label-conditioned reliability spread. Within each calibrated-confidence group, we measure how much $Z - \hat { c } ^ { \mathcal { A } }$ changes across predicted labels. A large spread indicates that equal-confidence predictions can have label-dependent cor rectness reliability. The random column is the 95th percentile after shufling labels within each group.
<table><tr><td>Dataset</td><td>Label spread</td><td>Random</td><td>Ratio</td><td>Seeds</td></tr><tr><td>ESCI-Rerank-US</td><td>26.7pp</td><td>3.8pp</td><td>6.9×</td><td>18/18</td></tr><tr><td>MSLR-WEB10K</td><td>59.8pp</td><td>4.3pp</td><td>13.8×</td><td>18/18</td></tr><tr><td>Amazon ESCI</td><td>60.8pp</td><td>8.1pp</td><td>7.5×</td><td>18/18</td></tr><tr><td>SciDocs</td><td>6.6pp</td><td>4.2pp</td><td>1.6×</td><td>18/18</td></tr><tr><td>WANDS</td><td>14.3pp</td><td>3.0pp</td><td>4.8×</td><td>18/18</td></tr><tr><td>Alloprof-Rerank</td><td>57.3pp</td><td>4.9pp</td><td>11.6×</td><td>18/18</td></tr></table>

A large value means that two predictions with similar calibrated confidence can have diferent residual errors depending on which label was predicted. Table 4 reports the average label spread across confidence groups for SMART as a representative calibrator. The random column shows how large the spread would be if predicted labels were shufled inside each confidence group. In every dataset, the label spread is larger than this random baseline, and the pattern appears in all three protocol seeds. Thus, the confidence-only assumption from Section 3 fails empirically. Residual reliability remains label-dependent within confidence groups.

The spread explains why label-wise reliability functions are useful. MSLR-WEB10K, Amazon ESCI, and Alloprof-Rerank show the largest spreads, far above their shufled-label baselines. These are also the datasets where the reliability reranking gains in Table 2 are large. ESCI-Rerank-US has a smaller but clear spread, while SciDocs has the weakest signal. This is consistent with the weaker AUPR-Error improvement above.

Figure 2 visualizes the learned functions $T _ { k } ( c )$ . The dashed iden tity line $q = c$ uses base confidence directly. The learned curves show that the same calibrated confidence can map to diferent reliability values depending on the predicted label. This illustrates that MRP is not merely smoothing confidence again. It estimates separate correctness reliability curves for diferent decision labels.

Table 5: Structural ablation averaged over all relevanceprediction settings. C-NLL denotes $\mathrm { N L L } _ { \mathrm { c o r r e c t } } .$ . Variants are defined in Section 3.5. MRP denotes the main label-wise 1D projection.
<table><tr><td>Variant</td><td></td><td>C-NLL↓ AUPR-Error↑</td><td>AURC↓</td><td>SelAcc@50↑</td></tr><tr><td>Conf.</td><td>0.6181</td><td>0.4876</td><td>0.2775</td><td>0.7279</td></tr><tr><td>Shared 1D</td><td>0.5720</td><td>0.4876</td><td>0.2775</td><td>0.7279</td></tr><tr><td>Label-only intercept</td><td>0.5132</td><td>0.6160</td><td>0.2302</td><td>0.7773</td></tr><tr><td>Per-label isotonic</td><td>0.5166</td><td>0.6148</td><td>0.2294</td><td>0.7782</td></tr><tr><td>MRP (Label-wise 1D)</td><td>0.4977</td><td>0.6185</td><td>0.2273</td><td>0.7800</td></tr><tr><td>Label-wise 2D</td><td>0.4975</td><td>0.6184</td><td>0.2274</td><td>0.7795</td></tr></table>

## 5.4 Structural Ablations

Table 5 shows which structural assumption is responsible for the gain. Shared 1D improves NLL<sub>correct</sub> over the confidence baseline, but it does not improve AUPR-Error, AURC, or SelAcc@50. This means that simply relearning a shared confidence-to-correctness curve can improve probability fit, but it does not improve risk reranking.

These variants form a nested diagnostic sequence. Shared 1D tests for a global confidence-to-correctness mismatch. The labelonly intercept tests whether the gain can be explained by static predicted-label bias, and per-label isotonic tests a nonparametric label-wise monotone mapping. MRP then asks whether the same label-conditioned structure can be represented by smooth monotone reliability curves.

The reranking metrics improve only after the decision label is introduced. The strong performance of the label-conditioned variants should not be read as evidence against MRP. Rather, it identifies the main residual structure that MRP is designed to model. Much of the residual signal after calibration is label-conditioned. MRP, the label-wise 1D model, provides a smooth monotone implementation of this idea by learning $T _ { k } ( c )$ for each label. Compared with per-label isotonic projection, this smooth lattice gives a clearer $\mathrm { N L L } _ { \mathrm { c o r r e c t } }$ improvement while maintaining comparable reranking quality. Its gains over label-only and per-label isotonic variants are modest, so the result should not be read as MRP dominating all label-wise alternatives. Instead, it shows that the key ingredient is label-conditioned residual reliability, with MRP serving as the main monotone projection used in this paper.

Label-wise 2D, which uses the top-runner gap, is nearly identical to Label-wise 1D. Its $\mathrm { N L L } _ { \mathrm { c o r r e c t } }$ is slightly lower, but AUPR-Error, AURC, and SelAcc@50 do not improve. This supports the final design choice. The main residual structure in these relevance datasets is the label-wise confidence-to-reliability curve $T _ { k } ( c )$ , not an additional two-dimensional decision-gap surface. We therefore use label-wise 1D projection as the main method.

As a robustness check, we test whether this signal is specific to the lightweight linear base predictor. On MSLR-WEB10K, a balanced LightGBM multiclass relevance predictor raises base accuracy from 0.394 to 0.518 and reduces ECE from 0.292 to 0.014. Even with this stronger and well-calibrated base, MRP improves the confidence baseline from 0.629 to 0.586 in $\mathrm { N L L } _ { \mathrm { c o r r e c t } } .$ , from 0.631 to 0.730 in AUPR-Error, and from 0.318 to 0.286 in AURC. This reinforces the same conclusion as Table 5: the persistent signal is label-conditioned residual reliability, and MRP is the smooth monotone projection used as our main variant.

## 6 Reliability-to-Probability Projection

## 6.1 Simplex View of Top-label Projection

A class probability vector is a point on the probability simplex. The vertex e<sub>�</sub> assigns probability one to class �. For a fixed decision $d ( x )$ the simplex center has top-label probability $1 / K ,$ , while the decisionlabel vertex $\mathbf { e } _ { d ( x ) }$ has top-label probability one. Thus, changing the confidence assigned to the fixed decision can be viewed as moving a probability vector toward or away from that vertex.

Temperature or power transforms trace an order-preserving path from the uniform distribution toward the argmax-class vertex. Related simplex views of calibrated predictive distributions are also used in recent uncertainty-calibration work [14]. This matters because MRP estimates a probability-valued scalar,

$$
q _ { \mathrm { M R P } } ^ { \mathcal { A } } ( x ) = T _ { d ( x ) } ( \hat { c } ^ { \mathcal { A } } ( x ) ) ,
$$

for the correctness event $Y = d ( x )$ , but uses it for reliability rerank ing rather than as a multiclass probability vector.

For a probability vector p, the power-normalized path

$$
p _ { k } ( \alpha ) = \frac { \mathcal P _ { k } ^ { \alpha } } { \sum _ { j } \mathcal P _ { j } ^ { \alpha } }
$$

gives a one-dimensional trajectory. A $\tan = 1 ,$ , it returns the original vector. As $\alpha  0 ,$ , it approaches the simplex center, and as $\alpha  \infty ,$ it concentrates on the largest-probability class. When $d ( x )$ remains the top class, this path sweeps the decision-label probability from $1 / K$ toward 1 without changing the class order.

This view lets us ask whether $q _ { \mathrm { M R P } } ^ { \mathcal { R } } ( x )$ can be realized as the top-label probability of a decision-preserving class probability vector. We define this optional extension as MRP-induced reliability calibration (MRC), a compatibility analysis between the learned reliability space and class-probability geometry.

## 6.2 MRC Construction

MRC starts from the calibrated vector $\hat { \mathbf { p } } ^ { \mathcal { A } } ( x )$ and the MRP target $q _ { \mathrm { M R P } } ^ { \mathcal { R } } ( x )$ . It moves $\hat { \mathbf { p } } ^ { \mathcal { A } } ( x )$ along a sample-wise power-temperature path so that the fixed top-label probability matches the target as

![](images/2d90c69f25379fcfdd24e9481f453a47cf27623e4a8b0c1ec79af3769c0b64c2.jpg)  
Figure 3: Schematic of MRC on the probability simplex. The numbered markers show the sequence from the uncalibrated prediction to the calibrated prediction, the MRP target contour $p _ { d ( x ) } = q _ { \mathrm { M R P } } ^ { \mathcal { A } } ( x )$ , and the MRC projection. The projection moves along the sample-wise temperature path and realizes the MRP target as a fixed top-label probability when feasible.

closely as possible:

$$
\hat { p } _ { k } ^ { \mathcal { A } , \alpha } ( x ) = \frac { ( \hat { p } _ { k } ^ { \mathcal { A } } ( x ) ) ^ { \alpha } } { \sum _ { j } ( \hat { p } _ { j } ^ { \mathcal { A } } ( x ) ) ^ { \alpha } }
$$

which is equivalent to sample-wise temperature scaling oflog $\hat { \mathbf { p } } ^ { \mathcal { A } } ( x )$ If $d ( x )$ is the top label and $q _ { \mathrm { M R P } } ^ { \mathcal { R } } ( x ) \in \mathsf { \bar { [ 1 / K , 1 ] } }$ , we can find $\alpha _ { i } \geq 0$ such that

$$
\hat { p } _ { d ( x _ { i } ) } ^ { \mathcal { A } , \alpha _ { i } } ( x _ { i } ) = q _ { \mathrm { M R P } } ^ { \mathcal { A } } ( x _ { i } ) .
$$

When an exact solution exists, MRC chooses it. Otherwise, it uses the closest feasible boundary point on the same path. With selected parameter $\alpha _ { i } ^ { \star }$

$$
\hat { \boldsymbol { \mathsf { p } } } ^ { \mathrm { M R C } } ( \boldsymbol { x } _ { i } ) = \hat { \boldsymbol { \mathsf { p } } } ^ { \mathcal { A } , \alpha _ { i } ^ { \star } } ( \boldsymbol { x } _ { i } ) .
$$

Unlike standard TS, MRC does not learn one global temperature. It chooses a sample-wise endpoint from $q _ { \mathrm { M R P } } ^ { \mathcal { R } } ( x ) . \mathrm { I f } q _ { \mathrm { M R P } } ^ { \mathcal { R } } ( x ) < 1 / K$ the target cannot be represented exactly on this path. If � (�) is no longer the top label after calibration, the decision-preserving path is not directly applicable. We report feasibility with the projection results. Figure 3 illustrates the process.

## 6.3 Projection Results and Feasibility

Table 6 tests whether MRP reliability estimates fit top-label probability geometry. MRC finds a sample-wise $\alpha _ { i } ^ { \star }$ on the power-temperature path and compares the resulting vector with the calibrated one. This is stricter than reliability reranking because the score must be representable as a top-label probability.

The results are mixed but informative. MRC often improves NLL and Brier score, and it can reduce ECE when base probabilities are poorly calibrated. However, it does not uniformly improve fixed ECE or top-label ECE. For strong calibrators, the projection can move away from the best calibration point. Thus, probability correction is more constrained than reliability reranking.

The feasibility columns make this point concrete. In Amazon ESCI and WANDS, only 7.2% and about 1.0% of predictions have $q _ { \mathrm { M R P } } < 1 / K$ , so most targets are feasible. In ESCI-Rerank-US, MSLR-WEB10K, and Alloprof-Rerank, this rate is larger, around 25–38%, so many low-reliability predictions hit the simplex boundary. Thus, MRC is best viewed as a compatibility analysis of whether the MRP reliability space embeds into probability geometry.

Table 6: Top-label simplex projection analysis. Base is the original calibrated vector, and MRC is the MRP-induced powertemperature projection. For Fixed ECE, Top ECE, NLL, and Brier, Δ is MRC minus Base. Blue bold negative Δ values indicate improvements, and lower is better for all four metrics. The $q < 1 / K$ column reports infeasible targets. Solved gives exact rates.
<table><tr><td rowspan="2">Dataset</td><td rowspan="2">K Cal.</td><td colspan="3">Fixed ECE↓</td><td colspan="3">Top ECE↓</td><td colspan="3">NLL↓</td><td colspan="3">Brier↓</td><td rowspan="2"></td><td rowspan="2">q &lt; 1/K Solved</td></tr><tr><td>Base</td><td>MRC</td><td>Δ</td><td>Base</td><td>MRC</td><td>Δ</td><td>Base</td><td>MRC</td><td>∆</td><td>Base</td><td>MRC Δ</td><td></td></tr><tr><td rowspan="5">ESCI-Rerank US</td><td rowspan="5">Uncal. TS DIAG 2</td><td>0.1835 0.0131</td><td>0.0494 0.0482</td><td>-0.1341 +0.0351</td><td>0.1835 0.0131</td><td>0.0494 0.0482</td><td>-0.1341 +0.0351</td><td>0.8323 0.6793</td><td>0.6543</td><td>-0.1780</td><td>0.5722</td><td>0.4620</td><td>-0.1103</td><td>37.6%</td><td>100.0%</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.6541</td><td>-0.0252</td><td>0.4863</td><td>0.4618</td><td>-0.0245</td><td>37.6%</td><td>100.0%</td></tr><tr><td>0.0111</td><td>0.0476</td><td>+0.0365</td><td>0.0111</td><td>0.0476</td><td>+0.0365</td><td>0.6795</td><td>0.6538</td><td>-0.0257</td><td>0.4865</td><td>0.4616</td><td>-0.0249</td><td>37.7%</td><td>100.0%</td></tr><tr><td>Spline 0.0153</td><td>0.0413</td><td>+0.0260</td><td>0.0114</td><td>0.0452</td><td>+0.0338</td><td>0.6802</td><td>0.6558</td><td>-0.0244</td><td>0.4872</td><td>0.4635</td><td>-0.0237</td><td>37.7%</td><td>91.7%</td></tr><tr><td>h-cal 0.1122 SMART 0.0074</td><td>0.0506 0.0478</td><td>-0.0616 +0.0404</td><td>0.1122 0.0074</td><td>0.0506 0.0478</td><td>-0.0616 +0.0404</td><td>0.7338 0.6802</td><td>0.6541 0.6543</td><td>-0.0797 -0.0259</td><td>0.5242</td><td>0.4618 0.4620</td><td>-0.0624 -0.0252</td><td>37.4% 38.5%</td><td>100.0% 100.0%</td></tr><tr><td rowspan="6">MSLR-</td><td>Uncal.</td><td>0.2922</td><td>0.0703</td><td>-0.2219</td><td>0.2922</td><td>0.0762</td><td>-0.2160</td><td>2.1775</td><td>1.2081</td><td>-0.9694</td><td>0.4872 0.8695</td><td>0.6274</td><td>-0.2422</td><td>31.9%</td><td>100.0%</td></tr><tr><td>TS</td><td>0.0697</td><td>0.0681</td><td>-0.0016</td><td>0.0697</td><td>0.0750</td><td>+0.0052</td><td>1.4052</td><td>1.2090</td><td>-0.1963</td><td>0.7302</td><td>0.6262</td><td>-0.1039</td><td>32.1%</td><td>100.0%</td></tr><tr><td>DIAG</td><td>0.0964</td><td>0.0785</td><td>-0.0178</td><td>0.0964</td><td>0.0857</td><td>-0.0107</td><td>1.3380</td><td>1.3085</td><td>-0.0294</td><td>0.7156</td><td>0.6321</td><td>-0.0835</td><td>32.0%</td><td>100.0%</td></tr><tr><td>Spline</td><td>0.0107</td><td>0.0545</td><td>+0.0438</td><td>0.0950</td><td>0.1151</td><td>+0.0201</td><td>1.4763</td><td>1.4506</td><td>-0.0256</td><td>0.7481</td><td>0.6973</td><td>-0.0508</td><td>31.9%</td><td>57.9%</td></tr><tr><td>h-cal</td><td>0.0549</td><td>0.0679</td><td>+0.0130</td><td>0.0548</td><td>0.0747</td><td>+0.0199</td><td>1.4231</td><td>1.2307</td><td>-0.1924</td><td>0.7330</td><td>0.6316</td><td>-0.1014</td><td>32.0%</td><td>99.9%</td></tr><tr><td rowspan="6"></td><td>SMART</td><td>0.1137</td><td>0.0709</td><td>-0.0428</td><td>0.1137</td><td>0.0772</td><td>-0.0365</td><td>1.3301</td><td>1.2178</td><td>-0.1123</td><td>0.6998</td><td>0.6296</td><td>-0.0702</td><td>32.0%</td><td>100.0%</td></tr><tr><td>Uncal.</td><td>0.0470</td><td></td><td></td><td>0.0470</td><td>0.0194</td><td></td><td></td><td></td><td></td><td></td><td>0.3496</td><td>-0.0212</td><td>7.2%</td><td>100.0%</td></tr><tr><td></td><td>0.0469</td><td>0.0136</td><td>-0.0334</td><td>0.0469</td><td>0.0194</td><td>-0.0276</td><td>0.7132</td><td>0.7191</td><td>+0.0060</td><td>0.3708</td><td></td><td></td><td>7.2%</td><td></td></tr><tr><td>TS</td><td>0.0453</td><td>0.0135</td><td>-0.0334</td><td></td><td></td><td>-0.0275</td><td>0.7107</td><td>0.7192</td><td>+0.0085</td><td>0.3710</td><td>0.3496</td><td>-0.0213</td><td></td><td>100.0%</td></tr><tr><td>DIAG</td><td>0.0188</td><td>0.0134 0.0122</td><td>-0.0318 -0.0066</td><td>0.0453 0.0178</td><td>0.0193 0.0177</td><td>-0.0260 0.0000</td><td>0.7103 0.7012</td><td>0.7230</td><td>+0.0126</td><td>0.3709 0.3672</td><td>0.3497 0.3497</td><td>-0.0213 -0.0176</td><td>7.2% 7.0%</td><td>100.0% 98.8%</td></tr><tr><td>Spline h-cal</td><td>0.0400</td><td>0.0137</td><td>-0.0263</td><td>0.0400</td><td>0.0192</td><td>-0.0208</td><td>0.7134</td><td>0.7078 0.7090</td><td>+0.0066 -0.0043</td><td>0.3700</td><td>0.3495</td><td>-0.0205</td><td>7.1%</td><td>100.0%</td></tr><tr><td rowspan="6">SciDocs</td><td>SMART</td><td>0.0206</td><td>0.0140</td><td>-0.0066</td><td>0.0206</td><td>0.0197</td><td>-0.0010</td><td>0.7024</td><td>0.7192</td><td>+0.0167</td><td>0.3674</td><td>0.3496</td><td>-0.0179</td><td>7.2%</td><td>100.0%</td></tr><tr><td>Uncal.</td><td>0.2097</td><td>0.0088</td><td>-0.2008</td><td>0.2097</td><td>0.0103</td><td>-0.1993</td><td>0.9131</td><td>0.6788</td><td>-0.2343</td><td>0.5962</td><td>0.4860</td><td>-0.1102</td><td>13.0%</td><td>100.0%</td></tr><tr><td>TS</td><td>0.0139</td><td>0.0074</td><td>-0.0065</td><td>0.0139</td><td>0.0078</td><td>-0.0061</td><td>0.6818</td><td>0.6789</td><td>-0.0029</td><td>0.4887</td><td>0.4862</td><td>-0.0025</td><td>8.6%</td><td>100.0%</td></tr><tr><td>DIAG</td><td>0.0081</td><td>0.0070</td><td>-0.0010</td><td>0.0081</td><td>0.0073</td><td>-0.0008</td><td>0.6815</td><td>0.6788</td><td>-0.0027</td><td>0.4884</td><td>0.4861</td><td>-0.0023</td><td>9.0%</td><td>100.0%</td></tr><tr><td>Spline</td><td>0.0152</td><td>0.0127</td><td>-0.0024</td><td>0.0136</td><td>0.0103</td><td>-0.0033</td><td>0.6819</td><td>0.6802</td><td>-0.0017</td><td>0.4889</td><td>0.4872</td><td>-0.0016</td><td>6.8%</td><td>89.0%</td></tr><tr><td>h-cal SMART</td><td>0.1417</td><td>0.0054</td><td>-0.1363</td><td>0.1417</td><td>0.0059</td><td>-0.1358</td><td>0.7755</td><td>0.6791</td><td>-0.0964</td><td>0.5434</td><td>0.4864</td><td>-0.0571</td><td>10.9%</td><td>100.0%</td></tr><tr><td rowspan="5">WANDS</td><td></td><td></td><td>0.0162 0.0119</td><td>-0.0043</td><td>0.0162</td><td>0.0126</td><td>-0.0036</td><td>0.6812</td><td>0.6797</td><td>-0.0016</td><td>0.4882</td><td>0.4868</td><td>-0.0014</td><td>8.8%</td><td>100.0%</td></tr><tr><td>Uncal.</td><td>0.0059</td><td>0.0077</td><td>+0.0018</td><td>0.0059</td><td>0.0081</td><td>+0.0022</td><td>0.5506</td><td>0.5517</td><td>+0.0011</td><td>0.3140</td><td>0.3124</td><td>-0.0016</td><td>1.0%</td><td>100.0%</td></tr><tr><td>TS</td><td>0.0058</td><td>0.0077</td><td>+0.0019</td><td>0.0058</td><td>0.0080</td><td>+0.0022</td><td>0.5506</td><td>0.5517</td><td>+0.0011</td><td>0.3140</td><td>0.3124</td><td>-0.0016</td><td>1.0% -0.0016 1.0%</td><td>100.0%</td></tr><tr><td>DIAG Spline</td><td>0.0058</td><td>0.0075</td><td>+0.0017</td><td>0.0058</td></table>

The central contribution remains MRP: a label-wise correctnessreliability space for reranking predictions after calibration. MRC shows that this space can sometimes be embedded through constrained top-label projection, but it should not be read as a replacement for standard multiclass calibration.

## 7 Limitations

MRP is designed for post-calibration reliability reranking, not for improving the underlying relevance model or replacing a multiclass calibrator. Because it preserves the predicted relevance label and calibrated class probabilities, it cannot fix errors that require chang ing the model decision. The fixed-decision protocol is useful when systems must decide which existing predictions to trust, but it does not address candidate generation, retrieval quality, or end-to-end relevance ranking.

The method is most useful when residual reliability difers across predicted labels after calibration. When this structure is weak, as in SciDocs, confidence already captures much of the available signal and MRP has less room to improve reranking. MRC is only a secondary analysis: not all correctness-reliability estimates are feasible top-label probabilities. Extending MRP to query-dependent reliability functions, richer decision contexts, and joint optimization with candidate ranking remains future work.

## 8 Conclusion

We study residual reliability reranking after calibration in relevance prediction and introduce MRP, a label-wise monotone projection method. Even when calibrated probabilities are fixed, predictions with similar confidence can have diferent correctness reliability depending on the predicted relevance label. Instead of changing class probabilities, MRP learns label-wise correctness-reliability curves and reranks fixed predictions by residual risk.

Across six information access datasets and multiple post-hoc calibrators, MRP broadly improves $\mathrm { N L L } _ { \mathrm { c o r r e c t } }$ and AURC, improves fallback performance on average, and improves AUPR-Error when label-conditioned residual signal is present, while preserving fullcoverage accuracy and ECE. The ablations show that the main signal comes from label-wise residual reliability rather than from a shared confidence remapping or an additional decision-gap dimension. These results suggest that calibration does not end the reliability problem. Calibrated confidence gives a meaningful probability scale, but systems that must decide which predictions to use, defer, or review still need reliability reranking.

## References

[1] Yan Chen, Shujian Liu, Zheng Liu, Weiyi Sun, Linas Baltrunas, and Benjamin Schroeder. 2022. Wands: Dataset for product search relevance assessment. In European Conference on Information Retrieval. Springer, 128–141.

[2] Arman Cohan, Sergey Feldman, Iz Beltagy, Doug Downey, and Daniel S Weld. 2020. Specter: Document-level representation learning using citation-informed transformers. In Proceedings ofthe 58th annual meeting ofthe association for computational linguistics. 2270–2282.

[3] Charles Corbière, Nicolas Thome, Avner Bar-Hen, Matthieu Cord, and Patrick Pérez. 2019. Addressing failure prediction by learning model confidence. Advances in neural information processing systems 32 (2019).

[4] Corinna Cortes, Giulia DeSalvo, and Mehryar Mohri. 2016. Learning with rejec tion. In International conference on algorithmic learning theory. Springer, 67–82.

[5] Nick Craswell, Bhaskar Mitra, Emine Yilmaz, Daniel Campos, andJimmy Lin. 2021. Ms marco: Benchmarking ranking models in the large-data regime. In Proceedings ofthe 44th international ACM SIGIR conference on research and development in information retrieval. 1566–1576

[6] Ran El-Yaniv et al. 2010. On the Foundations of Noise-free Selective Classification. Journal ofMachine Learning Research 11, 5 (2010).

[7] Yonatan Geifman and Ran El-Yaniv. 2017. Selective classification for deep neural networks. Advances in neural information processing systems 30 (2017).

[8] Yonatan Geifman and Ran El-Yaniv. 2019. Selectivenet: A deep neural network with an integrated reject option. In International conference on machine learning. PMLR, 2151–2159.

[9] Chuan Guo, Geof Pleiss, Yu Sun, and Kilian Q Weinberger. 2017. On calibration of modern neural networks. In International conference on machine learning. PMLR, 1321–1330.

[10] Haolan Guo, Linwei Tao, Haoyang Luo, Minjing Dong, and Chang Xu. 2025. Sample Margin-Aware Recalibration of Temperature Scaling. arXiv preprint arXiv:2506.23492 (2025).

[11] Chirag Gupta and Aaditya Ramdas. 2022. Top-label calibration and multiclass-tobinary reductions. In International Conference on Learning Representations.

[12] Kartik Gupta, Amir Rahimi, Thalaiyasingam Ajanthan, Thomas Mensink, Cristian Sminchisescu, and Richard Hartley. 2020. Calibration of neural networks using splines. arXiv preprint arXiv:2006.12800 (2020).

[13] Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, and Mingwei Chang. 2020. Retrieval augmented language model pre-training. In International conference on machine learning. PMLR, 3929–3938.

[14] Jakob Heiss, Sören Lambrecht, Jakob Weissteiner, Hanna Wutte, Žan Žurič, Josef Teichmann, and Bin Yu. 2026. JUCAL: Jointly Calibrating Aleatoric and Epistemic Uncertainty in Classification Tasks. arXiv preprint arXiv:2602.20153 (2026).

[15] Dan Hendrycks and Kevin Gimpel. 2016. A baseline for detecting misclassified and out-of-distribution examples in neural networks. arXiv preprint arXiv:1610.02136 (2016).

[16] Wenjian Huang, Guiping Cao, Jiahao Xia, Jingkun Chen, Hao Wang, and Jianguo Zhang. 2025. h-calibration: Rethinking Classifier Recalibration with Probabilistic Error-Bounded Objective. IEEE Transactions on Pattern Analysis and Machine Intelligence (2025).

[17] Gautier Izacard and Edouard Grave. 2021. Leveraging passage retrieval with generative models for open domain question answering. In Proceedings ofthe 16th conference ofthe european chapter ofthe association for computational linguistics: main volume. 874–880.

[18] Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih. 2020. Dense passage retrieval for opendomain question answering. In Proceedings ofthe 2020 conference on empirical methods in natural language processing (EMNLP). 6769–6781.

[19] Meelis Kull, Miquel Perello Nieto, Markus Kängsepp, Telmo Silva Filho, Hao Song, and Peter Flach. 2019. Beyond temperature scaling: Obtaining well-calibrated multi-class probabilities with dirichlet calibration. Advances in neural information processing systems 32 (2019).

[20] Ananya Kumar, Percy S Liang, and Tengyu Ma. 2019. Verified uncertainty calibration. Advances in neural information processing systems 32 (2019).

[21] Aviral Kumar, Sunita Sarawagi, and Ujjwal Jain. 2018. Trainable calibration measures for neural networks from kernel mean embeddings. In International Conference on Machine Learning. PMLR, 2805–2814.

[22] Antoine Lefebvre-Brossard, Stephane Gazaille, and Michel C Desmarais. 2023. Alloprof: a new french question-answer education dataset and its use in an information retrieval case study. arXiv preprint arXiv:2302.07738 (2023).

[23] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, et al. 2020. Retrieval-augmented generation for knowledge-intensive nlp tasks. Advances in neural information processing systems 33 (2020), 9459–9474.

[24] Tie-Yan Liu. 2009. Learning to Rank for Information Retrieval. Foundations and Trends in Information Retrieval 3, 3 (2009), 225–331. doi:10.1561/1500000016

[25] David Madras, Toni Pitassi, and Richard Zemel. 2018. Predict responsibly: improv ing fairness and accuracy by learning to defer. Advances in neural information processing systems 31 (2018).

[26] Microsoft Research. 2010. Microsoft Learning to Rank Datasets. https://www. microsoft.com/en-us/research/project/mslr/. Accessed: 2026-05-19.

[27] Hussein Mozannar and David Sontag. 2020. Consistent estimators for learning to defer to an expert. In International conference on machine learning. PMLR, 7076–7087.

[28] Mahdi Pakdaman Naeini, Gregory Cooper, and Milos Hauskrecht. 2015. Obtaining well calibrated probabilities using bayesian binning. In Proceedings ofthe AAAI conference on artificial intelligence, Vol. 29.

[29] Alexandru Niculescu-Mizil and Rich Caruana. 2005. Predicting good probabilities with supervised learning. In Proceedings ofthe 22nd international conference on Machine learning. 625–632

[30] Rodrigo Nogueira and Kyunghyun Cho. 2019. Passage Re-ranking with BERT. arXiv preprint arXiv:1901.04085 (2019).

[31] John Platt et al. 1999. Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods. Advances in large margin classifiers 10, 3 (1999), 61–74.

[32] Amir Rahimi, Amirreza Shaban, Ching-An Cheng, Richard Hartley, and Byron Boots. 2020. Intra order-preserving functions for calibration of multi-class neural networks. Advances in neural information processing systems 33 (2020), 13456– 13467.

[33] Chandan K Reddy, Lluís Màrquez, Fran Valero, Nikhil Rao, Hugo Zaragoza, Sambaran Bandyopadhyay, Arnab Biswas, Anlu Xing, and Karthik Subbian. 2022. Shopping queries dataset: A large-scale ESCI benchmark for improving product search. arXiv preprint arXiv:2206.06588 (2022).

[34] Stephen Robertson and Hugo Zaragoza. 2009. The probabilistic relevance framework: BM25 and beyond. Vol. 4. Now Publishers Inc.

[35] Nandan Thakur, Nils Reimers, Andreas Rücklé, Abhishek Srivastava, and Iryna Gurevych. 2021. Beir: A heterogenous benchmark for zero-shot evaluation of information retrieval models. arXiv preprint arXiv:2104.08663 (2021).

[36] Juozas Vaicenavicius, David Widmann, Carl Andersson, Fredrik Lindsten, Jacob Roll, and Thomas Schön. 2019. Evaluating model calibration in classification. In The 22nd international conference on artificial intelligence and statistics. PMLR, 3459–3467.

[37] Bianca Zadrozny and Charles Elkan. 2002. Transforming classifier scores into accurate multiclass probability estimates. In Proceedings of the eighth ACM SIGKDD international conference on Knowledge discovery and data mining. 694–699.