# $T C P _ { \alpha }$ : Margin-Controlled Confidence estimation for reliable Music Information Retrieval

Parampreet Singh<sup>∗</sup>, Anushka Singh<sup>‡</sup>, Sumit Kumar<sup>†</sup>, Vipul Arora<sup>§</sup>

Indian Institute of Technology, Kanpur, India

Email: <sup>∗</sup>params21@iitk.ac.in, <sup>‡</sup>anushkas22@iitk.ac.in, <sup>†</sup>krsumit@iitk.ac.in, <sup>§</sup>vipular@iitk.ac.in

Abstract—Deep neural networks are often overconfident, assigning high confidence even to incorrect predictions. Consequently, users lack a reliable signal for deciding when a prediction can be trusted. Post-hoc confidence estimation addresses this by training a lightweight auxiliary head over a frozen classifier. Existing targets, however, suffer from inherent ambiguity: they assign overlapping confidence values to correct and incorrect predictions, while errors near the decision boundary receive confidence scores indistinguishable from correct predictions. In this work, we propose $T C { \bar { P } } _ { \alpha } ,$ , a novel confidence target that resolves these limitations by introducing a margin-controlled penalty for misclassified samples. We prove that TCP guarantees complete separation between the target values of correct and incorrect predictions, with a separation margin that is independent of the number of classes and increases monotonically with the penalty parameter. Since accurate classifiers naturally produce very few errors, learning these targets results in a severely imbalanced regression problem. We therefore present a systematic study of training strategies for learning under this imbalance and identify an effective training configuration through extensive ablation studies. We evaluate the proposed approach on raga¯ identification, investigate its robustness under domain shift, and further validate it on frame-wise ornamentation detection without modifying the selected configuration. Across all settings, $T C P _ { \alpha }$ consistently outperforms existing confidence targets for failure prediction. Rejecting only the least-confident 8% of predictions improves the base model’s macro-F1 from 0.89 to 0.98, while fine-tuning the confidence head with only 5% labeled samples from a new corpus effectively restores performance under domain shift. It provides an effective and transferable framework for reliable failure prediction in practical music information retrieval systems.

Index Terms—Confidence Estimation, Failure Prediction, Music Information Retrieval, Raga¯ Identification, Ornamentation detection

## I. INTRODUCTION

D <sup>EEP</sup> <sup>learning</sup> <sup>has</sup> <sup>significantly</sup> <sup>advanced</sup> <sup>Music</sup> <sup>Infor-</sup>mation Retrieval (MIR), enabling state-of-the-art perfor- mation Retrieval (MIR), enabling state-of-the-art performance across tasks such as automatic transcription [1], [2], source separation [3], [4], instrument recognition [5], genre classification [6], raga¯ identification [7], and rhythmic-pattern analysis [8], [9]. Despite these advances, most deep learning models provide only a class prediction without indicating whether that prediction should be trusted. This absence of reliable confidence estimates limits their usefulness in applications such as music pedagogy [10], where an automated critique delivered with unwarranted certainty can mislead learners, and equally in recommendation systems, where an unreliable prediction shown with high confidence erodes user trust. Labeled data is a standing bottleneck across MIR, and automatically labeling archives with tags such as raga¯ or genre could ease the scarcity and cost of expert annotation [11]. Such a system is useful, however, only if it can identify the predictions it is likely to get wrong and defer them for expert review. Confidence estimation [12]–[14] addresses this by appending a score to each prediction, flagging low-confidence cases for manual review, relabeling, or rejection.

A range of methods have been proposed to estimate the confidence of a model’s prediction. The simplest is Maximum Class Probability (MCP) [15], which uses the predicted softmax probability as the confidence score. However, deep neural networks are systematically overconfident [12]–[14], causing MCP to assign high confidence even to incorrect predictions. Sampling- and ensemble-based approaches [16]–[18] provide improved uncertainty estimates but require multiple forward passes or retraining. Post-hoc confidence estimation methods, such as ConfidNet [12], leave the classifier frozen and train a lightweight auxiliary confidence head. Their effectiveness largely depends on the confidence target used for supervision. The original ConfidNet [12] uses the True Class Probability (TCP), which reduces overconfidence by using the probability assigned to the ground-truth class. However, its target values for correct and incorrect predictions overlap, and this ambiguity increases with the number of classes. Its normalized variant, $T C P _ { n }$ [12], eliminates this overlap for correct predictions but assigns near-boundary errors confidence targets indistinguishable from correct predictions. These limitations arise from the design of the confidence target itself rather than from the post-hoc framework, motivating the search for improved target formulations. Additionally, learning such confidence targets results in a highly imbalanced regression problem. Since the targets are computed from the predictions of an accurate base classifier, most samples receive high confidence targets, while only a small fraction corresponds to errors. Learning from such a skewed target distribution is therefore challenging.

In this work, we propose a modified target $\mathrm { T C P } _ { \alpha } ,$ , which augments the $\mathrm { T C P _ { n } }$ denominator with a penalty term active only on misclassified samples. The penalty term creates separation between mis-classified and correctly classified samples with a margin that is independent of the number of classes and widens monotonically with α. The target imbalance, however, persists regardless of how the target is constructed, since it is inherited from the base classifier itself. We therefore study a range of training strategies to address it, and identify the configuration that works best through ablations.

We first validate the proposed method on raga identifica-¯ tion [7], where the success–error imbalance is particularly severe. We then evaluate whether the same configuration transfers to frame-wise ornamentation detection [19], a task with different temporal granularity and class distribution. Finally, we study domain shift using a second raga corpus [20]. While¯ our experiments focus on Indian Art Music, the proposed framework can be readily applied to other deep classification problems requiring reliable failure prediction.

The contributions of this work are summarized as follows:

• We propose $\mathrm { T C P } _ { \alpha } ,$ , a novel confidence target for failure prediction that achieves complete separation between correct and incorrect predictions, independent of the number of classes.

• We provide theoretical guarantees establishing complete target separation and a margin independent of the number of classes.

• We identify the configuration that best handles the training imbalance on raga identification through a systematic¯ study of training strategies, and show that rejecting a very small fraction of samples yields large performance gains.

• We validate on frame-wise ornamentation detection, a task differing in granularity and imbalance, and achieve similar gains by rejection.

• We examine transfer under domain shift, and find that fine-tuning the confidence head on a few labeled target-domain samples restores well-separated confidence scores.

This work substantially extends our ICASSP workshop publication [21] by (1) providing theoretical guarantees for the proposed $\mathrm { T C P } _ { \alpha }$ targets; (2) analyzing a range of training strategies for learning the highly imbalanced targets; (3) formulating the problem explicitly as failure prediction rather than confidence calibration; and (4) examining the effect of domain shift.

All the codes are available at: https://tinyurl.com/tcp-alpha

## II. RELATED WORKS

Confidence and uncertainty estimation for deep neural networks have been studied extensively, with existing methods broadly falling into two categories. The first estimates uncertainty through the training process itself using Bayesian deep learning [16], Monte Carlo dropout [17], or deep ensembles [18]. While these methods generally improve uncertainty estimation, they require specialized training procedures or multiple stochastic forward passes. The second category comprises post-hoc confidence estimation methods, which operate on an already-trained classifier without modifying its decision boundary. Temperature scaling [14] calibrates the softmax probabilities to better match empirical accuracy, while Maximum Class Probability (MCP) uses the predicted softmax probability as a confidence score [15]. Other training-free scores include predictive entropy and the Top-2 Margin, which respectively use the full predictive distribution and the gap between the top two class probabilities [22], as well as the energy score, which derives a confidence measure directly from the pre-softmax logits [23]. DOCTOR [24] similarly utilizes the full softmax vector for failure prediction without additional training. ConfidNet [12] instead trains a lightweight auxiliary confidence head over a frozen classifier using the True Class Probability (TCP) as the regression target, thereby reducing the overconfidence exhibited by MCP on incorrect predictions. Our work builds upon this post-hoc confidence estimation framework by proposing a new target formulation that addresses the limitations of existing TCP-based targets.

An important distinction in this area is between confidence calibration andfailure prediction [25]. Calibration seeks probabilistic correctness, ensuring that predicted confidence matches empirical accuracy, and is commonly evaluated using metrics such as the Expected Calibration Error (ECE) [14], [26]. However, recent studies have shown that ECE can be insensitive to important calibration failures and does not always reflect the practical usefulness of confidence estimates [27], [28]. More importantly, good calibration does not necessarily imply good failure prediction. The work [25] demonstrate that several methods which improve calibration error or out-of-distribution detection can simultaneously reduce the separability between correct and incorrect predictions, which is precisely the opposite of the objective in failure prediction. Our goal is therefore not to estimate calibrated probabilities, but to learn a confidence score that reliably distinguishes correct predictions from incorrect ones, enabling unreliable predictions to be rejected, deferred for human review, or prioritized for relabeling. Accordingly, we evaluate using failure prediction metrics such as AUPR-Error, AUROC, and FPR@95%TPR [12] rather than calibration metrics like ECE. Failure prediction is also closely related to selective prediction or confidence-based abstention, which studies using confidence or uncertainty to trade predictive risk against coverage [29]–[32]. Complementary to this line of work, we focus on learning a more reliable confidence score that can be used within such frameworks.

Learning confidence targets presents another challenge, as the confidence targets are highly imbalanced. Classical imbalance mitigation techniques such as class reweighting, oversampling, and focal loss [33]–[35] are primarily designed for discrete class labels. Our setting is therefore closer to Deep Imbalanced Regression [36], which introduces Label Distribution Smoothing (LDS) and Feature Distribution Smoothing (FDS) for skewed continuous targets. However, unlike conventional regression problems, the imbalance in our setting is not an intrinsic property of the dataset but is induced by the high accuracy of the underlying classifier itself. We therefore investigate training strategies specifically tailored to this setting.

Within Music Information Retrieval (MIR), uncertainty and confidence estimation have received comparatively little attention. Existing studies have explored uncertainty in music genre classification [37], [38], musical improvisation [39], music emotion recognition [40], and melody estimation [41], [42].

In Indian Art Music, previous work has primarily focused on improving predictive performance for tasks such as raga¯ identification [7], [43]–[45] and ornamentation detection [19], without estimating the reliability of model predictions. Our earlier work [21] took the first step toward closing this gap by introducing a modified TCP target for these two tasks. The present work substantially extends that direction through a more rigorous theoretical treatment, improved learning strategies, and broader experimental validation.

## III. PRELIMINARIES AND PROBLEM FORMULATION

## A. Setup

Consider a labeled dataset $\boldsymbol { \mathcal { D } } = \{ ( x _ { i } , y _ { i } ^ { * } ) \} _ { i = 1 } ^ { N }$ of N i.i.d. samples, where $x _ { i } \in \mathbb { R } ^ { d }$ denotes the input sample and $y _ { i } ^ { * } \in$ $\mathcal { V } = \{ 1 , \ldots , K \}$ is its ground-truth class. Let $f _ { w }$ denote a K-class classifier with parameters w that induces a predictive distribution $\mathbb { P } ( Y \mid w , x )$ through a softmax layer, and let

$$
{ \hat { y } } \ = \ \arg \operatorname* { m a x } _ { k \in \mathcal { V } } \mathbb { P } ( Y = k \mid w , x ) .\tag{1}
$$

Throughout this paper, we denote the probability assigned to the ground-truth class as $p ^ { * }$ and to the predicted class as $\hat { p } \colon$

$$
p ^ { * } \ = \ \mathbb { P } ( Y = y ^ { * } \mid w , x ) , \qquad \hat { p } \ = \ \mathbb { P } ( Y = \hat { y } \mid w , x ) ,\tag{2}
$$

respectively. We partition the dataset into the success set $\mathcal { D } ^ { + } =$ $\left\{ i : \hat { y } _ { i } = y _ { i } ^ { * } \right\} ( f _ { w }$ makes a correct prediction) and the error set $\mathcal { D } ^ { - } = \{ i : \hat { y } _ { i } \neq y _ { i } ^ { * } \} \ ( f _ { w }$ makes an incorrect prediction), so that $\mathcal { D } ^ { + } \cup \mathcal { D } ^ { - } = \{ 1 , \ldots , N \}$

Following the post-hoc confidence estimation framework of ConfidNet [12], the base classifier $f _ { w }$ is trained once and then frozen. As shown in Fig. 1, our objective is to attach to every prediction a scalar confidence score $\hat { c } ( x ; \theta ) \in [ 0 , 1 ]$ , produced by an auxiliary head with parameters θ operating on the frozen backbone’s penultimate representation $\boldsymbol { z } ( \boldsymbol { x } ) \in \mathbb { R } ^ { d _ { f } }$ . The head is trained by regression against a confidence target $c ^ { * } ( x , y ^ { * } )$

$$
\mathcal { L } _ { \mathrm { c o n f } } ( \theta ; \mathcal { D } ) = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \bigl ( \hat { c } ( x _ { i } ; \theta ) - c ^ { * } ( x _ { i } , y _ { i } ^ { * } ) \bigr ) ^ { 2 } .\tag{3}
$$

The confidence target is a function of the ground-truth label and is therefore available only at training time. The task is to reproduce it at inference from $z ( x )$ alone. Consequently, the effectiveness of this framework depends critically on the design of the target $c ^ { * } ( x , y ^ { * } )$ , which is the primary focus of this work.

## B. Existing Targets and Their Failure Modes

We briefly review the existing target formulations and highlight the limitations that motivate our proposed target.

a) Maximum Class Probability (MCP): The simplest confidence score is $\mathrm { M C P } ( x ) ~ = ~ \hat { p }$ [15], which is just the probability of the predicted class for the base classifier. It suffers from the problem of models being over-confident. Since $\hat { p } \ge 1 / K$ always, MCP takes values in $[ 1 / K , 1 ]$ for both successes and for errors, which results in an overlap in confidence scores for both (Fig. 2).

![](images/c869090f3a9f63f26eaa48cd0adc458d9ec420a4253522593d4a0c407516fbdd.jpg)  
Fig. 1: Overview of the post-hoc confidence estimation module. The backbone and classifier are frozen after base training; only the confidence head is trained, against the $\mathrm { T C P } _ { \alpha }$ target derived from the frozen classifier’s softmax output and the ground-truth label.

b) True Class Probability (TCP): ConfidNet [12] instead learns $\mathrm { T C P } ( x , y ^ { * } ) = p ^ { * }$ [12], which is the probability associated with the True class instead of the predicted class. Two guarantees follow immediately:

$$
\begin{array} { r l } & { 1 ) ~ \mathrm { i f } ~ p ^ { * } > \frac { 1 } { 2 } ~ \mathrm { t h e n } ~ \hat { y } = y ^ { * } ; } \\ & { 2 ) ~ \mathrm { i f } ~ p ^ { * } < \frac { 1 } { K } ~ \mathrm { t h e n } ~ \hat { y } \neq y ^ { * } . } \end{array}
$$

Neither is vacuous, but together they leave an ambiguity band: because an error forces $p ^ { * } < \hat { p }$ and hence $p ^ { * } < \frac { 1 } { 2 }$ , the error targets occupy $[ 0 , \textstyle { \frac { 1 } { 2 } } )$ while the success targets occupy $\textstyle { \bigl [ } { \frac { 1 } { K } } , 1 { \bigr ] }$ Any target value in $\textstyle { \big [ } { \frac { 1 } { K } } , { \frac { 1 } { 2 } } { \big ) }$ is therefore attainable by a correct and by an incorrect prediction alike, and no threshold on TCP can separate the two. The width of this band grows with $K .$

c) Normalised TCP $( T C P _ { n } ) .$ To eliminate the ambiguity on the success side, [12] normalises the TCP score by the predicted-class probability,

$$
\mathrm { T C P _ { n } } ( x , y ^ { * } ) = \frac { p ^ { * } } { \hat { p } } ,\tag{4}
$$

which pins every success to exactly 1. The authors claim stronger theoretical guarantees compared to TCP, but this creates an additional problem for the error targets. Consider an error with $p ^ { * } = m - \varepsilon$ and ${ \hat { p } } = m + \varepsilon$ for some m and $\varepsilon > 0 ,$ Then $\mathrm { T C P _ { n } } \to 1 \mathrm { a s } \varepsilon \to 0 ^ { + }$ , even though the prediction is wrong. So, near-boundary errors are thus assigned targets indistinguishable from confident successes.

![](images/8128bcfa86b9dea2a441ad4d855bc9a391d4cc3d0bbe04fff9e22e4a3e0ae335.jpg)  
Fig. 2: Comparison of the target distributions induced by MCP, $\mathrm { T C P } _ { n } .$ , and the proposed $\mathrm { T C P } _ { \alpha }$ for PIM [7] dataset. The xaxis represents the targets and y-axis represents the density. $\mathrm { T C P } _ { \alpha }$ produces complete separation between success and error targets.

## IV. PROPOSED METHOD

## A. The $T C P _ { \alpha }$ Confidence Target

The limitations of TCP and $\mathrm { T C P } _ { n }$ arise because the corresponding targets cannot completely separate correct and incorrect predictions. We therefore introduce $\mathrm { T C P } _ { \alpha } ,$ , which modifies $\mathrm { T C P } _ { n }$ through an additive penalty that is active only for misclassified samples. We define:

$$
\mathrm { T C P } _ { \alpha } ( x , \hat { y } , y ^ { * } ) = \frac { p ^ { * } } { \hat { p } + \mathbb { I } [ \hat { y } \neq y ^ { * } ] \left( p ^ { * } + \alpha \right) } ,\tag{5}
$$

where $\alpha \geq 0$ is a penalty hyperparameter. The proposed formulation preserves the desirable property that every correctly classified sample receives a target of one, while explicitly pushing all error targets away from the success region. When $\hat { y } \neq y ^ { \ast }$ the denominator is inflated by $p ^ { * } + \alpha$ , which strictly depresses the target below its $\mathrm { T C P _ { n } }$ value. Fig. 2 shows a comparison of the MCP, $\mathrm { T C P _ { n } }$ and the proposed $\mathrm { T C P } _ { \alpha }$ targets.

It is worth noting that the $p ^ { * }$ term inside the penalty is what removes the near-boundary pathology of $\mathrm { T C P _ { n } }$ even at $\alpha = 0$ The constant α then provides continuous, monotone control over how far below the success targets the error targets are pushed.

## B. Theoretical Guarantees

Let $T ^ { + } = \{ \mathrm { T C P } _ { \alpha } ( x , \hat { y } , y ^ { * } ) : \hat { y } = y ^ { * } \}$ and $\begin{array} { r l } { \mathcal { T } ^ { - } } & { { } = } \end{array}$ $\{ \mathrm { T C P } _ { \alpha } ( x , \hat { y } , y ^ { * } ) : \hat { y } \neq y ^ { * } \}$ denote the sets of achievable success and error targets respectively.

proposition 1 (Success targets are exactly one). For any x with $\hat { y } = y ^ { * }$ and any $\alpha \ge 0 , \ \mathrm { T C P } _ { \alpha } ( x , \hat { y } , y ^ { * } ) = 1$ . Hence ${ \mathcal { T } } ^ { + } = \{ 1 \}$

Proof. When $\hat { y } = y ^ { * }$ the indicator in (5) vanishes and $\hat { p } = p ^ { * }$ So, $\mathrm { T C P } _ { \alpha } = p ^ { * } / p ^ { * } = 1$

proposition 2 (Tight ceiling on error targets). For any x with $\hat { y } \neq y ^ { * }$ and any $\alpha \geq 0 _ { i }$

$$
\mathrm { T C P } _ { \alpha } ( x , \hat { y } , y ^ { * } ) \ < \ \frac { 1 } { 2 ( 1 + \alpha ) } .\tag{6}
$$

The bound is tight: sup $\begin{array} { r } { \mathcal { T } ^ { - } = \frac { 1 } { 2 ( 1 + \alpha ) } } \end{array}$ , approached in the limit $\begin{array} { r } { \hat { p }  p ^ { * } , p ^ { * }  \frac { 1 } { 2 } } \end{array}$

Proof. For an error, (5) gives $\mathrm { T C P } _ { \alpha } = p ^ { * } / ( \hat { p } + p ^ { * } + \alpha )$ Strictness of the argmax gives $\hat { p } > p ^ { * }$ , hence

$$
\hat { p } + p ^ { * } + \alpha ~ > ~ 2 p ^ { * } + \alpha , \qquad \mathrm { s o } \qquad \mathrm { T C P } _ { \alpha } ~ < ~ \frac { p ^ { * } } { 2 p ^ { * } + \alpha } .\tag{7}
$$

Now, since $\hat { p } > p ^ { * }$ and $\hat { p } + p ^ { * } \leq 1$ , we have $2 p ^ { * } < \hat { p } + p ^ { * } \leq 1$ and hence $\begin{array} { r } { p ^ { * } < \frac { 1 } { 2 } } \end{array}$ . Eq: (7) becomes:

$$
\begin{array} { r } { \mathrm { T C P } _ { \alpha } < \frac { 1 / 2 } { 1 + \alpha } = \frac { 1 } { 2 ( 1 + \alpha ) } } \end{array}
$$

For tightness, fix $\alpha \geq 0$ and take the sequence $\begin{array} { r } { p _ { n } ^ { * } = \frac { 1 } { 2 } - \frac { 1 } { n } . } \end{array}$ $\begin{array} { r } { \hat { p } _ { n } = \frac { 1 } { 2 } - \frac { 1 } { 2 n } } \end{array}$ , which satisfies $\hat { p } _ { n } > p _ { n } ^ { * }$ and $\hat { p } _ { n } + p _ { n } ^ { * } < 1$ for all $n \geq 3 .$ . Thus, each pair $( p _ { n } ^ { * } , \hat { p } _ { n } )$ corresponds to a valid error prediction. Then,

$$
\begin{array} { c } { T C P _ { \alpha n } = \frac { \frac { 1 } { 2 } - \frac { 1 } { n } } { ( \frac { 1 } { 2 } - \frac { 1 } { 2 n } ) + ( \frac { 1 } { 2 } - \frac { 1 } { n } ) + \alpha } } \\ { = \frac { \frac { 1 } { 2 } - \frac { 1 } { n } } { 1 + \alpha - \frac { 3 } { 2 n } } \xrightarrow [ ] { n  \infty } \frac { 1 } { 2 ( 1 + \alpha ) } . } \end{array}\tag{8}
$$

Thus, the target assigned to an incorrect prediction is strictly bounded by $\frac { 1 } { 2 ( 1 + \alpha ) }$ , and this bound is tight ■

theorem 1 (Complete target separation and separation margin). For every $\alpha \geq 0 , \mathcal { T } ^ { + } \cap \mathcal { T } ^ { - } = \varnothing$ , and the separation margin is

$$
\Delta ( \alpha ) \ \triangleq \ \operatorname* { i n f } _ { v \in { \mathcal { T } } ^ { + } } v \ - \ \operatorname* { s u p } _ { v \in { \mathcal { T } } ^ { - } } v \ = \ 1 - \frac { 1 } { 2 ( 1 + \alpha ) } .\tag{9}
$$

The margin is independent of K, satisfies $\Delta ( 0 ) = { \textstyle { \frac { 1 } { 2 } } }$ , is strictly increasing in α, and $\Delta ( \alpha )  1$ as $\alpha  \infty .$

Proof. By Proposition 1, inf $\tau ^ { + } ~ = ~ 1$ . By Proposition 2, sup $\begin{array} { r } { \dot { \mathcal { T } } ^ { - } \stackrel { \cdot } { = } \frac { 1 } { 2 ( 1 + \alpha ) } < 1 } \end{array}$ . Hence, ${ \cal T } ^ { + } \cap { \cal T } ^ { - } = \emptyset$ , and

$$
\Delta ( \alpha ) = 1 - \frac { 1 } { 2 ( 1 + \alpha ) } .
$$

Finally,

$$
\frac { \mathrm { d } \Delta } { \mathrm { d } \alpha } = \frac { 1 } { 2 ( 1 + \alpha ) ^ { 2 } } > 0 ,
$$

so the margin increases strictly with $\alpha ,$ with $\Delta ( 0 ) = { \textstyle { \frac { 1 } { 2 } } }$ and $\Delta ( \alpha )  1 \ \mathrm { a s } \ \alpha  \infty .$ ■

proposition 3 (Monotone penalty). For any fixed error sample with $p ^ { * } > 0 , \mathrm { T C P } _ { \alpha }$ is strictly decreasing in α:

$$
\frac { \partial \mathrm { T C P } _ { \alpha } } { \partial \alpha } = - \frac { p ^ { * } } { ( \hat { p } + p ^ { * } + \alpha ) ^ { 2 } } < 0 .\tag{10}
$$

For success samples $\partial \mathrm { T C P } _ { \alpha } / \partial \alpha = 0$

Proof. For errors, ${ \cal C } \triangleq \hat { p } + p ^ { * } > 0$ is constant in α and $\mathrm { T C P } _ { \alpha } ~ = ~ p ^ { * } / ( C + \alpha ) ;$ ; differentiating gives the stated derivative, which is negative since $p ^ { * } \ > \ 0 .$ . For successes Proposition 1 gives $\mathrm { T C P } _ { \alpha } \equiv 1$ , whose derivative is zero. ■

Proposition 3 identifies α as a monotone margin-control parameter: increasing it depresses every error target while leaving every success target pinned at 1, so the two groups are driven apart uniformly and predictably.

Remark 1 (Contrast with the baselines). For MCP the two target ranges coincide, for TCP the ranges overlap on $[ \textstyle { \frac { 1 } { K } } , \textstyle { \frac { 1 } { 2 } } )$ a band that widens with the number of classes, so the guarantee degrades in exactly the many-class regime where confidence estimation is the most needed. For $\mathrm { T C P _ { n } }$ the ranges touch $( \operatorname* { s u p } T ^ { - } = \operatorname* { i n f } T ^ { + } = 1 )$ and the margin is zero. Theorem 1 shows that $\mathrm { T C P } _ { \alpha }$ is the only member of the family with a strictly positive, K-independent margin, and it attains this already at $\alpha \ : = \ : 0$ . The role of $\alpha > 0$ is therefore not to create separation but to enlarge it. Table I summarises all these differences.

TABLE I: Theoretical properties of confidence targets. $\tau ^ { + } , \tau ^ { - }$ are the achievable target ranges for successes and errors; $K$ is the number of classes; $\alpha \geq 0$ is the penalty of $\mathrm { T C P } _ { \alpha } .$
<table><tr><td>Criterion</td><td> $\tau ^ { + }$ </td><td>T⁻</td><td>Overlap</td><td>Margin  $\Delta$ </td></tr><tr><td>MCP</td><td></td><td> $\textstyle { \bigl [ } { \frac { 1 } { K } } , 1 { \bigr ] }$ </td><td>always</td><td>0</td></tr><tr><td>TCP</td><td> $\begin{array} { r } { \left[ \frac { 1 } { K } , 1 \right] } \\ { \left[ \frac { 1 } { K } , 1 \right] } \end{array}$ </td><td> $[ \hat { 0 } , \hat { \textstyle { \frac { 1 } { 2 } } } )$ </td><td> $\textstyle { \bigl [ } { \frac { 1 } { K } } , { \frac { \bar { 1 } } { 2 } } { \bigr ) }$ </td><td>0</td></tr><tr><td> $\mathrm { T C P _ { n } }$ </td><td>{1}</td><td>(0, 1)</td><td>at boundary</td><td>0</td></tr><tr><td> $\mathbf { T C P } _ { \alpha }$  (ours)</td><td>{1}</td><td> $\begin{array} { r } { \left( 0 , \frac { 1 } { 2 ( 1 + \alpha ) } \right) } \end{array}$ </td><td>none</td><td> $\begin{array} { r } { 1 - \frac { 1 } { 2 ( 1 + \alpha ) } } \end{array}$ </td></tr></table>

## C. Learning the Confidence Head

Since the base classifier is highly accurate, the target distribution induced by TCP<sub>α</sub> is extremely skewed, with only a small fraction corresponding to errors. Although the target provides complete separation between correct and incorrect predictions, the resulting regression problem is dominated by the abundant high-confidence samples, making informative low-confidence predictions difficult to learn.

To address this challenge, we employ an imbalance-aware training strategy consisting of stratified mini-batch sampling and class-conditional loss weighting. We enforce a fixed errorto-success ratio in every mini-batch during training. Let $r =$ $| { \mathcal { D } } _ { - } ^ { \mathrm { b a t c h } } | / | { \mathcal { D } } _ { + } ^ { \mathrm { b a t c h } } |$ be the desired ratio. Since $| \mathcal { D } _ { - } | \ll | \mathcal { D } _ { + } |$ , error samples are exhausted faster than successes under random sampling, so we maintain a circular buffer over D<sub>−</sub>. With $\pi ^ { ( t ) }$ the t-th random permutation of $\mathcal { D } _ { - } .$ , the buffer for epoch e concatenates successive permutations,

$$
\mathcal { B } _ { - } ( e ) = \pi ^ { ( 1 ) } \parallel \pi ^ { ( 2 ) } \parallel \cdots ,\tag{11}
$$

drawing fresh permutations until every mini-batch in the epoch is filled. Each error sample thus recurs several times before all successes are seen once, without repeating within a single permutation. We combine this with class-conditional weighting, which weights each sample by the inverse frequency of its base class in the error pool.

## V. EXPERIMENTAL SETUP

## A. Datasets

1) Raga Identification:¯ The Prasar Bharati Indian Music (PIM) dataset [7] contains 191 hours of polyphonic Hindustani concert recordings annotated with raga and tonic labels.¯ Following [7], we use the 12 most frequent ragas, segmented¯ into non-overlapping 30-second audio chunks. To evaluate robustness under domain shift, we use the Saraga-Hindustani dataset [20], an independently collected IAM corpus with different recording conditions and provenance. It contains recordings from 11 of the selected ragas, with substantially¯ fewer samples per class than PIM.

2) Ornamentation Detection: The Raga Ornamentation De-¯ tection corpus [19] contains mono-channel expert Hindustani vocal recordings with frame-wise labels for seven ornament types recorded by 2 expert Hindustani Musicians. We use audio from Expert-2, segmented into 10-second clips with frame-wise ornamentation labels.

TABLE II: Dataset statistics and the induced imbalance seen by the confidence head. |D<sup>−</sup>| is the number of base-model errors available for training the confidence head. K is the number of classes. Base F1 shows performance of the base classifier on which the confidence head is being trained.
<table><tr><td>Dataset</td><td>Unit</td><td> $K$ </td><td> $\vert \mathcal { D } ^ { + } \vert : \vert \mathcal { D } ^ { \cdot }$  1</td><td>Base F1</td></tr><tr><td>PIM</td><td>30s chunk</td><td>12</td><td>≈8.1:1</td><td>0.89</td></tr><tr><td>ROD</td><td>frame</td><td>7</td><td>≈2.2:1</td><td>0.69</td></tr><tr><td>Saraga</td><td>30s chunk</td><td>11</td><td>≈3.2:1</td><td>0.76</td></tr></table>

## B. Base Models

For raga identification ¯ , we follow [7] and use tonicnormalized chromagram features as input to a CNN–LSTM classifier trained with the cross-entropy loss on the PIM dataset. For ornamentation detection, we use the Temporal Convolutional Network proposed in [19], which operates on chromagram features using periodic padding and dilated convolutions.

Both classifiers are trained to convergence and subsequently frozen. The proposed confidence head is then attached to the penultimate feature representation while leaving the backbone and the original classification layer unchanged. For the domain-shift experiments on Saraga dataset, we directly evaluate the confidence head using the classifier trained on PIM, followed by fine-tuning on a few labeled Saraga samples.

Table II summarizes the datasets, the performance of the corresponding base classifiers, and the resulting success-toerror ratios. The final column highlights the central challenge addressed in this work: a classifier with a macro-F1 of approximately 0.89 produces only about one error for every eight correct predictions, resulting in a highly imbalanced confidence learning problem.

Unless otherwise stated, all ablation studies are performed on the PIM dataset. For the Saraga and ornamentation datasets, we report only the primary baselines and the final selected training strategy.

## C. Baselines

We compare the proposed method against two categories of confidence estimators. The first comprises training-free scores computed directly from the frozen classifier, while the second consists of learned confidence heads that differ only in the target used for supervision.

a) MCP: MCP, described in Section III-B, serves as the simplest confidence baseline.

b) Predictive Entropy: Predictive entropy [22] measures the uncertainty of the complete predictive distribution,

$$
\mathrm { P E } ( { \boldsymbol { x } } ) = - \sum _ { k = 1 } ^ { K } \mathbb { P } ( { \boldsymbol { Y } } = k \mid w , { \boldsymbol { x } } ) \log \mathbb { P } ( { \boldsymbol { Y } } = k \mid w , { \boldsymbol { x } } ) .\tag{12}
$$

Lower entropy indicates higher confidence; we therefore use $- \mathrm { P E } ( x )$ as the confidence score. Unlike MCP, it accounts for the entire predictive distribution rather than only the predicted class.

c) Top-2 Margin: The Top-2 Margin [22] measures the separation between the predicted class and its closest competitor,

$$
\operatorname { M a r g i n } ( x ) = { \hat { p } } - \operatorname* { m a x } _ { k \neq { \hat { y } } } \mathbb { P } ( Y = k \mid w , x ) .\tag{13}
$$

Larger margins correspond to higher confidence and indicate predictions that lie farther from the decision boundary.

d) Energy Score: The energy score [23] is computed directly from the pre-softmax logits,

$$
\mathrm { E } ( x ) ~ = ~ - T \log \sum _ { k = 1 } ^ { K } \exp \bigl ( g _ { k } ( x ) / T \bigr ) ,\tag{14}
$$

where $T = 1$ in our experiments. Higher confidence corresponds to lower energy, so we use $- \mathrm { E } ( x )$ as the confidence score.

e) MC Dropout: It approximates Bayesian model averaging by performing $T _ { \mathrm { m c } }$ stochastic forward passes with dropout enabled inference [17], yielding sampled weights $w _ { t }$ and the averaged predictive distribution

$$
\bar { p } _ { k } ( x ) \ = \ \frac { 1 } { T _ { \mathrm { m c } } } \sum _ { t = 1 } ^ { T _ { \mathrm { m c } } } \mathbb { P } ( Y = k \mid w _ { t } , x ) ,\tag{15}
$$

where $T _ { \mathrm { m c } } = 5 0$ in our experiments. The confidence score is taken as the negative predictive entropy of the averaged predictive distribution.

f) TCP and $\operatorname { T C P } _ { \mathrm { n } } .$ Here we train an auxiliary confidence head, implemented as a two-layer MLP with a sigmoid output using the target formulations described in Section III-B. To ensure a fair comparison, both are trained using the same confidence head architecture and optimization strategy as the proposed $\mathrm { T C P } _ { \alpha }$ model.

## D. Ablation Study Design

In addition to the proposed imbalance-aware training strategy described in Section IV-C, we evaluate several alternative approaches for learning the confidence head. under the highly imbalanced $\mathrm { T C P } _ { \alpha }$ target distribution.

1) Regression with Imbalance Correction: As a baseline, we first train the confidence head using the mean squared error loss of Eq. (3) without any imbalance handling. We then compare it with several regression strategies designed to alleviate target imbalance. Label Distribution Smoothing (LDS) [36] smooths the empirical target distribution and reweights samples according to the smoothed density, while its feature-space counterpart, Feature Distribution Smoothing (FDS) [36], performs smoothing in the learned feature representation. We also evaluate simple error upsampling, which creates duplicate copies of error samples during training.

2) Binary Discrimination: Instead of regressing continuous confidence targets, we also investigate treating confidence estimation as a binary discrimination problem by assigning each sample a success/error label $\ell _ { i }$ as:

$$
\ell _ { i } = \mathbb { I } \big [ i \in \mathcal { D } ^ { + } \big ] = \left\{ \begin{array} { l l } { 1 , } & { \hat { y } _ { i } = y _ { i } ^ { * } , } \\ { 0 , } & { \hat { y } _ { i } \ne y _ { i } ^ { * } , } \end{array} \right.\tag{16}
$$

TABLE III: Performance on raga identification (PIM). All¯ values are percentages, best in bold. (R) denotes the target trained with vanilla regression, (P) denotes training with our proposed training strategy (Sec: IV-C). T2M: Top-2-Margin score, MC-D: MC Dropout
<table><tr><td>Method</td><td>FPR@95↓</td><td>AUPR-E ↑</td><td>AUPR-S ↑</td><td>AUROC ↑</td></tr><tr><td>MCP</td><td>42.25</td><td>50.59</td><td>96.94</td><td>83.60</td></tr><tr><td>Entropy</td><td>41.00</td><td>53.70</td><td>97.10</td><td>84.49</td></tr><tr><td>T2M</td><td>43.00</td><td>44.25</td><td>96.70</td><td>82.53</td></tr><tr><td>Energy</td><td>37.50</td><td>56.87</td><td>97.47</td><td>86.25</td></tr><tr><td>MC-D</td><td>40.75</td><td>52.37</td><td>97.13</td><td>84.70</td></tr><tr><td>TCP (R)</td><td>46.02</td><td>25.35</td><td>98.85</td><td>85.11</td></tr><tr><td>TCPn (R)</td><td>61.74</td><td>32.64</td><td>98.42</td><td>81.74</td></tr><tr><td> $\mathrm { T C P } _ { \alpha }$  (R)</td><td>53.68</td><td>74.30</td><td>98.89</td><td>91.26</td></tr><tr><td>TCP (P)</td><td>12.24</td><td>83.13</td><td>99.76</td><td>97.37</td></tr><tr><td> $\mathrm { T C P } _ { n } ^ { \phantom { \dagger } } \left( \mathbf { \vec { P } } \right)$ </td><td>41.24</td><td>76.01</td><td>99.20</td><td>92.74</td></tr><tr><td>TCPα (P)</td><td>1.60</td><td>95.96</td><td>99.95</td><td>99.46</td></tr></table>

and training the confidence head using binary cross-entropy. We additionally evaluate focal loss [33], which emphasizes difficult minority samples by down-weighting well-classified examples. During inference, we take its predicted scalar output directly as the confidence score.

3) Per-Class Models: Finally, we investigate whether replacing a single global confidence head with class-specific models improves performance. Two binary discrimination variants are considered. The first one is trained using only the successes and errors of the corresponding class given by:

$$
\mathcal { D } _ { \mathrm { c l s } } ^ { ( k ) } = \underbrace { \{ ( x _ { i } , 1 ) : y _ { i } ^ { * } = k , \ : \hat { y } _ { i } = k \} } _ { \mathrm { c l a s s - } k \mathrm { ~ s u c c e s s e s } } \cup \underbrace { \{ ( x _ { i } , 0 ) : y _ { i } ^ { * } = k , \ : \hat { y } _ { i } \neq k \} } _ { \mathrm { c l a s s - } k \mathrm { ~ e r r o r s } } .\tag{17}
$$

Hence, we train k separate models. In another configuration, we also incorporate confusing negatives, i.e., samples from other classes incorrectly predicted as the target class by the base classifier, denoted by $\mathcal { H } _ { k }$ :

$$
\mathcal { H } _ { k } = \{ { \left( x _ { i } , 0 \right) } : i \in \mathcal { D } ^ { - } , \ \hat { y } _ { i } = k \} .\tag{18}
$$

So here, we work with: ${ \mathcal { D } } ^ { ( k ) } = { \mathcal { D } } _ { \mathrm { c l s } } ^ { ( k ) } \cup { \mathcal { H } } _ { k }$ , giving higher weight to the confusing $\mathcal { H } _ { k }$ in the regression loss. Additionally, we evaluate per-class regression models trained on $\mathcal { D } ^ { ( k ) }$ using the proposed $\mathrm { T C P } _ { \alpha }$ targets in place of binary targets $\ell _ { i } .$ . During inference, each sample is routed to the confidence head corresponding to its predicted class.

## E. Evaluation Metrics

We evaluate using the standard failure prediction metrics: AUPR-Error (AUPR-E), FPR@95%TPR, AUROC, and AUPR-Success (AUPR-S). Given the severe class imbalance, AUPR-E and FPR@95%TPR are the primary metrics because they evaluate performance on the rare error class. AUROC measures overall separability, while AUPR-S is reported only for completeness.

## VI. RESULTS AND DISCUSSION

## A. Comparison with Existing Baselines

Table III compares the proposed method against both training-free confidence scores and learned confidence estimators on the PIM dataset. Among the training-free methods, all confidence scores perform similarly, which is expected since MCP, predictive entropy, Top-2 Margin, and the energy score are deterministic functions of the same classifier output, while MC Dropout estimates uncertainty from multiple stochastic forward passes of the same model. Consequently, they induce similar rankings over correct and incorrect predictions. The energy score is the strongest training-free baseline, achieving an AUPR-E of 56.87 and an AUROC of 86.25, whereas MC Dropout offers no noticeable improvement despite requiring 50 forward passes.

Among the learned confidence models, $\mathrm { T C P } _ { \alpha }$ consistently outperforms both TCP and $\mathrm { T C P } _ { n }$ under identical training settings. Even with vanilla regression, $\mathrm { T C P } _ { \alpha }$ substantially improves failure prediction, confirming that the proposed target formulation itself is more informative. Applying the proposed imbalance-aware training strategy further improves all three learned targets, with $\mathrm { T C P } _ { \alpha }$ achieving the best overall performance $( \mathrm { A U P R - E } ~ = ~ 9 5 . 9 6 ~ $ $\mathrm { { A U R O C } } ~ = ~ 9 9 . 4 6$ and FPR $\textcircled { \omega } 9 5 \% \mathrm { T P R } = 1 . 6 0 )$ $\mathrm { T C P } _ { n }$ consistently performs the worst among the learned targets because its formulation assigns high target values to many near-boundary errors, making them difficult for the confidence head to distinguish from correct predictions. The rejection curves in Fig. 4 demonstrate the practical impact of this improvement: rejecting only the leastconfident $8 \%$ of predictions increases the retained macro-F1 from 0.89 to approximately 0.98. The confidence density plots (Fig. 3) further show that $\mathrm { T C P } _ { \alpha }$ produces substantially cleaner separation between correct and incorrect predictions than the existing targets.

![](images/83979de1bb67506870e1a0d1cf7fc07352b9055343f7b87054c797cc415209f4.jpg)  
Fig. 3: Distribution of predicted confidence scores for $\mathrm { T C P } _ { \alpha }$ on the PIM test set. The proposed target produces clear separation between correct and incorrect predictions.

## B. Ablation Studies

1) Training Strategy: Table IV compares different strategies for learning the $\mathrm { T C P } _ { \alpha }$ confidence head under severe target imbalance. All methods use the same confidence head and $\mathrm { T C P } _ { \alpha }$ target, differing only in the optimization strategy.

![](images/aa9d508c435ee46f555495b3763d9ea51bc717f4b76634870ab200a55fa42edd.jpg)  
Fig. 4: Macro-F1 of the retained predictions as increasingly low-confidence samples are rejected using confidence scores from baselines and proposed method.

Among the regression-based methods, LDS substantially improves over vanilla regression, but the confidence distributions remain considerably more diffuse than those produced by the proposed method. Instead of concentrating the correctly classified samples near confidence one, LDS spreads them over a broad high-confidence region while simultaneously shifting many error samples toward lower confidence. Consequently, the overlap between the two populations is reduced but not eliminated, with a noticeable number of successes still assigned relatively low confidence. The proposed training strategy, in contrast, produces a much sharper separation between the two distributions, resulting in substantially better failure prediction performance. FDS provides no benefit and error upsampling performs poorly, likely because repeatedly sampling the same small set of error examples increases overfitting without improving diversity. Treating confidence estimation as a binary discrimination problem using binary cross-entropy or focal loss is consistently inferior to confidence regression, indicating that the continuous $\mathrm { T C P } _ { \alpha }$ target contains richer supervision than hard success-error labels.

Treating confidence estimation as a binary discrimination problem using binary cross-entropy or focal loss is consistently inferior to confidence regression, indicating that the continuous $\mathrm { T C P } _ { \alpha }$ target contains richer supervision than hard successerror labels. Since binary models learn only two target values, even a few errors predicted as successes or a few successes assigned low confidence significantly degrade failure prediction metrics, as observed in their density plots. Among the per-class models, incorporating the confusing negatives $\mathcal { H } _ { k }$ improves performance over the basic per-class discriminator, confirming that hard negatives are beneficial. However, neither the per-class classifiers nor the per-class regressors outperform a single global confidence model. For the regressors in particular, the small number of error samples available within each class provides insufficient supervision to learn reliable class-specific confidence functions. Such approaches may become more effective on datasets with substantially larger numbers of error samples per class. The proposed imbalance-aware training strategy achieves the best performance by a large margin, improving AUPR-E from 74.30 to 95.96 while reducing FPR@95%TPR from 53.68 to only 1.60. The corresponding confidence distributions (Fig. 3) show a clear separation between correct and incorrect predictions, and we use this configuration for all subsequent experiments.

TABLE IV: Ablation studies for training $\mathrm { T C P } _ { \alpha }$ target on PIM. All rows use the same target and the same confidence head, and differ only in how the imbalance is handled. EU: Error Upsampling, $\mathrm { T C P } _ { \alpha } – \mathrm { P } ;$ proposed (Sec IV-C)
<table><tr><td>Strategy</td><td>FPR@95↓</td><td>AUPR-E ↑</td><td>AUPR-S ↑</td><td>AUROC ↑</td></tr><tr><td>reg</td><td>53.68</td><td>74.30</td><td>98.89</td><td>91.26</td></tr><tr><td>FDS</td><td>54.71</td><td>65.92</td><td>98.89</td><td>90.07</td></tr><tr><td>EU</td><td>66.14</td><td>29.91</td><td>98.34</td><td>81.67</td></tr><tr><td>LDS</td><td>21.98</td><td>85.70</td><td>98.94</td><td>94.42</td></tr><tr><td> $\mathrm { T C P } _ { \alpha } \ ( \mathrm { P } )$ </td><td>1.60</td><td>95.96</td><td>99.95</td><td>99.46</td></tr><tr><td>Binary</td><td>76.08</td><td>25.92</td><td>97.13</td><td>76.75</td></tr><tr><td>Focal</td><td>66.67</td><td>24.28</td><td>97.62</td><td>78.52</td></tr><tr><td>P-C</td><td>68.79</td><td>50.61</td><td>97.95</td><td>82.10</td></tr><tr><td>P-C-W</td><td>63.65</td><td>45.92</td><td>98.52</td><td>86.17</td></tr><tr><td>K-reg</td><td>77.26</td><td>29.52</td><td>96.99</td><td>79.14</td></tr></table>

TABLE V: Effect of the success-to-error ratio $r$ in each training batch on PIM. Batch size is fixed at 64.
<table><tr><td>r</td><td>FPR@95↓</td><td>AUPR-E ↑</td><td>AUROC ↑</td></tr><tr><td>5.4</td><td>39.36</td><td>91.09</td><td>95.08</td></tr><tr><td>3.3</td><td>10.62</td><td>97.41</td><td>97.50</td></tr><tr><td>2.2</td><td>1.60</td><td>95.96</td><td>99.46</td></tr><tr><td>1.6</td><td>7.53</td><td>89.21</td><td>98.14</td></tr><tr><td>1.1</td><td>7.63</td><td>82.19</td><td>98.25</td></tr><tr><td>0.6</td><td>10.17</td><td>73.69</td><td>97.31</td></tr><tr><td>0.3</td><td>13.28</td><td>52.81</td><td>95.19</td></tr></table>

2) Effect of the Success:Error Batch Ratio: Table V studies the influence of the success-to-error ratio r within each training mini-batch while keeping all other settings fixed. Very small numbers of error samples bias learning toward the dominant success class, leading to high FPR@95%TPR despite good average ranking. Conversely, increasing the proportion of error samples gradually degrades AUPR-E. For a ratio of 0.3:1, we observe that the error samples are concentrated below a confidence value of 0.5, but the confidence assigned to correctly classified samples becomes increasingly dispersed in the whole range of (0,1) instead of remaining concentrated near one. As a result, the overlap between the two populations increases, making them less distinguishable. A ratio of approximately 2.2:1 (20 error samples in a batch of 64) provides the best trade-off and is therefore used in all subsequent experiments.

3) Effect of the Penalty Parameter α: Table VI evaluates the influence of the penalty parameter α. Although Theorem 1 guarantees complete target separation even for $\alpha = 0$ introducing a positive penalty provides an additional margin between the success and error targets, making the regression problem easier to learn. Accordingly, all positive values of α outperform $\alpha = 0 ,$ , with the best performance obtained at $\alpha = 1 / K$ . Beyond this point, the performance varies only slightly. We use $\alpha = 1 / K$ throughout the experiments.

TABLE VI: Effect of the penalty α on $T C P _ { \alpha }$ performed on PIM dataset.
<table><tr><td>α</td><td>FPR@95↓</td><td>AUPR-E ↑</td><td>AUROC ↑</td></tr><tr><td>0</td><td>11.02</td><td>88.10</td><td>97.42</td></tr><tr><td> $1 / 2 K$ </td><td>3.48</td><td>91.15</td><td>97.29</td></tr><tr><td> $1 / K$ </td><td>1.60</td><td>95.96</td><td>99.46</td></tr><tr><td> $2 / K$ </td><td>6.31</td><td>90.20</td><td>96.90</td></tr><tr><td> $0 . 5$ </td><td>8.10</td><td>87.49</td><td>96.77</td></tr><tr><td>1</td><td>13.09</td><td>89.33</td><td>98.02</td></tr><tr><td>5</td><td>8.38</td><td>90.40</td><td>96.71</td></tr></table>

TABLE VII: Performance on Saraga dataset. The base classifier is the PIM model in all rows and is never retrained. reg represents vanilla regression training. $\mathrm { T C P } _ { \alpha } ( P _ { s } ) { : }$ best model trained on PIM fine-tuned on Saraga using s% labeled training samples, test set being fixed for all.
<table><tr><td>Model</td><td>FPR@95↓</td><td>AUPR-E ↑</td><td>AUPR-S ↑</td><td>AUROC ↑</td></tr><tr><td>MCP</td><td>73.38</td><td>60.39</td><td>89.85</td><td>80.61</td></tr><tr><td>reg</td><td>52.83</td><td>80.14</td><td>92.05</td><td>88.56</td></tr><tr><td> $\mathrm { T C P } _ { \alpha }$  (P0)</td><td>87.5</td><td>40.16</td><td>78.15</td><td>62.61</td></tr><tr><td>TCPα (P5)</td><td>5.0</td><td>97.90</td><td>99.50</td><td>98.21</td></tr><tr><td>TCPα  $\left( P _ { 1 0 } \right)$ </td><td>3.12</td><td>97.28</td><td>99.46</td><td>98.78</td></tr><tr><td> $\mathrm { T C P } _ { \alpha } \ ( P _ { 2 0 } )$ </td><td>2.50</td><td>97.92</td><td>99.37</td><td>98.60</td></tr><tr><td> $\mathrm { T C P } _ { \alpha } \ ( P _ { 4 0 } )$ </td><td>2.25</td><td>98.77</td><td>99.70</td><td>99.36</td></tr></table>

## C. Transfer to Domain Shift

Table VII evaluates the proposed confidence model under domain shift using the Saraga dataset. The base raga classifier¯ trained on PIM is kept fixed throughout, and only the confidence head is adapted. Directly applying the confidence head learned on PIM to Saraga $( \mathrm { T C P } _ { \alpha } ( P _ { 0 } ) )$ performs poorly, even below MCP, indicating that the learned confidence distribution is dataset-specific and does not transfer reliably across recording conditions.

Encouragingly, fine-tuning the confidence head using only 5% labeled samples from the target domain restores almost all of the performance, improving AUPR-E from 40.16 to 97.90 while reducing FPR@95%TPR from 87.5 to 5.0. Increasing the adaptation set beyond 5% yields only marginal improvements, suggesting that a small amount of labeled target-domain data is sufficient to adapt the confidence model to the new domain.

## D. Ornamentation Detection

Table VIII evaluates the proposed method on frame-wise ornamentation detection using the same configuration selected for raga identification.¯ $\mathrm { T C P } _ { \alpha }$ again achieves the best failure prediction performance across all metrics, improving AUPR-E to 86.01 while reducing FPR@95%TPR to 19.13. Similar to the raga identification task, the binary discrimination models¯ improve upon MCP but remain inferior to confidence regression, confirming that the continuous $\mathrm { T C P } _ { \alpha }$ target provides richer supervision than hard success-error labels.

TABLE VIII: Failure prediction on ornamentation detection. The configuration is the one selected on raga identification;¯ no additional search was performed.
<table><tr><td>Method</td><td>FPR@95↓</td><td>AUPR-E ↑</td><td>AUPR-S ↑</td><td>AUROC ↑</td></tr><tr><td>MCP</td><td>68.66</td><td>43.34</td><td>91.98</td><td>77.14</td></tr><tr><td>TCP</td><td>33.46</td><td>76.62</td><td>97.60</td><td>92.31</td></tr><tr><td> $\mathrm { T C P _ { n } }$ </td><td>47.88</td><td>71.21</td><td>96.77</td><td>89.22</td></tr><tr><td> $\mathrm { T C P } _ { \alpha }$ </td><td>19.13</td><td>86.01</td><td>98.70</td><td>95.80</td></tr><tr><td>Binary</td><td>35.30</td><td>81.58</td><td>97.74</td><td>92.90</td></tr><tr><td>Focal</td><td>35.10</td><td>81.41</td><td>97.91</td><td>92.96</td></tr></table>

![](images/f0f32d9e185a311f9e295042aa5c2564bf57e524b5299676cbf2178f01e7c4da.jpg)  
Fig. 5: Rejection Curves for Ornamentation detection task

The rejection curves in Fig. 5 further demonstrate the practical benefit of the proposed confidence estimator. Rejecting approximately 20% of the least-confident frames increases the macro-F1 from 0.69 to approximately 0.95, after which the curve saturates, indicating that most erroneous predictions have already been removed. In contrast, TCP and the binary models peak earlier and then decline, reflecting their weaker separation between correct and incorrect predictions.

![](images/9608baa01ad75e111422663b807d086a1f361834d144c31a835352154cdd80d7.jpg)  
Fig. 6: Density plot for the best performing model for Ornamentation detection

## VII. CONCLUSION AND FUTURE WORK

In this work, we propose $\mathrm { T C P } _ { \alpha }$ , a novel confidence target for post-hoc failure prediction that provides complete separation between correct and incorrect predictions with a margin independent of the number of classes. We study several training strategies for the resulting imbalanced regression problem and identify an imbalance-aware configuration that transfers across tasks. Experiments on raga identification and orna-¯ mentation detection demonstrate that the proposed approach consistently outperforms existing confidence targets and commonly used confidence scores. Rejecting predictions using the proposed confidence scores yields substantial improvements in the retained predictions, showing that a small amount of expert review directed by $\mathrm { T C P } _ { \alpha }$ is sufficient to make the remaining predictions highly reliable. We also observe that only a small amount of labeled target-domain data is sufficient to adapt the confidence head to a new domain.

A promising direction for future work is to develop classconditional confidence estimation that not only identifies unreliable predictions but also indicates plausible alternative classes. This is particularly useful for music analysis, where confusing two closely related ragas is fundamentally different¯ from confusing unrelated ones. Such information can provide more meaningful feedback in downstream applications such as music analysis and pedagogy.

## REFERENCES

[1] A. Saha, H.-U. Berendes, M. Muller, and B. Maman, “Snapping matters:¨ Context-aware onset refinement for automatic music transcription,” arXiv preprint arXiv:2606.11903, 2026.

[2] T. Kwon, D. Jeong, and J. Nam, “Towards efficient and real-time piano transcription using neural autoregressive models,” IEEE/ACM Transactions on Audio, Speech, and Language Processing (TASLP), vol. 32, pp. 5106–5116, 2024.

[3] B. Torres, G. Peeters, and G. Richard, “The inverse drum machine: Source separation through joint transcription and analysis-by-synthesis,” IEEE TASLP, vol. 34, pp. 84–95, 2026.

[4] S. Araki, N. Ito, R. Haeb-Umbach, G. Wichern, Z.-Q. Wang, and Y. Mitsufuji, “30+ years of source separation research: Achievements and future challenges,” in IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP), 2025, pp. 1–5.

[5] S. Xu, Y. Wang, Z. Li, F. Yu, and W. Li, “Leveraging timbre semantic descriptors for cross-dataset instrument recognition,” IEEE TASLP, vol. 33, pp. 1196–1207, 2025.

[6] Y. Zhang and T. Li, “Music genre classification with parallel convolutional neural networks and capuchin search algorithm,” Scientific reports, vol. 15, no. 1, p. 9580, 2025.

[7] P. Singh and V. Arora, “Explainable deep learning analysis for raga identification in indian art music,” IEEE TASLP, vol. 33, pp. 2302–2311, 2025.

[8] C.-W. Wu, C. Dittmar, C. Southall, R. Vogl, G. Widmer, J. Hockman, M. Muller, and A. Lerch, “A review of automatic drum transcription,”¨ IEEE TASLP, vol. 26, no. 9, pp. 1457–1483, 2018.

[9] R. B. Kodag and V. Arora, “Meta-learning-based percussion transcription and tala identification from low-resource audio,” IEEE TASLP, vol. 33, pp. 2749–2758, 2025.

[10] S. Kumar, S. Jaiswal, P. Singh, and V. Arora, “Automatic detection and analysis of singing mistakes for music learning by imitation,” IEEE TASLP, 2026.

[11] Q. Kong, C. Yu, Y. Xu, T. Iqbal, W. Wang, and M. D. Plumbley, “Weakly labelled audioset tagging with attention neural networks,” IEEE TASLP, vol. 27, no. 11, pp. 1791–1802, 2019.

[12] C. Corbiere, N. Thome, A. Bar-Hen, M. Cord, and P. P\` erez, “Addressing´ failure prediction by learning model confidence,” Advances in Neural Information Processing Systems (NeurIPS), vol. 32, 2019.

[13] C. Corbiere, N. Thome, A. Saporta, T.-H. Vu, M. Cord, and P. Perez, “Confidence estimation via auxiliary models,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 44, no. 10, pp. 6043– 6055, 2021.

[14] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, “On calibration of modern neural networks,” in International Conference on Machine Learning (ICML), 2017, pp. 1321–1330.

[15] D. Hendrycks and K. Gimpel, “A baseline for detecting misclassified and out-of-distribution examples in neural networks,” in International Conference on Learning Representations (ICLR), 2017.

[16] C. Blundell, J. Cornebise, K. Kavukcuoglu, and D. Wierstra, “Weight uncertainty in neural networks,” in ICML, 2015, pp. 1613–1622.

[17] Y. Gal and Z. Ghahramani, “Dropout as a bayesian approximation: Representing model uncertainty in deep learning,” in ICML, 2016, pp. 1050–1059.

[18] B. Lakshminarayanan, A. Pritzel, and C. Blundell, “Simple and scalable predictive uncertainty estimation using deep ensembles,” in NeurIPS, vol. 30, 2017.

[19] S. Kumar, P. Singh, and V. Arora, “Recognizing ornaments in vocal indian art music with active annotation,” IEEE TASLP, vol. 34, pp. 220– 229, 2026.

[20] A. Srinivasamurthy, S. Gulati, R. Caro Repetto, and X. Serra, “Saraga: Open datasets for research on indian art music,” Empirical Musicology Review, vol. 16, no. 1, p. 85–98, Dec. 2021.

[21] S. Kumar, P. Singh, and V. Arora, “Confidence-enhanced models for indian art music analysis,” in IEEE International Conference on Acoustics, Speech, and Signal Processing Workshops (ICASSPW), 2025, pp. 1–5.

[22] V.-L. Nguyen, M. H. Shaker, and E. Hullermeier, “How to measure un-¨ certainty in uncertainty sampling for active learning,” Machine Learning, vol. 111, no. 1, pp. 89–122, 2022.

[23] W. Liu, X. Wang, J. Owens, and Y. Li, “Energy-based out-of-distribution detection,” NeurIPS, vol. 33, pp. 21 464–21 475, 2020.

[24] F. Granese, M. Romanelli, D. Gorla, C. Palamidessi, and P. Piantanida, “DOCTOR: A Simple Method for Detecting Misclassification Errors,” in NeurIPS, 2021.

[25] F. Zhu, Z. Cheng, X.-Y. Zhang, and C.-L. Liu, “Rethinking confidence calibration for failure prediction,” in European Conference on Computer Vision (ECCV). Springer, 2022, pp. 518–536.

[26] M. P. Naeini, G. F. Cooper, and M. Hauskrecht, “Obtaining well calibrated probabilities using bayesian binning,” in AAAI Conference on Artificial Intelligence, 2015.

[27] M. Chidambaram, H. Lee, C. McSwiggen, and S. Rezchikov, “How flawed is ECE? an analysis via logit smoothing,” in ICML, 2024.

[28] J. Nixon, M. W. Dusenberry, L. Zhang, G. Jerfel, and D. Tran, “Measuring calibration in deep learning,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops, June 2019.

[29] Y. Geifman and R. El-Yaniv, “Selective classification for deep neural networks,” in NeurIPS, vol. 30, 2017.

[30] R. El-Yaniv and Y. Wiener, “On the foundations of noise-free selective classification,” Journal of Machine Learning Research, vol. 11, no. 53, pp. 1605–1641, 2010.

[31] V. Franc, D. Prusa, and V. Voracek, “Optimal strategies for reject option classifiers,” Journal of Machine Learning Research, vol. 24, no. 11, pp. 1–49, 2023.

[32] Y. Geifman and R. El-Yaniv, “Selectivenet: A deep neural network with an integrated reject option,” in ICML. PMLR, 2019, pp. 2151–2159.

[33] T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollar, “Focal loss for´ dense object detection,” in IEEE International Conference on Computer Vision (ICCV), 2017, pp. 2980–2988.

[34] Y. Cui, M. Jia, T.-Y. Lin, Y. Song, and S. Belongie, “Class-balanced loss based on effective number of samples,” in IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2019, pp. 9268– 9277.

[35] N. V. Chawla, K. W. Bowyer, L. O. Hall, and W. P. Kegelmeyer, “Smote: Synthetic minority over-sampling technique,” Journal of Artificial Intelligence Research, vol. 16, pp. 321–357, 2002.

[36] Y. Yang, K. Zha, Y.-C. Chen, H. Wang, and D. Katabi, “Delving into deep imbalanced regression,” in ICML, 2021, pp. 11 842–11 851.

[37] H. Lukashevich, S. Grollmisch, and J. Abeßer, “Quantifying uncertainty in music genre classification,” in Proceedings of The 49th Annual Conference on Acoustics DAGA. Hamburg, Germany, 2023, pp. 1378– 1381.

[38] T. Ye, S. Si, J. Wang, N. Cheng, and J. Xiao, “Uncertainty Calibration for Deep Audio Classifiers,” in Interspeech, 2022, pp. 1556–1560.

[39] T. Daikoku, “Temporal dynamics of uncertainty and prediction error in musical improvisation across different periods,” Scientific Reports, vol. 14, no. 1, p. 22297, 2024.

[40] J. Han, Z. Zhang, Z. Ren, and B. Schuller, “Exploring perception uncertainty for emotion recognition in dyadic conversation and music listening,” Cognitive Computation, vol. 13, pp. 231–240, 2021.

[41] A. Jaiswal, P. Singh, and V. Arora, “Improving active learning for melody estimation by disentangling uncertainties,” in IEEE ICASSP, 2026, pp. 3401–3405.

[42] K. R. Saxena and V. Arora, “Interactive singing melody extraction based on active adaptation,” IEEE TASLP, vol. 32, pp. 2729–2738, 2024.

[43] S. Chowdhuri, “Phononet: multi-stage deep neural networks for raga identification in hindustani classical music,” in ICMR, 2019.

[44] P. Singh, A. Mishra, A. Raina, and V. Arora, “Ontology-driven hierarchical learning for raga identification,” in 2025 National Conference on Communications (NCC), 2025, pp. 1–6.

[45] P. Singh, A. Gupta, A. Mishra, and V. Arora, “Identification and clustering of unseen ragas in indian art music,” in Proceedings of the 26th International Society for Music Information Retrieval Conference, 2025, pp. 811–818.