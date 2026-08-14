# CardioState-JEPA: Delay-Aware Cross-Modal Learning of a Shared Cardiac Representation

Hamza Shafiq<sup>1</sup> Hung Manh Pham<sup>2</sup> Bin Zhu<sup>2</sup> Pan Zhou<sup>2</sup> Jun Hu<sup>1</sup> Aaqib Saeed<sup>1∗</sup>

<sup>1</sup> Eindhoven University of Technology, Netherlands

<sup>2</sup> Singapore Management University, Singapore

## Abstract

Electrocardiography (ECG), photoplethysmography (PPG), and phonocardiography (PCG) provide complementary views of the same cardiac cycle, yet existing cardiac foundation models are trained for a single sensing modality, leaving the shared physiolog across sensors unexploited. We introduce CardioState-JEPA, a cardiac foundation model to learn a single shared representation jointly across ECG, PPG, and PCG, built on a physiology-aware joint-embedding predictive architecture. The model maps heterogeneous waveforms into a common token space, processes them with a single shared Transformer encoder, and learns by predicting masked latent cardiac states, placing the pretraining target on shared physiology rather than sensor-specific waveform appearance. To handle the temporal ofsets between electrical, mechanical, and hemodynamic events, cross-modal prediction uses a learned delay aligner that matches signals at the corresponding cardiac time. Because synchronized multi-sensor recordings are scarce, CardioState-JEPA first learns within-modality structure from abundant unimodal data and then uses paired data to align modalities in latent cardiac time. Evaluated as a frozen encoder across 25 downstream tasks spanning ECG, PPG, and PCG, our encoder improves average PPG classification by 8.2 AUROC points, PCG murmur detection by 18.8 AUROC points, and ECG classification by 15.5 AUROC points over the best self-supervised signal baseline and matches or exceeds cardiac models trained with privileged clinical text or supervised labels on several ECG benchmarks. These results establish that heterogeneous cardiac signals can mutually supervise a single foundation model of cardiac physiology.

## 1 Introduction

Cardiac signals are a common and important source of information in clinical diagnosis, long-term monitoring, wearable sensing, and everyday health assessment [24]. Among the many ways to observe cardiac activity, electrocardiography (ECG), photoplethysmography (PPG), and phonocardiography (PCG) remain especially important, as they capture electrical, hemodynamic, and acoustic aspects of the heart, respectively. These modalities are increasingly used across both clinical and daily settings, from arrhythmia analysis and heart sound screening to wearable monitoring of pulse, blood pressure, and physiological stress [29, 32, 38]. As shown in Figure 1, the three modalities are complementary observations of shared cardiac activity but provide diferent views of one underlying physiological process. Electrical activation initiates contraction, valve motion then produces the heart sounds, and blood ejection appears later as a peripheral pulse, so the three modalities observe the same beat at diferent physiological times.

Despite this physiological connection, machine learning for cardiac signals has largely progressed along separate modalityspecific paths. Recent advances in large-scale representation learning and foundation models have substantially improved performance for individual cardiac sensors, with ECG models supporting rhythm, conduction, and arrhythmia analysis [25, 28, 33], PPG models enabling wearable tasks such as heart rate estimation, blood pressure prediction, and vascular state assessment [35], and PCG models targeting acoustic signatures such as murmurs and valvular abnormalities [36]. However, these models are still typically trained and deployed within a single sensing modality, so structure learned from one view of cardiac physiology is not explicitly used to improve another. A unified cardiac representation model should instead exploit the fact that electrical, hemodynamic, and acoustic signals are diferent observations of the same underlying cardiac process.

![](images/7f113c3e6f417b07efae497cf8f8fb2589966c4b237f0b33fcd4f82beb0e8d27.jpg)  
Figure 1: Motivation of CardioState-JEPA. ECG, PPG, and PCG observe the same cardiac cycle at diferent physiological times, motivating a shared cardiac representation.

However, building such a model requires more than applying standard multimodal learning. ECG, PPG, and PCG are coupled by physiology but are shifted in time: PCG follows electrical activation via electromechanical coupling, while PPG arrives later via pulse transit. At the same time, the same cardiac event can appear as a sharp electrical deflection, an acoustic burst, or a smooth peripheral pulse wave. Consequently, naive timestamp alignment can compare diferent phases of the same beat, while raw cross-signal reconstruction can encourage sensor-specific shortcuts. These challenges motivate a learning objective that operates in a shared latent space and an alignment mechanism that explicitly accounts for physiological delay.

To address these challenges, we introduce CardioState-JEPA, a physiology-aware joint-embedding predictive architecture for unified cardiac representation learning across ECG, PPG, and PCG. The model maps heterogeneous waveforms into a common token space with lightweight modality-specific stems and processes them using a single shared Transformer encoder. Since the modalities do not share waveform appearance but do share latent cardiac structure, CardioState-JEPA learns by predicting masked latent cardiac states rather than reconstructing raw signals. This predictive formulation avoids emphasizing low-level sensor morphology and also removes the need for negative pairs, which can be ambiguous in physiological data because diferent subjects or segments may share rhythm, heart rate, or pathology.

CardioState-JEPA is also designed around the data regime of cardiac sensing. Large unimodal ECG, PPG, and PCG corpora are increasingly available, but synchronized multisensor recordings are comparatively scarce. We therefore use a two-stage curriculum. The model first learns within-modality structure from abundant unimodal recordings through intramodal masked latent prediction. It then uses paired recordings to align modalities through delay-aware cross-modal prediction, while continuing to sample unimodal data to preserve per-modality performance. Additional physiology-guided objectives encourage cardiac-cycle phase awareness, consistent delay estimates, and robust shared representations.

We evaluate CardioState-JEPA as a frozen encoder across 25 downstream tasks spanning ECG, PPG, and PCG, where a single shared encoder consistently outperforms strong modalityspecific baselines. Overall, our contributions are as follows:

• We formulate unified cardiac representation learning as the problem of recovering a shared latent cardiac state from heterogeneous ECG, PPG, and PCG observations that are shifted in time by physiological delay.

• We propose CardioState-JEPA, a joint-embedding predictive architecture that learns shared cardiac representations through intra-modal masked latent prediction and delayaware cross-modal latent prediction.

• We introduce a physiology-guided alignment strategy in which inter-modality delay is explicitly estimated and supervised, allowing electrical, hemodynamic, and acoustic signals to be aligned in latent cardiac time rather than by raw timestamp.

• We show that our pretrained encoder transfers across ECG, PPG, and PCG tasks, outperforming modality-specific baselines on average for PPG and PCG while remaining competitive on ECG, providing a unified alternative to separate per-sensor representation learning.

## 2 Related Work

General Time Series Foundation Models. General time series foundation models have become an important direction for learning transferable representations across forecasting, classification, and anomaly detection tasks [3, 39]. Most of these models rely on self-supervised learning to exploit large unlabeled time series corpora. Within this area, contrastive methods (TS2Vec [45], CoST [43]), masked-prediction methods inspired by masked autoencoders [12, 15], and joint-embedding predictive architectures [11] learn transferable representations by predicting latent rather than raw structure. Despite this progress, most general time series foundation models underperform on focused clinical tasks while still treating sensors or channels as independent streams and therefore do not explicitly model the structured correspondences that arise when multiple modalities observe the same underlying physical process.

Cardiac Signal Foundation Models. Building on progress in general self-supervised learning, recent work has focused on cardiac signals as a domain where large unlabeled datasets are increasingly accessible. For ECG, early self-supervised models such as ST-MEM [30] have demonstrated strong transfer to arrhythmia, rhythm, and morphology classification using masked prediction objectives on publicly available recordings. A recent line of ECG foundation models, including ECGFounder [23], ECG-FM [28], HeartLang [17], and D-BETA [33], further leverages large-scale paired labels or weak clinical reports as a multimodal framework to achieve higher performance, though at the cost of requiring privileged annotation. For PPG, PaPaGei [35] and AnyPPG [31] apply contrastive pretraining to wearable recordings and show strong performance in heart rate and blood pressure estimation. For PCG, general audio models such as AudioMAE [16] and CLAP [10] have been adapted to heart sound classification. However, all of these models are designed for a single sensing modality. CardioState-JEPA difers by training one shared encoder across ECG, PPG, and PCG simultaneously, treating the physiological correspondence between modalities as an additional supervisory signal rather than a complication to be avoided.

## 3 Method

## 3.1 Overview

CardioState-JEPA learns a single representation that serves ECG, PPG, and PCG together, instead of training a separate model per sensor. As shown in Figure 2, each modality is first mapped into a common token space by a light modalityspecific tokenizer, and all modalities are then processed by one shared Transformer encoder that produces a common cardiac code. Because the modalities are ofset in time by physiology, we train in two stages (detailed below): Stage I learns each modality’s structure from abundant unimodal data, and Stage II aligns modalities from scarce paired data through a learned delay.

![](images/a4c76f407d8fc9aff94048c61958701a78b5b0116c175b76b99e406f310f49a7.jpg)  
Figure 2: Overview of CardioState-JEPA. Stage I learns unimodal cardiac codes with masked latent prediction from abundant ECG, PPG, and PCG recordings. Stage II uses paired recordings for delay-aware cross-modal prediction, where a learned aligner estimates the physiological ofset so that one modality predicts another at the corresponding cardiac time.

## 3.2 Problem Formulation

We view ECG, PPG, and PCG not as three signal families but as three renderings of one hidden process, which makes precise what a shared representation should capture. A cardiac cycle produces a latent state $\mathbf { c } ( t ) \in \mathbb { R } ^ { d _ { c } }$ , and each sensor observes it through its own transduction and at its own point along the cardiovascular path:

$$
\mathbf { x } _ { m } ( t ) \ = \ O _ { m } ( \mathbf { c } \big ( t - \tau _ { m } ( t ) \big ) , \ \mathbf { u } _ { m } ( t ) \big ) ,\tag{1}
$$

where $O _ { m }$ is a modality-specific rendering, $\mathbf { u } _ { m }$ collects sensor nuisance such as noise and placement, and $\tau _ { m }$ is a physiological delay from electrical activation. These delays set cardiac signals apart from ordinary multimodal data: the ECG is nearly immediate, the PCG follows electromechanical coupling, and the PPG arrives only after pulse transit. Here, two modalities of the same beat thus share c but difer in rendering, nuisance, and relative delay $\tau _ { m  n } = \tau _ { n } - \tau _ { m }$ . Two requirements follow: First, since c is common to the whole cycle, the shared representation must be recoverable from a partial view of one modality. Second, since diferent sensors describe the same state at the same instant, their representations must agree once the delay $\tau _ { m  n }$ is removed. We do not model $\mathbf { u } _ { m }$ explicitly; predicting in latent space rather than reconstructing waveforms, together with pressure toward modality invariance, is enough to reduce reliance on it. The two requirements map onto the two stages: the first is met from abundant unimodal data, the second from the scarce paired data.

## 3.3 Modality Tokenizers and Shared Encoder

Since ECG, PPG, and PCG difer in sampling rate, channels, and morphology, we make them comparable with a modalityspecific tokenizer $f _ { m }$ before the shared encoder, so the encoder spends its capacity on cardiac structure rather than acquisition diferences. Given $\mathbf { x } _ { m } \in \mathbb { R } ^ { C _ { m } \times T _ { m } }$ , the tokenizer produces $\mathbf { h } _ { m } =$ $f _ { m } ( \mathbf { x } _ { m } ) \in \mathbb { R } ^ { N \times d }$ : a strided convolution whose stride is set per modality so all three emit tokens at a comparable rate, a multiscale depthwise block that adds context around short events such as the QRS complex, PPG upstroke, and first heart sound, and a linear projection, followed by sinusoidal positional and modality embeddings. A single shared Transformer encoder then processes these tokens with no modality-specific layers, so ECG, PPG, and PCG update the same parameters. Writing $\mathbf { H } _ { m } = g _ { \theta } ( \mathbf { h } _ { m } )$ , a shared projector maps each token to the cardiac code $\mathbf Z _ { m } = \pi ( \mathbf H _ { m } )$ used for both training and transfer. A momentum encoder $\bar { g } _ { \bar { \theta } } .$ , an exponential moving average of the tokenizers, encoder, and projector, is run without gradients on the unmasked signal to supply stable targets.

## 3.4 Stage I: Intra-Modal JEPA

Stage I enforces the first requirement, recovering the cardiac code from a partial view of one modality. We predict masked regions in latent space rather than reconstructing the waveform, because reconstruction rewards fitting sensor noise and fine morphology, exactly the nuisance we want the code to forget. Following the joint-embedding predictive principle [4], we mask a large fraction of the tokens in contiguous blocks, encode the visible context, and let a light predictor $q _ { \theta }$ match the momentum encoder’s code at the masked positions,

$$
\mathcal { L } _ { \mathrm { i n t r a } } = \frac { 1 } { | \mathcal { T } | } \sum _ { i \in \mathcal { T } } \mathcal { H } \big ( \mathrm { L N } ( \hat { \mathbf { z } } ^ { i } ) , \mathrm { L N } ( \bar { \mathbf { z } } ^ { i } ) \big ) ,\tag{2}
$$

where $\hat { \mathbf { z } } ^ { i }$ is the prediction, $\bar { \mathbf { z } } ^ { i }$ the stop-gradient target, LN LayerNorm, and H the Smooth-L1 loss. Because the masked blocks routinely span whole beats, the model cannot rely on local interpolation and must infer the missing state from neighboring cycles, which drives the code toward the rhythm and phase structure shared across modalities.

## 3.5 Stage II: Delay-Aware Cross-Modal JEPA

Intra-modal prediction recovers the code within each sensor but does not make sensors agree, the second requirement. Stage II predicts one modality from another on paired recordings, introduced after Stage I because alignment is meaningful only once each modality has reliable structure. Aligning tokens at equal timestamps would match an electrical event to a mechanical or hemodynamic one and teach surface appearance rather than shared state, so we predict through a learned delay. A delay head $h _ { \delta }$ maps the source code $\mathbf { z } _ { m } ^ { i }$ and a target-modality embedding $\mathbf { e } _ { n }$ to a bounded per-token ofset:

$$
\tau _ { m  n } ^ { i } = \tau _ { \mathrm { m a x } } \operatorname { t a n h } \big ( h _ { \delta } ( [ \mathbf { z } _ { m } ^ { i } ; \mathbf { e } _ { n } ] ) \big ) .\tag{3}
$$

and the target is gathered with a Gaussian kernel centered at the shifted time $t _ { i } + \tau ^ { i }$ over the in-band target tokens,

$$
a _ { i j } = \mathrm { s o f t m a x } _ { j } \bigg ( - \frac { 1 } { 2 } \Big ( \frac { t _ { j } - t _ { i } - \tau ^ { i } } { \sigma } \Big ) ^ { 2 } \bigg ) , \qquad \tilde { \mathbf { z } } _ { n } ^ { i } = \sum _ { j } a _ { i j } \bar { \mathbf { z } } _ { n } ^ { j } .\tag{4}
$$

A soft kernel is used rather than nearest-token selection because it is diferentiable in $\tau ^ { i }$ and thus gives the delay head a usable gradient, and it reduces to a soft interpolation when $\sigma$ is comparable to the token spacing. Each position is weighted by the certainty of its alignment,

$$
w _ { i } = 1 - H ( a _ { i } ) / { \log N _ { i } } ( N _ { i } > 1 ) , \qquad w _ { i } = 0 ( N _ { i } \le 1 ) ,\tag{5}
$$

with � the entropy of $a _ { i }$ and $N _ { i }$ the number of in-band tokens, so peaked, well-covered alignments approach one and ambiguous or empty ones are downweighted or excluded. The cross-modal objective predicts the aligned target under these weights,

$$
\mathcal { L } _ { \mathrm { c r o s s } } = \frac { \sum _ { i \in \mathcal { T } } w _ { i } \mathcal { H } \big ( \mathrm { L N } ( \hat { \mathbf { z } } _ { m n } ^ { i } ) , \mathrm { L N } ( \tilde { \mathbf { z } } _ { n } ^ { i } ) \big ) } { \sum _ { i \in \mathcal { T } } w _ { i } + \epsilon } ,\tag{6}
$$

where $\hat { \pmb { { z } } } _ { m n } ^ { i }$ comes from the same predictor $q _ { \theta }$ as in Eq. (2), now conditioned on $\tau ^ { i } ;$ sharing the predictor keeps both objectives in one space, with the two difering only in how the predictor is conditioned. Left unconstrained, the delay drifts toward trivial solutions, so we anchor it to physiology: beat detection gives a reference ofset per pair, the R-peak to first-heart-sound interval for ECG–PCG and the pulse arrival time for ECG–PPG, and we regress the estimate onto it,

$$
\mathcal { L } _ { \mathrm { d e l a y - s u p } } = \frac { 1 } { | \mathcal { T } | } \sum _ { i \in \mathcal { T } } \mathcal { H } ( \tau ^ { i } , \tau _ { \mathrm { a n c h o r } } ^ { i } ) .\tag{7}
$$

This supervision is applied only where reliable anchors exist; when too few clean beats are detected, the term is masked out while the delay head still runs and the window still contributes to $\mathcal { L } _ { \mathrm { c r o s s } } .$ , so noisy detection removes only the label, never the alignment path. The delay is therefore learned and physiologically supervised rather than fixed, and it is the mechanism that encourages a single shared space across electrical, acoustic, and hemodynamic sensing.

## 3.6 Auxiliary Objectives and Training

Our framework leverages two light terms to further shape the code: a cross-modal VICReg [5] term $\mathcal { L } _ { \mathrm { s t a t e } }$ pulls paired modalities into the same space while preventing collapse, and a phase term $\mathcal { L } _ { \mathrm { p h a s e } }$ predicts intra-beat phase from anchors, tying the code to the cardiac cycle. The full objective is:

$$
\begin{array} { r l } & { \mathcal { L } = \lambda _ { \mathrm { i n t r a } } \mathcal { L } _ { \mathrm { i n t r a } } + \lambda _ { \mathrm { c r o s s } } \mathcal { L } _ { \mathrm { c r o s s } } + \lambda _ { \mathrm { d e l a y } } \mathcal { L } _ { \mathrm { d e l a y - s u p } } } \\ & { \qquad + \lambda _ { \mathrm { s t a t e } } \mathcal { L } _ { \mathrm { s t a t e } } + \lambda _ { \mathrm { p h a s e } } \mathcal { L } _ { \mathrm { p h a s e } } . } \end{array}\tag{8}
$$

The curriculum follows the data asymmetry. In Stage I, each sample is a single modality, so only intra-modal prediction and the phase term are active; the cross-modal, delay, and state terms require paired inputs and switch on in Stage II, which warm-starts from Stage I and spends the scarce paired data on alignment while keeping unimodal data in the mixture so per-modality codes do not drift. At test time the encoder is frozen: a signal passes through its tokenizer, the shared encoder, and the projector, and the pooled code is read by a linear head, so every result measures transfer without encoder adaptation.

## 4 Experiments

## 4.1 Datasets and Downstream Tasks

In the first stage, the unimodal pretraining dataset includes MIMIC-IV-ECG [13] with 800K 12-lead ECG recordings at 500 Hz, PPG-EXT [29] as a large-scale PPG corpus at 125 Hz, and BMD-HS [1] for PCG recordings at 4000 Hz. In the second stage, we introduce cross-modal supervision by further using PPG-EXT [29] and VitalDB [21] for synchronous ECG-PPG pairs, EPHNOGRAM [18] for ECG-PCG pairs, and SensSmartTech [20] for trimodal recordings. Full dataset statistics are reported in the Appendix.

After pretraining, we assess whether the learned representation transfers to various downstream tasks. For ECG, we evaluate on PTB-XL [40] (with four subgroups: superclass, subclass, form, and rhythm) for arrhythmia classification, together with CPSC 2018 [27] and CSN [48], following [25]. For PPG, we use seventeen tasks covering arrhythmia detection, stress and activity recognition, respiratory rate, heart rate regression, blood pressure regression, $\mathrm { S p O } _ { 2 }$ estimation, and HRV estimation, using the pre-processed data from PulseLM [34]. For PCG, we evaluate murmur detection on CirCor DigiScope [32] and abnormal heart sound detection on CinC2016 [26]. The signal samples can be found in the Appendix. Finally, classification tasks are reported using macro-AUROC ×100, while regression tasks are reported using MAE.

<table><tr><td>Task</td><td>PaPaGei-S</td><td>PaPaGei-P</td><td>AnyPPG</td><td>PulsePPG</td><td>ChronosBolt</td><td>CardioState-JEPA</td></tr><tr><td>PPG Classification Tasks</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>WESAD ↑</td><td>65.3</td><td>65.7</td><td>70.1</td><td>66.8</td><td>64.0</td><td>74.5</td></tr><tr><td>DaLiA Activity ↑</td><td>68.4</td><td>71.1</td><td>79.0</td><td>77.3</td><td>77.2</td><td>90.3</td></tr><tr><td>MIMIC AF ↑</td><td>48.4</td><td>81.1</td><td>93.3</td><td>32.4</td><td>46.1</td><td>97.7</td></tr><tr><td>PPG Arrhythmia ↑</td><td>86.3</td><td>88.7</td><td>95.8</td><td>92.6</td><td>90.4</td><td>96.8</td></tr><tr><td>BIDMC RR ↑</td><td>51.3</td><td>38.2</td><td>53.2</td><td>43.5</td><td>38.0</td><td>78.4</td></tr><tr><td>UQVital RR ↑</td><td>40.6</td><td>41.8</td><td>41.8</td><td>29.5</td><td>30.5</td><td>44.6</td></tr><tr><td>Average (Classification) ↑</td><td>60.1</td><td>64.4</td><td>72.2</td><td>57.0</td><td>57.7</td><td>80.4</td></tr><tr><td>PPG Regression Tasks</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>DaLiA HR ↓</td><td>11.8</td><td>11.5</td><td>5.8</td><td>8.6</td><td>8.2</td><td>3.8</td></tr><tr><td>EarSet HR ↓</td><td>10.7</td><td>9.5</td><td>9.8</td><td>8.9</td><td>8.3</td><td>8.2</td></tr><tr><td>WildPPG HR ↓</td><td>10.0</td><td>9.9</td><td>5.4</td><td>7.7</td><td>7.5</td><td>5.4</td></tr><tr><td>Sensors DBP ↓</td><td>7.2</td><td>7.0</td><td>6.9</td><td>7.4</td><td>6.9</td><td>6.8</td></tr><tr><td>Sensors SBP ↓</td><td>20.0</td><td>19.4</td><td>16.7</td><td>18.6</td><td>18.0</td><td>16.8</td></tr><tr><td>UCI DBP ↓</td><td>7.8</td><td>7.9</td><td>6.7</td><td>6.9</td><td>7.2</td><td>5.8</td></tr><tr><td>UCI SBP ↓</td><td>18.5</td><td>18.1</td><td>15.2</td><td>16.2</td><td>17.4</td><td>10.3</td></tr><tr><td>BCG DBP ↓</td><td>10.1</td><td>12.9</td><td>4.5</td><td>11.2</td><td>10.8</td><td>2.8</td></tr><tr><td>BCG SBP ↓</td><td>13.7</td><td>10.2</td><td>13.5</td><td>12.7</td><td>11.8</td><td>4.4</td></tr><tr><td>UQVital SpO2↓</td><td>0.76</td><td>0.85</td><td>0.78</td><td>0.84</td><td>0.83</td><td>0.82</td></tr><tr><td>WildPPG RMSSD ↓</td><td>36.6</td><td>37.6</td><td>34.9</td><td>36.7</td><td>35.9</td><td>34.8</td></tr><tr><td>Average (Regression) ↓</td><td>13.4</td><td>13.2</td><td>10.9</td><td>12.3</td><td>12.1</td><td>9.1</td></tr></table>

Table 1: PPG Baseline Linear Probe Results (100% training data). ↑: macro-AUROC ×100 (higher is better); ↓: MAE (lower is better). Bold: best; underline: second best. Average (Classification) is the unweighted mean macro-AUROC across the 6 classification tasks. Average (Regression) is the unweighted mean MAE across the 11 regression tasks.

## 4.2 Baselines and Evaluation Protocol

Given the broad range of downstream tasks, we compare CardioState-JEPA against baselines specific to each modality. For ECG, we include ten self-supervised ECG-only methods: SimCLR [6], BYOL [14], BarlowTwins [46], MoCov3 [8], SimSiam [7], TS-TCC [9], CLOCS [19], ASTCL [41], CRT [47], and ST-MEM [30], following [25]. These models form the main comparison group because, like CardioState-JEPA, they do not rely on clinical reports or labels during representation learning. Meanwhile, we also report ECG-Founder [23] and ECG-Text models, including ECG-FM [28], HeartLang [17], AnyChat [22], ESI [44], MERL [25], and D-BETA [33]. These models are useful references, but they are not directly comparable because they use either largescale supervised labels or paired clinical reports. Therefore, in Table 2, bold and underline are assigned only within the selfsupervised ECG-only group, with CardioState-JEPA included in that group. The supervised and text-supervised models are shown separately to indicate how close a signal-only model can come to models trained with additional privileged information.

For PPG, we compare against PaPaGei-S, PaPaGei-P [35], AnyPPG [31], PulsePPG [37], and ChronosBolt [2]. For PCG, we use CLAP [10], AudioMAE [16] and StethoLM [42] as the main baselines. All methods are evaluated under a linear probing protocol where the pretrained encoder is frozen and only a task-specific linear head is trained. All downstream evaluations are performed using patient-disjoint splits. To further test label eficiency, ECG tasks are evaluated with 1%, 10%, and 100% of the training labels.

## 4.3 Implementation Details

Our model is implemented in PyTorch and uses a ViT-B shared encoder (12 layers, 12 attention heads, hidden dimension � = 768), trained on a single NVIDIA H100 GPU with AdamW. Stage I runs for 300K steps and Stage II for 200K steps with the cross-modal objectives enabled. Full optimization hyperparameters (learning-rate schedules, batch sizes, weight decay, and EMA momentum) are provided in the Appendix.

## 4.4 Main Results

We first examine whether a single frozen encoder can remain competitive across all three modalities. Tables 1, 2, and 3 summarize the PPG, ECG, and PCG results, respectively. Overall, CardioState-JEPA performs consistently across the three sensing domains, despite using the same shared encoder for all downstream tasks.

On PPG tasks, CardioState-JEPA improves the average classification AUROC from 72.2 to 80.4 compared with the strongest baseline, while also reducing the average regression MAE from 10.9 to 9.1. The gains are especially clear on activity recognition, atrial fibrillation detection, respiratory rate estimation, and blood pressure estimation. For ECG, CardioState-JEPA improves the average AUROC across all 18 settings by 15.5 points over the strongest self-supervised baseline (MoCo-v3; 84.1 vs. 68.5 in Table 2), outperforming every self-supervised ECG-only model on nearly all datasets and label fractions. While text-supervised and label-supervised ECG models still provide strong reference points, CardioState-JEPA matches or surpasses them on several benchmarks without using clinical reports or large supervised ECG labels. For PCG, the model achieves the best results on both CirCor murmur detection and CinC2016 abnormal heart-sound detection under the linear probing protocol. Since the same encoder is also used for ECG and PPG, this result supports the central hypothesis that shared cardiac pretraining can benefit acoustic heart-sound representation, rather than only electrical or optical signals.

<table><tr><td></td><td colspan="3">PTB-XL Super</td><td colspan="3">PTB-XL Sub</td><td colspan="3">PTB-XL Form</td><td colspan="3">PTB-XL Rhythm</td><td colspan="3">CPSC</td><td colspan="3">CSN</td><td></td></tr><tr><td>Method</td><td>1%</td><td>10%</td><td>100%</td><td>1%</td><td>10%</td><td>100%</td><td>1%</td><td>10%</td><td>100%</td><td>1%</td><td>10%</td><td>100%</td><td>1%</td><td>10%</td><td>100%</td><td>1%</td><td>10%</td><td>100%</td><td>Avg</td></tr><tr><td colspan="3">Self-supervised ECG-only models</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>SimCLR</td><td>63.4</td><td>69.8</td><td>73.5</td><td>60.8</td><td>68.3</td><td>73.4</td><td>55.0</td><td>57.0</td><td>62.5</td><td>51.4</td><td>69.4</td><td>77.7</td><td>59.8</td><td>68.5</td><td>76.5</td><td>59.0</td><td>67.3</td><td>73.2</td><td>65.9</td></tr><tr><td>BYOL</td><td>71.7</td><td>73.8</td><td>76.5</td><td>57.2</td><td>67.4</td><td>71.6</td><td>48.7</td><td>61.6</td><td>70.8</td><td>42.0</td><td>74.4</td><td>77.2</td><td>60.9</td><td>74.4</td><td>78.8</td><td>54.2</td><td>71.9</td><td>74.7</td><td>67.1</td></tr><tr><td>BarlowTwins</td><td>72.9</td><td>76.0</td><td>78.4</td><td>62.6</td><td>70.8</td><td>74.3</td><td>52.1</td><td>60.4</td><td>66.1</td><td>50.1</td><td>73.5</td><td>77.6</td><td>55.1</td><td>72.8</td><td>78.4</td><td>60.7</td><td>71.6</td><td>77.4</td><td>68.4</td></tr><tr><td>MoCo-v3</td><td>73.2</td><td>76.7</td><td>78.3</td><td>55.9</td><td>69.2</td><td>76.7</td><td>50.3</td><td>63.7</td><td>71.3</td><td>51.4</td><td>71.7</td><td>74.3</td><td>62.1</td><td>76.7</td><td>75.3</td><td>54.6</td><td>74.3</td><td>77.7</td><td>68.5</td></tr><tr><td>SimSiam</td><td>73.2</td><td>72.7</td><td>75.6</td><td>62.5</td><td>69.3</td><td>76.4</td><td>55.2</td><td>62.9</td><td>71.3</td><td>49.3</td><td>69.5</td><td>75.9</td><td>58.4</td><td>72.9</td><td>75.3</td><td>58.3</td><td>68.6</td><td>77.4</td><td>68.0</td></tr><tr><td>TS-TCC</td><td>70.7</td><td>75.9</td><td>78.9</td><td>53.5</td><td>67.0</td><td>77.9</td><td>48.0</td><td>61.8</td><td>71.2</td><td>43.3</td><td>69.5</td><td>78.2</td><td>57.1</td><td>73.6</td><td>78.7</td><td>55.3</td><td>68.5</td><td>76.8</td><td>67.0</td></tr><tr><td>CLOCS</td><td>68.9</td><td>73.4</td><td>76.3</td><td>57.9</td><td>72.6</td><td>76.2</td><td>52.0</td><td>58.0</td><td>72.7</td><td>47.2</td><td>71.9</td><td>76.3</td><td>59.6</td><td>77.8</td><td>77.5</td><td>54.4</td><td>71.9</td><td>76.1</td><td>67.8</td></tr><tr><td>ASTCL</td><td>72.5</td><td>77.3</td><td>81.0</td><td>61.9</td><td>68.8</td><td>76.5</td><td>44.1</td><td>60.9</td><td>67.0</td><td>52.4</td><td>72.0</td><td>76.1</td><td>57.9</td><td>77.0</td><td>79.5</td><td>56.4</td><td>70.9</td><td>75.8</td><td>68.2</td></tr><tr><td>CRT ST-MEM</td><td>69.7</td><td>78.2</td><td>77.2</td><td>62.0</td><td>70.8</td><td>78.7</td><td>46.4</td><td>59.5</td><td>68.7</td><td>47.4</td><td>73.5</td><td>74.4</td><td>58.0</td><td>76.4</td><td>82.0</td><td>56.2</td><td>73.7</td><td>78.8</td><td>68.4</td></tr><tr><td></td><td>61.1</td><td>66.9</td><td>71.4</td><td>54.1</td><td>57.9</td><td>63.6</td><td>55.7</td><td>60.0</td><td>66.1</td><td>51.1</td><td>65.4</td><td>74.9</td><td>56.7</td><td>63.3</td><td>70.4</td><td>59.8</td><td>66.9</td><td>71.4</td><td>63.2</td></tr><tr><td colspan="3">Label-supervised ECG model</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>ECGFounder</td><td>83.1</td><td>87.2</td><td>89.6</td><td>73.6</td><td>76.5</td><td>81.7</td><td>61.1</td><td>69.7</td><td>85.1</td><td>87.0</td><td>91.6</td><td>94.6</td><td>87.6</td><td>93.6</td><td>96.3</td><td>71.9</td><td>81.7</td><td>91.9</td><td>83.5</td></tr><tr><td colspan="3">ECG-Text foundation models</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>MERL</td><td>82.4</td><td>86.3</td><td>88.7</td><td>64.9</td><td>80.6</td><td>84.7</td><td>58.3</td><td>72.4</td><td>79.7</td><td>53.3</td><td>82.9</td><td>88.3</td><td>70.3</td><td>85.3</td><td>90.6</td><td>66.6</td><td>82.7</td><td>88.0</td><td>78.1</td></tr><tr><td>HeartLang</td><td>78.9</td><td>84.4</td><td>86.7</td><td>69.8</td><td>75.1</td><td>83.9</td><td>56.4</td><td>66.1</td><td>79.5</td><td>72.6</td><td>77.3</td><td>91.1</td><td>71.1</td><td>84.8</td><td>91.3</td><td>68.4</td><td>76.1</td><td>89.8</td><td>78.0</td></tr><tr><td>AnyChat</td><td>82.1</td><td>84.4</td><td>86.8</td><td>72.1</td><td>73.2</td><td>82.2</td><td>58.4</td><td>69.7</td><td>77.1</td><td>73.1</td><td>84.7</td><td>94.5</td><td>83.7</td><td>88.9</td><td>92.6</td><td>71.2</td><td>79.5</td><td>89.9</td><td>80.2</td></tr><tr><td>ESI</td><td>72.3</td><td>80.8</td><td>83.7</td><td>61.6</td><td>69.0</td><td>75.6</td><td>56.9</td><td>59.3</td><td>72.8</td><td>56.6</td><td>71.4</td><td>81.7</td><td>68.2</td><td>78.9</td><td>83.8</td><td>60.1</td><td>65.1</td><td>79.4</td><td>71.0</td></tr><tr><td>D-BETA</td><td>83.2</td><td>88.4 87.0</td><td>90.1</td><td>77.7</td><td>82.9</td><td>85.2</td><td>70.1</td><td>78.9</td><td>84.0</td><td>86.6</td><td>92.8</td><td>96.7</td><td>85.5</td><td>91.4</td><td>94.9</td><td>80.0</td><td>87.4</td><td>90.7</td><td>85.9</td></tr><tr><td>ECG-FM</td><td>81.3</td><td></td><td>89.4</td><td>71.7 71.0</td><td>76.9 79.1</td><td>83.0 86.4</td><td>64.6 60.5</td><td>76.3 70.0</td><td>86.1 83.9</td><td>77.0 87.8</td><td>91.0 92.4</td><td>96.3 96.4</td><td>84.7 87.6</td><td>92.6 93.4</td><td>95.7 95.8</td><td>76.0</td><td>87.7</td><td>95.0</td><td>84.0</td></tr><tr><td>CardioState-JEPA (Ours)</td><td>81.3</td><td>86.4</td><td>89.1</td></table>

Table 2: ECG linear probing results under 1%, 10%, and 100% label fractions across six datasets. Avg is the mean macro-AUROC across all 18 settings (six datasets × three label fractions). Label-supervised and ECG-text models are shown as reference model but are excluded from the SSL bold/underline comparison.

<table><tr><td>Method</td><td> $\mathbf { C i r C o r } \uparrow$ </td><td>CinC2016 ↑</td></tr><tr><td>StethoLM</td><td> $6 4 . 9 \pm 1 . 2$ </td><td> $4 5 . 7 \pm 0 . 4$ </td></tr><tr><td>CLAP</td><td> $7 5 . 3 \pm 1 . 8$ </td><td> $4 4 . 3 \pm 0 . 7 $ </td></tr><tr><td>AudioMAE</td><td> ${ \underline { { 7 9 . 1 } } } \pm 2 . 7$ </td><td> $6 2 . 4 \pm 0 . 7$ </td></tr><tr><td>CardioState-JEPA (Ours)</td><td> ${ \bf 9 7 . 9 \pm 0 . 1 }$ </td><td> ${ \bf 6 6 . 8 \pm 0 . 4 }$ </td></tr></table>

Table 3: PCG linear probe results on the murmur detection and abnormal heart sound detection tasks (macro-AUROC ×100, ↑; mean±std over 3 seeds).

## 4.5 Visualizing the Shared Space

While the downstream results show task-level transfer, they do not directly reveal how the shared cardiac space is organized. To inspect this, we visualize the pooled cardiac codes of held-out co-recorded ECG, PPG, and PCG samples using t-SNE at two points in training, after Stage I and after Stage II. Figure 3 shows that after Stage I the samples form clear modality-specific clusters, since each modality has only been trained in isolation. After Stage II the same samples become thoroughly interleaved across modalities, indicating that crossmodal training reshapes the space from a sensor-driven layout

![](images/90065ab621e707d089c7cf4d93d7a9486f77a0b593c942d38c8f86fc0b8b27ef.jpg)  
Figure 3: T-SNE of cardiac codes from co-recorded ECG, PPG, and PCG samples after Stage I (left) and Stage II (right).

into a modality-invariant one.

We quantify this shift with the modality silhouette of the pooled codes, which measures how separable the ECG, PPG, and PCG clusters are. It falls from 0.121 after Stage I to −0.006 after Stage II, so the sensor-specific separation is efectively removed, and the shared code becomes largely invariant to the sensing modality. Because downstream performance also improves from Stage I to Stage II, this mixing reflects genuine cross-modal alignment rather than a collapse of the representation.

## 4.6 Ablation Studies

Finally, we analyze which parts of CardioState-JEPA contribute to the observed gains. We start with the role of multimodal pretraining. The top block of Table 4 compares models pretrained with diferent modality combinations while keeping the architecture and training budget fixed. This comparison allows us to separate the efect of additional cardiac modalities from simple changes in model capacity. The full trimodal setting gives the best overall balance across the three domains, strongest on PPG regression and PCG and within noise of the best bimodal variant on ECG and PPG classification, indicating that the added modalities provide useful supervisory signal for single-modality downstream performance rather than a simple capacity increase.

<table><tr><td>Model / Variant</td><td>ECG Avg ↑</td><td>PPG Cls Avg ↑</td><td>PPG Reg Avg ↓</td><td>PCG Avg ↑</td></tr><tr><td colspan="5">Pretraining Modality Combination</td></tr><tr><td>ECG only</td><td> $8 9 . 7 \pm 0 . 5$ </td><td></td><td></td><td></td></tr><tr><td>PPG only</td><td></td><td> $7 8 . 5 \pm 2 . 6$ </td><td> $1 0 . 7 \pm 0 . 6$ </td><td></td></tr><tr><td>PCG only</td><td></td><td></td><td></td><td> $6 0 . 2 \pm 0 . 9$ </td></tr><tr><td> $\mathrm { E C G } + \mathrm { P P G }$ </td><td> ${ \bf 9 1 . 4 \pm 0 . 4 }$ </td><td> ${ \bf 8 0 . 6 \pm 2 . 8 }$ </td><td> $1 0 . 1 \pm 0 . 8$ </td><td></td></tr><tr><td> $\mathrm { E C G } + \mathrm { P C G }$ </td><td> $8 8 . 4 \pm 0 . 6 $ </td><td></td><td></td><td> $7 9 . 7 \pm 0 . 7$ </td></tr><tr><td>CardioState-JEPA (Ours)</td><td> $9 0 . 9 \pm 0 . 3 $ </td><td> $8 0 . 4 \pm 2 . 5$ </td><td> ${ \bf 9 . 1 \pm 0 . 6 }$ </td><td> $\mathbf { 8 2 . 3 \pm 0 . 2 }$ </td></tr><tr><td colspan="5">Self-Supervised Objective</td></tr><tr><td>SimCLR</td><td> $8 9 . 5 \pm 0 . 5$ </td><td> $7 0 . 1 \pm 2 . 9$ </td><td> $1 1 . 1 \pm 0 . 6$ </td><td> $8 0 . 0 \pm 0 . 7$ </td></tr><tr><td>BYOL</td><td> $8 9 . 3 \pm 0 . 4$ </td><td> $6 6 . 7 \pm 3 . 8$ </td><td> $1 1 . 0 \pm 0 . 5$ </td><td> $8 0 . 3 \pm 0 . 4$ </td></tr><tr><td>BarlowTwins</td><td> $8 8 . 6 \pm 0 . 4$ </td><td> $6 4 . 1 \pm 4 . 4$ </td><td> $1 1 . 3 \pm 0 . 5$ </td><td> $7 8 . 8 \pm 1 . 0$ </td></tr><tr><td>MAE</td><td> $8 4 . 4 \pm 0 . 5$ </td><td> $6 2 . 0 \pm 3 . 8$ </td><td> $1 1 . 8 \pm 0 . 5$ </td><td> $7 8 . 1 \pm 0 . 3$ </td></tr><tr><td>CardioState-JEPA (Ours)</td><td> ${ \bf 9 0 . 9 \pm 0 . 3 }$ </td><td> ${ \bf 8 0 . 4 \pm 2 . 5 }$ </td><td> ${ \bf 9 . 1 \pm 0 . 6 }$ </td><td> $\mathbf { 8 2 . 3 \pm 0 . 2 }$ </td></tr><tr><td colspan="5">Auxiliary Loss Terms</td></tr><tr><td>w/o Cross-modal prediction</td><td> $8 3 . 8 \pm 0 . 5$ </td><td> $6 3 . 9 \pm 3 . 5$ </td><td> $1 0 . 7 \pm 0 . 5$ </td><td> $7 8 . 4 \pm 0 . 4$ </td></tr><tr><td>w/o State alignment</td><td> $8 4 . 4 \pm 0 . 5$ </td><td> $6 6 . 3 \pm 3 . 5$ </td><td> $1 1 . 2 \pm 0 . 5$ </td><td> $7 9 . 3 \pm 0 . 5$ </td></tr><tr><td>w/o Delay modeling</td><td> $8 5 . 3 \pm 0 . 6$ </td><td> $6 8 . 9 \pm 3 . 7 $ </td><td> $1 1 . 1 \pm 0 . 6$ </td><td> $8 1 . 3 \pm 0 . 5$ </td></tr><tr><td>w/o Phase supervision</td><td> $8 8 . 7 \pm 0 . 5$ </td><td> $7 1 . 0 \pm 3 . 1$ </td><td> $1 0 . 4 \pm 0 . 6$ </td><td> $7 8 . 3 \pm 0 . 5$ </td></tr><tr><td>CardioState-JEPA (Ours)</td><td> ${ \bf 9 0 . 9 \pm 0 . 3 }$ </td><td> ${ \bf 8 0 . 4 \pm 2 . 5 }$ </td><td> ${ \bf 9 . 1 \pm 0 . 6 }$ </td><td> $\mathbf { 8 2 . 3 \pm 0 . 2 }$ </td></tr></table>

Table 4: Ablation studies on pretraining modalities, self-supervised objectives, and auxiliary losses.

To further isolate the efect of the learning objective, we replace the JEPA objective with SimCLR [6], BYOL [14], BarlowTwins [46], and MAE [15], while keeping the data and architecture unchanged. As shown in the middle block of Table 4, masked latent prediction gives the strongest overall results. This supports the use of JEPA for heterogeneous cardiac signals, where predicting latent structure is likely more useful than reconstructing raw waveforms or relying only on instance discrimination.

We also remove each auxiliary objective from the full model in the bottom block of Table 4. The results show that no single term explains all of the improvements. Instead, cross-modal prediction, state alignment, delay modeling, and phase supervision provide complementary benefits. This is consistent with the design of CardioState-JEPA, where the main prediction losses learn transferable representations while the auxiliary losses guide the representation toward physiologically meaningful structure. The Appendix further reports an inputlength ablation and a sensitivity analysis over the loss weights, both of which show that CardioState-JEPA is robust to these choices.

foundation model of latent cardiac physiology. We introduced CardioState-JEPA, a physiology-aware joint embedding predictive framework that learns from ECG, PPG, and PCG through intra-modal latent prediction and delay-aware cross-modal alignment. By predicting latent cardiac states and aligning modalities at the corresponding cardiac time, CardioState-JEPA enables electrical, hemodynamic, and acoustic signals to supervise one another. Our experiments show that the unified model transfers across modalities and surpasses strong self-supervised signal baselines on representative tasks. These results support our hypothesis that heterogeneous cardiac signals can provide mutual supervision for learning a shared physiological representation. We hope this motivates a shift from per-sensor cardiac models toward unified representations that exploit, rather than avoid, the physiological correspondence between modalities.

## 5 Conclusion

We studied whether cardiac representation learning can move beyond separate sensor-specific pretraining toward a shared

## Acknowledgments

This work was supported by the NWO AiNed Fellowship Grant of A.S., and in part by Google.org and the Google Cloud Research Credits program through the Gemini Academic Program. We also acknowledge the use of the Dutch National Supercomputer Snellius for essential computational tasks.

## References

[1] Shams Nafisa Ali, Afia Zahin, Samiul Based Shuvo, Nusrat Binta Nizam, Shoyad Ibn Sabur Khan Nuhash, Sayeed Sajjad Razin, S. M. Sakeef Sani, Farihin Rahman, Nawshad Binta Nizam, Farhat Binte Azam, Rakib Hossen, Sumaiya Ohab, Nawsabah Noor, and Taufiq Hasan. BUET multi-disease heart sound dataset: A comprehensive auscultation dataset for developing computer-aided diagnostic systems. Computer Methods and

Programs in Biomedicine Update, 9:100237, 2026. ISSN 2666- 9900. doi: 10.1016/j.cmpbup.2026.100237.

[2] Abdul Fatir Ansari, Lorenzo Stella, Caner Turkmen, Xiyuan Zhang, Pedro Mercado, Huibin Shen, Oleksandr Shchur, Syama S. Rangapuram, Sebastian Pineda Arango, Shubham Kapoor, Jasper Zschiegner, Danielle C. Maddix, Hao Wang, Michael W. Mahoney, Kari Torkkola, Andrew Gordon Wilson, Michael Bohlke-Schneider, and Yuyang Wang. Chronos: Learning the language of time series. Transactions on Machine Learning Research, 2024. URL https://openreview.net/forum? id=gerNCVqqtR.

[3] Abdul Fatir Ansari, Oleksandr Shchur, Jaris Küken, Andreas Auer, Boran Han, Pedro Mercado, Syama Sundar Rangapuram, Huibin Shen, Lorenzo Stella, Xiyuan Zhang, et al. Chronos-2: From univariate to universal forecasting. arXiv preprint arXiv:2510.15821, 2025.

[4] Mahmoud Assran, Quentin Duval, Ishan Misra, Piotr Bojanowski, Pascal Vincent, Michael Rabbat, Yann LeCun, and Nicolas Ballas. Self-supervised learning from images with a joint-embedding predictive architecture. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 15619–15629, 2023. doi: 10.1109/CVPR52729.2023.01499.

[5] Adrien Bardes, Jean Ponce, and Yann LeCun. VICReg: Varianceinvariance-covariance regularization for self-supervised learning. In International Conference on Learning Representations (ICLR), 2022. URL https://openreview.net/forum?id= xm6YD62D1Ub.

[6] Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton. A simple framework for contrastive learning of visual representations. In Proceedings ofthe 37th International Conference on Machine Learning, volume 119 of Proceedings ofMachine Learning Research, pages 1597–1607. PMLR, 2020.

[7] Xinlei Chen and Kaiming He. Exploring simple siamese representation learning. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 15750–15758, 2021. doi: 10.1109/CVPR46437.2021.01549.

[8] Xinlei Chen, Saining Xie, and Kaiming He. An empirical study of training self-supervised vision transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 9620–9629, 2021. doi: 10.1109/ICCV48922. 2021.00950.

[9] Emadeldeen Eldele, Mohamed Ragab, Zhenghua Chen, Min Wu, Chee Keong Kwoh, Xiaoli Li, and Cuntai Guan. Time-series representation learning via temporal and contextual contrasting. In Proceedings ofthe Thirtieth International Joint Conference on Artificial Intelligence (IJCAI), pages 2352–2359, 2021. doi: 10.24963/ijcai.2021/324.

[10] Benjamin Elizalde, Soham Deshmukh, Mahmoud Al Ismail, and Huaming Wang. CLAP: Learning audio concepts from natural language supervision. In IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 1–5, 2023. doi: 10.1109/ICASSP49357.2023.10095889.

[11] Sofiane Ennadir, Siavash Golkar, and Leopoldo Sarra. Joint embedding go temporal. In NeurIPS Workshop on Time Series

in the Age ofLarge Models, 2024. URL https://openreview.net/ forum?id=FIdbozebmy.

[12] Mononito Goswami, Konrad Szafer, Arjun Choudhry, Yifu Cai, Shuo Li, and Artur Dubrawski. Moment: A family of open timeseries foundation models. arXiv preprint arXiv:2402.03885, 2024.

[13] Brian Gow, Tom Pollard, Larry A. Nathanson, Alistair Johnson, Benjamin Moody, Chrystinne Fernandes, Nathaniel Greenbaum, Jonathan W. Waks, Parastou Eslami, Tanner Carbonati, Ashish Chaudhari, Elizabeth Herbst, Dana Moukheiber, Seth Berkowitz, Roger Mark, and Steven Horng. MIMIC-IV-ECG: Diagnostic electrocardiogram matched subset. PhysioNet, September 2023. doi: 10.13026/4nqg-sb35. URL https://physionet.org/content/ mimic-iv-ecg/1.0/. Version 1.0.

[14] Jean-Bastien Grill, Florian Strub, Florent Altché, Corentin Tallec, Pierre H. Richemond, Elena Buchatskaya, Carl Doersch, Bernardo Avila Pires, Zhaohan Daniel Guo, Mohammad Gheshlaghi Azar, Bilal Piot, Koray Kavukcuoglu, Rémi Munos, and Michal Valko. Bootstrap your own latent: A new approach to self-supervised learning. Advances in Neural Information Processing Systems, 33:21271–21284, 2020.

[15] Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, and Ross Girshick. Masked autoencoders are scalable vision learners. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 15979–15988, 2022. doi: 10.1109/CVPR52688.2022.01553.

[16] Po-Yao Huang, Hu Xu, Juncheng Li, Alexei Baevski, Michael Auli, Wojciech Galuba, Florian Metze, and Christoph Feichtenhofer. Masked autoencoders that listen. In Advances in Neural Information Processing Systems, volume 35, pages 28708–28720, 2022.

[17] Jiarui Jin, Haoyu Wang, Hongyan Li, Jun Li, Jiahui Pan, and Shenda Hong. Reading your heart: Learning ECG words and sentences via pre-training ECG language model. In International Conference on Learning Representations (ICLR), 2025. URL https://openreview.net/forum?id=6Hz1Ko087B.

[18] Arsalan Kazemnejad, Sajjad Karimi, Peiman Gordany, Gari D. Cliford, and Reza Sameni. An open-access simultaneous electrocardiogram and phonocardiogram database. Physiological Measurement, 45(5):055005, 2024. doi: 10.1088/1361- 6579/ad43af.

[19] Dani Kiyasseh, Tingting Zhu, and David A. Clifton. CLOCS: Contrastive learning of cardiac signals across space, time, and patients. In Proceedings ofthe 38th International Conference on Machine Learning, volume 139 of Proceedings ofMachine Learning Research, pages 5606–5615. PMLR, 2021.

[20] Aleksandar Lazović, Predrag Tadić, Natalija Ðorđević, Vladimir Atanasoski, Masa Tiosavljevic, Marija Ivanovic, Ljupco Hadzievski, Arsen Ristic, Vladan Vukcevic, and Jovana Petrovic. SensSmartTech database of cardiovascular signals synchronously recorded by an electrocardiograph, phonocardiograph, photoplethysmograph and accelerometer. PhysioNet, December 2024. doi: 10.13026/fy9p-n277. URL https: //physionet.org/content/senssmarttech/1.0.0/. Version 1.0.0.

[21] Hyung-Chul Lee, Yoonsang Park, Soo Bin Yoon, Seong Mi Yang, Dongnyeok Park, and Chul-Woo Jung. VitalDB, a highfidelity multi-parameter vital signs database in surgical patients. Scientific Data, 9(1):279, 2022. doi: 10.1038/s41597-022- 01411-5.

[22] Haitao Li, Ziyu Li, Yiheng Mao, Ziyi Liu, Zhoujian Sun, and Zhengxing Huang. anyECG-chat: A generalist ECG-MLLM for flexible ECG input and multi-task understanding. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, pages 597–605, 2026. doi: 10.1609/aaai.v40i1.37024.

[23] Jun Li, Aaron D. Aguirre, Valdery Moura Junior, Jiarui Jin, Che Liu, Lanhai Zhong, Chenxi Sun, Gari D. Cliford, M. Brandon Westover, and Shenda Hong. An electrocardiogram foundation model built on over 10 million recordings. NEJM AI, 2(7): AIoa2401033, 2025. doi: 10.1056/AIoa2401033.

[24] Jian Lin, Rumin Fu, Xinxiang Zhong, Peng Yu, Guoxin Tan, Wei Li, Huan Zhang, Yangfan Li, Lei Zhou, and Chengyun Ning. Wearable sensors and devices for real-time cardiovascular disease monitoring. Cell reports physical science, 2(8), 2021.

[25] Che Liu, Zhongwei Wan, Cheng Ouyang, Anand Shah, Wenjia Bai, and Rossella Arcucci. Zero-shot ECG classification with multimodal learning and test-time clinical knowledge enhancement. In Proceedings ofthe 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pages 31949–31963. PMLR, 2024. URL https://proceedings.mlr.press/v235/liu24bg.html.

[26] Chengyu Liu, David Springer, Qiao Li, Benjamin Moody, Ricardo Abad Juan, Francisco J. Chorro, Francisco Castells, Jose Millet Roig, Ikaro Silva, Alistair E. W. Johnson, Zeeshan Syed, Samuel E. Schmidt, Chrysa D. Papadaniil, Leontios Hadjileontiadis, Hosein Naseri, Ali Moukadem, Alain Dieterlen, Christian Brandt, Hong Tang, Maryam Samieinasab, Mohammad Reza Samieinasab, Reza Sameni, Roger G. Mark, and Gari D. Cliford. An open access database for the evaluation of heart sound algorithms. Physiological Measurement, 37(12): 2181–2213, 2016. doi: 10.1088/0967-3334/37/12/2181.

[27] Feifei Liu, Chengyu Liu, Lina Zhao, Xiangyu Zhang, Xiaoling Wu, Xiaoyan Xu, Yulin Liu, Caiyun Ma, Shoushui Wei, Zhiqiang He, Jianqing Li, and Eddie Ng Yin. An open access database for evaluating the algorithms of electrocardiogram rhythm and morphology abnormality detection. Journal of Medical Imaging and Health Informatics, 8(7):1368–1373, 2018. doi: 10.1166/ jmihi.2018.2442.

[28] Kaden McKeen, Sameer Masood, Augustin Toma, Barry Rubin, and Bo Wang. ECG-FM: An open electrocardiogram foundation model. JAMIA Open, 8(5):ooaf122, 2025. doi: 10.1093/jamiaopen/ooaf122.

[29] Mohammad Moulaeifard, Marie Kutscher, Philip J. Aston, Peter H. Charlton, and Nils Strodthof. MIMIC-III-Ext-PPG, a PPG-based benchmark dataset for cardiovascular and respiratory signal analysis. Scientific Data, 13:668, 2026. doi: 10.1038/s41597-026-07335-8.

[30] Yeongyeon Na, Minje Park, Yunwon Tae, and Sunghoon Joo. Guiding masked representation learning to capture spatiotemporal relationship of electrocardiogram. In International Conference on Learning Representations (ICLR), 2024. URL https://openreview.net/forum?id=WcOohbsF4H.

[31] Guangkun Nie, Gongzheng Tang, Yujie Xiao, Jun Li, Shun Huang, Deyun Zhang, Qinghao Zhao, and Shenda Hong. AnyPPG: An ECG-guided PPG foundation model trained on over 100,000 hours of recordings for holistic health profiling. arXiv preprint arXiv:2511.01747, 2025. doi: 10.48550/arXiv.2511.01747.

[32] Jorge Oliveira, Francesco Renna, Paulo Costa, Marcelo Nogueira, Cristina Oliveira, Andoni Elola, Carlos Ferreira, Alipio Jorge, Ali Bahrami Rad, Reza Sameni, Gari D. Cliford, and Miguel Coimbra. The CirCor DigiScope phonocardiogram dataset. PhysioNet, 2022. doi: 10.13026/tshs-mw03. URL https://physionet.org/content/circor-heart-sound/1.0.3/. Version 1.0.3.

[33] Hung Manh Pham, Aaqib Saeed, and Dong Ma. Boosting masked ECG-text auto-encoders as discriminative learners. In Proceedings of the 42nd International Conference on Machine Learning, 2025. URL https://openreview.net/forum?id= mM65b81LdM.

[34] Hung Manh Pham, Jinyang Wu, Xiao Ma, Yiming Zhang, Yixin Xu, Aaqib Saeed, Bin Zhu, Zhou Pan, and Dong Ma. PulseLM: A foundation dataset and benchmark for PPG-text learning. arXiv preprint arXiv:2603.03331, 2026. doi: 10.48550/arXiv. 2603.03331.

[35] Arvind Pillai, Dimitris Spathis, Fahim Kawsar, and Mohammad Malekzadeh. PaPaGei: Open foundation models for optical physiological signals. In International Conference on Learning Representations (ICLR), 2025. URL https://openreview.net/ forum?id=kYwTmlq6Vn.

[36] Matthew A. Reyna, Yashar Kiarashi, Andoni Elola, Jorge Oliveira, Francesco Renna, Annie Gu, Erick A. Perez Alday, Nadi Sadr, Ashish Sharma, Jacques Kpodonu, Sandra Mattos, Miguel T. Coimbra, Reza Sameni, Ali Bahrami Rad, and Gari D. Cliford. Heart murmur detection from phonocardiogram recordings: The george b. moody physionet challenge 2022. PLOS Digital Health, 2(9):e0000324, 2023. doi: 10.1371/journal.pdig.0000324.

[37] Mithun Saha, Maxwell A. Xu, Wanting Mao, Sameer Neupane, James M. Rehg, and Santosh Kumar. Pulse-PPG: An open-source field-trained PPG foundation model for wearable applications across lab and field settings. Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies, 9(3):126:1–126:35, 2025. doi: 10.1145/3749494.

[38] Philipp Schmidt, Attila Reiss, Robert Dürichen, et al. Introducing wesad, a multimodal dataset for wearable stress and afect detection. In Proceedings ofthe 20th ACM International Conference on Multimodal Interaction (ICMI), pages 400–408, 2018. doi: 10.1145/3242969.3242985.

[39] Xiaoming Shi, Shiyu Wang, Yuqi Nie, Dianqi Li, Zhou Ye, Qingsong Wen, and Ming Jin. Time-moe: Billion-scale time series foundation models with mixture of experts. In International conference on learning representations, volume 2025, pages 34635–34667, 2025.

[40] Patrick Wagner, Nils Strodthof, Ralf-Dieter Bousseljot, Dieter Kreiseler, Fatima I. Lunze, Wojciech Samek, and Tobias Schaefter. PTB-XL, a large publicly available electrocardiography dataset. Scientific Data, 7(1):154, 2020. doi: 10.1038/s41597-020-0495-6.

[41] Ning Wang, Panpan Feng, Zhaoyang Ge, Yanjie Zhou, Bing Zhou, and Zongmin Wang. Adversarial spatiotemporal contrastive learning for electrocardiogram signals. IEEE Transactions on Neural Networks and Learning Systems, 35(10): 13845–13859, 2024. doi: 10.1109/TNNLS.2023.3272153.

[42] Yishan Wang, Tsai-Ning Wang, Mathias Funk, and Aaqib Saeed. Stetholm: Audio language model for cardiopulmonary analysis across clinical tasks. Transactions on Machine Learning Research, 2026. URL https://arxiv.org/abs/2603.00355.

[43] Gerald Woo, Chenghao Liu, Doyen Sahoo, Akshat Kumar, and Steven Hoi. Cost: Contrastive learning of disentangled seasonaltrend representations for time series forecasting. arXiv preprint arXiv:2202.01575, 2022.

[44] Han Yu, Peikun Guo, and Akane Sano. ECG semantic integrator (ESI): A foundation ECG model pretrained with LLM-enhanced cardiological text. Transactions on Machine Learning Research, 2024. ISSN 2835-8856. URL https://openreview.net/forum? id=giEbq8Khcf.

[45] Zhihan Yue, Yujing Wang, Juanyong Duan, Tianmeng Yang, Congrui Huang, Yunhai Tong, and Bixiong Xu. Ts2vec: Towards universal representation of time series. In Proceedings ofthe AAAI conference on artificial intelligence, volume 36, pages 8980–8987, 2022.

[46] Jure Zbontar, Li Jing, Ishan Misra, Yann LeCun, and Stéphane Deny. Barlow twins: Self-supervised learning via redundancy reduction. In Proceedings ofthe 38th International Conference on Machine Learning, volume 139 of Proceedings ofMachine Learning Research, pages 12310–12320. PMLR, 2021.

[47] Wenrui Zhang, Ling Yang, Shijia Geng, and Shenda Hong. Self-supervised time series representation learning via cross reconstruction transformer. IEEE Transactions on Neural Networks and Learning Systems, 35(11):16129–16138, 2024. doi: 10.1109/TNNLS.2023.3292066.

[48] Jianwei Zheng, Jianming Zhang, Sidy Danioko, Hai Yao, Hangyuan Guo, and Cyril Rakovski. A 12-lead electrocardiogram database for arrhythmia research covering more than 10,000 patients. Scientific Data, 7(1):48, 2020. doi: 10.1038/s41597-020-0386-x.

## A Appendix

## A.1 Data and Training Configuration

Table 5 summarizes the pretraining and downstream linear probing settings used throughout our experiments. For each pretraining corpus we list the number of training and validation recordings, and for each downstream benchmark we list the number of classes together with the train, validation, and test splits. Every downstream task uses the same optimizer, epoch budget, batch size, and learning rate, so that the reported diferences reflect the quality of the frozen encoder rather than any per task tuning of the linear head. Keeping this protocol identical across ECG, PPG, and PCG also makes the comparison between the three sensing modalities easy to interpret, since only the input data and the linear head change from one task to the next. Tasks with a dash in the class column are regression tasks that we evaluate with mean absolute error, while all remaining tasks are classification tasks that we evaluate with macro-AUROC.

## A.2 Additional Implementation Details

The main paper reports the architecture and the high level training budget, and here we give the full optimization settings so that the results can be reproduced. Both pretraining stages use AdamW with weight decay 0.05. In Stage I the model is pretrained for 300K steps with batch size 196 and a peak learning rate of $1 . 2 \times 1 0 ^ { - 4 }$ that cosine decays to $1 \times 1 0 ^ { - 6 }$ . In Stage II the model is trained for a further 200K steps with batch size 64 and a peak learning rate of $6 \times 1 0 ^ { - 5 }$ that decays to $1 \times 1 0 ^ { - 6 }$ , with the cross modal objectives enabled. The shared encoder follows the ViT-B configuration with 12 layers, 12 attention heads, and hidden dimension 768. The momentum encoder is an exponential moving average whose momentum increases linearly from 0.998 to 0.9999 over the course of pretraining. All pretraining runs on a single NVIDIA H100 GPU. During downstream evaluation the encoder is frozen and only a linear head is trained, using the per task splits and the AdamW settings listed in Table 5.

## A.3 Training Procedure

Algorithm 1 summarizes the two stage curriculum. Stage I learns the structure of each modality from abundant unimodal data through masked latent prediction, and Stage II warm starts from Stage I and spends the scarce paired data on delay aware cross modal prediction while continuing to sample unimodal data so that the per modality codes do not drift.

## A.4 Input Length Ablation

We further study whether CardioState-JEPA stays stable when the length of the input window changes, because a representation that is sensitive to the exact segment duration is harder to deploy in practice. Table 6 reports four representative ECG task groups at window lengths of 2.5, 5, and 10 seconds, using 10% of the training labels. For each task group we give the score at every window length together with the mean and the standard deviation across the three settings. CardioState-JEPA reaches the smallest standard deviation in most task groups, which shows that its cardiac representation stays reliable regardless of how much signal context is available at test time. This robustness matters in real deployments, where the amount of clean signal that a sensor can capture varies with the recording conditions and with the patient.

```tcl
Algorithm 1: Two-stage training of CardioState-JEPA
Require: Unimodal corpora $\{ \mathcal { D } _ { m } \}$ for ECG, PPG, PCG;
paired corpora $\mathcal { P } ;$ tokenizers $\{ f _ { m } \} ;$ shared encoder $g _ { \theta } ;$
projector �; predictor ${ q } _ { \theta } ;$ delay head $h _ { \delta } ;$ momentum
encoder $\bar { g } _ { \bar { \theta } }$
1: Stage I, intra-modal prediction
2: for each pretraining step do
3: Sample a unimodal batch $x _ { m }$ from $\{ \mathcal { D } _ { m } \}$
4: Tokenize $h _ { m } \ = \ f _ { m } ( x _ { m } )$ and split into context and
masked blocks
5: Encode the context with $g _ { \theta }$ and predict the masked
codes with $q _ { \theta }$
6: Form stop-gradient targets with $\bar { g } _ { \bar { \theta } }$ on the unmasked
signal
7: Update � on $\lambda _ { \mathrm { i n t r a } } \mathcal { L } _ { \mathrm { i n t r a } } + \lambda _ { \mathrm { p h a s e } } \mathcal { L } _ { \mathrm { p h a s e } }$
8: Update �<sup>¯</sup> as an EMA of �
9: end for
10: Stage II, delay-aware cross-modal prediction, warm
started from Stage I
11: for each pretraining step do
12: With probability � sample a paired batch $\left( x _ { m } , x _ { n } \right)$ from
P, otherwise a unimodal batch
13: if the batch is paired then
14: Estimate the per token delay $\begin{array} { r l } { \tau _ { m  n } } & { { } = } \end{array}$
$\tau _ { \operatorname* { m a x } } \operatorname { t a n h } ( h _ { \delta } ( z _ { m } ) )$
15: Gather the aligned target with a Gaussian kernel
centered at $t + \tau$
16: Update � and � on $\lambda _ { \mathrm { c r o s s } } \mathcal { L } _ { \mathrm { c r o s s } } + \lambda _ { \mathrm { d e l a y } } \mathcal { L } _ { \mathrm { d e l a y - s u p } } +$
$\lambda _ { \mathrm { s t a t e } } \mathcal { L } _ { \mathrm { s t a t e } }$
17: else
18: Update � on $\lambda _ { \mathrm { i n t r a } } \mathcal { L } _ { \mathrm { i n t r a } } + \lambda _ { \mathrm { p h a s e } } \mathcal { L } _ { \mathrm { p h a s e } }$
19: end if
20: Update $\bar { \theta }$ as an EMA of �
21: end for
Ensure: Frozen shared encoder $g _ { \theta }$ for downstream linear
probing
```

## A.5 Sensitivity to Loss Weights

The full objective combines the intra-modal and cross-modal prediction losses with three auxiliary terms whose relative weights are $\lambda _ { \mathrm { c r o s s } } , \lambda _ { \mathrm { d e l a y } } , \mathrm { a n d } \lambda _ { \mathrm { s t a t e } }$ . To check that our results do not depend on a narrow choice of these weights, we vary each one around the value used by CardioState-JEPA while holding the architecture, the data, and all other weights fixed, and we re-evaluate the frozen encoder. Tables 7, 8, and 9 report the four aggregate metrics for the cross modal weight, the delay supervision weight, and the state weight respectively, with the CardioState-JEPA configuration $( \lambda _ { \mathrm { c r o s s } } = 1$ $\lambda _ { \mathrm { d e l a y } } = 1 , \lambda _ { \mathrm { s t a t e } } = 0 . 0 5 )$ shown as the reference in each table. ECG Avg is the mean macro-AUROC over the six ECG datasets at full labels, PPG Cls is the mean over the six PPG classification tasks, PPG Reg is the mean MAE over the eleven regression tasks, and PCG Avg is the mean over the two PCG tasks.

Across all three tables the default configuration used by CardioState-JEPA gives the best overall balance. It attains the strongest PPG classification score of 80.4 and the lowest PPG regression error of 9.1, and it matches the best ECG average of 90.9, while remaining within a few tenths of a point of the best PCG result. No perturbation improves more than one metric at a time, and the settings that raise PCG by a small margin, namely a larger state weight, do so at a clear cost to PPG classification. The ECG average is nearly constant across every setting, changing by less than one point as each weight is varied by a factor of two or more, which shows that the electrical representation is essentially insensitive to the exact loss weighting. PPG classification and regression are the most responsive metrics, and both are best at the default, which indicates that the chosen weights are well matched to the wearable tasks. PCG improves slightly as the state and delay weights increase, which is consistent with the role of these terms in pulling the acoustic modality toward the shared cardiac state, although the gain is small and does not outweigh the loss elsewhere. Taken together, these results show that CardioState-JEPA is robust to the exact loss weighting and that the default configuration is the best single choice across the three sensing modalities.

<table><tr><td>Dataset</td><td># Classes</td><td>Train</td><td>Valid</td><td>Test</td><td>Opt.</td><td>Epochs</td><td>BS</td><td>LR</td></tr><tr><td>ECG Pretraining</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>MIMIC-IV-ECG [13]</td><td>一</td><td>710,560</td><td>78,951</td><td></td><td>AdamW</td><td></td><td>196</td><td>1.2e-4</td></tr><tr><td>PPG Pretraining</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>PPG-EXT [29]</td><td>一</td><td>4,611,607</td><td>512,401</td><td></td><td>AdamW</td><td></td><td>196</td><td>1.2e-4</td></tr><tr><td>PCG Pretraining</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>BMD-HS [1]</td><td></td><td>3,436</td><td>382</td><td></td><td>AdamW</td><td></td><td>196</td><td>1.2e-4</td></tr><tr><td>ECG Downstream</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>PTB-XL Super [40]</td><td>5</td><td>17,084</td><td>2,146</td><td>2,158</td><td>AdamW</td><td>100</td><td>16</td><td>1e-3</td></tr><tr><td>PTB-XL Sub</td><td>23</td><td>17,084</td><td>2,146</td><td>2,158</td><td>AdamW</td><td>100</td><td>16</td><td>1e-3</td></tr><tr><td>PTB-XL Form</td><td>19</td><td>7,197</td><td>901</td><td>880</td><td>AdamW</td><td>100</td><td>16</td><td>1e-3</td></tr><tr><td>PTB-XL Rhythm</td><td>12</td><td>16,832</td><td>2,100</td><td>2,098</td><td>AdamW</td><td>100</td><td>16</td><td>1e-3</td></tr><tr><td>CPSC 2018 [27]</td><td>9</td><td>4,950</td><td>551</td><td>1,376</td><td>AdamW</td><td>100</td><td>16</td><td>1e-3</td></tr><tr><td>CSN [48]</td><td>38</td><td>16,546</td><td>1,860</td><td>4,620</td><td>AdamW</td><td>100</td><td>16</td><td>1e-3</td></tr><tr><td>PPG Downstream</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>WESAD</td><td>2</td><td>2,104</td><td>297</td><td>597</td><td>AdamW</td><td>100</td><td>16</td><td>1e-3</td></tr><tr><td>DaLiA</td><td>8</td><td>29,294</td><td>4,602</td><td>5,320</td><td>AdamW</td><td>100</td><td>16</td><td>1e-3</td></tr><tr><td>MIMIC AF</td><td>2</td><td>3,239</td><td>240</td><td>717</td><td>AdamW</td><td>100</td><td>16</td><td>1e-3</td></tr><tr><td>PPG Arrhythmia</td><td>2</td><td>36,820</td><td>4,764</td><td>5,243</td><td>AdamW</td><td>100</td><td>16</td><td>1e-3</td></tr><tr><td>BIDMC</td><td></td><td>9,412</td><td>1,652</td><td>1,398</td><td>AdamW</td><td>100</td><td>16</td><td>1e-3</td></tr><tr><td>UQVital EarSet</td><td></td><td>22,158</td><td>2,529</td><td>8,158</td><td>AdamW</td><td>100</td><td>16</td><td>1e-3</td></tr><tr><td></td><td></td><td>1,368</td><td>44</td><td>364</td><td>AdamW</td><td>100</td><td>16</td><td>1e-3</td></tr><tr><td>WildPPG</td><td>一</td><td>179,492</td><td>15,000</td><td>45,000</td><td>AdamW</td><td>100</td><td>16</td><td>1e-3</td></tr><tr><td>Sensors BP UCI BP</td><td>一</td><td>1,631 89,054</td><td>180 11,286</td><td>250</td><td>AdamW</td><td>100</td><td>16</td><td>1e-3</td></tr><tr><td>BCG BP</td><td>1</td><td>521</td><td>86</td><td>11,411 64</td><td>AdamW</td><td>100</td><td>16</td><td>1e-3</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>AdamW</td><td>100</td><td>16</td><td>1e-3</td></tr><tr><td>PCG Downstream</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>CirCor Murmur [32]</td><td>3</td><td>1,745</td><td>608</td><td>654</td><td>AdamW</td><td>100</td><td>16</td><td>1e-3</td></tr><tr><td>CinC [26]</td><td>2</td><td>2,006</td><td>627</td><td>607</td><td>AdamW</td><td>100</td><td>16</td><td>1e-3</td></tr></table>

Table 5: Training configurations for pretraining and downstream linear probing. Regression tasks are shown with a dash in the class column and are scored with mean absolute error, while classification tasks are scored with macro-AUROC.

## A.6 Representative Signal Examples

We group the representative waveforms by the role they play in CardioState-JEPA and present each group in its own figure, so that the reader can compare the modalities within a single training stage before comparing across stages.

Figure 4 shows the unimodal pretraining examples for ECG, PPG, and PCG. These recordings supply the abundant single modality data used in Stage I, where the model learns the structure of each modality on its own through masked latent prediction. Even within this group the three sensors look nothing alike, since each one renders the cardiac cycle through a diferent physical process, yet all of them carry the same rhythm and timing information that the shared encoder is trained to recover.

Figure 5 shows the synchronized paired and trimodal recordings used in Stage II. These co-recorded segments are far scarcer than the unimodal data, and they are the only source that ties the modalities together in time. The figure makes clear why alignment by raw timestamp is insuficient, since the same beat appears at a visibly diferent position in each sensor, which is exactly the ofset that the learned delay aligner is designed to correct.

Figure 6 shows the downstream evaluation examples drawn from the benchmarks listed in Table 5. These segments are never seen during pretraining and are used only to probe the frozen encoder. Although they difer from the pretraining data in morphology, sampling rate, and noise pattern, they describe the same underlying cardiac process, which is the property that lets a single shared representation transfer across all three sensing modalities.

## A.7 Attention Maps on Cardiac Signals

Figure 7 visualizes attention maps from CardioState-JEPA on representative ECG, PPG, and PCG signals. The examples show that the shared encoder concentrates on physiologically meaningful regions in every modality, namely sharp depolarization complexes in ECG, the pulse upstroke and systolic peak in PPG, and localized acoustic bursts in PCG. Because a single set of parameters produces all three maps, the figure also shows that the encoder has learned to recognize the modality specific signature of the same cardiac events rather than a single fixed waveform shape. This behavior matches the design goal of the model, which is to describe the shared cardiac cycle while remaining sensitive to how each sensor renders it.

<table><tr><td></td><td colspan="5">PTB-XL Super ↑</td><td colspan="5">PTB-XL Sub ↑</td><td colspan="5">PTB-XL Form ↑</td><td colspan="5">PTB-XL Rhythm ↑</td></tr><tr><td>Method</td><td>2.5s</td><td>5s</td><td>10s</td><td>Mean</td><td>Std</td><td>2.5s</td><td>5s</td><td>10s</td><td>Mean</td><td>Std</td><td>2.5s</td><td>5s</td><td>10s</td><td>Mean</td><td>Std</td><td>2.5s</td><td>5s</td><td>10s</td><td>Mean</td><td>Std</td></tr><tr><td>ST-MEM</td><td>65.3</td><td>69.5</td><td>69.2</td><td>68.0</td><td>1.9</td><td>58.9</td><td>60.7</td><td>60.0</td><td>59.9</td><td>0.7</td><td>52.4</td><td>58.1</td><td>53.5</td><td>54.7</td><td>2.5</td><td>62.6</td><td>58.7</td><td>62.2</td><td>61.2</td><td>1.8</td></tr><tr><td>ECG-FM</td><td>80.5</td><td>81.9</td><td>86.5</td><td>83.0</td><td>2.6</td><td>73.3</td><td>72.2</td><td>75.1</td><td>73.5</td><td>1.2</td><td>62.2</td><td>67.8</td><td>76.0</td><td>68.7</td><td>5.7</td><td>72.3</td><td>77.0</td><td>82.0</td><td>77.1</td><td>4.0</td></tr><tr><td>ECGFounder</td><td>83.3</td><td>86.5</td><td>87.0</td><td>85.6</td><td>1.6</td><td>74.8</td><td>76.0</td><td>76.2</td><td>75.7</td><td>0.6</td><td>62.6</td><td>68.2</td><td>69.3</td><td>66.7</td><td>2.9</td><td>73.5</td><td>81.7</td><td>90.2</td><td>81.8</td><td>6.8</td></tr><tr><td>CardioState-JEPA</td><td>85.2</td><td>85.6</td><td>86.4</td><td>85.7</td><td>0.5</td><td>78.4</td><td>78.7</td><td>79.1</td><td>78.7</td><td>0.3</td><td>65.6</td><td>68.3</td><td>70.0</td><td>68.0</td><td>1.8</td><td>87.0</td><td>87.3</td><td>92.4</td><td>88.9</td><td>2.5</td></tr></table>

Table 6: Input length ablation on ECG tasks (macro-AUROC ×100, ↑, 10% training data). Bold marks the best value in each column. Mean and Std are the mean and standard deviation across the three input lengths.
<table><tr><td>Variant</td><td>ECG Avg ↑</td><td>PPG Cls ↑</td><td>PPG Reg ↓</td><td>PCG Avg ↑</td></tr><tr><td> $\mathbf { C a r d i o S t a t e - J E P A } \ ( \lambda _ { \mathrm { c r o s s } } = 1 . 0 )$ </td><td>90.9</td><td>80.4</td><td>9.1</td><td>82.3</td></tr><tr><td> $\lambda _ { \mathrm { c r o s s } } = 0 . 5$ </td><td>90.9</td><td>78.5</td><td>10.34</td><td>79.4</td></tr><tr><td> $\lambda _ { \mathrm { c r o s s } } = 2 . 0$ </td><td>90.6</td><td>77.3</td><td>10.54</td><td>80.9</td></tr></table>

Table 7: Sensitivity to the cross-modal loss weight $\lambda _ { \mathrm { c r o s s } } .$ The CardioState-JEPA row uses the default weighting. Values other than MAE are macro- $\mathbf { A U R O C } \times 1 0 0$ , where higher is better, and lower is better for MAE.

![](images/e6a256c34213166e17ef10e20385c77245fd3a7a1c948aed2919aa22e7fdd59f.jpg)  
(a) ECG

![](images/cdc5f07a3c893bbff9433997f507f63a8197bbb6f1734a284dcf113fac3873ea.jpg)  
(a) ECG and PPG

![](images/bb646cfbd42574ecc80069114df67897e3e9da87f3b3553681a6a15927637bfd.jpg)

![](images/c8adaea3f4745e3758004f259eb79fc48068e0e2d26420f61a07a1d34a4da8cc.jpg)  
(b) ECG and PCG

(b) PPG  
![](images/257cc265762b47087e4890073487c6bdf9a3a364822de698625a1d14c86a2d21.jpg)  
(c) PCG  
Figure 4: Unimodal pretraining signals used in Stage I, shown one modality per row. ECG, PPG, and PCG recordings observe the same cardiac cycle through electrical, hemodynamic, and acoustic sensing, yet difer in morphology, sampling rate, and noise pattern.

![](images/5183d069c0b77523ea6c189888102d6a996bf1110fe865058241d1974b94b8ab.jpg)  
(c) Trimodal  
Figure 5: Synchronized paired and trimodal recordings used for Stage II delay aware cross modal alignment, shown one pairing per row. The same beat appears at a diferent time in each sensor, which motivates the learned physiological delay.

<table><tr><td>Variant</td><td>ECG Avg ↑</td><td>PPG Cls ↑</td><td>PPG Reg ↓</td><td>PCG Avg ↑</td></tr><tr><td>CardioState-JEPA  $( \lambda _ { \mathrm { d e l a y } } = 1 . 0 )$ </td><td>90.9</td><td>80.4</td><td>9.1</td><td>82.3</td></tr><tr><td> $\lambda _ { \mathrm { d e l a y } } = 0 . 5$ </td><td>90.8</td><td>78.7</td><td>9.68</td><td>79.3</td></tr><tr><td> $\lambda _ { \mathrm { d e l a y } } = 2 . 0$ </td><td>90.8</td><td>78.0</td><td>10.20</td><td>82.5</td></tr></table>

Table 8: Sensitivity to the delay supervision loss weight $\lambda _ { \mathrm { d e l a y } }$ . The CardioState-JEPA row uses the default weighting. Values other than MAE are macro-AUROC ×100, where higher is better, and lower is better for MAE.
<table><tr><td>Variant</td><td>ECG Avg ↑</td><td>PPG Cls ↑</td><td>PPG Reg ↓</td><td>PCG Avg ↑</td></tr><tr><td> $\mathbf { C a r d i o S t a t e - J E P A } \ ( \lambda _ { \mathrm { s t a t e } } = 0 . 0 5 )$ </td><td>90.9</td><td>80.4</td><td>9.1</td><td>82.3</td></tr><tr><td> $\lambda _ { \mathrm { s t a t e } } = 0 . 2 5$ </td><td>90.7</td><td>74.5</td><td>10.20</td><td>82.7</td></tr><tr><td> $\lambda _ { \mathrm { s t a t e } } = 1 . 0$ </td><td>90.2</td><td>75.8</td><td>9.65</td><td>82.7</td></tr></table>

Table 9: Sensitivity to the state loss weight $\lambda _ { \mathrm { s t a t e } }$ . The CardioState-JEPA row uses the default weighting. Values other than MAE are macro-AUROC ×100, where higher is better, and lower is better for MAE.

![](images/a328c50f3b8c0874c7b4a00af0aad9c6f7874e165db2a0112e175a562b3112d8.jpg)  
(a) ECG attention map

![](images/5436423f54e684b59012ee66f41f745ee56d108e879dc22f2372b8df84e334da.jpg)  
(a) ECG

![](images/fd08ecc0ca82e28209c9b2f0ab2bb9cbb955039e4965df97080377005727746c.jpg)  
(b) PPG

![](images/6380d8efd5be92eee804041c2057f7e71c0421f91c30c73e47dc76e28f9c3354.jpg)  
(c) PCG  
Figure 6: Downstream evaluation signals, shown one modality per row. These held out ECG, PPG, and PCG segments probe the frozen encoder and are never seen during pretraining.

![](images/be2ceab002fcdca4777db226eae4331ca564f17041ade939371f60132c54d693.jpg)  
(b) PPG attention map

![](images/4e7205c666a3f8ac420fc58a3279207eb91eee8a123c17d0905e0abe08a68167.jpg)  
(c) PCG attention map  
Figure 7: Attention visualizations from CardioState-JEPA on representative ECG, PPG, and PCG signals. The ECG map highlights sharp electrical events such as QRS complexes, the PPG map emphasizes pulse morphology including the upstroke and systolic peak, and the PCG map focuses on localized acoustic heart-sound events. These patterns indicate that the shared encoder attends to modality specific manifestations of the cardiac cycle while learning a common cardiac representation.

## A.8 Delay Alignment Visualization

To check that the delay aligner captures physiological timing rather than an arbitrary ofset, we visualize its output on paired recordings. For each R-peak $t _ { R }$ detected in the ECG, we mark the predicted event time $t _ { R } + \hat { \tau }$ on the target signal and compare it with the true cardiac event in that modality. As shown in Figure 8, the predicted time lines up with the systolic upstroke in the PPG and with the first heart sound $S _ { 1 }$ in the PCG, and it does so consistently from one beat to the next. This close agreement indicates that the learned delay reflects the true electromechanical and pulse transit timing between modalities rather than a value that merely minimizes the training loss. It also gives a direct and interpretable check on the central mechanism of CardioState-JEPA, since the alignment can be read against known cardiac landmarks.

![](images/5164bdae6785ebb1d1e17fe90c8bcae5bfa6928e2f31df507cbfd61c98ae8f54.jpg)  
Figure 8: Learned delay alignment on paired recordings. The top panel shows ECG to PPG alignment on VitalDB and the bottom panel shows ECG to PCG alignment on EPHNOGRAM. For each ECG R-peak $t _ { R } ,$ , the predicted event time $t _ { R } + \hat { \tau }$ (red dashed) is overlaid on the target signal alongside the actual event (green), namely the systolic upstroke for PPG and the first heart sound $S _ { 1 }$ for PCG. The learned delay tracks the physiological ofset between modalities.

## A.9 Task-Relevant Structure After Cross-Modal Pretraining

The shared space analysis showed that cross-modal training makes the pooled cardiac code invariant to the sensing modality, since the modality silhouette falls from 0.121 to −0.006. Here we show a complementary efect within a single modality, namely that the same training makes the PPG representation more discriminative for downstream classes. In both of the following figures we compare PPG features from the Stage I and Stage II checkpoints, so the comparison isolates the contribution of cross-modal pretraining rather than comparing against baselines.

Figure 9 shows the binary atrial fibrillation task. After Stage I the two classes overlap, and after Stage II they form clearer clusters, with the class silhouette rising from 0.03 to 0.12. This shows that pulling the modalities into a shared space does not blur the class structure that a downstream classifier needs, but instead makes it easier to separate.

Figure 10 shows the six-class arrhythmia task, where the efect is stronger. Stage I features are largely intermixed with a silhouette of 0.01, whereas after Stage II the classes organize into coherent groups with a silhouette of 0.09. Because absolute silhouette values are small for t-SNE embeddings of high dimensional features, we read these numbers as relative improvements rather than measures of separability. The modality analysis and the class analysis are computed over diferent labelings, so cross-modal training is expected to lower the modality silhouette while raising the class silhouette, and together the two figures indicate that the training removes sensor identity from the shared space while sharpening task relevant structure within each modality.

![](images/8493f4448bba474ec8213c08452d5db089e2e581dfea37410445b66f0b0a32e2.jpg)  
Figure 9: t-SNE of PPG features on the binary atrial fibrillation task (MIMICPerform-AF) from the Stage I (left) and Stage II (right) checkpoints, colored by class. The class silhouette rises from 0.03 to 0.12.

![](images/3d879beaad9ac729aa8737fb4e7785e2e85b1b68b5c48068fb2d870b3817ddab.jpg)  
Figure 10: t-SNE of PPG features on the six-class arrhythmia task from the Stage I (left) and Stage II (right) checkpoints, colored by class. The class silhouette rises from 0.01 to 0.09.

## A.10 Limitations

CardioState-JEPA has several limitations that also point to future work. First, the paired and trimodal corpora that drive cross modal alignment are much smaller than the unimodal corpora, so the amount of synchronized supervision is limited and the delay aligner is trained on comparatively few clean beats. Second, the PCG pretraining data is the smallest of the three modalities, which places more weight on cross modal transfer for acoustic tasks and may limit how far the acoustic representation can be pushed on its own. Third, on ECG the trimodal model is within noise of the strongest bimodal variant, so the benefit of adding a third modality is clearest for PPG and PCG rather than for ECG. Fourth, the delay aligner assumes that a reliable reference event can be detected on the source signal, so very noisy recordings where beats cannot be located fall back to unsupervised alignment. Finally, all results use a frozen encoder with linear probing, and we leave full fine tuning and larger encoder sizes to future work. We view these limitations as natural next steps rather than obstacles, since each one is tied to data availability or evaluation budget rather than to the core formulation.