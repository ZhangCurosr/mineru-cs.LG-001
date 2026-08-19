# Diff-DDoS: Realistic Cyber-Physical Attack Synthesis and Robust Detection for 5G-Enabled CPS Using Tabular Diffusion Models

Bilal Hussain , Senior Member, IEEE, Xiao Tang , Member, IEEE, Qinghe Du , Member, IEEE, Tan Li , Member, IEEE, Muhammad Azhar , Member, IEEE, and Danista Khan , Member, IEEE

Abstract— Deep learning-based DDoS detectors for 5Genabled cyber-physical systems face two challenges: scarce real-world labeled attack data and the unrealism of naive synthetic substitutes, which limit robustness against adaptive adversaries. Detectors trained on hand-crafted attacks with fixed scaling multipliers degrade catastrophically—F1-score drops of ∼47% to 100% depending on scenario—when confronted with realistic, distribution-preserving samples. We propose Diff-DDoS, a three-phase framework for realistic attack synthesis and robust detection using tabular diffusion models. Phase 1 establishes a baseline CNN cell-level detector on spatiotemporal grids derived from call detail records (CDRs). Phase 2 trains a tabular denoising diffusion probabilistic model (TabDDPM) on normal CDR aggregates to generate realistic attacks, exposing detector vulnerabilities. Phase 3 introduces adversarial diffusion training (ADT), which applies inverse classifier guidance to iteratively generate hard yet distribution-preserving adversarial samples until the detector converges. On a real-world Milano CDR dataset across SMS-flooding, silent-call, Internet-signaling, and blended scenarios, ResNet50 with ADT recovers F1-scores

of 79.62% (silent-call), 100% (Internet), and 92.79% (blended) while retaining near-perfect multiplier-test F1 on most scenarios. We further benchmark ADT against gradient-based adversarial training, conditional tabular GAN (CTGAN), and fixed-multiplier baselines under identical iterative training and validation-based threshold calibration; after calibration, ADT reaches 100% SMS F1 versus 47.3% for CTGAN and attains silent-call F1 on par with the strongest gradientbased adversarial-training baseline. SMS flooding remains challenging for lightweight architectures, though calibration enables near-perfect ResNet50 detection. These results establish tabular diffusion models as a practical tool for stress-testing and hardening intrusion detectors in datascarce 5G cyber-physical deployments.

Index Terms— 5G networks, adversarial training, cyberphysical systems, DDoS attacks, diffusion models.

## I. INTRODUCTION

(CPSs), including industrial monitoring, transportation, and energy services. This convergence enlarges the attack surface and creates new pathways for distributed denial-of-service (DDoS) attacks that threaten both network availability and physical processes [1]–[4], requiring multi-layer AI-enabled protection [5], [6]. Recent threat-landscape reports rank DDoS attacks among the most prominent cybersecurity threats in mobile and Internet of Things (IoT)-connected environments [7].

Detecting stealthy cellular attacks is particularly difficult. Silent-call abuse, SMS flooding, and Internet-signaling (signaling storm) attacks stress control-plane procedures rather than only consuming user-plane bandwidth [8]–[11], and signaling overload and layer-3 abuse remain credible threats in modern cellular architectures [10], [11].

This mismatch exposes a broader weakness: models trained on stylized attacks degrade sharply under distribution

At the monitoring layer, call detail record (CDR) aggregates capture cell-level temporal deviations in SMS, call, and Internet activity. Hussain et al. [4] showed they can be organized as $9 \times 9 \times 3$ spatiotemporal grids and classified by convolutional neural networks (CNNs) with strong performance on multiplier-based synthetic attacks. However, those multiplier attacks use fixed, large feature multipliers and are therefore easier to separate from normal traffic than realistic adversarial behavior.

shift [12], [13]. In cellular settings the problem is amplified by scarce, privacy-sensitive attack labels; naive perturbations can break traffic semantics [2], [4], [12], [13], leaving detectors brittle against attacks that stay near the normal traffic manifold.

To address this gap, we propose Diff-DDoS, a three-phase framework using tabular diffusion models to synthesize realistic attack samples and harden detectors against adaptive evasion. In Phase 1, we re-implement the CNN-based detector of [4] as a multiplier-based baseline (full pseudo-code in Supplementary Sec. S-A). In Phase 2, we train TabDDPM only on normal CDR aggregates to generate manifold-preserving attack samples, revealing the baseline detector’s poor generalization. In Phase 3, we introduce adversarial diffusion training (ADT), which uses guided diffusion sampling to generate hard on-manifold attacks and iteratively retrains the detector.

The main contributions of this work are as follows:

1) We re-implement the cell-level CNN baseline of [4] and establish reference performance on multiplier-based attacks.

2) We demonstrate that detectors trained only on multiplierbased attacks can fail severely when evaluated on diffusion-generated attacks drawn from the normal CDR manifold.

3) We introduce ADT, which combines TabDDPM-based generation with inverse classifier guidance to create hard, realistic attack samples for iterative robust training.

4) We evaluate Diff-DDoS across SMS, silent-call, Internet, and blended scenarios using SimpleCNN and 50- layer residual network (ResNet50), highlighting ADT robustness gains, validation-based threshold calibration, and a per-scenario learnability analysis showing that SMS flooding is capacity-limited on SimpleCNN but recoverable on ResNet50.

Sec. II covers related work; Sec. III background and problem setup; Sec. IV methodology; Sec. V experimental setup and evaluation protocols; Sec. VI results; Sec. VII limitations and conclusion. Extended derivations and full result grids are in the Supplementary Material.

## II. RELATED WORK

DDoS detection in 5G-enabled CPSs has advanced from rule-based to machine learning/deep learning (ML/DL) methods because signaling abuse, SMS flooding, and silent-call attacks can evade threshold defenses [2], [14]. Hussain et al. [4] showed CDR grids enable strong CNN performance on multiplier-based attacks; later work explored recurrent, federated, and quantum-inspired models [15]–[17].

CDR analytics also demonstrate useful spatiotemporal structure for traffic prediction and network monitoring [18]–[21]. Yet both DDoS detection and CDR analytics literatures rarely evaluate robustness to adaptive evasion close to normal traffic statistics [4], [15], [22], [23].

Deep intrusion detection systems (IDSs) degrade under distribution shift and adversarial manipulation [24]–[26]. Conditional tabular generative adversarial network (CTGAN) improves data augmentation but suffers from instability [27];

TabDDPM offers richer tabular distributions and higherfidelity synthesis [28], [29]. Gradient-based adversarial perturbations generate off-manifold samples [12], [13], [30], [31], while diffusion-based approaches show growing promise [32]– [34].

## III. BACKGROUND AND PROBLEM SETUP

## A. CDR Grid Representation and Detection Task

Each CDR snapshot is represented as a grid image $x \in$ $\mathbb { R } ^ { 9 \times 9 \times 3 }$ , where the three channels correspond to aggregated SMS-out, Call-out, and Internet traffic for each of the $9 \times 9 =$ 81 cells in a selected Milano [35] subgrid over one 10-minute interval within the 11:00–14:00 window. The detection task is formulated as a multilabel cell-level classification problem: for each grid image $x ,$ a CNN-based detector M with learnable weights θ outputs a vector

$$
\mathcal { M } ( x ; \theta ) = \hat { Y } \in ( 0 , 1 ) ^ { 8 1 } ,
$$

where each component $\hat { y } _ { k } \in ( 0 , 1 )$ is the sigmoid-activated predicted attack probability for cell $k \in \{ 1 , \ldots , 8 1 \}$ . The ground-truth label $Y \in \{ 0 , 1 \} ^ { 8 1 }$ assigns $Y _ { k } ~ = ~ 1$ if cell k is under attack and $Y _ { k } ~ = ~ 0$ otherwise. During adversarial diffusion training (Phase 3), θ is updated iteratively across rounds, so the detector after ADT round $r$ for attack scenario s is denoted $\mathcal { M } ( \cdot ; \theta _ { s } ^ { r } )$ (during round-r sampling, guidance uses $\theta _ { s } ^ { r - 1 } ;$ Sec. IV-D). We write $f _ { \boldsymbol { \theta } } \equiv \mathcal { M } ( \cdot ; \boldsymbol { \theta } )$ in algorithms when shorthand is needed.

The CDR-based attack synthesis of [4] applies scenariospecific multipliers to $| { \mathcal { C } } | = 4 0$ randomly selected cells, labeling them $Y _ { k } = 1$ (see Supplementary Algorithm S2). Four scenarios are evaluated throughout: SMS flooding (elevated SMSout), silent-call (suppressed outgoing calls), Internet-signaling (elevated Internet, approximating a signaling storm [4]), and blended (simultaneous SMS/Internet elevation and call suppression). Phase 1 uses fixed multipliers (169.5× SMS, −33.1× call clipped at zero, 43.3× Internet); these stylized attacks motivate the diffusion-based robustness evaluation and are not used to generate attacks in Phases 2–3 (Phase 2 test attacks are TabDDPM-generated; see Sec. IV-C). The preprocessing pipeline (62 CDR files, 11:00–14:00 aggregation, Num2Dims(·) row-major cell indexing) is given in Supplementary Algorithms S1–S2.

## B. Denoising Diffusion and Classifier Guidance

We use standard denoising diffusion probabilistic model (DDPM) notation [28], [36]: a fixed forward schedule corrupts clean data $x _ { 0 }$ into noisy samples $x _ { t }$ over $T$ steps; a learned reverse denoiser $\epsilon _ { \psi }$ (weights ψ, distinct from detector θ) predicts the injected noise. Our TabDDPM instantiation on 3D CDR aggregates is summarized in Sec. IV-C.1; full equations are in Supplementary Sec. S-B. DDPMs have been extended from images [37] to tabular data [28]; Diff-DDoS builds on the latter.

TABLE I: Main-text notation summary.
<table><tr><td>Symbol</td><td>Meaning</td></tr><tr><td> $\mathcal { M } ( \cdot ; \theta ) , f _ { \theta }$ </td><td>Multilabel CNN detector (81 sigmoid outputs per grid)</td></tr><tr><td> $\Xi _ { s } ^ { r } , \Xi _ { s } ^ { r , \mathrm { o r i g } } , \Xi _ { s } ^ { r , \mathrm { T a b } }$ </td><td>Metric bundles (Acc, Prec, Rec, F1, FPR) for attack scenario s at ADT round r; not the detector</td></tr><tr><td> $\theta , \theta _ { s } ^ { * } , \theta _ { s } ^ { r }$ </td><td>Detector learnable weights</td></tr><tr><td>∈ψ, φTabDDPM</td><td>TabDDPM denoiser network (weights ψ; distinct from θ)</td></tr><tr><td> $\phi _ { s } : \mathbb { R } ^ { 3 }  \mathbb { R } ^ { 3 }$ </td><td>Mild post-sample scenario scaling on 3D CDR aggregates (Phase 2)</td></tr><tr><td></td><td>MapToGrid(x, B) Copy background B, write aggregate  into attacked cells C, set  $Y _ { k } = 1 \mathrm { o n } \mathcal { C }$ </td></tr><tr><td>pattack</td><td>Mean detector score over attacked cells C during inverse guidance</td></tr></table>

Classifier guidance [37] steers the reverse process via classifier gradients:

$$
\begin{array} { r } { \nabla _ { x _ { t } } \log p ( x _ { t } \mid y ) = \underbrace { \nabla _ { x _ { t } } \log p ( x _ { t } ) } _ { \mathrm { u n c o n d i t i o n a l ~ s c o r e } } + s _ { \mathrm { c g } } \cdot \underbrace { \nabla _ { x _ { t } } \log p ( y \mid x _ { t } ) } _ { \mathrm { c l a s s i f i e r ~ g r a d i e n t } } , } \end{array}\tag{1}
$$

where $\nabla _ { x _ { t } }$ is the gradient with respect to the noisy sample $x _ { t } ,$ $s _ { \mathrm { c g } }$ controls conditioning strength, and the classifier gradient biases samples toward target class y. Diff-DDoS extends this idea via inverse guidance in Phase 3 (Sec. IV-D), steering samples away from attack-like detector scores while preserving the normal CDR manifold [33], [34].

## C. Notation Summary

Table I lists overloaded symbols used in the main text; the supplement provides an extended disambiguation table.

## IV. METHODOLOGY

Fig. 1 summarizes the Diff-DDoS pipeline: Phase 1 establishes a multiplier-trained CNN baseline; Phase 2 trains Tab-DDPM on normal 3D CDR aggregates and exposes baseline vulnerability to manifold-preserving attacks; Phase 3 applies adversarial diffusion training (ADT) with inverse classifier guidance to iteratively harden the detector. Offline attack synthesis is separated from online grid-level inference (Sec. VI-F).

## A. Data Representations and Attack-Steering Mechanisms

Normal tabular training rows are R<sub>norm</sub> = {(SMS sum, CALL sum, INTERNET sum)} extracted from raw CDR aggregates over the $9 \times 9$ subgrid; they are standardized with StandardScaler fit on ${ \mathcal { R } } _ { \mathrm { n o r m } }$ only. Num2Dims(k) maps cell index k to grid coordinates $( h , w )$ in row-major order. MapToGrid(x˜, B) copies background grid B, writes aggregate x˜ into attacked cells C via Num2Dims(·), and sets $Y _ { k } ~ = ~ 1$ on C. Phase 2 test grids use $B \in { \mathcal { A } }$ (abnormal-template pool); Phase 3 ADT training grids use a zero background $( B = { \bf 0 } ) \mathrm { { } }$ : only cells in C receive x˜; all others remain zero (Algorithm 2). Table II contrasts the three attack-steering mechanisms used across phases.

TABLE II: Attack-steering mechanisms across Diff-DDoS phases.
<table><tr><td>Mechanism</td><td>Phase</td><td>Action</td><td>Purpose</td></tr><tr><td>Fixed multipliers 1</td><td></td><td>169.5× -33.1× Call, 43.3× Internet</td><td>SMS, Stylized baseline</td></tr><tr><td>Heuristic 十</td><td>2</td><td>Mild γ ranges on tar- On-manifold</td><td></td></tr><tr><td>steering  $\phi _ { s }$ </td><td></td><td>get channel(s)</td><td>vulnerability test</td></tr><tr><td>Inverse classifier 3 guidance</td><td></td><td>Minimize Pattack to- Hard robust-training ward  $p _ { \mathrm { t a r g e t } }$ </td><td>samples</td></tr></table>

## B. Phase 1: Baseline Detector Training

The detector is a multilabel model with 81 sigmoid outputs per grid (Sec. III-A); Phase 1 trains on the union of normal grids (all-negative labels) and multiplier-attack grids (40 perturbed cells positive), without any TabDDPM-generated attacks. Phase 1 follows [4]; the full training procedure is in Supplementary Algorithm S3.

Data Preprocessing: Phase 1 uses the two-stage pipeline of Supplementary Algorithms S1–S2 (grid extraction and fixedmultiplier attack injection on $| \mathcal { C } | ~ = ~ 4 0 ~ $ cells; Table II). Attack labels are defined solely by synthesis; no statistical thresholding is applied.

Model Training: Two detector architectures are employed:

• SimpleCNN: A lightweight CNN with three convolution–pooling blocks followed by fully-connected layers.

• ResNet50: A deeper (50-layer) residual network adapted to $9 \times 9 \times 3$ inputs with an 81-way sigmoid output layer.

Both models are trained with a multilabel binary crossentropy (BCE) loss averaged over N training images and all 81 cells:

$$
\begin{array} { r l } {  { \mathcal { L } _ { \mathrm { B C E } } ( \theta ) = - \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \frac { 1 } { 8 1 } \sum _ { k = 1 } ^ { 8 1 } \Bigl [ y _ { i , k } \log \hat { y } _ { i , k } } } \\ & { \qquad + ( 1 - y _ { i , k } ) \log \bigl ( 1 - \hat { y } _ { i , k } \bigr ) \Bigr ] , } \end{array}\tag{2}
$$

where $y _ { i , k } ~ \in ~ \{ 0 , 1 \}$ is the ground-truth label and $\begin{array} { r l } { \hat { y } _ { i , k } } & { { } = } \end{array}$ $\mathcal { M } ( x _ { i } ; \theta ) _ { k } \in ( 0 , 1 )$ is the sigmoid-activated predicted attack probability for cell k in image i, produced by detector M with learnable weights θ.

Predictions are binarized at $\tau = 0 . 5 ;$ true positives, false positives, and false negatives (TP/FP/FN) are micro-averaged over $N _ { \mathrm { t e s t } } \times 8 1$ cell decisions to compute Accuracy, Precision, Recall, F1 score, False Positive Rate (FPR), and receiver operating characteristic area under the curve (ROC–AUC) (Sec. VI-G; dataset splits in Sec. V).

## C. Phase 2: TabDDPM-Based Vulnerability Demonstration

Phase 2 demonstrates that multiplier-trained detectors are vulnerable to realistic diffusion-based attacks. Let ${ \mathcal { X } } _ { n }$ denote the 781 normal-only grid images (614 train + 167 test) and A the 335 unperturbed abnormal-template grids produced by Supplementary Algorithms S1–S2 (fixed random seed 1; Sec. V). Let $\mathcal { X } _ { a } ^ { s }$ denote the scenario-specific abnormal grid working pool $( N _ { \mathrm { a b n } } ~ = ~ 3 3 5 )$ used in Phase 2 indexing.

Phase 1: Baseline Detector Training  
![](images/70d6d078e41793c0294dd6d49cc8a73d93085850d8c6fd725d75cbc229886614.jpg)  
Fig. 1: Overview of the proposed Diff-DDoS framework. Phase 1 trains baseline detectors on multiplier-based attacks, Phase 2 uses TabDDPM to generate realistic attacks from normal CDR aggregates, and Phase 3 applies ADT to generate hard onmanifold samples for iterative retraining.

Training attacks in $\mathcal { X } _ { a } ^ { s }$ (indices 0–166) remain multiplierbased, while test attacks (indices 167–334, $N _ { \mathrm { a t t a c k } } ^ { \mathrm { t e s t } } = 1 6 8 )$ are replaced with TabDDPM-generated grids. Let $\bar { \mathcal { X } } _ { a } ^ { s , \mathrm { T a b } }$ denote the TabDDPM-based test set: the 335-image split combining 167 held-out normal test grids with 168 TabDDPM test-attack grids X<sup>s</sup><sub>a</sub> [167:335] (i.e., $\bar { \mathcal { D } } _ { \mathrm { t e s t } } ^ { ( s ) }$ in Algorithm 1). Phase-2 F1 is $\mathrm { F 1 } _ { \mathrm { P h . 2 } } ^ { ( s ) }$ on this split with $\tau = 0 . 5$ (protocol P1; Table III); degradation is $\Delta _ { \mathrm { F 1 } } ^ { \boldsymbol { ( s ) } } = ( \mathrm { F 1 } _ { \mathrm { P h . 1 } } ^ { ( s ) } - \mathrm { F 1 } _ { \mathrm { P h . 2 } } ^ { ( \bar { s } ) } ) / \mathrm { F 1 } _ { \mathrm { P h . 1 } } ^ { ( s ) } \times 1 0 0 \%$

1) Tabular Diffusion Model for Aggregated CDRs: Attack synthesis operates on 3D CDR aggregates (SMS, CALL, INTERNET) per attacked-cell group, where scenario constraints are naturally expressed and realism is quantified (Sec. VI-C). A grid-image DDPM would couple unrelated cells and complicate per-channel control; we therefore model the low-dimensional aggregate manifold with TabDDPM [28] and map sampled aggregates back to $9 \times 9 \times 3$ grids via MapToGrid(·, B) (Sec. IV-A).

We adapt the DDPM framework [36] to tabular CDR features. Raw CDR rows are aggregated into a 3D feature vector $x _ { 0 } = \mathrm { ( S M S }$ sum, CALL sum, INTERNET sum) $\in \mathbb { R } ^ { 3 }$ per (cell, timeslot, day), which defines the normal CDR manifold. A fixed linear noise schedule $( T = 1 0 0 0 , \beta _ { \mathrm { m i n } } = 1 0 ^ { - 4 }$ $\beta _ { \mathrm { m a x } } = 0 . 0 2 )$ governs the forward diffusion, and the denoiser $\epsilon _ { \psi } - \mathrm { a }$ multilayer perceptron (MLP) with hidden dimensions [256, 512, 256] that takes the concatenation of $x _ { t }$ and the scalar diffusion step t (broadcast to the feature dimension)— is trained with the standard $\ell _ { 2 }$ noise-prediction objective ${ \mathcal { L } } _ { \mathrm { D D P M } } ( \psi )$ . The full forward/reverse equations, denoiser architecture, training objective, and posterior sampling are given in Supplementary Sec. S-B.

2) Realistic Attack Scenario Generation: TabDDPM is trained exclusively on normal aggregate CDR rows. Here $\textstyle { \mathcal { U } } ( a , b )$ denotes the continuous uniform distribution on $[ a , b ] ,$ and $\gamma _ { 1 } , \gamma _ { 2 } , \gamma _ { 3 }$ are independent per-sample scaling factors. To generate realistic attacks for a given scenario s, we:

1) Run reverse diffusion (Supplementary Sec. S-B) to obtain an aggregate $\mathbf { x } ^ { ( i ) } = ( \mathrm { S M S } , \mathrm { C A L L } , \mathrm { I N T E R N E T } ) \in$ $\mathbb { R } ^ { 3 }$ , applying lightweight heuristic steering on the target channel(s) every max(1, ⌊T /10⌋) timesteps (Table II; distinct from Phase 3 inverse classifier guidance and from post-sample $\phi _ { s }$ in item 2).

2) Apply mild scenario-specific constraints $\tilde { \mathbf { x } } ^ { ( i ) } $ $\phi _ { s } ( \mathbf { x } ^ { ( i ) } )$ on the target channel only: SMS, $\gamma _ { 1 } \sim \mathcal { U } ( 2 , 5 )$ silent-call, $\gamma _ { 2 } \sim \mathcal { U } ( 0 . 3 , 0 . 7 )$ with CALL ← max $\cdot ( 0 , \gamma _ { 2 } \cdot$ CALL); Internet, $\gamma _ { 3 } ~ \sim ~ \mathcal { U } ( 2 , 5 )$ ; Blended, all three applied jointly. Non-target channels are left unchanged (silent-call models resource-exhaustion via call suppression [8], [38]).

3) Map each constrained triple to a $9 \times 9 \times 3$ grid via MapToGrid(x˜<sup>(i)</sup>, B<sup>(i)</sup>) with $B ^ { ( i ) } \in { \mathcal { A } }$

The Phase 1 detectors are then evaluated on the full test split without retraining (results in Sec. VI-B).

## D. Phase 3: Adversarial Diffusion Training (ADT)

Phase 3 applies ADT: for each $\begin{array} { r l r l r l } { s } & { { } } & { \in } & { { } { \mathcal { S } } } & { } & { { } = } \end{array}$ {SMS, Call, Internet, Blended} and round $r \ = \ 1 , \ldots , R ,$ inverse-guided TabDDPM samples are appended to the training set and the detector is fine-tuned (Algorithm 2). Let $\mathcal { X } _ { \mathrm { a d v } } ^ { r }$ denote the adversarial grids generated in round r and ϕ<sub>TabDDPM</sub> the Phase 2 TabDDPM denoiser (Supplementary Sec. S-B).

$$
( { \mathcal { D } } _ { \mathrm { n o r m } } ,
$$

$$
\mathcal { M } ( \cdot ; \theta )
$$

$$
f _ { \theta } )
$$

$$
\{ f _ { \theta _ { s } ^ { * } } \} _ { s } ; \{ \mathrm { F 1 } _ { \mathrm { P h . 2 } } ^ { ( s ) } , \Delta _ { \mathrm { F 1 } } ^ { ( s ) } \} _ { s }
$$

$$
{ \mathcal { R } } _ { \mathrm { n o r m } }
$$

$$
{ \mathcal { R } } _ { \mathrm { n o r m } }
$$

$$
p _ { \psi }
$$

$$
\mathcal { R } _ { \mathrm { n o r m } }
$$

$$
s \in \{ \mathrm { S M S } ,
$$

$$
f _ { \theta }
$$

$$
\mathcal { D } _ { s } ^ { \mathrm { t r a i n } }
$$

$$
s 2 ;
$$

$$
\Rightarrow f _ { \theta _ { s } ^ { * } }
$$

$$
\mathcal { X } _ { a } ^ { s }
$$

$$
N _ { \mathrm { a b n } } = 3 3 5
$$

$$
i = 1 6 7
$$

$$
\{ N _ { \mathrm { a t t a c k } } ^ { \mathrm { t e s t } } = 1 6 8
$$

$$
\mathbf { x } ^ { ( i ) } 
$$

$$
\left( p _ { \psi } \right)
$$

$$
\bar { \iota ( 1 , \lfloor T / 1 0 \rfloor ) }
$$

$$
\tilde { \mathbf { x } } ^ { ( i ) }  \phi _ { s } ( \mathbf { x } ^ { ( i ) } )
$$

$$
\gamma
$$

$$
B ^ { ( i ) } \gets \mathcal { A } [ i ] ; \mathcal { X } _ { a } ^ { s } [ i ] \gets \mathrm { M a p T o G r i d } ( \tilde { \mathbf { x } } ^ { ( i ) } , B ^ { ( i ) } )
$$

$$
\mathcal { D } _ { \mathrm { t e s t } } ^ { ( s ) }
$$

$$
f _ { \theta _ { s } ^ { * } }
$$

$$
\mathcal { D } _ { \mathrm { t e s t } } ^ { ( s ) }
$$

$$
\Delta _ { \mathrm { F 1 } } ^ { ( s ) }
$$

$$
\mathrm { F 1 } _ { \mathrm { P h . 2 } } ^ { ( s ) } ,
$$

$$
\{ f _ { \theta _ { s } ^ { * } } \} _ { s } , \{ \mathrm { F 1 } _ { \mathrm { P h . 2 } } ^ { ( s ) } , \Delta _ { \mathrm { F 1 } } ^ { ( s ) } \} _ { s }
$$

1) Inverse Classifier Guidance: During each ADT round, inverse classifier guidance steers TabDDPM sampling away from attack-like detector scores. Let $\mathcal { M } ( \cdot ; \theta _ { s } ^ { r } )$ denote the detector after ADT round $r ~ ( \theta _ { s } ^ { 0 } = \theta _ { s } ^ { * }$ from Phase 1). During generation at round r, inverse guidance uses $\mathcal { M } ( \cdot ; \theta _ { s } ^ { r - 1 } )$ (Algorithm 2); we omit subscript s in Eqs. (3)–(5) when s is fixed. For a CDR aggregate x (or its clean estimate $\scriptstyle { \hat { x } } _ { 0 }$ during reverse diffusion), we form the grid $I _ { \mathrm { g r i d } } ( x ) =$ $\mathrm { M a p T o G r i d } ( x , B )$ by replicating x into all cells of ${ \mathcal { C } } =$ $\{ c _ { 1 } , \ldots , c _ { 4 0 } \}$ on a background grid B (for inverse guidance during ADT sampling, B is drawn from normal training grids; see ADT iterative training below). The scalar attack probability is the mean predicted score over C:

$$
p _ { \mathrm { a t t a c k } } = \frac { 1 } { | \mathcal { C } | } \sum _ { k \in \mathcal { C } } \mathcal { M } ( I _ { \mathrm { g r i d } } ( x ) ; \theta _ { s } ^ { r - 1 } ) _ { k } .\tag{3}
$$

Given target probability $p _ { \mathrm { t a r g e t } } = 0 . 3$ and guidance scale λ, the guidance term is

$$
\begin{array} { r } { \nabla _ { \mathrm { g u i d a n c e } } : = \lambda \cdot ( p _ { \mathrm { a t t a c k } } - p _ { \mathrm { t a r g e t } } ) \cdot \nabla _ { x } p _ { \mathrm { a t t a c k } } , } \end{array}\tag{4}
$$

where $\nabla _ { x } p _ { \mathrm { a t t a c k } }$ is the gradient of $p _ { \mathrm { a t t a c k } }$ with respect to x; the guided update is

$$
\hat { x } _ { 0 } \gets \hat { x } _ { 0 } - \nabla _ { \mathrm { g u i d a n c e } } .\tag{5}
$$

For non-blended scenarios, gradient components in non-target CDR channels are scaled by 0.7 before applying the update,

$$
\{ \theta _ { s } ^ { * } \} _ { s \in { \cal S } } ;
$$

$$
( \bar { \mathcal { X } } _ { s } ^ { \mathrm { t r a i n } } , \mathcal { Y } _ { s } ^ { \mathrm { t r a i n } } ) , \mathcal { X } _ { s } ^ { \mathrm { t e s t } } , \mathcal { X } _ { a } ^ { s , \prime }
$$

$$
R ,
$$

$$
N _ { \mathrm { A D T } } ,
$$

$$
\lambda ,
$$

$$
E _ { \mathrm { A D T } } ,
$$

$$
\{ \theta _ { s } ^ { R , * } \} _ { s } ; \{ \Xi _ { s } ^ { r } \} _ { r = 0 } ^ { R }
$$

$$
s \in \mathcal { S } = \{ \mathrm { S M S } , \mathrm { C a l l }
$$

$$
\theta _ { s } ^ { 0 } \gets \theta _ { s } ^ { * } ; \Xi _ { s } ^ { 0 } \gets \mathrm { \bar { E } v a l u a t e } ( \theta _ { s } ^ { 0 } , \mathcal { X } _ { a } ^ { s , \mathrm { T a b } } )
$$

$$
r = 1
$$

$$
\mathcal { X } _ { \mathrm { a d v } } ^ { r }  \emptyset ; f _ { \mathrm { g u i d } }  \operatorname* { m a x } ( 1 , \lfloor T / 1 0 \rfloor )
$$

$$
i = 1
$$

$$
N _ { \mathrm { A D T } }
$$

$$
x _ { T } \sim \mathcal { N } ( 0 , I ) ;
$$

$$
f _ { \mathrm { g u i d } } = 0 , t > 0 ,
$$

$$
( 3 ) ‐ ( 5 ) )
$$

$$
\mathcal { M } ( \cdot ; \theta _ { s } ^ { r - 1 } )
$$

$$
x _ { \mathrm { a d v } } \mathrm { ~  ~ { ~  ~ } ~ } \mathrm { M a p T o G r i d } ( \hat { x } _ { 0 } , B { = } 0 )
$$

$$
\mathcal { C }
$$

$$
{ \mathcal { X } } _ { \mathrm { a d v } } ^ { r } \gets { \mathcal { X } } _ { \mathrm { a d v } } ^ { r } \cup \{ x _ { \mathrm { a d v } } \}
$$

$$
\Rightarrow ( \mathcal { X } _ { s } ^ { \mathrm { t r a i n } , r } , \mathcal { \tilde { V } } _ { s } ^ { \mathrm { t r a i n } , r } )
$$

$$
\theta _ { s } ^ { r - 1 }
$$

$$
E _ { \mathrm { A D T } }
$$

$$
\Rightarrow \theta _ { s } ^ { r }
$$

$$
\Xi _ { s } ^ { r }  \{ \Xi _ { s } ^ { r , \mathrm { o r i g } } , \Xi _ { s } ^ { r , \mathrm { T a b } } \}
$$

$$
\mathscr { X } _ { s } ^ { \mathrm { t e s t } } , \mathscr { X } _ { a } ^ { s , \mathrm { T a b } }
$$

$$
\theta _ { s } ^ { R , * }  \theta _ { s } ^ { R }
$$

$$
\{ \theta _ { s } ^ { R , * } \} _ { s } , \{ \Xi _ { s } ^ { r } \} _ { r = 0 } ^ { R }
$$

preventing guidance from distorting unrelated traffic dimensions.

2) ADTIterative Training Procedure: Round-wise metrics are $\Xi _ { s } ^ { r } = \{ \Xi _ { s } ^ { r , \mathrm { o r i g } } , \Xi _ { s } ^ { r , \mathrm { T a b } } \}$ on the multiplier test split $\mathcal { X } _ { s } ^ { \mathrm { t e s t } }$ and TabDDPM test set $\dot { \mathscr { X } } _ { a } ^ { s , \mathrm { T a b } }$ , respectively; the final weights are $\theta _ { s } ^ { R , * }$ . During inverse guidance, $I _ { \mathrm { g r i d } } ( x )$ is formed with a randomly drawn normal training background. ADT-appended training grids use a zero background with the aggregate written only into C (Sec. IV-A). Hyperparameters (R, N<sub>ADT</sub>, E<sub>ADT</sub>, $\lambda , p _ { \mathrm { t a r g e t } } )$ are given in Sec. V.

## V. EXPERIMENTAL SETUP

Following [4], we use the Telecom Italia Milano CDR collection [35]: a $9 \times 9$ subgrid (81 cells), 18 time slots per day (11:00–14:00, 10-minute intervals), yielding 1,116 grid images per scenario split into 781 training (614 normal $+ ~ 1 6 7$ attack) and 335 test (167 normal + 168 attack) images. We fix random seed 1 for data shuffling and attack-cell selection $( | \mathcal { C } | = 4 0 )$

TabDDPM is trained on normal aggregate CDR rows (3D features) for 100 epochs (learning rate $1 0 ^ { - 4 } .$ , batch size 32). SimpleCNN and ResNet50 are trained for up to 100 epochs in Phases 1–2 (ResNet50: 10% validation split with early stopping to limit overfitting; SimpleCNN: full 100-epoch budget). Phase 3 ADT uses $R = 3$ rounds, $\lambda = 1 . 0 , N _ { \mathrm { A D T } } = 1 0 0$ samples per round and scenario, $E _ { \mathrm { A D T } } = 2 0$ fine-tuning epochs per round, and $p _ { \mathrm { t a r g e t } } = 0 . 3$ . We evaluate Accuracy, Precision, Recall, F1, FPR, and ROC AUC via micro-averaging over $N _ { \mathrm { t e s t } } \times 8 1$ cell decisions $( N _ { \mathrm { t e s t } } = 3 3 5 )$ . No normalization is applied to CNN inputs (consistent with [4]); TabDDPM aggregates are standardized with StandardScaler fit on normal data only. Implementation details follow Supplementary Algorithms S1–S3 and main-text Algorithms 1–2.

TABLE III: Reader’s guide to F1 evaluation protocols.
<table><tr><td>ID</td><td>Protocol</td><td>Main location</td></tr><tr><td>P1</td><td>335-grid TabDDPM test; τ = 0.5; cumulative Phase-3 ADT</td><td>Table IV</td></tr><tr><td>P2</td><td>Same test as P1; fresh 3- Fig. 4 (left) round schedule per method;  $\tau = 0 . 5$ </td><td></td></tr><tr><td>P3</td><td>Locked attack test; τF1max or τY from validation</td><td>Fig. 4 (right); Sec. VI-G</td></tr><tr><td>P4</td><td>Fresh-retrain  $\lambda / p _ { \mathrm { t a r g e t } }$  sweep; locked test</td><td>Table VII</td></tr><tr><td>P5</td><td>Multiplier test split;  $\tau = 0 . 5$ </td><td>Table V</td></tr></table>

## A. Evaluation Protocols for Reported F1

F1 is reported under protocols P1–P5; Table III maps each label to its evaluation setting.

## VI. EXPERIMENTAL RESULTS AND DISCUSSION

Extended derivations, sensitivity sweeps, and full result grids are in the Supplementary Material.

## A. Evaluation Scope

Our use of $\mathbf { \tilde { \Sigma } } ^ { 6 6 } 5 G ^ { \prime }$ refers to the deployment context—CPS services depending on cellular connectivity—rather than a claim the benchmark traces were collected from a 5G standalone (SA) core. We evaluate on the public Milano CDR corpus [35] for reproducible comparison with Hussain et al. [4]; coarse CDR aggregates remain generation-agnostic monitoring records used in contemporary 5G anomaly detection [23], [39]. No public labeled per-cell cellular DDoS CDR dataset exists; packet-/flow-level 5G corpora (5G-NIDD [40], NCSRD-DS-5GDDoS [41]) are not directly mappable to our $9 \times 9 \times 3$ grid representation. Empirical claims are scoped to robust CDRgrid detection under realistic synthetic stress tests in a 5Gmotivated CPS monitoring setting.

## B. Phase Progression Across Diff-DDoS

Table IV summarizes detector performance across phases (all values at $\tau = 0 . 5 ;$ Ph. 2–3 follow protocol P1 on the TabDDPM test split). Both architectures achieve near-perfect Phase 1 F1 $( 9 9 . 6 \% - 1 0 0 \% )$ on multiplier-based test attacks— expected because Phase 1 multipliers (Table II) deviate tens to hundreds of $\sigma$ from normal (Fig. 2). Without retraining, Phase 2 F1 collapses to near zero for SMS, Internet, and Blended on TabDDPM attacks; ResNet50 silent-call F1 falls to 7.6% despite greater model capacity (a comparatively deeper backbone). ADT (Phase 3) recovers ResNet50 F1 to 79.6% (silent-call), 100% (Internet), and 92.8% (Blended) at round 3; SMS F1 stays below 10% at $\tau = 0 . 5$ (a decision-rule artifact resolved in Sec. VI-G). SimpleCNN silent-call F1 declines under ADT (Sec. VI-H); for these sparse stealth scenarios (SMS flooding and silent-call), ResNet50 is the deployable backbone (Secs. VI-G–VI-H).

TABLE IV: Detector performance across phases (protocol P1; all values in $\% ;$ deployment-default $\tau \ = \ 0 . 5 )$ . Ph. 1: multiplier-based test attacks (Acc, Rec, F1); precision is 100% for all rows (omitted). Ph. 2: TabDDPM-generated attacks without retraining. Ph. 3: ADT rounds R1–R3. SC: SimpleCNN; R50: ResNet50.

<table><tr><td rowspan="2">Scenario</td><td colspan="3">Ph. 1</td><td rowspan="2">Ph. 2</td><td colspan="3">Ph. 3F1</td></tr><tr><td>Acc</td><td>Rec</td><td>F1 F1</td><td>R1</td><td>R2</td><td>R3</td></tr><tr><td>SMS (SC)</td><td>99.96</td><td>99.85</td><td>99.93</td><td>0</td><td>9.09</td><td>9.09</td><td>9.09</td></tr><tr><td>SMS  $( \mathbf { R 5 0 } ) ^ { \mathbf { a } }$  Call  $\overline { { { ( \mathrm { S C ) } } ^ { \mathbf { b } } } }$ </td><td>100 99.85</td><td>100 99.40</td><td>100 99.70</td><td>0 52.63</td><td>6.98 38.44</td><td>9.74 12.87</td><td>9.58 9.71</td></tr><tr><td>Call (R50)</td><td>99.83</td><td>99.32</td><td>99.66</td><td>7.56</td><td>77.22</td><td>81.52</td><td>79.62</td></tr><tr><td>Internet (SC) Internet (R50)</td><td>100 99.80</td><td>100 99.21</td><td>100</td><td>0</td><td>89.14</td><td>96.78</td><td>96.62</td></tr><tr><td></td><td></td><td></td><td>99.60</td><td>0</td><td>100</td><td>100</td><td>100</td></tr><tr><td>Blended (SC)</td><td>100</td><td>100</td><td>100</td><td>0</td><td>86.55</td><td>81.13</td><td>79.57</td></tr><tr><td>Blended (R50)</td><td>100</td><td>100</td><td>100</td><td>0</td><td>99.35</td><td>97.22</td><td>92.79</td></tr></table>

<sup>a</sup>ResNet50 SMS values below 10% reflect a fixed-τ artifact (Sec. VI-G).  
<sup>b</sup>SimpleCNN silent-call Phase 3 decline is interpreted in Sec. VI-H.

## C. Attack Realism of TabDDPM-Generated Attacks

We operationalize realistic as manifold-preserving: synthesized attacks remain within the statistical envelope of normal CDR traffic, modeling stealthy control-plane abuse rather than high-volume volumetric DDoS. We quantify distributional fidelity against the normal CDR aggregate manifold and benchmark against Phase 1 multiplier baselines [4].

For each test attack grid we extract the mean 3D aggregate over the 40 attacked cells (168 grids per scenario) and compare it to a normal reference pool of 5,000 cell-level 3D features from normal-only training grids, using target-channel mean absolute z-score as the primary realism metric and squared maximum mean discrepancy (MMD<sup>2</sup>) with a radial basis function (RBF) kernel as a distributional check (full definitions in Supplementary Sec. S-C). Separately, 5,000 unconstrained TabDDPM draws yield joint $\mathrm { M M D ^ { 2 } } = \mathrm { 0 . 5 0 }$ to that normal pool, confirming generative fidelity before scenario constraints. After mild scenario constraints, TabDDPM attacks deviate by only $\approx 0 . 6 \mathrm { - 3 . 5 } \sigma$ on the target channel, whereas multiplier attacks deviate by ≈1–172σ (Fig. 2; Supplementary Table SI). For silent-call, TabDDPM attacks deviate by only 0.59σ on CALL (target-channel $\mathrm { M M D ^ { 2 } } = 0 . 1 9 )$ , so Phase 2 traffic is statistically near-normal; this stealth regime helps explain why multiplier-trained ResNet50 F1 collapses to 7.6% on TabDDPM Call grids despite near-perfect Phase 1 multiplier scores. These results confirm TabDDPM as a manifoldpreserving stealth-attack synthesizer; limitations are discussed in Sec. VII.

## D. ADT Recovery and Multiplier Retention

Beyond Table IV (TabDDPM test, protocol P1), we assess architecture-specific recovery and multiplier-test retention (protocol P5).

SimpleCNN. On the same 335-grid TabDDPM protocol (Table IV, $\tau ~ = ~ 0 . 5 )$ , Internet $( 0 \% ~  ~ 9 6 . 6 \%$ at R3) and Blended $( 0 \%  7 9 . 6 \% )$ improve markedly. Silent-call F1 drops from 52.6% (Phase 2) to 9.7% at R3—Grad-CAM analysis (Sec. VI-H) shows ADT suppresses attack scores on TabDDPM silent-call samples (mean attack score on attacked cells $0 . 3 5  0 . 0 1 )$ while preserving localization; ResNet50

![](images/2e61299a6c20e5595c9a9c2dba765c9088127a2a8ad5eaf7f28d785e391116db.jpg)  
Fig. 2: Dual-axis realism summary on test attack grids (168 per scenario). Bars (left axis) show target-channel mean absolute z-score, and lines (right axis) show joint (3D) $\mathrm { { \bf M M D } ^ { 2 } }$ to normal, for TabDDPM and multiplier attacks across scenarios. Lower values indicate closer alignment with normal CDR aggregates (higher realism/stealth).

TABLE V: Retention on fixed-multiplier attacks (protocol P5; ResNet50): F1 (%) on the locked multiplier test split at $\tau =$ 0.5, before ADT (Phase 1) and after ADT round 3.
<table><tr><td>Scenario</td><td>Phase 1 (pre-ADT)</td><td>ADT round 3</td></tr><tr><td>SMS</td><td>100.00</td><td>100.00</td></tr><tr><td>Call</td><td>99.66</td><td>100.00</td></tr><tr><td>Internet</td><td>99.60</td><td>66.80</td></tr><tr><td>Blended</td><td>100.00</td><td>100.00</td></tr></table>

Call F1 still rises to 79.6%. SMS F1 remains near 9% throughout.

Cell-level localization. Per-cell accuracy heatmaps for the Blended scenario before and after ADT (Fig. 3) show that ADT converts a patchwork of inconsistent per-cell accuracies into a near-uniform $9 \times 9$ accuracy grid (≈ 100% per cell), resolving spatial inconsistencies across attacked locations.

Retention on multiplier-based tests. Table V shows that ResNet50 retains 100% F1 on SMS, silent-call, and Blended on the multiplier test split at $\tau = 0 . 5$ after ADT. For Internet, F1 decreases from 99.6% to 66.8%, while attack-cell recall remains 100% (TP= 6720, FN= 0), indicating a calibration shift driven by false positives on normal cells rather than loss of multiplier-attack recognition; increasing τ to 0.99 raises Internet F1 to 86.3%. SimpleCNN retains 100% F1 on all four scenarios after ADT round 3.

## E. Comparison with Generative and Adversarial-Training Baselines

We compare ADT against a fixed-multiplier-trained detector without iterative retraining [4], fast gradient sign method (FGSM) adversarial training, projected gradient descent (PGD) adversarial training $( \ell _ { \infty } , \ \varepsilon \ = \ 0 . 1 \times .$ 10 steps), and conditional tabular GAN (CTGAN) [27] (150 epochs on up to 20,000 normal cell-level CDR features, same mild scenario scaling as Phase 2). All iterative methods follow the Phase 3 schedule (3 rounds, 100 augmented grids/round, 20 finetuning epochs/round) on ResNet50, evaluated on the locked TabDDPM test set (the same 335-image split as protocol P1: 167 normal + 168 TabDDPM attack grids; indices fixed after Phase 2 generation). We report $\tau = 0 . 5$ as the deploymentdefault (attack-type labels are unavailable at deployment time) before presenting validation-calibrated results with per-method $\tau _ { \mathrm { F 1 m a x } }$ (Sec. VI-G). Fig. 4 summarizes round-3 F1 under protocols P2 (left) and P3 (right); compare methods within each panel, as the two panels apply distinct threshold rules (common $\tau = 0 . 5$ versus per-method $\tau _ { \mathrm { F 1 m a x } } )$

![](images/e82812320a8ecd12335984ec7d315c49f999aea56abc855f3a2b60dcc45c82fe.jpg)

![](images/f8a7186e68d17dc977a658e7bf94f040901d81b8e678c83575a46aef37871e9d.jpg)  
Fig. 3: Cell-level accuracy heatmaps for the Blended scenario before and after ADT.

![](images/dcc399a8a0065c4b0a8f60d541288a8b5c4d153150dd28340f90e6ef81663021.jpg)

![](images/66387db93b6baf7187a5a4b0fef99d929aed0790f507a71f0795c477660819b8.jpg)  
Fig. 4: Baseline comparison on the locked TabDDPM test set (ResNet50, round 3). Left (P2): F1 at deployment-default $\tau = 0 . 5$ (FGSM, PGD, and CTGAN each follow the identical 3-round schedule used for ADT). Right (P3): F1 at τ<sub>F1max</sub> (validation-selected; Sec. VI-G). Compare methods within each panel; each panel applies a single threshold rule. Numeric values appear in Supplementary Table SIII.

Results at $\tau ~ = ~ 0 . 5$ (iterative protocol). Fig. 4 (left panel) and Supplementary Table SIII report round-3 F1 under protocol P2 (iterative baseline-comparison: FGSM, PGD, and CTGAN each follow the same 3-round retraining schedule as ADT). Under this protocol $( \tau = 0 . 5$ , locked TabDDPM test), ADT is the only method reaching operational F1 on Blended without per-method tuning (95.1% vs. $\leq 7 7 . 2 \%$ for FGSM/PGD/CTGAN). On Call, ADT leads (80.4%; nextbest PGD 77.0%). On Internet, ADT ties PGD/CTGAN (66.8%). On SMS, PGD reports higher round-3 F1 at $\tau = 0 . 5$ (P2: 66.8% vs. ADT 7.2%); on the locked attack test split both achieve $\mathrm { \ A U C = 1 . 0 - \mathrm { - } a }$ decision-rule artifact resolved in Sec. VI-G (Fig. 4, right panel).

Validation-calibrated comparison. Fig. 4 (right panel) reports $\tau _ { \mathrm { F 1 m a x } }$ selected on the held-out attack validation split and applied once to the locked attack test (P3; distinct from the 335-grid $\tau = 0 . 5$ scores in Table IV). ADT and PGD are essentially tied on Call (90.5% vs. 89.5%; bootstrap 95% CIs [85.6, 94.7] and [83.3, 94.3], overlapping). The clearest gap is SMS: ADT/FGSM/PGD reach 100% while CTGAN remains at 47.3%. Internet and Blended approach 100% for all methods after calibration.

Is diffusion strictly necessary? FGSM and PGD are training-time perturbation methods that distort existing grids but cannot synthesize standalone manifold-consistent attack samples for independent stress-testing; PGD is competitive on

TABLE VI: Offline generative and augmentation cost (RTX 4090).
<table><tr><td>Method</td><td>Train (s) Aug. 100</td></tr><tr><td>FGSM / PGD</td><td> ${ \lesssim } 0 . 0 1$ </td></tr><tr><td>CTGAN</td><td>62.5 0.007</td></tr><tr><td>TabDDPM</td><td>1296.0</td></tr><tr><td>ADT round</td><td>21.7</td></tr></table>

Note: FGSM/PGD: no generative model; augmentation is on-the-fly grid perturbation. CTGAN and TabDDPM: one-time offline training. ADT round: inverse-guided TabDDPM generation plus 20-epoch detector fine-tuning per round.

TABLE VII: Inverse-guidance sensitivity summary (ResNet50, round-3 F1 %). The operational default row reports P1 (cumulative Phase-3 ADT on the full 335-grid TabDDPM test, Table IV); sweep rows report a fresh-retrain, locked-test oneat-a-time sweep. The full grid is in Supplementary Table SII.
<table><tr><td>Protocol</td><td>λ</td><td> $p _ { \mathrm { t } }$ </td><td>Blended</td><td>SMS</td><td>Internet</td></tr><tr><td>Operational default</td><td>1.0</td><td>0.3</td><td>92.8</td><td>9.6</td><td>100.0</td></tr><tr><td rowspan="3">Fresh-retrain sweep</td><td>1.0</td><td>0.3</td><td>66.8</td><td>100.0</td><td>66.8</td></tr><tr><td>2.0</td><td>0.3</td><td>66.8</td><td>100.0</td><td>100.0</td></tr><tr><td>1.0</td><td>0.1</td><td>100.0</td><td>100.0</td><td>65.9</td></tr></table>

Call but lacks CDR aggregate realism (Sec. VI-C). FGSM and PGD require no generative pre-training, whereas CTGAN trains ≈ 21× faster than TabDDPM (62.5 s vs. 1296.0 s; Table VI), but showed a large Blended round-wise regression (99.7% → 66.8% at $\tau \ : = \ : 0 . 5 )$ and calibrated SMS F1 of only 47.3% (P3; Supplementary Table SIII bottom block). ADT combines manifold-preserving synthesis with deployable Blended robustness at the default threshold—properties not jointly matched by the alternatives.

## F. Inverse-Guidance Sensitivity and Online Latency

ADT depends on guidance scale λ (steering strength) and p<sub>target</sub> (adversarial difficulty). We perform a one-at-a-time sweep on $\mathrm { R e s N e t 5 0 : ~ } \lambda \in \{ 0 . 2 5 , 0 . 5 , 1 . 0 , 2 . 0 \}$ at $p _ { \mathrm { t a r g e t } } { = } 0 . 3 .$ and $p _ { \mathrm { t a r g e t } } ~ \in ~ \{ 0 . 1 , 0 . 3 , 0 . 5 , 0 . 7 \} ~ \mathrm { a t } ~ \lambda \mathrm { = 1 . 0 } .$ Target-channel $\mathbf { M M D ^ { 2 } }$ stays flat (≈ 0.51–0.57), so neither hyperparameter materially affects on-manifold realism. On the fresh-retrain one-at-a-time sweep (P4; locked TabDDPM attack test, round-3 F1 at $\tau = 0 . 5 )$ in Table VII, robustness is sensitive and nonmonotonic: Internet F1 rises from 66.8% at λ=1.0 to 100% at λ=2.0; Blended F1 rises from 66.8% at $p _ { \mathrm { t a r g e t } } { = } 0 . 3$ to 100% at $p _ { \mathrm { t a r g e t } } { = } 0 . 1$ . The table also reports the operational Phase-3 default; the full sweep is in Supplementary Sec. S-D. (1.0, 0.3) is the fixed deployment compromise, not a global optimum.

Runtime CPS monitoring requires only the online forwardpass latencies in Table VIII. Diff-DDoS targets periodic CDR grid analytics, not per-packet or Ultra-Reliable and Low-Latency Communications (uRLLC)-level (∼1 ms) inference: each $9 \times 9 \times 3$ grid aggregates a 10-minute window (Sec. V), so the operative CPS latency budget is the inter-grid interval (600 s). SimpleCNN attains 35.7 ms mean (49.5 ms p95) and ResNet50 42.7 ms mean (60.4 ms p95) on an RTX 4090 reference GPU; worst batch-1 p95 is ≈10<sup>4</sup>× below the aggregation window (Supplementary Table SVIII). Offline pipeline cost is in Supplementary Sec. S-H (Table SVI).

TABLE VIII: Online single-grid inference latency for Phase 3 (ADT) detectors (batch = 1, scenario-averaged; RTX 4090, 500 iterations after 50 warmup). The per-scenario and batch-= 32 breakdown is in Supplementary Table SVIII.
<table><tr><td>Arch.</td><td>Mean (ms)</td><td>p95 (ms)</td><td>Grids/s</td></tr><tr><td>SimpleCNN</td><td>35.7</td><td>49.5</td><td>28.1</td></tr><tr><td>ResNet50</td><td>42.7</td><td>60.4</td><td>23.4</td></tr></table>

ROC Curves (Micro-averaged, Blended scenario)

![](images/8cb14cfca48b2600634495233fed1e339334219473e24912f6d66b80856185f7.jpg)

![](images/99e2b2b1483e73074d541687ff2e85a67b0634ad5a49836d4dfdbe23756c53c2.jpg)  
Fig. 5: ROC curves for the Blended scenario on TabDDPMgenerated attacks. Left: Phase 2 comparison of SimpleCNN and ResNet50. Right: ResNet50 before and after ADT in Phase 3.

## G. Operating Points: ROC-AUC and Threshold Calibration

Micro-averaged ROC AUC (full table in Supplementary Table SVII; representative Blended ROC curves in Fig. 5, with additional ROC/PR curves in the Supplement) confirms Phase 1 perfect separability (AUC= 1.0). Phase 2 reveals vulnerability: SimpleCNN drops to AUC= 0.43 (SMS) and 0.75 (Blended), while ResNet50 holds 0.90–0.99. After ADT, SimpleCNN recovers to 0.85 (SMS) and 0.96 (Blended); ResNet50 reaches 0.95–1.0. The AUC/F1 gap is a decisionrule artifact: for SMS the detector correctly ranks attacks (ResNet50 AUC up to 1.0 after ADT) yet $\tau = 0 . 5$ yields near-zero SMS F1 (Table IV; locked attack split in Fig. 6) because attack scores concentrate below 0.5; validation-based threshold calibration (below) resolves this.

We implement validation-based threshold calibration and cost-sensitive retraining for the two threshold-sensitive sparseattack scenarios (SMS flooding and silent-call). For each, τ<sub>F1max</sub> (validation F1-maximizing) and $\tau _ { \mathrm { Y } }$ (Youden’s ${ \boldsymbol { J } } =$ TPR−FPR) [42] are selected on a held-out validation split of the TabDDPM attack-only test grids (84 of 168 attack images, 50% split, seed 1) and applied once to the remaining 84 locked attack test grids (separate from the primary 335-image microaveraged evaluation in Table IV), so no test-set tuning occurs. Fig. 6 compares locked-test F1 at $\tau = 0 . 5 , \tau _ { \mathrm { F 1 m a x } } ,$ and τ<sub>Y</sub> for Phase 2 (TabDDPM, no ADT) versus Phase 3 (ADT) on SMS flooding and silent-call. For ResNet50 on SMS flooding, calibration alone resolves the apparent ADT failure: on the locked attack split (Fig. 6), F1 rises from 4.82% at $\tau = 0 . 5$ to 99.55% at τ = 0.05 (AUC= 1.0, FPR= 0%; Table SIV), confirming that ADT learned the attack and that the sub-10% SMS F1 at $\tau = 0 . 5$ in Table IV was a fixed-threshold artifact rather than a learning failure.

We also evaluate class-weighted BCE with positive-class weight $( w _ { + } \approx 8 . 5 )$ and focal loss $( \gamma = 2 , \alpha = 0 . 7 5 )$ [43] over five seeds. For SMS flooding, neither calibration nor cost-sensitive retraining raises SimpleCNN F1 above $1 2 . 3 \pm$ 8.6% (large seed variance), indicating a capacity limit on the lightweight backbone; we therefore adopt ResNet50 as the deployable detector for sparse attacks. For silent-call, weighted BCE raises mean locked-test AUC from 0.65 to $0 . 8 5 \pm 0 . 0 9$ , but F1 at $\tau _ { \mathrm { F 1 m a x } }$ remains only $4 8 . 5 \pm 2 5 . 3 \%$ with large seed variance. The full per-architecture grid, Youden thresholds with FPR, and cost-sensitive comparisons appear in Supplementary Sec. S-F (Table SIV); see also Fig. 6.

![](images/f191748a633d4c375857a9baaaa27e1590de73188125a92f2f1bbfc1ae3675df.jpg)

![](images/0c1adc66e89d824348aa0b838a75dea686c8a7a043175f9ba5f80e4072d706ec.jpg)

![](images/8b39880b53d816bce22ab3684662012df6a867ec8840d3774e26900ceef41db8.jpg)  
Fig. 6: Locked-test F1 at the default threshold versus validation-calibrated thresholds $( \tau _ { \mathrm { F 1 m a x } } , \tau _ { \mathrm { Y } } )$ for the SMS and silent-call scenarios.

## H. Grad-CAM Interpretability of the SimpleCNN Silent-Call Anomaly

SimpleCNN silent-call F1 falls from 52.6% (Phase 2) to 9.7% (Phase 3, Table IV), which appears to contradict the ADT robustness claim. We apply Gradient-weighted Class Activation Mapping (Grad-CAM) [44] to SimpleCNN’s last convolutional layer on three grid types—normal (real) CDR snapshots, TabDDPM silent-call attacks, and fixed-multiplier baseline attacks—comparing the Phase 2 detector with the Phase 3 ADT model over $N { = } 4 0$ test grids per condition. Warmer overlay colors in Fig. 7 indicate stronger Grad-CAM activation (where the model was most sensitive on that grid image); $\hat { p }$ is the mean predicted attack score on attacked cells, analogous to $p _ { \mathrm { a t t a c k } }$ in Sec. IV-D. The figure shows three illustrative TabDDPM silent-call grids in which Phase 2 assigns moderate $\hat { p }$ on attacked cells while Phase 3 drives $\hat { p }$ near zero, even though the fraction of Grad-CAM mass on attacked cells (attn) stays ≈ 50% in both phases.

Fig. 8 summarizes the score–saliency asymmetry quantitatively: across N=40 TabDDPM silent-call test grids, mean attacked-cell scores drop from $0 . 3 5 \pm \ : 0 . 4 7$ (Phase 2) to $0 . 0 1 { \pm } 0 . 0 7$ (Phase 3), matching the F1 collapse, while saliency mass on attacked cells stays comparable $( 0 . 5 1 { \pm } 0 . 0 1 \ \mathrm { v s } . 0 . 5 1 { \pm }$ 0.02). On multiplier-based attacks, mean scores remain near 1.0 in both phases, while Phase 3 saliency becomes diffuse $( 0 . 1 8 \pm 0 . 2 4 ~ \mathrm { v s . } ~ 0 . 5 1 \pm 0 . 0 1 $ ; Supplementary Table SV). ADT on SimpleCNN therefore overfits to stealth guidance samples and suppresses detection on manifold-preserving TabDDPM traffic rather than losing localization; ResNet50 still reaches 79.6% F1 under the same protocol. Full numerical Grad-CAM results are in Supplementary Sec. S-G (Table SV).

![](images/ed6cf489ef5c53205b093dbbe82c9440376e7f142b65ca7781930eb9d88d9760.jpg)

Fig. 7: Grad-CAM overlays for TabDDPM silent-call attacks (SimpleCNN): Call-channel input (left) and saliency maps for Phase 2 (trained on multiplier attacks, tested on TabDDPM silent-call grids) vs. Phase 3 ADT (same backbone after retraining). Boxed cells mark ground-truth attacked locations (green on the input panel, white on the overlays). Subplot titles report the mean predicted attack score on attacked cells $( \hat { p } )$ and the fraction of Grad-CAM saliency mass on those cells (attn).  
![](images/981baa313f9f8678425b1b224a79df15178a0f1c51bc1c4129a2cea9488841ee.jpg)

![](images/cd39ce0f79ac330a442027fa8690859b4a07e03fb6f6bcb4889e1d311d3485a5.jpg)  
Fig. 8: Quantitative Grad-CAM summary for SimpleCNN before vs. after ADT on attacked cells. $L e f t { \mathrm { : } }$ mean predicted attack score on attacked cells for TabDDPM and multiplier attack sets. Right: Grad-CAM saliency mass on attacked cells for the same sets. Bars compare Phase 2 and Phase 3 (ADT); error bars denote ±1 sample standard deviation over $N = 4 0$ test grids per condition.

## VII. LIMITATIONS AND CONCLUSION

We found that multiplier-trained CDR detectors fail on manifold-preserving attacks: Phase 2 F1 collapsed to near zero in three of four scenarios (Table IV), while ADT with ResNet50 recovered deployable F1 for silent-call (79.6%), Internet (100%), and Blended (92.8%), retaining near-perfect multiplier-test performance except an Internet calibration shift (Table V). This extends Hussain et al. [4] from multiplier baselines to manifold-preserving stress tests and applies inverseguided TabDDPM for iterative cellular IDS hardening. Calibration resolves the SMS threshold artifact (Sec. VI-G), Grad-CAM shows SimpleCNN can suppress stealth silent-call scores without losing localization (Sec. VI-H), and ADT matched or exceeded gradient- and GAN-based baselines (Sec. VI-E).

Three limitations bound external validity. No labeled percell cellular DDoS CDR ground truth exists (Sec. VI-A), so evaluation uses synthetic rather than operator-confirmed attack traces. Attack realism uses coarse 3D aggregate statistics— mean |z| and MMD<sup>2</sup> on SMS/CALL/INTERNET sums (Sec. VI-C)—a proxy that omits full grid spatiotemporal structure and does not establish field adversary fidelity. Finally, the Milano corpus reflects 3G/4G-era metropolitan traffic [35], and ten-minute grid monitoring targets periodic CPS analytics rather than sub-millisecond uRLLC control (Sec. VI-F).

Operator CDR validation and packet-level 5G corpora where mapping permits [40], [41] should confirm ADT transfer beyond Milano; future work includes adaptive adversaries and regional scale-up [6]. For 5G-connected CPS and IoT services, stealthy signaling abuse threatens network and physicalprocess availability; pre-deployment stress testing is a practical safeguard. Manifold-preserving TabDDPM attacks expose multiplier-trained brittleness, and ADT recovers robustness with ResNet50 and calibration, offering a workflow to harden cell-level DDoS monitors when labeled attack CDRs remain scarce.

## REFERENCES

[1] A. Humayed, J. Lin, F. Li, and B. Luo, “Cyber-physical systems security—a survey,” IEEE Internet Things J., vol. 4, no. 6, pp. 1802– 1831, 2017.

[2] L. He, Z. Yan, and M. Atiquzzaman, “LTE/LTE-A network security data collection and analysis for security measurement: A survey,” IEEE Access, vol. 6, pp. 4220–4242, 2018.

[3] 3GPP, “Study on the security aspects of the next generation system,” 3rd Generation Partnership Project (3GPP), Tech. Rep. TR 33.899, Rel. 14, v1.3.0, 2017.

[4] B. Hussain, Q. Du, B. Sun, and Z. Han, “Deep learning-based DDoSattack detection for cyber-physical system over 5G network,” IEEE Trans. Ind. Informat., vol. 17, no. 2, pp. 860–870, 2021.

[5] A. Alam, A. Umer, I. Ullah, and A. Alsayat, “AI-enabled cybersecurity framework for future 5G wireless infrastructures,” Sci. Rep., vol. 16, p. 7055, 2026.

[6] B. Hussain, M. Bilal, T. Li, H. Pervaiz, X. Tang, Q. Du, F. Ahmad, M. Azhar, and J. Zhang, “AI-native closed-loop security for 6Genabled cyber-physical systems: From edge detection to network-wide mitigation,” arXiv:2606.08173, 2026.

[7] ENISA, “ENISA threat landscape 2025,” European Union Agency for Cybersecurity (ENISA), Tech. Rep., Oct. 2025.

[8] G.-H. Tu, C.-Y. Li, C. Peng, and S. Lu, “How voice call technology poses security threats in 4G LTE networks,” in Proc. IEEE Conf. Commun. Netw. Secur. (CNS), 2015, pp. 442–450.

[9] I. Murynets and R. P. Jover, “Anomaly detection in cellular machineto-machine communications,” in Proc. IEEE Int. Conf. Commun. (ICC), 2013, pp. 2138–2143.

[10] D. K. Nguyen, R. E. Malki, and F. Rebecchi, “RRC signaling storm detection in O-RAN,” in Proc. IEEE Symp. Comput. Commun. (ISCC), 2025, pp. 1–7.

[11] H. Wen, P. Porras, V. Yegneswaran, A. Gehani, and Z. Lin, “5G-Spector: An O-RAN compliant layer-3 cellular attack detection service,” in Proc. 31st Annu. Netw. Distrib. Syst. Secur. Symp. (NDSS), 2024.

[12] K. He, D. D. Kim, and M. R. Asghar, “Adversarial machine learning for network intrusion detection systems: A comprehensive survey,” IEEE Commun. Surveys Tuts., vol. 25, no. 1, pp. 538–566, 2023.

[13] H. A. Alatwi and C. Morisset, “Adversarial machine learning in network intrusion detection domain: A systematic review,” arXiv:2112.03315, 2021.

[14] S. Mavoungou, G. Kaddoum, M. Taha, and G. Matar, “Survey on threats and attacks on mobile networks,” IEEE Access, vol. 4, pp. 4543–4572, 2016.

[15] D. Dahiya, “DDoS attacks detection in 5G networks: Hybrid model with statistical and higher-order statistical features,” Cybern. Syst., vol. 54, no. 6, pp. 888–913, 2023.

[16] P. Munaweera, S. Prasad, T. Hewa, Y. Siriwardhana, and M. Ylianttila, “Federated learning-powered DDoS attack detection for securing cyber physical systems in 5G and beyond networks,” in Proc. 14th Int. Conf. Internet Things (IoT), 2024, pp. 273–278.

[17] B. H. Swathi, R. Arvind, S. Benvin, R. Gupta, R. Maranan, and R. Jayanthi, “Securing 5G-enabled cyber-physical systems: An optimized reflection equivariant quantum neural network approach to DDoS attack detection,” in Proc. 3rd Int. Conf. Self Sustain. Artif. Intell. Syst. (ICSSAS), 2025, pp. 606–612.

[18] D. Naboulsi, M. Fiore, S. Ribot, and R. Stanica, “Large-scale mobile traffic analysis: A survey,” IEEE Commun. Surveys Tuts., vol. 18, no. 1, pp. 124–161, 2016.

[19] F. Xu, Y. Li, H. Wang, P. Zhang, and D. Jin, “Understanding mobile traffic patterns of large scale cellular towers in urban environment,” IEEE/ACM Trans. Netw., vol. 25, no. 2, pp. 1147–1161, 2017.

[20] A. Furno, M. Fiore, and R. Stanica, “Joint spatial and temporal classification of mobile traffic demands,” in Proc. IEEE INFOCOM, 2017, pp. 1–9.

[21] D. Naboulsi, R. Stanica, and M. Fiore, “Classifying call profiles in largescale mobile traffic datasets,” in Proc. IEEE INFOCOM, 2014, pp. 1806– 1814.

[22] K. Sultan, H. Ali, and Z. Zhang, “Call detail records driven anomaly detection and traffic prediction in mobile cellular networks,” IEEE Access, vol. 6, pp. 41 728–41 737, 2018.

[23] Z. Aziz and R. Bestak, “Insight into anomaly detection and prediction and mobile network security enhancement leveraging k-means clustering on call detail records,” Sensors, vol. 24, no. 6, p. 1716, 2024.

[24] N. Shone, T. N. Ngoc, V. D. Phai, and Q. Shi, “A deep learning approach to network intrusion detection,” IEEE Trans. Emerg. Topics Comput. Intell., vol. 2, no. 1, pp. 41–50, 2018.

[25] G. Apruzzese, M. Colajanni, L. Ferretti, A. Guido, and M. Marchetti, “On the effectiveness of machine and deep learning for cyber security,” in Proc. 10th Int. Conf. Cyber Conflict (CyCon), 2018, pp. 371–390.

[26] D. Arp, E. Quiring, F. Pendlebury, A. Warnecke, F. Pierazzi, C. Wressnegger, L. Cavallaro, and K. Rieck, “Dos and don’ts of machine learning in computer security,” in Proc. 31st USENIX Security Symp. (USENIX Security), 2022, pp. 3971–3988.

[27] L. Xu, M. Skoularidou, A. Cuesta-Infante, and K. Veeramachaneni, “Modeling tabular data using conditional GAN,” in Adv. Neural Inf. Process. Syst. (NeurIPS), 2019, pp. 7335–7345.

[28] A. Kotelnikov, D. Baranchuk, I. Rubachev, and A. Babenko, “TabD-DPM: Modelling tabular data with diffusion models,” in Proc. 40th Int. Conf. Mach. Learn. (ICML), 2023, pp. 17 564–17 579.

[29] M. C. Stoian, E. Giunchiglia, and T. Lukasiewicz, “A survey on deep learning approaches for tabular data generation: Utility, alignment, fidelity, privacy, diversity, and beyond,” Trans. Mach. Learn. Res., 2026. [Online]. Available: https://openreview.net/forum?id=RoShSRQQ67

[30] I. J. Goodfellow, J. Shlens, and C. Szegedy, “Explaining and harnessing adversarial examples,” in Proc. Int. Conf. Learn. Represent. (ICLR), 2015.

[31] A. Madry, A. Makelov, L. Schmidt, D. Tsipras, and A. Vladu, “Towards deep learning models resistant to adversarial attacks,” in Proc. Int. Conf. Learn. Represent. (ICLR), 2018.

[32] M. Alauthman, N. Aslam, A. Al-Qerem, A. Aldweesh, and P. Sureephong, “Generative adversarial networks for intrusion detection systems: A comprehensive survey of applications, challenges, and research directions,” Arab. J. Sci. Eng., vol. 51, no. 1, pp. 179–203, 2026.

[33] W. Nie, B. Guo, Y. Huang, C. Xiao, A. Vahdat, and A. Anandkumar, “Diffusion models for adversarial purification,” in Proc. 39th Int. Conf. Mach. Learn. (ICML), 2022.

[34] M. A. Merzouk, E. Beurier, R. Yaich, N. Boulahia-Cuppens, F. Cuppens, and F. Khomh, “Diffusion-based adversarial purification for intrusion detection,” in Data and Applications Security and Privacy XXXIX, ser. Lecture Notes in Computer Science. Springer, 2025, vol. 15722, pp. 351–370.

[35] G. Barlacchi, M. De Nadai, R. Larcher, A. Casella, C. Chitic, G. Torrisi, F. Antonelli, A. Vespignani, A. Pentland, and B. Lepri, “A multi-source dataset of urban life in the city of Milan and the province of Trentino,” Sci. Data, vol. 2, p. 150055, 2015.

[36] J. Ho, A. Jain, and P. Abbeel, “Denoising diffusion probabilistic models,” in Adv. Neural Inf. Process. Syst. (NeurIPS), vol. 33, 2020, pp. 6840– 6851.

[37] P. Dhariwal and A. Nichol, “Diffusion models beat GANs on image synthesis,” in Adv. Neural Inf. Process. Syst. (NeurIPS), vol. 34, 2021, pp. 8780–8794.

[38] T. Xie, C.-Y. Li, J. Tang, and G.-H. Tu, “How voice service threatens cellular-connected IoT devices in the operational 4G LTE networks,” in Proc. IEEE Int. Conf. Commun. (ICC), 2018, pp. 1–6.

[39] Z. Aziz and R. Bestak, “Modeling voice traffic patterns for anomaly detection and prediction in cellular networks based on CDR data,” IEEE Trans. Mobile Comput., vol. 23, no. 12, pp. 13 131–13 143, 2024.

[40] Y. Siriwardhana, S. Samarakoon, P. Porambage, M. Liyanage, S.-Y. Chang, J. Kim, J. Kim, and M. Ylianttila, “Descriptor: 5G wireless network intrusion detection dataset (5G-NIDD),” IEEE Data Descr., vol. 2, pp. 358–369, 2025.

[41] N. C. of Scientific Research “Demokritos” and S. H. (Greece), “NCSRD-DS-5GDDoS: 5G radio and core metrics containing sporadic DDoS attacks,” Zenodo, version v3.0, Oct. 2024.

[42] W. J. Youden, “Index for rating diagnostic tests,” Cancer, vol. 3, no. 1, pp. 32–35, 1950.

[43] T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollar, “Focal loss for´ dense object detection,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 42, no. 2, pp. 318–327, 2020.

[44] R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh, and D. Batra, “Grad-CAM: Visual explanations from deep networks via gradient-based localization,” Int. J. Comput. Vis., vol. 128, no. 2, pp. 336–359, 2020.

![](images/a288af6b7957b2e51d0bf208800b34ecf79d742a6bf9c5f3cc6c3a96908f0e45.jpg)

Bilal Hussain (Senior Member, IEEE) received the B.E. degree in electrical engineering from Bahria University, Pakistan, in 2010, the M.Sc. degree in information and communications engineering from the University of Leicester, Leicester, U.K., in 2011, and dual Ph.D. degrees in information and communications engineering from The Hong Kong Polytechnic University (PolyU), Hong Kong SAR, China, in 2021, and Xi’an Jiaotong University, Xi’an, China, in 2022. He is currently a Lecturer with the Division of Science,

Engineering, and Health Studies, School of Professional Education and Executive Development, PolyU. Previously, he was a Postdoctoral Fellow with the Centre for Advances in Reliability and Safety, Hong Kong. His research focuses on intelligent 6G networks, Agentic AI, real-time edge intelligence, and robust cyber-physical systems security.

![](images/cc5c947e32281a2290e7c1592b55c341bbe57b63f54d7cf26d09df3b05f96b23.jpg)

Xiao Tang (Member, IEEE) received the B.S. degree in information engineering (Elite Class Named After Tsien Hsue-shen) and the Ph.D. degree (Hons.) in information and communication engineering from Xi’an Jiaotong University, Xi’an, China, in 2011 and 2018, respectively. He is currently an Associate Professor with the School of Information and Communication Engineering, Xi’an Jiaotong University, Xi’an, China.

![](images/6292ca3ce8dfbe70524892e9cf935a6c0d8d56c9729904adbb044c172f3d5d5a.jpg)

Qinghe Du (Member, IEEE) received the B.S. and M.S. degrees in electrical engineering from Xi’an Jiaotong University, China, in 2001 and 2004, respectively, and the Ph.D. degree in computer engineering from Texas A&M University, College Station, TX, USA, in 2010. He is currently a Professor with the School of Information and Communications Engineering, Xi’an Jiaotong University. He has published over 150 technical papers in refereed technical journals and international academic conferences. His research interests widely cover related areas of mobile wireless communications and networking. He received several Best Paper Awards from the international academic conferences and technical journals. He served and/or serves as an Associate Editor for the IEEE TRANSAC-TIONS ON COMMUNICATIONS and IEEE COMMUNICATIONS LETTERS, an Area Editor for KSII Transactions on Internet and Information Systems, and an Editor for Electronics.

![](images/23240360e27d6e1a69ff673f905a1af9e410d338e8c02f876cc69a2e32d89361.jpg)

Tan Li (Member, IEEE) received the Ph.D. degree in computer science from City University of Hong Kong, Hong Kong SAR, China, in 2023, the M.Eng. degree in control engineering from the University of Science and Technology of China, Hefei, China, in 2019, and the B.Eng. degree in automation from Central South University, Changsha, China, in 2016. She is currently an Assistant Professor with the Department of Computer Science, The Hang Seng University of Hong Kong, Hong Kong SAR, China. Her research interests include federated learning and machine learning for wireless communications.

![](images/1467a3e464860ddfe4811a53e417a6dfc28a21a0eae7056be379cb8fb764df5e.jpg)

Muhammad Azhar (Member, IEEE) received the B.S. degree in computer science from the National University of Computer and Emerging Sciences, Islamabad, Pakistan, the master’s degree in computer science from Sejong University, Seoul, South Korea, and the Ph.D. degree in machine learning and data mining from Shenzhen University, Shenzhen, China. He is currently an Associate Professor and the Associate Head of the Department of Applied Data Science with Hong Kong Shue Yan University, Hong Kong

SAR, China. He also serves as the Director of the university’s Big Data Laboratory. His research interests include data mining, natural language processing, and advanced deep-learning techniques.

![](images/23a83321b43e2b8725a97360ba2f3cc5b35d2b964480980791a3890ab49cff85.jpg)

Danista Khan (Member, IEEE) received the Ph.D. degree in electrical and electronic engineering from The Hong Kong Polytechnic University (PolyU), Hong Kong SAR, China, specializing in applied deep-learning methods for wireless sensing. She received the master’s degree in electrical engineering from the University of Engineering and Technology, Lahore, Pakistan, and the bachelor’s degree in electrical engineering from The University of Lahore, Lahore, Pakistan. She is currently a Lecturer with the Division of Business and Hospitality Management, School of Professional Education and Executive Development, PolyU. Her research interests include wireless communications and networking, wireless sensing, and the Internet of Things.