# Training-Free Human-in-the-Loop Anomaly Detection via Memory Bank Correction

Ayusha Abbas, Saram Abbas, and Kabita Adhikari

Abstract—Anomaly detectors are hardest to deploy exactly where training data is scarcest: a newly commissioned production line has a handful of verified “golden" samples and no machinelearning engineer on the factory floor. We present a trainingfree human-in-the-loop framework in which a domain expert corrects a PatchCore detector by direct memory bank editing: no retraining, no gradients, no original training data. A false-positive correction inserts the reviewed image's normal patches through a self-calibrating novelty gate admitting only those beyond the median pool-normal nearest-neighbour distance. From a bank built on only ten golden samples, operator corrections close a median 66% of the gap to an uncorrected fully trained bank (mean 80%, raised by three categories that overshoot parity), significantly improving 12 of 15 MVTec AD categories and harming none: ten samples plus corrections outperform hundreds of samples without them. On already-trained banks the headroom is smaller and concentrated where the bank undersamples normal appearance (gated: toothbrush +0.10, metal nut +0.09, zipper +0.05, screw +0.05), and no category except grid is significantly harmed. Evaluation uses a held-out protocol (20 splits per category, Holmcorrected Wilcoxon), because corrected images entering the bank inflate naive evaluation toward AUROC 1.0 by memorisation. Passive and active querying are statistically indistinguishable; a matched-label-budget control attributes gains to deploymenttime label production at 43% of exhaustive-review cost; a defectmemory extension fails decisively. Feedback is simulated from ground truth; live expert trials, where mislabelling is costliest on small banks, remain future work.

Index Terms—Active learning, anomaly detection, human-inthe-loop, industrial inspection, memory bank correction, MVTec AD, PatchCore

## I. INTRODUCTION

Deep anomaly detection now achieves near-perfect discrimination on industrial inspection benchmarks [1], [2], but deployed models do not stay accurate. Normal product appearance drifts (new component variants, changed illumination, camera recalibration) and errors accumulate. The standard remedy, periodic retraining, requires new labelled data, training infrastructure, and machine learning (ML) expertise that is rarely present on the factory floor. The people who can retrain the model (ML engineers) are not the people who see its errors every day (quality engineers), so in regulated manufacturing environments a model correction can take weeks to months of escalation, retraining, and revalidation.

A second, sharper version of the same problem is cold start: a newly commissioned production line has almost no training data at all. In both situations the quality engineer already knows what the model does not: which flagged images are genuine defects and which are false alarms. Existing systems give them no way to put that knowledge into the model.

Detection failures carry industrial-scale costs: automotive original equipment manufacturers (OEMs) paid USD 51 billion in warranty claims in 2023 [3], a single end-of-line contamination excursion in semiconductor fabrication can exceed USD 21 million [4], and a defective-medicine prosecution carries six-figure penalties [5]. All three are downstream costs of defects that escaped inspection, and a defect grows more expensive at every stage it survives. A detector that degrades silently therefore does not just lose accuracy; it feeds escaped defects into the most expensive end of this cost structure.

Benchmark ceilings do not close this gap. PatchCore's headline 99.6% area under the receiver operating characteristic (ROC) curve (AUROC) on MVTec AD requires an ensemble of three large backbones at high resolution [1]; the leaner configurations that fit factory edge hardware give up several points of accuracy (our single-backbone deployment configuration averages 93.4%), and even the 99.6% ensemble, classifying 50,000 images a day, hands the quality team 200 misclassifications every day. PatchCore's authors themselves list deployment-time adaptation as an open problem [1].

This paper closes that gap with a training-free human-inthe-loop (HITL) framework: the quality engineer reviews a flagged error, makes the single binary judgment they already make daily (“normal" or “defect"), and the framework applies that judgment as a direct edit to the model's knowledge store, effective immediately. It needs no retraining, no ML engineer, and no access to the original training data. Figure 1 previews the outcome: one short review session clears a false alarm on images the loop never saw, while genuine defects remain flagged.

PatchCore [1] is uniquely suited to this because its knowledge is not buried in network weights: it stores normal appearance as an explicit memory bank of patch features. Adding patches from a falsely flagged normal image extends what the model accepts as normal; removing patches near a missed defect raises future scores for similar defects. Each correction is an interpretable edit to a data structure. Prior work modifies PatchCore banks automatically, via continual learning [6] or loss-driven optimisation [7], but still requires ML infrastructure; to our knowledge, ours is the first framework in which a domain expert corrects specific deploymenttime misclassifications by direct memory bank surgery.

HITL Correction Generalises: Held-Out Images Before and After One Review Session (seed 10, passive)

![](images/9710392099502d06ff70617ff5065b8ae51e887983ea0d83b052bb3688c3f112.jpg)  
Fig. 1. What training-free correction looks like on images the loop never touches. For toothbrush and metal nut, a held-out normal image is falsely flagged before correction (anomaly heatmap hot everywhere) and cleared after a single simulated review session (16–25 single-decision reviews for the seeds shown; the category means are \~13 and \~20), while a held-out defect remains confidently flagged, with the anomaly map concentrated on the defect itself (contaminated bristles; flipped part). Heatmaps share a colour scale within each row; scores are the image-level anomaly scores

## A. Contributions

We evaluate the framework on all 15 MVTec AD categories [2], 20 independent splits each, under Holm-corrected Wilcoxon tests. Our contributions:

• Cold-start recovery from ten samples. From a bank built on only 10 golden samples, operator corrections close a median 66% of the gap to an uncorrected fully trained bank (mean 80%, raised by three categories that overshoot parity), significantly improving 12 of 15 categories and harming none. Ten samples plus corrections outperform hundreds of samples without them. We do not claim curation dominates data: a corrected full-training bank still finishes higher (toothbrush 0.927 vs. 0.904, metal nut 0.989 vs. 0.927). The claim is about what is achievable on day one of a deployment, and this is the one setting in which every category either improves or is left untouched.

• A training-free correction mechanism. False positives (FPs) and false negatives (FNs) are resolved by direct memory bank edits, with no retraining, gradients, or training data. An FP/FN ablation shows FP insertion reproduces the entire gain while FN removal is a measured near no-op: the mechanism works by teaching normality, not by memorising defects.

• A self-calibrating novelty gate. A patch is inserted only if its nearest-neighbour distance to the bank exceeds the median pool-normal distance. On already-trained banks the gate eliminates the held-out degradation the raw loop causes on five of six well-covered categories, preserves every category that benefits, and is robust across a quantile sensitivity sweep, leaving grid as the single category still significantly harmed.

• A held-out evaluation protocol (plus AUCC, the areaunder-the-correction-curve efficiency metric): corrected images enter the bank, so naive evaluation on reviewed images inflates toward AUROC 1.0 by memorisation. Correcting a review pool while scoring a disjoint heldout set separates generalisation from memorisation, and reveals six categories where the ungated loop is flat-tonegative on mature banks, a boundary invisible to poolonly evaluation that we report as an explicit negative result. A matched-label-budget control shows the gains equal what the same labels would achieve at training time: the loop's value is producing those labels at deployment, at 43% of the cost of exhaustive review.

• A strategy equivalence result. Passive (random) and active (uncertainty) querying are statistically indistinguishable on every category, because bank edits act globally and the review budget exhausts the error pool under either ordering. No acquisition function is needed.

In summary: from ten golden samples a domain expert with no ML knowledge closes most of the gap to a fully trained detector, improving 12 of 15 categories and harming none, in tens of single-decision reviews. On mature banks the same loop raises held-out AUROC on the four categories whose banks undersample normal appearance, from 0.826 to 0.972 (toothbrush) and 0.900 to 0.998 (metal nut) in mean sessions of \~13 and \~20 reviews; those figures are for the ungated loop, and under the novelty-gated configuration we recommend for deployment the same categories reach 0.927 and 0.989, with no category except grid significantly harmed, so the loop is safe to leave on.

## II. RELATED WORK

Our work sits at the intersection of unsupervised anomaly detection, human-in-the-loop machine learning, and active learning query strategies.

## A. Unsupervised Anomaly Detection

Deep learning-based anomaly detection spans a broad taxonomy of reconstruction-based, density-based, and discriminative approaches [8]. Early anomaly detection approaches relied on reconstruction-based methods, such as convolutional autoencoders and generative adversarial networks (GANs) like AnoGAN [9], which detect anomalies by their high reconstruction error or latent space distance. These suffer from mode collapse, blurry reconstructions of high-frequency textures, and high computational overhead. A second family exploits pretrained convolutional neural network (CNN) features [10], showing that ImageNet representations transfer to industrial inspection without domain-specific training. Student-teacher frameworks [11] extend this by detecting anomalies through student-teacher feature discrepancy.

The state of the art is dominated by memory bank methods. SPADE [12] showed that nearest-neighbour comparison against a bank of pretrained features achieves strong localisation with no training; PatchCore [1] refines this with a greedy coreset of locally-aware patch features, reaching 99.6% mean AUROC on MVTec AD with a three-backbone ensemble at 320×320 resolution. That figure is a static benchmark result, and PatchCore's authors explicitly leave deploymenttime adaptation as future work. Our framework builds on PatchCore's architecture and answers that gap. PaDiM [13] and SimpleNet [14] are related feature-based alternatives.

## B. Human-in-the-Loop Machine Learning

Human-in-the-loop machine learning [15] incorporates human feedback into the learning process, classically via active learning [16]. In visual inspection, Rožanec et al. [17] combine active learning with explainable artificial intelligence (XAI) heat maps to reduce annotation burden for supervised retraining; HILAD [18] proposes bidirectional HITL for timeseries anomaly detection. Within anomaly detection, Das et al. [19], [20] and Bodor et al. [21] use expert feedback to reweight or re-optimise detectors, so the feedback still flows through retraining. Closest to our setting, Vishwakarma et al. [22] use human feedback on flagged samples to adapt the decision threshold with false-positive-rate guarantees, but never modify the representation; a follow-up additionally adapts a learned scoring network, but through periodic gradient optimisation [23]. We differ on all three axes: no retraining at all, corrections that edit the representation of normality itself, and responsibility shifted from ML engineers to the quality engineers who already review the errors. To our knowledge, ours is the first framework enabling human-driven, trainingfree correction of specific misclassified samples through direct memory bank surgery at deployment time.

## C. Active Learning and Query Strategies

Uncertainty sampling [24], the standard heuristic in active learning, selects the sample about which the model is least confident. Query-by-committee [25] and densityweighted methods [16] provide alternative strategies. A consistent finding is that uncertainty-based strategies outperform passive random sampling [16]. Our work tests this regularity for memory bank correction and finds that it does not transfer: under multi-seed held-out evaluation, uncertainty sampling fails to outperform random sampling, and we provide a geometric explanation for why. This finding has implications for both active learning theory and the design of HITL anomaly detection systems.

## D. Continual Learning and Test-Time Adaptation

Continual learning [26] and test-time adaptation [27] address deployment-time improvement through parameter updates, requiring careful regularisation against catastrophic forgetting. Within the PatchCore literature specifically, continual learning variants have emerged: PatchCoreCL [6] maintains per-task memory banks updated as new tasks arrive, and FR-PatchCore [7] updates the memory bank during training via a negative cosine similarity loss. A growing line of continual industrial anomaly detection likewise treats the memory bank or its coreset as the unit of adaptation: Li et al. [28] adapt the detector as new normal samples arrive under distribution shift, CADIC [29] incrementally updates embeddings within a fixed-size unified coreset shared across tasks, MECAD [30] assigns per-class experts with coreset replay buffers, and orthogonal LoRA banks [31] isolate per-task adaptation in low-rank parameter subspaces to preserve learned normality. These approaches share our insight that the memory bank is the modifiable component of memory-based detectors, but differ fundamentally in the update signal: all are driven by a stream of new, assumed-clean training data arriving at task or batch boundaries, whereas our framework is driven by an operator's dispositions of individual misclassified samples: error-targeted corrections at deployment time with no new training set, no optimisation objective, and no assumption about task structure. Mechanistically closest to our novelty gate (Section III-B2), TimeRep [32] augments a time-series memory bank at inference with representations whose nearestneighbour distance exceeds a quantile threshold, but as unsupervised drift tracking over a stream, not as gating of humanverified corrections.

Few-shot anomaly detection builds the detector itself from k samples: RegAD meta-learns a registration proxy across categories [33], WinCLIP transfers language-guided features zero-/few-shot [34], and GraphCore [35] and FastRecon [36] build compact memory banks directly from the k normals. All treat the residual gap to full-training performance as fixed once deployed. Our cold-start evaluation (Section III-E5) starts in the same regime, with a bank built from ten samples, and measures how much of that gap operator corrections then close; the contribution is the recovery-by-correction protocol, not a new few-shot detector. The framework itself is orthogonal: it applies on top of any memory bank model as a deploymenttime correction layer.

TABLE I  
POSITIONING OF OUR WORK RELATIVE TO PRIOR METHODS.
<table><tr><td>Method</td><td>No re- training</td><td>Human feedback</td><td>Deploy- ment-time</td><td>Specific correction</td></tr><tr><td>Autoencoder</td><td>x</td><td>x</td><td>x</td><td>x</td></tr><tr><td>AnoGAN</td><td>x</td><td>x</td><td>x</td><td>x</td></tr><tr><td>PatchCore</td><td> $\checkmark$ </td><td>x</td><td> $\pmb { \chi }$ </td><td>x</td></tr><tr><td>Few-shot AD</td><td>Partial</td><td>x</td><td>Partial</td><td>x</td></tr><tr><td>Continual learning</td><td> $\pmb { \chi }$ </td><td>x</td><td>√</td><td>x</td></tr><tr><td>Test-time adapt.</td><td>Partial</td><td>x</td><td>√</td><td>x</td></tr><tr><td>HITL + retraining</td><td>x</td><td>√</td><td>x</td><td>Partial</td></tr><tr><td>Ours</td><td> $\checkmark$ </td><td>√</td><td>√</td><td>√</td></tr></table>

## E. Positioning

Table I summarises how our approach relates to existing paradigms across four key desiderata.

## III. METHODOLOGY

We first summarise the PatchCore baseline, then present the HITL correction mechanism (including the novelty gate), the querying strategies, the AUCC metric, and the experimental protocols.

## A. PatchCore Baseline

1) Features and Memory Bank: PatchCore [1] represents normal appearance as an explicit memory bank of locallyaware patch features from a frozen WideResNet-50 [37] pretrained on ImageNet [38]. Feature maps are taken from backbone levels 2 (28×28, 512 channels; fine texture) and 3 (14×14, 1,024 channels; part-level structure): level 3 is bilinearly upsampled to 28×28 and concatenated channelwise with level 2, so each of the $2 8 \times 2 8 \ : = \ : 7 8 4$ spatial positions carries a 1,536-dimensional descriptor $p _ { i j } ( x )$ . We write $P ( x )$ for that set, $| P ( x ) | \ = \ 7 8 4$ . Backbone weights are fixed throughout coreset construction and every HITL experiment; no gradient is computed at any point.

The union of patch features over a category's defect-free training images is too large to query directly (some 200,000 vectors for a typical 200–300 image category), so PatchCore compresses it by greedy k-center coreset subsampling [39], retaining the subset that minimises the maximum distance from any training patch to its nearest retained entry. We keep 1% of training patches, giving banks $M \in \mathbb { R } ^ { N \times 1 5 3 6 }$ ranging from $N \ = \ 4 7 0$ (toothbrush, 60 training images) to 3,065 (hazelnut, 391). How undersampled a category's bank is turns out to predict where correction helps (Section IV).

2) Anomaly Scoring: At inference time, for a test image $x ,$ we extract its 784 patch features $P ( x )$ and compute the anomaly score as the maximum patch-level distance to the memory bank, each patch's distance being the mean over its $k = 9$ nearest bank entries:

$$
s ( x ) = \operatorname* { m a x } _ { p \in P ( x ) } d _ { 9 } ( p ) , \qquad d _ { 9 } ( p ) = \frac 1 9 \sum _ { m \in \Lambda \mathfrak { p } _ { 9 } ( p , M ) } \lVert p - m \rVert _ { 2 }\tag{1}
$$

where $\mathcal { N } _ { 9 } ( p , M )$ denotes the nine nearest entries of M to $p .$ The max identifies the single most anomalous patch, making the score sensitive to the small local defects typical of industrial inspection, and the same nine distances form the anomaly map used for localisation. Averaging over a small neighbourhood, rather than taking the 1-NN distance, makes the per-patch value less sensitive to a single unrepresentative bank entry. Retrieval uses a ball-tree index.

An image is classified anomalous if $s ( x ) \ \geq \ \tau .$ with $\tau$ the 90th percentile of scores on normal training images (no anomalous examples needed), recomputed after every correction round. Our implementation is a single WideResNet-50 at $2 2 4 \times 2 2 4$ without score reweighting, and departs from canonical PatchCore in using the mean-of-nine patch statistic of Eq. 1 in place of the 1-NN score; this is a deliberate implementation choice, applied identically to every baseline and every corrected bank reported here. It gives 0.934 mean AUROC, deliberately below PatchCore's 99.6% ensemble headline, because a model at 99.6% leaves no headroom for studying human-driven correction, whereas this deployable configuration has realistic error pools.

## B. Human-in-the-Loop Feedback Framework

Our framework (Fig. 2) is a post-deployment correction layer: it never touches the backbone, the training data, or any gradient. Every correction is an edit to the memory bank $M ,$ so it applies to any nearest-neighbour detector with an explicit feature store.

Correction proceeds over R rounds (Algorithm 1). In round $r \colon ( 1 )$ score all images against $M ^ { ( r ) } ;$ (2) form the error pool $E ^ { ( r ) } = \{ x$ : label(x) ≠ predict $( x , M ^ { ( r ) } , \tau ^ { ( r ) } ) \}$ ; (3) present the expert one image from $E ^ { ( r ) } ; ( 4 )$ the expert gives one binary label, “normal" or “defect"; (5) apply the corresponding correction to obtain $M ^ { ( r + 1 ) }$ . The loop stops at round R or when the error pool is exhausted. The entire human contribution per round is one label on one image.

1) False Positive Correction: Memory Bank Expansion: A false positive occurs when a normal image xFp receives an anomaly score $s ( x _ { \mathrm { F P } } ) \geq \tau$ , indicating that its patch features are anomalously distant from the current memory bank. The geometric interpretation is that $x _ { \mathrm { F P } }$ lies in a region of normal appearance space that is underrepresented in $M \colon$ a gap in the normal manifold that causes the nearest-neighbour distance to exceed the anomaly threshold despite the image being normal.

The correction adds all 784 patch features of $x _ { \mathrm { F P } }$ directly to the memory bank, filling this gap:

$$
M ^ { ( r + 1 ) } \gets M ^ { ( r ) } \cup P ( x _ { \mathrm { F P } } )\tag{2}
$$

This operation increases $| M |$ by 784 entries per $\mathrm { F P }$ correction. Without a size constraint, repeated FP corrections would cause the memory bank to grow without bound, increasing nearest-neighbour query time quadratically. To prevent this, we enforce a maximum bank size $N _ { \mathrm { m a x } }$ via random subsampling after each FP correction:

$$
M ^ { ( r + 1 ) } \gets \mathrm { R a n d o m S a m p l e } \left( M ^ { ( r ) } \cup P ( x _ { \mathrm { F P } } ) , N _ { \operatorname* { m a x } } \right)\tag{3}
$$

![](images/d246ee15c0c133a8a56a2c9b8aed105f77741f9af232d9629e0e76c3857626af.jpg)  
Fig. 2. Framework overview. A deployed PatchCore model flags errors; a quality engineer (not an ML engineer) resolves each with a single binary decision, which is applied as a direct edit to the memory bank and takes effect immediately. The conventional remedy (bottom, dashed) requires an ML team and a retraining pipeline.

applied whenever the insertion would leave $| M ^ { ( r + 1 ) } | > N _ { \mathrm { m a x } } .$

The value of $N _ { \mathrm { m a x } }$ is set uniformly across categories, as detailed in Section III-E3. The random subsampling preserves the global coverage properties of the coreset while bounding computational cost; a control experiment with the cap removed (Section III-E3) shows it is not a sensitive parameter.

2) Novelty-Gated Insertion: Adding all 784 patches of a corrected image is maximally effective when the bank genuinely undersamples normal appearance. On categories whose banks already cover the normal manifold well, however most inserted patches duplicate existing coverage, and this redundant insertion can perturb previously correct decisions (Section IV-D2). The framework therefore supports gating each candidate patch on novelty: a patch $p \ \in \ P ( x _ { \mathrm { F P } } )$ is inserted only if its nearest-neighbour distance to the current bank exceeds a threshold ${ \tau _ { \mathrm { n o v } } }$

$$
M ^ { ( r + 1 ) } \gets M ^ { ( r ) } \cup \big \{ p \in P ( x _ { \mathrm { F P } } ) : \operatorname* { m i n } _ { m \in M ^ { ( r ) } } \| p - m \| _ { 2 } > \tau _ { \mathrm { n o v } } \big \} ,\tag{4}
$$

where ${ \tau _ { \mathrm { n o v } } }$ is set once, before correction begins, as the median nearest-neighbour distance of all review-pool normal patches to the baseline bank (pool data only, never held-out images). Figure 3 shows the gate on a real toothbrush false positive: the half of its patches within existing coverage is rejected, the genuinely novel half inserted. The gate is selfcalibrating (permissive on undersampled banks, conservative on well-covered ones) and needs no tuning: outcomes change smoothly between the 0.25 and 0.75 quantiles (Section IV-B). The criterion has close relatives: it is an online, thresholdbased analogue of the greedy k-center coreset rule that built the bank [1], [39], asking of each candidate patch at insertion time the coverage question that coreset construction asks of the training set; and a quantile-thresholded 1-NN novelty filter has recently been used to keep a time-series memory bank nonredundant under unsupervised stream adaptation [32]. What is new is the role: applied to human-verified corrections, the filter is not an economy measure. Section IV-B shows it is the difference between a loop that harms well-covered categories and one that is safe to leave on. We evaluate the loop with and without the gate; gating is the recommended deployment configuration.

![](images/080d8d2eb308ca90d948218e19fee41ab6546187d55fa76e57f88d8cb7d41882.jpg)  
Fig. 3. The novelty gate on a real false positive (toothbrush, seed 10). The threshold τnov is the median 1-NN distance of all pool-normal patches to the baseline bank (black outline); of the flagged image's 784 patches, those within existing coverage (left of τ) are rejected and only the genuinely novel ones (right of τ) are inserted.

3) False Negative Correction: Memory Bank Surgery: A false negative occurs when an anomalous image xFN scores $s ( x _ { \mathrm { F N } } ) \ < \ \tau ;$ typically one or a few defect patches sit spuriously close to normal bank entries, inside the normal manifold.

The correction first identifies the anchor patch: the patch of $x _ { \mathrm { F N } }$ lying furthest from the current bank under the 1-NN distance,

$$
p ^ { * } = \underset { p \in P ( x _ { \mathrm { F N } } ) } { \mathrm { a r g m a x ~ } } \underset { m \in M } { \mathrm { m i n } } \| p - m \| _ { 2 }\tag{5}
$$

Note that Eq. 5 selects the anchor by 1-NN distance, whereas the image score of Eq. 1 is driven by the mean-of-nine statistic $d _ { 9 }$ . The two criteria do not in general identify the same patch, so removal is aimed at the most locally isolated patch rather than at the patch that actually set the score; Section IV-I shows this to be one of the two reasons FN correction behaves as a near no-op. Because $x _ { \mathrm { F N } }$ was classified normal, $p ^ { * }$ sits in a region of feature space that is ambiguously close to the normal manifold. The correction removes the $k _ { \mathrm { r e m o v e } } ~ = ~ 5$ nearest memory bank entries to $p ^ { * }$ , creating a gap in the normal manifold at precisely the location of the anomalous patch:

Algorithm 1 HITL Memory Bank Correction   
Require: M (memory bank), $\overline { { { X } _ { T } } }$ (test images + labels), R   
(rounds), strategy ∈ {passive, active}   
Ensure: ${ \dot { M } } ^ { ( R ) }$ (corrected bank), auroc\_curve, AUCC   
1: scores $ \operatorname { S C O R E A L L } ( X _ { T } , M )$   
2: AUROC0 ← AUROC(scores, labels $( X _ { T } ) )$   
3: corrected $ \emptyset ; \mathsf { a u r o c \_ c u r v e }  [ \mathrm { A U R O C _ { 0 } } ]$   
4: for $r = 1 , 2 , \ldots , R$ do   
5: scores(r) ← SCOREALL(XT, M)   
6: $\tau ^ { ( r ) } \gets \mathrm { p e r c e n t i l e } _ { 9 0 } ( \mathrm { s c o r e s } ^ { ( r ) } [ \mathrm { n o r m a l } ] )$   
7: $E ^ { ( r ) }  \{ x \ : \ \mathrm { l a b e l } ( x ) \neq \mathrm { p r e d i c t } ( x , M , \tau ^ { ( r ) } ) \}$ 1   
corrected   
8: if $E ^ { ( r ) } = \emptyset$ then   
9: break no errors remain   
10: end if   
11: x\* ← QUERYSTRATEGY $( E ^ { ( r ) }$ , strategy, $\tau ^ { ( r ) } )$   
Eq. 8/9   
12: [Human] inspects $x ^ { * } ;$ labels it normal or defective $\triangleright$   
single binary decision   
13: if $x ^ { * }$ is FP (human says normal, model said defective)   
then   
14: $M \gets \mathrm { F P C o R R E C T } ( x ^ { * } , M , N _ { \operatorname* { m a x } } )$   
15: else  FN: human says defective, model said normal   
16: $M \gets \mathrm { F N C o R R E C T } ( x ^ { * } , M , k _ { \mathrm { r e m o v e } } = 5 )$   
17: end if   
18: corrected ← corrected $\cup \{ x ^ { * } \}$   
19: $\mathrm { A U R O C } _ { r } \gets \mathrm { A U R O C } ( \mathrm { S c o r E A L L } ( X _ { T } , M )$ , labels)   
20: auroc\_curve.append(AUROCr)   
21: end for   
22: $g _ { r } \gets \operatorname* { m a x } ( 0 , \mathrm { A U R O C } _ { r } - \mathrm { A U R O C } _ { 0 } ) , r = 1 , \ldots , R$   
23: $\mathrm { A U C C }  \frac { \mathbf { \Phi } } { R ( 1 - \mathrm { A U R O C } _ { 0 } ) } \sum _ { r } g _ { r }$   
24: return M, auroc\_curve, AUCC

$$
I _ { \mathrm { r e m o v e } } = k \mathbf { N } \mathbf { N } ( p ^ { * } , M ^ { ( r ) } , k = 5 )\tag{6}
$$

$$
M ^ { ( r + 1 ) } \gets M ^ { ( r ) } \setminus \{ M _ { i } ^ { ( r ) } : i \in I _ { \mathrm { r e m o v e } } \}\tag{7}
$$

The excised gap raises future scores for patches falling in the same region, so similar defects are flagged; $k _ { \mathrm { r e m o v e } } = 5$ balances gap size against over-removal. After each correction (FP or FN) the bank index is rebuilt and all images rescored.

## C. Querying Strategies

The query strategy determines only the order in which errors are reviewed, not the correction mechanism. We evaluate two:

Passive querying draws uniformly at random from the error pool,

$$
x ^ { * } \sim \operatorname { U n i f o r m } ( E ^ { ( r ) } ) ,\tag{8}
$$

requiring no computation beyond the scoring pass and sampling FP and FN errors in proportion to their prevalence [16].

Active querying implements uncertainty sampling [16], [24], selecting the image whose score is closest to the decision threshold:

$$
x ^ { * } = \underset { x \in E ^ { ( r ) } } { \operatorname { a r g m i n } } | s ( x ) - \tau ^ { ( r ) } | .\tag{9}
$$

Its rationale, that boundary-adjacent samples are most informative, assumes local model updates. Section IV shows empirically that this assumption fails for memory bank correction, whose updates are global (Section V).

D. Evaluation Metric: Area Under the Correction Curve (AUCC)

AUROC before and after correction misses the efficiency of the process: how quickly improvement arrives over the review budget. The Area Under the Correction Curve (AUCC) captures it. With AUROCr the AUROC after round $r \ ( r \ = \ 0$ the baseline), define the per-round gain $\begin{array} { r l } { g _ { r } } & { { } = } \end{array}$ max $( 0 , \mathrm { A U R O C } _ { r } - \mathrm { A U R O C } _ { 0 } )$ and

$$
\mathsf { A U C C } = \frac { 1 } { R \left( 1 - \mathrm { A U R O C } _ { 0 } \right) } \sum _ { r = 1 } ^ { R } g _ { r } \ \in \ [ 0 , 1 ] ,\tag{10}
$$

i.e. the cumulative gain normalised by the area of immediate, sustained perfect correction. AUCC is comparable across baselines and rewards early improvement: the same final AUROC scores higher if reached in fewer reviews, matching the practical preference for fewer human interventions. It is an AUROC-specific instance of the area-under-the-learningcurve (ALC) family standard in active-learning evaluation [40], adapted to correction: gains are measured against an already-deployed baseline and normalised by the remaining headroom $1 - \mathsf { A U R O C } _ { 0 }$ , so scores are comparable across categories with different starting points.

## E. Experimental Setup

1) Dataset: All experiments use MVTec AD [2], the standard industrial anomaly detection benchmark: 5,354 images across 15 categories (5 textures, 10 objects) spanning 73 defect types, from fixed-pose rigid objects (bottle) to deformable configurations (cable) and random textures (leather, wood). Training sets contain only defect-free images; test sets contain both classes.

2) Category Selection: Stratified by Separation Index: We evaluate all 15 MVTec AD categories using passive and active querying strategies. The separation index σ, defined as the normalised distance between normal and anomalous score distributions, predicts correction utility and is used to interpret results across categories:

$$
\sigma = { \frac { \mu _ { \mathrm { a n o m a l o u s } } - \mu _ { \mathrm { n o r m a l } } } { \sigma _ { \mathrm { n o r m a l } } } }\tag{11}
$$

where μanomalous and $\mu _ { \mathrm { n o r m a l } }$ denote the mean anomaly score for each class, and $\sigma _ { \mathrm { n o r m a l } }$ denotes the standard deviation of the normal class score distribution. Normalising by $\sigma _ { \mathrm { n o r m a l } }$ (rather than a pooled variance) reflects how many normal-distribution standard deviations separate the two class means, a natural unit given that PatchCore's classification threshold itself is set from the normal score distribution. A high separation index indicates well-separated score distributions (easy category); a low index indicates overlapping distributions (hard category), as illustrated in Fig. 4. Table II lists all 15 categories with their separation indices, baseline AUROCs and dataset statistics; the baselines average 0.9342 and span 0.8139 (toothbrush) to 1.0000 (bottle, 1eather).

TABLE II  
ALL 15 MVTEC AD CATEGORIES, ORDERED BY SEPARATION INDEX σ (DESCENDING).
<table><tr><td>Category</td><td>Type</td><td>σ</td><td>Base.</td><td>Train</td><td>Test (g/a)</td></tr><tr><td>Bottle</td><td>Object</td><td>50.5</td><td>1.0000</td><td>209</td><td>20/63</td></tr><tr><td>Leather</td><td>Texture</td><td>23.1</td><td>1.0000</td><td>245</td><td>32/92</td></tr><tr><td>Carpet</td><td>Texture</td><td>9.5</td><td>0.9932</td><td>280</td><td>28/89</td></tr><tr><td>Wood</td><td>Texture</td><td>7.6</td><td>0.9912</td><td>247</td><td>19/60</td></tr><tr><td>Hazelnut</td><td>Object</td><td>7.0</td><td>0.9964</td><td>391</td><td>40/70</td></tr><tr><td>Tile</td><td>Texture</td><td>6.0</td><td>0.9776</td><td>230</td><td>33/84</td></tr><tr><td>Zipper</td><td>Object</td><td>4.9</td><td>0.8787</td><td>240</td><td>32/119</td></tr><tr><td>Cable</td><td>Object</td><td>4.6</td><td>0.9185</td><td>224</td><td>58/92</td></tr><tr><td>Capsule</td><td>Object</td><td>4.3</td><td>0.9290</td><td>219</td><td>23/109</td></tr><tr><td>Pill</td><td>Object</td><td>4.3</td><td>0.9242</td><td>267</td><td>26/141</td></tr><tr><td>Metal Nut</td><td>Object</td><td>4.3</td><td>0.8803</td><td>220</td><td>22/93</td></tr><tr><td>Transistor</td><td>Object</td><td>3.7</td><td>0.9558</td><td>213</td><td>60/40</td></tr><tr><td>Screw</td><td>Object</td><td>3.0</td><td>0.8703</td><td>320</td><td>41/119</td></tr><tr><td>Toothbrush</td><td>Object</td><td>2.6</td><td>0.8139</td><td>60</td><td>12/30</td></tr><tr><td>Grid</td><td>Texture</td><td>2.6</td><td>0.8839</td><td>264</td><td>21/57</td></tr></table>

Anomaly Score Distributions: Normal vs Anomalous (Four representative categories spanning the full difficulty range)  
![](images/08f9498bf52226fffc16fd95002fd31a78a06e8dab36f8fab714e1a21e21f847.jpg)  
Fig. 4. Anomaly score distributions for four representative categories spanning the full difficulty range (blue = normal, red = anomalous). Overlap decreases as σ increases, from heavily overlapping (toothbrush) to fully separated (leather).

3) Implementation Details and Bank Size Caps: The WideResNet-50 backbone uses ImageNet-pretrained weights loaded via the timm library (version 0.9+) and is kept frozen throughout. Input images are resized to 224×224 pixels and normalised with ImageNet mean [0.485, 0.456, 0.406] and standard deviation [0.229, 0.224, 0.225]. Features are extracted from backbone stages 2 and 3 using the features\_only interface. The memory bank nearest-neighbour index uses k = 9, Euclidean distance, and a ball-tree data structure (scikit-learn NearestNeighbors with $\mathtt { a l g o r i t h m } = \prime \mathtt { b a l l \_ t r e e } ^ { \prime }$ $\mathrm { n \_ j o b s { = } { - 1 } } )$ for efficient batch querying.

The memory bank is capped at $N _ { \mathrm { m a x } } = 1 2 { , } 0 0 0$ patches, uniformly across all categories: when an insertion would exceed the cap, the bank is randomly subsampled back to $N _ { \mathrm { m a x } }$ . The cap exists only to bound scoring cost; it is not a sensitive parameter. A 20-seed control run with the cap removed entirely changes no category's mean held-out delta except transistor (+0.021, the only Holm-significant shift) and doubles the scoring cost.

Each HITL experiment runs for $R = 3 0$ correction rounds per strategy per category. The ball-tree index is rebuilt after each correction to reflect the updated bank.

4) Held-Out Evaluation Protocol: A methodological subtlety of memory bank correction is that Eq. 2 adds the corrected image's own patches to the bank. If the same images are used both for correction and for evaluation, AUROC trends toward 1.0 partly because corrected normal images are subsequently scored against exact copies of their own features. That is memorisation, not generalisation. All headline results in this paper therefore use a held-out protocol.

For each experimental run, the category's test images are split 70/30 (stratified by label) into a review pool and a heldout set. Only pool images may enter the error pool $E ^ { ( r ) }$ and receive corrections; the decision threshold $\tau ^ { ( r ) }$ is computed from pool normals only. The held-out set is rescored against the updated bank after every correction round but never receives corrections and never influences querying or thresholding. Held-out AUROC after round r is the generalisation measure: it reflects whether corrections improve detection on images the loop has never seen. AUCC (Eq. 10) is likewise computed on the held-out curve. Pool AUROC is reported only as a secondary, self-consistency measure and is explicitly marked as inflated by memorisation wherever it appears.

Every (category, strategy) condition is run at 20 independent split seeds (10–29), each seed drawing a different pool/heldout partition and, for passive querying, a different random review order. Reported values are means over the 20 seeds with standard deviations; per-seed distributions are reported in Section IV-G. Statistical significance of per-category improvement is assessed with a two-sided Wilcoxon signed-rank test on the 20 paired per-seed held-out deltas (final minus baseline), and passive-vs-active differences with the same test on paired per-seed endpoints; both are Holm-corrected across the 15 categories at $\alpha = 0 . 0 5$ . Seeds 0–9 were reserved for earlier exploratory work and are excluded from all reported results. A uniform bank cap $N _ { \mathrm { m a x } } = 1 2 { , } 0 0 0$ is used for all categories under this protocol (Section III-E3).

5) Cold-Start Protocol: The protocol above corrects a fully trained bank. To evaluate the deployment scenario where little training data exists (a new production line with only 10–20 verified “golden" samples), we additionally run a cold-start protocol. For each seed, the initial bank is built from only $k ~ \in ~ \{ 1 0 , 2 0 \}$ training normals drawn with a seed-coupled random number generator: their patch features are extracted with the same backbone and reduced by greedy k-center coreset selection at the main protocol's 1% rate (78 patches at k = 10, 156 at $k = 2 0 )$ . The HITL loop then runs unchanged, with the same 70/30 pool/held-out splits as the full-training runs on matched seeds, and the budget extended to 90 rounds to accommodate the larger initial error pools. Because splits are seed-matched, the full-training result on the same seed is a paired reference: we report the recovery fraction, i.e. the fraction of the cold-start-to-full-training held-out gap closed by correction, and the median number of reviews to reach parity with the full-training bank.

All experiments are conducted on NVIDIA GPUs; timing details are reported in Section VI-A.

## IV. EXPERIMENTAL RESULTS

We present the full 15-category benchmark on fully trained banks, then the effect of the novelty gate (Section IV-B) and the cold-start setting (Section IV-C), followed by the analyses that explain the results: key categories, strategy equivalence, seed robustness, label economics, and the FP/FN asymmetry.

## A. Full 15-Category Results Overview

Table III reports held-out results for all 15 MVTec AD categories under the protocol of Section III-E4: means over 20 pool/held-out splits (seeds 10–29), ordered by baseline AUROC; “HO" is the held-out set, which never receives corrections.

The benchmark partitions into three regimes. Four categories gain +0.040 to +0.146 AUROC, on 15–20 of 20 seeds each: toothbrush (+0.146, 19/20), metal nut (+0.098, 20/20), zipper (+0.054, 20/20), screw (+0.040, 15/20). These are exactly the categories whose baseline errors are false positives from an undersampled normal manifold, which FP corrections repair. Five near-ceiling categories (baseline $\ge ~ 0 . 9 9 )$ are unaffected. Six categories are flat-to-negative, losing up to 0.047 AUROC (grid, transistor, pill, capsule, cable, tile): their pool AUROC still climbs (the memorisation artefact), but most inserted patches duplicate existing coverage, and this redundant insertion degrades held-out performance. A 20-seed control with the bank cap removed leaves these categories unchanged, ruling out subsampling churn as the cause; Section IV-B shows the novelty gate eliminates the degradation on five of the six.

These regimes survive formal testing. Applying a two-sided Wilcoxon signed-rank test to the 20 per-seed held-out deltas of each category (Holm-corrected across the 15 categories), all four winners are significantly positive (toothbrush, metal nut, zipper: adjusted $p < 1 0 ^ { - 4 }$ ;screw: adjusted $p = 0 . 0 3 2 )$ and four of the six flat-to-negative categories are significantly negative (grid, pill, tile, transistor; adjusted $p ~ \leq ~ 0 . 0 3 2 )$ which confirms that the degradation on those categories is systematic rather than noise. Cable and capsule are statistically indistinguishable from zero (adjusted $p = 0 . 0 5 7$ and 0.28). The same paired test applied to passive-vs-active per-seed differences finds no significant difference on any category (all unadjusted $p \ge 0 . 0 6 8 ;$ all adjusted $p = 1 )$ , formalising the strategy-equivalence claim of Section IV-E.

## B. Novelty-Gated Insertion Eliminates the Harm

The novelty gate (Section III-B2) changes the risk profile of the entire benchmark. With gating enabled (final column of

Table III), every flat-to-negative category except grid turns statistically neutral or positive: transistor $- 0 . 0 2 4  + 0 . 0 0 5 .$ pill $- 0 . 0 2 3  + 0 . 0 0 7 .$ cable $- 0 . 0 1 9  + 0 . 0 0 6 .$ tile $- 0 . 0 1 3 $ -0.001, capsule $- 0 . 0 0 9 \  \ + 0 . 0 0 3 ;$ the paired per-seed improvement over the ungated loop is Holm-significant on all five. All four winners remain Holm-significantly positive (toothbrush +0.101, metal nut +0.089, zipper +0.051, screw $+ 0 . 0 4 8 )$ ; the price of conservatism is a modest reduction on toothbrush (—0.044 vs. ungated, not Holm-significant across seeds) and metal nut (—0.009, Holm-significant but an order of magnitude smaller than the retained gain). The result is a deployment configuration in which no category except grid is significantly harmed, while every previous winner remains a winner. Figure 5(d) shows the effect directly on pill: the ungated trajectory drifts below baseline round by round, while the gated trajectory stays above it.

The gate's insertion rates show the self-calibration at work. On well-covered banks it makes corrections conservative (tile inserts a mean of only 65 of 784 candidate patches per FP correction, transistor 78) and the harm of redundant insertion vanishes; on undersampled banks it stays permissive (metal nut inserts 271, zipper 236) and the gains survive. Grid is the diagnostic exception: it inserts 230 patches per correction, more than metal nut, yet still degrades, so its errors are not a coverage problem at all; we return to it in Section IV-D2. The threshold requires no tuning: repeating the full sweep with ${ \tau _ { \mathrm { n o v } } }$ at the 0.25 and 0.75 quantiles of the same distance distribution shifts outcomes smoothly and modestly. At 0.25 the rescue of the negative categories weakens (of the four significantly negative categories, only transistor reaches non-negative); at 0.75 it strengthens (grid improves to —0.019) but a real cost appears on metal nut (—0.043 vs. ungated, Holm-significant, 0/20 seeds better). The median sits at the sweet spot, and no quantile exhibits knife-edge behaviour.

C. Cold-Start Deployment: Near-Full Performance from Ten Samples

Deployment-time correction delivers its largest gains where training data is scarcest. Table IV reports the cold-start protocol (Section III-E5) with $k ~ = ~ 1 0$ golden samples and the novelty gate enabled. Starting from banks of just 78 patches, the correction loop produces Holm-significant heldout improvements on 12 of 15 categories, with no category significantly harmed, and recovers on average 80% of the gap to the full-training bank (median 0.66 over the 11 categories with a meaningful gap). On two categories the corrected coldstart bank ends substantially above the full-training reference: metal nut (0.927 vs. 0.900) and toothbrush (0.904 vs. 0.826). Carpet and hazelnut also finish fractionally above their references $( \leq 0 . 0 0 3 )$ , but both start within 0.01 of ceiling, where the margin carries no weight. Operator-curated insertion can therefore beat exhaustive training data on quality as well as on labelling cost. Hazelnut, bottle, zipper and carpet reach or exceed parity within a median of 4–19 reviews (medians over the seeds that reach parity; Table IV reports how many of the 20 do). Figure 5(a-c) shows the three characteristic trajectories: crossing the full-training baseline (metal nut), converging to parity (zipper), and still climbing when the budget ends (screw).

TABLE III  
COMPLETE 15-CATEGORY HITL RESULTS UNDER THE HELD-OUT PROTOCOL (MEANS OVER 20 SEEDS). HO = HELD-OUT SET, NEVER CORRECTED. BOLD IMPROVEMENTS ARE CONSISTENT ACROSS SEEDS (≥15/20 SEEDS IMPROVED); NEGATIVE IMPROVEMENTS ARE SHOWN AS-IS; THE UNGATED CORRECTION LOOP IS NOT BENEFICIAL ON EVERY CATEGORY. “GATED ∆" IS THE PASSIVE LOOP WITH NOVELTY-GATED INSERTION (SECTION III-B2), THE RECOMMENDED DEPLOYMENT CONFIGURATION. \* MARKS A ∆ SIGNIFICANTLY DIFFERENT FROM ZERO (TWO-SIDED WILCOXON SIGNED-RANK ON PER-SEED DELTAS, HOLM-CORRECTED ACROSS 15 CATEGORIES, $\alpha = 0 . 0 5 )$ ; IT IS REPORTED FOR THE PASSIVE AND GATED CONFIGURATIONS. ACTIVE DELTAS ARE NOT SEPARATELY TESTED AGAINST ZERO, BECAUSE NO CATEGORY SHOWS A SIGNIFICANT PASSIVE-VS-ACTIVE DIFFERENCE UNDER THE SAME TEST.
<table><tr><td>Category</td><td>Type</td><td>Sep.(σ)</td><td>Full-test Base.</td><td>HO Base.</td><td>Pass. HO Final</td><td>Pass. ∆</td><td>Act. ∆</td><td>Gated ∆</td></tr><tr><td>Toothbrush</td><td>Object</td><td>2.6</td><td>0.8139</td><td>0.8264</td><td>0.9722</td><td>+0.146*</td><td>+0.146</td><td>+0.101*</td></tr><tr><td>Grid</td><td>Texture</td><td>2.6</td><td>0.8839</td><td>0.8662</td><td>0.8190</td><td>-0.047*</td><td>-0.056</td><td>-0.031*</td></tr><tr><td>Screw</td><td>Object</td><td>3.0</td><td>0.8703</td><td>0.8703</td><td>0.9105</td><td> $\mathbf { + 0 . 0 4 0 ^ { * } }$ </td><td>+0.043</td><td> $\mathbf { + 0 . 0 4 8 ^ { \ast } }$ </td></tr><tr><td>Transistor</td><td>Object</td><td>3.7</td><td>0.9558</td><td>0.9623</td><td>0.9379</td><td>-0.024*</td><td>-0.031</td><td>+0.005</td></tr><tr><td>Metal Nut</td><td>Object</td><td>4.3</td><td>0.8803</td><td>0.9000</td><td>0.9980</td><td>+0.098*</td><td>+0.098</td><td>+0.089*</td></tr><tr><td>Pill</td><td>Object</td><td>4.3</td><td>0.9242</td><td>0.9283</td><td>0.9051</td><td>-0.023*</td><td>-0.026</td><td>+0.007</td></tr><tr><td>Capsule</td><td>Object</td><td>4.3</td><td>0.9290</td><td>0.9403</td><td>0.9310</td><td>-0.009</td><td>-0.012</td><td>+0.003</td></tr><tr><td>Cable</td><td>Object</td><td>4.6</td><td>0.9185</td><td>0.9260</td><td>0.9069</td><td>-0.019</td><td>-0.018</td><td>+0.006</td></tr><tr><td>Zipper</td><td>Object</td><td>4.9</td><td>0.8787</td><td>0.8772</td><td>0.9317</td><td>+0.054*</td><td>+0.054</td><td>+0.051*</td></tr><tr><td>Tile</td><td>Texture</td><td>6.0</td><td>0.9776</td><td>0.9806</td><td>0.9679</td><td>-0.013*</td><td>-0.012</td><td>-0.001</td></tr><tr><td>Hazelnut</td><td>Object</td><td>7.0</td><td>0.9964</td><td>0.9960</td><td>0.9985</td><td>+0.002</td><td>+0.002</td><td>+0.003</td></tr><tr><td>Wood</td><td>Texture</td><td>7.6</td><td>0.9912</td><td>0.9898</td><td>0.9940</td><td>+0.004</td><td>+0.004</td><td>+0.004</td></tr><tr><td>Carpet</td><td>Texture</td><td>9.5</td><td>0.9932</td><td>0.9930</td><td>0.9961</td><td>+0.003</td><td>+0.003</td><td>+0.003</td></tr><tr><td>Leather</td><td>Texture</td><td>23.1</td><td>1.0000</td><td>1.0000</td><td>1.0000</td><td>0.000</td><td>0.000</td><td>0.000</td></tr><tr><td>Bottle</td><td>Object</td><td>50.5</td><td>1.0000</td><td>1.0000</td><td>0.9991</td><td>-0.001</td><td>-0.001</td><td>-0.000</td></tr></table>

The cold-start result survives both protocol variations. Without the gate $~ ( k ~ ) ~ = ~ 1 0$ ungated), 11 of 15 categories are Holm-significantly positive with mean recovery 0.89 and still no significant negatives; with k = 20 (156 patches, higher cold baselines and smaller gaps), 8 of 15 remain significantly positive with mean recovery 0.85. Neither the choice of k nor the gate produces the effect. Even grid, the open failure of the full-training setting, is neutral here (+0.038, not significant), and the flat-to-negative group of Table III becomes uniformly positive: when the bank genuinely undersamples normal appearance, every correction teaches something. The honest limits: capsule recovers only 37% of its gap, screw exhausts the 90-round budget at half recovery, and pill reaches 61% after a mean of 77 reviews. On categories with large error pools the loop is label-efficient, not label-free.

## D. Key Category Analysis

We analyse the two instructive extremes: toothbrush, the largest success, and the flat-to-negative group, the method's boundary.

1) Toothbrush: Maximum Feedback Benefit: Toothbrush is the hardest category in our evaluation (separation index 2.6σ, baseline AUROC 0.8139) and produces the largest heldout improvement observed: $0 . 8 2 6 ~  ~ 0 . 9 7 2 ~ ( + 0 . 1 4 6 )$ on images the correction loop never touches, improving on 19 of 20 splits (worst split -0.028) with a mean of only \~13 corrections. Pool AUROC reaches 1.000 on every split, but per Section III-E4 that figure partly reflects memorisation of corrected images and should not be quoted as the result.

Figure 6 shows held-out ROC curves before and after correction at the median seeds: the corrected curves dominate at every operating point, so the gain is not an artefact of the AUROC summary. Why do so few reviews suffice? Clearing one falsely flagged image adds its 784 patches to a bank of only 470, which lowers nearest-neighbour distances for every image (held-out included) whose patches fall near the newly covered region. One label redefines a region of normal appearance; this transfer to held-out data is what separates correction from memorisation. The consequence is visible per round: on toothbrush and metal nut the entire gain arrives in one or two decisive FP corrections (on toothbrush a single round contributes +0.139) and most rounds contribute nothing, while screw, with a larger and noisier error pool, accumulates its gain gradually. The loop's value lies in reaching those images, not in correction volume. The FP/FN ablation (Section IV-I)

Held-Out Trajectories: Cold-Start Recovery (a-c) and the Effect of the Novelty Gate (d) (mean ±1 std over 20 seeds)  
![](images/9d85d84991004d36123c890c4643d6c9edb23c12f41c22fdf4ee9ad7d14e1acc.jpg)

![](images/395f25ca6245aa2ce17b28af2ea9a1a9c18394f48d69b0c763732560ffb14478.jpg)

![](images/da2985425317c9c961d22338a7e19530c5cea236837b1c4164a71c739f1e662a.jpg)

![](images/8e0db340f59c6d66f99c2949e237a82bbd89dd40d8d75ab6719592a184d87cb0.jpg)  
Fig. 5. Held-out trajectories (mean ±1 std over 20 seeds). (a-c) Cold-start recovery from 10 golden samples with the novelty gate: metal nut crosses its full-training baseline around round 50 and ends above it; zipper converges to just below parity after a transient early dip (corrections on a 78-patch bank briefly disturb scores before coverage accumulates); screw is still climbing when the 90-round budget ends, far from parity (the budget-limited case). (d) Pill on a fully trained bank: the ungated loop drifts below baseline while the novelty-gated loop stays above it, showing the gate's effect in a single panel.

TABLE IV  
COLD-START RESULTS: INITIAL BANK BUILT FROM ONLY k = 10 TRAINING NORMALS (78 PATCHES), PASSIVE QUERYING WITH NOVELTY-GATED INSERTION, 90-ROUND BUDGET, MEANS OVER 20 SEEDS. "FULL REF." = FULL-TRAINING-BANK HELD-OUT BASELINE ON THE SAME SEED (PAIRED). RECOVERY = FRACTION OF THE COLD-TO-FULL GAP CLOSED (SHOWN WHERE THE GAP EXCEEDS 0.02). "LABELS TO PARITY" = MEDIAN REVIEWS TO REACH THE FULL-TRAINING REFERENCE, WITH THE NUMBER OF SEEDS REACHING IT. \* = HOLM-SIGNIFICANT ∆ (WILCOXON, α = 0.05); NO CATEGORY IS SIGNIFICANTLY NEGATIVE.
<table><tr><td>Category</td><td>Cold Base</td><td>Post-HITL</td><td>Full Ref.</td><td>Δ</td><td>Recovery</td><td>Reviews</td><td>Labels to Parity</td></tr><tr><td>Bottle</td><td>0.957</td><td>0.996</td><td>1.000</td><td>+0.039*</td><td>0.92</td><td>14</td><td>5 (7/20)</td></tr><tr><td>Cable</td><td>0.756</td><td>0.866</td><td>0.926</td><td> $+ 0 . 1 1 0 ^ { * }$ </td><td>0.65</td><td>50</td><td>35 (1/20)</td></tr><tr><td>Capsule</td><td>0.560</td><td>0.699</td><td>0.940</td><td> $+ 0 . 1 3 9 ^ { * }$ </td><td>0.37</td><td>80</td><td>— (0/20)</td></tr><tr><td>Carpet</td><td>0.987</td><td>0.996</td><td>0.993</td><td> $+ 0 . 0 0 8 ^ { * }$ </td><td></td><td>18</td><td>8 (2/20)</td></tr><tr><td>Grid</td><td>0.681</td><td>0.719</td><td>0.866</td><td> $+ 0 . 0 3 8$ </td><td>0.20</td><td>37</td><td>24 (6/20)</td></tr><tr><td>Hazelnut</td><td>0.921</td><td>0.998</td><td>0.996</td><td> $+ 0 . 0 7 7 ^ { \ast }$ </td><td>1.03</td><td>16</td><td>4 (14/20)</td></tr><tr><td>Leather</td><td>1.000</td><td>1.000</td><td>1.000</td><td> $+ 0 . 0 0 0$ </td><td></td><td>19</td><td>— (0/20)</td></tr><tr><td>Metal Nut</td><td>0.677</td><td>0.927</td><td>0.900</td><td> $+ 0 . 2 5 0 ^ { * }$ </td><td>1.12</td><td>42</td><td>36 (15/20)</td></tr><tr><td>Pill</td><td>0.579</td><td>0.791</td><td>0.928</td><td> $+ 0 . 2 1 2 ^ { \ast }$ </td><td>0.61</td><td>77</td><td>22 (1/20)</td></tr><tr><td>Screw</td><td>0.467</td><td>0.671</td><td>0.870</td><td>+0.204*</td><td>0.50</td><td>90</td><td>— (0/20)</td></tr><tr><td>Tile</td><td>0.968</td><td>0.972</td><td>0.981</td><td>+0.004</td><td></td><td>26</td><td>4 (2/20)</td></tr><tr><td>Toothbrush</td><td>0.732</td><td>0.904</td><td>0.826</td><td>+0.172*</td><td>1.82</td><td>12</td><td>8 (15/20)</td></tr><tr><td>Transistor</td><td>0.656</td><td>0.857</td><td>0.962</td><td> $+ 0 . 2 0 2 ^ { \ast }$ </td><td>0.66</td><td>32</td><td>23 (1/20)</td></tr><tr><td>Wood</td><td>0.982</td><td>0.989</td><td>0.990</td><td> $+ 0 . 0 0 7 ^ { \ast }$ </td><td></td><td>10</td><td>— (0/20)</td></tr><tr><td>Zipper</td><td>0.750</td><td>0.864</td><td>0.877</td><td> $+ 0 . 1 1 4 ^ { * }$ </td><td>0.89</td><td>44</td><td>19 (9/20)</td></tr></table>

Held-Out ROC Curves Before and After HITL Correction  
![](images/16e6172cc2f67aa2298b08a277576f0059d7750f813719f8592afed1bebad6e7.jpg)

![](images/6fb05690f81f5acb6a26ba554c1ce5de2d609602a22e66c42593432d241b8892.jpg)

![](images/b32ff2a80fd0a5949aee2b9f132c11611884b1f02f68b8141bbff21320409d66.jpg)  
Fig. 6. Held-out ROC curves before (grey) and after correction under passive (blue) and active (red dashed) querying, for the three largest-improvement categories at their median-improvement seed. The corrected ROC dominates the baseline everywhere; passive and active endpoints are nearly identical.

and the matched-budget control (Section IV-H) complete the mechanism account.

2) Where Correction Fails: The Flat-to-Negative Group: The six flat-to-negative categories share a profile: their banks are comparatively large, their remaining errors genuinely ambiguous, so FP corrections add patches the bank already covers, and the redundant insertion perturbs previously correct decisions. Grid is the clearest case (-0.047, degrading on 15/20 splits). The novelty gate (Section IV-B) resolves five of the six by refusing exactly these redundant insertions. Grid alone remains significantly negative when gated (—0.031): it inserts 230 patches per correction yet still degrades, so its failure is not a coverage problem. It is the method's one open failure in the mature-bank setting, though it is unharmed at cold start (Table IV). The practical rule: enable the gate by default and confirm per category with a held-out check.

## E. Strategy Comparison

Passive and active querying are statistically indistinguishable: a paired Wilcoxon test on per-seed held-out endpoints finds no significant difference on any category, and mean improvements agree within ±0.01 AUROC on all 15 categories (Table III); the residual differences sit well inside seed-to-seed variance, and the per-seed distributions of the two strategies overlap heavily on every category (Section IV-G). Mean held-out trajectories with 95% confidence bands overlap at every round on 14 of 15 categories (Fig. 7). The one exception concerns speed rather than destination: on metal nut, active querying reaches the ceiling in 4 rounds versus 9 for passive (per-seed means; mean AUCC 0.88 vs. 0.71), the only stretch where the bands separate, yet both converge to the same endpoint. Note that time-to-ceiling is shorter than session length: metal nut sessions run a mean of \~20 reviews, but the gain is complete well before the error pool is exhausted. An earlier single-seed finding that passive wins on 9 of 11 categories does not survive this evaluation, and we retract it; the mechanism (global bank updates plus a budget that exhausts the error pool under either ordering) is discussed in Section V.

Held-out AUCC follows the same pattern as the AUROC

Held-Out AUROC over Correction Rounds: Passive vs Active Querying (mean over 20 seeds; bands = 95% CI of the mean)

![](images/ec17abc5d5b35ecee4254c18b0422968cbc84210b9842e2ad0f812e90dc4a569.jpg)  
Fig. 7. Held-out AUROC over correction rounds for passive (blue) and active (red) querying, on six categories spanning the difficulty range (mean over 20 seeds; bands are the 95% confidence interval of the mean, t-based with $n = 2 0 { \mathrm { . } }$ not the seed-to-seed spread). The bands overlap at every round except on metal nut, where active querying rises faster over the first few rounds before both strategies converge to the same endpoint: a difference in speed, not destination.

deltas: correction utility concentrates in the low-σ categories with FP-dominated error pools (Table III), and passive and active AUCC agree on every category except metal nut.

## F. Ablation Study: $k _ { r e m o \nu e }$ Parameter

Ablating $k _ { \mathrm { r e m o v e } } ,$ the number of bank entries removed per FN correction, on toothbrush across $k \in \{ 1 , 3 , 5 , 9 \}$ gives identical results in every case (AUCC 0.9984, final AUROC 1.0000), because that session drew only FP corrections and the parameter never triggered. (Single seed 42, pool-only evaluation, predating the held-out protocol; the conclusion concerns the correction mechanism, not the split.) Toothbrush sessions are not generally FN-free: under the held-out protocol they average 5.0 FN corrections (Table V), which Section IV-I shows to be near no-ops. We use k = 5 throughout, large enough to open a gap in the manifold and small enough to avoid over-removal.

## G. Multi-Seed Validation

Every result in this paper is a mean over 20 independent pool/held-out splits (seeds 10-29); multi-seed validation is therefore built into the protocol rather than performed post hoc on selected categories. Figure 8 shows the full per-seed distribution of held-out improvement for every non-ceiling category and both strategies.

Three observations follow. First, the seed-consistent winners (toothbrush 19/20, metal nut 20/20, zipper 20/20, screw 15/20 seeds improved) are robust well beyond noise. Second, the flatto-negative categories are consistently flat-to-negative (grid degrades on 15/20 seeds), so their failure is systematic rather than sampling misfortune. Third, on hard categories a single seed can land anywhere within a third of the AUROC scale (toothbrush single-seed improvements range from —0.03 to +0.31), which explains why earlier single-seed comparisons of querying strategies produced unstable conclusions: individual seeds can and do flip the passive-vs-active winner. Any claim in this domain based on a single split should be treated as anecdote.

![](images/1c08855db792a6ff145ab4c36bb1a4e26939eaf80c196bcf58e332ef7e08e62f.jpg)  
Fig. 8. Per-seed distributions of held-out AUROC improvement (20 seeds per category; blue = passive, red = active). Toothbrush, metal nut, zipper and screw improve consistently; grid, transistor, pill, capsule, cable and tile centre at or below zero. Passive and active distributions overlap heavily on every category.

## H. Label Economics: Is the Gain Just Extra Training Data?

Since FP corrections append correctly-labelled normal images to the bank, does the held-out gain merely reflect data volume? We test this on toothbrush with matched budgets (20 confirmation seeds): the HITL loop reaches held-out AUROC 0.972 with 12.6 labels on average; labelling the same number of randomly chosen pool images reaches only 0.901; appending all pool normals (the more-data ceiling) reaches 0.974, but costs \~29 labels, because every pool image must be reviewed to establish which ones are normal. So yes, the gain is delivered through added labelled data, but the loop is what makes that data obtainable. At deployment, unlabelled data cannot be added safely (one mislabelled defect poisons the bank), so the reviewer is the only source; and errortriaged review reaches the more-data ceiling at 43% of its labelling cost, because random labelling wastes most labels on defects, where corrections are near no-ops. We claim not that the loop beats equivalent training data, but that it is the mechanism by which such data becomes obtainable, cheaply, after deployment.

## I. Ablation Study: FP vs. FN Correction

To isolate the contributions of FP and FN corrections, we rerun the toothbrush sessions with the reviewer restricted to a single error type (FP-only or FN-only, the skipped type excluded from review entirely) under the full held-out protocol (20 seeds, passive querying). Table V reports the result.

The asymmetry is stark. FP-only correction reproduces the entire improvement: its final held-out AUROC equals the combined loop's on every one of the 20 seeds (+0.146, Wilcoxon vs. zero $p < 1 0 ^ { - 4 } )$ , and it arrives in fewer reviews (7.5 vs. 12.6; AUCC 0.709 vs. 0.563), because no budget is spent on FN reviews that do not change the destination. FN-only correction achieves nothing: a mean of 8.8 removals per session leaves held-out AUROC at baseline on 13 of 20 seeds, with a residual mean that is slightly negative (—0.013, unadjusted $\begin{array} { c c l } { p } & { = } & { 0 . 0 3 4 ) } \end{array}$ . Removal is, if anything, mildly harmful here. Two causes compound. The first is the bank itself: toothbrush's errors come from gaps in normal coverage, which FP insertion fills directly, whereas FN removal deletes entries from a bank that is already too sparse. The second is a metric mismatch: the removal anchor is selected by 1-NN distance (Eq. 5) while the image score is the mean-of-nine statistic (Eq. 1), so the excised neighbourhood is frequently not the one that set the score. Removal is therefore both intrinsically risky on a minimal coreset and often aimed at the wrong location. Figure 9 extends the picture to all categories under the multi-seed protocol: every seed-consistent winner has an FP-dominated error pool.

TABLE V  
FP VS. FN CORRECTION ABLATION, TOOTHBRUSH PASSIVE, HELD-OUT PROTOCOL (MEANS OVER 20 SEEDS). RESTRICTED MODES EXCLUDE THE SKIPPED ERROR TYPE FROM REVIEW; ∆ IS HELD-OUT FINAL MINUS BASELINE (0.826).
<table><tr><td>Mode</td><td>HO Final</td><td>Δ</td><td>HO AUCC</td><td>FP/FN corr.</td></tr><tr><td>FP-only</td><td>0.9722</td><td>+0.146</td><td>0.709</td><td>7.5/0</td></tr><tr><td>Both</td><td>0.9722</td><td>+0.146</td><td>0.563</td><td>7.6/5.0</td></tr><tr><td>FN-only</td><td>0.8139</td><td>-0.013</td><td>0.004</td><td>0/8.8</td></tr></table>

![](images/57284ca2f3eb5431f6b28f2903cf55acc94dff0be749386dd418f0669c33a3e5.jpg)  
Fig. 9. FP vs. FN correction mix across all 15 categories (mean counts over 20 seeds). Blue = FP, red = FN. Bottle and leather are ceiling controls.

1) Why the Anchor Metric Is Not “Fixed": Protocol note: like the other parameter ablations, this investigation predates the held-out protocol and was run under pool-only evaluation at a single seed (42).

Section III notes that the removal anchor (Eq. 5, 1-NN) and the image score (Eq. 1, mean-of-nine) are different metrics, so removal frequently excises a neighbourhood other than the one that set the score. The obvious remedy is to re-anchor removal on the mean-of-nine patch that actually drives the score. We tested exactly that, on an FN-only correction sequence on toothbrush with $k _ { \mathrm { r e m o v e } }$ swept over {1, 2, 3, 5, 8}. Re-anchoring does give FN correction a real, non-zero effect, and that effect is destructive: every non-zero removal size collapses AUROC by roughly the same amount, from a baseline of 0.811 to a final 0.54–0.56, independent both of the anchor metric and of how many points are removed per correction.

The cause is the coreset itself. A false negative scores low precisely because its worst patch sits close to a legitimate normal bank entry; removing that neighbour to fix the one image also pushes away every other normal-looking patch in the same region, including patches from genuinely normal test images. On a bank that is already a minimal 1% coreset (470 patches for toothbrush) there is no safe removal size, so this is not a magnitude-tuning problem. We therefore retain the 1-NN anchor deliberately. Its mismatch with the scoring metric is what keeps FN correction close to a no-op, and on a minimal coreset a no-op is strictly preferable to a reliable means of destroying the bank. An effective FN mechanism would have to be additive rather than subtractive; the defect-memory pilot (Section V-B) tests the most obvious additive design and also fails.

## V. DISCUSSION

## A. Why Querying Strategy Barely Matters

The strategy-equivalence result (Section IV-E) contradicts the standard active learning assumption, and the mechanism explains why. Uncertainty sampling assumes updates are local: a boundary-adjacent sample mainly improves the boundary near itself, so sample choice matters. Memory bank corrections are global: inserting a false positive's patches changes nearestneighbour distances for every image, and the review budget exhausts the eligible error pool under any ordering. When every error is eventually corrected and every correction acts globally, order contributes nothing, so uncertainty sampling buys nothing. An earlier single-seed analysis had instead found passive querying winning on 9 of 11 categories; multi-seed validation (Section IV-G) exposed that as seed noise, and we retract it as a cautionary example for evaluation practice in this domain.

We also tested a centroid-distance query rule (FP: most centroid-distant normal; FN: most centroid-proximal anomaly) in single-seed pool-protocol experiments predating the heldout protocol: it matched passive on toothbrush, underperformed on metal nut, and was severely counterproductive on cable, in each case through a strong FP-selection bias. Query strategies that measurably beat random ordering for memory bank correction remain an open problem.

## B. Negative Result: Defect Memory Bank

Could FN corrections contribute symmetrically, by memorising operator-confirmed defect patches in a second bank and scoring images by the margin between distance-to-normal and distance-to-defect? A pre-registered pilot (pill, screw, cable; 20 seeds; passive) answers no, decisively: held-out AUROC collapses on pill (—0.226) and screw (—0.299, worse than the standard loop on 20/20 seeds; both Holm $p < 1 0 ^ { - 4 } )$ , with mid-run dips of 0.26–0.49 AUROC, while cable is indistinguishable from the standard loop. A defect bank built from \~28 rejected images over-generalises, pulling large regions of normal appearance toward “defect." Per the pre-registered gate, no full sweep was run. Together with the FP/FN ablation (Section IV-I), where FN removal is a measured near no-op, this fixes the paper's central mechanism claim: memory bank correction works by teaching normality, not by memorising defects.

## C. Bank Growth, and Why the Cap Is Not the Problem

Bank growth over correction rounds is largest exactly where correction helps most: toothbrush grows \~14× from its 470- patch original bank, because nearly every inserted patch covers new manifold. For the flat-to-negative group, one hypothesis is the opposite failure: that the cap's random subsampling churns away accumulated corrections. A 20-seed control with the cap removed refutes it: no held-out delta changes except transistor's (+0.021). The failures come from the corrections themselves, through redundant insertion into already-covered banks, which is precisely the failure mode the novelty gate removes (Section IV-B).

## D. What the Loop Cannot Fix

Two structural limits survive gating. First, globally-defined defects: cable's cable\_swap is locally identical to a normal cable (only the arrangement is wrong), and patch-level correction cannot encode global structure. Second, grid: it inserts 230 patches per correction yet still loses 0.031 AUROC in the mature-bank setting (Section IV-D2), so its errors are not a coverage problem; it is the method's one open failure, though it is unharmed at cold start. A general warning follows from the held-out protocol: on every failing category the pool metric still rises, so an operator watching in-loop accuracy would believe the loop is working while generalisation worsens. Monitoring must use never-corrected images.

## E. Practical Deployment Guidelines

Who: a quality engineer who already reviews model output. The expert contributes one binary label per reviewed image and needs no ML knowledge, Python, or training infrastructure.

When: enable the novelty gate by default; with it, no MVTec category except grid is significantly harmed, so the loop is safe to leave on. Gains are largest where errors are FPdominated from an undersampled normal manifold (+0.05 to +0.10 AUROC gated), and largest of all at cold start, where they reach +0.25 (Section IV-C). Confirm per category with a held-out check; never gate on in-loop accuracy (see above). Near-ceiling categories need no correction.

How: passive random querying, since active querying adds computation and configuration for no measurable benefit. Effective categories exhaust their errors in 13-20 corrections (screw, with its larger error pool, uses the full 30-round budget), i.e. roughly 13-20 minutes of review at one image per minute. Monitor a held-out sample and stop if it trends downward.

Infrastructure: none beyond the deployed model. Bank edits are CPU operations; no training data, no optimisation framework, no ML team.

## VI. CONCLUSION

We presented a training-free human-in-the-loop framework in which a quality engineer corrects a deployed anomaly detector by reviewing misclassified images: one binary decision per image, applied as a direct memory bank edit. Evaluation across all fifteen MVTec AD categories yields five findings.

First, the strongest deployment case is cold start. From a bank built on ten golden samples, corrections close a median 66% of the gap to an uncorrected fully trained bank (mean 80%, raised by three categories that overshoot parity), significantly improving 12 of 15 categories and harming none. Ten samples plus corrections outperform hundreds of samples without them. We are careful not to overstate this: a corrected full-training bank still finishes higher on the same categories (toothbrush 0.927 vs. 0.904, metal nut 0.989 vs. 0.927), so the result is about what is achievable on day one of a deployment, not about curation dominating data.

Second, on mature banks correction is effective where the bank undersamples normal appearance, and only there. On held-out images the loop never touches, toothbrush improves from AUROC 0.826 to 0.972 in a mean session of roughly 13 single-decision reviews and metal nut from 0.900 to 0.998 in roughly 20, consistently across 20 splits. At one image per minute, that is 13-20 minutes of review. Those endpoints are for the ungated loop; the novelty-gated configuration we recommend for deployment trades some of the gain for safety, reaching 0.927 and 0.989 respectively (third finding below). Five categories sit at ceiling and cannot move so the mature-bank gains are confined to four of fifteen.

Third, honest evaluation requires a held-out protocol, and acting on it produced the novelty gate. Corrected images enter the bank, so evaluation on reviewed images inflates toward AUROC 1.0 by memorisation. The held-out protocol reveals six categories where the raw loop is flatto-negative; diagnosing the cause (redundant insertion into already-covered banks) yielded the novelty gate, which eliminates the degradation on five of the six, preserves every winner, and leaves grid as the single open failure.

Fourth, querying strategy barely matters. Passive and active querying are statistically indistinguishable on every category: bank edits act globally, and the budget exhausts the error pool under either ordering. An earlier single-seed finding that passive wins on 9 of 11 categories did not survive multiseed validation; we retract it explicitly, alongside negative results for centroid-distance querying, for re-anchoring FN removal on the scoring metric, and for a defect-memory second bank. The mechanism is teaching normality, cheaply and in any order.

Fifth, the gains equal what the same labels would achieve at training time. The framework's contribution is not outperforming training data but producing correctly-labelled data at deployment, where unlabelled data cannot be added safely, at 43% of the labelling cost of exhaustive review, and applying it without retraining.

## A. Limitations

Our evaluation uses a research implementation on MVTec AD. With per-session feature caching, per-round compute is seconds on hard categories, rising toward a minute as the bank grows; production deployment would use approximate nearest-neighbour indexing. All headline results are means over 20 pool/held-out splits per category (Section III-E4); the centroid-querying investigation and the $k _ { \mathrm { r e m o v e } }$ parameter ablation predate this protocol and are marked as single-seed, poolprotocol results. Held-out sets are small on some categories (\~13 images for toothbrush), so single-split AUROCs are noisy: read means and seed win-counts (Fig. 8), not individual splits. Correction is image-level only; pixel-level correction remains future work.

## B. Future Work

• Batched correction: applying several corrections before rescoring may stabilise complex categories such as cable.

• Production-speed implementation: approximate nearest-neighbour indexing (FAISS, HNSW) and incremental rescoring would cut per-round latency to seconds, leaving expert review time (\~1 minute per image) as the only bottleneck.

• Pixel-level correction: letting the expert mark the anomalous region, not just the image label.

• Cross-category transfer: corrections for one product variant may transfer to related variants.

• Foundation model integration: applying the correction loop on top of zero-shot detectors [34], [41].

• Live human trial: replacing simulated feedback with genuine domain experts (Section VI-C) is the necessary next validation step.

## C. Threats to Validity

Simulated feedback: the expert is simulated from MVTec ground-truth labels and is always right. Real reviewers disagree with ground truth in ambiguous cases, fatigue, and drift; our results are an upper bound under perfect feedback, and a live trial is required before industrial claims.

Operator error is costliest at cold start: relaxing the perfect-feedback assumption does not cost the same in every regime. A single “normal" label on a true defect inserts up to 784 defect patches into the normal bank. Against a mature bank near the 12,000-patch cap that is a few per cent contamination; against a 78-patch cold-start bank it is an order of magnitude more feature mass than the bank itself Our strongest result is therefore also the one most exposed to imperfect feedback, and a live trial should establish an operator error rate, and the value of an escalation option that applies no correction, before the small-bank regime is deployed.

Single dataset: MVTec AD spans 15 diverse categories but not the full range of factory conditions (lighting variation, camera drift, novel defect types). Findings should be revalidated on proprietary data before being treated as general.

Mechanism specificity: direct bank editing applies to memory-bank detectors (PatchCore and coreset-based relatives). Reconstruction-, flow-, or classifier-based detectors expose no equivalent editable structure; we claim nothing beyond the memory-bank family.

Timing: reported wall-clock times are dominated by research-grade GPU rescoring. “Minutes of review" describes the human component and assumes the production-speed implementation above.

Prior work has modified PatchCore banks automatically; what this paper establishes is different in kind: the person who already reviews the errors can now fix them immediately, without an ML team. Where a formal retraining cycle takes weeks to months, a correction session takes 13-20 minutes. That shift, in both who can correct the model and how fast, is the framework's most useful property.

## DATA AND CODE AVAILABILITY

All experiments use the publicly available MVTec AD benchmark [2], obtainable from the MVTec Software GmbH website under its research licence; no proprietary or personal data were used. The correction-framework implementation and the experiment scripts that reproduce every reported result are available from the corresponding author on reasonable request.

## ACKNOWLEDGMENT

This research received no external funding.

## REFERENCES

[1] K. Roth, L. Pemula, J. Zepeda, B. Schölkopf, T. Brox, and P. Gehler, “Towards total recall in industrial anomaly detection," in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022, pp. 14318–14 328.

[2] P. Bergmann, M. Fauser, D. Sattlegger, and C. Steger, "MVTec AD — a comprehensive real-world dataset for unsupervised anomaly detection," in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019, pp. 9592–9600.

[3] Warranty Week, "Worldwide auto warranty report 2024," 2024, available: https://www.warrantyweek.com/archive/ww20241003.html.

[4] D. Hillier, “The most expensive defect," Semiconductor Digest, December 2014, available: https://sst.semiconductor-digest.com/2014/12/ the-most-expensive-defect/.

[5] MHRA, "Pharmaceutical company fined for manufacturing defective medicine," 2022, UK Government Press Release, March 2022.Available: https://www.gov.uk/government/news/ pharmaceutical-company-fined-for-manufacturing-defective-medicine.

[6] N. Bugarin, J. Bugaric, M. Barusco, D. Dalle Pezze, and G. A. Susto, “Unveiling the anomalies in an ever-changing world: A benchmark for pixel-level anomaly detection in continual learning," in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), 2024, pp. 4065–4074.

[7] Z. Jiang, Y. Zhang, Y. Wang, J. Li, and X. Gao, "FR-PatchCore: An industrial anomaly detection method for improving generalization," Sensors, vol. 24, no. 5, p. 1368, 2024.

[8] G. Pang, C. Shen, L. Cao, and A. van den Hengel, "Deep learning for anomaly detection: A review," ACM Computing Surveys, vol. 54, no. 2, pp. 1–38, 2021.

[9] T. Schlegl, P. Seeböck, S. M. Waldstein, U. Schmidt-Erfurth, and G. Langs, “Unsupervised anomaly detection with generative adversarial networks to guide marker discovery," in International Conference on Information Processing in Medical Imaging (IPMI), ser. Lecture Notes in Computer Science, vol. 10265. Springer, 2017, pp. 146–157.

[10] P. Napoletano, F. Piccoli, and R. Schettini, “Anomaly detection in nanofibrous materials by CNN-based self-similarity," Sensors, vol. 18, no. 1, p. 209, 2018.

[11] G. Wang, S. Han, E. Ding, and D. Huang, "Student-teacher feature pyramid matching for anomaly detection," in Proceedings of the British Machine Vision Conference (BMVC), 2021.

[12] N. Cohen and Y. Hoshen, “Sub-image anomaly detection with deep pyramid correspondences," 2020, arXiv:2005.02357.

[13] T. Defard, A. Setkov, A. Loesch, and R. Audigier, "PaDiM: A patch distribution modeling framework for anomaly detection and localization," in Pattern Recognition. ICPR International Workshops and Challenges, ser. Lecture Notes in Computer Science, vol. 12664. Springer, 2021, pp. 475–489.

[14] Z. Liu, Y. Zhou, Y. Xu, and Z. Wang, "SimpleNet: A simple network for image anomaly detection and localization," in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023, pp. 20 402–20 411.

[15] R. Munro Monarch, Human-in-the-Loop Machine Learning: Active Learning and Annotation for Human-Centered AI. Shelter Island, NY: Manning Publications, 2021.

[16] B. Settles, “Active learning literature survey," University of Wisconsin— Madison, Tech. Rep. 1648, 2010.

[17] J. M. Rožanec, E. Montini, V. Cutrona, D. Papamartzivanos, T. Klemenčič, B. Fortuna, D. Mladenić, E. Veliou, T. Giannetsos, and C. Emmanouilidis, “Human in the AI loop via xAI and active learning for visual inspection," in Artificial Intelligence in Manufacturing: Enabling Intelligent, Flexible and Cost-Effective Production Through AI, J. Soldatos, Ed. Springer, 2023, pp. 381–406.

[18] Z. Deng, X. Xuan, K.-L. Ma, and Z. Kong, "A reliable framework for human-in-the-loop anomaly detection in time series," ACM Transactions on Interactive Intelligent Systems, 2024, arXiv:2405.03234.

[19] S. Das, M. R. Islam, N. K. Jayakodi, and J. R. Doppa, "Active anomaly detection via ensembles," arXiv preprint arXiv:1809.06477, 2018.

[20] —, “Effectiveness of tree-based ensembles for anomaly discovery: Insights, batch and streaming active learning," Journal of Artificial Intelligence Research, vol. 80, pp. 127–172, 2024.

[21] H. Bodor, T. V. Hoang, and Z. Zhang, "Little help makes a big difference: Leveraging active learning to improve unsupervised time series anomaly detection," in Proceedings of the AIOPS Workshop, 19th International Conference on Service-Oriented Computing (ICSOC), 2021.

[22] H. Vishwakarma, H. Lin, and R. K. Vinayak, “Taming false positives in out-of-distribution detection with human feedback," in Proceedings of the 27th International Conference on Artificial Intelligence and Statistics (AISTATS), ser. Proceedings of Machine Learning Research, vol. 238, 2024, pp. 1486–1494.

[23] D. Yamada, H. Vishwakarma, and R. K. Vinayak, "Adaptive scoring and thresholding with human feedback for robust out-of-distribution detection," arXiv preprint arXiv:2505.02299, 2025.

[24] D. D. Lewis and W. A. Gale, “A sequential algorithm for training text classifiers," in Proceedings of the 17th Annual International ACM SIGIR Conference on Research and Development in Information Retrieval, 1994, pp. 3–12.

[25] H. S. Seung, M. Opper, and H. Sompolinsky, "Query by committee," in Proceedings of the Fifth Annual Workshop on Computational Learning Theory (COLT), 1992, pp. 287–294.

[26] M. De Lange, R. Aljundi, M. Masana, S. Parisot, X. Jia, A. Leonardis, G. Slabaugh, and T. Tuytelaars, “A continual learning survey: Defying forgetting in classification tasks," IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 44, no. 7, pp. 3366–3385, 2022.

[27] D. Wang, E. Shelhamer, S. Liu, B. Olshausen, and T. Darrell, "Tent: Fully test-time adaptation by entropy minimization," in International Conference on Learning Representations (ICLR), 2021.

[28] W. Li, J. Zhan, J. Wang, B. Xia, B.-B. Gao, J. Liu, C. Wang, and F. Zheng, “Towards continual adaptation in industrial anomaly detection," in Proceedings of the 30th ACM International Conference on Multimedia, 2022.

[29] G. Yang, Z. Deng, and J. Man, "CADIC: Continual anomaly detection based on incremental coreset," arXiv preprint arXiv:2511.08634, 2025.

[30] M. Dahmardeh and F. Setti, "MECAD: A multi-expert architecture for continual anomaly detection," in Proceedings of the International Conference on Image Analysis and Processing (ICIAP), 2025.

[31] W. Fang, H. Che, F. Ren, and Q. Lao, "Normality-preserving continual industrial anomaly detection via orthogonal LoRA banks," arXiv preprint arXiv:2606.02042, 2026

[32] C. S. Han and K. M. Lee, “Leveraging intermediate representations of time series foundation models for anomaly detection," arXiv preprint arXiv:2509.12650, 2025.

[33] C. Huang, H. Guan, A. Jiang, Y. Zhang, M. Spratling, and Y.-F. Wang, "Registration based few-shot anomaly detection," in Proceedings of the European Conference on Computer Vision (ECCV), 2022.

[34] J. Jeong, Y. Zou, T. Kim, D. Zhang, A. Ravichandran, and O. Dabeer, “"WinCLIP: Zero-/few-shot anomaly classification and segmentation," in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023, pp. 19 606–19 616.

[35] G. Xie, J. Wang, J. Liu, Y. Jin, and F. Zheng, "Pushing the limits of fewshot anomaly detection in industry vision: GraphCore," in International Conference on Learning Representations (ICLR), 2023.

[36] Z. Fang, X. Wang, H. Li, J. Liu, Q. Hu, and J. Xiao, "FastRecon: Fewshot industrial anomaly detection via fast feature reconstruction," in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2023, pp. 17 435–17 444

[37] S. Zagoruyko and N. Komodakis, "Wide residual networks," in Proceedings of the British Machine Vision Conference (BMVC). BMVA Press, 2016, pp. 87.1–87.12.

[38] J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei, "ImageNet: A large-scale hierarchical image database," in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2009, pp. 248–255.

[39] O. Sener and S. Savarese, “Active learning for convolutional neural networks: A core-set approach," in International Conference on Learning Representations (ICLR), 2018.

[40] I. Guyon, G. C. Cawley, G. Dror, and V. Lemaire, “Results of the active learning challenge," in Active Learning and Experimental Design Workshop (AISTATS 2010), ser. Proceedings of Machine Learning Research, vol. 16, 2011, pp. 19–45.

[41] Z. Gu, B. Zhu, G. Zhu, Y. Chen, M. Tang, and J. Wang, “AnomalyGPT: Detecting industrial anomalies using large vision-language models," in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 38, 2024, pp. 1932–1940.