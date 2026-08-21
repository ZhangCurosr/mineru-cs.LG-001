# Interpretable Feature Learning for RF Fingerprinting via Polar MKANs

Mikhail Krasnov<sup>†</sup>, Ljupcho Milosheski<sup>†</sup>, Carolina Fortuna

Jožef Stefan Institute, Ljubljana, Slovenia

{mikhail.krasnov, ljupcho.milosheski, carolina.fortuna}@ijs.si

Abstract—Radio frequency (RF) fingerprinting authenticates wireless devices from hardware-induced I/Q impairments, typically with deep-learning feature extractors that are accurate but opaque, limiting their use in security-critical settings. We propose Polar Monotonic Kolmogorov–Arnold Networks (Polar MKAN), a block-partitioned monotonic encoder on polar inputs in which each latent dimension depends exclusively on magnitude or phase, yielding channel separation and monotone responses by construction. On a synthetic gain/carrier frequency offset (CFO) benchmark, Polar MKAN reaches 57.2% DCI Disentanglement versus ≤ 12.9% for unpartitioned baselines. We further evaluate the detection-accuracy trade-off on real data and the sensitivity to blind CFO compensation.

Index Terms—RF fingerprinting, unknown emitter detection, interpretability, Kolmogorov–Arnold networks, monotonic networks, self-supervised learning, disentangled representations

## I. INTRODUCTION

RF fingerprinting exploits hardware-level imperfections such as carrier frequency offset (CFO), I/Q imbalance, and power-amplifier nonlinearities [1], to authenticate wireless devices without cryptographic overhead [2], [3]. Unknown emitter detection (UED) extends this capability to open-set scenarios in which previously unseen devices must be identified at inference time [4], [5]. Self-supervised approaches based on DL architectures [6], [7] are particularly suitable for this task due to their ability to learn from large amounts of unlabelled data, and recent work shows strong UED performance from such self-supervised feature spaces [8]. However, UED systems based on standard DL architectures remain opaque and intractable for an operator due to the large number of parameters standing between the latent features and any interpretable signal property. For security-sensitive applications such as spectrum monitoring and critical-infrastructure protection, this lack of transparency is a deployment barrier [9]. Existing approaches to UED interpretability rely on post-hoc methods such as gradient saliency [10], LIME [11], and SHAP [12], which provide local, sample-specific explanations without structural guarantees about global consistency. A complementary thread pursues disentangled representation learning for RF fingerprinting [13]–[15], separating device-relevant information from channel, receiver, modulation, and other nuisance factors through adversarial or regularised objectives. These methods share a fundamental limitation, namely that disentanglement is encouraged but not guaranteed in general [16], and the separation quality depends on training dynamics that cannot be verified at inference time. Kolmogorov–Arnold networks (KANs) [17], [18] offer an architectural route, since their visible spline functions enable global interpretability [8], but non-monotonic activations still yield complex input/output mappings. MKANs [19] enforce hard monotonicity constraints by making each spline activation monotonic, which fixes the sign of every input’s effect on every output and so makes the input/output mapping predictable. Certified-monotonic alternatives such as MonoKAN [20] exist, but we build on MKAN since its representation-learning guarantee (Theorem 1 of [19]) is established in the deep-clustering setting, certifying that a monotone extractor preserves the neighbourhood structure the clustering-based UED decision relies on. To the best of our knowledge, no prior work achieves an architecturally enforced separation of the magnitude and phase information channels in the RF-fingerprinting feature space. Building on the MKAN construction and UED workflow of [8], [19], this letter’s new contributions are the polar input representation and the blockpartitioned output, summarized as follows.

• An encoder whose latent space is split by construction, with dedicated latent dimensions wired exclusively to the magnitude and to the principal-phase inputs, so each feature reflects exactly one of the two physical channels.

• A synthetic-benchmark evaluation with known groundtruth device impairments, on which Polar MKAN recovers the magnitude/phase separation (57.2% DCI Disentanglement [21], [22] vs. ≤ 12.9% for unconstrained and polar-input-only baselines), and a quantified interpretability-performance trade-off on the UED task.

• A perturbation analysis on real WiSig [23] signals showing that the phase channels respond monotonically and stay separated from the magnitude channel, so a featurespace deviation decomposes into a magnitude- and a phase-channel component.

## II. BACKGROUND AND SYSTEM MODEL

We adopt the workflow of [8] depicted also in Fig. 1. A data modality (raw I/Q or polar coordinates) is mapped by a feature extractor (FE) $\bar { F : \mathbb { R } ^ { 2 N } } \to \mathbb { R } ^ { N ^ { * } }$ to a low-dimensional feature space. The FE is trained as the encoder of a self-supervised autoencoder, and at inference a cluster-based decision module compares test features with training clusters for UED. We keep the training and decision procedure fixed and modify only the FE and its input representation.

(a) Architecture  
![](images/fb7cf20fcd6fb7787878b111fdc6488fef6999dd7b32c6d9373115f4dd1b9c8f.jpg)  
Fig. 1: Self-supervised UED workflow.

## A. Monotonic KAN

A KAN layer [17] replaces fixed multi-layer perceptron (MLP) activations with learnable B-spline functions $\phi _ { i j } ( x _ { i } )$ on each edge, giving

$$
F ^ { ( j ) } ( \mathbf { x } ) = \sum _ { i } w _ { i j } ^ { ( s ) } \phi _ { i j } ( x _ { i } ) + w _ { i j } ^ { ( b ) } \mathrm { S i L U } ( x _ { i } ) .\tag{1}
$$

MKANs [19] constrain spline coefficients to be ordered, reparameterize connected weights through exponentiation, and replace the non-monotone SiLU base with ReLU, giving

$$
\widetilde { F } ^ { ( j ) } ( \mathbf { x } ) = \sum _ { i } e ^ { \widetilde { w } _ { i j } ^ { ( s ) } } \widetilde { \phi } _ { i j } ( x _ { i } ) + e ^ { \widetilde { w } _ { i j } ^ { ( b ) } } \mathrm { R e L U } ( x _ { i } ) + b _ { j } ,\tag{2}
$$

where $\widetilde { \phi } _ { i j }$ are monotone splines. Hence an output is componentwise nondecreasing in every input to which it is connected, while disconnected inputs have zero effect. Theorem 1 of [19] shows that an increasing FE can reproduce the semanticneighbourhood partition of an arbitrary FE with at most $2 \times$ feature dimensionality.

## B. Proposed Polar MKAN

For $x ^ { I } [ i ] + j x ^ { Q } [ i ]$ , we form principal-polar coordinates

$$
x ^ { I } [ i ] + j x ^ { Q } [ i ] = R [ i ] e ^ { j \phi [ i ] } , \qquad \phi [ i ] \in [ 0 , 2 \pi ) .\tag{3}
$$

The implementation feeds polar models the fixed numerical scaling $\widetilde { R } [ i ] \ : = \ : R [ i ] / 1 . 5$ and $\widetilde { \phi } [ i ] ~ = ~ \phi [ i ] / ( 2 \pi )$ . Only Polar MKAN imposes exclusive output connectivity, as illustrated in Fig. 2a, where $F _ { R }$ receives the magnitude block $\{ \widetilde { R } [ i ] \}$ while $F _ { \phi _ { 1 } }$ and $F _ { \phi _ { 2 } }$ receive the phase block $\{ \widetilde { \phi } [ i ] \}$ . Two outputs are allocated to phase because a substantial, often dominant, share of the discriminative fingerprint information is phaseencoded [2]. Polar CNN and Polar KAN use the same polar input, but without partitioned connectivity. The Polar MKAN architecture thus enforces channel-level separation without assuming that the two phase outputs isolate distinct physical factors. Formally,

$$
\begin{array} { r l r } & { } & { F _ { \phi _ { 1 } } = F _ { \phi _ { 1 } } ( \widetilde { \phi } _ { 1 } , \dots , \widetilde { \phi } _ { N } ) , } \\ & { } & { F _ { \phi _ { 2 } } = F _ { \phi _ { 2 } } ( \widetilde { \phi } _ { 1 } , \dots , \widetilde { \phi } _ { N } ) , } \\ & { } & { F _ { R } = F _ { R } ( \widetilde { R } _ { 1 } , \dots , \widetilde { R } _ { N } ) . } \end{array}\tag{4}
$$

Using $\succeq$ for componentwise ordering, monotonicity and the exclusive connections give the forward guarantees

$$
\begin{array} { r l } & { \widetilde { { \phi ^ { \prime } } } \succeq \widetilde { \phi } , \widetilde { { \bf R } } ^ { \prime } = \widetilde { { \bf R } } \Rightarrow F _ { \phi _ { k } } ^ { \prime } \ge F _ { \phi _ { k } } , k \in \{ 1 , 2 \} , \quad F _ { R } ^ { \prime } = F _ { R } , } \\ & { \widetilde { { \bf R } } ^ { \prime } \succeq \widetilde { { \bf R } } , \widetilde { \phi ^ { \prime } } = \widetilde { \phi } \Rightarrow F _ { R } ^ { \prime } \ge F _ { R } , \quad F _ { \phi _ { k } } ^ { \prime } = F _ { \phi _ { k } } . } \end{array}\tag{5}
$$

![](images/01343a674edf00613e88de23c6b7d760b4601445263ccfeb2f323219ddbc66d5.jpg)

![](images/a24751fe743cd402a8aa81451624300435bfc6359ab5a6f8054c1abb600f1567.jpg)  
Fig. 2: Polar MKAN. (a) Exclusive connections, where $F _ { \phi _ { 1 } } , F _ { \phi _ { 2 } }$ receive only phase inputs and $F _ { R }$ only magnitude inputs. (b) Idealised feature-space geometry away from the phase branch cut.

TABLE I: Composition of the synthetic dataset enabling a controlled disentanglement study.
<table><tr><td></td><td>Train</td><td>Evaluation</td></tr><tr><td>Devices</td><td>1000</td><td>100</td></tr><tr><td>Bursts/device</td><td>1</td><td>1</td></tr><tr><td>Burst length (N)</td><td>256</td><td>256</td></tr><tr><td>Samples/symbol (L)</td><td>4</td><td>4</td></tr><tr><td>SNR range (dB)</td><td>[10,30]</td><td>[10,30]</td></tr></table>

This means that raising every phase input while holding the magnitudes fixed can only raise the phase features, and leaves $F _ { R }$ untouched since it has no connection to the phase inputs. The second line mirrors this for magnitude. Monotonicity thus fixes the direction a feature may move, and the exclusive connections fix which features may move at all. Two conditions delimit these guarantees: (i) they run from an ordered input change to a feature change, so $\Delta F _ { \phi } > 0$ indicates a rise of the phase inputs, irrespective of its physical origin; (ii) they are stated in the principal-phase representation, so a phase shift is covered as long as every sample’s phase stays below $2 \pi ,$ since a sample that wraps back to 0 has decreased in this representation and the premise ${ \widetilde { \phi } } ^ { \prime } \succeq { \widetilde { \phi } }$ then fails.

## III. EVALUATION ON SIMULATED/SYNTHETIC DATASET

We construct a controlled RF-fingerprinting benchmark in which devices differ through fixed impairment parameters CFO, gain, I/Q imbalance, and AWGN model. No existing measured RF dataset has annotations that enables the study of disentanglement. We use the same impairment model for two complementary synthetic dataset evaluations: feature–factor analysis and UED evaluation. The feature–factor analysis uses 1,000 training and 100 held-out evaluation devices with one burst each (Table I), which gives broad coverage of the impairment space. The synthetic UED experiment uses 200 devices with 20 bursts each, since training, clustering, and open-set testing require repeated observations per device.

a) Signal and impairment model: Every burst uses the same fixed unit-power 16-QAM symbol sequence with levels $\{ - 3 , - 1 , 1 , 3 \}$ on both axes, upsampled by $L \ = \ 4$ and root-raised-cosine pulse-shaped (roll-off 0.35, span 8). The fixed sequence removes payload variation. Each device d has CFO $\nu _ { d } \sim \mathcal { N } ( 0 , ( 2 \times 1 0 ^ { - 3 } ) ^ { 2 } )$ and gain perturbation $g _ { d } \sim \mathcal { N } ( 0 , 0 . 0 7 ^ { 2 } )$ . With pulse-shaped $s [ n ]$

TABLE II: DCI Disentanglement (D) and Completeness (C) $( \% , \mathrm { m e a n } \pm 9 5 \%$ CI over 10 runs).
<table><tr><td>Architecture</td><td>D</td><td>C</td></tr><tr><td>CNN</td><td> $6 . 1 \pm 4 . 2$ </td><td> $6 . 7 \pm 3 . 4$ </td></tr><tr><td>KAN</td><td> $1 0 . 0 \pm 4 . 7$ </td><td> $9 . 8 \pm 4 . 0$ </td></tr><tr><td>MKAN</td><td> $7 . 4 \pm 3 . 1$ </td><td> $9 . 5 \pm 2 . 9$ </td></tr><tr><td>Polar CNN</td><td> $5 . 0 \pm 2 . 4$ </td><td> $5 . 3 \pm 1 . 8$ </td></tr><tr><td>Polar KAN</td><td> $1 2 . 9 \pm 4 . 8$ </td><td> $1 3 . 1 \pm 4 . 9$ </td></tr><tr><td>Polar MKAN</td><td> ${ \bf 5 7 . 2 \pm 6 . 0 }$ </td><td> ${ \bf 4 7 . 0 \pm 4 . 0 }$ </td></tr></table>

$$
u _ { d } [ n ] = s [ n ] e ^ { j 2 \pi \nu _ { d } n } .\tag{6}
$$

We additionally introduce complex I/Q imbalance as structured nuisance variation. Let $a _ { d } ~ \sim ~ \mathcal { N } ( 0 , 0 . 3 ^ { 2 } )$ dB, $\varphi _ { d } \sim$ $\mathcal { N } ( 0 , 0 . 0 3 ^ { 2 } )$ rad, and $\gamma _ { d } = 1 0 ^ { a _ { d } / 2 0 }$ . The equivalent-baseband coefficients are

$$
\begin{array} { r } { \mu _ { d } = \frac { 1 } { 2 } \big ( 1 + \gamma _ { d } e ^ { - j \varphi _ { d } } \big ) , \qquad \eta _ { d } = \frac { 1 } { 2 } \big ( 1 - \gamma _ { d } e ^ { j \varphi _ { d } } \big ) , } \end{array}\tag{7}
$$

giving

$$
x _ { d } [ n ] = ( 1 + g _ { d } ) \left[ \mu _ { d } u _ { d } [ n ] + \eta _ { d } u _ { d } ^ { \ast } [ n ] \right] .\tag{8}
$$

At zero imbalance $( \gamma _ { d } = 1 , \varphi _ { d } = 0 ) , \mu _ { d } = 1$ and $\eta _ { d } = 0$ . The imbalance is applied after the CFO/gain operation, matching the implementation. Because the image term contains $\boldsymbol { u } _ { d } ^ { * } [ { n } ]$ nonzero I/Q imbalance couples magnitude and phase, so the CFO effect is no longer purely phase-only.

Finally,

$$
y _ { d } [ n ] = x _ { d } [ n ] + w [ n ] , \qquad w [ n ] \sim { \mathcal { C N } } ( 0 , \sigma _ { w } ^ { 2 } ) ,\tag{9}
$$

where $\sigma _ { w } ^ { 2 } \ = \ P _ { x } / 1 0 ^ { \mathrm { S N R } / 1 0 }$ and $\mathbb { E } | w [ n ] | ^ { 2 } \ = \ \sigma _ { w } ^ { 2 }$ . Raw-I/Q models receive $( I , Q )$ directly, and polar models convert each burst to $( \widetilde { R } , \widetilde { \phi } )$ at loading time. The scored factor vector is $\pmb { \theta } _ { d } = ( g _ { d } , \nu _ { d } )$ , and the I/Q-imbalance parameters act as unscored structured nuisance variables.

b) Feature–factor analysis: All six autoencoders<sup>1</sup> are trained for 120 epochs with Adam (learning rate $1 0 ^ { - 2 } ) , N ^ { * } =$ 3, on the same synthetic training data. Polar CNN and Polar KAN serve as polar-input-only controls with unpartitioned encoders, while only the design of Polar MKAN enables the exclusive magnitude/phase output structure from Figure 2.

Following the DCI framework [21], [22], the evaluation estimates an $N ^ { * } \times K$ feature-importance matrix R by fitting one gradient-boosted regressor (100 estimators, maximum depth 3) from the learned coordinates to each of the $K = 2$ scored factors. For latent coordinate i,

$$
D _ { i } = 1 - { \frac { H ( P _ { i \cdot } ) } { \log K } } , \qquad P _ { i j } = { \frac { R _ { i j } } { \sum _ { k } R _ { i k } } } ,\tag{10}
$$

and overall disentanglement uses the canonical importance weighting

$$
D = \sum _ { i } \rho _ { i } D _ { i } , \qquad \rho _ { i } = \frac { \sum _ { j } R _ { i j } } { \sum _ { k , j } R _ { k j } } .\tag{11}
$$

![](images/a96ffe03512cb74453d94893ec8714d92495daa380694af3feeb5977ca87e310.jpg)  
Fig. 3: Sign consistency of the phase features vs. fraction of samples whose phase wraps (95% CI, 10 runs).

Completeness (C) is computed analogously across latent coordinates for each factor. Without I/Q imbalance, gain and CFO would align exactly with magnitude and phase, respectively. Under the full model used here, the image term introduces cross-channel coupling, so the correspondence is no longer exact. The experiment therefore evaluates whether the imposed magnitude/phase structure remains informative about the two device factors in the presence of structured nuisance variation. Table II shows that only Polar MKAN achieves substantial 57.2% D and 47.0% C values, while Polar KAN and Polar CNN show that polar coordinates alone do not recover comparable separation, achieving 12.9%, 5% D and 13.1%, 5.3%, respectively. The raw-I/Q MKAN shows that monotonicity in Cartesian coordinates is insufficient. The lower completeness, as shown in the last column of II is consistent with the two phase coordinates that share information about the single scored CFO factor.

c) Phase response validation: Eq. (5) guarantees the response sign only for wrap-free inputs, so the practically relevant property is how quickly sign consistency is lost once samples start crossing the 2π branch cut. To characterise this, we apply rotations in [−0.6, 0.6] rad to the synthetic evaluation set over 10 independently trained Polar MKANs and group the responses by the fraction of samples whose phase wraps, with the results shown in Fig. 3. Sign consistency decreases gradually with the wrapped fraction, from complete consistency when no sample wraps to roughly 70% below 2% wrapped samples, and to 20–30% once more than 10% of the samples cross the 2π. The sign reading of the phase features therefore remains reliable for small rotations, and degrades gradually as wrapping becomes widespread.

d) Synthetic UED: Factor alignment alone does not establish that the representation remains useful for detection, so we also run the clustering-based UED task of Fig. 1 on the synthetic data, mirroring the WiSig protocol of Section IV. A fixed 25% of the 200 devices are unknown and appear only in the test set, and for the rest 80% of bursts train and 20% test. Models train for 120 epochs with Adam (learning rate $1 0 ^ { - 2 } )$ , batch size 50 and input-noise standard deviation 0.01, using 80 clusters. Results are averaged over 10 training seeds.

e) Performance and interpretability: Fig. 4 compares synthetic UED performance with held-out DCI Disentanglement. Polar MKAN occupies a distinct high-disentanglement region while retaining 73.1% synthetic UED ROC AUC, within one standard deviation of raw-I/Q MKAN at 74.4% (Table III). Polar KAN, with the same input but neither monotonicity nor exclusive connections, reaches only 12.9% D, and raw-I/Q MKAN 7.4%, so the separation is associated with the combined block-partitioned monotonic construction rather than the polar transform alone. Taken together, the synthetic experiments evaluate two complementary properties. DCI measures alignment with known device factors, while UED measures whether the representation preserves device separability. Polar MKAN substantially improves the former while retaining useful detection performance.

![](images/92a88110f8b48d1408e19d97d2fe15366dd518d16dc51dfa937a0deba17dca49.jpg)  
Fig. 4: Synthetic UED ROC AUC vs. held-out DCI Disentanglement (95% CI, 10 runs).

## IV. VALIDATION AND EVALUATION ON WISIG

We validate and evaluate the proposed method versus selected alternatives on WiSig [23] SingleDay and ManyTx subsets, using length-256 I/Q and a single receiver (index 0) in both. SingleDay contains 28 emitters from one capture day, and ManyTx contains 150 emitters and samples from four days. Device-level open-set folds are generated by the same repository routine used in [8] (fold-ratio settings 7 and 5, respectively), and devices returned in each fold’s unknown list are excluded from that fold’s training data and labelled unknown at evaluation. Models are trained for 120 epochs with Adam (learning rate $1 0 ^ { - 3 } )$ , batch size 50, using 280 clusters for SingleDay and 1,000 for ManyTx. The number of clusters is set to 5–10× the number of classes, following the findings in [24]. Each fold is trained once, so the reported WiSig std aggregates device-split and training variability.

a) Amplitude/phase response: Fig. 5 illustrates the response on real signals. A common positive rotation leaves magnitudes unchanged. In raw I/Q it does not produce componentwise ordered inputs, so an MKAN can move latent coordinates in different directions, whereas in Polar MKAN a wrapfree rotation obeys (5), so $F _ { R }$ is unchanged and both phase outputs are nondecreasing. For a flagged sample, its deviation from the nearest known cluster centre, $( \Delta F _ { R } , \Delta F _ { \phi _ { 1 } } , \Delta F _ { \phi _ { 2 } } )$ therefore decomposes the anomaly into a magnitude-channel and a phase-channel component, measured in units of withincluster spread, and a positive $\Delta F _ { \phi }$ indicates an aggregate positive phase rotation relative to the cluster, with the sign interpretation inherited from (5) under its wrap-free condition. In an unconstrained feature space, the same deviation vector points in an arbitrary learned direction and supports no such reading.

![](images/f9b18503573d55cf01f87b9011171ad35b399d45a103e24644b597a431e59c99.jpg)

(a) Raw I/Q MKAN  
![](images/bf9451a2df428fbd60b977bb535dee106c1fc92dd4881eca6169a52f51f3de67.jpg)  
(b) Polar MKAN  
Fig. 5: Response to a positive carrier-phase rotation on WiSig ManyTx.

b) Performance–interpretability trade-off: Polar MKAN reaches 78.8% ROC AUC on SingleDay (KAN 89.8%) and 67.2% on ManyTx (KAN 73.2%) as summarized in Table III, quantifying the real-data cost of the stronger structural bias. The constrained model therefore trades approximately 11 AUC points on SingleDay and 6 points on ManyTx for the explicit magnitude/phase feature structure. Part of this gap is a dimensionality effect, since by Theorem 1 of [19], a monotone FE may need up to 2× the feature dimensionality of an unconstrained one to reproduce the same neighbourhood partition, so the fixed $N ^ { * } = 3$ disadvantages the monotone models.

c) Sensitivity to CFO suppression: Because CFO is a strong but temperature- and time-varying fingerprint [25], we evaluate sensitivity to its blind suppression on the synthetic UED benchmark and both WiSig subsets. For each burst, we estimate

$$
\hat { \nu } = \frac { 1 } { 2 \pi } \overline { { \mathrm { a r g } ( x [ n ] x ^ { * } [ n - 1 ] ) } }
$$

TABLE III: UED results (%). AUC and F1 uncompensated, $\mathrm { \ A U C _ { c o m p } }$ with blind CFO compensation before training and evaluation. Std over 10 seeds (synthetic) or device folds (WiSig). Best uncompensated in bold.
<table><tr><td></td><td colspan="3">Synthetic</td><td colspan="3">SingleDay</td><td colspan="3">ManyTx</td></tr><tr><td>FE</td><td>AUC</td><td>F1</td><td> $\mathbf { A U C } _ { \mathrm { c o m p } }$ </td><td>AUC</td><td>F1</td><td> $\mathbf { A U C } _ { \mathrm { c o m p } }$ </td><td>AUC</td><td>F1</td><td> $\mathbf { A U C } _ { \mathrm { c o m p } }$ </td></tr><tr><td>CNN</td><td> $6 2 . 1 { \pm } 2 . 2 $ </td><td> $2 4 . 8 { \pm } 3 . 6 $ </td><td> $5 1 . 1 { \pm } 1 . 7 $ </td><td> $7 1 . 0 { \pm } 2 . 5 $ </td><td> $3 6 . 8 { \pm } 5 . 4 $ </td><td> $6 6 . 8 { \pm } 2 . 4 $ </td><td> $6 4 . 7 { \pm } 1 . 8 $ </td><td> $2 8 . 8 { \pm } 3 . 9 $ </td><td> $6 2 . 3 { \pm } 1 . 3 $ </td></tr><tr><td>KAN</td><td> ${ \bf 7 8 . 4 \pm 4 . 2 }$ </td><td> $\mathbf { 6 2 . 9 2 } \mathrm { \pm 8 . 7 }$ </td><td> $5 1 . 1 { \pm } 1 . 1$ </td><td> ${ \bf 8 9 . 8 \pm } 3 . 6 $ </td><td>74.0±7.0</td><td> $7 6 . 2 { \pm } 4 . 6 $ </td><td> $7 3 . 2 { \pm } 2 . 6 $ </td><td> $\mathbf { 4 4 . 0 \pm 5 . 5 }$ </td><td> $6 7 . 1 \pm 3 . 1 $ </td></tr><tr><td>MKAN</td><td> $7 4 . 4 \pm 4 . 6$ </td><td> $5 3 . 3 { \pm } 8 . 2 $ </td><td> $4 9 . 8 { \pm } 1 . 4 $ </td><td> $7 9 . 7 { \pm } 5 . 8 $ </td><td> $6 1 . 3 { \pm } 7 . 9$ </td><td> $6 6 . 5 { \pm } 5 . 1 $ </td><td> $6 9 . 9 { \pm } 1 . 2 $ </td><td> $3 8 . 2 { \pm } 2 . 4 $ </td><td> $6 0 . 8 { \pm } 2 . 3 $ </td></tr><tr><td>Polar CNN</td><td> $5 3 . 6 { \pm } 1 . 6 $ </td><td> $2 0 . 6 { \pm } 2 . 2 $ </td><td> $5 0 . 4 \pm 0 . 9$ </td><td> $6 4 . 8 { \pm } 2 . 9$ </td><td> $2 7 . 4 \pm 5 . 9$ </td><td> $6 2 . 6 { \pm } 2 . 9$ </td><td> $6 1 . 7 { \pm } 1 . 1 $ </td><td> $2 4 . 1 { \pm } 2 . 0 $ </td><td> $6 0 . 9 { \pm } 1 . 0 $ </td></tr><tr><td>Polar KAN</td><td> $5 6 . 3 { \pm } 5 . 3 $ </td><td> $2 3 . 2 { \pm } 7 . 5 $ </td><td> $5 0 . 5 { \pm } 1 . 7 $ </td><td> $8 2 . 8 { \pm } 4 . 6 $ </td><td> $5 9 . 8 { \pm } 7 . 1 $ </td><td> $7 1 . 0 { \pm } 5 . 8 $ </td><td> $6 6 . 3 { \pm } 1 . 8 $ </td><td> $3 0 . 9 { \pm } 4 . 0 $ </td><td> $6 2 . 2 { \pm } 2 . 4 $ </td></tr><tr><td>Polar MKAN</td><td> $7 3 . 1 \pm 3 . 6 $ </td><td> $4 5 . 9 2 4 . 6 $ </td><td> $4 9 . 8 { \pm } 1 . 1 $ </td><td> $7 8 . 8 { \pm } 4 . 8 $ </td><td> $5 2 . 6 { \pm } 9 . 1 $ </td><td> $6 6 . 5 { \pm } 4 . 6 $ </td><td> $6 7 . 2 { \pm } 1 . 0 $ </td><td> $3 1 . 5 { \pm 2 . 1 }$ </td><td> $6 1 . 3 { \pm } 1 . 9$ </td></tr></table>

and apply $x [ n ]  x [ n ] e ^ { - j 2 \pi \hat { \nu } n }$ before training and evaluation, leaving all other UED settings unchanged. The $\mathrm { \ A U C _ { c o m p } }$ columns of Table III report the result. Compensation reduces AUC for every architecture.

On the synthetic task all fall to 50–53%, i.e., near chance, since CFO is essentially the only strong device-specific factor there. On WiSig the reductions are far smaller (about 1–14 points) and all models remain well above chance, indicating that real emitters carry substantial fingerprint information beyond CFO.

## V. CONCLUSION

Polar MKAN imposes a simple structural constraint on RF-fingerprinting features, in which one latent coordinate depends only on magnitudes and two depend only on principal phases, with componentwise nondecreasing mappings in each connected block. The synthetic gain/CFO benchmark shows substantially stronger feature–factor alignment than the evaluated unpartitioned baselines, while WiSig shows a measurable UED-accuracy cost. Future work should separate partitioning from monotonicity experimentally, use circular or referenceinvariant phase representations to remove the branch-cut limitation, and assess end-to-end attribution on flagged emitters.

## REFERENCES

[1] K. Sankhe, M. Belgiovine, F. Zhou, L. Angioloni, F. Restuccia, S. D’Oro, T. Melodia, S. Ioannidis, and K. Chowdhury, “No radio left behind: Radio fingerprinting through deep learning of physical-layer hardware impairments,” IEEE Transactions on Cognitive Communications and Networking, vol. 6, no. 1, pp. 165–178, 2019.

[2] N. Soltanieh, Y. Norouzi, Y. Yang, and N. C. Karmakar, “A review of radio frequency fingerprinting techniques,” IEEE Journal of Radio Frequency Identification, vol. 4, no. 3, pp. 222–233, 2020.

[3] L. Xie, L. Peng, J. Zhang, and A. Hu, “Radio frequency fingerprint identification for internet of things: A survey,” Security and Safety, vol. 3, p. 2023022, 2024.

[4] J. Robinson and S. Kuzdeba, “Novel device detection using rf fingerprints,” in 2021 IEEE 11th Annual Computing and Communication Workshop and Conference (CCWC). IEEE, 2021, pp. 0648–0654.

[5] S. Apfeld and A. Charlish, “Recognition of unknown radar emitters with machine learning,” IEEE Transactions on Aerospace and Electronic Systems, vol. 57, no. 6, pp. 4433–4447, 2021.

[6] M. Zhan, Y. Li, H. Cui, B. Li, J. Zhang, C. Li, and W. Wang, “Mcrff: A meta-contrastive learning-based rf fingerprinting method,” in MILCOM 2023-2023 IEEE Military Communications Conference (MILCOM). IEEE, 2023, pp. 391–396.

[7] X. Hao, Z. Feng, R. Liu, S. Yang, L. Jiao, and R. Luo, “Contrastive selfsupervised clustering for specific emitter identification,” IEEE Internet of Things Journal, vol. 10, no. 23, pp. 20 803–20 818, 2023.

[8] M. Krasnov, L. Milosheski, M. Mohorciˇ c, and C. Fortuna, “Designˇ principles of zero-shot self-supervised unknown emitter detectors,” arXiv preprint arXiv:2511.07026, 2025.

[9] C. Rudin, C. Chen, Z. Chen, H. Huang, L. Semenova, and C. Zhong, “Interpretable machine learning: Fundamental principles and 10 grand challenges,” Statistic Surveys, vol. 16, pp. 1–85, 2022.

[10] K. Simonyan, A. Vedaldi, and A. Zisserman, “Deep inside convolutional networks: Visualising image classification models and saliency maps,” arXiv preprint arXiv:1312.6034, 2013.

[11] M. T. Ribeiro, S. Singh, and C. Guestrin, ““Why should I trust you?”: Explaining the predictions of any classifier,” in Proc. ACM SIGKDD Int. Conf. Knowledge Discovery and Data Mining, 2016, pp. 1135–1144.

[12] S. M. Lundberg and S.-I. Lee, “A unified approach to interpreting model predictions,” in Advances in Neural Information Processing Systems (NeurIPS), 2017, pp. 4765–4774.

[13] R. Xie, W. Xu, J. Yu, A. Hu, D. W. K. Ng, and A. L. Swindlehurst, “Disentangled representation learning for RF fingerprint extraction under unknown channel statistics,” IEEE Transactions on Communications, vol. 71, no. 7, pp. 3946–3962, 2023.

[14] Y. Zhang et al., “Factorized disentangled representation learning for interpretable radio frequency fingerprint,” arXiv preprint arXiv:2508.12660, 2025.

[15] Y. Pan et al., “Cross-receiver generalization for RF fingerprint identification via feature disentanglement and adversarial training,” arXiv preprint arXiv:2510.09405, 2025.

[16] F. Locatello, S. Bauer, M. Lucic, G. Raetsch, S. Gelly, B. Schölkopf, and O. Bachem, “Challenging common assumptions in the unsupervised learning of disentangled representations,” in international conference on machine learning. PMLR, 2019, pp. 4114–4124.

[17] Z. Liu, Y. Wang, S. Vaidya, F. Ruehle, J. Halverson, M. Soljaciˇ c, T. Y.´ Hou, and M. Tegmark, “KAN: Kolmogorov–Arnold networks,” arXiv preprint arXiv:2404.19756, 2024.

[18] S. Somvanshi, S. A. Javed, M. M. Islam, D. Pandit, and S. Das, “A survey on kolmogorov-arnold network,” ACM Computing Surveys, 2024.

[19] M. Krasnov, B. Bertalanic, and C. Fortuna, “Monotonic kolmogorov-ˇ arnold networks: A theoretical and empirical study of monotonicity as an inductive bias,” arXiv preprint arXiv:2606.17886, 2026.

[20] A. Polo-Molina, D. Alfaya, and J. Portela, “Monokan: Certified monotonic kolmogorov-arnold network,” Neural Networks, p. 108278, 2025.

[21] C. Eastwood and C. K. I. Williams, “A framework for the quantitative evaluation of disentangled representations,” in International Conference on Learning Representations (ICLR), 2018.

[22] M.-A. Carbonneau, J. Zaidi, J. Boilard, and G. Gagnon, “Measuring disentanglement: A review of metrics,” IEEE transactions on neural networks and learning systems, vol. 35, no. 7, pp. 8747–8761, 2022.

[23] S. Hanna, S. Karunaratne, and D. Cabric, “Wisig: A large-scale wifi signal dataset for receiver and channel agnostic rf fingerprinting,” IEEE Access, vol. 10, p. 22808–22818, 2022.

[24] M. Krasnov, L. Milosheski, M. Mohorciˇ c, and C. Fortuna, “Novelˇ devices identification with deep clustering,” in 2025 IEEE International Conference on Machine Learning for Communication and Networking (ICMLCN). IEEE, 2025, pp. 1–6.

[25] L. Xie, L. Peng, and J. Zhang, “Towards robust rf fingerprint identification using spectral regrowth and carrier frequency offset,” in IEEE INFOCOM 2025 - IEEE Conference on Computer Communications, 2025, pp. 1–10.