# Atrial Fibrillation Detection with Arbitrary Leads via a Codebook-Based Reconstruction-Classification Framework

Hongtao Li<sup>a</sup>, Jia Wei<sup>b</sup>, Guoyao Li<sup>c</sup>, Yuchen Lei<sup>d</sup>, Guangnian Ma<sup>a</sup>, Jia Xiao<sup>a</sup>, Yuanjun Lai<sup>a</sup>, Shuzhen Lv<sup>a</sup> and Xueqiang Ouyang<sup>a,∗</sup>

<sup>a</sup>Information Center, The People’s Hospital of Baoan Shenzhen, Shenzhen, 518101, Guangdong, China

<sup>b</sup>School of Computer Science and Engineering, South China University of Technology, Guangzhou, 510006, Guangdong, China

<sup>c</sup>Department of Cardiology, The People’s Hospital of Baoan Shenzhen, Shenzhen, 518101, Guangdong, China

<sup>d</sup>Institutefor Advanced Study, Shenzhen University, Shenzhen, 518060, Guangdong, China

## A R T I C L E I N F O

Keywords:   
Atrial Fibrillation Detection   
Arbitrary Leads   
Codebook-based Learning   
Joint Reconstruction-Classification   
Contrastive Learning   
Graph Attention Network

## A BS T RA C T

Background and Objective: Reliable atrial fibrillation (AF) detection from electrocardiogram (ECG) signals remains challenging in real-world clinical settings due to variable lead configurations, crossdataset domain shifts, and pervasive physiological and technical artifacts. So we develop a robust and generalizable deep learning model for accurate AF detection.

Methods: We propose the Dual-Codebook Graph Collaborative Network (DCGCNet), a novel endto-end vector-quantized variational autoencoder that jointly performs AF classification and ECG reconstruction. DCGCNet introduces two key components: (1) a Local-Global Contrastive Module for learning noise-invariant representations, and (2) an Adaptive Codebook Vector Quantizer that dynamically refines codebook prototypes to better align with input data distributions, thereby preventing codebook collapse and enhancing generalization.

Results: DCGCNet achieves state-of-the-art performance in standard intra-dataset 12-lead evaluation and demonstrates exceptional cross-dataset generalization across seven diverse settings, consistently attaining AUC > 0.98 in all cases. Furthermore, it maintains high diagnostic accuracy under realistic noisy conditions, including baseline wander, powerline interference, and EMG artifacts.

Conclusions: DCGCNet establishes a new benchmark for robust, generalizable, and noise-resilient AF detection, showing strong potential for deployment in real-world clinical environments.

## 1. Introduction

Atrial Fibrillation (AF), the most prevalent cardiac arrhythmia in clinical practice, has increasingly become a critical global public health burden [9]. Recent epidemiological investigations have demonstrated that the number of individuals afected by AF worldwide has exceeded sixty million, and its prevalence continues to rise steadily, primarily driven by the global aging population and the changing lifestyles of modern societies [10]. AF is closely associated with a significantly elevated risk of severe complications, including ischemic stroke and heart failure, which not only compromise patients’ quality of life but also impose an enormous economic burden on healthcare systems across the globe [16]. Therefore, the development of eficient, accurate, and reliable automated AF detection technologies is of paramount importance for facilitating early clinical intervention, optimizing disease management, and ultimately improving long-term patient outcomes [49].

In clinical workflows, the electrocardiogram (ECG) remains the gold standard for AF diagnosis. However, conventional AF detection relies heavily on manual visual interpretation by specialized cardiologists, which is timeconsuming, labor-intensive, and prone to subjective bias, limiting its utility for large-scale screening and real-time monitoring. In recent years, Artificial Intelligence (AI),

![](images/8d1a90737159796535ea65d751e3d9c54773adc99b88637efc415922d5702216.jpg)  
Figure 1: Evolution of AF detection frameworks and our motivation. (a) Fixed-Lead models are inflexible to varying inputs. (b) Arbitrary-Lead models sufer from information scarcity. (c) Sequential frameworks decouple tasks, causing error propagation. (d) Our parallel framework jointly optimizes both tasks for superior performance.

particularly deep learning algorithms, has achieved transformative breakthroughs in ECG analysis, ofering a promising approach for automated AF detection [14].

Existing AF detection methodologies of deep learning exhibit a clear evolution in addressing the challenge of variable ECG lead configurations, as illustrated in Figure 1. The initial approach, Fixed-Lead Classification (Figure 1(a)), operates under the assumption of a standardized input, typically a fixed set of high-quality leads. While this paradigm can yield excellent performance in controlled clinical settings, its fundamental limitation is a complete lack of flexibility; it fails catastrophically when deployed in real-world scenarios involving diverse or non-standard lead setups [6][46][23].

To overcome this rigidity, research has shifted toward Arbitrary-Lead Classification (Figure 1(b)), which aims to develop models agnostic to the specific input leads. However, this direct approach is fundamentally limited by the incomplete diagnostic information inherent in non-standard or reduced-lead ECGs, as it lacks the comprehensive spatial perspective provided by a full 12-lead set, often resulting in suboptimal and unstable diagnostic performance [55].

A further approach, Sequential Reconstruction Classification (Figure 1(c)), bridges this gap by first reconstructing full standard 12-lead signals from arbitrary inputs via a dedicated network, before feeding the reconstruction into a separate classifier. Despite enhanced adaptability, this twostage pipeline has a critical flaw: decoupled reconstruction and classification objectives. The reconstructor optimizes for signal fidelity, which may misalign with the classifier’s need for discriminative, diagnosis-relevant features, thereby risking the amplification of noise or the discarding of crucial diagnostic information [40][44][54].

To address these limitations, we propose a Codebookbased Parallel Reconstruction-Classification Framework (Figure 1(d)). The framework employs a shared codebook that jointly governs both objectives. This discrete representation is naturally suited to signal reconstruction, learning a compact set of prototypical ECG patterns for high-fidelity recovery from incomplete inputs [51][29]. Simultaneously, by projecting latent features onto this finite set of learned prototypes, the codebook implicitly regularizes the classifier, filtering out instance-specific noise and enhancing generalization [32][37].The two tasks are synergistically coupled through the shared codebook: reconstruction populates it with physiologically plausible structures, while classification shapes it to prioritize diagnostically discriminative features. This unified formulation ensures the reconstructed signal is both anatomically faithful and optimally informative for robust classification. The main contributions of this work are summarized as follows:

1. We propose the first codebook-based joint learning framework for AF detection, DCGCNet—an end-to-end VQ-VAE that unifies classification and reconstruction via two specialized codebooks: a rhythm codebook for local discriminative patterns and a morphology codebook for global signal fidelity, enabling synergistic representation learning.

2. We introduce two tailored modules to boost robustness and generalization: a Local-Global Contrastive Module (LGCM) that enhances noise resilience through hierarchical contrastive learning on rhythm features, and an Adaptive Codebook Vector Quantizer (ACVQ) that dynamically aligns code vectors with the data manifold to prevent collapse and improve quantization stability.

3. We achieve state-of-the-art AF detection performance across diverse settings—demonstrating strong results in standard intra-dataset 12-lead evaluation, challenging cross-dataset arbitrary-lead generalization, and realistic noise conditions, highlighting clinical applicability.

## 2. Related Work

Automatic detection of AF increasingly leveraged deep learning models in recent years. Early approaches employed CNNs [26][46] and RNNs [24] to capture local patterns and long-range dependencies, respectively. Subsequent work introduced GNNs [42] to model spatial correlations among multi-lead ECG signals, while more recent architectures—such as Transformers [19] and Mamba [43]—enabled eficient modeling of global context, steadily improving detection performance. Existing methods largely fell into two paradigms: (1) fixed-lead training, which achieved strong performance on standard benchmarks but relied heavily on predefined lead configurations; and (2) random lead masking during training, which enhanced deployment flexibility but often compromised discriminative capability when critical leads were missing.

To mitigate information loss from limited leads, ECG lead reconstruction was explored as a preprocessing step, recovering a full 12-lead signal from a subset to provide richer input for downstream AF detection. Representative approaches included encoder-decoder CNNs/RNNs [34][22][11][25], GANs [56][20] that improved waveform realism through adversarial training, VAEs [12] that modeled probabilistic latent representations, DDPMs [30] that generated high-fidelity signals via iterative refinement, and echo state networks (ESNs) [18] that captured complex temporal dynamics with recurrent reservoir structures. However, these methods primarily prioritized signal fidelity and remained largely decoupled from the AF detection task, lacking explicit optimization for diagnostically relevant features.

Even under full-lead conditions, AF detectors often suffered from severe performance degradation. This degradation stemmed from two coexisting challenges: (1) domain shift caused by variations in acquisition hardware and patient demographics; and (2) signal corruption due to environmental and physiological noise. To address these issues, prior work explored test-time adaptation [47] and domain generalization [4] to improve cross-domain generalization, while data augmentation [13] was employed to enhance robustness against noisy perturbations. However, existing approaches typically treated signal reconstruction, AF detection, domain generalization, and noise robustness as largely decoupled objectives. As a result, they failed to leverage the synergies among these tasks within a coherent, unified architecture.

![](images/c3ff898bd105b081e8ac9f26d0d1021122ce52ef2991c013ad00aba06b0e9686.jpg)  
Figure 2: An end-to-end unified model integrating morphology-rhythm dual encoding, adaptive codebook quantization, local-global contrastive learning and heterogeneous graph interaction for multi-objective ECG analysis.

## 3. Method

To simultaneously address AF detection and multi-lead ECG reconstruction, we propose the Dual-Codebook Graph Collaborative Network (DCGCNet)—a unified framework that integrates four key components: morphology-rhythm dual encoding, adaptive codebook quantization, heterogeneous graph interaction, and local-global contrastive learning. As illustrated in Figure 2(a), these components operate within a multi-task architecture where the two objectives mutually reinforce each other, enabling DCGCNet to learn structured and multi-scale representations that jointly enhance diagnostic accuracy and reconstruction fidelity.

## 3.1. Rhythm-Morphology Dual Encoding

AF is defined by disorganized atrial activation and an irregular ventricular response [27]. AF detection algorithms primarily rely on irregular RR intervals as a surrogate marker. Although the absence of organized P waves supports the diagnosis, it is often obscured by noise or fibrillatory activity. Consequently, AF detection depends primarily on local rhythm dynamics, especially beat-to-beat RR variability. In contrast, multi-lead ECG reconstruction requires faithful preservation of global morphological integrity, including the amplitude, duration, and spatiotemporal alignment of P-QRS-T complexes across leads—features critical for accurate electrophysiological interpretation and clinical decision-making [41].

To explicitly account for the divergent signal characteristics underlying AF detection and ECG reconstruction, we propose two dedicated encoders with tailored receptive fields that extract rhythm- and morphology-specific representations from the input ECG $\mathbf { X } \in \mathbf { \mathbb { R } } ^ { B \times \bar { C } \times T }$ (�=12 leads, �=2500 sampling points, � is batchsize):

$$
\mathbf { H } _ { r } = \mathscr { E } _ { \theta } ^ { r h y t h m } ( \mathbf { X } ) \in \mathbb { R } ^ { B \times T ^ { \prime } \times D } ,
$$

$$
\mathbf { H } _ { m } = \mathcal { E } _ { \delta } ^ { m o r p h } ( \mathbf { X } ) \in \mathbb { R } ^ { B \times T ^ { \prime } \times D } ,\tag{1}
$$

(2)

where �<sup>′</sup>=625 and �=256. The rhythm encoder $\mathcal { E } _ { \theta } ^ { r h y t h m }$ adopts convolutional small kernels (�=5) to model finegrained irregularities characteristic of atrial fibrillation, whereas the morphology encoder $\mathcal { E } _ { \delta } ^ { m o r p h }$ utilizes larger kernels (�=15) to capture long-range structural dependencies essential for reconstruction.

## 3.2. Adaptive Codebook Vector Quantizer

The codebook in VQ-VAE provides a discrete latent representation that is inherently well-suited for generative modeling, making it highly efective for ECG reconstruction [51][29]. Moreover, by clustering semantically similar ECG segments into shared discrete codes, the codebook facilitates prototype-like learning that enhances feature consistency, improves generalization, and thereby enables robust AF detection even under distribution shifts [32][37].

To address the well-known issue of codebook collapse in vector quantization—where only a few codewords are actively used while others remain dormant [45]—we propose the ACVQ as shown in Figure 2(b). Unlike conventional static codebooks, ACVQ generates context-aware codewords by modeling interactions among prototype vectors via a lightweight Transformer-based module, termed VQBridge $B _ { \mathrm { v q } } \ [ 7 ]$ . Specifically, given a static codebook $\mathbf { Z } _ { \ast } \in \mathbb { R } ^ { K \times D }$ (where $\ast \in \quad \{ m , r \}$ denotes morphology or rhythm), the adaptive codebook $\overline { { \mathbf { Z } } } _ { \ast }$ is computed as:

$$
\mathbf { \overline { { Z } } _ { * } } = \mathbf { W } _ { \mathrm { o u t } } \cdot \mathrm { T r a n s f o r m e r } \big ( \mathbf { Z } _ { * } \cdot \mathbf { W } _ { \mathrm { i n } } \big ) , \quad * \in \{ m , r \} ,\tag{3}
$$

where $\mathbf { W _ { \mathrm { i n } } } ~ \in ~ \mathbb { R } ^ { D \times d ^ { \prime } }$ and $\mathbf { W _ { \mathrm { o u t } } } \ \in \ \mathbb { R } ^ { d ^ { \prime } \times D }$ are learnable projection matrices $( d ^ { \prime } = 2 5 6 )$ . This dynamic refinement encourages diverse codeword usage and mitigates dominance by a small subset of prototypes.

Then, we design a dual-codebook VQ-VAE that disentangles cardiac morphology and rhythm into two dedicated latent streams, leveraging the multi-scale features extracted by the hierarchical convolutional modules described in Section 3.1. The model employs two independent adaptive codebooks: a morphology codebook $\overline { { \mathbf { Z } } } _ { m }$ and a rhythm codebook $\overline { { \mathbf { Z } } } _ { r } ,$ each derived from its static counterpart via ACVQ. Given input features $\mathbf { H } _ { m }$ and $\mathbf { H } _ { r } .$ , quantization is performed separately through nearest-neighbor lookup:

$$
\mathbf { Z } _ { * , i } ^ { q } = \arg \operatorname* { m i n } _ { k \in \{ 1 , \ldots , K \} } \left\| \mathbf { H } _ { * , i } - \overline { { \mathbf { Z } } } _ { * , k } \right\| _ { 2 } ^ { 2 } , \quad * \in \{ m , r \} .\tag{4}
$$

The standard vector quantization loss is applied independently to each stream:

$$
\mathcal { L } _ { \mathrm { v q } _ { - } * } = \beta \mathopen { } \mathclose \bgroup \left\| \mathbf { Z } _ { * } ^ { q \mathrm { d e t a c h } } - \mathbf { H } _ { * } \aftergroup \egroup \right\| _ { 2 } ^ { 2 } + \mathopen { } \mathclose \bgroup \left\| \mathbf { Z } _ { * } ^ { q } - \mathbf { H } _ { * } ^ { \mathrm { d e t a c h } } \aftergroup \egroup \right\| _ { 2 } ^ { 2 } , \quad * \in \{ m , r \} ,\tag{5}
$$

with gradients estimated via the straight-through estimator.

## 3.3. Heterogeneous Graph Interaction

Although rhythm and morphology serve diferent diagnostic purposes, they are interrelated: temporal changes in waveform shape often reflect rhythm disturbances, and rhythm irregularities can also appear as subtle morphological changes across beats.

To better integrate complementary rhythm and morphology cues in a temporally coherent manner, we construct a heterogeneous graph $\mathcal { G } _ { \phi } = ( \mathcal { V } , \mathcal { E } )$ over the quantized token sequence of a single recording. The node set $\mathcal { V } = \{ r _ { t } , m _ { t } \} _ { t = 1 } ^ { T ^ { \prime } }$ contains rhythm and morphology tokens at each time step. Edges follow two principles as shown in Figure2 (a) $\mathcal { G } _ { \phi } \mathrm { : }$ (i) intra-modality continuity: each token connects to its immediate temporal neighbors within the same modality $( m _ { t }  m _ { t \pm 1 } , r _ { t }  r _ { t \pm 1 } ) ;$ (ii) inter-modality alignment: each morphology token $m _ { t }$ links to rhythm tokens at the same and adjacent positions $( r _ { t - 1 } , r _ { t } , r _ { t + 1 } )$ , and symmetrically for $r _ { t }$

This design explicitly captures cross-modal interactions—both synchronous and temporally ofset—within a three-step neighborhood, facilitating context-aware fusion of multi-modal temporal features for downstream tasks.

These structured relations are processed by a multihead GAT [52] that computes contextualized node representations through adaptive neighbor aggregation, $\mathbf { x } _ { i \_ n e w } =$ $\begin{array} { r } { \sum _ { j \in \mathcal { N } ( i ) } \alpha _ { i j } \mathbf { W } \mathbf { x } _ { j } } \end{array}$ , and:

$$
\alpha _ { i j } = \frac { \exp \left( \mathrm { L e a k y R e L U ( \mathbf { a } ^ { \top } [ \mathbf { W } \mathbf { x } _ { i } \mid \lvert \mathbf { W } \mathbf { x } _ { j } ] ) } \right) } { \sum _ { l \in \mathcal { N } ( i ) } \exp \left( \mathrm { L e a k y R e L U ( \mathbf { a } ^ { \top } [ \mathbf { W } \mathbf { x } _ { i } \mid \lvert \mathbf { W } \mathbf { x } _ { l } ] ) } \right) } ,\tag{6}
$$

where $\mathbf { x } _ { i / j }$ denotes the input feature of node $i / j$ , � is a learnable projection matrix, and $\alpha _ { i j }$ represents the attention weight assigned to neighbor � when updating node �.

## 3.4. Local-Global Contrastive Module

To improve robustness against common ECG artifacts, we adopt contrastive learning on rhythm features extracted from both the original view (�) and an augmented view $( \tilde { \mathbf { X } } )$ as described in Section 4.3. By enforcing consistency between (�) and (�<sup>̃</sup> ) derived from diferently perturbed versions of the same input, the model learns noise-invariant rhythm embeddings.

Specifically, we construct two complementary views of each sequence as shown in Figure 2(c): a global representation capturing overall temporal semantics, and a local representation preserving fine-grained temporal structure:

$$
\tilde { \mathbf { H } } _ { r } ^ { g } / \mathbf { H } _ { r } ^ { g } = \frac { 1 } { T ^ { \prime } } \sum _ { t = 1 } ^ { T ^ { \prime } } \tilde { \mathbf { H } } _ { r } / \mathbf { H } _ { r } \in \mathbb { R } ^ { D } ,\tag{7}
$$

$$
\tilde { \mathbf { H } } _ { r } ^ { l } / \mathbf { H } _ { r } ^ { l } = \mathrm { f l a t t e n } \big ( \{ \tilde { \mathbf { H } } _ { r } / \mathbf { H } _ { r } \} _ { t = 1 } ^ { T ^ { \prime } } \big ) \in \mathbb { R } ^ { T ^ { \prime } D } .\tag{8}
$$

For a mini-batch of � samples, we compute both global $( \tilde { \mathbf { H } } _ { r } ^ { g } , \mathbf { H } _ { r } ^ { g } )$ and local $( \tilde { \mathbf { H } } _ { r } ^ { l } , \mathbf { H } _ { r } ^ { l } )$ representations for each sample, resulting in 2� embeddings.

We apply the symmetric InfoNCE loss [8] to align each original–augmented pair while repelling other (negative) samples in the batch:

$$
\mathcal { L } _ { \mathrm { I n f o N C E } } ( \mathbf { u } , \mathbf { v } ) = - \log \frac { \exp ( \mathbf { u } ^ { \top } \mathbf { v } / \tau ) } { \sum _ { ( \mathbf { u } ^ { \prime } , \mathbf { v } ^ { \prime } ) \in B } \exp ( \mathbf { u } ^ { \top } \mathbf { v } ^ { \prime } / \tau ) } ,\tag{9}
$$

where (�, �) denotes a positive pair, $( \tilde { \mathbf { H } } _ { r } ^ { g } , \mathbf { H } _ { r } ^ { g } ) ( \mathcal { L } _ { \mathrm { c o n } _ { - } g } )$ and $( \tilde { \mathbf { H } } _ { r } ^ { l } , \mathbf { H } _ { r } ^ { l } ) ( \mathcal { L } _ { \mathrm { c o n } _ { - } l } ) .$ , and  is the set of all 2� embeddings in the current batch.

## 3.5. Multi-Objective Optimization

The total loss integrates six objectives that jointly address clinical requirements of AF detection, ECG reconstruction, domain generalization, and noise robustness:

$$
\begin{array} { r l } { \mathcal { L } _ { \mathrm { t o t a l } } = } & { \underbrace { \lambda _ { \mathrm { c l s } } \mathcal { L } _ { \mathrm { c l s } } } _ { \mathrm { A F D e t e c t i o n } } + \underbrace { \lambda _ { \mathrm { r e c } } \mathcal { L } _ { \mathrm { r e c } } } _ { \mathrm { E C G ~ R e c o n s t r u c t i o n } } } \\ & { + \underbrace { \lambda _ { \mathrm { v q } } \mathcal { L } _ { \mathrm { v q _ { \mathrm { { - } } r } q _ { \mathrm { { - } } r } } } + \lambda _ { \mathrm { v q } } \mathcal { L } _ { \mathrm { v q _ { \mathrm { { - } } m } } } } _ { \mathrm { Q u a n t i t a t i v e ~ L e a r n i n g } } + \underbrace { \lambda _ { \mathrm { c o n g } } \mathcal { L } _ { \mathrm { c o n _ { \mathrm { { - } } g } } } + \lambda _ { \mathrm { c o n l } } \mathcal { L } _ { \mathrm { c o n _ { \mathrm { { - } } } l } } } _ { \mathrm { C o n t r a s t i v e ~ L e a r n i n g } } , } \end{array}\tag{10}
$$

where $\mathcal { L } _ { \mathrm { c l s } } = \mathrm { C E } ( \mathbf { Y } , \hat { \mathbf { Y } } ) , \mathcal { L } _ { \mathrm { r e c } } = \| \mathbf { X } - \hat { \mathbf { X } } \| _ { 2 } .$

![](images/2f3cbbcd9d7a6a62b8f7bcf7861cddf8d1ce44d86f1de668a5ce28e8162f94f2.jpg)  
Figure 3: Experimental Protocol for Intra- and Cross-Dataset Evaluation of AF Detection Models.  
Table 1

Summary of the segmented datasets used in this study. The number of “Normal” and “AF” samples are reported after 10- second segmentation.

<table><tr><td>Dataset</td><td>Normal</td><td>AF</td><td>Hz</td><td>Country Leads</td></tr><tr><td>Chapman (Train)</td><td>6508</td><td>1416</td><td>500 China</td><td>12</td></tr><tr><td>CPSC2018 (Train)</td><td>1710</td><td>2215</td><td>500 China</td><td>12</td></tr><tr><td>Chapman (Test)</td><td>1617</td><td>364</td><td>500 China</td><td>12</td></tr><tr><td>CPSC2018 (Test)</td><td>423</td><td>531</td><td>500 China</td><td>12</td></tr><tr><td>AFDB</td><td>36718</td><td></td><td>22053 250 America</td><td>2</td></tr><tr><td>LTAF</td><td></td><td></td><td>187470 154590 128 America</td><td>2</td></tr><tr><td>CPSC2021</td><td></td><td></td><td>89570 47440 200 China</td><td>2</td></tr><tr><td>SPHDB</td><td></td><td></td><td>125000 125000 200 China</td><td>1</td></tr><tr><td>SHDBAF</td><td></td><td></td><td>614413 148651 200 Japan</td><td>2</td></tr></table>

## 4. Experiments

## 4.1. Datasets

To comprehensively evaluate the performance and generalizability of our proposed model for arbitrary AF detection and ECG reconstruction, we conducted experiments on seven publicly available ECG databases. The Chapman and CPSC2018 datasets were used for model development and internal validation via five-fold cross-validation. The remaining five datasets—AFDB, LTAF, CPSC2021, SPHDB, and SHDBAF—were reserved exclusively for external evaluation to assess the model’s generalizability across diverse populations, recording devices, and clinical settings, without any involvement in the training process. The complete experimental pipeline is illustrated in Figure 3.

Specifically, the Chapman dataset [58] is a large-scale 12-lead ECG database established through a collaboration between Chapman University, Shaoxing People’s Hospital, and Ningbo First Hospital. It comprises recordings from 45152 patients (500 Hz sampling rate, 10-second duration). All records are expert-annotated. For this study, we extracted a subset containing only “Normal” and “AF” labels, resulting in 9,905 segments.

The CPSC2018 dataset [38], originating from the 2018 China Physiological Signal Challenge, contains 6877 12- lead ECG recordings (500 Hz, 6–60 seconds in length) collected from 11 hospitals. The data encompasses various cardiac arrhythmias. We selected all samples labeled as “Normal” or “AF,” which amounted to 4889 segments after being uniformly segmented into 10-second clips.

For external zero-shot evaluation—where the model parameters remain fixed—we processed the following five independent databases, extracting only “Normal” and “AF” segments and segmenting all records into 10-second windows:

AFDB (MIT-BIH Atrial Fibrillation Database) [35]: Contains 23 fully annotated 10-hour Holter recordings (250 Hz, 2 leads) from Beth Israel Deaconess Medical Center, with beat-wise rhythm labels including AFIB, AFL, J, and N.

LTAF (Long-Term AF Database) [36]: Comprises 84 long-term ECG recordings (24–25 hours, 128 Hz, 2 leads) from subjects with paroxysmal or persistent AF. The heartbeat annotations (Normal/AF) were coordinated by MEDI-CALgorithmics and contributed to PhysioNet by Northwestern University.

CPSC2021 (the 2021 China Physiological Signal Challenge) [53]: A dataset specifically designed for paroxysmal AF (PAF) detection, featuring Holter recordings (200 Hz, 2 leads) with precise onset/ofset timestamps for AF episodes and three primary labels: AF, PAF, and Normal.

SPHDB (Shandong Provincial Hospital Database) [48]: Includes 24-hour wearable ECG recordings from 250 PAF patients (200 Hz, 1 lead), with all heartbeats clinically annotated as AF or non-AF.

SHDBAF (Saitama Heart Database Atrial Fibrillation) [50]: A Japanese open-source Holter database containing 128 24-hour recordings (200 Hz, 2 leads: modified CC5 and NASA). Of these, 98 recordings from 93 subjects have expert cardiologist-provided beat-wise annotations (AFIB, AFL, AT, PAT, N).

All ECG segments were resampled to a uniform sampling rate of 250 Hz prior to train and evaluation. The final statistics of the segmented datasets used in our experiments are summarized in Table 1.

## 4.2. Evaluation Metrics

We adopt distinct evaluation protocols for the two primary tasks. For the ECG lead reconstruction task, performance is measured using four signal fidelity metrics: the Pearson Correlation Coeficient (PCC) captures linear waveform similarity; the Coeficient of Determination (�<sup>2</sup>) reflects the proportion of variance in the ground-truth signal explained by the reconstruction; the Root Mean Squared Error (RMSE) emphasizes large deviations through quadratic weighting; and the Mean Absolute Error (MAE) provides an interpretable average error magnitude in microvolts (�V).

Table 3  
Table 2  
Performance Comparison of Diferent Models for Lead Reconstruction Performance on Chapman and CPSC2018 Datasets (mean ± std). ↑: higher is better; ↓: lower is better. Bold: best result; underlined: second-best result.
<table><tr><td rowspan="2">Model</td><td colspan="3">Chapman Lead I</td><td colspan="3">Chapman Lead I+II+V1</td><td colspan="3">CPSC2018 Lead I</td><td colspan="3">CPSC2018 Lead I+II+V1</td></tr><tr><td>PCC↑ R2↑</td><td>RMSE↓</td><td>MAE↓</td><td>PCC↑ R2↑</td><td>RMSE↓ MAE↓</td><td></td><td>PCC↑</td><td>R2↑</td><td>RMSE↓</td><td>MAE↓ PCC↑</td><td>R2↑</td><td>RMSE↓ MAE↓</td></tr><tr><td>ResCNN[34]</td><td>0.7828 0.6424</td><td>0.5983</td><td>0.3173</td><td>0.9349</td><td>0.8881 0.3402</td><td>0.1643</td><td>0.7519</td><td>0.5933</td><td>0.6406</td><td>0.3387</td><td>0.9132 0.8537 0.3853</td><td>0.1800</td></tr><tr><td>GRU[22]</td><td>(0.0012) (0.0016) 0.7770 0.6328</td><td>(0.0014) 0.6064</td><td>(0.0012) 0.3257</td><td>(0.0006) (0.0012) 0.9336 0.8844</td><td>(0.0020) 0.3460</td><td>(0.0016) 0.1768</td><td>(0.0013) 0.7541</td><td>(0.0053) 0.6025</td><td>(0.0042) (0.0016) 0.6332 0.3397</td><td>(0.0011) 0.9115</td><td>(0.0021) (0.0029) 0.8521 0.3875</td><td>(0.0022) 0.1910</td></tr><tr><td>EKGAN[20]</td><td>(0.0010) (0.0018) 0.7272 0.5342</td><td>(0.0015) 0.6801</td><td>(0.0023) 0.3714</td><td>(0.0011) (0.0021) 0.9139 0.8320</td><td>(0.0033) 0.4084</td><td>(0.0021) 0.1983</td><td>(0.0047) 0.7309</td><td>(0.0048) 0.5177</td><td>(0.0038) (0.0025) 0.6941 0.3652</td><td>(0.0006) 0.9190</td><td>(0.0014) (0.0019) 0.8441 0.3947</td><td>(0.0082) 0.1826</td></tr><tr><td>ESN[18]</td><td>(0.0011) (0.0122) 0.6977 0.5276</td><td>(0.0089) 0.6849</td><td>(0.0043) 0.3624</td><td>(0.0016) (0.0042) 0.9099 0.8261</td><td>(0.0050) 0.4156</td><td>(0.0047) 0.1921</td><td>(0.0025) 0.6617</td><td>(0.0202) 0.4825</td><td>(0.0144) 0.7191</td><td>(0.0048) (0.0026) 0.3754 0.8893</td><td>(0.0046) (0.0058) 0.7997 0.4474</td><td>(0.0043) 0.1988</td></tr><tr><td>DCGCNet(Ours)</td><td>(0.0003) (0.0014) 0.7865 0.6407</td><td>(0.0010) 0.5974</td><td>(0.0002) 0.3086</td><td>(0.0002) (0.0003) 0.9376 0.8811</td><td>(0.0004) 0.3437</td><td>(0.0003) 0.1530</td><td>(0.0012) 0.7583</td><td>(0.0010) 0.6012</td><td>(0.0007) 0.6313</td><td>(0.0006) (0.0004) 0.3259 0.9158</td><td>(0.0014) (0.0015) 0.8448 0.3938</td><td>(0.0010) 0.1714</td></tr></table>

Performance Comparison of Diferent Models for Twelve-Lead Atrial Fibrillation Detection on Chapman and CPSC2018 Datasets (mean ± std). Bold: best result
<table><tr><td rowspan="2">Model</td><td colspan="5">Chapman Dataset</td><td colspan="5">CPSC2018 Dataset</td></tr><tr><td>AUC</td><td>ACC</td><td>F1</td><td>Pre</td><td>Recall</td><td>AUC</td><td>ACC</td><td>F1</td><td>Pre</td><td>Recall</td></tr><tr><td rowspan="2">DNN[46]</td><td>0.9790</td><td>0.9332</td><td>0.8840</td><td>0.9016</td><td>0.8692</td><td>0.9841</td><td>0.9306</td><td>0.9301</td><td>0.9293</td><td>0.9330</td></tr><tr><td>(0.0027)</td><td>(0.0033)</td><td>(0.0053)</td><td>(0.0097)</td><td>(0.0080)</td><td>(0.0019)</td><td>(0.0099)</td><td>(0.0097)</td><td>(0.0090)</td><td>(0.0082)</td></tr><tr><td rowspan="2">DenseRNN[24]</td><td>0.9974</td><td>0.9694</td><td>0.9459</td><td>0.9759</td><td>0.9217</td><td>0.9965</td><td>0.9157</td><td>0.9156</td><td>0.9198</td><td>0.9238</td></tr><tr><td>(0.0003)</td><td>(0.0047)</td><td>(0.0089)</td><td>(0.0034)</td><td>(0.0137)</td><td>(0.0005)</td><td>(0.0201)</td><td>(0.0201)</td><td>(0.0158)</td><td>(0.0181)</td></tr><tr><td rowspan="2">ConvRGNN[42]</td><td>0.9627</td><td>0.9318</td><td>0.8832</td><td>0.8948</td><td>0.8739</td><td>0.9545</td><td>0.8889</td><td>0.8875</td><td>0.8878</td><td>0.8881</td></tr><tr><td>(0.0048)</td><td>(0.0040)</td><td>(0.0095)</td><td>(0.0086)</td><td>(0.0201)</td><td>(0.0091)</td><td>(0.0109)</td><td>(0.0112)</td><td>(0.0113)</td><td>(0.0120)</td></tr><tr><td rowspan="2">MSGformer[19]</td><td>0.9863</td><td>0.9556</td><td>0.9246</td><td>0.9322</td><td>0.9183</td><td>0.9962</td><td>0.9730</td><td>0.9727</td><td>0.9721</td><td>0.9735</td></tr><tr><td>(0.0033)</td><td>(0.0052)</td><td>(0.0102)</td><td>(0.0077)</td><td>(0.0194)</td><td>(0.0004)</td><td>(0.0044)</td><td>(0.0043)</td><td>(0.0049)</td><td>(0.0036)</td></tr><tr><td rowspan="2">ECGMamba[43]</td><td>0.9910</td><td>0.9637</td><td>0.9385</td><td>0.9458</td><td>0.9317</td><td>0.9931</td><td>0.9602</td><td>0.9597</td><td>0.9595</td><td>0.9607</td></tr><tr><td>(0.0012)</td><td>(0.0018)</td><td>(0.0034)</td><td>(0.0055)</td><td>(0.0076)</td><td>(0.0014)</td><td>(0.0036)</td><td>(0.0036)</td><td></td><td>(0.0039)</td></tr><tr><td rowspan="2">RawECGNet[4]</td><td>0.9951</td><td>0.9746</td><td>0.9583</td><td>0.9506</td><td>0.9670</td><td>0.9970</td><td>0.9819</td><td>0.9817</td><td>(0.0041) 0.9817</td><td>0.9819</td></tr><tr><td>(0.0027)</td><td>(0.0074)</td><td>(0.0126)</td><td>(0.0107)</td><td>(0.0187)</td><td>(0.0012)</td><td>(0.0041)</td><td>(0.0041)</td><td></td><td></td></tr><tr><td rowspan="2">TTDADualNet[47]</td><td>0.9946</td><td>0.9763</td><td>0.9611</td><td>0.9535</td><td>0.9693</td><td>0.9974</td><td>0.9811</td><td>0.9809</td><td>(0.0042) 0.9804</td><td>(0.0043) 0.9816</td></tr><tr><td>(0.0019)</td><td>(0.0025)</td><td>(0.0042)</td><td>(0.0043)</td><td>(0.0071)</td><td>(0.0019)</td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="2">DCGCNet(Ours)</td><td>0.9998</td><td>0.9966</td><td>0.9943</td><td>0.9924</td><td>0.9962</td><td>0.9991</td><td>(0.0060) 0.9937</td><td>(0.0060) 0.9936</td><td>(0.0062)</td><td>(0.0058)</td></tr><tr><td>(0.0001)</td><td>(0.0013)</td><td>(0.0021)</td><td>(0.0032)</td><td>(0.0015)</td><td>(0.0005)</td><td>(0.0017)</td><td>(0.0017)</td><td>0.9932 (0.0016)</td><td>0.9941 (0.0018)</td></tr></table>

For the AF detection task, we report five standard segment-level classification metrics: the Area Under the ROC Curve (AUC) assesses threshold-invariant discriminative ability; Accuracy (ACC) indicates overall correctness; the F1-score (F1) balances precision and recall via their harmonic mean; Precision (Pre) quantifies the reliability of positive predictions; and Recall measures sensitivity to true AF episodes.

## 4.3. Implementation Details

All experiments were implemented in PyTorch 2.4.1 with CUDA 11.5 acceleration and conducted on a workstation equipped with an NVIDIA GeForce RTX 3090 GPU (24 GB VRAM) and an Intel Core i7-12700KF CPU. The model architecture in Section III employed a codebook size of � = 512 and a vector quantization loss weight $\beta = 0 . 2 5$ The composite loss function was defined as $\mathcal { L } = \lambda _ { \mathrm { c l s } } \mathcal { L } _ { \mathrm { c l s } } \mathcal { + }$ + $\lambda _ { \mathrm { r e c } } \mathcal { L } _ { \mathrm { r e c } } + \lambda _ { \mathrm { v q } } \mathcal { L } _ { \mathrm { v q } } + \lambda _ { \mathrm { c o n g } } \mathcal { L } _ { \mathrm { c o n g } } + \lambda _ { \mathrm { c o n l } } \mathcal { L } _ { \mathrm { c o n l } } ,$ , with weights $\lambda _ { \mathrm { c l s } } = 1 , \lambda _ { \mathrm { r e c } } = \dot { 1 , } \lambda _ { \mathrm { v q } } = \mathsf { \bar { 0 } } . 2 , \bar { \lambda } _ { \mathrm { c o n g } } = 0 . 5$ , and $\lambda _ { \mathrm { c o n l } } = 0 . 1$ A Gumbel-Softmax temperature of $\tau = 0 . 1$ was used for discrete token sampling.

In our contrastive learning setup, we applied two ECGspecific data augmentation strategies to generate positive pairs for training: (i) additive Gaussian noise with standard deviation sampled uniformly from [0.1, 0.8] relative to the signal amplitude; and (ii) baseline wander with amplitude in [0.1, 0.8], and frequency in [0.05, 0.3] Hz. The model was optimized using the Adam optimizer with a batch size of 16, an initial learning rate of $1 \times 1 0 ^ { - 4 }$ , weight decay of $1 \times 1 0 ^ { - 4 }$ and trained for 50 epochs without learning rate scheduling.

## 4.4. Comparative Experiments

We compared our proposed DCGCNet against a diverse set of state-of-the-art baselines on two core tasks: ECG lead reconstruction and twelve-lead AF detection. To ensure a fair comparison, we reproduced all baseline methods under the same experimental setup.

For the reconstruction task, we evaluated all models on the Chapman and CPSC2018 datasets under two input configurations—single-lead (Lead I) and three-lead (Leads I+II+V1)—using four fidelity metrics: PCC, $R ^ { 2 }$ , RMSE, and MAE. Baseline methods include ResCNN[34] (Original task: reconstructed 12-lead ECG from leads I, II, V3 using residual CNNs), GRU[22] (Original task: from I and II via GRU), EKGAN[20] (Original task: from lead I via GAN),

(a) Train on the CPSC2018.

![](images/627bc688c4f2caaa16c7592fe4aa914350cf0c38f46afa0d15276a0717db295e.jpg)  
Figure 4: Cross-dataset generalization performance radar plots on AF detection comparing DCGCNet against baseline methods under three input strategies (fixed, arbitrary, reconstruction). (a) Models trained on CPSC2018 and evaluated on six external test sets. (b) Models trained on Chapman and tested on six datasets. Metrics shown include AUC, Accuracy, F1-score, Precision, and Recall.

and ESN[18] (Original task: from I and V1 using echo state networks). As shown in Table 2, DCGCNet consistently achieved the best or second-best performance across majority settings. Although ResCNN yielded slightly higher �<sup>2</sup> on Chapman Lead I and CPSC2018 Lead I+II+V1, DCGCNet exhibited more balanced improvements across all metrics. A reconstruction result was shown in Figure A.1

For AF detection, we benchmarked DCGCNet against seven recent deep learning architectures on the same datasets, reporting five standard classification metrics at the segment level. DNN[46] employed a standard deep neural network operating directly on 12-lead ECG inputs. DenseRNN[24] integrated bidirectional recurrent units into a densely connected residual architecture to model temporal dynamics across all leads. ConvRGNN[42] formulated the 12-lead

ECG as a graph and leveraged convolutional residual graph neural networks to capture inter-lead spatial dependencies. MSGformer[19] adopted a multi-scale grid attention mechanism to extract hierarchical temporal features from 12-lead signals. ECGMamba[43] utilized a bidirectional state space model for eficient sequence modeling and classification. Additionally, two recent generalizable approaches were included: RawECGNet[4], which incorporated batch normalization and an uncertainty-aware domain-shift module to improve robustness, and TTDADualNet[47], which addressed cross-dataset generalization through test-time data augmentation.

As summarized in Table 3, DCGCNet established a new state of the art. On the Chapman dataset, it achieved an AUC of 0.9998 and an accuracy of 0.9966, with an F1-score that surpassed the strongest baseline (TTDADualNet) by 3.32 percentage points. On CPSC2018, DCGCNet improved accuracy by 1.26% and F1-score by 1.27% over TTDADual-Net, while maintaining exceptional precision-recall balance. These results highlighted the efectiveness of our dual-path architecture and contextual graph-based representation in jointly capturing local rhythm dynamics and global interlead morphological dependencies. In addition, we gave the comparison of the results of the original literature in the TableA.1.

![](images/53cc900a42931fe162c1fb4a2ec648fa75edbda07cc0e730c2dccef74cfebff3.jpg)  
Figure 5: Robustness of DCGCNet and RawECGNet on AF detection under increasing levels of ECG artifacts. AUC performance is plotted across four training–testing configurations as a function of noise intensity, defined as the ratio of artifact amplitude to original ECG amplitude (Ratio). Each curve shows the mean AUC over five independent runs, with error bars indicating the corresponding standard deviation. The clinically relevant 10 dB SNR level (Ratio ≈ 0.32)—representative of typical ECG interference in wearable settings—is highlighted by white markers.

## 4.5. Generalization Experiments

To rigorously evaluate the cross-dataset generalization capability of the proposed DCGCNet, we conducted comprehensive experiments under two distinct training regimes: (1) models trained on Chapman and evaluated on six external test sets (AFDB, LTAF, CPSC2021, SPHDB, SHDBAF, and CPSC2018); and (2) models trained on CPSC2018 and tested on the same five datasets and Chapman. This design enabled a thorough assessment of model robustness across diverse acquisition protocols, patient populations, and labeling criteria.

For each generalization baseline method, RawECGNet[4] and TTDA-DualNet[47], we implemented three variants corresponding to diferent input handling strategies: Fixed: trained on standard 12-lead ECGs and directly applied to test data without adaptation; Arbitrary: trained on synthetically masked 12-lead ECGs (simulating variable lead availability) to learn from arbitrary lead configurations; Reconstruction: trained on full 12-lead ECGs but, during inference, first reconstructed missing leads using a ResCNN[34] before classification.

In contrast, our DCGCNet jointly optimized lead reconstruction and AF detection in an end-to-end manner, explicitly designed for arbitrary-lead inputs. All methods were reimplemented under identical experimental conditions—including data preprocessing, optimizer settings, and evaluation metrics—to ensure a fair comparison.

As visualized in Fig. 4, DCGCNet consistently achieved state-of-the-art performance across all test sets. Notably, when trained on Chapman and tested on CPSC2018 (both 12-lead datasets), the fixed and reconstruction variants yielded identical results, as no lead imputation was required—a pattern symmetrically observed in the reverse setting (CPSC2018 → Chapman). In more challenging cross-domain scenarios involving variable lead availability (e.g., AFDB, SPHDB), the arbitrary and reconstruction strategies substantially improved over thefixed baseline, yet still lagged behind DCGCNet by a clear margin in AUC and F1-score. These results demonstrated that the learnable codebook served as a unified representation bridge between reconstruction and classification, and its prototypical nature enabled it to generalize efectively across diverse clinical datasets.

## 4.6. Robust Experiments

To rigorously assess the robustness of DCGCNet under realistic clinical noise conditions, we conducted comprehensive experiments by progressively corrupting clean ECG segments with three dominant artifact types: baseline wander (BLW), powerline interference (PLI), and electromyographic (EMG) artifact. The corruption intensity was controlled by an SNR-equivalent parameter defined as the ratio

Table 6

of artifact amplitude to the original ECG amplitude (denoted as Ratio). Following established literature[3], which reported an average EMG artifact intensity of about 10 dB SNR $R a t i o \approx 0 . 3 2 ~ ,$ , we adopted this level as a clinically relevant reference for ambulatory ECG. Although BLW and PLI were not similarly quantified, we used the same 10 dB SNR for all three artifacts to ensure consistent evaluation. A specific example of adding noise was shown in Figure A.2.

Figure 5 showed the AUC performance of DCGC-Net across four cross-dataset configurations as Ratio increases from 0 to 1. Each curve represented the mean AUC over five independent runs, with error bars indicating the standard deviation. The results clearly demonstrated that DCGCNet consistently outperformed RawECGNet under all noise conditions. The 10 dB operating point (Ratio=0.32) was explicitly marked by white circular markers. The results demonstrated that DCGCNet maintained consistently high performance even under severe noise corruption, with only marginal degradation at the clinically relevant 10 dB level—underscoring its robustness and practical applicability in real-world ECG monitoring scenarios.

## 4.7. Ablation Experiments

We conducted ablation studies to assess the contribution of each component in DCGCNet across three settings: intra-dataset performance, cross-dataset generalization, and robustness to ECG noise.

Intra-dataset performance (Table 4) showed that the reconstruction (rec) and classification (cls) tasks were mutually beneficial—each task enhances the learning of features relevant to the other. This also demonstrated that removing either the rhythm or morphology encoder degraded AF detection and signal reconstruction, respectively, highlighting their complementary roles. The Heterogeneous Graph Interaction (HGI), Adaptive Codebook Vector Quantizer (ACVQ) and Local-Global Contrastive Module (LGCM) also contributed notably to reconstruction fidelity and classification accuracy. Cross-dataset generalization (Table 5) revealed that discrete representation learning was critical for domain transfer. Removing the ACVQ caused substantially larger AUC drops than ablating the basic Adaptive Codebook (AC), confirming ACVQ’s importance in bridging domain gaps. Robustness to noise (Table 6) demonstrated that both ACVQ and LGCM were essential under common artifacts—BLW, PLI, and EMG.

Together, these results validated that the integration of Rhythm-Morphology Dual Encoding, ACVQ, LGCM, and HGI was key to DCGCNet’s strong and reliable AF detection performance. In addition, the sensitivity analysis of the loss function was shown in the TableA.2.

## 4.8. Visualization

To investigate the generalization capability of the learned discrete morphological codebook, we visualized the usage frequency of individual codebook vectors across diferent classes and datasets. As shown in Figure 6, we compared intra-domain and cross-domain evaluation settings using ECG databases: CPSC2018 and Chapman. When the model was trained on CPSC2018 (Fig. 6a–b), the most frequently activated codebook entries for class 0 (e.g., indices 47, 353, 327) and class 1 (e.g., 197, 279, 296) remained highly consistent between the in-domain test set (CPSC2018) and the out-of-domain test set (Chapman). Similarly, a model trained on Chapman (Fig. 6c–d) exhibited stable usage of class-specific codebook vectors—such as indices 187, 144, and 423 for class 0, and 45, 325, 458 for class 1—across both its native test set and the CPSC2018 dataset. This consistency under distribution shift strongly suggests that the codebook does not merely memorize dataset-specific artifacts but instead encodes class-discriminative morphological primitives that are shared across populations and recording protocols. In addition, we presented Grad-CAM interpretability analysis in the FigureA.3.

Ablation study on intra-dataset: Lead I ECG reconstruction (PCC) and AF detection (AUC).
<table><tr><td rowspan="2">Setting</td><td colspan="2">Rec. (PCC)</td><td colspan="2">Cls. (AUC)</td></tr><tr><td>Chapman</td><td>CPSC2018</td><td>Chapman</td><td>CPSC2018</td></tr><tr><td>DCGCNet</td><td>0.7865</td><td>0.7583</td><td>0.9998</td><td>0.9991</td></tr><tr><td>w/o rec</td><td></td><td></td><td>0.9883</td><td>0.9749</td></tr><tr><td> $w / \circ$  cls</td><td>0.7751</td><td>0.7553</td><td></td><td></td></tr><tr><td> $\mathsf { w } / \mathsf { o }$  rhythm</td><td>0.7812</td><td>0.7521</td><td>0.9823</td><td>0.9754</td></tr><tr><td> $\mathsf { w } / \mathsf { o }$  morph</td><td>0.7205</td><td>0.6957</td><td>0.9986</td><td>0.9972</td></tr><tr><td> ${ \mathsf w } / { \mathsf o } ~ { \mathsf { A C } }$ </td><td>0.7398</td><td>0.7092</td><td>0.9865</td><td>0.9798</td></tr><tr><td> $\mathsf { w } / \mathsf { o } \mathsf { A C V Q }$ </td><td>0.7347</td><td>0.7043</td><td>0.9852</td><td>0.9784</td></tr><tr><td> $w / 0 L G C M \_ L$ </td><td>0.7768</td><td>0.7425</td><td>0.9948</td><td>0.9897</td></tr><tr><td> $w / \circ \mathsf { L G C M } _ { - } ^ { - } \mathsf { G }$ </td><td>0.7789</td><td>0.7462</td><td>0.9959</td><td>0.9915</td></tr><tr><td> $\mathsf { w } / \mathsf { o } \mathsf { L G C M }$ </td><td>0.7724</td><td>0.7438</td><td>0.9927</td><td>0.9873</td></tr><tr><td> $\mathsf { w } / \mathsf { o }$  HGl</td><td>0.7742</td><td>0.7487</td><td>0.9863</td><td>0.9804</td></tr></table>

Ablation study on cross-dataset generalization for AF detection (trained on Chapman).
<table><tr><td>Setting</td><td>CPSC2018(12 Leads) AFDB(2 Leads) SPHDB(1 Lead)</td><td></td></tr><tr><td>DCGCNet</td><td>0.9967</td><td>0.9947</td></tr><tr><td>w/o AC</td><td>0.9792</td><td>0.9624</td></tr><tr><td>w/o ACVQ</td><td>0.9658</td><td>0.9431</td></tr></table>

Ablation study on robustness to common ECG noises (ratio=0.333) —BLW, PLI, and EMG—within the intra-dataset.
<table><tr><td rowspan="2">Setting</td><td colspan="3">Chapman</td><td colspan="3">CPSC2018</td></tr><tr><td>BLW</td><td>PLI</td><td>EMG</td><td>BLW</td><td>PLI</td><td>EMG</td></tr><tr><td>DCGCNet</td><td>0.9941</td><td>0.9952</td><td>0.9889</td><td>0.9863</td><td>0.9914</td><td>0.9807</td></tr><tr><td>w/o ACVQ</td><td>0.9624</td><td>0.9681</td><td>0.9432</td><td>0.9415</td><td>0.9523</td><td>0.9218</td></tr><tr><td>w/o LGCM L</td><td>0.9832</td><td>0.9847</td><td>0.9603</td><td>0.9684</td><td>0.9721</td><td>0.9456</td></tr><tr><td>w/o LGCM G</td><td>0.9756</td><td>0.9802</td><td>0.9714</td><td>0.9592</td><td>0.9667</td><td>0.9523</td></tr><tr><td>w/o LGCM</td><td>0.9518</td><td>0.9584</td><td>0.9276</td><td>0.9237</td><td>0.9341</td><td>0.9024</td></tr></table>

## 5. Conclusion

In this work, we present DCGCNet, the first codebookbasedjoint learning framework for atrial fibrillation (AF) detection that simultaneously optimizes classification accuracy and ECG signal reconstruction. By introducing a rhythm codebook to capture local discriminative patterns and a morphology codebook to preserve global waveform fidelity, our model enables synergistic representation learning. Extensive experiments demonstrate state-of-the-art performance across diverse and challenging scenarios, including intradataset 12-lead evaluation, cross-dataset arbitrary-lead generalization, and noisy real-world conditions, highlighting its robustness and clinical applicability. Furthermore, we provide multi-level interpretability: statistical analysis of codebook usage reveals consistent activation patterns across domains, ofering insight into the model’s generalization mechanism; while Grad-CAM visualizations confirm that diagnostic decisions are grounded in clinically relevant ECG segments. These attributes collectively advance trustworthy AI for AF detection. Future work will explore extending the framework to multi-class arrhythmia diagnosis and real-time deployment on edge devices.

![](images/4c5545cff3142719cd85bb598860688591ed9e4475671d626817d3993fc9321e.jpg)

![](images/8c52b64dac216c80583b37df04d42dd7d92720a8f5f21829e1f36f98d9a9008a.jpg)

![](images/870b460da4b7689739e1e6f218fe653ad50057ce781945857e39ed4bcd8c9f80.jpg)  
(a) Train on the CPSC2018, Test on the CPSC2018.

![](images/5e1a92171b2328b6751018125b389c10b9c9dc028300fbbfadb4dedb09eca23b.jpg)

![](images/547e2f14912c4a5fec46d5818854c03b59e9e78cd281b169fa78a301ac4a6b0e.jpg)

![](images/c6d7d58003a22e2a3a74e28592516b7ad7e1cb696ce00bc7dbbc201b06bb3e94.jpg)

![](images/685b19e84d55ec4e16804eb375b7929ef13bb81749ce8eb844b9153ae2d5d9ec.jpg)  
(b) Train on the CPSC2018, Test on the Chapman.

![](images/7c0b975225787c26e8a2fc8d1f1f0ca169e50ebd14ab738b97797b27b8694b3d.jpg)

(c) Train on the Chapman, Test on the Chapman.  
![](images/d668b24ad5dc29b598e854b6d6de686c860b1ead3dcf6cab6f838165960b5daf.jpg)

![](images/c903db39ecdedf019daaaa28e22b0ad546ff0a37d5dfb58a7d2379eb7b0f7f96.jpg)

![](images/0a32a48340833fac13e18413f2261003e35a57319260d21bbe8dbf49e82c93d5.jpg)  
(d) Train on the Chapman, Test on the CPSC2018.  
Figure 6: Codebook usage patterns across training and testing domains reveal consistent, label-specific prototypes, demonstrating the generalizability of learned discrete representations. (a) Model trained on CPSC2018 and tested on CPSC2018; (b) Same model tested on Chapman (out-of-domain). (c) Model trained on Chapman and tested on Chapman; (d) Same model tested on CPSC2018 (out-of-domain).

## Declaration of Competing Interests

The authors declare no conflicts of interest.

## Funding Sources

This research received no specific grant.

## Declaration of Generative AI Use

The authors confirm that this manuscript was prepared with the assistance of the Qwen3 large language model for language polishing and formatting. The use of this AI tool did not influence the scientific content, data analysis, or interpretation of results presented in this work. All authors have reviewed and approved the final version of the manuscript.

## Data Statement

The following publicly available datasets were used in this study: Chapman: https://doi.org/10.13026/wgex-er52; CPSC2018: http://2018.icbeb.org/Challenge.html; AFDB: https://www.physionet.org/content/afdb/1.0.0/; LTAFDB: https://physionet.org/content/ltafdb/1.0.0/; CPSC2021: https://doi.org/10.13026/ksya-qw89; SPHDB: https://doi. org/10.17632/dvb5mnhfc4.1; SHDBAF: https://doi.org/10. 13026/n6yq-fq90. The source code is publicly available at https://github.com/Ou-Young-1999/DCGCNet.

## Author Contributions

Conceptualization: H. Li, X. Ouyang; Data curation: H. Li, J. Xiao, Y. Lai, S. Lv; Formal analysis: H. Li, X. Ouyang, J. Wei; Investigation: H. Li, X. Ouyang, J. Xiao, Y. Lai, S. Lv; Methodology: H. Li, X. Ouyang, J. Wei; Project administration: H. Li, X. Ouyang, J. Xiao; Resources: H. Li, J. Xiao; Software: X. Ouyang; Supervision: H. Li; Validation: H. Li, X. Ouyang, J. Wei, G. Li, Y. Lei, G. Ma; Visualization: X. Ouyang, Y. Lei; Writing – original draft: X. Ouyang; Writing – review & editing: H. Li, X. Ouyang, J. Wei, G. Li, Y. Lei, G. Ma.

## References

[1] Andersen, R.S., Peimankar, A., Puthusserypady, S., 2019. A deep learning approach for real-time detection of atrial fibrillation. Expert Systems with Applications 115, 465–473.

[2] Asgari, S., Mehrnia, A., Moussavi, M., 2015. Automatic detection of atrial fibrillation using stationary wavelet transform and support vector machine. Computers in biology and medicine 60, 132–142.

[3] Atanasoski, V., Petrović, J., Maneski, L.P., Miletić, M., Babić, M., Nikolić, A., Panescu, D., Ivanović, M.D., 2024. A morphologypreserving algorithm for denoising of emg-contaminated ecg signals. IEEE Open Journal of Engineering in Medicine and Biology 5, 296– 305.

[4] Ben-Moshe, N., Tsutsui, K., Brimer, S.B., Zvuloni, E., Sörnmo, L., Behar, J.A., 2024. Rawecgnet: Deep learning generalization for atrial fibrillation detection from the raw ecg. IEEE Journal of Biomedical and Health Informatics 28, 5180–5188.

[5] Biton, S., Aldhafeeri, M., Marcusohn, E., Tsutsui, K., Szwagier, T., Elias, A., Oster, J., Sellal, J.M., Suleiman, M., Behar, J.A., 2023. Generalizable and robust deep learning algorithm for atrial fibrillation diagnosis across geography, ages and sexes. NPJ Digital Medicine 6, 44.

[6] Cai, W., Chen, Y., Guo, J., Han, B., Shi, Y., Ji, L., Wang, J., Zhang, G., Luo, J., 2020. Accurate detection of atrial fibrillation from 12-lead ecg using deep neural network. Computers in biology and medicine 116, 103378.

[7] Chang, Y., Qin, J., Qiao, L., Wang, X., Zhu, Z., Ma, L., Wang, X., 2025. Scalable training for vector-quantized networks with 100% codebook utilization. arXiv preprint arXiv:2509.10140 .

[8] Chen, T., Kornblith, S., Norouzi, M., Hinton, G., 2020. A simple framework for contrastive learning of visual representations, in: International conference on machine learning, PmLR. pp. 1597–1607.

[9] Chugh, S.S., Havmoeller, R., Narayanan, K., Singh, D., Rienstra, M., Benjamin, E.J., Gillum, R.F., Kim, Y.H., McAnulty Jr, J.H., Zheng,

Z.J., et al., 2014. Worldwide epidemiology of atrial fibrillation: a global burden of disease 2010 study. Circulation 129, 837–847.

[10] Elliott, A.D., Middeldorp, M.E., Van Gelder, I.C., Albert, C.M., Sanders, P., 2023. Epidemiology and modifiable risk factors for atrial fibrillation. Nature Reviews Cardiology 20, 404–417.

[11] Epmoghaddam, D., Banta, A., Post, A., Razavi, M., Aazhang, B., 2025. Reconstructing 12-lead ecg from reduced lead sets using an encoder–decoder convolutional neural network. Biomedical Signal Processing and Control 104, 107486.

[12] Guan, X., Lai, Y., Jin, J., Li, J., Wang, H., Zhao, Q., Zhang, D., Geng, S., Hong, S., 2025. Reconstructing 12-lead ecg from 3-lead ecg using variational autoencoder to improve cardiac disease detection of wearable ecg devices. arXiv preprint arXiv:2510.11442 .

[13] Guhdar, M., Mstafa, R.J., Mohammed, A.O., 2025. A novel data augmentation strategy for robust deep learning classification of biomedical time-series data: Application to ecg and eeg analysis. arXiv preprint arXiv:2507.12645 .

[14] Hagiwara, Y., Fujita, H., Oh, S.L., Tan, J.H., San Tan, R., Ciaccio, E.J., Acharya, U.R., 2018. Computer-aided diagnosis of atrial fibrillation based on ecg signals: A review. Information Sciences 467, 99–114.

[15] He, R., Wang, K., Zhao, N., Liu, Y., Yuan, Y., Li, Q., Zhang, H., 2018. Automatic detection of atrial fibrillation based on continuous wavelet transform and 2d convolutional neural networks. Frontiers in physiology 9, 1206.

[16] Hindricks, G., Potpara, T., Dagres, N., Arbelo, E., Bax, J.J., Blomström-Lundqvist, C., Boriani, G., Castella, M., Dan, G.A., Dilaveris, P.E., et al., 2021. 2020 esc guidelines for the diagnosis and management of atrial fibrillation developed in collaboration with the european association for cardio-thoracic surgery (eacts) the task force for the diagnosis and management of atrial fibrillation of the european society of cardiology (esc) developed with the special contribution of the european heart rhythm association (ehra) of the esc. European heart journal 42, 373–498.

[17] Hirsch, G., Jensen, S.H., Poulsen, E.S., Puthusserypady, S., 2021. Atrial fibrillation detection using heart rate variability and atrial activity: A hybrid approach. Expert Systems with Applications 169, 114452.

[18] Jančiulevičiut¯ e, K., Sokas, D., Daukantas, S., Sörnmo, L., Petr˙ enas,˙ A., 2025. An echo state network for synthesizing the standard 12-lead ecg from a two-lead ecg obtained from a single touch of a wrist-worn device. Biomedical Signal Processing and Control 109, 108008.

[19] Ji, C., Wang, L., Qin, J., Liu, L., Han, Y., Wang, Z., 2024. Msgformer: A multi-scale grid transformer network for 12-lead ecg arrhythmia detection. Biomedical Signal Processing and Control 87, 105499.

[20] Joo, J., Joo, G., Kim, Y., Jin, M.N., Park, J., Im, H., 2023. Twelvelead ecg reconstruction from single-lead signals using generative adversarial networks, in: International Conference on Medical Image Computing and Computer-Assisted Intervention, Springer. pp. 184– 194.

[21] Kim, M., Shin, M., 2026. Enhancing ecg classification generalization through unified multi-dataset training. Sensors 26, 1830.

[22] Kim, M.G., Zhang, G., Li, M., Jung, J., 2025. Evaluating timeseries deep learning models for accurate and eficient reconstruction of clinical 12-lead ecg signals. IEEE Access .

[23] Kim, S., Lim, J., Jang, J., 2024. Seqafnet: A beat-wise sequential neural network for atrial fibrillation classification in adhesive patchtype electrocardiographs. IEEE Journal of Biomedical and Health Informatics 28, 5260–5269.

[24] Laghari, A.A., Sun, Y., Alhussein, M., Aurangzeb, K., Anwar, M.S., Rashid, M., 2023. Deep residual-dense network based on bidirectional recurrent neural network for atrial fibrillation detection. Scientific reports 13, 15109.

[25] Lence, A., Granese, F., Fall, A., Hanczar, B., Salem, J.E., Zucker, J.D., Prifti, E., 2025. Ecgrecover: a deep learning approach for electrocardiogram signal completion, in: Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 1, pp. 2359–2370.

[26] Li, H., Wei, J., Xiao, J., Lai, Y., Liu, M., Lv, S., Ouyang, X., 2026. Robust and generalizable atrial fibrillation detection from ecg using time-frequency fusion and supervised contrastive learning. arXiv preprint arXiv:2601.10202 .

[27] Lian, J., Wang, L., Muessig, D., 2011. A simple method to detect atrial fibrillation using rr intervals. The American journal of cardiology 107, 1494–1497.

[28] Lin, C., Lu, H., Sang, P., Pan, C., 2025. A knowledge embedded multimodal pseudo-siamese model for atrial fibrillation detection. Scientific Reports 15, 3133.

[29] Liu, H., Zhao, Z., Chen, X., Yu, R., She, Q., 2020. Using the vqvae to improve the recognition of abnormalities in short-duration 12- lead electrocardiogram records. Computer Methods and Programs in Biomedicine 196, 105639.

[30] Liu, M., Zhang, H., 2025. A personalized ecg lead reconstruction method based on denoising difusion model, in: Proceedings of the 2025 International Conference on Artificial Intelligence and Computational Intelligence, pp. 170–175.

[31] Liu, Q., Zhou, Y., He, J., Hou, J., Zhang, D., Cui, Q., Shi, F., Zhang, H., Zang, H., 2026. Af-ecgnet: An efective model for atrial fibrillation intelligent detection based on the improved transformer model. Biomedical Signal Processing and Control 116, 109493.

[32] Long, S., Zhou, Q., Jiang, X., Ying, C., Ma, L., Luo, Y., 2025. Domain generalization via discrete codebook learning, in: 2025 IEEE International Conference on Multimedia and Expo (ICME), IEEE. pp. 1–6.

[33] Ma, C., Liu, C., Wang, X., Li, Y., Wei, S., Lin, B.S., Li, J., 2022. A multistep paroxysmal atrial fibrillation scanning strategy in long-term ecgs. IEEE Transactions on Instrumentation and Measurement 71, 1–10.

[34] Mason, F., Pandey, A.C., Gadaleta, M., Topol, E.J., Muse, E.D., Quer, G., 2024. Ai-enhanced reconstruction of the 12-lead electrocardiogram via 3-leads with accurate clinical assessment. NPJ Digital Medicine 7, 201.

[35] Moody, G., 1983. A new method for detecting atrial fibrillation using rr intervals. Proc. Comput. Cardiol. 10, 227–230.

[36] Moody, G.E., 2004. Spontaneous termination of atrial fibrillation: a challenge from physionet and computers in cardiology 2004, in: Computers in Cardiology, 2004, IEEE. pp. 101–104.

[37] Musellim, S., Han, D.K., Jeong, J.H., Lee, S.W., 2022. Prototypebased domain generalization framework for subject-independent brain-computer interfaces, in: 2022 44th Annual International Conference of the IEEE Engineering in Medicine & Biology Society (EMBC), IEEE. pp. 711–714.

[38] Ng, E.Y.K., Liu, F., Liu, C., Zhao, L., Zhang, X., Wu, X., Xu, X., Liu, Y., Ma, C., Wei, S., He, Z., Li, J., 2018. An open access database for evaluating the algorithms of electrocardiogram rhythm and morphology abnormality detection. Journal of Medical Imaging and Health Informatics .

[39] Ng, Y., Liao, M.T., Chen, T.L., Lee, C.K., Chou, C.Y., Wang, W., 2023. Few-shot transfer learning for personalized atrial fibrillation detection using patient-based siamese network with single-lead ecg records. Artificial Intelligence in Medicine 144, 102644.

[40] Obianom, E.N., Ng, G.A., Li, X., 2025. Reconstruction of 12-lead ecg: a review of algorithms. Frontiers in Physiology 16, 1532284.

[41] Presacan, O., Dorobanţiu, A., Isaksen, J.L., Willi, T., Graf, C., Riegler, M.A., Sridhar, A.R., Kanters, J.K., Thambawita, V., 2025. Evaluating the feasibility of 12-lead electrocardiogram reconstruction from limited leads using deep learning. Communications medicine 5, 139.

[42] Qiang, Y., Dong, X., Liu, X., Yang, Y., Fang, Y., Dou, J., 2024a. Convrgnn: An eficient convolutional residual graph neural network for ecg classification. Computer Methods and Programs in Biomedicine 257, 108406.

[43] Qiang, Y., Dong, X., Liu, X., Yang, Y., Fang, Y., Dou, J., 2024b. Ecgmamba: Towards eficient ecg classification with bissm. arXiv preprint arXiv:2406.10098 .

[44] Rajotte, K.J., Islam, B., Huang, X., McManus, D.D., Clancy, E.A., 2025. Ecg statement classification and lead reconstruction using cnnbased models. IEEE Journal of Biomedical and Health Informatics

[45] Razavi, A., Van den Oord, A., Vinyals, O., 2019. Generating diverse high-fidelity images with vq-vae-2. Advances in neural information processing systems 32.

[46] Ribeiro, A.H., Ribeiro, M.H., Paixão, G.M., Oliveira, D.M., Gomes, P.R., Canazart, J.A., Ferreira, M.P., Andersson, C.R., Macfarlane, P.W., Meira Jr, W., et al., 2020. Automatic diagnosis of the 12-lead ecg using a deep neural network. Nature communications 11, 1760.

[47] Soleimani, M., Toosi, M.H., Mohammadi, S., Khalaj, B.H., 2025. Using test-time data augmentation for cross-domain atrial fibrillation detection from ecg signals. arXiv preprint arXiv:2503.13483 .

[48] Sun, Y., 2024. Shandong provincial hospital database (sphdb).

[49] Tsiartas, E., Nayak, D., Meade, A., 2025. Artificial intelligence capabilities in identifying atrial fibrillation using baseline sinus rhythm ecg: a systematic review. Open Heart 12.

[50] Tsutsui, K., Brimer, S.B., Ben-Moshe, N., Sellal, J.M., Oster, J., Mori, H., Ikeda, Y., Arai, T., Nakano, S., Kato, R., et al., 2025. Shdb-af: a japanese holter ecg database of atrial fibrillation. Scientific data 12, 454.

[51] Van Den Oord, A., Vinyals, O., et al., 2017. Neural discrete representation learning. Advances in neural information processing systems 30.

[52] Veličković, P., Cucurull, G., Casanova, A., Romero, A., Lio, P., Bengio, Y., 2017. Graph attention networks. arXiv preprint arXiv:1710.10903 .

[53] Wang, X., Ma, C., Zhang, X., Gao, H., Cliford, G.D., Liu, C., 2021. Paroxysmal atrial fibrillation events detection from dynamic ecg recordings: The 4th china physiological signal challenge 2021. Proc. PhysioNet , 1–83.

[54] Yan, T., Shi, H., Mu, L., Zheng, Z., Yao, L., Li, S., Chen, Z., Kang, X., Liang, S., Zhang, L., et al., 2026. Mms-net: A multi-task learning framework for 12-lead ecg reconstruction and disease classification from 3-lead inputs. Biomedical Signal Processing and Control 113, 108882.

[55] Yang, H.C., Hsieh, W.T., Chen, T.P.C., 2021. A mixed-domain self-attention network for multilabel cardiac irregularity classification using reduced-lead electrocardiogram, in: 2021 Computing in Cardiology (CinC), IEEE. pp. 01–04.

[56] Zhan, Z., Chen, J., Li, K., Huang, L., Xu, L., Bian, G.B., Millham, R., de Albuquerque, V.H.C., Wu, W., 2024. Conditional generative adversarial network driven variable-duration single-lead to 12-lead electrocardiogram reconstruction. Biomedical Signal Processing and Control 95, 106377.

[57] Zhang, X., Li, J., Cai, Z., Zhang, L., Chen, Z., Liu, C., 2021. Overfitting suppression training strategies for deep learning-based atrial fibrillation detection. Medical & Biological Engineering & Computing 59, 165–173.

[58] Zheng, J., Guo, H., Chu, H., 2022. A large scale 12-lead electrocardiogram database for arrhythmia study. PhysioNet Version 1.0.0.

[59] Zou, Y., Wang, P., Du, L., Chen, X., Li, Z., Song, J., Fang, Z., 2025. A multi-level multiple contrastive learning method for single-lead electrocardiogram atrial fibrillation detection. Bioengineering 12, 44.

[60] Zou, Y., Yu, X., Li, S., Mou, X., Du, L., Chen, X., Li, Z., Wang, P., Li, X., Du, M., et al., 2024. A generalizable and robust deep learning method for atrial fibrillation detection from long-term electrocardiogram. Biomedical Signal Processing and Control 90, 105797.

A. My Appendix

![](images/906f9e8b3f2cc0b35f9695b23537a8ec9f18646d53145d55ca30783c6318e03a.jpg)

![](images/84940df8ac91b7785609415e5461b88afae6723d7db61e7accda023db7ae489e.jpg)

(a)  
![](images/437133debdb972765a7e5b1170a4b0d03efd4febc35139820d7204309e1a8f1c.jpg)

![](images/831c32db271057fcb5e4617da9d9996455ebcad04b3f949765a528a1b400dc13.jpg)  
(b)  
Figure A.1: 12-lead ECG reconstruction results generated by our proposed DCGCNet. (a) Reconstruction using the combination of Leads I, II, and V1 as input. (b) Reconstruction using only Lead I as input. In both subplots, the ground-truth 12-lead ECG signals are plotted as solid black lines, while the DCGCNet-predicted reconstructions are shown as dashed red lines.

![](images/c62ecd53c01e7974fdfc3f70464630dbf6d784432f454c3b5f1fd478f803c62d.jpg)

![](images/f6575a9663f8fc2aa103c8dace57c462b1b1f088c0a1219c3ec8c79f7ce2d92f.jpg)

![](images/b374a58bd72fab0ec5b2377fcaaf3d718526c5709eb859038c0eb49c67f07eab.jpg)

![](images/cbca54528dc92593852f68e6343c09a0b6d759c6220f210f6f87a9dc9296b69f.jpg)

![](images/2b2b7a2e1c19a3bfa1bf47b97823bfed271fc659f7313e8611275ba87e4a7149.jpg)

![](images/86aa3ee7b66f3daedd9ed51febdc03d26bfcf174292096a62bc1464c583a2ffd.jpg)

![](images/292755b80010a9a8b12593057a63ebdbb2ab53892ff983f950405082b2cfc083.jpg)

![](images/d1dac886c7bff6875f5ee24f51f6f798a0e422d33ab03a05fcb7f594146eafae.jpg)

![](images/667463b5694f1ca8efb9345af3eaf3ecaf1da985911e000d69345f610b92216f.jpg)

![](images/b27dff36c961ea3884a5a7c2e9b6d14fe446fb3cfe64e3f1a1c4c0217c1c09b4.jpg)

![](images/489ca39cbf230c268c68fed526e080bbaeea056a5f04fdf0026ad9497efac219.jpg)

![](images/73dce167016f19a6fb497f2dc2510884386ea7e83df71e0f4b23ddda54e15bab.jpg)

![](images/9c30c71614f23dc047041ceaa62260f1672e31546b9dd67e23a6b806673a0331.jpg)

![](images/639db47dcbe7a6f46f691e6f56cf704a05a540a19063e5c712b17251e2892a63.jpg)

![](images/a04807eedc38c4079ac10dea8c126d39fbac7f3dd819c69ad7248903bb3c8f20.jpg)

![](images/c6ec6112908ebbc5f03834e7113f6546afaf3f58afc0639c1b5df85bf8da041b.jpg)

![](images/e3fd6c811c928081b2c87f763eb8a4191440ddb3bfa2e5adb6e2b6cfa83aa790.jpg)

![](images/82b4d9b67209bbb4f12cde3985a2265de6bf6a1e9453121ed3322c45d437cc6d.jpg)  
Figure A.2: Efect of clinically relevant additive noise on the original 12-lead ECG signal. Three common artifacts—baseline wander (caused by respiration or motion, simulated as low-frequency sinusoids <0.3 Hz), powerline interference (from mains electricity, modeled as a 50 Hz sine wave with random phase), and electromyographic (EMG) artifact (due to muscle activity, generated as bandpass-filtered white noise in 20–250 Hz)—are injected at intensity ratios of 0.1–0.5 relative to the ECG’s peak-to-peak amplitude, ensuring physiologically realistic corruption.

Comparison of AF detection performance with state-of-the-art methods across multiple datasets and cross-dataset settings. All results are directly reported from the original publications, ensuring a faithful reproduction of published claims. Metrics are presented as percentages (%). Abbreviations: CV = cross-validation; TL = transfer learning; CL = contrastive learning; DTW = dynamic time warping; AE = autoencoder; HN = hierarchical normalization. The proposed method $( ^ { \mathfrak { a } } \mathsf { O } \mathsf { u } \mathsf { r } ^ { \prime \prime } )$ demonstrates consistently competitive or superior performance, particularly in challenging cross-dataset generalization scenarios.
<table><tr><td>Train</td><td>Test</td><td>Input</td><td>AUC</td><td>ACC</td><td>F1</td><td>Precision</td><td>Recall</td><td>Method</td><td>Other</td></tr><tr><td>AFDB</td><td>AFDB</td><td>30s</td><td>99.50</td><td>97.10</td><td></td><td></td><td>97.00</td><td> $\mathsf { W a v e l e t } + \mathsf { S V M } [ 2 ]$ </td><td>2-fold CV</td></tr><tr><td>AFDB</td><td>AFDB</td><td>30 beats</td><td></td><td>97.60</td><td>97.10</td><td></td><td>98.00</td><td> $\mathsf { P e a k } + \mathsf { R F } [ 1 7 ]$ </td><td>4-fold CV</td></tr><tr><td>AFDB</td><td>AFDB</td><td>1.2s</td><td></td><td>99.23</td><td></td><td></td><td>99.41</td><td> $\mathsf { W a v e l e t } + \mathsf { C N N } [ 1 5 ]$ </td><td>4:1 split</td></tr><tr><td>AFDB</td><td>AFDB</td><td>30s</td><td></td><td>97.80</td><td></td><td></td><td>98.98</td><td> $\mathsf { C N N } + \mathsf { L S T M } [ 1 ]$ </td><td>5-fold CV</td></tr><tr><td>AFDB</td><td>AFDB</td><td>30s</td><td></td><td>93.00</td><td>92.64</td><td></td><td>89.83</td><td>Siamese + Few-shot TL[39]</td><td>5-fold CV</td></tr><tr><td>AFDB</td><td>AFDB</td><td>15s</td><td></td><td>99.11</td><td>97.00</td><td>99.46</td><td>95.40</td><td>Spatio-temporal Siamese[28]</td><td>8:1:1 split</td></tr><tr><td>AFDB</td><td>AFDB</td><td>5s</td><td></td><td>99.60</td><td></td><td>一</td><td>99.43</td><td>Transformer[31]</td><td> $9 5 \% / 5 \% / 5 \%$ </td></tr><tr><td>Wearable I</td><td>AFDB</td><td>10s</td><td></td><td>95.28</td><td>一</td><td></td><td>96.46</td><td> $\mathsf { L S T M } + \mathsf { C N N } [ \mathsf { 5 7 } ]$ </td><td>10-fold CV</td></tr><tr><td>CPSC2021</td><td>AFDB</td><td>30s</td><td>99.53</td><td>98.63</td><td>98.28</td><td>99.23</td><td>97.35</td><td> $\mathsf { R e s C N N + B i L S T M } [ 6 0 ]$ </td><td>5-fold CV</td></tr><tr><td>CPSC2021</td><td>AFDB</td><td>30s</td><td></td><td>98.18</td><td>98.10</td><td>98.33</td><td>97.90</td><td>Semi-supervised CL[59]</td><td>Cross-dataset</td></tr><tr><td>Chapman</td><td>AFDB</td><td>10s</td><td>99.51</td><td>97.55</td><td>97.40</td><td>97.30</td><td>97.49</td><td>Ours</td><td>5-fold CV</td></tr><tr><td>CPSC2021</td><td>CPSC2021</td><td>12 beats</td><td></td><td>95.19</td><td></td><td></td><td>96.78</td><td> $\mathsf { D T W } + \mathsf { A E } + \mathsf { S V M } [ 3 3 ]$ </td><td>5-fold CV</td></tr><tr><td>CPSC2021</td><td>CPSC2021</td><td>30s</td><td>99.70</td><td>98.63</td><td>98.18</td><td>97.34</td><td>99.08</td><td> $\mathsf { R e s C N N + B i L S T M } [ 6 0 ]$ </td><td>5-fold CV</td></tr><tr><td>CPSC2021</td><td>CPSC2021</td><td>5s</td><td></td><td>98.40</td><td></td><td></td><td>98.42</td><td> $\mathsf { T r a n s f o r m e r } [ 3 1 ]$ </td><td>95%/5%/5%</td></tr><tr><td>UVAF</td><td>CPSC2021</td><td>60 beats</td><td>99.00</td><td></td><td>95.00</td><td></td><td>95.00</td><td> $\mathsf { R e s N e t } + \mathsf { G R U } [ 5 ]$ </td><td>5-fold CV</td></tr><tr><td>Chapman</td><td>CPSC2021</td><td>10s</td><td>98.96</td><td>96.28</td><td>95.96</td><td>95.37</td><td>96.68</td><td>Ours</td><td>5-fold CV</td></tr><tr><td>AFDB</td><td>LTAF</td><td>30s</td><td></td><td>97.09</td><td>97.07</td><td></td><td>97.37</td><td>Siamese + Few-shot TL[39]</td><td>5-fold CV</td></tr><tr><td>CPSC2021</td><td>LTAF</td><td>30s</td><td>98.29</td><td>97.07</td><td>97.23</td><td>97.66</td><td>96.81</td><td>ResCNN + BiLSTM[60]</td><td>5-fold CV</td></tr><tr><td>CPSC2021</td><td>LTAF</td><td>30s</td><td></td><td>96.61</td><td>96.60</td><td>96.56</td><td>96.64</td><td>Semi-supervised CL[59]</td><td>Cross-dataset</td></tr><tr><td>Chapman</td><td>LTAF</td><td>10s</td><td>99.04</td><td>96.44</td><td>96.42</td><td>96.40</td><td>96.45</td><td>Ours</td><td>5-fold CV</td></tr><tr><td>UVAF</td><td>SHDBAF</td><td>60 beats</td><td>99.00</td><td></td><td>92.00</td><td></td><td>93.00</td><td>ResNet + GRU[5]</td><td>5-fold CV</td></tr><tr><td>Chapman</td><td>SHDBAF</td><td>10s</td><td>98.78</td><td>96.30</td><td>94.17</td><td>93.75</td><td>94.63</td><td>Ours</td><td>5-fold CV</td></tr><tr><td>Chapman</td><td>Chapman</td><td>10s</td><td>99.80</td><td>98.40</td><td>97.70</td><td></td><td></td><td>Supervised CL + HN[21]</td><td>8:1:1 split</td></tr><tr><td>Chapman</td><td>Chapman</td><td>10s</td><td>99.98</td><td>99.66</td><td>99.43</td><td>99.24</td><td>99.62</td><td>Ours</td><td>5-fold CV</td></tr><tr><td>CPSC2018</td><td>Chapman</td><td>10s</td><td>98.74</td><td>94.45</td><td>91.69</td><td>88.44</td><td>96.58</td><td>Ours</td><td>5-fold CV</td></tr><tr><td>CPSC2018</td><td>CPSC2018</td><td>10s</td><td>99.50</td><td>94.40</td><td>94.40</td><td></td><td></td><td>Supervised CL + HN[21]</td><td>8:1:1 split</td></tr><tr><td>CPSC2018</td><td>CPSC2018</td><td>10s</td><td>99.91</td><td>99.37</td><td>99.36</td><td>99.32</td><td>99.41</td><td>Ours</td><td>5-fold CV</td></tr><tr><td>Chapman</td><td>CPSC2018</td><td>10s</td><td>99.67</td><td>96.29</td><td>96.27</td><td>96.14</td><td>96.67</td><td>Ours</td><td>5-fold CV</td></tr></table>

Sensitivity analysis of model performance to auxiliary loss weights, with $\lambda _ { \mathrm { c l s } } = 1$ and $\lambda _ { \mathrm { r e c } } = 1$ fixed. Reported metrics are AUC and accuracy (ACC) on Chapman and CPSC2018 datasets.
<table><tr><td>Setting</td><td> $\lambda _ { \mathsf { v q } }$ </td><td> $\lambda _ { \tt c o n g }$ </td><td> $\lambda _ { \mathsf { c o n l } }$ </td><td>Chapman AUC</td><td>Chapman ACC</td><td>CPSC2018 AUC</td><td>CPSC2018 ACC</td></tr><tr><td>Default</td><td>0.2</td><td>0.5</td><td>0.1</td><td>99.98</td><td>99.66</td><td>99.91</td><td>99.37</td></tr><tr><td>Weaker global</td><td>0.2</td><td>0.1</td><td>0.1</td><td>99.74</td><td>98.83</td><td>99.48</td><td>98.26</td></tr><tr><td>Stronger global</td><td>0.2</td><td>1.0</td><td>0.1</td><td>99.80</td><td>99.03</td><td>99.62</td><td>98.63</td></tr><tr><td>Excessive global</td><td>0.2</td><td>3.0</td><td>0.1</td><td>99.22</td><td>97.59</td><td>98.51</td><td>96.54</td></tr><tr><td>Weaker local</td><td>0.2</td><td>0.5</td><td>0.02</td><td>99.66</td><td>98.54</td><td>99.35</td><td>98.07</td></tr><tr><td>Stronger local</td><td>0.2</td><td>0.5</td><td>0.20</td><td>99.70</td><td>98.75</td><td>99.54</td><td>98.42</td></tr><tr><td>Excessive local</td><td>0.2</td><td>0.5</td><td>0.50</td><td>99.11</td><td>97.20</td><td>98.34</td><td>96.29</td></tr><tr><td>Weaker VQ</td><td>0.05</td><td>0.5</td><td>0.1</td><td>99.51</td><td>98.24</td><td>99.08</td><td>97.53</td></tr><tr><td>Stronger VQ</td><td>0.50</td><td>0.5</td><td>0.1</td><td>99.34</td><td>97.83</td><td>98.76</td><td>97.05</td></tr><tr><td>Excessive VQ</td><td>1.00</td><td>0.5</td><td>0.1</td><td>98.59</td><td>96.20</td><td>97.67</td><td>95.28</td></tr></table>

![](images/5708d32b07da637b7f6695ba58aa79bde415b98b903791887fce8dcf388a825a.jpg)

![](images/8443fe3ff1c676cf31e5b1f14a9c732613eb3a8be2614fbb18e341c17519fe50.jpg)  
(b) Morphology  
Figure A.3: Grad-CAM visualizations validate the functional specialization of our dual-encoder architecture. Color intensity reflects gradient-based importance: warm colors (red/yellow) indicate regions most influential to each encoder’s output, while cool colors (blue) denote low attention. (a) The rhythm encoder—using small kernels—consistently attends to the R-wave, its interval regularity directly underpins rhythm assessment and atrial fibrillation detection. (b) The morphology encoder—employing larger kernels—focuses on the P-wave region, which carries structural information about atrial depolarization: organized P-waves signify sinus rhythm, whereas their absence or disorganization is characteristic of atrial fibrillation. This clear dissociation confirms our design rationale: distinct receptive fields efectively decouple temporal dynamics (rhythm) from waveform structure (morphology).