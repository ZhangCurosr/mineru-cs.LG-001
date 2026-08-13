# Beyond Local Power: Functional Connectivity Analysis for Subject-Independent Learning Style Recognition

Wiga Maulana Baihaqi

Dept. of Electrical and

Information Engineering, FacultyInformation Engineering, Faculty ormation Engineering, Faculty

Indriana Hidayah   
Dept. of Electrical and   
of Engineering   
Universitas Gadjah Mada   
Yogyakarta, Indonesia   
indriana.h@ugm.ac.id   
Noor Akhmad Setiawan   
Dept. of Electrical and   
Information Engineering, Faculty   
of Engineering   
Universitas Gadjah Mada   
Yogyakarta, Indonesia   
noorwewe@ugm.ac.id   
Sri Kusrohmaniah   
Department of Psychology,   
Faculty of Psychology   
Universitas Gadjah Mada   
Yogyakarta, Indonesia   
koes psi@ugm.ac.id

Abstract—Identifying individual learning styles optimizes pedagogical efficacy. While traditional questionnaires are structured, behavioral tracking methods require prolonged interaction log accumulation. To overcome these temporal constraints, this paper proposes an objective Electroencephalography (EEG) approach evaluating Phase Locking Value (PLV) connectivity against localized features across the Active–Reflective (AR) and Verbal–Visual (VV) Felder-Silverman dimensions. EEG signals were recorded from 28 participants during Raven’s Advanced Progressive Matrices tasks. Support Vector Machine classification used Leave-One-Subject-Out Cross-Validation (LOSO-CV) alongside a 70:30 intra-subject split. The VV dimension achieved 70.00% subject-<sup>[</sup> level accuracy driven by distinct fronto-occipital polarization. Conversely, the AR dimension yielded lower cross-subject generalizability (55.56%) due to overlapping executive networks and a “Systematic Neural Inversion” phenomenon, where stable individual connectivity signatures operated diametrically opposed to global boundaries (up to 20–0 voting margins). Ultimately, these outcomes demonstrate that rigid “one-size-fits-all” classifiers are bounded by biological diversity, emphasizing the need for future adaptive feature transformation techniques to bridge the crosssubject generalization gap.

Index Terms—EEG, FSLSM, learning styles, phase locking value, functional connectivity, SVM, brain-computer interface

## I. INTRODUCTION

Identifying individual learning styles is essential for optimizing overall pedagogical efficacy and enhancing the broader student learning experience [1]. While traditional self-report questionnaires like the Felder-Silverman Index of Learning Styles (ILS) offer a structured [2], widely adopted approach for identifying learning tendencies [3], alternative data-driven methods have emerged to infer learning styles from behavioral patterns or log data, aiming to eliminate the need for lengthy self-reports. However, these behavioral classifications are inherently time-consuming, often requiring multiple sessions across an entire course to accumulate sufficient interaction logs [4]. To overcome these temporal constraints, Electroencephalography (EEG) provides a rapid, direct alternative by recording cortical activity with excellent temporal resolution [5], allowing the immediate capture of short-term neurophysiological dynamics during specific tasks. Recent advancements have shifted from simple spectral clustering to advanced machine and deep learning architectures [6] utilizing time-domain, frequency-domain, and non-linear connectivity features [7], [8].

Despite these advances, existing deep learning models heavily rely on localized power features that are sensitive to anatomical differences between subjects, leading to poor generalization across different populations [9]. Furthermore, localized approaches treat brain regions in isolation, failing to account for the synchronized, large-scale networks that underpin complex learning activities. Neuroscientific evidence suggests that cognitive functions—such as information processing (Active–Reflective) or mental imagery (Verbal–Visual)— emerge from the cooperation of distributed neural networks [10]. Recent efforts have also explored adaptive waveletbased decomposition and multi-instance learning strategies to further improve EEG-based learning style recognition [11]. Specifically, phase synchronization between fronto-parietal and fronto-occipital circuits facilitates cognitive state switching. Phase Locking Value (PLV) effectively measures this synchrony independent of signal amplitude, mitigating individual physiological variations and providing neurophysiologically well-founded spatial priors that enhance classification accuracy and model interpretability [9].

This paper evaluates PLV against traditional localized features (PSD, entropy, and statistical measures) for FSLSM classification. Using Support Vector Machines (SVM), we classify the Active–Reflective and Verbal–Visual dimensions. To ensure robust generalization, we employ a dual-validation strategy: a standard 70:30 train-test split and a rigorous Leave-One-Subject-Out Cross-Validation (LOSO-CV) to assess subject-independent performance.

The contributions of this paper are threefold. First, it provides a complete feature evaluation across the frequency, time, complexity, and connection domains to discover highly accurate EEG biomarkers for the Felder-Silverman Learning Style Model (FSLSM). Second, to address the underlying neuronal heterogeneity in EEG data, this study establishes a subject-independent generalization framework applying a rigorous Leave-One-Subject-Out Cross-Validation (LOSO-CV) methodology. Finally, we conduct a full neurophysiological characterisation by directly mapping the FSLSM dimensions to phase-locking brain networks to understand the neurophysiological foundation underpinning individual learning styles.

The remainder of this paper is organized as follows: Section II details the methodology; Section III reports the results; Section IV discusses the findings; and Section V concludes the paper.

## II. METHOD

The step-wise scheme of the suggested research methodology for identifying the FSLSM facets using EEG data is illustrated in Fig. 1. The research has been divided into five parts: (1) recording and labeling EEG signals while participants were engaged in mental tasks, (2) preprocessing signals, (3) extracting features from different domains, (4) classifying signals using machine learning, and (5) evaluating the results and performing neurophysiological analysis. Through this extensive methodology, we intend to transmute the raw brainwave data into steady cognitive profiles while at the same time addressing the brain signal individual variability issue. Every step in the process, together with the specific computational and validation methods, is described in the following subsections.

## A. Participants

Twenty-eight healthy college students (aged 18–20) with no history of neurological or psychiatric disorders participated in this study. To ensure robust labeling and minimize “neural inversion” artifacts [6], only participants with a score of ≥ 5 (moderate-to-strong preference) on the Index of Learning Styles (ILS) survey were included; balanced learners (scores 1–3) were excluded. Following a balanced-class approach [6], we selected 18 participants for the Active–Reflective (AR) dimension (9 Active, 9 Reflective) and 10 for the Verbal– Visual (VV) dimension (5 Verbal, 5 Visual). This 50:50 distribution establishes a strict theoretical chance-level accuracy of 50.00%, eliminating majority class bias. The study was approved by the Research Ethics Committee of Universitas Indonesia Maju (No: 2649/Sket/Ka-Dept/RE/UIMA/VI/2025), and all participants provided written informed consent.

## B. EEG Data Acquisition

Continuous EEG signals were recorded at a 250 Hz sam pling rate using an OpenBCI Cyton board from eight electrodes (Fp1, Fp2, F3, F4, C3, C4, O1, and O2) with a left earlobe (A1) reference [12]. Cognitive stimulation was induced via 20 independent, 15-second trials of Raven’s Advanced Progressive Matrices (RAPM) Set II (Fig. 2). RAPM serves as a language-free, high-load cognitive catalyst that forces style specific processing strategies under the FSLSM framework. Specifically, the logical symbol rules within RAPM effectively elicit divergent processing behaviors: reflective learners naturally engage in deliberative introspection and meticulous hypothesis evaluation, whereas active learners instinctively deploy rapid, intuitive mental trial-and-error operations [6]. Simultaneously, solving these non-verbal matrices forces distinct input decoding strategies, where visual learners instinctively rely on spatial imagery and pattern manipulation, while verbal learners formulate internal sequential logical rules to decode the matrix structures [13]. Furthermore, the moderate difficulty level of RAPM prevents cognitive fatigue and excessive workload, thereby ensuring high signal integrity while capturing these distinct functional connectivity topologies. Fig. 2 illustrates the experimental setup, including the participant cap, Cyton board, and real-time monitoring via the OpenBCI GUI. The total recording time was approximately five minutes per participant.

## C. Preprocessing

To enhance signal quality while preserving phase information, raw EEG data were band-pass filtered between 8– 30 Hz using a fourth-order, zero-phase Butterworth filter [14]. The 15-second trials were segmented into 1-second nonoverlapping windows. To ensure reliable functional connectivity estimation, features were calculated at the whole-trial level by combining these windowed segments. In line with previous protocols, no additional artifact rejection or re-referencing was performed to maintain methodological consistency.

![](images/eb95f052857a20be65598d28ef740eabe68f19ca9ebfe63071407e2019855c87.jpg)  
Fig. 1. Proposed method: step-wise EEG-based FSLSM recognition pipeline.

![](images/c3aa0acd6cd76a56ea2da58207ad85ec3d7f9f5987ad675fcd342445869555e1.jpg)  
Fig. 2. Experimental setup for EEG data acquisition.

## D. Feature Extraction

Four feature sets were extracted across the time, frequency, complexity, and connectivity domains:

• Power Spectral Density (PSD) via Welch’s method for Alpha (8–13 Hz) and Beta (13–30 Hz) bands across 8 channels (16 features).

• Five statistical measures (mean, standard deviation, skewness, kurtosis, and RMS) per channel (40 features).

• Sample Entropy (SE) per channel to index cognitive load (8 features).

• Phase Locking Value (PLV) to quantify phase synchronization consistency between channel pairs.

The PLV of two channels i and $j$ is given by (1) [15]:

$$
P L V _ { i , j } = \left| \frac { 1 } { N } \sum _ { n = 1 } ^ { N } e ^ { j \left( \phi _ { i } \left( n \right) - \phi _ { j } \left( n \right) \right) } \right|\tag{1}
$$

where $\phi ( n )$ is the instantaneous phase acquired by the Hilbert Transform and N stands for the number of samples. For the 8- channel system, 28 unique pairs were computed per window, representing the global synchronization of the brain network.

## E. Classification and Evaluation Framework

To address inter-subject heterogeneity, individual Z-score normalization was applied to each participant’s data. Classification was performed using a Support Vector Machine (SVM) with a Radial Basis Function (RBF) kernel [16]. The SVM decision function for an input x is defined as (2):

$$
f ( \mathbf { x } ) = \mathrm { s i g n } \left( \sum _ { k = 1 } ^ { n _ { s v } } \alpha _ { k } y _ { k } K ( \mathbf { x } _ { k } , \mathbf { x } ) + b \right)\tag{2}
$$

here, $n _ { s v }$ is the number of support vectors, $\alpha _ { k }$ are the Lagrange multipliers, $y _ { k }$ are the class labels, b is the bias term, and $K ( \mathbf { x } _ { k } , \mathbf { x } )$ is the RBF kernel.

Subject-level predictions were determined using a hierarchical two-stage majority voting system across windows and trials. Model generalization was evaluated using two distinct validation schemes: an epoch-wise 70:30 split and Leave-One-Subject-Out Cross-Validation (LOSO-CV). The epoch-wise 70:30 split was conducted as an internal validation baseline to confirm model learnability and intra-subject feature consistency. Meanwhile, LOSO-CV was implemented to rigorously evaluate subject-independent performance while completely eliminating data leakage risks; notably, all core findings and subject-independent claims in this study rely exclusively on the LOSO-CV scheme.

Performance across both schemes was quantified using Accuracy, Precision, Recall, and F1-Score for each FSLSM dimension.

## F. Neurophysiological Connectivity Analysis

Functional connectivity was estimated by calculating the Phase Locking Value (PLV) between pairs of electrodes among a set of eight electrodes located at frontal (Fp1, Fp2, F3, F4), central (C3, C4), and occipital (O1, O2) sites. Class-specific connectivity patterns were obtained by retaining only the top 25% strongest synchronizations for each class. To detect features that best discriminate between classes, a difference connectivity matrix (∆PLV) was derived through a double threshold technique [17]. Edges were displayed only when they satisfied two requirements: (1) statistical significance by means of an independent t-test $( p \ : < \ : 0 . 0 5 )$ , and (2) a size among the top 20% of absolute differences (∆PLV > 80th percentile). Differential edges were assigned color codes (red for positive, blue for negative differences) and thicknesses proportional to their ∆PLV values. With this, the resulting networks represent statistically supported and neurophysiologically significant features of FSLSM.

TABLE I  
SUBJECT-LEVEL CLASSIFICATION PERFORMANCE: LOSO VALIDATION
<table><tr><td>Dim.</td><td>Feature Domain</td><td>Acc</td><td>Prec</td><td>Rec</td><td>F1</td></tr><tr><td rowspan="3">AR (Act-Ref)</td><td>PLV</td><td>55.56%</td><td>55.84%</td><td>55.56%</td><td>55.00%</td></tr><tr><td>Entropy</td><td>55.56%</td><td>56.92%</td><td>55.56%</td><td>53.25%</td></tr><tr><td>PSD (Alpha/Beta) Statistics</td><td>61.11% 55.56%</td><td>61.25% 56.92%</td><td>61.11% 55.56%</td><td>60.99% 53.25%</td></tr><tr><td rowspan="5">VV (Ver-Vis)</td><td>PLV</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>70.00%</td><td>70.83%</td><td>70.00%</td><td>69.70%</td></tr><tr><td>PSD Statistics</td><td>40.00%</td><td>40.00%</td><td>40.00%</td><td>40.00%</td></tr><tr><td></td><td>40.00%</td><td>40.00%</td><td>40.00%</td><td>40.00%</td></tr><tr><td>Entropy</td><td>40.00%</td><td>40.00%</td><td>40.00%</td><td>40.00%</td></tr></table>

## III. EXPERIMENTAL RESULTS

## A. Validation of the Feature Extraction Pipeline

The signal transformation pipeline is illustrated in Fig. 3. Panel (a) shows the raw EEG data from representative prefrontal (Fp1), central (C3), and occipital (O1) channels, which exhibit substantial DC offset and low-frequency baseline drifts. Panel (b) confirms the efficacy of the 8–30 Hz Butterworth band-pass filter, which transforms the raw data into stable, zero-mean signals within normal neurophysiological ranges by isolating Alpha and Beta oscillations. Finally, Panel (c) displays the derived Phase Locking Value (PLV) connectivity matrix for a Verbal learner. This heatmap captures distinct, noise-free spatial synchronization topologies that serve as the structural features for FSLSM classification.

## B. Comparative Performance of EEG Feature Domains

We evaluated classification performance across four domains: Phase Locking Value (PLV), Power Spectral Density (PSD), Sample Entropy, and Time-domain Statistics. Table I presents the Leave-One-Subject-Out (LOSO) cross-validation results, testing the model’s ability to generalize to unseen subjects.

As shown in Table I, PSD (Alpha/Beta) features yielded the highest accuracy for the Active–Reflective (AR) dimension (61.11%), indicating that AR styles are better captured by rhythmic oscillations and spectral power distributions. Conversely, PLV achieved the highest accuracy for the Verbal– Visual (VV) dimension (70.00%). This demonstrates that VV processing is predominantly a large-scale cortical network phenomenon relying on inter-regional synchronization (e.g., occipital-temporal integration) rather than localized power alterations.

TABLE II  
SUBJECT-LEVEL PERFORMANCE: 70:30 TRAIN-TEST SPLIT
<table><tr><td>Dimension</td><td>Feature Domain</td><td>Acc</td><td>Prec</td><td>Rec</td><td>F1</td></tr><tr><td>AR (Active-Refl.)</td><td>PLV PSD</td><td>100% 94.44%</td><td>100% 95.00%</td><td>100% 94.44%</td><td>100% 94.43%</td></tr><tr><td>VV (Verbal–Vis.)</td><td>PLV Statistics</td><td>100% 90.00%</td><td>100% 91.67%</td><td>100% 90.00%</td><td>100% 89.90%</td></tr></table>

TABLE III

STABILITY METRICS: MEAN ACC ± SD (LOSO)
<table><tr><td>Dim.</td><td>Feat.</td><td>Window Acc</td><td>Trial Acc</td><td>Subj. Acc</td></tr><tr><td>AR</td><td>PSD</td><td> $5 5 . 8 9 \% \pm 2 5 . 1 1 \%$ </td><td> $5 6 . 1 1 \% \pm 3 4 . 6 2 \%$ </td><td>61.11%</td></tr><tr><td>VV</td><td>PLV</td><td> $5 5 . 4 7 \% \pm 3 1 . 0 3 \%$ </td><td> $5 6 . 0 0 \% \pm 3 4 . 9 1 \%$ </td><td>70.00%</td></tr></table>

## C. Analysis of the Generalization Gap (LOSO vs. 70:30 Split)

A substantial “generalization gap” was observed when comparing the subject-independent LOSO scheme against the intra-subject 70:30 train-test split baseline (Table II).

In the 70:30 split, PLV achieved perfect accuracy (100.00%) for both dimensions. However, the sharp performance decline in LOSO validation highlights pronounced inter-subject variability. While the SVM model effectively learns individual neural “fingerprints” from known subjects, cross-subject generalization remains highly challenging due to the idiosyncratic nature of EEG signals.

## D. Stability and Majority Voting Efficiency

The hierarchical majority voting mechanism at the trial and subject levels effectively filtered out transient, noisy EEG segments to stabilize classification.

Table III demonstrates that aggregating window-level predictions into trial- and subject-level decisions consistently enhances classification stability. For instance, the AR subjectlevel accuracy (61.11%) outperforms its window-level baseline (55.89%). This indicates that while brief EEG epochs may be ambiguous, the cumulative majority vote over the entire session provides a more resilient reflection of a learner’s cognitive style.

## E. Neurophysiological Connectivity Analysis

Functional synchronization networks were visualized using PLV mapping to investigate the biological underpinnings of classification performance.

1) Active–Reflective (AR) Connectivity Profiles: Connectivity patterns (Fig. 4) reveal distinct synchronization densities. The Active style exhibits a streamlined network localized along the Frontal–Occipital axis (Fp2-O2, F4-O2). Conversely, the Reflective style displays a denser, more complex global integration across Frontal–Central–Occipital regions. The differential map (Fig. 4c) is dominated by blue edges (Reflective > Active), showing that the Reflective style recruits significantly higher global synchronization driven by intensive mental simulation and executive control [17]. Key discriminative features include dense inter-hemispheric coupling (Fp1-Fp2, C3-C4)

![](images/87aecf1c16e37026933475dc14cfabbbe9696ba3caff6da09e1a8014653533cf.jpg)

![](images/e0bd7251cea11ee4ea787c57868bcd34b6b5a0280c0a9b668eb53c78f984b076.jpg)  
Fig. 3. EEG data processing pipeline showing (a) raw signals, (b) 8–30 Hz filtered signals, and (c) the derived PLV matrix.

and robust long-range prefrontal-to-occipital connectivity. This topological overlap and shared executive circuitry explain why the classifier struggles to distinguish baseline synchronization profiles across unseen subjects during LOSO validation.

2) Verbal–Visual (VV) Connectivity Profiles: Neural signatures (Fig. 5) demonstrate clear spatial polarization. The Verbal style forms a dense hub centered in Left-Frontal and Prefrontal regions (Fp1, F3) associated with linguistic circuits and verbal rehearsal [17], [18]. Conversely, the Visual style activates localized Right-Hemisphere and posterior occipital pathways (Fp2-O2, F4-O2) tailored for spatial information processing [18]. Significant differences (Fig. 5c) confirm an overwhelming dominance of red edges (Verbal > Visual). This pronounced, distinct spatial divergence (Anterior vs. Posterior) directly accounts for the superior subject-level classification accuracy (70.00%) in the VV dimension compared to the overlapping functional networks of AR.

## IV. DISCUSSION

The Leave-One-Subject-Out (LOSO) evaluation reveals a prominent “Systematic Bias” rather than a stochastic tracking failure across subject-independent distributions. As demonstrated in the individual voting analysis (Table IV), several subjects achieved a 0% classification accuracy despite maintaining highly decisive voting margins (e.g., 20–0). Within the Active–Reflective (AR) dimension, multiple Reflective participants (such as Subjects 11, 12, 13, and 18) were classified as Active with 100% certainty. This empirical outcome uncovers a distinct “Systematic Neural Inversion” effect, indicating that while functional connectivity signatures remain highly stable within individuals, their biological manifestation can be mathematically oriented in a direction diametrically opposed to the population-averaged global model boundary. Similarly, the Verbal–Visual (VV) dimension exhibits a clear binary “Correct or Inverted” split (e.g., Subject 8), confirming that while Phase Locking Value (PLV) acts as a highly structured biomarker, its interpretation remains strictly person-specific.

These pronounced 20–0 voting errors provide critical empirical insights into the explicit limitations of utilizing a rigid, “one-size-fits-all” global classifier on volatile biological data. A generalized framework inherently presumes a homogeneous distribution of feature representations across an entire population; however, human neurophysiology sharply violates this assumption due to high inter-subject heterogeneity [19]. Because the absolute consistency of the misclassifications is not driven by random sensor noise, it confirms that while a populationaveraged model can successfully map prominent spatially polarized networks—such as the anterior-posterior divergence during Verbal–Visual processing—its generalizability remains strictly bounded by individual biological diversity. This limitation is heavily reflected in the substantial “generalization gap” where perfect intra-subject learnability (100.00% in the 70:30 split) drops significantly under cross-subject validation, particularly in the AR dimension, where the cross-subject PLV accuracy is limited to 55.56% due to topological overlaps in shared executive circuits [17], [18].

From a methodological perspective, choosing a binary classification paradigm in this preliminary study serves as a critical first step to mathematically isolate these raw discriminative boundaries without over-parameterizing the model.

![](images/e26772118e792ba7fa0736036e9acaf8ad042080fddaa0ddfeb63abb4629ff7d.jpg)

![](images/83c48b27461b7682ea4fd1c70d42f90cdf1b5a0fe15a2dceffc89789857c1f72.jpg)  
Fig. 5. Connectivity analysis for the VV dimension: (a) grand average Verbal, (b) grand average Visual, and (c) significant differences.

TABLE IV  
INDIVIDUAL VOTING ANALYSIS (SELECTED SUBJECTS)
<table><tr><td>Dim.</td><td>Subj.</td><td>True Label</td><td>Acc.</td><td>V1</td><td>V2</td></tr><tr><td>AR</td><td>3</td><td>Active</td><td>100%</td><td>20</td><td>0</td></tr><tr><td>AR</td><td>11</td><td>Reflective</td><td>0%</td><td>20</td><td>0</td></tr><tr><td>VV</td><td>1</td><td>Verbal</td><td>100%</td><td>20</td><td>0</td></tr><tr><td>VV</td><td>8</td><td>Visual</td><td>0%</td><td>20</td><td>0</td></tr></table>

Attempting to map high-dimensional phase-synchronization features to a continuous psychometric scale or a multi-class structure requires a substantially larger cohort to prevent severe overfitting. Given the current sample size (N = 28), the binary paradigm effectively restricts the parameter space of the SVM classifier, ensuring that the documented neural inversions are genuine representations of neurophysiological variance rather than statistical artifacts of overfitting. Furthermore, these findings underscore that standard subject-wise Z-score normalization remains insufficient to fully eliminate individual neural fingerprints [20]. Consequently, subsequent stages of this research framework will focus on structural methodological advancements specifically geared toward mitigating the cross-subject generalization gap. Future work will explore more robust distribution alignment strategies and adaptive feature transformation techniques to better harmonize the heterogeneous neural patterns across different individuals before expanding the scope to systematically incorporate the remaining Sensing–Intuitive and Sequential–Global dimensions of the FSLSM.

## V. CONCLUSION

This paper demonstrates that PLV serves as an effective quantitative biomarker for FSLSM recognition. Under LOSO-CV, the VV dimension achieved 70.00% accuracy due to distinct fronto-occipital polarization, whereas the AR dimension dropped to 55.56% due to topologically overlapping executive networks. A major contribution of this foundational study is the empirical evaluation of a rigid “one-size-fits-all” model, highlighted by the identification of the “Systematic Neural Inversion” phenomenon. The presence of high-confidence yet completely inverted predictions (20–0 voting margins) reveals that stable, individual functional connectivity signatures can operate in direct opposition to population-averaged boundaries. Ultimately, these outcomes establish a critical neurophysiological baseline, demonstrating that global models remain strictly bounded by individual biological diversity. These findings strongly motivate future methodological shifts toward adaptive feature transformation techniques specifically designed to mitigate the cross-subject generalization gap.

## ACKNOWLEDGMENT

This study was funded by the Indonesia Endowment Fund for Education (LPDP) under grant number SKPB-4803/LPDP/LPDP.3/2025.

## REFERENCES

[1] S.-R. Liou, C.-Y. Cheng, T.-P. Chu, C.-H. Chang, and H.-C. Liu, “Effectiveness of differentiated instruction on learning outcomes and learning satisfaction in the evidence-based nursing course: Empirical research quantitative,” Nurs. Open, vol. 10, no. 10, pp. 6794–6807, Oct. 2023.

[2] B. Hmedna, A. Bakki, A. E. Mezouary, and O. Baz, “Unlocking teachers’ potential: Moocls, a visualization tool for enhancing mooc teaching,” Smart Learning Environments, vol. 10, no. 1, p. 58, 2023.

[3] S. Shrestha, M. Joshi, A. Bashyal, A. Timilsina, and S. Subedi, “Integration of gamified elements and learning style data in online learning system,” 2023.

[4] N. Alzahrani, M. S. Ramzan, M. Meccawy, and H. Samra, “Learner behavior modeling framework through socio-cognitive features in elearning environment,” in 2025 International Conference on Artificial Intelligence, Computer, Data Sciences and Applications (ACDSA), 2025, pp. 1–6.

[5] A. Wijaya, N. A. Setiawan, and M. I. Shapiai, “Mapping research themes and future directions in learning style detection research: A bibliometric and content analysis,” The Electronic Journal of e-Learning, vol. 21, no. 4, p. 274, 2023.

[6] B. Zhang, C. Chai, Z. Yin, and Y. Shi, “Design and implementation of an eeg-based learning-style recognition mechanism,” Brain Sci., vol. 11, no. 5, 2021.

[7] R. Yuvaraj et al., “A machine learning framework for classroom eeg recording classification: Unveiling learning-style patterns,” Algorithms, vol. 17, no. 11, 2024.

[8] R. K. Saidala, S. Marri, U. Radder, R. M. Chethana, R. R. Bojja, and B. P. Shankar, “Real-time neural network system for identifying visual learners from raw eeg data,” in 3rd International Conference on Advances in Computing, Communication and Materials (ICACCM 2024), 2024.

[9] F. Abuhashish, R. Alrousan, H. A. Alkhazaleh, A. W. Arram, and I. Ismail, “Advanced eeg emotion recognition framework integrating fractal dimensions, connectivity metrics, and domain adaptive deep learning,” International Journal of Innovative Research and Scientific Studies, vol. 8, no. 6, pp. 1247–1265, Sep. 2025.

[10] R. Hall, M. Jackson, M. Maleki, and H. T. Crogman, “Modeling cognition through adaptive neural synchronization: a multimodal framework using eeg, fmri, and reinforcement learning,” Front. Comput. Neurosci., vol. 19, p. 1616472, 2025.

[11] A. Wijaya, M. S. Hasibuan, W. M. Baihaqi, R. Darmawan, R. Primartha, and N. A. Setiawan, “Improving eeg-based learning style detection with adaptive dwt and multi-instance learning,” International Journal of Intelligent Engineering and Systems, vol. 19, no. 1, pp. 389–401, Jan. 2026.

[12] V. Prabhakaran, J. A. Smith, J. E. Desmond, G. H. Glover, and J. D. E. Gabrieli, “Neural substrates of fluid reasoning: an fmri study of neocortical activation during performance of the raven’s progressive matrices test,” Cogn. Psychol., vol. 33, no. 1, pp. 43–63, 1997.

[13] S. Tortora, L. Tonin, S. Chisari, S. Micera, E. Menegatti, and F. Artoni, “Hybrid human-machine interface for gait decoding through bayesian fusion of eeg and emg classifiers,” Front. Neurorobot., vol. 14, 2020.

[14] J. J. Jui, I. T. Hettiarachchi, A. Bhatti, and D. Creighton, “Plvnet: Eegbased trust classification using phase locking value connectivity and deep neural networks,” Comput. Biol. Med., vol. 198, p. 111269, 2025.

[15] Z. Otarbay and A. Kyzyrkanov, “Svm-enhanced attention mechanisms for motor imagery eeg classification in brain-computer interfaces,” Front. Neurosci., vol. 19, 2025.

[16] A. P. Burgess, “On the interpretation of synchronization in eeg hyperscanning studies: a cautionary note,” Front. Hum. Neurosci., vol. 7, p. 881, 2013.

[17] Z. Zhang, P. Peng, S. B. Eickhoff, X. Lin, D. Zhang, and Y. Wang, “Neural substrates of the executive function construct, age-related changes, and task materials in adolescents and adults: Ale meta-analyses of 408 fmri studies,” Dev. Sci., vol. 24, no. 6, p. e13111, 2021.

[18] G. Darnai, G. Perlaki, G. Orsi, A. Ar<sup>´</sup> at´ o, A. Szente´ et al., “Language processing in internet use disorder: Task-based fmri study,” PLoS One, vol. 17, no. 6, p. e0269979, 2022.

[19] F. Wang, Y. Wan, M. Li, H. Huang, L. Li, X. Hou, J. Pan, Z. Wen, and J. Li, “Recent advances in fatigue detection algorithm based on eeg,” Intelligent Automation & Soft Computing, vol. 35, no. 3, pp. 3573–3586, 2023.

[20] A.-I. Karaiskou, C. Varon, C. A. Musluoglu, K. Alaerts, and M. D. Vos, “Eeg-based meditation decoding: tackling subject variability with spatial and temporal alignment,” Journal of Neural Engineering, vol. 22, no. 6, p. 66032, Dec. 2025.