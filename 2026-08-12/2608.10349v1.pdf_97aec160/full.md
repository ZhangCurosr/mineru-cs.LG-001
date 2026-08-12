# Beyond Detection Accuracy: Measuring Explanation Cost, Stability, and Utility for Resource-Aware IoT Intrusion Detection

Abdurrahman Tolay<sup>a,∗</sup>

<sup>a</sup>Independent Researcher, Istanbul, Türkiye

## Abstract

Machine-learning intrusion-detection studies commonly emphasize predictive accuracy while treating explanation generation as a computationally free post-processing step. This study jointly evaluates predictive efectiveness, explanation cost, local explanation stability, and selective explanation for binary Internet of Things (IoT) intrusion detection. A leakage-safe CICIoT2023 corpus was constructed with respect to exact 39-feature hashes through nonfinite-value handling, exact-feature deduplication, conservative original-label collision removal, and deterministic hash-level partitioning. Logistic Regression, Decision Tree, Random Forest, and XGBoost were evaluated on natural and balanced test distributions. The computational cost of tree-based Shapley additive explanations (TreeSHAP) was measured, stability was assessed under prediction-preserving perturbations, and validation-calibrated policies were used to allocate explanation workload. XGBoost provided the strongest overall predictive profile, while Random Forest produced the lowest false-positive rate. Explanation cost difered sharply by architecture: at 5,000 samples, TreeSHAP required 700.759 s for Random Forest and 1.471 s for XGBoost. Random Forest showed the strongest overall base-level explanation stability, while Decision Tree also remained highly stable in top-feature membership and attribution rank; XGBoost retained high rank and directional consistency but exhibited greater top-feature turnover and attribution-magnitude drift. On the balanced test, approximately 90% false-negative explanation coverage permitted compute savings of 28–32%, while approximately 95% coverage

permitted savings of 15–23%. Savings were substantially smaller under the attack-heavy natural prevalence. These results show that the operational value of explainable IoT intrusion detection depends on predictive quality, architecture-dependent explanation cost, local stability, workload prevalence, and selective invocation rather than on detection accuracy or explanation availability alone.

Keywords: Internet of Things, Intrusion Detection Systems, Explainable Artificial Intelligence, SHAP, Explanation Stability, Resource-Aware XAI, Cybersecurity

## 1. Introduction

Machine-learning intrusion-detection systems (IDSs) for Internet of Things (IoT) networks are commonly compared through predictive metrics, even though validation strategy, trafic composition, and deployment location materially afect the interpretation of those measurements [1, 2]. This evaluation is necessary, but it is not suficient for selecting a model for deployment at an edge gateway or another resource-constrained monitoring point. An operational detector must identify attacks while limiting false alarms, meet latency and memory constraints, and provide information that supports investigation when a decision requires human review. These requirements concern diferent properties of the system and cannot be inferred from accuracy alone.

The distinction is particularly important for highly imbalanced intrusiondetection data. When attack observations dominate a test distribution, a model can attain high accuracy while still misclassifying a substantial proportion of benign trafic. Such false positives consume analyst attention and may make an otherwise accurate detector impractical. Precision, recall, F1 score, false-positive rate, and precision–recall area under the curve (PR-AUC) therefore provide more informative views of security performance, provided that they are interpreted in relation to the class distribution on which they were measured.

Post-hoc explanation introduces a second deployment issue. Methods such as LIME and SHAP are designed to characterize individual predictions, and TreeSHAP specializes additive attribution for tree-based models [3, 4, 5]. Feature-attribution methods are nevertheless often used without placing their runtime on the same scale as prediction. In a constrained IoT setting, generating an explanation may require substantially more time and memory than producing the underlying class label. A detector that is inexpensive at inference time may become unsuitable if every prediction triggers a costly explanation. Predictive eficiency and explanation eficiency must consequently be measured separately.

Explanation availability also does not establish explanation reliability or operational value. Attribution sensitivity and adversarial manipulation have been demonstrated across post-hoc explanation settings [6, 7, 8]. An attribution may change under a small perturbation even when the model retains the same predicted class. Conversely, a stable attribution does not prove that the model relies on causal or semantically appropriate features. The usefulness of an explanation therefore depends on its computational cost, its behavior around the evaluated observation, and the decision context in which it is requested. These considerations motivate selective policies that reserve explanations for alerts, uncertain decisions, or other cases with a defined security priority.

This study develops that argument using CICIoT2023 [9] and a binary benign-versus-attack task. Logistic Regression implemented with the logisticloss SGDClassifier, Decision Tree, Random Forest, and XGBoost models are evaluated on a leakage-safe corpus constructed after non-finite-value removal, exact-feature deduplication, and conservative label-collision handling. Predictive behavior is assessed on both the natural cleaned test distribution and a balanced sensitivity test. The natural distribution represents the prevalence retained after corpus construction, whereas the balanced test reveals how prevalence-sensitive metrics change when benign and attack observations are equally represented.

The analysis proceeds in stages. It first establishes whether the models provide adequate attack detection and false-alarm control, then examines training cost, model size, and the integrity of a feature that receives unusually high native importance. It subsequently measures TreeSHAP cost, explanation stability under prediction-preserving perturbations, and selective-explanation cost–coverage trade-ofs. The purpose is not to argue that feature attribution makes an IDS trustworthy, but to determine whether explanatory analysis can be made computationally and operationally defensible in a resourceconstrained setting.

## 2. Related Work and Research Gap

## 2.1. IoT Intrusion Detection and CICIoT2023

Machine-learning IDS research spans signature, anomaly, and hybrid detection, with model choice, feature representation, class imbalance, validation strategy, and deployment location all afecting the meaning of reported performance [10, 1]. These dependencies are especially consequential in IoT networks, where heterogeneous devices and constrained monitoring points create diferent trafic and resource conditions from conventional enterprise networks [1, 2]. Consequently, a high score on a benchmark distribution is evidence about a particular experimental setting rather than direct evidence of feasibility on an operational gateway.

CICIoT2023 was constructed as a large-scale real-time IoT benchmark using a topology of 105 devices and an attack taxonomy covering 33 attacks in seven broad categories [9]. Its scale and family diversity have made it useful for binary and multiclass IDS experiments, but the authoritative benchmark description does not remove the need for study-specific audits of duplicates, label consistency, partition overlap, or prevalence efects. The present study therefore uses CICIoT2023 as its sole primary dataset while retaining original labels and attack families for diagnostics and constructing a binary target only after auditing the 39-feature representation.

Prior explainable IoT IDS research has proposed deep-learning frameworks, feature-attribution analyses, and surveys of opportunities for combining XAI with cyber defence [11, 12, 2]. These studies establish the relevance of explanations to IoT security analysis, but they also show that predictive evaluation, explanatory analysis, and deployment claims must be separated. The contribution here is not another classifier benchmark alone; it is an evaluation of the additional computational and operational consequences that arise when explanations are requested from otherwise competitive detectors.

## 2.2. Explainable AI for Intrusion Detection

LIME explains an individual prediction through a locally fitted surrogate model, whereas SHAP expresses a prediction through Shapley-value-based feature contributions [3, 4]. TreeSHAP provides algorithms specialized for decision trees and tree ensembles and supports local additive explanations that can be aggregated for broader model analysis [5]. These methods describe associations within a fitted predictor; their outputs do not by themselves establish causal relevance, semantic correctness, or analyst trust [13].

Applying SHAP or LIME to an IDS is already well established. Le et al. applied SHAP to provide global and local explanations of Decision Tree and Random Forest intrusion-detection decisions on IoTID20, NF-BoT-IoTv2, and NF-ToN-IoT-v2 [14]. Patil et al. used LIME as the explanatory component of an IDS evaluated on CICIDS2017 with Decision Tree, Random Forest, SVM, and ensemble or voting models [15]. Abou El Houda et al. and Keshk et al. developed explainable deep-learning frameworks for IoT intrusion detection [11, 12]. More recent work has systematized model and explainer comparisons. XAI-IDS examines feature importance across multiple network datasets and models [16], while Wang et al. study SHAP- and LIME-oriented transparency for IoT intrusion detection [17]. This literature makes clear that the use of SHAP is not itself a novel contribution. The remaining operational question is how explanatory cost, local repeatability, and invocation policy interact under a defined deployment workload.

## 2.3. Explanation Robustness, Stability, and Computational Cost

Post-hoc explanations can change under small input changes even when predictive behavior changes little, motivating explicit sensitivity and robustness evaluation [6, 7]. Explanation methods can also be manipulated or evaluated under adversarial conditions, further demonstrating that the presence of an explanation is not evidence of reliability [8]. In the IDS domain, E-XAI evaluates black-box explanation methods using descriptive accuracy, sparsity, stability, eficiency, robustness, and completeness across three network-intrusion datasets [18]. The gap addressed by the present study is therefore not the first consideration of XAI eficiency or stability in network intrusion detection.

Munilla and Khammas evaluate SHAP and LIME using DeepFool adversarial perturbations intended to stress the model decision boundary and alter model decisions [19]. The present study addresses a related but diferent construct. Its perturbations are bounded by validation-set statistics, restricted to selected continuous trafic features, accepted only when the predicted class remains unchanged, and required to satisfy an attack-confidence change no greater than 0.05. The two protocols consequently examine adversarial explanation robustness and local prediction-preserving explanation stability, respectively.

The TreeSHAP literature establishes eficient model-specific computation relative to generic Shapley-value estimation [5], but algorithmic eficiency does not determine the wall-clock burden of a particular trained tree architecture, batch size, or deployment workload. Likewise, broad XAI evaluation frameworks can include eficiency without measuring explanation latency relative to the underlying prediction across operational batch sizes. This study therefore reports absolute explanation time, per-sample latency, throughput, and the explanation-to-prediction ratio together. It additionally connects these measurements to selective invocation. Selective classification provides a methodological precedent for risk–coverage trade-ofs through abstention [20]. The present work transfers this resource-allocation principle to explanation invocation while leaving the IDS classification output unchanged.

## 2.4. Dataset Duplication, Leakage, and Evaluation Methodology

Redundant records are a longstanding concern in IDS benchmarking. The analysis of KDD Cup 99 by Tavallaee et al. showed that extensive duplication could bias learning and evaluation, and the derived NSL-KDD benchmark was designed in part to reduce this redundancy [21]. More generally, IDS surveys identify dataset construction, validation strategy, and realistic evaluation as recurring methodological limitations [10, 1]. The present work does not claim to discover the general duplicate-data problem. It provides a dataset-specific audit of CICIoT2023 and shows that 54.8176% of valid rows were exact duplicate instances in the evaluated 39-feature representation.

The audit also distinguishes three label levels: the original label, the attack family, and the binary target. Attack-to-attack original-label collisions remain consistent after mapping to Attack = 1, whereas benign–attack collisions do not. All cross-original-label hashes were nevertheless removed as a conservative semantic-consistency decision before deduplication. Deterministic hash-level partitioning then produced zero exact 39-feature-hash overlap among training, validation, and test partitions. This control is narrower and more precise than claiming that every form of dependence or leakage has been eliminated.

Table 1 summarizes the resulting research gap. The gap addressed by the present study is not the first consideration of XAI eficiency or stability in network intrusion detection. Rather, it is the joint operational evaluation of measured batch-dependent TreeSHAP cost relative to prediction, prediction-preserving local attribution stability, validation-calibrated selective explanation, prevalence-sensitive cost–coverage trade-ofs, and exact-featurehash-aware corpus construction.

Table 1: Comparison of representative explainable IDS studies and the present work. “Not explicitly evaluated” indicates that the item was not a stated empirical focus of the cited study.
<table><tr><td>Study</td><td>Dataset(s)</td><td>Detection models</td><td>XAI method</td><td>Explicit XAI cost?</td><td>Stability?</td><td>Selective XAI?</td><td>Hash-aware split?</td></tr><tr><td>Le et al. [14] IoTID20; NF-DT/RF</td><td>BoT-IoT-v2; ensemble- NF-ToN-IoT- tree v2</td><td>approach</td><td>SHAP</td><td>Not explicitlyNot explicitlyNot explicitlyNot explicitly evaluated</td><td>evaluated</td><td>evaluated</td><td>evaluated</td></tr><tr><td>Patil et al. [15]</td><td>CICIDS2017 DT; RF;</td><td>SVM; voting ensemble</td><td>LIME</td><td>Not explicitlyNot explicitlyNot explicitlyNot explicitly evaluated</td><td>evaluated</td><td>evaluated</td><td>evaluated</td></tr><tr><td>E-XAI [18]</td><td>CICIDS2017; Multiple NSL-KDD; RoEduNet- SIMARGL 2021</td><td>black-box models</td><td>SHAP; LIME</td><td>Evaluated</td><td></td><td>Evaluated Not explicitlyNot explicitly evaluated</td><td>evaluated</td></tr><tr><td>Wang et al. [17] Munilla and</td><td>IoT IDS datasets</td><td>Multiple MLSHAP; models CNN; DNN; SHAP;</td><td>LIME</td><td>Not explicitlyNot explicitlyNot explicitlyNot explicitly evaluated</td><td>evaluated</td><td>evaluated </td><td>evaluated</td></tr><tr><td>Khammas [19]</td><td>BoT-IoT; Edge-IIoT; N-BaIoT</td><td>LSTM; RF LIME</td><td></td><td>Not explicitly evaluated</td><td></td><td>Evaluated Not explicitlyNot explicitly evaluated</td><td>evaluated</td></tr><tr><td>Present study CICIoT2023 LR; DT; RF;SHAP</td><td></td><td>XGBoost</td><td>Tree</td><td>Yes</td><td>Evaluated</td><td>Yes</td><td>Yes</td></tr></table>

## 3. Research Motivation and Contributions

The immediate methodological motivation arises from the structure of CICIoT2023. The raw corpus contains extensive exact duplication. If rows were assigned independently to training and test partitions, identical 39- dimensional feature vectors could appear on both sides of the evaluation boundary. Performance measured under such a split could reflect exact feature-vector reuse rather than generalization to unseen observations. Label collisions create an additional annotation-consistency concern because some identical feature vectors are associated with diferent original labels or attack families. A rigorous predictive comparison must address these issues before model performance is interpreted.

A second motivation is the dependence of commonly reported metrics on the test distribution. The cleaned natural test set remains strongly attack dominated. Performance on this distribution is relevant to the constructed corpus, but it does not by itself show how a model behaves when benign trafic receives equal representation. Reporting a balanced sensitivity test alongside the natural test makes this dependence explicit without replacing the primary test distribution or merging CICIoT2023 with an external dataset.

The third motivation concerns the gap between prediction and explanation. The resource requirements of a classifier do not determine the cost of its post-hoc explanations. Moreover, neither native feature importance nor a single attribution plot establishes that a feature is indispensable to prediction or that an explanation is reliable. The targeted audit and ablation of Number illustrate why feature integrity and predictive dependence must be examined before explanation outputs are given operational meaning.

This work makes four connected contributions. First, it constructs a leakage-safe binary CICIoT2023 corpus through non-finite-value handling, exact-feature deduplication, conservative original-label collision removal, and deterministic hash-level partitioning. Second, it compares Logistic Regression, Decision Tree, Random Forest, and XGBoost under natural and balanced test distributions, emphasizing recall, F1, false-positive rate, and PR-AUC rather than treating accuracy as the primary criterion. Third, it measures prediction and model eficiency and audits the unusually influential Number feature through targeted ablation. Fourth, it empirically measures Tree-SHAP explanation cost, explanation stability under prediction-preserving perturbations, and selective-explanation cost–coverage trade-ofs.

The resulting manuscript is organized around trade-ofs rather than a universal model ranking. A model may provide the highest F1 score without producing the lowest false-positive rate; a compact single tree may remain competitive with larger ensembles; and an eficient predictor may still incur prohibitive explanation overhead. Separating these dimensions is necessary to move from detection accuracy toward operationally useful explainable intrusion detection.

The central question is therefore: when predictive performance is similar, what is the computational and operational cost of obtaining explanations, and can explanation workload be reduced without losing security-relevant explanatory coverage?

## 4. Research Questions

The study addresses four research questions that follow the progression from predictive performance to operational explanation.

RQ1: Predictive efectiveness. How do Logistic Regression, Decision Tree, Random Forest, and XGBoost compare on the binary CICIoT2023 task when recall, precision, F1 score, false-positive rate, and PR-AUC are prioritized over accuracy?

RQ2: Explanation cost. How much latency and throughput overhead does TreeSHAP introduce relative to prediction alone for the tree-based detectors, and how does this cost vary across model architecture and batch size?

RQ3: Explanation stability. How stable are local feature attributions under small prediction-preserving perturbations, and how does stability vary across model architecture, perturbation strength, and benign-versus-attack observations?

RQ4: Selective explanation. To what extent can security-aware or uncertainty-aware selection policies reduce explanation workload while retaining the decisions most relevant to operational review?

RQ1–RQ4 are answered by the empirical study. Keeping the questions separate prevents predictive efectiveness from being used as a proxy for explanation eficiency, stability, or utility.

## 5. Experimental Methodology

## 5.1. CICIoT2023 Dataset

CICIoT2023 is the sole dataset used in the present study [9]. The benchmark was constructed around 105 IoT devices and an attack taxonomy containing 33 attack types [9]. The raw corpus analyzed here contains 46,776,700 observations and 39 features. It includes benign trafic and attacks belonging to the DDoS, DoS, Recon, Web, BruteForce, Spoofing, and Mirai families. The primary classification task assigns benign observations to class 0 and all attacks to class 1. Original attack labels and family assignments are retained for descriptive and diagnostic analysis even though the fitted models address the binary task.

Table 2 reports the corrected family distribution in the raw corpus. DDoS and DoS trafic account for most observations, while benign trafic constitutes a much smaller proportion. This distribution motivates the use of classsensitive metrics and a separate balanced sensitivity evaluation. No external dataset was merged with CICIoT2023, and cross-dataset validation is outside the present experimental scope.

Table 2: Corrected family distribution in the raw CICIoT2023 corpus.
<table><tr><td>Family</td><td>Observations</td><td>Family</td><td>Observations</td></tr><tr><td>DDoS</td><td>33,984,450</td><td>DoS</td><td>7,845,120</td></tr><tr><td>Mirai</td><td>2,634,054</td><td>Benign</td><td>1,098,191</td></tr><tr><td>Recon</td><td>690,534</td><td>Spoofing</td><td>486,458</td></tr><tr><td>Web</td><td>24,829</td><td>BruteForce</td><td>13,064</td></tr></table>

## 5.2. Data-Quality Audit

The data-quality audit covered all 309 CSV files. It examined schema consistency and counted missing and infinite values before any model training or partitioning. Across the 46,776,700 raw rows, the audit identified 1,449 missing-value cells and 1,037 infinite-value cells. These are cell counts rather than row counts, and more than one invalid cell could occur in a single record. The non-finite cells were concentrated in 1,040 rows, all of which were removed. This operation left 46,775,660 valid observations. No schema inconsistency was found across the audited files.

The distinction between invalid cells and removed rows is retained in the reporting because adding the missing- and infinite-cell counts would overstate the number of excluded observations. Removal was performed before hashing so that duplicate and collision statistics were computed only over finite 39-dimensional feature vectors.

Table 3 summarizes the file-level and observation-level data-quality audit.

Table 3: Data-quality audit of the raw CICIoT2023 files.
<table><tr><td>Audit quantity</td><td>Count</td></tr><tr><td>CSV files audited</td><td>309</td></tr><tr><td>Raw rows</td><td>46,776,700</td></tr><tr><td>Missing-value cells Infinite-value cells</td><td>1,449</td></tr><tr><td>Rows removed for non-finite values</td><td>1,037</td></tr><tr><td>Schema inconsistencies</td><td>1,040</td></tr><tr><td>Valid rows after removal</td><td>0 46,775,660</td></tr></table>

## 5.3. Duplicate and Label-Collision Analysis

An exact hash was computed over the complete 39-feature vector of every valid observation. The 46,775,660 valid rows contained 21,134,351 unique feature vectors, 1,893,344 duplicate groups, and 25,641,309 duplicate instances. The resulting exact duplicate rate was 54.8176%. This rate is suficiently high that a naïve random row-level split could place identical feature vectors in training and test data, allowing exact feature-vector leakage and potentially optimistic performance estimates.

Duplicate feature vectors were also audited for label consistency. The analysis identified 533,528 hashes associated with more than one original label, 451,969 hashes crossing attack-family boundaries, and 905 hashes associated with both benign and attack observations. These categories are related and should not be summed as if they were disjoint. Benign-versus-attack collisions are inconsistent at the binary-target level because the same feature vector maps to both class 0 and class 1. In contrast, attack-to-attack original-label collisions remain binary-target consistent because every involved attack label maps to Attack = 1. The preprocessing procedure nevertheless excluded all cross-original-label hashes as a conservative methodological choice. This decision preserved semantic label consistency in the clean primary corpus and avoided retaining feature-identical observations carrying conflicting source annotations. After collision handling, 20,600,823 safe unique hashes remained.

Table 4 reports the duplicate and label-collision counts.

Table 4: Exact-duplicate and label-collision audit.
<table><tr><td>Audit quantity</td><td>Count</td></tr><tr><td>Valid rows</td><td>46,775,660</td></tr><tr><td>Unique 39-dimensional feature vectors</td><td>21,134,351</td></tr><tr><td>Duplicate groups</td><td>1,893,344</td></tr><tr><td>Duplicate instances</td><td>25,641,309</td></tr><tr><td>Exact duplicate rate</td><td>54.8176%</td></tr><tr><td>Cross-original-label collision hashes</td><td>533,528</td></tr><tr><td>Cross-family collision hashes</td><td>451,969</td></tr><tr><td>Benign-versus-attack collision hashes</td><td>905</td></tr><tr><td>Safe unique hashes after collision handling</td><td>20,600,823</td></tr></table>

## 5.4. Leakage-Safe Corpus Construction

The clean corpus was constructed in five ordered steps. Rows containing a missing or infinite value were removed. An exact hash was then computed over each 39-feature vector. As the conservative semantic-consistency choice described above, hashes associated with conflicting original labels were excluded, and the remaining exact duplicates were collapsed to one observation per feature vector. Finally, each surviving feature hash was assigned deterministically to a training, validation, or test partition.

The resulting corpus contained 20,600,823 unique observations. The nominal split was 70% training, 15% validation, and 15% test. Deterministic hash assignment produced 14,421,214 training observations, 3,089,510 validation observations, and 3,090,099 natural-test observations. The feature-hash overlap was zero for train versus validation, train versus test, and validation versus test. Exact feature-vector leakage across the three partitions was therefore eliminated.

Figure 1 summarizes the ordered construction and partitioning procedure.   
Table 5 reports the resulting partition sizes and class composition.

Table 5: Class composition of the leakage-safe corpus and balanced sensitivity test.
<table><tr><td>Partition</td><td>Total</td><td>Benign</td><td>Attack</td></tr><tr><td>Training (70%)</td><td>14,421,214</td><td>765,187</td><td>13,656,027</td></tr><tr><td>Validation (15%)</td><td>3,089,510</td><td>164,271</td><td>2,925,239</td></tr><tr><td>Natural test (15%)</td><td>3,090,099</td><td>163,513</td><td>2,926,586</td></tr><tr><td>Balanced sensitivity test</td><td>327,026</td><td>163,513</td><td>163,513</td></tr></table>

This procedure eliminates exact overlap, but it does not imply that all statistical dependence among related non-identical flows has been removed. The evaluation is therefore described specifically as leakage-safe with respect to exact 39-feature hashes.

## 5.5. Predictive Models

The model set represents four levels of functional and structural complexity. Logistic Regression provides a lightweight linear baseline and was implemented with SGDClassifier(loss="log\_loss") in scikit-learn [22]. It was trained for five epochs on all 14,421,214 training observations. A StandardScaler was fitted on the training data only and then applied without refitting to the validation and test sets. Balanced class weights were 9.4233266 for benign trafic and 0.5280165 for attack trafic. The principal classification threshold was 0.50. A threshold of 0.11 selected through validation-set F1 optimization was retained only for sensitivity analysis.

![](images/cc861bd212a6025ccafe74a1c162f222ace27218b2d501cc7ecebf6161e91139.jpg)  
Figure 1: Leakage-aware CICIoT2023 corpus-construction workflow. The procedure removes non-finite observations, conservatively excludes cross-original-label feature hashes, collapses exact duplicates, and partitions the surviving unique feature vectors deterministically.

Decision Tree, Random Forest, and XGBoost were trained on a common deterministic sample of 5,000,000 observations from the training partition. Holding this sample constant supports direct comparison among the three nonlinear models. The Decision Tree used max\_depth=20, min\_samples\_leaf=20, and balanced class weights. The Random Forest [23] contained 150 estimators, with max\_depth=20, min\_samples\_leaf=20, the square root of the feature count considered at each split, and balanced-subsample class weights.

XGBoost [24] used 300 estimators, a maximum depth of 8, a learning rate of 0.10, a row-subsampling proportion of 0.80, a column-subsampling proportion of 0.80, a minimum child weight of 5, and the histogram tree method. Balanced sample weighting was derived from the full training distribution. The global experiment seed was 2026. The test partitions were not used for threshold selection or model fitting.

## 5.6. Experimental Environment

Experiments were executed on a Microsoft Windows 11 Pro host equipped with an Intel(R) Core(TM) 7 240H processor, 10 physical cores, 16 logical processors, and 31.71 GB of system RAM. The operating-system version was 10.0.26200, build 26200. The Python version was 3.12.0. Principal package versions were NumPy 2.4.6, pandas 3.0.5, scikit-learn 1.9.0, XGBoost 3.4.0, SHAP 0.52.0, joblib 1.5.3, psutil 7.2.2, and PyArrow 25.0.0. The global random seed was 2026.

No explicit values were assigned to OMP\_NUM\_THREADS, MKL\_NUM\_THREADS, OPENBLAS\_NUM\_THREADS, or NUMEXPR\_NUM\_THREADS. This observation does not imply that the underlying libraries used a specific fixed thread count. All reported training, inference, and explanation timings are wall-clock measurements obtained in this hardware and software environment. They are environment dependent and must not be interpreted as direct measurements on a constrained edge device.

## 5.7. Reproducibility and Experimental Controls

Reproducibility controls were applied across data construction, model evaluation, explanation timing, stability analysis, and selective-policy calibration. The corpus split was deterministic at the complete 39-feature-hash level, and the global experiment seed was fixed at 2026. Decision Tree, Random Forest, and XGBoost used the same deterministic 5,000,000-observation training sample. The natural and balanced evaluations reused the same fitted models and fixed decision thresholds; neither distribution initiated retraining.

Selective-explanation thresholds were calibrated only from validation false negatives, and test labels were never used to choose a threshold. TreeSHAP cost used one fixed balanced master sample of 5,000 observations, with the prediction and adaptive explanation repeat schedules stated in Section 7. Perturbation geometry in the stability experiment was derived from validation-set interquartile ranges and percentile bounds. Full-test selective-explanation costs are identified as throughput-based projections rather than direct runtime measurements. The deterministic seeds, split construction, model configurations, evaluation protocols, and software environment are documented to support reproducibility. The experimental scripts, trained model artifacts, derived result files, and reproducibility documentation supporting this study are publicly available through the project repository and its archived Zenodo release (DOI: 10.5281/zenodo.21879038).

## 5.8. Evaluation Metrics

Let true positives (TP) denote attacks classified as attacks, false negatives (FN) denote missed attacks, false positives (FP) denote benign observations classified as attacks, and true negatives (TN) denote correctly classified benign observations. The principal threshold-dependent metrics are

$$
\mathrm { P r e c i s i o n } = \frac { \mathrm { T P } } { \mathrm { T P } + \mathrm { F P } } ,\tag{1}
$$

$$
{ \mathrm { R e c a l l } } = { \frac { \mathrm { T P } } { \mathrm { T P } + { \mathrm { F N } } } } ,\tag{2}
$$

$$
\mathrm { F 1 } = 2 \frac { \mathrm { P r e c i s i o n R e c a l l } } { \mathrm { P r e c i s i o n + R e c a l l } } ,\tag{3}
$$

$$
\mathrm { F P R } = \frac { \mathrm { F P } } { \mathrm { F P } + \mathrm { T N } } .\tag{4}
$$

PR-AUC is used as the principal threshold-independent summary because it characterizes the precision–recall trade-of and remains directly connected to attack-detection performance. Accuracy and receiver-operating-characteristic area under the curve (ROC-AUC) are reported as secondary metrics. Accuracy is not used as the primary basis for model selection because the cleaned corpus remains severely imbalanced. False-positive rate receives separate attention because benign false alarms create an operational burden that may not be visible in an aggregate F1 or accuracy value.

Computational measurements comprise training time, inference latency, and serialized model size. Prediction cost and explanation cost are treated separately. The TreeSHAP experiment measured prediction latency, explanation latency, prediction and explanation throughput, observed process-memory efects, and the explanation-to-inference overhead ratio at batch sizes of 1, 10, 100, 1,000, and 5,000.

## 5.9. Natural and Balanced Evaluation Protocol

The natural test set contains 3,090,099 observations: 163,513 benign and 2,926,586 attack observations. It preserves the attack-dominated prevalence of the cleaned corpus and is the principal test distribution. A balanced sensitivity test was constructed using all 163,513 benign test observations and 163,513 attack observations, for a total of 327,026. The balanced test does not supplant the natural test and is not used for tuning. It provides a controlled view of metric sensitivity when the two binary classes have equal representation.

Both tests use the same fitted models and fixed classification thresholds. Consequently, changes in precision, F1, or accuracy across the two tests reflect the evaluation distribution rather than retraining. The false-positive rate remains directly comparable because the benign observations are unchanged. PR-AUC is reported separately for each distribution because precision depends on prevalence. Results are interpreted jointly: the natural test represents performance on the cleaned corpus distribution, while the balanced test exposes conclusions that depend strongly on attack prevalence.

## 6. Predictive Results

Tables 6 and 7 report predictive performance on the natural and balanced test distributions, respectively.

## 6.1. Linear vs Nonlinear Models

The diference between Logistic Regression and the three tree-based models is visible on both test distributions. At the 0.50 threshold, Logistic Regression attained 0.828777 recall and 0.897914 F1 on the natural test, but its falsepositive rate was 0.308367. Its accuracy of 0.821520 therefore does not fully describe the operational weakness of the classifier: almost one third of benign observations were classified as attacks. The nonlinear models reduced the false-positive rate to 0.009718 or below while producing F1 scores above 0.984 on the same test distribution.

Table 6: Binary detection performance on the natural test set. Values are expressed as proportions.
<table><tr><td>Model</td><td>Accuracy</td><td>Precision</td><td>Recall</td><td>F1</td><td>FPR</td><td>ROC-AUC</td><td>PR-AUC</td></tr><tr><td>Logistic Regression</td><td>0.821520</td><td>0.979635</td><td>0.828777</td><td>0.897914</td><td>0.308367</td><td>0.816803</td><td>0.987239</td></tr><tr><td>Decision Tree</td><td>0.972772</td><td>0.999442</td><td>0.971794</td><td>0.985424</td><td>0.009718</td><td>0.992142</td><td>0.999379</td></tr><tr><td>Random Forest</td><td>0.970275</td><td>0.999869</td><td>0.968741</td><td>0.984059</td><td>0.002269</td><td>0.995555</td><td>0.999755</td></tr><tr><td>XGBoost</td><td>0.973098</td><td>0.999735</td><td>0.971852</td><td>0.985597</td><td>0.004605</td><td>0.996212</td><td>0.999791</td></tr></table>

Table 7: Binary detection performance on the balanced sensitivity test. Values are expressed as proportions.
<table><tr><td>Model</td><td>Accuracy</td><td>Precision</td><td>Recall</td><td>F1</td><td>FPR</td><td>ROC-AUC</td><td>PR-AUC</td></tr><tr><td>Logistic Regression</td><td>0.760692</td><td>0.729055</td><td>0.829751</td><td>0.776151</td><td>0.308367</td><td>0.817303</td><td>0.840948</td></tr><tr><td>Decision Tree</td><td>0.981228</td><td>0.990103</td><td>0.972173</td><td>0.981056</td><td>0.009718</td><td>0.992277</td><td>0.990255</td></tr><tr><td>Random Forest</td><td>0.983533</td><td>0.997665</td><td>0.969336</td><td>0.983296</td><td>0.002269</td><td>0.995602</td><td>0.996670</td></tr><tr><td>XGBoost</td><td>0.983842</td><td>0.995286</td><td>0.972290</td><td>0.983653</td><td>0.004605</td><td>0.996248</td><td>0.997126</td></tr></table>

The balanced evaluation makes the limitation of the linear model more explicit. Logistic Regression retained 0.829751 recall, but precision declined to 0.729055 and F1 to 0.776151. The false-positive rate remained 0.308367 because the benign evaluation set was unchanged. By contrast, the nonlinear models produced balanced-test F1 scores from 0.981056 to 0.983653. The observation is statistical rather than causal: the results show that the fitted linear decision function separated the evaluated trafic less efectively, but they do not identify why individual features interact nonlinearly.

Threshold selection further illustrates the danger of relying on a single aggregate objective. Validation-based F1 optimization selected a Logistic Regression threshold of 0.11. On the natural test, this threshold produced 0.887032 accuracy, 0.959895 precision, 0.919122 recall, 0.939066 F1, and 0.687321 FPR. On the balanced test, it produced 0.616067 accuracy, 0.572236 precision, 0.919456 recall, 0.705435 F1, and 0.687321 FPR. ROC-AUC and PR-AUC are threshold independent: they remained 0.816803 and 0.987239 on the natural test and 0.817303 and 0.840948 on the balanced test. Although the lower threshold improved natural-test recall and F1, it produced an operationally unacceptable increase in benign false positives. The 0.50 threshold is therefore retained as the principal baseline, and the 0.11 threshold is reported only as sensitivity analysis.

## 6.2. Ensemble Model Comparison

The three tree-based models were close in aggregate predictive performance, but their error profiles were not identical. XGBoost achieved the highest natural-test F1, 0.985597, and the highest balanced-test F1, 0.983653. It also produced the highest ROC-AUC and PR-AUC on both distributions: 0.996212 and 0.999791 on the natural test, and 0.996248 and 0.997126 on the balanced test. These observed metrics give XGBoost the strongest overall predictive profile among the evaluated models. However, the diferences among the tree-based models are small and are interpreted descriptively rather than as evidence of statistical superiority.

Random Forest produced the lowest false-positive rate, 0.002269, on both tests. Its natural-test precision was 0.999869 and its balanced-test precision was 0.997665, both higher than the corresponding XGBoost values. This false-alarm advantage was accompanied by lower recall: 0.968741 on the natural test and 0.969336 on the balanced test. A deployment that places greater cost on benign false alarms could therefore prefer Random Forest even though its F1 is slightly lower.

Decision Tree remained notably competitive despite its simpler structure. Its natural-test F1 of 0.985424 was close to XGBoost’s 0.985597, and its recall was 0.971794 compared with 0.971852 for XGBoost. On the balanced test, Decision Tree produced 0.981056 F1 and a 0.009718 false-positive rate. The single tree does not match the ensembles on every measure, but its performance is suficiently close that model complexity and resource cost become relevant selection criteria. Detection accuracy alone does not resolve this trade-of.

The distribution comparison also changes the interpretation of apparently high values. Precision and accuracy decrease or increase as class prevalence changes, even though each model and threshold remain fixed. PR-AUC likewise changes between the natural and balanced tests. These shifts reinforce the need to report the test distribution alongside every metric and to avoid generalizing a result beyond the evaluated trafic composition.

## 6.3. Eficiency Trade-ofs

Table 8 reports the completed training-time and serialized-size measurements. The comparison between Decision Tree, Random Forest, and XGBoost is based on the same 5,000,000-observation sample. Logistic Regression used all 14,421,214 training observations and five epochs, so its 38.14-second training time is not directly comparable with the common-sample tree results without accounting for the diferent training workloads.

Table 8: Recorded training time and serialized classifier-model size.
<table><tr><td>Model</td><td>Training observations</td><td>Training time (s)</td><td>Model size (MB)</td></tr><tr><td>Logistic Regression</td><td>14,421,214</td><td>38.14</td><td>0.002238</td></tr><tr><td>Decision Tree</td><td>5,000,000</td><td>13.72</td><td>≈0.498</td></tr><tr><td>Random Forest</td><td>5,000,000</td><td>99.28</td><td>22.466</td></tr><tr><td>XGBoost</td><td>5,000,000</td><td>35.02</td><td>0.895</td></tr></table>

Note: The separately serialized StandardScaler occupied 0.001433 MB; the classifier and scaler together occupied approximately 0.003671 MB.

Decision Tree required 13.72 seconds to train and approximately 0.498 MB of serialized storage on the experimental host. Its combination of compact serialized size, lower host training time, and F1 close to the ensemble models makes it a candidate for further evaluation in resource-constrained deployment settings rather than merely a weak baseline. These measurements do not demonstrate edge-device feasibility. Random Forest required 99.28 seconds and 22.466 MB, the largest host training time and model size among the three models trained on the common sample. The additional storage and training cost accompany the lowest observed false-positive rate, showing that eficiency and security error costs can point toward diferent choices.

XGBoost trained in 35.02 seconds on the experimental host and occupied 0.895 MB. Given its leading F1, ROC-AUC, and PR-AUC values, these measurements support the interpretation that XGBoost provides the strongest predictive/host-eficiency compromise among the completed experiments. This is not a universal ranking or evidence of edge-device feasibility. Decision Tree is smaller and faster to train, while Random Forest has the lowest false-positive rate. Moreover, serialized size and host training time do not establish inference cost, and none of these measurements establishes explanation cost. The Stage-5 latency and TreeSHAP benchmark is therefore necessary before deployment eficiency can be assessed, and actual constrained-hardware evaluation would still be required.

## 6.4. Feature-Integrity Analysis

Native XGBoost feature importance was heavily concentrated on Number. High native importance can arise because a feature provides useful split points, but it does not establish causal relevance, data leakage, or dependence that cannot be substituted by other features. A targeted integrity audit was therefore performed before using this importance pattern to motivate explanation claims.

The benign-versus-attack KS statistic for Number was 0.939147, the largest among the audited features. Other high values included 0.861050 for HTTPS, 0.806869 for ack\_flag\_number, 0.793179 for IAT, 0.780622 for Header\_Length, 0.775387 for Rate, and 0.735218 for ack\_count. These statistics show strong univariate distributional separation, but they do not identify whether the fitted model depends exclusively on any one feature.

Table 9 reports the KS statistics used in the targeted integrity audit.

Table 9: Benign-versus-attack KS statistics for features examined in the targeted integrity audit.
<table><tr><td>Feature</td><td>KS statistic</td><td>Feature</td><td>KS statistic</td></tr><tr><td>Number</td><td>0.939147</td><td>HTTPS</td><td>0.861050</td></tr><tr><td>ack_flag_number</td><td>0.806869</td><td>IAT</td><td>0.793179</td></tr><tr><td>Header_Length</td><td>0.780622</td><td>Rate</td><td>0.775387</td></tr><tr><td>ack_count</td><td>0.735218</td><td></td><td></td></tr></table>

Within the natural test set, Number had 99 unique values and an approximate unique-value ratio of 0.0032%. Its most common value accounted for approximately 88.74% of observations, and it was not flagged as highuniqueness or identifier-like. The benign distribution had a mean of approximately 9.998 and a median of 10, whereas the attack distribution had a mean of approximately 94.40 and a median of 100. The feature therefore separates broad trafic regimes while taking a small set of repeated values. This evidence neither proves nor disproves leakage; it establishes the need for a direct dependence test.

## 6.5. Number Feature Ablation

Decision Tree and XGBoost were retrained after removing Number, leaving 38 features. Each ablated model used the same deterministic 5,000,000- observation training sample as its 39-feature counterpart, so the comparison isolates the removal of the feature within the recorded protocol.

Table 10 compares the 39-feature and ablated configurations.

Removing Number changed XGBoost F1 by −0.000023 on the natural test and −0.000028 on the balanced test. Predictive performance was therefore virtually unchanged despite the feature’s high native importance and strong class separation. Decision Tree natural-test F1 decreased from 0.985424 to 0.984974, a change of −0.000450, whereas balanced-test F1 increased from 0.981056 to 0.981313, a change of +0.000257. Its false-positive rate decreased from 0.009718 to 0.008183, corresponding to ∆FPR = −0.001535.

Table 10: F1 and false-positive-rate results from the targeted Number ablation.
<table><tr><td>Model</td><td>Test distribution</td><td>Metric</td><td>39 features</td><td>Without Number</td><td>∆</td></tr><tr><td>XGBoost</td><td>Natural</td><td>F1</td><td>0.985597</td><td>0.985574</td><td>-0.000023</td></tr><tr><td>XGBoost</td><td>Balanced</td><td>F1</td><td>0.983653</td><td>0.983625</td><td>-0.000028</td></tr><tr><td>Decision Tree</td><td>Natural</td><td>F1</td><td>0.985424</td><td>0.984974</td><td>-0.000450</td></tr><tr><td>Decision Tree</td><td>Balanced</td><td>F1</td><td>0.981056</td><td>0.981313</td><td>+0.000257</td></tr><tr><td>Decision Tree</td><td>Both tests</td><td>FPR</td><td>0.009718</td><td>0.008183</td><td>-0.001535</td></tr></table>

The ablation demonstrates that XGBoost’s predictive efectiveness does not materially depend on Number alone. It also shows why native feature importance should not be equated with explanation quality or indispensable model dependence. More broadly, the result reinforces the central argument of this section: even after a detector achieves high predictive performance, model selection still requires evidence about resource cost and the behavior of the explanatory layer. The following sections therefore evaluate TreeSHAP cost, stability, and selective invocation relative to prediction alone.

## 7. Explanation Cost

TreeSHAP provides model-specific additive feature attributions for treebased predictors [4, 5]. Its cost was evaluated here for Decision Tree, Random Forest, and XGBoost using a deterministic balanced master sample of 5,000 observations containing 2,500 benign and 2,500 attack cases, each represented by the same 39 features used for prediction. Measurements were collected at batch sizes of 1, 10, 100, 1,000, and 5,000. Prediction timing used 10 repetitions at every batch size. TreeSHAP timing used an adaptive schedule of 10 repetitions for batches 1 and 10, five for batch 100, three for batch 1,000, and one for batch 5,000.

For model m and batch b, let $T _ { m , b } ^ { \mathrm { p r e d i c t } }$ denote prediction time and $T _ { m , b } ^ { \mathrm { e x p . } }$ plain denote TreeSHAP time. The explanation-overhead ratio is

$$
O _ { \mathrm { X A I } } ( m , b ) = \frac { T _ { m , b } ^ { \mathrm { e x p l a i n } } } { T _ { m , b } ^ { \mathrm { p r e d i c t } } } ,\tag{5}
$$

and the absolute incremental explanation time is

$$
C _ { \mathrm { X A I } } ( m , b ) = T _ { m , b } ^ { \mathrm { e x p l a i n } } - T _ { m , b } ^ { \mathrm { p r e d i c t } } .\tag{6}
$$

An overhead ratio is informative only in conjunction with absolute latency. When prediction takes tens of microseconds, a moderate absolute explanation time can produce a very large ratio. The analysis therefore considers absolute explanation latency, latency per sample, explanation throughput, and the explanation-to-prediction ratio jointly.

## 7.1. Scaling with Batch Size

Figure 2 shows that TreeSHAP cost increased with batch size for all three architectures, but at markedly diferent rates and absolute levels. Decision Tree prediction medians were 0.000049, 0.000054, 0.000060, 0.000122, and 0.000433 s across batches 1, 10, 100, 1,000, and 5,000. The reported TreeSHAP times were 0.001255, 0.011603, 0.113108, 1.123412, and 5.642689 s, producing overhead ratios of 25.46×, 216.06×, 1,894.60×, 9,196.98×, and 13,025.60×. Under the adaptive schedule, the 5,000-sample TreeSHAP value was one recorded measurement; the smaller-batch values summarized repeated timings. The large terminal ratio does not mean that Decision Tree explanations were more expensive than Random Forest explanations; it is driven in part by the exceptionally small Decision Tree prediction denominator.

Random Forest incurred the largest explanation burden. Prediction times were 0.014148, 0.014128, 0.019230, 0.024609, and 0.024835 s across the five batches, while the reported TreeSHAP times were 0.125572, 1.293386, 13.077294, 132.182032, and 700.758801 s. The 5,000-sample TreeSHAP value was one recorded measurement. The corresponding overhead ratios were 8.88×, 91.55×, 680.05×, 5,371.35×, and 28,217.04×.

XGBoost scaled more favorably. Prediction times were 0.000267, 0.000465, 0.000683, 0.001326, and 0.004109 s, while the reported TreeSHAP times were 0.003321, 0.009487, 0.034625, 0.277383, and 1.471088 s. The 5,000- sample TreeSHAP value was one recorded measurement. Its overhead ratios were 12.42×, 20.41×, 50.70×, 209.15×, and 357.99×. Although explanation remained more expensive than prediction, both the absolute time and scaling behavior were substantially more favorable than for Random Forest.

Table 11 summarizes the absolute and relative costs at the 5,000-sample operating point.

At 5,000 samples, Random Forest required approximately 476 times the absolute TreeSHAP time of XGBoost: 700.758801 s compared with 1.471088 s.

![](images/fd96701d0b32115c3b4b6aaea147fd65c36dda073e0a9898564283807b2368b0.jpg)  
Figure 2: TreeSHAP explanation-time scaling on the experimental host. Both axes are logarithmic; marker shapes and line styles distinguish models independently of color.

Table 11: TreeSHAP computational cost at the 5,000-sample operating point.
<table><tr><td>Model</td><td>Prediction (s)</td><td>Explanation (s)</td><td>µs/sample</td><td>Samples/s</td><td>Overhead</td></tr><tr><td>Decision Tree</td><td>0.000433</td><td>5.642689</td><td>1,128.54</td><td>886.10</td><td>13,025.60×</td></tr><tr><td>Random Forest</td><td>0.024835</td><td>700.758801</td><td>140,151.76</td><td>7.135</td><td>28,217.04×</td></tr><tr><td>XGBoost</td><td>0.004109</td><td>1.471088</td><td>294.22</td><td>3,398.84</td><td>357.99×</td></tr></table>

Its per-sample explanation latency was 140,151.76 µs and its explanation throughput was 7.135 samples/s. Decision Tree required 1,128.54 µs per sample and produced 886.10 explanations/s, whereas XGBoost required 294.22 µs per sample and produced 3,398.84 explanations/s. These results establish that predictive eficiency does not imply explanation eficiency. Random Forest combined strong predictive performance with an exceptionally high TreeSHAP burden, while XGBoost combined strong predictive performance with substantially higher explanation throughput.

## 7.2. TreeSHAP Numerical Validation

Decision Tree and Random Forest additivity residuals were efectively at numerical precision. For XGBoost, an initial probability-space diagnostic was not a valid additivity test because the default TreeSHAP output is additive in the model’s raw-margin, or log-odds, space. A separate validation used 50 observations, comprising 25 benign and 25 attack cases with 39 features. The raw-margin mean absolute error was $1 . 7 0 8 4 4 1 7 1 4 0 7 2 \times 1 0 ^ { - 6 }$ and the maximum absolute error was $4 . 3 3 3 6 5 2 6 6 8 7 2 0 \times 1 0 ^ { - 6 }$ . After applying the sigmoid transformation, probability mean absolute error was 3.101912290149× $1 0 ^ { - 8 }$ and maximum probability error was $2 . 1 4 0 3 5 7 0 9 5 1 7 1 \times 1 0 ^ { - 7 }$ . Classification agreement was 1.000000. These results confirm numerical consistency of the TreeSHAP reconstruction; they do not establish causal validity or semantic correctness of the attributions.

The Stage-5A memory statistic was an observed before/after process resident-set-size diference. It was not a true peak-memory measurement and may not capture transient allocation peaks. Memory results are therefore not interpreted as peak requirements.

## 8. Explanation Stability

An explanation should not be treated as locally reliable solely because it can be generated. Prior work has shown that feature attributions can be sensitive to small input changes and adversarial manipulation [6, 7, 8]; IDSspecific work has likewise evaluated explanation behavior under adversarial perturbations [18, 19]. The present protocol instead asks whether a small, semantically constrained perturbation that preserves both the predicted class and a similar confidence produces substantial changes in attribution structure. Stability was assessed for 100 base observations containing 50 benign and

50 attack cases. Ten candidate perturbations were generated for every base observation, with at most five accepted perturbations per sample and model.

For feature $j ,$ the perturbation scale was $\sigma _ { j } = 0 . 0 1 \mathrm { I Q R } _ { j } .$ , where interquartile ranges were estimated from the Stage-3 validation data. Perturbations were restricted to nine continuous trafic statistics: Rate, Tot sum, Min, Max, AVG, Std, Tot size, IAT, and Variance. Binary protocol and flag features and discrete or count-like variables were not perturbed. Candidate values were clipped to validation-set 0.5th–99.5th percentile ranges and constrained to be non-negative. A candidate $x ^ { \prime }$ was accepted only if

$$
f ( x ^ { \prime } ) = f ( x ) \quad { \mathrm { a n d } } \quad | p ( x ^ { \prime } ) - p ( x ) | \leq 0 . 0 5 ,\tag{7}
$$

where $p ( \cdot )$ is the model’s attack confidence.

Four complementary metrics were used. Top-5 Jaccard similarity measures membership overlap among the five largest absolute attributions. Spearman correlation measures agreement between absolute-attribution rankings. Cosine similarity measures signed directional agreement between complete attribution vectors. Normalized $L _ { 1 }$ change measures attribution-magnitude drift. Higher Jaccard, Spearman, and cosine values indicate greater stability, whereas lower normalized $L _ { 1 }$ values indicate greater stability.

## 8.1. Statistical Unit and Bootstrap Analysis

Stages 6A and 6B generated multiple accepted perturbations from the same original base observation. Perturbation pairs sharing a base observation are correlated and therefore should not be treated as statistically independent units. The original pair-level summaries are retained below because they describe the distribution of accepted perturbation pairs, but Stage 6C uses the original base observation as the statistical unit for uncertainty estimation.

Stage 6C first averaged each stability metric across all accepted perturbations belonging to the same base observation. It then computed the mean across these per-base means and obtained percentile-bootstrap 95% confidence intervals by resampling original base observations. The bootstrap used 10,000 repetitions and seed 2026. This analysis reused the recorded attribution comparisons and did not recompute SHAP values.

## 8.2. Local Stability at Epsilon 0.01

Table 12 reports the original perturbation-pair-level descriptive statistics. These summaries characterize a typical accepted perturbation pair but do not provide independent-sample uncertainty intervals.

Table 12: Perturbation-pair descriptive explanation stability at $\epsilon = 0 . 0 1$
<table><tr><td>Model</td><td>Accepted pairs</td><td>Base samples</td><td>Median Jaccard</td><td>Mean Jaccard</td><td>Median Spearman</td><td>Median cosine</td><td>Median norm.  $L _ { 1 }$ </td></tr><tr><td>Decision Tree</td><td>480</td><td>100/100</td><td>1.000</td><td>0.871156</td><td>0.985613</td><td>0.999818</td><td>0.036678</td></tr><tr><td>Random Forest</td><td>471</td><td>99/100</td><td>1.000</td><td>0.883834</td><td>0.994332</td><td>0.998687</td><td>0.044717</td></tr><tr><td>XGBoost</td><td>463</td><td>98/100</td><td>0.666667</td><td>0.781960</td><td>0.967406</td><td>0.996619</td><td>0.109284</td></tr></table>

Decision Tree and Random Forest exhibited very high local explanation stability. Both had a median Top-5 Jaccard similarity of 1.000, indicating complete median overlap among the five most highly attributed features. Random Forest produced the strongest attribution-rank stability, with a median Spearman correlation of 0.994332. Decision Tree had a median signed cosine similarity of 0.999818 and the smallest median normalized $L _ { 1 }$ change, 0.036678.

XGBoost retained high global rank and directional consistency, with median Spearman and cosine values of 0.967406 and 0.996619. It was nevertheless comparatively more sensitive at the local top-k and magnitude levels: median Top-5 Jaccard was 0.666667 and median normalized $L _ { 1 }$ change was 0.109284. This does not support the categorical statement that XGBoost explanations were unstable. Rather, under the tested perturbations, their leading-feature membership and attribution magnitudes were more sensitive than those of Decision Tree and Random Forest.

## 8.3. Base-Sample-Level Stability at Epsilon 0.010

Table 13 reports the Stage-6C mean of the per-base means. Random Forest exhibited the strongest overall base-sample-level stability at $\epsilon = 0 . 0 1 0$ combining the highest mean Top-5 Jaccard, Spearman, and signed-cosine values with the lowest mean normalized $L _ { 1 }$ drift. Decision Tree remained highly stable in top-feature membership and attribution rank. XGBoost retained high rank and directional agreement, but showed lower Top-5 overlap and greater normalized attribution drift.

The pair-level medians in Table 12 and the base-level means in Table 13 answer diferent questions. Pair-level medians describe the typical accepted perturbation pair, whereas base-level means give equal statistical weight to each original observation and expose observations with larger average drift. Their numerical diferences are therefore not contradictory.

Table 13: Base-sample-level explanation stability at $\epsilon = 0 . 0 1 0$ with percentile-bootstrap 95% confidence intervals. Each estimate is the mean of the per-base means.
<table><tr><td>Model</td><td>n</td><td>Mean Top-5 Jaccard</td><td>Mean Spearman</td><td>Mean signed cosine</td><td>Mean normalized  $L _ { 1 }$ </td></tr><tr><td></td><td></td><td>0.866442 0.885426</td><td>0.981194 0.984711</td><td>0.953836 Decision Tree 100 [0.845380, 0.886284] [0.978590, 0.983691] [0.939850, 0.966483] [0.129932, 0.196816] 0.970922</td><td>0.162103 0.096899</td></tr><tr><td>Random Forest</td><td>99</td><td></td><td></td><td>[0.863299, 0.906718] [0.981771, 0.987482] [0.962716, 0.978569] [0.081574, 0.113380]</td><td></td></tr><tr><td>XGBoost</td><td>98</td><td>0.783107</td><td>0.960863</td><td>0.954550 [0.762503, 0.803596] [0.956101, 0.965346] [0.945676, 0.962984] [0.169509, 0.210429]</td><td>0.189419</td></tr></table>

## 8.4. Class-Conditioned Explanation Stability

Table 14 stratifies the Stage-6C base-sample summaries by the original benign-versus-attack class. Across all three models, attack observations exhibited greater attribution-magnitude drift than benign observations. Attack observations also exhibited substantially lower signed-cosine similarity, particularly for Decision Tree and Random Forest. Rank and Top-5 behavior were model dependent, however, so these results do not support the claim that attack explanations were less stable on every metric.

TP/FN/TN case-type diagnostics were also inspected, but the falsenegative subgroups contained only three base observations per model and were therefore considered too small for inferential comparison.

## 8.5. Sensitivity to Perturbation Magnitude

The same protocol was repeated at $\epsilon \in \lbrace 0 . 0 0 5 , 0 . 0 1 0 , 0 . 0 2 0 \rbrace$ . Table 15 retains the original perturbation-pair descriptive summaries across these operating points.

Figure 3 visualizes the complementary perturbation-pair Top-5 overlap and normalized-magnitude responses across perturbation strengths.

At the perturbation-pair level, all models exhibited a monotonic degradation pattern as perturbation strength increased: mean Top-5 Jaccard, median Spearman, and median cosine similarity decreased, while normalized $L _ { 1 }$ drift increased. Cosine similarity declined only moderately, showing that signed directional agreement can remain high even as top-feature membership and attribution magnitude change. XGBoost median Top-5 Jaccard remained 0.6667 at all tested levels because Top-5 Jaccard takes discrete values; the decreasing mean Jaccard provided greater sensitivity to distributional change.

Table 14: Class-conditioned base-sample explanation stability at $\epsilon = 0 . 0 1 0$ . Bracketed values are percentile-bootstrap 95% confidence intervals.  
Panel A—Feature-set and rank stability
<table><tr><td>Model</td><td>Class</td><td>n</td><td>Mean Jaccard (95% CI)</td><td>Mean Spearman (95% CI)</td></tr><tr><td rowspan="2">Decision Tree</td><td>Attack</td><td>50</td><td>0.867000 [0.846667, 0.888000]</td><td>0.982396 [0.979386, 0.985166]</td></tr><tr><td>Benign</td><td>50</td><td>0.865885 [0.830378, 0.900282]</td><td>0.979993 [0.975611, 0.983950]</td></tr><tr><td rowspan="2">Random Forest</td><td>Attack</td><td>50</td><td>0.852095 [0.825429, 0.878857]</td><td>0.976826 [0.972201, 0.981037]</td></tr><tr><td>Benign</td><td>49</td><td>0.919436 [0.886199, 0.949433]</td><td>0.992757 [0.990812, 0.994481]</td></tr><tr><td rowspan="2">XGBoost</td><td>Attack</td><td>50</td><td>0.773651 [0.748633, 0.798605]</td><td>0.968088 [0.963037, 0.972425]</td></tr><tr><td>Benign</td><td>48</td><td>0.792956 [0.759623, 0.827185]</td><td>0.953336 [0.945549, 0.960726]</td></tr></table>

Panel B—Attribution-vector stability
<table><tr><td colspan="4"></td><td rowspan="2">Mean signed cosine Mean normalized  $L _ { 1 }$  (95% CI)</td></tr><tr><td>Model</td><td>Class</td><td>n</td><td>(95% CI)</td></tr><tr><td rowspan="2">Decision Tree</td><td>Attack</td><td>50</td><td>0.908699 [0.888549, 0.927694] [0.233248, 0.326965]</td><td>0.278686</td></tr><tr><td>Benign</td><td>50</td><td>0.998973</td><td>0.045520 [0.997981, 0.999560] [0.034810, 0.057681]</td></tr><tr><td rowspan="2">Random Forest Attack</td><td></td><td>50</td><td>0.945329</td><td>0.144477 [0.932882, 0.956561] [0.121431, 0.169037]</td></tr><tr><td>Benign</td><td>49</td><td>0.997038 [0.996163, 0.997851] [0.040989, 0.056242]</td><td>0.048350</td></tr><tr><td rowspan="2">XGBoost</td><td>Attack</td><td>50</td><td>0.927818</td><td>0.241949 [0.915846, 0.939559] [0.212975, 0.272373]</td></tr><tr><td>Benign</td><td>48</td><td>0.982395 [0.975728, 0.988317]</td><td>0.134700 [0.115206, 0.154728]</td></tr></table>

Table 15: Perturbation-pair descriptive explanation stability across perturbation strengths.
<table><tr><td>Model</td><td></td><td>Mean Jaccard Median Spearman</td><td></td><td>Median cosine Median norm.</td><td> $L _ { 1 }$ </td></tr><tr><td>Decision Tree</td><td>0.005</td><td>0.912587</td><td>0.993313</td><td>0.999967</td><td>0.015123</td></tr><tr><td rowspan="3"></td><td>0.010</td><td>0.871156</td><td>0.985613</td><td>0.999818</td><td>0.036678</td></tr><tr><td>0.020</td><td>0.827052</td><td>0.980547</td><td>0.999276</td><td>0.075716</td></tr><tr><td>Random Forest 0.005</td><td>0.913073</td><td>0.997368</td><td>0.999519</td><td>0.025085</td></tr><tr><td rowspan="4">XGBoost</td><td>0.010</td><td>0.883834</td><td>0.994332</td><td>0.998687</td><td>0.044717</td></tr><tr><td>0.020</td><td>0.857246</td><td>0.989676</td><td>0.996896</td><td>0.066850</td></tr><tr><td>0.005</td><td>0.813585</td><td>0.974896</td><td>0.998229</td><td>0.076566</td></tr><tr><td>0.010</td><td>0.781960</td><td>0.967406</td><td>0.996619</td><td>0.109284</td></tr><tr><td></td><td>0.020</td><td>0.747146</td><td>0.957283</td><td>0.994686</td><td>0.137975</td></tr></table>

![](images/b4a04401da64205f49699a58b47a041a05a4d40f430b5ba2ec4f0aee187c946c.jpg)

![](images/1ba382c997fd4dbfc941a643608a7c01f2092fc9d612286d120d9d413ca6076e.jpg)  
Figure 3: Perturbation-pair explanation stability as perturbation strength increases. The left panel reports mean Top-5 Jaccard similarity; the right panel reports median normalized $L _ { 1 }$ attribution drift. Marker shapes and line styles distinguish models independently of color.

The base-sample-level aggregation confirmed the same monotonic trend (Table 16). Mean Top-5 Jaccard decreased with perturbation strength for every model, while mean normalized $L _ { 1 }$ drift increased.

Table 16: Base-sample-level stability sensitivity across perturbation strengths. Values are means of the per-base means.
<table><tr><td>Model</td><td>€</td><td>Mean Top-5 Jaccard</td><td>Mean normalized  $L _ { 1 }$ </td></tr><tr><td>Decision Tree</td><td>0.005</td><td>0.912762</td><td>0.119848</td></tr><tr><td rowspan="5">Random Forest</td><td>0.010</td><td>0.866442</td><td>0.162103</td></tr><tr><td>0.020</td><td>0.822479</td><td>0.203172</td></tr><tr><td>0.005</td><td>0.912111</td><td>0.064236</td></tr><tr><td>0.010</td><td>0.885426</td><td>0.096899</td></tr><tr><td>0.020</td><td>0.858625</td><td>0.125522</td></tr><tr><td rowspan="3">XGBoost</td><td>0.005</td><td>0.817540</td><td>0.167672</td></tr><tr><td>0.010</td><td>0.783107</td><td>0.189419</td></tr><tr><td>0.020</td><td>0.748794</td><td>0.215619</td></tr></table>

Stages 6A and 6B reproduced the $\epsilon = 0 . 0 1 0$ operating point exactly at the base-sample level. For $\mathrm { T o p } { - } 5$ Jaccard, Spearman correlation, signed cosine similarity, and normalized $L _ { 1 }$ change, the mean and maximum absolute differences between the two stages were 0.0 for all three models. This agreement supports deterministic internal reproducibility of the stability pipeline.

## 9. Selective Explanation and Cost–Coverage Trade-ofs

Explain-all maximizes analytical coverage but may be unnecessary or computationally prohibitive. Selective prediction provides a broader methodological precedent for allocating automated actions according to confidence or risk [20]; selective explanation instead applies this resource-allocation principle to XAI invocation while leaving the detector’s class output unchanged. Stage 7A evaluated five policies. P0 explained every prediction. P1 explained predicted attacks only. P2 explained only uncertain predictions with $p ( \mathrm { a t t a c k } ) \in [ 0 . 4 0 , 0 . 6 0 ]$ . P3 explained predictions classified as attacks or falling in the uncertainty interval. P4 used a validation-calibrated risk trigger: every predicted attack was explained, together with predicted-benign cases whose attack probability exceeded a model-specific threshold.

P4 thresholds were derived from validation false negatives to target approximately 95% false-negative explanation coverage. They were 0.004588 for Decision Tree, 0.010637 for Random Forest, and 0.006504 for XGBoost. Thresholds were calibrated without test labels.

## 9.1. Selective Policies at the 95% Target

Table 17 reports the balanced-test results for the validation-calibrated P4 policy.

Table 17: Validation-calibrated selective explanation on the balanced test at the approximately 95% false-negative target.
<table><tr><td>Model</td><td>Threshold</td><td>Explained (%)</td><td>Saved (%)</td><td>True-attack coverage (%)</td><td>FN coverage (%)</td></tr><tr><td>Decision Tree</td><td>0.004588</td><td>84.7639</td><td>15.2361</td><td>99.8942</td><td>96.1978</td></tr><tr><td>Random Forest</td><td>0.010637</td><td>77.4214</td><td>22.5786</td><td>99.8483</td><td>95.0538</td></tr><tr><td>XGBoost</td><td>0.006504</td><td>76.5269</td><td>23.4731</td><td>99.8599</td><td>94.9459</td></tr></table>

On the balanced test, P1 reduced explanation workload by approximately 51%, but its false-negative explanation coverage was exactly zero because every missed attack was predicted benign. P2 saved more than 99.5% of workload but explained almost none of the true attacks, making it unsuitable as a standalone security policy. P3 recovered only a small fraction of false negatives, indicating that many missed attacks were not simply close to the 0.5 decision boundary. These outcomes motivate a risk-calibrated trigger rather than a narrow uncertainty-only policy.

P4 retained nearly complete true-attack coverage while reducing workload. XGBoost produced the largest balanced-test saving at the 95% target, 23.4731%, followed by Random Forest at 22.5786% and Decision Tree at 15.2361%. The achieved false-negative coverage varied slightly around the calibration target because thresholds were derived from validation data rather than tuned on test labels.

## 9.2. Prevalence Dependence

Selective-explanation savings were substantially smaller on the natural attack-dominated test distribution. At the P4 95% target, Decision Tree saved 1.7169% of explanation workload while covering 99.8838% of true attacks and 95.8812% of false negatives. Random Forest saved 2.5255%, with 99.8480% true-attack coverage and 95.1367% false-negative coverage. XGBoost saved

2.6071%, with 99.8624% true-attack coverage and 95.1115% false-negative coverage.

Selective-explanation savings are therefore prevalence-dependent. When attacks dominate the workload, a security policy that explains almost every attack necessarily retains a large explanation burden. The smaller natural-test saving is an operational property of the workload and policy objective rather than a failure of selective explanation.

## 9.3. Cost–Coverage Frontier

Stage 7B swept target validation false-negative explanation coverage over 50%, 70%, 80%, 90%, 95%, and 99%. For target C, the trigger threshold was the (1 − C)-quantile of validation false-negative attack probabilities. Test labels were not used for threshold calibration.

Figure 4 reports the balanced-test frontier produced by these validationcalibrated thresholds.

![](images/190b88ab10f63efda5198dfcfd288d277346a12f988e3a203d2b77a4169cdc6b.jpg)  
Figure 4: Balanced-test cost–coverage frontier using validation-calibrated false-negative risk thresholds. Marker shapes and line styles distinguish models independently of color.

Table 18 summarizes the principal 90% and 95% operating points.

At approximately 90% false-negative coverage on the balanced test, compute savings were 28.35% for Decision Tree, 31.17% for Random Forest, and 32.22% for XGBoost. At approximately 95% coverage, the corresponding savings were 15.24%, 22.58%, and 23.47%. All tested frontier points were non-dominated under the paired objectives of maximizing false-negative explanation coverage and compute saving. No single threshold is therefore universally optimal; the operating point depends on deployment risk tolerance and resource budget.

Table 18: Balanced-test cost–coverage frontier at the 90% and 95% validation targets.
<table><tr><td rowspan="2">Model</td><td colspan="2">90% target</td><td colspan="2">95% target</td></tr><tr><td>Test FN coverage (%)</td><td>Saved (%)</td><td>Test FN coverage (%)</td><td>Saved (%)</td></tr><tr><td>Decision Tree</td><td>90.1538</td><td>28.351263</td><td>96.1978</td><td>15.236097</td></tr><tr><td>Random Forest</td><td>90.2074</td><td>31.174891</td><td>95.0538</td><td>22.578633</td></tr><tr><td>XGBoost</td><td>89.8698</td><td>32.217316</td><td>94.9459</td><td>23.473057</td></tr></table>

The natural-test frontier was compressed by attack prevalence. At the 90% target, Decision Tree, Random Forest, and XGBoost saved 3.25%, 3.58%, and 3.66%, respectively. At the 95% target, savings were 1.72%, 2.53%, and 2.61%. The lower savings primarily reflect the requirement to explain nearly all attacks in a workload dominated by attacks.

Stage-7 full-test explanation times are throughput-based projections from the Stage-5A 5,000-sample operating point, not measured full-test runtimes. Under an explain-all host-runtime projection, Random Forest required approximately 120.30 h for the natural test, compared with approximately 0.253 h for XGBoost. These projected workloads emphasize architecture-dependent XAI cost but do not represent direct measurements on IoT hardware.

## 10. Discussion

## 10.1. Detection Accuracy Is Not an Adequate Deployment Criterion

The predictive experiments show why accuracy cannot determine deployment suitability in isolation. XGBoost, Random Forest, and Decision Tree produced closely grouped F1 scores, yet difered in false-positive behavior, serialized size, host training time, explanation cost, and local stability. Logistic Regression further showed that a threshold optimized for F1 under an attackdominated distribution can create an operationally undesirable false-positive rate. Model selection must therefore begin with predictive quality but cannot end there.

## 10.2. Explanation Cost Is Architecture-Dependent

The cost results expose a model-architecture efect that is not visible in detection metrics. At 5,000 samples, Random Forest TreeSHAP required 700.758801 s, whereas XGBoost required 1.471088 s. Random Forest explanations were highly stable, but their absolute cost and throughput were substantially less favorable. XGBoost explanations were dramatically cheaper and scaled more efectively. Decision Tree illustrates why overhead ratios cannot be interpreted alone: its 13,025.60× ratio was produced by a very small prediction time, while its absolute TreeSHAP time remained far below that of Random Forest.

## 10.3. Stability Is Multi-Dimensional

Stability depends on which aspect of an explanation is examined. Top-5 Jaccard measures feature-set membership, Spearman correlation measures rank agreement, signed cosine measures directional alignment, and normalized $L _ { 1 }$ captures magnitude drift. XGBoost demonstrates why a single measure is insuficient: cosine similarity remained 0.996619 at $\epsilon = 0 . 0 1$ , even though median Top-5 Jaccard was 0.666667 and normalized $L _ { 1 }$ drift was 0.109284. High directional agreement can therefore coexist with greater turnover among the most prominent features and larger changes in attribution magnitude.

Base-sample aggregation prevents observations with more accepted perturbations from receiving greater statistical weight and identifies Random Forest as the strongest overall base-level stability profile at $\epsilon = 0 . 0 1 0$ . The class-conditioned analysis adds a separate qualification: attack observations had greater magnitude drift and lower signed-cosine agreement across all three models, but Top-5 membership and rank diferences varied by architecture. Stability comparisons must therefore specify the metric, statistical unit, perturbation strength, and trafic class.

## 10.4. Selective Explanation as Resource Allocation

Selective explanation reframes XAI as a computational security resource rather than an automatic post-processing step. Explain-all provides maximum coverage at maximum cost. Explaining predicted attacks alone yields substantial savings on a balanced workload but never explains false negatives. Uncertainty-only invocation is inexpensive but provides inadequate attack coverage, while an attack-or-uncertain policy recovers only a limited fraction of missed attacks. Validation-calibrated risk creates a tunable compromise between false-negative explanation coverage and computational load without using test labels for threshold selection.

## 10.5. Practical Deployment Implications

The host-based measurements suggest three operational profiles that warrant subsequent device-level evaluation. For a lightweight gateway, Decision Tree may be attractive because its serialized model is compact and its absolute TreeSHAP cost is low relative to Random Forest, while its predictive performance remains competitive. This profile is a hypothesis for constrainedhardware evaluation, not evidence that the reported host timings transfer directly to an IoT gateway.

In a false-alarm-sensitive environment, Random Forest ofers the lowest observed false-positive rate, but its explanation throughput makes an explainall policy dificult to justify at high event volumes. Selective invocation is particularly important for this architecture because each explained case consumes substantially more host time. For high-throughput explainable monitoring, XGBoost ofers the strongest combined predictive and explanationcost profile on the tested host. The appropriate profile still depends on the deployed trafic prevalence, explanation budget, acceptable false-negative coverage, and the performance of the target hardware.

## 10.6. Model-Level Trade-of

Table 19 synthesizes the four evaluated dimensions: predictive performance, explanation cost, stability, and selective explanation.

Within the evaluated host-based configuration and CICIoT2023 binary task, XGBoost presented the most favorable combined predictive/explanationcost profile. This is not a universal model ranking. Random Forest ofered the lowest false-positive rate and strongest rank stability, while Decision Tree remained competitive with low absolute explanation cost and compact serialized size. The appropriate choice depends on the relative cost assigned to missed attacks, false alarms, explanatory throughput, local attribution sensitivity, and available resources.

Figure 5 summarizes the progression from detection performance to operational utility.

## 11. Limitations and Threats to Validity

Dataset scope. The primary empirical conclusions are based on CI-CIoT2023 and binary attack-versus-benign classification. The retained original labels and attack families support diagnostics, but the reported models were not evaluated as multiclass family classifiers.

Table 19: Cross-dimensional synthesis of the evaluated models.
<table><tr><td>Model</td><td>Predictive profile</td><td>Explanation cost</td><td>Stability</td><td>Selective-XAI implications</td></tr><tr><td>Decision Tree</td><td>Competitive detection with a compact serialized model</td><td>Low absolute TreeSHAP cost, despite a large ratio to very fast prediction</td><td>Very high feature-set, rank, directional, and magnitude</td><td>Compact baseline for further resource- constrained hardware</td></tr><tr><td>Random Forest</td><td>Strong detection and the lowest false-positive rate among the tree models</td><td>Extremely high TreeSHAP cost and lowest explanation throughput</td><td>Highest rank stability and high top-feature overlap</td><td>Selective invocation is especially important because each</td></tr><tr><td>XGBoost</td><td>Strongest overall predictive profile within</td><td>Lowest TreeSHAP cost at scale and highest explanation</td><td>High rank and directional consistency, with greater local top-k and</td><td>costly Most favorable combined predictive and explanation- cost profile on</td></tr></table>

![](images/5c35704eaa178218bcd4e4570a61ae50a4138c75fbad16e235b432f6eb81ebaf.jpg)  
Figure 5: Overall operational trade-of: predictive performance is the first stage, rather than the complete criterion, in evaluating resource-aware explainable intrusion detection.

External validity. No external-dataset validation has yet been conducted. The results do not establish cross-dataset generalization to Edge-IIoTset or another trafic environment.

Host-based timing. Training, prediction, and TreeSHAP measurements were obtained on the experimental host, not on a Raspberry Pi, production IoT gateway, or other constrained device. The measurements therefore do not demonstrate direct edge-device feasibility and are hardware dependent.

Memory measurement. Stage-5 memory observations used before/after process resident-set-size diferences. This method can miss transient allocation peaks and must not be interpreted as true peak-memory measurement.

TreeSHAP scope. Explanation-cost conclusions concern TreeSHAP for the evaluated Decision Tree, Random Forest, and XGBoost models. They should not be generalized automatically to LIME, model-agnostic SHAP variants, counterfactual methods, or other XAI procedures.

Perturbation validity. Stage-6 perturbations were restricted to selected continuous trafic statistics and bounded by validation-derived percentiles. They provide a controlled local sensitivity test but do not constitute complete protocol-level or causal trafic generation.

Interpretation of stability. Stable SHAP attributions do not prove that model reasoning is causal, semantically correct, or operationally appropriate. Stability measures repeatability under the tested neighborhood and explainer configuration.

Statistical uncertainty. Predictive metrics are point estimates from fixed fitted models and deterministic test partitions; repeated model training and formal significance testing were not performed. Accordingly, small diferences among predictive models are interpreted descriptively rather than as evidence of statistical superiority. For the Stage-6 stability analysis, dependence among multiple perturbations of the same observation was addressed through base-sample aggregation and 10,000-fold bootstrap confidence intervals across original base observations.

Projected selective-explanation cost. Full-dataset Stage-7 explanation times are throughput-based estimates derived from Stage-5A measurements at the 5,000-sample operating point. They are host-runtime projections rather than directly measured full-test runtimes.

Prevalence dependence. Selective-policy utility changed substantially between the balanced and natural test distributions. Savings measured under one prevalence should not be transferred to another workload without recalculation.

Operational utility. Utility in this study refers to computational workload, security-case coverage, and cost–coverage trade-ofs under selective explanation. No human-subject study was performed, and the results do not establish improvements in analyst trust, decision quality, or response efectiveness.

Binary focus. Multiclass attack-family detection may produce diferent predictive errors, explanation costs, local stability patterns, and selectionpolicy trade-ofs.

## 12. Conclusion

This study evaluated explainable IoT intrusion detection across predictive efectiveness, explanation cost, explanation stability, and selective invocation. For RQ1, the tree-based models achieved strong binary detection performance on the leakage-safe CICIoT2023 corpus. XGBoost produced the strongest overall predictive profile under the reported metrics, while Random Forest produced particularly low false-positive rates. These results also confirmed that accuracy alone obscures operationally important diferences in recall and benign false alarms.

For RQ2, TreeSHAP cost difered by orders of magnitude across model architectures. At 5,000 samples, Random Forest required approximately 700.8 s, compared with approximately 5.64 s for Decision Tree and 1.47 s for XGBoost. Prediction speed alone did not determine explanation speed, and relative overhead ratios required interpretation alongside absolute latency and throughput.

For RQ3, explanations were generally stable under the tested predictionpreserving perturbations, but stability was model dependent and multidimensional. Base-sample aggregation identified Random Forest as the strongest overall stability profile at $\epsilon = 0 . 0 1 0$ , while Decision Tree also exhibited strong top-feature and rank stability. XGBoost maintained high directional and rank consistency while showing greater top-k membership turnover and magnitude sensitivity. Stability declined systematically as perturbation strength increased. Attack observations exhibited greater magnitude drift and lower signed-cosine agreement than benign observations across all three models, although rank and Top-5 diferences remained architecture dependent.

For RQ4, validation-calibrated selective explanation reduced workload while retaining security-relevant coverage. On the balanced test, approximately 90% false-negative explanation coverage permitted savings of roughly 28–32%, depending on model. At approximately 95% coverage, savings ranged from roughly 15–23%. Benefits were much smaller on the attack-heavy natural test because a policy that explains nearly every attack must retain most of the workload.

The operational value of explainable IoT intrusion detection cannot be determined from detection accuracy or explanation availability alone. Predictive quality, architecture-dependent explanation cost, local stability, workload prevalence, and security-relevant coverage must be evaluated jointly. The present results establish these trade-ofs for the evaluated host configuration and CICIoT2023 binary task without claiming universal model superiority or direct constrained-device feasibility.

## Funding

This research did not receive any specific grant from funding agencies in the public, commercial, or not-for-profit sectors.

## CRediT authorship contribution statement

Abdurrahman Tolay: Conceptualization, Methodology, Software, Validation, Formal analysis, Investigation, Data curation, Visualization, Writing – original draft, Writing – review & editing.

## Declaration of competing interest

The author declares no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

## Data availability

The CICIoT2023 dataset used in this study is publicly available from its original source and is cited in the manuscript. The preprocessing, corpusaudit, predictive-evaluation, explanation-cost, stability-analysis, and selectiveexplanation scripts, together with trained model artifacts and derived experimental outputs, are publicly available through the reproducibility repository at GitHub (https://github.com/AbdurrahmanTolay/resource-aware-iot-xai) and its archived Zenodo release (DOI: 10.5281/zenodo.21879038).

Declaration of generative AI and AI-assisted technologies in the manuscript preparation process

During the preparation of this manuscript, the author used Prism for language checking, editorial refinement, and organization of the LaTeX manuscript. The research design, experimental methodology, implementation, data processing, analyses, numerical results, scientific interpretation, and final content were developed, verified, and approved by the author, who takes full responsibility for the work.

## References

[1] A. Khraisat, A. Alazab, A critical review of intrusion detection systems in the internet of things: Techniques, deployment strategy, validation strategy, attacks, public datasets and challenges, Cybersecurity 4 (2021) 18. doi:10.1186/s42400-021-00077-7.

[2] N. Moustafa, N. Koroniotis, M. Keshk, A. Y. Zomaya, Z. Tari, Explainable intrusion detection for cyber defences in the internet of things: Opportunities and solutions, IEEE Communications Surveys & Tutorials 25 (3) (2023) 1775–1807. doi:10.1109/COMST.2023.3280465.

[3] M. T. Ribeiro, S. Singh, C. Guestrin, “Why Should I Trust You?”: Explaining the predictions of any classifier, in: Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 2016, pp. 1135–1144. doi:10.1145/2939672.2939778.

[4] S. M. Lundberg, S.-I. Lee, A unified approach to interpreting model predictions, in: Advances in Neural Information Processing Systems, Vol. 30, 2017, pp. 4768–4777.

[5] S. M. Lundberg, G. Erion, H. Chen, A. DeGrave, J. M. Prutkin, B. Nair, R. Katz, J. Himmelfarb, N. Bansal, S.-I. Lee, From local explanations to global understanding with explainable AI for trees, Nature Machine Intelligence 2 (2020) 56–67. doi:10.1038/s42256-019-0138-9.

[6] A. Ghorbani, A. Abid, J. Zou, Interpretation of neural networks is fragile, in: Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 33, 2019, pp. 3681–3688. doi:10.1609/aaai.v33i01.33013681.

[7] C.-K. Yeh, C.-Y. Hsieh, A. Suggala, D. I. Inouye, P. K. Ravikumar, On the (in)fidelity and sensitivity of explanations, in: Advances in Neural Information Processing Systems, Vol. 32, 2019.

[8] D. Slack, S. Hilgard, E. Jia, S. Singh, H. Lakkaraju, Fooling LIME and SHAP: Adversarial attacks on post hoc explanation methods, in: Proceedings of the AAAI/ACM Conference on AI, Ethics, and Society, 2020, pp. 180–186. doi:10.1145/3375627.3375830.

[9] E. C. P. Neto, S. Dadkhah, R. Ferreira, A. Zohourian, R. Lu, A. A. Ghorbani, CICIoT2023: A real-time dataset and benchmark for large-scale attacks in IoT environment, Sensors 23 (13) (2023) 5941. doi:10.3390/s23135941.

[10] A. Khraisat, I. Gondal, P. Vamplew, J. Kamruzzaman, Survey of intrusion detection systems: Techniques, datasets and challenges, Cybersecurity 2 (2019) 20. doi:10.1186/s42400-019-0038-7.

[11] Z. Abou El Houda, B. Brik, L. Khoukhi, “Why Should I Trust Your IDS?”: An explainable deep learning framework for intrusion detection systems in internet of things networks, IEEE Open Journal of the Communications Society 3 (2022) 1164–1176. doi:10.1109/OJCOMS.2022.3188750.

[12] M. Keshk, N. Koroniotis, N. Pham, N. Moustafa, B. Turnbull, A. Y. Zomaya, An explainable deep learning-enabled intrusion detection framework in IoT networks, Information Sciences 639 (2023) 119000. doi:10.1016/j.ins.2023.119000.

[13] A. Barredo Arrieta, N. Díaz-Rodríguez, J. Del Ser, A. Bennetot, S. Tabik, A. Barbado, S. García, S. Gil-López, D. Molina, R. Benjamins, R. Chatila, F. Herrera, Explainable artificial intelligence (XAI): Concepts, taxonomies, opportunities and challenges toward responsible AI, Information Fusion 58 (2020) 82–115. doi:10.1016/j.infus.2019.12.012.

[14] T.-T.-H. Le, H. Kim, H. Kang, H. Kim, Classification and explanation for intrusion detection system based on ensemble trees and SHAP method, Sensors 22 (3) (2022) 1154. doi:10.3390/s22031154.

[15] S. Patil, V. Varadarajan, S. M. Mazhar, A. Sahibzada, N. Ahmed, O. Sinha, S. Kumar, K. Shaw, K. Kotecha, Explainable artificial intelligence for intrusion detection system, Electronics 11 (19) (2022) 3079. doi:10.3390/electronics11193079.

[16] O. Arreche, T. Guntur, M. Abdallah, XAI-IDS: Toward proposing an explainable artificial intelligence framework for enhancing network intrusion detection systems, Applied Sciences 14 (10) (2024) 4170. doi:10.3390/app14104170.

[17] Y. Wang, M. A. Azad, M. Zafar, A. Gul, Enhancing AI transparency in IoT intrusion detection using explainable AI techniques, Internet of Things 33 (2025) 101714. doi:10.1016/j.iot.2025.101714.

[18] O. Arreche, T. R. Guntur, J. W. Roberts, M. Abdallah, E-XAI: Evaluating black-box explainable AI frameworks for network intrusion detection, IEEE Access 12 (2024) 23954–23988. doi:10.1109/ACCESS.2024.3365140.

[19] J. Munilla, R. M. Khammas, Evaluation of explainable artificial intelligence in IoT intrusion detection systems under DeepFool adversarial conditions, Sensors 26 (10) (2026) 2924. doi:10.3390/s26102924.

[20] Y. Geifman, R. El-Yaniv, Selective classification for deep neural networks, in: Advances in Neural Information Processing Systems, Vol. 30, 2017, pp. 4878–4887.

[21] M. Tavallaee, E. Bagheri, W. Lu, A. A. Ghorbani, A detailed analysis of the KDD CUP 99 data set, in: 2009 IEEE Symposium on Computational Intelligence for Security and Defense Applications, 2009, pp. 1–6. doi:10.1109/CISDA.2009.5356528.

[22] F. Pedregosa, G. Varoquaux, A. Gramfort, V. Michel, B. Thirion, O. Grisel, M. Blondel, P. Prettenhofer, R. Weiss, V. Dubourg, J. Vanderplas, A. Passos, D. Cournapeau, M. Brucher, M. Perrot, E. Duchesnay, Scikit-learn: Machine learning in python, Journal of Machine Learning Research 12 (2011) 2825–2830.

[23] L. Breiman, Random forests, Machine Learning 45 (2001) 5–32. doi:10.1023/A:1010933404324.

[24] T. Chen, C. Guestrin, XGBoost: A scalable tree boosting system, in: Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 2016, pp. 785–794. doi:10.1145/2939672.2939785.