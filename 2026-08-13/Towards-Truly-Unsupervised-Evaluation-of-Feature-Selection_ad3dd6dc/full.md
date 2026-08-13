# Towards Truly Unsupervised Evaluation of Feature Selection

Hafiz Saud Arshad<sup>1[0009−0003−2107−5589]</sup>, Muhammad Rajabinasab<sup>1[0009−0006−7045−3998]</sup>, and Arthur Zimek<sup>1[0000−0001−7713−4208]</sup>

University of Southern Denmark, Odense, Denmark {saudarshad,rajabinasab,zimek}@imada.sdu.dk

Abstract. Feature selection is one of the most important and fundamental tasks in data mining, tackled by a family of methods with an established set of evaluation techniques to measure the quality of a specific method. Most of the methods commonly used for the unsupervised evaluation of feature selection algorithms sufer from critical design flaws which question their unsupervised nature. In this paper, we provide a critical discussion on the established allegedly unsupervised evaluation techniques, and shed light on the reasons why they are not truly unsupervised but, at best, supervised evaluation under an unsupervised downstream task. We also propose a novel, truly unsupervised evaluation framework to measure the quality of the feature selection algorithms without any form of information about the labels. The proposed framework utilizes unsupervised Principal Component Analysis, and optimal transport to measure the quality of the feature selection methods in a truly unsupervised manner.

Keywords: Unsupervised Evaluation · Feature Selection · Dimensionality Reduction · Subspace Similarity

## 1 Introduction

The curse of dimensionality is one of the fundamental challenges in machine learning and data mining. As dimensionality increases, both computational complexity and the dificulty of downstream tasks grow considerably. High dimensionality negatively impacts a wide range of applications, including but not limited to Nearest Neighbor search [17,37], classification [1], clustering [27], and outlier detection [2,36,33], among others. Dimensionality reduction consists of many diferent families of techniques, ranging from subspace learning [33] to feature extraction [3]. Among dimensionality reduction methods, Feature Selection remains particularly important because it preserves the semantics of the original dimensions and therefore maintains interpretability and explainability.

Feature selection is a fundamental machine learning and data mining task. Given a set of initial features $F ~ = ~ \{ f _ { 1 } , f _ { 2 } , . . . , f _ { n } \}$ , the problem of feature selection is defined as finding the optimal subset $S ^ { * } \subseteq F$ that maximizes a criterion function J(S) (e.g., the predictive power or the information content of the subset). Formally, this can be expressed as the optimization problem $S ^ { * } = \arg \operatorname* { m a x } _ { S \subseteq X } J ( S )$ . It can be set to be subject to the constraint $| S | = k < n$ or with a penalty term that encourages sparsity. The objective is to identify the most relevant features while eliminating redundant or irrelevant dimensions, thereby improving model generalization and reducing computational costs [13,21]. Feature selection can be conducted in both supervised and unsupervised settings. In supervised feature selection, information about class labels is used to facilitate the identification of informative features. In the unsupervised setting, the feature selection process is based on the intrinsic information of the features themselves (e.g., correlation and variance) [8].

Similar to dimensionality reduction, feature selection addresses the curse of dimensionality, the exponential expansion of the data space as the number of features increases, causing data to become so sparse that statistical distance and density metrics lose their meaningful distinction, leading to a degradation in the performance of many downstream machine learning and data mining tasks [12,43]. Unlike dimensionality reduction methods, feature selection ofers an important advantage: keeping the explainability properties of the original feature space while reducing the dimensionality. This is why, despite the existence of many successful methods for dimensionality reduction such as Principal Component Analysis (PCA), Autoencoders [7], t-distributed Stochastic Neighbor Embedding (t-SNE) [22], and Uniform Manifold Approximation and Projection (UMAP) [23], feature selection methods are still important for tackling the curse of dimensionality [13].

There are established evaluation techniques for measuring the quality of feature selection algorithms. In most cases, a combination of supervised and unsupervised evaluation is used which measures the quality of a feature selection method with respect to the performance in conducting a downstream task. There is however an important problem concerned with the unsupervised evaluation part, and that is the fact that the process is done by conducting clustering (usually using k-means) on the data representation using selected features, and checking the clustering accuracy with respect to the ground-truth labels. This is clearly problematic, as using any information about the labels is in contrast with the alleged unsupervised nature which we expect from the evaluation, rendering the current established so-called unsupervised way of evaluation, at best, as supervised evaluation under an unsupervised downstream task. Even not using the labels, and just using the number of distinct classes to select the number of clusters, should still be considered a violation of the unsupervised setting. Unsupervised feature selection methods require a truly unsupervised evaluation technique, and in many practical scenarios, ground-truth labels might actually not be available.

There are other evaluation metrics and methods proposed for the evaluation of feature selection algorithms. These techniques ofer ways of evaluating the entire feature selection process, instead of a single point [32], evaluating the amount of dimensionality reduction and the quality of feature selection at once [24], or the evaluation of the stability of feature selection algorithms [26,32], or of modelagnostic evaluation [34]. Only the last one can be considered an unsupervised metric, as it does not use the labels or any external information about the labels in the calculation of the metric.

In this paper, we propose a novel method for the truly unsupervised evaluation of feature selection algorithms, utilizing dimensionality reduction and inter-dataset similarity to measure the quality of the feature selection process. In this evaluation method, we take advantage of PCA as an informative unsupervised dimensionality reduction technique. We evaluate the performance of a feature selection algorithm A in selecting f features by calculating the interdataset similarity metric IDS between the subsets constituted by selecting f features using algorithm A, and the first f principal components. Inter-dataset similarity metrics measure the distributional similarities or diferences between two datasets. There are various inter-dataset similarity measures [34], especially in the field of optimal transport (OT) [29]. Metrics such as Earth Mover’s Distance (EMD) [39], Wasserstein Distance [41], Sliced Wasserstein Distance [25], and Sinkhorn [9] can give us useful insights on how similar two datasets are. Measuring the inter-dataset similarity between the two aforementioned subsets of the data allows us to rank the feature selection algorithms based on how closely to PCA they perform. This process hence favors a feature selection algorithm yielding close performance figures to PCA without compromising the explainability as a notion for unsupervised evaluation.

The remainder of this paper is structured as follows: In Section 2, we review the existing measures for evaluating feature selection algorithms. In Section 3, we showcase a detailed description of the proposed unsupervised evaluation framework. In Section 4, we present experimental results. In Section 5, we conclude the paper by presenting a summary, highlighting limitations, and outlining possible directions for future research.

## 2 Related Work

Feature selection algorithms are primarily evaluated based on their ability to enhance (or maintain) the performance of downstream machine learning tasks while simultaneously reducing the dimensionality. Evaluation methods for feature selection can be divided into two main categories: (i) performance assessment on downstream supervised or unsupervised tasks, typically using metrics computed for a fixed number of selected features, and (ii) measures that capture broader properties such as overall performance trends, model-agnostic quality, fitness gain, or stability.

## 2.1 Evaluation Based on Downstream Tasks

The most common evaluation approach involves measuring how feature selection afects the quality of a subsequent machine learning task, most frequently classification (supervised) or clustering (unsupervised). These evaluations are usually point-based, i.e., they report performance achieved with a specific number of selected features.

Supervised Evaluation: In supervised settings, the efectiveness of a feature selection method is typically quantified by training one or more classifiers on the reduced feature set and calculating standard classification performance measures.

Accuracy (ACC): Accuracy is defined as the proportion of correctly classified instances:

$$
\mathrm { A C C } = { \frac { \mathrm { T P } + \mathrm { T N } } { \mathrm { T P } + \mathrm { T N } + \mathrm { F P } + \mathrm { F N } } }\tag{1}
$$

where TP, TN, FP, and FN denote the number of true positives, true negatives, false positives, and false negatives, respectively.

Area Under the ROC Curve (AUC): The area under the receiver operating characteristic curve quantifies a classifier’s ability to discriminate between classes across all decision thresholds:

$$
\mathrm { A U C } = \int _ { 0 } ^ { 1 } \mathrm { T P R } ( \mathrm { F P R } ) d \mathrm { F P R }\tag{2}
$$

where TPR is the true positive rate and FPR is the false positive rate. Values of AUC close to 1 indicate excellent discriminative power, while $\mathrm { A U C } = 0 . 5$ corresponds to random guessing.

Supervised metrics are used widely for the evaluation of feature selection methods. However, their values strongly depend on the choice of classifier, hyperparameter settings, and the dataset characteristics. They can be deceptive, as classifiers might still perform reasonably well even on a poorly (or randomly) selected subset of features [31].

Unsupervised Evaluation: In the so-called unsupervised evaluation of feature selection, performance is frequently evaluated by applying a clustering algorithm (usually k-means) on the selected features and comparing the resulting clustering labels against ground-truth class labels using external validation measures, i.e., these evaluation measures actually are not unsupervised as such, but only use an unsupervised downstream task while employing supervised evaluation.

Clustering Accuracy (CLSACC) Clustering accuracy measures the agreement between predicted cluster labels and true class labels after finding an optimal label correspondence:

$$
{ \mathrm { C L S A C C } } = { \frac { 1 } { n } } \sum _ { i = 1 } ^ { n } \delta { \bigl ( } c _ { i } , \operatorname* { m a p } ( g _ { i } ) { \bigr ) }\tag{3}
$$

where n is the number of instances, $c _ { i }$ and $g _ { i }$ are the predicted cluster and true class labels for instance $i ,$ respectively, map(·) is the optimal assignment obtained via the Hungarian method [20], and $\delta ( \cdot , \cdot )$ is the Kronecker delta function [4].

Normalized Mutual Information (NMI): Normalized mutual information quantifies the shared information between the predicted clustering $C$ and the true partitioning indicated by the ground-truth labels $L ,$ normalized by their entropies:

$$
\mathrm { N M I } = \frac { \mathrm { M I } ( L , C ) } { \operatorname* { m a x } \bigl ( \mathrm { E n t r o p y } ( L ) , \mathrm { E n t r o p y } ( C ) \bigr ) }\tag{4}
$$

where $\operatorname { M I } ( \cdot , \cdot )$ denotes mutual information. The measure ranges from 0 (independent) to 1 (perfect agreement).

These external clustering metrics, while informative, still rely on groundtruth labels, which is in contrast with the unsupervised nature of the downstream task. If we aim to apply an unsupervised feature selection algorithm on a dataset which does not have any labels, this would be rendered inefective as much as the supervised evaluation.

## 2.2 Specialized Evaluation Measures

Several specialized metrics have been proposed to overcome limitations of singlepoint and downstream task-dependent evaluations by providing more holistic, model-agnostic, or robustness-oriented insights.

FSDEM Score: Feature Selection Dynamic Evaluation Metric (FSDEM) [32] aggregates performance across varying numbers of selected features. Let d be the total number of features, $P ( f )$ be the value of a downstream performance measure obtained using exactly f selected features, and $g ( x )$ be a continuous function approximated based on the observed pairs $\{ ( f , P ( f ) ) \} _ { f = 1 } ^ { d }$ . The FSDEM score is then defined as the normalized area under this curve over a chosen interval [a, b]:

$$
\mathrm { F S D E M } = { \frac { 1 } { b - a } } \int _ { a } ^ { b } g ( x ) d x .\tag{5}
$$

When the underlying performance measure $P$ is bounded in $[ 0 , 1 ] ,$ , so is FSDEM, with values closer to 1 indicating consistently good performance. FSDEM allows approximating the overall performance of the entire feature selection process based on a number of observations, leading to a more holistic overview compared to the common single-point evaluation.

Average Angle Diference (AAD): The Average Angle Diference [34], also formally denoted as ∆θ, provides a model-agnostic assessment of feature selection quality by quantifying the average change in the direction of the eigenvector corresponding to the first principal component when non-selected features are removed. Let F be the set of selected features and $F ^ { C }$ its complement. For each feature $f \in F ^ { C }$ , construct the modified dataset $\mathbf { D } _ { f ^ { 0 } }$ by setting all values of feature f to zero. The AAD is then calculated as:

$$
A A D = \frac { 1 } { | F ^ { C } | } \sum _ { f \in F ^ { C } } \varDelta \theta \big ( \mathbf { D } , \mathbf { D } _ { f ^ { 0 } } \big ) ,\tag{6}
$$

![](images/4bad5ae188bb1f64546f9c1cb0e56827ea4671dfffc915ca0d3e91c07c8bcc99.jpg)  
Fig. 1: Proposed method for truly unsupervised evaluation of feature selection.

where ∆θ denotes the angle diference between the eigenvectors corresponding to the first principal components. Smaller values of ∆θ (closer to 0) indicate that the removed features were largely redundant or uninformative, hence afecting the first and most important principal component the least.

## 3 Unsupervised Evaluation of Feature Selection

The core idea behind our framework is to decouple the evaluation from the original ground-truth labels. The overall structure of the method is presented in Fig. 1. The method operates as follows: We apply PCA for efective dimensionality reduction and select top f principal components. To evaluate the result of some feature selection algorithm, selecting f features, we calculate an interdataset similarity metric IDS for the two representations of the data resulting from PCA and from some feature selection method. The similarity between the reduced datasets with f selected features and the PCA-based representation with f components yields the final evaluation.

The objective is to identify feature selection algorithms that select features in a way that they are as close as possible to the representation obtained from PCA. The better a subset preserves the structure of the PCA representation, the higher it ranks.

The proposed framework constitutes a truly unsupervised approach for evaluating the quality of feature selection algorithms. The use of PCA as a baseline allows us to favor feature selection methods which select features which best resemble the principal components, aiming for a similar downstream performance figure while maintaining the explainability. The unsupervised nature of the proposed framework addresses the existing gap in the evaluation of feature selection algorithms by facilitating the truly unsupervised evaluation of the feature selection methods.

While we focus on the implementation of the framework using PCA, other dimensionality reduction methods could be used as well, depending on the desired bias of the quality measure. We experiment with a selection of diferent dataset similarity measures. Earth Mover’s Distance (OT\_EMD2) [5] measures the minimum cost of transforming one distribution into another by moving probability mass. Entropic-Regularized Sinkhorn Distance (OT\_SINKHORN2) [10] is a GPU-friendly approximation of EMD. Gromov-Wasserstein Distance (OT\_GW2) [30] compares distributions living in incomparable metric spaces by matching their internal pairwise geometry rather than point-to-point correspondences. And finally, Sliced Wasserstein Distance (OT\_Sliced\_SW) [6] approximates the Wasserstein distance by averaging one-dimensional projections of the distributions, yielding a scalable but lower-resolution similarity estimate.

As all the values reflected by optimal transport are a measure of distance or dissimilarity, we transform them into a similarity measure:

$$
\mathrm { O T _ { \mathrm { s i m } } = \frac { 1 } { O T } \quad \mathrm { i f ~ O T \neq 0 ~ e l s e ~ O T = + \infty ~ } }\tag{7}
$$

where OT is the optimal transport cost. As the distances are theoretically not bounded, we use +∞ as an indicator of maximum possible similarity which only happens when the distributions are exactly the same (e.g., passing the same dataset twice).

## 4 Experiments and Analyses

We perform our experiments using the FSEVAL benchmarking suite [38], a specialized framework designed for the extensive and comprehensive evaluation of feature selection algorithms. Below, we list the experimental setup, results and analysis.

## 4.1 Experimental Setup

The benchmark consists of 8 publicly available high-dimensional datasets commonly used in the unsupervised feature selection literature [21]. This collection spans biomedical data and other high-dimensional benchmarks, representing a variety of applications. These datasets include COIL20, Isolet, ORL, lung, lung\_discrete, warpAR10P, warpPIE10P, and Yale.

Unsupervised Feature Selection Methods Among the classical approaches, variance-based [15] and correlation-based methods [14] operate directly on feature statistics without any learning component. Laplacian Score (LS) [16] is a forward-filter method that operates in an unsupervised manner as part of the broader spectral feature selection framework [42], with computational complexity of $\bar { C _ { \mathrm { L S } } } = d n ^ { 2 }$ , where d denotes the number of features and n the number of samples.

On the more recent end, Subspace Clustering-based Feature Selection (SCFS) [28] integrates subspace learning, clustering, and sparse learning via a selfexpressive model combined with regularized regression to sparsely capture featurecluster correlations. The Variance-Covariance Subspace Distance approach (VCS-DFS) leverages inter-feature correlations by identifying feature subsets whose variance-covariance matrices exhibit minimal norm properties [19].

We evaluate five feature selection methods and augment them with the Random baseline [31] that assigns uniform random importance scores to study how the proposed metrics are able to discriminate the random behavior from other feature selection approaches. Hyperparameters follow the default or suggested values, e.g., Laplacian Score uses k = 5 for the nearest neighbor graph, VCSDFS uses $\rho = 0 . 5$ with 30 iterations, SCFS uses $\alpha = \beta = 1 . 0$

Evaluation Measures On the evaluation end, we compare nine feature selection evaluation metrics, organized into four groups:

– Supervised: These metrics use explicit supervisory knowledge e.g., groundtruth class labels. We use Classification Accuracy (ACC) and Area under the ROC Curve (AUC), computed with a Random Forest classifier (100 trees) under the stratified 5-fold cross validation, average over 5 repetitions (diferent splits).

– Pseudo-Unsupervised: These metrics are based on clustering as an unsupervised downstream task, but are still measured with respect to the groundtruth labels, including Clustering Accuracy (CLSACC) and Normalized Mutual Information (NMI), obtained from k-means with the number of clusters set to the number of ground-truth classes, averaged over 10 diferent random initializations.

– Model Agnostic: The Average Angle Diference (AAD) [32] does not use any label information and operates on the average change in the angle of the eigenvector corresponding to the first principle component caused by the removal of unselected features.

– Proposed (Truly Unsupervised): We create four instantiations of our framework, difering in the choice of the inter-dataset similarity metric while comparing the optimal transport between the subspace of selected features against the PCA-reduced representation. As discussed in Section 3, we are using the Earth Mover’s Distance (OT\_EMD2) [5], Entropic-Regularized Sinkhorn Distance (OT\_SINKHORN2) [10], Gromov-Wasserstein Distance (OT\_GW2) [30], and Sliced Wasserstein Distance (OT\_SLICED\_SW) [6].

Following the dynamic-evaluation philosophy of FSDEM [32], we assess each method at 10 feature fractions in low-percentage regime, 0-5 Percent with p ∈ {0.005, 0.010, 0.015, . . . , 0.05}, capturing the low-feature setting most relevant to high-dimensional data.

Rankings, Repetitions, and Selection For each (dataset, feature selection method, percentage) configuration, we proceed as follows. The feature selection method assigns a score to every feature; scores are sorted in descending order, and the top-f features are selected where f is determined by the percentage p following the FSDEM schedule [32]. Stochastic methods (here, only the Random baseline) are run 10 times and their resulting metric values are averaged; while deterministic methods are run once. We evaluate the selected set of features by each feature selection method using all 9 metrics.

![](images/0f69826fe1528d5c03cd4bb2d99d95cfd69a9f886be237a696016f7e82ccb758.jpg)  
Percentage of features selected (%)  
Fig. 2: Performance curves of the nine evaluation metrics on Isolet as a function of the percentage of features selected. For all measures, higher is better (distances are converted to similarities for OT-measures).

## 4.2 Results

Figure 2 shows that OT\_GW2 produces rankings partly aligning with the supervised block by placing VCSDFS and Correlation ranked lowest and Laplacian ranked highest. Although it places SCFS fourth and Variance second, misplaced, it still shows an overall agreement. Such disagreements are observed within Supervised block as well across diferent datasets. While OT\_SINKHORN2 and OT\_EMD2 produce the ranking largely inverted with respect to the supervised block, rating the two weakest selectors best. Here, we highlight that the local disruption between 1.5% and 3.5% is caused by the per-budget cost matrix normalization performed on the OT-metrics as a standard step: the unnormalized distance grows monotonically, while its denominator grows faster. Hence, we recommend readers not to compare OT scores across diferent percentages of selected features, rather observe the ranking order. In case of OT\_SLICED\_SW, the resolution degrades. Figure 3 shows the ranking produced by OT\_SLICED\_SW for the six feature selection methods and we see traces of agreement e.g., Random and SCFS are ranked top.

![](images/5b1b997304325eeccdba9a4633779e1a121da777e971637e41bc914b918bea84.jpg)  
Fig. 3: Ranking produced by Ot\_Sliced\_SW for Isolet dataset as a function of the percentage of features selected. 1 represents highest rank and 6 represents lowest rank.

These observations indicate that the proposed framework carries predictive validity: the proposed (label-free) metrics order selectors in a way that anticipates their label-based downstream performance. This should not be read as a claim of constructing validity nor as measuring our pipeline’s eficacy by its agreement with supervised metrics. Agreement should rather be seen as the downstream-task performance (here, classification) of the selected features that a practitioner would obtain by choosing methods according to the ranking provided by the proposed measure. Furthermore, a disagreement should not be seen as declaring a metric inefective. We should not expect them to produce a single shared ordering of the selectors; rather, examining where the rankings agree and where they diverge is itself informative about the evaluation landscape. The metrics provide us with information about a specific aspect of the quality of the feature selection process. Our framework measures similarity to the first f principal components which is in particular important as there is no loss of explainability imposed by feature selection, compared to the case of PCA, where the actual features are replaced by principal components.

Figure 4 reports the pairwise correlations between evaluation metrics, separately under Pearson and Spearman correlation. It is evident that OT\_GW2 and OT\_SLICED\_SW exhibit positive correlation with respect to traditional metrics, especially the supervised block(Pearson coeficients: 0.29 − 0.37; Spearman coeficients: 0.26−0.33), While OT\_EMD2 and OT\_Sinkhorn2 exhibit opposite patterns. Among model-agnostic alternatives, AAD correlates positively but relatively weakly with the supervised block than the proposed optimal transport-based metrics, indicating that the proposed framework tracks label-based downstream performance competitively without explicit supervision.

![](images/f9e26908879647f2947e853a64088df7d9bdd3022603bfbec0b3175bd47e0849.jpg)  
Fig. 4: Pairwise correlation between 9 evaluation metrics for the percentage of features selected in the range (0.5%–5%). The left plot reports Pearson correlations; and the right plot reports Spearman. Red cells mark metric-pair that ranks the feature selection methods consistently in the same direction, blue cells represent a pair that ranks them in opposite directions.

## 5 Conclusions, Limitations, and Future Work

This study takes a first step toward a broader question: whether feature selection can be evaluated without relying on information that the feature selection process itself is not assumed to have access to. We have approached this question by replacing label-dependent evaluation with an unsupervised comparison between the selected feature space and a reference representation derived from the data itself. The results provide encouraging evidence for the feasibility of this perspective, while also revealing several methodological and empirical challenges that remain to be addressed. The following subsections summarize the main conclusions, discuss the limitations of the current framework, and outline directions for developing a more comprehensive and robust approach to truly unsupervised evaluation.

Conclusions. In this paper, we argued that the established procedures for unsupervised evaluation of feature selection are not truly unsupervised. The procedures commonly used in the literature as unsupervised evaluation in fact rely on ground-truth labels. Hence, these techniques are technically supervised, merely using an unsupervised downstream task. To the best of our knowledge, only one truly unsupervised and model-agnostic metric currently exists for the evaluation of feature selection methods [34].

We proposed a truly unsupervised evaluation framework that, instead of exploiting ground-truth labels, evaluates a feature selection algorithm by measuring the optimal transport distance between its selected feature subset and a PCA-based reference subspace of the same size. Across eight high-dimensional datasets, the proposed metrics showed consistent correlations with established evaluation metrics under both correlation measures.

Our results also demonstrate that the choice of the underlying inter-dataset similarity measure has a substantial efect on the insights obtained from the evaluation. Importantly, a lower correlation with classification-based performance should not necessarily be interpreted as evidence that a metric provides a false or uninformative signal. Rather, it may indicate that the metric captures aspects of feature selection quality that are diferent from those reflected in downstream classification performance. This observation reinforces the importance of studying unsupervised evaluation metrics not merely according to their agreement with supervised performance, but also according to the properties of the selected representations that they actually measure.

Limitations. Despite providing preliminary steps toward truly unsupervised evaluation of feature selection, the proposed framework has several important limitations.

First, optimal transport methods can be computationally expensive, particularly for large datasets. Computationally eficient approximations, such as Sinkhorn-based methods [9], provide a practical alternative but may introduce stochasticity. This is particularly problematic for an evaluation metric, for which deterministic behavior is desirable. Given the same feature selection process and dataset, repeated evaluations should ideally produce identical results. Such stability is important when metric values are subsequently used for comparisons, statistical analysis, or ranking of feature selection methods.

Second, the use of PCA as the reference mechanism imposes a structural limitation on the framework. The number of principal components that can be obtained from PCA is bounded by min(n, d), where n is the number of instances and d is the number of features. Consequently, the proposed evaluation is either restricted to datasets satisfying n ≥ d, or limited to a partial view of the feature selection process when the number of selected features exceeds this bound. Ideally, an unsupervised evaluation metric should be applicable to datasets regardless of the relationship between their numbers of instances and features, and should be capable of evaluating the selection process across its full range of dimensionality reductions.

Finally, the empirical scope of this study is limited. The experiments cover a relatively small number of datasets and feature selection methods, and the observed relationships may therefore not generalize to the broader space of feature selection problems. Moreover, the current analysis primarily examines aggregate correlations rather than investigating how these relationships vary across individual datasets or diferent levels of dimensionality reduction.

Future Work and Broader Impact. The limitations above suggest several directions for future research. First, the proposed framework should be studied at a finer granularity. In particular, correlations should be examined separately for individual datasets and for diferent levels of dimensionality reduction. Repeated experiments should also be conducted to quantify and, where possible, eliminate the efects of stochasticity. A systematic comparison of diferent optimal transport formulations and their underlying design choices is likewise necessary to understand why they produce diferent evaluation signals. Such analyses would provide a more complete picture of the behavior of the proposed framework and help establish reliable recommendations for its use.

Second, the reliance on PCA as the unsupervised reference could be relaxed. Other dimensionality reduction techniques, such as autoencoders, could provide alternative reference representations and potentially overcome the dimensionality constraints of PCA while also allowing nonlinear structures to be captured. This could further enable a more detailed analysis of method rankings using Critical Diference diagrams based on Standard rank statistics [11] and MARS [35]. More generally, alternative dimensionality reduction techniques could transform the current approach into a dynamically guided evaluation framework. For example, multiple dimensionality reduction candidates could be generated and evaluated using internal dimensionality-reduction quality measures [40], with the best-performing representation for a given dataset subsequently used as the reference for feature selection evaluation.

Third, the raw values produced by the proposed metrics warrant further investigation. It should be determined whether their distributions have a meaningful interpretation and whether a correction-for-chance adjustment is necessary [18]. Such an adjustment could improve the interpretability and comparability of metric values across datasets and experimental settings.

Fourth, the properties of individual optimal transport measures provide an opportunity to simplify the framework. In particular, the Gromov-Wasserstein distance [30], used in OT\_GW2, can compare distributions defined on spaces of diferent dimensionality. This property could potentially eliminate the need for a reference subspace altogether, allowing the selected feature subset to be compared directly with the full-dimensional dataset.

Finally, the recurring loss of resolution observed for OT\_SLICED\_SW deserves a systematic investigation. Understanding why this behavior occurs, under which dataset characteristics it emerges, and which alternative inter-dataset similarity measures reproduce or avoid it could provide important insights into the design of future unsupervised evaluation metrics.

Taken together, these directions position the present study as a starting point rather than a final characterization of unsupervised feature selection evaluation. The limited number of datasets and methods, together with the methodological constraints imposed by optimal transport and PCA, mean that the current results should primarily be interpreted as a proof of concept. We therefore do not intend the observed correlations to constitute a definitive verdict on the superiority of the proposed metrics or on the quality of existing feature selection methods. Instead, the broader aim is to demonstrate the feasibility of evaluating feature selection without relying on ground-truth labels and to motivate a more rigorous, granular, and reproducible investigation of truly unsupervised evaluation in future work.

## References

1. Amsaleg, L., Bailey, J., Barbe, A., Erfani, S.M., Furon, T., Houle, M.E., Radovanovic, M., Nguyen, X.V.: High intrinsic dimensionality facilitates adversarial attack: Theoretical evidence. IEEE Trans. Inf. Forensics Secur. 16, 854–865 (2021)

2. Anderberg, A., Bailey, J., Campello, R.J.G.B., Houle, M.E., Marques, H.O., Radovanovic, M., Zimek, A.: Dimensionality-aware outlier detection. In: SDM. pp. 652–660. SIAM (2024)

3. Asadi Amiri, S., Rajabinasab, M.: Face recognition using color and edge orientation diference histogram. Journal of AI and Data Mining 9(1), 31–38 (2021)

4. Bishop, C.M.: Pattern Recognition and Machine Learning. Springer (2006), appendix or index reference to Kronecker delta δ<sub>ij</sub>

5. Bonneel, N., van de Panne, M., Paris, S., Heidrich, W.: Displacement interpolation using lagrangian mass transport. ACM Trans. Graph. 30(6), 158 (2011)

6. Bonneel, N., Rabin, J., Peyré, G., Pfister, H.: Sliced and radon wasserstein barycenters of measures. J. Math. Imaging Vis. 51(1), 22–45 (2015)

7. Bourlard, H., Kamp, Y.: Auto-association by multilayer perceptrons and singular value decomposition. Biological Cybernetics 59(4-5), 291–294 (1988)

8. Cai, J., Luo, J., Wang, S., Yang, S.: Feature selection in machine learning: A new perspective. Neurocomputing 300, 70–79 (2018)

9. Cuturi, M.: Sinkhorn distances: Lightspeed computation of optimal transport. In: Advances in Neural Information Processing Systems 26 (NIPS 2013) (2013)

10. Cuturi, M.: Sinkhorn distances: Lightspeed computation of optimal transport. In: Advances in Neural Information Processing Systems 26. pp. 2292–2300 (2013)

11. Demšar, J.: Statistical comparisons of classifiers over multiple data sets. Journal of Machine Learning Research 7, 1–30 (2006)

12. François, D., Wertz, V., Verleysen, M.: The concentration of fractional distances. IEEE Trans. Knowl. Data Eng. 19(7), 873–886 (2007)

13. Guyon, I., Elisseef, A.: An introduction to variable and feature selection. Journal of Machine Learning Research 3, 1157–1182 (2003)

14. Hall, M.A.: Correlation-based feature selection for machine learning. Ph.D. thesis, University of Waikato (1999)

15. He, X., Cai, D., Niyogi, P.: Laplacian score for feature selection. Advances in Neural Information Processing Systems 18 (2005)

16. He, X., Cai, D., Niyogi, P.: Laplacian score for feature selection. In: NIPS. pp. 507–514 (2005)

17. Houle, M.E., Kriegel, H., Kröger, P., Schubert, E., Zimek, A.: Can shared-neighbor distances defeat the curse of dimensionality? In: SSDBM. pp. 482–500 (2010)

18. Hubert, L., Arabie, P.: Comparing partitions. Journal of classification 2(1), 193– 218 (1985)

19. Karami, S., Saberi-Movahed, F., Tiwari, P., Marttinen, P., Vahdati, S.: Unsupervised feature selection based on variance-covariance subspace distance. Neural Networks 166, 188–203 (2023)

20. Kuhn, H.W.: The Hungarian Method for the assignment problem. Naval Research Logistics Quarterly 2(1-2), 83–97 (1955)

21. Li, J., Cheng, K., Wang, S., Morstatter, F., Trevino, R.P., Tang, J., Liu, H.: Feature selection: A data perspective. ACM Computing Surveys 50(6), 94:1–94:45 (2017)

22. Van der Maaten, L., Hinton, G.: Visualizing data using t-sne. Journal of Machine Learning Research 9(11), 2579–2605 (2008)

23. McInnes, L., Healy, J., Saul, N., Grossberger, L.: Umap: Uniform manifold approximation and projection. Journal of Open Source Software 3(29), 861 (2018)

24. Mostert, W., Malan, K.M., Engelbrecht, A.P.: A feature selection algorithm performance metric for comparative analysis. Algorithms 14(3), 100 (2021)

25. Nguyen, K., Nguyen, H., Pham, T., Ho, N.: Lightspeed geometric dataset distance via sliced optimal transport. arXiv preprint arXiv:2501.18901 (2025)

26. Nogueira, S., Sechidis, K., Brown, G.: On the stability of feature selection algorithms. Journal of Machine Learning Research 18(174), 1–54 (2017)

27. Okkels, C.B., Thordsen, E., Aumüller, M., Zimek, A., Schubert, E.: Approximate single-linkage clustering using graph-based indexes: Mst-based approaches and incremental searchers. In: SISAP. pp. 233–247 (2025)

28. Parsa, M.G., Zare, H., Ghatee, M.: Unsupervised feature selection based on adaptive similarity learning and subspace clustering. Eng. Appl. Artif. Intell. 95, 103855 (2020)

29. Peyré, G., Cuturi, M.: Computational optimal transport. Foundations and Trends in Machine Learning 11(5-6), 355–607 (2019)

30. Peyré, G., Cuturi, M., Solomon, J.: Gromov-wasserstein averaging of kernel and distance matrices. In: Proceedings of The 33rd International Conference on Machine Learning. vol. 48, pp. 2664–2672. PMLR (2016)

31. Rajabinasab, M., Houle, M.E., Chelly, O., Zimek, A.: Worse than random: The importance of a baseline for unsupervised feature selection. CoRR abs/2605.22973 (2026)

32. Rajabinasab, M., Lautrup, A.D., Hyrup, T., Zimek, A.: A dynamic evaluation metric for feature selection. In: Similarity Search and Applications. pp. 65–72. Springer Nature Switzerland, Cham (2025)

33. Rajabinasab, M., Lautrup, A.D., Schneider-Kamp, P., Zimek, A.: Towards semisupervised subspace learning for outlier detection in big data. In: Similarity Search and Applications. pp. 330–344 (2026)

34. Rajabinasab, M., Lautrup, A.D., Zimek, A.: Metrics for inter-dataset similarity with example applications in synthetic data and feature selection evaluation - extended version. CoRR abs/2501.09591 (2025)

35. Rajabinasab, M., Nejad, A.M., Zimek, A.: MARS: magnitude-aware rank statistics. CoRR abs/2605.23563 (2026)

36. Rajabinasab, M., Pakdaman, F., Gabbouj, M., Schneider-Kamp, P., Zimek, A.: Randomized PCA forest for outlier detection. CoRR abs/2508.12776 (2025)

37. Rajabinasab, M., Pakdaman, F., Zimek, A., Gabbouj, M.: Randomized pca forest for approximate k-nearest neighbor search. Expert Systems with Applications 281 (2025)

38. Rajabinasab, M., Zimek, A.: FSEVAL: feature selection evaluation toolbox and dashboard. CoRR abs/2604.18227 (2026)

39. Rubner, Y., Tomasi, C., Guibas, L.J.: The earth mover’s distance as a metric for image retrieval. International Journal of Computer Vision 40(2), 99–121 (2000)

40. Van Der Maaten, L., Postma, E.O., Van Den Herik, H.J., et al.: Dimensionality reduction: A comparative review. Journal of Machine Learning Research 10(1), 1–41 (2009)

41. Villani, C.: Optimal Transport: Old and New. Springer, Berlin, Heidelberg (2009)

42. Zhao, Z., Liu, H.: Spectral feature selection for supervised and unsupervised learning. In: ICML. pp. 1151–1157 (2007)

43. Zimek, A., Schubert, E., Kriegel, H.: A survey on unsupervised outlier detection in high-dimensional numerical data. Stat. Anal. Data Min. 5(5), 363–387 (2012)