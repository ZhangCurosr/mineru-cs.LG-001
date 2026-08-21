# DICS: Data-Informed Centroid Splitting for Decision Tree Classifiers

MD Saifur Rahman Mazumder<sup>∗</sup> Feng Yu<sup>†</sup>

Department of Mathematical Sciences University of Texas at El Paso

## Abstract

Decision tree–based models are widely used in machine learning due to their interpretability and strong empirical performance. However, training decision trees can be computationally expensive, particularly for large and high-dimensional datasets, largely due to the exhaustive search over candidate splits at each node. To improve computational eficiency, we propose Data-Informed Centroid Splitting (DICS), a clustering-based framework that constructs a compact and informative set of candidate splits using data-driven priors. By incorporating class-aware structure, DICS significantly reduces the split search space for classification tasks while preserving predictive performance. We further provide theoretical analysis showing that under the stated assumptions, DICS does not degrade the performance of classification trees compared to exhaustive split search. DICS can be incorporated into classification trees, random forests, and gradient-boosting models. Extensive experiments demonstrate that DICS achieves comparable accuracy while substantially reducing training time across synthetic and benchmark datasets, highlighting the benefit of integrating data-informed priors into split selection for scalable classification tree learning.

## 1 Introduction

Data-driven methods have become fundamental across a wide spectrum of real-world applications, where systems must extract insights from large and complex datasets to make reliable decisions. Tasks such as spam filtering, targeted advertising, fraud detection, and scientific discovery all depend on models that can capture intricate patterns while remaining computationally eficient. Among the broad family of machine learning techniques, decision tree–based models are particularly appealing due to their interpretability, flexibility, and strong empirical performance. Ensemble variants, including random forests [1] and gradient boosting machines [2], are especially efective, as they can model nonlinear relationships and higher-order feature interactions while maintaining relatively low computational overhead. Their hierarchical structure further enables them to handle heterogeneous data and provide interpretable decision rules, making them a key component of modern scalable learning systems.

Decision tree (DT) [3], also known as Classification and Regression Tree (CART), is a simple and interpretable model that produces predictions by recursively partitioning the feature space through a sequence of binary decisions. Despite its conceptual simplicity, training a decision tree can be computationally demanding. At each node, the algorithm evaluates a large number of candidate split points, which are typically derived from sorted feature values, and selects the optimal split according to a predefined criterion. This exhaustive search process is repeated independently across nodes, leading to significant computational costs as the number of samples, features, or tree depth increases.

To address these challenges, a substantial body of work has focused on improving the eficiency and scalability of tree construction. Early approaches such as SLIQ [4] and RainForest [5] employ presorting techniques, breadth-first growth strategies, and compact data representations to decouple scalability from split evaluation. QUEST [6] addresses a related problem by enabling fast tree construction while reducing variable-selection bias. [7] proposed dynamic split-point selection strategies that restrict the number of candidate thresholds for continuous features; however, these methods often rely on restrictive assumptions and may not scale well in high-dimensional settings. In the context of ensemble learning, Extremely Randomized Trees [8] further reduce computational cost by introducing randomness in both feature selection and split thresholds, thereby avoiding exhaustive search. More recent systems, such as XGBoost [9] and LightGBM [10], achieve substantial eficiency gains through approximate split finding, sparsity-aware computation, and histogram-based binning techniques. Collectively, these developments highlight that split selection remains a primary computational bottleneck in tree-based learning.

In this paper, we address the computational challenges of Classification Trees by exploiting structural information in the data to guide split selection. In classification tasks, the underlying data patterns provide valuable structural information that can guide split selection. Motivated by this observation, we propose Data-Informed Centroid Splitting (DICS), a method that leverages data-driven priors through clustering to construct a significantly reduced set of candidate splits, thereby substantially accelerating the tree training process.

Beyond conventional greedy tree induction, another line of work considers globally optimized sparse decision trees, where branch-and-bound and related search strategies optimize the complete tree structure rather than selecting splits greedily [11, 12, 13, 14]. Within this setting, McTavish et al. [15], for example, use a boosted tree ensemble as a reference model to identify informative thresholds for accelerating the optimization of sparse decision trees. While DICS shares the general motivation of restricting the continuous split space, it derives the candidates directly from the clustering structure of the input data without requiring an auxiliary predictive model. Clustering has also been incorporated more directly into tree construction, Clus-DTI [16] uses clustering to decompose a classification problem into simpler subproblems before inducing decision trees, while BDTKS [17] recursively applies K-means within a multivariate binary tree and converts centroidbased separations into hyperplane tests. In contrast, DICS uses clustering only to construct a reusable set of univariate feature–threshold candidates, leaving the underlying greedy tree objective unchanged and allowing the same candidate-generation strategy to be used with classification trees, random forests, and gradient-boosting models.

In summary, our contributions are as follows:

1. We propose Data-Informed Centroid Splitting (DICS), a clustering-driven approach for constructing compact candidate split sets in classification trees.

2. We provide theoretical analysis showing that decision trees built with DICS preserve predictive performance compared to standard exhaustive split search.

3. We conduct extensive experiments demonstrating that DICS achieves comparable accuracy while significantly improving computational eficiency.

## 2 Preliminaries

## 2.1 Classification trees

We consider a multi-class classification problem. Let ${ \bf X } = [ { \pmb x } _ { 1 } , \dots , { \pmb x } _ { N } ] \in \mathbb { R } ^ { P \times N }$ denote the feature matrix and $\pmb { y } \in [ K ] ^ { N }$ the corresponding labels, where N is the number of samples, P is the number of features, and K is the number of classes. Formally, a classification tree can be defined by $\begin{array} { r } { T ( \pmb { x } ; \pmb { \theta } ) = \sum _ { m = 1 } ^ { M } c _ { m } \mathbb { I } ( \pmb { x } \in R _ { m } ) } \end{array}$ where $\pmb { \theta } = \{ ( c _ { m } , R _ { m } ) \} _ { m = 1 } ^ { M }$ is the parameter, $R _ { m }$ denotes the m-th rectangle region, $c _ { m } \in [ K ]$ is the predicted class assigned to $R _ { m }$ , and $M$ is the total number of regions. To fit the model, we minimize the following empirical risk:

$$
\mathcal { L } ( \pmb { \theta } ) = \sum _ { i = 1 } ^ { N } \ell ( y _ { i } , T ( \pmb { x } _ { i } ; \pmb { \theta } ) ) = \sum _ { m = 1 } ^ { M } \sum _ { \pmb { x } _ { i } \in R _ { m } } \ell ( y _ { i } , c _ { m } ) .\tag{1}
$$

Solving (1) typically involves two stages: (1) learning the discrete tree structure $\lbrace R _ { m } \rbrace _ { m = 1 } ^ { M }$ , and (2) estimating the optimal labels $c _ { m }$ given the partition $\lbrace R _ { m } \rbrace _ { m = 1 } ^ { M }$ . However, the first step makes (1) non-diferentiable and computationally challenging. In fact, finding the optimal data partition induced by a decision tree is NP-complete [18].

Each region $R _ { m }$ corresponds to a leaf node and is formed by a sequence of recursive binary splits. Equivalently, every $R _ { m }$ can be expressed as the intersection of splitting constraints along the path from the root to the corresponding leaf: $\begin{array} { r } { R _ { m } = \bigcap _ { t \in \mathcal { P } _ { m } } \left\{ \pmb { x } \in \mathbb { R } ^ { P } : x _ { f _ { t } } \in S _ { t } \right\} } \end{array}$ , where $\mathcal { P } _ { m }$ denotes the set of internal nodes along the path to leaf $m , f _ { t }$ is the feature selected at node t, and $S _ { t } \in \{ ( - \infty , \tau _ { t } ) , [ \tau _ { t } , \infty ) \}$ specifies the split induced by threshold $\tau _ { t }$ . A feature-threshold pair $( f , \tau )$ defines a split that partitions the index set $[ N ] = \{ 1 , \dots , N \}$ into two subsets:

$$
\begin{array} { r } { \mathcal { T } _ { L } = \{ i \in \mathcal { I } : \mathbf { X } _ { f , i } < \tau \} , \quad \mathcal { T } _ { R } = \{ i \in \mathcal { I } : \mathbf { X } _ { f , i } \geq \tau \} . } \end{array}
$$

The search for the optimal splits is typically performed using a greedy, top-down strategy. Starting from the root node, all candidate splits are evaluated at each node. Specifically, for each feature $f ,$ , the samples are first sorted according to their values along that feature, and candidate thresholds are chosen as the midpoints between consecutive distinct values. The quality of a split $( f , \tau )$ is measured by the weighted post-split impurity:

$$
Q ( f , \tau ) = \frac { | \mathcal { T } _ { L } | } { N } G ( \mathcal { T } _ { L } ) + \frac { | \mathcal { T } _ { R } | } { N } G ( \mathcal { T } _ { R } ) .\tag{2}
$$

Here, $G ( \mathcal { T } )$ denotes the impurity of a node indexed by $\mathcal { T } .$ Common choices include the Gini index and entropy (deviance). To compute $G ( \mathcal { T } )$ , we first estimate the empirical class distribution within I as $\begin{array} { r } { \hat { \pi } _ { c } = \frac { 1 } { | \mathcal { T } | } \sum _ { i \in \mathcal { T } } \mathbb { I } ( y _ { i } = c ) } \end{array}$ . The Gini index is defined as $\begin{array} { r } { G ( \mathcal { T } ) = \sum _ { c = 1 } ^ { K } \hat { \pi } _ { c } ( 1 - \hat { \pi } _ { c } ) = 1 - \sum _ { c = 1 } ^ { K } \hat { \pi } _ { c } ^ { 2 } \left[ 1 9 , 3 \right] } \end{array}$ while the entropy is given by $\begin{array} { r } { G ( \mathcal { T } ) = - \sum _ { c = 1 } ^ { K } \hat { \pi } _ { c } \log \hat { \pi } _ { c } \left[ 2 0 \right] } \end{array}$ . This impurity-based criterion underlies the standard greedy split selection in classification trees and is widely used in tree-based methods [3].

## 2.2 Ensemble trees

Although a single decision tree ofers strong interpretability, it typically sufers from high variance, instability, and sensitivity to noise in the training data. In practice, ensemble tree methods are therefore often preferred. These approaches combine multiple decision trees to construct a stronger predictive model. The main types of ensemble trees include bagging (or bootstrap aggregating), Random Forest (RF, a specific instance of bagging) [1], boosting, and etc.

For classification tasks, an ensemble tree typically makes predictions via majority voting: ${ \hat { y } } ( { \pmb x } ) =$ $\begin{array} { r } { \arg \operatorname* { m a x } _ { c \in [ K ] } \sum _ { b = 1 } ^ { B } \mathbb { I } ( T _ { b } ( \pmb { x } ) = c ) } \end{array}$ , where $\{ T _ { b } \} _ { b = 1 } ^ { B }$ are the individual trees. In bagging, each tree $T _ { b }$ is trained on a bootstrap sample $( \mathbf { X } ^ { * b } , y ^ { * b } )$ . Random Forest further reduces variance by de-correlating trees, typically by restricting each tree to a random subset of features (of size $\lfloor { \sqrt { P } } \rfloor$ for classification). In boosting, trees are constructed sequentially, with each new tree designed to correct the errors made by the previous ones. Among these approaches, gradient boosting methods such as XGBoost [9] and LightGBM [10] are widely recognized for achieving strong predictive performance on tabular data.

## 2.3 Computational bottleneck

As discussed in earlier sections, the most challenging and computationally demanding component of a Decision Tree (DT) is learning the discrete tree structure, which is equivalent to searching for optimal splits (i.e., feature–threshold pairs). The standard approach adopts a greedy, top-down strategy: at each node, it exhaustively evaluates every split out of (N − 1)P candidates, corresponding to P features and up to (N − 1) candidate thresholds per feature (typically taken as midpoints between consecutive sorted sample values). As a result, the overall training cost scales rapidly with the sample size, tree depth, and ensemble size.

This large candidate space is the main computational bottleneck of DT training. To alleviate this issue, a popular alternative is the histogram-based method [9, 21, 10]. This approach discretizes each feature into a finite number of bins and constructs feature histograms, allowing the optimal split to be found by scanning histogram bins instead of raw data values. However, this binning strategy introduces additional approximation errors. Specifically, multiple distinct feature values are merged into the same bin, and candidate splits are restricted to bin boundaries only. This leads to two main drawbacks: (1) quantization error due to coarse discretization of continuous features, and (2) a trade-of between eficiency and accuracy that depends on the number of bins used. In particular, using fewer bins improves computational eficiency but increases information loss, while using more bins reduces quantization error at the cost of higher computational and memory overhead.

Nevertheless, both exhaustive search and histogram-based methods lack prior guidance on which splits are likely to be informative. This motivates our approach, which (i) constructs a compact, data-informed set of candidate splits, and (ii) enforces a fixed per-node evaluation budget, ensuring that the split-search cost remains predictable and controlled.

## 3 Methodology

In this section, we introduce a novel approach, termed Data-Informed Centroid Splitting (DICS), which precomputes a compact, data-informed set of candidate thresholds for use in classification trees (see Section 3.1). This strategy directly addresses the split-search bottleneck without modifying the underlying tree objective. We then describe how DICS can be integrated into standard classification tree-based frameworks, including classical decision trees, random forests [1], and gradient boosting systems, in Section 3.2. We further analyze the computational complexity of DICS in Section 3.3.

## 3.1 Data-Informed Centroid Splitting

A fundamental assumption in classification is that samples belonging to the same class are generally close to each other in the feature space. This principle underlies many methods, such as k-nearest neighbors and linear discriminant analysis [22, 23, 24]. Consequently, in ideal scenarios, one might expect the classification boundaries to roughly align with clusters formed by the data even without in the absence of label information. Although this alignment may not hold exactly in practice, clustering results can still provide valuable guidance and serve as useful prior information. A simple illustrative example is shown in the left panel of Figure 1: we generate $N = 6 0 0$ two-dimensional samples for two classes with $\pmb { x } | y = 0 \sim \mathcal { N } ( ( - 1 , 0 ) , \Sigma )$ and $\mathbf { \boldsymbol { x } } | \boldsymbol { y } = 1 \sim \mathcal { N } ( ( 1 , 1 ) , \Sigma )$ and $\Sigma = [ 0 . 2 5 , 0 . 1 ; 0 . 1 , 0 . 2 5 ]$ In this setting, the optimal Bayes decision boundary (labeled as classification boundary) is well approximated by the separator obtained from applying K-means clustering to the data.

![](images/5c52f3e60b0d3405034a8b14b7b4dd083a1d99cd2f1e014dbb0c7f25e12e4880.jpg)  
Figure 1: Illustrative example: (left) classification vs. clustering boundaries; (right) midpoint vs. variance-adjusted boundaries.

Moreover, clustering can typically be performed eficiently, for instance, via K-means [25]. Motivated by the observed alignment between clustering boundaries and classification decision boundaries, as well as the computational efectiveness of K-means, we propose a novel clusteringguided approach for generating candidate splits. Specifically, let $\{ \mu _ { c } \} _ { c = 1 } ^ { K } \subset \mathbb { R } ^ { P }$ be the centroids obtained from applying K-means clustering to the feature matrix X. Each centroid $\pmb { \mu } _ { c }$ represents a dominant mode of the underlying data distribution, providing a coarse yet stable summary of where the data concentrates.

We propose Data-Informed Centroid Splitting (DICS) approach, which leverages clustering boundaries as candidate classification boundaries, further yielding splits for tree-based models. In particular, for any pair of clusters $c _ { 1 }$ and $c _ { 2 } ,$ , a boundary can be defined based on their centroids. For a given feature $f ,$ this induces a candidate split of the form $\zeta ( \mu _ { c _ { 1 } } ^ { ( f ) } , \mu _ { c _ { 2 } } ^ { ( f ) } )$ . However, in the absence of labels, there is no unique “correct” boundary in clustering. In K-means, a point x is assigned to the cluster with the nearest centroid, i.e., arg min $\mathbf { \tau } _ { 1 < c < K } \| \pmb { x } - \pmb { \mu } _ { c } \|$ . The boundary between two clusters is thus given by the set of points satisfying $\| \pmb { x } - \pmb { \bar { \mu } } _ { c _ { 1 } } \| = \| \pmb { x } - \pmb { \mu } _ { c _ { 2 } } \|$ , which induces a Voronoi partition of the space [26]. Under this construction, the corresponding one-dimensional split is given by the midpoint, $\begin{array} { r } { \zeta ( \mu _ { c _ { 1 } } ^ { ( \dot { f } ) } , \dot { \mu } _ { c _ { 2 } } ^ { ( f ) } ) = \frac { 1 } { 2 } ( \mu _ { c _ { 1 } } ^ { ( f ) } + \mu _ { c _ { 2 } } ^ { ( f ) } ) } \end{array}$ . Nevertheless, this midpoint-based boundary may be suboptimal when clusters exhibit diferent variances. For example, if one class is significantly more dispersed than the other (as illustrated in the right panel of Figure 1), the midpoint can be biased toward the wider cluster. To address this issue, we propose a variance-adjusted boundary in which the distances from the centroids to the boundary are proportional to their standard deviations, i.e. $\begin{array} { r } { \frac { \zeta - \mu _ { c _ { 1 } } ^ { ( f ) } } { \mu _ { c _ { 2 } } ^ { ( f ) } - \zeta } = \frac { \hat { \sigma } _ { c _ { 1 } } } { \hat { \sigma } _ { c _ { 2 } } } } \end{array}$ , which implies that $\begin{array} { r } { \zeta ( \mu _ { c _ { 1 } } ^ { ( f ) } , \mu _ { c _ { 2 } } ^ { ( f ) } ) = \frac { \hat { \sigma } _ { c _ { 1 } } \mu _ { c _ { 2 } } ^ { ( f ) } + \hat { \sigma } _ { c _ { 2 } } \mu _ { c _ { 1 } } ^ { ( f ) } } { \hat { \sigma } _ { c _ { 1 } } + \hat { \sigma } _ { c _ { 2 } } } } \end{array}$ , where $\hat { \sigma } _ { c _ { 1 } }$ and $\hat { \sigma } _ { c _ { 2 } }$ are the standard deviations of feature $f$ within clusters $c _ { 1 }$ and $c _ { 2 } .$ , respectively. This formulation provides a more adaptive and robust candidate split when cluster spreads difer.

We also note that there are $\begin{array} { r } { \binom { K } { 2 } = \frac { K ( K - 1 ) } { 2 } } \end{array}$ possible pairs of centroids, whereas K clusters are present. When two centroids are very close, the boundary between them is unlikely to correspond to a meaningful split in the tree. To eliminate such redundant splits, we retain only m centroid pairs corresponding to the m largest inter-centroid distances $D _ { i j } = \| \pmb { \mu _ { i } } - \pmb { \mu _ { j } } \|$ . The resulting procedures are summarized in Algorithm 1.

Algorithm 1 Data-Informed Centroid Splitting (DICS)   
Require: Feature matrix $\mathbf { X } \in \mathbb { R } ^ { P \times N }$ , maximum pair count $\begin{array} { r } { m < \frac { K ( K - 1 ) } { 2 } } \end{array}$   
Ensure: Split-candidate dictionary S.   
1: Fit K-means on X with K clusters to obtain centroids $\{ \mu _ { c } \} _ { c = 1 } ^ { K }$   
2: Compute pairwise centroid distances: $D _ { i j }  \| \pmb { \mu _ { i } } - \pmb { \mu _ { j } } \| , \forall i < j$   
3: Select $\mathcal { P } $ the m pairs (i, j) with largest $D _ { i j }$   
4: for $f = 1$ to P do   
5: $\dot { T _ { f } }  \{ \big ( f , \zeta ( \mu _ { i } ^ { ( f ) } , \mu _ { j } ^ { ( f ) } ) \big ) : ( i , j ) \in \mathcal { P } \}$ , where $\zeta = ( \hat { \sigma } _ { c _ { 1 } } \mu _ { c _ { 2 } } ^ { ( f ) } + \hat { \sigma } _ { c _ { 2 } } \mu _ { c _ { 1 } } ^ { ( f ) } ) / ( \hat { \sigma } _ { c _ { 1 } } + \hat { \sigma } _ { c _ { 2 } } )$   
6: end for   
7: return $S = \cup _ { f = 1 } ^ { P } \{ T _ { f } \}$

## 3.2 Cluster-guided (ensemble) trees

The DICS algorithm (Algorithm 1) precalculates candidate splits guided by clustering, and the classification tree is subsequently grown from the resulting split-candidate dictionary S. The corresponding adaptations of the decision tree and random forest, referred to as the Cluster-Guided Classification Tree (CGCT) and Cluster-Guided Random Forest (CGRF), are summarized in Algorithm 2 (see Section A.1.1 for details) and Algorithm 3 (see Section A.1.2 for details), respectively.

As for gradient boosting machine, we propose Fast Classification Gradient Boosting Machine (FastC-GBM), which leverages candidate splits generated by DICS and incorporates the Gradientbased One-Side Sampling (GOSS) technique [10] and column (feature) subsampling technique [9, 10]. In particular, when evaluating the quality of a split $( f , \tau )$ , the computational cost is further reduced by sampling a subset of samples with small gradient magnitudes and subsampling features. The full procedure of FastC-GBM is summarized in Algorithm 4 (see Section A.1.3 for details).

## 3.3 Implementation and complexity

The candidate splits generated by DICS depend on the results of K-means. Since DICS only requires stable centroids, we adopt mini-batch K-means [27] together with K-means++ initialization [28]. Under this strategy, DICS introduces a one-time computational overhead of order O(TKbP), where T is the number of iterations, b is the mini-batch size, K is the number of clusters, and P is the feature dimension. This one-time is relative small compared to the split finding stage.

For split finding, DICS requires a computational cost of $\mathcal { O } ( m P U )$ , where m is the number of centroid pairs and U denotes the number of operations needed to evaluate the quality of a candidate split. In contrast, an exhaustive search incurs a cost of O(NP U), while histogram-based search reduces this to $\mathcal { O } ( H P U )$ , where H is the number of bins, typically set to 255 to balance eficiency and accuracy [10]. Consequently, DICS significantly reduces the per-tree training cost to a factor of m. When the number of classes K is small (which is often the case), $m = K ( K - 1 ) / 2$ remains small; for example, $m = 1 0$ when K = 5. When K is large, our sensitivity experiments (see Appendix A.3) show that, in practice, a small value of m is still suficient to achieve comparable accuracy. We further observe that accuracy remains stable across a range of tree depths (d) and mini-batch sizes (b).

## 4 Split-Level Stability Analysis

In this section, we provide a theoretical analysis of the proposed DICS algorithm. Our main result in Theorem 1 shows that the gain of the optimal split selected by DICS is close to that of the optimal split selected by a standard decision tree (DT). In particular, the diference between these two gains is bounded by $\mathcal { O } ( 1 / \sqrt { N } )$ , implying that as the number of samples N increases, the gap vanishes asymptotically.

Let the set of candidate splits of DT be given by $\mathcal { S } _ { \mathrm { D T } } = \{ ( f , \bar { \tau } _ { i } ^ { ( f ) } ) : i = 1 , \dots , N - 1 \}$ where $\bar { \tau } _ { i } ^ { ( f ) }$ are the midpoints of the sorted observation values along feature f. The set of candidate splits generated by DICS is denoted by $\mathcal { S } _ { \mathrm { D I C S } } = \{ ( f , \tilde { \tau } _ { i } ^ { ( f ) } ) : i = 1 , \dots , m \}$ where $\tilde { \tau } _ { i } ^ { ( f ) }$ are the thresholds produced by the DICS procedure. For any split $( f , \tau )$ , it defines a partition $\mathcal { P } _ { f , \tau } = \left( \mathcal { T } _ { L } , \mathcal { T } _ { R } \right)$ , where $\mathcal { T } _ { L } = \{ i \in [ N ]$ $\mathbf { X } _ { f , i } < \tau \big \}$ and $\mathcal { T } _ { R } = \left\{ i \in [ N ] : \mathbf { X } _ { f , i } \geq \tau \right\}$ are the corresponding index sets with sizes $n _ { L } = \left. \mathcal { T } _ { L } \right.$ and $n _ { R } = \left. \mathcal { T } _ { R } \right.$ . The optimal splits for DT and DICS are defined as $( \bar { f } ^ { * } , \bar { \tau } ^ { * } ) = \arg \operatorname* { m i n } _ { ( f , \tau ) \in S _ { \mathrm { D T } } } Q ( f , \tau )$ and $( \tilde { f } ^ { * } , \tilde { \tau } ^ { * } ) = \arg \operatorname* { m i n } _ { ( f , \tau ) \in S _ { \mathrm { D I C S } } } Q ( f , \tau )$ , respectively. Consequently, the corresponding gains for DT and DICS are given by $G ( \mathbb { Z } ) - Q ( { \bar { f } } ^ { * } , { \bar { \tau } } ^ { * } )$ and $G ( \mathbb { Z } ) \mathrm { ~ - ~ } Q ( \tilde { f } ^ { * } , \tilde { \tau } ^ { * } )$ , respectively. For notational convenience, we denote $\bar { Q } ^ { * } = { \cal Q } ( \bar { f } ^ { * } , \bar { \tau } ^ { * } )$ and $\tilde { Q } ^ { * } = Q ( \tilde { f } ^ { * } , \tilde { \tau } ^ { * } )$

Theorem 1. For every f, let $\begin{array} { r } { \bar { \tau } _ { * } ^ { ( f ) } = \arg \operatorname* { m i n } _ { \tau \in \{ \bar { \tau } _ { i } ^ { ( f ) } , 1 \leq i \leq N - 1 \} } Q ( f , \tau ) } \end{array}$ be the optimal threshold for DT and $\begin{array} { r } { \tilde { \tau } _ { * } ^ { ( f ) } = \arg \operatorname* { m i n } _ { \tau \in \{ \tilde { \tau } _ { i } ^ { ( f ) } , 1 \leq i \leq m \} } Q ( f , \tau ) } \end{array}$ be the optimal threshold for DICS, where $Q ( f , \tau )$ is the Gini impurity function. Suppose that $\begin{array} { r } { \alpha = \operatorname* { m i n } _ { f } \{ \frac { n _ { L } ^ { ( f ) } } { N } , \frac { n _ { R } ^ { ( f ) } } { N } \} > 0 } \end{array}$ . Further assume that a density regularity condition holds, $i . e . ,$ for every feature $f ,$ the marginal density $p ( x ^ { ( f ) } )$ is bounded above by a constant M in a neighborhood of the optimal split $\tau _ { * } ^ { ( f ) }$ . Then $\delta = d _ { H } \Big ( \mathcal { P } _ { f , \bar { \tau } _ { * } ^ { ( f ) } } , \mathcal { P } _ { f , \tilde { \tau } _ { * } ^ { ( f ) } } \Big ) = \mathcal { O } _ { P } ( \sqrt { N } )$ Moreover, the following bound holds:

$$
| \bar { Q } ^ { * } - \tilde { Q } ^ { * } | < \left( 8 + \frac { 1 } { \alpha ( 1 - \alpha ) } \cdot \frac { \delta } { N } \right) \frac { \delta } { N } .
$$

Proof. The proof is provided in Appendix A.2.

Here, $d _ { H } ( \mathcal { P } _ { 1 } , \mathcal { P } _ { 2 } )$ in the above theorem denotes the Hamming distance between two partitions, defined as $d _ { H } ( \mathcal { P } _ { 1 } , \mathcal { P } _ { 2 } ) = | \{ i \in [ N ] : z _ { i } \neq z _ { i } ^ { \prime } \} |$ where each partition $\mathcal { P } = ( \mathcal { I } _ { L } , \mathcal { I } _ { R } )$ is associated with a label vector $z \in \{ L , R \} ^ { N }$ defined by $z _ { i } = L$ if $i \in \mathcal { Z } _ { L }$ and $z _ { i } = R$ if $i \in \mathcal { I } _ { R }$

Remark 1. The assumption in Theorem 1 states that the data distribution is well-behaved in a neighborhood of the decision boundary. Such a condition is standard in statistical learning theory, as it rules out degenerate cases in which the marginal density vanishes or concentrates excessively near the optimal split. Under this assumption, Theorem 1 shows that the partitions induced by the optimal thresholds of DT and DICS difer on at most $\delta = \mathcal { O } _ { P } ( \sqrt { N } )$ samples. Although δ grows with N, the diference between the corresponding optimal gains satisfies $\mathcal { O } _ { P } ( 1 / \sqrt { N } )$ , and therefore converges to zero as N increases. We provide complementary empirical evidence for this split-level behavior in Appendix A.4. This bound is established at the root, where the DICS dictionary is estimated from the same distribution being split; at deeper nodes, the global dictionary is not guaranteed to satisfy the same alignment, and the depth-wise diagnostic in Appendix A.4 (Table 7) shows the cumulative retained gain ratio $R _ { d }$ declining from 100% at the root to 97.18% by depth 8.

Remark 2. Theorem 1 only establishes that the gain achieved by DICS is asymptotically close to that of DT, which does not imply that one method is necessarily better than the other, as the relationship between gain and classification accuracy is nontrivial. This theoretical result is also supported by our numerical experiments.

## 5 Numerical Experiments

We evaluate the proposed split-generation method, DICS, across several decision tree–based models, including (1) standard decision trees (DT), (2) random forests (RF), and (3) gradient boosting systems. The comparisons are conducted on both synthetic and real-world datasets. All results are obtained on a machine equipped with an Apple M4 Max chip and 64GB of unified memory.

Baseline Methods. As relatively few studies integrate clustering into tree construction, we restrict our comparison in the DT setting to BDTKS [17]. We exclude Clus-DTI [16], as it alters the underlying tree objective. Moreover, since BDTKS applies the K-means algorithm recursively at each node, it becomes computationally expensive for large datasets and is not readily compatible with ensemble methods. Therefore, we do not include it in the RF and boosting settings.

Parameters Setup. We use the Gini index as the splitting criterion to compute the quality $Q ( f , \tau )$ . The maximum tree depth is set to $D = 8$ for DT and RF, and $D = 3$ for boosting-based methods. The number of sampled candidate splits is fixed at $k = 1 0 0$ . For $K \leq 5 ,$ , we set the number of centroid pairs to $m = K ( K - 1 ) / 2 ;$ otherwise, we use $m = 2 5$ . The batch size for mini-batch K-means is set to 512.

<table><tr><td rowspan="2">(N, P)</td><td rowspan="2">Class</td><td colspan="4">Time (sec)↓</td><td rowspan="2">Speedup↑</td><td colspan="3"> $\mathrm { T e s t ~ A c c } \uparrow$ </td></tr><tr><td>DT</td><td>BDTKS</td><td>CGCT</td><td> $t _ { \mathrm { D T } } / t _ { \mathrm { C G C T } }$ </td><td>DT</td><td>BDTKS</td><td>CGCT</td></tr><tr><td rowspan="3">10000,1000</td><td>3</td><td>2.47 (0.03)</td><td></td><td>30.16</td><td>0.19 (0.02)</td><td>13.00</td><td>0.62 (0.00)</td><td>0.54</td><td>0.60 (0.00)</td></tr><tr><td>5</td><td>2.53 (0.01)</td><td></td><td>34.36 0.25</td><td>(0.02)</td><td>10.12</td><td>0.54 (0.00)</td><td>0.39</td><td>0.52 (0.00)</td></tr><tr><td>10</td><td>2.52 (0.00)</td><td>35.68</td><td>0.29 (0.02)</td><td>8.69</td><td>0.43</td><td>(0.00)</td><td>0.32</td><td>0.41 (0.00)</td></tr><tr><td rowspan="3">5000,7000</td><td>3</td><td>8.36 (0.10)</td><td></td><td>29.85 0.58</td><td>(0.05)</td><td>14.41</td><td>0.61 (0.01)</td><td>0.57</td><td>0.59 (0.01)</td></tr><tr><td>5</td><td>8.14 (0.03)</td><td></td><td>29.44 0.91</td><td>(0.04)</td><td>8.94</td><td>0.47 (0.01)</td><td>0.38</td><td>0.45 (0.00)</td></tr><tr><td>10</td><td>8.27 (0.02)</td><td>30.93</td><td>1.04 (0.08)</td><td>7.95</td><td></td><td>0.31 (0.00)</td><td>0.27</td><td>0.31 (0.01)</td></tr><tr><td rowspan="3">20000,5000</td><td>3</td><td>26.89 (0.26)</td><td>107.82</td><td>1.20</td><td>(0.07) 22.40</td><td></td><td>0.53 (0.00)</td><td>0.47</td><td>0.52 2 (0.00)</td></tr><tr><td>5</td><td>27.17</td><td>(0.04)</td><td>110.24 1.77</td><td>(0.07)</td><td>15.35</td><td>0.47 (0.00)</td><td>0.38</td><td>0.46 (0.00)</td></tr><tr><td>10</td><td>27.98 3 (0.05)</td><td>114.27</td><td>1.99 (0.09)</td><td>14.06</td><td></td><td>0.48 (0.00)</td><td>0.39</td><td>0.49 (0.00)</td></tr></table>

Table 1: Accuracy and training time (in seconds) for DT, BDTKS, and CGCT in synthetic data. The ratio t<sub>DT</sub>/t<sub>CGCT</sub> quantifies the training time improvement of CGCT relative to DT.

<table><tr><td rowspan="2">(N, P)</td><td rowspan="2">Class</td><td colspan="2">Time (sec)↓</td><td>Speedup↑</td><td colspan="2">Test Acc↑</td></tr><tr><td>RF</td><td>CGRF</td><td> $t _ { \mathrm { R F } } / t _ { \mathrm { C G R F } }$ </td><td>RF</td><td>CGRF</td></tr><tr><td rowspan="3">10000,1000</td><td>3</td><td>4.05 (0.13)</td><td>0.36 (0.04)</td><td>11.25</td><td>0.64 (0.00)</td><td>0.64 (0.00)</td></tr><tr><td>5</td><td>4.32 (0.02)</td><td>0.41 (0.03)</td><td>10.54</td><td>0.56 (0.01)</td><td>0.56 (0.00)</td></tr><tr><td>10</td><td>4.25 (0.04)</td><td>0.53 (0.02)</td><td>8.02</td><td>0.45 (0.01)</td><td>0.45 (0.01)</td></tr><tr><td rowspan="3">5000,7000</td><td>3</td><td>13.27 (0.45)</td><td>0.93 (0.10)</td><td>14.27</td><td>0.64 (0.00)</td><td>0.64 (0.01)</td></tr><tr><td>5</td><td>13.18 (0.05)</td><td>1.14 (0.05)</td><td>11.56</td><td>0.53 (0.00)</td><td>0.52 (0.00)</td></tr><tr><td>10</td><td>13.17 (0.04)</td><td>1.66 (0.08)</td><td>7.93</td><td>0.36 (0.00)</td><td>0.36 (0.00)</td></tr><tr><td rowspan="3">20000,5000</td><td>3</td><td>44.47 (1.45)</td><td>1.49 (0.04)</td><td>29.84</td><td>0.56 (0.00)</td><td>0.54 (0.00)</td></tr><tr><td>5</td><td>48.26 (1.07)</td><td>1.67 (0.10)</td><td>28.90</td><td>0.49 (0.00)</td><td>0.49 (0.00)</td></tr><tr><td>10</td><td>53.25 (0.60)</td><td>2.09 (0.06)</td><td>25.48</td><td>0.52 (0.00)</td><td>0.52 (0.00)</td></tr></table>

Table 2: Accuracy and training time (in seconds) for RF and CGRF in synthetic data. The ratio t<sub>RF</sub>/t<sub>CGRF</sub> quantifies the training time improvement of CGRF relative to RF.

<table><tr><td rowspan="2">(N, P)</td><td rowspan="2">Class</td><td colspan="3">Time (sec)↓</td><td colspan="2">Speedup↑</td><td colspan="3">Test Acc↑</td></tr><tr><td>XGBoost</td><td>LightGBM</td><td>FastC-GBM</td><td> $\rho _ { 1 }$ </td><td> $\rho _ { 2 }$ </td><td>XGBoost</td><td>LightGBM</td><td>FastC-GBM</td></tr><tr><td rowspan="3">10000,1000</td><td>3</td><td>1.62 (0.03)</td><td>1.16 (0.06)</td><td>0.32 (0.00)</td><td>5.06</td><td>3.62</td><td>0.68 (0.00)</td><td>0.68 (0.0)</td><td>0.65 (0.01)</td></tr><tr><td>5</td><td>2.66 (0.07)</td><td>1.81 (0.07)</td><td>0.35 (0.00)</td><td>7.60</td><td>5.17</td><td>0.59 (0.01)</td><td>0.60 (0.01)</td><td>0.59 (0.01)</td></tr><tr><td>10</td><td>5.03 (0.02)</td><td>3.31 (0.0)</td><td>0.51 (0.01)</td><td>9.86</td><td>6.49</td><td>0.47 (0.01)</td><td>0.46 (0.01)</td><td>0.45 (0.01)</td></tr><tr><td rowspan="3">5000,7000</td><td>3</td><td>6.62 (0.06)</td><td>3.09 (0.03)</td><td>0.24 (0.00)</td><td>27.58</td><td>12.87</td><td>0.63 (0.00)</td><td>0.63 (0.02)</td><td>0.61 (0.02)</td></tr><tr><td>5</td><td>11.25 (0.09)</td><td>4.96 (0.07)</td><td>0.28 (0.00)</td><td>40.18</td><td>17.71</td><td>0.51 (0.02)</td><td>0.50 (0.02)</td><td>0.50 (0.02)</td></tr><tr><td>10</td><td>18.06 (0.45)</td><td>9.80 (0.33)</td><td>0.36 (0.00)</td><td>50.17</td><td>27.22</td><td>0.35 (0.01)</td><td>0.35 (0.01)</td><td>0.34 (0.01)</td></tr><tr><td rowspan="3">20000,5000</td><td>3</td><td>8.52 (0.30)</td><td>5.28 (0.18)</td><td>0.66 (0.02)</td><td>12.90</td><td>8.00</td><td>0.57 (0.01)</td><td>0.57 (0.01)</td><td>0.56 (0.01)</td></tr><tr><td>5</td><td>15.57 (0.61)</td><td>8.41 (0.24)</td><td>0.48 (0.00)</td><td>32.44</td><td>17.52</td><td>0.51 (0.01)</td><td>0.51 (0.01)</td><td>0.49 (0.01)</td></tr><tr><td>10</td><td>31.41 (0.27)</td><td>14.72 (0.12)</td><td>1.01 (0.02)</td><td>31.10</td><td>14.57</td><td>0.53 (0.01)</td><td>0.53 (0.01)</td><td>0.53 (0.01)</td></tr></table>

Table 3: Accuracy and training time (in seconds) for XGBoost, LightGBM and FastC-GBM in synthetic data. The ratios $\rho _ { 1 } = t _ { \mathrm { X G B o o s t } } / t _ { \mathrm { F a s t C - G B M } }$ and $\rho _ { 2 } = t _ { \mathrm { L i g h t G B M } } / t _ { \mathrm { F a s t C } }$ <sub>−GBM</sub> quantify the training time improvement of FastC-GBM relative to XGBoost and LightGBM, respectively.

## 5.1 Synthetic data

Formulation. To evaluate the performance of the tree-based methods, we generate a multiclass synthetic dataset in which the class label depends on a mixture of linear and nonlinear feature efects. For each simulation, we sample N observations $\{ ( x _ { i } , y _ { i } ) \} _ { i = 1 } ^ { N }$ as follows. The P-dimensional feature vector $X \in \mathbb { R } ^ { P }$ is drawn from a multivariate normal distribution $X \sim { \mathcal { N } } ( 0 , \Sigma )$ , where $\Sigma _ { i i } = 1$ for all $i \in [ P ]$ and $\Sigma _ { i j } = \rho$ for all $i \neq j$ . Given $X$ , the label Y follows a categorical distribution, $Y \mid X \sim \operatorname { C a t e g o r i c a l } ( \theta _ { 1 } , \dots , \theta _ { K } )$ , where

$$
\theta _ { c } = P ( Y = c | X ) : = \frac { \exp ( s _ { c } ) } { \sum _ { c ^ { \prime } = 1 } ^ { K } \exp ( s _ { c ^ { \prime } } ) } , \quad \forall c = 1 , \dots , K
$$

with

$$
s _ { c } = \sum _ { j \in { \cal S } _ { 1 } } a _ { j } X _ { j } + \sum _ { j \in { \cal S } _ { 2 } } b _ { j } \sin ( \pi X _ { j } ) + \sum _ { j \in { \cal S } _ { 3 } } c _ { j } X _ { j } ^ { 2 } + \sum _ { k , l \in { \cal S } _ { 4 } \atop k < l } d _ { k l } X _ { k } X _ { l } + \eta .
$$

Here, the variable sets ${ \mathcal { S } } _ { 1 } , { \mathcal { S } } _ { 2 } , { \mathcal { S } } _ { 3 }$ , and $S _ { 4 }$ correspond to linear, sinusoidal, quadratic, and pairwise interaction efects, respectively. The collection $S _ { 1 } , S _ { 2 } , S _ { 3 } , S _ { 4 }$ forms a partition of $[ P ]$ , with cardinalities given by $| S _ { 1 } | = \lfloor P / 2 \rfloor , | S _ { 2 } | = \lfloor P / 4 \rfloor , | S _ { 3 } | = \lfloor P / 8 \rfloor$ , and $\begin{array} { r } { \left| S _ { 4 } \right| = P - \sum _ { k = 1 } ^ { 3 } \left| S _ { k } \right| } \end{array}$ . Moreover, the weights $a _ { j } , b _ { j } , c _ { j } , d _ { k l } , \eta \sim \mathcal { N } ( 0 , 1 )$

Results. For the synthetic datasets, we repeat each simulation 10 times, except for BDTKS, which is run only once due to its high training cost. The mean and standard deviation (reported in parentheses) of the training time and accuracy for all methods across the three settings with $\rho = 0 . 5$ are presented in Table 1–3. Additional results for the case $\rho = 0$ are provided in Appendix A.5.

From Table 1, CGCT delivers an impressive 8–22x speedup over the classical DT on synthetic data, with only a marginal trade-of in accuracy. In contrast, BDTKS does not demonstrate advantages in either training speed or accuracy. In the random forest and boosting settings, Table 2 and Table 3 demonstrate that DICS substantially accelerates training with negligible impact on accuracy (within 0.02). Specifically, CGRF is 8–30x faster than RF, while FastC-GBM is 3.6–27x faster than LightGBM and 5–50x faster than XGBoost.

<table><tr><td rowspan="2">Dataset</td><td colspan="3">Train time (s)↓</td><td rowspan="2">Speedup↑</td><td colspan="3">Test Acc↑</td></tr><tr><td>DT</td><td>CGCT BDTKS</td><td></td><td>tDT/tcGCT DT</td><td></td><td>CGCT BDTKS</td></tr><tr><td>Helena</td><td>1.30</td><td>0.12</td><td>470.25</td><td>10.83</td><td>0.28</td><td>0.26</td><td>0.16</td></tr><tr><td>Spambase</td><td>0.02</td><td>0.01</td><td>5.42</td><td>2.00</td><td>0.95</td><td>0.93</td><td>0.79</td></tr><tr><td>Santander</td><td>17.99</td><td>0.83</td><td>714.26</td><td>21.67</td><td>0.91</td><td>0.90</td><td>0.85</td></tr><tr><td>CIFAR-10</td><td>42.63</td><td>3.33</td><td>533.91</td><td>12.80</td><td>0.34</td><td>0.31</td><td>0.29</td></tr><tr><td>MNIST</td><td>3.27</td><td>0.99</td><td>124.42</td><td>3.30</td><td>0.83</td><td>0.79</td><td>0.91</td></tr><tr><td>Fashion-MNIST</td><td>7.27</td><td>1.07</td><td>176.54</td><td>6.80</td><td>0.80</td><td>0.76</td><td>0.80</td></tr></table>

Table 4: Accuracy and training time (in seconds) for DT, BDTKS, and CGCT in real datasets. The ratio t<sub>DT</sub>/t<sub>CGCT</sub> quantifies the training time improvement of CGCT relative to DT.

## 5.2 Real Datasets

Real Datasets. We use six real-world datasets spanning a variety of domains and scales, including Helena [29], Spambase [30], Santander [31], CIFAR-10 [32], MNIST [33], and Fashion-MNIST [34]. These datasets cover both tabular and image classification tasks, enabling a comprehensive evaluation of diferent model behaviors across heterogeneous data types. A summary of these datasets is given in Table 12 (Appendix).

Helena is a high-dimensional biological dataset from OpenML, commonly used for benchmarking feature selection and classification methods. Spambase is a classic UCI dataset for email spam detection based on word and character frequencies. The Santander dataset consists of anonymized customer transaction features for binary classification in a financial setting. CIFAR-10 is a standard image recognition benchmark containing 10 object categories with natural images, while MNIST is a widely used handwritten digit dataset for digit classification. Fashion-MNIST is a more challenging variant of MNIST that contains grayscale images of clothing items, providing a modern replacement for digit recognition benchmarks.

Results. The training time and accuracy of all methods across the three settings on real-world datasets are reported in Table 4–Table 6. Overall, tree-based algorithms enhanced with DICS achieve substantially improved training eficiency in these datasets. For the decision tree comparison in Table 4, CGCT is approximately 2-21x faster than DT, while its accuracy decreases slightly more than in the synthetic setting. In contrast, BDTKS is significantly slower and exhibits unstable performance across datasets. From Table 5, DICS accelerates the RF training by approximately 2.3-12.8x, while the accuracy drop remains small (up to 0.02). For boosting methods comparison in Table 6, the proposed FastC-GBM is consistently faster than LightGBM (up to 9.5x) and XGBoost (up to 18.75x). The only exception is the Santander dataset, where the performance degradation is likely due to implementation-level memory management issues on a large-scale dataset (N = 200k), rather than the algorithm itself. Finally, we observe that LightGBM performs unusually poorly on the Helena dataset, whereas FastC-GBM maintains stable performance, suggesting improved robustness compared to LightGBM, which is known to rely on a more aggressive splitting strategy that may afect stability in certain regimes.

## 6 Conclusion and Future Directions

We propose DICS, a novel approach for generating data-informed candidate splits in tree-based methods, resulting in a substantially reduced search space. Our theoretical analysis shows that, asymptotically, DICS yields candidate splits of comparable quality to those of classical decision trees, and therefore achieves similar predictive performance. Empirical results further demonstrate that DICS maintains comparable accuracy while significantly improving computational eficiency. Currently, the proposed approach focuses on accelerating classification tasks, which serves as the main limitation. An important direction for future work is to extend this framework by incorporating data-informed priors to similarly improve eficiency in regression settings.

<table><tr><td rowspan="2">Dataset</td><td colspan="2">Train time  $( \mathrm { s } ) \downarrow$ </td><td>Speedup↑</td><td colspan="2">Test Acc↑</td></tr><tr><td>RF</td><td>CGRF</td><td> $t _ { \mathrm { R F } } / t _ { \mathrm { C G R F } }$ </td><td>RF</td><td>CGRF</td></tr><tr><td>Helena</td><td>1.43 (0.12)</td><td>0.22 (0.02)</td><td>6.50</td><td>0.29 (0.00)</td><td>0.28 (0.00)</td></tr><tr><td>Spambase</td><td>0.09 (0.00)</td><td>0.03 (0.01)</td><td>3.00</td><td>0.95 (0.00)</td><td>0.94 (0.00)</td></tr><tr><td>Santander</td><td>22.04 (0.46)</td><td>3.12 (0.04)</td><td>7.06</td><td>0.90 (0.00)</td><td>0.90 (0.00)</td></tr><tr><td>CIFAR-10</td><td>55.96 (2.47)</td><td>4.37 (0.12)</td><td>12.80</td><td>0.40 (0.01)</td><td>0.38 (0.01)</td></tr><tr><td>MNIST</td><td>5.24 (0.25)</td><td>2.28 (0.02)</td><td>2.30</td><td>0.93 (0.00)</td><td>0.92 (0.00)</td></tr><tr><td>Fashion-MNIST 10.67 ~(0.05)</td><td></td><td>2.40 (0.03)</td><td>4.45</td><td>0.83 (0.00)</td><td>0.82 (0.00)</td></tr></table>

Table 5: Accuracy and training time (in seconds) for RF and CGRF in real datasets. The ratio t<sub>RF</sub>/t<sub>CGRF</sub> quantifies the training time improvement of CGRF relative to RF.
<table><tr><td></td><td colspan="3">Train time (s)↓</td><td colspan="3">Speedup↑</td><td colspan="3">Test Acc↑</td></tr><tr><td>Dataset</td><td>XGBoost</td><td>LightGBM</td><td>FastC-GBM</td><td> $\rho _ { 1 }$ </td><td> $\rho _ { 2 }$ </td><td></td><td>XGBoost</td><td>LightGBM</td><td>FastC-GBM</td></tr><tr><td>Helena</td><td>13.01 (0.14)</td><td>13.38 (0.15)</td><td>5.48 (0.26)</td><td>2.37</td><td>2.44</td><td>0.35</td><td>(0.00)</td><td>0.23 (0.00)</td><td>0.32 (0.00)</td></tr><tr><td>Spambase</td><td>0.20 (0.00)</td><td>0.16 (0.01)</td><td>0.06</td><td>(0.00)</td><td>3.34</td><td>2.67</td><td>0.95 (0.00)</td><td>0.95 (0.00)</td><td>0.94 (0.00)</td></tr><tr><td>Santander</td><td>0.91 (0.03)</td><td>1.09 (0.02)</td><td>1.12</td><td>(0.01)</td><td>0.81</td><td>0.97</td><td>0.90 (0.00)</td><td>0.90 (0.00)</td><td>0.90 (0.00)</td></tr><tr><td>CIFAR-10</td><td>52.50 (0.86)</td><td>26.57 (0.29)</td><td>2.80</td><td>(0.04)</td><td>18.75</td><td>9.49</td><td>0.45 (0.00)</td><td>0.48 (0.00)</td><td>0.46 (0.00)</td></tr><tr><td>MNIST</td><td>29.42 (0.07)</td><td>4.33 (0.01)</td><td>2.55</td><td>(0.04)</td><td>11.54 1.70</td><td></td><td>0.94 (0.00)</td><td>0.95 (0.00)</td><td>0.95 (0.00)</td></tr><tr><td>Fashion-MNIST</td><td>22.24 (0.11)</td><td>10.93 (0.10)</td><td>2.85</td><td>(0.04)</td><td>7.80</td><td>3.83</td><td>0.86 (0.00)</td><td>0.87 (0.00)</td><td>0.86 (0.00)</td></tr></table>

Table 6: Accuracy and training time (in seconds) for XGBoost, LightGBM and FastC-GBM in real datasets. The ratios $\rho _ { 1 } = t _ { \mathrm { X G B o o s t } } / t _ { \mathrm { F a s t C - G B M } }$ and $\rho _ { 2 } = t _ { \mathrm { L i g h t G B M } } / t _ { \mathrm { F a s t C - } }$ <sub>−GBM</sub> quantify the training time improvement of FastC-GBM relative to XGBoost and LightGBM, respectively.

## References

[1] Leo Breiman. Random forests. Machine learning, 45(1):5–32, 2001.

[2] Jerome H. Friedman. Greedy function approximation: a gradient boosting machine. Annals of statistics, pages 1189–1232, 2001.

[3] Leo Breiman, Jerome Friedman, Richard A Olshen, and Charles J Stone. Classification and regression trees, volume 8. Chapman and Hall/CRC, 2017.

[4] Manish Mehta, Rakesh Agrawal, and Jorma Rissanen. SLIQ: A fast scalable classifier for data mining. In International conference on extending database technology, pages 18–32. Springer, 1996.

[5] Johannes Gehrke, Raghu Ramakrishnan, and Venkatesh Ganti. Rainforest-a framework for fast decision tree construction of large datasets. In VLDB, volume 98, pages 416–427, 1998.

[6] Wei-Yin Loh and Yu-Shan Shih. Split selection methods for classification trees. Statistica sinica, pages 815–840, 1997.

[7] David Maxwell Chickering, Christopher Meek, and Robert Rounthwaite. Eficient determination of dynamic split points in a decision tree. In Proceedings 2001 IEEE international conference on data mining, pages 91–98. IEEE, 2001.

[8] Pierre Geurts, Damien Ernst, and Louis Wehenkel. Extremely randomized trees. Machine Learning, 63(1):3–42, 2006.

[9] Tianqi Chen and Carlos Guestrin. XGBoost: A scalable tree boosting system. In Proceedings of the 22nd acm sigkdd international conference on knowledge discovery and data mining, pages 785–794, 2016.

[10] Guolin Ke, Qi Meng, Thomas Finley, Taifeng Wang, Wei Chen, Weidong Ma, Qiwei Ye, and Tie-Yan Liu. LightGBM: A highly eficient gradient boosting decision tree. Advances in neural information processing systems, 30, 2017.

[11] Xiyang Hu, Cynthia Rudin, and Margo Seltzer. Optimal sparse decision trees. In Advances in Neural Information Processing Systems, volume 32, pages 7265–7273. Curran Associates, Inc., 2019.

[12] Rui Zhang, Rui Xin, Margo Seltzer, and Cynthia Rudin. Optimal sparse regression trees. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 37, pages 11172–11180, 2023.

[13] Jimmy Lin, Chudi Zhong, Diane Hu, Cynthia Rudin, and Margo Seltzer. Generalized and scalable optimal sparse decision trees. In Proceedings of the 37th International Conference on Machine Learning (ICML’20), volume 119, pages 6150–6160. PMLR, 2020.

[14] Gaël Aglin, Siegfried Nijssen, and Pierre Schaus. Learning optimal decision trees using caching branchand-bound search. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 34, pages 3146–3153, 2020.

[15] Hayden McTavish, Chudi Zhong, Reto Achermann, Ilias Karimalis, Jacques Chen, Cynthia Rudin, and Margo Seltzer. Fast sparse decision tree optimization via reference ensembles. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 36, pages 9604–9613, 2022.

[16] Rodrigo C Barros, André CPL R de Carvalho, Marcio R Basgalupp, and Marcos G Quiles. A clusteringbased decision tree induction algorithm. In 2011 11th International Conference on Intelligent Systems Design and Applications, pages 543–550. IEEE, 2011.

[17] Fei Wang, Quan Wang, Feiping Nie, Zhongheng Li, Weizhong Yu, and Fuji Ren. A linear multivariate binary decision tree classifier based on K-means splitting. Pattern Recognition, 107:107521, 2020.

[18] Hyafil Laurent and Ronald L. Rivest. Constructing optimal binary decision trees is NP-complete. Information processing letters, 5(1):15–17, 1976.

[19] Corrado Gini. Variabilità e mutabilità: contributo allo studio delle distribuzioni e delle relazioni statistiche. Tipogr. di P. Cuppini, 1912.

[20] Qing Ren Wang and Ching Y. Suen. Analysis and design of a decision tree based on entropy reduction and its application to large character set recognition. IEEE Transactions on Pattern Analysis and Machine Intelligence, PAMI-6(4):406–417, 1984.

[21] Ping Li, Qiang Wu, and Christopher Burges. McRank: Learning to rank using multiple classification and gradient boosting. Advances in neural information processing systems, 20, 2007.

[22] Thomas Cover and Peter Hart. Nearest neighbor pattern classification. IEEE transactions on information theory, 13(1):21–27, 1967.

[23] Richard O. Duda, Peter E. Hart, and David G. Stork. Pattern Classification. Wiley-Interscience, New York, 2 edition, 2001.

[24] Christopher M. Bishop and Nasser M. Nasrabadi. Pattern recognition and machine learning, volume 4. Springer, 2006.

[25] James B. McQueen. Some methods of classification and analysis of multivariate observations. In Proc. of 5th Berkeley Symposium on Math. Stat. and Prob., pages 281–297, 1967.

[26] Georges Voronoi. Nouvelles applications des paramètres continus à la théorie des formes quadratiques. deuxième mémoire. recherches sur les parallélloèdres primitifs. Journal für die reine und angewandte Mathematik (Crelles Journal), 1908(134):198–287, 1908.

[27] David Sculley. Web-scale k-means clustering. In Proceedings of the 19th international conference on World wide web, pages 1177–1178, 2010.

[28] David Arthur and Sergei Vassilvitskii. k-means++: the advantages of careful seeding. In Proceedings of the Eighteenth Annual ACM-SIAM Symposium on Discrete Algorithms, SODA ’07, page 1027–1035, USA, 2007. Society for Industrial and Applied Mathematics.

[29] OpenML. HELENA dataset. https://www.openml.org/d/41169, 2018.

[30] Mark Hopkins, Erik Reeber, George Forman, and Jaap Suermondt. Spambase. UCI Machine Learning Repository, 1999. OpenML dataset ID 44.

[31] Santander. Santander customer transaction prediction. https://www.kaggle.com, 2019.

[32] Alex Krizhevsky. Learning multiple layers of features from tiny images. University of Toronto, 2009.

[33] Yann LeCun, Léon Bottou, Yoshua Bengio, and Patrick Hafner. Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11):2278–2324, 1998.

[34] Han Xiao, Kashif Rasul, and Roland Vollgraf. Fashion-MNIST: a novel image dataset for benchmarking machine learning algorithms. arXiv preprint arXiv:1708.07747, 2017.

## A Technical appendices and supplementary material

In this Appendix, we provide the full details of the tree-based algorithms incorporating the DICS algorithm in Section A.1. The proof of the main theorem is presented in Section A.2. We further include a parameter sensitivity analysis in Section A.3, empirical validation of DICS Split Quality in Section A.4 as well as additional experimental results on synthetic data in Section A.5.

## A.1 Algorithmic Details

In this section, we present the detailed algorithms for tree-based methods incorporating DICS. In particular, we introduce the Cluster-Guided Classification Tree (CGCT), the Cluster-Guided Random Forest (CGRF), and the cluster-guided boosting method, referred to as the Fast Classification Gradient Boosting Machine (FastC-GBM).

## A.1.1 Cluster-Guided Classification Tree (CGCT)

DICS precalculates candidate splits S, which yields two main advantages: (a) the number of candidate splits is reduced to mP, significantly lowering the computational cost; (b) S focuses on representative thresholds that capture the primary variations in the data. As a result, it is suficient to explore only a subset of S to obtain high-quality splits, avoiding exhaustive evaluation. We refer to the resulting method as the Cluster-Guided Classification Tree (CGCT), summarized in Algorithm 2.

## A.1.2 Cluster-guided random forest (CGRF)

As an ensemble method, RF inherits the computational challenges of DT, since building each tree remains expensive when N is large, even though each split only considers a random subset of features (e.g., P for classification). To further accelerate training, the DICS algorithm can be incorporated by replacing the standard decision tree construction with the Cluster-Guided Classification Tree (CGCT) for each tree in the ensemble. We note that the split-candidate dictionary S obtained from bootstrap samples typically varies only slightly, as the perturbation induced by resampling has limited impact on the underlying K-means clustering. Therefore, it is unnecessary to recompute S for every CGCT, which helps reduce redundant computations. We refer to this method as the Cluster-Guided Random Forest (CGRF), which is summarized in Algorithm 3.

```latex
Algorithm 2 Cluster-Guided Classification Tree (CGCT)
Require: Feature matrix $\mathbf { X } \in \mathbb { R } ^ { P \times N }$ , labels $\pmb { y } \in [ K ] ^ { N }$ , split-candidate dictionary $\textit { S } \left( \mathrm { A l g . ~ } 1 \right)$
maximum depth $D ,$ number of sampled candidates k.
Ensure: Decision tree $T$
1: Function BuildTree $( \mathbf { X } , \pmb { y } , d )$
2: if $d \geq D$ or $| y | \le 2$ or $\textbf {  { y } }$ is pure then
3: return leaf node with majority class in $\textbf {  { y } }$
4: end if
5: (Split selection)
6: Sample a subset $\mathcal { C } \subset S$ with $| { \mathcal { C } } | = k$ (without replacement)
7: $( f ^ { * } , \tau ^ { * } )  \arg \operatorname* { m i n } _ { ( f , \tau ) \in { \mathcal { C } } } Q ( f , \tau )$
8: $S \gets S / \{ ( f ^ { * } , \tau ^ { * } ) \}$
9: (Partition)
10: $\mathbf { X } _ { L } , \mathbf { y } _ { L } \gets \{ ( \mathbf { x } _ { i } , y _ { i } ) : \mathbf { X } _ { f ^ { * } , i } < \tau ^ { * } \}$
11: ${ \bf X } _ { R } , { \pmb y } _ { R } \gets \{ ( { \pmb x } _ { i } , y _ { i } ) : { \bf X } _ { f ^ { * } , i } \geq \tau ^ { * } \}$
12: (Recursion)
13: $T _ { L } \gets \mathrm { B u m T R E E } ( \mathbf { X } _ { L } , \pmb { y } _ { L } , d + 1 )$
14: $T _ { R } \gets \mathrm { B u m T R E E } ( \mathbf { X } _ { R } , \pmb { y } _ { R } , d + 1 )$
15: return node $\left( f ^ { \ast } , \tau ^ { \ast } , T _ { L } , T _ { R } \right)$
16: return ${ \cal T } \gets \mathrm { B u m T R E E } ( \mathbf { X } , \pmb { y } , 0 )$
```

## A.1.3 Fast classification GBM (FastC-GBM)

For classification tasks, let $\begin{array} { r } { f _ { t , c } ( \pmb { x } ) = \sum _ { j = 1 } ^ { M } w _ { j , c } ^ { t } \mathbb { I } ( \pmb { x } \in R _ { j } ^ { t } ) , \forall c = 1 , \ldots , K , } \end{array}$ denote the tree model at iteration t for each class $c ;$ the resulting class scores are then transformed into probabilities via the softmax function. Each model $f _ { t } = [ f _ { t , 1 } , \dots , f _ { t , c } , \dots , f _ { t , K } ] ^ { \intercal } \in \mathbb { R } ^ { K }$ outputs a $K \cdot$ -dimensional vector. Gradient tree boosting system then proceeds by iteratively solving, at iteration t, the following optimization problem based on the current prediction $\hat { \pmb { y } } _ { i } ^ { ( t - 1 ) }$

$$
\mathcal { L } ^ { ( t ) } = \sum _ { i = 1 } ^ { N } \ell ( \pmb { y } _ { i } , \hat { \pmb { y } } _ { i } ^ { ( t - 1 ) } + f _ { t } ( \pmb { x } _ { i } ) ) + \Omega ( f _ { t } ) ,\tag{3}
$$

where the loss function is the multiclass cross-entropy

$$
\ell ( { \pmb y } _ { i } , \hat { \pmb y } _ { i } ) = - \sum _ { c = 1 } ^ { K } y _ { i c } \log p _ { i c } , \quad p _ { i c } = \frac { \exp ( \hat { y } _ { i c } ) } { \sum _ { l = 1 } ^ { K } \exp ( \hat { y } _ { i l } ) }
$$

and the regularization term is given by $\begin{array} { r } { \Omega ( f _ { t } ) = \gamma M + \frac { 1 } { 2 } \lambda \sum _ { c = 1 } ^ { K } | | w _ { : , c } ^ { t } | | ^ { 2 } } \end{array}$ . Taking a second-order approximation of (3) and expanding the regularization term yields

$$
\tilde { \mathcal { L } } ^ { ( t ) } = \sum _ { c = 1 } ^ { K } \sum _ { j = 1 } ^ { M } [ ( \sum _ { i \in \mathcal { T } _ { j } } g _ { i c } ) w _ { j , c } + \frac { 1 } { 2 } ( \lambda + \sum _ { i \in \mathcal { T } _ { j } } h _ { i c } ) w _ { j , c } ^ { 2 } ] + \gamma M ,\tag{4}
$$

where ${ \mathcal { T } } _ { j } = \{ i : { \pmb x } _ { i } \in R _ { j } ^ { t } \}$ denote the instance set of leaf $j$ and $g _ { i c } , h _ { i c }$ are the first and second order gradient statistics on the loss function for each class $^ { c , }$ whose formulas are given by

$$
g _ { i c } = p _ { i c } - y _ { i c } , \quad h _ { i c } = p _ { i c } ( 1 - p _ { i c } ) , \quad \forall c = 1 , \ldots , K .
$$

Algorithm 3 Cluster-Guided Random Forest (CGRF)   
Require: Feature matrix $\mathbf { X } \in \mathbb { R } ^ { P \times N }$ , labels $\pmb { y } \in [ K ] ^ { N }$ , number of trees $B ,$ number of clusters $m ,$   
maximum depth $D _ { ; }$ , number of sampled candidates $k ,$ feature subset size $\phi _ { p } = \lfloor \sqrt { P } \rfloor$   
Ensure: Forest $\mathcal { F } = \{ T _ { b } \} _ { b = 1 } ^ { B }$   
1: Initialize: $\pmb { S } \gets \mathrm { D I C S } ( \mathbf { X } , m )$   
2: for $b = 1$ to B do   
3: Draw a bootstrap sample $( \mathbf { X } ^ { ( b ) } , \pmb { y } ^ { ( b ) } )$ of size $N$   
4: Sample a feature subset $\mathcal { T } _ { b } \subset [ P ]$ with $| { \mathcal { T } } _ { b } | = \phi _ { p }$   
5: Construct restricted dictionary $S _ { b } \gets \{ T _ { f } \in { \cal S } : f \in \mathcal { T } _ { b } \}$   
6: Grow tree $T _ { b } \gets \mathrm { C G C T } ( \mathbf { X } ^ { ( b ) } , \pmb { y } ^ { ( b ) } , \mathcal { S } _ { b } , D , k )$   
7: end for   
8: return $\mathcal { F } = \{ T _ { b } \} _ { b = 1 } ^ { B }$

The optimal leaf weights for each region $R _ { j }$ and each class c, obtained by minimizing (4), are

$$
w _ { j , c } ^ { * } = - \frac { \sum _ { i \in \mathcal { T } _ { j } } g _ { i c } } { \lambda + \sum _ { i \in \mathcal { T } _ { j } } h _ { i c } } , \quad \forall c = 1 , \dots , K .\tag{5}
$$

Let $\mathcal { T } _ { L }$ and $\mathcal { T } _ { R }$ denote the instance sets of the left and right child nodes induced by a candidate split $( f , \tau )$ , and let $\mathcal { I } = \mathcal { I } _ { L } \cup \mathcal { I } _ { R }$ . Then the loss reduction is given by

$$
Q ( f , \tau ) = \frac { 1 } { 2 } \sum _ { c = 1 } ^ { K } \left[ \frac { ( \sum _ { i \in \mathcal { T } _ { L } } g _ { i c } ) ^ { 2 } } { \lambda + \sum _ { i \in \mathcal { T } _ { L } } h _ { i c } } + \frac { ( \sum _ { i \in \mathcal { T } _ { R } } g _ { i c } ) ^ { 2 } } { \lambda + \sum _ { i \in \mathcal { T } _ { R } } h _ { i c } } - \frac { ( \sum _ { i \in \mathcal { T } } g _ { i c } ) ^ { 2 } } { \lambda + \sum _ { i \in \mathcal { T } } h _ { i c } } \right] - \gamma .\tag{6}
$$

which can be used as evaluating the quality of the candidate split $Q ( f , \tau )$

By using DICS and incorporating the Gradient-based One-Side Sampling (GOSS) technique [10] and column (feature) subsampling technique [9, 10], our Fast Classification Gradient Boosting Machine (FastC-GBM) is summarized in Algorithm 4.

## A.2 Technical Proofs

Proof of Theorem 1. This proof consists of two parts. In Part (i), we show that under the density regularity condition, $\delta = d _ { H } ( \mathcal { P } _ { f , \bar { \tau } _ { * } ^ { ( f ) } } , \mathcal { P } _ { f , \tilde { \tau } _ { * } ^ { ( f ) } } ) = \mathcal { O } _ { P } ( \sqrt { N } )$ . In Part (ii), we establish that the diference between the two optimal gain values is bounded.

Part (i). Fix a feature $f ,$ and for notational simplicity write $\tau : = \tau ^ { ( f ) }$ . We first establish the convergence rate of the empirical centroids $\hat { \mu } _ { c }$ to the population means $\mu _ { c } .$ . Assume the feature $f$ is bounded in $[ a , b ]$ . By Hoefding’s inequality, for each class c and any $\varepsilon > 0$

$$
\mathbb { P } \big ( | \hat { \mu } _ { c } - \mu _ { c } | \ge \varepsilon \big ) \le 2 \exp \left( - \frac { 2 n _ { c } \varepsilon ^ { 2 } } { ( b - a ) ^ { 2 } } \right) ,
$$

where $n _ { c }$ is the number of samples in class c. Since $n _ { c } \asymp N$ , it follows that with probability at least $1 - \gamma$

$$
| \hat { \mu } _ { c } - \mu _ { c } | \leq ( b - a ) \sqrt { \frac { \log ( 2 / \gamma ) } { 2 n _ { c } } } = \mathcal { O } \biggl ( \frac { 1 } { \sqrt { N } } \biggr ) .
$$

In DICS, the candidate split is defined as $\begin{array} { r } { \tilde { \tau } = \zeta ( \hat { \mu } _ { c _ { 1 } } , \hat { \mu } _ { c _ { 2 } } ) = \frac { w _ { 1 } \hat { \mu } _ { c _ { 1 } } + w _ { 2 } \hat { \mu } _ { c _ { 2 } } } { w _ { 1 } + w _ { 2 } } } \end{array}$ , where $w _ { 1 } = \hat { \sigma } _ { c _ { 2 } }$ and $w _ { 2 } = \hat { \sigma } _ { c _ { 1 } }$ . The mapping $\zeta$ is linear and thus Lipschitz continuous with constant $L = \| \nabla \zeta \| _ { 2 } =$ $\begin{array} { r } { \frac { \sqrt { w _ { 1 } ^ { 2 } + w _ { 2 } ^ { 2 } } } { w _ { 1 } + w _ { 2 } } \leq 1 } \end{array}$ . Hence, $\begin{array} { r } { \left| { \tilde { \tau } } - \tau ^ { * } \right| \leq L \sum _ { c } \left| \hat { \mu } _ { c } - \mu _ { c } \right| = { \mathcal { O } } _ { P } \left( \frac { 1 } { \sqrt { N } } \right) } \end{array}$

Algorithm 4 Fast Classification Gradient Boosting Machine (FastC-GBM)   
Require: Feature matrix $\mathbf { X } \in \mathbb { R } ^ { P \times N }$ , labels $y \in [ K ] ^ { N }$ , split-candidate dictionary S (Alg. 1), number   
of boosting rounds $T ,$ maximum depth $D ,$ number of sampled candidates k, learning rate $\eta ,$   
GOSS large-gradient size $N _ { a } = \lfloor a \cdot N \rfloor$ , GOSS small-gradient sampling size $N _ { a } = \lfloor b \cdot N \rfloor$ , column   
(feature) subsampling size $N _ { f } = \lfloor c \cdot P \rfloor$   
Ensure: Additive multiclass boosted model $F _ { T } ( { \pmb x } )$   
1: Initialize: $F _ { 0 , c } ( \pmb { x } _ { i } ) = \log ( \hat { \pi } _ { c } ) , \forall c \in [ K ]$ , where $\hat { \pi } _ { c }$ is the empirical class proportion.   
2: for $t = 1$ to $T$ do   
3: (GOSS step)   
4: Update $g _ { i c } \gets p _ { i c } ^ { ( t - 1 ) } - \mathbb { I } ( y _ { i } = c ) , \quad \forall i \in [ N ] , \forall c \in [ K ] .$   
5: Update $h _ { i c }  p _ { i c } ^ { ( t - 1 ) } ( 1 - p _ { i c } ^ { ( t - 1 ) } ) , \quad \forall i \in [ N ] , \forall c \in [ K ] .$   
6: $\begin{array} { r } { A \gets \{ i \in [ N ] \ | \ i \in \mathrm { T o p } { - } ( N _ { a } ) \{ \| g _ { 1 } \| _ { 2 } , \dots , \| g _ { N } \| _ { 2 } \} \} . } \end{array}$   
7: Sample a subset $B \subset [ N ] \backslash \mathcal { A }$ with $| B | = N _ { b }$   
8: Update $\begin{array} { r } { g _ { i c } \gets \frac { 1 - a } { b } g _ { i c } , \quad \forall i \in \mathcal { B } , \forall c \in [ K ] } \end{array}$   
9: Update $\begin{array} { r } { h _ { i c }  \frac { 1 - a } { b } h _ { i c } , } \end{array}$ $\forall i \in B , \forall c \in [ K ]$   
10: (Split selection)   
11: Sample a subset $\mathcal { F } _ { \mathrm { s u b } } \subset [ P ]$ with $| \mathcal { F } _ { \mathrm { s u b } } | = N _ { f }$   
12: Sample a subset ${ \mathcal { C } } \subset S ^ { \prime } = \{ ( f , \tau ) \in S : f \in { \dot { \mathcal { F } } } _ { \mathrm { s u b } } \}$ with $| { \mathcal { C } } | = k$ (without replacement)   
13: $( f ^ { * } , \tau ^ { * } )  \arg \operatorname* { m i n } _ { ( f , \tau ) \in { \mathcal { C } } } Q ( f , \tau )$ where $Q ( f , \tau )$ is evaluated over $\mathcal { A } \cup \mathcal { B }$ using (6).   
14: $S \gets S / \{ ( f ^ { * } , \tau ^ { * } ) \}$   
15: Grow one leaf-wise tree $f _ { t }$ from the split $( f ^ { * } , \tau ^ { * } )$   
16: Update model $F _ { t } ( \pmb { x } _ { i } )  F _ { t - 1 } ( \pmb { x } _ { i } ) + \eta f _ { t } ( \pmb { x } _ { i } )$   
17: end for   
18: return $F _ { T } ( { \pmb x } )$

Let $\tau ^ { * }$ and $\bar { \tau } ^ { * }$ denote the population and empirical minimizers of the Gini impurity function $Q ( \tau )$ and $Q _ { N } ( \tau )$ , respectively. For a fixed feature $f ,$ the split is determined by the indicator $\mathbb { I } \{ X ^ { ( f ) } < \bar { \tau } \}$ which defines a step function with a discontinuity at $\tau$ . Hence, the empirical objective is piecewise constant and its maximizer lies on a grid of order statistics of size $1 / N$ . Standard results from change-point estimation imply $\begin{array} { r } { | \tau ^ { * } - \bar { \tau } ^ { * } | = \mathcal { O } _ { P } \big ( \frac { 1 } { N } \big ) } \end{array}$

Therefore, $\begin{array} { r } { | \tilde { \tau } - \bar { \tau } ^ { * } | \leq | \tilde { \tau } - \tau ^ { * } | + | \tau ^ { * } - \bar { \tau } ^ { * } | = \mathcal { O } _ { P } \Big ( \frac { 1 } { \sqrt { N } } \Big ) } \end{array}$ , which implies that $\begin{array} { r } { | \tilde { \tau } ^ { * } - \bar { \tau } ^ { * } | = \mathcal { O } _ { P } \Big ( \frac { 1 } { \sqrt { N } } \Big ) } \end{array}$ Define the Hamming distance

$$
\delta = \sum _ { i = 1 } ^ { N } \mathbb { I } \left\{ X _ { i } ^ { ( f ) } \in [ \operatorname* { m i n } ( \bar { \tau } ^ { * } , \tilde { \tau } ^ { * } ) , \operatorname* { m a x } ( \bar { \tau } ^ { * } , \tilde { \tau } ^ { * } ) ] \right\} .
$$

Assume the marginal density $p ( x ^ { ( f ) } )$ is bounded by M in a neighborhood of $\tau ^ { * }$ . Then

$$
\mathbb { E } [ \delta ] = N \int _ { \operatorname* { m i n } ( \bar { \tau } ^ { * } , \tilde { \tau } ^ { * } ) } ^ { \operatorname* { m a x } ( \bar { \tau } ^ { * } , \tilde { \tau } ^ { * } ) } p ( x ^ { ( f ) } ) d x \leq N M | \tilde { \tau } ^ { * } - \bar { \tau } ^ { * } | .
$$

Using $| \tilde { \tau } ^ { * } - \bar { \tau } ^ { * } | = \mathcal { O } _ { P } ( N ^ { - 1 / 2 } )$ , we obtain $\mathbb { E } [ \delta ] = \mathcal { O } ( \sqrt { N } )$ . By Markov’s inequality, for any $\varepsilon > 0$ $\begin{array} { r } { \mathbb { P } ( \delta > \varepsilon \sqrt { N } ) \le \frac { \mathbb { E } [ \delta ] } { \varepsilon \sqrt { N } } \to 0 , } \end{array}$ , as $N \to \infty$ . Hence, $\delta = \mathcal { O } _ { P } ( \sqrt { N } )$ , which completes the proof.

Part (ii). For notational simplicity, we write $\mathcal { P } = ( \mathcal { I } _ { L } , \mathcal { I } _ { R } )$ to denote the partition $\mathcal { P } _ { f , \bar { \tau } _ { * } ^ { ( f ) } }$ , and $\mathcal { P } ^ { \prime } = ( \mathbb { Z } _ { L } ^ { \prime } , \mathbb { Z } _ { R } ^ { \prime } )$ to denote the partition $\mathcal { P } _ { f , \tilde { \tau } _ { * } ^ { ( f ) } }$ . Then $\mathcal { P }$ has sizes $n _ { L } = \left| \mathcal { T } _ { L } \right|$ and $n _ { R } = \left. \mathcal { T } _ { R } \right.$ . The class counts within each partition are defined as $n _ { L , c } = | \{ i \in \mathbb { Z } _ { L } : y _ { i } = c \} |$ and $n _ { R , c } = | \{ i \in \mathcal { I } _ { R } : y _ { i } = c \}$ for each class $c \in [ K ]$

If $d _ { H } ( \mathcal { P } , \mathcal { P } ^ { \prime } ) = \delta$ , then there are $\delta$ samples are reassigned, either from $\mathcal { T } _ { L }$ to $\mathcal { T } _ { R }$ or vice versa. More precisely, let $\delta _ { 1 }$ denote the number of samples moving from $\mathcal { I } _ { L }$ to $\mathcal { T } _ { R }$ (referred to as $l e f t { - } t o { - } r i g h t )$ , and $\delta _ { 2 }$ the number moving from $\mathcal { T } _ { R }$ to $\mathcal { T } _ { L }$ (referred to as right-to-left), so that $\delta _ { 1 } + \delta _ { 2 } = \delta$ . Accordingly, the subset sizes become $n _ { L } ^ { \prime } = n _ { L } - \delta _ { 1 } + \delta _ { 2 }$ and $n _ { R } ^ { \prime } = n _ { R } + \delta _ { 1 } - \delta _ { 2 }$

We next examine the class counts in the left and right subsets. For each left-to-right move, there exists a class c such that $n _ { L , c }  n _ { L , c } - 1$ and $n _ { R , c }  n _ { R , c } + 1$ . Similarly, for each right-to-left move, there exists a class c such that $n _ { L , c }  n _ { L , c } + 1$ and $n _ { R , c }  n _ { R , c } - 1$

For a set $\mathcal { T }$ with $| { \mathcal { T } } | = n$ , we consider the Gini index in the form $\begin{array} { r } { G ( \mathcal { T } ) = 1 - \sum _ { c = 1 } ^ { K } \frac { n _ { c } ^ { 2 } } { n ^ { 2 } } } \end{array}$ , where $n _ { c }$ denotes the number of samples in I that belong to class c. Then the objective can be rewritten as

$$
Q ( \mathcal { P } ) = \frac { n _ { L } } { N } G ( \mathbb { Z } _ { L } ) + \frac { n _ { R } } { N } G ( \mathbb { Z } _ { R } ) = 1 - \frac { 1 } { N } \left( \frac { \sum _ { c = 1 } ^ { K } n _ { L , c } ^ { 2 } } { n _ { L } } + \frac { \sum _ { c = 1 } ^ { K } n _ { R , c } ^ { 2 } } { n _ { R } } \right)
$$

Hence the diference of two gains of $\mathcal { P }$ and ${ \mathcal { P } } ^ { \prime }$ can be found by

$$
Q ( \mathcal { P } ^ { \prime } ) - Q ( \mathcal { P } ) = \frac { 1 } { N } \left( \frac { \sum _ { c = 1 } ^ { K } ( n _ { L , c } ^ { \prime } ) ^ { 2 } } { n _ { L } ^ { \prime } } - \frac { \sum _ { c = 1 } ^ { K } n _ { L , c } ^ { 2 } } { n _ { L } } + \frac { \sum _ { c = 1 } ^ { K } ( n _ { R , c } ^ { \prime } ) ^ { 2 } } { n _ { R } ^ { \prime } } - \frac { \sum _ { c = 1 } ^ { K } n _ { R , c } ^ { 2 } } { n _ { R } } \right)\tag{7}
$$

Let $\begin{array} { r } { S _ { L } = \sum _ { c = 1 } ^ { K } n _ { L , c } ^ { 2 } } \end{array}$ and $\begin{array} { r } { S _ { R } = \sum _ { c = 1 } ^ { K } n _ { R , c } ^ { 2 } , } \end{array}$ and define $S _ { L } ^ { \prime } , S _ { R } ^ { \prime }$ analogously. Denote: $\begin{array} { r } { I _ { 1 } = \frac { S _ { L } ^ { \prime } } { n _ { L } ^ { \prime } } - \frac { S _ { L } } { n _ { L } } , I _ { 2 } = } \end{array}$ $\begin{array} { r } { \frac { S _ { R } ^ { \prime } } { n _ { R } ^ { \prime } } - \frac { S _ { R } } { n _ { R } } } \end{array}$ . We first bound $I _ { 1 }$ . Write

$$
I _ { 1 } = \frac { S _ { L } ^ { \prime } } { n _ { L } ^ { \prime } } - \frac { S _ { L } } { n _ { L } } = \frac { S _ { L } ^ { \prime } - S _ { L } } { n _ { L } } + S _ { L } ^ { \prime } \left( \frac { 1 } { n _ { L } ^ { \prime } } - \frac { 1 } { n _ { L } } \right) .
$$

Each moved sample changes exactly one class count by $\pm 1$ , so that $( n _ { L , c } ^ { \prime } ) ^ { 2 } - n _ { L , c } ^ { 2 } = \pm 2 n _ { L , c } + 1$ which contributes at most $| ( n _ { L , c } ^ { \prime } ) ^ { 2 } - n _ { L , c } ^ { 2 } | \leq 2 n _ { L } + 1 < 3 n _ { L }$ . So for $\delta$ samples are moved, so that

$$
| S _ { L } ^ { \prime } - S _ { L } | = \left| \sum _ { c = 1 } ^ { K } ( n _ { L , c } ^ { \prime } ) ^ { 2 } - \sum _ { c = 1 } ^ { K } n _ { L , c } ^ { 2 } \right| \leq \sum _ { c = 1 } ^ { K } | ( n _ { L , c } ^ { \prime } ) ^ { 2 } - n _ { L , c } ^ { 2 } | < 3 \delta n _ { L }\tag{8}
$$

Moreover, one can easily show that $S _ { L } ^ { \prime } \leq n _ { L } ^ { \prime 2 }$ . Combining the inequalities $S _ { L } ^ { \prime } \leq n _ { L } ^ { \prime 2 }$ and $| n _ { L } ^ { \prime } - n _ { L } | \leq \delta$ into (8) yields

$$
| I _ { 1 } | \le \frac { | S _ { L } ^ { \prime } - S _ { L } | } { n _ { L } } + \frac { | n _ { L } ^ { \prime } - n _ { L } | S _ { L } ^ { \prime } } { n _ { L } n _ { L } ^ { \prime } } < 3 \delta + \delta \frac { n _ { L } ^ { \prime } } { n _ { L } } = ( 4 + \frac { \delta } { n _ { L } } ) \delta .
$$

Similarly, we can show that $\begin{array} { r } { | I _ { 2 } | \le ( 4 + \frac { \delta } { n _ { R } } ) \delta } \end{array}$ . Substituting the inequalities of $I _ { 1 }$ and $I _ { 2 }$ into $( 7 )$ gives

$$
| Q ( \mathcal { P } ) - Q ( \mathcal { P } ^ { \prime } ) | < ( 8 + \frac { \delta } { n _ { L } } + \frac { \delta } { n _ { R } } ) \frac { \delta } { N } \le ( 8 + \frac { 1 } { \alpha ( 1 - \alpha ) } \cdot \frac { \delta } { N } ) \frac { \delta } { N } ,\tag{9}
$$

where $\begin{array} { r } { \alpha = \operatorname* { m i n } _ { f } \{ \frac { n _ { L } ^ { ( f ) } } { N } , \frac { n _ { R } ^ { ( f ) } } { N } \} } \end{array}$ . Finally, since the above bound holds uniformly over $f ,$ , we have

$$
\begin{array} { l } { \displaystyle \bar { Q } ^ { * } = \operatorname* { m i n } _ { f } Q ( f , \bar { \tau } _ { * } ^ { ( f ) } ) \leq \operatorname* { m i n } _ { f } Q ( f , \tilde { \tau } _ { * } ^ { ( f ) } ) + ( 8 + \displaystyle \frac { 1 } { \alpha ( 1 - \alpha ) } \cdot \frac { \delta } { N } ) \frac { \delta } { N } } \\ { \displaystyle \qquad = \tilde { Q } ^ { * } + ( 8 + \displaystyle \frac { 1 } { \alpha ( 1 - \alpha ) } \cdot \frac { \delta } { N } ) \frac { \delta } { N } } \end{array}
$$

and similarly,

$$
\tilde { Q } ^ { * } \le \bar { Q } ^ { * } + ( 8 + \frac { 1 } { \alpha ( 1 - \alpha ) } \cdot \frac { \delta } { N } ) \frac { \delta } { N } .
$$

Here we use the fact that $\begin{array} { r } { \bar { Q } ^ { * } = \operatorname* { m i n } _ { f } Q ( f , \bar { \tau } _ { * } ^ { ( f ) } ) = \operatorname* { m i n } _ { ( f , \tau ) \in S _ { \mathrm { D T } } } Q ( f , \tau ) } \end{array}$ and $\begin{array} { r } { \tilde { Q } ^ { * } = \operatorname* { m i n } _ { f } Q ( f , \tilde { \tau } _ { * } ^ { ( f ) } ) = } \end{array}$ $\begin{array} { r } { \operatorname* { m i n } _ { ( f , \tau ) \in S _ { \mathrm { D I C S } } } Q ( f , \tau ) } \end{array}$ . Combining the two inequalities above completes the proof.

## A.3 Parameter sensitivity

![](images/6f284cb906508763f340b4bea766329120f06e8540d4d144a9945ada497254aa.jpg)  
(a) Test accuracy vs. maximum number of cluster pairs (m) on CIFAR-10.

![](images/c251548fee2240b46735e28d51d2000216146c736d7a09e97e1b043365db2065.jpg)  
(b) Test accuracy vs. maximum number of featurethreshold pairs (k) on CIFAR-10.

![](images/eb387f6cd6180ec855b6313831aef223227e893b6f2c45b01b8db9ea69ea3ed4.jpg)  
(c) Test accuracy vs. maximum tree depth (D) on CIFAR-10.

![](images/45cf9d780100754f28321fd69fa1a1b5bbc8970a9c300638ba5091bf3424941d.jpg)  
(d) Test accuracy vs. mini-batch size (b) on CIFAR-10.  
Figure 2: Parameter sensitivity of CGCT on CIFAR-10 with respect to (a) the maximum number of cluster pairs (m), (b) the maximum number of feature-threshold pairs (k), (c) the maximum tree depth (D), and (d) the mini-batch size (b). Each point represents the mean test accuracy over 50 runs, and error bars denote ±3 standard deviations.

We analyze the sensitivity of CGCT to four key parameters: the maximum number of cluster pairs m from Algorithm 1, the length of the feature-threshold set k where $| { \mathcal { C } } | = k$ , the maximum tree depth D, and the mini-batch size b used in mini-batch K-means. These parameters control the number of candidate splits considered during tree construction, the complexity of the resulting tree, and the computational cost of candidate generation.

Figure 2(a) shows that increasing m improves accuracy at first, but the performance quickly stabilizes after a moderate value. This suggests that a small number of cluster pairs is suficient to capture useful split candidates, while additional pairs provide limited benefit.

Figure 2(b) illustrates a similar trend for k. Initially, accuracy increases as more feature-threshold pairs are considered. However, beyond a specific threshold, the improvement diminishes significantly, and the curve gradually flattens. This suggests that once a suficient number of informative candidates is included, adding more feature-threshold pairs yields only marginal benefit.

Figure 2(c) shows the efect of the maximum tree depth D. Accuracy improves substantially as the depth increases from shallow trees and gradually stabilizes at larger depths. The highest accuracy is obtained around $D = 8 – 1 0$ , while increasing the depth further provides no additional improvement and results in a slight decrease at $D = 1 2$ . This indicates that a moderate tree depth is suficient to capture the useful structure in the candidate splits without requiring unnecessarily deep trees.

Finally, Figure $2 ( \mathrm { d } )$ shows that CGCT is highly stable with respect to the mini-batch size b. Across a wide range of mini-batch sizes, from 32 to 2048, the test accuracy remains nearly unchanged. This suggests that the candidate splits generated by DICS are not highly sensitive to the mini-batch size used in K-means, and relatively small mini-batches are suficient to obtain stable predictive performance.

Overall, these results show that CGCT is not highly sensitive to large values of the aforementioned parameters. Moderate parameter choices are suficient to obtain stable accuracy while maintaining eficient split generation and tree construction.

## A.4 Empirical Validation of DICS Split Quality

We further evaluate the split quality of DICS from both tree-level and split-level perspectives. First, we independently grow exhaustive DT and CGCT on CIFAR-10 with maximum depth $D = 8$ allowing each method to select its own feature–threshold pair throughout tree construction. Table 7 reports the mean local gain, sample-weighted impurity reduction, and cumulative gain across depth. Overall, CGCT closely follows the impurity reduction achieved by exhaustive DT despite the two trees being grown independently. At the final splitting depth, CGCT achieves a cumulative gain of 0.123, compared with 0.127 for exhaustive DT, corresponding to 97.18% of the cumulative gain achieved by exhaustive split search.

We additionally examine the split-level behavior in Remark 1 using the synthetic-data setting by varying the sample size from N = 1,000 to $N = 2 0 { , } 0 0 0$ . For each sample size, DICS and exhaustive DT independently select their best root-level feature–threshold pair, and the results are averaged over 40 independent runs. As shown in Table 8, the mean gain gap $\Delta _ { G }$ decreases from 0.0039 to 0.0011 as N increases, while $\sqrt { N } \Delta _ { G }$ remains of comparable magnitude across the evaluated sample sizes. This behavior is consistent with the $O _ { P } ( N ^ { - 1 / 2 } )$ split-level rate in Remark 1 and provides additional empirical evidence that the optimal DICS gain approaches that of exhaustive DT as the sample size increases.

## A.5 Additional Experiments

This section presents additional experiments on synthetic data with $\rho = 0$ , where the generated features are independent of each other, as well as descriptions of the real datasets used in this paper. Additional comparisons of various algorithms are reported in Table 9–11. A summary of the real datasets is provided in Table 12.

<table><tr><td></td><td colspan="2">Mean Local Gain  $\bar { g } _ { d }$ </td><td colspan="2">Weighted Gain  ${ G } _ { d } ^ { ( w ) }$ </td><td colspan="2">Cumulative Gain  $C _ { d }$ </td><td rowspan="2"> $R _ { d }$  (%)</td></tr><tr><td>Depth</td><td>DT</td><td>CGCT</td><td>DT</td><td>CGCT</td><td>DT</td><td>CGCT</td></tr><tr><td>1</td><td>0.019</td><td>0.019</td><td>0.019</td><td>0.019</td><td>0.019</td><td>0.019</td><td>100.00</td></tr><tr><td>2</td><td>0.013</td><td>0.012</td><td>0.012</td><td>0.012</td><td>0.031</td><td>0.031</td><td>99.46</td></tr><tr><td>3</td><td>0.015</td><td>0.014</td><td>0.013</td><td>0.013</td><td>0.044</td><td>0.043</td><td>98.04</td></tr><tr><td>4</td><td>0.015</td><td>0.015</td><td>0.013</td><td>0.013</td><td>0.057</td><td>0.057</td><td>99.37</td></tr><tr><td>5</td><td>0.018</td><td>0.017</td><td>0.013</td><td>0.013</td><td>0.070</td><td>0.070</td><td>99.40</td></tr><tr><td>6</td><td>0.023</td><td>0.021</td><td>0.015</td><td>0.014</td><td>0.085</td><td>0.084</td><td>98.58</td></tr><tr><td>7</td><td>0.037</td><td>0.030</td><td>0.018</td><td>0.017</td><td>0.103</td><td>0.101</td><td>97.71</td></tr><tr><td>8</td><td>0.061</td><td>0.051</td><td>0.023</td><td>0.022</td><td>0.127</td><td>0.123</td><td>97.18</td></tr></table>

Table 7: Depth-wise impurity reduction of independently grown exhaustive DT and CGCT on CIFAR-10 with maximum depth $D = 8 .$ , where Depth 1 denotes the root level. For depth $d ,$ $\begin{array} { r } { \bar { g } _ { d } = | \mathcal { I } _ { d } | ^ { - 1 } \sum _ { j \in \mathcal { T } _ { d } } g _ { j } } \end{array}$ is the mean local Gini gain, $\begin{array} { r } { G _ { d } ^ { ( w ) } = \sum _ { j \in \mathcal { T } _ { d } } ( n _ { j } / N ) g _ { j } } \end{array}$ is the sample-weighted impurity reduction, $\begin{array} { r } { C _ { d } = \sum _ { r = 1 } ^ { d } G _ { r } ^ { ( w ) } } \end{array}$ is the cumulative weighted gain, and $R _ { d } = C _ { d } ^ { \mathrm { C G C T } } / C _ { d } ^ { \mathrm { D T } }$ is the cumulative gain retained by CGCT relative to exhaustive DT.

<table><tr><td>N</td><td>Gain Gap  $\Delta _ { G }$   $\sqrt { N } \Delta _ { G }$ </td></tr><tr><td>1,000 0.0039</td><td>0.1227</td></tr><tr><td>2,000</td><td>0.0025 0.1100</td></tr><tr><td>5,000 0.0017</td><td>0.1183</td></tr><tr><td>10,000</td><td>0.0012 0.1242</td></tr><tr><td>20,000 0.0011</td><td>0.1492</td></tr></table>

Table 8: Split-level gain gap between DICS and exhaustive DT across increasing sample sizes. Results are averaged over 40 independent runs.

<table><tr><td rowspan="2"> $( N , P )$ </td><td rowspan="2">Class</td><td colspan="3">Time  $( \sec ) \downarrow$ </td><td>Speedup↑</td><td colspan="3">ACC↑</td></tr><tr><td>DT</td><td>BDTKS</td><td>CGCT</td><td> $t _ { \mathrm { D T } } / t _ { \mathrm { C G C T } }$ </td><td>DT</td><td>BDTKS</td><td>CGCT</td></tr><tr><td rowspan="3">10000,1000</td><td>3</td><td>2.59 (0.02)</td><td>31.84</td><td>0.18 (0.02)</td><td>14.39</td><td>0.67 (0.00)</td><td>0.53</td><td>0.65 (0.00)</td></tr><tr><td>5</td><td>2.49 (0.00)</td><td>35.54</td><td>0.27 (0.02)</td><td>9.22</td><td>0.50 (0.00)</td><td>0.31</td><td>0.49 (0.00)</td></tr><tr><td>10</td><td>2.64 (0.00)</td><td>38.21</td><td>0.30 (0.02)</td><td>8.80</td><td>0.46 (0.00)</td><td>0.30</td><td>0.44 (0.01)</td></tr><tr><td rowspan="3">5000,7000</td><td>3</td><td>8.15 (0.09)</td><td>29.65</td><td>0.63 (0.03)</td><td>12.94</td><td>0.69 (0.00)</td><td>0.62</td><td>0.68 (0.01)</td></tr><tr><td>5</td><td>8.56 (0.04)</td><td>29.83</td><td>0.92 (0.05)</td><td>9.30</td><td>0.39 (0.00)</td><td>0.30</td><td>0.37 (0.01)</td></tr><tr><td>10</td><td>8.67 (0.02)</td><td>33.40</td><td>1.20 (0.05)</td><td>7.22</td><td>0.34 (0.00)</td><td>0.24</td><td>0.31 (0.01)</td></tr><tr><td rowspan="3">20000,5000</td><td>3</td><td>26.50 (0.33)</td><td>118.76</td><td>1.20 (0.10)</td><td>22.08</td><td>0.47 (0.00)</td><td>0.45</td><td>0.47 (0.00)</td></tr><tr><td>5</td><td>29.46 (0.04)</td><td>125.70</td><td>1.69 (0.01)</td><td>17.43</td><td>0.53 (0.00)</td><td>0.41</td><td>0.51 (0.00)</td></tr><tr><td>10</td><td>29.53 (0.04)</td><td>131.71</td><td>2.07 (0.07)</td><td>14.26</td><td>0.50 (0.00)</td><td>0.34</td><td>0.48 (0.00)</td></tr></table>

Table 9: Accuracy and training time (in seconds) for DT, BDTKS, and CGCT in real datasets. The ratio $t _ { \mathrm { D T } } / t _ { \mathrm { C G C T } }$ quantifies the training time improvement of CGCT relative to DT. Here $\rho = 0$

<table><tr><td rowspan="2">(N, P)</td><td rowspan="2">Class</td><td colspan="2">Time (sec)↓</td><td>Speedup↑</td><td colspan="2">ACC↑</td></tr><tr><td>RF</td><td>CGRF</td><td> $t _ { \mathrm { R F } } / t _ { \mathrm { C G R F } }$ </td><td>RF</td><td>CGRF</td></tr><tr><td rowspan="3">10000,1000</td><td>3</td><td>4.16 (0.04)</td><td>0.34 (0.02)</td><td>12.23</td><td>0.70 (0.00)</td><td>0.70 (0.00)</td></tr><tr><td>5</td><td>4.13 (0.05)</td><td>0.39 (0.00)</td><td>10.59</td><td>0.54 (0.00)</td><td>0.54 (0.01)</td></tr><tr><td>10</td><td>4.54 (0.00)</td><td>0.50 (0.02)</td><td>9.08</td><td>0.47 (0.00)</td><td>0.47 (0.00)</td></tr><tr><td rowspan="3">5000,7000</td><td>3</td><td>13.67 (0.46)</td><td>0.86 (0.07)</td><td>15.89</td><td>0.73 (0.00)</td><td>0.73 (0.01)</td></tr><tr><td>5</td><td>14.13 (0.15)</td><td>1.22 (0.03)</td><td>11.58</td><td>0.44 (0.01)</td><td>0.44 (0.00)</td></tr><tr><td>10</td><td>14.51 (0.16)</td><td>1.71 (0.07)</td><td>8.48</td><td>0.36 (0.01)</td><td>0.36 (0.01)</td></tr><tr><td rowspan="3">20000,5000</td><td>3</td><td>45.35 (1.86)</td><td>1.52 (0.05)</td><td>29.83</td><td>0.51 (0.00)</td><td>0.50 (0.01)</td></tr><tr><td>5</td><td>57.20 (1.31)</td><td>1.71 (0.08)</td><td>33.45</td><td>0.55 (0.00)</td><td>0.55 (0.00)</td></tr><tr><td>10</td><td>54.83 (0.24)</td><td>2.09 (0.04)</td><td>26.23</td><td>0.51 (0.00)</td><td>0.51 (0.00)</td></tr></table>

Table 10: Accuracy and training time (in seconds) for RF and CGRF in synthetic data. The ratio t /t<sub>CG</sub> quantifies the training time improvement of CGRF relative to RF. Here $\rho = 0$

<table><tr><td rowspan="2">(N, P)</td><td rowspan="2">Class</td><td colspan="3">Time (sec)↓</td><td colspan="2">Speedup↑</td><td colspan="4">ACC↑</td></tr><tr><td>XGBoost</td><td>LightGBM</td><td>CGXGB</td><td> $\rho _ { 1 }$ </td><td> $\rho _ { 1 }$ </td><td>XGBoost</td><td></td><td>LightGBM</td><td>CGXGB</td></tr><tr><td rowspan="3">10000,1000</td><td>3</td><td>1.43 (0.01)</td><td>0.97 (0.00)</td><td>0.30 (0.01)</td><td>4.77</td><td>3.23</td><td></td><td>0.71 (0.01)</td><td>0.71 (0.01)</td><td>0.70 (0.01)</td></tr><tr><td>5</td><td>2.36 (0.00)</td><td>1.59 (0.00)</td><td>0.37 (0.00)</td><td>6.38</td><td>4.30</td><td></td><td>0.54 (0.00)</td><td>0.55 (0.01)</td><td>0.55 (0.01)</td></tr><tr><td>10</td><td>4.50 (0.01)</td><td>2.93 (0.01)</td><td>0.44</td><td>(0.01)</td><td>10.23</td><td>6.66 0.48</td><td>(0.01)</td><td>0.47 (0.02)</td><td>0.45 (0.01)</td></tr><tr><td rowspan="3">5000,7000</td><td>3</td><td>6.73 (0.05)</td><td>3.13 (0.06)</td><td>0.24</td><td>(0.01)</td><td>28.04 13.04</td><td>0.75</td><td>(0.02)</td><td>0.75 (0.02)</td><td>0.74 (0.01)</td></tr><tr><td>5</td><td>10.81 (0.23)</td><td>4.80 (0.15)</td><td>0.29</td><td>(0.00)</td><td>37.28</td><td>16.55 0.41</td><td>(0.02)</td><td>0.41 (0.02)</td><td>0.38 (0.01)</td></tr><tr><td>10</td><td>18.16 (0.46)</td><td>9.30 (0.59)</td><td>0.37</td><td>(0.01) 49.08</td><td>25.13</td><td>0.35</td><td>(0.01)</td><td>0.34 (0.01)</td><td>0.33 (0.01)</td></tr><tr><td rowspan="3">20000,5000</td><td>3</td><td>9.40 (0.24)</td><td>5.73 (0.17)</td><td>0.68</td><td>(0.00)</td><td>13.82</td><td>8.43</td><td>0.54 (0.01)</td><td>0.55 (0.01)</td><td>0.52 (0.01)</td></tr><tr><td>5</td><td>15.06 (0.19)</td><td>8.13 (0.08)</td><td>0.77</td><td>(0.00)</td><td>19.56 10.56</td><td>0.55</td><td>(0.01)</td><td>0.55 (0.01)</td><td>0.54 (0.01)</td></tr><tr><td>10</td><td>31.04 (0.72)</td><td>15.10 (0.35)</td><td>1.09</td><td>(0.03)</td><td>28.48</td><td>13.85 0.51</td><td>(0.01)</td><td>0.51 (0.01)</td><td>0.50 (0.01)</td></tr></table>

Table 11: Accuracy and training time (in seconds) for XGBoost, LightGBM and FastC-GBM in synthetic data. The ratios $\rho _ { 1 } = t _ { \mathrm { X G B o o s t } } / t _ { \mathrm { F a s t C } }$ <sub>−GBM</sub> and $\rho _ { 2 } = t _ { \mathrm { L i g h t G B M } } / t _ { \mathrm { F a s t C } } .$ <sub>−GBM</sub> quantify the training time improvement of FastC-GBM relative to XGBoost and LightGBM, respectively. Here $\rho = 0$

<table><tr><td>Dataset</td><td>N</td><td>P</td><td>K</td></tr><tr><td>Helena</td><td>65,196</td><td>27</td><td>100</td></tr><tr><td>Spambase</td><td>4,601</td><td>57</td><td>2</td></tr><tr><td>Santander</td><td>200,000</td><td>200</td><td>2</td></tr><tr><td>CIFAR-10</td><td>60,000</td><td>3,072</td><td>10</td></tr><tr><td>MNIST</td><td>70,000</td><td>784</td><td>10</td></tr><tr><td>Fashion-MNIST</td><td>70,000</td><td>784</td><td>10</td></tr></table>

Table 12: Summary of datasets used in the experiments, including sample size (N), number of features (P), and number of classes (K).