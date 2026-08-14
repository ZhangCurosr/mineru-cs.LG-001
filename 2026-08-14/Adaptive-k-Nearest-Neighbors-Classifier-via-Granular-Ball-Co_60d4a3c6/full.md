# Adaptive k Nearest Neighbors Classifier via Granular Ball Computing

Xiaoyu Lian, Shuyin Xia\*, Hongxuan He, Lifeng Shen, Guoyin Wang, Xinbo Gao

Abstract—The k-Nearest Neighbor (KNN) algorithm is widely used across various tasks. The selection of the k value is a key issue because it significantly impacts performance. In this paper, an adaptive and efficient KNN approach via granularball computing is proposed. The method consists of two stages. In the training stage, the dataset is first coarsely partitioned to reduce the complexity of data distributions within a granular ball, and then the Fisher criterion is introduced to control ball splitting and stopping, yielding a multi-granularity granular ball representation. In the prediction stage, the nearest granular ball is first located through a weighted distance mechanism, and an adaptive neighborhood is then constructed around the test sample. The effective k value is dynamically determined by the actual number of samples contained in this neighborhood. The neighborhood induced by the nearest granular ball provides more stable local group information, thereby improving robustness against noise and local perturbations. Experimental results demonstrate that the proposed method outperforms existing KNN variants across multiple datasets in terms of both accuracy and efficiency. The code has been open-sourced for reproducibility: https://github.com/lianxiaoyu724/Adaptive-GBKNN.

Index Terms—KNN, Granular-Ball Computing, Granular Ball Splitting, Classification, Adaptivity.

## I. INTRODUCTION

supervised learning algorithm [1]. It computes Euclidean distances from the test instance to all training samples, subsequently identifying the k nearest neighbors for final classification through majority voting on their class labels. It is widely applied in applications such as speech recognition [2], pattern recognition [3]–[7], classification and clustering tasks [8]– [12]. However, the algorithm still has two limitations. In high-dimensional and large-scale datasets, the cost of distance computation and sorting increases sharply, leading to unacceptable computational complexity [13]. Moreover, the choice of k affects classification accuracy: a small k may cause overfitting, while a large k tends to include noisy samples, thereby compromising the robustness of the model.

The selection of the k value of the KNN algorithm plays a critical role in determining classification accuracy. As shown in Fig. 1(a), within the local distribution of Class 2, a few Class 1 samples can be regarded as noise. Setting $k \mathit { \Theta } = \mathit { \Theta } 3$ may cause the test sample to be misclassified as “Class 1.” Therefore, in the presence of noise, a larger k (e.g., k = 15) should be selected to enhance model robustness. Conversely, when the test sample lies near the decision boundary, as shown in Fig. 1(b), a smaller k is preferable to better capture local structural characteristics and improve classification precision. Currently, improved KNN algorithms that estimate the optimal k value [14] or use distance metrics [15], [16] have been widely applied in various research fields. The concept of neighborhood search has become a significant development direction for KNN. In 2007, Yiu et al. extended the traditional reverse nearest neighbor query by introducing the concept of reverse k-nearest neighbor (RKNN) search [17]. However, like traditional KNN, RKNN is also susceptible to the choice of k. The core challenge lies in effectively detecting the neighborhood structure for different datasets, which is especially difficult when the data characteristics are unknown.

![](images/46394cc09a8f7c912737afb51533e3efd09ecab7829991461d51136ce4d7c6da.jpg)  
Fig. 1: The effectiveness of k selection on KNN classification, in which the test sample should be classified to Class 2.

Regarding this challenge, the natural neighbor of an interesting idea was proposed, which has triggered extensive research [18]. Natural neighbors represent a scale-free neighbor relationship that better reflects the true characteristics of the data. Two data points, x and y, form a natural neighbor relationship if x considers y as a neighbor and y likewise considers x as a neighbor [19]. If each object in a dataset can find its natural neighbor, the dataset reaches a stable state. The natural neighborhood provides an adaptive alternative to the preset k in traditional kNN. Nevertheless, methods built on natural neighborhoods tend to rely heavily on the local geometric structure of data points [20]. When noise or outliers are present in the data, these anomalous points can significantly alter the neighborhood structure, leading to a decline in the stability and accuracy of the classification results. Especially when the data exhibits high-dimensional and complex distributions (e.g., non-convex shapes or multimodal distributions), the discriminative power of distance metrics decreases, and natural neighbor relationships may fail to accurately capture the boundary features between classes, negatively impacting classification performance.

In recent years, some studies have introduced granularball computing into KNN classification. By replacing original sample points with granular balls and making decisions based on the nearest granular ball, these methods improve classification efficiency and robustness to a certain extent [21], [22]. However, existing methods mainly rely on purity-related functions during granular ball generation, while purity is insufficient to fully characterize the geometric structure of samples within a granular ball. In the classification stage, they usually make decisions directly based on a single nearest granular ball, lacking explicit modeling of the local neighborhood of the query sample. Although a study has further combined granular-ball computing with natural neighborhood methods (MGKNN) [23], their granular ball partitioning processes still mainly rely on unsupervised mechanisms and cannot fully exploit class supervision information. Therefore, how to adaptively generate better supervised granular ball representations and, based on them, construct an adaptive neighborhood decision mechanism for query samples remains a worthwhile problem for further study.

To address the above limitations, this paper proposes an efficient and adaptive granular ball KNN (GBKNN) algorithm based on granular-ball computing. The algorithm initially simplifies the data distribution within each granular balls by splitting the original dataset containing n samples into $\sqrt { n }$ granular balls. The Fisher criterion is jointly introduced to control the adaptive generation of granular balls. Subsequently, an adaptive local neighborhood is constructed for the test sample according to its spatial relationship with the nearest granular ball, thereby adaptively determining the effective k value. Specifically, the contributions of this study include:

• A highly efficient and fully adaptive granular ball generation method is proposed. The method first partitions the data at a coarse granularity to reduce the complexity of data distribution in a ball, and then uses the Fisher criterion to guide ball splitting, thereby establishing an adaptive generation framework of granular balls that range from coarse to fine.

• A KNN method that constructs adaptive neighborhood relationships based on granular ball structures is proposed. Instead of classifying directly by the nearest granular ball, the method uses the nearest granular ball as a reference to build a local adaptive neighborhood for the test sample, and adaptively determines k by the actual number of samples contained in that neighborhood.

• Experimental results on multiple benchmark datasets show that GBKNN achieves superior overall performance in terms of classification accuracy, robustness, and time efficiency compared with several competing methods.

The remainder of this paper is organized as follows: Section II provides an overview of the evolution of KNN techniques, along with a discussion of relevant studies on granularball computing. In Section IV, a fully adaptive method for generating granular balls is introduced, followed by the design of an adaptive GBKNN algorithm built upon it. Section V reports the experimental results and delivers a comprehensive analysis of the algorithm’s performance. Finally, Section VI concludes the paper and highlights promising directions for future exploration.

## II. RELATED WORK

## A. KNN Classification Method

Neighborhood-based KNN algorithms have been extensively developed across various domains by integrating diverse strategies to overcome the limitations of traditional KNN [24]– [26]. These developments can be broadly categorized into three methodological directions.

First, some methods aim to reconstruct neighborhood structures to better reflect the true data manifold. To address the rigidity of fixed nearest-neighbor definitions, Yiu et al. [17] introduced the reverse KNN (RKNN) model, leveraging reverse nearest neighbor search to enhance query efficiency. Similarly, Brito et al. [27] proposed the mutual k-nearest neighbor (MKNN) model, which constructs mutual nearest neighbor graphs to improve robustness and reduce computational complexity for large datasets. Further, Zhu et al. [19] developed the natural neighbor (NaNEKNN) method, which eliminates the need for manually choosing the k parameter by identifying neighbors based on mutual proximity in the dataset. These methods redefine how neighborhoods are formed, providing a foundation for more adaptive KNN variants.

Second, adaptive strategies are employed to determine neighborhood sizes based on local density or distribution. Mullick et al. [28] introduced the adaptive KNN (AdaKNN), which utilizes local density and distribution patterns, integrated with neural networks, to improve classification performance. Similarly, OneStepKNN [29], parameterless KNN (PLKNN) [30], and Dr.k-NN [31] dynamically infer the optimal number of neighbors based on data characteristics, rather than relying on a pre-specified k value. These methods effectively alleviate the sensitivity to k and enhance KNN’s adaptability to complex data distributions.

Third, at the representation level, coarse-grained or structured approximations are used to enhance efficiency without significantly compromising accuracy. To address this, Ayyad et al. [32] proposed a strategy that defines circular weighted regions around test samples, incorporating SMKNN and LMKNN variants to boost classification accuracy, stability, and efficiency in gene expression data. More recently, Xie et al. [23] proposed the MGKNN algorithm, which leverages granular-ball computing to construct multi-granularity ball representations and neighborhood relationships while dynamically determining the k-value, thereby significantly improving the classification accuracy of KNN. However, this method still relies on the traditional KNN algorithm and does not fully utilize label information during the granular ball generation process, leaving room for further improvement.

## B. Granular-ball Computing

Granular-ball computing is an important model in granular computing, whose core idea is to adaptively construct granular balls of varying sizes to cover the sample space, thereby driving the learning process through granular ball-based representations. In terms of efficiency, the number of granular balls is significantly smaller than the number of samples, thus reducing computational overhead; in terms of robustness, the coarse-granularity nature of granular balls makes them less sensitive to sample noise; in terms of interpretability, the topological structure of granular balls provides multigranularity descriptions. For supervised datasets, granular ball generation is usually set to full coverage, i.e., the coverage ratio is 1, while ball compactness is typically ensured by the average radius. Given a dataset $D = \{ x _ { i } \mid i = 1 , 2 , \ldots , n \}$ , a commonly used supervised granular ball representation model is defined as:

$$
\begin{array} { r l } { } & { \underset { c _ { i } , r _ { i } , m } { \operatorname* { m i n } } m } \\ & { s . t . \ q u a l i t y ( G B _ { i } ) \geq \phi \left( x _ { j } \right) , i = 1 , 2 , . . . , m , j = 1 , 2 , . . . , n . } \end{array}\tag{1}
$$

where m is the number of granular balls, and $c _ { i }$ and $r _ { i }$ denote the center and radius of the i-th granular ball, respectively. The quality of each granular ball, quality(GB ), is regulated by the constraint function $\phi _ { x _ { j } }$ , thereby preventing the representation from degenerating into a few low-quality large balls. Hence, the key problem in granular ball generation is how to cover the sample space with as few granular balls as possible while maintaining high representation quality under the given constraints.

This model usually involves the joint optimization of the number, centers, radii, and quality constraints of granular balls, which leads to a highly non-convex optimization problem. Directly obtaining its global optimum is often computationally intractable. Therefore, inspired by the “global-first” strategy, this model is efficiently optimized and approximately solved through a coarse-to-fine granular ball splitting process with quality control. Initially, granular ball generation utilized the 2-means algorithm [21]. The purity threshold needs to be optimized within a specific range to adapt to different datasets and address label noise, but this process is computationally expensive. To address this challenge, Xia et al. proposed a novel granular ball generation method by controlling the weighted purity sum of sub-granular balls and accelerating the splitting process using a division method [22]. However, this method still requires improvements in stability and efficiency. Subsequently, Xie et al. [33] introduced the GBG (GBG++) method, based on an attention mechanism, which significantly improves efficiency and robustness while achieving stability. As a novel approach, granular-ball computing has inspired various theoretical methods in machine learning, including granular ball classifiers [34], granular ball clustering methods [35], granular ball neural networks [36], granular ball rough sets [37], and granular ball reinforcement learning [38]. Meanwhile, granular-ball computing enhances efficiency and interpretability, bringing benefits that foster broader research and application potential [39].

## III. PRELIMINARIES

Granular-ball computing takes granular balls as input units for learning. A granular ball is formally defined as follows [21].

Definition 1. Given a dataset $D = \{ x _ { i } , i = 1 , 2 , . . . , n \}$ , a granular ball GB split from $D$ is a spherical structure in the data space, represented by

$$
\begin{array} { r } { G B = \{ x _ { i } | x _ { i } \in D , \Delta ( x _ { i } - c ) \leq r \} , } \end{array}\tag{2}
$$

where $\Delta \left( x _ { i } - c \right)$ denotes the distance between a sample $x _ { i }$ and the center c. In this study, we adopt the Euclidean distance for computation. The center c and radius r ofthe granular ball are computed as follows:

$$
c = \frac { 1 } { \vert G B \vert } \sum _ { x _ { i } \in G B } x _ { i } , \ r = \frac { 1 } { \vert G B \vert } \sum _ { x _ { i } \in G B } \Delta \left( x _ { i } - c \right) ,\tag{3}
$$

where $| \cdot |$ denotes the cardinality of the set, i.e., the number of elements in the granular ball.

In classification problems, each granular ball, similar to a sample point, is assigned a class label, as defined below.

Definition 2. Let GB be a granular ball, and let $L = \{ l ^ { j } , j =$ $1 , 2 , \ldots , s \}$ denote the set of distinct class labels among its contained samples. The label of GB, denoted by $l _ { G B }$ , is defined as the label with the highest frequency among its samples:

$$
l _ { G B } = \arg \operatorname* { m a x } _ { l ^ { j } \in L } \left| \left\{ x _ { i } \in G B \mid l _ { x _ { i } } = l ^ { j } \right\} \right| ,\tag{4}
$$

where $l _ { x _ { i } }$ is the label of sample $x _ { i }$

According to Definition 2, the label of a granular ball is determined by the majority class of its contained samples, thus providing robustness to noise. The quality of a granular ball can be measured by its purity, defined as follows.

Definition 3. The purity of a granular ball GB, denoted by $P ,$ measures the proportion of samples in GB that share the granular ball’s label $l _ { G B } .$

$$
P = { \frac { | \{ x _ { i } \in G B \mid l _ { x _ { i } } = l _ { G B } \} | } { | G B | } } .\tag{5}
$$

In subsequent work, Xie et al. [33] introduced a splittingbased mechanism for granular ball generation, which begins with the entire dataset treated as a single granular ball. Specifically, the entire dataset is initially regarded as a granular ball, denoted by GB, which fails to meet the required quality criterion. The splitting process proceeds as follows:

Step 1: Compute the majority class label $l ^ { * }$ within the granular ball GB.

$$
l ^ { * } = a r g \operatorname* { m a x } _ { l ^ { j } \in L } \sum _ { i = 1 } ^ { n } \mathbb { I } ( l _ { x } = l ^ { j } ) ,\tag{6}
$$

where I(·) denotes the indicator function.

Step 2: Calculate the corresponding center $c _ { l ^ { * } }$ ∗ and radius $r _ { l ^ { \ast } }$ based on the majority class samples. To characterize the majority class region within the granular ball GB, first extract

![](images/f575714022780a632472feea8a35cb96763d30b8db26695887ecc402259de94d.jpg)  
(a) GBG++ on the ‘Two Moons’ dataset.

![](images/fe973758204f2a16b435ada9332181d32910c815b4330bdd3879c53b893ce293.jpg)

$$
\sqrt { n }
$$

![](images/d6961b8cb99be9a71eb73e45c4e98a7389220da2e432c8d1a8118e3d2188b26e.jpg)  
(c) GBG++ on the ‘Circles’ dataset.

![](images/40568066e7c57f2d3dc54368a1027cfb4565493c04ed20fcdc11fb9d6c7c18ec.jpg)  
(d) GBG++ (initial $\sqrt { n }$ split) on the ‘Circles’ dataset.  
Fig. 2: Comparison between GBG++ and GBG++ with initial $\sqrt { n }$ granular ball splitting on two non-linear datasets.

the subset of samples belonging to the majority class $l ^ { * }$ and compute its size:

$$
G B _ { l ^ { * } } = \left\{ x _ { i } \in G B \vert l _ { x _ { i } } = l ^ { * } \right\} , \quad N _ { l ^ { * } } = \left| G B _ { l ^ { * } } \right| .\tag{7}
$$

Then, calculate the center $c _ { l } { \mathrm { : } }$ ∗ and radius $r _ { l ^ { * } }$ of the majority class samples:

$$
c _ { l ^ { * } } = \frac { 1 } { N _ { l ^ { * } } } \sum _ { x _ { i } \in G B _ { l ^ { * } } } x _ { i } , \quad r _ { l ^ { * } } = \frac { 1 } { N _ { l ^ { * } } } \sum _ { x _ { i } \in G B _ { l ^ { * } } } \Delta \left( x _ { i } - c _ { l ^ { * } } \right) .\tag{8}
$$

Here, $\Delta ( \cdot )$ denotes the Euclidean distance.

Step 3: Construct a sub-granular ball using points within the computed radius.

$$
{ G B _ { k } } = \left\{ { x _ { i } } \in G B \mid \Delta \left( { x _ { i } } - { c _ { l ^ { * } } } \right) \leq r _ { l ^ { * } } \right\} .\tag{9}
$$

The iterative execution of Steps 1 to 3 continues until the number of remaining samples equals the number of class labels. This method has significantly improved the efficiency of granular ball generation. However, it exhibits limitations when dealing with complex nonlinear data structures, such as ‘Two Moons’ or ‘Circles’ datasets.

Specifically, the GBG++ method constructs each granular ball by calculating the center and radius based solely on the majority class samples, resulting in an approximately spherical shape. This geometric assumption implicitly presumes isotropic distributions, making it inadequate for fitting elongated or curved data structures. As a result, granular balls may mistakenly include structurally dissimilar samples or exclude similar ones, leading to structural errors. As illustrated in Fig. 2(a) and Fig. $2 ( \mathrm { c } ) ,$ , such mismatches hinder the model’s ability to represent high-dimensional, non-uniform data spaces effectively. To address this issue, we propose generating $\sqrt { n }$ granular balls from the global dataset to enable a more appropriate partitioning at the global level, simplifying the internal distribution within each ball and mitigating errors caused by limited local fitting capacity. Furthermore, existing methods typically require repeated tuning of the purity threshold to accommodate different datasets and label noise, which not only relies heavily on manual effort but also incurs considerable computational cost.

## IV. PROPOSED METHOD

## A. Motivation

Although existing granular ball-based KNN methods [21] have clear advantages over traditional KNN in classification efficiency and robustness, they still have limitations in both granular ball generation and decision making. First, in supervised granular ball generation, existing methods usually rely on specified purity thresholds or purity-related functions as the partition criterion. As a result, they focus mainly on label information and cannot fully capture the geometric distribution of samples within each granular ball. Second, in the prediction stage, these methods typically classify directly based on the nearest ball, lacking a mechanism to construct a local neighborhood around the test sample, and thus failing to reflect the true distribution near the test sample fully. Although MGKNN [23] enhances neighborhood modeling through granular ball clustering and enables adaptive adjustment of k, its granular ball generation still mainly relies on unsupervised mechanisms, making it difficult to fully exploit class supervision information and limiting its ability to represent true class boundaries. To address the above issues, this paper proposes a new supervised and fully adaptive granular ball generation method. The method first obtains an initial granular <sub>ball representation through</sub> √<sub>n-based coarse division and then</sub> uses the Fisher criterion, which captures both distributional representation and class discrimination, to control the granular ball generation process. The coarse-grained granular ball representation can effectively reduce the influence of noise points, thereby improving model robustness. In the decision stage, an adaptive local neighborhood is constructed for each test sample based on the nearest granular ball, and the value of k is automatically determined by the number of samples in the neighborhood. This not only further reduces the influence of small noisy granular balls but also better matches the decision mechanism of KNN.

The overall algorithm, as illustrated in Fig. 3, consists of two main modules: the granular ball generation and classification decision. First, the entire dataset is globally split into $\sqrt { n }$ initial granular balls to simplify data distributions within each ball (Fig. 3, Step 1). A coarse-to-fine splitting procedure is then performed, guided by the Fisher criterion that adaptively controls the splits without relying on preset thresholds. To further enhance decision boundary representation, a new purity lower bound based on label information is introduced, followed by a global overlap removal strategy to improve the structural clarity of granular balls. In the classification decision stage (Fig. 3, Module 2), test samples are not granularized. Instead, the nearest granular ball is located through a weighted distance mechanism, and the distance from the test sample to the farthest sample within that ball is used as the neighborhood radius $R _ { k N N }$ to construct an adaptive neighborhood centered on the test sample, thereby enabling efficient dynamic selection of k.

![](images/7bc48dda9d1269b03ebe221bc17fee90882db05da1fe6e71b37f2720623f2552.jpg)  
Fig. 3: The GBKNN, as illustrated in the overall flowchart, consists of two main modules: i) granular ball generation; ii) classification decision.

## B. Adaptive Granular Ball Generation

Initial Partitioning. In the generation of granular balls, directly splitting the entire dataset may be affected by the complexity of data distribution, particularly in the presence of nonlinear structures such as crescents or rings. This is because GBG++ determines the center and radius of each granular ball solely based on majority-class samples. Such geometric assumptions fail to accommodate elongated or curved data regions, making it difficult to align granular balls with the true shape of the data. As a result, the generated granular balls often exhibit excessive overlaps or leave out relevant samples, as shown in Fig. 2(a) and 2(c), ultimately degrading the model’s ability to represent intrinsic structures accurately. To better handle complex data structures, the dataset $D = \{ x _ { i } \mid$ $i = 1 , 2 , \ldots , n \}$ is initially partitioned into $\sqrt { n }$ granular balls using the k-means algorithm. Assume that the dataset contains s classes, and $D ^ { l }$ denotes the subset of samples in the l-th class. The initial center $o ^ { ( 0 ) }$ is randomly selected according to the proportion of each class, i.e.,

$$
\begin{array} { r } { o _ { j } ^ { ( 0 ) } = \mathrm { R a n d o m } \left( \left\{ x _ { i } \in D ^ { l } | l = 1 , 2 , . . , s \right\} \right) , j = 1 , 2 , . . . , \sqrt { n } . } \end{array}\tag{10}
$$

Each data point $x _ { i } \in D$ is assigned to the nearest centroid $o _ { j }$ using k-means clustering:

$$
G B _ { j ^ { * } } ^ { ( t ) } = \left\{ x _ { i } \in D | j ^ { * } = a r g \operatorname* { m i n } _ { j } \left\{ \Delta \left( x _ { i } , o _ { j } ^ { ( t ) } \right) \right\} \right\} ,\tag{11}
$$

where t is the iteration index. The centroid is then updated as:

$$
o _ { j } ^ { ( t + 1 ) } = \frac { 1 } { \left| G B _ { j } ^ { ( t ) } \right| } \sum _ { x _ { i } \in G B _ { j } ^ { t } } { x _ { i } } .\tag{12}
$$

The alternating iteration follows Eq. (11) and Eq. (12) until the centroids converge, ultimately yielding $\sqrt { n }$ stable granular balls and completing the initial data split.

To further validate the effectiveness of first performing $\sqrt { n }$ splitting, we conducted experiments on the crescent and ring-shaped datasets, and visualized the results using the GBG++ method, as shown in Fig. 2(b) and (d). Compared to directly applying the GBG++ method for splitting, the proposed strategy yields a more accurate approximation of the data distribution and better preserves the intrinsic geometric structure of the data. Specifically, on the crescent and ringshaped datasets, the GBG++ method generated 109 and 85 granular balls, respectively, whereas the number was significantly reduced to 21 and 62 when preceded by ${ \sqrt { n } } .$ -based splitting. This reduction is due to the initial $\sqrt { n }$ splitting, which effectively reduces the global mixing of data points, resulting in more stable subsequent granular ball partitioning and avoiding the computational overhead and overlap issues caused by excessive splitting. Therefore, this approach not only improves the data fitting capability but also optimizes the granular ball split efficiency, providing a better solution for applying the granular ball method to complex data structures.

Granular Ball Splitting Method: After generating $\sqrt { n }$ initial granular balls, we further refine the split. As reviewed in Section II-B, the GBG++ method employs an attentionbased splitting strategy [33], which demonstrates superior performance in terms of computational efficiency and stability compared to traditional methods such as k-means and $k \mathrm { - }$ partitioning. Specifically, GBG++ effectively focuses on key data regions, enabling more accurate granular ball splitting while minimizing redundant computations and enhancing overall efficiency. Therefore, we adopt the GBG++ approach described in Eq. (6), (7), (8) and (9) to further split the initial granular balls, ensuring structural stability and optimizing computational costs.

In classification tasks, the overlap between heterogeneous granular balls can lead to misclassification of samples in the overlapping regions, thereby affecting classification accuracy. Traditional methods typically perform deduplication only after the granular ball generation is complete. In contrast, this paper innovatively integrates the deduplication strategy into the granular ball generation process by executing deduplication after each split (as shown in Fig. 3). Further splitting is performed to reduce their size for overlapping heterogeneous granular balls, thereby optimizing subsequent granular ball splits. This strategy not only actively suppresses the accumulation of overlap, preventing excessive overlap caused by multiple rounds of splitting and improving the stability of granular ball splitting, but also gradually resolves the overlap issue during the splitting process, reducing the overall computational complexity of deduplication and improving computational efficiency.

Stop Condition: In the process of granular ball generation, the purity threshold is commonly used to guide the splitting of granular balls. However, due to the diverse characteristics of different datasets, the optimal value of the purity threshold often varies within a certain range, making it challenging to adaptively determine a fixed value in practice. Existing supervised adaptive granular ball generation methods usually determine whether to split a ball according to whether the weighted sum of the purities of the resulting child balls increases [22]. However, purity only measures the proportion of the dominant label within a granular ball and cannot fully characterize the geometric distribution of samples inside it. To address this limitation, this paper introduces a new criterion to control the granular ball generation process, namely the Fisher criterion, which is defined as follows. Suppose a granular ball GB contains samples from $L$ classes. Let $n _ { l }$ denote the number of samples in the l-th class, $\mu _ { l }$ the mean of the l-th class, and $\mu$ the overall mean of all samples in the granular ball. Then, the Fisher value of the granular ball is defined as:

$$
F = \frac { \sum _ { l = 1 } ^ { L } n _ { l } \left\| \mu _ { l } - \mu \right\| } { \sum _ { l = 1 } ^ { L } n _ { l } S _ { l } } ,\tag{13}
$$

where $S _ { l }$ represents the intra-class divergence of class l, defined as:

$$
S _ { l } = \frac { 1 } { n _ { l } } \sum _ { x _ { i } \in G B _ { l } } \left\| x _ { i } - \mu _ { l } \right\| ^ { 2 } .\tag{14}
$$

Here, $G B _ { l }$ represents the set of samples of class l.

Unlike the stopping criterion based on the weighted sum of granular balls’ purities, the Fisher criterion is not limited to measuring label consistency within a granular ball.

Instead, it evaluates both inter-class separability and intra-class compactness from the perspective of feature distribution. The weighted purity sum is essentially based on label proportions. While it can reflect the degree of class mixing within a granular ball, it cannot adequately characterize the geometric structure of samples in the feature space. In contrast, by jointly modeling between-class scatter and within-class scatter, the Fisher criterion can more comprehensively characterize a granular ball, thereby providing a finer and more robust basis for ball splitting and stopping. A small Fisher value indicates insufficient class separation or excessive within-class dispersion, suggesting that the granular ball remains mixed and should be further split.

Since a granular ball with purity equal to 1 is directly treated as a terminal ball, we further adopt a Fisher-based splitting criterion for non-pure granular balls. It should be noted that the Fisher criterion is not equivalent to purity, and therefore, we do not claim that it can theoretically guarantee a monotonic increase in purity. Instead, it focuses on the change in local structure before and after splitting. Specifically, when the weighted Fisher value of the sub-ball set is higher than that of the parent ball, the split achieves a better balance between inter-class separability and intra-class compactness, thereby improving the representation of the granular ball, and the split is accepted; otherwise, it is rejected, and the parent ball is retained.

Optimization of Granular Balls’ Quality: During granular ball partitioning, maintaining high purity within each granular ball is a key factor affecting classification performance. Existing splitting criteria may mistakenly accept some lowpurity parent balls, which can negatively affect the final classification accuracy. For example, the initial granular ball has a purity of $P = 5 5 / 1 0 0 = 0 . 5 5$ . After splitting, two child balls are obtained with purities $P _ { 1 } = 2 8 / 5 0 = 0 . 5 6$ and $P _ { 2 } = 2 7 / 5 0 = 0 . 5 4$ , respectively. In this case, although a split occurs, both child balls remain highly mixed, and neither interclass separability nor intra-class compactness is substantially improved. As a result, the weighted Fisher value of the child balls may be close to, or even lower than, that of the parent ball, so the Fisher criterion would judge the split as ineffective and stop further splitting. However, because the purity is still low, such granular balls may reduce subsequent classification accuracy and weaken the overall quality of the granular ball representation. To address the above issue, we propose a new category-adaptive purity lower bound $T _ { L _ { j } }$ , which is defined as follows:

$$
T _ { L _ { j } } = \frac { \sum _ { i = 1 } ^ { N } \left| G B _ { i } ^ { * } \right| } { \left| L _ { j } \right| } ,\tag{15}
$$

where $L _ { j }$ denotes the set of the j-th classes in the dataset D and N is the total number of granular balls set $\mathcal { G }$ generated after splitting.

This strategy allows for a more accurate assessment of the granular ball quality across different classes. It adaptively adjusts the purity for each class, ensuring that the parent granular ball stops splitting only after achieving sufficient quality. Additionally, it prevents low-quality granular balls from terminating prematurely, as shown in the diagram, thereby improving the overall granular balls’ quality. In addition, since the deoverlapping operation is applied only locally within sub-balls during the process, residual overlaps between different classes may persist. Therefore, a final global de-overlapping step is required to eliminate such inter-class overlaps and ensure a clear decision boundary.

## C. Classification Decision

Existing granular ball-based KNN methods usually classify directly according to the distance between the test sample and the nearest boundary granular ball, which can improve classification efficiency and robustness to some extent. However, such methods essentially rely on only a single nearest granular ball for decision making and do not explicitly construct a local neighborhood around the test sample. To address this issue, this paper proposes a novel neighborhood decision-making mechanism based on granular balls. Specifically, it first selects the granular ball whose boundary is closest to the test sample, and then uses the distance from the test sample to the farthest sample within that ball as the neighborhood radius, thereby constructing an adaptive neighborhood centered on the test sample. To address this issue, this paper innovatively proposes a sample-number-based weighting method, which dynamically adjusts the confidence of granular balls to mitigate the impact of noisy granular balls on classification decisions, thereby enhancing classification robustness. Based on this, the weighted distance is defined as follows:

Definition 4. Let $\mathcal { G } = \{ G B _ { i } \ | \ i = 1 , 2 , . . . , N \}$ be a granular ball set. The weighted distance between the test sample $x ^ { t e s t }$ and the ball $G B _ { i } \in \mathcal { G }$ is defined as:

$$
W d _ { i } \left( { x ^ { t e s t } , G B _ { i } } \right) = \left( 1 - \frac { { \left| G B _ { i } \right| } } { \sum _ { j = 1 } ^ { N } { \left| G B _ { j } \right| } } \right) \cdot \left( \Delta \left( { x ^ { t e s t } , o _ { i } } \right) - r _ { i } \right)\tag{16}
$$

where $| G B _ { i } |$ represents the number of samples contained in the ball $G B _ { i }$

The weighted distance enhances the reliability of granular ball labels by adjusting the boundary distance based on the proportion of sample counts, prioritizing granular balls with larger sample sizes and reducing the influence of noise or small granular balls on decision-making.

After obtaining the weighted nearest granular ball for the test sample, we further use the distance from the test sample to the farthest sample within that ball as the neighborhood radius, thereby constructing an adaptive neighborhood centered on the test sample. Since a granular ball consists of multiple similar samples, it can effectively reduce noise interference and provide more stable and reliable local group information than a single nearest sample point. The neighborhood induced by the nearest ball fully covers the samples within that ball while remaining compact. In other words, it preserves the reliable local information represented by the nearest granular ball without unnecessarily expanding the neighborhood.

Compared with existing KNN methods based on granularball computing, this approach explicitly constructs a strict neighborhood relation around the test sample, thus preserving the basic idea that labels are determined by neighborhoods. Compared with traditional KNN, the neighborhood here is first guided by the coarser granular ball, which gives the method advantages in both efficiency and robustness. Moreover, the actual number of samples contained in the neighborhood naturally determines the corresponding k value, so the neighborhood size no longer depends on manual setting but is adaptively determined by the local data distribution around the test sample. Therefore, the proposed method not only retains the efficiency and robustness of multi-granularity granular ball representation, but also preserves the essential KNN characteristic of making decisions based on local neighborhoods.

## D. Algorithm Procedure

This paper proposes an improved granular ball KNN classification method, which consists of the generation of granular balls and classification decision (as illustrated in Fig. 3). To optimize the initial granular ball partitioning and reduce the number of iterations, this paper directly splits the entire dataset into $\sqrt { n }$ granular balls and designs a new evaluation index to select the optimal initial granular ball division scheme, thereby simplifying the data distribution within each ball. Subsequently, the GBG++ method is employed for fast granular ball splitting, introducing weighted purity of sub-balls for dynamic control, achieving threshold-free and adaptive splitting. Considering that low-quality granular balls may remain after splitting, this paper further introduces an optimization of the granular ball quality stage, which refines the granular ball set based on a new purity threshold to improve ball quality and provide more reliable granular ball representations for subsequent classification decisions. In the classification stage, the proposed method first identifies the nearest granular ball of the test sample using a weighted distance measure, then constructs an adaptive neighborhood centered on the test sample based on that ball, and finally determines the label by using the sample information within the neighborhood. The detailed procedure is given in Algorithm 1.

## E. Theoretical Investigation

To further validate the effectiveness of the initial splitting method, a theorem is presented that proves directly that splitting $\sqrt { n }$ granular balls results in balls with better density compared to the binary splitting method. Specifically, the direct splitting method better maintains the density balance of the clusters during the division, leading to theoretically superior partition results, with these granular balls exhibiting a clear advantage in terms of density distribution.

Theorem 1. Let the dataset be $D = \{ x _ { 1 } , x _ { 2 } , \ldots , x _ { n } \} \subset \mathbb { R } ^ { d }$ and assume it is splitted into $\sqrt { n }$ granular balls $\{ G B _ { i } , i =$ $1 , 2 , . . . , \sqrt { n } \}$ , where $r _ { i }$ denotes the radius of the i-th ball. $\Omega _ { G B _ { i } }$ denotes the data space region that the granular ball $G B _ { i }$ covers. The average density $\bar { \rho }$ is defined as:

$$
\bar { \rho } = \frac { 1 } { \sqrt { n } } \sum _ { i = 1 } ^ { \sqrt { n } } \rho _ { i } , \ \rho _ { i } = \frac { 1 } { \sqrt { n } } \sum _ { i = 1 } ^ { \sqrt { n } } \frac { n _ { i } } { V _ { i } } ,\tag{17}
$$

Algorithm 1 GBKNN algorithm   
Input: Noise training set $D _ { n o i s y } ,$ test set $D _ { t e s t } ;$   
Output: The accuracy of GBKNN;   
1: Select $\sqrt { n }$ initial points randomly according to the label ratio multiple   
times to split $\sqrt { n }$ balls using k-means, and evaluate according to Eq. (32)   
to select the optimal granular ball set ${ \mathcal { G } } _ { \sqrt { n } } .$   
2: Initial $W = 0 , \bar { \mathcal { G } } _ { b a l l } \bar { \mathbf { \Psi } } = \emptyset , \mathcal { G } _ { n e x t } = \mathcal { G } _ { \sqrt { n } } ;$   
3: while $\mathcal { G } _ { n e x t } \neq \emptyset$ do   
4: Select one granular ball $G B _ { j }$ from $\mathcal { G } _ { n e x t } ,$ calculate the Fisher value   
W of $G B _ { j } \bar { ; }$   
5: Split the granular ball $G B _ { j }$ using the GBG++ method;   
Calculate the sum of all sub-balls’ Fisher value $F _ { G B _ { j } } ;$   
6: $\mathbf { i f } \ W < F _ { G B _ { i } }$ then   
7: Delete $\dot { G } \dot { B _ { j } }$ from $\mathcal { G } _ { n e x t } ,$ de-overlap heterogeneous balls in sub  
balls and add the resulting balls to $\mathcal { G } _ { n e x t } ;$   
8: else   
9: Add $G B _ { j }$ to $\mathcal { G } _ { b a l l }$ and remove it from $\mathcal { G } _ { n e x t } ;$   
10: end if   
11: Return to step 4.   
12: end while   
13: Get $T _ { L _ { j } }$ by the Eq. (15), and then use the GBG++ method to segment   
the balls with purity lower than $T _ { L _ { j } } ;$   
14: De-overlap heterogeneous granular balls;   
15: According to Eq. (16), the weighted distance $W _ { d }$ from each test sample   
to all granular balls is first computed to identify the nearest granular ball   
$G B ^ { * }$   
16: Then, the distance from the test sample to the farthest sample within   
that ball is taken as the radius $\begin{array} { r } { R _ { k N N } ( \bar { \boldsymbol { x } } ) = \operatorname* { m a x } _ { x _ { i } \in G B ^ { * } } \Delta ( \bar { \boldsymbol { x } } , \boldsymbol { x } _ { i } ) . } \end{array}$ . An   
adaptive neighborhood centered on the test sample can be constructed.   
17: It can get the label of the test sample can be obtained based on the   
neighborhood, and the final accuracy can be obtained.

where $n _ { i } = | G B _ { i } |$ denotes the number of samples in $G B _ { i } ,$ and $V _ { i }$ represents the volume of $G B _ { i } .$ . Consider the following two partitioning methods:

Method 1 (Global Partition): The $\sqrt { n }$ granular balls are obtained by directly applying the k-means algorithm over the entire dataset;

Method 2 (Recursive Partition): The $\sqrt { n }$ granular balls are generated via recursive binary splitting using 2-means.

The following conclusions are satisfied:

(i) If the data are uniformly distributed, then the average densities of samples in granular balls produced by both methods are approximately equal;

(ii) Assuming that the data are drawn from a Lipschitzcontinuous probability density function $f ( x ) ,$ , the average density of sample points in the granular balls generated by Method 1 is higher than that of Method 2, i.e., $\bar { \rho } ^ { ( 1 ) } > \bar { \rho } ^ { ( 2 ) }$

Proof. (i) Uniform Distribution.

Assume that the dataset is normalized to lie within the unit hypercube, so that all data points $x _ { i } \sim U ( [ 0 , 1 ] ^ { d } )$ are drawn from a uniform distribution over the unit hypercube. For Method 1, the k-means algorithm is directly applied to split the data into $\sqrt { n }$ granular balls. Since the data are uniformly distributed, each granular ball covers approximately the same region, with volume denoted by V, and the number of samples contained in each ball, $| G B _ { i } | \approx { \sqrt { n } }$ , is roughly equal. In $\mathbb { R } ^ { d }$ the volume of a ball [40] with radius r is given by

$$
V _ { d } ( r ) = \frac { \pi ^ { d / 2 } } { \Gamma ( 1 + d / 2 ) } r ^ { d } = \xi _ { d } r ^ { d } ,\tag{18}
$$

where Γ denotes the gamma function and $\xi _ { d }$ is a constant. When $d = 1$ , thus $\xi _ { 1 } = 2 ;$ hence, $V ( r ) = 2 r$ , which represents

the length of a line segment. When $d = 2 ,$ , we have $\xi _ { 2 } = \pi ;$ therefore, $V ( r ) = \pi r ^ { 2 }$ , which represents the area of a circle.

The average density of sample points in a granular ball is

$$
\bar { \rho } ^ { ( 1 ) } \approx \frac { \sqrt { n } } { V _ { d } } .\tag{19}
$$

For Method 2, under the assumption of uniform data distribution, the recursive splitting process is performed in a balanced manner, resulting in granular balls with equal volumes; similarly, we have

$$
\bar { \rho } ^ { ( 2 ) } \approx \frac { \sqrt { n } } { V _ { d } } .\tag{20}
$$

In summary, when the samples $x _ { i } \sim U ( [ 0 , 1 ] ^ { d } )$ are drawn from a uniform distribution over the unit hypercube, the granular balls generated by both methods have similar volumes and sample counts, and thus their densities tend to be approximately equal, i.e.,

$$
\bar { \rho } ^ { ( 1 ) } \approx \bar { \rho } ^ { ( 2 ) } .\tag{21}
$$

(ii) Non-uniform Distributions.

Assume that the data samples are drawn from a probability density function $f ( x )$ . On each granular ball $G B _ { i }$ , the function $f ( x )$ has a bounded gradient and satisfies the Lipschitz condition; that is, there exists a constant $L _ { i } > 0$ such that [41]:

$$
\forall x , y \in G B _ { i } , | f ( x ) - f ( y ) | \leq L _ { i } \left\| x - y \right\| .\tag{22}
$$

Since the samples follow the distribution $f ( x )$ , the expected number of samples in $G B _ { i }$ is approximately

$$
E \left[ \left| G B _ { i } \right| \right] \approx n \int _ { \Omega _ { G B _ { i } } } f \left( x \right) d x .\tag{23}
$$

We rewrite the function $f ( x )$ on $G B _ { i }$ as:

$$
f \left( x \right) = f \left( c _ { i } \right) + \left( f \left( x \right) - f \left( c _ { i } \right) \right) .\tag{24}
$$

Substitute this approximation Eq. (24) into the integral:

$$
\begin{array} { r } { \displaystyle \int _ { \Omega _ { G B _ { i } } } f \left( \boldsymbol { x } \right) d \boldsymbol { x } = \int _ { \Omega _ { G B _ { i } } } \left( f \left( c _ { i } \right) + \left( f \left( \boldsymbol { x } \right) - f \left( c _ { i } \right) \right) \right) d \boldsymbol { x } } \\ { \displaystyle = f \left( c _ { i } \right) V _ { i } + \int _ { \Omega _ { G B _ { i } } } \left( f \left( \boldsymbol { x } \right) - f \left( c _ { i } \right) \right) d \boldsymbol { x } . } \end{array}\tag{25}
$$

By multiplying by the total number of samples n, the expected number of samples can be estimated as:

$$
\begin{array} { l } { { \displaystyle E \left[ \left. G B _ { i } \right. \right] \approx n \int _ { \Omega _ { G B _ { i } } } f \left( x \right) d x } \ ~ } \\ { { \displaystyle ~ = n f \left( c _ { i } \right) V _ { i } + n \int _ { \Omega _ { G B _ { i } } } \left( f \left( x \right) - f \left( c _ { i } \right) \right) d x } \ ~ } \\ { { \displaystyle ~ = n f \left( c _ { i } \right) V _ { i } + \epsilon _ { i } } . } \end{array}\tag{26}
$$

Therefore, $\epsilon _ { i }$ represents the error in estimating $E [ | G B _ { i } | ] .$ —that is, the error caused by using $f \left( c _ { i } \right) V _ { i }$ instead of the integral $\int _ { \Omega _ { G B _ { i } } } \left( f \left( x \right) \right)$ dx. Since $f ( x )$ is Lipschitz continuous on $G B _ { i }$ that is,

$$
\left| f \left( x \right) - f \left( c _ { i } \right) \right| \leq L _ { i } \left\| x - c _ { i } \right\| \leq L _ { i } r _ { i } .\tag{27}
$$

Then, the corresponding error term can be bounded as follows:

$$
\epsilon _ { i } \leq n \int _ { \Omega _ { G B _ { i } } } \left( L _ { i } r _ { i } \right) d x \leq n L _ { i } r _ { i } V _ { i } .\tag{28}
$$

So the density $\rho _ { i }$ of samples in granular ball $G B _ { i }$ can be estimated as follows:

$$
\rho _ { i } = \frac { n _ { i } } { V _ { i } } = \frac { E \left[ | G B _ { i } | \right] } { V _ { i } } \approx n f \left( c _ { i } \right) + n L _ { i } r _ { i } ,\tag{29}
$$

where $f ( c _ { i } )$ denotes the value of the density function at the center $c _ { i }$ of the ball. For each granular ball, a larger $f \left( c _ { i } \right)$ and a smaller error term $n L _ { i } r _ { i }$ lead to a higher estimated density.

Method 1 generates $\sqrt { n }$ granular balls across the entire data space. The samples within each ball are as concentrated as possible, and $r _ { i }$ is small, which increases the concentration of samples within the clusters, i.e., larger $f ( c _ { i } )$ . Since k-means tends to generate balls with balanced sizes and small $r _ { i } ,$ the sample concentration within each ball is higher, leading to smaller error. As a result, according to Eq. (29), the estimated density $\rho _ { i }$ for each granular ball is larger. Therefore, Method 1 yields granular balls with higher individual densities $\rho _ { i }$ , and consequently achieves a larger average density:

$$
\bar { \rho } ^ { ( 1 ) } = \frac { 1 } { \sqrt { n } } \sum _ { i = 1 } ^ { \sqrt { n } } \rho _ { i } .\tag{30}
$$

Method 2 uses recursive 2-means splitting. Since each round of division only focuses on local information, the final centers $c _ { i }$ are no longer globally optimal and may fall into regions with lower density $f ( x )$ . This can result in granular balls with larger radii or uneven sample distributions, where $f \left( c _ { i } \right)$ is smaller, $r _ { i }$ is larger, and $L _ { i }$ is unstable, leading to larger error terms. Consequently, the average density $\bar { \rho } ^ { ( \breve { 2 } ) }$ tends to be lower. Therefore,

$$
\bar { \rho } ^ { ( 1 ) } > \bar { \rho } ^ { ( 2 ) } .\tag{31}
$$

Remark: In density estimation, the n $\mathbf { } _ { L _ { i } r _ { i } }$ term represents an upper bound for the error, not the actual trend of density increasing with radius $r _ { i } .$ This is a consequence of using the Lipschitz continuity to estimate the bias introduced in the integral approximation. As a result, the density approximation takes a form related to this argument. In fact, the density $\rho _ { i }$ does not necessarily increase with the radius $r _ { i } ,$ especially in non-uniform distributions, where the sample density in regions farther from the center of the ball typically decreases. Therefore, the main term $n f ( c _ { i } )$ should be considered the core estimate of the density, while the error term $n L _ { i } r _ { i }$ is only used to quantify the bias and does not reflect the actual trend of density change.

In the above method, the selection of initial points is crucial. Initially, we randomly selected initial points based on the proportion of samples with different labels. However, this random strategy does not guarantee the optimality of granular ball splitting, often requiring multiple computations and validation to determine the best partitioning, leading to high computational costs. Moreover, random initialization may result in an imbalanced distribution of initial granular balls, which can affect the stability of subsequent splits and the overall quality of the final granular balls.

To address the above issue, a heuristic criterion based on density evaluation is proposed to optimize the initial split of $\sqrt { n }$ granular balls. Specifically, we first generate $\sqrt { n }$ initial granular balls, denoted as ${ \mathcal { G } } _ { \sqrt { n } } .$ , and introduce the sum of granular balls’ densities as an evaluation metric to assess the quality of the partitioning. This criterion enhances the separability of the initial granular balls and provides a more stable and effective foundation for subsequent refinement. The mathematical definition is as follows:

$$
S u m _ { T } = \sum _ { j = 1 } ^ { \sqrt { n } } \rho _ { j } ,\tag{32}
$$

where $\rho _ { j }$ denotes the density estimate of the granular ball $G B _ { j } \in \mathcal { G } _ { \sqrt { n } } ,$ reflecting the compactness of the sample distribution within the ball. A higher density indicates that the samples are more tightly clustered, suggesting stronger representational and discriminative capabilities of the granular ball. By introducing a density-based evaluation criterion, a relatively optimal initial granular ball split can be achieved with fewer random selections. This approach effectively reduces uneven distributions caused by randomness, enhancing the stability and reproducibility of the splitting process. Moreover, it promotes a more compact and balanced initial structure, providing a more robust foundation for subsequent granular ball refinement. As a result, the overall computational cost is reduced, and generation efficiency is significantly improved.

## F. Time Complexity

The overall time complexity of the proposed algorithm can be considered from two stages: the granular ball generation stage and the classification decision stage. The granular ball generation stage primarily consists of the initial partitioning of granular balls and the dynamic splitting process based on the GBG++ method. In contrast, the classification decision stage uses a weighted nearest ball to construct a neighborhood centered on the predicted point and completes the classification. Next, we analyze the computational complexity of each stage separately.

1) Granular Ball Generation Stage: First, we apply $k \mathrm { - }$ means to obtain an initial division of the dataset into $\sqrt { n }$ granular balls, which simplifies subsequent computations. The time complexity of k-means is typically $O ( n k T _ { k } )$ , where $T _ { k }$ denotes the number of k-means iterations. By setting $k = \sqrt { n }$ , the time complexity of this initial division stage becomes $O ( n \sqrt { n } T _ { k } )$ . Subsequently, during the GBG++ iterative splitting stage, each ball is controlled based on the weighted Fisher sum of its split sub-balls, ultimately forming G granular balls Since the time complexity of $\mathrm { G B G + + }$ is close to linear, i.e., $O ( n )$ , and granular balls tend to stabilize during the intermediate stages, the overall complexity for granular ball splitting can be approximated as $O ( n )$ [33]. Furthermore, the de-overlapping operation is performed after each split, but it only affects a small number of adjacent granular balls, thus the computational load is relatively low and can be neglected. Therefore, the total time complexity for the entire granular ball generation stage can be expressed as $O ( n { \sqrt { n } } T _ { k } ) + O ( n ) = O ( n { \sqrt { n } } )$

2) Classification Decision Stage: In the classification stage, for each test sample, the weighted distances to all granular balls are first computed to identify the weighted nearest ball and initialize the neighborhood radius. Let the number of test samples be $M ,$ and the final number of granular balls be $N .$ The time complexity of this step is $O ( M N )$ . To avoid exhaustive search over all training samples, the proposed method further uses the geometric boundaries of granular balls to filter candidate neighborhoods. Specifically, only those granular balls whose boundary distance to the test sample is no greater than the neighborhood radius are considered to potentially intersect the neighborhood. Therefore, sample-level distance computation is performed only within these candidate balls. In this way, neighborhood construction consists of two levels: a coarse filtering at the granular ball level, followed by a fine filtering at the sample level within the candidate balls. Let $s _ { i }$ denote the number of samples in the candidate granular balls of the i-th test sample that require further distance computation. Then, the total complexity of the sample-level fine filtering over the whole test set is $\textstyle { \dot { O } } ( \sum _ { i = 1 } ^ { M } s _ { i } )$ . Accordingly, the total time complexity of the classification stage can be written as $\begin{array} { r } { O \left( M N + \sum _ { i = 1 } ^ { \bar { M } } s _ { i } \right) } \end{array}$ . If s¯ denotes the average number of samples that need to be further checked within the candidate granular balls for each test sample, the above complexity can be further simplified as $O \left( M \left( N + { \bar { s } } \right) \right)$ .

3) Overall Time Complexity Analysis: Traditional KNN is a lazy learning method, whose main computational cost is concentrated in the testing stage, with complexity $O ( M n )$ In contrast, the proposed method consists of two stages: a granular ball generation stage and a classification stage. The complexity of the granular ball generation stage is $O ( n { \sqrt { n } } )$ while the complexity of the classification stage is $O ( M ( N +$ s¯)). Therefore, the total complexity of the proposed method is $O ( n { \sqrt { n } } + M ( N + { \bar { s } } ) )$ . When the number of test samples is large and $N + \bar { s } \ll n$ , the additional cost introduced by preprocessing can be compensated by the efficiency gain in the subsequent classification stage, so that the proposed method achieves an overall computational advantage.

## V. EXPERIMENT DETAILS

This section conducts an experiment analysis from four main aspects: i) comparing the classification accuracy of different methods across various datasets; ii) analyzing the accuracy variation trends of each method under different noise conditions; iii) evaluating the runtime performance of different methods; iv) evaluating the classification accuracy and running time of GBKNN on large-scale datasets; v) verifying, through ablation studies, the effectiveness of the $\sqrt { n } { \mathrm { - b a s e d } }$ coarse partition, neighborhood construction, and purity lower-bound mechanism in improving model performance.

## A. Setup

Datasets: To verify the feasibility and effectiveness of the proposed method, experiments were conducted on 17 real-world datasets, including 12 commonly used datasets from the UCI Machine Learning Repository<sup>1</sup> and 5 newly added large-scale datasets (Santander<sup>2</sup>, creditcard<sup>3</sup>, cover-$\mathrm { t y p e ^ { 4 } , H i g \bar { g } s ^ { 4 } }$ , and supersymmetry<sup>1</sup>). The dataset ranges from 169 to 1,048,575 samples, with feature dimensions varying between 2 and 200 and the number of classes ranging from 2 to 9. Further details are presented in Table I.

TABLE I: Data information
<table><tr><td>NO.</td><td>Datasets</td><td>Sam</td><td>Dim</td><td>Classes</td></tr><tr><td>1</td><td>monks-2</td><td>169</td><td>7</td><td>2</td></tr><tr><td>2</td><td>heart1</td><td>294</td><td>13</td><td>2</td></tr><tr><td>3</td><td>cleveland</td><td>297</td><td>13</td><td>5</td></tr><tr><td>4</td><td>haberman</td><td>306</td><td>3</td><td>2</td></tr><tr><td>5</td><td>balance-scale</td><td>625</td><td>4</td><td>2</td></tr><tr><td>6</td><td>fourclass</td><td>862</td><td>2</td><td>2</td></tr><tr><td>7</td><td>phoneme</td><td>5,404</td><td>5</td><td>2</td></tr><tr><td>8</td><td>Anuran</td><td>7,195</td><td>22</td><td>7</td></tr><tr><td>9</td><td>mushroom</td><td>8,124</td><td>22</td><td>2</td></tr><tr><td>10</td><td>anneal</td><td>20,867</td><td>10</td><td>9</td></tr><tr><td>11</td><td>rssi</td><td>23,500</td><td>3</td><td>3</td></tr><tr><td>12</td><td>Skin NonSkin</td><td>35,961</td><td>3</td><td>2</td></tr><tr><td>13</td><td>Santander</td><td>199999</td><td>200</td><td>2</td></tr><tr><td>14</td><td>creditcard</td><td>284806</td><td>30</td><td>2</td></tr><tr><td>15</td><td>covertype</td><td>566601</td><td>10</td><td>2</td></tr><tr><td>16</td><td>Higgs</td><td>937882</td><td>24</td><td>2</td></tr><tr><td>17</td><td>supersymmetry</td><td>1048575</td><td>18</td><td>2</td></tr></table>

Baselines: The proposed method is compared with several baseline algorithms, including KNN [1], NaNEKNN [19], adaKNN variants [28], SMKNN [32], LMKNN [32], OneStep-KNN [29], PLKNN [30], MGNR [23], GBKNN 2019 [21] and the method based on sub-balls weighted purity and control ball division, and using weighted nearest granular ball decision-making, is denoted as GBKNN p.

Implementation details: For the traditional KNN algorithm, the value of k is set within the range [1, 15] with a step size of 2, and the optimal k is determined through hyperparameter tuning using the training set. In the generation of granular balls, our method first randomly selects the initial point 10 times to split $\sqrt { n }$ balls and then selects the optimal splitting result for subsequent calculations based on the evaluation index. To ensure the comparability of performance evaluation across different algorithms on the same dataset, classification accuracy (ACC) is adopted as the evaluation metric, which quantifies the consistency between classification results and ground truth labels. A higher ACC value indicates better classification performance. In the experiments, all datasets are randomly split into train and test sets in a $\mathrm { ~ 8 ~ : ~ 2 ~ }$ ratio to ensure fairness and robustness. Each dataset is augmented by introducing noise at different proportions: 0%, 5%, 10%, 15%, 20%, 25%, and 30%. It is important to note that noise is introduced only to the training set, while the validation and test sets remain free of noise, ensuring the reliability and validity of the results. For all experiments, our hardware experimental environment is an AMD Ryzen Threadripper PRO 5975WX

TABLE II: Average classification accuracy of different methods under various noise conditions
<table><tr><td>Data</td><td>monks-2</td><td>heart1</td><td>cleveland</td><td>haberman</td><td>balance-scale</td><td>fourclass</td><td>phonme</td><td>Anuran</td><td>mushroom</td><td>anneal</td><td>rssi</td><td>Skin Nonskin</td><td>Avg.</td></tr><tr><td>KNN</td><td>0.5798</td><td>0.6281</td><td>0.4680</td><td>0.6684</td><td>0.8949</td><td>0.9884</td><td>0.8437</td><td>0.9830</td><td>0.9876</td><td>0.9231</td><td>0.8585</td><td>0.9844</td><td>0.8173</td></tr><tr><td>NaNEKNN</td><td>0.7395</td><td>0.7044</td><td>0.4483</td><td>0.7347</td><td>0.8903</td><td>0.9527</td><td>0.8332</td><td>0.9526</td><td>0.9675</td><td>0.8689</td><td>0.8477</td><td>0.9883</td><td>0.8273</td></tr><tr><td>adaKNNGIHS</td><td>0.6807</td><td>0.6305</td><td>0.2266</td><td>0.5408</td><td>0.8880</td><td>0.9826</td><td>0.7846</td><td>0.8973</td><td>0.9935</td><td>0.7369</td><td>0.6919</td><td>0.9848</td><td>0.7532</td></tr><tr><td>adaKNN</td><td>0.6471</td><td>0.6404</td><td>0.4483</td><td>0.5714</td><td>0.8480</td><td>0.9701</td><td>0.7836</td><td>0.9626</td><td>0.9900</td><td>0.8852</td><td>0.7820</td><td>0.9874</td><td>0.7930</td></tr><tr><td>adaKNN2</td><td>0.6555</td><td>0.6232</td><td>0.4384</td><td>0.5893</td><td>0.8366</td><td>0.9709</td><td>0.7995</td><td>0.9520</td><td>0.9318</td><td>0.8834</td><td>0.8165</td><td>0.9819</td><td>0.7899</td></tr><tr><td>adaKNN2GIHS</td><td>0.5798</td><td>0.5788</td><td>0.2020</td><td>0.4923</td><td>0.8274</td><td>0.9635</td><td>0.7786</td><td>0.9042</td><td>0.9305</td><td>0.8174</td><td>0.7515</td><td>0.9634</td><td>0.7325</td></tr><tr><td>SMKNN</td><td>0.6218</td><td>0.7586</td><td>0.5320</td><td>0.7985</td><td>0.9211</td><td>0.8646</td><td>0.7955</td><td>0.9435</td><td>0.9537</td><td>0.9042</td><td>0.8280</td><td>0.8156</td><td>0.8114</td></tr><tr><td>LMKNN</td><td>0.6050</td><td>0.7759</td><td>0.4729</td><td>0.7806</td><td>0.9223</td><td>0.8796</td><td>0.7939</td><td>0.6941</td><td>0.9256</td><td>0.7315</td><td>0.8524</td><td>0.7200</td><td>0.7628</td></tr><tr><td>OneStepKNN</td><td>0.4580</td><td>0.6576</td><td>0.4483</td><td>0.7168</td><td>0.4400</td><td>0.5905</td><td>0.4262</td><td>0.5728</td><td>0.5029</td><td>0.7278</td><td>0.5240</td><td>0.6802</td><td>0.5621</td></tr><tr><td>PLKNN</td><td>0.6008</td><td>0.7389</td><td>0.4138</td><td>0.7500</td><td>0.8754</td><td>0.7965</td><td>0.7818</td><td>0.9237</td><td>0.9210</td><td>0.8906</td><td>0.7387</td><td>0.6988</td><td>0.7608</td></tr><tr><td>MGNR</td><td>0.7017</td><td>0.6379</td><td>0.4483</td><td>0.6199</td><td>0.8686</td><td>0.9718</td><td>0.8334</td><td>0.9684</td><td>0.9382</td><td>0.9177</td><td>0.8280</td><td>0.9714</td><td>0.8088</td></tr><tr><td>GBKNN_2019</td><td>0.7227</td><td>0.7118</td><td>0.5222</td><td>0.7168</td><td>0.8994</td><td>0.8937</td><td>0.8115</td><td>0.9831</td><td>0.9341</td><td>0.9168</td><td>0.8043</td><td>0.9175</td><td>0.8195</td></tr><tr><td>GBKNN_p</td><td>0.7143</td><td>0.7783</td><td>0.5369</td><td>0.7628</td><td>0.9637</td><td>0.9950</td><td>0.8401</td><td>0.9869</td><td>0.9924</td><td>0.9204</td><td>0.8884</td><td>0.9922</td><td>0.8622</td></tr><tr><td>GBKNN</td><td>0.7311</td><td>0.8030</td><td>0.5567</td><td>0.8291</td><td>0.9669</td><td>0.9967</td><td>0.8403</td><td>0.9858</td><td>0.9896</td><td>0.9286</td><td>0.8991</td><td>0.9766</td><td>0.8736</td></tr></table>

32-Cores 3.60 GHz, and the software experiment environment is Python 3.9.

## B. Main Results

In Table II, the average accuracy results of different methods under 7 noise levels are presented, with the optimal results highlighted in bold and the second-best results marked with underline. As shown in Table II, the GBKNN algorithm achieves the best or second-best accuracy on most datasets, significantly outperforming the other 13 algorithms. Particularly under various noise conditions, GBKNN and GBKNN p show notable improvements in average accuracy compared to the latest multi-granularity-based MGNR method. GBKNN further achieves better performance than GBKNN p. For example, on the ‘heart1’ dataset, the accuracy of the MGNR method is 0.6379, while the proposed GBKNN algorithm achieves an accuracy of 0.8030, surpassing MGNR by approximately 25.88%. On all datasets, the GBKNN method reaches an average accuracy of 0.8736, outperforming other algorithms. In comparison, the average accuracies of GBKNN 2019 and GBKNN p are 0.8195 and 0.8622, respectively. Therefore, GBKNN improves the average accuracy by approximately 6.60% and 1.32% compared with GBKNN 2019 and GBKNN p, respectively.

This result fully demonstrates the outstanding performance of the GBKNN algorithm in classification tasks, clearly surpassing existing methods. The superior performance of GBKNN mainly comes from its adaptive granular ball representation and neighborhood construction mechanism. Specifically, the initial partition into $\sqrt { n }$ coarse granular balls preserves the overall distribution while simplifying local data structures, thereby reducing the effect of complex nonlinear distributions. Compared with GBKNN p, GBKNN uses the Fisher criterion for granular ball partitioning. Unlike purity, the Fisher criterion simultaneously captures inter-class separability and intra-class compactness, thus providing more reasonable guidance for granular ball generation. Classifying based on coarse-grained granular balls helps reduce the impact of noise points and enhances robustness. In the decision stage, GBKNN first searches for the nearest granular ball by a weighted distance, and then uses the distance from the test sample to the farthest sample within that ball as the neighborhood radius to construct an adaptive neighborhood around the test sample. Since a granular ball consists of multiple similar samples, it provides more reliable local group information than a single nearest neighbor. Meanwhile, the neighborhood induced by the nearest granular ball can fully cover the reliable local sample group while remaining compact, which helps to enhance robustness and improve the overall classification accuracy.

## C. Effect of varying noise levels.

To evaluate the impact of different noise levels, multiple noise intensities were considered, with the results presented in Fig. 4. Overall, as the noise level increases, most algorithms exhibit varying degrees of performance degradation, highlighting the sensitivity of KNN-based methods to noise interference. Notably, OneStepKNN, adaKNNGHIS, and adaKNN2GHIS show the most significant fluctuations, particularly on datasets such as balance-scale and phoneme, where their accuracy drops substantially, limited in robustness to noisy conditions.

In contrast, GBKNN demonstrates notable stability and robustness. Across all 12 datasets, GBKNN consistently maintains high accuracy under high-noise conditions, with minimal performance decline. For example, on the ‘haberman’, and ‘rssi’ datasets, GBKNN maintains an accuracy above 0.75 even when noise reaches 30%. On the ‘heart1’ dataset, GBKNN consistently outperforms all other compared methods across all noise levels. This notable advantage is mainly attributed to the group-based decision mechanism of GBKNN. Unlike methods that rely on individual sample points, coarse-grained granular balls reduce the impact of noise, thereby providing more reliable and robust local information. Furthermore, the neighborhood radius is defined as the distance between the test sample and the farthest sample within the nearest granular ball, thus constructing an adaptive neighborhood centered at the test sample. Such a neighborhood can fully cover the relevant samples within the nearest granular ball while maintaining good local compactness, thereby effectively enhancing the discriminative stability of the model in noisy environments. This is also a key reason why GBKNN can maintain high accuracy and exhibit strong noise resistance even under high noise levels.

In comparison, traditional methods such as KNN, LMKNN, and PLKNN perform reasonably well in low-noise scenarios but degrade rapidly as noise increases, reflecting their lack of mechanisms to handle data perturbations. NaNEKNN and SMKNN exhibit performance close to GBKNN on specific datasets such as ‘mushroom’ and ‘fourclass’, but are generally less stable. Additionally, MGNR performs well on datasets like ‘heart1’ and ‘anneal’, suggesting that its natural neighbor strategy is effective in some tasks. However, its performance still suffers under high noise due to susceptibility to false neighbors. In summary, GBKNN exhibits strong adaptability and noise resilience across diverse data distributions and noise levels. It stands out as a leading approach among existing KNN variants in terms of both stability and accuracy, making it particularly suitable for noise-sensitive or structurally complex tasks.

<table><tr><td>-+- KNN</td><td>--NaNEKNN</td><td>-o-adaKNNGIHS</td><td>-+- adaKNN</td><td>--adaKNN2</td></tr><tr><td>-0- adaKNN2GIHS</td><td>-+- SMKNN</td><td>-LMKNN</td><td>-o- OneStepKNN</td><td>-+- PLKNN</td></tr><tr><td>-- MGNR</td><td>-GBKNN 2019</td><td>-0- GBKNN_p</td><td>GBKNN</td><td></td></tr></table>

![](images/1c8a0da9dd3f2c00558ba56109381869789171ef57ffad35335891670a262a1f.jpg)  
Fig. 4: Accuracy of different methods on six datasets under different noise conditions.

## D. Runtime Analysis

As shown in Table III, the runtime comparison of various algorithms under noise-free conditions is conducted on 12 datasets. Overall, traditional KNN demonstrates relatively high computational efficiency on some small-scale datasets, such as ‘heart1’, ‘cleveland’, and ‘haberman’, where it achieves relatively short runtimes. Meanwhile, several improved KNNbased methods also exhibit certain efficiency advantages on some datasets. In contrast, GBKNN p and GBKNN incur slightly higher runtimes on small-scale datasets, mainly due to the additional granular ball generation process before classification, which introduces extra preprocessing overhead. However, as the data scale increases, the efficiency advantage of granular ball-based methods becomes more evident. On datasets such as ‘phoneme’, ‘Anuran’, ‘mushroom’, ‘anneal’, and ‘Skin NonSkin’, both GBKNN p and GBKNN achieve significantly lower runtimes than most competing algorithms. For example, on the ‘anneal’ dataset, the runtime of GBKNN is only 0.3576, which is much faster than 4.2327 for GBKNN 2019 and 30.2779 for the traditional KNN. Similarly, on ‘phoneme’, ‘mushroom’, and ‘Skin NonSkin’, GBKNN achieves runtimes of 2.4616, 2.4396, and 3.2114, respectively, demonstrating strong computational efficiency. Overall, the granular ball-based representation effectively reduces the computational cost caused by sample-wise operations, leading to better scalability on medium and large scale datasets.

Further comparison among GBKNN 2019, GBKNN p, and GBKNN shows that GBKNN p is generally slightly faster than GBKNN, while GBKNN consistently outperforms GBKNN 2019 in terms of runtime. This difference mainly stems from their distinct granular ball generation and decision mechanisms. Specifically, GBKNN 2019 improves efficiency by replacing sample points with granular balls, but its splitting strategy is relatively inefficient. GBKNN p adopts a coarse division and controls granular ball splitting based on purity, and directly performs classification using the nearest granular ball, resulting in a simpler computational process. In contrast, GBKNN employs the Fisher criterion to guide granular ball splitting and stopping, enabling more refined partitioning. In the prediction stage, it constructs an adaptive neighborhood centered at the test sample based on the nearest granular ball rather than directly using it for classification. Although this neighborhood construction introduces additional computational cost, it also enhances classification performance.

In summary, the results in Table III indicate that both GBKNN p and GBKNN achieve favorable time efficiency while maintaining high classification performance. GBKNN p is more efficient in terms of computational speed, whereas GBKNN achieves a better balance between efficiency and discriminative power, demonstrating strong practical value.

## E. Results of large-scale datasets

Furthermore, to further validate the applicability of the proposed method in large-scale scenarios, experiments are conducted on five large-scale datasets, including ‘Santander’, ‘creditcard’, ‘covertype’, ‘Higgs’, and ‘supersymmetry’, and the results are reported in the table. Due to the large sample sizes and high computational cost of these datasets, some competing algorithms encounter issues such as excessive memory consumption or prohibitively long runtimes, making it difficult to complete the experiments within acceptable resource and time constraints. Therefore, complete comparison results are not available on these datasets. In this setting, the analysis primarily focuses on the classification performance and robustness of GBKNN on large-scale datasets.

TABLE IV: Accuracy and time performance of GBKNN on large-scale datasets at different noise levels.
<table><tr><td>Data</td><td>Santander</td><td>creditcard</td><td>covertype</td><td>Higgs</td><td>supersymmetry</td></tr><tr><td>0%</td><td>0.8979</td><td>0.9991</td><td>0.8183</td><td>0.5117</td><td>0.6973</td></tr><tr><td>5%</td><td>0.8979</td><td>0.9989</td><td>0.8179</td><td>0.5139</td><td>0.6948</td></tr><tr><td>10%</td><td>0.8979</td><td>0.9990</td><td>0.8170</td><td>0.5119</td><td>0.6920</td></tr><tr><td>15%</td><td>0.8979</td><td>0.9991</td><td>0.8126</td><td>0.5144</td><td>0.6932</td></tr><tr><td>20%</td><td>0.8979</td><td>0.9987</td><td>0.8153</td><td>0.5122</td><td>0.6926</td></tr><tr><td>25%</td><td>0.8979</td><td>0.9986</td><td>0.8053</td><td>0.5220</td><td>0.6921</td></tr><tr><td>30%</td><td>0.8979</td><td>0.9980</td><td>0.7973</td><td>0.5197</td><td>0.6857</td></tr><tr><td>Time(s)</td><td>4183.4577</td><td>7367.7777</td><td>11644.1096</td><td>17343.2118</td><td>17271.1339</td></tr></table>

As shown in Table IV, GBKNN maintains relatively stable performance under different noise levels. For example, on the ‘Santander’ dataset, the accuracy remains at 0.8979 across the noise range from 0% to 30%; on the ‘creditcard dataset, the accuracy consistently stays above 0.9980; and on the ‘Higgs’ dataset, the accuracy remains stable within the range of 0.5117–0.5220 with only minor fluctuations. For the ‘covertype’ and ‘supersymmetry’ datasets, although accuracy declines as noise level increases, it remains stable overall. These results indicate that GBKNN retains strong robustness and applicability when dealing with large-scale and high-complexity data. Taken together with the average runtime of all noise observations, it can be seen that although the algorithm still requires a certain amount of computation time for ultra-large datasets, it can complete the training and prediction processes under limited computational resources, demonstrating good scalability. Overall, these findings further verify the effectiveness of GBKNN in large-scale data scenarios and highlight the advantage of granular ball representation in reducing computational burden and improving algorithm feasibility for large datasets.

## F. Ablation Studies

As shown in Table V, all key designs improve the final <sub>performance, and the complete model integrating</sub> √<sub>n-based</sub> coarse initialization, the Fisher criterion, the lower bound $T _ { L }$ of quality, weighted boundary distance $W _ { d } ,$ and neighborhood decision achieves the best average result of 0.8217. To ensure a fair comparison in the ablation study, we fixed the $\sqrt { n }$ initial cluster centers and analyzed the effect of different designs on model performance based on this setting. Subsequently, according to the proportion of different class samples, the farthest sample from the already selected initial points was iteratively chosen until $\sqrt { n }$ initial points were selected. By prioritizing the selection of majority-class samples and incorporating a farthest-point strategy based on global distribution, the initial granular balls were ensured to cover the entire data space, leading to a more balanced granular ball division and reducing initial partitioning imbalance.

In the ablation study on the ${ \sqrt { n } } .$ -based coarse division, we compared the traditional granular ball initialization method, which starts the iterative process from the whole dataset, with the ${ \sqrt { n } } -$ -based coarse initialization strategy. The results are shown in Table V (Rows 4 and 6). It can be seen that introducing coarse division leads to higher classification accuracy on most datasets, with improvements exceeding 5%–10% on some datasets. The reason is that dividing the whole dataset into $\sqrt { n }$ coarse balls provides a simpler initial representation for the original complex distribution, which helps reduce the effect of nonlinear data distributions and avoids premature convergence to overly large granular balls, thereby providing a more reasonable basis for adaptive granular ball generation.

TABLE III: Running time (s) of different methods under noise-free conditions.
<table><tr><td>Data</td><td>monks-2</td><td>heart1</td><td>cleveland</td><td>haberman</td><td>balance-scale</td><td>fourclass</td><td>phoneme</td><td>Anuran</td><td>mushroom</td><td>anneal</td><td>rssi</td><td>Skin Nonskin</td></tr><tr><td>KNN</td><td>0.0290</td><td>0.0630</td><td>0.0634</td><td>0.0530</td><td>0.1620</td><td>0.2831</td><td>9.3909</td><td>33.9094</td><td>30.2797</td><td>0.3682</td><td>0.3942</td><td>28.3518</td></tr><tr><td>NaNEKNN</td><td>0.0240</td><td>0.1620</td><td>0.0810</td><td>0.0806</td><td>0.1620</td><td>0.2066</td><td>10.2117</td><td>48.7547</td><td>14.8327</td><td>0.6892</td><td>0.4001</td><td>25.5310</td></tr><tr><td>adaKNNGIHS</td><td>0.3621</td><td>0.6977</td><td>0.5754</td><td>0.6663</td><td>2.4360</td><td>4.2957</td><td>149.0276</td><td>269.1356</td><td>346.7465</td><td>3.7422</td><td>6.4180</td><td>455.3578</td></tr><tr><td>adaKNN</td><td>0.3375</td><td>0.7073</td><td>0.5743</td><td>0.6635</td><td>2.4263</td><td>4.3463</td><td>150.0538</td><td>271.5230</td><td>350.7985</td><td>3.7151</td><td>6.3709</td><td>453.9935</td></tr><tr><td>adaKNN2</td><td>0.1422</td><td>0.3383</td><td>0.3327</td><td>0.3172</td><td>1.2322</td><td>2.3371</td><td>74.4621</td><td>133.2459</td><td>175.4169</td><td>1.9179</td><td>3.2745</td><td>223.5811</td></tr><tr><td>adaKNN2GIHS</td><td>0.1400</td><td>0.3355</td><td>0.3329</td><td>0.3169</td><td>1.2068</td><td>2.1702</td><td>75.1372</td><td>132.3127</td><td>173.2620</td><td>1.8857</td><td>3.2012</td><td>221.6835</td></tr><tr><td>SMKNN</td><td>0.0150</td><td>0.0450</td><td>0.0236</td><td>0.0420</td><td>0.1950</td><td>0.5251</td><td>17.0071</td><td>32.0827</td><td>31.3878</td><td>0.3066</td><td>0.5468</td><td>42.4478</td></tr><tr><td>LMKNN</td><td>0.0160</td><td>0.0470</td><td>0.0240</td><td>0.0420</td><td>0.2017</td><td>0.6069</td><td>17.4207</td><td>34.4481</td><td>32.1668</td><td>0.3418</td><td>0.6053</td><td>44.9805</td></tr><tr><td>OneStepKNN</td><td>0.1097</td><td>0.1210</td><td>0.0581</td><td>0.0702</td><td>0.2921</td><td>0.4803</td><td>70.4204</td><td>156.3371</td><td>224.2986</td><td>0.3792</td><td>0.9113</td><td>334.7534</td></tr><tr><td>PLKNN</td><td>0.0290</td><td>0.0746</td><td>0.0360</td><td>0.0820</td><td>0.3993</td><td>0.9231</td><td>31.0830</td><td>45.8409</td><td>55.3659</td><td>0.4587</td><td>0.9540</td><td>79.5953</td></tr><tr><td>MGNR</td><td>0.0420</td><td>0.0630</td><td>0.0460</td><td>0.0540</td><td>0.1513</td><td>0.1960</td><td>3.8934</td><td>7.9492</td><td>6.5127</td><td>0.2326</td><td>0.2831</td><td>7.7279</td></tr><tr><td>GBKNN_2019</td><td>3.0944</td><td>4.7226</td><td>5.9680</td><td>5.1024</td><td>5.7159</td><td>1.3050</td><td>41.9653</td><td>8.8376</td><td>2.7996</td><td>4.4277</td><td>8.2658</td><td>3.7593</td></tr><tr><td>GBKNN_p</td><td>0.0990</td><td>0.1140</td><td>0.1210</td><td>0.1066</td><td>0.2160</td><td>0.1860</td><td>2.1095</td><td>2.0994</td><td>2.0175</td><td>0.2560</td><td>0.2826</td><td>2.8242</td></tr><tr><td>GBKNN</td><td>0.1130</td><td>0.1370</td><td>0.1320</td><td>0.1270</td><td>0.2560</td><td>0.2310</td><td>2.4616</td><td>4.3782</td><td>2.4396</td><td>0.3676</td><td>0.3490</td><td>3.2114</td></tr></table>

TABLE V: Comparison of classification accuracy of GBKNN under different key modules.
<table><tr><td>√n Purity fisher minball minball_Wd TL Neighborhood monks-2 heart1 cleveland haberman balance-scale fourclass phoneme Anuran mushroom anneal</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>rssi</td><td>Skin Nonskin</td><td>Ave.</td></tr><tr><td>√ √ √</td><td></td><td></td><td>√</td><td></td><td></td><td>0.4832</td><td>0.6798 0.4483</td><td></td><td>0.7423</td><td>0.8697</td><td>0.9759</td><td>0.8180</td><td>0.9798</td><td>0.9916</td><td>0.8915 0.8429</td><td></td><td>0.9880 0.8093</td></tr><tr><td>√</td><td></td><td></td><td>√</td><td></td><td></td><td>0.4916 0.6847</td><td>0.4581</td><td>0.7372</td><td></td><td>0.8697</td><td>0.9759</td><td>0.8182</td><td>0.9797</td><td>0.9917</td><td>0.8870 0.8470</td><td>0.9877</td><td>0.8107</td></tr><tr><td>√</td><td>√</td><td></td><td>√</td><td></td><td></td><td>0.5</td><td>0.7266 0.4778</td><td>0.7066</td><td></td><td>0.8754</td><td>0.9801</td><td>0.8103</td><td>0.9780</td><td>0.9897</td><td>0.8915 0.8490</td><td>0.9877</td><td>0.8144</td></tr><tr><td></td><td>√</td><td></td><td>√</td><td>√</td><td></td><td>0.6933 0.6059</td><td>0.4532</td><td>0.6301</td><td></td><td>0.8229</td><td>0.9352</td><td>0.8057</td><td>0.9666</td><td>0.9607</td><td>0.9105 0.7949</td><td>0.9211</td><td>0.7917</td></tr><tr><td>√</td><td>√</td><td></td><td></td><td>√</td><td></td><td>0.5294</td><td>0.7463 0.4532</td><td>0.7321</td><td></td><td>0.9234</td><td>0.9784</td><td>0.8248</td><td>0.9589</td><td>0.9829</td><td>0.8716 0.8558</td><td>0.9828</td><td>0.8200</td></tr><tr><td>√</td><td>√</td><td></td><td>√</td><td>√</td><td></td><td>0.53360.7340</td><td>0.4581</td><td>0.7526</td><td></td><td>0.9246</td><td>0.9809</td><td>0.8263</td><td>0.9436</td><td>0.9863</td><td>0.8761 0.8626</td><td>0.9820</td><td>0.8217</td></tr></table>

The ablation experiment results for Purity and Fisher control of granular ball generation are shown in Table V (Rows 2 and 3). The results show that replacing purity with the Fisher criterion as the control criterion for granular ball generation further improves the average performance of the model by 3.7%. This indicates that, compared with purity, which only considers the proportion of the dominant class, the Fisher criterion can simultaneously capture inter-class separability and intra-class compactness within a granular ball, thus enabling more accurate evaluation of ball quality and more effective guidance for adaptive granular ball division.

As shown in Table V (Rows 5 and 6), after introducing the quality lower-bound mechanism (i.e., T ), the classification accuracy improves on most datasets. In particular, the accuracy increases by 2.05% and 1.32% on the ‘haberman’ and ‘rssi datasets, respectively. This mechanism further refines those granular balls that do not satisfy the splitting condition but still have relatively low quality, thereby effectively improving the classification accuracy of GBKNN across different datasets.

The effectiveness of the weighted boundary distance can be verified by comparing Rows 1 and 2 in Table V. Under the same purity-based granular ball generation criterion, the average performance improves from 0.8093 to 0.8107. Unlike the unweighted version, which treats all granular balls equally during prediction, the weighted boundary distance reduces the influence of small noisy balls and enhances the role of larger balls in nearest-ball search. Since larger granular balls usually contain more representative samples and are less likely to be dominated by local noise, this mechanism improves the reliability of nearest-ball selection and thus enhances the robustness of the model in the presence of local noise and outliers.

As shown in Rows 3 and 6 of Table V, the decision strategy based on constructing a neighborhood around the test point outperforms direct decision making based on the nearest granular ball. Since this neighborhood is induced by the nearest granular ball, and its radius is defined as the distance from the test sample to the farthest sample within that ball, it can fully cover the reliable local sample group while remaining compact. More importantly, this mechanism allows k to be adaptively determined by the actual number of samples contained in the neighborhood, rather than being manually preset, which is more consistent with the basic KNN principle of making decisions based on local neighborhoods.

## VI. CONCLUSION

This paper proposes a new adaptive granular-ball k-nearest neighbor model. The method fits the original data distribution by generating granular balls. It first simplifies local splitting through $\sqrt { n }$ coarse divisions, and then achieves full adaptivity by controlling the splitting process with the weighted sum of the Fisher values of child balls. During coarse-granular ball partitioning, a density-based splitting criterion is introduced, and the $\sqrt { n }$ divisions strategy is optimized through rigorous mathematical analysis. In the decision stage, GBKNN constructs an adaptive neighborhood for each test sample based on the weighted nearest granular ball, so that the value of k is dynamically determined by the actual number of samples in the neighborhood, avoiding the dependence of traditional KNN on a fixed parameter setting. Extensive experiments on multiple benchmark datasets show that GBKNN outperforms traditional KNN and other competing methods in classification accuracy, time efficiency, and noise robustness, demonstrating strong overall advantages. In future work, we will further explore more efficient granular ball partition strategies to continuously improve classification performance. We will also extend GBKNN to broader application scenarios, such as imbalanced classification, open-set recognition, and semisupervised learning.

## REFERENCES

[1] P. Soucy and G. W. Mineau, “A simple knn algorithm for text categorization,” in Proceedings 2001 IEEE International Conference on Data Mining. IEEE, 2001, pp. 647–648.

[2] K. K. Paliwal and P. Rao, “Application of k-nearest-neighbor decision rule in vowel recognition,” IEEE Transactions on Pattern Analysis and Machine Intelligence, no. 2, pp. 229–231, 1983.

[3] F. Zhang and J. Feng, “High-resolution mobile fingerprint matching via deep joint knn-triplet embedding,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 31, no. 1, 2017.

[4] U. P. Wani, Y. Gatagat, and M. Thalor, “Handwritten character recognition using cnn, knn and svm,” International Journal of Technology Engineering Arts Mathematics Science, vol. 1, no. 2, pp. 2583–1224, 2022.

[5] A. A. A. Ali and S. Mallaiah, “Intelligent handwritten recognition using hybrid cnn architectures based-svm classifier with dropout,” Journal of King Saud University-Computer and Information Sciences, vol. 34, no. 6, pp. 3294–3300, 2022.

[6] J. Zhang, Y. Li, F. Shen, Y. He, H. Tan, and Y. He, “Hierarchical text classification with multi-label contrastive learning and knn,” Neurocomputing, vol. 577, p. 127323, 2024.

[7] M. N. Ab Wahab, A. Nazir, A. T. Z. Ren, M. H. M. Noor, M. F. Akbar, and A. S. A. Mohamed, “Efficientnet-lite and hybrid cnn-knn implementation for facial expression recognition on raspberry pi,” IEEE Access, vol. 9, pp. 134 065–134 080, 2021.

[8] J.-P. Heo, Z. Lin, and S.-E. Yoon, “Distance encoded product quantization for approximate k-nearest neighbor search in high-dimensional space,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 41, no. 9, pp. 2084–2097, 2018.

[9] H. Samet, “K-nearest neighbor finding using maxnearestdist,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 30, no. 2, pp. 243–252, 2007.

[10] B. Zhang and S. N. Srihari, “Fast k-nearest neighbor classification using cluster-based trees,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 26, no. 4, pp. 525–528, 2004.

[11] B. S. Kim and S. B. Park, “A fast k nearest neighbor finding algorithm based on the ordered partition,” IEEE Transactions on Pattern Analysis and Machine Intelligence, no. 6, pp. 761–766, 1986.

[12] J. E. Goin, “Classification bias of the k-nearest neighbor algorithm,” IEEE Transactions on Pattern Analysis and Machine Intelligence, no. 3, pp. 379–381, 1984.

[13] N. Ukey, Z. Yang, B. Li, G. Zhang, Y. Hu, and W. Zhang, “Survey on exact knn queries over high-dimensional data space,” Sensors, vol. 23, no. 2, p. 629, 2023.

[14] A. K. Ghosh, “On optimum choice of k in nearest neighbor classification,” Computational Statistics & Data Analysis, vol. 50, no. 11, pp. 3113–3123, 2006.

[15] C. Domeniconi, J. Peng, and D. Gunopulos, “Locally adaptive metric nearest-neighbor classification,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 24, no. 9, pp. 1281–1285, 2002.

[16] G. Bhattacharya, K. Ghosh, and A. S. Chowdhury, “An affinity-based new local distance function and similarity measure for knn algorithm,” Pattern Recognition Letters, vol. 33, no. 3, pp. 356–363, 2012.

[17] M. L. Yiu and N. Mamoulis, “Reverse nearest neighbors search in ad hoc subspaces,” IEEE Transactions on Knowledge and Data Engineering, vol. 19, no. 3, pp. 412–426, 2007.

[18] S. Pourbahrami and M. Hashemzadeh, “A geometric-based clustering method using natural neighbors,” Information Sciences, vol. 610, pp. 694–706, 2022.

[19] Q. Zhu, J. Feng, and J. Huang, “Natural neighbor: A self-adaptive neighborhood method without parameter k,” Pattern Recognition Letters, vol. 80, pp. 30–36, 2016.

[20] J. Yang, L. Yang, J. Zhang, Q. Liang, W. Wang, D. Tang, and T. Liu, “Gnan: A natural neighbor search algorithm based on universal gravitation,” Pattern Recognition, vol. 146, p. 110063, 2024.

[21] S. Xia, Y. Liu, X. Ding, G. Wang, H. Yu, and Y. Luo, “Granular ball computing classifiers for efficient, scalable and robust learning,” Information Sciences, vol. 483, pp. 136–152, 2019.

[22] S. Xia, X. Dai, G. Wang, X. Gao, and E. Giem, “An efficient and adaptive granular-ball generation method in classification problem,” IEEE Transactions on Neural Networks and Learning Systems, vol. 35, no. 4, pp. 5319–5331, 2022.

[23] J. Xie, X. Xiang, S. Xia, L. Jiang, G. Wang, and X. Gao, “Mgnr: A multigranularity neighbor relationship and its application in knn classification and clustering methods,” IEEE Transactions on Pattern Analysis and Machine Intelligence, 2024.

[24] Q. Huang and A. K. Tung, “Lightweight-yet-efficient: Revitalizing balltree for point-to-hyperplane nearest neighbor search,” in 2023 IEEE 39th International Conference on Data Engineering (ICDE). IEEE, 2023, pp. 436–449.

[25] Y. Peng, B. Choi, T. N. Chan, and J. Xu, “Lan: Learning-based approximate k-nearest neighbor search in graph databases,” in 2022 IEEE 38th International Conference on Data Engineering (ICDE), 2022, pp. 2508–2521.

[26] C. Gong, Y. Li, Y. Liu, P.-h. Wang, and Y. You, “Joint evidential k-nearest neighbor classification,” in 2022 IEEE 38th International Conference on Data Engineering (ICDE). IEEE, 2022, pp. 2113–2126.

[27] M. R. Brito, E. L. Chavez, A. J. Quiroz, and J. E. Yukich, “Connec-´ tivity of the mutual k-nearest-neighbor graph in clustering and outlier detection,” Statistics & Probability Letters, vol. 35, no. 1, pp. 33–42, 1997.

[28] S. S. Mullick, S. Datta, and S. Das, “Adaptive learning-based knearest neighbor classifiers with resilience to class imbalance,” IEEE Transactions on Neural Networks and Learning Systems, vol. 29, no. 11, pp. 5713–5725, 2018.

[29] S. Zhang and J. Li, “Knn classification with one-step computation,” IEEE Transactions on Knowledge and Data Engineering, vol. 35, no. 3, pp. 2711–2723, 2021.

[30] D. S. Jodas, L. A. Passos, A. Adeel, and J. P. Papa, “Pl-k nn: A parameterless nearest neighbors classifier,” in 2022 29th International Conference on Systems, Signals and Image Processing (IWSSIP). IEEE, 2022, pp. 1–4.

[31] S. Zhu, L. Xie, M. Zhang, R. Gao, and Y. Xie, “Distributionally robust weighted k-nearest neighbors,” Advances in Neural Information Processing Systems, vol. 35, pp. 29 088–29 100, 2022.

[32] S. M. Ayyad, A. I. Saleh, and L. M. Labib, “Gene expression cancer classification using modified k-nearest neighbors technique,” Biosystems, vol. 176, pp. 41–51, 2019.

[33] Q. Xie, Q. Zhang, S. Xia, F. Zhao, C. Wu, G. Wang, and W. Ding, “Gbg++: A fast and stable granular ball generation method for classification,” IEEE Transactions on Emerging Topics in Computational Intelligence, 2024.

[34] S. Xia, X. Lian, G. Wang, X. Gao, J. Chen, and X. Peng, “Gbsvm: an efficient and robust support vector machine framework via granularball computing,” IEEE Transactions on Neural Networks and Learning Systems, 2024.

[35] J. Xie, C. Hua, S. Xia, Y. Cheng, G. Wang, and X. Gao, “W-gbc: An adaptive weighted clustering method based on granular-ball structure,” in 2024 IEEE 40th International Conference on Data Engineering (ICDE). IEEE, 2024, pp. 914–925.

[36] S. Xia, X. Ma, Z. Liu, C. Liu, S. Zhao, and G. Wang, “Graph coarsening via supervised granular-ball for scalable graph neural network training,” in Proceedings ofthe AAAI Conference on Artificial Intelligence, vol. 39, no. 12, 2025, pp. 12 872–12 880.

[37] S. Xia, C. Wang, G. Wang, X. Gao, W. Ding, J. Yu, Y. Zhai, and Z. Chen, “Gbrs: A unified granular-ball learning model of pawlak rough set and neighborhood rough set,” IEEE Transactions on Neural Networks and Learning Systems, 2023.

[38] J. Liu, H. Jianye, Y. Ma, and S. Xia, “Unlock the cognitive generalization of deep reinforcement learning via granular ball representation,” in Forty-first International Conference on Machine Learning, 2024.

[39] H. Kuai, J. Chen, X. Tao, L. Cai, K. Imamura, H. Matsumoto, P. Liang, and N. Zhong, “Never-ending learning for explainable brain computing,” Advanced Science, vol. 11, no. 24, p. 2307647, 2024.

[40] G. Le Caer, “Circumspheres of sets of n+1 random points in the d-¨ dimensional euclidean unit ball (1 ≤ n ≤ d),” Journal of Mathematical Physics, vol. 58, no. 5, 2017.

[41] B. K. Sriperumbudur, A. Gretton, K. Fukumizu, B. Scholkopf, and¨ G. R. Lanckriet, “Hilbert space embeddings and metrics on probability measures,” The Journal of Machine Learning Research, vol. 11, pp. 1517–1561, 2010.