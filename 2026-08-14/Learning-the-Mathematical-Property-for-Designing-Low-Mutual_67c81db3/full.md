# Learning the Mathematical Property for Designing Low Mutual Coherence Binary Sensing Matrices

Rekha<sup>a</sup>, Santosh Singh<sup>a</sup>, and S. K. Neogy<sup>b</sup>

<sup>a</sup>Department of Mathematics, Shiv Nadar Institution of Eminence, Delhi, India. (e-mail: re824@snu.edu.in)

<sup>b</sup>Centre for Research on Economics and Data Analysis (CREDA), Indian Statistical Institute, New Delhi, India.

## Abstract

The essential part for the success of almost all learning-based methods relies on the quality and size of the data sets chosen, based on the application and problem. In this research work, we are constructing the sensing matrix, which is essential for the success of the compressive sensing technique. We have chosen a learning-based technique for the construction of the sensing matrix. The novelty and uniqueness of the proposed technique is that it does not use any data set and also does not use a specific application. It uses the mathematical property/constraint for the construction of the sensing matrix for the perfect recovery of the signal. The perfect recovery of signals is an old and still very challenging problem in real-world applications. There have been many techniques for the perfect recovery of signals. In late 2000, compressive sensing became a popular mathematical tool for the perfect recovery of sparse signals. The core of the compressive technique is the construction of the sensing matrix, which satisfies certain special properties such as restricted isometry property (RIP), null space property (NSP), and spark property (SP). All these properties are NP-hard problems and hence computationally challenging to solve. For all practical purposes, the construction of the sensing matrix needs to achieve low mutual coherence to achieve the perfect recovery of the signals. We have used a neural network for the construction of the sensing matrix, and this framework constructs a binary sensing matrix with low mutual coherence. The entries in the matrix are generated through a shared underlying rule. The proposed architecture is simple and does not use large-scale training data sets. Such uniqueness and novelty bring a drastic reduction in computational cost, and also, for the first time in literature, the use of a mathematical property for defining the loss function. In this proposed research work, the mutual coherence property has been used in the neural network framework. Such a neural network framework brings generality (application dependency), robustness, and reduces storage requirements.

The simulation results demonstrate the efectiveness of the proposed approach. The results show that the proposed learning-based method consistently outperforms

popular conventional methods, which are based on deterministic as well as randommatrix techniques.

Keywords: Compressive sensing, sensing matrix, mutual incoherence property, artificial neural network, binary matrices, random matrices.

## 1 Introduction

Compressive sensing (CS) has emerged as an eficient signal acquisition framework that enables the recovery of sparse signals from a reduced number of measurements. Among the various components of compressive sensing, the design of the sensing matrix plays a fundamental role in eficiently acquiring the signal and in determining the accuracy and reliability of signal reconstruction. For successful sparse signal recovery, the sensing matrix must satisfy certain theoretical conditions that have been extensively studied in the literature. Some of the most important properties includes the spark [17, 18], Null Space Property (NSP) [14, 17], Restricted Isometry Property (RIP) [9, 10, 12], and Mutual Incoherence Property (MIP) [6, 7, 20]. These properties provide theoretical guarantees for sparse signal reconstruction and have motivated the development of several techniques for constructing eficient sensing matrices.

Among the theoretical conditions used to evaluate sensing matrices, the Restricted Isometry Property (RIP) and the Mutual Incoherence Property (MIP) are the most widely studied. Although the RIP provides strong theoretical guarantees for sparse recovery, verifying whether a sensing matrix satisfies this property is an NP-hard problem [31]. In contrast, the Mutual Incoherence Property is computationally tractable and can be evaluated more eficiently. Therefore, this paper primarily focuses on the Mutual Incoherence Property because of its practical applicability and analytical convenience. A sensing matrix with low mutual coherence generally enhances the uniqueness and stability of sparse signal reconstruction. David Donoho and Michael Elad [18] further emphasized the significance of incoherence by showing that sparse recovery performance improves when the sensing matrix Φ is suficiently incoherent with the sparsifying basis Ψ. Consequently, the design of sensing matrices with minimum mutual coherence has become an important research direction in compressive sensing.

Over the years, various approaches have been proposed for sensing matrix design, accounting for factors such as computational complexity, storage eficiency, reconstruction accuracy, and adaptability to the underlying signal structure. The commonly studied sensing matrix constructions include random matrices, deterministic matrices, and learning-based approaches. Each category ofers distinct advantages and trade-ofs, making it suitable for diferent compressed sensing applications and implementation constraints.

Among these, random sensing matrices generated from Gaussian or Bernoulli ensembles are widely used because they satisfy important theoretical conditions with high probability and provide strong guarantees for sparse signal recovery [3, 8, 9, 11]. Random matrices are largely incoherent with most sparsifying bases. However, due to their dense and unstructured nature, such matrices require significant storage space and lead to high computational complexity during signal acquisition and reconstruction [16, 27].

To address these limitations, deterministic sensing matrix constructions have been investigated extensively [19, 20, 32, 34, 35]. These approaches aim to minimize mutual coherence between the sensing and sparsifying matrices. Many of these methods formulate the sensing matrix design problem as an optimization problem in which the corresponding Gram matrix is forced to approximate an ideal target Gram matrix. Such optimization problems are generally solved using iterative algorithms. While these methods often provide improved performance compared to conventional random matrices, their construction procedures remain computationally expensive and mathematically complex [16, 35].

In recent years, deep learning techniques have attracted significant attention in the field of compressive sensing, demonstrating remarkable success in both signal reconstruction and sensing matrix construction [1, 5, 25, 33]. Most learning-based approaches jointly optimize the sensing matrix and reconstruction network within a unified training framework. In these methods, both the sensing and recovery stages are represented as trainable layers of a neural network and are optimized using large training datasets. By learning the underlying structure of signals directly from data, these approaches often achieve superior reconstruction performance. However, they typically rely heavily on extensive training data and computationally intensive architectures. Furthermore, learning-based methods primarily focus on data-driven optimization rather than explicitly enforcing the theoretical properties required for compressive sensing [5,26,36]. As a result, theoretical guarantees such as incoherence or RIP are often not well established.

Motivated by the limitations of existing sensing matrix construction approaches, this article proposes a novel sensing matrix design framework that primarily aims to learn binary sensing matrices by optimizing coherence-based mathematical properties. As far as we know, this property-based learning approach is novel and not limited to this area; it can be applied across multiple disciplines. The main contributions of this article are summarized as follows:

• A novel framework is proposed for constructing sensing matrices with reduced mutual coherence, which is one of the most important properties in compressive sensing.

• The proposed method generates binary sensing matrices whose entries are constructed according to a common rule-based mechanism, which reduces computational complexity and storage requirements

• Unlike conventional learning-based approaches that employ deep, computationally intensive architectures, the proposed framework uses a simple neural network to eficiently generate the sensing matrix.

• The proposed model uses a combined loss function, which favors incoherence.

• Experimental results demonstrate that the proposed sensing matrix achieves lower mutual coherence compared to several existing sensing matrix construction methods.

## 2 Background and Preliminaries

In the first decade of the 2000s, Compressive Sensing (CS) theory emerged as a modern signal sampling technique. Unlike the traditional Nyquist sampling method, which often requires a large number of measurements, CS enables the acquisition of high-dimensional signals using far fewer samples. Despite this reduced sampling, the original signal can still be accurately reconstructed using suitable reconstruction algorithms. The fundamental idea behind compressive sensing is the sparsity assumption. A signal is considered k-sparse if it contains at most k non-zero coeficients in an appropriate representation domain. Therefore, CS assumes that the signal has a sparse representation in a suitable transform domain, enabling eficient sampling and accurate reconstruction.

Assume $\boldsymbol { s } \in \mathbb { R } ^ { N }$ is the high-dimensional original signal and $\Psi \in \mathbb { R } ^ { N \times N }$ is the sparsifying basis. In the sparsifying domain, the original signal can be represented with only a few coeficients and mathematically can be expressed as,

$$
\begin{array} { r } { s = \Psi x } \end{array}\tag{1}
$$

where $\pmb { x } \in \mathbb { R } ^ { N }$ is the sparse representation of s. Then, instead of measuring every sample, compressive sensing measures the compressed measurement with the help of the sensing matrix $\Phi \in \mathbb { R } ^ { M \times N }$ where $M \ll N$ . Using the sensing matrix, the N-dimensional signal is compressed into M measurements, and the measurement process can be represented as,

$$
\begin{array} { r } { y = \Phi s = \Phi \Psi x = A x } \end{array}\tag{2}
$$

where $\pmb { y } \in \mathbb { R } ^ { M }$ is the low-dimensional measurements. In the above formulation, the matrix Φ is the sensing matrix, and A is popularly known as the efective sensing matrix.

The success of compressive sensing largely depends on three key components: the sparsifying basis, the sensing matrix, and the recovery algorithm. Considerable research has been devoted to the development and improvement of each of these components. Among them, the sensing matrix plays a particularly important role because it directly influences both signal acquisition and reconstruction. Therefore, the design of an eficient sensing matrix is a fundamental aspect of compressive sensing.

## 2.1 Conditions for Perfect Signal Recovery

An efective sensing matrix should be able to capture the most significant information of a signal using only a small number of measurements while preserving the information required for accurate reconstruction. However, not every matrix possesses these desirable properties.

To address this challenge, researchers have proposed several mathematical conditions to enable sensing matrices to eficiently acquire signal information without significant loss, while ensuring reliable signal recovery from the compressed measurements. Among them, the Restricted Isometric Property (RIP) was the most popular suficient condition, which was introduced by Emmanuel Cand\`es and Terence Tao in 2005 [12, 13].

Definition 1 (Restricted Isometry Property). [12] A matrix $\pmb { A } \in \mathbb { R } ^ { M \times N }$ , satisfies the Restricted Isometry Property (RIP) of order k if there exists a isometry constant $\delta _ { k } \in ( 0 , 1 )$ such that

$$
( 1 - \delta _ { k } ) \| { \pmb x } \| _ { 2 } ^ { 2 } \leq \| { \pmb A } { \pmb x } \| _ { 2 } ^ { 2 } \leq ( 1 + \delta _ { k } ) \| { \pmb x } \| _ { 2 } ^ { 2 } .\tag{3}
$$

holds for all k-sparse signals.

RIP preserves the mutual distances between all sparse signals after mapping to the measurement space, ensuring that no two sparse signals map to the same measurement. If the efective sensing matrix A satisfies the restricted isometric property, then this is suficient for a variety of algorithms to be able to successfully recover the original signal. However, verifying whether a given matrix satisfies RIP is itself NP-hard [31]. Whereas the Mutual Incoherence Property (MIP) [18] provides a more practical and trackable approach for constructing the sensing matrix.

A matrix satisfies MIP if its mutual coherence is suficiently small, and mutual coherence measures the similarity between the columns of the sensing matrix; the mathematical definition is given below.

Definition 2 (Mutual Coherence). [15] Let $\pmb { A } = [ a _ { 1 } , a _ { 2 } , . . . a _ { N } ] , a _ { i } \in \mathbb { R } ^ { M }$ , then the mutual coherence of A is the maximum correlation between the columns of A, i.e.,

$$
\mu ( A ) = \operatorname* { m a x } _ { 1 \leq i , j \leq N } \frac { \lvert a _ { i } ^ { T } a _ { j } \rvert } { \lVert a _ { i } \rVert _ { 2 } \lVert a _ { j } \rVert _ { 2 } } .\tag{4}
$$

In 2003, David Donoho and Michael Elad [18], shows that if sparsity of the signal satisfies

$$
k < \frac { 1 } { 2 } \left( \frac { 1 } { \mu ( A ) } + 1 \right) ,\tag{5}
$$

then a k-sparse signal can be recovered exactly using either Basis Pursuit or Orthogonal Matching Pursuit (OMP). This result demonstrates that lower mutual coherence between the sensing matrix and the sparsifying basis allows the accurate reconstruction of signals with relatively denser sparse representations within the compressive sensing framework. Consequently, mutual coherence has emerged as a practical and computationally eficient alternative for evaluating and designing sensing matrices. Therefore, this paper proposes a novel approach to constructing a learning-based sensing matrix that aims to achieve lower mutual coherence.

## 2.2 Some Important Terminologies

In CS, accurate sparse recovery is generally achieved when the columns of the efective sensing matrix exhibit low mutual coherence. Therefore, for a given sparsifying basis Ψ, the objective is to design a sensing matrix Φ such that the coherence among the columns of A is minimized. However, directly minimizing the maximum mutual coherence is dificult because of its non-diferentiable objective. Therefore, to guide the learning process and to measure the performance of the proposed matrix, coherence and related majors were utilized and described below.

• Gram Matrix: For the efective sensing matrix A the corresponding Gram matrix is defined as

$$
G = A ^ { T } A .
$$

The diagonal entries of G represent the squared norms of the columns of A, which become unity when the columns are normalized. The of-diagonal entries represent the pairwise coherence between diferent columns of A. Since lower coherence improves sparse reconstruction performance, the of-diagonal elements of G are ideally expected to be as close to zero as possible.

• Maximum Mutual Coherence: Maximum mutual coherence of A measures the largest coherence value between any two distinct columns of the efective sensing matrix. It is defined as

$$
\mu ( A ) = \operatorname* { m a x } _ { i \neq j } \{ | g _ { i j } | \}
$$

where $g _ { i j }$ denotes the $( i , j )$ -th entry of the Gram matrix G. A smaller value of $\mu ( A )$ indicates lower worst-case coherence and generally leads to improved sparse recovery guarantees.

• Average Mutual Coherence: While maximum mutual coherence considers only the largest pairwise coherence, average mutual coherence evaluates the overall coherence distribution among the columns of the sensing matrix. It is defined as

$$
\mu _ { \mathrm { a v g } } ( A ) = \mathrm { m e a n } \left( \sum _ { i \neq j } \left| g _ { i j } \right| \right)
$$

A lower value of average mutual coherence implies that, on average, the columns of the sensing matrix are weakly correlated, thereby contributing to more stable and reliable sparse signal reconstruction.

• Total Mutual Coherence: Total mutual coherence measures the cumulative coherence among all distinct column pairs of the efective sensing matrix and is expressed

$$
\mu _ { \mathrm { t o t a l } } ( A ) = \sum _ { i \neq j } | g _ { i j } |
$$

Minimizing this quantity encourages a globally incoherent sensing matrix structure, which is beneficial for robust compressive sensing performance.

• Equiangular Tight Frames: Equiangular Tight Frames (ETF) are particularly important because they attain the lowest possible coherence bound under certain conditions. The definition of ETF is given below.

Definition 3 (Equiangular Tight Frame). [30] Let $\pmb { A } = [ a _ { 1 } , a _ { 2 } , . . . a _ { N } ]$ be an $M \times N$ matrix. Then A is said to be an equiangular tight frame (ETF) if it satisfies the three conditions below

1. $\| a _ { i } \| = 1 f o r i = 1 , 2 , . . . , N .$

2. There exists a constant d such that, $| \langle a _ { i } , a _ { j } \rangle | = d f o r i \neq j$ , here $\left. a _ { i } , a _ { j } \right.$ represents the inner product.

3. The columns of A form a tight frame, i.e., $\begin{array} { r } { \pmb { A } \pmb { A } ^ { T } = \frac { N } { M } \pmb { \mathrm { I } } } \end{array}$

To encourage highly incoherent measurements, ETF-inspired constraints are incorporated into the proposed sensing matrix learning framework. Such a structure promotes uniform coherence distribution and improves the robustness of sparse signal recovery.

• Binary Sensing Matrices: A sensing matrix is referred to as a binary sensing matrix when its entries are restricted to two discrete values, typically {0, 1} or {−1, 1}. Binary sensing matrices are particularly attractive for practical implementations due to their low storage requirements, reduced computational complexity, and hardwareeficient realization. These characteristics make them well-suited for eficient learning and deployment of sensing matrices in real-world compressive sensing systems. However, maintaining incoherence among the columns with this binary constraint is the main dificulty.

## 3 Existing Sensing Matrix Design

The performance of compressive sensing largely depends on the design of the sensing matrix. An eficient sensing matrix should be able to capture the essential information of a highdimensional sparse signal with a significantly smaller number of measurements while still enabling accurate signal reconstruction. Therefore, the construction of the sensing matrix remains a central research problem in compressive sensing.

As discussed in the above section, most of the theoretical properties, including the Null Space Property (NSP), the Restricted Isometry Property (RIP), and the Spark property, provide strong recovery guarantees. However, verifying or constructing matrices that strictly satisfy them is computationally dificult and often impractical for large-scale applications. Consequently, researchers have focused on more tractable alternatives, among which mutual coherence has emerged as one of the most widely adopted measures for sensing matrix design.

Based on coherence minimization and practical implementation considerations, sensing matrices can be broadly categorized into random, deterministic, and data-driven learned matrices. Each class ofers diferent trade-ofs in terms of recovery performance, computational complexity, storage requirements, and implementation feasibility.

## 3.1 Random Sensing Matrix

Random sensing matrices gained significant attention during the early development of compressive sensing due to their strong theoretical recovery guarantees. In these constructions, matrix entries are typically sampled from random distributions such as Gaussian or Bernoulli distributions. Random matrices satisfy the RIP with high probability and exhibit low mutual coherence with most sparsifying bases [3].

The major advantage of random sensing matrices lies in their universality and strong reconstruction capability. However, despite their theoretical efectiveness, they sufer from several practical limitations, including high storage complexity, large memory requirements, and dificulties in hardware implementation for large-scale systems.

## 3.2 Deterministic Sensing Matrix

To address the practical limitations of random constructions, researchers explored deterministic sensing matrix design strategies. Following the influential work of Michel Elad [20] on coherence optimization, several iterative and algebraic approaches were proposed to construct matrices with progressively reduced mutual coherence [32, 34, 35].

Deterministic matrices ofer advantages such as reproducibility, reduced randomness, and improved implementation feasibility. In many practical scenarios, they also demonstrate competitive or superior performance compared to purely random constructions. However, these methods often rely on complex algebraic formulations or computationally intensive iterative optimization procedures, which may reduce flexibility and increase design complexity.

## 3.3 Data-driven Learned Sensing Matrix

Recent advances in machine learning and deep neural networks have significantly influenced compressed sensing research, leading to the development of data-driven sensing and reconstruction frameworks. Unlike conventional approaches that construct sensing matrices based on analytical principles and theoretical guarantees, learning-based methods optimize sensing and reconstruction processes directly from training data to improve performance for specific

signal classes.

Initially, deep learning was primarily employed for signal reconstruction, while the sensing stage continued to rely on conventional random sensing matrices. Although these approaches achieved notable improvements in reconstruction quality, they did not address the fundamental limitations associated with random sensing matrices, particularly their high storage requirements and computational overhead. Consequently, the storage burden remained largely unchanged despite advances in reconstruction algorithms.

Mousavi et al. [26] employed stacked denoising autoencoders (SDAs) to jointly learn the measurement and recovery processes. This is among the first papers to use a neural network to learn a compressive sensing framework. Their framework was trained using the ImageNet dataset [28] and demonstrated the efectiveness of end-to-end learning for compressed sensing. Nevertheless, the approach relied heavily on extensive training data and computational resources.

Edler et al. [1] later proposed a comprehensive deep learning framework that jointly optimized the acquisition and reconstruction stages. By adapting the sensing process to the data distribution, the method achieved improved reconstruction accuracy compared with conventional approaches. However, as with many learned sensing strategies, theoretical guarantees such as the Restricted Isometry Property (RIP) and low mutual coherence were generally absent.

In 2017, Bora et al. [5] drew attention for introducing a generative-model-based reconstruction framework that exploits the underlying structure of signals. Their approach was trained on extensive datasets, including MNIST, a dataset of 60,000 handwritten digit images [22], and CelebA, a dataset of more than 200,000 celebrity face images [23]. While the method improved reconstruction performance by leveraging learned data priors, the sensing process remained dependent on data-driven training and large-scale datasets.

A successful deep learning-based compressed sensing framework, ReconNet, proposed by Lohit et al. [24], utilized a random Gaussian sensing matrix and focused exclusively on learning an efective reconstruction network. The model was trained using the large-scale ImageNet dataset [21], demonstrating the potential of deep learning for compressed sensing recovery while retaining conventional random measurements.

Zhang et al. [36] proposed ISTA-Net, a structured deep network inspired by the Iterative Shrinkage-Thresholding Algorithm (ISTA) [4]. In this framework, each iteration of the optimization algorithm was unfolded into a neural network layer, and the algorithm parameters were learned directly from data. Trained on the ImageNet dataset [21], the method achieved faster convergence and improved reconstruction performance. However, its efectiveness remained dependent on the availability of large training datasets and substantial computational resources.

A more direct attempt to learn the sensing matrix was made by Shi et al. [29] in 2019 through the CSNet framework, which jointly optimized sensing and reconstruction. The authors learned three diferent sensing matrix types: floating-point, binary, and bipolar matrices. Although binary and bipolar sensing matrices reduce storage requirements compared with dense random matrices, the learned matrices do not provide theoretical guarantees such as RIP or mutual incoherence. Furthermore, the framework was trained on the BSD500 dataset [2], requiring a considerable amount of training data. Since the sensing matrix is optimized for a particular data distribution, its performance may not generalize well to signals that difer significantly from the training set.

Overall, learning-based compressed sensing methods have demonstrated impressive reconstruction performance by adapting to the statistical characteristics of specific datasets. However, most existing approaches either continue to employ random sensing matrices, thereby retaining the associated storage and computational burdens, or rely on learned sensing matrices that require large-scale training data and expensive optimization procedures. Moreover, learned sensing matrices generally lack rigorous theoretical guarantees, such as RIP and mutual incoherence, and their performance is often closely tied to the training distribution, limiting their generality and applicability in lightweight, resource-constrained, or deployment-critical environments.

## 4 Motivation

Although substantial progress has been made in sensing matrix design, existing approaches still face important limitations. Random matrices provide strong theoretical guarantees but require significant storage and are often ineficient for practical implementation. Determin istic constructions improve structural control and reproducibility; however, many of these methods involve complicated algebraic designs or computationally intensive optimization processes. On the other hand, data-driven learned sensing matrices achieve adaptability and improved reconstruction performance but generally depend on deep learning frameworks, large training datasets, and high computational resources.

Consequently, existing methods are unable to simultaneously achieve the desirable properties of low storage complexity, low mutual coherence, simple construction, and lightweight learning. This limitation highlights the need for a more eficient sensing matrix design framework that balances theoretical efectiveness with practical implementation requirements.

Motivated by these challenges, this study introduces a lightweight framework for generating binary sensing matrices via shared neural parameterization. The proposed method is designed to minimize mutual coherence while substantially reducing storage complexity, thereby eliminating the need for deep network architectures and large-scale training datasets. In contrast to conventional learning-based approaches that rely heavily on datadriven structural learning, the proposed framework focuses on learning from the desired sensing properties themselves rather than from training data. This property-oriented learning strategy enhances the generalization performance of the sensing matrix while alleviating the computational and data burdens associated with extensive training. Consequently, the framework ofers an eficient, scalable, and practical solution for compressive sensing applications, particularly in resource-constrained environments.

## 5 Proposed Property-driven Learned Sensing Matrix Design

The performance of a compressive sensing framework is strongly influenced by the properties of the sensing matrix. Conventional random sensing matrices, although widely used, often impose significant storage overhead and present practical challenges for hardware imple mentation. Existing learning-based approaches aim to overcome these limitations; however, they often rely on large-scale training datasets and computationally intensive deep architectures. Moreover, such methods primarily learn the underlying structure of the training data, which limits their generalization and provides only limited theoretical guarantees regarding the properties of the resulting sensing matrix.

To address these challenges, this work proposes a binary sensing matrix constructed through a property-driven learning framework, in which the sensing matrix is optimized directly based on the mutual incoherence property rather than being learned from signal data. By focusing on the intrinsic theoretical characteristics required for efective compressive sensing, the proposed approach enhances generalization while maintaining practical suitability for eficient storage and hardware implementation.

## 5.1 Problem Formulation

The primary objective of this chapter is to construct a binary sensing matrix with low coherence, whose entries are generated through a shared underlying rule. From a classical optimization perspective, this problem involves a combinatorial search, making it computationally intractable for large-scale settings. To overcome this challenge, we employ a neural network-based framework. However, unlike conventional deep learning approaches, we avoid highly complex architectures and large-scale training datasets, as they increase computational overhead and complicate the overall design. Instead of learning from signal datasets, our approach aims to construct the sensing matrix by directly learning and optimizing its inherent mathematical property, namely coherence, which is fundamental to compressive sensing. By focusing on the sensing matrix property rather than the data itself, the proposed framework reduces storage requirements while improving its universality, robustness, and practical applicability across diverse signal domains.

Accordingly, we formulate this problem as an optimization problem dependent on the neural network parameters. Let $\scriptstyle A _ { \theta }$ denote the desired learned sensing matrix with parameters θ. The problem can then be formulated as

$$
{ \begin{array} { r l } { \underset { \theta } { \operatorname* { m i n } } } & { { \mathcal { L } } ( A _ { \theta } ) } \\ { { \mathrm { s u b j e c t ~ t o } } } & { A _ { \theta } ( i , j ) \in \left\{ { \frac { - 1 } { \sqrt { M } } } , { \frac { 1 } { \sqrt { M } } } \right\} , } \\ & { { \mathrm { C o l u m n s ~ o f ~ } } A _ { \theta } { \mathrm { ~ g e n e r a t e d ~ b y ~ s h a r e d - r u l e . } } } \end{array} }\tag{6}
$$

Here, $\mathcal { L } ( A _ { \theta } )$ denotes the loss function that promotes incoherence, which is discussed in the following section.

Our objective is to learn a binary sensing matrix $\pmb { A } _ { \theta } \in \mathbb { R } ^ { M \times N }$ such that

$$
a _ { i j } \in \left\{ \frac { - 1 } { \sqrt { M } } , \frac { 1 } { \sqrt { M } } \right\} .
$$

Instead of randomly selecting or directly optimizing the entries $a _ { i j }$ of $\scriptstyle A _ { \theta } .$ , we generate each column of $\scriptstyle A _ { \theta }$ using a neural network governed by a shared rule. Let

$$
z _ { j } \in \mathbb { R } ^ { M } , \forall j = 1 , 2 , . . . N
$$

denote the latent vectors distributed according to a predefined distribution, which are provided as inputs to the neural network. Then, a lightweight neural network

$$
\mathcal { N } _ { \theta } : \mathbb { R } ^ { M } \to \mathbb { R } ^ { M }
$$

maps each latent vector to an intermediate representation, $s _ { j }$ , i.e.,

$$
s _ { j } = \mathcal { N } _ { \theta } ( z _ { j } )
$$

from which the corresponding binary sensing matrix column is obtained by applying the sign operator and normalizing it by a factor of $\textstyle { \frac { 1 } { \sqrt { M } } }$

$$
a _ { j } = \frac { 1 } { \sqrt { M } } \mathrm { s g n } ( s _ { j } ) .
$$

Finally, the complete sensing matrix is constructed by concatenating all generated columns,

$$
\mathbf { } A _ { \theta } = [ a _ { 1 } ^ { T } , a _ { 2 } ^ { T } , \ldots , a _ { N } ^ { T } ]
$$

where each column is generated according to the shared rule, i.e.,

$$
a _ { j } = \frac { 1 } { \sqrt { M } } \mathrm { s g n } ( \mathcal { N } _ { \theta } ( z _ { j } ) ) .
$$

Here, the same neural network with shared parameters θ is used to generate all columns of the sensing matrix, while only the latent vector $z _ { j }$ varies across columns. Consequently, the neural network acts as a compact matrix generator, producing a sensing matrix that exhibits a low-coherence structure.

![](images/3ded1f63636caf45207a4a30756dcb8cba0c216a7bbfd952d94642b751a35fa0.jpg)  
Figure 1: Architecture of the Model

The sensing matrix is therefore generated column-wise via a shared neural mapping, yielding a deterministic, structured construction. This approach requires only compact storage, ensures reproducibility, and provides greater mathematical control over the matrix properties. As a result, it ofers significant advantages over conventional brute-force random matrix generation methods.

## 5.2 Model Architecture

To construct the proposed property-based learned sensing matrix using a shared generation rule, a very simple fully connected feedforward neural network was employed. The architecture of the proposed model is illustrated in Figure 1. The network consists of an input layer, two hidden layers, and an output layer. The model takes a latent vector of dimension M as the input. This input is first passed through a hidden dense layer comprising 128 neurons with the Rectified Linear Unit (ReLU) activation function. The output of this is then forwarded to another fully connected layer with 64 neurons and a linear activation function.

To enforce the binary constraint on the generated sensing matrix, the Straight-Through Estimator (STE)-based sign function is applied at the output layer. The resulting binary output is subsequently normalized by $\textstyle { \frac { 1 } { \sqrt { M } } }$ to ensure stable scaling and maintain the energy normalization of the sensing matrix. Finally, the generated vectors are transposed and stacked column-wise to construct the sensing matrix of size $6 4 \times 2 5 6$

The proposed model employs a simple neural network architecture rather than a complex

Model:"functional"
<table><tr><td rowspan=1 colspan=1>Layer (type)</td><td rowspan=1 colspan=1>Output Shape</td><td rowspan=1 colspan=1>Param #</td></tr><tr><td rowspan=1 colspan=1>input_layer (InputLayer)</td><td rowspan=1 colspan=1>(None, 64)</td><td rowspan=1 colspan=1>0</td></tr><tr><td rowspan=1 colspan=1>dense (Dense)</td><td rowspan=1 colspan=1>(None, 128)</td><td rowspan=1 colspan=1>8,320</td></tr><tr><td rowspan=1 colspan=1>dense_1 (Dense)</td><td rowspan=1 colspan=1>(None, 64)</td><td rowspan=1 colspan=1>8,256</td></tr><tr><td rowspan=1 colspan=1>lambda (Lambda)</td><td rowspan=1 colspan=1>(None, 64)</td><td rowspan=1 colspan=1>0</td></tr><tr><td rowspan=1 colspan=1>true_divide (TrueDivide)</td><td rowspan=1 colspan=1>(None, 64)</td><td rowspan=1 colspan=1>0</td></tr><tr><td rowspan=1 colspan=1>lambda_1 (Lambda)</td><td rowspan=1 colspan=1>(64, None)</td><td rowspan=1 colspan=1>0</td></tr></table>

Total params: 16,576 (64.75 KB)  
Trainable params: 16,576 (64.75 KB)  
Non-trainable params: 0 (0.00 B)

Figure 2: Summary of the Model

deep network. As a result, it requires fewer trainable parameters, which reduces both computational complexity and training cost. Despite its simplicity, the model achieves competitive performance. A summary of the network architecture is presented in Figure 2.

## 5.3 Loss Function

In this chapter, our objective is to design a binary sensing matrix by exploiting the incoherence property rather than relying on large training datasets, thereby reducing computational complexity and training overhead. As discussed in Section 2, the coherence among the columns of the sensing matrix A is closely related to the of-diagonal entries of its Gram matrix G. Since sensing matrices with low coherence provide stronger guarantees for sparse signal recovery, minimizing coherence becomes a key design objective.

In compressive sensing, the of-diagonal elements of the Gram matrix are particularly important because they quantify the pairwise correlations among the columns of the sensing matrix. Ideally, the columns should be nearly orthogonal, resulting in a Gram matrix that approximates the identity matrix, i.e., G ≈ I. Furthermore, the mutual coherence of A is determined by the maximum absolute of-diagonal entry of G. Therefore, reducing these of-diagonal values directly improves the recovery performance of sparse signals.

However, directly minimizing the maximum of-diagonal entry is challenging because the corresponding objective function is non-diferentiable and thus unsuitable for gradient-based optimization methods. To address this limitation, diferentiable surrogate loss functions are commonly employed to approximate the minimization of the worst-case pairwise correlation.

In this work, we adopt a combined loss function that simultaneously minimizes the overall coherence of the sensing matrix while penalizing the largest pairwise column correlations.

This approach promotes the generation of binary sensing matrices with low mutual coherence, thereby enhancing sparse signal recovery performance. The proposed loss function is defined as

$$
\mathcal { L } ( A _ { \theta } ) = \alpha _ { 1 } \mathcal { L } _ { p - n o r m } ( A _ { \theta } ) + \alpha _ { 2 } \mathcal { L } _ { L o g S u m } ( A _ { \theta } ) + \alpha _ { 3 } \mathcal { L } _ { t i g h t } ( A _ { \theta } )
$$

where $\alpha _ { 1 } , \alpha _ { 2 }$ and $\alpha _ { 3 }$ are weighting coeficients that control the relative contributions of the three loss components. $\mathcal { L } _ { p - n o r m } ( A _ { \theta } ) , \mathcal { L } _ { L o g S u m } ( A _ { \theta } )$ , and $\mathcal { L } _ { t i g h t } ( A _ { \theta } )$ are the $\mathcal { L } _ { p } \mathrm { - n o r m }$ loss, LogSumExp (LSE) loss, and tight-frame loss, respectively.

$L _ { p }$ norm Loss: One of the popular choice is $L _ { p }$ norm Loss.

$$
\mathcal { L } _ { p - n o r m } ( A _ { \theta } ) = \left( \sum _ { i \neq j } | G _ { i j } | ^ { p } \right) ^ { \frac { 1 } { p } }
$$

If $p = 1$ , then it is equal to the sum of elements, and if $p = 2$ , then it becomes the Frobenius norm of $G .$ . As $p  \infty$ then the above quantity approaches to maximum value of $G ,$ that is, $\begin{array} { r } { \mathcal { L } _ { p - n o r m }  \operatorname* { m a x } _ { i \neq j } | { \pmb { G } } _ { i j } | = \mu ( \pmb { A } ) } \end{array}$

• LogSumExp Loss: Instead of considering maximum absolute value of $G _ { \ast }$ , we considered smoother approximation to the maximum.

$$
\mathcal { L } _ { L o g S u m } ( \pmb { A } _ { \theta } ) = \frac { 1 } { \xi } \log \left( \sum _ { i \neq j } e ^ { \xi G _ { i j } } \right)
$$

As $\xi  \infty$ , this also converges to the maximum. It has a smooth gradient, which makes it numerically more stable.

• Tight Frame Loss: Since Equiangular tight frames (ETFs) can achieve the lowest possible bound of coherence, that is, the Welch bound, therefore, we consider tight frame loss. Since the tight frame satisfies $\begin{array} { r } { \pmb { A } \pmb { A } ^ { T } = \frac { N } { M } \mathbb { I } . } \end{array}$ , therefore we cosider the loss function

$$
\mathcal { L } _ { t i g h t } ( A _ { \theta } ) = \Vert A A ^ { T } - \frac { N } { M } \mathbb { I } \Vert _ { F } ^ { 2 }
$$

where $\| . \| _ { F }$ denotes the Frobenius norm.

By appropriately selecting the weighting coeficients, the optimization process can effectively balance reducing the average column correlation with suppressing the maximum pairwise correlation. Consequently, the proposed loss function directly targets the coherence properties of the sensing matrix, guiding the learning process toward constructing binary sensing matrices with improved incoherence and enhanced sparse recovery capability.

Algorithm 1 : Property-driven learned binary sensing matrix construction   
Input: N latent vector $z _ { j } \in \mathbb { R } ^ { M }$ sampled from some distribution, number of epochs(E),   
learning rate(l), weighting coeficients of the loss function $\alpha _ { 1 } , \alpha _ { 2 }$ , and , $\alpha _ { 3 } .$ loss function   
parameters $p$ and $\xi .$   
Output: Sensing matrix $\pmb { A } \in \mathbb { R } ^ { M \times N }$   
1. Initialization: Randomly initialize weights and biases of the network.   
2. Forward Propagation:   
a). Feed the latent vectors to the input layer with M neurons.   
b). Calculate the value of hidden layer neurons, $\mathrm { i . e . }$   
$H = A _ { 1 } ( Z W _ { 1 } + b _ { 1 } )$   
$S = A _ { 2 } ( H W _ { 2 } + b _ { 2 } )$   
where $A _ { 1 }$ and $A _ { 2 }$ are the activation functions of the hidden layers.   
c). Binary Constraint and Scale: Apply binary constraint and scale i.e.,   
$A _ { i n } = \frac { 1 } { \sqrt { M } } \mathrm { s g n } ( S )$   
d). Sensing Matrix Construction: The sensing matrix $A = A _ { i n } ^ { T }$   
3. Compute Loss: Calculate the loss by   
$\mathscr { L } ( \pmb { A } ) = \alpha _ { 1 } \mathscr { L } _ { p - n o r m } ( \pmb { A } ) + \alpha _ { 2 } \mathscr { L } _ { L o g S u m } ( \pmb { A } ) + \alpha _ { 3 } \mathscr { L } _ { t i g h t } ( \pmb { A } )$   
where $\mathcal { L } _ { p - n o r m } ( A ) , \mathcal { L } _ { L o g S u m } ( A )$ and $\mathcal { L } _ { t i g h t } ( A )$ are calculated as describe in section $5 .$   
4. Back Propagation: Compute gradient and update the network parameters using   
Adam optimizer.   
To learn the optimal parameters, repeat steps 2 to 4, E number of times.   
Return A

## 6 Simulation Result of Proposed Model

This section presents simulation results for the proposed learning-based method that learns a binary sensing matrix satisfying the incoherence property. The most important thing is to train our model; we don’t need a large amount of data. We feed our model a latent vector sample from a Gaussian distribution, and by learning from the incoherence property and following the common generation rule, it generates incoherent columns of the sensing matrix.

## 6.1 Training Data

To train the model, we don’t use a large amount of data. In fact, our model learns from the classical CS incoherence property rather than from data. Most learning-based methods use data for training; as a result, they learn data statistics, and their performance is mostly data-specific, leading to limited generalization.

In our model, instead of optimizing each entry of A, we optimize it column-wise. To generate the columns of the sensing matrix, we feed N latent vectors of dimension M sampled from a Gaussian distribution. And our model maps these latent vectors to the columns of the sensing matrix A.

## 6.2 Hyperparameter Tuning

To determine the optimal hyperparameters, extensive experiments were conducted by varying values of hyperparameters, including learning rate, the number of neurons in the hidden layer, the number of training epochs, and the weighting coeficients of the proposed loss function, namely $\alpha _ { 1 } , \alpha _ { 2 }$ , and $\alpha _ { 3 }$ . A grid search strategy was used to systematically explore diferent parameter combinations. During the search process, both the total loss $\mathcal { L } ( \cdot )$ and the maximum $\mu ( \cdot )$ mutual coherence were monitored, since the primary objective of the proposed approach is to minimize the maximum mutual coherence of the resulting sensing matrix.

Although several hyperparameter combinations yielded comparable total loss values, the final configuration was selected for its ability to simultaneously minimize both the total loss and the maximum mutual coherence. In particular, a hidden layer consisting of 128 neurons was adopted. While increasing the number of neurons to 256 resulted in a marginal reduction in total loss, it did not yield a noticeable improvement in the maximum mutual coherence. Therefore, the simpler architecture with 128 neurons was preferred due to its lower computational complexity.

The learning rate was set to 0.0001, and the weighting coeficients for the loss function were set to $\alpha _ { 1 } = 3 , \alpha _ { 2 } = 1$ , and $\alpha _ { 3 } = 0 . 5$ . Furthermore, the parameters $p = 8$ and $\xi =$ 30 were selected because larger values cause $\mathscr { L } _ { p - n o r m } ( \cdot )$ and $\mathcal { L } _ { L o g S u m } ( \cdot )$ to more closely approximate the maximum mutual coherence. Using this hyperparameter configuration, the proposed model was trained for 1000 epochs.

## 6.3 Complexity of Model

To construct the random sensing matrix of size $M \times N$ , the construction and storage complexity will be $O ( M N )$ , which may not seem much, but the structure of the random matrix makes it complex for further calculation, as the sensing matrix is the one that is used for signal acquisition as well as for reconstruction. These processes involve extensive calculations and matrix multiplication, making the system computationally intensive.

![](images/a40207f67b68cda60abe168229684460e3bf51443754f0ddbd62585d3f5281ae.jpg)  
(a) M = 64, N = 128

![](images/7f43f5b934e68e34e73ce83a2edde6ba8d8a9ef68fe3fd71eecccd224b2801c4.jpg)  
(b) M = 64, N = 256

![](images/c6e5a9a7bb3e9db0340ff427bab38bfd160ebb4be94e74155e1c537d3064d93d.jpg)  
(c) M = 128, N = 256  
Figure 3: Loss plot versus iteration for diferent sensing matrix dimensions (M, N)

Most existing learning-based matrices rely on very complex networks and large amounts of training data. Because of this, not only the construction complexity but also the storage complexity increases, as they depend on the network’s parameters, which are generally very large.

However, in our case, we used a very small fully connected neural network, and its storage complexity is almost equal to that of the random sensing matrix. After training the model, we just need to store the network parameters to construct a new sensing matrix, which provides much better incoherence guarantees than the random matrix. The matrix’s binary nature makes it much easier to handle and perform further calculations.

## 6.4 Model Performance and Comparison Result

To design a binary sensing matrix with low mutual coherence, we proposed a neural network framework using the coherence-based loss function described in Section 5. For the selected hyperparameters described in section 6.2, the model is trained for 1000 epochs across diferent sensing matrix dimensions (M, N). The convergence behavior of the proposed approach is illustrated through several coherence-related performance measures.

Figure 3 presents the convergence of the loss function over the varying iterations. A consistent decrease in loss value is observed, demonstrating the efectiveness of the optimization process and indicating that the learned sensing matrix progressively approaches the desired low-coherence structure.

Figure 4 shows the variation in the maximum mutual coherence over iterations. However, the maximum coherence does not decrease monotonically with iterations and exhibits local fluctuations. This reflects the switching of the worst-case coherence pair across iterations. Despite these fluctuations, the envelope of the curve decreases progressively, indicating that the overall coherence structure of the proposed sensing matrix is improving.

In contrast, average and total mutual coherence are less sensitive and exhibit smoother, strictly decreasing trends as shown in Figures 5 and 6, respectively. This further validates the capability of the proposed learning strategy to reduce overall column correlations.

To provide a visual interpretation of the learned matrix structure, Figure 7 compares the heatmaps of the Gram matrices corresponding to the initial sensing matrix and the learned sensing matrix for diferent values of (M, N). The first row shows the initial Gram matrices, while the second row presents the Gram matrices obtained after optimization. The learned matrices exhibit significantly weaker of-diagonal entries, indicating reduced inter-column correlations and improved incoherence properties.

![](images/ca98f6e7fc322f21df42efe5157a5c71c7ff5010343ff9698d46fc49e59d0b18.jpg)  
(a) M = 64, N = 128

![](images/3bb8b640743a0aa1f6fa91572406113c03c9a22f33ffea1692eb538e9541bbb7.jpg)  
(b) M = 64, N = 256

![](images/5cab577f93ff616afa9dfbc8065210688d88fe6a57496604185f101022a98567.jpg)  
(c) M = 128, N = 256

Figure 4: Maximum mutual coherence versus iteration for diferent sensing matrix dimensions (M, N)  
![](images/010282a4ee9a8704f9b3e9a6c02259a788fe231ef209863874e797933a44d9e8.jpg)  
(a) M = 64, N = 128

![](images/4bccaaa8245904a0eb5a59c8de6f5c87b7f789532f148e4db1947678f471d7c9.jpg)  
(b) M = 64, N = 256

![](images/65197209ee172677f3699ce9c5b10165bf233a0d8f536059e23955d4e24bd7c1.jpg)  
(c) M = 128, N = 256

Figure 5: Average mutual coherence versus iteration for diferent sensing matrix dimensions (M, N)  
![](images/0f53147c66d93a1215fbd059acd61236a827b13ff19600ce789bf28edc9248eb.jpg)  
(a) M = 64, N = 128

![](images/0f8daa35d37ca934c0c09356220f864bba67cc20400d73f4e212b1f23fb47398.jpg)  
(b) M = 64, N = 256

![](images/e3043b3700605192c15385d21d5e68ee6e79992500299c4bd1589a8893bb8e72.jpg)  
(c) M = 128, N = 256  
Figure 6: Total mutual coherence versus iteration for diferent sensing matrix dimensions (M, N)

Furthermore, we compare the proposed method with two commonly used random sensing matrices: (i) a Gaussian random matrix whose entries are independently drawn from a normal distribution with mean zero and variance one, and (ii) a Bernoulli random matrix whose entries take values $\pm \frac { 1 } { \sqrt { M } }$ with equal probability. The coherence characteristics of these matrices are compared with those of the proposed learned sensing matrix.

![](images/b7c1a4109a358bb73f81e83ce8d5aa71995ebe465b7b5d5dd0cf227227940142.jpg)

![](images/fe26a2c0b9cc953312b23909d302d61f3bce78efd4109c881e9b96aeb582141a.jpg)

![](images/37458f6dc43d2764c4acc5b2b6cea7b4c44ca593cea204f1b790b9817a0368da.jpg)

![](images/fe29c3ed7921fea1937fac4275f21b643164a89e18892a18ddf64f844055ef56.jpg)  
(a) M = 64, N = 128

![](images/6b88cf84566a2ed6a0696a4bf4a43e177430e2ff286f71f99116ac107f9ee461.jpg)  
(b) M = 64, N = 256

![](images/9477c7064972bb49cc53c044c7023219fb7ffa37fe32feb211bdd268e29cb2c2.jpg)  
(c) M = 128, N = 256  
Figure 7: Comparison of the heatmap of the Gram matrix entries corresponding to the initial sensing matrix and the proposed learned sensing matrix for diferent sensing matrix dimensions (M, N).

Table 1 presents a comparison of the maximum, average, and total mutual coherence values achieved by the proposed learned sensing matrix and randomly generated sensing matrices for diferent matrix dimensions (M, N). For the proposed approach, the reported values correspond to the averages obtained from 20 independent runs of the learning algorithm. For the random matrices, 20 were generated for each M and N case, and the maximum, average, and total coherence were computed for each matrix; the values reported in the table represent the corresponding mean value. As shown, the proposed learned sensing matrix consistently yields lower coherence measures across all considered matrix dimensions. These results demonstrate the efectiveness of the proposed learning framework in constructing highly incoherent sensing matrices, which are desirable for improved compressed sensing performance.

Figure 8 presents the histograms of the of-diagonal entries of the corresponding Gram matrices for diferent values of (M, N). The histogram associated with the proposed learned sensing matrix is more concentrated around zero and exhibits considerably shorter tails than those of the random matrices. This behavior indicates that the correlations between distinct columns are substantially reduced. In particular, the largest-magnitude of-diagonal entries are significantly smaller for the proposed method, which directly translates into lower

Table 1: Comparison of mutual coherence and related performance measures of the proposed learned sensing matrix and random sensing matrices (Gaussian and Bernoulli matrices) for diferent values of M and N.
<table><tr><td colspan="4">M = 64 and N = 128</td></tr><tr><td>Matrix</td><td>Maximum Mutual Coherence  $( \mu ( A ) )$ </td><td>Average Mutual Coherence  $\left( \mu _ { a v g } ( A ) \right)$ </td><td>Total Mutual Coherence  $( \mu _ { t o t a l } ( A ) )$ </td></tr><tr><td rowspan="3">Random Gaussian Random Bernoulli Proposed</td><td> $\approx 0 . 4 8 1$ </td><td> $\approx 0 . 0 9 9$ </td><td> $\approx 2 5 6 . 2 3$ </td></tr><tr><td> $\approx 0 . 4 9 6$ </td><td> $\approx 0 . 0 9 8$ </td><td> $\approx 2 5 3 . 3 4$ </td></tr><tr><td> $\approx \mathbf { 0 . 2 8 1 }$   $M = 6 4$ </td><td> $\approx \mathbf { 0 . 0 9 1 }$   $N = 2 5 6$ </td><td> $\approx 2 0 0 . 5 8$ </td></tr><tr><td colspan="4">and</td></tr><tr><td>Random Gaussian</td><td> $\approx 0 . 5 0 1$ </td><td> $\approx 0 . 0 9 9$ </td><td>≈1022.83</td></tr><tr><td rowspan="2">Random Bernoulli Proposed</td><td> $\approx 0 . 5 2 8$ </td><td> $\approx 0 . 0 9 8$ </td><td> $\approx 1 0 1 9 . 5 8$ </td></tr><tr><td> $\approx \mathbf { 0 . 3 4 3 }$ </td><td> $\approx \mathbf { 0 . 0 9 2 }$ </td><td> $\approx 8 5 4 . 3 7$ </td></tr><tr><td colspan="4"> $\overline { { M = 1 2 8 } }$  and  $\overline { { N = 2 5 6 } }$ </td></tr><tr><td rowspan="2">Random Gaussian Random Bernoulli Proposed</td><td> $\approx 0 . 3 6 8$ </td><td> $\approx 0 . 0 7 0$ </td><td> $\approx 5 1 0 . 5 2$ </td></tr><tr><td> $\approx 0 . 3 6 8$   $\approx \mathbf { 0 . 2 5 0 }$ </td><td> $\approx 0 . 0 7 0$   $\approx \mathbf { 0 . 0 6 8 }$ </td><td> $\approx 5 0 9 . 2 7$   $\approx 4 4 1 . 9 \ 7$ </td></tr></table>

![](images/5852ed9e0a7daf7a19c6a789f8aaaf3e5f725052b2acba7028ba5d82fcffce8c.jpg)

![](images/5d2bf31fc031fc0dd8a4aa9fc26bd272f253f5e854d23678b2e43fd883de3c2e.jpg)

![](images/41f66dd54d4616957a5a447ebf1ec6be607cfe382405f698097019fe4b55544e.jpg)  
(a) M = 64, N = 128  
(b) M = 64, N = 256  
(c) M = 128, N = 256  
Figure 8: Comparison of the coeficient distributions of the of-diagonal Gram entries of the proposed learned sensing matrix and existing random sensing matrices (Gaussian and Bernoulli) through histogram analysis for diferent sensing matrix dimensions (M, N).

maximum mutual coherence.

Overall, the experimental results, including the convergence analysis, Gram matrix visualization, histogram comparisons, and quantitative evaluations reported in Table 1, demonstrate that the proposed coherence-driven learning framework efectively constructs sensing matrices with substantially reduced maximum, average, and total mutual coherence. These findings confirm the superiority of the proposed method over conventional random-sensingmatrix constructions across a wide range of matrix dimensions.

## 7 Conclusion

In compressive sensing, the sensing matrix plays a crucial role in acquiring and accurately reconstructing signals from a limited number of measurements. Over the years, several methods for constructing sensing matrices have been proposed in the literature. However, many of these methods sufer from limitations such as high computational complexity, complicated construction procedures, or failure to satisfy the essential properties required for efective compressive sensing.

To address these challenges, this chapter proposed a learning-based approach for constructing a binary sensing matrix using an artificial neural network (ANN). Unlike conventional data-driven methods, the proposed network learns directly from the mutual incoherence property, a key characteristic of an efective sensing matrix.

A simple neural network with only a few layers was designed to optimize the sensing matrix. The objective was to minimize the matrix’s mutual coherence. However, since the maximum mutual coherence is not diferentiable, it cannot be used directly as a loss function. Therefore, diferentiable surrogate functions were employed, and a combined loss function was designed to reduce the overall correlation among the matrix columns while encouraging low maximum coherence.

The efectiveness of the proposed method was evaluated by comparing its coherence measures with those of randomly generated sensing matrices. In addition, the corresponding Gram matrices were analyzed to assess the correlation structure. The experimental results and visual comparisons demonstrate that the proposed approach produces sensing matrices with favorable coherence properties and achieves competitive performance.

## References

[1] Amir Adler, Michael Elad, and Michael Zibulevsky. Compressed learning: A deep neural network approach. arXiv preprint arXiv:1610.09615, 2016.

[2] Pablo Arbelaez, Michael Maire, Charless Fowlkes, and Jitendra Malik. Contour detection and hierarchical image segmentation. IEEE transactions on pattern analysis and machine intelligence, 33(5):898–916, 2010.

[3] Richard Baraniuk, Mark Davenport, Ronald DeVore, and Michael Wakin. A simple proof of the restricted isometry property for random matrices. Constructive approximation, 28:253–263, 2008.

[4] Amir Beck and Marc Teboulle. A fast iterative shrinkage-thresholding algorithm for linear inverse problems. SIAM journal on imaging sciences, 2(1):183–202, 2009.

[5] Ashish Bora, Ajil Jalal, Eric Price, and Alexandros G Dimakis. Compressed sensing using generative models. In International conference on machine learning, pages 537– 546. PMLR, 2017.

[6] Jean Bourgain, Stephen Dilworth, Kevin Ford, Sergei Konyagin, and Denka Kutzarova. Explicit constructions of rip matrices and related problems. Duke Mathematical Journal, 2011.

[7] Emmanuel Candes, Nathaniel Braun, and Michael Wakin. Sparse signal and image recovery from compressive samples. In 2007 4th IEEE International Symposium on Biomedical Imaging: From Nano to Macro, pages 976–979. IEEE, 2007.

[8] Emmanuel Candes and Justin Romberg. Sparsity and incoherence in compressive sampling. Inverse problems, 23(3):969–985, 2007.

[9] Emmanuel J Candes. The restricted isometry property and its implications for compressed sensing. Comptes rendus mathematique, 346(9-10):589–592, 2008.

[10] Emmanuel J Cand\`es, Justin Romberg, and Terence Tao. Robust uncertainty principles: Exact signal reconstruction from highly incomplete frequency information. IEEE Transactions on information theory, 52(2):489–509, 2006.

[11] Emmanuel J Candes, Justin K Romberg, and Terence Tao. Stable signal recovery from incomplete and inaccurate measurements. Communications on Pure and Applied Mathematics: A Journal Issued by the Courant Institute of Mathematical Sciences, 59(8):1207–1223, 2006.

[12] Emmanuel J Candes and Terence Tao. Decoding by linear programming. IEEE transactions on information theory, 51(12):4203–4215, 2005.

[13] Emmanuel J Candes and Terence Tao. Near-optimal signal recovery from random projections: Universal encoding strategies? IEEE transactions on information theory, 52(12):5406–5425, 2006.

[14] Albert Cohen, Wolfgang Dahmen, and Ronald DeVore. Compressed sensing and best k-term approximation. Journal of the American mathematical society, 22(1):211–231, 2009.

[15] Mark A Davenport, Marco F Duarte, Yonina C Eldar, and Gitta Kutyniok. Introduction to compressed sensing., 2012.

[16] Ronald A DeVore. Deterministic constructions of compressed sensing matrices. Journal of complexity, 23(4-6):918–925, 2007.

[17] David L Donoho. Compressed sensing. IEEE Transactions on information theory, 52(4):1289–1306, 2006.

[18] David L Donoho and Michael Elad. Optimally sparse representation in general (nonorthogonal) dictionaries via $\ell _ { 1 }$ minimization. Proceedings of the National Academy of Sciences, 100(5):2197–2202, 2003.

[19] Julio Martin Duarte-Carvajalino and Guillermo Sapiro. Learning to sense sparse signals: Simultaneous sensing matrix and sparsifying dictionary optimization. IEEE Transactions on Image Processing, 18(7):1395–1408, 2009.

[20] Michael Elad. Optimized projections for compressed sensing. IEEE Transactions on Signal Processing, 55(12):5695–5702, 2007.

[21] Alex Krizhevsky, Ilya Sutskever, and Geofrey E Hinton. Imagenet classification with deep convolutional neural networks. Advances in neural information processing systems, 25, 2012.

[22] Yann LeCun, L´eon Bottou, Yoshua Bengio, and Patrick Hafner. Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11):2278–2324, 1998.

[23] Ziwei Liu, Ping Luo, Xiaogang Wang, and Xiaoou Tang. Deep learning face attributes in the wild. In Proceedings of the IEEE international conference on computer vision, pages 3730–3738, 2015.

[24] Suhas Lohit, Kuldeep Kulkarni, Ronan Kerviche, Pavan Turaga, and Amit Ashok. Convolutional neural networks for noniterative reconstruction of compressively sensed images. IEEE Transactions on Computational Imaging, 4(3):326–340, 2018.

[25] Vishal Monga, Yuelong Li, and Yonina C Eldar. Algorithm unrolling: Interpretable, efficient deep learning for signal and image processing. IEEE Signal Processing Magazine, 38(2):18–44, 2021.

[26] Ali Mousavi, Ankit B Patel, and Richard G Baraniuk. A deep learning approach to structured signal recovery. In 2015 53rd annual allerton conference on communication, control, and computing (Allerton), pages 1336–1343. IEEE, 2015.

[27] Andrianiaina Ravelomanantsoa, Hassan Rabah, and Amar Rouane. Compressed sensing: A simple deterministic measurement matrix and a fast recovery algorithm. IEEE Transactions on Instrumentation and Measurement, 64(12):3405–3413, 2015.

[28] Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, et al. Imagenet large scale visual recognition challenge. International journal of computer vision, 115(3):211–252, 2015.

[29] Wuzhen Shi, Feng Jiang, Shaohui Liu, and Debin Zhao. Image compressed sensing using convolutional neural network. IEEE Transactions on Image Processing, 29:375– 388, 2019.

[30] M´aty´as A Sustik, Joel A Tropp, Inderjit S Dhillon, and Robert W Heath Jr. On the existence of equiangular tight frames. Linear Algebra and its applications, 426(2-3):619– 635, 2007.

[31] Andreas M Tillmann and Marc E Pfetsch. The computational complexity of the restricted isometry property, the nullspace property, and related concepts in compressed sensing. IEEE Transactions on Information Theory, 60(2):1248–1259, 2013.

[32] Yue Wang, Linlin Xue, Yuqian Yan, and Zhongpeng Wang. A novel complex-valued gaussian measurement matrix for image compressed sensing. Entropy, 25(9):1248, 2023.

[33] Shanshan Wu, Alex Dimakis, Sujay Sanghavi, Felix Yu, Daniel Holtmann-Rice, Dmitry Storcheus, Afshin Rostamizadeh, and Sanjiv Kumar. Learning a compressed sensing measurement matrix via gradient unrolling. In International Conference on Machine Learning, pages 6828–6839. PMLR, 2019.

[34] Jianping Xu, Yiming Pi, and Zongjie Cao. Optimized projection matrix for compressive sensing. EURASIP Journal on Advances in Signal Processing, 2010:1–8, 2010.

[35] Renjie Yi, Chen Cui, Biao Wu, and Yang Gong. A new method of measurement matrix optimization for compressed sensing based on alternating minimization. Mathematics, 9(4):329, 2021.

[36] Jian Zhang and Bernard Ghanem. Ista-net: Interpretable optimization-inspired deep network for image compressive sensing. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 1828–1837, 2018.