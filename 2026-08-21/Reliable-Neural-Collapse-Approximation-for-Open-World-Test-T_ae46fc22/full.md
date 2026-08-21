# Reliable Neural Collapse Approximation for Open-World Test-Time Adaptation

Jia-Qi Lin, Yuangang Pan, Chang-Dong Wang, Haizhang Zhang, Ivor W. Tsang, and Joey Tianyi Zhou

## Abstract

Test-Time Adaptation (TTA) methods aim to bridge the domain gap between the source and target domains. However, traditional TTA methods become inefective when the label distribution shift occurs, a challenge commonly referred to as an open-world scenario. In this paper, we introduce a new method named Reliable Neural Collapse approximation (ReNC) for Open-World Test-Time Adaptation (OWTTA). Specifically, we leverage neural collapse as a structural prior for reliable target-domain adaptation. Guided by this prior, we justify that the pre-trained classifier weights can serve as the prototypes of the source domain. By measuring the similarity between samples and prototypes, we filter out the Out-Of-Distribution (OOD) samples for reliable updates. Furthermore, we propose a neural collapse approximation mechanism to refine these prototypes, ensuring they can gradually adapt to the target domain while maintaining the neural collapse structure. Extensive experiments on several open-world benchmarks demonstrate the superiority of the proposed method. Our empirical analysis suggests that ReNC better preserves NC-related properties in the target domain, providing useful evi dence for explaining reliable OWTTA and ofering new insights for model design. Code is available at https://github.com/JiaqiLin-AI/ReNC.

## 1 Introduction

Test-Time Adaptation (TTA) has attracted significant attention [1, 2, 3] for its ability to improve the generalization of deep learning models without requiring access to the entire batch of data. It aims to bridge the distribution gap between training and testing data under conditions of data distribution shift, where the training and testing data originate from diferent domains. Existing TTA methods [4, 5, 6, 7] address domain adaptation by either aligning target features with the source distribution or minimizing the entropy of model outputs. Both approaches have shown impressive efectiveness in enhancing model performance under data distribution shifts.

A basic assumption of TTA is that the training and testing samples share the same label space. However, this assumption may be violated in more realistic open-world scenarios [8]. Under the open-world scenarios, the test data may not only contain seen classes, known as In-Distribution (ID) data, but also unseen classes, referred to as Out-Of-Distribution (OOD) data. This phenomenon, which we refer to as a label distribution shift, can render existing TTA methods inefective. For example, in medical imaging, data distribution shift and label distribution shift often occur due to diferences in acquisition devices and new disease that the model has not encountered during training [9]. Adapting models to simultaneous data distribution shift and label distribution shift is the known as Open-World Test-Time Adaptation (OWTTA), which remains a challenge for the community. There are two primary challenges in OWTTA: 1) The label distribution shift caused by OOD samples renders the model update process unreliable. 2) The domain gap introduced by the data distribution shift further increases the dificulty of efective adaptation.

![](images/15a878675c854859a42a5d30de3ebe6bcba39a109b870dfb7ef5716498841ea2.jpg)  
Figure 1: Motivation of ReNC. (a) Existing methods use fixed source-domain prototypes, which may mis match the target domain. (b) ReNC improves feature-prototype consistency and adapts prototypes toward target-domain neural collapse.

Most recently, a limited number of studies [8, 10, 11] have explored OWTTA as an efective strategy to enhance the generalization abilities of deep neural networks. They typically employ a two-step process [10, 11] to address the aforementioned challenges. First, the OOD samples are filtered out from the target data to mitigate the negative efects of label distribution shift. This is typically achieved by measuring the similarity between features and their nearest prototypes to distinguish between ID and OOD samples. Next, TTA is applied to the remaining ID samples to address the distribution gap caused by data distribution shift. However, these methods optimize the model using fixed prototypes, which are typically derived either from the source domain or the pre-trained classifier. In practice, source prototypes are often unavailable due to computational constraints [12, 13], privacy concerns [14, 15], or copyright restrictions [16]. Moreover, even when accessible, prototypes from the source domain often fail to align well with the target domain due to the data distribution shift. As illustrated in Figure 1a, existing method [8] push features toward fixed prototypes of the source domain leading to suboptimal performance due to the domain gap with the target domain.

In this paper, we introduce a new OWTTA method, Reliable Neural Collapse Approximation (ReNC), which addresses the aforementioned challenges by approximating the neural collapse structure in the target domain. Neural collapse [17] describes a terminal geometry of well-trained classification neural networks, where features become intra-class compact, class means align with classifier weights, and classification behaves like nearest-class-center prediction. Motivated by this property, we leverage neural collapse as a reliable structural prior to guide OWTTA. However, as neural collapse typically occurs only with labeled source data, it cannot be directly achieved on unlabeled target data. Therefore, our ReNC method is proposed to reliably approximate the neural collapse state in the target domain. As illustrated in Figure 1b, ReNC not only pushes target features toward their corresponding prototypes, but also updates the prototypes to approximate the target-domain neural collapse state. Specifically, considering that the pre-trained source model has already collapsed, we introduce a parameter-free mechanism to filter out OOD samples for reliable updates. This mechanism evaluates the similarity of the target domain features to prototypes, ensuring the adaptation process focuses on ID samples. Furthermore, we formulate neural collapse approximation as a progressive adaptation process that encourages reliable target features to approach their corresponding prototypes, while updating the prototypes with prior-regularized adaptation to preserve the underlying neural collapse geometry. Benefiting from reliable and eficient updates, ReNC achieves a better neural collapse state in the target domain without access to the entire target data batch. We summarize the contributions as follows:

• We are the first study to introduce the concept of neural collapse to OWTTA. And we justify that classifier weights are equivalent to source domain prototypes from the neural collapse perspective.

• We introduce a new OWTTA framework, called ReNC, featuring an eficient updating mechanism to approximate the neural collapse state in the target domain, which provides a reliable structural prior for TTA.

• Extensive experiments on several open-world benchmarks demonstrate the efectiveness of ReNC in OWTTA, and further analyses show that ReNC better preserves NC-related representation properties during target adaptation.

The rest of this paper is organized as follows: In Section 2, we define the problem definition of OWTTA and provide a brief review of the related literature. Section 3 introduces our ReNC method, progressing from the basic concepts to the final approach. In Section 4, we present a thorough evaluation based on extensive experiments to validate the efectiveness of the ReNC approach. Finally, Section 5 provides a summary of the key findings and concludes the paper.

## 2 Problem Definition and Literature Review

In this section, we first introduce the setting of closed-world test-time adaptation. Then, we extend the problem to the open-world scenario, where data contains label distribution shifts. Lastly, we compare this setting with OOD detection and universal domain adaptation, discussing the diferences between them and OWTTA.

## 2.1 Test-Time Adaptation

Data Distribution Shift: We consider the classification tasks where X and Y represent the data space and the label space, respectively. There are two distinct distributions over $\mathcal { X } \times \mathcal { V }$ , referred to as the source domain $\mathcal { D } _ { S }$ and the target domain $\mathcal { D } _ { T }$ . The data distribution shift occurs because the source data set $s _ { S }$ and the target data set $\boldsymbol { S _ { T } }$ are drawn independently from two diferent domains, $\mathcal { D } _ { S }$ and $\mathcal { D } _ { T }$ , respectively, resulting in diferences in their distributions:

$$
\begin{array} { r } { \mathcal { S } _ { S } = \left\{ \left( \mathbf { x } _ { i } , y _ { i } \right) \middle | \mathbf { x } _ { i } \in \mathcal { D } _ { S } \right\} , \quad \mathcal { S } _ { T } = \left\{ \mathbf { x } _ { i } \middle | \mathbf { x } _ { i } \in \mathcal { D } _ { T } \right\} . } \end{array}
$$

Various domain adaptation methods [18, 19] have been proposed to eliminate the domain gap between $\mathcal { D } _ { S }$ and $\mathcal { D } _ { T }$ . Among all these methods, Test-Time Adaptation (TTA) [20, 21] has gained significant attention for its assumption that the model cannot access source data and target labels, making it more suitable for real-world applications.

A number of TTA methods are inspired by the self-training and employ various techniques for unlabeled target data adaptation [7, 22]. Test-time entropy minimization (TENT) [23] is proposed to update only the parameters in the batch normalization layer by minimizing the entropy of logits. Source Hypothesis

Transfer (SHOT) [12] freezes the classifier of the source model and updates the feature extraction module by minimizing entropy while maintaining balanced distribution in the target domain.

Besides, the distribution alignment techniques are also utilized in TTA by treating the source domain distribution as an anchor for aligning the features of the target domain [24, 25, 26]. Test-time training method [27] modified pre-training procedure to acquire the statistic information of source domain without accessing source data when adaptation. Nevertheless, such a scheme is still impractical since it alternates the pre-training procedure. To tackle this issue, [28] improves TTA performance by saving the distribution of the source domain and aligning the high-confidence target features towards the distribution for updating without altering pre-training.

Additionally, self-supervised learning techniques are also widely used in TTA since they do not need to consider the semantic information of the target data [29]. For example, contrastive learning is introduced [30] to refine the online pseudo labeling process, leveraging the self-supervised techniques to enhance the target feature adaptation. The self-supervised training branch is introduced [31] to fine-tune the parameters of the feature encoder during adaptation.

While most test-time adaptation (TTA) methods have been developed under the assumption of static target distributions and focus primarily on visual data, recent works have begun exploring more complex and realistic settings. Some studies consider scenarios where the target domain distribution changes continuously over time [32, 33, 34, 35], and [36] further extend this to evolving environments where the proportions of existing classes vary dynamically, potentially leading to class imbalance. In parallel, [37] investigates TTA in the context of tabular data, and proposes a method that estimates the test-time label distribution using confident predictions and calibrates model outputs accordingly. Furthermore, Vision-Language Models (VLMs) [38] have also been explored for test-time adaptation through prompt tuning, where prompts are optimized without labels to improve both accuracy and calibration during inference [39].

## 2.2 Open-World Test-Time Adaptation

Label distribution shift: Existing TTA methods have achieved significant improvement, assuming that the source and target domain have the same label space. In the open-world scenario, however, this assumption may be violated [40], with the distribution shift existing in both the data space X and the label space Y, simultaneously.

This shift is characterized by the divergence between the source label space and the target label space. The label set of the target domain $\mathcal { V } _ { T }$ contains not only the labels from the source domain $\mathcal { { V } } _ { S }$ , but also additional labels from new classes, denoted as $y _ { O } { \mathrm { : } }$

$$
\mathcal { Y } _ { T } = \{ \mathcal { y } _ { S } \cup \mathcal { y } _ { O } \} , \quad \mathcal { y } _ { S } \cap \mathcal { y } _ { O } = \emptyset .
$$

Based on the label of samples, the target data set $\boldsymbol { S _ { T } }$ can be further partitioned into the In-Distribution (ID) sample set $\boldsymbol { \mathcal { S } } _ { T _ { I } }$ and the Out-Of-Distribution (OOD) sample set $\mathcal { D } _ { T _ { O } }$ :

$$
\begin{array} { r l } & { { \cal { S } } _ { { \cal { T } } _ { I } } = \left\{ { \bf { x } } _ { i } | { \bf { x } } _ { i } \in { \cal { D } } _ { { \cal { T } } _ { I } } , \Psi ( { \bf { x } } _ { i } ) \in \mathcal { V } _ { S } \right\} , } \\ & { { \cal { S } } _ { { \cal { T } } _ { O } } = \left\{ { \bf { x } } _ { i } | { \bf { x } } _ { i } \in { \cal { D } } _ { { \cal { T } } _ { O } } , \Psi ( { \bf { x } } _ { i } ) \in \mathcal { V } _ { O } \right\} , } \end{array}
$$

where $\mathcal { D } _ { T _ { I } }$ represents the domain of ID samples, $\mathcal { D } _ { T _ { O } }$ represents the domain of OOD samples, and Ψ is the ground truth mapping. OWTTA requires a robust method to ensure that the adaptation process focuses on the ID samples while minimizing the adverse impact of the OOD samples.

Few TTA methods have taken the open-world scenario into consideration. Open-World Test-Time Training (OWT3<sup>2</sup>) [8] distinguishes OOD samples based on the distance between target data and prototypes from the source domain. It then expands the prototype pool and aligns the ID samples with statistical information from the source domain for reliable adaptation. OSTTA [10] measures sample confidence based on the probability values of logits. It leverages the wisdom of crowds to select high-confidence samples and applies entropy minimization exclusively to these selected samples. The Unified Entropy Optimization (UniEnt) [11] method introduces two Gaussian Mixture Models (GMMs) to model ID and OOD samples based on their distance to the nearest prototypes. UniEnt then minimizes the entropy of ID samples while maximizing the entropy of OOD samples to enhance the TTA performance.

## 2.3 Related Branches

Out-of-Distribution Detection. OOD detection aims to identify samples outside the training distribution, which is important for reliable model deployment. Existing methods usually design post-hoc scoring functions based on logits [41, 42, 43], feature representations [44, 45, 46], or gradient information [47, 48]. Some studies further improve OOD detection by re-training classifiers with auxiliary OOD datasets [49, 50] or generated synthetic OOD samples [51, 52, 53]. OOD detection methods require access to entire batches of data at once and do not account for data distribution shifts. In contrast, OWTTA must adapt and make inferences instantly with new incoming batches, addressing both data and label distribution shifts simultaneously.

Universal Domain Adaptation. Similar to TTA, existing domain adaptation methods focus on data distribution shifts while neglecting label distribution shifts in the target domain. To provide more flexibility in real-world scenarios, universal (or open-set) domain adaptation methods have been proposed [3, 54, 55, 56]. These methods require access to both source and target data, making them impractical in real-world scenarios where source data is inaccessible due to privacy or security concerns. To address this issue, source-free universal domain adaptation methods [57, 16, 58, 59] have been developed, which only require target data for adaptation. Unlike universal domain adaptation, which requires access to entire batches of data, OWTTA handles a more challenging scenario. In OWTTA, while all test data can be processed at once, inference and adaptation must be performed instantly with new incoming batches.

Test-Time Augmentation. Test-time augmentation improves prediction robustness by generating multiple augmented views of each test sample and aggregating their predictions [60, 61, 62]. These methods mainly exploit input-level diversity to obtain more stable predictions. Test-time augmentation improves generalization under domain shifts by leveraging input-level diversity, whereas OWTTA focusing on adapting the model to a specific target domain in the presence of OOD interference.

## 3 Methodology

In this section, we propose Reliable Neural Collapse approximation (ReNC) for open-world test-time adaptation. The key motivation of ReNC is to exploit neural collapse as a structural prior for reliable target domain adaptation. Specifically, we theoretically justify that, when a well-trained source model exhibits an NC-like geometry, its classifier weights can be interpreted as class prototypes. Based on this prior, ReNC first identifies reliable target samples according to their distances to the nearest prototypes, thereby filtering out OOD samples for reliable updates. Then, it progressively refines the source-domain prototypes with reliable target features and updates the model parameters under a geometry-preserving constraint in an alternating manner, encouraging the target representation to evolve toward an NC-like structure. In this way, ReNC efectively adapts the model to the target domain while maintaining a compact, balanced, and well-structured representation space.

We start by developing a simple, reliable update rule and then progressively refine our approach to handle more practical scenarios in an open-world setting.

## 3.1 Reliable Update by Filtering Out OOD Samples

Given a model pre-trained on the source domain $F \left( \pmb \theta , \cdot \right)$ , its logits $\mathbf { p } _ { i } ^ { t }$ can be represented by:

$$
\mathbf { p } _ { i } ^ { t } = F \left( \pmb { \theta } , \mathbf { x } _ { i } ^ { t } \right) = \sigma \left( \mathbf { w } f \left( \pmb { \theta } , \mathbf { x } _ { i } ^ { t } \right) + \mathbf { b } \right) = \sigma \left( \mathbf { w } \mathbf { z } _ { i } ^ { t } + \mathbf { b } \right) ,
$$

![](images/e8bfb93be0ae300a55a7ea2dd61304bb71d3f46666032b09b6dfdbad235d3a65.jpg)  
(a) CIFAR10-C & Noise

![](images/fdb4a93db7c85f75697e4eef2833d9fe79cdee3171817b60deedc1c54a274e11.jpg)  
(b) CIFAR10-C & SVHN

Figure 2: Entropy of CIFAR10-C with OOD samples ((a) Gaussian noise and (b) SVHN) pre-trained on CIFAR10. Minimizing entropy facilitates adaptation to the target domain. However, indiscriminately min imizing all entropy in an open-world scenario would degrade the model performance.  
![](images/86884123ccad6fda1d3f6c9e98e0d9037ba1094fed00af294fd3771ce9c84919.jpg)  
(a) CIFAR10-C&Noise

![](images/186408c28bc772ec44ae32558124018943c4eef018ef1c87b5a6ac415f775687.jpg)  
(b) CIFAR10-C&SVHN  
Figure 3: t-SNE visualization of extracted features on CIFAR10-C with OOD samples ((a) Gaussian noise and (b) SVHN) pre-trained on CIFAR10. Prototypes are generated using the feature class-means of source samples.

where $\mathbf { x } _ { i } ^ { t }$ is the input at the t-th batch, $f ( \pmb \theta , \cdot )$ is the encoder, $\mathbf { z } _ { i } ^ { t }$ is the feature, $\sigma ( \cdot )$ is the activation function, and w and b denote the classifier weights and bias, respectively.

Existing closed-world TTA methods assume that, despite the presence of data distribution shifts, the entropy values [63] for target domain samples corresponding to each class remain relatively low during inferring. Based on this assumption, a straightforward way to adapt $F ( \pmb \theta , \cdot )$ to the target domain $\mathcal { D } _ { T }$ is to minimize the entropy of each logit:

$$
\mathcal { L } _ { e } \left( \pmb { \theta } \right) = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } H \left( \mathbf { p } _ { i } ^ { t } \right) ,\tag{1}
$$

where $H ( \cdot )$ is the entropy function.

However, in the open-world scenario, label distribution shift exists in test samples. As shown in Figure 2, the entropy of ID and OOD samples are mixed together, complicating the process of minimizing entropy. Since the goal of minimizing entropy is to encourage each logit to form a sharp (one-hot) distribution, simply reducing the entropy of all test samples can disrupt the adaptation process.

To enhance the reliability of the update process, we investigate a selective updating strategy to split the ID and OOD samples. As shown in Figure 3, ID and OOD samples exhibit distinct feature distributions, where ID samples tend to cluster around their corresponding class prototypes, computed by averaging the samples in each class. Inspired by few-shot learning [64], we aim to transform the problem of distinguishing between ID and OOD samples into a two-cluster clustering problem. To achieve this, we construct a similarity-based measure using multiple prototypes to diferentiate between ID and OOD samples.

![](images/a3276e66e95224b55dd7b579afa8289c7ce251fd403d5d55ebedacd1d843da6a.jpg)  
(a) CIFAR10-C & Noise

![](images/c516825f405d294be112579c10dda42b9bf4170b47bf7acde27509268c033026.jpg)  
(b) CIFAR10-C & SVHN  
Figure 4: Similarity scores of CIFAR10-C with OOD samples ((a) Gaussian noise and (b) SVHN) pre-trained on CIFAR10. The diferences in the similarity scores between ID and OOD samples provide a basis for their discrimination.

Specifically, given K prototypes from source domain as $\mathbf { C } = \{ \mathbf { c } _ { j } \} _ { j = 1 } ^ { K }$ , where $\mathbf { c } _ { j }$ is calculated by the mean value of the features from j-th class, the similarity score between the test sample and its nearest prototype is:

$$
s _ { i } ^ { t } = 1 - \operatorname* { m a x } _ { \mathbf { c } _ { j } \in \mathbf { C } } \frac { \mathbf { z } _ { i } ^ { t ^ { \top } } \mathbf { c } _ { j } } { \| \mathbf { z } _ { i } ^ { t } \| \| \mathbf { c } _ { j } \| } .\tag{2}
$$

Some existing method [11] utilize a symmetric Gaussian Mixture Models (GMMs) with two components to fit ID and OOD samples based on their similarity scores. However, it is inappropriate because each component does not adhere to a symmetric Gaussian distribution, as shown in Figure 4. Instead, we approach this problem by learning an adaptive threshold $\tau _ { t }$ to partition ID and OOD samples, efectively treating it as a clustering task:

$$
\begin{array} { l } { \displaystyle \operatorname* { m i n } _ { \tau _ { t } } \frac { 1 } { N ^ { + } } \sum _ { i } \Big [ s _ { i } ^ { t } - \frac { 1 } { N ^ { + } } \sum _ { j } \mathbf { 1 } ( s _ { j } ^ { t } \leq \tau _ { t } ) s _ { j } ^ { t } \Big ] ^ { 2 } } \\ { \displaystyle \qquad + \frac { 1 } { N ^ { - } } \sum _ { i } \Big [ s _ { i } ^ { t } - \frac { 1 } { N ^ { - } } \sum _ { j } \mathbf { 1 } ( s _ { j } ^ { t } > \tau _ { t } ) s _ { j } ^ { t } \Big ] ^ { 2 } , } \end{array}\tag{3}
$$

where $\begin{array} { r } { N ^ { + } = \sum _ { i } \mathbf { 1 } \left( s _ { i } ^ { t } \leq \tau _ { t } \right) } \end{array}$ and $\begin{array} { r } { N ^ { - } = \sum _ { i } \mathbf { 1 } \left( s _ { i } ^ { t } > \tau _ { t } \right) } \end{array}$ represent the number of ID and OOD samples respectively. Optimizing $\mathrm { E q . \ ( 3 ) }$ enables finding an optimal threshold $\tau _ { t } ^ { * }$ to minimize the intra-cluster variations. Then, the objective function can be revised into:

$$
\mathcal { L } _ { e } ^ { + } \left( \pmb { \theta } , \mathbf { C } \right) = \frac { 1 } { N ^ { + } } \sum _ { i = 1 } ^ { N ^ { + } } H \left( \mathbf { p } _ { i } ^ { t } \right) .\tag{4}
$$

By applying Eq. (4), reliable updates can be achieved by minimizing the entropy of $N ^ { + }$ samples. In this process, the prototypes C serve as reliable class anchors for distinguishing ID samples from OOD ones, and thus are crucial for stable target-domain updates.

## 3.2 Source-Free Update with Pre-trained Classifier Weights

As discussed in Subsection 3.1, reliable updates rely on access to class prototypes. However, in Open-World Test-Time Adaptation (OWTTA), such prototype information is unavailable, since only a pre-trained model

![](images/97753300e128d629504c896feb85dcd30ca3a2f497bf06db86bf57cc432cc00e.jpg)  
Figure 5: Simplex Equiangular Tight Frame (Simplex ETF) with diferent class numbers.

is given and unlabeled target samples arrive batch-by-batch. Therefore, we derive the prototypes directly from the pre-trained model itself.

Recent neural collapse theory [65] provides a principled explanation for the final state of supervised training. Specifically, when a suficiently large neural network is well-trained on the training data, it tends to exhibit the following properties.

Theorem 1 (Convergence to Simplex ETF [65]) For a suficiently large neural network, the classmeans of the embeddings centered at the global mean become both linearly separable and maximally distant when the model converges. These class-means lie on a sphere centered at the origin, forming a Simplex ETF:

$$
\begin{array} { r l } & { \frac { \left. \mathbf { c } _ { i } - \mathbf { c } _ { G } , \mathbf { c } _ { j } - \mathbf { c } _ { G } \right. } { \| \mathbf { c } _ { i } - \mathbf { c } _ { G } \| _ { 2 } \| \mathbf { c } _ { j } - \mathbf { c } _ { G } \| _ { 2 } } = \left\{ \begin{array} { l l } { 1 , } & { i = j } \\ { - \frac { 1 } { K - 1 } , } & { i \neq j , } \end{array} \right. , } \\ & { \| \mathbf { c } _ { i } - \mathbf { c } _ { G } \| _ { 2 } - \| \mathbf { c } _ { j } - \mathbf { c } _ { G } \| _ { 2 } = 0 , \quad \forall 1 \leq i , j \leq K , } \end{array}
$$

where $\begin{array} { r } { \mathbf { c } _ { j } = \frac { \sum _ { i = 1 } ^ { N } \mathbf { z } _ { i } \mathbb { I } ( \Psi ( \mathbf { z } _ { i } ) = j ) } { \sum _ { i = 1 } ^ { N } \mathbb { I } ( \Psi ( \mathbf { z } _ { i } ) = j ) } } \end{array}$ is the class-mean of j-th class and $\begin{array} { r } { \mathbf { c } _ { G } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \mathbf { z } _ { i } } \end{array}$ is the global mean of feature, both are computed under the neural collapse state.

Theorem 2 (Convergence to Self-duality [65]) For a suficiently large neural network, the last-layer linear classifiers reside in the dual vector space of the class-means when the model has converged:

$$
\mathcal { N } C _ { 3 } : = \left\| \frac { \mathbf { W } \overline { { \mathbf { C } } } } { \left\| \mathbf { W } \overline { { \mathbf { C } } } \right\| _ { F } } - \frac { 1 } { \sqrt { K - 1 } } \left( \mathbf { I } _ { K } - \frac { 1 } { K } \mathbf { 1 } _ { K } \mathbf { 1 } _ { K } ^ { \top } \right) \right\| _ { F } ,\tag{5}
$$

where $\mathbf { \overline { { C } } } = [ \mathbf { c } _ { 1 } - \mathbf { c } _ { G } , \dots , \mathbf { c } _ { K } - \mathbf { c } _ { G } ]$ represents the centered class-mean matrix.

Theorem 3 (Variability Collapse [65]) As training progresses, the features within each class converge to their respective class-means, causing within-class variation to collapse to zero. Let the within-class covariance matrix be denoted as $\boldsymbol { \Sigma } \mathbf { w } \in \mathbb { R } ^ { d \times d }$ and the between-class covariance matrix as $\Sigma _ { \mathbf { B } } \in \mathbb { R } ^ { d \times d }$ , defined as follows:

$$
\Sigma _ { \mathbf { W } } : = \frac { 1 } { N K } \sum _ { i = 1 } ^ { N } \sum _ { j = 1 } ^ { K } \mathbf { d } _ { i j } \mathbf { d } _ { i j } ^ { T } ,
$$

$$
\Sigma _ { \mathbf { B } } : = \frac { 1 } { K } \sum _ { j = 1 } ^ { K } \left( \mathbf { c } _ { j } - \mathbf { c } _ { G } \right) ( \mathbf { c } _ { j } - \mathbf { c } _ { G } ) ^ { T } ,
$$

where $\mathbf { d } _ { i j } = \mathbb { I } ( \Psi ( \mathbf { z } _ { i } ) = j ) \cdot ( \mathbf { z } _ { i } - \mathbf { c } _ { j } )$ is the distance between the i-th feature and its class-mean. Then, we can measure the extent of within-class variability collapse by analyzing the relative magnitudes of $\Sigma _ { \mathbf { B } }$ and Σ<sub>W</sub> for the learned features:

$$
\mathcal { N C } _ { 1 } : = \Sigma _ { \mathbf { B } } ^ { \dagger } \Sigma _ { \mathbf { W } }  0 ,\tag{6}
$$

where † denotes the pseudo-inverse.

These theorems indicate that, at convergence, both the class means and classifier weights tend to align with a Simplex ETF structure, as depicted in Figure 5. Since OWTTA starts from a pre-trained source model, we assume that the source model has been suficiently trained and thus approximately satisfies this neural collapse property. See Table 7 for empirical verification.

Assumption 1 The pre-trained model can be considered well-trained on the source domain, with its classifier weights collapsing into a Simplex ETF.

Assumption 2 Let $F ( \pmb \theta ^ { * } , \cdot ) = \sigma \big ( \mathbf { w } ^ { * } f ( \pmb \theta ^ { * } , \cdot ) + \mathbf { b } ^ { * } \big )$ be a classification network that has reached the neural collapse state, where $\mathbf { w } ^ { * }$ and b<sup>∗</sup> are its classifier weights and bias, respectively. Let ${ \bf z } = f ( \pmb { \theta } ^ { * } , { \bf x } )$ denote the feature representations of the training set $\mathcal { S } _ { S } \in \{ ( \mathbf { x } _ { i } , y _ { i } ) | \mathbf { x } _ { i } \in \mathcal { D } _ { S } \}$ . Then, for some suficiently small positive real ϵ, it holds that

$$
\frac { \| \mathbf { b } _ { j } ^ { * } \| } { \langle \mathbf { w } _ { : , j } ^ { * } , \mathbf { z } _ { \Psi ( \mathbf { z } ) = j } \rangle } < \epsilon , \quad \forall j .\tag{7}
$$

Under Assumption 2, the optimal bias $\mathbf { b } ^ { * }$ is negligible in the neural collapse state. See Table 4.5 for empirical verification. Accordingly, the bias term can be omitted, and the optimization problem max $\langle \mathbf { w } _ { : , j } , \mathbf { z } _ { \Psi ( \mathbf { z } ) = j } \rangle + \mathbf { b } _ { j }$ reduces to $\operatorname* { m a x } _ { \mathbf { w } _ { : , j } } \langle \mathbf { w } _ { : , j } , \mathbf { z } _ { \Psi ( \mathbf { z } ) = j } \rangle$ . This indicates that the classifier weight of each $\mathbf { w } _ { : , j } , \mathbf { b } _ { j }$ class is optimized toward the same direction as its corresponding class prototype. We formalize this relation in the following lemma.

Lemma 1 Let $F ( \pmb \theta ^ { * } , \cdot ) = \sigma \big ( \mathbf { w } ^ { * } f ( \pmb \theta ^ { * } , \cdot ) + \mathbf { b } ^ { * } \big )$ be a suficiently trained classification network that approximately satisfies the neural collapse property. Let $f ( { \pmb \theta } ^ { * } , \mathbf x )$ denote the feature representation of a source training sample x from ${ \cal S } _ { \cal S } = \{ ( { \bf x } _ { i } , y _ { i } ) \vert { \bf x } _ { i } \in { \cal D } _ { \cal S } \}$ . Let $\mathbf { c } _ { j }$ be the prototype of the j-th class. Then we have:

$$
\frac { \mathbf { c } _ { j } } { \| \mathbf { c } _ { j } \| } = \frac { \mathbf { w } _ { : , j } ^ { * } } { \| \mathbf { w } _ { : , j } ^ { * } \| } , \quad \forall j .\tag{8}
$$

Following Lemma 1, we use the normalized classifier weights as class prototypes under the source-free setting. Therefore, Eq. (3) can be computed at each batch step without accessing source data, enabling reliable updates with selected target samples.

## 3.3 Neural Collapse Approximation

We further consider the efect of data distribution shifts. The neural collapse structure learned by the source model reflects the source-domain feature geometry, but may become misaligned with target-domain representations due to the data distribution shifts. From the neural collapse perspective, this featureprototype misalignment between shifted target representations and source-domain prototypes becomes a key reason for performance degradation.

Proposition 1 Given a source domain training set ${ \mathcal { S } } _ { S } \in \{ ( { \bf x } _ { i } , y _ { i } ) | { \bf x } _ { i } \in { \mathcal { D } } _ { S } \}$ and a target domain test set $S _ { T } \in \{ { \bf x } _ { i } | { \bf x } _ { i } \in { \mathcal { D } } _ { T } \}$ . Under the neural collapse state, let $\mathbf { c } _ { S } ^ { \ast }$ and $\mathbf { c } _ { T } ^ { \ast }$ denote the prototypes of source and target domains, respectively, each forming a Simplex ETF:

$$
\begin{array} { r } { \mathbf { c } _ { S } ^ { * } \mathbf { c } _ { S } ^ { * \top } = \alpha \Big ( \mathbf { I } _ { K } - \frac { 1 } { K } \mathbf { 1 } _ { K } \mathbf { 1 } _ { K } ^ { \top } \Big ) , \mathbf { c } _ { T } ^ { * } \mathbf { c } _ { T } ^ { * \top } = \beta \Big ( \mathbf { I } _ { K } - \frac { 1 } { K } \mathbf { 1 } _ { K } \mathbf { 1 } _ { K } ^ { \top } \Big ) , } \end{array}
$$

where $\alpha$ and $\beta$ denote the respective scaling factors for the source and target Simplex ETFs.

By Proposition 1, if $\mathcal { D } _ { S } \ne \mathcal { D } _ { T }$ , then ${ \bf c } _ { S } ^ { * } \neq { \bf c } _ { T } ^ { * } ,$ i.e., the source and target prototypes difer due to the data distribution shift, which leads to classification errors. Based on this understanding, we aim to guide target representations toward a target-domain NC-inspired geometry during adaptation.

Lemma 2 Let $F ( \pmb { \theta } ^ { * } , \cdot ) = \sigma \big ( \mathbf { w } ^ { * } f ( \pmb { \theta } ^ { * } , \cdot ) + \mathbf { b } ^ { * } \big )$ be a classification network that has reached the neural collapse state with the training set $\mathcal { S } _ { S } \in \{ ( \mathbf { x } _ { i } , y _ { i } ) | \mathbf { x } _ { i } \in \mathcal { D } _ { S } \}$ . Then, for each training sample $\mathbf { x } _ { i } ,$ its logits vector $\mathbf { p } =$ $F ( \pmb { \theta } ^ { * } , \mathbf { x } _ { i } )$ is a one-hot vector, i.e.,

$$
\mathbf { p } _ { j } = { \left\{ \begin{array} { l l } { 1 , } & { \Psi ( \mathbf { x } _ { i } ) = j , } \\ { 0 , } & { o t h e r w i s e . } \end{array} \right. }
$$

As stated in Lemma 2, the logits of a classification network tend to degenerate into one-hot vectors under the neural collapse state. The entropy loss in Eq. (4) encourages confident predictions, which implicitly pushes each sample toward its corresponding class mean and promotes a more compact target-domain representation. This connection to neural collapse explains why entropy minimization can improve adaptation efectiveness in TTA.

However, as indicated by Proposition 1, distribution shifts may alter the target-domain feature geometry, making direct entropy minimization in Eq. (4) blindly push target features toward source-domain prototypes. Such indiscriminate updates may distort the embedding space and damage the structured representations required for reliable adaptation.

We therefore seek to approximate the neural collapse state in the target domain. Since exact neural collapse is dificult to achieve in the unsupervised setting, we use it as a reliable structural prior for target adaptation. Specifically, we jointly update the model parameters and prototypes, and further leverage embedding-structure maintenance and stochastic prototype updates to promote balanced and compact representations during adaptation.

## 3.3.1 Maintain Embedding Structure

According to neural collapse theory, learned class representations tend to form a balanced geometric structure, i.e., a Simplex ETF, at convergence. We regard this property as a structural prior and preserve the balanced latent embedding geometry during target adaptation. Let $\mathbf { w } ^ { * }$ denote the classifier weights, and let ${ \mathbf { z } } _ { \Psi ( \mathbf { z } ) = j }$ denote the feature representations assigned to the j-th class. Then we have:

$$
\begin{array} { l } { { \displaystyle { \hat { \bf p } } ^ { * } = \frac { 1 } { K } \sum _ { j = 1 } ^ { K } \sigma ( \langle { \bf w } ^ { * } , { \bf z } _ { \Psi ( { \bf z } ) = j } \rangle ) = \frac { 1 } { K } \sum _ { j = 1 } ^ { K } \sigma ( \langle { \bf w } ^ { * } , { \bf c } _ { j } \rangle ) } } \\ { { \displaystyle ~ = \frac { 1 } { K } \sum _ { j = 1 } ^ { K } \sigma ( [ - \frac { 1 } { K - 1 } , - \frac { 1 } { K - 1 } , \ldots , 1 , \ldots , - \frac { 1 } { K - 1 } ] ) } } \\ { { \displaystyle ~ = \frac { 1 } { K } { \bf 1 } _ { K } , } } \end{array}
$$

where $\hat { \mathbf { p } } ^ { * }$ denotes the ideal class-averaged prediction under neural collapse, corresponding to a uniform distribution over classes.

Guided by this uniform prior, we encourage the average logits of ID samples to approach $\scriptstyle { \frac { 1 } { K } } \mathbf { 1 } _ { K }$ , thereby maintaining a balanced embedding structure during adaptation:

$$
\mathcal { L } _ { d } \left( \pmb { \theta } \right) = D _ { K L } \left( \hat { \mathbf { p } } ^ { t } \bigg \| \frac { 1 } { K } \mathbf { 1 } _ { K } \right) ,\tag{9}
$$

where $\begin{array} { r } { \hat { \mathbf { p } } ^ { t } = \frac { 1 } { N _ { + } } \sum _ { i = 1 } ^ { N ^ { + } } \mathbf { p } _ { i } ^ { t } } \end{array}$ is the average of logits of t-th batch, $D _ { K L }$ is the KL-divergence, $\mathrm { ~ \bf ~ p ~ } ^ { t }$ is the output of ID samples. Eq. (9) regularizes the adaptation by encouraging balanced class-wise predictions, which facilitates the preservation of the neural collapse-inspired embedding structure.

## 3.3.2 Stochastic Prototype Update

As stated in Proposition 1, source-domain and target-domain prototypes may difer under distribution shifts. Therefore, we further investigate how to update prototypes during adaptation. In OWTTA, target samples arrive in a single-pass stream, and each mini-batch usually covers only a subset of classes. This naturally introduces a class-coverage imbalance during prototype updates. For example, when the number of classes is much larger than the batch size, many classes are absent from the current batch. Updating prototypes only with the current batch may therefore produce biased estimates of the target distribution, hindering reliable adaptation and degrading OWTTA performance.

To address this issue, we propose a stochastic prototype update mechanism that selectively updates only the prototypes corresponding to classes present in the current batch, while keeping others fixed. This strategy ensures that updates are more relevant to the incoming data and improves adaptation eficiency. To facilitate this, the objective function is extended as follows:

$$
\begin{array} { l } { { \displaystyle { \mathcal { L } } \left( { \pmb { \theta } } , { \bf C } \right) = { \mathcal { L } } _ { e } ^ { + } \left( { \pmb { \theta } } , { \bf C } \right) + \lambda { \mathcal { L } } _ { d } \left( { \pmb { \theta } } \right) + { \mathcal { L } } _ { s } \left( { \pmb { \theta } } , { \bf C } \right) } \ ~ } \\ { { \displaystyle ~ = \frac { 1 } { N ^ { + } } \sum _ { i = 1 } ^ { N ^ { + } } H \left( { \bf p } _ { i } ^ { t } \right) + \lambda D _ { K L } \left( { \bf p } ^ { t } \bigg \| \frac { 1 } { K } { \bf 1 } _ { K } \right) } \ ~ } \\ { { \displaystyle ~ + \frac { 1 } { N ^ { + } } \sum _ { i = 1 } ^ { N ^ { + } } \sum _ { j = 1 } ^ { K } \xi _ { i j } ^ { t } \left\| { \bf z } _ { i } ^ { t } - { \bf c } _ { j } ^ { t } \right\| _ { 2 } ^ { 2 } } , } \end{array}\tag{10}
$$

where λ is the trade-of parameter and $\xi \in \{ 0 , 1 \}$ is an indicator factor assigning $\mathbf { z } _ { i } ^ { t }$ into its nearest prototype:

$$
\begin{array} { r } { \xi _ { i j } ^ { t } = \left\{ \begin{array} { l l } { 1 , } & { \mathrm { i f ~ } j = \arg \operatorname* { m a x } _ { j } \frac { \exp \left( - \left\| \mathbf { z } _ { i } ^ { t } - \mathbf { c } _ { j } ^ { t - 1 } \right\| _ { 2 } ^ { 2 } \right) } { \sum _ { j } ^ { K } \exp \left( - \left\| \mathbf { z } _ { i } ^ { t } - \mathbf { c } _ { j } ^ { t - 1 } \right\| _ { 2 } ^ { 2 } \right) } , } \\ { 0 , } & { \mathrm { o t h e r w i s e . } } \end{array} \right. } \end{array}\tag{11}
$$

The first term of Eq. (10) is guided by the compactness prior in Theorem 3, encouraging each sample to move closer to its corresponding prototype. The second term is inspired by the simplex-ETF geometry in Theorem 1, encouraging a well-structured inter-class embedding space. The third term alleviates the prototype mismatch described in Proposition 1 by progressively adapting the prototypes toward the targetdomain feature distribution. These three terms jointly incorporate the neural collapse prior into unsupervised target adaptation, enabling the model to form a compact and balanced target-domain representation withou requiring label supervision.

## 3.3.3 Alternating Optimization

Unlike the network parameters θ, prototypes C can be unstable when directly updated by stochastic gradi ents. This instability occurs because the prototypes estimated from diferent mini-batches can vary significantly. To achieve the stochastic prototype update, we propose a decomposition-coordination optimization approach based on the Alternating Direction Method of Multipliers (ADMM) to optimize Eq. (10). In particular, we employ an auxiliary variable $\pmb { \mu }$ to alternately update the model parameters and prototypes:

$$
\begin{array} { l } { \displaystyle \mathcal { L } \left( \boldsymbol { \theta } , \boldsymbol { \mu } , \mathbf { C } \right) = \mathcal { L } _ { e } ^ { + } \left( \boldsymbol { \theta } , \boldsymbol { \mu } \right) + \lambda \mathcal { L } _ { d } \left( \boldsymbol { \theta } \right) + \mathcal { L } _ { s } \left( \boldsymbol { \theta } , \boldsymbol { \mu } \right) } \\ { \displaystyle \quad \quad + \rho \sum _ { j = 1 } ^ { K } \| \boldsymbol { \mu } _ { j } - \mathbf { c } _ { j } \| _ { 2 } ^ { 2 } , } \end{array}
$$

where $\rho$ is a trade-of parameter controlling the prototype updates between batches. Our decomposition-

coordination optimization technique consists of three iterative steps:

$$
\begin{array} { r l } & { \theta ^ { t } = \underset { \theta } { \arg \operatorname* { m i n } } \ \mathcal { L } \left( \theta , \mu ^ { t - 1 } , \mathbf { C } ^ { t - 1 } \right) , } \\ & { \mu ^ { t } = \underset { \mu } { \arg \operatorname* { m i n } } \ \mathcal { L } \left( \theta ^ { t } , \mu , \mathbf { C } ^ { t - 1 } \right) , } \\ & { \mathbf { C } ^ { t } = \underset { \mathbf { C } } { \arg \operatorname* { m i n } } \ \mathcal { L } \left( \theta ^ { t } , \mu ^ { t } , \mathbf { C } \right) . } \end{array}
$$

We then develop eficient update solutions for each sub-problem, allowing for neural collapse approximation.

For updating θ: Since no closed-form solution exists, we optimize $\pmb { \theta } ^ { t }$ using gradient descent:

$$
\pmb { \theta } ^ { t } = \pmb { \theta } ^ { t - 1 } - \epsilon \left. \frac { \partial \mathcal { L } \left( \pmb { \theta } , \pmb { \mu } ^ { t - 1 } , \mathbf { C } ^ { t - 1 } \right) } { \partial \pmb { \theta } } \right| _ { \pmb { \theta = \theta } ^ { t - 1 } } ,\tag{12}
$$

where ϵ is the step size. Eq. (12) can be executed using advanced gradient descent methods.

For updating $\pmb { \mu } \colon$ The sub-problem involves a two-objective optimization. By setting the gradient of the objective function to zero, we can obtain an analytic solution:

$$
\frac { \partial \mathcal { L } \left( \pmb { \theta } ^ { t } , \pmb { \mu } , \mathbf { C } ^ { t - 1 } \right) } { \partial \pmb { \mu } _ { j } } = 0 \implies \pmb { \mu } _ { j } ^ { t } = \frac { \rho \mathbf { c } _ { j } ^ { t - 1 } + \frac { \eta } { N ^ { + } } \sum _ { i = 1 } ^ { N ^ { + } } \xi _ { i j } ^ { t } \mathbf { z } _ { i } ^ { t } } { \rho + \frac { \eta } { N ^ { + } } \sum _ { i = 1 } ^ { N ^ { + } } \xi _ { i j } ^ { t } } ,\tag{13}
$$

where $\eta$ is a balance factor. The numerator of Eq. (13) controls the sample weights of each prototype, while the denominator represents the sample counts assigned to the class. Let $\begin{array} { r } { \kappa ~ = ~ { \frac { \rho } { \rho + { \frac { \eta } { N ^ { + } } } \sum _ { i = 1 } ^ { N ^ { + } } \xi _ { i j } ^ { t } } } } \end{array}$ and $\begin{array} { r } { \bar { \mathbf { c } } _ { j } ^ { t } = \frac { \sum _ { i = 1 } ^ { N ^ { + } } \xi _ { i j } ^ { t } \mathbf { z } _ { i } ^ { t } } { \sum _ { i = 1 } ^ { N ^ { + } } \xi _ { i j } ^ { t } } } \end{array}$ . Eq. (13) can further simplified to the following form:

$$
\pmb { \mu } _ { j } ^ { t } = \kappa \mathbf { c } _ { j } ^ { t - 1 } + ( 1 - \kappa ) \bar { \mathbf { c } } _ { j } ^ { t } , \quad j = 1 , 2 , \ldots , K ,\tag{14}
$$

where κ acts as a self-learning parameter that balances the influence between the previous prototype $\mathbf { c } _ { j } ^ { t - 1 }$ and the prototype calculated in the current batch, $\bar { \mathbf { c } } _ { j } ^ { t }$ . It is noteworthy that in Eq. (14), $\bar { \mathbf { c } } _ { j } ^ { t }$ is calculated based not only on the features assigned to the class but also on the number of samples allocated to it. In fact, Eq. (14) corresponds to the exponential moving average (EMA), which has been demonstrated that the use of it ensures both stability and eficiency in scenarios of mini-batch updates [66].

For updating C: By setting the gradient of the objective function to zero, the analytic solution of C is:

$$
\frac { \partial \mathcal { L } ( \pmb { \theta } ^ { t } , \pmb { \mu } ^ { t } , \mathbf { C } ) } { \partial \mathbf { c } _ { j } } = 0 \implies \mathbf { c } _ { j } ^ { t } = \pmb { \mu } _ { j } ^ { t } , \quad j = 1 , 2 , \ldots , K .\tag{15}
$$

To summarize, our ReNC method filters out the OOD samples at first and approximates the neural collapse state in the target domain by updating prototypes stochastically to achieve reliable and eficient OWTTA. Finally, the predictions $\hat { y } _ { i } ^ { t }$ for each batch can be obtained by:

$$
\hat { y } _ { i } ^ { t } = \left\{ \begin{array} { l l } { F ( \pmb { \theta } ^ { t } , \mathbf { x } _ { i } ^ { t } ) , } & { s _ { i } ^ { t } < \tau _ { t } ^ { * } , } \\ { | \mathcal { V } _ { S } | + 1 , } & { \mathrm { o t h e r w i s e } . } \end{array} \right.\tag{16}
$$

A detailed overview of the proposed method is provided in Algorithm 1.

```perl
Algorithm 1 Reliable Neural Collapse approximation (ReNC)
Require: Test samples x from the target domain $\boldsymbol { S _ { T } }$ , pre-trained model $F ( \pmb \theta , \cdot ) ;$
# Initialize prototypes
1: Initialization: Extract prototypes C from classifier weights w and initialize $\mu = \mathrm { C } ;$
2: for t ← 1 to $T$ do
# Partition data into ID $( N ^ { + } )$ and OOD (N<sup>−</sup>) samples
3: Calculate $s _ { i } ^ { t }$ and $\tau _ { t } ^ { * }$ by Eq. (2) and Eq. (3);
# Neural collapse approximation
4: Update $\pmb { \theta } ^ { t }$ by Eq. (12), update $\mu ^ { t }$ by Eq. (13), update $\mathbf { c } ^ { t }$ by $\operatorname { E q . }$ (15);
# Inference
5: Obtain $\hat { y } _ { i } ^ { t }$ by Eq. (16).
6: end for
```

## 4 Experiments

In this section, we empirically validate the proposed ReNC on several open-world benchmarks. We begin by describing the datasets, evaluation metrics, baselines.Next, we conduct extensive experiments on five open-world benchmarks to demonstrate the efectiveness of our ReNC. Additionally, we validate Assumption 1 and Assumption 2, and further discuss the connection between TTA and neural collapse. Finally, we conduct comprehensive analyses, including ablation studies, computational cost analysis, parameter sensi tivity analysis, diferent ID/OOD ratio settings, continual OWTTA evaluation, ViT-backbone evaluation, and significance tests, to further assess the robustness and efectiveness of our method.

Table 1: In-distribution Datasets Information
<table><tr><td>Datasets</td><td>#Images</td><td>#Classes</td><td>Target Domain</td></tr><tr><td>CIFAR10-C</td><td>10,000</td><td>10</td><td>Corruption</td></tr><tr><td>CIFAR100-C</td><td>10,000</td><td>100</td><td>Corruption</td></tr><tr><td>ImageNet-C</td><td>50,000</td><td>1,000</td><td>Corruption</td></tr><tr><td>ImageNet-R</td><td>30,000</td><td>200</td><td>Style Transfer</td></tr><tr><td>VisDA-C</td><td>55,388</td><td>12</td><td>Style Transfer</td></tr></table>

## 4.1 Open-world Benchmarks

The open-world benchmarks employed in our experiments include both In-Distribution (ID) and Out-Of-Distribution (OOD) samples. Specifically, the ID samples are drawn from two types of target domains, i.e., corruption and style transfer. For corruption domain, CIFAR10-C [67], CIFAR100-C [67], ImageNet-C [67] are introduced, while for style transfer domain, we have ImageNet-R [68] and VisDA-C [69]. Details of the ID samples are presented in Table 1. Besides, we select Gaussian noise and five real-world datasets with distributions diferent from the ID samples to serve as the OOD datasets: Noise, composed with random Gaussian noise, MNIST [70], SVHN [71], Tiny-ImageNet [72], CIFAR100-C and CIFAR10-C.

We concatenate ID and OOD samples at an OOD sample ratio of $\frac { | { \cal S } _ { T _ { O } } | } { | { \cal S } _ { T _ { I } } | } = 1$ to form the open-world benchmarks. Specifically, for CIFAR10-C and CIFAR100-C, the OOD samples include Noise, MNIST, Tiny-ImageNet, and CIFAR100-C/CIFAR10-C. For ImageNet-C, ImageNet-R, and VisDA-C, the OOD samples are Noise, MNIST, and SVHN.

## 4.2 Evaluation metrics

To evaluate the OWTTA performance, similar to [8], we introduce three evaluation metrics:

$$
\begin{array} { r l } & { A C C _ { I } = \frac { \sum _ { \mathbf { x } _ { i } \in \mathcal { D } _ { T _ { I } } } \mathbf { 1 } \left( \hat { y } _ { i } = \Psi \left( \mathbf { x } _ { i } \right) \right) \cdot \mathbf { 1 } \left( \Psi \left( \mathbf { x } _ { i } \right) \in \mathcal { V } _ { s } \right) } { \sum _ { \mathbf { x } _ { i } \in \mathcal { D } _ { T _ { I } } } \mathbf { 1 } \left( \Psi \left( \mathbf { x } _ { i } \right) \in \mathcal { V } _ { s } \right) } , } \\ & { A C C _ { O } = \frac { \sum _ { \mathbf { x } _ { i } \in \mathcal { D } _ { T _ { O } } } \mathbf { 1 } \left( \hat { y } _ { i } \in \mathcal { V } _ { O } \right) \cdot \mathbf { 1 } \left( \Psi \left( \mathbf { x } _ { i } \right) \in \mathcal { V } _ { O } \right) } { \sum _ { \mathbf { x } _ { i } \in \mathcal { D } _ { T _ { O } } } \mathbf { 1 } \left( \Psi \left( \mathbf { x } _ { i } \right) \in \mathcal { V } _ { O } \right) } , } \\ & { A C C _ { H } = 2 \cdot \frac { A C C _ { I } \cdot A C C _ { O } } { A C C _ { I } + A C C _ { O } } , } \end{array}
$$

where $A C C _ { I }$ and $A C C _ { O }$ evaluate the classification accuracy of ID and OOD samples, respectively. $A C C _ { H }$   
is the harmonic mean of $A C C _ { I }$ and $A C C _ { O }$ measuring the overall OWTTA performance.

## 4.3 Baselines

The baselines in our experiments are summarized as follows. TEST: The classification results are obtained directly from the pre-trained model without any further adaptation, serving as a baseline for comparison. BN [73]: The batch normalization layers of the pre-trained network are unfrozen, allowing them to adjust their parameters based on the test samples. TENT [23]: It introduces two afine parameters to adjus the batch normalization layers of the model and updates these parameters by minimizing the entropy loss of target samples. SHOT [12]: It keeps the classifier parameters fixed while using the pre-trained feature encoder as the initialization for learning in the target domain. OSTTA [10]: This method filters out low-confidence OOD samples and adapts the model by minimizing the entropy of high-confidence samples. EATA [5]: This method actively selects high-confidence test samples for adaptation while filtering out low confidence ones, and mitigates catastrophic forgetting through Fisher regularization. RMT [35]: It uses symmetrical cross-entropy as a consistency loss and leverages contrastive learning for feature alignment, reducing error accumulation in continual and gradual test-time adaptation. CoTTA[32]: This method mitigates error accumulation by generating weight-averaged and augmentation-averaged pseudo-labels and alleviates catastrophic forgetting via stochastic restoration of source pre-trained weights during continua test-time adaptation. UniEnt [11]: It uses a two-component GMMs to partition ID and OOD samples based on similarity scores between features and prototypes. The entropy of ID samples is minimized, while that of OOD samples is maximized for better adaptation. OWT3 [8]: This method adapts the model to the target domain by aligning it with statistical information from the source domain and generating new prototypes for OOD samples. However, in OWTTA, since source statistics are unavailable, we replace the prototypes in OWT3 with normalized classifier weights and remove the alignment component of the loss function.

For the baselines not tailored for OWTTA, we follow the approach in [8] for comparison. Specifically, samples from each batch are first partitioned into the ID and OOD samples using $\tau _ { t } ^ { * }$ calculated by Eq. (3), with the prototypes extracted by pre-trained classifier weights. Then, these methods are applied to the ID samples for the adaptation. Following [11], we report results for OSTTA and UniEnt combined with TENT. Besides, our evaluation metric difers from those used in OSTTA and UniEnt <sup>3</sup>, since we apply the stricter evaluation metrics, identical to those used in OWT3 [8].

## 4.4 Overall Comparisons in the OWTTA Scenario

In this section, we validate the efectiveness of the proposed method in the OWTTA scenario. The performance is evaluated using $A C C _ { I } , A C C _ { O }$ and $A C C _ { H }$ . We compare the proposed method with ten baselines across all the open-world benchmarks, including both corrupted and style transfer domains with OOD samples. Tables 2, 3 and 4 present the results for the corrupted domains. The results for the style transfer domain are shown in Tables 5 and 6. These experimental results reveal several noteworthy observations.

1) Our ReNC consistently achieves superior $A C C _ { H }$ across all baselines, validating its capability to efectively adapt to diverse datasets in the open-world scenarios. Specifically, on the ImageNet-C dataset, where existing methods struggle to detect OOD samples using fixed prototypes, ReNC achieves consistently higher $A C C _ { H }$ , highlighting the efectiveness of our prototype updating mechanism. Meanwhile, on ImageNet-R and VisDA-C datasets, where all methods have relatively high $A C C _ { O }$ , our method still improves ID classification accuracy by approximating the neural collapse state in the target domain.

2) UniEnt and OSTTA underperform compared to TENT on nearly all the open-world benchmarks, suggesting that GMMs and confidence-based partitioning mechanisms may not align well with their entropy based optimization goals. In contrast, our ReNC performs data partitioning and prototype updating within the same latent space, making it more robust since both processes are tightly coupled.

3) The classification accuracy decreases for all methods as the number of classes increases, especially for ImageNet-C, which contains 1000 classes. Our results in Table 4 indicate an improvement in $A C C _ { H }$ from 21.37% to 31.02%. This improvement can be attributed to the neural collapse approximation mechanism of ReNC. For the ImageNet-C dataset, the number of classes exceeds the batch size, resulting in imbalanced or even missing classes in each batch. While ReNC updates the prototypes stochastically, it efectively addresses this issue and thus achieves better performance compared to other methods.

In conclusion, our ReNC stochastically updates the prototypes to fit the target data through neural collapse approximation, operating without any source domain information. It achieves the best performance across all the open-world benchmarks.

Table 2: Open-world test-time adaptation results on CIFAR10-C with OOD samples. Best marked in bold, second best underlined.
<table><tr><td rowspan="2">CIFAR10-C</td><td colspan="3">Noise</td><td colspan="3">MNIST</td><td colspan="3">SVHN</td><td colspan="3">Tiny</td><td colspan="3">CIFAR100-C</td></tr><tr><td> $A C C _ { I }$ </td><td> $A C C _ { O }$ </td><td> $A C C _ { H }$ </td><td> $A C C _ { I }$ </td><td> $A C C _ { O }$ </td><td> $A C C _ { H }$ </td><td> $A C C _ { I }$ </td><td> $A C C _ { O }$ </td><td> $A C C _ { H }$ </td><td> $A C C _ { I }$ </td><td> $A C C _ { O }$ </td><td> $A C C _ { H }$ </td><td> $A C C _ { I }$ </td><td> $A C C _ { O }$ </td><td> $A C C _ { H }$ </td></tr><tr><td>TEST</td><td>69.19</td><td>99.96</td><td>81.78</td><td>63.61</td><td>91.11</td><td>74.92</td><td>63.54</td><td>90.91</td><td>74.80</td><td>61.09</td><td>77.48</td><td>68.32</td><td>61.27</td><td>76.58</td><td>68.07</td></tr><tr><td>BN</td><td>39.44</td><td>60.57</td><td>47.77</td><td>60.19</td><td>69.56</td><td>64.54</td><td>69.21</td><td>80.17</td><td>74.29</td><td>69.02</td><td>75.15</td><td>71.95</td><td>68.83</td><td>73.33</td><td>71.01</td></tr><tr><td>TENT</td><td>42.75</td><td>61.99</td><td>50.60</td><td>63.21</td><td>72.80</td><td>67.67</td><td>70.84</td><td>82.92</td><td>76.41</td><td>69.70</td><td>76.23</td><td>72.82</td><td>69.22</td><td>74.07</td><td>71.56</td></tr><tr><td>SHOT</td><td>42.71</td><td>60.72</td><td>50.15</td><td>64.57</td><td>73.51</td><td>68.75</td><td>71.12</td><td>84.37</td><td>77.18</td><td>69.51</td><td>77.06</td><td>73.09</td><td>69.10</td><td>74.76</td><td>71.82</td></tr><tr><td>OSTTA</td><td>53.43</td><td>13.69</td><td>21.80</td><td>53.88</td><td>36.94</td><td>43.83</td><td>62.76</td><td>41.20</td><td>49.74</td><td>74.85</td><td>32.38</td><td>45.20</td><td>76.39</td><td>29.80</td><td>42.87</td></tr><tr><td>EATA</td><td>40.65</td><td>64.14</td><td>49.76</td><td>58.04</td><td>72.63</td><td>64.52</td><td>66.62</td><td>82.29</td><td>73.63</td><td>66.72</td><td>77.37</td><td>71.65</td><td>66.83</td><td>75.00</td><td>70.68</td></tr><tr><td>RMT</td><td>45.50</td><td>44.32</td><td>44.90</td><td>73.35</td><td>99.66</td><td>84.50</td><td>71.97</td><td>88.65</td><td>79.44</td><td>68.65</td><td>78.11</td><td>73.08</td><td>66.84</td><td>74.51</td><td>70.47</td></tr><tr><td>CoTTA</td><td>35.89</td><td>68.20</td><td>47.03</td><td>60.20</td><td>69.25</td><td>64.41</td><td>69.20</td><td>80.11</td><td>74.26</td><td>69.07</td><td>75.14</td><td>71.98</td><td>68.67</td><td>73.65</td><td>71.07</td></tr><tr><td>UniEnt</td><td>39.29</td><td>59.97</td><td>47.48</td><td>59.70</td><td>69.79</td><td>64.35</td><td>68.47</td><td>80.33</td><td>73.93</td><td>68.58</td><td>75.46</td><td>71.86</td><td>68.80</td><td>73.50</td><td>71.07</td></tr><tr><td>OWT3</td><td>69.41</td><td>99.96</td><td>81.93</td><td>63.89</td><td>91.10</td><td>75.11</td><td>63.72</td><td>90.97</td><td>74.94</td><td>61.05</td><td>77.56</td><td>68.32</td><td>61.37</td><td>76.47</td><td>68.09</td></tr><tr><td>Ours</td><td>79.14</td><td>99.98</td><td>88.35</td><td>82.23</td><td>98.12</td><td>89.47</td><td>77.00</td><td>91.54</td><td>83.64</td><td>70.76</td><td>77.63</td><td>74.04</td><td>70.26</td><td>75.06</td><td>72.58</td></tr></table>

Table 3: Open-world test-time adaptation results on CIFAR100-C with OOD samples. Best marked in bold, second best underlined.
<table><tr><td rowspan="2">CIFAR100-C</td><td colspan="3">Noise</td><td colspan="3">MNIST</td><td colspan="3">SVHN</td><td colspan="3">Tiny</td><td colspan="3">CIFAR10-C</td></tr><tr><td> $A C C _ { I }$ </td><td> $A C C _ { O }$ </td><td> $A C C _ { H }$ </td><td>|ACCI</td><td> $A C C _ { O }$ </td><td> $A C C _ { H }$ </td><td> $A C C _ { I }$ </td><td> $A C C _ { O }$ </td><td> $A C C _ { H }$ </td><td> $A C C _ { I }$ </td><td> $A C C _ { O }$ </td><td> $A C C _ { H }$ </td><td>|ACCI</td><td> $A C C _ { O }$ </td><td> $A C C _ { H }$ </td></tr><tr><td>TEST</td><td>25.30</td><td>99.99</td><td>40.38</td><td>24.76</td><td>46.81</td><td>32.39</td><td>26.34</td><td>91.65</td><td>40.92</td><td>26.20</td><td>86.68</td><td>40.24</td><td>25.79</td><td>87.86</td><td>39.88</td></tr><tr><td>BN</td><td>21.22</td><td>93.29</td><td>34.58</td><td>26.09</td><td>93.21</td><td>40.77</td><td>30.60</td><td>95.42</td><td>46.34</td><td>31.41</td><td>89.69</td><td>46.53</td><td>31.95</td><td>89.93</td><td>47.15</td></tr><tr><td>TENT</td><td>24.45</td><td>95.75</td><td>38.95</td><td>27.50</td><td>91.49</td><td>42.29</td><td>32.05</td><td>95.17</td><td>47.95</td><td>31.33</td><td>89.64</td><td>46.43</td><td>32.08</td><td>89.93</td><td>47.29</td></tr><tr><td>SHOT</td><td>25.98</td><td>96.62</td><td>40.95</td><td>28.26</td><td>87.36</td><td>42.71</td><td>32.42</td><td>95.14</td><td>48.36</td><td>31.18</td><td>89.71</td><td>46.28</td><td>32.07</td><td>90.10</td><td>47.30</td></tr><tr><td>OSTTA</td><td>12.20</td><td>40.58</td><td>18.76</td><td>17.97</td><td>46.56</td><td>25.93</td><td>33.33</td><td>62.59</td><td>43.50</td><td>36.81</td><td>52.15</td><td>43.16</td><td>42.75</td><td>43.67</td><td>43.21</td></tr><tr><td>EATA</td><td>22.73</td><td>94.06</td><td>36.61</td><td>25.58</td><td>92.08</td><td>40.04</td><td>29.57</td><td>96.20</td><td>45.24</td><td>30.35</td><td>90.03</td><td>45.40</td><td>31.44</td><td>89.79</td><td>46.57</td></tr><tr><td>RMT</td><td>27.68</td><td>99.92</td><td>43.35</td><td>26.40</td><td>83.61</td><td>40.13</td><td>27.92</td><td>89.52</td><td>42.56</td><td>27.58</td><td>81.59</td><td>41.22</td><td>28.34</td><td>84.31</td><td>42.42</td></tr><tr><td>CoTTA</td><td>20.93</td><td>93.25</td><td>34.19</td><td>25.23</td><td>92.85</td><td>39.68</td><td>30.09</td><td>95.26</td><td>45.73</td><td>31.17</td><td>89.85</td><td>46.28</td><td>31.86</td><td>89.90</td><td>47.05</td></tr><tr><td>UniEnt</td><td>20.05</td><td>94.48</td><td>33.08</td><td>25.54</td><td>92.86</td><td>40.06</td><td>31.36</td><td>94.48</td><td>47.09</td><td>31.20</td><td>89.19</td><td>46.23</td><td>32.52</td><td>89.03</td><td>47.64</td></tr><tr><td>OWT3</td><td>25.54</td><td>99.99</td><td>40.69</td><td>24.51</td><td>45.51</td><td>31.86</td><td>26.48</td><td>91.70</td><td>41.09</td><td>26.34</td><td>86.68</td><td>40.40</td><td>25.93</td><td>87.82</td><td>40.04</td></tr><tr><td>Ours</td><td>48.87</td><td>97.74</td><td>65.16</td><td>38.75</td><td>94.31</td><td>54.93</td><td>40.96</td><td>78.22</td><td>53.77</td><td>36.27</td><td>71.91</td><td>48.22</td><td>35.57</td><td>76.75</td><td>48.61</td></tr></table>

Table 4: Open-world test-time adaptation results on ImageNet-C with OOD samples. Best marked in bold, second best underlined.
<table><tr><td rowspan="2">ImageNet-C</td><td colspan="7">Noise MNIST</td></tr><tr><td> $A C C _ { I }$   $A C C _ { O }$ </td><td>ACCH</td><td></td><td> $A C C _ { I }$  ACCo</td><td>ACCH</td><td> $A C C _ { I }$ </td><td> $A C C _ { O }$ </td><td>ACCH</td></tr><tr><td>TEST</td><td>11.79</td><td>40.85</td><td>18.30</td><td>11.33</td><td>59.57</td><td>19.04</td><td>11.62 82.04</td><td>20.36</td></tr><tr><td>BN</td><td>9.73</td><td>66.50</td><td>16.98</td><td>15.20</td><td>74.24 25.23</td><td>16.24</td><td>77.93</td><td>26.88</td></tr><tr><td>TENT</td><td>9.31</td><td>68.49</td><td>16.39</td><td>15.26</td><td>72.67 25.22</td><td>17.28</td><td>78.79</td><td>28.34</td></tr><tr><td>SHOT</td><td>9.70</td><td>64.57</td><td>16.87</td><td>14.76</td><td>71.69</td><td>24.48 17.70</td><td>79.63</td><td>28.96</td></tr><tr><td>OSTTA</td><td>13.90</td><td>65.34</td><td>22.92</td><td>20.54</td><td>57.34</td><td>30.25 21.93</td><td>59.43</td><td>32.04</td></tr><tr><td>EATA</td><td>15.92</td><td>88.86</td><td>27.00</td><td>20.72</td><td>93.21</td><td>33.90 20.69</td><td>93.14</td><td>33.86</td></tr><tr><td>RMT</td><td>30.38</td><td>99.76</td><td>46.58</td><td>11.92</td><td>90.93</td><td>21.08</td><td>12.30 96.85</td><td>21.83</td></tr><tr><td>CoTTA</td><td>8.75</td><td>67.48</td><td>15.49</td><td>13.78</td><td>82.82</td><td>23.63</td><td>14.13 74.66</td><td>23.76</td></tr><tr><td>UniEnt</td><td>8.29</td><td>73.44</td><td>14.90</td><td>14.13</td><td>79.71</td><td>24.00</td><td>14.81 82.88</td><td>25.13</td></tr><tr><td>OWT3</td><td>9.76</td><td>68.03</td><td>17.07</td><td>15.41</td><td>75.29</td><td>25.58</td><td>16.45 78.67</td><td>27.21</td></tr><tr><td>Ours</td><td>36.96 99.78</td><td></td><td>53.94</td><td>35.61 98.79</td><td></td><td>52.35</td><td>36.73 97.87</td><td>53.41</td></tr></table>

Table 5: Open-world test-time adaptation results on ImageNet-R with OOD samples. Best marked in bold, second best underlined.
<table><tr><td rowspan="2">ImageNet-R.</td><td colspan="3">Noise</td><td colspan="5">MNIST SVHN</td></tr><tr><td>ACC1</td><td> $A C C _ { O }$ </td><td>ACCH</td><td>|ACCI</td><td> $A C C o$ </td><td>ACCH</td><td>|ACCI ACCO</td><td>ACCH</td></tr><tr><td>TEST</td><td></td><td>24.71 100.00</td><td>39.63</td><td>23.60</td><td>99.81</td><td>38.17</td><td>22.46 98.89</td><td>36.61</td></tr><tr><td>BN</td><td>16.38</td><td>89.76</td><td>27.70</td><td>18.70</td><td>82.97 30.52</td><td>19.03</td><td>86.08</td><td>31.17</td></tr><tr><td>TENT</td><td>16.87</td><td>93.23</td><td>28.57</td><td>19.81</td><td>84.72 32.11</td><td>20.09</td><td>87.99</td><td>32.71</td></tr><tr><td>SHOT</td><td>17.27</td><td>96.24</td><td>29.28</td><td>19.70</td><td>81.43</td><td>31.72 20.44</td><td>89.61</td><td>33.29</td></tr><tr><td>OSTTA</td><td>25.46</td><td>58.59</td><td>35.50</td><td>22.46</td><td>49.67</td><td>30.93</td><td>25.66 58.02</td><td>35.58</td></tr><tr><td>EATA</td><td>16.21</td><td>91.71</td><td>27.55</td><td>18.95</td><td>86.24</td><td>31.07</td><td>18.95 88.37</td><td>31.21</td></tr><tr><td>RMT</td><td>8.99</td><td>73.27</td><td>16.02</td><td>12.27</td><td>93.79</td><td>21.70</td><td>16.36 77.63</td><td>27.02</td></tr><tr><td>CoTTA</td><td>15.58</td><td>85.41</td><td>26.35</td><td>17.40</td><td>86.71</td><td>28.98</td><td>17.51 83.04</td><td>28.92</td></tr><tr><td>UniEnt</td><td>10.56</td><td>72.46</td><td>18.43</td><td>15.01</td><td>79.59</td><td>25.26</td><td>15.84 83.72</td><td>26.64</td></tr><tr><td>OWT3</td><td>16.56</td><td>92.13</td><td>28.07</td><td>19.13</td><td>83.75</td><td>31.15</td><td>19.42 86.93</td><td>31.75</td></tr><tr><td>Ours</td><td>39.85</td><td>98.67</td><td>56.77</td><td>37.99</td><td>97.32</td><td>54.65</td><td>39.36 95.68</td><td>55.78</td></tr></table>

Table 6: Open-world test-time adaptation results on VisDA-C with OOD samples. Best marked in bold, second best underlined.
<table><tr><td rowspan="2">VisDA-C</td><td colspan="3">Noise</td><td colspan="3">MNIST</td><td colspan="3">SVHN</td></tr><tr><td> $A C C _ { I }$ </td><td>ACCO</td><td>ACCH</td><td></td><td>|ACCI ACCO</td><td>ACCH|</td><td>|ACCI ACCO</td><td></td><td>ACCH</td></tr><tr><td>TEST</td><td></td><td>41.27100.00</td><td>58.43</td><td>43.12</td><td>97.41</td><td>59.78</td><td>42.06</td><td>99.46</td><td>59.12</td></tr><tr><td>BN</td><td>45.48</td><td>99.64</td><td>62.45</td><td>42.06</td><td>79.96</td><td>55.12</td><td>42.80</td><td>88.77</td><td>57.75</td></tr><tr><td>TENT</td><td></td><td>51.57 100.00</td><td>68.05</td><td>41.50</td><td>79.96</td><td>54.64</td><td>44.57</td><td>88.40</td><td>59.26</td></tr><tr><td>SHOT</td><td>42.39</td><td>99.94</td><td>59.53</td><td>42.69</td><td>68.08</td><td>52.48</td><td>43.45</td><td>72.69</td><td>54.33</td></tr><tr><td>OSTTA</td><td>39.60</td><td>54.33</td><td>45.81</td><td>46.41</td><td>29.81</td><td>36.30</td><td>44.91</td><td>45.55</td><td>45.23</td></tr><tr><td>EATA</td><td>43.12</td><td>99.68</td><td>60.20</td><td>41.49</td><td>81.13</td><td>54.90</td><td>41.75</td><td>89.25</td><td>56.89</td></tr><tr><td>RMT</td><td>25.87</td><td>74.63</td><td>38.42</td><td>26.09</td><td>99.98</td><td>41.38</td><td>30.36</td><td>63.64</td><td>41.11</td></tr><tr><td>CoTTA</td><td>44.01</td><td>99.56</td><td>61.04</td><td>40.71</td><td>83.91</td><td>54.82</td><td>40.89</td><td>89.86</td><td>56.20</td></tr><tr><td>UniEnt</td><td>54.84</td><td>95.99</td><td>69.80</td><td>41.92</td><td>80.13</td><td>55.04</td><td>42.83</td><td>88.53</td><td>57.73</td></tr><tr><td>OWT3</td><td>46.12</td><td>99.99</td><td>63.12</td><td>42.14</td><td>77.39</td><td>54.57</td><td>43.04</td><td>91.60</td><td>58.56</td></tr><tr><td>Ours</td><td>55.96</td><td>99.63</td><td>71.67</td><td>47.42</td><td>99.17</td><td>64.16</td><td>53.93</td><td>98.23</td><td>69.63</td></tr></table>

## 4.5 Connections between TTA and Neural Collapse

In this section, we investigate the connection between TTA and neural collapse to validate that the proposed method aligns with the intrinsic properties of well-trained neural networks.

Neural collapse in the source domain: Assumption 1 states that the pre-trained model has already collapsed in the source domain. To validate this, we calculate the $\mathcal { N C } _ { 1 }$ and $\mathcal { N C } _ { 3 }$ of the pre-trained model using its training data. $\mathcal { N C } _ { 1 }$ reflects how well the features collapse to their class-means and $\mathcal { N C } _ { 3 }$ measures the self-duality between class-mean and the classifier weights. The lower the values of these two metrics, the closer the model is to achieving the neural collapse state. As shown in Table $^ { 7 , }$ the models trained on diferent datasets have indeed collapsed on their respective datasets as indicated by low $\mathcal { N C } _ { 1 }$ and $\mathcal { N C } _ { 3 }$ values<sup>4</sup>.

Table 7: $\mathcal { N C } _ { 1 }$ and $\mathcal { N C } _ { 3 }$ on each training dataset. Lower values indicate closer to the neural collapse state.
<table><tr><td>CIFAR10</td><td colspan="2">CIFAR100</td><td colspan="2">ImageNet</td><td colspan="2">VisDA</td></tr><tr><td> $\mathcal { N C } _ { 1 }$ </td><td> $\mathcal { N C } _ { 3 }$ </td><td> $\mathcal { N C } _ { 1 }$ </td><td> $\mathcal { N C } _ { 3 }$ </td><td> $\mathcal { N C } _ { 1 }$   $\mathcal { N C } _ { 3 }$ </td><td> $\mathcal { N C } _ { 1 }$ </td><td> $\mathcal { N C } _ { 3 }$ </td></tr><tr><td>0.17120.2466</td><td></td><td>0.7541 0.6235</td><td>2.3275</td><td>1.1968</td><td>0.3031</td><td>0.5146</td></tr></table>

Table 8: The ratio b to $\langle \mathbf { w } , \mathbf { c } \rangle$ quantifies their relative influence. Lower values imply that the bias term is increasingly negligible in the neural collapse state.
<table><tr><td>Dataset</td><td>CIFAR10</td><td>CIFAR100</td><td>ImageNet</td><td>VisDA</td></tr><tr><td>Ratio</td><td></td><td></td><td> $1 . 5 2 6 \times 1 0 ^ { - 2 } 7 . 6 9 3 \times 1 0 ^ { - 3 } 1 . 1 1 6 \times 1 0 ^ { - 3 } 1 . 8 7 3 \times 1 0 ^ { - 3 }$ </td><td></td></tr></table>

Empirical validation of Assumption 2: To verify the validity of Assumption 2, we conduct experiments on a series of classification neural networks. According to Theorem 3, in the context of neural collapse, features z have collapsed to their class-mean, i.e., prototype c. Therefore, to empirically validate this assumption, we measure the ratio of the bias term $\mathbf { b } _ { j }$ to the weight-prototype interaction $\langle \mathbf { w } _ { : , j } , \mathbf { c } _ { j } \rangle$ , i.e., $\begin{array} { r } { \frac { 1 } { K } \sum _ { j } ^ { K } \frac { \left\| \mathbf { b } _ { j } \right\| } { \left. \mathbf { w } _ { : , j } , \mathbf { c } _ { j } \right. } } \end{array}$ . As shown in Table 8, the weight-prototype interaction significantly outweighs the bias term, $\begin{array} { r } { \mathrm { i . e . } \frac { 1 } { K } \sum _ { j } ^ { K } \frac { \| \mathbf { b } _ { j } \| } { \langle \mathbf { w } _ { : , j } , \mathbf { c } _ { j } \rangle } \ll 1 } \end{array}$ . This empirical result validates Assumption 2, which posits that the efect of the bias term is negligible in the neural collapse state.

Neural collapse in the target domain: TTA algorithms update model parameters to align with the data distribution of the target domain. Their primary goal is to correct the inaccuracies caused by target samples producing incorrect labels when passed through the pre-trained model. Since the pre-trained model has well-collapsed in the source domain, following Theorem 2 and Theorem 3, each feature should collapse to its corresponding classifier weight vector. The inaccuracies occur because the features generated by the encoder do not align well with the pre-trained classifier. From this perspective, efective TTA should guide target representations toward a target-domain NC-like geometry. To support this claim, we measure the $\mathcal { N C } _ { 1 }$ and $\mathcal { N C } _ { 3 }$ on all the open-world benchmarks, and the results are presented in Tables 9-12.

Table 9: $\mathcal { N C } _ { 1 }$ and $\mathcal { N C } _ { 3 }$ on CIFAR10-C and CIFAR100-C with OOD samples. Best results are marked in bold, second best are underlined. The gray background indicates the highest $A C C _ { S } .$
<table><tr><td rowspan="2">CIFAR10-C</td><td colspan="2">Noise</td><td colspan="2">MNIST</td><td colspan="2">SVHN</td><td colspan="2">Tiny</td><td colspan="2">CIFAR100-C</td></tr><tr><td> $\mathcal { N C } _ { 1 }$ </td><td> $\mathcal { N C } _ { 3 }$ </td><td> $\mathcal { N C } _ { 1 }$ </td><td>NC3</td><td> $\mathcal { N C } _ { 1 }$ </td><td> $\mathcal { N C } _ { 3 }$ </td><td>1  ${ \mathcal { N C } } _ { 1 }$ </td><td> $\mathcal { N C } _ { 3 }$ </td><td> $\mathcal { N C } _ { 1 }$ </td><td> $\mathcal { N C } _ { 3 }$ </td></tr><tr><td>TEST</td><td>0.7386</td><td>0.3548</td><td>0.7386</td><td>0.3548</td><td>0.7386</td><td>0.3548</td><td>0.7386</td><td>0.3548</td><td>0.7386</td><td>0.3548</td></tr><tr><td>BN</td><td>1.5176</td><td>0.4034</td><td>0.8026</td><td>0.3637</td><td>0.6209</td><td>0.3189</td><td>0.6052</td><td>0.3204</td><td>0.5970</td><td>0.3193</td></tr><tr><td>TENT</td><td>1.4132</td><td>0.3917</td><td>0.7314</td><td>0.3497</td><td>0.5924</td><td>0.3151</td><td>0.5946</td><td>0.3179</td><td>0.5884</td><td>0.3179</td></tr><tr><td>SHOT</td><td>1.3871</td><td>0.3914</td><td>0.7057</td><td>0.3461</td><td>0.5815</td><td>0.3147</td><td>0.5914</td><td>0.3180</td><td>0.5863</td><td>0.3185</td></tr><tr><td>OSTTA</td><td>1.4157</td><td>0.4056</td><td>0.7922</td><td>0.3633</td><td>0.6154</td><td>0.3194</td><td>0.5944</td><td>0.3212</td><td>0.5852</td><td>0.3203</td></tr><tr><td>EATA</td><td>1.5056</td><td>0.4046</td><td>0.8125</td><td>0.3665</td><td>0.6307</td><td>0.3205</td><td>0.6157</td><td>0.3224</td><td>0.6065</td><td>0.3215</td></tr><tr><td>RMT</td><td>1.3723</td><td>0.4611</td><td>0.6609</td><td>0.3963</td><td>0.6399</td><td>0.3979</td><td>0.6970</td><td>0.4060</td><td>0.6858</td><td>0.4143</td></tr><tr><td>CoTTA</td><td>1.9769</td><td>0.4753</td><td>0.7999</td><td>0.3626</td><td>0.6208</td><td>0.3184</td><td>0.6074</td><td>0.3206</td><td>0.5963</td><td>0.3189</td></tr><tr><td>UniEnt</td><td>1.4784</td><td>0.4017</td><td>0.8185</td><td>0.3666</td><td>0.6270</td><td>0.3195</td><td>0.6101</td><td>0.3201</td><td>0.6039</td><td>0.3200</td></tr><tr><td>OWT3</td><td>0.7358</td><td>0.3545</td><td>0.7358</td><td>0.3545</td><td>0.7350</td><td>0.3545</td><td>0.7366</td><td>0.3545</td><td>0.7368</td><td>0.3545</td></tr><tr><td>Ours</td><td>0.5731</td><td>0.3202</td><td>0.5782</td><td>0.3174</td><td>0.5667</td><td>0.3203</td><td>0.5795</td><td>0.3195</td><td>0.5761</td><td>0.3202</td></tr><tr><td rowspan="2">CIFAR100-C</td><td colspan="2">Noise</td><td colspan="2">MNIST</td><td colspan="2">SVHN</td><td colspan="2">Tiny</td><td colspan="2">CIFAR10-C</td></tr><tr><td> $\mathcal { N C } _ { 1 }$ </td><td> $\mathcal { N C } _ { 3 }$ </td><td> $\mathcal { N C } _ { 1 }$ </td><td> $\mathcal { N C } _ { 3 }$ </td><td> $\mathcal { N C } _ { 1 }$ </td><td> $\mathcal { N C } _ { 3 }$ </td><td>1  $\mathcal { N C } _ { 1 }$ </td><td> $\mathcal { N C } _ { 3 }$ </td><td> $\mathcal { N C } _ { 1 }$ </td><td> $\mathcal { N C } _ { 3 }$ </td></tr><tr><td>TEST</td><td>9.5278</td><td>0.8884</td><td>9.5278</td><td>0.8884</td><td>9.5278</td><td>0.8884</td><td>9.5278</td><td>0.8884</td><td>9.5278</td><td>0.8884</td></tr><tr><td>BN</td><td>14.1927</td><td>0.9529</td><td>10.6814</td><td>0.9139</td><td>8.2031</td><td>0.8764</td><td>7.8558</td><td>0.8689</td><td>7.7276</td><td>0.8709</td></tr><tr><td>TENT</td><td>11.6919</td><td>0.9177</td><td>9.4333</td><td>0.8909</td><td>7.7855</td><td>0.8649</td><td>7.7511</td><td>0.8631</td><td>7.6456</td><td>0.8652</td></tr><tr><td>SHOT</td><td>10.5667</td><td>0.9027</td><td>8.8852</td><td>0.8815</td><td>7.6591</td><td>0.8608</td><td>7.7561</td><td>0.8610</td><td>7.6657</td><td>0.8631</td></tr><tr><td>OSTTA</td><td>13.3452</td><td>0.9586</td><td>10.3678</td><td>0.9193</td><td>7.9534</td><td>0.8755</td><td>7.4570</td><td>0.8673</td><td>6.9739</td><td>0.8634</td></tr><tr><td>EATA</td><td>13.0616</td><td>0.9414</td><td>10.9230</td><td>0.9124</td><td>7.9622</td><td>0.8725</td><td>7.6159</td><td>0.8682</td><td>7.3165</td><td>0.8675</td></tr><tr><td>RMT</td><td>8.1638</td><td>0.9008</td><td>8.2756</td><td>0.9038</td><td>8.2989</td><td>0.9036</td><td>8.5444</td><td>0.9172</td><td>8.0033</td><td>0.9041</td></tr><tr><td>CoTTA</td><td>14.6210</td><td>0.9554</td><td>11.2101</td><td>0.9200</td><td>8.4420</td><td>0.8792</td><td>7.9485</td><td>0.8704</td><td>7.8608</td><td>0.8727</td></tr><tr><td>UniEnt</td><td>14.2197</td><td>0.9363</td><td>11.7456</td><td>0.9174</td><td>8.3867</td><td>0.8729</td><td>8.2405</td><td>0.8713</td><td>8.1938</td><td>0.8757</td></tr><tr><td>OWT3</td><td>9.4840</td><td>0.8877</td><td>9.5121</td><td>0.8882</td><td>9.4805</td><td>0.8876</td><td>9.4835</td><td>0.8877</td><td>9.4799</td><td>0.8876</td></tr><tr><td>Ours</td><td>7.7409</td><td>0.8606</td><td>7.6476</td><td>0.8700</td><td>8.0640</td><td>0.8604</td><td>7.7153</td><td>0.8557</td><td>7.6808</td><td>0.8577</td></tr></table>

Table 10: $\mathcal { N C } _ { 1 }$ and $\mathcal { N C } _ { 3 }$ on ImageNet-C with OOD samples. Best results marked in bold, second best underlined. The gray background indicates the highest $A C C _ { S }$
<table><tr><td rowspan="2">ImageNet-C</td><td colspan="2">Noise</td><td colspan="2">MNIST</td><td colspan="2">SVHN</td></tr><tr><td> $\mathcal { N C } _ { 1 }$ </td><td> $\mathcal { N C } _ { 3 }$ </td><td> $\mathcal { N C } _ { 1 }$ </td><td> $\mathcal { N C } _ { 3 }$ </td><td> $\mathcal { N C } _ { 1 }$ </td><td> $\mathcal { N C } _ { 3 }$ </td></tr><tr><td>TEST</td><td>3.7824</td><td>1.2762</td><td>3.7824</td><td>1.2762</td><td>3.7824</td><td>1.2762</td></tr><tr><td>BN</td><td>7.6031</td><td>1.2733</td><td>6.0601</td><td>1.2620</td><td>6.7761</td><td>1.2581</td></tr><tr><td>TENT</td><td>7.4605</td><td>1.2755</td><td>5.6129</td><td>1.2635</td><td>6.3444</td><td>1.2574</td></tr><tr><td>SHOT</td><td>7.6685</td><td>1.2759</td><td>5.8075</td><td>1.2610</td><td>6.3542</td><td>1.2557</td></tr><tr><td>OSTTA</td><td>6.0655</td><td>1.2586</td><td>4.1373</td><td>1.2554</td><td>4.1407</td><td>1.2557</td></tr><tr><td>EATA</td><td>6.9685</td><td>1.2501</td><td>5.3900</td><td>1.2427</td><td>5.4280</td><td>1.2527</td></tr><tr><td>RMT</td><td>1.9977</td><td>1.4037</td><td>2.7490</td><td>1.3911</td><td>2.0345</td><td>1.4074</td></tr><tr><td>CoTTA</td><td>7.7565</td><td>1.2733</td><td>7.2648</td><td>1.2652</td><td>7.9593</td><td>1.2616</td></tr><tr><td>UniEnt</td><td>8.0681</td><td>1.2695</td><td>5.6453</td><td>1.2506</td><td>5.8017</td><td>1.2496</td></tr><tr><td>OWT3</td><td>7.4846</td><td>1.2724</td><td>5.9790</td><td>1.2609</td><td>6.7146</td><td>1.2571</td></tr><tr><td>Ours</td><td>5.6142</td><td>1.2418</td><td>5.3901</td><td>1.2436</td><td>5.3158</td><td>1.2434</td></tr></table>

Table 11: $\mathcal { N C } _ { 1 }$ and $\mathcal { N C } _ { 3 }$ on ImageNet-R with OOD samples. Best results marked in bold, second best underlined. The gray background indicates the highest $A C C _ { S }$
<table><tr><td rowspan="2">ImageNet-R</td><td colspan="2">Noise</td><td colspan="2">MNIST</td><td colspan="2">SVHN</td></tr><tr><td> $\mathcal { N C } _ { 1 }$ </td><td> $\mathcal { N C } _ { 3 }$ </td><td> $\mathcal { N C } _ { 1 }$ </td><td> $\mathcal { N C } _ { 3 }$ </td><td> $\mathcal { N C } _ { 1 }$ </td><td> $\mathcal { N C } _ { 3 }$ </td></tr><tr><td>TEST</td><td>20.6384</td><td>1.0441</td><td>20.6384</td><td>1.0441</td><td>20.6384</td><td>1.0441</td></tr><tr><td>BN</td><td>30.1058</td><td>1.0541</td><td>28.6891</td><td>1.0577</td><td>27.8901</td><td>1.0540</td></tr><tr><td>TENT</td><td>27.4597</td><td>1.0478</td><td>27.5288</td><td>1.0549</td><td>26.6687</td><td>1.0507</td></tr><tr><td>SHOT</td><td>26.1513</td><td>1.0437</td><td>27.2900</td><td>1.0539</td><td>25.9399</td><td>1.0480</td></tr><tr><td>OSTTA</td><td>25.8711</td><td>1.0483</td><td>24.8492</td><td>1.0584</td><td>24.3873</td><td>1.0555</td></tr><tr><td>EATA</td><td>29.5161</td><td>1.0521</td><td>28.0359</td><td>1.0552</td><td>27.4065</td><td>1.0521</td></tr><tr><td>RMT</td><td>44.8059</td><td>1.1647</td><td>37.8196</td><td>1.1126</td><td>39.1313</td><td>1.1318</td></tr><tr><td>CoTTA</td><td>32.4597</td><td>1.0595</td><td>31.2893</td><td>1.0632</td><td>30.3840</td><td>1.0595</td></tr><tr><td>UniEnt</td><td>33.4932</td><td>1.0751</td><td>26.7478</td><td>1.0536</td><td>26.0775</td><td>1.0478</td></tr><tr><td>OWT3</td><td>28.9696</td><td>1.0513</td><td>27.6794</td><td>1.0548</td><td>27.1448</td><td>1.0512</td></tr><tr><td>Ours</td><td>19.7501</td><td>1.0345</td><td>19.3365</td><td>1.0369</td><td>19.2924 1.0383</td><td></td></tr></table>

Table 12: $\mathcal { N C } _ { 1 }$ and $\mathcal { N C } _ { 3 }$ on VisDA-C with OOD samples. Best results marked in bold, second best underlined. The gray background indicates the highest $A C C _ { S }$
<table><tr><td rowspan="2">VisDA-C</td><td colspan="2">Noise</td><td colspan="2">MNIST</td><td colspan="2">SVHN</td></tr><tr><td> $\mathcal { N C } _ { 1 }$ </td><td> $\mathcal { N C } _ { 3 }$ </td><td> $\mathcal { N C } _ { 1 }$ </td><td> $\mathcal { N C } _ { 3 }$ </td><td> $\mathcal { N C } _ { 1 }$ </td><td> $\mathcal { N C } _ { 3 }$ </td></tr><tr><td>TEST</td><td>1.8591</td><td>0.8081</td><td>1.8591</td><td>0.8081</td><td>1.8591</td><td>0.8081</td></tr><tr><td>BN</td><td>1.3001</td><td>0.7268</td><td>1.2937</td><td>0.7217</td><td>1.2915</td><td>0.7126</td></tr><tr><td>TENT</td><td>1.1664</td><td>0.7034</td><td>1.3417</td><td>0.7287</td><td>1.1755</td><td>0.7005</td></tr><tr><td>SHOT</td><td>1.0559</td><td>0.6944</td><td>1.1516</td><td>0.7098</td><td>1.1388</td><td>0.7003</td></tr><tr><td>OSTTA</td><td>1.4202</td><td>0.7503</td><td>1.2309</td><td>0.748</td><td>1.2792</td><td>0.7308</td></tr><tr><td>EATA</td><td>1.2970</td><td>0.7272</td><td>1.2910</td><td>0.7215</td><td>1.2934</td><td>0.7130</td></tr><tr><td>RMT</td><td>6.2168</td><td>0.8892</td><td>6.0925</td><td>0.8393</td><td>4.8335</td><td>0.8780</td></tr><tr><td>CoTTA</td><td>1.3695</td><td>0.7298</td><td>1.3806</td><td>0.7229</td><td>1.3659</td><td>0.7158</td></tr><tr><td>UniEnt</td><td>1.1522</td><td>0.7019</td><td>1.3653</td><td>0.6829</td><td>1.3198</td><td>0.6778</td></tr><tr><td>OWT3</td><td>1.1679</td><td>0.7083</td><td>1.1764</td><td>0.7031</td><td>1.1955</td><td>0.6983</td></tr><tr><td>Ours</td><td>1.1548</td><td>0.7097</td><td>1.3135</td><td>0.7055</td><td>1.2832</td><td>0.7046</td></tr></table>

For the corrupted domain, we observe that stronger ID classification performance is often accompanied by better NC-related properties. For example, on CIFAR10-C with Noise and MNIST as OOD samples, ReNC achieves the lowest $\mathcal { N C } _ { 1 }$ and $\mathcal { N C } _ { 3 }$ values among all methods, together with the highest $A C C _ { S }$ . For CIFAR10-C with Tiny-ImageNet and CIFAR100-C as OOD samples, OSTTA obtains the best $A C C _ { S }$ , while ReNC still maintains competitive NC-related metrics. These results suggest that although classification accuracy is not solely determined by NC metrics, preserving compact and well-aligned representations can efectively facilitate the target adaptation process.

For the style-transfer domain, as shown in Tables 11 and 12, ImageNet-R follows a similar trend to the corrupted domain. VisDA-C presents a more challenging case, where SHOT achieves relatively strong collapse metrics but still sufers from poor ID classification performance, even falling below TEST in some settings. This discrepancy may be due to the distinct training and testing sets in VisDA-C, which cause the source-domain pre-trained classifier to poorly align with the target domain features. ReNC achieves comparable NC-related properties and strong adaptation performance, supporting the efectiveness of neural collapse approximation.

In general, neural collapse provides an informative perspective for understanding classification representations. The improved NC-related properties of ReNC ofer a plausible explanation for its better performance.

## 4.6 Ablation Study

In this section, we evaluate the impact of entropy minimization $( \mathcal { L } _ { e } ^ { + } )$ , the one-hot distribution prior $( \mathcal { L } _ { d } )$ , the alignment of samples with their prototypes $( \mathcal { L } _ { s } )$ , and finally, the efectiveness of Updating Prototypes (U.P.). The TTA performance is reported using $A C C _ { H }$ on all the open-world benchmarks, as shown in Table 13 and Table 14. From these tables, we have the following observations:

1) $\mathcal { L } _ { s }$ may degrade classification performance if the prototypes are not updated. For example, on the CIFAR100-C dataset, applying $\mathcal { L } _ { s }$ leads to a drop in $A C C _ { H }$ with OOD samples such as MNIST, SVHN, Tiny-ImageNet and CIFAR10-C. This may be because the prototypes, initialized by the weights of the pre trained classifier, do not fit well with the target domain. Pushing samples towards these fixed prototypes can mislead the classification process. 2) Updating prototypes generally enhances TTA performance on all the open-world benchmarks. This can be attributed to the proposed neural collapse approximation mechanism, which updates prototypes stochastically, enabling the prototypes to well align with the target domain. 3) Simply minimizing $\mathcal { L } _ { e } ^ { + }$ or $\mathcal { L } _ { d }$ can lead to degraded TTA performance. As shown in Table 14, when updating prototypes, optimizing the model with $\mathcal { L } _ { e } ^ { + }$ or $\mathcal { L } _ { d }$ alone does not provide optimal adaptation results. Overall, using any of these losses individually may not yield better adaptation results.

Our ReNC integrates all the losses and updating mechanisms into a unified pipeline. It encourages variability collapse by pushing samples toward their corresponding prototypes, while preserving balanced embedding structure, which supports more robust and efective adaptation.

Table 13: Ablation study on the CIFAR10-C and CIFAR100-C with OOD samples. Best results marked in bold, second best underlined.
<table><tr><td>CIFAR10-C</td><td></td><td>Noise</td><td>MNIST SVHN</td><td>Tiny</td><td>CIFAR100-C</td></tr><tr><td> $\mathcal { L } _ { e } ^ { + }$   $\mathcal { L } _ { d }$ </td><td>Ls U.P.|</td><td> $A C C _ { H }$   $A C C _ { H }$ </td><td> $A C C _ { H }$ </td><td> $A C C _ { H }$ </td><td> $A C C _ { H }$ </td></tr><tr><td>√ √√</td><td></td><td>86.05 86.07</td><td>78.60 82.10 78.56 82.06</td><td>69.25 69.27</td><td>68.04 68.00</td></tr><tr><td> $\sqrt { \mathrm { ~ { ~ \bf ~ \psi ~ } ~ } } \sqrt { \mathrm { ~ { ~ \bf ~ \psi ~ } ~ } } \sqrt { \mathrm { ~ { ~ \bf ~ \psi ~ } ~ } }$ </td><td>V √</td><td>86.05 88.02</td><td>78.62 82.06 89.21 83.22</td><td>69.27 73.77</td><td>68.00 72.13</td></tr><tr><td>√</td><td>√ √ √ √</td><td>88.07 88.04</td><td>89.24 89.21</td><td>83.30 73.42 83.32</td><td>71.47 71.81</td></tr><tr><td>V V</td><td>V √</td><td>88.35</td><td>89.47 83.64</td><td>73.34 74.04</td><td>72.58</td></tr><tr><td>CIFAR100-C</td><td></td><td>Noise</td><td>MNIST</td><td>SVHN Tiny</td><td>CIFAR10-C</td></tr><tr><td> $\mathcal { L } _ { e } ^ { + }$   $\mathcal { L } _ { d }$ </td><td> $\mathcal { L } _ { s }$  U.P.|</td><td> $A C C _ { H }$ </td><td> $A C C _ { H }$ </td><td> $A C C _ { H }$ </td><td></td></tr><tr><td>V</td><td></td><td>48.99</td><td></td><td> $A C C _ { H }$  46.21</td><td> $A C C _ { H }$ </td></tr><tr><td>√</td><td></td><td>49.28</td><td>43.49 43.50</td><td>46.22 46.29 46.43</td><td>46.51 46.61</td></tr><tr><td>V √</td><td>√</td><td>49.36</td><td>43.34</td><td>46.21 46.26</td><td>46.44</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>√</td><td>√ √</td><td>57.06</td><td>54.58</td><td>53.31</td><td>47.05 47.46</td></tr><tr><td></td><td>√ √</td><td>63.19</td><td>52.57</td><td>52.22 47.22</td><td>47.56</td></tr><tr><td></td><td>√ V</td><td>54.10</td><td>51.39</td><td>50.85 46.36</td><td>46.45</td></tr><tr><td></td><td>√ V</td><td>65.16</td><td>54.93</td><td>53.77</td><td>48.22</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>48.61</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

## 4.7 Computational Cost Analysis

In this section, we evaluate the computational cost of each method. Since OOD detection is performed in a batch-wise manner, we set the batch size to 64 and measure the per-sample running time across all benchmarks with Noise OOD samples. The results are reported in Table 15. Under the same experimental settings, TEST and BN show the lowest computational overhead. The proposed ReNC method, while incurring approximately 2–3 times higher cost, achieves significantly better performance. Compared with other OWTTA-specific methods, ReNC demonstrates competitive eficiency relative to UniEnt (0.003270 vs. 0.003955) while achieving superior results. Notably, ReNC runs substantially faster than OWT3 on the ImageNet-C dataset (0.003270 vs. 0.006559). These results suggest that ReNC achieves a favorable balance between computational cost and performance.

Table 14: Ablation study on the ImageNet-C, ImageNet-R and VisDA-C with OOD samples. Best results marked in bold, second best underlined.
<table><tr><td colspan="4">ImageNet-C</td><td>Noise</td><td>MNIST</td><td>SVHN</td></tr><tr><td> $\mathcal { L } _ { e } ^ { + }$ </td><td> $\mathcal { L } _ { d }$ </td><td> $\mathcal { L } _ { s }$ </td><td>U.P. |</td><td> $A C C _ { H }$ </td><td> $A C C _ { H }$ </td><td> $A C C _ { H }$ </td></tr><tr><td> $\checkmark$ </td><td></td><td></td><td></td><td>33.11</td><td>34.96</td><td>36.56</td></tr><tr><td> $\checkmark$ </td><td> $\checkmark$ </td><td></td><td></td><td>33.25</td><td>35.00</td><td>36.47</td></tr><tr><td> $\checkmark$ </td><td> $\checkmark$ </td><td> $\checkmark$ </td><td></td><td>32.27</td><td>35.33</td><td>36.44</td></tr><tr><td></td><td></td><td>√</td><td>√</td><td>52.84</td><td>52.10</td><td>52.45</td></tr><tr><td> $\checkmark$ </td><td></td><td>√</td><td>V</td><td>51.67</td><td>51.23</td><td>51.53</td></tr><tr><td></td><td> $\checkmark$ </td><td>√</td><td>V</td><td>51.64</td><td>51.17</td><td>51.23</td></tr><tr><td> $\checkmark$ </td><td> $\checkmark$ </td><td>√</td><td>V</td><td>53.94</td><td>52.35</td><td>53.41</td></tr><tr><td colspan="5">ImageNet-R</td><td>MNIST</td><td>SVHN</td></tr><tr><td> $\mathcal { L } _ { e } ^ { + }$ </td><td> $\mathcal { L } _ { d }$ </td><td> $\mathcal { L } _ { s }$ </td><td> $\mathrm { { U . P . } }$ </td><td> $A C C _ { H }$ </td><td> $A C C _ { H }$ </td><td> $A C C _ { H }$ </td></tr><tr><td> $\checkmark$ </td><td></td><td></td><td></td><td>42.54</td><td>39.69</td><td>36.64</td></tr><tr><td> $\checkmark$ </td><td> $\checkmark$ </td><td></td><td></td><td>42.62</td><td>39.69</td><td>36.64</td></tr><tr><td> $\checkmark$ </td><td> $\checkmark$ </td><td>V</td><td></td><td>42.74</td><td>39.66</td><td>36.58</td></tr><tr><td></td><td></td><td>V</td><td>V</td><td>56.45</td><td>54.16</td><td>55.26</td></tr><tr><td> $\checkmark$ </td><td></td><td>√</td><td>√</td><td>55.55</td><td>52.51</td><td>54.55</td></tr><tr><td></td><td>√</td><td>√</td><td>V</td><td>55.54</td><td>52.49</td><td>54.59</td></tr><tr><td> $\checkmark$ </td><td> $\checkmark$ </td><td> $\checkmark$ </td><td>V</td><td>56.77</td><td>54.65</td><td>55.78</td></tr><tr><td colspan="5">VisDA-C</td><td>MNIST</td><td>SVHN</td></tr><tr><td> $\mathcal { L } _ { e } ^ { + }$ </td><td> $\mathcal { L } _ { d }$ </td><td> $\mathcal { L } _ { s }$ </td><td>U.P.</td><td> $A C C _ { H }$ </td><td> $A C C _ { H }$ </td><td> $A C C _ { H }$ </td></tr><tr><td>V</td><td></td><td></td><td></td><td>47.07</td><td>52.81</td><td>52.94</td></tr><tr><td>√</td><td> $\checkmark$ </td><td></td><td></td><td>47.08</td><td>52.83</td><td>52.90</td></tr><tr><td>√</td><td>√</td><td>V</td><td></td><td>47.32</td><td>52.95</td><td>52.85</td></tr><tr><td></td><td></td><td>√</td><td>√</td><td>70.39</td><td>63.22</td><td>68.41</td></tr><tr><td>√</td><td></td><td>√</td><td>√</td><td>69.81</td><td>62.68</td><td>67.86</td></tr><tr><td></td><td>√</td><td>√</td><td>√</td><td>69.72</td><td>62.31</td><td>67.83</td></tr><tr><td>√</td><td>√</td><td>√</td><td>√</td><td>71.67</td><td>64.16</td><td>69.63</td></tr></table>

Table 15: Per-sample running time (in seconds) across all datasets, measured with Noise OOD samples.
<table><tr><td>Running Time</td><td>CIFAR10-C</td><td>CIFAR100-C</td><td>ImageNet-C</td><td>ImageNet-R VisDA-C</td><td></td></tr><tr><td>TEST</td><td>0.000623</td><td>0.000626</td><td>0.001139</td><td>0.001188</td><td>0.001165</td></tr><tr><td>BN</td><td>0.000632</td><td>0.000658</td><td>0.001163</td><td>0.001242</td><td>0.001203</td></tr><tr><td>TENT</td><td>0.003121</td><td>0.003208</td><td>0.005682</td><td>0.005821</td><td>0.005640</td></tr><tr><td>SHOT</td><td>0.002406</td><td>0.002496</td><td>0.004015</td><td>0.003910</td><td>0.003733</td></tr><tr><td>OSTTA</td><td>0.001202</td><td>0.001230</td><td>0.002643</td><td>0.002656</td><td>0.003850</td></tr><tr><td>EATA</td><td>0.001726</td><td>0.001830</td><td>0.002824</td><td>0.002860</td><td>0.003195</td></tr><tr><td>RMT</td><td>0.002328</td><td>0.002382</td><td>0.003816</td><td>0.003570</td><td>0.003591</td></tr><tr><td>CoTTA</td><td>0.005638</td><td>0.004163</td><td>0.036937</td><td>0.009987</td><td>0.019757</td></tr><tr><td>UniEnt</td><td>0.001898</td><td>0.001905</td><td>0.003955</td><td>0.004019</td><td>0.003683</td></tr><tr><td>OWT3</td><td>0.003217</td><td>0.003776</td><td>0.006559</td><td>0.006480</td><td>0.005694</td></tr><tr><td>Ours</td><td>0.001978</td><td>0.001872</td><td>0.003270</td><td>0.003493</td><td>0.004059</td></tr></table>

## 4.8 Parameter Analysis

This section explores the model stability with respect to hyper-parameters. Given that identifying the optimal balance parameter λ is still an unresolved issue in unsupervised learning, we adopt a parameter-tuning approach. Figure 6 reports experiments with varying $\lambda \in \{ 1 e - 3 , 1 e - 5 , 1 e - 2 , 5 e - 2 , 1 e - 1 , 5 e - 5 \}$ using the $A C C _ { H }$ metric. We observe that the performance remains stable across diferent parameter values on almost all the open-world benchmarks, with only minor variations. For the VisDA-C dataset, the proposed method also shows improved performance when λ is larger than 5e − 3 for diferent OOD samples. In brief, the proposed method is stable under diferent settings of λ, indicating its robustness.

![](images/3237ce65bd018f91851134e294e94ad519097df80406f26a98e8c20c9dc6ed7c.jpg)  
(a) CIFAR10-C

![](images/9b57b58acab3ce15052a0320d2d4ec3b8f3041f8201975e5f22eee859e0c4d1e.jpg)  
(b) CIFAR100-C

![](images/8c0f8818781720963c17e4edf6215dcfb6e07bd737216434c26f3b808e4271e1.jpg)  
(c) ImageNet-C

![](images/931dd663000b65dff60361261f59ccbaed2aa10791e6295db2914026dffc27f7.jpg)  
(d) ImageNet-R

![](images/137fd1fa8d8c67d5ed8e2880067091d96ca22b0e83590ed8ed809bb0f461bad5.jpg)  
(e) VisDA-C  
Figure 6: Parameter analysis of the optimal balance parameter λ in Eq. (10).

## 4.9 Model Performance across Diferent OOD Sample Ratios

This section validates the robustness of the proposed method against varying numbers of OOD samples. Since the ratio between ID and OOD samples can vary in real-world scenarios, we examine the impact of diferent OOD sample ratios. Specifically, we control the OOD sample ratio $\frac { | \mathcal { S } _ { T _ { O } } | } { | \mathcal { S } _ { T _ { I } } | }$ from 0.2 to 1.0 across all the open-world benchmarks, evaluating their $A C C _ { H }$ values. The results, presented in Figure 7, show that the proposed method maintains consistent performance across diferent OOD sample ratios, demonstrating its applicability to a variety of data ratio scenarios.

![](images/8af876d7e5e0bbcf6f9cc0151684927243ffde0ea853ef14b806eaabd466ff5c.jpg)  
(a) CIFAR10-C

![](images/7b8ac652077643d474b77cc1a31b43f292c03f6898db377a08845d6604c3f164.jpg)  
(b) CIFAR100-C

![](images/15a4c2b73777e76bbd95e44da50a72e04eed42fdc4ef80d1705ec6bfaa2c5437.jpg)  
(c) ImageNet-C

![](images/9f5a830f5100ee8b4e40942c1aa6fb3c5c22723137b6870f26ccb591cf090c3b.jpg)  
(d) ImageNet-R

![](images/389706b4b120679b407df518968b7b29b2dc9fddb07f874c1eae862884b35d97.jpg)  
(e) VisDA-C  
Figure 7: Open-world test-time adaptation results of ReNC at diferent OOD sample ratios.

## 4.10 Evaluation of Efectiveness on Continual OWTTA

In this section, we evaluate the efectiveness of ReNC under the continual OWTTA setting, where data distribution shifts occur and OOD samples continuously change during the adaptation process. Specifically, CIFAR10-C is used as the ID dataset. For the OOD samples, we introduce the Noise, MNIST, SVHN, Tiny ImageNet, and CIFAR100-C datasets, all presented sequentially in an online manner. The comparison results against all baseline methods are presented in Table 16, using $A C C _ { H }$ as the evaluation metric. As shown in the table, the proposed ReNC achieves the highest average performance on the continual CIFAR10- C open-world benchmark, demonstrating its strong adaptability and efectiveness in handling the continual OWTTA scenario.

Table 16: Continual OWTTA results on CIFAR10-C with OOD samples. Best marked in bold, second best underlined.
<table><tr><td>CIFAR10-C</td><td>Noise</td><td>MNIST</td><td>SVHN</td><td>Tiny</td><td>CIFAR100-C</td><td>Average</td></tr><tr><td>TEST</td><td>81.78</td><td>75.10</td><td>74.85</td><td>68.38</td><td>68.07</td><td>73.64</td></tr><tr><td>BN</td><td>47.77</td><td>64.19</td><td>74.18</td><td>71.72</td><td>71.32</td><td>65.83</td></tr><tr><td>TENT</td><td>50.60</td><td>67.37</td><td>76.44</td><td>72.75</td><td>71.76</td><td>67.78</td></tr><tr><td>SHOT</td><td>50.15</td><td>67.74</td><td>77.16</td><td>72.97</td><td>71.89</td><td>67.98</td></tr><tr><td>OSTTA</td><td>21.80</td><td>43.70</td><td>46.89</td><td>48.22</td><td>47.47</td><td>41.62</td></tr><tr><td>EATA</td><td>49.76</td><td>64.36</td><td>74.09</td><td>71.69</td><td>71.34</td><td>66.25</td></tr><tr><td>RMT</td><td>44.90</td><td>88.10</td><td>82.39</td><td>76.32</td><td>76.32</td><td>73.61</td></tr><tr><td>CoTTA</td><td>47.03</td><td>59.90</td><td>73.79</td><td>71.41</td><td>71.01</td><td>64.63</td></tr><tr><td>UniEnt</td><td>47.48</td><td>63.99</td><td>73.94</td><td>71.77</td><td>71.28</td><td>65.69</td></tr><tr><td>OWT3</td><td>81.93</td><td>75.18</td><td>74.94</td><td>68.45</td><td>68.08</td><td>73.72</td></tr><tr><td>Ours</td><td>88.35</td><td>86.60</td><td>88.62</td><td>75.63</td><td>72.11</td><td>82.26</td></tr></table>

## 4.11 Efectiveness with ViT backbone

In this section, we present additional experiments using a ViT [74] backbone. Specifically, we first pretrain a ViT model on the CIFAR10 training set as the source domain model, and then perform OWTTA experiments on the CIFAR10-C dataset with OOD samples. Table 17 summarizes the results in terms of $A C C _ { H }$ As shown, ReNC consistently outperforms all baseline methods, demonstrating its robust generalization capability across diferent backbone architectures.

Table 17: OWTTA results under ViT backbone, evaluated by ACC<sub>H</sub>. Best marked in bold, second best underlined.
<table><tr><td>CIFAR10-C</td><td>Noise</td><td>MNIST</td><td>SVHN</td><td>Tiny</td><td>CIFAR100-C</td></tr><tr><td>TEST</td><td>92.99</td><td>91.04</td><td>90.40</td><td>86.16</td><td>81.60</td></tr><tr><td>BN</td><td>91.81</td><td>90.57</td><td>87.89</td><td>84.80</td><td>80.34</td></tr><tr><td>TENT</td><td>89.15</td><td>92.47</td><td>90.89</td><td>86.23</td><td>82.12</td></tr><tr><td>SHOT</td><td>93.61</td><td>92.37</td><td>90.36</td><td>86.21</td><td>82.11</td></tr><tr><td>OSTTA</td><td>37.21</td><td>44.75</td><td>49.78</td><td>50.36</td><td>50.60</td></tr><tr><td>EATA</td><td>90.26</td><td>89.55</td><td>88.94</td><td>85.55</td><td>81.12</td></tr><tr><td>RMT</td><td>90.49</td><td>90.03</td><td>89.32</td><td>85.85</td><td>81.86</td></tr><tr><td>CoTTA</td><td>93.04</td><td>91.81</td><td>89.90</td><td>83.58</td><td>81.66</td></tr><tr><td>UniEnt</td><td>93.46</td><td>90.12</td><td>85.71</td><td>83.58</td><td>79.65</td></tr><tr><td>OWT3</td><td>93.67</td><td>92.64</td><td>90.94</td><td>86.21</td><td>81.92</td></tr><tr><td>Ours</td><td>94.88</td><td>93.40</td><td>91.61</td><td>87.06</td><td>82.72</td></tr></table>

## 4.12 Vision Language Models in OWTTA

In this section, we explore the applicability of Vision-Language Models (VLMs) [38] to OWTTA. While VLMs are well known for their strong zero-shot capabilities, they lack native OOD detection mechanisms and thus cannot be directly applied to the OWTTA setting. To address this limitation, we extend C-TPT [39], a CLIP-based test-time adaptation method, by incorporating a pre-trained ResNet-50 as the OOD detection module, resulting in C-TPT+.

As shown in Table 18, C-TPT+ ranks second on the ImageNet-R open-world benchmark, demonstrating its strong generalization ability. This result highlights the potential of VLM-based approaches in OWTTA. Notably, our ReNC is specifically designed for this task, with tailored components that support more efective

Table 18: Comparison of vision-language model extensions on ImageNet-R with OOD samples, evaluated by $A C C _ { H }$ . Best marked in bold, second best underlined.
<table><tr><td>ImageNet-R</td><td>Noise</td><td>MNIST</td><td>SVHN</td></tr><tr><td>TEST</td><td>39.63</td><td>38.17</td><td>36.61</td></tr><tr><td>BN</td><td>27.70</td><td>30.52</td><td>31.17</td></tr><tr><td>TENT</td><td>28.57</td><td>32.11</td><td>32.71</td></tr><tr><td>SHOT</td><td>29.28</td><td>31.72</td><td>33.29</td></tr><tr><td>OSTTA</td><td>35.50</td><td>30.93</td><td>35.58</td></tr><tr><td>EATA</td><td>27.55</td><td>31.07</td><td>31.21</td></tr><tr><td>RMT</td><td>16.02</td><td>21.70</td><td>27.02</td></tr><tr><td>CoTTA</td><td>26.35</td><td>28.98</td><td>28.92</td></tr><tr><td>UniEnt</td><td>18.43</td><td>25.26</td><td>26.64</td></tr><tr><td>OWT3</td><td>28.07</td><td>31.15</td><td>31.75</td></tr><tr><td>C-TPT+</td><td>45.39</td><td>42.40</td><td>40.38</td></tr><tr><td>Ours</td><td>56.77</td><td>54.65</td><td>55.78</td></tr></table>

learning and adaptation. As a result, it achieves the best overall performance. These findings suggest that while VLMs ofer impressive zero-shot and generalization capabilities, they still require task-specific adaptation to perform reliably in specialized scenarios like OWTTA.

## 4.13 Friedman Test for Significant Test

In this section, we conduct a significant test to evaluate the performance variations among the eight methods. Specifically, we utilize the Friedman test [75] under the assumption of equal performance across all methods as the null hypothesis. Pairwise comparisons are subsequently carried out using the Nemenyi post-hoc test [76]. The dataset for this evaluation consists of 627 instances, with each method independently assessed across nineteen datasets using three evaluation metrics. The Friedman test yields a test statistic of $\tau _ { F } = 1 8 . 2 6 6 .$ which surpasses the critical value of $F _ { 1 0 , 5 6 0 } = 1 . 8 4 7$ at a significance level of $\alpha = 0 . 0 5$ . Thus, the null hypothesis is rejected, confirming that there are statistically significant diferences among the eight methods at a significance level of 0.05. Further examination via the Nemenyi post-hoc test (See Figure 8) shows that our ReNC method demonstrates a clear performance advantage over the seven baseline methods.

![](images/fcd9564f47bcdc765305a93bd421488ae9bc7b45ec3c5b5e3002e5cedaa68ced.jpg)  
Figure 8: Nemenyi post-hoc analysis of all the baselines.

## 5 Conclusion

This paper introduces Reliable Neural Collapse approximation (ReNC), a method designed to address OWTTA reliably and efectively. We justify that pre-trained classifier weights can be viewed as prototypes within the context of neural collapse and develop a reliable update mechanism to filter out Out-Of-Distribution (OOD) samples during adaptation. Furthermore, we propose a neural collapse approximation mechanism for updating prototypes, enabling the model to gradually approach the neural collapse state in the target domain while maintaining the balanced structure. Our empirical results validate the efectiveness of ReNC and suggest that NC-related properties may provide useful evidence for understanding its improved classification performance. In future work, we will explore the potential of foundation models for the OWTTA.

## References

[1] S. Ben-David, J. Blitzer, K. Crammer, and F. Pereira, “Analysis of representations for domain adaptation,” in NeurIPS, 2006, pp. 137–144.

[2] S. Ben-David, J. Blitzer, K. Crammer, A. Kulesza, F. Pereira, and J. W. Vaughan, “A theory of learning from diferent domains,” Mach. Learn., vol. 79, no. 1-2, pp. 151–175, 2010.

[3] K. You, M. Long, Z. Cao, J. Wang, and M. I. Jordan, “Universal domain adaptation,” in CVPR, 2019, pp. 2720–2729.

[4] M. Boudiaf, R. M¨uller, I. B. Ayed, and L. Bertinetto, “Parameter-free online test-time adaptation,” in CVPR, 2022, pp. 8334–8343.

[5] S. Niu, J. Wu, Y. Zhang, Y. Chen, S. Zheng, P. Zhao, and M. Tan, “Eficient test-time model adaptation without forgetting,” in ICML, vol. 162, 2022, pp. 16 888–16 905.

[6] Z. Chi, Y. Wang, Y. Yu, and J. Tang, “Test-time fast adaptation for dynamic scene deblurring via meta-auxiliary learning,” in CVPR, 2021, pp. 9137–9146.

[7] S. Goyal, M. Sun, A. Raghunathan, and J. Z. Kolter, “Test time adaptation via conjugate pseudolabels,” in NeurIPS, 2022.

[8] Y. Li, X. Xu, Y. Su, and K. Jia, “On the robustness of open-world test-time training: Self-training with dynamic prototype expansion,” in ICCV, 2023, pp. 11 802–11 812.

[9] M. Yuan, Y. Xia, H. Dong, Z. Chen, J. Yao, M. Qiu, K. Yan, X. Yin, Y. Shi, X. Chen, Z. Liu, B. Dong, J. Zhou, L. Lu, L. Zhang, and L. Zhang, “Devil is in the queries: Advancing mask transformers for real-world medical image segmentation and out-of-distribution localization,” in CVPR, 2023, pp. 23 879–23 889.

[10] J. Lee, D. Das, J. Choo, and S. Choi, “Towards open-set test-time adaptation utilizing the wisdom of crowds in entropy minimization,” in ICCV, 2023, p. 16334.

[11] Z. Gao, X.-Y. Zhang, and C.-L. Liu, “Unified entropy optimization for open-set test-time adaptation,” in CVPR, 2024, pp. 23 975–23 984.

[12] J. Liang, D. Hu, and J. Feng, “Do we really need to access the source data? source hypothesis transfer for unsupervised domain adaptation,” in ICML, vol. 119, 2020, pp. 6028–6039.

[13] J. N. Kundu, N. Venkat, R. M. V., and R. V. Babu, “Universal source-free domain adaptation,” in CVPR, 2020, pp. 4543–4552.

[14] S. Liu, D. Zhang, and X. Hao, “Eficient deformable convolutional prompt for continual test-time adaptation in medical image segmentation,” in AAAI, 2025, pp. 5550–5557.

[15] R. Li, Q. Jiao, W. Cao, H. Wong, and S. Wu, “Model adaptation: Unsupervised domain adaptation without source data,” in CVPR, 2020, pp. 9638–9647.

[16] Z. Feng, C. Xu, and D. Tao, “Open-set hypothesis transfer with semantic consistency,” IEEE Trans. Image Process., vol. 30, pp. 6473–6484, 2021.

[17] J. Zhou, C. You, X. Li, K. Liu, S. Liu, Q. Qu, and Z. Zhu, “Are all losses created equal: A neural collapse perspective,” in NeurIPS, 2022.

[18] Y.-W. Luo, C.-X. Ren, X.-L. Xu, and Q. Liu, “Geometric understanding of discriminability and trans ferability for visual domain adaptation,” IEEE Trans. Pattern Anal. Mach. Intell., pp. 1–16, 2024.

[19] F. Huang, S. Song, and L. Zhang, “Gradient harmonization in unsupervised domain adaptation,” IEEE Trans. Pattern Anal. Mach. Intell., pp. 1–17, 2024.

[20] Y. Wu, Z. Chi, Y. Wang, K. N. Plataniotis, and S. Feng, “Test-time domain adaptation by learning domain-aware batch normalization,” in AAAI, 2024, pp. 15 961–15 969.

[21] Y. Su, X. Xu, and K. Jia, “Towards real-world test-time adaptation: Tri-net self-training with balanced normalization,” in AAAI, 2024, pp. 15 126–15 135.

[22] Y. Gandelsman, Y. Sun, X. Chen, and A. A. Efros, “Test-time training with masked autoencoders,” in NeurIPS, 2022.

[23] D. Wang, E. Shelhamer, S. Liu, B. A. Olshausen, and T. Darrell, “Tent: Fully test-time adaptation by entropy minimization,” in ICLR, 2021.

[24] Y. Su, X. Xu, and K. Jia, “Revisiting realistic test-time training: Sequential inference and adaptation by anchored clustering,” in NeurIPS, 2022.

[25] T. Kojima, Y. Matsuo, and Y. Iwasawa, “Robustifying vision transformer without retraining from scratch by test-time class-conditional feature alignment,” in IJCAI, 2022, pp. 1009–1016.

[26] M. J. Mirza, P. Jan´e-Soneira, W. Lin, M. Kozinski, H. Possegger, and H. Bischof, “Actmad: Activation matching to align distributions for test-time-training,” in CVPR, 2023, pp. 24 152–24 161.

[27] Y. Liu, P. Kothari, B. van Delft, B. Bellot-Gurlet, T. Mordan, and A. Alahi, “TTT++: when does self-supervised test-time training fail or thrive?” in NeurIPS, 2021, pp. 21 808–21 820.

[28] Y. Su, X. Xu, T. Li, and K. Jia, “Revisiting realistic test-time training: Sequential inference and adaptation by anchored clustering regularized self-training,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 46, no. 8, pp. 5524–5540, 2024.

[29] N. Hansen, R. Jangir, Y. Sun, G. Aleny\`a, P. Abbeel, A. A. Efros, L. Pinto, and X. Wang, “Selfsupervised policy adaptation during deployment,” in ICLR, 2021.

[30] D. Chen, D. Wang, T. Darrell, and S. Ebrahimi, “Contrastive test-time adaptation,” in CVPR, 2022, pp. 295–305.

[31] Y. Sun, X. Wang, Z. Liu, J. Miller, A. A. Efros, and M. Hardt, “Test-time training with self-supervision for generalization under distribution shifts,” in ICML, vol. 119, 2020, pp. 9229–9248.

[32] Q. Wang, O. Fink, L. V. Gool, and D. Dai, “Continual test-time domain adaptation,” in CVPR, 2022, pp. 7191–7201.

[33] S. Niu, J. Wu, Y. Zhang, Z. Wen, Y. Chen, P. Zhao, and M. Tan, “Towards stable test-time adaptation in dynamic wild world,” in ICLR, 2023.

[34] T. Gong, J. Jeong, T. Kim, Y. Kim, J. Shin, and S. Lee, “NOTE: robust continual test-time adaptation against temporal correlation,” in NeurIPS, 2022.

[35] M. D¨obler, R. A. Marsden, and B. Yang, “Robust mean teacher for continual and gradual test-time adaptation,” in CVPR, 2023, pp. 7704–7714.

[36] Z. Zhou, L. Guo, L. Jia, D. Zhang, and Y. Li, “ODS: test-time adaptation in the presence of open-world data shift,” in ICML, vol. 202, 2023, pp. 42 574–42 588.

[37] Z. Zhou, K. Yu, L. Guo, and Y. Li, “Fully test-time adaptation for tabular data,” in AAAI, 2025, pp. 23 027–23 035.

[38] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, G. Krueger, and I. Sutskever, “Learning transferable visual models from natural language supervision,” in ICML, vol. 139, 2021, pp. 8748–8763.

[39] H. S. Yoon, E. Yoon, J. T. J. Tee, M. A. Hasegawa-Johnson, Y. Li, and C. D. Yoo, “C-TPT: calibrated test-time prompt tuning for vision-language models via text feature dispersion,” in ICLR, 2024.

[40] Y. Qian, Y. Bai, Z. Zhang, P. Zhao, and Z. Zhou, “Handling new class in online label shift,” in ICDM, 2023, pp. 1283–1288.

[41] D. Hendrycks and K. Gimpel, “A baseline for detecting misclassified and out-of-distribution examples in neural networks,” in ICLR, 2017.

[42] W. Liu, X. Wang, J. D. Owens, and Y. Li, “Energy-based out-of-distribution detection,” in NeurIPS, 2020.

[43] Y. Sun, C. Guo, and Y. Li, “React: Out-of-distribution detection with rectified activations,” in NeurIPS, 2021, pp. 144–157.

[44] Y. Sun, Y. Ming, X. Zhu, and Y. Li, “Out-of-distribution detection with deep nearest neighbors,” in ICML, vol. 162, 2022, pp. 20 827–20 840.

[45] C. S. Sastry and S. Oore, “Detecting out-of-distribution examples with gram matrices,” in ICML, vol. 119, 2020, pp. 8491–8501.

[46] H. Wang, Z. Li, L. Feng, and W. Zhang, “Vim: Out-of-distribution with virtual-logit matching,” in CVPR, 2022, pp. 4911–4920.

[47] R. Huang, A. Geng, and Y. Li, “On the importance of gradients for detecting distributional shifts in the wild,” in NeurIPS, 2021, pp. 677–689.

[48] C. Igoe, Y. Chung, I. Char, and J. Schneider, “How useful are gradients for OOD detection really?” CoRR, vol. abs/2205.10439, 2022.

[49] Y. Li and N. Vasconcelos, “Background data resampling for outlier-aware classification,” in CVPR, 2020, pp. 13 215–13 224.

[50] Y. Ming, Y. Fan, and Y. Li, “POEM: out-of-distribution detection with posterior sampling,” in ICML, vol. 162, 2022, pp. 15 650–15 665.

[51] S. Vernekar, A. Gaurav, V. Abdelzad, T. Denouden, R. Salay, and K. Czarnecki, “Out-of-distribution detection in classifiers via generation,” CoRR, vol. abs/1910.04241, 2019.

[52] K. Lee, H. Lee, K. Lee, and J. Shin, “Training confidence-calibrated classifiers for detecting out-of distribution samples,” in ICLR, 2018.

[53] X. Du, Z. Wang, M. Cai, and Y. Li, “VOS: learning what you don’t know by virtual outlier synthesis,” in ICLR, 2022.

[54] W. Chang, Y. Shi, H. Tuan, and J. Wang, “Unified optimal transport framework for universal domain adaptation,” in NeurIPS, 2022.

[55] G. Li, G. Kang, Y. Zhu, Y. Wei, and Y. Yang, “Domain consensus clustering for universal domain adaptation,” in CVPR, 2021, pp. 9757–9766.

[56] K. Saito, D. Kim, S. Sclarof, and K. Saenko, “Universal domain adaptation through self supervision,” in NeurIPS, 2020.

[57] S. Qu, T. Zou, F. R¨ohrbein, C. Lu, G. Chen, D. Tao, and C. Jiang, “Upcycling models under domain and category shift,” in CVPR, 2023, pp. 20 019–20 028.

[58] J. Liang, D. Hu, J. Feng, and R. He, “UMAD: universal model adaptation under domain and category shift,” CoRR, vol. abs/2112.08553, 2021.

[59] S. Qu, T. Zou, L. He, F. R¨ohrbein, A. Knoll, G. Chen, and C. Jiang, “LEAD: learning decomposition for source-free universal domain adaptation,” in CVPR, 2024, pp. 23 334–23 343.

[60] D. Shanmugam, D. W. Blalock, G. Balakrishnan, and J. V. Guttag, “Better aggregation in test-time augmentation,” in ICCV, 2021, pp. 1194–1203.

[61] I. Kim, Y. Kim, and S. Kim, “Learning loss for test-time augmentation,” in NeurIPS, 2020.

[62] A. Jelea, A. N. Belbachir, and M. Leordeanu, “Learning from random subspace exploration: Generalized test-time augmentation with self-supervised distillation,” CoRR, 2025.

[63] M. I. Belghazi, A. Baratin, S. Rajeswar, S. Ozair, Y. Bengio, R. D. Hjelm, and A. C. Courville, “Mutual information neural estimation,” in ICML, vol. 80, 2018, pp. 530–539.

[64] J. Snell, K. Swersky, and R. S. Zemel, “Prototypical networks for few-shot learning,” in NeurIPS, 2017, pp. 4077–4087.

[65] Z. Zhu, T. Ding, J. Zhou, X. Li, C. You, J. Sulam, and Q. Qu, “A geometric analysis of neural collapse with unconstrained features,” in NeurIPS, 2021, pp. 29 820–29 834.

[66] S. Xie, Z. Zheng, L. Chen, and C. Chen, “Learning semantic representations for unsupervised domain adaptation,” in ICML, vol. 80, 2018, pp. 5419–5428.

[67] D. Hendrycks and T. G. Dietterich, “Benchmarking neural network robustness to common corruptions and perturbations,” in ICLR, 2019.

[68] D. Hendrycks, S. Basart, N. Mu, S. Kadavath, F. Wang, E. Dorundo, R. Desai, T. Zhu, S. Parajuli, M. Guo, D. Song, J. Steinhardt, and J. Gilmer, “The many faces of robustness: A critical analysis of out-of-distribution generalization,” in ICCV, 2021, pp. 8320–8329.

[69] X. Peng, B. Usman, N. Kaushik, J. Hofman, D. Wang, and K. Saenko, “Visda: The visual domain adaptation challenge,” CoRR, 2017.

[70] L. Deng, “The MNIST database of handwritten digit images for machine learning research [best of the web],” IEEE Signal Process. Mag., vol. 29, no. 6, pp. 141–142, 2012.

[71] Y. Netzer, T. Wang, A. Coates, A. Bissacco, B. Wu, A. Y. Ng et al., “Reading digits in natural images with unsupervised feature learning,” in NIPS workshop on deep learning and unsupervised feature learning, vol. 2011, no. 5, 2011, p. 7.

[72] Y. Le and X. Yang, “Tiny imagenet visual recognition challenge,” CS 231N, vol. 7, no. 7, p. 3, 2015.

[73] S. Iofe and C. Szegedy, “Batch normalization: Accelerating deep network training by reducing internal covariate shift,” in ICML, vol. 37, 2015, pp. 448–456.

[74] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, J. Uszkoreit, and N. Houlsby, “An image is worth 16x16 words: Transformers for image recognition at scale,” in ICLR, 2021.

[75] J. Demsar, “Statistical comparisons of classifiers over multiple data sets,” J. Mach. Learn. Res., vol. 7, pp. 1–30, 2006.

[76] P. B. Nemenyi, Distribution-free multiple comparisons. Princeton University, 1963.