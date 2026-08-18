Data-Eficient and Interpretable Classification of Circulating Tumor Cell Phenotypes in Microfluidic Devices via Deep Learning

Serena Su, Yifan Wang, Senwei Liang

• We proposed a data eficient method called SubSeq argumentation for Circulating Tumor Cell Classification.

• We developed a Grad-CAM based analysis to identify phenotype-specific trajectory regions inside the microfluidic channel and to reveal interactions between cell, microposts and fluid that drive classification.

• Our analysis revealed that the cell velocity captures the primary predictive signals, while position trajectory provides complementary information. Combining the two information yields the highest classification performance.

• We found that classification relies primarily on localized trajectory segments rather than complete cell trajectories, suggesting that targeted segment analysis could reduce imaging and computational demands in future laboratory experiments.

# Data-Eficient and Interpretable Classification of Circulating Tumor Cell Phenotypes in Microfluidic Devices via Deep Learning<sup>⋆</sup>

Serena Su<sup>a</sup>, Yifan Wang<sup>a,∗</sup> and Senwei Liang<sup>a,∗</sup>

<sup>a</sup>Department ofMathematics & Statistic, Texas Tech University, Lubbock, 79407, Texas, USA

A R T I C L E I N F O

Keywords:   
Circulating Tumor Cell   
Microfluidic Device   
Trajectory-based Classification   
Data Augmentation   
Interpretability

## A BS T R AC T

Accurate classification of circulating tumor cell (CTC) phenotypes can provide valuable information for assessing metastatic potential. Label-free microfluidic devices provide a hydrodynamic obstacle course that transforms subtle biophysical characteristics of CTCs, including size and deformability, into distinct kinematic trajectories. However, the highly nonlinear fluid–structure interactions governing these trajectories make the inverse problem of inferring cellular phenotype from trajectory data analytically intractable. While deep neural networks (DNNs) have emerged as a powerful approach for addressing this inverse problem, their efectiveness is constrained by the limited availability of trajectory data and the lack of physical interpretability inherent to their black-box nature.

To address these challenges, we propose an interpretable and data-eficient DNN framework for trajectory-based CTC classification. To mitigate the scarcity of data, we develop Subsequence (SubSeq), a targeted augmentation strategy that randomly extracts informative local trajectory segments during training to promote learning from localized patterns. To interpret the trained DNN, we further apply Gradient-weighted Class Activation Mapping (Selvaraju et al. 2017) to identify the trajectory features and physical regions of the microfluidic device that drive model predictions. Experimental results demonstrate that SubSeq improves classification accuracy over the evaluated baseline and augmentation methods. Furthermore, interpretability analysis suggests that localized trajectory segments contain substantial biophysical information relevant to accurate classification. This provides justification for SubSeq and also highlights the redundancy of full-length trajectories. More broadly, the proposed framework views microfluidic geometries as physical encoders of cellular mechanical properties, providing mechanistic insights that may inform the future design of diagnostic devices.

## 1. Introduction

Circulating tumor cells (CTCs) are malignant cells shed from primary or metastatic tumors into the bloodstream and are widely recognized as important mediators of cancer metastasis. Detecting and classifying CTCs provide valuable insights into cancer diagnosis, prognosis, treatment response, and recurrence risk (Pantel, Brakenhof and Brandt, 2008; Alix-Panabières and Pantel, 2014), making CTC analysis central to the development of liquid biopsy technologies. Beyond their mere detection in blood, comprehensive classification of CTC phenotypes is essential. CTCs exhibit substantial phenotypic heterogeneity in size, morphology, deformability, surface marker expression, and mechanical rigidity. These characteristics often reflect clinically relevant biological states, including epithelial–mesenchymal transition, metastatic potential, and therapeutic resistance (Yu, Bardia, Wittner, Stott, Smas, Ting, Isakof, Ciciliano, Wells, Shah, Concannon, Donaldson, Sequist, Brachtel, Sgroi, Baselga, Ramaswamy, Toner, Haber and Maheswaran, 2013; Joosse, Gorges and Pantel, 2015; Micalizzi, Maheswaran and Haber, 2017).

Traditional biochemical labeling methods can fail to classify CTC subpopulations that lack the targeted surface markers or undergo phenotypic changes during disease progression (Micalizzi et al., 2017). To overcome this limitation, label-free microfluidic devices have emerged as a powerful physics-based alternative that exploits the intrinsic biophysical properties of cells (Di Carlo, 2009; Whitesides, 2006; Chen, Li, Huang, Xie, Mai, Wang, Nguyen and Huang, 2014; Warkiani, Tay, Guan and Han, 2015). In these systems, cells are transported through microstructured channels (as shown in Figure 1a), such as arrays of stationary microposts, where they undergo continuous interactions with both the surrounding fluid and the channel structures. As a cell navigates through these microstructures, its intrinsic biophysical properties, such as size and deformability, govern its hydrodynamic response, leaving a distinctive signature on its trajectory. However, because the distributions of these properties often overlap across diferent cell phenotypes, the resulting trajectories exhibit highly nonlinear and coupled dynamics. Consequently, these complex motion patterns are dificult to describe using conventional analytical models or to classify with simple threshold-based sorting strategies.

![](images/b354833522f65e4ce60038909c06f0dc6fe2fe63eaa7c2a395faf8779cb95039.jpg)

![](images/efd91cfd02eab6604399f9b45a68099ee86521f02be2823a9fe8a1ed092a704d.jpg)  
Figure 1: Subfigure I. The hyperuniform microfluidic device is shown in (a), with a schematic illustration of the micropost arrangement presented in (b). Subfigure II. Experimentally observed trajectories of PC3 and SKBR3 cancer cells are shown in an overall view in (a) and in two enlarged views in (b) and (c), corresponding to the regions indicated by the labeled black boxes (Images courtesy of the Wei Li laboratory). PC3 cells are shown in green, whereas SKBR3 cells are shown in blue. Subfigure III. The numerical simulation demonstrates distinct trajectories for two simulated circulating tumor cell types, shown in red and yellow, with diferent elastic properties. Both cells are released from the same initial location (Rejuan et al., 2025).

Recent studies (Rejuan et al., 2025; Kumar, Wang, Zhan, Gardner, Thompson, Li and Canic, 2025) have leveraged deep neural network (DNN)-based methods to validate the efectiveness of the hyperuniform microfluidic device using numerically simulated cell trajectories for CTC phenotype classification. Figure 1. I (a) and (b) show the hyperuniform microfluidic device and its micropost arrangement, which is disordered yet statistically uniform. Subfigures in Figure 1. II presents experimentally observed trajectories of PC3 and SKBR3 cancer cells, shown in green and blue, respectively. The overall and enlarged views indicate that diferent cell types can follow distinct paths through the micropost array. Building on these experimental observations, the aforementioned studies employed fluid– structure interaction simulations to systematically investigate phenotype-dependent interactions among cells, fluid flow, and microstructures. As shown in Figure 1. III, two CTCs with diferent elastic properties, despite being released from the same initial location, develop distinct trajectories through their interactions with the surrounding fluid and microposts. Given that numerical simulations provide controlled access to these coupled interactions, which can be dificult to isolate and measure experimentally at suficient spatial and temporal resolution, the aforementioned studies used simulated trajectories to systematically characterize phenotype-dependent cell behavior. By combining these trajectories with advanced deep learning architectures, including convolutional neural networks (CNNs) and recurrent neural networks (RNNs), they demonstrated that cell trajectories contain suficient phenotype-specific information to support CTC classification. This prior work highlights the potential of hyperuniform physical architectures to translate subtle diferences in cellular mechanical properties into measurable and distinguishable trajectory signatures.

Motivated by these promising results, we seek to further improve the existing CTC classification approach by addressing two key limitations. (I): Because full fluid–structure interaction simulations are computationally expensive, the available trajectory dataset remains relatively small, which may limit model generalization. (II): These DNN models operate as black boxes, making their internal decision-making processes notoriously dificult to interpret. This lack of transparency poses a limitation in biomedical applications, where predictions must be grounded in verifiable physical mechanisms (Rudin, 2019). Moreover, limited interpretability potentially restricts physical insight into how trajectory patterns relate to device geometry, thereby hindering the rational design of improved microfluidic architectures.

In this study, we introduce an data-eficient and interpretable deep learning framework for CTC phenotype classification. To alleviate the limitation of scarce simulation data, we propose a novel data augmentation strategy, termed Subsequence (SubSeq) sampling, which randomly extracts continuous segments from full trajectories during training. This simple yet efective strategy significantly improves classification performance, outperforming alternative methods. Next, we apply Gradient-weighted Class Activation Mapping (Grad-CAM) to project the model’s classification decisions back onto the physical trajectory space (Selvaraju, Cogswell, Das, Vedantam, Parikh and Batra, 2017). Interestingly, the interpretability analysis shows that the trained models do not rely on the entire trajectory for accurate prediction; instead, they concentrate on specific, highly discriminative segments within the hyperuniform device. This observation provides a physical justification for the efectiveness of SubSeq, suggesting that localized cell– flow–structure interactions contain suficient biophysical information for phenotype discrimination, while substantial portions of the full trajectories exhibit dynamic redundancy.

The proposed framework integrates microfluidic design, trajectory generation, and interpretable learning within an encoder–decoder perspective. The hyperuniform micropost array can be viewed as a physical encoder that transforms diferences in CTC mechanical properties into distinct trajectory signatures through cell interactions with the device geometry and surrounding fluid. The deep learning model then acts as a decoder that uses these trajectories to classify CTC phenotypes, while Grad-CAM maps the model’s learned importance scores back onto the microfluidic geometry. This analysis improves classification performance under limited-data conditions and identifies regions of the device that may contribute to phenotype discrimination. Although further physical and experimental validation is required, these findings provide a foundation for investigating how interpretable learning could inform the future optimization of microfluidic device architectures. The main contributions of this work are summarized as follows:

• We propose a trajectory-based deep learning framework with a Subsequence (SubSeq) sampling strategy and integrated Grad-CAM analysis, enabling efective learning from limited simulation data while providing interpretable mappings from trajectory segments to model predictions.

• We show that the cell velocity captures the primary predictive signals, while the position trajectory provides complementary information. Combining the two pieces of information yields the highest classification performance.

• We demonstrate that localized cell–flow–structure interactions within a hyperuniform microfluidic device contain suficient biophysical information for CTC phenotype classification, revealing that full trajectories exhibit significant redundancy and providing mechanistic insight for the rational design of microfluidic architectures.

The remainder of this paper is organized as follows. Section 2 reviews related works. Section 3 introduces the problem setup for CTC classification. Section 4 presents the proposed SubSeq sampling and Grad-CAM-based interpretability method. Section 5 reports experimental results, and in the end, Section 6 concludes the paper.

## 2. Related Work

## 2.1. In Silico Microfluidic CTC Classification

Integrating physical modeling, rational device design, and phenotypic profiling provides a clear pathway toward label-free microfluidic platforms that both detect rare cells and assess their biological significance. Exploring these frontiers, Tan et al. developed a simulation framework for CTC transport and adhesion, demonstrating how flow conditions, device geometry, ligand density, and cell properties govern capture dynamics (Tan, Ding, Hood and Li, 2019). To optimize this capture mechanically, Wang and Li introduced a novel device featuring hyperuniform micropost arrangements; this structured, heterogeneous geometry creates unique cell–flow interactions that enable labelfree detection and trajectory-based classification (Wang and Li, 2024). Expanding on these analytical capabilities,

Joshi et al. reviewed how microfluidic platforms are moving beyond simple isolation to characterize clinically relevant phenotypes, such as cell size, deformability, adhesion, metabolism, and surface markers (Joshi, Ahmadi, Gardner, Bright, Wang and Li, 2025).

## 2.2. Data Augmentation for DNN

Data augmentation improves machine learning model generalization by artificially expanding training set diversity. It is important in data-scarce environments, where it mitigates overfitting by exposing models to a broader spectrum of input variations. In computer vision, standard image augmentations include geometric transformations (e.g., rotation, flipping, translation (He, Zhang, Ren and Sun, 2016)) and pixel-level perturbations like noise injection. Recently, advanced regularization techniques such as Cutout (Devries and Taylor, 2017) and Mixup (Zhang, Cissé, Dauphin and Lopez-Paz, 2017; Lin, Huang, Wang, Liu and Lin, 2024), alongside automated search strategies like AutoAugment (Cubuk, Zoph, Mane, Vasudevan and Le, 2019) and data engine (Liang, Su, Schulter, Garg, Zhao, Wu and Chandraker, 2024), have further advanced visual representation learning.

Migrating image augmentations to time-series data requires accounting for strict temporal dependencies, as arbitrary transformations can easily distort sequential dynamics and generate invalid samples. Consequently, specialized techniques are categorized into time-domain, frequency-domain (Gao, Song, Wen, Wang, Sun and Xu, 2020), and advanced learning-based approaches (Wen, Sun, Yang, Song, Gao, Wang and Xu, 2021). Within the time domain, methods manipulate raw sequences directly: window cropping extracts local segments to multiply samples (Chen and Shi, 2021), flipping reverses numerical signs for symmetric datasets, window warping alters local dynamics via temporal compression or extension (Wen et al., 2021), and noise injection adds minor variations without altering underlying semantic labels (Wen and Keyes, 2019).

## 2.3. DNN Interpretability

Understanding the decision-making process of DNNs is an important aspect of model evaluation, especially in domains such as medical diagnosis (Huang, Liang and Liang, 2025). In computer vision, Class Activation Maps (CAMs) are widely used to explain model predictions by highlighting image regions that contribute most to a target class. The original CAM method proposed by (Zhou, Khosla, Lapedriza Garcia, Oliva and Torralba, 2016) generates class-specific activation maps by linearly combining the final convolutional feature maps with class-specific weights. Since then, numerous CAM variants have been developed, broadly categorized into gradient-free methods (e.g., CAM, Score-CAM, Recipro-CAM) and gradient-based methods (e.g., Layer-CAM, Grad-CAM) (Minh, 2023). Among these, Gradient-weighted CAM (Grad-CAM) (Selvaraju, Das, Vedantam, Cogswell, Parikh and Batra, 2016) is one of the most widely adopted approaches. It has also been extended to time-series analysis, where it identifies important temporal segments, with successful applications in national demand forecasting (van Zyl, Ye and Naidoo, 2024) and manufacturing anomaly detection (Hyun, Yoo, Kim, Lee and Kim, 2024).

## 3. Preliminary

This section introduces the trajectory-based classification task for circulating tumor cell phenotypes and the convolutional neural network (CNN) used in the proposed framework.

## 3.1. CTC Trajectory and Phenotypes

Cell trajectories in the hyperuniform microfluidic device were generated using numerical simulations of a coupled fluid–structure interaction model (Rejuan et al., 2025). The surrounding blood plasma was modeled as an incompressible Newtonian fluid governed by the Navier–Stokes equations and solved using the lattice Boltzmann method. Each CTC membrane was represented by a spring-network model whose deformation was computed using the finite element method. The spring-network accounts for stretching, bending, and surface area and volume conservation efects, with its dynamics coupled to the fluid through immersed boundary method.

The dataset consists of � trajectory samples $\boldsymbol { D } = \{ ( \mathbf { X } ^ { ( i ) } , p ^ { ( i ) } ) \} _ { i = 1 } ^ { N }$ , where $\mathbf { X } ^ { ( i ) } = \left\{ ( x _ { t } ^ { ( i ) } , y _ { t } ^ { ( i ) } , z _ { t } ^ { ( i ) } , v _ { x , t } ^ { ( i ) } , v _ { y , t } ^ { ( i ) } , v _ { z , t } ^ { ( i ) } ) \right\} _ { t = 1 } ^ { T }$ represents the position (first three coordinates) and velocity (last three coordinates) time series of the �-th cell, and $p ^ { ( i ) } \in \{ 0 , 1 \}$ denotes its phenotype label, corresponding to soft and hard CTC types, respectively. The trajectory-based CTC phenotype classification task aims to learn a classifier that maps $\mathbf { X } ^ { ( i ) }$ to its corresponding CTC phenotype.

![](images/3733a0a92481368866599ec50f1c8001a0b99ace0ef26b430df3d195ef675b9c.jpg)  
Figure 2: Convolutional neural network for CTC classification (Kumar et al., 2025). The model processes the position and velocity inputs through parallel branches, concatenates the extracted features, and finally performs classification using a multilayer perceptron to produce the final phenotype classification.

However, generating cell trajectories is computationally demanding because each trajectory requires a complete time-dependent fluid–structure interaction simulation. As a result, only a limited number of training samples are available, motivating the use of efective data augmentation strategies to improve the generalization of DNN models.

## 3.2. CNN for CTC Classification

CNNs have demonstrated promising performance in trajectory-based CTC phenotype classification (Kumar et al., 2025). In this study, we adopt the CNN architecture illustrated in Figure 2 as the primary testbed for evaluating the proposed framework. Specifically, we use this architecture to quantify the efect of data augmentation on classification performance relative to the baseline training procedure and apply Grad-CAM to interpret the features that influence the trained model’s predictions. Although the present analysis focuses on this CNN architecture, the proposed framework can also be applied to other neural network models for sequential data.

As shown in Figure 2, the input trajectory is split into two sequences: the positional sequence $( x _ { t } , y _ { t } , z _ { t } ) _ { t = 1 } ^ { T }$ and the velocity sequence $( v _ { x , t } , v _ { y , t } , v _ { z , t } ) _ { t = 1 } ^ { T }$ . These two sequences are processed by two parallel branches with the same architecture. Each branch takes a three-channel time series as input and applies a one-dimensional convolutional layer with 256 filters of kernel size 3, followed by a ReLU activation and a max-pooling layer with kernel size 2. A second one-dimensional convolutional layer with 256 filters and kernel size 3 is then applied, again followed by a ReLU activation. Adaptive average pooling subsequently aggregates the temporal dimension into a fixed-length 64- dimensional feature vector. Finally, the feature vectors from the position and velocity branches are concatenated and fed into a multilayer perceptron to produce the final phenotype classification.

## 4. Proposed Methods for Improving and Interpreting CTC Classification

This section introduces the proposed SubSeq sampling method, a data augmentation technique designed to improve CTC classification performance, and Grad-CAM analysis, which is used to interpret the trained models and identify salient features of CTC trajectories.

## 4.1. Subsequence (SubSeq) Augmentation

Inspired by the intuition behind Cutout (Devries and Taylor, 2017), SubSeq intentionally limits the trajectory information available to the classifier during training. By exposing only a randomly selected temporal segment, the model is encouraged to learn discriminative local motion patterns rather than relying on complete trajectories.

Given a trajectory sample $\mathbf { X } ^ { ( i ) } = \left\{ ( x _ { t } ^ { ( i ) } , y _ { t } ^ { ( i ) } , z _ { t } ^ { ( i ) } , v _ { x , t } ^ { ( i ) } , v _ { y , t } ^ { ( i ) } , v _ { z , t } ^ { ( i ) } ) \right\} _ { t = 1 } ^ { T }$ , the proposed SubSeq method generates an augmented training input sample by randomly extracting a contiguous temporal segment from the original trajectory.

Let $r \in ( 0 , 1 ]$ denote the minimum subsequence ratio. The minimum subsequence ratio � controls the amount of information retained in each sample. The minimum allowable subsequence length is defined as $L _ { \mathrm { m i n } } = \operatorname* { m a x } ( \lfloor r T \rfloor$ , 1), where ⌊⋅⌋ denotes the floor operator, i.e., the greatest integer less than or equal to its argument. A subsequence length � is then randomly and uniformly sampled from $L \sim \mathcal { V } \left( \{ L _ { \operatorname* { m i n } } , L _ { \operatorname* { m i n } } + 1 , \cdots , T \} \right)$ , and a starting index � is uniformly selected from $s \sim \mathcal { V } ( \{ 1 , 2 , \dots , T - L + 1 \} )$

The extracted subsequence is $\mathbf { X } _ { \mathrm { s u b } } ^ { ( i ) } = \left\{ \mathbf { u } _ { t } ^ { ( i ) } \right\} _ { t = s } ^ { s + L - 1 }$ , where $\mathbf { u } _ { t } ^ { ( i ) } = ( x _ { t } ^ { ( i ) } , y _ { t } ^ { ( i ) } , z _ { t } ^ { ( i ) } , v _ { x , t } ^ { ( i ) } , v _ { y , t } ^ { ( i ) } , v _ { z , t } ^ { ( i ) } )$ . To maintain a fixed input length for the CNN, the extracted subsequence is padded with zeros to recover the original trajectory length �, $\mathbf { e . g . , } \widetilde { \mathbf { X } } ^ { ( i ) } = \left( \mathbf { u } _ { s } ^ { ( i ) } , \mathbf { u } _ { s + 1 } ^ { ( i ) } , \ldots , \mathbf { u } _ { s + L - 1 } ^ { ( i ) } , \underbrace { \mathbf { 0 , } \ldots , \mathbf { 0 } } _ { T - L } \right)$ . Consequently, the classifier is trained using only a randomly selected portion of each trajectory while preserving a consistent input size.

## 4.2. Importance Score Visualization via Grad-CAM

For interpretability analysis, we employ Grad-CAM to identify the trajectory features that are most influential to the trained model predictions. Grad-CAM leverages the gradients obtained through backpropagation of the target output with respect to intermediate feature maps to generate a coarse localization map to highlight the regions that contribute most to the prediction. Since the CNN as shown in Figure 2 contains separate branches for processing position and velocity inputs, Grad-CAM is applied to the final convolutional layer of either the position or velocity branch.

Given a target class �, Grad-CAM generates a class-discriminative localization map ReLU $\left( \sum _ { k } a _ { k } ^ { c } A ^ { k } \right)$ , where $A ^ { k }$ denotes the �-th feature map of the last convolutional layer and $a _ { \scriptscriptstyle k } ^ { c }$ represents its importance weight for class �. The weights are computed by globally averaging the gradients of the class logit $y ^ { c }$ with respect to the feature map activations, $\begin{array} { r } { a _ { k } ^ { c } = \frac { 1 } { Z } \sum _ { i , j } \frac { \partial y ^ { \bar { c } } } { \partial A _ { i j } ^ { k } } } \end{array}$ , where � is the total number of elements in the feature map. The resulting weights quantify the contribution of each feature map to the prediction of class �. A weighted combination of the feature maps is then computed to produce a coarse localization map. Finally, the ReLU operation retains only positive contributions, highlighting the regions that positively influence the prediction while suppressing those that negatively afect the target class. The resulting Grad-CAM map, $L _ { \mathrm { G r a d - C A M } } ^ { c } \in \bar { \mathbb { R } } ^ { T }$ , identifies the temporal regions that are most influential to the classifier decision. Applying Grad-CAM to each branch separately produces a one-dimensional importance score for every time step in that branch. Consequently, the position and velocity branches each generate a temporal importance map, totally two time series of feature importance values.

To project the temporal importance scores into physical space, we associate the importance score at each time step with the corresponding position or velocity. Specifically, we define a bounding box enclosing the trajectory domain and partition it into a uniform 2D spatial grid. The importance scores from all test trajectories are then averaged within the corresponding grid boxes according to their spatial coordinates. The resulting spatial importance heatmap captures the statistical distribution of phenotype-discriminative regions across the physical domain.

## 5. Experiments

In this section, we first validate the performance of SubSeq augmentation and then leverage Grad-CAM for model interpretation. For performance evaluation, we benchmark SubSeq across diverse data regimes (from data-rich to data-scarce) and compare it with other data augmentation methods. For interpretability analysis, we first examine the individual contributions of position versus velocity features, and then statistically aggregate the test-set Grad-CAM maps to identify the generalized spatial and kinematic regions that dictate the classification decisions.

## 5.1. Experimental Protocol

The dataset contains a total of 516 CTC trajectories, including 262 soft-cell and 254 hard-cell trajectories. Among them, 416 trajectories are used for training and 100 for testing. For the Grad-CAM analysis, we further consider a filtered subset of 50 test trajectories that traverse the entire microfluidic channel, consisting of 36 soft-cell and 14 hard-cell trajectories.

The CNNs were trained using the cross-entropy loss function and the Adam optimizer (Kingma and Ba, 2014). To ensure the statistical reliability of the results against variations in data partitioning and NN model initialization, we evaluated each configuration using 75 independent runs. For each fixed train-test ratio, these runs were distributed across 5 distinct train-test splits, with each split evaluated across 15 independent trials. Each run was trained for 2000 epochs, and the final results from all 75 runs were aggregated and visualized using boxplots to evaluate the model performance.

![](images/77edb418fe8f9d7653b9311c62623f6b4bc247ecff5e1b0d13375d62b0fa4b78.jpg)

![](images/be257cb133e401d3b33b7828fb46cef3602b98b88fcc0f4b742db64e46813b6b.jpg)  
Figure 3: Comparison of original and SubSeq model performance across varying train-test splits. Boxplots show the distribution of testing accuracy (left) and ROC-AUC (right), with paired boxes indicating models trained on identical data splits. The central line denotes the median, boxes indicate the interquartile range, whiskers represent the data range, and hollow circles indicate outliers. Results show that SubSeq consistently achieves higher median testing accuracy and ROC-AUC than the baseline.

Model performance is evaluated using testing accuracy and the area under the receiver operating characteristic curve (ROC-AUC). The ROC-AUC is calculated on a held-out test set to assess the model’s overall discriminative ability, providing a threshold-independent measure of classification performance. The ROC-AUC metric ranges from 0.5 to 1, where values closer to 1 indicate better classification performance.

## 5.2. Classification Performance with SubSeq

SubSeq across diverse data regimes. We first evaluate the impact of the SubSeq across diverse data regimes, ranging from data-rich to data-scarce scenarios, by comparing accuracy and ROC-AUC over varying train-test splits.

Figure 3 shows that while reducing training data expectedly degrades the performance of both models, the SubSeq model consistently outperforms the baseline. Across diverse data regimes, SubSeq delivers higher median ROC-AUC and testing accuracy. Additionally, the narrower spread of performance distributions of SubSeq reflects greater stability across independent trials. The performance gap in favor of SubSeq is particularly notable at lower training data fractions, which confirms that SubSeq mitigates overfitting and enhances generalization in data-scarce scenarios.

SubSeq against alternative data augmentation methods. We evaluate the efectiveness and stability of SubSeq by comparing it to alternative data augmentation methods, including Cutout (Devries and Taylor, 2017) and Mixup (Zhang et al., 2017). We provide implementation details in Appendix 6.

Figure 4 compares the testing accuracy and ROC-AUC across the three augmentation methods. For testing accuracy (left), SubSeq achieves a comparable median to Cutout but exhibits a significantly tighter distribution and a much higher lower bound, which mitigates the worst-case performance seen in Cutout. In terms of ROC-AUC (right), while Cutout yields the highest peak and median values, it also shows the largest variability. In contrast, SubSeq maintains a competitive median with reduced spread. Overall, while Cutout ofers strong peak potential, SubSeq provides the most consistent and stable performance improvements across both metrics, making it a more reliable data augmentation strategy for the CTC dataset.

## 5.3. Interpreting CTC Classification

Efect ofinputfeatures: position vs. velocity. To gain a better understanding of the role of diferent input features in model performance, we evaluate the respective contributions of positional and velocity information. Specifically, we conduct an ablation study comparing a baseline configuration that utilizes both position and velocity features against a modified variant that relies solely on velocity or position data. These models are trained using the same parameters outlined in section 5.1 and tested across varying train-test splits.

![](images/15f83f0382a39e950d139c236246a9c11d455712922a51b49b0015602c5aad9b.jpg)

![](images/89a8cea9a73c5a7681f9a3dbab365f914b76285c49e7bb6a42d3ace38d0a433e.jpg)  
Figure 4: Performance comparison of data augmentation strategies under an 80%/20% training/testing split. Boxplots display the distribution of testing accuracy (left) and ROC-AUC (right) across the SubSeq, Cutout, and Mixup methods. Results show that SubSeq ofers the most stable performance (lower variability) in both testing accuracy and ROC-AUC across the evaluated augmentation methods.

![](images/b916c1f3995079d13d282f5f4b77404e7351ea81971d5c88e8f3cf5bc1b0f6aa.jpg)

![](images/6b3860a10b1ac3de077b900cfbcfc5d38e1593aa9a75027a6cb2c24507c80d51.jpg)  
Figure 5: Efect of input representation on model performance. Boxplots display the distributions of testing accuracy (left) and ROC-AUC (right) for models trained using SubSeq with velocity-only/position-only inputs or combined position and velocity inputs across varying train–test splits. Results show that velocity provides most of the discriminative information, while combining position and velocity yields the best performance.

Figure 5 evaluates model performance using velocity-only or positional-only inputs versus combined position and velocity features. The velocity-only configuration delivers highly competitive results, indicating that velocity features drive the core predictive performance. Conversely, models relying solely on positional information perform significantly worse, proving that position alone is insuficient for accurate classification. Nevertheless, integrating both modalities consistently yields the best overall results across all metrics. This demonstrates that positional data provides complementary information that enhances the model predictive capability.

This behavior is further illustrated by the ROC curves shown in Figure 6. Quantitatively, the unaugmented baseline model utilizing combined position and velocity inputs yields an ROC-AUC of 0.8612. Introducing data augmentation boosts the performance to an AUC of 0.9076 for the velocity-only configuration and to a peak of 0.9347 for the full model incorporating both feature modalities. Visually, the velocity-only model exhibits a more rigid, stepwise curve profile compared to the smoother curve profile that includes both position and velocity inputs. This suggests that relying only on velocity data while omitting positional information leads to more discrete decision boundaries.

These findings indicate that velocity captures the essential characteristics needed for classification, whereas positional information acts as an important supplement. Combining both feature types is therefore vital to maximizing overall predictive capability.

![](images/b7021fe99071b2f8e3633fc575642b4a09b8f503915a43db37d94f76ecee4039.jpg)  
(a) Position + Velocity

![](images/cd7e9f13d8068e64cbbfc86cae7f61cc00ffb3024a95a1f18bdf36361e42626a.jpg)  
(b) SubSeq & Position only

![](images/1c8903f50e316bdc00ce00313b460ea1b1e8262617ab232cbed0b5a20420d709.jpg)  
(c) SubSeq & Velocity only

![](images/f8adee2fa3c44c553c71637f6e71fa3fb8388d2acb75a0e3c9641d0ed7f025b1.jpg)  
(d) SubSeq & Position + Velocity  
Figure 6: ROC-AUC under an 80/20 train–test split across diferent input and augmentation configurations. Panels illustrate models trained with: (Top Left) combined position and velocity inputs without data augmentation, (Top Right) Positiononly inputs with SubSeq, (Bottom Left) Velocity-only inputs with SubSeq, and (Bottom Right) combined position and velocity inputs with SubSeq. Results show that SubSeq improves the ROC-AUC from 0.8612 to 0.9347 when both position and velocity are used. We can see that velocity provides more discriminative information than position, while combining both features achieves the best classification performance.

Importance score interpretation. We employ Grad-CAM to visualize how the trained model utilizes input information when making predictions. Specifically, we analyze a model trained using SubSeq on an 80%/20% training/testing split, which achieved a testing accuracy of 91% and a ROC-AUC of 0.96. We first present trajectorylevel importance maps for individual test samples and then aggregate the importance scores across the test set to construct spatial importance heatmaps in Figures 7 and 8, respectively. For each prediction target (soft or hard cells), Grad-CAM is applied separately to the position and velocity branches, yielding four sets of importance maps that reveal how each input modality contributes to the model predictions.

Figure 7 presents the trajectory-level Grad-CAM importance scores for several representative test trajectories under both the soft- and hard-cell prediction targets. The visualizations show that the model consistently assigns high importance to only selected portions of each trajectory, rather than treating the entire trajectory as equally informative. This indicates that the classifier primarily relies on localized trajectory patterns when making its predictions.

To further investigate how feature importance is distributed across the spatial domain for diferent input modalities (position and velocity), we construct aggregated spatial importance heatmaps by projecting trajectory points onto the (�, �) plane, partitioning the domain into spatial bins, and averaging the Grad-CAM importance scores within each bin across all test trajectories, as described in Section 4. Figure 8 shows that the trained model does not rely uniformly on the entire microfluidic channel when classifying CTC phenotypes. Instead, the Grad-CAM heatmaps reveal localized regions of high importance, indicating that phenotype-discriminative information is concentrated in specific regions of the channel. This observation supports the hypothesis that the hyperuniform micropost array acts as a physical encoder: only specific cell–micropost and cell–fluid interactions within certain regions produce discriminative trajectory signatures that are critical for classification.

![](images/90edad75469f5221e0da9021907ef36704519d4263d45c6d8e8ec2304371c8dc.jpg)  
Figure 7: Grad-CAM importance maps for several representative test trajectories under the soft- and hard-cell prediction targets, for each input modality (position or velocity). Lighter colors indicate larger Grad-CAM importance scores. The labels on the right denote the specific conditions and outcomes for each trajectory. For example, label (a) indicates a Soft Cell trajectory analyzed via the Position Branch, which the model correctly classified with a probability of 1.0.

A clear distinction is observed between the position and velocity branches, as illustrated in Figure 8. For both soft and hard cells, the velocity-branch heatmaps exhibit more spatially localized regions of high importance, whereas the position-branch heatmaps display broader regions of influence. This suggests that velocity captures more localized and instantaneous information related to the cell’s hydrodynamic response, such as sudden acceleration, deceleration, or changes in direction near a micropost. In contrast, the position branch appears to encode more cumulative and global trajectory information.

Figure 8 further suggests that the two phenotypes are distinguished not simply by their overall trajectories, but by how their motion responds locally to the surrounding fluid and microposts. Comparing the Grad-CAM heatmaps for soft and hard cells, we observe that the regions of highest importance difer between the two phenotypes. This indicates that the model learns distinct, phenotype-specific trajectory signatures rather than relying on a single discriminative region within the device. For soft cells, the most influential regions may correspond to locations where cell deformation leads to larger local deviations, smoother navigation around microposts, or delayed recovery of momentum following micropost interactions. In contrast, for hard cells, the important regions may reflect locations where the motion exhibits sharper directional changes or more constrained trajectories. These observations are consistent with the underlying physical intuition: soft cells can deform and therefore tend to follow smoother, more compliant trajectories, whereas hard cells exhibit more rigid, constrained deflection patterns.

![](images/3652c343f481773bca32e2eaeb6f35b484428a42ab52156b5a3e92a17d36ec2b.jpg)  
Figure 8: Grad-CAM importance maps aggregated over test trajectories under the soft- and hard-cell prediction targets, for each input modality (position and velocity). Lighter colors indicate larger Grad-CAM importance scores. The labels on the right indicate the conditions for the aggregated Grad-CAM maps. For instance, label (a) corresponds to the aggregated importance map for Soft Cells evaluated through the Position Branch.

## 6. Discussion & Conclusion

In this work, we presented an interpretable and data-eficient deep learning framework for trajectory-based circulating tumor cell (CTC) phenotype classification in a hyperuniform microfluidic device. The proposed framework addresses two important challenges in trajectory-based learning: the limited availability of simulated training data and the lack of transparency in deep neural network predictions. To improve model generalization under datascarce conditions, we introduced the Subsequence (SubSeq) augmentation strategy, which trains the classifier using randomly sampled contiguous trajectory segments. To improve interpretability, we integrated Grad-CAM to identify the trajectory regions and corresponding physical locations within the microfluidic device that contribute most strongly to phenotype classification.

Experimental results demonstrate that SubSeq consistently improves classification performance across varying training data regimes while providing stable and reliable generalization compared to existing augmentation methods. Interpretability analysis further reveals that accurate predictions rely primarily on localized trajectory segments rather than complete trajectories, providing a physical explanation for the efectiveness of SubSeq. Moreover, the Grad-CAM visualizations show that velocity features contain the strongest predictive information, while positional information provides complementary context that further improves classification performance. Aggregated importance maps additionally indicate that discriminative information is concentrated within specific regions of the microfluidic channel and difers between soft and hard cell phenotypes, suggesting that distinct cell-fluid-structure interactions encode phenotype-specific mechanical signatures.

Beyond improving classification accuracy, this work presents an integrated framework in which the hyperuniform micropost array functions as a physical encoder, transforming intrinsic cellular mechanical properties into trajectory signatures that can be decoded by machine-learning models. By mapping learned importance back to the physical domain, the framework provides mechanistic insight into where and how discriminative information arises, potentially informing more eficient biomedical analysis and future microfluidic device design.

We also acknowledge several limitations of the present study. First, the dataset is derived entirely from numerical simulations and contains a relatively small number of trajectories representing only two mechanical phenotypes. Consequently, the extent to which the trained models generalize to experimentally observed cells and to a broader range of cellular mechanical properties remains uncertain. Second, the current study considers only a single microfluidic device configuration and one convolutional neural network architecture. Future work will therefore focus on validating the proposed framework using experimentally acquired trajectory data, incorporating more diverse cellular characteristics and microfluidic geometries, and evaluating the robustness and efectiveness of SubSeq across diferent neural network architectures. We will also investigate how model interpretability can be incorporated directly into the optimization of device architectures. More broadly, the proposed combination of data-eficient learning and physically grounded interpretability provides a general framework for trajectory-based analysis in microfluidics and other scientific applications where both predictive performance and mechanistic understanding are important.

## Appendix

Training parameters and auxiliary strategies, including learning rate scheduling (e.g. linear warm-up, cosine annealing) and noise injection, are systematically explored for each augmentation method. These techniques are incorporated selectively and only when they are observed to improve performance for a given method during fine-tuning. As a result, each augmentation approach is evaluated under its own best-performing configuration. Hyperparameters and auxillary strategies are tuned independently for each method using the same evaluation protocal, ensuring a fair and consistent basis for comparison.

The SubSeq-augmented model utilizes a learning rate of $3 \times 1 0 ^ { - 4 }$ , an activation probability of 0.99, and a minimum length ratio of 0.6 relative to the full sequence length.

SubSeq against alternative data augmentation methods. We consider a fixed 80∕20 train-test split. In this experiment, the SubSeq method uses the same parameters as outlined above. Cutout was implemented with a learning rate of $3 \times 1 0 ^ { - 4 }$ , a masking probability of 0.99, and a maximum mask width of 250. Mixup was implemented with a learning rate of $5 \times 1 0 ^ { - 4 }$ and a mixing coeficient of $\alpha = 0 . 5$

## Data & Code Availability

The CTC trajectory dataset used in this study was obtained from the GitHub repository: https://github.com/ imsanjoykb/Microfluidic-Device-Data-CTC\_Model.

The code required to reproduce the results presented in this paper is publicly available at: https://github.com/ sjsusu/ctc-classification-model. All code is released under the MIT License.

## Acknowledgement

S. L. acknowledges partial support from the Texas Tech University startup funds. Y. W. acknowledges partial support from the National Science Foundation under grant DMS-2247001, the Cancer Prevention and Research Institute of Texas (CPRIT) under grant RP260780, and Simons Foundation Travel Award. The authors thank Dr. Wei Li for insightful discussions regarding circulating tumor cell dynamics and microfluidic device design.

## CRediT authorship contribution statement

Serena Su: Methodology, Software, Original draft preparation. Yifan Wang: Conceptualization of this study, Methodology, Data curation, Original draft preparation. Senwei Liang: Conceptualization of this study, Methodology, Software, Original draft preparation.

## References

Alix-Panabières, C., Pantel, K., 2014. Challenges in circulating tumour cell research. Nature Reviews Cancer 14, 623–631. doi:10.1038/nrc3820.

Chen, W., Shi, K., 2021. Multi-scale attention convolutional neural network for time series classification. Neural Networks 136, 126– 140. URL: https://www.sciencedirect.com/science/article/pii/S0893608021000010, doi:https://doi.org/10.1016/j. neunet.2021.01.001.

Chen, Y., Li, P., Huang, P.H., Xie, Y., Mai, J.D., Wang, L., Nguyen, N.T., Huang, T.J., 2014. Rare cell isolation and analysis in microfluidics. Lab on a Chip 14, 626–645. doi:10.1039/C3LC90136J.

Cubuk, E.D., Zoph, B., Mane, D., Vasudevan, V., Le, Q.V., 2019. Autoaugment: Learning augmentation strategies from data, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).

Devries, T., Taylor, G.W., 2017. Improved regularization of convolutional neural networks with cutout. CoRR abs/1708.04552. URL: http: //arxiv.org/abs/1708.04552, arXiv:1708.04552.

Di Carlo, D., 2009. Inertial microfluidics. Lab on a Chip 9, 3038–3046. doi:10.1039/B912547G.

Gao, J., Song, X., Wen, Q., Wang, P., Sun, L., Xu, H., 2020. Robusttad: Robust time series anomaly detection via decomposition and convolutional neural networks. arXiv preprint arXiv:2002.09545 .

He, K., Zhang, X., Ren, S., Sun, J., 2016. Deep residual learning for image recognition, in: Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 770–778.

Huang, Z., Liang, S., Liang, M., 2025. A generic shared attention mechanism for various backbone neural networks. Neurocomputing 611, 128697.

Hyun, Y.J., Yoo, Y., Kim, Y., Lee, T., Kim, W., 2024. Encoding time series as images for anomaly detection in manufacturing processes using convolutional neural networks and grad-cam. International Journal of Precision Engineering and Manufacturing 25, 2583–2598. URL: https://doi.org/10.1007/s12541-024-01069-6, doi:10.1007/s12541-024-01069-6.

Joosse, S.A., Gorges, T.M., Pantel, K., 2015. Biology, detection, and clinical implications of circulating tumor cells. EMBO Molecular Medicine 7, 1–11. doi:10.15252/emmm.201303698.

Joshi, R., Ahmadi, H., Gardner, K., Bright, R.K., Wang, W., Li, W., 2025. Advances in microfluidic platforms for tumor cell phenotyping: from bench to bedside. Lab on a Chip 25, 856–883. URL: https://doi.org/10.1039/D4LC00403E, doi:10.1039/D4LC00403E.

Kingma, D.P., Ba, J., 2014. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980 .

Kumar, S., Wang, Y., Zhan, H., Gardner, K., Thompson, T., Li, W., Canic, S., 2025. Classification of circulating tumor cells using machine learning on microfluidic trajectory data, in: Huang, L., Greenhalgh, D. (Eds.), Proceedings of 17th International Conference on Machine Learning and Computing, Springer, Cham. pp. 456–467. doi:10.1007/978-3-031-94898-5\_34.

Liang, M., Su, J.C., Schulter, S., Garg, S., Zhao, S., Wu, Y., Chandraker, M., 2024. Aide: An automatic data engine for object detection in autonomous driving, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 14695–14706.

Lin, J., Huang, Z., Wang, K., Liu, L., Lin, L., 2024. Continuous value assignment: A doubly robust data augmentation for of-policy learning. IEEE Transactions on Neural Networks and Learning Systems 36, 8153–8165.

Micalizzi, D.S., Maheswaran, S., Haber, D.A., 2017. A conduit to metastasis: circulating tumor cell biology. Genes & Development 31, 1827–1840. doi:10.1101/gad.305805.117.

Minh, A.P.T., 2023. Overview of class activation maps for visualization explainability. URL: https://arxiv.org/abs/2309.14304, arXiv:2309.14304.

Pantel, K., Brakenhof, R.H., Brandt, B., 2008. Detection, clinical relevance and specific biological properties of disseminating tumour cells. Nature Reviews Cancer 8, 329–340. doi:10.1038/nrc2375.

Rejuan, R., Aulisa, E., Li, W., Thompson, T., Kumar, S., Canic, S., Wang, Y., 2025. Validation of a microfluidic device prototype for cancer detection and identification: Circulating tumor cells classification based on cell trajectory analysis leverag ing cell-based modeling and machine learning. International Journal for Numerical Methods in Biomedical Engineering 41, e70037. URL: https://onlinelibrary.wiley.com/doi/abs/10.1002/cnm.70037, doi:https://doi.org/10.1002/cnm.70037, arXiv:https://onlinelibrary.wiley.com/doi/pdf/10.1002/cnm.70037. e70037 CNM-Sep-24-0967.R1.

Rudin, C., 2019. Stop explaining black box machine learning models for high stakes decisions and use interpretable models instead. Nature Machine Intelligence 1, 206–215. doi:10.1038/s42256-019-0048-x.

Selvaraju, R.R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., Batra, D., 2017. Grad-CAM: Visual explanations from deep networks via gradient-based localization, in: Proceedings of the IEEE International Conference on Computer Vision (ICCV), pp. 618–626. doi:10.1109/ ICCV.2017.74.

Selvaraju, R.R., Das, A., Vedantam, R., Cogswell, M., Parikh, D., Batra, D., 2016. Grad-cam: Why did you say that? visual explanations from deep networks via gradient-based localization. CoRR abs/1610.02391. URL: http://arxiv.org/abs/1610.02391, arXiv:1610.02391.

Tan, J., Ding, Z., Hood, M., Li, W., 2019. Simulation of circulating tumor cell transport and adhesion in cell suspensions in microfluidic devices. Biomicrofluidics 13, 064105. URL: https://pmc.ncbi.nlm.nih.gov/articles/PMC6837944/, doi:10.1063/1.5129787.

van Zyl, C., Ye, X., Naidoo, R., 2024. Harnessing explainable artificial intelligence for feature selection in time series energy forecasting: A comparative analysis of grad-cam and shap. Applied Energy 353, 122079. URL: https://www.sciencedirect.com/science/article/ pii/S0306261923014435, doi:https://doi.org/10.1016/j.apenergy.2023.122079.

Wang, Y., Li, W., 2024. Innovative microfluidic device design detects circulating tumor cells. SIAM News. URL: https://www.siam. org/publications/siam-news/articles/innovative-microfluidic-device-design-detects-circulating-tumor-cells/. accessed 15 June 2026.

Warkiani, M.E., Tay, A.K.P., Guan, G., Han, J., 2015. Membrane-less microfiltration using inertial microfluidics. Scientific Reports 5, 11018. doi:10.1038/srep11018.

Wen, Q., Sun, L., Yang, F., Song, X., Gao, J., Wang, X., Xu, H., 2021. Time series data augmentation for deep learning: A survey, in: Proceedings of the Thirtieth International Joint Conference on Artificial Intelligence, International Joint Conferences on Artificial Intelligence Organization. pp. 4653–4660.

Wen, T., Keyes, R., 2019. Time series anomaly detection using convolutional neural networks and transfer learning. arXiv preprint:1905.13628 . Whitesides, G.M., 2006. The origins and the future of microfluidics. Nature 442, 368–373. doi:10.1038/nature05058.

Yu, M., Bardia, A., Wittner, B.S., Stott, S.L., Smas, M.E., Ting, D.T., Isakof, S.J., Ciciliano, J.C., Wells, M.N., Shah, A.M., Concannon, K.F., Donaldson, M.C., Sequist, L.V., Brachtel, E., Sgroi, D., Baselga, J., Ramaswamy, S., Toner, M., Haber, D.A., Maheswaran, S., 2013. Circulating breast tumor cells exhibit dynamic changes in epithelial and mesenchymal composition. Science 339, 580–584. doi:10.1126/science. 1228522.

Zhang, H., Cissé, M., Dauphin, Y.N., Lopez-Paz, D., 2017. mixup: Beyond empirical risk minimization. CoRR abs/1710.09412. URL: http://arxiv.org/abs/1710.09412, arXiv:1710.09412.

Zhou, B., Khosla, A., Lapedriza Garcia, A., Oliva, A., Torralba, A., 2016. Learning deep features for discriminative localization. URL: http://hdl.handle.net/1721.1/112986.