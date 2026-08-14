# Multi-perspective Imbalance-Conscious 6G Beamforming Optimization and Performance

Chukwunonso Henry Nwokoye

Blessing Oluchi Iloka

Chikwue V. Umeugoji

Liberal Arts & Professional Studies

Physics, Engineering & Computer Science

IT Services,

York University,

University of Hertfordshire

Joint Admissions & Matn. Board

Toronto, Canada

0000-0002-2534-3734

England, United Kingdom

Lagos, Nigeria

Boluchi23@gmail.com

Umeugoji.chikwue@gmail.com

Christopher Anene Egemba

AI Product Development

Lightenet Technologies Ltd.

United Kingdom

0009-0007-2228-8243

Nnenna D. Duroha

Physics, Engineering & Computer Science

University of Hertfordshire

England, United Kingdom

nnennadaisy@yahoo.com

Abstract—The study presents a systematic machine learning (ML) study of 6G-IoT beamforming optimization (6GBO) using supervised and unsupervised approaches. We compared the predictive power of network, environmental, device, and vision feature groups for 6GBO. Additionally, it addressed other unsupervised perspectives that can enhance 6GBO, including clustering network scenarios using methods such as K-means, DBSCAN, and hierarchical clustering. Several imbalance-aware experiments revealed that network features possess better prediction power than device, environmental and vision feature groups, as evidenced by their recall, F1-score and ROC-AUC values. For unsupervised ML exploration (assessed using Elbow, Silhouette score and Davies Bouldin Index methods), the results indicate that the deployment environment and type of device primarily influence clustering, rather than mobility-based attributes. Furthermore, the explainability analysis showed that bandwidth, IoT sensors, and mobility possess higher global feature importance across the feature groups. In the future, we would apply deep and reinforcement learning techniques to predict throughput/latency or to optimize rewards determined by performance indicators like SNR enhancement.

Index Terms—Sixth Generation beamforming optimization (6GBO), Internet of Things (IoT), machine learning (ML), imbalance handling, Clustering, eXplainable AI (XAI)

## I. INTRODUCTION

The integration of the Internet of Things (IoT) with sixthgeneration (6G) networks is poised to revolutionize global connectivity by enabling intelligent, ultra-reliable, and lowlatency communication across diverse sectors, including industrial automation, smart cities, and autonomous systems [1]. As 6G-IoT environments grow increasingly multifaceted [2], they face substantial challenges in dynamic situations where the allocation of resources and reliable communication are hindered by user mobility, network topologies, and rapid channel variations. In such circumstances, static beamforming techniques are insufficient; instead, smart and adaptive beamforming approaches capable of instantaneously responding to changing signal contexts are necessary. Beamforming is a signal processing method employed in sensor or antenna arrays to guide signal transmission or reception by modifying the amplitude and phase of each component, thereby improving signal quality and minimizing interference [3]. This phenomenon boosts signal strength, mitigates interference, and improves the overall performance of the network [4], [5].

With the advancement of wireless communication networks in the direction of 6G [2], beamforming has become a dependable method for attaining higher data rates, improved spectral efficiencies, and ultra-reliable low-latency communication (URLLC) [6]. Beamforming enhances transfer and reception by concentrating signal energies on certain devices or users, which is crucial in high-frequency bands like THz and mmWave, where signal reduction is unavoidable [4], [7]. However, conventional static beamforming techniques face considerable obstacles in dynamic 6G-IoT settings marked by user movement, high device density, and rapidly changing channel scenarios that demand real-time flexibility. Adaptive beamforming approaches, which are supported by machine learning (ML), have shown significant potential in overcoming these challenges [8]. ML models can actively assess real-time data, predict ideal beam trajectories, and independently adjust transmission techniques to enhance reliability and performance [9].

We believe there is a need to move beyond monolithic modeling of beamforming success by isolating distinct feature domains—such as network-level parameters, environmental context, or device characteristics—and comparing their relative predictive power. Such approaches allow researchers to ask systems-level questions like “Which factor category (network, environment, device, vision) most strongly determines beamforming success?” or “Are environmental conditions more critical than network configurations?” From a 6G resourceallocation and adaptive beamforming perspective, this analysis is highly insightful: it enables network designers to prioritize measurement and processing investments toward the domains that drive performance, rather than assuming all features carry equal significance. Considering these questions, this study aims to conduct a comparative analysis of 6G beamforming optimization (6GBO) using network, environmental, device and vision factors with ML classifiers.

## II. RELATED WORK

Prior research used supervised and unsupervised ML methods in conventional wireless systems for channel prediction and dynamic beamforming. Zhou et al. [10] conducted an examination of ML solutions in IoT-supported wireless networks, highlighting the use of decision trees and neural networks for link adaptation and interference reduction. Qin et al. [11] showed that deep learning algorithms can effectively simulate signal processing at the physical layer, offering reliable solutions for channel feedback and beam option selection in variable situations. Reinforcement learning (RL) has garnered interest for its capacity to manage sequential decision-making processes in beamforming. Ye et al. [12] used deep RL to enhance resource allocation in vehicular networks, demonstrating its efficacy in adjusting beam directions within quickly evolving topologies.

The integration of beamforming optimization and ML [13] is facilitating the development of scalable, intelligent, and adaptable wireless networks, thereby establishing a basis for the ongoing expansion and robustness of next-generation communication infrastructure [14]. The incorporation of ML into wireless communication networks has garnered considerable attention owing to its capacity to tackle the issues posed by dynamic settings, high mobility, and optimal resource allocation. Specifically, ML-driven beamforming approaches have demonstrated potential for performance improvement in nextgeneration networks, including 5G [13], and 6G [14], as well as intelligent reflecting surface communication technologies.

Our study is aimed at understanding the impact of these factors on beamforming optimization using machine learning models: Logistic Regression (LR), XGBoost (XGB), Random Forest (RF), Gradient Boosting (GB), Support Vector Machine (SVM) using a radial basis function (RBF) kernel, Multilayer Perceptron (MLP), AdaBoost (AB), Gradient Boosting (GB), LightGBM, Extra Trees, and K-Nearest Neighbors (KNN). In contrast to previous GitHub implementations that utilize the entire dataset [15], encompassing derived performance metrics, our study intentionally excludes the pre-decision feature groups (FGs)—environmental, network, device, and vision—to evaluate their individual predictive efficacy for beamforming optimization success.

## III. METHODOLOGY

We adopted the ML development life cycle, which includes data collection; preprocessing and feature engineering (normalization, handling missing values, and encoding); dataset splitting; modeling (using classifiers); evaluation; and results. The 6G IoT intelligent management dataset [16] used for the study was accessed from Kaggle. Input features for our study include network (frequency (GHz), transmit power (dBm), bandwidth (MHz), codebook size, and interference level (dB)); environment (obstacle density, mobility (m/s), and environment type (outdoor)); and device (number of antennas and device type (IoT sensor and smartphone)). These input features are available before the occurrence of optimization. While running the experiments, the features were defined as groups, i.e., network parameters (NP), environmental factors (EF), device characteristics (DC), vision attributes (VA), and all features (AF). Note that VAs are Scale-Invariant Feature Transform (SIFT) keypoints, which are also part of the 6G IoT intelligent management dataset [16]. Essentially, the target variable is the column called ”optimized”, which contains 0s and 1s for unsuccessful and successful 6GBO. To evaluate the performance of the models, we employed the following metrics: precision (Pre), recall (Rec), F1, ROC-AUC, and PR-AUC.

Prior to model training, missing values were replaced with the median and standardized, whereas categorical variables were substituted with the mode and subjected to one-hot encoding. Boolean attributes were transformed into floatingpoint representations. Additionally, a stratified division of 80:20 for training and testing was implemented with a constant random seed (42) to maintain class distributions. On class balance, the positive rate is 0.168 (i.e., 168 positives out of 1000). In every aspect of preprocessing—such as feature normalization, median imputation, one-hot encoding, Boolean feature transformation, and SMOTE oversampling (when applicable)—were integrated into an imblearn pipeline and implemented separately within each cross-validation training fold, consequently averting the issue of data leakage.

In our experiments, we utilized a 5-fold cross-validation (CV) technique for out-of-fold (OOF) threshold tuning (TT). By implication, the training dataset was partitioned into 5 stratified subsets, with 4 folds allocated for training and one fold designated for validation. The method was continuously repeated till every fold had functioned once as the validation set. The aggregated OOF prediction probabilities were utilized to establish the ideal classification threshold, which improved the F1-score in a situation of significant class imbalance. This approach is essentially distinct from GridSearchCV (GSCV), which was utilized largely for the optimization of hyperparameters (Table I) and the selection of models by determining the optimal parameter combinations according to cross-validation scoring measures.

TABLE I  
GRIDSEARCHCV HYPERPARAMETER SEARCH SPACE
<table><tr><td rowspan=1 colspan=1>Classifier</td><td rowspan=1 colspan=1>Hyperparameter Search Space</td></tr><tr><td rowspan=1 colspan=1>LR</td><td rowspan=1 colspan=1>C ∈ {0.01, 0.1, 1.0}</td></tr><tr><td rowspan=1 colspan=1>RF</td><td rowspan=1 colspan=1>nestimators ∈ {100, 200}, max_depth ∈ {None, 10}</td></tr><tr><td rowspan=1 colspan=1>XGB</td><td rowspan=1 colspan=1>nestimators ∈ {100, 200}, max_depth ∈ {3, 6},learning_rate ∈ {0.1, 0.01}</td></tr><tr><td rowspan=1 colspan=1>GB</td><td rowspan=1 colspan=1>nestimators ∈ {100, 200}, learning_rate ∈ {0.1, 0.01}</td></tr><tr><td rowspan=1 colspan=1>AdaBoost</td><td rowspan=1 colspan=1>nestimators ∈ {50, 100}, learning_rate ∈ {1.0, 0.1}</td></tr><tr><td rowspan=1 colspan=1>SVM</td><td rowspan=1 colspan=1>C ∈ {0.1, 1.0}, γ ∈ {scale, auto}</td></tr><tr><td rowspan=1 colspan=1>MLP</td><td rowspan=1 colspan=1>Hidden layers ∈ {(64, 32), (128, 64)}, α ∈ {10−4, 10−3}</td></tr></table>

Table II presents the ablation study protocol used to evaluate the effect of feature leakage on 6GBO classification. Therein, Model 2 input features include Network, Environment, Device and Vision (pre-decision only); Model 2 input features include performance metrics only, and Model 3 features include pre-decision and performance metrics (outcomes). Model 2 and 3 was implemented to illustrate the significant jump in evaluation metrics (accuracy/AUC), thus demonstrating target leakage. Other columns in Table II present methods applied in order to address class imbalance.

TABLE II  
ABLATION STUDY PROTOCOL
<table><tr><td rowspan=1 colspan=1>Model</td><td rowspan=1 colspan=1>Feature Configuration</td><td rowspan=1 colspan=1>SMOTE</td><td rowspan=1 colspan=1>TT</td><td rowspan=1 colspan=1>GSCV</td></tr><tr><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>Pre-decision (NP, EF, DC, VA)</td><td rowspan=1 colspan=1>Yes</td><td rowspan=1 colspan=1>Yes</td><td rowspan=1 colspan=1>No</td></tr><tr><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>Pre-decision (NP, EF, DC, VA)</td><td rowspan=1 colspan=1>Yes</td><td rowspan=1 colspan=1>No</td><td rowspan=1 colspan=1>Yes</td></tr><tr><td rowspan=1 colspan=1>1-Ensembles</td><td rowspan=1 colspan=1>Pre-decision (NP, EF, DC, VA)</td><td rowspan=1 colspan=1>Yes</td><td rowspan=1 colspan=1>Yes</td><td rowspan=1 colspan=1>No</td></tr><tr><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>Performance Metrics Only</td><td rowspan=1 colspan=1>Yes</td><td rowspan=1 colspan=1>No</td><td rowspan=1 colspan=1>No</td></tr><tr><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>Model 1 + Performance Metrics</td><td rowspan=1 colspan=1>No</td><td rowspan=1 colspan=1>No</td><td rowspan=1 colspan=1>No</td></tr></table>

## IV. RESULTS & DISCUSSION

The initial experiment reveals that class imbalance dominates the observed performance. Recall that the dataset’s positive rate (optimized = 1) is small, so many models converge to predicting the majority (non-optimized) class. As a consequence, overall accuracy is high (≈0.83) but provides a misleading impression of success: several classifiers (notably those trained only on device-level features) report precision = 0, recall = 0, and F1 = 0, indicating that they never detect any positive examples. This behavior matches the majority-class baseline (predicting all samples as negative) and shows that accuracy alone is not an informative metric in this setting. Results of our experiments are available at the following GitHub link: https://github.com/ChiNonsoHenry16/Multi-perspective-Imbalace-Aware-6G-Beatforming-Optimization.

Through preliminary experiments, a more discriminating view is obtained from the class-sensitive metrics and the probabilistic ranking metrics (ROC-AUC). ROC-AUC values across the experiments lie near random to weakly informative (approximately 0.44–0.57), with the best AUCs reaching roughly 0.56–0.57 for a few models trained on environmental or network features. Correspondingly, only a handful of models (e.g., RF and XGBoost on network/environmental features) produce non-zero recall and modest F1 scores (examples: Environmental RF F1 ≈ 0.21, Environmental XGBoost F1 ≈ 0.15, and Network XGBoost F1 ≈ 0.08). These non-zero values indicate there is some discriminative signal in network and environmental features, but it is weak and insufficient for reliable detection with the current data and training setup.

Due to the significant dataset class imbalance (positive class of 16.8%), the evaluation of model performance predominantly relied on Recall, F1-score, ROC-AUC, and PR-AUC, as these metrics offer a more nuanced evaluation of minority-class identification compared to classification accuracy. Accuracy is conveyed as a supplementary measure for thoroughness.

## A. Model 1: Pre-optimization Factors Only

Model 1 presents a deployable prediction pipeline that employs pre-decision data. Here, the classifiers were used on just the network, environmental, device and vision parameters alone, without the addition of the performance metrics. They are as follows: NP (Frequency, Transmit Power, Bandwidth, Codebook Size, Interference Level; EF (Obstacle Density, Mobility, Environment-Outdoor); DC (Number of Antennas, Device Type-IoT Sensor, Device Type-Smartphone); and vision attributes (VA) (SIFT). In other words, these parameters are derived pre-optimization and exclude the following: Beamforming Gain, Latency, Energy Consumption, Throughput, Beam Training Time, SNR Improvement, Processing Time, and Memory Usage.

Tables III and IV display the best-performing (BP) ML models for the classification of 6GBO at a decision threshold (of 0.5) and the tuned (adjusted) threshold conditions, respectively. The code implementation employed libraries for preprocessing, encoding (one-hot), SMOTE for handling imbalance, probability calibration, and threshold refinement to tackle the significant class imbalance.

At a 0.5 threshold, the network and environmental groups demonstrated a comparably better performance, with AdaBoost and XGBoost attaining the highest F1 values of 0.327 and 0.324, respectively. After TT, recall significantly enhanced every FG, with some models attaining recall values around or equal to 1.0. LightGBM achieved the highest F1-score of 0.301 for NP, while SVM (RBF) attained the same score of 0.301 for vision attributes. The results demonstrate that network attributes offer strong predictive signals for 6GBO, whereas environmental, vision, and device features supply supplementary data for ML classification. No individual classifier consistently excelled throughout all feature domains, indicating significant disparities in the fundamental data distributions across the feature spaces.

Numerous significant insights arise from Tables III and IV. Firstly, TT significantly enhanced recall among all FG, affirming the significance of imbalance-conscious classification for 6GBO predictions. Secondly, NP consistently demonstrated a comparably better overall performance, whereas environmental and device features yielded identical F1-scores, suggesting that contextual and device-specific information significantly contributes to beamforming efficacy. Thirdly, the utilization of all features did not substantially exceed the performance of network parameters alone, indicating that supplementary feature groups offer minimal additional predictive capability. Ultimately, various feature groups preferred distinct ML techniques, with LightGBM, XGBoost, LR, MLP, and SVM (RBF) identified as the most effective models for specific feature domains, underscoring the varied characteristics of 6GBO prediction tasks.

TABLE III  
BEST-PERFORMING MODELS AT DEFAULT THRESHOLD (0.5)
<table><tr><td>Feature Group (FG)</td><td>Model</td><td>Acc.</td><td>Prec.</td><td>Recall</td><td>F1</td></tr><tr><td>All Features (AF)</td><td>AdaBoost</td><td>0.645</td><td>0.224</td><td>0.441</td><td>0.297</td></tr><tr><td>Network Parameters (NP)</td><td>AdaBoost</td><td>0.505</td><td>0.212</td><td>0.706</td><td>0.327</td></tr><tr><td>Environmental Factors (EF)</td><td>XGBoost</td><td>0.520</td><td>0.213</td><td>0.676</td><td>0.324</td></tr><tr><td>Device Characteristics (DC)</td><td>XGBoost</td><td>0.170</td><td>0.170</td><td>1.000</td><td>0.291</td></tr><tr><td>Vision Attributes (VA)</td><td>AdaBoost</td><td>0.285</td><td>0.166</td><td>0.794</td><td>0.274</td></tr></table>

TABLE IV  
BEST PERFORMING MODELS AFTER THRESHOLD TUNING
<table><tr><td>FG</td><td>Model</td><td>Acc</td><td>Prec</td><td>Rec</td><td>F1</td><td>ROC-AUC</td><td>PR-AUC</td></tr><tr><td>AF</td><td>MLP</td><td>0.205</td><td>0.176</td><td>1.000</td><td>0.300</td><td>0.533</td><td>0.195</td></tr><tr><td>NP</td><td>LightGBM</td><td>0.210</td><td>0.177</td><td>1.000</td><td>0.301</td><td>0.562</td><td>0.202</td></tr><tr><td>EF</td><td>XGBoost</td><td>0.180</td><td>0.172</td><td>1.000</td><td>0.293</td><td>0.571</td><td>0.203</td></tr><tr><td>DC</td><td>LR</td><td>0.200</td><td>0.172</td><td>0.971</td><td>0.292</td><td>0.533</td><td>0.184</td></tr><tr><td>VA</td><td>SVM</td><td>0.280</td><td>0.180</td><td>0.912</td><td>0.301</td><td>0.466</td><td>0.164</td></tr></table>

Despite some classifiers attaining elevated accuracy@0.5 levels between 0.70 and 0.83, a majority of these models demonstrated significantly deficient recall and F1-score results owing to the pronounced class imbalance within the dataset.

Table V displays the top-performing models using SMOTE and optimized by GridSearchCV, as determined by F1-score. Across all groups, DC attained the best performance, with AdaBoost attaining the highest F1 score of 0.307 and an ROC-AUC of 0.564. The SVM (RBF) classification algorithm attained the highest F1 score of 0.256 for NP, while LR excelled for EF with a 0.266 F1 score. Considering the VA category (with SIFT keypoints), AdaBoost attained the highest F1-score of 0.240. These findings indicate that device-related characteristics yield a better predictive signal for 6GBO, but environmental, network, and VA-based parameters demonstrate moderate predictive efficacy when assessed in isolation. While the highest standalone F1 value in the GridSearchCV studies was attained by AdaBoost utilizing device-related characteristics (F1 = 0.307), network attributes consistently yielded superior ROC-AUC and PR-AUC values across many experimental configurations. Thus, NP-related properties seem to yield better discriminatory information for predicting 6GBO, whereas device characteristics present a desirable precision-recall equilibrium at specific classification levels.

TABLE V  
BEST GRIDSEARCHCV-TUNED MODELS BY FEATURE GROUP
<table><tr><td rowspan=1 colspan=1>FG</td><td rowspan=1 colspan=1>Best Model</td><td rowspan=1 colspan=1>Accuracy</td><td rowspan=1 colspan=1>Precision</td><td rowspan=1 colspan=1>Recall</td><td rowspan=1 colspan=1>F1</td><td rowspan=1 colspan=1>ROC-AUC</td></tr><tr><td rowspan=1 colspan=1>NP</td><td rowspan=1 colspan=1>SVM (RBF)</td><td rowspan=1 colspan=1>0.457</td><td rowspan=1 colspan=1>0.166</td><td rowspan=1 colspan=1>0.560</td><td rowspan=1 colspan=1>0.256</td><td rowspan=1 colspan=1>0.537</td></tr><tr><td rowspan=1 colspan=1>EF</td><td rowspan=1 colspan=1>LR</td><td rowspan=1 colspan=1>0.540</td><td rowspan=1 colspan=1>0.181</td><td rowspan=1 colspan=1>0.500</td><td rowspan=1 colspan=1>0.266</td><td rowspan=1 colspan=1>0.492</td></tr><tr><td rowspan=1 colspan=1>DC</td><td rowspan=1 colspan=1>AdaBoost</td><td rowspan=1 colspan=1>0.563</td><td rowspan=1 colspan=1>0.209</td><td rowspan=1 colspan=1>0.580</td><td rowspan=1 colspan=1>0.307</td><td rowspan=1 colspan=1>0.564</td></tr><tr><td rowspan=1 colspan=1>VA</td><td rowspan=1 colspan=1>AdaBoost</td><td rowspan=1 colspan=1>0.493</td><td rowspan=1 colspan=1>0.160</td><td rowspan=1 colspan=1>0.480</td><td rowspan=1 colspan=1>0.240</td><td rowspan=1 colspan=1>0.484</td></tr></table>

## B. Model 1: Ensembles

Here, we employed voting ensemble (VE) and stacking ensemble (SE) algorithms for 6GBO classification. Tables VI and VII contains several classification results following preprocessing, SMOTE-based imbalance reduction, and OOFTT. At the default TT of 0.5, numerous models attained commendable accuracy levels, especially within the AF and NP categories; yet, recall and F1-scores were rather low, signifying inadequate identification of the minority class due to the imbalance. Following TT, recall significantly enhanced across the majority of feature groups, with numerous models attaining recall values around or equal to 1.000.

The network FG exhibited the best predictive performance, with the VE attaining the largest ROC-AUC and PR-AUC of 0.585 and 0.224, respectively. Also, the SE for network parameters recorded the highest optimized F1-score (0.302). The environment and device features yielded a somewhat moderate performance, whereas the vision (SIFT) attributes exhibited relatively poor discriminative ability. The findings indicate that network-related characteristics are the primary contributors to the prediction of adaptive and dynamic 6GBO. Additionally, beyond the ML-based 5G beam selection analysis in Klautau, et al. [13], the ensembles (SE and VE) in our study achieved higher accuracies (Table VI). However, it is noteworthy that both studies were conducted using different datasets. Interestingly, the device attributes achieved the strongest F1-score (0.307) when GridSearchCV was employed.

TABLE VI  
CLASSIFICATION RESULTS AT DEFAULT THRESHOLD (0.5)
<table><tr><td rowspan=1 colspan=1>FG</td><td rowspan=1 colspan=1>Model</td><td rowspan=1 colspan=1>Acc</td><td rowspan=1 colspan=1>Pre</td><td rowspan=1 colspan=1>Rec</td><td rowspan=1 colspan=1>F1</td></tr><tr><td rowspan=1 colspan=1>AFAF</td><td rowspan=1 colspan=1>VESE</td><td rowspan=1 colspan=1>0.7800.810</td><td rowspan=1 colspan=1>0.2220.167</td><td rowspan=1 colspan=1>0.1180.029</td><td rowspan=1 colspan=1>0.1540.050</td></tr><tr><td rowspan=1 colspan=1>DCDC</td><td rowspan=1 colspan=1>SEVE</td><td rowspan=1 colspan=1>0.5300.170</td><td rowspan=1 colspan=1>0.1940.170</td><td rowspan=1 colspan=1>0.5591.000</td><td rowspan=1 colspan=1>0.2880.291</td></tr><tr><td rowspan=1 colspan=1>EFEF</td><td rowspan=1 colspan=1>SEVE</td><td rowspan=1 colspan=1>0.6900.580</td><td rowspan=1 colspan=1>0.1500.143</td><td rowspan=1 colspan=1>0.1760.294</td><td rowspan=1 colspan=1>0.1620.192</td></tr><tr><td rowspan=1 colspan=1>NPNP</td><td rowspan=1 colspan=1>SEVE</td><td rowspan=1 colspan=1>0.8000.745</td><td rowspan=1 colspan=1>0.3130.282</td><td rowspan=1 colspan=1>0.1470.324</td><td rowspan=1 colspan=1>0.2000.301</td></tr><tr><td rowspan=1 colspan=1>VAVA</td><td rowspan=1 colspan=1>VESE</td><td rowspan=1 colspan=1>0.4000.535</td><td rowspan=1 colspan=1>0.1230.127</td><td rowspan=1 colspan=1>0.4120.294</td><td rowspan=1 colspan=1>0.1890.177</td></tr></table>

TABLE VII

6GBO CLASSIFICATION RESULTS AFTER THRESHOLD TUNING
<table><tr><td rowspan=1 colspan=1>FG</td><td rowspan=1 colspan=1>Model</td><td rowspan=1 colspan=1>Acc</td><td rowspan=1 colspan=1>Pre</td><td rowspan=1 colspan=1>Rec</td><td rowspan=1 colspan=1>F1</td><td rowspan=1 colspan=1>ROC-AUC</td><td rowspan=1 colspan=1>PR-AUC</td></tr><tr><td rowspan=1 colspan=1>AFAF</td><td rowspan=1 colspan=1>VESE</td><td rowspan=1 colspan=1>0.1700.165</td><td rowspan=1 colspan=1>0.1700.162</td><td rowspan=1 colspan=1>1.0000.941</td><td rowspan=1 colspan=1>0.2910.277</td><td rowspan=1 colspan=1>0.4760.421</td><td rowspan=1 colspan=1>0.1810.156</td></tr><tr><td rowspan=1 colspan=1>DCDC</td><td rowspan=1 colspan=1>SEVE</td><td rowspan=1 colspan=1>0.1700.170</td><td rowspan=1 colspan=1>0.1700.170</td><td rowspan=1 colspan=1>1.0001.000</td><td rowspan=1 colspan=1>0.2910.291</td><td rowspan=1 colspan=1>0.4940.467</td><td rowspan=1 colspan=1>0.1740.168</td></tr><tr><td rowspan=1 colspan=1>EFEF</td><td rowspan=1 colspan=1>SEVE</td><td rowspan=1 colspan=1>0.1900.175</td><td rowspan=1 colspan=1>0.1700.168</td><td rowspan=1 colspan=1>0.9710.971</td><td rowspan=1 colspan=1>0.2890.286</td><td rowspan=1 colspan=1>0.5040.517</td><td rowspan=1 colspan=1>0.1710.187</td></tr><tr><td rowspan=1 colspan=1>NPNP</td><td rowspan=1 colspan=1>SEVE</td><td rowspan=1 colspan=1>0.4450.170</td><td rowspan=1 colspan=1>0.1920.170</td><td rowspan=1 colspan=1>0.7061.000</td><td rowspan=1 colspan=1>0.3020.291</td><td rowspan=1 colspan=1>0.5770.585</td><td rowspan=1 colspan=1>0.2160.224</td></tr><tr><td rowspan=1 colspan=1>VAVA</td><td rowspan=1 colspan=1>VESE</td><td rowspan=1 colspan=1>0.2100.235</td><td rowspan=1 colspan=1>0.1670.164</td><td rowspan=1 colspan=1>0.9120.853</td><td rowspan=1 colspan=1>0.2820.275</td><td rowspan=1 colspan=1>0.4110.408</td><td rowspan=1 colspan=1>0.1530.149</td></tr></table>

## C. Model 2: Performance Metrics Alone

The objective of Model 2 is to assess the extent to which post-hoc outcome factors independently forecast the optimized label. This model was implemented as a benchmark for validating leakage. The pre-optimization data were excluded in the experiment conducted here. The performance metrics include the following: Beamforming Gain, Latency, Energy Consumption, Throughput, Beam Training Time, SNR Improvement, Processing Time, and Memory Usage.

Table VIII summarizes the 6GBO findings derived just from the above-mentioned performance metrics. The evaluations included preprocessing and SMOTE-based imbalance management to assess the predictive strength of the post-optimization outcome on the optimized label. The findings demonstrated good classification performance across almost all ML models, with GB, AdaBoost, DT, HG, XGBoost, and LightGBM attaining excellent scores of 1.000 for all evaluation metrics. Likewise, RF attained nearly flawless performance, exhibiting an accuracy and ROC-AUC of 0.995 and 1.000, respectively. Even relatively worse models like KNN and logistic regression yielded impressive ROC-AUC values of 0.935 and 0.944, respectively.

The findings indicate a good correlation between the performance measures as well as the target label, thereby providing significant post-hoc information regarding beamforming performance. Consequently, employing these factors for real-time forecasting will artificially enhance classification performance and inadequately represent actual decision-making scenarios in adaptive 6GBO. To a large extent, our results thus confirm the necessity of limiting deployable prediction pipelines of 6GBO to pre-decision (pre-optimization) data. This distinction is essential for guaranteeing equitable evaluation and accurate assessment of intelligent 6GBO schemes in emerging wireless communication settings.

TABLE VIII  
BEST PERFORMING MODELS USING PERFORMANCE METRICS ONLY
<table><tr><td rowspan=1 colspan=1>Model</td><td rowspan=1 colspan=1>Accuracy</td><td rowspan=1 colspan=1>F1-Score</td><td rowspan=1 colspan=1>ROC-AUC</td><td rowspan=1 colspan=1>PR-AUC</td></tr><tr><td rowspan=1 colspan=1>GB</td><td rowspan=1 colspan=1>1.000</td><td rowspan=1 colspan=1>1.000</td><td rowspan=1 colspan=1>1.000</td><td rowspan=1 colspan=1>1.000</td></tr><tr><td rowspan=1 colspan=1>AdaBoost</td><td rowspan=1 colspan=1>1.000</td><td rowspan=1 colspan=1>1.000</td><td rowspan=1 colspan=1>1.000</td><td rowspan=1 colspan=1>1.000</td></tr><tr><td rowspan=1 colspan=1>Decision Tree</td><td rowspan=1 colspan=1>1.000</td><td rowspan=1 colspan=1>1.000</td><td rowspan=1 colspan=1>1.000</td><td rowspan=1 colspan=1>1.000</td></tr><tr><td rowspan=1 colspan=1>HGB</td><td rowspan=1 colspan=1>1.000</td><td rowspan=1 colspan=1>1.000</td><td rowspan=1 colspan=1>1.000</td><td rowspan=1 colspan=1>1.000</td></tr><tr><td rowspan=1 colspan=1>XGBoost</td><td rowspan=1 colspan=1>1.000</td><td rowspan=1 colspan=1>1.000</td><td rowspan=1 colspan=1>1.000</td><td rowspan=1 colspan=1>1.000</td></tr><tr><td rowspan=1 colspan=1>LightGBM</td><td rowspan=1 colspan=1>1.000</td><td rowspan=1 colspan=1>1.000</td><td rowspan=1 colspan=1>1.000</td><td rowspan=1 colspan=1>1.000</td></tr><tr><td rowspan=1 colspan=1>RF</td><td rowspan=1 colspan=1>0.995</td><td rowspan=1 colspan=1>0.985</td><td rowspan=1 colspan=1>1.000</td><td rowspan=1 colspan=1>1.000</td></tr><tr><td rowspan=1 colspan=1>Extra Trees</td><td rowspan=1 colspan=1>0.970</td><td rowspan=1 colspan=1>0.906</td><td rowspan=1 colspan=1>0.996</td><td rowspan=1 colspan=1>0.979</td></tr><tr><td rowspan=1 colspan=1>MLP</td><td rowspan=1 colspan=1>0.950</td><td rowspan=1 colspan=1>0.853</td><td rowspan=1 colspan=1>0.983</td><td rowspan=1 colspan=1>0.931</td></tr></table>

## D. Model 3: All Features (Model 1 + Model 2)

The objective of Model 3 is to assess the impact of integrating pre-decision attributes with post-optimization result metrics to examine the effects of target leakage. Table IX displays the classification outcomes achieved with the entire feature set, excluding SMOTE balancing and threshold adjustment. Multiple ensemble models, notably GB, AdaBoost, DT, XG-Boost, HGB, and LightGBM, attained excellent performance for every evaluation metric (1.000). Likewise, RF attained an exceptional result with accuracy and ROC-AUC of 0.985 and 1.000, respectively. Classical ML models, including LR and SVM (RBF), exhibited good prediction performance, attaining ROC-AUC values over 0.93.

While these results first imply a highly superior prediction of 6GBO, the perfect performance clearly suggests the existence of target leakage inside the feature space. The incorporation of post-hoc performance indicators furnishes the models with outcome-related data that is inherently linked to the optimized label. Our decision to remove SMOTE balancing and TT is to clearly show that this model achieved perfect performance without these imbalance-aware approaches. Thus, the classifiers are proficiently learning optimization results instead of forecasting circumstances based on actionable predecision parameters.

Like Sharif’s implementations [15], this model generated a significant enhancement in all metrics. Therefore, the findings also underscore the necessity of meticulously distinguishing pre-decision factors from post-optimization performance measurements in the development of smart beamforming systems. The entire feature configuration yields remarkably high classification values; nevertheless, these outcomes do not accurately represent practical deployment scenarios, as several included metrics are accessible only post-beamforming optimization. This discovery substantiates the requirement for the ablation technique presented above and endorses the utilization of predecision characteristics for equitable and functionally significant assessment of adaptive 6GBO designs.

TABLE IX  
ALL FEATURES WITHOUT SMOTE OR THRESHOLD TUNING
<table><tr><td rowspan=1 colspan=1>Model</td><td rowspan=1 colspan=1>Accuracy</td><td rowspan=1 colspan=1>F1-Score</td><td rowspan=1 colspan=1>ROC-AUC</td><td rowspan=1 colspan=1>PR-AUC</td></tr><tr><td rowspan=1 colspan=1>GB</td><td rowspan=1 colspan=1>1.000</td><td rowspan=1 colspan=1>1.000</td><td rowspan=1 colspan=1>1.000</td><td rowspan=1 colspan=1>1.000</td></tr><tr><td rowspan=1 colspan=1>AdaBoost</td><td rowspan=1 colspan=1>1.000</td><td rowspan=1 colspan=1>1.000</td><td rowspan=1 colspan=1>1.000</td><td rowspan=1 colspan=1>1.000</td></tr><tr><td rowspan=1 colspan=1>XGBoost</td><td rowspan=1 colspan=1>1.000</td><td rowspan=1 colspan=1>1.000</td><td rowspan=1 colspan=1>1.000</td><td rowspan=1 colspan=1>1.000</td></tr><tr><td rowspan=1 colspan=1>LightGBM</td><td rowspan=1 colspan=1>1.000</td><td rowspan=1 colspan=1>1.000</td><td rowspan=1 colspan=1>1.000</td><td rowspan=1 colspan=1>1.000</td></tr><tr><td rowspan=1 colspan=1>RF</td><td rowspan=1 colspan=1>0.985</td><td rowspan=1 colspan=1>0.954</td><td rowspan=1 colspan=1>1.000</td><td rowspan=1 colspan=1>1.000</td></tr><tr><td rowspan=1 colspan=1>SVM</td><td rowspan=1 colspan=1>0.905</td><td rowspan=1 colspan=1>0.732</td><td rowspan=1 colspan=1>0.960</td><td rowspan=1 colspan=1>0.862</td></tr><tr><td rowspan=1 colspan=1>LR</td><td rowspan=1 colspan=1>0.885</td><td rowspan=1 colspan=1>0.716</td><td rowspan=1 colspan=1>0.940</td><td rowspan=1 colspan=1>0.812</td></tr><tr><td rowspan=1 colspan=1>KNN</td><td rowspan=1 colspan=1>0.890</td><td rowspan=1 colspan=1>0.633</td><td rowspan=1 colspan=1>0.845</td><td rowspan=1 colspan=1>0.613</td></tr></table>

## E. Explainability using SHAP

The network FG performed better than other groups: AdaBoost (F1 = 0.327) (Table III); LightGBM (F1 = 0.327) (Table IV); and SVM (RBF) (F1 = 0.256) (Table V). However, we employed the AdaBoost model for explainability. To enhance the interpretability of the AdaBoost model for the network FG (F1 = 0.327), analyses of permutation importance (PI), modelspecific feature importance (MSFI), and SHAP global importance were conducted. The AdaBoost model performed relatively well in our analysis above. Among these synergistic explainability methods, codebook size, bandwidth, and frequency repeatedly surfaced as the most significant determinants of 6G beamforming enhancement. PI revealed Codebook Size (0.042) and Bandwidth (0.033) as the paramount attributes; however, AdaBoost’s MSFI prioritized Codebook Size (0.304), Transmit Power (0.286), and Bandwidth (0.207). Conversely, SHAP analysis for global importance, which measures the contribution of each feature to model predictions, revealed bandwidth (0.073), frequency (0.029), and codebook size (0.020) as the primary influences. The persistent significance of the network feature group through various explainability techniques suggests that codebook arrangement, bandwidth availability, and operating frequency are crucial in influencing 6GBO decisions.

For completeness, we added SHAP explainability for the device and environmental attributes. SHAP analysis was conducted to provide insight into the XGBoost models chosen for the DC (F1 = 0.291) and EF (F1 = 0.317) categories. In the DC explainability model (Fig. 3), the most significant feature was actually Device Type (IoT Sensor) with a SHAP value of 0.035, followed by Device Type (Smartphone) at 0.025 and Number of Antennas at 0.023, suggesting that device type exerts a more substantial influence on 6GBO than the number of antennas. In the EF explainability model (Fig. 4), Mobility was identified as the primary predictor (SHAP = 0.177), succeeded by Obstacle Density (0.099) and Environment Type (Indoor/Outdoor) (0.033). These results indicate that, in terms of XGBoost, user mobility and environmental factors have a more significant impact on device-specific traits than do device-specific traits, with mobility being the paramount environmental element.

![](images/8cc05d9d04b9166389c6163871ca5cb6e0507e68d3c56c499d7e0ce6fce93687.jpg)  
Fig. 1. Permutation Importance

![](images/6aee42ad833e4d6dd2d12e5754775c0e8c66bd3f290dd93d7de97f4d1bebbece.jpg)  
Fig. 2. Permutation Importance

The GB model utilizing the Model 2 (Performance Metrics) attained flawless classification results. PI, MSFI, and SHAP analysis 5 consistently recognized latency, throughput, and beamforming gain as the primary predictors, but the other performance metrics had minimal influence on model decisions. Latency demonstrated the greatest significance among all three techniques, succeeded by Throughput and Beamforming Gain. The nearly flawless prediction accuracy indicates a correlation between these variables and the optimization objective, serving as post-decision performance metrics, which results in target leakage when employed as predictive attributes. As a result, while performance metrics offer significant insights for retrospective analysis and network assessment, they are inadequate for real-time 6GB prediction, underscoring the necessity of relying solely on pre-decision (NP, EF, DC, VA) attributes in practical 6GBO frameworks.

![](images/27faaa8297ec67a2831a7718bf03944adc81ab915d5d4945a4011eff31eecf9a.jpg)  
Fig. 3. Global Importance for Device Characteristics

![](images/3e20214deacdeaf7efb9c444afa1cb91c14e94c11ff42de9c6f34ac6d4a49bbe.jpg)  
Fig. 4. Global Importance for Environmental Factors

## F. Clustering Network Scenarios

Here, unsupervised ML clustering approaches were employed to elucidate environmental and operational changes in 6GBO systems with the goal of aiming to identify various profiles of network scenarios. The analysis employed essential contextual characteristics, such as mobility, obstacle density, device type, and environment (outdoor), which jointly define evolving communication contexts and user scenarios. The clustering analysis utilizing performance indicators is designed exclusively for post-hoc (exploratory) study rather than for real-time 6GBO. This is because they can cause target leakage.

We applied three clustering methods—K-means, DBSCAN, and hierarchical clustering—to categorize network scenarios based on feature similarity and spatial density correlations. Prior to clustering, numerical features were standardized, and categorical/binary data were encoded using a preprocessing pipeline. Missing values were addressed with median imputation for numeric data and mode imputation for categorical data. Evaluation metrics, such as the Elbow method and Silhouette score (SS), were used to determine the optimal number of clusters (from 2 to 10 (Table 6)). Furthermore, we employed SelectKBest to identify 5 key performance features for network clustering, focusing on latency, throughput, beamforming gain, device type (smartphone), and SNR improvement to enhance cluster formation despite unsupervised clustering. Table X contains the feature importance for the initial set (contextual) and selected (performance-based) set.

K-Means Clusters (PCA-reduced 2D)  
![](images/b9d1e307a1be16d8e4dcce994d14ea4c534a2faff7fb6c8ce15b6a4577464bc2.jpg)

Fig. 5. Global Importance for Performance Metrics  
![](images/4e10ef26aa79040ef1602c1fce0ed6c3085ba16957d2c1616570574f2498a544.jpg)

![](images/606ce0c299169b7465e8ff183e040ee34673ab6d7f3f3e012c01d13c780e2468.jpg)  
Fig. 6. Elbow & SS Metrics for the Initial Features

The ideal number of clusters was calculated separately for each set (initial and selected) with the Elbow and SS techniques. For the initial set of features, k=4 was preferred (Fig. 6), as it aligned with the elbow point, preserved a commendably high SS, and produced comprehensible operational scenarios. The SS analysis for the SelectKBest-based feature set revealed that k=2 yielded the most pronounced cluster distinctness (Fig. 8); hence, it was used for the unsupervised analysis.

Principal Component Analysis (PCA) was utilized to reduce the preprocessed feature space to two dimensions for better visualization of K-Means clusters (Figs. 7 and 9), revealing distinct patterns in contextual features (mobility, deployment conditions, obstacle density, and IoT sensor presence) and performance-related features (latency, throughput, beamforming gain, device type (smartphone), and SNR improvement) within the 6GBO dataset.

![](images/2e1dd29acfad00cc391240f17fcacc768afade8aec27058eb95bc1f73322ecba.jpg)

Fig. 7. PCA Illustration for the Initial Features  
![](images/eea54e03f6fb0acd791a657de773f8922a8eba89cd39fb188fea576b333c8c5c.jpg)

![](images/e2eef819d0ff78862a914e5fdb94a5d7c16c8230af5c0c2bfde4af835c91cbe1.jpg)  
Fig. 8. Elbow & SS Metrics for the Selected Features

Table X juxtaposes the clustering results of K-Means, DB-SCAN, and Hierarchical Clustering, utilizing both feature sets. K-Means and hierarchical clustering demonstrated superior performance for the initial features, attaining an SS of 0.352 at k=4, whereas the SelectKBest-based features yielded a moderately distinguishable 2-cluster arrangement at k=2 with an SS of 0.221. These results suggest that the performance-based features inherently created two overarching behavioral groups, while the original feature set delineates a more intricate 4- cluster framework. With the performance-based parameters, DBSCAN failed to identify any clusters, categorizing each of the 1000 data points as noise. This indicates that density-based clustering may be inappropriate for this dataset or necessitates extensive parameter adjustment.

In the PCA diagram (Fig. 7), context features are delineated distinctly, while in Table XII, we summarized clusters against obstacle density (OD), mobility (M), environment outdoor (OU), device type IoT sensor (DTI), and interpretation. The examination of the cluster centroids identified four unique deployment scenarios, largely distinguished by environmental type and device category. The clusters precisely corresponded to (cluster 0) outdoor non-IoT devices, (cluster 1) indoor non-

K-Means Clusters (PCA-reduced 2D)  
![](images/ee1387e9f156d8faffec9d86a35c1fab588881260506507f38456c73dd130bb2.jpg)  
Fig. 9. PCA Illustration for the Selected Features

TABLE X  
FEATURE IMPORTANCE COMPARISON FOR CLUSTERING ANALYSIS
<table><tr><td rowspan=1 colspan=1>Feature</td><td rowspan=1 colspan=1>Importance</td><td rowspan=1 colspan=1>Feature Set</td></tr><tr><td rowspan=1 colspan=1>Mobility (m/s)Environment_OutdoorDevice Type_IoT SensorObstacle Density</td><td rowspan=1 colspan=1>1.180.810.560.00</td><td rowspan=1 colspan=1>Initial FeaturesInitial FeaturesInitial FeaturesInitial Features</td></tr><tr><td rowspan=1 colspan=1>Latency (ms)Throughput (Mbps)Beamforming Gain (dB)Device Type_SmartphoneSNR Improvement (dB)</td><td rowspan=1 colspan=1>212.27153.3872.672.331.59</td><td rowspan=1 colspan=1>Selected FeaturesSelected FeaturesSelected FeaturesSelected FeaturesSelected Features</td></tr></table>

IoT devices, (cluster 2) indoor IoT sensors, and (cluster 3) outdoor IoT sensors. Conversely, OD and mobility demonstrated minimal fluctuations among clusters, indicating that the clustering pattern was primarily influenced by the installation environment and type of device instead of mobility-based attributes.

Obstacle density, while theoretically pertinent to mmWave obstruction, was uninformative in this dataset. Its dynamic range may be insufficient, inaccurately quantified, or associated with other variables that have already accounted for its impact. Subsequent data collection should either expand its scope or employ more sophisticated blockage proxies, such as LiDAR-derived blockage probabilities or map-driven line of sight. The unsupervised structure within the dataset is predominantly contextual as opposed to performance-oriented.

Segmenting initially by environment and device type produces stable, interpretable cohorts; within each cohort, performance exhibits smooth variation and is more effectively modeled using regression or soft clustering techniques. Context serves as the primary determinant of separable network/user states, whereas KPI variables constitute a continuum that is unable to inherently divide into discrete clusters. This conclusion indicates that contextual considerations may be more significant than motion in differentiating operational beamforming situations within the 6GBO dataset.

TABLE XI  
CLUSTERING PERFORMANCE COMPARISON
<table><tr><td>S/No</td><td>Algorithm</td><td>K</td><td>SS</td><td>DBI</td></tr><tr><td>0</td><td>K-Means</td><td>4</td><td>0.352</td><td>1.320</td></tr><tr><td>1</td><td>Hierarchical</td><td>4</td><td>0.352</td><td>1.320</td></tr><tr><td>2</td><td>DBSCAN</td><td>4</td><td>0.313</td><td>1.204</td></tr><tr><td>3</td><td>K-Means (Selected Features)</td><td>2</td><td>0.221</td><td>1.846</td></tr><tr><td>4</td><td>Hierarchical (Selected Features)</td><td>2</td><td>0.221</td><td>1.846</td></tr></table>

TABLE XII  
CLUSTER PROFILES IDENTIFIED FROM CONTEXT FEATURES
<table><tr><td rowspan=1 colspan=1>Cluster</td><td rowspan=1 colspan=1>OD</td><td rowspan=1 colspan=1>M</td><td rowspan=1 colspan=1>OU</td><td rowspan=1 colspan=1>DTI</td><td rowspan=1 colspan=1>Interpretation</td></tr><tr><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>5.050</td><td rowspan=1 colspan=1>1.547</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>Outdoor, non-IoT Scenario</td></tr><tr><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>4.868</td><td rowspan=1 colspan=1>1.477</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>Indoor, non-IoT Scenario</td></tr><tr><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>4.771</td><td rowspan=1 colspan=1>1.500</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>Indoor, IoT Sensor Scenario</td></tr><tr><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>4.863</td><td rowspan=1 colspan=1>1.415</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>Outdoor, IoT Sensor Scenario</td></tr></table>

Fig. 9 illustrates the PCA visualization of the resultant clusters, demonstrating two moderately distinct groupings with a degree of overlap in the condensed 2-dim space. The identified overlap indicates that the SelectKBest-based features demonstrate gradual shifts between operational states instead of entirely separate classifications. Nonetheless, the clustering reveals that these performance-oriented and devicespecific characteristics inherently divide into two overarching behavioral profiles, offering a comprehensible depiction of network operating conditions. This outcome aligns with the SS analysis, which showed k=2 to be the ideal number of clusters for the chosen feature set.

In the PCA diagram (Fig. 9), the performance-related features are moderately distinguishable using clusters 0 and 1. We observed two moderately distinguishable operating profiles (OPs), and their differences were derived from the abovementioned selected features (Table XIII). Cluster 0 shows a non-smartphone profile, defined by reduced latency (5.41 ms), elevated throughput (549.28 Mbps), enhanced beamforming gain (17.80 dB), and superior SNR enhancement (12.69 dB). Conversely, Cluster 1 aligns with a smartphone-centric profile, demonstrating marginally elevated latency (5.47 ms), diminished throughput (526.36 Mbps), worse beamforming gain (17.19 dB), and a lesser SNR enhancement (12.42 dB). Note that the device (smartphone)-type split is clean (0.00 vs. 1.00). These results suggest that the chosen performance and device-specific attributes inherently divide the 6G network into two OPs with modest differences, offering a comprehensible post-hoc analysis of beamforming optimization performance.

Finally, it is noteworthy that TT significantly enhanced recall, with multiple models attaining values near 1.000, signifying that almost all 6GBO prospects were identified. Nonetheless, this enhancement was achieved at the cost of accuracy, which remained comparatively low, around 0.17–0.19. This indicates that numerous samples identified as needing optimization were incorrect positives, leading to an elevated false-alarm rate. In adaptive beamforming, such erroneous detections could lead the network to inappropriately initiate beam reconfiguration and training, codebook searches, or other optimization processes for links that do not need modification or adjustment [17]. These additional beam updates may elevate computational demands, energy usage, signaling traffic, and processing delays, thereby diminishing overall system efficiency [17]. In extremely dynamic 6G settings, emphasizing recall could be advantageous when the expense of overlooking a legitimate 6GBO opportunity surpasses that of executing an unwarranted update. A false negative can result in diminished link quality, lowered throughput, and heightened communication latency. Thus, the criterion for decision-making ought to be contingent upon the application and must weigh the operational expenses of superfluous beam adjustments against the potential for missed 6GBO prospects.

TABLE XIII  
CLUSTERING PROFILES FROM PERFORMANCE-BASED FEATURES
<table><tr><td>Selected Features</td><td>Cluster 0</td><td>Cluster 1</td><td>Difference</td></tr><tr><td>Latency (ms)</td><td>5.41</td><td>5.47</td><td>-0.06</td></tr><tr><td>Throughput (Mbps)</td><td>549.28</td><td>526.36</td><td>+22.92</td></tr><tr><td>Beamforming Gain (dB)</td><td>17.80</td><td>17.19</td><td>+0.61</td></tr><tr><td>Device Type (Smartphone) (0/1)</td><td>0.00</td><td>1.00</td><td></td></tr><tr><td>SNR Improvement (dB)</td><td>12.69</td><td>12.42</td><td>+0.27</td></tr></table>

## V. CONCLUSION

This paper addresses practical, systems-level questions that are of interest to researchers and practitioners working on edge intelligence and beam management in mmWave/6G deployments. Specifically, the study presents a systematic ML study of beamforming optimization for 6G-IoT, comparing the predictive power of network, environmental, and device feature groups and demonstrating how ML pipelines (with explainability and imbalance handling) can guide adaptive, low-latency beam management. Generally, the network parameters have stronger F1-score and ROC-AUC/PR-AUC values. However, the device attributes achieved the strongest F1-score when GridSearchCV was employed. Despite the substantial enhancement in detecting optimized 6G beamforming scenarios through threshold tuning, the resultant rise in false positives underscores the necessity for tailored threshold preference that reconciles detection efficacy with operational demands. In the future, we would compare pre-decision feature groups against a simple majority-class baseline and a calibrated probability baseline. Also, our comparative analysis would extend to predicting beamforming gain, latency, energy consumption, throughput, and SNR improvement using deep and reinforcement learning techniques.

## VI. ACKNOWLEDGMENT

This work was undertaken thanks in part to funding from the Connected Minds Program, supported by the Canada First Research Excellence Fund, Grant #CFREF-2022-00010.

## REFERENCES

[1] Z. Qadir, K. N. Le, N. Saeed, and H. S. Munawar, “Towards 6g internet of things: Recent advances, use cases, and open challenges,” ICT Express, vol. 9, no. 3, pp. 296–312, 2023.

[2] M. S. Akbar, Z. Hussain, M. Ikram, Q. Z. Sheng, and S. C. Mukhopadhyay, “On challenges of sixth-generation (6g) wireless networks: A comprehensive survey of requirements, applications, and security issues,” Journal of Network and Computer Applications, vol. 233, p. 104040, 2025.

[3] J. Benesty, I. Cohen, and J. Chen, A Brief Overview of Conventional Beamforming, pp. 13–21. Cham: Springer International Publishing, 2021.

[4] W. Saad, M. Bennis, and M. Chen, “A vision of 6g wireless systems: Applications, trends, technologies, and open research problems,” IEEE network, vol. 34, no. 3, pp. 134–142, 2019.

[5] M. Giordani, M. Polese, M. Mezzavilla, S. Rangan, and M. Zorzi, “Toward 6g networks: Use cases and technologies,” IEEE communications magazine, vol. 58, no. 3, pp. 55–61, 2020.

[6] T. Q. Duong, S. R. Khosravirad, C. She, P. Popovski, M. Bennis, and T. Q. Quek, Ultra-reliable and Low-Latency Communications (URLLC) theory and practice: Advances in 5G and beyond. John Wiley & Sons, 2023.

[7] R. W. Heath, N. Gonzalez-Prelcic, S. Rangan, W. Roh, and A. M. Sayeed, “An overview of signal processing techniques for millimeter wave mimo systems,” IEEE journal of selected topics in signal processing, vol. 10, no. 3, pp. 436–453, 2016.

[8] A. M. Elbir, K. V. Mishra, S. A. Vorobyov, and R. W. Heath, “Twentyfive years of advances in beamforming: From convex and nonconvex optimization to learning techniques,” IEEE Signal Processing Magazine, vol. 40, no. 4, pp. 118–131, 2023.

[9] H. Huang, J. Yang, H. Huang, Y. Song, and G. Gui, “Deep learning for super-resolution channel estimation and doa estimation based massive mimo system,” IEEE Transactions on Vehicular Technology, vol. 67, no. 9, pp. 8549–8560, 2018.

[10] L. Zhou, H. Yin, H. Zhao, J. Wei, D. Hu, and V. C. Leung, “A comprehensive survey of artificial intelligence applications in uavenabled wireless networks,” Digital Communications and Networks, vol. 12, no. 4, pp. 561–583, 2026.

[11] Z. Qin, H. Ye, G. Y. Li, and B.-H. F. Juang, “Deep learning in physical layer communications,” IEEE Wireless Communications, vol. 26, no. 2, pp. 93–99, 2019.

[12] H. Ye, G. Y. Li, and B.-H. F. Juang, “Deep reinforcement learning based resource allocation for v2v communications,” IEEE Transactions on Vehicular Technology, vol. 68, no. 4, pp. 3163–3173, 2019.

[13] A. Klautau, P. Batista, N. Gonzalez-Prelcic, Y. Wang, and R. W. Heath,´ “5g mimo data for machine learning: Application to beam-selection using deep learning,” in 2018 Information Theory and Applications Workshop (ITA), pp. 1–9, 2018.

[14] D. d. S. Brilhante, J. C. Manjarres, R. Moreira, L. de Oliveira Veiga, J. F. de Rezende, F. Muller, A. Klautau, L. Leonel Mendes, and F. A.¨ P. de Figueiredo, “A literature survey on ai-aided beamforming and beam management for 5g and 6g systems,” Sensors, vol. 23, no. 9, p. 4359, 2023.

[15] R. S. Sharif, “Optimized beamforming 6g xai.” https://github.com/ raadsr15/Optimized-Beamforming-6G-XAI, 2025. Accessed: Jul. 20, 2026.

[16] M. Ziya, “6g iot intelligent management dataset: Computer vision-assisted intelligent beamforming management in 6g networks.” https://www.kaggle.com/datasets/ziya07/ 6g-iot-intelligent-management-dataset, 2024. Accessed: 2025-10- 21.

[17] M. Giordani, M. Polese, A. Roy, D. Castor, and M. Zorzi, “A tutorial on beam management for 3gpp nr at mmwave frequencies,” IEEE Communications Surveys & Tutorials, vol. 21, no. 1, pp. 173–196, 2018.