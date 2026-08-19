# Conformal Prediction for Molecular Properties under Label Shift

Hyeonsu Lee Mogam Institute for Biomedical Research hyeonsu.lee@mogam.re.kr

Erkhembayar Jadamba Mogam Institute for Biomedical Research erkhembayar.jadamba@mogam.re.kr

Juyeon Kim Mogam Institute for Biomedical Research juyeon.kim@mogam.re.kr

Seungjin Choi Intellicode seungjin.choi.mlg@gmail.com

Hyunjin Shin Mogam Institute for Biomedical Research hyungjin.shin@mogam.re.kr

## Abstract

Drug discovery and development underpins healthcare but remains costly and failure-prone. A critical bottleneck lies in predicting molecular properties such as solubility, potency, and toxicity, which directly determine whether a candidate can advance from preclinical to clinical trials. Artificial Intelligence (AI) has accelerated this process, yet its reliability is often undermined by distribution shift, as experimental conditions frequently diverge from training data. In addition, conventional point predictions provide only single-value estimates, offering limited guidance for high-stakes experimental design. We address these challenges with a conformal prediction framework tailored to label shift. By weighting conformal scores using marginal label probability ratios, our method produces statistically rigorous prediction intervals without retraining. This enables robust uncertainty quantification even when property distributions drift, directly tackling one of the most pervasive obstacles to applying AI in real-world drug development. By moving beyond accuracy alone to provide actionable confidence measures, our approach enhances the trustworthiness of AI-driven predictions. This further aligns predictive modeling with regulatory demands for transparency and uncertainty reporting and ultimately supports more reliable decision-making in billion-dollar development pipelines.

## 1 Introduction

Drug discovery and development is characterized by prolonged timelines, substantial resource requirements, and a high likelihood of failure. More specifically, developing and bringing a single new drug to market can cost from \$314 million to \$2.8 billion and take over a decade, but failure rates can reach up to larger than 90% across the entire development life cycle [1]. This remarkable inefficiency underscores the growing importance of artificial intelligence (AI) because it can substantially reduce the need for chemical and biological experiments by making predictions on molecular properties such as solubility, bioavailability, or toxicity. However, the current performance of AI for this purpose does not appear to be fully compelling to drug development specialists. A key component contributing to this insufficient performance is the uncertainty embedded in prediction by AI, mainly originating from data characterization and model training. In response to this limitation, recent FDA guidance (2025) for artificial intelligence in drug and device development explicitly requires that AI systems should provide “appropriate confidence intervals” and “uncertainty estimates” when supporting regulatory submissions [2, 3]. This requirement indicates that actively considering the uncertainty, thereby improving the reliable predictions, will be a critical part of AI applications to drug discovery and development.

The reliability of AI predictions is often undermined by the problem of distribution shift [4], a scenario where test data differ substantially from the training data. This is particularly an urgent issue in drug discovery, where novel compounds frequently occupy chemical spaces unseen during training. The consequence is often summarized as overconfident yet unreliable predictions, which is an unacceptable risk when billions of dollars and patient outcomes are at stake.

To address this gap, reliable estimation of uncertainty is essential. Conformal prediction, also known as conformal inference, is a versatile and statistically principled framework that constructs prediction intervals around model outputs [5, 6, 7]. Its foremost advantage is its distribution-free and finite-sample validity, which guarantees that prediction intervals will contain the true label with a user-specified probability (e.g., 90%), regardless of dataset size or underlying distributional assumptions. This property represents a major improvement over many traditional statistical methods [7, 8]. Originally pioneered by Vladimir Vovk and his colleagues in the 1990s, the core mechanism involves a simple calibration step where a small holdout dataset is used to convert an arbitrary heuristic notion of uncertainty from a pre-trained model into a rigorous one, typically by computing conformal scores and their empirical quantiles [5, 7, 9].

This methodology is broadly applicable across various machine learning tasks, ranging from image classification to regression. It has also been significantly extended to address complex real-world challenges such as covariate shift, distribution drift, and the control of general risks. As a result, it has become an indispensable tool for reliable uncertainty quantification in high-stakes applications [6, 7]. In drug discovery, this unreliability often stems from two specific types of distribution shift: covariate shift [6] and label shift [10]. Covariate shift occurs when the distribution of molecular structures $( P ( x ) )$ changes between training and test. For instance, in drug discovery, covariate shift is observed when a model trained on diverse chemical libraries is applied to a new and more specialized set of molecules.

On the other hand, label shift, the primary target of this work, arises when the distribution of the target property $( P ( y ) )$ changes, while the conditional distribution of features given the label $P ( x \mid y )$ remains invariant [11, 12, 13]. This situation commonly arises when research priorities shift toward discovering molecules with property values underrepresented in the original training data, such as compounds exhibiting exceptionally high potency or low toxicity. Although various machine learning solutions have been developed to address distribution shifts [6, 10, 14, 15, 16] label shift remains relatively underexplored, particularly in continuous regression-based tasks that are frequently encountered in molecular property prediction[11, 13, 17].

Addressing issues related to distribution shifts is becoming increasingly important as more sophisticated AI models, such as large language models (LLMs) pretrained on large-scale chemical databases [18, 19, 20], are applied to explore the complex relationships between molecular structure and function. However, designing novel molecules with these AI models inherently requires highly accurate predictions under distribution shift. In general, AI models trained under the assumption of identically distributed data often fail to account for label shift, as this assumption typically leads to overconfident yet incorrect single-value predictions for novel and unseen molecules. This underscores the necessity of uncertainty quantification, and highlights that the estimated uncertainty must be integrated with AI predictions to produce realistic and robust prediction intervals, even for state-of-the-art LLM-based AI models.

In response to these challenges, we propose a new framework that generates reliable prediction intervals for molecular properties, even under significant label shift. Our method builds on conformal prediction, a machine learning technique that provides distribution-free and finite-sample guarantees on prediction intervals [6, 21]. Standard conformal prediction assumes exchangeability between training and test data, an assumption violated under label shift. To address this issue, we develop a scheme based on weighted conformal prediction. In our framework, the corrective weights are derived from the ratio of the target to the source label distributions, which we estimate using versatiel approaches such as black box shift estimation (BBSE) [11], regularized learning under label shifts (RLLS) [12], and maximum likelihood estimation (MLE) [13].These techniques enhance the practicality of our approach, as the label shift can be directly estimated from the outputs of both unbiased and biased predictive models.

In conclusion, our method effectively mitigates the adverse effects of label shift without requiring costly model retraining. It generates statistically rigorous prediction intervals that adapt to changing property distributions, thereby providing a more realistic assessment of a molecule’s potential. Overall, this work makes a key contribution to the development of reliable AI for drug discovery by offering a robust methodology that ensures models remain trustworthy when navigating the uncertain frontiers of novel chemical space.s

## 2 Methods

The overall design of our framework is illustrated in Figure 1. The pipeline consists of the following steps: (i) the base prediction model is trained using source training set to perform predictions in the label shift environment, (ii) to quantify label shift, importance weights, which represent the ratio of the target domain’s marginal label distribution to the source domain’s marginal label distribution, are estimated using weight set through methods such as BBSE, RLLS, and MLE, (iii) nonconformity scores, such as absolute residuals, are computed for each data point in calibration set based on the predictions of the trained model and their actual labels, and (iv) the weighted quantile of the nonconformity scores is calculated by incorporating the estimated importance weights, which is then used to construct statistically valid prediction intervals under label shift for new test points. This high-level schema highlights how our approach adapts standard conformal prediction to remain valid under label shift.

![](images/a0b5962eaab4a7daf2ab9def49850994aaa85de6d0209bc3df8d24e0c938d961.jpg)  
Figure 1. Schematic diagram of conformal prediction for molecular properties under label shift.

We now formalize our approach for conformal prediction under label shift.

## 2.1 Problem Formulation

Let the source data be $\mathcal { D } _ { \mathrm { s } } = \{ ( x _ { i } , y _ { i } ) \} _ { i = 1 } ^ { n }$ , where $x _ { i } \in \mathcal X$ is a molecular representation and $y _ { i } \in \mathbb { R }$ is the continuous property of interest. These data points are drawn from a distribution $p ( x , y )$ . We are also given a set of unlabeled data from a target domain, $\mathcal { D } _ { \mathfrak { t } } = \{ x _ { j } \} _ { j = n + 1 } ^ { n + m } \mathrm { . }$ , drawn from a different distribution $q ( x , y )$

The label shift assumption posits that the conditional distribution of features given the label remains constant across domains, while the marginal label distribution changes:

$$
p ( x | y ) = q ( x | y ) \quad { \mathrm { a n d } } \quad p ( y ) \neq q ( y )\tag{1}
$$

Our goal is to construct a prediction interval, $C ( x _ { \mathrm { t e s t } } )$ , for a new test input $x _ { \mathrm { t e s t } }$ from the target domain that satisfies the marginal coverage guarantee at a desired confidence level $1 - \alpha$

$$
\mathbb { P } ( y _ { \mathrm { t e s t } } \in C ( x _ { \mathrm { t e s t } } ) ) \geq 1 - \alpha\tag{2}
$$

## 2.2 Binning and Importance Weight Estimation

To apply classification-based shift estimation techniques, we first discretize the continuous response variable y into K bins, creating a pseudo-label $\tilde { y } = \bar { \mathsf { b i n } } ( y ) \in \{ 0 , 1 , \cdots , K - 1 \}$ These pseudo-labels are used only for estimating weights. The discretization was performed using equally sized bins, and the bin range was determined by the minimum and the maximum values of the data used to calculate the marginal probability ratios.

The importance weight for each bin is defined as the ratio of the target and source pseudo-label probabilities:

$$
w ( \widetilde { y } ) = \frac { q ( \widetilde { y } ) } { p ( \widetilde { y } ) }\tag{3}
$$

In practice, the source probabilities $p ( \widetilde { y } )$ are calculated from the empirical frequencies in a held-out portion of the source data. The target probabilities $q ( \widetilde { y } )$ are estimated from the unlabeled target data using methods like BBSE, RLLS, or MLE, which leverage the outputs of a model trained on the binned source data.

Since MLE could not directly estimate $q ( \widetilde { y } )$ from predictions, we adopted a probabilistic approach. For each sample, we modeled a Gaussian centered at the prediction with standard deviation equal to the root mean squared error (RMSE) from the weight set. The probability of the sample falling into a bin was then given by the cumulative distribution function (CDF) difference at the bin’s bounds. Any negative probabilities arising from numerical errors were set to zero, and the resulting probability vector was normalized to ensure that its elements summed to one.

## 2.3 Weighted Conformal Prediction under Label Shift

To ensure the statistical validity of our method, we partition the source data $\mathcal { D } _ { s }$ into three disjoint subsets, preventing data leakage between steps:

1. Proper training set $\left( \mathcal { D } _ { \mathrm { t r a i n } } \right)$ : Used to train the base prediction model, $f .$

2. Weights set $( { \mathcal { D } } _ { \mathrm { w e i g h t } } ) { \mathrm { : } }$ : Used to estimate the label shift importance weights.

3. Calibration set $( \mathcal { D } _ { \mathrm { c a l } } ) \colon$ Used to compute nonconformity scores and calibrate the prediction intervals.

The weighted conformal prediction algorithm then proceeds as follows:

1. For each point $( x _ { i } , y _ { i } )$ in the calibration set $\mathcal { D } _ { \mathrm { c a l } } .$ , compute a nonconformity score. For regression, this is typically the absolute residual:

$$
s _ { i } = | y _ { i } - f ( x _ { i } ) |\tag{4}
$$

2. Assign the corresponding estimated importance weight $\widehat { w } _ { i } = \widehat { w } ( \widetilde { y } _ { i } )$ to each score $s _ { i } .$ , where $\widetilde { y } _ { i }$ is the bin of the true label $y _ { i }$

3. Compute the weighted quantile $\widehat { q } _ { w }$ from the set of scores $\left\{ s _ { i } \right\}$ and weights $\{ \widehat { w } _ { i } \}$ . This quantile is the value that satisfies:

$$
\widehat { q } _ { w } = \operatorname* { i n f } \left\{ s : \frac { \sum _ { i = 1 } ^ { n _ { c a l } } \widehat { w } _ { i } \cdot \mathbb { I } \{ s _ { i } \leq s \} } { \sum _ { j = 1 } ^ { n _ { c a l } } \widehat { w } _ { j } } \geq 1 - \alpha \right\}\tag{5}
$$

4. For a new test point $x _ { \mathrm { t } } .$ , the final prediction interval is formed by centering the weighted quantile around the model’s point prediction:

$$
C ( x _ { \mathrm { t } } ) = [ f ( x _ { \mathrm { t } } ) - \widehat { q } _ { w } , \quad f ( x _ { \mathrm { t } } ) + \widehat { q } _ { w } ]\tag{6}
$$

By using this weighted quantile, the method corrects for the distributional shift and restores the marginal coverage guarantee under the label shift assumption.

## 3 Experimental Settings

## 3.1 TDC Solubility AqsolDB

Solubility AqSolDB [22] from the Therapeutics Data Commons (TDC) [23], which provides measurements of compound solubility in aqueous solutions. This dataset serves as a benchmark for studying molecular physicochemical properties and for developing predictive models of drug solubility. AqSolDB specifically provides solubility information, which is a critical factor in drug design and delivery systems, and consists of 9,982 compounds. For each compound, experimentally measured logarithmic solubility values $( l o g S )$ and molecular structure information are included.

## 3.2 Chemical Large Language Model Finetuning

The large language model (LLM) employed in this study is based on a BART [24] architecture and has been optimized for the analysis of chemical data. The model was pretrained utilizing approximately 200 million unlabeled SMILES (Simplified Molecular Input Line Entry System) [25] data collected from Chembl [26], PubChem [27], ZINC [28], Enamine [29], Coconut [30], and Drugbank [31]. Through this pretraining, the model acquired enriched representations specific to the chemical domain, encompassing approximately 250 million parameters. The fine-tuning process was conducted via full fine-tuning of the pretrained LLM.

## 3.3 Data Splitting

We conducted a conformal prediction simulation utilizing split conformal prediction methods to address continuous label shift. The objective of this simulation was to ensure that the marginal probability ratio-based weights are "exchangeable" between the nonconformity score distributions of the training and test datasets when the label distribution Y differs between them. By guaranteeing this exchangeability, the constructed prediction intervals satisfy a minimum coverage of $1 - \alpha$ in a distribution-free manner.

The experiment was repeated 1,000 times, and in each iteration, the original data is divided into two subsets: source data $\mathcal { D } _ { \mathrm { s } }$ and target data $\mathcal { D } _ { \mathrm { t } } .$ . The two subsets are split in a 60% to 40% ratio of the total data. Here, $\mathcal { D } _ { \mathrm { s } }$ is divided into three subsets $( \mathcal { D } _ { \mathrm { t r a i n } } , \mathcal { D } _ { \mathrm { w e i g h t } } , \bar { \mathcal { D } } _ { \mathrm { c a l } } )$ of equal size. $\mathcal { D } _ { \mathrm { t } }$ is split into two subsets $( \mathcal { D } _ { \mathrm { n o } }$ <sub>\_shift</sub>, $\mathcal { D } _ { \mathrm { s h i f t } } )$ to evaluate coverage performance under label shift conditions and without label shift. $\mathcal { D } _ { \mathrm { n o . } }$ represented 50% of $\mathcal { D } _ { \mathrm { t } }$ and corresponded to test data without label shift. $\mathcal { D } _ { \mathrm { s h i f t } }$ is generated by sampling with replacement from $\mathcal { D } _ { \mathrm { t } } .$ , excluding $\mathcal { D } _ { \mathrm { n o \_ s h i f t } }$ . During this sampling process, the probability of selecting each data point was proportional to a specific weight, where $\mathbf { \bar { \boldsymbol { w } } } ( y ) = \exp ( \bar { y } ^ { T } \beta )$ . These weights were assigned based on the magnitude of $y .$

## 4 Experimental Results

Split conformal prediction fails under label shift As expected, the traditional split conformal prediction method exhibited a significant decline in coverage performance on the label-shifted test set compared to the non-shifted test set. In Figure 2, the coverage distribution for the shifted data (red) is notably shifted to the left relative to the distribution for the non-shifted data (gray), with the average coverage falling substantially below the nominal target. These findings highlight the limitations and unreliability of standard uncertainty quantification methods under label shift conditions.

Ratios of marginal probabilities recover coverage loss from label shift Our proposed method, which integrates weighted conformal prediction (WCP) with marginal probability ratios estimated via BBSE, RLLS, and MLE with bias-corrected temperature scaling (BCTS), provides statistically valid and reliable prediction intervals for addressing continuous label shift problems. The experimental results demonstrate that the proposed approach achieves improved predictive coverage compared to the traditional split conformal prediction (CP) method. Specifically, WCP consistently outperformed CP in terms of average coverage, as evidenced by the coverage distribution shown in Figure 3. The coverage distribution of WCP shifted to the right relative to CP, indicating that WCP more frequently generated prediction intervals that included the true labels. This allowed WCP to effectively achieve the desired coverage level, even under label shift conditions. These results provide empirical validation that weighted conformal prediction, when its weights are derived from marginal probability ratios, can produce more robust and reliable prediction intervals. This approach proves especially effective in addressing challenges presented by label shift scenarios.

![](images/6a401132d8b9f97235ee6d45ba27a880ef6bfc2e8280f0e891f6a49989e697b1.jpg)  
Figure 2. The KDE distributions for 1000 repeated experiments using standard split conformal prediction and the proposed methods are shown. The triangular markers on the x-axis correspond to the mean value of each distribution. The gray and red lines indicate the coverage distributions obtained by applying standard split conformal prediction to the test data without and with label shift (non-uniformly subsampled), respectively.

![](images/cb33fac865b93189973284d6c827a65c07fc0ab88747b4591161fbca170856c8.jpg)  
Figure 3. The KDE distributions for 1000 repeated experiments using standard split conformal prediction and the proposed methods are shown. The triangular markers on the x-axis correspond to the mean value of each distribution. Coverage distributions are shown for the test data under label shift: the red line corresponds to standard split conformal prediction, while the purple, green, and light blue lines correspond to conformal prediction with weights calculated via BBSE, RLLS, and MLE (BCTS), respectively.

Approaches for estimating marginal probability ratios When comparing the performance of BBSE, RLLS, and MLE, we observed that MLE achieved the highest coverage, followed by RLLS and then BBSE (Figure 3). Moreover, MLE showed robust performance in coverage recovery with respect to the number of bins (Table 2). This improvement can be explained by two factors: first, the use of bias-corrected calibration reduces systematic bias across classes; and second, the MLE algorithm benefits from a theoretical guarantee of convergence to a global optimum [13]. Nevertheless, this improvement in coverage came with certain trade-offs. The prediction intervals generated by MLE were generally wider (Figure 4), whereas BBSE and RLLS produced relatively narrower intervals.

## 5 Limitation

Our approach requires splitting source data into training, weighting, and calibration sets, which can reduce effective training size and hurt performance in low-sample regimes. Data augmentation methods [32, 33] may mitigate this. Moreover, standard conformal intervals are suboptimal for

![](images/6813543b75210b4a39212994ff6f1ba6a03b479aebd5613ff7274a85a7ab5059.jpg)  
Figure 4. The KDE distributions for 1000 repeated experiments using standard split conformal prediction and the proposed methods are shown. The triangular markers on the x-axis correspond to the mean value of each distribution. Interval length distributions are shown for the test data under label shift: the red line corresponds to standard split conformal prediction, while the purple, green, and light blue lines correspond to conformal prediction with weights calculated via BBSE, RLLS, and MLE (BCTS), respectively.

heteroscedastic data; techniques such as Conformalized Quantile Regression (CQR) [21] could provide more adaptive intervals. Exploring these directions remains future work.

## 6 Summary

This paper presents a practical and statistically grounded framework for producing reliable prediction intervals for molecular property prediction under label shift. By weighting conformal prediction with estimates of the target label distribution—obtained via BBSE, RLLS, and MLE—our method restores the coverage guarantees that split conformal prediction loses under distribution shift. When tested on the AqSolDB dataset with a large-scale pretrained chemical language model, our weighted conformal prediction consistently achieves more robust coverage than traditional approaches, with no need for costly retraining. The method is compatible with various estimation techniques, while maximum likelihood–based corrections achieve the best performance in coverage recovery. Our key contribution lies in developing a generalizable and model-agnostic framework that addresses an essential gap in the reliability of molecular property prediction. By ensuring statistically rigorous uncertainty quantification under label shift, our approach advances AI-based drug discovery toward regulatory compliance and real-world adoption, ultimately increasing confidence in high-stakes decisions on which compounds progress through the development pipeline.

## References

[1] Olivier J Wouters, Martin McKee, and Jeroen Luyten. Estimated research and development investment needed to bring a new medicine to market, 2009-2018. Jama, 323(9):844–853, 2020.

[2] Center for Devices and Radiological Health. Artificial Intelligence-Enabled Device Software Functions: Lifecycle Management and Marketing Submission Recommendations. https://www.fda.gov/regulatory-information/search-fda-guidance-documents/artificialintelligence-enabled-device-software-functions-lifecycle-management-and-marketing, Mon, 01/06/2025 - 09:41.

[3] Center for Drug Evaluation and Research. Considerations for the Use of Artificial Intelligence To Support Regulatory Decision-Making for Drug and Biological Products. https://www.fda.gov/regulatory-information/search-fda-guidance-documents/considerationsuse-artificial-intelligence-support-regulatory-decision-making-drug-and-biological, Tue, 01/07/2025 - 14:26.

[4] Jose G. Moreno-Torres, Troy Raeder, Rocío Alaiz-Rodríguez, Nitesh V. Chawla, and Francisco Herrera. A unifying view on dataset shift in classification. Pattern Recognition, 45(1):521–530, 2012.

[5] Vladimir Vovk, Alexander Gammerman, and Glenn Shafer. Algorithmic learning in a random world. Springer, 2005.

[6] Ryan J Tibshirani, Rina Foygel Barber, Emmanuel Candes, and Aaditya Ramdas. Conformal prediction under covariate shift. Advances in Neural Information Processing Systems, 32, 2019.

[7] Anastasios N Angelopoulos and Stephen Bates. A gentle introduction to conformal prediction and distribution-free uncertainty quantification. arXiv preprint arXiv:2107.07511, 2021.

[8] Jing Lei and Larry Wasserman. Distribution-free prediction bands for non-parametric regression. Journal ofthe Royal Statistical Society Series B: Statistical Methodology, 76(1):71–96, 2014.

[9] Harris Papadopoulos, Kostas Proedrou, Volodya Vovk, and Alex Gammerman. Inductive Confidence Machines for Regression. In Tapio Elomaa, Heikki Mannila, and Hannu Toivonen, editors, Machine Learning: ECML 2002, pages 345–356. Springer, 2002.

[10] Siddhartha Laghuvarapu, Zhen Lin, and Jimeng Sun. Codrug: Conformal drug property prediction with density estimation under covariate shift. Advances in Neural Information Processing Systems, 36:37728–37747, 2023.

[11] Zachary Lipton, Yu-Xiang Wang, and Alexander Smola. Detecting and correcting for label shift with black box predictors. In International Conference on Machine Learning, pages 3122–3130. PMLR, 2018.

[12] Kamyar Azizzadenesheli, Anqi Liu, Fanny Yang, and Animashree Anandkumar. Regularized learning for domain adaptation under label shifts. In International Conference on Learning Representations, 2019.

[13] Amr Alexandari, Anshul Kundaje, and Avanti Shrikumar. Maximum likelihood with biascorrected calibration is hard-to-beat at label shift adaptation. In International Conference on Machine Learning, pages 222–232. PMLR, 2020.

[14] Leo Klarner, Tim GJ Rudner, Michael Reutlinger, Torsten Schindler, Garrett M Morris, Charlotte Deane, and Yee Whye Teh. Drug discovery under covariate shift with domain-informed prior distributions over functions. In International Conference on Machine Learning, pages 17176– 17197. PMLR, 2023.

[15] Fang Wu, Shuting Jin, Siyuan Li, and Stan Z Li. Instructor-inspired machine learning for robust molecular property prediction. Advances in Neural Information Processing Systems, 37:116202–116222, 2024.

[16] Jina Kim, Jeffrey Willette, Bruno Andreis, and Sung Ju Hwang. Robust molecular property prediction via densifying scarce labeled data. In ICML 2025 Generative AI and Biology (GenBio) Workshop, 2025.

[17] Wenwen Si, Sangdon Park, Insup Lee, Edgar Dobriban, and Osbert Bastani. PAC prediction sets under label shift. In The Twelfth International Conference on Learning Representations, 2024.

[18] Seyone Chithrananda, Gabriel Grand, and Bharath Ramsundar. Chemberta: large-scale selfsupervised pretraining for molecular property prediction. arXiv preprint arXiv:2010.09885, 2020.

[19] Carl Edwards, Tuan Lai, Kevin Ros, Garrett Honke, Kyunghyun Cho, and Heng Ji. Translation between molecules and natural language. In Proceedings ofthe 2022 Conference on Empirical Methods in Natural Language Processing, pages 375–413, Abu Dhabi, United Arab Emirates, December 2022. Association for Computational Linguistics.

[20] Jerret Ross, Brian Belgodere, Vijil Chenthamarakshan, Inkit Padhi, Youssef Mroueh, and Payel Das. Large-scale chemical language representations capture molecular structure and properties. Nature Machine Intelligence, 4(12):1256–1264, 2022.

[21] Yaniv Romano, Evan Patterson, and Emmanuel Candes. Conformalized quantile regression. Advances in neural information processing systems, 32, 2019.

[22] Murat Cihan Sorkun, Abhishek Khetan, and Süleyman Er. AqSolDB, a curated reference set of aqueous solubility and 2D descriptors for a diverse set of compounds. Scientific Data, 6(1):143, August 2019.

[23] Kexin Huang, Tianfan Fu, Wenhao Gao, Yue Zhao, Yusuf H Roohani, Jure Leskovec, Connor W. Coley, Cao Xiao, Jimeng Sun, and Marinka Zitnik. Therapeutics data commons: Machine learning datasets and tasks for drug discovery and development. In Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 1), 2021.

[24] Mike Lewis, Yinhan Liu, Naman Goyal, Marjan Ghazvininejad, Abdelrahman Mohamed, Omer Levy, Veselin Stoyanov, and Luke Zettlemoyer. BART: Denoising sequence-to-sequence pretraining for natural language generation, translation, and comprehension. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 7871–7880, Online, July 2020. Association for Computational Linguistics.

[25] David Weininger. Smiles, a chemical language and information system. 1. introduction to methodology and encoding rules. Journal of chemical information and computer sciences, 28(1):31–36, 1988.

[26] Barbara Zdrazil, Eloy Felix, Fiona Hunter, Emma J Manners, James Blackshaw, Sybilla Corbett, Marleen de Veij, Harris Ioannidis, David Mendez Lopez, Juan F Mosquera, Maria Paula Magarinos, Nicolas Bosc, Ricardo Arcila, Tevfik Kizilören, Anna Gaulton, A Patrícia Bento, Melissa F Adasme, Peter Monecke, Gregory A Landrum, and Andrew R Leach. The ChEMBL Database in 2023: A drug discovery platform spanning multiple bioactivity data types and time periods. Nucleic Acids Research, 52(D1):D1180–D1192, January 2024.

[27] Sunghwan Kim, Jie Chen, Tiejun Cheng, Asta Gindulyte, Jia He, Siqian He, Qingliang Li, Benjamin A Shoemaker, Paul A Thiessen, Bo Yu, Leonid Zaslavsky, Jian Zhang, and Evan E Bolton. PubChem 2025 update. Nucleic Acids Research, 53(D1):D1516–D1525, January 2025.

[28] John J. Irwin, Teague Sterling, Michael M. Mysinger, Erin S. Bolstad, and Ryan G. Coleman. ZINC: A Free Tool to Discover Chemistry for Biology. Journal ofChemical Information and Modeling, 52(7):1757–1768, July 2012.

[29] Alexander Shivanyuk, Sergey Ryabukhin, A.V. Bogolyubsky, D.M. Mykytenko, Alexander Chuprina, W. Heilman, A.N. Kostyuk, and A. Tolmachev. Enamine real database: Making chemical diversity real. Chimica Oggi, 25:58–59, 11 2007.

[30] Maria Sorokina, Peter Merseburger, Kohulan Rajan, Mehmet Aziz Yirik, and Christoph Steinbeck. COCONUT online: Collection of Open Natural Products database. Journal of Cheminformatics, 13(1):2, January 2021.

[31] Craig Knox, Mike Wilson, Christen M. Klinger, Mark Franklin, Eponine Oler, Alex Wilson, Allison Pon, Jordan Cox, Na Eun Lucy Chin, Seth A. Strawbridge, Marysol Garcia-Patino, Ray Kruger, Aadhavya Sivakumaran, Selena Sanford, Rahil Doshi, Nitya Khetarpal, Omolola Fatokun, Daphnee Doucet, Ashley Zubkowski, Dorsa Yahya Rayat, Hayley Jackson, Karxena Harford, Afia Anjum, Mahi Zakir, Fei Wang, Siyang Tian, Brian Lee, Jaanus Liigand, Harrison Peters, Ruo Qi Rachel Wang, Tue Nguyen, Denise So, Matthew Sharp, Rodolfo da Silva, Cyrella Gabriel, Joshua Scantlebury, Marissa Jasinski, David Ackerman, Timothy Jewison, Tanvir Sajed, Vasuk Gautam, and David S. Wishart. DrugBank 6.0: The DrugBank Knowledgebase for 2024. Nucleic Acids Research, 52(D1):D1265–D1275, January 2024.

[32] Huaxiu Yao, Yiping Wang, Linjun Zhang, James Y Zou, and Chelsea Finn. C-mixup: Improving generalization in regression. Advances in neural information processing systems, 35:3361–3376, 2022.

[33] Xinyi Wu, Yun Zhang, Jiahui Yu, Chengyun Zhang, Haoran Qiao, Yejian Wu, Xinqiao Wang, Zhipeng Wu, and Hongliang Duan. Virtual data augmentation method for reaction prediction. Scientific Reports, 12(1):17098, October 2022.

[34] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In International Conference on Learning Representations, 2019.

## A Detailed Experimental Settings

All models were trained on NVIDIA A100 SXM4 40GB GPUs. As the model requires approximately 7.5 GB of memory for training, it is also feasible to run it on GPUs with lower specifications. Detailed model hyperparameters are represented in Table 1.

Table 1. Hyperparameters for training BBSE, RLLS, and MLE
<table><tr><td>Hyperparameter</td><td>BBSE</td><td>RLLS</td><td>MLE</td></tr><tr><td>Optimizer</td><td>AdamW [34]</td><td>AdamW [34]</td><td>AdamW [34]</td></tr><tr><td>Adam betas</td><td>(0.9, 0.99)</td><td>(0.9, 0.99)</td><td>(0.9, 0.99)</td></tr><tr><td>Learning rate</td><td>5e-5</td><td>5e-5</td><td>5e-5</td></tr><tr><td>Weight decay</td><td>0.1</td><td>0.1</td><td>0.1</td></tr><tr><td>Warmup steps</td><td>100</td><td>100</td><td>100</td></tr><tr><td>Error rate α</td><td>0.1</td><td>0.1</td><td>0.1</td></tr><tr><td>Batch size</td><td>16</td><td>16</td><td>16</td></tr><tr><td>Max length</td><td>150</td><td>150</td><td>150</td></tr><tr><td>Label shift β</td><td>-0.5</td><td>-0.5</td><td>-0.5</td></tr><tr><td>BART hidden dim</td><td>768</td><td>768</td><td>768</td></tr><tr><td>Predictor hidden dims</td><td>[512, 256]</td><td>[512, 256]</td><td>[512, 256]</td></tr><tr><td>Calibration</td><td>None</td><td>None</td><td>BCTS</td></tr><tr><td>Epochs</td><td>10</td><td>10</td><td>10</td></tr></table>

## B Additional Results

Coverage recovery performance based on the number of bins We varied the number of bins in BBSE, RLLS, and MLE, and for each configuration, the mean and standard deviation of coverage and interval length were reported over 1000 trials (Tables 2). The characteristics of the coverage distribution vary with the number of bins used in applying WCP through label discretization. Specifically, with fewer bins, certain trials exhibited high coverage, but the overall coverage distribution showed greater variance. Conversely, as the number of bins increased, the variance of the overall coverage distribution decreased, resembling the distribution observed when CP was applied to a non-shifted test dataset. This phenomenon can be interpreted as follows: with fewer bins, the coarse discretization of nonconformity scores leads to unstable quantile estimation, causing irregular fluctuations in the length of prediction intervals and significantly increasing the variance of the overall coverage distribution. On the other hand, as the number of bins increases, the estimation errors caused by discretization are reduced, resulting in a more stable and narrower (relatively) coverage distribution.

Table 2. Comparison of average coverage and interval length across different bin numbers for BBSE, RLLS, and MLE methods.
<table><tr><td rowspan="2">Bins</td><td colspan="2">BBSE</td><td colspan="2">RLLS</td><td colspan="2">MLE (BCTS)</td></tr><tr><td>Coverage</td><td>Length</td><td>Coverage</td><td>Length</td><td>Coverage</td><td>Length</td></tr><tr><td>5</td><td>0.865 ±0.033</td><td>4.346 ±0.316</td><td>0.886 ±0.034</td><td>4.804 ±0.590</td><td>0.899 ±0.034</td><td>5.163 ±0.832</td></tr><tr><td>15</td><td>0.845 ±0.032</td><td>3.993 ±0.236</td><td>0.877 ±0.032</td><td>4.577 ±0.42</td><td>0.895 ±0.034</td><td>5.031 ±0.620</td></tr><tr><td>50</td><td>0.804 ±0.033</td><td>3.414 ±0.149</td><td>0.859 ±0.032</td><td>4.204 ±0.297</td><td>0.892 ±0.034</td><td>4.947 ±0.581</td></tr><tr><td>100</td><td>0.782 ±0.033</td><td>3.165 ±0.123</td><td>0.843 ±0.032</td><td>3.934 ±0.222</td><td>0.858 ±0.031</td><td>4.192 ±0.242</td></tr></table>

![](images/afea78beb44c116c1f6e48fd7e68f9ef524961be14864a08d6c4e52c89fd6d27.jpg)

![](images/c5eb7fc6f8d4a0c1ac7b18b570341b1c04e5bb826e1ea76aa203b480777bc9ed.jpg)  
Figure 5. The KDE coverage distribution (top) and interval length distribution (bottom) for 1000 repeated experiments using standard split conformal prediction and the proposed methods are shown. Each color represents the number of bins used to calculate marginal probability ratios via BBSE. The triangular markers on the x-axis indicate the mean value of each distribution.

![](images/f5bcec0027264250f8663d400a9def0c94e507746d2676e7cfbd22a90a6eaa4a.jpg)

![](images/cece40c5d1b96bf729a44c8d59683a6b51d1d3f5c4b1e17be9f09a1a1beab31d.jpg)  
Figure 6. The KDE coverage distribution (top) and interval length distribution (bottom) for 1000 repeated experiments using standard split conformal prediction and the proposed methods are shown. Each color represents the number of bins used to calculate marginal probability ratios via RLLS. The triangular markers on the x-axis indicate the mean value of each distribution.

![](images/2e593a22b7a560e3606eac1b3a5480bddc04f10b974df947fb6a261f22b261d2.jpg)

![](images/ae7a7e45dd551c3eb591ac8786f54c2cc12ede2ef8cab94e4e57b23fa306cf85.jpg)  
Figure 7. The KDE coverage distribution (top) and interval length distribution (bottom) for 1000 repeated experiments using standard split conformal prediction and the proposed methods are shown. Each color represents the number of bins used to calculate marginal probability ratios via MLE. The triangular markers on the x-axis indicate the mean value of each distribution.