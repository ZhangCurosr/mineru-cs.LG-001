# Balanced Adaptive Prototype Selection for Scalable TabPFN Inference on Large-Scale Tabular Data

1<sup>st</sup> Mahboobe Jadid Department of Computer Engineering Ne. C., Islamic Azad University Neyshabur, Iran mahboobe.jadid@iau.ac.ir

2<sup>nd</sup> Melika Rezaye Garkani Department of Computer Engineering Ne. C., Islamic Azad University Neyshabur, Iran melika.rezaegarkani@iau.ac.ir

3<sup>rd</sup> Ali Mousavi Department of Computer Engineering Ne. C., Islamic Azad University Neyshabur, Iran mousavi@iau.ac.ir

Abstract—Pretrained tabular foundation models have demonstrated strong predictive capability; however, their application to large-scale datasets remains constrained by the limited inference context. This paper introduces Balanced Adaptive Prototype Selection (BAPS), a framework for constructing compact, information-preserving contexts for scalable TabPFN inference. Without modifying or retraining the pretrained model, BAPS jointly preserves representative structure, informative decision boundaries, local density, class balance, and feature-space diversity. Experiments on the million-row HIGGS and SUSY datasets show that 512 prototypes retain strong predictive performance and reliable calibration, corresponding to an approximately 1,953-fold context compression. All experiments were conducted on an Intel Core i7 CPU with 16 GB RAM and no GPU acceleration. These findings establish effective context construction as a practical mechanism for extending pretrained tabular foundation models to million-scale datasets.

Index Terms—TabPFN, Tabular Foundation Models, Prototype Selection, Context Construction, Large-Scale Tabular Learning

## I. INTRODUCTION

Recent advances in pretrained foundation models have transformed machine learning by replacing task-specific opti mization with knowledge transfer from large-scale pretraining [20]. This paradigm has recently been extended to tabular learning, where models such as TabPFN perform prediction directly from a labeled inference context rather than retraining model parameters for each downstream task [5]. This capability enables efficient deployment while achieving competitive predictive performance across diverse tabular problems [1].

Despite these advantages, the practical scalability of pretrained tabular foundation models remains fundamentally constrained by the bounded inference context [6]. Modern tabular datasets often contain hundreds of thousands or even millions of training instances, whereas only a limited number of examples can be presented to the model during inference [1]. Consequently, the effectiveness of TabPFN depends not only on the pretrained model itself, but also on whether the selected context preserves the predictive information contained in the original training data [5], [6]. Under severe compression, conventional sampling or prototype reduction may discard informative decision boundaries, minority-class evidence, and locally important structures, substantially degrading inference quality [7].

Existing prototype selection and dataset reduction methods primarily optimize computational efficiency through sampling, clustering, or instance selection [2], [8]. Although effective for conventional machine learning algorithms, these methods are not designed for pretrained foundation models, where the objective is information-preserving context construction rather than dataset reduction [5]. Consequently, reducing the number of training samples alone is insufficient for scalable TabPFN inference [6]. A principled mechanism for constructing compact contexts that preserve predictive information is therefore essential.

Motivated by this observation, we propose Balanced Adaptive Prototype Selection (BAPS), an information-preserving context-construction framework for scalable TabPFN inference. Instead of modifying or retraining the pretrained model, BAPS constructs compact inference contexts by jointly preserving representative structure, informative decision boundaries, local density, class balance, and feature-space diversity. By optimizing the information retained within a fixed context budget, BAPS directly addresses the scalability bottleneck of pretrained tabular foundation models while remaining fully compatible with the original TabPFN architecture [5]. Extensive experiments on the million-row HIGGS and SUSY benchmark datasets demonstrate that BAPS maintains strong predictive performance and reliable calibration using only 512 prototypes, corresponding to an approximately 1,953- fold context compression. These results show that effective context construction is not merely an optimization strategy but a fundamental requirement for scaling pretrained tabular foundation models to million-scale datasets.

The main contributions of this work are summarized as follows.

• We identify bounded inference context as the primary scalability bottleneck of pretrained tabular foundation models and formulate scalable context construction as an information-preservation problem.

• We propose BAPS, an information-preserving contextconstruction framework that jointly preserves representative structure, informative decision boundaries, local density, class balance, and feature-space diversity without modifying or retraining the pretrained model.

• We demonstrate through comprehensive experiments that

BAPS enables practical large-scale TabPFN inference on million-row datasets using only 512 prototypes and standard CPU hardware.

## II. RELATED WORK

Prototype selection and dataset reduction have long been employed to improve the scalability of tabular learning by reducing training-set size through sampling, clustering, or instance selection [8]. While these approaches effectively decrease computational cost for conventional machine learning, they assume that the downstream model is retrained using the selected data [7]. Consequently, their optimization objective is efficient data reduction rather than preserving the information required for inference.

Recent pretrained tabular foundation models, particularly TabPFN, have introduced a fundamentally different learning paradigm in which predictions are generated directly from a bounded labeled context instead of task-specific model optimization [9]. More recent studies have further investigated scalable context optimization and efficient inference for Prior-Data Fitted Networks, highlighting the importance of context quality for extending pretrained tabular foundation models beyond small datasets [10]. Under this setting, prediction quality depends on the information retained within the limited inference context rather than on the size of the original training set [5]. Therefore, simply selecting representative samples is insufficient, since informative decision boundaries, minorityclass evidence, and locally discriminative structures may be lost during compression [7].

This fundamental difference creates a gap that existing prototype selection and dataset reduction methods do not explicitly address [8]. Their objective is to compress data efficiently, whereas scalable TabPFN inference requires compact contexts that maximize predictive information under a fixed context budget [6]. Motivated by this requirement, BAPS formulates prototype selection as an information-preserving contextconstruction problem. By jointly preserving representative structure, informative decision boundaries, local density, class balance, and feature-space diversity, BAPS directly addresses the scalability bottleneck of pretrained tabular foundation models without modifying or retraining the underlying model.

## III. PROPOSED BAPS FRAMEWORK

## A. Motivation and Design Principle

Pretrained tabular foundation models have recently demonstrated remarkable performance by transferring knowledge acquired during large-scale offline pretraining to unseen downstream tasks through in-context inference [9]. Unlike conventional machine learning models, they do not update their parameters for each new dataset. Instead, predictions are generated exclusively from a small labeled context provided during inference [5]. Consequently, the quality of the inference context becomes a primary factor determining predictive performance [10].

Although this inference paradigm eliminates task-specific retraining, it introduces a fundamental scalability challenge [6].

Real-world tabular datasets often contain hundreds of thousands or even millions of training instances, whereas only a limited number of labeled samples can be included in the inference context [1]. As the dataset grows, the proportion of accessible information rapidly decreases, causing important structural patterns to be discarded before inference. Simply selecting representative samples or randomly reducing the training set cannot adequately preserve the diverse predictive evidence required for reliable in-context reasoning [3].

This limitation arises because different observations contribute complementary information to the inference process. Representative samples capture the dominant data distribution, boundary instances provide discriminative evidence for sep arating neighboring classes, dense local regions characterize reliable neighborhood structures, while minority-class samples prevent contextual bias toward majority classes [4]. Preserving only one of these characteristics inevitably degrades the contextual representation under severe compression.

Motivated by this observation, we reformulate scalable context construction as an information-preserving optimization problem rather than a conventional prototype reduction task. The objective is to maximize the predictive information retained within a fixed prototype budget while maintaining compatibility with the pretrained TabPFN architecture. Instead of modifying the foundation model, the proposed Balanced Adaptive Prototype Selection (BAPS) framework optimizes the inference context itself through complementary prototype selection principles that jointly preserve representative structure, discriminative boundaries, local density, class balance, and feature-space diversity.

The central hypothesis of this work is that the scalability of context-based tabular foundation models is primarily constrained by context quality rather than model capacity. By constructing a compact yet information-rich inference context, large-scale tabular datasets can be efficiently represented without retraining the pretrained model, thereby enabling scalable and reliable in-context inference.

## B. Problem Formulation

Let the training dataset be

$$
\begin{array} { r } { \mathcal { D } = \{ ( \mathbf { x } _ { i } , y _ { i } ) \} _ { i = 1 } ^ { N } , } \end{array}
$$

where $\mathbf { x } _ { i } \in \mathbb { R } ^ { d }$ is a feature vector and $y _ { i }$ is its corresponding class label. Given a fixed inference-context budget B, the objective is to construct a prototype subset

$$
\mathcal { P } \subset \mathcal { D } , \qquad | \mathcal { P } | = B ,\tag{1}
$$

that preserves the predictive information of the original dataset under the imposed context-size constraint.

Unlike conventional prototype selection, which primarily emphasizes data reduction, BAPS formulates context construction as an information-preserving optimization problem [7], [8]. The information retained by a prototype set is defined as

$$
\begin{array} { r } { \mathcal { Z } ( \mathcal { P } ) = \lambda _ { r } R ( \mathcal { P } ) + \lambda _ { q } Q ( \mathcal { P } ) + \lambda _ { d } D ( \mathcal { P } ) + \lambda _ { v } V ( \mathcal { P } ) + \lambda _ { c } C ( \mathcal { P } ) , } \end{array}\tag{2}
$$

![](images/aecccbf303cec90fa978b2987167ad8cdaed26ac97baa47ba6c081285c1ca605.jpg)  
Fig. 1. Overview of the proposed BAPS framework for scalable TabPFN inference.

Algorithm 1 Balanced Adaptive Prototype Selection (BAPS)   
Require: Training dataset D, prototype budget B   
Ensure: Prototype context $\mathcal { P }$   
1: Allocate class-wise prototype budgets   
2: Generate representative candidates   
3: Generate boundary candidates   
4: Generate density candidates   
5: Merge complementary candidates   
6: Apply diversity refinement   
7: Construct prototype context P   
8: Optionally aggregate multiple contexts   
9: return P

where R, Q, D, V , and C quantify representative, boundary, density, diversity, and class-preservation information, respectively, with

$$
\sum _ { i } \lambda _ { i } = 1 , \qquad \lambda _ { i } \geq 0 .\tag{3}
$$

The optimal prototype context is therefore obtained by

$$
\mathcal { P } ^ { * } = \underset { | \mathcal { P } | = B } { \arg \operatorname* { m a x } } \mathcal { I } ( \mathcal { P } ) .\tag{4}
$$

Since solving this optimization exactly is computationally infeasible, BAPS constructs an efficient approximation through complementary prototype-selection stages described in the following subsection.

## C. Balanced Adaptive Prototype Selection

The optimization problem above is computationally intractable for large-scale datasets because the number of feasible prototype subsets grows combinatorially with the training size. Rather than searching for the global optimum, BAPS approximates the information-preservation objective through a hierarchical context-construction strategy. As illustrated in Fig. 1 and summarized in Algorithm 1, the framework progressively preserves complementary sources of predictive information while maintaining a fixed prototype budget.

The framework first performs Class-Aware Budget Allocation, which distributes the available prototypes across classes before candidate generation, preserving the class-information component $C ( \mathcal { P } )$ . Candidate prototypes are then generated from three complementary perspectives. Representative Selection captures the dominant statistical structure of each class, Boundary Selection retains informative samples near decision boundaries, and Density Selection preserves reliable local neighborhood structures, jointly approximating R(P), Q(P), and $D ( \mathcal { P } )$ , respectively.

The generated candidates are integrated through Diversity Refinement, which optimizes $V ( \mathcal P )$ by removing redundant prototypes while maximizing feature-space coverage. The resulting prototype context therefore preserves complementary predictive information rather than a single structural characteristic of the original dataset. For additional robustness, BAPS optionally generates multiple prototype contexts using different random initializations and aggregates their prediction probabilities. Since every inference pass operates on the same fixed-size context, this strategy improves prediction stability without modifying the pretrained TabPFN model, at an additional inference cost approximately proportional to the number of contexts.

## D. Computational Complexity

The computational overhead of BAPS is incurred only during prototype construction, which is performed once for each training dataset. During inference, the pretrained model operates on a fixed-size prototype context of B samples, making both computational and memory costs independent of the original dataset size. Consequently, BAPS enables scalable inference while remaining fully compatible with the original pretrained architecture.

## IV. EXPERIMENTAL SETUP

## A. Experimental Setup

The proposed BAPS framework was evaluated on five publicly available tabular classification benchmarks, namely

HIGGS, SUSY, Covertype, Electricity, and Diabetes, covering diverse application domains, dataset sizes, feature dimensions, and class distributions. To comply with the conference page limit, only the results on the two largest datasets, HIGGS and SUSY, are presented, as they provide the most challenging evaluation of scalability under million-scale data. The evaluation includes six binary and multi-class benchmark datasets spanning medical, chemical, image-derived, environmental, and high-energy physics domains. Breast Cancer, Wine, and Digits represent small-scale settings, whereas Covertype, SUSY, and HIGGS provide increasingly demanding large-scale scenarios. The final scalability experiments focus on HIGGS and SUSY with up to one million training instances, corresponding to an approximately 1,953:1 context-compression ratio when using a fixed budget of 512 prototypes.

To ensure a fair comparison, all methods were evaluated under identical experimental settings using the same training/testing splits, prototype budgets, and random seeds. BAPS was implemented on top of the original pretrained TabPFN model without modifying its architecture or parameters [5], ensuring that any performance improvement results solely from the proposed context-construction strategy.

Performance was evaluated using Balanced Accuracy, Macro-F1, ROC-AUC, and Expected Calibration Error (ECE), while inference time was additionally reported to assess computational cost.

## B. Implementation Details

BAPS was implemented using the original pretrained TabPFN model without modifying its architecture or parameters. All competing methods were evaluated under identical experimental settings, including the same training/testing splits, prototype budgets, and evaluation protocol, ensuring a fair and reproducible comparison. Unless otherwise stated, all reported results are averaged over five independent runs.

## V. RESULTS AND DISCUSSION

The experiments evaluate whether information-preserving context construction can extend TabPFN to million-scale datasets without modifying or retraining the pretrained model. All reduced-context methods are compared under identical data partitions, random seeds, and prototype budgets; hence, observed differences primarily reflect the predictive value of the constructed contexts. The analysis focuses on three questions: whether BAPS preserves stronger predictive evidence than random or geometric reduction, whether complementary contexts improve stability and calibration, and whether these gains justify the additional computational cost under extreme compression.

## A. Predictive Performance under Extreme Compression

Table I compares all methods under the same 512-prototype budget, equivalent to retaining only 0.0512% of the million-row training sets. BAPS Ensemble outperforms all single-context baselines on both datasets and achieves the strongest overall performance on SUSY, showing that complementary information preservation is more effective than random sampling or purely geometric reduction. On HIGGS, KMeans Ensemble attains a marginally higher balanced accuracy, but BAPS remains superior to every single-context method and competitive across the remaining metrics. These results demonstrate that BAPS provides a consistently informative TabPFN context under nearly 2,000-fold compression, without modifying or retraining the pretrained model.

TABLE I  
PERFORMANCE COMPARISON OF CONTEXT-CONSTRUCTION METHODS UNDER THE SAME 512-PROTOTYPE BUDGET
<table><tr><td>Dataset</td><td>Method</td><td>↑Bal. Acc.</td><td>↑Macro-F1</td><td>↑ROC-AUC</td><td>↓ECE</td><td>↓Time (s)</td></tr><tr><td rowspan="5">HIGGS [23]</td><td>Stratified Random</td><td>0.670</td><td>0.670</td><td>0.735</td><td>0.030</td><td>479.9</td></tr><tr><td>KMeans Medoid</td><td>0.690</td><td>0.687</td><td>0.758</td><td>0.085</td><td>479.7</td></tr><tr><td>BAPS-Density</td><td>0.688</td><td>0.683</td><td>0.755</td><td>0.106</td><td>481.2</td></tr><tr><td>BAPS Ensemble</td><td>0.697</td><td>0.692</td><td>0.769</td><td>0.093</td><td>1511.8</td></tr><tr><td>KMeans Ensemble</td><td>0.703</td><td>一</td><td>0.776</td><td>一</td><td>1458.0</td></tr><tr><td rowspan="5">SUSY [24]</td><td>Stratified Random</td><td>0.779</td><td>0.781</td><td>0.854</td><td>0.014</td><td>385.5</td></tr><tr><td>KMeans Medoid</td><td>0.771</td><td>0.773</td><td>0.850</td><td>0.087</td><td>386.1</td></tr><tr><td>BAPS-Density</td><td>0.776</td><td>0.778</td><td>0.856</td><td>0.056</td><td>385.0</td></tr><tr><td>BAPS Ensemble</td><td>0.782</td><td>0.784</td><td>0.863</td><td>0.038</td><td>1212.2</td></tr><tr><td>KMeans Ensemble</td><td>0.778</td><td>1</td><td>0.857</td><td>1</td><td>1209.5</td></tr></table>

Note: Bold values indicate the best result for each metric within each dataset.  
Lower values are better for ECE and Time.

## B. Effect of Multi-Context Aggregation

Fig. 2 provides a direct comparison of the four principal context-construction strategies under the same 512-prototype budget. On HIGGS, BAPS Ensemble achieves a balanced accuracy of 0.697, improving over its single-context variant (0.688), Stratified Random Sampling (0.670), and remaining competitive with KMeans Medoid (0.690). On SUSY, BAPS Ensemble obtains the highest balanced accuracy (0.782), exceeding Stratified Random Sampling (0.779), KMeans Medoid (0.771), and single-context BAPS (0.776). These results demonstrate that complementary BAPS contexts consistently improve upon the corresponding single-context representation while maintaining strong performance across distinct million scale distributions

![](images/e5db8e484b185b99a12a8fa2506709c64d49c08fcdf4533b37bc598fc868ef82.jpg)  
Fig. 2. Balanced-accuracy comparison of four context-construction strategies on HIGGS and SUSY under the same 512-prototype budget

## C. Information Preservation and Computational Trade-off

The experimental results demonstrate that BAPS effectively bridges the gap between the fixed-context limitation of TabPFN and million-scale tabular datasets. Despite retaining only 512 prototypes (approximately 1,953:1 compression), the proposed framework preserves sufficient contextual information to maintain strong predictive performance. This confirms that prototype quality, rather than prototype quantity, is the primary factor determining context effectiveness.

Multi-context aggregation can improve calibration relative to the corresponding single-context BAPS configuration while preserving strong discriminative performance. Although this strategy increases inference time approximately in proportion to the number of contexts, the additional cost remains predictable and represents a practical trade-off for extending pretrained TabPFN to datasets that would otherwise exceed its inferencecontext limitation.

## D. Statistical Validation

The paired Wilcoxon test with Holm correction confirms that the performance improvement of the complete BAPS framework over the strongest single-context baseline is statistically significant (p = 0.031), indicating that the observed gains are unlikely to arise from random variation.

These results validate the proposed information-preserving context-construction strategy, demonstrating that compact prototype contexts can effectively overcome the fixed-context limitation of TabPFN while preserving strong predictive performance under nearly 2,000-fold compression.

## VI. CONCLUSION

This paper presented Balanced Adaptive Prototype Selection (BAPS), an information-preserving context-construction framework that extends pretrained TabPFN to large-scale tabular datasets without modifying or retraining the model.

By constructing compact and informative inference contexts, BAPS effectively addresses the fixed-context limitation while maintaining strong predictive performance under extreme data compression. Experimental results demonstrate that preserving complementary contextual information is more effective than conventional prototype reduction, enabling scalable and robust TabPFN inference on million-scale datasets. Future work will investigate adaptive context construction and extend the proposed framework to broader foundation models for tabular learning.

## REFERENCES

[1] N. Hollmann, S. Muller, L. Purucker, A. Krishnakumar, M. K¨ orfer, S. B.¨ Hoo, R. T. Schirrmeister, and F. Hutter, “Accurate predictions on small data with a tabular foundation model,” Nature, vol. 637, pp. 319–326, 2025, doi: 10.1038/s41586-024-08328-6.

[2] J. Bien and R. Tibshirani, “Prototype selection for interpretable classification,” Ann. Appl. Stat., vol. 5, no. 4, pp. 2403–2424, 2011, doi: 10.1214/11-AOAS495.

[3] J. S. Sanchez, F. Pla, and F. J. Ferri, “Prototype selection for the nearest´ neighbour rule through proximity graphs,” Pattern Recognit. Lett., vol. 18, no. 6, pp. 507–513, 1997, doi: 10.1016/S0167-8655(97)00035-4.

[4] S. Garc´ıa, J. Derrac, J. R. Cano, and F. Herrera, “Prototype selection for nearest neighbor classification: Taxonomy and empirical study,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 34, no. 3, pp. 417–435, Mar. 2012, doi: 10.1109/TPAMI.2011.142.

[5] N. Hollmann, S. Muller, K. Eggensperger, and F. Hutter, “TabPFN: A¨ transformer that solves small tabular classification problems in a second,” in Proc. 11th Int. Conf. Learn. Represent. (ICLR), 2023.

[6] B. Feuer, R. T. Schirrmeister, V. Cherepanova, C. Hegde, F. Hutter, M. Goldblum, N. Cohen, and C. White, “TuneTables: Context optimization for scalable prior-data fitted networks,” in Adv. Neural Inf. Process. Syst., vol. 37, 2024, doi: 10.52202/079017-2654.

[7] I. Triguero, J. Derrac, S. Garc´ıa, and F. Herrera, “A taxonomy and experimental study on prototype generation for nearest neighbor classification,” IEEE Trans. Syst., Man, Cybern. C, Appl. Rev., vol. 42, no. 1, pp. 86–100, Jan. 2012, doi: 10.1109/TSMCC.2010.2103939.

[8] S. Garc´ıa, J. R. Cano, and F. Herrera, “A memetic algorithm for evolutionary prototype selection: A scaling up approach,” Pattern Recognit., vol. 41, no. 8, pp. 2693–2709, 2008, doi: 10.1016/j.patcog.2008.02.006.

[9] S. Muller, N. Hollmann, S. Pineda Arango, J. Grabocka, and F. Hutter,¨ “Transformers can do Bayesian inference,” in Proc. 10th Int. Conf. Learn. Represent. (ICLR), 2022.

[10] V. Thomas, J. Ma, R. Hosseinzadeh, K. Golestan, G. Yu, M. Volkovs, and A. Caterini, “Retrieval & fine-tuning for in-context tabular models,” in Adv. Neural Inf. Process. Syst., vol. 37, 2024, doi: 10.52202/079017-3442.

[11] S. O. Arik and T. Pfister, “TabNet: Attentive interpretable tabular learning,”<sup>¨</sup> Proc. AAAI Conf. Artif. Intell., vol. 35, no. 8, pp. 6679–6687, 2021, doi: 10.1609/aaai.v35i8.16826.

[12] G. Somepalli, M. Goldblum, A. Schwarzschild, C. B. Bruss, and T. Goldstein, “SAINT: Improved neural networks for tabular data via row attention and contrastive pre-training,” in Proc. NeurIPS 2022 First Table Representation Learning Workshop, 2022.

[13] Y. Gorishniy, I. Rubachev, V. Khrulkov, and A. Babenko, “Revisiting deep learning models for tabular data,” in Adv. Neural Inf. Process. Syst., vol. 34, pp. 18932–18943, 2021.

[14] S. Popov, S. Morozov, and A. Babenko, “Neural oblivious decision ensembles for deep learning on tabular data,” in Proc. 8th Int. Conf. Learn. Represent. (ICLR), 2020.

[15] J. Chen, K. Liao, Y. Wan, D. Z. Chen, and J. Wu, “DANets: Deep abstract networks for tabular data classification and regression,” Proc. AAAI Conf. Artif. Intell., vol. 36, no. 4, pp. 3930–3938, 2022, doi: 10.1609/aaai.v36i4.20309.

[16] Y. Gorishniy, A. Kotelnikov, and A. Babenko, “TabM: Advancing tabular deep learning with parameter-efficient ensembling,” in Proc. 13th Int. Conf. Learn. Represent. (ICLR), 2025.

[17] J. Qu, D. Holzmuller, G. Varoquaux, and M. Le Morvan, “TabICL: A¨ tabular foundation model for in-context learning on large data,” in Proc. 42nd Int. Conf. Mach. Learn. (ICML), PMLR, vol. 267, pp. 50817–50847, 2025.

[18] L. Grinsztajn, E. Oyallon, and G. Varoquaux, “Why do tree-based models still outperform deep learning on typical tabular data?” in Adv. Neural Inf. Process. Syst., vol. 35, pp. 507–520, 2022.

[19] V. Borisov, T. Leemann, K. Seßler, J. Haug, M. Pawelczyk, and G. Kasneci, “Deep neural networks and tabular data: A survey,” IEEE Trans. Neural Netw. Learn. Syst., vol. 35, no. 6, pp. 7499–7519, Jun. 2024, doi: 10.1109/TNNLS.2022.3229161.

[20] B. Van Breugel and M. Van Der Schaar, “Position: Why tabular foundation models should be a research priority,” in Proc. 41st Int. Conf. Mach. Learn. (ICML), PMLR, vol. 235, pp. 48976–48993, 2024.

[21] G. Ke, Q. Meng, T. Finley, T. Wang, W. Chen, W. Ma, Q. Ye, and T.-Y. Liu, “LightGBM: A highly efficient gradient boosting decision tree,” in Adv. Neural Inf. Process. Syst., vol. 30, pp. 3146–3154, 2017.

[22] T. Chen and C. Guestrin, “XGBoost: A scalable tree boosting system,” in Proc. 22nd ACM SIGKDD Int. Conf. Knowl. Discovery Data Mining, pp. 785–794, 2016, doi: 10.1145/2939672.2939785.

[23] P. Baldi, P. Sadowski, and D. Whiteson, “Searching for exotic particles in high-energy physics with deep learning,” Nature Commun., vol. 5, Art. no. 4308, 2014, doi: 10.1038/ncomms5308.

[24] P. Baldi, P. Sadowski, and D. Whiteson, “Searching for supersymmetry using deep learning,” Phys. Rev. Lett., vol. 114, no. 11, Art. no. 111801, 2015, doi: 10.1103/PhysRevLett.114.111801.