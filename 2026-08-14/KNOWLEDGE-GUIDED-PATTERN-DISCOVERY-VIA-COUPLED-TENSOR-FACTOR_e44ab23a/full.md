# KNOWLEDGE-GUIDED PATTERN DISCOVERY VIA COUPLED TENSOR FACTORIZATIONS

Gaute Johannessen<sup>⋆</sup> Geert Roelofvan der Ploeg<sup>†</sup> Evrim Acar<sup>†</sup>

<sup>⋆</sup> Faculty of Mathematics and Natural Sciences, University of Oslo, Oslo <sup>†</sup> Department of Data Science and Knowledge Discovery, Simula Metropolitan Center for Digital Engineering, Oslo

## ABSTRACT

In order to understand complex systems such as the human metabolome or human brain, different sensing technologies are used, generating complex data. These datasets are often multiway, i.e., with more than two axes of variation such as a subjects by metabolites by time array. While tensor factorizations have successfully revealed interpretable patterns from such complex data, they have so far been mainly data-driven. On the other hand, there is more to data – there are computational models (of these systems), which are rich sources of prior information. In this paper, we introduce a knowledgeguided approach that brings together data and computational models by jointly analyzing real data and simulated data (generated using a computational model) using coupled tensor factorizations with linear coupling. Our experiments on real metabolomics measurements demonstrate that guiding the analysis of such noisy data with simulated data improves the pattern discovery performance while also revealing potential discrepancies between data and computational models.

Index Terms— tensor factorizations, coupled tensor factorizations, knowledge-guided machine learning.

## 1. INTRODUCTION

Complex systems such as the human metabolome (i.e., the complete set of small biochemical compounds in the body) or the human brain produce high-dimensional, noisy measurements that evolve over time and with significant variation between individuals. Such datasets can often be arranged as higher-order tensors, e.g., subjects by metabolites by time tensors representing longitudinal metabolomics data [1], subjects by voxels by time tensors in neuroscience [2]. Tensor factorizations [3] have been successfully used to reveal interpretable patterns from such data in neuroscience [2, 4], metabolomics [1], and microbiome data analysis [5].

While tensor factorizations have proven effective, existing approaches are predominantly data-driven and do not exploit the rich prior information encoded in computational models of these systems. Computational models, grounded in known biochemistry and physiology, encode dynamics that may be difficult to infer from noisy observations. In recent years, there have been significant efforts to develop knowledgeguided machine learning methods [6, 7, 8, 9] to incorporate prior information to improve scalability, generalizability, explainability, and with much smaller data sizes. While these methods hold great potential, their progress has been primarily limited to inference/prediction tasks and data-rich domains. Recent work has introduced goal-oriented tensor factorizations preserving underlying physical quantities of interest via penalty terms [10]. Nevertheless, incorporation of such prior information from computational models in unsupervised settings where the goal is to reveal the underlying patterns (with high accuracy and reproducibility) and extract novel insights is relatively underdeveloped.

In this paper, we introduce a novel knowledge-guided approach for pattern discovery bringing together data and computational models. Suppose we have real (observational) data collected from a dynamic system – represented as a higherorder tensor, e.g., X with modes: metabolites, time, subjects. Using a computational model of the system, we also generate simulated data, e.g., tensor Y with modes: metabolites, time, virtual subjects. We then jointly analyze X and Y coupled in the features (i.e., metabolites) mode using coupled tensor factorizations to guide the analysis of noisy real data with relatively clean simulated data (Fig. 1). Coupled tensor factorizations offer a principled way to jointly analyze real and simulated data using tensor models, e.g., the CANDE-COMP/PARAFAC (CP) model, while extracting latent factors in the coupled mode linked to each other via a coupling relation. Since sets of features in real and simulated data may only partially overlap, we make use of coupled tensor factorizations with linear coupling [11, 12]. In this coupled setting, we consider different models for real data, i.e., CP and a more flexible PARAFAC2 model that accounts for individual variation by extracting subject-specific time profiles. Previously, real and simulated datasets were jointly analyzed [13]; however, unlike our approach, previous work was limited to a coupled CP model where only the matching features (e.g., metabolites) were included: only 6 out of the over 150 available in real data. In this paper, we guide real data analysis with simulated data in a more realistic setting by including all features in the real data. Our experiments using coupled models relying on both CP and PARAFAC2-based approaches demonstrate that coupling real data analysis with simulated data consistently improves pattern discovery.

![](images/f0cf828c77141f81b4e9ae9e06f427b7e37cd29213731254906cf09e89e6fdf5.jpg)  
Fig. 1: Coupled CP model jointly analyzing real data X with simulated data Y, where the factor matrices in the metabolites mode, $\mathbf { A } _ { \mathrm { v i r t u a l } }$ and $\mathbf { A } _ { \mathrm { r e a l } }$ , are linked via a coupling relation.

## 2. METHODS

We first discuss the tensor models used to extract patterns from the real data alone, and then introduce the coupled tensor factorization models used to jointly analyze real and simulated datasets.

## 2.1. Tensor Factorizations

The CP model [14, 15] approximates a higher-order tensor using a sum of rank-one tensors. An R-component CP model of a third-order tensor $\ b { \mathcal { X } } \in \mathbb { R } ^ { I \times J \times K }$ can be represented as

$$
\mathfrak { X } \approx \sum _ { r = 1 } ^ { R } \mathbf { a } _ { r } \circ \mathbf { b } _ { r } \circ \mathbf { c } _ { r } = \ [ \mathbf { A } , \mathbf { B } , \mathbf { C } ] ,\tag{1}
$$

where ◦ is the outer vector product; $\textbf { A } \in \mathbb { R } ^ { I \times R } , \textbf { B } \in$ $\mathbb { R } ^ { J \times R } , \mathbf { C } \ \in \ \mathbb { R } ^ { K \times R }$ are the factor matrices in each mode. Under mild conditions, the CP model is unique up to permutation and scaling ambiguities [3], which facilitates interpretation. If X is a metabolites by time by subjects tensor, the rth component may reveal groups of metabolites in $\mathbf { a } _ { r }$ their temporal profile in $ { \mathbf { b } } _ { r }$ and the corresponding subject stratification in $\mathbf { c } _ { r }$ . The CP model can be fit to X by solving the optimization problem:

$$
\operatorname* { m i n } _ { \mathbf { A } , \mathbf { B } , \mathbf { C } } \| \mathbf { A } - [ [ \mathbf { A } , \mathbf { B } , \mathbf { C } ] \| ^ { 2 } ,\tag{2}
$$

where $\| \cdot \|$ denotes the Frobenius norm.

The PARAFAC2 model [16] relaxes the CP model by allowing the captured patterns in one mode to change across slices. An R-component PARAFAC2 model approximates X as follows:

$$
\mathbf { X } _ { k } \approx \mathbf { A } \mathbf { D } _ { k } \mathbf { B } _ { k } ^ { \mathsf { T } } , \quad \mathbf { B } _ { k } ^ { \mathsf { T } } \mathbf { B } _ { k } = \Phi , \quad \mathrm { f o r } \ k = 1 , \ldots , K ,\tag{3}
$$

where $\mathbf { X } _ { k } \in \mathbb { R } ^ { I \times J }$ corresponds to the kth frontal slice of X, $\mathbf { D } _ { k } \in \mathbb { R } ^ { R \times R }$ is a diagonal matrix with $\mathbf { C } ( k , : ) = { \mathrm { d i a g } } ( \mathbf { D } _ { k } )$ i.e., the kth row of C has the diagonal entries of $\mathbf { D } _ { k }$ . The cross product matrix Φ $\in \mathbb { R } ^ { R \times R }$ is constant for all k, $k =$ $1 , . . . , K$ Similar to CP, PARAFAC2 has uniqueness properties which facilitate the interpretation of factors [17]. In PARAFAC2, A and C can reveal groups of metabolites and subject stratifications, as in the CP model. On the other hand, $\mathbf { B } _ { k } \ \in \ \mathbb { R } ^ { J \times R }$ represents the time profiles for subject k. Instead of a shared set of time profiles, i.e., B matrix in the CP model, PARAFAC2 allows each subject to have different time profiles – which has the promise to reveal subject-specific differences [18, 19, 20]. When fitting a PARAFAC2 model, we solve the following optimization problem:

$$
\begin{array} { r l } { \displaystyle \operatorname* { m i n } _ { \mathbf { A } , \{ \mathbf { B } _ { k } , \mathbf { D } _ { k } \} _ { k \leq K } } \sum _ { k = 1 } ^ { K } \| \mathbf { X } _ { k } - \mathbf { A } \mathbf { D } _ { k } \mathbf { B } _ { k } ^ { \top } \| ^ { 2 } } & { } \\ { \mathrm { s } . \mathrm { t } . \quad } & { \mathbf { B } _ { k } ^ { \top } \mathbf { B } _ { k } = \Phi , \mathrm { f o r } k = 1 , \dots , K . } \end{array}\tag{4}
$$

## 2.2. Coupled Tensor Factorizations

To utilize the prior knowledge encoded in simulated data, real data tensor X and simulated data tensor $\mathfrak { Y } \in \mathbb { R } ^ { I _ { v } \times J _ { v } \times K _ { \mathfrak { z } } }$ can be jointly analyzed using coupled tensor factorizations as in Fig. 1. Coupled tensor factorizations are extensions of tensor factorizations to multiple datasets [21, 22, 23]. For instance, X and Y coupled in the first mode, e.g., metabolites mode, can be jointly analyzed using a coupled CP model as follows:

$$
\begin{array} { r l r } { \underset { \mathbf { A } , \mathbf { B } , \mathbf { C } , } { \operatorname* { m i n } } } & { \parallel \mathcal { X } - \left\| \mathbf { A } , \mathbf { B } , \mathbf { C } \right\| \parallel ^ { 2 } + \parallel \mathcal { Y } - \left[ \mathbf { A } _ { v } , \mathbf { B } _ { v } , \mathbf { C } _ { v } \right] \parallel ^ { 2 } } & \\ { \mathbf { a } , \mathbf { A } _ { v } , \mathbf { B } _ { v } , \mathbf { C } _ { v } } & \\ { \mathrm { s . t . } } & { \mathbf { A } = \mathbf { H } _ { r } \pmb { \Delta } \mathrm { ~ a n d ~ } \mathbf { A } _ { v } = \mathbf { H } _ { v } \pmb { \Delta } , } & { ( 5 } \end{array}
$$

where $\mathbf { A } _ { v } \in \mathbb { R } ^ { I _ { v } \times R } , \mathbf { B } _ { v } \in \mathbb { R } ^ { J _ { v } \times R }$ and $\mathbf { C } _ { v } \in \mathbb { R } ^ { K _ { v } \times R }$ are the factor matrices of the simulated data in each mode; H and H are transformation matrices defining the coupling relation, i.e., indicating the matching metabolites in each dataset, and $\pmb { \Delta } \in \mathbb { R } ^ { I \times R }$ is the consensus matrix. Here we assume an R-component model for both datasets. When simulated data contains only a subset of the metabolites in the real data, we can define $\dot { \mathbf { H } } _ { r } = \mathbf { I } \in \mathbb { R } ^ { I \times I }$ and $\mathbf { H } _ { v } \in \mathbb { R } ^ { I _ { v } \times I }$ with a single 1 in each row matching the shared metabolites in X and all other entries zero. The coupling relation in (5) is relevant for our application of interest. More general linear coupling relations are also possible [11, 12] but not considered here.

Similarly, X and Y can be jointly analyzed using a coupled PARAFAC2-CP model [12] by solving the following optimization problem:

$$
\underset { \mathbf { A } _ { v } , \mathbf { B } _ { v } , \mathbf { C } _ { v } } { \operatorname* { m i n } } \sum _ { k = 1 } ^ { K } { \| \boldsymbol { \mathfrak { X } } _ { k } - \mathbf { A D } _ { k } \mathbf { B } _ { k } ^ { \intercal } \| ^ { 2 } } + \| \boldsymbol { \mathfrak { Y } } - \left[ \mathbf { A } _ { v } , \mathbf { B } _ { v } , \mathbf { C } _ { v } \right] \| ^ { 2 }\tag{6}
$$

$$
\begin{array} { r l } { \mathrm { s . t . } \quad } & { \mathbf { A } = \mathbf { H } _ { r } \mathbf { \Delta } \mathbf { \Delta } \mathbf { \Delta } , \ \mathbf { A } _ { v } = \mathbf { H } _ { v } \mathbf { \Delta } \mathbf { \Delta } \mathbf { \Delta } , } \\ { \quad } & { \mathbf { B } _ { k } ^ { \top } \mathbf { B } _ { k } = \Phi , \ \mathrm { f o r } \ k = 1 , \ldots , K , } \end{array}
$$

where real data X is approximated using a PARAFAC2 model to reveal subject-specific time profiles, Y is approximated using a CP model, and H and H can be defined as in (5).

## 3. EXPERIMENTS

## 3.1. Data

In our experiments, we use two metabolomics datasets: a real dataset from a meal challenge study and a simulated dataset generated using a human metabolic model. Both are arranged as metabolites by time points by subjects tensors. The goal is to reveal underlying patterns characterizing metabolic responses of subjects to a meal and study whether those patterns (in time and metabolites mode) can stratify subjects according to body composition and insulin resistance measures [1, 18]. Real data consists of measurements of samples collected during a meal challenge test from the $\mathrm { C O P S A C } _ { 2 0 0 0 }$ cohort [24] in which participants consumed a standardized meal after overnight fasting. Blood samples were taken at the fasting state (0hr) and seven time points after the meal (0.25hr, 0.5hr, 1hr, 1.5hr, 2hr, 2.5hr and 4hr), and measured using Nuclear Magnetic Resonance spectroscopy. In addition, we have measurements of insulin and C-peptide. Due to previously discovered sex-related differences in the metabolic response [1], we restricted our analysis to male subjects only. Four subjects were removed as outliers, as in previous studies [1, 13, 18]. We also removed seven subjects with missing insulin and C-peptide values across all time points. The data is a tensor of 161 metabolites × 8 time points × 133 subjects. Simulated data is from Li et al. [13] and was generated using a human whole-body metabolic model [25], which is a kinetic model defined by ordinary differential equations with metabolites as variables and kinetic constants as parameters, capturing multi-organ metabolic dynamics after a meal. Fifty virtual subjects were generated by randomly perturbing kinetic parameters to introduce individual variation, yielding cleaner metabolic dynamics than noisy real measurements. The simulated data contains concentrations of six blood metabolites/hormones: insulin (Ins), glucose (Glc), pyruvate (Pyr), lactate (Lac), alanine (Ala) and $\beta -$ hydroxybutyrate (Bhb), all of which are also available in the real data. These were measured at the same time points as in the real data, yielding a tensor of 6 metabolites × 8 time points × 50 virtual subjects.

## 3.2. Experimental setup

To assess the added value of joint analysis of real and simulated data, we compare the performance of (various) coupled models of real and simulated datasets with individual analysis of real data. We focus on the following settings:

CP (T0-corrected) vs. Linearly Coupled CP - CP (T0- corrected): We compare the CP model of the real data with the coupled CP model of real and simulated data with linear coupling in the metabolites mode as in (5). Datasets are T0- corrected, i.e., measurements at the first time point are subtracted from other time slices. Both datasets are centered across the subjects mode and scaled within the metabolites mode. We consider this setting since T0-correction was used in previous studies [1, 13].

CP vs. Linearly Coupled CP - CP: We compare the CP model of the real data with the coupled CP model of real and simulated data with linear coupling in the metabolites mode as in (5). When there is no T0-correction, datasets are nonnegative. We add nonnegativity constraints to all modes in (5), and scale the datasets within the metabolites mode.

PARAFAC2 vs. Linearly Coupled PARAFAC2 - CP: Finally, we compare a PARAFAC2 model of the real data with a PARAFAC2 model of the real data coupled with a CP model of the simulated data as in (6). We analyze the real data using a PARAFAC2 model to account for individual differences in time in terms of metabolic response to the meal. Again, there is no T0-correction and datasets are nonnegative. Nonnegativity constraints are added to all modes in (6), and datasets are scaled within the metabolites mode.

For all models, missing entries in real data (about 0.01%) are imputed by the mean of the corresponding subject and time fibers. For linearly coupled CP-CP models, we have a ridge penalty on all modes (which does not improve performance for other models; see Supplementary Material [SM]).

We assess the performance of the models in terms of pattern discovery based on the correlation between factors extracted from the subjects mode and additional subject information, i.e., body composition and insulin resistance measures including homeostatic model assessment for insulin resistance (HOMA-IR), muscle to fat ratio, fat percentage, muscle mass, weight, Body Mass Index (BMI), waist circumference, waist to height ratio, fat mass, fat mass index, and fatfree mass index (FFMI). Detailed descriptions are in [1].

Model selection. We select the number of components (R) when fitting (coupled) tensor factorizations by considering reproducibility and replicability of the components as well as their biological interpretation. Reproducibility measures the ability to recover a unique solution, while replicability measures whether similar patterns are found across overlapping subsets of the data [2]. Procedural details, reproducibility and replicability plots are included in the SM.

Implementation details. The CP models are fitted using cp wopt (from the Tensor Toolbox - version 3.7) using the limited memory BFGS algorithm. We use the Alternating Optimization (AO) - Alternating Direction Method of Multipliers (ADMM) framework for PARAFAC2 [26] and for linearly coupled tensor models [11, 12]. For all models, we use 100 random initializations and select the run with the lowest loss. The convergence is assessed using an absolute tolerance of $1 0 ^ { - 8 }$ and a relative tolerance of $1 0 ^ { - 6 }$ on the loss. The code and SM are available at: https://github.com/Gaute J1/CoupledModelsProject/.

![](images/21d826eb3fb5dfee5ef16d62c3f0d28371e1fab69ea742aaaf0d3aa526055289.jpg)  
Fig. 2: Correlations between subject coefficients and body composition and insulin resistance measures, for the component with the highest correlations in each model. Models are colored in pairs, comparing uncoupled (dark) and linearly coupled (light) factorizations.

## 4. RESULTS

Our results demonstrate that jointly analyzing noisy real data with simulated data improves the recovery of a biologically meaningful component. The clean simulated data has an underlying component modeling co-varying insulin and glucose levels. When coupled with real data, the simulated data guides the analysis and enables recovery of this component known to be strongly linked to a BMI-related phenotype [13]. In uncoupled models of real data, this component is either not captured because other metabolites dominate the model or is noisier. Therefore, the patterns extracted by the coupled models have consistently stronger correlations with body composition and insulin resistance measures (Fig. 2).

CP (T0-corrected) vs. Linearly Coupled CP-CP (T0- corrected): A 2-component CP model of T0-corrected real data reveals two metabolite factors, with the second one capturing a BMI-associated pattern [1]. However, this pattern is mainly dominated by metabolites other than insulin, glucose (see Supp Fig. 1, comp 2). When the real data is analyzed together with simulated data using a 3-component linearly coupled CP-CP model, we observe that coupling forces a pattern in the simulated data mainly capturing insulin and glucose into the model of the real data (see Supp Fig. 4, comp 3), yielding stronger BMI-associated correlations.

CP vs. Linearly Coupled CP-CP: A 6-component CP model of the real data with nonnegativity constraints reveals several factors relevant for BMI-related variables. Among them is a clear insulin-dominated factor, but glucose is not modeled well and has a small coefficient (Supp Fig. 7, comp 5). On the other hand, when real data is jointly analyzed with simulated data using a 6-component linearly coupled CP-CP model, the model finetunes this component, increasing the glucose coefficient while suppressing the other metabolites, resulting in a purer insulin/glucose component with stronger correlations with the variables of interest (Supp Fig. 10, comp 5).

PARAFAC2 vs. Linearly Coupled PARAFAC2-CP: The 6- component PARAFAC2 model of real data similarly recovers an insulin-dominated factor with a small glucose coefficient (Supp Fig. 13, comp 6). When the real data is jointly analyzed with simulated data using the 6-component linearly coupled PARAFAC2-CP model, again a cleaner insulin/glucose pattern (comp 6 in Fig. 3) is captured yielding stronger correlations with BMI-related variables. Fig. 3 shows the factors of the linearly coupled PARAFAC2-CP model, which allows for subject-specific time profiles in the real data. We only include the factors of this model here. All other models are in SM.

Fig. 4 highlights the insulin/glucose component across all models, illustrating how coupling consistently produces a cleaner pattern in terms of modeling insulin/glucose relation. Note that the simulated time profile of this component shows a sharper peak around 1 hour post-meal than observed in the real data, where the time pattern is much broader using CP and shows much more individual variation using PARAFAC2. This demonstrates a potential discrepancy between the real data and the computational model. Furthermore, coupled models (via the factors extracted from the subjects mode) show that there is much less individual variation in the simulated data compared to real data.

## 5. CONCLUSION

In this paper, we have introduced a knowledge guided approach for interpretable pattern discovery from complex data. We demonstrate that coupled tensor factorizations with linear coupling constraints are effective tools for bringing together data and computational models via joint analysis of real and simulated data. Our experiments on metabolomics data demonstrate that the proposed approach consistently provides cleaner patterns compared to the analysis of only real data. Besides, our results show that the coupled framework can reveal potential mismatches between real data and computational models. Future work should focus on whether conflicting information in simulated and real data can also be reliably extracted via coupled tensor factorizations, for instance, through unshared factors. We also plan to apply the proposed approach in neuroscience, where computational models and noisy high-dimensional measurements are common.

## 6. ACKNOWLEDGEMENTS

We thank the children and families of the COPS $\mathbf { A C } _ { 2 0 0 0 }$ cohort for their contribution, and the clinical team at COPSAC for conducting the clinical study.

## 7. REFERENCES

[1] S. Yan, L. Li, D. Horner, et al., “Characterizing human postprandial metabolic response using multiway

![](images/387c9c118d5b0ffe3ede351b1e2980e940b4da05c158ebd5458e10f08f63ef8c.jpg)

Fig. 3: 6-component linearly coupled PARAFAC2-CP model of real and simulated data, showing the linearly coupled metabolite factors (a), subject factors (c), simulated time profiles $( \mathbf { b } _ { v } )$ and virtual subject factors $( \mathbf { c } _ { v } )$ . Time profiles of the real data $( \mathbf { b } _ { k } )$ scaled by subject coefficients are shown using the mean (and standard error of the mean) of stratifications based on BMI and Insulin Resistance (IR), i.e., lower BMI: $\mathbf { B M I } < 2 5 ,$ higher BMI: BMI ≥ 25; NoIR: HOMA-IR≤ 2.91, IR: HOMA-IR > 2.91.  
![](images/5881a78edf37949edcaf70a4f908440a11301c0d57122d54f203a40f1634044c.jpg)  
Fig. 4: The component of interest $( \mathrm { i . e . }$ , the one modeling insulin/glucose) for each model, showing the linearly coupled metabolite factor (a), time profile (b), subject factor (c), simulated time profile $( \mathbf { b } _ { v } )$ and virtual subject factor $( \mathbf { c } _ { v } ) .$ . For PARAFAC2 and coupled PARAFAC2-CP, subject-specific time profiles scaled by the subject coefficients are shown using the mean (and standard error of the mean) of stratifications based on BMI and Insulin Resistance (IR) groups.

data analysis,” Metabolomics, vol. 20, no. 50, 2024.

[2] M. Mørup, E. Acar, and T. Adali, “Tensor and coupled decompositions for interpretable pattern discovery in multimodal functional neuroimaging data,” IEEE Signal Proc. Mag., vol. 42, no. 4, pp. 41–57, 2025.

[3] G. Ballard and T. G. Kolda, Tensor Decompositions for Data Science, Cambridge University Press, June 2025.

[4] C. Drieu, Z. Zhu, Z. Wang, et al., “Rapid emergence of latent knowledge in the sensory cortex drives learning,” Nature, vol. 641, no. 8064, pp. 960–970, 2025.

[5] C. Martino, L. Shenhav, C. Marotz, et al., “Contextaware dimensionality reduction deconvolutes gut microbial community dynamics,” Nature Biotechnology, vol. 39, pp. 165–168, 2021.

[6] G. E. Karniadakis, I. G. Kevrekidis, L. Lu, et al., “Physics-informed machine learning,” Nature Reviews Physics, vol. 3, no. 6, pp. 422–440, 2021.

[7] V. Monga, Y. Li, and Y. C. Eldar, “Algorithm unrolling: Interpretable, efficient deep learning for signal and image processing,” IEEE Signal Processing Magazine, vol. 38, no. 2, pp. 18–44, Mar. 2021.

[8] A. Karpatne, X. Jia, and V. Kumar, “Knowledge-guided machine learning: Current trends and future prospects,” arXiv:2403.15989, 2024.

[9] L. von Rueden, S. Mayer, K. Beckh, et al., “Informed machine learning - a taxonomy and survey of integrating prior knowledge into learning systems,” IEEE Trans. Knowl. Data Eng, pp. 1–1, 2023.

[10] D. M. Dunlavy, E. T. Phipps, H. Kolla, et al., “Goaloriented low-rank tensor decompositions for numerical simulation data,” arXiv:2508.11139, 2025.

[11] C. Schenker, J. E. Cohen, and E. Acar, “A flexible optimization framework for regularized matrix-tensor factorizations with linear couplings,” IEEE J. Sel. Topics Signal Process., vol. 15, no. 3, pp. 506–521, 2021.

[12] C. Schenker, X. Wang, D. Horner, et al., “PARAFAC2- based coupled matrix and tensor factorizations with constraints,” IEEE J. Sel. Topics Signal Process., vol. 19, no. 7, pp. 1461–1476, 2025.

[13] L. Li, H. Hoefsloot, B. M. Bakker, et al., “Longitudinal Metabolomics Data Analysis Informed by Mechanistic Models,” Metabolites, vol. 15, no. 1, 2025.

[14] R. A. Harshman, “Foundations of the PARAFAC procedure: Models and conditions for an ”explanatory” multimodal factor analysis,” UCLA Working Papers in Phonetics, vol. 16, pp. 1–84, 1970.

[15] J. D. Carroll and J.-J. Chang, “Analysis of individual differences in multidimensional scaling via an n-way generalization of “Eckart-Young” decomposition,” Psychometrika, vol. 35, no. 3, pp. 283–319, 1970.

[16] R. A. Harshman, “PARAFAC2: Mathematical and technical notes,” UCLA Working Papers in Phonetics, vol. 22, pp. 30–44, 1972.

[17] H.A.L. Kiers, J.M.F. Ten Berge, and R. Bro, “PARAFAC2—Part I. A direct fitting algorithm for the PARAFAC2 model,” Journal ofChemometrics, vol. 13, no. 3-4, pp. 275–294, 1999.

[18] C. Chatzis, D. Horner, R. Bro, et al., “Revealing subject-specific temporal patterns from longitudinal data,” bioRxiv, 2026.

[19] B. Erdos, C. Chatzis, J. Thorsen, et al., “Extracting hostspecific developmental signatures from longitudinal microbiome data,” PLOS Computational Biology, vol. 22, no. 7, pp. e1014486, 2026.

[20] K. H. Madsen, N. W. Churchill, and M. Mørup, “Quantifying functional connectivity in multi-subject fMRI data using component models: Quantifying functional connectivity,” Human Brain Mapping, vol. 38, no. 2, pp. 882–899, 2017.

[21] E. Acar, T. G. Kolda, and D. M. Dunlavy, “All-at-once optimization for coupled matrix and tensor factorizations,” arXiv:1105.3422, 2011.

[22] E. Acar, R. Bro, and A. Smilde, “Data fusion in metabolomics using coupled matrix and tensor factorizations,” Proceedings of the IEEE, vol. 103, no. 9, pp. 1602–1620, 2015.

[23] L. Sorber, M. Van Barel, and L. De Lathauwer, “Structured data fusion,” IEEE J. Sel. Topics Signal Process., vol. 9, no. 4, pp. 586–600, 2015.

[24] H. Bisgaard, “The Copenhagen Prospective Study on Asthma in Childhood (COPSAC): design, rationale, and baseline data from a longitudinal birth cohort study,” Annals of Allergy, Asthma & Immunology, vol. 93, no. 4, pp. 381–389, 2004.

[25] H. Kurata, “Virtual metabolic human dynamic model for pathological analysis and therapy design for diabetes,” iScience, vol. 24, no. 2, pp. 102101, 2021.

[26] M. Roald, C. Schenker, V. D. Calhoun, et al., “An AO-ADMM Approach to Constraining PARAFAC2 on All Modes,” SIAM Journal on Mathematics of Data Science, vol. 4, no. 3, pp. 1191–1222, 2022.