# OOD Detection for EEG-based Machine Learning in High-Risk Environments

Philipp Bomatter1\* and Henry Gouk1

1 School of Informatics, University of Edinburgh, United Kingdom.

\*Corresponding author(s). E-mail(s): philipp.bomatter@ed.ac.uk; Contributing authors: henry.gouk@ed.ac.uk;

## Abstract

Machine learning models for electroencephalography (EEG) analysis show great promise across a wide range of applications, but their deployment in high-risk domains is hindered by their vulnerability to distribution shifts. Encountering out-of-distribution (OOD) data can lead to catastrophic, overconfident predictive failures. While OOD detection methods can mitigate these risks, they remain heavily under-explored for EEG. Moreover, evaluations in the broader literature typically evaluate OOD detection performance in isolation, ignoring their practical impact on downstream applications. To bridge this gap, we introduce a benchmark for EEG OOD detection, evaluate a broad range of methods, and furthermore evaluate their value in two clinical downstream prediction task. Our results disentangle OOD detection and model uncertainty estimation capabilities, which are frequently conflated in the literature, provide actionable insights about the current state of the art for EEG OOD detection and model uncertainty estimation, and demonstrate how complementary methods for both aspects can be combined to form a robust safety net for the deployment of EEG-based machine learning models in real-world applications.

Keywords: EEG, OOD Detection, AI Safety, Robustness, Reliability

## 1 Introduction

Machine learning (ML) provides a promising avenue for supporting and enhancing the analysis of electroencephalography (EEG). However, especially in high-risk contexts like clinical applications, the use of machine learning-based EEG biomarkers is hindered by their vulnerability to distribution shifts. Machine learning models can fail catastrophically (and with high confidence) when they are presented with data that differs substantially from the data they were trained on, i.e. out-of-distribution (OOD) data [1]. Particularly for EEG, there are ample ways in which such shifts can be introduced, including different recording hardware, (accidental) mismatches in the preprocessing (e.g. filter settings or re-referencing), or differences in the patient population. While specific quality controls for some of these can be designed, part of the problem is that there can be unknown shifts that one does not expect. Furthermore, while for example individual differences between patients can be the cause of distribution shifts [2], only deploying a model for patients that the model has seen during training is often not an option. In such cases, it may not be clear a priori whether the EEG data from a new patient is similar enough to the data seen during training.

OOD detection can mitigate these risks by enabling a system to abstain from making a prediction or defer to a human expert when a case is likely out of distribution. Current OOD detection methods can broadly be categorized into two paradigms: discriminative and generative. Approaches such as Maximum Softmax Probability (MSP) [3], Energy scores [4], Out-of-Distribution Detector for Neural Networks (ODIN) [5], and Activation Shaping (ASH) [6] directly operate on the activations or outputs of a (discriminative) classification model. Conversely, methods based on a generative model—including likelihood estimates, Typicality [7], Density of State Estimation (DoSE) [8], and Signal in the Noise (SITN) [9—attempt to explicitly model the underlying distribution of the training data. While these methods have seen extensive development and benchmarking in domains like computer vision, their application to EEG data remains heavily under-explored. Furthermore, the standard evaluation paradigm in the broader OOD detection literature suffers from a significant disconnect from clinical utility. The vast majority of existing work evaluates OOD detection methods strictly in isolation, assessing only how well a method can separate an in-distribution dataset from a distinctly different out-of-distribution dataset (e.g. separating CIFAR-10 images from SVHN images). Crucially, such evaluations fail to reflect the downstream impact of OOD detection: when deploying a system for clinical diagnosis, the ultimate goal is not to merely detect OOD samples, but to understand how flagging such samples improves the reliability of the diagnosis.

To bridge this gap, this work presents a systematic evaluation of OOD detection for EEG, explicitly linking detection capabilities to downstream performance in clinical prediction tasks. We design a suite of practically relevant perturbations with different levels of severity and use them to benchmark a broad set of OOD detection methods spanning the aforementioned approaches based on both discriminative and generative models. In line with a recent theoretical critique and position paper [10, 11], our results disentangle two fundamentally distinct concepts that are frequently conflated in the literature: true OOD detection vs model uncertainty. Across perturbations, methods based on discriminative models were largely incapable of OOD detection while methods based on generative models achieved considerably better results, with increasing performance for higher perturbation severities. However, in downstream clinical evaluations—where model errors stem from both OOD data and difficult indistribution (ID) cases—we find that methods from both paradigms effectively predict model performance. While generative models recognise unfamiliar data, discriminative methods can serve as a valuable proxy for model uncertainty. Ultimately, we show that these complementary signals can be combined to form a more robust safety net for the deployment of EEG-based machine learning models in clinical and other high-risk applications.

In summary, our main contributions are as follows:

• We present an OOD detection evaluation framework for EEG that closes several gaps in the field. The framework comprises a suite of perturbations for the controlled creation of OOD EEG data, as well as investigations of downstream impacts in two clinically relevant prediction tasks.

• We benchmark a broad range of OOD detection methods, including both discriminative and generative approaches, the latter of which have not previously been probed in an EEG context, but achieved substantially better performance.

• We present strong empirical evidence that clearly disentangles the often conflated concepts of OOD detection and model uncertainty, a finding and experimental setup that is of broader interest to the OOD detection community beyond EEG.

• Our results provide actionable guidance on the current state of the art for EEG OOD detection and show how it can be combined with uncertainty estimation for the safer deployment of machine learning models in high-risk EEG applications.

## 2 Related Work

OOD Detection for EEG-based Machine Learning The EEG-specific literature on OOD detection is limited. In a recent preprint, Mulder et al. benchmarked several discriminative OOD detection methods in a motor imagery setting and found them to be largely ineffective [12]. Liu et al. proposed a temporal OOD detection method for EEG-based brain-computer interfaces (BCIs) and compared it against a range of standard discriminative OOD detection methods [13]. Our study investigates a more general setting than the latter and is focused on a clinical context rather than BCIs. Furthermore, while our results largely concur with their findings on discriminative methods, our experiments also include methods based on generative models, an important addition given that they displayed much better OOD detection capabilities. Concurrent work by Tveter et al. also introduced a range of perturbations to create EEG data with controlled distribution shifts [14]. However, their work is focused on the comparison of different ensemble learning methods and their robustness to distribution shifts rather than OOD detection.

OOD Detection Benchmarks in Other Domains Outside of EEG, the OOD detection literature is particularly advanced in computer vision. Notable works that have focused on the evaluation of different OOD detection methods are the OpenOOD benchmarks [15, 16], although many methods papers also include comprehensive comparisons against other methods [3–6, 8, 9]. Less mature, but more closely related to EEG, is the field of OOD detection for (more general) time series data. Within this field, the OOD detection benchmark for time series data by Gungor et al. is most relevant to our work [17]. However, like EEG-specific prior work, their benchmarks are limited to discriminative approaches, for which they reported relatively poor performance overall. Furthermore, a common thread across all these works is their primary reliance on semantic shifts to define OOD data, such as reserving a subset of classes as OOD or evaluating models across disjoint datasets. Guille-Escuret et al. recently criticised these approaches, arguing that an exclusive focus on semantic shifts neglects relevant (covariate) distribution shifts encountered by real-world systems [18]. Motivated by a practical perspective, we see the value of OOD detection in protecting against model failure, irrespective of the nature of the shift. Our work is therefore closely aligned with this view. Our work is also different in that we go beyond standard OOD detection evaluations to study impacts on downstream clinical applications and disentangle the concepts of true OOD detection and model uncertainty.

![](images/a0c2e7fa376a85c647dc4b903edfec8821cb7fc2b068e52f1fb5dc15f1307176.jpg)  
Fig. 1 OOD Detection Evaluation Concept. Starting from an EEG dataset with annotations for a prediction task (here TUAB with EEG normality labels), the data is first partitioned into training and test splits. The training data is then used to train two models: a generative model trained using only the EEG without labels (here a UNet trained via flow-matching), and a discriminative model trained through supervised learning (here a TCN model). The test data then serves as in-distribution test data, for which corresponding out-of-distribution data is generated through controlled perturbations. Finally, we evaluate a range of OOD detection methods based on either the generative or discriminative model. In addition to OOD detection performance, we also evaluate its impact on downstream performance, quantifying how the prediction accuracy of the discriminative model varies across samples with different OOD scores of the different methods.

## 3 Methods

Figure 1 provides an overview of the components that enable our EEG OOD detection evaluations. Starting from an EEG dataset with corresponding annotations for a (clinical) prediction task, samples are first partitioned into a training and test split. The training data is used to train two models: a generative model trained using only the EEG without labels (i.e. unsupervised), and a discriminative model trained through supervised learning. The test set then serves as in-distribution test data, for which we generate corresponding out-of-distribution data through controlled perturbations. Finally, we evaluate a range of OOD detection methods based on either the generative or discriminative model in two ways. First, we assess their OOD detection performance, i.e. how well their assigned OOD scores discriminate between the unperturbed ID test samples and their corresponding perturbed OOD versions. Second, we also evaluate the impact of OOD detection on downstream performance in the (clinical) prediction task, quantifying how the prediction accuracy of the discriminative model varies across samples with different OOD scores of the different methods.

Detailed descriptions for the different components are given in the following sections. We start with the EEG perturbation suite that forms the backbone of our benchmarks in Section 3.1. Section 3.2 then outlines the different OOD detection methods included in our benchmarks along with architecture and training details for the underlying models. Information about the datasets, prediction tasks, data splitting, and preprocessing is provided in Section 3.3.

## 3.1 OOD Perturbation Suite for EEG

We introduce a suite of controlled perturbations to create OOD samples from indistribution test data. These transformations were designed to simulate plausible distribution shifts that may be encountered in real-world EEG data, though they are fundamentally intended to serve as a structured, reproducible, and well-controlled testbed rather than an exhaustive modelling of reality. Furthermore, the perturbations were designed to be parametrisable, which (with the exception of the re-referencing perturbation) enables fine-grained control of the severity of the distribution shift. Finally, by applying these transformations directly to the ID test set, the resulting OOD data is perfectly paired with the unperturbed baseline, providing the methodological capacity to isolate the exact effect of a shift on model performance. The perturbations are illustrated in Figure 2 and described in the following paragraphs.

Sampling Rate The signal is resampled to a higher sampling rate (using bandlimited sinc interpolation via the torchaudio library) and then cropped to maintain the same number of samples. From the perspective of a model that expects data to be at a lower sampling rate, perturbed samples will appear smoothed. Sampling rate mismatches are surprisingly common in practice; for instance, a common issue in the evaluation of EEG foundation models is that they are applied to data with a different sampling rate than the model was pre-trained on. We control the severity of this perturbation by varying the amount by which the sampling rate is increased compared to the 100 Hz that the model was trained on, ranging from 125 Hz over 150 Hz to 200 Hz.

Channel Order The ordering of a subset of EEG channels is permuted. From the perspective of a model that was trained with a fixed given channel order, the perturbation will lead to strange topographies with, for example, eye blinks appearing in presumed occipital channels. In practice, perturbations like this could be introduced by accident with some recording devices where electrodes are plugged into the amplifier individually. We control the severity of this perturbation by varying the size of the subset of randomly selected channels which are permuted. We increase the subset from 10%, i.e. swapping just two channels in the 19-channel layout, to 50% and then 100%.

Reference Scheme The EEG channels are re-referenced to a different scheme. Starting from average-referenced data that is used to train the model, we re-reference to three different schemes: a monopolar vertex reference with Cz as the reference channel, a linked-temporal reference with the average of T7 and T8 as the reference, and a longitudinal double-banana bipolar montage with an additional transverse channel (T7-T8) to maintain the same channel count. Unlike the different levels for other perturbations, these three options do not have a clear ordering in terms of perturbation severity.

![](images/5a57a1469efd3148293dbe29e32e47d23d5ad7e42fa8289f09605f4a19f140eb.jpg)  
Fig. 2 Illustration of EEG OOD Perturbations. An unperturbed (ID) EEG segment is shown (top left panel) along with corresponding OOD samples obtained through the different perturbations. For visualisation purposes, only a subset of channels are shown.

High-pass Filter The signal is filtered with an additional zero-phase Butterworth high-pass filter, attenuating low-frequency content. Starting from the unperturbed data (high-pass filtered at 1 Hz), we vary the severity of this perturbation by increasing the cutoff frequency starting from 2 Hz to 4 Hz and 8 Hz, progressively removing more low-frequency activity.

Low-pass Filter Analogous to the high-pass filter perturbation described above, the signal is filtered with a low-pass filter. Starting from the unperturbed data (lowpass filtered at 50 Hz), we vary the severity of this perturbation by lowering the cutoff frequency starting from 45 Hz to 30 Hz and 15 Hz, progressively removing more high-frequency activity.

## 3.2 OOD Detection Methods

Our benchmarks include popular OOD detection methods grouped into two paradigms: methods based directly on discriminative classification models and methods based on generative models.

## 3.2.1 Discriminative-model-based Methods

In the following, we outline the OOD detection methods included in our benchmarks that are directly based on the discriminative model used for the given classification task. As the underlying discriminative model for all methods, we use the TCN architecture originally proposed by Bai et al. [19], which has since been adapted and widely used for EEG data [20] (see Appendix E for hyperparameters and training details).

Maximum Softmax Probability (MSP) MSP uses the maximum probability from the network's final softmax layer as a confidence score [3]. This is based on the assumption that the model will output lower maximum probabilities (reflecting higher uncertainty) when presented with OOD data compared to ID data.

Energy Score Instead of operating on the softmax outputs, which are normalised to sum to one, Liu et al. proposed computing an energy score directly from the model's unnormalised logits [4]. OOD samples are expected to produce higher energy scores.

Out-of-Distribution Detector for Neural Networks (ODIN) ODIN builds upon the standard MSP approach through two key modifications: applying temperature scaling to the model's logits prior to the softmax computation, and adding gradient-based perturbations to the input data that are expected to have a stronger impact on ID than OOD data [5]. Together, these modifications aim to improve the separability of ID and OOD samples with the resulting score. We adopt the hyperparamters reported in the original publication for the main results (T=1000, €=0.0014). We did not perform hyperparameter tuning because we do not assume access to OOD data in our experiments. Furthermore, the original publication suggests relatively good hyperparameter transferability.

Activation Shaping (ASH) Djurisic et al. introduced an activation shaping (ASH) method that intervenes in the forward pass of a model by pruning and transforming activations before they reach the final classification head, after which the energy score is computed on the resulting logits [6]. We specifically use the ASH-S version with a pruning percentage of 65%. While tuning of this hyperparameter would require access to OOD data, which we do not assume in our benchmarks, the authors report relatively stable performance across different parameter values in the original publication.

## 3.2.2 Generative-model-based Methods

In the following, we outline the OOD detection methods included in our benchmarks that are based on a generative model. As the underlying generative model for all methods, we use the UNet architecture following Ho et al. [21], as implemented by Dhariwal and Nichol [22], and train it using the flow matching objective by Lipman et al. [23] (see Appendix E for details). We use Euler's method (with 100 timesteps), implemented in the library by Lipman et al. [24], as ODE solver to obtain the likelihoods and noise samples required by the different methods.

Log-likelihood The log-likelihood is evaluated via the instantaneous change of variables formula, calculated as the sum of the log-probability of the latent representation and the negative integral of the vector field's divergence over the flow trajectory. Samples with low likelihood under the generative model are flagged as OOD, reflecting the assumption that the model assigns higher probability densities to ID data.

Typicality Typicality was introduced by Nalisnick et al. to correct for a failure mode of likelihoods in high dimensions where regions of high probability density can lie outside of the so-called typical set that contains almost all probability mass [7]. It is defined as the absolute difference between the negative log-likelihood and an estimate of the entropy. We estimate the entropy as the average negative log-likelihood over the (in-distribution) validation data, following experiments in [9] that showed negligible differences to estimation on the training data as in the original publication. Furthermore, we use the single-sample version as in [8].

Density of State Estimation (DoSE) DoSE measures the empirical density of multiple statistics computed on in-distribution data (as for Typicality, we use (indistribution) validation data following experiments in [9]) [8]. We specifically use the DoSEkDE version for flow-based models, which is based on three statistics: the loglikelihood, the log-probability of the latent representation, and the log-determinant of the Jacobian (corresponding to the integral of the vector field's divergence for the continuous-time flow matching model used in our experiments). A 1D Kernel Density Estimator (KDE) is fit to each statistic and the final OOD score is computed by summing the log-probabilities from the individual KDEs. As in the original publication, the default SciPy KDE implementation with a Gaussian kernel and automatic determination of the bandwidth parameter through Scott's rule is used, and following [9], the KDEs are fit on a random subset of 10,000 samples for computational efficiency.

Signal in the Noise (SITN) SITN uses the diffeomorphic properties of continuous normalising flows to detect OOD samples by checking if their corresponding noise samples (obtained through backwards integration along the probability flow ODE) are consistent with the Gaussian prior used during training [9].

## 3.3 Data

Experiments were conducted on two large datasets (TUAB, CAUEEG) with annotations for clinical prediction tasks. With more than 1,000 participants each, these datasets are among the largest EEG datasets available for research purposes, providing us with the necessary sample size for robust training and benchmarking of machine learning methods.

TUAB The Temple University Hospital Abnormal (TUAB) EEG Corpus [25, 26] comprises a demographically balanced and curated subset of more than 2,300 participants from the TUH EEG Corpus [27]. The data was collected in a hospital environment and recordings were annotated by board certified neurologists for the normality of the EEG according to standardised criteria [28]. We specifically used version 3.0.1 of the dataset and followed the data curation and splitting in [2], which respects the official train-test split and additionally defines a validation split with 10% of the training data. All splitting is performed on the participant level, i.e. all samples of a given participant are assigned to the same split. Data access requests should be directed to the dataset authors.

CAUEEG The Chung-Ang University Hospital EEG (CAUEEG) dataset [29] includes a subset of 1,122 participants to which diagnostic labels for the categories normal, mild cognitive impairment (MCI), or dementia were assigned by neurologists based on a neuropsychological examination. We used the official train-validation-test split without overlapping participants. Data access for academic and research purposes can be requested from the dataset authors.

Preprocessing The data preprocessing comprised the following steps: selection of a fixed channel subset (the 19 channels in the 10-20 system), band-pass filtering to 1-50 Hz, resampling to 100 Hz, and re-referencing to average reference. Finally, nonoverlapping epochs with 2s duration were used.

## 4 Results

## 4.1 OOD Detection Performance

Table 1 shows the Area Under the Receiver Operating Characteristic (AUROC) curve for the OOD detection performance across the different perturbations and severity levels for all methods on the TUAB dataset. Models for the discriminative and generative methods were trained on the train split of the (unperturbed) dataset and OOD detection methods were then evaluated by using the unperturbed test split of the dataset as ID samples and the perturbed test data as OOD samples.

Table 1 OOD detection performance on TUAB. Performance is reported in terms of AUROC and bold values highlight the best result for each perturbation and severity level. Discriminative-model-based methods (MSP, Energy, ODIN, ASH) perform largely at chance level, whereas generative-model-based methods achieve considerably better OOD detection with increasing performance for higher severity levels. SITN consistently achieves the best performance.
<table><tr><td>Perturbation</td><td>Severity</td><td>MSP</td><td>Energy</td><td>ODIN</td><td>ASH</td><td>LL</td><td>Typ.</td><td>DoSE</td><td>SITN</td></tr><tr><td rowspan="3">Resampling</td><td>125 Hz</td><td>0.517</td><td>0.500</td><td>0.514</td><td>0.533</td><td>0.369</td><td>0.559</td><td>0.556</td><td>0.947</td></tr><tr><td>150 Hz</td><td>0.457</td><td>0.479</td><td>0.453</td><td>0.534</td><td>0.275</td><td>0.646</td><td>0.647</td><td>0.981</td></tr><tr><td>200 Hz</td><td>0.383</td><td>0.451</td><td>0.377</td><td>0.522</td><td>0.143</td><td>0.816</td><td>0.826</td><td>0.989</td></tr><tr><td rowspan="3">Channel Shuffle</td><td>10%</td><td>0.549</td><td>0.494</td><td>0.549</td><td>0.513</td><td>0.607</td><td>0.521</td><td>0.543</td><td>0.754</td></tr><tr><td>50%</td><td>0.553</td><td>0.431</td><td>0.555</td><td>0.494</td><td>0.793</td><td>0.682</td><td>0.711</td><td>0.924</td></tr><tr><td>100%</td><td>0.526</td><td>0.391</td><td>0.529</td><td>0.460</td><td>0.823</td><td>0.726</td><td>0.748</td><td>0.942</td></tr><tr><td rowspan="3">Re-referencing</td><td>Cz</td><td>0.517</td><td>0.435</td><td>0.520</td><td>0.597</td><td>0.950</td><td>0.937</td><td>0.987</td><td>0.998</td></tr><tr><td>T7, T8</td><td>0.587</td><td>0.513</td><td>0.591</td><td>0.568</td><td>0.927</td><td>0.896</td><td>0.973</td><td>0.997</td></tr><tr><td>bipolar</td><td>0.752</td><td>0.621</td><td>0.757</td><td>0.719</td><td>0.940</td><td>0.919</td><td>0.962</td><td>0.996</td></tr><tr><td rowspan="3">High-pass Filter 4 Hz</td><td>2 Hz</td><td>0.514</td><td>0.558</td><td>0.511</td><td>0.506</td><td>0.475</td><td>0.504</td><td>0.503</td><td>0.604</td></tr><tr><td></td><td>0.531</td><td>0.603</td><td>0.526</td><td>0.550</td><td>0.445</td><td>0.510</td><td>0.510</td><td>0.749</td></tr><tr><td>8 Hz</td><td>0.446</td><td>0.684</td><td>0.439</td><td>0.637</td><td>0.367</td><td>0.546</td><td>0.544</td><td>0.872</td></tr><tr><td rowspan="3">Low-pass Filter</td><td>45 Hz</td><td>0.499</td><td>0.502</td><td>0.499</td><td>0.503</td><td>0.461</td><td>0.507</td><td>0.505</td><td>0.822</td></tr><tr><td>30 Hz</td><td>0.500</td><td>0.511</td><td>0.498</td><td>0.515</td><td>0.246</td><td>0.649</td><td>0.644</td><td>0.986</td></tr><tr><td>15 Hz</td><td>0.511</td><td>0.536</td><td>0.506</td><td>0.560</td><td>0.016</td><td>0.975</td><td>0.974</td><td>0.990</td></tr></table>

OOD detection methods operating on discriminative models that were trained in a supervised way on the normality prediction task (MSP, Energy, ODIN, ASH) were largely ineffective for differentiating between in- and out-of-distribution data. These methods performed close to chance level (AUROC of 0.5) for most conditions and showed inconsistent trends with perturbation severity. Energy scores, for instance, showed increasingly better OOD detection performance with higher severities of the high-pass filter perturbation, but decreasing performance for increasing severities of the resampling perturbation.

Generative models (in particular Typicality, DoSE, and SITN), on the other hand, achieved much better OOD detection overall with consistent severity-dependent performance improvements. SITN clearly emerged as the most effective OOD detection method with strong performance even at the lowest severity level and for perturbations where other methods performed close to chance level (see for example the channel shuffle perturbation at the lowest severity level where only two channels are flipped). Raw log-likelihoods worked well for certain perturbations (Channel Shuffle, Re-referencing), but not others, exhibiting a behaviour previously observed in the computer vision literature, which we discuss in more detail in Appendix D.

Evaluations on CAUEEG corroborate these results, except that DoSE narrowly outperformed SITN for the Re-referencing perturbation and the highest severity Lowpass Filter setting (see Appendix A). Additional results for Mahalanobis distances can be found in Appendix B.

## 4.2 Downstream Performance Impact

Panel A in Figure 3 shows the impact of OOD detection on the clinical downstream EEG normality prediction task on the TUAB dataset (see Section 3.3). The figure shows one method from each paradigm (MSP as a discriminative method and SITN as a generative method) where the resampling perturbation (severity 200 Hz) was used for the OOD data. For each method, the pooled in- and out-of-distribution samples were sorted according to the assigned OOD scores. Then we report the (downstream) classification accuracy of the discriminative model across different percentile bins of the OOD score (e.g. the performance across the samples with the lowest 0.1% MSP scores, the 0.1-1% lowest MSP scores, etc.). The arrows in the top left/right corners of the plots indicate the OOD direction for each method: for MSP, OOD samples are expected to have lower scores, whereas SITN should assign higher scores to OOD samples. Finally, the histograms show the sample counts in each percentile bin with the different colours indicating the proportion of ID and OOD samples.

For MSP, the proportion of ID and OOD samples is relatively similar across the bins, consistent with the poor OOD detection AUROC values reported in Section 4.1. MSP does not separate ID and OOD samples well. Despite this fact, MSP values are still predictive of model performance, except for a failure mode at very high confidence values. SITN, on the other hand, separates ID and OOD samples extremely well. It can be seen that the model performance is substantially lower on OOD data, dropping almost to chance level, such that OOD detection provides a very valuable safety net to prevent misclassifications.

A  
![](images/c1831698e109b2bed5bcfa51ef717ca9d856a7ccd63276e88c70a6a7b4cbd200.jpg)

![](images/8ba8dc1d9a48d7e9967114653a7ae7875b7faa633ab67e962ad7e3cb5634452c.jpg)

B  
![](images/48ffaa719845330b64fb1ad7bf0c38c131048ead263368af93d423c9893aeb54.jpg)

![](images/de971a0e4c7a1009e4092c44f2d88330e473a3fdcecbf574652edc4ac4512bf5.jpg)  
Fig. 3 Model performance for pathology prediction on the TUAB dataset. A ID samples and corresponding OOD samples obtained with the resampling perturbation (severity 200 Hz). Performance is shown in terms of accuracy across percentile bins, where the x ticks indicate the upper edge of the corresponding bin (i.e. the first bin contains samples from the 0-0.1 percentile, the second one 0.1-1, etc.). The histograms indicate the number of ID and OOD samples in each bin. The arrows in the top left/right of each plot indicate the OOD direction for the corresponding metric. See Appendix H for full downstream performance results across all perturbations and OOD detection methods. B The same visualisations restricted to only in-distribution data. MSP is a good indicator of model performance across in-distribution samples except for a failure mode at the highest MSP scores, whereas SITN provides little information about model performance between ID samples.

In Panel B, we present the same visualisations on ID data (i.e. the unperturbed test data) alone. It can be seen that MSP scores are much more predictive of model performance than SITN scores within ID samples. Generative approaches like SITN, which do not have access to the discriminative model used for the classification task, provide little indication for model performance within the majority of in-distribution samples, but they do not have the aforementioned failure mode of discriminative methods at very high confidence levels. A performance drop can be observed for ID samples with the most extreme (top 1%) SITN scores. These samples likely represent rare in-distribution samples to which the model had very low exposure during training.

Full results for all perturbations and OOD detection methods as well as evaluations on the dementia diagnosis task on CAUEEG can be found in Appendix H. While the effect on downstream performance was smaller for certain conditions (e.g. Low-pass Filter on CAUEEG; Figure H17), the general patterns are highly consistent.

![](images/ce55b9e686bcec737874c3da68c12b03caf93fb9941cf956444c5c41a5b00a24.jpg)  
Fig. 4 Decision Diagram for Combined OOD and Uncertainty Detection. Samples are jointly probed by an OOD detector and a model uncertainty estimator, each of which can trigger the system to abstain from prediction or defer to a human expert to prevent misclassifications.

Overall, these results clearly disentangle two aspects that influence downstream performance. Generative methods like SITN excel at true OOD detection, identifying samples outside the training distribution, for which model predictions should not be trusted because the model was not exposed to comparable data during training. Discriminative methods like MSP, on the other hand, are not suitable for OOD detection, but reflect model uncertainty. Even with ample exposure to similar samples during training, certain cases can be ambiguous and difficult for the model to classify. Model uncertainty can capture such cases and provide valuable information about how well the model is likely to perform across different in-distribution samples.

## 4.3 Combination of OOD Detection and Model Uncertainty

The results in Section 4.2 suggest that it could be extremely useful to combine OOD detection and model uncertainty methods. OOD detection methods can provide protection against highly confident mispredictions when the model encounters out-ofdistribution samples, while model uncertainty methods function well for in-distribution data, where they provide valuable estimates of the model performance.

We evaluated a system following the decision diagram in Figure 4, where samples are assessed by both an OOD detector and a model uncertainty estimator. The classifier then only provides predictions for samples that pass both tests and abstains from prediction otherwise. We used SITN for OOD detection, following the calibration procedure detailed in the original publication [9], which allows one to pick a threshold with access to only an in-distribution validation set based on an acceptable false positive rate (FPR). A target FPR of 1% was chosen for this example. To control model uncertainty, we first used isotonic regression (fitted on the in-distribution validation set) to map raw MSP values to calibrated probabilities. We then thresholded samples as too uncertain if the calibrated probability of the prediction being correct was below 60%. The choice of this threshold will in practice depend on the costs of misclassifications vs abstaining from prediction or deferring to a human expert.

Results on TUAB are shown in Table 2. Focusing on the baseline performance without any rejections (coverage 100%) first, it can be seen that the accuracy dropped from 80.5% on ID data to 67.5% when corresponding OOD samples (resampling perturbation at 200 Hz) were included in the evaluation. When SITN was used to reject OOD samples, performance remained stable across both conditions. Note also that, as expected, the coverage on ID data is close to the desired 1% FPR. Using MSP to reject high uncertainty samples improved performance on ID data (at the cost of a lower coverage). When OOD samples were included in the evaluation, MSP did not protect well against the drop in performance, with only marginally better accuracy than the baseline despite rejecting more than 10% of the samples. Combining the two methods resulted in the best accuracy across both conditions.

Table 2 Selective Classification Performance for OOD Detection and Model Uncertainty Estimation. OOD Detection (using SITN), Model Uncertainty (using MSP), and their combination (SITN+MSP) are compared on ID-only data and ID+OOD data. OOD data was generated with the resampling perturbation at 200 Hz. MSP provides an accuracy improvement on in-distribution data, but only SITN provides effective protection against OOD data. Combining both methods enables the highest accuracy across both in- and out-of-distribution data.
<table><tr><td></td><td colspan="2">ID Only</td><td colspan="2">ID and OOD</td></tr><tr><td>Rejection Method</td><td>Coverage</td><td>Accuracy</td><td>Coverage</td><td>Accuracy</td></tr><tr><td>Baseline (No Rejection)</td><td>100.0%</td><td>80.5%</td><td>100.0%</td><td>67.5%</td></tr><tr><td>SITN (OÒD Detection)</td><td>98.5%</td><td>80.8%</td><td>49.5%</td><td>80.7%</td></tr><tr><td>MSP (Model Uncertainty)</td><td>88.1%</td><td>83.8%</td><td>89.2%</td><td>69.3%</td></tr><tr><td>Combined (SITN + MSP)</td><td>86.9%</td><td>84.1%</td><td>43.7%</td><td>84.1%</td></tr></table>

## 5 Discussion and Conclusions

In this study, we introduced an OOD detection benchmark for EEG-based machine learning and systematically evaluated a broad spectrum of both discriminative and generative methods. Our results clearly demonstrate that standard discriminative methods are incapable of reliably detecting OOD EEG data under realistic covariate shifts, a finding that aligns with limited prior work in the EEG BCI literature [12, 13]. However, generative approaches—which to the best of our knowledge have not been previously benchmarked for EEG or general time-series OOD detection—yield substantially better performance. In particular, Signal in the Noise (SITN) establishes a new state of the art for EEG OOD detection.

Follow-up evaluations to assess the utility of OOD detection in two clinical downstream prediction tasks demonstrated two things. First, performance dropped substantially on out-of-distribution samples, emphasising the value of OOD detection to prevent mispredictions in high-risk applications. Second, both discriminative and generative methods were predictive of downstream performance, but for different reasons. Only generative methods protect against model failure by flagging OOD samples. Discriminative methods prevent mispredictions by highlighting samples with high model uncertainty, even if they are in-distribution. These results provide strong support for the position recently put forward by Li et al., which argues that discriminative methods “answer the wrong question for OOD detection"[10]. We furthermore demonstrated how the complementary capabilities of discriminative and generative methods can be combined to protect against both failure modes.

The present study has some limitations that highlight valuable directions for future work. While the creation of OOD data through the perturbations presented in this work has several advantages (unambiguous ground truth, perfect pairing with ID samples, large sample size, control of severity levels), the field would benefit from additional benchmarks and curated datasets to study a wider variety of distribution shifts. Designing such OOD detection setups is more challenging than in computer vision. While images from different datasets (e.g. CIFAR-10 vs SVHN) or different object categories (e.g. cats vs cars) are clearly distinct, comparing EEG segments across different datasets is more nuanced. Besides clear differences in recording hardware or preprocessing, aspects of which are already reflected in our perturbation experiments, the degree to which differences in patient populations manifest in EEG distribution shifts is hard to quantify. Even for comparatively well-studied variables like sex or age with known population-level differences [30–32], a lot of the variance is driven by individual differences. Moreover, cross-dataset label definitions are often not mutually exclusive. EEG segments in the TUAB dataset could in principle also be assigned a simultaneous label for the dementia diagnosis task in CAUEEG, such that CAUEEG segments need not necessarily be OOD for a TUAB-trained model (see exploratory analysis in Appendix F). Where possible, we recommend grounding OOD detection through simultaneous assessments of performance in a relevant downstream task as in our experiments. This ensures that progress in OOD detection will translate to superior safeguards in practical applications.

Our results also raise questions about benchmarking practices in the broader OOD detection literature. How can the poor performance of discriminative methods in our evaluations be reconciled with their success on widely used computer vision benchmarks like OpenOOD? We hypothesise that this discrepancy stems from fundamental differences in how OOD evaluation is framed. For one, established leaderboards frequently permit the use of auxiliary OOD data during training. Our setup, by contrast, strictly assumes no prior access to OOD data, reflecting a more realistic scenario where models encounter completely unanticipated shifts. Furthermore, OOD detection benchmarks are dominated by semantic shifts [18]. An influential survey paper by Yang et al. even frames this exclusive focus as part of the general definition of OOD detection, although it is acknowledged that other shifts may be considered OOD from a generalisation perspective or when considering high-risk applications [33]. We encourage future applied work to place practical utility at its centre. In clinical domains, the primary objective of OOD detection is to prevent model failure caused by unanticipated distribution shifts. Crucially, achieving this requires benchmarks designed to prevent the entanglement of true OOD detection and model uncertainty. As our results suggest, these are fundamentally distinct capabilities and conflating the two obscures what a given method actually achieves. By designing benchmarks that explicitly separate OOD detection from model uncertainty estimation, the field can make targeted progress on both fronts.

Acknowledgements. This project was supported by the Royal Academy of Engineering under the Research Fellowship programme.

## References

[1] Nguyen, A.M., Yosinski, J., Clune, J.: Deep neural networks are easily fooled: High confidence predictions for unrecognizable images. In: IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2015, Boston, MA, USA, June 7-12, 2015, pp. 427–436 (2015). https://doi.org/10.1109/CVPR.2015.7298640

[2] Bomatter, P., Gouk, H.: Is Limited Participant Diversity Impeding EEG-based Machine Learning? In: Proc. of NeurIPS, vol. 38, pp. 117447–117474 (2025)

[3] Hendrycks, D., Gimpel, K.: A baseline for detecting misclassified and out-ofdistribution examples in neural networks. In: Proc. of ICLR (2017)

[4] Liu, W., Wang, X., Owens, J.D., Li, Y.: Energy-based out-of-distribution detection. In: Proc. of NeurIPS (2020)

[5] Liang, S., Li, Y., Srikant, R.: Enhancing the reliability of out-of-distribution image detection in neural networks. In: Proc. of ICLR (2018)

[6] Djurisic, A., Bozanic, N., Ashok, A., Liu, R.: Extremely simple activation shaping for out-of-distribution detection. In: Proc. of ICLR (2023)

[7] Nalisnick, E., Matsukawa, A., Teh, Y.W., Lakshminarayanan, B.: Detecting Out-of-Distribution Inputs to Deep Generative Models Using Typicality. ArXiv preprint abs/1906.02994 (2019)

[8] Morningstar, W.R., Ham, C., Gallagher, A.G., Lakshminarayanan, B., Alemi, A.A., Dillon, J.V.: Density of states estimation for out of distribution detection. In: The 24th International Conference on Artificial Intelligence and Statistics, AISTATS 2021, April 13-15, 2021, Virtual Event. Proceedings of Machine Learning Research, vol. 130, pp. 3232–3240 (2021)

[9] Bomatter, P., Geary, J., Gouk, H.: The Signal in the Noise: OOD Detection Through Goodness-of-Fit Testing in Factorised Latent Spaces. ArXiv preprint abs/2605.22496 (2026)

[10] Li, Y.L., Lu, D., Kirichenko, P., Qiu, S., Rudner, T.G.J., Bruss, C.B., Wilson, A.G.: Position: Supervised Classifiers Answer the Wrong Questions for OOD Detection. In: Proceedings of the 42nd International Conference on Machine Learning, pp. 81689–81713 (2025)

[11] Li, Y.L., Lu, D., Kirichenko, P., Qiu, S., Rudner, T.G.J., Bruss, C.B., Wilson, A.G.: Out-of-Distribution Detection Methods Answer the Wrong Questions. ArXiv preprint abs/2507.01831 (2025)

[12] Mulder, M.Q., Valdenegro-Toro, M., Sburlea, A.I., Jong, I.P.d.: The Challenge of Out-Of-Distribution Detection in Motor Imagery BCIs. ArXiv preprint abs/2603.13324 (2026)

[13] Liu, C., Li, S., Tan, L., Wu, D.: Temporal Out-of-Distribution Detection for Asynchronous Motor Imagery Brain-Computer Interfaces. ArXiv preprint abs/2605.01014 (2026)

[14] Tveter, M., Tveitstøl, T., Hatlestad-Hall, C., Hammer, H.L., Haraldsen, I.R.J.H.: Uncertainty in deep learning for EEG under dataset shifts. Artificial Intelligence in Medicine 174, 103374 (2026) https://doi.org/10.1016/j.artmed.2026.103374

[15] Yang, J., Wang, P., Zou, D., Zhou, Z., Ding, K., Peng, W., Wang, H., Chen, G., Li, B., Sun, Y., Du, X., Zhou, K., Zhang, W., Hendrycks, D., Li, Y., Liu, Z.: Openood: Benchmarking generalized out-of-distribution detection. In: Proc. of NeurIPS (2022)

[16] Zhang, J., Yang, J., Wang, P., Wang, H., Lin, Y., Zhang, H., Sun, Y., Du, X., Li, Y., Liu, Z., Chen, Y., Li, H.: OpenOOD v1.5: Enhanced Benchmark for Out-of-Distribution Detection. ArXiv preprint abs/2306.09301 (2023)

[17] Gungor, O., Rios, A.S., Ahuja, N., Rosing, T.: TS-OOD: Evaluating Time-Series Out-of-Distribution Detection and Prospective Directions for Progress. ArXiv preprint abs/2502.15901 (2025)

[18] Guille-Escuret, C., Noël, P., Mitliagkas, I., Vázquez, D., Monteiro, J.: Expecting the unexpected: Towards broad out-of-distribution detection. In: Proc. of NeurIPS (2024)

[19] Bai, S., Kolter, J.Z., Koltun, V.: An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling. ArXiv preprint abs/1803.01271 (2018)

[20] Gemein, L.A.W., Schirrmeister, R.T., Chrabaszcz, P., Wilson, D., Boedecker, J., Schulze-Bonhage, A., Hutter, F., Ball, T.: Machine-learning-based diagnostics of EEG pathology. NeuroImage 220, 117021 (2020) https://doi.org/10.1016/j. neuroimage.2020.117021

[21] Ho, J., Jain, A., Abbeel, P.: Denoising diffusion probabilistic models. In: Proc. of NeurIPS (2020)

[22] Dhariwal, P., Nichol, A.Q.: Diffusion models beat gans on image synthesis. In: Proc. of NeurIPS, pp. 8780–8794 (2021)

[23] Lipman, Y., Chen, R.T.Q., Ben-Hamu, H., Nickel, M., Le, M.: Flow matching for generative modeling. In: Proc. of ICLR (2023)

[24] Lipman, Y., Havasi, M., Holderrieth, P., Shaul, N., Le, M., Karrer, B., Chen, R.T.Q., Lopez-Paz, D., Ben-Hamu, H., Gat, I.: Flow Matching Guide and Code. ArXiv preprint abs/2412.06264 (2024)

[25] López, S., Suarez, G., Jungreis, D., Obeid, I., Picone, J.: Automated Identification of Abnormal Adult EEGs. IEEE Signal Processing in Medicine and Biology Symposium (SPMB) 2015 (2015) https://doi.org/10.1109/SPMB.2015.7405423

[26] Diego, L.d., Isabel, S.: Automated Interpretation of Abnormal Adult Electroencephalograms (2017)

[27] Obeid, I., Picone, J.: The Temple University Hospital EEG Data Corpus. Frontiers in Neuroscience 10 (2016) https://doi.org/10.3389/fnins.2016.00196

[28] Rubin, D.I.: Clinical Neurophysiology, 5th ed edn. Contemporary Neurology Ser, (2021)

[29] Kim, M.-j., Youn, Y.C., Paik, J.: Deep learning-based EEG analysis to classify normal, mild cognitive impairment, and dementia: Algorithms and dataset. NeuroImage 272, 120054 (2023) https://doi.org/10.1016/j.neuroimage.2023.120054

[30] Aurlien, H., Gjerde, I.O., Aarseth, J.H., Eldøen, G., Karlsen, B., Skeidsvoll, H., Gilhus, N.E.: EEG background activity described by a large computerized database. Clinical Neurophysiology 115(3), 665–673 (2004) https://doi.org/10. 1016/j.clinph.2003.10.019

[31] Chiang, A.K.I., Rennie, C.J., Robinson, P.A., Albada, S.J., Kerr, C.C.: Age trends and sex differences of alpha rhythms including split alpha peaks. Clinical Neurophysiology 122(8), 1505–1517 (2011) https://doi.org/10.1016/j.clinph.2011.01. 040

[32] Cellier, D., Riddle, J., Petersen, I., Hwang, K.: The development of theta and alpha neural oscillations from ages 3 to 24 years. Developmental Cognitive Neuroscience 50, 100969 (2021) https://doi.org/10.1016/j.dcn.2021.100969

[33] Yang, J., Zhou, K., Li, Y., Liu, Z.: Generalized Out-of-Distribution Detection: A Survey. International Journal of Computer Vision 132(12), 5635–5662 (2024) https://doi.org/10.1007/s11263-024-02117-4

[34] Lee, K., Lee, K., Lee, H., Shin, J.: A simple unified framework for detecting out-ofdistribution samples and adversarial attacks. In: Proc. of NeurIPS, pp. 7167–7177 (2018)

[35] Nalisnick, E.T., Matsukawa, A., Teh, Y.W., Görür, D., Lakshminarayanan, B.: Do deep generative models know what they don't know? In: Proc. of ICLR (2019)

[36] Serrà, J., Álvarez, D., Gómez, V., Slizovskaia, O., Núñez, J.F., Luque, J.: Input

complexity and out-of-distribution detection with likelihood-based generative models. In: Proc. of ICLR (2020)

[37] Schirrmeister, R., Zhou, Y., Ball, T., Zhang, D.: Understanding anomaly detection with deep invertible networks through hierarchies of distributions and features. In: Proc. of NeurIPS (2020)

[38] Kirichenko, P., Izmailov, P., Wilson, A.G.: Why normalizing flows fail to detect out-of-distribution data. In: Proc. of NeurIPS (2020)

[39] Havtorn, J.D., Frellsen, J., Hauberg, S., Maaløe, L.: Hierarchical vaes know what they don't know. In: Proc. of ICML. Proceedings of Machine Learning Research, vol. 139, pp. 4117–4128 (2021)

[40] Kamkari, H., Ross, B.L., Cresswell, J.C., Caterini, A.L., Krishnan, R.G., Loaiza-Ganem, G.: A geometric explanation of the likelihood OOD detection paradox. In: Proc. of ICML (2024)

[41] Tangermann, M., Müller, K.-R., Aertsen, A., Birbaumer, N., Braun, C., Brunner, C., Leeb, R., Mehring, C., Miller, K.J., Mueller-Putz, G., Nolte, G., Pfurtscheller, G., Preissl, H., Schalk, G., Schlögl, A., Vidaurre, C., Waldert, S., Blankertz, B.: Review of the BCI Competition IV. Frontiers in Neuroscience 6 (2012) https: //doi.org/10.3389/fnins.2012.00055

Table A1 OOD detection performance in terms of AUROC across different perturbations and severity levels on CAUEEG. Bold values indicate the best performing metric for a given perturbation and severity. Discriminative-model-based methods (MSP, Energy, ODIN, ASH) perform largely at chance level, whereas generative-model-based methods achieve considerably better OOD detection with increasing performance for higher severity levels.
<table><tr><td>Perturbation</td><td>Severity</td><td>MSP</td><td>Energy</td><td>ODIN</td><td>ASH</td><td>LL</td><td>Typ.</td><td>DoSE</td><td>SITN</td></tr><tr><td rowspan="3">Resampling</td><td>125 Hz</td><td>0.531</td><td>0.517</td><td>0.540</td><td>0.540</td><td>0.286</td><td>0.587</td><td>0.760</td><td>0.956</td></tr><tr><td>150 Hz</td><td>0.497</td><td>0.520</td><td>0.514</td><td>0.590</td><td>0.166</td><td>0.732</td><td>0.886</td><td>0.984</td></tr><tr><td>200 Hz</td><td>0.470</td><td>0.531</td><td>0.495</td><td>0.658</td><td>0.056</td><td>0.906</td><td>0.965</td><td>0.992</td></tr><tr><td rowspan="3">Channel Shuffle</td><td>10%</td><td>0.496</td><td>0.502</td><td>0.505</td><td>0.518</td><td>0.622</td><td>0.552</td><td>0.711</td><td>0.759</td></tr><tr><td>50%</td><td>0.443</td><td>0.490</td><td>0.470</td><td>0.570</td><td>0.841</td><td>0.766</td><td>0.968</td><td>0.956</td></tr><tr><td>100%</td><td>0.408</td><td>0.483</td><td>0.436</td><td>0.583</td><td>0.879</td><td>0.819</td><td>0.982</td><td>0.968</td></tr><tr><td rowspan="3">Re-referencing</td><td>Cz</td><td>0.281</td><td>0.230</td><td>0.305</td><td>0.366</td><td>0.992</td><td>0.991</td><td>1.000</td><td>0.997</td></tr><tr><td>T7, T8</td><td>0.305</td><td>0.361</td><td>0.339</td><td>0.471</td><td>0.982</td><td>0.976</td><td>1.000</td><td>0.997</td></tr><tr><td>bipolar</td><td>0.426</td><td>0.439</td><td>0.454</td><td>0.562</td><td>0.990</td><td>0.989</td><td>1.000</td><td>0.997</td></tr><tr><td rowspan="3">High-pass Filter 4 Hz</td><td>2 Hz</td><td>0.520</td><td>0.640</td><td>0.512</td><td>0.532</td><td>0.455</td><td>0.507</td><td>0.566</td><td>0.606</td></tr><tr><td></td><td>0.507</td><td>0.685</td><td>0.494</td><td>0.578</td><td>0.408</td><td>0.518</td><td>0.594</td><td>0.756</td></tr><tr><td>8 Hz</td><td>0.518</td><td>0.792</td><td>0.494</td><td>0.721</td><td>0.291</td><td>0.589</td><td>0.626</td><td>0.899</td></tr><tr><td rowspan="3">Low-pass Filter</td><td>45 Hz</td><td>0.496</td><td>0.501</td><td>0.496</td><td>0.502</td><td>0.446</td><td>0.505</td><td>0.544</td><td>0.822</td></tr><tr><td>30 Hz</td><td>0.495</td><td>0.510</td><td>0.492</td><td>0.515</td><td>0.186</td><td>0.700</td><td>0.943</td><td>0.987</td></tr><tr><td>15 Hz</td><td>0.557</td><td>0.564</td><td>0.556</td><td>0.563</td><td>0.007</td><td>0.989</td><td>0.998</td><td>0.994</td></tr></table>

## Appendix B Mahalanobis Distance

We also evaluated the Mahalanobis distance [34], an OOD detection approach that falls into the discriminative category but differs from the methods detailed in Section 3.2 by modelling the density of the latent embedding space, rather than deriving a score based on the network's predictions.

Specifically, we first extracted the penultimate-layer feature representations from the pre-trained discriminative model, taking the activations immediately following the global average pooling layer. Using the in-distribution validation set, we estimated the class-conditional mean feature vectors alongside a single, shared covariance matrix tied across all classes. To ensure numerical stability, a small regularisation term $( \epsilon = 1 0 ^ { - 5 } )$ was added to the diagonal of the shared covariance matrix prior to computing its inverse (the shared precision matrix). During evaluation, the Mahalanobis distance was calculated between each test sample's feature vector and each of the class-conditional means using this shared precision matrix. A sample's final score was then defined as its minimum Mahalanobis distance across all available classes. OOD samples are expected to have higher scores indicating that their embeddings are situated further from the known in-distribution classes.

OOD detection results are shown in Table B2. In contrast to other discriminative methods, the Mahalanobis distance yielded superior OOD detection performance under the Resampling, Channel Shuffle, and Re-referencing perturbations. However it exhibited a severe vulnerability to the High-pass Filter perturbation, where it performed worse than random chance.

Table B2 Mahalanobis distance OOD detection performance in terms of AUROC across different perturbations and severity levels. Compared to other discriminative approaches, Mahalanobis distances show better performance for the Resampling, Channel Shuffle, and Re-referencing perturbations, but also a strong failure for the High-pass Filter perturbation, where they consistently underperform even random guessing.
<table><tr><td>Perturbation</td><td>Severity</td><td>TUAB</td><td>CAUEEG</td></tr><tr><td rowspan="3">Resampling</td><td>125 Hz</td><td>0.548</td><td>0.544</td></tr><tr><td>150 Hz</td><td>0.594</td><td>0.586</td></tr><tr><td>200 Hz</td><td>0.654</td><td>0.640</td></tr><tr><td rowspan="3">Channel Shuffle</td><td>10%</td><td>0.583</td><td>0.604</td></tr><tr><td>50%</td><td>0.741</td><td>0.763</td></tr><tr><td>100%</td><td>0.775</td><td>0.799</td></tr><tr><td rowspan="3">Re-referencing</td><td>Cz</td><td>0.851</td><td>0.925</td></tr><tr><td>T7, T8</td><td>0.794</td><td>0.845</td></tr><tr><td>bipolar</td><td>0.786</td><td>0.861</td></tr><tr><td rowspan="3">High-pass Filter 4 Hz</td><td>2 Hz</td><td>0.348</td><td>0.263</td></tr><tr><td></td><td>0.321</td><td>0.252</td></tr><tr><td>8Hz</td><td>0.285</td><td>0.153</td></tr><tr><td rowspan="3">Low-pass Filter</td><td>45 Hz</td><td>0.500</td><td>0.501</td></tr><tr><td>30 Hz</td><td>0.506</td><td>0.505</td></tr><tr><td>15 Hz</td><td>0.541</td><td>0.519</td></tr></table>

## Appendix C Sample Size Dependence

A potential limitation of generative methods could be a larger sample size requirement. To investigate this possibility, we performed a data scaling analysis where the amount of training data was varied while the validation and test data was kept the same. Starting from a subset of just 25 participants and 5 segments per participant, we increased the sample size in two different ways: either increasing the number of participants or the number of segments per participant. These experiments were conducted on the TUAB dataset and subsampling of participants was stratified by the normality label. We used the resampling perturbation at 200 Hz for OOD detection evaluations.

As shown in Figure C1, OOD detection performance generally increased with the amount of training data, where a larger number of participants (solid lines) was more impactful than an equal number of segments from a smaller participant cohort (dashed lines). Generative methods (SITN in particular) performed surprisingly well even with very limited data and dominated discriminative methods across all evaluated sample sizes. The failure mode for log-likelihoods where performance decreased with more training data is discussed in Appendix D.

![](images/891422e4a866a40caf724106a630095d4b25d27a15357c659faa460d907aebdb.jpg)  
Fig. C1 Sample Size Dependence of OOD Detection Performance. Starting from a baseline of just 25 participants and 5 segments per participant (= 125 samples in total), the training data was gradually increased by either increasing the number of participants (solid lines) or segments per participant (dashed lines). The sampling rate perturbation at 200 Hz was used to generate OOD samples. OOD detection performance generally increased with the amount of training data (see Appendix D for a discussion of the likelihood). Even in the smallest data regimes, generative methods dominated discriminative approaches. Note that MSP is occluded by ODIN, which showed almost identical performance across sample sizes.

## Appendix D Limitations of Likelihoods for OOD Detection

Several works in the computer vision literature have identified critical limitations of likelihoods for OOD detection [35-40], which motivated the development of methods like Typicality, DoSE, and SITN. In widely replicated experiments, it was shown that generative models trained on CIFAR-10 consistently assign higher likelihoods to samples from other datasets like MNIST or SVHN [35–38]. Investigating this issue, Serrà et al. further observed a strong correlation between likelihood and sample complexity (measured by PNG compression size), showing that "simpler" samples like constantcolour images were consistently assigned higher likelihoods than more “complex" ones.

We investigated and were able to replicate the same behaviour for EEG data. Panel A in Figure D2 shows the log-likelihood distribution across ID test samples (100 Hz) compared to the distributions for the corresponding OOD samples obtained through the resampling perturbation at different severities. It can be seen that OOD samples were consistently assigned higher likelihoods, with the highest likelihoods assigned to samples perturbed at the highest severity level. As mentioned in Section 3.1 and illustrated in Figure 2, samples perturbed with this perturbation appear increasingly smoothed with higher severities. Furthermore, Panel B in the same figure shows the (ID) EEG segments with the lowest and highest likelihood in the dataset. Consistent with the complexity observations by Serrà et al., the highest likelihoods are assigned to completely flat EEG segments, whereas high-amplitude artefacts resulted in particularly low likelihoods.

A  
![](images/c5758e47867fba7076195012d88a8796bb2bbedaf6650e20a32207cb40d1f632.jpg)

![](images/d37f33bc3ec99a34af32e5a283fe34e03edca8d735df59d7ce9d60dd18320e0e.jpg)  
Time (s)

![](images/c98ee7c08a8264e3c147e8a3f4e0cdced9958608a7165877a4579965459c54c4.jpg)  
Time (s)  
Fig. D2 Likelihood Limitations. A Distribution of log-likelihoods across ID samples (100 Hz) and OOD samples obtained with the resampling perturbation at increasing severities of 125, 150, and 200 Hz. As previously observed in the context of computer vision datasets, OOD samples are consistently assigned higher likelihoods than the in-distribution data. B EEG segments that were assigned the lowest and highest likelihoods, respectively. As in computer vision, where "simpler" images like constant-colour images are assigned high likelihoods, EEG segments with completely flat channels also result in high likelihoods. High-amplitude artefacts result in the opposite extreme and are assigned very small likelihoods.

## Appendix E Training Details

For the generative model, the UNet architecture following Ho et al. [21], as implemented in Dhariwal and Nichol [22] was used with the hyperparameters detailed in Table E3. Following the flow matching objective of Lipman et al. [23], we constructed a linear interpolant between a Gaussian noise sample $Z \sim { \mathcal { N } } ( 0 , I )$ and a data sample $X \sim P _ { d a t a } \colon X _ { t } = ( 1 - t ) Z + t X , \quad t \sim \mathcal { U } [ 0 , 1 ]$ and minimised the mean-squared error between the network output and the target vector field $X - Z$

For the discriminative model, a Temporal Convolutional Network (TCN) [19] was used with the hyperparameters detailed in Table E4. This model was trained using a supervised cross-entropy loss for the respective prediction task.

Both models were optimised with AdamW (learning rate $1 0 ^ { - 4 }$ , weight decay 0.01, $\beta _ { 1 } \ = \ 0 . 9 , \ \beta _ { 2 } \ = \ 0 . 9 9 9 , \ \epsilon \ = \ 1 0 ^ { - 8 } )$ and global gradient norm clipping (max norm 1.0). The generative and discriminative models were trained for 1,000,000 and 50,000 gradient steps, respectively, at a batch size of 128 and with weights restored to the checkpoint of lowest validation loss. It was empirically verified that the validation loss had converged within the maximum number of gradient steps for both models.

Table E3 UNet hyperparameters.
<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>Channels</td><td>64</td></tr><tr><td>Depth</td><td>2</td></tr><tr><td>Channel multipliers</td><td>1, 2, 2, 2</td></tr><tr><td>Heads</td><td>1</td></tr><tr><td>Attention resolution</td><td>8</td></tr><tr><td>Dropout</td><td>0.0</td></tr></table>

Table E4 TCN hyperparameters.
<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>Blocks</td><td>4</td></tr><tr><td>Filters</td><td>64</td></tr><tr><td>Kernel size</td><td>5</td></tr><tr><td>Dilations</td><td>1, 2, 4, 8</td></tr><tr><td>Dropout</td><td>0.0</td></tr></table>

## Appendix F Exploratory Cross-dataset Experiments

We conducted a number of cross-dataset experiments where we evaluated the TUABtrained methods on CAUEEG and conversely the CAUEEG-trained methods on TUAB. We consider this analysis exploratory as it is unclear to what degree samples from one of these datasets should be considered OOD for a model trained on the other dataset. Aside from potential (unquantified) hardware differences, participants in the TUAB dataset could in principle also be assigned a label from the dementia diagnosis task on CAUEEG, such that EEG recordings from one dataset should not necessarily be considered OOD for a model trained on the other one. Rather than treating this setup as a benchmarking task to evaluate OOD detection performance, we are interested to what degree cross-dataset samples result in high OOD scores and if there are differences across participant subgroups.

Table F5 shows AUROC values analogous to those reported for OOD detection performance. Overall, the different methods show moderate discrimination between the two datasets. Scores from DoSE and SITN, the two strongest OOD detection methods according to the results in Section 4.1, do not separate samples from the two datasets when trained on TUAB (AUROC < 0.55), but achieve AUROC values of 0.703 and 0.740 respectively when trained on CAUEEG. This would be consistent with CAUEEG forming a subset of the TUAB data distribution.

Table F5 Cross-dataset Discrimination Across OOD Detection Methods. Models were trained on one dataset and then evaluated using the test split of the training dataset as ID and the test split of the other dataset as OOD data (e.g. TUAB→CAUEEG denotes trained on TUAB and using CAUEEG as OOD). The degree to which OOD scores discriminate between samples from the two datasets is reported in terms of AUROC. Note that ID and OOD should be interpreted with a grain of salt in this setup as it is unclear to what degree cross-dataset samples are OOD.
<table><tr><td>Condition</td><td>MSP</td><td>Energy</td><td>ODIN</td><td>ASH</td><td>LL</td><td>Typ.</td><td>DoSE</td><td>SITN</td></tr><tr><td>TUAB → CAUEEG</td><td>0.594</td><td>0.478</td><td>0.598</td><td>0.517</td><td>0.669</td><td>0.432</td><td>0.462</td><td>0.544</td></tr><tr><td>CAUEEG → TUAB</td><td>0.489</td><td>0.558</td><td>0.477</td><td>0.514</td><td>0.398</td><td>0.643</td><td>0.703</td><td>0.740</td></tr></table>

In the following figures, we compare OOD scores across different subgroups within the target dataset. To do so, we aggregated OOD scores on the participant level by averaging across samples, resulting in an average OOD score per participant.

Furthermore, to facilitate comparisons, we normalised OOD scores by applying minmax scaling to map the raw outputs of each OOD detection method to a standard [0, 1] range. Additionally, scores were inverted where necessary, such that a higher normalised OOD score always indicates “more OOD".

Figure F3 shows normalised OOD scores across the different diagnosis labels on CAUEEG (normal, mild cognitive impairment, dementia) for the TUAB-trained models. Trends across diagnosis groups are inconsistent across methods and scores are strongly overlapping. Figure F4 shows the same scores across age, with the colours corresponding to the same diagnosis groups (normal in green, mci in orange, dementia in blue). No clear trends are discernable, which might be expected given that the age range of the TUAB training data spans the full age range of the CAUEEG dataset. Figure F5 shows normalised scores for the CAUEEG-trained models on TUAB, stratified by the EEG normality label. Although scores are strongly overlapping for all methods, abnormal EEGs tended to be assigned somewhat higher OOD scores by SITN. Finally, Figure F6 shows the same OOD scores across age on TUAB. As for CAUEEG, no clear trend is discernable, although in this case one might have expected younger participants to look more OOD given that only 14 out of 950 participants in the CAUEEG training data (less than 2%) are aged younger than 50.

![](images/b898996c8fa63134fac43444306a85909111464529ea41496bb5ae71700addf3.jpg)  
Fig. F3 Cross-dataset Subgroup Analysis of OOD Scores Across Diagnosis Labels on CAUEEG. The results show normalised participant-level OOD scores for samples from the CAUEEG dataset assigned by methods trained on TUAB. Scores are largely overlapping without a consistent trend across diagnosis groups.

![](images/5c8f7cb3b60674d67946fd9892c333820be59f928818d178998a24feaa0ebe74.jpg)

![](images/c3d3da7f2abe6342d42a1de352f1545735f07c8975206a907061bf36e522852a.jpg)

![](images/552a82f49f514ed58ddbb5e9de0da4b778b924ded8136ff8b09be27b721b69c4.jpg)

![](images/b84009aef50af48bbf3c88b70a8814cf7d582141df6e2def0ba2bf34aa72026c.jpg)  
Fig. F4 Cross-dataset Subgroup Analysis of OOD Scores Across Age on CAUEEG. The results show normalised participant-level OOD scores for samples from the CAUEEG dataset assigned by methods trained on TUAB. No clear trend is observable.

![](images/e35acb897462db478bf197d914af5f67e615dc275c2603a2c54c71209331ff70.jpg)  
Fig. F5 Cross-dataset Subgroup Analysis of OOD Scores Across EEG Normality Labels on TUAB. The results show normalised participant-level OOD scores for samples from the TUAB dataset assigned by methods trained on CAUEEG. Scores are largely overlapping, although SITN assigned somewhat higher OOD scores to abnormal EEG recordings.

![](images/1fe2487699191c3c3f3421e9ff0ffc5bb380ed77cd8f329711ca8f90ccac5f6b.jpg)  
Fig. F6 Cross-dataset Subgroup Analysis of OOD Scores Across Age on TUAB. The results show normalised participant-level OOD scores for samples from the TUAB dataset assigned by methods trained on CAUEEG. No clear trends are discernable.

## Appendix G Cross-participant BCI Experiment

We evaluated the different OOD detection methods in a cross-participant setup on the BNCI 2014-001 Motor Imagery dataset [41]. This dataset contains EEG data for a motor imagery task from nine participants. Two sessions (recorded on different days) are available for each participant, each of which in turn consists of six runs with 12 trials per class. In our setup, we only used trials for the left hand and right hand classes, resulting in 144 trials per session and participant. We used the official traintest split, which assigns the first session of each participant to train and the second to test. Additionally, we used the last run in the training session as validation data. For our cross-participant experiment, we trained the classification model and the OOD detection methods only on the training data of the first participant. We then evaluated the methods on the test data of the training participant as well as on the test data of all other participants.

Figure G7 shows the results with prediction accuracy for the motor imagery task and OOD scores aggregated to the participant level through averaging. For better visualisation, we flipped scores where necessary, such that higher scores on the xaxis always correspond to higher OOD scores. SITN is the only method to clearly separate the (ID) training participant from all (OOD) participants. On six out of eight of the unseen participants prediction accuracy for the motor imagery task dropped considerably below the within-participant test performance. MSP assigned the lowest OOD scores to the two participants where model performance was higher than for the training participant, however, it also assigned the third lowest OOD score to a participant where performance dropped below chance level and which was flagged as highly OOD by all generative methods as well as the Energy score and ASH.

![](images/aed16f8176890b52c153c0fe55c43bd95740813e142c539e5af7dc045c4b3ca2.jpg)  
Fig. G7 Cross-participant Experiments on the BNCI 2014-001 Motor Imagery Dataset. Numbers next to the points indicate participant id. Methods were trained on the training data from participant 1 and evaluated on the test data from all participants. OOD scores were aggregated on the participant-level through averaging and aligned such that higher scores always indicate more OOD. Only SITN clearly separates test data from the training participant (ID) from the other participants (OOD). MSP assigns the lowest OOD scores to two participants where model performance was even higher than for the training participant, but also the 3rd lowest score to the participant where model performance was lowest and which was flagged as highly OOD by most other methods.

## Appendix H Full Downstream Performance Results

The following pages show the full results for the downstream performance evaluations across both tasks and datasets (normality prediction on TUAB (binary classification, chance level 50%); dementia diagnosis on CAUEEG (3-way classification into normal, MCI, dementia, chance level ≈ 33%)) for all perturbations and OOD detection methods. Performance is shown in terms of accuracy across percentile bins of the OOD score, where the x ticks indicate the upper edge of the corresponding bin (i.e. the first bin contains samples from the 0-0.1 percentile, the second one 0.1-1, etc.). The histograms indicate the number of ID and OOD samples in each bin. The arrows in the top left/right of each plot indicate the OOD direction for the corresponding metric. The captions for each figure indicate the task and dataset as well as the perturbation used to generate the OOD data.

![](images/67896a0011bdbe4c0e7e857c57823892f82896870963160feee6f568d688110d.jpg)

![](images/7e93b23e51d62546a9a6679d141e32f2f248887ea8b2470360717ac859fd2ed1.jpg)

![](images/bca3f10f3c107c1c84a9100414b893d9aa8dae4b957979ec5c4ac572c4fa0e43.jpg)

![](images/ee3b3745002e799ef1e9c579ebf3d5669e252c569ead3598a37355649c43bccf.jpg)

![](images/5acff7a47fbd97e63e0153d4024b225d46534e53b7b4e9705b1c4cc4a9d84123.jpg)

![](images/8048513cc96aa8a4eb172020f18859cbbff55e58d27f71db0d3c96499a35bcac.jpg)

![](images/6b1991057c9752f5828782fdde1da0d250a283e88f414c0d09f3be749695757e.jpg)

![](images/27babc368bc02891bc496cb096062e89165d9c8454911550159f819cd9533011.jpg)  
Fig. H8 Normality Prediction on TUAB — Resampling Perturbation (200 Hz)

![](images/a82e04e6d6d8d97937f6b57da07a9b177e73178ef00c2370722359e5eaa4f463.jpg)

![](images/ed9270e633cd4531fcff4591bd04b596f24a5bccd7fd09bb40ef90b059dc5ea8.jpg)

![](images/40a048f9e5417396e60145503e465a74c61c1f53327ac2d2d0e59b1af4a6e212.jpg)

![](images/c88567c3a5c0e4c150d5e333d793124706eb15330fd3b4af4cffcee095376ad2.jpg)

![](images/4c3c39cb8a3979438368506bdcceaa6d33363aef1f359ca0c615e4b838547f9f.jpg)

![](images/44ecb8d61e72bf43a705ecaf8936758b0023ecb478a07df0eee160248799d7f8.jpg)

![](images/ea095af6a33fda14631b5b754e6c9ddb93ad4422761ec5c2ff790ebc60f1b536.jpg)

![](images/ad606e333ad4483347e2b49ad23ddfbda4bcbfaa8f8fd28cf053465098f90ab8.jpg)  
Fig. H9 Normality Prediction on TUAB — Channel Shuffle Perturbation (100%)

![](images/bb8c048aac4ef2f31c041868887302c2e2811d5a15851cc6c731c837017d81bd.jpg)

![](images/7a8c763826cb9b0e7eaed39e63be8c674c6b061e4253d0917729ce7c2bb4f681.jpg)

![](images/68960c8595cd7a50cb3ad3252da39b8908609faa3555be0b9591ddd223891477.jpg)

![](images/73f43c8e1f5d6c7b22d89f48bc0e6025ca3571ac6c33cc1b225a12f8ba5adeb6.jpg)

![](images/9a02b5422418e26a4f97b3a3a32893ed7da07331e9a342870141b17753d95729.jpg)

![](images/f5d3d0cf125a414cf134508bfb317f2c04b30434067c1aadeb20df84bf1f6135.jpg)

![](images/58d8b936e1312f382c3b6e4c9fa1521267436298ecfff6053a91953eeff051e0.jpg)

![](images/df05d1003ea9f4c80323769a78980c55d0c7d074a197aeb6b00e210364224513.jpg)  
Fig. H10 Normality Prediction on TUAB — Re-referencing Perturbation (bipolar) Note: The irregular bin sizes at the upper percentiles of the SITN plot are due to metric saturation, where a large number of tied maximum scores are grouped into the final bin.

![](images/064a5212a30751b5c6ee4c49dc11d2bfa5ed9e09865c3c6dae36c33d8fb98f03.jpg)

![](images/af49da1c3ee5615b13b56b7bed6140140dfde929b1744790fe76f1f87e955c49.jpg)

![](images/4f034b1768550ce0889c773323e8a1b94858f586ef236d3c7fc1cf8d6eefea1e.jpg)

![](images/92407e2af6df1f720ad4c6fbe86f23b985e9c26c0ec87c64d0053fcce8772d34.jpg)

![](images/313a19f622209bd0370f3f3a4e702ba3bd634f6898a026a2f404967a6799b3f7.jpg)

![](images/8c18525fbdb1280740bd08ad6d0dcb5f15facf3f481094bdb3d951074850bca7.jpg)

![](images/2b6cefde7ac8afa874090e1e6ef6fcd65a14b318b8bba1dd77fed8dcf6cc2f10.jpg)

![](images/2743a58343bb8682d60b1329f0deaf86b142b782ac88dcd3932ef979d401a83d.jpg)  
Fig. H11 Normality Prediction on TUAB — High-pass Filter Perturbation (8 Hz)

![](images/a3c2e0348db5f1d42d337cef06d9e241d850100a2f9fd836e95d555edf2a9c9c.jpg)

![](images/09390d47772498f9644c3463debac3fee6fd62ccd600d4a19451bac847d3b1a5.jpg)

![](images/d91381f5a92a399ed3b361da292e30b45118a9824ed63f9a7628ac8a1dc7f37d.jpg)

![](images/e110547b402940c60cc1de8bcd2a456a4819c6a969be2a851dc2450a640a687a.jpg)

![](images/f14f55d9d98ec85b7ea34d2e01b16676a2ee5bebf55be4175d0970e48aff08a7.jpg)

![](images/40f4edb97eca3fa4f106c948d21c5d9b821635ec009f70dc9d076b5b7726d154.jpg)

![](images/ba0d48aae75666dd22362e7b95712cb64db46fc969ad82d6b4b3eab889aa46c4.jpg)

![](images/0b6b78e3c7a11ff7184b52ce01c527f5cc86c3d9bc6d8b0dc34c43495cd5f02b.jpg)  
Fig. H12 Normality Prediction on TUAB — Low-pass Filter Perturbation (15 Hz)

![](images/f62a6e700905cf5656c5e639f750d323f9f01577110a1a299cb8070eee30c492.jpg)

![](images/4ea4ae232085ae5651f5c7fb481fa09df842dc40321a61140599f869cbb9b3ef.jpg)

![](images/aaa67b41e12b645c0880dcd4e68bddfe38217754dddbb924e791a65cca0ca67c.jpg)

![](images/10e8e3ef1326709ecdd6ae07ee668811cf9386d7510a18542ee829390c70d9b4.jpg)

![](images/03b5b5aaead3ef108f06ec126e6e209766e6e8a36074a1b6cbc59ad0540eab67.jpg)

![](images/4e94614023152616f47a7e4833327807e78b4665ed594d3da2b6ed429eacbb93.jpg)

![](images/b57673a9c79aa6caab2abf92688d1cae9261ecef20ddda128f65764d1c6a0c6f.jpg)

![](images/a3bfa29c538143f2dc97261784a47bf9a3d52c4008a2518e76af87b09e8d921a.jpg)  
Fig. H13 Dementia Diagnosis on CAUEEG — Resampling Perturbation (200 Hz)

![](images/6e70329ecb7d6b4c9b95c0e3bce1ece6f932dd573c89e1b7b8e277b348659dd1.jpg)

![](images/86f606fa0f903f40e93a2a74a49eb84a3322d33b4396159e5a245bda17bf6703.jpg)

![](images/309074faebbf97ba760abb2419562903c08c8e2af4aa08c14abec03dedf23e29.jpg)

![](images/af70c19b19a71caf9814d5714f9978227e350dcba5d86b27cc045373ff36e120.jpg)

![](images/c748220d3f208092009771632a989d48deb91048079d0d04e70a61c4f2b98c46.jpg)  
Fig. H14 Dementia Diagnosis on CAUEEG — Channel Shuffle Perturbation (100%)

![](images/8287062296dbcff70db79fbd8ce963aed533227136d885db01564f02f710c787.jpg)

![](images/ffe2a8afc2e0a46d2d1a36aa7f4e0aaaab144a7146817b407aa85c0bbc619957.jpg)

![](images/94bc5f1c00e8e74e9eef73eb6c67cf885713ce473683f12335fdd826802d5f90.jpg)

![](images/78447b5f08ee903a610186a73ad372da98caadbe7927004e7667303454f91c0c.jpg)

![](images/f7e0a85e694b48da840a0ef95aee1f31fb06c7e1a6fdc36aa006c331fb866af7.jpg)

![](images/38bb6285e4c855645ea64a8ae00cd4a5c73cd94a17001d0e7b474c799e4b70c1.jpg)

![](images/e5f05392be67cdad7ccb60d8181e9d0f08382315160bf77712b44348048eb9d7.jpg)

![](images/2792bba155b23c6cfee29bbea14554b8a38f66098938842b92a01e9c848e2816.jpg)  
Fig. H15 Dementia Diagnosis on CAUEEG — Re-referencing Perturbation (bipolar) Note: The irregular bin sizes at the upper percentiles of the SITN plot are due to metric saturation, where a large number of tied maximum scores are grouped into the final bin.

![](images/cfc9388f961dfecb621186955d13345fcb0918902f3dce831a63e94390b3343c.jpg)

![](images/1beb6ea3c0cc1669d9301fc20fcf70c10f3329b27920c174cb8202c8a57aa24c.jpg)

![](images/928d1c6dcd2ad4de0781b21c6f423b0d9ea21ba503f7640598087e4bceec3406.jpg)

![](images/c8118abf7f6287db7d3e768e22c7a34f22c0dc7bfe8d023f04285dcd807a80d2.jpg)

![](images/30c081f5a277e4eaf6de7e47d90b69a6df0b6205a963c1002ade52f5eb59b006.jpg)

![](images/cd87f01af99e2248299d132132cddb7ba6e99324590090047e0fd8ba533fa0dd.jpg)

![](images/131d45b4ddba88221a05fcc56201fb78fb533647f7353c9db15579199cb648d7.jpg)

![](images/bf8432fdf725466e16d61d7645b874d1c3842d25c960728c00200be989be112e.jpg)  
Fig. H16 Dementia Diagnosis on CAUEEG — High-pass Filter Perturbation (8 Hz)

![](images/4494b8c2e2fa19364a3ed5b1cb0411495daba734ccb8b1383521e841246211f9.jpg)

![](images/41b65efc2a9ab59f818a97d55a447787ff870fd1be4dbc38e8003c8a6c0c768b.jpg)

![](images/6a209d6995c816feb6365176aff5ab3f65ce19e41696f8d8be3d38ba10c23a29.jpg)

![](images/1e3f633c87b7608d4d7e6b5926ca941a3e40cb13b1730dd784562f7283bbe13d.jpg)

![](images/e1c932c48c5fa6011ef238383a1eb2b040cb68eed0bc72fa64d04a518382be82.jpg)

![](images/62efa39aff218536ead406ddb271e9720f3cc0e0ed3b07bcdfe48ddc127bf3ce.jpg)

![](images/78c077db58c15fe559f8895992aee1018f7353723052a20176a704968b326b5a.jpg)

![](images/bf45b107a10888a9000aa3413d1a306e2b1289ba0de7f726dc21631dd9d71a34.jpg)  
Fig. H17 Dementia Diagnosis on CAUEEG — Low-pass Filter Perturbation (15 Hz)