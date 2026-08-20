# NanoSleep: A Parameter-Efficient Hybrid Temporal Convolutional Network for Single-Channel Sleep Stage Classification

S M Asif Hossain and Shruti Kshirsagar

School of Computing, Wichita State University, Kansas, USA

## Abstract

Sleep stage classification from single-channel electroencephalography (EEG) is essential for wearable and homebased sleep monitoring. However, many deep learning models achieve high accuracy at the cost of large model sizes, which limits their deployment on resource-constrained devices. In this work, we present NanoSleep, a compact hybrid temporal convolutional network for automatic sleep stage classification. NanoSleep combines a learnable Sinc-convolutional front end, a dual-branch feature extractor that fuses multi-scale temporal and spectral representations, a gated dilated temporal convolutional backbone with channel recalibration, and a conditional random field for sequence-level decoding. We further employ a weighted calibrated focal loss to address class imbalance. We evaluate NanoSleep on the Sleep-EDF and Sleep-EDF-Expanded datasets using subject-wise cross-validation. The proposed model consistently outperforms six representative baseline methods, and an ablation study confirms the contribution of each major component. These results demonstrate that NanoSleep provides an effective balance between accuracy and efficiency, making it well suited for wearable devices, home-based sleep monitoring, and resource-constrained clinical applications.

Keywords: Conditional random field, deep learning, electroencephalogram, parameter efficiency, sleep stage classification, temporal convolutional network, wearable health monitoring.

## 1 Introduction

Sleep is a fundamental physiological process that supports memory consolidation, immune regulation, and the restoration of cognitive and metabolic functions [1]. Poor or insufficient sleep is associated with cardiovascular disease, cognitive decline, and several neurological and psychiatric disorders [2, 3]. Accurate sleep assessment is therefore essential for diagnosing and managing sleep-related disorders. The first step in this process is sleep stage classification, which divides an overnight recording into a sequence of discrete sleep stages. Polysomnography (PSG) is the clinical standard for sleep assessment. It simultaneously records the electroencephalogram (EEG), electrooculogram (EOG), electromyogram (EMG), and other physiological signals throughout the night. According to the American Academy of Sleep Medicine (AASM) guidelines, the recording is divided into consecutive 30-second epochs. Trained sleep technologists assign each epoch to one of five stages: wakefulness (W), non-rapid-eyemovement sleep (N1, N2, and N3), or rapid-eye-movement sleep (REM) [4]. Although this procedure provides reliable clinical assessment, it is labor-intensive, time-consuming, and subject to inter-scorer variability. Agreement between experienced scorers often falls below 90% [5]. Moreover, PSG requires specialized laboratories, expensive equipment, and multiple electrodes that may disturb natural sleep. These limitations have motivated the development of automatic sleep staging systems that operate with fewer physiological signals. Among the available physiological signals, EEG provides the most discriminative information for sleep staging because characteristic waveforms such as sleep spindles, Kcomplexes, and slow waves are directly reflected in brain activity [6]. Single-channel EEG is particularly attractive for wearable and home-based monitoring because it reduces hardware complexity and improves user comfort while preserving most stage-relevant information [7, 8]. As a result, automatic sleep stage classification using single-channel EEG has become an active area of research. Recent deep learning methods have substantially improved automatic sleep staging by learning discriminative representations directly from raw EEG signals. Convolutional neural networks effectively capture local temporal patterns, recurrent neural networks model sequential sleep dynamics, and attention-based architectures further improve contextual modeling. Despite these advances, several important challenges remain.

The first challenge is computational efficiency. Highperforming models such as SleepEEGNet [9] and XSleep-Net [10] contain millions of trainable parameters. Such models are difficult to deploy on wearable devices and resourceconstrained clinical hardware because of their memory, computational, and energy requirements. Although lightweight alternatives have been proposed [11, 12], they often sacrifice classification accuracy to reduce model size.

The second challenge is class imbalance. Sleep datasets contain substantially fewer N1 epochs than the other sleep stages. Consequently, many deep learning models perform poorly on this minority class [13, 14]. The third challenge is sequence modeling. Many methods classify each epoch independently and therefore ignore the physiological transition rules that govern normal sleep progression [15]. As a result, these models may produce isolated predictions that are inconsistent with biological sleep patterns.

To address these challenges, we propose NanoSleep, a compact hybrid network for automatic sleep stage classification from single-channel EEG. NanoSleep combines learned multiscale temporal representations with interpretable spectral features to improve feature learning while maintaining a small model size. The architecture integrates a learnable Sincconvolutional front end, a dual-branch feature extractor, a gated dilated temporal convolutional backbone with channel recalibration, and a conditional random field (CRF) decoder for sequence-level prediction. We further employ a weighted calibrated focal loss to address severe class imbalance during training.

The main contributions of this work are summarized as follows.

1. We propose NanoSleep, a compact hybrid architecture that combines learned multi-scale temporal representations with interpretable spectral features. The proposed model employs a gated dilated temporal convolutional backbone and a CRF decoder to achieve high classification accuracy with only 0.35 M trainable parameters.

2. We develop an efficient training framework that integrates learnable signal preprocessing, imbalance-aware optimization, and sequence-level decoding. An ablation study quantifies the contribution of each component and validates the proposed architecture.

3. We evaluate NanoSleep on the Sleep-EDF and Sleep-EDF-Expanded datasets using subject-wise crossvalidation. The proposed model consistently outperforms representative state-of-the-art methods while requiring substantially fewer parameters.

The remainder of this paper is organized as follows. Section 2 reviews related work. Section 3 presents the proposed NanoSleep framework. Section 4 describes the experimental setup. Section 5 reports and discusses the experimental results. Section 6 outlines the study limitations and future research directions. Finally, Section 7 concludes the paper.

## 2 Related Works

In this section, we discuss the related work in the sleep stage classification domain.

## 2.1 Conventional Feature-Based Classifiers

Early automatic sleep staging methods relied on handcrafted EEG features combined with conventional machine learning classifiers. Researchers extracted time-domain statistics, modulation spectrogram, spectral band powers, entropy measures, and nonlinear complexity features to characterize sleep stages [16–18]. Memar and Faradji combined spectral features with a random forest classifier and reported promising performance [19]. Hassan and Bhuiyan employed ensemble empirical mode decomposition with boosting classifiers for singlechannel EEG sleep staging [20]. Jiang et al. incorporated temporal information through a hidden Markov model [21], while Zhou et al. proposed a stacked ensemble with a classbalancing strategy to improve N1 recognition [13]. Although these methods are computationally efficient and interpretable, their performance depends heavily on manually designed features and often generalizes poorly across different datasets and recording conditions. These limitations motivated the development of deep learning methods that learn discriminative representations directly from EEG signals.

## 2.2 Deep Learning on Single-Channel EEG

Deep learning has become the dominant approach for automatic sleep stage classification. Tsinalis et al. introduced stacked sparse autoencoders for learning EEG representations from time-frequency images [22]. Sors et al. later demonstrated that convolutional neural networks can classify sleep stages directly from raw single-channel EEG without handcrafted features [23]. Supratak et al. proposed DeepSleepNet, which combines multi-scale convolutional feature extraction with bidirectional long short-term memory networks for temporal modeling [24]. Sandhu et al. [25] demonstrated the effectiveness of single-channel EEG for accurate, interpretable, and practical automated sleep staging. Subsequent studies further improved performance using deeper architectures and larger training datasets [26, 27]. More recently, Yang et al. introduced BIOT, a transformer-based backbone for large-scale biosignal representation learning that improves feature transfer across heterogeneous physiological datasets [28]. These studies demonstrate the effectiveness of learned representations but generally require large models with high computational and memory costs.

## 2.3 Sequence and Attention Architectures

Sleep stages exhibit strong temporal dependencies, motivating sequence-based learning methods. Phan et al. proposed SeqSleepNet, a hierarchical recurrent network for sequenceto-sequence sleep staging [29]. Mousavi et al. introduced SleepEEGNet, an attention-based encoder-decoder architecture that improves classification performance at the cost of a large parameter count [9]. Eldele et al. developed AttnSleep by combining convolutional feature extraction with multi-head attention [30]. Phan et al. later proposed XSleep-Net and SleepTransformer, which integrate multiple signal representations and transformer-based sequence modeling to achieve state-of-the-art performance [10,31]. Graph-based approaches, including GraphSleepNet and SalientSleepNet, have also been proposed to model spatial-temporal relationships and salient EEG patterns [32, 33].

Recent studies have further improved model generalization through transfer learning and domain adaptation. Hossain and Kshirsagar proposed a demographic-aware transfer learning framework for cross-cohort sleep staging [34]. Eldele et al. introduced ADAST, an attentive domain adaptation framework based on iterative self-training [35]. Tallal et al. proposed STDA-Net for unsupervised domain adaptation across sleep datasets [36]. These studies demonstrate the importance of robust feature learning for clinical deployment. However, most sequence and attention models achieve higher accuracy by substantially increasing model complexity.

## 2.4 Data Augmentation Strategies

Data augmentation has been widely adopted across various domains to enhance the generalization capability and robustness of models trained on speech [37], [38], image [39], [40], and physiological signal data [14, 41]. Data augmentation has been used to increase training diversity and reduce the severe imbalance among sleep stages. Sun et al. introduced an oversampling-based pretraining procedure that applies temporal shifts and additive white noise to minority-stage EEG epochs [14]. Fan et al. systematically evaluated repeated sampling, morphological transformations, signal segmentation and recombination, cross-dataset transfer, and generative adversarial network-based synthesis on the MASS and Sleep-EDF datasets [42]. Lee et al. proposed spectral band blending, in which selected frequency bands are exchanged between EEG signals to generate new samples while preserving stage-related spectral information [43]. Khalili and Mohammadzadeh Asl also incorporated augmented sequence samples into a temporal convolutional framework to improve the training of raw-EEG classifiers [44].

Generative and noise-based augmentation methods have received increasing attention [45], [46]. Ling et al. developed an improved deep convolutional generative adversarial network that generates continuous-wavelet time-frequency maps for minority sleep stages [47]. Huang et al. applied Gaussian noise augmentation to underrepresented polysomnography segments to improve class balance and N1 recognition [48]. Rommel et al. compared 13 EEG augmentation transformations and showed that their effectiveness depends on the task, dataset, and training regime [49]. These studies demonstrate that data augmentation can improve robustness and minority-stage recognition. However, the selected transformations must preserve physiologically meaningful EEG morphology and temporal structure.

## 2.5 Efficient and Compact Models

Several studies have focused on reducing the computational cost of deep sleep staging models. TinySleepNet substantially reduces the parameter count of DeepSleepNet while maintaining competitive performance [11]. DeepSleepNet-Lite further simplifies the architecture and provides calibrated uncertainty estimates [12]. Perslev et al. proposed U-Sleep, a fully convolutional segmentation framework designed for heterogeneous sleep cohorts [50]. Kuo and Chen investigated hybrid recurrent architectures for large-scale clinical sleep staging [15]. Temporal convolutional networks have also emerged as an attractive alternative because dilated convolutions provide large receptive fields with relatively few parameters while supporting parallel computation [51,52]. Previous studies have addressed class imbalance using oversampling, cost-sensitive learning, and focal loss [13, 14, 53]. Despite these advances, existing lightweight models generally sacrifice classification accuracy to achieve computational efficiency.

Existing studies have significantly advanced automatic sleep stage classification. However, several challenges remain. High-performing models often rely on large convolutional, recurrent, or transformer architectures that are unsuitable for wearable and resource-constrained devices. Lightweight models reduce computational cost but usually compromise classification accuracy. In addition, most methods rely primarily on either learned representations or handcrafted features rather than exploiting their complementary strengths. Finally, the recognition of minority sleep stages, particularly N1, remains challenging because of severe class imbalance. To address these limitations, we propose NanoSleep, a compact hybrid architecture for single-channel EEG sleep stage classification.

## 3 Methodology

In this section, we describe the NanoSleep framework in detail. Fig. 1 shows the complete framework: we preprocess the raw single-channel EEG, extract time-domain and spectral features in parallel, fuse them, process the fusion with a gated dilated TCN backbone, and decode the sleep stage predictions with a CRF. Here, we present the proposed model architecture and the training objective.

## 3.1 Proposed Model

The main contribution of this work is NanoSleep, a compact hybrid model with approximately 0.35 million parameters. We combine learned and interpretable representations, an efficient temporal backbone, and a learnable sequence-level decoder into a single architecture. We design the model to satisfy three key requirements. First, it captures the multi-scale morphology of sleep EEG. Second, it exploits the strong sequential regularities of overnight sleep. Third, it remains compact enough to run on wearable and point-of-care devices. The following subsections describe each component. Each component is now described in turn, and the complete data flow is depicted in Fig. 1.

## 3.1.1 Multi-Scale Time-Domain Branch

The first branch of the extractor learns time-domain representations directly from the preprocessed signal using a multi-scale convolution module. Sleep EEG contains patterns with different temporal durations. Some events, such as Kcomplexes, are brief. Others, such as slow oscillations, last several seconds. To capture this diversity, we apply three convolutional kernels in parallel. The small kernel captures short transient events. The medium kernel captures mid-length rhythms, such as sleep spindles. The large kernel captures slow waves. We concatenate the outputs of all three kernels. This operation produces a feature representation that encodes multiple temporal scales and reflects the multi-scale nature of sleep EEG waveforms.

![](images/b5a7c735a59c11148b8549bfed843b3e0beea21a9506deadf2b02e756b48a157.jpg)  
Figure 1: Overall architecture of the proposed NanoSleep framework for single-channel EEG sleep stage classification.

![](images/601d341f1ea55139156221a4211d393bdb2c0f7a6d9a3bc2cbe16659fefe2ccc.jpg)  
Figure 2: Power spectral density of the single-channel EEG for the five sleep stages, with the standard delta, theta, alpha, and beta frequency bands indicated.

## 3.1.2 Spectral Feature Branch

The second branch computes interpretable descriptors that complement the learned representations. We first compute the relative power spectral density in the standard delta, theta, alpha, and beta frequency bands. Relative power reduces sensitivity to inter-subject amplitude variation. Fig. 2 shows the spectral power distribution for each sleep stage. The figure confirms that these bands contain stage-discriminative information. We also compute multiscale dispersion entropy [54] and the Hjorth parameters of activity, mobility, and complexity [55]. These features describe the signal complexity and temporal dynamics across multiple scales. Next, we pass all descriptors through a small multilayer perceptron, called the spectral multilayer perceptron. This network transforms them into a compact feature vector. We intentionally retain this handcrafted spectral branch alongside the learned branch. The spectral features provide stable and physiologically meaningful information that generalizes well across subjects. The ablation study in Section 5.3 shows that removing this branch causes one of the largest performance drops.

## 3.1.3 Feature Fusion

The fusion stage combines the learned time-domain features with the transformed spectral and complexity features. We concatenate the outputs of both branches to form a unified feature representation. This design allows the model to leverage the flexibility of learned features and the robustness of handcrafted descriptors. It also preserves the physiological interpretability of the spectral features while improving the overall representation.

## 3.1.4 Gated Dilated Temporal Convolutional Backbone

The fused features are processed by a gated dilated temporal convolutional network, which serves as the sequence backbone of NanoSleep. We use dilated convolutions to capture long-range temporal context across many epochs. This design increases the receptive field without adding many parameters or requiring sequential computation [52, 56]. Each layer includes a gated linear unit [57]. The gate learns which temporal information to retain and which to suppress. We then apply a squeeze-and-excitation block [58] to recalibrate the feature channels. This block emphasizes informative channels and reduces the influence of less useful ones. Together, dilation, gating, and channel recalibration produce a compact and expressive backbone.

## 3.1.5 Sequence-Level Decoding

Sleep stages follow well-established physiological transition patterns. We refine the backbone predictions with a learnable linear-chain conditional random field (CRF) [59]. The CRF learns a $5 \times 5$ transition matrix that captures common stage transitions. For example, a direct transition from deep sleep to REM sleep is uncommon. For an input sequence and a candidate label sequence $y ,$ the CRF assigns the following score:

$$
s ( y ) = \sum _ { t = 1 } ^ { T } U _ { t } ( y _ { t } ) + \sum _ { t = 1 } ^ { T - 1 } A ( y _ { t } , y _ { t + 1 } ) ,\tag{1}
$$

where $U _ { t } ( y _ { t } )$ is the unary score that the backbone assigns to stage $y _ { t }$ at epoch $t , A ( y _ { t } , y _ { t + 1 } )$ is the learned transition score between consecutive stages, and T is the number of epochs in the sequence. During inference, we use the Viterbi algorithm [60] to find the label sequence with the highest score. This decoding step removes implausible isolated predictions. For example, it suppresses a single wake epoch that appears within a long period of deep sleep. As a result, the final predictions become more consistent with the natural progression of sleep stages

## 3.2 Training Objective

We design the training objective to address class imbalance and improve predictions at sleep stage transitions. The objective combines a weighted calibrated focal loss with the negative log-likelihood of the conditional random field (CRF). The focal loss [53] reduces the contribution of easy examples and focuses training on difficult epochs, such as those belonging to the N1 stage. For an epoch with ground-truth stage c and predicted probability $p _ { c } .$ , the per-epoch focal loss is

$$
\mathcal { L } _ { \mathrm { f o c a l } } = - w _ { c } ( 1 - p _ { c } ) ^ { \gamma _ { c } } \log ( p _ { c } ) ,\tag{2}
$$

where $w _ { c }$ is the class weight and $\gamma _ { c }$ is the focusing parameter of class c. We compute the class weights using a sub-linear function of the inverse class frequency. This strategy increases the importance of rare stages without neglecting the majority classes. We also use a class-specific focusing schedule. We assign larger focusing parameters to the difficult N1 and N3 stages and smaller values to the majority stages.

We calibrate the predicted probabilities through probability smoothing. This step discourages overconfident predictions and produces confidence scores that better reflect the model’s true uncertainty [61]. We further apply a boundaryaware penalty. We double the focal loss whenever a stage transition occurs between consecutive epochs. This penalty encourages the model to make more accurate predictions at sleep stage boundaries. The final training objective is the sum of the weighted calibrated focal loss and the CRF negative loglikelihood. We compute the CRF loss from the sequence score in (1).

## 4 Experimental Setup

In this section, we describe the experimental setup, including the dataset, preprocessing steps, data augmentation strategy, experimental configuration, and evaluation metrics used for sleep stage classification.

## 4.1 Datasets and Training Protocol

We evaluate the proposed model and benchmark methods on two publicly available sleep EEG datasets. These datasets include recordings from both healthy individuals and subjects with sleep disorders. This diversity allows us to assess both classification accuracy and generalization. The Sleep-EDF dataset and the Sleep-EDF-Expanded dataset are available through the PhysioNet repository [62–64]. The first dataset is Sleep-EDF. We use the Fpz-Cz EEG channel from overnight recordings sampled at 100 Hz. The second dataset is Sleep-EDF-Expanded. We again use the Fpz-Cz channel. This dataset contains many more subjects and recordings, making it a stronger benchmark for evaluating robustness and crosssubject generalization. We segment all recordings into 30,s epochs according to the AASM guidelines. We assign each epoch to one of five sleep stages: W, N1, N2, N3, or REM. For datasets that use the older scoring standard, we merge the N3 and N4 stages into a single N3 class. Table 1 summarizes the number of subjects, EEG channels, and sampling rates for all datasets. Fig. 3 shows representative EEG epochs for each sleep stage, and Fig. 4 shows a complete overnight hypnogram, both drawn from the Sleep-EDF dataset.

Table 1: Composition of the two datasets used in this study.
<table><tr><td>Property</td><td>Sleep-EDF</td><td>Sleep-EDF-Exp.</td></tr><tr><td>Subjects</td><td>20</td><td> $\overline { { 7 8 } }$ </td></tr><tr><td>EEG channel</td><td>Fpz-Cz</td><td>Fpz-Cz</td></tr><tr><td>Sampling rate</td><td>100 Hz</td><td>100 Hz</td></tr></table>

![](images/ff3a387604536f51ab87083058320475a3fd5c3872f032e0529d965e90e0eb81.jpg)  
Figure 3: Representative 30 s single-channel EEG epochs from the Fpz-Cz channel of the Sleep-EDF dataset for the five sleep stages. The N2 epoch exhibits sleep spindles and Kcomplexes, the N3 epoch is dominated by high-amplitude slow waves, and the REM epoch presents a low-amplitude mixedfrequency pattern.

To evaluate generalization, we perform subject-wise crossvalidation. We ensure that recordings from the same subject never appear in both the training and test sets within a fold. We use 20-fold cross-validation for Sleep-EDF and 10-fold crossvalidation for Sleep-EDF-Expanded. We select the number of folds according to the size of each dataset. Within every training fold, we reserve a subset of subjects for validation. We use this validation set for model selection and hyperparameter tuning.

![](images/a335d62ba51aeb3d6af01f33297c194fdee52271e731d9b2db65fd480a6b0d72.jpg)  
Figure 4: Full-night hypnogram of a representative subject from the Sleep-EDF dataset, illustrating the cyclic alternation of sleep stages across the night and the comparatively small proportion of the N1 stage.

## 4.2 Preprocessing Pipeline

We preprocess the EEG signals to reduce noise while preserving stage-specific patterns. The preprocessing pipeline consists of three stages: learnable spectral filtering, robust normalization, and wavelet denoising.

First, we replace the fixed band-pass filter used in conventional pipelines with a learnable Sinc-convolutional layer [65]. Traditional band-pass filters use manually selected cutoff frequencies that remain fixed during training. As a result, they may remove useful frequency information. In contrast, the Sinc-convolutional layer learns the cutoff frequencies directly from the data. Each filter is defined by a low cutoff frequency $f _ { 1 }$ and a high cutoff frequency $f _ { 2 }$ . The filter response is constructed from the difference of two sinc functions and a filter g is defined in the time domain as

$$
g [ n , f _ { 1 } , f _ { 2 } ] = 2 f _ { 2 } \mathrm { s i n c } ( 2 \pi f _ { 2 } n ) - 2 f _ { 1 } \mathrm { s i n c } ( 2 \pi f _ { 1 } n ) ,\tag{3}
$$

where n is the sample index and sinc $: ( x ) = \sin ( x ) / x$ . We learn only the two cutoff frequencies for each filter. This design keeps the number of trainable parameters low while allowing the network to discover the most informative frequency bands for sleep staging.

Next, we normalize each EEG epoch using the median and the median absolute deviation (MAD) instead of the mean and standard deviation. EEG recordings often contain highamplitude artifacts that distort conventional normalization. Robust statistics reduce the influence of these outliers. For an epoch x, we compute the normalized signal as

$$
\tilde { x } = \frac { x - \mathrm { m e d i a n } ( x ) } { \mathrm { M A D } ( x ) + \epsilon } ,\tag{4}
$$

where MAD(x) is the median absolute deviation and ϵ is a small constant that prevents division by zero. This normalization reduces inter-subject amplitude variation and improves the consistency of the input signals. Finally, we apply wavelet denoising using the Daubechies-4 wavelet with soft thresholding [66]. We first decompose the EEG signal into wavelet coefficients. We then apply soft thresholding to the detail coefficients and reconstruct the signal. This process removes lowamplitude noise while preserving clinically important waveforms, such as sleep spindles and K-complexes, which are essential for identifying the N2 stage.

![](images/efa588afefbec2e019f1c560decef035fda98458b62727a094197b21c74bf534.jpg)  
Figure 5: Effect of the data augmentation operations on a representative N2 epoch. Panel (a) shows the original epoch, panel (b) shows the epoch after brain-wave stretching, and panel (c) shows the epoch after brain-noise injection with a slow voltage drift.

## 4.3 Data Augmentation

We apply three data augmentation techniques during training to improve generalization and reduce the effect of class imbalance. These augmentations increase the diversity of the training data, especially for minority sleep stages. First, we apply brain-wave stretching. We slightly stretch or compress the time axis of each EEG epoch. This operation simulates the natural variation of brain rhythms across subjects and sleep cycles. Second, we apply brain-noise injection. We add realistic broadband noise and slow baseline drifts to the EEG signal. This augmentation mimics background brain activity and recording artifacts commonly observed in real-world sleep studies. Third, we use a contextual augmentation based on feature-space CutMix [67]. We exchange segments of the latent representations between sequences that belong to the same sleep stage. This operation encourages the model to learn more robust stage representations and improves discrimination at stage boundaries. Fig. 5 illustrates the effects of brain-wave stretching and brain-noise injection on a representative N2 epoch.

## 4.4 Experimental Configuration and Reproducibility

This subsection describes the implementation and training settings used in our experiments. These details allow the proposed method to be reproduced.

We use EEG signals sampled at 100 Hz. Each 30,s epoch therefore contains 3000 samples. The Sinc-convolutional front end contains 32 learnable band-pass filters, each with a length of 65 samples. We initialize the filters to cover the 0.5–40 Hz frequency range, which includes the main sleep rhythms. We perform wavelet denoising using the Daubechies-4 wavelet with four decomposition levels and a universal soft threshold.

The time-domain branch uses three parallel onedimensional convolutional layers with kernel sizes of $^ { 7 , }$ 25, and 51 samples. Each branch produces 32 feature maps to capture short-, medium-, and long-duration temporal patterns. The spectral multilayer perceptron contains two hidden layers with 64 and 32 neurons, respectively. Both layers use rectified linear unit (ReLU) activations. The temporal backbone consists of six residual blocks with dilation rates of 1, 2, 4, 8, 16, and 32. Each block uses a kernel size of 7 and 64 feature channels. Each residual block includes a gated linear unit, batch normalization [68], and dropout with a rate of 0.3 [69]. The squeeze-and-excitation module uses a channel reduction ratio of 8.

We train the model using the Adam optimizer [70]. The initial learning rate is $1 \times 1 0 ^ { - 3 }$ with cosine-annealing scheduling and a weight decay of $1 \times 1 0 ^ { - 4 }$ . We use a batch size of 32 sequences, where each sequence contains 20 consecutive epochs. We train the model for up to 120 epochs and apply early stopping if the validation performance does not improve for 15 consecutive epochs. For the focal loss in (2), we set the focusing parameter to 2.5 for the N1 and N3 stages and 1.0 for the W, N2, and REM stages. We set the probability-smoothing factor to 0.1. We apply each data augmentation method described in Section 4.3 online with a probability of 0.5.

The complete model contains approximately 0.35 million trainable parameters. We implement all experiments in Py-Torch and run them on a single NVIDIA RTX 3090 GPU with 24 GB of memory. We fix the random seed to 42 for every cross-validation fold to ensure reproducible data splits, model initialization, and training.

## 4.5 Evaluation Metrics

We evaluate NanoSleep using several performance metrics that measure overall accuracy, per-stage performance, and agreement with expert annotations. These metrics are widely used in automatic sleep staging studies [13, 24]. Let TP, TN, FP, and FN denote the numbers of true positive, true negative, false positive, and false negative epochs, respectively.

We first report the overall accuracy (ACC), which measures the proportion of correctly classified epochs:

$$
\mathrm { A C C } = { \frac { \mathrm { T P } + \mathrm { T N } } { \mathrm { T P } + \mathrm { T N } + \mathrm { F P } + \mathrm { F N } } } .\tag{5}
$$

Accuracy provides an overall measure of performance. However, it can be biased toward majority sleep stages in imbalanced datasets.

We therefore report precision and recall for each sleep stage. Precision measures how many predicted epochs are correct, while recall measures how many true epochs are correctly identified:

$$
\mathrm { P r e c i s i o n } = { \frac { \mathrm { T P } } { \mathrm { T P } + \mathrm { F P } } } , \qquad \mathrm { R e c a l l } = { \frac { \mathrm { T P } } { \mathrm { T P } + \mathrm { F N } } } .\tag{6}
$$

We also compute the F1-score, which balances precision and recall:

$$
\mathrm { F 1 } = \frac { \mathrm { 2 } \times \mathrm { P r e c i s i o n } \times \mathrm { R e c a l l } } { \mathrm { P r e c i s i o n } + \mathrm { R e c a l l } } .\tag{7}
$$

To account for class imbalance, we report the macro F1- score (MF1). This metric computes the average F1-score across all sleep stages:

$$
\mathrm { M F 1 } = \frac { 1 } { C } \sum _ { i = 1 } ^ { C } \mathrm { F } 1 i ,\tag{8}
$$

where $C = 5$ is the number of sleep stages and F1i is the F1- score for stage i. Since each stage contributes equally, MF1 reflects the model’s performance on both majority and minority classes.

Finally, we report Cohen’s kappa coefficient (κ), which measures the agreement between the model predictions and expert annotations after correcting for chance agreement [71]:

$$
\kappa = { \frac { p _ { o } - p _ { e } } { 1 - p _ { e } } } ,\tag{9}
$$

where $p _ { o }$ is the observed agreement and $p _ { e }$ is the expected agreement by chance. A κ value above 0.80 indicates outstanding agreement, while values between 0.61 and 0.80 indicate substantial agreement. We also present confusion matrices to visualize the classification performance and error patterns for each sleep stage on every dataset.

## 5 Results and Discussion

In this section, we evaluate the performance of NanoSleep on two benchmark sleep EEG datasets. We compare the proposed model with representative state-of-the-art methods under the same subject-wise cross-validation protocol. We then analyze the confusion matrices to examine stage-level errors and perform an ablation study to quantify the contribution of each component. Finally, we discuss the main findings and their implications for compact and accurate sleep stage classification.

## 5.1 Performance on the Sleep-EDF Dataset

We first evaluate NanoSleep on the Sleep-EDF dataset. This experiment assesses whether the proposed architecture improves classification performance while maintaining a compact model size. We compare NanoSleep with six representative baseline models using 20-fold subject-wise crossvalidation. We report overall accuracy (ACC), macro F1-score (MF1), Cohen’s kappa (κ), per-stage F1-score, and model size. Table 2 summarizes the results. NanoSleep achieves the best overall performance, with an accuracy of 86.5%, a macro F1- score of 82.1%, and a kappa coefficient of 0.81. It outperforms TinySleepNet by 1.1% in accuracy, 1.6% in macro F1- score, and 0.01 in kappa. NanoSleep also uses only 0.35 M parameters, compared with 1.3 M for TinySleepNet. Thus, it requires only 27% of the parameters while achieving higher performance.

NanoSleep also outperforms larger and more complex models. It improves accuracy by 2.2% over SleepEEG-Net, 4.6% over DeepSleepNet, 2.6% over XSleepNet, and 2.1% over AttnSleep. It also improves the macro F1-score by 2.4%, 5.5%, 3.4%, and 4.0%, respectively. These results show that NanoSleep consistently outperforms recurrent, attention-based, lightweight, and high-capacity architectures. At the stage level, NanoSleep achieves the highest F1-scores for W, N2, and REM, with values of 94.4%, 89.8%, and 87.0%, respectively. It improves W recognition by 4.3% over TinySleepNet and 5.2% over SleepEEGNet. It also achieves the best performance on N2, the dominant sleep stage, with a 1.3% improvement over TinySleepNet. For REM, NanoSleep exceeds the strongest competing model by 1.6%. AttnSleep achieves the highest F1-score for N3, but NanoSleep remains within 1.6% while providing better overall accuracy, macro F1-score, kappa, and a substantially smaller model. N1 is the only stage where NanoSleep is not the best. Its F1-score of 50.8% remains close to the highest reported value of 52.9%.

We next analyze the confusion matrix to identify the remaining classification errors. Figure 6(a) shows that W, N2, N3, and REM are classified with high reliability. Their recall values reach 94.0%, 90.5%, 89.0%, and 88.0%, respectively. In contrast, N1 remains the most challenging stage, with a recall of 47.5%. Most N1 errors occur in the neighboring stages. Specifically, 26.5% of N1 epochs are classified as N2, 14.0% as REM, and 11.5% as W. This behavior agrees with sleep physiology because N1 is a short transitional stage that shares characteristics with adjacent stages and often produces disagreement among expert scorers [5]. We also observe that 10.7% of N3 epochs are classified as N2. This confusion reflects the gradual transition between these two stages.

Overall, these results demonstrate that NanoSleep achieves an excellent accuracy-to-parameter trade-off. The hybrid feature extractor combines complementary temporal and spectral information, while the temporal backbone captures long-range sleep dependencies efficiently. Together, these components enable NanoSleep to outperform both lightweight and highcapacity models while maintaining a compact architecture.

## 5.2 Performance on the Sleep-EDF-Expanded Dataset

We next evaluate NanoSleep on the larger Sleep-EDF-Expanded dataset to assess its robustness on a more heterogeneous population. We perform 10-fold subject-wise crossvalidation and compare NanoSleep with the same baseline models.Table 3 summarizes the results. NanoSleep again achieves the best overall performance, with an accuracy of 84.3%, a macro F1-score of 79.8%, and a kappa coefficient of 0.78. It improves upon TinySleepNet by 1.2% in accuracy, 1.6% in macro F1-score, and 0.01 in kappa while using only 0.35 M parameters, compared with 1.3 M. These results show that the performance gains observed on the original Sleep-EDF dataset remain consistent on a larger and more diverse cohort.

NanoSleep also outperforms the remaining baseline models. It improves accuracy by 4.3% over SleepEEGNet, 6.5% over DeepSleepNet, 4.0% over XSleepNet, 3.0% over AttnSleep, and 4.0% over DeepSleepNet-Lite. It also improves the macro F1-score by 6.2%, 8.0%, 3.4%, 4.6%, and 4.6%, respectively. These results demonstrate that NanoSleep generalizes well across different model families while maintaining a substantially smaller parameter count. At the stage level, NanoSleep achieves the highest F1-scores for W, N1, N2, and REM, with values of 94.7%, 51.8%, 87.0%, and 84.2%, respectively. It consistently improves W recognition across both Sleep-EDF datasets, indicating effective separation between wakefulness and sleep. NanoSleep also achieves the best N1 performance, although this stage remains the most challenging. The weighted focal loss and contextual feature learning contribute to this improvement. For N2, NanoSleep improves the F1-score by 1.7% over TinySleepNet and 2.0% over AttnSleep. For REM, it improves by 3.9% over TinySleepNet and 10.0% over AttnSleep. AttnSleep achieves the highest F1-score for N3, but NanoSleep remains within 0.7% while achieving higher overall accuracy, macro F1-score, kappa, and better performance on the remaining stages. These consistent improvements indicate that NanoSleep generalizes well across subjects and does not overfit the smaller Sleep-EDF dataset.

We next analyze the confusion matrix to examine the remaining classification errors. Figure 6(b) shows that W, N2, and REM are recognized with high reliability. In contrast, N1 remains the most difficult stage, with a recall of 48.5%. Most N1 errors occur in neighboring stages, which is expected because N1 represents a transitional sleep stage. We also observe greater confusion between N3 and N2 than on the original Sleep-EDF dataset. Specifically, 16.0% of N3 epochs are classified as N2. This pattern is consistent with the broader age range and greater inter-subject variability of the Sleep-EDF-Expanded cohort, where reduced slow-wave activity makes the boundary between N2 and N3 less distinct.

Overall, these results demonstrate that NanoSleep maintains high accuracy and strong generalization on a larger and more heterogeneous dataset. The model consistently outperforms both lightweight and high-capacity baselines while using substantially fewer parameters. These findings support NanoSleep as an efficient and robust solution for automatic sleep stage classification.

## 5.3 Ablation Study

We perform an ablation study to quantify the contribution of each component in NanoSleep. We evaluate all variants on the Sleep-EDF dataset using the same 20-fold subject-wise crossvalidation protocol. We remove or replace one component at a time while keeping the remaining architecture unchanged. Table 4 summarizes the results. Removing the spectral branch reduces the macro F1-score from 82.1% to 77.9%. This result shows that the handcrafted spectral descriptors provide complementary information that the learned features do not fully capture. Replacing the multi-scale convolution with a singlebranch convolution produces the largest performance drop, reducing the macro F1-score to 76.7%. This finding highlights the importance of learning temporal patterns at multiple scales.

Table 2: Performance comparison on the Sleep-EDF dataset under 20-fold subject-wise cross-validation.
<table><tr><td>Model</td><td> $\overline { { \mathbf { A C C } \left( \% \right) } }$ </td><td> $\overline { { \mathbf { M F 1 } \left( \% \right) } }$ </td><td>κ</td><td>W</td><td>N1</td><td>N2</td><td>N3</td><td>REM</td><td>Param.</td></tr><tr><td>SleepEEGNet</td><td> $\overline { { 8 4 . 3 \pm 0 . 4 } }$ </td><td> $7 9 . 7 \pm 0 . 8$ </td><td> $\overline { { 0 . 7 9 \pm 0 . 0 2 } }$ </td><td> $\overline { { 8 9 . 2 \pm 0 . 5 } }$ </td><td> $\overline { { 5 2 . 2 \pm 0 . 8 } }$ </td><td> $8 6 . 8 \pm 0 . 3$ </td><td> $8 5 . 1 \pm 0 . 3$ </td><td> $\overline { { 8 5 . 0 \pm 0 . 3 } }$ </td><td>2.6M</td></tr><tr><td>DeepSleepNet</td><td> $8 1 . 9 \pm 0 . 3$ </td><td> $7 6 . 6 \pm 0 . 3$ </td><td> $0 . 7 6 \pm 0 . 0 4$ </td><td> $8 6 . 7 \pm 0 . 4$ </td><td> $4 5 . 5 \pm 0 . 3$ </td><td> $8 5 . 1 \pm 0 . 3$ </td><td> $8 3 . 3 \pm 0 . 4$ </td><td> $8 2 . 6 \pm 0 . 2$ </td><td>24.7M</td></tr><tr><td>XSleepNet</td><td> $8 3 . 9 \pm 0 . 3$ </td><td> $7 8 . 7 \pm 0 . 2$ </td><td> $0 . 7 7 \pm 0 . 0 4$ </td><td> $8 1 . 6 \pm 0 . 4$ </td><td> ${ \bf 5 2 . 9 \pm 0 . 3 }$ </td><td> $8 8 . 1 \pm 0 . 2 $ </td><td> $8 5 . 3 \pm 0 . 3$ </td><td> $8 5 . 4 \pm 0 . 3$ </td><td>5.8M</td></tr><tr><td>AttnSieep</td><td> $8 4 . 4 \pm 0 . 3$ </td><td> $7 8 . 1 \pm 0 . 3$ </td><td> $0 . 7 9 \pm 0 . 0 3$ </td><td> $8 9 . 7 \pm 0 . 3$ </td><td> $4 2 . 6 \pm 0 . 3$ </td><td> $8 8 . 8 \pm 0 . 3$ </td><td> ${ \bf 9 0 . 2 \pm 0 . 3 }$ </td><td> $7 9 . 0 \pm 0 . 4$ </td><td>0.52M</td></tr><tr><td>DeepSleepNet-Lite</td><td> $8 4 . 0 \pm 0 . 4$ </td><td> $7 8 . 0 \pm 0 . 3$ </td><td> $0 . 7 8 \pm 0 . 0 4$ </td><td> $8 7 . 1 \pm 0 . 4$ </td><td> $4 4 . 4 \pm 0 . 3$ </td><td> $8 7 . 9 \pm 0 . 2$ </td><td> $8 8 . 2 \pm 0 . 4$ </td><td> $8 2 . 4 \pm 0 . 3$ </td><td>0.6M</td></tr><tr><td>TinySleepNet</td><td> $8 5 . 4 \pm 0 . 3$ </td><td> $8 0 . 5 \pm 0 . 3$ </td><td> $0 . 8 0 \pm 0 . 0 4$ </td><td> $9 0 . 1 \pm 0 . 3$ </td><td> $5 1 . 4 \pm 0 . 3$ </td><td> $8 8 . 5 \pm 0 . 2 $ </td><td> $8 8 . 3 \pm 0 . 3$ </td><td> $8 4 . 3 \pm 0 . 4$ </td><td>1.3M</td></tr><tr><td>NanoSleep</td><td> ${ \bf 8 6 . 5 \pm 0 . 3 }$ </td><td> ${ \bf 8 2 . 1 \pm 0 . 3 }$ </td><td> ${ \bf 0 . 8 1 \pm 0 . 0 4 }$ </td><td> ${ \bf 9 4 . 4 \pm 0 . 4 }$ </td><td> $5 0 . 8 \pm 0 . 3$ </td><td> ${ \bf 8 9 . 8 \pm 0 . 4 }$ </td><td> $8 8 . 6 \pm 0 . 4$ </td><td> ${ \bf 8 7 . 0 \pm 0 . 3 }$ </td><td>0.35 M</td></tr></table>

Note: Results are reported as mean ± standard deviation over 20 folds. Bold indicates the best result in each column. Statistical significance was evaluated using a two-sided paired Wilcoxon signed-rank test, with a p-value < 0.05 considered statistically significant.

Table 3: Performance comparison on the Sleep-EDF-Expanded dataset under 10-fold subject-wise cross-validation.
<table><tr><td>Model</td><td>ACC (%)</td><td>MF1 (%)</td><td>κ</td><td>W</td><td>N1</td><td>N2</td><td>N3</td><td>REM</td><td>Param.</td></tr><tr><td>SleepEEGNet</td><td> $\overline { { 8 0 . 0 \pm 0 . 5 } }$ </td><td> $7 3 . 6 \pm 0 . 5$ </td><td> $\overline { { 0 . 7 3 \pm 0 . 0 3 } }$ </td><td> $9 1 . 7 \pm 0 . 5$ </td><td> $\overline { { 4 4 . 1 \pm 1 . 7 } }$ </td><td> $\overline { { 8 2 . 5 \pm 0 . 6 } }$ </td><td> $\overline { { 7 3 . 5 \pm 1 . 0 } }$ </td><td> $7 6 . 1 \pm 1 . 0$ </td><td>2.6M</td></tr><tr><td>DeepSleepNet</td><td> $7 7 . 8 \pm 0 . 5$ </td><td> $7 1 . 8 \pm 0 . 6$ </td><td> $0 . 7 0 \pm 0 . 0 3$ </td><td> $9 0 . 9 \pm 0 . 7$ </td><td> $4 5 . 0 \pm 2 . 2$ </td><td> $7 9 . 2 \pm 0 . 6$ </td><td> $7 2 . 7 \pm 0 . 9$ </td><td> $7 1 . 1 \pm 0 . 7$ </td><td>24.7M</td></tr><tr><td>XSleepNet</td><td> $8 0 . 3 \pm 0 . 5$ </td><td> $7 6 . 4 \pm 0 . 5$ </td><td> $0 . 7 2 \pm 0 . 0 2$ </td><td> $8 5 . 2 \pm 0 . 5$ </td><td> $4 9 . 4 \pm 1 . 4$ </td><td> $8 6 . 0 \pm 0 . 6$ </td><td> $7 9 . 8 \pm 0 . 8$ </td><td> $8 1 . 7 \pm 0 . 9$ </td><td>5.8M</td></tr><tr><td>AttnSleep</td><td> $8 1 . 3 \pm 0 . 5$ </td><td> $7 5 . 2 \pm 0 . 7$ </td><td> $0 . 7 4 \pm 0 . 0 3$ </td><td> $9 2 . 0 \pm 0 . 7$ </td><td> $4 2 . 9 \pm 1 . 0$ </td><td> $8 5 . 0 \pm 0 . 7$ </td><td> ${ \bf 8 2 . 1 \pm 0 . 6 }$ </td><td> $7 4 . 2 \pm 0 . 9$ </td><td>0.52M</td></tr><tr><td>DeepSleepNet-Lite</td><td> $8 0 . 3 \pm 0 . 5$ </td><td> $7 5 . 2 \pm 0 . 7$ </td><td> $0 . 7 3 \pm 0 . 0 2$ </td><td> $9 1 . 5 \pm 0 . 8$ </td><td> $4 6 . 4 \pm 1 . 0$ </td><td> $8 2 . 9 \pm 0 . 7$ </td><td> $7 9 . 2 \pm 0 . 8$ </td><td> $7 6 . 4 \pm 0 . 5$ </td><td>0.6M</td></tr><tr><td>TinySleepNet</td><td> $8 3 . 1 \pm 0 . 4$ </td><td> $7 8 . 2 \pm 0 . 5$ </td><td> $0 . 7 7 \pm 0 . 0 2$ </td><td> $9 2 . 8 \pm 0 . 7$ </td><td> $5 1 . 5 \pm 1 . 9$ </td><td> $8 5 . 3 \pm 0 . 7$ </td><td> $8 1 . 1 \pm 0 . 8$ </td><td> $8 0 . 3 \pm 0 . 9$ </td><td>1.3M</td></tr><tr><td>NanoSleep</td><td> ${ \bf 8 4 . 3 \pm 0 . 4 }$ </td><td> ${ \bf 7 9 . 8 \pm 0 . 5 }$ </td><td> ${ \bf 0 . 7 8 \pm 0 . 0 3 }$ </td><td> ${ \bf 9 4 . 7 \pm 0 . 6 }$ </td><td> ${ \bf 5 1 . 8 \pm 1 . 5 }$ </td><td> ${ \bf 8 7 . 0 \pm 0 . 6 }$ </td><td> $8 1 . 4 \pm 0 . 8$ </td><td> ${ \bf 8 4 . 2 \pm 1 . 0 }$ </td><td>0.35 M</td></tr></table>

Note: Results are reported as mean ± standard deviation over 10 folds. Bold indicates the best result in each column. Statistical significance was evaluated using a two-sided paired Wilcoxon signed-rank test, with a p-value < 0.05 considered statistically significant.

![](images/7878f113163d921425dfbf378866cb3ec76da77334b85341a280ff01a9f9e7e8.jpg)

![](images/ba5943ae04de8e93f3d29be4ac29246383566f7eb05a351a435304eedb26fca2.jpg)  
(a) Sleep-EDF dataset.  
(b) Sleep-EDF-Expanded dataset.  
Figure 6: Confusion matrices of NanoSleep on the two evaluation datasets. Rows denote the expert annotation and columns denote the model prediction.

Removing the conditional random field (CRF) decreases the macro F1-score to 79.9%. This result demonstrates the benefit of sequence-level decoding for suppressing isolated and physiologically implausible predictions. Removing the squeezeand-excitation block reduces the macro F1-score to 81.3%. Although the reduction is smaller, it confirms the usefulness of channel recalibration. Replacing the focal loss with the standard cross-entropy loss reduces the macro F1-score to 79.5%. The largest degradation occurs for the minority N1 stage, indicating that the imbalance-aware objective is essential for recognizing underrepresented classes. Removing data augmentation decreases the macro F1-score to 79.8%, showing that the proposed augmentation strategy improves model generalization. Finally, replacing the gated dilated temporal convolutional backbone with a bidirectional long short-term memory network reduces the macro F1-score to 79.8%. This result shows that the proposed backbone is both more compact and more effective than a conventional recurrent architecture.

Overall, the ablation study shows that every component contributes to the final performance. The largest macro F1 reductions are observed when removing the multi-scale convolution, the spectral branch, and the focal-loss objective. These findings validate the design of NanoSleep and demonstrate that learned temporal representations and handcrafted spectral features complement each other to achieve robust and accurate sleep stage classification.

## 6 Limitations and Future Work

This study has several limitations that motivate future research. First, we evaluate NanoSleep using classification metrics only. Although the model contains only 0.35 M parameters, we do not report computational efficiency metrics such as floatingpoint operations (FLOPs), inference latency, or memory footprint. Future work will measure these metrics on representative edge devices to validate the suitability of NanoSleep for real-time deployment. Second, the performance improvements over the strongest baseline models are consistent but relatively modest. Although we report fold-wise standard deviations and paired statistical significance tests, we do not report confidence intervals or subject-level uncertainty estimates. Future work will include these analyses to provide a more comprehensive assessment of the observed improvements. Third, we conduct the ablation study only on the Sleep-EDF dataset. Extending this analysis to the Sleep-EDF-Expanded dataset will provide stronger evidence that each component generalizes across different cohorts and recording conditions. Fourth, N1 remains the most challenging sleep stage. Future work will investigate transfer learning from larger cohorts [34] and domain adaptation techniques [35, 36] to improve the recognition of this minority stage. Finally, we evaluate NanoSleep on publicly available datasets that primarily include healthy individuals and subjects with common sleep disorders. Future work will validate the proposed model on larger and more diverse clinical populations to further assess its robustness and clinical applicability.

## 7 Conclusion

This paper presented NanoSleep, a compact hybrid network for single-channel EEG sleep stage classification. NanoSleep combines learned multi-scale temporal features with interpretable spectral features and models long-range temporal dependencies using a gated dilated temporal convolutional network and a conditional random field. Experiments on the

Sleep-EDF and Sleep-EDF-Expanded datasets demonstrated that NanoSleep consistently outperformed representative baseline models while using only 0.35 M parameters. The ablation study further confirmed the contribution of each major component. These results show that NanoSleep provides an effective balance between accuracy and efficiency, making it suitable for wearable devices and resource-constrained clinical applications. Future work will evaluate computational efficiency on embedded platforms and improve the recognition of the N1 sleep stage.

## Acknowledgment

The authors would like to thank the providers of the Sleep-EDF and Sleep-EDF-Expanded datasets for making their data publicly available.

## References

[1] Faith S Luyster, Patrick J Strollo Jr, Phyllis C Zee, and James K Walsh. Sleep: a health imperative. Sleep, 35(6):727–734, 2012.

[2] Katharina Wulff, Silvia Gatti, Joseph G Wettstein, and Russell G Foster. Sleep and circadian rhythm disruption in psychiatric and neurodegenerative disease. Nature Reviews Neuroscience, 11(8):589–599, 2010.

[3] Suzanne M Bertisch, Benjamin D Pollock, Murray A Mittleman, Daniel J Buysse, Lydia A Bazzano, Daniel J Gottlieb, and Susan Redline. Insomnia with objective short sleep duration and risk of incident cardiovascular disease and all-cause mortality: Sleep heart health study. Sleep, 41(6):zsy047, 2018.

[4] Richard B Berry, Rita Brooks, Charlene Gamaldo, Susan M Harding, Robin M Lloyd, Stuart F Quan, Matthew T Troester, and Bradley V Vaughn. Aasm scoring manual updates for 2017 (version 2.4). Journal of clinical sleep medicine, 13(5):665–666, 2017.

[5] Richard S Rosenberg and Steven Van Hout. The american academy of sleep medicine inter-scorer reliability program: sleep stage scoring. Journal of clinical sleep medicine, 9(1):81–87, 2013.

[6] Huy Phan and Kaare Mikkelsen. Automatic sleep staging of eeg signals: recent development, challenges, and future directions. Physiological Measurement, 43(4):04TR01, 2022.

[7] Syed Anas Imtiaz. A systematic review of sensing tech nologies for wearable sleep staging. Sensors, 21(5):1562, 2021.

[8] Kaare B Mikkelsen, David Bove Villadsen, Marit Otto,´ and Preben Kidmose. Automatic sleep staging using eareeg. Biomedical engineering online, 16(1):111, 2017.

Table 4: Ablation study of the proposed NanoSleep model on the Sleep-EDF dataset under 20-fold subject-wise crossvalidation.
<table><tr><td>Model Variant</td><td> $\overline { { \mathbf { A C C } \left( \% \right) } }$ </td><td> $\overline { { \mathbf { M F 1 } \left( \% \right) } }$ </td><td>κ</td><td>W</td><td>N1</td><td>N2</td><td>N3</td><td>REM</td></tr><tr><td>W/o spectral branch</td><td> $8 3 . 5 \pm 0 . 4$ </td><td> $7 7 . 9 \pm 0 . 5$ </td><td> $\overline { { 0 . 7 6 \pm 0 . 0 3 } }$ </td><td> $9 2 . 6 \pm 0 . 7$ </td><td> $\overline { { 4 3 . 5 \pm 1 . 5 } }$ </td><td> $\overline { { 8 6 . 2 \pm 0 . 4 } }$ </td><td> $\overline { { 8 4 . 5 \pm 0 . 4 } }$ </td><td> $\overline { { 8 3 . 0 \pm 0 . 7 } }$ </td></tr><tr><td>W/o multi-scale convolution (single branch)</td><td> $8 2 . 0 \pm 0 . 4$ </td><td> $7 6 . 7 \pm 0 . 6$ </td><td> $0 . 7 4 \pm 0 . 0 3$ </td><td> $8 8 . 8 \pm 0 . 5$ </td><td> $4 2 . 2 \pm 1 . 6$ </td><td> $8 5 . 1 \pm 0 . 5$ </td><td> $8 3 . 2 \pm 0 . 7$ </td><td> $8 4 . 2 \pm 0 . 5$ </td></tr><tr><td>W/o conditional random field (softmax decoding)</td><td> $8 4 . 8 \pm 0 . 5$ </td><td> $7 9 . 9 \pm 0 . 6$ </td><td> $0 . 7 9 \pm 0 . 0 3$ </td><td> $9 2 . 5 \pm 0 . 9$ </td><td> $4 7 . 3 \pm 1 . 3$ </td><td> $8 8 . 5 \pm 0 . 4$ </td><td> $8 6 . 0 \pm 0 . 6$ </td><td> $8 5 . 5 \pm 0 . 5$ </td></tr><tr><td>W/o squeeze-and-excitation block</td><td> $8 5 . 8 \pm 0 . 5$ </td><td> $8 1 . 3 \pm 0 . 5$ </td><td> $0 . 8 0 \pm 0 . 0 4$ </td><td> $9 4 . 1 \pm 0 . 6 $ </td><td> $4 9 . 2 \pm { 1 . 5 }$ </td><td> $8 9 . 2 \pm 0 . 4$ </td><td> $8 7 . 8 \pm 0 . 8$ </td><td> $8 6 . 3 \pm 0 . 7$ </td></tr><tr><td>W/o focal loss (standard cross-entropy)</td><td> $8 5 . 2 \pm 0 . 4$ </td><td> $7 9 . 5 \pm 0 . 6$ </td><td> $0 . 7 9 \pm 0 . 0 4$ </td><td> ${ \bf 9 4 . 8 \pm 0 . 4 }$ </td><td> $4 4 . 1 \pm 1 . 7$ </td><td> $8 9 . 0 \pm 0 . 5$ </td><td> $8 5 . 2 \pm 0 . 7$ </td><td> $8 4 . 5 \pm 0 . 7$ </td></tr><tr><td>W/o data augmentation</td><td> $8 4 . 5 \pm 0 . 4$ </td><td> $7 9 . 8 \pm 0 . 6$ </td><td> $0 . 7 8 \pm 0 . 0 2$ </td><td> $9 2 . 8 \pm 0 . 5$ </td><td> $4 7 . 0 \pm 1 . 7$ </td><td> $8 8 . 2 \pm 0 . 6 $ </td><td> $8 6 . 1 \pm 0 . 5$ </td><td> $8 5 . 0 \pm 0 . 6$ </td></tr><tr><td>Backbone replaced with bidirectional LSTM</td><td> $8 4 . 2 \pm 0 . 4$ </td><td> $7 9 . 8 \pm 0 . 5$ </td><td> $0 . 7 8 \pm 0 . 0 3$ </td><td> $9 2 . 5 \pm 0 . 6$ </td><td> $4 6 . 5 \pm 1 . 6$ </td><td> $8 7 . 9 \pm 0 . 4$ </td><td> $8 6 . 8 \pm 0 . 8$ </td><td> $8 5 . 5 \pm 0 . 4$ </td></tr><tr><td>Full NanoSleep model</td><td> ${ \bf 8 6 . 5 \pm 0 . 3 }$ </td><td> ${ \bf 8 2 . 1 \pm 0 . 3 }$ </td><td> ${ \bf 0 . 8 1 \pm 0 . 0 4 }$ </td><td>94.4 ± 0.4</td><td> ${ \bf 5 0 . 8 \pm 0 . 3 }$ </td><td>89.8 ± 0.4</td><td> ${ \bf 8 8 . 6 \pm 0 . 4 }$ </td><td> ${ \bf 8 7 . 0 \pm 0 . 3 }$ </td></tr><tr><td colspan="9">Note: Results are reported as mean ± standard deviation over 20 folds. Bold indicates the best result in each column. Statistical significance was evaluated using a two-sided paired Wilcoxon signed-rank test, with a p-value &lt; 0.05 considered statistically significant.</td></tr></table>

[9] Sajad Mousavi, Fatemeh Afghah, and U Rajendra Acharya. Sleepeegnet: Automated sleep stage scoring with sequence to sequence deep learning approach. PloS one, 14(5):e0216456, 2019.

[10] Huy Phan, Oliver Y. Chen, Minh C. Tran, Philipp Koch, Alfred Mertins, and Maarten De Vos. XSleepNet: Multi-View Sequential Model for Automatic Sleep Staging . IEEE Transactions on Pattern Analysis & Machine Intelligence, 44(09):5903–5915, September 2022.

[11] Akara Supratak and Yike Guo. Tinysleepnet: An efficient deep learning model for sleep stage scoring based on raw single-channel eeg. In 2020 42nd Annual International Conference of the IEEE Engineering in Medicine & Biology Society (EMBC), pages 641–644. IEEE, 2020.

[12] Luigi Fiorillo, Paolo Favaro, and Francesca Dalia Faraci. Deepsleepnet-lite: A simplified automatic sleep stage scoring model with uncertainty estimates. IEEE transactions on neural systems and rehabilitation engineering, 29:2076–2085, 2021.

[13] Jinjin Zhou, Guangsheng Wang, Junbiao Liu, Duanpo Wu, Weifeng Xu, Zimeng Wang, Jing Ye, Ming Xia, Ying Hu, and Yuanyuan Tian. Automatic sleep stage classification with single channel eeg signal based on two-layer stacked ensemble model. IEEE Access, 8:57283–57297, 2020.

[14] Chenglu Sun, Jiahao Fan, Chen Chen, Wei Li, and Wei Chen. A two-stage neural network for sleep stage classification based on feature learning, sequence learning, and data augmentation. IEEE Access, 7:109386–109397, 2019.

[15] Chih-En Kuo and Guan-Ting Chen. Automatic sleep staging based on a hybrid stacked lstm neural network: verification using large-scale dataset. IEEE access, 8:111837–111849, 2020.

[16] Ahnaf Rashik Hassan and Mohammed Imamul Hassan Bhuiyan. A decision support system for automatic sleep staging from eeg signals using tunable q-factor wavelet transform and spectral features. Journal of neuroscience methods, 271:107–118, 2016.

[17] Tarek Lajnef, Sahbi Chaibi, Perrine Ruby, Pierre-Emmanuel Aguera, Jean-Baptiste Eichenlaub, Mounir Samet, Abdennaceur Kachouri, and Karim Jerbi. Learning machines and sleeping brains: automatic sleep stage classification using decision-tree multi-class support vector machines. Journal ofneuroscience methods, 250:94– 105, 2015.

[18] Unaza Tallal, Rupesh Agrawal, and Shruti Kshirsagar. Modulation-based feature extraction for robust sleep stage classification across apnea-based cohorts. Biosensors, 16(1):56, 2026.

[19] Pejman Memar and Farhad Faradji. A novel multi-class eeg-based sleep stage classification system. IEEE Transactions on Neural Systems and Rehabilitation Engineering, 26:84–95, 2018.

[20] Ahnaf Rashik Hassan and Mohammed Imamul Hassan Bhuiyan. Automated identification of sleep states from eeg signals by means of ensemble empirical mode decomposition and random under sampling boosting. Computer methods and programs in biomedicine, 140:201– 210, 2017.

[21] Dihong JIANG, Ya nan LU, Yu MA, and Yuanyuan WANG. Robust sleep stage classification with singlechannel eeg signals using multimodal decomposition and hmm-based refinement. Expert Systems with Applications, 121:188–203, 2019.

[22] Orestis Tsinalis, Paul M Matthews, and Yike Guo. Automatic sleep stage scoring using time-frequency analysis and stacked sparse autoencoders. Annals of biomedical engineering, 44(5):1587–1597, 2016.

[23] Arnaud Sors, Stephane Bonnet, S´ ebastien Mirek, Lau-´ rent Vercueil, and Jean-Franc¸ois Payen. A convolutional neural network for sleep stage scoring from raw singlechannel eeg. Biomedical Signal Processing and Control, 42:107–114, 2018.

[24] Akara Supratak, Hao Dong, Chao Wu, and Yike Guo. Deepsleepnet: A model for automatic sleep stage scoring based on raw single-channel eeg. IEEE transactions on neural systems and rehabilitation engineering, 25(11):1998–2008, 2017.

[25] Gurinder Kaur Sandhu, Jude Koenig, Shruti Kshirsagar, and Ankita Shukla. Exploring explainable ai methods for single channel eeg sleep staging across ahi stratified obstructive sleep apnea cohorts. 2026.

[26] Siddharth Biswal, Haoqi Sun, Balaji Goparaju, M Brandon Westover, Jimeng Sun, and Matt T Bianchi. Expertlevel sleep scoring with deep neural networks. Journal of the American Medical Informatics Association, 25(12):1643–1650, 2018.

[27] Jens B Stephansen, Alexander N Olesen, Mads Olsen, Aditya Ambati, Eileen B Leary, Hyatt E Moore, Oscar Carrillo, Ling Lin, Fang Han, Han Yan, et al. Neural network analysis of sleep stages enables efficient diagnosis of narcolepsy. Nature communications, 9(1):5229, 2018.

[28] Chaoqi Yang, M Brandon Westover, and Jimeng Sun. Biot: Biosignal transformer for cross-data learning in the wild. Advances in Neural Information Processing Systems, 36:78240–78260, 2023.

[29] Huy Phan, Fernando Andreotti, Navin Cooray, Oliver Y Chen, and Maarten De Vos. Seqsleepnet: end-to-´ end hierarchical recurrent neural network for sequenceto-sequence automatic sleep staging. IEEE Transactions on Neural Systems and Rehabilitation Engineering, 27(3):400–410, 2019.

[30] Emadeldeen Eldele, Zhenghua Chen, Chengyu Liu, Min Wu, Chee-Keong Kwoh, Xiaoli Li, and Cuntai Guan. An attention-based deep learning approach for sleep stage classification with single-channel eeg. IEEE Transactions on Neural Systems and Rehabilitation Engineering, 29:809–818, 2021.

[31] Huy Phan, Kaare Mikkelsen, Oliver Y Chen, Philipp´ Koch, Alfred Mertins, and Maarten De Vos. Sleeptransformer: Automatic sleep staging with interpretability and uncertainty quantification. IEEE Transactions on Biomedical Engineering, 69(8):2456–2467, 2022.

[32] Ziyu Jia, Youfang Lin, Jing Wang, Ronghao Zhou, Xiaojun Ning, Yuanlai He, and Yaoshuai Zhao. Graphsleepnet: Adaptive spatial-temporal graph convolutional networks for sleep stage classification. In Proceedings of the Twenty-Ninth International Joint Conference on Artificial Intelligence, pages 1324–1330, 2020.

[33] Ziyu Jia, Youfang Lin, Jing Wang, Xuehui Wang, Peiyi Xie, and Yingbin Zhang. Salientsleepnet: Multimodal salient wave detection network for sleep staging. In Proceedings of the Thirtieth International Joint Conference on Artificial Intelligence, pages 2614–2620, 2021.

[34] S M Asif Hossain and Shruti Kshirsagar. Demographicaware transfer learning for sleep stage classification in clinical polysomnography. arXiv preprint arXiv:2605.02245, 2026.

[35] Emadeldeen Eldele, Mohamed Ragab, Zhenghua Chen, Min Wu, Chee-Keong Kwoh, Xiaoli Li, and Cuntai Guan. Adast: Attentive cross-domain eeg-based sleep staging framework with iterative self-training. IEEE Transactions on Emerging Topics in Computational Intelligence, 7(1):210–221, 2023.

[36] Unaza Tallal, Shruti Kshirsagar, and Ankita Shukla. Stda-net: Spectrogram-based domain adaptation for cross-dataset sleep stage classification. arXiv preprint arXiv:2605.06736, 2026.

[37] Shruti Kshirsagar and Tiago H Falk. Cross-language speech emotion recognition using bag-of-word representations, domain adaptation, and data augmentation. Sensors, 22(17):6445, 2022.

[38] Shruti Kshirsagar, Anurag Pendyala, and Tiago H Falk. Task-specific speech enhancement and data augmentation for improved multimodal emotion recognition under noisy conditions. Frontiers in Computer Science, 5:1039261, 2023.

[39] Asmae Mouradi and Shruti Kshirsagar. Robust building damage detection in cross-disaster settings using domain adaptation. arXiv preprint arXiv:2603.14694, 2026.

[40] Bharath Chandra Reddy Parupati, Shruti Kshirsagar, Rajiv Bagai, and Atri Dutta. Towards robust building damage detection: Leveraging augmentation and domain adaptation. In 2025 IEEE Green Technologies Conference (GreenTech), pages 163–167. IEEE, 2025.

[41] Thu Mains and Shruti Kshirsagar. A machine learning approach for integrating phonocardiogram and electrocardiogram data for heart sound detection. 12 2024.

[42] Jiahao Fan, Chenglu Sun, Chen Chen, Xinyu Jiang, Xiangyu Liu, Xian Zhao, Long Meng, Chenyun Dai, and Wei Chen. Eeg data augmentation: towards class imbalance problem in sleep staging tasks. Journal of Neural Engineering, 17(5):056017, 2020.

[43] Choel-Hui Lee, Hyun-Ji Kim, Jae-Wook Heo, Hakseung Kim, and Dong-Joo Kim. Improving sleep stage classification performance by single-channel eeg data augmentation via spectral band blending. In 2021 9th International Winter Conference on Brain-Computer Interface (BCI), pages 1–5. IEEE, 2021.

[44] Ebrahim Khalili and Babak Mohammadzadeh Asl. Automatic sleep stage classification using temporal convolutional neural network and new data augmentation technique from raw single-channel eeg. Computer Methods and Programs in Biomedicine, 204:106063, 2021.

[45] Shruti Rajendra Kshirsagar. Affective human-machine interfaces: towards multi-lingual, environment-robust emotion detection from speech. PhD thesis, Universite´ du Quebec, Institut national de la recherche scientifique,´ 2022.

[46] Shruti Rajendra Kshirsagar and Tiago Henrik Falk. Quality-aware bag of modulation spectrum features for robust speech emotion recognition. IEEE Transactions on Affective Computing, 13(4):1892–1905, 2022.

[47] Huang Ling, Yao Luyuan, Li Xinxin, and Dong Bingliang. Staging study of single-channel sleep eeg signals based on data augmentation. Frontiers in Public Health, 10:1038742, 2022.

[48] Xinyu Huang, Kimiaki Shirahama, Muhammad Tausif Irshad, Muhammad Adeel Nisar, Artur Piet, and Marcin Grzegorzek. Sleep stage classification in children using self-attention and gaussian noise data augmentation. Sensors, 23(7):3446, 2023.

[49] Cedric Rommel, Joseph Paillard, Thomas Moreau, and´ Alexandre Gramfort. Data augmentation for learning predictive models on eeg: a systematic comparison. Journal ofNeural Engineering, 19(6):066020, 2022.

[50] Mathias Perslev, Sune Darkner, Lykke Kempfner, Miki Nikolic, Poul Jørgen Jennum, and Christian Igel. Usleep: resilient high-frequency sleep staging. NPJ digital medicine, 4(1):72, 2021.

[51] Colin Lea, Michael D Flynn, Rene Vidal, Austin Reiter, and Gregory D Hager. Temporal convolutional networks for action segmentation and detection. In 2017 IEEE conference on computer vision and pattern recognition (CVPR), pages 1003–1012. IEEE, 2017.

[52] Shaojie Bai, J Zico Kolter, and Vladlen Koltun. An empirical evaluation of generic convolutional and recurrent networks for sequence modeling. arXiv preprint arXiv:1803.01271, 2018.

[53] Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, and Piotr Dollar. Focal loss for dense object detection.´ In Proceedings of the IEEE international conference on computer vision, pages 2980–2988, 2017.

[54] Mostafa Rostaghi and Hamed Azami. Dispersion entropy: A measure for time-series analysis. IEEE Signal Processing Letters, 23(5):610–614, 2016.

[55] Bo Hjorth. Eeg analysis based on time domain properties. Electroencephalography and clinical neurophysiology, 29(3):306–310, 1970.

[56] Fisher Yu and Vladlen Koltun. Multi-scale context aggregation by dilated convolutions. arXiv preprint arXiv:1511.07122, 2015.

[57] Yann N Dauphin, Angela Fan, Michael Auli, and David Grangier. Language modeling with gated convolutional networks. In International conference on machine learning, pages 933–941. PMLR, 2017.

[58] Jie Hu, Li Shen, and Gang Sun. Squeeze-and-excitation networks. In Proceedings of the IEEE conference on

computer vision and pattern recognition, pages 7132– 7141, 2018.

[59] John Lafferty, Andrew McCallum, and Fernando CN Pereira. Conditional random fields: Probabilistic models for segmenting and labeling sequence data. In Proceedings of the eighteenth international conference on machine learning, pages 282–289. Morgan Kaufmann Publishers Inc., 2001.

[60] Andrew J Viterbi. Error bounds for convolutional codes and an asymptotically optimum decoding algorithm. IEEE Transactions on Information Theory, 13(2):260– 269, 1967.

[61] Chuan Guo, Geoff Pleiss, Yu Sun, and Kilian Q Weinberger. On calibration of modern neural networks. In International conference on machine learning, pages 1321–1330. PMLR, 2017.

[62] Bob Kemp. Sleep-EDF Database Expanded, 2018.

[63] Tom Pollard, Benjamin E Moody, Li-wei H Lehman, Brian J Gow, Chrystinne Fernandes, Chen Xie, Alistair Johnson, Roger G Mark, and Thomas Heldt. Physionet as a global platform for biomedical research. Nature Health, 1:792–795, 2026.

[64] Bob Kemp, Aeilko H Zwinderman, Bert Tuk, Hilbert AC Kamphuisen, and Josefien JL Oberye. Analysis of a sleep-dependent neuronal feedback loop: the slowwave microcontinuity of the eeg. IEEE Transactions on Biomedical Engineering, 47(9):1185–1194, 2000.

[65] Mirco Ravanelli and Yoshua Bengio. Speaker recognition from raw waveform with sincnet. In 2018 IEEE spoken language technology workshop (SLT), pages 1021– 1028. IEEE, 2018.

[66] David L Donoho and Iain M Johnstone. Ideal spatial adaptation by wavelet shrinkage. Biometrika, 81(3):425– 455, 1994.

[67] Sangdoo Yun, Dongyoon Han, Seong Joon Oh, Sanghyuk Chun, Junsuk Choe, and Youngjoon Yoo. Cutmix: Regularization strategy to train strong classifiers with localizable features. In Proceedings of the IEEE/CVF international conference on computer vision, pages 6023–6032, 2019.

[68] Sergey Ioffe and Christian Szegedy. Batch normalization: Accelerating deep network training by reducing internal covariate shift. In International conference on machine learning, pages 448–456. PMLR, 2015.

[69] Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever, and Ruslan Salakhutdinov. Dropout: a simple way to prevent neural networks from overfitting. The journal of machine learning research, 15(1):1929– 1958, 2014.

[70] Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.

[71] Jacob Cohen. A coefficient of agreement for nominal scales. Educational and psychological measurement, 20(1):37–46, 1960.