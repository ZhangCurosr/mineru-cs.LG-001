# Attractor Image-Based Deep Learning of Arterial Pulse Waves for Age Classification

Sara Vardanega<sup>1</sup>, Patrick Segers <sup>2</sup>, Philip Aston <sup>3,4</sup>, Ernst Rietzschel<sup>5</sup>, Jordi Alastruey <sup>1∗</sup>, Manasi Nandi <sup>6∗</sup> \*\* These authors contributed equally to this work

<sup>1</sup>School of Biomedical Engineering and Imaging Sciences, King’s College London, London, UK,

<sup>2</sup>Institute of Biomedical Engineering and Technology, Ghent University, Ghent, Belgium,

<sup>3</sup>National Physical Laboratory, Teddington, UK,

<sup>4</sup>School of Mathematics and Physics, University of Surrey, Guildford, UK,

<sup>5</sup>Department of Cardiovascular Diseases, Ghent University Hospital, Ghent, Belgium,

<sup>6</sup>School of Cancer and Pharmaceutical Sciences, King’s College London, London, UK

Sara Vardanega, sara.vardanega@kcl.ac.uk

St Thomas’ Hospital, Westminster Bridge Rd, London SE1 7EH

## Abstract

Arterial pulse waveform morphology evolves with age, reflecting structural and functional changes in the cardiovascular system. Thus, vascular age is a valuable surrogate marker of cardiovascular health, and premature vascular ageing can indicate increased disease risk. Pulse wave analysis could support risk stratification in otherwise asymptomatic adults. We transformed pulse wave time-series data from photoplethysmography (PPG) and arterial tonometry into images, using the Symmetric Projection Attractor Reconstruction (SPAR) method. These SPAR images were used to train a convolutional neural network to classify healthy subjects into two closely spaced age groups (35–40 and 50–55 years). The model demonstrated consistent classification performance across internal and external test sets, achieving F1 scores above 70% for both PPG and tonometry signals. These results suggest that SPAR-derived pulse wave images contain discriminative morphological features even among healthy adults close in age. This proof-of-concept lays the groundwork for future research into the use of SPAR for early risk detection using smart wearables.

## 1 Introduction

Vascular ageing (VA) is a complex process that involves the gradual deterioration of arterial structure and function over time, negatively impacting organ function [4]. The gold-standard measurement for VA is carotid-femoral pulse wave velocity, but this requires trained personnel and is not routinely clinically available [7]. In healthy ageing, chronological and vascular ages typically correspond [5].

Deviations from this relationship, manifesting as premature VA, are associated with increased cardiovascular disease (CVD) risk. Therefore, early detection of premature VA is critical for timely prevention and management of CVD, which remains a leading global health burden. Non-invasive pulse wave signals from photoplethysmography (PPG) or arterial tonometry can help assess vascular age, through analysis of pulse wave morphology, which changes with age. PPG is an optical method used in clinical and wearable devices to measure pulse waves at sites like the wrist and finger [3]. Arterial tonometry, mainly used clinically, measures pressure from superficial arteries such as the radial or carotid arteries [9]. By comparing signal-based estimates of vascular age to a person’s chronological age, we hypothesise we could identify early VA in community-based settings.

This study focuses on image-based age classification leveraging PPG and tonometry-derived pulse waveforms, targeting two narrow yet clinically relevant age cohorts (35–40 and 50–55 years), representing a critical window during which individuals may begin to manifest subclinical CVD risk. Pulse waves were converted into images using the Symmetric Projection Attractor Reconstruction (SPAR) method, which condenses time-series data into a single image [1]. These images were used to train a convolutional neural network (CNN) to classify age, assigning each unseen image to the most likely of the two classes (≤40 or ≥50).

![](images/e455568a474cc781a66dc49ca32d8f8a7e6e3cdb84f1b20673be8bd49f200694.jpg)  
Figure 1: Full pipeline adopted in this study. Section a) illustrates the SPAR pipeline, which processes a raw signal into a SPAR attractor image (density plot). Section b) illustrates the CNN pipeline, where CNN takes as input the density plots and gives as output a class label.

## 2 Methods

## 2.1 Datasets

Two datasets were used in this study: Round 1 data from the Asklepios Study and the Vortal dataset (Table 1).

Both comprised recordings from participants free from diagnosed CVD at study initiation, collected in the supine position. The Asklepios Study [8] includes 2,524 individuals (30–59 years, 52% female) randomly sampled from two twinned Belgian communities. The exact chronological age of each participant was labelled. Arterial tonometry waveforms (20 s, 200 Hz) were acquired at a single centre by one trained operator using the same device. The Vortal dataset [2] contains finger PPG recordings from 56 subjects: 40 labelled as ’Young’ (18–35 years, 53% female) and 16 as ’Elderly’ (70+ years, 56% female). Data were collected in a London clinical trials unit (approx. 10 min per subject, 125 Hz).

## 2.2 Data selection

From the original Asklepios population, a subgroup was considered for this study. We included only participants aged 30-40 and 50-59 years. Obese subjects $\mathrm { ( B M I ~ \geq ~ 3 0 ~ k g / m ^ { 2 } } )$ and those with high blood pressure (systolic ≥ 140 mmHg or diastolic ≥ 90 mmHg) were excluded, following the 2024 ESC guidelines for hypertension classification [6]. For the Vortal dataset, we used the whole cohort, as no metadata were available. The characteristics of both populations, assumed to be healthy, are shown in Table 1.

Table 1: Population characteristics for each dataset used in this study.
<table><tr><td></td><td>Asklepios</td><td>Vortal</td></tr><tr><td>No. of participants</td><td>2,524</td><td>56</td></tr><tr><td>Sex (M/F)</td><td>1,223/1,301</td><td>27/29</td></tr><tr><td>Age range (years)</td><td>30-59</td><td>18-35,70+</td></tr><tr><td>Signal type</td><td>Tonometry</td><td>PPG</td></tr><tr><td>Raw signal length</td><td>20 s</td><td>10 min</td></tr><tr><td>Analysed signal length</td><td>20 s</td><td>20 s</td></tr></table>

## 2.3 Symmetric Projection Attractor Reconstruction (SPAR) method

The SPAR method [1] is a non-fiducial points-based method that combines mathematics and cardiovascular physiology to quantify pulse waveform morphology and variability. Given a raw pulse wave signal, the average cycle length is first estimated for the selected time window. Next, a three-dimensional (3D)

![](images/6b86403bbac5a322df0c7233e5b7bfd20edc3e09d00b8cf302edde2da0014329.jpg)

![](images/361b3fea660b968ea297db8fbbcbcefc070cc6f832fa73039cc80966bc2b41c2.jpg)

Table 2: Total and selected number of subjects in the Asklepios and Vortal datasets, along with the number of SPAR images used in each group.
<table><tr><td>Cohort</td><td>Total</td><td>Selected</td><td>Images</td></tr><tr><td>Asklepios 30-34</td><td>15</td><td>12</td><td>12</td></tr><tr><td>Asklepios 35-40</td><td>456</td><td>341</td><td>341</td></tr><tr><td>Asklepios 50-55</td><td>514</td><td>251</td><td>251</td></tr><tr><td>Asklepios 56-59</td><td>160</td><td>77</td><td>77</td></tr><tr><td>Vortal 18-35</td><td>40</td><td>40</td><td>1302</td></tr><tr><td>Vortal 70+</td><td>16</td><td>16</td><td>528</td></tr></table>

Figure 2: Examples of single tonometry (left) and PPG (right) pulse waveforms and their correspondent SPAR attractor images, generated using 20 seconds of data.

attractor is generated using three delay coordinates, each separated by onethird of the average cycle length. The 3D attractor is then projected onto a two-dimensional (2D) plane normal to the unit vector and converted into a density plot (Figure 1a). This plot is the final attractor image used to train and validate CNNs in this work.

For the Asklepios dataset, we used one 20-second segment per subject, resulting in one attractor image per participant. For the Vortal dataset, we extracted multiple non-overlapping 20-second segments from the 10-minute recordings, generating around 30 images per subject.

## 2.4 Model and metrics

The CNN used in this study was based on a simplified TinyVGG architecture originally presented by Wang et al. [10]. It comprises two convolutional blocks followed by a final classification layer (Figure 1b).

Model performance was evaluated using sensitivity $\mathrm { ( T P \ / \ ( T P + F N ) ) }$ , specificity (TN / (TN+FP)) and the F1 score, which is the harmonic mean of precision $\mathrm { ( T P \ / \ ( T P + F P ) ) }$ and sensitivity; with TP being the true positives, TN the true negatives, FP the false positives and FN the false negatives. In this study, the positive class corresponded to the 50-55 years group, and the negative class to the 35-40 years group. Therefore, sensitivity measured the model’s ability to correctly identify subjects aged ≥ 50 years, while specificity measured its ability to correctly classify those aged ≤ 40 years.

## 2.5 Training and testing

The model was trained on 80% of the Asklepios data from the 35-40 and 50-55 age groups. The remaining 20% was split equally into a validation set (10%) and a test set (10%). Given the relatively small size, 10-fold cross-validation was performed to improve robustness, thus obtaining 10 diferent sets of model weights which were evaluated separately when testing. The model was then tested on diferent test sets, reported in Table 2. When testing on the larger Asklepios population, the same 10-fold structure was maintained to minimise bias. The Vortal test set remained constant across all evaluations.

## 3 Results

The model reached average F1 scores of at least 70%, with sensitivity >67% and specificity >79% across all test sets (Table 3). Overall, performance improved with larger test sets and broader age ranges, suggesting robust model generalisation. The higher sensitivity observed in these sets indicates greater accuracy in classifying older subjects. Additionally, standard deviations decreased with broader test sets, reflecting more consistent performance across cross-validation folds.

Table 3: Model performance across diferent test sets.
<table><tr><td></td><td>F1 score (%) Sens (%) Spec (%)</td><td></td><td></td></tr><tr><td>Test Asklepios 35-40 and 50-55</td><td> $\overline { { 7 0 . 9 \pm 8 . 6 } }$ </td><td> $\overline { { 6 7 . 0 \pm 1 2 . 3 } }$ </td><td> $\overline { { 8 5 . 0 \pm 6 . 3 } }$ </td></tr><tr><td>Test Asklepios</td><td> $7 9 . 3 \pm 2 . 0$ </td><td> $7 0 . 5 \pm 3 . 1$ </td><td> $8 4 . 3 \pm 4 . 8$ </td></tr><tr><td>30-40 and 50-59</td><td></td><td></td><td></td></tr><tr><td>Test Vortal</td><td> $7 2 . 8 \pm 2 . 5$ </td><td> $8 6 . 9 \pm 5 . 9$ </td><td> $7 9 . 0 \pm 2 . 0$ </td></tr></table>

When evaluated on PPG signals, the model showed higher F1 scores compared to the baseline test on the Asklepios dataset, together with an increase in sensitivity and a slight decrease in specificity. These results highlight the ability of the model, combined with the SPAR method, to detect age-related morphological changes across both tonometry and PPG signals. Figure 2 illustrates these morphological diferences. In younger individuals (35-40 years for tonometry, 18-35 years for PPG), pulse waveforms display a distinct secondary peak, resulting in attractors with “looped” edges and closed centres. In contrast, older subjects (50-55 years for tonometry, 70+ years for PPG) exhibit attenuated or absent secondary peaks, producing more open attractors with reduced looping, highlighting age-related waveform changes.

## 4 Discussion and conclusion

We have presented a simple yet efective model that can classify chronological age within a healthy population free from CVD, using two non-invasive pulse wave signals: arterial tonometry and PPG. We purposefully focused on healthy individuals for whom chronological and vascular ages are expected to align. The strong classification performance achieved on both tonometry and PPG test sets shows that each signal type contains enough morphological information to distinguish between the two selected age groups. These results suggest that the model successfully captures age-related changes in the cardiovascular system that manifest as alterations in pulse wave morphology.

Importantly, the presence of noise in the tonometry recordings (Figure 2) did not impact the quality of the resulting attractor images, when compared to those derived from PPG. This indicates that the SPAR method is inherently robust to signal noise, preserving relevant morphological features despite variability in signal acquisition quality.

The main limitation of this work lies in the small size of the training set, which was mitigated through 10-fold cross-validation to improve model robustness and generalisability. Furthermore, the detailed selection criteria applied to the Asklepios dataset could not be applied to the Vortal dataset due to the absence of detailed metadata. As a result, the Vortal dataset may include individuals with high blood pressure or BMI. However, the large age diference between the younger and older Vortal groups supports its suitability as a proofof-concept PPG test set for age classification.

Model performance could be increased by implementing a more complex CNN model and by training on a larger and more diverse population. This would allow for evaluation of the interplay between model complexity and performance. Whilst CNNs can be directly applied to raw pulse waves signals, SPAR’s robustness on noisy signals and at-a-glance summary of multiple pulse waves as a single 2D image may be more intuitive for end users for visual interpretation.

In summary, while age-related diferences in pulse wave morphology are well established, this study demonstrates a SPAR-based CNN approach capable of classifying individuals into two closely spaced age ranges within a CVD-free population. Moreover, our findings indicate that tonometry and PPG signals are suficiently similar to enable this classification task. Given the infrastructure demands of gold-standard VA assessments such as pulse wave velocity, our findings support further research into the use of SPAR-transformed PPG signals from community or wearable devices for the early detection and stratification of cardiovascular risk.

## Acknowledgments

The project (22HLT01 QUMPHY) has received funding from the European Partnership on Metrology, co-financed from the European Union’s Horizon Europe Research and Innovation Programme and by the Participating States. Funding for King’s College London was provided by Innovate UK under the Horizon Europe Guarantee Extension, grant number 10087011.

## References

[1] Philip J Aston et al. Beyond HRV: attractor reconstruction using the entire cardiovascular waveform data for novel feature extraction. Physiological Measurement, 39(2):024001, 2018.

[2] Peter H Charlton, Timothy Bonnici, Lionel Tarassenko, David A Clifton, Richard Beale, and Peter J Watkinson. An assessment of algorithms to estimate respiratory rate from the electrocardiogram and photoplethysmogram. Physiological Measurement, 37(4):610, 2016.

[3] Peter H. Charlton et al. Wearable photoplethysmography for cardiovascular monitoring. Proceedings of the IEEE, 110(3):355–381, 2022.

[4] Rachel E Climie et al. Vascular ageing: moving from bench towards bedside. European Journal of Preventive Cardiology, 30(11):1101–1117, 2023.

[5] Magda R Hamczyk et al. Biological versus chronological aging: JACC focus seminar. Journal of the American College of Cardiology, 75(8):919– 930, 2020.

[6] John William McEvoy et al. 2024 ESC guidelines for the management of elevated blood pressure and hypertension. Eur Heart J, 45(38):3912–4018, 2024.

[7] Alexander Reshetnik et al. Oscillometric assessment of arterial stifness in everyday clinical practice. Hypertension Research, 40(2):140–145, 2017.

[8] Ernst-R Rietzschel et al. Rationale, design, methods and baseline characteristics of the Asklepios Study. European Journal of Preventive Cardiology, 14(2):179–191, 2007.

[9] Paolo Salvi et al. Noninvasive estimation of central blood pressure and analysis of pulse waves by applanation tonometry. Hypertension Research, 38(10):646–648, 2015.

[10] Zijie J Wang et al. CNN explainer: learning convolutional neural networks with interactive visualization. IEEE Transactions on Visualization and Computer Graphics, 27(2):1396–1406, 2020.