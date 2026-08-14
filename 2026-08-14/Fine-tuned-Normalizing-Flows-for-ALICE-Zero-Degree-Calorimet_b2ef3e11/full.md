# Fine-tuned Normalizing Flows for ALICE Zero Degree Calorimeter Fast Simulation

Emilia Majerz<sup>1[0009−0005−2034−0410]⋆</sup>, Jacek Otwinowski<sup>2[0000−0002−5471−6595]</sup>, Witold Dzwinel<sup>1[0000−0001−8321−5928]</sup>, and Jacek Kitowski<sup>1[0000−0003−3902−8310]</sup>

<sup>1</sup> AGH University of Krakow, Poland

<sup>2</sup> The Henryk Niewodniczański Institute of Nuclear Physics, Polish Academy of Sciences

Abstract. Simulating the ALICE Zero Degree Calorimeter (ZDC) neutron detector responses at the LHC is computationally expensive, requiring complex Monte Carlo chains. We develop a generative surrogate, focusing on Normalizing Flows (NFs). Through transfer learning, we pretrain on the full imbalanced dataset and fine-tune specialized models for diferent particle types (γ, n, Λ, K<sup>0</sup><sub>S</sub>, Σ<sup>+</sup>) using two gradual-unfreezing schemes. As standard ZDC metrics like Wasserstein distance overlook conditional structure, we introduce refined metrics: conditional weighted MAE, dispersion ratio, and Jaccard co-activation error, that better capture physics-relevant input-output dependencies and response variability. Our ensemble of fine-tuned models achieves a Wasserstein distance of 1.61 ± 0.02, outperforming baselines across all metrics. This work provides a generalizable NF-based framework for LHC detector simulation, combining NFs, conditional fine-tuning, and physics-motivated evaluation.

Keywords: High Energy Physics · Calorimeter Simulation · Normalizing Flows · Transfer Learning · Fine-tuning · Ensemble Methods · Fast Simulation

## 1 Introduction

The simulation of particle detectors at the LHC traditionally employs computationally demanding Monte Carlo simulation paradigms. This complexity restricts the volume of simulated collision data available for interpreting and calibrating real-world experimental data, driving researchers to pursue alternative methodologies. With their recent advances, fast simulations employing Generative Neural Networks ofer a promising alternative as a baseline for numerical models’ surrogates.

The Zero Degree Calorimeter (ZDC) of the ALICE experiment [7] is a particularly relevant target for fast simulation. Located 112.5 m from the Interaction Point 2 (IP2), it requires many steps of the Monte Carlo method to obtain the simulated response. The ZDC consists of two detectors on each side of IP2: the proton calorimeter (ZP) and the neutron calorimeter (ZN). The ZN contains a quartz fibre matrix of size 44 × 44 embedded in a high-density tungsten absorber, with dimensions of $7 . 0 4 \times 7 . 0 4 \times 1 0 0 \mathrm { c m } ^ { 3 }$ . When a particle enters the ZN, it initiates hadronic showers in the absorber, and the resulting charged particles produce Cherenkov light in the fibres. The readout is organised into four towers in a 2 × 2 layout (depicted as red and green squares in Fig. 1, presenting the experiment overview). The output of every other fibre is routed to a shared photomultiplier (PM), with four additional PMs collecting signals from each of the respective towers [7]. This structure makes the ZN a natural test case for imagebased surrogate modelling and for validating physically meaningful predictions.

![](images/385e07bf33f42452d93e6455def82923ce0c17c0639fdbd8f332abd6b7cb5014.jpg)  
Fig. 1: The relation between numerical simulation (Monte Carlo model) and neural network-based surrogate approach utilised in fast simulations.

Physics-Based Deep Learning (PBDL), also referred to as Scientific Machine Learning, combines physical modelling, numerical simulation, and modern deep learning. Here, we propose a supervised data-driven approach to build a fast surrogate for developing a fast surrogate of complex numerical models.

Fast calorimeter simulation using generative models has been widely studied. Early approaches used Generative Adversarial Networks [13] and autoencoders [13]. More recent work has explored Normalizing Flows (NFs) [12, 13, 2], difusion models [13], and flow matching [13, 19]. Several studies have also addressed ZN simulation specifically [3, 5, 11, 20, 4, 19, 16], but only few have investigated NFs for this detector [20, 16].

NFs are a suitable choice for this task because they provide exact likelihood estimation, invertibility, and strong expressive power. These properties are attractive in a physics setting, where predictive accuracy should be accompanied by interpretability and consistency with the underlying detector response. NFs are also well suited to tasks with substantial conditional variability, where the same input can correspond to multiple valid outputs, as in the ZN dataset.

The dataset is highly imbalanced: common particle types account for most responses, whereas rarer species are underrepresented. While this distribution reflects realistic physical conditions, it poses challenges for training predictive models, which may become biased toward dominant particle types. To address this, we pre-train a base model on the full multi-particle dataset and then finetune dedicated submodels for selected particle types using gradual unfreezing. During inference, the model corresponding to the input particle type is used, which improves performance across both frequent and rare categories. Although fine-tuning methods for NFs have been proposed [8, 18, 17, 2], to the best of our knowledge they have not been systematically evaluated for MAF-based simulators in a high-energy physics setting using physics-motivated metrics. In addition, we assess model behaviour with interpretability methods to check whether the learned dependencies are physically plausible.

Our work concerns the application of NFs in the simulation of the ZN responses, with a particular focus on transfer learning and fine-tuning. We extend the research on NFs presented in [20], and also make the following contributions:

1. We improve the MAF-based NF surrogate through fine-tuning and an ensemble approach.

2. We propose and compare two schemes for MAF-based NFs fine-tuning, utilising the gradual unfreezing technique.

3. We propose and compare four methods for noise removal after NFs training, observing significant diferences in model performance.

4. After carrying out a critical analysis of the metrics reported in the literature for ZDC surrogate evaluation, we identify their weaknesses, and propose refined variants.

5. We use SHAP [15] to assess whether the model relies on physically meaningful input variables, and show that the model’s predictions are primarily influenced by energy and momentum components, which is consistent with known detector response patterns.

The rest of the paper is organised as follows. Section 2 describes the dataset, Section 3 presents the methodology, and Section 4 reports the results and comparison with prior work. We conclude with an interpretability analysis of the ensemble.

## 2 Dataset

The dataset used in this study contains N=306,780 samples generated by the GEANT4 [1] software. Each sample is a pair consisting of a vector with features of particles produced by primary proton-proton collisions (called primary particles) – such as energy, three-momentum, position in three dimensions, mass, and charge – and a 44 × 44 matrix representing the ZN response. Each matrix element represents the number of photons detected in the corresponding fibre. The detector response is therefore treated as a single-channel image, where each pixel value corresponds to a photon count. To suppress low-signal random responses, we retain only matrices with at least 10 photons; after this filtering, the median photon count is 234.

An important property of the dataset is its strong class imbalance. The γ particles account for almost 61% of the data, n for over 23%, p for over 5%, Λ for over 3%, and the remaining 17 particle types together for around 7.5%. This imbalance may bias the model toward the dominant particle types and degrade performance on underrepresented ones. Possible remedies include rebalancing the dataset, applying standard techniques for imbalanced data [14], or training dedicated models for poorly represented particle types. Here, we investigate the last strategy.

Among the particles used in our experiments, the fractions of primary particles that hit the detector directly and produce a response of at least 10 photons are 45% for $\gamma ,$ 78% for $n ,$ 46% for Λ, 21% for $K _ { S } ^ { 0 }$ , and 0% for $\Sigma ^ { + }$ . When the fractions are lower, we should expect multiple showers and scattered responses induced by secondaries.

The diversity of detector responses is an additional challenge for any datadriven surrogate model. The same vector of input parameters can produce substantially diferent ZN responses. This happens because particles traverse a long distance between IP2 and the detector, during which secondary particles may be produced and primary trajectories may deviate from their original paths. Diferent particle types also exhibit diferent levels of response diversity: some produce fairly consistent showers, while others lead to much more variable shower positions and shapes. An illustrative example is shown in Fig. 2. To quantify this efect, we use the variance-based diversity measure introduced in [4]:

$$
f _ { \mathrm { d i v } } ( \mathbf { c } ) = \sum _ { i , j } \sqrt { \frac { 1 } { \lvert X _ { c } \rvert } \sum _ { t \in X _ { c } } \left( x _ { i j } ^ { t } - \mu _ { i j } \right) ^ { 2 } } ,\tag{1}
$$

where $\ v x _ { i j } ^ { t }$ is the pixel value at coordinates $i , j$ for the detector response $t , \mu _ { i j }$ is the mean pixel value at $i , j$ for the unique input vector $\mathbf { c } ,$ and $| X _ { c } |$ is the number of detector responses associated with input c. This quantity is positively correlated with particle mass and energy.

![](images/3fde50160f779fb145f03b3646a3c4f73e950b41170c2418380f4e78ba3c6f4b.jpg)  
Fig. 2: The illustration of the diversity of the detector responses. Each column represents three GEANT4 simulations of the same input particle.

## 3 Methodology

We use a three-stage approach:

1. Pre-train an NF model on the full, imbalanced dataset to capture general detector-response patterns.

2. Fine-tune the pre-trained model separately for selected particle types using the corresponding subsets.

3. Combine the fine-tuned models into an ensemble and verify its performance.

We split the dataset into training, validation, and test sets in a 70:10:20 ratio. Models with the best validation performance were selected for testing. During fine-tuning, we used only particle-specific samples from the corresponding splits to avoid data leakage. Each model was evaluated five times, generating five samples per input vector on the test set.

## 3.1 Two-stage modelling

In our initial experiments, we trained NFs directly on the primary particle features. To reduce the dynamic range of the detector-response images, we applied a log transform to the pixel values. Although the generated showers were visually plausible, the predicted photon counts did not match the data well. We therefore adopted a two-stage approach: the NF generates a normalized response shape, which is then rescaled by an auxiliary model predicting the expected total photon count. We also provided the expected photon count as an additional input feature during NF training, following [12].

We compared several models for predicting the photon sum, including NFs, SVMs, GBRs, and BNNs. The BNN performed best and was used in all experiments (MAE: 53.20, RMSE: 111.47). Although the absolute errors appear large, the median percentage error of the best BNN was 12%, which seems acceptable given the stochastic nature of the target.

For the NF model, we follow the configuration of [20], itself based on [12]. The model uses the MAF architecture with rational quadratic spline (RQS) transformations [6], whose parameters are produced by MADE blocks [9]. The NF maps the data distribution to a simple base distribution through a sequence of invertible transformations (Fig. 3). Sampling proceeds in the reverse direction: a point is drawn from the base distribution, the MADE blocks compute the spline parameters, and the transformations are applied successively. Permutation layers are inserted between spline blocks to increase flexibility. Compared with [20], we improved the baseline by replacing the conditional diagonal Normal base with a standard Normal and using the pixel-value recalculation post-processing scheme.

## 3.2 Performance metrics

Because multiple detector responses may be valid for the same input, direct pixelwise comparison is not meaningful. Following prior work on the ZN detector [3, 5, 11, 20, 4], we use the mean 1-Wasserstein distance between the distributions of

![](images/b91ae1e844ffbc3eba6bc916228c93a481e34ac8149a7d6b0f4335ce48b314bd.jpg)  
Fig. 3: Block diagram of the NF model used. RQS layers are interleaved with permutation (PERM) and inversion (INV - order reversal) layers.

photons collected by each PM. We refer to these groups as channels and report the average over $m = 5$ channels:

$$
W _ { 1 } ( w , \hat { w } ) = \frac { 1 } { m } \sum _ { i = 1 } ^ { m } \int _ { 0 } ^ { 1 } \left| F _ { w _ { i } } ^ { - 1 } ( z ) - F _ { \hat { w } _ { i } } ^ { - 1 } ( z ) \right| d z .\tag{2}
$$

Here, $F _ { w _ { i } } ^ { - 1 }$ is the inverse cumulative distribution function of channel distribution $w _ { i } .$ , and $\hat { w } _ { i }$ denotes the predicted distribution.

We compute channel values by applying five checkerboard masks to each response image and summing the values within predefined fibre groups. A schematic overview of the masks is shown in Fig. 4.

![](images/86fa712288ba889f3ad3bc98e442d620263131ad1ab991d9b697a504657122a7.jpg)  
Fig. 4: A schematic overview of checkerboard masks applied to detector responses to obtain channel values. Each mask is the same size as the response.

Although the Wasserstein distance captures global distributional similarity, it does not assess whether the response matches the input condition. Thus, we compute a conditional mean absolute error over unique inputs, as in [16]:

$$
\mathrm { M A E } _ { c o n d } ( w , \hat { w } ) = \frac { 1 } { \sum _ { k } p _ { k } } \sum _ { k = 1 } ^ { c } p _ { k } \cdot \frac { 1 } { m } \sum _ { i = 1 } ^ { m } \left| \overline { { w _ { i } ^ { k } } } - \overline { { \hat { w } _ { i } ^ { k } } } \right| .\tag{3}
$$

We use logarithmic weights, $p _ { k } = \log ( 1 + n _ { k } )$ , to reduce the influence of highly repeated conditions while still favouring well-sampled regions of phase space.

To quantify channel variability, we use a per-channel median absolute deviation (MAD) with shrinkage toward a particle-level prior. MAD is robust to outliers, which is useful for detector responses that may contain rare, extreme showers, and shrinkage further stabilises it in low-sample cases. We compute the prior $M A D _ { p a r t i c l e }$ as a weighted median MAD value computed for reference inputs with at least 5 responses available. For each unique input and channel:

$$
\mathrm { M A D } _ { u , c } = \mathrm { m e d i a n } _ { i } \left( \left| R _ { u , c } ^ { i } - \mathrm { m e d i a n } _ { j } ( R _ { u , c } ^ { j } ) \right| \right) ,\tag{4}
$$

$$
\widetilde { \mathrm { M A D } } _ { u , c } = \alpha _ { u } \mathrm { M A D } _ { u , c } + \left( 1 - \alpha _ { u } \right) \mathrm { M A D } _ { p a r t i c l e , c } , \qquad \alpha _ { u } = \frac { N _ { u } } { N _ { u } + \kappa } .\tag{5}
$$

Here, κ is the 10th percentile of the number of responses per particle type. We compare reference and generated variability using the dispersion ratio:

$$
\delta _ { u , c } = \frac { \left| \widetilde { \mathrm { M A D } } _ { u , c } ^ { g e n } - \widetilde { \mathrm { M A D } } _ { u , c } ^ { r e f } \right| } { \widetilde { \mathrm { M A D } } _ { u , c } ^ { g e n } + \widetilde { \mathrm { M A D } } _ { u , c } ^ { r e f } + \tau } ,\tag{6}
$$

which provides a bounded symmetric score, treating overdispersion and underdispersion equally. We compute this separately for each channel and report the mean over channels, followed by a weighted mean over unique inputs. Cases with fewer than two samples are excluded.

We also measure spatial co-activation between neighbouring channels using a Jaccard-based error. For each event, we define a binary hit indicator and compute the Jaccard index for each neighbouring pair $( c _ { 1 } , c _ { 2 } )$ :

$$
J _ { c _ { 1 } , c _ { 2 } } = \frac { \sum _ { i } \Im ( B _ { u , c _ { 1 } } ^ { i } = 1 \land B _ { u , c _ { 2 } } ^ { i } = 1 ) } { \sum _ { i } \Im ( B _ { u , c _ { 1 } } ^ { i } = 1 \lor B _ { u , c _ { 2 } } ^ { i } = 1 ) } , \qquad B _ { u , c } ^ { i } = \Im ( R _ { u , c } ^ { i } > 0 ) .\tag{7}
$$

If both channels are always zero, we set $J _ { c _ { 1 } , c _ { 2 } } = 1$ . The final score for a unique input is

$$
\mathrm { J a c c E r r } _ { u } = \frac { 1 } { P } \sum _ { \mathrm { p a i r s } } \left| J _ { u } ^ { g e n } - J _ { u } ^ { r e f } \right| ,\tag{8}
$$

where $P$ is the number of neighbouring channel pairs. We report the weighted mean over unique inputs.

## 3.3 Fine-tuning Normalizing Flows

After training on the full dataset, we fine-tune separate models for selected particle types. We focus on particle types for which the baseline Wasserstein score is relatively poor and exclude very rare classes from the fine-tuning study. The selected particles are $\gamma , n , A , K _ { S } ^ { 0 }$ , and $\Sigma ^ { + }$ . Although fine-tuning all particle types might yield further gains, our goal here is to validate the methodology rather than to optimise every class exhaustively.

Standard neural-network fine-tuning usually freezes early layers and retrains later ones. For NFs, we consider two gradual-unfreezing directions: from the base-distribution side toward the data side, and in the opposite direction. We evaluate both because the flow structure makes either choice plausible. In each step, we unfreeze one MADE block at a time.

## 3.4 Post-processing

Discrete-valued data, such as detector responses, are often modelled by adding small uniform noise so that each integer value corresponds to a small interval. We use the same idea here. In the two-stage setup, the NF first generates a normalized response shape, which is then rescaled by the predicted total photon count (including the noise contribution). After this rescaling, we remove the added noise using one of four schemes:

1. Apply the floor function.

2. Subtract the expected preprocessing noise and round to the nearest integer.

3. Apply the floor function and then renormalize and rescale the response to preserve the total photon count. Finish with rounding.

4. Subtract the expected preprocessing noise, round, and then renormalize and rescale. Finish with rounding.

The renormalization steps are used because direct floor or round operations may force some pixels to zero and distort the total photon count. After renormalization, we rescale the responses, and round the values to integers again.

## 3.5 Ensemble model

The fine-tuned models can be combined with the baseline model in an ensemble. For each input, we use the dedicated fine-tuned model if one exists; otherwise, we use the baseline model trained on the full dataset. The ensemble therefore assigns specialised models to particle types that benefit from fine-tuning while retaining a shared model for the rest. In principle, the submodels may also difer in architecture, but in this work we use a uniform MAF-based setup.

## 4 Results

We report means and standard deviations over five runs of generating responses on the test set, which covers 20% of the dataset.

## 4.1 Post-processing approaches

In Section 3.4, we introduced four methods for noise removal. The chosen postprocessing approach has a large efect on Wasserstein performance, and no single method is optimal across all particle types, as shown in Table 1. Method 3 is best overall, but particle-specific optima difer. Using the default post-processing method, the Wasserstein score for the whole dataset is $1 . 7 3 \pm 0 . 0 2$ . When a dedicated method is selected for each particle type, the score improves slightly to $1 . 7 0 \pm 0 . 0 2$

## 4.2 Transfer learning and fine-tuning

We fine-tuned the baseline NF model (Baseline\_100) for γ, n Λ, $K _ { s } ^ { 0 }$ , and $\Sigma +$ particles, separately. The model was first trained for 100 epochs and then finetuned for another 100 epochs using the two schemas introduced in Section 3.3: FT\_100\_ND, where unfreezing starts from layers close to the Normal distribution, and FT\_100\_DN, where unfreezing starts from layers close to the data. We also compared the results with a baseline model trained for 200 epochs (Baseline\_200). In addition, we trained separate models from scratch for each particle type (Individual), with 100 and 200 epochs, but these performed substantially worse than the other approaches.

Table 1: Comparison of noise-removal methods on samples generated by the best-performing baseline model, quantified by the Wasserstein score.
<table><tr><td></td><td>Method 1</td><td>Method 2</td><td>Method 3</td><td>Method 4</td></tr><tr><td>All</td><td> $\overline { { 2 . 6 0 \pm 0 . 0 3 } }$ </td><td> $\overline { { 4 . 3 3 \pm 0 . 0 2 } }$ </td><td> $\mathbf { 1 . 7 3 \pm 0 . 0 2 }$ </td><td> $\overline { { 2 . 4 6 \pm 0 . 0 2 } }$ </td></tr><tr><td>Gamma (γ)</td><td> $3 . 6 7 \pm 0 . 0 2$ </td><td> $3 . 9 8 \pm 0 . 0 1$ </td><td> ${ \bf 1 . 8 0 \pm 0 . 0 2 }$ </td><td> $1 . 9 2 \pm 0 . 0 2$ </td></tr><tr><td>Neutron (n)</td><td> ${ \bf 3 . 4 7 \pm 0 . 0 7 }$ </td><td> $9 . 5 5 \pm 0 . 0 3$ </td><td> $3 . 7 0 \pm 0 . 0 7$ </td><td> $5 . 9 6 \pm 0 . 0 6$ </td></tr><tr><td>Lambda (Λ)</td><td> ${ \bf 1 6 . 2 3 \pm 0 . 2 2 }$ </td><td> $1 8 . 6 3 \pm 0 . 1 7$ </td><td> $1 7 . 3 9 \pm 0 . 2 4$ </td><td> $1 9 . 6 0 \pm 0 . 2 0$ </td></tr><tr><td>K-Short  $( K _ { S } ^ { 0 } )$ </td><td> $1 2 . 5 2 \pm 0 . 1 2$ </td><td> ${ \bf 1 0 . 5 1 \pm 0 . 0 6 }$ </td><td></td><td> $1 1 . 6 6 \pm 0 . 0 7 1 1 . 4 4 \pm 0 . 1 1$ </td></tr><tr><td>Sigma+  $( \Sigma ^ { + } )$ </td><td> ${ \bf 3 3 . 9 4 \pm 0 . 3 5 }$ </td><td> $3 5 . 0 4 \pm 0 . 2 1$ </td><td> $4 0 . 7 0 \pm 0 . 1 6 4 1 . 1 1 \pm 0 . 1 5$ </td><td></td></tr></table>

Two-stage modelling: the NF model (second stage) analysis We first assess the NF models responsible for modelling shower position and shape independently of the model predicting the expected number of photons. To do this, we use the photon counts derived from the dataset as input.

The analysed particles difer substantially in sample size, in the average number of responses per unique input, and in their level of diversity (Eq. 1), as provided in Table 2. Here, particle diversity is computed as a weighted mean of per-unique-input diversities normalised to [0, 1], with weights given by the number of responses available for each input.

Table 2: Basic statistics of the dataset with respect to the analysed particle types.
<table><tr><td>Gamma (γ) Neutron (n) Lambda (Λ) K-Short</td><td></td><td></td><td></td><td> $\overline { { ( K _ { S } ^ { 0 } ) } }$ </td><td>Sigma+ (Σ+)</td></tr><tr><td>No. of test samples</td><td>37,329</td><td>14,310</td><td>1,861</td><td>1,305</td><td>247</td></tr><tr><td>No. of unique test inputs</td><td>608</td><td>207</td><td>45</td><td>87</td><td>19</td></tr><tr><td>Diversity</td><td>0.10</td><td>0.30</td><td>0.48</td><td>0.47</td><td>0.58</td></tr></table>

Table 3 shows the results of our experiments. None of the approaches is best in every case, but FT\_100\_ND stands out as the strongest overall. The $\Sigma ^ { + }$ case remains exploratory because only a small number of samples is available. With that caveat, FT\_100\_ND performs best in 3 out of 4 cases for the MAE<sub>cond</sub>, δ, and JaccErr metrics. These metrics are computed at the level of unique input vectors, preserving the physical input-output relationship. By contrast, the Wasserstein score is more global and can favour distributional similarity even when the conditional structure is less accurate. FT\_100\_DN performs better than the baseline in all metrics for the γ case, which suggests that this fine-tuning direction can also be useful for some datasets.

To assess whether the diferences between models are statistically significant, we performed bootstrap resampling of the unique-input level for the metrics $\mathrm { M A E } _ { c o n d } , ~ \delta $ , and JaccErr. We focus on FT\_100\_ND because it usually performs better than FT\_100\_DN, and on the Baseline\_200. We exclude Σ+ due to small number of samples available. First, we computed each metric at the unique-input level and averaged the results over five runs. Then, for each of the 10,000 bootstrap samples, we resampled unique inputs, preserving all associated responses and weights, and computed the diferences between the fine-tuned and baseline models. This yielded p-values, 95% confidence intervals, and the fraction of bootstrap samples favouring the fine-tuned model. The results are shown in Table 4. For the γ particles, all fine-tuned results are significantly better than the baselines. This suggests that, when enough particle-specific samples are available, fine-tuning is the most eficient way to obtain a surrogate model. For the other cases, the conclusions are less clear: some results are statistically significant in favour of the fine-tuned models, while none favour the baseline. Whenever the fine-tuned models outperform the baseline in Table 3, the majority of bootstrap samples follow the same trend. However, this analysis should be interpreted cautiously. In some cases, only a small number of samples is available per unique input. In addition, resampling the observed unique inputs changes the efective input distribution, which may not correspond to a physically plausible perturbation.

Table 3: Comparison of model performances for diferent particles and metrics, with input photon counts derived from the dataset. Baseline denotes models trained on the full dataset and FT\_100 denotes fine-tuned Baseline\_100 models. The best results are in bold.
<table><tr><td>Metric</td><td>Model</td><td></td><td>Gamma (γ)</td><td> $\overline { { { \mathrm { N e u t r o n ~ } } ( n ) } }$ </td><td> $\overline { { \mathrm { L a m b d a } \left( \varLambda \right) } }$ </td><td> $\overline { { \mathrm { K - S h o r t ~ } \left( K _ { S } ^ { 0 } \right) } }$ </td><td> $\overline { { \mathrm { S i g m a } + ( \Sigma ^ { + } ) } }$ </td></tr><tr><td>WS</td><td>Baseline_100</td><td></td><td> $\overline { { 1 . 0 2 \pm 0 . 0 2 } }$ </td><td> $\overline { { 2 . 5 1 \pm 0 . 0 3 } }$ </td><td> $5 . 6 2 \pm 0 . 2 9$ </td><td> $\overline { { 4 . 9 6 \pm 0 . 1 4 } }$ </td><td> $\overline { { 1 0 . 8 9 \pm 0 . 6 8 } }$ </td></tr><tr><td>WS</td><td> $\mathrm { B a s e l i n e \_ 2 0 0 }$ </td><td></td><td> $0 . 9 5 \pm 0 . 0 3$ </td><td> ${ \bf 2 . 2 1 \pm 0 . 0 3 }$ </td><td> ${ \bf 5 . 1 9 \pm 0 . 3 4 }$ </td><td> $5 . 0 7 \pm 0 . 1 6$ </td><td> $1 1 . 0 9 \pm 0 . 6 2$ </td></tr><tr><td>WS</td><td>FT_100_ND</td><td></td><td> $0 . 8 8 \pm 0 . 0 1$ </td><td> $2 . 9 0 \pm 0 . 0 4$ </td><td> $5 . 7 0 \pm 0 . 1 9$ </td><td> ${ \bf 4 . 9 4 \pm 0 . 1 5 }$ </td><td> ${ \bf 9 . 9 1 \pm 0 . 6 8 }$ </td></tr><tr><td>WS</td><td>FT_100_DN</td><td></td><td> $\mathbf { 0 . 8 7 \pm 0 . 0 1 }$ </td><td> $2 . 2 8 \pm 0 . 0 5$ </td><td> $5 . 5 6 \pm 0 . 2 4$ </td><td> $5 . 1 0 \pm 0 . 1 0$ </td><td> $1 0 . 3 9 \pm 0 . 4 0$ </td></tr><tr><td> $\overline { { \mathrm { M A E } _ { c o n d } } }$ </td><td>Baseline 100</td><td></td><td> $\overline { { 2 . 0 1 \pm 0 . 0 3 } }$ </td><td> $\overline { { 3 . 9 4 \pm 0 . 0 5 } }$ </td><td> $6 . 9 6 \pm 0 . 4 4$ </td><td> $\overline { { 6 . 0 3 \pm 0 . 0 8 } }$ </td><td> $\overline { { { \bf { 6 . 3 7 \pm 0 . 4 6 } } } }$ </td></tr><tr><td> $\mathrm { M A E } _ { c o n d }$ </td><td>Baseline 200</td><td></td><td> $1 . 8 6 \pm 0 . 0 2$ </td><td> ${ \bf 3 . 6 7 \pm 0 . 0 6 }$ </td><td> $5 . 9 5 \pm 0 . 3 2$ </td><td> $6 . 0 1 \pm 0 . 1 5$ </td><td> $6 . 5 1 \pm 0 . 2 0$ </td></tr><tr><td> $\mathrm { M A E } _ { c o n d }$ </td><td>FT_100_ND</td><td></td><td> ${ \bf 1 . 3 9 \pm 0 . 0 1 }$ </td><td> $3 . 8 5 \pm 0 . 0 7$ </td><td> ${ \bf 5 . 7 8 \pm 0 . 1 6 }$ </td><td> ${ \bf 5 . 5 8 \pm 0 . 1 3 }$ </td><td> $7 . 0 4 \pm 0 . 6 6$ </td></tr><tr><td> $\mathrm { M A E } _ { c o n d }$ </td><td>FT 100 DN</td><td></td><td> $1 . 6 5 \pm 0 . 0 1$ </td><td> $3 . 7 5 \pm 0 . 1 2$ </td><td> $5 . 8 7 \pm 0 . 4 2$ </td><td> $5 . 8 5 \pm 0 . 1 5$ </td><td> $6 . 5 3 \pm 0 . 2 9$ </td></tr><tr><td>δ</td><td>Baseline_100</td><td></td><td> $\overline { { 0 . 1 3 4 \pm 0 . 0 0 3 } }$ </td><td> $\overline { { 0 . 0 8 6 \pm 0 . 0 0 1 } }$ </td><td> $\overline { { 0 . 0 8 3 \pm 0 . 0 0 4 } }$ </td><td>0.230 ± 0.006</td><td> $\overline { { 0 . 2 7 9 \pm 0 . 0 0 8 } }$ </td></tr><tr><td>δ</td><td>Baseline_200</td><td></td><td> $0 . 1 2 4 \pm 0 . 0 0 2$ </td><td> $0 . 0 8 1 \pm 0 . 0 0 1$ </td><td> $0 . 0 7 9 \pm 0 . 0 0 2$ </td><td>0.238 ± 0.012</td><td> $\mathbf { 0 . 2 6 3 \pm 0 . 0 1 3 }$ </td></tr><tr><td>δ</td><td>FT_100</td><td></td><td> $\mathbf { N D 0 . 0 9 8 \pm 0 . 0 0 2 }$ </td><td> $\mathbf { 0 . 0 7 9 \pm 0 . 0 0 2 }$ </td><td> $\mathbf { 0 . 0 7 4 \pm 0 . 0 0 5 }$ </td><td>0.237 ± 0.008</td><td> $0 . 2 7 5 \pm 0 . 0 1 4$ </td></tr><tr><td>δ</td><td>FT_100</td><td></td><td> $\mathrm { \Delta D N ~ \ 0 . 1 1 9 \pm 0 . 0 0 2 }$ </td><td> $0 . 0 8 6 \pm 0 . 0 0 3$ </td><td>0.078 ± 0.006</td><td>0.244 ± 0.006</td><td> $0 . 2 9 2 \pm 0 . 0 2 0$ </td></tr><tr><td></td><td>JaccErr Baseline</td><td></td><td> $\overline { { 1 0 0 ~ 0 . 1 0 0 \pm 0 . 0 0 1 } }$ </td><td> $\overline { { 0 . 0 4 6 3 \pm 0 . 0 0 1 1 } }$ </td><td>0.079 ± 0.005</td><td>0.240 ± 0.005</td><td> $\overline { { 0 . 2 5 5 \pm 0 . 0 1 6 } }$ </td></tr><tr><td></td><td>JaccErr Baseline</td><td></td><td> $_ { 2 0 0 } ~ 0 . 0 9 8 \pm 0 . 0 0 1$ </td><td> $0 . 0 4 6 3 \pm 0 . 0 0 1 3$ </td><td>0.081 ± 0.006 0.247 ± 0.009</td><td></td><td> $0 . 2 4 8 \pm 0 . 0 1 8$ </td></tr><tr><td></td><td>JaccErr FT 100</td><td></td><td> $\mathbf { N D 0 . 0 9 1 } \pm \mathbf { 0 . 0 0 1 }$ </td><td>0.0455 ± 0.0016 0.083 ± 0.001 0.226 ± 0.006</td><td></td><td></td><td> $\mathbf { 0 . 2 2 8 \pm 0 . 0 1 7 }$ </td></tr><tr><td></td><td>JaccErr FT 100 DN</td><td></td><td> $0 . 0 9 5 \pm 0 . 0 0 2$ </td><td> $0 . 0 4 7 0 \pm 0 . 0 0 1 5$ </td><td> $0 . 0 8 0 \pm 0 . 0 0 3$ </td><td>0.240 ± 0.007</td><td> $0 . 2 5 0 \pm 0 . 0 0 7$ </td></tr></table>

Table 4: Comparison between FT\_100\_ND and the Baseline\_200 models for diferent particles based on bootstrap samples. Results with p-value < 0.05 are highlighted in dark green and those close to $\mathrm { p - v a l u e } = 0 . 0 5 \mathrm { ~ - ~ }$ in green .
<table><tr><td rowspan=1 colspan=4>Metric Result type Gamma(γ)Neutron(n)Lambda(∆)K-Short $\overline { { ( K _ { S } ^ { 0 } ) } }$ </td></tr><tr><td rowspan=1 colspan=1> $\overline { { \mathrm { M A E } _ { c o n d } } }$      CI</td><td rowspan=1 colspan=1>-0.603-0.347</td><td rowspan=1 colspan=1>-0.093 0.469 -1.334 0.805</td><td rowspan=1 colspan=1>-0.824 -0.025</td></tr><tr><td rowspan=1 colspan=1> $\mathrm { M A E } _ { c o n d }$   $\mathrm { P , P _ { - } v a l }$ </td><td rowspan=1 colspan=1>1.000, 0.000</td><td rowspan=1 colspan=1>0.104, 0.2080.620, 0.761</td><td rowspan=1 colspan=1>0.980, 0.041</td></tr><tr><td rowspan=1 colspan=1> $\overline { { \delta } }$          CI</td><td rowspan=1 colspan=1>-0.030-0.021</td><td rowspan=2 colspan=1>-0.005 0.002-0.011 0.0000.853, 0.2950.973, 0.054</td><td rowspan=2 colspan=1>-0.014 0.0110.589, 0.821</td></tr><tr><td rowspan=1 colspan=1>δ     P, P_val</td><td rowspan=1 colspan=1>1.000, 0.000</td></tr><tr><td rowspan=1 colspan=1>JaccErr      CI</td><td rowspan=1 colspan=1>-0.010 -0.005</td><td rowspan=1 colspan=1>-0.003 0.002 -0.006 0.009</td><td rowspan=1 colspan=1>-0.033-0.010</td></tr><tr><td rowspan=1 colspan=1>JaccErr   $\mathrm { P , P _ { - } v a l }$ </td><td rowspan=1 colspan=1>1.000, 0.000</td><td rowspan=1 colspan=1>0.712, 0.576 0.342, 0.685</td><td rowspan=1 colspan=1>1.000, 0.000</td></tr></table>

Two-stage modelling: the full pipeline analysis Here, we analyse the performance of the full two-stage pipeline. The results for all models and metrics are shown in Table 5. For each particle and setup, we report scores obtained using the best-performing noise-removal method with respect to the Wasserstein score. Because training models from scratch for each particle performs significantly worse than other approaches, it is excluded from the analysis.

Table 5: Comparison of model performances for diferent particles and metrics. Baseline denotes models trained on the full dataset and $F T$ \_100 denotes finetuned Baseline\_100 models. Superscripts indicate the noise removal method used. The best results are in bold.
<table><tr><td>Metric</td><td>Model</td><td></td><td>Gamma (γ)</td><td> $\overline { { \mathrm { N e u t r o n ~ } ( n ) } }$ </td><td>Lambda (Λ) K-Short (KS) Sigma+ (Σ+)</td><td></td><td></td></tr><tr><td>WS</td><td>Baseline 100</td><td></td><td> $\overline { { 1 . 8 0 \pm 0 . 0 2 ^ { 3 } } }$ </td><td> $\overline { { { \bf 3 . 4 7 \pm 0 . 0 7 ^ { \mathrm { T } } } } }$ </td><td> $\overline { { 1 6 . 2 3 \pm 0 . 2 2 } }$ </td><td> $\mathbf { 1 0 . 5 1 \pm 0 . 0 6 ^ { 2 } }$ </td><td> $\overline { { 3 3 . 9 4 \pm 0 . 3 5 } } ^ { \mathrm { T } }$ </td></tr><tr><td>WS</td><td>Baseline_200</td><td></td><td> $1 . 7 5 \pm 0 . 0 2 ^ { 3 }$ </td><td> $3 . 8 5 \pm 0 . 0 5 ^ { 1 }$ </td><td> $1 6 . 8 3 \pm 0 . 1 6 ^ { 1 }$ </td><td> $1 1 . 0 3 \pm 0 . 1 4 ^ { 2 }$ </td><td> $3 2 . 3 8 \pm 0 . 4 6 ^ { 1 }$ </td></tr><tr><td>WS</td><td>FT_100_ND</td><td></td><td> $\mathbf { 1 . 5 8 \pm 0 . 0 1 ^ { 3 } }$ </td><td> $4 . 5 3 \pm 0 . 0 5 ^ { 3 }$ </td><td> $\mathbf { 1 5 . 7 9 \pm 0 . 2 1 ^ { 1 } }$ </td><td> $1 0 . 9 4 \pm 0 . 2 3 ^ { 2 }$ </td><td> $\mathbf { 3 1 . 5 2 \pm 0 . 6 4 } ^ { 1 }$ </td></tr><tr><td>WS</td><td>FT 100 DN</td><td></td><td> $1 . 7 0 \pm 0 . 0 2 ^ { 3 }$ </td><td> $3 . 5 6 \pm 0 . 0 7 ^ { 1 }$ </td><td> $1 6 . 0 0 \pm 0 . 2 6 ^ { 1 }$ </td><td> $1 0 . 5 6 \pm 0 . 2 0 ^ { 2 }$ </td><td> $3 2 . 6 2 \pm 0 . 4 1 ^ { 1 }$ </td></tr><tr><td> $\overline { { \mathrm { M A E } _ { c o n d } } }$ </td><td>Baseline 100</td><td></td><td> $\overline { { 4 . 8 9 \pm 0 . 0 2 ^ { 3 } } }$ </td><td> $\overline { { 7 . 0 4 \pm 0 . 0 3 } } ^ { 1 }$ </td><td> $\overline { { 1 0 . 6 3 \pm 0 . 4 2 } } ^ { 1 }$ </td><td> $\overline { { 9 . 7 7 \pm 0 . 1 7 ^ { 2 } } }$ </td><td> $\overline { { 1 7 . 2 3 \pm 0 . 2 6 ^ { 1 } } }$ </td></tr><tr><td> $\mathrm { M A E } _ { c o n d }$ </td><td>Baseline_200</td><td></td><td> $4 . 7 8 \pm 0 . 0 2 ^ { 3 }$ </td><td> $6 . 9 5 \pm 0 . 0 8 ^ { 1 }$ </td><td> $9 . 7 0 \pm 0 . 1 8 ^ { 1 }$ </td><td> ${ \bf 9 . 5 9 \pm 0 . 0 3 ^ { 2 } }$ </td><td> ${ \bf 1 6 . 6 3 \pm 0 . 2 2 } ^ { 1 }$ </td></tr><tr><td> $\mathrm { M A E } _ { c o n d }$ </td><td>FT 100−ND</td><td></td><td> ${ \bf 4 . 5 3 \pm 0 . 0 1 ^ { 3 } }$ </td><td> $7 . 2 9 \pm 0 . 0 3 ^ { 3 }$ </td><td> $9 . 7 3 \pm 0 . 1 9 ^ { 1 }$ </td><td> $1 0 . 1 1 \pm 0 . 1 3 ^ { 2 }$ </td><td> $1 7 . 1 1 \pm 0 . 3 3 ^ { 1 }$ </td></tr><tr><td> $\mathrm { M A E } _ { c o n d }$ </td><td>FT_100_DN</td><td></td><td> $4 . 6 5 \pm 0 . 0 2 ^ { 3 }$ </td><td> ${ \bf 6 . 7 8 \pm 0 . 0 8 ^ { 1 } }$ </td><td> $\mathbf { 9 . 4 3 \pm 0 . 0 8 ^ { 1 } }$ </td><td> $9 . 8 8 \pm 0 . 1 9 ^ { 2 }$ </td><td> $1 7 . 6 3 \pm 0 . 1 4 ^ { 1 }$ </td></tr><tr><td> $\overline { { \delta } }$ </td><td>Baseline_100</td><td></td><td> $\overline { { 0 . 1 4 6 \pm 0 . 0 0 1 ^ { 3 } } }$ </td><td> $\mathbf { 0 . 1 4 5 \pm 0 . 0 0 4 } ^ { \mathrm { T } }$ </td><td> $\mathbf { 0 . 1 4 8 \pm 0 . 0 0 5 ^ { 1 } }$ </td><td> $\overline { { 0 . 3 0 6 \pm 0 . 0 0 6 ^ { 2 } } }$ </td><td> $\overline { { 0 . 3 3 7 \pm 0 . 0 0 8 ^ { 1 } } }$ </td></tr><tr><td> $\delta$ </td><td>Baseline_200</td><td></td><td> $0 . 1 3 7 \pm 0 . 0 0 2 ^ { 3 }$ </td><td> $0 . 1 5 2 \pm 0 . 0 0 2 ^ { 1 }$ </td><td> $0 . 1 5 4 \pm 0 . 0 0 6 ^ { 1 }$ </td><td> $0 . 2 9 6 \pm 0 . 0 0 6 ^ { 2 }$ </td><td> $0 . 3 3 4 \pm 0 . 0 1 3 ^ { 1 }$ </td></tr><tr><td> $\delta$ </td><td> $\mathrm { F T \_ 1 0 0 } _ { \cdot }$ </td><td>ND</td><td> $\mathbf { 0 . 1 2 0 \pm 0 . 0 0 1 ^ { 3 } }$ </td><td> $0 . 1 8 4 \pm 0 . 0 0 1 ^ { 3 }$ </td><td> $0 . 1 5 2 \pm 0 . 0 0 2 ^ { 1 }$ </td><td> $\mathbf { 0 . 2 8 6 \pm 0 . 0 0 8 ^ { 2 } }$ </td><td> $\mathbf { 0 . 3 2 8 \pm 0 . 0 1 0 ^ { 1 } }$ </td></tr><tr><td> $\delta$ </td><td> $\mathrm { F T \_ 1 0 0 } _ { \mathrm { - } }$ </td><td>DN</td><td> $0 . 1 3 3 \pm 0 . 0 0 2 ^ { 3 }$ </td><td> $0 . 1 5 4 \pm 0 . 0 0 2 ^ { 1 }$ </td><td> $0 . 1 5 0 \pm 0 . 0 0 4 ^ { 1 }$ </td><td> $0 . 2 8 9 \pm 0 . 0 0 8 ^ { 2 }$ </td><td> $0 . 3 3 3 \pm 0 . 0 1 3 ^ { 1 }$ </td></tr><tr><td>JaccErr Baseline_100</td><td></td><td></td><td> $\overline { { 0 . 1 1 3 \pm 0 . 0 0 1 ^ { 3 } } }$ </td><td> $\overline { { 0 . 0 5 7 0 \pm 0 . 0 0 1 6 ^ { 1 } } }$ </td><td> $\overline { { 0 . 1 0 0 \pm 0 . 0 0 3 ^ { 1 } } }$ </td><td> $\overline { { 0 . 2 4 5 \pm 0 . 0 0 6 ^ { 2 } } }$ </td><td> $\overline { { 0 . 3 1 3 \pm 0 . 0 1 5 ^ { 1 } } }$ </td></tr><tr><td>JaccErr Baseline 200</td><td></td><td></td><td> $0 . 1 1 1 \pm 0 . 0 0 1 ^ { 3 }$ </td><td> $\mathbf { 0 . 0 5 4 3 \pm 0 . 0 0 0 9 } ^ { 1 }$ </td><td> $0 . 0 9 3 \pm 0 . 0 0 4 ^ { 1 }$ </td><td> $0 . 2 4 7 \pm 0 . 0 0 7 ^ { 2 }$ </td><td> $0 . 3 0 5 \pm 0 . 0 0 7 ^ { 1 }$ </td></tr><tr><td>JaccErr</td><td> $\mathrm { F T \_ 1 0 0 } _ { \cdot }$ </td><td>ND</td><td> $\mathbf { 0 . 1 0 5 \pm 0 . 0 0 2 ^ { 3 } }$ </td><td> $0 . 0 5 5 9 \pm 0 . 0 0 2 0 ^ { 3 }$ </td><td> $\mathbf { 0 . 0 9 0 \pm 0 . 0 0 6 } ^ { 1 }$ </td><td> $\mathbf { 0 . 2 3 6 \pm 0 . 0 0 6 ^ { 2 } }$ </td><td> $\mathbf { 0 . 2 7 6 \pm 0 . 0 1 0 ^ { 1 } }$ </td></tr><tr><td>JaccErr</td><td> $\mathrm { F T } _ { - } ^ { - } 1 0 0 _ { - } ^ { - }$ </td><td>DN</td><td> $0 . 1 0 9 \pm 0 . 0 0 1 ^ { 3 }$ </td><td> $0 . 0 5 4 4 \pm 0 . 0 0 1 6 ^ { 1 }$ </td><td> $0 . 0 9 3 \pm 0 . 0 0 7 ^ { 1 }$ </td><td> $0 . 2 4 5 \pm 0 . 0 0 7 ^ { 2 }$ </td><td> $0 . 2 9 9 \pm 0 . 0 2 2 ^ { 1 }$ </td></tr></table>

Fine-tuning outperforms all other approaches for $\gamma , \varLambda ,$ and $\Sigma ^ { + }$ in terms of Wasserstein score. For $K _ { S } ^ { 0 }$ and $n ,$ fine-tuning results are only slightly worse than the baseline.

The best noise-removal method is consistent between baseline and fine-tuned models for each particle. The only exception is $n ,$ where floor-based methods with and without recalculation perform best in diferent settings. This suggests that the model behaviour is consistent and that a once-chosen noise-removal method for a given particle should also work well for similar models.

Compared to Table 3, the full pipeline shows worse metric values, especially for $\varLambda , K _ { S } ^ { 0 } $ , and $\Sigma ^ { + }$ . This indicates that the auxiliary photon-count predictor is the main bottleneck, as it fails to learn the input-to-photon mapping well for these particles.

The other metrics also generally favour fine-tuning. Interestingly, FT\_100\_DN sometimes performs best, unlike in the previous NF-only analysis. The metric that remains most consistent between Tables 3 and 5 is Jaccard index error; the only significant change in the best-performing model is for the Λ case, which is not surprising given the much worse Wasserstein score observed there.

The most stable results are again obtained for $\gamma ,$ which is the least diverse particle type and also the most abundant in the data set. Here, FT\_100\_ND outperforms the rest. For Σ<sup>+</sup>, FT\_100\_ND is the best in 3 of the 4 metrics and second best in the remaining one, which also makes it stand out. For the remaining cases, the ranking is less straightforward. We therefore rank the results within each metric and combine them into a single averaged value. For $n ,$ Baseline\_200 and $\mathrm { F T } _ { - } 1 0 0 \_ \mathrm { D N }$ perform similarly, although Baseline\_200 is only slightly better in Jaccard index error, which suggests that the FT\_100\_DN can be considered to be better overall. For Λ, $\mathrm { F T } _ { - } 1 0 0 \_ \mathrm { N D }$ and $\mathrm { F T \_ 1 0 0 \_ D N }$ also have the same mean ranking, and the slight advantage of FT\_100\_DN becomes visible only when Table 3 is taken into account. For $K _ { S } ^ { 0 }$ , Baseline\_100 and FT\_100\_ND have the same ranking, but the fine-tuned model is preferable when Table 3 is considered as well.

Across the four better-sampled particle types $( \gamma , n , \ A , K _ { S } ^ { 0 } )$ and the four metrics, the fine-tuned $\mathrm { F T _ { - } } 1 0 0 \_ \mathrm { N D }$ model is best overall in most comparisons. The diference between results presented in Tables 3 and 5 suggests that the main bottleneck in this study is the prediction of the expected photon number, so further work should focus on that stage.

We also want to highlight that fine-tuning is more eficient than training the baseline longer on the full dataset. Thus, in cases where the fine-tuning improves the Baseline\_200 or where it behaves similarly, it makes more sense to follow the fine-tuning approach, which saves time and reduces energy consumption.

## 4.3 Ensemble model evaluation

We construct the final ensemble by selecting the best-performing model for each particle: $\mathrm { F T \_ 1 0 0 \_ N D \ f o r } \gamma , K _ { S } ^ { 0 } .$ and $\Sigma ^ { + } ; \mathrm { F T \_ 1 0 0 \_ D N }$ for n and $\varLambda ;$ and Baseline\_100 for the remaining particles (six models total). Table 6 compares the ensemble against the baselines across all metrics. The ensemble outperforms both baselines, validating the multi-model approach. Sample ensemble outputs are shown in Fig. 5.

Table 6: Final comparison of the ensemble model against baselines, computed over the full test dataset.
<table><tr><td>WS</td><td> $\overline { { \mathrm { M A E } _ { c o n d } } }$ </td><td>δ</td><td>JaccErr</td></tr><tr><td>Baseline_100  $\overline { { 1 . 7 0 \pm 0 . 0 2 } }$ </td><td> $\overline { { 6 . 5 7 \pm 0 . 0 4 } }$ </td><td> $\overline { { 0 . 1 7 6 \pm 0 . 0 0 1 } }$ </td><td> $\overline { { 0 . 1 2 7 \pm 0 . 0 0 0 } }$ </td></tr><tr><td>Baseline 200  $1 . 7 6 \pm 0 . 0 2$ </td><td> $6 . 4 3 \pm 0 . 0 3$ </td><td> $0 . 1 7 2 \pm 0 . 0 0 1$ </td><td> $0 . 1 2 5 \pm 0 . 0 0 1$ </td></tr><tr><td>Ensemble  ${ \bf 1 . 6 1 \pm 0 . 0 2 }$ </td><td></td><td>6.29 ± 0.02 0.163 ± 0.001</td><td> $\mathbf { 0 . 1 2 1 \pm 0 . 0 0 1 }$ </td></tr></table>

## 4.4 Time performance

MAF-based NF architectures are fast in evaluating the likelihood, which allows for reasonably short training, and slow during sampling. Our models require around 160.0 ms for response generation, compared to e.g. 0.023 ms needed for a GAN (NVIDIA A100 40 GB GPU, batch size 256) [20]. However, it is possible to train an Inverse Autoregressive Flow (IAF) student, which mimics the behaviour of the previously trained MAF teacher. This way, we can expect to obtain around 420 times faster models (0.38 ms per sample), as in [16].

![](images/208fcd55c781ff6852e57ac288806288fb8b77f56ce7fa963970ba5e2ab37dae.jpg)  
Fig. 5: Sample results from the ensemble model. Three samples per input vector illustrate run-to-run variability.

## 4.5 Understanding the model

Good performance alone is insuficient; models should also exhibit physically plausible reasoning. We use SHAP analysis to verify the ensemble’s feature dependencies for three quantities: total photon count, and shower centre x and y coordinates. For photon count, we explain the BNN predictions directly. For coordinates, we train surrogate BNNs on NF-generated data through knowledge distillation [10], as a direct NF explanation is too slow.

The analysis indicates that the photon count depends primarily on Energy and momentum in the z direction $P _ { z }$ , with higher values producing more photons, as expected physically. Because we removed the sign from $P _ { z }$ (indicating ZDC side, negligible here), its contribution is similar to Energy due to the energymomentum relation $E ^ { 2 } = P _ { x } ^ { 2 } + P _ { y } ^ { 2 } + P _ { z } ^ { 2 } + m a s s ^ { 2 }$ (in the dataset, the $P _ { z }$ value is orders of magnitude bigger than $\check { P } _ { x } , P _ { y }$ and mass, thus dominating the E value). Shower x and y coordinates depend mainly on $P _ { x } , P _ { y } ,$ , and Energy, as expected from particle trajectory physics. The influence of charge reflects magnetic field deflection near the detector.

## 5 Conclusions

We applied Normalizing Flows (NFs) to simulate ZDC neutron calorimeter responses, demonstrating substantial improvements through transfer learning, finetuning, and ensembling. Our ensemble generator achieves a Wasserstein distance of 1.61 ± 0.02, outperforming all baselines across all metrics while maintaining physically plausible feature dependencies verified via SHAP analysis. Importantly, fine-tuning saves time and reduces energy consumption compared to a longer training on the full dataset.

Standard ZDC evaluation metrics, such as Wasserstein distance and perpixel MAE, have known limitations. We therefore introduce conditional metrics: weighted channel MAE, dispersion ratio, and Jaccard co-activation error, which better capture input-output physical consistency across diverse particle types and response variability.

Key methodological advances include:

1. Two fine-tuning schemes using gradual unfreezing, with one of them showing the strongest overall performance.

2. Four post-processing methods for noise removal, where particle-specific optimization yields further gains.

3. An ensemble combining specialised models that validates the multi-model approach.

The analysis reveals that photon-count prediction remains the primary bottleneck, suggesting clear directions for future improvement. While current inference is slower than some other generative alternatives, IAF distillation ofers a path to a substantial speedup. We also highlight the possibility of utilising diferent approaches to ensemble modelling, e.g., with a meta-model designed to assign a simulation task to a particular generative surrogate.

This work provides a generalizable procedure for constructing a data-driven surrogate model of a complex numerical simulation, specifically aimed at predicting detector responses to particle parameters recorded at the LHC. The combination of normalizing flows, conditional fine-tuning, and physics-motivated evaluation establishes a strong foundation for production-ready fast simulation.

Acknowledgments. We would like to thank Sandro Wenzel, PhD from CERN for his support in this work. This work is co-financed and in part supported by the Ministry of Science and Higher Education (Agreements No. 2022/WK/01 and 2023/WK/07) by the program entitled “PMW” and by the Ministry funds assigned to AGH University of Krakow. We gratefully acknowledge Polish high-performance computing infrastructure PLGrid (HPC Center: ACK Cyfronet AGH) for providing computer facilities and support within computational grants no. PLG/2023/016410, PLG/2024/017264 and PLG/2025/018322.

Disclosure of Interests. The authors have no competing interests to declare that are relevant to the content of this article.

## References

1. Agostinelli, S., et al.: GEANT4 - A Simulation Toolkit. Nuclear Instruments and Methods in Physics Research 506, 250–303 (2003). https://doi.org/10.1016/S0168- 9002(03)01368-8

2. Bierlich, C., et al.: Towards a data-driven model of hadronization using normalizing flows. SciPost Phys. 17, 045 (2024). https://doi.org/10.21468/SciPostPhys.17.2.045

3. Deja, K., Dubiński, J., Nowak, P., Wenzel, S., Spurek, P., Trzcinski, T.: End-toend sinkhorn autoencoder with noise generator. IEEE Access 9, 7211–7219 (2021). https://doi.org/10.1109/ACCESS.2020.3048622

4. Dubiński, J., Deja, K., Wenzel, S., Rokita, P., Trzcinski, T.: Selectively increasing the diversity of gan-generated samples. In: Tanveer, M., Agarwal, S., Ozawa, S., Ekbal, A., Jatowt, A. (eds.) Neural Information Processing. pp. 260–270. Springer International Publishing, Cham (2023)

5. Dubiński, J., Deja, K., Wenzel, S., Rokita, P., Trzciński, T.: Machine learning methods for simulating particle response in the zero degree calorimeter at the alice experiment, cern (2023)

6. Durkan, C., Bekasovs, A., Murray, I., Papamakarios, G.: Neural spline flows. Advances in Neural Information Processing Systems, vol. 32, pp. 7511–7522. Neural Information Processing Systems Foundation, Inc (Dec 2019)

7. Gallio, M., Klempt, W., Leistam, L., De Groot, J., Schukraft, J.: ALICE Zero-Degree Calorimeter (ZDC): Technical Design Report. Technical design report. AL-ICE, CERN, Geneva (1999)

8. Gambardella, A., Baydin, A.G., Torr, P.H.: Transflow learning: Repurposing flow models without retraining. arXiv preprint arXiv:1911.13270 (2019)

9. Germain, M., Gregor, K., Murray, I., Larochelle, H.: Made: Masked autoencoder for distribution estimation (2015)

10. Hinton, G., Vinyals, O., Dean, J.: Distilling the knowledge in a neural network (2015)

11. Kita, M., Dubiński, J., Rokita, P., Deja, K.: Generative difusion models for fast simulations of particle collisions at cern (2024)

12. Krause, C., Shih, D.: Fast and accurate simulations of calorimeter showers with normalizing flows. Physical Review D 107(11) (Jun 2023). https://doi.org/10.1103/physrevd.107.113003

13. Krause, C., et al.: Calochallenge 2022: A community challenge for fast calorimeter simulation (2024)

14. Kubat, M., Matwin, S., et al.: Addressing the curse of imbalanced training sets: one-sided selection. In: Icml. vol. 97, p. 179. Citeseer (1997)

15. Lundberg, S.M., Lee, S.I.: A unified approach to interpreting model predictions. In: Advances in Neural Information Processing Systems. vol. 30. Curran Associates, Inc. (2017)

16. Majerz, E., Dzwinel, W., Kitowski, J.: Inverse Autoregressive Flows for Zero Degree Calorimeter fast simulation. In: 39th Annual Conference on Neural Information Processing Systems: Includes Machine Learning and the Physical Sciences (ML4PS) (12 2025)

17. Malnick, S., Avidan, S., Fried, O.: Taming normalizing flows. In: Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV). pp. 4644–4654 (January 2024)

18. Siahkoohi, A., Rizzuti, G., Witte, P.A., Herrmann, F.J.: Faster uncertainty quantification for inverse problems with conditional normalizing flows (2020)

19. Wojnar, M.: Even faster simulations with flow matching: A study of zero degree calorimeter responses. Computer Physics Communications 319, 109936 (2026). https://doi.org/https://doi.org/10.1016/j.cpc.2025.109936

20. Wojnar, M., Majerz, E., Dzwinel, W.: Fast simulation of the zero degree calorimeter responses with generative neural networks. Computing and Software for Big Science 9(1), 1 (Jan 2025). https://doi.org/10.1007/s41781-025-00130-x