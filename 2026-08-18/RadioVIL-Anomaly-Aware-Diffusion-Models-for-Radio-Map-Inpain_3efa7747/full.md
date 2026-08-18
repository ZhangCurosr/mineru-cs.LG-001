# RadioVIL: Anomaly-Aware Diffusion Models for Radio Map Inpainting and Zero-Shot Vehicle Localization

Ruixin Zhao<sup>∗</sup>, Xiucheng Wang<sup>†</sup>, Qiming Zhang<sup>‡</sup>, Nan Cheng<sup>†</sup>, Ruijin Sun<sup>†</sup>, and Conghao Zhou<sup>†</sup>

<sup>∗</sup>School of Computer Science and Technology, Xidian University, Xi’an, 710071, China

<sup>†</sup>State Key Laboratory of ISN and School of Telecommunications Engineering, Xidian University, Xi’an, 710071, China <sup>‡</sup>School of Artificial Intelligence, Xidian University, Xi’an, 710071, China

Email: {24179100115, xcwang 1, 23009200991}@stu.xidian.edu.cn, dr.nan.cheng@ieee.org, sunruijin@xidian.edu.cn, conghao.zhou@ieee.org

Abstract—High-precision radio map construction is essential for emerging 6G Integrated Sensing and Communication (ISAC) applications, including digital twins and intelligent transportation. However, existing deep learning methods predominantly treat this as a pure image completion task, resulting in over-smoothed reconstructions that fundamentally erase highfrequency scattering signatures of dynamic physical entities such as hidden vehicles. To overcome this, we propose RadioVIL, an efficient two-stage framework that reformulates joint radio map inpainting and zero-shot vehicle localization as a priorguided physical inverse problem. Specifically, we first train a Denoising Diffusion Probabilistic Model (DDPM) to capture the structural generative prior of the environment. During inference from highly sparse measurements, we employ a Diffusion-based Mediating Intermediate Layer Optimization (DMILO) algorithm. By optimizing an L<sub>1</sub>-regularized sparse deviation term, DMILO mathematically isolates vehicle scattering anomalies layer-bylayer without unfolding the entire denoising chain. Extensive experiments demonstrate that while conventional reconstruction baselines fail to detect hidden vehicles, and the zero-shot diffusion baseline achieves only limited detection ability due to forced semantic harmonization, RadioVIL preserves authentic physical textures, yielding the best LPIPS of 0.0587 in our evaluation. Uniquely, it unlocks accurate zero-shot vehicle localization directly from sparse radio maps, securing a 75.20% Recall and a 3.31-meter average error, paving a robust way for ISAC at the 6G edge.

Index Terms—6G, Radio map, Denoising Diffusion Probabilistic Models, DMILO, Sparse Measurements, Vehicle Localization

## I. INTRODUCTION

The evolution toward 6G Integrated Sensing and Communication (ISAC) requires high-fidelity, environment-aware electromagnetic representations. Radio maps (RMs) link environmental geometry and materials to spatial signal strength, serving as digital twins for applications such as autonomous driving, zero-measurement path planning, and dynamic vehicle localization. However, crowdsourced sensing is limited by deployment costs, physical occlusions, and restricted access to drivable road regions. The resulting sparse measurements make signal-field reconstruction and dynamic target detection highly ill-posed.

Existing methods, from regression-based RadioUNet [1] to adversarial RME-GAN [2], mainly treat RM construction as pixel-level completion. Pixel-wise objectives such as Mean Squared Error favor overly smooth reconstructions. Although effective for macroscopic path-loss trends, they erase highfrequency local variations and vehicle scattering footprints, hindering hidden-target sensing.

Recently, Denoising Diffusion Probabilistic Models (DDPMs) [3] have shown strong ability in capturing complex data manifolds. In wireless domains, diffusion architectures have been extended to dynamic map construction [4], Helmholtz equation-informed physical modeling [5], 3D environment synthesis [6], and accelerated inference trajectories [7]. Efficient architectures such as RadioMamba have also been introduced to address the computational burden of high-resolution feature extraction [8].

Despite these advances, solving the joint inverse problem under extreme sparsity remains challenging. For instance, zero-shot inpainting methods such as RePaint [9] achieve strong visual completion but fail in physical RM reconstruction. RePaint enforces semantic harmonization and blends masked regions with the background. Since hidden vehicles act as physical scatterers that contradict the smooth structural prior, RePaint suppresses these anomalies and effectively “washes out” vehicle signatures. Moreover, enforcing physical constraints across the entire denoising graph incurs high memory cost and may disrupt the generative manifold.

To address this bottleneck, we propose RadioVIL, an efficient two-stage framework that reformulates RM inpainting and zero-shot vehicle localization as an anomaly-aware, prior-guided physical inverse problem. Instead of end-to-end hallucination or rigid pixel replacement, RadioVIL dynamically injects hard physical constraints into the generative process. A pre-trained DDPM provides a smooth structural prior, while a layer-wise optimization strategy [10] isolates scattering anomalies without unfolding the full denoising chain. By introducing an L<sub>1</sub>-regularized sparse deviation term, RadioVIL absorbs high-frequency measurement residuals that contradict the generative prior, preserving authentic electromagnetic signatures of hidden targets. The main contributions are summarized as follows:

1) We introduce a two-stage framework integrating DDPMs [3] with Diffusion-based Mediating Intermediate Layer Optimization (DMILO) [10] to solve the severely ill-posed radio map inpainting and sensing problem.

2) We propose an anomaly-aware layer-wise optimization strategy that injects hard physical constraints. By using an L -regularized sparse deviation term, RadioVIL isolates high-frequency scattering anomalies and overcomes the semantic over-smoothing flaw of RePaint and traditional deep learning methods.

3) We design a zero-shot vehicle extraction mechanism based on spatial morphology and Connected Component Analysis (CCA), enabling hidden vehicle localization directly from isolated anomalies without paired labels or auxiliary detection networks.

## II. PRELIMINARIES AND SYSTEM MODEL

## A. Denoising Diffusion Probabilistic Models

Denoising Diffusion Probabilistic Models (DDPMs) [3] have recently emerged as state-of-the-art generative frameworks, demonstrating exceptional capabilities in data synthesis and wireless environment reconstruction [4], [11]. Given a complete ground-truth data distribution D, a DDPM progressively corrupts a clean sample $\mathbf { \delta } _ { \mathbf { x } _ { 0 } } \sim \mathcal { D }$ into a sequence of noisy latent variables $\{ \pmb { x } _ { t } \} _ { t = 1 } ^ { T }$ through a predefined variance schedule $\{ \beta _ { t } \} _ { t = 1 } ^ { T }$ . By defining $\alpha _ { t } = 1 - \beta _ { t }$ and $\begin{array} { r } { \bar { \alpha } _ { t } = \prod _ { s = 1 } ^ { t } \alpha _ { s } } \end{array}$ , the forward marginal distribution admits a closed-form Gaussian transition:

$$
\begin{array} { r } { q ( \pmb { x } _ { t } | \pmb { x } _ { 0 } ) = \mathcal { N } \left( \pmb { x } _ { t } ; \sqrt { \bar { \alpha } _ { t } } \pmb { x } _ { 0 } , ( 1 - \bar { \alpha } _ { t } ) \mathbf { I } \right) . } \end{array}\tag{1}
$$

To invert this forward diffusion corruption, a neural network $\epsilon _ { \theta } ( x _ { t } , t )$ , typically parameterized as a U-Net architecture, is optimized to predict the injected Gaussian noise at each timestep. The network is trained by minimizing the variational lower bound (VLB) objective:

$$
\mathcal { L } _ { p r i o r } ( \theta ) = \mathbb { E } _ { x _ { 0 } , \epsilon , t } \left[ \left| \left| \epsilon - \epsilon _ { \theta } ( \sqrt { \bar { \alpha } _ { t } } x _ { 0 } + \sqrt { 1 - \bar { \alpha } _ { t } } \epsilon , t ) \right| \right| _ { 2 } ^ { 2 } \right] _ { }\tag{2}
$$

Once converged offline, the frozen denoising network effectively encapsulates the structural prior of the physical environment, serving as a powerful generative dictionary that represents the physically plausible data manifold.

## B. System Model and Problem Formulation

Let the physical space be discretized into a two-dimensional grid of size $H \times W$ . We define the underlying true radio map, which contains complete environmental information including dynamic hidden vehicles, as $\mathbf { R } ^ { * } \in \mathbb { R } ^ { H \times W }$ . The physical environment is characterized by strict topological constraints. To formally represent the sensing capability, we define a binary physical mask matrix $\mathbf { M } \in \{ 0 , 1 \} ^ { H \times W }$ . According to realistic urban configurations, the mask is constructed piecewise:

$$
\mathbf { M } ( p ) = \left\{ \begin{array} { l l } { 1 , } & { p \in \Omega _ { b l d g } \cup \Omega _ { b s } \cup \Omega _ { e d g e } \cup \Omega _ { r a n d } } \\ { 0 , } & { p \in \Omega _ { r o a d } \cup \Omega _ { v e h } } \end{array} \right. ,\tag{3}
$$

where observable regions include static buildings $\Omega _ { b l d g }$ , base stations $\Omega _ { b s }$ , narrow road-edge observation bands $\Omega _ { e d g e } ,$ , and 5%–10% random free-space samples $\Omega _ { r a n d } .$ . Drivable roads and potential vehicle regions are masked out to establish a zero-knowledge zone for vehicle locations.

To account for hidden vehicles, we decompose the complete radio map into a smooth background and sparse vehicleinduced scattering footprints:

$$
\mathbf { R } ^ { * } = \mathbf { R } _ { b g } ^ { * } + \mathcal { K } ( \pmb { \nu } _ { v e h } ^ { * } ) ,\tag{4}
$$

where $\mathbf { R } _ { b g } ^ { * }$ denotes the static-environment background, $\nu _ { v e h } ^ { * }$ represents sparse vehicle-induced scattering sources, and $\kappa ( \cdot )$ is an effective scattering footprint operator. The sparse observation is then modeled as

$$
\pmb { y } = \mathbf { M } \odot \left( \mathbf { R } _ { b g } ^ { * } + \mathcal { K } ( \pmb { \nu } _ { v e h } ^ { * } ) \right) + \epsilon ,\tag{5}
$$

where $\epsilon \sim \mathcal { N } ( 0 , \sigma _ { n } ^ { 2 } )$ . Although vehicle pixels are masked, their scattering footprints still perturb observable road-edge, base-station, and random samples. Hidden vehicles are therefore inferred from residuals on $\{ p | \mathbf { M } ( p ) = 1 \}$ rather than direct observations, preserving the strict zero-knowledge assumption while retaining indirect physical observability.

Given y and the pre-trained diffusion prior, our objective is to recover R<sup>∗</sup> and extract hidden vehicle coordinates. Fig. 1 summarizes the task setting: the antenna and static structural layout specify the propagation environment, the mask defines the accessible sensing domain, the measurement provides sparse radio observations, and the color radio map is the dense target field to be reconstructed.

## III. PROPOSED RADIOVIL FRAMEWORK

Fig. 2 presents RadioVIL as a constrained bilevel pipeline combining a structural prior with anomaly-aware measurements.

## A. Prior Adaptation and Layer-wise Deterministic Mapping

Offline, we train a DDPM backbone $f _ { \boldsymbol { \theta } ^ { \ast } } \left( \boldsymbol { x } _ { t } , t , \mathbf { H } \right)$ , conditioned on the structural environment map H, to model macroscopic radio propagation. By explicitly encoding building layouts and boundaries, this frozen prior effectively captures large-scale distance-dependent attenuation and shadowing. However, it lacks the capability to natively represent unpredictable, highly sparse scattering anomalies caused by dynamic vehicles. Guiding the reverse inference $\mathcal { G } ( \cdot )$ directly with sparse measurements y requires unfolding the full computational graph, incurring prohibitive memory costs and disrupting the trajectory. To avoid this, we adopt a layerwise DMILO strategy [10]. At each reverse timestep t, we introduce a trainable latent variable z to locally balance the generative prior and measurement constraints before proceeding.

![](images/7e91f7801104318ccd8f39271cdb54998acd333d7ad45d6983a4d78f89c95a88.jpg)  
(a) Antenna

![](images/a121c9e2ca16d21431f2387e0d2fabed514e4f978ff29748aca07c60a07166e7.jpg)  
(b) Mask

![](images/a2b9bfc34b5cd7b7beffbcd1f15e4999ea86d8f4b9bde59c1498bc5e9c3f2c54.jpg)  
(c) Measurement

![](images/bb5b121f29248dd2ee17f539330a3306a3e0733d005735f8e30c666d73660800.jpg)  
(d) Color Radio Map  
Fig. 1: Input modalities and target output of the proposed RadioVIL framework. Subfigures show the antenna or structural layout, the physical mask defining observable and hidden regions, the degraded sparse measurement, and the complete color radio map used as the reconstruction target.

## B. Anomaly-Aware Inversion Operator

Sparse measurements y contain high-frequency residuals induced by vehicle scattering, which are difficult to represent using the smooth DDPM prior alone. To decouple these two factors, we introduce a sparse deviation variable $\pmb { \nu } \in \mathbb { R } ^ { H \times W }$ At timestep $t ,$ the predicted measurement is

$$
\hat { \pmb { y } } = \mathbf { M } \odot \left( f _ { \theta ^ { * } } ( z , t , \mathbf { H } ) + \mathcal { K } ( \pmb { \nu } ) \right) ,\tag{6}
$$

where $f _ { \theta ^ { * } } ( z , t , \mathbf { H } )$ models the smooth background and $\kappa ( \nu )$ describes the scattering footprint induced by sparse hidden anomalies. This leads to the layer-wise objective

$$
\mathcal { I } _ { t } ( z , \pmb { \nu } ) = \frac { 1 } { 2 \sigma _ { n } ^ { 2 } } \left. \pmb { y } - \mathbf { M } \odot \left( f _ { \theta ^ { * } } ( z , t , \mathbf { H } ) + K ( \pmb { \nu } ) \right) \right. _ { F } ^ { 2 } + \lambda \lVert \pmb { \nu } \rVert _ { 1 } .\tag{7}
$$

Here, z adapts the smooth generative manifold to observed large-scale propagation, while ν explains sparse residuals that remain inconsistent with this prior. Even when vehicle pixels are unobserved, ν is inferable because $\kappa ( \nu )$ affects the observed boundary samples. The $L _ { 1 }$ penalty suppresses arbitrary residual fitting and encourages localized vehicleinduced anomalies. In this sense, ν does not simply hallucinate missing pixels; instead, it serves as a sparse inverse variable constrained by both the measured boundary evidence and the smooth DDPM manifold.

## C. Iterative Inversion and Zero-Shot Extraction

The procedure iteratively optimizes (7). At each t, we initialize $\textit { \textbf { z } }  \textit { \textbf { x } } _ { t }$ and $\nu  0$ , followed by K gradient

descent iterations to update the variables simultaneously:

$$
z \gets z - \eta _ { z } \nabla _ { z } \mathcal { T } _ { t } ( z , \nu ) ,
$$

$$
\pmb { \nu }  \pmb { \nu } - \eta _ { \nu } \nabla _ { \pmb { \nu } } \mathcal { I } _ { t } ( \pmb { z } , \pmb { \nu } ) .\tag{8}
$$

(9)

Upon convergence of the inner loop, the optimized state $z ^ { * }$ is passed to the standard sampling step to compute ${ \mathbf { } } x _ { t - 1 } .$ , and the defect $\nu ^ { * }$ is accumulated as $\hat { \pmb { \nu } } = \hat { \pmb { \nu } } + \pmb { \nu } ^ { * }$

After the diffusion loop concludes, we obtain the highfidelity background radio map $\mathbf { R } _ { e s t } = \pmb { x } _ { 0 }$ and the aggregated anomaly map νˆ. A threshold τ generates a binary activation mask $\mathbf { B } _ { v e h }$

$$
\mathbf { B } _ { v e h } ( p ) = \left\{ \begin{array} { l l } { 1 , } & { \mathrm { i f ~ } | \hat { \pmb { \nu } } ( p ) | > \tau } \\ { 0 , } & { \mathrm { o t h e r w i s e } } \end{array} \right. .\tag{10}
$$

By performing Connected Component Analysis (CCA) on $\mathbf { B } _ { v e h }$ , we effectively filter out isolated noise pixels. Each robustly isolated cluster $\mathcal { C } _ { k }$ corresponds to a physical vehicle. The continuous coordinates $( x _ { k } , y _ { k } )$ are derived by computing the center of mass based on the anomaly intensity:

$$
x _ { k } = \frac { \sum _ { \mathcal { C } _ { k } } x \cdot \vert \hat { \nu } ( x , y ) \vert } { \sum _ { \mathcal { C } _ { k } } \vert \hat { \nu } ( x , y ) \vert } , \quad y _ { k } = \frac { \sum _ { \mathcal { C } _ { k } } y \cdot \vert \hat { \nu } ( x , y ) \vert } { \sum _ { \mathcal { C } _ { k } } \vert \hat { \nu } ( x , y ) \vert } .\tag{11}
$$

This mechanism enables highly accurate zero-shot vehicle localization directly from the physically isolated anomalies, eliminating the need for paired labels or auxiliary detection networks.

## IV. EXPERIMENTAL RESULTS AND ANALYSIS

## A. Experimental Setup

We evaluate RadioVIL on the RadioMapSeer benchmark dataset [1], [12]. To avoid target leakage, the conditioning map contains only static environmental information, including buildings, roads, greenbelts, and base-station-related cues; vehicle locations are excluded from the model input and used only for evaluation. To mimic sparse crowdsourced sensing, we retain static infrastructure, base-station coordinates, a single-pixel observation band around greenbelts, and 5%– 10% randomly sampled free-space pixels. Vehicle regions are fully masked with $\mathbf { M } ( p ) = 0$ , forcing the method to infer hidden vehicles from boundary scattering residuals. This setting ensures that the localization task is not solved by direct visual access to vehicle regions, but by exploiting the indirect electromagnetic perturbations left in the observable measurements.

We compare RadioVIL against three representative deep learning paradigms:

• RadioUNet [1]: A state-of-the-art fully supervised regression model optimized for pixel-wise Mean Squared Error (MSE).

• RME-GAN [2]: A Generative Adversarial Network designed to hallucinate missing radio map data through adversarial training.

• RePaint [9]: A powerful zero-shot diffusion-based inpainting approach that conditions the unconditional generative process by forcibly replacing known pixels at each reverse timestep, aiming to harmoniously blend missing regions with the observed background.

![](images/681f5e5d48926348e86bb67f6b4ab470c616deba741d316c9b2320c3eaae01d7.jpg)  
Fig. 2: Overview of RadioVIL. The outer diffusion loop governs the sampling trajectory, while the inner DMILO loop minimizes J<sub>t</sub>. Through L<sub>1</sub>-regularized optimization of ν, RadioVIL decouples the smooth background manifold $f _ { \theta ^ { \ast } } ( z )$ from sparse scattering anomalies for target localization.

TABLE I: Quantitative Comparison of Radio Map Inpainting and Zero-Shot Vehicle Localization Performance
<table><tr><td rowspan="2">Method</td><td colspan="3">Inpainting Quality</td><td colspan="4">Vehicle Localization Performance</td></tr><tr><td>PSNR (dB) (1)</td><td>SSIM (↑)</td><td>LPIPS (4)</td><td>Recall (%) (1)</td><td>F1-Score (↑)</td><td>Mean Dist. (m) (↓)</td><td>P90 Dist. (m) (↓)</td></tr><tr><td>RadioUNet [1]</td><td>25.13</td><td>0.889</td><td>0.2107</td><td>0.00</td><td>0.00</td><td>N/A</td><td>N/A</td></tr><tr><td>RME-GAN [2]</td><td>25.50</td><td>0.9433</td><td>0.0889</td><td>0.00</td><td>0.00</td><td>N/A</td><td>N/A</td></tr><tr><td>RePaint [9]</td><td>23.35</td><td>0.8099</td><td>0.1010</td><td>11.44</td><td>0.1765</td><td>3.89</td><td>6.41</td></tr><tr><td>RadioVIL (Ours)</td><td>24.30</td><td>0.8731</td><td>0.0587</td><td>75.20</td><td>0.6799</td><td>3.31</td><td>7.40</td></tr><tr><td>Relative Gain vs. RePaint</td><td>+4.07%</td><td>+7.80%</td><td>41.88%↓</td><td>+557.34%</td><td>+285.21%</td><td>14.91%↓</td><td></td></tr></table>

Note: Relative gain is computed against RePaint; ↓ denotes relative reduction. RePaint’s P90 is computed on only 11.44% successful detections, so it relative P90 gain is not reported.

The radio map inpainting quality is quantitatively assessed using Peak Signal-to-Noise Ratio (PSNR), Structural Similarity Index (SSIM), and Learned Perceptual Image Patch Similarity (LPIPS). For hidden vehicle localization, a predicted centroid is designated as a True Positive (TP) if its Euclidean distance to a ground-truth centroid falls within a specific tolerance limit (set to 10 meters in our evaluation). We employ Recall to measure the proportion of successfully identified true vehicles out of all actual targets, and the F1-Score as a comprehensive harmonic mean to evaluate precision and robustness. Furthermore, we assess spatial accuracy using the Mean Distance, calculating the average Euclidean coordinate error in meters, and introduce the 90th Percentile Distance (P90) to strictly bound the maximum spatial error for 90% of the successful detections. It is important to note that these distance metrics are computed exclusively over the true positives.

During implementation, we employ the Adam optimizer for the layer-wise gradient descent. The learning rate for the mediating latent state z is set to $2 \times 1 0 ^ { - 2 }$ , while the learning rate for the sparse deviation variable ν is configured at $1 \times 1 0 ^ { - 3 }$ . The number of inner iterations is defined as $K = 2 0 0$ , and the outer diffusion loop operates for 5 reverse timesteps. To forcefully enforce physical constraints, the sparsity regularization weight is strictly fixed at $\lambda = 0 . 1$ and the anomaly activation threshold for extracting zero-shot vehicle masks is empirically established at $\tau = 0 . 5$

## B. Radio Map Inpainting Performance

The quantitative results reveal an important phenomenon in metric evaluation. Although traditional regression and GANbased baselines (RadioUNet, RME-GAN) achieve higher PSNRs (25.13 dB and 25.50 dB), PSNR tends to penalize

![](images/2fe0f56f74503a0d1ef7448c16f4a92863bdba71645b41b58820167a718da60d.jpg)  
RME-GAN  
RadioVIL (Ours)

Fig. 3: Visual comparison of radio map inpainting and vehicle localization across different scenes from the RadioMapSeer dataset. From left to right: RadioUNet, RME-GAN, RePaint, RadioVIL, and ground truth.

high-frequency deviations, encouraging overly smoothed reconstructions that erase vehicle scattering footprints.

Compared with the zero-shot diffusion baseline RePaint, RadioVIL shows clear advantages, achieving higher PSNR (24.30 dB vs. 23.35 dB), SSIM, and the best LPIPS score of 0.0587. Since RePaint is designed for visual semantic harmonization, it treats sharp physical scattering boundaries as visual noise and blends critical target signatures into the background. In contrast, RadioVIL separates the smooth environmental prior from scattering deviations through layerwise DMILO optimization, preserving physical textures essential for sensing.

## C. Vehicle Localization Performance

The handling of high-frequency anomalies directly determines vehicle localization performance. As shown in Table I, traditional methods fail catastrophically: both RadioUNet [1] and RME-GAN [2] detect no hidden vehicles, yielding F1- scores of 0. RePaint [9] performs only marginally better, with 11.44% Recall and an F1-Score of 0.1765.

Although RePaint reports a lower 90th Percentile Distance (6.41m vs. 7.40m), this value is computed over only 11.44% successful detections. Therefore, P90 should be interpreted jointly with Recall and F1-Score, where RadioVIL bounds the localization error for a much larger portion (75.20%) of hidden vehicles. As shown in Fig. 3, RePaint suppresses structural anomalies and blends masked regions into free space. Since it is not anomaly-aware, it treats high-frequency electromagnetic scattering footprints as noise, causing frequent missed detections.

In contrast, RadioVIL achieves robust localization under sparse and strictly masked observations. By applying zeroshot extraction to the isolated anomaly map νˆ, our framework obtains an F1-Score of 0.6799, recalls 75.20% of hidden vehicles, and achieves an average localization error of 3.31 meters.

TABLE II: Impact of Building Density on Performance
<table><tr><td>Buildings</td><td>Samples PSNR</td><td></td><td>SSIM</td><td>F1</td></tr><tr><td>&lt; 15 (Simple)</td><td>560</td><td>24.55</td><td>0.878</td><td>0.743</td></tr><tr><td>15-30</td><td>5440</td><td>24.12</td><td>0.868</td><td>0.675</td></tr><tr><td>&gt; 30 (Complex)</td><td>2000</td><td>24.56</td><td>0.874</td><td>0.681</td></tr></table>

![](images/f1fc9300534e42643b3d98a163ecaed8d60bf8957947fda16198477002aad932.jpg)  
Fig. 4: Performance stability vs. distance from base station.

These results demonstrate that RadioVIL bridges visual image completion and anomaly-aware physical environment sensing.

## D. Robustness to Environmental Factors

We investigate the robustness of RadioVIL against environmental complexity. As summarized in Table II, performance shows no clear monotonic degradation as the environment becomes more cluttered, indicating that RadioVIL does not rely on overly simplified propagation patterns or sparse building layouts. Even in dense urban scenes with more than 30 buildings, where blockage, shadowing, and multipath effects are more severe, the vehicle detection F1-Score decreases by only 8.3%. This suggests that RadioVIL can still exploit global structural priors and boundary scattering residuals under complex propagation conditions.

Furthermore, Fig. 4 shows that both radio map inpainting and vehicle localization remain relatively stable across transmission ranges from 50m to 800m. This indicates that RadioVIL relies on global environmental structural priors rather than local signal intensity alone, enabling reliable performance across the coverage area.

## V. CONCLUSION

This paper presented RadioVIL, an anomaly-aware diffusion framework for joint radio map inpainting and zero-shot hidden vehicle localization from sparse measurements. By combining a frozen DDPM prior with layer-wise DMILO optimization, RadioVIL reformulates radio map reconstruction as a prior-guided physical inverse problem rather than a conventional pixel-level completion task. The frozen diffusion model captures the smooth environmental propagation background, while the $L _ { 1 }$ -regularized sparse deviation variable isolates vehicle-induced scattering anomalies that cannot be explained by the smooth prior. This design enables RadioVIL to preserve high-frequency electromagnetic signatures that are often suppressed by regression-based, GAN-based, or semantic diffusion inpainting baselines.

The isolated anomaly map further enables CCA-based vehicle localization without paired vehicle labels or auxiliary detection networks, supporting zero-shot sensing under strict masking conditions. Overall, the results demonstrate that sparse wireless measurements, when guided by structural generative priors and anomaly-aware inverse optimization, can provide a promising sensing mechanism for future 6G ISAC systems.

## ACKNOWLEDGMENT

This work was supported by the National Key Research and Development Program of China (2024YFB907500), and the National Natural Science Foundation of China (62571402).

## REFERENCES

[1] R. Levie, C¸ . Yapar, G. Kutyniok, and G. Caire, “RadioUNet: Fast radio map estimation with convolutional neural networks,” IEEE Trans. Wireless Commun., vol. 20, no. 6, pp. 4001–4015, 2021.

[2] S. Zhang, A. Wijesinghe, and Z. Ding, “RME-GAN: A learning framework for radio map estimation based on conditional generative adversarial network,” IEEE Internet Things J., vol. 10, no. 20, pp. 18 016–18 027, 2023.

[3] J. Ho, A. Jain, and P. Abbeel, “Denoising diffusion probabilistic models,” Advances in neural information processing systems, vol. 33, pp. 6840– 6851, 2020.

[4] X. Wang, K. Tao, N. Cheng, Z. Yin, Z. Li, Y. Zhang, and X. Shen, “RadioDiff: An effective generative diffusion model for sampling free dynamic radio map construction,” IEEE Trans. Cognit. Commun. Networking, Early access, pp. 1–13, 2024.

[5] X. Wang, Q. Zhang, N. Cheng, R. Sun, Z. Li, S. Cui, and X. Shen, “Radiodiff-k2: Helmholtz equation informed generative diffusion model for multi-path aware radio map construction,” arXiv preprint arXiv:2504.15623, 2025.

[6] X. Wang, Q. Zhang, N. Cheng, J. Chen, Z. Zhang, Z. Li, S. Cui, and X. Shen, “Radiodiff-3d: A 3dx 3d radio map dataset and generative diffusion based benchmark for 6g environment-aware communication,” IEEE Transactions on Network Science and Engineering, 2025.

[7] X. Wang, P. Zheng, H. Jia, N. Cheng, R. Sun, C. Zhou, and X. Shen, “Radiodiff-flux: Efficient radio map construction via generative denoise diffusion model trajectory midpoint reuse,” IEEE Transactions on Cognitive Communications and Networking, vol. 12, pp. 4882–4895, 2025.

[8] H. Jia, N. Cheng, X. Wang, C. Zhou, R. Sun, and X. Shen, “Radiomamba: Breaking the accuracy-efficiency trade-off in radio map construction via a hybrid mamba-unet,” IEEE Transactions on Network Science and Engineering, 2025.

[9] A. Lugmayr, M. Danelljan, A. Romero, F. Yu, R. Timofte, and L. V. Gool, “Repaint: Inpainting using denoising diffusion probabilistic models,” in 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022, pp. 11 451–11 461. [Online]. Available: https://openaccess.thecvf.com/content/CVPR2022/ html/Lugmayr RePaint Inpainting Using Denoising Diffusion Probabilistic Models CVPR 2022 paper.html

[10] Y. Zheng, W. Li, and Z. Liu, “Integrating Intermediate Layer Optimization and Projected Gradient Descent for Solving Inverse Problems with Diffusion Models,” 2025.

[11] X. Wang, Z. Fang, N. Cheng, R. Sun, Z. Li et al., “Radiodiff-inverse: Diffusion enhanced bayesian inverse estimation for isac radio map construction,” arXiv preprint arXiv:2504.14298, 2025.

[12] C¸ . Yapar, F. Jaensch, R. Levie, G. Kutyniok, and G. Caire, “The first pathloss radio map prediction challenge,” in Proceedings of the 2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2023, pp. 1–2.