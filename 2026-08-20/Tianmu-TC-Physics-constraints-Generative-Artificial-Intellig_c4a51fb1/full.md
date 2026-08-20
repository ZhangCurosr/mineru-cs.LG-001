# Tianmu-TC: Physics-constraints Generative Artificial Intelligence for Global Tropical Cyclone Forecasting

Shiqi Zhang<sup>1†</sup>, Pan Mu<sup>1†</sup>, Cheng Huang<sup>1</sup>, Hanting Yan<sup>1</sup>, Yuchao Zhu<sup>1</sup>,

Jinglin Zhang<sup>2</sup>, Shengyong Chen<sup>3∗</sup>, Shoujuan Shu<sup>4∗</sup>, Cong Bai<sup>1∗</sup>

<sup>1</sup>College of Computer Science and Technology, Zhejiang University of Technology, Hangzhou & 310023, China.

<sup>2</sup>School of Control Science and Engineering, Shandong University, Jinan & 250061, China.

<sup>3</sup>School of Computer Science and Engineering, Tianjin University of Technology, Tianjin & 300384, China.

<sup>4</sup>School of Earth Sciences, Zhejiang University, Hangzhou & 310058, China.

<sup>∗</sup>Corresponding author. Email: csy@tjut.edu.cn, sjshu@zju.edu.cn, congbai@zjut.edu.cn

<sup>†</sup>These authors contributed equally to this work.

Tropical cyclones (TCs) pose severe risks from strong winds and heavy rainfall. However, forecasting their track and intensity remains challenging due to chaotic atmosphere and the rapid amplification ofinitial condition errors, leading to growing forecast uncertainty. While numerical weather prediction (NWP) and deep learning models have made progress, they remain computationally demanding and often fail under complex meteorological scenarios. Here, we present Tianmu-TC, a physics-constraints generative framework for global TC forecasting. Trained on Western North Pacific data, Tianmu-TC leverages physics-constraints to generate controllable outputs with reduced uncertainty thus improving forecast reliability. Experiments show Tianmu-TC outperforms deterministic and ensemble

meteorological artificial intelligence models and authoritative NWP systems such as ECMWF in global ocean basins, with significantly lower computational cost. We further show Tianmu-TC performs well in challenging scenarios such as data sparsity, anomaly tracks, rapid intensification and weakening. These findings suggest physics-constraints generative AI ofers a promising approach for reliable, eficient global TC forecasting.

## Introduction

Tropical cyclones (TCs) are complex weather systems that often bring strong winds and heavy rainfall, leading to natural disasters that cause significant damage to infrastructure and pose serious threats to human life and property (1). Therefore, accurate forecasting is crucial for efective prevention of disasters caused by TCs. The track of a TC tells us where it is currently located, while its intensity represents its strength and the potential severity of the damage it may cause. Forecasting these two attributes are crucial for determining the level of preparedness required. Specifically, the TC track is a combination of locations, including longitude and latitude of the TC center. TC intensity can represented by diferent attributes, with pressure and wind speed being the most representative. Pressure refers to the lowest atmospheric pressure (hPa) at the TC center, and wind speed to the maximum sustained wind speed (m/s) over two minutes near the TC center. These attributes are closely interconnected in physics (2, 3).

However, forecasting tropical cyclones inherently involves uncertainty, making accurate predictions a persistent challenge. The atmosphere is a chaotic system, meaning that even small errors in the initial conditions can amplify over time—a phenomenon widely known as the butterfly effect—ultimately degrading forecast accuracy and increasing uncertainty. This uncertainty becomes especially pronounced as the forecast horizon extends, with both track and intensity predictions becoming less reliable. For instance, the predicted location of the TC center may deviate by several hundred kilometers, while intensity forecasts often fail to capture abrupt changes such as rapid intensification or rapid weakening. In practice, this leads to wide forecast spreads and rapidly growing uncertainty.

To address these challenges, authoritative Numerical Weather Prediction (NWP) models (4– 6) adopt ensemble forecasting strategies, which simulate multiple scenarios by perturbing the initial conditions (7). This approach has been widely adopted by oficial agencies such as the European Centre for Medium-Range Weather Forecasts (ECMWF) (8), the National Hurricane Center (OFCL) (6), and the China Central Meteorological Observatory (CMO) (9). While ensemble methods help mitigate uncertainty, they require substantial computational resources to simulate global atmospheric processes (10), resulting in prohibitively long training and inference times. This makes them less suitable for real-time regional applications where rapid response is essential. As the number of uncertainties being considered grows, the required computational resources increase significantly. Thus, it is imperative to explore more eficient approaches for TC forecasting.

Recently, the advancement of generative models has provided a novel perspective for addressing the challenges of uncertainty (11–15) with substantially lower computational cost. By introducing controlled noise or perturbations, they technically circumvent the impact of initial errors and their iterative amplification, enabling the generation of future TC tracks and intensities. Furthermore, their ability to generate a large number of diverse members, similar to ensemble forecasting, makes them highly suitable for predicting future tropical cyclone tracks and intensities.

However, the diversity reflected in these ensemble members—while beneficial for representing uncertainty—can lead to overly dispersed outputs, including multiple plausible but divergent tracks, thereby compromising the precision required for disaster prevention and emergency response. To address this, we introduce physics-constraints into the generative process. These constraints regulate the stochasticity of the model, enabling a more balanced trade-of between diversity and accuracy. The constraint conditions are formally defined and illustrated in Figure 1.

Firstly, one common constraint in prediction tasks is the historical values of the target attribute. In TC forecasting, this typically includes the TC’s past track and intensity, which we refer to as the historical information constraint.

Secondly, the generation and evolution of a TC are influenced by large-scale atmospheric environment. To account for this, we introduce the atmospheric environment constraint, which incorporate the major large-scale circulation systems influencing TC activity include the South Asia High (SAH) in the upper troposphere, the WP Subtropical High (WPSH), and the monsoon trough (16–18).

![](images/c23cd56088f475ea6e694cb4978e0fd5298c8eed4d9495c267c2444d499c0378.jpg)

B  
![](images/e51611a81fb3d6beaff59b475246624058e97899666b404deecb6ae7ad2a1428.jpg)

![](images/827355129f39115617bd338d26e9b09dea8921d6f6f21b6f5dd14a3e545b668e.jpg)

D  
![](images/710fb6cb08218fe58cf67364f815d3dfa6e37ddca7a38d658a6aa76bc0dab1e3.jpg)  
Figure 1: Physics-constraints for TC forecasting. (A) Overall schematic: meteorological factors influencing the development of TCs. (B) Proposed physics-constraints govern the generation process and achieve a lower uncertainty score with the smallest prediction area. (C) The data composition of atmospheric environment constraint. The large-scale wind field consists of the zonal � and meridional wind � components from 200 hPa to 850 hPa. High-pressure systems are composed of the South Asian High (SAH, �200) and the Western Pacific Subtropical High (WPSH, �500), where � denotes geopotential height. (D) TC tilt metrics: calculated from vorticity data at three pressure levels. See details in Materials and Methods.

Finally, beyond environmental forcing, TC attributes are also impacted by internal structural changes, which are often overlooked in prior work. Therefore, we designed the internal structure variation constraint, explicitly incorporating structural changes into the prediction process to control the diversity of generated outputs.

The three constraints above are collectively referred to as the Physics-driven Meteorological Factor Constraints, namely, Physics-constraints. Based on these, we introduce Tianmu-TC, a physics-constrained generative difusion framework for global tropical cyclone forecasting, as shown in Figure 2. By integrating Physics-constraints, Tianmu-TC transforms track and intensity prediction into a conditional generative process, where the initially chaotic generation results progressively become clearer and more accurate. Our approach ofers a new paradigm for eficient and robust global TC forecasting, highlighting the practicality and potential of generative modeling as a forecasting approach.

![](images/e6b75dd60d5eeba1d8990f275d3f2d143b40ccbc0213aa04b0c4be26d9177046.jpg)  
Figure 2: Framework of Tianmu-TC. Tianmu-TC consists of three key components: (1) Constraint Encoder for preliminary constraint encoding, (2) Cross-Modal Fusion module to establish correlations between constraints, and (3) Constraint guiding denoising module to perform the final denoising process. The �<sub>�</sub> ∼ �<sub>0</sub> represents the TC prediction results under diferent difusion steps �. To facilitate clear visualization, only the predicted track is displayed here, whereas wind speed and pressure predictions are simultaneously generated. From $y _ { K }$ to $y _ { 0 } .$ , the blue dashed line gradually converges from noise to near ground truth. The single-step denoising process from $y _ { k }$ to $y _ { k - 1 }$ is implemented by the Physics-constraints Denoising Block, which operates under the guidance of physics- constraints.

Previous eforts to apply artificial intelligence to TC forecasting have largely focused on deterministic prediction by learning patterns from historical data, while often lacking explicit modeling of forecast uncertainty, as exemplified by classical spatiotemporal sequence models such as RNNs (19, 20), GRUs (21), Bi-GRU (22), LSTMs (23), and ConvLSTMs (24). More recent work incorporates multi-modal inputs—including historical TC attributes and gridded reanalysis variables—to better represent the surrounding atmospheric environment (25–32). In parallel, largescale meteorological foundation models (33) such as Pangu (34) and Fengwu (35), together with recent probabilistic or ensemble AI weather models such as GenCast (11) and NeuralGCM (36), have achieved impressive skill by directly forecasting global ERA5 reanalysis fields. Although these models can indirectly support TC forecasting through downstream tracking, they are not specifically optimized for TC-focused tasks. This often leads to underestimation of intensity, less compact ensemble distributions, and suboptimal performance in rare or extreme scenarios. Additionally, their dependence on globally scaled datasets and large model sizes incurs substantial training and inference costs, making them less practical for real-time or resource-constrained forecasting settings.

In summary, existing approaches struggle to balance forecasting accuracy, computational eficiency, and uncertainty reduction, and often underperform under complex or rare meteorological conditions, such as rapid changes in tropical cyclone intensity or track. Our goal is to enhance prediction reliability by efectively reducing uncertainty, without incurring additional computational cost.

## Results

Tianmu-TC takes 48 hours of TC historical data as input and generates forecasts for the next 24 or 72 hours. The input includes three core components: ① historical TC attributes from the International Best Track Archive for Climate Stewardship (IBTrACS (37)), ② ERA5-derived environmental fields representing large-scale atmospheric conditions, and ③ TC tilt metrics extracted from vorticity fields to capture internal structural dynamics. In this study, the IBTrACS best track records are adopted as the ground truth in all evaluations, following standard practice in deep learning-based forecasting.

To rigorously assess generalizability, we evaluate Tianmu-TC on a global test set spanning all ocean basins from 2017 to 2023. This design ensures coverage of a wide range of TC scenarios, including varying intensities, tracks, and environmental conditions. Notably, to avoid excessive training time and computational cost, the model is trained exclusively on Western North Pacific (WP) TCs from 1950 to 2016. This highlights the model’s ability to generalize beyond its training domain, demonstrating robust performance across unseen basins despite being trained regionally.

$$
c _ { 1 } { : }
$$

![](images/e70b1f6f7f24626f3f2783bec34ab6c3758c4968aadc3b550df0b3e056e1a6e7.jpg)

Ba  
![](images/8bbcdf67abcdb27964506137466d86f06fa0392c2bfe1843e09f60ae214a9de2.jpg)

Ca  
![](images/b487a60e9f0d9d527df00ca206739b356844f9af10f5c2ecc048d4102ee2cdfa.jpg)

![](images/c473dcd5cfdf401a1a317ecf19188c017f7d49d2e73455daab9fe927c93d0144.jpg)

Bb  
![](images/12ae16e42d8ee27705cb47f7308ceeb661a621ea6b66aa4315b560450b10375c.jpg)

Cb  
![](images/96f2af85fce6e99abd58455381a3a1dcfe9b2853e00b7d0abf61fc3c76cf804d.jpg)

![](images/130ead9c77b44f05a14e60b72f17759104faffc284e5c91923beeb19df5f8c24.jpg)

Bc  
![](images/1f56bf081b6bec9bed087e908278b802b85a312ecf87895685188f62a608d3e1.jpg)

Cc  
![](images/aae738146739ae87bcad8a9733b880bf83b3043dbebaf53de8a3b4d0b0cf1728.jpg)  
Figure 3: Results under diferent constraints. (A) Generation results of Typhoon Krosa at 00:00 UTC on August 16, 2019, under five constraint settings with decreasing difusion steps � (from 100 to 0). Shaded regions represent uncertainty. Applying all constraints $( C _ { 1 } , C _ { 2 }$ , and $C _ { 3 } ;$ rightmost column) results in the most concentrated predictions with the lowest uncertainty. (B) For the Typhoon Krosa at 00:00 UTC on August 16, 2019, corresponding to $( \mathrm { A } ) ,$ , the mean and variance of the predictions are visualized as spread distributions. Smaller variance leads to a steeper curve, indicating lower prediction uncertainty. (C) Average prediction errors across 146 TCs from 2017 to 2023. The rows correspond to (a) track, (b) pressure, and (c) wind speed.

The time resolution is $^ 6$ hours. For fair comparison, following $\mathrm { T C N } _ { M } \left( 3 I \right)$ , the sampling number was set to $^ { 6 , }$ meaning that the model generates $^ 6$ possible tendencies. Further details on the model architecture, training setup, and datasets are provided in the Materials and Methods section.

Tianmu-TC reduces forecast uncertainty through the incorporation ofphysics-driven constraints, which consists of the historical information constraint $( C _ { 1 } )$ , the atmospheric environment constraint $( C _ { 2 } )$ , and the TC internal structure variation constraint $( C _ { 3 } )$ . We compare its performance against authoritative numerical weather prediction systems, deep learning–based methods, and deterministic and ensemble large-scale meteorological foundation models across global test regions. Additionally, we examine the model’s efectiveness under challenging scenarios such as data sparsity, anomaly tracks, landfall events, rapid intensification and rapid weakening.

The results demonstrate that Tianmu-TC achieves a favorable balance between predictive accuracy and controlled output diversity, while ofering strong robustness and fast inference. Together with its compact and accurate multi-trend forecasts, these findings suggest that generative AI, when properly constrained by physical principles, constitutes a reliable and efective approach for advancing global TC forecasting.

## Physics-constraints reduce prediction uncertainty

A central challenge in TC forecasting is how to efectively reduce predictive uncertainty. Uncertainty arises naturally from the chaotic nature of the atmosphere and can be further amplified by model imperfections, making it essential to evaluate whether generated ensembles can provide not only accurate predictions but also reliable distributions. To address this, we investigate the role of incorporating physics-constraints into the generative process. We evaluate their efectiveness by comparing generation results under five settings: unconstrained generation (none of the constraints), generation with partial constraints $( C _ { 1 } , \ : C _ { 1 } \& C _ { 2 } , \ : C _ { 1 } \& C _ { 3 } )$ , and generation with full physics-constraints $( C _ { 1 } \& C _ { 2 } \& C _ { 3 } )$ . During reverse difusion, the forecasts evolve from broad, highly uncertain distributions at difusion step $k = 1 0 0$ to concentrated predictions near the ground truth at $k = 0$

These evaluations leads to three key observations:

1. Full physics-constraints yield the lowest uncertainty with the most concentrated prediction distributions. When all three constraints are applied, Tianmu-TC generates ensembles that converge tightly around the ground truth for both track and intensity. As shown in Figure 3Aa–Ac, the final outcomes (� = 0) under full constraints form compact clusters, whereas the unconstrained case produces substantially larger deviations—track errors are about six times and intensity errors about three times larger. This aspect is further illustrated by the probability density curves in Figure 3Ba–Bc, where the fully constrained results exhibit the steepest probability density curves, with data points highly concentrated, indicating concentrated forecasts and substantially reduced uncertainty. The Figure S4 presents the uncertainty score corresponding to each predicted region, calculated as the MSE loss between the predictions and the ground truth. This further supports the aforementioned conclusion.

2. Physics-constraints enable the forecast distribution to better align with the ground-truth distri bution, thereby improving both local accuracy and global consistency. As shown in Figure 3Ba–Bc, the probability density curves of Tianmu-TC exhibit peak values and shapes closest to those of the ground truth, indicating that predictions are not only accurate at specific points but also capture the broader structural trends of tropical cyclones. This distributional alignment highlights the model’s ability to reproduce realistic ensemble behavior, extending beyond reduced spread to better capturing the statistical consistency of forecasts with the ground truth.

3. Physics-constraints reduce average prediction errors. As shown in Figure 3Ca–Cc, incorporating all three constraints lowers the average errors across track, pressure, and intensity, confirming the higher accuracy of Tianmu-TC compared with alternative settings. Specifically for intensity prediction, the $C _ { 3 }$ (TC internal structure variation constraint) enhances forecasts by encoding vertical structural changes within the TC, consistent with the well-established relationship between cyclone structure and intensity evolution (38). This agreement reinforces the physical interpretability of the model.

The efectiveness of physics-constraints in reducing prediction uncertainty is consistently reflected in visualization patterns, distribution convergence, and average error metrics, thereby resulting in improved forecasting performance.

![](images/4f3ceff8addadf4b1cf876d3575a66768b3f32a3f7f7d5a4b9752436ad058b4f.jpg)  
Figure 4: Comparison of average forecast errors with large-scale meteorological models and the authoritative NWP system during the test period from 2017 to 2023. The testing regions cover six ocean basins: (A) Eastern North Pacific (EP), (B) North Atlantic (NA), (C) North Indian (NI), (D) South Indian (SI), (E) South Pacific (SP), and (F) Western North Pacific (WP). (G) Comparison of computational eficiency metrics with large-scale meteorological models.

## Generalizability on global ocean areas

Tropical cyclones occur across multiple ocean basins, making cross-regional generalization a critical requirement for forecasting models. Although Tianmu-TC is trained exclusively on Western North Pacific data, we evaluate its performance on global TC datasets spanning six basins to assess its transferability to real-world applications. As shown in Figure 4, Tianmu-TC outperforms large meteorological models such as Pangu (34) and Fengwu (35) in most regions, while achieving comparable overall performance. In addition, Figure 4G shows that Tianmu-TC achieves an eightfold reduction in training time, a tenfold reduction in inference time, and a threefold reduction in model size. Two main conclusions emerge:

1. Eficiency and performance superiority. Despite its lower training costs, smaller model size, and faster inference compared with large meteorological models, Tianmu-TC achieves comparable track prediction and significantly better intensity prediction. Notably, it significantly outperforms ECMWF (the ECMWF global model (8)) in the EP, NI, and SI basins (Figure 4A,C,D). In the NA, SP, and WP basins (Figure 4B,E,F), it delivers superior intensity prediction performance across all metrics and competitive track accuracy within 18 hours. Moreover, Tianmu-TC runs on a single NVIDIA A6000 GPU rather than supercomputing infrastructure, highlighting that deep learning can surpass operational NWP while requiring substantially fewer computational resources.

![](images/9943a9b98e98b9dc3e500e43d560b47c16f08e2e9547c390568293af2f83079b.jpg)  
Figure 5: Visualization of track predictions for representative TCs across six ocean basins. Each basin presents forecasts for a specific TC at three consecutive initialization times: (A) EP (Jimena), (B) NA (Rene), (C) NI (Ockhi), (D) SI (Enawo), (E) SP (Gita), and (F) WP (Sonca).

2. Cross-regional scalability. Although trained only on WP data, Tianmu-TC generalizes effectively to the other five basins (Figure 4A–E), capturing distinct TC dynamics in both hemispheres. This demonstrates robust adaptability to diverse environments and strong potential for global TC forecasting.

For broader comparison, we also benchmarked Tianmu-TC against authoritative forecasts provided by local meteorological agencies in Figure 4 and Figure S2. The oficial forecast issued by the National Hurricane Center (OFCL) (6) was assessed in the EP and NA basins, while the NWP system of the China Central Meteorological Observatory (CMO) (9) was evaluated in the WP basin. As shown in Figure 4A, Tianmu-TC significantly outperforms OFCL in the EP basin. Notably, Tianmu-TC surpasses for the first time the authoritative NWP model operated by the CMO across all metrics in the WP basin (Figure 4F).

Table 1: Results on special TC scenarios on global area from 2017 to 2021. Samples is the number of cases. Lower values indicate better performance. “Anomaly” consists of circular movement, turning on the ocean, and V-shaped tracks. “RI” denotes Rapid Intensification, and “RW” denotes Rapid Weakening of TCs.
<table><tr><td rowspan="2"></td><td rowspan="2">Category Scenario (Unit) Methods Samples</td><td rowspan="2"></td><td rowspan="2"></td><td colspan="3">Forecast Horizon</td></tr><tr><td>+6h</td><td>+12h +18h</td><td>+24h</td></tr><tr><td rowspan="8">Track</td><td rowspan="3">Anomaly (km)</td><td>Pangu</td><td>582</td><td>44.13</td><td>55.90 72.75</td><td>104.26</td></tr><tr><td>Fengwu</td><td>582</td><td>44.19</td><td>49.33 65.89</td><td>95.62</td></tr><tr><td>ECMWF</td><td>582</td><td>40.32 50.26</td><td>61.82</td><td>78.11</td></tr><tr><td rowspan="3">Landfall (km)</td><td>Ours</td><td>582</td><td>16.90</td><td>17.28 34.74</td><td>70.38</td></tr><tr><td>Pangu</td><td>304</td><td>47.86</td><td>62.09 87.55</td><td>125.50</td></tr><tr><td>Fengwu</td><td>304</td><td>43.9652.80</td><td>65.19</td><td>83.17</td></tr><tr><td>Ours</td><td>ECMWF</td><td>304</td><td>35.71</td><td>47.2457.08</td><td>76.26</td></tr><tr><td rowspan="5">RI (m/s) Intensity</td><td>Pangu</td><td>304 231</td><td>16.98 22.89</td><td>20.90 36.61</td><td>66.26</td></tr><tr><td>Fengwu</td><td>231</td><td></td><td>31.68 41.30</td><td>49.32 46.17</td></tr><tr><td>ECMWF</td><td></td><td>17.41</td><td>21.89 29.85 38.88</td><td>28.78</td></tr><tr><td>Ours</td><td>231 231</td><td>1.35</td><td>18.39 24.32 1.29</td><td>7.59</td></tr><tr><td>Pangu</td><td>320</td><td>28.69 22.05</td><td>4.41 16.07</td><td>12.01</td></tr><tr><td rowspan="3">RW (m/s)</td><td>Fengwu</td><td>320</td><td></td><td>27.87 21.24 14.94</td><td></td><td>10.49</td></tr><tr><td>ECMWF</td><td>320</td><td></td><td>13.7810.73</td><td>8.64</td><td>26.40</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>Ours</td><td>320</td><td>2.69</td><td>1.63</td><td>3.79</td><td>5.74</td></tr></table>

Most existing studies focus on the Western North Pacific, where tropical cyclones occur most frequently. Therefore, we further compared Tianmu-TC with deep learning–based approaches in the WP basin (Figure S3). These include single-task baselines using only historical track or intensity information (DLM (27) and TCIF-fusion (39)) and multi-task models (SGAN (40), GBRNN (20), MMSTN (28), $\mathrm { T C N } _ { M }$ (31), TC-Difuser (32) and TAM-CL (41)). Tianmu-TC consistently surpasses all of these methods across evaluation metrics, highlighting its advantages in both track and intensity prediction.

![](images/21c3c4bf22648ffc575274c95724f0d81d5d5e89bf6f21612b441aba4cbf2d46.jpg)  
Figure 6: Visualization of complex prediction scenarios. (A) Track prediction, (B) wind speed, and (C) pressure. For (A), four challenging track cases are illustrated: (Aa) circular track, (Ab) post-landfall prediction, (Ac) turning on the ocean, and (Ad) V-shaped movement. (Ba)–(Bb): rapid intensification; (Bc)–(Bd): rapid weakening. Rapid intensification is defined as a tropical cyclone’s wind speed increasing by more than 30 knots (15.43 m/s) within 24 hours, whereas rapid weakening refers to the opposite process. (C) displays the corresponding pressure prediction results for (B).

In addition to aggregated performance metrics, Figure 5 presents qualitative visualizations of track predictions across the six global ocean basins. For each basin, we selected a representative tropical cyclone and visualized its predicted tracks at three consecutive forecast timestamps. This enables the comparison of temporal continuity and accuracy of predictions among diferent models.

Tianmu-TC shows consistent alignment with ground truth tracks over time, capturing not only general movement direction but also subtle tracks turns, demonstrating its reliability and temporal coherence in global-scale forecasting.

Overall, these experiments underscore the practical potential of Tianmu-TC: it not only reduces training costs but also demonstrates exceptional competitiveness against both operational forecasts and state-of-the-art deep learning approaches.

## Comparison with ensemble meteorological AI model: GenCast & NeuralGCM

To compare with existing multi-trend forecasting methods, GenCast and NeuralGCM were further selected as representative global machine-learning weather forecasting models for supplementary evaluation. These models were chosen because they provide both deterministic and ensemble forecasts, making them suitable references for assessing the efectiveness and competitiveness of Tianmu-TC in multi-trend tropical cyclone forecasting. To ensure reproducibility, the publicly archived gridded forecast data for the year 2020 were used, and tropical cyclone centers and intensity diagnostics were derived directly from the original forecast fields. Details of the data sources, selected forecast variables, tracking rules, and intensity diagnosis are provided in Supplementary Methods (Sections A.2 and A.3).

This comparison is conducted only for the year 2020 because publicly released GenCast and NeuralGCM forecasts are only provided for this year, and this avoids potential temporal overlap with the training period of GenCast. Since these forecasts are provided at 12 h intervals, the evaluation is performed at the shared forecast horizons of +12 h and +24 h. In addition, because MSLP is not available in the original NeuralGCM configuration used here, only track prediction results are reported for NeuralGCM.

Table 2 shows the deterministic forecast comparison. Tianmu-TC clearly outperforms NeuralGCM in track prediction. This gap may be partly associated with the relatively coarse (0.7<sup>◦</sup>) spatial resolution of the public deterministic NeuralGCM forecasts, which limits the precision of grid-based TC center localization. In contrast, Tianmu-TC is trained with high-precision best-track coordinates and is specifically optimized for short-term TC track prediction. Compared with Gen Cast, Tianmu-TC achieves lower errors for the +12 h track, the overall track, and all intensity-related metrics, while its +24 h track error is only slightly higher. These results indicate that Tianmu-TC provides more accurate short-term tropical cyclone intensity estimates and comparable track prediction skill relative to GenCast. This comparison is also informative from the perspective of computational cost. As a global medium-range ensemble forecasting system, GenCast is built at a substantially larger scale: the original GenCast paper reports slightly more than 3.5 days of pretrain ing and under 1.5 days of 0.25° fine-tuning on 32 TPUv5 instances, while a single 15-day global forecast takes approximately 8 minutes on a Cloud TPUv5 device. Under a simple linear scaling by forecast length, this corresponds to approximately 32 s for a 24 h forecast. In contrast, Tianmu-TC is specifically designed for short-term multi-attribute tropical cyclone forecasting, requires only 1.23 s for prediction, and can be trained on a single A6000 GPU in approximately 2 days. These results suggest that, with substantially lower computational cost, a tropical-cyclone-specific model still has clear advantages for short-term tropical cyclone track and intensity forecasting.

Table 2: Comparison with deterministic NeuralGCM and GenCast forecasts. Results are reported on test samples for the year 2020 at the overlapping forecast horizons of +12 h and +24 h. Track error, pressure error, and wind speed error are measured in km, hPa, and m/s, respectively. Lower values indicate better performance. In “Inference time”, s denotes seconds.
<table><tr><td rowspan="2">Model</td><td rowspan="2">Samples</td><td colspan="3">Track Error (km)</td><td colspan="3">Pressure Error (hPa)</td><td colspan="3">Wind Speed Error (m/s)</td><td rowspan="2">Training</td><td rowspan="2">Inference time</td></tr><tr><td>+12h</td><td>+24h</td><td>Overall</td><td>+12h</td><td>+24h</td><td>Overall</td><td>+12h</td><td>1+24h</td><td>Overall time</td></tr><tr><td>NeuralGCM</td><td>684</td><td></td><td>67.52 83.28</td><td>75.40</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>3 weeks</td><td>11.9 s</td></tr><tr><td>GenCast</td><td>684</td><td>43.25</td><td>562.00</td><td>52.63</td><td>10.71</td><td>10.75</td><td>10.73</td><td>10.10</td><td>9.93</td><td>10.02</td><td>5 days</td><td>32 s</td></tr><tr><td>Tianmu-TC</td><td>684</td><td>17.52</td><td>66.37</td><td>41.95</td><td>0.39</td><td>1.46</td><td>0.92</td><td>0.72</td><td>2.11</td><td>1.41</td><td>2 days</td><td>1.23 s</td></tr></table>

For multi-trend prediction, a formulation analogous to ensemble forecasting in numerical weather prediction, Table 3 evaluates 50 sampled trends from four complementary perspectives. Mean Error and Std. Error denote the mean and standard deviation of the errors across the 50 ensemble members or sampled trends, respectively. Spread measures the dispersion of the 50 forecasts and is used as an uncertainty-related indicator of the forecast distribution. Specifically, for track prediction, Spread is computed as the 90th-percentile radius of the distances between the 50 ensemble-predicted TC positions and the ensemble center, while for pressure and wind speed it is computed from the dispersion of the 50 predicted scalar values. Best-of-50 evaluates the error of the best member among the 50 forecasts, reflecting whether the forecast distribution contains a

Table 3: Comparison with ensemble NeuralGCM and GenCast forecasts. Results are reported on test samples for the year 2020 at the overlapping forecast horizons of +12 h and +24 h. The overall value is computed over the combined +12 h and +24 h samples. NeuralGCM-ENS, GenCast-ENS and Tianmu-TC-ENS denote the ensemble versions of NeuralGCM, GenCast and Tianmu-TC, respectively. For a fair comparison, all ensemble-based methods are evaluated using 50 members.
<table><tr><td rowspan="2">Task</td><td rowspan="2">Model</td><td rowspan="2">Samples</td><td colspan="3">Mean Error</td><td colspan="3">Std. Error</td><td colspan="2">Spread</td><td colspan="3">Best-of-50</td></tr><tr><td>+12h</td><td>+24h</td><td>Overall +12h</td><td></td><td>+24h</td><td>Overall +12h</td><td></td><td>+24h Overall</td><td>+12h</td><td>+24h</td><td>Overall</td></tr><tr><td rowspan="3">Track (km)</td><td>NeuralGCM-ENS 521/505</td><td></td><td></td><td>91.75126.87</td><td></td><td>109.04 31.8857.18</td><td></td><td>44.33</td><td>95.02 145.47</td><td></td><td>119.85</td><td>62.36 67.16</td><td>64.72</td></tr><tr><td>GenCast-ENS</td><td>521/505 92.28175.74 133.36 51.91 104.05</td><td></td><td></td><td></td><td></td><td></td><td>77.57</td><td></td><td></td><td>96.40189.52142.24 24.76 39.86</td><td></td><td>32.19</td></tr><tr><td>Tianmu-TC-ENS</td><td>521/505</td><td>28.40 109.86</td><td></td><td>68.49</td><td>8.76</td><td>34.16</td><td>21.26</td><td>22.33 85.29</td><td>53.32</td><td>10.42</td><td>40.94</td><td>25.44</td></tr><tr><td rowspan="3">Pressure (hPa)</td><td>NeuralGCM-ENS</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1 1</td><td>1</td></tr><tr><td>GenCast-ENS</td><td>521/505</td><td>9.72</td><td>10.05</td><td>9.88</td><td>1.30</td><td>1.92</td><td>1.60</td><td>1.34 2.05</td><td>1.69</td><td>6.88</td><td>6.27</td><td>6.58</td></tr><tr><td>Tianmu-TC-ENS</td><td>521/505</td><td>1.66</td><td>4.90</td><td>3.26</td><td>0.59</td><td>1.72</td><td>1.14 0.67</td><td>1.94</td><td>1.29</td><td>0.58</td><td>1.72</td><td>1.14</td></tr><tr><td rowspan="3"></td><td>NeuralGCM-ENS</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1 1</td><td></td><td>1 1</td><td>1</td><td>1</td></tr><tr><td>Wind Speed (m/s) GenCast-ENS</td><td>521/505</td><td>9.34</td><td>9.28</td><td>9.31</td><td>1.14</td><td>1.52</td><td>1.33</td><td>1.18</td><td>1.60</td><td>1.39 6.72</td><td>6.15</td><td>6.44</td></tr><tr><td>Tianmu-TC-ENS</td><td>521/505</td><td>1.01</td><td>3.16</td><td>2.07</td><td>0.36</td><td>1.06</td><td>0.70</td><td>0.42 1.21</td><td>0.81</td><td>0.34</td><td>1.22</td><td>0.77</td></tr></table>

highly accurate candidate.

First, Tianmu-TC-ENS improves the accuracy and stability of multi-trend prediction. For track prediction, Tianmu-TC-ENS reduces the overall Mean Error from 109.04 km and 133.36 km to 68.49 km compared with NeuralGCM-ENS and GenCast-ENS, respectively. The overall Std. Error is also reduced from 44.33 km and 77.57 km to 21.26 km, indicating that the sampled trends are not only closer to the ground truth on average, but also more stable across diferent members. For intensity prediction, Tianmu-TC-ENS similarly reduces the overall Mean Error from 9.88 hPa to 3.26 hPa for pressure and from 9.31 m/s to 2.07 m/s for wind speed, further confirming its advantage in multi-attribute TC prediction.

Second, Tianmu-TC-ENS produces a more compact forecast distribution while preserving accurate candidates. For track prediction, the overall Spread is reduced from 119.85 km and 142.24 km to 53.32 km, showing that Tianmu-TC-ENS substantially suppresses unnecessary spatial dispersion among the 50 trends. For pressure and wind speed, Tianmu-TC-ENS also reduces the overall Spread from 1.69 hPa to 1.29 hPa and from 1.39 m/s to 0.81 m/s, respectively. Meanwhile, the Best-of-50 results show that this compactness does not come from a simple collapse of the forecast distribution: Tianmu-TC-ENS achieves the best overall Best-of-50 errors for track, pressure, and wind speed prediction, although GenCast-ENS is slightly better for the +24 h track Best-of-50 case. Overall, these results provide quantitative evidence that the physically constrained multi-trend formulation of Tianmu-TC-ENS can improve the accuracy of sampled TC evolution trends while reducing unnecessary predictive dispersion. This supports the central objective of Tianmu-TC-ENS,

![](images/94685d7b9a33228421b5023c71138da19d361e563fac51bf44f5a629789d3538.jpg)  
Figure 7: Lifecycle visualization of multi-trend track forecasts for Typhoon ATSANI (WP, 2020). The main panel shows the rolling 12-h forecast path over the TC lifecycle, where each method is represented by the mean track of 50 ensemble members (i.e., multiple possible tracks), and the black curve denotes the best-track positions. The numbered panels show the spatial distributions of the 50 ensemble-predicted TC positions for representative cases selected from the main panel, covering early, middle, and late lifecycle stages as well as unusual looping-track predictions. In each numbered panel, circles indicate the spatial dispersion of the forecast members, and the values �, �, and � denote the corresponding dispersion radii of NeuralGCM-ENS, GenCast-ENS, and Tianmu-TC-ENS, respectively. The black star denotes the best-track position. The selected cases reveal diferent performance regimes, including accurate and biased situations.

namely generating physically consistent and uncertainty-compact multi-trend forecasts for TC track and intensity prediction.

Figure 7 further evaluates the lifecycle behavior of multi-trend track forecasts for Typhoon AT-SANI. Rather than focusing on a single initialization time, the figure examines forecasts throughout the entire TC lifecycle. ATSANI is selected because it provides diverse and challenging forecast scenarios, including unusual looping-track predictions, a pronounced track deflection near southern Taiwan, and subsequent evolution close to coastal and mountainous regions. These scenarios enable forecast stability to be evaluated under varying environmental and topographic conditions. Specifically, each method is represented by the mean track of 50 ensemble members, allowing a direct and fair comparison of lifecycle-scale track evolution, the dispersion of 50 ensemblepredicted TC positions, and the ability of the forecast distribution to cover the best-track position. As shown in the predicted positions panels of Figure 7, the values �, �, and � denote the dispersion radii of NeuralGCM-ENS, GenCast-ENS, and Tianmu-TC-ENS, respectively, with smaller values indicating lower spatial spread among the 50 forecasts. Tianmu-TC-ENS produces a more compact multi-trend forecast distribution, indicating lower forecast uncertainty while still covering the best-track position in most selected cases. In cases①–⑤, the orange Tianmu-TC-ENS predicted positions are concentrated within a much smaller circle than the blue NeuralGCM-ENS and green GenCast-ENS predicted positions, and the best-track position is generally located inside or close to the Tianmu-TC-ENS distribution. For example, Tianmu-TC-ENS shows notably smaller dispersion in cases①, ②, and ④, where the orange predicted positions remain tightly clustered around the best-track position. Meanwhile, NeuralGCM-ENS and GenCast-ENS usually exhibit larger circles, indicating stronger ensemble spread. This larger spread can also be beneficial, as it enables the global ensemble forecasts to cover the best-track position in several cases, although at the cost of higher uncertainty. These results suggest that Tianmu-TC-ENS provides a more concentrated multi-trend forecast while maintaining useful best-track coverage in most stages of the ATSANI lifecycle.

The selected cases also reveal several limitations. In case⑥, Tianmu-TC-ENS shows the largest positional bias among the three methods, whereas GenCast-ENS places its members closer to the best-track position. This error may be related to the relatively fast storm displacement in the preceding stages, which increases the dificulty of short-term track extrapolation. In case⑦, all three methods exhibit large errors, while NeuralGCM-ENS and GenCast-ENS show particularly large dispersion. This is likely associated with the late-stage land interaction and the rapidly changing storm environment near landfall. In addition, although GenCast-ENS usually benefits from its larger spread, it fails to cover the best-track position in case③. These results indicate that Tianmu-TC-ENS can substantially reduce excessive ensemble dispersion and provide stable multi-trend forecasts in most situations. However, achieving both low dispersion and reliable best-track coverage remains an important direction for future improvement.

## Robust forecasting of abrupt track changes and RI/RW events

To evaluate the performance of Tianmu-TC under challenging conditions, we examined its predictions for abrupt track changes and rapid intensity changes, which remain particularly dificult to forecast (42). As shown in Table 1, across abrupt track changes (denoted as anomaly in the table), landfall, rapid intensification (RI), and rapid weakening (RW) cases, Tianmu-TC consistently achieves the lowest forecast errors compared with the large meteorological models and operational NWP models (Pangu, Fengwu, ECMWF), demonstrating superior robustness under complex conditions. Anomaly includes circular movement, turning on the ocean, and V-shaped tracks; landfall refers to samples in which landfall occurs during at least half of the forecast period (24 hours). For track anomalies, the errors of Tianmu-TC are reduced by up to a factor of six relative to Pangu and by a factor of three relative to Fengwu. In landfall prediction, it substantially improves accuracy over ECMWF and achieves more than a twofold error reduction compared with Pangu. For intensity variation, Tianmu-TC reduces RI errors to less than one-tenth of those from operational NWP models and also achieves the best performance under RW cases. Since ECMWF dataset is limited, we additionally report comparisons on larger samples (aligned with Tianmu-TC, Pangu, Fengwu, see Table S1). The performance trends remain consistent, further validating the robustness of Tianmu-TC.

Besides, the corresponding visualization results are shown in Figure 6. For tracks, four representative scenarios were analyzed: circular movement, post-landfall prediction, turning on the ocean, and V-shaped tracks. For intensity, we focused on cases of rapid intensification and weakening. It can be observed that Tianmu-TC efectively captures the dynamics and transitions in these scenarios, providing reliable track forecasts even under irregular movements and maintaining high accuracy in predicting rapid intensity changes.

These results confirm that the proposed physics-constrained generative framework is capable of handling the most dificult TC scenarios, where both track and intensity variations remain highly uncertain in conventional approaches.

## Accurate real-time prediction under data sparsity

Given that the ERA5 reanalysis data used to encode the atmospheric environment constraint are not available in real time, we examined Tianmu-TC’s performance in cases where ERA5 data were missing during testing (see Table 4). In such cases, critical environmental inputs—including wind fields (�, �) and geopotential height (�) representing high-pressure systems—were unavailable.

By comparing Rows 9 and 12 in Table 4, we find that the absence of ERA5 data during inference has negligible impact on prediction accuracy. Notably, Tianmu-TC maintains comparable performance using only the TC historical attributes from the previous 24 hours.

This finding highlights the model’s robustness under data sparsity, where only partial inputs are available, and confirms its practical potential as a real-time, end-to-end TC forecasting method, even in the absence of comprehensive environmental data.

## Intensity prediction is dependent on track

A core challenge in tropical cyclone forecasting is the simultaneous prediction of both track and intensity. Traditionally, these two tasks have been modeled independently, reflecting the fact that they are governed by diferent dominant physical mechanisms: track prediction is mainly influenced by the steering flow, such as the position and strength of the subtropical high (43), while intensity prediction depends more heavily on factors like sea surface temperature, vertical wind shear, and ocean-atmosphere interactions (44, 45).

However, growing evidence suggests a strong coupling between track and intensity. In our proposed approach, both attributes are predicted jointly in an end-to-end framework, allowing the model to learn potential cross-task dependencies. As shown in Figure 8A, this joint prediction leads to superior performance compared to separate forecasting. Furthermore, the saliency analysis in

Table 4: Testing results under missing input conditions. The last row corresponds to the complete model (Tianmu-TC). $X _ { h i s } , X _ { E R A 5 } , X _ { t i l t }$ refer to TC historical attribute values, ERA5 data, and TC tilt values, respectively. +6ℎ to +24ℎ indicate prediction horizons in hours. Lower values represent better performance.
<table><tr><td rowspan="2">Row Real-time</td><td rowspan="2"></td><td colspan="2">Data</td><td rowspan="2"></td><td colspan="3">Track Error (km)</td><td rowspan="2">Pressure Error (hPa)</td><td rowspan="2">Wind Speed Error (m/s)</td><td rowspan="2">+6h+12h +18h +24h+6h +12h +18h 1+24h</td></tr><tr><td> $X _ { h i s }$ </td><td>XERA5</td><td> $X _ { t i l t }$ </td><td>+6h +12h</td></tr><tr><td>1</td><td>yes</td><td>1</td><td>1</td><td>1</td><td></td><td>114.88 228.64 331.53</td><td>+18h +24h 428.40 2.78</td><td>4.93 6.04</td><td>6.941.57 2.843.72</td><td>4.50</td></tr><tr><td>2</td><td>yes</td><td>6h</td><td>1</td><td>1</td><td>57.30</td><td></td><td>104.16 146.70 191.52 2.203.08</td><td></td><td>3.624.151.34 1.93 32.26</td><td>2.58</td></tr><tr><td>3</td><td>yes</td><td>12h</td><td>1</td><td>1</td><td>34.71</td><td>57.52</td><td>83.13</td><td>116.12 1.31 1.94 </td><td>1.91 2.62 0.62 1.25 1.17</td><td>1.54</td></tr><tr><td>4</td><td>yes</td><td>18h</td><td>1</td><td>1</td><td>26.43</td><td>39.97</td><td>61.88</td><td>93.991.10 2.03</td><td>1.97 2.67 0.59 1.13 1.09</td><td>1.47</td></tr><tr><td>5</td><td>yes</td><td>24h</td><td>1</td><td>1</td><td>20.97</td><td>30.21</td><td>46.45 77.19</td><td>0.96 1.60 </td><td>1.61 2.33 0.50 0.86 0.86</td><td>1.25</td></tr><tr><td>6</td><td>yes</td><td>30h</td><td>1</td><td>1</td><td>19.12</td><td>26.48</td><td>43.10 74.86</td><td></td><td>1.61 2.28 0.500.83 0.82</td><td>1.23</td></tr><tr><td>7</td><td>yes</td><td>36h</td><td>1</td><td>1</td><td>18.21</td><td>23.60</td><td>39.81 71.61</td><td>0.991.61 </td><td>5 2.23 0.49 0.81 0.80</td><td>1.21</td></tr><tr><td>8</td><td>yes</td><td>42h</td><td>1</td><td>1</td><td>17.99</td><td>22.77</td><td>38.70</td><td>0.961.54 1.55 70.16 0.96 1.53 </td><td>1.562.24 0.49 0.81 0.81</td><td>1.20</td></tr><tr><td>9</td><td>yes</td><td>48h</td><td>1</td><td>1</td><td>17.80</td><td>21.89</td><td>37.51 69.03</td><td>0.96 1.52</td><td>1.55 2.23 0.49 0.80 0.80</td><td>1.20</td></tr><tr><td>10</td><td>no</td><td>48h</td><td>√</td><td>1</td><td>17.80</td><td>21.89</td><td>37.51 69.03</td><td>0.96 1.52 1.55</td><td>5 2.23 0.49 0.80 0.80</td><td>1.20</td></tr><tr><td>11</td><td>no</td><td>48h</td><td>1</td><td>√</td><td>17.80</td><td>21.91</td><td>37.55 69.08</td><td>0.96 1.52 1.55</td><td>2.23 0.49 0.80</td><td>0.80 1.20</td></tr><tr><td>12</td><td>no</td><td>48h</td><td>√</td><td>√</td><td>17.80</td><td>21.91</td><td>37.55 69.08</td><td>0.96 1.52 1.55</td><td>2.23 0.49 0.80</td><td>0.80 1.20</td></tr></table>

Figure 8B highlights how intensity forecasts are sensitive to track-related features, providing further evidence for the interdependence between these two TC attributes.

A key conclusion is that while TC track and intensity are physically interconnected, their dependencies are asymmetric. Track prediction appears to be largely independent of intensity, whereas intensity prediction is strongly dependent on the track. As illustrated in Figure 8, this is primarily because the environmental field is the dominant driver of TC intensity changes. As a TC moves along its track, it encounters diferent environmental conditions (e.g., ocean heat content, wind shear), which in turn influence its structure and intensity. Thus, accurate track forecasts provide crucial context for predicting intensity evolution, reinforcing the necessity of jointly modeling both attributes.

![](images/aead46e3d588d4bf6b2e7bece3471fbc55362cff6f241d4add8c8a95b2995313.jpg)

![](images/efd071fc6666188cd81c23856f33a0d0d6d74546192d9c03ae9c272d98afcd7a.jpg)

![](images/17ac9f25a3bb27a014d1286aead444efd49960c854fdd7857232e43d74a400ec.jpg)

![](images/ccae9e3f79e70c7b009bb85dc7dfb12103b464eb91b9f2a59f0a369f7303aa5b.jpg)

![](images/cb8db286e3ac856c82ddc45e747e2e6ef64d93dcfe0b482b5f89881356de3a98.jpg)

Bc  
![](images/1b2eb627b19787c296d1e16a26a0b5799e3b4ea0a486d6544ff599d0047785d8.jpg)  
Figure 8: Analyzing the correlation between track and intensity prediction. (A) Ablation results when predictions are made individually and jointly. (B) Saliency maps of historical attribute values. The columns correspond to (a) track, (b) pressure, and (c) wind speed.

## Discussion

As shown in Table 5, Tianmu-TC’s intensity forecasting maintains a clear advantage over the China Meteorological Observatory (CMO), and the European Centre for Medium-Range Weather Forecasts global model, even at the +72 h forecast horizon. However, it must be acknowledged that this work still has some limitations. Track forecasting performance becomes suboptimal beyond 24 h, mainly due to the lack of explicit atmospheric dynamic constraints, and limited long-range temporal dependency modeling capability. Addressing these issues will be the focus of our future work.

## Materials and Methods

## Experimental Setup

For fair comparison, following $\mathrm { T C N } _ { M } (  { 3 } { I } )$ , the sampling number is 6, which means that the model generates 6 possible tendencies. Similarly, Tianmu-TC inputs 48 hours $( n = 8$ , the time resolution is

Table 5: Compare the prediction errors at a 72-hour forecast horizon with authoritative NWP methods in WP basin. The testing years are from 2017 to 2023.
<table><tr><td colspan="2"></td><td colspan="5">Prediction Horizon (hours)</td></tr><tr><td>Error Type</td><td>Method</td><td>+6h</td><td>+12h</td><td>+24h</td><td>+48h</td><td>+72h</td></tr><tr><td rowspan="3">Track Error (km)</td><td>CMO</td><td>32.4</td><td>52.9</td><td>75.5</td><td>132.1</td><td>201.1</td></tr><tr><td>ECMWF</td><td>42.3</td><td>47.0</td><td>65.8</td><td>116.8</td><td>193.6</td></tr><tr><td>Tianmu-TC</td><td>17.8</td><td>21.9</td><td>69.1</td><td>238.7</td><td>438.9</td></tr><tr><td rowspan="3">Pressure Error (hPa)</td><td>CMO</td><td>2.6</td><td>4.37</td><td>6.25</td><td>9.01</td><td>11.25</td></tr><tr><td>ECMWF</td><td>10.9</td><td>11.2</td><td>11.8</td><td>12.9</td><td>11.7</td></tr><tr><td>Tianmu-TC</td><td>1.0</td><td>1.5</td><td>2.1</td><td>4.8</td><td>6.2</td></tr><tr><td rowspan="3">Wind Speed Error (m/s)</td><td>CMO</td><td>1.5</td><td>2.3</td><td>3.5</td><td>4.9</td><td>6.0</td></tr><tr><td>ECMWF</td><td>7.2</td><td>7.2</td><td>7.2</td><td>7.4</td><td>6.9</td></tr><tr><td>Tianmu-TC</td><td>0.5</td><td>0.8</td><td>1.2</td><td>3.0</td><td>4.0</td></tr></table>

6 hours) of TC historical data and outputs 24 hours (� = 4) of future TC attribute data. Tianmu-TC was deployed on the PyTorch framework. Training was performed using Adam Optimizer with a learning rate of 0.0001, a batch size of 256, and a duration of 48 hours.

All experiments, including other deep learning methods used for comparison, were conducted on an NVIDIA RTX A6000 GPU. Random seeds were fixed during both training and testing to ensure reproducibility. The number of training epochs was set to 350 based on model convergence criteria. Absolute errors were computed between predicted results and ground truth for track (distance in km), pressure (hPa), and wind speed (m/s).

## Data preparation and availability

We organize a new dataset, which mainly consists of three parts: historical TC attributes, ERA5 data representing external environment, and TC tilt metrics representing TC internal structure. This dataset encompasses all the 1786 TCs data from 1950 to 2023 over the Western North Pacific. So the datasets contain suficiently diverse conditions of TCs. 80% of the TC data from 1950 to 2016 was allocated for training, 20% for validation, and the data from 2017 to 2023 were reserved for testing. Environmental data from MGTCF—including future move direction, intensity class and future intensity change direction—were also utilized. Further details are described below.

Historical TC attributes are derived from International Best Track Archive for Climate Stewardship (IBTrACS (37)), including essential TC center information such as longitude (<sup>◦</sup>), latitude ( ), pressure (hPa) and wind speed (m/s). Time resolution is 6 hours.

ERA5 data are derived from the fifth-generation atmospheric reanalysis of the global climate (ERA5) (46) by the European Centre for Medium-Range Weather Forecasts (ECMWF). We selected the geopotential height data at 200 hPa and 500 hPa, represented as �200 and �500, which indicate high-pressure systems, as well as the large-scale wind field data, including �200, �500, �700, �850, �200, �500, �700, and �850. Here, � represents the zonal wind component (east-west direction), and � represents the meridional wind component (north-south direction). The 200 hPa data reflect upper-level information of tropical cyclones, the 500 hPa data represent mid-level information, while the 700 hPa and 850 hPa data capture lower-level (near-surface) information of tropical cyclones. The image size is $2 0 ^ { \circ } \times 2 0 ^ { \circ }$ , centered on the TC eye, covering a radius of $1 0 ^ { \circ }$ around the TC’s eye. The spatial resolution is 0.25<sup>◦</sup>, resulting in a data size of $8 1 \times 8 1$

TC tilt metrics are derived from vorticity data, which are calculated from wind field data. As shown in Figure 1D, vorticity at 200 hPa, 500 hPa, and 850 hPa are first computed using the corresponding � and � components for each pressure level, as described in Equation 1. The resulting vorticity data for these three pressure levels are referred to as ��200, ��500, and ��850, respectively. Next, taking ��200 as an example, as shown in Figure 1D, the image size is $8 1 \times 8 1$ The TC eye is marked by a black diamond at the center of the image, located at position (41, 41), denoted as $( \diamond _ { x } , \diamond _ { y } )$ . The three local minima closest to the TC eye are marked as red circles, and their average coordinates are calculated to obtain the center of the three local minima, which is marked as a blue triangle $( \triangle _ { x } ^ { p l } , \triangle _ { y } ^ { p l } ) . p l \in \{ 2 0 0 , 5 0 0 , 8 5 0 \}$ refers to pressure level.

$$
\nu o = \frac { \partial \nu } { \partial x } - \frac { \partial u } { \partial y }\tag{1}
$$

$$
\partial x = \partial y \times \cos \left( { \frac { \pi } { 1 8 0 } } \times { \mathrm { a v g } } { \mathrm { . l a t i t u d e } } \right)\tag{2}
$$

where $\partial y = 1 1 1 . 3 2 \times 0 . 2 5 , 1 1 1 . 3 2$ represents the distance in kilometers corresponding to 1 degree of longitude. The value 0.25 refers to the spatial resolution. Thus, �� represents the distance between data points in the longitudinal direction, with units in kilometers. Similarly, �� represents

the distance between data points in the latitudinal direction, which varies depending on the latitude. Here, it is simplified to the average latitude (avg latitude). Since the training data is based on tropical cyclones in the WP, the average latitude is set to 15<sup>◦</sup>.

$$
D _ { \mathrm { c e n 2 0 0 } } = \sqrt { \left( \triangle _ { x } ^ { 2 0 0 } - \diamondsuit _ { x } \right) ^ { 2 } + \left( \triangle _ { y } ^ { 2 0 0 } - \diamondsuit _ { y } \right) ^ { 2 } }\tag{3}
$$

$$
D _ { \mathrm { c e n } 8 5 0 } = \sqrt { \left( \triangle _ { x } ^ { 8 5 0 } - \triangle _ { x } \right) ^ { 2 } + \left( \triangle _ { y } ^ { 8 5 0 } - \triangle _ { y } \right) ^ { 2 } }\tag{4}
$$

$$
D _ { 2 8 } = \frac { 1 } { 1 + \sqrt { { { \left( \triangle _ { x } ^ { 2 0 0 } - \triangle _ { x } ^ { 8 5 0 } \right) } ^ { 2 } } + { { \left( \triangle _ { y } ^ { 2 0 0 } - \triangle _ { y } ^ { 8 5 0 } \right) } ^ { 2 } } } }\tag{5}
$$

$$
D _ { 5 8 } = \frac { 1 } { 1 + \sqrt { { { \left( \triangle _ { x } ^ { 5 0 0 } - \triangle _ { x } ^ { 8 5 0 } \right) } ^ { 2 } } + { { \left( \triangle _ { y } ^ { 5 0 0 } - \triangle _ { y } ^ { 8 5 0 } \right) } ^ { 2 } } } }\tag{6}
$$

where $D _ { 2 8 }$ represents the alignment between 200 hPa and 850 hPa; $D _ { 5 8 }$ represents the alignment between 500 hPa and 850 hPa. $D _ { \mathrm { c e n 2 0 0 } }$ and $D _ { \mathrm { c e n 8 5 0 } }$ measure the distance between the TC eye and local minima center at 200 hPa and 850 hPa, respectively. The same calculations were performed for each historical time step, resulting in a sequential series of TC tilt metrics.

## Tianmu-TC network design

Overall, Tianmu-TC can be divided into three main modules: constraint encoder, cross-modal fusion, and constraint guiding denoising. The constraint guiding denoising consists of � Physicsconstraints Denoising Blocks.

## Constraint Encoder: How to design the 3 constraints?

LSTM-based Encoder for encoding TC track and intensity, which is designed based on the encoder from Trajectron++ (47). The input data has a shape of (�, 4), where $T = 8 .$ , representing the historical attributes of the tropical cyclone over $8 \times 6$ hours = 48 hours. The four values correspond to longitude, latitude, pressure, and wind speed. After encoding, the output $F _ { 1 }$ has a shape of (256).

CNN-based Encoder encodes selected ERA5 data. The data includes �200, �500, �700, �850, �200, �500, �700, �850, �200, and �500, totaling 10 variables. Therefore, the data shape is (�, 10, 81, 81), where $T = 8$ . The predicted attributes are all the attributes in and around the TC center, which we define as a highly task-relevant area. To enhance task-relevant information, the Central Information Enhancement (CenIE) block enhances the center point of the original ERA5 data (see in the Figure S1 of supplementary materials). As a result, the input data size becomes (�, 20, 81, 81). We implement channel fusion through 6 layers of CNN with ReLU activation functions and 3 layers of spatial attention. This process outputs data with a shape of (�, 1, 16, 16), where the channel dimension � is reduced from 20 to 1, and the height (�) and width (�) are downscaled.

Another LSTM-based encoder encodes TC tilt metrics. The input data has a shape of $( T , 4 )$ where $T = 8$ . The 4 values represent four metrics: $D _ { \mathrm { c e n 2 0 0 } } , D _ { \mathrm { c e n 8 5 0 } } , D _ { 2 8 } , D _ { 5 8 }$ . A single-layer LSTM is used to encode the data, resulting in $F _ { 3 }$ with a size of 256.

## Cross-modal Fusion: Establishing relationships among the three constraints.

The outputs from the constraint encoder have shapes of $F _ { 1 }$ (256), (�, 1, 16, 16), and $F _ { 3 }$ (256), respectively. From the input data shapes, it can also be observed that TC track and intensity, as well as TC tilt values, are one-dimensional attribute values, while ERA5 data is two-dimensional image data. Therefore, it is necessary to match and fuse these two modalities.

As shown in Figure 2, we select the encoding result (256) of TC track and intensity, which is semantically most relevant to the prediction target, and first reshape it into a shape of (1, 16, 16). We initialize � to 0, with a shape of (1, 16, 16).

The iteration is performed over each historical time step, with � ranging from 1 to 8. At $T = 1$ the corresponding slice from (�, 1, 16, 16) with shape (1, 16, 16) is selected and concatenated with the reshaped encodings of TC track, intensity, and �, yielding a final shape of (3, 16, 16). This is then sent into the SA (self-attention module) to capture semantic associations among the three, producing a new � with a shape of (1, 16, 16). This process is repeated to update � until $T = 8$ After reshape the final �, the feature size is (256), which is the result of temporal fusion of ERA5 data, denoted as $F _ { 2 }$

Finally, a 4-layer Transformer encoder is used to fuse $F _ { 1 } , F _ { 2 }$ , and $F _ { 3 }$ to obtain $C _ { 1 } , C _ { 2 }$ , and $C _ { 3 }$ In fact, these three constraints are fused into a complete feature of size (256), denoted as �.

## Constraint guiding denoising

A difusion sequence is $( y _ { 0 } , y _ { 1 } , . . . , y _ { K } )$ , where � is the maximum difusion step. The difusion process gradually adds noise into the ground truth region $y _ { 0 }$ of TC forecasting. Correspondingly, the reverse difusion sequence is $( y _ { K } , y _ { K - 1 } , . . . , y _ { 0 } )$ . This process utilizes historical data as conditions, and gradually denoises the standard Gaussian $y _ { K }$ to obtain the 1D future TC attributes $y _ { 0 }$ . This process reduces prediction noise.

The difusion process is defined as $q ( y _ { k } | y _ { 0 } ) : = N ( y _ { k } ; \sqrt { \bar { \alpha _ { k } } } y _ { 0 } , ( 1 - \bar { \alpha _ { k } } ) \mathrm { I } )$ , thus

$$
y _ { k } = \sqrt { \bar { \alpha _ { k } } } y _ { 0 } + \sqrt { ( 1 - \bar { \alpha _ { k } } ) } \varepsilon\tag{7}
$$

$\begin{array} { r } { \bar { \alpha _ { k } } = \prod _ { s = 1 } ^ { k } \alpha _ { s } , \alpha _ { 1 } , \alpha _ { 2 } , . . . \alpha _ { k } } \end{array}$ are fixed variance schedulers, the noise variable $\varepsilon \sim { \cal N } ( 0 , \mathrm { I } )$ . As the difusion step $k$ increases, $y _ { k } \sim N ( 0 , \mathrm { I } )$ . This means $z _ { 0 }$ is gradually destroyed into a Gaussian noise distribution. The training loss is:

$$
L ( \theta , \psi ) = \mathbb { E } _ { \varepsilon , z _ { 0 } , k } | | \varepsilon - \varepsilon _ { ( \theta , \psi ) } ( y _ { k } , k , X ) | |\tag{8}
$$

$\theta$ are parameters of Physics-constraints Denoising model, $\psi$ are parameters of constraint encoder and cross-modal fusion, � are the input of the model.

The generation of increasingly reliable TC predictions is formulated as a reverse difusion process, guided by three specifically designed constraints that progressively reduce uncertainty.

The reverse difusion process with Bi-condition is defined as:

$$
p _ { \theta } ( y _ { k - 1 } | y _ { k } , C ) : = N ( y _ { k - 1 } ; \mu _ { \theta } ( y _ { k } , k , C ) ; \Sigma _ { \theta } ( y _ { k } , k ) )\tag{9}
$$

$$
\mu _ { \theta } ( \cdot ) = \frac { 1 } { \sqrt { \alpha _ { k } } } ( y _ { k } - \frac { \beta _ { k } } { \sqrt { 1 - \bar { \alpha _ { k } } } } \varepsilon _ { \theta } ( y _ { k } , k , C ) )\tag{10}
$$

$$
\Sigma _ { \theta } ( \cdot ) = \sigma _ { k } ^ { 2 } \mathbf { I } = \beta _ { k } \mathbf { I }\tag{11}
$$

thus:

$$
y _ { k - 1 } = { \frac { 1 } { \sqrt { \alpha _ { k } } } } \left( y _ { k } - { \frac { \beta _ { k } } { \sqrt { 1 - { \bar { \alpha _ { k } } } } } } \varepsilon _ { \theta } ( y _ { k } , k , C ) \right) + { \sqrt { \beta _ { k } } } e\tag{12}
$$

In Equation 12, $e$ is a random variable in standard Gaussian Distribution, $\beta _ { k } \ = \ 1 - \alpha _ { k }$ $\varepsilon _ { \theta }$ is implemented by the Physics-constraints denoising block in Figure 2. The output of the block is:

$$
y _ { k } ^ { \prime } = \varepsilon _ { \theta } ( y _ { k } , k , C )\tag{13}
$$

inspired by MID (48), $\varepsilon _ { \theta }$ is implemented mainly by a Transformer encoder and 4-layer gating units, which use constraints � to generate Gate and bias, and perform a gating transform for latent noise �<sub>�</sub> .

## References and Notes

1. T. R. Knutson, J. L. McBride, J. Chan, K. Emanuel, G. Holland, C. Landsea, I. Held, J. P. Kossin, A. K. Srivastava, M. Sugi, Tropical cyclones and climate change. Nat. Geosci. 3, 157–163 (2010).

2. J. Callaghan, R. Smith, The relationship between maximum surface wind speeds and central pressure in tropical cyclones. Aust. Meteorol. Mag. 47(3), 191–202 (1998).

3. J. A. Knaf, R. M. Zehr, Reexamination of tropical cyclone wind–pressure relationships. Weather Forecast. 22(1), 71–88 (2007).

4. P. Bauer, A. Thorpe, G. Brunet, The quiet revolution of numerical weather prediction. Nature 525(7567), 47–55 (2015).

5. R. Kimura, Numerical weather prediction. J. Wind Eng. Ind. Aerodyn. 90(12-15), 1403–1414 (2002).

6. National Hurricane Center, Forecast Verification. National Hurricane Center, https://www. nhc.noaa.gov/verification/verify3.shtml. Accessed August 5, 2025.

7. Leutbecher, Martin and Palmer, Tim N, Ensemble forecasting, In Journal of computational physics, 227, 7, 3515–3539 (2008).

8. ECMWF, IFS documentation-Cy37r2, operational implementation. (European Centre for Medium-Range Weather Forecasts, 2011).

9. Central Meteorological Observatory CMO, Typhoon network of Central Meteorological Observatory, 2019, http://typhoon.nmc.cn/web.html (Accessed: 2003-03-10).

10. F. Sanders, A. C. Pike, J. P. Gaertner, A barotropic model for operational prediction of tracks of tropical storms. J. Appl. Meteorol. Climatol. 14, 265–280 (1975).

11. I. Price, A. Sanchez-Gonzalez, F. Alet, T. R. Andersson, A. El-Kadi, D. Masters, T. Ewalds, J. Stott, S. Mohamed, P. Battaglia, et al., Probabilistic weather forecasting with machine learning. Nature 637, 84–90 (2025).

12. Z. Gao, X. Shi, B. Han, H. Wang, X. Jin, D. Maddix, C. Li, Y. Wang, et al., PreDif: Precipitation nowcasting with latent difusion models. Advances in Neural Information Processing Systems 36, 78621–78656 (2023).

13. J. Liu, C. Xu, S. Han, L. Song, P. Wang, T. Zhang, Uncertainty-aware precipitation nowcasting with difusion model simulating precipitation evolution processes. Eng. Appl. Artif. Intell. 178, 115140 (2026).

14. Z. Chen, B. Peng, T. Zhai, D. Adu-Ampratwum, X. Ning, Generating 3D small binding molecules using shape-conditioned difusion models with guidance. Nat. Mach. Intell. 1–13 (2025).

15. M. Mardani, N. Brenowitz, Y. Cohen, J. Pathak, C. Y. Chen, C. C. Liu, M. Pritchard, et al., Residual corrective difusion modeling for km-scale atmospheric downscaling. Commun. Earth Environ. 6, 124 (2025).

16. C. Wang, B. Wang, Impacts of the South Asian high on tropical cyclone genesis in the South China Sea. Clim. Dyn. 56, 2279–2288 (2021).

17. L. Wu, Z. Wen, R. Huang, R. Wu, Possible linkage between the monsoon trough variability and the tropical cyclone activity over the western North Pacific. Mon. Weather Rev. 140, 140–150 (2012).

18. Y. Sun, Z. Zhong, L. Yi, T. Li, M. Chen, H. Wan, Y. Wang, K. Zhong, Dependence of the relationship between the tropical cyclone track and western Pacific subtropical high intensity on initial storm size: A numerical investigation. J. Geophys. Res. Atmos. 120, 11–451 (2015).

19. M. Moradi Kordmahalleh, M. Gorji Sefidmazgi, A. Homaifar, A sparse recurrent neural network for trajectory prediction of Atlantic hurricanes. In Proc. Genetic and Evolutionary Comput. Conf. (GECCO), 957–964 (2016).

20. S. Alemany, J. Beltran, A. Perez, S. Ganzfried, Predicting hurricane trajectories using a recurrent neural network. In Proc. AAAI Conf. Artif. Intell. 33, 468–475 (2019).

21. K. Cho, B. van Merrienboer, C. Gulcehre, D. Bahdanau, F. Bougares, H. Schwenk, Y. Bengio,¨ Learning phrase representations using RNN encoder–decoder for statistical machine translation. In Proc. 2014 Conf. Empir. Methods Nat. Lang. Process. (EMNLP), 1724–1734 (2014).

22. T. Song, Y. Li, F. Meng, P. Xie, D. Xu, A novel deep learning model by BiGRU with attention mechanism for tropical cyclone track prediction in the Northwest Pacific. J. Appl. Meteorol. Climatol. 61, 3–12 (2022).

23. S. Gao, P. Zhao, B. Pan, Y. Li, M. Zhou, J. Xu, S. Zhong, Z. Shi, A nowcasting model for the prediction of typhoon tracks based on a long short term memory neural network. Acta Oceanol. Sin. 37, 8–12 (2018).

24. S. Kim, H. Kim, J. Lee, S. Yoon, S. E. Kahou, K. Kashinath, Mr. Prabhat, Deep-Hurricane-Tracker: Tracking and forecasting extreme climate events. In Proc. IEEE Winter Conf. Appl. Comput. Vis. (WACV), 1761–1769 (2019).

25. X. Geng, Z. Liu, Z. Shi, Spatio-Temporal Alignment and Track-To-Velocity Module for Tropical Cyclone Forecast. Remote Sens. 15, 4938 (2023).

26. Y. Wu, X. Geng, Z. Liu, Z. Shi, Tropical cyclone forecast using multitask deep learning framework. IEEE Geosci. Remote Sens. Lett. 19, 1–5 (2021).

27. B. Pan, X. Xu, Z. Shi, Tropical cyclone intensity prediction based on recurrent neural networks. Electron. Lett. 55, 413–415 (2019).

28. C. Huang, C. Bai, S. Chan, J. Zhang, MMSTN: A multi-modal spatial-temporal network for tropical cyclone short-term prediction. Geophys. Res. Lett. 49, e2021GL096898 (2022).

29. C. Huang, C. Bai, S. Chan, J. Zhang, Y. Wu, MGTCF: Multi-generator tropical cyclone forecasting with heterogeneous meteorological data. In Proc. AAAI Conf. Artif. Intell. 37, 5096– 5104 (2023).

30. X. Wang, K. Chen, L. Liu, T. Han, B. Li, L. Bai, Global tropical cyclone intensity forecasting with multi-modal multi-scale causal autoregressive model. ICASSP 2025-2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 1–5 (2025).

31. C. Huang and P. Mu and J. Zhang and S. Chan and S. Zhang and H. Yan and S. Chen and C. Bai, Benchmark dataset and deep learning method for global tropical cyclone forecasting. Nature Commun. 16(1), 5923 (2025).

32. S. Zhang and P. Mu and C. Huang and J. Zhang and C. Bai, TC-Difuser: Bi-Condition Multi-Modal Difusion for Tropical Cyclone Forecasting. Proc. AAAI Conf. Artif. Intell. 39, 1120-1128 (2025).

33. C. Bodnar, W. P. Bruinsma, A. Lucic, M. Stanley, A. Allen, J. Brandstetter, et al., A foundation model for the Earth system. Nature 641, 1180–1187 (2025).

34. K. Bi, L. Xie, H. Zhang, X. Chen, X. Gu, Q. Tian, Accurate medium-range global weather forecasting with 3D neural networks. Nature 619, 533–538 (2023).

35. K. Chen, T. Han, F. Ling, J. Gong, L. Bai, X. Wang, J.-J. Luo, B. Fei, W. Zhang, X. Chen, The operational medium-range deterministic weather forecasting can be extended beyond a 10-day lead time. Commun. Earth Environ. 6(1), 518 (2025).

36. D. Kochkov, J. Yuval, I. Langmore, P. Norgaard, J. Smith, G. Mooers, M. Klower, J. Lottes, S.¨ Rasp, P. D¨uben, et al., Neural general circulation models for weather and climate. Nature 632, 1060–1066 (2024).

37. Knapp, K. R., Kruk, M. C., Levinson, D. H., Diamond, H. J., Neumann, C. J. The international best track archive for climate stewardship (ibtracs): Unifying tropical cyclone data. Bull. Am. Meteorol. Soc. 91, 363 – 376 (2010).

38. Y. Wang, C.-C. Wu, Current understanding of tropical cyclone structure and intensity changes—a review. Meteorol. Atmos. Phys. 87(4), 257–278 (2004).

39. C. Wang, X. Li, G. Zheng, Tropical cyclone intensity forecasting using model knowledge guided deep learning model. Environ. Res. Lett. 19, 024006 (2024).

40. A. Gupta, J. Johnson, L. Fei-Fei, S. Savarese, A. Alahi, Social GAN: Socially acceptable trajectories with generative adversarial networks. In Proc. IEEE Conf. Comput. Vis. Pattern Recognit., 2255–2264 (2018).

41. T. Li, M. Lai, S. Nie, H. Liu, Z. Liang, W. Lv, Tropical cyclone trajectory based on satellite remote sensing prediction and time attention mechanism ConvLSTM model. Big Data Res. 36, 100439 (2024).

42. C.-Y. Lee, M. K. Tippett, A. H. Sobel, S. J. Camargo, Rapid intensification and the bimodal distribution of tropical cyclone intensity. Nat. Commun. 7(1), 10625 (2016).

43. Q. Wu, X. Wang, L. Tao, Interannual and interdecadal impact of Western North Pacific Subtropical High on tropical cyclone activity. Clim. Dyn. 54(3), 2237–2248 (2020).

44. M. L. M. Wong, J. C. L. Chan, Tropical cyclone intensity in vertical wind shear. J. Atmos. Sci. 61(15), 1859–1876 (2004).

45. L. R. Schade, Tropical cyclone intensity and sea surface temperature. J. Atmos. Sci. 57(18), 3122–3130 (2000).

46. H. Hersbach, B. Bell, P. Berrisford, G. Biavati, A. Horanyi, J. Mu ´ noz Sabater, J. Nicolas, C.˜ Peubey, R. Radu, I. Rozum, ERA5 hourly data on single levels from 1979 to present. Copernicus Climate Change Serv. (C3S) Climate Data Store (CDS) 10, 10.24381 (2018).

47. T. Salzmann, B. Ivanovic, P. Chakravarty, M. Pavone, Trajectron++: Dynamically-feasible trajectory forecasting with heterogeneous data. In Proc. Eur. Conf. Comput. Vis. (ECCV), 683– 700 (2020).

48. T. Gu, G. Chen, J. Li, C. Lin, Y. Rao, J. Zhou, J. Lu, Stochastic trajectory prediction via motion indeterminacy difusion. In Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 17113–17122 (2022).

## Acknowledgments

Funding: This work is partially supported by Zhejiang Provincial Natural Science Foundation of China under Grant No. RG25F020001 and LR21F020002, as well as the Natural Science Foundation of China under Grant No. 62202429.

Author contributions: Conceptualization: S.Z., P.M., C.H., C.B., and S.S. Methodology: S.Z., P.M., S.S., C.H., H.Y., and Y.Z. Software: S.Z., P.M., Y.Z., C.H. Validation: S.Z., P.M., H.Y., Y.Z., C.B. and S.S. Formal analysis: S.Z., P.M., C.B., S.S., H.Y., C.H., J.Zh., and S.C. Investigation: S.Z., P.M., C.H., S.S., and C.B. Resources: C.B., S.S., S.C., and J.Zh. Data curation: S.Z., C.H., Y.Z., and H.Y. Writing—original draft: S.Z., P.M. Writing—review and editing: C.B., S.S., P.M., C.H., J.Zh., and S.C. Visualization: S.Z., P.M., and S.S. Supervision: C.B. and S.S. Project administration: C.B. and S.S. Funding acquisition: C.B.

Competing interests: There are no competing interests to declare.

Data and materials availability: Our dataset consists of three main components:

1. Historical TC attributes, sourced from our previously organized dataset, available at https://zenodo.org/records/15009527.

2. ERA5 data, provided by ECMWF, which can be accessed from the https://cds.climate.copernicus.eu/.

3. TC tilt metrics, derived from vorticity data, which are computed using wind field data. The processing code is available at https://github.com/Zjut-MultimediaPlus/Tianmu-TC.

In our model, these datasets are preprocessed into PKL files, which can be obtained from Google Drive: https://drive.google.com/file/d/1XpfByEZkZHAybXgB5p2YsR5KZhHrtVei/view?usp=drive link and https://drive.google.com/file/d/1aiJaUH035YOIbsS9Q1Y9GGmyKW1HiJU1/view?usp=drive link.

All the code and processed data are public on Github ( https://github.com/Zjut-MultimediaPlus/Tianmu-TC). This code will be updated and changed over time.

Supplementary materials

Supplementary Text

Table S1

Figures S1 to S4

Movies S1 to S3

# Supplementary Materials for Tianmu-TC: Physics-constraints Generative Artificial Intelligence for Global Tropical Cyclone Forecasting

Shiqi Zhang<sup>†</sup>, Pan Mu<sup>†</sup>, Cheng Huang, Hanting Yan, Yuchao Zhu,

Jinglin Zhang, Shengyong Chen<sup>∗</sup>, Shoujuan Shu<sup>∗</sup>, Cong Bai<sup>∗</sup>

<sup>∗</sup>Corresponding author. Email: csy@tjut.edu.cn, sjshu@zju.edu.cn, congbai@zjut.edu.cn <sup>†</sup>These authors contributed equally to this work.

This PDF file includes:

Supplementary Text

Table S1

Figures S1 to S4

Captions for Movies S1 to S3

Other Supplementary Materials for this manuscript:

Movies S1 to S3

## Supplementary Text

## A.1 Difusion Models for Multi-Trend Generation

Difusion models are inherently well-suited for multi-trend generation due to their ability to capture complex, multi-modal data distributions. Their probabilistic framework allows for flexible sampling, enabling the generation of diverse trends by conditioning on specific features. Additionally, the iterative refinement mechanism in the reverse difusion process ensures scalability and accurate representation of overlapping patterns. Empirical successes in generative tasks further validate their compatibility with multi-trend dynamics, making difusion models a natural choice for such challenges.

For fair comparison, following $\mathrm { T C N } _ { M }$ (31), the sampling number is 6, which means that the model generates 6 possible tendencies. Similarly, Tianmu-TC inputs 48 hours $( n = 8$ , the time resolution is 6 hours) of TC historical data and outputs 24 hours $( m = 4 )$ of 1D future TC attribute data.

## A.2 NeuralGCM Tropical Cyclone Tracking

NeuralGCM provides publicly available forecast data for 2020 (https://neuralgcm.readthedocs io/en/latest/neuralgcm\_datasets.html). In this study, tropical cyclone tracking is per formed according to the available forecast variables and the tracking parameters reported in Table I7 of the original NeuralGCM paper. The original NeuralGCM setting uses sea level pressure rather than MSLP, while the released forecast variables used in this study do not include SLP. Therefore, NeuralGCM cannot be used for intensity prediction and is only evaluated for track prediction.

For the deterministic forecast, we use the finest-resolution $0 . 7 ^ { \circ }$ NeuralGCM forecast data provided oficially (gs://weatherbench2/datasets/neuralgcm\_deterministic). According to Table I7 of the original NeuralGCM paper, vorticity tracking is used for tropical cyclone tracking. The criteria are: vorticity at 850 hPa change of $\pm 0 . 0 0 0 0 6 \ \mathrm { s ^ { - 1 } }$ over 5.5 GCD, together with a diference between geopotential surfaces at 300 and 500 hPa that decreases by at least $- 2 5 . 8 ~ \mathrm { m } ^ { 2 } \mathrm { s } ^ { - 2 }$ over 6.5 GCD. Here, GCD denotes great-circle distance. For the ensemble forecast (gs://weatherbench2/datasets/neuralgcm\_ens), due to the high computational cost, the original NeuralGCM paper reduced the highest resolution to $1 . 4 ^ { \circ }$ for ensemble forecasts. The tracking method is consistent with that used for the deterministic forecast, and the ensemble size is 50 members.

## A.3 GenCast Tropical Cyclone Tracking and Intensity Extraction

This section describes how tropical cyclone centers and intensity variables were extracted from the public GenCast forecast archive (gs://weatherbench2/datasets/gencast). We first summarize the original TempestExtremes tracker used in GenCast, then describe the modifications required by the limited pressure-level variables available in WeatherBench2, followed by the candidate review procedure and the final intensity extraction strategy.

The original GenCast used TempestExtremes v2.1 to track tropical cyclones. The tracker consists of two main stages. In the first stage, DetectNodes identifies candidate cyclone centers at each time step. Candidate locations are first determined from local minima in mean sea level pressure (MSL), and each low-pressure center is required to be associated with a suficiently compact closed low-pressure structure. In addition, the original tracker requires the candidate storm to be colocated with an upper-level warm-core structure in the 500–300 hPa geopotential thickness field. In the second stage, StitchNodes links candidate nodes across time into physically plausible cyclone trajectories, and further filters non-tropical-cyclone trajectories using criteria related to the maximum displacement between adjacent time steps, the maximum temporal gap, the minimum track duration, wind speed, surface elevation, and latitude. Because GenCast forecasts are evaluated at 12 h intervals, the original study adjusted the StitchNodes connection range from the default $8 ^ { \circ }$ to $1 2 ^ { \circ }$ in great-circle distance. This larger range allows candidate centers at adjacent forecast times to be linked even when a tropical cyclone travels farther within a 12 h interval.

In this study, the public WeatherBench2 GenCast archive provides pressure-level variables only at 500, 700, and 850 hPa. The 300 hPa geopotential height required to reproduce the original 300– 500 hPa warm-core criterion is therefore unavailable. As a result, the original TempestExtremes tracking protocol used in GenCast cannot be fully reproduced. To remain as consistent as possible with the original tracking rationale under the constraints of the public archive, we retained the core DetectNodes criteria based on local MSLP minima and closed low-pressure structures. Following the $1 2 ^ { \circ }$ displacement scale used for 12 h tracking in GenCast, we implemented a $1 2 ^ { \circ }$ local search window for each lead time. The search is anchored at the best-track center for the first lead time and, in the stepwise procedure, at the previously predicted center for the subsequent lead time. It should be emphasized that the goal of this study is not to redetect the genesis, termination, or complete life cycle of all global cyclones. Instead, for each given best-track sample, we diagnose the +12 h and +24 h tropical cyclone centers and intensities from the original GenCast gridded forecast fields. Therefore, the full global StitchNodes trajectory construction, minimum-duration filtering, and elevation and latitude filtering are not applied.

Specifically, for each initialization time and forecast lead time, MSLP, 850 hPa horizontal winds, and 10 m winds are first extracted within a 12° latitude–longitude window around the tracking anchor. The tracking anchor is initialized as the best-track center at the forecast initialization time. In the stepwise tracking procedure, the search anchor for the later lead time is updated to the predicted center from the previous lead time. Candidate centers are generated from local minima in the MSLP field, sorted by their MSLP values in ascending order, and at most 500 candidates are examined. Each candidate is then tested for the presence of a suficiently compact closed low-pressure structure.

Because the 300 hPa warm-core criterion cannot be reproduced, 850 hPa relative vorticity is further used as an auxiliary constraint on the low-level cyclonic structure. The 850 hPa relative vorticity is computed from the 850 hPa horizontal wind field. For candidates in the Northern Hemisphere, cyclonic vorticity corresponds to a positive vorticity core, so the maximum relative vorticity within a 1° radius of the candidate center is used. For candidates in the Southern Hemisphere, cyclonic vorticity corresponds to a negative vorticity core, so the minimum relative vorticity within a 1° radius of the candidate center is used. This criterion does not require the vorticity center and the MSLP minimum to fall on exactly the same grid point; rather, it requires a consistent low-level cyclonic structure near the candidate low-pressure center.

The final center is selected using a hierarchical priority strategy. Among the four lowest local MSLP minima, candidates that satisfy both the SLP closed-contour criterion and the 850 hPa vorticity-structure criterion, and that are close to the tracking anchor, are given the highest priority. All candidates are retained, and whether each candidate passes the SLP and SLP+VORT criteria is recorded for subsequent quality diagnosis.

Because this tracker cannot fully reproduce the TempestExtremes stitching procedure used in the original GenCast, we further performed candidate-level quality review of the automatic tracking results. For every 2020 test sample and lead time, we saved the candidate table, MSLP field, 850 hPa vorticity field, 10 m wind field, and the corresponding visualizations. The review was based on three criteria. First, the selected center should be a local MSLP minimum located within the innermost closed low-pressure structure, rather than a grid point on the outer edge of the closed low or along a low-pressure trough. Second, the candidate should be associated with a consistent cyclonic vorticity core in the 850 hPa relative vorticity field, namely a local positive-vorticity maximum in the Northern Hemisphere and a local negative-vorticity minimum in the Southern Hemisphere. Third, the predicted centers at +12 h and +24 h should be temporally continuous, avoiding jumps to unrelated low-pressure systems between adjacent lead times. For the small number of samples with clear jumps in the automatic tracking results, corrections were restricted to selecting, from the candidate table saved during automatic tracking, a candidate center satisfying the above physicalstructure and temporal-continuity criteria. No new center coordinates were manually specified outside the saved candidate set. To ensure traceability, we saved and provided the predicted tracks, track errors, candidate tables, and visualizations of MSLP, 850 hPa relative vorticity, and 10 m wind speed for all 2020 test samples. Overall, the reviewed centers are consistent with the closed MSLP structure, the low-level cyclonic vorticity structure, and the temporal continuity between adjacent lead times.

After the diagnostic center is determined, GenCast intensity diagnostics are further extracted. The intensity task in this study includes two variables: pressure and wind speed. Pressure refers to the lowest atmospheric pressure at the TC center, in hPa, and wind speed refers to the two-minute maximum sustained wind speed near the TC center, in m/s. For the GenCast gridded forecast fields, we extract the minimum MSLP within a 1° radius of the diagnostic center as the minimum central pressure diagnostic, and the maximum 10 m wind speed within a 3° radius as the near-center wind speed diagnostic. It should be noted that the public GenCast archive provides instantaneous gridded wind fields at discrete forecast times, rather than two-minute maximum sustained wind observations. Therefore, the GenCast wind speed error reported in this study should be interpreted as the diagnostic error of the near-center maximum wind speed derived from the instantaneous 10 m wind field, rather than a strict error of the two-minute maximum sustained wind speed.

Table S1: Extended-Sample Comparison on special TC scenarios on global area from 2017 to 2021. Samples is the number of cases. Lower values indicate better performance. “Anomaly” consists of circular movement, turning on the ocean, and V-shaped tracks. “RI” denotes Rapid Intensification, and “RW” denotes Rapid Weakening.
<table><tr><td rowspan="2"></td><td rowspan="2">Category Scenario (Unit) Methods Samples</td><td rowspan="2"></td><td rowspan="2"></td><td colspan="4">Forecast Horizon</td></tr><tr><td>+6h</td><td>+12h</td><td>+18h</td><td>+24h</td></tr><tr><td rowspan="5">Track</td><td rowspan="3">Anomaly (km)</td><td>Pangu</td><td>1103</td><td>56.25</td><td>62.96</td><td>77.58</td><td>110.94</td></tr><tr><td>Fengwu</td><td>1103</td><td>44.19</td><td>49.33</td><td>65.89</td><td>95.62</td></tr><tr><td>Ours</td><td>1103</td><td>18.08</td><td>17.72</td><td>36.93</td><td>73.57</td></tr><tr><td rowspan="3">Landfall (km)</td><td>Pangu</td><td>582</td><td></td><td>59.84 76.14</td><td>101.26</td><td>6131.16</td></tr><tr><td>Fengwu</td><td>582</td><td></td><td>52.33 62.04</td><td>76.14</td><td>93.80</td></tr><tr><td>Ours</td><td>582</td><td></td><td>17.59 19.06</td><td>38.86</td><td>71.78</td></tr><tr><td rowspan="5">Intensity</td><td rowspan="3">RI (m/s)</td><td>Pangu</td><td>330</td><td>20.79</td><td>29.41</td><td>39.17</td><td>47.02</td></tr><tr><td>Fengwu</td><td>330</td><td></td><td>19.92 27.71</td><td>36.82</td><td>43.99</td></tr><tr><td>Ours</td><td>330</td><td>1.37</td><td>1.31</td><td>4.63</td><td>8.04</td></tr><tr><td>Pangu</td><td>464</td><td></td><td>28.17 21.53</td><td></td><td>15.50</td><td>11.37</td></tr><tr><td rowspan="2">RW (m/s)</td><td>Fengwu</td><td>464</td><td></td><td>27.42 20.76</td><td></td><td>14.48</td><td>10.08</td></tr><tr><td>Ours</td><td>464</td><td></td><td>2.70</td><td>1.70</td><td>3.89</td><td>5.78</td></tr></table>

![](images/bae544d45848cf1e4729b2e2d06460928bcff7ebc0f0638225f10b673a7de2ae.jpg)  
Figure S1: Central Information Enhancement (CenIE) block. Considering that the center of selected ERA5 data corresponds to the location of the TC center, specifically, the longitude and latitude coordinates in the track task. Besides, pressure represents the lowest pressure near the TC center, and wind speed denotes the two-minute maximum sustained wind speed near the TC center. In other words, the predicted attributes are all the attributes in and around the TC center. Therefore, we define TC center as a highly task-relevant area. To enhance task-relevant information in the selected ERA5 data, CenIE block enhances the center point $E _ { a , b } ^ { t + i ^ { \prime } }$ of original 2D data $E _ { t + i } ^ { \prime }$ and $E _ { t + i } = c o n c a t ( E _ { t + i } ^ { \prime } , E _ { t + i } ^ { \prime } - E _ { a , b } ^ { t + i ^ { \prime } } )$ , where � in the figure denotes channel concatenation, $i \in [ 1 , n ]$ 2 the � and � denote the coordinates in the geopotential height map, � and � denote the width and height of the map, and � denotes the length of the historical data. That is, the value of each pixel point in $E _ { t + i } ^ { \prime }$ is subtracted from that of the current central point $E _ { a , b } ^ { t + i ^ { \prime } }$ , and concatenated with the original $E _ { t + i } ^ { \prime }$ . Finally, CenIE block outputs the enhancement of central information $E _ { t + i }$ . This data preprocessing method is very simple and does not need GPU resources.

Aa  
![](images/1cfb8a4d4ada6f420adbb0f3344a016c80ca2d36628e5cec213745a49577b51f.jpg)

Ab  
![](images/d002d746082d33185587edcf9bf86439f388d30909d259910f95a457d6d9a6ac.jpg)

![](images/3b62722e84d75070671ffb9ad93271aba7cd35457f9220d698cbcc87a0128269.jpg)

![](images/666ece7c1ed5b60d3903a666ff0522b483150c9f9afbe4fe915717b178d3fe21.jpg)  
Figure S2: Comparison with additional operational models in the EP and NA basins. We include authoritative forecasting systems issued by the National Hurricane Center (NHC): the oficial forecast (OFCL), and the statistical-dynamical model combination OCD5, which merges CLP5 (track) and DSHP (intensity). Other models include EMXI (ECMWF global model from the previous cycle), HCCA (a weighted consensus of multiple dynamical and statistical models), as well as recent large models Pangu and Fengwu. Testing period: from 2017 to 2023.

A  
![](images/e9634ec7afda272e4eeec3ca4752b7d64d0c7336494e2f698051b3ff0d61b119.jpg)

![](images/486a79f167eab078b55a691767d589ba53ab8e5d7abfeaafa9ef44b255ffeed9.jpg)

C  
![](images/605cd6d6a9e7e3d576a7d91e8ddb77658284568bd365866709865fe429cd992d.jpg)  
Figure S3: More comparison methods on WP.

![](images/1ecf31dab9f147682670eb482c9c35aacffaea7a74b80989a262051e81d50a5c.jpg)

<table><tr><td rowspan=1 colspan=1>255.468</td><td rowspan=1 colspan=1>337.455</td><td rowspan=1 colspan=1>156.905</td><td rowspan=1 colspan=1>144.027</td><td rowspan=1 colspan=1>257.727</td></tr><tr><td rowspan=1 colspan=1>268.173</td><td rowspan=1 colspan=1>262.739</td><td rowspan=1 colspan=1>114.549</td><td rowspan=1 colspan=1>90.274</td><td rowspan=1 colspan=1>178.358</td></tr><tr><td rowspan=1 colspan=1>286.144</td><td rowspan=1 colspan=1>193.821</td><td rowspan=1 colspan=1>73.088</td><td rowspan=1 colspan=1>50.749</td><td rowspan=1 colspan=1>102.834</td></tr><tr><td rowspan=1 colspan=1>312.656</td><td rowspan=1 colspan=1>141.832</td><td rowspan=1 colspan=1>45.099</td><td rowspan=1 colspan=1>31.989</td><td rowspan=1 colspan=1>46.791</td></tr><tr><td rowspan=1 colspan=1>341.232</td><td rowspan=1 colspan=1>114.481</td><td rowspan=1 colspan=1>31.384</td><td rowspan=1 colspan=1>30.842</td><td rowspan=1 colspan=1>17.961</td></tr><tr><td rowspan=1 colspan=1>354.135</td><td rowspan=1 colspan=1>101.826</td><td rowspan=1 colspan=1>26.025</td><td rowspan=1 colspan=1>28.838</td><td rowspan=1 colspan=1>14.290</td></tr></table>

<table><tr><td rowspan=1 colspan=1>129.056</td><td rowspan=1 colspan=1>62.808</td><td rowspan=1 colspan=1>87.137</td><td rowspan=1 colspan=1>115.579</td><td rowspan=1 colspan=1>36.373</td></tr><tr><td rowspan=1 colspan=1>115.359</td><td rowspan=1 colspan=1>57.781</td><td rowspan=1 colspan=1>72.382</td><td rowspan=1 colspan=1>107.298</td><td rowspan=1 colspan=1>31.581</td></tr><tr><td rowspan=1 colspan=1>89.606</td><td rowspan=1 colspan=1>46.939</td><td rowspan=1 colspan=1>52.447</td><td rowspan=1 colspan=1>81.493</td><td rowspan=1 colspan=1>23.125</td></tr><tr><td rowspan=1 colspan=1>58.912</td><td rowspan=1 colspan=1>31.047</td><td rowspan=1 colspan=1>30.389</td><td rowspan=1 colspan=1>47.775</td><td rowspan=1 colspan=1>13.149</td></tr><tr><td rowspan=1 colspan=1>28.698</td><td rowspan=1 colspan=1>13.715</td><td rowspan=1 colspan=1>11.021</td><td rowspan=1 colspan=1>18.340</td><td rowspan=1 colspan=1>4.562</td></tr><tr><td rowspan=1 colspan=1>11.820</td><td rowspan=1 colspan=1>2.711</td><td rowspan=1 colspan=1>2.165</td><td rowspan=1 colspan=1>4.993</td><td rowspan=1 colspan=1>0.700</td></tr></table>

<table><tr><td rowspan=1 colspan=1>24.112</td><td rowspan=1 colspan=1>25.266</td><td rowspan=1 colspan=1>33.492</td><td rowspan=1 colspan=1>37.073</td><td rowspan=1 colspan=1>21.920</td></tr><tr><td rowspan=1 colspan=1>22.085</td><td rowspan=1 colspan=1>19.979</td><td rowspan=1 colspan=1>29.717</td><td rowspan=1 colspan=1>35.916</td><td rowspan=1 colspan=1>18.703</td></tr><tr><td rowspan=1 colspan=1>17.653</td><td rowspan=1 colspan=1>12.824</td><td rowspan=1 colspan=1>22.834</td><td rowspan=1 colspan=1>28.264</td><td rowspan=1 colspan=1>14.386</td></tr><tr><td rowspan=1 colspan=1>11.471</td><td rowspan=1 colspan=1>5.516</td><td rowspan=1 colspan=1>13.687</td><td rowspan=1 colspan=1>16.366</td><td rowspan=1 colspan=1>8.231</td></tr><tr><td rowspan=1 colspan=1>5.689</td><td rowspan=1 colspan=1>1.310</td><td rowspan=1 colspan=1>5.084</td><td rowspan=1 colspan=1>5.741</td><td rowspan=1 colspan=1>2.462</td></tr><tr><td rowspan=1 colspan=1>2.786</td><td rowspan=1 colspan=1>1.056</td><td rowspan=1 colspan=1>0.795</td><td rowspan=1 colspan=1>0.804</td><td rowspan=1 colspan=1>0.263</td></tr><tr><td></td><td rowspan=1 colspan=1>C1</td><td rowspan=1 colspan=1>C1,C2</td><td></td><td></td></tr></table>

Figure S4: Quantitative results of uncertainty for (a) track, (b) pressure and (c) wind speed.

Caption for Movie S1. The reduction of prediction uncertainty in track forecasting. Animated version of Figure 3(a). The red solid line represents the ground truth, the cyan dashed lines represent the predictions of 6 tendencies. Under the guidance of proposed physics-constrained generative AI, six tendencies gradually converge into a region with low uncertainty.

Caption for Movie S2. The reduction of prediction uncertainty in pressure forecasting. Animated version of Figure 3(b). The blue solid line represents the ground truth, the cyan dashed lines represent the predictions of 6 tendencies.

Caption for Movie S3. The reduction of prediction uncertainty in wind speed forecasting. Animated version of Figure 3(c). The yellow solid line represents the ground truth, the cyan dashed lines represent the predictions of 6 tendencies.