# MAGPIE-Net: Predicting short-duration heavy-rainfall events in station neighborhoods from multitemporal FY-4A AGRI observations

Preprint

Xiang Lin<sup>1</sup>, Yunying Li<sup>∗1,2</sup>, Chengzhi Ye<sup>3,4</sup>, Zitong Chen<sup>1</sup>, and Jing Sun<sup>1</sup>

<sup>1</sup>College of Meteorology and Oceanography, National University of Defense Technology, Changsha 410073, China <sup>2</sup>Key Laboratory of High Impact Weather (Special), China Meteorological Administration, Changsha 410073, China <sup>3</sup>Hunan Key Laboratory of Meteorological Disaster Prevention and Reduction, Changsha 410118, China <sup>4</sup>Institute of Meteorological Science of Hunan Province, Changsha 410118, China

## Abstract

Short-duration heavy-rainfall warning determines whether 1 h rainfall will exceed a threshold within a target-station neighborhood over the next few hours. Multitemporal infrared and water-vapor observations from the Fengyun-4A Advanced Geostationary Radiation Imager (FY-4A AGRI) capture cloud-top cooling, moisture evolution, and cloud expansion before substantial surface rainfall develops. However, most deep-learning nowcasting methods convert these signals into local warnings by post-processing gridded precipitation predictions, preventing station-neighborhood event targets from directly supervising the satellite-tostation learning pathway. We propose MAGPIE-Net, which embeds a geographically adaptive, diferentiable grid-to-station mapping in a pathway combining convection-initiation features, multiscale encoding, and auxiliary gridded precipitation diagnosis. Station-neighborhood event losses thereby constrain the satellite representation and its mapping to irregular station locations for 0–3 h event prediction. In independent 2023 warm-season tests over central and eastern China, critical success index (CSI) values under the primary 40 km/20 mm h<sup>−1</sup> definition were 0.371, 0.304, and 0.238 at 0–1, 1–2, and 2–3 h. Across episodes, MAGPIE-Net achieved a detection rate of 65.1% and a mean lead time of 64.6 min, compared with 23.6% and 18.3 min for the best gridded-output baseline, and remained superior for smaller neighborhoods and the 50 mm h<sup>−1</sup> threshold. During the critical early-warning stage, when antecedent 1 h rainfall within 40 km remained below 1 mm, MAGPIE-Net detected 51.9% of episodes with a mean lead time of 38.5 min. These results show that event-oriented satelliteto-station modeling converts multitemporal geostationary cloud and moisture observations into local heavy-rainfall warnings more efectively than gridded-precipitation modeling.

Keywords Short-duration heavy rainfall · FY-4A AGRI · Station-neighborhood warning · Event-oriented modeling · Deep learning

## 1 Introduction

Short-duration heavy rainfall develops rapidly and can trigger damaging flash floods and urban flooding, particularly in small catchments and densely populated areas (Fowler et al., 2021; Westra et al., 2014). Over eastern China, extreme hourly rainfall is highly localized, and warm-season events are often associated with mesoscale convective systems (Zhang and Zhai, 2011; Chen et al., 2013). Urban drainage systems and small catchments can respond within hours, making hourly rainfall monitoring and warning central to severe-convective-weather operations (Zheng et al., 2015). This task may become increasingly important as hourly precipitation extremes intensify in a warming climate (Prein et al., 2017). Operational precipitation nowcasting has long relied on weather radar, whose frequent, high-resolution scans resolve the location, intensity, and movement of precipitation systems (Wilson et al., 1998; Germann and Zawadzki, 2002; Imhof et al., 2022). During the earliest stages of convective development, however, radar echoes may remain weak after cumulus clouds begin to grow vertically. A distinct precipitation echo may emerge only after rapid cloud development has begun (Roberts and Rutledge, 2003).

Geostationary satellite imagers can observe cloud development before precipitation echoes become pronounced. Infrared and water-vapor image sequences capture cloud-top cooling, vertical growth, horizontal expansion, glaciation-related spectral changes, and moistening in the middle and upper troposphere. These signals provide radiative indicators of convective development and have supported convection-initiation (CI) nowcasting across successive generations of geostationary satellites (Mecikalski and Bedka, 2006; Sieglaf et al., 2011; Walker et al., 2012; Lee et al., 2017). Early studies linked cloud-top cooling and multispectral changes to first radar echoes. The associated cloud-growth signals could precede the first significant radar echo by approximately 30–45 min (Roberts and Rutledge, 2003; Mecikalski et al., 2010). Later systems incorporated object tracking and machine-learning detection (Zinner et al., 2008; Han et al., 2019). Recent Himawari-8 studies produced 30-min CI forecasts and detected local convective storms approximately 30 min to 1 h before radar in representative South China cases (Li et al., 2024; Yang et al., 2024). These studies establish satellite-observed cloud evolution as an observational basis for earlier warning while convection is still developing.

Geostationary satellites complement this early observational capability with broader spatial coverage. Efective weather-radar coverage can be constrained by network distribution, terrain blockage, and increasing beam height with range, particularly over ofshore waters, mountainous terrain, and remote regions (Saltikof et al., 2019; Lee et al., 2021). In contrast, geostationary infrared and water-vapor channels continuously observe broad land and ocean domains during both day and night. The Fengyun-4A Advanced Geostationary Radiation Imager (FY-4A AGRI) provides frequent multispectral information on cloud-top temperature, atmospheric moisture, and cloud-phase-related structure (Yang et al., 2017). These characteristics allow AGRI to provide continuous information on convective development where radar coverage is limited and make it suitable for examining how regional cloud and moisture evolution supports local warning.

Broad observational availability, however, does not remove the physical dificulty of converting satelliteobserved cloud signals into surface-rainfall warnings. Reviews of satellite precipitation retrieval identify the indirect radiance–rainfall relationship as a fundamental source of uncertainty (Stephens and Kummerow, 2007; Kidd and Levizzani, 2011). Surface rainfall development depends on updraft strength, cloud microphysics, low-level moisture supply, boundary-layer convergence, vertical wind shear, terrain forcing, and storm organization (Doswell et al., 1996; Houze, 2012, 2014). Similar cloud-top cooling can therefore accompany diferent rainfall outcomes. Storm motion and vertical wind shear can also displace the surface rainfal maximum from the coldest cloud top. Analyses over China show substantial diferences between the spatia extent of convective cold clouds and the associated surface rainfall (Chen et al., 2025). Satellite observations indicate convective development without uniquely specifying the location, intensity, or spatial structure of subsequent surface rainfall. A satellite-based warning model must therefore identify the cloud and moisture evolution patterns associated with subsequent heavy rainfall while allowing the warning target to accommodate spatial displacement between cloud structures and surface rain cores.

Based on this consideration, we formulate the warning target as a station-neighborhood heavy-rainfall event. An event occurs when 1 h accumulated rainfall reaches or exceeds a prescribed threshold within a specified radius of a target station. A single rain gauge samples only a small part of a convective system, while an intense rain core can shift as the system develops and moves. Future rainfall fields may difer in the exact location and fine-scale morphology of their rain cores. They nevertheless represent the same station-neighborhood event when threshold-exceeding rainfall occurs within the same target-station neighborhood. The event target therefore focuses prediction on whether the area surrounding a station will be afected while accommodating uncertainty in rain-core location.

This definition accords with neighborhood, scale-aware, and object-based verification methods for highresolution precipitation forecasts (Ebert, 2008; Roberts and Lean, 2008; Gilleland et al., 2009; Davis et al., 2006; Gilleland et al., 2010). These methods accommodate small spatial displacements instead of treating them as complete forecast failures. Point-to-area sensitivity studies have similarly examined how neighborhood size afects short-duration heavy-rainfall verification (Tian et al., 2015). Operational practice provides direct precedent. China’s national standard GB/T 44213-2024 recommends circular-neighborhood fuzzy verification and identifies 40 km as the preferred radius (Standardization Administration of China, 2024). Earlier descriptions of National Meteorological Center operations document the same station-centered 40-km pointto-area method (Zheng et al., 2015). The U.S. Weather Prediction Center likewise expresses excessive-rainfall risk as the probability of exceeding flash-flood guidance within 40 km of a point (Erickson et al., 2021; Burke et al., 2023). These approaches preserve a location-specific warning target while accommodating displacement and scale uncertainty in localized convective rainfall.

Despite the operational relevance of station-neighborhood events, the prevailing deep-learning formulation for precipitation nowcasting remains gridded. Early radar-nowcasting systems framed prediction as convolutional sequence modeling or image-to-image translation (Shi et al., 2015, 2017; Ayzel et al., 2020; Trebing et al., 2021). Subsequent probabilistic and longer-range systems continued to generate spatial precipitation fields (Sønderby et al., 2020; Pulkkinen et al., 2019; Ravuri et al., 2021; Espeholt et al., 2022). Recent Transformer-based, physics-conditioned, and satellite-driven methods retain the same gridded formulation (Gao et al., 2022; Zhang et al., 2023; Park et al., 2025). These models describe the spatial distribution and evolution of precipitation, but their prediction endpoint couples several requirements. The model must determine whether heavy rainfall develops while resolving its location, intensity, morphology, and movement. Satellite cloudtop and water-vapor observations constrain these properties unevenly. For localized and relatively rare heavy-rainfall events, neighborhood threshold exceedance is only one property that the predicted field must reproduce. Station-neighborhood warnings are then derived after training through thresholding, interpolation, or neighborhood aggregation. Training consequently optimizes the gridded precipitation field, whereas the final warning decision concerns a station-neighborhood event. With grid-to-station conversion outside the trained model, event-prediction errors cannot directly constrain satellite feature learning or the spatia mapping to individual stations.

Figure 1 contrasts the two pathways. The conventional approach derives station-neighborhood events from a gridded precipitation forecast after model training. The event-oriented pathway instead incorporates a learnable grid-to-station mapping and station-neighborhood event supervision into the model, allowing the final event loss to shape both the regional satellite representation and its mapping to target stations.

To implement this formulation, we develop MAGPIE-Net for predicting station-neighborhood heavy-rainfall events over the next 0–3 h from multitemporal FY-4A AGRI observations. The model derives CI features from rapidly developing cloud regions and encodes regional cloud and moisture evolution at multiple scales. An auxiliary diagnosis head projects the regional satellite representation onto three forecast-window-specific gridded precipitation fields. The geographically adaptive SetConv (GA-SetConv) decoder maps these fields to station-local representations using location- and lead-dependent spatial support while incorporating geographic attributes and cyclical time encodings. A multi-head module converts the resulting representations into event probabilities for multiple neighborhood radii and rainfall thresholds. Joint training propagates station-neighborhood event losses through the grid-to-station mapping, directly linking satellite representation learning to the final warning target.

We compare MAGPIE-Net with gridded-output baselines on an independent 2023 warm-season test set over central and eastern China, using matched station-neighborhood event definitions. The analysis spans forecast windows, event definitions, and stages of rainfall development, while controlled configurations identify component contributions. These comparisons assess whether the event-oriented satellite-to-station pathway translates cloud and moisture evolution in multitemporal AGRI observations into station-neighborhood heavy-rainfall event probabilities more efectively than the conventional gridded-output pathway.

## 2 Data

## 2.1 Study Region

The study domain spans central and eastern China from $2 0 . 4 0 ^ { \circ }$ to $3 8 . 2 8 ^ { \circ } \mathrm { N }$ and from $1 0 7 . 0 4 ^ { \circ }$ to $1 2 4 . 9 2 ^ { \circ } \mathrm { E }$ (Fig. 2). It encompasses inland and coastal regions, plains, and mountainous terrain. Warm-season shortduration heavy rainfall is associated with frontal and warm-sector convection, mesoscale convective systems, tropical-cyclone rainbands, and terrain-related forcing (Chen et al., 2013; Luo et al., 2017; Lonfat et al., 2004; Houze, 2012). This combination of precipitation regimes and terrain settings provides a suitable setting for evaluating satellite-based station-neighborhood heavy-rainfall warning.

## 2.2 FY-4A AGRI Observations

The satellite input is derived from FY-4A AGRI products provided by the National Satellite Meteorological Center of the China Meteorological Administration. We use channels $7 \mathrm { - } 1 4 .$ , which span the mid-wave infrared, water-vapor, thermal-infrared window, split-window, and $\mathrm { C O _ { 2 } }$ absorption bands and are available during both day and night. Together, these channels describe cloud-top temperature, cloud phase, and moisture in the middle and upper troposphere. Changes between successive scans capture cloud-top cooling, cloud growth, horizontal expansion, and moisture evolution, thereby describing the evolution of regional cloud and moisture fields (Yang et al., 2017; Mecikalski et al., 2010). Because the radiance response of each channel also depends on cloud vertical structure and optical and microphysical properties (Sun et al., 2025), the eight channels are used jointly as multitemporal predictors. No individual channel is treated as a direct indicator of surface rainfall.

![](images/8e77368eaca180014a40bf8d4c2b118d2a3f82cd6e468cefeb299927cceda850.jpg)  
Figure 1: Comparison of the conventional gridded-output pathway and the event-oriented satellite-to-station modeling pathway for satellite-based short-duration heavy-rainfall warning. Both pathways use regional multichannel geostationary satellite sequences. In the conventional pathway, model training and verification are based on gridded precipitation, and station-neighborhood events are subsequently derived through radius-based neighborhood aggregation. The event-oriented pathway incorporates a learnable grid-to-station mapping and uses station-neighborhood heavy-rainfall events as the common target for training, verification, and warning output.

Table 1 summarizes the spectral channels used as model input. After geolocation and spatial registration, all observations are remapped to a common 448 × 448 latitude–longitude grid covering the study domain, with an approximate spacing of 0.04<sup>◦</sup>. Each AGRI scan is therefore represented by eight spatially aligned channel fields on the regional grid.

## 2.3 Ground-Station Precipitation Observations

Ground-station precipitation observations were obtained from the National Meteorological Information Center of the China Meteorological Administration. The dataset reports 1 h accumulated precipitation at 15 min intervals for national meteorological stations and regional automatic weather stations. At each valid time $t ,$ the available record includes every station with a valid observation, its precipitation accumulation over the preceding hour, and its longitude, latitude, and elevation. We denote the set of valid stations by $S _ { t }$ . For station s, $P _ { s } ( t )$ denotes the 1 h accumulated precipitation, and $\boldsymbol { r _ { s } } = \left( \lambda _ { s } , \phi _ { s } , h _ { s } \right)$ contains the geographic attributes. These records form the basis of the station-neighborhood heavy-rainfall event labels.

(a) National-station distribution and marginal counts  
![](images/dd185b8edc88bb0a5b80654e1ba334d8cc3f68bfd6e74066e707a5a752112896.jpg)

![](images/3fece4d992b38c40f841d4e165975cdd87cd4035c66889ee80643778729be8df.jpg)

(b) Location in China  
![](images/314dbd51cac7f942b8c349414e0b15db0b2223f2442ffb0716e3ae1e864ad283.jpg)

(c) Regional automatic weather station density  
![](images/87a6239466869efcdc6a02111ef7f8406f9f2a76347b73e6e6c8dbc4a3ae948d.jpg)  
Figure 2: Study domain and station networks. (a) Distribution of national meteorological stations, with marginal counts by longitude and latitude. (b) Location of the study domain within China. (c) Gaussian smoothed relative density of regional automatic weather stations in the precipitation dataset.

Table 1: FY-4A AGRI channels used as model input.
<table><tr><td>Channel</td><td>Type</td><td>Central wavelength</td><td>Native spatial resolution</td></tr><tr><td>7</td><td>Mid-wave infrared (high gain)</td><td>3.75 µm</td><td>2 kmª</td></tr><tr><td>8</td><td>Mid-wave infrared (low gain)</td><td>3.75 µm</td><td>4 km</td></tr><tr><td>9</td><td>Upper-tropospheric water vapor</td><td>6.25 µm</td><td>4 km</td></tr><tr><td>10</td><td>Mid-tropospheric water vapor</td><td>7.10µm</td><td>4km</td></tr><tr><td>11</td><td>Long-wave infrared</td><td>8.50μm</td><td>4km</td></tr><tr><td>12</td><td>Thermal-infrared window</td><td>10.70 µm</td><td>4 km</td></tr><tr><td>13</td><td>Split-window infrared</td><td>12.00µm</td><td>4km</td></tr><tr><td>14</td><td> $\mathrm { C O _ { 2 } }$  absorption</td><td>13.50µm</td><td>4 km</td></tr></table>

<sup>a</sup> Channel 7 has a native spatial resolution of 2 km, but the oficial AGRI product used here provides this channel at 4 km. All selected channels are subsequently remapped to the common grid with an approximate spacing of 0.04<sup>◦</sup>.

## 2.4 Station-Neighborhood Heavy-Rainfall Event Definition

For a target station $s ,$ consider the 1 h window ending at time t. A station-neighborhood heavy-rainfall event occurs when at least one station within radius R of s records a 1 h accumulation that meets threshold q. The neighborhood calculation includes every national meteorological station and regional automatic weather station with a valid rainfall record at t. For $s \in S _ { t }$ , the neighborhood is defined as

$$
\mathcal { N } _ { R } ( s , t ) = \{ s _ { i } \in S _ { t } \mid d ( s _ { i } , s ) \le R \}\tag{1}
$$

where $d ( s _ { i } , s )$ denotes the geographic distance between stations $s _ { i }$ and s. The maximum 1 h accumulated precipitation within the neighborhood is

$$
P _ { s } ^ { R , \operatorname* { m a x } } ( t ) = \operatorname* { m a x } _ { s _ { i } \in \mathcal { N } _ { R } ( s , t ) } P _ { s _ { i } } ( t )\tag{2}
$$

and the corresponding binary event label is

$$
y _ { s } ^ { R , q } ( t ) = \left\{ \begin{array} { l l } { 1 , } & { P _ { s } ^ { R , \operatorname* { m a x } } ( t ) \geq 2 0 , \quad q = 2 0 , } \\ { 1 , } & { P _ { s } ^ { R , \operatorname* { m a x } } ( t ) > 5 0 , \quad q = 5 0 , } \\ { 0 , } & { \mathrm { o t h e r w i s e } . } \end{array} \right.\tag{3}
$$

Thus, $y _ { s } ^ { R , q } ( t ) = 1$ indicates that the event criterion is satisfied at target station s during the 1 h window ending at t.

Three neighborhood radii, $R \in \{ 4 0 , 2 0 , 1 0 \}$ km, are paired with two rainfall thresholds, $q \in \{ 2 0 , 5 0 \} \mathrm { m m h ^ { - 1 } }$ to produce six event definitions. The complete label map at time t is written as

$$
\begin{array} { r } { \mathbf { Y } ( t ) = \left\{ \left( r _ { s } , \left\{ y _ { s } ^ { R , q } ( t ) \right\} _ { R \in \left\{ 4 0 , 2 0 , 1 0 \right\} , q \in \left\{ 2 0 , 5 0 \right\} } \right) \mid s \in \mathcal { S } _ { t } \right\} } \end{array}\tag{4}
$$

At each valid time, $\mathbf { Y } ( t )$ forms an irregular station-based label map containing the geographic attributes and six binary event labels for every valid station. The $4 0 \mathrm { k m / \bar { 2 0 } m m h ^ { - 1 } }$ combination is used as the primary event definition. The remaining five definitions serve as auxiliary targets during joint training and support evaluation under stricter spatial or rainfall-intensity criteria. Smaller radii require more precise event localization, whereas the $\mathrm { 5 0 m m } \dot { \mathrm { h } } ^ { \dot { - } 1 }$ threshold focuses on more intense short-duration rainfall.

## 2.5 Sample Construction and Data Split

Each forecasting sample is defined by an initialization time $t _ { 0 }$ . The satellite input consists of four AGRI scans acquired at 15 min intervals before initialization:

$$
\mathcal { T } _ { t _ { 0 } } ^ { \mathrm { s a t } } = \{ t _ { 0 } - 6 0 \operatorname* { m i n } , t _ { 0 } - 4 5 \operatorname* { m i n } , t _ { 0 } - 3 0 \operatorname* { m i n } , t _ { 0 } - 1 5 \operatorname* { m i n } \}\tag{5}
$$

Let $\mathbf { B } ( t ) \in \mathbb { R } ^ { 8 \times H \times W }$ denote the eight-channel AGRI field at time t. The four scans are concatenated along the channel dimension to form the satellite input:

$$
X _ { t _ { 0 } } ^ { \mathrm { s a t } } = \operatorname { C o n c a t } _ { \mathrm { c h } } \left[ \mathbf { B } ( t ) \right] _ { t \in \mathcal { T } _ { t _ { 0 } } ^ { \mathrm { s a t } } } \in \mathbb { R } ^ { 3 2 \times H \times W }\tag{6}
$$

The three forecast windows are

$$
H _ { k } = ( t _ { 0 } + ( k - 1 ) \ln , t _ { 0 } + k \ln ] \qquad k = 1 , 2 , 3\tag{7}
$$

and the corresponding targets are the station-neighborhood event-label maps ending 1, 2, and 3 h after initialization:

$$
{ \bf Y } _ { t _ { 0 } } ^ { \mathrm { t a r } } = [ { \bf Y } ( t _ { 0 } + 1 \mathrm { h } ) , { \bf Y } ( t _ { 0 } + 2 \mathrm { h } ) , { \bf Y } ( t _ { 0 } + 3 \mathrm { h } ) ]\tag{8}
$$

Since each label map is constructed from the 1 h accumulated precipitation ending at its valid time, the three targets correspond to the 0–1, 1–2, and 2–3 h forecast windows, respectively.

Station masks distinguish the locations used as prediction targets from those used only as neighborhood observations. National meteorological stations have more consistent quality control and temporal coverage, so they are used for supervision during training and validation and for the primary independent-test evaluation. Regional automatic weather stations contribute to the neighborhood rainfall observations used to construct the event labels but are excluded from supervised targets and primary national-station metrics. The modeling domain contains 987 national stations, of which 973 within the central evaluation crop are retained for the primary test metrics to reduce boundary efects.

Samples are restricted to May through September and split strictly by calendar year. Data from 2018–2021 are used for training, data from 2022 for validation and hyperparameter selection, and data from 2023 for independent testing. This temporal separation keeps the entire 2023 warm season outside model fitting and selection. To reduce the dominance of event-free conditions during training, we retain 10% of samples with no positive event label across the supervised target maps and all samples containing at least one positive label. The validation and test sets retain their original distributions. The final training, validation, and independent test sets contain 26,121, 3,136, and 3,097 samples, respectively.

## 3 MAGPIE-Net: An Event-Oriented Satellite-to-Station Architecture

## 3.1 Overall Architecture

Figure 3 summarizes the overall architecture of MAGPIE-Net. The model comprises a convection-initiation (CI) feature-construction module, a multiscale satellite encoder, an auxiliary gridded precipitation diagnosis head, a GA-SetConv grid-to-station decoder, and a multi-head event prediction module. It preserves cloudand rainfall-related spatial structures on the regional grid before mapping them to irregular target stations. Because this mapping is diferentiable, station-neighborhood event losses constrain both the shared satellite representation and its mapping to target stations.

![](images/e64f7cdebd582de704a673b355d47881e9f82d702756ea06a4f4624da4088893.jpg)  
Figure 3: Overall architecture of MAGPIE-Net. The input AGRI sequence is used to construct CI features, which are combined with the original AGRI channels by the multiscale satellite encoder. The encoder produces a regional satellite representation, from which the auxiliary diagnosis head generates three gridded precipitation fields. GA-SetConv maps these fields to target stations, and the multi-head module predicts event probabilities for three forecast windows. CBR denotes a convolution–batch-normalization–ReLU block. For clarity, repeated indices are omitted from the GA-SetConv inset; the complete notation is defined in Section 3.4.

For an initialization time $t _ { 0 } .$ the model receives the input AGRI sequence $X _ { t _ { 0 } } ^ { \mathrm { s a t } }$ defined in Section 2.5 and a set of target stations $S ^ { \mathrm { t a r } }$ . Let $\mathcal { R } = \{ 4 0 , 2 0 , 1 0 \}$ km and $\mathcal { Q } = \{ 2 0 , 5 0 \}$ mm $\mathrm { h } ^ { - 1 }$ denote the neighborhood radii and rainfall thresholds. The complete model mapping is expressed as

$$
\begin{array} { r l } & { \left[ \widehat { \mathbf { P } } _ { 1 } ( t _ { 0 } ) , \widehat { \mathbf { P } } _ { 2 } ( t _ { 0 } ) , \widehat { \mathbf { P } } _ { 3 } ( t _ { 0 } ) \right] = \mathrm { M A G P I E } _ { \Theta } \left( X _ { t _ { 0 } } ^ { \mathrm { s a t } } , \mathcal { S } ^ { \mathrm { t a r } } \right) , } \\ & { \qquad \widehat { \mathbf { P } } _ { k } ( t _ { 0 } ) = \left\{ \widehat { p } _ { s , k } ^ { R , q } ( t _ { 0 } ) ~ \Big | ~ s \in \mathcal { S } ^ { \mathrm { t a r } } , R \in \mathcal { R } , q \in \mathcal { Q } \right\} , \qquad k = 1 , 2 , 3 . } \end{array}\tag{9}
$$

For each forecast window $H _ { k } , \widehat { \mathbf { P } } _ { k } ( t _ { 0 } )$ contains event probabilities for every target station and radius–threshold combination. The 40 km/20 mm h<sup>−1</sup> head corresponds to the primary event definition, while the remaining five heads represent stricter spatial or rainfall-intensity criteria. Each $\hat { p } _ { s , k } ^ { R , q } ( t _ { 0 } )$ lies in [0, 1]. Binary event predictions are obtained by applying decision thresholds selected on the validation set.

The CI feature-construction module identifies rapid cloud development in the AGRI sequence and represents its spatial extent and timing. The multiscale satellite encoder combines the original AGRI channels with the CI features to derive a regional satellite representation. The auxiliary diagnosis head projects this representation onto three gridded precipitation fields. GA-SetConv maps the gridded features to station-local representations and incorporates geographic attributes and cyclical time encodings. The multi-head event prediction module converts these representations into event probabilities.

## 3.2 Convection-Initiation Feature Construction

Rapid cloud-top cooling, together with spectral changes in infrared and water-vapor channels, is commonly associated with developing convection (Mecikalski and Bedka, 2006; Mecikalski et al., 2010; Walker et al., 2012). MAGPIE-Net represents these signals using motion-compensated pixel-level convection-initiation (CI) masks that identify localized regions of rapid cloud development. The detection criteria are based on the physical indicators used by Peng et al. (Peng et al., 2025), with relaxed thresholds to retain a broader candidate set.

CI masks are constructed for the two candidate times $\tau \in \{ t _ { 0 } - 3 0 \mathrm { m i n } , t _ { 0 } - 1 5 \mathrm { m i n } \}$ . For each τ, the AGRI observations at $\tau - 3 0 , \tau - 1 5$ , and τ are used. Bidirectional motion fields are estimated from the 10.7 µm images using the Dense Inverse Search (DIS) optical-flow algorithm in OpenCV with the medium preset. These fields trace each pixel at τ through the two preceding scans, so cooling is calculated along the estimated cloud trajectory rather than between fixed grid cells. Candidate pixels must also satisfy forward–backward flow consistency and displacement limits before the spectral and cooling criteria in Table 2 are applied.

Table 2: Parameters and criteria for constructing CI candidate masks.
<table><tr><td>Item</td><td>Specification or condition</td></tr><tr><td>Candidate times</td><td> $\tau = t _ { 0 } - 3 0$  min and  $t _ { 0 } - 1 5 \mathrm { m i n } .$  each evaluated using the three scans at  $\tau - 3 0 , \tau - 1 5 ,$  and τ</td></tr><tr><td>Motion consistency</td><td>Backward trajectories remain within the grid, displacement in each 15-min step is at most 12 pixels, and forward–backward error is at most 1.5 pixels</td></tr><tr><td>Trajectory notation</td><td> $\widetilde { T } _ { 1 0 . 7 }$  denotes a temperature sampled along the backward-traced cloud trajectory</td></tr><tr><td>Cold cloud top</td><td> $T _ { 1 0 . 7 } ( \tau , g ) < 2 7 3 \mathrm { K }$ </td></tr><tr><td>Continuous cooling</td><td> $\widetilde { T } _ { 1 0 . 7 } ( \tau - 1 5 , g ) - \widetilde { T } _ { 1 0 . 7 } ( \tau - 3 0 , g ) \leq - 4 \mathrm { K }$  and  $T _ { 1 0 . 7 } ( \tau , g ) - \widetilde { T } _ { 1 0 . 7 } ( \tau - 1 5 , g ) \leq$   $- 4 \mathrm { K }$ </td></tr><tr><td>Water-vapor-infrared difference</td><td> $T _ { 7 . 1 } ( \tau , g ) - T _ { 1 0 . 7 } ( \tau , g ) > - 3 0 \mathrm { K }$ </td></tr><tr><td>Split-window difference Three-channel combination</td><td> $T _ { 1 2 . 0 } ( \tau , g ) - T _ { 1 0 . 7 } ( \tau , g ) > - 3 \mathrm { K }$ </td></tr><tr><td></td><td> $T _ { 8 . 5 } ( \dot { \tau } , g ) + T _ { 1 2 . 0 } ( \dot { \tau } , g ) - 2 T _ { 1 0 . 7 } ( \tau , g ) > - 5 \mathrm { K }$ </td></tr><tr><td>Spatial coherence</td><td>The connected candidate component contains at least 4 pixels and covers at least  $6 4 \mathrm { k m ^ { 2 } }$ </td></tr></table>

The relatively permissive thresholds favor candidate retention, so some identified cores may not subsequently produce heavy rainfall. The masks serve as auxiliary inputs rather than event labels. Evaluated together with the full multispectral context, they allow the encoder to distinguish candidates associated with subsequent station-neighborhood heavy rainfall from less informative cloud-development signals.

The candidate masks obtained from the input AGRI sequence are merged into a single field:

$$
C _ { t _ { 0 } } ( g ) = \operatorname* { m a x } _ { \tau } I _ { \tau } ( g )\tag{10}
$$

where $I _ { \tau } ( g )$ denotes the motion-compensated pixel-level CI mask at candidate time $\tau .$ The merged field $C _ { t _ { 0 } }$ marks locations where rapid cloud development is detected at either of the two candidate times.

Cloud-top features and the associated surface-rainfall area may separate as convective systems grow and move (Chen et al., 2025). A Gaussian filter is therefore applied to extend the candidate signal into the surrounding cloud region. The smoothed signal is represented by a three-channel field aligned with the three forecast windows:

$$
H _ { t _ { 0 } } = \mathrm { R e p } _ { 3 } \left\{ \mathrm { N o r m } \left[ \mathcal { K } _ { \sigma } * C _ { t _ { 0 } } \right] \right\} \in \mathbb { R } ^ { 3 \times H \times W }\tag{11}
$$

where Norm[·] scales the smoothed field to [0, 1], and $\mathrm { R e p } _ { 3 }$ repeats the same normalized field across three channels, one for each forecast window. A Gaussian kernel with a radius of 10 pixels and $\sigma = 5$ pixels is used. These channels provide the shared encoder with a common representation of the spatial context surrounding CI candidates. Forecast-window-specific responses to this context are learned by the subsequent preprocessing and station-decoding branches.

A timing field $A _ { t _ { 0 } }$ records the most recent CI detection at each grid cell. Its value is $2 / 3$ for a most recent detection at $t _ { 0 } - 3 0$ min, 1 for a detection at $t _ { 0 } - 1 5$ min, and 0 where neither mask contains a candidate. The merged candidate field, the spatially enhanced field, and the timing field are concatenated to form the CI feature set:

$$
Z _ { t _ { 0 } } ^ { \operatorname { C I } } = { \operatorname { C o n c a t } } _ { \operatorname { c h } } \left( C _ { t _ { 0 } } , H _ { t _ { 0 } } , A _ { t _ { 0 } } \right) \in \mathbb { R } ^ { 5 \times H \times W }\tag{12}
$$

## 3.3 Multiscale Encoding with Auxiliary Precipitation Supervision

Convective cloud and precipitation systems span a wide range of spatial scales. Rapidly cooling cloud tops and newly formed convective cores are localized, whereas cloud clusters, anvils, and the surrounding moisture environment extend over much larger areas (Mecikalski and Bedka, 2006; Houze, 2014). The encoder must therefore retain fine-scale cloud-development features while also representing the broader environment that afects storm growth and rainfall organization.

The input AGRI sequence and CI features are concatenated along the channel dimension:

$$
\widetilde { X } _ { t _ { 0 } } = \mathrm { C o n c a t _ { c h } } \left( X _ { t _ { 0 } } ^ { \mathrm { s a t } } , Z _ { t _ { 0 } } ^ { \mathrm { C I } } \right) \in \mathbb { R } ^ { 3 7 \times H \times W }\tag{13}
$$

The multiscale encoder follows a UNet++ architecture with nested skip connections (Ronneberger et al., 2015; Zhou et al., 2018). It uses four resolution levels with 64, 128, 256, and 512 feature channels. The nested skip connections combine broad cloud and moisture patterns from deeper levels with higher-resolution cloud structures to produce the regional satellite representation

$$
F _ { t _ { 0 } } = E _ { \theta } \left( \widetilde { X } _ { t _ { 0 } } \right)\tag{14}
$$

The regional representation is projected onto gridded precipitation diagnosis fields for the three forecast windows through an auxiliary $1 \times { \dot { 1 } }$ regression head:

$$
\widehat { \mathbf { G } } ( t _ { 0 } ) = D _ { \phi } \left( F _ { t _ { 0 } } \right) = \left[ \hat { G } _ { 1 } ( t _ { 0 } ) , \hat { G } _ { 2 } ( t _ { 0 } ) , \hat { G } _ { 3 } ( t _ { 0 } ) \right] \in \mathbb { R } ^ { 3 \times H \times W }\tag{15}
$$

Each diagnosis field represents precipitation-related spatial structure for one forecast window. Together, the three fields provide the gridded input to the station decoder.

The auxiliary reference field $G _ { k } ^ { * } ( t _ { 0 } )$ is generated from station precipitation observations through Barnes objective analysis (Barnes, 1964) on the same 0.04<sup>◦</sup> grid as the satellite input. This spatially continuous target describes the location and broad organization of observed surface rainfall. Supervision against this target encourages the regional satellite representation to preserve precipitation-related spatial structure for subsequent station-neighborhood event prediction.

## 3.4 GA-SetConv: Geographically Adaptive Grid-to-Station Decoding

The encoder produces features on a regular grid, whereas the warning targets are defined at irregular station locations. GA-SetConv connects these representations through SetConv-style kernel aggregation, following permutation-invariant set representations and conditional neural processes for mappings between irregular samples and continuous fields (Zaheer et al., 2017; Garnelo et al., 2018; Gordon et al., 2020). Rainfall organization varies with geographic setting, while storm motion and vertical wind shear can displace surface rainfall from the corresponding cloud-top features (Houze, 2012, 2014; Chen et al., 2025). The required spatial support can therefore difer among target stations and forecast windows. GA-SetConv uses an axis-aligned learnable kernel whose east–west and north–south length scales are conditioned on the target-station attributes and forecast window.

The gridded precipitation diagnosis fields are first processed by a separate U-Net branch for each forecast window:

$$
U _ { k } ( t _ { 0 } ) = P _ { \eta _ { k } } \left( { \widehat { \mathbf { G } } } ( t _ { 0 } ) \right) , \qquad k = 1 , 2 , 3\tag{16}
$$

where $U _ { k } ( t _ { 0 } )$ contains the gridded features used to predict events in forecast window $H _ { k }$ . The three branches generate forecast-window-specific features, allowing the spatial mapping to represent changes in existing rainfall structure, storm movement, spatial expansion, and new-cell development across forecast windows.

Let $( x _ { g } , y _ { g } )$ and $( x _ { s } , y _ { s } )$ denote the normalized coordinates of grid cell $g$ and target station s, respectively. For each forecast window, a geographic scale network uses the station attributes $\boldsymbol { r _ { s } } = \left( \lambda _ { s } , \phi _ { s } , h _ { s } \right)$ defined in Section 2.3 to estimate the east–west and north–south length scales:

$$
\begin{array} { r } { \left[ \ell _ { x , s , k } \right] = \exp \left[ h _ { \psi _ { k } } ( r _ { s } ) \right] } \end{array}\tag{17}
$$

The corresponding kernel is

$$
K _ { s , k } ( g ) = \exp \left[ - \frac { ( x _ { g } - x _ { s } ) ^ { 2 } } { 2 \ell _ { x , s , k } ^ { 2 } } - \frac { ( y _ { g } - y _ { s } ) ^ { 2 } } { 2 \ell _ { y , s , k } ^ { 2 } } \right]\tag{18}
$$

The two length scales independently control the east–west and north–south support. The aggregation region can therefore vary by target station and forecast window while remaining axis aligned.

For each target station and forecast window, the total kernel weight and normalized station-local feature are

$$
\begin{array} { r l } & { { d _ { s , k } = \displaystyle \sum _ { g \in \mathcal { G } } K _ { s , k } ( g ) } , } \\ & { { { \cal V } _ { s , k } = \displaystyle \frac { \sum _ { g \in \mathcal { G } } K _ { s , k } ( g ) U _ { k } ( t _ { 0 } , g ) } { d _ { s , k } + \epsilon } } . } \end{array}\tag{19}
$$

Dividing by the total kernel weight keeps the magnitude of $V _ { s , k }$ comparable as the learned aggregation region expands or contracts. The total weight $d _ { s , k }$ is retained as an additional input, allowing the station decoder to account for diferences in the efective spatial support.

The normalized station-local feature $V _ { s , k }$ and total kernel weight $d _ { s , k }$ are combined with the station attributes $r _ { s }$ and the cyclical encoding $\gamma _ { t _ { 0 } , k }$ of the valid time:

$$
Z _ { s , k } ^ { \mathrm { s t a } } = M _ { \xi _ { k } } \left( V _ { s , k } , d _ { s , k } , r _ { s } , \gamma _ { t _ { 0 } , k } \right)\tag{20}
$$

The station attributes represent persistent spatial diferences through longitude, latitude, and elevation, whereas the cyclical encoding describes seasonal and diurnal timing (Chen et al., 2013). Together with $V _ { s , k }$ and $d _ { s , k }$ , these variables form the station-local representation for each forecast window.

The station-local representation is passed to six event heads:

$$
\begin{array} { r l r } { \ell _ { s , k } ^ { R , q } = C _ { R , q } \left( Z _ { s , k } ^ { \mathrm { s t a } } \right) , } & { } & \\ { \hat { p } _ { s , k } ^ { R , q } ( t _ { 0 } ) = \sigma \left( \ell _ { s , k } ^ { R , q } \right) , } & { } & { R \in \mathcal { R } , \ q \in \mathcal { Q } . } \end{array}\tag{21}
$$

The 40 km/20 mm $\mathrm { h } ^ { - 1 }$ head produces the primary station-neighborhood heavy-rainfall event probability. The other five heads correspond to smaller neighborhoods and the higher rainfall threshold, providing additional constraints on spatial localization and rainfall intensity. A parallel regression branch supplies continuous station-rainfall information during training.

GA-SetConv therefore converts the gridded precipitation features derived from the shared regional representation into station-local representations, which the multi-head event prediction module converts into event probabilities at irregular station locations. Its axis-aligned spatial support adapts independently in the east–west and north–south directions for each target station and forecast window. Each forecast-window preprocessing branch uses a two-level U-Net with a base width of 32. The geographic scale network has two hidden layers with 32 units each, and the station decoder produces a 64-dimensional representation before the event heads.

## 3.5 Multi-Task Training Objective

The final warning target is defined by station-neighborhood event labels, making the event loss the principal station-level supervision. Because binary labels describe threshold exceedance but not rainfall intensity or the spatial organization of the precipitation system, the training objective also includes gridded precipitationdiagnosis and station-rainfall regression losses. These auxiliary terms constrain regional rainfall structure and local intensity, respectively.

For neighborhood radius R and rainfall threshold q, the event loss is defined using positive-weighted binary cross-entropy. Following the notation in Section 2.4, the observed label for forecast window $H _ { k }$ is $\mathop { y _ { s } ^ { R , q } } ( t _ { 0 } + k \mathrm { h } )$

$$
\mathcal { L } _ { R , q } = \frac { \sum _ { t _ { 0 } , s , k } M _ { s , k } ( t _ { 0 } ) \mathrm { B C E } _ { \alpha } \left( \ell _ { s , k } ^ { R , q } ( t _ { 0 } ) , y _ { s } ^ { R , q } ( t _ { 0 } + k \mathrm { h } ) \right) } { \sum _ { t _ { 0 } , s , k } M _ { s , k } ( t _ { 0 } ) + \epsilon }\tag{22}
$$

where $M _ { s , k } ( t _ { 0 } )$ indicates whether the corresponding station label is available, and

$$
\mathrm { B C E } _ { \alpha } ( \ell , y ) = - \alpha y \log \sigma ( \ell ) - ( 1 - y ) \log \left[ 1 - \sigma ( \ell ) \right]\tag{23}
$$

The positive-event weight is set to $\alpha = 2$ . Together with the downsampling of event-free training samples described in Section 2.5, this weighting gives greater influence to the less frequent positive events during optimization.

The losses from the six event heads are averaged:

$$
{ \overline { { \mathcal { L } } } } _ { \mathrm { e v e n t } } = { \frac { 1 } { 6 } } \sum _ { R \in { \mathcal { R } } } \sum _ { q \in { \mathcal { Q } } } { \mathcal { L } } _ { R , q }\tag{24}
$$

Joint optimization of the six event heads provides the shared station representation with supervision across neighborhood scales and rainfall thresholds, allowing the same representation to support all six event definitions.

The gridded diagnosis loss is

$$
\mathcal { L } _ { \mathrm { g r i d } } = \frac { \sum _ { t _ { 0 } , k , g } M _ { k } ^ { \mathrm { g r i d } } ( t _ { 0 } , g ) \left[ \hat { G } _ { k } ( t _ { 0 } , g ) - G _ { k } ^ { * } ( t _ { 0 } , g ) \right] ^ { 2 } } { \sum _ { t _ { 0 } , k , g } M _ { k } ^ { \mathrm { g r i d } } ( t _ { 0 } , g ) + \epsilon }\tag{25}
$$

where $M _ { k } ^ { \mathrm { g r i d } } ( t _ { 0 } , g )$ indicates whether the gridded reference value is valid. The masked loss is evaluated only where the Barnes reference field is valid and constrains the intermediate diagnosis fields and shared regional representation to preserve observed precipitation structure.

A parallel station-rainfall branch predicts the 1 h accumulated precipitation at each supervised station. The corresponding loss is

$$
\mathcal { L } _ { \mathrm { r a i n } } = \frac { 1 } { N _ { \mathrm { r a i n } } } \sum _ { t _ { 0 } , s , k } M _ { s , k } ^ { \mathrm { r a i n } } ( t _ { 0 } ) \left[ \widehat { P } _ { s , k } ^ { \mathrm { s t a } } ( t _ { 0 } ) - P _ { s } ( t _ { 0 } + k \mathrm { h } ) \right] ^ { 2 }\tag{26}
$$

where $P _ { s } ( t _ { 0 } + k \mathbf { h } )$ is the observed 1 h accumulated precipitation defined in Section 2.3, $M _ { s , k } ^ { \mathrm { r a i n } } ( t _ { 0 } )$ indicates the availability of the corresponding observation, and $N _ { \mathrm { r a i n } }$ is the number of valid station–time rainfall observations. This branch retains variations in local rainfall intensity that are not represented by the binary event labels.

The station-related loss and the complete training objective are

$$
\begin{array} { r } { \begin{array} { c } { \mathscr { L } _ { \mathrm { s t a t i o n } } = \displaystyle \frac { 2 \overline { { \mathscr { L } } } _ { \mathrm { e v e n t } } + \mathscr { L } _ { \mathrm { r a i n } } } { 3 } , } \\ { \mathscr { L } = w _ { \mathrm { g r i d } } \mathscr { L } _ { \mathrm { g r i d } } + w _ { \mathrm { s t a t i o n } } \mathscr { L } _ { \mathrm { s t a t i o n } } . } \end{array} } \end{array}\tag{27}
$$

The factor of 2 gives the aggregated event loss twice the weight of station-rainfall regression within the stationrelated objective. Both $w _ { \mathrm { g r i d } }$ and $w _ { \mathrm { s t a t i o n } }$ are set to 1. Together, the three losses supervise complementary aspects of the model: the gridded loss constrains regional precipitation structure, the station-rainfall loss preserves local intensity variation, and the event loss aligns the complete satellite-to-station pathway with the warning target.

## 4 Results

The experiments address two questions: whether the pre-initialization AGRI sequence contains information relevant to subsequent station-neighborhood heavy rainfall, and whether MAGPIE-Net converts this information into local event probabilities more efectively than gridded-output baselines.

## 4.1 Experimental Setup and Evaluation Protocol

MAGPIE-Net was trained for 15 epochs with a batch size of 2 using the AdamW optimizer. The initial learning rate was $1 \times 1 0 ^ { - 4 }$ , and the weight decay was $1 \times 1 0 ^ { - 3 }$ . The learning rate was halved after two consecutive epochs without an improvement in validation loss, and the gradient norm was clipped at 5. Model parameters from the epoch with the lowest validation loss in 2022 were used for evaluation on the independent 2023 test set.

All models receive the same four-scan FY-4A AGRI sequence and use the same year-based data split. The adapted Earthformer (Gao et al., 2022), PhyDNet (Le Guen and Thome, 2020), NPM (Park et al., 2025), and NowcastNet (Zhang et al., 2023) baselines follow the conventional gridded-output pathway shown in Fig. 1 and predict gridded 1 h precipitation for the three forecast windows. MAGPIE-Net internally derives the CI features described in Section 3.2 from the same AGRI sequence and directly predicts station-neighborhood heavy-rainfall event probabilities.

To compare the models at the final warning target, each gridded precipitation forecast is converted into a station-neighborhood event using the same radius-based neighborhood aggregation:

$$
\begin{array} { r l } & { \hat { G } _ { s , k } ^ { R , \operatorname* { m a x } } ( t _ { 0 } ) = \underset { \scriptstyle d _ { \mathrm { h a v } } ( g , s ) \leq R } { \operatorname* { m a x } } \quad \hat { G } _ { k } ( t _ { 0 } , g ) , } \\ & { } \\ & { \hat { y } _ { s , k } ^ { R , q , \mathrm { g r i d } } ( t _ { 0 } ) = \left\{ \begin{array} { l l } { 1 , } & { \hat { G } _ { s , k } ^ { R , \operatorname* { m a x } } ( t _ { 0 } ) \geq 2 0 , \quad q = 2 0 , } \\ { 1 , } & { \hat { G } _ { s , k } ^ { R , \operatorname* { m a x } } ( t _ { 0 } ) > 5 0 , \quad q = 5 0 , } \\ { 0 , } & { \mathrm { o t h e r w i s e } . } \end{array} \right. } \end{array}\tag{28}
$$

Here, $d _ { \mathrm { h a v } } ( \boldsymbol { g } , \boldsymbol { s } )$ is the haversine distance between grid cell $g$ and target station s. The six event definitions combine $R \in \{ 4 0 , 2 0 , 1 0 \}$ km with $q \in \{ 2 0 , 5 0 \} \mathrm { m m } \tilde { \mathrm { h } } ^ { - 1 }$ . The gridded-output baselines retain physical rainfall units, so their neighborhood maxima are compared directly with $q .$ The maximum is taken over the full verification radius rather than the smaller local neighborhood typically used for operational grid-to-station mapping. This gives the gridded baselines the same spatial tolerance as the event definition: thresholdexceeding precipitation predicted anywhere within R produces a positive forecast for the target station. MAGPIE-Net directly predicts event probabilities and therefore requires probability decision thresholds. For each event head, the threshold was selected by maximizing CSI on the 2022 validation set, shared across the three forecast windows, and then fixed for evaluation on the independent 2023 test set.

Performance at individual station–time instances is evaluated using the critical success index (CSI), probability of detection (POD), and false alarm ratio (FAR):

$$
\mathrm { C S I } = \frac { T P } { T P + F P + F N } , \qquad \mathrm { P O D } = \frac { T P } { T P + F N } , \qquad \mathrm { F A R } = \frac { F P } { T P + F P }\tag{29}
$$

where $T P , F P ,$ and FN denote hits, false alarms, and misses, respectively. The counts are accumulated over all valid station–time instances for each event definition and forecast window.

Episode-level verification uses the primary 40 km $1 / 2 0 \mathrm { m m h ^ { - 1 } }$ event definition, with each positive label representing an event in the 1 h window ending at its valid time. Because the labels are updated every 15 min, consecutive positives at the same station during sustained rainfall are grouped into one heavy-rainfall episode. The reference time $t _ { e }$ is the end of its first positive window. An episode is considered detected if, no later than $t _ { e } ,$ , the model issues a positive forecast for any 1 h event window belonging to that episode. For each detected episode, the chronologically earliest episode window with a positive forecast and the earliest initialization available for that window are retained; warning lead time is measured from that initialization to $t _ { e }$ . Undetected episodes are assigned a lead time of 0 min when calculating the mean episode warning lead time.

Antecedent rainfall characterizes the surface-rainfall state immediately before episode onset. For each episode, $P _ { \mathrm { a n t e } } = P _ { s } ^ { 4 0 , \operatorname* { m a x } } ( t _ { e } - 1 \mathrm { h } )$ is the maximum 1 h accumulated rainfall within 40 km during the preceding nonoverlapping hour. The three below-threshold groups, $P _ { \mathrm { a n t e } } < 1 \ \mathrm { m m } , 1 \leq P _ { \mathrm { a n t e } } < 5 \ \mathrm { m m }$ , and $5 \leq P _ { \mathrm { a n t e } } < 2 0$ mm, contain 9,843, 10,838, and 23,121 episodes, respectively. Another 8,480 episodes have $P _ { \mathrm { a n t e } } \geq 2 0$ mm; in these cases, rainfall had already reached the primary event threshold before at least one subsequent 15-min label fell below the threshold and was followed by renewed exceedance. These episodes are excluded from the stratified analysis because they do not begin from a below-threshold antecedent state.

## 4.2 Primary Station-Neighborhood Warning Performance

As shown in Fig. 4, MAGPIE-Net achieves CSI values of 0.371, 0.304, and 0.238 for the 0–1, 1–2, and 2–3 h forecast windows, respectively, under the primary event definition. The corresponding values for the best-performing gridded-output baseline, NowcastNet, are 0.167, 0.107, and 0.069. All models lose skill as the forecast lead time increases, consistent with growing uncertainty in storm movement, convective growth and decay, and new-cell development. MAGPIE-Net retains a larger proportion of its 0–1 h CSI than NowcastNet, and its relative CSI advantage therefore widens across the three forecast windows. Because all models receive the same four direct AGRI scans, this comparison evaluates how their respective prediction pathways convert cloud evolution into final warning decisions.

The POD and FAR show how this diference changes with forecast lead time. During the first forecast hour, the higher POD of MAGPIE-Net is accompanied by a moderate increase in FAR. Rapid cloud-top cooling and cloud expansion often accompany active convective growth, although some developing clouds later weaken or produce only limited surface rainfall. During the 1–2 h and 2–3 h windows, MAGPIE-Net has both substantially higher POD and lower FAR than NowcastNet. The event-oriented pathway therefore maintains a more favorable balance between missed events and false alarms as uncertainty in rainfall location and intensity increases.

At the episode level, MAGPIE-Net detects 65.1% of the observed heavy-rainfall episodes, compared with 23.6% for NowcastNet. The mean episode warning lead time is 64.6 min for MAGPIE-Net and 18.3 min for NowcastNet. Together, the higher episode detection rate and longer episode warning lead time show that MAGPIE-Net can use the input AGRI sequence to anticipate impending heavy-rainfall episodes within target-station neighborhoods.

![](images/ac2670ff013578b4746b51b1bb2eb6016d4c76cff3f4641203e1d7141aec6a71.jpg)

![](images/937fdf9ca6ad2053513475a806f669f823824578bc9671b48b59e65bd491a85d.jpg)

![](images/affbf76378b40ef72dfa4c1033db3e3b3d83d6d7ad9ba39ec108fd5756d54232.jpg)

(d) Episode Detection Rate (e) Mean Episode Warning Lead Time  
![](images/a27c9ada758d87afa1983970db596cab60c944345f67bc37954a213bc4dd1ced.jpg)

![](images/86edb66e53dff14b1d6b2237be641f6ab1294c94c947a620b8884a5f4c8d58b8.jpg)  
Figure 4: Station-neighborhood warning performance under the primary 40 km $/ 2 0$ mm $\mathrm { h } ^ { - 1 }$ event definition. Gridded-output predictions are converted into station-neighborhood events using the same radius-based neighborhood aggregation. (a)–(c) Sample-level CSI, POD, and FAR for the three forecast windows. (d) Episode detection rate. (e) Mean episode warning lead time, with undetected episodes assigned 0 min.

The antecedent-rainfall analysis examines how episode-level performance changes with the amount of surface rainfall present before the first threshold exceedance (Fig. 5). Both models improve as $P _ { \mathrm { a n t e } }$ increases. This pattern is consistent with greater persistence and spatial organization of the cloud–rain system after surface rainfall begins to develop, providing stronger cues for identifying the subsequent threshold exceedance.

The $P _ { \mathrm { a n t e } } < 1$ mm class represents an important early-warning stage in which substantial surface rainfall has not yet developed. Among the 9,843 episodes in this class, MAGPIE-Net detects 51.9% and provides a mean episode warning lead time of 38.5 min. NowcastNet detects 18.6% of the same episodes, with a mean episode warning lead time of 11.5 min. This result shows that MAGPIE-Net can use cloud and moisture evolution observed by AGRI before forecast initialization to anticipate the rapid transition from little or no rainfall to heavy rainfall within the target-station neighborhood.

Across all three antecedent-rainfall classes, the episode detection rate of MAGPIE-Net is approximately 2.8 times that of NowcastNet. Its mean episode warning lead-time advantage increases from 27.0 min for $P _ { \mathrm { a n t e } } < 1$ mm to 51.0 min for $5 \leq P _ { \mathrm { a n t e } } < 2 0$ mm. The advantage is therefore evident across all three below-threshold antecedent states, from near-zero surface rainfall to rainfall approaching the event threshold.

![](images/db1105f5e155e56d7c0e8a3344dc301e486fc9fe9d368d2f5170ac3180eb70be.jpg)  
NowcastNet

![](images/c9beaa3101668979980c44e9b5b9ceba04164c7bf1fdd46645b1df3b57808d77.jpg)  
Figure 5: Episode-level warning performance stratified by antecedent rainfall within the 40-km neighborhood under the primary event definition. $P _ { \mathrm { a n t e } }$ denotes the maximum 1 h accumulated rainfall during the nonoverlapping hour immediately preceding the first threshold-exceeding window. (a) Episode detection rate. (b) Mean episode warning lead time, with undetected episodes assigned 0 min. Numbers beneath the categories indicate the observed episode counts.

## 4.3 Contribution of the Event-Oriented Pathway and GA-SetConv

Three configurations are evaluated with MAGPIE-Net as the reference. MAGPIE-Net is the complete model. The grid-trained backbone uses the same UNet++ multiscale satellite encoder as MAGPIE-Net but is optimized only against gridded precipitation fields and contains no station-event prediction pathway. Its comparison with MAGPIE-Net evaluates the overall change from gridded-output training to the complete event-oriented satellite-to-station pathway. The bilinear event model retains the CI features, multiscale encoder, auxiliary gridded precipitation diagnosis, and multi-head station-event supervision of MAGPIE-Net, while replacing GA-SetConv with bilinear interpolation. Its comparison with MAGPIE-Net therefore isolates the contribution of GA-SetConv.

Table 3 reports CSI averaged over the three forecast windows. Relative to the grid-trained backbone, MAGPIE-Net increases the mean CSI across the six event definitions from 0.0154 to 0.1433, with improvements for every radius–threshold combination. This overall diference reflects the combined efect of the CI features, direct station-event supervision, and trainable GA-SetConv mapping within the complete event-oriented pathway. The improvement is particularly pronounced for the 50 mm h<sup>−1</sup> events, for which the CSI of the grid-trained backbone is close to zero. Extreme hourly rainfall is often concentrated in localized high-intensity rain cores, so a small error in the position or intensity of a thresholded gridded forecast can change the resulting station-neighborhood classification. Direct event supervision and trainable spatial conversion allow the satellite representation to be optimized against this final classification target.

Replacing bilinear interpolation with GA-SetConv further increases the overall mean CSI from 0.1331 to 0.1433, corresponding to a relative gain of 7.7%. Improvements occur under all six event definitions and become larger for smaller neighborhoods and the higher rainfall threshold. These stricter definitions are more sensitive to spatial displacement between cloud and rainfall structures and to local geographic conditions. Unlike the fixed local mapping used by bilinear interpolation, GA-SetConv adjusts its axis-aligned east–west and north–south spatial support for each target station and forecast window.

Table 3: Controlled comparison of the event-oriented pathway and grid-to-station decoding. Values are CSI averaged over the three forecast windows.
<table><tr><td>Model</td><td>40/20</td><td>20/20</td><td>10/20</td><td>40/50</td><td>20/50</td><td>10/50</td><td>Mean</td></tr><tr><td>Grid-trained backbone</td><td>0.0456</td><td>0.0291</td><td>0.0165</td><td>0.0007</td><td>0.0004</td><td>0.0002</td><td>0.0154</td></tr><tr><td>Bilinear event model</td><td>0.2873</td><td>0.1802</td><td>0.1086</td><td>0.1226</td><td>0.0703</td><td>0.0294</td><td>0.1331</td></tr><tr><td>MAGPIE-Net</td><td>0.3043</td><td>0.1909</td><td>0.1183</td><td>0.1315</td><td>0.0772</td><td>0.0376</td><td>0.1433</td></tr></table>

The notation 40/20 denotes 40 km $/ 2 0 \mathrm { m m h } ^ { - 1 }$ , and the remaining columns follow the same convention.

Taken together, the controlled configurations show that the complete event-oriented pathway establishes the main skill gain over gridded-output training, while GA-SetConv provides a consistent additional improvement across all six event definitions.

## 4.4 Performance Across Event Definitions and Spatiotemporal Subsets

## 4.4.1 Neighborhood Radius and Rainfall Threshold

The results in Fig. 6 indicate that MAGPIE-Net achieves the highest CSI across all 18 combinations of $R = 4 0 , 2 0$ , and 10 km, $q = 2 0$ and $\mathrm { 5 0 m m h ^ { - 1 } }$ , and the three forecast windows. Its advantage extends from the $4 0 \mathrm { k m / 2 0 \mathrm { m m h ^ { - 1 } } }$ definition to the localized $1 0 \mathrm { k m / 5 0 m m h ^ { - 1 } }$ definition, showing that the event-oriented pathway maintains a consistent advantage as the warning target requires more precise spatial localization and greater rainfall intensity.

CSI decreases for all models as the neighborhood radius becomes smaller and the rainfall threshold becomes higher, reflecting the increasing dificulty of the task. A 40-km neighborhood accommodates some displacement among the cloud-top structures observed by the satellite, the evolving surface-rainfall area, and the target station. At a radius of 10 km, even small location errors can change the event classification of short-lived convective rain cores. Extreme hourly rainfall at $\mathrm { 5 0 m m h ^ { - 1 } }$ is usually confined to a small area and depends on the local co-occurrence of strong ascent, abundant moisture, and eficient cloud microphysical processes. The coldest or most rapidly expanding cloud region does not necessarily coincide with the maximum surface rainfall (Chen et al., 2025). The CSI variation under stricter event definitions therefore reflects both event rarity and spatial-scale diferences between cloud-top radiative signals and surface rainfall.

## 4.4.2 Temporal and Regional Stratification

As shown in Fig. 7, daytime samples, defined as 06:00–18:00 Beijing Time (BJT), have slightly higher CSI and POD than nighttime samples and contain approximately twice as many positive events under the primary event definition. The higher daytime scores coincide with both the larger number of positive events and meteorological conditions that often favor locally initiated convection (Chen et al., 2013; Luo et al., 2017). Successive infrared observations can track the rapidly evolving cloud structures associated with such development.

After sunset, MAGPIE-Net retains station-neighborhood warning performance across all three forecast windows. Infrared and water-vapor channels continue to observe cloud-top and moisture evolution in nocturnal mesoscale convective systems, coastal convection, and tropical-cyclone rainbands (Chen et al., 2013; Lonfat et al., 2004). The observed day–night diferences occur alongside changes in event frequency, convective organization, and the relationship between upper-cloud evolution and surface rainfall.

Performance is strongest in July and August, when positive events are most frequent and the episode detection rates reach 66.4% and 70.5%, respectively. These months combine larger positive-event samples with frequent deep convection, organized mesoscale systems, and tropical-weather influences that often produce persistent and spatially coherent upper-cloud structures (Chen et al., 2013; Lonfat et al., 2004). May, June, and September contain fewer positive events and a broader mixture of frontal, warm-sector, and terrain-influenced rainfall (Luo et al., 2017; Houze, 2012). Across all five months, episode detection ranges from 48.0% in September to 70.5% in August, and the mean warning lead time ranges from 47.0 to 74.5 min.

![](images/1fd37b6de5d0d97be47411eb211afb5e831d16bed6d9d8406fc986341dd5a95a.jpg)

![](images/6efa772ff4fd317b59a4bfb54922b3708f7839e95a57161daa95563cd8b6bc83.jpg)

![](images/d83602ba8d01a2fcc99acbbd19beba9a7303e7218bb6b383ed02c49e4d8af40f.jpg)

![](images/e9c104990183d2f916d1c0fe96e44747018d3b2e5572bdaf419ef86104912a0b.jpg)

![](images/72223ea5f54259e7ec647f512dd9b80af5babaab7069235de39ea4a7bb951eb6.jpg)

![](images/824e2963ca0c911d368106f9a1883a1dd06b098673db059af675da8c50d124b6.jpg)  
Figure 6: Cross-model CSI across neighborhood radii and rainfall thresholds. Gridded-output predictions are evaluated using the same radius-based neighborhood aggregation as the station-neighborhood event definition. Panels (a)–(c) correspond to 20 mm h<sup>−1</sup>, and panels (d)–(f) correspond to 50 mm $\mathrm { h } ^ { - 1 }$ . Columns represent neighborhood radii of 40, 20, and 10 km. Each panel shows the 0–1, 1–2, and 2–3 h forecast windows.

Regional diferences are moderate. The Southeast Coastal Belt has the highest CSI and lowest FAR across all three forecast windows, while its POD remains comparable to the leading regional values. The Mountain Belt has lower POD and higher FAR at longer lead times. These contrasts are consistent with regional diferences among organized coastal and tropical-cyclone rainbands, terrain-influenced precipitation, and warm-sector rainfall, which can alter the spatial correspondence between upper-cloud evolution and surface rainfall (Luo et al., 2017; Lonfat et al., 2004; Houze, 2012). Despite these regional variations, the decline in warning skill with forecast lead time is similar across all five partitions.

## 4.5 CI-Feature Ablation and Spatial Attribution

Two experiments separated the contribution of CI features to warning performance from their spatial influence on the trained model. In the retraining ablation, MAGPIE-Net was trained without the five CI feature channels while retaining the same input AGRI sequence, event heads, and GA-SetConv decoder. The model with CI features detected 34,053 episodes, with an episode detection rate of 65.1% and a mean warning lead time of 64.6 min. The retrained model without CI detected 29,838 episodes, with a detection rate of 57.1% and a mean warning lead time of 59.2 min (Fig. 8a–c). Thus, including CI features allowed the model to detect additional heavy-rainfall episodes and provide earlier warnings.

An inference-time masking experiment then examined where CI information afected the output of the trained model. MAGPIE-Net was evaluated with the full input and the CI-masked input while its parameters and all other inputs remained unchanged. The event-probability diference was largest near the CI core and decreased with distance (Fig. 8d–i). This response was strongest in the 0–1 h window and weakened with forecast lead time. The mean diference for positive station–time events was approximately six to ten times that for negative events. The localized radial response and the stronger response for positive events indicate that the model used CI features selectively rather than raising probabilities uniformly around every CI candidate. This pattern is consistent with the permissive CI construction, in which the candidate masks identify possible regions of rapid cloud development and the surrounding AGRI context helps distinguish candidates associated with subsequent heavy rainfall.

![](images/beac87b34405675d67873962121ba106052a8760f6688d9b5fc09b242639ef88.jpg)

![](images/445fd29aa9a8d0a45bb0bcc93657fa2e6a931e24cb9498c2916f31251d5d382d.jpg)

![](images/f002dab1769e8d936fb41b95440bf0fdd6c681f4ae6d90f8b8cd0d8e8fd8125a.jpg)

![](images/7c4b8278974b6a1a6b61786169c08eb57aabf900c33236a108587332b26828e3.jpg)

(b) Forecast window: 0-1 h  
![](images/d585e2c282ef87cee929fa338d3724ef74d6418fd9653fa821ca441259a58ba7.jpg)

![](images/451554f609e5dde8964cc1b99de974b3e3aa438ef9b76168ea75aac950b55cde.jpg)

![](images/3befabd082aa2337d95efed07a772e4ff29b9766a3223d81e86fd2b1085925fa.jpg)

![](images/10523cc231871a61c14eba885c4d24505384d4598b2c32f3e8c1443f2ddf6835.jpg)

(c)  
![](images/ce2384d718a8add12c435f3b829a3318a77cee55208523754fbe4312a28928a3.jpg)

![](images/f1cd7921026faf5d96627ac5931154ac63819d9e52a61ce813cd1a7d88d6dd1b.jpg)  
Figure 7: Performance of MAGPIE-Net across temporal and regional subsets under the primary event definition. (a) Sample-level CSI, POD, FAR, and positive-event counts for daytime and nighttime samples. (b) Monthly sample-level metrics and positive-event counts for the three forecast windows, together with episode detection rate and mean episode warning lead time. (c) Regional partition and sample-level CSI, POD, and FAR across the five regions.

(a)  
![](images/95c693b9f75cd3041513d5a904f444fdc8be678ad9c5ca6634894b26354fff6a.jpg)

(b)  
![](images/96753149bedc810de6028f270dd5554c57f6ca58e499b5fb97dc109e214e0d01.jpg)

![](images/d00e4c490f79c4edf84bf0c7574964d13f21153b466e7040b88bd40636d13e76.jpg)

![](images/bb3d75a55c2f702b6595557e7920cc53c0587733b87303927b39fc60083997f8.jpg)

![](images/da1c0401b27f3f0750a4e047389da3f4053e859d3ef9802cfb738e91db16b20a.jpg)

![](images/5035d522aafc0039e118fd6703679f4ca64ebb2a9572cc98780f51a3b786eb10.jpg)

![](images/287c3131ea24116ae20adc4405a8f35a145307abe1b9758344269216fe052d67.jpg)

(h)  
![](images/708a29fa745d08339faa992ab95ed17d7750c213c1a3ce53e39e9fafa0214ba9.jpg)

(i)  
![](images/26e774b8afafe3a02209d3bc89887e497d2d5e29b1770ff6964b1c8882817970.jpg)  
Figure 8: CI-feature ablation and inference-time spatial attribution under the primary event definition. (a)–(c) Detected episode count, episode detection rate, and mean warning lead time for independently trained MAGPIE-Net models without and with CI features. (d)–(f) CI-centered station-neighborhood event-probability profiles obtained with the full input and the CI-masked input. (g)–(i) Event-probability diferences between the full and CI-masked inputs; the insets compare the mean diferences for positive and negative station–time events.

## 4.6 Representative Heavy-Rainfall Cases

## 4.6.1 Rapid Local Convective Development From Near-Zero Antecedent Rainfall

Figure 9 presents a localized convective heavy-rainfall episode over coastal southern Guangxi, initialized at 22:00 UTC on 10 August 2023 (06:00 BJT on 11 August). During the non-overlapping hour before the first threshold-exceeding window, the maximum 1 h accumulated rainfall across the 40-km neighborhoods of the three stations in the highlighted component was only 0.5 mm. The latest input AGRI image showed a localized cold-cloud feature near the Beibu Gulf coast. All three stations in the highlighted component then became positive during the 1–2 h forecast window. The case therefore represents a rapid transition from near-zero antecedent surface rainfall to localized station-neighborhood heavy rainfall.

MAGPIE-Net detected all three newly positive stations, whereas PhyDNet, NPM, EarthFormer, and NowcastNet missed all three. Within the enlarged local domain, MAGPIE-Net produced three hits, no misses, and four false alarms, corresponding to a local CSI of 0.429; none of the gridded-output baselines produced a hit. The case provides a direct example of MAGPIE-Net identifying localized heavy-rainfall development from the evolving cloud field before substantial surface rainfall had formed.

Observed | 1-2 h

Observed | 0-1 h  
Observed | 2-3 h Observed zoom | 1-2 h  
![](images/3d89e8ed69f68878d80cdc65f8d132cd8116c2afa33a5db41fcb82316c8952fe.jpg)  
Figure 9: Station-neighborhood event predictions for a localized convective heavy-rainfall episode over coastal southern Guangxi, initialized at 22:00 UTC on 10 August 2023 (06:00 BJT on 11 August). The top row shows the latest input AGRI 10.7 µm image at 21:45 UTC (05:45 BJT), observed positive events in the three forecast windows, and an enlarged view of the newly positive component during 1–2 h. The rightmost column uses the same AGRI 10.7 µm background for all rows. The remaining rows show predictions from PhyDNet, NPM, NowcastNet, and MAGPIE-Net. EarthFormer, omitted for clarity, also missed all three stations in the highlighted component. Cyan, orange, red, and blue markers denote hits, false alarms, misses, and observed positive events, respectively.

## 4.6.2 Organized Tropical-Cyclone Rainfall During Typhoon Doksuri

Figure 10 presents a contrasting, spatially organized rainfall case during the landfall of Typhoon Doksuri on 28 July 2023. The forecast was initialized at 00:00 UTC (08:00 BJT), approximately two hours before landfall near Jinjiang, Fujian. At initialization, the AGRI image showed a mature central cloud shield and spiral cloud bands approaching southeastern China. Over the following three hours, the observed station-neighborhood events expanded along the Fujian–Zhejiang coast as the tropical-cyclone circulation and associated rainbands moved inland.

The gridded-output models produced fragmented warnings and missed substantial parts of the coastal event band. MAGPIE-Net produced a more continuous band of hits and achieved the highest case-level CSI in all three forecast windows. The diference was most pronounced during 1–2 h and 2–3 h, as the afected area expanded along the coast. Some false alarms occurred beneath the broad cloud shield but outside the strongest surface rainbands. Their spatial distribution is consistent with the asymmetric rainfall structure of tropical cyclones (Lonfat et al., 2004) and the displacement between convective cold-cloud features and surface-rainfall maxima (Chen et al., 2025). Across the three windows, MAGPIE-Net followed the observed coastal expansion of the rainband more continuously than the gridded-output models.

The two cases span complementary stages of cloud–rain evolution. The southern Guangxi case follows localized heavy-rainfall development from near-zero antecedent surface rainfall, whereas the Doksuri case follows the inland expansion of a mature and organized tropical-cyclone rainband. MAGPIE-Net identified the newly positive southern Guangxi component missed by all gridded-output baselines and more continuously captured the coastal expansion of the Doksuri rainband.

Latest AGRI 10.7-μm image

Observed | 0-1 h

Observed | 1-2 h

# Observed | 2-3 h Observed zoom | 2-3 h

![](images/06a5f25bd4df4a6924aca785a954a7f905bd1dfb3651f75090017c8aa7369dec.jpg)

Figure 10: Station-neighborhood event predictions during the landfall of Typhoon Doksuri, initialized at 00:00 UTC (08:00 BJT) on 28 July 2023. The top row shows the latest input AGRI 10.7 µm image, observed positive events in the three forecast windows, and an enlarged coastal view during 2–3 h. The rightmost column uses the same AGRI 10.7 µm background for all rows. The remaining rows show predictions from PhyDNet, NPM, NowcastNet, and MAGPIE-Net. Cyan, orange, red, and blue markers denote hits, false alarms, misses, and observed positive events, respectively.

## 5 Conclusions

We developed MAGPIE-Net, an event-oriented satellite-to-station model that uses multitemporal FY-4A AGRI observations to predict station-neighborhood heavy-rainfall events at lead times of 0–3 h. The model combines convection-initiation feature construction, multiscale satellite encoding, an auxiliary gridded precipitation diagnosis, and geographically adaptive grid-to-station decoding. This design preserves the regional organization of cloud- and rainfall-related features, while the station-neighborhood event target directly guides their conversion into local station-neighborhood event probabilities.

In an independent evaluation over central and eastern China during the 2023 warm season, MAGPIE-Net outperformed the gridded-output baselines across all three forecast windows under the primary event definition. This advantage persisted under smaller neighborhood radii and the 50 mm $\mathrm { h } ^ { - 1 }$ threshold. When consecutive events at the same station were grouped into heavy-rainfall episodes, MAGPIE-Net achieved an episode detection rate of 65.1% and a mean episode warning lead time of 64.6 min, compared with 23.6% and 18.3 min for the best-performing gridded-output baseline. During the critical early-warning stage, when rainfall during the preceding hour remained below 1 mm within the 40-km neighborhood, MAGPIE-Net still detected 51.9% of the subsequent episodes and provided a mean episode warning lead time of 38.5 min. Together, these results show that continuous AGRI observations of cloud and moisture evolution can provide warning-relevant information for station-neighborhood heavy rainfall before substantial surface rainfall develops.

The configuration experiments clarify how MAGPIE-Net converts satellite information into stationneighborhood warnings. The complete event-oriented configuration produced the largest improvement over the gridded-output pathway, while GA-SetConv provided an additional gain through station- and lead-dependent spatial aggregation. Relative to the retrained no-CI model, including CI features increased the episode detection rate and mean warning lead time. Inference-time masking further showed that the probability response to CI information was concentrated near rapidly developing clouds.

Performance is lowest for spatially localized extreme-rainfall events under the smaller-neighborhood and higher-intensity definitions. AGRI cloud-top and water-vapor observations do not directly resolve low-level moisture convergence, warm-rain microphysics, or terrain-induced near-surface processes, and cloud-top features can be displaced from surface rainfall. Future work can combine AGRI observations with radar echoes, lightning, recent station rainfall, and atmospheric-state information to complement AGRI with information about the near-surface environment and hydrometeor structure. Evaluation across other climate regions and geostationary satellite sensors can further assess model transferability and probability calibration.

Taken together, these results show that, compared with conventional gridded-output modeling, the eventoriented satellite-to-station pathway more efectively translates multitemporal AGRI observations into local short-duration heavy-rainfall warnings.

## References

Georgy Ayzel, Tobias Schefer, and Maik Heistermann. RainNet v1.0: a convolutional neural network for radarbased precipitation nowcasting. Geosci. Model Dev., 13(6):2631–2644, 2020. doi: 10.5194/gmd-13-2631-2020.

Stanley L. Barnes. A technique for maximizing details in numerical weather map analysis. J. Appl. Meteorol., 3(4):396–409, 1964. doi: 10.1175/1520-0450(1964)003<0396:ATFMDI>2.0.CO;2.

Patrick C. Burke, Alex Lamers, Gregory Carbin, Michael J. Erickson, Mark Klein, Marc Chenard, Jennifer McNatt, and Lance Wood. The Excessive Rainfall Outlook at the Weather Prediction Center: operational definition, construction, and real-time collaboration. Bull. Am. Meteorol. Soc., 104(3):E542–E562, 2023. doi: 10.1175/BAMS-D-21-0281.1.

Jiong Chen, Yongguang Zheng, Xiaoling Zhang, and Peijun Zhu. Distribution and diurnal variation of warm-season short-duration heavy rainfall in relation to the MCSs in China. Acta Meteorol. Sin., 27(6): 868–888, 2013. doi: 10.1007/s13351-013-0605-x.

Zitong Chen, Yunying Li, Jing Sun, Xiong Hu, and Chao Zhang. Summer surface rainfall deviations from convective cold-cloud shields over China. Geophys. Res. Lett., 52(23):e2025GL117889, 2025. doi: 10.1029/2025GL117889.

Christopher Davis, Barbara Brown, and Randy Bullock. Object-based verification of precipitation forecasts. part I: Methodology and application to mesoscale rain areas. Mon. Weather Rev., 134(7):1772–1784, 2006. doi: 10.1175/MWR3145.1.

Charles A. Doswell, Harold E. Brooks, and Robert A. Maddox. Flash flood forecasting: an ingredientsbased methodology. Weather Forecast., 11(4):560–581, 1996. doi: 10.1175/1520-0434(1996)011<0560: FFFAIB>2.0.CO;2.

Elizabeth E. Ebert. Fuzzy verification of high-resolution gridded forecasts: a review and proposed framework. Meteorol. Appl., 15(1):51–64, 2008. doi: 10.1002/met.25.

Michael J. Erickson, Benjamin Albright, and James A. Nelson. Verifying and redefining the Weather Prediction Center’s Excessive Rainfall Outlook forecast product. Weather Forecast., 36(1):325–340, 2021. doi: 10.1175/WAF-D-20-0020.1.

Lasse Espeholt, Shreya Agrawal, Casper Sønderby, Manoj Kumar, Jonathan Heek, Carla Bromberg, Cenk Gazen, Rob Carver, Marcin Andrychowicz, Jason Hickey, Aaron Bell, and Nal Kalchbrenner. Deep learning for twelve hour precipitation forecasts. Nat. Commun., 13:5145, 2022. doi: 10.1038/s41467-022-32483-x.

Hayley J. Fowler, Geert Lenderink, Andreas F. Prein, Seth Westra, Richard P. Allan, Nikolina Ban, Renaud Barbero, Peter Berg, Stephen Blenkinsop, Hong X. Do, Selma B. Guerreiro, Jan O. Haerter, Elizabeth J. Kendon, Elizabeth Lewis, Christoph Schär, Ashish Sharma, Gabriele Villarini, Conrad Wasko, and Xuebin Zhang. Anthropogenic intensification of short-duration rainfall extremes. Nat. Rev. Earth Environ., 2: 107–122, 2021. doi: 10.1038/s43017-020-00128-6.

Zhihan Gao, Xingjian Shi, Hao Wang, Yi Zhu, Yuyang Wang, Mu Li, and Dit-Yan Yeung. Earthformer: exploring space-time transformers for Earth system forecasting. In Advances in Neural Information Processing Systems, volume 35, pages 25390–25403. Curran Associates, Inc., 2022. doi: 10.52202/068431-1841. URL https://proceedings.neurips.cc/paper\_files/paper/2022/hash/ a2affd71d15e8fedffe18d0219f4837a-Abstract-Conference.html.

Marta Garnelo, Dan Rosenbaum, Christopher Maddison, Tiago Ramalho, David Saxton, Murray Shanahan, Yee Whye Teh, Danilo Rezende, and S. M. Ali Eslami. Conditional neural processes. In Proceedings of the 35th International Conference on Machine Learning, volume 80 of Proceedings of Machine Learning Research, pages 1704–1713. PMLR, 2018. URL https://proceedings.mlr.press/v80/garnelo18a.html.

Urs Germann and Isztar Zawadzki. Scale-dependence of the predictability of precipitation from continental radar images. part I: Description of the methodology. Mon. Weather Rev., 130(12):2859–2873, 2002. doi: 10.1175/1520-0493(2002)130<2859:SDOTPO>2.0.CO;2.

Eric Gilleland, David Ahijevych, Barbara G. Brown, Barbara Casati, and Elizabeth E. Ebert. Intercomparison of spatial forecast verification methods. Weather Forecast., 24(5):1416–1430, 2009. doi: 10.1175/2009WAF2222269.1.

Eric Gilleland, David A. Ahijevych, Barbara G. Brown, and Elizabeth E. Ebert. Verifying forecasts spatially. Bull. Am. Meteorol. Soc., 91(10):1365–1376, 2010. doi: 10.1175/2010BAMS2819.1.

Jonathan Gordon, Wessel P. Bruinsma, Andrew Y. K. Foong, James Requeima, Yann Dubois, and Richard E. Turner. Convolutional conditional neural processes. International Conference on Learning Representations, 2020. URL https://openreview.net/forum?id=Skey4eBYPS.

Daehyeon Han, Juhyun Lee, Jungho Im, Seongmun Sim, Sanggyun Lee, and Hyangsun Han. A novel framework of detecting convective initiation combining automated sampling, machine learning, and repeated model tuning from geostationary satellite data. Remote Sens., 11(12):1454, 2019. doi: 10.3390/rs11121454.

Robert A. Houze, Jr. Orographic efects on precipitating clouds. Rev. Geophys., 50(1):RG1001, 2012. doi: 10.1029/2011RG000365.

Robert A. Houze, Jr. Cloud Dynamics. Academic Press, Oxford, 2 edition, 2014. ISBN 9780123742667.

Ruben O. Imhof, Claudia C. Brauer, Karel-Jan van Heeringen, Remko Uijlenhoet, and Albrecht H. Weerts. Large-sample evaluation of radar rainfall nowcasting for flood early warning. Water Resour. Res., 58(3): e2021WR031591, 2022. doi: 10.1029/2021WR031591.

Chris Kidd and Vincenzo Levizzani. Status of satellite precipitation retrievals. Hydrol. Earth Syst. Sci., 15 (4):1109–1116, 2011. doi: 10.5194/hess-15-1109-2011.

Vincent Le Guen and Nicolas Thome. Disentangling physical dynamics from unknown factors for unsupervised video prediction. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 11471–11481, 2020. doi: 10.1109/CVPR42600.2020.01149.

Sanggyun Lee, Hyangsun Han, Jungho Im, Eunna Jang, and Myong-In Lee. Detection of deterministic and probabilistic convection initiation using Himawari-8 Advanced Himawari Imager data. Atmos. Meas. Tech., 10(5):1859–1874, 2017. doi: 10.5194/amt-10-1859-2017.

Yoonjin Lee, Christian D. Kummerow, and Imme Ebert-Uphof. Applying machine learning methods to detect convection using Geostationary Operational Environmental Satellite-16 (GOES-16) advanced baseline imager (ABI) data. Atmos. Meas. Tech., 14(4):2699–2716, 2021. doi: 10.5194/amt-14-2699-2021.

Yang Li, Yubao Liu, Yueqin Shi, Baojun Chen, Fanhui Zeng, Zhaoyang Huo, and Hang Fan. Probabilistic convective initiation nowcasting using Himawari-8 AHI with explainable deep learning models. Mon. Weather Rev., 152(1):363–385, 2024. doi: 10.1175/MWR-D-22-0216.1.

Manuel Lonfat, Frank D. Marks, Jr., and Shuyi S. Chen. Precipitation distribution in tropical cyclones using the Tropical Rainfall Measuring Mission (TRMM) Microwave Imager: a global perspective. Mon. Weather Rev., 132(7):1645–1660, 2004. doi: 10.1175/1520-0493(2004)132<1645:PDITCU>2.0.CO;2.

Yali Luo, Renhe Zhang, Qilin Wan, Bin Wang, Wai Kin Wong, Zhiqun Hu, Ben Jong-Dao Jou, Yanluan Lin, Richard H. Johnson, Chih-Pei Chang, Yuejian Zhu, Xubin Zhang, Hui Wang, Rudi Xia, Juhui Ma, Da-Lin Zhang, Mei Gao, Yijun Zhang, Xi Liu, Yangruixue Chen, Huijun Huang, Xinghua Bao, Zheng Ruan, Zhehu Cui, Zhiyong Meng, Jiaxiang Sun, Mengwen Wu, Hongyan Wang, Xindong Peng, Weimiao Qian, Kun Zhao, and Yanjiao Xiao. The Southern China Monsoon Rainfall Experiment (SCMREX). Bull. Am. Meteorol. Soc., 98(5):999–1013, 2017. doi: 10.1175/BAMS-D-15-00235.1.

John R. Mecikalski and Kristopher M. Bedka. Forecasting convective initiation by monitoring the evolution of moving cumulus in daytime GOES imagery. Mon. Weather Rev., 134(1):49–78, 2006. doi: 10.1175/ MWR3062.1.

John R. Mecikalski, Wayne M. MacKenzie, Jr., Marianne Koenig, and Sam Muller. Cloud-top properties of growing cumulus prior to convective initiation as measured by Meteosat Second Generation. part I: Infrared fields. J. Appl. Meteorol. Climatol., 49(3):521–534, 2010. doi: 10.1175/2009JAMC2344.1.

Young-Jae Park, Doyi Kim, Minseok Seo, Hae-Gon Jeon, and Yeji Choi. Data-driven precipitation nowcasting using satellite imagery. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pages 28284–28292, 2025. doi: 10.1609/aaai.v39i27.35049.

L. Peng, Y. Li, C. Ye, and X. Ou. A new convective initiation definition and its characteristics in Central and Eastern China based on Fengyun-4A satellite cloud imagery. Remote Sens., 17(24):4053, 2025. doi: 10.3390/rs17244053.

Andreas F. Prein, Roy M. Rasmussen, Kyoko Ikeda, Changhai Liu, Martyn P. Clark, and Greg J. Holland. The future intensification of hourly precipitation extremes. Nat. Clim. Change, 7(1):48–52, 2017. doi: 10.1038/nclimate3168.

Seppo Pulkkinen, Daniele Nerini, Andrés A. Pérez Hortal, Carlos Velasco-Forero, Alan Seed, Urs Germann, and Loris Foresti. Pysteps: an open-source Python library for probabilistic precipitation nowcasting (v1.0). Geosci. Model Dev., 12(10):4185–4219, 2019. doi: 10.5194/gmd-12-4185-2019.

Suman Ravuri, Karel Lenc, Matthew Willson, Dmitry Kangin, Remi Lam, Piotr Mirowski, Megan Fitzsimons, Maria Athanassiadou, Sheleem Kashem, Sam Madge, Rachel Prudden, Amol Mandhane, Aidan Clark, Andrew Brock, Karen Simonyan, Raia Hadsell, Niall Robinson, Ellen Clancy, Alberto Arribas, and Shakir Mohamed. Skilful precipitation nowcasting using deep generative models of radar. Nature, 597(7878): 672–677, 2021. doi: 10.1038/s41586-021-03854-z.

Nigel M. Roberts and Humphrey W. Lean. Scale-selective verification of rainfall accumulations from high-resolution forecasts of convective events. Mon. Weather Rev., 136(1):78–97, 2008. doi: 10.1175/ 2007MWR2123.1.

Rita D. Roberts and Steven Rutledge. Nowcasting storm initiation and growth using GOES-8 and WSR-88D data. Weather Forecast., 18(4):562–584, 2003. doi: 10.1175/1520-0434(2003)018<0562:NSIAGU>2.0.CO;2.

Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-Net: convolutional networks for biomedical image segmentation. In Medical Image Computing and Computer-Assisted Intervention – MICCAI 2015, volume 9351 of Lecture Notes in Computer Science, pages 234–241. Springer, Cham, 2015. doi: 10.1007/978-3-319-24574-4\_28.

Elena Saltikof, Katja Friedrich, Joshua Soderholm, Katharina Lengfeld, Brian R. Nelson, Andreas Becker, Rainer Hollmann, Bernard Urban, Maik Heistermann, and Caterina Tassone. An overview of using weather radar for climatological studies: successes, challenges, and potential. Bull. Am. Meteorol. Soc., 100(9): 1739–1752, 2019. doi: 10.1175/BAMS-D-18-0166.1.

Xingjian Shi, Zhourong Chen, Hao Wang, Dit-Yan Yeung, Wai-kin Wong, and Wang-chun Woo. Convolutional LSTM network: a machine learning approach for precipitation nowcasting. In Advances in Neural Information Processing Systems, volume 28, pages 802–810. Cur-

ran Associates, Inc., 2015. URL https://proceedings.neurips.cc/paper\_files/paper/2015/hash/ 07563a3fe3bbe7e3ba84431ad9d055af-Abstract.html.

Xingjian Shi, Zhihan Gao, Leonard Lausen, Hao Wang, Dit-Yan Yeung, Wai-kin Wong, and Wang-chun Woo. Deep learning for precipitation nowcasting: a benchmark and a new model. In Advances in Neural Information Processing Systems, volume 30, pages 5617–5627, 2017. URL https://proceedings.neurips. cc/paper\_files/paper/2017/hash/a6db4ed04f1621a119799fd3d7545d3d-Abstract.html.

Justin M. Sieglaf, Lee M. Cronce, Wayne F. Feltz, Kristopher M. Bedka, Michael J. Pavolonis, and Andrew K. Heidinger. Nowcasting convective storm initiation using satellite-based box-averaged cloud-top cooling and cloud-type trends. J. Appl. Meteorol. Climatol., 50(1):110–126, 2011. doi: 10.1175/2010JAMC2496.1.

Casper Kaae Sønderby, Lasse Espeholt, Jonathan Heek, Mostafa Dehghani, Avital Oliver, Tim Salimans, Shreya Agrawal, Jason Hickey, and Nal Kalchbrenner. MetNet: a neural weather model for precipitation forecasting. arXiv preprint arXiv:2003.12140, 2020. doi: 10.48550/arXiv.2003.12140.

Standardization Administration of China. Weather forecast verification—severe convective weather. National Standard GB/T 44213-2024, Standardization Administration of China, Beijing, China, 2024. URL https: //openstd.samr.gov.cn/bzgk/std/newGbInfo?hcno=FE3805309734A2E082424559B3C7F6A4. Accessed 12 August 2026.

Graeme L. Stephens and Christian D. Kummerow. The remote sensing of clouds and precipitation from space: a review. J. Atmos. Sci., 64(11):3742–3765, 2007. doi: 10.1175/2006JAS2375.1.

Jing Sun, Yunying Li, Hao Hu, Qian Li, Chengzhi Ye, Yining Shi, and Zitong Chen. Impact of cloud vertical structure perturbations on the retrieval of cloud optical thickness and efective radius from FY4A/AGRI. Atmos. Chem. Phys., 25(22):16347–16361, 2025. doi: 10.5194/acp-25-16347-2025.

Fuyou Tian, Yongguang Zheng, Tao Zhang, Dongyan Mao, Wenyuan Tang, Qingliang Zhou, Jianhua Sun, and Sixiong Zhao. Sensitivity analysis of short-duration heavy rainfall related diagnostic parameters with point-area verification. J. Appl. Meteorol. Sci., 26(4):385–396, 2015. doi: 10.11898/1001-7313.20150401.

Kevin Trebing, Tomasz Stańczyk, and Siamak Mehrkanoon. SmaAt-UNet: Precipitation nowcasting using a small attention-UNet architecture. Pattern Recognit. Lett., 145:178–186, 2021. doi: 10.1016/j.patrec.2021. 01.036.

John R. Walker, Wayne M. MacKenzie, Jr., John R. Mecikalski, and Christopher P. Jewett. An enhanced geostationary satellite-based convective initiation algorithm for 0–2-h nowcasting with object tracking. J. Appl. Meteorol. Climatol., 51(11):1931–1949, 2012. doi: 10.1175/JAMC-D-11-0246.1.

Seth Westra, Hayley J. Fowler, Jason P. Evans, Lisa V. Alexander, Peter Berg, Fiona Johnson, Elizabeth J. Kendon, Geert Lenderink, and Nigel M. Roberts. Future changes to the intensity and frequency of short-duration extreme rainfall. Rev. Geophys., 52(3):522–555, 2014. doi: 10.1002/2014RG000464.

James W. Wilson, N. Andrew Crook, Cynthia K. Mueller, Juanzhen Sun, and Michael Dixon. Nowcasting thunderstorms: a status report. Bull. Am. Meteorol. Soc., 79(10):2079–2099, 1998. doi: 10.1175/1520-0477(1998)079<2079:NTASR>2.0.CO;2.

Chunlei Yang, Huiling Yuan, Feng Zhang, Meng Xie, Yan Wang, and Geng-Ming Jiang. Convective initiation nowcasting in South China using physics-augmented random forest models and geostationary satellites. Earth Space Sci., 11(7):e2024EA003571, 2024. doi: 10.1029/2024EA003571.

Jun Yang, Zhiqing Zhang, Caiying Wei, Feng Lu, and Qiang Guo. Introducing the new generation of Chinese geostationary weather satellites, Fengyun-4. Bull. Am. Meteorol. Soc., 98(8):1637–1658, 2017. doi: 10.1175/BAMS-D-16-0065.1.

Manzil Zaheer, Satwik Kottur, Siamak Ravanbakhsh, Barnabás Póczos, Ruslan Salakhutdinov, and Alexander J. Smola. Deep sets. In Advances in Neural Information Processing Systems, volume 30, pages 3391–3401, 2017. URL https://proceedings.neurips.cc/paper/2017/hash/ f22e4747da1aa27e363d86d40ff442fe-Abstract.html.

Huan Zhang and Panmao Zhai. Temporal and spatial characteristics of extreme hourly precipitation over eastern China in the warm season. Adv. Atmos. Sci., 28(5):1177–1183, 2011. doi: 10.1007/s00376-011-0020-0.

Yuchen Zhang, Mingsheng Long, Kaiyuan Chen, Lanxiang Xing, Ronghua Jin, Michael I. Jordan, and Jianmin Wang. Skilful nowcasting of extreme precipitation with NowcastNet. Nature, 619(7970):526–532, 2023. doi: 10.1038/s41586-023-06184-4.

Yongguang Zheng, Kanghui Zhou, Jie Sheng, Yinjing Lin, Fuyou Tian, Wenyuan Tang, Yu Lan, and Wenjian Zhu. Advances in techniques of monitoring, forecasting and warning of severe convective weather. J. Appl. Meteorol. Sci., 26(6):641–657, 2015. doi: 10.11898/1001-7313.20150601.

Zongwei Zhou, Md Mahfuzur Rahman Siddiquee, Nima Tajbakhsh, and Jianming Liang. UNet++: a nested U-Net architecture for medical image segmentation. In Deep Learning in Medical Image Analysis and Multimodal Learning for Clinical Decision Support, Lecture Notes in Computer Science, pages 3–11. Springer, Cham, 2018. doi: 10.1007/978-3-030-00889-5\_1.

Tobias Zinner, Hermann Mannstein, and Arnold Taferner. Cb-TRAM: Tracking and monitoring severe convection from onset over rapid development to mature phase using multi-channel Meteosat-8 SEVIRI data. Meteorol. Atmos. Phys., 101(3–4):191–210, 2008. doi: 10.1007/s00703-008-0290-y.