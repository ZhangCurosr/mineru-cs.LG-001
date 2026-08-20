# Scalable Geospatial Machine Learning for Power-Line Asset Risk: Integrating Remote Sensing for Lightning and Vegetation Risk Modelling

Artur Sokolovskya, Bhavik Meraia, Moe Jafaria, Muen Chena

aSA Power Networks, 1 Anzac Highway, Keswick SA, Adelaide, 5035, Australia

## Abstract

Electric power networks are increasingly exposed to weather-sensitive failure mechanisms that require asset-level, spatially explicit risk modelling for effective intervention planning. This study contributes a modular, robust, and explainable probability-of-failure (PoF) modelling framework for utility asset management. The central contribution is an asset-level architecture that can be scaled to new environmental data sources and additional PoF types without reworking the underlying pipeline. This is particularly relevant for industry settings, where risk models must remain operationally maintainable while adapting to changing data availability, asset-management priorities, and climate-driven hazard conditions

We demonstrate the framework for vegetation-related and lightning-related failure modes using a harmonised geospatial machine-learning pipeline. The implementation integrates multi-source predictors, including topography (SRTM) vegetation condition (MODIS Normalised Difference Vegetation Index - NDVI) lightning climatology (LIS VHRMC), OpenStreetMap-derived proximity features, and utility operational records. The resulting architecture is computationally efficient, operationally extensible, and suitable for utility-scale

deployment. It provides actionable asset-level risk stratification for inspection prioritisation, vegetation management, asset hardening, and resilience planning, supporting earlier intervention and more climate-resilient network operations.

Keywords: Energy Grid, Asset Management, Power Systems, Climate Resilience, Infrastructure Resilience

## 1. Introduction

## 1.1. Background

Electric power networks are increasingly exposed to weather-sensitive failure mechanisms that are spatially heterogeneous and operationally costly. In this study, we focus on two hazard-specific probabilities of failure (PoF): lightning PoF, defined as the probability that an asset fails due to lightningrelated exposure, and vegetation PoF, defined as the probability that an asset fails due to vegetation-related interactions (e.g., mechanical impact, or flashover events). Prior evidence shows that severe-weather impacts are non-uniform across networks, motivating asset-level modelling rather than aggregate system averages (Mukherjee et al., 2018).

From an environmental-process perspective, both hazards are dynamic. Lightning regimes may shift under warming climatic conditions (Romps et al., 2014), while vegetation dryness is also influenced by climate variability and change (Abatzoglou and Williams, 2016). Geospatial Earth-observation products therefore provide an important foundation for scalable risk modelling: terrain information from SRTM (Farr et al., 2007), vegetation-condition signals from MODIS indices (Didan, 2021), and long-term lightning climatology from LIS (Cecil et al., 2014). NDVI and related indices are particularly relevant as interpretable proxies of vegetation status in disturbance-oriented applications (Pettorelli et al., 2005).

## 1.2. The Research Gap

Despite progress in outage and hazard modelling, three limitations persist in the literature and in many operational implementations. First, studies frequently treat lightning and vegetation risks separately, limiting integrated asset-prioritisation decisions when both mechanisms can affect the same infrastructure. Second, many models remain too coarse spatially to support asset-level interventions (e.g., targeted inspection, trimming, hardening, or surge-protection planning). Third, hazard proxies are often not tightly coupled with utility failure records, reducing practical predictive value and limiting explainability for field deployment (Mukherjee et al., 2018)

Accordingly, there remains a need for a scalable and extensible PoF modelling design that uses harmonised geospatial predictors and observed utility failure outcomes, while supporting hazard-specific model instances for vegetation, lightning, and additional failure modes (e.g., animal-related faults).

## 1.3. Research Question

This study asks whether a scalable and extensible asset-level PoF framework, built on harmonised geospatial predictors and utility failure records, can produce operationally useful risk ranking and interpretable hazard-specific outputs for utility decision support. Under this question, the primary objective is methodological and operational validation within the available network context, rather than strict forward-in-time forecasting or geographic-domain transfer testing.

## 1.4. Contributions

This study makes four main contributions to asset-risk modelling for utility networks.

• Modular multi-PoF formulation: We develop an asset-level framework with a shared geospatial-machine-learning pipeline and hazardspecific model instances, demonstrated for vegetation PoF and lightning PoF, enabling consistent cross-hazard risk assessment without forcing a single combined model.

• Efficient and scalable modelling pipeline: We implement a computationally efficient approach suitable for utility-scale deployment, combining multi-scale spatial feature engineering with a modelling architecture that can be operationalised across asset fleets of millions of assets.

• Extensible data and target design: The framework is designed to be modular, allowing utilities to introduce new target variables (e.g., animal-related PoF), incorporate additional data sources, and expand the feature space as new operational and environmental datasets become available.

• Operationally meaningful asset-level granularity: Results indicate that asset-level PoF modelling captures historical failure patterns with useful fidelity, supporting more targeted and effective interventions for maintenance prioritisation and resilience planning.

## 1.5. Outline of the Study

The remainder of this paper is organised as follows. Section 2 describes the data sources, preprocessing pipeline, spatial feature engineering, and machine-learning architecture for modular multi-PoF estimation using hazardspecific models. Section 3 reports model validation, spatial risk patterns, explainability analyses, and operational risk stratification outputs. Section 4 interprets the findings, discusses limitations and practical implications, and outlines future directions toward near-real-time risk modelling. A glossary of acronyms is provided in Appendix A.

## 2. Materials and Methods

## 2.1. Data sources

The modelling framework integrates multi-source Earth-observation, volunteered geographic information, and utility operational data to characterise asset-level environmental exposure. Topographic context is represented using the NASA Shuttle Radar Topography Mission (SRTM) Global 3 arcsecond (V003) digital elevation model (Farr et al., 2007). Built-environment and land-water context are derived from OpenStreetMap (OSM) XML data (Contributors), including coastline, inland-water, and building layers reconstructed into analysis-ready vector geometries. The conceptual data flow is illustrated if Fig 1.

![](images/c4fdf6ae4b35087c775a7c8981f05523a28d9edb47c507a75b9eb0645442452e.jpg)  
Figure 1: Architecture of the data processing pipeline, illustrating the disparate handling of gridded environmental data and the reconstruction of vector geometries to assemble the final feature space.

Vegetation conditions are represented using the MODIS/Terra Vegetation Indices product (MOD13Q1, Collection 6.1; 16-day composites, 250 m, sinusoidal grid) (Didan, 2021). From this product, we derive NDVI-based predictors as proxies for vegetation condition and spatial biomass dynamics, consistent with established ecological remote-sensing practice (Pettorelli et al., 2005). To align with industrial delivery constraints, NDVI features were computed from exactly one year of the most recent available observations. Lightning exposure is characterised using the LIS 0.1° Very High-Resolution Gridded Lightning Monthly Climatology (VHRMC; V1) (Cecil et al., 2014), which provides long-term spatial patterns of lightning occurrence. Supervised labels are obtained from the utility's historical failurenotification archive (binary failure/no-failure by asset and failure mode).

Taken together, these sources define the broader modelling feature space, spanning topographic attributes, proximity measures, vegetation and lightning indicators, prior-failure information, and asset/network-context variables. The full list of model features is summarised in Table 1, and the derivation of these variables is described below.

## 2.2. Data preprocessing

All datasets were harmonised into a common asset-centred geospatial representation prior to feature extraction. Asset coordinates were used as analysis anchors, and multi-scale regions of interest (200 m to 50 km) were constructed to capture both near-field effects (e.g., local relief, building proximity, short-range vegetation context) and broader mesoscale exposure (e.g. regional lightning climatology).

Raster data streams (SRTM, MODIS NDVI, and LIS VHRMC) were spatially queried within each analysis window and summarised through scaleappropriate descriptive statistics. Vector layers derived from OSM were spatially intersected with the same windows to compute proximity- and densityoriented predictors (e.g., distance to nearest water body, coastline, or building element). This harmonised processing ensures that predictors from heterogeneous sources are comparable at asset level and suitable for downstream machine-learning ingestion.

Failure-notification records were quality controlled, standardised, and linked to the corresponding asset identifiers to generate binary targets for each PoF type. The resulting analytical table combines engineered geospatial predictors, operational-history indicators, and asset/network-context variables in a single supervised-learning matrix. Table 1 summarises the feature

space at a conceptual level.

Table 1: Full list of model features, descriptions, and spatial search constraints.
<table><tr><td>Category</td><td>Feature</td><td>Description</td><td>Search Radius</td></tr><tr><td>Topographic</td><td>Elevation Std. Dev.</td><td>Measure of local terrain roughness derived from the SRTM DEM.</td><td>400m</td></tr><tr><td>Topographic</td><td>Relative Elevation</td><td>Difference between asset elevation and the local neighbourhood mean.</td><td>400m</td></tr><tr><td>Topographic</td><td>Absolute Elevation</td><td>Absolute elevation at the asset location from the SRTM DEM.</td><td>400m</td></tr><tr><td>Topographic</td><td>Elevation Gradient</td><td>Local terrain slope calculated from nearby DEM cells.</td><td>400m</td></tr><tr><td>Proximity</td><td>Distance to Coastline</td><td>Minimum geodesic distance to the nearest coastline.</td><td>15km</td></tr><tr><td>Proximity</td><td></td><td>Distance to Waterbody Minimum geodesic distance to the nearest inland waterbody.</td><td>7km</td></tr><tr><td>Proximity</td><td>Distance to Building</td><td>Minimum geodesic distance to the nearest building.</td><td>200m</td></tr><tr><td>Environmental</td><td>Mean NDVI</td><td>Local mean NDVI as a proxy for vegetation condition.</td><td>400m</td></tr><tr><td>Environmental</td><td>Max NDVI Change</td><td>Maximum positive NDVI change representing recent biomass growth.</td><td>400m</td></tr><tr><td></td><td>Environmental Lightning Intensity</td><td>Regional lightning-climatology intensity summary.</td><td>50 km</td></tr><tr><td>Asset/Network Aggregated Voltage</td><td></td><td>Aggregated asset voltage class.</td><td>Exact match</td></tr><tr><td>Asset/Network SCONRR</td><td></td><td>Network-context classification (e.g., Rural, Metro, CBD).</td><td>Exact match</td></tr><tr><td>Asset/Network Tree Overhang Flag</td><td></td><td>Indicator of observed tree overhang near the asset.</td><td>Exact match</td></tr></table>

## 2.3. Data Processing and Spatial Optimisation

The integration of high-resolution remote sensing grids with expansive vector databases (such as OSM) across a regional-scale utility network presents significant computational challenges. The primary objective of this data integration was to construct a comprehensive feature space for each utility asset, capturing its specific environmental exposure across key domains: Key feature domains and representative engineered variables are summarised in Table 1.

To efficiently compute these features at scale, data preprocessing was executed end-to-end in the Snowflake cloud data platform. A highly optimised geospatial pipeline was developed in Snowflake to project, join, and aggregate these disparate spatial layers relative to the discrete point coordinates of each utility asset.

All geospatial transformations, feature engineering, intermediate table materialisation, and quality-control checks were performed within Snowflake. This unified cloud-native implementation ensured scalable distributed execution, consistent reproducibility across pipeline stages, and operational readiness for utility-scale deployment.

## 2.3.1. Optimised Geospatial Pruning

A core methodological challenge was computing precise proximity metrics, such as the exact distance to the nearest coastline or vegetation stand, without executing computationally prohibitive Cartesian products across the entire regional dataset. To resolve this, we implemented a two-step geospatial pruning algorithm as shown in Fig 2.

A Step 1 : Bounding-box Filter  
![](images/3b93c14f9d04194622ac66e53d2d7f121c49fbc5f65bfe810a80b2b3de213a5a.jpg)  
B Step 2: Calculate actual distances  
Figure 2: Schematic representation of the two-step geospatial pruning algorithm. A dynamic latitudinal/longitudinal bounding box (Panel A) efficiently filters candidate points before exact geodesic distance calculations are applied (Panel B).

First, a dynamic, numeric bounding box (BBox) was calculated for each asset based on the target search radius (R). To account for the Earth's curvature, the longitudinal delta (∆λ) was dynamically scaled by the latitude (φ) of the asset point, effectively pre-filtering candidate geometries using simple numeric inequalities:

$$
\Delta \lambda = \frac { R } { c \cdot \operatorname* { m a x } ( | \cos ( \phi ) | , \epsilon ) }\tag{1}
$$

where $c \approx 1 1 1$ , 320 m represents the nominal distance per degree of latitude, and ε is a safeguard constant to prevent division by zero near the poles

Following this rapid numeric join, Snowflake geospatial SQL functions were applied only to the pruned candidate subsets. Specifically, ST\_DwITHIN was used as a spatial filter to retain only candidates within the target search distance of each asset, and ST\_DISTANCE was then used to calculate the exact final distance to the remaining candidates. This approach reduced computational overhead by orders of magnitude while preserving geometric accuracy.

## 2.3.2. Vector Geometry Reconstruction

Anthropogenic and hydrological boundaries from OpenStreetMap XML extracts required morphological reconstruction before distance calculations. Raw geographic nodes were parsed and linked to their corresponding spatial entities (e.g., natural coastline and building layers). Using sequential node indexing, discrete coordinate pairs were flattened and concatenated into continuous Well-Known Text (WKT) LINESTRING geometries. This vectorisation step enabled precise distance-to-edge measurements for localised assets rather than less accurate centroid-based approximations.

## 2.4. Geospatial Feature Engineering

Based on the spatial interactions between the infrastructure and the surrounding environment, a localised feature space was engineered for each functional location. The spatial search radii were tuned based on the morphological characteristics of the targeted variable. The high-level view is illustrated in Fig 3.

ENVIRONMENTAL PROXIMITY ASSESSMENT  
![](images/5ef0bc8f1f1320083f9351b06ba825b12c243f02ea348cbde0cddb5371259666.jpg)  
Figure 3: Conceptual visualisation of multi-scale feature engineering. Concentric search radii define the spatial context for urban sheltering (200 m), localised topographic anomalies (400 m), and broader environmental exposure (up to 50 km).

Topographic Features. Terrain variability is a critical factor in localised weather exposure. We utilised a 400 m spatial radius to extract absolute elevation profiles from the underlying DEM. From the localised grid, we derived standard deviation of elevation (to represent terrain roughness), relative elevation deviation (the asset's elevation minus the local 200 m mean), and the nearest gradient slope.

Proximity and Anthropogenic Features. Using the reconstructed OSM line geometries, exact spatial distances were calculated to contextualise the asset's immediate surroundings. We established varying search thresholds: 15 km for coastlines (capturing maritime atmospheric influences), 7km for inland waterbodies, and 200 m for building footprints (representing varying degrees of urban sheltering).

Environmental and Meteorological Features. Vegetation density heavily influences the risk of secondary lightning impacts (e.g., localised fires or tree fall). The mean and maximum change of the Normalised Difference Vegetation Index (NDVI) were extracted within a 400 m radius to quantify adjacent biomass vitality. These NDVI statistics were derived from a one-year, mostrecent observation window. Finally, regional climatic exposure was captured by querying the maximum intensity of historical lightning strikes within a broad 50 km radius, utilising the VHRMC climatology grid.

## 2.4.1. Target Variable Formulation

To construct the target variables for the subsequent binary classification model, historical maintenance and outage logs were integrated into the spatial feature matrix. Specifically, fault notifications from the utility's enterprise asset management system over a five-year period were extracted and analysed. Incidents definitively attributed to either lightning strikes or vegetation interference were isolated and matched to their respective functional locations.

These historical failure records were transformed into binary labels (i.e., the presence or absence of a vegetation or lightning-related fault at a given asset). It is critical to clarify the objective of this classification framework: rather than strictly predicting deterministic future failures at individual poles, these historical labels are utilised to train the model to assess the underlying probability of failure exposure given an asset's specific environmental, topographic, and spatial context. This probabilistic approach enables the identification of high-risk operational environments across the network, facilitating broad, targeted mitigation strategies over specific, localised predictions.

## 2.5. Final Data Preparation and Imputation

Prior to model ingestion, the assembled feature matrix required systematic treatment of missing values generated during the spatial joining processes (e.g., when no geometric feature was found within the defined search radius).

Continuous environmental variables, including terrain elevation gradients, local NDVI indices, and lightning climatology intensities, were subjected to mean imputation to represent average regional baselines in the absence of local anomalies. Conversely, unjoined spatial proximity measures required domain-specific constant imputation. Assets with no buildings or coastline within their respective search buffers were assigned values slightly beyond the maximum search distance (e.g., 17,000 m for coastlines/buildings and 9,000 m for waterbodies), effectively encoding them as "distant" without introducing algorithmic artefacts. Asset metadata (such as asset class and urban/rural categorisation) were explicitly cast as categorical data types to leverage native categorical handling algorithms.

## 2.6. Machine Learning Architecture

To model the non-linear interactions between the engineered geospatial features and asset failure probabilities, a Light Gradient Boosting Machine (LightGBM) architecture was selected (Ke et al., 2017). LightGBM is highly effective for large-scale spatial datasets due to its gradient-based one-side sampling (GOSS) and exclusive feature bundling (EFB), alongside its native, optimised handling of discrete categorical variables without requiring highly sparse one-hot encodings.

While predicting multiple failure types could theoretically be consolidated into a single multi-target model, two distinct binary classification models were trained independently: one evaluating vegetation-related failure probability and the other evaluating lightning-related failure probability. This decoupled architecture was intentionally chosen to prioritise an industry-robust approach; isolating the target variables ensures that subsequent model analysis can be performed easily and diagnostic explainability, specifically using Shapley values, can be achieved with minimal operational overhead for domain experts. To avoid generating inferences on training data and ensure unbiased probability estimates, the modelling process utilised a 5-Fold Stratified Cross-Validation framework. This approach produced clean out-of-fold predictions, while the stratified sampling ensured that the minority class (fault occurrences) was proportionally represented across all training and validation splits, properly accounting for the highly imbalanced nature of utility failure datasets.

The models were regularised to prevent overfitting, utilising an intentionally shallow maximum tree depth of 4, a high minimum child sample threshold of 30, a learning rate of 0.03, and an ensemble size of 700 boosting iterations.

## 2.6.1. Model Evaluation and Probabilistic Output

Standard accuracy is often misleading in highly imbalanced spatial risk datasets. Therefore, model performance was evaluated using Out-Of-Fold (OOF) Area Under the Receiver Operating Characteristic Curve (ROC-AUC) and Out-Of-Fold Area Under the Precision-Recall Curve (PR-AUC). While ROC-AUC remains threshold-independent and informative for global class separability, PR-AUC is particularly suitable for rare-event settings because it directly emphasises positive-class precision and recall, making it more sensitive to minority-class performance under class imbalance (Davis and Goadrich, 2006; Saito and Rehmsmeier, 2015). The OOF methodology provides robust, unbiased performance estimates across the entire network geometry by aggregating validation predictions from all 5 independent folds.

To improve the transparency of the predictive spatial model, SHapley Additive exPlanations (SHAP) (Lundberg and Lee, 2017) were calculated. SHAP values disaggregate the final probability outputs, allowing for an intuitive understanding of the relative importance and directional impact of each individual topographic, meteorological, and proximity feature on the asset's risk profile.

Finally, the raw probability outputs generated by the cross-validated models $( P _ { O O F } )$ represent the likelihood of a failure event occurring over the historical observation horizon (H). To make these insights operationally actionable for asset management, the probabilities were mathematically annualised $\left( P _ { a n n } \right)$ under the standard constant-hazard assumption used in exponential reliability models, where survival over time t is represented as $R ( t ) = e ^ { - \lambda t }$ (NIST/SEMATECH, 2003; Meeker and Escobar, 1998):

$$
P _ { a n n } = 1 - ( 1 - P _ { O O F } ) ^ { \frac { 1 } { H } }\tag{2}
$$

The resulting annualised probabilities provide a standardised risk metric, allowing utility planners to prioritise spatial clusters of assets facing elevated environmental exposure.

## 3. Results

This section presents the reported data diagnostics, model-validation results, and feature-effect analyses for the vegetation and lightning probabilityof-failure models. It first provides an overview of the variable names used in the model-ready dataset, then summarises distributional diagnostics, crossvalidated predictive performance, and model explainability outputs

## 3.1. Data

For reference, the analytical variable names used in the model-ready dataset are listed in Table 2. These names correspond directly to the higherlevel feature groups summarised in Table 1.

Table 2: Model feature names used in the analytical dataset.
<table><tr><td>Variable Name</td><td>Manuscript Feature Category</td><td></td></tr><tr><td>FEATURE_1_ELEVATION_STDEV_M</td><td>Elevation Std. Dev.</td><td>Topographic</td></tr><tr><td>FEATURE_2_ELEVATION_DEVIATION_M</td><td>Relative Elevation</td><td>Topographic</td></tr><tr><td>FEATURE_3_NEAREST_ELEVATION_M</td><td>Absolute Elevation</td><td>Topographic</td></tr><tr><td>FEATURE_4_NEAREST_GRADIENT</td><td>Elevation Gradient</td><td>Topographic</td></tr><tr><td>FEATURE_5_DISTANCE_TO_SEA_M</td><td>Distance to Coastline</td><td>Proximity</td></tr><tr><td>FEATURE_6_DISTANCE_TO_WATER_M</td><td>Distance to Waterbody</td><td>Proximity</td></tr><tr><td>FEATURE_7_DISTANCE_TO_BUILDING_M</td><td>Distance to Building</td><td>Proximity</td></tr><tr><td>FEATURE_8_NEAREST_MEAN_NDVI</td><td>Mean NDVI</td><td>Environmental</td></tr><tr><td>FEATURE_8_NEAREST_MAX_CHANGE_NDVI</td><td>Max NDVI Change</td><td>Environmental</td></tr><tr><td>FEATURE_9_NEAREST_LIGHTNING_INTENSITY</td><td>Lightning Intensity</td><td>Environmental</td></tr><tr><td>AGGREGATED_VOLTAGE_KV</td><td>Operational Voltage</td><td>Asset/Network</td></tr><tr><td>SCONRR</td><td>SCONRR region</td><td>Asset/Network</td></tr><tr><td>IS_TREE_OVERHANG</td><td>Tree Overhang Flag</td><td>Asset/Network</td></tr></table>

Due to confidentiality constraints, we limit the reported diagnostics to non-sensitive distributions only. In this section, we provide a concise narrative summary of key encoded variables and qualitative distribution signals used for data-quality checks and model-readiness assessment. Additionally, we share the feature distribution panels in Appendix B.

SCONRR region is encoded as {0: Rural, 1: Metro, 2: CBD}. The full vegetation and lightning distribution panels are provided in Figures B.8 and B.9, respectively.

The data diagnostics indicate that SCONRR captures distinct network contexts (Rural, Metro, and CBD). The tree-overhang indicator, SCONRR, and aggregated voltage class display clear class-separation patterns in the distributions. These contrasts are consistent with the feature-effect interpretation in Section 3.3. In addition, the full distribution panels indicate class imbalance for both hazard targets.

## 3.2. Model performance and validation

Cross-validated discrimination performance is shown separately for the vegetation and lightning models in Figures 4 and 5. The ROC and Precision-Recall curves indicate stable fold-level behaviour and support the suitability of the framework for imbalanced utility-failure prediction tasks. Notably, fold-level stability is lower for the lightning model, which is most visible in the PR-AUC curves. We highlight that the observed model performance is conditioned by the geolocation-unaware training/test data split; this is further discussed in the Limitations subsection of Section 4.

![](images/4a2d16f7e3a78ef0e9ad95ae946e59ae7885f31c16ad9e8e8d238e5d56408654.jpg)

![](images/d493989a5f01785fe76a95187ee7453708bd30f761d4f61d105ca93df41d3b95.jpg)  
Figure 4: Cross-validated ROC and Precision-Recall curves for the vegetation-related failure model.

![](images/fae992218261e1183f8eb9042161347ae19a92d072ef68b59c576436020c26f1.jpg)

![](images/9290b2b5254bedb02bc02cd57325b542acf8454847f1660472d35e92faa4971f.jpg)  
Figure 5: Cross-validated ROC and Precision-Recall curves for the lightning-related failure model.

## 3.3. Feature effects and model explainability

SHAP global summaries for both hazard-specific models are shown in Figures 6 and 7. Across both models, topographic context, vegetation-condition indicators, lightning-climatology variables, and spatial proximity measures contribute strongly to risk stratification, while the relative ordering of feature importance differs by failure mechanism.

![](images/3cc26a43870e3b615ff56b567721d587bb816378d640a1aa61e8fe2cc20b3fe5.jpg)  
Figure 6: Global SHAP summary for the vegetation-related failure model.

For the vegetation failure model, the three most influential features are asset voltage, distance to water, and the tree-overhang indicator. Additional separation is observed for SCONRR category and local elevation standard deviation.

The vegetation model indicates that lower aggregated voltage classes are associated with higher predicted vegetation-failure risk. Shorter distance to water and the presence of tree overhang are also associated with elevated risk, consistent with their high SHAP contribution. In addition, higher mean NDVI values correspond to increased predicted risk, and larger SCONRR values (Metro and CBD categories) are similarly associated with higher model outputs.

![](images/00c69dc685d7d3329a97608b8d5c835de9d32b1962f118a520e3c5d6660d4a74.jpg)  
Figure 7: Global SHAP summary for the lightning-related failure model

For the lightning failure model, the three most influential features are distance to water, distance to the nearest building, and local asset elevation relative to surrounding terrain. Additional separation is observed for SCONRR category, the tree-overhang indicator, and asset voltage, with the latter two showing smaller effects on model output.

Inspection of the lightning-model SHAP patterns indicates the following:

• Shorter distance to buildings is associated with lower predicted lightningfailure risk.

• Rural SCONRR region class is associated with higher risk relative to Metro/CBD classes.

• Higher local absolute elevation at the asset location is associated with increased predicted risk.

• Higher local elevation variability (terrain ruggedness) is associated with reduced risk, indicating elevated risk for assets in flatter surrounding terrain.

• The presence of tree overhang and higher-voltage equipment is associated with modest risk reduction in this model

## 4. Discussion

The developed framework demonstrates that multi-source geospatial feature engineering and gradient-boosted probabilistic modelling can support more targeted asset risk management under lightning and vegetation exposure. By integrating terrain, vegetation, hydrological, and built-environment predictors at multiple spatial scales, the approach provides interpretable risk estimates that can complement conventional rule-based maintenance planning.

## 4.1. Analysis of Findings

The SHAP-based feature contributions indicate that the models capture physically and operationally plausible exposure mechanisms rather than relying on purely statistical artefacts. For vegetation-related failures, the dominant predictors (asset voltage class, distance to water, and tree-overhang flag) align with known field conditions. Shorter distance to water is a proxy for more persistent vegetation growth potential and higher branch-fall susceptibility in some species (e.g., eucalyptus/gum trees under high-moisture growth conditions), while direct overhang is an immediate mechanism for contactrelated events. The positive contribution of higher NDVI similarly aligns with increased local biomass availability. The SCONRR effect suggests that network context (Rural/Metro/CBD) modulates baseline exposure through differences in asset surroundings and operating environments. In particular Metro and CBD areas are subject to stricter regulations that can limit preemptive vegetation management for low-voltage lines, which may contribute to higher residual vegetation-related risk

For lightning-related failures, the SHAP pattern indicates two interacting drivers: broader regional exposure and immediate asset siting characteristics. Higher local absolute elevation contributes positively to risk, consistent with greater strike susceptibility of more exposed assets. Greater terrain ruggedness around an asset is associated with lower modelled risk. Distance-to-building effects may reflect shielding or correlated infrastructure context in denser built environments. The SCONRR region contribution appears strongly indicative of whether assets benefit from city-infrastructure lightning-protection conditions, making it a key modifier of lightning risk.

Taken together, these patterns support the practical interpretability of the framework: the dominant predictors are not only statistically influential but also actionable for utility decision-making, because they map to interventions such as vegetation management, local inspection prioritisation, and location-aware hardening strategies.

## 4.2. Limitations

This study has several limitations that should be considered when interpreting the reported performance and operational implications.

First, the modelling strategy is deliberately scoped rather than benchmarkdriven. We use a modular architecture with separate binary models for each PoF type, and we do not perform formal benchmarking against alternative model families (e.g., random forests, XGBoost, neural networks). This choice prioritises a scalable, interpretable, and operationally maintainable framework under utility constraints; however, it means that the present study does not claim that the selected model class is universally optimal.

Second, the validation design does not enforce explicit geospatial or temporal train-test separation. Instead, we use stratified cross-validation to obtain statistically efficient estimates under severe class imbalance. As a result, reported metrics may be optimistic for strict deployment scenarios involving transfer to unseen regions or forecasting into future periods, where spatial autocorrelation and temporal dependence are typically weaker between training and test distributions. It should be emphasised that rigorous spatial-block and temporal-holdout evaluation requires long, consistently structured historical records that are not always available in utility settings This data-availability constraint is therefore a practical limitation of both this case study and the broader domain.

Third, NDVI temporal coverage is limited to one year of recent observations due to delivery constraints. Although this window supports operational implementation, it may under-represent inter-annual vegetation variability, lagged ecological responses, and rare climatic conditions; consequently, vegetation-related PoF estimates may be less sensitive to longerterm environmental dynamics than models trained on multi-year vegetation histories.

Overall, these design choices remain consistent with the study objective: to establish and demonstrate a scalable, extensible, asset-level PoF framework that is operationally useful in a real utility context, rather than to provide definitive estimates of out-of-region transferability or long-horizon forecasting skill.

## 4.3. Future Work

A key next step is transitioning from retrospective risk estimation to nearreal-time risk monitoring. The MODIS/Terra Vegetation Indices product (MOD13Q1), used here for NDVI-derived predictors, is updated at 16-day intervals, enabling regular refresh of vegetation-condition signals. In parallel, integrating temporally resolved lightning observations, rather than long-term climatology alone, would allow explicit modelling of short- to medium-term lightning activity trends. Together, these updates create a pathway to a near-real-time risk monitoring platform that can re-prioritise assets as environmental conditions evolve.

A second high-value extension is vision-driven feature generation from operational imagery. Utility inspection photos can be analysed with computervision pipelines to extract structured indicators such as conductor-vegetation clearance, branch overhang severity, pole or hardware condition, and local obstruction context. These image-derived variables could be fused with the current geospatial predictors to improve both near-field exposure characterisation and operational explainability.

Relatedly, street-level panoramic imagery (e.g., Google Street View or equivalent platforms) could support scalable external feature extraction to enhance direct inspection imagery. Such data can provide repeatable corridorlevel context for vegetation encroachment, built-environment shielding, and asset-visibility proxies at large scale. Future work can evaluate multimodal modelling that combines satellite, tabular geospatial, LiDAR, and street view inputs under a common asset-level risk framework. This direction is especially relevant under climate change, where shifts in extreme-weather regimes, fuel conditions, and storm behaviour may alter baseline failure risk over time and require adaptive, continuously updated decision support.

An implication of the condition-based design is its potential transferability across geographies. Because the model is structured around observable asset conditions and environmental exposure variables, rather than only location-specific identifiers, patterns that are rare or out-of-distribution in one service territory may already be represented in the historical data of another. This is particularly important under climate change: conditions that have not yet been common in one region may begin to materialise over time, while analogous combinations of heat, vegetation condition, terrain, water proximity, and lightning exposure may already exist elsewhere. Although formal geographic transfer testing is outside the scope of this study, the modular feature design highlights a pathway toward cross-region learning, where utilities can use broader historical experience to anticipate emerging local risks without reworking the core architecture.

## 5. Conclusion

This study demonstrates that a modular, geospatially harmonised machinelearning framework can deliver operationally meaningful asset-level PoF estimation for both vegetation-related and lightning-related failure modes. By integrating multi-source Earth-observation data, built-environment context, and utility failure records, the approach captures hazard-specific exposure patterns while maintaining a shared, scalable modelling pipeline.

The results indicate that the dominant predictors are interpretable and actionable for field deployment, supporting targeted interventions such as vegetation management, inspection prioritisation, and location-aware hardening. At the same time, the reported performance should be interpreted within study scope, including the absence of strict spatial/temporal holdouts and formal cross-model benchmarking.

Overall, the framework provides a practical foundation for risk-informed utility asset management and can be extended toward near-real-time decision support through temporally refreshed environmental inputs and vision-driven features from inspection and street-level imagery.

## Acknowledgements

The authors thank Ali Walsh for her support throughout the project. The authors also thank Ashleigh Lamb for her design work on the figures.

## References

Abatzoglou, J.T., Williams, A.P., 2016. Impact of anthropogenic climate change on wildfire across western US forests. Proceedings of the National

Academy of Sciences 113, 11770–11775. doi:10.1073/pnas.1607171113.

Cecil, D.J., Buechler, D.E., Blakeslee, R.J., 2014. Gridded lightning climatology from TRMM-LIS and OTD: Dataset description. Atmospheric Research 135–136, 404–414. doi:10.1016/j.atmosres.2012.06.028.

Contributors, O., . Openstreetmap. URL: https://www.openstreetmap. org. data retrieved from https://www.openstreetmap.org.

Davis, J., Goadrich, M., 2006. The relationship between precision-recall and roc curves, in: Proceedings of the 23rd International Conference on Machine Learning, pp. 233–240. doi:10.1145/1143844.1143874.

Didan, K., 2021. MOD13Q1 MODIS/Terra Vegetation Indices 16-Day L3 Global 250m SIN Grid V061. NASA EOSDIS Land Processes DAAC. doi:10.5067/MODIS/MOD13Q1.061.

Farr, T.G., Rosen, P.A., Caro, E., Crippen, R., Duren, R., Hensley, S., Kobrick, M., Paller, M., Rodriguez, E., Roth, L., Seal, D., Shaffer, S. Shimada, J., Umland, J., Werner, M., Oskin, M., Burbank, D., Alsdorf, D., 2007. The Shuttle Radar Topography Mission. Reviews of Geophysics 45, RG2004. doi:10.1029/2005RG000183.

Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., Liu, T.Y., 2017. Lightgbm: A highly efficient gradient boosting decision tree. Advances in Neural Information Processing Systems 30, 3146–3154. URL: https://proceedings.neurips.cc/paper/2017/ file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf.

Lundberg, S.M., Lee, S.I., 2017. A unified approach to interpreting model predictions, in: Proceedings of the 31st International Conference on Neural Information Processing Systems, Curran Associates Inc., Red Hook, NY, USA. p. 4768–4777.

Meeker, W.Q., Escobar, L.A., 1998. Statistical Methods for Reliability Data. John Wiley & Sons, New York.

Mukherjee, S., Nateghi, R., Hastak, M., 2018. A multi-hazard approach to assess severe weather-induced major power outage risks in the united states. Reliability Engineering & System Safety 175, 283–305. doi:10. 1016/j.ress.2018.03.015.

NIST/SEMATECH, 2003. Engineering statistics handbook: Assessing product reliability. National Institute of Standards and Technology. URL: https://www.itl.nist.gov/div898/handbook/apr/apr.htm. accessed 22 May 2026.

Pettorelli, N., Vik, J.O., Mysterud, A., Gaillard, J.M., Tucker, C.J., Stenseth, N.C., 2005. Using the satellite-derived NDVI to assess ecological responses to environmental change. Trends in Ecology & Evolution 20, 503–510. doi:10.1016/j.tree.2005.05.011.

Romps, D.M., Seeley, J.T., Vollaro, D., Molinari, J., 2014. Projected increase in lightning strikes in the united states due to global warming. Science 346, 851–854. doi:10.1126/science.1259100.

Saito, T., Rehmsmeier, M., 2015. The precision-recall plot is more informative than the roc plot when evaluating binary classifiers on imbalanced datasets. PLOS ONE 10, e0118432. doi:10.1371/journal.pone.0118432.

## Appendices

Appendix A Glossary of acronyms

CBD Central Business District.

DEM Digital Elevation Model.

LIS Lightning Imaging Sensor.

LightGBM Light Gradient Boosting Machine.

MOD13Q1 MODIS/Terra Vegetation Indices product identifier.

MODIS Moderate Resolution Imaging Spectroradiometer.

NASA National Aeronautics and Space Administration.

NDVI Normalised Difference Vegetation Index.

OOF Out-of-fold.

OSM OpenStreetMap.

PoF Probability of failure.

PR Precision-Recall.

ROC Receiver Operating Characteristic.

SAPN SA Power Networks.

SCONRR Steering Committee on National Regulatory Reporting. Utility network-context region classification variable used in the asset data (e.g., Rural, Metro, CBD).

SHAP SHapley Additive exPlanations.

SQL Structured Query Language.

SRTM Shuttle Radar Topography Mission.

VHRMC Very High-Resolution Gridded Lightning Monthly Climatology.

WKT Well-Known Text.

XML Extensible Markup Language.

Appendix B Feature set

Figures B.8 and B.9 provide the full feature-distribution diagnostics for the vegetation and lightning failure models, respectively.

![](images/f6ea410340d5f587e35251c4c6a0194944b4d8f526504133095f93d7cf4d9bdc.jpg)

![](images/197e8e4b8978b40a61054515721461d9cadb70200c3a3694f41614d2925903b6.jpg)

![](images/5e305fad2651445b3265c6e371317646f6dafc5498e9aff861f4c67a63d42211.jpg)

![](images/0b305fcc2778b757c42ddc60d4cf635a70cb303120c024850a0a72c1121b0a58.jpg)

![](images/4e92ad875a8c6ccc145084ab9209f32f3980bcf51b2ce1f8814c718ce686562c.jpg)

![](images/1979cb525f6fd219d8fc12d3267cac9267954294ce0150d53e21dc683a982f3e.jpg)

![](images/374c26997973d7be16866b2358081264d13e797d2a0903fb4f6ca1a0f9412cf6.jpg)

![](images/e44056183e3c9b631b5ca858ff77afc02f8216de27c223175425c12b41afe847.jpg)

![](images/6528ae4d891b44e1fe19f3d94d563489654c08e8907b855158e6d634b3fcb00b.jpg)

![](images/8e7e4ec9bbe379d9c3d92d59123b2281057442d6c7870eda94e520557d47fdde.jpg)

![](images/2ac2106b2a7076dc733d522dad90ab665c20caf81c8f14fd95d56e2c4cb92047.jpg)

![](images/ee4381b9fd7ed001fcccd9fce0ef2a796a415647452e2319faff84c41f62d457.jpg)

![](images/7427b1da399baecf936b775c535c2a93055857e1903aff6403e8c47c9f3f6a94.jpg)  
Figure B.8: Feature distributions for the vegetation-related failure model input space.

![](images/c4fc35b597356831f5af23d1504b04d1f9a87d6076dd0d968c6af2fdba185cab.jpg)  
Figure B.9: Feature distributions for the lightning-related failure model input space.